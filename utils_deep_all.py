import numpy as np
import os
import itertools
import gymnax
import jax
import jax.numpy as jnp
import equinox as eqx # Standard for JAX NNs
import optax
import matplotlib.pyplot as plt
import pickle

  
class QNetwork(eqx.Module):
    layers: list
    activation: callable = eqx.static_field()
    obs_dim: int = eqx.static_field()
    act_dim: int = eqx.static_field()
    T: float = eqx.static_field() 

    def __init__(self, key, env, hidden_size=100):
        # Environment-aware dimensions
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        
        # input_size = [normalized_x, normalized_t]
        input_size = 2 
        
        k1, k2, k3,k4,k5, k6 = jax.random.split(key, 6)

        self.layers = [
            eqx.nn.Linear(input_size, hidden_size, key=k1),
            eqx.nn.Linear(hidden_size, hidden_size, key=k2),
            eqx.nn.Linear(hidden_size, hidden_size, key=k3),
            eqx.nn.Linear(hidden_size, hidden_size, key=k4),
            eqx.nn.Linear(hidden_size, hidden_size, key=k5),
            eqx.nn.Linear(hidden_size, self.act_dim, key=k6),
        ]
        self.activation = jax.nn.relu
        self.T = env.time_steps

    def __call__(self, x, t):
        # Ensure both inputs are floating point tracers
        x_f = jnp.asarray(x).astype(jnp.float32)
        t_f = jnp.asarray(t).astype(jnp.float32)
        
        # Normalize
        # Division will result in float32 because the numerators are float32
        norm_x = x_f / (jnp.maximum(self.obs_dim, 1)-1)
        norm_t = t_f / (self.T-1)
        
        # Create the input vector [s, t]
        h = jnp.array([norm_x, norm_t])
        
        # Forward pass
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        
        return self.layers[-1](h)
    


class ReplayBuffer(eqx.Module):
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_states: jnp.ndarray
    done: jnp.ndarray
    times: jnp.ndarray
    capacity: int
    pointer: int
    size: int

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.pointer = 0
        self.size = 0
        self.states = jnp.zeros((capacity), dtype=jnp.int32)
        self.actions = jnp.zeros((capacity), dtype=jnp.int32)
        self.rewards = jnp.zeros((capacity))
        self.next_states = jnp.zeros((capacity), dtype=jnp.int32)
        self.done = jnp.zeros((capacity), dtype = jnp.bool)
        self.times = jnp.zeros((capacity), dtype=jnp.int32)



def add_batch(buffer: ReplayBuffer, batch):
    # batch = dict of arrays of shape (B, ...)
    B = batch.states.shape[0]
    cap = buffer.capacity
    ptr = buffer.pointer

    def update_array(arr, new_arr):
        if arr.ndim == 1:
            start = (ptr,)
        else:
            start = (ptr,) + (0,) * (arr.ndim - 1)
        return jax.lax.dynamic_update_slice(arr, new_arr, start)

    new_states = update_array(buffer.states, batch.states)
    new_actions = update_array(buffer.actions, batch.actions)
    new_rewards = update_array(buffer.rewards, batch.rewards)
    new_next_states = update_array(buffer.next_states, batch.next_states)
    new_done = update_array(buffer.done, batch.done)
    new_times = update_array(buffer.times, batch.times)

    new_pointer = (ptr + B) % cap
    new_size = jnp.minimum(buffer.size + B, cap)

    new_buffer = eqx.tree_at(
        lambda b: (b.states, b.actions, b.rewards, b.next_states, b.done, b.times, b.pointer, b.size),
        buffer,
        (new_states, new_actions, new_rewards, new_next_states, new_done, new_times, new_pointer, new_size)
    )

    return new_buffer


def sample_batch(key, buffer: ReplayBuffer, batch_size):
    # Only sample from valid entries
    valid_size = buffer.size
    idx = jax.random.randint(key, (batch_size,), 0, valid_size)
    
    batch = {
        'states': buffer.states[idx],
        'actions': buffer.actions[idx],
        'rewards': buffer.rewards[idx],
        'next_states': buffer.next_states[idx],
        'done': buffer.done[idx],
        'times': buffer.times[idx],
    }
    return batch

        


@jax.jit(static_argnums=(1,))  # Index 0 is model, Index 1 is env
def get_action_probs_from_jax_model(model, env):
    """
    Returns the (T, X, U) action probability table.
    Deterministic policy: model(x, t_norm)
    """
    T = env.time_steps
    X = env.obs_dim
    U = env.act_dim

    def compute_t_probs(t):
        # Vectorize over all possible states X
        # Note: model is an Equinox module (PyTree), which JIT handles automatically
        logits = jax.vmap(lambda x: model(x, t))(jnp.arange(X))
        best_action = jnp.argmax(logits, axis=-1)
        return jax.nn.one_hot(best_action, U)

    # Vectorize over all time steps T
    return jax.vmap(compute_t_probs)(jnp.arange(T))



def sample_mu_jax(env, models_pool, idxs, key):
    # Split key at the start
    reset_key, scan_key = jax.random.split(key)
    
    def policy_step(state, _):
        # EnvState already contains a key, but we split it for step randomness
        xs = state.xs
        t = state.t
        
        # Policy evaluation (deterministic argmax, no key needed)
        qs_all = jnp.stack([jax.vmap(lambda x: m(x, t))(xs) for m in models_pool])
        chosen_qs = qs_all[idxs, jnp.arange(env.num_agents), :]
        actions = jnp.argmax(chosen_qs, axis=-1)
        
        mu_t = jnp.bincount(xs, length=env.obs_dim) / env.num_agents
        
        # env.step uses the state.key internally and returns a new state with a new key
        _, next_state, _, _ = env.step(state, actions, mu_t)
        return next_state, mu_t

    init_state = env.reset(reset_key)
    final_state, mu_history = jax.lax.scan(policy_step, init_state, None, length=env.time_steps)
    
    # Return mu and the NEW key from the final state to the caller
    return mu_history, final_state.key

def to_action_probs_from_models(env, models_pool, idxs):
    """
    Calculates aggregate action probabilities (T, X, U).
    Models are now non-population dependent (ignore mu).
    """
    T = env.time_steps
    X = env.obs_dim
    U = env.act_dim
    
    # Distribution of models in the population
    counts = jnp.bincount(idxs, length=len(models_pool))
    sigma = counts / len(idxs)
    
    def compute_t_probs(t):
        def get_model_pi(model):
            # model call only takes (x, t_norm)
            logits = jax.vmap(lambda x: model(x, t))(jnp.arange(X))
            return jax.nn.one_hot(jnp.argmax(logits, axis=-1), U)

        # Average policies: (num_models, X, U) -> (X, U)
        model_pis = jnp.stack([get_model_pi(m) for m in models_pool])
        return jnp.tensordot(sigma, model_pis, axes=1)

    return jax.vmap(compute_t_probs)(jnp.arange(T))


def eval_curr_reward_jax(env, mu, deviating_action_probs):
    num_states = env.obs_dim
    def backward_step(v_next, t):
        p_t = env.get_P(mu[t]) #
        r_t = env.get_R(mu[t]) #
        q_t = r_t + jnp.einsum('ijk,k->ji', p_t, v_next) #
        v_curr = jnp.sum(deviating_action_probs[t] * q_t, axis=-1) #
    
        return v_curr, q_t

    # Scan backwards from T-1 to 0
    timesteps = jnp.arange(env.time_steps - 1, -1, -1)
    v_zero, all_qs = jax.lax.scan(backward_step, jnp.zeros(num_states), timesteps) #
    
    return v_zero, jnp.flip(all_qs, axis=0) #


def sample_mu_jax(env, models_pool, idxs, key):
    # Split key at the start
    reset_key, scan_key = jax.random.split(key)
    
    def policy_step(state, _):
        # EnvState already contains a key, but we split it for step randomness
        xs = state.xs
        t = state.t
        
        # Policy evaluation (deterministic argmax, no key needed)
        qs_all = jnp.stack([jax.vmap(lambda x: m(x, t))(xs) for m in models_pool])
        chosen_qs = qs_all[idxs, jnp.arange(env.num_agents), :]
        actions = jnp.argmax(chosen_qs, axis=-1)
        
        mu_t = jnp.bincount(xs, length=env.obs_dim) / env.num_agents
        
        # env.step uses the state.key internally and returns a new state with a new key
        _, next_state, _, _ = env.step(state, actions, mu_t)
        return next_state, mu_t

    init_state = env.reset(reset_key, num_agents=env.num_agents)
    final_state, mu_history = jax.lax.scan(policy_step, init_state, None, length=env.time_steps)
    
    # Return mu and the NEW key from the final state to the caller
    return mu_history, final_state.key


def train_sampling_BR_buffer(env, target_mu, iterations=1000, batch_size = 100, lr=1e-3, tau=0.005, key=None):
    print(lr)
    if key is None:
        key = jax.random.PRNGKey(42)
    model_key, rollout_key = jax.random.split(key)
    
    # 1. Initialize Model & Optimizer
    model = QNetwork(model_key, env,)
    # target_model = model
    target_model = jax.tree_util.tree_map(lambda x: x, model)

    optimizer = optax.chain(
    optax.clip_by_global_norm(1), # This stops the loss from exploding
    optax.adam(lr)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    buffer = ReplayBuffer(100000)

    def loss_fn_sample(online_model, target_model, sample):
        # Slice at time t
        s      = sample['states']       # (N,)
        a      = sample['actions']      # (N,)
        r      = sample['rewards']      # (N,)
        s_next = sample['next_states']  # (N,)
        done   = sample['done']       # (N,)
        t = sample['times']

        t_next = t+1

        # Target Q
        # q_next_all = jax.vmap(target_model)(s_next, t_next)  # (N, A)
        # q_next_max = jnp.max(q_next_all, axis=-1)            # (N,)

        # y = r + (1.0 - done) * q_next_max
        # y = jax.lax.stop_gradient(y)
        # Double DQN: Use online network to SELECT actions, target network to EVALUATE them
        q_next_online = jax.vmap(online_model)(s_next, t_next)  # (N, A) - online network
        best_actions = jnp.argmax(q_next_online, axis=-1)        # (N,) - select with online

        q_next_target = jax.vmap(target_model)(s_next, t_next)  # (N, A) - target network
        q_next_max = jnp.take_along_axis(
            q_next_target, 
            best_actions[:, None], 
            axis=-1
        ).squeeze()  # (N,) - evaluate with target

        y = r + (1.0 - done) * q_next_max
        y = jax.lax.stop_gradient(y)
        
        # Online Q
        q_all = jax.vmap(online_model)(s, t)              # (N, A)
        q = jnp.take_along_axis(q_all, a[:, None], axis=-1).squeeze()

        return jnp.mean(optax.huber_loss(targets = y, predictions= q,delta = 1 ))
        return jnp.mean((q - y) ** 2)
    
    loss_and_grad = eqx.filter_value_and_grad(loss_fn_sample)

    @eqx.filter_jit
    def train_step(carry, step_idx):
        online_model, target_model, opt_state, buffer, key = carry

        epsilon_start = 1.0
        epsilon_end = 0.1
        epsilon_decay_steps = iterations // 2  # Decay over first half of training
        
        epsilon = jnp.maximum(
            epsilon_end, 
            epsilon_start - (epsilon_start - epsilon_end) * (step_idx / epsilon_decay_steps)
        )

        # Generate fresh trajectories for this update step
        new_rollout_key, key_sample, next_key = jax.random.split(key, 3)
        trajectory = jax.lax.stop_gradient(env.rollout(online_model, target_mu,num_agents=100,key= new_rollout_key, epsilon = epsilon))

        flattened_batch = jax.tree_util.tree_map(lambda x: x.reshape(-1), trajectory)

        buffer = add_batch(buffer, flattened_batch)

        # for _ in range(5):
        sample = sample_batch(key_sample, buffer, batch_size)

        # loss_sample = loss_fn_sample(online_model, target_model, sample)
        
        loss_val, grads = loss_and_grad(online_model, target_model, sample)
        
        updates, next_opt_state = optimizer.update(grads, opt_state)
        next_online_model = eqx.apply_updates(online_model, updates)

        target_model_next = jax.tree_util.tree_map(lambda x: x, next_online_model)

        # grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads) if g is not None))
        # jax.debug.print("Step: {x}, Loss: {y}, Grad Norm: {z}, next_key {sa}", x=step_idx, y=loss_val, z=grad_norm, sa=sample.states)
        # jax.debug.print("eps:{e}",e = epsilon )
        # Polyak averaging for target model
        # next_target_model = jax.tree_util.tree_map(
        #     lambda o, t: tau * o + (1 - tau) * t,
        #     next_online_model, target_model
        # )
        # slide = (step_idx)//tau
        # step0 = 1000
        # ratio = 2

        # # compute the next update step
        # update_step = step0 * ratio**slide
        is_update_step = (step_idx % tau == 0)
        # jax.debug.print('update: {up}, idx = {id}, slide = {s}', up= is_update_step, id = step_idx, s = )

        # If true, copy next_online_model to next_target_model
        # If false, keep the old target_model
        next_target_model = jax.lax.cond(
            is_update_step,
            lambda _: target_model_next, # "True" branch
            lambda _: target_model,      # "False" branch
            operand=None
        )

        return (next_online_model, next_target_model, next_opt_state,buffer, next_key), loss_val

    # Training Loop
    init_carry = (model, target_model, opt_state,buffer, rollout_key)
    (final_model, _, _,buffer, _), loss_hist = jax.lax.scan(
        train_step, init_carry, jnp.arange(iterations)
    )

    return final_model, loss_hist, buffer




def save_run(save_dir, run_name, models_pool, idxs, metrics):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Save models (JAX pytrees → pickle is OK)
    with open(os.path.join(save_dir, f"{run_name}_models.pkl"), "wb") as f:
        pickle.dump(models_pool, f)

    # 2. Save idxs separately (lightweight)
    np.save(os.path.join(save_dir, f"{run_name}_idxs.npy"), np.array(idxs))

    # 3. Save metrics separately (inspect without models)
    with open(os.path.join(save_dir, f"{run_name}_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

def load_metrics(save_dir, run_name):
    import pickle
    with open(os.path.join(save_dir, f"{run_name}_metrics.pkl"), "rb") as f:
        return pickle.load(f)

def deep_deterministic_fictitious_play_sampling(
    env, iterations, num_agents, batch_size,
    iterations_br, lr, key, tau,
    save_dir=None,
    run_name=None,
):
    """
    JAX-based Deep Deterministic Fictitious Play for non-population dependent policies.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Track metrics
    metrics = {
        "exploitability": [],
        "br_rewards": [],
        "curr_rewards": []
    }
    
    # 1. Initialize Pool with the Environment-Aware QNetwork
    init_model_key, key = jax.random.split(key)
    # Note: QNetwork now requires the 'env' argument
    models_pool = [QNetwork(init_model_key, env)]
    
    # idxs[i] is the index of the model in models_pool used by agent i
    idxs = jnp.zeros(num_agents, dtype=jnp.int32) 
 
    for iteration in range(1, iterations + 1):
        print(f"\n--- Fictitious Iteration {iteration} ---")
        print(f"Training Deep Best Response...")
        # Calculate aggregate policy of the population
        model_key, sample_key, key  = jax.random.split(key,3)

        mu,_= sample_mu_jax(env, models_pool, idxs,sample_key)

        new_br_model, br_loss_hist,_ = train_sampling_BR_buffer(
            env, target_mu = mu, iterations=iterations_br, lr=lr, key = model_key,tau=tau, batch_size=batch_size
        )
        plt.plot(br_loss_hist)
        plt.yscale('log')
        plt.show()
        # 4. Evaluate Exploitability
        # Get action probs for just the NEW Best Response
        br_probs = get_action_probs_from_jax_model(new_br_model, env)
        pop_action_probs = to_action_probs_from_models(env, models_pool, idxs)
        # Value of BR vs Population Policy
        v_0_vec, _ = eval_curr_reward_jax(env, mu, br_probs)
        # Value of Population Policy vs Population Policy
        v_curr_vec, _ = eval_curr_reward_jax(env, mu, pop_action_probs)
        
        # Dot product with initial state distribution mu_0 to get scalar expected value
        V_br = float(jnp.dot(env.mu_0, v_0_vec))
        V_curr = float(jnp.dot(env.mu_0, v_curr_vec))
        exploitability = V_br - V_curr
        log_msg = f"Exploitability: {exploitability:.4f} | BR Reward: {V_br:.4f} | Pop Reward: {V_curr:.4f}"
        print(log_msg, flush=True)

        metrics["exploitability"].append(exploitability)
        metrics["br_rewards"].append(V_br)
        metrics["curr_rewards"].append(V_curr)

        # 5. Update Pool
        models_pool.append(new_br_model)
        new_model_idx = len(models_pool) - 1

        # 6. Update Agent Assignments (Standard Fictitious Play rule: 1/(t+1))
        key, switch_key = jax.random.split(key)
        # Probability of switching to the new best response
        prob_switch = 1.0 / (iteration + 1)
        use_new_br = jax.random.uniform(switch_key, (num_agents,)) < prob_switch
        
        # Update idxs: agents either keep their old model or take the newest one
        idxs = jnp.where(use_new_br, new_model_idx, idxs)

    if save_dir is not None and run_name is not None:
        save_run(
            save_dir=save_dir,
            run_name=run_name,
            models_pool=models_pool,
            idxs=idxs,
            metrics=metrics,
        )

    return mu, models_pool, idxs,  metrics




