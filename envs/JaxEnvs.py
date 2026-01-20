import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from abc import ABC, abstractmethod

# 1. State container for JAX functional logic
class EnvState(NamedTuple):
    xs: jnp.ndarray
    t: int
    key: jax.random.PRNGKey

class Trajectory(NamedTuple):
    states: jnp.ndarray      # (T, num_agents)
    actions: jnp.ndarray     # (T, num_agents)
    times: jnp.ndarray       # (T, num_agents)
    rewards: jnp.ndarray     # (T, num_agents)
    next_states: jnp.ndarray # (T, num_agents)
    done: jnp.ndarray        # (T, num_agents)



# 2. Base Class (keeping your exact structure)
class MFMARLEnv(ABC):
    def __init__(self, observation_space, action_space, 
                 time_steps, num_agents, mu_0, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        self.time_steps = time_steps
        self.num_agents = num_agents
        self.mu_0 = mu_0

    @abstractmethod
    def reset(self, key): pass

    @abstractmethod
    def step(self, state, action): pass

# 3. SIS JAX Implementation
class SISJax(MFMARLEnv):
    def __init__(self, infection_rate=0.8, recovery_rate=0.2, time_steps=50,
                 initial_infection_prob=0.2, c_I=2.5, c_P=0.8,
                 delta_t=0.9, num_agents=300):
        
        # State 0: Susceptible (Healthy), State 1: Infected
        # Action 0: Passive, Action 1: Distancing
        observation_space = type('Space', (), {'n': 2})()
        action_space = type('Space', (), {'n': 2})()
        mu_0 = jnp.array([1 - initial_infection_prob, initial_infection_prob])

        super().__init__(observation_space, action_space, time_steps, num_agents, mu_0)

        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.delta_t = delta_t
        self.c_I = c_I
        self.c_P = c_P
        self.obs_dim = 2
        self.act_dim = 2

    def reset(self, key, num_agents):
        reset_key, next_key = jax.random.split(key)
        xs = jax.random.choice(reset_key, jnp.arange(self.obs_dim), 
                               shape=(num_agents,), p=self.mu_0)
        return EnvState(xs=xs, t=0, key=next_key)

    def next_states(self, t, xs, us, noise, mu_t):
        """
        Matrix-consistent transition logic.
        noise: shape (num_agents, 2) where [:, 0] is recovery noise and [:, 1] is infection noise.
        """
        # mu_t[1] is the fraction of infected agents
        prob_recovery = self.recovery_rate * self.delta_t
        prob_infection = self.infection_rate * mu_t[1] * self.delta_t
        
        # We use the provided noise (uniform [0,1]) to determine transitions
        recoveries = noise[:, 0] < prob_recovery
        infections = noise[:, 1] < prob_infection
        
        # Action 1 (Distancing) protects perfectly against infection in this model
        # (Current Infected & Didn't Recover) OR (Current Healthy & Not Distancing & Infected)
        new_xs = (xs * (1 - recoveries)) + ((1 - xs) * (1 - us) * infections)
        return new_xs.astype(jnp.int32)

    def reward(self, t, xs, us, mu_t):
        """Standard reward function: cost of being infected + cost of action."""
        return - (self.c_I * xs) - (self.c_P * us)

    def get_P(self, mu_t):
        """
        Constructs the (U, X, X) transition matrix.
        X=0: Healthy, X=1: Infected
        """
        X = self.obs_dim
        U = self.act_dim
        
        p_inf = jnp.minimum(1.0, self.infection_rate * mu_t[1] * self.delta_t)
        p_rec = self.recovery_rate * self.delta_t
        
        # For each action, we define the (From, To) transitions
        # Action 0 (Passive)
        P0 = jnp.array([
            [1 - p_inf, p_inf], # From Healthy
            [p_rec, 1 - p_rec]  # From Infected
        ])
        
        # Action 1 (Distancing) - Prob of infection becomes 0
        P1 = jnp.array([
            [1.0, 0.0],         # From Healthy (Protected)
            [p_rec, 1 - p_rec]  # From Infected
        ])
        
        return jnp.stack([P0, P1])

    def get_R(self, mu_t):
        """Constructs the (X, U) reward matrix."""
        # Using vmap over the reward function for consistency
        def get_r_for_u(u_val):
            return self.reward(None, jnp.arange(self.obs_dim), jnp.full((self.obs_dim,), u_val), mu_t)
        
        return jax.vmap(get_r_for_u)(jnp.arange(self.act_dim)).T

    def step(self, state: EnvState, actions: jnp.ndarray, mu_t: jnp.ndarray):
        """Refactored step using Matrix Logic for perfect consistency."""
        key, noise_key = jax.random.split(state.key)
        
        # 1. Rewards from matrix
        R = self.get_R(mu_t)
        rewards = R[state.xs, actions]
        
        # 2. Transitions from matrix
        P = self.get_P(mu_t)
        probs = P[actions, state.xs, :]
        
        # 3. Sample next states
        next_xs = jax.random.categorical(noise_key, jnp.log(probs + 1e-10), axis=-1)
        
        next_state = EnvState(xs=next_xs, t=state.t + 1, key=key)
        done = next_state.t >= self.time_steps
        
        return next_xs, next_state, rewards, done

    def rollout(self, model, target_mu, num_agents, key, epsilon = 0.1):
        # Pre-calculate matrices for speed as before
        P_horizon = jax.vmap(self.get_P)(target_mu)
        R_horizon = jax.vmap(self.get_R)(target_mu)

        def body_fun(state, t):
            
            # 1. Get Greedy Actions from the Model
            logits = jax.vmap(lambda x: model(x, t))(state.xs)
            greedy_actions = jnp.argmax(logits, axis=-1)
            
            # 2. E-Greedy Exploration Logic
            # Split the state key ONCE at the beginning
            key_eps, key_rand, key_step, next_key = jax.random.split(state.key, 4)
            
            # Generate random actions for all agents
            rand_actions = jax.random.randint(key_rand, (num_agents,), 0, self.act_dim)
            
            # Decide which agents explore
            actions = jnp.where(
                jax.random.uniform(key_eps, (num_agents,)) < epsilon,
                rand_actions,
                greedy_actions
            )
            
            # 3. Environment Transition (using actions)
            curr_xs = state.xs
            rewards = R_horizon[t, curr_xs, actions]
            probs = P_horizon[t, actions, curr_xs, :]
            
            # Sample next states
            next_xs = jax.random.categorical(key_step, jnp.log(probs + 1e-10), axis=-1)
            
            # Update state with the fresh next_key
            next_state = EnvState(xs=next_xs, t=state.t + 1, key=next_key)
            
            # 4. Record Trajectory
            transition = Trajectory(
                states=curr_xs,
                actions=actions,
                times=jnp.full((num_agents,), t),
                rewards=rewards,
                next_states=next_xs,
                done=jnp.full((num_agents,), next_state.t >= self.time_steps)
            )
            return next_state, transition

        initial_state = self.reset(key, num_agents)
        _, trajectory = jax.lax.scan(body_fun, initial_state, jnp.arange(self.time_steps))
        return trajectory

class SpatialBeachJax(MFMARLEnv):
    """
    Spatial Beach Bar on a 1D Torus.
    State (xs): Integer position [0, nb_states - 1].
    Action (us): 0: Left (-1), 1: Stay (0), 2: Right (+1).
    Reward: -Distance_to_Bar - (Coefficient * Local_Density) - abs(action)
    """
    def __init__(self, nb_states=21, time_steps=20, num_agents=300, 
                 bar_pos=None, congestion_coeff=5.0):
        
        observation_space = type('Space', (), {'n': nb_states})()
        action_space = type('Space', (), {'n': 3})() 
        
        # Start with a uniform distribution across the beach
        mu_0 = jnp.ones(nb_states) / nb_states

        super().__init__(observation_space, action_space, time_steps, num_agents, mu_0)

        self.nb_states = nb_states
        self.bar_pos = bar_pos if bar_pos is not None else nb_states // 2
        self.congestion_coeff = congestion_coeff
        self.obs_dim = nb_states
        self.act_dim = 3 
    
    def reset(self, key, num_agents):
        reset_key, next_key = jax.random.split(key)
        xs = jax.random.choice(reset_key, jnp.arange(self.obs_dim), 
                               shape=(num_agents,), p=self.mu_0)
        return EnvState(xs=xs, t=0, key=next_key)

    def next_states(self, t, xs, us, noise):
        """
        Updated: Noise is now passed in as an argument (sampled externally).
        noise: array of shape (num_agents,) containing values in {-1, 0, 1}
        """
        # us is [0, 1, 2], map to [-1, 0, 1]
        displacements = us - 1 
        
        # Periodic boundaries
        next_xs = (xs + displacements + noise) % self.nb_states
        return next_xs.astype(jnp.int32)

    def reward(self, t, xs, us, mu):
        """
        The agent wants to be at bar_pos but hates crowds.
        mu: The Mean Field (probability mass at each of the nb_states).
        """
        # 1. Shortest distance to Bar on the Torus
        dist = jnp.abs(xs - self.bar_pos)
        dist = jnp.minimum(dist, self.nb_states - dist)
        
        # 2. Continuous Congestion Penalty
        # Penalty is simply a linear function of the density at the agent's spot
        local_density = mu[xs] 
        congestion_penalty = self.congestion_coeff * local_density
        
        return -dist.astype(jnp.float32) - congestion_penalty - jnp.abs(us)

    def step(self, state: EnvState, actions: jnp.ndarray, target_mu_t: jnp.ndarray):
        """
        Step function refactored to use Matrix Logic for consistency.
        """
        key, noise_key = jax.random.split(state.key)
        
        # 1. Get Reward from Matrix Logic
        # R has shape (X, U). We index it using state.xs and actions.
        R = self.get_R(target_mu_t)
        rewards = R[state.xs, actions]
        
        # 2. Get Transition Probabilities from Matrix Logic
        # P has shape (U, X, X). 
        P = self.get_P(target_mu_t)
        
        # Extract transition distribution for each agent based on their action and current state
        # probs shape: (num_agents, X)
        probs = P[actions, state.xs, :]
        
        # 3. Sample next states based on the probabilities in P
        # jax.random.categorical allows vectorized sampling across the agent dimension
        next_xs = jax.random.categorical(noise_key, jnp.log(probs + 1e-10), axis=-1)
        
        next_state = EnvState(xs=next_xs, t=state.t + 1, key=key)
        done = next_state.t >= self.time_steps
        
        return next_xs, next_state, rewards, done

    def rollout(self, model, target_mu, num_agents, key, epsilon = 0.1):
        # Pre-calculate matrices for speed as before
        P_horizon = jax.vmap(self.get_P)(target_mu)
        R_horizon = jax.vmap(self.get_R)(target_mu)

        def body_fun(state, t):
            
            # 1. Get Greedy Actions from the Model
            logits = jax.vmap(lambda x: model(x, t))(state.xs)
            greedy_actions = jnp.argmax(logits, axis=-1)
            
            # 2. E-Greedy Exploration Logic
            # Split the state key ONCE at the beginning
            key_eps, key_rand, key_step, next_key = jax.random.split(state.key, 4)
            
            # Generate random actions for all agents
            rand_actions = jax.random.randint(key_rand, (num_agents,), 0, self.act_dim)
            
            # Decide which agents explore
            actions = jnp.where(
                jax.random.uniform(key_eps, (num_agents,)) < epsilon,
                rand_actions,
                greedy_actions
            )
            
            # 3. Environment Transition (using actions)
            curr_xs = state.xs
            rewards = R_horizon[t, curr_xs, actions]
            probs = P_horizon[t, actions, curr_xs, :]
            
            # Sample next states
            next_xs = jax.random.categorical(key_step, jnp.log(probs + 1e-10), axis=-1)
            
            # Update state with the fresh next_key
            next_state = EnvState(xs=next_xs, t=state.t + 1, key=next_key)
            
            # 4. Record Trajectory
            transition = Trajectory(
                states=curr_xs,
                actions=actions,
                times=jnp.full((num_agents,), t),
                rewards=rewards,
                next_states=next_xs,
                done=jnp.full((num_agents,), next_state.t >= self.time_steps)
            )
            return next_state, transition

        initial_state = self.reset(key, num_agents)
        _, trajectory = jax.lax.scan(body_fun, initial_state, jnp.arange(self.time_steps))
        return trajectory
    
    def get_P(self, mu_t):
        X = self.nb_states
        U = 3
        noise_vals = jnp.array([-1, 0, 1])
        prob = 1.0 / 3.0

        def get_transition_for_action_noise(u_val, n_val):
            # Now u_val and n_val will be scalars (0-dim)
            all_xs = jnp.arange(X)
            us_vec = jnp.full((X,), u_val)
            ns_vec = jnp.full((X,), n_val)
            
            # next_states logic: (xs + (u-1) + n) % X
            next_s = self.next_states(None, all_xs, us_vec, ns_vec)
            
            # Returns (X, X) matrix for this specific u and n
            return jax.nn.one_hot(next_s, X)

        # 1. Map over Noise for a single action: returns (3, X, X)
        def get_P_for_single_action(u_val):
            # vmap over noise_vals (axis 0)
            noise_mats = jax.vmap(get_transition_for_action_noise, in_axes=(None, 0))(u_val, noise_vals)
            # Average the 3 matrices: (X, X)
            return jnp.mean(noise_mats, axis=0)

        # 2. Map over Actions: returns (3, X, X)
        P = jax.vmap(get_P_for_single_action)(jnp.arange(U))
        
        return P
    

    def get_R(self, mu_t, action_probs=None):
        """
        Constructs the Reward Matrix R [State, Action].
        Matches the logic of: -dist - congestion - |us|
        """
        X = self.nb_states
        U = 3
        xs = jnp.arange(X)
        us = jnp.arange(U) # 0, 1, 2
        
        # 1. Torus distance to bar
        dist = jnp.abs(xs - self.bar_pos)
        dist = jnp.minimum(dist, X - dist)
        
        # 2. Congestion penalty (local density at x)
        # mu_t is (X,), so congestion is (X,)
        congestion = self.congestion_coeff * mu_t
        
        # 3. Action cost: |us|
        # Note: If us=[0, 1, 2], then Action 2 (Right) is more expensive than Action 0 (Left).
        action_cost = jnp.abs(us)
        
        # Combine: R[x, u] = (-dist[x] - congestion[x]) - action_cost[u]
        # We use broadcasting: (X, 1) - (1, U)
        state_reward = -dist.astype(jnp.float32) - congestion
        R = state_reward[:, None] - action_cost[None, :]
        
        return R