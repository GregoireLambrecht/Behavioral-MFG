from args_parser import *
from utils_deep_all import deep_deterministic_fictitious_play_sampling
import numpy as np
import jax

def main():
    args = parse_args()
    env, all_params = extract_env_all_params(args)
        
    seed_int = all_params['seed']
    seed = jax.random.PRNGKey(seed_int)
    print(f"--- Running {args.env_name} | Seed: {seed_int} ---")
    
    # Now all_params contains 'folder_name' and 'file_name'
    deep_deterministic_fictitious_play_sampling(env, 
                                                iterations=100,
                                                num_agents=1000,
                                                iterations_br=100000,
                                                lr=8e-4,
                                                key=seed, 
                                                tau = 10000,
                                                batch_size=500,
                                                save_dir=all_params['folder_name'],
                                                run_name=all_params['file_name']
                                                )
if __name__ == "__main__":
    main()