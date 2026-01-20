import argparse
from envs.JaxEnvs import *

def parse_args():
    parser = argparse.ArgumentParser(description="Behavioral MFG, DRL")

    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--env_name', type=str, default='beachbar')
    parser.add_argument('--folder_name', type=str, required=True)
    parser.add_argument('--file_name', type=str, default='run')

    return parser.parse_args()


def extract_env_all_params(args):
    all_params = vars(args).copy()

    if args.env_name == 'beachbar':
        env = SpatialBeachJax(num_agents=1000, time_steps=20, nb_states=21)

    elif args.env_name == 'SIS':
        env = SISJax(
            infection_rate=0.8,
            recovery_rate=0.2,
            num_agents=1000,
            time_steps=20,
            delta_t=0.5,
            c_I=2,
            c_P=0.8,
        )
    else:
        raise ValueError(f"Unknown env_name: {args.env_name}")

    return env, all_params
