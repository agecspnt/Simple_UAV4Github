import argparse
import os
import torch
import numpy as np
from runner import Runner
from common.arguments import get_common_args, get_mixer_args
from Multi_UAV_env import Multi_UAV_env_multi_level_practical_multicheck

import cProfile
import pstats
from io import StringIO
import time

profiler = cProfile.Profile()
profiler_printed_by_runner = False

def main_function_to_profile(prof):
    global profiler_printed_by_runner
    args = get_common_args()
    if args.alg.lower() == 'vdn':
        args = get_mixer_args(args) 
    
    if args.log_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        parent_log_dir = args.log_dir 
        args.log_dir = os.path.join(parent_log_dir, timestamp)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        print(f"All logs for this run will be saved in: {args.log_dir}")
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.log_dir = os.path.join(".", timestamp + "_logs")
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        print(f"Warning: Original args.log_dir not specified. All logs for this run will be saved in: {args.log_dir}")
    
    hyperparameters_file_path = os.path.join(args.log_dir, "hyperparameters.txt")
    with open(hyperparameters_file_path, 'w') as f:
        f.write(f"Hyperparameters for run: {args.log_dir}\n")
        f.write("-"*40 + "\n")
        for arg_name, arg_value in sorted(vars(args).items()):
            f.write(f"{arg_name}: {arg_value}\n")
    print(f"Hyperparameters saved to: {hyperparameters_file_path}")

    args.profiler_instance = prof
    args.runner_should_print_profile = not args.evaluate 
    args.profiler_printed_by_runner_flag_ref = globals()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    env = Multi_UAV_env_multi_level_practical_multicheck(args) 

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"] 

    runner = Runner(env, args)

    if args.evaluate:
        print("Starting evaluation...")
        prof.enable()
        runner.evaluate()
        prof.disable()
        
        if prof.stats:
            print("\n" + "="*50 + "\nPROFILE RESULTS (Evaluation):" + "="*50)
            s_io = StringIO()
            ps = pstats.Stats(prof, stream=s_io).sort_stats('cumulative')
            ps.print_stats(50)
            print(s_io.getvalue())
            profiler_printed_by_runner = True
    else:
        print("Starting training...")
        runner.run()

if __name__ == '__main__':
    profiler.enable()
    try:
        main_function_to_profile(profiler)
    finally:
        profiler.disable()
        if not profiler_printed_by_runner:
            s_io = StringIO()
            ps = pstats.Stats(profiler, stream=s_io).sort_stats('cumulative')
            
            output = s_io.getvalue()
            if output and not output.isspace() and "0 function calls" not in output:
                print("\n" + "="*50 + "\nPROFILE RESULTS (Main Fallback - End of Script):" + "="*50)
                print(output) 