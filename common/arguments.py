import argparse

def get_common_args():
    parser = argparse.ArgumentParser("Common arguments for multi-agent reinforcement learning")
    parser.add_argument('--n_steps', type=int, default=2000000, help='total number of steps for training')
    parser.add_argument('--n_episodes', type=int, default=1, help='total number of episodes for training')
    parser.add_argument('--episode_limit', type=int, default=50, help='maximum episode length')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use last action for recurrent policies')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=5000, help='how often to evaluate the model')
    parser.add_argument('--evaluate_episodes', type=int, default=32, help='number of episodes for evaluation')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--grad_norm_clip', type=float, default=10, help='gradient norm clipping')
    parser.add_argument('--save_cycle', type=int, default=5000, help='how often to save the model')
    parser.add_argument('--target_update_cycle', type=int, default=200, help='how often to update the target network')
    parser.add_argument('--cuda', action='store_true', help='enable CUDA (if available and desired)', default=True)
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--log_dir', type=str, default='./log/', help='directory to save logs')
    parser.add_argument('--model_dir', type=str, default='./model/', help='directory to save models')
    parser.add_argument('--alg', type=str, default='vdn', help='the algorithm to train the agent')
    parser.add_argument('--epsilon', type=float, default=1.0, help='initial epsilon for exploration')
    parser.add_argument('--min_epsilon', type=float, default=0.05, help='minimum epsilon for exploration')
    parser.add_argument('--anneal_steps', type=int, default=50000, help='steps for annealing epsilon')
    parser.add_argument('--buffer_size', type=int, default=5000, help='the size of replay buffer')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    
    parser.add_argument('--evaluate', action='store_true', default=False, help='whether to run evaluation mode instead of training')

    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='GRU hidden layer size for agent Q network')
    parser.add_argument('--qmix_hidden_dim', type=int, default=64, help='Hidden layer size for QMix network (if used)')
    parser.add_argument('--two_hyper_layers', type=bool, default=False, help='Whether to use two hyper layers for QMix an option')
    parser.add_argument('--hyper_hidden_dim', type=int, default=64, help='Hidden layer size for QMix hyper network (if used)')

    parser.add_argument('--map', type=str, default='default_map', help='Name of the map or scenario, used for organizing saved models')

    parser.add_argument('--tqdm_mininterval', type=float, default=0.5, help='Minimum interval (seconds) for tqdm progress bar updates.')

    parser.add_argument('--visualize_latest_eval', action='store_true', default=True, help='Whether to visualize the latest evaluation episode trajectories.')
    parser.add_argument('--save_visualization_plot', action='store_true', default=True, help='Whether to save the visualization plot instead of displaying it.')

    args = parser.parse_args()
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / args.anneal_steps if args.anneal_steps > 0 else 0
    return args

def get_mixer_args(args):
    if args.alg.lower() == 'vdn':
        pass
    elif args.alg.lower() == 'qmix':
        args.qmix_hidden_dim = getattr(args, 'qmix_hidden_dim', 64)
        args.two_hyper_layers = getattr(args, 'two_hyper_layers', False)
        args.hyper_hidden_dim = getattr(args, 'hyper_hidden_dim', 64)
    return args

if __name__ == '__main__':
    common_args = get_common_args()
    print("--- Common Arguments ---")
    for k, v in vars(common_args).items():
        print(f"{k}: {v}")

    class ArgsForMixerTest:
        def __init__(self):
            self.alg = 'qmix'

    args_with_mixer = get_mixer_args(common_args)
    print("--- Arguments with Mixer Config (example for QMix if alg=qmix) ---")
    if common_args.alg == 'qmix':
        print(f"QMix Hidden Dim: {args_with_mixer.qmix_hidden_dim}")
        print(f"Two Hyper Layers: {args_with_mixer.two_hyper_layers}")
        print(f"Hyper Hidden Dim: {args_with_mixer.hyper_hidden_dim}")
    elif common_args.alg == 'vdn':
        print("VDN does not have specific mixer arguments in this setup.")

    print("Note: To see QMix specific args, run script with --alg qmix") 