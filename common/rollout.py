import numpy as np
import torch
import os
import time

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon if hasattr(args, 'anneal_epsilon') else 0.0
        self.min_epsilon = args.min_epsilon

        anneal_rate_info = 'N/A'
        if hasattr(self.args, 'anneal_steps') and self.args.anneal_steps > 0 and self.anneal_epsilon > 0:
            anneal_rate_info = self.anneal_epsilon / self.args.anneal_steps
        print(f"Initialized RolloutWorker. Default Epsilon (if not passed to generate_episode): {self.epsilon}, Min Epsilon: {self.min_epsilon}, Anneal Rate (config): {anneal_rate_info}")

    def generate_episode(self, episode_num=None, evaluate=False, epsilon=None, log_output_dir=None):
        if self.args.cuda:
            self.agents.policy.init_hidden(1)

        o, u, r, s, avail_u, u_onehot, terminated, padded = [], [], [], [], [], [], [], []
        o_next, s_next, avail_u_next = [], [], []
        episode_env_info = []

        obs_all_agents, state = self.env.reset() 
        
        terminated_flag = False
        win_flag = False
        episode_reward = 0
        step = 0

        if epsilon is None:
            current_epsilon = 0.0 if evaluate else self.epsilon
        else:
            current_epsilon = epsilon

        last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        while not terminated_flag and step < self.episode_limit:
            obs_tensor_list = [torch.tensor(obs_i, dtype=torch.float32) for obs_i in obs_all_agents]
            
            actions = []
            actions_onehot = []

            for agent_id in range(self.n_agents):
                avail_actions_agent = self.env.get_avail_agent_actions(agent_id) if hasattr(self.env, 'get_avail_agent_actions') else np.ones(self.n_actions)
                
                action_int = self.agents.choose_action(obs_all_agents[agent_id], 
                                                       last_action[agent_id], 
                                                       agent_id, 
                                                       avail_actions_agent, 
                                                       current_epsilon, 
                                                       evaluate)
                
                action_onehot = np.zeros(self.n_actions)
                action_onehot[action_int] = 1
                
                actions.append(action_int)
                actions_onehot.append(action_onehot)
                last_action[agent_id] = action_onehot

            reward, terminated_flag, env_info = self.env.step(actions) 
            episode_env_info.append(env_info)
            
            obs_all_agents_next = self.env.get_obs()
            state_next = self.env.get_state()

            o.append(obs_all_agents)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            r.append([reward])
            avail_u.append([self.env.get_avail_agent_actions(i) if hasattr(self.env, 'get_avail_agent_actions') else np.ones(self.n_actions) for i in range(self.n_agents)])
            
            o_next.append(obs_all_agents_next)
            s_next.append(state_next)
            avail_u_next.append([self.env.get_avail_agent_actions(i) if hasattr(self.env, 'get_avail_agent_actions') else np.ones(self.n_actions) for i in range(self.n_agents)])

            padded.append([0.])
            terminated.append([1.0 if terminated_flag else 0.])

            episode_reward += reward
            step += 1
            
            obs_all_agents = obs_all_agents_next
            state = state_next

            if evaluate and hasattr(self.env, 'render'):
                self.env.render()

        for _ in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            s.append(np.zeros(self.state_shape))
            u.append(np.zeros((self.n_agents, 1)))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminated.append([1.0])

        episode_transitions = {
            'o': np.array(o),
            's': np.array(s),
            'u': np.array(u),
            'r': np.array(r),
            'o_next': np.array(o_next),
            's_next': np.array(s_next),
            'avail_u': np.array(avail_u),
            'avail_u_next': np.array(avail_u_next),
            'u_onehot': np.array(u_onehot),
            'padded': np.array(padded),
            'terminated': np.array(terminated),
            'env_info': episode_env_info
        }
        
        stats = {
            "episode_reward": episode_reward,
            "episode_length": step,
            "epsilon": current_epsilon
        }
        if 'is_success' in env_info:
            stats["is_success"] = env_info['is_success']
        
        log_file_path = None
        if evaluate and log_output_dir is not None:
            log_file_path = self.write_log(episode_transitions, stats, episode_num, log_output_dir)
        elif evaluate and self.args.log_dir is not None:
            print(f"Warning: Evaluation logging triggered but specific log_output_dir not provided. Falling back to self.args.log_dir: {self.args.log_dir}")
            log_file_path = self.write_log(episode_transitions, stats, episode_num, self.args.log_dir)

        return episode_transitions, stats, log_file_path

    def write_log(self, episode_data, episode_stats, episode_num, target_log_dir):
        if episode_num is not None and isinstance(episode_num, str) and "_ep" in episode_num:
            try:
                ep_val = int(episode_num.split("_ep")[-1])
                if ep_val != 0:
                    return None
            except ValueError:
                pass

        if not target_log_dir:
            print("Warning: write_log called without a target_log_dir. Skipping log.")
            return None

        if not os.path.exists(target_log_dir):
            try:
                os.makedirs(target_log_dir)
            except OSError as e:
                print(f"Error creating log directory {target_log_dir}: {e}")
                return None
        
        timestamp = time.strftime("%Y%m%d-%H%M%S") 
        log_file_name = f"{self.args.alg}_{timestamp}_eval_ep{episode_num if episode_num is not None else 'undef'}.log"
        log_path = os.path.join(target_log_dir, log_file_name)

        with open(log_path, "w") as f:
            f.write(f"Episode: {episode_num if episode_num is not None else 'N/A'}, Algorithm: {self.args.alg}\n")
            f.write(f"Total Reward: {episode_stats['episode_reward']:.2f}, Length: {episode_stats['episode_length']}\n")
            if "is_success" in episode_stats:
                f.write(f"Success: {episode_stats['is_success']}\n")

            if hasattr(self.env, 'BS_locations'):
                bs_locs = self.env.BS_locations
                if isinstance(bs_locs, np.ndarray):
                    bs_locs_list = bs_locs.tolist()
                else:
                    bs_locs_list = list(bs_locs)
                
                bs_locs_str = "[" + ", ".join([f"[{loc[0]:.1f}, {loc[1]:.1f}]" for loc in bs_locs_list]) + "]"
                f.write(f"BS Locations: {bs_locs_str}\n")
            else:
                f.write(f"BS Locations: Not available in env object or attribute name mismatch\n")

            f.write("-" * 30 + "\n")
            f.write("Step | Agent Actions | Reward | Terminated | UAV Locations (if available) | Collisions (from env_info)\n")
            f.write("-" * 30 + "\n")

            for step_idx in range(episode_stats['episode_length']):
                actions_step = episode_data['u'][step_idx].flatten()
                reward_step = episode_data['r'][step_idx][0]
                terminated_step = episode_data['terminated'][step_idx][0]
                env_info_step = episode_data['env_info'][step_idx] if 'env_info' in episode_data and step_idx < len(episode_data['env_info']) else {}
                
                log_line = f"{step_idx:4d} | {actions_step} | {reward_step:7.2f} | {terminated_step:3.0f} | "
                
                state_at_step = episode_data['s'][step_idx]
                num_av = self.args.n_agents
                loc_start_idx = num_av + (num_av * self.env.n_bs if hasattr(self.env, 'n_bs') else num_av * 1)
                
                uav_locs_norm = state_at_step[loc_start_idx : loc_start_idx + num_av * 2].reshape(num_av, 2)
                uav_locs_real = uav_locs_norm * self.env.max_dist if hasattr(self.env, 'max_dist') else uav_locs_norm * 1500

                locations_str = "[" + ", ".join([f"({loc[0]:.1f},{loc[1]:.1f})" for loc in uav_locs_real]) + "]"
                log_line += f"{locations_str} | "
                
                collisions_str = "N/A"
                if 'collisions' in env_info_step:
                    if isinstance(env_info_step['collisions'], (list, np.ndarray)) and len(env_info_step['collisions']) == self.n_agents:
                        collisions_str = "[" + ", ".join(["C" if coll else "-" for coll in env_info_step['collisions']]) + "]"
                    else:
                        collisions_str = str(env_info_step['collisions'])
                
                log_line += f"{collisions_str}"

                f.write(log_line + "\n")
            
            f.write("-" * 30 + "\n")
            print(f"Evaluation log written to {log_path}")
        return log_path

if __name__ == '__main__':
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    from common.arguments import get_common_args
    
    class MockEnv:
        def __init__(self, args_env):
            self.n_a_agents = args_env.n_agents
            self.n_actions = args_env.n_actions
            self.episode_limit = args_env.episode_limit
            self.obs_shape = args_env.obs_shape
            self.state_shape = args_env.state_shape
            self.max_dist = 1500
            self.n_bs = 1

            self._step = 0

        def reset(self):
            self._step = 0
            obs = [np.random.rand(self.obs_shape) for _ in range(self.n_a_agents)]
            state = np.random.rand(self.state_shape)
            return obs, state

        def step(self, actions):
            self._step += 1
            reward = np.random.rand() - 0.5
            done = self._step >= self.episode_limit
            info = {'is_success': done and np.random.rand() > 0.5, 'collisions': np.random.randint(0,2)}
            
            obs_next = [np.random.rand(self.obs_shape) for _ in range(self.n_a_agents)]
            state_next = np.random.rand(self.state_shape)
            self.current_obs = obs_next 
            self.current_state = state_next
            return reward, done, info

        def get_obs(self):
            return [np.random.rand(self.obs_shape) for _ in range(self.n_a_agents)]
        
        def get_state(self):
            return np.random.rand(self.state_shape)

        def get_avail_agent_actions(self, agent_id):
            return np.ones(self.n_actions)

    class MockAgents:
        def __init__(self, args_agents):
            self.n_agents = args_agents.n_agents
            self.n_actions = args_agents.n_actions
            self.args = args_agents
            self.policy = type('MockPolicy', (object,), {'init_hidden': lambda self_policy, bs: None})()

        def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate):
            if evaluate:
                return np.argmax(np.random.rand(self.n_actions) * avail_actions)
            if np.random.rand() < epsilon:
                return np.random.choice(np.where(avail_actions == 1)[0])
            return np.argmax(np.random.rand(self.n_actions) * avail_actions)

    print("--- Testing RolloutWorker ---")
    class TestArgs:
        def __init__(self):
            self.episode_limit = 5
            self.n_actions = 4
            self.n_agents = 2
            self.state_shape = 10
            self.obs_shape = 8
            self.epsilon = 0.5
            self.min_epsilon = 0.01
            self.anneal_epsilon = (self.epsilon - self.min_epsilon) / 100
            self.cuda = False
            self.log_dir = "./test_logs_rollout/"
            self.alg = "test_alg"
            self.anneal_steps = 100

    test_args = TestArgs()

    mock_env = MockEnv(test_args)
    mock_agents = MockAgents(test_args)

    worker = RolloutWorker(mock_env, mock_agents, test_args)

    print("\n--- Generating Training Episode ---")
    episode_data_train, stats_train, _ = worker.generate_episode(episode_num=1, evaluate=False)
    print(f"Training Episode Stats: {stats_train}")
    assert stats_train['episode_length'] <= test_args.episode_limit
    assert 'o' in episode_data_train
    assert episode_data_train['o'].shape == (test_args.episode_limit, test_args.n_agents, test_args.obs_shape)
    
    print("\n--- Generating Evaluation Episode (with logging) ---")
    if os.path.exists(test_args.log_dir):
        import shutil
        shutil.rmtree(test_args.log_dir)

    episode_data_eval, stats_eval, log_file_path = worker.generate_episode(episode_num=1, evaluate=True)
    print(f"Evaluation Episode Stats: {stats_eval}")
    assert stats_eval['epsilon'] == 0
    assert os.path.exists(test_args.log_dir), "Log directory was not created."
    log_files = os.listdir(test_args.log_dir)
    assert len(log_files) > 0, "Log file was not created in the log directory."
    print(f"Log file created: {os.path.join(test_args.log_dir, log_files[0])}")

    print("\nRolloutWorker tests completed.")
    if os.path.exists(test_args.log_dir):
        import shutil
        shutil.rmtree(test_args.log_dir) 