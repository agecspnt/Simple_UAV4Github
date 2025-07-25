import numpy as np
import threading

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit

        self.buffer = {
            'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
            's': np.empty([self.size, self.episode_limit, self.state_shape]),
            'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
            'r': np.empty([self.size, self.episode_limit, 1]),
            'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
            's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
            'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
            'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
            'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
            'padded': np.empty([self.size, self.episode_limit, 1]),
            'terminated': np.empty([self.size, self.episode_limit, 1])
        }
        
        self.lock = threading.Lock()
        
        self.current_idx = 0
        self.current_size = 0

    def store_episode(self, episode_batch):
        with self.lock:
            idx = self._get_storage_idx(1)
            for key in self.buffer.keys():
                self.buffer[key][idx] = episode_batch[key]
    
    def sample(self, batch_size):
        temp_buffer = {}
        with self.lock:
            episode_idxs = np.random.randint(0, self.current_size, batch_size)
            
            for key in self.buffer.keys():
                temp_buffer[key] = self.buffer[key][episode_idxs]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc

        self.current_size = min(self.size, self.current_size + inc)
        
        if inc == 1:
            return idx[0]
        return idx

    def __len__(self):
        return self.current_size

if __name__ == '__main__':
    class DummyArgs:
        def __init__(self):
            self.n_actions = 4
            self.n_agents = 3
            self.state_shape = 12
            self.obs_shape = 10
            self.buffer_size = 5
            self.episode_limit = 2

    args = DummyArgs()
    replay_buffer = ReplayBuffer(args)

    print(f"--- Replay Buffer Initialized ---")
    print(f"Size: {replay_buffer.size}, Current Size: {len(replay_buffer)}")
    print(f"Episode limit: {replay_buffer.episode_limit}")

    episode_len_test = args.episode_limit 
    dummy_episode = {
        'o': np.random.rand(episode_len_test, args.n_agents, args.obs_shape),
        's': np.random.rand(episode_len_test, args.state_shape),
        'u': np.random.randint(0, args.n_actions, size=(episode_len_test, args.n_agents, 1)),
        'r': np.random.rand(episode_len_test, 1),
        'o_next': np.random.rand(episode_len_test, args.n_agents, args.obs_shape),
        's_next': np.random.rand(episode_len_test, args.state_shape),
        'avail_u': np.random.randint(0, 2, size=(episode_len_test, args.n_agents, args.n_actions)),
        'avail_u_next': np.random.randint(0, 2, size=(episode_len_test, args.n_agents, args.n_actions)),
        'u_onehot': np.eye(args.n_actions)[np.random.randint(0, args.n_actions, size=(episode_len_test, args.n_agents, 1)).reshape(-1)].reshape(episode_len_test, args.n_agents, args.n_actions),
        'padded': np.zeros((episode_len_test, 1)),
        'terminated': np.zeros((episode_len_test, 1))
    }
    dummy_episode['terminated'][episode_len_test-1,0] = 1

    print("\n--- Storing Episodes ---")
    num_episodes_to_store = 7
    for i in range(num_episodes_to_store):
        current_episode_data = {k: v + i if v.dtype in [np.float64, np.float32] else v for k,v in dummy_episode.items()}
        
        replay_buffer.store_episode(current_episode_data)
        print(f"Stored episode {i+1}. Buffer current_idx: {replay_buffer.current_idx}, current_size: {len(replay_buffer)}")
        assert len(replay_buffer) == min(i + 1, args.buffer_size), "Buffer size incorrect after store"

    assert len(replay_buffer) == args.buffer_size, "Buffer not full after enough stores."
    assert replay_buffer.current_idx == num_episodes_to_store % args.buffer_size if num_episodes_to_store >= args.buffer_size else num_episodes_to_store , "current_idx incorrect after wrap around"

    print("\n--- Sampling from Buffer ---")
    sample_batch_size = 2
    if len(replay_buffer) >= sample_batch_size:
        sampled_batch = replay_buffer.sample(sample_batch_size)
        print(f"Sampled batch of {sample_batch_size} episodes.")
        for key, value in sampled_batch.items():
            print(f"  Key '{key}', Shape: {value.shape}")
            if key == 'o':
                assert value.shape == (sample_batch_size, args.episode_limit, args.n_agents, args.obs_shape), f"Shape mismatch for {key}"
            elif key == 's':
                assert value.shape == (sample_batch_size, args.episode_limit, args.state_shape), f"Shape mismatch for {key}"
            elif key == 'r' or key == 'padded' or key == 'terminated':
                 assert value.shape == (sample_batch_size, args.episode_limit, 1), f"Shape mismatch for {key}"
        print("Sampling test passed.")
    else:
        print(f"Skipping sampling test as buffer size ({len(replay_buffer)}) is less than sample batch size ({sample_batch_size}).")

    print("ReplayBuffer tests completed.") 