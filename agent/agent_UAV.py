import numpy as np
import torch

class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        if args.alg == 'vdn':
            from policy.vdn_UAV import VDN
            self.policy = VDN(args)
        else:
            raise Exception("Unsupported algorithm: {}".format(args.alg))

        self.epsilon = args.epsilon if hasattr(args, 'epsilon') else 0.05
        self.min_epsilon = args.min_epsilon if hasattr(args, 'min_epsilon') else 0.01
        self.anneal_epsilon = args.anneal_epsilon if hasattr(args, 'anneal_epsilon') else (self.epsilon - self.min_epsilon) / 100000
        print(f"Initialized Agents with Epsilon: {self.epsilon}, Min Epsilon: {self.min_epsilon}, Anneal Steps: {(self.epsilon - self.min_epsilon) / self.anneal_epsilon if self.anneal_epsilon > 0 else 'N/A'}")

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        inputs = obs.copy()

        if not evaluate and np.random.uniform() < epsilon:
            if avail_actions is not None:
                avail_actions_ind = np.nonzero(avail_actions)[0]
                if len(avail_actions_ind) == 0:
                    action = np.random.choice(self.n_actions)
                else:
                    action = np.random.choice(avail_actions_ind)
            else:
                action = np.random.choice(self.n_actions)
        else:
            q_values_tensor = self.policy.get_agent_q_values(inputs, agent_num) 
            
            q_values_np = q_values_tensor.squeeze(0).cpu().detach().numpy()

            if avail_actions is not None:
                q_values_np[avail_actions == 0] = -float('inf')
            
            action = np.argmax(q_values_np)
        
        return action

    def train(self, batch, train_step, epsilon=None):
        loss = self.policy.learn(batch, train_step)
        return loss

    def save_model(self, path, train_step):
        self.policy.save_model(path)
        print(f"Agent's policy model saved (identifier: {train_step}) to directory: {path}")

    def load_model(self, path):
        self.policy.load_model(path)
        print(f"Loaded model from {path}")

    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self, epsilon_val):
        self.epsilon = epsilon_val

if __name__ == '__main__':
    class DummyArgs:
        def __init__(self):
            self.n_actions = 4
            self.n_agents = 3
            self.state_shape = 12
            self.obs_shape = 10
            self.alg = 'vdn'
            self.epsilon = 0.5
            self.min_epsilon = 0.01
            self.anneal_epsilon = (self.epsilon - self.min_epsilon) / 5000
            self.use_cuda = False
            self.rnn_hidden_dim = 64
            self.lr = 0.0005
            self.gamma = 0.99
            self.target_update_cycle = 200
            self.grad_norm_clip = 10
            self.device = torch.device("cuda" if self.use_cuda else "cpu")

    args = DummyArgs()
    
    class MockVDNPolicy:
        def __init__(self, args_policy):
            self.n_agents = args_policy.n_agents
            self.n_actions = args_policy.n_actions
            self.device = args_policy.device
            self.eval_mlps = [lambda obs_tensor: torch.rand(obs_tensor.size(0), self.n_actions).to(self.device) 
                              for _ in range(self.n_agents)] 

        def get_agent_q_values(self, obs_tensor, agent_num):
            if agent_num < len(self.eval_mlps):
                return torch.rand(obs_tensor.shape[0], self.n_actions).to(self.device)
            else:
                raise ValueError(f"Agent number {agent_num} out of range for policy.")

        def learn(self, batch, train_step):
            print(f"MockVDNPolicy: learn called at train_step {train_step} with batch of size {len(batch['o'])}")
            return np.random.rand() 
            
        def save_model(self, path, train_step):
            print(f"MockVDNPolicy: save_model called for path {path}, step {train_step}")

        def load_model(self, path):
            print(f"MockVDNPolicy: load_model called for path {path}")

    import sys
    mock_policy_module = type(sys)('policy.vdn_UAV')
    mock_policy_module.VDN = MockVDNPolicy
    sys.modules['policy.vdn_UAV'] = mock_policy_module

    print("--- Testing Agents Class Initialization ---")
    agents_manager = Agents(args)
    print(f"Agents manager initialized with policy: {type(agents_manager.policy)}")

    print("\n--- Testing choose_action ---")
    dummy_obs_shape = (args.obs_shape,) if isinstance(args.obs_shape, int) else args.obs_shape
    obs_agent_0 = np.random.rand(*dummy_obs_shape) 
    last_action_agent_0 = np.zeros(args.n_actions)
    agent_id_0 = 0
    avail_actions_agent_0 = np.array([1, 1, 0, 1])

    chosen_action_explore = agents_manager.choose_action(obs_agent_0, last_action_agent_0, agent_id_0, avail_actions_agent_0, epsilon=1.0, evaluate=False)
    print(f"Chosen action (explore, epsilon=1.0, agent 0): {chosen_action_explore}")
    assert avail_actions_agent_0[chosen_action_explore] == 1, "Exploratory action chose an unavailable action"

    chosen_action_greedy = agents_manager.choose_action(obs_agent_0, last_action_agent_0, agent_id_0, avail_actions_agent_0, epsilon=0.0, evaluate=False)
    print(f"Chosen action (greedy, epsilon=0.0, agent 0): {chosen_action_greedy}")
    assert avail_actions_agent_0[chosen_action_greedy] == 1, "Greedy action chose an unavailable action"

    chosen_action_eval = agents_manager.choose_action(obs_agent_0, last_action_agent_0, agent_id_0, avail_actions_agent_0, epsilon=1.0, evaluate=True)
    print(f"Chosen action (evaluate=True, agent 0): {chosen_action_eval}")
    assert avail_actions_agent_0[chosen_action_eval] == 1, "Evaluation action chose an unavailable action"
    assert chosen_action_eval == chosen_action_greedy, "Greedy and Evaluate actions should match for same obs if Q-values are deterministic"

    print("\n--- Testing Epsilon Annealing (conceptual) ---")
    initial_eps = agents_manager.get_epsilon()
    for _ in range(5):
        agents_manager.choose_action(obs_agent_0, last_action_agent_0, agent_id_0, avail_actions_agent_0, agents_manager.get_epsilon(), evaluate=False)
    print(f"Epsilon after some steps: {agents_manager.get_epsilon()} (initial: {initial_eps})")
    assert agents_manager.get_epsilon() < initial_eps or agents_manager.get_epsilon() == args.min_epsilon

    print("\n--- Testing train method ---")
    batch_size = 2
    dummy_batch = {
        'o': np.random.rand(batch_size, args.n_agents, args.episode_limit if hasattr(args, 'episode_limit') else 50, args.obs_shape),
    }
    
    simple_dummy_batch = {'o': [None]*batch_size}
    
    mock_episode_len = 5
    mock_batch_for_train = {
        'o': np.random.rand(batch_size, mock_episode_len, args.n_agents, args.obs_shape),
    }

    loss = agents_manager.train(mock_batch_for_train, train_step=100)
    print(f"Loss from dummy train call: {loss}")

    print("\n--- Testing Save/Load Model ---")
    agents_manager.save_model("./dummy_agent_model", 100)
    agents_manager.load_model("./dummy_agent_model")

    del sys.modules['policy.vdn_UAV']
    print("\nAgent class tests completed (using mock policy).") 