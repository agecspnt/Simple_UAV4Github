import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_units=[64, 64]):
        super(MLP, self).__init__()
        layers = []
        prev_units = input_shape
        for units in hidden_units:
            layers.append(nn.Linear(prev_units, units))
            layers.append(nn.ReLU())
            prev_units = units
        layers.append(nn.Linear(prev_units, n_actions))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        q_values = self.fc2(h_out)
        return q_values, h_out

class D3QN(nn.Module):
    def __init__(self, input_shape, args):
        super(D3QN, self).__init__()
        self.args = args
        
        self.feature_layer_size = args.rnn_hidden_dim if hasattr(args, 'rnn_hidden_dim') else 64 

        self.feature_net = MLP(input_shape, self.feature_layer_size, 
                               hidden_units=[self.feature_layer_size])

        self.advantage_net = MLP(self.feature_layer_size, args.n_actions, 
                                 hidden_units=[self.feature_layer_size // 2])

        self.value_net = MLP(self.feature_layer_size, 1, 
                             hidden_units=[self.feature_layer_size // 2])

    def forward(self, x):
        features = self.feature_net(x)
        
        advantage = self.advantage_net(features)
        value = self.value_net(features)
        
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values 