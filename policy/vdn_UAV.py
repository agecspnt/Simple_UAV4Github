import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from network.base_net import D3QN
from network.vdn_net import VDNNet

class VDN:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape_agent = self.obs_shape

        self.eval_mlp = D3QN(input_shape_agent, args) 
        self.target_mlp = D3QN(input_shape_agent, args)

        self.eval_vdn_net = VDNNet()
        self.target_vdn_net = VDNNet()

        self.eval_mlp.to(self.args.device)
        self.target_mlp.to(self.args.device)
        self.eval_vdn_net.to(self.args.device)
        self.target_vdn_net.to(self.args.device)

        self.target_mlp.load_state_dict(self.eval_mlp.state_dict())

        self.optimizer = optim.Adam(self.eval_mlp.parameters(), lr=self.args.lr)

        self.scaler = GradScaler(enabled=self.args.cuda)

    def learn(self, batch, train_step, epsilon=None):
        episode_num = batch['o'].shape[0]
        max_episode_len = self.args.episode_limit
        self.init_hidden(episode_num)
        
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        
        s = batch['s'][:, :-1]
        r = batch['r'][:, :-1]
        u = batch['u'][:, :-1]
        terminated = batch['terminated'][:, :-1].float()
        obs = batch['o'][:, :-1]
        obs_next = batch['o'][:, 1:]
        mask = (1 - batch["padded"][:, :-1]).float()

        s = s.to(self.args.device)
        r = r.to(self.args.device)
        u = u.to(self.args.device)
        terminated = terminated.to(self.args.device)
        obs = obs.to(self.args.device)
        obs_next = obs_next.to(self.args.device)
        mask = mask.to(self.args.device)

        with autocast(enabled=self.args.cuda):
            q_evals_agents = []
            for i in range(self.n_agents):
                agent_obs = obs[:, :, i, :]
                agent_obs_reshaped = agent_obs.reshape(-1, self.obs_shape) 
                
                q_values_all_actions_agent = self.eval_mlp(agent_obs_reshaped)
                
                q_values_all_actions_agent = q_values_all_actions_agent.reshape(episode_num, max_episode_len -1, self.n_actions)
                
                action_taken_by_agent_i = u[:, :, i]
                q_taken_for_agent = torch.gather(q_values_all_actions_agent, dim=2, index=action_taken_by_agent_i)
                q_evals_agents.append(q_taken_for_agent)

            q_evals = torch.cat(q_evals_agents, dim=2)

            q_total_eval = self.eval_vdn_net(q_evals)

            q_target_next_individual_max_values = []
            for i in range(self.n_agents):
                agent_obs_next = obs_next[:, :, i, :]
                agent_obs_next_reshaped = agent_obs_next.reshape(-1, self.obs_shape)
                
                q_next_agent_target_all_actions = self.target_mlp(agent_obs_next_reshaped)
                
                q_next_agent_target_all_actions = q_next_agent_target_all_actions.reshape(episode_num, max_episode_len - 1, self.n_actions)
                
                q_next_agent_target_max, _ = torch.max(q_next_agent_target_all_actions, dim=2, keepdim=True)
                q_target_next_individual_max_values.append(q_next_agent_target_max)

            q_target_next_max_per_agent = torch.cat(q_target_next_individual_max_values, dim=2)

            q_total_target_next = self.target_vdn_net(q_target_next_max_per_agent)
            
            targets = r + self.args.gamma * q_total_target_next * (1 - terminated)

            td_error = (q_total_eval - targets.detach())
            
            masked_td_error = td_error * mask
            
            loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        
        if hasattr(self.args, 'grad_norm_clip') and self.args.grad_norm_clip is not None:
             torch.nn.utils.clip_grad_norm_(self.eval_mlp.parameters(), self.args.grad_norm_clip)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_mlp.load_state_dict(self.eval_mlp.state_dict())

        return loss.item()

    def init_hidden(self, batch_size):
        pass

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        q_net_path = os.path.join(path, "q_network.pth")
        torch.save(self.eval_mlp.state_dict(), q_net_path)
        print(f"Model Q-network saved to: {q_net_path}")

    def load_model(self, path):
        q_net_path = os.path.join(path, "q_network.pth")
        try:
            self.eval_mlp.load_state_dict(torch.load(q_net_path, map_location=lambda storage, loc: storage))
            self.target_mlp.load_state_dict(self.eval_mlp.state_dict()) 
            print(f"Q-Network loaded from {q_net_path}")
        except FileNotFoundError:
            print(f"Error: Q-Network model file not found at {q_net_path}")
            raise
        except Exception as e:
            print(f"Error loading Q-Network model from {q_net_path}: {e}")
            raise

    def get_q_values_for_actions(self, obs_batch, actions_batch):
        if not isinstance(obs_batch, torch.Tensor):
            obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=self.args.device if self.args.cuda else "cpu")
        if not isinstance(actions_batch, torch.Tensor):
            actions_batch = torch.tensor(actions_batch, dtype=torch.long, device=self.args.device if self.args.cuda else "cpu")

        q_all_actions = self.eval_mlp(obs_batch)
        q_taken = torch.gather(q_all_actions, dim=1, index=actions_batch)
        return q_taken

    def get_agent_q_values(self, agent_obs_np, agent_num):
        agent_obs_tensor = torch.tensor(agent_obs_np, dtype=torch.float32).unsqueeze(0)
        agent_obs_tensor = agent_obs_tensor.to(self.args.device)
        
        q_values = self.eval_mlp(agent_obs_tensor)
        return q_values 