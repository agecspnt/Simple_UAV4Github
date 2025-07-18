import numpy as np
import math
import random

class Multi_UAV_env_multi_level_practical_multicheck:
    def __init__(self, args):
        self.n_a_agents = 3
        self.n_bs = 1
        self.n_channel = 2

        self.sinr_AV = 6
        self.sinr_AV_real = 10**(self.sinr_AV / 10)

        self.pw_AV = 30
        self.pw_AV_real = (10**(self.pw_AV / 10)) / 1000

        self.pw_BS = 40
        self.pw_BS_real = (10**(self.pw_BS / 10)) / 1000

        self.g_main = 10
        self.g_side = 1

        self.N0 = -120
        self.N0_real = (10**(self.N0 / 10)) / 1000

        self.reference_AV = -60
        self.reference_AV_real = 10**(self.reference_AV / 10)

        self.env_PL_exp = 3

        self.height_BS = 15
        self.height_AV = 150

        self.v_max = 100
        self.delta_t = 2

        self.n_speed = 2
        self.vel_actions = np.array([0, 100], dtype=float)

        self.BS_locations = np.zeros((self.n_bs, 2), dtype=float)
        self.bs_radius = 200

        self.init_location = np.array([[-250.0, 400.0], [-30.33, 930.33], [-30.33, -130.33]], dtype=float)
        self.dest_location = np.array([[1250.0, 400.0], [1030.33, -130.33], [1030.33, 930.33]], dtype=float)

        self.trajectory = np.zeros((self.n_a_agents, 6))
        for i in range(self.n_a_agents):
            delta_x = self.dest_location[i,0] - self.init_location[i,0]
            delta_y = self.dest_location[i,1] - self.init_location[i,1]
            if delta_x == 0:
                self.trajectory[i, 0] = np.inf if delta_y > 0 else -np.inf
            else:
                self.trajectory[i, 0] = delta_y / delta_x
            self.trajectory[i, 1] = self.init_location[i,1] - self.trajectory[i,0] * self.init_location[i,0]
            self.trajectory[i, 2] = self.init_location[i,0]
            self.trajectory[i, 3] = self.init_location[i,1]
            self.trajectory[i, 4] = self.dest_location[i,0]
            self.trajectory[i, 5] = self.dest_location[i,1]

        self.max_dist = 1500
        self.check_num = 5

        self.collision_threshold = 10

        self.episode_limit = args.episode_limit if hasattr(args, 'episode_limit') else 50
        self.arrive_reward = 10
        self.all_arrived_reward = 2000
        self.conflict_reward = -100
        self.collision_reward = -100
        self.movement_penalty = -1

        self.n_actions = self.n_speed * self.n_channel

        self.current_location = np.copy(self.init_location)
        self.is_arrived = np.zeros(self.n_a_agents, dtype=bool)
        self.is_collision = np.zeros(self.n_a_agents, dtype=bool)
        self.episode_step = 0
        self.recorded_arrive = []

        self.args = args

    def get_state_size(self):
        return self.n_a_agents + (self.n_a_agents * self.n_bs) + (self.n_a_agents * 2)

    def get_state(self):
        arrive_state = self.is_arrived.astype(float).flatten()

        bs_association_flags = np.zeros(self.n_a_agents * self.n_bs)
        for i in range(self.n_a_agents):
            bs_association_flags[i * self.n_bs + 0] = 1.0 
        
        location_norm = self.current_location.flatten() / self.max_dist

        state = np.concatenate((arrive_state, bs_association_flags, location_norm))
        return state

    def get_obs_size(self):
        obs_size_val = 0
        obs_size_val += 1
        obs_size_val += 1
        obs_size_val += (self.n_a_agents - 1) * (self.n_bs + 1 + 1)
        obs_size_val += (self.n_a_agents - 1)
        obs_size_val += self.n_a_agents
        return obs_size_val

    def get_obs_agent(self, agent_id):
        obs = []

        my_loc = self.current_location[agent_id]
        my_dest = self.dest_location[agent_id]
        if self.is_arrived[agent_id]:
            obs.append(0.0)
        else:
            dist_to_dest = np.linalg.norm(my_dest - my_loc)
            obs.append(dist_to_dest / self.max_dist)

        obs.append(1.0)

        for other_id in range(self.n_a_agents):
            if other_id == agent_id:
                continue
            
            other_loc = self.current_location[other_id]

            bs_assoc_other = np.zeros(self.n_bs)
            if self.n_bs > 0:
                 bs_assoc_other[0] = 1.0 
            obs.extend(bs_assoc_other.tolist())

            other_bs_loc_3d = np.append(self.BS_locations[0], self.height_BS)
            my_loc_3d = np.append(my_loc, self.height_AV)
            dist_to_other_bs = np.linalg.norm(other_bs_loc_3d - my_loc_3d)
            obs.append(dist_to_other_bs / self.max_dist)

            dist_to_other_agent = np.linalg.norm(my_loc_3d - np.append(other_loc, self.height_AV))
            obs.append(dist_to_other_agent / self.max_dist)

        for other_id in range(self.n_a_agents):
            if other_id == agent_id:
                continue
            obs.append(1.0 if self.is_arrived[other_id] else 0.0)

        agent_one_hot = np.zeros(self.n_a_agents)
        agent_one_hot[agent_id] = 1.0
        obs.extend(agent_one_hot.tolist())
        
        return np.array(obs, dtype=float)

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_a_agents)]
        return agents_obs

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.n_actions,
            "n_agents": self.n_a_agents,
            "episode_limit": self.episode_limit,
            "agent_num": self.n_a_agents,
            "action_space": [np.arange(self.n_actions)] * self.n_a_agents
        }
        return env_info

    def _calculate_path_loss(self, loc1_2d, loc2_2d, h1, h2):
        loc1_3d = np.append(loc1_2d, h1)
        loc2_3d = np.append(loc2_2d, h2)
        dist_3d = np.linalg.norm(loc1_3d - loc2_3d)
        if dist_3d < 1e-6:
            dist_3d = 1e-6
        path_loss_linear = self.reference_AV_real * (dist_3d ** self.env_PL_exp)
        if path_loss_linear == 0:
            return np.inf 
        return 1.0 / path_loss_linear

    def _calculate_sinr(self, agent_id, channel_idx, current_agent_locations, agent_actions_decoded):
        uav_loc_2d = current_agent_locations[agent_id]
        bs_loc_2d = self.BS_locations[0]
        
        gain_signal = self._calculate_path_loss(uav_loc_2d, bs_loc_2d, self.height_AV, self.height_BS)
        if np.isinf(gain_signal):
             received_signal_power = 0
        else:
             received_signal_power = self.pw_AV_real * (self.g_main**2) * gain_signal

        total_interference_power = 0

        for interferer_id in range(self.n_a_agents):
            if interferer_id == agent_id:
                continue

            _, interferer_channel_idx = agent_actions_decoded[interferer_id]
            interferer_vel_idx, _ = agent_actions_decoded[interferer_id]
            interferer_is_active = (not self.is_arrived[interferer_id]) and (self.vel_actions[interferer_vel_idx] > 0)

            if interferer_channel_idx == channel_idx and interferer_is_active:
                interferer_loc_2d = current_agent_locations[interferer_id]
                gain_interf_uav_to_bs = self._calculate_path_loss(interferer_loc_2d, bs_loc_2d, self.height_AV, self.height_BS)
                if np.isinf(gain_interf_uav_to_bs):
                    interf_power_contrib = 0
                else:
                    interf_power_contrib = self.pw_AV_real * self.g_main * self.g_side * gain_interf_uav_to_bs
                total_interference_power += interf_power_contrib

        denominator = total_interference_power + self.N0_real
        if denominator == 0:
            return np.inf
        
        sinr_real = received_signal_power / denominator
        return sinr_real

    def delta_location(self, vel_level_idx, agent_id):
        vel = self.vel_actions[vel_level_idx]
        moving_dist = vel * self.delta_t

        if moving_dist == 0:
            return np.array([0.0, 0.0])

        slope_param = self.trajectory[agent_id, 0]
        
        current_pos = self.current_location[agent_id]
        dest_pos = self.dest_location[agent_id]
        
        direction_vector = dest_pos - current_pos
        dist_to_dest = np.linalg.norm(direction_vector)

        if dist_to_dest < 1e-6:
             return np.array([0.0,0.0])

        if moving_dist >= dist_to_dest:
            return direction_vector

        unit_direction_vector = direction_vector / dist_to_dest
        delta_x = unit_direction_vector[0] * moving_dist
        delta_y = unit_direction_vector[1] * moving_dist
        
        return np.array([delta_x, delta_y])

    def check_action_collision(self, inter_locations_all_agents):
        collision_flags = np.zeros(self.n_a_agents, dtype=bool)
        for i in range(self.n_a_agents):
            for j in range(i + 1, self.n_a_agents):
                collided_this_pair = False
                for k_check_point in range(self.check_num):
                    loc_i_2d = inter_locations_all_agents[i, k_check_point, :]
                    loc_j_2d = inter_locations_all_agents[j, k_check_point, :]
                    
                    dist_sq = (loc_i_2d[0] - loc_j_2d[0])**2 + \
                              (loc_i_2d[1] - loc_j_2d[1])**2
                    
                    if dist_sq < self.collision_threshold**2:
                        collision_flags[i] = True
                        collision_flags[j] = True
                        collided_this_pair = True
                        break
        return collision_flags

    def step(self, actions):
        self.episode_step += 1
        rewards = np.zeros(self.n_a_agents)
        
        agent_actions_decoded = []
        for i in range(self.n_a_agents):
            action_int = actions[i]
            vel_level_idx = action_int // self.n_channel
            channel_idx = action_int % self.n_channel
            agent_actions_decoded.append((vel_level_idx, channel_idx))

        prev_locations = np.copy(self.current_location)
        inter_locations = np.zeros((self.n_a_agents, self.check_num, 2))

        for i in range(self.n_a_agents):
            delta_loc = np.array([0.0, 0.0])
            if not self.is_arrived[i]:
                vel_idx, _ = agent_actions_decoded[i]
                delta_loc = self.delta_location(vel_idx, i)
                self.current_location[i] += delta_loc
            
            for k in range(self.check_num):
                if self.check_num == 1:
                    inter_locations[i, k, :] = prev_locations[i] + delta_loc
                else:
                    inter_locations[i, k, :] = prev_locations[i] + k * (delta_loc / (self.check_num - 1))
        
        for i in range(self.n_a_agents):
            if not self.is_arrived[i]:
                current_dist_to_dest = np.linalg.norm(self.current_location[i] - self.dest_location[i])
                if current_dist_to_dest < 10.0:
                    self.is_arrived[i] = True
                    self.current_location[i] = np.copy(self.dest_location[i])
                    if i not in self.recorded_arrive:
                        rewards[i] += self.arrive_reward
                        self.recorded_arrive.append(i)

        self.is_collision = self.check_action_collision(inter_locations)

        step_reward_val = 0

        for i in range(self.n_a_agents):
            if not self.is_arrived[i]:
                step_reward_val += self.movement_penalty
        
        current_step_global_reward = 0
        
        _reward_this_step = 0
        
        for i in range(self.n_a_agents):
            if not self.is_arrived[i]:
                _reward_this_step += self.movement_penalty
            
            if self.is_collision[i]:
                _reward_this_step += self.collision_reward

        for i in range(self.n_a_agents):
            vel_idx, ch_idx = agent_actions_decoded[i]
            is_active_for_comms = (not self.is_arrived[i]) and (self.vel_actions[vel_idx] > 0)
            
            if is_active_for_comms:
                sinr_val = self._calculate_sinr(i, ch_idx, self.current_location, agent_actions_decoded)
                if sinr_val < self.sinr_AV_real:
                    _reward_this_step += self.conflict_reward
        
        for i in range(self.n_a_agents):
            if self.is_arrived[i]:
                if i not in self.recorded_arrive:
                    _reward_this_step += self.arrive_reward
                    self.recorded_arrive.append(i)

        if len(self.recorded_arrive) == self.n_a_agents:
            if not hasattr(self, 'all_arrived_bonus_given_flag') or not self.all_arrived_bonus_given_flag:
                 all_currently_arrived = True
                 for i in range(self.n_a_agents):
                     if not self.is_arrived[i]:
                         all_currently_arrived = False
                         break
                 if all_currently_arrived:
                    _reward_this_step += self.all_arrived_reward
                    self.all_arrived_bonus_given_flag = True
        
        terminated = False
        if self.episode_step >= self.episode_limit:
            terminated = True
        
        if len(self.recorded_arrive) == self.n_a_agents:
            all_currently_arrived_check = True
            for i in range(self.n_a_agents):
                if not self.is_arrived[i]:
                    all_currently_arrived_check = False; break
            if all_currently_arrived_check:
                 terminated = True

        info = {
            'is_success': len(self.recorded_arrive) == self.n_a_agents and terminated,
            'collisions': np.sum(self.is_collision),
        }
        if terminated and not hasattr(self, 'all_arrived_bonus_given_flag'):
            self.all_arrived_bonus_given_flag = False

        final_reward = _reward_this_step
        
        return final_reward, terminated, info

    def _clear_episode_flags(self):
        if hasattr(self, 'all_arrived_bonus_given_flag'):
            del self.all_arrived_bonus_given_flag
        self.recorded_arrive = []

    def reset(self):
        self._clear_episode_flags()
        self.episode_step = 0
        self.current_location = np.copy(self.init_location)
        self.is_arrived = np.zeros(self.n_a_agents, dtype=bool)
        self.is_collision = np.zeros(self.n_a_agents, dtype=bool)

        for i in range(self.n_bs):
            angle = random.uniform(0, 2 * math.pi)
            r = self.bs_radius * math.sqrt(random.uniform(0, 1))
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            self.BS_locations[i] = [x, y]
        return self.get_obs(), self.get_state()

if __name__ == '__main__':
    class DummyArgs:
        def __init__(self):
            self.episode_limit = 60

    args = DummyArgs()
    env = Multi_UAV_env_multi_level_practical_multicheck(args)

    print("--- Environment Test ---")
    print(f"Number of Agents: {env.n_a_agents}")
    print(f"Number of Actions: {env.n_actions}")
    print(f"State Size: {env.get_state_size()}")
    print(f"Observation Size: {env.get_obs_size()}")
    print(f"Max episode steps: {env.episode_limit}")

    obs, state = env.reset()
    print("\nInitial Observations (Agent 0):", obs[0])
    print("Initial State:", state)
    assert len(obs[0]) == env.get_obs_size(), "Obs size mismatch"
    assert len(state) == env.get_state_size(), "State size mismatch"

    total_reward_acc = 0
    done = False
    current_step = 0

    print("\n--- Running a Sample Episode ---")
    while not done and current_step < env.episode_limit:
        current_step += 1
        random_actions = [np.random.randint(0, env.n_actions) for _ in range(env.n_a_agents)]
        
        reward, done, info = env.step(random_actions)
        
        next_obs = env.get_obs()
        next_state = env.get_state()
        
        total_reward_acc += reward
        
        print(f"Step: {current_step}, Actions: {random_actions}, Reward: {reward:.2f}, Done: {done}")
        if 'collisions' in info and info['collisions'] > 0:
             print(f"  Collision detected this step for UAVs: {env.is_collision}")

        if done:
            print(f"Episode finished after {current_step} steps.")
            print(f"Final reward: {total_reward_acc:.2f}")
            print(f"Success: {info.get('is_success', False)}")
            break
    
    if not done:
        print(f"Episode timed out after {env.episode_limit} steps.")
        print(f"Final reward: {total_reward_acc:.2f}")

    print("\n--- Testing get_env_info ---")
    env_info_dict = env.get_env_info()
    print(env_info_dict)
    assert env_info_dict["state_shape"] == env.get_state_size()
    assert env_info_dict["obs_shape"] == env.get_obs_size()
    assert env_info_dict["n_actions"] == env.n_actions
    assert env_info_dict["n_agents"] == env.n_a_agents

    print("\n--- Testing specific scenarios (conceptual) ---")
    env.reset()
    env.current_location[0] = np.array([100.0, 100.0])
    env.current_location[1] = np.array([100.5, 100.5])
    env.current_location[2] = np.array([500.0, 500.0])

    test_inter_locs = np.zeros((env.n_a_agents, env.check_num, 2))
    for ag_idx in range(env.n_a_agents):
        for ch_idx in range(env.check_num):
            test_inter_locs[ag_idx, ch_idx, :] = env.current_location[ag_idx] 
    
    collision_results = env.check_action_collision(test_inter_locs)
    print(f"Manual collision check for UAVs at almost same spot: {collision_results}")
    assert collision_results[0] == True, "Collision not detected for UAV 0"
    assert collision_results[1] == True, "Collision not detected for UAV 1"
    assert collision_results[2] == False, "Collision incorrectly detected for UAV 2"

    env.reset()
    env.current_location[0] = np.array([0.0, 0.0])
    env.current_location[1] = np.array([10.0, 0.0])
    env.current_location[2] = np.array([500.0, 20.0])
    
    decoded_actions_test = [(0,0), (0,0), (0,1)]
    env.is_arrived[:] = False

    sinr_uav0 = env._calculate_sinr(agent_id=0, channel_idx=0, 
                                   current_agent_locations=env.current_location, 
                                   agent_actions_decoded=decoded_actions_test)
    print(f"SINR for UAV0 (with UAV1 interfering on same channel 0): {sinr_uav0:.4e} (linear), dB: {10*np.log10(sinr_uav0) if sinr_uav0 > 0 else -np.inf:.2f}")

    sinr_uav2 = env._calculate_sinr(agent_id=2, channel_idx=1,
                                   current_agent_locations=env.current_location,
                                   agent_actions_decoded=decoded_actions_test)
    print(f"SINR for UAV2 (on channel 1, no other same-channel interferers): {sinr_uav2:.4e} (linear), dB: {10*np.log10(sinr_uav2) if sinr_uav2 > 0 else -np.inf:.2f}")
    
    if sinr_uav0 < env.sinr_AV_real:
        print(f"UAV0 SINR ({10*np.log10(sinr_uav0):.2f} dB) is BELOW threshold ({env.sinr_AV} dB). Conflict penalty should apply.")
    else:
        print(f"UAV0 SINR ({10*np.log10(sinr_uav0):.2f} dB) is ABOVE threshold ({env.sinr_AV} dB). No conflict penalty for UAV0.")

    print("\nBasic tests completed.") 