import numpy as np
import os
from common.replay_buffer import ReplayBuffer
from common.rollout import RolloutWorker
from agent.agent_UAV import Agents
import torch
from tqdm import tqdm
import collections
import subprocess
import time
import matplotlib.pyplot as plt
import visualizer
from io import StringIO
import pstats

class Runner:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.agents = Agents(args)

        self.buffer = ReplayBuffer(args)

        self.rolloutWorker = RolloutWorker(self.env, self.agents, self.args)

        self.log_dir = args.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        self.convergence_timesteps = []
        self.convergence_avg_rewards = []
        self.convergence_avg_losses = []

        print(f"Runner initialized with algorithm: {args.alg}")
        print(f"Number of agents: {args.n_agents}, Number of actions: {args.n_actions}")
        print(f"Episode limit: {args.episode_limit}, Total training steps: {args.n_steps}")
        print(f"Buffer size: {args.buffer_size}, Batch size: {args.batch_size}")
        print(f"Epsilon: init={self.epsilon:.2f}, min={self.min_epsilon:.2f}, anneal_steps={args.anneal_steps}")
        print(f"Target update cycle: {args.target_update_cycle}, Evaluate cycle: {args.evaluate_cycle}, Save cycle: {args.save_cycle}")
        print(f"Device: {args.device}")

    def run(self):
        print("\nStarting training run...")
        time_steps = 0
        episode_num = 0
        last_evaluate_T = 0
        last_save_T = 0

        train_rewards = collections.deque(maxlen=100)
        train_episode_lengths = collections.deque(maxlen=100)
        train_losses = collections.deque(maxlen=1000)

        first_eval_done_for_profiling = False

        with tqdm(total=self.args.n_steps, desc="Training Progress", mininterval=self.args.tqdm_mininterval, unit="step") as pbar:
            while time_steps < self.args.n_steps:
                episode_num += 1
                current_epsilon_for_rollout = max(self.min_epsilon, self.epsilon)
                
                episode_batch, episode_stats, _ = self.rolloutWorker.generate_episode(
                    episode_num=episode_num, 
                    epsilon=current_epsilon_for_rollout,
                    evaluate=False
                )
                episode_reward = episode_stats["episode_reward"]
                episode_len = episode_stats["episode_length"]
                
                train_rewards.append(episode_reward)
                train_episode_lengths.append(episode_len)
                self.buffer.store_episode(episode_batch)
                
                num_train_steps_this_episode = 0
                if self.buffer.current_size >= self.args.batch_size:
                    for _ in range(episode_len):
                        batch = self.buffer.sample(self.args.batch_size)
                        loss = self.agents.train(batch, time_steps + num_train_steps_this_episode)
                        train_losses.append(loss)
                        num_train_steps_this_episode +=1
                
                pbar.update(episode_len)
                time_steps += episode_len
                
                if self.epsilon > self.min_epsilon:
                    if self.args.anneal_steps > 0:
                         self.epsilon = self.args.epsilon - (self.args.epsilon - self.min_epsilon) * (min(time_steps, self.args.anneal_steps) / self.args.anneal_steps)
                    else: 
                        self.epsilon = self.min_epsilon
                
                log_stats = {
                    "Episode": episode_num,
                    "Epsilon": f"{current_epsilon_for_rollout:.3f}",
                    "Avg Reward (Tr)~": f"{np.mean(train_rewards):.2f}" if train_rewards else "N/A",
                    "Avg Length (Tr)~": f"{np.mean(train_episode_lengths):.2f}" if train_episode_lengths else "N/A",
                    "Avg Loss~": f"{np.mean(train_losses):.4f}" if train_losses else "N/A",
                    "Buffer": f"{self.buffer.current_size}/{self.buffer.size}"
                }
                pbar.set_postfix(log_stats)

                if (time_steps - last_evaluate_T) >= self.args.evaluate_cycle:
                    pbar.set_description_str(f"Running evaluation at T={time_steps}")
                    
                    if train_rewards: 
                        self.convergence_timesteps.append(time_steps)
                        self.convergence_avg_rewards.append(np.mean(train_rewards))
                    if train_losses: 
                        self.convergence_avg_losses.append(np.mean(train_losses)) 
                    else: 
                        if self.convergence_timesteps and len(self.convergence_avg_losses) < len(self.convergence_timesteps):
                             self.convergence_avg_losses.append(np.nan) 

                    eval_start_time = time.time() 
                    self.evaluate(episode_num, time_steps)
                    eval_duration = time.time() - eval_start_time 
                    pbar.write(f"Evaluation at T={time_steps} (Episode {episode_num}) finished in {eval_duration:.2f} seconds.") 
                    
                    last_evaluate_T = time_steps
                    pbar.set_description_str("Training Progress") 

                    if not first_eval_done_for_profiling and hasattr(self.args, 'runner_should_print_profile') and self.args.runner_should_print_profile:
                        if hasattr(self.args, 'profiler_instance'):
                            pbar.write("Disabling profiler and printing results after first evaluation...")
                            self.args.profiler_instance.disable() 
                            
                            s_io_runner = StringIO()
                            ps_runner = pstats.Stats(self.args.profiler_instance, stream=s_io_runner).sort_stats('cumulative')
                            ps_runner.print_stats(100)
                            profile_output_runner = s_io_runner.getvalue()

                            if profile_output_runner and not profile_output_runner.isspace() and "0 function calls" not in profile_output_runner:
                                pbar.write("\n" + "="*50 + "\nPROFILE RESULTS (After First Evaluation in Training):" + "="*50)
                                pbar.write(profile_output_runner)
                            else:
                                pbar.write("DEBUG: Profiler output was considered empty/trivial or had 0 calls. Raw output (if any) was:")
                                pbar.write(profile_output_runner)
                            
                            if hasattr(self.args, 'profiler_printed_by_runner_flag_ref') and 'profiler_printed_by_runner' in self.args.profiler_printed_by_runner_flag_ref:
                                self.args.profiler_printed_by_runner_flag_ref['profiler_printed_by_runner'] = True
                            
                            self.args.runner_should_print_profile = False
                        first_eval_done_for_profiling = True

                if (time_steps - last_save_T) >= self.args.save_cycle and time_steps > 0:
                    pbar.set_description_str(f"Saving model at T={time_steps}")
                    self.save_models(time_steps) 
                    last_save_T = time_steps
                    pbar.set_description_str("Training Progress")
        
        print(f"\nTraining finished after {time_steps} timesteps and {episode_num} episodes.")
        
        if hasattr(self.args, 'runner_should_print_profile') and self.args.runner_should_print_profile:
            if hasattr(self.args, 'profiler_instance'):
                self.args.profiler_instance.disable() 
                
                s_io_fallback = StringIO()
                ps_fallback = pstats.Stats(self.args.profiler_instance, stream=s_io_fallback).sort_stats('cumulative')
                ps_fallback.print_stats(100)
                profile_output_fallback = s_io_fallback.getvalue()

                if profile_output_fallback and not profile_output_fallback.isspace() and "0 function calls" not in profile_output_fallback:
                    print("\n" + "="*50 + "\nPROFILE RESULTS (Training Ended Before First Eval Triggered Profile Print):" + "="*50)
                    print(profile_output_fallback)
                else:
                    print("DEBUG: Fallback profiler output was considered empty/trivial or had 0 calls. Raw output (if any) was:")
                    print(profile_output_fallback)
                
                if hasattr(self.args, 'profiler_printed_by_runner_flag_ref') and 'profiler_printed_by_runner' in self.args.profiler_printed_by_runner_flag_ref:
                     self.args.profiler_printed_by_runner_flag_ref['profiler_printed_by_runner'] = True
                self.args.runner_should_print_profile = False

        self.evaluate(episode_num, time_steps) 
        self.save_models(time_steps) 

    def evaluate(self, current_episode_num=0, current_time_steps=0):
        print(f"\nEvaluating model... Training Episode: {current_episode_num}, Timesteps: {current_time_steps}")
        total_rewards = []
        total_steps = []

        visualization_save_dir = os.path.join(self.args.log_dir, "visualizations")
        if not os.path.exists(visualization_save_dir):
            os.makedirs(visualization_save_dir)

        first_log_file_path = None

        run_specific_log_dir = self.args.log_dir 
        if not run_specific_log_dir:
            print("Warning: self.args.log_dir is not set. Cannot save detailed episode logs.")

        for eval_ep_num in tqdm(range(self.args.evaluate_episodes), desc="Evaluation Episodes", leave=False, mininterval=self.args.tqdm_mininterval):
            log_episode_identifier = f"eval_T{current_time_steps}_ep{eval_ep_num}"
            
            episode_data, episode_stats, log_file_path_this_ep = self.rolloutWorker.generate_episode(
                episode_num=log_episode_identifier, 
                evaluate=True, 
                epsilon=0.0,
                log_output_dir=run_specific_log_dir
            )
            total_rewards.append(episode_stats["episode_reward"])
            total_steps.append(episode_stats["episode_length"])

            if eval_ep_num == 0 and log_file_path_this_ep:
                first_log_file_path = log_file_path_this_ep

        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_steps = np.mean(total_steps)
        
        print(f"Evaluation Results (over {self.args.evaluate_episodes} episodes):")
        print(f"  Avg Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
        print(f"  Avg Steps: {avg_steps:.2f}")
        print("-"*40)
        
        if first_log_file_path and hasattr(self.args, 'visualize_latest_eval') and self.args.visualize_latest_eval:
            print(f"Attempting to visualize trajectories: {first_log_file_path}")
            try:
                visualizer_args_list = ["python", "visualizer.py", first_log_file_path]
                
                if hasattr(self.args, 'save_visualization_plot') and self.args.save_visualization_plot:
                    visualizer_args_list.extend(["--save_dir", visualization_save_dir])

                subprocess.run(visualizer_args_list, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running visualizer.py for trajectories: {e}")
            except FileNotFoundError:
                print(f"Error: visualizer.py not found or python not in PATH.")
        
        if hasattr(self.args, 'visualize_latest_eval') and self.args.visualize_latest_eval:
            convergence_plot_path = os.path.join(visualization_save_dir, "convergence_curve.png")
            print(f"Attempting to generate convergence curve: {convergence_plot_path}")
            try:
                current_losses_len = len(self.convergence_avg_losses)
                expected_len = len(self.convergence_timesteps)
                if current_losses_len < expected_len:
                    padded_losses = list(self.convergence_avg_losses) + [np.nan] * (expected_len - current_losses_len)
                else:
                    padded_losses = self.convergence_avg_losses

                if self.convergence_timesteps:
                    visualizer.plot_convergence_curve(
                        self.convergence_timesteps,
                        self.convergence_avg_rewards,
                        padded_losses,
                        convergence_plot_path
                    )
                    print(f"Convergence curve saved to {convergence_plot_path}")
                else:
                    print("No data yet to plot convergence curve.")

            except Exception as e:
                print(f"Error generating convergence curve: {e}")

    def save_models(self, identifier):
        model_path = os.path.join(self.args.model_dir, str(identifier))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        self.agents.save_model(model_path, identifier) 

    def load_models(self, path):
        print(f"Loading models from {path}")
        self.agents.load_model(path)
        self.epsilon = self.min_epsilon