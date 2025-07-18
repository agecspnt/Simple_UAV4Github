import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
from typing import List, Dict, Tuple, Optional
import re

from Multi_UAV_env import Multi_UAV_env_multi_level_practical_multicheck
from agent.agent_UAV import Agents
from openai import OpenAI

api_key = "use your own key"

client = OpenAI(api_key=api_key)

class ModelEvaluator:
    def __init__(self, model_path: str, args):
        self.model_path = model_path
        self.args = args
        self.env = Multi_UAV_env_multi_level_practical_multicheck(args)
        self.agents = Agents(args)
        self.load_model()
        
    def load_model(self):
        try:
            self.agents.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def evaluate_scenario(self, bs_location: List[List[float]], max_steps: int = 11) -> Dict:
        self.env.reset()
        self.env.BS_locations = np.array(bs_location)
        
        obs = self.env.get_obs()
        state = self.env.get_state()
        
        total_reward = 0
        step_count = 0
        done = False
        episode_actions = []
        
        while not done and step_count < max_steps:
            actions = []
            
            for agent_id in range(self.env.n_a_agents):
                obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
                if self.args.cuda:
                    obs_tensor = obs_tensor.cuda()
                
                q_values = self.agents.policy.get_agent_q_values(obs[agent_id], agent_id)
                action = q_values.argmax().item()
                actions.append(action)
            
            episode_actions.append(actions.copy())
            
            reward, done, info = self.env.step(actions)
            total_reward += reward
            step_count += 1
            
            obs = self.env.get_obs()
            state = self.env.get_state()
        
        return {
            'total_reward': total_reward,
            'steps': step_count,
            'success': info.get('is_success', False),
            'actions': episode_actions,
            'collisions': info.get('collisions', 0)
        }

class ChatGPTEvaluator:
    def __init__(self):
        self.system_prompt = """You are simulating 3 UAVs navigating from fixed initial positions to specific destinations(fixed straight line). 
Each UAV selects one of 4 discrete actions at each step: (velocity, channel), where:
- Velocity levels: 0 = stay, 1 = 100m/s
- Channels: 0 or 1
- Action ID = velocity * 2 + channel ∈ {0, 1, 2, 3}

There is 1 base station randomly placed within a 200m radius around the origin, used for maintaining communication quality.
At each step, all 3 UAVs choose actions simultaneously.

Your goal is to predict a sequence of 11 steps (one per line), each containing 3 integers representing the actions of UAV1, UAV2, and UAV3.

Initial positions (fixed):
- UAV1: [-250.0, 400.0] → [1250.0, 400.0]
- UAV2: [-30.33, 930.33] → [1030.33, -130.33]
- UAV3: [-30.33, -130.33] → [1030.33, 930.33]

Some examples:
[
  {
    "id": 1,
    "input": "[[-14.798741849521901, 93.3243096177203]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 2,
    "input": "[[84.08579057136284, 106.91524688814226]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 2. 2.], [3. 1. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 2.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 3,
    "input": "[[54.408355174320576, -64.08086028638142]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [0. 2. 3.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 4,
    "input": "[[-157.43567421920204, -98.00752738874569]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 3.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 3.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 5,
    "input": "[[-112.28148638473567, 33.640116681791476]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [0. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 6,
    "input": "[[-26.30822353055179, -106.76085242905822]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 3.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 7,
    "input": "[[133.2932400065467, 110.98365026119164]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [0. 2. 3.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 8,
    "input": "[[-64.00233644152553, -74.85246219982534]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 9,
    "input": "[[41.5183367005526, -105.36057372525583]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 3.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 10,
    "input": "[[117.24876359028478, -74.99987566111726]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [1. 2. 3.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 3.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 11,
    "input": "[[-3.1093756383286153, -190.56766331567707]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [0. 2. 3.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 3.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 12,
    "input": "[[-124.90734671544493, -53.26001373579176]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [0. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 2.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 13,
    "input": "[[169.26708570357036, 45.31825256735777]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 2.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 14,
    "input": "[[33.64970438706573, 103.66610975318314]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [1. 2. 3.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 15,
    "input": "[[129.38174061634993, -91.00472419812901]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [0. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 2.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 16,
    "input": "[[-29.070649832134045, 38.78001852127635]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 17,
    "input": "[[152.48363574548958, -46.47046453208972]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 2. 2.], [3. 1. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 2.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 18,
    "input": "[[151.75308359963827, -113.55458138891966]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 3.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 3.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 19,
    "input": "[[-6.6872837918776185, 106.90184282998]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 2. 2.], [3. 1. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 2.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 20,
    "input": "[[-44.41392766745752, 24.459924845783853]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 3.], [3. 1. 2.], [2. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 3.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 21,
    "input": "[[-182.2450931826297, 4.476015454892539]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 2.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 22,
    "input": "[[-85.20953483950946, -103.60615728170626]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [0. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 23,
    "input": "[[185.98990041368552, 32.66087591077309]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [1. 2. 3.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 24,
    "input": "[[119.76654543103824, -7.00570468308494]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [1. 2. 3.], [3. 1. 2.], [3. 1. 2.], [2. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 3.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 25,
    "input": "[[17.40026986282237, -104.92789087451595]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 1.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 26,
    "input": "[[-145.57922000181586, -122.02281083311408]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 2.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 27,
    "input": "[[-46.77009396584732, 122.82417141989643]]",
    "output": "[1. 3. 2.], [1. 3. 2.], [1. 2. 3.], [3. 1. 2.], [2. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 3.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 28,
    "input": "[[-22.17965884307347, -139.785607469715]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [1. 2. 2.], [3. 1. 2.], [3. 1. 2.], [3. 3. 2.], [3. 3. 2.], [2. 2. 2.], [2. 2. 2.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 29,
    "input": "[[-182.04335433615327, -43.6529612751129]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [0. 2. 2.], [3. 1. 2.], [3. 2. 2.], [3. 1. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 2.], [2. 2. 2.], [2. 0. 2.]"
  },
  {
    "id": 30,
    "input": "[[71.48998310057539, 169.8125480525834]]",
    "output": "[0. 3. 2.], [1. 3. 2.], [0. 2. 3.], [3. 1. 2.], [3. 1. 2.], [2. 3. 2.], [3. 3. 2.], [2. 3. 2.], [2. 2. 3.], [2. 2. 2.], [2. 0. 2.]"
  }
]

Think step by step about the optimal path for each UAV considering:
1. The base station position for communication
2. The shortest path to destination
3. Avoiding collisions between UAVs
4. Channel allocation for communication quality

Please provide your reasoning and then output the action sequence."""
        
        self.env = None
        
    def get_action_sequence(self, bs_location: List[List[float]], max_retries: int = 3) -> Optional[List[List[int]]]:
        user_prompt = f"""Base station location: {bs_location}

Generate an optimal 11-step action sequence for the 3 UAVs considering:
1. Efficient path to destinations
2. Collision avoidance
3. Communication quality with base station
4. Channel allocation optimization

Provide the sequence in the exact format requested."""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                ai_response = response.choices[0].message.content
                action_sequence = self.extract_action_sequence(ai_response)
                
                if action_sequence:
                  return action_sequence
                else:
                    print(f"Failed to extract valid sequence from response (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    continue
            except Exception as e:
                print(f"Error in API call (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise
        
        return None
    
    def extract_action_sequence(self, response_text: str) -> Optional[List[List[int]]]:
        pattern = r'\[(\d+)[,\s]+(\d+)[,\s]+(\d+)\]'
        matches = re.findall(pattern, response_text)
        
        if len(matches) < 10:
                    return None
        
        action_sequence = []
        for match in matches[:11]:
            try:
                actions = [int(match[0]), int(match[1]), int(match[2])]
                if all(0 <= a <= 3 for a in actions):
                    action_sequence.append(actions)
                else:
                    return None
            except ValueError:
              return None
    
        return action_sequence
    
    def evaluate_scenario(self, bs_location: List[List[float]], args) -> Dict:
        if self.env is None:
            self.env = Multi_UAV_env_multi_level_practical_multicheck(args)
        
        action_sequence = self.get_action_sequence(bs_location)
        
        if action_sequence is None:
            return {
                'total_reward': -1000,
                'steps': 11,
                'success': False,
                'actions': [],
                'collisions': 0,
                'error': 'Failed to generate valid action sequence'
            }
        
        self.env.reset()
        self.env.BS_locations = np.array(bs_location)
        
        total_reward = 0
        step_count = 0
        done = False
        
        for step_actions in action_sequence:
            if done:
                break
                
            reward, done, info = self.env.step(step_actions)
            total_reward += reward
            step_count += 1
        
        return {
            'total_reward': total_reward,
            'steps': step_count,
            'success': info.get('is_success', False),
            'actions': action_sequence,
            'collisions': info.get('collisions', 0)
        }

def create_evaluation_args():
    class EvalArgs:
          def __init__(self):
            self.n_steps = 2000000
            self.n_episodes = 1
            self.episode_limit = 50
            self.last_action = True
            self.reuse_network = True
            self.gamma = 0.99
            self.optimizer = "Adam"
            self.evaluate_cycle = 5000
            self.evaluate_episodes = 32
            self.lr = 5e-4
            self.grad_norm_clip = 10
            self.save_cycle = 5000
            self.target_update_cycle = 200
            self.cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if self.cuda else "cpu")
            self.seed = 123
            self.log_dir = './log/'
            self.model_dir = './model/'
            self.alg = 'vdn'
            self.epsilon = 0.0
            self.min_epsilon = 0.0
            self.anneal_steps = 50000
            self.anneal_epsilon = 0.0
            self.buffer_size = 5000
            self.batch_size = 64
            self.evaluate = True
            self.rnn_hidden_dim = 64
            self.qmix_hidden_dim = 64
            self.two_hyper_layers = False
            self.hyper_hidden_dim = 64
            self.map = 'default_map'
            self.tqdm_mininterval = 0.5
            
    return EvalArgs()

def generate_test_scenarios(num_scenarios: int = 100) -> List[List[List[float]]]:
    scenarios = []
    for i in range(num_scenarios):
        bs_location = []
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, 200)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        bs_location.append([x, y])
        scenarios.append(bs_location)
    return scenarios

def run_comparison(model_path: str, num_scenarios: int = 100, output_dir: str = "comparison_results"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    full_output_dir = os.path.join(output_dir, f"comparison_results_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    
    args = create_evaluation_args()
    
    env_info = Multi_UAV_env_multi_level_practical_multicheck(args).get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    
    model_evaluator = ModelEvaluator(model_path, args)
    chatgpt_evaluator = ChatGPTEvaluator()
    
    scenarios = generate_test_scenarios(num_scenarios)
    
    results = {
        'model_results': [],
        'chatgpt_results': [],
        'scenarios': scenarios
    }
    
    print(f"Running comparison on {num_scenarios} scenarios...")
    
    for i, scenario in enumerate(tqdm(scenarios, desc="Evaluating scenarios")):
        model_result = model_evaluator.evaluate_scenario(scenario)
        chatgpt_result = chatgpt_evaluator.evaluate_scenario(scenario, args)
        
        results['model_results'].append(model_result)
        results['chatgpt_results'].append(chatgpt_result)
        
        if i % 10 == 0:
            print(f"Completed {i+1}/{num_scenarios} scenarios")
    
    results_file = os.path.join(full_output_dir, "comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    analyze_and_visualize_results(results, full_output_dir)
        
    return results
    
def analyze_and_visualize_results(results: Dict, output_dir: str):
    model_results = results['model_results']
    chatgpt_results = results['chatgpt_results']
    
    model_rewards = [r['total_reward'] for r in model_results]
    chatgpt_rewards = [r['total_reward'] for r in chatgpt_results]
    
    model_success_rate = sum(1 for r in model_results if r['success']) / len(model_results)
    chatgpt_success_rate = sum(1 for r in chatgpt_results if r['success']) / len(chatgpt_results)
    
    model_avg_reward = np.mean(model_rewards)
    chatgpt_avg_reward = np.mean(chatgpt_rewards)
    
    model_avg_steps = np.mean([r['steps'] for r in model_results])
    chatgpt_avg_steps = np.mean([r['steps'] for r in chatgpt_results])
    
    model_avg_collisions = np.mean([r['collisions'] for r in model_results])
    chatgpt_avg_collisions = np.mean([r['collisions'] for r in chatgpt_results])
    
    print("\n=== Comparison Results ===")
    print(f"Model Average Reward: {model_avg_reward:.2f}")
    print(f"ChatGPT Average Reward: {chatgpt_avg_reward:.2f}")
    print(f"Model Success Rate: {model_success_rate:.2%}")
    print(f"ChatGPT Success Rate: {chatgpt_success_rate:.2%}")
    print(f"Model Average Steps: {model_avg_steps:.2f}")
    print(f"ChatGPT Average Steps: {chatgpt_avg_steps:.2f}")
    print(f"Model Average Collisions: {model_avg_collisions:.2f}")
    print(f"ChatGPT Average Collisions: {chatgpt_avg_collisions:.2f}")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(model_rewards, bins=30, alpha=0.7, label='Model', color='blue')
    plt.hist(chatgpt_rewards, bins=30, alpha=0.7, label='ChatGPT', color='red')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    methods = ['Model', 'ChatGPT']
    success_rates = [model_success_rate, chatgpt_success_rate]
    plt.bar(methods, success_rates, color=['blue', 'red'])
    plt.ylabel('Success Rate')
    plt.title('Success Rate Comparison')
    plt.ylim(0, 1)
    
    plt.subplot(2, 3, 3)
    avg_rewards = [model_avg_reward, chatgpt_avg_reward]
    plt.bar(methods, avg_rewards, color=['blue', 'red'])
    plt.ylabel('Average Reward')
    plt.title('Average Reward Comparison')
    
    plt.subplot(2, 3, 4)
    avg_steps = [model_avg_steps, chatgpt_avg_steps]
    plt.bar(methods, avg_steps, color=['blue', 'red'])
    plt.ylabel('Average Steps')
    plt.title('Average Steps Comparison')
    
    plt.subplot(2, 3, 5)
    avg_collisions = [model_avg_collisions, chatgpt_avg_collisions]
    plt.bar(methods, avg_collisions, color=['blue', 'red'])
    plt.ylabel('Average Collisions')
    plt.title('Average Collisions Comparison')
    
    plt.subplot(2, 3, 6)
    plt.scatter(model_rewards, chatgpt_rewards, alpha=0.6)
    plt.xlabel('Model Reward')
    plt.ylabel('ChatGPT Reward')
    plt.title('Reward Correlation')
    min_reward = min(min(model_rewards), min(chatgpt_rewards))
    max_reward = max(max(model_rewards), max(chatgpt_rewards))
    plt.plot([min_reward, max_reward], [min_reward, max_reward], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    summary = {
        'model_avg_reward': model_avg_reward,
        'chatgpt_avg_reward': chatgpt_avg_reward,
        'model_success_rate': model_success_rate,
        'chatgpt_success_rate': chatgpt_success_rate,
        'model_avg_steps': model_avg_steps,
        'chatgpt_avg_steps': chatgpt_avg_steps,
        'model_avg_collisions': model_avg_collisions,
        'chatgpt_avg_collisions': chatgpt_avg_collisions
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis saved to {output_dir}")

def find_latest_model(model_dir: str = "./model") -> str:
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    model_dirs = [d for d in os.listdir(model_dir) 
                  if os.path.isdir(os.path.join(model_dir, d)) and d.isdigit()]
    
    if not model_dirs:
        raise FileNotFoundError(f"No models found in {model_dir}")
    
    model_dirs.sort(key=int, reverse=True)
    latest_model_path = os.path.join(model_dir, model_dirs[0])
    
    q_network_path = os.path.join(latest_model_path, "q_network.pth")
    if not os.path.exists(q_network_path):
        raise FileNotFoundError(f"Model file not found: {q_network_path}")
    
    return latest_model_path

def main():
    parser = argparse.ArgumentParser(description="Compare RL model with ChatGPT")
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--num_scenarios', type=int, default=100, help='Number of test scenarios')
    parser.add_argument('--output_dir', type=str, default='comparison_results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.model_path is None:
        try:
            model_path = find_latest_model()
            print(f"Using latest model: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    else:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"Error: Model path does not exist: {model_path}")
            return
    
    try:
        results = run_comparison(model_path, args.num_scenarios, args.output_dir)
        print("Comparison completed successfully!")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 