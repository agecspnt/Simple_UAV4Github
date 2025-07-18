#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集生成脚本
用于加载训练好的强化学习模型进行1000次评估，生成BS位置和智能体动作的数据集

生成的数据格式:
{"input": "[[-46.7, -36.6]]", "output": "[2. 0. 3.], [1. 0. 3.], [3. 0. 2.], [3. 1. 2.], ..."}
input: BS坐标位置
output: Agent Actions序列
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import json
import random
import math
from glob import glob

# Import project modules
from Multi_UAV_env import Multi_UAV_env_multi_level_practical_multicheck
from agent.agent_UAV import Agents
from common.rollout import RolloutWorker


# 硬编码运行次数
NUM_EVALUATIONS = 10000


def create_evaluation_args():
    """创建评估专用的参数配置"""
    class EvalArgs:
        def __init__(self):
            # 基础参数
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
            self.cuda = True
            self.seed = 123
            self.log_dir = './log/'
            self.model_dir = './model/'
            
            # 算法相关
            self.alg = 'vdn'
            self.epsilon = 0.0  # 评估时不探索
            self.min_epsilon = 0.0
            self.anneal_steps = 50000
            self.anneal_epsilon = 0.0
            self.buffer_size = 5000
            self.batch_size = 64
            
            # 评估模式
            self.evaluate = True
            
            # 网络参数
            self.rnn_hidden_dim = 64
            self.qmix_hidden_dim = 64
            self.two_hyper_layers = False
            self.hyper_hidden_dim = 64
            
            # 地图名称
            self.map = 'default_map'
            
            # 进度条更新间隔
            self.tqdm_mininterval = 0.5
            
    return EvalArgs()


class DatasetGenerator:
    """数据集生成器类"""
    
    def __init__(self, args, model_path):
        """
        初始化生成器
        
        Args:
            args: 命令行参数
            model_path: 模型路径
        """
        self.args = args
        self.model_path = model_path
        
        # 禁用评估日志记录，避免生成不必要的日志文件
        self.args.log_dir = None
        
        # 创建时间戳日志目录
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.dataset_dir = os.path.join("./datasets", f"dataset_{timestamp}")
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        
        print(f"=== 数据集生成开始 ===")
        print(f"模型路径: {model_path}")
        print(f"数据集保存到: {self.dataset_dir}")
        print(f"计划生成 {NUM_EVALUATIONS} 条数据")
        
        # 初始化环境
        self.env = Multi_UAV_env_multi_level_practical_multicheck(args)
        
        # 获取环境信息
        env_info = self.env.get_env_info()
        self.args.n_actions = env_info["n_actions"]
        self.args.n_agents = env_info["n_agents"] 
        self.args.state_shape = env_info["state_shape"]
        self.args.obs_shape = env_info["obs_shape"]
        self.args.episode_limit = env_info["episode_limit"]
        
        print(f"环境信息:")
        print(f"  智能体数量: {self.args.n_agents}")
        print(f"  动作空间大小: {self.args.n_actions}")
        print(f"  状态空间大小: {self.args.state_shape}")
        print(f"  观测空间大小: {self.args.obs_shape}")
        print(f"  最大回合长度: {self.args.episode_limit}")
        
        # 初始化智能体
        self.agents = Agents(args)
        
        # 加载模型
        self.load_model()
        
        # 初始化rollout worker
        self.rollout_worker = RolloutWorker(self.env, self.agents, self.args)
        
        # 用于存储数据集
        self.dataset = []
            
    def load_model(self):
        """加载训练好的模型"""
        try:
            print(f"正在加载模型: {self.model_path}")
            self.agents.load_model(self.model_path)
            print("✅ 模型加载成功!")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)
    
    def set_random_bs_position(self):
        """设置随机的BS位置"""
        # 使用与环境reset相同的逻辑，在半径范围内随机生成
        bs_radius = getattr(self.env, 'bs_radius', 200)  # 默认半径200米
        
        for i in range(self.env.n_bs):
            # 生成随机角度
            angle = random.uniform(0, 2 * math.pi)
            # 生成随机半径 (平方根确保圆内均匀分布)
            r = bs_radius * math.sqrt(random.uniform(0, 1))
            # 转换为笛卡尔坐标
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            self.env.BS_locations[i] = [x, y]
            
    def run_single_evaluation(self, eval_idx):
        """
        运行单次评估
        
        Args:
            eval_idx: 评估索引
            
        Returns:
            dict: 包含BS位置和动作序列的数据条目，如果失败则返回None
        """
        try:
            # 设置不同的随机种子确保每次BS位置不同
            random.seed(self.args.seed + eval_idx)
            np.random.seed(self.args.seed + eval_idx)
            
            # 重置环境 (这会随机化BS位置)
            self.env.reset()
            
            # 再次设置随机BS位置确保每次都不同
            self.set_random_bs_position()
            
            # 获取当前BS位置
            bs_positions = self.env.BS_locations.tolist()
            
            # 运行一个回合
            episode_data, episode_stats, _ = self.rollout_worker.generate_episode(
                episode_num=f"dataset_ep{eval_idx}",
                evaluate=True,
                epsilon=0.0,  # 纯贪心策略，不探索
                log_output_dir=None  # 不生成日志文件
            )
            
            # 提取智能体动作序列
            actions = episode_data['u']  # 形状: (episode_limit, n_agents, 1)
            episode_length = episode_stats["episode_length"]
            
            # 只取实际步数的动作，去掉填充的部分
            valid_actions = actions[:episode_length]  # (episode_length, n_agents, 1)
            
            # 转换动作格式: 从 (episode_length, n_agents, 1) 到 列表字符串
            action_sequence = []
            for step in range(episode_length):
                step_actions = valid_actions[step].flatten()  # (n_agents,)
                # 格式化为 "[action1. action2. action3.]"
                action_str = "[" + " ".join([f"{int(a)}." for a in step_actions]) + "]"
                action_sequence.append(action_str)
            
            # 创建数据条目
            data_entry = {
                "input": str(bs_positions),  # BS位置
                "output": ", ".join(action_sequence)  # 动作序列
            }
            
            if eval_idx % 100 == 0:
                print(f"完成第 {eval_idx + 1}/{NUM_EVALUATIONS} 次评估")
                
            return data_entry
            
        except Exception as e:
            print(f"第 {eval_idx + 1} 次评估失败: {e}")
            return None
            
    def generate_dataset(self):
        """
        生成完整数据集
        """
        print(f"\n=== 开始生成数据集 ===")
        
        success_count = 0
        failure_count = 0
        
        for eval_idx in range(NUM_EVALUATIONS):
            data_entry = self.run_single_evaluation(eval_idx)
            
            if data_entry is not None:
                self.dataset.append(data_entry)
                success_count += 1
            else:
                failure_count += 1
                
        # 保存数据集到JSON文件
        dataset_file = os.path.join(self.dataset_dir, "uav_dataset.json")
        
        try:
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=2)
            
            print(f"\n=== 数据集生成完成 ===")
            print(f"成功生成: {success_count} 条数据")
            print(f"失败次数: {failure_count} 次")
            print(f"数据集文件: {dataset_file}")
            
            # 打印几个示例数据
            print(f"\n示例数据:")
            for i, entry in enumerate(self.dataset[:3]):
                print(f"样本 {i+1}:")
                print(f"  Input (BS位置): {entry['input']}")
                print(f"  Output (动作序列): {entry['output'][:100]}...")  # 只显示前100字符
                
        except Exception as e:
            print(f"保存数据集失败: {e}")


def find_latest_model(model_dir="./model"):
    """查找最新的训练模型"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        
    # 获取所有模型目录
    model_dirs = [d for d in os.listdir(model_dir) 
                  if os.path.isdir(os.path.join(model_dir, d)) and d.isdigit()]
    
    if not model_dirs:
        raise FileNotFoundError(f"在 {model_dir} 中未找到任何模型")
        
    # 按步数排序，获取最新的
    model_dirs.sort(key=int, reverse=True)
    latest_model_path = os.path.join(model_dir, model_dirs[0])
    
    # 验证模型文件是否存在
    q_network_path = os.path.join(latest_model_path, "q_network.pth")
    if not os.path.exists(q_network_path):
        raise FileNotFoundError(f"模型文件不存在: {q_network_path}")
        
    return latest_model_path, int(model_dirs[0])


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="强化学习模型数据集生成脚本")
    parser.add_argument('--model_step', type=int, default=None,
                       help='指定要使用的模型步数')
    parser.add_argument('--model_path', type=str, default=None,
                       help='指定模型路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    eval_args = parser.parse_args()
    
    # 创建评估参数配置
    args = create_evaluation_args()
    
    # 重写种子参数
    args.seed = eval_args.seed
    
    # 设置设备
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {args.device}")
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # 确定模型路径
    if eval_args.model_path:
        model_path = eval_args.model_path
        if not os.path.exists(model_path):
            print(f"指定的模型路径不存在: {model_path}")
            sys.exit(1)
        model_step = os.path.basename(model_path)
    elif eval_args.model_step:
        model_path = f"./model/{eval_args.model_step}"
        if not os.path.exists(model_path):
            print(f"指定步数的模型不存在: {model_path}")
            sys.exit(1)
        model_step = eval_args.model_step
    else:
        # 自动查找最新模型
        try:
            model_path, model_step = find_latest_model()
            print(f"自动找到最新模型: 步数 {model_step}")
        except FileNotFoundError as e:
            print(f"{e}")
            sys.exit(1)
    
    # 创建数据集生成器并运行
    try:
        generator = DatasetGenerator(args, model_path)
        generator.generate_dataset()
        
    except KeyboardInterrupt:
        print("\n数据集生成被用户中断")
    except Exception as e:
        print(f"数据集生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 