import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import argparse
from typing import List, Dict, Tuple, Any
import time
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
import pandas as pd

def load_comparison_results(results_path: str) -> Dict[str, Any]:
    """Load comparison results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_action_sequences(results: Dict[str, Any]) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    """Extract action sequences from model and ChatGPT results."""
    model_sequences = []
    chatgpt_sequences = []
    
    for result in results['model_results']:
        if 'actions' in result and result['actions']:
            model_sequences.append(result['actions'])
    
    for result in results['chatgpt_results']:
        if 'actions' in result and result['actions']:
            chatgpt_sequences.append(result['actions'])
    
    return model_sequences, chatgpt_sequences

def calculate_action_diversity(sequences: List[List[List[int]]]) -> Dict[str, float]:
    """Calculate various diversity metrics for action sequences."""
    if not sequences:
        return {}
    
    all_actions = []
    for sequence in sequences:
        for step in sequence:
            all_actions.extend(step)
    
    action_counts = Counter(all_actions)
    total_actions = len(all_actions)
    
    probabilities = [count / total_actions for count in action_counts.values()]
    action_entropy = entropy(probabilities, base=2)
    
    unique_actions = len(set(all_actions))
    action_variance = np.var(list(action_counts.values()))
    
    return {
        'entropy': action_entropy,
        'unique_actions': unique_actions,
        'variance': action_variance,
        'total_actions': total_actions,
        'action_distribution': dict(action_counts)
    }

def calculate_sequence_diversity(sequences: List[List[List[int]]]) -> Dict[str, float]:
    """Calculate diversity metrics for entire sequences."""
    if not sequences:
        return {}
    
    sequence_strings = []
    for sequence in sequences:
        sequence_str = str(sequence)
        sequence_strings.append(sequence_str)
    
    unique_sequences = len(set(sequence_strings))
    total_sequences = len(sequence_strings)
    sequence_uniqueness = unique_sequences / total_sequences if total_sequences > 0 else 0
    
    sequence_counts = Counter(sequence_strings)
    sequence_probs = [count / total_sequences for count in sequence_counts.values()]
    sequence_entropy = entropy(sequence_probs, base=2)
    
    return {
        'unique_sequences': unique_sequences,
        'total_sequences': total_sequences,
        'uniqueness_ratio': sequence_uniqueness,
        'sequence_entropy': sequence_entropy
    }

def calculate_step_diversity(sequences: List[List[List[int]]]) -> List[Dict[str, float]]:
    """Calculate diversity metrics for each step across all sequences."""
    if not sequences:
        return []
    
    max_steps = max(len(seq) for seq in sequences)
    step_diversities = []
    
    for step_idx in range(max_steps):
        step_actions = []
        for sequence in sequences:
            if step_idx < len(sequence):
                step_actions.append(tuple(sequence[step_idx]))
        
        if step_actions:
            step_counts = Counter(step_actions)
            total_actions = len(step_actions)
            
            probs = [count / total_actions for count in step_counts.values()]
            step_entropy = entropy(probs, base=2)
            
            unique_actions = len(set(step_actions))
            uniqueness_ratio = unique_actions / total_actions if total_actions > 0 else 0
            
            step_diversities.append({
                'step': step_idx,
                'entropy': step_entropy,
                'unique_actions': unique_actions,
                'total_actions': total_actions,
                'uniqueness_ratio': uniqueness_ratio
            })
    
    return step_diversities

def calculate_pairwise_distances(sequences: List[List[List[int]]]) -> Dict[str, float]:
    """Calculate pairwise distances between sequences."""
    if len(sequences) < 2:
        return {}
    
    max_len = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        padded_seq = seq + [[-1, -1, -1]] * (max_len - len(seq))
        flattened = [action for step in padded_seq for action in step]
        padded_sequences.append(flattened)
    
    sequences_array = np.array(padded_sequences)
    
    distances = pairwise_distances(sequences_array, metric='hamming')
    
    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances[distances > 0]),
        'max_distance': np.max(distances)
    }

def analyze_agent_behavior(sequences: List[List[List[int]]], num_agents: int = 3) -> Dict[int, Dict[str, Any]]:
    """Analyze individual agent behavior patterns."""
    agent_behaviors = {}
    
    for agent_id in range(num_agents):
        agent_actions = []
        for sequence in sequences:
            for step in sequence:
                if agent_id < len(step):
                    agent_actions.append(step[agent_id])
        
        if agent_actions:
            action_counts = Counter(agent_actions)
            total_actions = len(agent_actions)
            
            probs = [count / total_actions for count in action_counts.values()]
            agent_entropy = entropy(probs, base=2)
            
            agent_behaviors[agent_id] = {
                'action_distribution': dict(action_counts),
                'entropy': agent_entropy,
                'most_common_action': action_counts.most_common(1)[0][0],
                'total_actions': total_actions
            }
    
    return agent_behaviors

def visualize_diversity_analysis(model_diversity: Dict, chatgpt_diversity: Dict, output_dir: str):
    """Create comprehensive diversity analysis visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods = ['Model', 'ChatGPT']
    
    action_entropies = [model_diversity['action']['entropy'], chatgpt_diversity['action']['entropy']]
    axes[0, 0].bar(methods, action_entropies, color=['blue', 'red'])
    axes[0, 0].set_title('Action Entropy')
    axes[0, 0].set_ylabel('Entropy (bits)')
    
    sequence_entropies = [model_diversity['sequence']['sequence_entropy'], chatgpt_diversity['sequence']['sequence_entropy']]
    axes[0, 1].bar(methods, sequence_entropies, color=['blue', 'red'])
    axes[0, 1].set_title('Sequence Entropy')
    axes[0, 1].set_ylabel('Entropy (bits)')
    
    uniqueness_ratios = [model_diversity['sequence']['uniqueness_ratio'], chatgpt_diversity['sequence']['uniqueness_ratio']]
    axes[0, 2].bar(methods, uniqueness_ratios, color=['blue', 'red'])
    axes[0, 2].set_title('Sequence Uniqueness Ratio')
    axes[0, 2].set_ylabel('Uniqueness Ratio')
    axes[0, 2].set_ylim(0, 1)
    
    if 'step' in model_diversity and 'step' in chatgpt_diversity:
        model_step_entropies = [step['entropy'] for step in model_diversity['step']]
        chatgpt_step_entropies = [step['entropy'] for step in chatgpt_diversity['step']]
        
        steps = range(max(len(model_step_entropies), len(chatgpt_step_entropies)))
        axes[1, 0].plot(steps[:len(model_step_entropies)], model_step_entropies, 'b-', label='Model', marker='o')
        axes[1, 0].plot(steps[:len(chatgpt_step_entropies)], chatgpt_step_entropies, 'r-', label='ChatGPT', marker='s')
        axes[1, 0].set_title('Step-wise Entropy')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Entropy (bits)')
        axes[1, 0].legend()
    
    if 'distance' in model_diversity and 'distance' in chatgpt_diversity:
        mean_distances = [model_diversity['distance']['mean_distance'], chatgpt_diversity['distance']['mean_distance']]
        axes[1, 1].bar(methods, mean_distances, color=['blue', 'red'])
        axes[1, 1].set_title('Mean Pairwise Distance')
        axes[1, 1].set_ylabel('Mean Distance')
    
    model_actions = list(model_diversity['action']['action_distribution'].keys())
    model_counts = list(model_diversity['action']['action_distribution'].values())
    chatgpt_actions = list(chatgpt_diversity['action']['action_distribution'].keys())
    chatgpt_counts = list(chatgpt_diversity['action']['action_distribution'].values())
    
    all_actions = sorted(set(model_actions + chatgpt_actions))
    model_action_counts = [model_diversity['action']['action_distribution'].get(action, 0) for action in all_actions]
    chatgpt_action_counts = [chatgpt_diversity['action']['action_distribution'].get(action, 0) for action in all_actions]
    
    x = np.arange(len(all_actions))
    width = 0.35
    
    axes[1, 2].bar(x - width/2, model_action_counts, width, label='Model', color='blue', alpha=0.7)
    axes[1, 2].bar(x + width/2, chatgpt_action_counts, width, label='ChatGPT', color='red', alpha=0.7)
    axes[1, 2].set_title('Action Distribution')
    axes[1, 2].set_xlabel('Action')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(all_actions)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diversity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_agent_behavior_heatmap(model_behaviors: Dict, chatgpt_behaviors: Dict, output_dir: str):
    """Create heatmaps showing agent behavior patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    model_data = []
    chatgpt_data = []
    
    for agent_id in range(3):
        if agent_id in model_behaviors:
            model_dist = model_behaviors[agent_id]['action_distribution']
            model_data.append([model_dist.get(action, 0) for action in range(4)])
    else:
            model_data.append([0, 0, 0, 0])
        
        if agent_id in chatgpt_behaviors:
            chatgpt_dist = chatgpt_behaviors[agent_id]['action_distribution']
            chatgpt_data.append([chatgpt_dist.get(action, 0) for action in range(4)])
        else:
            chatgpt_data.append([0, 0, 0, 0])
    
    model_df = pd.DataFrame(model_data, index=[f'Agent {i}' for i in range(3)], 
                           columns=[f'Action {i}' for i in range(4)])
    chatgpt_df = pd.DataFrame(chatgpt_data, index=[f'Agent {i}' for i in range(3)], 
                             columns=[f'Action {i}' for i in range(4)])
    
    sns.heatmap(model_df, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Model Agent Behavior')
    
    sns.heatmap(chatgpt_df, annot=True, fmt='d', cmap='Reds', ax=axes[1])
    axes[1].set_title('ChatGPT Agent Behavior')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_behavior_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()

def comprehensive_diversity_analysis(results_path: str, output_dir: str = None):
    """Perform comprehensive diversity analysis on comparison results."""
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f"diversity_analysis_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = load_comparison_results(results_path)
    model_sequences, chatgpt_sequences = extract_action_sequences(results)
    
    print(f"Loaded {len(model_sequences)} model sequences and {len(chatgpt_sequences)} ChatGPT sequences")
    
    model_diversity = {
        'action': calculate_action_diversity(model_sequences),
        'sequence': calculate_sequence_diversity(model_sequences),
        'step': calculate_step_diversity(model_sequences),
        'distance': calculate_pairwise_distances(model_sequences),
        'agent_behavior': analyze_agent_behavior(model_sequences)
    }
    
    chatgpt_diversity = {
        'action': calculate_action_diversity(chatgpt_sequences),
        'sequence': calculate_sequence_diversity(chatgpt_sequences),
        'step': calculate_step_diversity(chatgpt_sequences),
        'distance': calculate_pairwise_distances(chatgpt_sequences),
        'agent_behavior': analyze_agent_behavior(chatgpt_sequences)
    }
    
    diversity_results = {
        'model_diversity': model_diversity,
        'chatgpt_diversity': chatgpt_diversity,
        'comparison_summary': {
            'model_action_entropy': model_diversity['action']['entropy'],
            'chatgpt_action_entropy': chatgpt_diversity['action']['entropy'],
            'model_sequence_entropy': model_diversity['sequence']['sequence_entropy'],
            'chatgpt_sequence_entropy': chatgpt_diversity['sequence']['sequence_entropy'],
            'model_uniqueness_ratio': model_diversity['sequence']['uniqueness_ratio'],
            'chatgpt_uniqueness_ratio': chatgpt_diversity['sequence']['uniqueness_ratio'],
            'model_mean_distance': model_diversity['distance'].get('mean_distance', 0),
            'chatgpt_mean_distance': chatgpt_diversity['distance'].get('mean_distance', 0)
        }
    }
    
    results_file = os.path.join(output_dir, 'diversity_results.json')
    with open(results_file, 'w') as f:
        json.dump(diversity_results, f, indent=2)
    
    print("\n=== Diversity Analysis Results ===")
    print(f"Model Action Entropy: {model_diversity['action']['entropy']:.3f}")
    print(f"ChatGPT Action Entropy: {chatgpt_diversity['action']['entropy']:.3f}")
    print(f"Model Sequence Entropy: {model_diversity['sequence']['sequence_entropy']:.3f}")
    print(f"ChatGPT Sequence Entropy: {chatgpt_diversity['sequence']['sequence_entropy']:.3f}")
    print(f"Model Uniqueness Ratio: {model_diversity['sequence']['uniqueness_ratio']:.3f}")
    print(f"ChatGPT Uniqueness Ratio: {chatgpt_diversity['sequence']['uniqueness_ratio']:.3f}")
    
    if model_diversity['distance'] and chatgpt_diversity['distance']:
        print(f"Model Mean Distance: {model_diversity['distance']['mean_distance']:.3f}")
        print(f"ChatGPT Mean Distance: {chatgpt_diversity['distance']['mean_distance']:.3f}")
    
    visualize_diversity_analysis(model_diversity, chatgpt_diversity, output_dir)
    create_agent_behavior_heatmap(model_diversity['agent_behavior'], chatgpt_diversity['agent_behavior'], output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return diversity_results

def main():
    parser = argparse.ArgumentParser(description="Analyze output diversity of RL model vs ChatGPT")
    parser.add_argument('--results_path', type=str, required=True, 
                        help='Path to comparison results JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_path):
        print(f"Error: Results file not found: {args.results_path}")
        return
    
    try:
        results = comprehensive_diversity_analysis(args.results_path, args.output_dir)
        print("Diversity analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 