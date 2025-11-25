"""
Comprehensive evaluation of transfer quality with multiple metrics.

Metrics:
1. Vector Similarity Metrics (on test set):
   - MSE (Mean Squared Error)
   - Cosine Similarity
   - L2 Distance
   - Pearson Correlation

2. Reconstruction Quality:
   - Per-dimension error analysis
   - Top-k dimension preservation

3. Transfer Generalization:
   - Train vs Test performance comparison
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transferability.scripts.transfer import LinearTransfer, MLPTransfer, load_vectors


def convert_to_serializable(obj):
    """Convert numpy/torch types to Python native types for JSON serialization."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def compute_similarity_metrics(source_vectors, target_vectors, transferred_vectors):
    """Compute comprehensive similarity metrics."""
    metrics = {}
    
    # MSE
    mse = nn.MSELoss()(transferred_vectors, target_vectors).item()
    metrics['mse'] = mse
    
    # Cosine Similarity (average across all vectors)
    cosine_sim = nn.CosineSimilarity(dim=1)(transferred_vectors, target_vectors)
    metrics['cosine_similarity_mean'] = cosine_sim.mean().item()
    metrics['cosine_similarity_std'] = cosine_sim.std().item()
    metrics['cosine_similarity_min'] = cosine_sim.min().item()
    metrics['cosine_similarity_max'] = cosine_sim.max().item()
    
    # L2 Distance
    l2_dist = torch.norm(transferred_vectors - target_vectors, p=2, dim=1)
    metrics['l2_distance_mean'] = l2_dist.mean().item()
    metrics['l2_distance_std'] = l2_dist.std().item()
    
    # Pearson Correlation (average across dimensions)
    correlations = []
    for dim in range(transferred_vectors.shape[1]):
        corr, _ = pearsonr(
            transferred_vectors[:, dim].cpu().numpy(),
            target_vectors[:, dim].cpu().numpy()
        )
        correlations.append(corr)
    metrics['pearson_correlation_mean'] = np.mean(correlations)
    metrics['pearson_correlation_std'] = np.std(correlations)
    
    # Relative Error
    relative_error = torch.abs(transferred_vectors - target_vectors) / (torch.abs(target_vectors) + 1e-8)
    metrics['relative_error_mean'] = relative_error.mean().item()
    metrics['relative_error_median'] = relative_error.median().item()
    
    return metrics


def evaluate_top_k_preservation(source_vectors, target_vectors, transferred_vectors, k=100):
    """Evaluate how well top-k dimensions are preserved."""
    metrics = {}
    
    # For each vector, find top-k dimensions in target
    # and check if they're also top-k in transferred
    overlaps = []
    for i in range(target_vectors.shape[0]):
        target_topk = torch.topk(torch.abs(target_vectors[i]), k).indices
        transferred_topk = torch.topk(torch.abs(transferred_vectors[i]), k).indices
        
        # Compute overlap
        overlap = len(set(target_topk.cpu().numpy()) & set(transferred_topk.cpu().numpy()))
        overlaps.append(overlap / k)
    
    metrics[f'top_{k}_overlap_mean'] = np.mean(overlaps)
    metrics[f'top_{k}_overlap_std'] = np.std(overlaps)
    
    return metrics


def evaluate_transfer_method(config, source_model, target_model, method, train_indices, test_indices, device):
    """Evaluate a single transfer method."""
    print(f"\nEvaluating {method} transfer: {source_model} -> {target_model}")
    
    # Load vectors
    vectors_dir = config['transfer']['output']['vectors_dir']
    num_concepts = config['transfer']['total_concepts']
    
    source_vectors = load_vectors(vectors_dir, source_model, num_concepts)
    target_vectors = load_vectors(vectors_dir, target_model, num_concepts)
    
    if source_vectors is None or target_vectors is None:
        print("Error: Could not load vectors")
        return None
    
    # Split into train and test
    source_train = source_vectors[train_indices]
    source_test = source_vectors[test_indices]
    target_train = target_vectors[train_indices]
    target_test = target_vectors[test_indices]
    
    # Load or apply transfer model
    if method == 'identity':
        # Check dimensions
        if source_vectors.shape[1] != target_vectors.shape[1]:
            print(f"Skipping identity transfer: Dimension mismatch ({source_vectors.shape[1]} vs {target_vectors.shape[1]})")
            return None
            
        # No transformation
        transferred_train = source_train.to(device)
        transferred_test = source_test.to(device)
    else:
        # Load trained transfer model
        transfer_models_dir = config['transfer']['output']['transfer_models_dir']
        model_path = os.path.join(transfer_models_dir, f"{source_model}_to_{target_model}_{method}.pt")
        
        if not os.path.exists(model_path):
            print(f"Warning: Transfer model not found: {model_path}")
            return None
        
        # Create and load model
        input_dim = source_vectors.shape[1]
        output_dim = target_vectors.shape[1]
        
        if method == 'linear':
            transfer_model = LinearTransfer(input_dim, output_dim).to(device)
        elif method == 'mlp':
            transfer_model = MLPTransfer(
                input_dim, output_dim,
                hidden_dims=config['transfer']['mlp']['hidden_dims']
            ).to(device)
        
        transfer_model.load_state_dict(torch.load(model_path))
        transfer_model.eval()
        
        with torch.no_grad():
            transferred_train = transfer_model(source_train.to(device))
            transferred_test = transfer_model(source_test.to(device))
    
    # Compute metrics on train set
    print("  Computing train metrics...")
    train_metrics = compute_similarity_metrics(
        source_train.to(device),
        target_train.to(device),
        transferred_train
    )
    train_metrics.update(evaluate_top_k_preservation(
        source_train.to(device),
        target_train.to(device),
        transferred_train,
        k=100
    ))
    
    # Compute metrics on test set
    print("  Computing test metrics...")
    test_metrics = compute_similarity_metrics(
        source_test.to(device),
        target_test.to(device),
        transferred_test
    )
    test_metrics.update(evaluate_top_k_preservation(
        source_test.to(device),
        target_test.to(device),
        transferred_test,
        k=100
    ))
    
    # Compute generalization gap
    generalization_metrics = {
        'mse_gap': test_metrics['mse'] - train_metrics['mse'],
        'cosine_gap': train_metrics['cosine_similarity_mean'] - test_metrics['cosine_similarity_mean'],
        'l2_gap': test_metrics['l2_distance_mean'] - train_metrics['l2_distance_mean'],
    }
    
    return {
        'train': train_metrics,
        'test': test_metrics,
        'generalization': generalization_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive transfer evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(config['transfer']['seed'])
    np.random.seed(config['transfer']['seed'])
    
    # Create train/test split
    total_concepts = config['transfer']['total_concepts']
    train_concepts = config['transfer']['train_concepts']
    
    # Deterministic split to match transfer.py
    # transfer.py trains on the first `train_concepts`
    indices = np.arange(total_concepts)
    train_indices = indices[:train_concepts]
    test_indices = indices[train_concepts:]
    
    print(f"Train concepts: {train_indices.tolist()}")
    print(f"Test concepts: {test_indices.tolist()}")
    
    # Evaluate all combinations
    all_results = {}
    
    models = config['transfer']['models']
    methods = config['transfer']['transfer_methods']
    
    for source_model_config in models:
        for target_model_config in models:
            source_name = source_model_config['name']
            target_name = target_model_config['name']
            
            for method in methods:
                key = f"{source_name}_to_{target_name}_{method}"
                print(f"\n{'='*60}")
                print(f"Evaluating: {key}")
                print(f"{'='*60}")
                
                results = evaluate_transfer_method(
                    config, source_name, target_name, method,
                    train_indices, test_indices, args.device
                )
                
                if results:
                    all_results[key] = results
                    
                    # Print summary
                    print(f"\nTrain Performance:")
                    print(f"  MSE: {results['train']['mse']:.6f}")
                    print(f"  Cosine Sim: {results['train']['cosine_similarity_mean']:.4f}")
                    print(f"\nTest Performance:")
                    print(f"  MSE: {results['test']['mse']:.6f}")
                    print(f"  Cosine Sim: {results['test']['cosine_similarity_mean']:.4f}")
                    print(f"\nGeneralization:")
                    print(f"  MSE Gap: {results['generalization']['mse_gap']:.6f}")
                    print(f"  Cosine Gap: {results['generalization']['cosine_gap']:.4f}")
    
    # Save results
    output_dir = config['transfer']['output']['evaluation_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to serializable format
    serializable_results = convert_to_serializable(all_results)
    
    results_path = os.path.join(output_dir, "comprehensive_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save split indices
    split_path = os.path.join(output_dir, "train_test_split.json")
    with open(split_path, 'w') as f:
        json.dump({
            'train_indices': train_indices.tolist(),
            'test_indices': test_indices.tolist()
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
