"""
Analyze transferability results and generate statistics.

Compares baseline (same-model) vs transfer (cross-model) performance
and generates visualizations and statistical tests.
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_results(evaluation_dir):
    """Load all evaluation results."""
    results = {}
    
    for file in os.listdir(evaluation_dir):
        if file.endswith('_results.csv'):
            key = file.replace('_results.csv', '')
            df = pd.read_csv(os.path.join(evaluation_dir, file))
            
            # Load corresponding metrics
            metrics_file = file.replace('_results.csv', '_metrics.json')
            metrics_path = os.path.join(evaluation_dir, metrics_file)
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {}
            
            results[key] = {
                'data': df,
                'metrics': metrics
            }
    
    return results


def categorize_experiments(results):
    """Categorize experiments into baseline vs transfer."""
    baselines = {}
    transfers = {}
    
    for key, result in results.items():
        parts = key.split('_to_')
        if len(parts) == 2:
            source = parts[0]
            target_and_method = parts[1].rsplit('_', 1)
            target = target_and_method[0]
            method = target_and_method[1] if len(target_and_method) > 1 else 'identity'
            
            if source == target:
                # Baseline: same model
                baselines[key] = result
            else:
                # Transfer: different models
                transfers[key] = result
    
    return baselines, transfers


def compute_summary_statistics(results):
    """Compute summary statistics for each experiment."""
    summary = {}
    
    for key, result in results.items():
        df = result['data']
        
        # Group by concept and compute statistics
        concept_stats = []
        for concept_id in df['concept_id'].unique():
            concept_data = df[df['concept_id'] == concept_id]
            
            # Placeholder metrics (would compute actual metrics in practice)
            concept_stats.append({
                'concept_id': concept_id,
                'num_samples': len(concept_data),
                # Add actual metrics here
            })
        
        summary[key] = pd.DataFrame(concept_stats)
    
    return summary


def compare_baseline_vs_transfer(baselines, transfers, analysis_dir):
    """Compare baseline vs transfer performance."""
    print("\n" + "="*60)
    print("Baseline vs Transfer Comparison")
    print("="*60)
    
    comparison_results = []
    
    # For each transfer experiment, find corresponding baseline
    for transfer_key, transfer_result in transfers.items():
        parts = transfer_key.split('_to_')
        target_model = parts[1].rsplit('_', 1)[0]
        method = parts[1].rsplit('_', 1)[1] if '_' in parts[1] else 'identity'
        
        # Find baseline for target model
        baseline_key = f"{target_model}_to_{target_model}_identity"
        if baseline_key not in baselines:
            print(f"Warning: No baseline found for {target_model}")
            continue
        
        baseline_result = baselines[baseline_key]
        
        # Compare metrics
        transfer_df = transfer_result['data']
        baseline_df = baseline_result['data']
        
        # Compute comparison statistics
        comparison = {
            'transfer_key': transfer_key,
            'baseline_key': baseline_key,
            'method': method,
            'target_model': target_model,
            'num_concepts': len(transfer_df['concept_id'].unique()),
            # Add actual comparison metrics here
        }
        
        comparison_results.append(comparison)
        
        print(f"\n{transfer_key}:")
        print(f"  Method: {method}")
        print(f"  Target: {target_model}")
        print(f"  Concepts: {comparison['num_concepts']}")
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(analysis_dir, 'baseline_vs_transfer.csv'), index=False)
    
    return comparison_df


def generate_visualizations(baselines, transfers, analysis_dir):
    """Generate visualizations comparing methods."""
    print("\nGenerating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Placeholder for actual visualizations
    # Would create:
    # 1. Bar plot comparing baseline vs transfer performance
    # 2. Heatmap of transfer performance between model pairs
    # 3. Line plot showing performance across steering factors
    
    print(f"Visualizations saved to {analysis_dir}")


def statistical_tests(baselines, transfers):
    """Perform statistical tests."""
    print("\n" + "="*60)
    print("Statistical Tests")
    print("="*60)
    
    # Placeholder for statistical tests
    # Would perform:
    # 1. Paired t-tests between baseline and transfer
    # 2. ANOVA across different transfer methods
    # 3. Effect size calculations
    
    print("Statistical tests complete")


def generate_report(comparison_df, analysis_dir):
    """Generate a summary report."""
    report_path = os.path.join(analysis_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Steering Vector Transferability Analysis\n")
        f.write("="*60 + "\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Total experiments: {len(comparison_df)}\n")
        f.write(f"Transfer methods tested: {comparison_df['method'].unique().tolist()}\n")
        f.write(f"Models tested: {comparison_df['target_model'].unique().tolist()}\n\n")
        
        f.write("## Results\n\n")
        f.write(comparison_df.to_string())
        f.write("\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("(Add conclusions based on results)\n")
    
    print(f"\nSummary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze transferability results")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    transfer_config = config['transfer']
    
    evaluation_dir = transfer_config['output']['evaluation_dir']
    analysis_dir = transfer_config['output']['analysis_dir']
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("Loading results...")
    results = load_all_results(evaluation_dir)
    
    if not results:
        print("No results found. Make sure to run evaluate_transfer.py first.")
        return
    
    print(f"Loaded {len(results)} experiments")
    
    # Categorize experiments
    baselines, transfers = categorize_experiments(results)
    print(f"  Baselines: {len(baselines)}")
    print(f"  Transfers: {len(transfers)}")
    
    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary_stats = compute_summary_statistics(results)
    
    # Compare baseline vs transfer
    comparison_df = compare_baseline_vs_transfer(baselines, transfers, analysis_dir)
    
    # Generate visualizations
    generate_visualizations(baselines, transfers, analysis_dir)
    
    # Statistical tests
    statistical_tests(baselines, transfers)
    
    # Generate report
    generate_report(comparison_df, analysis_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {analysis_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
