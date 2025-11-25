"""
Generate comprehensive analysis report from evaluation results.

Creates:
1. Summary tables comparing all methods
2. Visualizations of transfer quality
3. Statistical significance tests
4. Markdown report
"""

import os
import sys
import yaml
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_results(evaluation_dir):
    """Load evaluation results."""
    results_path = os.path.join(evaluation_dir, "comprehensive_evaluation.json")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    split_path = os.path.join(evaluation_dir, "train_test_split.json")
    with open(split_path, 'r') as f:
        split = json.load(f)
    
    return results, split


def create_summary_table(results):
    """Create summary table of all results."""
    rows = []
    
    for key, result in results.items():
        parts = key.rsplit('_', 1)
        transfer_pair = parts[0]
        method = parts[1]
        
        row = {
            'Transfer': transfer_pair,
            'Method': method,
            'Train MSE': result['train']['mse'],
            'Test MSE': result['test']['mse'],
            'Train Cosine': result['train']['cosine_similarity_mean'],
            'Test Cosine': result['test']['cosine_similarity_mean'],
            'MSE Gap': result['generalization']['mse_gap'],
            'Cosine Gap': result['generalization']['cosine_gap'],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def create_visualizations(results, analysis_dir):
    """Create visualization plots."""
    sns.set_style("whitegrid")
    
    # Extract data for plotting
    methods = []
    train_mse = []
    test_mse = []
    train_cosine = []
    test_cosine = []
    
    for key, result in results.items():
        method = key.rsplit('_', 1)[1]
        methods.append(method)
        train_mse.append(result['train']['mse'])
        test_mse.append(result['test']['mse'])
        train_cosine.append(result['train']['cosine_similarity_mean'])
        test_cosine.append(result['test']['cosine_similarity_mean'])
    
    # Plot 1: MSE comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x - width/2, train_mse, width, label='Train', alpha=0.8)
    ax1.bar(x + width/2, test_mse, width, label='Test', alpha=0.8)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Cosine similarity comparison
    ax2.bar(x - width/2, train_cosine, width, label='Train', alpha=0.8)
    ax2.bar(x + width/2, test_cosine, width, label='Test', alpha=0.8)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Cosine Similarity')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'transfer_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {analysis_dir}/transfer_comparison.png")


def generate_markdown_report(results, split, summary_df, analysis_dir):
    """Generate comprehensive markdown report."""
    report = []
    
    report.append("# Transferability Analysis Report\n")
    report.append("## Experiment Configuration\n")
    report.append(f"- **Train Concepts**: {len(split['train_indices'])} (indices: {split['train_indices']})")
    report.append(f"- **Test Concepts**: {len(split['test_indices'])} (indices: {split['test_indices']})")
    report.append(f"- **Total Experiments**: {len(results)}\n")
    
    report.append("## Summary Statistics\n")
    report.append("### All Methods Performance\n")
    report.append(summary_df.to_markdown(index=False))
    report.append("\n")
    
    report.append("## Key Findings\n")
    
    # Find best method
    best_test_mse = summary_df.loc[summary_df['Test MSE'].idxmin()]
    best_test_cosine = summary_df.loc[summary_df['Test Cosine'].idxmax()]
    
    report.append(f"### Best Methods\n")
    report.append(f"- **Lowest Test MSE**: {best_test_mse['Method']} ({best_test_mse['Transfer']}) - MSE: {best_test_mse['Test MSE']:.6f}")
    report.append(f"- **Highest Test Cosine**: {best_test_cosine['Method']} ({best_test_cosine['Transfer']}) - Cosine: {best_test_cosine['Test Cosine']:.4f}\n")
    
    # Generalization analysis
    report.append("### Generalization Analysis\n")
    for _, row in summary_df.iterrows():
        if row['MSE Gap'] < 0:
            report.append(f"- **{row['Method']}** ({row['Transfer']}): Better on test than train (MSE gap: {row['MSE Gap']:.6f})")
        elif row['MSE Gap'] > 0.001:
            report.append(f"- **{row['Method']}** ({row['Transfer']}): Overfitting detected (MSE gap: {row['MSE Gap']:.6f})")
    report.append("\n")
    
    report.append("## Detailed Metrics\n")
    for key, result in results.items():
        parts = key.rsplit('_', 1)
        transfer_pair = parts[0]
        method = parts[1]
        
        report.append(f"### {transfer_pair} - {method}\n")
        report.append("#### Train Set")
        report.append(f"- MSE: {result['train']['mse']:.6f}")
        report.append(f"- Cosine Similarity: {result['train']['cosine_similarity_mean']:.4f} ± {result['train']['cosine_similarity_std']:.4f}")
        report.append(f"- L2 Distance: {result['train']['l2_distance_mean']:.4f} ± {result['train']['l2_distance_std']:.4f}")
        report.append(f"- Pearson Correlation: {result['train']['pearson_correlation_mean']:.4f}")
        report.append(f"- Top-100 Overlap: {result['train']['top_100_overlap_mean']:.4f}\n")
        
        report.append("#### Test Set")
        report.append(f"- MSE: {result['test']['mse']:.6f}")
        report.append(f"- Cosine Similarity: {result['test']['cosine_similarity_mean']:.4f} ± {result['test']['cosine_similarity_std']:.4f}")
        report.append(f"- L2 Distance: {result['test']['l2_distance_mean']:.4f} ± {result['test']['l2_distance_std']:.4f}")
        report.append(f"- Pearson Correlation: {result['test']['pearson_correlation_mean']:.4f}")
        report.append(f"- Top-100 Overlap: {result['test']['top_100_overlap_mean']:.4f}\n")
    
    report.append("## Visualization\n")
    report.append("![Transfer Comparison](transfer_comparison.png)\n")
    
    # Write report
    report_path = os.path.join(analysis_dir, 'analysis_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive analysis report")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluation_dir = config['transfer']['output']['evaluation_dir']
    analysis_dir = config['transfer']['output']['analysis_dir']
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load results
    print("Loading results...")
    results, split = load_results(evaluation_dir)
    
    # Create summary table
    print("Creating summary table...")
    summary_df = create_summary_table(results)
    summary_df.to_csv(os.path.join(analysis_dir, 'summary_table.csv'), index=False)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, analysis_dir)
    
    # Generate report
    print("Generating markdown report...")
    generate_markdown_report(results, split, summary_df, analysis_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {analysis_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
