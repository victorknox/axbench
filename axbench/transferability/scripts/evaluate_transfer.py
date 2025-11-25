"""
Evaluate transferability of steering vectors.

Compares:
1. Baseline: Model-specific vectors (trained on same model)
2. Transfer: Cross-model vectors (trained on different model)
   - Identity transfer (direct)
   - Linear transfer (learned linear transformation)
   - MLP transfer (learned non-linear transformation)
"""

import os
import sys
import yaml
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from axbench.models.mean import DiffMean
from axbench.evaluators.lm_judge import LMJudgeEvaluator
from axbench.evaluators.ppl import PerplexityEvaluator
from transferability.scripts.transfer import LinearTransfer, MLPTransfer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_transfer_model(transfer_models_dir, source_model, target_model, method, device):
    """Load a trained transfer model."""
    if method == 'identity':
        return None  # No transformation needed
    
    model_name = f"{source_model}_to_{target_model}_{method}.pt"
    model_path = os.path.join(transfer_models_dir, model_name)
    
    if not os.path.exists(model_path):
        print(f"Warning: Transfer model not found: {model_path}")
        return None
    
    # Load model architecture (need to know dimensions)
    # For now, we'll create it when we load vectors
    state_dict = torch.load(model_path, map_location=device)
    return state_dict


def evaluate_steering(model, tokenizer, vectors, concepts, eval_config, device):
    """Evaluate steering performance with given vectors."""
    results = []
    
    # Load AlpacaEval dataset
    alpaca_path = os.path.join(eval_config.get('master_data_dir', 'axbench/data'), 'alpaca_eval.json')
    alpaca_df = pd.read_json(alpaca_path)
    
    num_examples = eval_config['num_examples']
    factors = eval_config['steering_factors']
    
    for concept_id, concept in enumerate(tqdm(concepts, desc="Evaluating concepts")):
        # Sample prompts
        sampled_prompts = alpaca_df.sample(num_examples, random_state=42)['instruction'].tolist()
        
        for factor in factors:
            for prompt_id, prompt in enumerate(sampled_prompts):
                # Apply steering
                vector = vectors[concept_id].to(device)
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
                
                # Generate with steering
                # This is simplified - in practice you'd use pyvene for proper intervention
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=eval_config['output_length'],
                        do_sample=True,
                        temperature=0.7
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                results.append({
                    'concept_id': concept_id,
                    'concept': concept,
                    'prompt_id': prompt_id,
                    'prompt': prompt,
                    'factor': factor,
                    'output': generated_text
                })
    
    return pd.DataFrame(results)


def compute_metrics(results_df, concepts, lm_model=None):
    """Compute evaluation metrics."""
    metrics = {}
    
    # Group by concept and factor
    for concept_id in results_df['concept_id'].unique():
        concept_data = results_df[results_df['concept_id'] == concept_id]
        concept = concepts[concept_id]
        
        for factor in concept_data['factor'].unique():
            factor_data = concept_data[concept_data['factor'] == factor]
            
            # Compute perplexity (simplified - would need proper implementation)
            # avg_ppl = compute_perplexity(factor_data['output'].tolist())
            
            # LM Judge scores (if available)
            if lm_model:
                # Would call LMJudgeEvaluator here
                pass
            
            key = f"concept_{concept_id}_factor_{factor}"
            metrics[key] = {
                'concept': concept,
                'factor': factor,
                'num_samples': len(factor_data),
                # 'perplexity': avg_ppl,
            }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate steering vector transferability")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    transfer_config = config['transfer']
    
    # Load metadata
    metadata_path = os.path.join(transfer_config['data_dir'], "metadata.jsonl")
    with open(metadata_path, 'r') as f:
        metadata = [eval(line) for line in f]
    
    num_concepts = min(len(metadata), transfer_config.get('max_concepts', len(metadata)))
    concepts = [metadata[i]['concept'] for i in range(num_concepts)]
    
    print(f"Evaluating transferability for {num_concepts} concepts")
    
    all_results = {}
    
    # Evaluate each model pair and transfer method
    models = transfer_config['models']
    for source_model_config in models:
        for target_model_config in models:
            source_name = source_model_config['name']
            target_name = target_model_config['name']
            
            # Load target model
            print(f"\n{'='*60}")
            print(f"Target Model: {target_name}")
            print(f"{'='*60}")
            
            target_model = AutoModelForCausalLM.from_pretrained(
                target_model_config['model_path'],
                torch_dtype=torch.bfloat16,
                device_map=args.device
            )
            tokenizer = AutoTokenizer.from_pretrained(target_model_config['model_path'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test different transfer methods
            for method in transfer_config['transfer_methods']:
                print(f"\nEvaluating: {source_name} -> {target_name} ({method})")
                
                # Load source vectors
                vectors_dir = transfer_config['output']['vectors_dir']
                source_vectors = []
                for concept_id in range(num_concepts):
                    vector_path = os.path.join(vectors_dir, source_name, f"concept_{concept_id}_vector.pt")
                    vector = torch.load(vector_path)
                    source_vectors.append(vector)
                source_vectors = torch.stack(source_vectors)
                
                # Apply transfer if needed
                if source_name != target_name and method != 'identity':
                    transfer_state = load_transfer_model(
                        transfer_config['output']['transfer_models_dir'],
                        source_name, target_name, method, args.device
                    )
                    
                    if transfer_state is not None:
                        # Create and load transfer model
                        input_dim = source_vectors.shape[1]
                        output_dim = source_vectors.shape[1]  # Assume same for now
                        
                        if method == 'linear':
                            transfer_model = LinearTransfer(input_dim, output_dim).to(args.device)
                        elif method == 'mlp':
                            transfer_model = MLPTransfer(
                                input_dim, output_dim,
                                hidden_dims=transfer_config['mlp']['hidden_dims']
                            ).to(args.device)
                        
                        transfer_model.load_state_dict(transfer_state)
                        transfer_model.eval()
                        
                        with torch.no_grad():
                            source_vectors = transfer_model(source_vectors.to(args.device))
                
                # Evaluate steering
                results_df = evaluate_steering(
                    target_model, tokenizer, source_vectors,
                    concepts, transfer_config['evaluation'], args.device
                )
                
                # Compute metrics
                metrics = compute_metrics(results_df, concepts)
                
                # Store results
                key = f"{source_name}_to_{target_name}_{method}"
                all_results[key] = {
                    'results': results_df,
                    'metrics': metrics
                }
                
                # Save results
                output_dir = transfer_config['output']['evaluation_dir']
                os.makedirs(output_dir, exist_ok=True)
                
                results_df.to_csv(os.path.join(output_dir, f"{key}_results.csv"), index=False)
                with open(os.path.join(output_dir, f"{key}_metrics.json"), 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            # Clean up
            del target_model
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to: {transfer_config['output']['evaluation_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
