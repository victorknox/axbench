"""
Generate DiffMean steering vectors for multiple models.

This script trains DiffMean vectors on different models using the same concepts,
so we can test transferability between models.
"""

import os
import sys
import yaml
import torch
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from axbench.models.mean import DiffMean
from axbench.utils.constants import *


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_training_data(data_dir, concept_id):
    """Load training data for a specific concept, including shared negatives."""
    train_data_path = os.path.join(data_dir, "train_data.parquet")
    df = pd.read_parquet(train_data_path)
    
    # Get positive examples for this concept
    positive_df = df[df['concept_id'] == concept_id]
    
    # Get shared negative examples (concept_id == -1)
    negative_df = df[df['concept_id'] == -1]
    
    # Combine them
    concept_df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    return concept_df





def main():
    parser = argparse.ArgumentParser(description="Generate DiffMean vectors for multiple models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--start_concept", type=int, default=0, help="Start concept index")
    parser.add_argument("--end_concept", type=int, default=None, help="End concept index")
    parser.add_argument("--model_filter", type=str, default=None, help="Filter to specific model name")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    transfer_config = config['transfer']
    
    # Get metadata to know how many concepts
    metadata_path = os.path.join(transfer_config['data_dir'], "metadata.jsonl")
    with open(metadata_path, 'r') as f:
        metadata = [eval(line) for line in f]
    
    max_concepts = min(len(metadata), transfer_config.get('max_concepts', len(metadata)))
    start_idx = args.start_concept
    end_idx = args.end_concept if args.end_concept is not None else max_concepts
    end_idx = min(end_idx, max_concepts)
    
    print(f"Generating DiffMean vectors for concepts {start_idx} to {end_idx}")
    
    # Filter models if requested
    models_to_process = transfer_config['models']
    if args.model_filter:
        models_to_process = [m for m in models_to_process if args.model_filter in m['name']]
        print(f"Filtering for models matching: {args.model_filter}")
    
    print(f"Models: {[m['name'] for m in models_to_process]}")
    
    # Iterate over models first to avoid reloading
    for model_config in models_to_process:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_config['name']}")
        print(f"{'='*60}")
        
        model_name = model_config['model_path']
        layer = model_config['layer']
        
        # Load model and tokenizer ONCE
        print(f"Loading model: {model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=args.device
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.padding_side = "right"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
            
        # Create training args
        from dataclasses import dataclass
        @dataclass
        class TrainingArgs:
            batch_size: int = 6
            n_epochs: int = 1
            lr: float = 0.001
            weight_decay: float = 0.0
        
        training_args = TrainingArgs()
        
        # Initialize DiffMean wrapper (we'll re-init the inner part for each concept)
        # Actually, DiffMean takes model in init.
        
        # Iterate over concepts
        for concept_id in tqdm(range(start_idx, end_idx), desc=f"Concepts for {model_config['name']}"):
            concept_name = metadata[concept_id]['concept']
            # print(f"Processing concept {concept_id}: {concept_name}")
            
            # Check if vectors already exist? 
            # For now, just overwrite or append.
            
            # Load training data
            train_df = load_training_data(transfer_config['data_dir'], concept_id)
            train_df['labels'] = (train_df['category'] == 'positive').astype(int)
            
            # Initialize DiffMean for this concept
            diffmean = DiffMean(
                model=model,
                tokenizer=tokenizer,
                layer=layer,
                component="res",
                device=args.device,
                training_args=training_args
            )
            
            diffmean.make_model(mode="train", low_rank_dimension=1)
            diffmean.train(train_df, prefix_length=1)
            
            # Save vectors
            model_output_dir = Path(transfer_config['output']['vectors_dir']) / model_config['name']
            model_output_dir.mkdir(parents=True, exist_ok=True)
            diffmean.save(model_output_dir, concept_id=concept_id)
            
        # Clean up model after all concepts are done
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("Vector generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
