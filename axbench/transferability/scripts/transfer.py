"""
Learn transfer functions to adapt steering vectors from one model to another.

Supports:
- Identity: No transformation (direct transfer)
- Linear: Learn a linear transformation matrix
- MLP: Learn a non-linear multi-layer perceptron
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class LinearTransfer(nn.Module):
    """Linear transformation for vector transfer."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
    
    def forward(self, x):
        return self.linear(x)


class MLPTransfer(nn.Module):
    """Non-linear MLP for vector transfer."""
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256], activation='relu'):
        super().__init__()
        layers = []
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


def load_vectors(vectors_dir, model_name, num_concepts):
    """Load DiffMean vectors for a model.
    
    AxBench saves all concept vectors in a single file with shape [num_concepts, hidden_dim].
    """
    model_dir = os.path.join(vectors_dir, model_name)
    weight_path = os.path.join(model_dir, "DiffMean_weight.pt")
    
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found at {weight_path}")
        return None
    
    # Load all vectors at once
    all_vectors = torch.load(weight_path, weights_only=False)
    
    # Convert to float32 for compatibility with linear layers
    all_vectors = all_vectors.float()
    
    # Check if we have the right number of concepts
    if all_vectors.shape[0] != num_concepts:
        print(f"Warning: Expected {num_concepts} concepts but found {all_vectors.shape[0]}")
        num_concepts = min(num_concepts, all_vectors.shape[0])
    
    print(f"Loaded {num_concepts} vectors with dimension {all_vectors.shape[1]}")
    
    return all_vectors[:num_concepts]


def train_transfer_model(source_vectors, target_vectors, config, method='linear', device='cuda'):
    """Train a transfer model to map source vectors to target vectors."""
    print(f"\nTraining {method} transfer model...")
    
    input_dim = source_vectors.shape[1]
    output_dim = target_vectors.shape[1]
    
    # Create model
    if method == 'linear':
        model = LinearTransfer(input_dim, output_dim).to(device)
        lr = config['linear']['learning_rate']
        num_epochs = config['linear']['num_epochs']
        batch_size = config['linear']['batch_size']
    elif method == 'mlp':
        model = MLPTransfer(
            input_dim, output_dim,
            hidden_dims=config['mlp']['hidden_dims'],
            activation=config['mlp']['activation']
        ).to(device)
        lr = config['mlp']['learning_rate']
        num_epochs = config['mlp']['num_epochs']
        batch_size = config['mlp']['batch_size']
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    source_vectors = source_vectors.to(device)
    target_vectors = target_vectors.to(device)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Mini-batch training
        num_samples = source_vectors.shape[0]
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            source_batch = source_vectors[batch_indices]
            target_batch = target_vectors[batch_indices]
            
            optimizer.zero_grad()
            predictions = model(source_batch)
            loss = criterion(predictions, target_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (num_samples / batch_size)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    print(f"Training complete. Best loss: {best_loss:.6f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Learn transfer functions for steering vectors")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--source_model", type=str, required=True, help="Source model name")
    parser.add_argument("--target_model", type=str, required=True, help="Target model name")
    parser.add_argument("--method", type=str, default="linear", choices=['identity', 'linear', 'mlp'],
                        help="Transfer method")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    transfer_config = config['transfer']
    
    # Get number of concepts
    metadata_path = os.path.join(transfer_config['data_dir'], "metadata.jsonl")
    with open(metadata_path, 'r') as f:
        metadata = [eval(line) for line in f]
    num_concepts = min(len(metadata), transfer_config.get('max_concepts', len(metadata)))
    
    print(f"Loading vectors for {num_concepts} concepts")
    print(f"Source: {args.source_model} -> Target: {args.target_model}")
    
    # Load vectors
    num_concepts = config['transfer']['total_concepts']
    vectors_dir = transfer_config['output']['vectors_dir']
    source_vectors = load_vectors(vectors_dir, args.source_model, num_concepts)
    target_vectors = load_vectors(vectors_dir, args.target_model, num_concepts)
    
    if source_vectors is None or target_vectors is None:
        return
        
    # Select training concepts
    train_concepts = config['transfer'].get('train_concepts', num_concepts)
    print(f"Training on first {train_concepts} concepts (Total available: {num_concepts})")
    
    source_train = source_vectors[:train_concepts]
    target_train = target_vectors[:train_concepts]
    
    print(f"Source vectors shape: {source_vectors.shape}")
    print(f"Target vectors shape: {target_vectors.shape}")
    
    # Handle identity transfer (no training needed)
    if args.method == 'identity':
        print("\nIdentity transfer - no training needed")
        print("Vectors will be used directly without transformation")
        return
    
    # Train transfer model
    transfer_model = train_transfer_model(
        source_vectors, target_vectors,
        transfer_config, method=args.method,
        device=args.device
    )
    
    # Save transfer model
    output_dir = transfer_config['output']['transfer_models_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = f"{args.source_model}_to_{args.target_model}_{args.method}.pt"
    save_path = os.path.join(output_dir, model_name)
    torch.save(transfer_model.state_dict(), save_path)
    
    print(f"\nSaved transfer model to: {save_path}")
    
    # Evaluate transfer quality
    transfer_model.eval()
    with torch.no_grad():
        source_vectors = source_vectors.to(args.device)
        target_vectors = target_vectors.to(args.device)
        
        transferred = transfer_model(source_vectors)
        mse = nn.MSELoss()(transferred, target_vectors).item()
        cosine_sim = nn.CosineSimilarity(dim=1)(transferred, target_vectors).mean().item()
    
    print(f"\nTransfer Quality Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Cosine Similarity: {cosine_sim:.6f}")


if __name__ == "__main__":
    main()
