#!/bin/bash

# Test script - runs just the first step to verify everything works

set -e

CONFIG="axbench/transferability/configs/transfer_config.yaml"
DEVICE="cuda:0"

echo "=========================================="
echo "Testing Transferability Setup"
echo "=========================================="

echo ""
echo "Checking training data..."
ls -lh axbench/concept10/prod_2b_l20_v1/generate/

echo ""
echo "Testing vector generation for Gemma-2-2B (first concept only)..."
python -c "
import sys
sys.path.insert(0, 'axbench')
import yaml
import pandas as pd

# Load config
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Check data
data_path = config['transfer']['data_dir'] + '/train_data.parquet'
df = pd.read_parquet(data_path)
print(f'Training data shape: {df.shape}')
print(f'Concepts: {df[\"concept_id\"].nunique()}')
print(f'Columns: {list(df.columns)}')
print('\\nFirst few rows:')
print(df.head())
"

echo ""
echo "=========================================="
echo "Setup looks good! Ready to run experiment."
echo "=========================================="
