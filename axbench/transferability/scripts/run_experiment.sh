#!/bin/bash

# Main experiment runner for steering vector transferability
# This script runs the complete pipeline:
# 1. Generate DiffMean vectors for multiple models
# 2. Learn transfer functions (linear and MLP)
# 3. Evaluate transferability
# 4. Generate analysis and statistics

set -e  # Exit on error

CONFIG="axbench/transferability/configs/transfer_config.yaml"
DEVICE="cuda:0"

echo "=========================================="
echo "Steering Vector Transferability Experiment"
echo "=========================================="

# Step 1: Generate DiffMean vectors
echo ""
echo "Step 1: Generating DiffMean vectors for all models..."
python axbench/transferability/scripts/generate_vectors.py \
    --config $CONFIG \
    --device $DEVICE

# Step 2: Learn transfer functions
echo ""
echo "Step 2: Learning transfer functions..."

# Get model names from config (simplified - assumes gemma and llama)
SOURCE_MODEL="gemma-2-2b"
TARGET_MODEL="llama-3.1-8b"

# Linear transfer: gemma -> llama
echo "  Training linear transfer: $SOURCE_MODEL -> $TARGET_MODEL"
python axbench/transferability/scripts/transfer.py \
    --config $CONFIG \
    --source_model $SOURCE_MODEL \
    --target_model $TARGET_MODEL \
    --method linear \
    --device $DEVICE

# MLP transfer: gemma -> llama
echo "  Training MLP transfer: $SOURCE_MODEL -> $TARGET_MODEL"
python axbench/transferability/scripts/transfer.py \
    --config $CONFIG \
    --source_model $SOURCE_MODEL \
    --target_model $TARGET_MODEL \
    --method mlp \
    --device $DEVICE

# Linear transfer: llama -> gemma
echo "  Training linear transfer: $TARGET_MODEL -> $SOURCE_MODEL"
python axbench/transferability/scripts/transfer.py \
    --config $CONFIG \
    --source_model $TARGET_MODEL \
    --target_model $SOURCE_MODEL \
    --method linear \
    --device $DEVICE

# MLP transfer: llama -> gemma
echo "  Training MLP transfer: $TARGET_MODEL -> $SOURCE_MODEL"
python axbench/transferability/scripts/transfer.py \
    --config $CONFIG \
    --source_model $TARGET_MODEL \
    --target_model $SOURCE_MODEL \
    --method mlp \
    --device $DEVICE

# Step 3: Evaluate transferability
echo ""
echo "Step 3: Evaluating transferability..."
python axbench/transferability/scripts/evaluate_transfer.py \
    --config $CONFIG \
    --device $DEVICE

# Step 4: Generate analysis
echo ""
echo "Step 4: Generating analysis and statistics..."
python axbench/transferability/scripts/analyze_results.py \
    --config $CONFIG

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "Results saved to: axbench/transferability/results/"
echo "=========================================="
