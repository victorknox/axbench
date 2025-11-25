#!/bin/bash

# Complete Transferability Pipeline
# Usage: bash run_full_pipeline.sh <source_model> <target_model> <source_layer> <target_layer>
#
# Example:
#   bash run_full_pipeline.sh gemma-2-2b-it gemma-2-2b-it 20 10

set -e

# Parse arguments
SOURCE_MODEL_PATH=${1:-"google/gemma-2-2b-it"}
TARGET_MODEL_PATH=${2:-"google/gemma-2-2b-it"}
SOURCE_LAYER=${3:-20}
TARGET_LAYER=${4:-10}
TOTAL_CONCEPTS=${5:-10}
TRAIN_CONCEPTS=${6:-8}
TEST_CONCEPTS=${7:-2}
DATA_DIR=${8:-"axbench/concept10/prod_2b_l20_v1/generate"}

# Derive model names from paths
SOURCE_MODEL_NAME=$(echo $SOURCE_MODEL_PATH | sed 's/\//-/g' | sed 's/google-//g')-l${SOURCE_LAYER}
TARGET_MODEL_NAME=$(echo $TARGET_MODEL_PATH | sed 's/\//-/g' | sed 's/google-//g')-l${TARGET_LAYER}

CONFIG="axbench/transferability/configs/pipeline_config.yaml"
DEVICE="cuda:0"

echo "=========================================="
echo "Complete Transferability Pipeline"
echo "=========================================="
echo "Source: $SOURCE_MODEL_NAME ($SOURCE_MODEL_PATH, layer $SOURCE_LAYER)"
echo "Target: $TARGET_MODEL_NAME ($TARGET_MODEL_PATH, layer $TARGET_LAYER)"
echo "Concepts: $TOTAL_CONCEPTS (Train: $TRAIN_CONCEPTS, Test: $TEST_CONCEPTS)"
echo "Data: $DATA_DIR"
echo ""

# Create temporary config
cat > $CONFIG << EOF
transfer:
  # Models to test
  models:
    - name: "$SOURCE_MODEL_NAME"
      model_path: "$SOURCE_MODEL_PATH"
      layer: $SOURCE_LAYER
    - name: "$TARGET_MODEL_NAME"
      model_path: "$TARGET_MODEL_PATH"
      layer: $TARGET_LAYER
  
  # Train/test split
  train_concepts: $TRAIN_CONCEPTS
  test_concepts: $TEST_CONCEPTS
  total_concepts: $TOTAL_CONCEPTS
  
  # Data configuration
  concept_path: "axbench/data/gemma-2-2b_20-gemmascope-res-16k.json"
  data_dir: "$DATA_DIR"
  
  # Transfer methods
  transfer_methods:
    - "identity"
    - "linear"
    - "mlp"
  
  # Linear transfer settings
  linear:
    learning_rate: 0.001
    num_epochs: 100
    batch_size: 32
  
  # MLP transfer settings
  mlp:
    hidden_dims: [512, 256]
    learning_rate: 0.001
    num_epochs: 100
    batch_size: 32
    activation: "relu"
  
  # Output directories
  output:
    vectors_dir: "axbench/transferability/results/vectors"
    transfer_models_dir: "axbench/transferability/results/transfer_models"
    evaluation_dir: "axbench/transferability/results/evaluation"
    analysis_dir: "axbench/transferability/results/analysis"
  
  seed: 42
EOF

echo "Step 1: Generating DiffMean vectors..."
python axbench/transferability/scripts/generate_vectors.py \
    --config $CONFIG \
    --device $DEVICE

echo ""
echo "Step 2: Training transfer functions..."

# Linear transfer: source -> target
echo "  Training linear transfer: $SOURCE_MODEL_NAME -> $TARGET_MODEL_NAME"
python axbench/transferability/scripts/transfer.py \
    --config $CONFIG \
    --source_model $SOURCE_MODEL_NAME \
    --target_model $TARGET_MODEL_NAME \
    --method linear \
    --device $DEVICE

# MLP transfer: source -> target
echo "  Training MLP transfer: $SOURCE_MODEL_NAME -> $TARGET_MODEL_NAME"
python axbench/transferability/scripts/transfer.py \
    --config $CONFIG \
    --source_model $SOURCE_MODEL_NAME \
    --target_model $TARGET_MODEL_NAME \
    --method mlp \
    --device $DEVICE

# If different models, train reverse direction too
if [ "$SOURCE_MODEL_NAME" != "$TARGET_MODEL_NAME" ]; then
    echo "  Training linear transfer: $TARGET_MODEL_NAME -> $SOURCE_MODEL_NAME"
    python axbench/transferability/scripts/transfer.py \
        --config $CONFIG \
        --source_model $TARGET_MODEL_NAME \
        --target_model $SOURCE_MODEL_NAME \
        --method linear \
        --device $DEVICE
    
    echo "  Training MLP transfer: $TARGET_MODEL_NAME -> $SOURCE_MODEL_NAME"
    python axbench/transferability/scripts/transfer.py \
        --config $CONFIG \
        --source_model $TARGET_MODEL_NAME \
        --target_model $SOURCE_MODEL_NAME \
        --method mlp \
        --device $DEVICE
fi

echo ""
echo "Step 3: Evaluating transferability..."
python axbench/transferability/scripts/evaluate_comprehensive.py \
    --config $CONFIG \
    --device $DEVICE

echo ""
echo "Step 4: Generating analysis report..."
python axbench/transferability/scripts/analyze_comprehensive.py \
    --config $CONFIG

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Results saved to: axbench/transferability/results/"
echo "=========================================="
