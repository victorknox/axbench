#!/bin/bash
set -e

# Configuration
CONFIG="axbench/transferability/configs/pipeline_config_50.yaml"
DEVICE="cuda:0"

# Create config for 50 concepts
cat > $CONFIG << EOF
transfer:
  # Models to test
  models:
    - name: "gemma-2-2b-it-l20"
      model_path: "google/gemma-2-2b-it"
      layer: 20
    - name: "gemma-2-9b-it-l30"
      model_path: "google/gemma-2-9b-it"
      layer: 30
    - name: "meta-llama-Llama-3.1-8B-Instruct-l20"
      model_path: "meta-llama/Llama-3.1-8B-Instruct"
      layer: 20
  
  # Train/test split
  train_concepts: 40
  test_concepts: 10
  total_concepts: 50
  
  # Data configuration
  concept_path: "axbench/data/gemma-2-2b_20-gemmascope-res-16k.json"
  data_dir: "axbench/concept500/prod_2b_l20_v1/generate"
  
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
  
  # Output directories (pointing to where vectors were saved)
  output:
    vectors_dir: "axbench/transferability/results/vectors"
    transfer_models_dir: "axbench/transferability/results_50/transfer_models"
    evaluation_dir: "axbench/transferability/results_50/evaluation"
    analysis_dir: "axbench/transferability/results_50/analysis"
  
  seed: 42
EOF

echo "=========================================="
echo "Finishing 50-Concept Experiment"
echo "=========================================="

# Experiment 1: Gemma 2B -> Gemma 9B
# echo "------------------------------------------"
# echo "Experiment 1: Gemma 2B -> Gemma 9B"
# echo "------------------------------------------"

# echo "Training linear transfer..."
# python axbench/transferability/scripts/transfer.py \
#     --config $CONFIG \
#     --source_model "gemma-2-2b-it-l20" \
#     --target_model "gemma-2-9b-it-l30" \
#     --method linear \
#     --device $DEVICE

# echo "Training MLP transfer..."
# python axbench/transferability/scripts/transfer.py \
#     --config $CONFIG \
#     --source_model "gemma-2-2b-it-l20" \
#     --target_model "gemma-2-9b-it-l30" \
#     --method mlp \
#     --device $DEVICE

# echo "Evaluating..."
# python axbench/transferability/scripts/evaluate_comprehensive.py \
#     --config $CONFIG \
#     --device $DEVICE

# Experiment 2: Gemma 2B -> Llama 3.1 8B
echo ""
echo "------------------------------------------"
echo "Experiment 2: Gemma 2B -> Llama 3.1 8B"
echo "------------------------------------------"

echo "Training linear transfer..."
python axbench/transferability/scripts/transfer.py \
    --config $CONFIG \
    --source_model "gemma-2-2b-it-l20" \
    --target_model "meta-llama-Llama-3.1-8B-Instruct-l20" \
    --method linear \
    --device $DEVICE

echo "Training MLP transfer..."
python axbench/transferability/scripts/transfer.py \
    --config $CONFIG \
    --source_model "gemma-2-2b-it-l20" \
    --target_model "meta-llama-Llama-3.1-8B-Instruct-l20" \
    --method mlp \
    --device $DEVICE

echo "Evaluating..."
python axbench/transferability/scripts/evaluate_comprehensive.py \
    --config $CONFIG \
    --device $DEVICE

echo "=========================================="
echo "Analysis..."
echo "=========================================="
python axbench/transferability/scripts/analyze_comprehensive.py \
    --config $CONFIG

echo "Done!"
