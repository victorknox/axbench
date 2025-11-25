#!/bin/bash

# Run cross-model transfer experiments
# Usage: bash run_cross_model_experiments.sh <hf_token>

HF_TOKEN=$1

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HuggingFace token required"
    echo "Usage: bash run_cross_model_experiments.sh <hf_token>"
    exit 1
fi

export HF_TOKEN=$HF_TOKEN
# Also set for huggingface_hub
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

echo "=========================================="
echo "Starting Cross-Model Experiments"
echo "=========================================="

# Experiment 1: Gemma 2B -> Gemma 9B
echo ""
echo "------------------------------------------"
echo "Experiment 1: Gemma 2B -> Gemma 9B"
echo "------------------------------------------"
bash axbench/transferability/scripts/run_full_pipeline.sh \
    "google/gemma-2-2b-it" \
    "google/gemma-2-9b-it" \
    20 \
    30 \
    50 \
    40 \
    10 \
    "axbench/concept500/prod_2b_l20_v1/generate"

# Experiment 2: Gemma 2B -> Llama 3.1 8B
echo ""
echo "------------------------------------------"
echo "Experiment 2: Gemma 2B -> Llama 3.1 8B"
echo "------------------------------------------"
bash axbench/transferability/scripts/run_full_pipeline.sh \
    "google/gemma-2-2b-it" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    20 \
    20 \
    50 \
    40 \
    10 \
    "axbench/concept500/prod_2b_l20_v1/generate"

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
