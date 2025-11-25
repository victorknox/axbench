#!/bin/bash

# Quick setup script for transferability experiment
# Generates minimal training data needed for the experiment

set -e

echo "=========================================="
echo "Setting up Transferability Experiment"
echo "=========================================="

CONFIG="axbench/demo/sweep/simple.yaml"
DUMP_DIR="axbench/demo"

# Step 1: Generate training data (requires OpenAI API key)
echo ""
echo "Step 1: Generating training data for concepts..."
echo "This will use OpenAI API to generate training examples."
echo ""

# Check if OpenAI key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY is not set!"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

python axbench/scripts/generate.py \
    --config $CONFIG \
    --dump_dir $DUMP_DIR \
    --mode training

echo ""
echo "=========================================="
echo "Setup complete! Training data generated."
echo "You can now run the transferability experiment:"
echo "  bash axbench/transferability/scripts/run_experiment.sh"
echo "=========================================="
