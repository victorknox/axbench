# Transferability Experiment - Ready to Run!

## ✅ Setup Complete

- **GPU**: NVIDIA A40 (46GB) - Sufficient  
- **Training Data**: `axbench/concept10/prod_2b_l20_v1/generate/` (10 concepts)
- **Config**: Updated, OpenAI evaluation disabled
- **Concepts**: Using 5 out of 10 available concepts

## Quick Start

### Run Full Experiment

```bash
cd /workspace/axbench
bash axbench/transferability/scripts/run_experiment.sh
```

This will:
1. Generate DiffMean vectors for Gemma-2-2B and Llama-3.1-8B (~15 min)
2. Train transfer functions (linear + MLP) (~10 min)
3. Evaluate all combinations (~30 min)
4. Generate analysis and statistics (~5 min)

**Total time**: ~1 hour

### Or Run Step-by-Step

```bash
cd /workspace/axbench

# 1. Generate vectors (both models, 5 concepts)
python axbench/transferability/scripts/generate_vectors.py \
  --config axbench/transferability/configs/transfer_config.yaml \
  --device cuda:0

# 2. Train linear transfer: Gemma -> Llama
python axbench/transferability/scripts/transfer.py \
  --config axbench/transferability/configs/transfer_config.yaml \
  --source_model gemma-2-2b \
  --target_model llama-3.1-8b \
  --method linear \
  --device cuda:0

# 3. Train MLP transfer: Gemma -> Llama
python axbench/transferability/scripts/transfer.py \
  --config axbench/transferability/configs/transfer_config.yaml \
  --source_model gemma-2-2b \
  --target_model llama-3.1-8b \
  --method mlp \
  --device cuda:0

# 4. Evaluate
python axbench/transferability/scripts/evaluate_transfer.py \
  --config axbench/transferability/configs/transfer_config.yaml \
  --device cuda:0

# 5. Analyze results
python axbench/transferability/scripts/analyze_results.py \
  --config axbench/transferability/configs/transfer_config.yaml
```

## What to Expect

### Outputs

Results will be saved to `axbench/transferability/results/`:
- `vectors/` - DiffMean vectors for each model
- `transfer_models/` - Trained transfer functions
- `evaluation/` - Steering performance CSVs and metrics
- `analysis/` - Comparison statistics and plots

### Key Comparisons

The experiment will compare:
- **Baseline**: Gemma vectors on Gemma, Llama vectors on Llama
- **Identity Transfer**: Gemma vectors on Llama (no adaptation)
- **Linear Transfer**: Gemma vectors → linear transform → Llama
- **MLP Transfer**: Gemma vectors → MLP → Llama

You'll get statistics showing which transfer method works best!
