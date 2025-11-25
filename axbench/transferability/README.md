# Steering Vector Transferability Experiments

## Overview

This directory contains experiments to test whether DiffMean steering vectors learned on one model can transfer to another model, with optional linear or non-linear adaptation.

## Experiment Design

### Research Question
Can steering vectors learned on Model A effectively steer Model B's behavior?

### Approach

1. **Train DiffMean vectors** on two different models (e.g., Gemma-2-2B and Llama-3.1-8B)
2. **Transfer methods**:
   - **Identity**: Use Model A's vectors directly on Model B
   - **Linear**: Learn a linear transformation matrix to adapt vectors
   - **Non-linear**: Learn an MLP to adapt vectors
3. **Evaluate**: Compare steering performance of:
   - Baseline: Model-specific vectors (trained on same model)
   - Transfer: Cross-model vectors (trained on different model)

### Metrics
- Steering effectiveness (LM judge scores)
- Perplexity
- Rule following accuracy

## Directory Structure

```
transferability/
├── configs/          # YAML configs for experiments
├── scripts/          # Python scripts for the pipeline
│   ├── generate_vectors.py    # Train DiffMean on multiple models
│   ├── transfer.py             # Learn transfer functions
│   ├── evaluate.py             # Evaluate transferred vectors
│   └── run_experiment.py       # Main experiment runner
└── results/          # Experiment outputs
```

## Usage

### 1. Generate DiffMean vectors for multiple models

```bash
python axbench/transferability/scripts/generate_vectors.py \
  --config axbench/transferability/configs/transfer_config.yaml
```

### 2. Learn transfer functions (optional)

```bash
python axbench/transferability/scripts/transfer.py \
  --source_model gemma-2-2b \
  --target_model llama-3.1-8b \
  --method linear  # or 'mlp'
```

### 3. Evaluate transferability

```bash
python axbench/transferability/scripts/evaluate.py \
  --config axbench/transferability/configs/transfer_config.yaml
```

### 4. Run full experiment

```bash
bash axbench/transferability/scripts/run_experiment.sh
```

## Expected Outputs

- `results/vectors/` - DiffMean vectors for each model
- `results/transfer_models/` - Learned transfer functions
- `results/evaluation/` - Steering performance metrics
- `results/analysis/` - Statistical comparisons and visualizations
