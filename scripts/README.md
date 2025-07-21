# CortexGPT Scripts

This directory contains utility scripts for training and using CortexGPT.

## Directory Structure

```
scripts/
├── train_cortexgpt.py    # Main training script (simplified interface)
├── generate.py           # Text generation with trained models
├── download_data.py      # Unified data download system
├── data/                 # Data preparation scripts
│   └── create_demo_data.py
└── demos/                # Demo applications
    ├── minimal_demo.py
    ├── learning_effect_demo.py
    └── natural_language_demo.py
```

## Quick Start

### 1. Download Data
```bash
# List available datasets
uv run scripts/download_data.py --list

# Download specific dataset
uv run scripts/download_data.py --dataset english_large

# Download all English datasets
uv run scripts/download_data.py --all --category english
```

### 2. Train Model
```bash
# Quick training with demo data
uv run scripts/train_cortexgpt.py --dataset demo --epochs 10

# Train with large English dataset
uv run scripts/train_cortexgpt.py --dataset english_large --epochs 20

# Train with Korean dataset
uv run scripts/train_cortexgpt.py --dataset klue --epochs 20
```

### 3. Generate Text
```bash
# Generate text with trained model
uv run scripts/generate.py \
    --checkpoint checkpoints/model_best.pt \
    --prompt "The future of AI is" \
    --max-length 100
```

### 4. Run Demos
```bash
# Minimal generation demo
uv run scripts/demos/minimal_demo.py

# Real-time learning demo
uv run scripts/demos/learning_effect_demo.py

# Interactive chat demo
uv run scripts/demos/natural_language_demo.py
```

## Training Options

### Basic Options
- `--dataset`: Choose dataset (demo, english_large, korean_large, etc.)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate

### Model Options
- `--dim`: Model dimension (256, 512, 768)
- `--bge-stage`: BGE training stage (1=adapters only, 2=full fine-tuning)

### Advanced Options
- `--wandb`: Enable Weights & Biases logging
- `--resume`: Resume from checkpoint
- `--checkpoint-dir`: Directory to save checkpoints

## Notes

- All models use BGE-M3 embeddings by default
- Use `uv run` to execute scripts (not `python`)
- Check GPU memory with smaller batch sizes if OOM occurs