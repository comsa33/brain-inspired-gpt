# English-Only Training Guide

This guide shows how to train BrainGPT models using only English data.

## Quick Start

### 1. Prepare English-Only Data

```bash
# Option A: Wikipedia English (small, quick)
uv run data/openwebtext/prepare_simple.py

# Option B: C4 Dataset (high quality, larger)
uv run data/openwebtext/prepare_c4.py --max-samples 100000
```

### 2. Train with V2 Model (Recommended)

```bash
# Basic training
uv run brain_gpt/training/train_brain_gpt_v2.py --no-wandb

# With custom settings
uv run brain_gpt/training/train_brain_gpt_v2.py \
  --batch-size 8 \
  --learning-rate 6e-4 \
  --max-steps 10000 \
  --no-wandb
```

### 3. Train with Original Model

```bash
# RTX 3090 optimized version
uv run brain_gpt/training/train_brain_gpt_3090.py --no-wandb

# Simple training script
uv run brain_gpt/training/train_simple.py
```

## Using Specific English Datasets

### Wikipedia English Only

The `prepare_simple.py` script downloads both English and Korean Wikipedia by default. The data is mixed in the training files.

### C4 Dataset (English Only)

C4 is a cleaned Common Crawl dataset containing only English text:

```bash
# Prepare C4 data
uv run data/openwebtext/prepare_c4.py --max-samples 100000

# Train with C4 data
uv run brain_gpt/training/train_multilingual.py \
  --data-dirs data/c4 \
  --language-sampling none \
  --no-wandb
```

### Custom English Dataset

If you have your own English text files:

```python
# Create a simple preparation script
import numpy as np
from brain_gpt.core.multilingual_tokenizer import MultilingualBrainTokenizer

# Load your English texts
texts = ["Your English text here...", "Another text..."]

# Tokenize
tokenizer = MultilingualBrainTokenizer()
tokens = []
for text in texts:
    tokens.extend(tokenizer.encode(text, language='en'))

# Save as binary file
tokens_array = np.array(tokens, dtype=np.uint16)
tokens_array.tofile('data/custom/english_train.bin')
```

## Training Tips

### For Best Performance

1. **Use V2 Model**: It's 3-5x faster and more stable
   ```bash
   uv run brain_gpt/training/train_brain_gpt_v2.py --compile
   ```

2. **Larger Batch Size**: V2 can handle larger batches
   ```bash
   --batch-size 16 --gradient-accumulation-steps 2
   ```

3. **Enable PyTorch 2.0 Compile**:
   ```bash
   --compile  # Adds 10-20% speed improvement
   ```

### Memory Settings

- **RTX 3090 (24GB)**:
  - V2: batch_size=8-16
  - V1: batch_size=2-4

- **RTX 3080 (10GB)**:
  - Use smaller model config
  - Enable gradient checkpointing

### Monitoring Training

Without wandb:
```bash
# Training prints progress every 10 steps
Step 100: loss=4.5632, lr=6.00e-04, tokens/s=15420
```

With wandb:
```bash
# Remove --no-wandb flag
uv run brain_gpt/training/train_brain_gpt_v2.py
```

## Example Commands

### Fastest English Training
```bash
# Quick test (small model, few steps)
uv run brain_gpt/training/train_brain_gpt_v2.py \
  --batch-size 8 \
  --max-steps 1000 \
  --no-wandb
```

### High-Quality English Model
```bash
# Prepare C4 data (once)
uv run data/openwebtext/prepare_c4.py --max-samples 500000

# Train longer
uv run brain_gpt/training/train_brain_gpt_v2.py \
  --data-dir data/c4 \
  --batch-size 8 \
  --max-steps 50000 \
  --learning-rate 3e-4 \
  --compile
```

### Testing Your Model
```bash
# Run benchmark
uv run benchmark_v1_vs_v2.py

# Test features
uv run test_v2_features.py
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size
- Enable gradient_checkpointing
- Use smaller model (modify n_layer, n_embd in training script)

### Slow Training
- Use V2 model instead of V1
- Enable --compile flag
- Check GPU utilization with `nvidia-smi`

### Poor Generation Quality
- Train for more steps (at least 10,000)
- Use larger/better dataset (C4 recommended)
- Adjust learning rate (try 3e-4 to 6e-4)