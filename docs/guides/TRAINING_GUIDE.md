# CortexGPT Training Guide

## 🚀 Quick Start (Recommended)

For RTX 3090 users who want to start training immediately:

```bash
# Fast mode - Best for experiments
uv run scripts/train.py --mode fast --epochs 10 --wandb

# This will:
# - Auto-detect your GPU and optimize settings
# - Use learning rate 1e-4 (not the old 5e-5)
# - Load data with 8 workers for speed
# - Train ~20x faster than before!
```

## 📊 Training Modes

| Mode | Memory | Speed | Features | When to Use |
|------|--------|-------|----------|-------------|
| `fast` | 8-10GB | ~1s/iter | Minimal features | Quick experiments, testing |
| `standard` | 12-15GB | ~2s/iter | Phase 1+2 features | Balanced training |
| `full` | 20GB+ | ~5s/iter | All features | Research, production |

## 🎯 Common Commands

### Basic Training
```bash
# Fast training with demo data
uv run scripts/train.py --mode fast --epochs 5

# Standard training with custom data
uv run scripts/train.py \
    --train-data data/your_train.bin \
    --val-data data/your_val.bin \
    --mode standard \
    --epochs 20 \
    --wandb
```

### Resume Training
```bash
# Continue from checkpoint
uv run scripts/train.py \
    --resume checkpoints/cortex/cortex_gpt_best.pt \
    --mode fast \
    --epochs 10
```

### Custom Settings
```bash
# Override auto-detected settings
uv run scripts/train.py \
    --mode fast \
    --batch-size 16 \
    --lr 2e-4 \
    --dim 768 \
    --epochs 10
```

## 🔧 Performance Optimization

### Key Improvements in v2.1
1. **Fixed Learning Rate**: Now uses 1e-4 (was 5e-5) for faster convergence
2. **Optimized Data Loading**: 8 workers by default (was 2)
3. **Smart GPU Detection**: Auto-configures for your hardware
4. **Reduced Warmup**: 5% instead of 10% for faster learning

### Expected Performance

**Before (Old Scripts)**:
- Speed: ~40 seconds/iteration
- Loss decrease: 0.00006/iteration
- Time to convergence: 60+ days

**After (New Script)**:
- Speed: ~1-2 seconds/iteration
- Loss decrease: 0.005-0.01/iteration
- Time to convergence: 1-2 days

## 📈 Monitoring Training

### Real-time Monitoring
```bash
# GPU usage (should be >90%)
watch -n 1 nvidia-smi

# Training logs (if using wandb)
# Check https://wandb.ai/your-username/cortex-gpt

# Loss progression
tail -f wandb/latest-run/logs/debug.log | grep loss
```

### What to Look For
✅ **Good Signs**:
- GPU utilization >90%
- Loss decreasing 0.005-0.01 per iteration
- Smooth loss curve without spikes
- Training speed ~1-2s/iteration

❌ **Problems**:
- GPU utilization <50% → Increase batch size or workers
- Loss barely changing → Check learning rate
- OOM errors → Reduce batch size or use smaller mode
- Very slow → Check num_workers setting

## 🧠 Advanced Features

### Using Neuroscience Features
```bash
# Standard mode includes homeostasis
uv run scripts/train.py --mode standard --epochs 20

# Full mode includes all neuroscience features
uv run scripts/train.py --mode full --epochs 20
```

### Memory Requirements by Feature
- Base model: ~8GB
- + Homeostasis: +3GB
- + Sleep-wake cycles: +3GB
- + Episodic memory: +5GB
- + All features: ~20GB+

## 💡 Tips & Tricks

### For Different GPUs
- **RTX 3090 (24GB)**: Use any mode, `fast` recommended for experiments
- **RTX 3080 (10GB)**: Use `fast` mode, maybe `standard` with reduced batch
- **RTX 3070 (8GB)**: Use `fast` mode only
- **Smaller GPUs**: Use `fast` mode with `--batch-size 2`

### Data Preparation
```bash
# Convert JSONL to binary format
uv run cortexgpt/data/prepare_data.py \
    --input-file your_data.jsonl \
    --output-file your_data.bin

# Create demo data
uv run scripts/data/create_demo_data.py
```

### Troubleshooting

**OOM Errors**:
```bash
# Reduce batch size
--batch-size 4

# Use gradient accumulation
--gradient-accumulation 4

# Use fast mode
--mode fast
```

**Slow Training**:
```bash
# Increase workers
--num-workers 12

# Check GPU usage
nvidia-smi

# Use larger batch if GPU allows
--batch-size 16
```

**Poor Convergence**:
```bash
# Increase learning rate
--lr 2e-4

# Reduce warmup
--warmup-ratio 0.02

# Try different mode
--mode standard
```

## 📚 Legacy Training (Not Recommended)

The old training script is still available but has performance issues:

```bash
# Old method - 20x slower, poor learning rate
uv run scripts/train_cortexgpt.py --epochs 10

# Why it's slow:
# - Learning rate too small (5e-5)
# - Poor data loading (few workers)
# - Excessive parameter group scaling
```

## 🎯 Recommendations

1. **Always use the new `train.py` script**
2. **Start with `fast` mode for experiments**
3. **Monitor GPU usage to ensure efficiency**
4. **Use W&B for detailed monitoring**
5. **Adjust batch size for your GPU memory**

Remember: The new optimized training is **20x faster** with **better convergence**!