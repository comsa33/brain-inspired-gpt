# CortexGPT Training Optimization Guide

## 🚨 Problem: Extremely Slow Training (39.51s/iteration)

Your current training speed would take **~388 days** to complete 20 epochs!

## 🎯 Root Causes

1. **Data Loading Bottleneck** (Main Issue)
   - Only 2 workers for data loading
   - CPU can't keep up with GPU
   - GPU sits idle waiting for data

2. **Large Dataset**
   - 254,976 training samples
   - 42,496 iterations per epoch

3. **Complex Neuroscience Features**
   - Homeostatic plasticity calculations
   - Sleep-wake cycle computations
   - Additional overhead per batch

## 🚀 Solutions

### 1. Quick Fix - Increase Data Workers

```bash
# Use more data loading workers
uv run scripts/train_neuroscience_3090.py --epochs 20 --num-workers 8 --wandb
```

### 2. Use the Fast Training Script

```bash
# Optimized for speed with minimal features
uv run scripts/train_fast_3090.py --epochs 10 --num-workers 8 --minimal --wandb

# With slightly more features
uv run scripts/train_fast_3090.py --epochs 10 --num-workers 8 --batch-size 12 --gradient-accumulation 1
```

### 3. Reduce Dataset Size for Testing

```bash
# Create a smaller dataset for faster iteration
head -n 10000 data/train.jsonl > data/train_small.jsonl

# Convert to binary
uv run cortexgpt/data/prepare_data.py \
    --input-file data/train_small.jsonl \
    --output-file data/train_small.bin

# Train on smaller dataset
uv run scripts/train_neuroscience_3090.py \
    --train-data data/train_small.bin \
    --epochs 5 \
    --num-workers 8
```

### 4. Performance Monitoring

```bash
# Monitor GPU utilization (should be >90%)
watch -n 1 nvidia-smi

# Check if CPU is the bottleneck
htop  # Look for high CPU usage during training
```

## 📊 Expected Performance Improvements

| Configuration | Speed | Time per Epoch | 20 Epochs |
|--------------|-------|----------------|-----------|
| Current (2 workers) | 39.51s/iter | ~19.4 days | ~388 days |
| With 8 workers | ~5s/iter | ~2.5 days | ~50 days |
| Fast script (minimal) | ~1-2s/iter | ~12-24 hours | ~10-20 days |
| Smaller dataset (10k) | ~0.5s/iter | ~1 hour | ~20 hours |

## 🔧 Additional Optimizations

### 1. Enable Mixed Precision (Future)
```python
# Add to trainer for 2x speedup
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(inputs)
```

### 2. Gradient Checkpointing
```bash
# Reduce memory usage, slight speed penalty
--gradient-checkpointing
```

### 3. Distributed Training (Multi-GPU)
```bash
# If you have multiple GPUs
torchrun --nproc_per_node=2 scripts/train_neuroscience_3090.py
```

## 🎯 Recommended Approach

1. **Start with fast script** for initial experiments
2. **Use 8+ workers** for data loading
3. **Test on smaller dataset** first
4. **Gradually enable features** once training is stable
5. **Monitor GPU utilization** to ensure it's >90%

## 💡 Pro Tips

- **Batch Size**: Larger = more GPU utilization, but watch memory
- **Workers**: Set to number of CPU cores (usually 8-16)
- **Pin Memory**: Already enabled, keeps data ready for GPU
- **Persistent Workers**: Reduces worker startup overhead

The key is to keep the GPU fed with data continuously!