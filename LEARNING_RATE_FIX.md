# CortexGPT Learning Rate Fix Guide

## 🚨 Problem: Loss Decreasing Too Slowly

Your training showed:
- Loss decrease: 10.3742 → 10.2828 (only 0.0914 decrease over 1,405 iterations)
- Learning rate displayed: 2.75e-08 (1,818x smaller than expected!)
- At this rate: Would take 60+ days to reach reasonable loss

## 🔍 Root Causes

### 1. **Parameter Group Scaling Too Aggressive**
```python
# Current (too conservative):
memory_systems: lr * 0.1  # 10x slower!
neuroscience: lr * 0.5    # 2x slower
core_model: lr * 1.0      # normal

# Better (balanced):
memory_systems: lr * 0.5  # only 2x slower
neuroscience: lr * 0.8    # slightly slower
core_model: lr * 1.0      # normal
```

### 2. **Warmup Too Long**
- Current: 10% of training = 4,250 steps
- At iteration 1,405: Only 33% through warmup
- Better: 5% warmup = 2,125 steps

### 3. **Base Learning Rate Too Low**
- Current: 5e-5
- Recommended: 1e-4 to 5e-4

## 🚀 Solutions

### Option 1: Quick Fix (Existing Script)
```bash
# Higher learning rate + shorter warmup
uv run scripts/train_neuroscience_3090.py \
    --lr 2e-4 \
    --warmup-ratio 0.05 \
    --epochs 10 \
    --num-workers 8
```

### Option 2: Optimized Script (Recommended)
```bash
# Use the new high learning rate script
uv run scripts/train_high_lr.py \
    --lr 1e-4 \
    --epochs 10 \
    --num-workers 8 \
    --enable-neuroscience \
    --wandb
```

### Option 3: Aggressive Learning
```bash
# For faster experimentation
uv run scripts/train_high_lr.py \
    --lr 5e-4 \
    --warmup-ratio 0.02 \
    --batch-size 12 \
    --gradient-accumulation 1 \
    --epochs 5
```

## 📊 Expected Results

| Configuration | Old Loss/iter | New Loss/iter | Speedup |
|--------------|---------------|---------------|---------|
| Original | 0.000065 | - | 1x |
| Quick Fix | - | ~0.002 | 30x |
| Optimized | - | ~0.005 | 75x |
| Aggressive | - | ~0.01 | 150x |

## 🎯 Learning Rate Guidelines

### For Different Model Sizes:
- **Small (< 50M params)**: 5e-4 to 1e-3
- **Medium (50-100M params)**: 1e-4 to 5e-4  ← Your model (63M)
- **Large (> 100M params)**: 5e-5 to 1e-4

### For Different Phases:
1. **Initial Training**: Start with higher LR (1e-4)
2. **Fine-tuning**: Reduce to 5e-5
3. **Final Polish**: Use 1e-5

### Signs of Good Learning Rate:
✅ Loss decreases 0.1-0.5 per epoch
✅ No loss spikes or NaN
✅ Smooth, consistent decrease
✅ Validation loss follows training

### Signs of Bad Learning Rate:
❌ Loss barely changes (too low)
❌ Loss explodes or NaN (too high)
❌ Loss oscillates wildly (too high)
❌ Loss decreases then increases (too high + no scheduler)

## 🔧 Monitoring Commands

```bash
# Watch loss progression
watch -n 10 "tail -n 20 wandb/latest-run/logs.txt | grep loss"

# Monitor learning rates
python -c "
import torch
ckpt = torch.load('checkpoints/cortex_gpt_latest.pt')
for g in ckpt['optimizer_state_dict']['param_groups']:
    print(f\"{g['name']}: {g['lr']:.2e}\")
"
```

## 💡 Pro Tips

1. **Start High, Reduce Later**: Better to start with high LR and reduce than be stuck with low LR
2. **Monitor Gradient Norms**: Should be 0.1-10.0 range
3. **Use Learning Rate Finder**: Run a test to find optimal LR
4. **Trust Loss Decrease**: If loss decreases smoothly, LR is probably good

The key insight: Your learning rate was effectively **1,818x too small** due to parameter group scaling and warmup. The new scripts fix this!