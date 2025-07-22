# Training Tips for CortexGPT

## Common Issues and Solutions

### 1. Too Many `<unk>` Tokens

**Problem**: Model generates mostly `<unk>` tokens during training.

**Cause**: Tokenizer vocabulary is too small or wasn't trained on the actual data.

**Solution**: 
- The tokenizer now automatically trains on actual dataset content
- Ensure vocabulary size is at least 5,000 for basic experiments
- For production, use 30,000-50,000 vocabulary size

### 2. Model Not Learning (High Loss)

**Problem**: Loss stays high (>10) and doesn't decrease.

**Solutions**:
- Start with smaller model: `--dim 256` instead of default 768
- Use higher learning rate initially: `--lr 1e-3`
- Reduce batch size: `--batch-size 2`
- Check if data is loaded correctly with `demo_tokenizer.py`

### 3. Recommended Training Commands

#### For Initial Testing (Overfitting Test)
```bash
# Test if model can memorize small dataset
uv run test_overfit.py
```

#### For Demo Training
```bash
# Small model, high learning rate, few epochs
uv run cortexgpt/training/train_realtime.py \
    --dataset demo \
    --dim 256 \
    --lr 1e-3 \
    --batch-size 4 \
    --epochs 20 \
    --vocab-size 10000
```

#### For Real Dataset Training
```bash
# Larger model, normal learning rate, more epochs
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --dim 512 \
    --lr 3e-4 \
    --batch-size 8 \
    --gradient-accumulation 4 \
    --epochs 50 \
    --vocab-size 30000 \
    --wandb
```

### 4. Memory Optimization

If you run out of GPU memory:
```bash
# Reduce batch size and increase gradient accumulation
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --batch-size 2 \
    --gradient-accumulation 16 \
    --dim 256
```

### 5. Monitoring Training

Use wandb for better monitoring:
```bash
uv add wandb
uv run cortexgpt/training/train_realtime.py --dataset demo --wandb
```

### 6. Debugging Tokenization

Check if tokenizer is working correctly:
```bash
uv run demo_tokenizer.py
```

This will show:
- Vocabulary size
- Token distribution (Korean vs English)
- Sample tokenization results

## Model Architecture Guidelines

### Small Model (for testing)
- `--dim 256`: Hidden dimension
- `--vocab-size 10000`: Vocabulary size
- Good for: Quick experiments, overfitting tests

### Medium Model (for demos)
- `--dim 512`: Hidden dimension  
- `--vocab-size 30000`: Vocabulary size
- Good for: Demo training, small datasets

### Large Model (for production)
- `--dim 768`: Hidden dimension
- `--vocab-size 50000`: Vocabulary size  
- Good for: Real training, large datasets

## Expected Training Behavior

### First 10 Epochs
- Loss should drop from ~10 to ~5
- Model starts generating some real words (not just `<unk>`)
- Korean and English tokens both appear

### After 50 Epochs  
- Loss should be below 3.0
- Model generates coherent phrases
- Can complete simple prompts

### Overfitting Test
- Loss should go below 0.1 within 100 epochs
- Model perfectly memorizes training samples
- If this doesn't work, there's a fundamental issue

## Troubleshooting Checklist

1. ✅ Check tokenizer vocabulary size (should be >1000)
2. ✅ Verify data is loaded (check batch contents)
3. ✅ Start with small model (dim=256)
4. ✅ Use high learning rate initially (1e-3)
5. ✅ Run overfitting test first
6. ✅ Monitor with wandb for better insights