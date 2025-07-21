# CortexGPT Project Status

## ✅ Completed Tasks

### 1. Fixed OOM Issues for RTX 3090
- Created `train_cortexgpt_consumer_gpu.py` with GPU-specific profiles
- Implemented gradient accumulation support in unified trainer
- Reduced default configurations for consumer GPUs

### 2. Neuroscience Features Support
- Created `train_neuroscience_3090.py` for selective feature enabling
- Added memory-optimized configurations for Phase 2 features
- Updated READMEs with detailed neuroscience training instructions

### 3. Documentation Updates
- Updated README.md and README_KR.md with:
  - Consumer GPU configurations
  - Neuroscience feature usage
  - Memory optimization techniques
  - Troubleshooting guides

### 4. Code Cleanup
- Removed old phase-specific implementations (phase2, phase3, stable)
- Moved documentation to docs/ directory
- Cleaned up test and temporary files
- Removed wandb logs

## 📁 Current Project Structure

### Core Files (Unified Implementation)
- `cortexgpt/models/cortex_gpt.py` - Main model interface
- `cortexgpt/models/cortex_gpt_unified.py` - Unified implementation with all phases
- `cortexgpt/training/train_cortex_gpt.py` - Unified trainer with gradient accumulation

### Training Scripts
- `scripts/train_cortexgpt.py` - Main training script
- `scripts/train_cortexgpt_consumer_gpu.py` - Consumer GPU optimized training
- `scripts/train_neuroscience_3090.py` - Neuroscience features for RTX 3090
- `scripts/quick_start_unified.py` - Quick start guide with auto-detection

### Documentation
- `README.md` - Main documentation (updated)
- `README_KR.md` - Korean documentation (updated)
- `PHASE1_SUMMARY.md` - Phase 1 development history
- `FIXES_SUMMARY.md` - Recent fixes documentation
- `docs/` - Additional technical documentation

## 🚀 Ready-to-Use Commands

### For RTX 3090 Users

#### Basic Training (Minimal Features)
```bash
uv run scripts/train_cortexgpt_consumer_gpu.py --auto-detect --epochs 10
```

#### Neuroscience Features
```bash
# Both homeostasis and sleep-wake (default)
uv run scripts/train_neuroscience_3090.py --epochs 20

# Only homeostasis (lower memory)
uv run scripts/train_neuroscience_3090.py --homeostasis-only --epochs 20
```

#### Quick Start
```bash
uv run scripts/quick_start_unified.py
# Choose option 2 for consumer GPU optimized
```

### Memory Usage Guidelines

| Configuration | Memory Usage | Features |
|--------------|--------------|----------|
| Minimal | 8-10GB | Base model only |
| Phase 1 | 10-12GB | + Stability features |
| + Homeostasis | 12-15GB | + Homeostatic plasticity |
| + Sleep-Wake | 15-18GB | + Sleep-wake cycles |
| Full (Default) | >20GB | All features enabled |

## 🔧 Key Improvements

1. **Gradient Accumulation**: Enables larger effective batch sizes on limited memory
2. **Selective Feature Enabling**: Turn on only needed features to save memory
3. **Auto GPU Detection**: Automatically configures for your GPU
4. **Memory Monitoring**: Built-in warnings and recommendations

## 📝 Notes

- Default configuration requires >20GB memory (not suitable for consumer GPUs)
- Use provided scripts for consumer GPU training
- Monitor GPU memory with `watch -n 1 nvidia-smi`
- Start with minimal features and gradually enable more

The project is now fully functional on consumer GPUs with intelligent memory management!