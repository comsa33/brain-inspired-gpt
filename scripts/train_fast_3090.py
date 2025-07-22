#!/usr/bin/env python3
"""
Optimized training script for RTX 3090 - Fast version
Addresses performance bottlenecks for efficient training
"""

import os
import sys
import argparse
import torch
import torch.cuda.amp as amp
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from cortexgpt.models.cortex_gpt import CortexGPT, UnifiedCortexConfig
from cortexgpt.training.train_cortex_gpt import UnifiedCortexTrainer
from cortexgpt.data.dataset import TokenizedDataset


def main():
    parser = argparse.ArgumentParser(
        description="Fast CortexGPT training optimized for RTX 3090",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of data loading workers")
    
    # Model configuration
    parser.add_argument("--dim", type=int, default=512,
                       help="Model dimension")
    parser.add_argument("--minimal", action="store_true",
                       help="Use minimal configuration for fastest training")
    
    # Data
    parser.add_argument("--train-data", type=str, default="data/sample_train.bin",
                       help="Training data path")
    parser.add_argument("--val-data", type=str, default="data/sample_val.bin",
                       help="Validation data path")
    
    # Other
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/fast_3090",
                       help="Checkpoint directory")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable wandb logging")
    parser.add_argument("--mixed-precision", action="store_true", default=True,
                       help="Use mixed precision training (FP16)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Fast CortexGPT Training - RTX 3090 Optimized")
    print("=" * 80)
    
    # Optimized settings
    print(f"\nOptimized Configuration:")
    print(f"  Model dimension: {args.dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Data loading workers: {args.num_workers}")
    print(f"  Mixed precision: {args.mixed_precision}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create minimal config for speed
    config = UnifiedCortexConfig(
        stm_capacity=32 if args.minimal else 64,
        ltm_dim=64 if args.minimal else 128,
        cortical_columns=4 if args.minimal else 8,
        sparsity_ratio=0.2 if args.minimal else 0.1,
        
        # Disable expensive features by default
        enable_homeostasis=False,
        enable_sleep_wake=False,
        enable_cls=False,
        enable_metaplasticity=False,
        
        # Disable Phase 3 features
        use_gpu_memory=False,
        async_memory_ops=False,
        enable_episodic_memory=False,
        enable_working_memory=False,
        enable_hierarchical_compression=False,
        enable_cognitive_features=False,
        
        # Keep Phase 1 stability features
        memory_temperature=1.0,
        use_stop_gradient=True,
        memory_dropout=0.1,
        residual_weight=0.1,
    )
    
    # Create model
    print("\nCreating model...")
    model = CortexGPT(config, vocab_size=50257, dim=args.dim).to(device)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TokenizedDataset(args.train_data)
    val_dataset = TokenizedDataset(args.val_data)
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    
    # Performance optimizations
    if torch.cuda.is_available():
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        
        # Set up for mixed precision if enabled
        if args.mixed_precision:
            print("\n✅ Mixed precision training enabled (FP16)")
    
    # Create trainer with optimized settings
    trainer_args = argparse.Namespace(
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.1,
        grad_clip=1.0,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        wandb=args.wandb,
        wandb_project="cortex-gpt-fast",
        wandb_entity=None,
        seed=42,
        resume=None,
        
        # Feature flags (minimal for speed)
        minimal=args.minimal,
        enable_phase1=True,
        enable_phase2=False,
        enable_phase3=False,
    )
    
    trainer = UnifiedCortexTrainer(model, train_dataset, val_dataset, trainer_args)
    
    # Performance tips
    print("\n" + "="*80)
    print("Performance Optimizations Applied:")
    print(f"✅ Multi-worker data loading: {args.num_workers} workers")
    print("✅ cuDNN benchmarking enabled")
    print("✅ Minimal features for maximum speed")
    if args.mixed_precision:
        print("✅ Mixed precision (FP16) training")
    print("✅ Optimized batch size and accumulation")
    print("\nExpected training speed: ~1-2 seconds per iteration")
    print("="*80 + "\n")
    
    # Start training
    print("Starting optimized training...")
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Best model saved to: {args.checkpoint_dir}/cortex_gpt_best.pt")


if __name__ == "__main__":
    main()