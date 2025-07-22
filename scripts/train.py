#!/usr/bin/env python3
"""
Unified CortexGPT Training Script
Optimized for both performance and functionality
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from cortexgpt.models.cortex_gpt import CortexGPT, UnifiedCortexConfig
from cortexgpt.training.train_cortex_gpt import UnifiedCortexTrainer
from cortexgpt.data.dataset import TokenizedDataset


def detect_gpu():
    """Detect GPU and recommend settings"""
    if not torch.cuda.is_available():
        return None, {}
    
    gpu_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # GPU profiles with optimized settings
    if "3090" in gpu_name or memory_gb >= 24:
        return "RTX 3090", {
            "batch_size": 12,
            "gradient_accumulation": 1,
            "dim": 512,
            "lr": 1e-4,
            "num_workers": 8
        }
    elif "3080" in gpu_name or memory_gb >= 10:
        return "RTX 3080", {
            "batch_size": 8,
            "gradient_accumulation": 2,
            "dim": 384,
            "lr": 1e-4,
            "num_workers": 6
        }
    elif "3070" in gpu_name or memory_gb >= 8:
        return "RTX 3070", {
            "batch_size": 4,
            "gradient_accumulation": 4,
            "dim": 256,
            "lr": 1e-4,
            "num_workers": 4
        }
    else:
        return "Other GPU", {
            "batch_size": 2,
            "gradient_accumulation": 8,
            "dim": 256,
            "lr": 5e-5,
            "num_workers": 4
        }


def main():
    parser = argparse.ArgumentParser(
        description="CortexGPT Training - Unified and Optimized",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training mode
    parser.add_argument("--mode", type=str, choices=["fast", "standard", "full"], 
                       default="fast", help="Training mode: fast (minimal), standard, or full (all features)")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--epochs", type=int, default=10,
                           help="Number of epochs")
    train_group.add_argument("--batch-size", type=int, default=None,
                           help="Batch size (auto-detected if not specified)")
    train_group.add_argument("--gradient-accumulation", type=int, default=None,
                           help="Gradient accumulation steps")
    train_group.add_argument("--lr", type=float, default=None,
                           help="Learning rate (auto-detected if not specified)")
    train_group.add_argument("--warmup-ratio", type=float, default=0.05,
                           help="Warmup ratio (5% recommended)")
    train_group.add_argument("--num-workers", type=int, default=None,
                           help="Data loading workers (auto-detected if not specified)")
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--dim", type=int, default=None,
                           help="Model dimension (auto-detected if not specified)")
    model_group.add_argument("--vocab-size", type=int, default=50257,
                           help="Vocabulary size")
    
    # Data
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--train-data", type=str, default="data/sample_train.bin",
                           help="Training data path")
    data_group.add_argument("--val-data", type=str, default="data/sample_val.bin",
                           help="Validation data path")
    
    # Other options
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/cortex",
                       help="Checkpoint directory")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="cortex-gpt",
                       help="W&B project name")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Auto-detect GPU and set defaults
    gpu_name, gpu_config = detect_gpu() or ("CPU", {})
    
    # Apply auto-detected settings if not manually specified
    if args.batch_size is None:
        args.batch_size = gpu_config.get("batch_size", 8)
    if args.gradient_accumulation is None:
        args.gradient_accumulation = gpu_config.get("gradient_accumulation", 2)
    if args.lr is None:
        args.lr = gpu_config.get("lr", 1e-4)
    if args.num_workers is None:
        args.num_workers = gpu_config.get("num_workers", 4)
    if args.dim is None:
        args.dim = gpu_config.get("dim", 512)
    
    print("=" * 80)
    print("CortexGPT Unified Training")
    print("=" * 80)
    print(f"\nDetected: {gpu_name}")
    print(f"Mode: {args.mode}")
    print(f"\nConfiguration:")
    print(f"  Model dimension: {args.dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.lr:.2e}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    print(f"  Data workers: {args.num_workers}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"\n✅ CUDA enabled with cuDNN benchmark")
    
    # Configure based on mode
    if args.mode == "fast":
        print("\n🚀 Fast mode: Minimal features for maximum speed")
        config = UnifiedCortexConfig(
            stm_capacity=32,
            ltm_dim=64,
            cortical_columns=4,
            sparsity_ratio=0.2,
            # Disable all advanced features
            enable_homeostasis=False,
            enable_sleep_wake=False,
            enable_cls=False,
            enable_metaplasticity=False,
            use_gpu_memory=False,
            async_memory_ops=False,
            enable_episodic_memory=False,
            enable_working_memory=False,
            enable_hierarchical_compression=False,
            enable_cognitive_features=False,
        )
    elif args.mode == "standard":
        print("\n⚡ Standard mode: Balanced features and performance")
        config = UnifiedCortexConfig(
            stm_capacity=64,
            ltm_dim=128,
            cortical_columns=8,
            sparsity_ratio=0.1,
            # Enable Phase 1 & selective Phase 2
            enable_homeostasis=True,
            enable_sleep_wake=False,
            enable_cls=False,
            enable_metaplasticity=False,
            # Disable Phase 3
            use_gpu_memory=False,
            async_memory_ops=False,
            enable_episodic_memory=False,
            enable_working_memory=False,
            enable_hierarchical_compression=False,
            enable_cognitive_features=False,
        )
    else:  # full
        print("\n🧠 Full mode: All features enabled (requires more memory)")
        config = UnifiedCortexConfig(
            stm_capacity=128,
            ltm_dim=256,
            cortical_columns=16,
            sparsity_ratio=0.05,
            # Enable all features
            enable_homeostasis=True,
            enable_sleep_wake=True,
            enable_cls=True,
            enable_metaplasticity=True,
            use_gpu_memory=torch.cuda.is_available(),
            async_memory_ops=True,
            enable_episodic_memory=True,
            enable_working_memory=True,
            enable_hierarchical_compression=True,
            enable_cognitive_features=True,
        )
    
    # Create model
    print("\nCreating model...")
    model = CortexGPT(config, vocab_size=args.vocab_size, dim=args.dim).to(device)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e9:.2f} GB (fp32)")
    
    # Load datasets
    print("\nLoading datasets...")
    if not os.path.exists(args.train_data):
        print(f"⚠️  Training data not found at {args.train_data}")
        print("Run: uv run scripts/data/create_demo_data.py")
        return
    
    train_dataset = TokenizedDataset(args.train_data)
    val_dataset = TokenizedDataset(args.val_data)
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    
    # Create trainer
    trainer_args = argparse.Namespace(
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.1,
        grad_clip=1.0,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=None,
        seed=args.seed,
        resume=args.resume,
        
        # Feature flags based on mode
        minimal=(args.mode == "fast"),
        enable_phase1=True,
        enable_phase2=(args.mode in ["standard", "full"]),
        enable_phase3=(args.mode == "full"),
    )
    
    trainer = UnifiedCortexTrainer(model, train_dataset, val_dataset, trainer_args)
    
    # Training tips
    print("\n" + "="*80)
    print("Training Tips:")
    print("✅ Optimized learning rate and warmup for faster convergence")
    print("✅ Multi-worker data loading enabled")
    print("✅ cuDNN benchmarking for better GPU performance")
    if args.mode == "fast":
        print("✅ Minimal mode for maximum training speed")
    print("\nMonitor training:")
    print("  GPU: watch -n 1 nvidia-smi")
    print("  Logs: tail -f wandb/latest-run/logs/debug.log")
    print("="*80 + "\n")
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Best model: {args.checkpoint_dir}/cortex_gpt_best.pt")


if __name__ == "__main__":
    main()