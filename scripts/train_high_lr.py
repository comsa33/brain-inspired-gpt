#!/usr/bin/env python3
"""
High learning rate training script for faster convergence
Fixes the slow learning issue in CortexGPT
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


class HighLRTrainer(UnifiedCortexTrainer):
    """Modified trainer with better learning rate configuration"""
    
    def _create_optimizer(self):
        """Create optimizer with more balanced learning rates"""
        # More balanced parameter groups
        param_groups = [
            # Memory systems - only 2x slower (not 10x)
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(x in n for x in ['stm', 'ltm', 'episodic', 'working', 'memory'])],
                'lr': self.args.lr * 0.5,  # Changed from 0.1 to 0.5
                'name': 'memory_systems'
            },
            # Neuroscience components - slightly slower
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(x in n for x in ['homeostatic', 'oscillator', 'cls', 'metaplastic'])],
                'lr': self.args.lr * 0.8,  # Changed from 0.5 to 0.8
                'name': 'neuroscience'
            },
            # Core model - full learning rate
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(x in n for x in ['stm', 'ltm', 'episodic', 'working', 'memory',
                                                      'homeostatic', 'oscillator', 'cls', 'metaplastic'])],
                'lr': self.args.lr,
                'name': 'core_model'
            }
        ]
        
        # Filter out empty groups
        param_groups = [g for g in param_groups if len(list(g['params'])) > 0]
        
        print("\nParameter groups:")
        for g in param_groups:
            num_params = sum(p.numel() for p in g['params'])
            print(f"  {g['name']}: {num_params:,} params, lr={g['lr']:.2e}")
        
        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),  # More standard betas
            eps=1e-8
        )


def main():
    parser = argparse.ArgumentParser(
        description="High learning rate CortexGPT training for faster convergence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4, 2x higher)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                       help="Warmup ratio (default: 0.05, shorter warmup)")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of data loading workers")
    
    # Model configuration
    parser.add_argument("--dim", type=int, default=512,
                       help="Model dimension")
    parser.add_argument("--enable-neuroscience", action="store_true",
                       help="Enable neuroscience features")
    
    # Data
    parser.add_argument("--train-data", type=str, default="data/sample_train.bin",
                       help="Training data path")
    parser.add_argument("--val-data", type=str, default="data/sample_val.bin",
                       help="Validation data path")
    
    # Other
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/high_lr",
                       help="Checkpoint directory")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable wandb logging")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("High Learning Rate CortexGPT Training")
    print("=" * 80)
    
    # Configuration
    print(f"\nOptimized Learning Rate Configuration:")
    print(f"  Base LR: {args.lr:.2e} (2x default)")
    print(f"  Memory systems LR: {args.lr * 0.5:.2e} (was {args.lr * 0.1:.2e})")
    print(f"  Neuroscience LR: {args.lr * 0.8:.2e} (was {args.lr * 0.5:.2e})")
    print(f"  Warmup ratio: {args.warmup_ratio} (shorter)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create config
    config = UnifiedCortexConfig(
        stm_capacity=64,
        ltm_dim=128,
        cortical_columns=8,
        sparsity_ratio=0.1,
        
        # Enable neuroscience if requested
        enable_homeostasis=args.enable_neuroscience,
        enable_sleep_wake=args.enable_neuroscience,
        enable_cls=False,
        enable_metaplasticity=False,
        
        # Disable Phase 3 for memory
        use_gpu_memory=False,
        async_memory_ops=False,
        enable_episodic_memory=False,
        enable_working_memory=False,
        enable_hierarchical_compression=False,
        enable_cognitive_features=False,
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
    
    # Calculate warmup steps
    steps_per_epoch = len(train_dataset) // (args.batch_size * args.gradient_accumulation)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    print(f"\nTraining Schedule:")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Max LR reached at step: {warmup_steps}")
    
    # Create trainer with custom trainer class
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
        wandb_project="cortex-gpt-high-lr",
        wandb_entity=None,
        seed=42,
        resume=None,
        
        # Feature flags
        minimal=False,
        enable_phase1=True,
        enable_phase2=args.enable_neuroscience,
        enable_phase3=False,
    )
    
    trainer = HighLRTrainer(model, train_dataset, val_dataset, trainer_args)
    
    # Tips
    print("\n" + "="*80)
    print("Learning Rate Optimization Tips:")
    print("✅ Higher base learning rate (1e-4 vs 5e-5)")
    print("✅ Balanced parameter group rates (0.5x, 0.8x, 1.0x)")
    print("✅ Shorter warmup (5% vs 10%)")
    print("✅ Better optimizer betas (0.9, 0.999)")
    print("\nExpected improvements:")
    print("- Faster initial loss decrease")
    print("- Better gradient flow through all components")
    print("- Reach good performance in fewer epochs")
    print("="*80 + "\n")
    
    # Start training
    print("Starting training with optimized learning rates...")
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Best model saved to: {args.checkpoint_dir}/cortex_gpt_best.pt")


if __name__ == "__main__":
    main()