#!/usr/bin/env python3
"""
Unified CortexGPT Training Script
Integrates all Phase 1-3 improvements for stable, efficient, brain-inspired training.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from cortexgpt.models.cortex_gpt import CortexGPT, UnifiedCortexConfig
from cortexgpt.training.train_cortex_gpt import UnifiedCortexTrainer
from cortexgpt.data.dataset import TokenizedDataset


def main():
    parser = argparse.ArgumentParser(
        description="Train Unified CortexGPT with all enhancements",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--vocab-size", type=int, default=50257,
                           help="Vocabulary size")
    model_group.add_argument("--dim", type=int, default=768,
                           help="Model dimension")
    model_group.add_argument("--stm-capacity", type=int, default=128,
                           help="Short-term memory capacity")
    model_group.add_argument("--ltm-dim", type=int, default=256,
                           help="Long-term memory dimension")
    model_group.add_argument("--cortical-columns", type=int, default=16,
                           help="Number of cortical columns")
    model_group.add_argument("--sparsity-ratio", type=float, default=0.05,
                           help="Sparsity ratio for columns")
    
    # Phase selection
    phase_group = parser.add_argument_group("Phase Selection")
    phase_group.add_argument("--enable-phase1", action="store_true", default=True,
                           help="Enable Phase 1 stability features")
    phase_group.add_argument("--enable-phase2", action="store_true", default=True,
                           help="Enable Phase 2 neuroscience features")
    phase_group.add_argument("--enable-phase3", action="store_true", default=True,
                           help="Enable Phase 3 performance features")
    phase_group.add_argument("--minimal", action="store_true",
                           help="Minimal configuration (disable all advanced features)")
    
    # Phase 1: Stability features
    stability_group = parser.add_argument_group("Phase 1: Stability Features")
    stability_group.add_argument("--memory-temperature", type=float, default=1.0,
                               help="Temperature for memory gating")
    stability_group.add_argument("--use-stop-gradient", action="store_true", default=True,
                               help="Stop gradient on memory retrieval")
    stability_group.add_argument("--memory-dropout", type=float, default=0.1,
                               help="Dropout rate for memory")
    stability_group.add_argument("--residual-weight", type=float, default=0.1,
                               help="Weight for residual connections")
    
    # Phase 2: Neuroscience features
    neuro_group = parser.add_argument_group("Phase 2: Neuroscience Features")
    neuro_group.add_argument("--enable-homeostasis", action="store_true", default=True,
                           help="Enable homeostatic plasticity")
    neuro_group.add_argument("--enable-sleep-wake", action="store_true", default=True,
                           help="Enable sleep-wake cycles")
    neuro_group.add_argument("--enable-cls", action="store_true", default=True,
                           help="Enable complementary learning systems")
    neuro_group.add_argument("--target-firing-rate", type=float, default=0.1,
                           help="Target firing rate for homeostasis")
    neuro_group.add_argument("--consolidation-cycle", type=int, default=1000,
                           help="Steps per sleep-wake cycle")
    
    # Phase 3: Performance features
    perf_group = parser.add_argument_group("Phase 3: Performance Features")
    perf_group.add_argument("--use-gpu-memory", action="store_true", default=True,
                          help="Use GPU-accelerated memory")
    perf_group.add_argument("--async-memory", action="store_true", default=True,
                          help="Enable async memory operations")
    perf_group.add_argument("--enable-episodic", action="store_true", default=True,
                          help="Enable episodic memory")
    perf_group.add_argument("--enable-working", action="store_true", default=True,
                          help="Enable working memory")
    perf_group.add_argument("--episodic-capacity", type=int, default=10000,
                          help="Episodic memory capacity")
    perf_group.add_argument("--working-memory-slots", type=int, default=8,
                          help="Working memory slots")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--batch-size", type=int, default=16,
                           help="Batch size")
    train_group.add_argument("--gradient-accumulation", type=int, default=1,
                           help="Gradient accumulation steps")
    train_group.add_argument("--epochs", type=int, default=20,
                           help="Number of epochs")
    train_group.add_argument("--lr", type=float, default=5e-5,
                           help="Learning rate")
    train_group.add_argument("--warmup-ratio", type=float, default=0.1,
                           help="Warmup ratio")
    train_group.add_argument("--weight-decay", type=float, default=0.1,
                           help="Weight decay")
    train_group.add_argument("--grad-clip", type=float, default=1.0,
                           help="Gradient clipping")
    
    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--train-data", type=str, default="data/sample_train.bin",
                          help="Training data path")
    data_group.add_argument("--val-data", type=str, default="data/sample_val.bin",
                          help="Validation data path")
    data_group.add_argument("--num-workers", type=int, default=4,
                          help="Data loading workers")
    
    # Other configuration
    other_group = parser.add_argument_group("Other Configuration")
    other_group.add_argument("--checkpoint-dir", type=str, default="checkpoints/cortex_unified",
                           help="Checkpoint directory")
    other_group.add_argument("--seed", type=int, default=42,
                           help="Random seed")
    other_group.add_argument("--wandb", action="store_true",
                           help="Enable W&B logging")
    other_group.add_argument("--wandb-project", type=str, default="cortex-gpt-unified",
                           help="W&B project name")
    other_group.add_argument("--wandb-entity", type=str, default=None,
                           help="W&B entity")
    
    args = parser.parse_args()
    
    # Handle minimal mode
    if args.minimal:
        args.enable_phase1 = False
        args.enable_phase2 = False
        args.enable_phase3 = False
        args.enable_homeostasis = False
        args.enable_sleep_wake = False
        args.enable_cls = False
        args.enable_episodic = False
        args.enable_working = False
        args.use_gpu_memory = False
        args.async_memory = False
    
    # Print configuration
    print("=" * 80)
    print("Unified CortexGPT Training")
    print("=" * 80)
    print(f"Model Configuration:")
    print(f"  Dimension: {args.dim}")
    print(f"  Vocabulary: {args.vocab_size}")
    print(f"  STM Capacity: {args.stm_capacity}")
    print(f"  Cortical Columns: {args.cortical_columns}")
    print(f"\nEnabled Features:")
    print(f"  Phase 1 (Stability): {args.enable_phase1}")
    print(f"  Phase 2 (Neuroscience): {args.enable_phase2}")
    print(f"  Phase 3 (Performance): {args.enable_phase3}")
    if args.enable_phase2:
        print(f"    - Homeostasis: {args.enable_homeostasis}")
        print(f"    - Sleep-Wake: {args.enable_sleep_wake}")
        print(f"    - CLS: {args.enable_cls}")
    if args.enable_phase3:
        print(f"    - GPU Memory: {args.use_gpu_memory}")
        print(f"    - Episodic Memory: {args.enable_episodic}")
        print(f"    - Working Memory: {args.enable_working}")
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create configuration
    config = UnifiedCortexConfig(
        # Core
        stm_capacity=args.stm_capacity,
        ltm_dim=args.ltm_dim,
        cortical_columns=args.cortical_columns,
        sparsity_ratio=args.sparsity_ratio,
        
        # Phase 1: Stability
        memory_temperature=args.memory_temperature if args.enable_phase1 else 1.0,
        use_stop_gradient=args.use_stop_gradient and args.enable_phase1,
        memory_dropout=args.memory_dropout if args.enable_phase1 else 0.0,
        residual_weight=args.residual_weight if args.enable_phase1 else 0.0,
        use_soft_sparsity=args.enable_phase1,
        
        # Phase 2: Neuroscience
        enable_homeostasis=args.enable_homeostasis and args.enable_phase2,
        enable_sleep_wake=args.enable_sleep_wake and args.enable_phase2,
        enable_cls=args.enable_cls and args.enable_phase2,
        enable_metaplasticity=args.enable_phase2,
        target_firing_rate=args.target_firing_rate,
        consolidation_cycle=args.consolidation_cycle,
        
        # Phase 3: Performance
        use_gpu_memory=args.use_gpu_memory and args.enable_phase3 and torch.cuda.is_available(),
        async_memory_ops=args.async_memory and args.enable_phase3,
        enable_episodic_memory=args.enable_episodic and args.enable_phase3,
        enable_working_memory=args.enable_working and args.enable_phase3,
        enable_hierarchical_compression=args.enable_phase3,
        episodic_capacity=args.episodic_capacity,
        working_memory_slots=args.working_memory_slots,
        enable_cognitive_features=args.enable_phase3,
    )
    
    # Create model
    print("\nCreating Unified CortexGPT model...")
    model = CortexGPT(config, args.vocab_size, args.dim)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e9:.2f} GB (fp32)")
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        train_dataset = TokenizedDataset(args.train_data)
        val_dataset = TokenizedDataset(args.val_data)
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Val samples: {len(val_dataset):,}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please prepare data using the data preparation scripts.")
        return
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = UnifiedCortexTrainer(model, train_dataset, val_dataset, args)
    
    # Training tips
    print("\n" + "="*80)
    print("Training Tips:")
    if args.enable_phase1:
        print("- Phase 1: Loss spike detection and recovery enabled")
        print("- Phase 1: Temperature-controlled memory gating active")
    if args.enable_phase2:
        print("- Phase 2: Homeostatic plasticity will stabilize activations")
        print("- Phase 2: Sleep-wake cycles will modulate learning")
    if args.enable_phase3:
        print("- Phase 3: Advanced memory systems active")
        print("- Phase 3: Monitor memory utilization and throughput")
    print("="*80 + "\n")
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(model, 'shutdown'):
            model.shutdown()
    
    # Final summary
    print(f"\nTraining complete! Check:")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  Best model: {args.checkpoint_dir}/cortex_gpt_best.pt")
    
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()