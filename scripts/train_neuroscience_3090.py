#!/usr/bin/env python3
"""
Neuroscience-focused training for RTX 3090
Optimized to run Phase 2 features within 24GB memory constraints
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
        description="Train CortexGPT with Neuroscience features on RTX 3090",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--epochs", type=int, default=20,
                           help="Number of epochs")
    train_group.add_argument("--lr", type=float, default=5e-5,
                           help="Learning rate")
    train_group.add_argument("--warmup-ratio", type=float, default=0.1,
                           help="Warmup ratio")
    
    # Neuroscience options
    neuro_group = parser.add_argument_group("Neuroscience Options")
    neuro_group.add_argument("--full-neuroscience", action="store_true",
                           help="Enable all neuroscience features (may OOM)")
    neuro_group.add_argument("--homeostasis-only", action="store_true",
                           help="Enable only homeostatic plasticity")
    neuro_group.add_argument("--sleep-wake-only", action="store_true",
                           help="Enable only sleep-wake cycles")
    
    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--train-data", type=str, default="data/sample_train.bin",
                          help="Training data path")
    data_group.add_argument("--val-data", type=str, default="data/sample_val.bin",
                          help="Validation data path")
    
    # Other
    other_group = parser.add_argument_group("Other Configuration")
    other_group.add_argument("--checkpoint-dir", type=str, default="checkpoints/neuro_3090",
                           help="Checkpoint directory")
    other_group.add_argument("--wandb", action="store_true",
                           help="Enable W&B logging")
    
    args = parser.parse_args()
    
    # RTX 3090 Optimized Configuration
    print("=" * 80)
    print("CortexGPT Neuroscience Training - RTX 3090 Optimized")
    print("=" * 80)
    
    # Memory-efficient settings for 3090
    BATCH_SIZE = 6  # Reduced from 16
    GRADIENT_ACCUMULATION = 3  # Effective batch size: 18
    MODEL_DIM = 512  # Reduced from 768
    STM_CAPACITY = 64  # Reduced from 128
    LTM_DIM = 128  # Reduced from 256
    CORTICAL_COLUMNS = 8  # Reduced from 16
    
    # Reduced memory features
    EPISODIC_CAPACITY = 1000  # Reduced from 10000
    WORKING_MEMORY_SLOTS = 4  # Reduced from 8
    
    print(f"Configuration for RTX 3090 (24GB):")
    print(f"  Model dimension: {MODEL_DIM}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  STM capacity: {STM_CAPACITY}")
    print(f"  Cortical columns: {CORTICAL_COLUMNS}")
    
    # Determine which features to enable
    enable_homeostasis = False
    enable_sleep_wake = False
    enable_cls = False
    enable_metaplasticity = False
    
    if args.full_neuroscience:
        print("\n⚠️  WARNING: Full neuroscience mode may cause OOM!")
        print("   Enabling all Phase 2 features...")
        enable_homeostasis = True
        enable_sleep_wake = True
        enable_cls = True
        enable_metaplasticity = True
    elif args.homeostasis_only:
        print("\n✅ Enabling homeostatic plasticity only")
        enable_homeostasis = True
    elif args.sleep_wake_only:
        print("\n✅ Enabling sleep-wake cycles only")
        enable_sleep_wake = True
    else:
        # Default: both homeostasis and sleep-wake
        print("\n✅ Enabling homeostasis + sleep-wake (default)")
        enable_homeostasis = True
        enable_sleep_wake = True
    
    print("\nEnabled Neuroscience Features:")
    print(f"  Homeostatic Plasticity: {enable_homeostasis}")
    print(f"  Sleep-Wake Cycles: {enable_sleep_wake}")
    print(f"  Complementary Learning: {enable_cls}")
    print(f"  BCM Metaplasticity: {enable_metaplasticity}")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Set memory fraction to prevent fragmentation
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Create configuration
    config = UnifiedCortexConfig(
        # Core - Reduced sizes
        stm_capacity=STM_CAPACITY,
        ltm_dim=LTM_DIM,
        cortical_columns=CORTICAL_COLUMNS,
        sparsity_ratio=0.1,  # Increased sparsity
        
        # Phase 1: Stability (always enabled)
        memory_temperature=1.0,
        use_stop_gradient=True,
        memory_dropout=0.1,
        residual_weight=0.1,
        use_soft_sparsity=True,
        
        # Phase 2: Neuroscience (selective)
        enable_homeostasis=enable_homeostasis,
        enable_sleep_wake=enable_sleep_wake,
        enable_cls=enable_cls,
        enable_metaplasticity=enable_metaplasticity,
        target_firing_rate=0.1,
        consolidation_cycle=1000,
        
        # Phase 3: Performance (mostly disabled for memory)
        use_gpu_memory=False,  # Disable GPU memory to save VRAM
        async_memory_ops=False,
        enable_episodic_memory=False,  # Disable to save memory
        enable_working_memory=False,  # Disable to save memory
        enable_hierarchical_compression=False,
        enable_cognitive_features=False,
    )
    
    # Create model
    print("\nCreating CortexGPT model...")
    model = CortexGPT(config, vocab_size=50257, dim=MODEL_DIM)
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e9:.2f} GB (fp32)")
    
    # Estimate memory usage
    param_memory = total_params * 4 / 1e9  # fp32
    activation_memory = BATCH_SIZE * MODEL_DIM * 1024 * 4 / 1e9
    estimated_memory = param_memory + activation_memory * 3  # Rough estimate
    print(f"\nEstimated Memory Usage:")
    print(f"  Parameters: {param_memory:.2f} GB")
    print(f"  Activations: {activation_memory:.2f} GB per batch")
    print(f"  Total Estimate: {estimated_memory:.2f} GB")
    
    if estimated_memory > 20:
        print("\n⚠️  WARNING: Estimated memory usage may exceed 3090 capacity!")
        print("   Consider reducing batch size or disabling more features.")
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        train_dataset = TokenizedDataset(args.train_data)
        val_dataset = TokenizedDataset(args.val_data)
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Val samples: {len(val_dataset):,}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("\nCreating sample data...")
        os.makedirs("data", exist_ok=True)
        sample_size = 10000
        train_data = np.random.randint(0, 50257, size=sample_size, dtype=np.uint16)
        val_data = np.random.randint(0, 50257, size=sample_size//10, dtype=np.uint16)
        
        train_data.tofile("data/sample_train.bin")
        val_data.tofile("data/sample_val.bin")
        
        train_dataset = TokenizedDataset("data/sample_train.bin")
        val_dataset = TokenizedDataset("data/sample_val.bin")
    
    # Create trainer
    print("\nInitializing trainer...")
    # Create args object for trainer
    trainer_args = argparse.Namespace(
        batch_size=BATCH_SIZE,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.1,
        grad_clip=1.0,
        num_workers=2,  # Reduced workers
        checkpoint_dir=args.checkpoint_dir,
        wandb=args.wandb,
        wandb_project="cortex-gpt-neuro-3090",
        wandb_entity=None,
        # Phase flags for trainer
        enable_phase1=True,
        enable_phase2=True,
        enable_phase3=False,
    )
    
    trainer = UnifiedCortexTrainer(model, train_dataset, val_dataset, trainer_args)
    
    # Training tips
    print("\n" + "="*80)
    print("Training Tips for Neuroscience Features on RTX 3090:")
    print("- Monitor GPU memory with: watch -n 1 nvidia-smi")
    print("- If OOM occurs, try --homeostasis-only or --sleep-wake-only")
    print("- Gradient accumulation is set to 3 for effective batch size of 18")
    print("- Phase 3 features (episodic/working memory) are disabled to save memory")
    print("- Consider using mixed precision training for additional savings")
    print("="*80 + "\n")
    
    # Memory management settings
    if torch.cuda.is_available():
        # Clear cache before training
        torch.cuda.empty_cache()
        
        # Set environment variable for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
        print("\nTraining completed successfully!")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "="*80)
            print("ERROR: Out of Memory!")
            print("Try these solutions:")
            print("1. Use --homeostasis-only or --sleep-wake-only")
            print("2. Reduce batch size further (current: {})".format(BATCH_SIZE))
            print("3. Disable more features")
            print("4. Use the minimal trainer instead")
            print("="*80)
        else:
            print(f"\nTraining failed with error: {e}")
        raise
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Cleanup
        if hasattr(model, 'shutdown'):
            model.shutdown()
    
    print(f"\nTraining complete! Check:")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  Best model: {args.checkpoint_dir}/cortex_gpt_best.pt")


if __name__ == "__main__":
    main()