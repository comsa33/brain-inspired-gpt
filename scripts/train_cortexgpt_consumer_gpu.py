#!/usr/bin/env python3
"""
CortexGPT Training Script optimized for Consumer GPUs (3090 and below)
Provides memory-efficient configurations and gradient accumulation support.
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


# GPU Configuration Profiles
GPU_PROFILES = {
    "3090": {
        "memory_gb": 24,
        "batch_size": 4,
        "gradient_accumulation": 4,  # Effective batch size: 16
        "model_dim": 512,  # Reduced from 768
        "phases": {
            "minimal": True,  # Start with minimal for testing
            "phase1": True,   # Basic stability features
            "phase2": False,  # Disable neuroscience features initially
            "phase3": False,  # Disable advanced memory features initially
        }
    },
    "3080": {
        "memory_gb": 10,
        "batch_size": 2,
        "gradient_accumulation": 8,  # Effective batch size: 16
        "model_dim": 384,
        "phases": {
            "minimal": True,
            "phase1": True,
            "phase2": False,
            "phase3": False,
        }
    },
    "3070": {
        "memory_gb": 8,
        "batch_size": 1,
        "gradient_accumulation": 16,  # Effective batch size: 16
        "model_dim": 256,
        "phases": {
            "minimal": True,
            "phase1": True,
            "phase2": False,
            "phase3": False,
        }
    },
    "1660": {
        "memory_gb": 6,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "model_dim": 256,
        "phases": {
            "minimal": True,
            "phase1": False,
            "phase2": False,
            "phase3": False,
        }
    }
}


def detect_gpu_profile():
    """Auto-detect GPU and return appropriate profile"""
    if not torch.cuda.is_available():
        return None
        
    gpu_name = torch.cuda.get_device_name(0).lower()
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Match GPU profiles
    for profile_name, profile in GPU_PROFILES.items():
        if profile_name in gpu_name:
            return profile_name, profile
            
    # Default based on memory
    if memory_gb >= 20:
        return "3090", GPU_PROFILES["3090"]
    elif memory_gb >= 10:
        return "3080", GPU_PROFILES["3080"]
    elif memory_gb >= 8:
        return "3070", GPU_PROFILES["3070"]
    else:
        return "1660", GPU_PROFILES["1660"]


def main():
    parser = argparse.ArgumentParser(
        description="Train CortexGPT on Consumer GPUs with optimized settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # GPU selection
    gpu_group = parser.add_argument_group("GPU Configuration")
    gpu_group.add_argument("--gpu-profile", type=str, choices=list(GPU_PROFILES.keys()),
                          help="GPU profile to use (auto-detected if not specified)")
    gpu_group.add_argument("--auto-detect", action="store_true", default=True,
                          help="Auto-detect GPU and use appropriate settings")
    
    # Model configuration overrides
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--vocab-size", type=int, default=50257,
                           help="Vocabulary size")
    model_group.add_argument("--dim", type=int, default=None,
                           help="Model dimension (overrides GPU profile)")
    model_group.add_argument("--stm-capacity", type=int, default=64,
                           help="Short-term memory capacity")
    model_group.add_argument("--ltm-dim", type=int, default=128,
                           help="Long-term memory dimension")
    model_group.add_argument("--cortical-columns", type=int, default=8,
                           help="Number of cortical columns")
    model_group.add_argument("--sparsity-ratio", type=float, default=0.1,
                           help="Sparsity ratio for columns")
    
    # Phase configuration
    phase_group = parser.add_argument_group("Phase Configuration")
    phase_group.add_argument("--enable-all-phases", action="store_true",
                           help="Enable all phases (override GPU profile)")
    phase_group.add_argument("--phase1-only", action="store_true",
                           help="Enable only Phase 1 stability features")
    phase_group.add_argument("--minimal", action="store_true",
                           help="Minimal configuration (no advanced features)")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--batch-size", type=int, default=None,
                           help="Batch size (overrides GPU profile)")
    train_group.add_argument("--gradient-accumulation", type=int, default=None,
                           help="Gradient accumulation steps (overrides GPU profile)")
    train_group.add_argument("--epochs", type=int, default=10,
                           help="Number of epochs")
    train_group.add_argument("--lr", type=float, default=5e-5,
                           help="Learning rate")
    train_group.add_argument("--warmup-ratio", type=float, default=0.1,
                           help="Warmup ratio")
    train_group.add_argument("--weight-decay", type=float, default=0.1,
                           help="Weight decay")
    train_group.add_argument("--grad-clip", type=float, default=1.0,
                           help="Gradient clipping")
    
    # Memory optimization
    memory_group = parser.add_argument_group("Memory Optimization")
    memory_group.add_argument("--fp16", action="store_true", default=True,
                            help="Use mixed precision training")
    memory_group.add_argument("--gradient-checkpointing", action="store_true", default=True,
                            help="Use gradient checkpointing")
    memory_group.add_argument("--offload-optimizer", action="store_true",
                            help="Offload optimizer states to CPU")
    
    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--train-data", type=str, default="data/sample_train.bin",
                          help="Training data path")
    data_group.add_argument("--val-data", type=str, default="data/sample_val.bin",
                          help="Validation data path")
    data_group.add_argument("--num-workers", type=int, default=2,
                          help="Data loading workers")
    
    # Other configuration
    other_group = parser.add_argument_group("Other Configuration")
    other_group.add_argument("--checkpoint-dir", type=str, default="checkpoints/cortex_consumer",
                           help="Checkpoint directory")
    other_group.add_argument("--seed", type=int, default=42,
                           help="Random seed")
    other_group.add_argument("--wandb", action="store_true",
                           help="Enable W&B logging")
    other_group.add_argument("--wandb-project", type=str, default="cortex-gpt-consumer",
                           help="W&B project name")
    
    args = parser.parse_args()
    
    # Auto-detect GPU if requested
    if args.auto_detect and not args.gpu_profile:
        detected = detect_gpu_profile()
        if detected:
            profile_name, profile = detected
            print(f"Auto-detected GPU: {profile_name}")
            args.gpu_profile = profile_name
        else:
            print("No CUDA GPU detected, using CPU mode")
            profile = {
                "batch_size": 1,
                "gradient_accumulation": 16,
                "model_dim": 256,
                "phases": {"minimal": True, "phase1": False, "phase2": False, "phase3": False}
            }
    else:
        profile = GPU_PROFILES.get(args.gpu_profile, GPU_PROFILES["3090"])
    
    # Apply GPU profile settings (if not overridden)
    if args.dim is None:
        args.dim = profile["model_dim"]
    if args.batch_size is None:
        args.batch_size = profile["batch_size"]
    if args.gradient_accumulation is None:
        args.gradient_accumulation = profile["gradient_accumulation"]
    
    # Apply phase configuration
    if args.minimal:
        phases = {"phase1": False, "phase2": False, "phase3": False}
    elif args.phase1_only:
        phases = {"phase1": True, "phase2": False, "phase3": False}
    elif args.enable_all_phases:
        phases = {"phase1": True, "phase2": True, "phase3": True}
    else:
        phases = profile["phases"]
    
    # Print configuration
    print("=" * 80)
    print("CortexGPT Training - Consumer GPU Optimized")
    print("=" * 80)
    print(f"GPU Profile: {args.gpu_profile if args.gpu_profile else 'CPU'}")
    print(f"Model Configuration:")
    print(f"  Dimension: {args.dim}")
    print(f"  Vocabulary: {args.vocab_size}")
    print(f"  STM Capacity: {args.stm_capacity}")
    print(f"  Cortical Columns: {args.cortical_columns}")
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Gradient Accumulation: {args.gradient_accumulation}")
    print(f"  Effective Batch Size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"\nEnabled Features:")
    print(f"  Phase 1 (Stability): {phases.get('phase1', False)}")
    print(f"  Phase 2 (Neuroscience): {phases.get('phase2', False)}")
    print(f"  Phase 3 (Performance): {phases.get('phase3', False)}")
    print(f"\nMemory Optimizations:")
    print(f"  Mixed Precision (FP16): {args.fp16}")
    print(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
    print(f"  Optimizer Offloading: {args.offload_optimizer}")
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
        memory_temperature=1.0 if phases.get('phase1', False) else 1.0,
        use_stop_gradient=phases.get('phase1', False),
        memory_dropout=0.1 if phases.get('phase1', False) else 0.0,
        residual_weight=0.1 if phases.get('phase1', False) else 0.0,
        use_soft_sparsity=phases.get('phase1', False),
        
        # Phase 2: Neuroscience
        enable_homeostasis=phases.get('phase2', False),
        enable_sleep_wake=phases.get('phase2', False),
        enable_cls=phases.get('phase2', False),
        enable_metaplasticity=phases.get('phase2', False),
        
        # Phase 3: Performance
        use_gpu_memory=phases.get('phase3', False) and torch.cuda.is_available(),
        async_memory_ops=False,  # Disable async for stability
        enable_episodic_memory=phases.get('phase3', False),
        enable_working_memory=phases.get('phase3', False),
        enable_hierarchical_compression=phases.get('phase3', False),
        episodic_capacity=1000,  # Reduced for consumer GPUs
        working_memory_slots=4,  # Reduced for consumer GPUs
        enable_cognitive_features=phases.get('phase3', False),
    )
    
    # Create model
    print("\nCreating CortexGPT model...")
    model = CortexGPT(config, args.vocab_size, args.dim)
    
    # Apply memory optimizations
    if args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"  Model size: {total_params * 4 / 1e9:.2f} GB (fp32)")
    print(f"  Model size: {total_params * 2 / 1e9:.2f} GB (fp16)")
    
    # Estimate memory usage
    param_memory = total_params * (2 if args.fp16 else 4) / 1e9
    activation_memory = args.batch_size * args.dim * 1024 * (2 if args.fp16 else 4) / 1e9
    estimated_memory = param_memory + activation_memory * 2  # Rough estimate
    print(f"\nEstimated Memory Usage:")
    print(f"  Parameters: {param_memory:.2f} GB")
    print(f"  Activations: {activation_memory:.2f} GB per batch")
    print(f"  Total Estimate: {estimated_memory:.2f} GB")
    
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
        # Create sample data if not found
        os.makedirs("data", exist_ok=True)
        sample_size = 10000
        train_data = np.random.randint(0, args.vocab_size, size=sample_size, dtype=np.uint16)
        val_data = np.random.randint(0, args.vocab_size, size=sample_size//10, dtype=np.uint16)
        
        train_data.tofile("data/sample_train.bin")
        val_data.tofile("data/sample_val.bin")
        
        train_dataset = TokenizedDataset("data/sample_train.bin")
        val_dataset = TokenizedDataset("data/sample_val.bin")
        print(f"  Created sample train data: {len(train_dataset):,}")
        print(f"  Created sample val data: {len(val_dataset):,}")
    
    # Create trainer with gradient accumulation support
    print("\nInitializing trainer...")
    # Add gradient accumulation to args for trainer
    args.enable_phase1 = phases.get('phase1', False)
    args.enable_phase2 = phases.get('phase2', False) 
    args.enable_phase3 = phases.get('phase3', False)
    
    trainer = UnifiedCortexTrainer(model, train_dataset, val_dataset, args)
    
    # Training tips
    print("\n" + "="*80)
    print("Training Tips for Consumer GPUs:")
    print("- Using gradient accumulation for larger effective batch size")
    print("- Monitor GPU memory usage with nvidia-smi")
    print("- If OOM occurs, reduce batch size or model dimension")
    print("- Start with minimal/phase1 only, then gradually enable features")
    print("- Consider using fp16 and gradient checkpointing")
    print("="*80 + "\n")
    
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
            print("1. Reduce batch size (current: {})".format(args.batch_size))
            print("2. Reduce model dimension (current: {})".format(args.dim))
            print("3. Disable advanced features (use --minimal)")
            print("4. Enable optimizer offloading (--offload-optimizer)")
            print("5. Use a smaller GPU profile")
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
    
    # Final summary
    print(f"\nTraining complete! Check:")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  Best model: {args.checkpoint_dir}/cortex_gpt_best.pt")
    
    if args.wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    main()