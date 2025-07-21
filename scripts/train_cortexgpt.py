#!/usr/bin/env python3
"""
CortexGPT Training Script
Simple and clean training interface for CortexGPT
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Train CortexGPT model")
    
    # Basic arguments
    parser.add_argument("--dataset", type=str, default="demo",
                       choices=["demo", "english_small", "english_large", "korean_small", 
                               "korean_large", "wikitext", "openwebtext", "c4_en", 
                               "klue", "combined"],
                       help="Dataset to use for training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    
    # Model arguments
    parser.add_argument("--dim", type=int, default=768,
                       help="Model dimension")
    parser.add_argument("--use-bge", action="store_true", default=True,
                       help="Use BGE-M3 hybrid embeddings (always enabled)")
    parser.add_argument("--bge-stage", type=int, default=1, choices=[1, 2],
                       help="BGE training stage (1=freeze BGE, 2=fine-tune all)")
    
    # Other arguments
    parser.add_argument("--wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Build command
    cmd = [
        "uv", "run", "cortexgpt/training/train_realtime.py",
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--dim", str(args.dim),
        "--checkpoint-dir", args.checkpoint_dir,
        "--gradient-accumulation", "4"  # Default value
    ]
    
    # Always use BGE embeddings
    cmd.extend(["--use-bge-embeddings", "--embedding-stage", str(args.bge_stage)])
    
    if args.wandb:
        cmd.append("--wandb")
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    # Print command
    print("🚀 Starting CortexGPT training...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Execute
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()