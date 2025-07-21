#!/usr/bin/env python3
"""
Training script that uses async multiprocessing for fast data loading
Solves the 20+ minute startup problem with large datasets
"""

import subprocess
import sys
import os

def main():
    print("🚀 Training CortexGPT with Async Multiprocessing")
    print("=" * 50)
    print()
    print("This script uses async multiprocessing to speed up data loading.")
    print("It solves the issue where training took 20+ minutes to start.")
    print()
    
    # Example command for wikitext dataset
    cmd = [
        "uv", "run", "cortexgpt/training/train_realtime.py",
        "--dataset", "wikitext",
        "--dim", "512",
        "--vocab-size", "30000",
        "--batch-size", "8",
        "--gradient-accumulation", "4",
        "--lr", "3e-4",
        "--epochs", "10",
        "--num-workers", "4",  # Important: use multiple workers for async loading
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("Features:")
    print("✅ Async multiprocessing tokenization")
    print("✅ Parallel data loading across multiple workers")
    print("✅ Starts training in seconds instead of minutes")
    print("✅ Memory-efficient streaming")
    print()
    
    # Add wandb flag if requested
    if "--wandb" in sys.argv:
        cmd.append("--wandb")
        print("📊 W&B logging enabled")
    
    # Run the training
    subprocess.run(cmd)


if __name__ == "__main__":
    main()