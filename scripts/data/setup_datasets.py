#!/usr/bin/env python3
"""
Quick setup script to download and prepare datasets for CortexGPT training
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 CortexGPT Dataset Setup")
    print("=" * 50)
    
    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except:
        print("❌ uv not found. Please install uv first.")
        return
    
    # Install required packages
    print("\n📦 Installing required packages...")
    packages = ["datasets", "requests", "tqdm", "wandb"]
    
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.run(["uv", "add", package], capture_output=True)
    
    print("✅ Packages installed")
    
    # Download datasets
    print("\n📥 Downloading datasets...")
    print("   This will download small samples of popular datasets")
    
    # Run download script
    download_script = "cortexgpt/data/download_datasets.py"
    if Path(download_script).exists():
        subprocess.run(["uv", "run", download_script])
    else:
        print(f"❌ {download_script} not found")
        return
    
    # Prepare datasets
    print("\n🔧 Preparing datasets...")
    prepare_script = "cortexgpt/data/prepare_datasets.py"
    if Path(prepare_script).exists():
        subprocess.run(["uv", "run", prepare_script])
    else:
        print(f"❌ {prepare_script} not found")
        return
    
    print("\n✅ Dataset setup complete!")
    print("\n📋 Next steps:")
    print("1. Train with demo data:")
    print("   uv run cortexgpt/training/train_realtime.py --dataset demo --epochs 3")
    print("\n2. Train with real data (after downloading):")
    print("   uv run cortexgpt/training/train_realtime.py --dataset klue --epochs 10 --wandb")
    print("\n3. Resume training:")
    print("   uv run cortexgpt/training/train_realtime.py --dataset klue --resume auto")
    print("\n4. Monitor with wandb:")
    print("   wandb login  # First time only")
    print("   Then add --wandb flag to training")


if __name__ == "__main__":
    main()