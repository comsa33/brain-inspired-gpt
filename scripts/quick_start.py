#!/usr/bin/env python3
"""
Quick start script for CortexGPT
Guides users through data download and training
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"🧠 {text}")
    print(f"{'='*60}\n")


def run_command(cmd):
    """Run command and return success status"""
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print_header("CortexGPT Quick Start")
    
    print("Welcome to CortexGPT! This script will help you get started.\n")
    
    # Step 1: Check for demo data
    data_dir = Path("data")
    demo_train = data_dir / "sample_train.bin"
    demo_val = data_dir / "sample_val.bin"
    
    if not demo_train.exists() or not demo_val.exists():
        print("📥 Demo data not found. Creating demo dataset...")
        if run_command(["uv", "run", "scripts/data/create_demo_data.py"]):
            print("✅ Demo data created!")
        else:
            print("❌ Failed to create demo data.")
            return
    else:
        print("✅ Demo data found!")
    
    # Step 2: Choose training mode
    print_header("Choose Training Mode")
    
    print("Select training mode:")
    print("1. Fast mode - Quick experiments (recommended)")
    print("2. Standard mode - Balanced features")
    print("3. Full mode - All features (requires 20GB+ GPU)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    mode_map = {
        "1": "fast",
        "2": "standard",
        "3": "full"
    }
    
    if choice == "4":
        print("Exiting...")
        return
    
    if choice not in mode_map:
        print("❌ Invalid choice.")
        return
    
    mode = mode_map[choice]
    
    # Step 3: Configure training
    print_header("Training Configuration")
    
    print(f"Training mode: {mode}")
    print("Default settings will be auto-detected based on your GPU.")
    
    epochs = input("\nNumber of epochs (default: 5): ").strip()
    if not epochs:
        epochs = "5"
    
    use_wandb = input("Enable W&B logging? (y/n, default: n): ").strip().lower()
    
    # Step 4: Start training
    print_header("Starting Training")
    
    print("\n🚀 Starting optimized training...")
    print("Expected performance:")
    print("- Training speed: ~1-2 seconds per iteration")
    print("- Much faster than legacy training!")
    
    cmd = [
        "uv", "run", "scripts/train.py",
        "--mode", mode,
        "--epochs", epochs,
        "--checkpoint-dir", "checkpoints/quickstart"
    ]
    
    if use_wandb == 'y':
        cmd.append("--wandb")
    
    if run_command(cmd):
        print("\n✅ Training complete!")
        
        # Step 5: Test generation
        print_header("Testing Text Generation")
        
        test = input("\nWould you like to test text generation? (y/n): ").strip().lower()
        
        if test == 'y':
            prompt = input("Enter a prompt (or press Enter for default): ").strip()
            if not prompt:
                prompt = "The future of AI is"
            
            cmd = [
                "uv", "run", "scripts/generate.py",
                "--checkpoint", "checkpoints/quickstart/cortex_gpt_best.pt",
                "--prompt", prompt,
                "--max-length", "50"
            ]
            
            print(f"\n🎯 Generating text...")
            run_command(cmd)
    else:
        print("\n❌ Training failed. Please check the error messages above.")
    
    # Step 6: Next steps
    print_header("Next Steps")
    
    print("🎉 Congratulations! You've completed the quick start.\n")
    print("Next steps:")
    print("1. Download larger datasets:")
    print("   uv run scripts/download_data.py --dataset english_large")
    print("\n2. Train with your data:")
    print("   uv run scripts/train.py --train-data your_data.bin --mode fast --epochs 10")
    print("\n3. Try different modes:")
    print("   uv run scripts/train.py --mode standard  # More features")
    print("   uv run scripts/train.py --mode full      # All features")
    print("\n4. Monitor training:")
    print("   - GPU: watch -n 1 nvidia-smi")
    print("   - Logs: tail -f wandb/latest-run/logs/debug.log")
    
    print("\nFor more options: uv run scripts/train.py --help")
    print("\nHappy training! 🚀")


if __name__ == "__main__":
    main()