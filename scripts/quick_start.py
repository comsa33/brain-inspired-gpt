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
    
    # Step 1: Check data
    print("1️⃣ Checking available datasets...")
    result = subprocess.run(
        ["uv", "run", "scripts/download_data.py", "--list"],
        capture_output=True,
        text=True
    )
    
    if "❌ Not downloaded" in result.stdout:
        print("\n📥 No datasets found. Let's download some!")
        
        # Offer options
        print("\nWhich dataset would you like to start with?")
        print("1. Demo (quick test, 1K samples)")
        print("2. English Large (50K samples)")
        print("3. Korean Large (50K samples)")
        print("4. KLUE (Korean, from Hugging Face)")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        dataset_map = {
            "1": "demo",
            "2": "english_large",
            "3": "korean_large",
            "4": "klue"
        }
        
        if choice in dataset_map:
            dataset = dataset_map[choice]
            print(f"\n📥 Downloading {dataset}...")
            if run_command(["uv", "run", "scripts/download_data.py", "--dataset", dataset]):
                print("✅ Download complete!")
            else:
                print("❌ Download failed. Please check your internet connection.")
                return
        else:
            print("❌ Invalid choice.")
            return
    else:
        # Parse available datasets
        print("\n✅ Found existing datasets!")
        dataset = "demo"  # Default
    
    # Step 2: Start training
    print_header("Starting Training")
    
    print(f"Ready to train with '{dataset}' dataset!\n")
    print("Training configuration:")
    print("- Model dimension: 256 (small, for testing)")
    print("- Epochs: 5")
    print("- BGE-M3 embeddings: Enabled")
    print("- Checkpoint directory: checkpoints/quickstart")
    
    proceed = input("\nProceed with training? (y/n): ").strip().lower()
    
    if proceed == 'y':
        print("\n🚀 Starting training...")
        cmd = [
            "uv", "run", "scripts/train_cortexgpt.py",
            "--dataset", dataset,
            "--epochs", "5",
            "--dim", "256",
            "--batch-size", "4",
            "--checkpoint-dir", "checkpoints/quickstart"
        ]
        
        if run_command(cmd):
            print("\n✅ Training complete!")
            
            # Step 3: Test generation
            print_header("Testing Text Generation")
            
            print("Let's test the trained model!\n")
            
            test = input("Would you like to test text generation? (y/n): ").strip().lower()
            
            if test == 'y':
                prompt = input("Enter a prompt (or press Enter for default): ").strip()
                if not prompt:
                    prompt = "The future of AI is"
                
                cmd = [
                    "uv", "run", "scripts/generate.py",
                    "--checkpoint", "checkpoints/quickstart/model_best.pt",
                    "--prompt", prompt,
                    "--max-length", "50"
                ]
                
                print(f"\n🎯 Generating text...")
                run_command(cmd)
        else:
            print("\n❌ Training failed. Please check the error messages above.")
    
    print_header("Next Steps")
    
    print("🎉 Congratulations! You've completed the quick start.\n")
    print("Next steps:")
    print("1. Download larger datasets: uv run scripts/download_data.py --all")
    print("2. Train longer: uv run scripts/train_cortexgpt.py --dataset english_large --epochs 20")
    print("3. Try demos: uv run scripts/demos/learning_effect_demo.py")
    print("4. Read the docs: Check README.md for advanced options")
    
    print("\nHappy training! 🚀")


if __name__ == "__main__":
    main()