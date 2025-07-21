#!/usr/bin/env python3
"""
Quick start script for Unified CortexGPT
Works with the new unified trainer and provides memory-efficient options
"""

import os
import sys
import subprocess
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"🧠 {text}")
    print(f"{'='*60}\n")


def check_gpu():
    """Check GPU availability and memory"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        return gpu_name, gpu_memory
    else:
        print("⚠️  No GPU detected, will use CPU (slow)")
        return None, 0


def create_sample_data():
    """Create sample binary data for quick testing"""
    print("\n📝 Creating sample data...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create small sample data
    vocab_size = 50257
    train_size = 100000  # 100K tokens
    val_size = 10000     # 10K tokens
    
    # Generate random token data
    train_data = np.random.randint(0, vocab_size, size=train_size, dtype=np.uint16)
    val_data = np.random.randint(0, vocab_size, size=val_size, dtype=np.uint16)
    
    # Save to binary files
    train_path = data_dir / "sample_train.bin"
    val_path = data_dir / "sample_val.bin"
    
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    
    print(f"✅ Created sample training data: {train_path} ({train_size:,} tokens)")
    print(f"✅ Created sample validation data: {val_path} ({val_size:,} tokens)")
    
    return str(train_path), str(val_path)


def get_training_config(gpu_memory):
    """Get appropriate training configuration based on GPU memory"""
    if gpu_memory >= 20:  # 3090 or better
        return {
            "profile": "3090",
            "batch_size": 4,
            "gradient_accumulation": 4,
            "dim": 512,
            "phases": ["minimal", "phase1"]
        }
    elif gpu_memory >= 10:  # 3080
        return {
            "profile": "3080",
            "batch_size": 2,
            "gradient_accumulation": 8,
            "dim": 384,
            "phases": ["minimal"]
        }
    elif gpu_memory >= 8:  # 3070
        return {
            "profile": "3070",
            "batch_size": 1,
            "gradient_accumulation": 16,
            "dim": 256,
            "phases": ["minimal"]
        }
    else:  # Low memory or CPU
        return {
            "profile": "low_memory",
            "batch_size": 1,
            "gradient_accumulation": 16,
            "dim": 256,
            "phases": ["minimal"]
        }


def run_command(cmd):
    """Run command and return success status"""
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def main():
    print_header("Unified CortexGPT Quick Start")
    
    print("Welcome to CortexGPT! This script will help you get started with the unified model.\n")
    
    # Step 1: Check GPU
    gpu_name, gpu_memory = check_gpu()
    
    # Step 2: Create or check data
    print_header("Data Preparation")
    
    train_path = Path("data/sample_train.bin")
    val_path = Path("data/sample_val.bin")
    
    if not train_path.exists() or not val_path.exists():
        train_path, val_path = create_sample_data()
    else:
        print("✅ Found existing sample data")
        print(f"   Training: {train_path}")
        print(f"   Validation: {val_path}")
    
    # Step 3: Get training configuration
    print_header("Training Configuration")
    
    config = get_training_config(gpu_memory)
    
    print(f"Based on your hardware, using configuration:")
    print(f"  Profile: {config['profile']}")
    print(f"  Model dimension: {config['dim']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation']}")
    print(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation']}")
    print(f"  Enabled phases: {', '.join(config['phases'])}")
    
    # Step 4: Choose training mode
    print("\nTraining options:")
    print("1. Quick test (2 epochs, minimal features)")
    print("2. Consumer GPU optimized (10 epochs, auto-configured)")
    print("3. Full training (20 epochs, all features - requires high-end GPU)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Quick test
        cmd = [
            "uv", "run", "scripts/train_cortexgpt.py",
            "--train-data", str(train_path),
            "--val-data", str(val_path),
            "--epochs", "2",
            "--batch-size", str(config['batch_size']),
            "--gradient-accumulation", str(config['gradient_accumulation']),
            "--dim", str(config['dim']),
            "--minimal",
            "--checkpoint-dir", "checkpoints/quickstart"
        ]
    elif choice == "2":
        # Consumer GPU optimized
        cmd = [
            "uv", "run", "scripts/train_cortexgpt_consumer_gpu.py",
            "--train-data", str(train_path),
            "--val-data", str(val_path),
            "--epochs", "10",
            "--auto-detect",
            "--checkpoint-dir", "checkpoints/consumer_gpu"
        ]
    elif choice == "3":
        # Full training (warning for low-end GPUs)
        if gpu_memory < 24:
            print("\n⚠️  WARNING: Full training requires at least 24GB GPU memory!")
            print("   Your GPU has {:.1f}GB. This may cause out-of-memory errors.".format(gpu_memory))
            proceed = input("   Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Aborting.")
                return
        
        cmd = [
            "uv", "run", "scripts/train_cortexgpt.py",
            "--train-data", str(train_path),
            "--val-data", str(val_path),
            "--epochs", "20",
            "--batch-size", "16",
            "--gradient-accumulation", "1",
            "--checkpoint-dir", "checkpoints/full"
        ]
    else:
        print("❌ Invalid choice.")
        return
    
    # Step 5: Start training
    print_header("Starting Training")
    
    print("\n🚀 Starting training with command:")
    print(" ".join(cmd))
    print()
    
    if run_command(cmd):
        print("\n✅ Training complete!")
        
        # Step 6: Test generation (optional)
        print_header("Next Steps")
        
        print("🎉 Congratulations! You've completed the quick start.\n")
        print("Next steps:")
        print("1. Monitor GPU usage: watch nvidia-smi")
        print("2. Try different configurations: edit the training script")
        print("3. Load real data: prepare your own datasets")
        print("4. Enable more features: gradually enable phase2 and phase3")
        print("5. Check logs: tensorboard --logdir checkpoints/")
        
        # Offer generation test
        test = input("\nWould you like to test text generation? (y/n): ").strip().lower()
        
        if test == 'y':
            # Find the checkpoint
            checkpoint_dirs = {
                "1": "checkpoints/quickstart",
                "2": "checkpoints/consumer_gpu",
                "3": "checkpoints/full"
            }
            checkpoint_dir = checkpoint_dirs.get(choice, "checkpoints/quickstart")
            
            # Look for best model
            best_model = Path(checkpoint_dir) / "cortex_gpt_best.pt"
            if not best_model.exists():
                # Try other common names
                for name in ["model_best.pt", "checkpoint_final.pt", "cortex_gpt_recovery.pt"]:
                    alt_path = Path(checkpoint_dir) / name
                    if alt_path.exists():
                        best_model = alt_path
                        break
            
            if best_model.exists():
                print(f"\n📝 Testing generation with model: {best_model}")
                
                # Create a simple generation script inline
                gen_script = '''
import torch
from cortexgpt.models.cortex_gpt import CortexGPT

# Load model
checkpoint = torch.load("{}", map_location='cpu')
config = checkpoint['config']
model = CortexGPT(config, vocab_size=50257, dim=config.ltm_dim)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate
prompt = torch.randint(0, 50257, (1, 10))  # Random prompt
with torch.no_grad():
    output = model.generate(prompt, max_length=50)
print("Generated tokens:", output.shape)
print("(Real text generation requires a proper tokenizer)")
'''.format(best_model)
                
                # Save and run
                gen_path = Path("test_generation.py")
                gen_path.write_text(gen_script)
                
                subprocess.run(["uv", "run", str(gen_path)])
                gen_path.unlink()  # Clean up
            else:
                print(f"❌ No model checkpoint found in {checkpoint_dir}")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("- Out of memory: Reduce batch_size or model dimension")
        print("- Missing dependencies: Run 'uv sync'")
        print("- Data issues: Check data file paths")
    
    print("\nHappy training! 🚀")


if __name__ == "__main__":
    main()