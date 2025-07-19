#!/usr/bin/env python3
"""
Prepare all datasets for Brain-Inspired GPT training
This script helps users easily download and prepare various datasets
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time


def run_command(cmd: str, description: str):
    """Run a command with proper error handling"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for Brain-Inspired GPT')
    parser.add_argument('--datasets', nargs='+', 
                        choices=['korean', 'wikipedia', 'fineweb', 'redpajama', 'all'],
                        default=['korean', 'wikipedia'],
                        help='Datasets to prepare')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Maximum samples per dataset (for testing)')
    parser.add_argument('--full', action='store_true',
                        help='Download full datasets (warning: very large)')
    
    args = parser.parse_args()
    
    # Expand 'all' option
    if 'all' in args.datasets:
        args.datasets = ['korean', 'wikipedia', 'fineweb', 'redpajama']
    
    print("🧠 Brain-Inspired GPT Dataset Preparation")
    print(f"Datasets to prepare: {args.datasets}")
    print(f"Max samples: {args.max_samples if not args.full else 'Full dataset'}")
    
    start_time = time.time()
    success_count = 0
    
    # Prepare Korean datasets
    if 'korean' in args.datasets:
        cmd = "uv run brain_gpt/training/prepare_korean_hf_datasets.py"
        if not args.full:
            cmd += f" --max-texts {args.max_samples}"
        
        if run_command(cmd, "Preparing Korean datasets (KLUE, KorQuAD)"):
            success_count += 1
    
    # Prepare Wikipedia datasets
    if 'wikipedia' in args.datasets:
        cmd = "uv run data/openwebtext/prepare_simple.py"
        cmd += " --datasets wikipedia wikipedia-ko"
        if not args.full:
            cmd += f" --max-samples {args.max_samples}"
        
        if run_command(cmd, "Preparing Wikipedia (English + Korean)"):
            success_count += 1
    
    # Prepare FineWeb educational dataset
    if 'fineweb' in args.datasets:
        cmd = "uv run data/openwebtext/prepare_fineweb.py"
        cmd += " --dataset-type fineweb-edu"
        if not args.full:
            cmd += f" --max-samples {args.max_samples}"
        
        if run_command(cmd, "Preparing FineWeb-Edu (high-quality educational content)"):
            success_count += 1
    
    # Prepare RedPajama-v2 dataset
    if 'redpajama' in args.datasets:
        cmd = "uv run data/openwebtext/prepare_redpajama.py"
        cmd += " --config sample --languages en ko"
        if not args.full:
            cmd += f" --max-samples {args.max_samples}"
        
        if run_command(cmd, "Preparing RedPajama-v2 (multilingual web data)"):
            success_count += 1
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Dataset preparation complete!")
    print(f"Successfully prepared: {success_count}/{len(args.datasets)} datasets")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print('='*60)
    
    # Show next steps
    print("\n📚 Next steps:")
    print("1. Check prepared datasets:")
    print("   ls -la data/*/")
    
    print("\n2. Start training with multilingual data:")
    print("   uv run brain_gpt/training/train_multilingual.py")
    
    print("\n3. Or train with specific datasets:")
    print("   # Korean-focused:")
    print("   uv run brain_gpt/training/train_korean.py")
    print("   # English-focused:")
    print("   uv run brain_gpt/training/train_brain_gpt_3090.py --data-dir data/fineweb")
    
    print("\n4. Monitor GPU usage during training:")
    print("   watch -n 1 nvidia-smi")


if __name__ == "__main__":
    main()