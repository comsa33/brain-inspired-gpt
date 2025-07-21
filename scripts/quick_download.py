#!/usr/bin/env python3
"""
Quick download script for getting started with real datasets
Downloads pre-processed subsets for faster training startup
"""

import os
import sys
import subprocess
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Quick dataset download for CortexGPT")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       choices=["wikitext", "english_large", "korean_large"],
                       help="Dataset to download")
    
    args = parser.parse_args()
    
    print(f"📥 Quick downloading {args.dataset}...")
    
    # For wikitext, download from Hugging Face but limit size
    if args.dataset == "wikitext":
        print("Downloading WikiText-103 subset...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets library...")
            subprocess.run(["uv", "pip", "install", "datasets"], check=True)
            from datasets import load_dataset
        
        # Download only a subset for faster loading
        dataset = load_dataset(
            "wikitext", 
            "wikitext-103-raw-v1",
            split="train[:5000]"  # Only first 5000 samples
        )
        
        # Save to JSONL
        import json
        from pathlib import Path
        
        output_dir = Path("data/datasets/wikitext")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "data.jsonl", 'w', encoding='utf-8') as f:
            for item in dataset:
                text = item['text'].strip()
                if len(text) > 100:  # Skip very short texts
                    json.dump({"text": text}, f, ensure_ascii=False)
                    f.write('\n')
        
        print(f"✅ Downloaded {args.dataset} (5000 samples)")
    
    else:
        # Use the regular download script
        subprocess.run([
            "uv", "run", "scripts/download_data.py",
            "--dataset", args.dataset
        ])
    
    print("\n💡 To train:")
    print(f"   uv run scripts/train_cortexgpt.py --dataset {args.dataset} --epochs 10")


if __name__ == "__main__":
    main()