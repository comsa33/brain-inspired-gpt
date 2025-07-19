#!/usr/bin/env python3
"""
Quick dataset preparation with working datasets
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
    parser = argparse.ArgumentParser(description='Quick dataset preparation with working datasets')
    parser.add_argument('--skip-korean', action='store_true',
                        help='Skip Korean datasets (faster)')
    
    args = parser.parse_args()
    
    print("🧠 Brain-Inspired GPT Quick Dataset Preparation")
    print("Using verified working datasets only")
    
    start_time = time.time()
    success_count = 0
    
    # Prepare Korean datasets (already working)
    if not args.skip_korean:
        cmd = "uv run brain_gpt/training/prepare_korean_hf_datasets.py --max-texts 50000"
        if run_command(cmd, "Preparing Korean datasets (KLUE, KorQuAD)"):
            success_count += 1
    
    # Prepare Wikipedia (already working)
    cmd = "uv run data/openwebtext/prepare_simple.py --datasets wikipedia --max-samples 50000"
    if run_command(cmd, "Preparing Wikipedia (English)"):
        success_count += 1
    
    # Try C4 dataset
    print("\n" + "="*60)
    print("📝 Creating C4 dataset preparation script...")
    print("="*60)
    
    c4_script = '''#!/usr/bin/env python3
"""
C4 Dataset Preparation
Clean Common Crawl dataset
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict
import datasets
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from brain_gpt.core.multilingual_tokenizer import MultilingualBrainTokenizer


def prepare_c4(output_dir: str = 'data/c4', max_samples: int = 50000):
    """Prepare C4 dataset"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading C4 dataset...")
    
    # Load C4 dataset
    dataset = datasets.load_dataset(
        'allenai/c4',
        'en',
        split='train',
        streaming=True,
        trust_remote_code=True
    )
    
    tokenizer = MultilingualBrainTokenizer()
    texts = []
    
    with tqdm(desc="Processing C4", total=max_samples) as pbar:
        for i, doc in enumerate(dataset):
            if i >= max_samples:
                break
                
            text = doc.get('text', '')
            if text and len(text.strip()) > 100:
                texts.append({
                    'text': text,
                    'language': 'en',
                    'source': 'c4'
                })
                pbar.update(1)
    
    print(f"\\nCollected {len(texts)} texts")
    
    # Split and save
    np.random.shuffle(texts)
    split_idx = int(0.95 * len(texts))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Save metadata
    metadata = {
        'source': 'c4',
        'total_texts': len(texts),
        'train_texts': len(train_texts),
        'val_texts': len(val_texts),
        'language': 'en'
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Tokenize and save
    print("Tokenizing train split...")
    train_tokens = []
    for text_data in tqdm(train_texts):
        tokens = tokenizer.encode(text_data['text'], language='en')
        tokens = [t for t in tokens if 0 <= t < 65536]
        train_tokens.extend(tokens)
    
    train_array = np.array(train_tokens, dtype=np.uint16)
    train_array.tofile(output_path / 'c4_train.bin')
    print(f"Saved {len(train_array):,} tokens to c4_train.bin")
    
    print("Tokenizing val split...")
    val_tokens = []
    for text_data in tqdm(val_texts):
        tokens = tokenizer.encode(text_data['text'], language='en')
        tokens = [t for t in tokens if 0 <= t < 65536]
        val_tokens.extend(tokens)
    
    val_array = np.array(val_tokens, dtype=np.uint16)
    val_array.tofile(output_path / 'c4_val.bin')
    print(f"Saved {len(val_array):,} tokens to c4_val.bin")
    
    print(f"\\nDataset saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/c4')
    parser.add_argument('--max-samples', type=int, default=50000)
    args = parser.parse_args()
    
    prepare_c4(args.output_dir, args.max_samples)
'''
    
    # Save C4 script
    c4_script_path = Path("data/openwebtext/prepare_c4.py")
    c4_script_path.write_text(c4_script)
    os.chmod(c4_script_path, 0o755)
    
    # Run C4 preparation
    cmd = "uv run data/openwebtext/prepare_c4.py --max-samples 50000"
    if run_command(cmd, "Preparing C4 dataset (Common Crawl)"):
        success_count += 1
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Quick dataset preparation complete!")
    print(f"Successfully prepared: {success_count} datasets")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print('='*60)
    
    # Show next steps
    print("\n📚 Prepared datasets:")
    print("1. Korean: data/korean_hf/ (KLUE + KorQuAD)")
    print("2. Wikipedia: data/simple/ (English + Korean)")
    print("3. C4: data/c4/ (High-quality web text)")
    
    print("\n🚀 Next steps:")
    print("1. Start multilingual training:")
    print("   uv run brain_gpt/training/train_multilingual.py \\")
    print("     --data-dirs data/korean_hf data/simple data/c4")
    
    print("\n2. Or train with specific focus:")
    print("   # Korean-focused:")
    print("   uv run brain_gpt/training/train_korean.py")
    print("   # English-focused:")
    print("   uv run brain_gpt/training/train_brain_gpt_3090.py --data-dir data/c4")
    
    print("\n3. Test the model:")
    print("   uv run test_multilingual.py")


if __name__ == "__main__":
    main()