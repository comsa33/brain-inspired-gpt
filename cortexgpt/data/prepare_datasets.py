#!/usr/bin/env python3
"""
Prepare downloaded datasets for CortexGPT training
Handles tokenization, chunking, and format conversion
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import random

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cortexgpt.tokenization.multilingual_tokenizer import MultilingualTokenizer

class DatasetPreparer:
    """Prepares raw text datasets for training"""
    
    def __init__(
        self,
        tokenizer: Optional[MultilingualTokenizer] = None,
        block_size: int = 1024,
        num_workers: int = None
    ):
        self.tokenizer = tokenizer or self._create_tokenizer()
        self.block_size = block_size
        self.num_workers = num_workers or mp.cpu_count()
        
    def _create_tokenizer(self) -> MultilingualTokenizer:
        """Create and train tokenizer on sample data"""
        print("Creating tokenizer...")
        tokenizer = MultilingualTokenizer(vocab_size=50000)
        
        # Train on sample texts
        sample_texts = []
        
        # Load sample texts from various sources
        data_paths = [
            "data/datasets/klue/data.jsonl",
            "data/datasets/korean_wiki/data.jsonl", 
            "data/datasets/wikipedia_en/data.jsonl"
        ]
        
        for path in data_paths:
            if Path(path).exists():
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 1000:  # Sample 1000 lines from each
                            break
                        data = json.loads(line)
                        sample_texts.append(data['text'])
        
        if sample_texts:
            print(f"Training tokenizer on {len(sample_texts)} samples...")
            tokenizer.learn_bpe(sample_texts, verbose=True)
        else:
            print("Warning: No sample data found, using default tokenizer")
            
        return tokenizer
    
    def prepare_jsonl_dataset(
        self,
        input_path: str,
        output_dir: str,
        split_ratio: float = 0.95
    ) -> Dict[str, int]:
        """Prepare a JSONL dataset for training"""
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📚 Processing {input_path}...")
        
        # First pass: count lines
        total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
        
        # Calculate split
        train_size = int(total_lines * split_ratio)
        
        # Prepare output files
        train_file = output_dir / "train.bin"
        val_file = output_dir / "val.bin"
        
        train_tokens = []
        val_tokens = []
        
        # Process file
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=total_lines, desc="Tokenizing")):
                try:
                    data = json.loads(line)
                    text = data.get('text', '') or data.get('content', '')
                    
                    if not text:
                        continue
                    
                    # Tokenize
                    tokens = self.tokenizer.encode(text)
                    
                    # Add to appropriate split
                    if i < train_size:
                        train_tokens.extend(tokens)
                    else:
                        val_tokens.extend(tokens)
                        
                    # Save periodically to avoid memory issues
                    if len(train_tokens) > 100_000_000:  # 100M tokens
                        self._save_tokens(train_tokens, train_file, append=True)
                        train_tokens = []
                        
                    if len(val_tokens) > 10_000_000:  # 10M tokens
                        self._save_tokens(val_tokens, val_file, append=True)
                        val_tokens = []
                        
                except Exception as e:
                    print(f"Error processing line {i}: {e}")
                    continue
        
        # Save remaining tokens
        if train_tokens:
            self._save_tokens(train_tokens, train_file, append=True)
        if val_tokens:
            self._save_tokens(val_tokens, val_file, append=True)
        
        # Get final sizes
        train_size = os.path.getsize(train_file) // 2  # uint16 = 2 bytes
        val_size = os.path.getsize(val_file) // 2
        
        stats = {
            "train_tokens": train_size,
            "val_tokens": val_size,
            "total_tokens": train_size + val_size,
            "vocab_size": len(self.tokenizer.vocab)
        }
        
        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Prepared dataset:")
        print(f"   Train: {train_size:,} tokens")
        print(f"   Val: {val_size:,} tokens")
        print(f"   Vocab: {stats['vocab_size']:,} tokens")
        
        return stats
    
    def _save_tokens(self, tokens: List[int], output_path: Path, append: bool = False) -> None:
        """Save tokens to binary file"""
        mode = 'ab' if append else 'wb'
        tokens_array = np.array(tokens, dtype=np.uint16)
        
        with open(output_path, mode) as f:
            tokens_array.tofile(f)
    
    def prepare_all_datasets(self, base_dir: str = "data/datasets") -> None:
        """Prepare all downloaded datasets"""
        base_dir = Path(base_dir)
        
        # Find all JSONL files
        jsonl_files = list(base_dir.glob("*/data.jsonl"))
        
        if not jsonl_files:
            print("❌ No datasets found. Run download_datasets.py first!")
            return
        
        print(f"\n🔧 Found {len(jsonl_files)} datasets to prepare")
        
        # Prepare each dataset
        for jsonl_path in jsonl_files:
            dataset_name = jsonl_path.parent.name
            output_dir = base_dir / dataset_name / "prepared"
            
            if output_dir.exists():
                print(f"⏭️  Skipping {dataset_name} (already prepared)")
                continue
                
            self.prepare_jsonl_dataset(jsonl_path, output_dir)
        
        print("\n✅ All datasets prepared!")
        
    def create_combined_dataset(
        self,
        dataset_dirs: List[str],
        output_dir: str,
        language_ratios: Dict[str, float] = {"ko": 0.5, "en": 0.5}
    ) -> None:
        """Create a combined dataset from multiple sources with language balancing"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔀 Creating combined dataset with ratios: {language_ratios}")
        
        # Collect all prepared datasets
        all_datasets = []
        for dataset_dir in dataset_dirs:
            dataset_path = Path(dataset_dir)
            metadata_path = dataset_path / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                all_datasets.append({
                    "path": dataset_path,
                    "metadata": metadata,
                    "train_file": dataset_path / "train.bin",
                    "val_file": dataset_path / "val.bin"
                })
        
        if not all_datasets:
            print("❌ No prepared datasets found!")
            return
        
        # Combine datasets
        # This is a simplified version - in practice you'd want to:
        # 1. Sample from each dataset according to language ratios
        # 2. Shuffle properly
        # 3. Handle very large datasets efficiently
        
        print(f"📊 Combining {len(all_datasets)} datasets...")
        
        # For now, just create symlinks to use multiple datasets
        combined_config = {
            "datasets": [str(d["path"]) for d in all_datasets],
            "language_ratios": language_ratios,
            "total_tokens": sum(d["metadata"]["total_tokens"] for d in all_datasets)
        }
        
        with open(output_dir / "combined_config.json", 'w') as f:
            json.dump(combined_config, f, indent=2)
        
        print(f"✅ Combined dataset config created")
        print(f"   Total tokens: {combined_config['total_tokens']:,}")


def main():
    """Prepare datasets for training"""
    preparer = DatasetPreparer()
    
    # Prepare all downloaded datasets
    preparer.prepare_all_datasets()
    
    # Create a combined dataset
    dataset_dirs = [
        "data/datasets/klue/prepared",
        "data/datasets/korean_wiki/prepared",
        "data/datasets/wikipedia_en/prepared"
    ]
    
    preparer.create_combined_dataset(
        dataset_dirs,
        "data/datasets/combined",
        language_ratios={"ko": 0.4, "en": 0.6}
    )


if __name__ == "__main__":
    main()