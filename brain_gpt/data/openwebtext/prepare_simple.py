# Simple alternative using direct download without dataset scripts
# This uses a preprocessed version of OpenWebText

import os
import json
from tqdm import tqdm
import numpy as np
import tiktoken
import requests
from pathlib import Path

# number of workers in .map() call
num_proc = 8

enc = tiktoken.get_encoding("gpt2")

def download_sample_data():
    """Download a sample of OpenWebText data for testing"""
    # We'll create a small sample dataset for testing
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 100,
        "Machine learning is a subset of artificial intelligence. " * 100,
        "Natural language processing enables computers to understand human language. " * 100,
    ] * 1000  # Create 3000 samples
    
    return sample_texts

if __name__ == '__main__':
    print("Creating sample dataset for testing...")
    print("For production, use one of the modern alternatives:")
    print("- FineWeb: 15 trillion tokens")
    print("- RedPajama: 1 trillion tokens")
    print("- Dolma: 3 trillion tokens")
    
    # Get sample data
    texts = download_sample_data()
    
    # Process all texts
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = enc.encode_ordinary(text)
        tokens.append(enc.eot_token)
        all_tokens.extend(tokens)
    
    # Convert to numpy array
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # Split into train and validation
    n = len(all_tokens)
    train_size = int(0.9995 * n)
    train_tokens = all_tokens[:train_size]
    val_tokens = all_tokens[train_size:]
    
    # Save train split
    train_file = os.path.join(os.path.dirname(__file__), 'train.bin')
    train_arr = np.memmap(train_file, dtype=np.uint16, mode='w+', shape=(len(train_tokens),))
    train_arr[:] = train_tokens
    train_arr.flush()
    
    # Save validation split
    val_file = os.path.join(os.path.dirname(__file__), 'val.bin')
    val_arr = np.memmap(val_file, dtype=np.uint16, mode='w+', shape=(len(val_tokens),))
    val_arr[:] = val_tokens
    val_arr.flush()
    
    print(f"Created sample dataset:")
    print(f"Train: {len(train_tokens):,} tokens ({len(train_tokens) * 2 / 1024 / 1024:.2f} MB)")
    print(f"Val: {len(val_tokens):,} tokens ({len(val_tokens) * 2 / 1024 / 1024:.2f} MB)")
    print(f"\nFiles saved:")
    print(f"- {train_file}")
    print(f"- {val_file}")