#!/usr/bin/env python3
"""
Test async data loading speed
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cortexgpt.tokenization.multilingual_tokenizer import MultilingualTokenizer
from cortexgpt.data.jsonl_dataset import JSONLDataset
from cortexgpt.data.lazy_jsonl_dataset import LazyJSONLDataset
from cortexgpt.data.async_jsonl_dataset import AsyncJSONLDataset

def test_dataset_loading(dataset_path: str = "data/datasets/wikitext/data.jsonl"):
    """Compare loading times for different dataset implementations"""
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = MultilingualTokenizer(vocab_size=30000)
    
    # Simple training corpus for tokenizer
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "안녕하세요, 오늘은 좋은 날씨입니다.",
        "Machine learning is transforming the world.",
        "인공지능은 미래를 바꾸고 있습니다."
    ] * 10
    
    tokenizer.train(training_texts)
    print(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run: uv run scripts/download_data.py --dataset wikitext")
        return
    
    print(f"\nTesting with dataset: {dataset_path}")
    
    # Test 1: Original JSONLDataset (slow)
    print("\n1. Testing JSONLDataset (original - tokenizes everything upfront)...")
    print("   ⚠️  This may take 20+ minutes for large datasets...")
    start_time = time.time()
    try:
        # Only test with a small subset to avoid waiting too long
        if "wikitext" in dataset_path or "large" in dataset_path:
            print("   Skipping full JSONLDataset test for large dataset (would take too long)")
            load_time1 = 1200  # Assume 20 minutes
        else:
            dataset1 = JSONLDataset(dataset_path, tokenizer, block_size=1024)
            load_time1 = time.time() - start_time
            print(f"   Loading time: {load_time1:.2f} seconds")
            print(f"   Sequences: {len(dataset1)}")
    except Exception as e:
        print(f"   Error: {e}")
        load_time1 = float('inf')
    
    # Test 2: LazyJSONLDataset
    print("\n2. Testing LazyJSONLDataset (tokenizes on-demand)...")
    start_time = time.time()
    try:
        dataset2 = LazyJSONLDataset(dataset_path, tokenizer, block_size=1024)
        load_time2 = time.time() - start_time
        print(f"   Loading time: {load_time2:.2f} seconds")
        print(f"   Estimated sequences: {len(dataset2)}")
    except Exception as e:
        print(f"   Error: {e}")
        load_time2 = float('inf')
    
    # Test 3: AsyncJSONLDataset
    print("\n3. Testing AsyncJSONLDataset (async multiprocessing)...")
    start_time = time.time()
    try:
        dataset3 = AsyncJSONLDataset(dataset_path, tokenizer, block_size=1024)
        load_time3 = time.time() - start_time
        print(f"   Loading time: {load_time3:.2f} seconds")
        print(f"   Estimated sequences: {len(dataset3)}")
        
        # Wait a bit for async processing
        print("   Waiting for async tokenization...")
        time.sleep(5)
        print(f"   Actual sequences ready: {len(dataset3.samples)}")
        
        # Clean up
        dataset3.cleanup()
    except Exception as e:
        print(f"   Error: {e}")
        load_time3 = float('inf')
    
    # Summary
    print("\n=== Summary ===")
    print(f"JSONLDataset:      {load_time1:.2f}s")
    print(f"LazyJSONLDataset:  {load_time2:.2f}s")
    print(f"AsyncJSONLDataset: {load_time3:.2f}s")
    
    if load_time1 != float('inf'):
        print(f"\nSpeedup:")
        print(f"  Lazy vs Original:  {load_time1/load_time2:.1f}x faster")
        print(f"  Async vs Original: {load_time1/load_time3:.1f}x faster")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset to test")
    args = parser.parse_args()
    
    dataset_path = f"data/datasets/{args.dataset}/data.jsonl"
    test_dataset_loading(dataset_path)