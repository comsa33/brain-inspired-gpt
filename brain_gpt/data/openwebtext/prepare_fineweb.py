# Alternative script using FineWeb-Edu dataset
# FineWeb-Edu is a high-quality educational subset of web data

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
num_proc = 8

# number of workers in load_dataset() call
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Loading FineWeb-Edu dataset...")
    print("This dataset contains 1.3 trillion tokens of educational content")
    
    # FineWeb-Edu is specifically filtered for educational quality
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True  # Use streaming for large dataset
    )
    
    # For demonstration, let's process a smaller sample
    sample_size = 100000  # Process first 100k examples
    
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    # Process the dataset
    processed_data = []
    for i, example in enumerate(tqdm(dataset, desc="Processing samples")):
        if i >= sample_size:
            break
        processed = process(example)
        processed_data.append(processed)
    
    # Create train/val split
    val_size = int(0.0005 * len(processed_data))
    train_data = processed_data[:-val_size]
    val_data = processed_data[-val_size:]
    
    # Save train split
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        total_len = sum(item['len'] for item in split_data)
        filename = os.path.join(os.path.dirname(__file__), f'fineweb_{split_name}.bin')
        dtype = np.uint16
        
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_len,))
        
        idx = 0
        for item in tqdm(split_data, desc=f'Writing {filename}'):
            arr[idx : idx + item['len']] = item['ids']
            idx += item['len']
        
        arr.flush()
        print(f"Saved {total_len} tokens to {filename}")