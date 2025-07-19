# Alternative script using RedPajama-1T dataset (1 trillion tokens)
# RedPajama is a more modern and larger dataset compared to OpenWebText

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
    print("Loading RedPajama-1T dataset...")
    print("Note: This is a 1 trillion token dataset, much larger than OpenWebText")
    
    # RedPajama-1T includes multiple data sources
    # You can specify which subset to use:
    # - 'common_crawl' (878B tokens)
    # - 'c4' (175B tokens)
    # - 'github' (59B tokens)
    # - 'wikipedia' (24B tokens)
    # - 'book' (26B tokens)
    # - 'arxiv' (28B tokens)
    # - 'stackexchange' (20B tokens)
    
    # Load a specific subset (e.g., common_crawl)
    dataset = load_dataset(
        "togethercomputer/RedPajama-Data-1T", 
        "common_crawl",
        streaming=True  # Use streaming for large dataset
    )
    
    # For demonstration, let's process a smaller sample
    # In production, you would process the full dataset
    sample_size = 100000  # Process first 100k examples
    
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    # Process the dataset
    processed_data = []
    for i, example in enumerate(tqdm(dataset['train'], desc="Processing samples")):
        if i >= sample_size:
            break
        processed = process(example)
        processed_data.append(processed)
    
    # Save to binary file
    total_len = sum(item['len'] for item in processed_data)
    filename = os.path.join(os.path.dirname(__file__), 'redpajama_sample.bin')
    dtype = np.uint16
    
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_len,))
    
    idx = 0
    for item in tqdm(processed_data, desc=f'Writing {filename}'):
        arr[idx : idx + item['len']] = item['ids']
        idx += item['len']
    
    arr.flush()
    print(f"Saved {total_len} tokens to {filename}")