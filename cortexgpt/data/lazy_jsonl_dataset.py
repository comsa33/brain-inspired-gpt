"""
Lazy JSONL dataset that tokenizes on-demand for faster startup
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional
import random


class LazyJSONLDataset(Dataset):
    """Dataset that loads JSONL files lazily and tokenizes on-demand"""
    
    def __init__(self, data_path: str, tokenizer, block_size: int = 1024, max_samples: Optional[int] = None):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_samples = max_samples
        
        # Just store text data, don't tokenize yet
        self.texts = []
        
        # Load text data (fast)
        if self.data_path.is_file():
            files = [self.data_path]
        else:
            files = list(self.data_path.glob("*.jsonl"))[:5]  # Limit files for faster loading
        
        print(f"Loading texts from {len(files)} file(s)...")
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if max_samples and len(self.texts) >= max_samples:
                        break
                    try:
                        data = json.loads(line)
                        text = data.get('text', '') or data.get('content', '')
                        if text and len(text) > 100:  # Skip very short texts
                            self.texts.append(text)
                    except:
                        continue
                        
            if max_samples and len(self.texts) >= max_samples:
                break
        
        print(f"Loaded {len(self.texts)} texts (lazy loading enabled)")
        
        # Estimate number of sequences (rough estimate)
        avg_text_len = sum(len(t) for t in self.texts[:100]) / min(100, len(self.texts))
        avg_tokens_per_text = avg_text_len * 0.75  # Rough estimate
        self.estimated_sequences = int(len(self.texts) * avg_tokens_per_text / block_size)
        print(f"Estimated sequences: ~{self.estimated_sequences}")
        
        # Cache for tokenized sequences
        self.cache = {}
        self.cache_size = 1000  # Cache up to 1000 sequences
    
    def __len__(self):
        # Return estimated length for faster initialization
        return max(self.estimated_sequences, 100)
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Map idx to text and position
        text_idx = idx % len(self.texts)
        
        # Tokenize the selected text
        text = self.texts[text_idx]
        tokens = self.tokenizer.encode(text)
        
        # If tokens too short, combine with next text
        while len(tokens) < self.block_size + 1 and text_idx + 1 < len(self.texts):
            text_idx += 1
            next_text = self.texts[text_idx]
            next_tokens = self.tokenizer.encode(next_text)
            tokens.extend([self.tokenizer.special_tokens.get('<eos>', 2)])
            tokens.extend(next_tokens)
        
        # Extract a random window from tokens
        if len(tokens) > self.block_size + 1:
            start_idx = random.randint(0, len(tokens) - self.block_size - 1)
            tokens = tokens[start_idx:start_idx + self.block_size + 1]
        else:
            # Pad if necessary
            while len(tokens) < self.block_size + 1:
                tokens.append(self.tokenizer.special_tokens.get('<pad>', 0))
        
        # Create input/target pairs
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[idx] = (input_ids, target_ids)
        
        return input_ids, target_ids