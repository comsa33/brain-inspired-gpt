"""
JSONL dataset for demo training when .bin files are not available
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
import random


class JSONLDataset(Dataset):
    """Dataset that loads from JSONL files and tokenizes on the fly"""
    
    def __init__(self, data_path: str, tokenizer, block_size: int = 1024):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.samples = []
        
        # Load all texts
        texts = []
        if self.data_path.is_file():
            # Single file
            files = [self.data_path]
        else:
            # Directory
            files = list(self.data_path.glob("*.jsonl"))
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get('text', '') or data.get('content', '')
                        if text:
                            texts.append(text)
                    except:
                        continue
        
        print(f"Loaded {len(texts)} texts from {len(files)} file(s)")
        
        # Tokenize all texts and create samples
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.special_tokens.get('<eos>', 2))
        
        print(f"Total tokens: {len(all_tokens)}")
        
        # Create overlapping sequences
        stride = block_size // 2
        for i in range(0, len(all_tokens) - block_size + 1, stride):
            self.samples.append(all_tokens[i:i + block_size])
        
        print(f"Created {len(self.samples)} training sequences")
        
        # If too few samples, duplicate them
        if len(self.samples) < 100:
            original_samples = self.samples.copy()
            while len(self.samples) < 100:
                self.samples.extend(original_samples)
            print(f"Duplicated to {len(self.samples)} sequences")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # Input is all tokens except last, target is all tokens except first
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids


def create_jsonl_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    block_size: int = 1024,
    batch_size: int = 8,
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create dataloaders from JSONL files"""
    
    # Create datasets
    train_dataset = JSONLDataset(train_path, tokenizer, block_size)
    val_dataset = JSONLDataset(val_path, tokenizer, block_size) if val_path else None
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = None
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader