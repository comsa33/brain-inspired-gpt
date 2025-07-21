"""
Simple dataset class for CortexGPT training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class TokenizedDataset(Dataset):
    """
    Simple tokenized dataset for language modeling.
    """
    
    def __init__(self, data_path: str, block_size: int = 1024):
        self.block_size = block_size
        
        # Load tokenized data
        if os.path.exists(data_path):
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        else:
            # Create dummy data for testing
            print(f"Warning: {data_path} not found. Using dummy data.")
            self.data = np.random.randint(0, 50000, size=(100000,), dtype=np.uint16)
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a chunk of data
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        # Return as dictionary for compatibility with trainer
        return {
            'input_ids': x,
            'labels': y
        }