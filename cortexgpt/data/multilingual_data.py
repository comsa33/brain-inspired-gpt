"""
Multilingual data handling for Brain-Inspired GPT with streaming support.
"""

import os
from typing import Iterator, Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from pathlib import Path
import random


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for multilingual training that loads data on-the-fly.
    """
    
    def __init__(
        self,
        data_dir: str,
        block_size: int = 1024,
        stride: int = 512,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize streaming dataset.
        
        Args:
            data_dir: Directory containing tokenized .bin files
            block_size: Size of each text block
            stride: Stride for sliding window
            shuffle: Whether to shuffle files
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.stride = stride
        self.shuffle = shuffle
        self.seed = seed
        
        # Find all .bin files
        self.files = list(self.data_dir.glob("*.bin"))
        if not self.files:
            raise ValueError(f"No .bin files found in {data_dir}")
        
        print(f"Found {len(self.files)} data files")
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Yield data samples.
        """
        # Get worker info for multi-process data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process
            file_list = self.files
        else:
            # Multiple workers - split files
            per_worker = len(self.files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.files)
            file_list = self.files[start:end]
        
        # Shuffle files if needed
        if self.shuffle:
            rng = random.Random(self.seed + (worker_info.id if worker_info else 0))
            file_list = list(file_list)
            rng.shuffle(file_list)
        
        # Process each file
        for file_path in file_list:
            try:
                # Load data
                data = np.memmap(file_path, dtype=np.uint16, mode='r')
                
                # Create sliding windows
                for i in range(0, len(data) - self.block_size - 1, self.stride):
                    # Extract block
                    block = data[i:i + self.block_size + 1]
                    
                    # Convert to tensors
                    x = torch.from_numpy(block[:-1].astype(np.int64))
                    y = torch.from_numpy(block[1:].astype(np.int64))
                    
                    yield x, y
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue


class TokenizedDataset(Dataset):
    """
    Simple tokenized dataset for validation/testing.
    """
    
    def __init__(self, data_path: str, block_size: int = 1024):
        """
        Initialize tokenized dataset.
        
        Args:
            data_path: Path to tokenized .bin file
            block_size: Size of each text block
        """
        self.data_path = Path(data_path)
        self.block_size = block_size
        
        # Load data
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.n_samples = (len(self.data) - 1) // block_size
        
        print(f"Loaded {len(self.data):,} tokens ({self.n_samples:,} samples)")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        """
        start = idx * self.block_size
        end = start + self.block_size + 1
        
        block = self.data[start:end]
        x = torch.from_numpy(block[:-1].astype(np.int64))
        y = torch.from_numpy(block[1:].astype(np.int64))
        
        return x, y


def create_dataloaders(
    train_dir: str,
    val_path: Optional[str] = None,
    block_size: int = 1024,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for training and validation.
    
    Args:
        train_dir: Directory containing training data files
        val_path: Path to validation data file (optional)
        block_size: Size of each text block
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle training data
        pin_memory: Whether to pin memory for GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create training dataset
    train_dataset = StreamingDataset(
        data_dir=train_dir,
        block_size=block_size,
        shuffle=shuffle
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    # Create validation dataset if path provided
    val_loader = None
    if val_path and os.path.exists(val_path):
        val_dataset = TokenizedDataset(
            data_path=val_path,
            block_size=block_size
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader


def create_sample_data(output_dir: str, num_samples: int = 1000, vocab_size: int = 50000):
    """
    Create sample tokenized data for testing.
    
    Args:
        output_dir: Directory to save sample data
        num_samples: Number of samples to generate
        vocab_size: Vocabulary size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random token sequences
    for i in range(3):  # Create 3 files
        data = np.random.randint(0, vocab_size, size=num_samples, dtype=np.uint16)
        
        output_path = os.path.join(output_dir, f"sample_{i}.bin")
        data.tofile(output_path)
        
        print(f"Created {output_path} with {num_samples} tokens")


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data("data/sample_multilingual", num_samples=10000)
    
    # Test the dataloader
    train_loader, _ = create_dataloaders(
        train_dir="data/sample_multilingual",
        batch_size=4,
        block_size=256,
        num_workers=0
    )
    
    # Get a batch
    for i, (x, y) in enumerate(train_loader):
        print(f"Batch {i}: x.shape={x.shape}, y.shape={y.shape}")
        if i >= 2:
            break