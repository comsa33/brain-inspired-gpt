"""
Streaming JSONL dataset that processes data on-the-fly without loading everything into memory
"""

import json
import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from typing import Iterator, Optional
import random
from itertools import cycle


class StreamingJSONLDataset(IterableDataset):
    """Streaming dataset that processes JSONL files without loading all data into memory"""
    
    def __init__(self, data_path: str, tokenizer, block_size: int = 1024, 
                 shuffle_buffer_size: int = 10000):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Get file list
        if self.data_path.is_file():
            self.files = [self.data_path]
        else:
            self.files = list(self.data_path.glob("*.jsonl"))
        
        print(f"Streaming from {len(self.files)} file(s)")
        
        # Quick scan to estimate dataset size
        self.estimated_samples = 0
        for file_path in self.files[:1]:  # Just check first file
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, _ in enumerate(f):
                    if i >= 1000:  # Sample first 1000 lines
                        self.estimated_samples = i * len(self.files) * 10  # Rough estimate
                        break
        
        if self.estimated_samples == 0:
            self.estimated_samples = 10000  # Default estimate
        
        print(f"Estimated samples: ~{self.estimated_samples}")
        
    def __iter__(self) -> Iterator[tuple]:
        # Buffer for shuffling
        buffer = []
        
        # Token accumulator for creating sequences
        token_buffer = []
        
        # Cycle through files for continuous iteration
        for file_path in cycle(self.files):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get('text', '') or data.get('content', '')
                        
                        if not text or len(text) < 50:
                            continue
                        
                        # Tokenize text
                        tokens = self.tokenizer.encode(text)
                        token_buffer.extend(tokens)
                        token_buffer.append(self.tokenizer.special_tokens.get('<eos>', 2))
                        
                        # Create sequences from token buffer
                        while len(token_buffer) >= self.block_size + 1:
                            sequence = token_buffer[:self.block_size + 1]
                            token_buffer = token_buffer[self.block_size:]  # No overlap
                            
                            # Add to shuffle buffer
                            buffer.append(sequence)
                            
                            # Yield from buffer when full
                            if len(buffer) >= self.shuffle_buffer_size:
                                # Shuffle and yield
                                random.shuffle(buffer)
                                for seq in buffer:
                                    input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                                    target_ids = torch.tensor(seq[1:], dtype=torch.long)
                                    yield input_ids, target_ids
                                buffer = []
                                
                    except Exception as e:
                        continue
        
        # Yield remaining buffer
        if buffer:
            random.shuffle(buffer)
            for seq in buffer:
                input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                target_ids = torch.tensor(seq[1:], dtype=torch.long)
                yield input_ids, target_ids
                
    def __len__(self):
        # Return estimated length for progress tracking
        return self.estimated_samples


def create_streaming_dataloader(
    data_path: str,
    tokenizer,
    block_size: int = 1024,
    batch_size: int = 8,
    num_workers: int = 0
) -> DataLoader:
    """Create a streaming dataloader"""
    
    dataset = StreamingJSONLDataset(data_path, tokenizer, block_size)
    
    # For IterableDataset, we don't use shuffle parameter
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Drop incomplete batches
    )
    
    return dataloader