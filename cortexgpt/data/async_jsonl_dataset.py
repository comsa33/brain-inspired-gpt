"""
Asynchronous multiprocessing JSONL dataset for ultra-fast loading
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import multiprocessing as mp
from multiprocessing import Pool, Queue, Process
import queue
import threading
import time
from tqdm import tqdm
import numpy as np


class AsyncJSONLDataset(Dataset):
    """Dataset that uses async multiprocessing for fast data loading and tokenization"""
    
    def __init__(self, data_path: str, tokenizer, block_size: int = 1024, 
                 num_workers: int = None, prefetch_factor: int = 10):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.prefetch_factor = prefetch_factor
        
        # Determine number of workers
        if num_workers is None:
            self.num_workers = min(mp.cpu_count() - 1, 8)
        else:
            self.num_workers = num_workers
            
        print(f"Initializing AsyncJSONLDataset with {self.num_workers} workers...")
        
        # Get file list
        if self.data_path.is_file():
            self.files = [self.data_path]
        else:
            self.files = list(self.data_path.glob("*.jsonl"))
        
        # Quick parallel scan to count lines and prepare chunks
        self.file_chunks = self._prepare_file_chunks()
        self.total_chunks = len(self.file_chunks)
        
        # Start background tokenization
        self._start_async_tokenization()
        
        # Estimate total sequences
        self.estimated_sequences = self.total_chunks * 10  # Rough estimate
        print(f"Total chunks to process: {self.total_chunks}")
        
    def _prepare_file_chunks(self) -> List[Tuple[Path, int, int]]:
        """Prepare file chunks for parallel processing"""
        chunks = []
        chunk_size = 100  # Process 100 lines at a time
        
        print("Scanning files...")
        with Pool(self.num_workers) as pool:
            # Count lines in parallel
            file_line_counts = pool.map(self._count_lines, self.files)
            
            # Create chunks
            for file_path, line_count in zip(self.files, file_line_counts):
                for start_line in range(0, line_count, chunk_size):
                    end_line = min(start_line + chunk_size, line_count)
                    chunks.append((file_path, start_line, end_line))
        
        return chunks
    
    @staticmethod
    def _count_lines(file_path: Path) -> int:
        """Count lines in a file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def _start_async_tokenization(self):
        """Start asynchronous tokenization in background"""
        self.tokenized_queue = mp.Queue(maxsize=self.prefetch_factor * self.num_workers)
        self.chunk_queue = mp.Queue()
        
        # Add all chunks to queue
        for chunk in self.file_chunks:
            self.chunk_queue.put(chunk)
        
        # Add stop signals
        for _ in range(self.num_workers):
            self.chunk_queue.put(None)
        
        # Start worker processes
        self.workers = []
        for i in range(self.num_workers):
            p = Process(
                target=self._tokenize_worker,
                args=(i, self.chunk_queue, self.tokenized_queue, 
                      self.tokenizer.vocab, self.tokenizer.merges,
                      self.tokenizer.special_tokens, self.block_size)
            )
            p.daemon = True
            p.start()
            self.workers.append(p)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_progress)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Collect results asynchronously
        self.samples = []
        self.collection_thread = threading.Thread(target=self._collect_results)
        self.collection_thread.daemon = True
        self.collection_thread.start()
    
    @staticmethod
    def _tokenize_worker(worker_id: int, chunk_queue: Queue, result_queue: Queue,
                        vocab: dict, merges: list, special_tokens: dict, block_size: int):
        """Worker process for tokenization"""
        # Recreate tokenizer in worker process
        from cortexgpt.tokenization.multilingual_tokenizer import MultilingualTokenizer
        tokenizer = MultilingualTokenizer()
        tokenizer.vocab = vocab
        tokenizer.merges = merges
        tokenizer.special_tokens = special_tokens
        tokenizer.reverse_vocab = {v: k for k, v in vocab.items()}
        
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            
            file_path, start_line, end_line = chunk
            
            try:
                # Process chunk
                texts = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i < start_line:
                            continue
                        if i >= end_line:
                            break
                        
                        try:
                            data = json.loads(line)
                            text = data.get('text', '') or data.get('content', '')
                            if text and len(text) > 50:
                                texts.append(text)
                        except:
                            continue
                
                # Tokenize texts
                all_tokens = []
                for text in texts:
                    tokens = tokenizer.encode(text)
                    all_tokens.extend(tokens)
                    all_tokens.append(special_tokens.get('<eos>', 2))
                
                # Create sequences
                sequences = []
                for i in range(0, len(all_tokens) - block_size, block_size):
                    sequences.append(all_tokens[i:i + block_size + 1])
                
                # Send results
                if sequences:
                    result_queue.put((worker_id, sequences))
                    
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                continue
    
    def _monitor_progress(self):
        """Monitor tokenization progress"""
        processed = 0
        start_time = time.time()
        
        with tqdm(total=self.total_chunks, desc="Tokenizing") as pbar:
            while processed < self.total_chunks:
                time.sleep(0.1)
                current_processed = self.total_chunks - self.chunk_queue.qsize() + self.num_workers
                if current_processed > processed:
                    pbar.update(current_processed - processed)
                    processed = current_processed
    
    def _collect_results(self):
        """Collect tokenized results from workers"""
        collected = 0
        
        while collected < self.total_chunks:
            try:
                worker_id, sequences = self.tokenized_queue.get(timeout=1.0)
                self.samples.extend(sequences)
                collected += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Collection error: {e}")
                break
        
        # Shuffle samples
        import random
        random.shuffle(self.samples)
        
        print(f"✅ Tokenization complete! Total sequences: {len(self.samples)}")
    
    def __len__(self):
        # Wait for some samples to be ready
        timeout = 30  # Wait up to 30 seconds
        start_time = time.time()
        
        while len(self.samples) == 0 and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        return max(len(self.samples), self.estimated_sequences)
    
    def __getitem__(self, idx):
        # Wait for samples if necessary
        while idx >= len(self.samples):
            time.sleep(0.01)
            if not self.collection_thread.is_alive():
                # If collection is done but we don't have enough samples, wrap around
                idx = idx % len(self.samples)
                break
        
        sequence = self.samples[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids
    
    def cleanup(self):
        """Clean up worker processes"""
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()


def create_async_dataloaders(
    train_path: str,
    val_path: Optional[str],
    tokenizer,
    block_size: int = 1024,
    batch_size: int = 8,
    num_workers: int = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create async dataloaders with multiprocessing"""
    
    print("🚀 Creating async dataloaders with multiprocessing...")
    
    # Create datasets
    train_dataset = AsyncJSONLDataset(
        train_path, tokenizer, block_size, num_workers
    )
    
    val_dataset = None
    if val_path and Path(val_path).exists():
        val_dataset = AsyncJSONLDataset(
            val_path, tokenizer, block_size, 
            num_workers=max(1, num_workers // 2)  # Use fewer workers for validation
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Dataset handles its own multiprocessing
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader