#!/usr/bin/env python3
"""
Simple training script for Brain-Inspired GPT
Minimal configuration for testing and demos
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer


class SimpleDataset(Dataset):
    """Simple dataset for training"""
    def __init__(self, data_path, block_size=1024):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def train_simple():
    """Simple training loop"""
    print("🧠 Brain-Inspired GPT - Simple Training")
    print("=" * 60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Small model config for RTX 3090
    config = BrainGPTConfig()
    config.n_layer = 6      # Very small
    config.n_head = 8
    config.n_embd = 512
    config.block_size = 512
    config.n_cortical_columns = 16
    config.column_size = 32  # 16 * 32 = 512
    config.gradient_checkpointing = False  # Disable for simplicity
    
    print(f"Device: {device}")
    print(f"Model config: {config.n_layer} layers, {config.n_embd} hidden size")
    
    # Create model
    model = BrainGPT(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M")
    print(f"Effective parameters: {total_params*(1-config.sparsity_base)/1e6:.1f}M")
    
    # Create dataset
    data_path = Path("data/openwebtext/train.bin")
    if not data_path.exists():
        print("❌ Training data not found. Run:")
        print("   uv run brain_gpt/training/create_sample_data.py")
        return
        
    dataset = SimpleDataset(data_path, block_size=config.block_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training loop
    model.train()
    print("\n🚀 Starting training...")
    
    losses = []
    start_time = time.time()
    
    for step, (x, y) in enumerate(dataloader):
        if step >= 100:  # Train for 100 steps
            break
            
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
                
            # Only compute loss on the output positions
            if logits.size(1) == 1:
                # Model outputs only last position
                y = y[:, -1:]
                
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}: loss = {loss.item():.4f}, time = {elapsed:.1f}s")
            
    # Summary
    print("\n" + "="*60)
    print("Training Summary:")
    print(f"  Steps trained: {len(losses)}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Avg loss: {np.mean(losses):.4f}")
    print(f"  Time: {time.time() - start_time:.1f}s")
    
    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': len(losses),
        'loss': losses[-1],
    }
    
    save_path = Path("checkpoints/brain_gpt_simple.pt")
    save_path.parent.mkdir(exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"\n✅ Model saved to {save_path}")
    
    # Quick generation test
    print("\n🎯 Testing generation...")
    model.eval()
    
    # Create a simple prompt
    tokenizer = MultilingualBrainTokenizer()
    prompt = "The future of AI"
    tokens = tokenizer.encode(prompt)[:10]
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.generate(x, max_new_tokens=20, temperature=0.8)
        
    print(f"Prompt: {prompt}")
    print(f"Generated {len(output[0]) - len(tokens)} new tokens")
    print("\n✅ Training complete!")
    

if __name__ == "__main__":
    train_simple()