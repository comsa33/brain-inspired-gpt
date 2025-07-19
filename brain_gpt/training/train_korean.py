#!/usr/bin/env python3
"""
Training script for Brain-Inspired GPT with Korean dataset
Uses the prepared Korean HuggingFace dataset
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer


class KoreanDataset(Dataset):
    """Korean dataset for training"""
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


def train_korean():
    """Train Brain-Inspired GPT on Korean dataset"""
    print("🧠 Brain-Inspired GPT - Korean Training")
    print("==" * 30)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Small model config for RTX 3090
    config = BrainGPTConfig()
    config.n_layer = 8      # Small but deeper than train_simple
    config.n_head = 8
    config.n_embd = 512
    config.block_size = 1024  # Larger context for Korean
    config.n_cortical_columns = 16
    config.column_size = 32  # 16 * 32 = 512
    config.vocab_size = 65536  # Support full Korean vocab
    config.gradient_checkpointing = True  # Enable for memory efficiency
    
    print(f"Device: {device}")
    print(f"Model config: {config.n_layer} layers, {config.n_embd} hidden size")
    print(f"Block size: {config.block_size}")
    print(f"Vocab size: {config.vocab_size}")
    
    # Create model
    model = BrainGPT(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M")
    print(f"Effective parameters: {total_params*(1-config.sparsity_base)/1e6:.1f}M")
    
    # Create datasets
    train_path = Path("data/korean_hf/korean_hf_train.bin")
    val_path = Path("data/korean_hf/korean_hf_val.bin")
    
    if not train_path.exists():
        print("❌ Korean training data not found. Run:")
        print("   uv run brain_gpt/training/prepare_korean_hf_datasets.py")
        return
        
    print(f"\n📊 Dataset info:")
    train_size = train_path.stat().st_size // 2  # uint16 = 2 bytes
    val_size = val_path.stat().st_size // 2
    print(f"   Training tokens: {train_size:,}")
    print(f"   Validation tokens: {val_size:,}")
    
    train_dataset = KoreanDataset(train_path, block_size=config.block_size)
    val_dataset = KoreanDataset(val_path, block_size=config.block_size)
    
    # Smaller batch size for memory efficiency
    batch_size = 2 if config.gradient_checkpointing else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer with Korean-specific learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=1e-5
    )
    
    # Training settings
    num_epochs = 1
    eval_interval = 100
    save_interval = 500
    max_steps = 1000  # Limit for demo
    
    # Training loop
    model.train()
    print("\n🚀 Starting Korean training...")
    
    train_losses = []
    val_losses = []
    step = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n📖 Epoch {epoch + 1}/{num_epochs}")
        
        epoch_losses = []
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (x, y) in enumerate(pbar):
            if step >= max_steps:
                break
                
            x, y = x.to(device), y.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                    
                # Handle different output shapes
                if logits.size(1) == 1:
                    # Model outputs only last position
                    y_target = y[:, -1:]
                else:
                    # Model outputs all positions
                    y_target = y
                    
                # Compute loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y_target.view(-1)
                )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track loss
            loss_val = loss.item()
            train_losses.append(loss_val)
            epoch_losses.append(loss_val)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_val:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Evaluation
            if step % eval_interval == 0 and step > 0:
                model.eval()
                val_loss = evaluate(model, val_loader, device)
                val_losses.append(val_loss)
                model.train()
                
                print(f"\nStep {step}: train_loss={loss_val:.4f}, val_loss={val_loss:.4f}")
            
            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                save_checkpoint(model, optimizer, config, step, loss_val)
            
            step += 1
            
        # Epoch summary
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f}")
    
    # Training summary
    elapsed = time.time() - start_time
    print("\n" + "=="*30)
    print("🎓 Training Summary:")
    print(f"  Total steps: {step}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"  Final val loss: {val_losses[-1]:.4f}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Tokens/second: {step * batch_size * config.block_size / elapsed:.0f}")
    
    # Save final model
    final_path = save_checkpoint(model, optimizer, config, step, train_losses[-1], final=True)
    
    # Test generation with Korean
    print("\n🎯 Testing Korean generation...")
    test_korean_generation(model, config, device)
    
    print("\n✅ Korean training complete!")
    print(f"Model saved to: {final_path}")


def evaluate(model, val_loader, device, max_batches=50):
    """Evaluate model on validation set"""
    losses = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
                
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                    
                # Handle different output shapes
                if logits.size(1) == 1:
                    # Model outputs only last position
                    y_target = y[:, -1:]
                else:
                    # Model outputs all positions
                    y_target = y
                    
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y_target.view(-1)
                )
                
            losses.append(loss.item())
    
    return np.mean(losses)


def save_checkpoint(model, optimizer, config, step, loss, final=False):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': step,
        'loss': loss,
    }
    
    if final:
        save_path = Path("checkpoints/brain_gpt_korean_final.pt")
    else:
        save_path = Path(f"checkpoints/brain_gpt_korean_step{step}.pt")
        
    save_path.parent.mkdir(exist_ok=True)
    torch.save(checkpoint, save_path)
    
    return save_path


def test_korean_generation(model, config, device):
    """Test generation with Korean prompts"""
    model.eval()
    tokenizer = MultilingualBrainTokenizer()
    
    # Korean test prompts
    prompts = [
        "인공지능의 미래는",
        "한국의 전통 문화",
        "서울은 대한민국의",
        "기계 학습 기술이",
    ]
    
    print("\n🇰🇷 Korean Generation Examples:")
    print("-" * 60)
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, language='ko')[:50]  # Limit prompt length
        x = torch.tensor(tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model.generate(
                x, 
                max_new_tokens=30,
                temperature=0.8,
                top_k=50
            )
        
        generated_text = tokenizer.decode(output[0].cpu().numpy())
        print(f"\n📝 Prompt: {prompt}")
        print(f"📖 Generated: {generated_text}")
        print("-" * 60)


if __name__ == "__main__":
    train_korean()