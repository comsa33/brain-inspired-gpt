#!/usr/bin/env python3
"""
Brain-Inspired GPT Training Script optimized for RTX 3090
Smaller model configuration to fit in 24GB VRAM
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
import argparse
from dataclasses import dataclass
import wandb
from torch.amp import GradScaler, autocast
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer


@dataclass
class TrainingConfig:
    """Training configuration optimized for RTX 3090"""
    # Model
    n_layer: int = 12  # Reduced from 48
    n_head: int = 16   # Reduced from 32
    n_embd: int = 1024 # Reduced from 2048
    block_size: int = 1024
    vocab_size: int = 65536
    
    # Brain-inspired parameters
    n_cortical_columns: int = 32
    column_size: int = 32  # 32 * 32 = 1024
    n_dendrites: int = 4
    dendritic_input_size: int = 512
    top_k_dendrites: int = 2
    
    # Training
    batch_size: int = 2  # Very small for memory efficiency
    gradient_accumulation_steps: int = 8  # Effective batch size = 16
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 100
    max_steps: int = 5000
    eval_interval: int = 100
    save_interval: int = 500
    
    # Optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    compile_model: bool = False  # Disable compilation for now
    
    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "brain-gpt"
    wandb_run_name: str = "brain-gpt-3090-optimized"


class OpenWebTextDataset(Dataset):
    """Dataset for OpenWebText or Korean data"""
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


class BrainGPTTrainer:
    """Trainer for Brain-Inspired GPT optimized for RTX 3090"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        
        # Initialize model
        self.model_config = BrainGPTConfig()
        # Override defaults with our optimized values
        self.model_config.n_layer = config.n_layer
        self.model_config.n_head = config.n_head
        self.model_config.n_embd = config.n_embd
        self.model_config.block_size = config.block_size
        self.model_config.vocab_size = config.vocab_size
        self.model_config.n_cortical_columns = config.n_cortical_columns
        self.model_config.column_size = config.column_size
        self.model_config.gradient_checkpointing = config.gradient_checkpointing
        
        print("Initializing Brain-Inspired GPT model...")
        self.model = BrainGPT(self.model_config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {total_params/1e6:.1f}M parameters")
        print(f"Effective parameters (with sparsity): {total_params*(1-self.model_config.sparsity_base)/1e6:.1f}M")
        
        # Memory usage estimate
        param_memory = total_params * 4 / 1e9  # 4 bytes per param
        optimizer_memory = param_memory * 2  # Adam uses 2x param memory
        activation_memory = config.batch_size * config.block_size * config.n_embd * config.n_layer * 4 / 1e9
        total_memory = param_memory + optimizer_memory + activation_memory
        print(f"Estimated memory usage: {total_memory:.2f}GB")
        
        # Initialize datasets
        self.train_dataset, self.val_dataset = self._load_datasets()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler('cuda') if config.mixed_precision else None
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config)
            )
            wandb.watch(self.model, log_freq=100)
            
        self.step = 0
        self.best_val_loss = float('inf')
        
    def _load_datasets(self):
        """Load training and validation datasets"""
        print("Creating datasets...")
        
        # Check for Korean data first
        korean_train = Path(self.config.data_dir) / "korean_hf" / "korean_hf_train.bin"
        korean_val = Path(self.config.data_dir) / "korean_hf" / "korean_hf_val.bin"
        
        if korean_train.exists() and korean_val.exists():
            print("Using Korean dataset")
            train_path = korean_train
            val_path = korean_val
        else:
            # Fall back to OpenWebText
            train_path = Path(self.config.data_dir) / "openwebtext" / "train.bin"
            val_path = Path(self.config.data_dir) / "openwebtext" / "val.bin"
            
            if not train_path.exists():
                raise FileNotFoundError(
                    f"No training data found. Please run:\n"
                    f"  uv run brain_gpt/training/prepare_korean_hf_datasets.py\n"
                    f"  or\n"
                    f"  uv run brain_gpt/training/create_sample_data.py"
                )
        
        train_dataset = OpenWebTextDataset(train_path, self.config.block_size)
        val_dataset = OpenWebTextDataset(val_path, self.config.block_size)
        
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def _get_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}...")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.model.train()
        losses = []
        start_time = time.time()
        
        data_iter = iter(train_loader)
        
        for step in range(self.config.max_steps):
            self.step = step
            
            # Accumulate gradients
            total_loss = 0
            self.optimizer.zero_grad()
            
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                if self.config.mixed_precision:
                    with autocast('cuda'):
                        logits = self.model(x)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        
                        # Handle different output shapes
                        if logits.size(1) == 1:
                            y_target = y[:, -1:]
                        else:
                            y_target = y
                            
                        loss = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y_target.view(-1)
                        )
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    logits = self.model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    
                    if logits.size(1) == 1:
                        y_target = y[:, -1:]
                    else:
                        y_target = y
                        
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y_target.view(-1)
                    )
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
            
            # Optimizer step
            if self.config.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Record loss
            losses.append(total_loss)
            
            # Logging
            if step % 10 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (step + 1) * self.config.batch_size * self.config.block_size * self.config.gradient_accumulation_steps / elapsed
                lr = self.scheduler.get_last_lr()[0]
                
                print(f"Step {step}: loss={total_loss:.4f}, lr={lr:.2e}, tokens/s={tokens_per_sec:.0f}")
                
                if self.config.use_wandb:
                    wandb.log({
                        'train/loss': total_loss,
                        'train/lr': lr,
                        'train/tokens_per_sec': tokens_per_sec,
                        'train/step': step
                    })
            
            # Evaluation
            if step % self.config.eval_interval == 0 and step > 0:
                val_loss = self.evaluate()
                print(f"Validation loss: {val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(best=True)
                
                self.model.train()
            
            # Save checkpoint
            if step % self.config.save_interval == 0 and step > 0:
                self.save_checkpoint()
        
        # Final save
        self.save_checkpoint(final=True)
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        losses = []
        for i, (x, y) in enumerate(val_loader):
            if i >= 50:  # Evaluate on subset
                break
                
            x, y = x.to(self.device), y.to(self.device)
            
            if self.config.mixed_precision:
                with autocast('cuda'):
                    logits = self.model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    
                    if logits.size(1) == 1:
                        y_target = y[:, -1:]
                    else:
                        y_target = y
                        
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y_target.view(-1)
                    )
            else:
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                if logits.size(1) == 1:
                    y_target = y[:, -1:]
                else:
                    y_target = y
                    
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y_target.view(-1)
                )
            
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        
        if self.config.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/step': self.step
            })
        
        return avg_loss
    
    def save_checkpoint(self, best=False, final=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model_config,
            'step': self.step,
            'best_val_loss': self.best_val_loss,
        }
        
        if final:
            path = Path(self.config.checkpoint_dir) / "brain_gpt_3090_final.pt"
        elif best:
            path = Path(self.config.checkpoint_dir) / "brain_gpt_3090_best.pt"
        else:
            path = Path(self.config.checkpoint_dir) / f"brain_gpt_3090_step{self.step}.pt"
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Also test generation
        if final or best:
            self.test_generation()
    
    @torch.no_grad()
    def test_generation(self):
        """Test model generation"""
        self.model.eval()
        
        tokenizer = MultilingualBrainTokenizer()
        prompts = [
            "The future of artificial intelligence",
            "인공지능의 미래는",  # Korean
            "Once upon a time",
        ]
        
        print("\n" + "="*60)
        print("Generation Examples:")
        
        for prompt in prompts:
            # Try to detect language
            if any(ord(c) > 127 for c in prompt):
                lang = 'ko'
            else:
                lang = 'en'
                
            tokens = tokenizer.encode(prompt, language=lang)[:50]
            x = torch.tensor(tokens).unsqueeze(0).to(self.device)
            
            output = self.model.generate(
                x,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50
            )
            
            generated = tokenizer.decode(output[0].cpu().numpy())
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated}")
        
        print("="*60 + "\n")
        self.model.train()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        use_wandb=not args.no_wandb
    )
    
    # Create trainer
    trainer = BrainGPTTrainer(config)
    
    # Resume from checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.step = checkpoint['step']
        print(f"Resumed from step {trainer.step}")
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()