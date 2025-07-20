#!/usr/bin/env python3
"""
Brain-Inspired GPT V2 Training Script
Optimized for efficiency and performance on RTX 3090
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
from typing import Optional, Dict, Any
import wandb
from torch.amp import GradScaler, autocast
import math
from contextlib import nullcontext

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain_v2 import BrainGPTv2
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer


@dataclass
class TrainingConfig:
    """Optimized training configuration for BrainGPTv2"""
    # Model - smaller config for RTX 3090
    n_layer: int = 12
    n_head: int = 16  
    n_embd: int = 768  # Smaller than v1 for efficiency
    block_size: int = 1024
    vocab_size: int = 65536  # Match tokenizer vocab size
    
    # Training
    batch_size: int = 4  # Reduced for memory efficiency with episodic memory
    gradient_accumulation_steps: int = 8  # Effective batch size = 32
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    max_steps: int = 10000
    eval_interval: int = 100
    save_interval: int = 1000
    
    # Optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = False  # Disable for now due to dtype issues
    compile_model: bool = False  # Disable compile for debugging
    
    # Memory settings
    use_memory: bool = False  # Temporarily disable to avoid OOM during training
    memory_write_prob: float = 0.1
    
    # Paths
    data_dir: str = "data/simple"
    checkpoint_dir: str = "checkpoints/v2"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "brain-gpt-v2"
    log_interval: int = 10
    
    # Hardware optimization
    num_workers: int = 4
    pin_memory: bool = True
    non_blocking: bool = True
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine" or "linear"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


class EfficientDataset(Dataset):
    """Memory-mapped dataset for efficient data loading"""
    
    def __init__(self, data_path: str, block_size: int = 1024):
        self.block_size = block_size
        # Memory map for efficiency
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.length = len(self.data) - block_size - 1
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Get chunk
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        # Clamp values to vocab size to prevent index errors
        # The tokenizer uses 65536 vocab size
        vocab_size = 65536
        x = torch.clamp(x, 0, vocab_size - 1)
        y = torch.clamp(y, 0, vocab_size - 1)
        
        return x, y


class BrainGPTv2Trainer:
    """Optimized trainer for BrainGPTv2"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model_config = BrainGPTConfig()
        self.model_config.n_layer = config.n_layer
        self.model_config.n_head = config.n_head
        self.model_config.n_embd = config.n_embd
        self.model_config.block_size = config.block_size
        self.model_config.vocab_size = config.vocab_size
        self.model_config.gradient_checkpointing = config.gradient_checkpointing
        
        print("Initializing BrainGPTv2...")
        self.model = BrainGPTv2(self.model_config)
        
        # Compile model if using PyTorch 2.0+
        if config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with PyTorch 2.0...")
            self.model = torch.compile(self.model)
            
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params/1e6:.1f}M")
        print(f"Trainable parameters: {trainable_params/1e6:.1f}M")
        
        # Initialize tokenizer
        self.tokenizer = MultilingualBrainTokenizer()
        
        # Load datasets
        self.train_dataset = self._load_dataset('train')
        self.val_dataset = self._load_dataset('val')
        
        # Initialize optimizer with parameter groups
        self.optimizer = self._init_optimizer()
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if config.mixed_precision and self.device == 'cuda' else None
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=vars(config),
                name=f"brain-gpt-v2-{time.strftime('%Y%m%d-%H%M%S')}"
            )
            wandb.watch(self.model, log_freq=100)
        
        self.step = 0
        self.best_val_loss = float('inf')
        
    def _load_dataset(self, split: str) -> EfficientDataset:
        """Load dataset split"""
        data_path = Path(self.config.data_dir) / f"simple_{split}.bin"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
            
        dataset = EfficientDataset(str(data_path), self.config.block_size)
        print(f"Loaded {split} dataset: {len(dataset):,} samples")
        
        return dataset
    
    def _init_optimizer(self):
        """Initialize optimizer with parameter groups"""
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8
        )
        
        return optimizer
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for current step"""
        # Warmup
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        
        # After warmup
        progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        
        if self.config.lr_scheduler == "cosine":
            # Cosine decay
            return self.config.min_lr + (self.config.learning_rate - self.config.min_lr) * \
                   0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            # Linear decay
            return self.config.min_lr + (self.config.learning_rate - self.config.min_lr) * \
                   (1.0 - progress)
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training on {self.device}...")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        
        self.model.train()
        start_time = time.time()
        
        # Training metrics
        running_loss = 0.0
        running_tokens = 0
        
        data_iter = iter(train_loader)
        
        for step in range(self.config.max_steps):
            self.step = step
            
            # Update learning rate
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Accumulate gradients
            total_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                
                x, y = batch
                x = x.to(self.device, non_blocking=self.config.non_blocking)
                y = y.to(self.device, non_blocking=self.config.non_blocking)
                
                # Mixed precision context
                ctx = autocast('cuda', dtype=torch.float16) if self.config.mixed_precision else nullcontext()
                
                with ctx:
                    # Forward pass
                    logits, loss = self.model(x, targets=y, use_memory=self.config.use_memory)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                running_loss += loss.item()
                running_tokens += x.numel()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Logging
            if step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = running_tokens / elapsed
                avg_loss = running_loss / self.config.log_interval
                
                print(f"Step {step}: loss={avg_loss:.4f}, lr={lr:.2e}, "
                      f"tokens/s={tokens_per_sec:.0f}, time={elapsed/60:.1f}min")
                
                if self.config.use_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lr': lr,
                        'train/tokens_per_sec': tokens_per_sec,
                        'train/step': step,
                    })
                
                running_loss = 0.0
            
            # Evaluation
            if step % self.config.eval_interval == 0 and step > 0:
                val_loss = self.evaluate()
                print(f"Validation loss: {val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(best=True)
                
                if self.config.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/best_loss': self.best_val_loss,
                        'val/step': step,
                    })
                
                self.model.train()
            
            # Save checkpoint
            if step % self.config.save_interval == 0 and step > 0:
                self.save_checkpoint()
                self.test_generation()
        
        # Final save
        self.save_checkpoint(final=True)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/3600:.1f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,  # Larger batch for eval
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        losses = []
        for i, (x, y) in enumerate(val_loader):
            if i >= 50:  # Evaluate on subset
                break
                
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            # Forward pass without memory
            _, loss = self.model(x, targets=y, use_memory=False)
            losses.append(loss.item())
        
        return np.mean(losses)
    
    def save_checkpoint(self, best=False, final=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_config': self.model_config,
            'step': self.step,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if final:
            path = Path(self.config.checkpoint_dir) / "brain_gpt_v2_final.pt"
        elif best:
            path = Path(self.config.checkpoint_dir) / "brain_gpt_v2_best.pt"
        else:
            path = Path(self.config.checkpoint_dir) / f"brain_gpt_v2_step{self.step}.pt"
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    @torch.no_grad()
    def test_generation(self):
        """Test generation quality"""
        self.model.eval()
        
        test_prompts = [
            "The future of artificial intelligence",
            "Once upon a time",
            "def hello_world():",
            "The meaning of life is",
        ]
        
        print("\n" + "="*60)
        print("Generation Examples:")
        
        for prompt in test_prompts:
            tokens = self.tokenizer.encode(prompt, language='en')[:50]
            x = torch.tensor(tokens).unsqueeze(0).to(self.device)
            
            # Generate with memory and ACT
            output = self.model.generate(
                x,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
                use_memory=True,
                use_act=True
            )
            
            generated = self.tokenizer.decode(output[0].cpu().numpy())
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated}")
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train BrainGPTv2')
    parser.add_argument('--data-dir', type=str, default='data/simple',
                        help='Directory containing training data')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=6e-4)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    parser.add_argument('--compile', action='store_true', help='Use PyTorch 2.0 compile')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        use_wandb=not args.no_wandb,
        compile_model=args.compile
    )
    
    # Create trainer
    trainer = BrainGPTv2Trainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.step = checkpoint['step']
        trainer.best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from checkpoint at step {trainer.step}")
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()