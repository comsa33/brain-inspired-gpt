"""
Unified CortexGPT Trainer with all enhancements
Combines stability mechanisms, neuroscience features, and performance optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import os
from pathlib import Path
from tqdm import tqdm

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed, logging will be limited")


class UnifiedCortexTrainer:
    """Unified trainer for CortexGPT with all phase improvements"""
    
    def __init__(self, model, train_dataset, val_dataset, args):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.args = args
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Optimizer with parameter groups
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        self.scheduler = self._create_scheduler(total_steps, warmup_steps)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.loss_history = []
        self.recent_losses = []
        
        # Phase 1: Loss spike detection
        self.loss_spike_threshold = 3.0
        self.recovery_checkpoint = None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if available
        if args.wandb and WANDB_AVAILABLE:
            self._init_wandb()
            
    def _create_optimizer(self):
        """Create optimizer with different learning rates for components"""
        # Parameter groups based on unified model components
        param_groups = [
            # Memory systems - lower learning rate
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(x in n for x in ['stm', 'ltm', 'episodic', 'working', 'memory'])],
                'lr': self.args.lr * 0.1,
                'name': 'memory_systems'
            },
            # Neuroscience components - medium learning rate
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(x in n for x in ['homeostatic', 'oscillator', 'cls', 'metaplastic'])],
                'lr': self.args.lr * 0.5,
                'name': 'neuroscience'
            },
            # Core model - normal learning rate
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(x in n for x in ['stm', 'ltm', 'episodic', 'working', 'memory',
                                                      'homeostatic', 'oscillator', 'cls', 'metaplastic'])],
                'lr': self.args.lr,
                'name': 'core_model'
            }
        ]
        
        # Filter out empty groups
        param_groups = [g for g in param_groups if len(list(g['params'])) > 0]
        
        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8
        )
        
    def _create_scheduler(self, total_steps, warmup_steps):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
            
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        config = vars(self.args)
        config.update({
            'model_type': 'unified_cortex_gpt',
            'phase_1_stability': True,
            'phase_2_neuroscience': True,
            'phase_3_performance': True
        })
        
        wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            config=config,
            name=f"unified-cortex-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        
        # Watch model
        wandb.watch(self.model, log="all", log_freq=100)
        
    def detect_loss_spike(self, current_loss: float) -> bool:
        """Detect training instability from Phase 1"""
        if len(self.recent_losses) < 10:
            return False
            
        recent_avg = np.mean(self.recent_losses[-10:])
        if current_loss > self.loss_spike_threshold * recent_avg:
            return True
        return False
        
    def train(self):
        """Main training loop with all enhancements"""
        print(f"Starting Unified CortexGPT training on {self.device}")
        print(f"Model config: {self.model.config}")
        
        for epoch in range(self.args.epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate(epoch)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best', epoch, val_loss)
                
            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'epoch_{epoch+1}', epoch, val_loss)
                
            # Log epoch summary
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Best Val Loss: {self.best_val_loss:.4f}")
            
            # Log model stats
            stats = self.model.get_stats()
            print(f"  Model Stats: {stats}")
            
        # Cleanup
        if hasattr(self.model, 'shutdown'):
            self.model.shutdown()
            
        print("\nTraining completed!")
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with stability mechanisms and gradient accumulation"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Get gradient accumulation steps
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        # Zero gradients at start
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device) if 'labels' in batch else input_ids
            
            # Forward pass
            outputs = self.model(input_ids)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            
            # Phase 1: Loss spike detection (on unscaled loss)
            if self.detect_loss_spike(loss.item() * gradient_accumulation_steps):
                print(f"\nLoss spike detected: {loss.item() * gradient_accumulation_steps:.4f}")
                if self.recovery_checkpoint is not None:
                    print("Recovering from checkpoint...")
                    self.load_checkpoint(self.recovery_checkpoint)
                    continue
                    
            # Backward pass
            loss.backward()
            
            # Perform optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.args.grad_clip
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update progress bar with gradient norm
                pbar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'grad': f'{grad_norm.item():.2f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
            else:
                # Update progress bar without gradient norm
                pbar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'accumulating': f'{(batch_idx % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
            
            # Update metrics (with unscaled loss)
            total_loss += loss.item() * gradient_accumulation_steps
            self.recent_losses.append(loss.item() * gradient_accumulation_steps)
            if len(self.recent_losses) > 100:
                self.recent_losses.pop(0)
            
            # Only increment global step when optimizer steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                self.global_step += 1
                
                # Logging
                if self.args.wandb and WANDB_AVAILABLE and self.global_step % 10 == 0:
                    wandb.log({
                        'train/loss': loss.item() * gradient_accumulation_steps,
                        'train/grad_norm': grad_norm.item() if 'grad_norm' in locals() else 0,
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })
                
            # Save recovery checkpoint periodically
            if self.global_step % 1000 == 0:
                self.recovery_checkpoint = self.save_checkpoint(
                    'recovery', epoch, loss.item() * gradient_accumulation_steps
                )
                
        # Handle any remaining gradients
        if (num_batches % gradient_accumulation_steps) != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.args.grad_clip
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        return total_loss / num_batches
        
    def validate(self, epoch: int) -> float:
        """Validation with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        total_perplexity = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device) if 'labels' in batch else input_ids
                
                outputs = self.model(input_ids)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                total_loss += loss.item()
                total_perplexity += torch.exp(loss).item()
                
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        # Log validation metrics
        if self.args.wandb and WANDB_AVAILABLE:
            metrics = {
                'val/loss': avg_loss,
                'val/perplexity': avg_perplexity,
                'epoch': epoch
            }
            
            # Add model stats
            stats = self.model.get_stats()
            for key, value in stats.items():
                metrics[f'model/{key}'] = value
                
            wandb.log(metrics)
            
        return avg_loss
        
    def save_checkpoint(self, name: str, epoch: int, loss: float) -> str:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'global_step': self.global_step,
            'config': self.model.config,
            'args': self.args
        }
        
        path = self.checkpoint_dir / f'cortex_gpt_{name}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        return str(path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from {path}")
        

def create_trainer(model, train_dataset, val_dataset, args):
    """Factory function to create unified trainer"""
    return UnifiedCortexTrainer(model, train_dataset, val_dataset, args)