"""
Training script for CortexGPT with memory consolidation and continuous learning.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cortexgpt.models.cortex_gpt import CortexGPT, MemoryConfig
from cortexgpt.data.dataset import TokenizedDataset


class CortexGPTTrainer:
    """
    Trainer for CortexGPT with special handling for memory systems.
    """
    
    def __init__(
        self,
        model: CortexGPT,
        train_dataset: Dataset,
        val_dataset: Dataset,
        args: argparse.Namespace
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.args = args
        
        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Optimizer with different learning rates for different components
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        
        # Mixed precision training
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize wandb
        if args.wandb:
            wandb.init(
                project="cortex-gpt",
                name=f"cortex-gpt-{time.strftime('%Y%m%d-%H%M%S')}",
                config=vars(args)
            )
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for memory systems"""
        param_groups = [
            # Core model parameters
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if "stm" not in n and "ltm" not in n and "consolidator" not in n],
                "lr": self.args.lr
            },
            # STM parameters (faster learning)
            {
                "params": [p for n, p in self.model.named_parameters() if "stm" in n],
                "lr": self.args.lr * 2
            },
            # LTM parameters (slower learning)
            {
                "params": [p for n, p in self.model.named_parameters() if "ltm" in n],
                "lr": self.args.lr * 0.5
            },
            # Consolidator parameters
            {
                "params": [p for n, p in self.model.named_parameters() if "consolidator" in n],
                "lr": self.args.lr * 0.8
            }
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with memory consolidation"""
        self.model.train()
        
        total_loss = 0
        total_tokens = 0
        consolidation_events = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle tuple dataset output
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
            else:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
            
            # Forward pass with mixed precision
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                logits = self.model(input_ids)
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += (labels != -100).sum().item()
            
            # Periodic memory consolidation (simulate sleep cycles)
            if batch_idx % self.args.consolidation_interval == 0:
                self._consolidate_memories()
                consolidation_events += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "lr": self.scheduler.get_last_lr()[0],
                "consolidations": consolidation_events
            })
            
            # Log to wandb
            if self.args.wandb and batch_idx % 100 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/perplexity": math.exp(loss.item()),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/consolidations": consolidation_events,
                    "train/stm_size": len(self.model.stm.keys),
                    "train/ltm_size": self.model.ltm.index.ntotal
                })
        
        return {
            "loss": total_loss / len(self.train_dataset),
            "perplexity": math.exp(total_loss / total_tokens),
            "consolidations": consolidation_events
        }
    
    def _consolidate_memories(self):
        """Trigger memory consolidation (sleep-like process)"""
        with torch.no_grad():
            self.model.consolidator.consolidate(self.model.stm, self.model.ltm)
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        stm_hits = 0
        ltm_hits = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # Handle tuple dataset output
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
            else:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += (labels != -100).sum().item()
        
        metrics = {
            "val_loss": total_loss / len(self.val_dataset),
            "val_perplexity": math.exp(total_loss / total_tokens)
        }
        
        return metrics
    
    def train(self):
        """Main training loop"""
        best_val_loss = float("inf")
        
        for epoch in range(self.args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.args.epochs} ===")
            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            print(f"Train metrics: {train_metrics}")
            
            # Evaluate
            val_metrics = self.evaluate()
            print(f"Val metrics: {val_metrics}")
            
            # Save checkpoint
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(epoch, val_metrics)
                print("Saved best checkpoint!")
            
            # Log to wandb
            if self.args.wandb:
                wandb.log({
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **val_metrics
                })
            
            # Generate samples
            if epoch % self.args.sample_interval == 0:
                self._generate_samples()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "args": self.args
        }
        
        path = Path(self.args.checkpoint_dir) / f"cortex_gpt_best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    @torch.no_grad()
    def _generate_samples(self):
        """Generate sample outputs to monitor quality"""
        self.model.eval()
        
        prompts = [
            "The future of artificial intelligence",
            "Once upon a time",
            "def hello_world():",
            "The meaning of life is"
        ]
        
        print("\n=== Generated Samples ===")
        
        for prompt in prompts:
            # Tokenize prompt (simplified - use actual tokenizer in production)
            input_ids = torch.randint(0, 50000, (1, 10)).to(self.device)
            
            # Generate
            output = self.model.generate(input_ids, max_length=50)
            
            print(f"Prompt: {prompt}")
            print(f"Generated: [Token IDs: {output.tolist()[0][:20]}...]")
            print()


def main():
    parser = argparse.ArgumentParser(description="Train CortexGPT")
    
    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--stm-capacity", type=int, default=128)
    parser.add_argument("--ltm-dim", type=int, default=256)
    parser.add_argument("--cortical-columns", type=int, default=16)
    parser.add_argument("--sparsity-ratio", type=float, default=0.05)
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--consolidation-interval", type=int, default=500)
    parser.add_argument("--sample-interval", type=int, default=1)
    
    # Other arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/cortex")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create model
    config = MemoryConfig(
        stm_capacity=args.stm_capacity,
        ltm_dim=args.ltm_dim,
        cortical_columns=args.cortical_columns,
        sparsity_ratio=args.sparsity_ratio
    )
    
    model = CortexGPT(config, args.vocab_size, args.dim)
    
    # Load datasets
    train_dataset = TokenizedDataset("data/sample_train.bin")
    val_dataset = TokenizedDataset("data/sample_val.bin")
    
    # Create trainer
    trainer = CortexGPTTrainer(model, train_dataset, val_dataset, args)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()