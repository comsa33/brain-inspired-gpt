"""
Real-time Training Script for CortexGPT with Memory Optimization

Features:
- Memory-efficient training with gradient accumulation
- Real-time learning integration
- Multi-language support
- Adaptive batch sizing to prevent OOM
"""

import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import psutil
import GPUtil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: uv add wandb")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cortexgpt.models.realtime_cortex import RealTimeCortexGPT, AdvancedMemoryConfig
from cortexgpt.tokenization.multilingual_tokenizer import MultilingualTokenizer
from cortexgpt.data.multilingual_data import create_dataloaders, StreamingDataset
from cortexgpt.data.jsonl_dataset import create_jsonl_dataloaders
from cortexgpt.learning.realtime_learner import RealTimeLearner


class MemoryEfficientTrainer:
    """
    Memory-efficient trainer with adaptive batch sizing and gradient accumulation
    """
    
    def __init__(
        self,
        model: RealTimeCortexGPT,
        tokenizer: MultilingualTokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: argparse.Namespace
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        
        # Device setup with memory monitoring
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Memory management
        self.base_batch_size = args.batch_size
        self.current_batch_size = args.batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation
        self.memory_threshold = 0.9  # 90% GPU memory usage threshold
        
        # Optimizer with memory-efficient settings
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        # For streaming datasets, estimate steps based on typical dataset size
        estimated_steps_per_epoch = 1000  # Reasonable estimate for our data
        total_steps = estimated_steps_per_epoch * args.epochs // self.gradient_accumulation_steps
        warmup_steps = int(0.1 * total_steps)
        
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision with memory optimization
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Real-time learner
        self.realtime_learner = RealTimeLearner(model, tokenizer)
        
        # Monitoring
        self.memory_stats = []
        self.loss_history = []
        self.global_step = 0
        
        # Initialize wandb if requested
        if args.wandb and WANDB_AVAILABLE:
            import datetime
            run_name = f"cortexgpt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=args.wandb_project or "cortexgpt",
                name=run_name,
                config={
                    "model": "RealTimeCortexGPT",
                    "batch_size": args.batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "learning_rate": args.lr,
                    "warmup_steps": warmup_steps,
                    "total_steps": total_steps,
                    "epochs": args.epochs,
                    "stm_capacity": model.config.stm_capacity,
                    "ltm_capacity": model.config.ltm_capacity,
                    "archive_capacity": model.config.archive_capacity,
                    "vocab_size": len(tokenizer.vocab),
                    "model_dim": model.dim,
                    "dataset": args.dataset,
                    "checkpoint_dir": args.checkpoint_dir,
                }
            )
            
            # Watch model
            wandb.watch(model, log="all", log_freq=100)
            
        elif args.wandb and not WANDB_AVAILABLE:
            print("❌ wandb requested but not installed. Continuing without wandb.")
    
    def _create_optimizer(self):
        """Create memory-efficient optimizer"""
        # Group parameters by type for different learning rates
        param_groups = [
            # Embeddings - lower LR
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if "embedding" in n],
                "lr": self.args.lr * 0.5,
                "weight_decay": 0.0
            },
            # Memory systems - adaptive LR
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(x in n for x in ["stm", "ltm", "archive"])],
                "lr": self.args.lr * 0.8,
                "weight_decay": self.args.weight_decay * 0.5
            },
            # Core model - standard LR
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(x in n for x in ["embedding", "stm", "ltm", "archive"])],
                "lr": self.args.lr,
                "weight_decay": self.args.weight_decay
            }
        ]
        
        # Use AdamW with memory-efficient settings
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),  # More aggressive beta2 for stability
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False  # Fused optimizer for memory efficiency
        )
        
        return optimizer
    
    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Linear schedule with warmup"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            gpu_usage = gpu.memoryUsed / gpu.memoryTotal
            gpu_free = gpu.memoryFree
        else:
            gpu_usage = 0.0
            gpu_free = 0.0
        
        cpu_usage = psutil.virtual_memory().percent / 100.0
        
        return {
            'gpu_usage': gpu_usage,
            'gpu_free_mb': gpu_free,
            'cpu_usage': cpu_usage
        }
    
    def adjust_batch_size(self, memory_stats: Dict[str, float]):
        """Dynamically adjust batch size based on memory usage"""
        gpu_usage = memory_stats['gpu_usage']
        
        if gpu_usage > self.memory_threshold and self.current_batch_size > 1:
            # Reduce batch size
            self.current_batch_size = max(1, self.current_batch_size // 2)
            self.gradient_accumulation_steps *= 2
            print(f"Reducing batch size to {self.current_batch_size} due to high memory usage")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        elif gpu_usage < 0.7 and self.current_batch_size < self.base_batch_size:
            # Increase batch size if memory allows
            self.current_batch_size = min(self.base_batch_size, self.current_batch_size * 2)
            self.gradient_accumulation_steps = max(1, self.gradient_accumulation_steps // 2)
            print(f"Increasing batch size to {self.current_batch_size}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with memory optimization"""
        self.model.train()
        
        total_loss = 0
        total_tokens = 0
        batch_count = 0
        accumulation_count = 0
        
        # Start real-time learning thread
        if self.args.realtime_learning:
            self.realtime_learner.start()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get memory stats
            memory_stats = self.get_memory_usage()
            self.memory_stats.append(memory_stats)
            
            # Adjust batch size if needed
            if batch_idx % 100 == 0:
                self.adjust_batch_size(memory_stats)
            
            # Move batch to device
            # Handle both dictionary and tuple formats
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch.get('target_ids', batch['input_ids']).to(self.device)
                attention_mask = batch.get('attention_mask', torch.ones_like(batch['input_ids'])).to(self.device)
            else:
                # Tuple format (input_ids, target_ids)
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                attention_mask = torch.ones_like(input_ids)
            
            # Truncate batch if needed
            if input_ids.size(0) > self.current_batch_size:
                input_ids = input_ids[:self.current_batch_size]
                target_ids = target_ids[:self.current_batch_size]
                attention_mask = attention_mask[:self.current_batch_size]
            
            # Forward pass with mixed precision
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = self.model(input_ids, real_time=False)
                
                # Calculate loss with attention mask
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1),
                    ignore_index=self.tokenizer.special_tokens['<pad>'],
                    reduction='none'
                )
                
                # Apply attention mask
                loss = (loss * attention_mask.view(-1)).sum() / attention_mask.sum()
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            accumulation_count += 1
            
            # Update weights after accumulation
            if accumulation_count % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step
                self.scheduler.step()
                
                batch_count += 1
            
            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_tokens += attention_mask.sum().item()
            
            # Memory consolidation
            if batch_idx % self.args.consolidation_interval == 0:
                self.model.consolidate_memories()
                
                # Clear cache after consolidation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Update global step
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'ppl': math.exp(min(loss.item() * self.gradient_accumulation_steps, 10)),
                'lr': self.scheduler.get_last_lr()[0],
                'gpu': f"{memory_stats['gpu_usage']:.1%}",
                'bs': self.current_batch_size
            })
            
            # Log to wandb
            if self.args.wandb and WANDB_AVAILABLE and batch_idx % 100 == 0:
                wandb.log({
                    'train/loss': loss.item() * self.gradient_accumulation_steps,
                    'train/perplexity': math.exp(min(loss.item() * self.gradient_accumulation_steps, 10)),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/gpu_usage': memory_stats['gpu_usage'],
                    'train/batch_size': self.current_batch_size,
                    'memory/stm_size': len(self.model.stm.memories),
                    'memory/ltm_size': len(self.model.ltm.memories),
                    'memory/archive_size': self.model.archive.index.ntotal
                })
            
            # Process real-time queries if available
            if self.args.realtime_learning and batch_idx % 10 == 0:
                self._process_realtime_queue()
        
        # Stop real-time learning
        if self.args.realtime_learning:
            self.realtime_learner.stop()
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        perplexity = math.exp(min(avg_loss, 10))
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens': total_tokens,
            'batches': batch_count
        }
    
    def _process_realtime_queue(self):
        """Process queries from real-time queue"""
        processed = 0
        max_process = 5  # Process up to 5 queries
        
        while processed < max_process and not self.realtime_learner.learning_queue.empty():
            try:
                example = self.realtime_learner.learning_queue.get_nowait()
                
                # Process query
                response, metadata = self.realtime_learner.process_query(
                    example.query,
                    context=example.context,
                    learn=True
                )
                
                processed += 1
                
            except:
                break
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {'loss': 0.0, 'perplexity': 1.0}
        """Evaluate model with memory efficiency"""
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        lang_losses = {'ko': [], 'en': [], 'mixed': []}
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move batch to device
            # Handle both dictionary and tuple formats
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch.get('target_ids', batch['input_ids']).to(self.device)
                attention_mask = batch.get('attention_mask', torch.ones_like(batch['input_ids'])).to(self.device)
            else:
                # Tuple format (input_ids, target_ids)
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                attention_mask = torch.ones_like(input_ids)
            
            # Forward pass
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = self.model(input_ids, real_time=False)
                
                # Calculate loss
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1),
                    ignore_index=self.tokenizer.special_tokens['<pad>'],
                    reduction='none'
                )
                
                # Apply attention mask
                loss = (loss * attention_mask.view(-1)).sum() / attention_mask.sum()
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += attention_mask.sum().item()
            
            # Track language-specific performance
            for i in range(input_ids.size(0)):
                text = self.tokenizer.decode(input_ids[i].tolist())
                lang = self.realtime_learner.detect_language(text)
                if lang in lang_losses:
                    lang_losses[lang].append(loss.item())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader.dataset)
        perplexity = math.exp(min(avg_loss, 10))
        
        metrics = {
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
            'val_tokens': total_tokens
        }
        
        # Language-specific metrics
        for lang, losses in lang_losses.items():
            if losses:
                metrics[f'val_loss_{lang}'] = np.mean(losses)
                metrics[f'val_ppl_{lang}'] = math.exp(min(np.mean(losses), 10))
        
        return metrics
    
    def train(self, start_epoch: int = 0):
        """Main training loop with memory optimization"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(start_epoch, self.args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.args.epochs} ===")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train metrics: {train_metrics}")
            
            # Evaluate
            val_metrics = self.evaluate()
            print(f"Val metrics: {val_metrics}")
            
            # Save checkpoint
            current_val_loss = val_metrics.get('val_loss', val_metrics.get('loss', float('inf')))
            is_best = current_val_loss < best_val_loss
            
            if is_best:
                best_val_loss = current_val_loss
                patience_counter = 0
                print("📈 New best model!")
            else:
                patience_counter += 1
            
            # Save checkpoint (always save latest, mark best separately)
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if patience_counter >= self.args.patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
            
            # Log to wandb
            if self.args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **val_metrics
                })
            
            # Generate samples
            if epoch % self.args.sample_interval == 0:
                self._generate_samples()
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final cleanup
        if self.args.realtime_learning:
            self.realtime_learner.save_state(f"{self.args.checkpoint_dir}/realtime_final")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = True):
        """Save model checkpoint"""
        checkpoint_path = Path(self.args.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_name = "model_best.pt" if is_best else f"model_epoch_{epoch}.pt"
        self.model.save_checkpoint(str(checkpoint_path / model_name))
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'args': vars(self.args),
            'memory_stats': self.memory_stats[-100:] if self.memory_stats else [],
            'global_step': self.global_step if hasattr(self, 'global_step') else 0
        }
        
        state_name = "training_state_best.pt" if is_best else f"training_state_epoch_{epoch}.pt"
        torch.save(training_state, checkpoint_path / state_name)
        
        # Always save as latest for easy resumption
        torch.save(training_state, checkpoint_path / "training_state_latest.pt")
        self.model.save_checkpoint(str(checkpoint_path / "model_latest.pt"))
        
        # Save tokenizer
        self.tokenizer.save(str(checkpoint_path / "tokenizer.json"))
        
        print(f"✅ Saved checkpoint: epoch {epoch}, best={is_best}")
    
    def load_checkpoint(self, checkpoint_path: str = None):
        """Load checkpoint for resuming training"""
        if checkpoint_path is None:
            checkpoint_path = Path(self.args.checkpoint_dir) / "training_state_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
            
        if not checkpoint_path.exists():
            print(f"⚠️  No checkpoint found at {checkpoint_path}")
            return 0  # Start from epoch 0
            
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        
        # Load training state
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Restore optimizer and scheduler
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        # Restore other state
        start_epoch = state['epoch'] + 1
        self.memory_stats = state.get('memory_stats', [])
        self.global_step = state.get('global_step', 0)
        
        # Load model
        model_path = checkpoint_path.parent / "model_latest.pt"
        if model_path.exists():
            self.model.load_checkpoint(str(model_path))
            print(f"✅ Loaded model from {model_path}")
        
        print(f"✅ Resuming from epoch {start_epoch}")
        return start_epoch
    
    @torch.no_grad()
    def _generate_samples(self):
        """Generate sample outputs in both languages"""
        self.model.eval()
        
        prompts = {
            'en': [
                "The future of artificial intelligence is",
                "def fibonacci(n):",
                "Once upon a time, there was"
            ],
            'ko': [
                "인공지능의 미래는",
                "안녕하세요, 오늘은",
                "기계학습이란"
            ]
        }
        
        print("\n=== Generated Samples ===")
        
        for lang, lang_prompts in prompts.items():
            print(f"\n{lang.upper()} Samples:")
            for prompt in lang_prompts:
                # Tokenize
                input_ids = self.tokenizer.encode(prompt)
                input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Generate
                max_new_tokens = 50
                for _ in range(max_new_tokens):
                    with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = self.model(input_tensor, real_time=False)
                    
                    # Get next token
                    next_token_logits = outputs[0, -1, :]
                    next_token = torch.multinomial(
                        F.softmax(next_token_logits / 0.8, dim=-1), 1
                    )
                    
                    # Append to sequence
                    input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                    
                    # Stop at EOS
                    if next_token.item() == self.tokenizer.special_tokens['<eos>']:
                        break
                
                # Decode
                generated = self.tokenizer.decode(input_tensor[0].tolist())
                print(f"Prompt: {prompt}")
                print(f"Generated: {generated}")
                print()


def main():
    parser = argparse.ArgumentParser(description="Train Real-time CortexGPT")
    
    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--stm-capacity", type=int, default=64)
    parser.add_argument("--ltm-capacity", type=int, default=10000)
    parser.add_argument("--archive-capacity", type=int, default=100000)
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    
    # Real-time learning
    parser.add_argument("--realtime-learning", action="store_true")
    parser.add_argument("--consolidation-interval", type=int, default=500)
    parser.add_argument("--sample-interval", type=int, default=1)
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data", 
                       help="Base directory for datasets")
    parser.add_argument("--dataset", type=str, default="demo",
                       choices=["demo", "combined", "wikipedia", "klue", "openwebtext"],
                       help="Dataset to use for training")
    parser.add_argument("--korean-ratio", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=1024)
    
    # Other arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/realtime")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="cortexgpt", 
                       help="Wandb project name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set memory optimization
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create model configuration
    config = AdvancedMemoryConfig(
        stm_capacity=args.stm_capacity,
        ltm_capacity=args.ltm_capacity,
        archive_capacity=args.archive_capacity
    )
    
    # Create or load tokenizer
    print("Setting up tokenizer...")
    tokenizer_path = Path(args.checkpoint_dir) / "tokenizer.json"
    
    if tokenizer_path.exists() and args.resume:
        # Load existing tokenizer when resuming
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = MultilingualTokenizer(vocab_size=args.vocab_size)
        tokenizer.load(str(tokenizer_path))
        actual_vocab_size = len(tokenizer.vocab)
        print(f"Loaded tokenizer with vocab size: {actual_vocab_size}")
    else:
        # Create new tokenizer and train on actual data
        tokenizer = MultilingualTokenizer(vocab_size=args.vocab_size)
        
        # Collect training texts from actual datasets
        training_texts = []
        
        # For demo dataset, use the actual data
        if args.dataset == "demo":
            demo_files = list(Path(args.data_dir).glob("*.jsonl"))
            for demo_file in demo_files[:5]:  # Use up to 5 demo files
                if demo_file.exists():
                    with open(demo_file, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i >= 1000:  # Limit lines per file
                                break
                            try:
                                data = json.loads(line)
                                text = data.get('text', '') or data.get('content', '')
                                if text:
                                    training_texts.append(text)
                            except:
                                continue
        else:
            # For other datasets, load from the raw data
            dataset_paths = {
                "klue": f"{args.data_dir}/datasets/klue/data.jsonl",
                "korean_wiki": f"{args.data_dir}/datasets/korean_wiki/data.jsonl",
                "wikipedia": f"{args.data_dir}/datasets/wikipedia_en/data.jsonl",
                "openwebtext": f"{args.data_dir}/datasets/openwebtext/data.jsonl",
                "combined": [
                    f"{args.data_dir}/datasets/klue/data.jsonl",
                    f"{args.data_dir}/datasets/korean_wiki/data.jsonl",
                    f"{args.data_dir}/datasets/wikipedia_en/data.jsonl"
                ]
            }
            
            if args.dataset in dataset_paths:
                paths = dataset_paths[args.dataset]
                if not isinstance(paths, list):
                    paths = [paths]
                    
                for path in paths:
                    if Path(path).exists():
                        with open(path, 'r', encoding='utf-8') as f:
                            for i, line in enumerate(f):
                                if i >= 2000:  # Sample more lines for better vocabulary
                                    break
                                try:
                                    data = json.loads(line)
                                    text = data.get('text', '') or data.get('content', '')
                                    if text:
                                        training_texts.append(text)
                                except:
                                    continue
        
        # If no training texts found, use a comprehensive default corpus
        if not training_texts:
            print("Warning: No dataset texts found, using default corpus")
            training_texts = [
                # English samples
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Python is a high-level programming language.",
                "Neural networks are inspired by biological neurons.",
                "Deep learning has revolutionized computer vision.",
                "Natural language processing enables machines to understand text.",
                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "class Model: def __init__(self): self.weights = None",
                # Korean samples
                "안녕하세요! 오늘은 좋은 날씨입니다.",
                "인공지능은 미래의 핵심 기술입니다.",
                "한국어 자연어 처리는 매우 중요한 연구 분야입니다.",
                "기계 학습은 데이터로부터 패턴을 학습합니다.",
                "딥러닝은 인공 신경망을 사용합니다.",
                "파이썬은 가장 인기 있는 프로그래밍 언어 중 하나입니다.",
                "함수형 프로그래밍은 부작용을 최소화합니다.",
                "객체 지향 프로그래밍은 캡슐화를 제공합니다.",
            ] * 50  # Repeat to get more samples
        
        print(f"Training tokenizer on {len(training_texts)} text samples...")
        tokenizer.learn_bpe(training_texts, verbose=True)
        
        # Save the tokenizer immediately after training
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        
        actual_vocab_size = len(tokenizer.vocab)
        print(f"Created tokenizer with vocab size: {actual_vocab_size}")
    
    # Create model with actual vocab size
    actual_vocab_size = len(tokenizer.vocab)
    if actual_vocab_size < 1000:
        print(f"⚠️  Warning: Vocabulary size is very small ({actual_vocab_size}). Consider using more training data for tokenizer.")
    
    model = RealTimeCortexGPT(config, actual_vocab_size, args.dim)
    print(f"Created model with vocab size: {actual_vocab_size}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    
    # Determine data paths based on dataset choice
    if args.dataset == "demo":
        # Use JSONL files for demo dataset
        train_path = f"{args.data_dir}/train.jsonl"
        val_path = f"{args.data_dir}/val.jsonl"
        
        # Check if demo data exists
        if not Path(train_path).exists():
            print(f"❌ Demo data not found!")
            print(f"   Please run: python scripts/data/create_demo_data.py")
            return
            
        # Use JSONL dataloader for demo
        train_loader, val_loader = create_jsonl_dataloaders(
            train_path=train_path,
            val_path=val_path if Path(val_path).exists() else None,
            tokenizer=tokenizer,
            block_size=args.block_size,
            batch_size=args.batch_size,
            num_workers=0  # JSONL dataset doesn't support multiprocessing well
        )
    elif args.dataset == "combined":
        train_dir = f"{args.data_dir}/datasets/combined"
        val_path = None
        
        train_loader, val_loader = create_dataloaders(
            train_dir=train_dir,
            val_path=val_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            block_size=args.block_size
        )
    else:
        # For other datasets, use JSONL files directly
        dataset_path = f"{args.data_dir}/datasets/{args.dataset}/data.jsonl"
        
        if Path(dataset_path).exists():
            print(f"Using JSONL data for {args.dataset} dataset...")
            # Split data for training/validation
            from cortexgpt.data.jsonl_dataset import JSONLDataset
            
            # Create train/val split
            full_dataset = JSONLDataset(dataset_path, tokenizer, block_size=args.block_size)
            train_size = int(0.95 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            ) if val_size > 0 else None
        else:
            # Try prepared .bin files as fallback
            train_dir = f"{args.data_dir}/datasets/{args.dataset}/prepared"
            val_path = f"{args.data_dir}/datasets/{args.dataset}/prepared/val.bin"
            
            # Check if prepared data exists
            if not Path(train_dir).exists():
                print(f"❌ Dataset '{args.dataset}' not found!")
                print(f"   Please run: python cortexgpt/data/download_datasets.py")
                print(f"   Then: python cortexgpt/data/prepare_datasets.py")
                return
            
            train_loader, val_loader = create_dataloaders(
                train_dir=train_dir,
                val_path=val_path if Path(val_path).exists() else None,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                block_size=args.block_size
            )
    
    # Create trainer
    trainer = MemoryEfficientTrainer(model, tokenizer, train_loader, val_loader, args)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        if args.resume == "auto":
            # Auto-resume from latest checkpoint
            start_epoch = trainer.load_checkpoint()
        else:
            # Resume from specific checkpoint
            start_epoch = trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(start_epoch)


if __name__ == "__main__":
    main()