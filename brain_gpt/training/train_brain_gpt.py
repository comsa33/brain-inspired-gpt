"""
Training script for Brain-Inspired GPT with curriculum learning
Optimized for RTX 3090 with advanced features
"""

import os
import math
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer, KoreanDataCollator
from core.sparse_modules import AdaptiveSparsityGate


@dataclass
class TrainingArguments:
    # Model
    model_name: str = "brain-gpt-3090"
    
    # Data
    train_data_dir: str = "./data"
    korean_data_dir: str = "./data/korean"
    
    # Training
    batch_size: int = 2  # Small for 3090 memory
    gradient_accumulation_steps: int = 16  # Effective batch size = 32
    max_iters: int = 100000
    learning_rate: float = 3e-4
    warmup_iters: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 5
    sparsity_warmup_iters: int = 10000
    language_curriculum: bool = True
    
    # Efficiency
    compile_model: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 100
    checkpoint_interval: int = 5000
    output_dir: str = "./output"
    wandb_project: str = "brain-gpt"
    
    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"


class CurriculumScheduler:
    """
    Implements brain-inspired curriculum learning
    Gradually increases task complexity like human learning
    """
    
    def __init__(self, args: TrainingArguments, config: BrainGPTConfig):
        self.args = args
        self.config = config
        self.current_stage = 0
        
        # Define curriculum stages (like brain development)
        self.stages = [
            # Stage 1: Dense learning (infant brain)
            {
                "name": "dense_learning",
                "iters": 5000,
                "sparsity": 0.0,
                "languages": ["en"],
                "sequence_length": 512,
                "learning_rate_mult": 1.0,
            },
            # Stage 2: Initial pruning (early childhood)
            {
                "name": "initial_pruning", 
                "iters": 10000,
                "sparsity": 0.5,
                "languages": ["en"],
                "sequence_length": 1024,
                "learning_rate_mult": 1.0,
            },
            # Stage 3: Language introduction (language acquisition)
            {
                "name": "multilingual",
                "iters": 20000,
                "sparsity": 0.8,
                "languages": ["en", "ko"],
                "sequence_length": 1536,
                "learning_rate_mult": 0.8,
            },
            # Stage 4: Advanced sparsity (adolescent brain)
            {
                "name": "advanced_sparse",
                "iters": 30000,
                "sparsity": 0.95,
                "languages": ["en", "ko", "mixed"],
                "sequence_length": 2048,
                "learning_rate_mult": 0.5,
            },
            # Stage 5: Fine-tuning (adult brain)
            {
                "name": "fine_tuning",
                "iters": float('inf'),
                "sparsity": 0.98,
                "languages": ["en", "ko", "mixed", "code"],
                "sequence_length": 2048,
                "learning_rate_mult": 0.3,
            },
        ]
        
    def get_stage(self, iteration: int) -> Dict:
        """Get current curriculum stage"""
        cumulative_iters = 0
        
        for stage in self.stages:
            cumulative_iters += stage["iters"]
            if iteration < cumulative_iters:
                return stage
                
        return self.stages[-1]  # Final stage
        
    def get_sparsity(self, iteration: int) -> float:
        """Get target sparsity for current iteration"""
        stage = self.get_stage(iteration)
        
        # Smooth transition between stages
        if iteration < self.args.sparsity_warmup_iters:
            # Linear warmup
            progress = iteration / self.args.sparsity_warmup_iters
            return stage["sparsity"] * progress
        
        return stage["sparsity"]
        
    def get_learning_rate(self, base_lr: float, iteration: int) -> float:
        """Adjust learning rate based on curriculum stage"""
        stage = self.get_stage(iteration)
        
        # Cosine decay with warmup
        if iteration < self.args.warmup_iters:
            return base_lr * iteration / self.args.warmup_iters
            
        # Cosine decay
        progress = (iteration - self.args.warmup_iters) / (self.args.max_iters - self.args.warmup_iters)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return base_lr * stage["learning_rate_mult"] * cosine_decay


class MultilingualDataset(Dataset):
    """
    Dataset that mixes English and Korean data with curriculum awareness
    """
    
    def __init__(
        self,
        data_dir: str,
        korean_data_dir: str,
        tokenizer: MultilingualBrainTokenizer,
        block_size: int = 2048,
        languages: List[str] = ["en", "ko"],
        mix_ratio: Dict[str, float] = None
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.languages = languages
        self.mix_ratio = mix_ratio or {"en": 0.7, "ko": 0.3}
        
        # Load data files
        self.data_files = {}
        
        # English data
        if "en" in languages:
            en_train_path = os.path.join(data_dir, "openwebtext", "train.bin")
            if os.path.exists(en_train_path):
                self.data_files["en"] = np.memmap(en_train_path, dtype=np.uint16, mode='r')
                
        # Korean data
        if "ko" in languages:
            ko_train_path = os.path.join(korean_data_dir, "train_korean.bin")
            if os.path.exists(ko_train_path):
                self.data_files["ko"] = np.memmap(ko_train_path, dtype=np.uint16, mode='r')
                
        # Calculate total size
        self.total_size = sum(len(data) for data in self.data_files.values())
        
    def __len__(self):
        return self.total_size // self.block_size
        
    def __getitem__(self, idx):
        # Select language based on mix ratio
        lang = np.random.choice(
            list(self.mix_ratio.keys()),
            p=list(self.mix_ratio.values())
        )
        
        # Get data from selected language
        if lang in self.data_files:
            data = self.data_files[lang]
            
            # Random position
            max_pos = len(data) - self.block_size - 1
            if max_pos <= 0:
                # Fallback to beginning if data too small
                pos = 0
            else:
                pos = np.random.randint(0, max_pos)
                
            # Extract chunk
            chunk = torch.from_numpy(data[pos:pos + self.block_size + 1].astype(np.int64))
            
            return {
                "input_ids": chunk[:-1],
                "labels": chunk[1:],
                "language": lang
            }
        else:
            # Fallback: return zeros
            return {
                "input_ids": torch.zeros(self.block_size, dtype=torch.long),
                "labels": torch.zeros(self.block_size, dtype=torch.long),
                "language": "en"
            }


class BrainGPTTrainer:
    """
    Trainer implementing brain-inspired training strategies
    """
    
    def __init__(
        self,
        model: BrainGPT,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Move model to device
        self.device = torch.device(args.device)
        self.model = self.model.to(self.device)
        
        # Compile model if requested
        if args.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
            
        # Initialize optimizer
        self.optimizer = self.configure_optimizers()
        
        # Mixed precision
        self.scaler = GradScaler() if args.mixed_precision else None
        
        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(args, model.config)
        
        # Metrics tracking
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "sparsity": [],
            "active_neurons": [],
            "energy_consumed": []
        }
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
    def configure_optimizers(self):
        """Configure AdamW optimizer with weight decay"""
        # Separate parameters by weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.dim() >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
                    
        optim_groups = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        # Use fused AdamW if available
        use_fused = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
            **extra_args
        )
        
        return optimizer
        
    def train(self):
        """Main training loop with brain-inspired curriculum"""
        model = self.model
        args = self.args
        
        # Initialize wandb
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.model_name,
                config=vars(args)
            )
            
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        model.train()
        iter_num = 0
        best_eval_loss = float('inf')
        
        print(f"Starting training on {args.device}...")
        print(f"Model has {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
        
        for epoch in range(args.max_iters // len(train_loader) + 1):
            for batch in train_loader:
                if iter_num >= args.max_iters:
                    break
                    
                # Get curriculum stage
                stage = self.curriculum.get_stage(iter_num)
                
                # Update learning rate
                lr = self.curriculum.get_learning_rate(args.learning_rate, iter_num)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
                # Update model sparsity
                target_sparsity = self.curriculum.get_sparsity(iter_num)
                self._update_model_sparsity(target_sparsity)
                
                # Forward pass
                loss, metrics = self._training_step(batch, stage)
                
                # Backward pass
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Gradient accumulation
                if (iter_num + 1) % args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if args.grad_clip > 0:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        
                    # Optimizer step
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad(set_to_none=True)
                    
                # Logging
                if iter_num % args.log_interval == 0:
                    self._log_metrics(iter_num, loss.item(), metrics, lr)
                    
                # Evaluation
                if iter_num % args.eval_interval == 0:
                    eval_loss = self.evaluate()
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_checkpoint(iter_num, is_best=True)
                        
                # Regular checkpoint
                if iter_num % args.checkpoint_interval == 0:
                    self.save_checkpoint(iter_num)
                    
                iter_num += 1
                
        print("Training completed!")
        
    def _training_step(self, batch: Dict, stage: Dict) -> Tuple[torch.Tensor, Dict]:
        """Single training step with brain-inspired features"""
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        language = batch.get("language", ["en"] * len(input_ids))
        
        # Select language adapter based on batch
        lang_id = language[0] if isinstance(language, list) else language
        
        # Forward pass with mixed precision
        if self.scaler:
            with autocast():
                logits, loss = self.model(input_ids, labels, language_id=lang_id)
        else:
            logits, loss = self.model(input_ids, labels, language_id=lang_id)
            
        # Collect metrics
        metrics = {
            "loss": loss.item(),
            "perplexity": torch.exp(loss).item(),
            "active_neurons": self._count_active_neurons(),
            "energy": self.model.energy_consumed.item() if hasattr(self.model, 'energy_consumed') else 0
        }
        
        return loss, metrics
        
    def _update_model_sparsity(self, target_sparsity: float):
        """Update model sparsity based on curriculum"""
        # This would update sparsity masks in the model
        # For now, just track the target
        pass
        
    def _count_active_neurons(self) -> float:
        """Count percentage of active neurons"""
        total_neurons = 0
        active_neurons = 0
        
        for module in self.model.modules():
            if hasattr(module, 'get_active_neurons'):
                active_neurons += module.get_active_neurons()
                total_neurons += 1
                
        return active_neurons / total_neurons if total_neurons > 0 else 0
        
    def _log_metrics(self, iter_num: int, loss: float, metrics: Dict, lr: float):
        """Log training metrics"""
        print(f"Iter {iter_num}: loss={loss:.4f}, ppl={metrics['perplexity']:.2f}, "
              f"lr={lr:.2e}, active={metrics['active_neurons']:.1%}")
              
        if self.args.wandb_project:
            wandb.log({
                "train/loss": loss,
                "train/perplexity": metrics['perplexity'],
                "train/learning_rate": lr,
                "train/active_neurons": metrics['active_neurons'],
                "train/energy": metrics['energy'],
                "train/iteration": iter_num
            })
            
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate model on validation set"""
        if not self.eval_dataset:
            return 0.0
            
        self.model.eval()
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        total_loss = 0
        total_tokens = 0
        
        for i, batch in enumerate(eval_loader):
            if i >= self.args.eval_iters:
                break
                
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            logits, loss = self.model(input_ids, labels)
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.numel()
            
        avg_loss = total_loss / max(1, i)
        
        print(f"Evaluation loss: {avg_loss:.4f}")
        
        if self.args.wandb_project:
            wandb.log({
                "eval/loss": avg_loss,
                "eval/perplexity": math.exp(avg_loss)
            })
            
        self.model.train()
        
        return avg_loss
        
    def save_checkpoint(self, iter_num: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter_num": iter_num,
            "config": self.model.config,
            "metrics": self.metrics
        }
        
        if is_best:
            path = os.path.join(self.args.output_dir, "best_model.pt")
        else:
            path = os.path.join(self.args.output_dir, f"checkpoint_{iter_num}.pt")
            
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


def main():
    """Main training function"""
    # Parse arguments
    args = TrainingArguments()
    
    # Initialize config
    config = BrainGPTConfig()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = MultilingualBrainTokenizer()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MultilingualDataset(
        args.train_data_dir,
        args.korean_data_dir,
        tokenizer,
        block_size=config.block_size,
        languages=["en", "ko"],
        mix_ratio={"en": 0.7, "ko": 0.3}
    )
    
    eval_dataset = MultilingualDataset(
        args.train_data_dir,
        args.korean_data_dir,
        tokenizer,
        block_size=config.block_size,
        languages=["en", "ko"],
        mix_ratio={"en": 0.7, "ko": 0.3}
    )
    
    # Initialize model
    print("Initializing Brain-Inspired GPT model...")
    model = BrainGPT(config)
    
    # Initialize trainer
    trainer = BrainGPTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Start training
    trainer.train()
    

if __name__ == "__main__":
    main()