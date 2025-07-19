#!/usr/bin/env python3
"""
Multilingual Brain-Inspired GPT Training
Supports training on mixed datasets with proper language handling
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import wandb
from torch.amp import GradScaler, autocast

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer


@dataclass
class MultilingualTrainingConfig:
    """Training configuration for multilingual models"""
    # Model
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    block_size: int = 1024
    vocab_size: int = 65536
    
    # Brain-inspired parameters
    n_cortical_columns: int = 32
    column_size: int = 32
    n_dendrites: int = 4
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 200
    max_steps: int = 10000
    eval_interval: int = 100
    save_interval: int = 1000
    
    # Multilingual
    language_weights: Dict[str, float] = None  # e.g., {'en': 0.5, 'ko': 0.3, 'mixed': 0.2}
    language_sampling: str = 'balanced'  # 'balanced', 'proportional', 'uniform'
    
    # Optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Paths
    data_dirs: List[str] = None
    checkpoint_dir: str = "checkpoints/multilingual"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "brain-gpt-multilingual"


class MultilingualDataset(Dataset):
    """Dataset that combines multiple language datasets"""
    
    def __init__(self, data_paths: List[Tuple[str, str]], block_size: int = 1024):
        """
        Args:
            data_paths: List of (path, language) tuples
            block_size: Sequence length
        """
        self.block_size = block_size
        self.datasets = []
        self.languages = []
        self.dataset_sizes = []
        
        for path, language in data_paths:
            if os.path.exists(path):
                data = np.memmap(path, dtype=np.uint16, mode='r')
                self.datasets.append(data)
                self.languages.append(language)
                self.dataset_sizes.append(len(data))
                print(f"Loaded {language} dataset from {path}: {len(data):,} tokens")
        
        self.total_size = sum(self.dataset_sizes)
        
        # Calculate dataset offsets
        self.offsets = [0]
        for size in self.dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)
    
    def __len__(self):
        return max(0, self.total_size - self.block_size)
    
    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = 0
        local_idx = idx
        
        for i, offset in enumerate(self.offsets[1:]):
            if idx < offset - self.block_size:
                dataset_idx = i
                local_idx = idx - self.offsets[i]
                break
        else:
            dataset_idx = len(self.datasets) - 1
            local_idx = idx - self.offsets[-1]
        
        # Ensure we don't go out of bounds
        if local_idx + self.block_size + 1 > len(self.datasets[dataset_idx]):
            # Wrap around to beginning of dataset
            local_idx = 0
        
        # Get data
        data = self.datasets[dataset_idx]
        chunk = data[local_idx:local_idx + self.block_size + 1]
        
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        # Add language info
        language = self.languages[dataset_idx]
        
        return x, y, language


class LanguageBalancedSampler(torch.utils.data.Sampler):
    """Sampler that balances languages during training"""
    
    def __init__(self, dataset: MultilingualDataset, 
                 language_weights: Optional[Dict[str, float]] = None,
                 sampling_strategy: str = 'balanced'):
        self.dataset = dataset
        self.language_weights = language_weights or {}
        self.sampling_strategy = sampling_strategy
        
        # Create indices for each language
        self.language_indices = {}
        current_offset = 0
        
        for i, (size, lang) in enumerate(zip(dataset.dataset_sizes, dataset.languages)):
            valid_size = size - dataset.block_size
            if valid_size > 0:
                indices = list(range(current_offset, current_offset + valid_size))
                self.language_indices[lang] = indices
                current_offset += size
        
        # Calculate sampling probabilities
        self._calculate_probabilities()
    
    def _calculate_probabilities(self):
        """Calculate sampling probabilities for each language"""
        if self.sampling_strategy == 'balanced':
            # Equal probability for each language
            num_langs = len(self.language_indices)
            self.probs = {lang: 1.0 / num_langs for lang in self.language_indices}
        
        elif self.sampling_strategy == 'proportional':
            # Probability proportional to dataset size
            total_samples = sum(len(indices) for indices in self.language_indices.values())
            self.probs = {
                lang: len(indices) / total_samples 
                for lang, indices in self.language_indices.items()
            }
        
        elif self.sampling_strategy == 'weighted':
            # Use provided weights
            total_weight = sum(self.language_weights.get(lang, 1.0) 
                              for lang in self.language_indices)
            self.probs = {
                lang: self.language_weights.get(lang, 1.0) / total_weight
                for lang in self.language_indices
            }
    
    def __iter__(self):
        # Generate indices with language balancing
        languages = list(self.language_indices.keys())
        probs = [self.probs[lang] for lang in languages]
        
        while True:
            # Sample a language
            lang = np.random.choice(languages, p=probs)
            
            # Sample an index from that language
            indices = self.language_indices[lang]
            idx = np.random.choice(indices)
            
            yield idx
    
    def __len__(self):
        return len(self.dataset)


def multilingual_collate_fn(batch):
    """Custom collate function for multilingual dataset"""
    if len(batch[0]) == 3:
        # Batch contains (x, y, language) tuples
        xs, ys, languages = zip(*batch)
        x_batch = torch.stack(xs)
        y_batch = torch.stack(ys)
        return x_batch, y_batch, languages
    else:
        # Regular batch
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)


class MultilingualTrainer:
    """Trainer for multilingual Brain-Inspired GPT"""
    
    def __init__(self, config: MultilingualTrainingConfig):
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
        self.model_config.n_cortical_columns = config.n_cortical_columns
        self.model_config.column_size = config.column_size
        self.model_config.gradient_checkpointing = config.gradient_checkpointing
        
        print("Initializing multilingual Brain-Inspired GPT...")
        self.model = BrainGPT(self.model_config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {total_params/1e6:.1f}M parameters")
        
        # Initialize tokenizer
        self.tokenizer = MultilingualBrainTokenizer()
        
        # Load datasets
        self.train_dataset, self.val_datasets = self._load_datasets()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.mixed_precision else None
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=vars(config)
            )
            wandb.watch(self.model, log_freq=100)
        
        self.step = 0
        self.best_val_loss = float('inf')
    
    def _load_datasets(self) -> Tuple[MultilingualDataset, Dict[str, Dataset]]:
        """Load all datasets"""
        print("\nLoading datasets...")
        
        # Find all available datasets
        train_paths = []
        val_paths = {}
        
        for data_dir in self.config.data_dirs:
            data_path = Path(data_dir)
            
            # Check for different dataset types
            # Use set to avoid duplicates
            train_files = set()
            for pattern in ['*_train.bin', '*train.bin']:
                train_files.update(data_path.glob(pattern))
            
            for train_file in train_files:
                    # Detect language from metadata or filename
                    metadata_file = data_path / 'metadata.json'
                    language = 'en'  # default
                    
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            languages = metadata.get('languages', metadata.get('language_stats', {}).keys())
                            # For mixed datasets, use 'mixed' or the dominant language
                            if len(languages) > 1:
                                language = 'mixed'
                            else:
                                language = list(languages)[0] if languages else 'en'
                    elif 'korean' in str(train_file) or 'ko' in str(train_file):
                        language = 'ko'
                    
                    train_paths.append((str(train_file), language))
                    
                    # Find corresponding validation file
                    val_file = train_file.parent / train_file.name.replace('train', 'val')
                    if val_file.exists():
                        val_paths[language] = str(val_file)
        
        if not train_paths:
            raise ValueError(f"No training data found in {self.config.data_dirs}")
        
        # Create multilingual dataset
        train_dataset = MultilingualDataset(train_paths, self.config.block_size)
        
        # Create validation datasets per language
        val_datasets = {}
        for lang, val_path in val_paths.items():
            val_datasets[lang] = MultilingualDataset([(val_path, lang)], self.config.block_size)
        
        print(f"Total training samples: {len(train_dataset):,}")
        print(f"Languages: {train_dataset.languages}")
        
        return train_dataset, val_datasets
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting multilingual training on {self.device}...")
        
        # Create data loader with language balancing
        if self.config.language_sampling != 'none':
            sampler = LanguageBalancedSampler(
                self.train_dataset,
                self.config.language_weights,
                self.config.language_sampling
            )
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=2,
                pin_memory=True,
                collate_fn=multilingual_collate_fn
            )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                collate_fn=multilingual_collate_fn
            )
        
        self.model.train()
        start_time = time.time()
        
        data_iter = iter(train_loader)
        language_counts = {lang: 0 for lang in self.train_dataset.languages}
        
        for step in range(self.config.max_steps):
            self.step = step
            
            # Accumulate gradients
            total_loss = 0
            self.optimizer.zero_grad()
            
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                
                if len(batch) == 3:
                    x, y, language = batch
                    # Update language statistics
                    for lang in language:
                        language_counts[lang] += 1
                else:
                    x, y = batch
                    language = ['unknown'] * x.size(0)
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                if self.config.mixed_precision:
                    with autocast('cuda'):
                        # Pass targets to model for proper training mode
                        logits, model_loss = self.model(x, targets=y)
                        
                        # Use model's computed loss if available
                        if model_loss is not None:
                            loss = model_loss / self.config.gradient_accumulation_steps
                        else:
                            # Fallback to manual loss computation
                            if isinstance(logits, tuple):
                                logits = logits[0]
                        
                            
                            # Ensure proper shapes
                            if logits.dim() == 2:
                                # logits is (batch_size, vocab_size) - single token prediction
                                # y should be (batch_size,)
                                if y.dim() == 2:
                                    y = y[:, -1]  # Take last token
                            else:
                                # logits is (batch_size, seq_len, vocab_size)
                                logits = logits.view(-1, logits.size(-1))
                                y = y.view(-1)
                            
                            loss = nn.functional.cross_entropy(logits, y)
                            loss = loss / self.config.gradient_accumulation_steps
                else:
                    # Pass targets to model for proper training mode
                    logits, model_loss = self.model(x, targets=y)
                    
                    # Use model's computed loss if available
                    if model_loss is not None:
                        loss = model_loss / self.config.gradient_accumulation_steps
                    else:
                        # Fallback to manual loss computation
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        
                        # Ensure proper shapes (same as above)
                        if logits.dim() == 2:
                            if y.dim() == 2:
                                y = y[:, -1]
                        else:
                            logits = logits.view(-1, logits.size(-1))
                            y = y.view(-1)
                        
                        loss = nn.functional.cross_entropy(logits, y)
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
            
            # Logging
            if step % 10 == 0:
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                
                # Calculate language distribution
                total_samples = sum(language_counts.values())
                lang_dist = {
                    lang: count / total_samples if total_samples > 0 else 0
                    for lang, count in language_counts.items()
                }
                
                print(f"Step {step}: loss={total_loss:.4f}, lr={lr:.2e}")
                print(f"  Language distribution: {lang_dist}")
                
                if self.config.use_wandb:
                    wandb.log({
                        'train/loss': total_loss,
                        'train/lr': lr,
                        'train/step': step,
                        **{f'train/lang_{lang}': dist for lang, dist in lang_dist.items()}
                    })
            
            # Evaluation
            if step % self.config.eval_interval == 0 and step > 0:
                val_losses = self.evaluate()
                print(f"Validation losses: {val_losses}")
                
                avg_val_loss = np.mean(list(val_losses.values()))
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.save_checkpoint(best=True)
                
                self.model.train()
            
            # Save checkpoint
            if step % self.config.save_interval == 0 and step > 0:
                self.save_checkpoint()
                self.test_generation()
        
        # Final save
        self.save_checkpoint(final=True)
        print(f"\nTraining complete in {(time.time() - start_time)/60:.1f} minutes")
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation sets for each language"""
        self.model.eval()
        val_losses = {}
        
        for lang, val_dataset in self.val_datasets.items():
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=multilingual_collate_fn
            )
            
            losses = []
            for i, batch in enumerate(val_loader):
                if i >= 50:  # Evaluate on subset
                    break
                
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Pass targets for proper evaluation
                logits, loss = self.model(x, targets=y)
                
                # If model doesn't compute loss, do it manually
                if loss is None:
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )
                
                losses.append(loss.item())
            
            val_losses[lang] = np.mean(losses) if losses else float('inf')
        
        if self.config.use_wandb:
            wandb.log({
                **{f'val/loss_{lang}': loss for lang, loss in val_losses.items()},
                'val/loss_avg': np.mean(list(val_losses.values())),
                'val/step': self.step
            })
        
        return val_losses
    
    def save_checkpoint(self, best=False, final=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model_config,
            'training_config': vars(self.config),
            'step': self.step,
            'best_val_loss': self.best_val_loss,
        }
        
        if final:
            path = Path(self.config.checkpoint_dir) / "multilingual_final.pt"
        elif best:
            path = Path(self.config.checkpoint_dir) / "multilingual_best.pt"
        else:
            path = Path(self.config.checkpoint_dir) / f"multilingual_step{self.step}.pt"
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    @torch.no_grad()
    def test_generation(self):
        """Test generation in multiple languages"""
        self.model.eval()
        
        test_prompts = [
            ("The future of artificial intelligence", "en"),
            ("인공지능의 미래는", "ko"),
            ("Once upon a time", "en"),
            ("옛날 옛적에", "ko"),
            ("def hello_world():", "en"),
        ]
        
        print("\n" + "="*60)
        print("Generation Examples:")
        
        for prompt, expected_lang in test_prompts:
            tokens = self.tokenizer.encode(prompt, language=expected_lang)[:50]
            x = torch.tensor(tokens).unsqueeze(0).to(self.device)
            
            output = self.model.generate(
                x,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50
            )
            
            generated = self.tokenizer.decode(output[0].cpu().numpy())
            print(f"\nPrompt ({expected_lang}): {prompt}")
            print(f"Generated: {generated}")
        
        print("="*60 + "\n")
        self.model.train()


def main():
    parser = argparse.ArgumentParser(description='Train multilingual Brain-Inspired GPT')
    parser.add_argument('--data-dirs', nargs='+', 
                        default=['data/korean_hf', 'data/simple', 'data/fineweb', 'data/redpajama_v2'],
                        help='Directories containing training data')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--language-sampling', type=str, default='balanced',
                        choices=['balanced', 'proportional', 'weighted', 'none'])
    parser.add_argument('--no-wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Create config
    config = MultilingualTrainingConfig(
        data_dirs=args.data_dirs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        language_sampling=args.language_sampling,
        use_wandb=not args.no_wandb
    )
    
    # Filter to existing directories
    config.data_dirs = [d for d in config.data_dirs if os.path.exists(d)]
    if not config.data_dirs:
        print("No valid data directories found. Please prepare datasets first:")
        print("  uv run data/openwebtext/prepare_simple.py")
        print("  uv run brain_gpt/training/prepare_korean_hf_datasets.py")
        return
    
    print(f"Using data directories: {config.data_dirs}")
    
    # Create trainer
    trainer = MultilingualTrainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()