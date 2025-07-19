#!/usr/bin/env python3
"""
Training script for smaller Brain-Inspired GPT model
Optimized for RTX 3090 with limited memory
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer
from train_brain_gpt import BrainGPTTrainer, parse_args


def create_small_model_config():
    """Create a smaller model configuration for RTX 3090"""
    config = BrainGPTConfig()
    
    # Smaller model that fits in RTX 3090
    config.n_layer = 12  # Reduced from 48
    config.n_head = 16   # Reduced from 32
    config.n_embd = 768  # Reduced from 2048
    config.block_size = 1024  # Reduced from 2048
    
    # Adjust cortical columns to match
    config.n_cortical_columns = 16
    config.column_size = 48  # 16 * 48 = 768
    
    # Keep brain-inspired features
    config.sparsity_base = 0.95
    config.n_dendrites = 8
    config.gradient_checkpointing = True
    config.mixed_precision = True
    
    return config


def main():
    """Main training function with smaller model"""
    # Parse arguments
    args = parse_args()
    
    # Override some args for small model
    args.model_name = "brain-gpt-small"
    args.batch_size = 4
    args.gradient_accumulation_steps = 8
    args.max_iters = 10000
    args.eval_interval = 100
    args.save_interval = 500
    args.log_interval = 10
    
    print("🧠 Brain-Inspired GPT Training (Small Model)")
    print("=" * 60)
    
    # Create small model config
    config = create_small_model_config()
    
    # Print model size
    total_params = config.n_layer * (
        4 * config.n_embd * config.n_embd +  # MLP
        3 * config.n_embd * config.n_embd +  # QKV
        config.n_embd * config.n_embd        # Output projection
    ) + config.n_embd * config.vocab_size * 2  # Embeddings
    
    effective_params = total_params * (1 - config.sparsity_base)
    
    print(f"Model Configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Hidden size: {config.n_embd}")
    print(f"  Attention heads: {config.n_head}")
    print(f"  Sequence length: {config.block_size}")
    print(f"  Total parameters: ~{total_params/1e6:.0f}M")
    print(f"  Effective parameters: ~{effective_params/1e6:.0f}M")
    print(f"  Estimated memory: ~{total_params * 4 / 1e9:.1f}GB")
    
    # Initialize trainer with small model
    trainer = BrainGPTTrainer(args, config)
    
    # Start training
    print("\n🚀 Starting training...")
    trainer.train()
    

if __name__ == "__main__":
    main()