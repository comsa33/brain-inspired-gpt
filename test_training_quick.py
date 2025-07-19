#!/usr/bin/env python3
"""
Quick test of multilingual training
"""

import time
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brain_gpt.training.train_multilingual import MultilingualTrainingConfig, MultilingualTrainer


def main():
    # Quick test config
    config = MultilingualTrainingConfig(
        data_dirs=['data/korean_hf'],
        batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=5,
        eval_interval=10,
        use_wandb=False,
        language_sampling='none'  # Disable balanced sampling for speed
    )
    
    print("Starting quick training test...")
    start_time = time.time()
    
    # Create trainer
    trainer = MultilingualTrainer(config)
    
    # Time a few training steps
    step_times = []
    
    # Manually do a few training steps to measure time
    train_loader = torch.utils.data.DataLoader(
        trainer.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Single thread for testing
    )
    
    data_iter = iter(train_loader)
    trainer.model.train()
    
    for i in range(3):
        step_start = time.time()
        
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            break
            
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
            
        x, y = x.to(trainer.device), y.to(trainer.device)
        
        # Forward pass
        logits, loss = trainer.model(x, targets=y)
        
        # Backward pass
        if loss is not None:
            loss.backward()
            
        step_time = time.time() - step_start
        step_times.append(step_time)
        print(f"Step {i}: {step_time:.3f}s, loss={loss.item():.4f}")
    
    total_time = time.time() - start_time
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average step time: {avg_step_time:.3f}s")
    print(f"Estimated time for 100 steps: {avg_step_time * 100:.1f}s")
    
    # Test generation
    print("\nTesting generation...")
    trainer.test_generation()


if __name__ == "__main__":
    main()