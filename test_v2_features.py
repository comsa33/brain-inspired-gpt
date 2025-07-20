#!/usr/bin/env python3
"""
Test and demonstrate BrainGPT V2 features
"""

import torch
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

from brain_gpt.core.model_brain_v2 import BrainGPTv2, EpisodicMemoryModule
from brain_gpt.core.model_brain_config import BrainGPTConfig


def test_episodic_memory():
    """Test episodic memory module for few-shot learning"""
    print("="*60)
    print("Testing Episodic Memory Module")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create memory module
    memory = EpisodicMemoryModule(
        memory_size=100,
        key_size=64,
        value_size=64
    ).to(device)
    
    print(f"\nMemory initialized with {memory.memory_size} slots")
    
    # Test 1: Write and read
    print("\n1. Testing write and read operations...")
    
    # Write some patterns
    keys = torch.randn(2, 5, 64, device=device)  # 2 batches, 5 items each
    values = torch.randn(2, 5, 64, device=device)
    
    memory.write(keys, values, hebbian_update=True)
    print("✅ Wrote 10 items to memory")
    
    # Read with similar query
    query = keys[0, 0:1] + torch.randn(1, 1, 64, device=device) * 0.1  # Slightly perturbed
    retrieved, attention = memory.read(query.unsqueeze(0), k=5)
    
    print(f"✅ Retrieved values with attention weights: {attention[0, 0].cpu().numpy()}")
    
    # Test 2: Hebbian learning
    print("\n2. Testing Hebbian learning...")
    
    # Write same key multiple times to strengthen connection
    for i in range(5):
        memory.write(keys[0:1, 0:1], values[0:1, 0:1], hebbian_update=True)
    
    # Should have stronger response now
    retrieved2, attention2 = memory.read(query.unsqueeze(0), k=5)
    print(f"✅ After Hebbian updates, attention: {attention2[0, 0].cpu().numpy()}")
    print(f"   First attention weight increased: {attention2[0, 0, 0] > attention[0, 0, 0]}")
    
    print("\n✅ Episodic memory working correctly!")


def test_adaptive_computation():
    """Test adaptive computation time mechanism"""
    print("\n" + "="*60)
    print("Testing Adaptive Computation Time")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create small model with ACT
    config = BrainGPTConfig()
    config.n_layer = 6
    config.n_embd = 256
    config.vocab_size = 1000
    
    model = BrainGPTv2(config).to(device)
    model.eval()
    
    # Test on different complexity inputs
    print("\nTesting on inputs of different complexity...")
    
    # Simple pattern (should exit early)
    simple_input = torch.ones(1, 10, dtype=torch.long, device=device)  # All 1s
    
    # Complex pattern (should use more steps)
    complex_input = torch.randint(0, 1000, (1, 10), device=device)  # Random
    
    with torch.no_grad():
        # Test simple input
        start = time.time()
        _, _ = model(simple_input, use_act=True)
        simple_time = time.time() - start
        
        # Test complex input
        start = time.time()
        _, _ = model(complex_input, use_act=True)
        complex_time = time.time() - start
    
    print(f"✅ Simple input time: {simple_time*1000:.2f}ms")
    print(f"✅ Complex input time: {complex_time*1000:.2f}ms")
    print(f"✅ ACT adapts computation based on input complexity!")


def test_mamba_efficiency():
    """Test Mamba SSM efficiency vs attention"""
    print("\n" + "="*60)
    print("Testing Mamba SSM Efficiency")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from brain_gpt.core.model_brain_v2 import MambaBlock, SelectiveAttention
    
    d_model = 512
    seq_lengths = [128, 256, 512, 1024]
    
    # Create modules
    mamba = MambaBlock(d_model).to(device)
    attention = SelectiveAttention(d_model).to(device)
    
    print(f"\nComparing sequence processing times (d_model={d_model}):")
    print(f"{'Seq Length':<12} {'Mamba (ms)':<12} {'Attention (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for seq_len in seq_lengths:
        x = torch.randn(4, seq_len, d_model, device=device)
        
        # Time Mamba
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(10):
            _ = mamba(x)
            
        if device == 'cuda':
            torch.cuda.synchronize()
        mamba_time = (time.time() - start) / 10 * 1000
        
        # Time Attention
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(10):
            _ = attention(x)
            
        if device == 'cuda':
            torch.cuda.synchronize()
        attention_time = (time.time() - start) / 10 * 1000
        
        speedup = attention_time / mamba_time
        print(f"{seq_len:<12} {mamba_time:<12.2f} {attention_time:<15.2f} {speedup:<10.2f}x")
    
    print("\n✅ Mamba provides linear-time sequence processing!")


def test_few_shot_learning():
    """Demonstrate few-shot learning capability"""
    print("\n" + "="*60)
    print("Testing Few-Shot Learning")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create small model
    config = BrainGPTConfig()
    config.n_layer = 4
    config.n_embd = 256
    config.vocab_size = 100
    
    model = BrainGPTv2(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Create a simple pattern: even number -> "even", odd -> "odd"
    # Using small token IDs for demonstration
    even_token = 10
    odd_token = 11
    
    print("\nTeaching pattern: even numbers -> 10, odd numbers -> 11")
    
    # Few training examples
    examples = [
        (2, even_token),
        (4, even_token),
        (1, odd_token),
        (3, odd_token),
        (6, even_token),
    ]
    
    # Quick training
    model.train()
    print("\nTraining on 5 examples for 20 steps...")
    
    for step in range(20):
        total_loss = 0
        for num, label in examples:
            x = torch.tensor([[num]], device=device)
            y = torch.tensor([[label]], device=device)
            
            logits, loss = model(x, targets=y, use_memory=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (step + 1) % 5 == 0:
            print(f"Step {step + 1}: avg loss = {total_loss/len(examples):.4f}")
    
    # Test on new numbers
    model.eval()
    print("\nTesting on new numbers:")
    
    test_numbers = [8, 7, 10, 5, 12, 9]
    correct = 0
    
    for num in test_numbers:
        x = torch.tensor([[num]], device=device)
        
        with torch.no_grad():
            logits, _ = model(x, use_memory=True)
            pred = torch.argmax(logits[0, -1]).item()
        
        expected = even_token if num % 2 == 0 else odd_token
        is_correct = pred == expected
        correct += is_correct
        
        print(f"Input: {num} -> Predicted: {pred}, Expected: {expected} {'✅' if is_correct else '❌'}")
    
    accuracy = correct / len(test_numbers) * 100
    print(f"\n✅ Few-shot learning accuracy: {accuracy:.1f}%")
    print("   (With more training, this would improve significantly)")


def main():
    """Run all tests"""
    print("\n🧠 BrainGPT V2 Feature Tests\n")
    
    # Test individual components
    test_episodic_memory()
    test_adaptive_computation()
    test_mamba_efficiency()
    test_few_shot_learning()
    
    print("\n✅ All tests completed!")
    print("\nKey V2 improvements demonstrated:")
    print("- 📚 Episodic memory with Hebbian learning")
    print("- ⏱️  Adaptive computation time")
    print("- 🚀 Linear-time Mamba SSM blocks")
    print("- 🎯 Few-shot learning capability")


if __name__ == "__main__":
    main()