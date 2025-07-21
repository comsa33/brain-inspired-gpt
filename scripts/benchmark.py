#!/usr/bin/env python3
"""
Benchmark script for CortexGPT
Tests model performance and memory efficiency
"""

import os
import sys
import time
import torch
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortexgpt.models.realtime_cortex import RealTimeCortexGPT, AdvancedMemoryConfig
from cortexgpt.tokenization.multilingual_tokenizer import MultilingualTokenizer


def benchmark_generation(model, tokenizer, prompts, max_length=100):
    """Benchmark text generation speed"""
    model.eval()
    device = next(model.parameters()).device
    
    total_time = 0
    total_tokens = 0
    
    with torch.no_grad():
        for prompt in prompts:
            # Encode
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids]).to(device)
            
            # Generate
            start_time = time.time()
            
            generated = input_ids.copy()
            for _ in range(max_length):
                outputs = model(input_tensor)
                next_token_logits = outputs[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                
                if next_token == tokenizer.special_tokens.get('<eos>', 2):
                    break
                
                generated.append(next_token)
                input_tensor = torch.tensor([generated]).to(device)
            
            end_time = time.time()
            
            # Stats
            generation_time = end_time - start_time
            tokens_generated = len(generated) - len(input_ids)
            
            total_time += generation_time
            total_tokens += tokens_generated
    
    return {
        'total_time': total_time,
        'total_tokens': total_tokens,
        'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
        'avg_time_per_prompt': total_time / len(prompts)
    }


def benchmark_memory(model):
    """Benchmark memory usage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved
        }
    else:
        return {
            'allocated_mb': 0,
            'reserved_mb': 0
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CortexGPT model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--num-prompts", type=int, default=10,
                       help="Number of prompts to test")
    parser.add_argument("--max-length", type=int, default=50,
                       help="Maximum generation length")
    
    args = parser.parse_args()
    
    print("🏃 CortexGPT Benchmark")
    print("=" * 50)
    
    # Load model
    print("📥 Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Extract configuration
    config = checkpoint.get('config', AdvancedMemoryConfig())
    vocab_size = checkpoint.get('vocab_size', 50000)
    dim = checkpoint.get('dim', 768)
    
    # Load tokenizer
    tokenizer_path = Path(args.checkpoint).parent / 'tokenizer.json'
    if tokenizer_path.exists():
        tokenizer = MultilingualTokenizer()
        tokenizer.load(str(tokenizer_path))
    else:
        print("⚠️  Tokenizer not found, using default")
        tokenizer = MultilingualTokenizer(vocab_size=vocab_size)
    
    # Create model
    model = RealTimeCortexGPT(
        config=config,
        vocab_size=len(tokenizer.vocab),
        dim=dim,
        use_hybrid_embeddings=True,
        tokenizer=tokenizer
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✅ Model loaded (device: {device})")
    print(f"   Vocab size: {len(tokenizer.vocab)}")
    print(f"   Model dim: {dim}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "Machine learning can help us",
        "인공지능의 미래는",
        "기계 학습이 우리에게",
        "The most important thing in life is",
        "가장 중요한 것은",
        "Technology has changed how we",
        "기술은 우리의 삶을",
        "In the next decade, we will see",
        "앞으로 10년 안에 우리는"
    ][:args.num_prompts]
    
    # Benchmark generation speed
    print(f"\n⚡ Benchmarking generation speed ({len(test_prompts)} prompts)...")
    gen_stats = benchmark_generation(model, tokenizer, test_prompts, args.max_length)
    
    print(f"\n📊 Generation Statistics:")
    print(f"   Total time: {gen_stats['total_time']:.2f} seconds")
    print(f"   Total tokens: {gen_stats['total_tokens']}")
    print(f"   Tokens/second: {gen_stats['tokens_per_second']:.1f}")
    print(f"   Avg time/prompt: {gen_stats['avg_time_per_prompt']:.2f} seconds")
    
    # Benchmark memory
    mem_stats = benchmark_memory(model)
    print(f"\n💾 Memory Usage:")
    print(f"   Allocated: {mem_stats['allocated_mb']:.1f} MB")
    print(f"   Reserved: {mem_stats['reserved_mb']:.1f} MB")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()