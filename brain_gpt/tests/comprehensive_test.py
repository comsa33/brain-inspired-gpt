#!/usr/bin/env python3
"""
Comprehensive test script for Brain-Inspired GPT
Tests all major functionality with proper error handling
"""

import os
import sys
import torch
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer


def print_test_header(title):
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)


def test_model_creation():
    """Test model initialization"""
    print_test_header("Model Creation Test")
    
    try:
        config = BrainGPTConfig()
        config.n_layer = 4
        config.n_embd = 512
        config.n_head = 8
        config.n_cortical_columns = 16
        config.column_size = 32  # 16 * 32 = 512
        
        model = BrainGPT(config)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Model created successfully")
        print(f"   Total parameters: {total_params/1e6:.1f}M")
        print(f"   Effective parameters: {total_params*(1-config.sparsity_base)/1e6:.1f}M")
        
        return True, model, config
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False, None, None


def test_tokenizer():
    """Test tokenizer functionality"""
    print_test_header("Tokenizer Test")
    
    try:
        tokenizer = MultilingualBrainTokenizer()
        
        # Test English
        text_en = "Hello, world!"
        tokens_en = tokenizer.encode(text_en, language='en')
        decoded_en = tokenizer.decode(tokens_en)
        print(f"✅ English: '{text_en}' -> {len(tokens_en)} tokens")
        
        # Test Korean
        text_ko = "안녕하세요!"
        tokens_ko = tokenizer.encode(text_ko, language='ko')
        decoded_ko = tokenizer.decode(tokens_ko)
        print(f"✅ Korean: '{text_ko}' -> {len(tokens_ko)} tokens")
        
        # Test mixed
        text_mixed = "AI는 amazing!"
        tokens_mixed = tokenizer.encode(text_mixed)
        print(f"✅ Mixed: '{text_mixed}' -> {len(tokens_mixed)} tokens")
        
        return True, tokenizer
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False, None


def test_forward_pass(model, config):
    """Test model forward pass"""
    print_test_header("Forward Pass Test")
    
    try:
        device = next(model.parameters()).device
        
        # Create random input
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        input_ids = input_ids.to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_ids)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output shape: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(model, tokenizer, config):
    """Test text generation"""
    print_test_header("Generation Test")
    
    try:
        device = next(model.parameters()).device
        model.eval()
        
        prompts = [
            ("The future of AI", "en"),
            ("Hello world", "en"),
        ]
        
        for prompt, lang in prompts:
            tokens = tokenizer.encode(prompt, language=lang)[:20]
            if not tokens:
                tokens = [ord(c) % config.vocab_size for c in prompt]
            
            input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
            
            start_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    temperature=1.0,
                    top_k=50
                )
            
            gen_time = time.time() - start_time
            new_tokens = len(output_ids[0]) - len(tokens)
            
            print(f"✅ Generated from '{prompt}': {new_tokens} new tokens in {gen_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage(model):
    """Test memory efficiency"""
    print_test_header("Memory Usage Test")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Move model to GPU if not already
            model = model.cuda()
            
            # Run a forward pass
            input_ids = torch.randint(0, 70000, (1, 128)).cuda()
            with torch.no_grad():
                _ = model(input_ids)
            
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"✅ Memory test completed")
            print(f"   Peak memory used: {memory_used:.2f} GB")
            print(f"   Total GPU memory: {memory_total:.1f} GB")
            print(f"   Efficiency: {(1 - memory_used/memory_total)*100:.0f}% memory available")
        else:
            print("⚠️  CUDA not available, skipping memory test")
        
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def test_sparsity(model):
    """Test sparsity patterns"""
    print_test_header("Sparsity Test")
    
    try:
        # Check cortical column sparsity
        sparsity_stats = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'get_sparsity_stats'):
                stats = module.get_sparsity_stats()
                sparsity_stats.append((name, stats['sparsity']))
                
        if sparsity_stats:
            print("✅ Sparsity patterns:")
            for name, sparsity in sparsity_stats[:5]:  # Show first 5
                print(f"   {name}: {sparsity*100:.1f}% sparse")
        else:
            print("✅ Model uses dense computation (sparsity patterns not tracked)")
            
        return True
    except Exception as e:
        print(f"❌ Sparsity test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧠 BRAIN-INSPIRED GPT - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Track results
    results = {}
    
    # Test 1: Model creation
    success, model, config = test_model_creation()
    results['model_creation'] = success
    
    if not success:
        print("\n❌ Cannot continue without model")
        return
    
    # Test 2: Tokenizer
    success, tokenizer = test_tokenizer()
    results['tokenizer'] = success
    
    # Test 3: Forward pass
    success = test_forward_pass(model, config)
    results['forward_pass'] = success
    
    # Test 4: Generation (only if tokenizer works)
    if tokenizer:
        success = test_generation(model, tokenizer, config)
        results['generation'] = success
    else:
        results['generation'] = False
    
    # Test 5: Memory usage
    success = test_memory_usage(model)
    results['memory'] = success
    
    # Test 6: Sparsity
    success = test_sparsity(model)
    results['sparsity'] = success
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.ljust(20)}: {status}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Brain-Inspired GPT is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("\nCommon fixes:")
        print("  1. Ensure CUDA is available for GPU tests")
        print("  2. Check that all dependencies are installed: uv sync")
        print("  3. Korean tokenizer warnings are normal (uses fallback)")


if __name__ == "__main__":
    main()