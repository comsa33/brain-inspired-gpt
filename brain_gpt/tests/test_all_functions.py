#!/usr/bin/env python3
"""
Comprehensive test suite for Brain-Inspired GPT
Tests all major functions: data loading, training, inference, and performance
"""

import os
import sys
import torch
import pytest
import tempfile
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer, KoreanDataCollator
from core.sparse_modules import CorticalColumnLinear, SpikingCorticalColumn
from training.prepare_korean_data import KoreanDatasetPreprocessor


class TestBrainGPT:
    """Test suite for Brain-Inspired GPT"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = BrainGPTConfig()
        config.n_layer = 4  # Smaller for testing
        config.n_embd = 512
        config.n_head = 8
        config.block_size = 128
        return config
    
    @pytest.fixture
    def model(self, config):
        """Create test model"""
        return BrainGPT(config)
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer"""
        return MultilingualBrainTokenizer(cache_dir="./test_tokenizer_cache")
    
    def test_model_initialization(self, model, config):
        """Test model initialization and parameter count"""
        print("\n🧪 Testing Model Initialization...")
        
        # Check model structure
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'lm_head')
        assert len(model.transformer.h) == config.n_layer
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model initialized with {total_params/1e6:.1f}M parameters")
        assert total_params > 0
        
        # Check sparsity
        effective_params = total_params * (1 - config.sparsity_base)
        print(f"✅ Effective parameters (with sparsity): {effective_params/1e6:.1f}M")
        
    def test_tokenizer_basic(self, tokenizer):
        """Test basic tokenizer functionality"""
        print("\n🧪 Testing Tokenizer...")
        
        # Test English
        text_en = "Hello, world!"
        tokens_en = tokenizer.encode(text_en, language='en')
        decoded_en = tokenizer.decode(tokens_en)
        print(f"✅ English: '{text_en}' -> {len(tokens_en)} tokens")
        assert len(tokens_en) > 0
        
        # Test Korean
        text_ko = "안녕하세요, 세계!"
        tokens_ko = tokenizer.encode(text_ko, language='ko')
        decoded_ko = tokenizer.decode(tokens_ko)
        print(f"✅ Korean: '{text_ko}' -> {len(tokens_ko)} tokens")
        assert len(tokens_ko) > 0
        
        # Test mixed
        text_mixed = "Hello 안녕하세요!"
        tokens_mixed = tokenizer.encode(text_mixed)
        decoded_mixed = tokenizer.decode(tokens_mixed)
        print(f"✅ Mixed: '{text_mixed}' -> {len(tokens_mixed)} tokens")
        assert len(tokens_mixed) > 0
        
    def test_language_detection(self, tokenizer):
        """Test language detection"""
        print("\n🧪 Testing Language Detection...")
        
        test_cases = [
            ("Hello world", "en"),
            ("안녕하세요", "ko"),
            ("def hello():", "code"),
            ("AI와 machine learning", "ko"),  # Mixed but predominantly Korean
        ]
        
        for text, expected_lang in test_cases:
            detected = tokenizer.detect_language(text)
            print(f"✅ '{text}' detected as: {detected}")
            assert detected == expected_lang
            
    def test_forward_pass(self, model, tokenizer, config):
        """Test model forward pass"""
        print("\n🧪 Testing Forward Pass...")
        
        # Create sample input
        text = "The future of AI is"
        tokens = tokenizer.encode(text)[:config.block_size]
        input_ids = torch.tensor(tokens).unsqueeze(0)
        
        # Forward pass
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            output = model(input_ids)
            
        assert output.shape == (1, len(tokens), config.vocab_size)
        print(f"✅ Forward pass successful: {output.shape}")
        
    def test_generation(self, model, tokenizer, config):
        """Test text generation"""
        print("\n🧪 Testing Text Generation...")
        
        device = next(model.parameters()).device
        model.eval()
        
        prompts = [
            ("The future of AI", "en"),
            ("인공지능의 미래", "ko"),
        ]
        
        for prompt, lang in prompts:
            tokens = tokenizer.encode(prompt, language=lang)
            input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=20,
                    temperature=0.8,
                    language_id=lang
                )
                
            generated = tokenizer.decode(output[0].tolist())
            print(f"✅ Generated from '{prompt}': {len(output[0])} tokens")
            assert len(output[0]) > len(tokens)
            
    def test_sparse_modules(self):
        """Test sparse module functionality"""
        print("\n🧪 Testing Sparse Modules...")
        
        # Test CorticalColumnLinear
        linear = CorticalColumnLinear(512, 256, num_columns=8)
        x = torch.randn(1, 10, 512)
        output = linear(x)
        assert output.shape == (1, 10, 256)
        print("✅ CorticalColumnLinear forward pass successful")
        
        # Check sparsity
        sparsity = linear.get_sparsity_stats()
        print(f"✅ Sparsity: {sparsity['sparsity']:.1%}")
        assert sparsity['sparsity'] > 0.5
        
    def test_data_collator(self, tokenizer):
        """Test Korean data collator"""
        print("\n🧪 Testing Data Collator...")
        
        collator = KoreanDataCollator(tokenizer, max_length=128)
        
        examples = [
            {"text": "Hello world", "language": "en"},
            {"text": "안녕하세요", "language": "ko"},
        ]
        
        batch = collator(examples)
        
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'language_ids' in batch
        assert batch['input_ids'].shape == (2, 128)
        print("✅ Data collator working correctly")
        
    def test_memory_efficiency(self, model, config):
        """Test memory efficiency"""
        print("\n🧪 Testing Memory Efficiency...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model = model.cuda()
            
            # Run forward pass
            input_ids = torch.randint(0, config.vocab_size, (1, config.block_size)).cuda()
            
            with torch.no_grad():
                output = model(input_ids)
                
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"✅ Peak memory usage: {memory_used:.2f} GB")
            
            # Check if memory usage is reasonable
            expected_memory = (sum(p.numel() for p in model.parameters()) * 4) / 1e9  # 4 bytes per param
            efficiency = expected_memory / memory_used
            print(f"✅ Memory efficiency: {efficiency:.1%}")
        else:
            print("⚠️  CUDA not available, skipping memory test")
            
    def test_training_step(self, model, tokenizer, config):
        """Test single training step"""
        print("\n🧪 Testing Training Step...")
        
        device = next(model.parameters()).device
        model.train()
        
        # Create batch
        texts = ["The future is", "AI will"]
        batch_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)[:config.block_size]
            tokens += [0] * (config.block_size - len(tokens))  # Pad
            batch_tokens.append(tokens)
            
        input_ids = torch.tensor(batch_tokens).to(device)
        targets = input_ids.clone()
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        assert loss.item() > 0
        print(f"✅ Training loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.parameters() if p.requires_grad)
        assert has_grads
        print("✅ Gradients computed successfully")
        
    @pytest.mark.slow
    def test_performance_benchmarks(self, model, tokenizer, config):
        """Test performance benchmarks"""
        print("\n🧪 Testing Performance Benchmarks...")
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping performance test")
            return
            
        model = model.cuda().eval()
        
        # Warm up
        input_ids = torch.randint(0, config.vocab_size, (1, config.block_size)).cuda()
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_ids)
                
        # Benchmark
        num_iterations = 100
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_ids)
                
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        tokens_per_second = (num_iterations * config.block_size) / total_time
        print(f"✅ Throughput: {tokens_per_second:.0f} tokens/sec")
        print(f"✅ Latency: {total_time/num_iterations*1000:.1f} ms/batch")
        
    def test_korean_data_preprocessing(self):
        """Test Korean data preprocessing"""
        print("\n🧪 Testing Korean Data Preprocessing...")
        
        preprocessor = KoreanDatasetPreprocessor()
        
        # Test text cleaning
        dirty_text = "안녕하세요!!!   세계~~~ ㅋㅋㅋ"
        clean_text = preprocessor.clean_text(dirty_text)
        print(f"✅ Cleaned: '{dirty_text}' -> '{clean_text}'")
        
        # Test sentence splitting
        text = "안녕하세요. 오늘 날씨가 좋네요! AI는 미래입니다."
        sentences = preprocessor.split_sentences(text)
        print(f"✅ Split into {len(sentences)} sentences")
        assert len(sentences) == 3
        
    def test_end_to_end_pipeline(self, config):
        """Test complete pipeline from data to generation"""
        print("\n🧪 Testing End-to-End Pipeline...")
        
        # Initialize components
        tokenizer = MultilingualBrainTokenizer()
        model = BrainGPT(config)
        
        if torch.cuda.is_available():
            model = model.cuda()
            
        # Prepare data
        train_texts = [
            "Artificial intelligence is transforming the world.",
            "인공지능은 세상을 변화시키고 있습니다.",
            "def hello_world():\n    print('Hello!')",
        ]
        
        # Train for a few steps
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        for i, text in enumerate(train_texts * 2):
            tokens = tokenizer.encode(text)[:config.block_size]
            if len(tokens) < 2:
                continue
                
            input_ids = torch.tensor(tokens[:-1]).unsqueeze(0)
            targets = torch.tensor(tokens[1:]).unsqueeze(0)
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                targets = targets.cuda()
                
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 3 == 0:
                print(f"✅ Step {i}: Loss = {loss.item():.4f}")
                
        # Test generation after training
        model.eval()
        test_prompt = "AI is"
        tokens = tokenizer.encode(test_prompt)
        input_ids = torch.tensor(tokens).unsqueeze(0)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=10)
            
        generated = tokenizer.decode(output[0].tolist())
        print(f"✅ Generated: '{generated}'")
        
        print("\n✅ End-to-end pipeline test completed successfully!")


def run_all_tests():
    """Run all tests with detailed output"""
    print("🧠 Brain-Inspired GPT Comprehensive Test Suite")
    print("=" * 60)
    
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_all_tests()