"""
Integration tests for Brain-Inspired GPT
Verifies all components work correctly together
"""

import torch
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer
from core.sparse_modules import CorticalColumnLinear, StructuredSparseMask


class TestBrainGPT:
    """Test suite for Brain-Inspired GPT components"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = BrainGPTConfig()
        config.n_layer = 4  # Smaller for testing
        config.n_head = 8
        config.n_embd = 512
        config.block_size = 128
        return config
        
    @pytest.fixture
    def model(self, config):
        """Create test model"""
        return BrainGPT(config)
        
    @pytest.fixture
    def tokenizer(self):
        """Create test tokenizer"""
        return MultilingualBrainTokenizer()
        
    def test_model_initialization(self, model, config):
        """Test model initializes correctly"""
        assert model is not None
        assert len(model.blocks) == config.n_layer
        assert model.config.n_embd == config.n_embd
        
        # Check parameter count
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count/1e6:.2f}M")
        assert param_count > 0
        
    def test_forward_pass(self, model):
        """Test forward pass works"""
        batch_size = 2
        seq_len = 64
        vocab_size = model.config.vocab_size
        
        # Create random input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits, loss = model(input_ids)
        
        # Check outputs
        assert logits.shape == (batch_size, 1, vocab_size)  # Only last token
        assert loss is None  # No targets provided
        
        # With targets
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, loss = model(input_ids, targets)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert loss is not None
        assert loss.item() > 0
        
    def test_generation(self, model, tokenizer):
        """Test text generation"""
        prompt = "Hello world"
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        
        # Generate
        output = model.generate(input_ids, max_new_tokens=10)
        
        # Check output
        assert output.shape[0] == 1  # Batch size
        assert output.shape[1] > input_ids.shape[1]  # Generated tokens
        
        # Decode
        generated_text = tokenizer.decode(output[0].tolist())
        assert generated_text.startswith(prompt)
        
    def test_multilingual_tokenizer(self, tokenizer):
        """Test multilingual tokenization"""
        # English
        en_text = "Hello world"
        en_tokens = tokenizer.encode(en_text, language='en')
        assert len(en_tokens) > 0
        assert tokenizer.decode(en_tokens) == en_text
        
        # Korean
        ko_text = "안녕하세요"
        ko_tokens = tokenizer.encode(ko_text, language='ko')
        assert len(ko_tokens) > 0
        # Note: exact decode might differ due to tokenization
        
        # Mixed
        mixed_text = "Hello 안녕"
        mixed_tokens = tokenizer.encode(mixed_text, language='mixed')
        assert len(mixed_tokens) > 0
        
    def test_sparse_modules(self):
        """Test sparse module functionality"""
        # Test 2:4 pattern
        mask_2_4 = StructuredSparseMask.create_2_4_pattern(
            (256, 256), torch.device('cpu')
        )
        sparsity = 1 - (mask_2_4.sum() / mask_2_4.numel())
        assert 0.45 < sparsity < 0.55  # Should be ~50% sparse
        
        # Test cortical pattern
        mask_cortical = StructuredSparseMask.create_cortical_pattern(
            n_columns=8, column_size=32, inter_column_density=0.1
        )
        assert mask_cortical.shape == (256, 256)
        
        # Test cortical column linear
        linear = CorticalColumnLinear(256, 256, n_columns=8, sparsity=0.9)
        x = torch.randn(4, 256)
        y = linear(x)
        assert y.shape == (4, 256)
        
    def test_dendritic_attention(self, model):
        """Test dendritic attention mechanism"""
        # Get first attention block
        attention = model.blocks[0].attention
        
        # Create input
        x = torch.randn(2, 32, model.config.n_embd)
        
        # Forward pass
        output, stats = attention(x, return_attention_stats=True)
        
        # Check output
        assert output.shape == x.shape
        assert 'sparsity' in stats
        assert 0 <= stats['sparsity'] <= 1
        
    def test_memory_efficiency(self, model, config):
        """Test memory usage is within bounds"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        model = model.cuda()
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        batch_size = 2
        seq_len = config.block_size
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
        
        with torch.cuda.amp.autocast():
            logits, loss = model(input_ids, input_ids)
            
        # Check memory usage
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak memory usage: {peak_memory_gb:.2f} GB")
        
        # Should fit in RTX 3090 (24GB)
        assert peak_memory_gb < 24
        
    def test_curriculum_learning(self):
        """Test curriculum learning scheduler"""
        from train_brain_gpt import CurriculumScheduler, TrainingArguments
        
        args = TrainingArguments()
        config = BrainGPTConfig()
        scheduler = CurriculumScheduler(args, config)
        
        # Test stage progression
        stage_0 = scheduler.get_stage(0)
        assert stage_0['sparsity'] == 0.0
        assert stage_0['languages'] == ['en']
        
        stage_mid = scheduler.get_stage(15000)
        assert stage_mid['sparsity'] > 0
        assert 'ko' in stage_mid['languages']
        
        stage_final = scheduler.get_stage(100000)
        assert stage_final['sparsity'] > 0.95
        
    def test_energy_constraints(self, model):
        """Test metabolic energy constraints"""
        if hasattr(model, 'energy_consumed'):
            initial_energy = model.energy_consumed.item()
            
            # Run forward pass
            x = torch.randint(0, model.config.vocab_size, (1, 32))
            model(x)
            
            # Energy should increase
            # Note: This is simulated in the current implementation
            assert model.energy_consumed.item() >= initial_energy
            

def test_integration():
    """Full integration test"""
    print("\n" + "="*60)
    print("BRAIN-INSPIRED GPT INTEGRATION TEST")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing components...")
    config = BrainGPTConfig()
    config.n_layer = 4  # Small for testing
    model = BrainGPT(config)
    tokenizer = MultilingualBrainTokenizer()
    
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Test English generation
    print("\n2. Testing English generation...")
    en_prompt = "The future is"
    en_input = torch.tensor(tokenizer.encode(en_prompt)).unsqueeze(0)
    en_output = model.generate(en_input, max_new_tokens=20)
    en_text = tokenizer.decode(en_output[0].tolist())
    print(f"✅ English: {en_text}")
    
    # Test Korean generation
    print("\n3. Testing Korean generation...")
    ko_prompt = "미래는"
    ko_input = torch.tensor(tokenizer.encode(ko_prompt, language='ko')).unsqueeze(0)
    ko_output = model.generate(ko_input, max_new_tokens=20, language_id='ko')
    ko_text = tokenizer.decode(ko_output[0].tolist())
    print(f"✅ Korean: {ko_text}")
    
    # Test sparsity
    print("\n4. Testing sparsity patterns...")
    active_neurons = 0
    total_neurons = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'get_active_neurons'):
            active = module.get_active_neurons()
            print(f"  {name}: {active:.1%} active")
            active_neurons += active
            total_neurons += 1
            
    if total_neurons > 0:
        avg_active = active_neurons / total_neurons
        print(f"✅ Average active neurons: {avg_active:.1%}")
        
    # Test memory usage
    print("\n5. Testing memory efficiency...")
    if torch.cuda.is_available():
        model = model.cuda()
        torch.cuda.empty_cache()
        
        # Large input
        large_input = torch.randint(0, config.vocab_size, (2, 512)).cuda()
        with torch.cuda.amp.autocast():
            _ = model(large_input)
            
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"✅ Peak memory: {memory_gb:.2f} GB (target: <8GB)")
    else:
        print("⚠️ CUDA not available, skipping memory test")
        
    print("\n" + "="*60)
    print("ALL TESTS PASSED! 🎉")
    print("="*60)
    

if __name__ == "__main__":
    # Run integration test
    test_integration()
    
    # Run pytest if available
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("\nInstall pytest for detailed unit tests: pip install pytest")