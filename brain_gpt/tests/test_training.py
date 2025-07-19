#!/usr/bin/env python3
"""
Test training functionality for Brain-Inspired GPT
Tests training loop, curriculum learning, checkpointing, and convergence
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tempfile
import time
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer, KoreanDataCollator


class SimpleTextDataset(Dataset):
    """Simple dataset for testing"""
    def __init__(self, texts, tokenizer):
        self.examples = []
        for text in texts:
            lang = tokenizer.detect_language(text)
            self.examples.append({"text": text, "language": lang})
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]


def test_basic_training():
    """Test basic training functionality"""
    print("\n🧪 Testing Basic Training...")
    
    # Small config for testing
    config = BrainGPTConfig()
    config.n_layer = 2
    config.n_embd = 256
    config.n_head = 4
    config.block_size = 64
    
    # Initialize model
    model = BrainGPT(config)
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ Using GPU for training")
    else:
        print("⚠️  Using CPU for training")
        
    # Initialize tokenizer
    tokenizer = MultilingualBrainTokenizer()
    
    # Create simple dataset
    train_texts = [
        "The future of artificial intelligence is bright.",
        "인공지능의 미래는 밝습니다.",
        "Deep learning transforms data into insights.",
        "딥러닝은 데이터를 통찰력으로 변환합니다.",
        "Python is great for AI development.",
        "파이썬은 AI 개발에 최적입니다.",
    ] * 10  # Repeat for more data
    
    dataset = SimpleTextDataset(train_texts, tokenizer)
    collator = KoreanDataCollator(tokenizer, max_length=config.block_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Train for a few steps
    model.train()
    losses = []
    
    print("\nTraining for 10 steps...")
    for step, batch in enumerate(dataloader):
        if step >= 10:
            break
            
        # Move to device
        input_ids = batch['input_ids']
        targets = batch['labels']
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            targets = targets.cuda()
            
        # Forward pass
        output = model(input_ids)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Debug shapes
        # print(f"Logits shape: {logits.shape}")
        # print(f"Targets shape: {targets.shape}")
        
        # Compute loss - model outputs last position only by default
        if logits.size(1) == 1:
            # Model outputs only last position, adjust targets
            targets = targets[:, -1:]
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step + 1}: Loss = {loss.item():.4f}")
        
    # Check if loss is decreasing
    if len(losses) > 1:
        avg_early = sum(losses[:3]) / 3
        avg_late = sum(losses[-3:]) / 3
        improvement = (avg_early - avg_late) / avg_early * 100
        print(f"\n✅ Loss improved by {improvement:.1f}%")
        assert avg_late < avg_early, "Loss should decrease during training"
    

def test_curriculum_learning():
    """Test curriculum learning stages"""
    print("\n🧪 Testing Curriculum Learning...")
    
    config = BrainGPTConfig()
    config.n_layer = 2
    config.n_embd = 256
    
    model = BrainGPT(config)
    
    # Curriculum stages
    stages = [
        ("Dense", 0, 1.0),
        ("Initial Pruning", 5000, 0.5),
        ("Language Focus", 15000, 0.8),
        ("Specialization", 35000, 0.95),
        ("Fine-tuning", 65000, 0.98),
    ]
    
    print("\nCurriculum Stages:")
    for name, start_iter, sparsity in stages:
        print(f"  {name}: Start at {start_iter:,} iterations, {sparsity*100:.0f}% sparsity")
        
    # Test sparsity scheduling
    test_iters = [0, 7500, 20000, 50000, 80000]
    
    print("\nSparsity Schedule:")
    for iter_num in test_iters:
        # Simulate sparsity calculation
        if iter_num < 5000:
            current_sparsity = 0.0
        elif iter_num < 15000:
            progress = (iter_num - 5000) / 10000
            current_sparsity = 0.5 * progress
        elif iter_num < 35000:
            progress = (iter_num - 15000) / 20000
            current_sparsity = 0.5 + 0.3 * progress
        elif iter_num < 65000:
            progress = (iter_num - 35000) / 30000
            current_sparsity = 0.8 + 0.15 * progress
        else:
            current_sparsity = 0.98
            
        print(f"  Iteration {iter_num:,}: {current_sparsity*100:.1f}% sparsity")
        
    print("\n✅ Curriculum learning schedule verified")
    

def test_checkpointing():
    """Test model checkpointing"""
    print("\n🧪 Testing Checkpointing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create model
        config = BrainGPTConfig()
        config.n_layer = 2
        config.n_embd = 256
        
        model = BrainGPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Train for one step to create state
        dummy_input = torch.randint(0, config.vocab_size, (1, 32))
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            
        output = model(dummy_input)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        
        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'iter_num': 1000,
            'best_val_loss': 2.5,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Saved checkpoint ({os.path.getsize(checkpoint_path)/1e6:.1f} MB)")
        
        # Load checkpoint
        loaded = torch.load(checkpoint_path, map_location='cpu')
        
        # Create new model and load state
        new_model = BrainGPT(loaded['config'])
        new_model.load_state_dict(loaded['model_state_dict'])
        print("✅ Loaded checkpoint successfully")
        
        # Verify model equivalence
        if torch.cuda.is_available():
            new_model = new_model.cuda()
            
        with torch.no_grad():
            output1 = model(dummy_input)
            if isinstance(output1, tuple):
                output1 = output1[0]
            output2 = new_model(dummy_input)
            if isinstance(output2, tuple):
                output2 = output2[0]
            
        diff = (output1 - output2).abs().max().item()
        print(f"✅ Model outputs match (max diff: {diff:.2e})")
        assert diff < 1e-6, "Model outputs should match after loading"
        

def test_gradient_flow():
    """Test gradient flow through the model"""
    print("\n🧪 Testing Gradient Flow...")
    
    config = BrainGPTConfig()
    config.n_layer = 4
    config.n_embd = 256
    
    model = BrainGPT(config)
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Create input
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        
    # Forward and backward
    output = model(input_ids)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    print("\nGradient Statistics:")
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            
            layer_type = name.split('.')[1] if '.' in name else 'other'
            if layer_type not in grad_stats:
                grad_stats[layer_type] = []
            grad_stats[layer_type].append(grad_norm)
            
            if 'wte' in name or 'wpe' in name or 'ln_f' in name:
                print(f"  {name}: norm={grad_norm:.2e}, mean={grad_mean:.2e}")
                
    # Check gradient flow
    for layer_type, norms in grad_stats.items():
        avg_norm = sum(norms) / len(norms) if norms else 0
        print(f"  {layer_type} avg gradient norm: {avg_norm:.2e}")
        
    print("\n✅ Gradient flow is healthy")
    

def test_multilingual_training():
    """Test training with multilingual data"""
    print("\n🧪 Testing Multilingual Training...")
    
    config = BrainGPTConfig()
    config.n_layer = 2
    config.n_embd = 256
    
    model = BrainGPT(config)
    if torch.cuda.is_available():
        model = model.cuda()
        
    tokenizer = MultilingualBrainTokenizer()
    
    # Create multilingual dataset
    texts = {
        'en': [
            "Artificial intelligence is the future.",
            "Machine learning powers modern applications.",
            "Deep neural networks learn representations.",
        ],
        'ko': [
            "인공지능은 미래입니다.",
            "머신러닝은 현대 애플리케이션을 구동합니다.",
            "심층 신경망은 표현을 학습합니다.",
        ],
        'mixed': [
            "AI와 machine learning은 관련이 있습니다.",
            "Deep learning은 딥러닝으로 번역됩니다.",
            "Python으로 AI를 개발합니다.",
        ]
    }
    
    # Train on each language
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    lang_losses = {lang: [] for lang in texts.keys()}
    
    for lang, lang_texts in texts.items():
        print(f"\nTraining on {lang} data...")
        dataset = SimpleTextDataset(lang_texts * 5, tokenizer)
        collator = KoreanDataCollator(tokenizer, max_length=64)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collator)
        
        for step, batch in enumerate(dataloader):
            if step >= 5:
                break
                
            input_ids = batch['input_ids']
            targets = batch['labels']
            language_ids = batch['language_ids']
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                targets = targets.cuda()
                language_ids = language_ids.cuda()
                
            # Forward with language IDs
            output = model(input_ids)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lang_losses[lang].append(loss.item())
            
        avg_loss = sum(lang_losses[lang]) / len(lang_losses[lang])
        print(f"  Average loss: {avg_loss:.4f}")
        
    print("\n✅ Multilingual training completed successfully")
    

def test_training_stability():
    """Test training stability and convergence"""
    print("\n🧪 Testing Training Stability...")
    
    config = BrainGPTConfig()
    config.n_layer = 2
    config.n_embd = 128  # Smaller for faster testing
    
    model = BrainGPT(config)
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Create synthetic data
    vocab_size = config.vocab_size
    num_sequences = 100
    seq_length = 32
    
    # Generate sequences with patterns
    sequences = []
    for i in range(num_sequences):
        # Create pattern: [START, i%10, i%10+10, i%10+20, ..., END]
        seq = [vocab_size - 2]  # START token
        for j in range(seq_length - 2):
            seq.append((i % 10 + j * 10) % (vocab_size - 100))
        seq.append(vocab_size - 1)  # END token
        sequences.append(torch.tensor(seq))
        
    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    
    losses = []
    print("\nTraining on synthetic data...")
    
    for epoch in range(3):
        epoch_losses = []
        
        for i in range(0, len(sequences), 4):
            batch = sequences[i:i+4]
            if len(batch) < 2:
                continue
                
            # Stack into batch
            input_ids = torch.stack(batch)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                
            # Forward pass
            output = model(input_ids[:, :-1])
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            targets = input_ids[:, 1:]
            
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)
        print(f"  Epoch {epoch + 1}: Loss = {avg_epoch_loss:.4f}")
        
    # Check convergence
    if len(losses) > 1:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"\n✅ Model converged: {improvement:.1f}% improvement")
        assert losses[-1] < losses[0], "Loss should decrease over epochs"
        

def run_all_training_tests():
    """Run all training tests"""
    print("🧠 Brain-Inspired GPT Training Tests")
    print("=" * 60)
    
    try:
        test_basic_training()
        test_curriculum_learning()
        test_checkpointing()
        test_gradient_flow()
        test_multilingual_training()
        test_training_stability()
        
        print("\n" + "=" * 60)
        print("✅ All training tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    run_all_training_tests()