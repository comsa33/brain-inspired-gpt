#!/usr/bin/env python3
"""
Simple overfitting test for CortexGPT
Tests if the model can memorize a small dataset
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortexgpt.models.cortex_gpt import CortexGPT
from cortexgpt.models.realtime_cortex import RealTimeCortexGPT, AdvancedMemoryConfig
from cortexgpt.tokenization.multilingual_tokenizer import MultilingualTokenizer


class SimpleDataset(Dataset):
    """Simple dataset for overfitting test"""
    
    def __init__(self, texts, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []
        
        # Combine all texts into one long sequence
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.special_tokens.get('<eos>', 2))  # Add EOS between texts
        
        # Create overlapping sequences
        for i in range(0, len(all_tokens) - block_size + 1, block_size // 4):
            self.data.append(all_tokens[i:i + block_size])
        
        # If we still don't have enough data, repeat the sequences
        if len(self.data) < 10:
            original_data = self.data.copy()
            for _ in range(10):
                self.data.extend(original_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Input is all tokens except last, target is all tokens except first
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids


def test_overfit():
    """Test if model can overfit on a small dataset"""
    
    # Small corpus for testing
    test_texts = [
        "The cat sat on the mat.",
        "안녕하세요. 오늘 날씨가 좋습니다.",
        "Machine learning is amazing.",
        "인공지능은 미래입니다.",
        "def hello(): return 'world'",
    ]
    
    print("🧪 Overfitting Test for CortexGPT")
    print(f"Test texts: {len(test_texts)}")
    
    # Create tokenizer
    print("\n1. Creating tokenizer...")
    tokenizer = MultilingualTokenizer(vocab_size=5000)
    
    # Train tokenizer on test texts + some extras for better vocabulary
    train_texts = test_texts * 20  # Repeat to get more samples
    train_texts.extend([
        "The quick brown fox jumps over the lazy dog.",
        "Python is a programming language.",
        "한국어는 아름다운 언어입니다.",
        "기계 학습은 재미있습니다.",
    ] * 10)
    
    tokenizer.learn_bpe(train_texts, verbose=True)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Create dataset
    print("\n2. Creating dataset...")
    dataset = SimpleDataset(test_texts, tokenizer, block_size=32)  # Smaller block size
    print(f"Dataset size: {len(dataset)} sequences")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty! Check tokenization.")
        return
        
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"Batch size: 2, Total batches: {len(dataloader)}")
    
    # Create model
    print("\n3. Creating model...")
    config = AdvancedMemoryConfig(
        stm_capacity=32,
        ltm_capacity=100,
        archive_capacity=1000
    )
    
    model = RealTimeCortexGPT(config, len(tokenizer.vocab), dim=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop
    print("\n4. Training to overfit...")
    print("=" * 50)
    
    for epoch in range(100):  # Many epochs to ensure overfitting
        total_loss = 0
        model.train()
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            outputs = model(input_ids, real_time=False)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        # Generate samples every 10 epochs
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}, Loss: {avg_loss:.4f}")
            
            # Test generation
            model.eval()
            with torch.no_grad():
                for i, text in enumerate(test_texts[:3]):
                    print(f"\nOriginal: {text}")
                    
                    # Encode prompt (first few tokens)
                    tokens = tokenizer.encode(text)
                    prompt_tokens = tokens[:3]  # Use first 3 tokens as prompt
                    prompt_text = tokenizer.decode(prompt_tokens)
                    print(f"Prompt: {prompt_text}")
                    
                    # Generate
                    input_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)
                    
                    generated = prompt_tokens.copy()
                    for _ in range(20):  # Generate 20 tokens
                        with torch.no_grad():
                            outputs = model(input_ids, real_time=False)
                            next_token_logits = outputs[0, -1, :]
                            next_token = torch.argmax(next_token_logits)
                            generated.append(next_token.item())
                            
                            # Update input
                            input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
                            
                            # Stop at EOS if present
                            if next_token.item() == tokenizer.special_tokens.get('<eos>', -1):
                                break
                    
                    generated_text = tokenizer.decode(generated)
                    print(f"Generated: {generated_text}")
            
            print("=" * 50)
    
    print("\n✅ Overfitting test complete!")
    print("\nIf the model successfully memorized the training data,")
    print("it should generate text very similar to the originals.")
    print("\nIf not, there may be issues with:")
    print("- Model architecture")
    print("- Learning rate")
    print("- Tokenization")


if __name__ == "__main__":
    test_overfit()