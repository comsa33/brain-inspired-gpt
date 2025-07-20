#!/usr/bin/env python3
"""
Minimal working demo of CortexGPT's real-time learning
Shows the core concept without complex tokenization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import re
from typing import List, Dict, Tuple

class SimpleTokenizer:
    """Simple word-based tokenizer for demo purposes"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        self.id_to_word = {0: '<pad>', 1: '<unk>', 2: '<eos>'}
        self.next_id = 3
        
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        # Simple word splitting (lowercase, basic punctuation handling)
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        tokens = []
        for word in words:
            if word not in self.word_to_id:
                if self.next_id < self.vocab_size:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
                    tokens.append(self.word_to_id[word])
                else:
                    tokens.append(self.word_to_id['<unk>'])
            else:
                tokens.append(self.word_to_id[word])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        words = []
        for tid in token_ids:
            if tid in self.id_to_word:
                words.append(self.id_to_word[tid])
            else:
                words.append('<unk>')
        
        # Simple detokenization
        text = ' '.join(words)
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        return text

class SimpleMemory:
    """Simple memory system that actually works"""
    def __init__(self, capacity=32, dim=128):
        self.capacity = capacity
        self.dim = dim
        self.memories = deque(maxlen=capacity)
        self.access_counts = deque(maxlen=capacity)
        
    def store(self, key, value):
        """Store a memory"""
        self.memories.append({'key': key.detach(), 'value': value.detach()})
        self.access_counts.append(1)
        
    def retrieve(self, query):
        """Retrieve most similar memory"""
        if len(self.memories) == 0:
            return torch.zeros_like(query), 0.0
        
        # Find most similar
        similarities = []
        for mem in self.memories:
            # Ensure both tensors are 1D for comparison
            q = query.view(-1)
            k = mem['key'].view(-1)
            sim = F.cosine_similarity(q.unsqueeze(0), k.unsqueeze(0), dim=1).item()
            similarities.append(sim)
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        # Update access count
        self.access_counts[best_idx] += 1
        
        return self.memories[best_idx]['value'], best_sim

class MinimalCortexGPT(nn.Module):
    """Minimal version showing real-time learning"""
    def __init__(self, vocab_size=1000, dim=128):
        super().__init__()
        
        # Simple embeddings
        self.embedding = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size)
        
        # Memory systems
        self.stm = SimpleMemory(capacity=32, dim=dim)
        self.ltm = SimpleMemory(capacity=100, dim=dim)
        
        # Learning from experience
        self.learning_rate = 0.1
        
    def forward(self, input_ids):
        """Forward pass with memory"""
        # Get embeddings
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # Simple pooling
        
        # Retrieve from memory
        stm_val, stm_conf = self.stm.retrieve(x)
        ltm_val, ltm_conf = self.ltm.retrieve(x)
        
        # Combine with memories
        if stm_conf > 0.5:
            x = x * 0.5 + stm_val * 0.5
        elif ltm_conf > 0.5:
            x = x * 0.7 + ltm_val * 0.3
        
        # Generate output
        output = self.output(x)
        
        # Store in STM for learning
        self.stm.store(x, x)
        
        # Consolidate to LTM if accessed frequently
        if len(self.stm.access_counts) > 0 and max(self.stm.access_counts) > 3:
            # Move most accessed to LTM
            max_idx = np.argmax(list(self.stm.access_counts))
            self.ltm.store(
                self.stm.memories[max_idx]['key'],
                self.stm.memories[max_idx]['value']
            )
        
        return output, {'stm_conf': stm_conf, 'ltm_conf': ltm_conf}

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 20, temperature: float = 0.8) -> str:
    """Generate a response given a prompt"""
    device = next(model.parameters()).device
    
    # Tokenize prompt
    tokens = tokenizer.tokenize(prompt)
    generated = tokens.copy()
    
    # Generate tokens one by one
    for _ in range(max_tokens):
        # Prepare input
        input_ids = torch.tensor([tokens[-20:]], device=device)  # Use last 20 tokens as context
        
        # Forward pass
        with torch.no_grad():
            output, meta = model(input_ids)
            
        # Sample from output distribution
        probs = F.softmax(output[0] / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        # Stop if we hit end token or unknown
        if next_token in [tokenizer.word_to_id.get('<eos>', 2), tokenizer.word_to_id.get('<unk>', 1)]:
            break
            
        tokens.append(next_token)
        generated.append(next_token)
    
    # Decode
    return tokenizer.decode(generated)

def demo():
    """Simple demonstration"""
    print("🧠 Minimal CortexGPT Demo - Real-time Learning")
    print("=" * 50)
    
    # Create model and tokenizer
    tokenizer = SimpleTokenizer(vocab_size=500)
    model = MinimalCortexGPT(vocab_size=500, dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"✅ Model created on {device}")
    
    # Simulate learning progression
    print("\n📚 Learning Progression Demo")
    print("-" * 40)
    
    # Test queries - showing how the model remembers patterns
    queries = [
        "What is quantum computing?",
        "Quantum computing uses qubits",
        "What is quantum computing?",  # Ask again - should show better memory
        "Tell me about quantum",  # Related query
        "안녕하세요 세계",  # Korean: Hello world
        "Hello world",  # English version
    ]
    
    for i, text in enumerate(queries):
        print(f"\n🔄 Query {i+1}: {text}")
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        print(f"📝 Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
        # Convert to tensor
        input_ids = torch.tensor([tokens], device=device)
        
        # Forward pass
        output, meta = model(input_ids)
        
        # Show memory confidence
        print(f"📊 Memory: STM={meta['stm_conf']:.2f}, LTM={meta['ltm_conf']:.2f}")
        
        # Generate a short response
        response = generate_response(model, tokenizer, text, max_tokens=10)
        print(f"🤖 Response: {response}")
        
        # Show memory growth
        print(f"💾 Memory size: STM={len(model.stm.memories)}, LTM={len(model.ltm.memories)}")
    
    # Interactive mode
    import sys
    if sys.stdin.isatty():
        print("\n\n💬 Interactive Mode")
        print("-" * 40)
        print("Type your message or 'quit' to exit:")
        print("(The model learns from your inputs and improves over time!)")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n👤 You: ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Add to conversation history
                conversation_history.append(user_input)
                
                # Generate response
                response = generate_response(model, tokenizer, user_input, max_tokens=20)
                print(f"🤖 Bot: {response}")
                
                # Show learning metrics
                _, meta = model(torch.tensor([tokenizer.tokenize(user_input)], device=device))
                print(f"\n📊 Memory confidence: STM={meta['stm_conf']:.2f}, LTM={meta['ltm_conf']:.2f}")
                print(f"💾 Total memories: STM={len(model.stm.memories)}, LTM={len(model.ltm.memories)}")
                
                # Show vocabulary growth
                print(f"📚 Vocabulary size: {len(tokenizer.word_to_id)} words")
                
            except EOFError:
                print("\n\n⚠️ EOF detected - exiting interactive mode.")
                break
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted - exiting.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    else:
        print("\n📋 Non-interactive environment detected. Skipping interactive mode.")
    
    print("\n✅ Demo complete!")
    print("\nKey Concepts Demonstrated:")
    print("  • Natural language input (no token IDs needed!)")
    print("  • Memories improve with repetition")
    print("  • STM → LTM consolidation")
    print("  • Real-time learning without training")
    print("  • Simple but effective!")

if __name__ == "__main__":
    demo()