#!/usr/bin/env python3
"""
CortexGPT Text Generation Script
Simple interface for generating text with trained models
"""

import argparse
import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortexgpt.models.realtime_cortex import RealTimeCortexGPT, AdvancedMemoryConfig
from cortexgpt.tokenization.multilingual_tokenizer import MultilingualTokenizer


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generate text from a prompt"""
    model.eval()
    device = next(model.parameters()).device
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    # Generate
    generated = input_ids.copy()
    
    # Get valid token range
    vocab_size = len(tokenizer.vocab)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model output
            outputs = model(input_tensor)
            
            # Get next token probabilities
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Mask out invalid token IDs
            next_token_logits[vocab_size:] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Ensure token is valid
            if next_token >= vocab_size:
                next_token = tokenizer.special_tokens.get('<unk>', 1)
            
            # Check for EOS
            if next_token == tokenizer.special_tokens.get('<eos>', 2):
                break
            
            # Append token
            generated.append(next_token)
            input_tensor = torch.tensor([generated]).to(device)
    
    # Decode
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Generate text with CortexGPT")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for generation")
    parser.add_argument("--max-length", type=int, default=100,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    print("🤖 Loading CortexGPT model...")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Extract configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config
        config = AdvancedMemoryConfig()
    
    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.checkpoint), 'tokenizer.json')
    if os.path.exists(tokenizer_path):
        tokenizer = MultilingualTokenizer()
        tokenizer.load(tokenizer_path)
    else:
        print("⚠️  Tokenizer not found, using default")
        tokenizer = MultilingualTokenizer(vocab_size=50000)
    
    # Create model
    vocab_size = checkpoint.get('vocab_size', len(tokenizer.vocab))
    dim = checkpoint.get('dim', 768)
    
    model = RealTimeCortexGPT(
        config=config,
        vocab_size=vocab_size,
        dim=dim,
        use_hybrid_embeddings=True,  # Always use BGE-M3
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
    
    print(f"✅ Model loaded successfully! (device: {device})")
    print(f"\n📝 Prompt: {args.prompt}")
    print("\n🎯 Generating...\n")
    
    # Generate text
    generated = generate_text(
        model, tokenizer, args.prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    print("=" * 50)
    print(generated)
    print("=" * 50)


if __name__ == "__main__":
    main()