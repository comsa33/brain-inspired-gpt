#!/usr/bin/env python3
"""
Test multilingual capabilities of Brain-Inspired GPT
Shows how the model handles different languages
"""

import os
import sys
import torch
from pathlib import Path

# Add brain_gpt to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brain_gpt.core.model_brain import BrainGPT
from brain_gpt.core.model_brain_config import BrainGPTConfig
from brain_gpt.core.multilingual_tokenizer import MultilingualBrainTokenizer


def load_latest_checkpoint():
    """Find and load the latest checkpoint"""
    checkpoint_dirs = [
        Path("checkpoints/multilingual"),
        Path("checkpoints"),
    ]
    
    for ckpt_dir in checkpoint_dirs:
        if not ckpt_dir.exists():
            continue
            
        # Look for best checkpoint first
        best_ckpt = ckpt_dir / "multilingual_best.pt"
        if best_ckpt.exists():
            return best_ckpt
            
        best_ckpt = ckpt_dir / "brain_gpt_3090_best.pt"
        if best_ckpt.exists():
            return best_ckpt
            
        # Find latest checkpoint by step number
        ckpts = list(ckpt_dir.glob("*_step*.pt"))
        if ckpts:
            return max(ckpts, key=lambda p: int(p.stem.split('step')[-1]))
    
    return None


def test_generation():
    """Test text generation in multiple languages"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Find checkpoint
    checkpoint_path = load_latest_checkpoint()
    if not checkpoint_path:
        print("❌ No checkpoint found. Please train a model first:")
        print("   uv run brain_gpt/training/train_multilingual.py")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', checkpoint.get('model_config'))
    
    model = BrainGPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize tokenizer
    tokenizer = MultilingualBrainTokenizer()
    
    # Test prompts in different languages
    test_cases = [
        # English prompts
        {
            'prompt': "The future of artificial intelligence is",
            'language': 'en',
            'max_tokens': 50,
            'temperature': 0.8
        },
        {
            'prompt': "Once upon a time in a distant galaxy",
            'language': 'en',
            'max_tokens': 60,
            'temperature': 0.9
        },
        # Korean prompts
        {
            'prompt': "인공지능의 미래는",
            'language': 'ko',
            'max_tokens': 50,
            'temperature': 0.8
        },
        {
            'prompt': "옛날 옛적에 깊은 산속에",
            'language': 'ko',
            'max_tokens': 60,
            'temperature': 0.9
        },
        # Mixed language
        {
            'prompt': "AI와 machine learning의 차이점은",
            'language': 'ko',
            'max_tokens': 50,
            'temperature': 0.7
        },
        # Code-like prompt
        {
            'prompt': "def calculate_fibonacci(n):",
            'language': 'en',
            'max_tokens': 80,
            'temperature': 0.5
        }
    ]
    
    print("\n" + "="*80)
    print("🌐 Multilingual Generation Test")
    print("="*80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}/{len(test_cases)}")
        print(f"Language: {test['language']}")
        print(f"Prompt: {test['prompt']}")
        print("-" * 40)
        
        # Tokenize
        tokens = tokenizer.encode(
            test['prompt'], 
            language=test['language'],
            add_language_markers=True
        )
        
        # Limit initial tokens
        tokens = tokens[:50]
        x = torch.tensor(tokens).unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                x,
                max_new_tokens=test['max_tokens'],
                temperature=test['temperature'],
                top_k=50,
                top_p=0.95
            )
        
        # Decode
        generated_text = tokenizer.decode(output[0].cpu().numpy())
        
        print(f"Generated: {generated_text}")
        
        # Analyze language distribution in output
        output_tokens = output[0].cpu().numpy()
        en_tokens = sum(1 for t in output_tokens if t < tokenizer.base_vocab_size)
        ko_tokens = sum(1 for t in output_tokens 
                       if tokenizer.base_vocab_size <= t < tokenizer.base_vocab_size + tokenizer.korean_vocab_size)
        
        print(f"Token distribution: EN={en_tokens}, KO={ko_tokens}, Other={len(output_tokens)-en_tokens-ko_tokens}")
    
    print("\n" + "="*80)
    print("✅ Testing complete!")
    print("="*80)
    
    # Show model info
    print(f"\nModel info:")
    print(f"- Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"- Layers: {config.n_layer}")
    print(f"- Hidden size: {config.n_embd}")
    print(f"- Vocabulary size: {config.vocab_size}")
    print(f"- Device: {device}")
    
    if 'training_config' in checkpoint:
        train_cfg = checkpoint['training_config']
        print(f"\nTraining info:")
        print(f"- Trained on: {train_cfg.get('data_dirs', 'Unknown')}")
        print(f"- Language sampling: {train_cfg.get('language_sampling', 'Unknown')}")
        print(f"- Steps: {checkpoint.get('step', 'Unknown')}")


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Test multilingual Brain-Inspired GPT')
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint')
    args = parser.parse_args()
    
    if args.checkpoint:
        # Override checkpoint loading
        global load_latest_checkpoint
        load_latest_checkpoint = lambda: Path(args.checkpoint)
    
    test_generation()


if __name__ == "__main__":
    main()