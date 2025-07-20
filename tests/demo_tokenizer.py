#!/usr/bin/env python3
"""
Demo script to test tokenizer functionality
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortexgpt.tokenization.multilingual_tokenizer import MultilingualTokenizer


def demo_tokenizer():
    """Demonstrate tokenizer functionality"""
    
    print("🔤 Tokenizer Demo")
    print("=" * 50)
    
    # Create tokenizer
    tokenizer = MultilingualTokenizer(vocab_size=10000)
    
    # Training corpus with diverse text
    training_texts = [
        # English
        "Hello, world! How are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a high-level programming language.",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "Natural language processing enables computers to understand human language.",
        
        # Korean
        "안녕하세요! 오늘 날씨가 정말 좋네요.",
        "인공지능은 미래의 핵심 기술입니다.",
        "한국어 자연어 처리는 매우 중요한 연구 분야입니다.",
        "파이썬은 배우기 쉬운 프로그래밍 언어입니다.",
        "기계 학습은 데이터로부터 패턴을 학습합니다.",
        "딥러닝은 인공 신경망을 사용하는 기계 학습의 한 분야입니다.",
        
        # Mixed
        "AI (인공지능) is changing the world.",
        "Python과 JavaScript는 인기 있는 프로그래밍 언어입니다.",
        "Machine Learning과 딥러닝의 차이점은 무엇인가요?",
    ] * 10  # Repeat for better BPE learning
    
    print(f"Training tokenizer on {len(training_texts)} texts...")
    tokenizer.learn_bpe(training_texts, verbose=True)
    
    print(f"\n✅ Tokenizer created with vocabulary size: {len(tokenizer.vocab)}")
    
    # Show special tokens
    print("\n📌 Special tokens:")
    for name, token_id in tokenizer.special_tokens.items():
        print(f"  {name}: {token_id}")
    
    # Test tokenization
    test_texts = [
        "Hello, world!",
        "안녕하세요!",
        "Machine learning is amazing.",
        "인공지능은 미래입니다.",
        "def hello(): print('world')",
    ]
    
    print("\n🧪 Tokenization tests:")
    print("-" * 50)
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        print(f"\nOriginal: {text}")
        print(f"Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"Decoded: {decoded}")
        print(f"Token count: {len(tokens)}")
        
        # Show individual tokens
        if len(tokens) <= 15:
            token_strs = []
            for token_id in tokens:
                # Get token string from reverse_vocab mapping
                token_str = tokenizer.reverse_vocab.get(token_id, f"<UNK:{token_id}>")
                if isinstance(token_str, str) and token_str.startswith('##'):
                    token_str = token_str[2:]  # Remove ## prefix
                token_strs.append(repr(token_str))
            print(f"Token strings: {' '.join(token_strs)}")
    
    # Check vocabulary coverage
    print("\n📊 Vocabulary analysis:")
    
    # Count token types
    special_count = sum(1 for token in tokenizer.vocab.keys() if token.startswith('<') and token.endswith('>'))
    korean_count = sum(1 for token in tokenizer.vocab.keys() if any('\uac00' <= c <= '\ud7af' for c in token))
    english_count = sum(1 for token in tokenizer.vocab.keys() if token.isalpha() and all(c < '\u0080' for c in token))
    
    print(f"  Total vocabulary: {len(tokenizer.vocab)}")
    print(f"  Special tokens: {special_count}")
    print(f"  Korean tokens: {korean_count}")
    print(f"  English tokens: {english_count}")
    print(f"  Other tokens: {len(tokenizer.vocab) - special_count - korean_count - english_count}")
    
    # Test unknown token handling
    print("\n🔍 Unknown token handling:")
    unknown_text = "这是中文文本 🚀 Emoji test!"
    tokens = tokenizer.encode(unknown_text)
    decoded = tokenizer.decode(tokens)
    print(f"Original: {unknown_text}")
    print(f"Decoded: {decoded}")
    print(f"Contains <unk>: {'<unk>' in decoded}")


if __name__ == "__main__":
    demo_tokenizer()