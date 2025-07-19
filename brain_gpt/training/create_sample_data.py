#!/usr/bin/env python3
"""
Create sample training data for Brain-Inspired GPT
Creates both English and Korean synthetic data for testing
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.multilingual_tokenizer import MultilingualBrainTokenizer


def create_sample_data():
    """Create sample training data files"""
    print("🔧 Creating sample training data...")
    
    # Initialize tokenizer
    tokenizer = MultilingualBrainTokenizer()
    
    # Create data directories
    data_dir = Path("data")
    en_dir = data_dir / "openwebtext"
    ko_dir = data_dir / "korean"
    
    en_dir.mkdir(parents=True, exist_ok=True)
    ko_dir.mkdir(parents=True, exist_ok=True)
    
    # English sample texts
    en_texts = [
        "The future of artificial intelligence is bright. Machine learning continues to advance rapidly.",
        "Deep neural networks have revolutionized computer vision and natural language processing.",
        "Python is the most popular programming language for data science and machine learning.",
        "Transformers have become the dominant architecture in modern NLP applications.",
        "The brain is an incredibly efficient information processing system.",
        "Neurons communicate through electrical and chemical signals.",
        "Sparse representations are key to biological intelligence.",
        "Energy efficiency is crucial for sustainable AI development.",
        "Large language models require significant computational resources.",
        "Research in brain-inspired computing is advancing rapidly.",
    ] * 100  # Repeat to create more data
    
    # Korean sample texts
    ko_texts = [
        "인공지능의 미래는 매우 밝습니다. 머신러닝 기술이 빠르게 발전하고 있습니다.",
        "딥러닝은 컴퓨터 비전과 자연어 처리 분야에 혁명을 일으켰습니다.",
        "파이썬은 데이터 과학과 머신러닝을 위한 가장 인기 있는 프로그래밍 언어입니다.",
        "트랜스포머는 현대 자연어 처리의 주요 아키텍처가 되었습니다.",
        "뇌는 놀라울 정도로 효율적인 정보 처리 시스템입니다.",
        "뉴런은 전기적, 화학적 신호를 통해 소통합니다.",
        "희소 표현은 생물학적 지능의 핵심입니다.",
        "에너지 효율성은 지속 가능한 AI 개발에 매우 중요합니다.",
        "대규모 언어 모델은 상당한 컴퓨팅 리소스를 필요로 합니다.",
        "뇌에서 영감을 받은 컴퓨팅 연구가 빠르게 발전하고 있습니다.",
    ] * 100  # Repeat to create more data
    
    # Tokenize English data
    print("📝 Tokenizing English data...")
    en_tokens = []
    for text in en_texts:
        tokens = tokenizer.encode(text, language='en')
        en_tokens.extend(tokens)
    
    # Save English data
    en_array = np.array(en_tokens, dtype=np.uint16)
    en_train_path = en_dir / "train.bin"
    en_array.tofile(en_train_path)
    print(f"✅ Created {en_train_path} with {len(en_array):,} tokens")
    
    # Also create validation data (smaller)
    en_val_array = en_array[:10000]  # First 10k tokens for validation
    en_val_path = en_dir / "val.bin"
    en_val_array.tofile(en_val_path)
    print(f"✅ Created {en_val_path} with {len(en_val_array):,} tokens")
    
    # Tokenize Korean data
    print("📝 Tokenizing Korean data...")
    ko_tokens = []
    for text in ko_texts:
        tokens = tokenizer.encode(text, language='ko')
        ko_tokens.extend(tokens)
    
    # Save Korean data
    ko_array = np.array(ko_tokens, dtype=np.uint16)
    ko_train_path = ko_dir / "train_korean.bin"
    ko_array.tofile(ko_train_path)
    print(f"✅ Created {ko_train_path} with {len(ko_array):,} tokens")
    
    # Also create validation data
    ko_val_array = ko_array[:10000]  # First 10k tokens for validation
    ko_val_path = ko_dir / "val_korean.bin"
    ko_val_array.tofile(ko_val_path)
    print(f"✅ Created {ko_val_path} with {len(ko_val_array):,} tokens")
    
    # Create metadata
    metadata = {
        "en_train_tokens": len(en_array),
        "en_val_tokens": len(en_val_array),
        "ko_train_tokens": len(ko_array),
        "ko_val_tokens": len(ko_val_array),
        "vocab_size": tokenizer.get_vocab_size(),
    }
    
    print("\n📊 Dataset Summary:")
    print(f"   English train: {metadata['en_train_tokens']:,} tokens")
    print(f"   English val: {metadata['en_val_tokens']:,} tokens")
    print(f"   Korean train: {metadata['ko_train_tokens']:,} tokens")
    print(f"   Korean val: {metadata['ko_val_tokens']:,} tokens")
    print(f"   Vocabulary size: {metadata['vocab_size']:,}")
    
    return metadata


def create_tiny_test_data():
    """Create tiny dataset for quick testing"""
    print("\n🔧 Creating tiny test dataset...")
    
    # Initialize tokenizer
    tokenizer = MultilingualBrainTokenizer()
    
    # Create directories
    data_dir = Path("data/tiny")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Very simple data
    texts = [
        "Hello world! This is a test.",
        "안녕하세요! 테스트입니다.",
        "AI is amazing. 인공지능은 놀랍습니다.",
    ] * 10
    
    tokens = []
    for text in texts:
        tokens.extend(tokenizer.encode(text))
    
    # Save
    array = np.array(tokens, dtype=np.uint16)
    train_path = data_dir / "train.bin"
    array.tofile(train_path)
    
    print(f"✅ Created {train_path} with {len(array):,} tokens")
    

if __name__ == "__main__":
    # Create sample data
    metadata = create_sample_data()
    
    # Also create tiny test data
    create_tiny_test_data()
    
    print("\n✅ Sample data creation complete!")
    print("\n🚀 You can now run training with:")
    print("   uv run brain_gpt/training/train_brain_gpt.py")