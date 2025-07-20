#!/usr/bin/env python3
"""
Create demo training data for CortexGPT
This creates small demo datasets for quick testing
"""

import json
import os
from pathlib import Path

def create_demo_data():
    """Create demo training data in JSONL format"""
    
    # Demo texts - mix of Korean and English
    demo_texts = [
        # English technical
        {"text": "The quick brown fox jumps over the lazy dog.", "language": "en"},
        {"text": "Machine learning is a subset of artificial intelligence.", "language": "en"},
        {"text": "Python is a high-level programming language.", "language": "en"},
        {"text": "Neural networks are inspired by biological neurons.", "language": "en"},
        {"text": "Deep learning has revolutionized computer vision.", "language": "en"},
        {"text": "Natural language processing enables machines to understand text.", "language": "en"},
        {"text": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)", "language": "en"},
        {"text": "class Model: def __init__(self): self.weights = None", "language": "en"},
        {"text": "Transformers have become the dominant architecture in NLP.", "language": "en"},
        {"text": "Attention is all you need for sequence modeling.", "language": "en"},
        
        # Korean technical
        {"text": "안녕하세요! 오늘은 좋은 날씨입니다.", "language": "ko"},
        {"text": "인공지능은 미래의 핵심 기술입니다.", "language": "ko"},
        {"text": "한국어 자연어 처리는 매우 중요한 연구 분야입니다.", "language": "ko"},
        {"text": "기계 학습은 데이터로부터 패턴을 학습합니다.", "language": "ko"},
        {"text": "딥러닝은 인공 신경망을 사용합니다.", "language": "ko"},
        {"text": "파이썬은 가장 인기 있는 프로그래밍 언어 중 하나입니다.", "language": "ko"},
        {"text": "함수형 프로그래밍은 부작용을 최소화합니다.", "language": "ko"},
        {"text": "객체 지향 프로그래밍은 캡슐화를 제공합니다.", "language": "ko"},
        {"text": "트랜스포머는 자연어 처리의 혁명을 가져왔습니다.", "language": "ko"},
        {"text": "어텐션 메커니즘은 시퀀스 모델링의 핵심입니다.", "language": "ko"},
        
        # Mixed language
        {"text": "AI (인공지능) is changing the world.", "language": "mixed"},
        {"text": "Python과 JavaScript는 인기 있는 프로그래밍 언어입니다.", "language": "mixed"},
        {"text": "Machine Learning과 딥러닝의 차이점은 무엇인가요?", "language": "mixed"},
        {"text": "Transformer 모델은 NLP 분야를 혁신했습니다.", "language": "mixed"},
        {"text": "GPU를 사용하면 딥러닝 훈련이 빠릅니다.", "language": "mixed"},
        
        # Conversational
        {"text": "Hello! How are you today?", "language": "en"},
        {"text": "I'm doing great, thank you for asking!", "language": "en"},
        {"text": "오늘 뭐 하셨어요?", "language": "ko"},
        {"text": "코딩을 하고 있었어요. 재미있네요!", "language": "ko"},
        {"text": "What's your favorite programming language?", "language": "en"},
    ]
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Write training data
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    
    # Split data 80/20
    split_idx = int(len(demo_texts) * 0.8)
    train_data = demo_texts[:split_idx]
    val_data = demo_texts[split_idx:]
    
    # Write training data (repeat to have more samples)
    with open(train_file, 'w', encoding='utf-8') as f:
        for _ in range(10):  # Repeat 10 times
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Write validation data
    with open(val_file, 'w', encoding='utf-8') as f:
        for _ in range(5):  # Repeat 5 times
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("✅ Created demo data files:")
    print(f"   {train_file}: {split_idx * 10} samples")
    print(f"   {val_file}: {(len(demo_texts) - split_idx) * 5} samples")
    
    # Also create a mini test file
    test_file = data_dir / "test.jsonl"
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in demo_texts[:5]:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"   {test_file}: 5 samples")


if __name__ == "__main__":
    create_demo_data()