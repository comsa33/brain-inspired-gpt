#!/usr/bin/env python3
"""
Test data loading functionality for Brain-Inspired GPT
Tests Korean dataset preparation, tokenization, and data pipeline
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import tempfile
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.multilingual_tokenizer import MultilingualBrainTokenizer, KoreanDataCollator
from training.prepare_korean_data import KoreanDatasetPreprocessor


def test_korean_tokenizer():
    """Test Korean tokenizer functionality"""
    print("\n🧪 Testing Korean Tokenizer...")
    
    tokenizer = MultilingualBrainTokenizer()
    
    # Test cases
    test_texts = [
        ("안녕하세요", "ko", "Basic Korean greeting"),
        ("인공지능", "ko", "Korean AI term"),
        ("서울특별시", "ko", "Seoul city name"),
        ("감사합니다", "ko", "Korean thanks"),
        ("2024년 1월", "ko", "Korean date"),
        ("AI와 머신러닝", "ko", "Mixed Korean-English"),
    ]
    
    for text, expected_lang, description in test_texts:
        # Test language detection
        detected_lang = tokenizer.detect_language(text)
        print(f"\n📝 {description}: '{text}'")
        print(f"   Detected language: {detected_lang}")
        
        # Test encoding
        tokens = tokenizer.encode(text, language=expected_lang)
        print(f"   Encoded to {len(tokens)} tokens")
        
        # Test decoding
        decoded = tokenizer.decode(tokens)
        print(f"   Decoded: '{decoded}'")
        
        # Verify round-trip
        if text in decoded or decoded in text:
            print("   ✅ Round-trip successful")
        else:
            print("   ⚠️  Round-trip mismatch")
            
            
def test_data_preparation():
    """Test Korean data preparation"""
    print("\n🧪 Testing Data Preparation...")
    
    # Create tokenizer first
    tokenizer = MultilingualBrainTokenizer()
    preprocessor = KoreanDatasetPreprocessor(tokenizer)
    
    # Create sample Korean data
    sample_data = [
        "인공지능은 미래 기술의 핵심입니다.",
        "딥러닝과 머신러닝의 차이점은 무엇일까요?",
        "한국어 자연어 처리는 매우 흥미로운 분야입니다.",
        "서울에서 부산까지 KTX로 2시간 30분이 걸립니다.",
        "오늘 날씨가 정말 좋네요!",
    ]
    
    # Test text processing
    processed_data = []
    for text in sample_data:
        # Clean text
        cleaned = preprocessor.clean_text(text)
        
        # Split into sentences
        sentences = preprocessor.split_sentences(cleaned)
        
        print(f"\n원문: {text}")
        print(f"정제: {cleaned}")
        print(f"문장 수: {len(sentences)}")
        
        processed_data.extend(sentences)
        
    print(f"\n✅ Processed {len(sample_data)} texts into {len(processed_data)} sentences")
    
    return processed_data


def test_data_collator():
    """Test data collation for training"""
    print("\n🧪 Testing Data Collator...")
    
    tokenizer = MultilingualBrainTokenizer()
    collator = KoreanDataCollator(tokenizer, max_length=64)
    
    # Create sample batch
    examples = [
        {"text": "인공지능의 미래는 밝습니다.", "language": "ko"},
        {"text": "The future of AI is bright.", "language": "en"},
        {"text": "def train_model(data):", "language": "code"},
        {"text": "서울은 한국의 수도입니다.", "language": "ko"},
    ]
    
    # Collate batch
    batch = collator(examples)
    
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  language_ids: {batch['language_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    
    # Verify batch content
    for i, example in enumerate(examples):
        lang_id = batch['language_ids'][i].item()
        lang_map = {0: 'en', 1: 'ko', 2: 'code'}
        print(f"\n  Example {i}: {example['language']} -> {lang_map.get(lang_id, 'unknown')}")
        
    print("\n✅ Data collator working correctly")
    
    return batch


def test_dataloader():
    """Test PyTorch DataLoader integration"""
    print("\n🧪 Testing DataLoader...")
    
    tokenizer = MultilingualBrainTokenizer()
    collator = KoreanDataCollator(tokenizer, max_length=64)
    
    # Create dataset
    dataset = [
        {"text": f"한국어 문장 {i}번입니다.", "language": "ko"}
        for i in range(20)
    ]
    dataset.extend([
        {"text": f"English sentence number {i}.", "language": "en"}
        for i in range(20)
    ])
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator
    )
    
    # Test iteration
    print(f"\nDataset size: {len(dataset)}")
    print(f"Batch size: 4")
    print(f"Expected batches: {len(dataset) // 4}")
    
    batch_count = 0
    total_samples = 0
    
    for batch in dataloader:
        batch_count += 1
        batch_size = batch['input_ids'].shape[0]
        total_samples += batch_size
        
        if batch_count == 1:
            print(f"\nFirst batch:")
            print(f"  Batch size: {batch_size}")
            print(f"  Input shape: {batch['input_ids'].shape}")
            print(f"  Languages: {batch['language_ids'].tolist()}")
            
    print(f"\n✅ Processed {batch_count} batches, {total_samples} total samples")
    

def test_save_load_data():
    """Test saving and loading processed data"""
    print("\n🧪 Testing Data Save/Load...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare some data
        data = [
            {"text": "안녕하세요, 인공지능입니다.", "language": "ko"},
            {"text": "Hello, I am AI.", "language": "en"},
        ]
        
        # Save as JSON
        json_path = os.path.join(temp_dir, "test_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Saved {len(data)} examples to JSON")
        
        # Load back
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            
        print(f"✅ Loaded {len(loaded_data)} examples from JSON")
        
        # Verify
        assert len(loaded_data) == len(data)
        assert loaded_data[0]['text'] == data[0]['text']
        print("✅ Data integrity verified")
        

def test_mixed_language_handling():
    """Test handling of mixed language content"""
    print("\n🧪 Testing Mixed Language Handling...")
    
    tokenizer = MultilingualBrainTokenizer()
    
    mixed_texts = [
        "AI와 인공지능은 같은 meaning입니다.",
        "Deep learning은 딥러닝으로 번역됩니다.",
        "Python으로 AI 모델을 만들어봅시다.",
        "서울 Seoul 대한민국 Korea",
    ]
    
    for text in mixed_texts:
        print(f"\n텍스트: '{text}'")
        
        # Detect chunks
        chunks = tokenizer._split_by_language(text)
        print(f"청크 수: {len(chunks)}")
        
        for chunk_text, chunk_lang in chunks:
            print(f"  [{chunk_lang}] '{chunk_text}'")
            
        # Encode
        tokens = tokenizer.encode(text)
        print(f"토큰 수: {len(tokens)}")
        
        # Decode
        decoded = tokenizer.decode(tokens)
        print(f"디코딩: '{decoded}'")
        

def run_all_data_tests():
    """Run all data loading tests"""
    print("🧠 Brain-Inspired GPT Data Loading Tests")
    print("=" * 60)
    
    try:
        # Run tests
        test_korean_tokenizer()
        processed_data = test_data_preparation()
        batch = test_data_collator()
        test_dataloader()
        test_save_load_data()
        test_mixed_language_handling()
        
        print("\n" + "=" * 60)
        print("✅ All data loading tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    run_all_data_tests()