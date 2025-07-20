#!/usr/bin/env python3
"""
자연어 입력을 받는 CortexGPT 데모
실제로 한국어/영어 텍스트를 입력하고 응답을 받을 수 있습니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cortexgpt.models.realtime_cortex import RealTimeCortexGPT, AdvancedMemoryConfig
from cortexgpt.tokenization.multilingual_tokenizer import MultilingualTokenizer
from cortexgpt.learning.realtime_learner import RealTimeLearner


def main():
    print("🧠 CortexGPT 자연어 대화 데모")
    print("=" * 50)
    
    # 모델 설정
    config = AdvancedMemoryConfig(
        stm_capacity=64,
        ltm_capacity=1000,
        learning_rate_stm=0.1,
        self_feedback_rate=0.05
    )
    
    # 모델 생성
    print("\n📦 모델 초기화 중...")
    model = RealTimeCortexGPT(config, vocab_size=10000, dim=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ {device}에서 모델 로드 완료")
    
    # 토크나이저 생성
    print("\n🔤 토크나이저 준비 중...")
    tokenizer = MultilingualTokenizer(vocab_size=10000)
    
    # 샘플 텍스트로 토크나이저 학습
    sample_texts = [
        # 한국어 대화
        "안녕하세요", "안녕하세요, 반갑습니다", "오늘 날씨가 좋네요",
        "네, 정말 좋은 날씨예요", "무엇을 도와드릴까요?",
        "인공지능에 대해 알고 싶어요", "인공지능은 컴퓨터가 인간처럼 학습하는 기술입니다",
        "기계학습이란 무엇인가요?", "기계학습은 데이터로부터 패턴을 찾는 방법입니다",
        "딥러닝과 머신러닝의 차이는?", "딥러닝은 인공신경망을 사용하는 머신러닝의 한 분야입니다",
        
        # 영어 대화
        "Hello", "Hello, nice to meet you", "How are you today?",
        "I'm doing well, thank you", "What can I help you with?",
        "Tell me about AI", "AI is technology that allows computers to learn like humans",
        "What is machine learning?", "Machine learning is finding patterns from data",
        "What's the difference between deep learning and machine learning?",
        "Deep learning is a subset of machine learning using neural networks",
        
        # 일상 대화
        "오늘 뭐 먹었어요?", "점심으로 김치찌개를 먹었어요",
        "What did you have for lunch?", "I had a sandwich",
        "좋은 하루 되세요", "Have a nice day",
        "감사합니다", "Thank you", "천만에요", "You're welcome"
    ] * 10
    
    tokenizer.learn_bpe(sample_texts, verbose=False)
    print(f"✅ 토크나이저 준비 완료 (어휘 크기: {len(tokenizer.vocab)})")
    
    # 학습 시스템 설정
    print("\n🎓 실시간 학습 시스템 시작...")
    learner = RealTimeLearner(model, tokenizer)
    learner.start()
    print("✅ 학습 시스템 활성화")
    
    # 초기 학습 - 기본 대화 패턴
    print("\n📚 기본 대화 패턴 학습 중...")
    basic_conversations = [
        ("안녕하세요", "안녕하세요! 반갑습니다."),
        ("Hello", "Hello! Nice to meet you."),
        ("오늘 날씨 어때요?", "오늘은 좋은 날씨네요."),
        ("How's the weather?", "It's a nice day today."),
        ("감사합니다", "천만에요!"),
        ("Thank you", "You're welcome!"),
    ]
    
    for query, expected in basic_conversations:
        response, _ = learner.process_query(query, learn=True)
        print(f"  학습: {query} → {expected[:20]}...")
    
    print("\n✅ 기본 학습 완료!")
    
    # 대화 시작
    print("\n💬 대화를 시작합니다! (종료: 'quit', 상태: 'stats')")
    print("한국어와 영어 모두 사용 가능합니다.")
    print("-" * 50)
    
    while True:
        try:
            # 사용자 입력
            user_input = input("\n👤 당신: ")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                show_stats(learner, model)
                continue
            
            # 응답 생성
            response, metadata = learner.process_query(user_input, learn=True)
            
            # 응답 표시
            print(f"🤖 CortexGPT: {response}")
            
            # 메타데이터 표시
            print(f"   [품질: {metadata['quality_score']:.2f}, "
                  f"언어: {metadata['language']}, "
                  f"STM: {metadata['confidence']['stm']:.2f}, "
                  f"LTM: {metadata['confidence']['ltm']:.2f}]")
            
            # 학습 효과 표시
            if metadata['learned']:
                print("   ✅ 이 대화로부터 학습했습니다!")
            
        except KeyboardInterrupt:
            print("\n\n👋 대화를 종료합니다.")
            break
        except EOFError:
            print("\n\n⚠️ EOF detected - 비대화형 환경입니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            if "EOF" in str(e):
                break
    
    # 종료
    print("\n🛑 시스템 종료 중...")
    learner.stop()
    
    # 최종 통계
    print("\n📊 최종 대화 통계")
    print("-" * 40)
    stats = learner.stats
    print(f"총 대화 수: {stats['total_queries']}")
    print(f"학습된 대화: {stats['total_learned']}")
    print(f"평균 품질: {stats['avg_quality']:.2f}")
    print(f"언어 분포: 한국어={stats['languages']['ko']}, "
          f"영어={stats['languages']['en']}, "
          f"혼합={stats['languages']['mixed']}")
    
    print("\n✅ 데모 종료!")
    print("\nCortexGPT의 특징:")
    print("  • 대화하면서 실시간으로 학습")
    print("  • 자주 나오는 패턴은 장기 기억으로 저장")
    print("  • 한국어와 영어를 자연스럽게 처리")
    print("  • 스스로 응답을 평가하고 개선")


def show_stats(learner, model):
    """현재 시스템 상태 표시"""
    print("\n📊 시스템 상태")
    print("-" * 40)
    
    # 학습 통계
    stats = learner.stats
    print(f"총 대화: {stats['total_queries']}")
    print(f"학습된 대화: {stats['total_learned']}")
    print(f"평균 품질: {stats['avg_quality']:.2f}")
    
    # 언어 분포
    print(f"\n언어 사용:")
    total_langs = sum(stats['languages'].values())
    if total_langs > 0:
        for lang, count in stats['languages'].items():
            percentage = (count / total_langs) * 100
            print(f"  {lang}: {count} ({percentage:.1f}%)")
    
    # 메모리 상태
    print(f"\n메모리 사용:")
    print(f"  STM: {len(model.stm.memories)} / {model.config.stm_capacity}")
    print(f"  LTM: {len(model.ltm.memories)} / {model.config.ltm_capacity}")
    print(f"  Archive: {model.archive.index.ntotal} / {model.config.archive_capacity}")
    
    # 학습률
    if stats['learning_rate_history']:
        avg_lr = np.mean(list(stats['learning_rate_history']))
        print(f"\n평균 학습률: {avg_lr:.6f}")


if __name__ == "__main__":
    main()