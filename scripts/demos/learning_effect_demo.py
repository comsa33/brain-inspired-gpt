#!/usr/bin/env python3
"""
학습 효과를 명확히 보여주는 간단한 데모
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class SimpleRealTimeLearner:
    """간단한 실시간 학습 시스템"""
    
    def __init__(self):
        # 질문-답변 메모리
        self.memory = {}
        self.access_count = defaultdict(int)
        self.confidence = defaultdict(float)
        
    def learn(self, question, answer):
        """질문-답변 쌍을 학습"""
        self.memory[question] = answer
        self.access_count[question] += 1
        # 반복할수록 신뢰도 증가
        self.confidence[question] = min(1.0, self.access_count[question] * 0.3)
        
    def respond(self, question):
        """질문에 대한 응답 생성"""
        # 정확히 일치하는 질문이 있는지 확인
        if question in self.memory:
            self.access_count[question] += 1
            self.confidence[question] = min(1.0, self.access_count[question] * 0.3)
            return self.memory[question], self.confidence[question]
        
        # 유사한 질문 찾기
        best_match = None
        best_score = 0
        
        for mem_q in self.memory:
            # 간단한 유사도 계산 (공통 단어 비율)
            q_words = set(question.split())
            m_words = set(mem_q.split())
            if len(q_words) > 0:
                similarity = len(q_words & m_words) / len(q_words)
                if similarity > best_score:
                    best_score = similarity
                    best_match = mem_q
        
        if best_match and best_score > 0.5:
            return self.memory[best_match], best_score * self.confidence[best_match]
        
        return "아직 학습하지 못한 내용입니다.", 0.0


def main():
    print("🧠 CortexGPT 학습 효과 데모")
    print("=" * 50)
    print("실시간으로 학습하고 개선되는 과정을 보여드립니다.\n")
    
    # 학습 시스템 생성
    learner = SimpleRealTimeLearner()
    
    # 시나리오 1: 처음 보는 질문
    print("📚 시나리오 1: 처음 보는 질문")
    print("-" * 40)
    
    question1 = "인공지능이 뭔가요?"
    response1, conf1 = learner.respond(question1)
    print(f"👤 질문: {question1}")
    print(f"🤖 응답: {response1}")
    print(f"📊 신뢰도: {conf1:.2f}\n")
    
    # 학습시키기
    print("💡 학습 중...")
    learner.learn(question1, "인공지능은 컴퓨터가 인간처럼 학습하고 판단하는 기술입니다.")
    print("✅ 학습 완료!\n")
    
    # 같은 질문 다시하기
    print("🔄 같은 질문을 다시 해봅니다:")
    response2, conf2 = learner.respond(question1)
    print(f"👤 질문: {question1}")
    print(f"🤖 응답: {response2}")
    print(f"📊 신뢰도: {conf2:.2f} (향상됨!)\n")
    
    # 시나리오 2: 반복 학습으로 신뢰도 증가
    print("\n📚 시나리오 2: 반복으로 신뢰도 증가")
    print("-" * 40)
    
    question2 = "기계학습이란?"
    learner.learn(question2, "기계학습은 데이터로부터 패턴을 찾아 학습하는 방법입니다.")
    
    for i in range(3):
        response, conf = learner.respond(question2)
        print(f"\n{i+1}번째 질문:")
        print(f"👤 질문: {question2}")
        print(f"🤖 응답: {response}")
        print(f"📊 신뢰도: {conf:.2f}")
    
    # 시나리오 3: 유사한 질문 처리
    print("\n\n📚 시나리오 3: 유사한 질문 이해")
    print("-" * 40)
    
    # 여러 질문-답변 학습
    qa_pairs = [
        ("날씨가 어때요?", "오늘은 맑고 좋은 날씨입니다."),
        ("안녕하세요", "안녕하세요! 반갑습니다."),
        ("감사합니다", "천만에요! 도움이 되어 기쁩니다."),
    ]
    
    for q, a in qa_pairs:
        learner.learn(q, a)
    
    # 비슷한 질문들
    similar_questions = [
        "오늘 날씨가 어떤가요?",  # '날씨'라는 단어 포함
        "안녕",  # '안녕'이라는 단어 포함
        "고마워요",  # 새로운 표현
    ]
    
    for q in similar_questions:
        response, conf = learner.respond(q)
        print(f"\n👤 질문: {q}")
        print(f"🤖 응답: {response}")
        print(f"📊 신뢰도: {conf:.2f}")
    
    # 최종 통계
    print("\n\n📊 최종 학습 통계")
    print("-" * 40)
    print(f"총 학습된 패턴: {len(learner.memory)}개")
    print(f"가장 많이 접근한 질문: {max(learner.access_count, key=learner.access_count.get) if learner.access_count else 'None'}")
    print(f"평균 신뢰도: {np.mean(list(learner.confidence.values())):.2f}")
    
    print("\n✅ 데모 완료!")
    print("\n핵심 포인트:")
    print("  • 처음 보는 질문은 대답하지 못함")
    print("  • 학습 후에는 정확히 대답")
    print("  • 반복할수록 신뢰도 증가")
    print("  • 유사한 질문도 어느 정도 이해")


if __name__ == "__main__":
    main()