<div align="center">

# 🧠 CortexGPT

**인간 두뇌에서 영감을 받은 실시간 학습 언어 모델**

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Model Size](https://img.shields.io/badge/model-768D-purple.svg)
![Memory](https://img.shields.io/badge/memory-STM→LTM→Archive-orange.svg)

[English](README.md) | [한국어](#한국어)

</div>

## 한국어

### 🏛️ 아키텍처

```mermaid
graph TB
    subgraph "CortexGPT 모델"
        Input["📥 입력 레이어<br/>• 다국어 토크나이저<br/>• 한국어/영어 지원<br/>• BPE 인코딩"]
        
        Transformer["🤖 트랜스포머 코어<br/>• 멀티 헤드 어텐션<br/>• 피드 포워드 네트워크<br/>• 레이어 정규화<br/>• 잔차 연결"]
        
        subgraph "메모리 시스템"
            STM["💭 STM (단기 기억)<br/>• 용량: 64<br/>• 빠른 접근<br/>• 최근 상호작용"]
            LTM["🧠 LTM (장기 기억)<br/>• 용량: 10,000<br/>• 통합된 지식<br/>• 빈번한 패턴"]
            Archive["📚 Archive (보관 메모리)<br/>• 용량: 100,000<br/>• 압축 저장<br/>• 드물게 사용되는 지식"]
        end
        
        Learner["🎓 실시간 학습기<br/>• 온라인 학습<br/>• 메모리 통합<br/>• 자기 평가"]
        
        Output["📤 출력 레이어<br/>• 토큰 생성<br/>• 신뢰도 점수<br/>• 언어 감지"]
    end
    
    Input --> |"인코딩된 토큰"| Transformer
    Transformer --> |"컨텍스트 저장"| STM
    STM --> |"통합<br/>(자주 사용)"| LTM
    LTM --> |"보관<br/>(드물게 사용)"| Archive
    STM --> |"현재 컨텍스트"| Learner
    LTM --> |"검색된 지식"| Learner
    Learner --> |"가중치 업데이트"| Transformer
    Transformer --> |"예측"| Output
    Learner -.-> |"업데이트"| STM
    Learner -.-> |"전송"| LTM
    
    style Input fill:#e6f3ff
    style Transformer fill:#e6f3ff
    style STM fill:#ffe6e6
    style LTM fill:#e6ffe6
    style Archive fill:#e6e6ff
    style Learner fill:#e6f3ff
    style Output fill:#e6f3ff
```

### 🌟 핵심 특징

- **실시간 학습**: 훈련/추론 구분 없이 지속적으로 학습
- **인간과 유사한 메모리**: STM(단기) → LTM(장기) → Archive(보관) 시스템
- **자기 개선**: 스스로 평가하고 개선하는 메커니즘
- **다국어 지원**: 한국어와 영어를 자연스럽게 처리
- **메모리 효율성**: OOM 방지를 위한 적응형 배치 크기 조정
- **체크포인트 지원**: 중단 후 훈련 재개 가능

### 🚀 빠른 시작

#### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/comsa33/cortexgpt.git
cd cortexgpt

# 모든 의존성 설치
uv sync

# 또는 모니터링 도구 포함 설치
uv sync --extra monitoring
```

#### 2. 데모 데이터 생성

```bash
# 데모 훈련 데이터 생성
uv run scripts/data/create_demo_data.py
```

#### 3. 기본 기능 테스트

```bash
# 토크나이저 테스트
uv run tests/demo_tokenizer.py

# 모델 학습 가능 여부 테스트 (과적합 테스트)
uv run tests/test_overfit.py
```

#### 4. 훈련

```bash
# 빠른 데모 훈련 (작은 모델, 빠름)
uv run cortexgpt/training/train_realtime.py \
    --dataset demo \
    --dim 256 \
    --lr 1e-3 \
    --epochs 20

# wandb로 모니터링
uv run cortexgpt/training/train_realtime.py \
    --dataset demo \
    --dim 512 \
    --wandb

# 실제 데이터셋으로 훈련 (설정 후)
uv run scripts/data/setup_datasets.py  # 데이터셋 다운로드 및 준비
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --batch-size 4 \
    --gradient-accumulation 8 \
    --epochs 50 \
    --wandb

# 중단된 훈련 재개
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --resume auto
```

#### 5. 데모 실행

```bash
# 최소 생성 데모
uv run scripts/demos/minimal_demo.py

# 실시간 학습 데모
uv run scripts/demos/learning_effect_demo.py

# 대화형 채팅 데모
uv run scripts/demos/natural_language_demo.py
```

### 📖 상세 사용 가이드

#### 사전 훈련된 모델 사용하기

```bash
# 체크포인트를 로드하고 텍스트 생성
uv run cortexgpt/inference/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "인공지능의 미래는" \
    --max-length 100

# 훈련된 모델로 대화형 채팅
uv run cortexgpt/inference/chat.py \
    --checkpoint checkpoints/best_model.pt \
    --temperature 0.8
```

#### 실시간 학습 데모

실시간 학습 데모는 CortexGPT가 상호작용을 통해 어떻게 학습하는지 보여줍니다:

```bash
# 학습 효과 데모 실행
uv run scripts/demos/learning_effect_demo.py
```

이 데모는 다음을 보여줍니다:
- 지식 없이 초기 응답
- 사용자 피드백으로부터 학습
- 학습 후 개선된 응답
- 시간에 따른 메모리 통합

#### 커스텀 훈련

커스텀 데이터셋의 경우, JSONL 파일로 데이터를 생성하세요:

```json
{"text": "여기에 훈련 텍스트를 입력하세요"}
{"text": "또 다른 훈련 예제"}
```

그런 다음 훈련:

```bash
# 커스텀 데이터셋 준비
uv run cortexgpt/data/prepare_custom.py \
    --input your_data.jsonl \
    --output data/custom

# 커스텀 데이터로 훈련
uv run cortexgpt/training/train_realtime.py \
    --dataset custom \
    --vocab-size 30000 \
    --epochs 50
```

#### 메모리 시스템 설정

다양한 사용 사례에 맞게 메모리 시스템 매개변수를 조정하세요:

```bash
# 빠른 실험을 위한 작은 메모리
uv run cortexgpt/training/train_realtime.py \
    --stm-capacity 32 \
    --ltm-capacity 1000 \
    --archive-capacity 10000

# 프로덕션을 위한 큰 메모리
uv run cortexgpt/training/train_realtime.py \
    --stm-capacity 128 \
    --ltm-capacity 50000 \
    --archive-capacity 500000
```

#### API 사용법

```python
from cortexgpt import CortexGPT, MultilingualTokenizer

# 모델과 토크나이저 초기화
model = CortexGPT.from_pretrained("checkpoints/best_model.pt")
tokenizer = MultilingualTokenizer.from_pretrained("checkpoints/tokenizer.json")

# 텍스트 생성
prompt = "기계 학습이란"
inputs = tokenizer.encode(prompt)
outputs = model.generate(inputs, max_length=100)
response = tokenizer.decode(outputs)
print(response)

# 실시간 학습
from cortexgpt.learning import RealTimeLearner

learner = RealTimeLearner(model, tokenizer)
learner.start()  # 백그라운드 학습 시작

# 학습과 함께 쿼리 처리
response, metadata = learner.process_query(
    "기계 학습이란 무엇인가요?",
    learn=True
)
print(f"응답: {response}")
print(f"신뢰도: {metadata['confidence']}")
```

#### 훈련 모니터링

Weights & Biases를 사용하여 상세한 모니터링:

```bash
# 먼저 wandb에 로그인
wandb login

# 모니터링과 함께 훈련
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --wandb \
    --wandb-project "cortexgpt-experiments" \
    --wandb-name "run-001"
```

모니터링 항목:
- 훈련/검증 손실
- 학습률 스케줄
- 메모리 시스템 사용량
- 샘플 생성
- 성능 메트릭

### 🌍 실제 데이터셋으로 훈련하기

#### 1단계: 데이터셋 다운로드

```bash
# 샘플 데이터셋 다운로드 (KLUE, 위키피디아 등)
uv run cortexgpt/data/download_datasets.py
```

다음 데이터셋의 샘플을 다운로드합니다:
- **KLUE**: 한국어 언어 이해 평가 데이터셋
- **한국어 위키피디아**: 한국어 백과사전 문서
- **영어 위키피디아**: 영어 백과사전 문서
- **OpenWebText**: 웹 크롤링 데이터 (샘플)

#### 2단계: 데이터셋 준비 (선택사항)

훈련 스크립트는 JSONL 파일을 자동으로 처리하지만, 더 빠른 로딩을 위해 사전 처리할 수 있습니다:

```bash
# 다운로드한 모든 데이터셋 준비
uv run cortexgpt/data/prepare_datasets.py
```

#### 3단계: 실제 데이터로 훈련

##### 한국어 데이터셋 (KLUE)
```bash
# KLUE 데이터셋으로 훈련
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --dim 512 \
    --vocab-size 30000 \
    --batch-size 8 \
    --gradient-accumulation 4 \
    --lr 3e-4 \
    --epochs 10 \
    --wandb
```

##### 영어 데이터셋 (위키피디아)
```bash
# 영어 위키피디아로 훈련
uv run cortexgpt/training/train_realtime.py \
    --dataset wikipedia \
    --dim 512 \
    --vocab-size 30000 \
    --batch-size 8 \
    --gradient-accumulation 4 \
    --lr 3e-4 \
    --epochs 10 \
    --wandb
```

##### 한국어-영어 혼합 훈련
```bash
# 결합된 데이터셋으로 훈련
uv run cortexgpt/training/train_realtime.py \
    --dataset combined \
    --korean-ratio 0.4 \
    --dim 768 \
    --vocab-size 50000 \
    --batch-size 4 \
    --gradient-accumulation 8 \
    --lr 2e-4 \
    --epochs 20 \
    --wandb
```

#### 4단계: 훈련 재개

훈련이 중단된 경우:

```bash
# 최신 체크포인트에서 재개
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --resume auto \
    --wandb

# 특정 체크포인트에서 재개
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --resume checkpoints/realtime/model_best.pt \
    --wandb
```

#### 훈련 팁

1. **작게 시작하기**: 테스트를 위해 `--dim 256`과 `--vocab-size 10000`으로 시작
2. **메모리 모니터링**: OOM 발생 시 `--batch-size 2`를 사용하고 `--gradient-accumulation` 증가
3. **학습률**: 작은 모델은 `1e-3`, 큰 모델은 `3e-4`로 시작
4. **어휘 크기**: 
   - 한국어만: 20,000-30,000
   - 영어만: 30,000-40,000
   - 혼합: 40,000-50,000

### 📊 사용 가능한 데이터셋

| 데이터셋 | 언어 | 설명 |
|---------|------|------|
| `demo` | 혼합 | 작은 테스트 데이터셋 (기본값) |
| `klue` | 한국어 | 한국어 언어 이해 평가 |
| `wikipedia` | 영어 | 위키피디아 문서 |
| `korean_wiki` | 한국어 | 한국어 위키피디아 |
| `openwebtext` | 영어 | GPT-2 훈련용 웹 텍스트 |
| `combined` | 혼합 | 여러 데이터셋 조합 |

### 🏗️ 프로젝트 구조

```
my-efficient-gpt/
├── cortexgpt/              # 메인 패키지
│   ├── models/            # 모델 아키텍처
│   ├── learning/          # 실시간 학습 시스템
│   ├── tokenization/      # 다국어 토크나이저
│   ├── data/             # 데이터 로딩 유틸리티
│   └── training/         # 훈련 스크립트
├── scripts/
│   ├── data/             # 데이터 준비 스크립트
│   └── demos/            # 데모 애플리케이션
├── tests/                # 테스트 스크립트
├── docs/                 # 문서
└── data/                 # 훈련 데이터
```

### 💡 작동 원리

#### 메모리 흐름
```
새로운 입력 → STM (빠른 접근)
     ↓ (자주 사용)
    LTM (통합된 지식)
     ↓ (오래 미사용)
   Archive (압축 저장)
```

#### 학습 과정
1. **첫 질문**: "아직 학습하지 못한 내용입니다"
2. **학습 후**: 정확한 답변 제공
3. **반복 시**: 신뢰도 증가 (0.6 → 0.9 → 1.0)

### 📈 훈련 옵션

```bash
# 모델 아키텍처
--dim               # 히든 차원 (256/512/768, 기본값: 768)
--vocab-size        # 토크나이저 어휘 크기 (기본값: 50000)

# 훈련 파라미터
--batch-size        # 배치 크기 (기본값: 8)
--gradient-accumulation  # 그래디언트 누적 단계 (기본값: 4)
--epochs           # 에폭 수 (기본값: 10)
--lr              # 학습률 (기본값: 3e-4)

# 메모리 시스템
--stm-capacity     # 단기 기억 용량 (기본값: 64)
--ltm-capacity     # 장기 기억 용량 (기본값: 10000)
--archive-capacity # 보관 용량 (기본값: 100000)

# 모니터링 및 체크포인팅
--wandb           # Weights & Biases 로깅 활성화
--wandb-project   # W&B 프로젝트 이름
--checkpoint-dir  # 체크포인트 디렉토리
--resume         # 체크포인트에서 재개 (auto/경로)
```

### 🚀 권장 훈련 설정

#### 테스트 및 개발
```bash
# 빠른 테스트를 위한 작은 모델
--dim 256 --lr 1e-3 --batch-size 4 --epochs 20
```

#### 데모 훈련
```bash
# 데모를 위한 중간 모델
--dim 512 --lr 5e-4 --batch-size 8 --gradient-accumulation 4
```

#### 프로덕션 훈련
```bash
# 실제 훈련을 위한 큰 모델
--dim 768 --lr 3e-4 --batch-size 4 --gradient-accumulation 8 --wandb
```

### 🔬 연구 및 개발

CortexGPT는 여러 신경과학 개념을 구현합니다:

- **헤비안 학습**: "함께 발화하는 뉴런은 함께 연결된다"
- **메모리 통합**: STM에서 LTM으로의 점진적 전이
- **선택적 주의**: 관련 정보에 집중
- **지속적 학습**: 잊지 않고 새로운 작업 학습

### 📝 인용

```bibtex
@software{cortexgpt2024,
  author = {Ruo Lee},
  title = {CortexGPT: Real-time Learning Language Model},
  year = {2025},
  email = {comsa333@gmail.com}
}
```

### 📄 라이선스

MIT 라이선스 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

Made with ❤️ by Ruo Lee