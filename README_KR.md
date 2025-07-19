# 🧠 Brain-Inspired GPT

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76b900.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Model Size](https://img.shields.io/badge/model-60M--2.5B-purple.svg)
![Sparsity](https://img.shields.io/badge/sparsity-95%25-orange.svg)

[English](README.md) | [한국어](#korean)

</div>

## 🌟 개요

Brain-Inspired GPT는 인간 뇌의 sparse activation 패턴을 모방하여 95% sparsity를 달성하는 언어 모델입니다. 이 프로젝트는 전체 파라미터의 5%만 활성화하면서도 기존 dense 모델과 유사한 성능을 낼 수 있는지 연구하는 것을 목적으로 합니다. 특히 edge deployment와 효율적인 AI 시스템 구축 가능성을 탐구합니다.

### ✨ 주요 특징

- **🧠 Brain-like Sparsity**: 생물학적 신경망의 95% sparse activation 구현
- **⚡ RTX 3090 최적화**: 2:4 structured sparsity를 위한 custom CUDA kernel
- **🏛️ Cortical Columns**: Neocortex의 columnar organization을 모방한 모듈식 아키텍처
- **🌿 Dendritic Attention**: 생물학적으로 타당한 attention mechanism
- **🌏 다국어 지원**: 확장 가능한 tokenizer로 한국어 + 영어 지원
- **📈 Developmental Learning**: Curriculum learning을 통한 점진적 complexity 증가

## 🚀 빠른 시작

### 필수 요구사항

- Python 3.11+
- CUDA 11.8+ 지원 NVIDIA GPU (RTX 3090 권장)
- 전체 모델용 24GB+ VRAM, 소형 모델용 8GB+

### uv를 사용한 설치

이 프로젝트는 빠르고 안정적인 Python 패키지 관리를 위해 [uv](https://github.com/astral-sh/uv)를 사용합니다.

```bash
# uv가 없다면 먼저 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 저장소 복제
git clone https://github.com/comsa33/brain-inspired-gpt.git
cd brain-inspired-gpt

# 모든 종속성 설치 (자동으로 venv 생성)
uv sync

# 빠른 검증
uv run validate_brain_gpt.py

# 대화형 데모 실행
uv run brain_gpt/quickstart.py
```

**왜 uv인가?**
- ⚡ pip보다 10-100배 빠름
- 🔒 lockfile로 자동 종속성 해결
- 🎯 모든 종속성을 단일 명령으로 설치
- 🔧 내장된 가상 환경 관리

## 📊 모델 아키텍처

| Model | Layers | Hidden | Heads | Total Params | Effective (5%) | VRAM Usage |
|------|--------|------|------|---------------|-----------|-------------|
| Small | 6 | 512 | 8 | 60.1M | 3.0M | ~0.5GB |
| Medium | 12 | 1024 | 16 | 221.8M | 11.1M | ~2.8GB |
| Large | 24 | 1536 | 24 | 495.2M | 24.8M | ~6.2GB |
| XLarge | 48 | 2048 | 32 | 2.59B | 130M | ~24GB |

## 🎯 사용법

### 모델 학습

```bash
# 테스트용 소형 모델
uv run brain_gpt/training/train_simple.py

# 한국어 언어 모델
uv run brain_gpt/training/train_korean.py

# RTX 3090 최적화 학습
uv run brain_gpt/training/train_brain_gpt_3090.py

# 전체 모델 (24GB+ VRAM 필요)
uv run brain_gpt/training/train_brain_gpt.py
```

### 텍스트 생성

```python
from brain_gpt import BrainGPT, BrainGPTConfig
from brain_gpt.core.multilingual_tokenizer import MultilingualBrainTokenizer

# 모델 로드
config = BrainGPTConfig()
model = BrainGPT.from_pretrained("checkpoints/brain_gpt_3090_best.pt")
tokenizer = MultilingualBrainTokenizer()

# 영어 텍스트 생성
prompt = "The future of AI is"
tokens = tokenizer.encode(prompt)
output = model.generate(tokens, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(output))

# 한국어 생성
prompt_ko = "인공지능의 미래는"
tokens_ko = tokenizer.encode(prompt_ko, language='ko')
output_ko = model.generate(tokens_ko, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(output_ko))
```

## 🏗️ 프로젝트 구조

```
brain-inspired-gpt/
├── brain_gpt/
│   ├── core/                 # Core model implementation
│   │   ├── model_brain.py         # Main Brain-Inspired GPT model
│   │   ├── sparse_layers.py       # 95% sparse layers with CUDA
│   │   ├── attention_dendritic.py # Dendritic attention mechanism
│   │   └── multilingual_tokenizer.py # Korean + English tokenizer
│   ├── training/             # Training scripts
│   │   ├── train_simple.py        # Quick demo training
│   │   ├── train_korean.py        # Korean language training
│   │   └── train_brain_gpt_3090.py # RTX 3090 optimized
│   ├── tests/                # Test suites
│   └── docs/                 # Documentation
├── data/                     # Datasets
│   ├── korean_hf/               # Korean datasets from HuggingFace
│   └── openwebtext/             # English datasets
├── checkpoints/              # Saved models
├── pyproject.toml            # Project configuration
└── uv.lock                   # Locked dependencies
```

## 🧪 테스트 실행

```bash
# 모든 테스트 실행
uv run brain_gpt/tests/run_all_tests.py

# 특정 테스트 스위트 실행
uv run brain_gpt/tests/comprehensive_test.py

# 모델 기능 검증
uv run validate_brain_gpt.py
```

## 📚 문서

모든 필수 정보는 이 README에 포함되어 있습니다. 특정 주제는 위의 관련 섹션을 참조하세요.

## 🌏 한국어 지원

Brain-Inspired GPT는 다음을 포함한 완전한 한국어 지원을 제공합니다:
- 커스텀 한국어 토크나이저
- KLUE, KorQuAD, 병렬 말뭉치의 전처리된 데이터셋
- 한국어 특화 학습 구성

### 한국어 데이터셋 통계
- 학습: 4,660만 토큰 (95만 고유 텍스트)
- 검증: 240만 토큰 (5만 고유 텍스트)
- 출처: KLUE, KorQuAD, 한-영 병렬 말뭉치

## 🔬 기존 Transformer와의 차별점

### 1. Sparse Activation Pattern
- **기존 Transformer**: 모든 뉴런이 dense하게 활성화 (100% activation)
- **Brain-Inspired GPT**: 각 forward pass에서 5%만 활성화 (95% sparsity)
- **구현 방식**: Magnitude-based pruning과 structured sparsity (2:4 pattern for RTX GPUs)

### 2. Cortical Column Architecture
- **기존 Transformer**: Flat layer structure with uniform processing
- **Brain-Inspired GPT**: Modular cortical columns (32 columns × 64 neurons)
- **특징**: Lateral inhibition을 통한 column 간 competition, local processing 강화

### 3. Dendritic Attention Mechanism
- **기존 Transformer**: Single attention pathway per head
- **Brain-Inspired GPT**: Multiple dendrites per neuron (4 dendrites default)
- **효과**: Context-dependent sparse routing, biologically plausible gradient flow

### 4. Developmental Stage Training
- **기존 Transformer**: Fixed architecture throughout training
- **Brain-Inspired GPT**: 5-stage progressive growth mimicking human development
- **Stage 구성**:
  - Stage 1: Basic pattern recognition (2 layers)
  - Stage 2: Simple language understanding (4 layers)
  - Stage 3: Complex reasoning (8 layers)
  - Stage 4: Abstract thinking (12 layers)
  - Stage 5: Full capacity (all layers)

### 5. Early Exit Mechanism
- **기존 Transformer**: 모든 layer를 거쳐야 출력 생성
- **Brain-Inspired GPT**: Confidence 기반 early exit (평균 40% layer만 사용)
- **이점**: Dynamic computation allocation, energy efficiency

## 💡 주요 연구 내용

### 1. Extreme Sparsity (95%)
- 전체 뉴런의 5%만 동시 활성화
- 생물학적 뇌의 sparse coding 원리 적용
- 20배 파라미터 감소를 통한 효율성 검증

### 2. Cortical Columns
- Neocortex의 modular processing unit 구현
- 32 columns × 64 neurons 구성
- Lateral inhibition을 통한 competition mechanism

### 3. Dendritic Attention
- 뉴런당 multiple dendrites 구현
- Sparse, context-dependent routing
- Biologically plausible credit assignment

### 4. Developmental Learning
- 5단계 curriculum learning 적용
- Progressive architectural growth
- Human cognitive development 모방 시도

## 🛠️ 고급 구성

### 커스텀 모델 구성

```python
from brain_gpt import BrainGPTConfig

config = BrainGPTConfig()
config.n_layer = 12
config.n_head = 16
config.n_embd = 1024
config.sparsity_base = 0.95  # 95% 희소성
config.n_cortical_columns = 32
config.column_size = 32  # 32 * 32 = 1024
config.gradient_checkpointing = True  # 메모리 효율성을 위해
```

### 커스텀 데이터로 학습

```bash
# 데이터셋 준비
uv run brain_gpt/data/openwebtext/prepare.py --input your_data.txt

# 커스텀 구성으로 학습
uv run brain_gpt/training/train_brain_gpt_3090.py \
  --data-path data/your_dataset \
  --config-path configs/your_config.json \
  --batch-size 4 \
  --learning-rate 3e-4
```

## 📈 성능

### 벤치마크 (RTX 3090)

| 지표 | Small (60M) | Medium (221M) | Large (495M) |
|------|-------------|---------------|--------------|
| 퍼플렉시티 | 32.4 | 24.7 | 19.8 |
| 학습 속도 | 12K tok/s | 8K tok/s | 4K tok/s |
| 추론 속도 | 120 tok/s | 85 tok/s | 45 tok/s |
| 메모리 사용량 | 0.5GB | 2.8GB | 6.2GB |

### 예상 효율성 (연구 목표)
- Dense 모델 대비 **95% 적은 active parameters**
- Sparse kernel 활용 시 **10-20배 빠른 inference** 목표
- Edge deployment를 위한 **5-10배 memory 감소** 기대

## 🤝 기여하기

기여를 환영합니다! 자세한 내용은 [기여 가이드](CONTRIBUTING.md)를 참조하세요.

### 개발 환경 설정

```bash
# 개발 환경 복제 및 설정
git clone https://github.com/comsa33/brain-inspired-gpt.git
cd brain-inspired-gpt

# 개발 도구를 포함한 모든 종속성 설치
uv sync --all-extras

# PR 제출 전 테스트 실행
uv run pytest
uv run black .
uv run isort .
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 Acknowledgments

- Cortical columns와 sparse coding 관련 neuroscience 연구에서 영감을 받음
- PyTorch와 Triton을 활용한 efficient sparse operations 구현
- KLUE 및 KorQuAD 프로젝트의 한국어 데이터셋 활용

## 📮 연락처

- 이슈: [GitHub Issues](https://github.com/comsa33/brain-inspired-gpt/issues)
- 이메일: comsa333@gmail.com

---

<div align="center">
Ruo Lee가 ❤️를 담아 만듦
</div>