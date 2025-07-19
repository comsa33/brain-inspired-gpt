# 🧠 뇌 영감 GPT (Brain-Inspired GPT)

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

Brain-Inspired GPT는 95% 희소성과 생물학적 영감을 받은 아키텍처를 통해 인간 뇌의 효율성을 모방하는 혁신적인 언어 모델입니다. 파라미터의 5%만 사용하면서도 밀집 모델과 비슷한 성능을 달성하여, 엣지 배포와 효율적인 AI 연구에 이상적입니다.

### ✨ 주요 특징

- **🧠 뇌와 같은 희소성**: 생물학적 신경망을 모방한 95% 희소 활성화
- **⚡ RTX 3090 최적화**: 2:4 구조적 희소성을 위한 커스텀 CUDA 커널
- **🏛️ 피질 기둥**: 신피질 조직에서 영감을 받은 모듈식 아키텍처
- **🌿 수상돌기 주의**: 생물학적으로 타당한 주의 메커니즘
- **🌏 다국어 지원**: 확장 가능한 토크나이저로 한국어 + 영어 지원
- **📈 발달 학습**: 커리큘럼 학습을 통한 점진적 복잡성 증가

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

| 모델 | 레이어 | 히든 | 헤드 | 전체 파라미터 | 유효 (5%) | VRAM 사용량 |
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
│   ├── core/                 # 핵심 모델 구현
│   │   ├── model_brain.py         # 메인 Brain-Inspired GPT 모델
│   │   ├── sparse_layers.py       # CUDA를 사용한 95% 희소 레이어
│   │   ├── attention_dendritic.py # 수상돌기 주의 메커니즘
│   │   └── multilingual_tokenizer.py # 한국어 + 영어 토크나이저
│   ├── training/             # 학습 스크립트
│   │   ├── train_simple.py        # 데모용 빠른 학습
│   │   ├── train_korean.py        # 한국어 언어 학습
│   │   └── train_brain_gpt_3090.py # RTX 3090 최적화
│   ├── tests/                # 종합 테스트
│   └── docs/                 # 추가 문서
├── data/                     # 데이터셋
│   ├── korean_hf/               # HuggingFace의 한국어 데이터셋
│   └── openwebtext/             # 영어 데이터셋
├── checkpoints/              # 저장된 모델
├── pyproject.toml            # 프로젝트 구성 및 종속성
└── uv.lock                   # 잠긴 종속성 버전
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

## 💡 주요 혁신

### 1. 극한 희소성 (95%)
- 언제나 뉴런의 5%만 활성화
- 생물학적 뇌 효율성과 일치
- 최소한의 성능 손실로 20배 파라미터 감소

### 2. 피질 기둥
- 신피질과 같은 모듈식 처리 단위
- 일반적인 구성: 32개 기둥 × 64개 뉴런
- 경쟁을 위한 측면 억제

### 3. 수상돌기 주의
- 뉴런당 다중 수상돌기
- 희소하고 문맥 의존적인 라우팅
- 생물학적으로 타당한 신용 할당

### 4. 발달 학습
- 단순에서 복잡으로 5단계 커리큘럼
- 점진적 아키텍처 성장
- 인간 인지 발달 모방

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

### 효율성 향상
- 밀집 모델보다 **95% 적은 활성 파라미터**
- 희소 커널로 **10-20배 빠른 추론**
- 엣지 배포를 위한 **5-10배 메모리 감소**

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

## 🙏 감사의 말

- 피질 기둥과 희소 코딩에 대한 신경과학 연구에서 영감을 받음
- 효율적인 희소 연산을 위해 PyTorch와 Triton으로 구축
- KLUE 및 KorQuAD 프로젝트의 한국어 데이터셋

## 📮 연락처

- 이슈: [GitHub Issues](https://github.com/comsa33/brain-inspired-gpt/issues)
- 이메일: comsa333@gmail.com

---

<div align="center">
Ruo Lee가 ❤️를 담아 만듦
</div>