# 📚 Brain-Inspired GPT Dataset Guide

This guide provides detailed information about available datasets and how to use them effectively for training Brain-Inspired GPT models.

## 🌟 Quick Start

For first-time users, we recommend starting with these commands:

```bash
# Prepare basic datasets (Korean + Wikipedia)
uv run prepare_all_datasets.py --datasets korean wikipedia

# Start multilingual training
uv run brain_gpt/training/train_multilingual.py --language-sampling balanced

# Test the model
uv run test_multilingual.py
```

## 📊 Available Datasets

### 1. **RedPajama-v2** (Recommended for Large-Scale Training)
- **Size**: 30 trillion tokens
- **Languages**: English, German, French, Spanish, Italian
- **Quality**: Includes 40+ quality annotations per document
- **Best for**: Large-scale multilingual models

```bash
# Sample dataset (for testing)
uv run data/openwebtext/prepare_redpajama.py --config sample --max-samples 10000

# Full dataset (warning: very large)
uv run data/openwebtext/prepare_redpajama.py --config default --languages en ko
```

### 2. **FineWeb-Edu** (Recommended for High-Quality Models)
- **Size**: 1.3 trillion tokens (educational subset)
- **Language**: English (primarily)
- **Quality**: Exceptionally high educational content
- **Best for**: Models focused on informative, educational responses

```bash
# Educational content
uv run data/openwebtext/prepare_fineweb.py --dataset-type fineweb-edu --max-samples 50000

# Full FineWeb (15T tokens)
uv run data/openwebtext/prepare_fineweb.py --dataset-type fineweb
```

### 3. **Korean Datasets** (Essential for Korean Support)
- **Size**: 50M+ tokens
- **Sources**: KLUE, KorQuAD, parallel corpora
- **Quality**: Curated Korean NLP datasets
- **Best for**: Korean language understanding

```bash
# Prepare Korean datasets
uv run brain_gpt/training/prepare_korean_hf_datasets.py --max-texts 100000
```

### 4. **Wikipedia** (Good for Quick Testing)
- **Size**: ~20B tokens per language
- **Languages**: 300+ available
- **Quality**: Encyclopedia-quality content
- **Best for**: Quick tests, factual knowledge

```bash
# English + Korean Wikipedia
uv run data/openwebtext/prepare_simple.py --datasets wikipedia wikipedia-ko
```

## 🔧 Dataset Features

### Quality Filtering

All dataset scripts include advanced quality filtering:

- **Perplexity-based filtering**: Remove low-quality or gibberish text
- **Educational scoring**: Prioritize informative content (FineWeb-Edu)
- **Language detection**: Accurate language identification
- **Length filtering**: Remove too short or too long documents
- **Deduplication**: Remove duplicate content

### Memory-Efficient Processing

- **Streaming support**: Process large datasets without loading into memory
- **Chunk processing**: Process data in manageable chunks
- **Binary format**: Efficient storage as uint16 tokens

## 💡 Training Strategies

### 1. **Balanced Multilingual Training**

For models that perform well across languages:

```bash
# Prepare multiple datasets
uv run prepare_all_datasets.py --datasets korean wikipedia fineweb

# Train with balanced sampling
uv run brain_gpt/training/train_multilingual.py \
  --data-dirs data/korean_hf data/simple data/fineweb \
  --language-sampling balanced
```

### 2. **Korean-Focused Training**

For models optimized for Korean:

```bash
# Prepare Korean datasets
uv run brain_gpt/training/prepare_korean_hf_datasets.py

# Train with Korean focus
uv run brain_gpt/training/train_korean.py
```

### 3. **High-Quality English Training**

For models with exceptional English capabilities:

```bash
# Prepare FineWeb-Edu
uv run data/openwebtext/prepare_fineweb.py --dataset-type fineweb-edu

# Train on educational content
uv run brain_gpt/training/train_brain_gpt_3090.py --data-dir data/fineweb
```

## 📈 Dataset Statistics

| Dataset | Total Size | Languages | Download Time* | Training Quality |
|---------|------------|-----------|----------------|------------------|
| Wikipedia | ~20B tokens | 300+ | 10-30 min | Good for factual knowledge |
| Korean HF | 50M+ tokens | Korean | 5-15 min | Essential for Korean |
| FineWeb-Edu | 1.3T tokens | English | 1-3 hours | Excellent quality |
| RedPajama-v2 | 30T tokens | 5 languages | 6-12 hours | Massive scale |

*Download times are approximate and depend on internet speed

## 🛠️ Advanced Options

### Custom Quality Thresholds

```bash
# Higher quality threshold (more selective)
uv run data/openwebtext/prepare_redpajama.py \
  --quality-threshold 0.8 \
  --max-samples 50000

# Lower threshold (more data)
uv run data/openwebtext/prepare_fineweb.py \
  --quality-threshold 0.5
```

### Language-Specific Filtering

```bash
# Only specific languages from RedPajama
uv run data/openwebtext/prepare_redpajama.py \
  --languages en de fr \
  --max-samples 100000
```

### Custom Dataset Mixing

```python
# In your training script
from brain_gpt.training.train_multilingual import MultilingualTrainingConfig

config = MultilingualTrainingConfig(
    data_dirs=['data/korean_hf', 'data/fineweb', 'data/simple'],
    language_weights={
        'ko': 0.4,  # 40% Korean
        'en': 0.5,  # 50% English  
        'mixed': 0.1  # 10% other
    },
    language_sampling='weighted'
)
```

## 🚀 Performance Tips

1. **Start Small**: Test with `--max-samples 10000` first
2. **Use SSD**: Store datasets on fast SSD for better training speed
3. **Parallel Downloads**: Scripts use multiple workers by default
4. **Monitor Disk Space**: Full datasets can be very large (100GB+)
5. **GPU Memory**: Adjust batch size based on your GPU memory

## 📝 Dataset Licenses

- **RedPajama-v2**: Apache 2.0
- **FineWeb**: ODC-By 1.0
- **Wikipedia**: CC BY-SA 3.0
- **Korean Datasets**: Various (check individual licenses)

Always check and comply with dataset licenses for your use case.

## 🤝 Contributing New Datasets

To add support for new datasets:

1. Create a preparation script in `data/openwebtext/`
2. Follow the existing script patterns
3. Include quality filtering and language detection
4. Test with small samples first
5. Submit a pull request

For questions or issues, please open an issue on GitHub.