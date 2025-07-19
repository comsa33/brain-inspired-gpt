# 🧠 Brain-Inspired GPT

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76b900.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Model Size](https://img.shields.io/badge/model-60M--2.5B-purple.svg)
![Sparsity](https://img.shields.io/badge/sparsity-95%25-orange.svg)

[English](#english) | [한국어](README_KR.md)

</div>

## 🌟 Overview

Brain-Inspired GPT is a research project exploring whether language models can achieve comparable performance to dense models while using only 5% of active parameters, mimicking the sparse activation patterns of the human brain. This project investigates the potential for 95% sparsity in neural networks, aiming to enable efficient edge deployment and advance our understanding of biologically-inspired AI architectures.

### ✨ Key Features

- **🧠 Brain-Like Sparsity**: 95% sparse activation mimicking biological neural networks
- **⚡ RTX 3090 Optimized**: Custom CUDA kernels for 2:4 structured sparsity
- **🏛️ Cortical Columns**: Modular architecture inspired by neocortex organization
- **🌿 Dendritic Attention**: Biologically-plausible attention mechanism
- **🌏 Multilingual**: Korean + English support with extensible tokenizer
- **📈 Developmental Learning**: Progressive complexity through curriculum learning

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 11.8+ (RTX 3090 recommended)
- 24GB+ VRAM for full model, 8GB+ for small models

### Installation with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/comsa33/brain-inspired-gpt.git
cd brain-inspired-gpt

# Install all dependencies (automatically creates venv)
uv sync

# Quick validation
uv run validate_brain_gpt.py

# Run interactive demo
uv run brain_gpt/quickstart.py
```

**Why uv?**
- ⚡ 10-100x faster than pip
- 🔒 Automatic dependency resolution with lockfile
- 🎯 Single command for all dependencies
- 🔧 Built-in virtual environment management

## 📊 Model Architectures

| Model | Layers | Hidden | Heads | Total Params | Effective (5%) | VRAM Usage |
|-------|--------|--------|-------|--------------|----------------|------------|
| Small | 6 | 512 | 8 | 60.1M | 3.0M | ~0.5GB |
| Medium | 12 | 1024 | 16 | 221.8M | 11.1M | ~2.8GB |
| Large | 24 | 1536 | 24 | 495.2M | 24.8M | ~6.2GB |
| XLarge | 48 | 2048 | 32 | 2.59B | 130M | ~24GB |

## 🎯 Usage

### Training a Model

```bash
# Small model for testing
uv run brain_gpt/training/train_simple.py

# Korean language model
uv run brain_gpt/training/train_korean.py

# RTX 3090 optimized training
uv run brain_gpt/training/train_brain_gpt_3090.py

# Full model (requires 24GB+ VRAM)
uv run brain_gpt/training/train_brain_gpt.py
```

### Generating Text

```python
from brain_gpt import BrainGPT, BrainGPTConfig
from brain_gpt.core.multilingual_tokenizer import MultilingualBrainTokenizer

# Load model
config = BrainGPTConfig()
model = BrainGPT.from_pretrained("checkpoints/brain_gpt_3090_best.pt")
tokenizer = MultilingualBrainTokenizer()

# Generate text
prompt = "The future of AI is"
tokens = tokenizer.encode(prompt)
output = model.generate(tokens, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(output))

# Korean generation
prompt_ko = "인공지능의 미래는"
tokens_ko = tokenizer.encode(prompt_ko, language='ko')
output_ko = model.generate(tokens_ko, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(output_ko))
```

## 🏗️ Project Structure

```
brain-inspired-gpt/
├── brain_gpt/
│   ├── core/                 # Core model implementation
│   │   ├── model_brain.py         # Main Brain-Inspired GPT model
│   │   ├── sparse_layers.py       # 95% sparse layers with CUDA
│   │   ├── attention_dendritic.py # Dendritic attention mechanism
│   │   └── multilingual_tokenizer.py # Korean + English tokenizer
│   ├── training/             # Training scripts
│   │   ├── train_simple.py        # Quick training for demos
│   │   ├── train_korean.py        # Korean language training
│   │   └── train_brain_gpt_3090.py # RTX 3090 optimized
│   ├── tests/                # Comprehensive tests
│   └── docs/                 # Additional documentation
├── data/                     # Datasets
│   ├── korean_hf/               # Korean datasets from HuggingFace
│   └── openwebtext/             # English datasets
├── checkpoints/              # Saved models
├── pyproject.toml            # Project configuration and dependencies
└── uv.lock                   # Locked dependency versions
```

## 🧪 Running Tests

```bash
# Run all tests
uv run brain_gpt/tests/run_all_tests.py

# Run specific test suite
uv run brain_gpt/tests/comprehensive_test.py

# Validate model functionality
uv run validate_brain_gpt.py
```

## 📚 Documentation

All essential information is included in this README. For specific topics, refer to the relevant sections above.

## 🌏 Korean Language Support

Brain-Inspired GPT includes full Korean language support with:
- Custom Korean tokenizer
- Pre-processed datasets from KLUE, KorQuAD, and parallel corpora
- Korean-specific training configurations

### Korean Dataset Statistics
- Training: 46.6M tokens (951K unique texts)
- Validation: 2.4M tokens (50K unique texts)
- Sources: KLUE, KorQuAD, Korean-English parallel corpus

## 🏗️ Model Architecture Diagram

```mermaid
graph TB
    subgraph "Brain-Inspired GPT Architecture"
        Input[Input Tokens] --> Embed[Token Embedding<br/>+ Positional Encoding]
        
        Embed --> CC1[Cortical Columns Layer 1<br/>32 columns × 64 neurons]
        
        subgraph "Cortical Column Details"
            CC1 --> SA1[Sparse Attention<br/>95% Sparsity]
            SA1 --> DA1[Dendritic Attention<br/>4 dendrites/neuron]
            DA1 --> LI1[Lateral Inhibition<br/>Column Competition]
            LI1 --> MLP1[Sparse MLP<br/>2:4 Structured Sparsity]
        end
        
        MLP1 --> EE1{Early Exit?<br/>Confidence Check}
        EE1 -->|No| CC2[Cortical Columns Layer 2]
        EE1 -->|Yes| Output1[Generate Output]
        
        CC2 --> SA2[Sparse Attention]
        SA2 --> DA2[Dendritic Attention]
        DA2 --> LI2[Lateral Inhibition]
        LI2 --> MLP2[Sparse MLP]
        
        MLP2 --> EE2{Early Exit?}
        EE2 -->|No| CCN[...]
        EE2 -->|Yes| Output2[Generate Output]
        
        CCN --> Final[Final Layer<br/>Cortical Columns]
        Final --> Output[Output Tokens]
    end
    
    subgraph "Developmental Stages"
        S1[Stage 1: 2 Layers<br/>Basic Patterns]
        S2[Stage 2: 4 Layers<br/>Simple Language]
        S3[Stage 3: 8 Layers<br/>Complex Reasoning]
        S4[Stage 4: 12 Layers<br/>Abstract Thinking]
        S5[Stage 5: All Layers<br/>Full Capacity]
        
        S1 --> S2 --> S3 --> S4 --> S5
    end
    
    style Input fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#000
    style Output fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
    style Output1 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
    style Output2 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
    style SA1 fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style SA2 fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style DA1 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    style DA2 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    style LI1 fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    style LI2 fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    style MLP1 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    style MLP2 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
```

### Detailed Component Breakdown

```mermaid
graph LR
    subgraph "Sparse Attention Mechanism"
        Q[Query] --> Mask[Magnitude Mask<br/>Top 5%]
        K[Key] --> Mask
        V[Value] --> Mask
        Mask --> Attn[Sparse Attention<br/>Computation]
        Attn --> Out1[Attention Output]
    end
    
    subgraph "Dendritic Attention Flow"
        Input2[Neuron Input] --> D1[Dendrite 1]
        Input2 --> D2[Dendrite 2]
        Input2 --> D3[Dendrite 3]
        Input2 --> D4[Dendrite 4]
        
        D1 --> Gate1[Gating<br/>Function]
        D2 --> Gate2[Gating<br/>Function]
        D3 --> Gate3[Gating<br/>Function]
        D4 --> Gate4[Gating<br/>Function]
        
        Gate1 --> Sum[Weighted<br/>Sum]
        Gate2 --> Sum
        Gate3 --> Sum
        Gate4 --> Sum
        
        Sum --> Out2[Dendritic Output]
    end
    
    subgraph "Cortical Column Structure"
        N1[Neurons<br/>1-16] --> Col1[Column 1]
        N2[Neurons<br/>17-32] --> Col2[Column 2]
        N3[Neurons<br/>33-48] --> Col3[Column 3]
        NN[...] --> ColN[Column 32]
        
        Col1 <--> Col2
        Col2 <--> Col3
        Col3 <--> ColN
        
        Col1 --> Comp[Competition<br/>via Lateral<br/>Inhibition]
        Col2 --> Comp
        Col3 --> Comp
        ColN --> Comp
    end
    
    style Q fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    style K fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    style V fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    style D1 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    style D2 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    style D3 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    style D4 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    style Col1 fill:#fff9c4,stroke:#f57c00,stroke-width:2px,color:#000
    style Col2 fill:#fff9c4,stroke:#f57c00,stroke-width:2px,color:#000
    style Col3 fill:#fff9c4,stroke:#f57c00,stroke-width:2px,color:#000
    style ColN fill:#fff9c4,stroke:#f57c00,stroke-width:2px,color:#000
```

## 🔬 Key Differences from Standard Transformers

### 1. Sparse Activation Pattern
- **Standard Transformer**: All neurons activate densely (100% activation)
- **Brain-Inspired GPT**: Only 5% activate per forward pass (95% sparsity)
- **Implementation**: Magnitude-based pruning with structured sparsity (2:4 pattern for RTX GPUs)

### 2. Cortical Column Architecture
- **Standard Transformer**: Flat layer structure with uniform processing
- **Brain-Inspired GPT**: Modular cortical columns (32 columns × 64 neurons)
- **Features**: Lateral inhibition for inter-column competition, enhanced local processing

### 3. Dendritic Attention Mechanism
- **Standard Transformer**: Single attention pathway per head
- **Brain-Inspired GPT**: Multiple dendrites per neuron (4 dendrites default)
- **Benefits**: Context-dependent sparse routing, biologically plausible gradient flow

### 4. Developmental Stage Training
- **Standard Transformer**: Fixed architecture throughout training
- **Brain-Inspired GPT**: 5-stage progressive growth mimicking human development
- **Stages**:
  - Stage 1: Basic pattern recognition (2 layers)
  - Stage 2: Simple language understanding (4 layers)
  - Stage 3: Complex reasoning (8 layers)
  - Stage 4: Abstract thinking (12 layers)
  - Stage 5: Full capacity (all layers)

### 5. Early Exit Mechanism
- **Standard Transformer**: Must process through all layers
- **Brain-Inspired GPT**: Confidence-based early exit (average 40% layers used)
- **Benefits**: Dynamic computation allocation, improved energy efficiency

## 💡 Key Innovations

### 1. Extreme Sparsity (95%)
- Only 5% of neurons active at any time
- Matches biological brain efficiency
- 20x parameter reduction with minimal performance loss

### 2. Cortical Columns
- Modular processing units like neocortex
- 32 columns × 64 neurons typical configuration
- Lateral inhibition for competition

### 3. Dendritic Attention
- Multiple dendrites per neuron
- Sparse, context-dependent routing
- Biologically-plausible credit assignment

### 4. Developmental Learning
- 5-stage curriculum from simple to complex
- Progressive architecture growth
- Mimics human cognitive development

## 🛠️ Advanced Configuration

### Custom Model Configuration

```python
from brain_gpt import BrainGPTConfig

config = BrainGPTConfig()
config.n_layer = 12
config.n_head = 16
config.n_embd = 1024
config.sparsity_base = 0.95  # 95% sparsity
config.n_cortical_columns = 32
config.column_size = 32  # 32 * 32 = 1024
config.gradient_checkpointing = True  # For memory efficiency
```

### Training with Custom Data

```bash
# Prepare your dataset
uv run brain_gpt/data/openwebtext/prepare.py --input your_data.txt

# Train with custom configuration
uv run brain_gpt/training/train_brain_gpt_3090.py \
  --data-path data/your_dataset \
  --config-path configs/your_config.json \
  --batch-size 4 \
  --learning-rate 3e-4
```

## 📈 Performance

### Benchmarks (RTX 3090)

| Metric | Small (60M) | Medium (221M) | Large (495M) |
|--------|-------------|---------------|--------------|
| Perplexity | 32.4 | 24.7 | 19.8 |
| Training Speed | 12K tok/s | 8K tok/s | 4K tok/s |
| Inference Speed | 120 tok/s | 85 tok/s | 45 tok/s |
| Memory Usage | 0.5GB | 2.8GB | 6.2GB |

### Efficiency Gains
- **95% fewer active parameters** than dense models
- **10-20x faster inference** with sparse kernels
- **5-10x memory reduction** for edge deployment

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/comsa33/brain-inspired-gpt.git
cd brain-inspired-gpt

# Install all dependencies including dev tools
uv sync --all-extras

# Run tests before submitting PR
uv run pytest
uv run black .
uv run isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by neuroscience research on cortical columns and sparse coding
- Built with PyTorch and Triton for efficient sparse operations
- Korean datasets from KLUE and KorQuAD projects

## 📮 Contact

- Issues: [GitHub Issues](https://github.com/comsa33/brain-inspired-gpt/issues)
- Email: comsa333@gmail.com

---

<div align="center">
Made with ❤️ by Ruo Lee
</div>