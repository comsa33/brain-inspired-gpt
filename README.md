<div align="center">

# 🧠 CortexGPT

**Real-time Learning Language Model Inspired by Human Brain**

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Model Size](https://img.shields.io/badge/model-768D-purple.svg)
![Memory](https://img.shields.io/badge/memory-STM→LTM→Archive-orange.svg)

[English](#english) | [한국어](README_KR.md)

</div>

## English

### 🏛️ Architecture

```mermaid
graph TB
    subgraph "CortexGPT Model"
        Input["📥 Input Layer<br/>• Multilingual Tokenizer<br/>• Korean/English Support<br/>• BPE Encoding"]
        
        Transformer["🤖 Transformer Core<br/>• Multi-Head Attention<br/>• Feed Forward Network<br/>• Layer Normalization<br/>• Residual Connections"]
        
        subgraph "Memory System"
            STM["💭 STM (Short-Term Memory)<br/>• Capacity: 64<br/>• Fast Access<br/>• Recent Interactions"]
            LTM["🧠 LTM (Long-Term Memory)<br/>• Capacity: 10,000<br/>• Consolidated Knowledge<br/>• Frequent Patterns"]
            Archive["📚 Archive Memory<br/>• Capacity: 100,000<br/>• Compressed Storage<br/>• Rarely Used Knowledge"]
        end
        
        Learner["🎓 Real-Time Learner<br/>• Online Learning<br/>• Memory Consolidation<br/>• Self-Evaluation"]
        
        Output["📤 Output Layer<br/>• Token Generation<br/>• Confidence Scoring<br/>• Language Detection"]
    end
    
    Input --> |"Encoded Tokens"| Transformer
    Transformer --> |"Store Context"| STM
    STM --> |"Consolidate<br/>(Frequent Use)"| LTM
    LTM --> |"Archive<br/>(Rare Use)"| Archive
    STM --> |"Current Context"| Learner
    LTM --> |"Retrieved Knowledge"| Learner
    Learner --> |"Updated Weights"| Transformer
    Transformer --> |"Predictions"| Output
    Learner -.-> |"Update"| STM
    Learner -.-> |"Transfer"| LTM
    
    style Input fill:#e6f3ff
    style Transformer fill:#e6f3ff
    style STM fill:#ffe6e6
    style LTM fill:#e6ffe6
    style Archive fill:#e6e6ff
    style Learner fill:#e6f3ff
    style Output fill:#e6f3ff
```

### 🌟 Key Features

- **Real-time Learning**: Continuously learns without train/inference distinction
- **Human-like Memory**: STM (Short-term) → LTM (Long-term) → Archive system
- **Self-Improvement**: Self-evaluation and improvement mechanisms
- **Multilingual Support**: Natural processing of Korean and English
- **Memory Efficiency**: Adaptive batch sizing to prevent OOM
- **Checkpoint Support**: Resume training after interruptions

### 🚀 Quick Start

#### 1. Installation

```bash
# Clone repository
git clone https://github.com/comsa33/cortexgpt.git
cd cortexgpt

# Install all dependencies
uv sync

# Or install with optional monitoring tools
uv sync --extra monitoring
```

#### 2. Create Demo Data

```bash
# Create demo training data
uv run scripts/data/create_demo_data.py
```

#### 3. Test Basic Functionality

```bash
# Test tokenizer
uv run tests/demo_tokenizer.py

# Test if model can learn (overfitting test)
uv run tests/test_overfit.py
```

#### 4. Training

```bash
# Quick demo training (small model, fast)
uv run cortexgpt/training/train_realtime.py \
    --dataset demo \
    --dim 256 \
    --lr 1e-3 \
    --epochs 20

# Monitor with wandb
uv run cortexgpt/training/train_realtime.py \
    --dataset demo \
    --dim 512 \
    --wandb

# Train with real datasets (after setup)
uv run scripts/data/setup_datasets.py  # Download & prepare datasets
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --batch-size 4 \
    --gradient-accumulation 8 \
    --epochs 50 \
    --wandb

# Resume interrupted training
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --resume auto
```

#### 5. Run Demos

```bash
# Minimal generation demo
uv run scripts/demos/minimal_demo.py

# Real-time learning demo
uv run scripts/demos/learning_effect_demo.py

# Interactive chat demo
uv run scripts/demos/natural_language_demo.py
```

### 📖 Detailed Usage Guide

#### Using Pre-trained Models

```bash
# Load a checkpoint and generate text
uv run cortexgpt/inference/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "The future of AI is" \
    --max-length 100

# Interactive chat with a trained model
uv run cortexgpt/inference/chat.py \
    --checkpoint checkpoints/best_model.pt \
    --temperature 0.8
```

#### Real-time Learning Demo

The real-time learning demo shows how CortexGPT learns from interactions:

```bash
# Run the learning effect demo
uv run scripts/demos/learning_effect_demo.py
```

This demonstrates:
- Initial response without knowledge
- Learning from user feedback
- Improved responses after learning
- Memory consolidation over time

#### Custom Training

For custom datasets, create a JSONL file with your data:

```json
{"text": "Your training text here"}
{"text": "Another training example"}
```

Then train:

```bash
# Prepare your custom dataset
uv run cortexgpt/data/prepare_custom.py \
    --input your_data.jsonl \
    --output data/custom

# Train on custom data
uv run cortexgpt/training/train_realtime.py \
    --dataset custom \
    --vocab-size 30000 \
    --epochs 50
```

#### Memory System Configuration

Adjust memory system parameters for different use cases:

```bash
# Small memory for quick experiments
uv run cortexgpt/training/train_realtime.py \
    --stm-capacity 32 \
    --ltm-capacity 1000 \
    --archive-capacity 10000

# Large memory for production
uv run cortexgpt/training/train_realtime.py \
    --stm-capacity 128 \
    --ltm-capacity 50000 \
    --archive-capacity 500000
```

#### API Usage

```python
from cortexgpt import CortexGPT, MultilingualTokenizer

# Initialize model and tokenizer
model = CortexGPT.from_pretrained("checkpoints/best_model.pt")
tokenizer = MultilingualTokenizer.from_pretrained("checkpoints/tokenizer.json")

# Generate text
prompt = "인공지능의 미래는"
inputs = tokenizer.encode(prompt)
outputs = model.generate(inputs, max_length=100)
response = tokenizer.decode(outputs)
print(response)

# Real-time learning
from cortexgpt.learning import RealTimeLearner

learner = RealTimeLearner(model, tokenizer)
learner.start()  # Start background learning

# Process queries with learning
response, metadata = learner.process_query(
    "What is machine learning?",
    learn=True
)
print(f"Response: {response}")
print(f"Confidence: {metadata['confidence']}")
```

#### Monitoring Training

Use Weights & Biases for detailed monitoring:

```bash
# First login to wandb
wandb login

# Train with monitoring
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --wandb \
    --wandb-project "cortexgpt-experiments" \
    --wandb-name "run-001"
```

Monitor:
- Training/validation loss
- Learning rate schedule
- Memory system usage
- Sample generations
- Performance metrics

### 🌍 Training with Real Datasets

#### Step 1: Download Datasets

```bash
# Download sample datasets (KLUE, Wikipedia, etc.)
uv run cortexgpt/data/download_datasets.py
```

This downloads samples from:
- **KLUE**: Korean Language Understanding dataset
- **Korean Wikipedia**: Korean encyclopedia articles
- **English Wikipedia**: English encyclopedia articles
- **OpenWebText**: Web crawl data (sample)

#### Step 2: Prepare Datasets (Optional)

The training script automatically handles JSONL files, but you can pre-process for faster loading:

```bash
# Prepare all downloaded datasets
uv run cortexgpt/data/prepare_datasets.py
```

#### Step 3: Train on Real Data

##### Korean Dataset (KLUE)
```bash
# Train on KLUE dataset
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

##### English Dataset (Wikipedia)
```bash
# Train on English Wikipedia
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

##### Mixed Korean-English Training
```bash
# Train on combined datasets
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

#### Step 4: Resume Training

If training is interrupted:

```bash
# Resume from latest checkpoint
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --resume auto \
    --wandb

# Resume from specific checkpoint
uv run cortexgpt/training/train_realtime.py \
    --dataset klue \
    --resume checkpoints/realtime/model_best.pt \
    --wandb
```

#### Training Tips

1. **Start Small**: Begin with `--dim 256` and `--vocab-size 10000` for testing
2. **Monitor Memory**: Use `--batch-size 2` and increase `--gradient-accumulation` if OOM
3. **Learning Rate**: Start with `1e-3` for small models, `3e-4` for larger ones
4. **Vocabulary Size**: 
   - Korean only: 20,000-30,000
   - English only: 30,000-40,000
   - Mixed: 40,000-50,000

### 📊 Available Datasets

| Dataset | Language | Description |
|---------|----------|-------------|
| `demo` | Mixed | Small test dataset (default) |
| `klue` | Korean | Korean Language Understanding Evaluation |
| `wikipedia` | English | Wikipedia articles |
| `korean_wiki` | Korean | Korean Wikipedia |
| `openwebtext` | English | Web text for GPT-2 training |
| `combined` | Mixed | Multiple datasets combined |

### 🏗️ Project Structure

```
my-efficient-gpt/
├── cortexgpt/              # Main package
│   ├── models/            # Model architectures
│   ├── learning/          # Real-time learning system
│   ├── tokenization/      # Multilingual tokenizer
│   ├── data/             # Data loading utilities
│   └── training/         # Training scripts
├── scripts/
│   ├── data/             # Data preparation scripts
│   └── demos/            # Demo applications
├── tests/                # Test scripts
├── docs/                 # Documentation
└── data/                 # Training data
```

### 💡 How It Works

#### Memory Flow
```
New Input → STM (Fast Access)
     ↓ (Frequently Used)
    LTM (Consolidated Knowledge)
     ↓ (Long Unused)
   Archive (Compressed Storage)
```

#### Learning Process
1. **First Query**: "Not learned yet"
2. **After Learning**: Provides accurate answer
3. **With Repetition**: Confidence increases (0.6 → 0.9 → 1.0)

### 📈 Training Options

```bash
# Model Architecture
--dim               # Hidden dimension (256/512/768, default: 768)
--vocab-size        # Tokenizer vocabulary size (default: 50000)

# Training Parameters
--batch-size        # Batch size (default: 8)
--gradient-accumulation  # Gradient accumulation steps (default: 4)
--epochs           # Number of epochs (default: 10)
--lr              # Learning rate (default: 3e-4)

# Memory System
--stm-capacity     # Short-term memory capacity (default: 64)
--ltm-capacity     # Long-term memory capacity (default: 10000)
--archive-capacity # Archive capacity (default: 100000)

# Monitoring & Checkpointing
--wandb           # Enable Weights & Biases logging
--wandb-project   # W&B project name
--checkpoint-dir  # Checkpoint directory
--resume         # Resume from checkpoint (auto/path)
```

### 🚀 Recommended Training Configurations

#### Testing & Development
```bash
# Small model for quick testing
--dim 256 --lr 1e-3 --batch-size 4 --epochs 20
```

#### Demo Training
```bash
# Medium model for demos
--dim 512 --lr 5e-4 --batch-size 8 --gradient-accumulation 4
```

#### Production Training
```bash
# Large model for real training
--dim 768 --lr 3e-4 --batch-size 4 --gradient-accumulation 8 --wandb
```

### 🔬 Research & Development

CortexGPT implements several neuroscience-inspired concepts:

- **Hebbian Learning**: "Neurons that fire together, wire together"
- **Memory Consolidation**: Gradual transfer from STM to LTM
- **Selective Attention**: Focus on relevant information
- **Continual Learning**: Learn new tasks without forgetting

### 📝 Citation

```bibtex
@software{cortexgpt2024,
  author = {Ruo Lee},
  title = {CortexGPT: Real-time Learning Language Model},
  year = {2025},
  email = {comsa333@gmail.com}
}
```

### 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

Made with ❤️ by Ruo Lee