# CortexGPT Architecture

## Overview

CortexGPT is a real-time learning language model inspired by human brain memory systems. It implements a three-tier memory architecture that mimics how humans process and retain information.

## Architecture Diagram

The architecture diagram in the README uses Mermaid syntax, which is automatically rendered by GitHub. You can view it directly in the README file.

## Components

### 1. Input Layer
- **Multilingual Tokenizer**: Supports Korean and English text processing
- **BPE Encoding**: Byte Pair Encoding for efficient tokenization
- **Language Detection**: Automatic language identification

### 2. Transformer Core
- **Multi-Head Attention**: 8 attention heads for parallel processing
- **Feed Forward Network**: Two-layer MLP with GELU activation
- **Layer Normalization**: Applied before each sub-layer
- **Residual Connections**: Skip connections for gradient flow

### 3. Memory System

#### STM (Short-Term Memory)
- **Capacity**: 64 entries
- **Purpose**: Store recent interactions and context
- **Access**: O(1) fast lookup
- **Features**:
  - Attention-based retrieval
  - Automatic overflow to LTM
  - Context preservation

#### LTM (Long-Term Memory)
- **Capacity**: 10,000 entries
- **Purpose**: Store consolidated knowledge from repeated patterns
- **Access**: FAISS-based similarity search
- **Features**:
  - Vector similarity search
  - Importance-based ranking
  - Gradual consolidation from STM

#### Archive Memory
- **Capacity**: 100,000 entries
- **Purpose**: Long-term storage for rarely accessed knowledge
- **Access**: Compressed vector search
- **Features**:
  - Memory compression
  - Lazy loading
  - Periodic cleanup

### 4. Real-Time Learner
- **Online Learning**: Updates model weights during inference
- **Memory Consolidation**: Transfers knowledge between memory tiers
- **Self-Evaluation**: Monitors performance and adjusts learning
- **Features**:
  - Hebbian learning rules
  - Confidence scoring
  - Adaptive learning rates

### 5. Output Layer
- **Token Generation**: Next token prediction
- **Confidence Scoring**: Uncertainty estimation
- **Language Detection**: Output language identification

## Memory Flow

```
1. New input → Tokenization → Transformer processing
2. Context stored in STM for immediate recall
3. Frequently accessed STM entries → LTM consolidation
4. Rarely accessed LTM entries → Archive compression
5. Real-time learner monitors all memory operations
6. Continuous weight updates based on memory patterns
```

## Key Innovations

1. **Three-Tier Memory**: Mimics human memory organization
2. **Real-Time Learning**: No separate training/inference phases
3. **Memory Consolidation**: Automatic knowledge transfer
4. **Multilingual Support**: Native Korean and English processing
5. **Adaptive Batch Sizing**: Prevents OOM errors dynamically

## Technical Specifications

- **Model Dimensions**: 256/512/768 (configurable)
- **Vocabulary Size**: 10,000-50,000 tokens
- **Memory Update Frequency**: Every 500 steps
- **Learning Rate**: Adaptive (1e-3 to 3e-4)
- **Attention Heads**: 8
- **Feed Forward Dimension**: 4x hidden dimension

## Implementation Details

The architecture is implemented across several key modules:

- `cortexgpt/models/cortex_gpt.py`: Base transformer architecture
- `cortexgpt/models/realtime_cortex.py`: Memory system and real-time learning
- `cortexgpt/learning/realtime_learner.py`: Online learning algorithms
- `cortexgpt/tokenization/multilingual_tokenizer.py`: BPE tokenizer

## Future Enhancements

1. **Episodic Memory**: Add specific event recall capability
2. **Working Memory**: Implement task-specific temporary storage
3. **Sleep Consolidation**: Offline memory reorganization
4. **Attention Visualization**: Real-time attention pattern display
5. **Multi-Modal Support**: Extend to image and audio processing