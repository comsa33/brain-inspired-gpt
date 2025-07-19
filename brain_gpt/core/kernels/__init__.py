"""
Optimized CUDA kernels for Brain-Inspired GPT
"""

from .dendritic_attention_cuda import (
    DendriticAttentionCUDA,
    cortical_sparse_mm,
    CorticalColumnSparseMM
)