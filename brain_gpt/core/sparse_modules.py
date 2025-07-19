"""
Sparse modules implementing cortical column architecture
Optimized for RTX 3090 with 2:4 structured sparsity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np

try:
    from kernels import cortical_sparse_mm
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA kernels not available, using PyTorch fallback")


class StructuredSparseMask:
    """
    Creates and manages structured sparsity patterns optimized for tensor cores
    """
    
    @staticmethod
    def create_2_4_pattern(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """Create 2:4 structured sparsity pattern (50% sparse)"""
        mask = torch.zeros(shape, device=device)
        
        # Create 2:4 pattern along last dimension
        for i in range(0, shape[1], 4):
            if i + 4 <= shape[1]:
                # Select 2 random positions out of 4
                indices = torch.randperm(4)[:2] + i
                mask[:, indices] = 1.0
            else:
                # Handle remainder
                remaining = shape[1] - i
                indices = torch.randperm(remaining)[:remaining//2] + i
                mask[:, indices] = 1.0
                
        return mask
    
    @staticmethod
    def create_block_sparse_pattern(
        shape: Tuple[int, int], 
        block_size: int = 32,
        sparsity: float = 0.9,
        device: torch.device = None
    ) -> torch.Tensor:
        """Create block sparse pattern aligned with GPU architecture"""
        mask = torch.zeros(shape, device=device)
        
        n_blocks_h = (shape[0] + block_size - 1) // block_size
        n_blocks_w = (shape[1] + block_size - 1) // block_size
        total_blocks = n_blocks_h * n_blocks_w
        
        # Select active blocks
        n_active = int(total_blocks * (1 - sparsity))
        active_blocks = torch.randperm(total_blocks)[:n_active]
        
        for block_idx in active_blocks:
            block_i = block_idx // n_blocks_w
            block_j = block_idx % n_blocks_w
            
            start_i = block_i * block_size
            end_i = min(start_i + block_size, shape[0])
            start_j = block_j * block_size
            end_j = min(start_j + block_size, shape[1])
            
            mask[start_i:end_i, start_j:end_j] = 1.0
            
        return mask
    
    @staticmethod
    def create_cortical_pattern(
        n_columns: int,
        column_size: int,
        inter_column_density: float = 0.1,
        device: torch.device = None
    ) -> torch.Tensor:
        """Create cortical column connectivity pattern"""
        size = n_columns * column_size
        mask = torch.zeros(size, size, device=device)
        
        # Dense connections within columns
        for i in range(n_columns):
            start = i * column_size
            end = start + column_size
            mask[start:end, start:end] = 1.0
        
        # Sparse connections between columns
        for i in range(n_columns):
            for j in range(n_columns):
                if i != j:
                    start_i = i * column_size
                    end_i = start_i + column_size
                    start_j = j * column_size
                    end_j = start_j + column_size
                    
                    # Random sparse connections
                    inter_mask = torch.rand(column_size, column_size, device=device) < inter_column_density
                    mask[start_i:end_i, start_j:end_j] = inter_mask.float()
                    
        return mask


class CorticalColumnLinear(nn.Module):
    """
    Linear layer with cortical column organization and structured sparsity
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_columns: int = 32,
        sparsity: float = 0.95,
        use_2_4_pattern: bool = True,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_columns = n_columns
        self.column_size = out_features // n_columns
        self.sparsity = sparsity
        self.use_2_4_pattern = use_2_4_pattern
        
        # Initialize weight with proper scaling
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Create sparsity mask
        if use_2_4_pattern and out_features % 4 == 0:
            self.mask = StructuredSparseMask.create_2_4_pattern(
                (out_features, in_features), device=torch.cuda.current_device()
            )
        else:
            self.mask = StructuredSparseMask.create_cortical_pattern(
                n_columns, self.column_size, 
                inter_column_density=1-sparsity,
                device=torch.cuda.current_device()
            )
            
        self.register_buffer('sparse_mask', self.mask)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Xavier initialization scaled for sparsity
        fan_in = self.in_features * (1 - self.sparsity)
        fan_out = self.out_features * (1 - self.sparsity)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        
        with torch.no_grad():
            self.weight.normal_(0, std)
            # Apply mask to initial weights
            self.weight.mul_(self.sparse_mask)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply structured sparsity mask
        sparse_weight = self.weight * self.sparse_mask
        
        if CUDA_AVAILABLE and x.is_cuda:
            # Use optimized CUDA kernel
            output = cortical_sparse_mm(x, sparse_weight.t())
        else:
            # Fallback to PyTorch
            output = F.linear(x, sparse_weight, self.bias)
            
        return output
    
    def get_active_neurons(self) -> float:
        """Return percentage of active neurons"""
        return (self.sparse_mask.sum() / self.sparse_mask.numel()).item()


class SpikingCorticalColumn(nn.Module):
    """
    Spiking neural network implementation of cortical column
    Uses leaky integrate-and-fire neurons with STDP-like learning
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_columns: int = 32,
        threshold: float = 1.0,
        decay: float = 0.9,
        refractory_period: int = 5
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_columns = n_columns
        self.column_size = out_features // n_columns
        self.threshold = threshold
        self.decay = decay
        self.refractory_period = refractory_period
        
        # Synaptic weights
        self.weight = CorticalColumnLinear(
            in_features, out_features, n_columns, 
            sparsity=0.95, use_2_4_pattern=True
        )
        
        # Neuron state
        self.register_buffer('membrane_potential', torch.zeros(1, out_features))
        self.register_buffer('refractory_count', torch.zeros(1, out_features))
        self.register_buffer('spike_history', torch.zeros(1, out_features, 10))
        
        # Adaptive threshold
        self.adaptive_threshold = nn.Parameter(torch.ones(n_columns) * threshold)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        batch_size = x.shape[0]
        
        # Ensure state tensors match batch size
        if self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(batch_size, self.out_features, device=x.device)
            self.refractory_count = torch.zeros(batch_size, self.out_features, device=x.device)
            
        # Synaptic input
        synaptic_input = self.weight(x)
        
        # Update membrane potential (only for non-refractory neurons)
        active_mask = (self.refractory_count == 0).float()
        self.membrane_potential = (
            self.decay * self.membrane_potential + synaptic_input
        ) * active_mask
        
        # Generate spikes
        threshold_per_neuron = self.adaptive_threshold.repeat_interleave(self.column_size)
        spikes = (self.membrane_potential > threshold_per_neuron).float()
        
        # Reset spiking neurons
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        
        # Update refractory period
        self.refractory_count = torch.maximum(
            self.refractory_count - 1,
            spikes * self.refractory_period
        )
        
        # Update spike history
        self.spike_history = torch.cat([
            self.spike_history[:, :, 1:],
            spikes.unsqueeze(2)
        ], dim=2)
        
        # Compute output (spike rate over recent history)
        output = self.spike_history.mean(dim=2)
        
        # Statistics
        stats = {
            'spike_rate': spikes.mean().item(),
            'active_neurons': (spikes > 0).float().mean().item(),
            'membrane_mean': self.membrane_potential.mean().item(),
            'membrane_std': self.membrane_potential.std().item()
        }
        
        return output, stats


class HierarchicalSparseMLP(nn.Module):
    """
    MLP with hierarchical sparsity mimicking cortical hierarchy
    Early layers: 10% active, Middle: 5%, Deep: 2%
    """
    
    def __init__(
        self,
        config,
        layer_idx: int,
        hidden_multiplier: int = 4
    ):
        super().__init__()
        hidden_size = config.n_embd * hidden_multiplier
        
        # Determine sparsity based on layer depth
        self.sparsity = config.get_sparse_pattern(layer_idx)
        
        # Use appropriate sparsity pattern
        if layer_idx < config.n_layer // 3:
            # Early layers - block sparse
            self.w1 = CorticalColumnLinear(
                config.n_embd, hidden_size,
                n_columns=16, sparsity=self.sparsity,
                use_2_4_pattern=False
            )
            self.w2 = CorticalColumnLinear(
                hidden_size, config.n_embd,
                n_columns=16, sparsity=self.sparsity,
                use_2_4_pattern=False
            )
        else:
            # Later layers - 2:4 structured sparsity
            self.w1 = CorticalColumnLinear(
                config.n_embd, hidden_size,
                n_columns=32, sparsity=self.sparsity,
                use_2_4_pattern=True
            )
            self.w2 = CorticalColumnLinear(
                hidden_size, config.n_embd,
                n_columns=32, sparsity=self.sparsity,
                use_2_4_pattern=True
            )
            
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        h = self.activation(h)
        h = self.w2(h)
        return h
    
    def get_sparsity_info(self) -> dict:
        return {
            'target_sparsity': self.sparsity,
            'w1_active': self.w1.get_active_neurons(),
            'w2_active': self.w2.get_active_neurons()
        }


class AdaptiveSparsityGate(nn.Module):
    """
    Dynamically adjusts sparsity based on input complexity
    Mimics metabolic constraints in biological neurons
    """
    
    def __init__(self, dim: int, min_active: float = 0.01, max_active: float = 0.1):
        super().__init__()
        self.dim = dim
        self.min_active = min_active
        self.max_active = max_active
        
        # Complexity estimation network
        self.complexity_net = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.GELU(),
            nn.Linear(dim // 8, 1),
            nn.Sigmoid()
        )
        
        # Gating network
        self.gate_net = nn.Linear(dim, dim)
        
        # Running statistics for normalization
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
        self.momentum = 0.1
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Estimate complexity
        complexity = self.complexity_net(x.mean(dim=1, keepdim=True))
        
        # Determine sparsity level
        active_ratio = self.min_active + (self.max_active - self.min_active) * complexity
        k = (active_ratio * self.dim).int().clamp(min=1)
        
        # Compute importance scores
        importance = self.gate_net(x).abs()
        
        # Update running statistics
        if self.training:
            with torch.no_grad():
                mean = importance.mean(dim=(0, 1))
                var = importance.var(dim=(0, 1))
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        
        # Normalize importance scores
        importance = (importance - self.running_mean) / (self.running_var.sqrt() + 1e-6)
        
        # Select top-k neurons
        topk_vals, topk_idx = torch.topk(importance, k.item(), dim=-1)
        
        # Create sparse mask
        mask = torch.zeros_like(importance)
        mask.scatter_(-1, topk_idx, 1.0)
        
        # Apply mask with straight-through estimator
        sparse_x = x * mask + (x - x.detach()) * (1 - mask)
        
        stats = {
            'complexity': complexity.mean().item(),
            'active_ratio': active_ratio.mean().item(),
            'active_neurons': k.float().mean().item()
        }
        
        return sparse_x, stats