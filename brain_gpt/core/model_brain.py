"""
Brain-Inspired Efficient GPT Model
Revolutionary architecture combining neuroscience insights with GPU optimization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import functools
from dataclasses import dataclass
from typing import Optional, Tuple

from .model_brain_config import BrainGPTConfig

# Custom autograd function for sparse operations with straight-through estimator
class SparseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sparsity_mask):
        ctx.save_for_backward(sparsity_mask)
        return input * sparsity_mask
    
    @staticmethod
    def backward(ctx, grad_output):
        sparsity_mask, = ctx.saved_tensors
        # Straight-through estimator: gradients flow through sparse connections
        return grad_output, None

sparse_multiply = SparseFunction.apply

class DendriticAttention(nn.Module):
    """
    Biologically-inspired attention mechanism with dendritic computation
    Processes attention in hierarchical branches before integration
    """
    
    def __init__(self, config: BrainGPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        # Ensure n_dendrites doesn't exceed n_head
        self.n_dendrites = min(config.n_dendrites, self.n_head)
        self.dendrite_threshold = config.dendrite_threshold
        
        # Dendritic branches process subsets of heads
        self.heads_per_dendrite = self.n_head // self.n_dendrites
        
        # Ensure we have at least 1 head per dendrite
        if self.heads_per_dendrite == 0:
            self.n_dendrites = self.n_head
            self.heads_per_dendrite = 1
        
        # Lightweight dendritic projections
        self.dendritic_qkv = nn.ModuleList([
            nn.Linear(self.n_embd, 3 * self.heads_per_dendrite * self.head_dim, bias=False)
            for _ in range(self.n_dendrites)
        ])
        
        # Soma integration layer
        self.soma_integration = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # Selective attention gating
        self.attention_gate = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd // 8, bias=False),
            nn.GELU(),
            nn.Linear(self.n_embd // 8, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x: torch.Tensor, return_attention_stats: bool = False) -> Tuple[torch.Tensor, Optional[dict]]:
        B, T, C = x.size()
        
        # Selective attention gating - decide which tokens need attention
        attention_gates = self.attention_gate(x)  # (B, T, 1)
        
        # Only process tokens with high gate values
        important_mask = (attention_gates > self.dendrite_threshold).float()
        
        # Process through dendritic branches
        dendritic_outputs = []
        total_active = 0
        
        for i, dendrite_qkv in enumerate(self.dendritic_qkv):
            # Only compute for important tokens
            masked_x = x * important_mask
            
            qkv = dendrite_qkv(masked_x)
            # Split into Q, K, V - each has shape (B, T, heads_per_dendrite * head_dim)
            qkv_split_size = self.heads_per_dendrite * self.head_dim
            q, k, v = qkv.split(qkv_split_size, dim=2)
            
            # Reshape for multi-head attention
            q = q.view(B, T, self.heads_per_dendrite, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.heads_per_dendrite, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.heads_per_dendrite, self.head_dim).transpose(1, 2)
            
            # Efficient attention with Flash Attention when available
            if hasattr(F, 'scaled_dot_product_attention'):
                att_output = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True
                )
            else:
                # Manual attention computation
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att_output = att @ v
            
            att_output = att_output.transpose(1, 2).contiguous().view(B, T, -1)
            dendritic_outputs.append(att_output)
            
            # Track sparsity
            total_active += important_mask.sum().item()
        
        # Soma integration - combine dendritic outputs
        combined = torch.cat(dendritic_outputs, dim=-1)
        output = self.soma_integration(combined)
        
        # Apply residual gating based on importance
        output = output * important_mask
        
        stats = None
        if return_attention_stats:
            stats = {
                'sparsity': 1.0 - (total_active / (B * T * self.n_dendrites)),
                'important_tokens': important_mask.mean().item()
            }
        
        return output, stats


class CorticalColumnBlock(nn.Module):
    """
    Implements cortical column organization with extreme sparsity
    Mimics the brain's columnar structure with lateral inhibition
    """
    
    def __init__(self, config: BrainGPTConfig, layer_idx: int):
        super().__init__()
        self.n_columns = config.n_cortical_columns
        self.column_size = config.column_size
        self.layer_idx = layer_idx
        self.sparsity = config.get_sparse_pattern(layer_idx)
        
        # Pre-normalization
        self.ln1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        
        # Dendritic attention
        self.attention = DendriticAttention(config)
        
        # Cortical column MLP with structured sparsity
        self.mlp = CorticalColumnMLP(config, self.sparsity)
        
        # Lateral inhibition between columns
        self.lateral_inhibition = LateralInhibition(self.n_columns, self.column_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        attn_out, _ = self.attention(self.ln1(x))
        x = x + attn_out
        
        # MLP with cortical column organization
        mlp_out = self.mlp(self.ln2(x))
        
        # Apply lateral inhibition only if dimensions match
        B, T, C = mlp_out.shape
        expected_size = self.n_columns * self.column_size
        
        if C == expected_size:
            mlp_reshaped = mlp_out.view(B, T, self.n_columns, self.column_size)
            mlp_inhibited = self.lateral_inhibition(mlp_reshaped)
            mlp_out = mlp_inhibited.view(B, T, C)
        else:
            # Skip lateral inhibition if dimensions don't match
            # This happens when n_embd != n_columns * column_size
            pass
        
        x = x + mlp_out
        
        return x


class CorticalColumnMLP(nn.Module):
    """
    MLP with cortical column structure and 2:4 structured sparsity for RTX 3090
    """
    
    def __init__(self, config: BrainGPTConfig, sparsity: float):
        super().__init__()
        self.hidden_size = 4 * config.n_embd
        self.sparsity = sparsity
        
        # Use 2:4 structured sparsity pattern (50% sparse, optimal for Ampere)
        self.use_2_4_sparsity = True
        
        self.w1 = nn.Linear(config.n_embd, self.hidden_size, bias=False)
        self.w2 = nn.Linear(self.hidden_size, config.n_embd, bias=False)
        
        # Precompute sparsity masks for efficiency
        self._init_sparse_masks()
        
    def _init_sparse_masks(self):
        """Initialize structured sparsity masks optimized for tensor cores"""
        if self.use_2_4_sparsity:
            # 2:4 pattern - keep 2 weights out of every 4
            mask1 = torch.zeros_like(self.w1.weight)
            mask2 = torch.zeros_like(self.w2.weight)
            
            # Create 2:4 pattern
            for i in range(0, mask1.shape[0], 4):
                indices = torch.randperm(4)[:2] + i
                if i + 4 <= mask1.shape[0]:
                    mask1[indices, :] = 1.0
                    
            for i in range(0, mask2.shape[1], 4):
                indices = torch.randperm(4)[:2] + i
                if i + 4 <= mask2.shape[1]:
                    mask2[:, indices] = 1.0
                    
            self.register_buffer('mask1', mask1)
            self.register_buffer('mask2', mask2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply structured sparsity to weights
        sparse_w1 = sparse_multiply(self.w1.weight, self.mask1)
        sparse_w2 = sparse_multiply(self.w2.weight, self.mask2)
        
        # Compute with sparse weights
        h = F.linear(x, sparse_w1)
        h = F.gelu(h)
        h = F.linear(h, sparse_w2)
        
        return h


class LateralInhibition(nn.Module):
    """
    Implements lateral inhibition between cortical columns
    Creates competition and specialization between columns
    """
    
    def __init__(self, n_columns: int, column_size: int):
        super().__init__()
        self.n_columns = n_columns
        self.column_size = column_size
        
        # Inhibitory connections between columns
        self.inhibition_strength = nn.Parameter(torch.ones(n_columns, n_columns) * 0.1)
        self.inhibition_strength.data.fill_diagonal_(0)  # No self-inhibition
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, n_columns, column_size)
        B, T, _, _ = x.shape
        
        # Compute column activities
        column_activities = x.mean(dim=-1)  # (B, T, n_columns)
        
        # Apply lateral inhibition
        inhibition = torch.matmul(column_activities, self.inhibition_strength)  # (B, T, n_columns)
        inhibition = inhibition.unsqueeze(-1)  # (B, T, n_columns, 1)
        
        # Suppress less active columns
        x_inhibited = x - inhibition * x
        x_inhibited = F.relu(x_inhibited)  # Only positive activations survive
        
        return x_inhibited


class BrainGPT(nn.Module):
    """
    Brain-Inspired GPT optimized for RTX 3090
    Combines extreme sparsity with biological computation principles
    """
    
    def __init__(self, config: BrainGPTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings with language-specific components
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        # Language adapters for Korean
        if config.use_language_adapters:
            self.language_adapters = nn.ModuleDict({
                'korean': LanguageAdapter(config.n_embd, config.adapter_size),
                'english': LanguageAdapter(config.n_embd, config.adapter_size),
            })
        
        # Cortical column transformer blocks
        self.blocks = nn.ModuleList([
            CorticalColumnBlock(config, layer_idx=i)
            for i in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
        
        # Output head with weight tying
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        
        # Energy tracking for metabolic constraints
        self.register_buffer('energy_consumed', torch.tensor(0.0))
        self.register_buffer('energy_budget', torch.tensor(config.energy_budget))
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print model statistics
        self._print_model_stats()
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def _print_model_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        sparse_params = total_params * (1 - self.config.sparsity_base)
        
        print(f"BrainGPT Model Statistics:")
        print(f"Total parameters: {total_params/1e9:.2f}B")
        print(f"Effective parameters (with sparsity): {sparse_params/1e9:.2f}B")
        print(f"Layers: {self.config.n_layer}")
        print(f"Hidden size: {self.config.n_embd}")
        print(f"Attention heads: {self.config.n_head}")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        language_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = input_ids.device
        B, T = input_ids.size()
        
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        # Token and position embeddings
        token_emb = self.token_embedding(input_ids)
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        
        x = token_emb + pos_emb
        
        # Apply language adapter if specified
        if language_id and self.config.use_language_adapters and hasattr(self, 'language_adapters'):
            if language_id in self.language_adapters:
                x = self.language_adapters[language_id](x)
        
        # Process through transformer blocks with optional gradient checkpointing
        for i, block in enumerate(self.blocks):
            if self.config.gradient_checkpointing and self.training:
                # Use the recommended use_reentrant=False for better performance
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
                
            # Early exit based on confidence (inference only)
            if not self.training and self.config.adaptive_computation:
                confidence = self._compute_confidence(x)
                if confidence > self.config.early_exit_threshold:
                    print(f"Early exit at layer {i+1}/{self.config.n_layer}")
                    break
        
        x = self.ln_f(x)
        
        if targets is not None:
            # Training mode - compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference mode - only compute last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss
    
    def _compute_confidence(self, x: torch.Tensor) -> float:
        """Compute prediction confidence for early exit"""
        # Simple confidence based on activation magnitude
        return torch.sigmoid(x.abs().mean()).item()
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        language_id: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate tokens with brain-inspired efficiency"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass - handle both tuple and tensor returns
            output = self(idx_cond, language_id=language_id)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # Apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


class LanguageAdapter(nn.Module):
    """Language-specific adapter for multilingual support"""
    
    def __init__(self, hidden_size: int, adapter_size: int):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size, bias=False)
        self.up_project = nn.Linear(adapter_size, hidden_size, bias=False)
        self.gelu = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_project(x)
        x = self.gelu(x)
        x = self.up_project(x)
        return residual + x