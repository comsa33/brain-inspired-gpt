"""
Hybrid Embedding Layer combining BGE-M3 with CortexGPT's memory systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import numpy as np
from pathlib import Path


class HybridEmbeddingLayer(nn.Module):
    """
    Hybrid embedding layer that combines BGE-M3 embeddings with CortexGPT's memory-aware system.
    
    This layer provides:
    - Pre-trained multilingual embeddings from BGE-M3
    - Memory-context aware projections
    - Adaptive combination of embeddings
    - Efficient caching mechanisms
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        bge_model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        cache_embeddings: bool = True,
        special_tokens: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.bge_model_name = bge_model_name
        self.use_fp16 = use_fp16
        self.cache_embeddings = cache_embeddings
        
        # Try to load BGE-M3 model
        self.bge_model = None
        self.use_bge = False
        try:
            from FlagEmbedding import BGEM3FlagModel
            self.bge_model = BGEM3FlagModel(bge_model_name, use_fp16=use_fp16)
            self.bge_dim = 1024  # BGE-M3 output dimension
            self.use_bge = True
            print(f"Successfully loaded BGE-M3 model: {bge_model_name}")
        except Exception as e:
            print(f"Warning: Could not load BGE-M3 model: {e}")
            print("Falling back to basic embeddings")
        
        # Fallback basic embeddings
        self.basic_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(8192, dim)  # Support up to 8192 tokens
        
        if self.use_bge:
            # Adapter layers to match dimensions
            self.bge_adapter = nn.Sequential(
                nn.Linear(self.bge_dim, dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim)
            )
            
            # Memory-aware projection
            self.memory_projection = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim)
            )
            
            # Learnable combination gate
            self.combination_gate = nn.Parameter(torch.tensor(0.7))
            
            # Language-specific adapters for better multilingual support
            self.lang_adapters = nn.ModuleDict({
                'ko': nn.Linear(dim, dim),
                'en': nn.Linear(dim, dim),
                'mixed': nn.Linear(dim, dim)
            })
        
        # Special token embeddings (always learnable)
        self.special_tokens = special_tokens or {}
        if self.special_tokens:
            self.special_embeddings = nn.Embedding(len(self.special_tokens), dim)
        
        # Embedding cache
        if self.cache_embeddings:
            self.embedding_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
        
        # Initialize adapter weights for stability
        if self.use_bge:
            self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize adapter weights for stable training"""
        # Initialize BGE adapter with Xavier initialization
        for module in self.bge_adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize language adapters as near-identity
        for lang, adapter in self.lang_adapters.items():
            if isinstance(adapter, nn.Linear):
                nn.init.eye_(adapter.weight)
                if adapter.bias is not None:
                    nn.init.zeros_(adapter.bias)
                # Add small noise to break symmetry
                with torch.no_grad():
                    adapter.weight.add_(torch.randn_like(adapter.weight) * 0.01)
        
        # Clamp combination gate to reasonable range
        with torch.no_grad():
            self.combination_gate.data = torch.clamp(self.combination_gate.data, 0.3, 0.7)
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on character ranges."""
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3' or '\u3131' <= c <= '\u3163')
        total_chars = len(text)
        
        if total_chars == 0:
            return 'mixed'
        
        korean_ratio = korean_chars / total_chars
        if korean_ratio > 0.7:
            return 'ko'
        elif korean_ratio < 0.3:
            return 'en'
        else:
            return 'mixed'
    
    def get_bge_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings from BGE-M3 model."""
        if not self.use_bge or self.bge_model is None:
            return None
        
        # Check cache first
        if self.cache_embeddings:
            cache_key = str(hash(tuple(texts)))
            if cache_key in self.embedding_cache:
                self.cache_hits += 1
                return self.embedding_cache[cache_key]
            self.cache_misses += 1
        
        # Get embeddings from BGE-M3
        with torch.no_grad():
            # BGE-M3 returns dict with 'dense_vecs', 'sparse_vecs', etc.
            outputs = self.bge_model.encode(
                texts,
                batch_size=len(texts),
                max_length=512,  # Adjust based on your needs
                return_dense=True,
                return_sparse=False,  # We only use dense for now
                return_colbert_vecs=False
            )
            embeddings = outputs['dense_vecs']
            
            # Convert to tensor
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings)
            
            # Move to correct device and dtype
            device = next(self.parameters()).device
            dtype = torch.float16 if self.use_fp16 else torch.float32
            embeddings = embeddings.to(device=device, dtype=dtype)
        
        # Cache the result
        if self.cache_embeddings and len(self.embedding_cache) < 1000:  # Limit cache size
            self.embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        tokenizer=None,
        memory_context: Optional[torch.Tensor] = None,
        use_bge: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Forward pass combining BGE-M3 and basic embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            tokenizer: Tokenizer for converting IDs to text (required for BGE)
            memory_context: Optional memory context [batch_size, dim]
            use_bge: Override whether to use BGE embeddings
        
        Returns:
            embeddings: Combined embeddings [batch_size, seq_len, dim]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Decide whether to use BGE
        use_bge = use_bge if use_bge is not None else self.use_bge
        
        # Get position embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(position_ids)
        
        # Get token embeddings
        if use_bge and self.bge_model is not None and tokenizer is not None:
            # Convert token IDs to text for BGE
            texts = []
            for ids in input_ids:
                # Handle special tokens
                valid_ids = ids[ids >= 0]  # Remove padding
                text = tokenizer.decode(valid_ids.tolist())
                texts.append(text)
            
            # Get BGE embeddings
            bge_embeds = self.get_bge_embeddings(texts)
            
            if bge_embeds is not None:
                # BGE returns [batch_size, embed_dim], we need [batch_size, seq_len, dim]
                # Expand to sequence length
                bge_embeds = bge_embeds.unsqueeze(1).expand(batch_size, seq_len, -1)
                
                # Adapt BGE embeddings to model dimension
                adapted_embeds = self.bge_adapter(bge_embeds)
                
                # Apply language-specific adaptation
                lang_adapted_embeds = []
                for i, text in enumerate(texts):
                    lang = self.detect_language(text)
                    if lang in self.lang_adapters:
                        lang_adapted = self.lang_adapters[lang](adapted_embeds[i])
                    else:
                        lang_adapted = self.lang_adapters['mixed'](adapted_embeds[i])
                    lang_adapted_embeds.append(lang_adapted)
                adapted_embeds = torch.stack(lang_adapted_embeds)
                
                # Get basic embeddings as well
                basic_embeds = self.basic_embedding(input_ids)
                
                # Combine BGE and basic embeddings
                if memory_context is not None:
                    # Expand memory context to sequence length
                    memory_context_expanded = memory_context.unsqueeze(1).expand(batch_size, seq_len, -1)
                    
                    # Combine with memory context
                    combined_input = torch.cat([adapted_embeds, memory_context_expanded], dim=-1)
                    memory_enhanced = self.memory_projection(combined_input)
                    
                    # Final combination
                    embeddings = (self.combination_gate * memory_enhanced + 
                                 (1 - self.combination_gate) * basic_embeds)
                else:
                    # Combine without memory context
                    embeddings = (self.combination_gate * adapted_embeds + 
                                 (1 - self.combination_gate) * basic_embeds)
            else:
                # Fallback to basic embeddings if BGE fails
                embeddings = self.basic_embedding(input_ids)
        else:
            # Use basic embeddings only
            embeddings = self.basic_embedding(input_ids)
        
        # Add position embeddings
        embeddings = embeddings + pos_embeds
        
        return embeddings
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.embedding_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses + 1e-8)
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class EmbeddingAdapterTrainer:
    """
    Two-stage training strategy for hybrid embeddings.
    Stage 1: Freeze BGE, train adapters only
    Stage 2: Fine-tune everything with different learning rates
    """
    
    def __init__(self, model, base_lr: float = 3e-4):
        self.model = model
        self.base_lr = base_lr
        
        # Identify parameter groups
        self.adapter_params = []
        self.bge_params = []
        self.other_params = []
        
        for name, param in model.named_parameters():
            if 'bge_adapter' in name or 'memory_projection' in name or 'lang_adapters' in name:
                self.adapter_params.append(param)
            elif 'bge_model' in name:
                self.bge_params.append(param)
            else:
                self.other_params.append(param)
    
    def get_stage1_optimizer(self, model):
        """Stage 1: Train adapters only, freeze BGE."""
        # Freeze BGE parameters
        for name, param in model.named_parameters():
            if 'bge_model' in name:
                param.requires_grad = False
        
        # Return optimizer for adapter parameters
        param_groups = [
            {'params': self.adapter_params, 'lr': self.base_lr},
            {'params': self.other_params, 'lr': self.base_lr}
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    def get_stage2_optimizer(self, model):
        """Stage 2: Fine-tune everything with different learning rates."""
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        # Different learning rates for different components
        param_groups = [
            {'params': self.bge_params, 'lr': self.base_lr * 0.1},  # Lower LR for pre-trained
            {'params': self.adapter_params, 'lr': self.base_lr},
            {'params': self.other_params, 'lr': self.base_lr}
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=0.01)