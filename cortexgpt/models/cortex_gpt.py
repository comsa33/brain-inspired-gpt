"""
CortexGPT: Brain-Inspired Language Model with Dual Memory System

This implements a biologically-inspired architecture with:
- Short-term memory (STM) for immediate processing
- Long-term memory (LTM) for consolidated knowledge  
- Real-time memory consolidation
- Sparse cortical activation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import faiss
from collections import deque
import math


@dataclass
class MemoryConfig:
    """Configuration for memory systems"""
    stm_capacity: int = 128  # Short-term memory capacity
    ltm_dim: int = 256  # Dimension of LTM embeddings
    compression_ratio: int = 32  # LTM compression factor
    consolidation_threshold: int = 3  # Repetitions before consolidation
    retrieval_top_k: int = 5  # Number of memories to retrieve
    cortical_columns: int = 16  # Number of parallel processing columns
    sparsity_ratio: float = 0.05  # Only 5% of neurons active


class ShortTermMemory(nn.Module):
    """
    Short-term memory buffer with attention-based retrieval.
    Mimics human working memory with limited capacity.
    """
    
    def __init__(self, capacity: int, dim: int):
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        
        # Memory slots (key-value pairs)
        self.keys = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        self.access_counts = deque(maxlen=capacity)
        
        # Attention mechanism for retrieval
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        
    def store(self, key: torch.Tensor, value: torch.Tensor):
        """Store item in STM, potentially evicting oldest"""
        self.keys.append(key.detach())
        self.values.append(value.detach())
        self.access_counts.append(1)
        
    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve from STM using attention mechanism"""
        if len(self.keys) == 0:
            return torch.zeros_like(query), torch.zeros(query.shape[0] if query.dim() > 1 else 1, device=query.device)
            
        # Store original dimension for later
        original_dim = query.dim()
        
        # Handle query dimensions
        if query.dim() == 1:
            query = query.unsqueeze(0)  # Add batch dimension
            
        # Project query
        q = self.query_proj(query)  # [batch, dim]
        
        # Stack all keys and compute attention
        keys_tensor = torch.stack(list(self.keys))  # [n_items, dim]
        # Handle stored keys that might have batch dimension
        if keys_tensor.dim() == 3:
            # Flatten batch dimension from stored keys
            keys_tensor = keys_tensor.view(-1, keys_tensor.size(-1))
        k = self.key_proj(keys_tensor)  # [n_items, dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.t()) / math.sqrt(self.dim)  # [batch, n_items]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Retrieve weighted combination of values
        values_tensor = torch.stack(list(self.values))  # [n_items, dim] or [n_items, batch, dim]
        
        # Handle stored values that might have batch dimension
        if values_tensor.dim() == 3:
            # Flatten batch dimension from stored values
            values_tensor = values_tensor.view(-1, values_tensor.size(-1))
        
        # Match dimensions for matrix multiplication
        if attn_weights.size(1) != values_tensor.size(0):
            # Adjust attention weights to match number of values
            n_values = values_tensor.size(0)
            if attn_weights.size(1) > n_values:
                attn_weights = attn_weights[:, :n_values]
            else:
                # Pad with zeros if needed
                padding = torch.zeros(attn_weights.size(0), n_values - attn_weights.size(1), 
                                    device=attn_weights.device)
                attn_weights = torch.cat([attn_weights, padding], dim=1)
        
        retrieved = torch.matmul(attn_weights, values_tensor)  # [batch, dim]
        
        # Update access counts for retrieved items
        max_indices = torch.argmax(attn_weights, dim=-1)
        for idx in max_indices:
            if idx < len(self.access_counts):
                self.access_counts[idx] += 1
                
        # Return with proper dimensions
        confidence = attn_weights.max(dim=-1)[0]
        if original_dim == 1:
            return retrieved.squeeze(0), confidence.squeeze(0)
        else:
            return retrieved, confidence
    
    def get_consolidation_candidates(self, threshold: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get items that should be consolidated to LTM"""
        candidates = []
        for i, count in enumerate(self.access_counts):
            if count >= threshold:
                candidates.append((self.keys[i], self.values[i]))
        return candidates


class LongTermMemory(nn.Module):
    """
    Long-term memory with compressed storage and fast retrieval.
    Uses learned compression and vector similarity search.
    """
    
    def __init__(self, input_dim: int, compressed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        
        # Compression/decompression networks
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, compressed_dim)
        )
        
        self.decompressor = nn.Sequential(
            nn.Linear(compressed_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )
        
        # Initialize FAISS index for fast retrieval
        self.index = faiss.IndexFlatL2(compressed_dim)
        self.stored_values = []
        
    def consolidate(self, key: torch.Tensor, value: torch.Tensor):
        """Compress and store in LTM"""
        # Ensure float32 for compression network
        key = key.float()
        value = value.float()
        
        # Compress the key for indexing
        compressed_key = self.compressor(key)
        
        # Add to FAISS index
        self.index.add(compressed_key.detach().cpu().numpy())
        self.stored_values.append(value.detach())
        
    def retrieve(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve k most similar memories"""
        if self.index.ntotal == 0:
            return torch.zeros_like(query), torch.zeros(query.shape[0] if query.dim() > 1 else 1, device=query.device)
            
        # Ensure float32 for compression network
        query_float = query.float()
        
        # Compress query
        compressed_query = self.compressor(query_float)
        
        # Search in FAISS
        distances, indices = self.index.search(compressed_query.detach().cpu().numpy(), k)
        
        # Retrieve and aggregate values
        retrieved_values = []
        for batch_idx, batch_indices in enumerate(indices):
            batch_values = []
            for idx in batch_indices:
                if idx != -1 and idx < len(self.stored_values):
                    batch_values.append(self.stored_values[idx])
            
            if batch_values:
                # Weight by inverse distance
                weights = 1.0 / (distances[batch_idx] + 1e-6)
                weights = weights / weights.sum()
                weighted_value = sum(w * v for w, v in zip(weights, batch_values))
                # Decompress back to original dimension
                decompressed = self.decompressor(weighted_value.float())
                # Convert back to original dtype
                decompressed = decompressed.to(query.dtype)
                retrieved_values.append(decompressed)
            else:
                # Ensure proper shape even when no values found
                zero_value = torch.zeros_like(query[batch_idx])
                if zero_value.dim() == 1:
                    zero_value = zero_value.unsqueeze(0)
                retrieved_values.append(zero_value)
                
        retrieved = torch.stack(retrieved_values)
        confidence = 1.0 / (distances.min(axis=1) + 1e-6)
        
        return retrieved, torch.tensor(confidence, device=query.device)


class CorticalColumn(nn.Module):
    """
    A single cortical column that processes information sparsely.
    Each column specializes in different aspects of the input.
    """
    
    def __init__(self, dim: int, sparsity_ratio: float = 0.05):
        super().__init__()
        self.dim = dim
        self.sparsity_ratio = sparsity_ratio
        
        # Specialized transformations
        self.transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # Gating mechanism for sparse activation
        self.gate = nn.Linear(dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input with sparse activation"""
        # Compute gate scores
        gate_scores = self.gate(x).squeeze(-1)  # [batch]
        
        # Apply sparsity (only top-k% activate)
        batch_size = x.size(0)
        k = max(1, int(batch_size * self.sparsity_ratio))
        top_k_values, top_k_indices = torch.topk(gate_scores, k)
        
        # Create sparse mask
        mask = torch.zeros_like(gate_scores)
        mask.scatter_(0, top_k_indices, 1.0)
        
        # Apply transformation only to active items
        output = torch.zeros_like(x)
        if mask.sum() > 0:
            active_x = x[mask.bool()]
            active_output = self.transform(active_x)
            # Ensure same dtype
            output[mask.bool()] = active_output.to(output.dtype)
            
        return output, mask


class MemoryConsolidator(nn.Module):
    """
    Handles the transfer of information from STM to LTM.
    Implements sleep-like consolidation cycles.
    """
    
    def __init__(self, config: MemoryConfig, dim: int):
        super().__init__()
        self.config = config
        self.dim = dim
        # Consolidation network takes two embeddings (key + value) of size dim each
        self.consolidation_network = nn.Sequential(
            nn.Linear(dim * 2, config.ltm_dim),
            nn.ReLU(),
            nn.Linear(config.ltm_dim, config.ltm_dim)
        )
        
    def consolidate(self, stm: ShortTermMemory, ltm: LongTermMemory):
        """Transfer repeated patterns from STM to LTM"""
        candidates = stm.get_consolidation_candidates(self.config.consolidation_threshold)
        
        for key, value in candidates:
            # Process through consolidation network - ensure float32
            combined = torch.cat([key.float(), value.float()], dim=-1)
            consolidated = self.consolidation_network(combined)
            
            # Store in LTM
            ltm.consolidate(key, consolidated)
            
            # Remove from STM access counts to prevent re-consolidation
            # (In full implementation, would remove the item)


class CortexGPT(nn.Module):
    """
    Main CortexGPT model combining all components.
    """
    
    def __init__(self, config: MemoryConfig, vocab_size: int, dim: int):
        super().__init__()
        self.config = config
        self.dim = dim
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_encoding = self._create_positional_encoding(5000, dim)
        
        # Memory systems
        self.stm = ShortTermMemory(config.stm_capacity, dim)
        self.ltm = LongTermMemory(dim, config.ltm_dim)
        self.consolidator = MemoryConsolidator(config, dim)
        
        # Cortical columns
        self.columns = nn.ModuleList([
            CorticalColumn(dim, config.sparsity_ratio)
            for _ in range(config.cortical_columns)
        ])
        
        # Column aggregation
        self.column_mixer = nn.Linear(dim * config.cortical_columns, dim)
        
        # Output projection
        self.output_proj = nn.Linear(dim, vocab_size)
        
        # Memory integration
        self.memory_gate = nn.Linear(dim * 3, 3)  # STM, LTM, current
        
    def _create_positional_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        """Create decaying positional encoding"""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           -(math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Apply decay to simulate temporal context fading
        decay = torch.exp(-0.1 * torch.arange(max_len).float()).unsqueeze(1)
        pe = pe * decay
        
        return pe
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory augmentation and sparse processing.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed tokens
        x = self.token_embedding(input_ids)  # [batch, seq_len, dim]
        
        # Add positional encoding
        # Ensure positional encoding matches embedding dimension
        if self.positional_encoding.size(1) != x.size(2):
            # Recreate positional encoding with correct dimension
            self.positional_encoding = self._create_positional_encoding(5000, x.size(2))
            
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0).to(device)  # [1, seq_len, dim]
        x = x + pos_enc
        
        # Process through cortical columns (sparse activation)
        column_outputs = []
        active_masks = []
        
        for column in self.columns:
            output, mask = column(x.view(-1, self.dim))
            column_outputs.append(output)
            active_masks.append(mask)
            
        # Aggregate column outputs
        column_stack = torch.stack(column_outputs, dim=-1)  # [batch*seq, dim, n_columns]
        column_stack = column_stack.view(batch_size, seq_len, -1)
        x = self.column_mixer(column_stack)
        
        # Memory augmentation
        outputs = []
        
        for t in range(seq_len):
            current = x[:, t, :]  # [batch, dim]
            
            # Retrieve from memories
            stm_value, stm_score = self.stm.retrieve(current)
            ltm_value, ltm_score = self.ltm.retrieve(current)
            
            # Ensure all tensors have same dimensions as current [batch, dim]
            while stm_value.dim() > 2:
                stm_value = stm_value.squeeze(0)
            while ltm_value.dim() > 2:
                ltm_value = ltm_value.squeeze(0)
            
            # Handle case where memory values might be 1D
            if stm_value.dim() == 1:
                stm_value = stm_value.unsqueeze(0)
            if ltm_value.dim() == 1:
                ltm_value = ltm_value.unsqueeze(0)
                
            # Ensure batch dimensions match
            if stm_value.size(0) != current.size(0):
                stm_value = stm_value.expand(current.size(0), -1)
            if ltm_value.size(0) != current.size(0):
                ltm_value = ltm_value.expand(current.size(0), -1)
                
            # Gate memory contributions
            memory_inputs = torch.cat([current, stm_value, ltm_value], dim=-1)
            gates = F.softmax(self.memory_gate(memory_inputs), dim=-1)
            
            # Combine current and memory
            output = (gates[:, 0:1] * current + 
                     gates[:, 1:2] * stm_value + 
                     gates[:, 2:3] * ltm_value)
            
            outputs.append(output)
            
            # Store in STM for future retrieval
            self.stm.store(current, output)
            
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, dim]
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        # Periodic consolidation (every N steps during training)
        if self.training and torch.rand(1).item() < 0.1:  # 10% chance
            self.consolidator.consolidate(self.stm, self.ltm)
            
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100) -> torch.Tensor:
        """Generate text with memory-augmented decoding"""
        self.eval()
        
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.forward(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == 2:  # EOS token
                    break
                    
        return input_ids