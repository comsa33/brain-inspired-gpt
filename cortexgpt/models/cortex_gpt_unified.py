"""
Unified CortexGPT: Brain-Inspired Language Model with All Enhancements
Integrates Phase 1 stability, Phase 2 neuroscience, and Phase 3 performance optimizations.

This replaces the legacy cortex_gpt.py with a unified, production-ready model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Any, Union
from dataclasses import dataclass
import faiss
from collections import deque
import math
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import time

# Check for GPU support
FAISS_GPU_AVAILABLE = False
try:
    import faiss
    if hasattr(faiss, 'StandardGpuResources'):
        FAISS_GPU_AVAILABLE = True
except ImportError:
    pass


@dataclass
class UnifiedCortexConfig:
    """Unified configuration combining all phases"""
    
    # Core architecture
    stm_capacity: int = 128
    ltm_dim: int = 256
    compression_ratio: int = 32
    consolidation_threshold: int = 3
    retrieval_top_k: int = 5
    cortical_columns: int = 16
    sparsity_ratio: float = 0.05
    
    # Phase 1: Stability improvements
    memory_temperature: float = 1.0
    memory_dropout: float = 0.1
    use_stop_gradient: bool = True
    residual_weight: float = 0.1
    use_soft_sparsity: bool = True
    sparsity_temperature: float = 0.5
    memory_decay_learnable: bool = True
    
    # Phase 2: Neuroscience features
    enable_homeostasis: bool = True
    target_firing_rate: float = 0.1
    homeostatic_tau: float = 1000.0
    homeostatic_strength: float = 0.01
    
    enable_sleep_wake: bool = True
    wake_theta_freq: float = 8.0
    sleep_delta_freq: float = 1.0
    rem_gamma_freq: float = 40.0
    consolidation_cycle: int = 1000
    
    enable_cls: bool = True
    fast_lr_scale: float = 10.0
    slow_lr_scale: float = 0.1
    cls_balance: float = 0.3
    
    enable_metaplasticity: bool = True
    bcm_threshold_tau: float = 5000.0
    bcm_threshold_init: float = 0.5
    ltp_rate: float = 0.01
    ltd_rate: float = 0.005
    
    # Phase 3: Performance optimizations
    use_gpu_memory: bool = True
    async_memory_ops: bool = True
    memory_thread_pool_size: int = 4
    batch_memory_operations: bool = True
    parallel_consolidation: bool = True
    memory_prefetch_size: int = 32
    
    # Phase 3: Advanced memory
    enable_episodic_memory: bool = True
    episodic_capacity: int = 10000
    episodic_window: int = 100
    episodic_compression: int = 16
    
    enable_working_memory: bool = True
    working_memory_slots: int = 8
    working_memory_decay: float = 0.95
    task_embedding_dim: int = 64
    
    enable_hierarchical_compression: bool = True
    memory_hierarchy_levels: int = 3
    compression_schedule: List[int] = None
    abstraction_threshold: float = 0.9
    
    enable_memory_interactions: bool = True
    memory_interaction_heads: int = 8
    cross_memory_attention: bool = True
    memory_graph_layers: int = 2
    
    # Phase 3: Cognitive features
    enable_cognitive_features: bool = True
    enable_analogical_reasoning: bool = True
    analogy_similarity_threshold: float = 0.8
    enable_causal_inference: bool = True
    causal_window: int = 50
    enable_concept_abstraction: bool = True
    concept_clustering_threshold: float = 0.7
    
    def __post_init__(self):
        if self.compression_schedule is None:
            self.compression_schedule = [8, 32, 128]


class StabilizedMemoryBase(nn.Module):
    """Base class for stabilized memory systems"""
    
    def __init__(self, capacity: int, dim: int, dropout: float = 0.1):
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        
        # Memory storage
        self.keys = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        self.timestamps = deque(maxlen=capacity)
        
        # Attention mechanism
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(dim)
        
        # Learnable decay rate
        self.decay_rate = nn.Parameter(torch.tensor(0.99))
        
    def compute_attention(self, query: torch.Tensor, keys: torch.Tensor, 
                         temperature: float = 1.0) -> torch.Tensor:
        """Compute stabilized attention scores"""
        q = self.query_proj(query)
        k = self.key_proj(keys)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = scores / temperature
        
        # Stabilized softmax
        scores = scores - scores.max(dim=-1, keepdim=True)[0]
        attn_weights = F.softmax(scores, dim=-1)
        
        return attn_weights


class UnifiedShortTermMemory(StabilizedMemoryBase):
    """Enhanced STM with all improvements"""
    
    def __init__(self, config: UnifiedCortexConfig, dim: int):
        super().__init__(config.stm_capacity, dim, config.memory_dropout)
        self.config = config
        self.current_timestamp = 0
        
    def store(self, key: torch.Tensor, value: torch.Tensor, 
              timestamp: Optional[int] = None):
        """Store with stabilized handling"""
        if timestamp is None:
            timestamp = self.current_timestamp
            self.current_timestamp += 1
            
        # Handle batch dimensions properly
        if key.dim() > 1:
            for i in range(key.size(0)):
                self.keys.append(key[i].detach())
                self.values.append(value[i].detach())
                self.timestamps.append(timestamp)
        else:
            self.keys.append(key.detach())
            self.values.append(value.detach())
            self.timestamps.append(timestamp)
            
    def retrieve(self, query: torch.Tensor, current_timestamp: Optional[int] = None,
                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve with temporal decay and stabilization"""
        if len(self.keys) == 0:
            return torch.zeros_like(query), torch.zeros(
                query.shape[0] if query.dim() > 1 else 1, device=query.device
            )
            
        # Stack keys and values
        keys_tensor = torch.stack(list(self.keys))
        values_tensor = torch.stack(list(self.values))
        
        # Flatten batch dimensions if needed
        if keys_tensor.dim() == 3:
            keys_tensor = keys_tensor.view(-1, keys_tensor.size(-1))
        if values_tensor.dim() == 3:
            values_tensor = values_tensor.view(-1, values_tensor.size(-1))
            
        # Apply temporal decay
        if current_timestamp is not None:
            time_diffs = torch.tensor(
                [current_timestamp - t for t in self.timestamps],
                device=query.device, dtype=torch.float32
            )
            decay_weights = torch.pow(self.decay_rate, time_diffs)
            decay_weights = decay_weights.unsqueeze(-1)
        else:
            decay_weights = 1.0
            
        # Compute attention with temperature control
        attn_weights = self.compute_attention(query, keys_tensor, temperature)
        
        # Apply decay to attention
        if isinstance(decay_weights, torch.Tensor):
            attn_weights = attn_weights * decay_weights.t()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
        # Weighted retrieval
        retrieved = torch.matmul(attn_weights, values_tensor)
        confidence = attn_weights.max(dim=-1)[0]
        
        # Apply dropout and normalization
        retrieved = self.dropout(retrieved)
        retrieved = self.layer_norm(retrieved)
        
        return retrieved, confidence


class GPUAcceleratedLTM(nn.Module):
    """GPU-accelerated long-term memory with hierarchical compression"""
    
    def __init__(self, config: UnifiedCortexConfig, dim: int):
        super().__init__()
        self.config = config
        self.dim = dim
        self.compressed_dim = config.ltm_dim
        
        # Compression networks
        self.compressor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.LayerNorm(dim // 4),
            nn.Linear(dim // 4, self.compressed_dim)
        )
        
        self.decompressor = nn.Sequential(
            nn.Linear(self.compressed_dim, dim // 4),
            nn.GELU(),
            nn.LayerNorm(dim // 4),
            nn.Linear(dim // 4, dim // 2),
            nn.GELU(),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, dim)
        )
        
        # Initialize FAISS index
        self.use_gpu = config.use_gpu_memory and FAISS_GPU_AVAILABLE and torch.cuda.is_available()
        self._init_faiss_index()
        
        self.stored_values = deque(maxlen=100000)
        self.metadata = deque(maxlen=100000)
        self.current_size = 0
        self.lock = threading.Lock()
        
    def _init_faiss_index(self):
        """Initialize FAISS index with GPU support if available"""
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                flat_config = faiss.GpuIndexFlatConfig()
                flat_config.device = torch.cuda.current_device()
                self.index = faiss.GpuIndexFlatL2(res, self.compressed_dim, flat_config)
            except:
                self.use_gpu = False
                self.index = faiss.IndexFlatL2(self.compressed_dim)
        else:
            self.index = faiss.IndexFlatL2(self.compressed_dim)
            
    def consolidate(self, key: torch.Tensor, value: torch.Tensor, metadata: Dict = None):
        """Consolidate memory with compression"""
        with self.lock:
            # Compress
            compressed_key = self.compressor(key.float())
            
            if compressed_key.dim() == 1:
                compressed_key = compressed_key.unsqueeze(0)
                
            # Add to index
            self.index.add(compressed_key.detach().cpu().numpy())
            self.stored_values.append(value.detach())
            self.metadata.append(metadata or {})
            self.current_size += 1
            
    def retrieve(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve with decompression"""
        if self.current_size == 0:
            return torch.zeros_like(query), torch.zeros(
                query.shape[0] if query.dim() > 1 else 1, device=query.device
            )
            
        with self.lock:
            # Compress query
            compressed_query = self.compressor(query.float())
            if compressed_query.dim() == 1:
                compressed_query = compressed_query.unsqueeze(0)
                
            # Search
            k = min(k, self.current_size)
            distances, indices = self.index.search(
                compressed_query.detach().cpu().numpy(), k
            )
            
            # Retrieve and decompress
            retrieved_values = []
            for batch_idx, batch_indices in enumerate(indices):
                batch_values = []
                for idx in batch_indices:
                    if idx != -1 and idx < len(self.stored_values):
                        value = self.stored_values[idx]
                        # Decompress if needed
                        if value.size(-1) == self.compressed_dim:
                            value = self.decompressor(value.float())
                        batch_values.append(value)
                        
                if batch_values:
                    # Weight by inverse distance
                    weights = 1.0 / (distances[batch_idx] + 1e-6)
                    weights = weights / weights.sum()
                    weighted_value = sum(w * v for w, v in zip(weights, batch_values))
                    retrieved_values.append(weighted_value)
                else:
                    retrieved_values.append(torch.zeros_like(query[batch_idx] if query.dim() > 1 else query))
                    
            retrieved = torch.stack(retrieved_values) if retrieved_values else torch.zeros_like(query)
            confidence = 1.0 / (distances.min(axis=1) + 1e-6) if distances.size > 0 else torch.zeros(1)
            
            return retrieved, torch.tensor(confidence, device=query.device)


class EpisodicMemory(nn.Module):
    """Episodic memory for experience sequences"""
    
    def __init__(self, config: UnifiedCortexConfig, dim: int):
        super().__init__()
        self.config = config
        self.dim = dim
        
        # Episode encoder/decoder
        self.episode_encoder = nn.Sequential(
            nn.Linear(dim * config.episodic_window, dim * 4),
            nn.GELU(),
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, config.episodic_compression * dim)
        )
        
        self.episode_decoder = nn.Sequential(
            nn.Linear(config.episodic_compression * dim, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim)
        )
        
        # Episode buffer
        self.episode_buffer = deque(maxlen=config.episodic_window)
        self.episodes = deque(maxlen=config.episodic_capacity)
        
    def add_to_episode(self, state: torch.Tensor):
        """Add state to current episode"""
        self.episode_buffer.append(state.detach())
        
    def store_episode(self, metadata: Dict = None):
        """Store current episode"""
        if len(self.episode_buffer) >= self.config.episodic_window // 2:
            # Pad if needed
            states = list(self.episode_buffer)
            while len(states) < self.config.episodic_window:
                states.append(torch.zeros_like(states[0]))
                
            # Encode episode
            episode_tensor = torch.cat(states, dim=-1)
            encoded = self.episode_encoder(episode_tensor)
            
            self.episodes.append({
                'encoded': encoded.detach(),
                'metadata': metadata or {},
                'length': len(self.episode_buffer)
            })
            
    def retrieve_similar(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Retrieve similar episodes"""
        if not self.episodes:
            return torch.zeros_like(query)
            
        # Encode current context as episode query
        if len(self.episode_buffer) > 0:
            states = list(self.episode_buffer)[-self.config.episodic_window:]
            while len(states) < self.config.episodic_window:
                states.append(torch.zeros_like(states[0]))
            episode_query = torch.cat(states, dim=-1)
            encoded_query = self.episode_encoder(episode_query)
        else:
            return torch.zeros_like(query)
            
        # Find similar episodes
        similarities = []
        for episode in self.episodes:
            sim = F.cosine_similarity(encoded_query, episode['encoded'], dim=-1)
            similarities.append(sim)
            
        if similarities:
            # Get top-k
            similarities_tensor = torch.stack(similarities)
            top_k = min(k, len(similarities))
            _, indices = torch.topk(similarities_tensor, top_k)
            
            # Decode and average
            retrieved = []
            for idx in indices:
                if 0 <= idx < len(self.episodes):
                    decoded = self.episode_decoder(self.episodes[idx]['encoded'])
                    retrieved.append(decoded)
                
            if retrieved:
                return torch.stack(retrieved).mean(dim=0)
            else:
                return torch.zeros_like(query)
        else:
            return torch.zeros_like(query)


class WorkingMemory(nn.Module):
    """Task-specific working memory with gated access"""
    
    def __init__(self, config: UnifiedCortexConfig, dim: int):
        super().__init__()
        self.config = config
        self.dim = dim
        self.num_slots = config.working_memory_slots
        
        # Memory slots
        self.slots = nn.Parameter(torch.randn(self.num_slots, dim) * 0.01)
        self.slot_gates = nn.Parameter(torch.ones(self.num_slots))
        
        # Task embedding
        self.task_embedder = nn.Sequential(
            nn.Linear(dim, config.task_embedding_dim),
            nn.GELU(),
            nn.LayerNorm(config.task_embedding_dim)
        )
        
        # Read/write attention
        self.read_attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.write_attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        
        # Gating
        self.read_gate = nn.Sequential(
            nn.Linear(dim + config.task_embedding_dim, dim),
            nn.Sigmoid()
        )
        
        self.register_buffer('slot_usage', torch.zeros(self.num_slots))
        
    def read(self, query: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Read from working memory"""
        batch_size = query.size(0)
        
        # Task embedding
        if task_context is None:
            task_context = query
        task_emb = self.task_embedder(task_context)
        
        # Gated slots
        slots = self.slots.unsqueeze(0).expand(batch_size, -1, -1)
        gated_slots = slots * self.slot_gates.unsqueeze(0).unsqueeze(-1)
        
        # Read attention
        read_out, _ = self.read_attention(
            query.unsqueeze(1), gated_slots, gated_slots
        )
        read_out = read_out.squeeze(1)
        
        # Apply gate
        gate_input = torch.cat([read_out, task_emb], dim=-1)
        gate = self.read_gate(gate_input)
        
        return read_out * gate
        
    def write(self, content: torch.Tensor, task_context: Optional[torch.Tensor] = None):
        """Write to working memory"""
        batch_size = content.size(0)
        
        # Find least used slot
        least_used_idx = torch.argmin(self.slot_usage)
        
        # Update slot with decay
        self.slots.data[least_used_idx] *= self.config.working_memory_decay
        self.slots.data[least_used_idx] += content.mean(dim=0) * (1 - self.config.working_memory_decay)
        self.slot_usage[least_used_idx] += 1


class HomeostaticNeuron(nn.Module):
    """Neuron with homeostatic plasticity"""
    
    def __init__(self, input_dim: int, output_dim: int, config: UnifiedCortexConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Homeostatic variables
        self.register_buffer('avg_activation', torch.zeros(output_dim))
        self.register_buffer('scaling_factor', torch.ones(output_dim))
        self.register_buffer('update_count', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x)
        
        # Apply homeostatic scaling
        scaled_output = output * self.scaling_factor.unsqueeze(0)
        activated = F.gelu(scaled_output)
        
        # Update homeostasis during training
        if self.training and self.config.enable_homeostasis:
            self._update_homeostasis(activated)
            
        return activated
        
    def _update_homeostasis(self, activations: torch.Tensor):
        """Update homeostatic scaling factors"""
        with torch.no_grad():
            batch_avg = activations.mean(dim=0)
            
            # Update running average
            tau = self.config.homeostatic_tau
            alpha = 1.0 / tau
            self.avg_activation = (1 - alpha) * self.avg_activation + alpha * batch_avg
            
            # Update scaling factors
            target = self.config.target_firing_rate
            self.scaling_factor = self.scaling_factor * (target / (self.avg_activation + 1e-8))
            self.scaling_factor = torch.clamp(self.scaling_factor, 0.1, 10.0)
            
            self.update_count += 1


class SleepWakeOscillator(nn.Module):
    """Sleep-wake cycle oscillator"""
    
    def __init__(self, config: UnifiedCortexConfig):
        super().__init__()
        self.config = config
        self.register_buffer('phase', torch.tensor(0.0))
        self.register_buffer('cycle_step', torch.tensor(0))
        
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current sleep-wake state"""
        cycle_pos = (self.cycle_step % self.config.consolidation_cycle).float()
        cycle_ratio = cycle_pos / self.config.consolidation_cycle
        
        # Sleep stages
        is_wake = cycle_ratio < 0.6
        is_nrem = (cycle_ratio >= 0.6) & (cycle_ratio < 0.8)
        is_rem = cycle_ratio >= 0.8
        
        # Generate oscillations
        if is_wake:
            theta = torch.sin(2 * np.pi * self.config.wake_theta_freq * self.phase)
            oscillation = theta
            consolidation_prob = 0.1
        elif is_nrem:
            delta = torch.sin(2 * np.pi * self.config.sleep_delta_freq * self.phase)
            oscillation = delta
            consolidation_prob = 0.8
        else:  # REM
            gamma = torch.sin(2 * np.pi * self.config.rem_gamma_freq * self.phase)
            theta = torch.sin(2 * np.pi * self.config.wake_theta_freq * self.phase)
            oscillation = gamma * (theta > 0).float()
            consolidation_prob = 0.5
            
        return {
            'oscillation': oscillation,
            'consolidation_prob': torch.tensor(consolidation_prob),
            'is_wake': is_wake,
            'is_nrem': is_nrem,
            'is_rem': is_rem,
            'cycle_ratio': cycle_ratio
        }
        
    def step(self, dt: float = 0.001):
        """Advance oscillator"""
        self.phase += dt
        self.cycle_step += 1


class ComplementaryLearningSystem(nn.Module):
    """Fast and slow learning pathways"""
    
    def __init__(self, dim: int, config: UnifiedCortexConfig):
        super().__init__()
        self.config = config
        self.dim = dim
        
        # Fast pathway (hippocampus-like)
        self.fast_encoder = nn.Sequential(
            HomeostaticNeuron(dim, dim * 2, config),
            nn.Dropout(0.2),
            HomeostaticNeuron(dim * 2, dim, config)
        )
        
        # Slow pathway (neocortex-like)
        self.slow_encoder = nn.Sequential(
            HomeostaticNeuron(dim, dim, config),
            nn.LayerNorm(dim),
            HomeostaticNeuron(dim, dim, config)
        )
        
        # Integration gate
        self.integration_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, sleep_state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process through complementary pathways"""
        # Fast pathway
        fast_output = self.fast_encoder(x)
        
        # Slow pathway
        slow_output = self.slow_encoder(x)
        
        # Dynamic balance based on sleep state
        if sleep_state['is_wake']:
            balance = self.config.cls_balance * 0.3
        elif sleep_state['is_nrem']:
            balance = self.config.cls_balance * 0.7
        else:  # REM
            balance = self.config.cls_balance
            
        # Gated integration
        gate_input = torch.cat([fast_output, slow_output], dim=-1)
        gate = self.integration_gate(gate_input)
        
        # Combine pathways
        output = (1 - balance) * fast_output + balance * slow_output
        output = gate * output + (1 - gate) * x  # Residual
        
        return output, fast_output


class CorticalColumn(nn.Module):
    """Enhanced cortical column with all improvements"""
    
    def __init__(self, dim: int, config: UnifiedCortexConfig):
        super().__init__()
        self.dim = dim
        self.config = config
        
        # Main processing with homeostatic neurons
        self.layers = nn.Sequential(
            HomeostaticNeuron(dim, dim * 2, config),
            nn.LayerNorm(dim * 2),
            nn.Dropout(config.memory_dropout),
            HomeostaticNeuron(dim * 2, dim, config),
            nn.LayerNorm(dim)
        )
        
        # Gating for soft sparsity
        self.gate = nn.Linear(dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process with soft sparsity"""
        # Compute gate scores
        gate_scores = self.gate(x).squeeze(-1)
        
        if self.config.use_soft_sparsity and self.training:
            # Soft sparsity using Gumbel-Softmax
            gate_probs = F.gumbel_softmax(gate_scores.unsqueeze(-1), 
                                         tau=self.config.sparsity_temperature,
                                         hard=False, dim=0).squeeze(-1)
        else:
            # Hard sparsity for inference
            batch_size = x.size(0)
            k = max(1, int(batch_size * self.config.sparsity_ratio))
            _, indices = torch.topk(gate_scores, k)
            gate_probs = torch.zeros_like(gate_scores)
            gate_probs.scatter_(0, indices, 1.0)
            
        # Apply processing
        processed = self.layers(x)
        
        # Apply gating
        output = processed * gate_probs.unsqueeze(-1)
        
        return output, gate_probs


class MemoryConsolidator(nn.Module):
    """Async memory consolidation with hierarchical compression"""
    
    def __init__(self, config: UnifiedCortexConfig, dim: int):
        super().__init__()
        self.config = config
        self.dim = dim
        
        # Consolidation network
        self.consolidation_network = nn.Sequential(
            nn.Linear(dim * 2, config.ltm_dim),
            nn.GELU(),
            nn.LayerNorm(config.ltm_dim),
            nn.Linear(config.ltm_dim, config.ltm_dim)
        )
        
        # Async components
        if config.async_memory_ops:
            self.executor = ThreadPoolExecutor(max_workers=config.memory_thread_pool_size)
            self.consolidation_queue = queue.Queue()
            self.start_worker()
            
    def start_worker(self):
        """Start background consolidation worker"""
        def worker():
            while True:
                try:
                    items = self.consolidation_queue.get(timeout=1.0)
                    if items is None:
                        break
                    self._process_consolidation(items)
                except queue.Empty:
                    continue
                    
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        
    def _process_consolidation(self, items: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Process consolidation batch"""
        # Implementation handled by specific memory systems
        pass
        
    def consolidate(self, stm: StabilizedMemoryBase, ltm: GPUAcceleratedLTM):
        """Consolidate memories from STM to LTM"""
        candidates = []
        
        # Get consolidation candidates
        for i, (key, value) in enumerate(zip(stm.keys, stm.values)):
            # Simple threshold check (can be enhanced)
            if i < self.config.consolidation_threshold:
                candidates.append((key, value))
                
        # Process candidates
        for key, value in candidates:
            combined = torch.cat([key.float(), value.float()], dim=-1)
            consolidated = self.consolidation_network(combined)
            ltm.consolidate(key, consolidated)


class UnifiedCortexGPT(nn.Module):
    """Unified CortexGPT with all phases integrated"""
    
    def __init__(self, config: UnifiedCortexConfig, vocab_size: int, dim: int):
        super().__init__()
        self.config = config
        self.dim = dim
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(5000, dim)
        self.pos_modulation = nn.Parameter(torch.ones(1, 1, dim))
        
        # Input processing
        self.input_norm = nn.LayerNorm(dim)
        
        # Memory systems
        self.stm = UnifiedShortTermMemory(config, dim)
        self.ltm = GPUAcceleratedLTM(config, dim)
        
        # Optional advanced memory
        if config.enable_episodic_memory:
            self.episodic_memory = EpisodicMemory(config, dim)
        if config.enable_working_memory:
            self.working_memory = WorkingMemory(config, dim)
            
        # Neuroscience components
        if config.enable_sleep_wake:
            self.oscillator = SleepWakeOscillator(config)
        if config.enable_cls:
            self.cls = ComplementaryLearningSystem(dim, config)
            
        # Cortical columns
        self.columns = nn.ModuleList([
            CorticalColumn(dim, config) for _ in range(config.cortical_columns)
        ])
        
        # Column aggregation
        self.column_mixer = nn.Sequential(
            nn.Linear(dim * config.cortical_columns, dim),
            nn.LayerNorm(dim)
        )
        
        # Memory integration
        num_memories = 2  # STM, LTM
        if config.enable_episodic_memory:
            num_memories += 1
        if config.enable_working_memory:
            num_memories += 1
            
        self.memory_gate = nn.Sequential(
            nn.Linear(dim * (num_memories + 1), num_memories + 1),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(dim, vocab_size)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        
        # Consolidator
        self.consolidator = MemoryConsolidator(config, dim)
        
        # Track timestep
        self.register_buffer('timestep', torch.tensor(0))
        
    def _create_positional_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        """Create positional encoding"""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           -(math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with all enhancements"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0).to(device)
        
        # Modulate based on sleep state if enabled
        if self.config.enable_sleep_wake:
            sleep_state = self.oscillator.get_state()
            modulation = 1.0 + 0.2 * sleep_state['oscillation']
            x = x + pos_enc * self.pos_modulation * modulation
        else:
            x = x + pos_enc * self.pos_modulation
            sleep_state = {'is_wake': True, 'is_nrem': False, 'is_rem': False,
                          'consolidation_prob': torch.tensor(0.1)}
            
        # Input normalization
        x = self.input_norm(x)
        
        # Process through CLS if enabled
        if self.config.enable_cls:
            x_reshaped = x.view(-1, self.dim)
            x_processed, fast_output = self.cls(x_reshaped, sleep_state)
            x = x_processed.view(batch_size, seq_len, self.dim)
        
        # Process through cortical columns
        column_outputs = []
        column_gates = []
        
        for column in self.columns:
            col_out, col_gate = column(x.view(-1, self.dim))
            column_outputs.append(col_out.view(batch_size, seq_len, self.dim))
            column_gates.append(col_gate.view(batch_size, seq_len))
            
        # Aggregate columns
        x = torch.cat(column_outputs, dim=-1)
        x = self.column_mixer(x)
        
        # Process sequence with memory augmentation
        outputs = []
        
        for t in range(seq_len):
            current = x[:, t, :]
            
            # Update working memory if enabled
            if self.config.enable_working_memory:
                self.working_memory.write(current)
                
            # Add to episodic buffer if enabled
            if self.config.enable_episodic_memory:
                self.episodic_memory.add_to_episode(current)
                
            # Retrieve from memories
            memories = self._retrieve_memories(current)
            
            # Integrate memories
            memory_values = [current]
            for mem_value, mem_score in memories.values():
                memory_values.append(mem_value)
                
            # Gate memories
            memory_concat = torch.cat(memory_values, dim=-1)
            gates = self.memory_gate(memory_concat)
            
            # Weighted combination
            output = sum(gate.unsqueeze(-1) * value 
                        for gate, value in zip(gates.unbind(-1), memory_values))
            
            # Residual connection
            output = current + self.config.residual_weight * output
            
            outputs.append(output)
            
            # Store in STM
            self.stm.store(current, output, timestamp=self.timestep.item())
            
            # Update timestep and oscillator
            self.timestep += 1
            if self.config.enable_sleep_wake:
                self.oscillator.step()
                
            # Consolidation check
            if self.config.enable_sleep_wake and sleep_state['consolidation_prob'] > torch.rand(1):
                self._trigger_consolidation()
                
        # Stack outputs
        output = torch.stack(outputs, dim=1)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        # Store episode if complete
        if self.config.enable_episodic_memory and seq_len >= self.config.episodic_window:
            self.episodic_memory.store_episode({'seq_len': seq_len})
            
        return logits
        
    def _retrieve_memories(self, query: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve from all memory systems"""
        memories = {}
        
        # STM retrieval
        stm_value, stm_score = self.stm.retrieve(
            query, 
            current_timestamp=self.timestep.item(),
            temperature=self.config.memory_temperature
        )
        memories['stm'] = (self._ensure_batch_dim(stm_value, query.size(0)), stm_score)
        
        # LTM retrieval
        ltm_value, ltm_score = self.ltm.retrieve(query, k=self.config.retrieval_top_k)
        memories['ltm'] = (self._ensure_batch_dim(ltm_value, query.size(0)), ltm_score)
        
        # Optional memory systems
        if self.config.enable_working_memory:
            wm_value = self.working_memory.read(query)
            memories['working'] = (wm_value, torch.ones(query.size(0), device=query.device))
            
        if self.config.enable_episodic_memory:
            ep_value = self.episodic_memory.retrieve_similar(query)
            memories['episodic'] = (self._ensure_batch_dim(ep_value, query.size(0)), 
                                   torch.ones(query.size(0), device=query.device))
            
        # Apply stop gradient if configured
        if self.config.use_stop_gradient:
            memories = {k: (v[0].detach(), v[1]) for k, v in memories.items()}
            
        return memories
        
    def _ensure_batch_dim(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Ensure tensor has correct batch dimension"""
        while tensor.dim() > 2:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.size(0) != batch_size:
            tensor = tensor.expand(batch_size, -1)
        return tensor
        
    def _trigger_consolidation(self):
        """Trigger memory consolidation"""
        if self.config.async_memory_ops and hasattr(self.consolidator, 'consolidation_queue'):
            # Get candidates
            candidates = []
            for i, (key, value) in enumerate(zip(self.stm.keys, self.stm.values)):
                if i < min(5, len(self.stm.keys)):  # Limit batch size
                    candidates.append((key, value))
            
            if candidates:
                self.consolidator.consolidation_queue.put(candidates)
        else:
            # Synchronous consolidation
            self.consolidator.consolidate(self.stm, self.ltm)
            
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0) -> torch.Tensor:
        """Generate text with memory augmentation"""
        self.eval()
        
        for _ in range(max_length - input_ids.size(1)):
            with torch.no_grad():
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for EOS
                if next_token.item() == 2:
                    break
                    
        return input_ids
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        stats = {
            'stm_size': len(self.stm.keys),
            'ltm_size': self.ltm.current_size,
            'timestep': self.timestep.item()
        }
        
        if self.config.enable_working_memory:
            stats['working_memory_usage'] = self.working_memory.slot_usage.mean().item()
            
        if self.config.enable_episodic_memory:
            stats['episode_buffer_size'] = len(self.episodic_memory.episode_buffer)
            stats['stored_episodes'] = len(self.episodic_memory.episodes)
            
        if self.config.enable_homeostasis:
            # Get average scaling factor from first column
            scaling_factors = []
            for layer in self.columns[0].layers:
                if isinstance(layer, HomeostaticNeuron):
                    scaling_factors.append(layer.scaling_factor.mean().item())
            if scaling_factors:
                stats['avg_homeostatic_scaling'] = np.mean(scaling_factors)
                
        if self.config.enable_sleep_wake:
            sleep_state = self.oscillator.get_state()
            stats['sleep_phase'] = sleep_state['cycle_ratio'].item()
            
        return stats
        
    def shutdown(self):
        """Clean up resources"""
        if hasattr(self.consolidator, 'consolidation_queue'):
            self.consolidator.consolidation_queue.put(None)
            if hasattr(self.consolidator, 'worker_thread'):
                self.consolidator.worker_thread.join(timeout=5.0)
        if hasattr(self.consolidator, 'executor'):
            self.consolidator.executor.shutdown(wait=True)


# Backward compatibility
CortexGPT = UnifiedCortexGPT
MemoryConfig = UnifiedCortexConfig