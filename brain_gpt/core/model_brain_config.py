"""
Brain-Inspired GPT Configuration for RTX 3090
Optimized for 24GB VRAM with maximum efficiency
"""

from dataclasses import dataclass
import torch

@dataclass
class BrainGPTConfig:
    # Model dimensions optimized for RTX 3090 tensor cores (multiples of 16)
    vocab_size: int = 70288  # Base 50257 + 20000 Korean tokens + padding to multiple of 16
    block_size: int = 2048   # Longer context with efficient attention
    
    # Architecture - larger model that fits in 24GB through extreme sparsity
    n_layer: int = 48        # Deep like human cortex (6 layers × 8 regions)
    n_head: int = 32         # More heads but sparse activation
    n_embd: int = 2048       # Wider model compensating for sparsity
    
    # Brain-inspired parameters
    sparsity_base: float = 0.95      # 5% active neurons like brain
    sparsity_schedule: str = "developmental"  # Gradual pruning
    n_cortical_columns: int = 32     # Cortical column organization
    column_size: int = 64             # Neurons per column
    
    @property
    def cortical_size(self) -> int:
        """Total size of cortical columns"""
        return self.n_cortical_columns * self.column_size
    
    # Dendritic attention parameters
    n_dendrites: int = 8              # Dendrites per neuron group
    dendrite_threshold: float = 0.3   # Spike threshold
    selective_attention_ratio: float = 0.2  # Only 20% tokens get full attention
    
    # Efficiency parameters
    use_flash_attention: bool = True
    use_sparse_kernels: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: bool = True      # bf16 for A100/3090
    
    # Korean language parameters
    korean_vocab_size: int = 20000
    use_language_adapters: bool = True
    adapter_size: int = 256
    
    # Memory optimization for RTX 3090
    micro_batch_size: int = 2         # Small batch to fit in memory
    gradient_accumulation: int = 16   # Effective batch size 32
    activation_checkpointing: bool = True
    
    # Energy efficiency (simulated metabolic constraints)
    energy_budget: float = 1.0
    adaptive_computation: bool = True
    early_exit_threshold: float = 0.9
    
    # Training
    dropout: float = 0.0              # No dropout needed with sparsity
    bias: bool = False                # No bias for efficiency
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    
    def __post_init__(self):
        # Validate configuration for RTX 3090
        self._validate_memory_usage()
        
    def _validate_memory_usage(self):
        """Estimate memory usage to ensure it fits in 24GB"""
        # Parameters (with sparsity considered)
        embedding_params = self.vocab_size * self.n_embd
        attention_params = self.n_layer * 4 * self.n_embd * self.n_embd * (1 - self.sparsity_base)
        mlp_params = self.n_layer * 8 * self.n_embd * self.n_embd * (1 - self.sparsity_base)
        
        total_params = (embedding_params + attention_params + mlp_params) / 1e9
        
        # Activations (with checkpointing)
        activation_memory = self.micro_batch_size * self.block_size * self.n_embd * 4 / 1e9
        
        # Optimizer states (Adam needs 2x parameters)
        optimizer_memory = total_params * 2
        
        total_memory = total_params + activation_memory + optimizer_memory
        
        print(f"Estimated memory usage: {total_memory:.2f}GB")
        print(f"Parameters: {total_params:.2f}B (with {self.sparsity_base*100:.0f}% sparsity)")
        
        assert total_memory < 20, f"Model too large for RTX 3090: {total_memory:.2f}GB > 20GB"
        
    def get_sparse_pattern(self, layer_idx):
        """Get sparsity pattern for specific layer (increases with depth)"""
        if layer_idx < self.n_layer // 3:
            return 0.90  # Early layers: 10% active
        elif layer_idx < 2 * self.n_layer // 3:
            return 0.95  # Middle layers: 5% active  
        else:
            return 0.98  # Deep layers: 2% active