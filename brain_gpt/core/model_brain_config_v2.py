"""
Configuration for Brain-Inspired GPT V2
Optimized for efficiency and performance
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class BrainGPTConfigV2:
    """Configuration for BrainGPTv2 model"""
    
    # Model architecture
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 32000  # Reduced for efficiency
    
    # SSM parameters
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    
    # Memory parameters
    memory_size: int = 5000
    memory_key_size: int = 768
    memory_value_size: int = 768
    memory_write_prob: float = 0.1
    hebbian_lr: float = 0.1
    
    # Attention parameters (for selective attention layers)
    attention_ratio: float = 0.1  # Only attend to 10% of tokens
    attention_layers: List[int] = None  # Which layers use attention (default: every 4th)
    
    # Adaptive computation
    use_act: bool = True
    act_max_steps: int = 12
    act_threshold: float = 0.99
    
    # Training
    gradient_checkpointing: bool = True
    dropout: float = 0.1
    
    # Optimization
    use_flash_attention: bool = True
    compile_model: bool = True  # PyTorch 2.0 compile
    
    def __post_init__(self):
        # Set default attention layers if not specified
        if self.attention_layers is None:
            # Every 4th layer uses attention
            self.attention_layers = [i for i in range(self.n_layer) if i % 4 == 0]
        
        # Ensure memory sizes match embedding size
        if self.memory_key_size != self.n_embd:
            self.memory_key_size = self.n_embd
        if self.memory_value_size != self.n_embd:
            self.memory_value_size = self.n_embd


# Preset configurations
def get_model_config(model_size: str = "small") -> BrainGPTConfigV2:
    """Get preset model configuration"""
    
    configs = {
        "tiny": BrainGPTConfigV2(
            n_layer=6,
            n_head=6,
            n_embd=384,
            block_size=512,
            vocab_size=16000,
            memory_size=1000,
        ),
        "small": BrainGPTConfigV2(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=1024,
            vocab_size=32000,
            memory_size=5000,
        ),
        "medium": BrainGPTConfigV2(
            n_layer=24,
            n_head=16,
            n_embd=1024,
            block_size=2048,
            vocab_size=32000,
            memory_size=10000,
        ),
        "large": BrainGPTConfigV2(
            n_layer=32,
            n_head=20,
            n_embd=1280,
            block_size=2048,
            vocab_size=50000,
            memory_size=20000,
        ),
    }
    
    return configs.get(model_size, configs["small"])