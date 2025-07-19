# Configuration for Brain-Inspired Efficient GPT
# Optimized for RTX 3090 with Korean language support

# Model architecture - Brain-inspired with extreme sparsity
model_type = 'brain_gpt'  # New model type
n_layer = 48  # Deep architecture (6 cortical layers × 8 regions)
n_head = 32  # More heads for dendritic branching
n_embd = 2048  # Wider to compensate for sparsity
block_size = 2048  # Longer context with efficient attention
vocab_size = 70288  # Base 50257 + 20K Korean + padding
bias = False  # No bias for efficiency
dropout = 0.0  # No dropout needed with sparsity

# Brain-inspired parameters
use_brain_architecture = True
sparsity_base = 0.95  # 95% sparse like brain
n_cortical_columns = 32
column_size = 64
use_dendritic_attention = True
n_dendrites = 8
selective_attention_ratio = 0.2  # Only 20% tokens get full attention
use_lateral_inhibition = True

# Efficiency features
use_flash_attention = True
use_2_4_structured_sparsity = True  # RTX 3090 optimization
gradient_checkpointing = True
compile_model = True  # torch.compile
mixed_precision = True  # bf16

# Korean language support
use_multilingual = True
korean_vocab_size = 20000
use_language_adapters = True
adapter_size = 256

# Training hyperparameters
batch_size = 2  # Small for RTX 3090 memory
gradient_accumulation_steps = 16  # Effective batch = 32
max_iters = 100000
learning_rate = 3e-4
warmup_iters = 2000
lr_decay_iters = 100000
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Curriculum learning (brain development stages)
use_curriculum_learning = True
curriculum_stages = [
    {"name": "dense", "iters": 5000, "sparsity": 0.0, "languages": ["en"]},
    {"name": "pruning", "iters": 10000, "sparsity": 0.5, "languages": ["en"]},
    {"name": "multilingual", "iters": 20000, "sparsity": 0.8, "languages": ["en", "ko"]},
    {"name": "sparse", "iters": 30000, "sparsity": 0.95, "languages": ["en", "ko", "mixed"]},
    {"name": "expert", "iters": float('inf'), "sparsity": 0.98, "languages": ["all"]},
]

# Energy efficiency (metabolic constraints)
use_energy_constraints = True
energy_budget = 1.0
adaptive_computation = True
early_exit_threshold = 0.9

# Data configuration
dataset = 'mixed'  # Mix of English and Korean
data_dir = './data'
korean_data_dir = './data/korean'
language_mix_ratio = {"en": 0.6, "ko": 0.3, "code": 0.05, "mixed": 0.05}

# Evaluation
eval_interval = 500
eval_iters = 100
log_interval = 10

# Checkpointing
always_save_checkpoint = True
checkpoint_interval = 5000
out_dir = 'out-brain-gpt'

# Logging
wandb_log = True
wandb_project = 'brain-gpt'
wandb_run_name = 'brain-gpt-korean-3090'

# System
device = 'cuda'
dtype = 'bfloat16'  # Better for RTX 3090
backend = 'nccl'

# Initialize for efficient GPT modules if needed
init_from = 'scratch'