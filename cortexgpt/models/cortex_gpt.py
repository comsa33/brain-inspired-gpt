"""
CortexGPT: Brain-Inspired Language Model
This file now imports from the unified implementation that includes all enhancements.
"""

# Import everything from the unified implementation
from .cortex_gpt_unified import *

# Maintain backward compatibility
__all__ = [
    'UnifiedCortexConfig',
    'UnifiedCortexGPT',
    'CortexGPT',
    'MemoryConfig',
    'UnifiedShortTermMemory',
    'GPUAcceleratedLTM',
    'EpisodicMemory',
    'WorkingMemory',
    'HomeostaticNeuron',
    'SleepWakeOscillator',
    'ComplementaryLearningSystem',
    'CorticalColumn',
    'MemoryConsolidator'
]