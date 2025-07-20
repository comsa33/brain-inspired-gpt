"""
Real-time Learning System for CortexGPT

This implements continuous learning from external queries with:
- Stream processing of incoming queries
- Self-supervised learning from responses
- Adaptive learning rate based on confidence
- Multi-language support (Korean/English)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import deque
import threading
import queue
import time
from dataclasses import dataclass
import json
import os

from cortexgpt.models.realtime_cortex import RealTimeCortexGPT, AdvancedMemoryConfig
from cortexgpt.tokenization.multilingual_tokenizer import MultilingualTokenizer


@dataclass
class LearningExample:
    """A single learning example from real-time interaction"""
    query: str
    response: str
    context: Optional[str]
    quality_score: float
    timestamp: float
    language: str  # 'ko', 'en', 'mixed'
    metadata: Dict[str, Any]


class RealTimeLearner:
    """
    Manages real-time learning for CortexGPT
    """
    
    def __init__(
        self,
        model: RealTimeCortexGPT,
        tokenizer: MultilingualTokenizer,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        # Configuration
        self.config = config or {
            'learning_batch_size': 1,
            'learning_interval': 10,  # seconds
            'min_quality_threshold': 0.5,
            'max_learning_rate': 0.01,
            'min_learning_rate': 0.0001,
            'adaptive_lr': True,
            'self_play_enabled': True,
            'self_play_interval': 60,  # seconds
            'memory_consolidation_interval': 300,  # 5 minutes
            'save_checkpoint_interval': 3600,  # 1 hour
        }
        
        # Learning components
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['max_learning_rate'],
            weight_decay=0.01
        )
        
        # Learning queue and history
        self.learning_queue = queue.Queue(maxsize=10000)
        self.learning_history = deque(maxlen=1000)
        self.response_cache = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'total_learned': 0,
            'avg_quality': 0.0,
            'languages': {'ko': 0, 'en': 0, 'mixed': 0},
            'learning_rate_history': deque(maxlen=100),
            'memory_stats': {
                'stm_size': 0,
                'ltm_size': 0,
                'archive_size': 0
            }
        }
        
        # Threading for continuous learning
        self.is_running = False
        self.learning_thread = None
        self.consolidation_thread = None
        self.self_play_thread = None
        
    def detect_language(self, text: str) -> str:
        """Detect if text is Korean, English, or mixed"""
        korean_chars = sum(1 for char in text if '가' <= char <= '힣')
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 'en'
        
        korean_ratio = korean_chars / total_chars
        
        if korean_ratio > 0.8:
            return 'ko'
        elif korean_ratio < 0.2:
            return 'en'
        else:
            return 'mixed'
    
    def process_query(
        self,
        query: str,
        context: Optional[str] = None,
        learn: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Process a real-time query and optionally learn from it"""
        
        # Detect language
        language = self.detect_language(query)
        self.stats['languages'][language] += 1
        
        # Tokenize query
        if context:
            full_input = f"{context} {query}"
        else:
            full_input = query
        
        input_ids = self.tokenizer.encode(full_input)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        
        # Move to device
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Get response from model
        with torch.no_grad():
            output_logits, metadata = self.model.process_real_time_query(
                input_tensor,
                learn=learn
            )
        
        # Generate response tokens
        response_ids = self.generate_response(output_logits)
        response_text = self.tokenizer.decode(response_ids)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(
            query, response_text, metadata
        )
        
        # Create learning example
        if learn and quality_score > self.config['min_quality_threshold']:
            example = LearningExample(
                query=query,
                response=response_text,
                context=context,
                quality_score=quality_score,
                timestamp=time.time(),
                language=language,
                metadata=metadata
            )
            
            # Add to learning queue
            try:
                self.learning_queue.put_nowait(example)
            except queue.Full:
                # Remove oldest example if queue is full
                try:
                    self.learning_queue.get_nowait()
                    self.learning_queue.put_nowait(example)
                except:
                    pass
        
        # Update statistics
        self.stats['total_queries'] += 1
        self.stats['avg_quality'] = (
            self.stats['avg_quality'] * 0.99 + quality_score * 0.01
        )
        
        # Prepare response metadata
        response_metadata = {
            'quality_score': quality_score,
            'language': language,
            'memory_usage': metadata.get('memory_gates', None),
            'confidence': metadata.get('memory_confidence', {}),
            'learned': learn and quality_score > self.config['min_quality_threshold']
        }
        
        return response_text, response_metadata
    
    def generate_response(
        self,
        logits: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8
    ) -> List[int]:
        """Generate response tokens from logits"""
        response_ids = []
        
        # Get vocabulary size
        vocab_size = logits.size(-1)
        
        # Generate tokens
        for i in range(max_length):
            # Get logits for current position
            if logits.dim() == 3:  # [batch, seq, vocab]
                if i < logits.size(1):
                    current_logits = logits[0, i, :]
                else:
                    # Use last position logits
                    current_logits = logits[0, -1, :]
            else:  # [batch, vocab]
                current_logits = logits[0]
            
            # Apply temperature
            current_logits = current_logits / temperature
            
            # Avoid generating special tokens too often
            unk_token = self.tokenizer.special_tokens.get('<unk>', 1)
            pad_token = self.tokenizer.special_tokens.get('<pad>', 0)
            current_logits[unk_token] -= 5.0  # Penalize UNK token
            current_logits[pad_token] -= 10.0  # Strongly penalize PAD token
            
            probs = F.softmax(current_logits, dim=-1)
            
            # Sample token with top-k filtering
            top_k = min(50, vocab_size)
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # Renormalize
            top_probs = top_probs / top_probs.sum()
            
            # Sample from top-k
            sample_idx = torch.multinomial(top_probs, 1).item()
            next_token = top_indices[sample_idx].item()
            
            response_ids.append(next_token)
            
            # Stop at EOS token
            if next_token == self.tokenizer.special_tokens.get('<eos>', 2):
                break
        
        return response_ids
    
    def calculate_quality_score(
        self,
        query: str,
        response: str,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate quality score for learning"""
        score = 0.0
        
        # Response length appropriateness
        query_len = len(query.split())
        response_len = len(response.split())
        length_ratio = response_len / max(query_len, 1)
        
        if 0.5 <= length_ratio <= 3.0:
            score += 0.3
        elif 0.2 <= length_ratio <= 5.0:
            score += 0.1
        
        # Memory confidence
        if 'memory_confidence' in metadata:
            avg_confidence = np.mean(list(metadata['memory_confidence'].values()))
            score += 0.3 * avg_confidence
        
        # Language consistency
        query_lang = self.detect_language(query)
        response_lang = self.detect_language(response)
        if query_lang == response_lang or query_lang == 'mixed':
            score += 0.2
        
        # Response coherence (from self-evaluation)
        if metadata.get('quality_scores'):
            scores = metadata['quality_scores']
            score += 0.2 * np.mean([
                scores.get('coherence', 0),
                scores.get('relevance', 0),
                scores.get('confidence', 0)
            ])
        
        return min(score, 1.0)
    
    def learn_from_example(self, example: LearningExample):
        """Learn from a single example"""
        # Prepare inputs
        input_ids = self.tokenizer.encode(example.query)
        target_ids = self.tokenizer.encode(example.response)
        
        # Ensure we have valid tokens
        if not input_ids:
            input_ids = [self.tokenizer.special_tokens.get('<unk>', 1)]
        if not target_ids:
            target_ids = [self.tokenizer.special_tokens.get('<unk>', 1)]
        
        # Truncate to reasonable length
        max_len = 512
        input_ids = input_ids[:max_len]
        target_ids = target_ids[:max_len]
        
        # Make sure they're the same length for loss calculation
        min_len = min(len(input_ids), len(target_ids))
        input_ids = input_ids[:min_len]
        target_ids = target_ids[:min_len]
        
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        target_tensor = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0)
        
        # Move to device
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        
        # Adaptive learning rate based on quality
        if self.config['adaptive_lr']:
            lr_scale = example.quality_score
            current_lr = (
                self.config['min_learning_rate'] +
                (self.config['max_learning_rate'] - self.config['min_learning_rate']) * lr_scale
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            self.stats['learning_rate_history'].append(current_lr)
        
        # Forward pass
        self.model.train()
        output = self.model(input_tensor, real_time=False)
        
        # Calculate loss with quality weighting
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            target_tensor.view(-1),
            reduction='none'
        )
        weighted_loss = (loss * example.quality_score).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update
        self.optimizer.step()
        
        # Update statistics
        self.stats['total_learned'] += 1
        self.learning_history.append({
            'timestamp': example.timestamp,
            'loss': weighted_loss.item(),
            'quality': example.quality_score,
            'language': example.language
        })
    
    def self_play_learning(self):
        """Generate self-play examples for learning"""
        # Sample from previous responses for context
        if len(self.response_cache) < 5:
            return
        
        # Create synthetic query by combining previous elements
        context_idx = np.random.randint(0, len(self.response_cache))
        context = self.response_cache[context_idx]
        
        # Generate a follow-up question (simplified version)
        prompts = {
            'ko': [
                "이것에 대해 더 설명해주세요:",
                "다음은 무엇인가요?",
                "예시를 들어주세요:",
                "왜 그런가요?"
            ],
            'en': [
                "Can you explain more about:",
                "What comes next?",
                "Can you give an example of:",
                "Why is that?"
            ]
        }
        
        # Detect language of context
        lang = self.detect_language(context)
        if lang == 'mixed':
            lang = 'en'
        
        prompt = np.random.choice(prompts[lang])
        synthetic_query = f"{prompt} {context[:50]}..."
        
        # Process as regular query
        response, metadata = self.process_query(
            synthetic_query,
            context=context,
            learn=True
        )
        
        # Cache response for future use
        if metadata['quality_score'] > 0.7:
            self.response_cache.append(response)
    
    def continuous_learning_loop(self):
        """Main learning loop that runs in background"""
        while self.is_running:
            try:
                # Get examples from queue
                examples = []
                deadline = time.time() + self.config['learning_interval']
                
                while time.time() < deadline and len(examples) < self.config['learning_batch_size']:
                    try:
                        example = self.learning_queue.get(timeout=1)
                        examples.append(example)
                    except queue.Empty:
                        continue
                
                # Learn from collected examples
                if examples:
                    for example in examples:
                        self.learn_from_example(example)
                
                # Update memory statistics
                self.stats['memory_stats'] = {
                    'stm_size': len(self.model.stm.memories),
                    'ltm_size': len(self.model.ltm.memories),
                    'archive_size': self.model.archive.index.ntotal
                }
                
            except Exception as e:
                print(f"Learning loop error: {e}")
                time.sleep(1)
    
    def memory_consolidation_loop(self):
        """Periodic memory consolidation"""
        while self.is_running:
            try:
                time.sleep(self.config['memory_consolidation_interval'])
                self.model.consolidate_memories()
                print(f"Memory consolidation completed. Stats: {self.stats['memory_stats']}")
            except Exception as e:
                print(f"Consolidation error: {e}")
    
    def self_play_loop(self):
        """Self-play learning loop"""
        while self.is_running:
            try:
                time.sleep(self.config['self_play_interval'])
                if self.config['self_play_enabled']:
                    self.self_play_learning()
            except Exception as e:
                print(f"Self-play error: {e}")
    
    def start(self):
        """Start all learning threads"""
        self.is_running = True
        
        # Start continuous learning
        self.learning_thread = threading.Thread(
            target=self.continuous_learning_loop,
            daemon=True
        )
        self.learning_thread.start()
        
        # Start memory consolidation
        self.consolidation_thread = threading.Thread(
            target=self.memory_consolidation_loop,
            daemon=True
        )
        self.consolidation_thread.start()
        
        # Start self-play
        self.self_play_thread = threading.Thread(
            target=self.self_play_loop,
            daemon=True
        )
        self.self_play_thread.start()
        
        print("Real-time learning system started")
    
    def stop(self):
        """Stop all learning threads"""
        self.is_running = False
        
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        if self.consolidation_thread:
            self.consolidation_thread.join(timeout=5)
        if self.self_play_thread:
            self.self_play_thread.join(timeout=5)
        
        print("Real-time learning system stopped")
    
    def save_state(self, path: str):
        """Save complete learner state"""
        state = {
            'model_checkpoint': f"{path}_model.pt",
            'stats': self.stats,
            'learning_history': list(self.learning_history),
            'response_cache': list(self.response_cache),
            'config': self.config
        }
        
        # Save model
        self.model.save_checkpoint(state['model_checkpoint'])
        
        # Save learner state
        with open(f"{path}_learner.json", 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, path: str):
        """Load complete learner state"""
        # Load learner state
        with open(f"{path}_learner.json", 'r') as f:
            state = json.load(f)
        
        # Load model
        self.model.load_checkpoint(state['model_checkpoint'])
        
        # Restore state
        self.stats = state['stats']
        self.learning_history = deque(state['learning_history'], maxlen=1000)
        self.response_cache = deque(state['response_cache'], maxlen=100)
        self.config.update(state['config'])
        
        print(f"Loaded state with {self.stats['total_queries']} queries processed")


def create_realtime_system(
    vocab_size: int = 50000,
    dim: int = 768,
    checkpoint: Optional[str] = None
) -> Tuple[RealTimeCortexGPT, MultilingualTokenizer, RealTimeLearner]:
    """Create a complete real-time learning system"""
    
    # Create configuration
    config = AdvancedMemoryConfig()
    
    # Create model
    model = RealTimeCortexGPT(config, vocab_size, dim)
    
    # Create tokenizer
    tokenizer = MultilingualTokenizer(vocab_size=vocab_size)
    
    # Create learner
    learner = RealTimeLearner(model, tokenizer)
    
    # Load checkpoint if provided
    if checkpoint and os.path.exists(f"{checkpoint}_learner.json"):
        learner.load_state(checkpoint)
    
    return model, tokenizer, learner