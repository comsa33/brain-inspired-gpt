"""
Optimized inference script for Brain-Inspired GPT
Shows how to use the model efficiently in production
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import time
from typing import List, Optional, Dict
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer


class BrainGPTInference:
    """
    Optimized inference engine for Brain-Inspired GPT
    Includes batching, caching, and RTX 3090 optimizations
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        compile_model: bool = True,
        use_cache: bool = True
    ):
        self.device = torch.device(device)
        
        # Initialize model
        print("Loading Brain-Inspired GPT for inference...")
        self.config = BrainGPTConfig()
        self.model = BrainGPT(self.config)
        
        if model_path:
            self.load_checkpoint(model_path)
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Compile model for faster inference
        if compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            
        # Initialize tokenizer
        self.tokenizer = MultilingualBrainTokenizer()
        
        # Cache for common prompts
        self.cache = {} if use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        print(f"✅ Model ready for inference on {self.device}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded checkpoint from {path}")
        
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        language: Optional[str] = None,
        return_stats: bool = False
    ) -> Dict:
        """
        Generate text from prompt with advanced sampling
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature (0.0 = greedy)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            language: Language hint ('en', 'ko', or None for auto-detect)
            return_stats: Whether to return generation statistics
            
        Returns:
            Dictionary with generated text and optional statistics
        """
        # Check cache
        cache_key = (prompt, max_length, temperature, top_k, top_p, language)
        if self.cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
            
        self.cache_misses += 1
        
        # Start timing
        start_time = time.time()
        
        # Detect language if not specified
        if language is None:
            language = self.tokenizer.detect_language(prompt)
            
        # Tokenize
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, language=language)
        ).unsqueeze(0).to(self.device)
        
        # Generate with mixed precision
        with autocast():
            output_ids = self._generate_tokens(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                language_id=language if language in ['en', 'ko'] else None
            )
            
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist())
        
        # Calculate statistics
        generation_time = time.time() - start_time
        tokens_generated = len(output_ids[0]) - len(input_ids[0])
        tokens_per_second = tokens_generated / generation_time
        
        result = {
            "text": generated_text,
            "language": language,
            "tokens_generated": tokens_generated,
            "time_seconds": generation_time,
            "tokens_per_second": tokens_per_second
        }
        
        if return_stats:
            result["stats"] = {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "model_energy": self.model.energy_consumed.item() if hasattr(self.model, 'energy_consumed') else 0,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
            }
            
        # Cache result
        if self.cache and len(self.cache) < 1000:  # Limit cache size
            self.cache[cache_key] = result
            
        return result
        
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        language_id: Optional[str]
    ) -> torch.Tensor:
        """
        Core token generation with advanced sampling
        """
        batch_size = input_ids.shape[0]
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Check if we've hit the context limit
            if input_ids.shape[1] >= self.config.block_size:
                # Use sliding window
                input_ids = input_ids[:, -self.config.block_size + 1:]
                
            # Forward pass
            with autocast():
                logits, _ = self.model(input_ids, language_id=language_id)
                
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_keep = torch.topk(logits, min(top_k, logits.shape[-1]))[1]
                logits_mask = torch.full_like(logits, float('-inf'))
                logits_mask.scatter_(1, indices_to_keep, 0)
                logits = logits + logits_mask
                
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Find cutoff
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter back
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS token
            if next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                break
                
        return input_ids
        
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        Generate text for multiple prompts efficiently
        """
        results = []
        
        # Process in batches for efficiency
        batch_size = 4  # Optimal for RTX 3090
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Generate for each prompt in batch
            # (True batching would require padding and attention masks)
            for prompt in batch_prompts:
                result = self.generate(prompt, **kwargs)
                results.append(result)
                
        return results
        
    def stream_generate(
        self,
        prompt: str,
        max_length: int = 100,
        **kwargs
    ):
        """
        Stream generation token by token
        Useful for real-time applications
        """
        # Tokenize prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt)
        ).unsqueeze(0).to(self.device)
        
        # Generate tokens one at a time
        for i in range(max_length):
            with torch.no_grad(), autocast():
                logits, _ = self.model(input_ids)
                
            # Get next token
            next_token_logits = logits[0, -1, :] / kwargs.get('temperature', 0.8)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode single token
            token_text = self.tokenizer.decode([next_token.item()])
            
            yield token_text
            
            # Append token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS
            if next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                break
                
    def benchmark_inference(self):
        """Benchmark inference performance"""
        print("\n🚀 Benchmarking Brain-GPT Inference Performance")
        print("-" * 50)
        
        test_prompts = [
            ("English", "The future of artificial intelligence is"),
            ("Korean", "인공지능 기술의 발전은"),
            ("Code", "def optimize_neural_network("),
        ]
        
        total_time = 0
        total_tokens = 0
        
        # Warmup
        print("Warming up...")
        for _ in range(5):
            self.generate("Hello world", max_length=10)
            
        print("\nRunning benchmark...")
        
        for lang, prompt in test_prompts:
            print(f"\n{lang}: {prompt}")
            
            result = self.generate(
                prompt,
                max_length=50,
                temperature=0.8,
                return_stats=True
            )
            
            print(f"Generated: {result['text'][len(prompt):][:50]}...")
            print(f"Speed: {result['tokens_per_second']:.0f} tokens/sec")
            print(f"Time: {result['time_seconds']:.3f}s")
            
            total_time += result['time_seconds']
            total_tokens += result['tokens_generated']
            
        # Summary
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        print(f"Average speed: {total_tokens/total_time:.0f} tokens/sec")
        print(f"Peak memory: {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB")
        print(f"Cache hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%")
        

def main():
    """Example usage of Brain-GPT inference"""
    
    # Initialize inference engine
    engine = BrainGPTInference(
        model_path=None,  # Use randomly initialized model for demo
        compile_model=True,
        use_cache=True
    )
    
    # Example 1: Simple generation
    print("\n📝 Example 1: Simple Generation")
    result = engine.generate(
        "The brain-inspired AI model",
        max_length=50,
        temperature=0.8
    )
    print(f"Generated: {result['text']}")
    print(f"Speed: {result['tokens_per_second']:.0f} tokens/sec")
    
    # Example 2: Korean generation
    print("\n🇰🇷 Example 2: Korean Generation")
    result = engine.generate(
        "뇌과학 기반의 인공지능은",
        max_length=50,
        language="ko"
    )
    print(f"Generated: {result['text']}")
    
    # Example 3: Batch generation
    print("\n📚 Example 3: Batch Generation")
    prompts = [
        "AI will change",
        "The future is",
        "Technology enables"
    ]
    results = engine.batch_generate(prompts, max_length=30)
    for prompt, result in zip(prompts, results):
        print(f"{prompt} → {result['text'][len(prompt):]}")
        
    # Example 4: Streaming
    print("\n🌊 Example 4: Streaming Generation")
    print("Streaming: ", end='', flush=True)
    for token in engine.stream_generate("Once upon a time", max_length=30):
        print(token, end='', flush=True)
        time.sleep(0.05)  # Simulate real-time
    print()
    
    # Run benchmark
    engine.benchmark_inference()
    

if __name__ == "__main__":
    main()