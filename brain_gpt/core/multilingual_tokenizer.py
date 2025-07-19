"""
Multilingual tokenizer with Korean language support
Optimized for memory efficiency and brain-inspired processing
"""

import os
import json
import pickle
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
import tiktoken
import sentencepiece as spm
from collections import defaultdict


class MultilingualBrainTokenizer:
    """
    Efficient multilingual tokenizer with Korean support
    Uses hierarchical encoding inspired by language processing in the brain
    """
    
    def __init__(
        self,
        base_vocab_size: int = 50257,
        korean_vocab_size: int = 20000,
        cache_dir: str = "./tokenizer_cache"
    ):
        self.base_vocab_size = base_vocab_size
        self.korean_vocab_size = korean_vocab_size
        self.total_vocab_size = base_vocab_size + korean_vocab_size + 31  # Padding to multiple of 16
        self.cache_dir = cache_dir
        
        # Language markers
        self.lang_tokens = {
            'ko_start': base_vocab_size,
            'ko_end': base_vocab_size + 1,
            'en_start': base_vocab_size + 2,
            'en_end': base_vocab_size + 3,
            'code_start': base_vocab_size + 4,
            'code_end': base_vocab_size + 5,
        }
        
        # Initialize tokenizers
        self._init_tokenizers()
        
        # Language detection patterns
        self.korean_pattern = self._compile_korean_pattern()
        
        # Cache for frequent tokens
        self.token_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _init_tokenizers(self):
        """Initialize base and Korean tokenizers"""
        # GPT-2 tokenizer for English
        self.gpt2_tokenizer = tiktoken.get_encoding("gpt2")
        
        # Korean tokenizer (would need to be trained on Korean corpus)
        self.korean_tokenizer = None  # Placeholder for trained SentencePiece model
        self._init_korean_tokenizer()
        
    def _init_korean_tokenizer(self):
        """Initialize or load Korean SentencePiece tokenizer"""
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        korean_model_path = os.path.join(self.cache_dir, "korean_tokenizer.model")
        
        if os.path.exists(korean_model_path):
            # Load existing model
            self.korean_tokenizer = spm.SentencePieceProcessor()
            self.korean_tokenizer.Load(korean_model_path)
        else:
            # Create a mock tokenizer for demonstration
            # In production, this would be trained on a large Korean corpus
            print("Warning: Korean tokenizer not found. Using character-level fallback.")
            self.korean_tokenizer = KoreanCharacterTokenizer(
                vocab_size=self.korean_vocab_size,
                base_offset=self.base_vocab_size + 10
            )
            
    def _compile_korean_pattern(self):
        """Compile regex pattern for Korean text detection"""
        import re
        # Hangul syllables, Hangul Jamo, and Hangul compatibility Jamo
        korean_ranges = [
            (0xAC00, 0xD7A3),  # Hangul syllables
            (0x1100, 0x11FF),  # Hangul Jamo
            (0x3130, 0x318F),  # Hangul compatibility Jamo
            (0xA960, 0xA97F),  # Hangul Jamo Extended-A
            (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
        ]
        
        pattern = '|'.join([f'[\\u{start:04X}-\\u{end:04X}]' for start, end in korean_ranges])
        return re.compile(pattern)
        
    def detect_language(self, text: str) -> str:
        """Detect primary language of text"""
        if not text:
            return 'en'
            
        # Count characters by type
        korean_chars = len(self.korean_pattern.findall(text))
        total_chars = len(text)
        
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        
        # Determine language
        if korean_ratio > 0.3:
            return 'ko'
        elif any(keyword in text for keyword in ['def ', 'class ', 'import ', 'function', 'const ']):
            return 'code'
        else:
            return 'en'
            
    def encode(
        self,
        text: str,
        add_language_markers: bool = True,
        max_length: Optional[int] = None,
        language: Optional[str] = None
    ) -> List[int]:
        """
        Encode text with language-aware tokenization
        Uses brain-inspired chunking for efficiency
        """
        if not text:
            return []
            
        # Check cache
        cache_key = (text[:100], add_language_markers, language)  # Use first 100 chars as key
        if cache_key in self.token_cache:
            self.cache_hits += 1
            cached_result = self.token_cache[cache_key]
            if len(text) <= 100:
                return cached_result
                
        self.cache_misses += 1
        
        # Detect language if not specified
        if language is None:
            language = self.detect_language(text)
            
        # Split text into language chunks for mixed content
        chunks = self._split_by_language(text)
        
        all_tokens = []
        
        for chunk_text, chunk_lang in chunks:
            if add_language_markers:
                all_tokens.append(self.lang_tokens[f'{chunk_lang}_start'])
                
            if chunk_lang == 'ko':
                tokens = self._encode_korean(chunk_text)
            elif chunk_lang == 'code':
                tokens = self._encode_code(chunk_text)
            else:
                tokens = self._encode_english(chunk_text)
                
            all_tokens.extend(tokens)
            
            if add_language_markers:
                all_tokens.append(self.lang_tokens[f'{chunk_lang}_end'])
                
        # Apply max length
        if max_length and len(all_tokens) > max_length:
            all_tokens = all_tokens[:max_length]
            
        # Cache result if text is short
        if len(text) <= 100:
            self.token_cache[cache_key] = all_tokens
            
        return all_tokens
        
    def _split_by_language(self, text: str) -> List[Tuple[str, str]]:
        """Split text into chunks by language"""
        chunks = []
        current_chunk = []
        current_lang = None
        
        # Simple character-by-character analysis
        for char in text:
            if self.korean_pattern.match(char):
                detected_lang = 'ko'
            elif char.isascii():
                detected_lang = 'en'
            else:
                detected_lang = current_lang or 'en'
                
            if current_lang != detected_lang:
                if current_chunk:
                    chunks.append((''.join(current_chunk), current_lang))
                current_chunk = [char]
                current_lang = detected_lang
            else:
                current_chunk.append(char)
                
        if current_chunk:
            chunks.append((''.join(current_chunk), current_lang))
            
        return chunks
        
    def _encode_english(self, text: str) -> List[int]:
        """Encode English text using GPT-2 tokenizer"""
        return self.gpt2_tokenizer.encode_ordinary(text)
        
    def _encode_korean(self, text: str) -> List[int]:
        """Encode Korean text with special handling for Hangul structure"""
        if self.korean_tokenizer:
            if hasattr(self.korean_tokenizer, 'encode_as_ids'):
                # Real SentencePiece tokenizer
                tokens = self.korean_tokenizer.encode_as_ids(text)
            else:
                # Fallback character tokenizer
                tokens = self.korean_tokenizer.encode(text)
                
            # Offset tokens to Korean vocabulary range
            return [t + self.base_vocab_size + 10 for t in tokens]
        else:
            # Ultimate fallback: character-level encoding
            return [ord(c) % self.korean_vocab_size + self.base_vocab_size + 10 for c in text]
            
    def _encode_code(self, text: str) -> List[int]:
        """Encode code with special handling for programming constructs"""
        # For now, use GPT-2 tokenizer which handles code reasonably well
        return self.gpt2_tokenizer.encode_ordinary(text)
        
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text"""
        if not tokens:
            return ""
            
        text_parts = []
        current_tokens = []
        current_mode = 'en'
        
        for token in tokens:
            # Check for language markers
            if token in self.lang_tokens.values():
                # Process accumulated tokens
                if current_tokens:
                    text_parts.append(self._decode_tokens(current_tokens, current_mode))
                    current_tokens = []
                    
                # Update mode based on marker
                for lang_name, lang_token in self.lang_tokens.items():
                    if token == lang_token and '_start' in lang_name:
                        current_mode = lang_name.replace('_start', '')
                        break
            else:
                current_tokens.append(token)
                
        # Process remaining tokens
        if current_tokens:
            text_parts.append(self._decode_tokens(current_tokens, current_mode))
            
        return ''.join(text_parts)
        
    def _decode_tokens(self, tokens: List[int], mode: str) -> str:
        """Decode tokens based on language mode"""
        if mode == 'ko':
            # Adjust tokens back to Korean tokenizer range
            adjusted_tokens = [t - self.base_vocab_size - 10 for t in tokens
                             if t >= self.base_vocab_size + 10]
            if self.korean_tokenizer and hasattr(self.korean_tokenizer, 'decode'):
                return self.korean_tokenizer.decode(adjusted_tokens)
            else:
                # Fallback character decoding
                return ''.join([chr(t) for t in adjusted_tokens if t < 0x110000])
        else:
            # English/Code decoding
            valid_tokens = [t for t in tokens if t < self.base_vocab_size]
            if valid_tokens:
                return self.gpt2_tokenizer.decode(valid_tokens)
            return ""
            
    def get_vocab_size(self) -> int:
        """Return total vocabulary size"""
        return self.total_vocab_size
        
    def save(self, path: str):
        """Save tokenizer configuration"""
        config = {
            'base_vocab_size': self.base_vocab_size,
            'korean_vocab_size': self.korean_vocab_size,
            'total_vocab_size': self.total_vocab_size,
            'lang_tokens': self.lang_tokens,
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from configuration"""
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        tokenizer = cls(
            base_vocab_size=config['base_vocab_size'],
            korean_vocab_size=config['korean_vocab_size']
        )
        tokenizer.lang_tokens = config['lang_tokens']
        
        return tokenizer


class KoreanCharacterTokenizer:
    """
    Fallback character-level tokenizer for Korean
    Used when proper SentencePiece model is not available
    """
    
    def __init__(self, vocab_size: int, base_offset: int):
        self.vocab_size = vocab_size
        self.base_offset = base_offset
        
        # Common Korean characters and Jamo
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Initialize with common Hangul syllables
        syllables = []
        for i in range(0xAC00, min(0xAC00 + vocab_size - 1000, 0xD7A4)):
            syllables.append(chr(i))
            
        # Add Jamo
        for i in range(0x1100, 0x1200):
            syllables.append(chr(i))
            
        # Create mappings
        for i, char in enumerate(syllables[:vocab_size - 100]):
            self.char_to_id[char] = i
            self.id_to_char[i] = char
            
    def encode(self, text: str) -> List[int]:
        """Encode Korean text to character IDs"""
        tokens = []
        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                # Unknown character - use hash
                tokens.append(hash(char) % (self.vocab_size - 100) + 100)
        return tokens
        
    def decode(self, tokens: List[int]) -> str:
        """Decode character IDs back to Korean text"""
        chars = []
        for token in tokens:
            if token in self.id_to_char:
                chars.append(self.id_to_char[token])
            else:
                # Unknown token - skip
                pass
        return ''.join(chars)


class KoreanDataCollator:
    """
    Data collator that handles Korean-specific linguistic features
    Optimized for brain-inspired processing patterns
    """
    
    def __init__(self, tokenizer: MultilingualBrainTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate examples with Korean-aware padding and attention masks
        """
        # Extract texts and languages
        texts = [ex.get('text', '') for ex in examples]
        languages = [ex.get('language', None) for ex in examples]
        
        # Tokenize all texts
        all_input_ids = []
        all_attention_masks = []
        all_language_ids = []
        
        for text, lang in zip(texts, languages):
            # Encode with language markers
            input_ids = self.tokenizer.encode(
                text,
                add_language_markers=True,
                max_length=self.max_length,
                language=lang
            )
            
            # Pad or truncate
            if len(input_ids) < self.max_length:
                padding_length = self.max_length - len(input_ids)
                input_ids = input_ids + [0] * padding_length  # 0 is padding token
                attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length
            else:
                input_ids = input_ids[:self.max_length]
                attention_mask = [1] * self.max_length
                
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            
            # Language ID for adapter selection
            lang_id = 0 if lang == 'en' else 1 if lang == 'ko' else 2
            all_language_ids.append(lang_id)
            
        # Convert to tensors
        batch = {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_masks, dtype=torch.long),
            'language_ids': torch.tensor(all_language_ids, dtype=torch.long),
        }
        
        # Add labels for language modeling (shifted input_ids)
        batch['labels'] = batch['input_ids'].clone()
        batch['labels'][batch['attention_mask'] == 0] = -100  # Ignore padding in loss
        
        return batch