"""
Multilingual tokenizer for CortexGPT supporting Korean and English.
Uses BPE with special handling for Korean characters.
"""

import json
import regex as re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import unicodedata
from collections import defaultdict
import numpy as np


class MultilingualTokenizer:
    """
    A BPE tokenizer that properly handles Korean and English text.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, int]] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<unk>": 1,
            "<eos>": 2,
            "<bos>": 3,
            "<mask>": 4,
        }
        
        # Korean-specific tokens
        self.korean_special = {
            "<ko_start>": 5,
            "<ko_end>": 6,
            "<en_start>": 7,
            "<en_end>": 8,
        }
        self.special_tokens.update(self.korean_special)
        
        # Initialize vocabulary
        self.vocab = {}
        self.reverse_vocab = {}
        self.merges = []
        
        # Regex patterns for tokenization
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        # Korean-specific patterns
        self.korean_pat = re.compile(r'[\u3131-\u3163\uac00-\ud7a3]+')
        self.jamo_pat = re.compile(r'[\u1100-\u11FF\u3130-\u318F\uA960-\uA97F]+')
        
    def _get_pairs(self, word: List[str]) -> set:
        """Get all adjacent pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _is_korean(self, text: str) -> bool:
        """Check if text contains Korean characters."""
        return bool(self.korean_pat.search(text))
    
    def _split_korean_jamo(self, char: str) -> List[str]:
        """Split Korean character into jamo (consonants and vowels)."""
        if not self._is_korean(char):
            return [char]
            
        # Decompose Korean character
        decomposed = unicodedata.normalize('NFD', char)
        return list(decomposed)
    
    def _tokenize_korean(self, text: str) -> List[str]:
        """Special tokenization for Korean text."""
        tokens = []
        
        # Split into syllables
        syllables = list(text)
        
        for syllable in syllables:
            # Option 1: Keep syllables intact
            tokens.append(syllable)
            
            # Option 2: Decompose to jamo for better generalization
            # jamos = self._split_korean_jamo(syllable)
            # tokens.extend(jamos)
            
        return tokens
    
    def pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into words."""
        words = []
        
        # Split by language
        current_lang = None
        current_word = []
        
        for char in text:
            if self._is_korean(char):
                if current_lang != 'ko':
                    if current_word:
                        words.append(''.join(current_word))
                    current_word = []
                    current_lang = 'ko'
                current_word.append(char)
            else:
                if current_lang == 'ko':
                    if current_word:
                        words.append(''.join(current_word))
                    current_word = []
                    current_lang = 'en'
                current_word.append(char)
        
        if current_word:
            words.append(''.join(current_word))
        
        # Further tokenize each word
        tokenized = []
        for word in words:
            if self._is_korean(word):
                tokenized.append("<ko_start>")
                tokenized.extend(self._tokenize_korean(word))
                tokenized.append("<ko_end>")
            else:
                # Use regex for English
                tokenized.extend(re.findall(self.pat, word))
        
        return tokenized
    
    def learn_bpe(self, texts: List[str], verbose: bool = True):
        """Learn BPE merges from a corpus."""
        # Count word frequencies
        word_freqs = defaultdict(int)
        
        for text in texts:
            words = self.pre_tokenize(text)
            for word in words:
                word_freqs[word] += 1
        
        # Initialize vocabulary with characters
        vocab = defaultdict(int)
        for word, freq in word_freqs.items():
            for char in word:
                vocab[char] += freq
        
        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
        
        # Add character-level tokens
        current_idx = len(self.special_tokens)
        for char, freq in vocab.items():
            if char not in self.vocab and freq >= self.min_frequency:
                self.vocab[char] = current_idx
                self.reverse_vocab[current_idx] = char
                current_idx += 1
        
        # Split words into characters
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word)
        
        # Learn merges
        num_merges = self.vocab_size - current_idx
        
        for i in range(num_merges):
            # Stop if we've reached vocab size limit
            if current_idx >= self.vocab_size:
                break
                
            # Count pairs
            pair_freqs = defaultdict(int)
            
            for word, freq in word_freqs.items():
                split = splits[word]
                if len(split) == 1:
                    continue
                    
                for j in range(len(split) - 1):
                    pair = (split[j], split[j + 1])
                    pair_freqs[pair] += freq
            
            # Find most frequent pair
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Skip if frequency too low
            if pair_freqs[best_pair] < self.min_frequency:
                break
            
            # Merge the best pair
            self.merges.append(best_pair)
            
            # Add to vocabulary only if we have space
            if current_idx < self.vocab_size:
                new_token = best_pair[0] + best_pair[1]
                self.vocab[new_token] = current_idx
                self.reverse_vocab[current_idx] = new_token
                current_idx += 1
            
            # Update splits
            new_splits = {}
            for word, split in splits.items():
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                        new_split.append(split[i] + split[i + 1])
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                new_splits[word] = new_split
            splits = new_splits
            
            if verbose and (i + 1) % 1000 == 0:
                print(f"Learned {i + 1} merges...")
        
        if verbose:
            print(f"Vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subwords."""
        words = self.pre_tokenize(text)
        tokens = []
        
        for word in words:
            # Skip special tokens
            if word in self.special_tokens:
                tokens.append(word)
                continue
            
            # Apply BPE
            word_tokens = list(word)
            
            # Apply merges
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge[0] and word_tokens[i + 1] == merge[1]:
                        word_tokens = word_tokens[:i] + [merge[0] + merge[1]] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        ids = []
        
        # Get the actual vocab size (maximum valid token ID + 1)
        max_valid_id = len(self.vocab) - 1
        
        for token in tokens:
            if token in self.vocab:
                token_id = self.vocab[token]
                # Ensure token ID is within bounds
                if token_id <= max_valid_id:
                    ids.append(token_id)
                else:
                    ids.append(self.vocab["<unk>"])
            else:
                ids.append(self.vocab["<unk>"])
                
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        
        for id in ids:
            if id in self.reverse_vocab:
                token = self.reverse_vocab[id]
                # Skip all special tokens except for content-relevant ones
                if token in self.special_tokens:
                    # Only skip control tokens, not content tokens
                    if token in ["<pad>", "<unk>", "<eos>", "<bos>", "<mask>", 
                               "<ko_start>", "<ko_end>", "<en_start>", "<en_end>"]:
                        continue
                tokens.append(token)
            else:
                # Skip invalid IDs instead of adding <unk>
                continue
        
        # Join tokens
        text = "".join(tokens)
        
        # Clean up spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.special_tokens = data["special_tokens"]
        self.vocab_size = data["vocab_size"]
        self.min_frequency = data["min_frequency"]
        
        # Rebuild reverse vocab
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}


def create_tokenizer_from_corpus(
    corpus_path: str,
    save_path: str,
    vocab_size: int = 50000
) -> MultilingualTokenizer:
    """Create and train a tokenizer from a corpus file."""
    
    print("Creating multilingual tokenizer...")
    tokenizer = MultilingualTokenizer(vocab_size=vocab_size)
    
    # Load corpus
    print(f"Loading corpus from {corpus_path}...")
    texts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            if len(texts) >= 100000:  # Limit for training
                break
    
    # Learn BPE
    print("Learning BPE merges...")
    tokenizer.learn_bpe(texts)
    
    # Save tokenizer
    print(f"Saving tokenizer to {save_path}...")
    tokenizer.save(save_path)
    
    # Test
    test_texts = [
        "Hello, world!",
        "안녕하세요, 세계!",
        "The future of artificial intelligence",
        "인공지능의 미래",
        "def hello_world():",
        "함수 정의하기"
    ]
    
    print("\nTokenization examples:")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"Original: {text}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Decoded: {decoded}")
        print()
    
    return tokenizer


if __name__ == "__main__":
    # Example usage
    tokenizer = MultilingualTokenizer(vocab_size=10000)
    
    # Test corpus
    test_corpus = [
        "Hello, world! How are you today?",
        "안녕하세요! 오늘 어떻게 지내세요?",
        "The quick brown fox jumps over the lazy dog.",
        "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
        "Machine learning is fascinating.",
        "기계 학습은 매우 흥미롭습니다.",
        "def calculate_sum(a, b): return a + b",
        "함수를 정의하고 사용하는 방법"
    ]
    
    # Learn BPE
    tokenizer.learn_bpe(test_corpus, verbose=True)
    
    # Test tokenization
    for text in test_corpus[:4]:
        print(f"\nText: {text}")
        tokens = tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        ids = tokenizer.encode(text)
        print(f"IDs: {ids}")
        decoded = tokenizer.decode(ids)
        print(f"Decoded: {decoded}")