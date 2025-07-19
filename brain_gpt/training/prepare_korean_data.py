"""
Korean dataset preparation for Brain-Inspired GPT
Handles multiple Korean datasets with linguistic awareness
"""

import os
import sys
import json
import numpy as np
import re
from tqdm import tqdm
from typing import List, Dict, Optional
import datasets
from datasets import load_dataset, concatenate_datasets
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.multilingual_tokenizer import MultilingualBrainTokenizer


class KoreanDatasetPreprocessor:
    """
    Processes Korean datasets with awareness of linguistic features
    """
    
    def __init__(
        self,
        tokenizer: MultilingualBrainTokenizer,
        output_dir: str = "./data/korean",
        max_length: int = 2048
    ):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.max_length = max_length
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Korean-specific text processing
        self.honorific_levels = ['formal', 'polite', 'informal']
        self.particle_list = ['은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로']
        
    def clean_text(self, text: str) -> str:
        """Clean Korean text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove repeated punctuation
        text = re.sub(r'([!?~]){2,}', r'\1', text)
        # Remove laughing patterns like ㅋㅋㅋ
        text = re.sub(r'[ㅋㅎ]{3,}', '', text)
        return text.strip()
        
    def split_sentences(self, text: str) -> List[str]:
        """Split Korean text into sentences"""
        # Simple sentence splitting on Korean sentence endings
        sentences = re.split(r'[.!?]+', text)
        # Filter empty sentences
        return [s.strip() for s in sentences if s.strip()]
        
    def prepare_datasets(self) -> Dict[str, datasets.Dataset]:
        """
        Load and prepare multiple Korean datasets
        """
        print("Loading Korean datasets...")
        
        all_datasets = {}
        
        # 1. Korean Wikipedia
        try:
            wiki_dataset = self._prepare_korean_wikipedia()
            if wiki_dataset:
                all_datasets['wikipedia'] = wiki_dataset
        except Exception as e:
            print(f"Failed to load Korean Wikipedia: {e}")
            
        # 2. Korean dialogue dataset (if available)
        try:
            dialogue_dataset = self._prepare_korean_dialogue()
            if dialogue_dataset:
                all_datasets['dialogue'] = dialogue_dataset
        except Exception as e:
            print(f"Failed to load Korean dialogue: {e}")
            
        # 3. Korean news dataset
        try:
            news_dataset = self._prepare_korean_news()
            if news_dataset:
                all_datasets['news'] = news_dataset
        except Exception as e:
            print(f"Failed to load Korean news: {e}")
            
        # 4. Create synthetic Korean data for testing
        synthetic_dataset = self._create_synthetic_korean()
        all_datasets['synthetic'] = synthetic_dataset
        
        return all_datasets
        
    def _prepare_korean_wikipedia(self) -> Optional[datasets.Dataset]:
        """Load and process Korean Wikipedia"""
        try:
            # Load Korean Wikipedia (small subset for testing)
            dataset = load_dataset(
                "wikipedia", 
                "20231101.ko",
                split="train[:1000]",  # Small subset for testing
                trust_remote_code=True
            )
            
            # Process Wikipedia articles
            def process_wiki(example):
                text = example.get('text', '')
                
                # Clean and normalize text
                text = self._normalize_korean_text(text)
                
                # Tokenize
                tokens = self.tokenizer.encode(
                    text,
                    add_language_markers=True,
                    max_length=self.max_length,
                    language='ko'
                )
                
                return {
                    'input_ids': tokens,
                    'length': len(tokens),
                    'language': 'ko',
                    'domain': 'encyclopedia'
                }
                
            dataset = dataset.map(
                process_wiki,
                remove_columns=dataset.column_names,
                desc="Processing Korean Wikipedia"
            )
            
            return dataset
            
        except Exception as e:
            print(f"Error loading Korean Wikipedia: {e}")
            return None
            
    def _prepare_korean_dialogue(self) -> Optional[datasets.Dataset]:
        """Prepare Korean conversational data"""
        # This would load actual Korean dialogue datasets
        # For now, return None as placeholder
        return None
        
    def _prepare_korean_news(self) -> Optional[datasets.Dataset]:
        """Prepare Korean news articles"""
        # This would load actual Korean news datasets
        # For now, return None as placeholder
        return None
        
    def _create_synthetic_korean(self) -> datasets.Dataset:
        """Create synthetic Korean data for testing"""
        print("Creating synthetic Korean dataset...")
        
        synthetic_examples = []
        
        # Example sentences covering different linguistic features
        templates = [
            # Formal polite
            ("안녕하세요. 저는 {name}입니다.", "formal"),
            ("오늘 날씨가 매우 좋습니다.", "formal"),
            ("한국어를 공부하고 있습니다.", "formal"),
            
            # Informal
            ("안녕! 나는 {name}이야.", "informal"),
            ("오늘 날씨 진짜 좋다.", "informal"),
            ("한국어 공부하고 있어.", "informal"),
            
            # Questions
            ("어디에 가십니까?", "formal"),
            ("뭐 먹을래?", "informal"),
            ("이것은 무엇입니까?", "formal"),
            
            # Complex sentences
            ("저는 서울에서 태어났지만 부산에서 자랐습니다.", "formal"),
            ("비가 오면 집에 있을 거예요.", "polite"),
            ("책을 읽는 것을 좋아해요.", "polite"),
            
            # Technical/Modern
            ("인공지능 기술이 빠르게 발전하고 있습니다.", "formal"),
            ("스마트폰으로 쇼핑하는 것이 편리합니다.", "formal"),
            ("프로그래밍을 배우고 싶어요.", "polite"),
        ]
        
        # Korean names for templates
        names = ["철수", "영희", "민수", "지영", "성민", "은지"]
        
        # Generate variations
        for _ in range(1000):  # Create 1000 examples
            template, formality = templates[np.random.randint(len(templates))]
            
            # Fill in template
            text = template
            if "{name}" in text:
                text = text.format(name=np.random.choice(names))
                
            # Add variations
            if np.random.random() > 0.5:
                # Add particles
                particle = np.random.choice(self.particle_list)
                text += f" 그것{particle} 좋아요."
                
            # Tokenize
            tokens = self.tokenizer.encode(
                text,
                add_language_markers=True,
                max_length=self.max_length,
                language='ko'
            )
            
            synthetic_examples.append({
                'input_ids': tokens,
                'length': len(tokens),
                'language': 'ko',
                'formality': formality,
                'domain': 'synthetic'
            })
            
        # Convert to dataset
        dataset = datasets.Dataset.from_list(synthetic_examples)
        
        return dataset
        
    def _normalize_korean_text(self, text: str) -> str:
        """Normalize Korean text for consistent processing"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove zero-width spaces
        text = text.replace('\u200b', '')
        
        return text
        
    def create_training_files(self, datasets_dict: Dict[str, datasets.Dataset]):
        """
        Create memory-mapped training files from datasets
        """
        print("Creating training files...")
        
        # Combine all datasets
        all_examples = []
        
        for name, dataset in datasets_dict.items():
            print(f"Processing {name} dataset with {len(dataset)} examples")
            all_examples.extend(dataset)
            
        # Shuffle examples
        np.random.shuffle(all_examples)
        
        # Split into train/validation
        n_examples = len(all_examples)
        n_train = int(0.95 * n_examples)
        
        train_examples = all_examples[:n_train]
        val_examples = all_examples[n_train:]
        
        # Create memory-mapped files
        self._create_memmap_file(train_examples, 'train_korean.bin')
        self._create_memmap_file(val_examples, 'val_korean.bin')
        
        # Save metadata
        metadata = {
            'n_train': len(train_examples),
            'n_val': len(val_examples),
            'max_length': self.max_length,
            'vocab_size': self.tokenizer.get_vocab_size(),
            'datasets': list(datasets_dict.keys())
        }
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Created training files with {len(train_examples)} train and {len(val_examples)} validation examples")
        
    def _create_memmap_file(self, examples: List[Dict], filename: str):
        """Create memory-mapped file from examples"""
        filepath = os.path.join(self.output_dir, filename)
        
        # Calculate total tokens
        total_tokens = sum(ex['length'] for ex in examples)
        
        # Create memory-mapped array
        dtype = np.uint16  # Assuming vocab size < 65536
        arr = np.memmap(filepath, dtype=dtype, mode='w+', shape=(total_tokens,))
        
        # Fill array
        idx = 0
        for ex in tqdm(examples, desc=f"Writing {filename}"):
            tokens = ex['input_ids']
            arr[idx:idx + len(tokens)] = tokens
            idx += len(tokens)
            
        arr.flush()
        print(f"Created {filepath} with {total_tokens:,} tokens")


class KoreanDataAugmenter:
    """
    Augment Korean data with linguistic variations
    """
    
    def __init__(self):
        self.formality_mappings = {
            '습니다': ['어요', '어'],  # Formal -> Polite/Informal
            '입니다': ['이에요', '이야'],
            '합니다': ['해요', '해'],
        }
        
    def augment_formality(self, text: str) -> List[str]:
        """Create formality variations of Korean text"""
        variations = [text]
        
        for formal, informal_options in self.formality_mappings.items():
            if formal in text:
                for informal in informal_options:
                    variation = text.replace(formal, informal)
                    if variation != text:
                        variations.append(variation)
                        
        return variations
        
    def augment_particles(self, text: str) -> List[str]:
        """Create particle variations"""
        # Korean particles can often be swapped for similar meanings
        particle_alternatives = {
            '은': ['는'],
            '이': ['가'],
            '을': ['를'],
        }
        
        variations = [text]
        
        for particle, alternatives in particle_alternatives.items():
            if particle in text:
                for alt in alternatives:
                    variation = text.replace(particle, alt)
                    if variation != text:
                        variations.append(variation)
                        
        return variations


def main():
    """Main function to prepare Korean datasets"""
    # Initialize tokenizer
    print("Initializing multilingual tokenizer...")
    tokenizer = MultilingualBrainTokenizer()
    
    # Initialize processor
    processor = KoreanDatasetPreprocessor(tokenizer)
    
    # Prepare datasets
    datasets_dict = processor.prepare_datasets()
    
    # Create training files
    processor.create_training_files(datasets_dict)
    
    print("Korean dataset preparation complete!")
    

if __name__ == "__main__":
    main()