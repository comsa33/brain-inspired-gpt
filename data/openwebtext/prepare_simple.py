#!/usr/bin/env python3
"""
Simple Dataset Preparation for Quick Testing
Uses smaller, readily available datasets
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import datasets
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from brain_gpt.core.multilingual_tokenizer import MultilingualBrainTokenizer


class SimpleDatasetDownloader:
    """Download and prepare simple datasets for quick testing"""
    
    DATASET_CONFIGS = {
        'wikipedia': {
            'path': 'wikimedia/wikipedia',
            'name': '20231101.en',
            'language': 'en',
            'description': 'English Wikipedia articles'
        },
        'wikipedia-ko': {
            'path': 'wikimedia/wikipedia', 
            'name': '20231101.ko',
            'language': 'ko',
            'description': 'Korean Wikipedia articles'
        },
        'bookcorpus': {
            'path': 'bookcorpus/bookcorpus',
            'name': 'plain_text',
            'language': 'en',
            'description': 'Books corpus for language modeling'
        },
        'cc100-en': {
            'path': 'cc100',
            'name': 'en',
            'language': 'en',
            'description': 'Common Crawl corpus in English'
        },
        'cc100-ko': {
            'path': 'cc100',
            'name': 'ko',
            'language': 'ko',
            'description': 'Common Crawl corpus in Korean'
        },
        'oscar': {
            'path': 'oscar-corpus/OSCAR-2301',
            'name': 'en',
            'language': 'en',
            'description': 'OSCAR multilingual corpus'
        }
    }
    
    def __init__(self,
                 output_dir: str = 'data/simple',
                 datasets: Optional[List[str]] = None,
                 max_samples_per_dataset: int = 10000,
                 min_text_length: int = 100,
                 max_text_length: int = 10000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default to Wikipedia datasets for quick testing
        self.datasets = datasets or ['wikipedia', 'wikipedia-ko']
        self.max_samples = max_samples_per_dataset
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        
        # Initialize tokenizer
        self.tokenizer = MultilingualBrainTokenizer()
        
        print(f"Simple Dataset Downloader initialized")
        print(f"Datasets: {self.datasets}")
        print(f"Output directory: {self.output_dir}")
        
    def download_and_process(self):
        """Download and process all configured datasets"""
        all_texts = []
        dataset_stats = {}
        
        for dataset_name in self.datasets:
            if dataset_name not in self.DATASET_CONFIGS:
                print(f"Unknown dataset: {dataset_name}, skipping...")
                continue
            
            config = self.DATASET_CONFIGS[dataset_name]
            print(f"\nProcessing {dataset_name}: {config['description']}")
            
            try:
                # Try to load dataset
                texts = self._load_and_process_dataset(dataset_name, config)
                
                if texts:
                    all_texts.extend(texts)
                    dataset_stats[dataset_name] = len(texts)
                    print(f"Collected {len(texts)} texts from {dataset_name}")
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                # Try alternative loading method
                texts = self._load_alternative_dataset(dataset_name, config)
                if texts:
                    all_texts.extend(texts)
                    dataset_stats[dataset_name] = len(texts)
        
        # Save combined dataset
        if all_texts:
            self._save_dataset(all_texts, dataset_stats)
        
        return len(all_texts)
    
    def _load_and_process_dataset(self, name: str, config: Dict) -> List[Dict]:
        """Load and process a single dataset"""
        texts = []
        
        try:
            # Load dataset with streaming for memory efficiency
            dataset = datasets.load_dataset(
                config['path'],
                config.get('name'),
                split='train',
                streaming=True,
                trust_remote_code=True
            )
            
            with tqdm(desc=f"Processing {name}", total=self.max_samples) as pbar:
                for i, item in enumerate(dataset):
                    if i >= self.max_samples:
                        break
                    
                    # Extract text based on dataset format
                    text = self._extract_text(item, name)
                    
                    if text and self.min_text_length <= len(text) <= self.max_text_length:
                        texts.append({
                            'text': text,
                            'language': config['language'],
                            'source': name,
                            'metadata': {
                                'dataset': name,
                                'index': i
                            }
                        })
                    
                    pbar.update(1)
        
        except Exception as e:
            print(f"Streaming failed for {name}: {e}")
            # Try non-streaming approach
            try:
                dataset = datasets.load_dataset(
                    config['path'],
                    config.get('name'),
                    split='train[:5000]',  # Load only first 5000 samples
                    trust_remote_code=True
                )
                
                for i, item in enumerate(dataset):
                    if i >= self.max_samples:
                        break
                    
                    text = self._extract_text(item, name)
                    
                    if text and self.min_text_length <= len(text) <= self.max_text_length:
                        texts.append({
                            'text': text,
                            'language': config['language'],
                            'source': name,
                            'metadata': {
                                'dataset': name,
                                'index': i
                            }
                        })
            
            except Exception as e2:
                print(f"Non-streaming also failed for {name}: {e2}")
        
        return texts
    
    def _load_alternative_dataset(self, name: str, config: Dict) -> List[Dict]:
        """Try alternative dataset loading methods"""
        texts = []
        
        # Use alternative datasets
        alternatives = {
            'wikipedia': 'wikipedia simple english',
            'wikipedia-ko': 'KETI-AIR/korquad',
            'bookcorpus': 'bookcorpusopen/bookcorpusopen',
            'cc100-en': 'allenai/c4',
            'cc100-ko': 'KETI-AIR/korquad',
            'oscar': 'oscar'
        }
        
        if name in alternatives:
            print(f"Trying alternative dataset for {name}...")
            # Implementation would go here
            # For now, return empty list
        
        return texts
    
    def _extract_text(self, item: Dict, dataset_name: str) -> str:
        """Extract text from dataset item based on dataset format"""
        # Common field names for text
        text_fields = ['text', 'content', 'passage', 'document', 'sentence', 'paragraph']
        
        # Dataset-specific handling
        if 'wikipedia' in dataset_name:
            return item.get('text', item.get('content', ''))
        elif dataset_name == 'bookcorpus':
            return item.get('text', '')
        elif 'cc100' in dataset_name:
            return item.get('text', '')
        elif dataset_name == 'oscar':
            return item.get('text', item.get('content', ''))
        
        # Generic extraction
        for field in text_fields:
            if field in item and item[field]:
                return item[field]
        
        # If item is a string, return it directly
        if isinstance(item, str):
            return item
        
        return ''
    
    def _save_dataset(self, texts: List[Dict], stats: Dict):
        """Save processed dataset"""
        print(f"\nSaving {len(texts)} texts...")
        
        # Shuffle texts
        np.random.shuffle(texts)
        
        # Split into train/validation
        split_idx = int(0.95 * len(texts))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        # Language statistics
        lang_stats = {}
        for text in texts:
            lang = text['language']
            lang_stats[lang] = lang_stats.get(lang, 0) + 1
        
        # Save metadata
        metadata = {
            'source': 'simple_datasets',
            'datasets': list(stats.keys()),
            'dataset_stats': stats,
            'language_stats': lang_stats,
            'total_texts': len(texts),
            'train_texts': len(train_texts),
            'val_texts': len(val_texts),
            'min_text_length': self.min_text_length,
            'max_text_length': self.max_text_length,
            'download_date': time.strftime('%Y-%m-%d')
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Tokenize and save
        self._tokenize_and_save(train_texts, 'train')
        self._tokenize_and_save(val_texts, 'val')
        
        print(f"Dataset saved to {self.output_dir}")
        print(f"Language distribution: {lang_stats}")
    
    def _tokenize_and_save(self, texts: List[Dict], split: str):
        """Tokenize texts and save as binary files"""
        print(f"Tokenizing {split} split with {len(texts)} texts...")
        
        all_tokens = []
        
        for text_data in tqdm(texts, desc=f"Tokenizing {split}"):
            text = text_data['text']
            language = text_data['language']
            
            # Tokenize with language info
            tokens = self.tokenizer.encode(
                text,
                add_language_markers=True,
                language=language,
                max_length=2048  # Limit token length
            )
            
            # Filter to valid range
            tokens = [t for t in tokens if 0 <= t < 65536]
            
            if tokens:
                all_tokens.extend(tokens)
        
        # Save as numpy array
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        output_path = self.output_dir / f'simple_{split}.bin'
        tokens_array.tofile(output_path)
        
        print(f"Saved {len(tokens_array):,} tokens to {output_path}")
        
        # Save sample texts for inspection
        sample_path = self.output_dir / f'simple_{split}_samples.jsonl'
        with open(sample_path, 'w', encoding='utf-8') as f:
            for text_data in texts[:50]:  # First 50 samples
                sample = {
                    'text': text_data['text'][:500] + '...',
                    'language': text_data['language'],
                    'source': text_data['source']
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Download simple datasets for testing')
    parser.add_argument('--output-dir', type=str, default='data/simple',
                        help='Output directory for processed data')
    parser.add_argument('--datasets', nargs='+', 
                        default=['wikipedia', 'wikipedia-ko'],
                        help='Datasets to download')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Maximum samples per dataset')
    parser.add_argument('--min-length', type=int, default=100,
                        help='Minimum text length')
    parser.add_argument('--max-length', type=int, default=10000,
                        help='Maximum text length')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = SimpleDatasetDownloader(
        output_dir=args.output_dir,
        datasets=args.datasets,
        max_samples_per_dataset=args.max_samples,
        min_text_length=args.min_length,
        max_text_length=args.max_length
    )
    
    # Download and process
    start_time = time.time()
    num_texts = downloader.download_and_process()
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Total texts processed: {num_texts:,}")
    
    # Show next steps
    print("\nNext steps:")
    print("1. Run training with mixed datasets:")
    print("   uv run brain_gpt/training/train_brain_gpt_3090.py --data-dir data/simple")
    print("\n2. Or prepare larger datasets:")
    print("   uv run data/openwebtext/prepare_fineweb.py --max-samples 100000")
    print("   uv run data/openwebtext/prepare_redpajama.py --config sample")


if __name__ == "__main__":
    main()