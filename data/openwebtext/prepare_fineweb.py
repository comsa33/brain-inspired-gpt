#!/usr/bin/env python3
"""
FineWeb Dataset Preparation
15 trillion tokens of high-quality web data
FineWeb-Edu: 1.3T tokens of educational content
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
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from brain_gpt.core.multilingual_tokenizer import MultilingualBrainTokenizer


class FineWebDownloader:
    """Download and prepare FineWeb/FineWeb-Edu dataset"""
    
    DATASET_CONFIGS = {
        'fineweb': {
            'path': 'HuggingFaceFW/fineweb',
            'name': 'default',
            'splits': ['train'],
            'description': '15T tokens of high-quality web data'
        },
        'fineweb-edu': {
            'path': 'HuggingFaceFW/fineweb-edu',
            'name': 'default',
            'splits': ['train'],
            'description': '1.3T tokens of educational content'
        },
        'fineweb-edu-5.4': {
            'path': 'HuggingFaceFW/fineweb-edu',
            'name': 'sample-350BT',  # Use sample for testing
            'splits': ['train'],
            'description': '5.4T tokens of high educational content'
        }
    }
    
    def __init__(self,
                 output_dir: str = 'data/fineweb',
                 dataset_type: str = 'fineweb-edu',
                 max_samples: Optional[int] = None,
                 chunk_size: int = 10000,
                 num_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_type = dataset_type
        self.config = self.DATASET_CONFIGS[dataset_type]
        self.max_samples = max_samples
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        
        # Initialize tokenizer
        self.tokenizer = MultilingualBrainTokenizer()
        
        print(f"FineWeb Downloader initialized")
        print(f"Dataset: {dataset_type} - {self.config['description']}")
        print(f"Output directory: {self.output_dir}")
        
    def download_and_process(self):
        """Download and process FineWeb dataset"""
        all_texts = []
        
        print(f"\nLoading {self.dataset_type} dataset...")
        
        try:
            # Load dataset with streaming for memory efficiency
            dataset = datasets.load_dataset(
                self.config['path'],
                name=self.config['name'],
                split='train',
                streaming=True,
                trust_remote_code=True
            )
            
            # Process documents
            texts = self._process_documents(dataset)
            
            print(f"\nCollected {len(texts)} texts")
            
            # Save processed data
            if texts:
                self._save_dataset(texts)
            
            return len(texts)
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Trying alternative configuration...")
            
            # Fallback to sample dataset
            try:
                dataset = datasets.load_dataset(
                    'HuggingFaceFW/fineweb-edu',
                    name='sample-10BT',
                    split='train',
                    streaming=True,
                    trust_remote_code=True
                )
                
                texts = self._process_documents(dataset)
                
                if texts:
                    self._save_dataset(texts)
                
                return len(texts)
                
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return 0
    
    def _process_documents(self, dataset) -> List[Dict]:
        """Process documents from FineWeb"""
        texts = []
        
        with tqdm(desc="Processing documents", unit=" docs") as pbar:
            for i, doc in enumerate(dataset):
                if self.max_samples and i >= self.max_samples:
                    break
                
                # Extract text
                text = doc.get('text', '')
                if not text or len(text.strip()) < 100:
                    continue
                
                # Extract metadata
                metadata = {
                    'source': 'fineweb',
                    'dataset': self.dataset_type,
                    'doc_id': doc.get('id', f'doc_{i}'),
                    'dump': doc.get('dump', 'unknown'),
                    'url': doc.get('url', ''),
                    'date': doc.get('date', ''),
                    'metadata': doc.get('metadata', {})
                }
                
                # Educational score for FineWeb-Edu
                if 'edu' in self.dataset_type:
                    edu_score = doc.get('educational_score', 
                                      doc.get('score', 0.5))
                    metadata['educational_score'] = edu_score
                
                # Detect language (FineWeb is primarily English)
                language = self._detect_language(text)
                
                texts.append({
                    'text': text,
                    'language': language,
                    'metadata': metadata
                })
                
                pbar.update(1)
        
        return texts
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # FineWeb is primarily English, but check for other languages
        sample = text[:1000]
        
        # Check for non-ASCII characters
        non_ascii_count = sum(1 for c in sample if ord(c) > 127)
        
        if non_ascii_count / len(sample) > 0.1:
            # Likely non-English, use tokenizer's detection
            return self.tokenizer.detect_language(text)
        
        return 'en'
    
    def _save_dataset(self, texts: List[Dict]):
        """Save processed dataset"""
        # Convert generator to list if needed
        if hasattr(texts, '__iter__') and not isinstance(texts, list):
            all_texts = []
            for text in texts:
                all_texts.append(text)
            texts = all_texts
        
        print(f"\nSaving {len(texts)} texts...")
        
        # Split into train/validation
        np.random.shuffle(texts)
        split_idx = int(0.95 * len(texts))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        # Save metadata
        metadata = {
            'source': 'fineweb',
            'dataset_type': self.dataset_type,
            'total_texts': len(texts),
            'train_texts': len(train_texts),
            'val_texts': len(val_texts),
            'languages': list(set(t['language'] for t in texts)),
            'download_date': time.strftime('%Y-%m-%d')
        }
        
        # Add educational stats for FineWeb-Edu
        if 'edu' in self.dataset_type:
            edu_scores = [t['metadata'].get('educational_score', 0) 
                         for t in texts if 'educational_score' in t['metadata']]
            if edu_scores:
                metadata['educational_stats'] = {
                    'mean_score': np.mean(edu_scores),
                    'min_score': np.min(edu_scores),
                    'max_score': np.max(edu_scores),
                    'std_score': np.std(edu_scores)
                }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Tokenize and save
        self._tokenize_and_save(train_texts, 'train')
        self._tokenize_and_save(val_texts, 'val')
        
        print(f"Dataset saved to {self.output_dir}")
    
    def _tokenize_and_save(self, texts: List[Dict], split: str):
        """Tokenize texts and save as binary files"""
        print(f"Tokenizing {split} split with {len(texts)} texts...")
        
        all_tokens = []
        
        # Process in batches for memory efficiency
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Tokenizing {split}"):
            batch = texts[i:i + batch_size]
            
            for text_data in batch:
                text = text_data['text']
                language = text_data['language']
                
                # Tokenize with language info
                tokens = self.tokenizer.encode(
                    text,
                    add_language_markers=True,
                    language=language
                )
                
                # Filter to valid range
                tokens = [t for t in tokens if 0 <= t < 65536]
                
                if tokens:
                    all_tokens.extend(tokens)
        
        # Save as numpy array
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        output_path = self.output_dir / f'{self.dataset_type}_{split}.bin'
        tokens_array.tofile(output_path)
        
        print(f"Saved {len(tokens_array):,} tokens to {output_path}")
        
        # Save sample texts for inspection
        sample_path = self.output_dir / f'{self.dataset_type}_{split}_samples.jsonl'
        with open(sample_path, 'w', encoding='utf-8') as f:
            for text_data in texts[:100]:  # First 100 samples
                sample = {
                    'text': text_data['text'][:500] + '...',  # First 500 chars
                    'language': text_data['language'],
                    'metadata': text_data['metadata']
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Download and prepare FineWeb dataset')
    parser.add_argument('--output-dir', type=str, default='data/fineweb',
                        help='Output directory for processed data')
    parser.add_argument('--dataset-type', type=str, default='fineweb-edu',
                        choices=['fineweb', 'fineweb-edu', 'fineweb-edu-5.4'],
                        help='FineWeb dataset variant')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help='Chunk size for processing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = FineWebDownloader(
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers
    )
    
    # Download and process
    start_time = time.time()
    num_texts = downloader.download_and_process()
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Total texts processed: {num_texts:,}")


if __name__ == "__main__":
    main()