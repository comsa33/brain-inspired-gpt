#!/usr/bin/env python3
"""
RedPajama-Data-v2 Dataset Preparation
30 trillion tokens with quality annotations
Supports: English, French, Spanish, German, Italian
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import requests
from tqdm import tqdm
import datasets
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from brain_gpt.core.multilingual_tokenizer import MultilingualBrainTokenizer


class RedPajamaV2Downloader:
    """Download and prepare RedPajama-Data-v2 dataset"""
    
    DATASET_CONFIGS = {
        'default': {
            'name': 'default',
            'base_url': 'togethercomputer/RedPajama-Data-V2',
            'languages': ['en', 'de', 'fr', 'es', 'it'],
            'quality_signals': [
                'ccnet_bucket',
                'ccnet_language_score',
                'ccnet_length',
                'ccnet_nlines',
                'ccnet_original_length',
                'ccnet_original_nlines',
                'ccnet_perplexity',
                'rps_doc_curly_bracket',
                'rps_doc_frac_all_caps_words',
                'rps_doc_frac_lines_end_with_ellipsis',
                'rps_doc_frac_no_alph_words',
                'rps_doc_lorem_ipsum',
                'rps_doc_mean_word_length',
                'rps_doc_stop_word_fraction',
                'rps_doc_symbol_to_word_ratio'
            ]
        },
        'sample': {
            'name': 'sample',
            'base_url': 'togethercomputer/RedPajama-Data-V2-sample',
            'languages': ['en'],
            'quality_signals': ['ccnet_bucket', 'ccnet_perplexity']
        }
    }
    
    def __init__(self, 
                 output_dir: str = 'data/redpajama_v2',
                 config: str = 'sample',
                 languages: Optional[List[str]] = None,
                 max_samples: Optional[int] = None,
                 quality_threshold: float = 0.7,
                 num_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self.DATASET_CONFIGS[config]
        self.languages = languages or self.config['languages']
        self.max_samples = max_samples
        self.quality_threshold = quality_threshold
        self.num_workers = num_workers
        
        # Initialize tokenizer
        self.tokenizer = MultilingualBrainTokenizer()
        
        print(f"RedPajama-V2 Downloader initialized")
        print(f"Config: {config}")
        print(f"Languages: {self.languages}")
        print(f"Output directory: {self.output_dir}")
        
    def download_and_process(self):
        """Download and process RedPajama-V2 dataset"""
        all_texts = []
        
        for lang in self.languages:
            print(f"\nProcessing language: {lang}")
            
            try:
                # Load dataset from HuggingFace
                if self.config['name'] == 'sample':
                    # Try different sample configurations
                    try:
                        dataset = datasets.load_dataset(
                            'togethercomputer/RedPajama-Data-1T-Sample',
                            split='train',
                            streaming=True,
                            trust_remote_code=True
                        )
                    except:
                        # Fallback to direct language subset
                        dataset = datasets.load_dataset(
                            'togethercomputer/RedPajama-Data-1T',
                            lang,
                            split='train[:1000]',  # Sample first 1000 docs
                            streaming=True,
                            trust_remote_code=True
                        )
                else:
                    # Full dataset with language filtering
                    dataset = datasets.load_dataset(
                        self.config['base_url'],
                        name=lang,
                        split='train',
                        streaming=True,
                        trust_remote_code=True
                    )
                
                # Process documents
                lang_texts = self._process_documents(dataset, lang)
                all_texts.extend(lang_texts)
                
                print(f"Collected {len(lang_texts)} high-quality texts for {lang}")
                
            except Exception as e:
                print(f"Error processing {lang}: {e}")
                continue
        
        # Save processed data
        if all_texts:
            self._save_dataset(all_texts)
        
        return len(all_texts)
    
    def _process_documents(self, dataset, language: str) -> List[Dict]:
        """Process documents with quality filtering"""
        texts = []
        
        with tqdm(desc=f"Processing {language}", unit=" docs") as pbar:
            for i, doc in enumerate(dataset):
                if self.max_samples and i >= self.max_samples:
                    break
                
                # Extract text and quality signals
                text = doc.get('raw_content', '')
                if not text or len(text.strip()) < 100:
                    continue
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(doc)
                
                if quality_score >= self.quality_threshold:
                    texts.append({
                        'text': text,
                        'language': language,
                        'quality_score': quality_score,
                        'metadata': {
                            'source': 'redpajama_v2',
                            'doc_id': doc.get('doc_id', f'{language}_{i}'),
                            'quality_signals': {
                                signal: doc.get(signal, None)
                                for signal in self.config['quality_signals']
                                if signal in doc
                            }
                        }
                    })
                
                pbar.update(1)
        
        return texts
    
    def _calculate_quality_score(self, doc: Dict) -> float:
        """Calculate quality score based on multiple signals"""
        scores = []
        
        # CCNet bucket score (0 is highest quality)
        bucket = doc.get('ccnet_bucket', 3)
        if bucket is not None:
            scores.append(1.0 - (bucket / 3.0))
        
        # Perplexity score (lower is better, normalize to 0-1)
        perplexity = doc.get('ccnet_perplexity', 1000)
        if perplexity is not None:
            # Normalize perplexity (assuming range 0-2000)
            scores.append(max(0, 1.0 - (perplexity / 2000.0)))
        
        # Language score (higher is better)
        lang_score = doc.get('ccnet_language_score', 0.5)
        if lang_score is not None:
            scores.append(lang_score)
        
        # Document length score (prefer medium-length documents)
        length = doc.get('ccnet_length', 0)
        if length > 0:
            # Optimal length around 1000-5000 characters
            if 1000 <= length <= 5000:
                scores.append(1.0)
            elif length < 1000:
                scores.append(length / 1000.0)
            else:
                scores.append(max(0.3, 1.0 - (length - 5000) / 50000.0))
        
        # Fraction of stop words (higher is better for natural text)
        stop_word_frac = doc.get('rps_doc_stop_word_fraction', 0.3)
        if stop_word_frac is not None:
            scores.append(min(1.0, stop_word_frac * 2))
        
        # Calculate average score
        return sum(scores) / len(scores) if scores else 0.0
    
    def _save_dataset(self, texts: List[Dict]):
        """Save processed dataset"""
        print(f"\nSaving {len(texts)} texts...")
        
        # Split into train/validation
        np.random.shuffle(texts)
        split_idx = int(0.95 * len(texts))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        # Save metadata
        metadata = {
            'source': 'redpajama_v2',
            'version': '2.0',
            'languages': list(set(t['language'] for t in texts)),
            'total_texts': len(texts),
            'train_texts': len(train_texts),
            'val_texts': len(val_texts),
            'quality_threshold': self.quality_threshold,
            'download_date': time.strftime('%Y-%m-%d')
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Tokenize and save
        self._tokenize_and_save(train_texts, 'train')
        self._tokenize_and_save(val_texts, 'val')
        
        print(f"Dataset saved to {self.output_dir}")
    
    def _tokenize_and_save(self, texts: List[Dict], split: str):
        """Tokenize texts and save as binary files"""
        print(f"Tokenizing {split} split...")
        
        all_tokens = []
        
        for text_data in tqdm(texts, desc=f"Tokenizing {split}"):
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
        output_path = self.output_dir / f'redpajama_v2_{split}.bin'
        tokens_array.tofile(output_path)
        
        print(f"Saved {len(tokens_array):,} tokens to {output_path}")
        
        # Also save as JSONL for debugging
        jsonl_path = self.output_dir / f'redpajama_v2_{split}.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for text_data in texts[:100]:  # Save first 100 for inspection
                f.write(json.dumps(text_data, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Download and prepare RedPajama-V2 dataset')
    parser.add_argument('--output-dir', type=str, default='data/redpajama_v2',
                        help='Output directory for processed data')
    parser.add_argument('--config', type=str, default='sample', choices=['sample', 'default'],
                        help='Dataset configuration (sample for testing, default for full)')
    parser.add_argument('--languages', nargs='+', default=None,
                        help='Languages to download (default: all available)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per language (default: all)')
    parser.add_argument('--quality-threshold', type=float, default=0.7,
                        help='Quality score threshold (0-1)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = RedPajamaV2Downloader(
        output_dir=args.output_dir,
        config=args.config,
        languages=args.languages,
        max_samples=args.max_samples,
        quality_threshold=args.quality_threshold,
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