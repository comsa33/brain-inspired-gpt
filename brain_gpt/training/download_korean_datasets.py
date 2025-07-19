#!/usr/bin/env python3
"""
Download and prepare Korean datasets for Brain-Inspired GPT training
Focuses on freely available, high-quality Korean text datasets
"""

import os
import sys
import json
import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.multilingual_tokenizer import MultilingualBrainTokenizer


class KoreanDatasetDownloader:
    """Download and prepare Korean datasets"""
    
    def __init__(self, data_dir: str = "data/korean_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = MultilingualBrainTokenizer()
        
    def download_file(self, url: str, filename: str, desc: str = "Downloading"):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        filepath = self.data_dir / filename
        
        with open(filepath, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))
                    
        return filepath
        
    def download_korean_wikipedia(self):
        """Download Korean Wikipedia dump"""
        print("\n📥 Downloading Korean Wikipedia...")
        
        # Use a recent stable dump
        wiki_url = "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2"
        wiki_file = "kowiki-latest-pages-articles.xml.bz2"
        
        try:
            # Download if not exists
            wiki_path = self.data_dir / wiki_file
            if not wiki_path.exists():
                print(f"Downloading from {wiki_url}")
                self.download_file(wiki_url, wiki_file, "Korean Wikipedia")
            else:
                print(f"✅ Wikipedia dump already exists: {wiki_path}")
                
            # Extract and process
            extracted_path = self.data_dir / "kowiki.xml"
            if not extracted_path.exists():
                print("Extracting Wikipedia dump...")
                import bz2
                with bz2.open(wiki_path, 'rb') as f_in:
                    with open(extracted_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        
            return extracted_path
            
        except Exception as e:
            print(f"❌ Error downloading Wikipedia: {e}")
            return None
            
    def process_wikipedia(self, xml_path: Path, max_articles: int = 10000):
        """Process Wikipedia XML and extract text"""
        print(f"\n📝 Processing Wikipedia articles (max {max_articles})...")
        
        articles = []
        count = 0
        
        try:
            # Parse XML
            for event, elem in ET.iterparse(xml_path, events=('start', 'end')):
                if event == 'end' and elem.tag.endswith('page'):
                    # Extract title and text
                    title_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}title')
                    text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}text')
                    
                    if title_elem is not None and text_elem is not None:
                        title = title_elem.text
                        text = text_elem.text
                        
                        if text and len(text) > 100:  # Skip short articles
                            articles.append({
                                'title': title,
                                'text': text[:5000]  # Limit text length
                            })
                            count += 1
                            
                            if count % 1000 == 0:
                                print(f"  Processed {count} articles...")
                                
                            if count >= max_articles:
                                break
                                
                    # Clear element to save memory
                    elem.clear()
                    
        except Exception as e:
            print(f"❌ Error processing Wikipedia: {e}")
            
        print(f"✅ Extracted {len(articles)} articles")
        return articles
        
    def download_klue_datasets(self):
        """Download KLUE benchmark datasets using Hugging Face"""
        print("\n📥 Downloading KLUE datasets...")
        
        try:
            from datasets import load_dataset
            
            klue_tasks = ['ynat', 'sts', 'nli', 'ner', 're', 'dp', 'mrc', 'wos']
            klue_data = {}
            
            for task in klue_tasks:
                try:
                    print(f"  Downloading KLUE-{task}...")
                    dataset = load_dataset('klue', task)
                    klue_data[task] = dataset
                    print(f"  ✅ Downloaded {task}: {len(dataset['train'])} train examples")
                except Exception as e:
                    print(f"  ⚠️  Failed to download {task}: {e}")
                    
            return klue_data
            
        except ImportError:
            print("❌ Please install datasets library: pip install datasets")
            return None
            
    def download_korquad(self):
        """Download KorQuAD dataset"""
        print("\n📥 Downloading KorQuAD...")
        
        try:
            from datasets import load_dataset
            
            # Download KorQuAD 1.0
            dataset = load_dataset("squad_kor_v1")
            
            print(f"✅ Downloaded KorQuAD:")
            print(f"   Train: {len(dataset['train'])} examples")
            print(f"   Validation: {len(dataset['validation'])} examples")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Error downloading KorQuAD: {e}")
            return None
            
    def download_korpora_datasets(self):
        """Download datasets from Korpora"""
        print("\n📥 Downloading Korpora datasets...")
        
        try:
            from Korpora import Korpora
            
            # List of useful datasets for LLM training
            corpus_names = [
                'kowikitext',
                'namuwikitext',
                'korean_chatbot_data',
                'nsmc',  # Movie reviews
                'kornli',
                'korsts',
            ]
            
            korpora_data = {}
            
            for corpus_name in corpus_names:
                try:
                    print(f"  Downloading {corpus_name}...")
                    corpus = Korpora.load(corpus_name)
                    korpora_data[corpus_name] = corpus
                    print(f"  ✅ Downloaded {corpus_name}")
                except Exception as e:
                    print(f"  ⚠️  Failed to download {corpus_name}: {e}")
                    
            return korpora_data
            
        except ImportError:
            print("❌ Please install Korpora: pip install Korpora")
            return None
            
    def create_training_data(self, texts: List[str], output_file: str):
        """Convert texts to tokenized training data"""
        print(f"\n🔧 Creating training data: {output_file}")
        
        all_tokens = []
        
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = self.tokenizer.encode(text, language='ko')
            all_tokens.extend(tokens)
            
        # Convert to numpy array
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        
        # Save
        output_path = self.data_dir / output_file
        tokens_array.tofile(output_path)
        
        print(f"✅ Saved {len(tokens_array):,} tokens to {output_path}")
        return output_path
        
    def prepare_combined_dataset(self):
        """Combine all downloaded datasets into training files"""
        print("\n🔧 Preparing combined training dataset...")
        
        all_texts = []
        
        # 1. Process Wikipedia if available
        wiki_path = self.data_dir / "kowiki.xml"
        if wiki_path.exists():
            articles = self.process_wikipedia(wiki_path, max_articles=5000)
            wiki_texts = [article['text'] for article in articles]
            all_texts.extend(wiki_texts)
            print(f"  Added {len(wiki_texts)} Wikipedia articles")
            
        # 2. Add sample Korean texts (for demo)
        sample_texts = [
            "인공지능 기술이 빠르게 발전하면서 우리의 일상생활에 많은 변화를 가져오고 있습니다.",
            "한국어는 한글이라는 독특한 문자 체계를 가지고 있으며, 이는 세종대왕이 창제했습니다.",
            "서울은 대한민국의 수도로서 천만 명 이상의 인구가 거주하는 대도시입니다.",
            "K-pop과 K-drama는 전 세계적으로 인기를 얻으며 한류 열풍을 이끌고 있습니다.",
            "한국의 전통 음식인 김치는 발효 식품으로 건강에 매우 좋습니다.",
            "5G 네트워크 기술에서 한국은 세계 최초로 상용화에 성공했습니다.",
            "한국의 교육열은 세계적으로 유명하며, 대학 진학률이 매우 높습니다.",
            "제주도는 한국의 대표적인 관광지로 아름다운 자연경관을 자랑합니다.",
            "한국의 IT 산업은 삼성전자와 LG전자를 중심으로 세계 시장을 선도하고 있습니다.",
            "한글날은 매년 10월 9일로, 한글 창제를 기념하는 국경일입니다.",
        ] * 100  # Repeat for more data
        
        all_texts.extend(sample_texts)
        print(f"  Added {len(sample_texts)} sample texts")
        
        # Create training and validation splits
        split_idx = int(len(all_texts) * 0.9)
        train_texts = all_texts[:split_idx]
        val_texts = all_texts[split_idx:]
        
        # Create tokenized files
        train_path = self.create_training_data(train_texts, "korean_train.bin")
        val_path = self.create_training_data(val_texts, "korean_val.bin")
        
        return train_path, val_path
        

def main():
    """Main function to download and prepare Korean datasets"""
    print("🇰🇷 Korean Dataset Downloader for Brain-Inspired GPT")
    print("=" * 60)
    
    downloader = KoreanDatasetDownloader()
    
    # 1. Try to download datasets
    print("\n1️⃣ Attempting to download datasets...")
    
    # Wikipedia (might be large, so optional)
    # wiki_path = downloader.download_korean_wikipedia()
    
    # KLUE (requires datasets library)
    # klue_data = downloader.download_klue_datasets()
    
    # KorQuAD (requires datasets library)
    # korquad_data = downloader.download_korquad()
    
    # Korpora (requires Korpora library)
    # korpora_data = downloader.download_korpora_datasets()
    
    # 2. For now, create a combined dataset with available data
    print("\n2️⃣ Creating combined training dataset...")
    train_path, val_path = downloader.prepare_combined_dataset()
    
    print("\n" + "="*60)
    print("✅ Dataset preparation complete!")
    print(f"   Training data: {train_path}")
    print(f"   Validation data: {val_path}")
    
    print("\n📚 To download more datasets, install:")
    print("   pip install datasets  # For KLUE, KorQuAD")
    print("   pip install Korpora   # For Korpora datasets")
    
    print("\n🚀 To train with Korean data:")
    print("   uv run brain_gpt/training/train_simple.py")
    

if __name__ == "__main__":
    main()