#!/usr/bin/env python3
"""
Unified data download system for CortexGPT
Supports multiple datasets with various download methods
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Optional, Callable
from tqdm import tqdm
import random
from abc import ABC, abstractmethod

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DatasetDownloader(ABC):
    """Abstract base class for dataset downloaders"""
    
    def __init__(self, name: str, output_dir: Path):
        self.name = name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def download(self) -> bool:
        """Download the dataset. Return True if successful."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, str]:
        """Get dataset information"""
        pass


class GeneratedDataset(DatasetDownloader):
    """Generate synthetic datasets"""
    
    def __init__(self, name: str, output_dir: Path, generator_func: Callable, 
                 num_samples: int = 5000):
        super().__init__(name, output_dir)
        self.generator_func = generator_func
        self.num_samples = num_samples
    
    def download(self) -> bool:
        print(f"📝 Generating {self.name} dataset...")
        texts = self.generator_func(self.num_samples)
        
        output_file = self.output_dir / "data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in tqdm(texts, desc=f"Writing {self.name}"):
                json.dump({"text": text, "language": self.get_language()}, 
                         f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Generated {len(texts)} samples for {self.name}")
        return True
    
    def get_language(self) -> str:
        if "korean" in self.name.lower() or "klue" in self.name.lower():
            return "ko"
        return "en"
    
    def get_info(self) -> Dict[str, str]:
        return {
            "type": "generated",
            "samples": str(self.num_samples),
            "language": self.get_language()
        }


class HuggingFaceDataset(DatasetDownloader):
    """Download datasets from Hugging Face"""
    
    def __init__(self, name: str, output_dir: Path, hf_dataset: str, 
                 hf_config: Optional[str] = None, split: str = "train[:10000]"):
        super().__init__(name, output_dir)
        self.hf_dataset = hf_dataset
        self.hf_config = hf_config
        self.split = split
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
        except ImportError:
            print("❌ Installing datasets library...")
            os.system("uv pip install datasets")
            from datasets import load_dataset
        
        print(f"📥 Downloading {self.name} from Hugging Face...")
        
        try:
            if self.hf_config:
                dataset = load_dataset(self.hf_dataset, self.hf_config, split=self.split)
            else:
                dataset = load_dataset(self.hf_dataset, split=self.split)
            
            output_file = self.output_dir / "data.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in tqdm(dataset, desc=f"Processing {self.name}"):
                    text = item.get('text', item.get('content', ''))
                    if len(text) > 50:  # Skip very short texts
                        json.dump({"text": text, "language": self.get_language()}, 
                                 f, ensure_ascii=False)
                        f.write('\n')
            
            print(f"✅ Downloaded {self.name} successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {self.name}: {e}")
            return False
    
    def get_language(self) -> str:
        if "ko" in self.hf_dataset or "korean" in self.name.lower():
            return "ko"
        return "en"
    
    def get_info(self) -> Dict[str, str]:
        return {
            "type": "huggingface",
            "dataset": self.hf_dataset,
            "split": self.split,
            "language": self.get_language()
        }


# Generator functions for synthetic data
def generate_demo_texts(num_samples: int) -> List[str]:
    """Generate simple demo texts"""
    en_samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Python is a versatile programming language.",
        "Data science combines statistics and programming.",
        "Neural networks learn from examples."
    ]
    
    ko_samples = [
        "안녕하세요. 오늘 날씨가 좋습니다.",
        "인공지능은 미래의 핵심 기술입니다.",
        "한국어 자연어 처리는 중요한 연구 분야입니다.",
        "딥러닝은 놀라운 성과를 보여주고 있습니다.",
        "기계 학습으로 많은 문제를 해결할 수 있습니다."
    ]
    
    texts = []
    for i in range(num_samples):
        if i % 2 == 0:
            texts.append(random.choice(en_samples))
        else:
            texts.append(random.choice(ko_samples))
    return texts


def generate_english_texts(num_samples: int) -> List[str]:
    """Generate diverse English texts"""
    topics = [
        "artificial intelligence", "machine learning", "deep learning",
        "natural language processing", "computer vision", "robotics",
        "data science", "quantum computing", "blockchain", "cybersecurity",
        "cloud computing", "edge computing", "IoT", "5G technology",
        "biotechnology", "renewable energy", "space exploration"
    ]
    
    patterns = [
        "The field of {topic} has revolutionized modern technology.",
        "Researchers in {topic} are making breakthrough discoveries.",
        "Understanding {topic} is crucial for future innovations.",
        "Recent advances in {topic} show promising results.",
        "The application of {topic} extends across many industries."
    ]
    
    texts = []
    for _ in range(num_samples):
        pattern = random.choice(patterns)
        topic = random.choice(topics)
        text = pattern.format(topic=topic)
        
        # Add more sentences
        if random.random() > 0.5:
            text += f" This technology enables new possibilities in various domains."
            text += f" The impact on society will be significant in coming years."
        
        texts.append(text)
    
    return texts


def generate_korean_texts(num_samples: int) -> List[str]:
    """Generate diverse Korean texts"""
    topics = [
        "인공지능", "기계학습", "딥러닝", "자연어처리", "컴퓨터비전",
        "로봇공학", "데이터과학", "양자컴퓨팅", "블록체인", "사이버보안",
        "클라우드컴퓨팅", "엣지컴퓨팅", "사물인터넷", "5G기술",
        "생명공학", "신재생에너지", "우주탐사"
    ]
    
    patterns = [
        "{topic} 분야는 현대 기술을 혁신하고 있습니다.",
        "{topic} 연구자들은 획기적인 발견을 하고 있습니다.",
        "{topic}을(를) 이해하는 것은 미래 혁신에 중요합니다.",
        "{topic}의 최근 발전은 유망한 결과를 보여줍니다.",
        "{topic}의 응용은 많은 산업에 걸쳐 있습니다."
    ]
    
    texts = []
    for _ in range(num_samples):
        pattern = random.choice(patterns)
        topic = random.choice(topics)
        text = pattern.format(topic=topic)
        
        # Add more sentences
        if random.random() > 0.5:
            text += " 이 기술은 다양한 분야에서 새로운 가능성을 열어줍니다."
            text += " 앞으로 사회에 미칠 영향은 매우 클 것입니다."
        
        texts.append(text)
    
    return texts


# Dataset registry
DATASETS = {
    # Demo datasets
    "demo": GeneratedDataset(
        "demo", 
        Path("data/datasets/demo"),
        generate_demo_texts,
        1000
    ),
    
    # English datasets
    "english_small": GeneratedDataset(
        "english_small",
        Path("data/datasets/english_small"),
        generate_english_texts,
        5000
    ),
    
    "english_large": GeneratedDataset(
        "english_large",
        Path("data/datasets/english_large"),
        generate_english_texts,
        50000
    ),
    
    # Korean datasets
    "korean_small": GeneratedDataset(
        "korean_small",
        Path("data/datasets/korean_small"),
        generate_korean_texts,
        5000
    ),
    
    "korean_large": GeneratedDataset(
        "korean_large",
        Path("data/datasets/korean_large"),
        generate_korean_texts,
        50000
    ),
    
    # Hugging Face datasets
    "wikitext": HuggingFaceDataset(
        "wikitext",
        Path("data/datasets/wikitext"),
        "wikitext",
        "wikitext-103-raw-v1",
        "train[:10000]"
    ),
    
    "openwebtext": HuggingFaceDataset(
        "openwebtext",
        Path("data/datasets/openwebtext"),
        "Skylion007/openwebtext",
        None,
        "train[:10000]"
    ),
    
    "c4_en": HuggingFaceDataset(
        "c4_en",
        Path("data/datasets/c4_en"),
        "c4",
        "en",
        "train[:5000]"
    ),
    
    # Korean Hugging Face datasets
    "klue": HuggingFaceDataset(
        "klue",
        Path("data/datasets/klue"),
        "klue/klue",
        "tc",  # Text classification subset
        "train[:10000]"
    ),
}


def list_datasets():
    """List all available datasets"""
    print("\n📚 Available Datasets for CortexGPT\n")
    print(f"{'Name':<20} {'Type':<15} {'Language':<10} {'Status'}")
    print("-" * 60)
    
    for name, dataset in DATASETS.items():
        info = dataset.get_info()
        status = "✅ Downloaded" if (dataset.output_dir / "data.jsonl").exists() else "❌ Not downloaded"
        print(f"{name:<20} {info['type']:<15} {info['language']:<10} {status}")


def download_dataset(name: str) -> bool:
    """Download a specific dataset"""
    if name not in DATASETS:
        print(f"❌ Unknown dataset: {name}")
        print("   Use --list to see available datasets")
        return False
    
    dataset = DATASETS[name]
    return dataset.download()


def download_all_datasets(category: Optional[str] = None):
    """Download all datasets or datasets in a category"""
    if category:
        filtered = {k: v for k, v in DATASETS.items() 
                   if category in k or category in v.get_info().get('language', '')}
    else:
        filtered = DATASETS
    
    print(f"\n📥 Downloading {len(filtered)} datasets...\n")
    
    success = 0
    for name, dataset in filtered.items():
        if dataset.download():
            success += 1
        print()
    
    print(f"\n✅ Successfully downloaded {success}/{len(filtered)} datasets")


def main():
    parser = argparse.ArgumentParser(
        description="Unified data download system for CortexGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  %(prog)s --list
  
  # Download a specific dataset
  %(prog)s --dataset english_large
  
  # Download all English datasets
  %(prog)s --all --category english
  
  # Download all Korean datasets
  %(prog)s --all --category korean
  
  # Download all datasets
  %(prog)s --all
        """
    )
    
    parser.add_argument("--list", action="store_true",
                       help="List all available datasets")
    parser.add_argument("--dataset", type=str,
                       help="Download a specific dataset")
    parser.add_argument("--all", action="store_true",
                       help="Download all datasets")
    parser.add_argument("--category", type=str,
                       choices=["english", "korean", "demo"],
                       help="Filter datasets by category (use with --all)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.list, args.dataset, args.all]):
        parser.print_help()
        return
    
    print("🌐 CortexGPT Data Download System\n")
    
    if args.list:
        list_datasets()
    elif args.dataset:
        download_dataset(args.dataset)
    elif args.all:
        download_all_datasets(args.category)
    
    print("\n💡 To train with a dataset:")
    print("   uv run scripts/train_cortexgpt.py --dataset <dataset_name> --epochs 20")


if __name__ == "__main__":
    main()