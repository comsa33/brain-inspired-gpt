#!/usr/bin/env python3
"""
Prepare Korean datasets from Hugging Face for Brain-Inspired GPT
Uses freely available Korean datasets without requiring registration
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.multilingual_tokenizer import MultilingualBrainTokenizer


class KoreanHFDatasetPreparer:
    """Prepare Korean datasets from Hugging Face"""
    
    def __init__(self, data_dir: str = "data/korean_hf"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = MultilingualBrainTokenizer()
        
    def load_klue_datasets(self) -> Dict[str, List[str]]:
        """Load KLUE datasets and extract text"""
        print("\n📥 Loading KLUE datasets from Hugging Face...")
        
        try:
            from datasets import load_dataset
            
            all_texts = []
            
            # KLUE-TC (Topic Classification) - contains news articles
            try:
                print("  Loading KLUE-TC (News articles)...")
                dataset = load_dataset("klue", "ynat")
                for split in ['train', 'validation']:
                    for item in dataset[split]:
                        title = item.get('title', '')
                        # YNAT dataset has news titles
                        if title:
                            all_texts.append(title)
                print(f"  ✅ Loaded {len(all_texts)} news titles")
            except Exception as e:
                print(f"  ⚠️  Failed to load KLUE-TC: {e}")
                
            # KLUE-STS (Semantic Textual Similarity) - sentence pairs
            try:
                print("  Loading KLUE-STS (Sentence pairs)...")
                sts_texts = []
                dataset = load_dataset("klue", "sts")
                for split in ['train', 'validation']:
                    for item in dataset[split]:
                        sts_texts.append(item['sentence1'])
                        sts_texts.append(item['sentence2'])
                all_texts.extend(sts_texts)
                print(f"  ✅ Loaded {len(sts_texts)} sentences from STS")
            except Exception as e:
                print(f"  ⚠️  Failed to load KLUE-STS: {e}")
                
            # KLUE-NLI (Natural Language Inference)
            try:
                print("  Loading KLUE-NLI (Premise-hypothesis pairs)...")
                nli_texts = []
                dataset = load_dataset("klue", "nli")
                for split in ['train', 'validation']:
                    for item in dataset[split]:
                        nli_texts.append(item['premise'])
                        nli_texts.append(item['hypothesis'])
                all_texts.extend(nli_texts)
                print(f"  ✅ Loaded {len(nli_texts)} sentences from NLI")
            except Exception as e:
                print(f"  ⚠️  Failed to load KLUE-NLI: {e}")
                
            return {'klue': all_texts}
            
        except ImportError:
            print("❌ datasets library not found. Install with: pip install datasets")
            return {}
            
    def load_korquad_dataset(self) -> Dict[str, List[str]]:
        """Load KorQuAD dataset"""
        print("\n📥 Loading KorQuAD dataset...")
        
        try:
            from datasets import load_dataset
            
            all_texts = []
            
            # Load KorQuAD v1.0
            dataset = load_dataset("squad_kor_v1")
            
            for split in ['train', 'validation']:
                for item in dataset[split]:
                    # Extract context (passage)
                    context = item.get('context', '')
                    if context:
                        all_texts.append(context)
                        
                    # Also add questions and answers for variety
                    question = item.get('question', '')
                    if question:
                        all_texts.append(question)
                        
                    answers = item.get('answers', {})
                    if answers and 'text' in answers:
                        for answer in answers['text']:
                            if answer:
                                all_texts.append(answer)
                                
            print(f"✅ Loaded {len(all_texts)} texts from KorQuAD")
            return {'korquad': all_texts}
            
        except Exception as e:
            print(f"❌ Failed to load KorQuAD: {e}")
            return {}
            
    def load_nsmc_dataset(self) -> Dict[str, List[str]]:
        """Load NSMC (Naver Sentiment Movie Corpus)"""
        print("\n📥 Loading NSMC dataset...")
        
        try:
            from datasets import load_dataset
            
            all_texts = []
            
            # Load NSMC
            dataset = load_dataset("nsmc")
            
            for split in ['train', 'test']:
                for item in dataset[split]:
                    document = item.get('document', '')
                    if document:
                        all_texts.append(document)
                        
            print(f"✅ Loaded {len(all_texts)} movie reviews from NSMC")
            return {'nsmc': all_texts}
            
        except Exception as e:
            print(f"❌ Failed to load NSMC: {e}")
            return {}
            
    def load_korean_parallel_corpora(self) -> Dict[str, List[str]]:
        """Load Korean parallel corpora"""
        print("\n📥 Loading Korean parallel corpora...")
        
        try:
            from datasets import load_dataset
            
            all_texts = []
            
            # Try to load Korean-English parallel data
            try:
                dataset = load_dataset("Helsinki-NLP/opus-100", "en-ko")
                for split in ['train', 'validation', 'test']:
                    if split in dataset:
                        for item in dataset[split]:
                            ko_text = item['translation']['ko']
                            if ko_text:
                                all_texts.append(ko_text)
                print(f"✅ Loaded {len(all_texts)} Korean texts from parallel corpus")
            except Exception as e:
                print(f"⚠️  Failed to load parallel corpus: {e}")
                
            return {'parallel': all_texts}
            
        except Exception as e:
            print(f"❌ Failed to load parallel corpora: {e}")
            return {}
            
    def create_high_quality_samples(self) -> List[str]:
        """Create high-quality Korean text samples for training"""
        print("\n📝 Creating high-quality Korean samples...")
        
        samples = [
            # Technology and AI
            "인공지능 기술의 발전은 우리 사회에 혁명적인 변화를 가져오고 있습니다. 특히 자연어 처리 분야에서의 발전은 인간과 기계 간의 소통을 더욱 원활하게 만들고 있습니다.",
            "딥러닝 알고리즘의 발전으로 컴퓨터 비전, 음성 인식, 자연어 이해 등 다양한 분야에서 인간 수준의 성능을 달성하고 있습니다.",
            "한국은 5G 네트워크 기술을 세계 최초로 상용화하여 정보통신 분야의 선도국가로 자리매김했습니다.",
            
            # Korean culture and history
            "한글은 세종대왕이 1443년에 창제한 과학적이고 체계적인 문자 체계로, 유네스코 세계기록유산에 등재되어 있습니다.",
            "한국의 전통 음식인 김치는 발효 과학의 정수를 보여주는 건강식품으로, 세계적으로 그 가치를 인정받고 있습니다.",
            "K-pop과 K-drama로 대표되는 한류는 전 세계에 한국 문화를 알리는 중요한 역할을 하고 있습니다.",
            
            # Science and education
            "한국의 교육 시스템은 높은 학업 성취도로 유명하며, 특히 수학과 과학 분야에서 세계적인 경쟁력을 보여주고 있습니다.",
            "생명공학 기술의 발전으로 난치병 치료와 신약 개발에 새로운 가능성이 열리고 있습니다.",
            "기후 변화에 대응하기 위한 신재생 에너지 기술 개발이 전 세계적으로 활발히 진행되고 있습니다.",
            
            # Economy and society
            "한국 경제는 제조업과 서비스업을 중심으로 빠르게 성장하여 선진국 대열에 합류했습니다.",
            "스타트업 생태계의 활성화로 혁신적인 기업들이 계속해서 등장하고 있으며, 유니콘 기업도 증가하고 있습니다.",
            "고령화 사회로의 진입에 따라 복지 정책과 의료 시스템의 개선이 중요한 과제로 떠오르고 있습니다.",
            
            # Literature and philosophy
            "한국 문학은 고전 문학부터 현대 문학까지 다양한 주제와 형식으로 인간의 삶과 정서를 표현해 왔습니다.",
            "동양 철학의 전통 속에서 한국은 유교, 불교, 도교의 사상을 독특하게 융합하여 발전시켰습니다.",
            "현대 한국 사회는 전통과 현대, 동양과 서양의 가치관이 공존하는 다원적 문화를 형성하고 있습니다.",
        ]
        
        # Add more variations
        extended_samples = []
        for sample in samples:
            extended_samples.append(sample)
            # Add slight variations
            extended_samples.append(sample + " 이러한 발전은 앞으로도 계속될 것으로 예상됩니다.")
            extended_samples.append("최근 연구에 따르면, " + sample)
            
        print(f"✅ Created {len(extended_samples)} high-quality samples")
        return extended_samples
        
    def prepare_training_data(self, texts: List[str], output_name: str) -> Path:
        """Convert texts to tokenized binary format"""
        print(f"\n🔧 Preparing {output_name}...")
        
        # Tokenize all texts
        all_tokens = []
        
        for text in tqdm(texts, desc="Tokenizing"):
            if text and len(text.strip()) > 0:
                tokens = self.tokenizer.encode(text, language='ko')
                if tokens:
                    all_tokens.extend(tokens)
                    # Add a separator token
                    all_tokens.append(0)  # Use 0 as separator
                    
        # Convert to numpy array - use uint32 to avoid overflow
        # Filter out tokens that are out of range
        valid_tokens = [t for t in all_tokens if 0 <= t < 65536]
        tokens_array = np.array(valid_tokens, dtype=np.uint16)
        
        # Save
        output_path = self.data_dir / f"{output_name}.bin"
        tokens_array.tofile(output_path)
        
        print(f"✅ Saved {len(tokens_array):,} tokens to {output_path}")
        return output_path
        
    def create_comprehensive_dataset(self):
        """Create a comprehensive Korean dataset from all available sources"""
        print("\n🔧 Creating comprehensive Korean dataset...")
        
        all_texts = []
        
        # 1. Load datasets from Hugging Face
        datasets = {}
        datasets.update(self.load_klue_datasets())
        datasets.update(self.load_korquad_dataset())
        datasets.update(self.load_nsmc_dataset())
        datasets.update(self.load_korean_parallel_corpora())
        
        # 2. Add high-quality samples
        high_quality_samples = self.create_high_quality_samples()
        all_texts.extend(high_quality_samples * 10)  # Repeat for more data
        
        # 3. Combine all dataset texts
        for dataset_name, texts in datasets.items():
            print(f"\nAdding {len(texts)} texts from {dataset_name}")
            all_texts.extend(texts)
            
        # Remove duplicates and empty texts
        unique_texts = list(set(text.strip() for text in all_texts if text and len(text.strip()) > 0))
        print(f"\nTotal unique texts: {len(unique_texts)}")
        
        # Shuffle
        import random
        random.shuffle(unique_texts)
        
        # Split into train and validation
        split_idx = int(len(unique_texts) * 0.95)
        train_texts = unique_texts[:split_idx]
        val_texts = unique_texts[split_idx:]
        
        print(f"\nTrain texts: {len(train_texts)}")
        print(f"Validation texts: {len(val_texts)}")
        
        # Create binary files
        train_path = self.prepare_training_data(train_texts, "korean_hf_train")
        val_path = self.prepare_training_data(val_texts, "korean_hf_val")
        
        return train_path, val_path
        

def main():
    """Main function"""
    print("🇰🇷 Korean Dataset Preparation from Hugging Face")
    print("=" * 60)
    
    preparer = KoreanHFDatasetPreparer()
    
    # Check if datasets library is available
    try:
        import datasets
        print("✅ datasets library is available")
        
        # Create comprehensive dataset
        train_path, val_path = preparer.create_comprehensive_dataset()
        
        print("\n" + "="*60)
        print("✅ Dataset preparation complete!")
        print(f"   Training data: {train_path}")
        print(f"   Validation data: {val_path}")
        
        # Update training script to use new data
        print("\n🚀 To train with this Korean dataset:")
        print("   1. Update data paths in train_brain_gpt.py")
        print("   2. Run: uv run brain_gpt/training/train_simple.py")
        
    except ImportError:
        print("❌ datasets library not found!")
        print("\nTo use Hugging Face Korean datasets:")
        print("  1. Install: pip install datasets")
        print("  2. Run this script again")
        print("\nAlternatively, use the basic Korean dataset:")
        print("  uv run brain_gpt/training/download_korean_datasets.py")
        

if __name__ == "__main__":
    main()