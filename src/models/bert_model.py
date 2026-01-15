"""
BERT News Classifier - COMPLETE FIXED VERSION
Supports: Kaggle ZIP, label mapping, device auto-detection
"""

import torch
import numpy as np
import zipfile
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)


class BERTNewsClassifier:
    """BERT-based news classifier with full production support"""
    
    def __init__(
        self, 
        model_path: str = None, 
        kaggle_zip_path: str = None,
        device: str = None,
        label_encoder_path: str = None
    ):
        """
        Initialize BERT classifier
        
        Args:
            model_path: Path to model directory
            kaggle_zip_path: Path to Kaggle ZIP file (extracts automatically)
            device: 'cuda' or 'cpu' (auto-detect if None)
            label_encoder_path: Path to label mapping JSON
        """
        
        # Device detection
        self.device = self._detect_device(device)
        logger.info(f"🎯 Using device: {self.device}")
        
        # Extract Kaggle ZIP if provided
        if kaggle_zip_path and os.path.exists(kaggle_zip_path):
            logger.info(f"📦 Extracting Kaggle model: {kaggle_zip_path}")
            model_path = self._extract_kaggle_zip(kaggle_zip_path)
        
        self.model_path = model_path
        if not self.model_path:
            raise ValueError("Must provide model_path or kaggle_zip_path")
        
        # Load model and tokenizer
        self._load_model()
        
        # Load label mapping (CRITICAL for correct predictions!)
        self._load_label_mapping(label_encoder_path)
        
        logger.info(f"✅ Model ready with {len(self.classes)} classes")
    
    @staticmethod
    def _detect_device(device: str = None) -> str:
        """Auto-detect device if not specified"""
        if device:
            return device
        
        if torch.cuda.is_available():
            logger.info("🚀 CUDA available - using GPU")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🚀 Apple MPS available - using GPU")
            return "mps"
        else:
            logger.info("💻 Using CPU")
            return "cpu"
    
    def _extract_kaggle_zip(self, zip_path: str) -> str:
        """
        Extract Kaggle ZIP and find model directory
        
        Expected structure:
        your_model.zip
        ├── week4_bert_final/        ← Model files here
        │   ├── pytorch_model.bin
        │   ├── config.json
        │   ├── tokenizer_config.json
        │   ├── special_tokens_map.json
        │   └── vocab.txt
        """
        
        extract_dir = "./models/kaggle_extracted"
        Path(extract_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 Extracting ZIP: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        
        # Find model directory (contains pytorch_model.bin or config.json)
        model_dir = None
        for root, dirs, files in os.walk(extract_dir):
            if 'pytorch_model.bin' in files or 'config.json' in files:
                model_dir = root
                logger.info(f"✅ Found model at: {model_dir}")
                break
        
        if not model_dir:
            raise FileNotFoundError(
                f"❌ No pytorch_model.bin or config.json found in {zip_path}\n"
                f"Expected structure:\n"
                f"  your_model.zip/\n"
                f"  ├── week4_bert_final/\n"
                f"  │   ├── pytorch_model.bin\n"
                f"  │   ├── config.json\n"
                f"  │   └── tokenizer files..."
            )
        
        return model_dir
    
    def _load_model(self):
        """Load BERT model and tokenizer"""
        try:
            logger.info(f"🤖 Loading model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("✅ Tokenizer loaded")
            
            # Load model config to get num_labels
            with open(f"{self.model_path}/config.json") as f:
                config = json.load(f)
                num_labels = config.get('num_labels', 8)
            
            logger.info(f"📊 Model has {num_labels} labels")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=num_labels
            ).to(self.device)
            
            self.model.eval()
            logger.info(f"✅ Model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_label_mapping(self, label_encoder_path: str = None):
        """
        Load label mapping from saved encoder or infer from config
        
        Priority:
        1. Provided label_encoder_path
        2. label_mapping.json in model directory
        3. Infer from model config
        """
        
        label_mapping = None
        
        # Try provided path
        if label_encoder_path and os.path.exists(label_encoder_path):
            logger.info(f"📂 Loading labels from: {label_encoder_path}")
            with open(label_encoder_path) as f:
                label_mapping = json.load(f)
        
        # Try model directory
        model_label_path = f"{self.model_path}/label_mapping.json"
        if label_mapping is None and os.path.exists(model_label_path):
            logger.info(f"📂 Found label mapping in model dir")
            with open(model_label_path) as f:
                label_mapping = json.load(f)
        
        # Fallback: infer from config (if classes key exists)
        if label_mapping is None:
            try:
                with open(f"{self.model_path}/config.json") as f:
                    config = json.load(f)
                    # Check if id2label mapping exists in config
                    if 'id2label' in config:
                        self.classes = [
                            config['id2label'][str(i)] 
                            for i in range(config['num_labels'])
                        ]
                        logger.info(f"✅ Loaded {len(self.classes)} classes from config: {self.classes}")
                        return
            except:
                pass
        
        # If we got label_mapping from file
        if label_mapping is not None:
            if isinstance(label_mapping, list):
                self.classes = label_mapping
            elif isinstance(label_mapping, dict):
                # Handle both {'class1': 0, ...} and {'0': 'class1', ...}
                if all(isinstance(k, int) for k in label_mapping.keys()):
                    self.classes = [label_mapping[i] for i in range(len(label_mapping))]
                else:
                    self.classes = sorted(label_mapping.keys(), 
                                        key=lambda x: label_mapping[x])
            logger.info(f"✅ Loaded {len(self.classes)} classes: {self.classes}")
            return
        
        # Final fallback: use default 8 classes (original dataset)
        logger.warning("⚠️  No label mapping found - using default 8 classes")
        self.classes = [
            "BUSINESS", "ENTERTAINMENT", "HEALTH", "NATION",
            "SCIENCE", "SPORTS", "TECHNOLOGY", "WORLD"
        ]
    
    def predict(self, text: str) -> Dict:
        """
        Predict topic for single text
        
        Returns:
            {
                'topic': str,
                'confidence': float,
                'probabilities': {class: float, ...}
            }
        """
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Get top prediction
            predicted_idx = np.argmax(probabilities)
            predicted_topic = self.classes[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            return {
                'topic': predicted_topic,
                'confidence': confidence,
                'probabilities': {
                    cls: float(prob) 
                    for cls, prob in zip(self.classes, probabilities)
                }
            }
        
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Batch prediction for multiple texts
        
        Args:
            texts: List of article titles
        
        Returns:
            List of prediction dicts
        """
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Process each prediction
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            
            results = []
            for i, probs in enumerate(probabilities):
                predicted_idx = np.argmax(probs)
                predicted_topic = self.classes[predicted_idx]
                confidence = float(probs[predicted_idx])
                
                results.append({
                    'text': texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i],
                    'topic': predicted_topic,
                    'confidence': confidence
                })
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Batch prediction error: {e}")
            raise
    
    def predict_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-K predictions
        
        Args:
            text: Article title
            k: Number of top predictions
        
        Returns:
            List of (topic, confidence) tuples sorted by confidence
        """
        
        result = self.predict(text)
        probs = result['probabilities']
        
        top_k = sorted(
            probs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return top_k
    
    @property
    def num_labels(self) -> int:
        """Number of output classes"""
        return len(self.classes)


# ============================================
# UTILITY FUNCTION: Save label mapping
# ============================================

def save_label_mapping(classes: List[str], output_path: str):
    """Save label mapping for later use"""
    mapping = {str(i): cls for i, cls in enumerate(classes)}
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"✅ Saved label mapping to {output_path}")