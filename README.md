# BERT News Classifier - End-to-End ML System Design

## 📋 Project Overview

Production-ready news article classification system using BERT fine-tuning. Implements data-centric ML principles with systematic error analysis and temporal validation.

**Performance:**
- Week 1 (Baseline): 78% accuracy
- Week 2 (Data Engineering): 82% accuracy  
- Week 3 (TF-IDF + Features): 85% accuracy
- Week 4 (BERT Fine-tuning): **90%+ accuracy**

## 🏗️ Project Structure

```
bert-news-classifier/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI server
│   │   └── routes.py        # Prediction endpoints
│   ├── models/
│   │   ├── bert_model.py    # BERT wrapper
│   │   └── preprocessing.py # Text cleaning
│   └── database/
│       ├── postgres_client.py
│       └── redis_client.py
├── data/
│   ├── labelled_newscatcher_dataset_CLEANED.csv
│   ├── week2_train_FIXED.csv
│   ├── week2_val_FIXED.csv
│   └── week2_test_FIXED.csv
├── notebooks/
│   ├── MLSystemDesign.ipynb  # Main notebook
│   ├── 01_temporal_distribution_analysis.png
│   ├── 02_domain_topic_entropy_analysis.png
│   └── ...analysis outputs
├── models/
│   └── kaggle_models/        # Pre-trained models
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── requirements.txt
├── docker-compose.yml
├── README.md
└── setup_project.py
```

## 🎯 Key Findings (Data-Centric AI)

### 1. **Temporal Distribution**
- ✅ Low entropy across years (robust to time shifts)
- 📊 Balanced temporal splits for validation

### 2. **Domain Leakage Risk**
- 🔴 HIGH-risk domains detected (Bloomberg, Reuters)
- 📌 Created out-of-domain test set: `week2_ood_test.csv`

### 3. **Label Quality**
- ✅ <3% label overlap (excellent clarity)
- 📋 Validated 50 samples per topic: `week2_validation_sample_50.csv`

### 4. **Data Cleaning**
- Removed outliers: titles <3 or >30 words (0.2% loss)
- Cleaned dataset: `labelled_newscatcher_dataset_CLEANED.csv`

## 🚀 Quick Start

### **Installation**

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/bert-news-classifier.git
cd bert-news-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Run FastAPI Server**

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

Visit: http://localhost:8000/docs

### **Run Batch Prediction**

```bash
python scripts/batch_test.py
```

### **Run Jupyter Notebook**

```bash
jupyter notebook MLSystemDesign.ipynb
```

## 📊 Model Comparison

| Model | Approach | Val Acc | Test Acc | Training |
|-------|----------|---------|----------|----------|
| **Baseline** | Embedding + GlobalAvgPool | 78% | 78% | 10 epochs |
| **TF-IDF** | Logistic Regression | 76% | 76% | < 1s |
| **TF-IDF + Features** | + Text/Temporal features | 79% | 79% | < 2s |
| **BERT** | DistilBERT fine-tuned | **91%** | **90%** | 2 epochs |

## 🔍 Error Analysis

Per-topic accuracy (BERT):
- SPORTS: 94% ✅ (easy, low vocabulary richness)
- BUSINESS: 92% ✅ (strong domain signals)
- ENTERTAINMENT: 88% (temporal drift)
- WORLD: 87%
- TECHNOLOGY: 86% (overlaps with SCIENCE)
- SCIENCE: 81% (hardest, high diversity)
- HEALTH: 79% (COVID temporal shift)
- NATION: 77% (confusion with WORLD)

👉 See `week3_confusion_matrix.png` for details

## 🗂️ Data Files (Not Committed - Download Separately)

Due to size limits, data files are not in repo. Download from:

1. **Original Dataset** (large):
   ```
   data/labelled_newscatcher_dataset.csv (500MB+)
   ```

2. **Week 2 Cleaned Splits** (recommended, 80MB):
   ```
   week2_train_FIXED.csv
   week2_val_FIXED.csv
   week2_test_FIXED.csv
   ```

3. **Pre-trained Models** (not committed):
   ```
   models/kaggle_models/bert_weighted_model.zip
   week4_bert_final/
   ```

## 🐳 Docker Deployment

```bash
# Build image
docker-compose build

# Run containers
docker-compose up

# API available at http://localhost:8000
# PostgreSQL at localhost:5433
# Redis at localhost:6379
```

## 📈 Model Metrics

### Validation Set (Week 4)
- **Accuracy:** 91%
- **Macro F1:** 0.88
- **Weighted F1:** 0.91
- **Top-2 Accuracy:** 97%

### Out-of-Domain Test
- **OOD Accuracy:** 87% (-3% generalization gap)
- **Domain-specific domains:** -5% to -8%

## 🔄 Training Pipeline

```bash
# Week 1: Data Analysis & Cleaning
python -c "
import pandas as pd
df = pd.read_csv('data/labelled_newscatcher_dataset.csv', sep=';')
# → labelled_newscatcher_dataset_CLEANED.csv (0.2% loss)
"

# Week 2: Stratified Splits
python setup_project.py
# → week2_train_FIXED.csv, week2_val_FIXED.csv, week2_test_FIXED.csv

# Week 3: Feature Engineering + TF-IDF
jupyter notebook MLSystemDesign.ipynb
# Cell: "SETUP & LOAD Week 2 SPLITS" → "ADD CUSTOM FEATURES"
# Result: 85% accuracy

# Week 4: BERT Fine-tuning
jupyter notebook MLSystemDesign.ipynb
# Cell: "BERT MODEL + TRAINING SETUP"
# Result: 90%+ accuracy
```

## 🎓 Learning Outcomes

This project demonstrates:

1. **Data-Centric AI Principles**
   - Quality over quantity
   - Systematic error analysis
   - Label ambiguity detection

2. **Production ML Systems**
   - FastAPI inference server
   - Async batch processing
   - Redis caching + PostgreSQL logging
   - Docker containerization

3. **Rigorous Evaluation**
   - Time-stratified K-fold splits
   - Out-of-domain generalization testing
   - Per-class performance analysis

4. **Feature Engineering**
   - Text vectorization (TF-IDF, BERT embeddings)
   - Special character preservation
   - Temporal features (news cycle patterns)

## 📝 Configuration

See `setup_project.py` for:
- Dataset paths
- Model hyperparameters
- Database credentials (use `.env` file!)

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Use CPU
export CUDA_VISIBLE_DEVICES=""
```

### Redis Connection Error
```bash
# Start Redis
redis-server
# Or in Docker
docker run -d -p 6379:6379 redis:latest
```

### PostgreSQL Connection Error
```bash
# Docker Compose starts it automatically
docker-compose up -d
```

## 📚 References

- [Data-Centric AI](https://www.deeplearning.ai/short-courses/data-centric-ai/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://huggingface.co/transformers/)



---

**⭐ If this helps, please star the repo!**

Last updated: 2025-01-15