"""
Complete setup script for BERT News Classifier project
Run this ONCE to create the entire project structure
"""

import os
import sys
from pathlib import Path

def create_directory_structure(base_path="bert-news-classifier"):
    """Create all necessary directories"""
    
    directories = [
        f"{base_path}/.github/workflows",
        f"{base_path}/src/api/routes",
        f"{base_path}/src/api/middleware",
        f"{base_path}/src/models",
        f"{base_path}/src/database",
        f"{base_path}/src/utils",
        f"{base_path}/src/config",
        f"{base_path}/models/bert_weighted_model",
        f"{base_path}/tests",
        f"{base_path}/scripts",
        f"{base_path}/infra/terraform",
        f"{base_path}/infra/kubernetes",
        f"{base_path}/docker",
        f"{base_path}/data/raw",
        f"{base_path}/data/processed",
        f"{base_path}/notebooks",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_init_files(base_path="bert-news-classifier"):
    """Create __init__.py files"""
    
    init_files = [
        f"{base_path}/src/__init__.py",
        f"{base_path}/src/api/__init__.py",
        f"{base_path}/src/api/routes/__init__.py",
        f"{base_path}/src/api/middleware/__init__.py",
        f"{base_path}/src/models/__init__.py",
        f"{base_path}/src/database/__init__.py",
        f"{base_path}/src/utils/__init__.py",
        f"{base_path}/src/config/__init__.py",
        f"{base_path}/tests/__init__.py",
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")

if __name__ == "__main__":
    base_path = "bert-news-classifier"
    print("🚀 Creating BERT News Classifier Project Structure...\n")
    create_directory_structure(base_path)
    create_init_files(base_path)
    print("\n✅ Project structure created successfully!")