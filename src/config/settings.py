import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_TITLE = "BERT News Classifier API"
    API_VERSION = "1.0.0"
    
    MODEL_PATH = "./models/bert_weighted_model"
    KAGGLE_MODEL_ZIP = os.getenv("KAGGLE_MODEL_ZIP", None)
    MAX_SEQ_LENGTH = 128
    
    API_KEY = os.getenv("API_KEY", "test-key")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEVICE = os.getenv("DEVICE", "cpu")
    
    TOPICS = [
        "ENTERTAINMENT", "HEALTH", "TECHNOLOGY", "WORLD",
        "BUSINESS", "SPORTS", "NATION", "SCIENCE"
    ]

settings = Settings()