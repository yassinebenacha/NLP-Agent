"""
Configuration settings for the NLP Analysis Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
RESULTS_DIR = PROJECT_ROOT / "results"

# Model configurations
SENTIMENT_CONFIG = {
    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "max_length": 512,
    "batch_size": 16,
    "device": "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"
}

TOPIC_MODELING_CONFIG = {
    "bertopic": {
        "min_topic_size": 10,
        "n_gram_range": (1, 2),
        "calculate_probabilities": True
    },
    "lda": {
        "num_topics": 10,
        "passes": 10,
        "alpha": "auto",
        "eta": "auto"
    }
}

NER_CONFIG = {
    "spacy_model": "en_core_web_sm",
    "custom_entities": ["PERSON", "ORG", "GPE", "MONEY", "DATE"]
}

SUMMARIZATION_CONFIG = {
    "extractive": {
        "model_name": "facebook/bart-large-cnn",
        "max_length": 150,
        "min_length": 50,
        "num_sentences": 3
    },
    "abstractive": {
        "model_name": "t5-small",
        "max_length": 200,
        "min_length": 50
    }
}

# Visualization settings
PLOT_CONFIG = {
    "style": "seaborn-v0_8",
    "figsize": (12, 8),
    "dpi": 300,
    "color_palette": "viridis"
}

# Data processing settings
PREPROCESSING_CONFIG = {
    "remove_stopwords": True,
    "remove_punctuation": True,
    "lowercase": True,
    "remove_numbers": False,
    "min_word_length": 2,
    "max_word_length": 50
}

# Evaluation metrics
EVALUATION_CONFIG = {
    "sentiment": ["accuracy", "precision", "recall", "f1"],
    "topic_modeling": ["coherence", "perplexity"],
    "summarization": ["rouge-1", "rouge-2", "rouge-l", "bert-score"],
    "ner": ["precision", "recall", "f1"]
}

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, 
                  MODELS_DIR, VISUALIZATIONS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
