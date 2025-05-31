"""
Data preprocessing utilities for NLP tasks
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Optional, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for NLP tasks
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 remove_numbers: bool = False,
                 min_word_length: int = 2,
                 max_word_length: int = 50,
                 language: str = 'english'):
        """
        Initialize the text preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert to lowercase
            remove_numbers: Whether to remove numbers
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
            language: Language for stopwords
        """
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.language = language
        
        # Initialize components
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(language))
        
        self.lemmatizer = WordNetLemmatizer()
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Some features may not work.")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Filter by word length
        tokens = [token for token in tokens 
                 if self.min_word_length <= len(token) <= self.max_word_length]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, 
                           df: pd.DataFrame, 
                           text_column: str,
                           new_column: str = 'processed_text') -> pd.DataFrame:
        """
        Preprocess text in a pandas DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text
            new_column: Name of the new column for processed text
            
        Returns:
            DataFrame with processed text column
        """
        df = df.copy()
        df[new_column] = df[text_column].apply(self.preprocess_text)
        return df
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)
    
    def get_word_frequency(self, texts: List[str], top_n: int = 20) -> dict:
        """
        Get word frequency from a list of texts
        
        Args:
            texts: List of texts
            top_n: Number of top words to return
            
        Returns:
            Dictionary of word frequencies
        """
        from collections import Counter
        
        all_words = []
        for text in texts:
            processed_text = self.preprocess_text(text)
            all_words.extend(processed_text.split())
        
        word_freq = Counter(all_words)
        return dict(word_freq.most_common(top_n))


def create_sample_dataset(size: int = 1000) -> pd.DataFrame:
    """
    Create a sample dataset for testing NLP models
    
    Args:
        size: Number of samples to generate
        
    Returns:
        DataFrame with sample text data
    """
    # Sample texts for different categories
    positive_texts = [
        "I love this product! It's amazing and works perfectly.",
        "Great service and excellent quality. Highly recommended!",
        "This is the best purchase I've ever made. So happy!",
        "Outstanding performance and great value for money.",
        "Fantastic experience! Will definitely buy again."
    ]
    
    negative_texts = [
        "This product is terrible. Complete waste of money.",
        "Poor quality and bad customer service. Very disappointed.",
        "Worst experience ever. Would not recommend to anyone.",
        "Broken after one day. Completely useless.",
        "Overpriced and underdelivered. Very unsatisfied."
    ]
    
    neutral_texts = [
        "The product arrived on time and matches the description.",
        "Standard quality for the price. Nothing special.",
        "It works as expected. No complaints, no praise.",
        "Average product with typical features.",
        "Delivered as promised. Meets basic requirements."
    ]
    
    # Generate random samples
    np.random.seed(42)
    data = []
    
    for i in range(size):
        sentiment = np.random.choice(['positive', 'negative', 'neutral'], 
                                   p=[0.4, 0.3, 0.3])
        
        if sentiment == 'positive':
            text = np.random.choice(positive_texts)
        elif sentiment == 'negative':
            text = np.random.choice(negative_texts)
        else:
            text = np.random.choice(neutral_texts)
        
        # Add some variation
        text = text + f" Sample {i+1}."
        
        data.append({
            'id': i+1,
            'text': text,
            'sentiment': sentiment,
            'length': len(text),
            'word_count': len(text.split())
        })
    
    return pd.DataFrame(data)
