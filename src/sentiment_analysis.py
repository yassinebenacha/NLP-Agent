"""
Sentiment Analysis module using multiple approaches
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Traditional ML imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Deep learning imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from transformers import TrainingArguments, Trainer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Some features will be limited.")

# TextBlob for simple sentiment
from textblob import TextBlob

from config import SENTIMENT_CONFIG


class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis class supporting multiple approaches
    """
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # Initialize transformer model if available
        if TRANSFORMERS_AVAILABLE:
            self._init_transformer_model()
    
    def _init_transformer_model(self):
        """Initialize pre-trained transformer model for sentiment analysis"""
        try:
            model_name = SENTIMENT_CONFIG["model_name"]
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.transformer_model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"Loaded transformer model: {model_name}")
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            self.sentiment_pipeline = None
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get sentiment using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Convert polarity to sentiment categories
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'confidence': abs(polarity)
        }
    
    def transformer_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Get sentiment using transformer model
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment and confidence
        """
        if not self.sentiment_pipeline:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        try:
            # Truncate text if too long
            max_length = SENTIMENT_CONFIG["max_length"]
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.sentiment_pipeline(text)[0]
            
            # Map labels to standard format
            label_map = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive'
            }
            
            sentiment = label_map.get(result['label'], result['label'].lower())
            
            return {
                'sentiment': sentiment,
                'confidence': result['score']
            }
        except Exception as e:
            print(f"Error in transformer sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    text_column: str, 
                    label_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for traditional ML models
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Initialize vectorizer if not exists
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True
            )
            X = self.vectorizer.fit_transform(df[text_column])
        else:
            X = self.vectorizer.transform(df[text_column])
        
        # Convert labels to numeric
        y = df[label_column].map(self.label_mapping)
        
        return X, y
    
    def train_traditional_models(self, 
                               df: pd.DataFrame, 
                               text_column: str, 
                               label_column: str,
                               test_size: float = 0.2) -> Dict[str, Dict]:
        """
        Train traditional ML models for sentiment analysis
        
        Args:
            df: Training DataFrame
            text_column: Name of text column
            label_column: Name of label column
            test_size: Proportion of test data
            
        Returns:
            Dictionary with model results
        """
        # Prepare data
        X, y = self.prepare_data(df, text_column, label_column)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(random_state=42, probability=True),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'classification_report': report,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Store best model
            self.models[name] = model
            
            print(f"{name} accuracy: {accuracy:.4f}")
        
        return results
    
    def predict_sentiment(self, 
                         text: Union[str, List[str]], 
                         method: str = 'transformer') -> Union[Dict, List[Dict]]:
        """
        Predict sentiment for text(s)
        
        Args:
            text: Input text or list of texts
            method: Method to use ('transformer', 'textblob', or model name)
            
        Returns:
            Sentiment prediction(s)
        """
        if isinstance(text, str):
            texts = [text]
            single_text = True
        else:
            texts = text
            single_text = False
        
        results = []
        
        for txt in texts:
            if method == 'transformer':
                result = self.transformer_sentiment(txt)
            elif method == 'textblob':
                result = self.textblob_sentiment(txt)
            elif method in self.models:
                # Use traditional ML model
                if self.vectorizer is None:
                    result = {'sentiment': 'neutral', 'confidence': 0.0}
                else:
                    X = self.vectorizer.transform([txt])
                    pred = self.models[method].predict(X)[0]
                    proba = self.models[method].predict_proba(X)[0] if hasattr(self.models[method], 'predict_proba') else [0.33, 0.33, 0.34]
                    
                    result = {
                        'sentiment': self.reverse_label_mapping[pred],
                        'confidence': max(proba)
                    }
            else:
                result = {'sentiment': 'neutral', 'confidence': 0.0}
            
            results.append(result)
        
        return results[0] if single_text else results
    
    def analyze_sentiment_distribution(self, 
                                     df: pd.DataFrame, 
                                     text_column: str,
                                     method: str = 'transformer') -> pd.DataFrame:
        """
        Analyze sentiment distribution in a dataset
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            method: Method to use for sentiment analysis
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for idx, text in df[text_column].items():
            sentiment_result = self.predict_sentiment(text, method=method)
            results.append({
                'index': idx,
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence']
            })
        
        sentiment_df = pd.DataFrame(results)
        
        # Add to original dataframe
        result_df = df.copy()
        result_df['predicted_sentiment'] = sentiment_df['sentiment']
        result_df['sentiment_confidence'] = sentiment_df['confidence']
        
        return result_df
    
    def compare_methods(self, 
                       texts: List[str]) -> pd.DataFrame:
        """
        Compare different sentiment analysis methods
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame comparing different methods
        """
        results = []
        
        for i, text in enumerate(texts):
            result = {'text': text[:100] + '...' if len(text) > 100 else text}
            
            # TextBlob
            textblob_result = self.textblob_sentiment(text)
            result['textblob_sentiment'] = textblob_result['sentiment']
            result['textblob_confidence'] = textblob_result['confidence']
            
            # Transformer
            if TRANSFORMERS_AVAILABLE:
                transformer_result = self.transformer_sentiment(text)
                result['transformer_sentiment'] = transformer_result['sentiment']
                result['transformer_confidence'] = transformer_result['confidence']
            
            # Traditional models
            for model_name in self.models:
                model_result = self.predict_sentiment(text, method=model_name)
                result[f'{model_name}_sentiment'] = model_result['sentiment']
                result[f'{model_name}_confidence'] = model_result['confidence']
            
            results.append(result)
        
        return pd.DataFrame(results)
