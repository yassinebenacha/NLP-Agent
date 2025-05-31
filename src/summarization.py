"""
Text Summarization module using extractive and abstractive methods
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Basic text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx

# Transformers for advanced summarization
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Some summarization features will be limited.")

# Sklearn for TF-IDF based summarization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SUMMARIZATION_CONFIG


class TextSummarizer:
    """
    Comprehensive text summarization class supporting multiple approaches
    """
    
    def __init__(self):
        self.models = {}
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize transformer models if available
        if TRANSFORMERS_AVAILABLE:
            self._init_transformer_models()
    
    def _init_transformer_models(self):
        """Initialize transformer models for summarization"""
        try:
            # Extractive summarization model
            extractive_model = SUMMARIZATION_CONFIG["extractive"]["model_name"]
            self.extractive_pipeline = pipeline(
                "summarization",
                model=extractive_model,
                tokenizer=extractive_model
            )
            
            # Abstractive summarization model
            abstractive_model = SUMMARIZATION_CONFIG["abstractive"]["model_name"]
            self.abstractive_pipeline = pipeline(
                "summarization",
                model=abstractive_model,
                tokenizer=abstractive_model
            )
            
            print("Transformer summarization models loaded successfully")
        except Exception as e:
            print(f"Error loading transformer models: {e}")
            self.extractive_pipeline = None
            self.abstractive_pipeline = None
    
    def preprocess_text_for_summarization(self, text: str) -> List[str]:
        """
        Preprocess text for summarization
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            # Remove extra whitespace
            sentence = ' '.join(sentence.split())
            
            # Skip very short sentences
            if len(sentence.split()) > 3:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """
        Calculate similarity between two sentences
        
        Args:
            sent1: First sentence
            sent2: Second sentence
            
        Returns:
            Similarity score
        """
        # Tokenize and remove stopwords
        words1 = [word.lower() for word in word_tokenize(sent1) 
                 if word.lower() not in self.stop_words and word.isalpha()]
        words2 = [word.lower() for word in word_tokenize(sent2) 
                 if word.lower() not in self.stop_words and word.isalpha()]
        
        # Create vocabulary
        all_words = list(set(words1 + words2))
        
        # Create vectors
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        for word in words1:
            if word in all_words:
                vector1[all_words.index(word)] += 1
        
        for word in words2:
            if word in all_words:
                vector2[all_words.index(word)] += 1
        
        # Calculate cosine similarity
        return 1 - cosine_distance(vector1, vector2)
    
    def textrank_summarization(self, 
                              text: str, 
                              num_sentences: int = 3) -> Dict[str, Union[str, List[str]]]:
        """
        Extractive summarization using TextRank algorithm
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
            
        Returns:
            Dictionary with summary information
        """
        # Preprocess text
        sentences = self.preprocess_text_for_summarization(text)
        
        if len(sentences) <= num_sentences:
            return {
                'summary': ' '.join(sentences),
                'sentences': sentences,
                'method': 'textrank',
                'compression_ratio': 1.0
            }
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self.calculate_sentence_similarity(
                        sentences[i], sentences[j]
                    )
        
        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Rank sentences
        ranked_sentences = sorted(
            ((scores[i], s, i) for i, s in enumerate(sentences)), 
            reverse=True
        )
        
        # Select top sentences
        selected_sentences = sorted(
            ranked_sentences[:num_sentences], 
            key=lambda x: x[2]  # Sort by original order
        )
        
        summary_sentences = [s[1] for s in selected_sentences]
        summary = ' '.join(summary_sentences)
        
        compression_ratio = len(summary) / len(text)
        
        return {
            'summary': summary,
            'sentences': summary_sentences,
            'method': 'textrank',
            'compression_ratio': compression_ratio,
            'sentence_scores': {i: scores[i] for i in range(len(sentences))}
        }
    
    def tfidf_summarization(self, 
                           text: str, 
                           num_sentences: int = 3) -> Dict[str, Union[str, List[str]]]:
        """
        Extractive summarization using TF-IDF
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
            
        Returns:
            Dictionary with summary information
        """
        # Preprocess text
        sentences = self.preprocess_text_for_summarization(text)
        
        if len(sentences) <= num_sentences:
            return {
                'summary': ' '.join(sentences),
                'sentences': sentences,
                'method': 'tfidf',
                'compression_ratio': 1.0
            }
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores (sum of TF-IDF values)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Rank sentences
        ranked_indices = sentence_scores.argsort()[::-1]
        
        # Select top sentences and sort by original order
        selected_indices = sorted(ranked_indices[:num_sentences])
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = ' '.join(summary_sentences)
        
        compression_ratio = len(summary) / len(text)
        
        return {
            'summary': summary,
            'sentences': summary_sentences,
            'method': 'tfidf',
            'compression_ratio': compression_ratio,
            'sentence_scores': {i: float(sentence_scores[i]) for i in range(len(sentences))}
        }
    
    def transformer_summarization(self, 
                                 text: str, 
                                 method: str = 'extractive',
                                 max_length: Optional[int] = None,
                                 min_length: Optional[int] = None) -> Dict[str, Union[str, float]]:
        """
        Summarization using transformer models
        
        Args:
            text: Input text
            method: 'extractive' or 'abstractive'
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Dictionary with summary information
        """
        if not TRANSFORMERS_AVAILABLE:
            return {
                'summary': text[:200] + '...',
                'method': f'transformer_{method}',
                'compression_ratio': 0.1,
                'error': 'Transformers not available'
            }
        
        pipeline_model = None
        config_key = method
        
        if method == 'extractive' and self.extractive_pipeline:
            pipeline_model = self.extractive_pipeline
        elif method == 'abstractive' and self.abstractive_pipeline:
            pipeline_model = self.abstractive_pipeline
        
        if not pipeline_model:
            return {
                'summary': text[:200] + '...',
                'method': f'transformer_{method}',
                'compression_ratio': 0.1,
                'error': f'{method} pipeline not available'
            }
        
        try:
            # Set parameters
            if max_length is None:
                max_length = SUMMARIZATION_CONFIG[config_key]["max_length"]
            if min_length is None:
                min_length = SUMMARIZATION_CONFIG[config_key]["min_length"]
            
            # Truncate text if too long
            max_input_length = 1024  # Most models have this limit
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            # Generate summary
            summary_result = pipeline_model(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            summary = summary_result[0]['summary_text']
            compression_ratio = len(summary) / len(text)
            
            return {
                'summary': summary,
                'method': f'transformer_{method}',
                'compression_ratio': compression_ratio
            }
        
        except Exception as e:
            return {
                'summary': text[:200] + '...',
                'method': f'transformer_{method}',
                'compression_ratio': 0.1,
                'error': str(e)
            }
    
    def summarize_text(self, 
                      text: str, 
                      method: str = 'textrank',
                      num_sentences: int = 3,
                      max_length: Optional[int] = None,
                      min_length: Optional[int] = None) -> Dict[str, Union[str, float]]:
        """
        Summarize text using specified method
        
        Args:
            text: Input text
            method: Summarization method
            num_sentences: Number of sentences (for extractive methods)
            max_length: Maximum length (for transformer methods)
            min_length: Minimum length (for transformer methods)
            
        Returns:
            Dictionary with summary information
        """
        if method == 'textrank':
            return self.textrank_summarization(text, num_sentences)
        elif method == 'tfidf':
            return self.tfidf_summarization(text, num_sentences)
        elif method == 'transformer_extractive':
            return self.transformer_summarization(text, 'extractive', max_length, min_length)
        elif method == 'transformer_abstractive':
            return self.transformer_summarization(text, 'abstractive', max_length, min_length)
        else:
            print(f"Unknown method: {method}")
            return {
                'summary': text[:200] + '...',
                'method': method,
                'compression_ratio': 0.1,
                'error': f'Unknown method: {method}'
            }
    
    def summarize_dataset(self, 
                         df: pd.DataFrame, 
                         text_column: str,
                         method: str = 'textrank',
                         num_sentences: int = 3) -> pd.DataFrame:
        """
        Summarize texts in a dataset
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            method: Summarization method
            num_sentences: Number of sentences in summary
            
        Returns:
            DataFrame with summaries
        """
        results = []
        
        for idx, text in df[text_column].items():
            summary_result = self.summarize_text(
                text, 
                method=method, 
                num_sentences=num_sentences
            )
            
            results.append({
                'index': idx,
                'summary': summary_result['summary'],
                'compression_ratio': summary_result['compression_ratio'],
                'method': summary_result['method']
            })
        
        summary_df = pd.DataFrame(results)
        
        # Add to original dataframe
        result_df = df.copy()
        result_df['summary'] = summary_df['summary']
        result_df['compression_ratio'] = summary_df['compression_ratio']
        result_df['summarization_method'] = summary_df['method']
        
        return result_df
    
    def compare_summarization_methods(self, 
                                    text: str, 
                                    num_sentences: int = 3) -> pd.DataFrame:
        """
        Compare different summarization methods
        
        Args:
            text: Input text
            num_sentences: Number of sentences for extractive methods
            
        Returns:
            DataFrame comparing different methods
        """
        methods = ['textrank', 'tfidf']
        
        if TRANSFORMERS_AVAILABLE:
            methods.extend(['transformer_extractive', 'transformer_abstractive'])
        
        results = []
        
        for method in methods:
            try:
                summary_result = self.summarize_text(
                    text, 
                    method=method, 
                    num_sentences=num_sentences
                )
                
                results.append({
                    'method': method,
                    'summary': summary_result['summary'],
                    'compression_ratio': summary_result['compression_ratio'],
                    'summary_length': len(summary_result['summary']),
                    'error': summary_result.get('error', None)
                })
            
            except Exception as e:
                results.append({
                    'method': method,
                    'summary': 'Error occurred',
                    'compression_ratio': 0.0,
                    'summary_length': 0,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
