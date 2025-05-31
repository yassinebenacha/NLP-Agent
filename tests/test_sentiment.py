"""
Unit tests for sentiment analysis module
"""

import unittest
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_analysis import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SentimentAnalyzer()
        self.sample_texts = [
            "I love this product! It's amazing.",
            "This is terrible. I hate it.",
            "It's okay, nothing special.",
            "Best purchase ever! Highly recommended.",
            "Worst experience. Complete waste of money."
        ]
        self.sample_labels = ['positive', 'negative', 'neutral', 'positive', 'negative']
    
    def test_textblob_sentiment(self):
        """Test TextBlob sentiment analysis"""
        text = "I love this product!"
        result = self.analyzer.textblob_sentiment(text)
        
        self.assertIn('sentiment', result)
        self.assertIn('polarity', result)
        self.assertIn('subjectivity', result)
        self.assertIn('confidence', result)
        self.assertIn(result['sentiment'], ['positive', 'negative', 'neutral'])
    
    def test_predict_sentiment_textblob(self):
        """Test sentiment prediction using TextBlob"""
        text = "This is an amazing product!"
        result = self.analyzer.predict_sentiment(text, method='textblob')
        
        self.assertIsInstance(result, dict)
        self.assertIn('sentiment', result)
        self.assertIn('confidence', result)
    
    def test_predict_sentiment_list(self):
        """Test sentiment prediction for list of texts"""
        results = self.analyzer.predict_sentiment(self.sample_texts[:3], method='textblob')
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIn('sentiment', result)
            self.assertIn(result['sentiment'], ['positive', 'negative', 'neutral'])
    
    def test_compare_methods(self):
        """Test comparison of different sentiment analysis methods"""
        comparison_df = self.analyzer.compare_methods(self.sample_texts[:3])
        
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertIn('text', comparison_df.columns)
        self.assertIn('textblob_sentiment', comparison_df.columns)
        self.assertEqual(len(comparison_df), 3)
    
    def test_empty_text(self):
        """Test handling of empty text"""
        result = self.analyzer.predict_sentiment("", method='textblob')
        
        self.assertIsInstance(result, dict)
        self.assertIn('sentiment', result)
    
    def test_invalid_method(self):
        """Test handling of invalid method"""
        result = self.analyzer.predict_sentiment("Test text", method='invalid_method')
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['sentiment'], 'neutral')
        self.assertEqual(result['confidence'], 0.0)


class TestSentimentAnalysisIntegration(unittest.TestCase):
    """Integration tests for sentiment analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SentimentAnalyzer()
        
        # Create sample DataFrame
        self.sample_df = pd.DataFrame({
            'text': [
                "I love this product! It's amazing.",
                "This is terrible. I hate it.",
                "It's okay, nothing special.",
                "Best purchase ever!",
                "Worst experience ever."
            ],
            'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
        })
    
    def test_analyze_sentiment_distribution(self):
        """Test sentiment distribution analysis"""
        result_df = self.analyzer.analyze_sentiment_distribution(
            self.sample_df, 'text', method='textblob'
        )
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('predicted_sentiment', result_df.columns)
        self.assertIn('sentiment_confidence', result_df.columns)
        self.assertEqual(len(result_df), len(self.sample_df))
    
    def test_sentiment_consistency(self):
        """Test sentiment analysis consistency"""
        text = "This is an amazing product that I absolutely love!"
        
        # Run multiple times to check consistency
        results = []
        for _ in range(5):
            result = self.analyzer.predict_sentiment(text, method='textblob')
            results.append(result['sentiment'])
        
        # All results should be the same for TextBlob (deterministic)
        self.assertEqual(len(set(results)), 1)


if __name__ == '__main__':
    unittest.main()
