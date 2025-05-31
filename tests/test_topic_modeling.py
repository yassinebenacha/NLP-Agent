"""
Unit tests for topic modeling module
"""

import unittest
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from topic_modeling import TopicModeler


class TestTopicModeler(unittest.TestCase):
    """Test cases for TopicModeler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.modeler = TopicModeler()
        self.sample_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual information.",
            "Data science combines statistics and programming for insights."
        ]
    
    def test_initialization(self):
        """Test TopicModeler initialization"""
        self.assertIsInstance(self.modeler.models, dict)
        self.assertIsInstance(self.modeler.vectorizers, dict)
        self.assertIsInstance(self.modeler.topic_results, dict)
    
    def test_preprocess_for_lda(self):
        """Test LDA preprocessing"""
        vectorizer, doc_term_matrix = self.modeler.preprocess_for_lda(self.sample_texts)
        
        self.assertIsNotNone(vectorizer)
        self.assertIsNotNone(doc_term_matrix)
        self.assertEqual(doc_term_matrix.shape[0], len(self.sample_texts))
    
    def test_sklearn_lda_training(self):
        """Test sklearn LDA training"""
        results = self.modeler.train_sklearn_lda(self.sample_texts, n_topics=2)
        
        self.assertIn('model', results)
        self.assertIn('topics', results)
        self.assertIn('perplexity', results)
        self.assertEqual(len(results['topics']), 2)
    
    def test_gensim_lda_training(self):
        """Test Gensim LDA training"""
        results = self.modeler.train_gensim_lda(self.sample_texts, n_topics=2)
        
        self.assertIn('model', results)
        self.assertIn('topics', results)
        self.assertIn('coherence', results)
        self.assertEqual(len(results['topics']), 2)
    
    def test_compare_models(self):
        """Test model comparison"""
        comparison_df = self.modeler.compare_models(self.sample_texts, n_topics=2)
        
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertIn('model', comparison_df.columns)
        self.assertIn('n_topics', comparison_df.columns)


if __name__ == '__main__':
    unittest.main()
