"""
Unit tests for summarization module
"""

import unittest
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from summarization import TextSummarizer


class TestTextSummarizer(unittest.TestCase):
    """Test cases for TextSummarizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.summarizer = TextSummarizer()
        self.sample_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        in contrast to the natural intelligence displayed by humans and animals. 
        Leading AI textbooks define the field as the study of "intelligent agents": 
        any device that perceives its environment and takes actions that maximize 
        its chance of successfully achieving its goals. Colloquially, the term 
        "artificial intelligence" is often used to describe machines that mimic 
        "cognitive" functions that humans associate with the human mind, such as 
        "learning" and "problem solving". As machines become increasingly capable, 
        tasks considered to require "intelligence" are often removed from the 
        definition of AI, a phenomenon known as the AI effect.
        """
    
    def test_initialization(self):
        """Test TextSummarizer initialization"""
        self.assertIsInstance(self.summarizer.models, dict)
        self.assertIsInstance(self.summarizer.stop_words, set)
    
    def test_preprocess_text_for_summarization(self):
        """Test text preprocessing for summarization"""
        sentences = self.summarizer.preprocess_text_for_summarization(self.sample_text)
        
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 0)
        
        # Check that sentences are cleaned
        for sentence in sentences:
            self.assertIsInstance(sentence, str)
            self.assertGreater(len(sentence.split()), 3)
    
    def test_calculate_sentence_similarity(self):
        """Test sentence similarity calculation"""
        sent1 = "Artificial intelligence is demonstrated by machines."
        sent2 = "AI is shown by computer systems."
        
        similarity = self.summarizer.calculate_sentence_similarity(sent1, sent2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_textrank_summarization(self):
        """Test TextRank summarization"""
        result = self.summarizer.textrank_summarization(self.sample_text, num_sentences=2)
        
        self.assertIn('summary', result)
        self.assertIn('sentences', result)
        self.assertIn('method', result)
        self.assertIn('compression_ratio', result)
        
        self.assertEqual(result['method'], 'textrank')
        self.assertIsInstance(result['summary'], str)
        self.assertLessEqual(len(result['sentences']), 2)
    
    def test_tfidf_summarization(self):
        """Test TF-IDF summarization"""
        result = self.summarizer.tfidf_summarization(self.sample_text, num_sentences=2)
        
        self.assertIn('summary', result)
        self.assertIn('sentences', result)
        self.assertIn('method', result)
        self.assertIn('compression_ratio', result)
        
        self.assertEqual(result['method'], 'tfidf')
        self.assertIsInstance(result['summary'], str)
        self.assertLessEqual(len(result['sentences']), 2)
    
    def test_summarize_text(self):
        """Test general text summarization"""
        # Test TextRank method
        result = self.summarizer.summarize_text(
            self.sample_text, 
            method='textrank', 
            num_sentences=2
        )
        
        self.assertIn('summary', result)
        self.assertIn('method', result)
        self.assertEqual(result['method'], 'textrank')
    
    def test_compare_summarization_methods(self):
        """Test comparison of summarization methods"""
        comparison_df = self.summarizer.compare_summarization_methods(
            self.sample_text, 
            num_sentences=2
        )
        
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertIn('method', comparison_df.columns)
        self.assertIn('summary', comparison_df.columns)
        self.assertIn('compression_ratio', comparison_df.columns)
        
        # Should have at least textrank and tfidf methods
        methods = comparison_df['method'].tolist()
        self.assertIn('textrank', methods)
        self.assertIn('tfidf', methods)
    
    def test_summarize_dataset(self):
        """Test dataset summarization"""
        df = pd.DataFrame({
            'text': [self.sample_text, self.sample_text[:200]]
        })
        
        result_df = self.summarizer.summarize_dataset(
            df, 
            'text', 
            method='textrank', 
            num_sentences=1
        )
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('summary', result_df.columns)
        self.assertIn('compression_ratio', result_df.columns)
        self.assertIn('summarization_method', result_df.columns)
        self.assertEqual(len(result_df), 2)


if __name__ == '__main__':
    unittest.main()
