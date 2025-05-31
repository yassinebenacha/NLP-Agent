"""
Unit tests for NER module
"""

import unittest
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ner import NamedEntityRecognizer


class TestNamedEntityRecognizer(unittest.TestCase):
    """Test cases for NamedEntityRecognizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ner = NamedEntityRecognizer()
        self.sample_texts = [
            "Apple Inc. was founded by Steve Jobs in Cupertino.",
            "Microsoft Corporation is located in Redmond, Washington.",
            "Google was established by Larry Page and Sergey Brin.",
            "Amazon was founded by Jeff Bezos in 1994.",
            "Tesla is led by Elon Musk."
        ]
    
    def test_initialization(self):
        """Test NER initialization"""
        self.assertIsInstance(self.ner.models, dict)
        self.assertIsInstance(self.ner.entity_stats, dict)
    
    def test_nltk_ner(self):
        """Test NLTK NER extraction"""
        text = "Apple Inc. was founded by Steve Jobs."
        entities = self.ner.nltk_ner(text)
        
        self.assertIsInstance(entities, list)
        # Check that entities have required fields
        for entity in entities:
            self.assertIn('text', entity)
            self.assertIn('label', entity)
            self.assertIn('start', entity)
            self.assertIn('end', entity)
    
    def test_extract_entities_nltk(self):
        """Test entity extraction using NLTK"""
        text = "Microsoft Corporation is in Redmond."
        entities = self.ner.extract_entities(text, method='nltk')
        
        self.assertIsInstance(entities, list)
    
    def test_filter_entities_by_type(self):
        """Test entity filtering by type"""
        entities = [
            {'text': 'Apple', 'label': 'ORGANIZATION', 'start': 0, 'end': 5},
            {'text': 'Steve Jobs', 'label': 'PERSON', 'start': 20, 'end': 30},
            {'text': 'Cupertino', 'label': 'GPE', 'start': 34, 'end': 43}
        ]
        
        filtered = self.ner.filter_entities_by_type(entities, ['PERSON'])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['text'], 'Steve Jobs')
    
    def test_get_entities_by_confidence(self):
        """Test entity filtering by confidence"""
        entities = [
            {'text': 'Apple', 'label': 'ORG', 'confidence': 0.9},
            {'text': 'Steve', 'label': 'PERSON', 'confidence': 0.3},
            {'text': 'Cupertino', 'label': 'GPE', 'confidence': 0.8}
        ]
        
        filtered = self.ner.get_entities_by_confidence(entities, min_confidence=0.5)
        self.assertEqual(len(filtered), 2)
    
    def test_analyze_entities_in_dataset(self):
        """Test entity analysis in dataset"""
        df = pd.DataFrame({'text': self.sample_texts})
        result_df = self.ner.analyze_entities_in_dataset(df, 'text', method='nltk')
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('total_entities', result_df.columns)
        self.assertIn('unique_entity_types', result_df.columns)
        self.assertIn('entities', result_df.columns)


if __name__ == '__main__':
    unittest.main()
