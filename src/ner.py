"""
Named Entity Recognition module using spaCy and transformers
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')

# spaCy for NER
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. NER features will be limited.")

# Transformers for advanced NER
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Some NER features will be limited.")

# NLTK for basic NER
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize

from config import NER_CONFIG


class NamedEntityRecognizer:
    """
    Comprehensive Named Entity Recognition class
    """
    
    def __init__(self):
        self.models = {}
        self.entity_stats = {}
        
        # Initialize spaCy model
        if SPACY_AVAILABLE:
            self._init_spacy_model()
        
        # Initialize transformer model
        if TRANSFORMERS_AVAILABLE:
            self._init_transformer_model()
    
    def _init_spacy_model(self):
        """Initialize spaCy NER model"""
        try:
            model_name = NER_CONFIG["spacy_model"]
            self.nlp = spacy.load(model_name)
            print(f"Loaded spaCy model: {model_name}")
        except OSError:
            print(f"spaCy model '{NER_CONFIG['spacy_model']}' not found. Please install it.")
            self.nlp = None
    
    def _init_transformer_model(self):
        """Initialize transformer NER model"""
        try:
            # Use a pre-trained NER model
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple"
            )
            print(f"Loaded transformer NER model: {model_name}")
        except Exception as e:
            print(f"Error loading transformer NER model: {e}")
            self.ner_pipeline = None
    
    def spacy_ner(self, text: str) -> List[Dict]:
        """
        Extract named entities using spaCy
        
        Args:
            text: Input text
            
        Returns:
            List of entities with their information
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0,  # spaCy doesn't provide confidence scores
                'description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def transformer_ner(self, text: str) -> List[Dict]:
        """
        Extract named entities using transformer model
        
        Args:
            text: Input text
            
        Returns:
            List of entities with their information
        """
        if not self.ner_pipeline:
            return []
        
        try:
            # Split long texts into chunks
            max_length = 512
            if len(text) > max_length:
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            else:
                chunks = [text]
            
            all_entities = []
            offset = 0
            
            for chunk in chunks:
                entities = self.ner_pipeline(chunk)
                
                for entity in entities:
                    all_entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'start': entity['start'] + offset,
                        'end': entity['end'] + offset,
                        'confidence': entity['score']
                    })
                
                offset += len(chunk)
            
            return all_entities
        
        except Exception as e:
            print(f"Error in transformer NER: {e}")
            return []
    
    def nltk_ner(self, text: str) -> List[Dict]:
        """
        Extract named entities using NLTK
        
        Args:
            text: Input text
            
        Returns:
            List of entities with their information
        """
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity chunking
            tree = ne_chunk(pos_tags)
            
            entities = []
            current_entity = []
            current_label = None
            start_pos = 0
            
            for i, item in enumerate(tree):
                if hasattr(item, 'label'):  # It's a named entity
                    if current_label != item.label():
                        # Save previous entity if exists
                        if current_entity:
                            entity_text = ' '.join(current_entity)
                            entities.append({
                                'text': entity_text,
                                'label': current_label,
                                'start': start_pos,
                                'end': start_pos + len(entity_text),
                                'confidence': 0.8  # NLTK doesn't provide confidence
                            })
                        
                        # Start new entity
                        current_entity = [item[0][0]]
                        current_label = item.label()
                        start_pos = text.find(item[0][0], start_pos)
                    else:
                        # Continue current entity
                        current_entity.append(item[0][0])
                else:
                    # Save current entity if exists
                    if current_entity:
                        entity_text = ' '.join(current_entity)
                        entities.append({
                            'text': entity_text,
                            'label': current_label,
                            'start': start_pos,
                            'end': start_pos + len(entity_text),
                            'confidence': 0.8
                        })
                        current_entity = []
                        current_label = None
            
            # Save last entity if exists
            if current_entity:
                entity_text = ' '.join(current_entity)
                entities.append({
                    'text': entity_text,
                    'label': current_label,
                    'start': start_pos,
                    'end': start_pos + len(entity_text),
                    'confidence': 0.8
                })
            
            return entities
        
        except Exception as e:
            print(f"Error in NLTK NER: {e}")
            return []
    
    def extract_entities(self, 
                        text: str, 
                        method: str = 'spacy') -> List[Dict]:
        """
        Extract named entities using specified method
        
        Args:
            text: Input text
            method: Method to use ('spacy', 'transformer', 'nltk')
            
        Returns:
            List of entities
        """
        if method == 'spacy':
            return self.spacy_ner(text)
        elif method == 'transformer':
            return self.transformer_ner(text)
        elif method == 'nltk':
            return self.nltk_ner(text)
        else:
            print(f"Unknown method: {method}")
            return []
    
    def analyze_entities_in_dataset(self, 
                                   df: pd.DataFrame, 
                                   text_column: str,
                                   method: str = 'spacy') -> pd.DataFrame:
        """
        Analyze entities in a dataset
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            method: Method to use for NER
            
        Returns:
            DataFrame with entity analysis
        """
        results = []
        
        for idx, text in df[text_column].items():
            entities = self.extract_entities(text, method=method)
            
            # Count entities by type
            entity_counts = {}
            for entity in entities:
                label = entity['label']
                entity_counts[label] = entity_counts.get(label, 0) + 1
            
            results.append({
                'index': idx,
                'total_entities': len(entities),
                'unique_entity_types': len(entity_counts),
                'entity_counts': entity_counts,
                'entities': entities
            })
        
        analysis_df = pd.DataFrame(results)
        
        # Add to original dataframe
        result_df = df.copy()
        result_df['total_entities'] = analysis_df['total_entities']
        result_df['unique_entity_types'] = analysis_df['unique_entity_types']
        result_df['entities'] = analysis_df['entities']
        
        return result_df
    
    def get_entity_statistics(self, 
                             df: pd.DataFrame, 
                             text_column: str,
                             method: str = 'spacy') -> Dict:
        """
        Get comprehensive entity statistics
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            method: Method to use for NER
            
        Returns:
            Dictionary with entity statistics
        """
        all_entities = []
        entity_type_counts = {}
        entity_text_counts = {}
        
        for text in df[text_column]:
            entities = self.extract_entities(text, method=method)
            all_entities.extend(entities)
            
            for entity in entities:
                label = entity['label']
                text_val = entity['text']
                
                # Count by type
                entity_type_counts[label] = entity_type_counts.get(label, 0) + 1
                
                # Count by text
                entity_text_counts[text_val] = entity_text_counts.get(text_val, 0) + 1
        
        # Calculate statistics
        total_entities = len(all_entities)
        unique_entity_types = len(entity_type_counts)
        unique_entity_texts = len(entity_text_counts)
        
        # Most common entities
        most_common_types = sorted(entity_type_counts.items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
        most_common_texts = sorted(entity_text_counts.items(), 
                                 key=lambda x: x[1], reverse=True)[:20]
        
        # Average confidence (if available)
        confidences = [e['confidence'] for e in all_entities if 'confidence' in e]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        statistics = {
            'total_entities': total_entities,
            'unique_entity_types': unique_entity_types,
            'unique_entity_texts': unique_entity_texts,
            'entity_type_counts': entity_type_counts,
            'most_common_types': most_common_types,
            'most_common_texts': most_common_texts,
            'average_confidence': avg_confidence,
            'method_used': method
        }
        
        self.entity_stats[method] = statistics
        return statistics
    
    def compare_ner_methods(self, texts: List[str]) -> pd.DataFrame:
        """
        Compare different NER methods
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame comparing different methods
        """
        results = []
        
        for i, text in enumerate(texts):
            result = {
                'text_id': i,
                'text_preview': text[:100] + '...' if len(text) > 100 else text
            }
            
            # Try each method
            methods = ['spacy', 'transformer', 'nltk']
            
            for method in methods:
                try:
                    entities = self.extract_entities(text, method=method)
                    
                    # Count entities by type
                    entity_counts = {}
                    for entity in entities:
                        label = entity['label']
                        entity_counts[label] = entity_counts.get(label, 0) + 1
                    
                    result[f'{method}_total'] = len(entities)
                    result[f'{method}_types'] = len(entity_counts)
                    result[f'{method}_entities'] = entity_counts
                    
                except Exception as e:
                    result[f'{method}_total'] = 0
                    result[f'{method}_types'] = 0
                    result[f'{method}_entities'] = {}
                    print(f"Error with {method}: {e}")
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def filter_entities_by_type(self, 
                               entities: List[Dict], 
                               entity_types: List[str]) -> List[Dict]:
        """
        Filter entities by specific types
        
        Args:
            entities: List of entities
            entity_types: List of entity types to keep
            
        Returns:
            Filtered list of entities
        """
        return [entity for entity in entities if entity['label'] in entity_types]
    
    def get_entities_by_confidence(self, 
                                  entities: List[Dict], 
                                  min_confidence: float = 0.5) -> List[Dict]:
        """
        Filter entities by confidence threshold
        
        Args:
            entities: List of entities
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of entities
        """
        return [entity for entity in entities 
                if entity.get('confidence', 1.0) >= min_confidence]
