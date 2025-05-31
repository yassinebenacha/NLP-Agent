"""
Evaluation utilities for NLP models
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# ROUGE metrics for summarization
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available. ROUGE metrics will be limited.")

# BERT Score for semantic similarity
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: bert-score not available. BERT Score metrics will be limited.")

# Topic modeling evaluation
try:
    from gensim.models import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

from config import EVALUATION_CONFIG


class ModelEvaluator:
    """
    Comprehensive evaluation class for NLP models
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
    
    def evaluate_classification(self, 
                              y_true: List, 
                              y_pred: List,
                              y_pred_proba: Optional[List] = None,
                              labels: Optional[List] = None,
                              task_name: str = "classification") -> Dict[str, Any]:
        """
        Evaluate classification model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            labels: Label names
            task_name: Name of the task
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'task_name': task_name
        }
        
        # ROC AUC for binary/multiclass classification
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    if len(y_pred_proba[0]) == 2:
                        auc = roc_auc_score(y_true, [p[1] for p in y_pred_proba])
                    else:
                        auc = roc_auc_score(y_true, y_pred_proba)
                else:
                    # Multiclass classification
                    auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                
                results['roc_auc'] = auc
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
                results['roc_auc'] = None
        
        self.evaluation_results[task_name] = results
        return results
    
    def evaluate_sentiment_analysis(self, 
                                   y_true: List, 
                                   y_pred: List,
                                   y_pred_proba: Optional[List] = None) -> Dict[str, Any]:
        """
        Evaluate sentiment analysis model
        
        Args:
            y_true: True sentiment labels
            y_pred: Predicted sentiment labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        return self.evaluate_classification(
            y_true, y_pred, y_pred_proba, 
            labels=['negative', 'neutral', 'positive'],
            task_name='sentiment_analysis'
        )
    
    def evaluate_ner(self, 
                    true_entities: List[List[Dict]], 
                    pred_entities: List[List[Dict]]) -> Dict[str, Any]:
        """
        Evaluate Named Entity Recognition model
        
        Args:
            true_entities: List of true entity lists for each document
            pred_entities: List of predicted entity lists for each document
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert entities to sets for comparison
        true_entity_sets = []
        pred_entity_sets = []
        
        for true_ents, pred_ents in zip(true_entities, pred_entities):
            # Create sets of (start, end, label) tuples
            true_set = set((ent['start'], ent['end'], ent['label']) for ent in true_ents)
            pred_set = set((ent['start'], ent['end'], ent['label']) for ent in pred_ents)
            
            true_entity_sets.append(true_set)
            pred_entity_sets.append(pred_set)
        
        # Calculate metrics
        total_true = sum(len(s) for s in true_entity_sets)
        total_pred = sum(len(s) for s in pred_entity_sets)
        total_correct = sum(len(true_set & pred_set) 
                           for true_set, pred_set in zip(true_entity_sets, pred_entity_sets))
        
        precision = total_correct / total_pred if total_pred > 0 else 0
        recall = total_correct / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Entity-level metrics by type
        entity_types = set()
        for true_ents in true_entities:
            entity_types.update(ent['label'] for ent in true_ents)
        for pred_ents in pred_entities:
            entity_types.update(ent['label'] for ent in pred_ents)
        
        type_metrics = {}
        for entity_type in entity_types:
            type_true = sum(sum(1 for ent in true_ents if ent['label'] == entity_type) 
                           for true_ents in true_entities)
            type_pred = sum(sum(1 for ent in pred_ents if ent['label'] == entity_type) 
                           for pred_ents in pred_entities)
            type_correct = sum(len(set((ent['start'], ent['end'], ent['label']) 
                                     for ent in true_ents if ent['label'] == entity_type) &
                                 set((ent['start'], ent['end'], ent['label']) 
                                     for ent in pred_ents if ent['label'] == entity_type))
                              for true_ents, pred_ents in zip(true_entities, pred_entities))
            
            type_precision = type_correct / type_pred if type_pred > 0 else 0
            type_recall = type_correct / type_true if type_true > 0 else 0
            type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0
            
            type_metrics[entity_type] = {
                'precision': type_precision,
                'recall': type_recall,
                'f1_score': type_f1,
                'support': type_true
            }
        
        results = {
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1': f1,
            'total_true_entities': total_true,
            'total_predicted_entities': total_pred,
            'total_correct_entities': total_correct,
            'entity_type_metrics': type_metrics,
            'task_name': 'named_entity_recognition'
        }
        
        self.evaluation_results['ner'] = results
        return results
    
    def evaluate_summarization(self, 
                              reference_summaries: List[str], 
                              generated_summaries: List[str]) -> Dict[str, Any]:
        """
        Evaluate text summarization using ROUGE and BERT Score
        
        Args:
            reference_summaries: Reference/gold summaries
            generated_summaries: Generated summaries
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {'task_name': 'text_summarization'}
        
        # ROUGE metrics
        if ROUGE_AVAILABLE:
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for ref, gen in zip(reference_summaries, generated_summaries):
                scores = self.rouge_scorer.score(ref, gen)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            
            # Average ROUGE scores
            results['rouge1'] = np.mean(rouge_scores['rouge1'])
            results['rouge2'] = np.mean(rouge_scores['rouge2'])
            results['rougeL'] = np.mean(rouge_scores['rougeL'])
            results['rouge_scores_detail'] = rouge_scores
        
        # BERT Score
        if BERT_SCORE_AVAILABLE:
            try:
                P, R, F1 = bert_score(generated_summaries, reference_summaries, lang='en')
                results['bert_score_precision'] = P.mean().item()
                results['bert_score_recall'] = R.mean().item()
                results['bert_score_f1'] = F1.mean().item()
            except Exception as e:
                print(f"Error calculating BERT Score: {e}")
                results['bert_score_precision'] = None
                results['bert_score_recall'] = None
                results['bert_score_f1'] = None
        
        # Basic metrics
        avg_ref_length = np.mean([len(ref.split()) for ref in reference_summaries])
        avg_gen_length = np.mean([len(gen.split()) for gen in generated_summaries])
        compression_ratio = avg_gen_length / avg_ref_length if avg_ref_length > 0 else 0
        
        results['average_reference_length'] = avg_ref_length
        results['average_generated_length'] = avg_gen_length
        results['compression_ratio'] = compression_ratio
        
        self.evaluation_results['summarization'] = results
        return results
    
    def evaluate_topic_modeling(self, 
                               model, 
                               texts: List[str],
                               model_type: str = 'gensim') -> Dict[str, Any]:
        """
        Evaluate topic modeling using coherence metrics
        
        Args:
            model: Trained topic model
            texts: Original texts
            model_type: Type of model ('gensim', 'sklearn', 'bertopic')
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {'task_name': 'topic_modeling', 'model_type': model_type}
        
        if model_type == 'gensim' and GENSIM_AVAILABLE:
            try:
                # Preprocess texts
                processed_texts = [text.lower().split() for text in texts]
                
                # Create dictionary
                from gensim import corpora
                dictionary = corpora.Dictionary(processed_texts)
                
                # Calculate coherence
                coherence_model = CoherenceModel(
                    model=model, 
                    texts=processed_texts, 
                    dictionary=dictionary, 
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
                results['coherence_c_v'] = coherence_score
                
                # Calculate perplexity if available
                if hasattr(model, 'log_perplexity'):
                    corpus = [dictionary.doc2bow(text) for text in processed_texts]
                    perplexity = model.log_perplexity(corpus)
                    results['perplexity'] = perplexity
                
            except Exception as e:
                print(f"Error evaluating Gensim model: {e}")
                results['coherence_c_v'] = None
                results['perplexity'] = None
        
        elif model_type == 'sklearn':
            try:
                # For sklearn LDA, we can calculate perplexity
                if hasattr(model, 'perplexity'):
                    # This would require the document-term matrix
                    # results['perplexity'] = model.perplexity(doc_term_matrix)
                    pass
            except Exception as e:
                print(f"Error evaluating sklearn model: {e}")
        
        elif model_type == 'bertopic':
            try:
                # BERTopic doesn't have traditional coherence metrics
                # But we can calculate some basic statistics
                topic_info = model.get_topic_info()
                results['num_topics'] = len(topic_info) - 1  # Exclude outlier topic
                results['num_outliers'] = topic_info[topic_info['Topic'] == -1]['Count'].sum()
                results['avg_topic_size'] = topic_info[topic_info['Topic'] != -1]['Count'].mean()
            except Exception as e:
                print(f"Error evaluating BERTopic model: {e}")
        
        self.evaluation_results['topic_modeling'] = results
        return results
    
    def generate_evaluation_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive evaluation report
        
        Returns:
            DataFrame with evaluation summary
        """
        report_data = []
        
        for task_name, results in self.evaluation_results.items():
            row = {'task': task_name}
            
            if task_name in ['sentiment_analysis', 'classification']:
                row.update({
                    'accuracy': results.get('accuracy'),
                    'precision': results.get('precision'),
                    'recall': results.get('recall'),
                    'f1_score': results.get('f1_score'),
                    'roc_auc': results.get('roc_auc')
                })
            
            elif task_name == 'ner':
                row.update({
                    'precision': results.get('overall_precision'),
                    'recall': results.get('overall_recall'),
                    'f1_score': results.get('overall_f1'),
                    'total_entities': results.get('total_true_entities')
                })
            
            elif task_name == 'summarization':
                row.update({
                    'rouge1': results.get('rouge1'),
                    'rouge2': results.get('rouge2'),
                    'rougeL': results.get('rougeL'),
                    'bert_score_f1': results.get('bert_score_f1'),
                    'compression_ratio': results.get('compression_ratio')
                })
            
            elif task_name == 'topic_modeling':
                row.update({
                    'coherence': results.get('coherence_c_v'),
                    'perplexity': results.get('perplexity'),
                    'num_topics': results.get('num_topics'),
                    'model_type': results.get('model_type')
                })
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def compare_models(self, 
                      model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models for the same task
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            DataFrame comparing models
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'model': model_name}
            
            # Extract relevant metrics based on task type
            if 'accuracy' in results:
                row.update({
                    'accuracy': results.get('accuracy'),
                    'precision': results.get('precision'),
                    'recall': results.get('recall'),
                    'f1_score': results.get('f1_score')
                })
            
            if 'rouge1' in results:
                row.update({
                    'rouge1': results.get('rouge1'),
                    'rouge2': results.get('rouge2'),
                    'rougeL': results.get('rougeL')
                })
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
