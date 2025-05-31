"""
Topic Modeling module using LDA and BERTopic
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Traditional topic modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# Modern topic modeling
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("Warning: BERTopic not available. Some features will be limited.")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    import pyLDAvis.sklearn as sklearnvis
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False
    print("Warning: pyLDAvis not available. Visualization features will be limited.")

from config import TOPIC_MODELING_CONFIG


class TopicModeler:
    """
    Comprehensive topic modeling class supporting LDA and BERTopic
    """
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.topic_results = {}
        
        # Initialize BERTopic components if available
        if BERTOPIC_AVAILABLE:
            self._init_bertopic_components()
    
    def _init_bertopic_components(self):
        """Initialize BERTopic components"""
        try:
            # Sentence transformer for embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # UMAP for dimensionality reduction
            self.umap_model = UMAP(
                n_neighbors=15, 
                n_components=5, 
                min_dist=0.0, 
                metric='cosine',
                random_state=42
            )
            
            # HDBSCAN for clustering
            self.hdbscan_model = HDBSCAN(
                min_cluster_size=TOPIC_MODELING_CONFIG["bertopic"]["min_topic_size"],
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            print("BERTopic components initialized successfully")
        except Exception as e:
            print(f"Error initializing BERTopic components: {e}")
            self.sentence_model = None
    
    def preprocess_for_lda(self, texts: List[str], min_df: int = 2, max_df: float = 0.8) -> Tuple[Any, Any]:
        """
        Preprocess texts for LDA modeling
        
        Args:
            texts: List of texts
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            
        Returns:
            Tuple of (vectorizer, document-term matrix)
        """
        # Use CountVectorizer for LDA
        vectorizer = CountVectorizer(
            max_features=1000,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        self.vectorizers['lda'] = vectorizer
        
        return vectorizer, doc_term_matrix
    
    def train_sklearn_lda(self, 
                         texts: List[str], 
                         n_topics: int = 10,
                         random_state: int = 42) -> Dict[str, Any]:
        """
        Train LDA model using scikit-learn
        
        Args:
            texts: List of texts
            n_topics: Number of topics
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with model results
        """
        print(f"Training sklearn LDA with {n_topics} topics...")
        
        # Preprocess texts
        vectorizer, doc_term_matrix = self.preprocess_for_lda(texts)
        
        # Train LDA model
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=100,
            learning_method='batch'
        )
        
        lda_model.fit(doc_term_matrix)
        
        # Get topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': top_weights
            })
        
        # Calculate perplexity
        perplexity = lda_model.perplexity(doc_term_matrix)
        
        # Store results
        results = {
            'model': lda_model,
            'vectorizer': vectorizer,
            'doc_term_matrix': doc_term_matrix,
            'topics': topics,
            'perplexity': perplexity,
            'n_topics': n_topics
        }
        
        self.models['sklearn_lda'] = results
        print(f"sklearn LDA training completed. Perplexity: {perplexity:.2f}")
        
        return results
    
    def train_gensim_lda(self, 
                        texts: List[str], 
                        n_topics: int = 10,
                        passes: int = 10,
                        alpha: str = 'auto',
                        eta: str = 'auto') -> Dict[str, Any]:
        """
        Train LDA model using Gensim
        
        Args:
            texts: List of texts
            n_topics: Number of topics
            passes: Number of passes through the corpus
            alpha: Alpha parameter
            eta: Eta parameter
            
        Returns:
            Dictionary with model results
        """
        print(f"Training Gensim LDA with {n_topics} topics...")
        
        # Preprocess texts for Gensim
        processed_texts = [text.lower().split() for text in texts]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_texts)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            random_state=42,
            passes=passes,
            alpha=alpha,
            eta=eta,
            per_word_topics=True
        )
        
        # Get topics
        topics = []
        for topic_id in range(n_topics):
            topic_words = lda_model.show_topic(topic_id, topn=10)
            words = [word for word, _ in topic_words]
            weights = [weight for _, weight in topic_words]
            
            topics.append({
                'topic_id': topic_id,
                'words': words,
                'weights': weights
            })
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            model=lda_model, 
            texts=processed_texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Store results
        results = {
            'model': lda_model,
            'dictionary': dictionary,
            'corpus': corpus,
            'topics': topics,
            'coherence': coherence_score,
            'n_topics': n_topics
        }
        
        self.models['gensim_lda'] = results
        print(f"Gensim LDA training completed. Coherence: {coherence_score:.4f}")
        
        return results
    
    def train_bertopic(self, 
                      texts: List[str],
                      min_topic_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train BERTopic model
        
        Args:
            texts: List of texts
            min_topic_size: Minimum topic size
            
        Returns:
            Dictionary with model results
        """
        if not BERTOPIC_AVAILABLE:
            print("BERTopic not available")
            return {}
        
        print("Training BERTopic model...")
        
        if min_topic_size is None:
            min_topic_size = TOPIC_MODELING_CONFIG["bertopic"]["min_topic_size"]
        
        # Update HDBSCAN min_cluster_size
        self.hdbscan_model.min_cluster_size = min_topic_size
        
        # Initialize BERTopic
        topic_model = BERTopic(
            embedding_model=self.sentence_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            calculate_probabilities=TOPIC_MODELING_CONFIG["bertopic"]["calculate_probabilities"],
            verbose=True
        )
        
        # Fit the model
        topics, probabilities = topic_model.fit_transform(texts)
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        
        # Get topics in a structured format
        topics_structured = []
        for topic_id in topic_model.get_topics():
            if topic_id != -1:  # Exclude outlier topic
                topic_words = topic_model.get_topic(topic_id)
                words = [word for word, _ in topic_words]
                weights = [weight for _, weight in topic_words]
                
                topics_structured.append({
                    'topic_id': topic_id,
                    'words': words,
                    'weights': weights
                })
        
        # Store results
        results = {
            'model': topic_model,
            'topics': topics,
            'probabilities': probabilities,
            'topic_info': topic_info,
            'topics_structured': topics_structured,
            'n_topics': len(topic_model.get_topics()) - 1  # Exclude outlier topic
        }
        
        self.models['bertopic'] = results
        print(f"BERTopic training completed. Found {results['n_topics']} topics")
        
        return results
    
    def get_document_topics(self, 
                           texts: List[str], 
                           model_type: str = 'bertopic') -> List[Dict]:
        """
        Get topic assignments for documents
        
        Args:
            texts: List of texts
            model_type: Type of model to use
            
        Returns:
            List of topic assignments
        """
        if model_type not in self.models:
            print(f"Model {model_type} not trained yet")
            return []
        
        results = []
        
        if model_type == 'bertopic' and BERTOPIC_AVAILABLE:
            model = self.models[model_type]['model']
            topics, probabilities = model.transform(texts)
            
            for i, (topic, prob) in enumerate(zip(topics, probabilities)):
                results.append({
                    'document_id': i,
                    'topic': topic,
                    'probability': prob.max() if prob is not None else 0.0
                })
        
        elif model_type == 'sklearn_lda':
            model_data = self.models[model_type]
            vectorizer = model_data['vectorizer']
            lda_model = model_data['model']
            
            doc_term_matrix = vectorizer.transform(texts)
            doc_topic_probs = lda_model.transform(doc_term_matrix)
            
            for i, probs in enumerate(doc_topic_probs):
                topic = np.argmax(probs)
                probability = probs[topic]
                
                results.append({
                    'document_id': i,
                    'topic': topic,
                    'probability': probability
                })
        
        elif model_type == 'gensim_lda':
            model_data = self.models[model_type]
            lda_model = model_data['model']
            dictionary = model_data['dictionary']
            
            for i, text in enumerate(texts):
                processed_text = text.lower().split()
                bow = dictionary.doc2bow(processed_text)
                doc_topics = lda_model.get_document_topics(bow)
                
                if doc_topics:
                    topic, probability = max(doc_topics, key=lambda x: x[1])
                else:
                    topic, probability = -1, 0.0
                
                results.append({
                    'document_id': i,
                    'topic': topic,
                    'probability': probability
                })
        
        return results
    
    def compare_models(self, texts: List[str], n_topics: int = 10) -> pd.DataFrame:
        """
        Compare different topic modeling approaches
        
        Args:
            texts: List of texts
            n_topics: Number of topics for LDA models
            
        Returns:
            DataFrame comparing model performance
        """
        results = []
        
        # Train sklearn LDA
        sklearn_results = self.train_sklearn_lda(texts, n_topics=n_topics)
        results.append({
            'model': 'sklearn_lda',
            'n_topics': n_topics,
            'perplexity': sklearn_results['perplexity'],
            'coherence': None
        })
        
        # Train Gensim LDA
        gensim_results = self.train_gensim_lda(texts, n_topics=n_topics)
        results.append({
            'model': 'gensim_lda',
            'n_topics': n_topics,
            'perplexity': None,
            'coherence': gensim_results['coherence']
        })
        
        # Train BERTopic
        if BERTOPIC_AVAILABLE:
            bertopic_results = self.train_bertopic(texts)
            results.append({
                'model': 'bertopic',
                'n_topics': bertopic_results['n_topics'],
                'perplexity': None,
                'coherence': None
            })
        
        return pd.DataFrame(results)
