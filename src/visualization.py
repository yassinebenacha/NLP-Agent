"""
Visualization utilities for NLP analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Word cloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("Warning: wordcloud not available. Word cloud visualizations will be limited.")

# Network analysis
import networkx as nx

# Topic modeling visualization
try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    import pyLDAvis.sklearn as sklearnvis
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False

from config import PLOT_CONFIG


class NLPVisualizer:
    """
    Comprehensive visualization class for NLP analysis
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer
        
        Args:
            style: Matplotlib style to use
        """
        self.style = style
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup plotting style and parameters"""
        plt.style.use(self.style)
        sns.set_palette(PLOT_CONFIG["color_palette"])
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = PLOT_CONFIG["figsize"]
        plt.rcParams['figure.dpi'] = PLOT_CONFIG["dpi"]
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_sentiment_distribution(self, 
                                   df: pd.DataFrame, 
                                   sentiment_column: str = 'sentiment',
                                   title: str = 'Sentiment Distribution') -> plt.Figure:
        """
        Plot sentiment distribution
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Name of sentiment column
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        sentiment_counts = df[sentiment_column].value_counts()
        ax1.bar(sentiment_counts.index, sentiment_counts.values)
        ax1.set_title(f'{title} - Counts')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        
        # Pie chart
        ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax2.set_title(f'{title} - Proportions')
        
        plt.tight_layout()
        return fig
    
    def plot_sentiment_confidence(self, 
                                 df: pd.DataFrame,
                                 sentiment_column: str = 'sentiment',
                                 confidence_column: str = 'confidence') -> plt.Figure:
        """
        Plot sentiment confidence distribution
        
        Args:
            df: DataFrame with sentiment and confidence data
            sentiment_column: Name of sentiment column
            confidence_column: Name of confidence column
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Box plot of confidence by sentiment
        sns.boxplot(data=df, x=sentiment_column, y=confidence_column, ax=ax)
        ax.set_title('Sentiment Confidence Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Confidence Score')
        
        plt.tight_layout()
        return fig
    
    def plot_text_length_distribution(self, 
                                     df: pd.DataFrame,
                                     text_column: str = 'text',
                                     title: str = 'Text Length Distribution') -> plt.Figure:
        """
        Plot text length distribution
        
        Args:
            df: DataFrame with text data
            text_column: Name of text column
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Calculate text lengths
        text_lengths = df[text_column].str.len()
        word_counts = df[text_column].str.split().str.len()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Character length distribution
        ax1.hist(text_lengths, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title(f'{title} - Character Length')
        ax1.set_xlabel('Number of Characters')
        ax1.set_ylabel('Frequency')
        ax1.axvline(text_lengths.mean(), color='red', linestyle='--', 
                   label=f'Mean: {text_lengths.mean():.1f}')
        ax1.legend()
        
        # Word count distribution
        ax2.hist(word_counts, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title(f'{title} - Word Count')
        ax2.set_xlabel('Number of Words')
        ax2.set_ylabel('Frequency')
        ax2.axvline(word_counts.mean(), color='red', linestyle='--',
                   label=f'Mean: {word_counts.mean():.1f}')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def create_word_cloud(self, 
                         texts: List[str],
                         title: str = 'Word Cloud',
                         max_words: int = 100) -> plt.Figure:
        """
        Create word cloud visualization
        
        Args:
            texts: List of texts
            title: Plot title
            max_words: Maximum number of words to display
            
        Returns:
            Matplotlib figure
        """
        if not WORDCLOUD_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'WordCloud not available', 
                   ha='center', va='center', fontsize=20)
            ax.set_title(title)
            return fig
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=max_words,
            colormap='viridis'
        ).generate(combined_text)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16)
        
        return fig
    
    def plot_topic_distribution(self, 
                               topic_info: pd.DataFrame,
                               title: str = 'Topic Distribution') -> plt.Figure:
        """
        Plot topic distribution
        
        Args:
            topic_info: DataFrame with topic information
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Filter out outlier topic if present
        if 'Topic' in topic_info.columns:
            plot_data = topic_info[topic_info['Topic'] != -1]
        else:
            plot_data = topic_info
        
        # Bar plot of topic sizes
        if 'Count' in plot_data.columns:
            ax.bar(range(len(plot_data)), plot_data['Count'])
            ax.set_xlabel('Topic ID')
            ax.set_ylabel('Number of Documents')
        else:
            # Fallback if Count column not available
            ax.bar(range(len(plot_data)), [1] * len(plot_data))
            ax.set_xlabel('Topic ID')
            ax.set_ylabel('Topic Count')
        
        ax.set_title(title)
        ax.set_xticks(range(len(plot_data)))
        
        plt.tight_layout()
        return fig
    
    def plot_topic_words(self, 
                        topics: List[Dict],
                        n_words: int = 10,
                        title: str = 'Top Words per Topic') -> plt.Figure:
        """
        Plot top words for each topic
        
        Args:
            topics: List of topic dictionaries
            n_words: Number of words to show per topic
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        n_topics = len(topics)
        n_cols = min(3, n_topics)
        n_rows = (n_topics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_topics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, topic in enumerate(topics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            words = topic['words'][:n_words]
            weights = topic['weights'][:n_words] if 'weights' in topic else [1] * len(words)
            
            # Horizontal bar plot
            y_pos = np.arange(len(words))
            ax.barh(y_pos, weights)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_title(f"Topic {topic.get('topic_id', i)}")
            ax.set_xlabel('Weight')
        
        # Hide empty subplots
        for i in range(n_topics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_entity_distribution(self, 
                                entity_stats: Dict,
                                title: str = 'Named Entity Distribution') -> plt.Figure:
        """
        Plot named entity distribution
        
        Args:
            entity_stats: Dictionary with entity statistics
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Entity type distribution
        entity_types = entity_stats.get('most_common_types', [])
        if entity_types:
            types, counts = zip(*entity_types)
            ax1.bar(types, counts)
            ax1.set_title('Entity Types Distribution')
            ax1.set_xlabel('Entity Type')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
        
        # Most common entities
        common_entities = entity_stats.get('most_common_texts', [])
        if common_entities:
            entities, counts = zip(*common_entities[:10])  # Top 10
            ax2.barh(range(len(entities)), counts)
            ax2.set_yticks(range(len(entities)))
            ax2.set_yticklabels(entities)
            ax2.invert_yaxis()
            ax2.set_title('Most Common Entities')
            ax2.set_xlabel('Count')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, 
                             cm: np.ndarray,
                             labels: List[str],
                             title: str = 'Confusion Matrix') -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, 
                             comparison_df: pd.DataFrame,
                             metric: str = 'f1_score',
                             title: str = 'Model Comparison') -> plt.Figure:
        """
        Plot model comparison
        
        Args:
            comparison_df: DataFrame with model comparison results
            metric: Metric to compare
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if metric in comparison_df.columns:
            models = comparison_df['model'] if 'model' in comparison_df.columns else comparison_df.index
            scores = comparison_df[metric]
            
            bars = ax.bar(models, scores)
            ax.set_title(f'{title} - {metric.replace("_", " ").title()}')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                if not pd.isna(score):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_sentiment_plot(self, 
                                        df: pd.DataFrame,
                                        text_column: str = 'text',
                                        sentiment_column: str = 'sentiment',
                                        confidence_column: str = 'confidence') -> go.Figure:
        """
        Create interactive sentiment analysis plot
        
        Args:
            df: DataFrame with sentiment data
            text_column: Name of text column
            sentiment_column: Name of sentiment column
            confidence_column: Name of confidence column
            
        Returns:
            Plotly figure
        """
        fig = px.scatter(
            df, 
            x=range(len(df)), 
            y=confidence_column,
            color=sentiment_column,
            hover_data=[text_column],
            title='Interactive Sentiment Analysis Results'
        )
        
        fig.update_layout(
            xaxis_title='Document Index',
            yaxis_title='Confidence Score',
            hovermode='closest'
        )
        
        return fig
    
    def save_plot(self, 
                 fig: plt.Figure, 
                 filename: str, 
                 save_dir: str = 'visualizations') -> str:
        """
        Save plot to file
        
        Args:
            fig: Matplotlib figure
            filename: Filename to save
            save_dir: Directory to save in
            
        Returns:
            Full path to saved file
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        plt.close(fig)
        
        return filepath
