"""
NLP Analysis Project

A comprehensive NLP analysis toolkit for sentiment analysis, topic modeling,
named entity recognition, and text summarization.
"""

__version__ = "1.0.0"
__author__ = "Data Science Intern Candidate"

from . import config
from . import data_preprocessing
from . import sentiment_analysis
from . import topic_modeling
from . import ner
from . import summarization
from . import evaluation
from . import visualization

__all__ = [
    "config",
    "data_preprocessing", 
    "sentiment_analysis",
    "topic_modeling",
    "ner", 
    "summarization",
    "evaluation",
    "visualization"
]
