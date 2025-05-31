"""
NLP Agent - Streamlit Application

Essential NLP modules for the Streamlit web application.
"""

__version__ = "1.0.0"
__author__ = "NLP Agent Developer"

from . import data_preprocessing
from . import sentiment_analysis
from . import simple_data_preprocessing

__all__ = [
    "data_preprocessing",
    "sentiment_analysis",
    "simple_data_preprocessing"
]
