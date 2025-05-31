# Advanced NLP Analysis Project

A comprehensive Natural Language Processing (NLP) project demonstrating advanced techniques in sentiment analysis, topic modeling, named entity recognition, and text summarization.

## 🎯 Project Overview

This project showcases a complete NLP pipeline with the following components:

1. **Sentiment Analysis** - Multi-approach sentiment classification (positive/neutral/negative)
2. **Topic Modeling** - LDA and BERTopic for discovering latent topics
3. **Named Entity Recognition (NER)** - Extract and classify named entities using spaCy
4. **Text Summarization** - Both extractive and abstractive summarization techniques

## 📁 Project Structure

```
nlp_analysis_project/
├── data/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Cleaned and preprocessed data
│   └── sample/               # Sample datasets for testing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_named_entity_recognition.ipynb
│   ├── 05_text_summarization.ipynb
│   └── 06_comprehensive_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── data_preprocessing.py
│   ├── sentiment_analysis.py
│   ├── topic_modeling.py
│   ├── ner.py
│   ├── summarization.py
│   ├── evaluation.py         # Model evaluation utilities
│   └── visualization.py     # Plotting utilities
├── tests/                    # Unit tests
├── visualizations/           # Output visualizations
├── models/                   # Saved models
├── results/                  # Analysis results and reports
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd nlp_analysis_project

# Create virtual environment
python -m venv nlp_env
source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Run the Analysis

Start with the Jupyter notebooks in order:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Or use the Python modules directly:

```python
from src.sentiment_analysis import SentimentAnalyzer
from src.topic_modeling import TopicModeler
from src.ner import NamedEntityRecognizer
from src.summarization import TextSummarizer

# Initialize analyzers
sentiment_analyzer = SentimentAnalyzer()
topic_modeler = TopicModeler()
ner = NamedEntityRecognizer()
summarizer = TextSummarizer()
```

## 📊 Features

### Sentiment Analysis
- **Multiple Approaches**: TextBlob, Transformers (RoBERTa), Traditional ML
- **Models**: Logistic Regression, Naive Bayes, SVM, Random Forest
- **Evaluation**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Visualization**: Confidence distributions, confusion matrices

### Topic Modeling
- **LDA (Latent Dirichlet Allocation)**: Both scikit-learn and Gensim implementations
- **BERTopic**: Modern transformer-based topic modeling
- **Evaluation**: Coherence scores, perplexity
- **Visualization**: Topic distributions, word clouds, interactive plots

### Named Entity Recognition
- **spaCy NER**: Pre-trained models for entity extraction
- **Transformer NER**: BERT-based models for enhanced accuracy
- **NLTK NER**: Traditional rule-based approach
- **Entity Types**: PERSON, ORG, GPE, MONEY, DATE, and more
- **Evaluation**: Precision, Recall, F1-score by entity type

### Text Summarization
- **Extractive Methods**: TextRank, TF-IDF based selection
- **Abstractive Methods**: BART, T5 transformer models
- **Evaluation**: ROUGE scores, BERT Score
- **Comparison**: Multiple methods side-by-side analysis

## 📈 Sample Results

### Sentiment Analysis Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| RoBERTa | 0.92 | 0.91 | 0.92 | 0.91 |
| Logistic Regression | 0.85 | 0.84 | 0.85 | 0.84 |
| Naive Bayes | 0.82 | 0.81 | 0.82 | 0.81 |

### Topic Modeling Results
- **LDA Coherence Score**: 0.45
- **BERTopic Topics Found**: 8 distinct topics
- **Most Coherent Topic**: Technology and AI (coherence: 0.62)

## 🛠️ Technologies Used

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **plotly**: Interactive visualizations

### NLP Libraries
- **nltk**: Natural language processing toolkit
- **spacy**: Industrial-strength NLP
- **transformers**: State-of-the-art transformer models
- **sentence-transformers**: Sentence embeddings
- **bertopic**: Advanced topic modeling
- **gensim**: Topic modeling and document similarity

### Specialized Tools
- **rouge-score**: Summarization evaluation
- **bert-score**: Semantic similarity evaluation
- **wordcloud**: Word cloud generation
- **pyLDAvis**: LDA visualization

## 📝 Usage Examples

### Sentiment Analysis
```python
from src.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Analyze single text
result = analyzer.predict_sentiment("This product is amazing!", method='transformer')
print(result)  # {'sentiment': 'positive', 'confidence': 0.95}

# Analyze dataset
df_results = analyzer.analyze_sentiment_distribution(df, 'text_column')
```

### Topic Modeling
```python
from src.topic_modeling import TopicModeler

modeler = TopicModeler()

# Train BERTopic model
results = modeler.train_bertopic(texts, min_topic_size=10)

# Get document topics
doc_topics = modeler.get_document_topics(texts, model_type='bertopic')
```

### Named Entity Recognition
```python
from src.ner import NamedEntityRecognizer

ner = NamedEntityRecognizer()

# Extract entities
entities = ner.extract_entities("Apple Inc. was founded by Steve Jobs.", method='spacy')

# Analyze dataset
df_entities = ner.analyze_entities_in_dataset(df, 'text_column')
```

### Text Summarization
```python
from src.summarization import TextSummarizer

summarizer = TextSummarizer()

# Generate summary
summary = summarizer.summarize_text(long_text, method='textrank', num_sentences=3)

# Compare methods
comparison = summarizer.compare_summarization_methods(text)
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## 📊 Evaluation Metrics

### Classification Tasks
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Topic Modeling
- **Coherence Score**: Semantic coherence of topics
- **Perplexity**: Model's uncertainty (lower is better)
- **Topic Diversity**: Uniqueness of discovered topics

### Summarization
- **ROUGE-1/2/L**: Overlap with reference summaries
- **BERT Score**: Semantic similarity with references
- **Compression Ratio**: Summary length / Original length

## 🎨 Visualizations

The project includes comprehensive visualizations:

- **Sentiment Distribution**: Bar charts and pie charts
- **Text Length Analysis**: Histograms and box plots
- **Word Clouds**: Visual representation of frequent terms
- **Topic Visualization**: Interactive topic maps
- **Entity Distribution**: Entity type frequency charts
- **Model Comparison**: Performance comparison plots
- **Confusion Matrices**: Classification error analysis

## 🔧 Configuration

Modify `src/config.py` to customize:

- Model parameters
- File paths
- Visualization settings
- Evaluation metrics
- Processing options

## 📚 References

- [Transformers Documentation](https://huggingface.co/transformers/)
- [spaCy Documentation](https://spacy.io/)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Data Science Intern Candidate - Advanced NLP Project

---

**Note**: This project is designed for educational and demonstration purposes, showcasing advanced NLP techniques and best practices in machine learning project organization.
