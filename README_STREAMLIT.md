# 🤖 NLP Agent - Streamlit Interface

A comprehensive Natural Language Processing analysis tool with an intuitive web interface built with Streamlit.

## 🌟 Features

### 📊 Data Exploration
- **Text Statistics**: Character count, word count, sentence count, average word length
- **Word Frequency Analysis**: Interactive bar charts of most frequent words
- **Word Clouds**: Visual representation of text content (if wordcloud library available)
- **Download Results**: Export word frequency data as CSV

### 😊 Sentiment Analysis
- **Multiple Methods**: TextBlob, VADER, or simple rule-based analysis
- **Confidence Scores**: Gauge visualization of prediction confidence
- **Sentence-level Analysis**: Breakdown of sentiment for individual sentences
- **Visual Indicators**: Emoji and color-coded sentiment display

### 🎯 Topic Modeling
- **LDA Integration**: Uses pre-trained LDA models from your notebooks
- **Topic Distribution**: Interactive charts showing topic probabilities
- **Word Importance**: Top words for each discovered topic
- **Fallback Analysis**: Simple keyword-based topic detection when models unavailable

### 🏷️ Named Entity Recognition
- **spaCy Integration**: Advanced NER using spaCy models
- **Entity Highlighting**: Color-coded entity visualization in text
- **Entity Statistics**: Distribution charts and summary metrics
- **Pattern-based Fallback**: Simple regex patterns when spaCy unavailable
- **Filtering**: Interactive entity type filtering

### 📝 Text Summarization
- **Multiple Methods**: Frequency-based, TF-IDF, and transformer-based (BART)
- **Customizable Length**: Adjustable number of sentences in summary
- **Compression Metrics**: Detailed analysis of summary quality
- **Side-by-side View**: Original text and summary comparison
- **Quality Analysis**: Sentence and word reduction statistics

## 🚀 Quick Start

### Option 1: Simple Launch
```bash
python run_app.py
```

### Option 2: Manual Launch
```bash
# Install requirements
pip install -r requirements_streamlit.txt

# Launch Streamlit
streamlit run app.py
```

### Option 3: Direct Streamlit
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn nltk textblob
streamlit run app.py
```

## 📋 Requirements

### Core Requirements (Always Needed)
- `streamlit>=1.28.0`
- `pandas>=1.5.0`
- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`
- `plotly>=5.0.0`
- `scikit-learn>=1.1.0`
- `nltk>=3.8`
- `textblob>=0.17.0`

### Optional Advanced Features
- `spacy>=3.4.0` + `en_core_web_sm` model (for advanced NER)
- `transformers>=4.20.0` + `torch>=1.12.0` (for BART summarization)
- `wordcloud>=1.9.0` (for word cloud visualization)

## 🎛️ Interface Guide

### 🧭 Navigation
- **Sidebar Menu**: Choose between different NLP tools
- **System Status**: View loaded models and available features
- **Text Input Options**: Type, upload file, or use sample data

### 📝 Text Input Methods

1. **Type/Paste Text**: Direct text input in sidebar
2. **Upload File**: Support for .txt and .csv files
3. **Sample Data**: Pre-loaded examples for each domain

### 📊 Results & Visualizations
- **Interactive Charts**: Plotly-based visualizations with zoom/pan
- **Downloadable Results**: CSV export for all analysis results
- **Real-time Processing**: Instant analysis as you change inputs
- **Error Handling**: Graceful fallbacks when advanced features unavailable

## 🔧 Configuration

### Model Integration
The app automatically detects and uses pre-trained models from your `models/` folder:
- `models/lda_model.pkl` - LDA topic model
- `models/tfidf_vectorizer.pkl` - TF-IDF vectorizer

### Custom Modules
The app imports your custom NLP modules from `src/`:
- `src/data_preprocessing.py`
- `src/sentiment_analysis.py`
- `src/topic_modeling.py`
- `src/ner.py`
- `src/summarization.py`

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```
Solution: The app includes fallback methods for all features
- Missing spaCy → Simple regex-based NER
- Missing transformers → TF-IDF summarization
- Missing wordcloud → Text-based word frequency display
```

**2. Model Loading Errors**
```
Solution: Check that model files exist in models/ folder
- Run your training notebooks first to generate models
- App will use simple alternatives if models unavailable
```

**3. Streamlit Not Found**
```bash
pip install streamlit
# or
python run_app.py  # Auto-installs requirements
```

### Performance Tips
- **Large Texts**: App automatically truncates very long texts for processing
- **Memory Usage**: Advanced models (BART) require significant RAM
- **Processing Time**: First run may be slower due to model loading

## 📱 Usage Examples

### Example 1: News Article Analysis
1. Select "📊 Data Exploration"
2. Choose "Use Sample Data" → "Technology News"
3. View word frequency and statistics
4. Switch to "😊 Sentiment Analysis" for sentiment classification
5. Try "🎯 Topic Modeling" to discover themes

### Example 2: Document Summarization
1. Select "📝 Text Summarization"
2. Upload a long document or paste text (>50 words)
3. Adjust summary length with slider
4. Compare different summarization methods
5. Download results for further use

### Example 3: Entity Extraction
1. Select "🏷️ Named Entity Recognition"
2. Input text with names, organizations, locations
3. View highlighted entities in text
4. Filter by entity type
5. Export entity list as CSV

## 🎨 Customization

### Adding New Features
1. Create new function in `app.py`
2. Add menu item in sidebar navigation
3. Follow existing pattern for error handling and fallbacks

### Styling
- Modify CSS in the `st.markdown()` sections
- Customize colors in entity highlighting
- Adjust chart themes in Plotly configurations

## 📞 Support

### Getting Help
1. Check console output for error messages
2. Verify all requirements are installed
3. Ensure model files exist if using advanced features
4. Try fallback methods if advanced features fail

### Feature Requests
The interface is designed to be modular and extensible. New NLP features can be easily added following the existing patterns.

---

**🎉 Enjoy exploring your NLP Agent with this intuitive interface!**
