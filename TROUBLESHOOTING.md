# 🔧 NLP Agent Troubleshooting Guide

## 🚨 Common Issues and Solutions

### Issue 1: "Text Preprocessor not available"

**Cause**: Missing NLTK dependencies or spaCy model

**Solutions**:
```bash
# Option 1: Quick fix
python quick_fix.py

# Option 2: Manual installation
pip install nltk spacy
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm

# Option 3: Use simplified version (already included)
# The app will automatically use simple_data_preprocessing.py as fallback
```

### Issue 2: "spaCy NER not available"

**Cause**: spaCy or English model not installed

**Solutions**:
```bash
# Install spaCy and model
pip install spacy
python -m spacy download en_core_web_sm

# Alternative: The app includes pattern-based NER fallback
# It will work without spaCy, just with reduced accuracy
```

### Issue 3: "Sentiment Analyzer not available"

**Cause**: Missing TextBlob or other dependencies

**Solutions**:
```bash
# Install TextBlob
pip install textblob

# Download TextBlob corpora
python -c "import textblob; textblob.download_corpora()"

# Alternative: The app includes simple sentiment analysis fallback
```

### Issue 4: Import errors when running Streamlit

**Cause**: Missing core packages

**Solutions**:
```bash
# Install all requirements
pip install -r requirements_streamlit.txt

# Or install manually
pip install streamlit pandas numpy matplotlib plotly scikit-learn
```

### Issue 5: Models not loading (LDA, TF-IDF)

**Cause**: Model files don't exist in models/ folder

**Solutions**:
1. Run the training notebooks first:
   ```bash
   jupyter notebook notebooks/03_topic_modeling_fixed.ipynb
   ```
2. Or the app will use simple topic analysis as fallback

### Issue 6: Streamlit app won't start

**Cause**: Various dependency issues

**Solutions**:
```bash
# Check Streamlit installation
pip install streamlit

# Launch with verbose output
streamlit run app.py --logger.level=debug

# Check for port conflicts
streamlit run app.py --server.port=8502
```

## 🛠️ Step-by-Step Fix Process

### Step 1: Run Quick Fix
```bash
python quick_fix.py
```

### Step 2: Test Dependencies
```bash
python test_streamlit_app.py
```

### Step 3: Launch App
```bash
streamlit run app.py
```

### Step 4: Check Status
- Look at the sidebar "System Status" section
- Green checkmarks = working
- Yellow warnings = using fallbacks
- Red errors = need fixing

## 🔍 Debugging Tips

### Check Python Environment
```bash
# Check Python version (should be 3.7+)
python --version

# Check installed packages
pip list | grep -E "(streamlit|pandas|numpy|spacy|nltk)"

# Check if models exist
ls -la models/
```

### Check Module Imports
```python
# Test imports manually
import streamlit as st
import pandas as pd
import numpy as np

# Test custom modules
import sys
sys.path.append('src')
from simple_data_preprocessing import TextPreprocessor
```

### Check Streamlit Logs
- Look at the terminal where you ran `streamlit run app.py`
- Check for specific error messages
- Note which modules failed to load

## 🎯 Fallback Features

The app is designed to work even with missing dependencies:

### Text Preprocessing
- **Full version**: Uses NLTK, spaCy, advanced tokenization
- **Fallback**: Simple regex-based preprocessing

### Sentiment Analysis
- **Full version**: TextBlob, VADER, transformer models
- **Fallback**: Simple keyword-based sentiment

### Topic Modeling
- **Full version**: Pre-trained LDA models
- **Fallback**: Simple keyword clustering

### Named Entity Recognition
- **Full version**: spaCy with en_core_web_sm model
- **Fallback**: Regex pattern matching

### Text Summarization
- **Full version**: BART transformer, TF-IDF
- **Fallback**: Frequency-based extractive summarization

## 📞 Getting Help

### If Nothing Works
1. **Check Python version**: Must be 3.7 or higher
2. **Create new environment**:
   ```bash
   python -m venv nlp_env
   source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate
   pip install streamlit pandas numpy matplotlib plotly
   streamlit run app.py
   ```
3. **Use minimal version**: The app will work with just basic packages

### Error Messages to Look For
- `ModuleNotFoundError`: Missing package → install with pip
- `OSError: [E050]`: spaCy model missing → download with spacy download
- `LookupError`: NLTK data missing → download with nltk.download()
- `FileNotFoundError`: Model files missing → run training notebooks

### Performance Issues
- **Slow loading**: First run downloads models, subsequent runs faster
- **Memory issues**: Advanced models (BART) need 4GB+ RAM
- **Processing time**: Large texts are automatically truncated

## ✅ Success Indicators

When everything works correctly, you should see:
- ✅ LDA model loaded
- ✅ TF-IDF vectorizer loaded  
- ✅ Text Preprocessor loaded
- ✅ Sentiment Analyzer loaded
- ✅ spaCy NER loaded

Even with some ❌ or ⚠️ indicators, the app will still work with reduced functionality.

## 🚀 Quick Start (Minimal Setup)

If you just want to get started quickly:

```bash
# Install only essentials
pip install streamlit pandas numpy matplotlib plotly scikit-learn

# Run app (will use fallback methods)
streamlit run app.py
```

The app is designed to be robust and will provide a good experience even without all advanced features!
