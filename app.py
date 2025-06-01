#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Agent - Advanced Text Analysis Platform
A comprehensive NLP analysis tool with multiple features
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
import sys
import pickle
import base64

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="NLP Agent - YASSINE BEN ACHA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-header {
        font-size: 2rem;
        color: #2e8b57;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .contact-info {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .contact-item {
        margin: 0.5rem 0;
        font-size: 1rem;
        color: #333;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #666;
        text-align: center;
        padding: 8px 0;
        font-size: 0.85rem;
        border-top: 1px solid #dee2e6;
        z-index: 999;
    }
    .footer a:hover {
        color: #1f77b4 !important;
        text-decoration: underline !important;
    }
    .contact-info a:hover {
        color: #1f77b4 !important;
        text-decoration: underline !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""

def load_models():
    """Load pre-trained models"""
    models = {}
    
    try:
        # Load LDA model
        if os.path.exists('models/lda_model.pkl'):
            with open('models/lda_model.pkl', 'rb') as f:
                models['lda'] = pickle.load(f)
            st.sidebar.success("✅ LDA model loaded")
        
        # Load TF-IDF vectorizer
        if os.path.exists('models/tfidf_vectorizer.pkl'):
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                models['tfidf'] = pickle.load(f)
            st.sidebar.success("✅ TF-IDF vectorizer loaded")
            
    except Exception as e:
        st.sidebar.warning(f"⚠️ Model loading error: {str(e)}")
    
    return models

def import_nlp_modules():
    """Import NLP modules with error handling"""
    modules = {}

    try:
        from data_preprocessing import TextPreprocessor
        # Try to create preprocessor with minimal dependencies
        modules['preprocessor'] = TextPreprocessor(
            remove_stopwords=False,  # Avoid NLTK dependency issues
            remove_punctuation=True,
            lowercase=True
        )
        st.sidebar.success("✅ Text Preprocessor loaded")
    except Exception as e:
        # Try simplified preprocessor
        try:
            from simple_data_preprocessing import TextPreprocessor
            modules['preprocessor'] = TextPreprocessor(
                remove_stopwords=True,
                remove_punctuation=True,
                lowercase=True
            )
            st.sidebar.success("✅ Simple Text Preprocessor loaded")
        except Exception as e2:
            st.sidebar.warning(f"⚠️ Text Preprocessor not available: {str(e)[:30]}...")
            modules['preprocessor'] = None

    try:
        from sentiment_analysis import SentimentAnalyzer
        modules['sentiment'] = SentimentAnalyzer()
        st.sidebar.success("✅ Sentiment Analyzer loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Sentiment Analyzer not available: {str(e)[:50]}...")
        modules['sentiment'] = None

    try:
        import spacy
        # Try different spaCy model loading approaches
        try:
            modules['nlp'] = spacy.load("en_core_web_sm")
            st.sidebar.success("✅ spaCy NER loaded")
        except OSError:
            # Try alternative model names
            try:
                modules['nlp'] = spacy.load("en")
                st.sidebar.success("✅ spaCy NER loaded (en model)")
            except OSError:
                st.sidebar.info("ℹ️ Using pattern-based NER (spaCy model not available)")
                modules['nlp'] = None
    except Exception as e:
        st.sidebar.info("ℹ️ Using pattern-based NER (spaCy not installed)")
        modules['nlp'] = None

    return modules

def create_download_link(df, filename, text="Download CSV"):
    """Create a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 NLP Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Comprehensive Natural Language Processing Analysis Tool**")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Load models and modules
    with st.sidebar.expander("🔧 System Status", expanded=False):
        models = load_models()
        modules = import_nlp_modules()
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Choose Analysis Tool:",
        [
            "🏠 Home",
            "📊 Data Exploration", 
            "😊 Sentiment Analysis",
            "🎯 Topic Modeling",
            "🏷️ Named Entity Recognition",
            "📝 Text Summarization"
        ]
    )
    
    # Text input section (common for all pages)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Text Input")
    
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Type/Paste Text", "Upload File", "Use Sample Data"]
    )
    
    current_text = ""
    
    if input_method == "Type/Paste Text":
        current_text = st.sidebar.text_area(
            "Enter your text:",
            height=150,
            placeholder="Paste your text here for analysis..."
        )
    
    elif input_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=['txt', 'csv']
        )
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                current_text = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                st.sidebar.write("CSV Preview:")
                st.sidebar.dataframe(df.head())
                text_column = st.sidebar.selectbox("Select text column:", df.columns)
                current_text = " ".join(df[text_column].astype(str).tolist())
    
    elif input_method == "Use Sample Data":
        sample_texts = {
            "Technology News": "Apple Inc. announced revolutionary new features for the iPhone at their headquarters in Cupertino, California. The company's CEO Tim Cook highlighted artificial intelligence capabilities and improved battery life.",
            "Movie Review": "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and the cinematography was breathtaking. I would definitely recommend it to anyone who enjoys great storytelling.",
            "Business Article": "The stock market experienced significant volatility today as investors reacted to the Federal Reserve's latest interest rate decision. Technology stocks led the decline while energy sectors showed resilience.",
            "Health News": "Recent studies published in the Journal of Medicine show promising results for a new treatment approach. Researchers at Harvard University conducted extensive trials with positive outcomes.",
            "Climate Report": "Global temperatures continue to rise according to the latest climate data. Scientists emphasize the urgent need for renewable energy adoption and carbon emission reduction strategies."
        }
        
        selected_sample = st.sidebar.selectbox("Choose sample text:", list(sample_texts.keys()))
        current_text = sample_texts[selected_sample]
        st.sidebar.text_area("Sample text:", current_text, height=100, disabled=True)
    
    # Update session state
    st.session_state.current_text = current_text
    
    # Contact information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
        <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">🤖 AI Engineer & ML Expert</h4>
        <div style="font-size: 0.9rem;">
            <div><strong>YASSINE BEN ACHA</strong></div>
            <div>📱 <a href="https://wa.me/212696545641" target="_blank" style="text-decoration: none; color: inherit;">+212 696 545 641 (WhatsApp)</a></div>
            <div>✉️ <a href="mailto:yassinebenacha1@gmail.com" style="text-decoration: none; color: inherit;">yassinebenacha1@gmail.com</a></div>
            <div>🌐 <a href="https://portfolio-pro-phi.vercel.app" target="_blank" style="text-decoration: none; color: inherit;">Mon portfolio</a></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Exploration":
        show_data_exploration(current_text, modules)
    elif page == "😊 Sentiment Analysis":
        show_sentiment_analysis(current_text, modules)
    elif page == "🎯 Topic Modeling":
        show_topic_modeling(current_text, models, modules)
    elif page == "🏷️ Named Entity Recognition":
        show_ner_analysis(current_text, modules)
    elif page == "📝 Text Summarization":
        show_text_summarization(current_text, modules)

    # Add footer with copyright
    st.markdown("""
    <div class="footer">
        © 2025 YASSINE BEN ACHA | NLP Agent - Advanced Text Analysis Platform |
        📱 <a href="https://wa.me/212696545641" target="_blank" style="text-decoration: none; color: inherit;">WhatsApp</a> |
        ✉️ <a href="mailto:yassinebenacha1@gmail.com" style="text-decoration: none; color: inherit;">yassinebenacha1@gmail.com</a> |
        🌐 <a href="https://portfolio-pro-phi.vercel.app" target="_blank" style="text-decoration: none; color: inherit;">Portfolio</a>
    </div>
    """, unsafe_allow_html=True)

def show_home_page():
    """Display home page"""

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ## Welcome to NLP Agent! 🚀

        This comprehensive tool provides advanced Natural Language Processing capabilities
        for analyzing and understanding text data.

        ### 🛠️ Available Features:
        """)

        features = [
            ("📊 Data Exploration", "Analyze text statistics, word frequencies, and generate word clouds"),
            ("😊 Sentiment Analysis", "Classify text sentiment as positive, negative, or neutral"),
            ("🎯 Topic Modeling", "Discover hidden topics using LDA and advanced algorithms"),
            ("🏷️ Named Entity Recognition", "Extract people, organizations, locations, and other entities"),
            ("📝 Text Summarization", "Generate concise summaries using extractive methods")
        ]

        for feature, description in features:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{feature}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        ### 🚀 Getting Started:
        1. **Choose a tool** from the sidebar navigation
        2. **Input your text** using one of the three methods:
           - Type or paste text directly
           - Upload a text or CSV file
           - Use provided sample data
        3. **Analyze** and explore the results with interactive visualizations
        4. **Download** results for further use

        ### 💡 Tips:
        - For best results, use text with at least 100 words
        - Multiple documents can be analyzed by uploading CSV files
        - All processing is done locally for privacy
        """)

        # Contact Information
        st.markdown("""
        <div class="contact-info">
            <h3 style="color: #17a2b8; margin-bottom: 1rem;">🤖 AI Engineer & Machine Learning Expert</h3>
            <div class="contact-item"><strong>🧑‍💼 YASSINE BEN ACHA</strong></div>
            <div class="contact-item">📱 <a href="https://wa.me/212696545641" target="_blank" style="text-decoration: none; color: inherit;">+212 696 545 641 (WhatsApp)</a></div>
            <div class="contact-item">✉️ <a href="mailto:yassinebenacha1@gmail.com" style="text-decoration: none; color: inherit;">yassinebenacha1@gmail.com</a></div>
            <div class="contact-item">🌐 <a href="https://portfolio-pro-phi.vercel.app" target="_blank" style="text-decoration: none; color: inherit;">Mon portfolio</a></div>
        </div>
        """, unsafe_allow_html=True)

def show_data_exploration(text, modules):
    """Data exploration page"""
    st.markdown('<h2 class="feature-header">📊 Data Exploration</h2>', unsafe_allow_html=True)

    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
        return

    with st.spinner("🔄 Analyzing text..."):
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📏 Characters", len(text))
        with col2:
            words = text.split()
            st.metric("📝 Words", len(words))
        with col3:
            sentences = text.count('.') + text.count('!') + text.count('?')
            st.metric("📄 Sentences", max(1, sentences))
        with col4:
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            st.metric("📊 Avg Word Length", f"{avg_word_length:.1f}")

        # Word frequency analysis
        st.subheader("🔤 Word Frequency Analysis")

        # Simple word frequency (fallback if modules not available)
        words_clean = [word.lower().strip('.,!?";') for word in words if len(word) > 2]
        word_freq = pd.Series(words_clean).value_counts().head(20)

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart
            fig = px.bar(
                x=word_freq.values,
                y=word_freq.index,
                orientation='h',
                title="Top 20 Most Frequent Words",
                labels={'x': 'Frequency', 'y': 'Words'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Word cloud simulation (text-based if wordcloud not available)
            st.subheader("☁️ Word Cloud")
            try:
                from wordcloud import WordCloud
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(text)

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except ImportError:
                # Text-based word cloud
                st.write("**Most Frequent Words:**")
                for word, freq in word_freq.head(10).items():
                    st.write(f"{'█' * min(freq, 20)} {word} ({freq})")

        # Download results
        st.subheader("💾 Download Results")
        results_df = pd.DataFrame({
            'Word': word_freq.index,
            'Frequency': word_freq.values
        })

        st.markdown(create_download_link(results_df, "word_frequency.csv", "📥 Download Word Frequencies"), unsafe_allow_html=True)

def show_sentiment_analysis(text, modules):
    """Sentiment analysis page"""
    st.markdown('<h2 class="feature-header">😊 Sentiment Analysis</h2>', unsafe_allow_html=True)

    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
        return

    with st.spinner("🔄 Analyzing sentiment..."):
        # Simple sentiment analysis fallback
        st.info("ℹ️ Using basic sentiment analysis")

        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            sentiment = "positive"
            confidence = min(0.8, 0.5 + (pos_count - neg_count) * 0.1)
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = min(0.8, 0.5 + (neg_count - pos_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5

        col1, col2 = st.columns(2)
        with col1:
            emoji_map = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
            st.markdown(f"<h1 style='text-align: center;'>{emoji_map[sentiment]}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>{sentiment.title()}</h3>", unsafe_allow_html=True)
        with col2:
            st.metric("🎯 Confidence", f"{confidence:.2f}")

def show_topic_modeling(text, models, modules):
    """Topic modeling page"""
    st.markdown('<h2 class="feature-header">🎯 Topic Modeling</h2>', unsafe_allow_html=True)

    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
        return

    st.info("ℹ️ Using simple topic analysis")

    # Simple keyword extraction
    words = text.lower().split()
    words_clean = [word.strip('.,!?";') for word in words if len(word) > 3]
    word_freq = pd.Series(words_clean).value_counts()

    # Group words into simple "topics" based on common themes
    tech_words = ['technology', 'computer', 'software', 'digital', 'internet', 'data', 'artificial', 'intelligence']
    business_words = ['business', 'market', 'economy', 'financial', 'investment', 'company', 'revenue', 'profit']
    health_words = ['health', 'medical', 'treatment', 'patient', 'doctor', 'medicine', 'hospital', 'disease']

    topics = {
        'Technology': sum(word_freq.get(word, 0) for word in tech_words),
        'Business': sum(word_freq.get(word, 0) for word in business_words),
        'Health': sum(word_freq.get(word, 0) for word in health_words),
        'General': len(words_clean) - sum(word_freq.get(word, 0) for word in tech_words + business_words + health_words)
    }

    # Normalize
    total = sum(topics.values())
    if total > 0:
        topics = {k: v/total for k, v in topics.items()}

    # Display results
    topic_df = pd.DataFrame(list(topics.items()), columns=['Topic', 'Score'])
    topic_df = topic_df[topic_df['Score'] > 0].sort_values('Score', ascending=False)

    if not topic_df.empty:
        fig = px.pie(topic_df, values='Score', names='Topic', title="Simple Topic Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Topic Scores")
        for _, row in topic_df.iterrows():
            st.metric(f"🎯 {row['Topic']}", f"{row['Score']:.2f}")

def show_ner_analysis(text, modules):
    """Named Entity Recognition page"""
    st.markdown('<h2 class="feature-header">🏷️ Named Entity Recognition</h2>', unsafe_allow_html=True)

    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
        return

    st.info("ℹ️ Using simple pattern-based entity extraction")

    # Simple pattern-based entity extraction
    import re
    entities = []

    # Simple patterns
    person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    org_pattern = r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b'

    persons = re.findall(person_pattern, text)
    orgs = re.findall(org_pattern, text)

    for person in persons:
        entities.append((person, 'PERSON'))
    for org in orgs:
        entities.append((org, 'ORG'))

    if entities:
        entity_df = pd.DataFrame(entities, columns=['Text', 'Type'])
        st.dataframe(entity_df, use_container_width=True)
    else:
        st.warning("⚠️ No entities found in the text.")

def show_text_summarization(text, modules):
    """Text summarization page"""
    st.markdown('<h2 class="feature-header">📝 Text Summarization</h2>', unsafe_allow_html=True)

    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
        return

    # Simple extractive summarization
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

    if len(sentences) < 3:
        st.warning("⚠️ Text too short for summarization. Please provide at least 3 sentences.")
        return

    # Simple scoring based on word frequency
    words = text.lower().split()
    word_freq = pd.Series(words).value_counts()

    sentence_scores = []
    for sentence in sentences:
        score = sum(word_freq.get(word.lower(), 0) for word in sentence.split())
        sentence_scores.append((sentence, score))

    # Get top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:3]
    summary = '. '.join([sent[0] for sent in top_sentences])

    st.subheader("📄 Summary")
    st.write(summary)

    st.subheader("📊 Summary Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Sentences", len(sentences))
        st.metric("Summary Sentences", len(top_sentences))
    with col2:
        compression_ratio = len(top_sentences) / len(sentences)
        st.metric("Compression Ratio", f"{compression_ratio:.2f}")

if __name__ == "__main__":
    main()
