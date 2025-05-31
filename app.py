#!/usr/bin/env python3
"""
NLP Agent - Streamlit Interface
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
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="NLP Agent",
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
                st.sidebar.warning("⚠️ spaCy model not found - using pattern-based NER")
                modules['nlp'] = None
    except Exception as e:
        st.sidebar.warning(f"⚠️ spaCy NER not available: {str(e)[:30]}...")
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

        # Try to use the sentiment analyzer module
        if modules.get('sentiment'):
            try:
                result = modules['sentiment'].textblob_sentiment(text)
                sentiment = result['sentiment']
                confidence = result['confidence']

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Sentiment emoji
                    emoji_map = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
                    st.markdown(f"<h1 style='text-align: center;'>{emoji_map.get(sentiment, '❓')}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>{sentiment.title()}</h3>", unsafe_allow_html=True)

                with col2:
                    st.metric("🎯 Confidence", f"{confidence:.2f}")

                with col3:
                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence"},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 1], 'color': "gray"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.9}}))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Sentiment analysis error: {str(e)}")

        else:
            # Fallback simple sentiment analysis
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

        # Detailed analysis
        st.subheader("📊 Detailed Analysis")

        # Split text into sentences for sentence-level analysis
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

        if len(sentences) > 1:
            st.write(f"**Analyzing {len(sentences)} sentences:**")

            sentence_sentiments = []
            for i, sentence in enumerate(sentences[:10]):  # Limit to first 10 sentences
                # Simple sentence sentiment
                pos_words_in_sent = sum(1 for word in ['good', 'great', 'excellent', 'love'] if word in sentence.lower())
                neg_words_in_sent = sum(1 for word in ['bad', 'terrible', 'hate', 'awful'] if word in sentence.lower())

                if pos_words_in_sent > neg_words_in_sent:
                    sent_sentiment = "Positive"
                    color = "🟢"
                elif neg_words_in_sent > pos_words_in_sent:
                    sent_sentiment = "Negative"
                    color = "🔴"
                else:
                    sent_sentiment = "Neutral"
                    color = "🟡"

                st.write(f"{color} **Sentence {i+1}:** {sent_sentiment}")
                st.write(f"   _{sentence[:100]}{'...' if len(sentence) > 100 else ''}_")

def show_topic_modeling(text, models, modules):
    """Topic modeling page"""
    st.markdown('<h2 class="feature-header">🎯 Topic Modeling</h2>', unsafe_allow_html=True)

    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
        return

    with st.spinner("🔄 Discovering topics..."):

        # Check if we have pre-trained models
        if models.get('lda') and models.get('tfidf'):
            try:
                # Use pre-trained models
                st.success("✅ Using pre-trained LDA model")

                # Transform text using existing vectorizer
                tfidf_matrix = models['tfidf'].transform([text])
                topic_distribution = models['lda'].transform(tfidf_matrix)[0]

                # Get topic information
                n_topics = len(topic_distribution)
                feature_names = models['tfidf'].get_feature_names_out()

                # Display topic distribution
                st.subheader("📊 Topic Distribution")

                topic_data = []
                for i, prob in enumerate(topic_distribution):
                    topic_data.append({'Topic': f'Topic {i}', 'Probability': prob})

                topic_df = pd.DataFrame(topic_data)

                # Bar chart of topic probabilities
                fig = px.bar(
                    topic_df,
                    x='Topic',
                    y='Probability',
                    title="Topic Probability Distribution",
                    color='Probability',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show top topics
                top_topics = sorted(enumerate(topic_distribution), key=lambda x: x[1], reverse=True)[:3]

                st.subheader("🏆 Top 3 Topics")

                for rank, (topic_idx, prob) in enumerate(top_topics):
                    with st.expander(f"🎯 Topic {topic_idx} (Probability: {prob:.3f})"):
                        # Get top words for this topic
                        topic_words = models['lda'].components_[topic_idx]
                        top_word_indices = topic_words.argsort()[-10:][::-1]
                        top_words = [feature_names[i] for i in top_word_indices]
                        top_weights = [topic_words[i] for i in top_word_indices]

                        # Display words
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Top Words:**")
                            for word, weight in zip(top_words, top_weights):
                                st.write(f"• {word}: {weight:.3f}")

                        with col2:
                            # Word importance chart
                            word_df = pd.DataFrame({
                                'Word': top_words[:5],
                                'Weight': top_weights[:5]
                            })
                            fig = px.bar(word_df, x='Weight', y='Word', orientation='h',
                                       title=f"Top Words in Topic {topic_idx}")
                            st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error using pre-trained models: {str(e)}")
                st.info("🔄 Falling back to simple topic analysis...")
                show_simple_topic_analysis(text)
        else:
            st.info("ℹ️ Pre-trained models not available. Using simple topic analysis.")
            show_simple_topic_analysis(text)

def show_simple_topic_analysis(text):
    """Simple topic analysis fallback"""

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

    with st.spinner("🔄 Extracting entities..."):

        entities = []

        # Try to use spaCy NER
        if modules.get('nlp'):
            try:
                doc = modules['nlp'](text)
                entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
                st.success("✅ Using spaCy NER model")

            except Exception as e:
                st.error(f"❌ spaCy NER error: {str(e)}")
                entities = []

        if not entities:
            # Fallback simple NER
            st.info("ℹ️ Using simple pattern-based entity extraction")
            entities = extract_simple_entities(text)

        if entities:
            # Display entity statistics
            entity_types = [ent[1] for ent in entities]
            entity_counts = pd.Series(entity_types).value_counts()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Entity Type Distribution")
                fig = px.bar(
                    x=entity_counts.values,
                    y=entity_counts.index,
                    orientation='h',
                    title="Entity Types Found",
                    labels={'x': 'Count', 'y': 'Entity Type'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("🏷️ Entity Summary")
                for entity_type, count in entity_counts.items():
                    st.metric(f"{entity_type}", count)

            # Display entities table
            st.subheader("📋 Extracted Entities")

            entity_df = pd.DataFrame(entities, columns=['Text', 'Type', 'Start', 'End'])

            # Add entity type filter
            selected_types = st.multiselect(
                "Filter by entity type:",
                options=entity_df['Type'].unique(),
                default=entity_df['Type'].unique()
            )

            filtered_df = entity_df[entity_df['Type'].isin(selected_types)]
            st.dataframe(filtered_df, use_container_width=True)

            # Highlighted text
            st.subheader("🎨 Highlighted Text")
            highlighted_text = highlight_entities_in_text(text, entities)
            st.markdown(highlighted_text, unsafe_allow_html=True)

            # Download results
            st.subheader("💾 Download Results")
            st.markdown(create_download_link(entity_df, "entities.csv", "📥 Download Entities"), unsafe_allow_html=True)

        else:
            st.warning("⚠️ No entities found in the text.")

def extract_simple_entities(text):
    """Enhanced pattern-based entity extraction"""
    import re

    entities = []

    # Enhanced patterns for different entity types
    patterns = {
        'PERSON': [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Title First Last
            r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b'  # First M. Last
        ],
        'ORG': [
            r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation|Co)\b',
            r'\b(?:Apple|Google|Microsoft|Amazon|Facebook|Tesla|Netflix|IBM|Oracle)\b',
            r'\b[A-Z][A-Z]+\b',  # Acronyms like NASA, FBI
            r'\b[A-Z][a-z]+ (?:University|College|Institute|School)\b'
        ],
        'GPE': [
            r'\b(?:United States|California|New York|London|Paris|Tokyo|Beijing|Washington|Boston|Chicago|Los Angeles|San Francisco|Seattle|Miami|Dallas|Houston|Philadelphia|Phoenix|San Diego|San Antonio|Detroit|San Jose|Austin|Jacksonville|Fort Worth|Columbus|Charlotte|Memphis|Baltimore|El Paso|Milwaukee|Denver|Nashville|Las Vegas|Portland|Oklahoma City|Tucson|Albuquerque|Atlanta|Colorado Springs|Raleigh|Omaha|Miami|Oakland|Minneapolis|Tulsa|Cleveland|Wichita|Arlington)\b',
            r'\b(?:Canada|Mexico|Brazil|Argentina|Chile|Peru|Colombia|Venezuela|Ecuador|Bolivia|Uruguay|Paraguay|Guyana|Suriname|French Guiana)\b',
            r'\b(?:England|France|Germany|Italy|Spain|Portugal|Netherlands|Belgium|Switzerland|Austria|Sweden|Norway|Denmark|Finland|Poland|Czech Republic|Hungary|Romania|Bulgaria|Greece|Turkey|Russia|Ukraine|Belarus|Lithuania|Latvia|Estonia)\b',
            r'\b(?:China|Japan|India|South Korea|Thailand|Vietnam|Malaysia|Singapore|Indonesia|Philippines|Australia|New Zealand)\b'
        ],
        'MONEY': [
            r'\$[\d,]+(?:\.\d{2})?(?:\s?(?:million|billion|trillion))?',
            r'[\d,]+(?:\.\d{2})?\s?(?:dollars|USD|EUR|GBP|JPY)',
            r'€[\d,]+(?:\.\d{2})?',
            r'£[\d,]+(?:\.\d{2})?'
        ],
        'DATE': [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(?:today|yesterday|tomorrow|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
        ],
        'TIME': [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b',
            r'\b(?:morning|afternoon|evening|night|midnight|noon)\b'
        ],
        'EMAIL': [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ],
        'PHONE': [
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        ],
        'URL': [
            r'https?://[^\s]+',
            r'www\.[^\s]+\.[a-z]{2,}'
        ]
    }

    for entity_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Avoid duplicates
                entity_text = match.group()
                start, end = match.start(), match.end()

                # Check if this entity overlaps with existing ones
                overlap = False
                for existing_entity in entities:
                    existing_start, existing_end = existing_entity[2], existing_entity[3]
                    if (start < existing_end and end > existing_start):
                        overlap = True
                        break

                if not overlap:
                    entities.append((entity_text, entity_type, start, end))

    return entities

def highlight_entities_in_text(text, entities):
    """Highlight entities in text with colors"""

    # Color mapping for entity types
    colors = {
        'PERSON': '#ffeb3b',
        'ORG': '#4caf50',
        'GPE': '#2196f3',
        'MONEY': '#ff9800',
        'DATE': '#9c27b0',
        'MISC': '#607d8b'
    }

    # Sort entities by start position (reverse order for replacement)
    sorted_entities = sorted(entities, key=lambda x: x[2], reverse=True)

    highlighted = text
    for entity_text, entity_type, start, end in sorted_entities:
        color = colors.get(entity_type, '#gray')
        replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{entity_text} ({entity_type})</span>'
        highlighted = highlighted[:start] + replacement + highlighted[end:]

    return highlighted

def show_text_summarization(text, modules):
    """Text summarization page"""
    st.markdown('<h2 class="feature-header">📝 Text Summarization</h2>', unsafe_allow_html=True)

    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
        return

    # Check text length
    if len(text.split()) < 50:
        st.warning("⚠️ Text is too short for meaningful summarization. Please provide at least 50 words.")
        return

    with st.spinner("🔄 Generating summary..."):

        # Summarization parameters
        col1, col2 = st.columns(2)
        with col1:
            num_sentences = st.slider("Number of sentences in summary:", 1, 5, 3)
        with col2:
            summary_method = st.selectbox("Summarization method:",
                                        ["Frequency-based", "TF-IDF", "Advanced (if available)"])

        # Generate summary
        if summary_method == "Frequency-based":
            summary = frequency_based_summary(text, num_sentences)
            method_used = "Frequency-based extractive summarization"

        elif summary_method == "TF-IDF":
            summary = tfidf_based_summary(text, num_sentences)
            method_used = "TF-IDF extractive summarization"

        else:  # Advanced
            try:
                # Try to use transformers if available
                from transformers import pipeline
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

                # Truncate text if too long
                max_length = 1024
                if len(text) > max_length:
                    text_truncated = text[:max_length]
                    st.info(f"ℹ️ Text truncated to {max_length} characters for processing")
                else:
                    text_truncated = text

                result = summarizer(text_truncated,
                                  max_length=min(150, len(text_truncated)//4),
                                  min_length=30,
                                  do_sample=False)
                summary = result[0]['summary_text']
                method_used = "BART transformer-based abstractive summarization"

            except ImportError:
                st.info("ℹ️ Advanced summarization not available. Using TF-IDF method.")
                summary = tfidf_based_summary(text, num_sentences)
                method_used = "TF-IDF extractive summarization (fallback)"
            except Exception as e:
                st.error(f"❌ Advanced summarization error: {str(e)}")
                summary = tfidf_based_summary(text, num_sentences)
                method_used = "TF-IDF extractive summarization (fallback)"

        # Display results
        st.subheader("📊 Summary Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            compression_ratio = (1 - len(summary) / len(text)) * 100
            st.metric("📉 Compression Ratio", f"{compression_ratio:.1f}%")
        with col2:
            st.metric("📏 Original Length", f"{len(text)} chars")
        with col3:
            st.metric("📝 Summary Length", f"{len(summary)} chars")

        # Method used
        st.info(f"🔧 Method used: {method_used}")

        # Display original and summary side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Original Text")
            st.text_area("", text, height=300, disabled=True, key="original")

        with col2:
            st.subheader("📝 Generated Summary")
            st.text_area("", summary, height=300, disabled=True, key="summary")

        # Summary quality metrics
        st.subheader("📊 Summary Analysis")

        original_sentences = len([s for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()])
        summary_sentences = len([s for s in summary.replace('!', '.').replace('?', '.').split('.') if s.strip()])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Original Sentences", original_sentences)
        with col2:
            st.metric("📝 Summary Sentences", summary_sentences)
        with col3:
            sentence_reduction = (1 - summary_sentences / max(original_sentences, 1)) * 100
            st.metric("📉 Sentence Reduction", f"{sentence_reduction:.1f}%")
        with col4:
            words_original = len(text.split())
            words_summary = len(summary.split())
            word_reduction = (1 - words_summary / max(words_original, 1)) * 100
            st.metric("🔤 Word Reduction", f"{word_reduction:.1f}%")

        # Download summary
        st.subheader("💾 Download Summary")
        summary_data = {
            'Original Text': [text],
            'Summary': [summary],
            'Method': [method_used],
            'Compression Ratio': [f"{compression_ratio:.1f}%"]
        }
        summary_df = pd.DataFrame(summary_data)
        st.markdown(create_download_link(summary_df, "summary.csv", "📥 Download Summary"), unsafe_allow_html=True)

def frequency_based_summary(text, num_sentences=3):
    """Simple frequency-based extractive summarization"""
    import re
    from collections import Counter

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    if len(sentences) <= num_sentences:
        return text

    # Calculate word frequencies
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    words_filtered = [word for word in words if word not in stop_words and len(word) > 2]
    word_freq = Counter(words_filtered)

    # Score sentences
    sentence_scores = {}
    for sentence in sentences:
        words_in_sentence = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(word_freq.get(word, 0) for word in words_in_sentence)
        sentence_scores[sentence] = score

    # Get top sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]

    # Maintain original order
    summary_sentences = []
    for sentence in sentences:
        if any(sentence == top[0] for top in top_sentences):
            summary_sentences.append(sentence)

    return '. '.join(summary_sentences) + '.'

def tfidf_based_summary(text, num_sentences=3):
    """TF-IDF based extractive summarization"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import re

        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        if len(sentences) <= num_sentences:
            return text

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Calculate sentence scores (sum of TF-IDF values)
        sentence_scores = tfidf_matrix.sum(axis=1).A1

        # Get top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]

        # Maintain original order
        summary_sentences = [sentences[i] for i in sorted(top_indices)]

        return '. '.join(summary_sentences) + '.'

    except ImportError:
        # Fallback to frequency-based
        return frequency_based_summary(text, num_sentences)

if __name__ == "__main__":
    main()
