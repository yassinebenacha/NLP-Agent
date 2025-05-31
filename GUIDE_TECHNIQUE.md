# 🔧 Guide Technique - NLP Agent

## 📋 **Table des Matières**
1. [Architecture Technique](#architecture-technique)
2. [Structure du Code](#structure-du-code)
3. [Modules et Classes](#modules-et-classes)
4. [Algorithmes Implémentés](#algorithmes-implémentés)
5. [Gestion des Erreurs](#gestion-des-erreurs)
6. [Performance et Optimisation](#performance-et-optimisation)
7. [Extension et Personnalisation](#extension-et-personnalisation)

---

## 🏗️ **Architecture Technique**

### **Stack Technologique**
```
Frontend:    Streamlit (Python Web Framework)
Backend:     Python 3.8+
ML/NLP:      scikit-learn, NLTK, TextBlob
Viz:         Plotly, Matplotlib
Deploy:      Streamlit Cloud + GitHub
Storage:     Pickle (modèles), Local (temporaire)
```

### **Flux de Données**
```
Input Text → Preprocessing → Feature Extraction → ML Models → Visualization → Output
     ↓              ↓              ↓               ↓            ↓           ↓
  Streamlit    TextPreprocessor  TfidfVectorizer   LDA      Plotly     Streamlit
   Widget         Class           sklearn         Model     Charts      Display
```

### **Diagramme d'Architecture**
```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT APP (app.py)                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    Home     │  │ Data Explor │  │   Sentiment Anal    │  │
│  │   Page      │  │   ation     │  │      ysis           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │   Topic     │  │    NER      │                          │
│  │  Modeling   │  │  Analysis   │                          │
│  └─────────────┘  └─────────────┘                          │
├─────────────────────────────────────────────────────────────┤
│                      SRC MODULES                           │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │ data_preprocessing│  │    sentiment_analysis.py      │  │
│  │      .py        │  │                                 │  │
│  └─────────────────┘  └─────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     MODELS FOLDER                          │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │  lda_model.pkl  │  │   tfidf_vectorizer.pkl         │  │
│  └─────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 **Structure du Code**

### **app.py - Application Principale**
```python
# Structure principale
def main():
    setup_page_config()      # Configuration Streamlit
    load_custom_css()        # Styles CSS personnalisés
    modules = load_modules() # Chargement des modules NLP
    
    # Navigation sidebar
    analysis_type = st.sidebar.selectbox(...)
    
    # Routage vers les fonctionnalités
    if analysis_type == "🏠 Home":
        show_home_page()
    elif analysis_type == "📊 Data Exploration":
        show_data_exploration(text, modules)
    # ... autres fonctionnalités
```

### **Fonctions Principales**
```python
# Chargement des modules
def load_modules() -> Dict[str, Any]:
    """Charge tous les modules NLP avec gestion d'erreurs"""
    
# Pages de fonctionnalités
def show_data_exploration(text: str, modules: Dict) -> None:
def show_sentiment_analysis(text: str, modules: Dict) -> None:
def show_topic_modeling(text: str, models: Dict, modules: Dict) -> None:
def show_named_entity_recognition(text: str, modules: Dict) -> None:
def show_text_summarization(text: str, modules: Dict) -> None:

# Utilitaires
def get_text_input() -> str:
def create_download_link(data: str, filename: str) -> str:
```

---

## 🧩 **Modules et Classes**

### **1. data_preprocessing.py**

#### **Classe TextPreprocessor**
```python
class TextPreprocessor:
    def __init__(self, 
                 remove_stopwords: bool = True,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 remove_numbers: bool = False,
                 min_word_length: int = 2,
                 max_word_length: int = 50,
                 language: str = 'english'):
        # Initialisation avec fallbacks intelligents
```

#### **Méthodes Principales**
```python
def clean_text(self, text: str) -> str:
    """Nettoyage de base du texte"""
    # Suppression URLs, emails, HTML tags
    # Normalisation des espaces
    
def preprocess_text(self, text: str) -> str:
    """Pipeline complet de préprocessing"""
    # 1. Nettoyage → 2. Tokenisation → 3. Filtrage → 4. Lemmatisation
    
def extract_sentences(self, text: str) -> List[str]:
    """Extraction de phrases avec fallback"""
    # NLTK sent_tokenize ou regex fallback
    
def get_word_frequency(self, texts: List[str], top_n: int = 20) -> dict:
    """Calcul de fréquence des mots"""
```

#### **Gestion des Fallbacks**
```python
# Imports avec gestion d'erreurs
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Utilisation conditionnelle
if NLTK_AVAILABLE:
    tokens = word_tokenize(text)
else:
    tokens = re.findall(r'\b\w+\b', text)  # Fallback regex
```

### **2. sentiment_analysis.py**

#### **Classe SentimentAnalyzer**
```python
class SentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        # Configuration inline (pas de fichier config externe)
        self.config = {
            "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "max_length": 512
        }
```

#### **Méthodes d'Analyse**
```python
def textblob_sentiment(self, text: str) -> Dict[str, float]:
    """Analyse avec TextBlob - rapide et simple"""
    
def transformer_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
    """Analyse avec modèles Transformer (si disponible)"""
    
def predict_sentiment(self, text: Union[str, List[str]], method: str = 'textblob'):
    """Interface unifiée pour toutes les méthodes"""
```

---

## 🤖 **Algorithmes Implémentés**

### **1. TF-IDF (Term Frequency-Inverse Document Frequency)**

#### **Implémentation**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_vectorizer():
    return TfidfVectorizer(
        max_features=1000,           # Vocabulaire limité
        ngram_range=(1, 2),          # Uni et bigrammes
        stop_words='english',        # Suppression mots vides
        lowercase=True,              # Normalisation casse
        min_df=2,                    # Fréquence minimum
        max_df=0.95                  # Fréquence maximum
    )
```

#### **Utilisation**
```python
# Entraînement
vectorizer = create_tfidf_vectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Transformation nouveau texte
new_vector = vectorizer.transform([new_text])
```

### **2. LDA (Latent Dirichlet Allocation)**

#### **Configuration**
```python
from sklearn.decomposition import LatentDirichletAllocation

def create_lda_model(n_topics=5):
    return LatentDirichletAllocation(
        n_components=n_topics,       # Nombre de sujets
        random_state=42,             # Reproductibilité
        max_iter=100,               # Itérations maximum
        learning_method='batch',     # Méthode d'apprentissage
        learning_offset=50.0,        # Paramètre d'apprentissage
        doc_topic_prior=None,        # Prior Dirichlet documents
        topic_word_prior=None        # Prior Dirichlet mots
    )
```

#### **Pipeline Complet**
```python
def topic_modeling_pipeline(texts, n_topics=3):
    # 1. Préprocessing
    preprocessor = TextPreprocessor()
    processed_texts = [preprocessor.preprocess_text(text) for text in texts]
    
    # 2. Vectorisation TF-IDF
    vectorizer = create_tfidf_vectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # 3. Modélisation LDA
    lda_model = create_lda_model(n_topics)
    lda_model.fit(tfidf_matrix)
    
    # 4. Extraction des sujets
    feature_names = vectorizer.get_feature_names_out()
    topics = extract_topics(lda_model, feature_names)
    
    return lda_model, vectorizer, topics
```

### **3. Named Entity Recognition**

#### **Méthode spaCy (si disponible)**
```python
def spacy_ner(text):
    if SPACY_AVAILABLE and nlp:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    return []
```

#### **Méthode Pattern-based (fallback)**
```python
def pattern_based_ner(text):
    entities = []
    
    # Patterns pour différents types d'entités
    patterns = {
        'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
        'ORG': r'\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC)\b',
        'GPE': r'\b(?:Paris|London|New York|Tokyo)\b',
        'DATE': r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'
    }
    
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            entities.append((match.group(), entity_type))
    
    return entities
```

### **4. Text Summarization**

#### **Méthode TF-IDF**
```python
def tfidf_based_summary(text, num_sentences=3):
    # 1. Segmentation en phrases
    sentences = sent_tokenize(text)
    
    # 2. Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # 3. Calcul des scores de phrases
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    
    # 4. Sélection des meilleures phrases
    ranked_sentences = sorted(enumerate(sentence_scores), 
                            key=lambda x: x[1], reverse=True)
    
    # 5. Reconstruction du résumé
    summary_indices = sorted([i for i, _ in ranked_sentences[:num_sentences]])
    summary = ' '.join([sentences[i] for i in summary_indices])
    
    return summary
```

#### **Méthode Frequency-based**
```python
def frequency_based_summary(text, num_sentences=3):
    # 1. Préprocessing et comptage des mots
    words = preprocess_text(text).split()
    word_freq = Counter(words)
    
    # 2. Score des phrases basé sur la fréquence
    sentences = sent_tokenize(text)
    sentence_scores = {}
    
    for sentence in sentences:
        sentence_words = preprocess_text(sentence).split()
        score = sum(word_freq[word] for word in sentence_words)
        sentence_scores[sentence] = score / len(sentence_words)
    
    # 3. Sélection des meilleures phrases
    ranked_sentences = sorted(sentence_scores.items(), 
                            key=lambda x: x[1], reverse=True)
    
    return ' '.join([sent for sent, _ in ranked_sentences[:num_sentences]])
```

---

## 🛡️ **Gestion des Erreurs**

### **Stratégie de Robustesse**
```python
def safe_execution(func, fallback_func=None, *args, **kwargs):
    """Exécution sécurisée avec fallback"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.warning(f"Primary method failed: {e}")
        if fallback_func:
            try:
                return fallback_func(*args, **kwargs)
            except Exception as e2:
                st.error(f"Fallback method also failed: {e2}")
        return None
```

### **Validation des Entrées**
```python
def validate_text_input(text):
    """Validation et nettoyage des entrées utilisateur"""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    if len(text) < 10:
        raise ValueError("Text too short for meaningful analysis")
    
    if len(text) > 50000:
        st.warning("Text is very long, truncating to 50,000 characters")
        text = text[:50000]
    
    return text.strip()
```

### **Gestion des Modèles Manquants**
```python
def load_model_safely(model_path):
    """Chargement sécurisé des modèles"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning(f"Model not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
```

---

## ⚡ **Performance et Optimisation**

### **Caching Streamlit**
```python
@st.cache_data
def load_and_preprocess_text(text):
    """Cache le préprocessing pour éviter les recalculs"""
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_text(text)

@st.cache_resource
def load_models():
    """Cache le chargement des modèles (une seule fois)"""
    models = {}
    models['lda'] = load_model_safely('models/lda_model.pkl')
    models['tfidf'] = load_model_safely('models/tfidf_vectorizer.pkl')
    return models
```

### **Optimisation Mémoire**
```python
def process_large_text(text, chunk_size=1000):
    """Traitement par chunks pour les gros textes"""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    results = []
    
    for chunk in chunks:
        result = process_chunk(chunk)
        results.append(result)
    
    return combine_results(results)
```

### **Lazy Loading**
```python
class LazyNLPModules:
    """Chargement paresseux des modules lourds"""
    def __init__(self):
        self._spacy_model = None
        self._transformer_model = None
    
    @property
    def spacy_model(self):
        if self._spacy_model is None:
            self._spacy_model = self._load_spacy()
        return self._spacy_model
```

---

## 🔧 **Extension et Personnalisation**

### **Ajouter une Nouvelle Fonctionnalité**

#### **1. Créer le Module**
```python
# src/new_feature.py
class NewFeatureAnalyzer:
    def __init__(self):
        pass
    
    def analyze(self, text):
        # Implémentation de la nouvelle fonctionnalité
        return results
```

#### **2. Intégrer dans app.py**
```python
# Dans load_modules()
try:
    from new_feature import NewFeatureAnalyzer
    modules['new_feature'] = NewFeatureAnalyzer()
    st.sidebar.success("✅ New Feature loaded")
except Exception as e:
    st.sidebar.error(f"❌ New Feature failed: {e}")
    modules['new_feature'] = None

# Ajouter dans la navigation
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Tool:",
    ["🏠 Home", "📊 Data Exploration", ..., "🆕 New Feature"]
)

# Créer la fonction d'affichage
def show_new_feature(text, modules):
    st.markdown('<h2 class="feature-header">🆕 New Feature</h2>', 
                unsafe_allow_html=True)
    
    if modules['new_feature']:
        results = modules['new_feature'].analyze(text)
        # Affichage des résultats
    else:
        st.error("New Feature not available")
```

### **Personnaliser les Styles**
```python
def load_custom_css():
    st.markdown("""
    <style>
    .feature-header {
        color: #1f77b4;
        font-size: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
```

### **Ajouter de Nouveaux Modèles**
```python
# Entraîner et sauvegarder un nouveau modèle
def train_and_save_model(data, model_path):
    model = SomeMLModel()
    model.fit(data)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")

# Charger dans l'application
@st.cache_resource
def load_custom_model():
    return load_model_safely('models/custom_model.pkl')
```

---

## 📊 **Monitoring et Debugging**

### **Logging**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_with_logging(text, method):
    logger.info(f"Starting analysis with method: {method}")
    start_time = time.time()
    
    try:
        result = perform_analysis(text, method)
        duration = time.time() - start_time
        logger.info(f"Analysis completed in {duration:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
```

### **Métriques de Performance**
```python
def track_performance(func):
    """Décorateur pour tracker les performances"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        st.sidebar.metric("Execution Time", f"{end_time - start_time:.2f}s")
        st.sidebar.metric("Memory Usage", f"{(end_memory - start_memory) / 1024 / 1024:.1f} MB")
        
        return result
    return wrapper
```

**Votre NLP Agent est maintenant techniquement documenté et prêt pour l'extension! 🚀**
