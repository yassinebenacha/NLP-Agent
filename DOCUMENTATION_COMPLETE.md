# 📚 NLP Agent - Documentation Complète

## 🎯 **Vue d'Ensemble du Projet**

### **Qu'est-ce que NLP Agent?**
NLP Agent est une **application web complète** développée avec Streamlit qui offre **5 fonctionnalités d'analyse de texte** avancées. C'est un projet parfait pour démontrer vos compétences en **Data Science**, **NLP**, et **développement web**.

### **Technologies Utilisées**
- **Frontend**: Streamlit (interface web interactive)
- **Backend**: Python avec bibliothèques NLP
- **Machine Learning**: scikit-learn, NLTK, TextBlob
- **Visualisation**: Plotly, Matplotlib
- **Déploiement**: Streamlit Cloud + GitHub

---

## 🏗️ **Architecture du Projet**

### **Structure des Fichiers**
```
NLP-Agent/
├── 📄 app.py                    # Application principale Streamlit
├── 📁 src/                      # Modules Python
│   ├── data_preprocessing.py    # Préprocessing de texte
│   ├── sentiment_analysis.py    # Analyse de sentiment
│   └── simple_data_preprocessing.py  # Version simplifiée
├── 📁 models/                   # Modèles pré-entraînés
│   ├── lda_model.pkl           # Modèle LDA pour topic modeling
│   └── tfidf_vectorizer.pkl    # Vectoriseur TF-IDF
├── 📄 requirements.txt         # Dépendances Python
├── 📄 .gitignore              # Fichiers à ignorer par Git
└── 📄 README.md               # Documentation du projet
```

### **Flux de Données**
```
Texte d'entrée → Préprocessing → Analyse → Visualisation → Résultats
```

---

## 🔧 **Fonctionnalités Détaillées**

### **1. 📊 Data Exploration (Exploration de Données)**

#### **Objectif**
Analyser les caractéristiques statistiques et structurelles du texte.

#### **Fonctionnalités**
- **Statistiques de base**: Nombre de mots, phrases, caractères
- **Analyse de fréquence**: Mots les plus fréquents
- **Visualisations**: Graphiques interactifs avec Plotly
- **Nuage de mots**: Représentation visuelle des mots importants

#### **Code Principal**
```python
def show_data_exploration(text, modules):
    # Calcul des statistiques
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    # Visualisation avec Plotly
    word_freq = pd.Series(words).value_counts().head(10)
    fig = px.bar(x=word_freq.values, y=word_freq.index)
    st.plotly_chart(fig)
```

### **2. 😊 Sentiment Analysis (Analyse de Sentiment)**

#### **Objectif**
Déterminer l'émotion (positive, négative, neutre) exprimée dans le texte.

#### **Méthodes Utilisées**
1. **TextBlob**: Analyse rapide basée sur des lexiques
2. **Patterns**: Analyse basée sur des règles linguistiques
3. **Machine Learning**: Classification avec scikit-learn

#### **Code Principal**
```python
def analyze_sentiment(text):
    # TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Classification
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'
```

### **3. 🎯 Topic Modeling (Modélisation de Sujets)**

#### **Objectif**
Identifier automatiquement les thèmes principaux dans le texte.

#### **Algorithme: LDA (Latent Dirichlet Allocation)**
- **Principe**: Chaque document est un mélange de sujets
- **Sortie**: Distribution de probabilité sur les sujets
- **Visualisation**: Graphiques de distribution des sujets

#### **Code Principal**
```python
# Entraînement du modèle LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform([text])

lda = LatentDirichletAllocation(n_components=3)
lda.fit(tfidf_matrix)
```

### **4. 🏷️ Named Entity Recognition (Reconnaissance d'Entités)**

#### **Objectif**
Identifier et classifier les entités nommées (personnes, lieux, organisations).

#### **Méthodes**
1. **spaCy** (si disponible): Modèle pré-entraîné
2. **Patterns regex** (fallback): Reconnaissance basée sur des motifs

#### **Types d'Entités Détectées**
- **PERSON**: Noms de personnes
- **ORG**: Organisations, entreprises
- **GPE**: Lieux géopolitiques (villes, pays)
- **DATE**: Dates et expressions temporelles

### **5. 📝 Text Summarization (Résumé de Texte)**

#### **Objectif**
Créer automatiquement un résumé concis du texte original.

#### **Méthodes Implémentées**
1. **TF-IDF**: Sélection des phrases les plus importantes
2. **Fréquence**: Basé sur la fréquence des mots
3. **Position**: Privilégie les premières phrases

#### **Code Principal**
```python
def tfidf_based_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calcul des scores
    scores = tfidf_matrix.sum(axis=1).A1
    ranked_sentences = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
    # Sélection des meilleures phrases
    summary_sentences = [sentences[i] for i, _ in ranked_sentences[:num_sentences]]
    return ' '.join(summary_sentences)
```

---

## 🛠️ **Installation et Configuration**

### **Étape 1: Cloner le Projet**
```bash
git clone https://github.com/yassinebenacha/NLP-Agent.git
cd NLP-Agent
```

### **Étape 2: Créer un Environnement Virtuel**
```bash
# Créer l'environnement
python -m venv nlp_env

# Activer l'environnement
# Windows:
nlp_env\Scripts\activate
# Mac/Linux:
source nlp_env/bin/activate
```

### **Étape 3: Installer les Dépendances**
```bash
pip install -r requirements.txt
```

### **Étape 4: Télécharger les Données NLTK**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### **Étape 5: Lancer l'Application**
```bash
streamlit run app.py
```

---

## 📦 **Dépendances Expliquées**

### **requirements.txt**
```
streamlit>=1.28.0      # Framework web pour l'interface
pandas>=1.5.0          # Manipulation de données
numpy>=1.21.0          # Calculs numériques
matplotlib>=3.5.0      # Visualisations statiques
plotly>=5.0.0          # Visualisations interactives
scikit-learn>=1.1.0    # Machine learning
nltk>=3.8              # Traitement du langage naturel
textblob>=0.17.0       # Analyse de sentiment simple
```

### **Pourquoi ces Bibliothèques?**
- **Streamlit**: Interface web sans JavaScript
- **Pandas**: Manipulation efficace des données textuelles
- **Plotly**: Graphiques interactifs pour l'exploration
- **scikit-learn**: Algorithmes ML robustes et optimisés
- **NLTK**: Outils NLP complets et bien documentés

---

## 🎨 **Interface Utilisateur**

### **Navigation Sidebar**
```python
# Sélection de la fonctionnalité
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Tool:",
    ["🏠 Home", "📊 Data Exploration", "😊 Sentiment Analysis", 
     "🎯 Topic Modeling", "🏷️ Named Entity Recognition", 
     "📝 Text Summarization"]
)
```

### **Zone de Saisie de Texte**
```python
# Options d'entrée multiples
input_method = st.radio("Choose input method:", 
                       ["✍️ Type text", "📁 Upload file", "📋 Use sample"])

if input_method == "✍️ Type text":
    text = st.text_area("Enter your text:", height=200)
elif input_method == "📁 Upload file":
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
```

### **Affichage des Résultats**
```python
# Colonnes pour organiser l'affichage
col1, col2 = st.columns(2)

with col1:
    st.metric("Word Count", len(words))
    
with col2:
    st.metric("Sentence Count", num_sentences)
```

---

## 🧠 **Algorithmes et Concepts NLP**

### **1. Préprocessing de Texte**

#### **Étapes du Préprocessing**
```python
def preprocess_text(text):
    # 1. Nettoyage de base
    text = re.sub(r'http\S+', '', text)  # Supprimer URLs
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer ponctuation

    # 2. Normalisation
    text = text.lower()  # Minuscules

    # 3. Tokenisation
    tokens = word_tokenize(text)

    # 4. Suppression des mots vides
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    # 5. Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)
```

#### **Pourquoi Préprocesser?**
- **Réduction du bruit**: Éliminer les éléments non informatifs
- **Normalisation**: Uniformiser le format du texte
- **Optimisation**: Réduire la dimensionnalité des données

### **2. TF-IDF (Term Frequency-Inverse Document Frequency)**

#### **Formule Mathématique**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)

où:
TF(t,d) = (Nombre d'occurrences de t dans d) / (Nombre total de mots dans d)
IDF(t) = log(Nombre total de documents / Nombre de documents contenant t)
```

#### **Implémentation**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=1000,      # Limiter le vocabulaire
    ngram_range=(1, 2),     # Unigrammes et bigrammes
    stop_words='english'    # Supprimer mots vides
)

tfidf_matrix = vectorizer.fit_transform(documents)
```

### **3. LDA (Latent Dirichlet Allocation)**

#### **Principe**
- Chaque **document** est un mélange de **sujets**
- Chaque **sujet** est un mélange de **mots**
- Algorithme probabiliste pour découvrir ces mélanges

#### **Paramètres Importants**
```python
lda = LatentDirichletAllocation(
    n_components=5,         # Nombre de sujets
    random_state=42,        # Reproductibilité
    max_iter=100,          # Nombre d'itérations
    learning_method='batch' # Méthode d'apprentissage
)
```

### **4. Analyse de Sentiment avec TextBlob**

#### **Métriques**
- **Polarity**: [-1, 1] (négatif → positif)
- **Subjectivity**: [0, 1] (objectif → subjectif)

#### **Classification**
```python
def classify_sentiment(polarity):
    if polarity > 0.1:
        return "Positive 😊"
    elif polarity < -0.1:
        return "Negative 😞"
    else:
        return "Neutral 😐"
```

---

## 🚀 **Déploiement sur Streamlit Cloud**

### **Étape 1: Préparer le Repository GitHub**
```bash
# Vérifier que tous les fichiers sont inclus
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### **Étape 2: Configurer Streamlit Cloud**
1. Aller sur [share.streamlit.io](https://share.streamlit.io)
2. Se connecter avec GitHub
3. Sélectionner le repository `NLP-Agent`
4. Choisir `app.py` comme fichier principal
5. Cliquer sur "Deploy"

### **Étape 3: Vérifier le Déploiement**
L'application devrait afficher:
- ✅ **LDA model loaded**
- ✅ **TF-IDF vectorizer loaded**
- ✅ **Text Preprocessor loaded**
- ✅ **Sentiment Analyzer loaded**
- ⚠️ **spaCy NER not available** (normal)

---

## 🔧 **Gestion des Erreurs et Fallbacks**

### **Stratégie de Robustesse**
```python
# Exemple de fallback pour spaCy
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    # Utiliser des patterns regex comme fallback
```

### **Gestion des Dépendances Optionnelles**
```python
def safe_import(module_name, fallback_func=None):
    try:
        return __import__(module_name)
    except ImportError:
        if fallback_func:
            return fallback_func()
        return None
```

---

## 📊 **Métriques et Évaluation**

### **Métriques de Performance**
- **Temps de traitement**: Mesure de la vitesse d'analyse
- **Précision**: Qualité des résultats (pour sentiment)
- **Couverture**: Pourcentage d'entités détectées

### **Validation des Résultats**
```python
def validate_sentiment_analysis(predictions, ground_truth):
    from sklearn.metrics import accuracy_score, classification_report

    accuracy = accuracy_score(ground_truth, predictions)
    report = classification_report(ground_truth, predictions)

    return accuracy, report
```

---

## 🎯 **Cas d'Usage et Applications**

### **1. Analyse de Feedback Client**
- Analyser les avis produits
- Identifier les points d'amélioration
- Mesurer la satisfaction client

### **2. Veille Médiatique**
- Surveiller les mentions de marque
- Analyser le sentiment des articles
- Identifier les sujets tendance

### **3. Analyse de Contenu**
- Résumer des documents longs
- Extraire les entités importantes
- Classifier le contenu par thème

### **4. Recherche Académique**
- Analyser des corpus de textes
- Identifier les thèmes de recherche
- Extraire des insights quantitatifs

---

## 🏆 **Avantages pour votre Profil**

### **Compétences Démontrées**
- **Data Science**: Preprocessing, ML, évaluation
- **NLP**: Sentiment, NER, topic modeling, summarization
- **Développement Web**: Interface utilisateur interactive
- **DevOps**: Déploiement cloud, gestion des dépendances
- **Gestion de Projet**: Structure, documentation, versioning

### **Points Forts du Projet**
- **Complet**: 5 fonctionnalités NLP différentes
- **Robuste**: Gestion d'erreurs et fallbacks
- **Professionnel**: Interface propre et intuitive
- **Déployé**: Application web accessible publiquement
- **Documenté**: Code commenté et documentation complète

---

## 📚 **Ressources pour Approfondir**

### **Livres Recommandés**
- "Natural Language Processing with Python" (NLTK Book)
- "Speech and Language Processing" (Jurafsky & Martin)
- "Hands-On Machine Learning" (Aurélien Géron)

### **Cours en Ligne**
- CS224N (Stanford NLP Course)
- Fast.ai NLP Course
- Coursera NLP Specialization

### **Documentation Officielle**
- [Streamlit Docs](https://docs.streamlit.io)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [NLTK Documentation](https://www.nltk.org)

---

## 🎉 **Conclusion**

Votre **NLP Agent** est un projet complet qui démontre une maîtrise solide des technologies modernes de Data Science et NLP. Il combine:

- **Théorie**: Algorithmes NLP avancés
- **Pratique**: Implémentation robuste
- **Déploiement**: Application web fonctionnelle
- **Documentation**: Guide complet et professionnel

**C'est un excellent atout pour vos candidatures de stage et votre portfolio professionnel! 🚀**
