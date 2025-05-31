# 🚀 Guide d'Utilisation - NLP Agent

## 📋 **Table des Matières**
1. [Démarrage Rapide](#démarrage-rapide)
2. [Guide des Fonctionnalités](#guide-des-fonctionnalités)
3. [Exemples Pratiques](#exemples-pratiques)
4. [Troubleshooting](#troubleshooting)
5. [FAQ](#faq)

---

## 🏁 **Démarrage Rapide**

### **Accès à l'Application**
- **URL Déployée**: [Votre lien Streamlit Cloud]
- **Local**: `streamlit run app.py`

### **Interface Principale**
```
┌─────────────────────────────────────┐
│  🤖 NLP Agent                       │
├─────────────────────────────────────┤
│  Navigation (Sidebar):              │
│  ✅ LDA model loaded                │
│  ✅ TF-IDF vectorizer loaded        │
│  ✅ Text Preprocessor loaded        │
│  ✅ Sentiment Analyzer loaded       │
│  ⚠️ spaCy NER not available         │
│                                     │
│  Choose Analysis Tool:              │
│  🏠 Home                           │
│  📊 Data Exploration               │
│  😊 Sentiment Analysis             │
│  🎯 Topic Modeling                 │
│  🏷️ Named Entity Recognition       │
│  📝 Text Summarization             │
└─────────────────────────────────────┘
```

---

## 🎯 **Guide des Fonctionnalités**

### **1. 🏠 Home (Page d'Accueil)**

#### **Objectif**
Présentation du projet et navigation vers les fonctionnalités.

#### **Utilisation**
1. Lisez la description du projet
2. Consultez les fonctionnalités disponibles
3. Sélectionnez une analyse dans la sidebar

---

### **2. 📊 Data Exploration**

#### **Objectif**
Analyser les caractéristiques statistiques de votre texte.

#### **Étapes d'Utilisation**
1. **Saisir le texte**:
   ```
   Méthodes d'entrée:
   ✍️ Type text    - Saisie manuelle
   📁 Upload file  - Fichier .txt ou .csv
   📋 Use sample   - Données d'exemple
   ```

2. **Analyser les résultats**:
   - **Statistiques de base**: Mots, phrases, caractères
   - **Fréquence des mots**: Top 10 des mots les plus fréquents
   - **Graphiques**: Visualisations interactives
   - **Nuage de mots**: Représentation visuelle

3. **Interpréter les métriques**:
   ```
   📊 Métriques Affichées:
   - Word Count: Nombre total de mots
   - Sentence Count: Nombre de phrases
   - Character Count: Nombre de caractères
   - Average Word Length: Longueur moyenne des mots
   - Vocabulary Size: Nombre de mots uniques
   ```

#### **Exemple Pratique**
```
Texte d'entrée: "I love this product! It works great and the quality is excellent."

Résultats:
- Word Count: 12
- Sentence Count: 2
- Character Count: 71
- Top words: love, product, works, great, quality, excellent
```

---

### **3. 😊 Sentiment Analysis**

#### **Objectif**
Déterminer l'émotion exprimée dans le texte.

#### **Méthodes Disponibles**
1. **TextBlob**: Analyse rapide et simple
2. **Pattern-based**: Règles linguistiques
3. **Ensemble**: Combinaison de méthodes

#### **Étapes d'Utilisation**
1. **Entrer le texte** à analyser
2. **Sélectionner la méthode** d'analyse
3. **Interpréter les résultats**:
   ```
   Résultats Affichés:
   😊 Sentiment: Positive/Negative/Neutral
   📊 Confidence: Score de confiance (0-1)
   📈 Polarity: Score de polarité (-1 à +1)
   🎯 Subjectivity: Objectivité vs Subjectivité (0-1)
   ```

#### **Guide d'Interprétation**
```
Polarity Score:
+0.5 à +1.0  → Très Positif 😍
+0.1 à +0.5  → Positif 😊
-0.1 à +0.1  → Neutre 😐
-0.5 à -0.1  → Négatif 😞
-1.0 à -0.5  → Très Négatif 😡

Confidence Score:
0.8 - 1.0    → Très fiable ✅
0.6 - 0.8    → Fiable ✅
0.4 - 0.6    → Modéré ⚠️
0.0 - 0.4    → Peu fiable ❌
```

#### **Exemples**
```
Texte: "I absolutely love this product!"
→ Sentiment: Positive 😊
→ Polarity: +0.8
→ Confidence: 0.9

Texte: "This is terrible, worst purchase ever."
→ Sentiment: Negative 😞
→ Polarity: -0.7
→ Confidence: 0.85

Texte: "The product arrived on time."
→ Sentiment: Neutral 😐
→ Polarity: 0.0
→ Confidence: 0.6
```

---

### **4. 🎯 Topic Modeling**

#### **Objectif**
Identifier automatiquement les thèmes principaux dans le texte.

#### **Algorithme Utilisé**
- **LDA (Latent Dirichlet Allocation)**
- **TF-IDF Vectorization**

#### **Étapes d'Utilisation**
1. **Entrer un texte long** (minimum 50 mots recommandé)
2. **Configurer les paramètres**:
   ```
   Paramètres Ajustables:
   - Number of Topics: 2-10 (défaut: 3)
   - Max Features: 50-500 (défaut: 100)
   ```
3. **Analyser les résultats**:
   - **Topic Distribution**: Probabilité de chaque sujet
   - **Top Words per Topic**: Mots caractéristiques
   - **Visualizations**: Graphiques de distribution

#### **Interprétation des Résultats**
```
Topic 1 (30%): technology, software, development, programming
→ Sujet: Développement technologique

Topic 2 (45%): business, market, sales, customer, revenue
→ Sujet: Business et ventes

Topic 3 (25%): team, project, management, deadline, meeting
→ Sujet: Gestion de projet
```

#### **Conseils d'Utilisation**
- **Textes longs**: Meilleurs résultats avec 200+ mots
- **Textes cohérents**: Éviter les textes trop disparates
- **Nombre de sujets**: Commencer avec 3-5 sujets

---

### **5. 🏷️ Named Entity Recognition**

#### **Objectif**
Identifier et classifier les entités nommées dans le texte.

#### **Types d'Entités Détectées**
```
👤 PERSON: Noms de personnes
🏢 ORG: Organisations, entreprises
🌍 GPE: Lieux géopolitiques (villes, pays)
📅 DATE: Dates et expressions temporelles
💰 MONEY: Montants monétaires
📊 PERCENT: Pourcentages
🔢 CARDINAL: Nombres
```

#### **Méthodes Utilisées**
1. **spaCy** (si disponible): Modèle pré-entraîné
2. **Pattern-based** (fallback): Expressions régulières

#### **Étapes d'Utilisation**
1. **Entrer le texte** contenant des entités
2. **Visualiser les résultats**:
   - **Highlighted Text**: Texte avec entités surlignées
   - **Entity Table**: Tableau des entités trouvées
   - **Statistics**: Nombre d'entités par type

#### **Exemple Pratique**
```
Texte: "Apple Inc. was founded by Steve Jobs in Cupertino in 1976."

Entités Détectées:
👤 PERSON: Steve Jobs
🏢 ORG: Apple Inc.
🌍 GPE: Cupertino
📅 DATE: 1976
```

---

### **6. 📝 Text Summarization**

#### **Objectif**
Créer automatiquement un résumé concis du texte original.

#### **Méthodes Disponibles**
1. **TF-IDF Based**: Sélection des phrases les plus importantes
2. **Frequency Based**: Basé sur la fréquence des mots
3. **Position Based**: Privilégie les premières phrases

#### **Paramètres Configurables**
```
Summary Length:
- Short (1-2 sentences)
- Medium (3-4 sentences)  
- Long (5+ sentences)

Summary Ratio:
- 10% du texte original
- 25% du texte original
- 50% du texte original
```

#### **Étapes d'Utilisation**
1. **Entrer un texte long** (minimum 100 mots)
2. **Choisir la méthode** de résumé
3. **Configurer la longueur** souhaitée
4. **Analyser le résumé**:
   - **Original Text Length**: Longueur du texte original
   - **Summary Length**: Longueur du résumé
   - **Compression Ratio**: Taux de compression
   - **Key Sentences**: Phrases clés sélectionnées

#### **Conseils d'Utilisation**
- **Textes structurés**: Meilleurs résultats avec des paragraphes
- **Longueur optimale**: 200-1000 mots pour l'original
- **Méthode TF-IDF**: Recommandée pour textes techniques
- **Méthode Position**: Bonne pour articles de presse

---

## 💡 **Exemples Pratiques**

### **Exemple 1: Analyse d'Avis Client**
```
Texte: "I recently purchased this laptop and I'm extremely satisfied! 
The performance is outstanding, battery life is excellent, and the 
design is sleek. Customer service was also very helpful. However, 
the price is a bit high, but overall it's worth the investment."

Workflow:
1. Data Exploration → Statistiques de base
2. Sentiment Analysis → Sentiment: Positive (0.7)
3. Named Entity Recognition → Produit: laptop
4. Text Summarization → "Performance outstanding, excellent battery, worth investment"
```

### **Exemple 2: Analyse d'Article de Presse**
```
Texte: Long article sur l'économie française...

Workflow:
1. Topic Modeling → Sujets: économie, politique, entreprises
2. Named Entity Recognition → Lieux: France, Paris; Personnes: Emmanuel Macron
3. Text Summarization → Résumé en 3 phrases clés
4. Sentiment Analysis → Sentiment global de l'article
```

---

## 🔧 **Troubleshooting**

### **Problèmes Courants**

#### **1. "No text provided"**
**Cause**: Champ de texte vide
**Solution**: Entrer du texte ou utiliser les données d'exemple

#### **2. "Text too short for analysis"**
**Cause**: Texte insuffisant pour l'analyse
**Solution**: 
- Data Exploration: Minimum 10 mots
- Topic Modeling: Minimum 50 mots
- Summarization: Minimum 100 mots

#### **3. "spaCy NER not available"**
**Cause**: spaCy non installé (normal en déploiement)
**Solution**: Utilise automatiquement les patterns regex

#### **4. Résultats incohérents**
**Cause**: Texte de mauvaise qualité ou trop court
**Solution**: 
- Nettoyer le texte (supprimer caractères spéciaux)
- Utiliser des textes plus longs et cohérents
- Essayer différentes méthodes d'analyse

### **Optimisation des Performances**
```
Pour de meilleurs résultats:
✅ Textes en anglais (optimisé pour l'anglais)
✅ Textes bien structurés (paragraphes, phrases complètes)
✅ Longueur appropriée selon la fonctionnalité
✅ Contenu cohérent et thématique
```

---

## ❓ **FAQ**

### **Q: Quelles langues sont supportées?**
**R**: Principalement l'anglais. Support limité pour d'autres langues.

### **Q: Quelle est la taille maximale de texte?**
**R**: Recommandé: 10,000 caractères maximum pour des performances optimales.

### **Q: Les données sont-elles sauvegardées?**
**R**: Non, toutes les analyses sont temporaires et locales.

### **Q: Comment améliorer la précision des résultats?**
**R**: 
- Utiliser des textes plus longs
- Nettoyer le texte avant analyse
- Choisir la méthode appropriée selon le type de contenu

### **Q: L'application fonctionne-t-elle hors ligne?**
**R**: Non, nécessite une connexion internet pour Streamlit Cloud.

### **Q: Comment exporter les résultats?**
**R**: Utiliser les boutons "Download" disponibles dans chaque section.

---

## 🎯 **Conseils d'Utilisation Avancée**

### **Workflow Recommandé**
1. **Commencer par Data Exploration** → Comprendre le texte
2. **Sentiment Analysis** → Évaluer le ton général
3. **Topic Modeling** → Identifier les thèmes (si texte long)
4. **Named Entity Recognition** → Extraire les entités importantes
5. **Text Summarization** → Créer un résumé (si texte long)

### **Cas d'Usage Spécifiques**
- **Feedback client**: Sentiment + NER
- **Articles de presse**: Topic Modeling + Summarization
- **Documents techniques**: Data Exploration + Summarization
- **Réseaux sociaux**: Sentiment + Data Exploration

**Votre NLP Agent est maintenant prêt à analyser tous vos textes! 🚀**
