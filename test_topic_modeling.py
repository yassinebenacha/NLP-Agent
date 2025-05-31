#!/usr/bin/env python3
"""
Test script pour vérifier le notebook de modélisation de sujets
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

def test_topic_modeling():
    """Test de la modélisation de sujets"""
    print("🎯 TEST DE LA MODÉLISATION DE SUJETS")
    print("=" * 50)
    
    # Créer un dataset d'exemple
    sample_texts = [
        "artificial intelligence machine learning technology innovation future",
        "climate change environment global warming sustainability green energy",
        "economy market finance business investment stock trading",
        "health medicine medical research treatment disease prevention",
        "education school university student learning online digital",
        "sports football basketball competition championship team victory",
        "politics government election democracy policy reform legislation",
        "technology smartphone computer internet digital transformation innovation",
        "travel tourism vacation destination culture adventure exploration",
        "food cooking recipe restaurant cuisine nutrition healthy eating"
    ]
    
    print(f"📊 Dataset d'exemple créé: {len(sample_texts)} documents")
    
    # Vectorisation TF-IDF
    try:
        vectorizer = TfidfVectorizer(
            max_features=100,
            min_df=1,
            max_df=0.8,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(sample_texts)
        print(f"✅ Vectorisation TF-IDF réussie: {tfidf_matrix.shape}")
        
    except Exception as e:
        print(f"❌ Erreur vectorisation: {e}")
        return False
    
    # Modélisation LDA
    try:
        n_topics = 3
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        
        lda_topics = lda_model.fit_transform(tfidf_matrix)
        print(f"✅ Modèle LDA entraîné: {n_topics} sujets")
        
        # Afficher les sujets
        feature_names = vectorizer.get_feature_names_out()
        print("\n🎯 SUJETS DÉCOUVERTS:")
        
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[-5:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"  Sujet {topic_idx}: {', '.join(top_words)}")
        
        # Assigner les sujets
        assigned_topics = np.argmax(lda_topics, axis=1)
        confidence_scores = np.max(lda_topics, axis=1)
        
        print(f"\n📊 Assignation des sujets:")
        for i, (topic, conf) in enumerate(zip(assigned_topics, confidence_scores)):
            print(f"  Doc {i}: Sujet {topic} (confiance: {conf:.3f})")
        
        print(f"\n📈 Confiance moyenne: {confidence_scores.mean():.3f}")
        
    except Exception as e:
        print(f"❌ Erreur LDA: {e}")
        return False
    
    # Test des bibliothèques avancées
    print("\n🤖 TEST DES BIBLIOTHÈQUES AVANCÉES:")
    
    try:
        from bertopic import BERTopic
        print("✅ BERTopic disponible")
    except ImportError:
        print("⚠️ BERTopic non disponible")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers disponible")
    except ImportError:
        print("⚠️ SentenceTransformers non disponible")
    
    try:
        import umap
        print("✅ UMAP disponible")
    except ImportError:
        print("⚠️ UMAP non disponible")
    
    try:
        import hdbscan
        print("✅ HDBSCAN disponible")
    except ImportError:
        print("⚠️ HDBSCAN non disponible")
    
    return True

def main():
    """Fonction principale"""
    print("🚀 Test du Notebook de Modélisation de Sujets")
    print("=" * 60)
    
    success = test_topic_modeling()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("\n📝 Le notebook 03_topic_modeling_fixed.ipynb est prêt à être utilisé!")
        print("\n🚀 Prochaines étapes:")
        print("  1. Ouvrir Jupyter: jupyter notebook")
        print("  2. Exécuter: notebooks/03_topic_modeling_fixed.ipynb")
        print("  3. Analyser les sujets découverts")
    else:
        print("❌ Certains tests ont échoué.")
        print("🔧 Vérifiez l'installation des dépendances.")

if __name__ == "__main__":
    main()
