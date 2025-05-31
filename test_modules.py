#!/usr/bin/env python3
"""
Script de test simple pour vérifier les modules NLP
"""

import sys
import os
sys.path.append('src')

def test_basic_imports():
    """Test des imports de base"""
    print("🔍 Test des imports de base...")
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ Pandas, NumPy, Matplotlib OK")
    except Exception as e:
        print(f"❌ Erreur imports de base: {e}")
        return False
    
    try:
        import nltk
        import spacy
        print("✅ NLTK, spaCy OK")
    except Exception as e:
        print(f"❌ Erreur imports NLP: {e}")
        return False
    
    return True

def test_data_preprocessing():
    """Test du module de préprocessing"""
    print("\n🔍 Test du module de préprocessing...")
    
    try:
        from data_preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        test_text = "Ceci est un test! Avec de la ponctuation..."
        result = preprocessor.preprocess_text(test_text)
        
        print(f"✅ Préprocessing OK: '{result}'")
        return True
    except Exception as e:
        print(f"❌ Erreur préprocessing: {e}")
        return False

def test_sentiment_analysis():
    """Test du module d'analyse de sentiment"""
    print("\n🔍 Test du module d'analyse de sentiment...")
    
    try:
        from sentiment_analysis import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        test_text = "J'adore ce produit!"
        result = analyzer.textblob_sentiment(test_text)
        
        print(f"✅ Analyse de sentiment OK: {result['sentiment']}")
        return True
    except Exception as e:
        print(f"❌ Erreur analyse de sentiment: {e}")
        return False

def test_spacy_model():
    """Test du modèle spaCy"""
    print("\n🔍 Test du modèle spaCy...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Apple Inc. was founded by Steve Jobs.")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        print(f"✅ Modèle spaCy OK: {entities}")
        return True
    except Exception as e:
        print(f"❌ Erreur modèle spaCy: {e}")
        return False

def main():
    """Fonction principale"""
    print("🚀 Test des Modules NLP")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_data_preprocessing,
        test_sentiment_analysis,
        test_spacy_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Résultats: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("\n📝 Prochaines étapes:")
        print("  1. Ouvrir Jupyter: jupyter notebook")
        print("  2. Exécuter: notebooks/01_data_exploration_fixed.ipynb")
    else:
        print("⚠️ Certains tests ont échoué.")
        print("🔧 Vérifiez l'installation des dépendances.")

if __name__ == "__main__":
    main()
