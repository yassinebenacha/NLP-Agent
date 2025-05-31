#!/usr/bin/env python3
"""
Script d'installation des bibliothèques NLP avancées
"""

import subprocess
import sys

def install_package(package):
    """Installer un package avec pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Installation des bibliothèques avancées"""
    print("🚀 INSTALLATION DES BIBLIOTHÈQUES NLP AVANCÉES")
    print("=" * 60)
    
    # Packages pour BERTopic et modélisation avancée
    advanced_packages = [
        "sentence-transformers",
        "umap-learn", 
        "hdbscan",
        "bertopic",
        "plotly",
        "kaleido"  # Pour sauvegarder les graphiques plotly
    ]
    
    print("📦 Installation des packages avancés...")
    
    success_count = 0
    for package in advanced_packages:
        print(f"\n🔄 Installation de {package}...")
        if install_package(package):
            print(f"✅ {package} installé avec succès!")
            success_count += 1
        else:
            print(f"❌ Échec de l'installation de {package}")
    
    print(f"\n📊 Résultat: {success_count}/{len(advanced_packages)} packages installés")
    
    if success_count == len(advanced_packages):
        print("\n🎉 TOUTES LES BIBLIOTHÈQUES AVANCÉES INSTALLÉES!")
        print("\n✅ Fonctionnalités disponibles:")
        print("  🤖 BERTopic - Modélisation de sujets avec BERT")
        print("  🔤 SentenceTransformers - Embeddings de phrases")
        print("  📊 UMAP - Réduction de dimensionnalité")
        print("  🎯 HDBSCAN - Clustering hiérarchique")
        print("  📈 Plotly - Visualisations interactives")
        
        print("\n🚀 Vous pouvez maintenant utiliser toutes les fonctionnalités avancées!")
        
    elif success_count > 0:
        print(f"\n⚠️ Installation partielle ({success_count}/{len(advanced_packages)})")
        print("🔧 Certaines fonctionnalités avancées peuvent ne pas être disponibles")
        print("📊 Le notebook fonctionnera avec LDA classique")
        
    else:
        print("\n❌ Aucune bibliothèque avancée installée")
        print("📊 Utilisation de LDA classique uniquement")
    
    print("\n📝 Pour tester l'installation:")
    print("  python test_topic_modeling.py")

if __name__ == "__main__":
    main()
