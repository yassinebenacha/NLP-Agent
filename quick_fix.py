#!/usr/bin/env python3
"""
Quick fix for NLP Agent dependencies
Run this to install missing packages and download spaCy model
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def download_spacy_model():
    """Download spaCy English model"""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy English model downloaded successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to download spaCy model")
        return False

def main():
    print("🔧 Quick Fix for NLP Agent")
    print("=" * 40)
    
    # Essential packages for Streamlit app
    essential_packages = [
        "streamlit",
        "pandas", 
        "numpy",
        "matplotlib",
        "plotly",
        "scikit-learn"
    ]
    
    print("📦 Installing essential packages...")
    for package in essential_packages:
        install_package(package)
    
    # Install spaCy and model
    print("\n🤖 Installing spaCy...")
    if install_package("spacy"):
        print("📥 Downloading spaCy English model...")
        download_spacy_model()
    
    # Install optional packages
    print("\n⚡ Installing optional packages...")
    optional_packages = ["wordcloud", "textblob", "nltk"]
    for package in optional_packages:
        install_package(package)
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except:
        print("⚠️ NLTK data download failed")
    
    print("\n" + "=" * 40)
    print("🎉 Quick fix completed!")
    print("\n🚀 Now restart your Streamlit app:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
