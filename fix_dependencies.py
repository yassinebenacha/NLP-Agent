#!/usr/bin/env python3
"""
Fix dependencies for NLP Agent Streamlit app
This script will install missing packages and download required models
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def install_basic_packages():
    """Install basic required packages"""
    packages = [
        "streamlit",
        "pandas", 
        "numpy",
        "matplotlib",
        "seaborn", 
        "plotly",
        "scikit-learn",
        "nltk",
        "textblob"
    ]
    
    print("📦 Installing basic packages...")
    for package in packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️ Failed to install {package}, but continuing...")

def install_optional_packages():
    """Install optional packages"""
    print("\n🚀 Installing optional advanced packages...")
    
    # Try to install spaCy
    if run_command("pip install spacy", "Installing spaCy"):
        # Try to download the English model
        run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")
    
    # Try to install other optional packages
    optional_packages = ["wordcloud", "transformers", "torch"]
    for package in optional_packages:
        run_command(f"pip install {package}", f"Installing {package} (optional)")

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    nltk_downloads = [
        "punkt",
        "stopwords", 
        "wordnet",
        "averaged_perceptron_tagger",
        "maxent_ne_chunker",
        "words"
    ]
    
    for data in nltk_downloads:
        run_command(f'python -c "import nltk; nltk.download(\'{data}\')"', f"Downloading NLTK {data}")

def create_simple_preprocessor():
    """Create a simplified preprocessor that doesn't require all dependencies"""
    
    simple_preprocessor_code = '''"""
Simple text preprocessor that works without heavy dependencies
"""

import re
import string

class SimpleTextPreprocessor:
    """Simplified text preprocessor"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'this', 'that', 'these', 'those', 'a', 'an', 'as', 'if', 'it', 'its'
        }
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\\S+@\\S+', '', text)
        
        return text
    
    def preprocess_text(self, text):
        """Simple preprocessing"""
        text = self.clean_text(text)
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra spaces
        text = re.sub(r'\\s+', ' ', text).strip()
        
        return text

# Create a fallback TextPreprocessor class
TextPreprocessor = SimpleTextPreprocessor
'''
    
    # Write the simple preprocessor
    with open('src/simple_preprocessor.py', 'w') as f:
        f.write(simple_preprocessor_code)
    
    print("✅ Created simple preprocessor fallback")

def test_imports():
    """Test if imports work"""
    print("\n🧪 Testing imports...")
    
    test_modules = [
        ('streamlit', 'st'),
        ('pandas', 'pd'), 
        ('numpy', 'np'),
        ('matplotlib.pyplot', 'plt'),
        ('plotly.express', 'px'),
        ('sklearn.feature_extraction.text', 'TfidfVectorizer'),
        ('nltk', 'nltk'),
        ('textblob', 'TextBlob')
    ]
    
    for module, alias in test_modules:
        try:
            exec(f"import {module} as {alias}")
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
    
    # Test spaCy separately
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy with en_core_web_sm model")
    except:
        print("❌ spaCy or en_core_web_sm model")

def main():
    """Main function"""
    print("🔧 NLP Agent Dependency Fixer")
    print("=" * 50)
    
    # Install packages
    install_basic_packages()
    install_optional_packages()
    download_nltk_data()
    create_simple_preprocessor()
    
    # Test everything
    test_imports()
    
    print("\n" + "=" * 50)
    print("🎉 Dependency fixing completed!")
    print("\n📝 Next steps:")
    print("1. Restart your Streamlit app: streamlit run app.py")
    print("2. Check the sidebar for module status")
    print("3. If issues persist, some features will use fallback methods")
    
    print("\n💡 Tips:")
    print("- If spaCy model fails, NER will use simple pattern matching")
    print("- If transformers fail, summarization will use TF-IDF")
    print("- The app is designed to work with minimal dependencies")

if __name__ == "__main__":
    main()
