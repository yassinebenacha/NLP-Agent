#!/usr/bin/env python3
"""
Installation verification script for NLP Analysis Project
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            module = importlib.import_module(module_name, package_name)
        else:
            module = importlib.import_module(module_name)
        return True, f"✅ {module_name} imported successfully"
    except ImportError as e:
        return False, f"❌ {module_name} failed to import: {e}"
    except Exception as e:
        return False, f"❌ {module_name} error: {e}"

def main():
    """Run installation verification"""
    print("🔍 NLP Project Installation Verification")
    print("=" * 50)
    
    # Core libraries
    core_libs = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'plotly'
    ]
    
    # NLP libraries
    nlp_libs = [
        'nltk',
        'spacy',
        'textblob',
        'transformers',
        'torch',
        'gensim'
    ]
    
    # Optional libraries
    optional_libs = [
        'wordcloud',
        'jupyter',
        'pytest'
    ]
    
    all_passed = True
    
    print("\n📦 Core Libraries:")
    for lib in core_libs:
        success, message = test_import(lib)
        print(f"  {message}")
        if not success:
            all_passed = False
    
    print("\n🔤 NLP Libraries:")
    for lib in nlp_libs:
        success, message = test_import(lib)
        print(f"  {message}")
        if not success:
            all_passed = False
    
    print("\n⚙️ Optional Libraries:")
    for lib in optional_libs:
        success, message = test_import(lib)
        print(f"  {message}")
    
    # Test spaCy model
    print("\n🧠 spaCy Model Test:")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Apple Inc. was founded by Steve Jobs.")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"  ✅ spaCy model works! Found entities: {entities}")
    except Exception as e:
        print(f"  ❌ spaCy model failed: {e}")
        all_passed = False
    
    # Test our custom modules
    print("\n🏗️ Custom Modules:")
    sys.path.append('src')
    
    custom_modules = [
        'config',
        'data_preprocessing',
        'sentiment_analysis',
        'topic_modeling',
        'ner',
        'summarization',
        'evaluation',
        'visualization'
    ]
    
    for module in custom_modules:
        success, message = test_import(module)
        print(f"  {message}")
        if not success:
            all_passed = False
    
    # Test basic functionality
    print("\n🧪 Basic Functionality Test:")
    try:
        from data_preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess_text("This is a test sentence!")
        print(f"  ✅ Text preprocessing works: '{result}'")
    except Exception as e:
        print(f"  ❌ Text preprocessing failed: {e}")
        all_passed = False
    
    try:
        from sentiment_analysis import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        result = analyzer.textblob_sentiment("I love this!")
        print(f"  ✅ Sentiment analysis works: {result['sentiment']}")
    except Exception as e:
        print(f"  ❌ Sentiment analysis failed: {e}")
        all_passed = False
    
    # Final result
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL CORE COMPONENTS WORKING! Ready to run notebooks.")
        print("\n📝 Next steps:")
        print("  1. Run: jupyter notebook")
        print("  2. Open: notebooks/01_data_exploration.ipynb")
        print("  3. Execute all cells to verify functionality")
    else:
        print("⚠️ Some components failed. Check error messages above.")
        print("\n🔧 Try installing missing packages:")
        print("  pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    main()
