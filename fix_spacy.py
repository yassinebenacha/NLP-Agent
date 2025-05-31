#!/usr/bin/env python3
"""
Fix spaCy installation issues
The "cannot import name util" error is usually due to version conflicts
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def fix_spacy():
    """Fix spaCy installation issues"""
    print("🔧 Fixing spaCy Installation")
    print("=" * 40)
    
    # Step 1: Uninstall existing spaCy
    print("🗑️ Removing existing spaCy installation...")
    run_command("pip uninstall spacy -y", "Uninstalling spaCy")
    
    # Step 2: Clean pip cache
    print("🧹 Cleaning pip cache...")
    run_command("pip cache purge", "Cleaning pip cache")
    
    # Step 3: Install latest spaCy
    print("📦 Installing latest spaCy...")
    success = run_command("pip install spacy", "Installing spaCy")
    
    if success:
        # Step 4: Download English model
        print("📥 Downloading English model...")
        model_success = run_command("python -m spacy download en_core_web_sm", "Downloading en_core_web_sm")
        
        if not model_success:
            print("⚠️ Model download failed, trying alternative...")
            run_command("python -m spacy download en", "Downloading en model")
    
    # Step 5: Test installation
    print("\n🧪 Testing spaCy installation...")
    test_spacy()

def test_spacy():
    """Test if spaCy works correctly"""
    try:
        import spacy
        print("✅ spaCy import successful")
        
        # Test model loading
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ en_core_web_sm model loaded")
            
            # Test basic functionality
            doc = nlp("Apple Inc. is a technology company.")
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            print(f"✅ NER test successful: {entities}")
            
        except OSError:
            try:
                nlp = spacy.load("en")
                print("✅ en model loaded (alternative)")
            except OSError:
                print("❌ No spaCy models available")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ spaCy import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ spaCy test failed: {e}")
        return False

def alternative_solution():
    """Provide alternative if spaCy still doesn't work"""
    print("\n🔄 Alternative Solution")
    print("=" * 40)
    
    print("If spaCy still doesn't work, the app will use pattern-based NER.")
    print("This provides basic entity recognition without spaCy.")
    
    print("\n📝 Manual spaCy installation (if needed):")
    print("1. Create new environment:")
    print("   python -m venv spacy_env")
    print("   source spacy_env/bin/activate  # Windows: spacy_env\\Scripts\\activate")
    print("2. Install spaCy:")
    print("   pip install spacy")
    print("   python -m spacy download en_core_web_sm")
    
    print("\n💡 The Streamlit app will work fine without spaCy!")
    print("   - Pattern-based NER will be used instead")
    print("   - All other features remain fully functional")

def main():
    """Main function"""
    print("🩺 spaCy Diagnostic and Fix Tool")
    print("=" * 50)
    
    # Check current status
    print("🔍 Checking current spaCy status...")
    if test_spacy():
        print("🎉 spaCy is already working correctly!")
        return
    
    # Try to fix
    fix_spacy()
    
    # Final test
    print("\n🏁 Final Test")
    print("=" * 20)
    if test_spacy():
        print("🎉 spaCy fix successful!")
        print("✅ Your Streamlit app should now show 'spaCy NER loaded'")
    else:
        print("⚠️ spaCy fix unsuccessful")
        alternative_solution()
    
    print("\n🚀 Next steps:")
    print("1. Restart your Streamlit app: streamlit run app.py")
    print("2. Check the sidebar for updated status")
    print("3. The app will work even if spaCy shows warnings")

if __name__ == "__main__":
    main()
