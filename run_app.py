#!/usr/bin/env python3
"""
NLP Agent Launcher
Simple script to launch the Streamlit app with proper setup
"""

import subprocess
import sys
import os

def check_streamlit():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_requirements():
    """Install basic requirements"""
    print("📦 Installing basic requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "streamlit", "pandas", "numpy", "matplotlib", 
            "seaborn", "plotly", "scikit-learn", "nltk", "textblob"
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main launcher function"""
    print("🚀 NLP Agent Launcher")
    print("=" * 40)
    
    # Check if Streamlit is available
    if not check_streamlit():
        print("⚠️ Streamlit not found. Installing requirements...")
        if install_requirements():
            print("✅ Requirements installed successfully!")
        else:
            print("❌ Failed to install requirements.")
            print("Please run: pip install -r requirements_streamlit.txt")
            return
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("❌ app.py not found in current directory")
        return
    
    print("🌟 Launching NLP Agent...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("  - Use Ctrl+C to stop the app")
    print("  - Refresh browser if needed")
    print("  - Check terminal for any error messages")
    print("\n" + "=" * 40)
    
    # Launch Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 NLP Agent stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main()
