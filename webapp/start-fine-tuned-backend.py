#!/usr/bin/env python3
"""
Startup script for TruthLens Backend with Fine-tuned Model
"""
import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask',
        'flask-cors',
        'torch',
        'transformers',
        'sentence-transformers',
        'numpy',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_model_files():
    """Check if fine-tuned model files exist."""
    print("\n🔍 Checking model files...")
    
    model_path = Path(__file__).parent.parent / "models" / "claim-detection-roberta-base"
    
    if model_path.exists():
        print(f"✅ Fine-tuned model found at: {model_path}")
        return True
    else:
        print(f"⚠️ Fine-tuned model not found at: {model_path}")
        print("The backend will fall back to the base model.")
        return False

def check_environment():
    """Check environment variables."""
    print("\n🔍 Checking environment...")
    
    # Check for API keys
    serper_key = os.getenv('SERPER_API_KEY')
    bing_key = os.getenv('BING_API_KEY')
    
    if serper_key:
        print("✅ SERPER_API_KEY found")
    elif bing_key:
        print("✅ BING_API_KEY found")
    else:
        print("⚠️ No search API keys found (SERPER_API_KEY or BING_API_KEY)")
        print("Web search functionality will be disabled.")
    
    # Check port
    port = os.getenv('PORT', '5000')
    print(f"✅ Server will run on port: {port}")

def start_backend():
    """Start the backend server."""
    print("\n🚀 Starting TruthLens Backend with Fine-tuned Model")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return False
    
    # Check model files
    check_model_files()
    
    # Check environment
    check_environment()
    
    # Start the server
    backend_script = Path(__file__).parent / "truthlens-backend-with-fine-tuned.py"
    
    if not backend_script.exists():
        print(f"\n❌ Backend script not found: {backend_script}")
        return False
    
    print(f"\n🎯 Starting backend: {backend_script}")
    print("=" * 60)
    
    try:
        # Run the backend
        subprocess.run([sys.executable, str(backend_script)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Backend failed to start: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("🎯 TruthLens Backend with Fine-tuned Model")
    print("=" * 50)
    
    success = start_backend()
    
    if success:
        print("\n✅ Backend started successfully")
    else:
        print("\n❌ Failed to start backend")
        sys.exit(1)

if __name__ == "__main__":
    main()
