#!/usr/bin/env python3
"""
TruthLens API Deployment Script
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check required packages
    required_packages = [
        'fastapi', 'uvicorn', 'requests', 'sentence_transformers',
        'torch', 'transformers', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def check_api_keys():
    """Check if API keys are configured."""
    print("🔍 Checking API keys...")
    
    required_keys = ['NEWS_API_KEY', 'GUARDIAN_API_KEY', 'GOOGLE_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("Set them as environment variables or in .env file")
        return False
    
    print("✅ API keys configured")
    return True

def test_api():
    """Test the API functionality."""
    print("🔍 Testing API...")
    
    try:
        # Start server in background
        print("Starting server...")
        server_process = subprocess.Popen([
            sys.executable, "app_enhanced.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(15)
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code != 200:
            print("❌ Health check failed")
            return False
        
        # Test claim verification
        test_claim = {
            "claim": "The Earth is flat",
            "context": "Test claim"
        }
        
        response = requests.post(
            "http://localhost:8000/verify",
            json=test_claim,
            timeout=30
        )
        
        if response.status_code != 200:
            print("❌ Claim verification failed")
            return False
        
        result = response.json()
        print(f"✅ API test successful - Verdict: {result.get('verdict', 'Unknown')}")
        
        # Stop server
        server_process.terminate()
        server_process.wait()
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def create_production_config():
    """Create production configuration files."""
    print("🔧 Creating production configuration...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path("env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("✅ Created .env file from template")
        else:
            print("⚠️  No env.example found")
    
    # Create Procfile for Heroku
    procfile = Path("Procfile")
    if not procfile.exists():
        procfile.write_text("web: uvicorn app_enhanced:app --host 0.0.0.0 --port $PORT")
        print("✅ Created Procfile for Heroku deployment")
    
    # Create runtime.txt for Python version
    runtime_file = Path("runtime.txt")
    if not runtime_file.exists():
        runtime_file.write_text("python-3.11.7")
        print("✅ Created runtime.txt")

def main():
    """Main deployment script."""
    print("🚀 TruthLens API Deployment Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check API keys
    if not check_api_keys():
        print("⚠️  Continue without API keys? (y/n): ", end="")
        if input().lower() != 'y':
            sys.exit(1)
    
    # Create production config
    create_production_config()
    
    # Test API
    if not test_api():
        print("❌ API test failed")
        sys.exit(1)
    
    print("\n✅ Deployment preparation complete!")
    print("\n📋 Next steps:")
    print("1. Set your API keys in .env file")
    print("2. Choose deployment method:")
    print("   - Docker: docker-compose up --build")
    print("   - Heroku: git push heroku main")
    print("   - Manual: python app_enhanced.py")
    print("3. Test your deployment")
    print("\n📚 See README_DEPLOYMENT.md for detailed instructions")

if __name__ == "__main__":
    main()
