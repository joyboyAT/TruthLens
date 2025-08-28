#!/usr/bin/env python3
"""
Startup script for TruthLens FastAPI Backend
"""
import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
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
        print("Install them with: pip install -r requirements_fastapi.txt")
        return False
    
    return True

def check_environment():
    """Check environment variables."""
    print("\n🔍 Checking environment...")
    
    # Check port
    port = os.getenv('PORT', '8000')
    print(f"✅ Server will run on port: {port}")
    
    # Check for API keys (optional)
    serper_key = os.getenv('SERPER_API_KEY')
    bing_key = os.getenv('BING_API_KEY')
    
    if serper_key:
        print("✅ SERPER_API_KEY found")
    elif bing_key:
        print("✅ BING_API_KEY found")
    else:
        print("⚠️ No search API keys found (SERPER_API_KEY or BING_API_KEY)")
        print("Web search functionality will be disabled.")

def start_fastapi():
    """Start the FastAPI server."""
    print("\n🚀 Starting TruthLens FastAPI Backend")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return False
    
    # Check environment
    check_environment()
    
    # Start the server
    fastapi_script = Path(__file__).parent / "truthlens_fastapi.py"
    
    if not fastapi_script.exists():
        print(f"\n❌ FastAPI script not found: {fastapi_script}")
        return False
    
    print(f"\n🎯 Starting FastAPI server: {fastapi_script}")
    print("=" * 60)
    
    try:
        # Run the FastAPI server
        subprocess.run([sys.executable, str(fastapi_script)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 FastAPI server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FastAPI server failed to start: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("🎯 TruthLens FastAPI Backend")
    print("=" * 50)
    
    success = start_fastapi()
    
    if success:
        print("\n✅ FastAPI server started successfully")
    else:
        print("\n❌ Failed to start FastAPI server")
        sys.exit(1)

if __name__ == "__main__":
    main()
