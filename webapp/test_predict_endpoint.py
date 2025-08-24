#!/usr/bin/env python3
"""
Simple test script for the TruthLens FastAPI predict endpoint
"""

import requests
import json
import time

def test_predict_endpoint():
    """Test the /predict endpoint with a sample claim."""
    
    # Test data
    test_data = {
        "text": "COVID-19 vaccines cause autism in children.",
        "input_type": "text",
        "max_claims": 1,
        "max_evidence_per_claim": 1
    }
    
    # API endpoint
    url = "http://localhost:8000/predict"
    
    try:
        print("🧪 Testing TruthLens FastAPI /predict endpoint...")
        print(f"📤 Sending request to: {url}")
        print(f"📝 Test data: {json.dumps(test_data, indent=2)}")
        print("-" * 60)
        
        # Send POST request
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📋 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Received response:")
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"❌ ERROR! Status code: {response.status_code}")
            print(f"Error response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running on http://localhost:8000?")
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Server might be still loading models.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_health_endpoint():
    """Test the /health endpoint."""
    
    try:
        print("\n🏥 Testing /health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check successful:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("🚀 TruthLens FastAPI Test Script")
    print("=" * 60)
    
    # Test health first
    test_health_endpoint()
    
    # Wait a bit for server to be ready
    print("\n⏳ Waiting 5 seconds for server to be ready...")
    time.sleep(5)
    
    # Test predict endpoint
    test_predict_endpoint()
