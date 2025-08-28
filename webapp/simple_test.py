#!/usr/bin/env python3
"""
Simple test script for TruthLens FastAPI Backend
"""
import requests
import json
import time

def test_api():
    """Test the FastAPI backend."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing TruthLens FastAPI Backend")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   TruthLens available: {data.get('truthlens_available', False)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server")
        print("   Make sure the FastAPI server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Predict endpoint
    print("\n2. Testing predict endpoint...")
    test_data = {
        "text": "COVID-19 vaccines cause autism in children.",
        "input_type": "text",
        "max_claims": 1,
        "max_evidence_per_claim": 1
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Predict endpoint working")
            print(f"   Claims processed: {data.get('summary', {}).get('claims_processed', 0)}")
            print(f"   Total time: {data.get('summary', {}).get('total_processing_time', 0):.3f}s")
            print(f"   Phases completed: {data.get('summary', {}).get('phases_completed', 0)}")
            
            # Show pipeline phases
            pipeline_results = data.get('pipeline_results', {})
            for phase_name, phase_data in pipeline_results.items():
                status = phase_data.get('status', 'unknown')
                proc_time = phase_data.get('processing_time', 0)
                print(f"     {phase_name}: {status} ({proc_time:.3f}s)")
            
            return True
        else:
            print(f"❌ Predict failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    if success:
        print("\n🎉 All tests passed! FastAPI backend is working correctly.")
        print("\n🚀 Next steps:")
        print("1. Open webapp/frontend/index.html in your browser")
        print("2. Test with Postman or other API tools")
        print("3. Deploy to production")
    else:
        print("\n❌ Tests failed. Check the output above for details.")
