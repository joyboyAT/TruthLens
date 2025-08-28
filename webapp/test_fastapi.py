#!/usr/bin/env python3
"""
Test script for TruthLens FastAPI Backend
Tests the /predict endpoint locally
"""
import requests
import json
import time
from pathlib import Path

def test_health_check(base_url="http://localhost:8000"):
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   TruthLens available: {data.get('truthlens_available', False)}")
            print(f"   Pipeline components: {data.get('pipeline_components', {})}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to FastAPI server")
        print("   Make sure the server is running on http://localhost:8000")
        return False

def test_pipeline_status(base_url="http://localhost:8000"):
    """Test the pipeline status endpoint."""
    print("\n🔍 Testing pipeline status...")
    
    try:
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Pipeline status retrieved")
            print(f"   TruthLens available: {data.get('truthlens_available', False)}")
            print(f"   Pipeline components: {data.get('pipeline_components', {})}")
            return True
        else:
            print(f"❌ Pipeline status failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to FastAPI server")
        return False

def test_predict_endpoint(base_url="http://localhost:8000"):
    """Test the predict endpoint."""
    print("\n🔍 Testing predict endpoint...")
    
    test_cases = [
        {
            "text": "COVID-19 vaccines cause autism in children.",
            "input_type": "text",
            "max_claims": 3,
            "max_evidence_per_claim": 2
        },
        {
            "text": "5G towers emit dangerous radiation that causes cancer. The weather is nice today.",
            "input_type": "text",
            "max_claims": 5,
            "max_evidence_per_claim": 3
        },
        {
            "text": "Scientists discovered a new planet in our solar system.",
            "input_type": "text",
            "max_claims": 2,
            "max_evidence_per_claim": 2
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: '{test_case['text'][:50]}...'")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/predict",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"      ✅ Prediction successful")
                print(f"      Processing time: {request_time:.3f}s")
                print(f"      Total pipeline time: {data.get('summary', {}).get('total_processing_time', 0):.3f}s")
                print(f"      Claims processed: {data.get('summary', {}).get('claims_processed', 0)}")
                print(f"      Evidence retrieved: {data.get('summary', {}).get('evidence_retrieved', 0)}")
                print(f"      Phases completed: {data.get('summary', {}).get('phases_completed', 0)}")
                
                # Show pipeline phases
                pipeline_results = data.get('pipeline_results', {})
                for phase_name, phase_data in pipeline_results.items():
                    status = phase_data.get('status', 'unknown')
                    proc_time = phase_data.get('processing_time', 0)
                    print(f"        {phase_name}: {status} ({proc_time:.3f}s)")
                
            else:
                print(f"      ❌ Prediction failed: {response.status_code}")
                print(f"      Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("      ❌ Connection error")
            return False
        except Exception as e:
            print(f"      ❌ Exception: {e}")
            return False
    
    return True

def test_curl_commands(base_url="http://localhost:8000"):
    """Show curl commands for testing."""
    print("\n📋 Curl commands for testing:")
    print("=" * 60)
    
    # Health check
    print("1. Health Check:")
    print(f"curl -X GET '{base_url}/health'")
    print()
    
    # Pipeline status
    print("2. Pipeline Status:")
    print(f"curl -X GET '{base_url}/status'")
    print()
    
    # Predict endpoint
    print("3. Predict Endpoint:")
    test_data = {
        "text": "COVID-19 vaccines cause autism in children.",
        "input_type": "text",
        "max_claims": 3,
        "max_evidence_per_claim": 2
    }
    
    curl_command = f"""curl -X POST '{base_url}/predict' \\
  -H 'Content-Type: application/json' \\
  -d '{json.dumps(test_data, indent=2)}'"""
    
    print(curl_command)
    print()
    
    # Pretty print response
    print("4. Pretty print response:")
    print(f"curl -X POST '{base_url}/predict' \\")
    print("  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(test_data)}' | python -m json.tool")

def main():
    """Main test function."""
    print("🧪 Testing TruthLens FastAPI Backend")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Run tests
    tests = [
        ("Health Check", test_health_check, base_url),
        ("Pipeline Status", test_pipeline_status, base_url),
        ("Predict Endpoint", test_predict_endpoint, base_url)
    ]
    
    results = []
    
    for test_name, test_func, *args in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func(*args)
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Show curl commands
    test_curl_commands(base_url)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FastAPI backend is working correctly.")
        print("\n🚀 Next steps:")
        print("1. Start the frontend and connect to this API")
        print("2. Test with Postman or other API testing tools")
        print("3. Deploy to production")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
