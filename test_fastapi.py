#!/usr/bin/env python3
"""
Test script for TruthLens FastAPI app
Tests the /verify endpoint with the specified claim.
"""

import requests
import json
import time

def test_fastapi_app():
    """Test the FastAPI app endpoints."""
    
    base_url = "http://localhost:8000"
    
    print("🚀 TESTING TRUTHLENS FASTAPI APP")
    print("="*50)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Root endpoint
    print("\n2️⃣ Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"   Message: {response.json()['message']}")
            print(f"   Components available: {response.json()['components_available']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 3: Sources endpoint
    print("\n3️⃣ Testing sources endpoint...")
    try:
        response = requests.get(f"{base_url}/sources")
        if response.status_code == 200:
            print("✅ Sources endpoint passed")
            sources = response.json()['available_sources']
            for source in sources:
                print(f"   - {source['name']}: {source['status']}")
        else:
            print(f"❌ Sources endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Sources endpoint error: {e}")
    
    # Test 4: Verify endpoint with the specified claim
    print("\n4️⃣ Testing verify endpoint...")
    test_claim = "COVID-19 vaccines cause autism"
    
    payload = {
        "claim": test_claim
    }
    
    try:
        print(f"   Testing claim: '{test_claim}'")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/verify",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            print("✅ Verify endpoint passed")
            result = response.json()
            
            print(f"   Claim: {result['claim']}")
            print(f"   Sources checked: {', '.join(result['sources_checked'])}")
            print(f"   Verification badge: {result['verification_badge']}")
            print(f"   Evidence strength: {result['evidence_strength']}")
            print(f"   Total articles: {result['total_articles']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   API response time: {processing_time:.2f}s")
            
            # Show source breakdown
            if result['source_breakdown']:
                print("   Source breakdown:")
                for source, count in result['source_breakdown'].items():
                    print(f"     - {source}: {count} articles")
            
            # Show top article details
            if result['details']:
                print("   Top articles:")
                for i, article in enumerate(result['details'][:3], 1):
                    print(f"     {i}. {article['title'][:60]}...")
                    print(f"        Source: {article['source_name']} | Similarity: {article['similarity_score']:.3f}")
                    print(f"        URL: {article['url']}")
            
        else:
            print(f"❌ Verify endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Verify endpoint error: {e}")
    
    # Test 5: Advanced verification endpoint
    print("\n5️⃣ Testing advanced verification endpoint...")
    try:
        response = requests.post(
            f"{base_url}/verify-advanced",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Advanced verification endpoint passed")
            result = response.json()
            
            print(f"   Verdict: {result['verdict']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   News articles: {result['news_articles']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            
            if result['fact_check_result']:
                print(f"   Fact-check result: {result['fact_check_result'].get('verdict', 'Unknown')}")
            
        elif response.status_code == 503:
            print("⚠️ Advanced verification not available (pipeline not initialized)")
        else:
            print(f"❌ Advanced verification failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Advanced verification error: {e}")
    
    print("\n" + "="*50)
    print("🎯 TESTING COMPLETED")
    print("="*50)

if __name__ == "__main__":
    print("Make sure the FastAPI app is running with: uvicorn app:app --reload")
    print("Then run this test script.")
    
    # Wait a moment for user to start the app
    input("\nPress Enter when the FastAPI app is running...")
    
    test_fastapi_app()
