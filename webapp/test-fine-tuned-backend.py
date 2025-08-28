#!/usr/bin/env python3
"""
Test script for TruthLens Backend with Fine-tuned Model
"""
import requests
import json
import time
from pathlib import Path

def test_backend_health(base_url="http://localhost:5000"):
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   TruthLens available: {data.get('truthlens_available', False)}")
            print(f"   Fine-tuned available: {data.get('fine_tuned_available', False)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend server")
        print("   Make sure the backend is running on http://localhost:5000")
        return False

def test_model_status(base_url="http://localhost:5000"):
    """Test the model status endpoint."""
    print("\n🔍 Testing model status...")
    
    try:
        response = requests.get(f"{base_url}/api/model-status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model status retrieved")
            print(f"   Fine-tuned model: {data.get('fine_tuned_model', {}).get('status', 'unknown')}")
            print(f"   Accuracy: {data.get('fine_tuned_model', {}).get('accuracy', 'unknown')}")
            print(f"   Vector retriever: {data.get('vector_retriever', {}).get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Model status failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend server")
        return False

def test_claim_extraction(base_url="http://localhost:5000"):
    """Test claim extraction with fine-tuned model."""
    print("\n🔍 Testing claim extraction...")
    
    test_texts = [
        "COVID-19 vaccines cause autism in children.",
        "5G towers emit dangerous radiation that causes cancer.",
        "The weather is nice today.",
        "Scientists discovered a new planet in our solar system.",
        "I like pizza."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Test {i}: '{text}'")
        
        try:
            response = requests.post(
                f"{base_url}/api/claims",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                claims = data.get('claims', [])
                detection_method = data.get('model_info', {}).get('detection_method', 'unknown')
                
                print(f"      ✅ Extracted {len(claims)} claims")
                print(f"      Method: {detection_method}")
                print(f"      Time: {data.get('processing_time', 0):.3f}s")
                
                for j, claim in enumerate(claims, 1):
                    confidence = claim.get('checkworthiness', 0)
                    print(f"        Claim {j}: {claim.get('text', '')[:50]}... (confidence: {confidence:.3f})")
            else:
                print(f"      ❌ Failed: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("      ❌ Connection error")
            return False
    
    return True

def test_full_analysis(base_url="http://localhost:5000"):
    """Test full text analysis with claims and evidence."""
    print("\n🔍 Testing full analysis...")
    
    test_text = "COVID-19 vaccines cause autism in children. 5G towers emit dangerous radiation that causes cancer."
    
    try:
        response = requests.post(
            f"{base_url}/api/analyze",
            json={"text": test_text},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            claims = data.get('claims', [])
            evidence = data.get('total_evidence', 0)
            detection_method = data.get('model_info', {}).get('detection_method', 'unknown')
            
            print(f"✅ Full analysis completed")
            print(f"   Claims extracted: {len(claims)}")
            print(f"   Evidence retrieved: {evidence}")
            print(f"   Detection method: {detection_method}")
            print(f"   Processing time: {data.get('processing_time', 0):.3f}s")
            
            # Show details
            for i, claim in enumerate(claims, 1):
                print(f"\n   Claim {i}: {claim.get('text', '')}")
                print(f"   Confidence: {claim.get('checkworthiness', 0):.3f}")
                claim_evidence = claim.get('evidence', [])
                print(f"   Evidence sources: {len(claim_evidence)}")
                
                for j, ev in enumerate(claim_evidence[:2], 1):  # Show first 2 evidence
                    print(f"     Evidence {j}: {ev.get('title', '')[:50]}...")
            
            return True
        else:
            print(f"❌ Full analysis failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
        return False

def test_evidence_search(base_url="http://localhost:5000"):
    """Test evidence search functionality."""
    print("\n🔍 Testing evidence search...")
    
    test_query = "COVID-19 vaccines autism"
    
    try:
        response = requests.post(
            f"{base_url}/api/evidence",
            json={"query": test_query, "max_results": 3},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            evidence = data.get('evidence', [])
            
            print(f"✅ Evidence search completed")
            print(f"   Results: {len(evidence)}")
            print(f"   Processing time: {data.get('processing_time', 0):.3f}s")
            
            for i, ev in enumerate(evidence, 1):
                print(f"   Evidence {i}: {ev.get('title', '')[:50]}...")
                print(f"     Source: {ev.get('source_type', 'unknown')}")
                print(f"     Score: {ev.get('relevance_score', 0):.3f}")
            
            return True
        else:
            print(f"❌ Evidence search failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
        return False

def main():
    """Main test function."""
    print("🧪 Testing TruthLens Backend with Fine-tuned Model")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    # Run tests
    tests = [
        ("Health Check", test_backend_health, base_url),
        ("Model Status", test_model_status, base_url),
        ("Claim Extraction", test_claim_extraction, base_url),
        ("Evidence Search", test_evidence_search, base_url),
        ("Full Analysis", test_full_analysis, base_url)
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
        print("🎉 All tests passed! Fine-tuned backend is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
