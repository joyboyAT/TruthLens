#!/usr/bin/env python3
"""
Simple test script for TruthLens FastAPI app
Tests the basic functionality without starting a server.
"""

import asyncio
import json
from app_clean import app, search_newsapi, search_guardian, calculate_similarity

async def test_api_functions():
    """Test the API functions directly."""
    
    print("🧪 TESTING TRUTHLENS API FUNCTIONS")
    print("="*50)
    
    # Test 1: Test similarity calculation
    print("\n1️⃣ Testing similarity calculation...")
    claim = "COVID-19 vaccines cause autism"
    title = "Study finds no link between COVID-19 vaccines and autism"
    
    similarity = calculate_similarity(claim, title)
    print(f"   Claim: {claim}")
    print(f"   Title: {title}")
    print(f"   Similarity: {similarity:.3f}")
    
    if similarity > 0:
        print("   ✅ Similarity calculation working")
    else:
        print("   ❌ Similarity calculation failed")
    
    # Test 2: Test News API search (mock)
    print("\n2️⃣ Testing News API search function...")
    try:
        # This will make a real API call
        articles = await search_newsapi("COVID-19", max_results=3)
        print(f"   News API returned {len(articles)} articles")
        
        if articles:
            print("   ✅ News API search working")
            for i, article in enumerate(articles[:2], 1):
                print(f"     {i}. {article['title'][:60]}...")
        else:
            print("   ⚠️ News API returned no articles (may be rate limited)")
            
    except Exception as e:
        print(f"   ❌ News API search failed: {e}")
    
    # Test 3: Test Guardian API search (mock)
    print("\n3️⃣ Testing Guardian API search function...")
    try:
        # This will make a real API call
        articles = await search_guardian("COVID-19", max_results=3)
        print(f"   Guardian API returned {len(articles)} articles")
        
        if articles:
            print("   ✅ Guardian API search working")
            for i, article in enumerate(articles[:2], 1):
                print(f"     {i}. {article['title'][:60]}...")
        else:
            print("   ⚠️ Guardian API returned no articles")
            
    except Exception as e:
        print(f"   ❌ Guardian API search failed: {e}")
    
    # Test 4: Test the complete verification flow
    print("\n4️⃣ Testing complete verification flow...")
    try:
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            print("   ✅ Health endpoint working")
            print(f"   Status: {response.json()['status']}")
        else:
            print(f"   ❌ Health endpoint failed: {response.status_code}")
        
        # Test root endpoint
        response = client.get("/")
        if response.status_code == 200:
            print("   ✅ Root endpoint working")
            print(f"   Message: {response.json()['message']}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
        
        # Test sources endpoint
        response = client.get("/sources")
        if response.status_code == 200:
            print("   ✅ Sources endpoint working")
            sources = response.json()['available_sources']
            for source in sources:
                print(f"     - {source['name']}: {source['status']}")
        else:
            print(f"   ❌ Sources endpoint failed: {response.status_code}")
            
    except ImportError:
        print("   ⚠️ FastAPI test client not available, skipping endpoint tests")
    except Exception as e:
        print(f"   ❌ Endpoint testing failed: {e}")
    
    print("\n" + "="*50)
    print("🎯 FUNCTIONALITY TESTING COMPLETED")
    print("="*50)

def test_models():
    """Test the Pydantic models."""
    print("\n🔧 TESTING PYDANTIC MODELS")
    print("="*30)
    
    try:
        from app_clean import ClaimInput, ArticleDetail, VerificationResponse
        
        # Test ClaimInput
        claim_data = {"claim": "COVID-19 vaccines cause autism"}
        claim_input = ClaimInput(**claim_data)
        print(f"   ✅ ClaimInput model working: {claim_input.claim}")
        
        # Test ArticleDetail
        article_data = {
            "title": "Test Article",
            "url": "https://example.com",
            "source": "Test Source",
            "source_name": "TestAPI",
            "similarity_score": 0.85
        }
        article_detail = ArticleDetail(**article_data)
        print(f"   ✅ ArticleDetail model working: {article_detail.title}")
        
        # Test VerificationResponse
        response_data = {
            "claim": "Test claim",
            "sources_checked": ["News API"],
            "verification_badge": "✅ Verified",
            "evidence_strength": "Strong",
            "details": [article_detail],
            "processing_time": 1.5,
            "timestamp": "2024-01-01T00:00:00",
            "total_articles": 1,
            "source_breakdown": {"TestAPI": 1}
        }
        verification_response = VerificationResponse(**response_data)
        print(f"   ✅ VerificationResponse model working: {verification_response.evidence_strength}")
        
    except Exception as e:
        print(f"   ❌ Model testing failed: {e}")

if __name__ == "__main__":
    print("🚀 TRUTHLENS API FUNCTIONALITY TEST")
    print("="*50)
    
    # Test models first
    test_models()
    
    # Test async functions
    asyncio.run(test_api_functions())
    
    print("\n🎉 All tests completed!")
    print("\nTo start the server, run:")
    print("   uvicorn app_clean:app --host 127.0.0.1 --port 8000")
    print("\nThen test with:")
    print("   curl -X POST \"http://localhost:8000/verify\" \\")
    print("        -H \"Content-Type: application/json\" \\")
    print("        -d '{\"claim\": \"COVID-19 vaccines cause autism\"}'")
