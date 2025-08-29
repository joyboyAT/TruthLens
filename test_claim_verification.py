#!/usr/bin/env python3
"""
Simple test script for TruthLens claim verification
"""

import requests
import json
import time

def test_claim_verification():
    """Test the claim verification endpoint"""
    
    # Test claims
    test_claims = [
        "The Earth is flat",
        "Climate change is real",
        "Vaccines cause autism",
        "The moon landing was fake"
    ]
    
    base_url = "http://localhost:8000"
    
    # First check if server is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server health check failed")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test each claim
    for claim in test_claims:
        print(f"\n🔍 Testing claim: '{claim}'")
        
        try:
            payload = {
                "claim": claim,
                "context": "Test claim from verification script"
            }
            
            response = requests.post(
                f"{base_url}/verify",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Verdict: {result.get('verdict', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.1%}")
                print(f"   Evidence Strength: {result.get('evidence_strength', 'Unknown')}")
                print(f"   Articles Found: {result.get('total_articles', 0)}")
                print(f"   Processing Time: {result.get('processing_time', 0):.2f}s")
                
                # Show stance distribution
                stance_dist = result.get('stance_distribution', {})
                if stance_dist:
                    print(f"   Stance Distribution: {stance_dist}")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        
        # Wait between requests
        time.sleep(2)

if __name__ == "__main__":
    print("🚀 TruthLens Claim Verification Test")
    print("=" * 50)
    test_claim_verification()
