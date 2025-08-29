#!/usr/bin/env python3
"""
Google Fact Check API Test
Tests the Google Fact Check API with known fact-checked claims.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Google Fact Check API
try:
    from src.verification.google_factcheck_api import GoogleFactCheckAPI, FactCheckResult
    GOOGLE_FACTCHECK_AVAILABLE = True
except ImportError:
    GOOGLE_FACTCHECK_AVAILABLE = False
    print("Error: Google Fact Check API not available")

def test_google_factcheck_api():
    """Test the Google Fact Check API with various claims."""
    
    # Google Fact Check API key
    api_key = "AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc"
    
    if not GOOGLE_FACTCHECK_AVAILABLE:
        print("Google Fact Check API not available")
        return False
    
    try:
        # Initialize API
        print("Initializing Google Fact Check API...")
        factcheck_api = GoogleFactCheckAPI(api_key)
        print("✅ Google Fact Check API initialized successfully")
        
        # Test claims that are likely to have been fact-checked
        test_claims = [
            "COVID-19 vaccines cause autism",
            "5G causes coronavirus",
            "Bill Gates wants to microchip people",
            "Hydroxychloroquine cures COVID-19",
            "The Earth is flat",
            "Climate change is a hoax",
            "Vaccines contain microchips",
            "Face masks cause oxygen deprivation",
            "The moon landing was fake",
            "Chemtrails are real"
        ]
        
        print(f"\nTesting {len(test_claims)} claims with Google Fact Check API...")
        print("="*80)
        
        results = []
        
        for i, claim in enumerate(test_claims, 1):
            print(f"\n--- Test {i}/{len(test_claims)} ---")
            print(f"Claim: {claim}")
            
            try:
                # Verify claim
                result = factcheck_api.verify_claim(claim)
                
                if result:
                    print(f"✅ Found fact-check result!")
                    print(f"   Verdict: {result.verdict}")
                    print(f"   Confidence: {result.confidence:.1%}")
                    print(f"   Publisher: {result.publisher}")
                    print(f"   Rating: {result.rating}")
                    print(f"   URL: {result.url}")
                    print(f"   Explanation: {result.explanation}")
                    
                    results.append({
                        "claim": claim,
                        "found": True,
                        "verdict": result.verdict,
                        "confidence": result.confidence,
                        "publisher": result.publisher,
                        "rating": result.rating,
                        "url": result.url,
                        "explanation": result.explanation
                    })
                else:
                    print("❌ No fact-check result found")
                    results.append({
                        "claim": claim,
                        "found": False,
                        "verdict": "NOT ENOUGH INFO",
                        "confidence": 0.0,
                        "publisher": "None",
                        "rating": "None",
                        "url": "None",
                        "explanation": "No fact-check result found in Google Fact Check API"
                    })
                
            except Exception as e:
                print(f"❌ Error verifying claim: {e}")
                results.append({
                    "claim": claim,
                    "found": False,
                    "error": str(e)
                })
            
            # Rate limiting
            if i < len(test_claims):
                time.sleep(1)
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        found_count = sum(1 for r in results if r.get("found", False))
        print(f"Claims with fact-check results: {found_count}/{len(test_claims)}")
        
        if found_count > 0:
            verdict_counts = {}
            for result in results:
                if result.get("found", False):
                    verdict = result.get("verdict", "UNKNOWN")
                    verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            
            print("\nVerdict Distribution:")
            for verdict, count in verdict_counts.items():
                print(f"  {verdict}: {count}")
            
            publishers = set()
            for result in results:
                if result.get("found", False):
                    publishers.add(result.get("publisher", "Unknown"))
            
            print(f"\nPublishers found: {len(publishers)}")
            for publisher in sorted(publishers):
                print(f"  - {publisher}")
        
        # Save results
        output_file = "google_factcheck_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        return found_count > 0
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_claim(claim_text: str):
    """Test a single claim with Google Fact Check API."""
    api_key = "AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc"
    
    if not GOOGLE_FACTCHECK_AVAILABLE:
        print("Google Fact Check API not available")
        return False
    
    try:
        factcheck_api = GoogleFactCheckAPI(api_key)
        print(f"Testing claim: {claim_text}")
        
        result = factcheck_api.verify_claim(claim_text)
        
        if result:
            print(f"✅ Found fact-check result!")
            print(f"   Verdict: {result.verdict}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Publisher: {result.publisher}")
            print(f"   Rating: {result.rating}")
            print(f"   URL: {result.url}")
            print(f"   Explanation: {result.explanation}")
            return True
        else:
            print("❌ No fact-check result found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Test single claim provided as argument
        claim = " ".join(sys.argv[1:])
        success = test_single_claim(claim)
    else:
        # Test multiple claims
        success = test_google_factcheck_api()
    
    if success:
        print("\n✅ Google Fact Check API test completed successfully!")
        return 0
    else:
        print("\n❌ Google Fact Check API test failed!")
        return 1

if __name__ == "__main__":
    exit(main())
