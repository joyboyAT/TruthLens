#!/usr/bin/env python3
"""
Automated TruthLens Pipeline Test
Runs the pipeline with predefined test claims for automated testing.
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

from tests.test_simple_pipeline import SimpleTruthLensPipeline, SimplePipelineResult

def test_multiple_claims():
    """Test the pipeline with multiple predefined claims."""
    
    test_claims = [
        "COVID-19 vaccines cause autism in children according to recent studies.",
        "5G towers emit dangerous radiation that causes cancer in nearby residents.",
        "Water boils at 100 degrees Celsius at sea level under normal atmospheric pressure.",
        "The Earth is flat and NASA has been hiding this fact for decades.",
        "Drinking 8 glasses of water per day is essential for good health.",
        "Modi embarks on Japan, China trip; to meet Xi & Putin but won't attend Beijing's contentious Victory Day parade"
    ]
    
    print("Automated TruthLens Pipeline Test")
    print("="*60)
    
    # Initialize pipeline
    pipeline = SimpleTruthLensPipeline()
    
    results = []
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n--- Test {i}/{len(test_claims)} ---")
        print(f"Claim: {claim}")
        
        # Process claim
        result = pipeline.process_claim(claim)
        results.append(result)
        
        # Print summary
        print(f"Verdict: {result.verdict}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        print(f"Success: {result.success}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    successful_tests = sum(1 for r in results if r.success)
    print(f"Successful Tests: {successful_tests}/{len(results)}")
    
    verdict_counts = {}
    for result in results:
        verdict = result.verdict
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    
    print("\nVerdict Distribution:")
    for verdict, count in verdict_counts.items():
        print(f"  {verdict}: {count}")
    
    avg_confidence = sum(r.confidence for r in results if r.success) / max(1, successful_tests)
    print(f"\nAverage Confidence: {avg_confidence:.1%}")
    
    total_time = sum(r.processing_time for r in results)
    print(f"Total Processing Time: {total_time:.3f}s")
    print(f"Average Time per Claim: {total_time/len(results):.3f}s")
    
    # Save detailed results
    output_file = "automated_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return successful_tests == len(results)

def test_single_claim(claim_text: str):
    """Test the pipeline with a single claim."""
    print(f"Testing claim: {claim_text}")
    
    pipeline = SimpleTruthLensPipeline()
    result = pipeline.process_claim(claim_text)
    
    print(f"Verdict: {result.verdict}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Explanation: {result.explanation}")
    
    return result.success

def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Test single claim provided as argument
        claim = " ".join(sys.argv[1:])
        success = test_single_claim(claim)
    else:
        # Test multiple claims
        success = test_multiple_claims()
    
    if success:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
