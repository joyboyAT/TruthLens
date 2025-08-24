#!/usr/bin/env python3
"""
Debug script to test claim detection logic
"""

import sys
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

def test_claim_detection():
    """Test the claim detection logic with our test sentence."""
    
    try:
        from extractor.claim_detector import is_claim, _heuristic_is_claim
        
        test_sentence = "COVID-19 vaccines cause autism in children."
        
        print("🔍 Testing Claim Detection Logic")
        print("=" * 50)
        print(f"Test sentence: '{test_sentence}'")
        print()
        
        # Test heuristic detection
        print("📊 Testing Heuristic Detection:")
        is_claim_heuristic, prob_heuristic = _heuristic_is_claim(test_sentence)
        print(f"  Is claim: {is_claim_heuristic}")
        print(f"  Probability: {prob_heuristic:.3f}")
        print()
        
        # Test full detection (with ML if available)
        print("🤖 Testing Full Detection (with ML):")
        is_claim_full, prob_full = is_claim(test_sentence)
        print(f"  Is claim: {is_claim_full}")
        print(f"  Probability: {prob_full:.3f}")
        print()
        
        # Test with different thresholds
        print("🎯 Testing Different Thresholds:")
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            is_claim_thresh, prob_thresh = is_claim(test_sentence, threshold=threshold)
            print(f"  Threshold {threshold}: {is_claim_thresh} (prob: {prob_thresh:.3f})")
        print()
        
        # Test with other sentences for comparison
        print("🔍 Testing Other Sentences:")
        test_sentences = [
            "The Earth revolves around the Sun.",
            "Water boils at 100 degrees Celsius.",
            "I love pizza.",
            "What a nice day!",
            "5G towers emit dangerous radiation.",
            "Government announces new economic package."
        ]
        
        for sentence in test_sentences:
            is_claim_result, prob_result = is_claim(sentence)
            print(f"  '{sentence}' -> {is_claim_result} (prob: {prob_result:.3f})")
            
    except Exception as e:
        print(f"❌ Error testing claim detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_claim_detection()
