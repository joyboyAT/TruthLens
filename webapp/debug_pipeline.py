#!/usr/bin/env python3
"""
Debug script to test the entire pipeline step by step
"""

import sys
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

def test_pipeline_step_by_step():
    """Test the pipeline step by step to see where claims are lost."""
    
    try:
        # Set environment variable to disable ML
        import os
        os.environ["TRUTHLENS_DISABLE_ML"] = "1"
        
        from extractor.pipeline import process_text, _split_sentences
        from extractor.claim_detector import is_claim
        from extractor.claim_extractor import extract_claim_spans
        from extractor.atomicizer import to_atomic
        from extractor.context import analyze_context
        from extractor.ranker import score_claim
        
        test_text = "COVID-19 vaccines cause autism in children."
        
        print("🔍 Testing Pipeline Step by Step")
        print("=" * 50)
        print(f"Test text: '{test_text}'")
        print()
        
        # Step 1: Split sentences
        print("📝 Step 1: Split sentences")
        sentences = _split_sentences(test_text)
        print(f"  Sentences: {sentences}")
        print()
        
        # Step 2: Claim detection for each sentence
        print("🎯 Step 2: Claim detection")
        for i, sent in enumerate(sentences):
            is_claim_result, prob = is_claim(sent)
            print(f"  Sentence {i+1}: '{sent}' -> {is_claim_result} (prob: {prob:.3f})")
        print()
        
        # Step 3: Test the full pipeline
        print("🔄 Step 3: Full pipeline processing")
        results = process_text(test_text)
        print(f"  Total claims found: {len(results)}")
        for i, claim in enumerate(results):
            print(f"  Claim {i+1}:")
            print(f"    ID: {claim['id']}")
            print(f"    Text: '{claim['text']}'")
            print(f"    Subject: '{claim['subject']}'")
            print(f"    Predicate: '{claim['predicate']}'")
            print(f"    Object: '{claim['object']}'")
            print(f"    Checkworthiness: {claim['checkworthiness']:.3f}")
            print()
        
        # Step 4: Test individual components
        print("🔧 Step 4: Individual component testing")
        for sent in sentences:
            print(f"  Testing sentence: '{sent}'")
            
            # Claim detection
            is_claim_result, prob = is_claim(sent)
            print(f"    Claim detection: {is_claim_result} (prob: {prob:.3f})")
            
            if is_claim_result:
                # Claim spans
                spans = extract_claim_spans(sent)
                print(f"    Claim spans: {spans}")
                
                # Atomic claims
                for span in spans:
                    atomic_claims = to_atomic(span.get("text", ""), None) or []
                    print(f"    Atomic claims: {atomic_claims}")
                    
                    for ac in atomic_claims:
                        # Context analysis
                        ctx = analyze_context(ac.get("text", ""), sent)
                        print(f"    Context: {ctx}")
                        
                        # Scoring
                        score = score_claim(ac.get("text", ""))
                        print(f"    Score: {score:.3f}")
            print()
            
    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline_step_by_step()
