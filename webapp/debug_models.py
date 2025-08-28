#!/usr/bin/env python3
"""
Debug script to test TruthLens models directly
"""
import sys
from pathlib import Path

# Add the parent directory to Python path to import TruthLens modules
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

print("🔍 Testing TruthLens Models...")
print(f"📁 Current directory: {current_dir}")
print(f"🔧 Parent directory: {parent_dir}")

try:
    print("\n1. Testing imports...")
    from extractor.pipeline import process_text
    from extractor.claim_detector import is_claim
    from extractor.claim_extractor import extract_claim_spans
    from extractor.atomicizer import to_atomic
    from extractor.context import analyze_context
    from extractor.ranker import score_claim
    print("✅ All imports successful")
    
    print("\n2. Testing claim detection...")
    test_text = "COVID-19 vaccines cause autism in children"
    print(f"Test text: {test_text}")
    
    # Test individual components
    print("\n3. Testing is_claim...")
    label, prob = is_claim(test_text)
    print(f"is_claim result: label={label}, probability={prob}")
    
    print("\n4. Testing extract_claim_spans...")
    spans = extract_claim_spans(test_text)
    print(f"extract_claim_spans result: {spans}")
    
    print("\n5. Testing to_atomic...")
    if spans:
        atomic_claims = to_atomic(spans[0].get("text", ""), None)
        print(f"to_atomic result: {atomic_claims}")
    else:
        print("No spans to atomicize")
    
    print("\n6. Testing analyze_context...")
    ctx = analyze_context(test_text, test_text)
    print(f"analyze_context result: {ctx}")
    
    print("\n7. Testing score_claim...")
    score = score_claim(test_text)
    print(f"score_claim result: {score}")
    
    print("\n8. Testing full pipeline...")
    atomic_claims = process_text(test_text)
    print(f"process_text result: {atomic_claims}")
    
    if atomic_claims:
        print("\n✅ Pipeline working! Claims extracted:")
        for i, claim in enumerate(atomic_claims):
            print(f"  Claim {i+1}:")
            print(f"    ID: {claim.get('id')}")
            print(f"    Text: {claim.get('text')}")
            print(f"    Subject: {claim.get('subject')}")
            print(f"    Predicate: {claim.get('predicate')}")
            print(f"    Object: {claim.get('object')}")
            print(f"    Checkworthiness: {claim.get('checkworthiness')}")
            print(f"    Context: {claim.get('context')}")
    else:
        print("\n❌ Pipeline returned empty claims")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
