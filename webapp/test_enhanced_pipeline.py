#!/usr/bin/env python3
"""
Test script for the Enhanced TruthLens Pipeline
Demonstrates dynamic evidence retrieval and NLI verification
"""

import sys
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

def test_enhanced_pipeline():
    """Test the enhanced pipeline with various input texts."""
    
    try:
        from src.enhanced_pipeline import process_text_enhanced
        
        # Test cases
        test_cases = [
            "COVID-19 vaccines cause autism in children.",
            "The Earth is flat.",
            "Water boils at 100 degrees Celsius at sea level.",
            "5G towers emit dangerous radiation that causes cancer.",
            "The Great Wall of China is visible from space.",
            "I love pizza and ice cream."
        ]
        
        print("🚀 Testing Enhanced TruthLens Pipeline")
        print("=" * 60)
        print("This pipeline uses dynamic evidence retrieval and NLI verification")
        print("It works with ANY input text, not just pre-detected claims!")
        print()
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"🧪 Test Case {i}: '{test_text}'")
            print("-" * 50)
            
            try:
                # Process with enhanced pipeline
                result = process_text_enhanced(test_text, max_evidence=3)
                
                # Display results
                print(f"📊 Verdict: {result['summary']['verdict']}")
                print(f"🎯 Confidence: {result['summary']['confidence']:.2f}")
                print(f"📝 Reasoning: {result['summary']['reasoning']}")
                print(f"📚 Evidence Sources: {result['evidence_retrieval']['sources_checked']}")
                print(f"⏱️ Processing Time: {result['summary']['processing_time']:.2f}s")
                
                # Show evidence details
                print(f"📖 Evidence Retrieved: {len(result['evidence_retrieval']['evidence'])} items")
                for j, evidence in enumerate(result['evidence_retrieval']['evidence'], 1):
                    print(f"  {j}. {evidence['source']}: {evidence['title'][:50]}...")
                
                # Show verification details
                print(f"🔍 Verification Results: {len(result['verification']['verification_results'])}")
                for j, verification in enumerate(result['verification']['verification_results'], 1):
                    print(f"  {j}. {verification['source']}: {verification['stance']} (confidence: {verification['confidence_score']:.2f})")
                
            except Exception as e:
                print(f"❌ Error processing test case {i}: {e}")
            
            print("\n" + "="*60 + "\n")
            
    except Exception as e:
        print(f"❌ Error testing enhanced pipeline: {e}")
        import traceback
        traceback.print_exc()

def test_individual_components():
    """Test individual components of the enhanced pipeline."""
    
    try:
        from src.evidence_retrieval.dynamic_evidence_retriever import get_dynamic_evidence
        from src.verification.enhanced_verifier import verify_text_with_evidence
        
        test_text = "COVID-19 vaccines cause autism in children."
        
        print("🔧 Testing Individual Components")
        print("=" * 40)
        
        # Test dynamic evidence retrieval
        print("📚 Testing Dynamic Evidence Retrieval...")
        evidence = get_dynamic_evidence(test_text, max_evidence=3)
        print(f"✅ Retrieved {len(evidence)} evidence items")
        for i, ev in enumerate(evidence, 1):
            print(f"  {i}. {ev.source}: {ev.title[:50]}...")
        
        # Test enhanced verification
        print("\n🔍 Testing Enhanced Verification...")
        evidence_dicts = [
            {
                'content': ev.content,
                'source': ev.source
            } for ev in evidence
        ]
        verification_result = verify_text_with_evidence(test_text, evidence_dicts)
        print(f"✅ Verdict: {verification_result['verdict']}")
        print(f"✅ Confidence: {verification_result['confidence']:.2f}")
        print(f"✅ Reasoning: {verification_result['reasoning']}")
        
    except Exception as e:
        print(f"❌ Error testing individual components: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 Enhanced TruthLens Pipeline Test Suite")
    print("=" * 60)
    
    # Test individual components first
    test_individual_components()
    print("\n" + "="*60 + "\n")
    
    # Test full pipeline
    test_enhanced_pipeline()
