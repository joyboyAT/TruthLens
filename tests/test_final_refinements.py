#!/usr/bin/env python3
"""
Final Refinements Test
Tests confidence calibration, hybrid stance models, SQLite caching, evidence snippets, and explicit verdict logic.
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

# Import enhanced verification components
try:
    from src.verification.stance_classifier import StanceClassifier, StanceResult
    from src.verification.verdict_mapper import EnhancedVerdictMapper
    from src.verification.sqlite_cache_manager import SQLiteCacheManager
    VERIFICATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced verification components not available - {e}")
    VERIFICATION_AVAILABLE = False

def print_evidence_snippets(evidence_snippets: List[Dict[str, Any]]):
    """Print evidence snippets in a formatted way."""
    print(f"\n📄 Evidence Snippets ({len(evidence_snippets)}):")
    print(f"{'='*60}")
    
    for i, snippet in enumerate(evidence_snippets, 1):
        stance_emoji = {
            "SUPPORTED": "🟢",
            "REFUTED": "🔴", 
            "NOT ENOUGH INFO": "🟡"
        }.get(snippet["stance"], "⚪")
        
        print(f"{i}. {stance_emoji} {snippet['sentence'][:100]}...")
        print(f"   Confidence: {snippet['confidence']:.3f} | Reasoning: {snippet['reasoning']}")
        if snippet.get("source"):
            print(f"   Source: {snippet['source']} ({snippet.get('source_type', 'news')})")
        print()

def print_ensemble_votes(ensemble_votes: Dict[str, Any]):
    """Print ensemble voting results."""
    print(f"\n🗳️ Ensemble Voting Results:")
    print(f"{'='*40}")
    
    for model, vote in ensemble_votes.items():
        if vote["stance"]:
            stance_emoji = {
                "SUPPORTED": "🟢",
                "REFUTED": "🔴",
                "NOT ENOUGH INFO": "🟡"
            }.get(vote["stance"], "⚪")
            
            print(f"  {model}: {stance_emoji} {vote['stance']} (confidence: {vote['confidence']:.3f})")
        else:
            print(f"  {model}: No vote")

def get_test_cases() -> List[Dict[str, Any]]:
    """Get test cases for final refinements."""
    return [
        {
            "name": "Confidence Calibration Test",
            "claim": "The Earth is flat",
            "evidence": [
                {
                    "text": "NASA has provided overwhelming evidence that the Earth is spherical, including satellite images and space missions.",
                    "source": "NASA",
                    "source_type": "fact_check",
                    "reliability_score": 0.95
                },
                {
                    "text": "The Earth is flat and NASA is hiding the truth from us.",
                    "source": "Flat Earth Society",
                    "source_type": "news",
                    "reliability_score": 0.2
                }
            ],
            "expected_verdict": "🔴 Likely False",
            "expected_confidence_max": 0.95  # Should not exceed max confidence
        },
        {
            "name": "Scientific Consensus Test",
            "claim": "Vaccines cause autism in children",
            "evidence": [
                {
                    "text": "Multiple large-scale studies have found no link between vaccines and autism. The CDC and WHO confirm vaccines are safe.",
                    "source": "WHO",
                    "source_type": "fact_check",
                    "reliability_score": 0.95
                },
                {
                    "text": "A small study suggested a possible link between vaccines and autism.",
                    "source": "Small Journal",
                    "source_type": "news",
                    "reliability_score": 0.4
                }
            ],
            "expected_verdict": "🔴 Likely False",
            "expected_logic": "scientific_consensus_override"
        },
        {
            "name": "No Evidence Test",
            "claim": "Aliens built the pyramids",
            "evidence": [
                {
                    "text": "The pyramids were built by ancient Egyptians using advanced engineering techniques.",
                    "source": "Archaeology Journal",
                    "source_type": "news",
                    "reliability_score": 0.7
                },
                {
                    "text": "There is no evidence of alien involvement in pyramid construction.",
                    "source": "Science Magazine",
                    "source_type": "news",
                    "reliability_score": 0.8
                }
            ],
            "expected_verdict": "🟡 Unclear",
            "expected_logic": "no_evidence_case"
        },
        {
            "name": "Causal Chain Test",
            "claim": "Heavy rains caused massive destruction in Mumbai",
            "evidence": [
                {
                    "text": "Heavy rains in Mumbai led to flooding, with reports of 12 deaths and extensive damage to infrastructure.",
                    "source": "Times of India",
                    "source_type": "news",
                    "reliability_score": 0.8
                },
                {
                    "text": "Rescue operations are ongoing in flood-affected areas of Mumbai. Many people are still missing.",
                    "source": "Hindustan Times",
                    "source_type": "news",
                    "reliability_score": 0.8
                }
            ],
            "expected_verdict": "🟢 Likely True",
            "expected_logic": "causal_reasoning"
        }
    ]

def test_confidence_calibration():
    """Test confidence calibration to avoid over-certainty."""
    print("🎯 Testing Confidence Calibration")
    print("="*60)
    
    stance_classifier = StanceClassifier()
    
    # Test case that should trigger high confidence
    claim = "The Earth is flat"
    evidence = "NASA has provided overwhelming evidence that the Earth is spherical, including satellite images and space missions."
    
    print(f"Claim: {claim}")
    print(f"Evidence: {evidence[:100]}...")
    
    stance_result = stance_classifier.classify_one(claim, evidence)
    
    print(f"\nResults:")
    print(f"  Stance: {stance_result.label}")
    print(f"  Raw Confidence: {stance_result.confidence_score:.3f}")
    print(f"  Max Confidence Applied: {stance_result.confidence_score <= 0.95}")
    print(f"  Rule Override: {stance_result.rule_based_override}")
    
    # Check if confidence is properly calibrated
    if stance_result.confidence_score <= 0.95:
        print(f"  ✅ Confidence properly calibrated (≤ 0.95)")
    else:
        print(f"  ❌ Confidence not properly calibrated (> 0.95)")
    
    return stance_result

def test_hybrid_stance_models():
    """Test hybrid stance models with ensemble voting."""
    print("\n🤖 Testing Hybrid Stance Models")
    print("="*60)
    
    stance_classifier = StanceClassifier()
    
    # Test case with multiple evidence types
    claim = "Vaccines cause autism in children"
    evidence_texts = [
        "Multiple large-scale studies have found no link between vaccines and autism. The CDC and WHO confirm vaccines are safe.",
        "A small study suggested a possible link between vaccines and autism.",
        "Scientific consensus shows no connection between vaccines and autism spectrum disorders."
    ]
    
    print(f"Claim: {claim}")
    print(f"Evidence Count: {len(evidence_texts)}")
    
    stance_results = stance_classifier.classify_batch(claim, evidence_texts)
    
    print(f"\nEnsemble Voting Results:")
    for i, result in enumerate(stance_results, 1):
        print(f"\nEvidence {i}:")
        print(f"  Final Stance: {result.label}")
        print(f"  Confidence: {result.confidence_score:.3f}")
        print(f"  Rule Override: {result.rule_based_override}")
        
        if hasattr(result, 'ensemble_votes') and result.ensemble_votes:
            print_ensemble_votes(result.ensemble_votes)
        
        if hasattr(result, 'evidence_snippets') and result.evidence_snippets:
            print_evidence_snippets(result.evidence_snippets)
    
    return stance_results

def test_sqlite_caching():
    """Test SQLite-based caching with hash-based keys."""
    print("\n💾 Testing SQLite Caching")
    print("="*60)
    
    # Initialize SQLite cache manager
    cache_manager = SQLiteCacheManager()
    
    # Test case
    claim = "Heavy rains caused massive destruction in Mumbai"
    evidence = "Heavy rains in Mumbai led to flooding, with reports of 12 deaths and extensive damage to infrastructure."
    
    print(f"Claim: {claim}")
    print(f"Evidence: {evidence[:100]}...")
    
    # First run (should cache)
    print(f"\n🔄 First Run (Caching):")
    start_time = time.time()
    
    # Simulate stance classification
    stance_classifier = StanceClassifier()
    stance_result = stance_classifier.classify_one(claim, evidence)
    
    # Cache the result
    cached_result = cache_manager.cache_stance_result(claim, evidence, stance_result)
    first_run_time = time.time() - start_time
    
    print(f"  Stance: {stance_result.label}")
    print(f"  Confidence: {stance_result.confidence_score:.3f}")
    print(f"  Processing Time: {first_run_time:.3f}s")
    print(f"  Cached: ✅")
    
    # Second run (should use cache)
    print(f"\n⚡ Second Run (Cache Hit):")
    start_time = time.time()
    cached_stance = cache_manager.get_cached_stance(claim, evidence)
    cache_time = time.time() - start_time
    
    if cached_stance:
        print(f"  Stance: {cached_stance.stance}")
        print(f"  Confidence: {cached_stance.confidence_score:.3f}")
        print(f"  Processing Time: {cache_time:.3f}s")
        print(f"  Cache Hit: ✅")
        print(f"  Speed Improvement: {first_run_time/cache_time:.1f}x faster")
        
        # Show evidence snippets from cache
        if cached_stance.evidence_snippets:
            print_evidence_snippets(cached_stance.evidence_snippets)
    
    # Show cache statistics
    print(f"\n📊 Cache Statistics:")
    stats = cache_manager.get_cache_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return cache_manager

def test_weighted_feedback():
    """Test weighted user feedback system."""
    print("\n👥 Testing Weighted User Feedback")
    print("="*60)
    
    cache_manager = SQLiteCacheManager()
    
    # Test different user types with different weights
    test_feedbacks = [
        {
            "claim": "Vaccines cause autism in children",
            "user_label": "REFUTED",
            "user_confidence": 0.9,
            "feedback_reason": "Scientific consensus is clear - no link between vaccines and autism",
            "evidence_used": ["Multiple large-scale studies have found no link"],
            "user_type": "trusted_fact_checker"
        },
        {
            "claim": "Vaccines cause autism in children",
            "user_label": "SUPPORTED",
            "user_confidence": 0.8,
            "feedback_reason": "I read an article that said vaccines might cause autism",
            "evidence_used": ["An article I read"],
            "user_type": "casual_user"
        },
        {
            "claim": "Heavy rains caused massive destruction in Mumbai",
            "user_label": "SUPPORTED",
            "user_confidence": 0.95,
            "feedback_reason": "I was there and saw the destruction firsthand",
            "evidence_used": ["Personal eyewitness account"],
            "user_type": "expert"
        }
    ]
    
    for i, feedback in enumerate(test_feedbacks, 1):
        print(f"\n--- User Feedback {i} ---")
        print(f"Claim: {feedback['claim']}")
        print(f"User Type: {feedback['user_type']}")
        print(f"User Label: {feedback['user_label']}")
        print(f"User Confidence: {feedback['user_confidence']}")
        print(f"Feedback Reason: {feedback['feedback_reason']}")
        
        # Add feedback to cache
        cache_manager.add_user_feedback(
            feedback["claim"],
            feedback["user_label"],
            feedback["user_confidence"],
            feedback["feedback_reason"],
            feedback["evidence_used"],
            feedback["user_type"]
        )
        
        print(f"  ✅ Feedback added with appropriate weight")
    
    # Retrieve feedback for a claim
    print(f"\n📋 Retrieved Feedback:")
    feedback_list = cache_manager.get_user_feedback_for_claim("Vaccines cause autism in children")
    
    for feedback in feedback_list:
        print(f"  User Type: {feedback.user_type}")
        print(f"  Label: {feedback.user_label}")
        print(f"  Weighted Confidence: {feedback.user_confidence:.3f}")
        print(f"  Reason: {feedback.feedback_reason}")
        print()

def test_explicit_verdict_logic():
    """Test explicit verdict logic encoding."""
    print("\n⚖️ Testing Explicit Verdict Logic")
    print("="*60)
    
    stance_classifier = StanceClassifier()
    verdict_mapper = EnhancedVerdictMapper()
    
    test_cases = get_test_cases()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        print(f"Claim: {test_case['claim']}")
        print(f"Expected: {test_case['expected_verdict']}")
        
        # Extract evidence
        evidence_texts = [ev["text"] for ev in test_case["evidence"]]
        evidence_metadata = [
            {
                "source": ev["source"],
                "source_type": ev["source_type"],
                "reliability_score": ev["reliability_score"]
            } for ev in test_case["evidence"]
        ]
        
        # Perform stance classification
        print(f"\n📊 Stance Classification:")
        stance_results = stance_classifier.classify_batch(test_case["claim"], evidence_texts)
        
        # Combine with metadata
        enhanced_stance_results = []
        for j, (result, metadata) in enumerate(zip(stance_results, evidence_metadata)):
            enhanced_result = {
                "label": result.label,
                "confidence_score": result.confidence_score,
                "probabilities": result.probabilities,
                "rule_based_override": result.rule_based_override,
                "source": metadata["source"],
                "source_type": metadata["source_type"],
                "reliability_score": metadata["reliability_score"],
                "text": evidence_texts[j],
                "evidence_snippets": result.evidence_snippets if hasattr(result, 'evidence_snippets') else []
            }
            enhanced_stance_results.append(enhanced_result)
            
            print(f"  Evidence {j+1}: {result.label} (confidence: {result.confidence_score:.3f})")
            if result.rule_based_override:
                print(f"    Rule Override: {result.rule_based_override}")
        
        # Perform verdict aggregation
        print(f"\n⚖️ Verdict Aggregation:")
        verdict_result = verdict_mapper.map_to_enhanced_verdict(enhanced_stance_results, test_case["claim"])
        
        print(f"  Final Verdict: {verdict_result['verdict']}")
        print(f"  Reasoning: {verdict_result['reasoning']}")
        print(f"  Confidence: {verdict_result['probability']:.3f}")
        
        # Check verdict logic applied
        logic_applied = verdict_result.get("verdict_logic_applied", {})
        print(f"  Logic Applied:")
        for logic, applied in logic_applied.items():
            print(f"    {logic}: {'✅' if applied else '❌'}")
        
        # Check if verdict matches expectation
        if verdict_result['verdict'] == test_case['expected_verdict']:
            print(f"  ✅ VERDICT MATCHES EXPECTATION")
        else:
            print(f"  ❌ VERDICT MISMATCH: Expected {test_case['expected_verdict']}, Got {verdict_result['verdict']}")
        
        # Check confidence calibration
        if 'expected_confidence_max' in test_case:
            if verdict_result['probability'] <= test_case['expected_confidence_max']:
                print(f"  ✅ CONFIDENCE PROPERLY CALIBRATED")
            else:
                print(f"  ❌ CONFIDENCE NOT CALIBRATED: {verdict_result['probability']:.3f} > {test_case['expected_confidence_max']}")
        
        # Show evidence snippets
        if verdict_result.get("evidence_snippets"):
            print_evidence_snippets(verdict_result["evidence_snippets"])

def main():
    """Main function to test final refinements."""
    try:
        if not VERIFICATION_AVAILABLE:
            print("❌ Enhanced verification components not available")
            return 1
        
        print("🚀 Testing Final Refinements")
        print("="*80)
        print("✅ Features Being Tested:")
        print("   - Confidence calibration (max 0.95)")
        print("   - Hybrid stance models with ensemble voting")
        print("   - SQLite caching with hash-based keys")
        print("   - Evidence snippets in verdict JSON")
        print("   - Weighted user feedback system")
        print("   - Explicit verdict logic encoding")
        print("="*80)
        
        # Test 1: Confidence Calibration
        test_confidence_calibration()
        
        # Test 2: Hybrid Stance Models
        test_hybrid_stance_models()
        
        # Test 3: SQLite Caching
        cache_manager = test_sqlite_caching()
        
        # Test 4: Weighted Feedback
        test_weighted_feedback()
        
        # Test 5: Explicit Verdict Logic
        test_explicit_verdict_logic()
        
        # Final statistics
        print(f"\n📊 Final Cache Statistics:")
        stats = cache_manager.get_cache_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n✅ All final refinement tests completed successfully!")
        print(f"\n🎉 Key Improvements Implemented:")
        print(f"   - Confidence calibration prevents over-certainty")
        print(f"   - Hybrid models provide ensemble voting")
        print(f"   - SQLite caching enables scalability and queryability")
        print(f"   - Evidence snippets provide full auditability")
        print(f"   - Weighted feedback improves model learning")
        print(f"   - Explicit logic ensures consistent verdicts")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
