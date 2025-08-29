#!/usr/bin/env python3
"""
Demo script for Enhanced TruthLens Pipeline
Showcases the improvements with specific examples from the requirements.
"""

import os
import logging
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_enhanced_pipeline():
    """Demo the enhanced TruthLens pipeline with specific examples."""
    
    # Example claims from the requirements
    example_claims = [
        "Nanded floods caused massive destruction",
        "COVID-19 vaccines cause autism in children",
        "AI will replace jobs in the next decade",
        "Climate change is a hoax",
        "The Earth is flat"
    ]
    
    print("🚀 Enhanced TruthLens Pipeline Demo")
    print("=" * 50)
    print()
    
    # Check if we can import the enhanced components
    try:
        from src.enhanced_truthlens_pipeline import create_enhanced_pipeline, EnhancedTruthLensPipeline
        print("✅ Enhanced components imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import enhanced components: {e}")
        print("This demo requires the enhanced components to be properly installed.")
        return
    
    # Check for API keys
    news_api_key = os.getenv('NEWS_API_KEY')
    google_api_key = os.getenv('GOOGLE_FACTCHECK_API_KEY')
    
    if not news_api_key:
        print("⚠️  NEWS_API_KEY not found in environment variables")
        print("   Set NEWS_API_KEY to test with real news data")
        print("   For demo purposes, we'll show the structure without real API calls")
        demo_structure_only()
        return
    
    print(f"✅ News API key found")
    if google_api_key:
        print(f"✅ Google Fact Check API key found")
    else:
        print("⚠️  Google Fact Check API key not found (optional)")
    
    print()
    
    # Create enhanced pipeline
    try:
        pipeline = create_enhanced_pipeline(news_api_key, google_api_key)
        print("✅ Enhanced TruthLens pipeline created successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        return
    
    # Demo with example claims
    for i, claim in enumerate(example_claims, 1):
        print(f"🔍 Example {i}: {claim}")
        print("-" * 40)
        
        try:
            # Analyze the claim
            result = pipeline.analyze_claim(claim, max_articles=10)
            
            # Display results
            display_analysis_result(result)
            
        except Exception as e:
            print(f"❌ Error analyzing claim: {e}")
        
        print()
        print("=" * 50)
        print()

def demo_structure_only():
    """Demo the structure without real API calls."""
    print("📋 Enhanced TruthLens Pipeline Structure")
    print("=" * 40)
    print()
    
    print("🔧 Key Components:")
    print("1. Enhanced Stance Classifier")
    print("   - Threshold tuning: support_prob > 0.6, contradict_prob > 0.6")
    print("   - Rule-based signals for explicit contradictions")
    print("   - Better causal reasoning for destruction/impact claims")
    print("   - Scientific consensus handling")
    print()
    
    print("2. Enhanced Verdict Aggregator")
    print("   - Weighted voting: Support > 40% → Likely True")
    print("   - Contradict > 40% → Likely False")
    print("   - Strong fact-check integration")
    print("   - Better handling of scientific consensus claims")
    print()
    
    print("3. Enhanced Fact-Checking API")
    print("   - Google Fact Check API integration")
    print("   - Snopes, PolitiFact, Science Feedback support")
    print("   - Multiple source aggregation")
    print()
    
    print("4. Enhanced Semantic Search")
    print("   - Sentence transformers for semantic similarity")
    print("   - Article deduplication and clustering")
    print("   - Better relevance ranking")
    print()
    
    print("📊 Example Analysis Structure:")
    example_result = {
        "claim": "COVID-19 vaccines cause autism in children",
        "verdict": "Likely False",
        "confidence": 0.95,
        "reasoning": "Scientific consensus claim: refuted by scientific evidence",
        "stance_distribution": {"support": 0, "contradict": 0, "neutral": 0},
        "stance_percentages": {"support": 0.0, "contradict": 0.0, "neutral": 0.0},
        "fact_check_result": {
            "verdict": "REFUTED",
            "confidence": 0.95,
            "source": "Google Fact Check"
        },
        "rule_based_overrides": ["scientific_consensus"],
        "evidence_summary": "Claim contradicts established scientific consensus"
    }
    
    print(json.dumps(example_result, indent=2))
    print()
    
    print("🎯 Key Improvements Demonstrated:")
    print("✅ Scientific consensus claims default to False")
    print("✅ Rule-based overrides for explicit contradictions")
    print("✅ Better causal reasoning (floods → deaths/rescue = evidence of destruction)")
    print("✅ Improved thresholds (0.6 instead of higher values)")
    print("✅ 40% thresholds for verdict aggregation")
    print("✅ Multiple fact-checking sources")

def display_analysis_result(result):
    """Display the analysis result in a readable format."""
    
    # Verdict with emoji
    verdict_emoji = {
        "Likely True": "🟢",
        "Likely False": "🔴", 
        "Unclear": "🟡",
        "Not Enough Info": "⚪",
        "Error": "❌"
    }
    
    emoji = verdict_emoji.get(result.verdict, "❓")
    print(f"Verdict: {emoji} {result.verdict}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Reasoning: {result.reasoning}")
    
    # Stance distribution
    print(f"\n📊 Stance Distribution:")
    for stance, count in result.stance_distribution.items():
        percentage = result.stance_percentages.get(stance, 0)
        print(f"   {stance.capitalize()}: {count} ({percentage:.1%})")
    
    # Fact-check result
    if result.fact_check_result:
        print(f"\n🔍 Fact-Check Result:")
        fc = result.fact_check_result
        print(f"   Verdict: {fc.get('verdict', 'Unknown')}")
        print(f"   Source: {fc.get('source', 'Unknown')}")
        print(f"   Confidence: {fc.get('confidence', 0):.1%}")
    
    # Rule-based overrides
    if result.rule_based_overrides:
        print(f"\n⚡ Rule-Based Overrides:")
        for override in result.rule_based_overrides:
            print(f"   - {override}")
    
    # Evidence summary
    print(f"\n📝 Evidence Summary:")
    print(f"   {result.evidence_summary}")
    
    # Search quality
    if result.search_summary and 'search_quality' in result.search_summary:
        quality = result.search_summary['search_quality']
        print(f"\n🔍 Search Quality: {quality}")
    
    # Processing time
    print(f"\n⏱️  Processing Time: {result.processing_time:.2f}s")

def run_specific_examples():
    """Run specific examples mentioned in the requirements."""
    print("🎯 Running Specific Examples from Requirements")
    print("=" * 50)
    print()
    
    # Example 1: Nanded floods
    print("🌊 Example 1: 'Nanded floods caused massive destruction'")
    print("Expected: Should detect deaths, rescue ops, army relief as evidence of destruction")
    print("Current issue: 0 supporting, 0 contradicting, 7 neutral = 🔴 Likely False (29%)")
    print("Improvement: Better causal reasoning to map 'floods → deaths/rescue' as support")
    print()
    
    # Example 2: COVID vaccines
    print("💉 Example 2: 'COVID-19 vaccines cause autism in children'")
    print("Expected: Should be 🔴 False due to scientific consensus")
    print("Current issue: 2 supporting, 0 contradicting = 🟡 Unclear (54.9%)")
    print("Improvement: Scientific consensus claims default to False")
    print()
    
    # Example 3: AI jobs
    print("🤖 Example 3: 'AI will replace jobs'")
    print("Expected: Better stance detection with improved thresholds (0.6)")
    print("Current issue: Higher thresholds causing most claims to be unclear")
    print("Improvement: Lower thresholds + fine-tuned NLI model")
    print()

if __name__ == "__main__":
    print("🚀 Starting Enhanced TruthLens Pipeline Demo")
    print()
    
    # Run the demo
    demo_enhanced_pipeline()
    
    print()
    print("🎯 Key Improvements Summary:")
    print("1. ✅ Threshold tuning: support_prob > 0.6, contradict_prob > 0.6")
    print("2. ✅ Enhanced stance detection with rule-based signals")
    print("3. ✅ Improved verdict aggregation: Support > 40% → Likely True, Contradict > 40% → Likely False")
    print("4. ✅ Multiple fact-checking sources (Google, Snopes, PolitiFact, Science Feedback)")
    print("5. ✅ Semantic search with deduplication and clustering")
    print("6. ✅ Better causal reasoning for destruction/impact claims")
    print("7. ✅ Scientific consensus handling (vaccines & autism → False)")
    print()
    print("🔧 To use the enhanced pipeline:")
    print("   from src.enhanced_truthlens_pipeline import create_enhanced_pipeline")
    print("   pipeline = create_enhanced_pipeline(news_api_key, google_api_key)")
    print("   result = pipeline.analyze_claim('Your claim here')")
    print()
    print("📚 For more details, check the enhanced component files:")
    print("   - src/verification/enhanced_stance_classifier.py")
    print("   - src/verification/enhanced_verdict_aggregator.py")
    print("   - src/verification/enhanced_factcheck_api.py")
    print("   - src/evidence_retrieval/enhanced_semantic_search.py")
    print("   - src/enhanced_truthlens_pipeline.py")
