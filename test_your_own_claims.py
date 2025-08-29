#!/usr/bin/env python3
"""
Interactive Test Script for Your Own Claims
Test your own claims with the enhanced TruthLens pipeline.
"""

import os
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_your_claim(claim: str):
    """Test a specific claim with the enhanced pipeline."""
    print(f"\n🔍 Testing Your Claim: '{claim}'")
    print("=" * 70)
    
    try:
        # Import enhanced components
        from src.verification.enhanced_stance_classifier import EnhancedStanceClassifier
        from src.verification.enhanced_verdict_aggregator import EnhancedVerdictAggregator
        from src.verification.enhanced_factcheck_api import EnhancedFactCheckAPI
        from src.evidence_retrieval.enhanced_semantic_search import EnhancedSemanticSearch
        
        # Initialize components
        stance_classifier = EnhancedStanceClassifier()
        verdict_aggregator = EnhancedVerdictAggregator()
        semantic_search = EnhancedSemanticSearch()
        
        google_api_key = os.getenv('GOOGLE_FACTCHECK_API_KEY')
        if google_api_key:
            fact_check_api = EnhancedFactCheckAPI(google_api_key)
            print("✅ Fact-Check API initialized with Google API key")
        else:
            fact_check_api = None
            print("⚠️  Google Fact Check API key not available")
        
        # Step 1: Check fact-check sources (REAL API)
        fact_check_result = None
        if fact_check_api:
            try:
                print("🔍 Checking fact-check sources (REAL API)...")
                fact_check_result = fact_check_api.get_best_fact_check(claim)
                if fact_check_result:
                    print(f"   ✅ Found: {fact_check_result.verdict} (confidence: {fact_check_result.confidence:.3f})")
                    print(f"      Source: {fact_check_result.best_source['name']}")
                    print(f"      Explanation: {fact_check_result.explanation[:100]}...")
                else:
                    print("   ⚠️  No fact-check result found")
            except Exception as e:
                print(f"   ❌ Fact-check error: {e}")
        
        # Step 2: Create mock articles for stance detection
        mock_articles = create_mock_articles_for_claim(claim)
        print(f"📰 Created {len(mock_articles)} mock articles for stance detection")
        
        # Step 3: Perform semantic search and ranking
        try:
            print("🔍 Performing semantic search and ranking...")
            search_results = semantic_search.search_and_rank_articles(claim, mock_articles)
            print(f"   ✅ Semantic search completed: {len(search_results)} results")
            if search_results:
                print(f"      Top result score: {search_results[0].semantic_score:.3f}")
        except Exception as e:
            print(f"   ❌ Semantic search error: {e}")
        
        # Step 4: Perform stance detection
        stance_results = []
        for j, article in enumerate(mock_articles):
            try:
                stance_result = stance_classifier.classify_stance(claim, article)
                stance_results.append({
                    'stance': stance_result.stance,
                    'confidence': stance_result.confidence,
                    'reasoning': stance_result.reasoning,
                    'rule_based_override': stance_result.rule_based_override,
                    'evidence_sentences': stance_result.evidence_sentences
                })
                
                print(f"   Article {j+1}: {stance_result.stance} (confidence: {stance_result.confidence:.3f})")
                if stance_result.rule_based_override:
                    print(f"      Rule-based override: {stance_result.rule_based_override}")
                if stance_result.evidence_sentences:
                    print(f"      Evidence: {stance_result.evidence_sentences[0][:80]}...")
                
            except Exception as e:
                print(f"   ❌ Stance detection error: {e}")
        
        # Step 5: Aggregate verdict
        if stance_results:
            try:
                # Convert fact-check result to dict format if available
                fact_check_dict = None
                if fact_check_result:
                    fact_check_dict = fact_check_result.to_dict()
                
                verdict = verdict_aggregator.aggregate_verdict(
                    claim, stance_results, fact_check_dict, len(mock_articles)
                )
                
                print(f"\n🎯 FINAL VERDICT: {verdict.verdict}")
                print(f"   Confidence: {verdict.confidence:.1%}")
                print(f"   Reasoning: {verdict.reasoning}")
                
                # Display stance distribution
                if verdict.stance_distribution:
                    print(f"   Stance Distribution: {dict(verdict.stance_distribution)}")
                
                # Display evidence summary
                print(f"   Evidence Summary: {verdict.evidence_summary}")
                
                # Display fact-check override if any
                if verdict.fact_check_override:
                    print(f"   Fact-Check Override: {verdict.fact_check_override}")
                
            except Exception as e:
                print(f"   ❌ Verdict aggregation error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing claim: {e}")
        logger.error(f"Error testing claim '{claim}': {e}")
        return False

def create_mock_articles_for_claim(claim: str) -> List[Dict[str, Any]]:
    """Create realistic mock articles for testing stance detection."""
    # Create articles that might support or contradict the claim
    articles = [
        {
            'title': f'News about: {claim}',
            'description': f'Recent developments related to {claim}',
            'content': f'This article discusses the claim that {claim}. Various experts have weighed in on this topic with different perspectives. The article presents multiple viewpoints and evidence related to this claim.'
        },
        {
            'title': f'Analysis of: {claim}',
            'description': f'Comprehensive analysis of the claim with expert opinions',
            'content': f'A detailed analysis of the claim "{claim}" reveals multiple aspects that need consideration. Research and evidence are presented from various sources. Experts provide their insights on this topic.'
        },
        {
            'title': f'Expert opinion on: {claim}',
            'description': f'What leading experts say about this claim',
            'content': f'Leading experts in the field have provided their opinions on the claim that {claim}. Their insights offer valuable perspective on this issue. The article examines the evidence and expert consensus.'
        },
        {
            'title': f'Research findings on: {claim}',
            'description': f'Scientific research and studies related to this claim',
            'content': f'Recent research has examined the claim that {claim}. Multiple studies have been conducted to investigate this topic. The findings provide important insights into the validity of this claim.'
        }
    ]
    
    return articles

def main():
    """Main interactive function."""
    print("🚀 Enhanced TruthLens Pipeline - Test Your Own Claims")
    print("=" * 70)
    print()
    print("🎯 This script will test your claims using:")
    print("   ✅ Real Google Fact Check API integration")
    print("   ✅ Enhanced stance detection with 0.6 thresholds")
    print("   ✅ Improved verdict aggregation with 40% thresholds")
    print("   ✅ Semantic search and ranking")
    print("   ✅ Rule-based overrides for scientific consensus")
    print()
    
    # Test some example claims first
    example_claims = [
        "Nanded floods caused massive destruction",
        "COVID-19 vaccines cause autism in children"
    ]
    
    print("🧪 Testing Example Claims First:")
    for claim in example_claims:
        test_your_claim(claim)
        print("\n" + "-" * 70)
    
    # Now let user input their own claims
    print("\n🎯 Now Test Your Own Claims!")
    print("Enter claims to test (type 'quit' to exit):")
    
    while True:
        try:
            user_claim = input("\n🔍 Enter your claim: ").strip()
            
            if user_claim.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_claim:
                print("⚠️  Please enter a valid claim")
                continue
            
            # Test the user's claim
            test_your_claim(user_claim)
            
            # Ask if they want to continue
            continue_test = input("\n🔍 Test another claim? (y/n): ").strip().lower()
            if continue_test not in ['y', 'yes', '']:
                print("👋 Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
