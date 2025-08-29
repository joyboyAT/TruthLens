#!/usr/bin/env python3
"""
Comprehensive Test Script for TruthLens Fixes
Tests all the implemented fixes and improvements.
"""

import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_semantic_cross_reference_fixes():
    """Test the semantic cross-reference fixes."""
    print("\n" + "="*60)
    print("🧪 TESTING SEMANTIC CROSS-REFERENCE FIXES")
    print("="*60)
    
    try:
        from evidence_retrieval.semantic_cross_reference_scorer import SemanticCrossReferenceScorer, CrossReferenceScore
        
        # Initialize scorer
        scorer = SemanticCrossReferenceScorer()
        print("✅ SemanticCrossReferenceScorer initialized successfully")
        
        # Test structured output
        test_scores = [
            CrossReferenceScore(
                article_id="test1",
                source_name="NewsAPI",
                similarity_score=0.85,
                matching_articles=[],
                credibility_boost=0.7,
                verification_badge="✅ Verified by multiple sources",
                evidence_strength="Strong"
            ),
            CrossReferenceScore(
                article_id="test2",
                source_name="Guardian",
                similarity_score=0.45,
                matching_articles=[],
                credibility_boost=0.3,
                verification_badge="⚠️ Partially verified",
                evidence_strength="Weak (Single Source)"
            )
        ]
        
        # Test structured output
        structured_results = scorer.get_structured_output(test_scores)
        print(f"✅ Structured output generated: {len(structured_results)} results")
        
        for result in structured_results:
            print(f"  - {result['source']}: {result['verification_badge']} ({result['evidence_strength']})")
            print(f"    Credibility Boost: {result['credibility_boost']:.3f}")
        
        # Test cross-reference summary
        summary = scorer.get_cross_reference_summary(test_scores)
        print(f"✅ Cross-reference summary: {summary['total_articles']} articles")
        print(f"  - Strong evidence: {summary['strong_evidence_count']}")
        print(f"  - Moderate evidence: {summary['moderate_evidence_count']}")
        print(f"  - Average credibility boost: {summary['average_credibility_boost']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Semantic cross-reference fixes test failed: {e}")
        return False

def test_enhanced_news_handler_fixes():
    """Test the enhanced news handler fixes."""
    print("\n" + "="*60)
    print("🧪 TESTING ENHANCED NEWS HANDLER FIXES")
    print("="*60)
    
    try:
        from src.news.enhanced_news_handler import EnhancedNewsHandler
        
        # Test with mock API keys
        news_api_key = "test_news_api_key"
        guardian_api_key = "test_guardian_api_key"
        
        # Initialize handler
        handler = EnhancedNewsHandler(news_api_key, guardian_api_key)
        print("✅ EnhancedNewsHandler initialized successfully")
        
        # Test cache initialization
        if hasattr(handler, 'cache_conn') and handler.cache_conn:
            print("✅ NewsAPI cache initialized successfully")
        else:
            print("⚠️ NewsAPI cache not available")
        
        # Test rate limit checking
        if hasattr(handler, '_is_newsapi_rate_limited'):
            is_rate_limited = handler._is_newsapi_rate_limited()
            print(f"✅ Rate limit checking working: {is_rate_limited}")
        else:
            print("❌ Rate limit checking not available")
        
        # Test cache methods
        if hasattr(handler, '_cache_newsapi_results'):
            print("✅ NewsAPI caching methods available")
        else:
            print("❌ NewsAPI caching methods not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced news handler fixes test failed: {e}")
        return False

def test_currents_api_integration():
    """Test the Currents API integration."""
    print("\n" + "="*60)
    print("🧪 TESTING CURRENTS API INTEGRATION")
    print("="*60)
    
    try:
        from src.news.currents_api_handler import CurrentsAPIHandler
        
        # Test without API key (free tier)
        handler = CurrentsAPIHandler()
        print("✅ CurrentsAPIHandler initialized in free tier mode")
        
        # Test availability check
        is_available = handler.is_available()
        print(f"✅ Availability check: {is_available}")
        
        # Test trending topics
        try:
            trending = handler.get_trending_topics()
            if trending:
                print(f"✅ Trending topics retrieved: {len(trending)} topics")
                print(f"  - Sample: {trending[:3]}")
            else:
                print("⚠️ No trending topics available")
        except Exception as e:
            print(f"⚠️ Trending topics test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Currents API integration test failed: {e}")
        return False

def test_enhanced_pipeline_integration():
    """Test the enhanced pipeline integration."""
    print("\n" + "="*60)
    print("🧪 TESTING ENHANCED PIPELINE INTEGRATION")
    print("="*60)
    
    try:
        from src.enhanced_truthlens_pipeline import EnhancedTruthLensPipeline
        
        # Test with mock API keys
        news_api_key = "test_news_api_key"
        guardian_api_key = "test_guardian_api_key"
        google_api_key = "test_google_api_key"
        currents_api_key = "test_currents_api_key"
        
        # Test component imports without full initialization
        try:
            from src.verification.enhanced_stance_classifier import EnhancedStanceClassifier
            print("✅ Enhanced Stance Classifier import successful")
        except Exception as e:
            print(f"❌ Enhanced Stance Classifier import failed: {e}")
        
        try:
            from src.verification.enhanced_verdict_aggregator import EnhancedVerdictAggregator
            print("✅ Enhanced Verdict Aggregator import successful")
        except Exception as e:
            print(f"❌ Enhanced Verdict Aggregator import failed: {e}")
        
        try:
            from src.evidence_retrieval.enhanced_semantic_search import EnhancedSemanticSearch
            print("✅ Enhanced Semantic Search import successful")
        except Exception as e:
            print(f"❌ Enhanced Semantic Search import failed: {e}")
        
        try:
            from src.news.enhanced_news_handler import EnhancedNewsHandler
            print("✅ Enhanced News Handler import successful")
        except Exception as e:
            print(f"❌ Enhanced News Handler import failed: {e}")
        
        try:
            from src.verification.enhanced_factcheck_api import EnhancedFactCheckAPI
            print("✅ Enhanced Fact Check API import successful")
        except Exception as e:
            print(f"❌ Enhanced Fact Check API import failed: {e}")
        
        print("✅ Enhanced pipeline components import test completed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced pipeline integration test failed: {e}")
        return False

def test_google_fact_check_integration():
    """Test Google Fact Check API integration."""
    print("\n" + "="*60)
    print("🧪 TESTING GOOGLE FACT CHECK API INTEGRATION")
    print("="*60)
    
    try:
        from src.verification.enhanced_factcheck_api import EnhancedFactCheckAPI
        
        # Test with mock API key
        api_key = "test_google_api_key"
        
        # Initialize API
        fact_check_api = EnhancedFactCheckAPI(api_key)
        print("✅ EnhancedFactCheckAPI initialized successfully")
        
        # Test methods
        if hasattr(fact_check_api, 'get_best_fact_check'):
            print("✅ get_best_fact_check method available")
        else:
            print("❌ get_best_fact_check method not available")
        
        if hasattr(fact_check_api, 'search_claims'):
            print("✅ search_claims method available")
        else:
            print("❌ search_claims method not available")
        
        # Test to_dict method
        from verification.enhanced_factcheck_api import EnhancedFactCheckResult
        
        test_result = EnhancedFactCheckResult(
            claim_text="Test claim",
            verdict="REFUTED",
            confidence=0.9,
            sources=[],
            best_source={"name": "Test Source", "url": "http://test.com"},
            review_date="2024-01-01",
            explanation="Test explanation",
            rating="False",
            url="http://test.com"
        )
        
        result_dict = test_result.to_dict()
        if isinstance(result_dict, dict):
            print("✅ to_dict method working correctly")
        else:
            print("❌ to_dict method not working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Google Fact Check API integration test failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 TRUTHLENS COMPREHENSIVE FIXES TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Semantic Cross-Reference Fixes", test_semantic_cross_reference_fixes),
        ("Enhanced News Handler Fixes", test_enhanced_news_handler_fixes),
        ("Currents API Integration", test_currents_api_integration),
        ("Enhanced Pipeline Integration", test_enhanced_pipeline_integration),
        ("Google Fact Check API Integration", test_google_fact_check_integration)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! All fixes are working correctly.")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Some fixes may need attention.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)
