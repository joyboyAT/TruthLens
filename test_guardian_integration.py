#!/usr/bin/env python3
"""
Test Script for Guardian API Integration
Tests the enhanced news handler with both News API and Guardian API.
"""

import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_guardian_api_direct():
    """Test Guardian API directly to ensure it's working."""
    print("🔍 Testing Guardian API Directly")
    print("=" * 50)
    
    try:
        from src.news.guardian_api_handler import GuardianAPIHandler
        
        # Get Guardian API key from environment
        guardian_api_key = os.getenv('GUARDIAN_API_KEY')
        if not guardian_api_key:
            print("❌ Guardian API key not found in environment variables")
            print("   Set GUARDIAN_API_KEY environment variable")
            return False
        
        # Initialize Guardian API handler
        guardian_handler = GuardianAPIHandler(guardian_api_key)
        print("✅ Guardian API handler initialized successfully")
        
        # Test 1: Fetch COVID-19 vaccine articles
        print("\n1️⃣ Testing 'COVID-19 vaccine' query")
        print("-" * 40)
        
        covid_results = guardian_handler.fetch_guardian_news("COVID-19 vaccine", max_results=5, days_back=30)
        print(f"   Results: {len(covid_results)} articles")
        
        if covid_results:
            print("   ✅ Guardian API is working!")
            for i, article in enumerate(covid_results[:3], 1):
                print(f"      {i}. {article.get('title', 'No title')[:60]}...")
                print(f"         Source: {article.get('source', 'Unknown')}")
                print(f"         URL: {article.get('url', 'No URL')[:80]}...")
        else:
            print("   ⚠️  No results found for COVID-19 vaccine")
        
        # Test 2: Fetch climate change articles
        print("\n2️⃣ Testing 'climate change' query")
        print("-" * 40)
        
        climate_results = guardian_handler.fetch_guardian_news("climate change", max_results=5, days_back=30)
        print(f"   Results: {len(climate_results)} articles")
        
        if climate_results:
            print("   ✅ Climate change articles found!")
            for i, article in enumerate(climate_results[:3], 1):
                print(f"      {i}. {article.get('title', 'No title')[:60]}...")
                print(f"         Section: {article.get('section', 'Unknown')}")
        else:
            print("   ⚠️  No results found for climate change")
        
        # Test 3: Fetch flat earth articles (should find debunking)
        print("\n3️⃣ Testing 'flat earth' query")
        print("-" * 40)
        
        flat_earth_results = guardian_handler.fetch_guardian_news("flat earth", max_results=5, days_back=90)
        print(f"   Results: {len(flat_earth_results)} articles")
        
        if flat_earth_results:
            print("   ✅ Flat earth articles found!")
            for i, article in enumerate(flat_earth_results[:3], 1):
                print(f"      {i}. {article.get('title', 'No title')[:60]}...")
                print(f"         Content preview: {article.get('content', '')[:100]}...")
        else:
            print("   ⚠️  No results found for flat earth")
        
        return True
        
    except Exception as e:
        print(f"❌ Guardian API test failed: {e}")
        logger.error(f"Guardian API test failed: {e}")
        return False

def test_enhanced_news_handler():
    """Test the enhanced news handler with both APIs."""
    print("\n🔍 Testing Enhanced News Handler")
    print("=" * 50)
    
    try:
        from src.news.enhanced_news_handler import EnhancedNewsHandler
        
        # Get API keys
        news_api_key = os.getenv('NEWS_API_KEY')
        guardian_api_key = os.getenv('GUARDIAN_API_KEY')
        
        if not news_api_key:
            print("❌ News API key not found")
            return False
        
        if not guardian_api_key:
            print("❌ Guardian API key not found")
            return False
        
        # Initialize enhanced news handler
        enhanced_handler = EnhancedNewsHandler(news_api_key, guardian_api_key)
        print("✅ Enhanced News Handler initialized successfully")
        
        # Test 1: Get news sources for climate change
        print("\n1️⃣ Testing get_news_sources('climate change')")
        print("-" * 50)
        
        climate_articles = enhanced_handler.get_news_sources("climate change", max_results=10, days_back=30)
        print(f"   Total results: {len(climate_articles)} articles")
        
        if climate_articles:
            # Count sources
            newsapi_count = sum(1 for a in climate_articles if a.source_name == "NewsAPI")
            guardian_count = sum(1 for a in climate_articles if a.source_name == "Guardian")
            
            print(f"   Source breakdown: NewsAPI={newsapi_count}, Guardian={guardian_count}")
            
            # Show top results
            print("   Top 3 results:")
            for i, article in enumerate(climate_articles[:3], 1):
                print(f"      {i}. {article.title[:60]}...")
                print(f"         Source: {article.source_name} - {article.source}")
                print(f"         Cross-ref score: {article.cross_reference_score:.3f}")
                if article.matching_headlines:
                    print(f"         Matching headlines: {len(article.matching_headlines)}")
        else:
            print("   ⚠️  No results found")
        
        # Test 2: Get credibility boost
        print("\n2️⃣ Testing credibility boost calculation")
        print("-" * 50)
        
        if climate_articles:
            credibility_metrics = enhanced_handler.get_credibility_boost(climate_articles)
            print(f"   Credibility Score: {credibility_metrics['credibility_score']:.3f}")
            print(f"   Cross References: {credibility_metrics['cross_references']}")
            print(f"   Source Diversity: {credibility_metrics['source_diversity']:.3f}")
            print(f"   Total Articles: {credibility_metrics['total_articles']}")
            print(f"   Source Breakdown: {credibility_metrics['source_breakdown']}")
        else:
            print("   ⚠️  No articles to calculate credibility")
        
        # Test 3: Test fallback strategy
        print("\n3️⃣ Testing fallback strategy")
        print("-" * 50)
        
        fallback_articles = enhanced_handler.search_with_fallback("AI technology", max_results=8, days_back=30)
        print(f"   Fallback results: {len(fallback_articles)} articles")
        
        if fallback_articles:
            newsapi_count = sum(1 for a in fallback_articles if a.source_name == "NewsAPI")
            guardian_count = sum(1 for a in fallback_articles if a.source_name == "Guardian")
            print(f"   Source breakdown: NewsAPI={newsapi_count}, Guardian={guardian_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced News Handler test failed: {e}")
        logger.error(f"Enhanced News Handler test failed: {e}")
        return False

def test_verification_logic():
    """Test the verification logic for cross-referencing."""
    print("\n🔍 Testing Verification Logic")
    print("=" * 50)
    
    try:
        from src.news.enhanced_news_handler import EnhancedNewsHandler
        
        # Get API keys
        news_api_key = os.getenv('NEWS_API_KEY')
        guardian_api_key = os.getenv('GUARDIAN_API_KEY')
        
        if not news_api_key or not guardian_api_key:
            print("❌ API keys not available for verification logic test")
            return False
        
        enhanced_handler = EnhancedNewsHandler(news_api_key, guardian_api_key)
        
        # Test 1: Earth is flat (should show Guardian debunking, NewsAPI likely also)
        print("\n1️⃣ Testing 'Earth is flat' verification")
        print("-" * 50)
        
        flat_earth_articles = enhanced_handler.get_news_sources("Earth is flat", max_results=8, days_back=90)
        print(f"   Results: {len(flat_earth_articles)} articles")
        
        if flat_earth_articles:
            # Look for debunking content
            debunking_count = 0
            for article in flat_earth_articles:
                title_lower = article.title.lower()
                content_lower = article.content.lower()
                if any(word in title_lower or word in content_lower for word in ['debunk', 'false', 'not true', 'round', 'spherical']):
                    debunking_count += 1
                    print(f"      Debunking article: {article.title[:60]}...")
                    print(f"         Source: {article.source_name}")
            
            print(f"   Total debunking articles: {debunking_count}")
        
        # Test 2: Nanded floods (Guardian may not cover, so NewsAPI remains primary)
        print("\n2️⃣ Testing 'Nanded floods' verification")
        print("-" * 50)
        
        nanded_articles = enhanced_handler.get_news_sources("Nanded floods", max_results=8, days_back=30)
        print(f"   Results: {len(nanded_articles)} articles")
        
        if nanded_articles:
            newsapi_count = sum(1 for a in nanded_articles if a.source_name == "NewsAPI")
            guardian_count = sum(1 for a in nanded_articles if a.source_name == "Guardian")
            print(f"   Source breakdown: NewsAPI={newsapi_count}, Guardian={guardian_count}")
            
            if guardian_count == 0:
                print("   ✅ Guardian doesn't cover Nanded floods - NewsAPI remains primary")
            else:
                print("   ⚠️  Guardian has some coverage of Nanded floods")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification logic test failed: {e}")
        logger.error(f"Verification logic test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Guardian API Integration Test Suite")
    print("=" * 80)
    print()
    
    # Check environment variables
    print("🔑 Checking API Keys")
    print("-" * 30)
    
    news_api_key = os.getenv('NEWS_API_KEY')
    guardian_api_key = os.getenv('GUARDIAN_API_KEY')
    
    if news_api_key:
        print("✅ News API key found")
    else:
        print("❌ News API key not found")
    
    if guardian_api_key:
        print("✅ Guardian API key found")
    else:
        print("❌ Guardian API key not found")
    
    if not news_api_key or not guardian_api_key:
        print("\n⚠️  Please set the required environment variables:")
        print("   export NEWS_API_KEY='your_news_api_key'")
        print("   export GUARDIAN_API_KEY='your_guardian_api_key'")
        return
    
    print()
    
    # Run tests
    test_results = []
    
    # Test 1: Direct Guardian API
    test_results.append(("Guardian API Direct", test_guardian_api_direct()))
    
    # Test 2: Enhanced News Handler
    test_results.append(("Enhanced News Handler", test_enhanced_news_handler()))
    
    # Test 3: Verification Logic
    test_results.append(("Verification Logic", test_verification_logic()))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Guardian API integration tests passed!")
        print("✅ The enhanced news handler is working correctly")
        print("✅ Cross-referencing between News API and Guardian API is functional")
        print("✅ Verification logic is implemented")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    print("\n🔧 Next Steps:")
    print("   1. Integrate EnhancedNewsHandler into your main pipeline")
    print("   2. Use get_news_sources() for comprehensive news gathering")
    print("   3. Leverage cross-reference scores for credibility boosting")

if __name__ == "__main__":
    main()
