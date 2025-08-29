#!/usr/bin/env python3
"""
Single Claim Demo
Demonstrates the enhanced claim analyzer with a single claim and fallback to mock data.
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

# Import enhanced claim analyzer
try:
    from src.claim_analyzer import EnhancedClaimAnalyzer, ClaimAnalysisResult, ConfidenceBadge
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced claim analyzer not available - {e}")
    ANALYZER_AVAILABLE = False

def print_confidence_badge(badge: ConfidenceBadge):
    """Print a formatted confidence badge."""
    print(f"\n{'='*60}")
    print(f"CONFIDENCE BADGE")
    print(f"{'='*60}")
    print(f"{badge.emoji} {badge.level}")
    print(f"Confidence: {badge.confidence:.1%}")
    print(f"Color: {badge.color.upper()}")
    print(f"Reasoning: {badge.reasoning}")
    print(f"{'='*60}")

def print_analysis_result(result: ClaimAnalysisResult):
    """Print a formatted analysis result."""
    print(f"\n{'='*80}")
    print(f"SINGLE CLAIM ANALYSIS DEMO")
    print(f"{'='*80}")
    
    print(f"Original Claim: {result.original_claim}")
    print(f"Claim Type: {result.claim_type}")
    print(f"Processing Time: {result.processing_time:.2f} seconds")
    print(f"Sources Checked: {', '.join(result.sources_checked)}")
    
    print(f"\nSearch Phrases Extracted ({len(result.extracted_phrases)}):")
    for i, phrase in enumerate(result.extracted_phrases, 1):
        print(f"  {i}. {phrase}")
    
    print(f"\nNews Articles Found ({len(result.news_articles)}):")
    if result.news_articles:
        for i, article in enumerate(result.news_articles[:5], 1):  # Show top 5
            print(f"  {i}. {article['title']}")
            print(f"     Source: {article['source']}")
            print(f"     Relevance: {article['relevance_score']:.2f}")
            print(f"     Search Phrase: {article['search_phrase']}")
            print()
    else:
        print("  ⚠️  No news articles found (API quota may be exhausted)")
    
    if result.google_factcheck_result:
        print(f"\nGoogle Fact Check Result:")
        google = result.google_factcheck_result
        print(f"  Verdict: {google['verdict']}")
        print(f"  Confidence: {google['confidence']:.1%}")
        print(f"  Publisher: {google['publisher']}")
        print(f"  Rating: {google['rating']}")
        print(f"  URL: {google['url']}")
    
    # Print confidence badge
    if result.confidence_badge:
        print_confidence_badge(result.confidence_badge)
    
    print(f"\nAnalysis Summary:")
    print(result.analysis_summary)
    
    print(f"{'='*80}")

def get_mock_news_articles(claim: str) -> List[Dict[str, Any]]:
    """Generate mock news articles for demonstration when API quota is exhausted."""
    claim_lower = claim.lower()
    
    # Mock articles based on claim content
    mock_articles = []
    
    if "nanded" in claim_lower and "flood" in claim_lower:
        mock_articles = [
            {
                "title": "Nanded floods cause massive destruction in Maharashtra",
                "description": "Heavy rains in Nanded district have led to severe flooding, causing widespread damage to property and infrastructure.",
                "url": "https://example.com/nanded-floods",
                "source": "Times of India",
                "published_at": "2024-01-15T10:00:00Z",
                "relevance_score": 0.95,
                "search_phrase": "Nanded floods"
            },
            {
                "title": "Maharashtra floods: Nanded district worst affected",
                "description": "Nanded district in Maharashtra has been severely affected by floods, with reports of massive destruction.",
                "url": "https://example.com/maharashtra-floods",
                "source": "Hindustan Times",
                "published_at": "2024-01-14T15:30:00Z",
                "relevance_score": 0.88,
                "search_phrase": "Nanded floods"
            }
        ]
    elif "covid" in claim_lower and "vaccine" in claim_lower:
        mock_articles = [
            {
                "title": "Fact Check: COVID-19 vaccines do not cause autism",
                "description": "Multiple studies have confirmed that COVID-19 vaccines are safe and do not cause autism in children.",
                "url": "https://example.com/covid-vaccine-fact-check",
                "source": "Reuters Fact Check",
                "published_at": "2024-01-10T12:00:00Z",
                "relevance_score": 0.92,
                "search_phrase": "COVID-19 vaccines"
            }
        ]
    elif "earth" in claim_lower and "flat" in claim_lower:
        mock_articles = [
            {
                "title": "Fact Check: The Earth is not flat, NASA evidence confirms",
                "description": "Scientific evidence from NASA and other space agencies confirms that the Earth is spherical, not flat.",
                "url": "https://example.com/earth-not-flat",
                "source": "NASA",
                "published_at": "2024-01-05T09:00:00Z",
                "relevance_score": 0.98,
                "search_phrase": "Earth flat"
            }
        ]
    else:
        # Generic mock articles
        mock_articles = [
            {
                "title": f"Fact Check: {claim}",
                "description": f"Recent analysis of the claim '{claim}' shows mixed evidence requiring further investigation.",
                "url": "https://example.com/fact-check",
                "source": "Fact Check Organization",
                "published_at": "2024-01-12T11:00:00Z",
                "relevance_score": 0.75,
                "search_phrase": "generic"
            }
        ]
    
    return mock_articles

def analyze_claim_with_fallback(analyzer: EnhancedClaimAnalyzer, claim: str) -> ClaimAnalysisResult:
    """
    Analyze a claim with fallback to mock data if API quota is exhausted.
    """
    try:
        # Try to analyze with real APIs
        result = analyzer.analyze_claim(claim)
        
        # If no news articles found, add mock data for demonstration
        if not result.news_articles:
            print("⚠️  No news articles found from API (quota may be exhausted)")
            print("🔄 Adding mock news articles for demonstration...")
            
            mock_articles = get_mock_news_articles(claim)
            result.news_articles = mock_articles
            
            # Recalculate confidence with mock data
            from src.claim_analyzer import ClaimAnalysisResult, ConfidenceBadge
            
            # Simple confidence calculation based on mock data
            if mock_articles:
                confidence = 0.6  # Moderate confidence for mock data
                reasoning = f"Analysis based on {len(mock_articles)} mock news articles (API quota exhausted)"
            else:
                confidence = 0.0
                reasoning = "No news articles available for analysis"
            
            # Update confidence badge
            if confidence >= 0.7:
                badge = ConfidenceBadge("Likely True", confidence, "green", "🟢", reasoning)
            elif confidence >= 0.4:
                badge = ConfidenceBadge("Unclear", confidence, "yellow", "🟡", reasoning)
            else:
                badge = ConfidenceBadge("Likely False", confidence, "red", "🔴", reasoning)
            
            result.confidence_badge = badge
            
            # Update sources
            result.sources_checked.append("Mock Data (API quota exhausted)")
            
            # Update analysis summary
            result.analysis_summary = f"""
Enhanced Claim Analysis Summary (with Mock Data):
- Original Claim: {claim}
- Claim Type: {result.claim_type}
- Search Phrases: {', '.join(result.extracted_phrases)}
- News Articles Found: {len(result.news_articles)} (mock data)
- Google Fact Check: {'Available' if result.google_factcheck_result else 'Not available'}
- Confidence: {badge.emoji} {badge.level} ({confidence:.1%})
- Reasoning: {reasoning}
- Note: Using mock data due to API quota exhaustion
"""
        
        return result
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        print("🔄 Falling back to mock analysis...")
        
        # Create a basic result with mock data
        mock_articles = get_mock_news_articles(claim)
        confidence = 0.5  # Default confidence for mock data
        reasoning = "Analysis based on mock data due to API errors"
        
        badge = ConfidenceBadge("Unclear", confidence, "yellow", "🟡", reasoning)
        
        return ClaimAnalysisResult(
            original_claim=claim,
            extracted_phrases=["mock phrase"],
            news_articles=mock_articles,
            confidence_badge=badge,
            analysis_summary=f"Mock analysis for: {claim}",
            sources_checked=["Mock Data (API Error)"],
            processing_time=0.1,
            claim_type="general"
        )

def main():
    """Main function to demonstrate single claim analysis."""
    try:
        # API keys
        news_api_key = "9c7e59e19af34bb8adb97d0a8bec458d"
        google_api_key = "AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc"
        
        if not ANALYZER_AVAILABLE:
            print("❌ Enhanced claim analyzer not available")
            return 1
        
        # Initialize enhanced claim analyzer
        print("🚀 Initializing Enhanced Claim Analyzer for Demo...")
        analyzer = EnhancedClaimAnalyzer(news_api_key, google_api_key)
        print("✅ Enhanced Claim Analyzer initialized successfully")
        
        # Test claim
        test_claim = "Nanded floods caused massive destruction in Maharashtra"
        
        print(f"\n🔍 Analyzing single claim: {test_claim}")
        print("="*80)
        print("✅ Features demonstrated:")
        print("   - Phrase extraction from claims")
        print("   - News API integration with rate limiting")
        print("   - Google Fact Check API integration")
        print("   - Confidence badge generation")
        print("   - Fallback to mock data if API quota exhausted")
        print("="*80)
        
        # Analyze the claim with fallback
        result = analyze_claim_with_fallback(analyzer, test_claim)
        
        # Print results
        print_analysis_result(result)
        
        # Save result
        output_file = "single_claim_demo_result.json"
        
        json_result = {
            "original_claim": result.original_claim,
            "claim_type": result.claim_type,
            "extracted_phrases": result.extracted_phrases,
            "news_articles": result.news_articles,
            "google_factcheck_result": result.google_factcheck_result,
            "confidence_badge": {
                "level": result.confidence_badge.level,
                "confidence": result.confidence_badge.confidence,
                "color": result.confidence_badge.color,
                "emoji": result.confidence_badge.emoji,
                "reasoning": result.confidence_badge.reasoning
            } if result.confidence_badge else None,
            "analysis_summary": result.analysis_summary,
            "sources_checked": result.sources_checked,
            "processing_time": result.processing_time
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Demo result saved to: {output_file}")
        print("\n✅ Single claim analysis demo completed successfully!")
        
        return 0
            
    except Exception as e:
        print(f"\n❌ Demo failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
