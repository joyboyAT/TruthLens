#!/usr/bin/env python3
"""
Claim Analyzer Test
Tests the claim analyzer with keyword extraction, news search, and confidence badges.
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

# Import claim analyzer
try:
    from src.claim_analyzer import ClaimAnalyzer, ClaimAnalysisResult, ConfidenceBadge
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Claim analyzer not available - {e}")
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
    print(f"CLAIM ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    print(f"Original Claim: {result.original_claim}")
    print(f"Processing Time: {result.processing_time:.2f} seconds")
    print(f"Sources Checked: {', '.join(result.sources_checked)}")
    
    print(f"\nKeywords Extracted ({len(result.extracted_keywords)}):")
    for i, keyword in enumerate(result.extracted_keywords, 1):
        print(f"  {i}. {keyword}")
    
    print(f"\nNews Articles Found ({len(result.news_articles)}):")
    for i, article in enumerate(result.news_articles[:5], 1):  # Show top 5
        print(f"  {i}. {article['title']}")
        print(f"     Source: {article['source']}")
        print(f"     Relevance: {article['relevance_score']:.2f}")
        print(f"     Keyword: {article['search_keyword']}")
        print()
    
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

def get_test_claims() -> List[str]:
    """Get test claims for analysis."""
    return [
        "Nanded floods caused massive destruction in Maharashtra",
        "COVID-19 vaccines cause autism in children",
        "5G technology causes coronavirus",
        "The Earth is flat and NASA has been hiding this fact",
        "Climate change is a hoax created by scientists",
        "Maharashtra heavy rains led to 8 deaths in Nanded",
        "Artificial intelligence will replace all human jobs",
        "Renewable energy is more expensive than fossil fuels"
    ]

def main():
    """Main function to test the claim analyzer."""
    try:
        # API keys
        news_api_key = "9c7e59e19af34bb8adb97d0a8bec458d"
        google_api_key = "AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc"
        
        if not ANALYZER_AVAILABLE:
            print("❌ Claim analyzer not available")
            return 1
        
        # Initialize claim analyzer
        print("🚀 Initializing Claim Analyzer...")
        analyzer = ClaimAnalyzer(news_api_key, google_api_key)
        print("✅ Claim Analyzer initialized successfully")
        
        # Get test claims
        test_claims = get_test_claims()
        
        print(f"\n🔍 Testing {len(test_claims)} claims...")
        print("="*80)
        
        results = []
        
        for i, claim in enumerate(test_claims, 1):
            print(f"\n--- Test {i}/{len(test_claims)} ---")
            print(f"Claim: {claim}")
            
            try:
                # Analyze the claim
                result = analyzer.analyze_claim(claim)
                results.append(result)
                
                # Print results
                print_analysis_result(result)
                
                # Rate limiting between claims
                if i < len(test_claims):
                    print("⏳ Waiting 2 seconds before next analysis...")
                    time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error analyzing claim: {e}")
                continue
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        successful_analyses = len(results)
        print(f"Successful Analyses: {successful_analyses}/{len(test_claims)}")
        
        if successful_analyses > 0:
            # Confidence badge distribution
            badge_counts = {}
            for result in results:
                if result.confidence_badge:
                    level = result.confidence_badge.level
                    badge_counts[level] = badge_counts.get(level, 0) + 1
            
            print(f"\nConfidence Badge Distribution:")
            for level, count in badge_counts.items():
                emoji = "🟢" if level == "Likely True" else "🟡" if level == "Unclear" else "🔴"
                print(f"  {emoji} {level}: {count}")
            
            # Average processing time
            avg_time = sum(r.processing_time for r in results) / len(results)
            print(f"\nAverage Processing Time: {avg_time:.2f} seconds per claim")
            
            # Sources used
            all_sources = set()
            for result in results:
                all_sources.update(result.sources_checked)
            print(f"\nSources Used: {', '.join(all_sources)}")
        
        # Save detailed results
        output_file = "claim_analysis_results.json"
        
        # Convert results to JSON-serializable format
        json_results = []
        for result in results:
            json_result = {
                "original_claim": result.original_claim,
                "extracted_keywords": result.extracted_keywords,
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
            json_results.append(json_result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        if successful_analyses == len(test_claims):
            print("\n✅ All claim analyses completed successfully!")
            return 0
        else:
            print(f"\n⚠️ {len(test_claims) - successful_analyses} analyses failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
