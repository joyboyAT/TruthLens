#!/usr/bin/env python3
"""
Interactive Claim Analyzer Test
Allows users to input any claim and get confidence analysis with badges.
"""

import sys
import os
import time
from pathlib import Path

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
    for i, article in enumerate(result.news_articles[:3], 1):  # Show top 3
        print(f"  {i}. {article['title']}")
        print(f"     Source: {article['source']}")
        print(f"     Relevance: {article['relevance_score']:.2f}")
        print()
    
    if result.google_factcheck_result:
        print(f"\nGoogle Fact Check Result:")
        google = result.google_factcheck_result
        print(f"  Verdict: {google['verdict']}")
        print(f"  Confidence: {google['confidence']:.1%}")
        print(f"  Publisher: {google['publisher']}")
        print(f"  Rating: {google['rating']}")
    
    # Print confidence badge
    if result.confidence_badge:
        print_confidence_badge(result.confidence_badge)
    
    print(f"\nAnalysis Summary:")
    print(result.analysis_summary)
    
    print(f"{'='*80}")

def get_user_claim() -> str:
    """Get claim input from user."""
    print("\n🚀 TruthLens Interactive Claim Analyzer")
    print("="*60)
    print("Enter a claim to analyze (or press Enter for examples):")
    
    user_input = input("\nClaim: ").strip()
    
    if not user_input:
        # Example claims
        examples = [
            "Nanded floods caused massive destruction in Maharashtra",
            "COVID-19 vaccines cause autism in children",
            "5G technology causes coronavirus",
            "The Earth is flat and NASA has been hiding this fact",
            "Climate change is a hoax created by scientists",
            "Maharashtra heavy rains led to 8 deaths in Nanded",
            "Artificial intelligence will replace all human jobs",
            "Renewable energy is more expensive than fossil fuels"
        ]
        
        print("\nExample claims:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        
        choice = input("\nSelect an example (1-8) or press Enter for example 1: ").strip()
        try:
            choice_idx = int(choice) - 1 if choice else 0
            if 0 <= choice_idx < len(examples):
                return examples[choice_idx]
        except ValueError:
            pass
        
        return examples[0]
    
    return user_input

def main():
    """Main function for interactive claim analysis."""
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
        
        while True:
            # Get user claim
            claim = get_user_claim()
            
            if claim.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            print(f"\n🔍 Analyzing claim: {claim}")
            print("Please wait...")
            
            try:
                # Analyze the claim
                result = analyzer.analyze_claim(claim)
                
                # Print results
                print_analysis_result(result)
                
                # Ask if user wants to analyze another claim
                another = input("\nAnalyze another claim? (y/n): ").strip().lower()
                if another not in ['y', 'yes']:
                    print("👋 Goodbye!")
                    break
                
            except Exception as e:
                print(f"❌ Error analyzing claim: {e}")
                continue
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
