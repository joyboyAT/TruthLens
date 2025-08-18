#!/usr/bin/env python3
"""
Demonstration script for TruthLens Phase 3 Fact-Check Connectors.

This script demonstrates how to use the fact-check connectors to search for claims
across multiple fact-checking sources including PIB, PolitiFact, BOOM Live, and AltNews.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add the parent directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from connectors.factcheck import (
    PIBFactCheckConnector,
    PolitiFactConnector,
    BOOMLiveConnector,
    AltNewsConnector,
    FactCheckAggregator
)


async def demo_individual_connectors():
    """Demonstrate individual connector functionality."""
    print("🔍 Testing Individual Fact-Check Connectors")
    print("=" * 60)
    
    # Test PIB Fact Check
    print("\n1. PIB Fact Check Connector")
    print("-" * 30)
    pib = PIBFactCheckConnector()
    print(f"   Name: {pib.name}")
    print(f"   Base URL: {pib.base_url}")
    print(f"   RSS URL: {pib.rss_url}")
    print(f"   Rate Limit: {pib.rate_limit} requests/minute")
    
    # Test PolitiFact
    print("\n2. PolitiFact Connector")
    print("-" * 30)
    politifact = PolitiFactConnector()
    print(f"   Name: {politifact.name}")
    print(f"   Base URL: {politifact.base_url}")
    print(f"   RSS URL: {politifact.rss_url}")
    print(f"   Rate Limit: {politifact.rate_limit} requests/minute")
    
    # Test BOOM Live
    print("\n3. BOOM Live Connector")
    print("-" * 30)
    boom = BOOMLiveConnector()
    print(f"   Name: {boom.name}")
    print(f"   Base URL: {boom.base_url}")
    print(f"   RSS URL: {boom.rss_url}")
    print(f"   Rate Limit: {boom.rate_limit} requests/minute")
    
    # Test AltNews
    print("\n4. AltNews Connector")
    print("-" * 30)
    altnews = AltNewsConnector()
    print(f"   Name: {altnews.name}")
    print(f"   Base URL: {altnews.base_url}")
    print(f"   RSS URL: {altnews.rss_url}")
    print(f"   Rate Limit: {altnews.rate_limit} requests/minute")


async def demo_claim_search():
    """Demonstrate claim search functionality."""
    print("\n\n🔍 Testing Claim Search Functionality")
    print("=" * 60)
    
    # Sample claims for testing
    test_claims = [
        "Climate change is caused by human activities",
        "Vaccines cause autism",
        "The Earth is flat",
        "जलवायु परिवर्तन मानवीय गतिविधियों के कारण होता है",  # Hindi
        "காலநிலை மாற்றம் மனித செயல்பாடுகளால் ஏற்படுகிறது"   # Tamil
    ]
    
    print(f"Testing with {len(test_claims)} sample claims:")
    for i, claim in enumerate(test_claims, 1):
        print(f"   {i}. {claim}")
    
    print("\nNote: This is a demonstration. Actual RSS fetching and article scraping")
    print("would require network access and may be rate-limited by the sources.")
    
    # Create aggregator
    aggregator = FactCheckAggregator()
    print(f"\n✅ Fact-Check Aggregator initialized with {len(aggregator.connectors)} sources:")
    for source_name in aggregator.connectors.keys():
        print(f"   - {source_name}")


async def demo_evidence_processing():
    """Demonstrate evidence processing and normalization."""
    print("\n\n📝 Testing Evidence Processing")
    print("=" * 60)
    
    # Create a test connector
    from connectors.factcheck import FactCheckConnector
    
    test_connector = FactCheckConnector("https://example.com", "Test Connector")
    
    # Sample article data
    sample_article = {
        'title': 'Fact Check: Climate Change Claims Debunked',
        'url': 'https://example.com/fact-check-climate-change',
        'published': datetime.now(),
        'summary': 'A comprehensive fact check of common climate change misconceptions.',
        'full_text': 'This is the full text content of the fact check article...',
        'author': 'Fact Checker Team',
        'domain': 'example.com'
    }
    
    print("Sample Article Data:")
    print(f"   Title: {sample_article['title']}")
    print(f"   URL: {sample_article['url']}")
    print(f"   Author: {sample_article['author']}")
    print(f"   Summary: {sample_article['summary']}")
    
    # Normalize to evidence
    evidence = test_connector.normalize_to_evidence(sample_article)
    
    print("\nNormalized Evidence:")
    print(f"   ID: {evidence.id}")
    print(f"   Source Type: {evidence.source_type}")
    print(f"   Language: {evidence.language}")
    print(f"   Domain: {evidence.domain}")
    print(f"   Title: {evidence.title}")
    print(f"   Snippet: {evidence.snippet}")
    print(f"   Published: {evidence.published_at}")
    print(f"   Metadata Categories: {evidence.metadata.categories}")
    print(f"   Verification Status: {evidence.metadata.verification_status}")


async def demo_fuzzy_matching():
    """Demonstrate fuzzy matching capabilities."""
    print("\n\n🎯 Testing Fuzzy Matching")
    print("=" * 60)
    
    from fuzzywuzzy import fuzz
    
    # Sample claims and fact-check titles
    test_pairs = [
        ("Climate change is caused by humans", "Fact Check: Climate Change and Human Activities"),
        ("Vaccines cause autism", "Debunked: Vaccines and Autism Link"),
        ("Earth is flat", "Fact Check: Is the Earth Really Flat?"),
        ("जलवायु परिवर्तन", "जलवायु परिवर्तन पर तथ्य जांच"),  # Hindi
        ("காலநிலை மாற்றம்", "காலநிலை மாற்றம் பற்றிய உண்மை சோதனை")  # Tamil
    ]
    
    print("Fuzzy Matching Examples:")
    print("Claim -> Fact-Check Title -> Similarity Score")
    print("-" * 70)
    
    for claim, fact_check_title in test_pairs:
        # Calculate different similarity scores
        ratio_score = fuzz.ratio(claim.lower(), fact_check_title.lower())
        partial_ratio = fuzz.partial_ratio(claim.lower(), fact_check_title.lower())
        token_sort_ratio = fuzz.token_sort_ratio(claim.lower(), fact_check_title.lower())
        
        print(f"Claim: {claim}")
        print(f"Title: {fact_check_title}")
        print(f"Scores - Ratio: {ratio_score}, Partial: {partial_ratio}, Token Sort: {token_sort_ratio}")
        print("-" * 70)


def demo_configuration():
    """Demonstrate configuration and settings."""
    print("\n\n⚙️ Configuration and Settings")
    print("=" * 60)
    
    print("Fact-Check Connector Configuration:")
    print()
    
    # Display connector configurations
    connectors_config = {
        "PIB Fact Check": {
            "base_url": "https://pib.gov.in",
            "rss_url": "https://pib.gov.in/rss/fact-check.xml",
            "rate_limit": 5,
            "description": "Government of India Press Information Bureau"
        },
        "PolitiFact": {
            "base_url": "https://www.politifact.com",
            "rss_url": "https://www.politifact.com/feeds/fact-check.xml",
            "rate_limit": 15,
            "description": "Pulitzer Prize-winning fact-checking website"
        },
        "BOOM Live": {
            "base_url": "https://www.boomlive.in",
            "rss_url": "https://www.boomlive.in/feed",
            "rate_limit": 10,
            "description": "Indian fact-checking and news verification platform"
        },
        "AltNews": {
            "base_url": "https://www.altnews.in",
            "rss_url": "https://www.altnews.in/feed",
            "rate_limit": 8,
            "description": "Indian fact-checking website focusing on misinformation"
        }
    }
    
    for name, config in connectors_config.items():
        print(f"{name}:")
        print(f"   Base URL: {config['base_url']}")
        print(f"   RSS URL: {config['rss_url']}")
        print(f"   Rate Limit: {config['rate_limit']} requests/minute")
        print(f"   Description: {config['description']}")
        print()


async def main():
    """Main demonstration function."""
    print("🚀 TRUTHLENS PHASE 3 - FACT-CHECK CONNECTOR DEMONSTRATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Run demonstrations
        await demo_individual_connectors()
        await demo_claim_search()
        await demo_evidence_processing()
        await demo_fuzzy_matching()
        demo_configuration()
        
        print("\n" + "=" * 80)
        print("✅ FACT-CHECK CONNECTOR DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\n📋 Next Steps:")
        print("1. Install dependencies: pip install -r connectors/requirements_factcheck.txt")
        print("2. Run tests: python -m pytest tests/test_factcheck.py -v")
        print("3. Test with real data: Modify demo script to enable actual RSS fetching")
        print("4. Add more fact-check sources as needed")
        print("5. Integrate with the main evidence retrieval pipeline")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This might be due to missing dependencies or import issues.")
        print("Please ensure all required packages are installed.")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
