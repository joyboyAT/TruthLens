"""
Example usage of TruthLens Evidence schema.

This script demonstrates how to create, manipulate, and serialize Evidence objects
with various types of data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from schemas.evidence import (
    Evidence, 
    EvidenceScores, 
    SourceType, 
    SupportLabel
)
from datetime import datetime, timezone
import json


def create_sample_evidence():
    """Create sample evidence objects for demonstration."""
    
    # Sample 1: News Article
    news_evidence = Evidence(
        id="ev_news_001",
        claim_id="claim_climate_001",
        source_type=SourceType.NEWS_ARTICLE,
        url="https://www.reuters.com/climate-change-study",
        domain="reuters.com",
        title="New Study Shows Accelerated Climate Change Effects",
        published_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        language="en",
        snippet="A comprehensive study published in Nature reveals that climate change effects are accelerating faster than previously predicted...",
        full_text="Full article content would go here with detailed analysis of the climate change study...",
        support_label=SupportLabel.SUPPORTS,
        scores=EvidenceScores(relevance=0.9, freshness=0.8, source=0.95),
        metadata={
            "author": "Jane Smith",
            "section": "Science",
            "word_count": 1200,
            "tags": ["climate", "science", "research"]
        }
    )
    
    # Sample 2: Research Paper
    research_evidence = Evidence(
        id="ev_research_001",
        claim_id="claim_climate_001",
        source_type=SourceType.RESEARCH_PAPER,
        url="https://www.nature.com/articles/climate-study-2024",
        domain="nature.com",
        title="Global Climate Patterns: A Comprehensive Analysis",
        published_at=datetime(2023, 12, 20, 14, 0, 0, tzinfo=timezone.utc),
        language="en",
        snippet="This peer-reviewed research examines global climate patterns over the past century...",
        full_text="Abstract: This study presents a comprehensive analysis of global climate patterns...",
        support_label=SupportLabel.SUPPORTS,
        scores=EvidenceScores(relevance=0.95, freshness=0.7, source=0.98),
        metadata={
            "doi": "10.1038/climate.2024.001",
            "authors": ["Dr. John Doe", "Dr. Sarah Johnson"],
            "journal": "Nature",
            "volume": "615",
            "pages": "45-52",
            "peer_reviewed": True
        }
    )
    
    # Sample 3: Social Media Post (refuting evidence)
    social_evidence = Evidence(
        id="ev_social_001",
        claim_id="claim_climate_001",
        source_type=SourceType.SOCIAL_MEDIA,
        url="https://twitter.com/user/status/123456789",
        domain="twitter.com",
        title="Twitter post about climate change",
        published_at=datetime(2024, 1, 18, 16, 45, 0, tzinfo=timezone.utc),
        language="en",
        snippet="Climate change is a hoax! The data doesn't support these claims...",
        full_text="Climate change is a hoax! The data doesn't support these claims. I've looked at the research and it's all fabricated...",
        support_label=SupportLabel.REFUTES,
        scores=EvidenceScores(relevance=0.6, freshness=0.9, source=0.2),
        metadata={
            "platform": "Twitter",
            "user_followers": 5000,
            "verified": False,
            "engagement": {"likes": 150, "retweets": 25, "replies": 10}
        }
    )
    
    return [news_evidence, research_evidence, social_evidence]


def demonstrate_evidence_operations():
    """Demonstrate various operations on Evidence objects."""
    
    print("=== TruthLens Evidence Schema Demo ===\n")
    
    # Create sample evidence
    evidence_list = create_sample_evidence()
    
    # Demonstrate serialization
    print("1. Evidence Serialization Examples:")
    print("-" * 50)
    
    for i, evidence in enumerate(evidence_list, 1):
        print(f"\nEvidence {i}: {evidence.source_type.value.upper()}")
        print(f"Title: {evidence.title}")
        print(f"Support: {evidence.support_label.value}")
        print(f"Final Score: {evidence.scores.final:.3f}")
        
        # Convert to JSON
        json_str = evidence.to_json()
        print(f"JSON Length: {len(json_str)} characters")
        
        # Demonstrate round-trip serialization
        reconstructed = Evidence.from_json(json_str)
        print(f"Reconstruction successful: {reconstructed.id == evidence.id}")
    
    # Demonstrate chunk management
    print("\n\n2. Chunk Management Demo:")
    print("-" * 50)
    
    evidence = evidence_list[0]  # Use news evidence
    print(f"Original chunk count: {len(evidence.chunk_ids)}")
    
    # Add chunks
    evidence.add_chunk("chunk_001")
    evidence.add_chunk("chunk_002")
    evidence.add_chunk("chunk_003")
    print(f"After adding chunks: {len(evidence.chunk_ids)}")
    print(f"Chunk IDs: {evidence.chunk_ids}")
    
    # Remove a chunk
    evidence.remove_chunk("chunk_002")
    print(f"After removing chunk_002: {evidence.chunk_ids}")
    
    # Demonstrate score updates
    print("\n\n3. Score Update Demo:")
    print("-" * 50)
    
    print(f"Initial scores: {evidence.scores.to_dict()}")
    
    # Update scores
    evidence.update_scores(relevance=0.95, freshness=0.85)
    print(f"After updating relevance and freshness: {evidence.scores.to_dict()}")
    
    evidence.update_scores(source=0.92)
    print(f"After updating source score: {evidence.scores.to_dict()}")
    
    # Demonstrate metadata handling
    print("\n\n4. Metadata Handling Demo:")
    print("-" * 50)
    
    # Add metadata
    evidence.metadata["fact_check_status"] = "verified"
    evidence.metadata["last_updated"] = "2024-01-20"
    
    print(f"Metadata keys: {list(evidence.metadata.keys())}")
    print(f"Fact check status: {evidence.metadata.get('fact_check_status')}")
    
    # Demonstrate enum usage
    print("\n\n5. Enum Values Demo:")
    print("-" * 50)
    
    print("Available Source Types:")
    for source_type in SourceType:
        print(f"  - {source_type.value}")
    
    print("\nAvailable Support Labels:")
    for label in SupportLabel:
        print(f"  - {label.value}")
    
    # Demonstrate text hashing
    print("\n\n6. Text Hashing Demo:")
    print("-" * 50)
    
    print(f"Full text hash: {evidence.full_text_hash}")
    print(f"Hash length: {len(evidence.full_text_hash)} characters")
    
    # Test hash consistency
    same_evidence = Evidence(
        id="ev_test_hash",
        claim_id="claim_test",
        source_type=SourceType.NEWS_ARTICLE,
        url="https://example.com",
        domain="example.com",
        title="Test",
        full_text=evidence.full_text
    )
    
    print(f"Hash consistency: {evidence.full_text_hash == same_evidence.full_text_hash}")


def demonstrate_database_schemas():
    """Demonstrate database schema usage."""
    
    print("\n\n7. Database Schema Demo:")
    print("-" * 50)
    
    try:
        from schemas.database import (
            get_bigquery_schema,
            get_postgres_schema,
            create_all_postgres_schemas
        )
        
        # Show BigQuery schema for evidence_raw
        bigquery_schema = get_bigquery_schema("evidence_raw")
        print(f"BigQuery evidence_raw table has {len(bigquery_schema['fields'])} fields")
        
        # Show PostgreSQL schema creation
        postgres_schemas = create_all_postgres_schemas()
        print(f"PostgreSQL schema creation: {len(postgres_schemas)} tables")
        
        print("Database schemas loaded successfully!")
        
    except ImportError as e:
        print(f"Database schemas not available: {e}")


if __name__ == "__main__":
    try:
        demonstrate_evidence_operations()
        demonstrate_database_schemas()
        
        print("\n\n=== Demo Completed Successfully! ===")
        print("The Evidence schema is working correctly with:")
        print("- Proper instantiation and validation")
        print("- JSON serialization/deserialization")
        print("- Score calculations and updates")
        print("- Chunk management")
        print("- Metadata handling")
        print("- Text hashing")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
