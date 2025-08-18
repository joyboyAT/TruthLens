"""
Tests for TruthLens Evidence schema.

This module tests the Evidence dataclass functionality including:
- Instantiation with various data types
- Serialization to JSON
- Deserialization from JSON
- Score calculations
- Chunk management
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import patch

# Import the schemas to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from schemas.evidence import (
    Evidence, 
    EvidenceScores, 
    SourceType, 
    SupportLabel
)


class TestEvidenceScores:
    """Test EvidenceScores dataclass."""
    
    def test_evidence_scores_creation(self):
        """Test creating EvidenceScores with default values."""
        scores = EvidenceScores()
        assert scores.relevance == 0.0
        assert scores.freshness == 0.0
        assert scores.source == 0.0
        assert scores.final == 0.0
    
    def test_evidence_scores_custom_values(self):
        """Test creating EvidenceScores with custom values."""
        scores = EvidenceScores(
            relevance=0.8,
            freshness=0.6,
            source=0.9,
            final=0.8
        )
        assert scores.relevance == 0.8
        assert scores.freshness == 0.6
        assert scores.source == 0.9
        assert scores.final == 0.8
    
    def test_evidence_scores_to_dict(self):
        """Test converting EvidenceScores to dictionary."""
        scores = EvidenceScores(relevance=0.7, freshness=0.5, source=0.8, final=0.7)
        scores_dict = scores.to_dict()
        
        expected = {
            "relevance": 0.7,
            "freshness": 0.5,
            "source": 0.8,
            "final": 0.7
        }
        assert scores_dict == expected
    
    def test_evidence_scores_from_dict(self):
        """Test creating EvidenceScores from dictionary."""
        data = {
            "relevance": 0.6,
            "freshness": 0.4,
            "source": 0.7,
            "final": 0.6
        }
        scores = EvidenceScores.from_dict(data)
        
        assert scores.relevance == 0.6
        assert scores.freshness == 0.4
        assert scores.source == 0.7
        assert scores.final == 0.6


class TestEvidence:
    """Test Evidence dataclass."""
    
    def test_evidence_creation_minimal(self):
        """Test creating Evidence with minimal required fields."""
        evidence = Evidence(
            id="ev_001",
            claim_id="claim_001",
            source_type=SourceType.NEWS_ARTICLE,
            url="https://example.com/article",
            domain="example.com",
            title="Test Article"
        )
        
        assert evidence.id == "ev_001"
        assert evidence.claim_id == "claim_001"
        assert evidence.source_type == SourceType.NEWS_ARTICLE
        assert evidence.url == "https://example.com/article"
        assert evidence.domain == "example.com"
        assert evidence.title == "Test Article"
        assert evidence.language == "en"
        assert evidence.support_label == SupportLabel.NEUTRAL
        assert evidence.chunk_ids == []
        assert evidence.metadata == {}
    
    def test_evidence_creation_full(self):
        """Test creating Evidence with all fields."""
        published_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        retrieved_at = datetime(2024, 1, 20, 14, 30, 0, tzinfo=timezone.utc)
        
        evidence = Evidence(
            id="ev_002",
            claim_id="claim_002",
            source_type=SourceType.RESEARCH_PAPER,
            url="https://research.org/paper",
            domain="research.org",
            title="Research Study on Climate Change",
            published_at=published_at,
            retrieved_at=retrieved_at,
            language="en",
            snippet="This study examines the impact of climate change...",
            full_text="Full research paper content...",
            support_label=SupportLabel.SUPPORTS,
            scores=EvidenceScores(relevance=0.9, freshness=0.7, source=0.8),
            metadata={"doi": "10.1234/paper", "authors": ["Dr. Smith"]}
        )
        
        assert evidence.published_at == published_at
        assert evidence.retrieved_at == retrieved_at
        assert evidence.snippet == "This study examines the impact of climate change..."
        assert evidence.full_text == "Full research paper content..."
        assert evidence.support_label == SupportLabel.SUPPORTS
        assert evidence.scores.relevance == 0.9
        assert evidence.metadata["doi"] == "10.1234/paper"
    
    def test_evidence_text_hash_auto_computation(self):
        """Test automatic computation of text hash when full_text is provided."""
        evidence = Evidence(
            id="ev_003",
            claim_id="claim_003",
            source_type=SourceType.NEWS_ARTICLE,
            url="https://example.com/article",
            domain="example.com",
            title="Test Article",
            full_text="This is some test content for hashing."
        )
        
        # Hash should be automatically computed
        assert evidence.full_text_hash != ""
        assert len(evidence.full_text_hash) == 16  # xxhash64 produces 16-char hex
    
    def test_evidence_to_dict(self):
        """Test converting Evidence to dictionary."""
        published_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        evidence = Evidence(
            id="ev_004",
            claim_id="claim_004",
            source_type=SourceType.BLOG_POST,
            url="https://blog.com/post",
            domain="blog.com",
            title="Blog Post Title",
            published_at=published_at,
            full_text="Blog content here",
            scores=EvidenceScores(relevance=0.8, freshness=0.6, source=0.7)
        )
        
        evidence_dict = evidence.to_dict()
        
        # Check required fields
        assert evidence_dict["id"] == "ev_004"
        assert evidence_dict["claim_id"] == "claim_004"
        assert evidence_dict["source_type"] == "blog_post"
        assert evidence_dict["url"] == "https://blog.com/post"
        assert evidence_dict["domain"] == "blog.com"
        assert evidence_dict["title"] == "Blog Post Title"
        
        # Check datetime fields
        assert evidence_dict["published_at"] == "2024-01-15T12:00:00+00:00"
        assert "retrieved_at" in evidence_dict  # Should be current time
        
        # Check enum fields
        assert evidence_dict["source_type"] == "blog_post"
        assert evidence_dict["support_label"] == "neutral"
        
        # Check scores
        assert evidence_dict["scores"]["relevance"] == 0.8
        assert evidence_dict["scores"]["freshness"] == 0.6
        assert evidence_dict["scores"]["source"] == 0.7
    
    def test_evidence_to_json(self):
        """Test converting Evidence to JSON string."""
        evidence = Evidence(
            id="ev_005",
            claim_id="claim_005",
            source_type=SourceType.NEWS_ARTICLE,
            url="https://news.com/article",
            domain="news.com",
            title="News Article",
            full_text="News content"
        )
        
        json_str = evidence.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["id"] == "ev_005"
        assert parsed["claim_id"] == "claim_005"
        assert parsed["source_type"] == "news_article"
        assert parsed["url"] == "https://news.com/article"
    
    def test_evidence_from_dict(self):
        """Test creating Evidence from dictionary."""
        data = {
            "id": "ev_006",
            "claim_id": "claim_006",
            "source_type": "government_document",
            "url": "https://gov.org/doc",
            "domain": "gov.org",
            "title": "Government Document",
            "published_at": "2024-01-10T10:00:00+00:00",
            "retrieved_at": "2024-01-20T15:00:00+00:00",
            "language": "en",
            "snippet": "Government document snippet",
            "full_text_hash": "abc123hash",
            "chunk_ids": ["chunk_1", "chunk_2"],
            "support_label": "supports",
            "scores": {
                "relevance": 0.9,
                "freshness": 0.8,
                "source": 0.95,
                "final": 0.9
            },
            "metadata": {"document_type": "report"}
        }
        
        evidence = Evidence.from_dict(data)
        
        assert evidence.id == "ev_006"
        assert evidence.claim_id == "claim_006"
        assert evidence.source_type == SourceType.GOVERNMENT_DOCUMENT
        assert evidence.url == "https://gov.org/doc"
        assert evidence.domain == "gov.org"
        assert evidence.title == "Government Document"
        assert evidence.published_at.year == 2024
        assert evidence.published_at.month == 1
        assert evidence.published_at.day == 10
        assert evidence.language == "en"
        assert evidence.snippet == "Government document snippet"
        assert evidence.full_text_hash == "abc123hash"
        assert evidence.chunk_ids == ["chunk_1", "chunk_2"]
        assert evidence.support_label == SupportLabel.SUPPORTS
        assert evidence.scores.relevance == 0.9
        assert evidence.scores.freshness == 0.8
        assert evidence.scores.source == 0.95
        assert evidence.scores.final == 0.9
        assert evidence.metadata["document_type"] == "report"
    
    def test_evidence_from_json(self):
        """Test creating Evidence from JSON string."""
        json_str = '''
        {
            "id": "ev_007",
            "claim_id": "claim_007",
            "source_type": "social_media",
            "url": "https://twitter.com/post",
            "domain": "twitter.com",
            "title": "Social Media Post",
            "language": "en",
            "support_label": "refutes",
            "scores": {
                "relevance": 0.7,
                "freshness": 0.9,
                "source": 0.3,
                "final": 0.6
            }
        }
        '''
        
        evidence = Evidence.from_json(json_str)
        
        assert evidence.id == "ev_007"
        assert evidence.claim_id == "claim_007"
        assert evidence.source_type == SourceType.SOCIAL_MEDIA
        assert evidence.url == "https://twitter.com/post"
        assert evidence.domain == "twitter.com"
        assert evidence.title == "Social Media Post"
        assert evidence.language == "en"
        assert evidence.support_label == SupportLabel.REFUTES
        assert evidence.scores.relevance == 0.7
        assert evidence.scores.freshness == 0.9
        assert evidence.scores.source == 0.3
        assert evidence.scores.final == 0.6
    
    def test_evidence_chunk_management(self):
        """Test adding and removing chunks."""
        evidence = Evidence(
            id="ev_008",
            claim_id="claim_008",
            source_type=SourceType.NEWS_ARTICLE,
            url="https://example.com/article",
            domain="example.com",
            title="Test Article"
        )
        
        # Add chunks
        evidence.add_chunk("chunk_001")
        evidence.add_chunk("chunk_002")
        assert evidence.chunk_ids == ["chunk_001", "chunk_002"]
        
        # Add duplicate chunk (should not add)
        evidence.add_chunk("chunk_001")
        assert evidence.chunk_ids == ["chunk_001", "chunk_002"]
        
        # Remove chunk
        evidence.remove_chunk("chunk_001")
        assert evidence.chunk_ids == ["chunk_002"]
        
        # Remove non-existent chunk
        evidence.remove_chunk("chunk_999")
        assert evidence.chunk_ids == ["chunk_002"]
    
    def test_evidence_score_updates(self):
        """Test updating evidence scores."""
        evidence = Evidence(
            id="ev_009",
            claim_id="claim_009",
            source_type=SourceType.NEWS_ARTICLE,
            url="https://example.com/article",
            domain="example.com",
            title="Test Article"
        )
        
        # Initial scores should be 0.0
        assert evidence.scores.relevance == 0.0
        assert evidence.scores.freshness == 0.0
        assert evidence.scores.source == 0.0
        assert evidence.scores.final == 0.0
        
        # Update individual scores
        evidence.update_scores(relevance=0.8, freshness=0.6)
        assert evidence.scores.relevance == 0.8
        assert evidence.scores.freshness == 0.6
        assert evidence.scores.source == 0.0
        
        # Final score should be recalculated
        expected_final = 0.8 * 0.4 + 0.6 * 0.2 + 0.0 * 0.4
        assert abs(evidence.scores.final - expected_final) < 0.001
        
        # Update source score
        evidence.update_scores(source=0.9)
        assert evidence.scores.source == 0.9
        
        # Final score should be recalculated again
        expected_final = 0.8 * 0.4 + 0.6 * 0.2 + 0.9 * 0.4
        assert abs(evidence.scores.final - expected_final) < 0.001
    
    def test_evidence_enum_values(self):
        """Test all enum values are accessible."""
        # Test SourceType enum values
        assert SourceType.NEWS_ARTICLE == "news_article"
        assert SourceType.RESEARCH_PAPER == "research_paper"
        assert SourceType.GOVERNMENT_DOCUMENT == "government_document"
        assert SourceType.SOCIAL_MEDIA == "social_media"
        assert SourceType.BLOG_POST == "blog_post"
        assert SourceType.VIDEO == "video"
        assert SourceType.PODCAST == "podcast"
        assert SourceType.OTHER == "other"
        
        # Test SupportLabel enum values
        assert SupportLabel.SUPPORTS == "supports"
        assert SupportLabel.REFUTES == "refutes"
        assert SupportLabel.NEUTRAL == "neutral"
        assert SupportLabel.MIXED == "mixed"
        assert SupportLabel.UNCLEAR == "unclear"


class TestEvidenceEdgeCases:
    """Test edge cases and error handling."""
    
    def test_evidence_empty_text_hash(self):
        """Test evidence with empty full_text doesn't compute hash."""
        evidence = Evidence(
            id="ev_010",
            claim_id="claim_010",
            source_type=SourceType.NEWS_ARTICLE,
            url="https://example.com/article",
            domain="example.com",
            title="Test Article",
            full_text=""
        )
        
        assert evidence.full_text_hash == ""
    
    def test_evidence_none_published_at(self):
        """Test evidence with None published_at."""
        evidence = Evidence(
            id="ev_011",
            claim_id="claim_011",
            source_type=SourceType.NEWS_ARTICLE,
            url="https://example.com/article",
            domain="example.com",
            title="Test Article",
            published_at=None
        )
        
        evidence_dict = evidence.to_dict()
        assert evidence_dict["published_at"] is None
    
    def test_evidence_metadata_persistence(self):
        """Test that metadata is properly preserved through serialization."""
        metadata = {
            "author": "John Doe",
            "tags": ["politics", "fact-check"],
            "nested": {"level1": {"level2": "value"}},
            "numbers": [1, 2, 3],
            "boolean": True
        }
        
        evidence = Evidence(
            id="ev_012",
            claim_id="claim_012",
            source_type=SourceType.NEWS_ARTICLE,
            url="https://example.com/article",
            domain="example.com",
            title="Test Article",
            metadata=metadata
        )
        
        # Test metadata preservation
        assert evidence.metadata == metadata
        
        # Test through serialization cycle
        evidence_dict = evidence.to_dict()
        evidence_json = evidence.to_json()
        evidence_reconstructed = Evidence.from_json(evidence_json)
        
        assert evidence_reconstructed.metadata == metadata


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
