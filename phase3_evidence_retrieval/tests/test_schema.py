"""
Tests for TruthLens Phase 3 Evidence Retrieval System schemas.

This module tests the Pydantic models and utility functions for evidence data structures.
"""

import pytest
import json
from datetime import datetime, timezone
from typing import Dict, Any

# Import the schemas to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from schemas.evidence import (
    SourceType,
    SupportLabel,
    Language,
    EvidenceScores,
    EvidenceMetadata,
    TextChunk,
    RawEvidence,
    ProcessedEvidence,
    EvidenceChunk,
    RetrievalQuery,
    RetrievalResult,
    EvidenceBatch,
    ProcessingStatus,
    ProcessingResult,
    EvidenceSummary,
    evidence_to_dict,
    evidence_from_dict,
    evidence_to_json,
    evidence_from_json,
    validate_evidence_url,
    get_evidence_domain,
    calculate_text_hash,
    estimate_reading_time,
    get_language_from_text
)


class TestEvidenceScores:
    """Test EvidenceScores model."""
    
    def test_evidence_scores_creation(self):
        """Test creating EvidenceScores with default values."""
        scores = EvidenceScores()
        assert scores.relevance == 0.0
        assert scores.freshness == 0.0
        assert scores.source == 0.0
        assert scores.factuality == 0.0
        assert scores.final == 0.0
    
    def test_evidence_scores_custom_values(self):
        """Test creating EvidenceScores with custom values."""
        scores = EvidenceScores(
            relevance=0.8,
            freshness=0.6,
            source=0.9,
            factuality=0.7
        )
        assert scores.relevance == 0.8
        assert scores.freshness == 0.6
        assert scores.source == 0.9
        assert scores.factuality == 0.7
        # Final score should be calculated automatically
        expected_final = 0.8 * 0.4 + 0.6 * 0.2 + 0.9 * 0.3 + 0.7 * 0.1
        assert abs(scores.final - expected_final) < 0.001
    
    def test_evidence_scores_validation(self):
        """Test score validation bounds."""
        # Valid scores
        scores = EvidenceScores(relevance=0.5, freshness=0.5, source=0.5, factuality=0.5)
        assert scores.relevance == 0.5
        
        # Invalid scores should raise validation error
        with pytest.raises(ValueError):
            EvidenceScores(relevance=1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            EvidenceScores(freshness=-0.1)  # < 0.0


class TestEvidenceMetadata:
    """Test EvidenceMetadata model."""
    
    def test_evidence_metadata_creation(self):
        """Test creating EvidenceMetadata with various fields."""
        metadata = EvidenceMetadata(
            author="John Doe",
            publication_date=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            word_count=1000,
            tags=["science", "research"],
            categories=["academic"],
            sentiment="positive",
            toxicity_score=0.1,
            factuality_score=0.9,
            source_reputation=0.8,
            verification_status="verified",
            fact_check_rating="true",
            citations=["https://example.com/paper1", "https://example.com/paper2"],
            doi="10.1234/paper.2024.001",
            journal="Nature",
            volume="615",
            issue="1",
            pages="45-52",
            peer_reviewed=True,
            funding_source="National Science Foundation",
            conflicts_of_interest="None declared",
            accessibility="open_access",
            license="CC-BY-4.0",
            copyright="2024 John Doe"
        )
        
        assert metadata.author == "John Doe"
        assert metadata.word_count == 1000
        assert metadata.tags == ["science", "research"]
        assert metadata.sentiment == "positive"
        assert metadata.toxicity_score == 0.1
        assert metadata.factuality_score == 0.9
        assert metadata.doi == "10.1234/paper.2024.001"
        assert metadata.peer_reviewed is True
    
    def test_evidence_metadata_defaults(self):
        """Test EvidenceMetadata default values."""
        metadata = EvidenceMetadata()
        assert metadata.author is None
        assert metadata.tags == []
        assert metadata.categories == []
        assert metadata.entities == []
        assert metadata.keywords == []
        assert metadata.citations == []
        assert metadata.corrections == []
        assert metadata.custom_fields == {}


class TestTextChunk:
    """Test TextChunk model."""
    
    def test_text_chunk_creation(self):
        """Test creating TextChunk with all fields."""
        chunk = TextChunk(
            chunk_id="chunk_001",
            text="This is a sample text chunk for testing purposes.",
            chunk_index=0,
            start_char=0,
            end_char=50,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            embedding_model="intfloat/multilingual-e5-large",
            language=Language.ENGLISH,
            metadata={"confidence": 0.95, "topic": "science"}
        )
        
        assert chunk.chunk_id == "chunk_001"
        assert chunk.text == "This is a sample text chunk for testing purposes."
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 50
        assert chunk.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert chunk.embedding_model == "intfloat/multilingual-e5-large"
        assert chunk.language == Language.ENGLISH
        assert chunk.metadata["confidence"] == 0.95
    
    def test_text_chunk_minimal(self):
        """Test creating TextChunk with minimal required fields."""
        chunk = TextChunk(
            chunk_id="chunk_002",
            text="Minimal chunk",
            chunk_index=1,
            start_char=51,
            end_char=63
        )
        
        assert chunk.chunk_id == "chunk_002"
        assert chunk.text == "Minimal chunk"
        assert chunk.embedding is None
        assert chunk.embedding_model is None
        assert chunk.language == Language.ENGLISH
        assert chunk.metadata is None


class TestRawEvidence:
    """Test RawEvidence model."""
    
    def test_raw_evidence_creation(self):
        """Test creating RawEvidence with all fields."""
        evidence = RawEvidence(
            id="ev_001",
            claim_id="claim_001",
            source_type=SourceType.FACT_CHECK,
            url="https://example.com/fact-check",
            domain="example.com",
            title="Fact Check: Climate Change Claims",
            published_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            language=Language.ENGLISH,
            raw_html="<html><body>Fact check content</body></html>",
            raw_text="Fact check content here",
            snippet="A comprehensive fact check of climate change claims",
            full_text="Full fact check content with detailed analysis...",
            text_hash="abc123hash",
            metadata=EvidenceMetadata(author="Fact Checker", tags=["climate", "fact-check"]),
            source_config={"rate_limit": 10, "timeout": 30},
            retrieval_metadata={"scraper": "beautifulsoup", "processing_time": 1.5}
        )
        
        assert evidence.id == "ev_001"
        assert evidence.claim_id == "claim_001"
        assert evidence.source_type == SourceType.FACT_CHECK
        assert evidence.url == "https://example.com/fact-check"
        assert evidence.domain == "example.com"
        assert evidence.title == "Fact Check: Climate Change Claims"
        assert evidence.language == Language.ENGLISH
        assert evidence.raw_html == "<html><body>Fact check content</body></html>"
        assert evidence.raw_text == "Fact check content here"
        assert evidence.text_hash == "abc123hash"
        assert evidence.metadata.author == "Fact Checker"
        assert evidence.source_config["rate_limit"] == 10
    
    def test_raw_evidence_minimal(self):
        """Test creating RawEvidence with minimal required fields."""
        evidence = RawEvidence(
            id="ev_002",
            source_type=SourceType.NEWS,
            url="https://news.com/article",
            domain="news.com",
            title="News Article"
        )
        
        assert evidence.id == "ev_002"
        assert evidence.source_type == SourceType.NEWS
        assert evidence.url == "https://news.com/article"
        assert evidence.domain == "news.com"
        assert evidence.title == "News Article"
        assert evidence.claim_id is None
        assert evidence.published_at is None
        assert evidence.language == Language.ENGLISH
        assert evidence.retrieved_at is not None  # Should be set automatically


class TestProcessedEvidence:
    """Test ProcessedEvidence model."""
    
    def test_processed_evidence_creation(self):
        """Test creating ProcessedEvidence with all fields."""
        # Create chunks first
        chunks = [
            TextChunk(
                chunk_id="chunk_001",
                text="First chunk of evidence",
                chunk_index=0,
                start_char=0,
                end_char=25,
                embedding=[0.1, 0.2, 0.3],
                embedding_model="test-model",
                language=Language.ENGLISH
            ),
            TextChunk(
                chunk_id="chunk_002",
                text="Second chunk of evidence",
                chunk_index=1,
                start_char=26,
                end_char=50,
                embedding=[0.4, 0.5, 0.6],
                embedding_model="test-model",
                language=Language.ENGLISH
            )
        ]
        
        evidence = ProcessedEvidence(
            id="ev_processed_001",
            raw_evidence_id="ev_001",
            claim_id="claim_001",
            source_type=SourceType.ACADEMIC,
            url="https://academic.com/paper",
            domain="academic.com",
            title="Academic Research Paper",
            published_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            retrieved_at=datetime(2024, 1, 20, 15, 0, 0, tzinfo=timezone.utc),
            language=Language.ENGLISH,
            snippet="Research findings on climate change",
            full_text="Complete research paper content with detailed analysis...",
            text_hash="processed_hash_123",
            chunks=chunks,
            support_label=SupportLabel.SUPPORTS,
            scores=EvidenceScores(relevance=0.9, freshness=0.7, source=0.8, factuality=0.9),
            metadata=EvidenceMetadata(
                author="Dr. Smith",
                doi="10.1234/paper.2024.001",
                journal="Nature",
                peer_reviewed=True
            ),
            processing_metadata={"pipeline_version": "1.0", "processing_time": 2.5},
            quality_flags=["high_quality", "peer_reviewed"],
            accessibility_score=0.8,
            readability_score=0.7,
            complexity_score=0.6
        )
        
        assert evidence.id == "ev_processed_001"
        assert evidence.raw_evidence_id == "ev_001"
        assert evidence.source_type == SourceType.ACADEMIC
        assert evidence.chunks == chunks
        assert len(evidence.chunks) == 2
        assert evidence.support_label == SupportLabel.SUPPORTS
        assert evidence.scores.relevance == 0.9
        assert evidence.metadata.author == "Dr. Smith"
        assert evidence.metadata.peer_reviewed is True
        assert evidence.quality_flags == ["high_quality", "peer_reviewed"]
        assert evidence.accessibility_score == 0.8


class TestRetrievalQuery:
    """Test RetrievalQuery model."""
    
    def test_retrieval_query_creation(self):
        """Test creating RetrievalQuery with all fields."""
        query = RetrievalQuery(
            claim="Climate change is caused by human activities",
            max_results=20,
            min_relevance=0.7,
            sources=[SourceType.FACT_CHECK, SourceType.ACADEMIC, SourceType.GOVERNMENT],
            language=Language.ENGLISH,
            date_range={
                "start": datetime(2020, 1, 1, tzinfo=timezone.utc),
                "end": datetime(2024, 12, 31, tzinfo=timezone.utc)
            },
            domains=["example.com", "academic.org"],
            exclude_domains=["fake-news.com"],
            keywords=["climate", "human", "emissions"],
            exclude_keywords=["denial", "hoax"],
            support_label=SupportLabel.SUPPORTS,
            min_freshness=0.6,
            min_source_trust=0.8,
            include_metadata=True,
            include_chunks=True,
            deduplicate=True,
            rerank=True,
            cache_results=True
        )
        
        assert query.claim == "Climate change is caused by human activities"
        assert query.max_results == 20
        assert query.min_relevance == 0.7
        assert SourceType.FACT_CHECK in query.sources
        assert query.language == Language.ENGLISH
        assert query.domains == ["example.com", "academic.org"]
        assert query.exclude_domains == ["fake-news.com"]
        assert query.keywords == ["climate", "human", "emissions"]
        assert query.support_label == SupportLabel.SUPPORTS
        assert query.min_freshness == 0.6
        assert query.min_source_trust == 0.8
        assert query.include_metadata is True
        assert query.include_chunks is True
        assert query.deduplicate is True
        assert query.rerank is True
        assert query.cache_results is True
    
    def test_retrieval_query_defaults(self):
        """Test RetrievalQuery default values."""
        query = RetrievalQuery(claim="Test claim")
        
        assert query.claim == "Test claim"
        assert query.max_results == 10
        assert query.min_relevance == 0.3
        assert query.sources == []
        assert query.language == Language.ENGLISH
        assert query.domains == []
        assert query.exclude_domains == []
        assert query.keywords == []
        assert query.exclude_keywords == []
        assert query.support_label is None
        assert query.min_freshness is None
        assert query.min_source_trust is None
        assert query.include_metadata is True
        assert query.include_chunks is True
        assert query.deduplicate is True
        assert query.rerank is True
        assert query.cache_results is True
    
    def test_retrieval_query_validation(self):
        """Test RetrievalQuery validation."""
        # Valid query
        query = RetrievalQuery(claim="Test claim", max_results=50)
        assert query.max_results == 50
        
        # Invalid max_results should raise validation error
        with pytest.raises(ValueError):
            RetrievalQuery(claim="Test claim", max_results=0)  # < 1
        
        with pytest.raises(ValueError):
            RetrievalQuery(claim="Test claim", max_results=101)  # > 100
        
        # Invalid min_relevance should raise validation error
        with pytest.raises(ValueError):
            RetrievalQuery(claim="Test claim", min_relevance=1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            RetrievalQuery(claim="Test claim", min_relevance=-0.1)  # < 0.0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_evidence_to_dict(self):
        """Test converting evidence to dictionary."""
        evidence = RawEvidence(
            id="ev_001",
            source_type=SourceType.NEWS,
            url="https://example.com/article",
            domain="example.com",
            title="Test Article"
        )
        
        evidence_dict = evidence_to_dict(evidence)
        assert isinstance(evidence_dict, dict)
        assert evidence_dict["id"] == "ev_001"
        assert evidence_dict["source_type"] == "news"
        assert evidence_dict["url"] == "https://example.com/article"
        assert evidence_dict["domain"] == "example.com"
        assert evidence_dict["title"] == "Test Article"
    
    def test_evidence_from_dict(self):
        """Test creating evidence from dictionary."""
        data = {
            "id": "ev_002",
            "source_type": "fact_check",
            "url": "https://factcheck.org/article",
            "domain": "factcheck.org",
            "title": "Fact Check Article"
        }
        
        evidence = evidence_from_dict(data, "raw")
        assert isinstance(evidence, RawEvidence)
        assert evidence.id == "ev_002"
        assert evidence.source_type == SourceType.FACT_CHECK
        assert evidence.url == "https://factcheck.org/article"
        assert evidence.domain == "factcheck.org"
        assert evidence.title == "Fact Check Article"
    
    def test_evidence_to_json(self):
        """Test converting evidence to JSON string."""
        evidence = RawEvidence(
            id="ev_003",
            source_type=SourceType.ACADEMIC,
            url="https://academic.org/paper",
            domain="academic.org",
            title="Academic Paper"
        )
        
        json_str = evidence_to_json(evidence)
        assert isinstance(json_str, str)
        
        # Parse back to verify
        parsed = json.loads(json_str)
        assert parsed["id"] == "ev_003"
        assert parsed["source_type"] == "academic"
        assert parsed["url"] == "https://academic.org/paper"
    
    def test_evidence_from_json(self):
        """Test creating evidence from JSON string."""
        json_str = '''
        {
            "id": "ev_004",
            "source_type": "government",
            "url": "https://gov.org/report",
            "domain": "gov.org",
            "title": "Government Report"
        }
        '''
        
        evidence = evidence_from_json(json_str, "raw")
        assert isinstance(evidence, RawEvidence)
        assert evidence.id == "ev_004"
        assert evidence.source_type == SourceType.GOVERNMENT
        assert evidence.url == "https://gov.org/report"
        assert evidence.domain == "gov.org"
        assert evidence.title == "Government Report"
    
    def test_validate_evidence_url(self):
        """Test URL validation."""
        assert validate_evidence_url("https://example.com/article") is True
        assert validate_evidence_url("http://example.com/article") is True
        assert validate_evidence_url("https://example.com") is True
        assert validate_evidence_url("invalid-url") is False
        assert validate_evidence_url("") is False
    
    def test_get_evidence_domain(self):
        """Test domain extraction."""
        assert get_evidence_domain("https://example.com/article") == "example.com"
        assert get_evidence_domain("http://subdomain.example.org/path") == "subdomain.example.org"
        assert get_evidence_domain("https://www.news.com") == "www.news.com"
        assert get_evidence_domain("invalid-url") == ""
        assert get_evidence_domain("") == ""
    
    def test_calculate_text_hash(self):
        """Test text hash calculation."""
        text = "This is test text for hashing"
        hash1 = calculate_text_hash(text)
        hash2 = calculate_text_hash(text)
        
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length
        assert hash1 == hash2  # Same text should produce same hash
        
        # Different text should produce different hash
        hash3 = calculate_text_hash("Different text")
        assert hash1 != hash3
    
    def test_estimate_reading_time(self):
        """Test reading time estimation."""
        text = "This is a short text with ten words for testing purposes."
        reading_time = estimate_reading_time(text)
        
        assert isinstance(reading_time, int)
        assert reading_time >= 1  # Should be at least 1 minute
        
        # Test with different words per minute
        reading_time_fast = estimate_reading_time(text, words_per_minute=300)
        reading_time_slow = estimate_reading_time(text, words_per_minute=100)
        assert reading_time_fast < reading_time_slow
    
    def test_get_language_from_text(self):
        """Test language detection."""
        # Test English
        english_text = "This is English text for testing language detection."
        assert get_language_from_text(english_text) == Language.ENGLISH
        
        # Test Hindi (if langdetect supports it)
        # hindi_text = "यह हिंदी पाठ है भाषा पहचान के लिए।"
        # assert get_language_from_text(hindi_text) == Language.HINDI
        
        # Test empty text
        assert get_language_from_text("") == Language.ENGLISH
        
        # Test very short text
        assert get_language_from_text("Hi") == Language.ENGLISH


class TestEnums:
    """Test enum values."""
    
    def test_source_type_values(self):
        """Test SourceType enum values."""
        assert SourceType.FACT_CHECK == "fact_check"
        assert SourceType.NEWS == "news"
        assert SourceType.WIKIPEDIA == "wikipedia"
        assert SourceType.GOVERNMENT == "government"
        assert SourceType.ACADEMIC == "academic"
        assert SourceType.SOCIAL_MEDIA == "social_media"
        assert SourceType.BLOG == "blog"
        assert SourceType.VIDEO == "video"
        assert SourceType.PODCAST == "podcast"
        assert SourceType.OTHER == "other"
    
    def test_support_label_values(self):
        """Test SupportLabel enum values."""
        assert SupportLabel.SUPPORTS == "supports"
        assert SupportLabel.REFUTES == "refutes"
        assert SupportLabel.NEUTRAL == "neutral"
        assert SupportLabel.MIXED == "mixed"
        assert SupportLabel.UNCLEAR == "unclear"
    
    def test_language_values(self):
        """Test Language enum values."""
        assert Language.ENGLISH == "en"
        assert Language.HINDI == "hi"
        assert Language.SPANISH == "es"
        assert Language.FRENCH == "fr"
        assert Language.GERMAN == "de"
        assert Language.CHINESE == "zh"
        assert Language.ARABIC == "ar"
        assert Language.RUSSIAN == "ru"
        assert Language.PORTUGUESE == "pt"
        assert Language.JAPANESE == "ja"
        assert Language.KOREAN == "ko"
        assert Language.ITALIAN == "it"
        assert Language.DUTCH == "nl"
        assert Language.SWEDISH == "sv"
        assert Language.NORWEGIAN == "no"
        assert Language.DANISH == "da"
        assert Language.FINNISH == "fi"
        assert Language.POLISH == "pl"
        assert Language.TURKISH == "tr"
        assert Language.GREEK == "el"
    
    def test_processing_status_values(self):
        """Test ProcessingStatus enum values."""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PROCESSING == "processing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
        assert ProcessingStatus.CANCELLED == "cancelled"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
