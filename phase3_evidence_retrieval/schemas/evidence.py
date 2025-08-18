"""
Evidence schemas for TruthLens Phase 3: Evidence Retrieval System.

This module defines Pydantic models for evidence data structures used throughout
the evidence retrieval pipeline, including raw evidence, processed evidence,
and retrieval results.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl
import json


class SourceType(str, Enum):
    """Types of evidence sources."""
    FACT_CHECK = "fact_check"
    NEWS = "news"
    WIKIPEDIA = "wikipedia"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    SOCIAL_MEDIA = "social_media"
    BLOG = "blog"
    VIDEO = "video"
    PODCAST = "podcast"
    OTHER = "other"


class SupportLabel(str, Enum):
    """Evidence support labels for claims."""
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class Language(str, Enum):
    """Supported languages for evidence processing."""
    ENGLISH = "en"
    HINDI = "hi"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    ARABIC = "ar"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    JAPANESE = "ja"
    KOREAN = "ko"
    ITALIAN = "it"
    DUTCH = "nl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"
    POLISH = "pl"
    TURKISH = "tr"
    GREEK = "el"


class EvidenceScores(BaseModel):
    """Scores for evidence quality and relevance."""
    relevance: float = Field(0.0, ge=0.0, le=1.0, description="Relevance to the claim")
    freshness: float = Field(0.0, ge=0.0, le=1.0, description="Recency of the evidence")
    source: float = Field(0.0, ge=0.0, le=1.0, description="Source credibility score")
    factuality: float = Field(0.0, ge=0.0, le=1.0, description="Factual accuracy score")
    final: float = Field(0.0, ge=0.0, le=1.0, description="Final weighted score")
    
    @validator('final', pre=True, always=True)
    def calculate_final_score(cls, v, values):
        """Calculate final score if not provided."""
        if v == 0.0 and all(key in values for key in ['relevance', 'freshness', 'source', 'factuality']):
            # Weighted average calculation
            weights = {'relevance': 0.4, 'freshness': 0.2, 'source': 0.3, 'factuality': 0.1}
            return sum(values[key] * weights[key] for key in weights.keys())
        return v


class EvidenceMetadata(BaseModel):
    """Additional metadata for evidence."""
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    word_count: Optional[int] = None
    reading_time: Optional[int] = None  # in minutes
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    sentiment: Optional[str] = None
    toxicity_score: Optional[float] = None
    factuality_score: Optional[float] = None
    source_reputation: Optional[float] = None
    verification_status: Optional[str] = None
    fact_check_rating: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    doi: Optional[str] = None
    isbn: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    peer_reviewed: Optional[bool] = None
    funding_source: Optional[str] = None
    conflicts_of_interest: Optional[str] = None
    retraction_notice: Optional[str] = None
    corrections: List[str] = Field(default_factory=list)
    social_media_metrics: Optional[Dict[str, Any]] = None
    engagement_metrics: Optional[Dict[str, Any]] = None
    accessibility: Optional[str] = None
    license: Optional[str] = None
    copyright: Optional[str] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    """A chunk of text from evidence with embeddings."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="The text content of the chunk")
    chunk_index: int = Field(..., description="Order of the chunk in the document")
    start_char: int = Field(..., description="Starting character position")
    end_char: int = Field(..., description="Ending character position")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    language: Language = Field(Language.ENGLISH, description="Language of the chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional chunk metadata")


class RawEvidence(BaseModel):
    """Raw evidence data as retrieved from sources."""
    id: str = Field(..., description="Unique identifier for the evidence")
    claim_id: Optional[str] = Field(None, description="Associated claim identifier")
    source_type: SourceType = Field(..., description="Type of evidence source")
    url: HttpUrl = Field(..., description="Source URL")
    domain: str = Field(..., description="Source domain")
    title: str = Field(..., description="Evidence title")
    published_at: Optional[datetime] = Field(None, description="Publication date")
    retrieved_at: datetime = Field(default_factory=datetime.utcnow, description="Retrieval timestamp")
    language: Language = Field(Language.ENGLISH, description="Content language")
    raw_html: Optional[str] = Field(None, description="Raw HTML content")
    raw_text: Optional[str] = Field(None, description="Raw text content")
    snippet: Optional[str] = Field(None, description="Short excerpt")
    full_text: Optional[str] = Field(None, description="Complete text content")
    text_hash: Optional[str] = Field(None, description="Hash of the text content")
    metadata: EvidenceMetadata = Field(default_factory=EvidenceMetadata, description="Additional metadata")
    source_config: Dict[str, Any] = Field(default_factory=dict, description="Source-specific configuration")
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict, description="Retrieval process metadata")


class ProcessedEvidence(BaseModel):
    """Processed evidence with NLP enhancements."""
    id: str = Field(..., description="Unique identifier for the evidence")
    raw_evidence_id: str = Field(..., description="Reference to raw evidence")
    claim_id: Optional[str] = Field(None, description="Associated claim identifier")
    source_type: SourceType = Field(..., description="Type of evidence source")
    url: HttpUrl = Field(..., description="Source URL")
    domain: str = Field(..., description="Source domain")
    title: str = Field(..., description="Evidence title")
    published_at: Optional[datetime] = Field(None, description="Publication date")
    retrieved_at: datetime = Field(..., description="Retrieval timestamp")
    language: Language = Field(..., description="Content language")
    snippet: str = Field(..., description="Short excerpt")
    full_text: str = Field(..., description="Complete text content")
    text_hash: str = Field(..., description="Hash of the text content")
    chunks: List[TextChunk] = Field(default_factory=list, description="Text chunks")
    support_label: SupportLabel = Field(SupportLabel.NEUTRAL, description="Support label for the claim")
    scores: EvidenceScores = Field(default_factory=EvidenceScores, description="Quality scores")
    metadata: EvidenceMetadata = Field(..., description="Enhanced metadata")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing pipeline metadata")
    quality_flags: List[str] = Field(default_factory=list, description="Quality assessment flags")
    accessibility_score: Optional[float] = Field(None, description="Content accessibility score")
    readability_score: Optional[float] = Field(None, description="Text readability score")
    complexity_score: Optional[float] = Field(None, description="Content complexity score")


class EvidenceChunk(BaseModel):
    """Evidence chunk for vector search and retrieval."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    evidence_id: str = Field(..., description="Reference to the evidence")
    claim_id: Optional[str] = Field(None, description="Associated claim identifier")
    text: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Order of the chunk")
    embedding: List[float] = Field(..., description="Vector embedding")
    embedding_model: str = Field(..., description="Model used for embedding")
    language: Language = Field(..., description="Language of the chunk")
    relevance_score: float = Field(0.0, description="Relevance to the claim")
    support_label: SupportLabel = Field(SupportLabel.NEUTRAL, description="Support label")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class RetrievalQuery(BaseModel):
    """Query for evidence retrieval."""
    claim: str = Field(..., description="The claim to find evidence for")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results")
    min_relevance: float = Field(0.3, ge=0.0, le=1.0, description="Minimum relevance score")
    sources: List[SourceType] = Field(default_factory=list, description="Preferred source types")
    language: Language = Field(Language.ENGLISH, description="Preferred language")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range filter")
    domains: List[str] = Field(default_factory=list, description="Preferred domains")
    exclude_domains: List[str] = Field(default_factory=list, description="Excluded domains")
    keywords: List[str] = Field(default_factory=list, description="Required keywords")
    exclude_keywords: List[str] = Field(default_factory=list, description="Excluded keywords")
    support_label: Optional[SupportLabel] = Field(None, description="Preferred support label")
    min_freshness: Optional[float] = Field(None, description="Minimum freshness score")
    min_source_trust: Optional[float] = Field(None, description="Minimum source trust score")
    include_metadata: bool = Field(True, description="Include detailed metadata")
    include_chunks: bool = Field(True, description="Include text chunks")
    deduplicate: bool = Field(True, description="Remove duplicate evidence")
    rerank: bool = Field(True, description="Apply reranking")
    cache_results: bool = Field(True, description="Cache retrieval results")


class RetrievalResult(BaseModel):
    """Result of evidence retrieval."""
    query: RetrievalQuery = Field(..., description="The original query")
    evidence: List[ProcessedEvidence] = Field(default_factory=list, description="Retrieved evidence")
    total_found: int = Field(0, description="Total number of evidence found")
    total_retrieved: int = Field(0, description="Total number of evidence retrieved")
    processing_time_ms: int = Field(0, description="Processing time in milliseconds")
    cache_hit: bool = Field(False, description="Whether result was served from cache")
    sources_used: List[SourceType] = Field(default_factory=list, description="Sources that were queried")
    languages_found: List[Language] = Field(default_factory=list, description="Languages found in results")
    date_range_covered: Optional[Dict[str, datetime]] = Field(None, description="Date range covered by results")
    quality_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of quality scores")
    support_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of support labels")
    source_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of source types")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional retrieval metadata")


class EvidenceBatch(BaseModel):
    """Batch of evidence for processing."""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    evidence: List[RawEvidence] = Field(..., description="Evidence in the batch")
    batch_size: int = Field(..., description="Number of evidence in the batch")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Batch creation timestamp")
    priority: int = Field(1, ge=1, le=10, description="Processing priority")
    source_type: SourceType = Field(..., description="Source type for the batch")
    processing_status: str = Field("pending", description="Current processing status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")


class ProcessingStatus(str, Enum):
    """Status of evidence processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingResult(BaseModel):
    """Result of evidence processing."""
    evidence_id: str = Field(..., description="Evidence identifier")
    status: ProcessingStatus = Field(..., description="Processing status")
    processed_evidence: Optional[ProcessedEvidence] = Field(None, description="Processed evidence")
    chunks_created: int = Field(0, description="Number of chunks created")
    embeddings_generated: int = Field(0, description="Number of embeddings generated")
    processing_time_ms: int = Field(0, description="Processing time in milliseconds")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


class EvidenceSummary(BaseModel):
    """Summary statistics for evidence."""
    total_evidence: int = Field(0, description="Total number of evidence")
    total_chunks: int = Field(0, description="Total number of chunks")
    source_distribution: Dict[SourceType, int] = Field(default_factory=dict, description="Distribution by source type")
    language_distribution: Dict[Language, int] = Field(default_factory=dict, description="Distribution by language")
    support_distribution: Dict[SupportLabel, int] = Field(default_factory=dict, description="Distribution by support label")
    quality_distribution: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Distribution by quality scores")
    date_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution by date")
    domain_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution by domain")
    average_scores: Dict[str, float] = Field(default_factory=dict, description="Average quality scores")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


# Utility functions
def evidence_to_dict(evidence: Union[RawEvidence, ProcessedEvidence]) -> Dict[str, Any]:
    """Convert evidence to dictionary."""
    return evidence.dict()


def evidence_from_dict(data: Dict[str, Any], evidence_type: str = "processed") -> Union[RawEvidence, ProcessedEvidence]:
    """Create evidence from dictionary."""
    if evidence_type == "raw":
        return RawEvidence(**data)
    else:
        return ProcessedEvidence(**data)


def evidence_to_json(evidence: Union[RawEvidence, ProcessedEvidence]) -> str:
    """Convert evidence to JSON string."""
    return evidence.json(indent=2)


def evidence_from_json(json_str: str, evidence_type: str = "processed") -> Union[RawEvidence, ProcessedEvidence]:
    """Create evidence from JSON string."""
    data = json.loads(json_str)
    return evidence_from_dict(data, evidence_type)


def validate_evidence_url(url: str) -> bool:
    """Validate evidence URL format."""
    try:
        HttpUrl(url)
        return True
    except:
        return False


def get_evidence_domain(url: str) -> str:
    """Extract domain from evidence URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return ""


def calculate_text_hash(text: str) -> str:
    """Calculate hash of text content."""
    import hashlib
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time in minutes."""
    word_count = len(text.split())
    return max(1, round(word_count / words_per_minute))


def get_language_from_text(text: str) -> Language:
    """Detect language from text content."""
    try:
        from langdetect import detect
        lang_code = detect(text)
        # Map language codes to our enum
        lang_mapping = {
            'en': Language.ENGLISH,
            'hi': Language.HINDI,
            'es': Language.SPANISH,
            'fr': Language.FRENCH,
            'de': Language.GERMAN,
            'zh': Language.CHINESE,
            'ar': Language.ARABIC,
            'ru': Language.RUSSIAN,
            'pt': Language.PORTUGUESE,
            'ja': Language.JAPANESE,
            'ko': Language.KOREAN,
            'it': Language.ITALIAN,
            'nl': Language.DUTCH,
            'sv': Language.SWEDISH,
            'no': Language.NORWEGIAN,
            'da': Language.DANISH,
            'fi': Language.FINNISH,
            'pl': Language.POLISH,
            'tr': Language.TURKISH,
            'el': Language.GREEK
        }
        return lang_mapping.get(lang_code, Language.ENGLISH)
    except:
        return Language.ENGLISH
