"""
Evidence schema for TruthLens fact-checking system.

This module defines the Evidence dataclass and related schemas for storing
and managing evidence data used in claim verification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import xxhash
from enum import Enum


class SourceType(str, Enum):
    """Types of evidence sources."""
    NEWS_ARTICLE = "news_article"
    RESEARCH_PAPER = "research_paper"
    GOVERNMENT_DOCUMENT = "government_document"
    SOCIAL_MEDIA = "social_media"
    BLOG_POST = "blog_post"
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


@dataclass
class EvidenceScores:
    """Scores for evidence quality and relevance."""
    relevance: float = 0.0  # How relevant is this evidence to the claim
    freshness: float = 0.0  # How recent is this evidence
    source: float = 0.0     # Source credibility score
    final: float = 0.0      # Final weighted score
    
    def to_dict(self) -> Dict[str, float]:
        """Convert scores to dictionary."""
        return {
            "relevance": self.relevance,
            "freshness": self.freshness,
            "source": self.source,
            "final": self.final
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'EvidenceScores':
        """Create EvidenceScores from dictionary."""
        return cls(**data)


@dataclass
class Evidence:
    """Evidence dataclass for storing claim verification data."""
    
    # Core identifiers
    id: str
    claim_id: str
    
    # Source metadata
    source_type: SourceType
    url: str
    domain: str
    title: str
    
    # Temporal information
    published_at: Optional[datetime] = None
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    
    # Content information
    language: str = "en"
    snippet: str = ""
    full_text: str = ""
    full_text_hash: str = ""
    
    # Chunking and embeddings
    chunk_ids: List[str] = field(default_factory=list)
    
    # Classification and scoring
    support_label: SupportLabel = SupportLabel.NEUTRAL
    scores: EvidenceScores = field(default_factory=EvidenceScores)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.full_text_hash and self.full_text:
            self.full_text_hash = self._compute_text_hash()
    
    def _compute_text_hash(self) -> str:
        """Compute hash of full text using xxhash."""
        if not self.full_text:
            return ""
        return xxhash.xxh64(self.full_text.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Evidence to dictionary for serialization."""
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "source_type": self.source_type.value,
            "url": self.url,
            "domain": self.domain,
            "title": self.title,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "retrieved_at": self.retrieved_at.isoformat(),
            "language": self.language,
            "snippet": self.snippet,
            "full_text_hash": self.full_text_hash,
            "chunk_ids": self.chunk_ids,
            "support_label": self.support_label.value,
            "scores": self.scores.to_dict(),
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert Evidence to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Evidence':
        """Create Evidence from dictionary."""
        # Handle datetime fields
        if data.get('published_at'):
            data['published_at'] = datetime.fromisoformat(data['published_at'])
        if data.get('retrieved_at'):
            data['retrieved_at'] = datetime.fromisoformat(data['retrieved_at'])
        
        # Handle enum fields
        if data.get('source_type'):
            data['source_type'] = SourceType(data['source_type'])
        if data.get('support_label'):
            data['support_label'] = SupportLabel(data['support_label'])
        
        # Handle scores
        if data.get('scores'):
            data['scores'] = EvidenceScores.from_dict(data['scores'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Evidence':
        """Create Evidence from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def add_chunk(self, chunk_id: str) -> None:
        """Add a chunk ID to the evidence."""
        if chunk_id not in self.chunk_ids:
            self.chunk_ids.append(chunk_id)
    
    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a chunk ID from the evidence."""
        if chunk_id in self.chunk_ids:
            self.chunk_ids.remove(chunk_id)
    
    def update_scores(self, **kwargs) -> None:
        """Update evidence scores."""
        for key, value in kwargs.items():
            if hasattr(self.scores, key):
                setattr(self.scores, key, value)
        
        # Recalculate final score if individual scores changed
        if any(key in kwargs for key in ['relevance', 'freshness', 'source']):
            self._recalculate_final_score()
    
    def _recalculate_final_score(self) -> None:
        """Recalculate the final weighted score."""
        # Simple weighted average - can be customized based on requirements
        weights = {
            'relevance': 0.4,
            'freshness': 0.2,
            'source': 0.4
        }
        
        self.scores.final = sum(
            getattr(self.scores, score) * weight 
            for score, weight in weights.items()
        )
