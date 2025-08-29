"""
Hybrid Evidence Retrieval for TruthLens Phase 3
Combines trusted sources, vector search, and web search for comprehensive evidence retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

from .vector_search import VectorEvidenceRetriever
from .grounded_search import GroundedSearcher
from .trusted_sources import TrustedSourcesDatabase, TrustedSourcesAPI, SourceType, TrustedContent

logger = logging.getLogger(__name__)


@dataclass
class EvidenceRetrievalResult:
    """Result of evidence retrieval with metadata."""
    claim_id: str
    evidence_list: List[Dict[str, Any]]
    total_evidence: int
    trusted_evidence: int
    vector_evidence: int
    web_evidence: int
    fact_check_evidence: int = 0
    retrieval_time: float = 0.0
    freshness_score: float = 0.0
    reliability_score: float = 0.0
    errors: List[str] = None
    semantic_scores: List[float] = None
    clusters: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.semantic_scores is None:
            self.semantic_scores = []
        if self.clusters is None:
            self.clusters = []


class HybridEvidenceRetriever:
    """
    Hybrid evidence retrieval system for Phase 3.
    
    Features:
    - Trusted sources integration (PIB, WHO, RBI)
    - Vector-based semantic search
    - Web search augmentation
    - Freshness bias implementation
    - Reliability scoring
    - Content deduplication
    """
    
    def __init__(
        self,
        vector_retriever: Optional[VectorEvidenceRetriever] = None,
        grounded_retriever: Optional[GroundedSearcher] = None,
        trusted_database: Optional[TrustedSourcesDatabase] = None,
        max_results: int = 10,
        freshness_bias_days: int = 14
    ):
        """
        Initialize the hybrid evidence retriever.
        
        Args:
            vector_retriever: Vector search retriever
            grounded_retriever: Web search retriever
            trusted_database: Trusted sources database
            max_results: Maximum number of results to return
            freshness_bias_days: Days for freshness bias (recent content gets priority)
        """
        self.max_results = max_results
        self.freshness_bias_days = freshness_bias_days
        
        # Initialize retrievers
        self.vector_retriever = vector_retriever
        self.grounded_retriever = grounded_retriever
        self.trusted_database = trusted_database
        
        # Initialize if not provided
        if self.vector_retriever is None:
            self.vector_retriever = VectorEvidenceRetriever(use_gpu=False)
        
        if self.grounded_retriever is None:
            # Create a mock grounded searcher for testing
            from .grounded_search import make_default_client
            client = make_default_client()
            if client:
                self.grounded_retriever = GroundedSearcher(client)
            else:
                # Create a mock searcher for testing
                self.grounded_retriever = None
        
        if self.trusted_database is None:
            self.trusted_database = TrustedSourcesDatabase()
    
    def retrieve_evidence(self, claim: str, claim_id: str) -> EvidenceRetrievalResult:
        """
        Retrieve evidence for a claim using hybrid approach.
        
        Args:
            claim: The claim to find evidence for
            claim_id: Unique identifier for the claim
            
        Returns:
            EvidenceRetrievalResult with all retrieved evidence
        """
        start_time = time.time()
        errors = []
        
        try:
            # Step 1: Search trusted sources
            trusted_evidence = self._search_trusted_sources(claim)
            logger.info(f"Found {len(trusted_evidence)} trusted evidence items")
            
            # Step 2: Vector search
            vector_evidence = self._search_vector_database(claim)
            logger.info(f"Found {len(vector_evidence)} vector evidence items")
            
            # Step 3: Web search (if needed)
            web_evidence = []
            if len(trusted_evidence) + len(vector_evidence) < self.max_results // 2:
                web_evidence = self._search_web(claim)
                logger.info(f"Found {len(web_evidence)} web evidence items")
            
            # Step 4: Combine and rank evidence
            combined_evidence = self._combine_and_rank_evidence(
                trusted_evidence, vector_evidence, web_evidence
            )
            
            # Step 5: Calculate metrics
            retrieval_time = time.time() - start_time
            freshness_score = self._calculate_freshness_score(combined_evidence)
            reliability_score = self._calculate_reliability_score(combined_evidence)
            
            return EvidenceRetrievalResult(
                claim_id=claim_id,
                evidence_list=combined_evidence[:self.max_results],
                total_evidence=len(combined_evidence),
                trusted_evidence=len(trusted_evidence),
                vector_evidence=len(vector_evidence),
                web_evidence=len(web_evidence),
                retrieval_time=retrieval_time,
                freshness_score=freshness_score,
                reliability_score=reliability_score,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Evidence retrieval failed: {str(e)}")
            retrieval_time = time.time() - start_time
            
            return EvidenceRetrievalResult(
                claim_id=claim_id,
                evidence_list=[],
                total_evidence=0,
                trusted_evidence=0,
                vector_evidence=0,
                web_evidence=0,
                retrieval_time=retrieval_time,
                freshness_score=0.0,
                reliability_score=0.0,
                errors=errors
            )
    
    def _search_trusted_sources(self, claim: str) -> List[Dict[str, Any]]:
        """Search trusted sources for evidence including fact-checking sources."""
        try:
            evidence_list = []
            
            # 1. Get fact-checking results (highest priority)
            fact_check_content = self.trusted_database.get_fact_check_results(claim)
            for content in fact_check_content:
                source = self.trusted_database.sources.get(content.source_id)
                evidence_list.append({
                    "id": content.id,
                    "text": content.full_text,
                    "snippet": content.snippet,
                    "url": content.url,
                    "source": source.name if source else "Unknown",
                    "source_type": "fact_check",
                    "reliability_score": source.reliability_score if source else 0.9,
                    "relevance_score": content.relevance_score,
                    "published_date": content.published_date.isoformat(),
                    "freshness_score": self._calculate_content_freshness(content.published_date),
                    "tags": content.tags,
                    "fact_check_rating": content.fact_check_rating,
                    "verdict": content.verdict
                })
            
            # 2. Search other trusted sources
            trusted_content = self.trusted_database.search_content(
                query=claim,
                max_results=self.max_results // 2,
                min_relevance=0.3
            )
            
            for content in trusted_content:
                source = self.trusted_database.sources.get(content.source_id)
                evidence_list.append({
                    "id": content.id,
                    "text": content.full_text,
                    "snippet": content.snippet,
                    "url": content.url,
                    "source": source.name if source else "Unknown",
                    "source_type": "trusted",
                    "reliability_score": source.reliability_score if source else 0.8,
                    "relevance_score": content.relevance_score,
                    "published_date": content.published_date.isoformat(),
                    "freshness_score": self._calculate_content_freshness(content.published_date),
                    "tags": content.tags
                })
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"Trusted sources search failed: {e}")
            return []
    
    def _search_vector_database(self, claim: str) -> List[Dict[str, Any]]:
        """Search vector database for evidence using enhanced semantic search."""
        try:
            # Search in vector database with clustering
            vector_results = self.vector_retriever.search_evidence(
                query=claim,
                top_k=self.max_results // 2,
                apply_clustering=True
            )
            
            # Convert to evidence format
            evidence_list = []
            for result in vector_results:
                evidence_list.append({
                    "id": result.evidence.id,
                    "text": result.evidence.full_text,
                    "snippet": result.evidence.snippet,
                    "url": result.evidence.url,
                    "source": result.evidence.domain,
                    "source_type": "vector",
                    "reliability_score": 0.7,  # Default for vector search
                    "relevance_score": result.similarity_score,
                    "published_date": result.evidence.published_date.isoformat() if hasattr(result.evidence, 'published_date') else datetime.now().isoformat(),
                    "freshness_score": 0.5,  # Default freshness
                    "tags": []
                })
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _search_web(self, claim: str) -> List[Dict[str, Any]]:
        """Search web for additional evidence."""
        try:
            # Search web using grounded search
            if self.grounded_retriever is None:
                return []
            
            web_results = self.grounded_retriever.search(
                claim_text=claim,
                top_k=self.max_results // 4
            )
            
            # Convert to evidence format
            evidence_list = []
            for result in web_results:
                evidence_list.append({
                    "id": result.id,
                    "text": result.full_text,
                    "snippet": result.snippet,
                    "url": result.url,
                    "source": result.domain,
                    "source_type": "web",
                    "reliability_score": 0.5,  # Lower reliability for web search
                    "relevance_score": result.metadata.get("score", 0.5) if result.metadata else 0.5,
                    "published_date": result.published_at.isoformat() if result.published_at else datetime.now().isoformat(),
                    "freshness_score": 0.5,
                    "tags": []
                })
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    def _combine_and_rank_evidence(
        self, 
        trusted_evidence: List[Dict[str, Any]], 
        vector_evidence: List[Dict[str, Any]], 
        web_evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine and rank evidence from all sources."""
        # Combine all evidence
        all_evidence = trusted_evidence + vector_evidence + web_evidence
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_evidence = []
        for evidence in all_evidence:
            url = evidence.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_evidence.append(evidence)
            elif not url:
                unique_evidence.append(evidence)
        
        # Calculate combined scores
        for evidence in unique_evidence:
            # Combine relevance, reliability, and freshness
            relevance = evidence.get("relevance_score", 0.5)
            reliability = evidence.get("reliability_score", 0.5)
            freshness = evidence.get("freshness_score", 0.5)
            
            # Weighted combination
            combined_score = (
                0.4 * relevance + 
                0.4 * reliability + 
                0.2 * freshness
            )
            
            # Boost trusted sources
            if evidence.get("source_type") == "trusted":
                combined_score *= 1.2
            
            evidence["combined_score"] = combined_score
        
        # Sort by combined score
        unique_evidence.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return unique_evidence
    
    def _calculate_content_freshness(self, published_date: datetime) -> float:
        """Calculate freshness score for content."""
        days_old = (datetime.now() - published_date).days
        
        if days_old <= 1:
            return 1.0
        elif days_old <= 7:
            return 0.9
        elif days_old <= 30:
            return 0.7
        elif days_old <= 90:
            return 0.5
        else:
            return 0.3
    
    def _calculate_freshness_score(self, evidence_list: List[Dict[str, Any]]) -> float:
        """Calculate overall freshness score for evidence set."""
        if not evidence_list:
            return 0.0
        
        freshness_scores = [evidence.get("freshness_score", 0.5) for evidence in evidence_list]
        return sum(freshness_scores) / len(freshness_scores)
    
    def _calculate_reliability_score(self, evidence_list: List[Dict[str, Any]]) -> float:
        """Calculate overall reliability score for evidence set."""
        if not evidence_list:
            return 0.0
        
        reliability_scores = [evidence.get("reliability_score", 0.5) for evidence in evidence_list]
        return sum(reliability_scores) / len(reliability_scores)
    
    def update_trusted_sources(self):
        """Update content from all trusted sources."""
        try:
            api = TrustedSourcesAPI(self.trusted_database)
            api.update_all_sources()
            logger.info("Updated trusted sources successfully")
        except Exception as e:
            logger.error(f"Failed to update trusted sources: {e}")
    
    def get_evidence_statistics(self, result: EvidenceRetrievalResult) -> Dict[str, Any]:
        """Get detailed statistics about evidence retrieval."""
        return {
            "total_evidence": result.total_evidence,
            "trusted_evidence": result.trusted_evidence,
            "vector_evidence": result.vector_evidence,
            "web_evidence": result.web_evidence,
            "retrieval_time": result.retrieval_time,
            "freshness_score": result.freshness_score,
            "reliability_score": result.reliability_score,
            "average_relevance": sum(e.get("relevance_score", 0) for e in result.evidence_list) / max(1, len(result.evidence_list)),
            "source_distribution": {
                "trusted": len([e for e in result.evidence_list if e.get("source_type") == "trusted"]),
                "vector": len([e for e in result.evidence_list if e.get("source_type") == "vector"]),
                "web": len([e for e in result.evidence_list if e.get("source_type") == "web"])
            },
            "errors": result.errors
        }


def get_hybrid_evidence_retriever() -> HybridEvidenceRetriever:
    """Get or create the hybrid evidence retriever."""
    return HybridEvidenceRetriever()


def retrieve_evidence_for_claim(claim: str, claim_id: str, max_results: int = 10) -> EvidenceRetrievalResult:
    """
    Convenience function to retrieve evidence for a claim.
    
    Args:
        claim: The claim to find evidence for
        claim_id: Unique identifier for the claim
        max_results: Maximum number of results to return
        
    Returns:
        EvidenceRetrievalResult with retrieved evidence
    """
    retriever = HybridEvidenceRetriever(max_results=max_results)
    return retriever.retrieve_evidence(claim, claim_id)
