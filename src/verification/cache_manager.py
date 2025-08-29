#!/usr/bin/env python3
"""
Cache Manager for TruthLens Verification System
Provides caching for API results, stance detection, and verdicts with explainability features.
"""

import json
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


@dataclass
class CachedStanceResult:
    """Cached stance detection result with explainability."""
    claim_hash: str
    evidence_hash: str
    stance: str
    confidence_score: float
    probabilities: Dict[str, float]
    rule_based_override: Optional[str]
    evidence_sentences: List[str]  # Sentences used for stance detection
    highlighted_sentences: List[Dict[str, Any]]  # Sentences with highlighting info
    timestamp: datetime
    model_version: str


@dataclass
class CachedVerdictResult:
    """Cached verdict result with full explainability."""
    claim_hash: str
    verdict: str
    confidence_badge: str
    reasoning: str
    weighted_votes: Dict[str, float]
    fact_check_override: Optional[str]
    stance_results: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    evidence_count: int
    timestamp: datetime
    processing_time: float


@dataclass
class UserFeedback:
    """User feedback for claims to improve stance detection."""
    claim_hash: str
    user_label: str  # SUPPORTED, REFUTED, NOT ENOUGH INFO
    user_confidence: float
    feedback_reason: str
    timestamp: datetime
    evidence_used: List[str]


class CacheManager:
    """
    Comprehensive cache manager for TruthLens verification system.
    
    Features:
    - Caching for API results, stance detection, and verdicts
    - Explainability with highlighted sentences
    - User feedback collection for model improvement
    - Automatic cache expiration and cleanup
    """
    
    def __init__(self, cache_dir: str = "data/cache", max_cache_age_days: int = 30):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_cache_age_days: Maximum age of cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_age = timedelta(days=max_cache_age_days)
        
        # Cache file paths
        self.stance_cache_file = self.cache_dir / "stance_cache.pkl"
        self.verdict_cache_file = self.cache_dir / "verdict_cache.pkl"
        self.api_cache_file = self.cache_dir / "api_cache.pkl"
        self.feedback_file = self.cache_dir / "user_feedback.json"
        
        # In-memory caches for faster access
        self.stance_cache: Dict[str, CachedStanceResult] = {}
        self.verdict_cache: Dict[str, CachedVerdictResult] = {}
        self.api_cache: Dict[str, Any] = {}
        self.user_feedback: List[UserFeedback] = []
        
        # Load existing cache
        self._load_cache()
        
        # Clean up old cache entries
        self._cleanup_old_cache()
    
    def _load_cache(self):
        """Load existing cache from files."""
        try:
            # Load stance cache
            if self.stance_cache_file.exists():
                with open(self.stance_cache_file, 'rb') as f:
                    self.stance_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.stance_cache)} stance cache entries")
            
            # Load verdict cache
            if self.verdict_cache_file.exists():
                with open(self.verdict_cache_file, 'rb') as f:
                    self.verdict_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.verdict_cache)} verdict cache entries")
            
            # Load API cache
            if self.api_cache_file.exists():
                with open(self.api_cache_file, 'rb') as f:
                    self.api_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.api_cache)} API cache entries")
            
            # Load user feedback
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
                    self.user_feedback = [
                        UserFeedback(**item) for item in feedback_data
                    ]
                logger.info(f"Loaded {len(self.user_feedback)} user feedback entries")
                
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save cache to files."""
        try:
            # Save stance cache
            with open(self.stance_cache_file, 'wb') as f:
                pickle.dump(self.stance_cache, f)
            
            # Save verdict cache
            with open(self.verdict_cache_file, 'wb') as f:
                pickle.dump(self.verdict_cache, f)
            
            # Save API cache
            with open(self.api_cache_file, 'wb') as f:
                pickle.dump(self.api_cache, f)
            
            # Save user feedback
            feedback_data = [asdict(feedback) for feedback in self.user_feedback]
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _cleanup_old_cache(self):
        """Remove old cache entries."""
        cutoff_time = datetime.now() - self.max_cache_age
        
        # Clean stance cache
        old_stance_keys = [
            key for key, value in self.stance_cache.items()
            if value.timestamp < cutoff_time
        ]
        for key in old_stance_keys:
            del self.stance_cache[key]
        
        # Clean verdict cache
        old_verdict_keys = [
            key for key, value in self.verdict_cache.items()
            if value.timestamp < cutoff_time
        ]
        for key in old_verdict_keys:
            del self.verdict_cache[key]
        
        # Clean API cache
        old_api_keys = [
            key for key, value in self.api_cache.items()
            if isinstance(value, dict) and value.get('timestamp', 0) < cutoff_time.timestamp()
        ]
        for key in old_api_keys:
            del self.api_cache[key]
        
        if old_stance_keys or old_verdict_keys or old_api_keys:
            logger.info(f"Cleaned up {len(old_stance_keys)} stance, {len(old_verdict_keys)} verdict, {len(old_api_keys)} API cache entries")
            self._save_cache()
    
    def _generate_hash(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.lower().encode()).hexdigest()
    
    def _extract_evidence_sentences(self, evidence_text: str, claim: str) -> List[Dict[str, Any]]:
        """
        Extract and highlight sentences from evidence that are relevant to the claim.
        
        Args:
            evidence_text: The evidence text
            claim: The claim being verified
            
        Returns:
            List of highlighted sentences with relevance information
        """
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', evidence_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract claim keywords
        claim_keywords = set(re.findall(r'\b\w+\b', claim.lower()))
        claim_keywords = {kw for kw in claim_keywords if len(kw) > 3}  # Filter short words
        
        highlighted_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Calculate relevance score
            keyword_matches = sum(1 for kw in claim_keywords if kw in sentence_lower)
            relevance_score = keyword_matches / len(claim_keywords) if claim_keywords else 0
            
            # Check for negation patterns
            negation_patterns = [
                r'\b(not true|false|fake|hoax|debunked|disproven|refuted|denied|incorrect|wrong)\b',
                r'\b(no evidence|no proof|unfounded|baseless|misleading|misinformation)\b',
                r'\b(does not|did not|do not|cannot|could not|would not|should not)\b'
            ]
            
            has_negation = any(re.search(pattern, sentence_lower) for pattern in negation_patterns)
            
            # Check for support patterns
            support_patterns = [
                r'\b(confirmed|verified|true|accurate|correct|proven|established)\b',
                r'\b(evidence shows|studies confirm|research indicates)\b'
            ]
            
            has_support = any(re.search(pattern, sentence_lower) for pattern in support_patterns)
            
            # Determine sentence type
            if has_negation:
                sentence_type = "contradicting"
            elif has_support:
                sentence_type = "supporting"
            elif relevance_score > 0.3:
                sentence_type = "relevant"
            else:
                sentence_type = "neutral"
            
            highlighted_sentences.append({
                "sentence": sentence,
                "relevance_score": relevance_score,
                "sentence_type": sentence_type,
                "has_negation": has_negation,
                "has_support": has_support,
                "keyword_matches": keyword_matches
            })
        
        return highlighted_sentences
    
    def get_cached_stance(self, claim: str, evidence_text: str) -> Optional[CachedStanceResult]:
        """Get cached stance result if available."""
        claim_hash = self._generate_hash(claim)
        evidence_hash = self._generate_hash(evidence_text)
        cache_key = f"{claim_hash}_{evidence_hash}"
        
        if cache_key in self.stance_cache:
            cached_result = self.stance_cache[cache_key]
            if datetime.now() - cached_result.timestamp < self.max_cache_age:
                logger.debug(f"Cache hit for stance: {claim[:50]}...")
                return cached_result
        
        return None
    
    def cache_stance_result(self, claim: str, evidence_text: str, stance_result: Any, 
                          model_version: str = "facebook/bart-large-mnli") -> CachedStanceResult:
        """Cache stance detection result with explainability."""
        claim_hash = self._generate_hash(claim)
        evidence_hash = self._generate_hash(evidence_text)
        cache_key = f"{claim_hash}_{evidence_hash}"
        
        # Extract highlighted sentences
        highlighted_sentences = self._extract_evidence_sentences(evidence_text, claim)
        
        # Create cached result
        cached_result = CachedStanceResult(
            claim_hash=claim_hash,
            evidence_hash=evidence_hash,
            stance=stance_result.label,
            confidence_score=stance_result.confidence_score,
            probabilities=stance_result.probabilities,
            rule_based_override=stance_result.rule_based_override,
            evidence_sentences=[s["sentence"] for s in highlighted_sentences],
            highlighted_sentences=highlighted_sentences,
            timestamp=datetime.now(),
            model_version=model_version
        )
        
        # Store in cache
        self.stance_cache[cache_key] = cached_result
        self._save_cache()
        
        logger.debug(f"Cached stance result for: {claim[:50]}...")
        return cached_result
    
    def get_cached_verdict(self, claim: str) -> Optional[CachedVerdictResult]:
        """Get cached verdict result if available."""
        claim_hash = self._generate_hash(claim)
        
        if claim_hash in self.verdict_cache:
            cached_result = self.verdict_cache[claim_hash]
            if datetime.now() - cached_result.timestamp < self.max_cache_age:
                logger.debug(f"Cache hit for verdict: {claim[:50]}...")
                return cached_result
        
        return None
    
    def cache_verdict_result(self, claim: str, verdict_result: Dict[str, Any], 
                           processing_time: float) -> CachedVerdictResult:
        """Cache verdict result."""
        claim_hash = self._generate_hash(claim)
        
        # Create cached result
        cached_result = CachedVerdictResult(
            claim_hash=claim_hash,
            verdict=verdict_result["verdict"],
            confidence_badge=verdict_result["confidence_badge"],
            reasoning=verdict_result["reasoning"],
            weighted_votes=verdict_result["weighted_votes"],
            fact_check_override=verdict_result.get("fact_check_override"),
            stance_results=verdict_result["stance_results"],
            citations=verdict_result["citations"],
            evidence_count=verdict_result["evidence_count"],
            timestamp=datetime.now(),
            processing_time=processing_time
        )
        
        # Store in cache
        self.verdict_cache[claim_hash] = cached_result
        self._save_cache()
        
        logger.debug(f"Cached verdict result for: {claim[:50]}...")
        return cached_result
    
    def get_cached_api_result(self, api_name: str, query: str) -> Optional[Any]:
        """Get cached API result if available."""
        cache_key = f"{api_name}_{self._generate_hash(query)}"
        
        if cache_key in self.api_cache:
            cached_result = self.api_cache[cache_key]
            if isinstance(cached_result, dict) and 'timestamp' in cached_result:
                cache_time = datetime.fromtimestamp(cached_result['timestamp'])
                if datetime.now() - cache_time < self.max_cache_age:
                    logger.debug(f"Cache hit for API {api_name}: {query[:50]}...")
                    return cached_result['data']
        
        return None
    
    def cache_api_result(self, api_name: str, query: str, result: Any):
        """Cache API result."""
        cache_key = f"{api_name}_{self._generate_hash(query)}"
        
        self.api_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.now().timestamp()
        }
        self._save_cache()
        
        logger.debug(f"Cached API result for {api_name}: {query[:50]}...")
    
    def add_user_feedback(self, claim: str, user_label: str, user_confidence: float, 
                         feedback_reason: str, evidence_used: List[str]):
        """Add user feedback for claim verification."""
        claim_hash = self._generate_hash(claim)
        
        feedback = UserFeedback(
            claim_hash=claim_hash,
            user_label=user_label,
            user_confidence=user_confidence,
            feedback_reason=feedback_reason,
            timestamp=datetime.now(),
            evidence_used=evidence_used
        )
        
        self.user_feedback.append(feedback)
        self._save_cache()
        
        logger.info(f"Added user feedback for claim: {claim[:50]}...")
    
    def get_explainability_report(self, claim: str, evidence_text: str) -> Dict[str, Any]:
        """Generate explainability report for a claim-evidence pair."""
        cached_stance = self.get_cached_stance(claim, evidence_text)
        
        if not cached_stance:
            return {"error": "No cached stance result found"}
        
        # Analyze highlighted sentences
        supporting_sentences = [
            s for s in cached_stance.highlighted_sentences 
            if s["sentence_type"] == "supporting"
        ]
        
        contradicting_sentences = [
            s for s in cached_stance.highlighted_sentences 
            if s["sentence_type"] == "contradicting"
        ]
        
        relevant_sentences = [
            s for s in cached_stance.highlighted_sentences 
            if s["sentence_type"] == "relevant"
        ]
        
        return {
            "claim": claim,
            "evidence_preview": evidence_text[:200] + "...",
            "stance": cached_stance.stance,
            "confidence": cached_stance.confidence_score,
            "rule_based_override": cached_stance.rule_based_override,
            "supporting_sentences": supporting_sentences,
            "contradicting_sentences": contradicting_sentences,
            "relevant_sentences": relevant_sentences,
            "all_highlighted_sentences": cached_stance.highlighted_sentences,
            "model_version": cached_stance.model_version,
            "timestamp": cached_stance.timestamp.isoformat()
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "stance_cache_size": len(self.stance_cache),
            "verdict_cache_size": len(self.verdict_cache),
            "api_cache_size": len(self.api_cache),
            "user_feedback_count": len(self.user_feedback),
            "cache_directory": str(self.cache_dir),
            "max_cache_age_days": self.max_cache_age.days
        }
    
    def clear_cache(self):
        """Clear all cache."""
        self.stance_cache.clear()
        self.verdict_cache.clear()
        self.api_cache.clear()
        self.user_feedback.clear()
        self._save_cache()
        logger.info("All cache cleared")
