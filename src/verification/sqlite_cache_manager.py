#!/usr/bin/env python3
"""
SQLite Cache Manager for TruthLens Verification System
Provides scalable and queryable caching with hash-based keys for consistency.
"""

import json
import logging
import hashlib
import sqlite3
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
    evidence_sentences: List[str]
    highlighted_sentences: List[Dict[str, Any]]
    ensemble_votes: Dict[str, Any]
    evidence_snippets: List[Dict[str, Any]]
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
    scientific_consensus_override: Optional[str]
    no_evidence_case: bool
    stance_results: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    evidence_snippets: List[Dict[str, Any]]
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
    user_type: str  # trusted_fact_checker, casual_user, expert
    timestamp: datetime
    evidence_used: List[str]


class SQLiteCacheManager:
    """
    SQLite-based cache manager for TruthLens verification system.
    
    Features:
    - SQLite database for scalable and queryable caching
    - Hash-based cache keys for consistency
    - User feedback with weighted importance
    - Automatic cache expiration and cleanup
    - Evidence snippets and explainability
    """
    
    def __init__(self, db_path: str = "data/cache/truthlens_cache.db", max_cache_age_days: int = 30):
        """
        Initialize the SQLite cache manager.
        
        Args:
            db_path: Path to SQLite database file
            max_cache_age_days: Maximum age of cache entries in days
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_age = timedelta(days=max_cache_age_days)
        
        # Initialize database
        self._init_database()
        
        # Clean up old cache entries
        self._cleanup_old_cache()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create stance cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stance_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        claim_hash TEXT NOT NULL,
                        evidence_hash TEXT NOT NULL,
                        cache_key TEXT UNIQUE NOT NULL,
                        stance TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        probabilities TEXT NOT NULL,
                        rule_based_override TEXT,
                        evidence_sentences TEXT NOT NULL,
                        highlighted_sentences TEXT NOT NULL,
                        ensemble_votes TEXT NOT NULL,
                        evidence_snippets TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create verdict cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS verdict_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        claim_hash TEXT UNIQUE NOT NULL,
                        verdict TEXT NOT NULL,
                        confidence_badge TEXT NOT NULL,
                        reasoning TEXT NOT NULL,
                        weighted_votes TEXT NOT NULL,
                        fact_check_override TEXT,
                        scientific_consensus_override TEXT,
                        no_evidence_case BOOLEAN NOT NULL,
                        stance_results TEXT NOT NULL,
                        citations TEXT NOT NULL,
                        evidence_snippets TEXT NOT NULL,
                        evidence_count INTEGER NOT NULL,
                        processing_time REAL NOT NULL,
                        timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create API cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        api_name TEXT NOT NULL,
                        query_hash TEXT NOT NULL,
                        cache_key TEXT UNIQUE NOT NULL,
                        result_data TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create user feedback table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        claim_hash TEXT NOT NULL,
                        user_label TEXT NOT NULL,
                        user_confidence REAL NOT NULL,
                        feedback_reason TEXT NOT NULL,
                        user_type TEXT NOT NULL,
                        evidence_used TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_stance_cache_key ON stance_cache(cache_key)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_stance_cache_timestamp ON stance_cache(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_verdict_cache_hash ON verdict_cache(claim_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_verdict_cache_timestamp ON verdict_cache(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_cache_key ON api_cache(cache_key)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_cache_timestamp ON api_cache(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_hash ON user_feedback(claim_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON user_feedback(timestamp)")
                
                conn.commit()
                logger.info(f"SQLite cache database initialized at: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
    
    def _generate_hash(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.lower().encode()).hexdigest()
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from multiple arguments."""
        combined = "_".join(str(arg) for arg in args)
        return self._generate_hash(combined)
    
    def _json_serialize(self, obj: Any) -> str:
        """Serialize object to JSON string."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.dumps(obj, default=str, ensure_ascii=False)
    
    def _json_deserialize(self, json_str: str) -> Any:
        """Deserialize JSON string to object."""
        try:
            return json.loads(json_str)
        except:
            return json_str
    
    def _cleanup_old_cache(self):
        """Remove old cache entries."""
        cutoff_time = datetime.now() - self.max_cache_age
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean stance cache
                cursor.execute("DELETE FROM stance_cache WHERE timestamp < ?", (cutoff_time.isoformat(),))
                stance_deleted = cursor.rowcount
                
                # Clean verdict cache
                cursor.execute("DELETE FROM verdict_cache WHERE timestamp < ?", (cutoff_time.isoformat(),))
                verdict_deleted = cursor.rowcount
                
                # Clean API cache
                cursor.execute("DELETE FROM api_cache WHERE timestamp < ?", (cutoff_time.isoformat(),))
                api_deleted = cursor.rowcount
                
                # Clean user feedback (keep longer)
                feedback_cutoff = datetime.now() - timedelta(days=90)  # Keep feedback for 90 days
                cursor.execute("DELETE FROM user_feedback WHERE timestamp < ?", (feedback_cutoff.isoformat(),))
                feedback_deleted = cursor.rowcount
                
                conn.commit()
                
                if stance_deleted or verdict_deleted or api_deleted or feedback_deleted:
                    logger.info(f"Cleaned up {stance_deleted} stance, {verdict_deleted} verdict, {api_deleted} API, {feedback_deleted} feedback cache entries")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old cache: {e}")
    
    def get_cached_stance(self, claim: str, evidence_text: str) -> Optional[CachedStanceResult]:
        """Get cached stance result if available."""
        claim_hash = self._generate_hash(claim)
        evidence_hash = self._generate_hash(evidence_text)
        cache_key = self._generate_cache_key(claim_hash, evidence_hash)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM stance_cache 
                    WHERE cache_key = ? AND timestamp > ?
                """, (cache_key, (datetime.now() - self.max_cache_age).isoformat()))
                
                row = cursor.fetchone()
                if row:
                    logger.debug(f"Cache hit for stance: {claim[:50]}...")
                    return self._row_to_stance_result(row)
                
        except Exception as e:
            logger.error(f"Error retrieving cached stance: {e}")
        
        return None
    
    def cache_stance_result(self, claim: str, evidence_text: str, stance_result: Any, 
                          model_version: str = "facebook/bart-large-mnli") -> CachedStanceResult:
        """Cache stance detection result with explainability."""
        claim_hash = self._generate_hash(claim)
        evidence_hash = self._generate_hash(evidence_text)
        cache_key = self._generate_cache_key(claim_hash, evidence_hash)
        
        # Extract evidence snippets
        evidence_snippets = stance_result.evidence_snippets if hasattr(stance_result, 'evidence_snippets') else []
        
        # Create cached result
        cached_result = CachedStanceResult(
            claim_hash=claim_hash,
            evidence_hash=evidence_hash,
            stance=stance_result.label,
            confidence_score=stance_result.confidence_score,
            probabilities=stance_result.probabilities,
            rule_based_override=stance_result.rule_based_override,
            evidence_sentences=stance_result.evidence_sentences if hasattr(stance_result, 'evidence_sentences') else [],
            highlighted_sentences=stance_result.highlighted_sentences if hasattr(stance_result, 'highlighted_sentences') else [],
            ensemble_votes=stance_result.ensemble_votes if hasattr(stance_result, 'ensemble_votes') else {},
            evidence_snippets=evidence_snippets,
            timestamp=datetime.now(),
            model_version=model_version
        )
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO stance_cache 
                    (claim_hash, evidence_hash, cache_key, stance, confidence_score, 
                     probabilities, rule_based_override, evidence_sentences, 
                     highlighted_sentences, ensemble_votes, evidence_snippets, 
                     model_version, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    claim_hash, evidence_hash, cache_key, cached_result.stance,
                    cached_result.confidence_score, self._json_serialize(cached_result.probabilities),
                    cached_result.rule_based_override, self._json_serialize(cached_result.evidence_sentences),
                    self._json_serialize(cached_result.highlighted_sentences),
                    self._json_serialize(cached_result.ensemble_votes),
                    self._json_serialize(cached_result.evidence_snippets),
                    model_version, cached_result.timestamp.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error caching stance result: {e}")
        
        logger.debug(f"Cached stance result for: {claim[:50]}...")
        return cached_result
    
    def get_cached_verdict(self, claim: str) -> Optional[CachedVerdictResult]:
        """Get cached verdict result if available."""
        claim_hash = self._generate_hash(claim)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM verdict_cache 
                    WHERE claim_hash = ? AND timestamp > ?
                """, (claim_hash, (datetime.now() - self.max_cache_age).isoformat()))
                
                row = cursor.fetchone()
                if row:
                    logger.debug(f"Cache hit for verdict: {claim[:50]}...")
                    return self._row_to_verdict_result(row)
                
        except Exception as e:
            logger.error(f"Error retrieving cached verdict: {e}")
        
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
            scientific_consensus_override=verdict_result.get("scientific_consensus_override"),
            no_evidence_case=verdict_result.get("no_evidence_case", False),
            stance_results=verdict_result["stance_results"],
            citations=verdict_result["citations"],
            evidence_snippets=verdict_result.get("evidence_snippets", []),
            evidence_count=verdict_result["evidence_count"],
            timestamp=datetime.now(),
            processing_time=processing_time
        )
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO verdict_cache 
                    (claim_hash, verdict, confidence_badge, reasoning, weighted_votes,
                     fact_check_override, scientific_consensus_override, no_evidence_case,
                     stance_results, citations, evidence_snippets, evidence_count,
                     processing_time, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    claim_hash, cached_result.verdict, cached_result.confidence_badge,
                    cached_result.reasoning, self._json_serialize(cached_result.weighted_votes),
                    cached_result.fact_check_override, cached_result.scientific_consensus_override,
                    cached_result.no_evidence_case, self._json_serialize(cached_result.stance_results),
                    self._json_serialize(cached_result.citations),
                    self._json_serialize(cached_result.evidence_snippets),
                    cached_result.evidence_count, cached_result.processing_time,
                    cached_result.timestamp.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error caching verdict result: {e}")
        
        logger.debug(f"Cached verdict result for: {claim[:50]}...")
        return cached_result
    
    def get_cached_api_result(self, api_name: str, query: str) -> Optional[Any]:
        """Get cached API result if available."""
        query_hash = self._generate_hash(query)
        cache_key = self._generate_cache_key(api_name, query_hash)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT result_data FROM api_cache 
                    WHERE cache_key = ? AND timestamp > ?
                """, (cache_key, (datetime.now() - self.max_cache_age).isoformat()))
                
                row = cursor.fetchone()
                if row:
                    logger.debug(f"Cache hit for API {api_name}: {query[:50]}...")
                    return self._json_deserialize(row[0])
                
        except Exception as e:
            logger.error(f"Error retrieving cached API result: {e}")
        
        return None
    
    def cache_api_result(self, api_name: str, query: str, result: Any):
        """Cache API result."""
        query_hash = self._generate_hash(query)
        cache_key = self._generate_cache_key(api_name, query_hash)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO api_cache 
                    (api_name, query_hash, cache_key, result_data, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    api_name, query_hash, cache_key,
                    self._json_serialize(result), datetime.now().isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error caching API result: {e}")
        
        logger.debug(f"Cached API result for {api_name}: {query[:50]}...")
    
    def add_user_feedback(self, claim: str, user_label: str, user_confidence: float, 
                         feedback_reason: str, evidence_used: List[str], 
                         user_type: str = "casual_user"):
        """Add user feedback for claim verification with weighted importance."""
        claim_hash = self._generate_hash(claim)
        
        # Determine feedback weight based on user type
        user_weights = {
            "trusted_fact_checker": 1.0,
            "expert": 0.8,
            "casual_user": 0.5
        }
        feedback_weight = user_weights.get(user_type, 0.5)
        
        feedback = UserFeedback(
            claim_hash=claim_hash,
            user_label=user_label,
            user_confidence=user_confidence * feedback_weight,  # Apply weight
            feedback_reason=feedback_reason,
            user_type=user_type,
            timestamp=datetime.now(),
            evidence_used=evidence_used
        )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_feedback 
                    (claim_hash, user_label, user_confidence, feedback_reason, 
                     user_type, evidence_used, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    claim_hash, feedback.user_label, feedback.user_confidence,
                    feedback.feedback_reason, feedback.user_type,
                    self._json_serialize(feedback.evidence_used),
                    feedback.timestamp.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error adding user feedback: {e}")
        
        logger.info(f"Added user feedback for claim: {claim[:50]}... (weight: {feedback_weight})")
    
    def get_user_feedback_for_claim(self, claim: str) -> List[UserFeedback]:
        """Get user feedback for a specific claim."""
        claim_hash = self._generate_hash(claim)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM user_feedback 
                    WHERE claim_hash = ?
                    ORDER BY timestamp DESC
                """, (claim_hash,))
                
                rows = cursor.fetchall()
                feedback_list = []
                
                for row in rows:
                    feedback = UserFeedback(
                        claim_hash=row[1],
                        user_label=row[2],
                        user_confidence=row[3],
                        feedback_reason=row[4],
                        user_type=row[5],
                        timestamp=datetime.fromisoformat(row[7]),
                        evidence_used=self._json_deserialize(row[6])
                    )
                    feedback_list.append(feedback)
                
                return feedback_list
                
        except Exception as e:
            logger.error(f"Error retrieving user feedback: {e}")
            return []
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count entries in each table
                cursor.execute("SELECT COUNT(*) FROM stance_cache")
                stance_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM verdict_cache")
                verdict_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM api_cache")
                api_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM user_feedback")
                feedback_count = cursor.fetchone()[0]
                
                # Get database size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    "stance_cache_size": stance_count,
                    "verdict_cache_size": verdict_count,
                    "api_cache_size": api_count,
                    "user_feedback_count": feedback_count,
                    "database_path": str(self.db_path),
                    "database_size_mb": round(db_size / (1024 * 1024), 2),
                    "max_cache_age_days": self.max_cache_age.days
                }
                
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}
    
    def _row_to_stance_result(self, row) -> CachedStanceResult:
        """Convert database row to CachedStanceResult."""
        return CachedStanceResult(
            claim_hash=row[1],
            evidence_hash=row[2],
            stance=row[4],
            confidence_score=row[5],
            probabilities=self._json_deserialize(row[6]),
            rule_based_override=row[7],
            evidence_sentences=self._json_deserialize(row[8]),
            highlighted_sentences=self._json_deserialize(row[9]),
            ensemble_votes=self._json_deserialize(row[10]),
            evidence_snippets=self._json_deserialize(row[11]),
            timestamp=datetime.fromisoformat(row[13]),
            model_version=row[12]
        )
    
    def _row_to_verdict_result(self, row) -> CachedVerdictResult:
        """Convert database row to CachedVerdictResult."""
        return CachedVerdictResult(
            claim_hash=row[1],
            verdict=row[2],
            confidence_badge=row[3],
            reasoning=row[4],
            weighted_votes=self._json_deserialize(row[5]),
            fact_check_override=row[6],
            scientific_consensus_override=row[7],
            no_evidence_case=bool(row[8]),
            stance_results=self._json_deserialize(row[9]),
            citations=self._json_deserialize(row[10]),
            evidence_snippets=self._json_deserialize(row[11]),
            evidence_count=row[12],
            timestamp=datetime.fromisoformat(row[14]),
            processing_time=row[13]
        )
    
    def clear_cache(self):
        """Clear all cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM stance_cache")
                cursor.execute("DELETE FROM verdict_cache")
                cursor.execute("DELETE FROM api_cache")
                cursor.execute("DELETE FROM user_feedback")
                conn.commit()
                logger.info("All cache cleared")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_explainability_report(self, claim: str, evidence_text: str) -> Dict[str, Any]:
        """Generate explainability report for a claim-evidence pair."""
        cached_stance = self.get_cached_stance(claim, evidence_text)
        
        if not cached_stance:
            return {"error": "No cached stance result found"}
        
        # Analyze highlighted sentences
        supporting_sentences = [
            s for s in cached_stance.highlighted_sentences 
            if s.get("sentence_type") == "supporting"
        ]
        
        contradicting_sentences = [
            s for s in cached_stance.highlighted_sentences 
            if s.get("sentence_type") == "contradicting"
        ]
        
        relevant_sentences = [
            s for s in cached_stance.highlighted_sentences 
            if s.get("sentence_type") == "relevant"
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
            "ensemble_votes": cached_stance.ensemble_votes,
            "evidence_snippets": cached_stance.evidence_snippets,
            "model_version": cached_stance.model_version,
            "timestamp": cached_stance.timestamp.isoformat()
        }
