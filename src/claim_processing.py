"""
Phase 2 - Claim Extraction & Ranking Integration
Comprehensive integration of claim extraction, atomicization, and ranking
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from extractor.pipeline import process_text, AtomicClaimJSON
from extractor.claim_detector import is_claim
from extractor.claim_extractor import extract_claim_spans
from extractor.atomicizer import to_atomic
from extractor.context import analyze_context
from extractor.ranker import score_claim


@dataclass
class ClaimProcessingResult:
    """Result of claim processing with all metadata."""
    claim_id: str
    original_text: str
    atomic_claims: List[AtomicClaimJSON]
    total_claims: int
    checkworthy_claims: int
    average_score: float
    processing_time: float
    errors: List[str]


class ClaimProcessor:
    """
    Comprehensive claim processing pipeline for Phase 2.
    
    Features:
    - Text preprocessing and normalization
    - Claim detection and extraction
    - Atomic claim decomposition
    - Checkworthiness scoring
    - Context analysis
    - Quality filtering
    """
    
    def __init__(self, min_checkworthiness: float = 0.3):
        """
        Initialize the claim processor.
        
        Args:
            min_checkworthiness: Minimum score for claims to be considered checkworthy
        """
        self.min_checkworthiness = min_checkworthiness
    
    def process_claims(self, text: str) -> ClaimProcessingResult:
        """
        Process text and extract checkworthy claims.
        
        Args:
            text: Input text to process
            
        Returns:
            ClaimProcessingResult with all extracted claims and metadata
        """
        import time
        start_time = time.time()
        
        claim_id = str(uuid.uuid4())
        errors = []
        
        try:
            # Process text into atomic claims
            atomic_claims = process_text(text)
            
            # Filter by checkworthiness
            checkworthy_claims = [
                claim for claim in atomic_claims 
                if claim["checkworthiness"] >= self.min_checkworthiness
            ]
            
            # Calculate statistics
            total_claims = len(atomic_claims)
            checkworthy_count = len(checkworthy_claims)
            avg_score = sum(c["checkworthiness"] for c in atomic_claims) / max(1, total_claims)
            
            processing_time = time.time() - start_time
            
            return ClaimProcessingResult(
                claim_id=claim_id,
                original_text=text,
                atomic_claims=checkworthy_claims,  # Only return checkworthy claims
                total_claims=total_claims,
                checkworthy_claims=checkworthy_count,
                average_score=avg_score,
                processing_time=processing_time,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Processing failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return ClaimProcessingResult(
                claim_id=claim_id,
                original_text=text,
                atomic_claims=[],
                total_claims=0,
                checkworthy_claims=0,
                average_score=0.0,
                processing_time=processing_time,
                errors=errors
            )
    
    def extract_single_claim(self, text: str) -> Optional[AtomicClaimJSON]:
        """
        Extract a single claim from text.
        
        Args:
            text: Input text
            
        Returns:
            AtomicClaimJSON if a checkworthy claim is found, None otherwise
        """
        try:
            # Check if text is a claim
            label, prob = is_claim(text)
            if not label:
                return None
            
            # Extract claim spans
            spans = extract_claim_spans(text)
            if not spans:
                spans = [{"text": text, "start": 0, "end": len(text), "conf": prob}]
            
            # Process first span
            span = spans[0]
            atomic_claims = to_atomic(span.get("text", ""), None) or []
            
            if not atomic_claims:
                atomic_claims = [{"text": span.get("text", ""), "subject": "", "predicate": "", "object": ""}]
            
            # Take first atomic claim
            ac = atomic_claims[0]
            claim_text = ac.get("text", "").strip()
            
            if not claim_text:
                return None
            
            # Analyze context and score
            ctx = analyze_context(claim_text, text)
            score = score_claim(claim_text)
            
            # Only return if checkworthy
            if score >= self.min_checkworthiness:
                return AtomicClaimJSON(
                    id=str(uuid.uuid4()),
                    text=claim_text,
                    subject=ac.get("subject", "").strip(),
                    predicate=ac.get("predicate", "").strip(),
                    object=ac.get("object", "").strip(),
                    context=ctx,
                    checkworthiness=float(max(0.0, min(1.0, score)))
                )
            
            return None
            
        except Exception:
            return None
    
    def batch_process(self, texts: List[str]) -> List[ClaimProcessingResult]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of ClaimProcessingResult objects
        """
        results = []
        for text in texts:
            result = self.process_claims(text)
            results.append(result)
        return results
    
    def get_claim_statistics(self, result: ClaimProcessingResult) -> Dict[str, Any]:
        """
        Get detailed statistics about processed claims.
        
        Args:
            result: ClaimProcessingResult object
            
        Returns:
            Dictionary with statistics
        """
        if not result.atomic_claims:
            return {
                "total_claims": 0,
                "checkworthy_claims": 0,
                "average_score": 0.0,
                "score_distribution": {},
                "processing_time": result.processing_time,
                "success_rate": 0.0
            }
        
        # Calculate score distribution
        scores = [claim["checkworthiness"] for claim in result.atomic_claims]
        score_distribution = {
            "high": len([s for s in scores if s >= 0.7]),
            "medium": len([s for s in scores if 0.4 <= s < 0.7]),
            "low": len([s for s in scores if s < 0.4])
        }
        
        # Calculate success rate
        success_rate = result.checkworthy_claims / max(1, result.total_claims)
        
        return {
            "total_claims": result.total_claims,
            "checkworthy_claims": result.checkworthy_claims,
            "average_score": result.average_score,
            "score_distribution": score_distribution,
            "processing_time": result.processing_time,
            "success_rate": success_rate,
            "errors": result.errors
        }


def process_claims_for_verification(text: str, min_score: float = 0.3) -> List[Dict[str, Any]]:
    """
    Convenience function to process claims for verification pipeline.
    
    Args:
        text: Input text
        min_score: Minimum checkworthiness score
        
    Returns:
        List of claim dictionaries ready for verification
    """
    processor = ClaimProcessor(min_checkworthiness=min_score)
    result = processor.process_claims(text)
    
    # Convert to verification-ready format
    claims_for_verification = []
    for claim in result.atomic_claims:
        claims_for_verification.append({
            "id": claim["id"],
            "text": claim["text"],
            "subject": claim["subject"],
            "predicate": claim["predicate"],
            "object": claim["object"],
            "checkworthiness": claim["checkworthiness"],
            "context": claim["context"]
        })
    
    return claims_for_verification


def extract_top_claims(text: str, top_k: int = 3, min_score: float = 0.3) -> List[Dict[str, Any]]:
    """
    Extract top-k most checkworthy claims from text.
    
    Args:
        text: Input text
        top_k: Number of top claims to return
        min_score: Minimum checkworthiness score
        
    Returns:
        List of top-k claim dictionaries
    """
    processor = ClaimProcessor(min_checkworthiness=min_score)
    result = processor.process_claims(text)
    
    # Sort by checkworthiness and take top-k
    sorted_claims = sorted(
        result.atomic_claims, 
        key=lambda x: x["checkworthiness"], 
        reverse=True
    )[:top_k]
    
    # Convert to dictionary format
    top_claims = []
    for claim in sorted_claims:
        top_claims.append({
            "id": claim["id"],
            "text": claim["text"],
            "subject": claim["subject"],
            "predicate": claim["predicate"],
            "object": claim["object"],
            "checkworthiness": claim["checkworthiness"],
            "context": claim["context"]
        })
    
    return top_claims
