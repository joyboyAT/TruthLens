#!/usr/bin/env python3
"""
Enhanced Verifier with Google Fact Check API Integration
Uses Google Fact Check API and NLI models for comprehensive claim verification.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Import Google Fact Check API
try:
    from .google_factcheck_api import GoogleFactCheckAPI, FactCheckResult
    GOOGLE_FACTCHECK_AVAILABLE = True
except ImportError:
    GOOGLE_FACTCHECK_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """Result of claim verification."""
    claim_text: str
    evidence_text: str
    stance: str  # SUPPORTED, REFUTED, NOT ENOUGH INFO
    confidence_score: float
    source: str
    reasoning: str
    google_factcheck_result: Optional[Dict[str, Any]] = None

class EnhancedVerifier:
    """Enhanced verifier using Google Fact Check API and NLI models."""
    
    def __init__(self, google_api_key: Optional[str] = None, nli_model_name: str = "roberta-large-mnli"):
        """
        Initialize the enhanced verifier.
        
        Args:
            google_api_key: Google Fact Check API key
            nli_model_name: NLI model to use for verification
        """
        self.google_factcheck = None
        self.nli_model = None
        self.nli_tokenizer = None
        
        # Initialize Google Fact Check API
        if GOOGLE_FACTCHECK_AVAILABLE and google_api_key:
            try:
                self.google_factcheck = GoogleFactCheckAPI(google_api_key)
                logger.info("Google Fact Check API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Fact Check API: {e}")
                self.google_factcheck = None
        
        # Initialize NLI model
        try:
            logger.info(f"Loading NLI model: {nli_model_name}")
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
            self.nli_model.eval()
            logger.info("NLI model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading NLI model: {e}")
            logger.info("Falling back to rule-based verification")
            self.nli_model = None
            self.nli_tokenizer = None
    
    def verify_claim_with_google_factcheck(self, claim: str) -> Optional[FactCheckResult]:
        """Verify claim using Google Fact Check API."""
        if not self.google_factcheck:
            return None
        
        try:
            logger.info(f"Verifying claim with Google Fact Check API: {claim[:50]}...")
            result = self.google_factcheck.verify_claim(claim)
            
            if result:
                logger.info(f"Found Google Fact Check result: {result.verdict} (confidence: {result.confidence:.1%})")
                return result
            else:
                logger.info("No Google Fact Check result found")
                return None
                
        except Exception as e:
            logger.error(f"Error verifying claim with Google Fact Check API: {e}")
            return None
    
    def verify_with_nli_model(self, claim: str, evidence: str) -> tuple[str, float]:
        """Verify claim against evidence using NLI model."""
        if not self.nli_model or not self.nli_tokenizer:
            return self._rule_based_verification(claim, evidence)
        
        try:
            # Prepare input for NLI model
            inputs = self.nli_tokenizer(
                claim,
                evidence,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                scores = probabilities[0].numpy()
            
            # Map NLI labels to our stance
            # NLI labels: 0=entailment, 1=neutral, 2=contradiction
            entailment_score = scores[0]  # SUPPORTED
            contradiction_score = scores[2]  # REFUTED
            neutral_score = scores[1]  # NOT ENOUGH INFO
            
            # Determine stance based on highest score
            if entailment_score > contradiction_score and entailment_score > neutral_score:
                stance = "SUPPORTED"
                confidence = float(entailment_score)
            elif contradiction_score > entailment_score and contradiction_score > neutral_score:
                stance = "REFUTED"
                confidence = float(contradiction_score)
            else:
                stance = "NOT ENOUGH INFO"
                confidence = float(neutral_score)
            
            return stance, confidence
            
        except Exception as e:
            logger.error(f"Error with NLI model verification: {e}")
            return self._rule_based_verification(claim, evidence)
    
    def _rule_based_verification(self, claim: str, evidence: str) -> tuple[str, float]:
        """Fallback rule-based verification."""
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Keywords that suggest contradiction
        contradiction_keywords = [
            "false", "debunked", "myth", "hoax", "disproven", "incorrect",
            "no evidence", "misleading", "inaccurate", "wrong"
        ]
        
        # Keywords that suggest support
        support_keywords = [
            "true", "confirmed", "verified", "proven", "fact", "correct",
            "evidence shows", "studies confirm", "research proves"
        ]
        
        # Count keywords
        contradiction_count = sum(1 for keyword in contradiction_keywords if keyword in evidence_lower)
        support_count = sum(1 for keyword in support_keywords if keyword in evidence_lower)
        
        # Determine stance
        if contradiction_count > support_count:
            return "REFUTED", 0.6
        elif support_count > contradiction_count:
            return "SUPPORTED", 0.6
        else:
            return "NOT ENOUGH INFO", 0.3
    
    def verify_text_against_evidence(self, claim_text: str, evidence_list: List[Dict[str, Any]]) -> List[VerificationResult]:
        """Verify claim against multiple evidence sources."""
        results = []
        
        # First, try Google Fact Check API
        google_result = self.verify_claim_with_google_factcheck(claim_text)
        
        if google_result:
            # Use Google Fact Check result as primary verification
            result = VerificationResult(
                claim_text=claim_text,
                evidence_text=f"Google Fact Check: {google_result.rating}",
                stance=google_result.verdict,
                confidence_score=google_result.confidence,
                source="Google Fact Check API",
                reasoning=google_result.explanation,
                google_factcheck_result={
                    "publisher": google_result.publisher,
                    "rating": google_result.rating,
                    "url": google_result.url,
                    "review_date": google_result.review_date
                }
            )
            results.append(result)
            logger.info(f"Primary verification from Google Fact Check: {google_result.verdict}")
        
        # If no Google Fact Check result, use evidence list with NLI model
        if not google_result and evidence_list:
            logger.info("No Google Fact Check result found, using evidence list with NLI model")
            
            for i, evidence in enumerate(evidence_list[:3]):  # Limit to top 3 evidence
                evidence_text = evidence.get('snippet', '') or evidence.get('text', '')
                source = evidence.get('source', f'Evidence {i+1}')
                
                if evidence_text:
                    stance, confidence = self.verify_with_nli_model(claim_text, evidence_text)
                    
                    result = VerificationResult(
                        claim_text=claim_text,
                        evidence_text=evidence_text[:200] + "..." if len(evidence_text) > 200 else evidence_text,
                        stance=stance,
                        confidence_score=confidence,
                        source=source,
                        reasoning=f"Verified using NLI model against {source}"
                    )
                    results.append(result)
        
        # If still no results, create a default result
        if not results:
            logger.info("No verification results found, creating default result")
            result = VerificationResult(
                claim_text=claim_text,
                evidence_text="No evidence available",
                stance="NOT ENOUGH INFO",
                confidence_score=0.0,
                source="No sources",
                reasoning="No evidence or fact-check results available for verification"
            )
            results.append(result)
        
        return results
    
    def get_overall_verdict(self, verification_results: List[VerificationResult]) -> Dict[str, Any]:
        """Get overall verdict from multiple verification results."""
        if not verification_results:
            return {
                "verdict": "NOT ENOUGH INFO",
                "confidence": 0.0,
                "reasoning": "No verification results available"
            }
        
        # If we have Google Fact Check result, use it as primary
        google_results = [r for r in verification_results if r.google_factcheck_result]
        if google_results:
            primary_result = google_results[0]
            return {
                "verdict": primary_result.stance,
                "confidence": primary_result.confidence_score,
                "reasoning": primary_result.reasoning,
                "source": "Google Fact Check API"
            }
        
        # Otherwise, aggregate results
        stances = [r.stance for r in verification_results]
        confidences = [r.confidence_score for r in verification_results]
        
        # Count stances
        stance_counts = {}
        for stance in stances:
            stance_counts[stance] = stance_counts.get(stance, 0) + 1
        
        # Get most common stance
        if stance_counts:
            most_common_stance = max(stance_counts, key=stance_counts.get)
            avg_confidence = sum(confidences) / len(confidences)
            
            return {
                "verdict": most_common_stance,
                "confidence": avg_confidence,
                "reasoning": f"Based on {len(verification_results)} verification results",
                "source": "NLI Model"
            }
        else:
            return {
                "verdict": "NOT ENOUGH INFO",
                "confidence": 0.0,
                "reasoning": "No clear verification results"
            }
