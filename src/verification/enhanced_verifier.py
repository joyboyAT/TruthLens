#!/usr/bin/env python3
"""
Enhanced Verifier using NLI for stance detection
Works with any input text and evidence, not just pre-detected claims
"""

import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class VerificationResult:
    """Result of verification using NLI."""
    claim_text: str
    evidence_text: str
    stance: str  # "entailment", "contradiction", "neutral"
    confidence_score: float
    source: str
    reasoning: str

class EnhancedVerifier:
    """
    Enhanced verifier that uses NLI models for stance detection.
    Works with any input text and evidence.
    """
    
    def __init__(self):
        self.nli_model = None
        self.tokenizer = None
        self._load_nli_model()
    
    def _load_nli_model(self):
        """Load the NLI model for stance detection."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            model_name = "roberta-large-mnli"
            print(f"Loading NLI model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.nli_model = self.nli_model.to('cuda')
                print("NLI model loaded on GPU")
            else:
                print("NLI model loaded on CPU")
                
        except Exception as e:
            print(f"Error loading NLI model: {e}")
            print("Falling back to rule-based verification")
            self.nli_model = None
    
    def verify_text_against_evidence(self, text: str, evidence_list: List[Any]) -> List[VerificationResult]:
        """
        Verify any input text against a list of evidence.
        
        Args:
            text: Input text to verify (can be any text, not just claims)
            evidence_list: List of evidence objects (DynamicEvidence or dict)
            
        Returns:
            List of verification results
        """
        results = []
        
        for evidence in evidence_list:
            # Handle both DynamicEvidence objects and dictionaries
            if hasattr(evidence, 'content'):
                # DynamicEvidence object
                evidence_content = evidence.content
                evidence_source = evidence.source
            else:
                # Dictionary
                evidence_content = evidence.get('content', '')
                evidence_source = evidence.get('source', 'Unknown')
            
            if evidence_content:
                result = self._verify_single_pair(text, evidence_content, evidence_source)
                results.append(result)
        
        return results
    
    def _verify_single_pair(self, claim: str, evidence: str, source: str) -> VerificationResult:
        """Verify a single claim-evidence pair using NLI."""
        
        if self.nli_model is not None:
            return self._nli_verification(claim, evidence, source)
        else:
            return self._rule_based_verification(claim, evidence, source)
    
    def _nli_verification(self, claim: str, evidence: str, source: str) -> VerificationResult:
        """Use NLI model for verification."""
        try:
            import torch
            
            # Prepare input for NLI
            # NLI models expect: "premise" + "hypothesis" format
            # We treat evidence as premise and claim as hypothesis
            premise = evidence[:500]  # Limit evidence length
            hypothesis = claim
            
            # Tokenize
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
            # NLI labels: 0=entailment, 1=neutral, 2=contradiction
            probs = probabilities[0].cpu().numpy()
            stance_idx = probs.argmax()
            confidence = float(probs[stance_idx])
            
            # Map to stance labels
            stance_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
            stance = stance_map.get(stance_idx, "neutral")
            
            # Generate reasoning
            reasoning = self._generate_reasoning(stance, confidence, claim, evidence)
            
            return VerificationResult(
                claim_text=claim,
                evidence_text=evidence[:200] + "..." if len(evidence) > 200 else evidence,
                stance=stance,
                confidence_score=confidence,
                source=source,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"Error in NLI verification: {e}")
            return self._rule_based_verification(claim, evidence, source)
    
    def _rule_based_verification(self, claim: str, evidence: str, source: str) -> VerificationResult:
        """Enhanced rule-based verification with source-specific logic."""
        import re
        
        # Simple keyword matching
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Extract key terms from claim
        claim_terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', claim_lower))
        
        # Count matching terms
        matching_terms = sum(1 for term in claim_terms if term in evidence_lower)
        total_terms = len(claim_terms)
        
        # Source-specific adjustments
        source_confidence_boost = 0.0
        if source.lower() in ['snopes', 'politifact', 'factcheck.org']:
            source_confidence_boost = 0.2  # Boost for fact-checking sources
        elif source.lower() == 'wikipedia':
            source_confidence_boost = 0.1  # Boost for Wikipedia
        
        # Look for contradiction indicators
        contradiction_indicators = ['false', 'debunked', 'misleading', 'no evidence', 'unfounded', 'hoax', 'myth']
        contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in evidence_lower)
        
        # Look for support indicators
        support_indicators = ['true', 'confirmed', 'verified', 'accurate', 'correct', 'supported by']
        support_count = sum(1 for indicator in support_indicators if indicator in evidence_lower)
        
        if total_terms == 0:
            confidence = 0.5
            stance = "neutral"
        else:
            match_ratio = matching_terms / total_terms
            
            # Adjust based on contradiction/support indicators
            if contradiction_count > support_count:
                stance = "contradiction"
                confidence = min(0.9, 0.6 + contradiction_count * 0.1 + source_confidence_boost)
            elif support_count > contradiction_count:
                stance = "entailment"
                confidence = min(0.9, 0.5 + match_ratio * 0.3 + support_count * 0.1 + source_confidence_boost)
            elif match_ratio > 0.6:
                stance = "entailment"
                confidence = min(0.8, 0.5 + match_ratio * 0.3 + source_confidence_boost)
            elif match_ratio < 0.3:
                stance = "contradiction"
                confidence = min(0.8, 0.4 + (1 - match_ratio) * 0.4 + source_confidence_boost)
            else:
                stance = "neutral"
                confidence = 0.5 + source_confidence_boost
        
        reasoning = self._generate_reasoning(stance, confidence, claim, evidence)
        
        return VerificationResult(
            claim_text=claim,
            evidence_text=evidence[:200] + "..." if len(evidence) > 200 else evidence,
            stance=stance,
            confidence_score=confidence,
            source=source,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, stance: str, confidence: float, claim: str, evidence: str) -> str:
        """Generate human-readable reasoning for the verification result."""
        
        if stance == "entailment":
            if confidence > 0.8:
                return f"The evidence strongly supports this claim. The information in the evidence directly relates to and confirms the statement."
            else:
                return f"The evidence provides some support for this claim, though the connection may not be definitive."
        
        elif stance == "contradiction":
            if confidence > 0.8:
                return f"The evidence contradicts this claim. The information in the evidence directly refutes the statement."
            else:
                return f"The evidence appears to contradict this claim, though the contradiction may not be definitive."
        
        else:  # neutral
            return f"The evidence neither clearly supports nor contradicts this claim. More specific information would be needed to make a determination."
    
    def get_overall_verdict(self, verification_results: List[VerificationResult]) -> Dict[str, Any]:
        """Get overall verdict from multiple verification results."""
        
        if not verification_results:
            return {
                "verdict": "Not Verified",
                "confidence": 0.0,
                "reasoning": "No evidence available for verification."
            }
        
        # Calculate weighted average based on source reliability
        source_weights = {
            "Wikipedia": 0.8,
            "Fact-Checking Organizations": 0.9,
            "News": 0.7,
            "Web Search": 0.6,
            "Unknown": 0.5
        }
        
        total_weight = 0
        weighted_score = 0
        
        for result in verification_results:
            weight = source_weights.get(result.source, 0.5)
            
            # Convert stance to score
            if result.stance == "entailment":
                score = result.confidence_score
            elif result.stance == "contradiction":
                score = 1 - result.confidence_score
            else:  # neutral
                score = 0.5
            
            weighted_score += score * weight * result.confidence_score
            total_weight += weight * result.confidence_score
        
        if total_weight == 0:
            overall_score = 0.5
        else:
            overall_score = weighted_score / total_weight
        
        # Determine verdict
        if overall_score >= 0.7:
            verdict = "Likely True"
        elif overall_score <= 0.3:
            verdict = "Likely False"
        else:
            verdict = "Unclear"
        
        # Generate overall reasoning
        reasoning = self._generate_overall_reasoning(verification_results, overall_score)
        
        return {
            "verdict": verdict,
            "confidence": overall_score,
            "reasoning": reasoning,
            "evidence_count": len(verification_results)
        }
    
    def _generate_overall_reasoning(self, results: List[VerificationResult], overall_score: float) -> str:
        """Generate overall reasoning based on all verification results."""
        
        if not results:
            return "No evidence was found to verify this claim."
        
        # Count stances
        stances = [r.stance for r in results]
        entailment_count = stances.count("entailment")
        contradiction_count = stances.count("contradiction")
        neutral_count = stances.count("neutral")
        
        if entailment_count > contradiction_count and entailment_count > neutral_count:
            return f"Multiple sources ({entailment_count} out of {len(results)}) support this claim, suggesting it is likely true."
        elif contradiction_count > entailment_count and contradiction_count > neutral_count:
            return f"Multiple sources ({contradiction_count} out of {len(results)}) contradict this claim, suggesting it is likely false."
        else:
            return f"The evidence is mixed or unclear. {len(results)} sources were checked, but no clear consensus emerged."

# Convenience function for easy integration
def verify_text_with_evidence(text: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify any text against evidence and return overall verdict."""
    verifier = EnhancedVerifier()
    results = verifier.verify_text_against_evidence(text, evidence_list)
    return verifier.get_overall_verdict(results)
