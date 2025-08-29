#!/usr/bin/env python3
"""
Enhanced Stance Classifier for TruthLens
Implements improved thresholds (0.6) and rule-based signals for better stance detection.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class EnhancedStanceResult:
    """Enhanced stance detection result with confidence and evidence."""
    stance: str  # "support", "contradict", "neutral"
    confidence: float
    evidence_sentences: List[str]
    reasoning: str
    rule_based_override: Optional[str] = None
    model_probabilities: Optional[Dict[str, float]] = None

class EnhancedStanceClassifier:
    """
    Enhanced stance classifier with improved thresholds and rule-based signals.
    
    Key improvements:
    - Threshold tuning: support_prob > 0.6, contradict_prob > 0.6
    - Rule-based signals for explicit contradictions
    - Better causal reasoning for destruction/impact claims
    - Scientific consensus handling
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize the enhanced stance classifier.
        
        Args:
            model_name: NLI model to use (default: facebook/bart-large-mnli)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Improved thresholds as requested
        self.support_threshold = 0.6
        self.contradict_threshold = 0.6
        
        # Rule-based contradiction keywords (highest priority)
        self.contradiction_keywords = [
            'false', 'debunked', 'misleading', 'inaccurate', 'wrong', 'fake', 'hoax',
            'myth', 'disproven', 'refuted', 'denied', 'rejected', 'contradicted',
            'fact check', 'misinformation', 'disinformation', 'not true', 'untrue',
            'no evidence', 'no link', 'no connection', 'does not cause', 'does not lead to',
            'never happened', 'never occurred', 'baseless', 'unfounded'
        ]
        
        # Support keywords for verification
        self.support_keywords = [
            'confirmed', 'verified', 'true', 'accurate', 'correct', 'proven', 'fact',
            'study shows', 'research confirms', 'evidence shows', 'scientists say',
            'official', 'announced', 'reported', 'confirmed by', 'verified by'
        ]
        
        # Causal reasoning patterns for destruction/impact claims
        self.destruction_indicators = [
            r'\b(deaths?|fatalities|casualties|killed|died|perished)\b',
            r'\b(rescue|evacuation|emergency|disaster|catastrophe)\b',
            r'\b(missing|trapped|stranded|isolated)\b',
            r'\b(damage|destruction|devastation|ruined|destroyed)\b',
            r'\b(flood|flooding|water|inundated|submerged)\b',
            r'\b(heavy rains?|torrential|downpour|storm)\b',
            r'\b(army|military|relief|aid|assistance)\b',
            r'\b(displaced|homeless|evacuated)\b'
        ]
        
        # Scientific consensus topics
        self.scientific_consensus_topics = [
            'vaccine autism', 'vaccines cause autism', 'vaccine autism link',
            'earth flat', 'earth is flat', 'flat earth',
            'climate change hoax', 'global warming hoax', 'climate change fake',
            '5g coronavirus', '5g covid', '5g causes coronavirus',
            'moon landing fake', 'moon landing hoax'
        ]
        
        # Compile regex patterns
        self.destruction_regex = re.compile('|'.join(self.destruction_indicators), re.IGNORECASE)
        
        # Initialize NLI model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded NLI model {model_name} on {self.device}")
            logger.info(f"Using thresholds: support={self.support_threshold}, contradict={self.contradict_threshold}")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            self.tokenizer = None
            self.model = None
    
    def classify_stance(self, claim: str, article: Dict[str, Any]) -> EnhancedStanceResult:
        """
        Classify stance with improved logic and rule-based signals.
        
        Args:
            claim: The claim to verify
            article: Article containing title, description, content
            
        Returns:
            EnhancedStanceResult with stance, confidence, and reasoning
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()
        
        text_to_analyze = f"{title} {description} {content}"
        
        # Step 1: Check for rule-based contradictions (highest priority)
        contradiction_result = self._check_rule_based_contradiction(text_to_analyze)
        if contradiction_result:
            return contradiction_result
        
        # Step 2: Check for rule-based support
        support_result = self._check_rule_based_support(text_to_analyze)
        if support_result:
            return support_result
        
        # Step 3: Check for scientific consensus claims
        if self._is_scientific_consensus_claim(claim):
            return self._handle_scientific_consensus_claim(claim, text_to_analyze)
        
        # Step 4: Check for causal reasoning (destruction/impact claims)
        if self._is_causal_claim(claim):
            causal_result = self._analyze_causal_stance(claim, text_to_analyze)
            if causal_result:
                return causal_result
        
        # Step 5: Use NLI model if available
        if self.model and self.tokenizer:
            nli_result = self._classify_with_nli(claim, text_to_analyze)
            if nli_result:
                return nli_result
        
        # Step 6: Default to neutral
        return EnhancedStanceResult(
            stance="neutral",
            confidence=0.5,
            evidence_sentences=[],
            reasoning="No clear stance detected"
        )
    
    def _check_rule_based_contradiction(self, text: str) -> Optional[EnhancedStanceResult]:
        """Check for explicit contradiction keywords."""
        for keyword in self.contradiction_keywords:
            if keyword in text:
                evidence_sentences = self._extract_evidence_sentences(text, keyword)
                return EnhancedStanceResult(
                    stance="contradict",
                    confidence=0.9,
                    evidence_sentences=evidence_sentences,
                    reasoning=f"Article explicitly contains contradiction keyword: '{keyword}'",
                    rule_based_override="contradiction_keyword"
                )
        return None
    
    def _check_rule_based_support(self, text: str) -> Optional[EnhancedStanceResult]:
        """Check for explicit support keywords."""
        for keyword in self.support_keywords:
            if keyword in text:
                evidence_sentences = self._extract_evidence_sentences(text, keyword)
                return EnhancedStanceResult(
                    stance="support",
                    confidence=0.8,
                    evidence_sentences=evidence_sentences,
                    reasoning=f"Article explicitly contains support keyword: '{keyword}'",
                    rule_based_override="support_keyword"
                )
        return None
    
    def _is_scientific_consensus_claim(self, claim: str) -> bool:
        """Check if claim is about a topic with strong scientific consensus."""
        claim_lower = claim.lower()
        return any(topic in claim_lower for topic in self.scientific_consensus_topics)
    
    def _handle_scientific_consensus_claim(self, claim: str, text: str) -> EnhancedStanceResult:
        """Handle claims with strong scientific consensus."""
        # For consensus claims, look for refutation evidence
        if any(keyword in text for keyword in self.contradiction_keywords):
            evidence_sentences = self._extract_evidence_sentences(text, "scientific consensus")
            return EnhancedStanceResult(
                stance="contradict",
                confidence=0.95,
                evidence_sentences=evidence_sentences,
                reasoning="Scientific consensus claim with explicit refutation evidence",
                rule_based_override="scientific_consensus"
            )
        
        # Default to contradict for consensus claims (they're usually false)
        return EnhancedStanceResult(
            stance="contradict",
            confidence=0.9,
            evidence_sentences=[],
            reasoning="Scientific consensus claim: refuted by scientific evidence",
            rule_based_override="scientific_consensus"
        )
    
    def _is_causal_claim(self, claim: str) -> bool:
        """Check if claim is causal."""
        causal_words = ['causes', 'caused', 'leads to', 'led to', 'results in', 'resulted in', 'destruction', 'damage']
        return any(word in claim.lower() for word in causal_words)
    
    def _analyze_causal_stance(self, claim: str, text: str) -> Optional[EnhancedStanceResult]:
        """Analyze stance for causal claims by looking for evidence of cause-effect relationships."""
        claim_lower = claim.lower()
        
        # Extract cause and effect from claim
        cause_effect = self._extract_cause_effect(claim_lower)
        if not cause_effect:
            return None
        
        cause, effect = cause_effect
        
        # Look for evidence of the effect in the article
        effect_evidence = self._find_effect_evidence(effect, text)
        
        if effect_evidence:
            # If we find evidence of the effect, it supports the causal claim
            return EnhancedStanceResult(
                stance="support",
                confidence=0.7,
                evidence_sentences=effect_evidence,
                reasoning=f"Found evidence of effect '{effect}' supporting causal claim",
                rule_based_override="causal_reasoning"
            )
        
        # Look for evidence that contradicts the causal relationship
        contradiction_evidence = self._find_contradiction_evidence(cause, effect, text)
        
        if contradiction_evidence:
            return EnhancedStanceResult(
                stance="contradict",
                confidence=0.7,
                evidence_sentences=contradiction_evidence,
                reasoning=f"Found evidence contradicting causal relationship between '{cause}' and '{effect}'",
                rule_based_override="causal_reasoning"
            )
        
        return None
    
    def _extract_cause_effect(self, claim: str) -> Optional[Tuple[str, str]]:
        """Extract cause and effect from a causal claim."""
        patterns = [
            r'(\w+(?:\s+\w+){0,3})\s+(?:causes?|caused|leads?\s+to|led\s+to|results?\s+in)\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})\s+(?:trigger(s|ed)?|creates?|produces?|generates?)\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})\s+(?:floods?|flooding)\s+(?:caused|causes?|led\s+to)\s+(\w+(?:\s+\w+){0,3})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        
        return None
    
    def _find_effect_evidence(self, effect: str, text: str) -> List[str]:
        """Find evidence of an effect in the text."""
        evidence = []
        
        # Look for destruction indicators
        if 'destruction' in effect.lower() or 'damage' in effect.lower():
            matches = self.destruction_regex.findall(text)
            if matches:
                # Find sentences containing destruction indicators
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in matches):
                        evidence.append(sentence.strip())
        
        return evidence[:3]  # Return top 3 evidence sentences
    
    def _find_contradiction_evidence(self, cause: str, effect: str, text: str) -> List[str]:
        """Find evidence contradicting a causal relationship."""
        evidence = []
        
        # Look for sentences that suggest the effect didn't happen
        negation_patterns = [
            r'no\s+\w+', r'not\s+\w+', r'never\s+\w+', r'does\s+not', r'did\s+not'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(re.search(pattern, sentence_lower) for pattern in negation_patterns):
                if cause.lower() in sentence_lower or effect.lower() in sentence_lower:
                    evidence.append(sentence.strip())
        
        return evidence[:3]
    
    def _classify_with_nli(self, claim: str, text: str) -> Optional[EnhancedStanceResult]:
        """Classify stance using NLI model with improved thresholds."""
        try:
            # Prepare input for NLI model
            inputs = self.tokenizer(
                claim,
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            # Map probabilities to labels
            contradict_prob = probabilities[0]  # contradiction
            neutral_prob = probabilities[1]     # neutral
            support_prob = probabilities[2]     # entailment
            
            model_probs = {
                "contradict": float(contradict_prob),
                "neutral": float(neutral_prob),
                "support": float(support_prob)
            }
            
            # Apply improved thresholds
            if support_prob > self.support_threshold:
                return EnhancedStanceResult(
                    stance="support",
                    confidence=float(support_prob),
                    evidence_sentences=[],
                    reasoning=f"NLI model: support probability {support_prob:.3f} > {self.support_threshold}",
                    model_probabilities=model_probs
                )
            elif contradict_prob > self.contradict_threshold:
                return EnhancedStanceResult(
                    stance="contradict",
                    confidence=float(contradict_prob),
                    evidence_sentences=[],
                    reasoning=f"NLI model: contradict probability {contradict_prob:.3f} > {self.contradict_threshold}",
                    model_probabilities=model_probs
                )
            else:
                return EnhancedStanceResult(
                    stance="neutral",
                    confidence=float(neutral_prob),
                    evidence_sentences=[],
                    reasoning=f"NLI model: neutral probability {neutral_prob:.3f} (no threshold exceeded)",
                    model_probabilities=model_probs
                )
                
        except Exception as e:
            logger.error(f"Error in NLI classification: {e}")
            return None
    
    def _extract_evidence_sentences(self, text: str, keyword: str) -> List[str]:
        """Extract sentences containing evidence for a keyword."""
        sentences = re.split(r'[.!?]+', text)
        evidence_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if keyword.lower() in sentence.lower() and len(sentence) > 20:
                evidence_sentences.append(sentence)
        
        return evidence_sentences[:3]  # Return top 3 evidence sentences
