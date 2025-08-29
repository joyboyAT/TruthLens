#!/usr/bin/env python3
"""
Enhanced Verdict Aggregator for TruthLens
Implements improved weighted voting logic and better fact-check integration.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class VerdictResult:
    """Result of verdict aggregation."""
    verdict: str  # "Likely True", "Likely False", "Unclear", "Not Enough Info"
    confidence: float
    reasoning: str
    stance_distribution: Dict[str, int]
    stance_percentages: Dict[str, float]
    fact_check_override: Optional[str] = None
    evidence_summary: str = ""

class EnhancedVerdictAggregator:
    """
    Enhanced verdict aggregator with improved logic.
    
    Key improvements:
    - Weighted voting: Support > 40% → Likely True, Contradict > 40% → Likely False
    - Strong fact-check integration: Google Fact Check overrides weak news signals
    - Better handling of scientific consensus claims
    - Improved confidence calculation
    """
    
    def __init__(self):
        """Initialize the enhanced verdict aggregator."""
        # Improved thresholds as requested
        self.support_percentage_threshold = 0.4  # 40% support = Likely True
        self.contradict_percentage_threshold = 0.4  # 40% contradict = Likely False
        
        # Confidence thresholds for verdict levels
        self.likely_true_threshold = 0.6
        self.likely_false_threshold = 0.4
        
        # Scientific consensus topics that should default to False
        self.scientific_consensus_topics = [
            'vaccine autism', 'vaccines cause autism', 'vaccine autism link',
            'earth flat', 'earth is flat', 'flat earth',
            'climate change hoax', 'global warming hoax', 'climate change fake',
            '5g coronavirus', '5g covid', '5g causes coronavirus',
            'moon landing fake', 'moon landing hoax'
        ]
    
    def aggregate_verdict(self, 
                         claim: str,
                         stance_results: List[Dict[str, Any]],
                         fact_check_result: Optional[Dict[str, Any]] = None,
                         total_articles: int = 0) -> VerdictResult:
        """
        Aggregate verdict using improved weighted voting logic.
        
        Args:
            claim: The claim being verified
            stance_results: List of stance detection results
            fact_check_result: Google Fact Check or other fact-check result
            total_articles: Total number of articles analyzed
            
        Returns:
            VerdictResult with final verdict and reasoning
        """
        if not stance_results and not fact_check_result:
            return VerdictResult(
                verdict="Not Enough Info",
                confidence=0.0,
                reasoning="No evidence available for analysis",
                stance_distribution={},
                stance_percentages={},
                evidence_summary="No articles or fact-check results found"
            )
        
        # Count stances
        stance_counts = defaultdict(int)
        for result in stance_results:
            stance = result.get('stance', 'neutral')
            stance_counts[stance] += 1
        
        total_stances = len(stance_results)
        
        # Calculate percentages
        stance_percentages = {}
        for stance in ['support', 'contradict', 'neutral']:
            stance_percentages[stance] = stance_counts[stance] / total_stances if total_stances > 0 else 0
        
        # Check for scientific consensus claims first
        if self._is_scientific_consensus_claim(claim):
            return self._handle_scientific_consensus_verdict(
                claim, stance_percentages, fact_check_result
            )
        
        # Check fact-check result (highest priority)
        if fact_check_result:
            return self._handle_fact_check_verdict(
                claim, stance_percentages, fact_check_result
            )
        
        # Apply improved weighted voting logic
        return self._apply_weighted_voting(stance_percentages, total_stances)
    
    def _is_scientific_consensus_claim(self, claim: str) -> bool:
        """Check if claim is about a topic with strong scientific consensus."""
        claim_lower = claim.lower()
        return any(topic in claim_lower for topic in self.scientific_consensus_topics)
    
    def _handle_scientific_consensus_verdict(self, 
                                           claim: str,
                                           stance_percentages: Dict[str, float],
                                           fact_check_result: Optional[Dict[str, Any]]) -> VerdictResult:
        """Handle verdicts for scientific consensus claims."""
        support_pct = stance_percentages.get('support', 0)
        contradict_pct = stance_percentages.get('contradict', 0)
        
        # If fact-check confirms refutation, use that
        if fact_check_result:
            verdict = fact_check_result.get('verdict', '').upper()
            if 'REFUTED' in verdict or 'FALSE' in verdict:
                return VerdictResult(
                    verdict="Likely False",
                    confidence=0.95,
                    reasoning=f"Scientific consensus claim refuted by fact-check: {verdict}",
                    stance_distribution={'support': 0, 'contradict': 0, 'neutral': 0},
                    stance_percentages=stance_percentages,
                    fact_check_override="scientific_consensus_refuted",
                    evidence_summary=f"Fact-check verdict: {verdict}"
                )
        
        # For consensus claims, high support percentage indicates misinformation
        if support_pct > 0.2:  # Even 20% support for consensus claims is concerning
            return VerdictResult(
                verdict="Likely False",
                confidence=0.9,
                reasoning=f"Scientific consensus claim with {support_pct:.1%} support (likely misinformation)",
                stance_distribution={'support': 0, 'contradict': 0, 'neutral': 0},
                stance_percentages=stance_percentages,
                fact_check_override="scientific_consensus_misinformation",
                evidence_summary=f"High support ({support_pct:.1%}) for scientifically refuted claim"
            )
        
        # Default to refuted for consensus claims
        return VerdictResult(
            verdict="Likely False",
            confidence=0.85,
            reasoning="Scientific consensus claim: refuted by scientific evidence",
            stance_distribution={'support': 0, 'contradict': 0, 'neutral': 0},
            stance_percentages=stance_percentages,
            fact_check_override="scientific_consensus_default",
            evidence_summary="Claim contradicts established scientific consensus"
        )
    
    def _handle_fact_check_verdict(self,
                                  claim: str,
                                  stance_percentages: Dict[str, float],
                                  fact_check_result: Dict[str, Any]) -> VerdictResult:
        """Handle verdicts when fact-check result is available."""
        verdict = fact_check_result.get('verdict', '').upper()
        confidence = fact_check_result.get('confidence', 0.8)
        
        support_pct = stance_percentages.get('support', 0)
        contradict_pct = stance_percentages.get('contradict', 0)
        
        # Fact-check result overrides news analysis
        if 'REFUTED' in verdict or 'FALSE' in verdict:
            return VerdictResult(
                verdict="Likely False",
                confidence=max(0.8, confidence),  # At least 80% confidence
                reasoning=f"Fact-check result: {verdict} (confidence: {confidence:.1%}). News analysis: {support_pct:.1%} support, {contradict_pct:.1%} contradict",
                stance_distribution={'support': 0, 'contradict': 0, 'neutral': 0},
                stance_percentages=stance_percentages,
                fact_check_override="refuted",
                evidence_summary=f"Fact-check verdict: {verdict}"
            )
        elif 'SUPPORTED' in verdict or 'TRUE' in verdict:
            return VerdictResult(
                verdict="Likely True",
                confidence=max(0.8, confidence),  # At least 80% confidence
                reasoning=f"Fact-check result: {verdict} (confidence: {confidence:.1%}). News analysis: {support_pct:.1%} support, {contradict_pct:.1%} contradict",
                stance_distribution={'support': 0, 'contradict': 0, 'neutral': 0},
                stance_percentages=stance_percentages,
                fact_check_override="supported",
                evidence_summary=f"Fact-check verdict: {verdict}"
            )
        else:
            # Mixed or unclear from fact-check, use news analysis as tiebreaker
            return self._apply_weighted_voting(stance_percentages, sum(stance_percentages.values()))
    
    def _apply_weighted_voting(self, 
                              stance_percentages: Dict[str, float],
                              total_stances: int) -> VerdictResult:
        """Apply improved weighted voting logic."""
        support_pct = stance_percentages.get('support', 0)
        contradict_pct = stance_percentages.get('contradict', 0)
        neutral_pct = stance_percentages.get('neutral', 0)
        
        # Apply 40% thresholds as requested
        if support_pct > self.support_percentage_threshold:
            confidence = 0.6 + (support_pct * 0.3)  # Base 0.6 + up to 0.3 for high support
            return VerdictResult(
                verdict="Likely True",
                confidence=min(0.9, confidence),
                reasoning=f"Weighted voting: {support_pct:.1%} support > {self.support_percentage_threshold:.1%} threshold",
                stance_distribution={'support': int(support_pct * total_stances), 'contradict': int(contradict_pct * total_stances), 'neutral': int(neutral_pct * total_stances)},
                stance_percentages=stance_percentages,
                evidence_summary=f"Strong support evidence ({support_pct:.1%}) exceeds threshold"
            )
        elif contradict_pct > self.contradict_percentage_threshold:
            confidence = 0.1 + (contradict_pct * 0.2)  # Base 0.1 + up to 0.2 for high contradiction
            return VerdictResult(
                verdict="Likely False",
                confidence=min(0.8, confidence),
                reasoning=f"Weighted voting: {contradict_pct:.1%} contradict > {self.contradict_percentage_threshold:.1%} threshold",
                stance_distribution={'support': int(support_pct * total_stances), 'contradict': int(contradict_pct * total_stances), 'neutral': int(neutral_pct * total_stances)},
                stance_percentages=stance_percentages,
                evidence_summary=f"Strong contradiction evidence ({contradict_pct:.1%}) exceeds threshold"
            )
        else:
            # Mixed or unclear evidence
            if total_stances == 0:
                return VerdictResult(
                    verdict="Not Enough Info",
                    confidence=0.0,
                    reasoning="No articles found to analyze",
                    stance_distribution={},
                    stance_percentages=stance_percentages,
                    evidence_summary="No evidence available"
                )
            else:
                # Calculate confidence based on evidence distribution
                max_pct = max(support_pct, contradict_pct, neutral_pct)
                if max_pct < 0.3:  # Very mixed evidence
                    confidence = 0.3
                    verdict = "Unclear"
                else:
                    confidence = 0.5
                    verdict = "Unclear"
                
                return VerdictResult(
                    verdict=verdict,
                    confidence=confidence,
                    reasoning=f"Mixed evidence: {support_pct:.1%} support, {contradict_pct:.1%} contradict, {neutral_pct:.1%} neutral (no threshold exceeded)",
                    stance_distribution={'support': int(support_pct * total_stances), 'contradict': int(contradict_pct * total_stances), 'neutral': int(neutral_pct * total_stances)},
                    stance_percentages=stance_percentages,
                    evidence_summary=f"Mixed evidence with no clear majority (max: {max_pct:.1%})"
                )
    
    def create_confidence_badge(self, verdict_result: VerdictResult) -> Dict[str, Any]:
        """Create a confidence badge from verdict result."""
        if verdict_result.verdict == "Likely True":
            return {
                "level": "Likely True",
                "confidence": verdict_result.confidence,
                "color": "green",
                "emoji": "🟢",
                "reasoning": verdict_result.reasoning
            }
        elif verdict_result.verdict == "Likely False":
            return {
                "level": "Likely False",
                "confidence": verdict_result.confidence,
                "color": "red",
                "emoji": "🔴",
                "reasoning": verdict_result.reasoning
            }
        elif verdict_result.verdict == "Unclear":
            return {
                "level": "Unclear",
                "confidence": verdict_result.confidence,
                "color": "yellow",
                "emoji": "🟡",
                "reasoning": verdict_result.reasoning
            }
        else:  # Not Enough Info
            return {
                "level": "Not Enough Info",
                "confidence": 0.0,
                "color": "gray",
                "emoji": "⚪",
                "reasoning": verdict_result.reasoning
            }
    
    def get_evidence_summary(self, stance_results: List[Dict[str, Any]]) -> str:
        """Generate a summary of evidence used for the verdict."""
        if not stance_results:
            return "No evidence available"
        
        # Count evidence types
        evidence_counts = defaultdict(int)
        rule_based_overrides = []
        
        for result in stance_results:
            stance = result.get('stance', 'neutral')
            evidence_counts[stance] += 1
            
            # Track rule-based overrides
            override = result.get('rule_based_override')
            if override:
                rule_based_overrides.append(override)
        
        summary_parts = []
        
        # Add stance distribution
        if evidence_counts:
            stance_summary = ", ".join([f"{count} {stance}" for stance, count in evidence_counts.items()])
            summary_parts.append(f"Evidence: {stance_summary}")
        
        # Add rule-based overrides
        if rule_based_overrides:
            override_summary = ", ".join(set(rule_based_overrides))
            summary_parts.append(f"Rule-based signals: {override_summary}")
        
        return ". ".join(summary_parts) if summary_parts else "Evidence analysis completed"
