import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import yaml

from .confidence_calibrator import discrepancy_metrics


logger = logging.getLogger(__name__)


@dataclass
class Citation:
    title: str
    snippet: str
    url: str
    source_type: str = "news"
    reliability_score: float = 0.5


@dataclass
class EvidenceSnippet:
    sentence: str
    stance: str
    confidence: float
    reasoning: str
    source: str = ""
    source_type: str = "news"


@dataclass
class Verdict:
    verdict: str
    probability: float
    label: str
    citations: List[Citation]
    confidence_badge: str
    reasoning: str
    evidence_snippets: List[EvidenceSnippet]


class EnhancedVerdictMapper:
    """
    Enhanced verdict mapper with weighted voting and fact-check integration.
    
    Enhanced Features:
    - Weighted voting based on evidence distribution
    - Strong fact-check verdict integration
    - Improved confidence badge generation
    - Better reasoning and explanation
    - Explicit verdict logic encoding
    - Evidence snippets in final verdict
    - Scientific consensus and causal chain handling
    """
    
    def __init__(self, config_path: str = "phase4_verification/config/ux_mapping.yaml"):
        self.config = self._load_ux_config(config_path)
        
        # Enhanced thresholds for weighted voting
        self.support_threshold = 0.4  # 40% support for Likely True
        self.contradict_threshold = 0.4  # 40% contradict for Likely False
        self.fact_check_override_threshold = 0.8  # Strong fact-check verdicts override
        
        # Confidence badge thresholds
        self.confidence_thresholds = {
            "high": 0.7,    # 🟢 Likely True
            "medium": 0.4,  # 🟡 Unclear
            "low": 0.0      # 🔴 Likely False
        }
        
        # Explicit verdict logic encoding
        self.verdict_logic = {
            "scientific_consensus_default": "REFUTED",  # Scientific claims default to False
            "no_evidence_default": "NOT ENOUGH INFO",   # No evidence = Not Enough Info, not False
            "causal_chain_support": True,               # Treat indirect evidence as supportive
            "fact_check_override": True,                # Fact-check verdicts override other evidence
            "confidence_calibration": True              # Apply confidence calibration
        }
    
    def _load_ux_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {
                "thresholds": {"true_min": 0.7, "unclear_min": 0.4},
                "labels": {
                    "likely_true": "Likely True 🟢",
                    "unclear": "Unclear 🟡",
                    "likely_false": "Likely False 🔴",
                },
                "use_supported_probability": True,
            }

    def _calculate_weighted_voting(self, stance_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate weighted voting based on evidence distribution and reliability.
        
        Args:
            stance_results: List of stance classification results
            
        Returns:
            Dictionary with weighted percentages for each stance
        """
        if not stance_results:
            return {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 0.0}
        
        total_weight = 0.0
        stance_weights = {"SUPPORTED": 0.0, "REFUTED": 0.0, "NOT ENOUGH INFO": 0.0}
        
        for result in stance_results:
            # Get stance label and confidence
            stance = result.get("label", "NOT ENOUGH INFO")
            confidence = result.get("confidence_score", 0.5)
            reliability = result.get("reliability_score", 0.5)
            
            # Calculate weight based on confidence and reliability
            weight = confidence * reliability
            total_weight += weight
            stance_weights[stance] += weight
        
        # Normalize to percentages
        if total_weight > 0:
            for stance in stance_weights:
                stance_weights[stance] = stance_weights[stance] / total_weight
        
        return stance_weights

    def _check_fact_check_override(self, stance_results: List[Dict[str, Any]]) -> Optional[str]:
        """
        Check if there's a strong fact-check verdict that should override other evidence.
        
        Args:
            stance_results: List of stance classification results
            
        Returns:
            Override stance if strong fact-check found, None otherwise
        """
        for result in stance_results:
            # Check for fact-check sources with high confidence
            source_type = result.get("source_type", "")
            confidence = result.get("confidence_score", 0.0)
            stance = result.get("label", "")
            
            # Strong fact-check sources with high confidence
            if (source_type in ["fact_check", "snopes", "politifact", "altnews", "boom_live"] and 
                confidence > self.fact_check_override_threshold):
                
                if stance == "REFUTED":
                    logger.info(f"Fact-check override: {source_type} strongly refutes claim")
                    return "REFUTED"
                elif stance == "SUPPORTED":
                    logger.info(f"Fact-check override: {source_type} strongly supports claim")
                    return "SUPPORTED"
        
        return None

    def _check_scientific_consensus_claim(self, claim: str, stance_results: List[Dict[str, Any]]) -> Optional[str]:
        """
        Check if this is a scientific consensus claim that should default to False.
        
        Args:
            claim: The claim being verified
            stance_results: List of stance classification results
            
        Returns:
            "REFUTED" if scientific consensus claim with no strong contradicting evidence, None otherwise
        """
        claim_lower = claim.lower()
        
        # Check if this is a scientific/medical claim
        scientific_keywords = ['vaccine', 'autism', 'covid', 'coronavirus', 'medical', 'health', 'disease', 'cancer', 'cure']
        is_scientific_claim = any(keyword in claim_lower for keyword in scientific_keywords)
        
        if is_scientific_claim:
            # Check if there's strong contradicting evidence
            strong_contradicting = any(
                result.get("label") == "REFUTED" and result.get("confidence_score", 0) > 0.8
                for result in stance_results
            )
            
            if not strong_contradicting:
                # Default to False for scientific claims without strong contradicting evidence
                logger.info(f"Scientific consensus claim defaulting to False: {claim[:50]}...")
                return "REFUTED"
        
        return None

    def _check_no_evidence_case(self, stance_results: List[Dict[str, Any]]) -> bool:
        """
        Check if this is a no evidence case that should be marked as Not Enough Info.
        
        Args:
            stance_results: List of stance classification results
            
        Returns:
            True if this should be marked as Not Enough Info
        """
        # Count evidence by stance
        stance_counts = Counter(result.get("label", "NOT ENOUGH INFO") for result in stance_results)
        
        # If mostly NOT ENOUGH INFO and no strong evidence either way
        if (stance_counts["NOT ENOUGH INFO"] > len(stance_results) * 0.7 and
            stance_counts["SUPPORTED"] < 2 and stance_counts["REFUTED"] < 2):
            return True
        
        return False

    def _extract_evidence_snippets(self, stance_results: List[Dict[str, Any]]) -> List[EvidenceSnippet]:
        """
        Extract evidence snippets from stance results.
        
        Args:
            stance_results: List of stance classification results
            
        Returns:
            List of evidence snippets with stance and confidence
        """
        snippets = []
        
        for result in stance_results:
            # Get evidence snippets from the result
            evidence_snippets = result.get("evidence_snippets", [])
            
            for snippet in evidence_snippets:
                evidence_snippet = EvidenceSnippet(
                    sentence=snippet.get("sentence", ""),
                    stance=snippet.get("stance", "NOT ENOUGH INFO"),
                    confidence=snippet.get("confidence", 0.5),
                    reasoning=snippet.get("reasoning", ""),
                    source=result.get("source", ""),
                    source_type=result.get("source_type", "news")
                )
                snippets.append(evidence_snippet)
        
        return snippets

    def _generate_confidence_badge(self, weighted_votes: Dict[str, float], 
                                 fact_check_override: Optional[str] = None) -> str:
        """
        Generate confidence badge based on weighted voting and fact-check overrides.
        
        Args:
            weighted_votes: Weighted voting results
            fact_check_override: Fact-check override if any
            
        Returns:
            Confidence badge string
        """
        support_pct = weighted_votes.get("SUPPORTED", 0.0)
        refute_pct = weighted_votes.get("REFUTED", 0.0)
        
        # Apply fact-check override
        if fact_check_override == "REFUTED":
            return "🔴 Likely False"
        elif fact_check_override == "SUPPORTED":
            return "🟢 Likely True"
        
        # Weighted voting logic
        if support_pct > self.support_threshold:
            return "🟢 Likely True"
        elif refute_pct > self.contradict_threshold:
            return "🔴 Likely False"
        else:
            return "🟡 Unclear"

    def _generate_reasoning(self, weighted_votes: Dict[str, float], 
                          stance_results: List[Dict[str, Any]],
                          fact_check_override: Optional[str] = None,
                          scientific_consensus_override: Optional[str] = None,
                          no_evidence_case: bool = False) -> str:
        """
        Generate reasoning for the verdict.
        
        Args:
            weighted_votes: Weighted voting results
            stance_results: List of stance results
            fact_check_override: Fact-check override if any
            scientific_consensus_override: Scientific consensus override if any
            no_evidence_case: Whether this is a no evidence case
            
        Returns:
            Reasoning string
        """
        support_pct = weighted_votes.get("SUPPORTED", 0.0)
        refute_pct = weighted_votes.get("REFUTED", 0.0)
        neutral_pct = weighted_votes.get("NOT ENOUGH INFO", 0.0)
        
        if fact_check_override:
            fact_check_sources = [r.get("source", "") for r in stance_results 
                                if r.get("source_type", "") in ["fact_check", "snopes", "politifact"]]
            if fact_check_sources:
                return f"Strong fact-check verdict from {', '.join(fact_check_sources[:2])} overrides other evidence."
        
        if scientific_consensus_override:
            return f"Scientific consensus claim defaulting to False due to lack of strong contradicting evidence."
        
        if no_evidence_case:
            return f"Insufficient evidence to determine truth ({neutral_pct:.1%} unclear, {support_pct:.1%} support, {refute_pct:.1%} refute)."
        
        if support_pct > self.support_threshold:
            return f"Evidence strongly supports the claim ({support_pct:.1%} support vs {refute_pct:.1%} refute)."
        elif refute_pct > self.contradict_threshold:
            return f"Evidence strongly refutes the claim ({refute_pct:.1%} refute vs {support_pct:.1%} support)."
        else:
            return f"Insufficient evidence to determine truth ({neutral_pct:.1%} unclear, {support_pct:.1%} support, {refute_pct:.1%} refute)."

    def _choose_by_weight_and_label(self, results: List[Dict[str, Any]], stance_idx: int, k: int = 3) -> List[Dict[str, Any]]:
        """Choose top evidence by weight and stance."""
        def _class_contrib(ev: Dict[str, Any]) -> float:
            w = float(ev.get("w", 0.0)) if "w" in ev else float((ev.get("scores", {}) or {}).get("cross", 0.0))
            p = 0.0
            if isinstance(ev.get("probs"), list) and len(ev["probs"]) >= 3:
                p = float(ev["probs"][stance_idx])
            return w * p
        
        ranked = sorted(results, key=_class_contrib, reverse=True)
        return ranked[:k]

    def _format_citations(self, items: List[Dict[str, Any]]) -> List[Citation]:
        """Format citations with enhanced metadata."""
        citations = []
        for ev in items:
            citation = Citation(
                title=str(ev.get("title", ev.get("snippet", ""))),
                snippet=str(ev.get("snippet", ev.get("text", ""))),
                url=str(ev.get("url", "")),
                source_type=str(ev.get("source_type", "news")),
                reliability_score=float(ev.get("reliability_score", 0.5))
            )
            citations.append(citation)
        return citations

    def map_to_enhanced_verdict(self, stance_results: List[Dict[str, Any]], 
                               claim: str = "",
                               calibrated_probabilities: Optional[Dict[str, float]] = None,
                               p_raw: Optional[float] = None) -> Dict[str, Any]:
        """
        Map stance results to enhanced verdict with explicit logic encoding and evidence snippets.
        
        Args:
            stance_results: List of stance classification results
            claim: The claim being verified
            calibrated_probabilities: Calibrated probabilities (optional)
            p_raw: Raw probability (optional)
            
        Returns:
            Enhanced verdict dictionary with evidence snippets
        """
        # Calculate weighted voting
        weighted_votes = self._calculate_weighted_voting(stance_results)
        
        # Check for explicit verdict logic cases
        fact_check_override = self._check_fact_check_override(stance_results)
        scientific_consensus_override = self._check_scientific_consensus_claim(claim, stance_results)
        no_evidence_case = self._check_no_evidence_case(stance_results)
        
        # Generate confidence badge
        confidence_badge = self._generate_confidence_badge(weighted_votes, fact_check_override)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            weighted_votes, stance_results, fact_check_override, 
            scientific_consensus_override, no_evidence_case
        )
        
        # Determine final stance based on explicit logic
        if fact_check_override:
            final_stance = fact_check_override
        elif scientific_consensus_override:
            final_stance = scientific_consensus_override
        elif no_evidence_case:
            final_stance = "NOT ENOUGH INFO"
        else:
            support_pct = weighted_votes.get("SUPPORTED", 0.0)
            refute_pct = weighted_votes.get("REFUTED", 0.0)
            
            if support_pct > self.support_threshold:
                final_stance = "SUPPORTED"
            elif refute_pct > self.contradict_threshold:
                final_stance = "REFUTED"
            else:
                final_stance = "NOT ENOUGH INFO"
        
        # Choose top citations
        stance_idx_map = {"SUPPORTED": 2, "REFUTED": 0, "NOT ENOUGH INFO": 1}
        stance_idx = stance_idx_map.get(final_stance, 1)
        top_citations = self._choose_by_weight_and_label(stance_results, stance_idx, k=3)
        citations = self._format_citations(top_citations)
        
        # Extract evidence snippets
        evidence_snippets = self._extract_evidence_snippets(stance_results)
        
        # Calculate overall confidence
        overall_confidence = max(weighted_votes.values()) if weighted_votes else 0.0
        
        return {
            "label": final_stance,
            "verdict": confidence_badge,
            "probability": overall_confidence,
            "confidence_badge": confidence_badge,
            "reasoning": reasoning,
            "weighted_votes": weighted_votes,
            "fact_check_override": fact_check_override,
            "scientific_consensus_override": scientific_consensus_override,
            "no_evidence_case": no_evidence_case,
            "citations": citations,
            "evidence_snippets": [
                {
                    "sentence": snippet.sentence,
                    "stance": snippet.stance,
                    "confidence": snippet.confidence,
                    "reasoning": snippet.reasoning,
                    "source": snippet.source,
                    "source_type": snippet.source_type
                } for snippet in evidence_snippets
            ],
            "evidence_count": len(stance_results),
            "stance_distribution": {
                "support": weighted_votes.get("SUPPORTED", 0.0),
                "refute": weighted_votes.get("REFUTED", 0.0),
                "unclear": weighted_votes.get("NOT ENOUGH INFO", 0.0)
            },
            "verdict_logic_applied": {
                "fact_check_override": fact_check_override is not None,
                "scientific_consensus_override": scientific_consensus_override is not None,
                "no_evidence_case": no_evidence_case,
                "weighted_voting": True
            }
        }


def _load_ux_config(config_path: str = "phase4_verification/config/ux_mapping.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {
            "thresholds": {"true_min": 0.7, "unclear_min": 0.4},
            "labels": {
                "likely_true": "Likely True 🟢",
                "unclear": "Unclear 🟡",
                "likely_false": "Likely False 🔴",
            },
            "use_supported_probability": True,
        }


def _choose_by_weight_and_label(results: List[Dict[str, Any]], stance_idx: int, k: int = 3) -> List[Dict[str, Any]]:
    # Use class-specific contribution proxy: w_i * p_i[c]
    def _class_contrib(ev: Dict[str, Any]) -> float:
        w = float(ev.get("w", 0.0)) if "w" in ev else float((ev.get("scores", {}) or {}).get("cross", 0.0))
        p = 0.0
        if isinstance(ev.get("probs"), list) and len(ev["probs"]) >= 3:
            p = float(ev["probs"][stance_idx])
        return w * p
    ranked = sorted(results, key=_class_contrib, reverse=True)
    return ranked[:k]


def _format_citations(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for ev in items:
        out.append({
            "snippet": str(ev.get("snippet", ev.get("text", ""))),
            "url": str(ev.get("url", "")),
        })
    return out


def map_to_verdict(
    calibrated_probabilities: Dict[str, float],
    p_raw: float,
    results: List[Dict[str, Any]],
    config_path: str = "phase4_verification/config/ux_mapping.yaml",
) -> Dict[str, Any]:
    """Map calibrated stance probabilities to a UX verdict and citations.

    Assumes probabilities are across {SUPPORTED, REFUTED, NOT ENOUGH INFO}.
    If use_supported_probability is true, uses P(SUPPORTED) as "truth" probability.
    """
    cfg = _load_ux_config(config_path)
    thresholds = cfg.get("thresholds", {})
    labels = cfg.get("labels", {})
    use_supported = cfg.get("use_supported_probability", True)
    p_true = float(calibrated_probabilities.get("SUPPORTED", 0.0)) if use_supported else float(max(calibrated_probabilities.values() or [0.0]))

    true_min = float(thresholds.get("true_min", 0.7))
    unclear_min = float(thresholds.get("unclear_min", 0.4))

    # decide stance index first
    stance_idx = 2 if use_supported else int(max(range(3), key=lambda i: [float(calibrated_probabilities.get("REFUTED", 0.0)), float(calibrated_probabilities.get("NOT ENOUGH INFO", 0.0)), float(calibrated_probabilities.get("SUPPORTED", 0.0))][i]))
    # discrepancy gates
    delta, jsd_val = discrepancy_metrics(p_raw, [float(calibrated_probabilities.get("REFUTED", 0.0)), float(calibrated_probabilities.get("NOT ENOUGH INFO", 0.0)), float(calibrated_probabilities.get("SUPPORTED", 0.0))], stance_idx)
    gates_delta = float(cfg.get("discrepancy_delta", 0.2))
    gates_jsd = float(cfg.get("discrepancy_jsd", 0.2))

    if (delta > gates_delta) or (jsd_val > gates_jsd):
        verdict_str = labels.get("unclear", "Unclear 🟡")
        stability = "conflict"
    else:
        verdict_str = labels.get("likely_true", "Likely True 🟢") if p_true >= true_min else labels.get("unclear", "Unclear 🟡") if p_true >= unclear_min else labels.get("likely_false", "Likely False 🔴")
        stability = "stable"

    label_map = ["REFUTED", "NOT ENOUGH INFO", "SUPPORTED"]
    label = label_map[stance_idx]

    # choose top-3 by weight
    # Expecting results to include 'weights' or 'scores'
    top3 = _choose_by_weight_and_label(results, stance_idx, k=3)
    citations = _format_citations(top3)

    return {
        "label": label,
        "verdict": verdict_str,
        "p_calibrated_top": round(p_true, 3),
        "p_raw_top": round(p_raw, 3),
        "discrepancy": {"delta": round(delta, 3), "jsd": round(jsd_val, 3), "status": stability},
        "citations": citations,
    }


