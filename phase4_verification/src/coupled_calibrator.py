"""
Coupled Confidence Calibration with Balanced Fusion

This module implements a coupled fusion approach that combines:
1. Bayesian strict fusion (multiplicative)
2. Average fusion (additive) 
3. Variance-based confidence estimation
4. Temperature scaling for calibration

The coupled approach provides more robust confidence estimates
by balancing strict and lenient fusion strategies.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

LABELS = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Apply temperature-scaled softmax to logits."""
    if T <= 0:
        raise ValueError("Temperature must be positive")
    
    # Numerical stability: subtract max before exp
    logits_scaled = (logits - np.max(logits)) / T
    exp_logits = np.exp(logits_scaled)
    return exp_logits / np.sum(exp_logits)


def bayesian_fusion(probs_list: List[np.ndarray]) -> np.ndarray:
    """Strict Bayesian fusion by multiplying probabilities."""
    if not probs_list:
        return np.array([1/3, 1/3, 1/3])
    
    # Start with ones and multiply by each probability distribution
    fused = np.ones_like(probs_list[0])
    for p in probs_list:
        # Clip to avoid numerical issues
        p_clipped = np.clip(p, 1e-9, 1.0)
        fused *= p_clipped
    
    # Normalize
    fused /= np.sum(fused)
    return fused


def coupled_fusion(probs_list: List[np.ndarray], beta: float = 0.4) -> Tuple[np.ndarray, float]:
    """
    Coupled fusion combining Bayesian and average fusion with confidence estimation.
    
    Args:
        probs_list: List of probability distributions
        beta: Coupling parameter (0 = pure Bayesian, 1 = pure average)
    
    Returns:
        Tuple of (fused_probabilities, confidence_score)
    """
    if not probs_list:
        return np.array([1/3, 1/3, 1/3]), 0.0
    
    # Average probabilities
    avg_probs = np.mean(probs_list, axis=0)
    
    # Bayesian strict fusion
    bayes_probs = bayesian_fusion(probs_list)
    
    # Variance-based confidence estimation
    stacked = np.stack(probs_list, axis=0)
    variance = np.var(stacked, axis=0).mean()
    confidence = max(0.0, min(1.0, 1.0 - variance))  # Clamp between 0 and 1
    
    # Coupled probability adjustment
    fused = (1 - beta) * bayes_probs + beta * (confidence * avg_probs)
    
    # Ensure numerical stability
    fused = np.clip(fused, 1e-9, 1.0)  # Avoid zeros
    fused /= np.sum(fused)  # Normalize
    
    return fused, confidence


def compute_evidence_weights(evidence_items: List[Dict[str, Any]], 
                           domain_priors: Dict[str, float],
                           recency_tau_days: float = 365.0) -> List[float]:
    """
    Compute weights for evidence items based on similarity, domain trust, and recency.
    
    Args:
        evidence_items: List of evidence dictionaries with scores, domain, age_days
        domain_priors: Dictionary mapping domains to trust scores
        recency_tau_days: Decay parameter for recency scoring
    
    Returns:
        List of weights for each evidence item
    """
    weights = []
    
    for item in evidence_items:
        # Similarity score (cross-encoder or cosine)
        scores = item.get("scores", {}) or {}
        sim = float(scores.get("cross", scores.get("cosine", 0.0)))
        sim = max(0.0, min(1.0, sim))  # Clamp to [0, 1]
        
        # Domain trust
        domain = str(item.get("domain", "")).lower()
        trust = domain_priors.get(domain, domain_priors.get("default", 0.6))
        
        # Recency factor
        recency = 1.0
        age_days = item.get("age_days")
        if age_days is not None:
            try:
                recency = np.exp(-float(age_days) / max(1e-6, float(recency_tau_days)))
            except Exception:
                recency = 1.0
        
        # Combined weight
        weight = sim * trust * recency
        weights.append(float(weight))
    
    return weights


def calibrate_confidence_coupled(results: List[Dict[str, Any]], 
                                T: float = 1.5, 
                                beta: float = 0.4,
                                domain_priors: Dict[str, float] = None,
                                recency_tau_days: float = 365.0) -> Dict[str, Any]:
    """
    Calibrate confidence using coupled fusion approach.
    
    Args:
        results: List of evidence results with logits/probs
        T: Temperature for softmax scaling
        beta: Coupling parameter for fusion
        domain_priors: Domain trust scores
        recency_tau_days: Recency decay parameter
    
    Returns:
        Dictionary with calibrated probabilities and confidence
    """
    if not results:
        return {
            "SUPPORTED": 1/3,
            "REFUTED": 1/3, 
            "NOT ENOUGH INFO": 1/3,
            "confidence": 0.0
        }
    
    # Default domain priors if not provided
    if domain_priors is None:
        domain_priors = {
            "who.int": 1.0,
            "cdc.gov": 1.0,
            "nih.gov": 0.95,
            "nature.com": 0.90,
            "default": 0.6
        }
    
    # Extract logits and convert to probabilities
    probs_list = []
    for result in results:
        logits = result.get("logits", result.get("raw_logits"))
        if logits is not None:
            probs = softmax(np.array(logits), T)
        else:
            # Fallback to existing probabilities
            probs = np.array(result.get("probs", [1/3, 1/3, 1/3]))
        probs_list.append(probs)
    
    # Apply coupled fusion
    fused_probs, confidence = coupled_fusion(probs_list, beta=beta)
    
    # Apply evidence weighting if available
    if len(results) > 1:
        weights = compute_evidence_weights(results, domain_priors, recency_tau_days)
        # Weight the fused probabilities by evidence quality
        weighted_probs = np.zeros_like(fused_probs)
        total_weight = sum(weights)
        
        if total_weight > 0:
            for i, (probs, weight) in enumerate(zip(probs_list, weights)):
                weighted_probs += (weight / total_weight) * probs
            
            # Blend weighted and fused
            final_probs = 0.7 * fused_probs + 0.3 * weighted_probs
            final_probs = np.clip(final_probs, 1e-9, 1.0)
            final_probs /= np.sum(final_probs)
        else:
            final_probs = fused_probs
    else:
        final_probs = fused_probs
    
    return {
        "SUPPORTED": float(final_probs[0]),
        "REFUTED": float(final_probs[1]),
        "NOT ENOUGH INFO": float(final_probs[2]),
        "confidence": float(confidence),
        "fusion_confidence": float(confidence),
        "evidence_count": len(results)
    }


def calibrate_confidence_weighted_coupled(results: List[Dict[str, Any]], 
                                         T: float = 1.5,
                                         beta: float = 0.4) -> Dict[str, Any]:
    """
    Weighted coupled calibration that considers evidence quality.
    
    This is a simplified version that uses the coupled approach
    with basic weighting based on evidence scores.
    """
    if not results:
        return {
            "SUPPORTED": 1/3,
            "REFUTED": 1/3,
            "NOT ENOUGH INFO": 1/3,
            "confidence": 0.0
        }
    
    # Extract and weight probabilities
    weighted_probs = []
    total_weight = 0.0
    
    for result in results:
        # Get weight from cross-encoder score
        scores = result.get("scores", {}) or {}
        weight = float(scores.get("cross", scores.get("cosine", 0.5)))
        weight = max(0.1, min(1.0, weight))  # Clamp to reasonable range
        
        # Get probabilities
        logits = result.get("logits", result.get("raw_logits"))
        if logits is not None:
            probs = softmax(np.array(logits), T)
        else:
            probs = np.array(result.get("probs", [1/3, 1/3, 1/3]))
        
        weighted_probs.append(weight * probs)
        total_weight += weight
    
    if total_weight > 0:
        # Normalize weighted probabilities
        avg_weighted = np.sum(weighted_probs, axis=0) / total_weight
    else:
        avg_weighted = np.array([1/3, 1/3, 1/3])
    
    # Apply coupled fusion to weighted average
    fused_probs, confidence = coupled_fusion([avg_weighted], beta=beta)
    
    return {
        "SUPPORTED": float(fused_probs[0]),
        "REFUTED": float(fused_probs[1]),
        "NOT ENOUGH INFO": float(fused_probs[2]),
        "confidence": float(confidence),
        "evidence_count": len(results)
    }


# Test function
def test_coupled_calibration():
    """Test the coupled calibration with mock data."""
    # Mock logits
    mock_logits = [
        np.array([2.0, -1.0, 0.0]),
        np.array([1.5, -0.5, 0.2]),
        np.array([1.8, -0.2, 0.1])
    ]
    
    # Convert to results format
    mock_results = [
        {"logits": logits.tolist(), "probs": softmax(logits, 1.0).tolist()}
        for logits in mock_logits
    ]
    
    # Test basic coupled calibration
    result = calibrate_confidence_coupled(mock_results, T=1.5, beta=0.4)
    
    print("=== Coupled Calibration Test ===")
    print("Final Probabilities:", {k: f"{v:.3f}" for k, v in result.items() if k != "weights"})
    print("Confidence:", f"{result['confidence']:.3f}")
    
    # Test weighted coupled calibration
    result_weighted = calibrate_confidence_weighted_coupled(mock_results, T=1.5, beta=0.4)
    
    print("\n=== Weighted Coupled Calibration Test ===")
    print("Final Probabilities:", {k: f"{v:.3f}" for k, v in result_weighted.items() if k not in ["weights", "num_evidence"]})
    print("Confidence:", f"{result_weighted['confidence']:.3f}")
    print("Weights:", [f"{w:.3f}" for w in result_weighted["weights"]])


if __name__ == "__main__":
    test_coupled_calibration()
