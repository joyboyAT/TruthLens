"""
Tests for coupled confidence calibration.
"""

import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.coupled_calibrator import (
    softmax, bayesian_fusion, coupled_fusion, 
    calibrate_confidence_coupled, calibrate_confidence_weighted_coupled,
    compute_evidence_weights
)


def test_softmax():
    """Test softmax function with temperature scaling."""
    logits = np.array([1.0, 2.0, 0.5])
    
    # Test with T=1.0
    probs = softmax(logits, T=1.0)
    assert np.allclose(probs.sum(), 1.0)
    assert probs[1] > probs[0] > probs[2]  # 2.0 > 1.0 > 0.5
    
    # Test with T=2.0 (smoother)
    probs_smooth = softmax(logits, T=2.0)
    assert np.allclose(probs_smooth.sum(), 1.0)
    # Should be more uniform with higher temperature
    assert np.std(probs_smooth) < np.std(probs)


def test_bayesian_fusion():
    """Test Bayesian fusion of probability distributions."""
    probs1 = np.array([0.8, 0.1, 0.1])
    probs2 = np.array([0.7, 0.2, 0.1])
    probs3 = np.array([0.6, 0.3, 0.1])
    
    fused = bayesian_fusion([probs1, probs2, probs3])
    assert np.allclose(fused.sum(), 1.0)
    # Should favor the class with highest agreement
    assert fused[0] > fused[1] > fused[2]


def test_coupled_fusion():
    """Test coupled fusion combining Bayesian and average approaches."""
    probs1 = np.array([0.8, 0.1, 0.1])
    probs2 = np.array([0.7, 0.2, 0.1])
    probs3 = np.array([0.6, 0.3, 0.1])
    
    fused, confidence = coupled_fusion([probs1, probs2, probs3], beta=0.4)
    assert np.allclose(fused.sum(), 1.0)
    assert 0.0 <= confidence <= 1.0
    # Should still favor the first class
    assert fused[0] > fused[1] > fused[2]


def test_calibrate_confidence_coupled():
    """Test coupled confidence calibration."""
    results = [
        {"logits": [2.0, -1.0, 0.0]},
        {"logits": [1.5, -0.5, 0.2]},
        {"logits": [1.8, -0.2, 0.1]}
    ]
    
    result = calibrate_confidence_coupled(results, T=1.5, beta=0.4)
    
    assert "SUPPORTED" in result
    assert "REFUTED" in result
    assert "NOT ENOUGH INFO" in result
    assert "confidence" in result
    
    # Check probabilities sum to 1
    total_prob = sum(result[label] for label in ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"])
    assert abs(total_prob - 1.0) < 1e-6
    
    # Should favor SUPPORTED given the logits
    assert result["SUPPORTED"] > result["REFUTED"]
    assert result["SUPPORTED"] > result["NOT ENOUGH INFO"]


def test_calibrate_confidence_weighted_coupled():
    """Test weighted coupled calibration."""
    results = [
        {"logits": [2.0, -1.0, 0.0], "scores": {"cross": 0.9}},
        {"logits": [1.5, -0.5, 0.2], "scores": {"cross": 0.7}},
        {"logits": [1.8, -0.2, 0.1], "scores": {"cross": 0.8}}
    ]
    
    result = calibrate_confidence_weighted_coupled(results, T=1.5, beta=0.4)
    
    assert "SUPPORTED" in result
    assert "REFUTED" in result
    assert "NOT ENOUGH INFO" in result
    assert "confidence" in result
    assert "evidence_count" in result
    
    # Check probabilities sum to 1
    total_prob = sum(result[label] for label in ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"])
    assert abs(total_prob - 1.0) < 1e-6
    
    assert result["evidence_count"] == 3


def test_compute_evidence_weights():
    """Test evidence weight computation."""
    evidence_items = [
        {"scores": {"cross": 0.9}, "domain": "who.int", "age_days": 10},
        {"scores": {"cross": 0.7}, "domain": "blog.example.com", "age_days": 100},
        {"scores": {"cross": 0.8}, "domain": "cdc.gov", "age_days": 30}
    ]
    
    domain_priors = {
        "who.int": 1.0,
        "cdc.gov": 1.0,
        "default": 0.6
    }
    
    weights = compute_evidence_weights(evidence_items, domain_priors, recency_tau_days=365.0)
    
    assert len(weights) == 3
    assert all(0.0 <= w <= 1.0 for w in weights)
    # WHO should have highest weight (high score + high trust + recent)
    assert weights[0] > weights[2] > weights[1]


def test_edge_cases():
    """Test edge cases."""
    # Empty results
    result = calibrate_confidence_coupled([], T=1.5, beta=0.4)
    assert result["SUPPORTED"] == 1/3
    assert result["REFUTED"] == 1/3
    assert result["NOT ENOUGH INFO"] == 1/3
    assert result["confidence"] == 0.0
    
    # Single result
    single_result = [{"logits": [1.0, 0.0, 0.0]}]
    result = calibrate_confidence_coupled(single_result, T=1.0, beta=0.4)
    assert result["SUPPORTED"] > 0.5  # Should favor SUPPORTED
    
    # Zero temperature (should raise error)
    with pytest.raises(ValueError):
        softmax(np.array([1.0, 2.0, 0.5]), T=0.0)
    
    # Negative temperature (should raise error)
    with pytest.raises(ValueError):
        softmax(np.array([1.0, 2.0, 0.5]), T=-1.0)
