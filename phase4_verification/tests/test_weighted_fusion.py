import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.confidence_calibrator import weighted_logit_fusion


def test_weighted_logit_fusion_linear():
    priors = {"default": 1.0}
    results = [
        {"domain": "a.com", "scores": {"cross": 1.0}, "logits": [1.0, 0.0, 0.0]},
        {"domain": "b.com", "scores": {"cross": 0.5}, "logits": [0.0, 1.0, 0.0]},
    ]
    Z, w = weighted_logit_fusion(results, recency_tau_days=365, domain_priors=priors)
    # With trust=1 and recency=1, weights equal sim (cross)
    assert len(w) == 2
    assert abs(Z[0] - 1.0) < 1e-6
    assert abs(Z[1] - 0.5) < 1e-6

