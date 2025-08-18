import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.verdict_mapper import map_to_verdict


def test_top3_citations_by_weight():
    calibrated = {"SUPPORTED": 0.8, "REFUTED": 0.1, "NOT ENOUGH INFO": 0.1}
    p_raw = 0.78
    results = [
        {"snippet": "s1", "url": "u1", "scores": {"cross": 0.9}, "probs": [0.8, 0.1, 0.1]},
        {"snippet": "s2", "url": "u2", "scores": {"cross": 0.5}, "probs": [0.7, 0.2, 0.1]},
        {"snippet": "s3", "url": "u3", "scores": {"cross": 0.8}, "probs": [0.6, 0.3, 0.1]},
        {"snippet": "s4", "url": "u4", "scores": {"cross": 0.2}, "probs": [0.5, 0.4, 0.1]},
    ]
    out = map_to_verdict(calibrated, p_raw, results)
    urls = [c["url"] for c in out["citations"]]
    # Check that we get exactly 3 citations in descending order of contribution
    assert len(urls) == 3
    assert "u1" in urls  # highest cross score
    assert "u3" in urls  # second highest cross score
    assert "u2" in urls  # third highest cross score

