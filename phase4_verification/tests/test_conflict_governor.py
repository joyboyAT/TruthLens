import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.verdict_mapper import map_to_verdict


def test_conflict_gate_cases():
    # Case A: near agreement → not forced to yellow
    calibrated = {"SUPPORTED": 0.74, "REFUTED": 0.10, "NOT ENOUGH INFO": 0.16}
    p_raw = 0.70
    results = [
        {"snippet": "a", "url": "u1", "scores": {"cross": 0.9}},
        {"snippet": "b", "url": "u2", "scores": {"cross": 0.8}},
        {"snippet": "c", "url": "u3", "scores": {"cross": 0.7}},
    ]
    out = map_to_verdict(calibrated, p_raw, results)
    assert out["verdict"].startswith("Likely True") or out["discrepancy"]["status"] == "stable"

    # Case B: conflict → force yellow
    calibrated = {"SUPPORTED": 0.62, "REFUTED": 0.20, "NOT ENOUGH INFO": 0.18}
    p_raw = 0.18
    out2 = map_to_verdict(calibrated, p_raw, results)
    assert out2["verdict"].startswith("Unclear")
    assert out2["discrepancy"]["status"] == "conflict"

