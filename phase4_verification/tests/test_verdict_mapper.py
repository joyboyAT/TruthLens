import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.verdict_mapper import map_to_verdict


def test_verdict_mapper_rules_and_citations():
    calibrated = {"SUPPORTED": 0.75, "REFUTED": 0.15, "NOT ENOUGH INFO": 0.10}
    p_raw = 0.70
    evidence = [
        {"title": "WHO report", "snippet": "Evidence supports claim.", "url": "https://who.int/a", "scores": {"cross": 0.9}, "probs": [0.75, 0.15, 0.10]},
        {"title": "CDC study", "snippet": "High quality data.", "url": "https://cdc.gov/b", "scores": {"cross": 0.8}, "probs": [0.70, 0.20, 0.10]},
        {"title": "Reuters", "snippet": "News coverage.", "url": "https://reuters.com/c", "scores": {"cross": 0.7}, "probs": [0.65, 0.25, 0.10]},
        {"title": "Extra", "snippet": "Should be ignored.", "url": "https://example.com/d", "scores": {"cross": 0.3}, "probs": [0.50, 0.40, 0.10]},
    ]

    v = map_to_verdict(calibrated, p_raw, evidence)
    print("\nVerdict:", v["verdict"], "P=", round(v["p_calibrated_top"], 3))
    assert v["verdict"].startswith("Likely True")
    assert v["label"] == "SUPPORTED"
    assert 0.0 <= v["p_calibrated_top"] <= 1.0
    assert len(v["citations"]) == 3
    assert v["citations"][0]["url"] == "https://who.int/a"


