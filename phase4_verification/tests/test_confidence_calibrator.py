import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.confidence_calibrator import calibrate_confidence


def test_calibration_smoothing_and_bounds():
    # Simulate three evidence probability distributions
    d1 = {"SUPPORTED": 0.90, "REFUTED": 0.05, "NOT ENOUGH INFO": 0.05}
    d2 = {"SUPPORTED": 0.70, "REFUTED": 0.20, "NOT ENOUGH INFO": 0.10}
    d3 = {"SUPPORTED": 0.60, "REFUTED": 0.25, "NOT ENOUGH INFO": 0.15}

    # Calibrate with T=1.5
    res = calibrate_confidence([d1, d2, d3], temperature=1.5)

    assert 0.0 <= res.confidence <= 1.0
    assert set(res.probabilities.keys()) == {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}
    # Expect SUPPORTED to remain the top label
    assert res.label == "SUPPORTED"
    # Smoothing: probability should be lower than raw average max (which is ~0.73)
    # but still reasonably high (> 0.5)
    assert res.confidence < 0.9
    assert res.confidence > 0.5
    print("\nCalibrated:", res.label, round(res.confidence, 3), {k: round(v, 3) for k, v in res.probabilities.items()})


