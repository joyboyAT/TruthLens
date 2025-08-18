import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.stance_classifier import StanceClassifier, classify_stance


@pytest.mark.slow
def test_stance_classifier_basic_refuted():
    claim = "Earth is flat"
    evidence = "NASA shows Earth is spherical with satellite imagery and physics."

    res = classify_stance(claim, evidence)
    print("\nStance:", res.label)
    print("Probabilities:", {k: round(v, 3) for k, v in res.probabilities.items()})
    assert res.label in {"REFUTED", "SUPPORTED", "NOT ENOUGH INFO"}
    # Expect REFUTED to be the top label for this pair
    assert res.label == "REFUTED"

