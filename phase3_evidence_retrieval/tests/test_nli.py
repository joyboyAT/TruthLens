"""
Tests for NLI labeling (support/contradict/neutral) with English + Hindi toy pairs.
"""

import os
import sys
import pytest

# Ensure package import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from reasoning.nli_labeler import NLIClassifier, NLIConfig


@pytest.mark.slow
def test_nli_toy_pairs_en_hi():
	# Prefer multilingual for mixed-language support
	clf = NLIClassifier(NLIConfig(use_multilingual=True, batch_size=4))

	pairs = [
		# (claim, evidence_text, expected_label)
		("COVID-19 vaccines reduce severe illness", "Clinical trials show vaccines reduce hospitalization and severe disease.", "entail"),
		("COVID-19 vaccines reduce severe illness", "There is no effect of vaccines on any disease severity.", "contradict"),
		("COVID-19 vaccines reduce severe illness", "Chocolate cake recipe ingredients include flour and sugar.", "neutral"),
		# Hindi
		("टीके गंभीर बीमारी को कम करते हैं", "क्लिनिकल ट्रायल बताते हैं कि टीके अस्पताल में भर्ती और गंभीर बीमारी को घटाते हैं।", "entail"),
		("टीके गंभीर बीमारी को कम करते हैं", "टीकों का गंभीर बीमारी पर कोई प्रभाव नहीं होता।", "contradict"),
		("टीके गंभीर बीमारी को कम करते हैं", "यह एक चॉकलेट केक की रेसिपी है।", "neutral"),
	]

	preds = clf.classify_pairs([(c, e) for c, e, _ in pairs])
	correct = 0
	for (c, e, exp), pr in zip(pairs, preds):
		print(f"Claim: {c}")
		print(f"Evidence: {e}")
		print(f"Pred: {pr['label']} (entail={pr['entail']:.3f}, contradict={pr['contradict']:.3f}, neutral={pr['neutral']:.3f})")
		print()
		if pr["label"] == exp:
			correct += 1

	assert correct >= 4, f"Expected ≥4/6 correct, got {correct}/6"
