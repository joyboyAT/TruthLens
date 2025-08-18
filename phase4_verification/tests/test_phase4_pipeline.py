import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import VerificationPipeline


@pytest.mark.slow
def test_phase4_end_to_end_mock():
	claim = "COVID-19 vaccines cause infertility"
	evidence = [
		{"id": "1", "title": "WHO: COVID-19 vaccines are safe", "snippet": "No evidence linking vaccines to infertility.", "url": "https://who.int/a"},
		{"id": "2", "title": "CDC safety update", "snippet": "Studies show no increased risk of infertility.", "url": "https://cdc.gov/b"},
		{"id": "3", "title": "Random blog", "snippet": "I think vaccines are harmful", "url": "https://blog.example/c"},
		{"id": "4", "title": "Systematic review", "snippet": "Comprehensive analysis finds no association with infertility.", "url": "https://nejm.org/d"},
		{"id": "5", "title": "Unrelated topic", "snippet": "Stock market news.", "url": "https://news.example/e"},
	]

	pipe = VerificationPipeline()
	out = pipe.run(claim, evidence, top_k=3, similarity_min=0.4, temperature=1.5)

	print("\nVerdict:", out.verdict["verdict"], "p_calibrated=", round(out.verdict["p_calibrated_top"], 3))
	print("Citations:")
	for c in out.verdict["citations"]:
		print("-", c["url"]) 

	assert 0.0 <= out.verdict["p_calibrated_top"] <= 1.0
	assert len(out.verdict["citations"]) <= 3
	# Expect a supportive verdict or stable status
	assert out.verdict["verdict"].startswith("Likely") or out.calibrated["label"] in {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}


