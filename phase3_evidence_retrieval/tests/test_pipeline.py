"""
End-to-end pipeline test for TruthLens Phase 3.
"""

import os
import sys
import pytest

# Ensure package import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline.retrieve_pipeline import RetrievePipeline


@pytest.mark.slow
def test_pipeline_end_to_end_print_table():
	pipe = RetrievePipeline()
	claims = [
		"Climate change is caused by human activities",
		"COVID-19 vaccines reduce severe illness",
		"Renewable energy reduces carbon emissions",
		"GDP growth in India exceeded 8% last quarter",
		"Malaria vaccine approved by WHO",
	]
	for cl in claims:
		res = pipe.orchestrate(cl)
		rows = res.get("results", [])
		print("\nrank | domain | label | final_score | published_at | title")
		for r in rows[:5]:
			print(f"{r['rank']} | {r['domain']} | {r['label']} | {r['final_score']:.3f} | {r['published_at']} | {r['title'][:60]}")
		# Not asserting heavy end-to-end guarantees; minimal sanity check
		assert isinstance(rows, list)
