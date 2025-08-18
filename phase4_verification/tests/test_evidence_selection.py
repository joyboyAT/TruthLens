import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evidence_selection import EvidenceSelector, EvidenceItem


@pytest.mark.slow
def test_select_top_evidence_simple():
	claim = "COVID-19 vaccines cause infertility"
	evidence_docs = [
		EvidenceItem(id="1", title="WHO: COVID-19 vaccines are safe", snippet="No evidence linking vaccines to infertility.", url="https://who.int/a"),
		EvidenceItem(id="2", title="CDC safety update", snippet="Studies show no increased risk of infertility.", url="https://cdc.gov/b"),
		EvidenceItem(id="3", title="Random blog", snippet="I think vaccines are harmful", url="https://blog.example/c"),
		EvidenceItem(id="4", title="Systematic review", snippet="Comprehensive analysis finds no association with infertility.", url="https://nejm.org/d"),
		EvidenceItem(id="5", title="Unrelated topic", snippet="Stock market news.", url="https://news.example/e"),
	]

	selector = EvidenceSelector()
	selected = selector.select_top_evidence(claim, evidence_docs, top_k=3, similarity_min=0.4)

	assert len(selected) <= 3
	# Expect the obviously relevant ones to score higher than unrelated
	ids_by_rank = [ev.id for ev, _ in selected]
	assert "5" not in ids_by_rank  # unrelated should be filtered
	print("\nSelected evidence (id, score):", [(ev.id, round(score, 3)) for ev, score in selected])

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evidence_selection import select_top_evidence


def test_select_top_evidence_top3():
	claim = "COVID-19 vaccines cause infertility"
	evidence = [
		{"id": "e1", "text": "Large studies show COVID-19 vaccines do not cause infertility in men or women."},
		{"id": "e2", "text": "WHO states vaccines are safe and effective; no link to infertility has been found."},
		{"id": "e3", "text": "Anecdotal reports online claimed fertility issues, but controlled trials refute this."},
		{"id": "e4", "text": "The economy grew by 5 percent last quarter according to official estimates."},
		{"id": "e5", "text": "Mask mandates reduced transmission in several regions."},
	]

	results = select_top_evidence(claim, evidence, top_k=3, similarity_min=0.3)
	assert isinstance(results, list)
	assert len(results) <= 3
	# Expect health-related evidence outranks unrelated economic statement
	ids = [r['id'] for r in results]
	assert 'e4' not in ids
	# Similarities should be in descending order
	if len(results) > 1:
		for i in range(1, len(results)):
			assert results[i-1]['similarity'] >= results[i]['similarity']


