"""
Tests for de-duplication & clustering with SimHash and URL canonicalization.
"""

import os
import sys
import pytest

# Ensure package import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cleaning.dedup import DedupItem, deduplicate, DedupConfig


def test_dedup_mirrored_copies_single_cluster():
	text = "Vaccines reduce severe illness and hospitalization. Clinical trials confirm this finding."
	items = [
		DedupItem(evidence_id="e1", chunk_id="c1", url="https://www.reuters.com/article/abc?utm=xyz", text=text, score=0.8),
		DedupItem(evidence_id="e2", chunk_id="c2", url="https://reuters.com/article/abc", text=text + " ", score=0.7),
		DedupItem(evidence_id="e3", chunk_id="c3", url="https://www.reuters.com/article/abc#section", text=text, score=0.9),
	]
	clusters = deduplicate(items, DedupConfig(hamming_threshold=3))
	assert len(clusters) == 1
	cl = clusters[0]
	# representative should be highest score (e3)
	assert cl.representative.evidence_id == "e3"
	# provenance should contain all three URLs
	assert len(cl.provenance_urls) == 3
