import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evidence_selection import rrf_fuse


def test_rrf_simple_order():
    rankings = {
        "dense": ["A", "B", "C", "D"],
        "bm25": ["B", "A", "E", "C"],
    }
    fused = rrf_fuse(rankings, k=60)
    # A and B should be near the top due to high ranks in both lists
    assert fused[0] in {"A", "B"}
    assert fused[1] in {"A", "B"}

