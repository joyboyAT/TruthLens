import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extractor.ranker import score_claim

def test_ranker():
    s1 = score_claim('5G towers cause COVID-19.')
    s2 = score_claim('I love pizza.')
    assert s1 > 0.8 and s2 < 0.2
