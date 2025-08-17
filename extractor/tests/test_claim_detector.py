import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['TRUTHLENS_DISABLE_ML']='1'
from extractor.claim_detector import is_claim

def test_claim_detector():
    ok, prob = is_claim('The earth is flat.')
    assert ok and prob > 0.8
    ok2, prob2 = is_claim('What a nice day!')
    assert (not ok2) and prob2 <= 0.6
