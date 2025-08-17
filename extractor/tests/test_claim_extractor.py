import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['TRUTHLENS_DISABLE_ML']='1'
from extractor.claim_extractor import extract_claim_spans

def test_claim_extractor():
    s = 'The company fired 100 employees last month.'
    spans = extract_claim_spans(s)
    assert any('fired 100 employees last month' in sp.get('text','') for sp in spans)
