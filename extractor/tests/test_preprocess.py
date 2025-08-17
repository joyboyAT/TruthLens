import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extractor.preprocess import normalize_whitespace, split_sentences

def test_preprocess():
    assert normalize_whitespace('  a   b ') == 'a b'
    sents = split_sentences('A. B? C!')
    assert sents == ['A.', 'B?', 'C!']
