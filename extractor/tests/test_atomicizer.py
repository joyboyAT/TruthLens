import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extractor.atomicizer import to_atomic

def test_atomicizer():
    claims = to_atomic('A bought B and C in 2020', None)
    texts = [c.get('text','').strip().lower() for c in claims]
    assert 'a bought b in 2020' in texts and 'a bought c in 2020' in texts
