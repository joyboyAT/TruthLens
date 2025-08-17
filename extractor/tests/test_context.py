import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extractor.context import analyze_context

def test_context():
    ctx = analyze_context('vaccines cause X', 'According to an unverified tweet, vaccines might cause X.')
    assert ctx.get('modality') == 'speculative'
    assert ctx.get('attribution') == 'an unverified tweet'
