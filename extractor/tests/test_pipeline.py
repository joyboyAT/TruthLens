import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['TRUTHLENS_DISABLE_ML']='1'
from extractor.pipeline import process_text

def test_pipeline():
    out = process_text('5G towers cause COVID-19. I love pizza. According to an unverified tweet, vaccines might cause X.')
    assert isinstance(out, list) and all('checkworthiness' in c and 'context' in c for c in out)
