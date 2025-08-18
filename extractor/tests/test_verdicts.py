import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extractor.verdicts import map_verdict


def test_map_verdict():
	assert map_verdict(0.8).startswith("🟢 Likely True")
	assert map_verdict(0.55).startswith("🟡 Unclear")
	assert map_verdict(0.2).startswith("🔴 Likely False")
