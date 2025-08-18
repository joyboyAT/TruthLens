import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase6_output_ux.verdict_mapping import map_verdict


def test_verdict_mapping():
	assert map_verdict(0.8).startswith("🟢 Likely True")
	assert map_verdict(0.55).startswith("🟡 Unclear")
	assert map_verdict(0.2).startswith("🔴 Likely False")
