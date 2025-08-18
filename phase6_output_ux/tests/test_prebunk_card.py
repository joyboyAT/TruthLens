import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase6_output_ux.prebunk_card import build_prebunk_card


def test_prebunk_card():
	cues = ["Fear Appeal", "Conspiracy"]
	tips = {
		"Fear Appeal": "Fear is used to manipulate. Verify with WHO/CDC.",
		"Conspiracy": "Conspiracies lack proof. Check transparent sources.",
	}
	card = build_prebunk_card(cues, tips)
	assert card["title"] == "Prebunk Tips"
	assert card["tips"] == [
		"Fear is used to manipulate. Verify with WHO/CDC.",
		"Conspiracies lack proof. Check transparent sources.",
	]
