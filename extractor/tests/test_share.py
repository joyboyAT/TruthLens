import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extractor.share import build_share_card


def test_build_share_card(capsys):
	claim = "COVID vaccine kills in 2 years"
	verdict = "🔴 Likely False"
	evidence = {"highlight": "WHO confirms vaccines are safe", "url": "https://pib.gov/some/path"}
	tip = "Fear is used to manipulate. Verify with WHO/CDC."
	text = build_share_card(claim, verdict, evidence, tip)
	print(text)
	captured = capsys.readouterr().out
	assert "Claim: COVID vaccine kills in 2 years" in captured
	assert "Verdict: 🔴 Likely False" in captured
	assert "Evidence: WHO confirms vaccines are safe (pib.gov/" in captured
	assert "Tip: Fear is used to manipulate. Verify with WHO/CDC." in captured
