import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase6_output_ux.evidence_cards import build_evidence_cards


def test_evidence_cards():
	inp = [
		{"text": "WHO confirms COVID vaccines are safe.", "url": "https://pib.gov/1", "source": "PIB"},
		{"text": "CDC reports updated guidance.", "url": "https://cdc.gov/2", "source": "CDC"},
		{"text": "Lancet publishes new meta-analysis.", "url": "https://thelancet.com/3", "source": "The Lancet"},
	]
	cards = build_evidence_cards(inp)
	assert isinstance(cards, list) and len(cards) == 3
	assert cards[0]["source"] == "PIB"
	assert "highlight" in cards[0]
	assert cards[0]["url"].startswith("https://")
