from __future__ import annotations

from phase6_output_ux.verdict_mapping import map_verdict
from phase6_output_ux.evidence_cards import build_evidence_cards
from phase6_output_ux.cue_badges import generate_cue_badges
from phase6_output_ux.prebunk_card import build_prebunk_card
from phase6_output_ux.share_card import build_share_card

if __name__ == "__main__":
	claim = "COVID vaccine kills in 2 years"
	score = 0.12
	verdict = map_verdict(score)
	evidence = build_evidence_cards([
		{"text": "WHO confirms vaccines are safe", "url": "https://pib.gov/some/path", "source": "PIB"},
	])[0]
	badges = generate_cue_badges(["Fear Appeal", "Conspiracy"])
	card = build_prebunk_card(
		["Fear Appeal", "Conspiracy"],
		{"Fear Appeal": "Fear is used to manipulate. Verify with WHO/CDC.", "Conspiracy": "Conspiracies lack proof. Check transparent sources."},
	)
	share = build_share_card(claim, verdict.split(" — ")[0], evidence, card["tips"][0])
	print("Badges:", badges)
	print("Prebunk Tips:", card["tips"]) 
	print()
	print(share)
