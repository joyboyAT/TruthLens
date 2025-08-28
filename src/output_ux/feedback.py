from __future__ import annotations

from typing import Literal

from extractor.feedback import collect_feedback as _collect


def collect_feedback(claim_id: str, user_choice: Literal["thumbs_up", "thumbs_down"]) -> str:
	return _collect(claim_id, user_choice)
