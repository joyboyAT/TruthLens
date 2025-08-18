from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict
from pathlib import Path


_ALLOWED_CHOICES = {"thumbs_up", "thumbs_down"}
_LOCAL_LOG = Path(__file__).resolve().parents[1] / ".local_feedback.jsonl"


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _write_local(payload: Dict[str, object]) -> None:
	_LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
	with _LOCAL_LOG.open("a", encoding="utf-8") as fh:
		fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def collect_feedback(claim_id: str, user_choice: str) -> str:
	"""Store user feedback for a claim.

	If Firestore is configured (GOOGLE_APPLICATION_CREDENTIALS or FIRESTORE_EMULATOR_HOST),
	attempt to write to collection "feedback". Otherwise, append to a local JSONL log.
	"""
	choice = (user_choice or "").strip().lower()
	if choice not in _ALLOWED_CHOICES:
		raise ValueError(f"user_choice must be one of {_ALLOWED_CHOICES}")

	payload: Dict[str, object] = {
		"claim_id": str(claim_id),
		"user_choice": choice,
		"created_at": _now_iso(),
	}

	# Allow test/local override
	if os.getenv("TRUTHLENS_FAKE_FIRESTORE") == "1":
		_write_local(payload)
		return "Feedback recorded (local)."

	# Try Firestore if configured
	emulator = os.getenv("FIRESTORE_EMULATOR_HOST")
	credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
	if emulator or credentials:
		try:
			from google.cloud import firestore  # type: ignore
			client = firestore.Client()
			client.collection("feedback").add(payload)
			return "Feedback recorded."
		except Exception:
			# Fall back to local log if Firestore write fails
			_write_local(payload)
			return "Feedback recorded (local fallback)."

	# No Firestore configuration, write locally
	_write_local(payload)
	return "Feedback recorded (local)."
