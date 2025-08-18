import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extractor.feedback import collect_feedback


def test_collect_feedback_local(tmp_path, monkeypatch):
	# Route local log to a temp file
	log_file = tmp_path / "_fb.jsonl"
	monkeypatch.setenv("TRUTHLENS_FAKE_FIRESTORE", "1")
	# Patch module-level path by shadowing environment; implementation uses env flag
	msg = collect_feedback("claim-123", "thumbs_up")
	assert "Feedback recorded" in msg
	# When TRUTHLENS_FAKE_FIRESTORE=1, writing goes to default local log path.
	# We can't easily inject the path without changing the code; just assert function returns.
	# Optionally, call again with invalid choice to ensure validation works
	try:
		collect_feedback("claim-123", "invalid")
		assert False, "Expected ValueError"
	except ValueError:
		pass

    
