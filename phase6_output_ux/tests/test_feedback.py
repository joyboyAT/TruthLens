import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase6_output_ux.feedback import collect_feedback


def test_feedback_local(monkeypatch):
	monkeypatch.setenv('TRUTHLENS_FAKE_FIRESTORE', '1')
	msg = collect_feedback('phase6-claim-1', 'thumbs_up')
	assert 'Feedback recorded' in msg
