import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Phase_7.pipeline.pipeline import run_pipeline


def test_pipeline_run_schema():
	text = "The company fired 100 employees last month."
	out = run_pipeline(text)
	assert isinstance(out, dict) and "results" in out
	if out["results"]:
		it = out["results"][0]
		assert set(["claim","context"]).issubset(it.keys())
		assert "context_info" in it
		assert "evidence" in it and "evidence_cards" in it
		assert "verification" in it and "cue_badges" in it and "prebunk" in it
		assert "share" in it
