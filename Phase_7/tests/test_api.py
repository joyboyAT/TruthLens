import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient
from Phase_7.app import app


def test_verify_endpoint():
	client = TestClient(app)
	payload = {
		"text": "The company fired 100 employees last month.",
		"evidence": [{"text":"WHO confirms vaccines are safe","url":"https://pib.gov","source":"PIB"}],
		"cues": ["Fear Appeal","Conspiracy"],
		"prebunk_tips": {"Fear Appeal":"Verify with WHO/CDC.","Conspiracy":"Check transparent sources."}
	}
	r = client.post("/verify", json=payload)
	assert r.status_code == 200
	data = r.json()
	assert "results" in data
