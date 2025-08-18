"""
API tests for TruthLens Retrieval API.
"""

import os
import sys
import pytest

# Ensure package import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.retrieve import app


@pytest.mark.asyncio
async def test_retrieve_evidence_and_detail():
	client = TestClient(app)
	claims = [
		"Climate change is caused by human activities",
		"COVID-19 vaccines reduce severe illness",
		"Renewable energy policy",
	]

	for claim in claims:
		resp = client.post("/retrieve-evidence", json={"claim_text": claim, "lang": "en", "entities": []})
		assert resp.status_code == 200
		data = resp.json()
		assert "top_evidence" in data and isinstance(data["top_evidence"], list)
		assert len(data["top_evidence"]) > 0
		first = data["top_evidence"][0]
		assert "published_at" in first and isinstance(first["published_at"], str)

		# Fetch detail
		evid = first["id"]
		detail = client.get(f"/evidence/{evid}")
		assert detail.status_code == 200
		j = detail.json()
		assert j["id"] == evid
		assert "provenance" in j and isinstance(j["provenance"], list)
