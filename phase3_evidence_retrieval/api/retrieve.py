"""
TruthLens Phase 3 - Retrieval API (FastAPI)

Endpoints:
- POST /retrieve-evidence {claim_text, lang, entities[]} → returns top_evidence, diagnostics, audit
- GET /evidence/{id} → detailed view with provenance + chunks
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from retrieval.grounded_search import make_default_client, GroundedSearcher  # type: ignore
from connectors.wikipedia import WikipediaConnector  # type: ignore
from scoring.evidence_score import score_evidence, ScoreWeights  # type: ignore
from schemas.evidence import RawEvidence, EvidenceMetadata, SourceType, Language  # type: ignore


class RetrieveRequest(BaseModel):
	claim_text: str
	lang: Optional[str] = Field(default="en")
	entities: Optional[List[str]] = None
	max_results: int = 5


class EvidenceScoreOut(BaseModel):
	relevance: float
	freshness: float
	source: float
	final: float


class EvidenceOut(BaseModel):
	id: str
	title: str
	url: str
	domain: str
	published_at: str
	support_label: Optional[str] = None
	scores: EvidenceScoreOut


class RetrieveResponse(BaseModel):
	top_evidence: List[EvidenceOut]
	diagnostics: Dict[str, Any]
	audit: Dict[str, Any]


app = FastAPI(title="TruthLens Retrieval API", version="0.1")


# Simple in-memory store (for demo/testing)
class EvidenceStore:
	def __init__(self) -> None:
		self._store: Dict[str, Dict[str, Any]] = {}
	def put(self, ev: RawEvidence, scores: Dict[str, float]) -> None:
		self._store[ev.id] = {
			"evidence": ev,
			"scores": scores,
			"provenance": [ev.url],
			"chunks": [],
		}
	def get(self, evid: str) -> Optional[Dict[str, Any]]:
		return self._store.get(evid)


app.state.evidence_store = EvidenceStore()


def _fallback_published(ev: RawEvidence) -> datetime:
	return ev.published_at or ev.retrieved_at or datetime.now(timezone.utc)


def _to_out(ev: RawEvidence, scores: Dict[str, float]) -> EvidenceOut:
	# Ensure published_at
	pub = _fallback_published(ev)
	return EvidenceOut(
		id=ev.id,
		title=ev.title,
		url=ev.url,
		domain=ev.domain,
		published_at=pub.isoformat(),
		support_label=getattr(ev, "support_label", None),
		scores=EvidenceScoreOut(**scores),
	)


def _search_sources(claim: str, entities: Optional[List[str]], max_results: int) -> List[RawEvidence]:
	client = make_default_client()
	if client is not None:
		searcher = GroundedSearcher(client)
		return searcher.search(claim, entities=entities, top_k=max_results, days=90)
	# Wikipedia fallback (background)
	connector = WikipediaConnector(lang="en")
	items: List[RawEvidence] = []
	ents = entities or [claim]
	for e in ents[:max_results]:
		ev = connector.fetch_background(e)
		if ev:
			# Ensure published_at exists for response contracts
			if ev.published_at is None:
				# Use retrieved_at as published surrogate
				ev.published_at = ev.retrieved_at or datetime.now(timezone.utc)
			items.append(ev)
	return items


@app.post("/retrieve-evidence", response_model=RetrieveResponse)
async def retrieve_evidence(req: RetrieveRequest) -> RetrieveResponse:
	claim = req.claim_text.strip()
	if not claim:
		raise HTTPException(status_code=400, detail="claim_text is required")

	# Search
	evidence_items = _search_sources(claim, req.entities, max_results=req.max_results)

	# Score & store
	weights = ScoreWeights(relevance=0.5, freshness=0.3, source=0.2)
	out_items: List[EvidenceOut] = []
	sources_used: List[str] = []
	publish_dates: List[str] = []
	for ev in evidence_items:
		# For demo, set a default relevance from title overlap
		raw_rel = 2.0 if any(tok.lower() in (ev.title or "").lower() for tok in claim.split()[:3]) else 0.5
		scores = score_evidence(ev, raw_rel, w=weights)
		app.state.evidence_store.put(ev, scores)
		out_items.append(_to_out(ev, scores))
		sources_used.append(ev.domain)
		publish_dates.append(_fallback_published(ev).isoformat())

	# Diagnostics/audit
	diagnostics = {
		"sources_queried": list(dict.fromkeys(sources_used)),
		"time_window_days": 90 if make_default_client() is not None else None,
	}
	audit = {
		"published_at_list": publish_dates,
		"count": len(out_items),
	}

	return RetrieveResponse(top_evidence=out_items, diagnostics=diagnostics, audit=audit)


class EvidenceDetailOut(BaseModel):
	id: str
	title: str
	url: str
	domain: str
	published_at: str
	scores: EvidenceScoreOut
	provenance: List[str]
	chunks: List[Dict[str, Any]]


@app.get("/evidence/{evid}", response_model=EvidenceDetailOut)
async def get_evidence_detail(evid: str) -> EvidenceDetailOut:
	rec = app.state.evidence_store.get(evid)
	if not rec:
		raise HTTPException(status_code=404, detail="evidence not found")
	ev: RawEvidence = rec["evidence"]
	scores: Dict[str, float] = rec["scores"]
	prov: List[str] = rec["provenance"]
	chunks: List[Dict[str, Any]] = rec["chunks"]
	pub = _fallback_published(ev)
	return EvidenceDetailOut(
		id=ev.id,
		title=ev.title,
		url=ev.url,
		domain=ev.domain,
		published_at=pub.isoformat(),
		scores=EvidenceScoreOut(**scores),
		provenance=prov,
		chunks=chunks,
	)
