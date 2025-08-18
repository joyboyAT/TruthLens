"""
Retrieve Pipeline Orchestrator for TruthLens Phase 3.

Flow:
  factcheck → grounded_search → wikipedia → chunk → embed → retrieve → rerank → NLI → score → dedup → cache

Returns: ranked evidence with labels, reasons (NLI/confidence), freshness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from connectors.factcheck import FactCheckAggregator  # type: ignore
from retrieval.grounded_search import make_default_client, GroundedSearcher  # type: ignore
from connectors.wikipedia import WikipediaConnector  # type: ignore
from nlp.chunking import Chunker, ChunkerConfig  # type: ignore
from nlp.embeddings import EmbeddingEncoder  # type: ignore
from retrieval.vector_search import VectorSearcher  # type: ignore
from retrieval.rerank import Reranker  # type: ignore
from reasoning.nli_labeler import NLIClassifier, NLIConfig  # type: ignore
from scoring.evidence_score import score_evidence, ScoreWeights  # type: ignore
from cleaning.dedup import DedupItem, deduplicate, DedupConfig  # type: ignore
from schemas.evidence import RawEvidence, EvidenceMetadata, SourceType, Language, TextChunk  # type: ignore


@dataclass
class PipelineConfig:
	max_search_results: int = 5
	max_chunks_per_doc: int = 8
	rerank_top_k: int = 10
	use_multilingual_nli: bool = True
	chunk_model: str = "intfloat/multilingual-e5-large"
	embedding_model: str = "intfloat/multilingual-e5-large"
	reranker_model: str = "BAAI/bge-reranker-v2-m3"
	weights: ScoreWeights = ScoreWeights(relevance=0.5, freshness=0.3, source=0.2)


class RetrievePipeline:
	"""End-to-end retrieval and labeling pipeline."""

	def __init__(self, cfg: Optional[PipelineConfig] = None) -> None:
		self.cfg = cfg or PipelineConfig()
		# Components
		self.chunker = Chunker(ChunkerConfig(model_name=self.cfg.chunk_model, max_tokens=512, overlap_sentences=1))
		self.encoder = EmbeddingEncoder(model_name=self.cfg.embedding_model)
		self.vec_search = VectorSearcher(self.encoder)
		self.reranker = Reranker(model_name=self.cfg.reranker_model)
		self.nli = NLIClassifier(NLIConfig(use_multilingual=self.cfg.use_multilingual_nli))
		# Searchers
		self.factchecks = FactCheckAggregator()
		client = make_default_client()
		self.searcher = GroundedSearcher(client) if client else None
		self.wikipedia = WikipediaConnector(lang="en")

	def _fetch_sources(self, claim: str, entities: Optional[List[str]]) -> List[RawEvidence]:
		items: List[RawEvidence] = []
		# Fact-checks (best if quick)
		try:
			fc = self.factchecks.search_all_sources([claim], max_results=self.cfg.max_search_results)
			for _, lst in fc.items():
				for (_c, ev, _s) in lst:
					items.append(ev)
		except Exception:
			pass
		# Grounded search
		if self.searcher is not None:
			try:
				items.extend(self.searcher.search(claim, entities=entities, top_k=self.cfg.max_search_results, days=90))
			except Exception:
				pass
		# Wikipedia fallback
		if not items:
			ents = entities or [claim]
			for e in ents[: self.cfg.max_search_results]:
				try:
					w = self.wikipedia.fetch_background(e)
					if w:
						if w.published_at is None:
							w.published_at = w.retrieved_at or datetime.now(timezone.utc)
						items.append(w)
				except Exception:
					pass
		# Deduplicate identical urls
		seen = set()
		uniq: List[RawEvidence] = []
		for ev in items:
			key = (ev.domain, ev.url)
			if key in seen:
				continue
			seen.add(key)
			uniq.append(ev)
		return uniq[: self.cfg.max_search_results]

	def _chunk_embed(self, ev: RawEvidence) -> List[TextChunk]:
		text = ev.full_text or ev.snippet or ev.title or ""
		chunks = self.chunker.chunk(text, base_id=f"{ev.id}", language=Language.ENGLISH)
		chunks = chunks[: self.cfg.max_chunks_per_doc]
		self.encoder.encode_chunks(chunks, normalize=True)
		return chunks

	def _rank_chunks(self, claim: str, chunks: List[TextChunk]) -> List[Tuple[TextChunk, float]]:
		# First-stage: vector cosine
		idx_scores = self.vec_search.search(claim, chunks, top_k=min(self.cfg.rerank_top_k, len(chunks) or 1))
		seed_order = [chunks[i] for i, _ in idx_scores]
		if not seed_order:
			seed_order = chunks
		# Rerank
		return self.reranker.rerank(claim, seed_order, top_k=min(self.cfg.rerank_top_k, len(seed_order)))

	def _label_with_nli(self, claim: str, top_chunks: List[TextChunk]) -> List[Dict[str, Any]]:
		pairs = [(claim, c.text) for c in top_chunks]
		return self.nli.classify_pairs(pairs)

	def orchestrate(self, claim: str, lang: str = "en", entities: Optional[List[str]] = None) -> Dict[str, Any]:
		# 1) gather sources
		sources = self._fetch_sources(claim, entities)
		if not sources:
			return {"results": [], "diagnostics": {"reason": "no_sources"}}

		# 2) chunk + embed per source
		all_chunks: List[Tuple[RawEvidence, List[TextChunk]]] = []
		for ev in sources:
			ch = self._chunk_embed(ev)
			if ch:
				all_chunks.append((ev, ch))

		# 3) rank chunks globally
		global_chunks: List[TextChunk] = []
		ev_for_chunk: Dict[str, RawEvidence] = {}
		for ev, chs in all_chunks:
			for c in chs:
				global_chunks.append(c)
				ev_for_chunk[c.chunk_id] = ev

		ranked_pairs = self._rank_chunks(claim, global_chunks)
		top_chunks = [c for c, s in ranked_pairs]
		rel_scores = {c.chunk_id: float(s) for c, s in ranked_pairs}

		# 4) NLI labeling for top chunks
		nli_preds = self._label_with_nli(claim, top_chunks)
		label_by_chunk: Dict[str, Dict[str, Any]] = {}
		for c, pr in zip(top_chunks, nli_preds):
			label_by_chunk[c.chunk_id] = pr

		# 5) Score at evidence level using top chunk from that evidence
		best_chunk_per_ev: Dict[str, Tuple[TextChunk, float]] = {}
		for c, s in ranked_pairs:
			ev = ev_for_chunk[c.chunk_id]
			if ev.id not in best_chunk_per_ev:
				best_chunk_per_ev[ev.id] = (c, s)

		results_raw: List[Dict[str, Any]] = []
		for ev_id, (chunk, rel) in best_chunk_per_ev.items():
			ev = ev_for_chunk[chunk.chunk_id]
			pred = label_by_chunk.get(chunk.chunk_id, {"label": "neutral", "entail": 0.0, "contradict": 0.0, "neutral": 1.0})
			scores = score_evidence(ev, rel, w=self.cfg.weights)
			results_raw.append({
				"evidence": ev,
				"chunk": chunk,
				"relevance_raw": rel,
				"label": pred["label"],
				"nli": pred,
				"scores": scores,
			})

		# 6) Deduplicate
		dedup_items: List[DedupItem] = []
		for r in results_raw:
			ev: RawEvidence = r["evidence"]
			chunk: TextChunk = r["chunk"]
			score = r["scores"]["final"]
			text = ev.full_text or chunk.text or ev.snippet or ev.title
			dedup_items.append(DedupItem(evidence_id=ev.id, chunk_id=chunk.chunk_id, url=ev.url, text=text, score=score))
		clusters = deduplicate(dedup_items, DedupConfig())

		# Keep representative rows
		rep_ids = set(c.representative.evidence_id for c in clusters)
		final_rows = [r for r in results_raw if r["evidence"].id in rep_ids]
		# Sort by final score desc
		final_rows.sort(key=lambda r: r["scores"]["final"], reverse=True)

		# 7) Prepare output bundle
		bundle: List[Dict[str, Any]] = []
		for rank, r in enumerate(final_rows, start=1):
			ev: RawEvidence = r["evidence"]
			pub = ev.published_at or ev.retrieved_at or datetime.now(timezone.utc)
			bundle.append({
				"rank": rank,
				"id": ev.id,
				"domain": ev.domain,
				"label": r["label"],
				"final_score": r["scores"]["final"],
				"published_at": pub.isoformat(),
				"title": ev.title,
				"reasons": r["nli"],
			})

		diagnostics = {
			"num_sources": len(sources),
			"num_chunks": len(global_chunks),
			"reranked": len(ranked_pairs),
		}
		return {"results": bundle, "diagnostics": diagnostics}
