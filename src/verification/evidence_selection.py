import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import yaml
import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
try:
    from rank_bm25 import BM25Okapi
except Exception:  # Optional dependency
    BM25Okapi = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    id: str
    title: str
    snippet: str
    url: str
    domain: Optional[str] = None
    published_at: Optional[str] = None
    full_text: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)


def _load_calib(path: str = "config/calibration.yaml") -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {
            "cosine_floor": 0.60,
            "topk_dense": 100,
            "topk_rrf": 30,
            "topk_cross_encoder": 20,
            "final_k": 3,
        }


def rrf_fuse(rankings: Dict[str, List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion.
    rankings maps method->list of doc_ids by descending score.
    score(doc) = sum_m 1 / (k + rank_m(doc))."""
    scores: Dict[str, float] = {}
    for _name, docs in rankings.items():
        for rank_idx, doc_id in enumerate(docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_idx)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def cross_encoder_rerank(claim: str, docs: List[EvidenceItem], top_k: int) -> List[EvidenceItem]:
    if not docs:
        return []
    text_docs = [(d, (d.snippet or d.full_text or d.title or "")) for d in docs]
    try:
        model = CrossEncoder("cross-encoder/roberta-large-ms-marco")
        pairs = [(claim, txt) for _, txt in text_docs]
        scores = model.predict(pairs).tolist()
        for (d, _), s in zip(text_docs, scores):
            d.scores["cross"] = float(s)
    except Exception as e:
        # Offline fallback: lexical overlap as proxy
        logger.warning(f"Cross-encoder unavailable; using offline overlap heuristic. Error: {e}")
        claim_set = set((claim or "").lower().split())
        for d, txt in text_docs:
            toks = set((txt or "").lower().split())
            inter = len(claim_set & toks)
            denom = max(1, len(claim_set) + len(toks) - inter)
            d.scores["cross"] = float(inter / denom)
    return sorted(docs, key=lambda d: d.scores.get("cross", 0.0), reverse=True)[:top_k]


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return 0.5 * (s[mid - 1] + s[mid])
    return s[mid]


class EvidenceSelector:
    """Selects evidence snippets by fused ranking (dense + BM25) and cross-encoder reranking."""

    def __init__(self, model_name: str = "sentence-transformers/all-roberta-large-v1") -> None:
        self._offline = os.environ.get("TRUTHLENS_FORCE_OFFLINE", "0") == "1"
        if self._offline:
            self.model = None
            logger.warning("TRUTHLENS_FORCE_OFFLINE=1 → using offline embedding stub")
        else:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                # Offline fallback: lightweight, deterministic embedding via hashing
                self.model = None
                self._offline = True
                logger.warning(f"Failed to load embedding model '{model_name}'. Falling back to offline stub. Error: {e}")

    def _cheap_embed(self, texts: List[str], dim: int = 384) -> np.ndarray:
        vecs = np.zeros((len(texts), dim), dtype=float)
        for i, t in enumerate(texts):
            tokens = (t or "").lower().split()
            for tok in tokens:
                h = hash(tok) % dim
                vecs[i, h] += 1.0
            n = np.linalg.norm(vecs[i])
            if n > 0:
                vecs[i] = vecs[i] / n
        return vecs

    def _evidence_text(self, ev: EvidenceItem) -> str:
        return (ev.snippet or ev.full_text or ev.title or "").strip()

    def select_top_evidence(
        self,
        claim: str,
        evidence_list: List[EvidenceItem],
        top_k: int = 3,
        similarity_min: float = 0.6,
    ) -> List[Tuple[EvidenceItem, float]]:
        if not claim or not evidence_list:
            return []

        cfg = _load_calib()
        cosine_floor = float(cfg.get("cosine_floor", similarity_min))
        topk_rrf = int(cfg.get("topk_rrf", 30))
        topk_cross = int(cfg.get("topk_cross_encoder", 20))
        final_k = int(cfg.get("final_k", top_k))

        texts = [self._evidence_text(ev) for ev in evidence_list]
        nonempty_idxs = [i for i, t in enumerate(texts) if t]
        if not nonempty_idxs:
            return []

        # Dense cosine (offline-friendly)
        if self._offline:
            claim_emb = self._cheap_embed([claim])
            cand_embs = self._cheap_embed([texts[i] for i in nonempty_idxs])
        else:
            claim_emb = self.model.encode([claim], normalize_embeddings=True)
            cand_embs = self.model.encode([texts[i] for i in nonempty_idxs], normalize_embeddings=True)
        sims = cosine_similarity(claim_emb, cand_embs)[0]
        id_by_idx = {i: evidence_list[i].id for i in nonempty_idxs}
        cosine_map: Dict[str, float] = {id_by_idx[i]: float(sims[j]) for j, i in enumerate(nonempty_idxs)}
        dense_ranked_ids = [doc_id for doc_id, _ in sorted(cosine_map.items(), key=lambda x: x[1], reverse=True)]

        # BM25 (optional)
        bm25_scores: Dict[str, float] = {}
        bm25_ranked_ids: List[str] = []
        if BM25Okapi is not None:
            tokenized_corpus = [(texts[i] or "").lower().split() for i in nonempty_idxs]
            bm25 = BM25Okapi(tokenized_corpus)
            query_tokens = (claim or "").lower().split()
            scores = bm25.get_scores(query_tokens)
            for j, i in enumerate(nonempty_idxs):
                bm25_scores[id_by_idx[i]] = float(scores[j])
            bm25_ranked_ids = [doc_id for doc_id, _ in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)]

        # RRF fusion
        rankings = {"dense": dense_ranked_ids}
        if bm25_ranked_ids:
            rankings["bm25"] = bm25_ranked_ids
        fused_ids = rrf_fuse(rankings, k=60)[:topk_rrf]
        # Attach RRF and component scores
        rrf_positions = {doc_id: rank for rank, doc_id in enumerate(fused_ids, start=1)}

        # Prepare candidate docs
        id_to_item = {ev.id: ev for ev in evidence_list}
        candidates: List[EvidenceItem] = []
        for doc_id in fused_ids:
            ev = id_to_item.get(doc_id)
            if not ev:
                continue
            ev.scores["cosine"] = float(cosine_map.get(doc_id, 0.0))
            if bm25_scores:
                ev.scores["bm25"] = float(bm25_scores.get(doc_id, 0.0))
            ev.scores["rrf"] = 1.0 / (60 + rrf_positions[doc_id])
            candidates.append(ev)

        # Cross-encoder rerank
        reranked = cross_encoder_rerank(claim, candidates, top_k=topk_cross)

        # Dynamic median filter on cross scores
        cross_scores = [ev.scores.get("cross", 0.0) for ev in reranked]
        cross_med = _median(cross_scores)

        # Keep if (cosine >= floor) OR (cross >= median)
        filtered = [
            ev for ev in reranked
            if (ev.scores.get("cosine", 0.0) >= cosine_floor) or (ev.scores.get("cross", 0.0) >= cross_med)
        ]

        # Final top-k and return with cosine score as the tuple's float (no padding)
        final = sorted(filtered, key=lambda d: (d.scores.get("cross", 0.0), d.scores.get("cosine", 0.0)), reverse=True)[:final_k]
        return [(ev, ev.scores.get("cosine", 0.0)) for ev in final]


def drop_neutral_evidence(nli_probs: List[float], cfg_path: str = "config/calibration.yaml") -> bool:
    cfg = _load_calib(cfg_path)
    thr = float(cfg.get("nei_max_prob_keep", 0.45))
    m = max(float(x) for x in (nli_probs or [])) if nli_probs else 0.0
    return m < thr


def select_top_evidence(
    claim: str,
    evidence_list: List[Dict[str, Any]],
    top_k: int = 3,
    similarity_min: float = 0.6,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> List[Dict[str, Any]]:
    """Convenience functional wrapper.

    Input evidence items can be dicts with keys: id, title, snippet, url, domain, published_at, full_text
    Returns list of dicts with added 'similarity'.
    """
    selector = EvidenceSelector(model_name=model_name)
    items = [
        EvidenceItem(
            id=str(d.get("id", i)),
            title=str(d.get("title", "")),
            snippet=str(d.get("snippet", "")),
            url=str(d.get("url", "")),
            domain=d.get("domain"),
            published_at=d.get("published_at"),
            full_text=d.get("full_text"),
        )
        for i, d in enumerate(evidence_list)
    ]
    selected = selector.select_top_evidence(claim, items, top_k=top_k, similarity_min=similarity_min)
    return [
        {
            "id": ev.id,
            "title": ev.title,
            "snippet": ev.snippet,
            "url": ev.url,
            "domain": ev.domain,
            "published_at": ev.published_at,
            "full_text": ev.full_text,
            "similarity": score,
            "scores": ev.scores,
        }
        for ev, score in selected
    ]

