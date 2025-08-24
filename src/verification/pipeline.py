import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .evidence_selection import EvidenceSelector, EvidenceItem, drop_neutral_evidence
from .stance_classifier import StanceClassifier
from .confidence_calibrator import aggregate_scores
from .verdict_mapper import map_to_verdict


logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    claim: str
    selected_evidence: List[Dict[str, Any]]
    stance_results: List[Dict[str, Any]]
    calibrated: Dict[str, Any]
    verdict: Dict[str, Any]


class VerificationPipeline:
    """Phase 4 end-to-end: select evidence → NLI → calibrate → verdict mapping."""

    def __init__(self, emb_model: str = "sentence-transformers/all-roberta-large-v1", nli_model: str = "roberta-large-mnli") -> None:
        self.selector = EvidenceSelector(model_name=emb_model)
        self.nli = StanceClassifier(model_name=nli_model)

    def run(self, claim: str, evidence_list: List[Dict[str, Any]], top_k: int = 3, similarity_min: float = 0.6, temperature: float = 1.5) -> PipelineOutput:
        # 1) Evidence selection
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
        selected = self.selector.select_top_evidence(claim, items, top_k=top_k, similarity_min=similarity_min)
        selected_dicts: List[Dict[str, Any]] = []
        for ev, score in selected:
            selected_dicts.append({
                "id": ev.id,
                "title": ev.title,
                "snippet": ev.snippet,
                "url": ev.url,
                "domain": ev.domain,
                "published_at": ev.published_at,
                "full_text": ev.full_text,
                "similarity": score,
            })

        # 2) Stance classification (premise=evidence, hypothesis=claim)
        texts = [d.get("snippet") or d.get("full_text") or d.get("title") or "" for d in selected_dicts]
        nli_results = self.nli.classify_batch(
            claim,
            texts,
            evidence_ids=[d.get("id") for d in selected_dicts],
            evidence_scores=[d.get("scores") for d in selected_dicts],
        )
        # Build standardized results with ordered logits/probs: [SUP, REF, NEI]
        stance_results: List[Dict[str, Any]] = []
        for d, r in zip(selected_dicts, nli_results):
            # roberta-large-mnli: logits order [ref, nei, sup]
            z_ref, z_nei, z_sup = (r.raw_logits or [0.0, 0.0, 0.0])
            # probs dict to vector in [sup, ref, nei]
            probs_dict = r.probabilities
            p_sup = float(probs_dict.get("SUPPORTED", 0.0))
            p_ref = float(probs_dict.get("REFUTED", 0.0))
            p_nei = float(probs_dict.get("NOT ENOUGH INFO", 0.0))
            # derive age_days if possible (published_at ISO)
            age_days = None
            try:
                import datetime as _dt
                pub = d.get("published_at")
                if isinstance(pub, str) and pub:
                    ts = _dt.datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=_dt.timezone.utc)
                    age_days = int(((_dt.datetime.now(_dt.timezone.utc) - ts).total_seconds()) // (24 * 3600))
            except Exception:
                age_days = None
            stance_results.append({
                "evidence_id": d["id"],
                "label": r.label,
                "logits": [z_sup, z_ref, z_nei],
                "probs": [p_sup, p_ref, p_nei],
                "scores": d.get("scores", {}),
                "domain": d.get("domain"),
                "age_days": age_days,
                "published_at": d.get("published_at"),
                "url": d["url"],
                "title": d["title"],
                "snippet": d["snippet"],
                "similarity": d["similarity"],
            })

        # Prune neutral evidence if more than 1 remains; keep at least best one
        pruned: List[Dict[str, Any]] = []
        for item in stance_results:
            if drop_neutral_evidence(item["probs"]):
                continue
            pruned.append(item)
        if pruned:
            stance_results = pruned
        else:
            # keep best one by cross score if all neutral
            stance_results = sorted(stance_results, key=lambda x: (x.get("scores", {}).get("cross", 0.0), x.get("scores", {}).get("cosine", 0.0)), reverse=True)[:1]

        # 3) Weighted fusion + calibration (with optional coupled calibration)
        if hasattr(self, 'use_coupled_calibration') and self.use_coupled_calibration:
            from .coupled_calibrator import calibrate_confidence_coupled
            coupled_result = calibrate_confidence_coupled(stance_results, T=temperature, beta=0.4)
            calibrated_probs = {
                "SUPPORTED": float(coupled_result["SUPPORTED"]),
                "REFUTED": float(coupled_result["REFUTED"]),
                "NOT ENOUGH INFO": float(coupled_result["NOT ENOUGH INFO"]),
            }
            # Find top class
            top_class = max(range(3), key=lambda i: [calibrated_probs["SUPPORTED"], calibrated_probs["REFUTED"], calibrated_probs["NOT ENOUGH INFO"]][i])
            calibrated = {
                "label": ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"][top_class],
                "confidence": float(coupled_result["confidence"]),
                "probabilities": calibrated_probs,
                "temperature": temperature,
                "beta": 0.4,
            }
            # Create mock agg for compatibility
            agg = {
                "p_raw_for_top": float(max((r.get("probs", [0.0, 0.0, 0.0])[top_class] for r in stance_results), default=0.0)),
                "margin": 0.0,
                "net_evidence": 0.0,
                "contrib": []
            }
        else:
            agg = aggregate_scores(stance_results)
            calibrated_vec = agg["p_calibrated"]  # [SUP, REF, NEI]
            # Convert to dict keyed by labels
            calibrated_probs = {
                "SUPPORTED": float(calibrated_vec[0]),
                "REFUTED": float(calibrated_vec[1]),
                "NOT ENOUGH INFO": float(calibrated_vec[2]),
            }
            calibrated = {
                "label": ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"][int(max(range(3), key=lambda i: calibrated_vec[i]))],
                "confidence": float(max(calibrated_vec)),
                "probabilities": calibrated_probs,
                "temperature": None,
            }

        # 4) Verdict mapping
        verdict = map_to_verdict(calibrated["probabilities"], agg["p_raw_for_top"], stance_results)
        # enrich with margin/net_evidence and contrib
        verdict["margin"] = round(float(agg.get("margin", 0.0)), 3)
        verdict["net_evidence"] = round(float(agg.get("net_evidence", 0.0)), 3)
        verdict.setdefault("explanations", {})
        verdict["explanations"]["notes"] = ["stance-aware fusion", f"temperature={temperature}"]
        if hasattr(self, 'use_coupled_calibration') and self.use_coupled_calibration:
            verdict["explanations"]["notes"].append("coupled-calibration")
        verdict["explanations"]["contrib_top"] = agg.get("contrib", [])[:3]

        return PipelineOutput(
            claim=claim,
            selected_evidence=selected_dicts,
            stance_results=stance_results,
            calibrated=calibrated,
            verdict=verdict,
        )


