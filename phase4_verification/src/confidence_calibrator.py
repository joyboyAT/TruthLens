import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np

import math
import yaml
from datetime import datetime, timezone


logger = logging.getLogger(__name__)

LABELS = ["REFUTED", "NOT ENOUGH INFO", "SUPPORTED"]


@dataclass
class CalibrationResult:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    temperature: float


def _normalize(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in probs.values())
    if total <= 0:
        uniform = 1.0 / len(LABELS)
        return {k: uniform for k in LABELS}
    return {k: max(0.0, float(v)) / total for k, v in probs.items()}


def _average_distributions(dists: List[Dict[str, float]]) -> Dict[str, float]:
    if not dists:
        return {k: 1.0 / len(LABELS) for k in LABELS}
    agg = {k: 0.0 for k in LABELS}
    for d in dists:
        for k in LABELS:
            agg[k] += float(d.get(k, 0.0))
    avg = {k: v / len(dists) for k, v in agg.items()}
    return _normalize(avg)


def _softmax_from_logits(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _apply_temperature(avg_probs: Dict[str, float], temperature: float) -> Dict[str, float]:
    # Convert probs to pseudo-logits via log, apply 1/T, then softmax
    eps = 1e-12
    logits = [math.log(max(eps, avg_probs[k])) for k in LABELS]
    scaled = [x / max(1e-6, temperature) for x in logits]
    sm = _softmax_from_logits(scaled)
    return {k: sm[i] for i, k in enumerate(LABELS)}


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _domain_trust(domain: str, priors: Dict[str, float]) -> float:
    domain = (domain or "").lower()
    # simple exact/substring match fallback
    for key, val in priors.items():
        if key == "default":
            continue
        if "*" in key:
            # very simple glob-like contains
            pattern = key.replace("*", "")
            if pattern and pattern in domain:
                return float(val)
        elif key in domain:
            return float(val)
    return float(priors.get("default", 0.6))


def compute_weights(evidence_item: Dict[str, Any], recency_tau_days: int, domain_priors: Dict[str, float]) -> float:
    scores = evidence_item.get("scores", {}) or {}
    sim = float(scores.get("cross", scores.get("cosine", 0.0)))
    # min-max normalize similarity assuming [0,1] already for cosine/cross; clamp
    sim = max(0.0, min(1.0, sim))

    domain = str(evidence_item.get("domain") or "")
    trust = _domain_trust(domain, domain_priors)

    # recency factor
    recency = 1.0
    try:
        pub = evidence_item.get("published_at")
        if isinstance(pub, str) and pub:
            # try parse ISO format
            published_at = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        elif isinstance(pub, datetime):
            published_at = pub
        else:
            published_at = None
        if published_at is not None:
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - published_at).total_seconds() / (24 * 3600)
            tau = max(1e-6, float(recency_tau_days))
            recency = math.exp(-age_days / tau)
    except Exception:
        recency = 1.0

    return float(sim * trust * recency)


def weighted_logit_fusion(results: List[Dict[str, Any]], recency_tau_days: int, domain_priors: Dict[str, float]) -> Tuple[List[float], List[float]]:
    # results: each has keys: 'logits' (list length 3), and evidence metadata including 'scores','domain','published_at'
    Z = [0.0, 0.0, 0.0]
    weights: List[float] = []
    for r in results:
        w = compute_weights(r, recency_tau_days, domain_priors)
        weights.append(w)
        logits = r.get("logits") or r.get("raw_logits") or [0.0, 0.0, 0.0]
        for i in range(3):
            Z[i] += w * float(logits[i])
    return Z, weights


def temperature_scaling(Z: List[float], T: float) -> List[float]:
    scaled = [z / max(1e-6, T) for z in Z]
    return _softmax_from_logits(scaled)


def compute_selection_weight(item: Dict[str, Any], domain_priors: Dict[str, float], tau: float) -> float:
    scores = item.get("scores", {}) or {}
    sim = float(scores.get("cross", scores.get("cosine", 0.0)))
    sim = max(0.0, min(1.0, sim))
    trust = _domain_trust(str(item.get("domain") or ""), domain_priors)
    rec = 1.0
    if item.get("age_days") is not None:
        try:
            rec = math.exp(- float(item["age_days"]) / float(max(1e-6, tau)))
        except Exception:
            rec = 1.0
    return float(sim * trust * rec)


def stance_aware_logit_fusion(results: List[Dict[str, Any]], domain_priors: Dict[str, float], tau: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    C = 3
    Z = np.zeros(C, dtype=float)
    contrib: List[Dict[str, Any]] = []
    for item in results:
        w_i = compute_selection_weight(item, domain_priors, tau)
        p_i = np.array(item.get("probs", [0.0, 0.0, 0.0]), dtype=float)
        z_i = np.array(item.get("logits", [0.0, 0.0, 0.0]), dtype=float)
        for c in range(C):
            w_ic = w_i * p_i[c]
            Z[c] += w_ic * z_i[c]
        contrib.append({"evidence_id": item.get("evidence_id"), "w": float(w_i), "p": p_i.tolist()})
    return Z, contrib


def temperature_softmax(Z: np.ndarray, T: float) -> np.ndarray:
    Zs = Z / float(T)
    Zs = Zs - Zs.max()
    e = np.exp(Zs)
    return e / e.sum()


def aggregate_scores(results: List[Dict[str, Any]], config_path: str = "config/calibration.yaml", priors_path: str = "config/domain_priors.yaml") -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    priors = _load_yaml(priors_path)
    T = float(cfg.get("temperature", 1.2))
    tau = float(cfg.get("recency_tau_days", 365))
    Z, contrib = stance_aware_logit_fusion(results, priors, tau)
    p_cal = temperature_softmax(Z, T)
    c_star = int(np.argmax(p_cal))
    p_raw = max((float(r.get("probs", [0.0, 0.0, 0.0])[c_star]) for r in results), default=0.0)
    # margin and net evidence
    sorted_idx = np.argsort(-p_cal)
    margin = float(p_cal[sorted_idx[0]] - p_cal[sorted_idx[1]])
    idx_sup, idx_ref = 0, 1
    S = 0.0
    for r in results:
        S += compute_selection_weight(r, priors, tau) * (float(r.get("probs", [0.0, 0.0, 0.0])[idx_sup]) - float(r.get("probs", [0.0, 0.0, 0.0])[idx_ref]))
    return {
        "p_calibrated": p_cal.tolist(),
        "p_raw_for_top": float(max(p_raw, 1e-6)),
        "top_class": c_star,
        "margin": margin,
        "net_evidence": S,
        "contrib": contrib,
    }


def jsd(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def discrepancy_metrics(p_raw: float, p_cal_vec: List[float], c_star: int) -> Tuple[float, float]:
    p_cal = float(p_cal_vec[c_star])
    delta = abs(p_raw - p_cal)
    r = np.full(3, (1 - p_raw) / 2.0)
    r[c_star] = p_raw
    return delta, float(jsd(r.tolist(), p_cal_vec))


 


def calibrate_confidence(
    results: List[Dict[str, float]], temperature: float = 1.5
) -> CalibrationResult:
    """
    Calibrate stance confidence given a list of probability distributions
    over labels {SUPPORTED, REFUTED, NOT ENOUGH INFO}.

    Steps:
      1) Average distributions across evidences
      2) Temperature scaling (higher T → smoother / less confident)
      3) Normalize and pick argmax as final stance
    """
    avg = _average_distributions(results)
    calibrated = _apply_temperature(avg, temperature)
    # Ensure normalization after temperature scaling
    calibrated = _normalize(calibrated)
    # Argmax and confidence
    label = max(calibrated.items(), key=lambda kv: kv[1])[0]
    confidence = float(calibrated[label])
    return CalibrationResult(label=label, confidence=confidence, probabilities=calibrated, temperature=temperature)


