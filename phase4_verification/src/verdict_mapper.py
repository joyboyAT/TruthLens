import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .confidence_calibrator import discrepancy_metrics


logger = logging.getLogger(__name__)


@dataclass
class Citation:
    title: str
    snippet: str
    url: str


@dataclass
class Verdict:
    verdict: str
    probability: float
    label: str
    citations: List[Citation]


def _load_ux_config(config_path: str = "phase4_verification/config/ux_mapping.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {
            "thresholds": {"true_min": 0.7, "unclear_min": 0.4},
            "labels": {
                "likely_true": "Likely True 🟢",
                "unclear": "Unclear 🟡",
                "likely_false": "Likely False 🔴",
            },
            "use_supported_probability": True,
        }


def _choose_by_weight_and_label(results: List[Dict[str, Any]], stance_idx: int, k: int = 3) -> List[Dict[str, Any]]:
    # Use class-specific contribution proxy: w_i * p_i[c]
    def _class_contrib(ev: Dict[str, Any]) -> float:
        w = float(ev.get("w", 0.0)) if "w" in ev else float((ev.get("scores", {}) or {}).get("cross", 0.0))
        p = 0.0
        if isinstance(ev.get("probs"), list) and len(ev["probs"]) >= 3:
            p = float(ev["probs"][stance_idx])
        return w * p
    ranked = sorted(results, key=_class_contrib, reverse=True)
    return ranked[:k]


def _format_citations(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for ev in items:
        out.append({
            "snippet": str(ev.get("snippet", ev.get("text", ""))),
            "url": str(ev.get("url", "")),
        })
    return out


def map_to_verdict(
    calibrated_probabilities: Dict[str, float],
    p_raw: float,
    results: List[Dict[str, Any]],
    config_path: str = "phase4_verification/config/ux_mapping.yaml",
) -> Dict[str, Any]:
    """Map calibrated stance probabilities to a UX verdict and citations.

    Assumes probabilities are across {SUPPORTED, REFUTED, NOT ENOUGH INFO}.
    If use_supported_probability is true, uses P(SUPPORTED) as "truth" probability.
    """
    cfg = _load_ux_config(config_path)
    thresholds = cfg.get("thresholds", {})
    labels = cfg.get("labels", {})
    use_supported = cfg.get("use_supported_probability", True)
    p_true = float(calibrated_probabilities.get("SUPPORTED", 0.0)) if use_supported else float(max(calibrated_probabilities.values() or [0.0]))

    true_min = float(thresholds.get("true_min", 0.7))
    unclear_min = float(thresholds.get("unclear_min", 0.4))

    # decide stance index first
    stance_idx = 2 if use_supported else int(max(range(3), key=lambda i: [float(calibrated_probabilities.get("REFUTED", 0.0)), float(calibrated_probabilities.get("NOT ENOUGH INFO", 0.0)), float(calibrated_probabilities.get("SUPPORTED", 0.0))][i]))
    # discrepancy gates
    delta, jsd_val = discrepancy_metrics(p_raw, [float(calibrated_probabilities.get("REFUTED", 0.0)), float(calibrated_probabilities.get("NOT ENOUGH INFO", 0.0)), float(calibrated_probabilities.get("SUPPORTED", 0.0))], stance_idx)
    gates_delta = float(cfg.get("discrepancy_delta", 0.2))
    gates_jsd = float(cfg.get("discrepancy_jsd", 0.2))

    if (delta > gates_delta) or (jsd_val > gates_jsd):
        verdict_str = labels.get("unclear", "Unclear 🟡")
        stability = "conflict"
    else:
        verdict_str = labels.get("likely_true", "Likely True 🟢") if p_true >= true_min else labels.get("unclear", "Unclear 🟡") if p_true >= unclear_min else labels.get("likely_false", "Likely False 🔴")
        stability = "stable"

    label_map = ["REFUTED", "NOT ENOUGH INFO", "SUPPORTED"]
    label = label_map[stance_idx]

    # choose top-3 by weight
    # Expecting results to include 'weights' or 'scores'
    top3 = _choose_by_weight_and_label(results, stance_idx, k=3)
    citations = _format_citations(top3)

    return {
        "label": label,
        "verdict": verdict_str,
        "p_calibrated_top": round(p_true, 3),
        "p_raw_top": round(p_raw, 3),
        "discrepancy": {"delta": round(delta, 3), "jsd": round(jsd_val, 3), "status": stability},
        "citations": citations,
    }


