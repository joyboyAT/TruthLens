"""
Fit temperature T for calibration on a dev set.

Input JSONL lines with fields per claim:
{
  "evidence": [ {"logits": [z_sup,z_ref,z_nei], "probs": [..], "scores": {...}, "domain": "...", "age_days": ...}, ... ],
  "label": 0|1|2  # class index for [SUP, REF, NEI]
}

This script fuses logits with stance-aware fusion and optimizes T to minimize NLL.
Writes learned T to config/calibration.yaml.
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.confidence_calibrator import stance_aware_logit_fusion, temperature_softmax, _load_yaml


def nll_for_T(T: float, data, priors, tau: float) -> float:
    losses = []
    for row in data:
        Z, _ = stance_aware_logit_fusion(row["evidence"], priors, tau)
        p = temperature_softmax(Z, T)
        y = int(row["label"])  # 0..2
        losses.append(-float(np.log(max(1e-12, p[y]))))
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True, help="Path to dev JSONL")
    ap.add_argument("--out", default="config/calibration.yaml")
    args = ap.parse_args()

    data = []
    with open(args.dev, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    cfg = _load_yaml("config/calibration.yaml")
    priors = _load_yaml("config/domain_priors.yaml")
    tau = float(cfg.get("recency_tau_days", 365))

    # Simple grid search for demo
    grid = np.linspace(0.5, 3.0, 26)
    best_T, best_loss = None, float("inf")
    for T in grid:
        loss = nll_for_T(T, data, priors, tau)
        if loss < best_loss:
            best_loss, best_T = loss, T

    # Write back
    cfg["temperature"] = float(best_T)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Best T={best_T:.3f}, NLL={best_loss:.4f} written to {args.out}")


if __name__ == "__main__":
    main()


