"""Paired token-cluster intervals for strict-training representative refits."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "reviewer_priority4_20260922_v1"
sys.path.insert(0, str(ROOT / "results" / "fusion_uncertainty_20260922_v1" / "code"))
import run_fusion_uncertainty as uncertainty


def main() -> None:
    records = []
    for collection in ("BAYC", "MAYC"):
        for training in ("full", "strict", "strict_without_one_wei"):
            path = OUT / f"stage2_{collection}_{training}_refit_predictions.npz"
            with np.load(path, allow_pickle=False) as src:
                mask = src["strict_mask"]
                y = src["y_lrp"][mask]
                token = src["tokens"][mask]
                preds = np.column_stack((src["metadata_pred"][mask], src["early_pred"][mask]))
            assert len(y) >= 1000
            point = np.sqrt(np.mean((y[:, None]-preds)**2, axis=0))
            samples, _ = uncertainty.grouped_bootstrap(y, preds, token, equal_token=False,
                                                        seed=20260908)
            delta = samples[:, 1]-samples[:, 0]
            low, high = uncertainty.bounds(delta, 0.95)
            records.append({"collection": collection, "training_cohort": training,
                            "evaluation_cohort": "strict_clean_flag", "n_transactions": len(y),
                            "n_tokens": len(np.unique(token)), "metadata_rmse": float(point[0]),
                            "early_rmse": float(point[1]), "early_minus_metadata_rmse": float(point[1]-point[0]),
                            "paired_token_ci95_low": low, "paired_token_ci95_high": high,
                            "bootstrap_replicates": 2000})
    with (OUT / "stage2_refit_token_intervals.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for row in records:
        print(f"{row['collection']} {row['training_cohort']}: ΔRMSE={row['early_minus_metadata_rmse']:+.6f}, CI95=[{row['paired_token_ci95_low']:+.6f},{row['paired_token_ci95_high']:+.6f}]")


if __name__ == "__main__":
    main()
