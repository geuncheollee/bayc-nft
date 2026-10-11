"""Paired token-bootstrap intervals for the selected residual corrections."""
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
    rows = []
    for collection in ("BAYC", "MAYC"):
        for analysis, filename, use_strict in (
            ("all_oof_residuals_all_evaluation", f"stage3_{collection}_residual_evaluation.npz", False),
            ("strict_oof_residuals_strict_evaluation", f"stage3_{collection}_strict_residual_evaluation.npz", True),
            ("strict_no_onewei_oof_residuals_strict_evaluation", f"stage3_{collection}_strict_no_onewei_residual_evaluation.npz", True),
        ):
            with np.load(OUT / filename, allow_pickle=False) as src:
                mask = src["strict_mask"] if use_strict else np.ones(len(src["y_lrp"]), dtype=bool)
                y = src["y_lrp"][mask]
                token = src["tokens"][mask]
                preds = np.column_stack((src["metadata_pred"][mask], src["residual_corrected_pred"][mask]))
            point = np.sqrt(np.mean((y[:, None]-preds)**2, axis=0))
            samples, _ = uncertainty.grouped_bootstrap(y, preds, token, equal_token=False, seed=20260908)
            low, high = uncertainty.bounds(samples[:, 1]-samples[:, 0], 0.95)
            rows.append({"collection": collection, "analysis": analysis, "n_transactions": len(y),
                         "n_tokens": len(np.unique(token)), "metadata_rmse": float(point[0]),
                         "residual_corrected_rmse": float(point[1]), "delta_rmse": float(point[1]-point[0]),
                         "token_bootstrap_ci95_low": low, "token_bootstrap_ci95_high": high,
                         "replicates": 2000})
    with (OUT / "stage3_residual_intervals.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['collection']} {row['analysis']}: ΔRMSE={row['delta_rmse']:+.6f}, CI95=[{row['token_bootstrap_ci95_low']:+.6f},{row['token_bootstrap_ci95_high']:+.6f}]")


if __name__ == "__main__":
    main()
