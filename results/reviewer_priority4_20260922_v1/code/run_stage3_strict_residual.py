"""Prespecified strict-flag sensitivity for the residual image correction.

Uses the same archived metadata OOF predictions but fits the correction using
only transaction rows satisfying the independent 2026-09-08 cleaning audit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "reviewer_priority4_20260922_v1"
sys.path.insert(0, str(OUT / "code"))
import run_stage3_residual as stage3


def main() -> None:
    if not (OUT / "stage3_residual_report.json").exists():
        raise RuntimeError("Complete primary residual analysis before strict sensitivity")
    report = {"design": "same metadata OOF predictions as stage3; strict-flag-only image residual training and evaluation",
              "test_status": "retrospective descriptive, previously inspected 2025+ data", "collections": {}}
    with threadpool_limits(limits=3):
        for collection in ("BAYC", "MAYC"):
            choice = stage3.stage2.chosen(collection)
            with np.load(OUT / f"stage3_{collection}_metadata_oof.npz", allow_pickle=False) as src:
                rows = src["source_rows"].copy()
                tokens = src["tokens"].copy()
                times = src["times"].copy()
                y = src["y_lrp"].copy()
                metadata_pred = src["metadata_oof_pred"].copy()
            eligibility_path = ROOT / "revision" / "data_audit_20260908" / f"{collection.lower()}_trade_eligibility.jsonl"
            wanted = set(map(int, rows))
            flags = {r["source_row"]: r["strict_clean"] for r in stage3.stage2.lines(eligibility_path)
                     if r["source_row"] in wanted}
            assert len(flags) == len(rows)
            clean = np.asarray([flags[int(row)] for row in rows], dtype=bool)
            q23 = (times >= "2024-04-01") & (times < "2024-10-01") & clean
            q4 = (times >= "2024-10-01") & (times < "2025-01-01") & clean
            x = stage3.image_rows(collection, choice["early_encoder"], tokens)
            residual = y-metadata_pred
            grid = []
            for alpha in stage3.ALPHAS:
                adjustment = stage3.correction(x[q23], residual[q23], x[q4], alpha)
                pred = metadata_pred[q4]+adjustment
                grid.append({"alpha": alpha, "strict_q4_rmse": float(np.sqrt(np.mean((y[q4]-pred)**2))),
                             "strict_q4_partial_r2": float(1-np.sum((y[q4]-pred)**2)/np.sum((y[q4]-metadata_pred[q4])**2))})
            best = min(grid, key=lambda row: (row["strict_q4_rmse"], row["alpha"]))
            with np.load(OUT / f"stage2_{collection}_full_refit_predictions.npz", allow_pickle=False) as src:
                eval_strict = src["strict_mask"].copy()
            with np.load(OUT / f"stage3_{collection}_residual_evaluation.npz", allow_pickle=False) as src:
                eval_tokens = src["tokens"].copy()
                yy = src["y_lrp"].copy()
                base = src["metadata_pred"].copy()
            x_eval = stage3.image_rows(collection, choice["early_encoder"], eval_tokens)
            correction = stage3.correction(x[clean], residual[clean], x_eval, best["alpha"])
            candidate = base+correction
            result = {"encoder": choice["early_encoder"], "strict_train_oof_rows": int(clean.sum()),
                      "strict_q2_q3_rows": int(q23.sum()), "strict_q4_validation_rows": int(q4.sum()),
                      "grid": grid, "selected_alpha": best["alpha"],
                      "strict_q4_partial_r2": best["strict_q4_partial_r2"]}
            for name, mask in (("all_evaluation", np.ones(len(yy), dtype=bool)), ("strict_evaluation", eval_strict)):
                baseline_sse = float(np.sum((yy[mask]-base[mask])**2))
                candidate_sse = float(np.sum((yy[mask]-candidate[mask])**2))
                result[name] = {"n": int(mask.sum()),
                                "metadata_rmse": float(np.sqrt(baseline_sse/mask.sum())),
                                "residual_corrected_rmse": float(np.sqrt(candidate_sse/mask.sum())),
                                "predictive_partial_r2": 1-candidate_sse/baseline_sse}
            report["collections"][collection] = result
            np.savez_compressed(OUT / f"stage3_{collection}_strict_residual_evaluation.npz",
                                y_lrp=yy, metadata_pred=base, residual_corrected_pred=candidate,
                                strict_mask=eval_strict, tokens=eval_tokens)
            target_path = ROOT / "revision" / "target_pipeline_20260909" / f"{collection.lower()}_development_targets.jsonl"
            one_wei_rows = {r["source_row"] for r in stage3.stage2.lines(target_path)
                            if r["source_row"] in wanted and r["price_eth"] <= 1e-18*(1+1e-9)}
            clean_no_onewei = clean & ~np.isin(rows, list(one_wei_rows))
            q23_nw = (times >= "2024-04-01") & (times < "2024-10-01") & clean_no_onewei
            q4_nw = (times >= "2024-10-01") & (times < "2025-01-01") & clean_no_onewei
            grid_nw = []
            for alpha in stage3.ALPHAS:
                adj = stage3.correction(x[q23_nw], residual[q23_nw], x[q4_nw], alpha)
                pred = metadata_pred[q4_nw]+adj
                grid_nw.append({"alpha": alpha, "strict_no_onewei_q4_rmse": float(np.sqrt(np.mean((y[q4_nw]-pred)**2))),
                                "strict_no_onewei_q4_partial_r2": float(1-np.sum((y[q4_nw]-pred)**2)/np.sum((y[q4_nw]-metadata_pred[q4_nw])**2))})
            best_nw = min(grid_nw, key=lambda row: (row["strict_no_onewei_q4_rmse"], row["alpha"]))
            correction_nw = stage3.correction(x[clean_no_onewei], residual[clean_no_onewei], x_eval, best_nw["alpha"])
            pred_nw = base+correction_nw
            strict_sse_nw = float(np.sum((yy[eval_strict]-pred_nw[eval_strict])**2))
            strict_sse_meta = float(np.sum((yy[eval_strict]-base[eval_strict])**2))
            result["strict_without_one_wei"] = {
                "training_oof_rows": int(clean_no_onewei.sum()), "excluded_one_wei_oof_rows": int((clean & ~clean_no_onewei).sum()),
                "strict_q4_validation_rows": int(q4_nw.sum()), "grid": grid_nw,
                "selected_alpha": best_nw["alpha"], "strict_q4_partial_r2": best_nw["strict_no_onewei_q4_partial_r2"],
                "strict_evaluation_rmse": float(np.sqrt(strict_sse_nw/eval_strict.sum())),
                "strict_evaluation_predictive_partial_r2": 1-strict_sse_nw/strict_sse_meta,
            }
            np.savez_compressed(OUT / f"stage3_{collection}_strict_no_onewei_residual_evaluation.npz",
                                y_lrp=yy, metadata_pred=base, residual_corrected_pred=pred_nw,
                                strict_mask=eval_strict, tokens=eval_tokens)
            (OUT / "stage3_strict_residual_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"stage3 strict {collection}: alpha={best['alpha']}, strict Q4 partial R²={best['strict_q4_partial_r2']:.6f}, strict eval partial R²={result['strict_evaluation']['predictive_partial_r2']:.6f}", flush=True)
            print(f"stage3 strict+no1wei {collection}: alpha={best_nw['alpha']}, strict Q4 partial R²={best_nw['strict_no_onewei_q4_partial_r2']:.6f}, strict eval partial R²={result['strict_without_one_wei']['strict_evaluation_predictive_partial_r2']:.6f}", flush=True)


if __name__ == "__main__":
    main()
