"""Retrospective paired bootstrap of frozen valuation models against LRP=0."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "fusion_vs_zero_uncertainty_20260922_v1"
IMAGE = ROOT / "results" / "image_asinh_transaction_2022_2024_eight_regressors_20260920_v1"
SOURCE_SCRIPT = ROOT / "results" / "fusion_uncertainty_20260922_v1" / "code" / "run_fusion_uncertainty.py"
spec = importlib.util.spec_from_file_location("fusion_uncertainty_core", SOURCE_SCRIPT)
assert spec is not None and spec.loader is not None
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

NAMES = ("zero", "metadata", "image", "early", "late")
FAMILY_WIDE_CONFIDENCE = 1.0 - 0.05 / 8.0  # Four models times two collections.


def selected_image(collection: str) -> tuple[Path, dict[str, str]]:
    rows = [row for row in core.read_csv(IMAGE / "pre2025_image_model_selection_112_combinations.csv")
            if row["collection"] == collection]
    row = min(rows, key=lambda item: float(item["pre2025_validation_transaction_rmse"]))
    path = IMAGE / "predictions" / f"{collection}_{row['encoder']}_{row['family']}.npz"
    return path, row


def verify_zero(collection: str, y: np.ndarray, tokens: np.ndarray) -> dict:
    rows = core.read_csv(core.META / "baseline_results.csv")
    baseline = next(row for row in rows if row["collection"] == collection
                    and row["sample"] == "evaluation_2025_plus" and row["baseline"] == "LRP_zero")
    transaction, equal = core.point_scores(y, np.zeros((len(y), 1)), tokens)
    assert np.isclose(transaction[0], float(baseline["transaction_rmse"]), atol=1e-12, rtol=0)
    assert np.isclose(equal[0], float(baseline["equal_token_rmse"]), atol=1e-12, rtol=0)
    return {"transaction_rmse": float(transaction[0]), "equal_token_rmse": float(equal[0])}


def records(collection: str, points: np.ndarray, replicas: np.ndarray,
            *, resampling: str, metric: str) -> list[dict]:
    rows = []
    for index, name in enumerate(NAMES[1:], start=1):
        delta = replicas[:, index] - replicas[:, 0]
        utility = 100 * (replicas[:, 0] - replicas[:, index]) / replicas[:, 0]
        row = {
            "collection": collection, "resampling": resampling, "metric": metric,
            "model": name, "reference": "LRP_zero",
            "model_rmse": float(points[index]), "zero_rmse": float(points[0]),
            "delta_rmse_model_minus_zero": float(points[index] - points[0]),
            "utility_percent": float(100 * (points[0] - points[index]) / points[0]),
            "bootstrap_probability_model_lower_rmse": float(np.mean(delta < 0)),
            "iterations": core.ITERATIONS,
        }
        for confidence, label in ((0.95, "95"), (0.975, "97_5"),
                                  (FAMILY_WIDE_CONFIDENCE, "99_375")):
            row[f"delta_rmse_ci{label}_low"], row[f"delta_rmse_ci{label}_high"] = core.bounds(delta, confidence)
            row[f"utility_ci{label}_low"], row[f"utility_ci{label}_high"] = core.bounds(utility, confidence)
        rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "completion.json").exists():
        raise FileExistsError("Completed output exists; no result is overwritten")
    rows = []
    audit = []
    for collection in ("BAYC", "MAYC"):
        y, base_preds, tokens, blocks, source_audit = core.load_aligned(collection)
        baseline = verify_zero(collection, y, tokens)
        image_path, image_selection = selected_image(collection)
        with np.load(image_path, allow_pickle=False) as source:
            for key, expected in (("source_rows", np.asarray([
                    item["source_row"] for item in core.read_jsonl(
                        core.TARGET / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl")
                ])), ("tokens", tokens), ("y_lrp", y)):
                if not np.array_equal(source[key].astype(str) if key == "tokens" else source[key], expected):
                    raise ValueError(f"{collection}: image {key} is not row-aligned")
            image_pred = source["predictions_lrp"].copy()
        image_evaluation = next(
            row for row in core.read_csv(IMAGE / "fixed_out_of_time_image_evaluation_112_combinations.csv")
            if row["collection"] == collection and row["encoder"] == image_selection["encoder"]
            and row["family"] == image_selection["family"]
        )
        image_rmse = float(np.sqrt(np.mean((y - image_pred) ** 2)))
        if not np.isclose(image_rmse, float(image_evaluation["evaluation_transaction_rmse"]), atol=1e-12, rtol=0):
            raise ValueError(f"{collection}: selected image score does not reproduce result CSV")
        preds = np.column_stack((np.zeros(len(y)), base_preds[:, 0], image_pred,
                                 base_preds[:, 1], base_preds[:, 2]))
        transaction, equal = core.point_scores(y, preds, tokens)
        token_transaction, token_equal = core.grouped_bootstrap(
            y, preds, tokens, equal_token=True, seed=core.SEED
        )
        block_transaction, _ = core.grouped_bootstrap(
            y, preds, blocks, equal_token=False, seed=core.SEED
        )
        rows.extend(records(collection, transaction, token_transaction,
                            resampling="paired_token_cluster", metric="transaction_rmse"))
        rows.extend(records(collection, equal, token_equal,
                            resampling="paired_token_cluster", metric="equal_token_rmse"))
        rows.extend(records(collection, transaction, block_transaction,
                            resampling="paired_14day_calendar_block", metric="transaction_rmse"))
        np.savez_compressed(OUT / f"{collection}_bootstrap_replicates.npz",
                            model_names=np.asarray(NAMES),
                            token_transaction_rmse=token_transaction,
                            token_equal_token_rmse=token_equal,
                            block_transaction_rmse=block_transaction)
        audit.append({
            "collection": collection, "n_transactions": len(y),
            "n_tokens": int(np.unique(tokens).size), "n_14day_blocks": int(np.unique(blocks).size),
            "baseline_reproduced": baseline, "selected_image_encoder": image_selection["encoder"],
            "selected_image_family": image_selection["family"],
            "selected_image_validation_rmse": float(image_selection["pre2025_validation_transaction_rmse"]),
            "selected_image_file": str(image_path.relative_to(ROOT)),
            "selected_image_sha256": core.sha256(image_path),
            "image_row_alignment_and_score_reproduced": True,
            "other_model_audit": source_audit,
        })
        print(f"{collection}: {len(y)} paired rows, zero RMSE {baseline['transaction_rmse']:.9f}", flush=True)
    core.write_csv(OUT / "vs_zero_comparisons.csv", rows)
    core.write_json(OUT / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_descriptive_uncertainty",
        "reference": "LRP=0 prediction on identical evaluation rows",
        "seed": core.SEED, "iterations": core.ITERATIONS,
        "resampling": "paired token-cluster and nonoverlapping UTC 14-day calendar-block bootstrap",
        "confidence_intervals": "percentile 95%, 97.5% (two collections), 99.375% (eight transaction-RMSE comparisons)",
        "interpretation": "fixed-prediction sampling uncertainty only; previously examined evaluation set; no model-selection uncertainty",
        "audits": audit,
    })
    core.write_json(OUT / "completion.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True, "comparison_rows": len(rows), "collections": 2,
        "iterations_per_resampling_per_collection": core.ITERATIONS,
        "all_row_and_metric_checks_passed": True,
    })


if __name__ == "__main__":
    main()
