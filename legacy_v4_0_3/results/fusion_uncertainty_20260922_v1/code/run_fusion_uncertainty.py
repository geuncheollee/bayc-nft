"""Paired uncertainty analysis for the frozen TF-IDF fusion benchmarks.

This is a retrospective analysis of an already examined 2025+ evaluation set.
It quantifies uncertainty conditional on fixed, pre-2025-selected predictions;
it does not rerun model selection or provide a fresh confirmatory test.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "fusion_uncertainty_20260922_v1"
META = ROOT / "results" / "metadata_asinh_transaction_2022_2024_eight_regressors_20260920_v1"
EARLY = ROOT / "results" / "early_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260921_v1"
LATE = ROOT / "results" / "late_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260922_v2"
TARGET = ROOT / "revision" / "target_pipeline_20260909"
SEED = 20260908
ITERATIONS = 2000
MODEL_NAMES = ("metadata", "early", "late")
COMPARISONS = (
    ("early_vs_late", "early", "late"),
    ("early_vs_metadata", "early", "metadata"),
    ("late_vs_metadata", "late", "metadata"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_paths(collection: str) -> tuple[dict[str, Path], dict[str, float]]:
    meta_row = next(
        row for row in read_csv(META / "tfidf_constrained_metadata_selection.csv")
        if row["collection"] == collection
    )
    early_row = next(
        row for row in read_csv(EARLY / "development_selected_early_model.csv")
        if row["collection"] == collection and row["overall_selected_pre2025"] == "True"
    )
    late_row = next(
        row for row in read_csv(LATE / "development_selected_late_model.csv")
        if row["collection"] == collection and row["overall_selected_pre2025"] == "True"
    )
    paths = {
        "metadata": META / "predictions" / f"{collection}_TF_IDF_{meta_row['family']}.npz",
        "early": EARLY / "predictions" / f"{collection}_{early_row['encoder']}_{early_row['family']}.npz",
        "late": LATE / "predictions" / f"{collection}_{late_row['encoder']}_{late_row['image_family']}.npz",
    }
    expected = {
        "metadata": float(meta_row["evaluation_transaction_rmse"]),
        "early": float(early_row["evaluation_transaction_rmse"]),
        "late": float(late_row["evaluation_transaction_rmse"]),
    }
    return paths, expected


def load_aligned(collection: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    paths, expected = selected_paths(collection)
    data = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as source:
            data[name] = {key: source[key].copy() for key in (
                "source_rows", "tokens", "y_lrp", "predictions_lrp"
            )}
    first = data["metadata"]
    for name in MODEL_NAMES[1:]:
        for key in ("source_rows", "tokens", "y_lrp"):
            if not np.array_equal(first[key], data[name][key]):
                raise ValueError(f"{collection}: {name} {key} is not row-aligned")
    n = {"BAYC": 5120, "MAYC": 13019}[collection]
    if len(first["y_lrp"]) != n or len(np.unique(first["source_rows"])) != n:
        raise ValueError(f"{collection}: evaluation row count or source-row uniqueness mismatch")

    target_path = TARGET / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl"
    target = read_jsonl(target_path)
    if len(target) != n:
        raise ValueError(f"{collection}: target row count mismatch")
    for key, values in (
        ("source_rows", np.asarray([row["source_row"] for row in target])),
        ("tokens", np.asarray([str(row["token_id"]) for row in target])),
        ("y_lrp", np.asarray([row["y_log_relative_price"] for row in target], dtype=np.float64)),
    ):
        if not np.array_equal(first[key].astype(str) if key == "tokens" else first[key], values):
            raise ValueError(f"{collection}: {key} differs from canonical target file")
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [datetime.fromisoformat(row["time"]) for row in target]
    if any(not start <= item < datetime(2026, 4, 14, tzinfo=timezone.utc) for item in times):
        raise ValueError(f"{collection}: timestamp outside fixed evaluation window")
    blocks = np.asarray([(item - start).days // 14 for item in times], dtype=np.int64)
    y = first["y_lrp"]
    preds = np.column_stack([data[name]["predictions_lrp"] for name in MODEL_NAMES])
    if not np.isfinite(y).all() or not np.isfinite(preds).all():
        raise ValueError(f"{collection}: non-finite prediction or target")
    observed = np.sqrt(np.mean((y[:, None] - preds) ** 2, axis=0))
    for i, name in enumerate(MODEL_NAMES):
        if not np.isclose(observed[i], expected[name], rtol=0, atol=1e-12):
            raise ValueError(f"{collection}: {name} RMSE does not reproduce result CSV")
    audit = {
        "collection": collection,
        "n_transactions": n,
        "n_tokens": int(np.unique(first["tokens"]).size),
        "n_14day_blocks": int(np.unique(blocks).size),
        "prediction_files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                             for name, path in paths.items()},
        "target_file": {"path": str(target_path.relative_to(ROOT)), "sha256": sha256(target_path)},
        "row_alignment_passed": True,
        "published_metric_reproduction_passed": True,
    }
    return y, preds, first["tokens"].astype(str), blocks, audit


def point_scores(y: np.ndarray, preds: np.ndarray, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inverse, token_counts = np.unique(tokens, return_inverse=True, return_counts=True)
    residual2 = (y[:, None] - preds) ** 2
    transaction = np.sqrt(np.mean(residual2, axis=0))
    equal_token = np.asarray([
        np.sqrt(np.mean(np.bincount(inverse, weights=residual2[:, col]) / token_counts))
        for col in range(preds.shape[1])
    ])
    return transaction, equal_token


def grouped_bootstrap(y: np.ndarray, preds: np.ndarray, groups: np.ndarray,
                      *, equal_token: bool, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    _, inverse, group_counts = np.unique(groups, return_inverse=True, return_counts=True)
    n_groups = len(group_counts)
    squared = (y[:, None] - preds) ** 2
    group_sse = np.column_stack([
        np.bincount(inverse, weights=squared[:, col], minlength=n_groups)
        for col in range(preds.shape[1])
    ])
    group_mse = group_sse / group_counts[:, None]
    transaction = np.empty((ITERATIONS, preds.shape[1]), dtype=np.float64)
    equal = np.empty_like(transaction) if equal_token else None
    generator = np.random.default_rng(seed)
    for start in range(0, ITERATIONS, 128):
        stop = min(start + 128, ITERATIONS)
        draws = generator.integers(0, n_groups, size=(stop - start, n_groups))
        denominator = group_counts[draws].sum(axis=1)
        for col in range(preds.shape[1]):
            transaction[start:stop, col] = np.sqrt(group_sse[draws, col].sum(axis=1) / denominator)
            if equal is not None:
                equal[start:stop, col] = np.sqrt(group_mse[draws, col].mean(axis=1))
    return transaction, equal


def bounds(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail], method="linear")
    return float(low), float(high)


def comparison_rows(collection: str, point: np.ndarray, samples: np.ndarray,
                    *, resampling: str, metric: str) -> list[dict]:
    index = {name: i for i, name in enumerate(MODEL_NAMES)}
    rows = []
    for label, candidate, reference in COMPARISONS:
        candidate_i, reference_i = index[candidate], index[reference]
        delta = samples[:, candidate_i] - samples[:, reference_i]
        utility = 100 * (samples[:, reference_i] - samples[:, candidate_i]) / samples[:, reference_i]
        delta95 = bounds(delta, 0.95)
        delta975 = bounds(delta, 0.975)
        utility95 = bounds(utility, 0.95)
        utility975 = bounds(utility, 0.975)
        rows.append({
            "collection": collection, "resampling": resampling, "metric": metric,
            "comparison": label, "candidate": candidate, "reference": reference,
            "candidate_rmse": float(point[candidate_i]), "reference_rmse": float(point[reference_i]),
            "delta_rmse_candidate_minus_reference": float(point[candidate_i] - point[reference_i]),
            "delta_rmse_ci95_low": delta95[0], "delta_rmse_ci95_high": delta95[1],
            "delta_rmse_ci97_5_low": delta975[0], "delta_rmse_ci97_5_high": delta975[1],
            "utility_percent": float(100 * (point[reference_i] - point[candidate_i]) / point[reference_i]),
            "utility_ci95_low": utility95[0], "utility_ci95_high": utility95[1],
            "utility_ci97_5_low": utility975[0], "utility_ci97_5_high": utility975[1],
            "bootstrap_probability_candidate_lower_rmse": float(np.mean(delta < 0)),
            "iterations": ITERATIONS,
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "completion.json").exists():
        raise FileExistsError("Completed output exists; no result is overwritten")
    all_rows = []
    audits = []
    for collection in ("BAYC", "MAYC"):
        y, preds, tokens, blocks, audit = load_aligned(collection)
        audits.append(audit)
        transaction_point, equal_point = point_scores(y, preds, tokens)
        token_transaction, token_equal = grouped_bootstrap(
            y, preds, tokens, equal_token=True, seed=SEED
        )
        block_transaction, _ = grouped_bootstrap(
            y, preds, blocks, equal_token=False, seed=SEED
        )
        all_rows.extend(comparison_rows(collection, transaction_point, token_transaction,
                                        resampling="paired_token_cluster", metric="transaction_rmse"))
        all_rows.extend(comparison_rows(collection, equal_point, token_equal,
                                        resampling="paired_token_cluster", metric="equal_token_rmse"))
        all_rows.extend(comparison_rows(collection, transaction_point, block_transaction,
                                        resampling="paired_14day_calendar_block", metric="transaction_rmse"))
        np.savez_compressed(
            OUT / f"{collection}_bootstrap_replicates.npz",
            model_names=np.asarray(MODEL_NAMES),
            token_transaction_rmse=token_transaction,
            token_equal_token_rmse=token_equal,
            block_transaction_rmse=block_transaction,
        )
        print(f"{collection}: {audit['n_transactions']} aligned rows, "
              f"{audit['n_tokens']} tokens, {audit['n_14day_blocks']} 14-day blocks", flush=True)

    write_csv(OUT / "paired_comparisons.csv", all_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_descriptive_uncertainty",
        "protocol": "revision/Analysis Protocol 20260908.md, Section 6",
        "seed": SEED, "iterations": ITERATIONS,
        "token_resampling": "sample unique token IDs with replacement; keep all corresponding trades; same sample for all models",
        "calendar_resampling": "non-overlapping UTC 14-day blocks anchored 2025-01-01; sample blocks with replacement; same sample for all models",
        "confidence_interval": "percentile intervals of paired bootstrap replicates",
        "interpretation": "fixed-prediction sampling uncertainty only; no training or model-selection uncertainty; previously examined evaluation set",
        "audits": audits,
    }
    write_json(OUT / "run_manifest.json", manifest)
    selected = [row for row in all_rows if row["resampling"] == "paired_token_cluster"
                and row["metric"] == "transaction_rmse"]
    lines = [
        "# Paired uncertainty for TF-IDF fusion models", "",
        "Retrospective 2025+ evaluation on identical transactions. Negative ΔRMSE means the first model is better. "
        "CIs condition on frozen predictions and do not account for training or selection uncertainty.", "",
        "| Collection | Comparison | ΔRMSE | 95% token-cluster CI | 97.5% token-cluster CI |", "|---|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['collection']} | {row['candidate']} − {row['reference']} | "
            f"{row['delta_rmse_candidate_minus_reference']:.6f} | "
            f"[{row['delta_rmse_ci95_low']:.6f}, {row['delta_rmse_ci95_high']:.6f}] | "
            f"[{row['delta_rmse_ci97_5_low']:.6f}, {row['delta_rmse_ci97_5_high']:.6f}] |"
        )
    lines += ["", "Full transaction/equal-token and time-block comparisons: `paired_comparisons.csv`.", ""]
    (OUT / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "completion.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True, "collections": 2, "comparison_rows": len(all_rows),
        "iterations_per_resampling_per_collection": ITERATIONS,
        "row_alignment_passed": all(item["row_alignment_passed"] for item in audits),
        "published_metric_reproduction_passed": all(item["published_metric_reproduction_passed"] for item in audits),
    })


if __name__ == "__main__":
    main()
