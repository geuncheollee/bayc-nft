"""Temporal token-balanced BAYC metadata experiment.

All tuning and fitting use transactions through 2024-12-31.  Each token has
equal total training influence by fitting one token row to its mean fold-fitted
robust-asinh target.  Specifications are frozen before the fixed retrospective
2025-01-01--2026-04-13 evaluation labels are loaded.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "metadata_temporal_token_balanced_20260920_v1"
SOURCE_CODE = ROOT / "results" / "metadata_asinh_nine_regressors_20260920" / "code"
CANONICAL_CODE = ROOT / "results" / "seven_encoder_full_rerun" / "code"
sys.path.insert(0, str(SOURCE_CODE))
sys.path.insert(0, str(CANONICAL_CODE))

import canonical_runner as cr
import run_experiment as common


SEED = 20260908
ENCODINGS = ("one-hot", "TF-IDF")
FAMILIES = tuple(common.FAMILIES)
TUNING_QUARTERS = tuple(cr.FINAL_Q)
EXPECTED_FITS = len(ENCODINGS) * (
    len(TUNING_QUARTERS) * sum(len(common.GRIDS[family]) for family in FAMILIES)
    + len(FAMILIES)
)


def now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def token_balanced_metrics(y: np.ndarray, pred: np.ndarray, tokens: np.ndarray) -> dict:
    _, inverse, counts = np.unique(tokens.astype(str), return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights /= weights.sum()
    residual = y - pred
    mean_y = float(np.dot(weights, y))
    mse = float(np.dot(weights, residual**2))
    sst = float(np.dot(weights, (y - mean_y) ** 2))
    return {
        "n_transactions": int(len(y)), "n_tokens": int(len(counts)),
        "rmse": float(np.sqrt(mse)), "mae": float(np.dot(weights, np.abs(residual))),
        "r2": float(1.0 - mse / sst), "target_sd": float(np.sqrt(sst)),
        "mse": mse,
    }


def transaction_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    residual = y - pred
    mse = float(np.mean(residual**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / sst), "target_sd": float(np.std(y)),
    }


def group_mean(values: np.ndarray, inverse: np.ndarray, mask: np.ndarray, n_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    indices = inverse[mask]
    counts = np.bincount(indices, minlength=n_tokens)
    sums = np.bincount(indices, weights=values[mask], minlength=n_tokens)
    present = np.flatnonzero(counts)
    means = np.full(n_tokens, np.nan, dtype=np.float64)
    means[present] = sums[present] / counts[present]
    return means, present


def transaction_predictions(pred_token: np.ndarray, token_indices: np.ndarray, inverse: np.ndarray,
                            mask: np.ndarray, n_tokens: int) -> np.ndarray:
    lookup = np.full(n_tokens, np.nan, dtype=np.float64)
    lookup[token_indices] = pred_token
    result = lookup[inverse[mask]]
    assert np.isfinite(result).all()
    return result


def fit_model(prep: dict, family: str, config: dict, y_token: np.ndarray):
    if family == "PLS" and config["components"] > prep["rank"]:
        return None, {"valid": False, "error": "components exceed rank"}
    X = prep["raw_train"] if family in common.TREE_FAMILIES else prep["ztrain"]
    try:
        model = common.build_model(family, config, "development")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y_token)
        valid = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        if family == "PLS":
            valid &= len(model.n_iter_) == config["components"]
        return model, {"valid": bool(valid), "warnings": [str(item.message) for item in caught]}
    except Exception as error:
        return None, {"valid": False, "error": repr(error)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    (OUT / "predictions").mkdir()
    start_time = time.time()

    data = cr.load_development("BAYC")
    assert len(data["y"]) == 58_635
    assert np.all(data["times"] < "2025-01-01")
    tx_tokens = data["tokens"].astype(str)
    unique_tokens, first, inverse = np.unique(tx_tokens, return_index=True, return_inverse=True)
    token_X = data["X"][first]
    assert np.array_equal(data["X"], token_X[inverse])
    assert "token_id" not in {str(column).lower() for column in data["columns"]}
    token_data = {"X": token_X, "tokens": unique_tokens}
    n_tokens = len(unique_tokens)

    source_paths = [
        Path(__file__), Path(common.__file__), Path(cr.__file__), Path(cr.base.__file__),
        cr.base.OLD / "metadata_normalized.jsonl",
        cr.base.TARGET / "bayc_development_targets.jsonl",
        cr.base.TARGET / "bayc_temporal_test_targets.NOT_FOR_SELECTION.jsonl",
    ]
    specification = {
        "collection": "BAYC",
        "development": "inception through 2024-12-31",
        "fixed_retrospective_out_of_time_evaluation": "2025-01-01 through 2026-04-13",
        "target": "fold-fitted robust asinh of transaction LRP",
        "training_unit": "one row per token with mean transformed target over training-period transactions",
        "training_weight": "equal total weight per token",
        "selection_metric": "pooled 2024 Q2-Q4 equal-token RMSE on original LRP scale",
        "encodings": ENCODINGS, "families": FAMILIES, "grids": common.GRIDS,
        "evaluation_policy": "evaluation labels loaded only after all specifications and fitted models are frozen",
        "seed": SEED,
    }
    write_json(OUT / "experiment_specification.json", specification)
    write_json(OUT / "run_manifest.json", {
        "start_utc": now(), "expected_fits": EXPECTED_FITS,
        "python": sys.version, "machine": platform.platform(),
        "available_ram_gib": psutil.virtual_memory().available / 2**30,
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_paths],
    })

    tuning_rows = []
    scores: dict[tuple[str, str], list[dict]] = {}
    completed = 0
    for encoding in ENCODINGS:
        quarter_cache = {}
        for origin, end in TUNING_QUARTERS:
            train_mask = data["times"] < origin
            valid_mask = (data["times"] >= origin) & (data["times"] < end)
            assert train_mask.any() and valid_mask.any()
            parameters = common.fit_target_transform(data["y"][train_mask])
            z_all = common.target_forward(data["y"], parameters)
            mean_z, train_token_indices = group_mean(z_all, inverse, train_mask, n_tokens)
            valid_token_indices = np.unique(inverse[valid_mask])
            prep = common.prepare(token_data, train_token_indices, valid_token_indices, encoding)
            quarter_cache[origin] = {
                "train_mask": train_mask, "valid_mask": valid_mask,
                "train_tokens": train_token_indices, "valid_tokens": valid_token_indices,
                "parameters": parameters, "mean_z": mean_z, "prep": prep,
            }
            for family in FAMILIES:
                for config in common.GRIDS[family]:
                    tick = time.perf_counter()
                    model, status = fit_model(prep, family, config, mean_z[train_token_indices])
                    if status["valid"]:
                        Xvalid = prep["raw_valid"] if family in common.TREE_FAMILIES else prep["zvalid"]
                        pred_token = common.target_inverse(common.predict_model(model, Xvalid), parameters)
                        pred_tx = transaction_predictions(pred_token, valid_token_indices, inverse, valid_mask, n_tokens)
                        metric = token_balanced_metrics(data["y"][valid_mask], pred_tx, tx_tokens[valid_mask])
                    else:
                        metric = None
                    tuning_rows.append({
                        "encoding": encoding, "family": family, "origin": origin, "end": end,
                        "config": json.dumps(config, sort_keys=True), "valid": status["valid"],
                        "equal_token_n": None if metric is None else metric["n_tokens"],
                        "equal_token_mse": None if metric is None else metric["mse"],
                        "equal_token_rmse": None if metric is None else metric["rmse"],
                        "seconds": time.perf_counter() - tick,
                    })
                    completed += 1
                    write_json(OUT / "progress.json", {
                        "stage": "pre2025_tuning", "completed_fits": completed,
                        "expected_fits": EXPECTED_FITS, "encoding": encoding,
                        "quarter": origin, "family": family,
                    })
                    del model
        # Pool token-quarter MSE for deterministic selection.
        for family in FAMILIES:
            family_scores = []
            for config in common.GRIDS[family]:
                config_text = json.dumps(config, sort_keys=True)
                parts = [row for row in tuning_rows if row["encoding"] == encoding and row["family"] == family and row["config"] == config_text]
                valid = len(parts) == len(TUNING_QUARTERS) and all(row["valid"] for row in parts)
                pooled_mse = (
                    sum(float(row["equal_token_mse"]) * int(row["equal_token_n"]) for row in parts)
                    / sum(int(row["equal_token_n"]) for row in parts)
                ) if valid else None
                family_scores.append({"config": config, "valid": valid, "mse_lrp": pooled_mse})
            scores[encoding, family] = family_scores
        del quarter_cache
        gc.collect()
    write_csv(OUT / "pre2025_tuning_all_configs.csv", tuning_rows)

    # Freeze configurations using only pre-2025 validation scores, then refit.
    all_train_mask = data["times"] < "2025-01-01"
    final_parameters = common.fit_target_transform(data["y"][all_train_mask])
    final_z = common.target_forward(data["y"], final_parameters)
    final_mean_z, all_train_token_indices = group_mean(final_z, inverse, all_train_mask, n_tokens)
    empty = np.asarray([], dtype=int)
    freeze = {"created_utc": now(), "evaluation_labels_loaded_before_freeze": False, "models": {}}
    development_selection = []
    training_results = []
    for encoding in ENCODINGS:
        prep = common.prepare(token_data, all_train_token_indices, empty, encoding)
        for family in FAMILIES:
            ordered = common.ordered_choices(family, scores[encoding, family])
            if not ordered:
                raise RuntimeError(f"No valid pre-2025 configuration: {encoding} {family}")
            selected = ordered[0]
            model, status = fit_model(prep, family, selected["config"], final_mean_z[all_train_token_indices])
            if not status["valid"]:
                raise RuntimeError(f"Final refit failed: {encoding} {family}")
            Xtrain = prep["raw_train"] if family in common.TREE_FAMILIES else prep["ztrain"]
            pred_train_token = common.target_inverse(common.predict_model(model, Xtrain), final_parameters)
            pred_train_tx = transaction_predictions(
                pred_train_token, all_train_token_indices, inverse, all_train_mask, n_tokens
            )
            macro = token_balanced_metrics(data["y"], pred_train_tx, tx_tokens)
            micro = transaction_metrics(data["y"], pred_train_tx)
            key = f"{encoding}__{family}"
            model_path = OUT / "models" / f"{encoding.replace('-', '_')}_{family}.joblib"
            joblib.dump({
                "encoding": prep["encoding"], "scaler": prep["scaler"], "model": model,
                "family": family, "metadata_encoding": encoding, "columns": data["columns"],
                "target_parameters": final_parameters, "training_tokens": unique_tokens[all_train_token_indices],
            }, model_path, compress=3)
            freeze["models"][key] = {
                "encoding": encoding, "family": family, "selected_config": selected["config"],
                "pooled_pre2025_validation_equal_token_rmse": float(np.sqrt(selected["mse_lrp"])),
                "model_path": str(model_path.relative_to(OUT)), "model_sha256": sha256(model_path),
                "target_parameters": final_parameters,
            }
            development_selection.append({
                "collection": "BAYC", "encoding": encoding, "family": family,
                "selected_config": json.dumps(selected["config"], sort_keys=True),
                "pre2025_validation_equal_token_rmse": float(np.sqrt(selected["mse_lrp"])),
            })
            training_results.append({
                "collection": "BAYC", "encoding": encoding, "family": family,
                **{f"train_equal_token_{name}": value for name, value in macro.items() if name != "mse"},
                **{f"train_transaction_{name}": value for name, value in micro.items()},
            })
            completed += 1
            write_json(OUT / "progress.json", {
                "stage": "final_refit_pre2025", "completed_fits": completed,
                "expected_fits": EXPECTED_FITS, "encoding": encoding, "family": family,
            })
        del prep
        gc.collect()
    assert completed == EXPECTED_FITS
    freeze["development_selected_model"] = min(
        development_selection, key=lambda row: row["pre2025_validation_equal_token_rmse"]
    )
    write_json(OUT / "selection_freeze.json", freeze)
    write_csv(OUT / "pre2025_model_selection_18_combinations.csv", development_selection)
    write_csv(OUT / "training_performance_18_combinations.csv", training_results)

    # Evaluation labels are loaded only after the immutable freeze above exists.
    eval_rows = cr.lines(cr.base.TARGET / "bayc_temporal_test_targets.NOT_FOR_SELECTION.jsonl")
    assert len(eval_rows) == 5_120
    assert all("2025-01-01" <= row["time"] < "2026-04-14" for row in eval_rows)
    metadata = {(row["collection"], int(row["token_id"])): row for row in cr.lines(cr.base.OLD / "metadata_normalized.jsonl")}
    y_eval = np.asarray([row["y_log_relative_price"] for row in eval_rows], dtype=np.float64)
    tokens_eval = np.asarray([str(row["token_id"]) for row in eval_rows])
    rows_eval = np.asarray([row["source_row"] for row in eval_rows])
    unique_eval_tokens, first_eval, inverse_eval = np.unique(tokens_eval, return_index=True, return_inverse=True)
    known_eval_token = np.isin(unique_eval_tokens, unique_tokens[all_train_token_indices])
    final_results = []
    for encoding in ENCODINGS:
        for family in FAMILIES:
            key = f"{encoding}__{family}"
            spec = freeze["models"][key]
            model_path = OUT / spec["model_path"]
            assert sha256(model_path) == spec["model_sha256"]
            bundle = joblib.load(model_path)
            X_eval_token = np.asarray([
                [metadata["BAYC", int(token)][column] for column in bundle["columns"]]
                for token in unique_eval_tokens
            ], dtype=object)
            raw = common.dense(bundle["encoding"].transform(X_eval_token))
            features = raw if family in common.TREE_FAMILIES else bundle["scaler"].transform(raw)
            pred_eval_token = common.target_inverse(common.predict_model(bundle["model"], features), bundle["target_parameters"])
            pred_eval = pred_eval_token[inverse_eval]
            assert np.isfinite(pred_eval).all()
            macro = token_balanced_metrics(y_eval, pred_eval, tokens_eval)
            micro = transaction_metrics(y_eval, pred_eval)
            known_tx = known_eval_token[inverse_eval]
            subgroup = {}
            for name, mask in (("known", known_tx), ("unseen", ~known_tx)):
                if mask.any():
                    subgroup.update({
                        f"{name}_equal_token_{metric}": value
                        for metric, value in token_balanced_metrics(y_eval[mask], pred_eval[mask], tokens_eval[mask]).items()
                        if metric != "mse"
                    })
                else:
                    subgroup[f"{name}_equal_token_n_transactions"] = 0
                    subgroup[f"{name}_equal_token_n_tokens"] = 0
            final_results.append({
                "collection": "BAYC", "encoding": encoding, "family": family,
                "development_selected": key == f'{freeze["development_selected_model"]["encoding"]}__{freeze["development_selected_model"]["family"]}',
                "selected_config": json.dumps(spec["selected_config"], sort_keys=True),
                "pre2025_validation_equal_token_rmse": spec["pooled_pre2025_validation_equal_token_rmse"],
                **{f"evaluation_equal_token_{name}": value for name, value in macro.items() if name != "mse"},
                **{f"evaluation_transaction_{name}": value for name, value in micro.items()},
                **subgroup,
            })
            np.savez_compressed(
                OUT / "predictions" / f"{encoding.replace('-', '_')}_{family}.npz",
                source_rows=rows_eval, tokens=tokens_eval, y_lrp=y_eval, predictions_lrp=pred_eval,
                known_token=known_tx,
            )
    write_csv(OUT / "fixed_out_of_time_evaluation_18_combinations.csv", final_results)

    # Baselines use training information only.
    _, train_inverse, train_counts = np.unique(tx_tokens, return_inverse=True, return_counts=True)
    train_weights = 1.0 / train_counts[train_inverse]
    train_weights /= train_weights.sum()
    train_token_z_mean = float(np.dot(train_weights, final_z))
    constant_lrp = float(common.target_inverse(np.asarray([train_token_z_mean]), final_parameters)[0])
    baselines = []
    for sample, y, tokens in (("training", data["y"], tx_tokens), ("fixed_out_of_time_evaluation", y_eval, tokens_eval)):
        for baseline, value in (("LRP_zero", 0.0), ("training_token_asinh_mean", constant_lrp)):
            macro = token_balanced_metrics(y, np.full(len(y), value), tokens)
            micro = transaction_metrics(y, np.full(len(y), value))
            baselines.append({
                "sample": sample, "baseline": baseline, "constant_prediction": value,
                **{f"equal_token_{name}": val for name, val in macro.items() if name != "mse"},
                **{f"transaction_{name}": val for name, val in micro.items()},
            })
    write_csv(OUT / "baseline_results.csv", baselines)

    ranked = sorted(final_results, key=lambda row: row["evaluation_equal_token_rmse"])
    selected_key = f'{freeze["development_selected_model"]["encoding"]}__{freeze["development_selected_model"]["family"]}'
    selected_eval = next(row for row in final_results if f'{row["encoding"]}__{row["family"]}' == selected_key)
    lines = [
        "# BAYC temporal token-balanced metadata experiment", "",
        "All selection used pre-2025 data. The fixed retrospective out-of-time evaluation covers 2025-01-01 through 2026-04-13.", "",
        f'Pre-2025 development-selected model: **{freeze["development_selected_model"]["encoding"]} + {freeze["development_selected_model"]["family"]}**.', "",
        "| Rank | Encoding | Family | Train token RMSE | Pre-2025 validation token RMSE | 2025+ token RMSE | 2025+ token MAE | 2025+ token R2 | 2025+ transaction RMSE |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    training_lookup = {(row["encoding"], row["family"]): row for row in training_results}
    for rank, row in enumerate(ranked, 1):
        train = training_lookup[row["encoding"], row["family"]]
        lines.append(
            f'| {rank} | {row["encoding"]} | {row["family"]} | {train["train_equal_token_rmse"]:.9f} | '
            f'{row["pre2025_validation_equal_token_rmse"]:.9f} | {row["evaluation_equal_token_rmse"]:.9f} | '
            f'{row["evaluation_equal_token_mae"]:.9f} | {row["evaluation_equal_token_r2"]:.9f} | '
            f'{row["evaluation_transaction_rmse"]:.9f} |'
        )
    (OUT / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for item in json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))["inputs"]:
        assert sha256(ROOT / item["path"]) == item["sha256"]
    write_json(OUT / "completion.json", {
        "passed": True, "expected_fits": EXPECTED_FITS, "actual_fits": completed,
        "evaluation_loaded_after_freeze": True, "combinations": len(final_results),
        "development_selected_model": freeze["development_selected_model"],
        "development_selected_model_evaluation": selected_eval,
        "best_evaluation_descriptive_not_for_selection": ranked[0],
        "evaluation_known_tokens": int(known_eval_token.sum()),
        "evaluation_unseen_tokens": int((~known_eval_token).sum()),
        "elapsed_seconds": time.time() - start_time, "manuscript_modified": False,
    })
    write_json(OUT / "progress.json", {
        "stage": "complete", "completed_fits": completed, "expected_fits": EXPECTED_FITS,
    })


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        main()
