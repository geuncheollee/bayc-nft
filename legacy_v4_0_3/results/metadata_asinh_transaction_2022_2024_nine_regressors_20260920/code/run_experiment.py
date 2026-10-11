"""Robust-asinh metadata benchmark for BAYC and MAYC, trained on 2022-2024 transactions.

Two metadata representations and eight regression families are tuned using only
2024 Q2-Q4 forward validation.  The fixed retrospective 2025+ evaluation data
are loaded only after all 36 final specifications have been frozen.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "metadata_asinh_transaction_2022_2024_eight_regressors_20260920_v1"
COMMON_CODE = ROOT / "results" / "metadata_asinh_nine_regressors_20260920" / "code"
TEMPORAL_CODE = ROOT / "results" / "metadata_temporal_token_balanced_20260920" / "code"
CANONICAL_CODE = ROOT / "results" / "seven_encoder_full_rerun" / "code"
for path in (COMMON_CODE, TEMPORAL_CODE, CANONICAL_CODE):
    sys.path.insert(0, str(path))

import canonical_runner as cr
import run_experiment as common
import run_temporal_token_balanced as temporal


SEED = 20260908
COLLECTIONS = ("BAYC", "MAYC")
ENCODINGS = ("one-hot", "TF-IDF")
FAMILIES = tuple(family for family in common.FAMILIES if family != "RandomForest")
TRAIN_START = "2022-01-01"
TRAIN_END = "2025-01-01"
TUNING_QUARTERS = tuple(cr.FINAL_Q)
GRID_FITS = len(COLLECTIONS) * len(ENCODINGS) * len(TUNING_QUARTERS) * sum(
    len(common.GRIDS[family]) for family in FAMILIES
)
FINAL_FITS = len(COLLECTIONS) * len(ENCODINGS) * len(FAMILIES)
EXPECTED_FITS = GRID_FITS + FINAL_FITS

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def progress(stage: str, completed: int, **extra) -> None:
    write_json(
        OUT / "progress.json",
        {
            "updated_utc": now(), "stage": stage, "completed_fits": completed,
            "expected_fits": EXPECTED_FITS, "percent": 100 * completed / EXPECTED_FITS,
            **extra,
        },
    )


def windowed_development(collection: str) -> dict:
    full = cr.load_development(collection)
    keep = (full["times"] >= TRAIN_START) & (full["times"] < TRAIN_END)
    data = {
        "y": full["y"][keep], "tokens": full["tokens"][keep],
        "rowids": full["rowids"][keep], "times": full["times"][keep],
        "X": full["X"][keep], "columns": full["columns"],
    }
    assert len(data["y"]) and np.all(data["times"] >= TRAIN_START) and np.all(data["times"] < TRAIN_END)
    assert np.isfinite(data["y"]).all()
    return data


def fit_checkpoint(data: dict, collection: str, encoding: str, family: str, config: dict,
                   origin: str, end: str, prep: dict, parameters: dict, train: np.ndarray,
                   valid: np.ndarray, completed: int) -> tuple[dict, int]:
    config_index = common.GRIDS[family].index(config)
    path = OUT / "checkpoints" / collection / encoding.replace("-", "_") / origin / family / f"{config_index}.json"
    if path.exists():
        row = read_json(path)
        assert row["config"] == config
        assert row["training_row_sha256"] == array_sha(data["rowids"][train])
        assert row["validation_row_sha256"] == array_sha(data["rowids"][valid])
        return row, completed + 1

    transformed = common.target_forward(data["y"], parameters)
    started = time.perf_counter()
    model, status = temporal.fit_model(prep, family, config, transformed[train])
    if status["valid"]:
        features = prep["raw_valid"] if family in common.TREE_FAMILIES else prep["zvalid"]
        prediction = common.target_inverse(common.predict_model(model, features), parameters)
        micro = temporal.transaction_metrics(data["y"][valid], prediction)
        macro = temporal.token_balanced_metrics(data["y"][valid], prediction, data["tokens"][valid].astype(str))
        finite = bool(np.isfinite(prediction).all())
    else:
        micro = macro = None
        finite = False
    row = {
        "collection": collection, "encoding": encoding, "family": family, "config": config,
        "origin": origin, "end": end, "valid": bool(status["valid"] and finite),
        "status": status, "train_n": int(len(train)), "validation_n": int(len(valid)),
        "training_min": str(data["times"][train].min()), "training_max": str(data["times"][train].max()),
        "training_row_sha256": array_sha(data["rowids"][train]),
        "validation_row_sha256": array_sha(data["rowids"][valid]),
        "target_parameters": parameters, "dimension": int(prep["dimension"]),
        "transaction_mse": None if micro is None else micro["rmse"] ** 2,
        "transaction_rmse": None if micro is None else micro["rmse"],
        "equal_token_mse": None if macro is None else macro["mse"],
        "equal_token_rmse": None if macro is None else macro["rmse"],
        "equal_token_n": None if macro is None else macro["n_tokens"],
        "seconds": time.perf_counter() - started,
    }
    write_json(path, row)
    completed += 1
    progress("pre2025_tuning", completed, collection=collection, encoding=encoding,
             family=family, quarter=origin, config_index=config_index)
    del model
    return row, completed


def pooled_scores(rows: list[dict], collection: str, encoding: str, family: str) -> list[dict]:
    output = []
    for config in common.GRIDS[family]:
        parts = [row for row in rows if row["collection"] == collection and row["encoding"] == encoding
                 and row["family"] == family and row["config"] == config]
        valid = len(parts) == len(TUNING_QUARTERS) and all(row["valid"] for row in parts)
        mse = (
            sum(row["transaction_mse"] * row["validation_n"] for row in parts)
            / sum(row["validation_n"] for row in parts)
        ) if valid else None
        output.append({"config": config, "valid": valid, "mse_lrp": mse})
    return output


def main() -> None:
    if (OUT / "completion.json").exists():
        raise FileExistsError(f"Completed result already exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    (OUT / "predictions").mkdir(exist_ok=True)
    started = time.time()

    source_paths = [
        Path(__file__), Path(common.__file__), Path(temporal.__file__), Path(cr.__file__),
        cr.base.OLD / "metadata_normalized.jsonl",
    ]
    for collection in COLLECTIONS:
        source_paths.extend([
            cr.base.TARGET / f"{collection.lower()}_development_targets.jsonl",
            cr.base.TARGET / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl",
        ])
    specification = {
        "collections": COLLECTIONS,
        "training_window": "2022-01-01 through 2024-12-31",
        "training_unit": "one uniformly weighted row per eligible transaction",
        "target": "fold-fitted robust-asinh of LRP; z=asinh((LRP-training median)/training normalized MAD)",
        "target_parameter_policy": "estimated only from the corresponding training transactions",
        "encodings": ENCODINGS, "families": FAMILIES, "grids": common.GRIDS,
        "selection": "pooled 2024 Q2-Q4 transaction-weighted RMSE on original LRP after inverse transform",
        "evaluation": "fixed retrospective out-of-time window 2025-01-01 through 2026-04-13; unavailable until freeze",
        "reported_scale": "original LRP", "seed": SEED,
    }
    spec_path = OUT / "experiment_specification.json"
    if not spec_path.exists():
        write_json(spec_path, specification)
        write_json(OUT / "run_manifest.json", {
            "start_utc": now(), "expected_fits": EXPECTED_FITS, "python": sys.version,
            "machine": platform.platform(), "specification_sha256": sha256(spec_path),
            "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_paths],
        })
    else:
        assert read_json(spec_path) == json.loads(json.dumps(specification))
    manifest = read_json(OUT / "run_manifest.json")
    assert sha256(spec_path) == manifest["specification_sha256"]
    for item in manifest["inputs"]:
        assert sha256(ROOT / item["path"]) == item["sha256"]

    data_by_collection = {collection: windowed_development(collection) for collection in COLLECTIONS}
    cohort_rows = []
    for collection, data in data_by_collection.items():
        cohort_rows.append({
            "collection": collection, "training_transactions": len(data["y"]),
            "training_tokens": len(np.unique(data["tokens"])),
            "training_min_time": min(data["times"]), "training_max_time": max(data["times"]),
        })
    write_csv(OUT / "cohort_audit.csv", cohort_rows)

    completed = len(list((OUT / "checkpoints").rglob("*.json"))) if (OUT / "checkpoints").exists() else 0
    tuning_rows = []
    for collection, data in data_by_collection.items():
        for encoding in ENCODINGS:
            for origin, end in TUNING_QUARTERS:
                train = np.flatnonzero((data["times"] >= TRAIN_START) & (data["times"] < origin))
                valid = np.flatnonzero((data["times"] >= origin) & (data["times"] < end))
                assert len(train) and len(valid) and data["times"][train].max() < data["times"][valid].min()
                parameters = common.fit_target_transform(data["y"][train])
                prep = common.prepare(data, train, valid, encoding)
                for family in FAMILIES:
                    for config in common.GRIDS[family]:
                        row, completed = fit_checkpoint(
                            data, collection, encoding, family, config, origin, end,
                            prep, parameters, train, valid, completed,
                        )
                        tuning_rows.append(row)
                del prep
                gc.collect()
    write_csv(OUT / "pre2025_tuning_all_configs.csv", [
        {
            **{key: value for key, value in row.items() if key not in {"config", "status", "target_parameters"}},
            "config": json.dumps(row["config"], sort_keys=True),
            "status": json.dumps(row["status"], sort_keys=True),
            "target_parameters": json.dumps(row["target_parameters"], sort_keys=True),
        }
        for row in tuning_rows
    ])

    # Fit all final models and freeze every specification before reading evaluation labels.
    freeze = {"created_utc": now(), "evaluation_labels_loaded_before_freeze": False, "models": {}}
    selection_rows = []
    training_rows = []
    for collection, data in data_by_collection.items():
        all_train = np.arange(len(data["y"]), dtype=int)
        empty = np.asarray([], dtype=int)
        parameters = common.fit_target_transform(data["y"])
        transformed = common.target_forward(data["y"], parameters)
        for encoding in ENCODINGS:
            prep = common.prepare(data, all_train, empty, encoding)
            for family in FAMILIES:
                scores = pooled_scores(tuning_rows, collection, encoding, family)
                choices = common.ordered_choices(family, scores)
                if not choices:
                    raise RuntimeError(f"No valid configuration for {collection} {encoding} {family}")
                model = status = selected = None
                attempts = []
                for choice in choices:
                    candidate, candidate_status = temporal.fit_model(prep, family, choice["config"], transformed)
                    attempts.append({"config": choice["config"], "status": candidate_status})
                    completed += 1
                    if candidate_status["valid"]:
                        model, status, selected = candidate, candidate_status, choice
                        break
                if model is None:
                    raise RuntimeError(f"Final refit failed: {collection} {encoding} {family}: {attempts}")
                features = prep["raw_train"] if family in common.TREE_FAMILIES else prep["ztrain"]
                pred_train = common.target_inverse(common.predict_model(model, features), parameters)
                macro = temporal.token_balanced_metrics(data["y"], pred_train, data["tokens"].astype(str))
                micro = temporal.transaction_metrics(data["y"], pred_train)
                key = f"{collection}__{encoding}__{family}"
                model_path = OUT / "models" / f"{collection}_{encoding.replace('-', '_')}_{family}.joblib"
                joblib.dump({
                    "encoding": prep["encoding"], "scaler": prep["scaler"], "model": model,
                    "family": family, "metadata_encoding": encoding, "collection": collection,
                    "columns": data["columns"], "target_parameters": parameters,
                    "training_tokens": np.unique(data["tokens"]),
                }, model_path, compress=3)
                freeze["models"][key] = {
                    "collection": collection, "encoding": encoding, "family": family,
                    "selected_config": selected["config"],
                    "pre2025_validation_transaction_rmse": float(np.sqrt(selected["mse_lrp"])),
                    "model_path": str(model_path.relative_to(OUT)), "model_sha256": sha256(model_path),
                    "target_parameters": parameters, "attempts": attempts,
                }
                selection_rows.append({
                    "collection": collection, "encoding": encoding, "family": family,
                    "selected_config": json.dumps(selected["config"], sort_keys=True),
                    "pre2025_validation_transaction_rmse": float(np.sqrt(selected["mse_lrp"])),
                })
                training_rows.append({
                    "collection": collection, "encoding": encoding, "family": family,
                    **{f"training_equal_token_{name}": value for name, value in macro.items() if name != "mse"},
                    **{f"training_transaction_{name}": value for name, value in micro.items()},
                })
                progress("final_refit", completed, collection=collection, encoding=encoding, family=family)
                del model
            del prep
            gc.collect()
    for collection in COLLECTIONS:
        candidates = [row for row in selection_rows if row["collection"] == collection]
        freeze.setdefault("development_selected", {})[collection] = min(
            candidates, key=lambda row: (row["pre2025_validation_transaction_rmse"], ENCODINGS.index(row["encoding"]), FAMILIES.index(row["family"]))
        )
    write_csv(OUT / "pre2025_model_selection_32_combinations.csv", selection_rows)
    write_csv(OUT / "training_performance_32_combinations.csv", training_rows)
    write_json(OUT / "selection_freeze.json", freeze)
    progress("selection_frozen_loading_evaluation", completed)

    # Fixed retrospective evaluation is intentionally loaded only after freeze.
    metadata = {(row["collection"], int(row["token_id"])): row for row in cr.lines(cr.base.OLD / "metadata_normalized.jsonl")}
    final_rows = []
    baseline_rows = []
    evaluation_counts = {}
    for collection, data in data_by_collection.items():
        eval_path = cr.base.TARGET / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl"
        eval_rows = cr.lines(eval_path)
        assert len(eval_rows) and all("2025-01-01" <= row["time"] < "2026-04-14" for row in eval_rows)
        y_eval = np.asarray([row["y_log_relative_price"] for row in eval_rows], dtype=np.float64)
        tokens_eval = np.asarray([str(row["token_id"]) for row in eval_rows])
        source_eval = np.asarray([row["source_row"] for row in eval_rows])
        unique_eval, inverse_eval = np.unique(tokens_eval, return_inverse=True)
        training_tokens = np.unique(data["tokens"].astype(str))
        known_unique = np.isin(unique_eval, training_tokens)
        known_transaction = known_unique[inverse_eval]
        evaluation_counts[collection] = {"transactions": len(y_eval), "tokens": len(unique_eval)}
        for encoding in ENCODINGS:
            for family in FAMILIES:
                key = f"{collection}__{encoding}__{family}"
                spec = freeze["models"][key]
                model_path = OUT / spec["model_path"]
                assert sha256(model_path) == spec["model_sha256"]
                bundle = joblib.load(model_path)
                X_unique = np.asarray([
                    [metadata[collection, int(token)][column] for column in bundle["columns"]]
                    for token in unique_eval
                ], dtype=object)
                raw = common.dense(bundle["encoding"].transform(X_unique))
                features = raw if family in common.TREE_FAMILIES else bundle["scaler"].transform(raw)
                prediction_unique = common.target_inverse(
                    common.predict_model(bundle["model"], features), bundle["target_parameters"]
                )
                prediction = prediction_unique[inverse_eval]
                assert np.isfinite(prediction).all()
                macro = temporal.token_balanced_metrics(y_eval, prediction, tokens_eval)
                micro = temporal.transaction_metrics(y_eval, prediction)
                subgroups = {}
                for name, mask in (("known", known_transaction), ("unseen", ~known_transaction)):
                    if mask.any():
                        values = temporal.token_balanced_metrics(y_eval[mask], prediction[mask], tokens_eval[mask])
                        subgroups.update({f"{name}_equal_token_{metric}": value for metric, value in values.items() if metric != "mse"})
                selected_model = freeze["development_selected"][collection]
                is_selected = encoding == selected_model["encoding"] and family == selected_model["family"]
                final_rows.append({
                    "collection": collection, "encoding": encoding, "family": family,
                    "development_selected": is_selected,
                    "selected_config": json.dumps(spec["selected_config"], sort_keys=True),
                    "pre2025_validation_transaction_rmse": spec["pre2025_validation_transaction_rmse"],
                    **{f"evaluation_equal_token_{name}": value for name, value in macro.items() if name != "mse"},
                    **{f"evaluation_transaction_{name}": value for name, value in micro.items()},
                    **subgroups,
                })
                np.savez_compressed(
                    OUT / "predictions" / f"{collection}_{encoding.replace('-', '_')}_{family}.npz",
                    source_rows=source_eval, tokens=tokens_eval, y_lrp=y_eval,
                    predictions_lrp=prediction, known_token=known_transaction,
                )

        parameters = common.fit_target_transform(data["y"])
        transformed = common.target_forward(data["y"], parameters)
        constant = float(common.target_inverse(np.asarray([transformed.mean()]), parameters)[0])
        for sample, y, tokens in (("training_2022_2024", data["y"], data["tokens"].astype(str)),
                                  ("evaluation_2025_plus", y_eval, tokens_eval)):
            for baseline, value in (("LRP_zero", 0.0), ("training_transaction_asinh_mean", constant)):
                macro = temporal.token_balanced_metrics(y, np.full(len(y), value), tokens)
                micro = temporal.transaction_metrics(y, np.full(len(y), value))
                baseline_rows.append({
                    "collection": collection, "sample": sample, "baseline": baseline,
                    "constant_prediction": value,
                    **{f"equal_token_{name}": val for name, val in macro.items() if name != "mse"},
                    **{f"transaction_{name}": val for name, val in micro.items()},
                })

    write_csv(OUT / "fixed_out_of_time_evaluation_32_combinations.csv", final_rows)
    write_csv(OUT / "baseline_results.csv", baseline_rows)

    training_lookup = {(row["collection"], row["encoding"], row["family"]): row for row in training_rows}
    summary = [
        "# Robust-asinh metadata benchmark: 2022-2024 transaction fits", "",
        "All tuning used pre-2025 data. Scores are on the original LRP scale after inverse robust-asinh transformation.", "",
    ]
    for collection in COLLECTIONS:
        selected = freeze["development_selected"][collection]
        summary.extend([
            f"## {collection}", "",
            f"Development-selected model: **{selected['encoding']} + {selected['family']}**.", "",
            "| Rank | Encoding | Family | Train transaction RMSE | Pre-2025 validation transaction RMSE | 2025+ transaction RMSE | 2025+ transaction MAE | 2025+ transaction R² | 2025+ equal-token RMSE |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        ranked = sorted([row for row in final_rows if row["collection"] == collection],
                        key=lambda row: row["evaluation_transaction_rmse"])
        for rank, row in enumerate(ranked, 1):
            train = training_lookup[collection, row["encoding"], row["family"]]
            summary.append(
                f"| {rank} | {row['encoding']} | {row['family']} | {train['training_transaction_rmse']:.9f} | "
                f"{row['pre2025_validation_transaction_rmse']:.9f} | {row['evaluation_transaction_rmse']:.9f} | "
                f"{row['evaluation_transaction_mae']:.9f} | {row['evaluation_transaction_r2']:.9f} | "
                f"{row['evaluation_equal_token_rmse']:.9f} |"
            )
        summary.append("")
    (OUT / "analysis_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    for item in manifest["inputs"]:
        assert sha256(ROOT / item["path"]) == item["sha256"]
    write_json(OUT / "completion.json", {
        "completed_utc": now(), "passed": True, "minimum_expected_fits": EXPECTED_FITS,
        "actual_fits_or_resumed_checkpoints": completed, "combinations": len(final_rows),
        "training_cohorts": cohort_rows, "evaluation_counts": evaluation_counts,
        "development_selected": freeze["development_selected"],
        "evaluation_loaded_after_freeze": True, "elapsed_seconds": time.time() - started,
        "manuscript_modified": False, "prior_results_overwritten": False,
    })
    progress("complete", completed)


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        main()
