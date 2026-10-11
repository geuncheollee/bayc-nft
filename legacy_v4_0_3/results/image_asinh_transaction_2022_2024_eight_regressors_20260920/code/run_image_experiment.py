"""Seven-encoder image-only benchmark under the revised 2022-2024 specification."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "image_asinh_transaction_2022_2024_eight_regressors_20260920_v1"
REGISTRY_PATH = ROOT / "results" / "seven_encoder_pca1024_rerun" / "embedding_registry.json"
PCA_CODE = ROOT / "results" / "seven_encoder_pca1024_rerun" / "code"
COMMON_CODE = ROOT / "results" / "metadata_asinh_nine_regressors_20260920" / "code"
TEMPORAL_CODE = ROOT / "results" / "metadata_temporal_token_balanced_20260920" / "code"
CANONICAL_CODE = ROOT / "results" / "seven_encoder_full_rerun" / "code"
for path in (COMMON_CODE, TEMPORAL_CODE, CANONICAL_CODE, PCA_CODE):
    sys.path.insert(0, str(path))

import canonical_runner as cr
import pca_runner as pr
import run_experiment as common
import run_temporal_token_balanced as temporal


SEED = 20260908
COLLECTIONS = ("BAYC", "MAYC")
ENCODERS = ("DINOv2", "CLIP", "SigLIP2", "SAM", "SDXL_VAE", "DreamSim", "AIM")
REDUCED = {"SDXL_VAE", "DreamSim"}
FAMILIES = tuple(family for family in common.FAMILIES if family != "RandomForest")
TREE_FAMILIES = {"HistGradientBoosting", "XGBoost", "LightGBM"}
TRAIN_START = "2022-01-01"
TRAIN_END = "2025-01-01"
TUNING_QUARTERS = tuple(cr.FINAL_Q)
GRID_SIZE = sum(len(common.GRIDS[family]) for family in FAMILIES)
EXPECTED_TUNING_FITS = len(COLLECTIONS) * len(ENCODERS) * len(TUNING_QUARTERS) * GRID_SIZE
EXPECTED_FINAL_FITS = len(COLLECTIONS) * len(ENCODERS) * len(FAMILIES)
EXPECTED_FITS = EXPECTED_TUNING_FITS + EXPECTED_FINAL_FITS


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
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def progress(stage: str, completed: int, **extra) -> None:
    write_json(OUT / "progress.json", {
        "updated_utc": now(), "stage": stage, "completed_fits": completed,
        "expected_fits": EXPECTED_FITS, "percent": 100 * completed / EXPECTED_FITS,
        **extra,
    })


def finite_matrix(matrix, block: int = 256) -> bool:
    return all(np.isfinite(np.asarray(matrix[start:start + block])).all()
               for start in range(0, len(matrix), block))


def load_encoder_data(collection: str, encoder: str, registry: dict) -> dict:
    base = cr.load_development(collection)
    keep = (base["times"] >= TRAIN_START) & (base["times"] < TRAIN_END)
    info = registry[encoder][collection]
    ids = np.asarray(info["token_ids"], dtype=np.int64)
    assert len(ids) == len(np.unique(ids))
    matrix_path = ROOT / info["matrix"]
    assert sha256(matrix_path) == info["matrix_sha256"]
    image = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    assert image.shape == (len(ids), info["dimensions"])
    mapping = {int(token): index for index, token in enumerate(ids)}
    tokens = base["tokens"][keep]
    missing = sorted(set(map(int, tokens)) - set(mapping))
    if missing:
        raise RuntimeError(f"{collection} {encoder} missing tokens: {missing[:20]}")
    data = {
        "collection": collection, "encoder": encoder,
        "y": base["y"][keep], "tokens": tokens,
        "rowids": base["rowids"][keep], "times": base["times"][keep],
        "image": image, "feature_tokens": ids,
        "index": np.asarray([mapping[int(token)] for token in tokens], dtype=np.int64),
        "native_dimension": int(image.shape[1]), "native_dtype": str(image.dtype),
        "matrix_path": str(matrix_path.relative_to(ROOT)),
        "matrix_sha256": info["matrix_sha256"], "_native_sha256": info["matrix_sha256"],
        "_encoder": encoder, "_collection": collection, "_sample": "original_2022_start",
    }
    assert np.array_equal(ids[data["index"]], tokens)
    assert np.isfinite(data["y"]).all()
    return data


def projected_image(data: dict, train: np.ndarray, origin: str) -> tuple[np.ndarray, dict | None]:
    if data["encoder"] not in REDUCED:
        return data["image"], None
    folder = OUT / "reduced_embeddings" / data["collection"] / data["encoder"] / origin
    _, matrix = pr.fit_projection(data, train, origin, folder, components=1024)
    manifest = read_json(folder / "manifest.json")
    return matrix, {
        "type": "training-fold-only PCA", "components": 1024,
        "manifest": str((folder / "manifest.json").relative_to(OUT)),
        "manifest_sha256": sha256(folder / "manifest.json"),
        "matrix": str((folder / "embeddings_1024.npy").relative_to(OUT)),
        "matrix_sha256": manifest["embeddings_sha256"],
        "pca": str((folder / "pca.joblib").relative_to(OUT)),
        "pca_sha256": manifest["pca_sha256"],
    }


def prepare_image(data: dict, image: np.ndarray, train: np.ndarray, valid: np.ndarray,
                  transformed_target: np.ndarray) -> dict:
    train_feature_rows = data["index"][train]
    valid_feature_rows = data["index"][valid]
    unique, inverse, counts = np.unique(train_feature_rows, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=transformed_target[train]) / counts
    raw_unique = np.asarray(image[unique], dtype=np.float64)
    mask = np.ptp(raw_unique, axis=0) > 0
    if not mask.any():
        raise ValueError("All image dimensions are constant")
    scaler = StandardScaler().fit(raw_unique[:, mask], sample_weight=counts)
    z_unique = scaler.transform(raw_unique[:, mask])
    z_valid = scaler.transform(np.asarray(image[valid_feature_rows][:, mask], dtype=np.float64)) if len(valid) else np.empty((0, int(mask.sum())))
    selected = np.linspace(0, len(unique) - 1, min(256, len(unique)), dtype=int)
    rank = int(np.linalg.matrix_rank(z_unique[selected] - z_unique[selected].mean(axis=0)))
    weighted_mean = np.average(z_unique, axis=0, weights=counts)
    assert np.isfinite(z_unique).all() and np.isfinite(z_valid).all()
    return {
        "unique": unique, "inverse": inverse, "counts": counts, "means": means,
        "z_unique": z_unique, "z_valid": z_valid, "mask": mask, "scaler": scaler,
        "rank": rank, "weighted_mean": weighted_mean,
        "dimension": int(mask.sum()), "valid_feature_rows": valid_feature_rows,
    }


def fit_model(prep: dict, family: str, config: dict, transformed_target_train: np.ndarray):
    started = time.perf_counter()
    model = offset = None
    prediction = None
    try:
        if family == "PLS" and config["components"] > prep["rank"]:
            raise ValueError("components exceed deterministic rank lower bound")
        model = common.build_model(family, config, "development")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if family == "PLS":
                centered_x = (prep["z_unique"] - prep["weighted_mean"]) * np.sqrt(prep["counts"] / 2)[:, None]
                target_mean = float(np.mean(transformed_target_train))
                centered_y = (prep["means"] - target_mean) * np.sqrt(prep["counts"] / 2)
                signed_x = np.concatenate([centered_x, -centered_x])
                signed_y = np.concatenate([centered_y, -centered_y])
                model.fit(signed_x, signed_y)
                offset = {"x": prep["weighted_mean"], "y": target_mean}
                prediction = (np.asarray(model.predict(prep["z_valid"] - offset["x"])).reshape(-1) + offset["y"]
                              if len(prep["z_valid"]) else np.empty(0, dtype=np.float64))
            elif family == "LinearSVR":
                model = cr.vision.CountedPrimalSVR(config["C"], config["epsilon"]).fit_counts(
                    prep["z_unique"], prep["inverse"], transformed_target_train
                )
                prediction = (np.asarray(model.predict(prep["z_valid"])).reshape(-1)
                              if len(prep["z_valid"]) else np.empty(0, dtype=np.float64))
            else:
                model.fit(prep["z_unique"], prep["means"], sample_weight=prep["counts"])
                prediction = (common.predict_model(model, prep["z_valid"])
                              if len(prep["z_valid"]) else np.empty(0, dtype=np.float64))
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        finite = bool(np.isfinite(prediction).all())
        if family == "PLS":
            converged &= len(model.n_iter_) == config["components"]
            finite &= bool(np.isfinite(model.coef_).all())
        status = {
            "valid": bool(converged and finite), "converged": bool(converged), "finite": finite,
            "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in caught],
            "n_iter": np.asarray(getattr(model, "n_iter_", [])).tolist(),
            "seconds": time.perf_counter() - started,
        }
        if family == "LinearSVR":
            status.update(primal_objective=float(model.objective_),
                          primal_gradient_norm=float(model.gradient_norm_),
                          objective_gap_upper_bound=float(model.objective_gap_upper_bound_))
    except Exception as error:
        status = {"valid": False, "error": repr(error), "seconds": time.perf_counter() - started}
    return model, offset, prediction, status


def fit_checkpoint(data: dict, image: np.ndarray, projection: dict | None, train: np.ndarray,
                   valid: np.ndarray, origin: str, end: str, prep: dict, parameters: dict,
                   transformed: np.ndarray, family: str, config: dict, completed: int) -> tuple[dict, int]:
    number = common.GRIDS[family].index(config)
    path = OUT / "checkpoints" / data["collection"] / data["encoder"] / origin / family / f"{number}.json"
    if path.exists():
        row = read_json(path)
        assert row["config"] == config
        assert row["training_row_sha256"] == array_sha(data["rowids"][train])
        return row, completed + 1
    model, offset, pred_z, status = fit_model(prep, family, config, transformed[train])
    if status["valid"]:
        pred_lrp = common.target_inverse(pred_z, parameters)
        micro = temporal.transaction_metrics(data["y"][valid], pred_lrp)
        macro = temporal.token_balanced_metrics(data["y"][valid], pred_lrp, data["tokens"][valid].astype(str))
    else:
        micro = macro = None
    row = {
        "collection": data["collection"], "encoder": data["encoder"], "family": family,
        "config": config, "origin": origin, "end": end, "valid": bool(status["valid"]),
        "status": status, "train_n": len(train), "validation_n": len(valid),
        "training_min": str(data["times"][train].min()), "training_max": str(data["times"][train].max()),
        "training_row_sha256": array_sha(data["rowids"][train]),
        "validation_row_sha256": array_sha(data["rowids"][valid]),
        "target_parameters": parameters, "analysis_dimension": prep["dimension"],
        "native_dimension": data["native_dimension"], "projection": projection,
        "transaction_mse": None if micro is None else micro["rmse"] ** 2,
        "transaction_rmse": None if micro is None else micro["rmse"],
        "equal_token_mse": None if macro is None else macro["mse"],
        "equal_token_rmse": None if macro is None else macro["rmse"],
    }
    write_json(path, row)
    completed += 1
    progress("pre2025_tuning", completed, collection=data["collection"], encoder=data["encoder"],
             quarter=origin, family=family, config_index=number,
             available_ram_gib=psutil.virtual_memory().available / 2**30)
    del model
    return row, completed


def pooled_scores(rows: list[dict], collection: str, encoder: str, family: str) -> list[dict]:
    scores = []
    for config in common.GRIDS[family]:
        parts = [row for row in rows if row["collection"] == collection and row["encoder"] == encoder
                 and row["family"] == family and row["config"] == config]
        valid = len(parts) == 3 and all(row["valid"] for row in parts)
        mse = (sum(row["transaction_mse"] * row["validation_n"] for row in parts)
               / sum(row["validation_n"] for row in parts)) if valid else None
        scores.append({"config": config, "valid": valid, "mse_lrp": mse})
    return scores


def main() -> None:
    if (OUT / "completion.json").exists():
        raise FileExistsError(f"Completed output exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("models", "predictions", "checkpoints", "reduced_embeddings"):
        (OUT / folder).mkdir(exist_ok=True)
    started = time.time()
    registry = read_json(REGISTRY_PATH)
    specification = {
        "stage": "image-only regression",
        "collections": COLLECTIONS, "encoders": ENCODERS, "families": FAMILIES,
        "grids": {family: common.GRIDS[family] for family in FAMILIES},
        "training_window": "2022-01-01 through 2024-12-31",
        "training_unit": "uniform transaction weight, computed exactly with repeated-feature sufficient statistics",
        "target": "fold-fitted robust-asinh LRP; all selection and reporting on inverse-transformed original LRP",
        "selection": "pooled 2024 Q2-Q4 transaction-weighted RMSE",
        "evaluation": "fixed retrospective 2025-01-01 through 2026-04-13; loaded only after freeze",
        "dimensions": "stored representation for DINOv2/CLIP/SigLIP2/SAM/AIM; fold-local PCA1024 for SDXL_VAE/DreamSim",
        "random_forest_excluded": True, "seed": SEED,
    }
    spec_path = OUT / "experiment_specification.json"
    if not spec_path.exists():
        write_json(spec_path, specification)
        inputs = [Path(__file__), REGISTRY_PATH, Path(common.__file__), Path(temporal.__file__), Path(pr.__file__), Path(cr.__file__)]
        for collection in COLLECTIONS:
            inputs.extend([cr.base.TARGET / f"{collection.lower()}_development_targets.jsonl",
                           cr.base.TARGET / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl"])
        for encoder in ENCODERS:
            for collection in COLLECTIONS:
                info = registry[encoder][collection]
                inputs.extend([ROOT / info["matrix"], ROOT / info["manifest"]])
        write_json(OUT / "run_manifest.json", {
            "start_utc": now(), "expected_fits": EXPECTED_FITS, "python": sys.version,
            "machine": platform.platform(), "available_ram_gib": psutil.virtual_memory().available / 2**30,
            "specification_sha256": sha256(spec_path),
            "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in inputs],
        })
    else:
        assert read_json(spec_path) == json.loads(json.dumps(specification))
    manifest = read_json(OUT / "run_manifest.json")
    assert sha256(spec_path) == manifest["specification_sha256"]

    # Full input integrity audit before any fitting.
    audit_rows = []
    for collection in COLLECTIONS:
        for encoder in ENCODERS:
            data = load_encoder_data(collection, encoder, registry)
            unique_transactions = np.unique(data["tokens"])
            audit_rows.append({
                "collection": collection, "encoder": encoder,
                "matrix_path": data["matrix_path"], "matrix_sha256": data["matrix_sha256"],
                "native_dimension": data["native_dimension"], "dtype": data["native_dtype"],
                "matrix_rows": len(data["image"]), "registry_token_ids": len(data["feature_tokens"]),
                "duplicate_registry_token_ids": len(data["feature_tokens"]) - len(np.unique(data["feature_tokens"])),
                "training_transactions": len(data["y"]), "training_tokens": len(unique_transactions),
                "missing_training_tokens": 0, "row_alignment_verified": True,
                "finite_matrix": finite_matrix(data["image"]),
            })
            del data
            gc.collect()
    if not all(row["finite_matrix"] and row["row_alignment_verified"] and row["missing_training_tokens"] == 0 for row in audit_rows):
        raise RuntimeError("Embedding audit failed")
    write_csv(OUT / "embedding_audit.csv", audit_rows)

    # Existing checkpoints are counted as the traversal encounters them.  Starting
    # from their count would double-count resumed work in progress.json.
    completed = 0
    tuning_rows = []
    data_cache: dict[tuple[str, str], dict] = {}
    for collection in COLLECTIONS:
        for encoder in ENCODERS:
            data = load_encoder_data(collection, encoder, registry)
            data_cache[collection, encoder] = data
            for origin, end in TUNING_QUARTERS:
                train = np.flatnonzero((data["times"] >= TRAIN_START) & (data["times"] < origin))
                valid = np.flatnonzero((data["times"] >= origin) & (data["times"] < end))
                assert len(train) and len(valid) and data["times"][train].max() < data["times"][valid].min()
                parameters = common.fit_target_transform(data["y"][train])
                transformed = common.target_forward(data["y"], parameters)
                image, projection = projected_image(data, train, origin)
                prep = prepare_image(data, image, train, valid, transformed)
                for family in FAMILIES:
                    for config in common.GRIDS[family]:
                        row, completed = fit_checkpoint(data, image, projection, train, valid, origin, end,
                                                        prep, parameters, transformed, family, config, completed)
                        tuning_rows.append(row)
                del prep, image
                gc.collect()
            del data_cache[collection, encoder]
            gc.collect()
    write_csv(OUT / "pre2025_tuning_all_configs.csv", [{
        **{key: value for key, value in row.items() if key not in {"config", "status", "target_parameters", "projection"}},
        "config": json.dumps(row["config"], sort_keys=True),
        "status": json.dumps(row["status"], sort_keys=True),
        "target_parameters": json.dumps(row["target_parameters"], sort_keys=True),
        "projection": json.dumps(row["projection"], sort_keys=True),
    } for row in tuning_rows])

    # Refit all 112 candidates; freeze before loading any 2025+ target.
    freeze = {"created_utc": now(), "evaluation_labels_loaded_before_freeze": False, "models": {}}
    selection_rows, training_rows = [], []
    for collection in COLLECTIONS:
        for encoder in ENCODERS:
            data = load_encoder_data(collection, encoder, registry)
            train = np.arange(len(data["y"]), dtype=int)
            empty = np.asarray([], dtype=int)
            parameters = common.fit_target_transform(data["y"])
            transformed = common.target_forward(data["y"], parameters)
            image, projection = projected_image(data, train, "2025-01-01")
            prep = prepare_image(data, image, train, empty, transformed)
            for family in FAMILIES:
                scores = pooled_scores(tuning_rows, collection, encoder, family)
                choices = common.ordered_choices(family, scores)
                attempts = []
                model = offset = selected = None
                for choice in choices:
                    candidate, candidate_offset, _, status = fit_model(prep, family, choice["config"], transformed)
                    attempts.append({"config": choice["config"], "status": status})
                    completed += 1
                    if status["valid"]:
                        model, offset, selected = candidate, candidate_offset, choice
                        break
                if model is None:
                    raise RuntimeError(
                        f"Final refit failed: {collection} {encoder} {family}; attempts={attempts}"
                    )
                if family == "PLS":
                    pred_z = np.asarray(model.predict(prep["z_unique"] - offset["x"])).reshape(-1) + offset["y"]
                else:
                    pred_z = common.predict_model(model, prep["z_unique"])
                pred_unique = common.target_inverse(pred_z, parameters)
                lookup = np.full(len(data["feature_tokens"]), np.nan)
                lookup[prep["unique"]] = pred_unique
                train_pred = lookup[data["index"]]
                assert np.isfinite(train_pred).all()
                micro = temporal.transaction_metrics(data["y"], train_pred)
                macro = temporal.token_balanced_metrics(data["y"], train_pred, data["tokens"].astype(str))
                key = f"{collection}__{encoder}__{family}"
                model_path = OUT / "models" / f"{collection}_{encoder}_{family}.joblib"
                joblib.dump({
                    "model": model, "offset": offset, "scaler": prep["scaler"], "mask": prep["mask"],
                    "family": family, "encoder": encoder, "collection": collection,
                    "target_parameters": parameters, "projection": projection,
                    "training_tokens": np.unique(data["tokens"]),
                }, model_path, compress=3)
                freeze["models"][key] = {
                    "collection": collection, "encoder": encoder, "family": family,
                    "selected_config": selected["config"],
                    "pre2025_validation_transaction_rmse": float(np.sqrt(selected["mse_lrp"])),
                    "model_path": str(model_path.relative_to(OUT)), "model_sha256": sha256(model_path),
                    "projection": projection, "target_parameters": parameters, "attempts": attempts,
                }
                selection_rows.append({
                    "collection": collection, "encoder": encoder, "family": family,
                    "native_dimension": data["native_dimension"], "analysis_dimension": prep["dimension"],
                    "selected_config": json.dumps(selected["config"], sort_keys=True),
                    "pre2025_validation_transaction_rmse": float(np.sqrt(selected["mse_lrp"])),
                })
                training_rows.append({
                    "collection": collection, "encoder": encoder, "family": family,
                    **{f"training_equal_token_{name}": value for name, value in macro.items() if name != "mse"},
                    **{f"training_transaction_{name}": value for name, value in micro.items()},
                })
                progress("final_refit", completed, collection=collection, encoder=encoder, family=family)
                del model
            del prep, image, data
            gc.collect()
    for collection in COLLECTIONS:
        for encoder in ENCODERS:
            candidates = [row for row in selection_rows if row["collection"] == collection and row["encoder"] == encoder]
            freeze.setdefault("selected_image_learner_by_encoder", {}).setdefault(collection, {})[encoder] = min(
                candidates, key=lambda row: (row["pre2025_validation_transaction_rmse"], FAMILIES.index(row["family"]))
            )
    write_csv(OUT / "pre2025_image_model_selection_112_combinations.csv", selection_rows)
    write_csv(OUT / "training_performance_112_combinations.csv", training_rows)
    write_json(OUT / "selection_freeze.json", freeze)
    progress("selection_frozen_loading_evaluation", completed)

    final_rows = []
    for collection in COLLECTIONS:
        eval_rows = cr.lines(cr.base.TARGET / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl")
        y_eval = np.asarray([row["y_log_relative_price"] for row in eval_rows], dtype=np.float64)
        tokens_eval = np.asarray([str(row["token_id"]) for row in eval_rows])
        source_eval = np.asarray([row["source_row"] for row in eval_rows])
        assert len(y_eval) == {"BAYC": 5120, "MAYC": 13019}[collection]
        assert all("2025-01-01" <= row["time"] < "2026-04-14" for row in eval_rows)
        unique_tokens, inverse = np.unique(tokens_eval, return_inverse=True)
        for encoder in ENCODERS:
            info = registry[encoder][collection]
            ids = np.asarray(info["token_ids"], dtype=np.int64)
            mapping = {int(token): index for index, token in enumerate(ids)}
            feature_rows = np.asarray([mapping[int(token)] for token in unique_tokens])
            native = np.load(ROOT / info["matrix"], mmap_mode="r", allow_pickle=False)
            for family in FAMILIES:
                key = f"{collection}__{encoder}__{family}"
                spec = freeze["models"][key]
                model_path = OUT / spec["model_path"]
                assert sha256(model_path) == spec["model_sha256"]
                bundle = joblib.load(model_path)
                if bundle["projection"] is None:
                    image = native
                else:
                    projection_path = OUT / bundle["projection"]["matrix"]
                    assert sha256(projection_path) == bundle["projection"]["matrix_sha256"]
                    image = np.load(projection_path, mmap_mode="r", allow_pickle=False)
                raw = np.asarray(image[feature_rows][:, bundle["mask"]], dtype=np.float64)
                z = bundle["scaler"].transform(raw)
                if family == "PLS":
                    offset = bundle["offset"]
                    pred_z_unique = np.asarray(bundle["model"].predict(z - offset["x"])).reshape(-1) + offset["y"]
                else:
                    pred_z_unique = common.predict_model(bundle["model"], z)
                pred_unique = common.target_inverse(pred_z_unique, bundle["target_parameters"])
                prediction = pred_unique[inverse]
                macro = temporal.token_balanced_metrics(y_eval, prediction, tokens_eval)
                micro = temporal.transaction_metrics(y_eval, prediction)
                selected = freeze["selected_image_learner_by_encoder"][collection][encoder]
                final_rows.append({
                    "collection": collection, "encoder": encoder, "family": family,
                    "encoder_learner_selected_pre2025": family == selected["family"],
                    "selected_config": json.dumps(spec["selected_config"], sort_keys=True),
                    "pre2025_validation_transaction_rmse": spec["pre2025_validation_transaction_rmse"],
                    **{f"evaluation_equal_token_{name}": value for name, value in macro.items() if name != "mse"},
                    **{f"evaluation_transaction_{name}": value for name, value in micro.items()},
                })
                np.savez_compressed(OUT / "predictions" / f"{collection}_{encoder}_{family}.npz",
                                    source_rows=source_eval, tokens=tokens_eval, y_lrp=y_eval,
                                    predictions_lrp=prediction)
            del native
            gc.collect()
    write_csv(OUT / "fixed_out_of_time_image_evaluation_112_combinations.csv", final_rows)

    selected_rows = [row for row in final_rows if row["encoder_learner_selected_pre2025"]]
    write_csv(OUT / "selected_image_learner_by_encoder.csv", selected_rows)
    summary = [
        "# Image-only robust-asinh benchmark", "",
        "All learner and hyperparameter choices use pooled 2024 Q2-Q4 transaction RMSE. The 2025+ results are descriptive and were loaded only after selection freeze.", "",
    ]
    for collection in COLLECTIONS:
        summary.extend([f"## {collection}", "",
                        "| Encoder | Selected learner | Pre-2025 validation RMSE | 2025+ transaction RMSE | 2025+ equal-token RMSE |",
                        "|---|---|---:|---:|---:|"])
        for encoder in ENCODERS:
            row = next(item for item in selected_rows if item["collection"] == collection and item["encoder"] == encoder)
            summary.append(f"| {encoder} | {row['family']} | {row['pre2025_validation_transaction_rmse']:.9f} | {row['evaluation_transaction_rmse']:.9f} | {row['evaluation_equal_token_rmse']:.9f} |")
        summary.append("")
    (OUT / "analysis_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    for item in manifest["inputs"]:
        assert sha256(ROOT / item["path"]) == item["sha256"]
    write_json(OUT / "completion.json", {
        "completed_utc": now(), "passed": True, "expected_fits": EXPECTED_FITS,
        "actual_fits": completed, "candidate_combinations": len(final_rows),
        "evaluation_loaded_after_freeze": True, "elapsed_seconds": time.time() - started,
        "manuscript_modified": False, "prior_results_overwritten": False,
    })
    progress("complete", completed)


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        main()
