"""Seven-encoder TF-IDF early-fusion benchmark under the robust-asinh protocol.

Each distinct training NFT is one document for fold-local metadata TF-IDF.
Metadata and image blocks are standardized separately within each training
fold and concatenated before fitting all eight regressors.
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
import psutil
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "early_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260921_v1"
IMAGE_RESULT = ROOT / "results" / "image_asinh_transaction_2022_2024_eight_regressors_20260920_v1"
REGISTRY_PATH = ROOT / "results" / "seven_encoder_pca1024_rerun" / "embedding_registry.json"
COMMON_CODE = ROOT / "results" / "metadata_asinh_nine_regressors_20260920" / "code"
TEMPORAL_CODE = ROOT / "results" / "metadata_temporal_token_balanced_20260920" / "code"
CANONICAL_CODE = ROOT / "results" / "seven_encoder_full_rerun" / "code"
IMAGE_CODE = ROOT / "results" / "image_asinh_transaction_2022_2024_eight_regressors_20260920" / "code"

sys.path.insert(0, str(COMMON_CODE))
import run_experiment as common

sys.path.insert(0, str(TEMPORAL_CODE))
import run_temporal_token_balanced as temporal

sys.path.insert(0, str(CANONICAL_CODE))
import canonical_runner as cr

sys.path.insert(0, str(IMAGE_CODE))
import run_image_experiment as image_run


SEED = 20260908
COLLECTIONS = ("BAYC", "MAYC")
ENCODERS = ("DINOv2", "CLIP", "SigLIP2", "SAM", "SDXL_VAE", "DreamSim", "AIM")
REDUCED = {"SDXL_VAE", "DreamSim"}
FAMILIES = tuple(family for family in common.FAMILIES if family != "RandomForest")
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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
    write_json(
        OUT / "progress.json",
        {
            "updated_utc": now(),
            "stage": stage,
            "completed_fits": completed,
            "expected_fits": EXPECTED_FITS,
            "percent": 100 * min(completed, EXPECTED_FITS) / EXPECTED_FITS,
            **extra,
        },
    )


def metadata_lookup() -> dict[tuple[str, int], dict]:
    return {
        (row["collection"], int(row["token_id"])): row
        for row in cr.lines(cr.base.OLD / "metadata_normalized.jsonl")
    }


def load_encoder_data(collection: str, encoder: str, registry: dict, metadata: dict) -> dict:
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
    columns = cr.base.FEATURES + (["generation"] if collection == "MAYC" else [])
    X = np.asarray(
        [[metadata[collection, int(token)][column] for column in columns] for token in ids],
        dtype=object,
    )
    data = {
        "collection": collection,
        "encoder": encoder,
        "y": base["y"][keep],
        "tokens": tokens,
        "rowids": base["rowids"][keep],
        "times": base["times"][keep],
        "X": X,
        "columns": columns,
        "image": image,
        "feature_tokens": ids,
        "index": np.asarray([mapping[int(token)] for token in tokens], dtype=np.int64),
        "native_dimension": int(image.shape[1]),
        "native_dtype": str(image.dtype),
        "matrix_path": str(matrix_path.relative_to(ROOT)),
        "matrix_sha256": info["matrix_sha256"],
        "_native_sha256": info["matrix_sha256"],
        "_encoder": encoder,
        "_collection": collection,
        "_sample": "original_2022_start",
    }
    assert np.array_equal(ids[data["index"]], tokens)
    assert np.isfinite(data["y"]).all()
    return data


def projected_image(data: dict, train: np.ndarray, origin: str) -> tuple[np.ndarray, dict | None]:
    if data["encoder"] not in REDUCED:
        return data["image"], None
    folder = IMAGE_RESULT / "reduced_embeddings" / data["collection"] / data["encoder"] / origin
    manifest_path = folder / "manifest.json"
    matrix_path = folder / "embeddings_1024.npy"
    pca_path = folder / "pca.joblib"
    manifest = read_json(manifest_path)
    identity = manifest["identity"]
    assert identity["origin"] == origin
    assert identity["native_sha256"] == data["matrix_sha256"]
    assert identity["training_row_sha256"] == array_sha(data["rowids"][train])
    assert sha256(matrix_path) == manifest["embeddings_sha256"]
    assert sha256(pca_path) == manifest["pca_sha256"]
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    assert matrix.shape == (len(data["feature_tokens"]), 1024)
    return matrix, {
        "type": "reused training-fold-only PCA",
        "components": 1024,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "matrix": str(matrix_path.relative_to(ROOT)),
        "matrix_sha256": manifest["embeddings_sha256"],
        "pca": str(pca_path.relative_to(ROOT)),
        "pca_sha256": manifest["pca_sha256"],
    }


def prepare_early(
    data: dict,
    image: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    transformed: np.ndarray,
) -> dict:
    working = dict(data)
    working["image"] = image
    working["y"] = transformed
    prepared = image_run.prepare_image(working, image, train, valid, transformed)
    unique = prepared["unique"]
    counts = prepared["counts"]
    # IDF documents are unique training tokens, not transaction rows.
    onehot = OneHotEncoder(
        handle_unknown="ignore", drop=None, sparse_output=True, dtype=np.float64
    )
    unique_binary = onehot.fit_transform(data["X"][unique])
    tfidf = TfidfTransformer(**common.TFIDF_SETTINGS).fit(unique_binary)
    raw_unique_unfiltered = common.dense(tfidf.transform(unique_binary))
    constant_filter = VarianceThreshold(threshold=0).fit(raw_unique_unfiltered)
    encoding = Pipeline(
        [
            ("onehot", onehot),
            ("tfidf", tfidf),
            ("constant_filter", constant_filter),
            ("dense", FunctionTransformer(common.dense, validate=False)),
        ]
    )
    raw_unique = constant_filter.transform(raw_unique_unfiltered)
    metadata_scaler = StandardScaler().fit(raw_unique, sample_weight=counts)
    z_metadata_unique = metadata_scaler.transform(raw_unique)
    if len(valid):
        raw_valid = encoding.transform(data["X"][data["index"][valid]])
        z_metadata_valid = metadata_scaler.transform(raw_valid)
    else:
        z_metadata_valid = np.empty((0, raw_unique.shape[1]), dtype=np.float64)
    z_unique = np.ascontiguousarray(
        np.concatenate((z_metadata_unique, prepared["z_unique"]), axis=1),
        dtype=np.float64,
    )
    z_valid = np.ascontiguousarray(
        np.concatenate((z_metadata_valid, prepared["z_valid"]), axis=1),
        dtype=np.float64,
    )
    metadata_dimension = int(raw_unique.shape[1])
    image_dimension = int(prepared["dimension"])
    sampled = np.linspace(0, len(unique) - 1, min(256, len(unique)), dtype=int)
    rank = int(
        np.linalg.matrix_rank(z_unique[sampled] - z_unique[sampled].mean(axis=0))
    )
    weighted_mean = np.average(z_unique, axis=0, weights=counts)
    document_frequency = np.asarray(unique_binary.sum(axis=0)).reshape(-1)
    expected_idf = np.log((1 + len(unique)) / (1 + document_frequency)) + 1
    np.testing.assert_allclose(tfidf.idf_, expected_idf, rtol=1e-14, atol=1e-14)
    state = {
        "mode": "early",
        "encoding": encoding,
        "metadata_scaler": metadata_scaler,
        "image_mask": prepared["mask"],
        "image_scaler": prepared["scaler"],
        "metadata_columns": data["columns"],
    }
    assert z_unique.shape[1] == metadata_dimension + image_dimension
    assert np.isfinite(z_unique).all() and np.isfinite(z_valid).all()
    return {
        "unique": unique,
        "inverse": prepared["inverse"],
        "counts": counts,
        "means": prepared["means"],
        "z_unique": z_unique,
        "z_valid": z_valid,
        "rank": rank,
        "weighted_mean": weighted_mean,
        "dimension": int(z_unique.shape[1]),
        "metadata_dimension": metadata_dimension,
        "image_dimension": image_dimension,
        "idf_document_count": int(len(unique)),
        "state": state,
    }


def fit_checkpoint(
    data: dict,
    train: np.ndarray,
    valid: np.ndarray,
    origin: str,
    end: str,
    prep: dict,
    projection: dict | None,
    parameters: dict,
    transformed: np.ndarray,
    family: str,
    config: dict,
    completed: int,
) -> tuple[dict, int]:
    number = common.GRIDS[family].index(config)
    stem = OUT / "checkpoints" / data["collection"] / data["encoder"] / origin / family / str(number)
    info_path = stem.with_suffix(".json")
    prediction_path = stem.with_suffix(".npy")
    if info_path.exists():
        row = read_json(info_path)
        assert row["config"] == config
        assert row["training_row_sha256"] == array_sha(data["rowids"][train])
        assert row["validation_row_sha256"] == array_sha(data["rowids"][valid])
        if row["valid"]:
            assert sha256(prediction_path) == row["prediction_sha256"]
        return row, completed + 1
    if psutil.virtual_memory().available < 1.5 * 2**30:
        raise MemoryError("Less than 1.5 GiB available; checkpoints retained for resume")
    model, offset, pred_z, status = image_run.fit_model(prep, family, config, transformed[train])
    if status["valid"]:
        pred_lrp = common.target_inverse(pred_z, parameters)
        micro = temporal.transaction_metrics(data["y"][valid], pred_lrp)
        macro = temporal.token_balanced_metrics(
            data["y"][valid], pred_lrp, data["tokens"][valid].astype(str)
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("xb") as stream:
            np.save(stream, pred_lrp, allow_pickle=False)
        prediction_sha256 = sha256(prediction_path)
    else:
        micro = macro = None
        prediction_sha256 = None
    row = {
        "collection": data["collection"],
        "encoder": data["encoder"],
        "family": family,
        "config": config,
        "origin": origin,
        "end": end,
        "valid": bool(status["valid"]),
        "status": status,
        "train_n": int(len(train)),
        "validation_n": int(len(valid)),
        "training_min": str(data["times"][train].min()),
        "training_max": str(data["times"][train].max()),
        "training_row_sha256": array_sha(data["rowids"][train]),
        "validation_row_sha256": array_sha(data["rowids"][valid]),
        "target_parameters": parameters,
        "metadata_dimension": prep["metadata_dimension"],
        "idf_document_count": prep["idf_document_count"],
        "image_dimension": prep["image_dimension"],
        "early_dimension": prep["dimension"],
        "native_image_dimension": data["native_dimension"],
        "projection": projection,
        "prediction_file": str(prediction_path.relative_to(OUT)) if status["valid"] else None,
        "prediction_sha256": prediction_sha256,
        "transaction_mse": None if micro is None else micro["rmse"] ** 2,
        "transaction_rmse": None if micro is None else micro["rmse"],
        "equal_token_mse": None if macro is None else macro["mse"],
        "equal_token_rmse": None if macro is None else macro["rmse"],
    }
    write_json(info_path, row)
    completed += 1
    progress(
        "pre2025_tuning",
        completed,
        collection=data["collection"],
        encoder=data["encoder"],
        quarter=origin,
        family=family,
        config_index=number,
        available_ram_gib=psutil.virtual_memory().available / 2**30,
    )
    del model
    return row, completed


def pooled_scores(rows: list[dict], collection: str, encoder: str, family: str) -> list[dict]:
    scores = []
    for config in common.GRIDS[family]:
        parts = [
            row
            for row in rows
            if row["collection"] == collection
            and row["encoder"] == encoder
            and row["family"] == family
            and row["config"] == config
        ]
        valid = len(parts) == len(TUNING_QUARTERS) and all(row["valid"] for row in parts)
        mse = (
            sum(row["transaction_mse"] * row["validation_n"] for row in parts)
            / sum(row["validation_n"] for row in parts)
            if valid
            else None
        )
        scores.append({"config": config, "valid": valid, "mse_lrp": mse})
    return scores


def main() -> None:
    if (OUT / "completion.json").exists():
        raise FileExistsError(f"Completed output exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("models", "predictions", "checkpoints"):
        (OUT / folder).mkdir(exist_ok=True)
    started = time.time()
    registry = read_json(REGISTRY_PATH)
    metadata = metadata_lookup()
    specification = {
        "stage": "early-fusion regression",
        "collections": COLLECTIONS,
        "encoders": ENCODERS,
        "metadata_encoding": "training-fold-fitted category-aware one-hot followed by NFT-document TF-IDF",
        "tfidf_settings": {
            **common.TFIDF_SETTINGS,
            "document_unit": "one distinct NFT token in each training portion",
            "constant_filter": "zero-variance training-token columns; same support as repeated transactions",
        },
        "fusion": "separately standardized metadata and image blocks concatenated within each training fold",
        "families": FAMILIES,
        "grids": {family: common.GRIDS[family] for family in FAMILIES},
        "training_window": "2022-01-01 through 2024-12-31",
        "training_unit": "uniform transaction weight, computed with repeated-feature sufficient statistics",
        "target": "fold-fitted robust-asinh LRP; selection and reporting on inverse-transformed original LRP",
        "selection": "pooled 2024 Q2-Q4 transaction-weighted RMSE",
        "evaluation": "fixed retrospective 2025-01-01 through 2026-04-13; loaded only after freeze",
        "dimensions": "native image representation except reused fold-local PCA1024 for SDXL_VAE/DreamSim",
        "random_forest_excluded": True,
        "validation_predictions_persisted": True,
        "seed": SEED,
    }
    spec_path = OUT / "experiment_specification.json"
    source_paths = [
        Path(__file__),
        REGISTRY_PATH,
        Path(common.__file__),
        Path(temporal.__file__),
        Path(cr.__file__),
        Path(image_run.__file__),
        cr.base.OLD / "metadata_normalized.jsonl",
    ]
    for collection in COLLECTIONS:
        source_paths.extend(
            [
                cr.base.TARGET / f"{collection.lower()}_development_targets.jsonl",
                cr.base.TARGET / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl",
            ]
        )
        for encoder in ENCODERS:
            info = registry[encoder][collection]
            source_paths.extend([ROOT / info["matrix"], ROOT / info["manifest"]])
            if encoder in REDUCED:
                for origin, _ in TUNING_QUARTERS:
                    folder = IMAGE_RESULT / "reduced_embeddings" / collection / encoder / origin
                    source_paths.extend(
                        [folder / "manifest.json", folder / "embeddings_1024.npy", folder / "pca.joblib"]
                    )
                folder = IMAGE_RESULT / "reduced_embeddings" / collection / encoder / "2025-01-01"
                source_paths.extend(
                    [folder / "manifest.json", folder / "embeddings_1024.npy", folder / "pca.joblib"]
                )
    if not spec_path.exists():
        write_json(spec_path, specification)
        write_json(
            OUT / "run_manifest.json",
            {
                "start_utc": now(),
                "expected_fits": EXPECTED_FITS,
                "python": sys.version,
                "machine": platform.platform(),
                "available_ram_gib": psutil.virtual_memory().available / 2**30,
                "specification_sha256": sha256(spec_path),
                "inputs": [
                    {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                    for path in dict.fromkeys(source_paths)
                ],
            },
        )
    else:
        assert read_json(spec_path) == json.loads(json.dumps(specification))
    manifest = read_json(OUT / "run_manifest.json")
    assert sha256(spec_path) == manifest["specification_sha256"]

    cohort_rows = []
    for collection in COLLECTIONS:
        for encoder in ENCODERS:
            data = load_encoder_data(collection, encoder, registry, metadata)
            cohort_rows.append(
                {
                    "collection": collection,
                    "encoder": encoder,
                    "training_transactions": len(data["y"]),
                    "training_tokens": len(np.unique(data["tokens"])),
                    "registry_tokens": len(data["feature_tokens"]),
                    "native_image_dimension": data["native_dimension"],
                    "training_min": str(data["times"].min()),
                    "training_max": str(data["times"].max()),
                    "token_alignment_verified": bool(
                        np.array_equal(data["feature_tokens"][data["index"]], data["tokens"])
                    ),
                }
            )
            del data
            gc.collect()
    write_csv(OUT / "cohort_audit.csv", cohort_rows)

    completed = 0
    tuning_rows = []
    for collection in COLLECTIONS:
        for encoder in ENCODERS:
            data = load_encoder_data(collection, encoder, registry, metadata)
            for origin, end in TUNING_QUARTERS:
                train = np.flatnonzero((data["times"] >= TRAIN_START) & (data["times"] < origin))
                valid = np.flatnonzero((data["times"] >= origin) & (data["times"] < end))
                assert len(train) and len(valid) and data["times"][train].max() < data["times"][valid].min()
                parameters = common.fit_target_transform(data["y"][train])
                transformed = common.target_forward(data["y"], parameters)
                image, projection = projected_image(data, train, origin)
                prep = prepare_early(data, image, train, valid, transformed)
                for family in FAMILIES:
                    for config in common.GRIDS[family]:
                        row, completed = fit_checkpoint(
                            data,
                            train,
                            valid,
                            origin,
                            end,
                            prep,
                            projection,
                            parameters,
                            transformed,
                            family,
                            config,
                            completed,
                        )
                        tuning_rows.append(row)
                del prep, image
                gc.collect()
            del data
            gc.collect()
    write_csv(
        OUT / "pre2025_tuning_all_configs.csv",
        [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"config", "status", "target_parameters", "projection"}
                },
                "config": json.dumps(row["config"], sort_keys=True),
                "status": json.dumps(row["status"], sort_keys=True),
                "target_parameters": json.dumps(row["target_parameters"], sort_keys=True),
                "projection": json.dumps(row["projection"], sort_keys=True),
            }
            for row in tuning_rows
        ],
    )

    freeze = {"created_utc": now(), "evaluation_labels_loaded_before_freeze": False, "models": {}}
    selection_rows, training_rows = [], []
    for collection in COLLECTIONS:
        for encoder in ENCODERS:
            data = load_encoder_data(collection, encoder, registry, metadata)
            train = np.arange(len(data["y"]), dtype=int)
            empty = np.asarray([], dtype=int)
            parameters = common.fit_target_transform(data["y"])
            transformed = common.target_forward(data["y"], parameters)
            image, projection = projected_image(data, train, "2025-01-01")
            prep = prepare_early(data, image, train, empty, transformed)
            for family in FAMILIES:
                scores = pooled_scores(tuning_rows, collection, encoder, family)
                choices = common.ordered_choices(family, scores)
                attempts = []
                model = offset = selected = None
                for choice in choices:
                    candidate, candidate_offset, _, status = image_run.fit_model(
                        prep, family, choice["config"], transformed
                    )
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
                    pred_z = (
                        np.asarray(model.predict(prep["z_unique"] - offset["x"])).reshape(-1)
                        + offset["y"]
                    )
                else:
                    pred_z = common.predict_model(model, prep["z_unique"])
                pred_unique = common.target_inverse(pred_z, parameters)
                lookup = np.full(len(data["feature_tokens"]), np.nan)
                lookup[prep["unique"]] = pred_unique
                train_pred = lookup[data["index"]]
                assert np.isfinite(train_pred).all()
                micro = temporal.transaction_metrics(data["y"], train_pred)
                macro = temporal.token_balanced_metrics(
                    data["y"], train_pred, data["tokens"].astype(str)
                )
                key = f"{collection}__{encoder}__{family}"
                model_path = OUT / "models" / f"{collection}_{encoder}_{family}.joblib"
                joblib.dump(
                    {
                        "model": model,
                        "offset": offset,
                        "state": prep["state"],
                        "family": family,
                        "encoder": encoder,
                        "collection": collection,
                        "columns": data["columns"],
                        "target_parameters": parameters,
                        "projection": projection,
                        "training_tokens": np.unique(data["tokens"]),
                    },
                    model_path,
                    compress=3,
                )
                freeze["models"][key] = {
                    "collection": collection,
                    "encoder": encoder,
                    "family": family,
                    "selected_config": selected["config"],
                    "pre2025_validation_transaction_rmse": float(np.sqrt(selected["mse_lrp"])),
                    "model_path": str(model_path.relative_to(OUT)),
                    "model_sha256": sha256(model_path),
                    "projection": projection,
                    "target_parameters": parameters,
                    "attempts": attempts,
                }
                selection_rows.append(
                    {
                        "collection": collection,
                        "encoder": encoder,
                        "family": family,
                        "metadata_dimension": prep["metadata_dimension"],
                        "idf_document_count": prep["idf_document_count"],
                        "image_dimension": prep["image_dimension"],
                        "early_dimension": prep["dimension"],
                        "selected_config": json.dumps(selected["config"], sort_keys=True),
                        "pre2025_validation_transaction_rmse": float(np.sqrt(selected["mse_lrp"])),
                    }
                )
                training_rows.append(
                    {
                        "collection": collection,
                        "encoder": encoder,
                        "family": family,
                        **{
                            f"training_equal_token_{name}": value
                            for name, value in macro.items()
                            if name != "mse"
                        },
                        **{f"training_transaction_{name}": value for name, value in micro.items()},
                    }
                )
                progress("final_refit", completed, collection=collection, encoder=encoder, family=family)
                del model
            del prep, image, data
            gc.collect()
    for collection in COLLECTIONS:
        for encoder in ENCODERS:
            candidates = [
                row
                for row in selection_rows
                if row["collection"] == collection and row["encoder"] == encoder
            ]
            freeze.setdefault("selected_early_learner_by_encoder", {}).setdefault(collection, {})[
                encoder
            ] = min(
                candidates,
                key=lambda row: (
                    row["pre2025_validation_transaction_rmse"],
                    FAMILIES.index(row["family"]),
                ),
            )
        all_candidates = [row for row in selection_rows if row["collection"] == collection]
        freeze.setdefault("development_selected", {})[collection] = min(
            all_candidates,
            key=lambda row: (
                row["pre2025_validation_transaction_rmse"],
                ENCODERS.index(row["encoder"]),
                FAMILIES.index(row["family"]),
            ),
        )
    write_csv(OUT / "pre2025_early_model_selection_112_combinations.csv", selection_rows)
    write_csv(OUT / "training_performance_112_combinations.csv", training_rows)
    write_json(OUT / "selection_freeze.json", freeze)
    progress("selection_frozen_loading_evaluation", completed)

    final_rows = []
    for collection in COLLECTIONS:
        eval_rows = cr.lines(
            cr.base.TARGET / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl"
        )
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
                    projection_path = ROOT / bundle["projection"]["matrix"]
                    assert sha256(projection_path) == bundle["projection"]["matrix_sha256"]
                    image = np.load(projection_path, mmap_mode="r", allow_pickle=False)
                X_unique = np.asarray(
                    [
                        [metadata[collection, int(token)][column] for column in bundle["columns"]]
                        for token in unique_tokens
                    ],
                    dtype=object,
                )
                z = cr.vision.transform(
                    bundle["state"],
                    X_unique,
                    np.asarray(image[feature_rows], dtype=np.float64),
                )
                if family == "PLS":
                    offset = bundle["offset"]
                    pred_z_unique = (
                        np.asarray(bundle["model"].predict(z - offset["x"])).reshape(-1)
                        + offset["y"]
                    )
                else:
                    pred_z_unique = common.predict_model(bundle["model"], z)
                pred_unique = common.target_inverse(
                    pred_z_unique, bundle["target_parameters"]
                )
                prediction = pred_unique[inverse]
                assert np.isfinite(prediction).all()
                macro = temporal.token_balanced_metrics(y_eval, prediction, tokens_eval)
                micro = temporal.transaction_metrics(y_eval, prediction)
                per_encoder = freeze["selected_early_learner_by_encoder"][collection][encoder]
                overall = freeze["development_selected"][collection]
                final_rows.append(
                    {
                        "collection": collection,
                        "encoder": encoder,
                        "family": family,
                        "encoder_learner_selected_pre2025": family == per_encoder["family"],
                        "overall_selected_pre2025": encoder == overall["encoder"]
                        and family == overall["family"],
                        "selected_config": json.dumps(spec["selected_config"], sort_keys=True),
                        "pre2025_validation_transaction_rmse": spec[
                            "pre2025_validation_transaction_rmse"
                        ],
                        **{
                            f"evaluation_equal_token_{name}": value
                            for name, value in macro.items()
                            if name != "mse"
                        },
                        **{f"evaluation_transaction_{name}": value for name, value in micro.items()},
                    }
                )
                np.savez_compressed(
                    OUT / "predictions" / f"{collection}_{encoder}_{family}.npz",
                    source_rows=source_eval,
                    tokens=tokens_eval,
                    y_lrp=y_eval,
                    predictions_lrp=prediction,
                )
            del native
            gc.collect()
    write_csv(OUT / "fixed_out_of_time_early_evaluation_112_combinations.csv", final_rows)
    selected_rows = [row for row in final_rows if row["encoder_learner_selected_pre2025"]]
    overall_rows = [row for row in final_rows if row["overall_selected_pre2025"]]
    write_csv(OUT / "selected_early_learner_by_encoder.csv", selected_rows)
    write_csv(OUT / "development_selected_early_model.csv", overall_rows)

    summary = [
        "# Early-fusion robust-asinh benchmark",
        "",
        "Unique-training-token TF-IDF metadata and image blocks are standardized separately within each training fold and concatenated. All choices use pooled 2024 Q2-Q4 transaction RMSE; 2025+ results are descriptive.",
        "",
    ]
    for collection in COLLECTIONS:
        summary.extend(
            [
                f"## {collection}",
                "",
                "| Encoder | Selected learner | Pre-2025 validation RMSE | 2025+ transaction RMSE | 2025+ equal-token RMSE |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for encoder in ENCODERS:
            row = next(
                item
                for item in selected_rows
                if item["collection"] == collection and item["encoder"] == encoder
            )
            summary.append(
                f"| {encoder} | {row['family']} | {float(row['pre2025_validation_transaction_rmse']):.9f} | "
                f"{float(row['evaluation_transaction_rmse']):.9f} | "
                f"{float(row['evaluation_equal_token_rmse']):.9f} |"
            )
        summary.append("")
        chosen = next(row for row in overall_rows if row["collection"] == collection)
        summary.append(
            f"Overall development-selected early fusion: **{chosen['encoder']} + {chosen['family']}**."
        )
        summary.append("")
    (OUT / "analysis_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    for item in manifest["inputs"]:
        assert sha256(ROOT / item["path"]) == item["sha256"]
    write_json(
        OUT / "completion.json",
        {
            "completed_utc": now(),
            "passed": True,
            "expected_fits": EXPECTED_FITS,
            "actual_fit_attempts": completed,
            "candidate_combinations": len(final_rows),
            "evaluation_loaded_after_freeze": True,
            "elapsed_seconds": time.time() - started,
            "manuscript_modified": False,
            "prior_results_overwritten": False,
        },
    )
    progress("complete", completed)


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        main()
