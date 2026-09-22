"""TF-IDF metadata plus image-prediction late fusion, with pre-2025 selection.

Each collection uses its development-selected TF-IDF metadata learner. Each
image encoder/family uses the configuration selected in the completed image-only
benchmark. Missing validation predictions are recomputed with those frozen
configurations; final out-of-time predictions are reused only after selection
and weight freeze. No prior result directory is modified.
"""

from __future__ import annotations

import csv
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
EARLY_CODE = ROOT / "results" / "early_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260921" / "code"
sys.path.insert(0, str(EARLY_CODE))
import run_early_fusion_tfidf_experiment as early


OUT = ROOT / "results" / "late_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260922_v2"
METADATA_RESULT = ROOT / "results" / "metadata_asinh_transaction_2022_2024_eight_regressors_20260920_v1"
IMAGE_RESULT = early.IMAGE_RESULT
COLLECTIONS = early.COLLECTIONS
ENCODERS = early.ENCODERS
FAMILIES = early.FAMILIES
QUARTERS = early.TUNING_QUARTERS
WEIGHTS = tuple(index / 10 for index in range(11))
EXPECTED_METADATA_FITS = len(COLLECTIONS) * len(QUARTERS)
EXPECTED_IMAGE_FITS = len(COLLECTIONS) * len(ENCODERS) * len(FAMILIES) * len(QUARTERS)
EXPECTED_FITS = EXPECTED_METADATA_FITS + EXPECTED_IMAGE_FITS
EXPECTED_CANDIDATES = len(COLLECTIONS) * len(ENCODERS) * len(FAMILIES)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_prediction(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float64), allow_pickle=False)
    temporary.replace(path)
    return early.sha256(path)


def report(stage: str, completed: int, **extra) -> None:
    early.write_json(
        OUT / "progress.json",
        {
            "updated_utc": early.now(),
            "stage": stage,
            "completed_fits": completed,
            "expected_fits": EXPECTED_FITS,
            "percent": 100 * min(completed, EXPECTED_FITS) / EXPECTED_FITS,
            **extra,
        },
    )


def selections() -> tuple[dict, dict]:
    metadata_rows = read_csv(METADATA_RESULT / "pre2025_model_selection_32_combinations.csv")
    image_rows = read_csv(IMAGE_RESULT / "pre2025_image_model_selection_112_combinations.csv")
    metadata = {}
    image = {}
    for collection in COLLECTIONS:
        candidates = [
            row for row in metadata_rows
            if row["collection"] == collection and row["encoding"] == "TF-IDF"
        ]
        assert len(candidates) == len(FAMILIES)
        metadata[collection] = min(
            candidates,
            key=lambda row: (
                float(row["pre2025_validation_transaction_rmse"]),
                FAMILIES.index(row["family"]),
            ),
        )
        for encoder in ENCODERS:
            for family in FAMILIES:
                matches = [
                    row for row in image_rows
                    if row["collection"] == collection
                    and row["encoder"] == encoder
                    and row["family"] == family
                ]
                assert len(matches) == 1
                image[collection, encoder, family] = matches[0]
    assert len(image) == EXPECTED_CANDIDATES
    return metadata, image


def quarter_indices(data: dict, origin: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    train = np.flatnonzero(
        (data["times"] >= early.TRAIN_START) & (data["times"] < origin)
    )
    valid = np.flatnonzero((data["times"] >= origin) & (data["times"] < end))
    assert len(train) and len(valid)
    assert data["times"][train].max() < data["times"][valid].min()
    return train, valid


def validate_saved(row: dict, path: Path, config: dict, rowids: np.ndarray) -> np.ndarray:
    assert row["config"] == config
    assert row["validation_row_sha256"] == early.array_sha(rowids)
    assert row["prediction_sha256"] == early.sha256(path)
    values = np.load(path, allow_pickle=False)
    assert len(values) == len(rowids) and np.isfinite(values).all()
    return values


def metadata_fold(
    data: dict, collection: str, origin: str, end: str, selected: dict, completed: int
) -> tuple[np.ndarray, int]:
    train, valid = quarter_indices(data, origin, end)
    family = selected["family"]
    config = json.loads(selected["selected_config"])
    path = OUT / "checkpoints" / "metadata" / collection / origin / f"{family}.json"
    pred_path = path.with_suffix(".npy")
    if path.exists():
        row = early.read_json(path)
        return validate_saved(row, pred_path, config, data["rowids"][valid]), completed + 1
    transaction_data = dict(data)
    transaction_data["X"] = data["X"][data["index"]]
    prep = early.common.prepare_tfidf(transaction_data, train, valid)
    parameters = early.common.fit_target_transform(data["y"][train])
    transformed = early.common.target_forward(data["y"], parameters)
    started = time.perf_counter()
    model, status = early.temporal.fit_model(prep, family, config, transformed[train])
    if not status["valid"]:
        raise RuntimeError(f"Selected TF-IDF model invalid: {collection} {origin} {status}")
    features = prep["raw_valid"] if family in early.common.TREE_FAMILIES else prep["zvalid"]
    predicted = early.common.target_inverse(
        early.common.predict_model(model, features), parameters
    )
    assert len(predicted) == len(valid) and np.isfinite(predicted).all()
    mse = float(np.mean((data["y"][valid] - predicted) ** 2))
    old_index = early.common.GRIDS[family].index(config)
    old_path = (
        METADATA_RESULT / "checkpoints" / collection / "TF_IDF"
        / origin / family / f"{old_index}.json"
    )
    old = early.read_json(old_path)
    assert old["valid"] and old["config"] == config
    assert old["training_row_sha256"] == early.array_sha(data["rowids"][train])
    assert old["validation_row_sha256"] == early.array_sha(data["rowids"][valid])
    if not np.isclose(mse, old["transaction_mse"], rtol=1e-7, atol=1e-9):
        raise RuntimeError(f"TF-IDF checkpoint reproduction failed: {collection} {origin}")
    prediction_sha = save_prediction(pred_path, predicted)
    early.write_json(
        path,
        {
            "collection": collection, "origin": origin, "end": end,
            "family": family, "config": config, "status": status,
            "train_n": len(train), "validation_n": len(valid),
            "training_row_sha256": early.array_sha(data["rowids"][train]),
            "validation_row_sha256": early.array_sha(data["rowids"][valid]),
            "validation_target_sha256": early.array_sha(data["y"][valid]),
            "prediction_sha256": prediction_sha, "transaction_mse": mse,
            "reference_checkpoint": str(old_path.relative_to(ROOT)),
            "reference_transaction_mse": old["transaction_mse"],
            "fit_seconds": time.perf_counter() - started,
        },
    )
    completed += 1
    report("pre2025_validation_predictions", completed, collection=collection,
           component="metadata", quarter=origin, family=family)
    del model, prep, transaction_data
    return predicted, completed


def image_fold(
    data: dict, collection: str, encoder: str, origin: str, end: str,
    family: str, selected: dict, prepared: dict, transformed: np.ndarray,
    train: np.ndarray, valid: np.ndarray, completed: int
) -> tuple[np.ndarray, int]:
    config = json.loads(selected["selected_config"])
    path = OUT / "checkpoints" / "image" / collection / encoder / origin / f"{family}.json"
    pred_path = path.with_suffix(".npy")
    if path.exists():
        row = early.read_json(path)
        return validate_saved(row, pred_path, config, data["rowids"][valid]), completed + 1
    started = time.perf_counter()
    model, _, pred_z, status = early.image_run.fit_model(
        prepared, family, config, transformed[train]
    )
    if not status["valid"]:
        raise RuntimeError(
            f"Selected image model invalid: {collection} {encoder} {origin} {family} {status}"
        )
    parameters = early.common.fit_target_transform(data["y"][train])
    predicted = early.common.target_inverse(pred_z, parameters)
    assert len(predicted) == len(valid) and np.isfinite(predicted).all()
    mse = float(np.mean((data["y"][valid] - predicted) ** 2))
    old_index = early.common.GRIDS[family].index(config)
    old_path = (
        IMAGE_RESULT / "checkpoints" / collection / encoder
        / origin / family / f"{old_index}.json"
    )
    old = early.read_json(old_path)
    assert old["valid"] and old["config"] == config
    assert old["training_row_sha256"] == early.array_sha(data["rowids"][train])
    assert old["validation_row_sha256"] == early.array_sha(data["rowids"][valid])
    if not np.isclose(mse, old["transaction_mse"], rtol=1e-7, atol=1e-9):
        raise RuntimeError(
            f"Image checkpoint reproduction failed: {collection} {encoder} {origin} {family}"
        )
    prediction_sha = save_prediction(pred_path, predicted)
    early.write_json(
        path,
        {
            "collection": collection, "encoder": encoder, "origin": origin,
            "end": end, "family": family, "config": config, "status": status,
            "train_n": len(train), "validation_n": len(valid),
            "training_row_sha256": early.array_sha(data["rowids"][train]),
            "validation_row_sha256": early.array_sha(data["rowids"][valid]),
            "validation_target_sha256": early.array_sha(data["y"][valid]),
            "prediction_sha256": prediction_sha, "transaction_mse": mse,
            "reference_checkpoint": str(old_path.relative_to(ROOT)),
            "reference_transaction_mse": old["transaction_mse"],
            "fit_seconds": time.perf_counter() - started,
        },
    )
    completed += 1
    report("pre2025_validation_predictions", completed, collection=collection,
           component="image", encoder=encoder, quarter=origin, family=family)
    del model
    return predicted, completed


def main() -> None:
    if (OUT / "completion.json").exists():
        raise FileExistsError(f"Completed output exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(exist_ok=True)
    started = time.time()
    metadata_choice, image_choice = selections()
    registry = early.read_json(early.REGISTRY_PATH)
    metadata_lookup = early.metadata_lookup()
    specification = {
        "stage": "TF-IDF metadata plus image prediction late fusion",
        "collections": COLLECTIONS, "encoders": ENCODERS, "image_families": FAMILIES,
        "metadata_selection": {
            collection: {
                "family": metadata_choice[collection]["family"],
                "config": json.loads(metadata_choice[collection]["selected_config"]),
            }
            for collection in COLLECTIONS
        },
        "image_configuration": "one selected configuration per encoder and family from completed image-only pre-2025 tuning",
        "weight_grid": WEIGHTS, "prediction_formula": "(1 - image_weight) * metadata_prediction + image_weight * image_prediction",
        "target": "fold-fitted robust-asinh LRP; combine and score inverse-transformed original-LRP predictions",
        "training_window": "2022-01-01 through 2024-12-31",
        "selection": "pooled 2024 Q2-Q4 transaction-weighted RMSE on original LRP",
        "evaluation": "fixed retrospective 2025-01-01 through 2026-04-13; prior evaluation predictions loaded after selection freeze",
        "expected_recomputed_fits": EXPECTED_FITS,
        "prior_results_overwritten": False,
    }
    spec_path = OUT / "experiment_specification.json"
    if not spec_path.exists():
        early.write_json(spec_path, specification)
        source_paths = [
            Path(__file__), Path(early.__file__), Path(early.common.__file__),
            Path(early.temporal.__file__), Path(early.image_run.__file__),
            early.REGISTRY_PATH,
            METADATA_RESULT / "completion.json",
            METADATA_RESULT / "pre2025_model_selection_32_combinations.csv",
            IMAGE_RESULT / "completion.json",
            IMAGE_RESULT / "pre2025_image_model_selection_112_combinations.csv",
            early.cr.base.OLD / "metadata_normalized.jsonl",
        ]
        early.write_json(
            OUT / "run_manifest.json",
            {
                "start_utc": early.now(), "expected_fits": EXPECTED_FITS,
                "python": sys.version, "machine": platform.platform(),
                "specification_sha256": early.sha256(spec_path),
                "inputs": [
                    {"path": str(path.relative_to(ROOT)), "sha256": early.sha256(path)}
                    for path in source_paths
                ],
            },
        )
    else:
        assert early.read_json(spec_path) == json.loads(json.dumps(specification))
    manifest = early.read_json(OUT / "run_manifest.json")
    assert early.sha256(spec_path) == manifest["specification_sha256"]

    completed = 0
    metadata_predictions = {}
    validation_targets = {}
    validation_rows = {}
    for collection in COLLECTIONS:
        data = early.load_encoder_data(collection, "DINOv2", registry, metadata_lookup)
        for origin, end in QUARTERS:
            _, valid = quarter_indices(data, origin, end)
            prediction, completed = metadata_fold(
                data, collection, origin, end, metadata_choice[collection], completed
            )
            metadata_predictions[collection, origin] = prediction
            validation_targets[collection, origin] = data["y"][valid].copy()
            validation_rows[collection, origin] = data["rowids"][valid].copy()
        del data
        gc.collect()

    candidate_rows = []
    weight_rows = []
    for collection in COLLECTIONS:
        for encoder in ENCODERS:
            data = early.load_encoder_data(collection, encoder, registry, metadata_lookup)
            image_predictions = {}
            for origin, end in QUARTERS:
                train, valid = quarter_indices(data, origin, end)
                assert np.array_equal(data["rowids"][valid], validation_rows[collection, origin])
                assert np.array_equal(data["y"][valid], validation_targets[collection, origin])
                parameters = early.common.fit_target_transform(data["y"][train])
                transformed = early.common.target_forward(data["y"], parameters)
                image, _ = early.projected_image(data, train, origin)
                working = dict(data)
                working["image"] = image
                prepared = early.image_run.prepare_image(
                    working, image, train, valid, transformed
                )
                for family in FAMILIES:
                    predicted, completed = image_fold(
                        data, collection, encoder, origin, end, family,
                        image_choice[collection, encoder, family], prepared,
                        transformed, train, valid, completed
                    )
                    image_predictions[origin, family] = predicted
                del prepared, image, working
                gc.collect()
            for family in FAMILIES:
                y = np.concatenate([validation_targets[collection, origin] for origin, _ in QUARTERS])
                metadata_pred = np.concatenate([metadata_predictions[collection, origin] for origin, _ in QUARTERS])
                image_pred = np.concatenate([image_predictions[origin, family] for origin, _ in QUARTERS])
                assert len(y) == len(metadata_pred) == len(image_pred)
                scores = []
                for weight in WEIGHTS:
                    prediction = (1 - weight) * metadata_pred + weight * image_pred
                    mse = float(np.mean((y - prediction) ** 2))
                    scores.append((mse, weight))
                    weight_rows.append(
                        {"collection": collection, "encoder": encoder,
                         "image_family": family, "image_weight": weight,
                         "validation_n": len(y), "pre2025_validation_transaction_mse": mse,
                         "pre2025_validation_transaction_rmse": float(np.sqrt(mse))}
                    )
                best_mse, best_weight = min(scores, key=lambda item: (item[0], item[1]))
                candidate_rows.append(
                    {"collection": collection, "encoder": encoder,
                     "metadata_family": metadata_choice[collection]["family"],
                     "image_family": family,
                     "metadata_config": metadata_choice[collection]["selected_config"],
                     "image_config": image_choice[collection, encoder, family]["selected_config"],
                     "image_weight": best_weight, "validation_n": len(y),
                     "pre2025_validation_transaction_rmse": float(np.sqrt(best_mse)),
                     "pre2025_validation_transaction_mse": best_mse}
                )
            del data, image_predictions
            gc.collect()
    assert completed == EXPECTED_FITS and len(candidate_rows) == EXPECTED_CANDIDATES
    write_csv(OUT / "pre2025_weight_grid_1232_combinations.csv", weight_rows)
    write_csv(OUT / "pre2025_late_model_selection_112_combinations.csv", candidate_rows)
    chosen = {}
    for collection in COLLECTIONS:
        chosen[collection] = min(
            (row for row in candidate_rows if row["collection"] == collection),
            key=lambda row: (
                row["pre2025_validation_transaction_mse"],
                ENCODERS.index(row["encoder"]),
                FAMILIES.index(row["image_family"]),
                WEIGHTS.index(row["image_weight"]),
            ),
        )
    freeze = {
        "created_utc": early.now(),
        "evaluation_predictions_loaded_before_freeze": False,
        "selected": chosen,
        "weight_grid": WEIGHTS,
        "metadata_prediction_source": str(METADATA_RESULT.relative_to(ROOT)),
        "image_prediction_source": str(IMAGE_RESULT.relative_to(ROOT)),
    }
    early.write_json(OUT / "selection_freeze.json", freeze)
    report("selection_frozen_loading_evaluation", completed)

    final_rows = []
    for collection in COLLECTIONS:
        meta_family = metadata_choice[collection]["family"]
        meta_path = METADATA_RESULT / "predictions" / f"{collection}_TF_IDF_{meta_family}.npz"
        with np.load(meta_path, allow_pickle=False) as source:
            reference = {key: source[key].copy() for key in source.files}
        assert len(reference["y_lrp"]) == {"BAYC": 5120, "MAYC": 13019}[collection]
        for row in (item for item in candidate_rows if item["collection"] == collection):
            image_path = IMAGE_RESULT / "predictions" / f"{collection}_{row['encoder']}_{row['image_family']}.npz"
            with np.load(image_path, allow_pickle=False) as source:
                for key in ("source_rows", "tokens", "y_lrp"):
                    assert np.array_equal(reference[key], source[key])
                image_prediction = source["predictions_lrp"]
            weight = row["image_weight"]
            prediction = (1 - weight) * reference["predictions_lrp"] + weight * image_prediction
            assert np.isfinite(prediction).all()
            micro = early.temporal.transaction_metrics(reference["y_lrp"], prediction)
            macro = early.temporal.token_balanced_metrics(
                reference["y_lrp"], prediction, reference["tokens"].astype(str)
            )
            selected = (
                row["encoder"] == chosen[collection]["encoder"]
                and row["image_family"] == chosen[collection]["image_family"]
            )
            final_rows.append(
                {**row, "overall_selected_pre2025": selected,
                 **{f"evaluation_transaction_{key}": value for key, value in micro.items()},
                 **{f"evaluation_equal_token_{key}": value for key, value in macro.items() if key != "mse"}}
            )
            np.savez_compressed(
                OUT / "predictions" / f"{collection}_{row['encoder']}_{row['image_family']}.npz",
                source_rows=reference["source_rows"], tokens=reference["tokens"],
                y_lrp=reference["y_lrp"], predictions_lrp=prediction,
            )
    write_csv(OUT / "fixed_out_of_time_late_evaluation_112_combinations.csv", final_rows)
    selected_rows = [row for row in final_rows if row["overall_selected_pre2025"]]
    assert len(selected_rows) == len(COLLECTIONS)
    write_csv(OUT / "development_selected_late_model.csv", selected_rows)
    lines = [
        "# TF-IDF metadata plus image late-fusion benchmark", "",
        "All component models, image weights, and global candidates are selected using pre-2025 validation only. The 2025+ scores are descriptive.", "",
        "| Collection | Metadata | Image | Image weight | Pre-2025 RMSE | 2025+ transaction RMSE | 2025+ equal-token RMSE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['collection']} | TF-IDF + {row['metadata_family']} | "
            f"{row['encoder']} + {row['image_family']} | {row['image_weight']:.1f} | "
            f"{row['pre2025_validation_transaction_rmse']:.9f} | "
            f"{row['evaluation_transaction_rmse']:.9f} | "
            f"{row['evaluation_equal_token_rmse']:.9f} |"
        )
    (OUT / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for item in manifest["inputs"]:
        assert early.sha256(ROOT / item["path"]) == item["sha256"]
    early.write_json(
        OUT / "completion.json",
        {
            "completed_utc": early.now(), "passed": True,
            "expected_fits": EXPECTED_FITS, "actual_fits": completed,
            "candidate_combinations": len(final_rows),
            "weight_combinations": len(weight_rows),
            "evaluation_predictions_loaded_after_freeze": True,
            "elapsed_seconds": time.time() - started,
            "prior_results_overwritten": False, "manuscript_modified": False,
        },
    )
    report("complete", completed)


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        main()
