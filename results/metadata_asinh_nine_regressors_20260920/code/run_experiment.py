"""BAYC metadata experiment with fold-fitted robust-asinh targets.

Two metadata representations (one-hot and NFT-document TF-IDF) are evaluated
with nine regression families under the existing nested chronological design.
No manuscript or prior result directory is modified.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_OUT = Path(__file__).resolve().parents[1]
ROOT = SCRIPT_OUT.parents[1]
# The initial preflight stopped before fitting because its expected-quarter
# assertion omitted the three final-tuning quarters. Preserve that aborted
# directory and write the corrected execution to a separate v2 directory.
OUT = ROOT / "results" / "metadata_asinh_nine_regressors_20260920_v2"
BOOST = ROOT / "results" / "metadata_boosting_benchmark_20260918"
CANONICAL = ROOT / "results" / "seven_encoder_full_rerun" / "code"
sys.path.insert(0, str(BOOST / "packages"))
sys.path.insert(0, str(BOOST / "code"))
sys.path.insert(0, str(CANONICAL))

import joblib
import lightgbm
import numpy as np
import psutil
import sklearn
import xgboost
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import canonical_runner as cr
import run_boosting as boost


SEED = 20260908
COLLECTION = "BAYC"
ENCODINGS = ["one-hot", "TF-IDF"]
FAMILIES = [
    "OLS", "Ridge", "ElasticNet", "PLS", "LinearSVR",
    "HistGradientBoosting", "XGBoost", "LightGBM", "RandomForest",
]
TREE_FAMILIES = {"HistGradientBoosting", "XGBoost", "LightGBM", "RandomForest"}
RF_GRID = [
    {"max_features": max_features, "min_samples_leaf": min_leaf}
    for max_features, min_leaf in itertools.product((0.5, 1.0), (1, 5, 20))
]
GRIDS = dict(
    cr.GRIDS,
    OLS=[{}],
    XGBoost=boost.GRID,
    LightGBM=boost.GRID,
    RandomForest=RF_GRID,
)
TFIDF_SETTINGS = dict(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)
EXPECTED_FITS = 1372  # 2 encodings * (10*65 quarter + 3*9 outer + 9 final refits)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def write_json_replace(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv_once(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def log_event(**value) -> None:
    record = dict(utc=now(), **value)
    with (OUT / "execution_journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def update_progress(stage: str, **extra) -> None:
    completed = len(list((OUT / "checkpoints").rglob("*.json"))) if (OUT / "checkpoints").exists() else 0
    write_json_replace(
        OUT / "progress.json",
        dict(
            updated_utc=now(), stage=stage, completed_fits=completed,
            expected_fits=EXPECTED_FITS, percent=100 * completed / EXPECTED_FITS,
            **extra,
        ),
    )


def array_sha(values: np.ndarray) -> str:
    return cr.array_sha(np.asarray(values))


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def fit_target_transform(y_train: np.ndarray) -> dict:
    center = float(np.median(y_train))
    scale = float(stats.median_abs_deviation(y_train, scale="normal"))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Robust-asinh scale is non-positive")
    return {"center": center, "normalized_MAD": scale}


def target_forward(y: np.ndarray, parameters: dict) -> np.ndarray:
    return np.arcsinh((np.asarray(y) - parameters["center"]) / parameters["normalized_MAD"])


def target_inverse(z: np.ndarray, parameters: dict) -> np.ndarray:
    result = parameters["center"] + parameters["normalized_MAD"] * np.sinh(np.asarray(z))
    return np.asarray(result, dtype=np.float64).reshape(-1)


def prepare_onehot(data: dict, train: np.ndarray, valid: np.ndarray) -> dict:
    encoder = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False, dtype=np.float64)
    raw_train = encoder.fit_transform(data["X"][train])
    constant_filter = VarianceThreshold(threshold=0).fit(raw_train)
    raw_train = constant_filter.transform(raw_train)
    raw_valid = constant_filter.transform(encoder.transform(data["X"][valid])) if len(valid) else np.empty((0, raw_train.shape[1]))
    encoding = Pipeline([("onehot", encoder), ("constant_filter", constant_filter)])
    scaler = StandardScaler().fit(raw_train)
    ztrain = scaler.transform(raw_train)
    zvalid = scaler.transform(raw_valid) if len(valid) else raw_valid
    sample = ztrain[np.linspace(0, len(train) - 1, min(1024, len(train)), dtype=int)]
    rank = int(np.linalg.matrix_rank(sample - sample.mean(axis=0)))
    assert np.isfinite(raw_train).all() and np.isfinite(raw_valid).all()
    return dict(encoding=encoding, scaler=scaler, raw_train=raw_train, raw_valid=raw_valid,
                ztrain=ztrain, zvalid=zvalid, rank=rank, dimension=raw_train.shape[1],
                representation="one-hot")


def prepare_tfidf(data: dict, train: np.ndarray, valid: np.ndarray) -> dict:
    # One distinct NFT is one IDF document. Repeated sales do not redefine rarity.
    unique_tokens, first = np.unique(data["tokens"][train], return_index=True)
    unique_X = data["X"][train[first]]
    onehot = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=True, dtype=np.float64)
    unique_binary = onehot.fit_transform(unique_X)
    tfidf = TfidfTransformer(**TFIDF_SETTINGS).fit(unique_binary)
    before_filter = dense(tfidf.transform(onehot.transform(data["X"][train])))
    constant_filter = VarianceThreshold(threshold=0).fit(before_filter)
    raw_train = constant_filter.transform(before_filter)
    encoding = Pipeline([("onehot", onehot), ("tfidf", tfidf), ("constant_filter", constant_filter)])
    raw_valid = dense(encoding.transform(data["X"][valid])) if len(valid) else np.empty((0, raw_train.shape[1]))
    scaler = StandardScaler().fit(raw_train)
    ztrain = scaler.transform(raw_train)
    zvalid = scaler.transform(raw_valid) if len(valid) else raw_valid
    df = np.asarray(unique_binary.sum(axis=0)).reshape(-1)
    expected_idf = np.log((1 + len(unique_tokens)) / (1 + df)) + 1
    np.testing.assert_allclose(tfidf.idf_, expected_idf, rtol=1e-14, atol=1e-14)
    sample = ztrain[np.linspace(0, len(train) - 1, min(1024, len(train)), dtype=int)]
    rank = int(np.linalg.matrix_rank(sample - sample.mean(axis=0)))
    assert np.isfinite(raw_train).all() and np.isfinite(raw_valid).all()
    return dict(encoding=encoding, scaler=scaler, raw_train=raw_train, raw_valid=raw_valid,
                ztrain=ztrain, zvalid=zvalid, rank=rank, dimension=raw_train.shape[1],
                representation="TF-IDF", unique_training_tokens=len(unique_tokens), idf=expected_idf)


def prepare(data: dict, train: np.ndarray, valid: np.ndarray, encoding: str) -> dict:
    return prepare_onehot(data, train, valid) if encoding == "one-hot" else prepare_tfidf(data, train, valid)


def build_model(family: str, config: dict, phase: str):
    if family == "OLS":
        return LinearRegression(fit_intercept=True, n_jobs=1)
    if family in {"XGBoost", "LightGBM"}:
        return boost.build(family, config)
    if family == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300, criterion="squared_error", bootstrap=True,
            max_features=config["max_features"], min_samples_leaf=config["min_samples_leaf"],
            max_depth=None, random_state=SEED, n_jobs=3,
        )
    return cr.phase_model(family, config, "development" if phase != "final_refit" else "final")


def predict_model(model, X: np.ndarray) -> np.ndarray:
    if isinstance(model, lightgbm.LGBMRegressor):
        return np.asarray(model.booster_.predict(X, num_threads=3), dtype=np.float64).reshape(-1)
    return np.asarray(model.predict(X), dtype=np.float64).reshape(-1)


def config_index(family: str, config: dict) -> int:
    return GRIDS[family].index(config)


def fit_one(data: dict, train: np.ndarray, valid: np.ndarray, prep: dict, encoding: str,
            family: str, config: dict, phase: str):
    stem = OUT / "checkpoints" / encoding.replace("-", "_") / phase / family / str(config_index(family, config))
    info_path = stem.with_suffix(".json")
    pred_path = stem.with_suffix(".npz")
    if info_path.exists():
        info = read_json(info_path)
        assert info["config"] == config
        assert info["training_row_sha256"] == array_sha(data["rowids"][train])
        assert info["validation_row_sha256"] == array_sha(data["rowids"][valid])
        if info["valid"] and len(valid):
            assert sha256(pred_path) == info["prediction_sha256"]
            saved = np.load(pred_path)
            return saved["predictions_lrp"], saved["predictions_asinh"], info
        return None, None, info

    cr.check_training(data, train, valid, str(data["times"][valid].min())[:10] if len(valid) else "2025-01-01")
    if psutil.virtual_memory().available < 1.2 * 2**30:
        raise MemoryError("Less than 1.2 GiB available; checkpoints retained for resume")
    target_parameters = fit_target_transform(data["y"][train])
    y_train_asinh = target_forward(data["y"][train], target_parameters)
    y_valid_asinh = target_forward(data["y"][valid], target_parameters) if len(valid) else np.empty(0)
    Xtrain = prep["raw_train"] if family in TREE_FAMILIES else prep["ztrain"]
    Xvalid = prep["raw_valid"] if family in TREE_FAMILIES else prep["zvalid"]
    start = time.perf_counter()
    model = None
    pred_asinh = pred_lrp = None
    try:
        if family == "PLS" and config["components"] > prep["rank"]:
            raise ValueError("requested PLS components exceed deterministic rank lower bound")
        model = build_model(family, config, phase)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(Xtrain, y_train_asinh)
            pred_asinh = predict_model(model, Xvalid) if len(valid) else np.empty(0)
        pred_lrp = target_inverse(pred_asinh, target_parameters)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        finite = bool(np.isfinite(pred_asinh).all() and np.isfinite(pred_lrp).all())
        if family == "PLS":
            converged &= len(model.n_iter_) == config["components"]
            finite &= bool(np.isfinite(model.coef_).all())
        info = dict(
            valid=bool(converged and finite), converged=bool(converged), finite=bool(finite),
            warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
            n_iter=np.asarray(getattr(model, "n_iter_", [])).tolist(),
            fitted_rank=int(model.rank_) if family == "OLS" else None,
        )
    except MemoryError:
        raise
    except Exception as error:
        info = dict(valid=False, error=repr(error))

    info.update(
        collection=COLLECTION, encoding=encoding, family=family, config=config, phase=phase,
        train_n=len(train), n=len(valid), training_max=str(data["times"][train].max()),
        training_row_sha256=array_sha(data["rowids"][train]),
        validation_row_sha256=array_sha(data["rowids"][valid]),
        target_parameters=target_parameters, dimension=prep["dimension"], rank_lower_bound=prep["rank"],
        seconds=time.perf_counter() - start,
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    if info["valid"]:
        if len(valid):
            np.savez_compressed(
                pred_path, predictions_lrp=pred_lrp, predictions_asinh=pred_asinh,
                y_lrp=data["y"][valid], y_asinh=y_valid_asinh,
                tokens=data["tokens"][valid], source_rows=data["rowids"][valid],
            )
            info.update(
                prediction_sha256=sha256(pred_path),
                sse_lrp=float(np.sum((data["y"][valid] - pred_lrp) ** 2)),
                sse_asinh=float(np.sum((y_valid_asinh - pred_asinh) ** 2)),
            )
        if phase == "final_refit":
            model_path = stem.with_suffix(".joblib")
            joblib.dump(
                dict(
                    encoding=prep["encoding"], scaler=prep["scaler"], model=model,
                    family=family, metadata_encoding=encoding, columns=data["columns"],
                    target_parameters=target_parameters,
                ),
                model_path,
                compress=3,
            )
            info.update(model_path=str(model_path.relative_to(OUT)), model_sha256=sha256(model_path))
    write_json_once(info_path, info)
    log_event(stage="fit_completed", encoding=encoding, family=family, phase=phase,
              config_index=config_index(family, config), valid=info["valid"], seconds=round(info["seconds"], 3))
    update_progress("fitting", encoding=encoding, family=family, phase=phase)
    del model
    return (pred_lrp, pred_asinh, info) if info["valid"] else (None, None, info)


def masks(data: dict, origin: str, end: str | None = None):
    train = np.flatnonzero(data["times"] < origin)
    valid = np.flatnonzero((data["times"] >= origin) & (data["times"] < end)) if end else np.asarray([], dtype=int)
    cr.check_training(data, train, valid, origin)
    return train, valid


def inner_scores(quarter_infos: dict, schedule: list, encoding: str, family: str) -> list[dict]:
    results = []
    for index, config in enumerate(GRIDS[family]):
        infos = [quarter_infos[encoding, origin, end, family][index] for origin, end in schedule]
        valid = all(info["valid"] for info in infos)
        results.append(dict(
            config=config, valid=valid,
            mse_lrp=sum(info["sse_lrp"] for info in infos) / sum(info["n"] for info in infos) if valid else None,
            mse_asinh=sum(info["sse_asinh"] for info in infos) / sum(info["n"] for info in infos) if valid else None,
        ))
    return results


def ordered_choices(family: str, scores: list[dict]) -> list[dict]:
    eligible = [row for row in scores if row["valid"]]
    if not eligible:
        return []
    if family in {"Ridge", "ElasticNet", "PLS", "LinearSVR", "HistGradientBoosting"}:
        canonical_scores = [dict(config=row["config"], valid=True, mse=row["mse_lrp"]) for row in eligible]
        canonical_order = cr.meta.ordered_valid(family, canonical_scores)
        by_config = {json.dumps(row["config"], sort_keys=True): row for row in eligible}
        return [by_config[json.dumps(row["config"], sort_keys=True)] for row in canonical_order]
    if family in {"XGBoost", "LightGBM"}:
        return sorted(
            eligible,
            key=lambda row: (
                round(row["mse_lrp"] / 1e-12) * 1e-12,
                row["config"]["max_leaf_nodes"], -row["config"]["l2_regularization"],
                row["config"]["learning_rate"],
            ),
        )
    if family == "RandomForest":
        return sorted(
            eligible,
            key=lambda row: (
                round(row["mse_lrp"] / 1e-12) * 1e-12,
                -row["config"]["min_samples_leaf"], row["config"]["max_features"],
            ),
        )
    return eligible  # OLS has exactly one configuration.


def metric_record(y, pred, tokens) -> dict:
    metrics = cr.base.metrics(np.asarray(y), np.asarray(pred), np.asarray(tokens))
    return {
        "n": int(metrics["n"]), "rmse": float(metrics["rmse"]), "mae": float(metrics["mae"]),
        "r2": float(metrics["r2"]), "equal_token_rmse": float(metrics["equal_token_rmse"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [
        Path(__file__), Path(cr.__file__), Path(cr.base.__file__), Path(cr.meta.__file__), Path(boost.__file__),
        cr.base.OLD / "metadata_normalized.jsonl",
        cr.base.COHORT / "image_ready_split_manifest_v1.json",
        cr.base.TARGET / "bayc_development_targets.jsonl",
        cr.base.TARGET / "bayc_temporal_test_targets.NOT_FOR_SELECTION.jsonl",
        ROOT / "revision" / "data_audit_20260908" / "analysis_specification_v1.json",
    ]
    specification = {
        "collection": COLLECTION,
        "target": "Fold-fitted robust asinh: z=asinh((LRP-training_median)/training_normalized_MAD); predictions inverse-transformed before selection and scoring",
        "target_parameter_policy": "Each inner, outer, and final fit estimates median and normalized MAD only from its own training rows",
        "encodings": ENCODINGS,
        "families": FAMILIES,
        "grids": GRIDS,
        "random_forest_fixed": {"n_estimators": 300, "criterion": "squared_error", "bootstrap": True, "max_depth": None, "n_jobs": 3, "random_state": SEED},
        "tfidf": {**TFIDF_SETTINGS, "document_unit": "one distinct NFT token in each training portion"},
        "selection_metric": "original-LRP MSE after inverse robust-asinh transformation",
        "validation": "canonical D1-D3 nested expanding-window validation; final tuning on 2024 Q2-Q4",
        "evaluation": "fixed retrospective out-of-time window 2025-01-01 through 2026-04-13, loaded only after freeze",
        "seed": SEED,
        "scope": "exploratory target-transform experiment; no primary reselection and no manuscript modification",
    }
    spec_path = OUT / "experiment_specification.json"
    if not spec_path.exists():
        write_json_once(spec_path, specification)
    else:
        assert read_json(spec_path) == specification
    manifest_path = OUT / "run_manifest.json"
    if not manifest_path.exists():
        write_json_once(
            manifest_path,
            dict(
                start_utc=now(), expected_fits=EXPECTED_FITS, python=sys.version,
                numpy=np.__version__, scipy=stats.__version__ if hasattr(stats, "__version__") else None,
                sklearn=sklearn.__version__, xgboost=xgboost.__version__, lightgbm=lightgbm.__version__,
                machine=platform.platform(), available_ram_gib=psutil.virtual_memory().available / 2**30,
                specification_sha256=sha256(spec_path),
                inputs=[{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_paths],
            ),
        )
    manifest = read_json(manifest_path)
    assert sha256(spec_path) == manifest["specification_sha256"]
    for item in manifest["inputs"]:
        assert sha256(ROOT / item["path"]) == item["sha256"]

    update_progress("loading_development")
    data = cr.load_development(COLLECTION)
    assert len(data["y"]) == 58_635
    schedules = {
        fold["name"]: [(q["train_end_exclusive"][:10], q["validation_end_exclusive"][:10]) for q in fold["inner_quarter_schedule"]]
        for fold in data["split"]["folds"]
    }
    schedules["final"] = cr.FINAL_Q
    unique_quarters = sorted(set(itertools.chain.from_iterable(schedules.values())))
    assert len(unique_quarters) == 10

    quarter_infos: dict = {}
    for encoding in ENCODINGS:
        for origin, end in unique_quarters:
            train, valid = masks(data, origin, end)
            prep = prepare(data, train, valid, encoding)
            for family in FAMILIES:
                infos = []
                for config in GRIDS[family]:
                    _, _, info = fit_one(data, train, valid, prep, encoding, family, config, "quarter_" + origin)
                    infos.append(info)
                quarter_infos[encoding, origin, end, family] = infos
            del prep
            gc.collect()

    outer_rows = []
    pooled: dict[tuple[str, str], dict[str, list]] = {
        (encoding, family): {"y": [], "pred": [], "tokens": [], "y_asinh": [], "pred_asinh": []}
        for encoding in ENCODINGS for family in FAMILIES
    }
    baseline_parts = {encoding: {"y": [], "pred_mean": [], "pred_zero": [], "tokens": []} for encoding in ENCODINGS}
    for encoding in ENCODINGS:
        for fold in data["split"]["folds"]:
            train = np.asarray([data["rowmap"][int(row)] for row in fold["train_source_rows"]])
            valid = np.asarray([data["rowmap"][int(row)] for row in fold["validation_source_rows"]])
            origin = {"D1": "2023-01-01", "D2": "2023-07-01", "D3": "2024-01-01"}[fold["name"]]
            cr.check_training(data, train, valid, origin)
            prep = prepare(data, train, valid, encoding)
            target_parameters = fit_target_transform(data["y"][train])
            transformed_train_mean = float(target_forward(data["y"][train], target_parameters).mean())
            baseline_prediction = np.full(len(valid), target_inverse(np.asarray([transformed_train_mean]), target_parameters)[0])
            bp = baseline_parts[encoding]
            bp["y"].append(data["y"][valid]); bp["pred_mean"].append(baseline_prediction)
            bp["pred_zero"].append(np.zeros(len(valid))); bp["tokens"].append(data["tokens"][valid])
            for family in FAMILIES:
                scores = inner_scores(quarter_infos, schedules[fold["name"]], encoding, family)
                choices = ordered_choices(family, scores)
                attempts = []
                pred_lrp = pred_asinh = None
                for choice in choices:
                    candidate_lrp, candidate_asinh, info = fit_one(
                        data, train, valid, prep, encoding, family, choice["config"], "outer_" + fold["name"]
                    )
                    attempts.append(info)
                    if info["valid"]:
                        pred_lrp, pred_asinh = candidate_lrp, candidate_asinh
                        break
                if pred_lrp is None:
                    raise RuntimeError(f"No valid outer model: {encoding} {family} {fold['name']}")
                parameters = attempts[-1]["target_parameters"]
                y_asinh = target_forward(data["y"][valid], parameters)
                pool = pooled[encoding, family]
                pool["y"].append(data["y"][valid]); pool["pred"].append(pred_lrp)
                pool["tokens"].append(data["tokens"][valid]); pool["y_asinh"].append(y_asinh)
                pool["pred_asinh"].append(pred_asinh)
                outer_rows.append(
                    dict(
                        collection=COLLECTION, encoding=encoding, family=family, fold=fold["name"],
                        selected_config=attempts[-1]["config"], target_parameters=parameters,
                        selection_scores=scores, attempts=attempts,
                        metrics_lrp=metric_record(data["y"][valid], pred_lrp, data["tokens"][valid]),
                        metrics_asinh=metric_record(y_asinh, pred_asinh, data["tokens"][valid]),
                    )
                )
            del prep
            gc.collect()

    development_results = []
    for encoding in ENCODINGS:
        for family in FAMILIES:
            pool = pooled[encoding, family]
            y_all = np.concatenate(pool["y"]); pred_all = np.concatenate(pool["pred"])
            tok_all = np.concatenate(pool["tokens"]); ya_all = np.concatenate(pool["y_asinh"])
            pa_all = np.concatenate(pool["pred_asinh"])
            metrics = metric_record(y_all, pred_all, tok_all)
            transformed_metrics = metric_record(ya_all, pa_all, tok_all)
            development_results.append(
                dict(
                    collection=COLLECTION, encoding=encoding, family=family,
                    **{f"LRP_{key}": value for key, value in metrics.items()},
                    **{f"asinh_{key}": value for key, value in transformed_metrics.items()},
                )
            )
    baseline_results = []
    for encoding in ENCODINGS:
        bp = baseline_parts[encoding]
        y_all = np.concatenate(bp["y"]); tok_all = np.concatenate(bp["tokens"])
        for name, key in (("training_asinh_mean", "pred_mean"), ("LRP_zero", "pred_zero")):
            baseline_results.append(dict(sample="development_outer_D1_D3", encoding=encoding, baseline=name,
                                         **metric_record(y_all, np.concatenate(bp[key]), tok_all)))

    freeze = {"created_utc": now(), "evaluation_labels_loaded_before_freeze": False, "models": {}}
    train_all, empty = masks(data, "2025-01-01")
    assert len(train_all) == 58_635 and len(empty) == 0
    for encoding in ENCODINGS:
        prep = prepare(data, train_all, empty, encoding)
        for family in FAMILIES:
            scores = inner_scores(quarter_infos, schedules["final"], encoding, family)
            choices = ordered_choices(family, scores)
            attempts = []
            final_info = None
            for choice in choices:
                _, _, info = fit_one(data, train_all, empty, prep, encoding, family, choice["config"], "final_refit")
                attempts.append(info)
                if info["valid"]:
                    final_info = info
                    break
            if final_info is None:
                raise RuntimeError(f"No valid final model: {encoding} {family}")
            key = f"{encoding}__{family}"
            freeze["models"][key] = {
                "encoding": encoding, "family": family, "selected_config": final_info["config"],
                "selection_scores": scores, "attempts": attempts,
                "model_path": final_info["model_path"], "model_sha256": final_info["model_sha256"],
                "target_parameters": final_info["target_parameters"],
            }
        del prep
        gc.collect()

    freeze_path = OUT / "selection_freeze.json"
    if not freeze_path.exists():
        write_json_once(freeze_path, freeze)
    else:
        prior = read_json(freeze_path)
        assert prior["models"] == freeze["models"]
        freeze = prior
    write_json_once(OUT / "outer_fold_audit.json", outer_rows) if not (OUT / "outer_fold_audit.json").exists() else None
    write_csv_once(OUT / "development_results_18_combinations.csv", development_results) if not (OUT / "development_results_18_combinations.csv").exists() else None
    write_csv_once(OUT / "baseline_results.csv", baseline_results) if not (OUT / "baseline_results.csv").exists() else None
    update_progress("selection_frozen_loading_evaluation")

    # Fixed retrospective out-of-time evaluation is loaded only after the freeze.
    eval_rows = cr.lines(cr.base.TARGET / "bayc_temporal_test_targets.NOT_FOR_SELECTION.jsonl")
    assert len(eval_rows) == 5_120
    assert all("2025-01-01" <= row["time"] < "2026-04-14" for row in eval_rows)
    metadata = {(row["collection"], int(row["token_id"])): row for row in cr.lines(cr.base.OLD / "metadata_normalized.jsonl")}
    y_eval = np.asarray([row["y_log_relative_price"] for row in eval_rows], dtype=np.float64)
    tok_eval = np.asarray([row["token_id"] for row in eval_rows])
    row_eval = np.asarray([row["source_row"] for row in eval_rows])
    final_results = []
    (OUT / "final_predictions").mkdir(parents=True, exist_ok=True)
    for encoding in ENCODINGS:
        for family in FAMILIES:
            spec = freeze["models"][f"{encoding}__{family}"]
            model_path = OUT / spec["model_path"]
            assert sha256(model_path) == spec["model_sha256"]
            bundle = joblib.load(model_path)
            X = np.asarray(
                [[metadata[COLLECTION, int(row["token_id"])][column] for column in bundle["columns"]] for row in eval_rows],
                dtype=object,
            )
            raw = dense(bundle["encoding"].transform(X))
            features = raw if family in TREE_FAMILIES else bundle["scaler"].transform(raw)
            pred_asinh = predict_model(bundle["model"], features)
            pred_lrp = target_inverse(pred_asinh, bundle["target_parameters"])
            y_asinh = target_forward(y_eval, bundle["target_parameters"])
            assert np.isfinite(pred_lrp).all() and np.isfinite(pred_asinh).all()
            metrics = metric_record(y_eval, pred_lrp, tok_eval)
            transformed_metrics = metric_record(y_asinh, pred_asinh, tok_eval)
            np.savez_compressed(
                OUT / "final_predictions" / f"{encoding.replace('-', '_')}_{family}.npz",
                source_rows=row_eval, tokens=tok_eval, y_lrp=y_eval, predictions_lrp=pred_lrp,
                y_asinh=y_asinh, predictions_asinh=pred_asinh,
            )
            final_results.append(
                dict(
                    collection=COLLECTION, encoding=encoding, family=family,
                    selected_config=json.dumps(spec["selected_config"], sort_keys=True),
                    target_center=bundle["target_parameters"]["center"],
                    target_normalized_MAD=bundle["target_parameters"]["normalized_MAD"],
                    **{f"LRP_{key}": value for key, value in metrics.items()},
                    **{f"asinh_{key}": value for key, value in transformed_metrics.items()},
                )
            )
            log_event(stage="final_evaluation", encoding=encoding, family=family,
                      LRP_RMSE=metrics["rmse"], asinh_RMSE=transformed_metrics["rmse"])

    # Evaluation baselines are fixed from all development targets.
    final_target = fit_target_transform(data["y"])
    train_asinh_mean = float(target_forward(data["y"], final_target).mean())
    mean_lrp = target_inverse(np.asarray([train_asinh_mean]), final_target)[0]
    for encoding in ENCODINGS:
        baseline_results.append(dict(sample="fixed_retrospective_evaluation", encoding=encoding,
                                     baseline="training_asinh_mean", **metric_record(y_eval, np.full(len(y_eval), mean_lrp), tok_eval)))
        baseline_results.append(dict(sample="fixed_retrospective_evaluation", encoding=encoding,
                                     baseline="LRP_zero", **metric_record(y_eval, np.zeros(len(y_eval)), tok_eval)))

    write_csv_once(OUT / "final_evaluation_18_combinations.csv", final_results)
    # Rewrite the baseline file once to append evaluation rows; the initial file
    # is an intermediate checkpoint in a new output directory, not a prior result.
    with (OUT / "baseline_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(baseline_results[0]))
        writer.writeheader(); writer.writerows(baseline_results)

    for item in manifest["inputs"]:
        assert sha256(ROOT / item["path"]) == item["sha256"]
    ranked_dev = sorted(development_results, key=lambda row: row["LRP_rmse"])
    ranked_final = sorted(final_results, key=lambda row: row["LRP_rmse"])
    summary = [
        "# BAYC robust-asinh metadata experiment",
        "",
        "Two metadata representations and nine regression families were refitted from scratch. Model selection used only development data; final scores use the fixed retrospective out-of-time window.",
        "",
        "## Development D1-D3 pooled ranking (original-LRP scale after inverse transform)",
        "",
        "| Rank | Encoding | Family | RMSE | MAE | R2 | Equal-token RMSE |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked_dev, 1):
        summary.append(f'| {rank} | {row["encoding"]} | {row["family"]} | {row["LRP_rmse"]:.9f} | {row["LRP_mae"]:.9f} | {row["LRP_r2"]:.9f} | {row["LRP_equal_token_rmse"]:.9f} |')
    summary.extend([
        "", "## Fixed retrospective out-of-time evaluation", "",
        "| Rank | Encoding | Family | RMSE | MAE | R2 | Equal-token RMSE |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(ranked_final, 1):
        summary.append(f'| {rank} | {row["encoding"]} | {row["family"]} | {row["LRP_rmse"]:.9f} | {row["LRP_mae"]:.9f} | {row["LRP_r2"]:.9f} | {row["LRP_equal_token_rmse"]:.9f} |')
    summary.extend([
        "",
        "Primary comparison metrics are reported on the original LRP scale after inverse transformation. Asinh-scale metrics are retained in the CSV files. This exploratory run does not change the certified primary model or any manuscript.",
    ])
    (OUT / "analysis_summary.md").write_text("\n".join(summary), encoding="utf-8")
    write_json_once(
        OUT / "completion.json",
        dict(
            completed_utc=now(), passed=True, expected_fits=EXPECTED_FITS,
            actual_fits=len(list((OUT / "checkpoints").rglob("*.json"))),
            combinations=18, evaluation_loaded_after_freeze=True,
            best_development={key: ranked_dev[0][key] for key in ("encoding", "family", "LRP_rmse", "LRP_mae", "LRP_r2")},
            best_evaluation_descriptive={key: ranked_final[0][key] for key in ("encoding", "family", "LRP_rmse", "LRP_mae", "LRP_r2")},
            manuscript_modified=False, prior_results_overwritten=False,
        ),
    )
    update_progress("complete")


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        main()
