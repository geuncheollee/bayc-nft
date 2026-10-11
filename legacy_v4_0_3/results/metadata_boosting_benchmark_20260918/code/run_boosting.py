"""Additive nested-temporal metadata boosting benchmarks; no primary reselection."""
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

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
MAIN = ROOT / "results/seven_encoder_pca1024_rerun"
OLS = ROOT / "results/metadata_ols_benchmark_20260918"
sys.path.insert(0, str(OUT / "packages"))
sys.path.insert(0, str(ROOT / "results/seven_encoder_full_rerun/code"))
import joblib
import numpy as np
import sklearn
import xgboost
import lightgbm
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
import canonical_runner as cr

SEED = 20260908
FAMILIES = ["XGBoost", "LightGBM"]
GRID = [dict(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=reg)
        for lr, leaves, reg in itertools.product((0.05, 0.1), (15, 31), (1, 10))]
FIXED = dict(n_estimators=300, subsample=1.0, colsample_bytree=1.0,
             reg_alpha=0.0, max_bin=255, random_state=SEED, n_jobs=3)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csvsave(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def log(obj):
    obj = dict(utc=now(), **obj)
    with (OUT / "execution_journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(obj) + "\n")
    print(json.dumps(obj), flush=True)


def build(family, config):
    common = dict(FIXED, learning_rate=config["learning_rate"], reg_lambda=config["l2_regularization"])
    if family == "XGBoost":
        return xgboost.XGBRegressor(**common, objective="reg:squarederror", tree_method="hist",
                                  device="cpu", grow_policy="lossguide", max_depth=0,
                                  max_leaves=config["max_leaf_nodes"], min_child_weight=20,
                                  verbosity=0)
    return lightgbm.LGBMRegressor(**common, objective="regression", boosting_type="gbdt",
                                 num_leaves=config["max_leaf_nodes"], max_depth=-1,
                                 min_child_samples=20, subsample_freq=0, deterministic=True,
                                 force_col_wise=True, verbosity=-1)


def predict(model, X):
    # Booster API avoids sklearn's feature-name warning for anonymous dense input.
    return np.asarray(model.booster_.predict(X, num_threads=3) if isinstance(model, lightgbm.LGBMRegressor)
                      else model.predict(X), dtype=np.float64).reshape(-1)


def masks(data, origin, end=None):
    train = np.flatnonzero(data["times"] < origin)
    valid = np.flatnonzero((data["times"] >= origin) & (data["times"] < end)) if end else np.asarray([], dtype=int)
    cr.check_training(data, train, valid, origin)
    return train, valid


def prepare(data, train, valid):
    encoding = Pipeline(cr.base.model_pipeline("Ridge", dict(alpha=1)).steps[:2])
    Xtrain = encoding.fit_transform(data["X"][train])
    Xvalid = encoding.transform(data["X"][valid]) if len(valid) else np.empty((0, Xtrain.shape[1]))
    assert np.isfinite(Xtrain).all() and np.isfinite(Xvalid).all()
    assert all(set(cats).issubset(set(data["X"][train, i])) for i, cats in enumerate(encoding.named_steps["onehot"].categories_))
    return encoding, Xtrain, Xvalid


def fit(data, train, valid, encoding, Xtrain, Xvalid, collection, family, config, phase):
    stem = OUT / "checkpoints" / collection / phase / family / str(GRID.index(config))
    path = stem.with_suffix(".json")
    if path.exists():
        info = read(path)
        assert info["config"] == config
        assert info["training_row_sha256"] == cr.array_sha(data["rowids"][train])
        assert info["validation_row_sha256"] == cr.array_sha(data["rowids"][valid])
        if info["valid"]:
            assert sha(stem.with_suffix(".npy")) == info["prediction_sha256"]
            assert sha(stem.with_suffix(".joblib")) == info["model_sha256"]
            return np.load(stem.with_suffix(".npy")), info
        return None, info
    start = time.perf_counter()
    model = build(family, config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(Xtrain, data["y"][train])
        pred = predict(model, Xvalid) if len(valid) else np.empty(0)
        probe = predict(model, Xtrain[:20])
    finite = bool(np.isfinite(pred).all() and np.isfinite(probe).all())
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    info = dict(collection=collection, family=family, config=config, phase=phase,
                valid=bool(finite and converged), n=len(valid), train_n=len(train),
                training_max=str(data["times"][train].max()), dimension=Xtrain.shape[1],
                training_row_sha256=cr.array_sha(data["rowids"][train]),
                validation_row_sha256=cr.array_sha(data["rowids"][valid]),
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                seconds=time.perf_counter() - start)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if info["valid"]:
        np.save(stem.with_suffix(".npy"), pred)
        joblib.dump(dict(encoding=encoding, model=model), stem.with_suffix(".joblib"))
        info.update(model_path=str(stem.with_suffix(".joblib").relative_to(OUT)),
                    model_sha256=sha(stem.with_suffix(".joblib")),
                    prediction_sha256=sha(stem.with_suffix(".npy")),
                    sse=float(np.sum((data["y"][valid] - pred) ** 2)))
    save(path, info)
    log(dict(stage="fit_completed", collection=collection, family=family, phase=phase,
             config_index=GRID.index(config), valid=info["valid"], seconds=round(info["seconds"], 2)))
    return pred if info["valid"] else None, info


def choose(scores):
    eligible = [r for r in scores if r["valid"]]
    assert eligible, "No valid configuration; never silently change cohort"
    best = min(r["mse"] for r in eligible)
    return min([r for r in eligible if r["mse"] <= best + 1e-12],
               key=lambda r: (r["config"]["max_leaf_nodes"], -r["config"]["l2_regularization"], r["config"]["learning_rate"]))


def pooled_scores(quarter_results, schedule, family):
    results = []
    for index, config in enumerate(GRID):
        infos = [quarter_results[(origin, end, family)][index] for origin, end in schedule]
        valid = all(info["valid"] for info in infos)
        results.append(dict(config=config, valid=valid,
                            mse=sum(i["sse"] for i in infos) / sum(i["n"] for i in infos) if valid else None))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(cr.__file__), Path(cr.base.__file__),
               cr.base.OLD / "metadata_normalized.jsonl", cr.base.COHORT / "image_ready_split_manifest_v1.json",
               MAIN / "selected_primary_models.json", OLS / "development_comparison.csv"]
    for collection in cr.COLLECTIONS:
        sources.extend([cr.base.TARGET / f"{collection.lower()}_development_targets.jsonl",
                        cr.base.TARGET / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl",
                        MAIN / "development/original" / collection / "metadata/summary.json"])
    if not (OUT / "run_manifest.json").exists():
        save(OUT / "run_manifest.json", dict(start_utc=now(), seed=SEED, python=sys.version,
             numpy=np.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
             lightgbm=lightgbm.__version__, machine=platform.platform(), git_commit=None,
             fixed_parameters=FIXED, grid=GRID, inputs=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in sources],
             protocol="Original D1-D3 nested expanding validation and final 2024 Q2-Q4 tuning; training-only category-aware one-hot and zero-variance removal; unscaled tree inputs",
             xgboost_parameters="CPU hist, lossguide, max_depth=0, min_child_weight=20, squared error",
             lightgbm_parameters="CPU deterministic, force_col_wise, max_depth=-1, min_child_samples=20, squared error",
             tie_rule="MSE tolerance 1e-12, fewer leaves, higher L2, lower learning rate",
             evaluation="fixed retrospective out-of-time evaluation window; no early stopping or evaluation-based tuning",
             warning="Parameter grids have matched sizes but library regularization and split criteria are not algebraically equivalent; bounded-grid additive benchmarks, not comprehensive optimum claims.",
             references=["https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html", "https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html"]))
    manifest = read(OUT / "run_manifest.json")
    for record in manifest["inputs"]:
        assert sha(ROOT / record["path"]) == record["sha256"]
    development, frozen = [], {}
    with (OLS / "development_comparison.csv").open(encoding="utf-8", newline="") as stream:
        prior = list(csv.DictReader(stream))
    for collection in cr.COLLECTIONS:
        data = cr.load_development(collection)
        assert len(data["y"]) == {"BAYC": 58635, "MAYC": 113823}[collection]
        schedules = {}
        for fold in data["split"]["folds"]:
            schedules[fold["name"]] = [(q["train_end_exclusive"][:10], q["validation_end_exclusive"][:10]) for q in fold["inner_quarter_schedule"]]
        schedules["final"] = cr.FINAL_Q
        unique_quarters = sorted(set(itertools.chain.from_iterable(schedules.values())))
        quarters = {}
        for origin, end in unique_quarters:
            train, valid = masks(data, origin, end)
            assert len(valid)
            encoding, Xtrain, Xvalid = prepare(data, train, valid)
            for family in FAMILIES:
                reports = []
                for config in GRID:
                    _, info = fit(data, train, valid, encoding, Xtrain, Xvalid, collection, family, config, "quarter_" + origin)
                    reports.append(info)
                quarters[origin, end, family] = reports
            del encoding, Xtrain, Xvalid
            gc.collect()
        family_parts = {family: [[], [], []] for family in FAMILIES}
        outer_reports = []
        for fold in data["split"]["folds"]:
            train = np.asarray([data["rowmap"][int(r)] for r in fold["train_source_rows"]])
            valid = np.asarray([data["rowmap"][int(r)] for r in fold["validation_source_rows"]])
            origin = {"D1": "2023-01-01", "D2": "2023-07-01", "D3": "2024-01-01"}[fold["name"]]
            cr.check_training(data, train, valid, origin)
            encoding, Xtrain, Xvalid = prepare(data, train, valid)
            for family in FAMILIES:
                scores = pooled_scores(quarters, schedules[fold["name"]], family)
                config = choose(scores)["config"]
                pred, info = fit(data, train, valid, encoding, Xtrain, Xvalid, collection, family, config, "outer_" + fold["name"])
                assert info["valid"]
                parts = family_parts[family]
                parts[0].append(data["y"][valid]); parts[1].append(pred); parts[2].append(data["tokens"][valid])
                outer_reports.append(dict(collection=collection, family=family, fold=fold["name"],
                                          origin=origin, selection_scores=scores, selected_config=config,
                                          fit=info, metrics=cr.base.metrics(data["y"][valid], pred, data["tokens"][valid])))
                np.savez_compressed(OUT / f"{collection}_{family}_{fold['name']}_predictions.npz",
                                    predictions=pred, y=data["y"][valid], tokens=data["tokens"][valid], source_rows=data["rowids"][valid])
            del encoding, Xtrain, Xvalid
            gc.collect()
        for row in prior:
            if row["collection"] == collection:
                development.append(dict(collection=collection, family=row["family"], n=int(row["n"]),
                     RMSE=float(row["RMSE"]), MAE=float(row["MAE"]), R2=float(row["R2"]),
                     equal_token_RMSE=float(row["equal_token_RMSE"]), existing_selected_Mstar=row["existing_selected_Mstar"] == "True"))
        train, valid = masks(data, "2025-01-01")
        encoding, Xtrain, Xvalid = prepare(data, train, valid)
        frozen[collection] = {}
        for family in FAMILIES:
            y, pred, tok = [np.concatenate(part) for part in family_parts[family]]
            metrics = cr.base.metrics(y, pred, tok)
            development.append(dict(collection=collection, family=family, n=metrics["n"], RMSE=metrics["rmse"],
                  MAE=metrics["mae"], R2=metrics["r2"], equal_token_RMSE=metrics["equal_token_rmse"], existing_selected_Mstar=False))
            scores = pooled_scores(quarters, schedules["final"], family)
            config = choose(scores)["config"]
            _, info = fit(data, train, valid, encoding, Xtrain, Xvalid, collection, family, config, "final_refit")
            assert info["valid"]
            frozen[collection][family] = dict(selected_config=config, selection_scores=scores,
                                            model_path=info["model_path"], model_sha256=info["model_sha256"],
                                            columns=data["columns"], training_max=info["training_max"], development_RMSE=metrics["rmse"])
        if not (OUT / f"{collection}_outer_audit.json").exists():
            save(OUT / f"{collection}_outer_audit.json", outer_reports)
        del data, encoding, Xtrain, Xvalid, family_parts
        gc.collect()
    if not (OUT / "selection_freeze.json").exists():
        save(OUT / "selection_freeze.json", dict(utc=now(), specifications=frozen,
             no_evaluation_labels_loaded_before_freeze=True, primary_selection_unchanged=True))
    assert read(OUT / "selection_freeze.json")["specifications"] == frozen
    if not (OUT / "development_comparison.csv").exists():
        csvsave(OUT / "development_comparison.csv", development)
    final = []
    for collection in cr.COLLECTIONS:
        rows = cr.lines(cr.base.TARGET / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl")
        assert len(rows) == {"BAYC": 5120, "MAYC": 13019}[collection]
        assert all("2025-01-01" <= r["time"] < "2026-04-14" for r in rows)
        metadata = {(r["collection"], int(r["token_id"])): r for r in cr.lines(cr.base.OLD / "metadata_normalized.jsonl")}
        y = np.asarray([r["y_log_relative_price"] for r in rows])
        tok = np.asarray([r["token_id"] for r in rows])
        for family in FAMILIES:
            spec = frozen[collection][family]
            assert sha(OUT / spec["model_path"]) == spec["model_sha256"]
            bundle = joblib.load(OUT / spec["model_path"])
            X = np.asarray([[metadata[collection, int(r["token_id"])][c] for c in spec["columns"]] for r in rows], dtype=object)
            encoded = bundle["encoding"].transform(X)
            assert np.isfinite(encoded).all() and np.isfinite(y).all()
            pred = predict(bundle["model"], encoded)
            assert np.isfinite(pred).all()
            metrics = cr.base.metrics(y, pred, tok)
            np.savez_compressed(OUT / f"{collection}_{family}_final_predictions.npz",
                                predictions=pred, y=y, tokens=tok, source_rows=np.asarray([r["source_row"] for r in rows]))
            final.append(dict(collection=collection, family=family, **metrics))
            log(dict(stage="final_evaluation", collection=collection, family=family, metrics=metrics))
    if not (OUT / "boosting_final_evaluation.csv").exists():
        csvsave(OUT / "boosting_final_evaluation.csv", final)
    for record in manifest["inputs"]:
        assert sha(ROOT / record["path"]) == record["sha256"]
    if not (OUT / "completion.json").exists():
        save(OUT / "completion.json", dict(end_utc=now(), passed=True, expected_fits=336,
             actual_checkpoints=len(list((OUT / "checkpoints").rglob("*.json"))),
             primary_selection_unchanged=True, manuscript_modified=False, verification_pending=True))


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        main()
