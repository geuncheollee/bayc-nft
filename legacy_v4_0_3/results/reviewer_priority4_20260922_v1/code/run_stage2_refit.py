"""Refit selected TF-IDF metadata and early models under strict trade flags.

The model families/hyperparameters are fixed from prior pre-2025 selection.
Full-cohort refits first check numerical consistency with the archived runs.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "reviewer_priority4_20260922_v1"
sys.path.insert(0, str(ROOT / "results" / "metadata_asinh_nine_regressors_20260920" / "code"))
import run_experiment as common
metadata_code = ROOT / "results" / "metadata_asinh_transaction_2022_2024_nine_regressors_20260920" / "code" / "run_experiment.py"
metadata_spec = importlib.util.spec_from_file_location("metadata_eight_run", metadata_code)
metadata_run = importlib.util.module_from_spec(metadata_spec)
metadata_spec.loader.exec_module(metadata_run)
sys.path.insert(0, str(ROOT / "results" / "early_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260921" / "code"))
import run_early_fusion_tfidf_experiment as early_run
temporal = metadata_run.temporal
cr = metadata_run.cr


def lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prediction(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        return source["predictions_lrp"].copy()


def chosen(collection: str) -> dict:
    meta_folder = metadata_run.OUT
    early_folder = early_run.OUT
    meta_select = list(csv.DictReader((meta_folder / "tfidf_constrained_metadata_selection.csv").open(encoding="utf-8-sig", newline="")))
    early_select = list(csv.DictReader((early_folder / "development_selected_early_model.csv").open(encoding="utf-8-sig", newline="")))
    m = next(r for r in meta_select if r["collection"] == collection)
    e = next(r for r in early_select if r["collection"] == collection)
    mf = json.loads((meta_folder / "selection_freeze.json").read_text(encoding="utf-8"))
    ef = json.loads((early_folder / "selection_freeze.json").read_text(encoding="utf-8"))
    return {"metadata_family": m["family"], "metadata_config": mf["models"][f"{collection}__TF-IDF__{m['family']}"]["selected_config"],
            "early_encoder": e["encoder"], "early_family": e["family"],
            "early_config": ef["models"][f"{collection}__{e['encoder']}__{e['family']}"]["selected_config"]}


def fit_metadata(data: dict, train: np.ndarray, eval_rows: list[dict], metadata: dict,
                 family: str, config: dict) -> tuple[np.ndarray, dict]:
    params = common.fit_target_transform(data["y"][train])
    prep = common.prepare(data, train, np.asarray([], dtype=np.int64), "TF-IDF")
    target_z = common.target_forward(data["y"][train], params)
    model, status = temporal.fit_model(prep, family, config, target_z)
    if not status["valid"]:
        raise RuntimeError(f"Metadata fit invalid: {status}")
    tokens = np.unique([int(r["token_id"]) for r in eval_rows])
    features = np.asarray([[metadata[data["collection"], int(token)][column] for column in data["columns"]]
                           for token in tokens], dtype=object)
    raw = common.dense(prep["encoding"].transform(features))
    matrix = raw if family in common.TREE_FAMILIES else prep["scaler"].transform(raw)
    pred_unique = common.target_inverse(common.predict_model(model, matrix), params)
    lookup = dict(zip(tokens, pred_unique))
    result = np.asarray([lookup[int(r["token_id"])] for r in eval_rows])
    return result, {"training_transactions": int(len(train)), "training_tokens": int(len(np.unique(data["tokens"][train]))),
                    "target_parameters": params, "fit_status": status}


def fit_early(data: dict, train: np.ndarray, eval_rows: list[dict], metadata: dict,
              family: str, config: dict) -> tuple[np.ndarray, dict]:
    params = common.fit_target_transform(data["y"][train])
    target_z = common.target_forward(data["y"], params)
    image = data["image"]
    prep = early_run.prepare_early(data, image, train, np.asarray([], dtype=np.int64), target_z)
    model, offset, _, status = early_run.image_run.fit_model(prep, family, config, target_z[train])
    if not status["valid"]:
        raise RuntimeError(f"Early fit invalid: {status}")
    tokens = np.unique([int(r["token_id"]) for r in eval_rows])
    mapping = {int(t): i for i, t in enumerate(data["feature_tokens"])}
    rows = np.asarray([mapping[int(token)] for token in tokens])
    X = np.asarray([[metadata[data["collection"], int(token)][column] for column in data["columns"]]
                    for token in tokens], dtype=object)
    Z = cr.vision.transform(prep["state"], X, np.asarray(image[rows], dtype=np.float64))
    if family == "PLS":
        pred_z = np.asarray(model.predict(Z - offset["x"])).reshape(-1) + offset["y"]
    else:
        pred_z = common.predict_model(model, Z)
    pred_unique = common.target_inverse(pred_z, params)
    lookup = dict(zip(tokens, pred_unique))
    result = np.asarray([lookup[int(r["token_id"])] for r in eval_rows])
    return result, {"training_transactions": int(len(train)), "training_tokens": int(len(np.unique(data["tokens"][train]))),
                    "target_parameters": params, "fit_status": status, "feature_dimension": prep["dimension"]}


def main() -> None:
    assert (OUT / "stage2_fixed_prediction_sensitivity.csv").exists()
    metadata = early_run.metadata_lookup()
    report = {"method": "fixed pre-2025-selected family/config, full-versus-strict training refit; 2025+ retrospective evaluation",
              "collections": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for collection in ("BAYC", "MAYC"):
        options = chosen(collection)
        eval_rows = list(lines(ROOT / "revision" / "target_pipeline_20260909" / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl"))
        eligible_path = ROOT / "revision" / "data_audit_20260908" / f"{collection.lower()}_trade_eligibility.jsonl"
        eligibility = {r["source_row"]: r["strict_clean"] for r in lines(eligible_path)}
        eval_strict = np.asarray([eligibility[r["source_row"]] for r in eval_rows])
        y = np.asarray([r["y_log_relative_price"] for r in eval_rows])
        meta_data = metadata_run.windowed_development(collection)
        meta_data["collection"] = collection
        early_data = early_run.load_encoder_data(collection, options["early_encoder"],
                                                 early_run.read_json(early_run.REGISTRY_PATH), metadata)
        assert np.array_equal(meta_data["rowids"], early_data["rowids"])
        full = np.arange(len(meta_data["y"]), dtype=np.int64)
        strict = np.asarray([i for i in full if eligibility[int(meta_data["rowids"][i])]], dtype=np.int64)
        development_path = ROOT / "revision" / "target_pipeline_20260909" / f"{collection.lower()}_development_targets.jsonl"
        one_wei_ids = {r["source_row"] for r in lines(development_path)
                       if "2022-01-01" <= r["time"] < "2025-01-01" and r["price_eth"] <= 1e-18*(1+1e-9)}
        strict_no_onewei = np.asarray([i for i in strict if int(meta_data["rowids"][i]) not in one_wei_ids], dtype=np.int64)
        collection_report = {"selected": options, "refits": {}}
        collection_report["training_exact_one_wei_rows"] = len(one_wei_ids)
        with threadpool_limits(limits=3):
            for label, train in (("full", full), ("strict", strict),
                                 ("strict_without_one_wei", strict_no_onewei)):
                meta_pred, meta_status = fit_metadata(meta_data, train, eval_rows, metadata,
                                                     options["metadata_family"], options["metadata_config"])
                early_pred, early_status = fit_early(early_data, train, eval_rows, metadata,
                                                    options["early_family"], options["early_config"])
                all_rmse = [float(np.sqrt(np.mean((y-p)**2))) for p in (meta_pred, early_pred)]
                strict_rmse = [float(np.sqrt(np.mean((y[eval_strict]-p[eval_strict])**2))) for p in (meta_pred, early_pred)]
                collection_report["refits"][label] = {
                    "metadata": meta_status, "early": early_status,
                    "evaluation_all_metadata_rmse": all_rmse[0], "evaluation_all_early_rmse": all_rmse[1],
                    "evaluation_strict_metadata_rmse": strict_rmse[0], "evaluation_strict_early_rmse": strict_rmse[1],
                    "evaluation_strict_early_minus_metadata_rmse": strict_rmse[1]-strict_rmse[0],
                }
                if label == "full":
                    archived_m = prediction(metadata_run.OUT / "predictions" / f"{collection}_TF_IDF_{options['metadata_family']}.npz")
                    archived_e = prediction(early_run.OUT / "predictions" / f"{collection}_{options['early_encoder']}_{options['early_family']}.npz")
                    collection_report["archived_reproduction"] = {
                        "metadata_max_abs_prediction_gap": float(np.max(np.abs(meta_pred-archived_m))),
                        "early_max_abs_prediction_gap": float(np.max(np.abs(early_pred-archived_e))),
                        "metadata_rmse_gap": all_rmse[0]-float(np.sqrt(np.mean((y-archived_m)**2))),
                        "early_rmse_gap": all_rmse[1]-float(np.sqrt(np.mean((y-archived_e)**2))),
                    }
                    if abs(collection_report["archived_reproduction"]["metadata_rmse_gap"]) > 0.001 or abs(collection_report["archived_reproduction"]["early_rmse_gap"]) > 0.001:
                        raise RuntimeError(f"Full refit does not reproduce archived metrics: {collection_report['archived_reproduction']}")
                np.savez_compressed(OUT / f"stage2_{collection}_{label}_refit_predictions.npz",
                                    source_rows=np.asarray([r["source_row"] for r in eval_rows]), y_lrp=y,
                                    tokens=np.asarray([str(r["token_id"]) for r in eval_rows]),
                                    metadata_pred=meta_pred, early_pred=early_pred, strict_mask=eval_strict)
                print(f"stage2 {collection} {label}: train={len(train)} strict-test ΔRMSE={strict_rmse[1]-strict_rmse[0]:+.6f}", flush=True)
        report["collections"][collection] = collection_report
        (OUT / "stage2_refit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
