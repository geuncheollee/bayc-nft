"""Past-only metadata residual prediction from the selected image representation.

Metadata residuals are generated out of fold in 2024 Q2-Q4. Ridge image
correction is tuned on Q4 after fitting on Q2-Q3, then refitted on all three
OOF quarters and assessed on the previously examined 2025+ period.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "reviewer_priority4_20260922_v1"
sys.path.insert(0, str(OUT / "code"))
import run_stage2_refit as stage2
sys.path.insert(0, str(ROOT / "results" / "fusion_uncertainty_20260922_v1" / "code"))
import run_fusion_uncertainty as uncertainty

common = stage2.common
metadata_run = stage2.metadata_run
temporal = stage2.temporal
cr = stage2.cr
ALPHAS = (0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def metadata_oof(collection: str, data: dict, family: str, config: dict) -> tuple[np.ndarray, list[dict]]:
    y = data["y"]
    oof = np.full(len(y), np.nan, dtype=np.float64)
    rows = []
    for origin, end in cr.FINAL_Q:
        train = np.flatnonzero((data["times"] >= "2022-01-01") & (data["times"] < origin))
        valid = np.flatnonzero((data["times"] >= origin) & (data["times"] < end))
        assert len(train) and len(valid)
        params = common.fit_target_transform(y[train])
        prep = common.prepare(data, train, valid, "TF-IDF")
        model, status = temporal.fit_model(prep, family, config, common.target_forward(y[train], params))
        if not status["valid"]:
            raise RuntimeError(f"{collection} {origin}: {status}")
        Xvalid = prep["raw_valid"] if family in common.TREE_FAMILIES else prep["zvalid"]
        pred = common.target_inverse(common.predict_model(model, Xvalid), params)
        oof[valid] = pred
        checkpoint_path = metadata_run.OUT / "checkpoints" / collection / "TF_IDF" / origin / family / f"{common.GRIDS[family].index(config)}.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["training_row_sha256"] == array_sha(data["rowids"][train])
        assert checkpoint["validation_row_sha256"] == array_sha(data["rowids"][valid])
        rmse = float(np.sqrt(np.mean((y[valid]-pred)**2)))
        assert abs(rmse-checkpoint["transaction_rmse"]) < 1e-9
        rows.append({"collection": collection, "quarter_start": origin, "quarter_end": end,
                     "training_transactions": len(train), "validation_transactions": len(valid),
                     "metadata_rmse_reproduced": rmse, "metadata_checkpoint_rmse": checkpoint["transaction_rmse"],
                     "training_source_rows_sha256": array_sha(data["rowids"][train]),
                     "validation_source_rows_sha256": array_sha(data["rowids"][valid])})
        print(f"stage3 {collection} metadata OOF {origin}: n={len(valid)}, RMSE={rmse:.6f}", flush=True)
    return oof, rows


def image_rows(collection: str, encoder: str, tokens: np.ndarray) -> np.ndarray:
    registry = stage2.early_run.read_json(stage2.early_run.REGISTRY_PATH)
    info = registry[encoder][collection]
    assert encoder not in stage2.early_run.REDUCED
    image = np.load(ROOT / info["matrix"], mmap_mode="r", allow_pickle=False)
    mapping = {int(token): index for index, token in enumerate(info["token_ids"])}
    positions = np.asarray([mapping[int(token)] for token in tokens], dtype=np.int64)
    matrix = np.asarray(image[positions], dtype=np.float64)
    assert np.isfinite(matrix).all()
    return matrix


def correction(xtrain: np.ndarray, residual: np.ndarray, xpred: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0:
        return np.zeros(len(xpred), dtype=np.float64)
    scaler = StandardScaler().fit(xtrain)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-6)
    model.fit(scaler.transform(xtrain), residual)
    return np.asarray(model.predict(scaler.transform(xpred)), dtype=np.float64)


def main() -> None:
    if not (OUT / "stage2_refit_report.json").exists():
        raise RuntimeError("Complete step 2 strict-training refit before step 3")
    report = {"design": "time ordered metadata OOF residual + native image Ridge correction",
              "alpha_grid_including_no_correction": list(ALPHAS),
              "test_status": "retrospective, repeatedly inspected 2025+ evaluation; descriptive only",
              "collections": {}}
    quarter_rows = []
    with threadpool_limits(limits=3):
        for collection in ("BAYC", "MAYC"):
            data = metadata_run.windowed_development(collection)
            choice = stage2.chosen(collection)
            oof, rows = metadata_oof(collection, data, choice["metadata_family"], choice["metadata_config"])
            quarter_rows.extend(rows)
            q23 = (data["times"] >= "2024-04-01") & (data["times"] < "2024-10-01")
            q4 = (data["times"] >= "2024-10-01") & (data["times"] < "2025-01-01")
            all_oof = q23 | q4
            assert np.isfinite(oof[all_oof]).all()
            image = image_rows(collection, choice["early_encoder"], data["tokens"])
            residual = data["y"] - oof
            grid = []
            for alpha in ALPHAS:
                adj = correction(image[q23], residual[q23], image[q4], alpha)
                pred = oof[q4] + adj
                grid.append({"alpha": alpha, "q4_rmse": float(np.sqrt(np.mean((data["y"][q4]-pred)**2))),
                             "q4_partial_r2": float(1 - np.sum((data["y"][q4]-pred)**2) /
                                                    np.sum((data["y"][q4]-oof[q4])**2))})
            best = min(grid, key=lambda row: (row["q4_rmse"], row["alpha"]))
            eval_rows = list(stage2.lines(ROOT / "revision" / "target_pipeline_20260909" / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl"))
            y_eval, preds, token_eval, _, _ = uncertainty.load_aligned(collection)
            assert np.array_equal(y_eval, [r["y_log_relative_price"] for r in eval_rows])
            x_eval = image_rows(collection, choice["early_encoder"], np.asarray([r["token_id"] for r in eval_rows]))
            correction_eval = correction(image[all_oof], residual[all_oof], x_eval, best["alpha"])
            pred_eval = preds[:, 0] + correction_eval
            base_sse = float(np.sum((y_eval-preds[:, 0])**2))
            corrected_sse = float(np.sum((y_eval-pred_eval)**2))
            base_rmse = float(np.sqrt(base_sse/len(y_eval)))
            corrected_rmse = float(np.sqrt(corrected_sse/len(y_eval)))
            samples, _ = uncertainty.grouped_bootstrap(y_eval, np.column_stack((preds[:, 0], pred_eval)),
                                                        token_eval, equal_token=False, seed=20260908)
            delta = samples[:, 1]-samples[:, 0]
            lo, hi = uncertainty.bounds(delta, 0.95)
            report["collections"][collection] = {
                "metadata_family": choice["metadata_family"], "image_encoder": choice["early_encoder"],
                "oof_training_rows_q2_q3": int(q23.sum()), "q4_selection_rows": int(q4.sum()),
                "all_oof_residual_training_rows": int(all_oof.sum()), "grid": grid, "chosen_alpha": best["alpha"],
                "q4_chosen_partial_r2": best["q4_partial_r2"], "evaluation_metadata_rmse": base_rmse,
                "evaluation_residual_corrected_rmse": corrected_rmse,
                "evaluation_delta_rmse_corrected_minus_metadata": corrected_rmse-base_rmse,
                "evaluation_predictive_partial_r2": 1-corrected_sse/base_sse,
                "evaluation_delta_rmse_token_bootstrap_95ci": [lo, hi],
            }
            np.savez_compressed(OUT / f"stage3_{collection}_metadata_oof.npz", source_rows=data["rowids"][all_oof],
                                tokens=data["tokens"][all_oof], times=data["times"][all_oof].astype(str),
                                y_lrp=data["y"][all_oof], metadata_oof_pred=oof[all_oof])
            np.savez_compressed(OUT / f"stage3_{collection}_residual_evaluation.npz",
                                source_rows=np.asarray([r["source_row"] for r in eval_rows]), tokens=token_eval,
                                y_lrp=y_eval, metadata_pred=preds[:, 0], residual_corrected_pred=pred_eval)
            print(f"stage3 {collection}: alpha={best['alpha']}, Q4 partial R²={best['q4_partial_r2']:.6f}, eval partial R²={1-corrected_sse/base_sse:.6f}", flush=True)
            (OUT / "stage3_residual_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "stage3_metadata_quarter_reproduction.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(quarter_rows[0]))
        writer.writeheader()
        writer.writerows(quarter_rows)


if __name__ == "__main__":
    main()
