"""Three pre-2025 rolling-origin refits and descriptive selection stability.

Fixed representatives are the previously selected metadata, early, and late
models. Each quarter uses only earlier transactions to fit. Per-quarter
candidate/weight winners are diagnostics, not independent test performance.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "reviewer_priority4_20260922_v1"
sys.path.insert(0, str(OUT / "code"))
import run_stage3_residual as stage3

stage2 = stage3.stage2
common = stage2.common
cr = stage2.cr
early_run = stage2.early_run
image_run = early_run.image_run


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def best_candidate(rows: list[dict], collection: str, origin: str) -> dict:
    usable = [r for r in rows if r["collection"] == collection and r["origin"] == origin and r["valid"] == "True"]
    assert usable
    best = min(usable, key=lambda r: (float(r["transaction_mse"]), r.get("encoder", r.get("encoding", "")), r["family"], r["config"]))
    return {"encoder_or_encoding": best.get("encoder", best.get("encoding")), "family": best["family"],
            "config": json.loads(best["config"]), "same_quarter_selected_rmse": float(best["transaction_rmse"]),
            "valid_candidates": len(usable)}


def image_oof(collection: str, encoder: str, family: str, config: dict,
              metadata_data: dict) -> dict[str, np.ndarray]:
    registry = image_run.read_json(image_run.REGISTRY_PATH)
    data = image_run.load_encoder_data(collection, encoder, registry)
    assert np.array_equal(data["rowids"], metadata_data["rowids"])
    result = {}
    for origin, end in cr.FINAL_Q:
        train = np.flatnonzero((data["times"] >= "2022-01-01") & (data["times"] < origin))
        valid = np.flatnonzero((data["times"] >= origin) & (data["times"] < end))
        params = common.fit_target_transform(data["y"][train])
        transformed = common.target_forward(data["y"], params)
        projected, _ = early_run.projected_image(data, train, origin)
        prep = image_run.prepare_image(data, projected, train, valid, transformed)
        model, _, pred_z, status = image_run.fit_model(prep, family, config, transformed[train])
        if not status["valid"]:
            raise RuntimeError(f"Image rolling fit invalid: {collection} {origin} {status}")
        pred = common.target_inverse(pred_z, params)
        checkpoint_path = image_run.OUT / "checkpoints" / collection / encoder / origin / family / f"{common.GRIDS[family].index(config)}.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["training_row_sha256"] == stage3.array_sha(data["rowids"][train])
        rmse = float(np.sqrt(np.mean((data["y"][valid]-pred)**2)))
        assert abs(rmse-checkpoint["transaction_rmse"]) < 1e-8
        result[origin] = pred
        print(f"stage4 {collection} {encoder}+{family} image OOF {origin}: RMSE={rmse:.6f}", flush=True)
    return result


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y-pred)**2)))


def main() -> None:
    if not (OUT / "stage3_strict_residual_report.json").exists():
        raise RuntimeError("Complete step 3 before rolling stability analysis")
    meta_tuning = csv_rows(stage2.metadata_run.OUT / "pre2025_tuning_all_configs.csv")
    meta_tuning = [r for r in meta_tuning if r["encoding"] == "TF-IDF"]
    early_tuning = csv_rows(early_run.OUT / "pre2025_tuning_all_configs.csv")
    late_rows = csv_rows(ROOT / "results" / "late_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260922_v2" / "development_selected_late_model.csv")
    records = []
    report = {"design": "three expanding-window 2024 Q2-Q4 fits of fixed representatives plus quarter-specific selection diagnostics",
              "quarter_winner_is_not_independent_holdout": True, "collections": {}}
    with threadpool_limits(limits=3):
        for collection in ("BAYC", "MAYC"):
            data = stage2.metadata_run.windowed_development(collection)
            choice = stage2.chosen(collection)
            late = next(r for r in late_rows if r["collection"] == collection)
            assert late["metadata_family"] == choice["metadata_family"]
            image_preds = image_oof(collection, late["encoder"], late["image_family"],
                                    json.loads(late["image_config"]), data)
            with np.load(OUT / f"stage3_{collection}_metadata_oof.npz", allow_pickle=False) as src:
                ids = src["source_rows"].copy()
                meta_oof = src["metadata_oof_pred"].copy()
            index = {int(row): float(pred) for row, pred in zip(ids, meta_oof)}
            target = list(stage2.lines(ROOT / "revision" / "target_pipeline_20260909" / f"{collection.lower()}_development_targets.jsonl"))
            target_map = {r["source_row"]: r for r in target}
            eligible_path = ROOT / "revision" / "data_audit_20260908" / f"{collection.lower()}_trade_eligibility.jsonl"
            wanted = set(map(int, ids))
            strict_map = {r["source_row"]: r["strict_clean"] for r in stage2.lines(eligible_path) if r["source_row"] in wanted}
            assert len(strict_map) == len(wanted)
            w_frozen = float(late["image_weight"])
            pooled = {"y": [], "meta": [], "image": []}
            for origin, end in cr.FINAL_Q:
                valid = np.flatnonzero((data["times"] >= origin) & (data["times"] < end))
                yy = data["y"][valid]
                rowids = data["rowids"][valid]
                mm = np.asarray([index[int(row)] for row in rowids])
                ii = image_preds[origin]
                assert len(ii) == len(yy)
                config = choice["early_config"]
                path = early_run.OUT / "checkpoints" / collection / choice["early_encoder"] / origin / choice["early_family"] / f"{common.GRIDS[choice['early_family']].index(config)}.npy"
                early = np.load(path, allow_pickle=False)
                early_info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                assert early_info["validation_row_sha256"] == stage3.array_sha(rowids)
                assert abs(rmse(yy, early)-early_info["transaction_rmse"]) < 1e-9
                late_pred = (1-w_frozen)*mm + w_frozen*ii
                strict = np.asarray([strict_map[int(row)] for row in rowids], dtype=bool)
                one_wei = np.asarray([target_map[int(row)]["price_eth"] <= 1e-18*(1+1e-9) for row in rowids])
                weights = np.linspace(0, 1, 11)
                same_quarter_weight = min(weights, key=lambda w: (np.sum((yy-((1-w)*mm+w*ii))**2), w))
                for name, mask in (("all", np.ones(len(yy), dtype=bool)), ("strict", strict)):
                    records.append({"collection": collection, "quarter_start": origin, "quarter_end": end,
                                    "sample": name, "transactions": int(mask.sum()),
                                    "strict_transactions": int(strict.sum()), "one_wei_transactions": int((one_wei & mask).sum()),
                                    "metadata_rmse": rmse(yy[mask], mm[mask]),
                                    "early_rmse": rmse(yy[mask], early[mask]),
                                    "late_rmse": rmse(yy[mask], late_pred[mask]),
                                    "early_minus_metadata_rmse": rmse(yy[mask], early[mask])-rmse(yy[mask], mm[mask]),
                                    "late_minus_metadata_rmse": rmse(yy[mask], late_pred[mask])-rmse(yy[mask], mm[mask]),
                                    "frozen_late_image_weight": w_frozen,
                                    "same_quarter_optimal_image_weight_descriptive": float(same_quarter_weight),
                                    "quarter_metadata_winner": json.dumps(best_candidate(meta_tuning, collection, origin)),
                                    "quarter_early_winner": json.dumps(best_candidate(early_tuning, collection, origin))})
                pooled["y"].append(yy)
                pooled["meta"].append(mm)
                pooled["image"].append(ii)
                print(f"stage4 {collection} {origin}: early-meta={records[-2]['early_minus_metadata_rmse']:+.6f}, late-meta={records[-2]['late_minus_metadata_rmse']:+.6f}, weight={float(same_quarter_weight):.1f}", flush=True)
            py, pm, pi = (np.concatenate(pooled[key]) for key in ("y", "meta", "image"))
            pooled_late_rmse = rmse(py, (1-w_frozen)*pm+w_frozen*pi)
            assert abs(pooled_late_rmse-float(late["pre2025_validation_transaction_rmse"])) < 1e-8
            full_rows = [r for r in records if r["collection"] == collection and r["sample"] == "all"]
            report["collections"][collection] = {
                "fixed_late_pooled_rmse_reproduced": pooled_late_rmse,
                "fixed_late_original_pooled_rmse": float(late["pre2025_validation_transaction_rmse"]),
                "early_better_than_metadata_quarters": sum(r["early_minus_metadata_rmse"] < 0 for r in full_rows),
                "late_better_than_metadata_quarters": sum(r["late_minus_metadata_rmse"] < 0 for r in full_rows),
                "same_quarter_weight_choices": [r["same_quarter_optimal_image_weight_descriptive"] for r in full_rows],
                "quarter_metadata_winners": [json.loads(r["quarter_metadata_winner"]) for r in full_rows],
                "quarter_early_winners": [json.loads(r["quarter_early_winner"]) for r in full_rows],
            }
            (OUT / "stage4_rolling_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "stage4_rolling_quarters.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
