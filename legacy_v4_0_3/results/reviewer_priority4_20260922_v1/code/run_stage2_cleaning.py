"""Paired fixed-model cleaning and repeat-sale sensitivity on evaluated trades.

This is descriptive sensitivity conditional on previously selected and fitted
models. Filtering evaluation rows does not re-estimate the training models.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "reviewer_priority4_20260922_v1"
sys.path.insert(0, str(ROOT / "results" / "fusion_uncertainty_20260922_v1" / "code"))
import run_fusion_uncertainty as uncertainty


def lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not (OUT / "stage1_evaluation_audit.json").exists():
        raise RuntimeError("Complete step 1 before step 2")
    records = []
    audit = {}
    for collection in ("BAYC", "MAYC"):
        y, preds, tokens, _, source_audit = uncertainty.load_aligned(collection)
        target_path = ROOT / "revision" / "target_pipeline_20260909" / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl"
        target = list(lines(target_path))
        eligibility_path = ROOT / "revision" / "data_audit_20260908" / f"{collection.lower()}_trade_eligibility.jsonl"
        source_ids = {r["source_row"] for r in target}
        flags = {r["source_row"]: r for r in lines(eligibility_path) if r["source_row"] in source_ids}
        assert len(target) == len(y) == len(flags)
        assert all(r["source_row"] in flags for r in target)
        times = [r["time"] for r in target]
        strict = np.array([flags[r["source_row"]]["strict_clean"] for r in target], dtype=bool)
        reverse = np.array([flags[r["source_row"]]["reverse_pair_24h"] for r in target], dtype=bool)
        multi = np.array([flags[r["source_row"]]["observed_multi_nft_tx"] for r in target], dtype=bool)
        one_wei = np.array([r["price_eth"] <= 1e-18 * (1+1e-9) for r in target], dtype=bool)
        low = np.array([r["price_eth"] < 1e-6 for r in target], dtype=bool)
        first = np.zeros(len(target), dtype=bool)
        observed = set()
        for index in np.argsort(times, kind="stable"):
            token = tokens[index]
            if token not in observed:
                first[index] = True
                observed.add(token)
        frequency = Counter(tokens)
        frequency_histogram = Counter(frequency.values())
        audit[collection] = {
            "evaluation_rows": len(y), "tokens": len(frequency),
            "strict_flag_true": int(strict.sum()), "reverse_pair_flag_true": int(reverse.sum()),
            "multi_nft_tx_flag_true": int(multi.sum()), "one_wei_rows": int(one_wei.sum()),
            "below_1e_minus_6_eth_rows": int(low.sum()),
            "repeat_sale_token_frequency": dict(sorted(frequency_histogram.items())),
            "model_alignment": source_audit,
        }
        development_path = ROOT / "revision" / "target_pipeline_20260909" / f"{collection.lower()}_development_targets.jsonl"
        training = [r for r in lines(development_path) if "2022-01-01" <= r["time"] < "2025-01-01"]
        audit[collection]["training_2022_2024"] = {
            "transactions": len(training), "tokens": len({r["token_id"] for r in training})}
        training_ids = {r["source_row"] for r in training}
        training_flags = {r["source_row"]: r for r in lines(eligibility_path) if r["source_row"] in training_ids}
        assert len(training_flags) == len(training)
        audit[collection]["training_2022_2024"].update({
            "strict_clean_flag_true": sum(training_flags[r["source_row"]]["strict_clean"] for r in training),
            "reverse_pair_flag_true": sum(training_flags[r["source_row"]]["reverse_pair_24h"] for r in training),
            "multi_nft_tx_flag_true": sum(training_flags[r["source_row"]]["observed_multi_nft_tx"] for r in training),
            "one_wei_rows": sum(r["price_eth"] <= 1e-18 * (1+1e-9) for r in training),
            "below_1e_minus_6_eth_rows": sum(r["price_eth"] < 1e-6 for r in training),
        })
        masks = {
            "all_evaluated": np.ones(len(y), dtype=bool),
            "strict_clean_flag": strict,
            "exclude_reverse_pair_24h": ~reverse,
            "exclude_observed_multi_nft_tx": ~multi,
            "exclude_exact_one_wei": ~one_wei,
            "exclude_below_1e_minus_6_eth": ~low,
            "first_evaluation_sale_per_token": first,
        }
        for label, mask in masks.items():
            yy, pp, tt = y[mask], preds[mask], tokens[mask]
            trans, equal = uncertainty.point_scores(yy, pp, tt)
            ci = {}
            if label in {"all_evaluated", "strict_clean_flag", "first_evaluation_sale_per_token"}:
                draws, _ = uncertainty.grouped_bootstrap(yy, pp, tt, equal_token=False,
                                                         seed=20260908)
                for name, candidate in (("early", 1), ("late", 2)):
                    delta = draws[:, candidate] - draws[:, 0]
                    lo, hi = uncertainty.bounds(delta, 0.95)
                    ci[f"{name}_minus_metadata_token_ci95_low"] = lo
                    ci[f"{name}_minus_metadata_token_ci95_high"] = hi
            records.append({
                "collection": collection, "analysis": label,
                "n_transactions": len(yy), "n_tokens": len(set(tt)),
                "metadata_transaction_rmse": float(trans[0]),
                "early_transaction_rmse": float(trans[1]),
                "late_transaction_rmse": float(trans[2]),
                "early_minus_metadata_transaction_rmse": float(trans[1] - trans[0]),
                "late_minus_metadata_transaction_rmse": float(trans[2] - trans[0]),
                "metadata_equal_token_rmse": float(equal[0]),
                "early_equal_token_rmse": float(equal[1]),
                "late_equal_token_rmse": float(equal[2]),
                "early_minus_metadata_equal_token_rmse": float(equal[1] - equal[0]),
                "late_minus_metadata_equal_token_rmse": float(equal[2] - equal[0]),
                **ci,
            })
        print(f"{collection}: {len(y)} evaluated transactions; strict {strict.sum()}, one-wei {one_wei.sum()}, first-sale {first.sum()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "stage2_fixed_prediction_sensitivity.csv", records)
    (OUT / "stage2_filter_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
