"""Audit the available final-test boundary and prior reuse of its labels."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "reviewer_priority4_20260922_v1"
RUNS = {
    "metadata": ROOT / "results" / "metadata_asinh_transaction_2022_2024_eight_regressors_20260920_v1",
    "image": ROOT / "results" / "image_asinh_transaction_2022_2024_eight_regressors_20260920_v1",
    "early": ROOT / "results" / "early_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260921_v1",
    "late": ROOT / "results" / "late_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260922_v2",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def bounds(path: Path) -> dict:
    low, high, count = None, None, 0
    after_cutoff = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            stamp = row["block_time"]
            low = stamp if low is None or stamp < low else low
            high = stamp if high is None or stamp > high else high
            after_cutoff += stamp >= "2026-04-15"
            count += 1
    return {"rows": count, "min_time": low, "max_time": high,
            "rows_on_or_after_2026_04_15": after_cutoff, "sha256": sha(path)}


def main() -> None:
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "reviewer response priority 1; retrospective evaluation provenance audit",
        "raw_trade_sources": {}, "prior_runs": {},
        "claim_policy": {
            "run_level_pre_evaluation_freeze": True,
            "study_level_single_use_untouched_test": False,
            "reason": "The same 2025+ test labels were evaluated by metadata, image, early, and late analyses, and the published-in-workspace study protocol evolved after viewing this period.",
            "new_complete_day_after_2026_04_13_in_frozen_local_sources": False,
            "permitted_description": "retrospective out-of-time evaluation with repeated study-level test inspection",
            "remaining_editor_request": "a genuinely unused held-out set cannot be supplied from these frozen local raw files",
        },
    }
    for collection in ("BAYC", "MAYC"):
        path = ROOT / "revision" / f"{collection.lower()}_all_trades.csv"
        report["raw_trade_sources"][collection] = bounds(path)
        assert report["raw_trade_sources"][collection]["rows_on_or_after_2026_04_15"] == 0
    dates = []
    for name, folder in RUNS.items():
        freeze_path = folder / "selection_freeze.json"
        completion_path = folder / "completion.json"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        assert completion["passed"] is True
        date = freeze["created_utc"]
        dates.append(date)
        report["prior_runs"][name] = {
            "folder": str(folder.relative_to(ROOT)), "freeze_utc": date,
            "freeze_sha256": sha(freeze_path), "completion_sha256": sha(completion_path),
            "run_level_freeze_flag": freeze.get("evaluation_labels_loaded_before_freeze",
                                                freeze.get("evaluation_predictions_loaded_before_freeze")),
        }
        assert report["prior_runs"][name]["run_level_freeze_flag"] is False
    assert dates == sorted(dates)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage1_evaluation_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"stage": 1, "raw_max_times": {k: v["max_time"] for k, v in report["raw_trade_sources"].items()},
                      "freeze_dates": dates, "new_untouched_local_test_available": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
