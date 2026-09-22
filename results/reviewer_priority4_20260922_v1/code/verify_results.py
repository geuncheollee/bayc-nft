"""Cross-check the four priority-analysis artifacts and freeze their hashes."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "reviewer_priority4_20260922_v1"


def read_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def rows(name: str):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    stage1 = read_json("stage1_evaluation_audit.json")
    assert len(stage1["prior_runs"]) == 4
    assert all(run["run_level_freeze_flag"] is False for run in stage1["prior_runs"].values())
    assert stage1["claim_policy"]["study_level_single_use_untouched_test"] is False
    assert all(source["rows_on_or_after_2026_04_15"] == 0 for source in stage1["raw_trade_sources"].values())

    assert len(rows("stage2_fixed_prediction_sensitivity.csv")) == 14
    stage2 = read_json("stage2_refit_report.json")
    stage2_intervals = rows("stage2_refit_token_intervals.csv")
    assert len(stage2_intervals) == 6
    for collection in ("BAYC", "MAYC"):
        reproduction = stage2["collections"][collection]["archived_reproduction"]
        assert abs(reproduction["metadata_rmse_gap"]) < 1e-12
        assert abs(reproduction["early_rmse_gap"]) < 1e-12
        for row in (r for r in stage2_intervals if r["collection"] == collection):
            value = stage2["collections"][collection]["refits"][row["training_cohort"]]
            assert abs(float(row["early_minus_metadata_rmse"])-value["evaluation_strict_early_minus_metadata_rmse"]) < 1e-12

    stage3 = read_json("stage3_residual_report.json")
    stage3_strict = read_json("stage3_strict_residual_report.json")
    assert len(rows("stage3_metadata_quarter_reproduction.csv")) == 6
    assert len(rows("stage3_residual_intervals.csv")) == 6
    assert set(stage3["collections"]) == set(stage3_strict["collections"]) == {"BAYC", "MAYC"}
    for collection in ("BAYC", "MAYC"):
        assert "strict_without_one_wei" in stage3_strict["collections"][collection]

    stage4 = read_json("stage4_rolling_report.json")
    quarter = rows("stage4_rolling_quarters.csv")
    forward = rows("stage4_forward_selected_quarters.csv")
    assert len(quarter) == 12 and len(forward) == 4
    for collection in ("BAYC", "MAYC"):
        entry = stage4["collections"][collection]
        assert abs(entry["fixed_late_pooled_rmse_reproduced"]-entry["fixed_late_original_pooled_rmse"]) < 1e-8
        assert len(entry["same_quarter_weight_choices"]) == 3
        assert len([r for r in quarter if r["collection"] == collection and r["sample"] == "all"]) == 3
        assert len([r for r in forward if r["collection"] == collection]) == 2

    report = OUT / "analysis_summary.md"
    content = report.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", content):
        assert (OUT / target).resolve().exists(), target
    files = sorted(path for path in OUT.rglob("*") if path.is_file()
                   and path.name != "completion.json" and not path.name.endswith(".pyc"))
    manifest = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "status": "four_priority_analyses_completed_with_no_fresh_untouched_test_available",
        "stage1": "audit complete; truly unused final test unavailable in frozen local data",
        "stage2": "selected metadata and early model strict-cleaning/refit sensitivity complete",
        "stage3": "representative past-only OOF image residual correction complete",
        "stage4": "three rolling quarters and two forward selection transfers complete",
        "files": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}
                  for path in files],
    }
    (OUT / "completion.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verified_files": len(files), "stage2_intervals": len(stage2_intervals),
                      "stage3_intervals": len(rows("stage3_residual_intervals.csv")),
                      "stage4_quarter_rows": len(quarter), "forward_selections": len(forward)}))


if __name__ == "__main__":
    main()
