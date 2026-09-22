"""Evaluate quarter-selected configurations in the next calendar quarter.

Every next-quarter model was fit on a past-only prefix in the archived tuning
grid. The source grids were already inspected during this retrospective study.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "reviewer_priority4_20260922_v1"
sys.path.insert(0, str(OUT / "code"))
import run_stage4_rolling as stage4


def find_row(rows: list[dict], collection: str, origin: str, winner: dict) -> dict:
    matches = [r for r in rows if r["collection"] == collection and r["origin"] == origin
               and r["family"] == winner["family"] and json.loads(r["config"]) == winner["config"]
               and r.get("encoder", r.get("encoding")) == winner["encoder_or_encoding"]]
    assert len(matches) == 1
    return matches[0]


def main() -> None:
    if not (OUT / "stage4_rolling_report.json").exists():
        raise RuntimeError("Complete step 4 rolling fits first")
    metadata = [r for r in stage4.csv_rows(stage4.stage2.metadata_run.OUT / "pre2025_tuning_all_configs.csv")
                if r["encoding"] == "TF-IDF"]
    early = stage4.csv_rows(stage4.early_run.OUT / "pre2025_tuning_all_configs.csv")
    records = []
    origins = [origin for origin, _ in stage4.cr.FINAL_Q]
    for collection in ("BAYC", "MAYC"):
        for prev, future in zip(origins[:-1], origins[1:]):
            m_choice = stage4.best_candidate(metadata, collection, prev)
            e_choice = stage4.best_candidate(early, collection, prev)
            m_future = find_row(metadata, collection, future, m_choice)
            e_future = find_row(early, collection, future, e_choice)
            assert m_future["valid"] == e_future["valid"] == "True"
            assert m_future["validation_n"] == e_future["validation_n"]
            records.append({
                "collection": collection, "selection_quarter": prev, "forward_quarter": future,
                "forward_validation_transactions": int(m_future["validation_n"]),
                "metadata_selected_family": m_choice["family"],
                "metadata_selected_config": json.dumps(m_choice["config"], sort_keys=True),
                "early_selected_encoder": e_choice["encoder_or_encoding"],
                "early_selected_family": e_choice["family"],
                "early_selected_config": json.dumps(e_choice["config"], sort_keys=True),
                "prior_quarter_metadata_rmse_used_for_selection": m_choice["same_quarter_selected_rmse"],
                "prior_quarter_early_rmse_used_for_selection": e_choice["same_quarter_selected_rmse"],
                "forward_metadata_rmse": float(m_future["transaction_rmse"]),
                "forward_early_rmse": float(e_future["transaction_rmse"]),
                "forward_early_minus_metadata_rmse": float(e_future["transaction_rmse"])-float(m_future["transaction_rmse"]),
            })
    path = OUT / "stage4_forward_selected_quarters.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
