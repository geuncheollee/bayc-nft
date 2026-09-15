#!/usr/bin/env python3
"""Self-contained validation for the public v3.3.6.4 release.

This suite intentionally avoids transaction rows, NFT images, feature matrices,
model binaries, row-level predictions, signed approvals, and sealed test targets.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPEC_PATH = ROOT / "public_templates" / "final_execution_specification.PUBLIC.json"
PIPELINE_PATH = (
    ROOT
    / "revision"
    / "final_execution_pipeline_20260914_v3_3_6_4"
    / "code"
    / "pipeline.py"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def import_pipeline():
    helper_dir = ROOT / "revision" / "code"
    sys.path.insert(0, str(helper_dir))
    spec = importlib.util.spec_from_file_location("public_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pipeline module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_json_documents() -> None:
    json_files = sorted(
        p for p in ROOT.rglob("*.json") if ".git" not in p.relative_to(ROOT).parts
    )
    if len(json_files) < 10:
        raise AssertionError(f"expected at least 10 public JSON records, found {len(json_files)}")
    for path in json_files:
        load_json(path)


def check_completed_specification() -> None:
    spec = load_json(SPEC_PATH)
    meta = spec["specification_metadata"]
    assert meta["status"] == "EXECUTION_COMPLETED_AND_PUBLIC_RECORD_PREPARED"
    assert "execution_status" in meta["execution_policy"]
    rendered = json.dumps(spec, sort_keys=True)
    assert "PROPOSED_PENDING_USER_DECISION" not in rendered
    assert "unexecuted_status" not in rendered
    for decision in spec["governance_decisions_status"].values():
        assert decision["status"] == "USER_APPROVED_OPTION_A"
    assert spec["statistical_inference_and_ci"]["bootstrap"]["iterations"] == 2000
    assert spec["statistical_inference_and_ci"]["bootstrap"]["random_seed"] == 20260908


def check_result_statuses_and_bindings() -> None:
    selected = load_json(RESULTS / "selected_configurations.PUBLIC.json")
    refit = load_json(RESULTS / "refit_summary.json")
    metrics = load_json(RESULTS / "test_metrics_summary.json")
    intervals = load_json(RESULTS / "confidence_intervals_summary.json")
    custody = load_json(RESULTS / "evaluation_custody.json")
    source_hashes = load_json(RESULTS / "SOURCE_HASHES.json")

    assert selected["execution_profile"] == "recommended_16"
    assert selected["_metadata"]["approved_scope_decisions_path"] == "WITHHELD_NONPUBLIC_APPROVAL_RECORD"
    assert refit["status"] == "REFIT_COMPLETED" and refit["total_models"] == 16
    assert metrics["status"] == "EVALUATION_COMPLETED"
    assert intervals["status"] == "INFERENCE_COMPLETED"
    assert custody["status"] == "EVALUATION_COMPLETED"
    assert intervals["iterations"] == 2000 and intervals["seed"] == 20260908

    binding = source_hashes["binding"]
    for payload in (selected, refit, metrics, intervals, custody):
        assert payload["approved_execution_manifest_sha256"] == binding["approved_execution_manifest_sha256"]
        assert payload["approved_scope_decisions_sha256"] == binding["approved_scope_decisions_sha256"]
    for payload in (metrics, intervals, custody):
        assert payload["freeze_manifest_sha256"] == binding["freeze_manifest_sha256"]

    for filename in (
        "refit_summary.json",
        "test_metrics_summary.json",
        "confidence_intervals_summary.json",
        "evaluation_custody.json",
    ):
        assert sha256_file(RESULTS / filename) == source_hashes["source_sha256"][filename]


def check_published_primary_results() -> None:
    metrics = load_json(RESULTS / "test_metrics_summary.json")["metrics"]["original_v1"]
    assert metrics["BAYC"]["metadata_baseline"]["n"] == 5120
    assert metrics["MAYC"]["metadata_baseline"]["n"] == 13019
    assert_close(metrics["BAYC"]["metadata_baseline"]["rmse"], 0.34238705387599533)
    assert_close(metrics["BAYC"]["augmented_candidate"]["rmse"], 0.34238705387599533)
    assert_close(metrics["BAYC"]["augmented_candidate"]["utility_percent"], 0.0)
    assert_close(metrics["MAYC"]["metadata_baseline"]["rmse"], 0.16527195758122926)
    assert_close(metrics["MAYC"]["augmented_candidate"]["rmse"], 0.16839478099360022)
    assert_close(metrics["MAYC"]["augmented_candidate"]["utility_percent"], -1.8895059138124721)


def check_published_intervals() -> None:
    collections = load_json(RESULTS / "confidence_intervals_summary.json")["collections"]["original_v1"]
    bayc = collections["BAYC"]["paired_token_cluster_bootstrap"]
    mayc = collections["MAYC"]["paired_token_cluster_bootstrap"]
    assert bayc["augmented_delta_rmse"]["ci_95"] == [0.0, 0.0]
    assert bayc["confirmatory_bonferroni_975"]["delta_rmse_ci_97_5"] == [0.0, 0.0]
    expected = [0.0009483901314478501, 0.005032100262932901]
    for actual, target in zip(mayc["augmented_delta_rmse"]["ci_95"], expected):
        assert_close(actual, target)
    expected_bonf = [0.0006967152783451166, 0.005174714521115443]
    for actual, target in zip(
        mayc["confirmatory_bonferroni_975"]["delta_rmse_ci_97_5"], expected_bonf
    ):
        assert_close(actual, target)


def check_python_sources_compile() -> None:
    for path in sorted(ROOT.rglob("*.py")):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")


def check_pipeline_smoke() -> None:
    pipeline = import_pipeline()
    assert pipeline.CountedPrimalSVR is not None
    assert pipeline.is_exact_one_wei_trade({"amount_raw": "1"}) is True
    assert pipeline.is_exact_one_wei_trade({"price_eth": "0.000000000000000001"}) is True
    assert pipeline.is_exact_one_wei_trade({"amount_raw": 2}) is False
    try:
        pipeline.is_exact_one_wei_trade({"price_eth": "0.0000000000000000005"})
    except ValueError:
        pass
    else:
        raise AssertionError("sub-wei value did not fail closed")
    assert pipeline.build_regressor("Ridge", {"alpha": 1.0}).alpha == 1.0
    hgb = pipeline.build_regressor(
        "HistGradientBoosting",
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 10.0},
    )
    assert hgb.random_state == 20260908 and hgb.early_stopping is False


def check_no_restricted_payloads() -> None:
    forbidden_suffixes = {".npy", ".npz", ".joblib", ".pkl", ".pickle", ".safetensors", ".docx", ".pdf"}
    forbidden_names = {"freeze_manifest.json"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.relative_to(ROOT).parts:
            continue
        if path.name.endswith(".NOT_FOR_SELECTION.jsonl"):
            raise AssertionError(f"sealed target present: {path}")
        if path.suffix.lower() in forbidden_suffixes or path.name in forbidden_names:
            raise AssertionError(f"restricted payload present: {path}")


def check_data_boundary_documented() -> None:
    text = (ROOT / "DATA_ACCESS.md").read_text(encoding="utf-8")
    assert "original Dune SQL" in text
    assert "byte-for-byte reconstruction" in text
    assert "does not" in text
    assert "structured input bundle" in text


def check_repository_manifest() -> None:
    manifest_path = ROOT / "REPRODUCIBILITY_MANIFEST.json"
    manifest = load_json(manifest_path)
    expected = {}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if (
            not path.is_file()
            or ".git" in relative.parts
            or "__pycache__" in relative.parts
            or path == manifest_path
        ):
            continue
        expected[relative.as_posix()] = sha256_file(path)
    recorded = {row["path"]: row["sha256"] for row in manifest["files"]}
    assert manifest["file_count"] == len(recorded)
    assert recorded == expected


CHECKS = [
    ("json_documents", check_json_documents),
    ("completed_specification", check_completed_specification),
    ("result_statuses_and_bindings", check_result_statuses_and_bindings),
    ("published_primary_results", check_published_primary_results),
    ("published_intervals", check_published_intervals),
    ("python_sources_compile", check_python_sources_compile),
    ("pipeline_smoke", check_pipeline_smoke),
    ("no_restricted_payloads", check_no_restricted_payloads),
    ("data_boundary_documented", check_data_boundary_documented),
    ("repository_manifest", check_repository_manifest),
]


def main() -> int:
    failures = []
    for name, function in CHECKS:
        try:
            function()
            print(f"[PASS] {name}")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
    print(f"PUBLIC RELEASE QA: {len(CHECKS) - len(failures)}/{len(CHECKS)} PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
