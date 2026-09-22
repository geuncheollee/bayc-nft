"""Read-only input audit for a fresh seven-encoder rerun; no model fitting."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
REV = ROOT / "revision"
ENCODERS = ["DINOv2", "CLIP", "SigLIP2", "SAM", "SDXL_VAE", "DreamSim", "AIM"]
COLLECTIONS = ["BAYC", "MAYC"]
COUNTS = {"BAYC": {"development": 58635, "evaluation": 5120},
          "MAYC": {"development": 113823, "evaluation": 13019}}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(s) for s in f if s.strip()]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def csvsave(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                     for k, v in r.items()} for r in rows)


def locate():
    old = read(REV / "Encoder Development Registry 20260909 v4.json")
    registry = {}
    for encoder, key in [("DINOv2", "dinov2_fullframe"), ("CLIP", "clip_native"), ("SigLIP2", "siglip2")]:
        registry[encoder] = old[key]["collections"]
    search = REV / "four_encoder_extension_20260917"
    for encoder, stem in [("SAM", "sam"), ("SDXL_VAE", "sdxl_vae"),
                          ("DreamSim", "dreamsim"), ("AIM", "aimv2")]:
        registry[encoder] = {}
        for collection in COLLECTIONS:
            matrices = list(search.rglob(f"{collection.lower()}_{stem}_features.npy"))
            # Pilot and full artifacts may coexist: choose the completed full directory only.
            matrices = [p for p in matrices if p.parent.name == "full_" + stem]
            if len(matrices) != 1:
                raise ValueError(f"Ambiguous/missing full embedding: {encoder}/{collection}: {matrices}")
            matrix = matrices[0]
            manifest = matrix.with_name(f"{collection.lower()}_{stem}_manifest.jsonl")
            registry[encoder][collection] = {
                "matrix": str(matrix.relative_to(ROOT)),
                "manifest": str(manifest.relative_to(ROOT)),
                "state": str((matrix.parent / "state.json").relative_to(ROOT))}
    return registry


def main():
    if (OUT / "run_manifest.json").exists():
        raise RuntimeError("Audit already exists; do not overwrite it. Use a separate run directory.")
    started = now()
    master_path = REV / "image_validation_20260909/master_tokens_v1.jsonl"
    master = {(r["collection"], int(r["token_id"])): r for r in lines(master_path)}
    metadata_path = REV / "data_audit_20260908/metadata_normalized.jsonl"
    metadata = {(r["collection"], int(r["token_id"])): r for r in lines(metadata_path)}
    split_path = REV / "image_validation_20260909/image_ready_split_manifest_v1.json"
    split = read(split_path)
    registry = locate()
    input_paths = {master_path, metadata_path, split_path}
    cohorts, issues, audit, details = [], [], [], {}
    target_tokens = {}
    for collection in COLLECTIONS:
        target_tokens[collection] = {}
        for stage, filename in [("development", "development_targets.jsonl"),
                                ("evaluation", "temporal_test_targets.NOT_FOR_SELECTION.jsonl")]:
            path = REV / "target_pipeline_20260909" / f"{collection.lower()}_{filename}"
            input_paths.add(path)
            rows = lines(path)
            # Evaluation labels are deliberately not extracted: integrity audit uses identifiers/dates only.
            ids = [int(r["source_row"]) for r in rows]
            tokens = [int(r["token_id"]) for r in rows]
            times = [r["time"] for r in rows]
            bounded = (all(t < "2025-01-01" for t in times) if stage == "development"
                       else all("2025-01-01" <= t < "2026-04-14" for t in times))
            split_matches = (set(ids) == set(split[collection]["development_source_rows"])) if stage == "development" else True
            bad_meta = sorted(set(t for t in tokens if (collection, t) not in metadata))
            count_ok = len(rows) == COUNTS[collection][stage]
            ok = count_ok and len(set(ids)) == len(ids) and bounded and split_matches and not bad_meta
            record = dict(collection=collection,stage=stage,transactions=len(rows),
                          expected_transactions=COUNTS[collection][stage],unique_tokens=len(set(tokens)),
                          duplicate_source_rows=len(ids)-len(set(ids)),time_min=min(times),time_max=max(times),
                          temporal_boundary_pass=bounded,split_manifest_pass=split_matches,
                          missing_metadata_tokens=bad_meta,passed=ok)
            cohorts.append(record)
            target_tokens[collection][stage] = dict(zip(ids, tokens))
            if not ok:
                issues.append(dict(kind="cohort_failure", **record))
    for encoder in ENCODERS:
        for collection in COLLECTIONS:
            info = registry[encoder][collection]
            matrix_path, manifest_path = ROOT / info["matrix"], ROOT / info["manifest"]
            input_paths.update([matrix_path, manifest_path])
            matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
            manifest = lines(manifest_path)
            tokens = np.full(matrix.shape[0], -1, dtype=np.int64)
            row_counts = Counter()
            mismatch_hashes, mismatch_images, wrong_collections = [], [], []
            for implicit_idx, row in enumerate(manifest):
                idx = int(row.get("feature_row_idx", implicit_idx))
                token = int(row["token_id"])
                row_counts[idx] += 1
                if idx < 0 or idx >= len(tokens):
                    issues.append(dict(kind="manifest_row_out_of_bounds",encoder=encoder,collection=collection,row=idx))
                    continue
                tokens[idx] = token
                expected_hash = row.get("vector_sha256", row.get("feature_sha256"))
                if expected_hash and hashlib.sha256(np.ascontiguousarray(matrix[idx]).tobytes()).hexdigest() != expected_hash:
                    mismatch_hashes.append(token)
                ledger = master.get((collection, token))
                if ledger is None or ledger["image_sha256"] != row.get("image_sha256"):
                    mismatch_images.append(token)
                if row.get("collection", collection) != collection:
                    wrong_collections.append(token)
            expected = {t for (c,t),r in master.items() if c == collection and r.get("image_data_ready",False)}
            actual = set(int(t) for t in tokens)
            duplicate_tokens = sorted(t for t,n in Counter(tokens.tolist()).items() if n > 1)
            missing, extra = sorted(expected-actual), sorted(actual-expected)
            finite_counts = dict(nan=0,positive_inf=0,negative_inf=0)
            low = np.full(matrix.shape[1], np.inf)
            high = np.full(matrix.shape[1], -np.inf)
            for begin in range(0,len(matrix),128):
                block = matrix[begin:begin+128]
                finite_counts["nan"] += int(np.isnan(block).sum())
                finite_counts["positive_inf"] += int(np.isposinf(block).sum())
                finite_counts["negative_inf"] += int(np.isneginf(block).sum())
                low = np.minimum(low, np.min(block,axis=0))
                high = np.maximum(high, np.max(block,axis=0))
            join_issues = {}
            token_map = {int(t): i for i,t in enumerate(tokens)}
            for stage, source_map in target_tokens[collection].items():
                lost = [{"source_row":r,"token_id":t} for r,t in source_map.items() if t not in token_map]
                alignment_ok = not lost and all(int(tokens[token_map[t]]) == t for t in source_map.values())
                join_issues[stage] = dict(missing_transactions=lost,row_alignment_pass=alignment_ok)
            matrix_hash, manifest_hash = sha(matrix_path), sha(manifest_path)
            original_hash_ok = (not info.get("matrix_sha256") or info["matrix_sha256"] == matrix_hash)
            original_hash_ok &= not info.get("manifest_sha256") or info["manifest_sha256"] == manifest_hash
            info.update(matrix_sha256=matrix_hash,manifest_sha256=manifest_hash,
                        dimensions=int(matrix.shape[1]),dtype=str(matrix.dtype),
                        token_id_key="token_id",row_mapping="feature_row_idx" if "feature_row_idx" in manifest[0] else "manifest line order",
                        token_ids=tokens.tolist())
            ok = (len(manifest)==len(matrix) and len(row_counts)==len(matrix) and max(row_counts.values(),default=0)==1
                  and not (missing or extra or duplicate_tokens or mismatch_hashes or mismatch_images or wrong_collections)
                  and not sum(finite_counts.values()) and original_hash_ok
                  and all(v["row_alignment_pass"] for v in join_issues.values()))
            record = dict(collection=collection,encoder=encoder,matrix_path=info["matrix"],manifest_path=info["manifest"],
                          token_id_key="token_id",row_mapping=info["row_mapping"],n_rows=len(matrix),
                          n_unique_tokens=len(actual),expected_tokens=len(expected),native_dimension=matrix.shape[1],dtype=str(matrix.dtype),
                          duplicate_token_count=len(duplicate_tokens),missing_token_count=len(missing),extra_token_count=len(extra),
                          **finite_counts,constant_dimensions=int((low==high).sum()),
                          vector_hash_mismatches=len(mismatch_hashes),image_hash_mismatches=len(mismatch_images),
                          canonical_input_hash_pass=original_hash_ok,
                          development_missing_transactions=len(join_issues["development"]["missing_transactions"]),
                          evaluation_missing_transactions=len(join_issues["evaluation"]["missing_transactions"]),passed=bool(ok))
            audit.append(record)
            details[encoder+"/"+collection] = dict(missing_tokens=missing,extra_tokens=extra,duplicate_tokens=duplicate_tokens,
                                                  mismatched_vector_tokens=mismatch_hashes,mismatched_image_tokens=mismatch_images,
                                                  joins=join_issues,constant_dimension_indices=np.flatnonzero(low==high).tolist())
            if not ok:
                issues.append(dict(kind="embedding_failure", **record))
            print(json.dumps(record),flush=True)
            if info.get("state"):
                input_paths.add(ROOT/info["state"])
    csvsave(OUT / "embedding_audit.csv", audit)
    csvsave(OUT / "cohort_audit.csv", cohorts)
    save(OUT / "embedding_registry.json", registry)
    save(OUT / "audit_details.json", dict(issues=issues,embedding_details=details,
                                         evaluation_labels_used_for_selection=False))
    scripts = [Path(__file__), REV/"code/metadata_baseline_pipeline.py",REV/"code/full_metadata_comparison.py",
               REV/"code/encoder_development_v4.py",REV/"code/counted_primal_svr_v3.py",
               REV/"final_execution_pipeline_20260914_v3_3_6_4/code/pipeline.py"]
    specs = [REV/"Encoder Development Execution Specification 20260909 v4.json",
             REV/"Metadata Comparison Execution Specification 20260909.json",
             REV/"Analysis Amendment 20260909.json",REV/"data_audit_20260908/analysis_specification_v1.json",
             REV/"final_execution_pipeline_20260914_v3_3_6_4/final_execution_specification.json"]
    input_paths.update(specs)
    git = subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True)
    memory = psutil.virtual_memory()
    sd = max(r["native_dimension"] for r in audit)
    nt = max(r["transactions"] for r in cohorts if r["stage"]=="development")
    save(OUT / "run_manifest.json",dict(status="INPUT_AUDIT_COMPLETE" if not issues else "INPUT_AUDIT_FAILED",
        started_utc=started,audit_finished_utc=now(),analysis_finished_utc=None,seed=20260908,
        evaluation_window_description="fixed retrospective out-of-time evaluation window",
        evaluation_start_utc="2025-01-01T00:00:00Z",evaluation_end_exclusive_utc="2026-04-14T00:00:00Z",
        all_decisions_cutoff_exclusive="2025-01-01T00:00:00Z",git_commit=git.stdout.strip() if git.returncode==0 else None,
        git_warning=git.stderr.strip() if git.returncode else None,
        machine=dict(platform=platform.platform(),processor=platform.processor(),hostname=platform.node(),cpu_count=psutil.cpu_count(),
                     physical_memory_bytes=memory.total,available_memory_bytes=memory.available),
        environment=dict(python=sys.version,executable=sys.executable,
                         packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}),
        input_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(input_paths)],
        canonical_scripts=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in scripts],
        embedding_integrity_pass=not issues,cohort_integrity_pass=all(r["passed"] for r in cohorts),
        resource_estimate=dict(native_sdxl_dimension=sd,dense_en_transaction_array_bytes=nt*sd*8,
                               dense_en_gram_bytes=sd*sd*8,note="Lower bounds, excluding other arrays, models and copies.")))
    if issues:
        raise RuntimeError("Input audit failed: do not start regression; see audit_details.json")


if __name__ == "__main__":
    main()
