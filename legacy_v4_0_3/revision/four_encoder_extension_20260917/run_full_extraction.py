"""Resumable full-cohort extraction, enabled only after a passed pilot.

Usage: python run_full_extraction.py --encoder sam --execute
The program never loads price, target, temporal-split, or evaluation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_pilot import (DEVICE, LOADERS, MASTER, MODEL_SPECS, OUT, ROOT,
                       gpu_status, package_versions, prepare_image, sha256_array)


def read_items() -> dict[str, list[dict]]:
    result = {"BAYC": [], "MAYC": []}
    for line in MASTER.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["collection"] in result and item["image_data_ready"]:
            result[item["collection"]].append(item)
    return {key: sorted(value, key=lambda row: int(row["token_id"])) for key, value in result.items()}


def passed_pilot(name: str) -> dict:
    path = OUT / f"pilot_{name}_summary.json"
    if not path.exists():
        raise RuntimeError(f"missing pilot result: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    record = data.get("encoders", {}).get(name, {})
    if record.get("status") != "PASSED":
        raise RuntimeError(f"pilot did not pass for {name}: {record.get('status')}; {record.get('error')}")
    if len(record.get("dimension", [])) != 1:
        raise RuntimeError(f"pilot did not establish one dimension for {name}")
    return record


def run(name: str) -> None:
    pilot = passed_pilot(name)
    dimension = pilot["dimension"][0]
    destination = OUT / f"full_{name}"
    destination.mkdir(exist_ok=True)
    state_path = destination / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed": {}}
    cohorts = read_items()
    model = None
    total_started = time.perf_counter()
    try:
        model, embed = LOADERS[name]()
        for collection, items in cohorts.items():
            matrix_path = destination / f"{collection.lower()}_{name}_features.npy"
            manifest_path = destination / f"{collection.lower()}_{name}_manifest.jsonl"
            expected = len(items)
            if matrix_path.exists() and matrix_path.stat().st_size:
                matrix = np.lib.format.open_memmap(matrix_path, mode="r+", dtype=np.float32,
                                                   shape=(expected, dimension))
            else:
                matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32,
                                                   shape=(expected, dimension))
            done = set(state["completed"].get(collection, []))
            by_token = {str(item["token_id"]): item for item in items}
            index_by_token = {str(item["token_id"]): index for index, item in enumerate(items)}
            prior_rows = []
            if manifest_path.exists():
                prior_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
            rows = {str(row["token_id"]): row for row in prior_rows}
            # A process can be interrupted after an array row and state entry
            # have been flushed but before the collection manifest is emitted.
            # Preserve that valid completed work by reconstructing only the
            # missing manifest rows from the frozen source image and mmap row.
            done.update(rows)
            recovered = sorted(done.difference(rows), key=int)
            if recovered:
                pilot_method = pilot.get("output_method", ["unknown"])[0]
                for token in recovered:
                    item = by_token.get(token)
                    if item is None:
                        raise RuntimeError(f"{collection}:{token} is in state but absent from frozen ledger")
                    index = index_by_token[token]
                    saved_vector = matrix[index]
                    if not np.isfinite(saved_vector).all():
                        raise RuntimeError(f"{collection}:{token} recovered row has non-finite values")
                    rows[token] = {
                        "collection": collection, "token_id": item["token_id"],
                        # The feature was written only after prepare_image()
                        # verified this frozen-ledger hash in the interrupted
                        # extraction. Reuse it rather than re-reading every
                        # source image merely to rebuild the manifest.
                        "image_sha256": item["image_sha256"], "vector_sha256": sha256_array(saved_vector),
                        "dimension": dimension, "output_method": pilot_method,
                        "manifest_reconstructed_from_flushed_state": True,
                    }
                state.setdefault("manifest_recovery", {})[collection] = {
                    "reconstructed_rows": len(recovered),
                    "reason": "state/matrix persisted before collection manifest",
                }
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            started = time.perf_counter()
            for index, item in enumerate(items):
                token = str(item["token_id"])
                if token in done:
                    continue
                image, observed_hash = prepare_image(item)
                vector, output_method = embed(image)
                if vector.shape != (1, dimension):
                    raise RuntimeError(f"{collection}:{token} output shape {vector.shape}; expected (1,{dimension})")
                if not np.isfinite(vector).all():
                    raise RuntimeError(f"{collection}:{token} has non-finite values")
                matrix[index] = vector[0]
                matrix.flush()
                rows[token] = {
                    "collection": collection, "token_id": item["token_id"],
                    "image_sha256": observed_hash, "vector_sha256": sha256_array(vector[0]),
                    "dimension": dimension, "output_method": output_method,
                }
                done.add(token)
                state["completed"][collection] = sorted(done, key=int)
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                if len(done) % 25 == 0 or len(done) == expected:
                    print(f"{name} {collection}: {len(done)}/{expected}", flush=True)
            if len(done) != expected:
                raise RuntimeError(f"{collection} incomplete: {len(done)}/{expected}")
            ordered_rows = [rows[str(item["token_id"])] for item in items]
            manifest_path.write_text("\n".join(json.dumps(row) for row in ordered_rows) + "\n", encoding="utf-8")
            matrix.flush()
            npy_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
            state.setdefault("collection_results", {})[collection] = {
                "tokens": expected, "dimension": dimension, "feature_file": matrix_path.name,
                "feature_sha256": npy_hash, "manifest_file": manifest_path.name,
                "elapsed_seconds": time.perf_counter() - started,
                "all_finite": bool(np.isfinite(matrix).all()),
            }
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
    state.update({
        "encoder": name, "spec": MODEL_SPECS[name], "pilot_matrix_sha256": pilot.get("matrix_sha256"),
        "master_ledger": str(MASTER.relative_to(ROOT)), "packages": package_versions(),
        "gpu_status_after": gpu_status(), "total_elapsed_seconds": time.perf_counter() - total_started,
    })
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"encoder": name, "status": "COMPLETE", "seconds": state["total_elapsed_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=list(LOADERS))
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        raise SystemExit("Review the passed pilot, then supply --execute.")
    run(arguments.encoder)
