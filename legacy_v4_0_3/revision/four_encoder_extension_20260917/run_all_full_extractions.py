"""Run the four passed full-cohort extractors sequentially and resumably.

The program deliberately invokes one encoder at a time.  Each child retains
the per-token state/manifest behaviour in ``run_full_extraction.py``.  A
failure stops the queue, preserving completed work and making the error
visible in ``full_extraction_queue.json`` rather than silently continuing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "pilot_output"
STATE_PATH = OUT / "full_extraction_queue.json"
ENCODERS = ("sam", "sdxl_vae", "dreamsim", "aimv2")


def encoder_complete(name: str) -> bool:
    state_path = OUT / f"full_{name}" / "state.json"
    if not state_path.exists():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    results = state.get("collection_results", {})
    return set(results) == {"BAYC", "MAYC"} and all(
        result.get("all_finite") for result in results.values()
    )


def main() -> None:
    queue = {"status": "RUNNING", "started_unix": time.time(), "encoders": {}}
    STATE_PATH.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    for name in ENCODERS:
        if encoder_complete(name):
            queue["encoders"][name] = {"status": "ALREADY_COMPLETE"}
            STATE_PATH.write_text(json.dumps(queue, indent=2), encoding="utf-8")
            continue
        started = time.time()
        command = [sys.executable, "-u", str(HERE / "run_full_extraction.py"),
                   "--encoder", name, "--execute"]
        completed = subprocess.run(command, cwd=HERE, check=False)
        queue["encoders"][name] = {
            "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
            "returncode": completed.returncode,
            "elapsed_seconds": time.time() - started,
        }
        STATE_PATH.write_text(json.dumps(queue, indent=2), encoding="utf-8")
        if completed.returncode != 0:
            queue["status"] = "STOPPED_ON_FAILURE"
            STATE_PATH.write_text(json.dumps(queue, indent=2), encoding="utf-8")
            raise SystemExit(f"{name} failed; queue stopped without starting later encoders")
    queue["status"] = "COMPLETE"
    queue["completed_unix"] = time.time()
    STATE_PATH.write_text(json.dumps(queue, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
