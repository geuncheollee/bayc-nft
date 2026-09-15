#!/usr/bin/env python3
"""Regenerate manuscript Figures 3 and 4 from public summary JSON only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    return paths


def figure_3(metrics: dict, output_dir: Path) -> list[Path]:
    collections = ("BAYC", "MAYC")
    roles = (
        ("metadata_baseline", "Metadata baseline", "////"),
        ("mandatory_early_benchmark", "Early fusion", "...."),
        ("augmented_candidate", "Selected late fusion", "xx"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    for axis, collection in zip(axes, collections):
        block = metrics["metrics"]["original_v1"][collection]
        values = [block[key]["rmse"] for key, _, _ in roles]
        bars = axis.bar(
            range(len(roles)), values, color="white", edgecolor="black", linewidth=1.0
        )
        for bar, (_, _, hatch) in zip(bars, roles):
            bar.set_hatch(hatch)
        axis.set_xticks(range(len(roles)), [label for _, label, _ in roles], rotation=18, ha="right")
        axis.set_ylabel("Out-of-time RMSE")
        axis.set_title(collection, weight="bold")
        axis.grid(axis="y", color="0.87", linewidth=0.7)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Out-of-sample RMSE across prespecified model roles", weight="bold")
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "Figure_3_RMSE_Model_Roles")
    plt.close(fig)
    return paths


def figure_4(metrics: dict, intervals: dict, output_dir: Path) -> list[Path]:
    collections = ("BAYC", "MAYC")
    points, nominal, corrected = [], [], []
    for collection in collections:
        points.append(metrics["metrics"]["original_v1"][collection]["augmented_candidate"]["utility_percent"])
        block = intervals["collections"]["original_v1"][collection]["paired_token_cluster_bootstrap"]
        nominal.append(block["augmented_utility_percent"]["ci_95"])
        corrected.append(block["confirmatory_bonferroni_975"]["utility_percent_ci_975"])
    values = points + [x for pair in nominal + corrected for x in pair] + [0.0, 1.0]
    xmin, xmax = min(values), max(values)
    padding = max((xmax - xmin) * 0.12, 0.25)
    fig, axis = plt.subplots(figsize=(7.2, 3.55))
    y_positions = [1, 0]
    for y, label, point, ci95, ci975, hatch in zip(
        y_positions, collections, points, nominal, corrected, ("////", "xx")
    ):
        axis.plot(ci975, [y, y], color="0.25", linewidth=1.3, zorder=2)
        axis.plot(ci95, [y, y], color="black", linewidth=5.0, solid_capstyle="butt", zorder=3)
        axis.scatter([point], [y], s=78, facecolor="white", edgecolor="black", linewidth=1.3, hatch=hatch, zorder=4)
        axis.text(point, y + 0.18, f"{point:.2f}%", ha="center", va="bottom", fontsize=8, weight="bold")
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--", label="No incremental utility (0%)")
    axis.axvline(1.0, color="0.35", linewidth=1.0, linestyle=":", label="Prespecified practical guideline (1%)")
    axis.set_xlim(xmin - padding, xmax + padding)
    axis.set_ylim(-0.55, 1.55)
    axis.set_yticks(y_positions, collections)
    axis.set_xlabel("Incremental utility U (percentage points)")
    axis.set_title("Paired token-clustered bootstrap uncertainty for selected late fusion", weight="bold")
    axis.grid(axis="x", color="0.87", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=False, fontsize=7.5)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(axis="y", length=0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    paths = save_figure(fig, output_dir, "Figure_4_Utility_Intervals")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "test_metrics_summary.json"
    intervals_path = args.results_dir / "confidence_intervals_summary.json"
    metrics, intervals = read_json(metrics_path), read_json(intervals_path)
    outputs = figure_3(metrics, args.output_dir) + figure_4(metrics, intervals, args.output_dir)
    manifest = {
        "status": "PUBLIC_FIGURES_REGENERATED",
        "inputs": {
            metrics_path.name: sha256_file(metrics_path),
            intervals_path.name: sha256_file(intervals_path),
        },
        "outputs": {path.name: sha256_file(path) for path in outputs},
    }
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

