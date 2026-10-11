"""
DINOv2 Full-Frame (No-Crop) Feature Extraction Pipeline for BAYC & MAYC
Strict adherence to:
- revision/Analysis Protocol 20260908.md
- revision/Revision Plan 20260908.md
- revision/image_validation_20260909/cohort_release_v1.json

Key Requirements:
1. Model: facebook/dinov2-large, revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c
2. Input Cohort: revision/image_validation_20260909/master_tokens_v1.jsonl (BAYC: 9,366, MAYC: 12,459)
3. Protocol Preprocessing:
   - EXIF transpose -> RGBA mode -> Center-pad nonsquare with neutral gray RGB(128,128,128)
   - Straight-alpha composite over solid RGB(128,128,128,255) background -> RGB PIL image
4. Full-Frame (No-Crop) Resize:
   - Square input resized directly to 224x224 (do_center_crop=False, size={"height": 224, "width": 224})
   - Center crop strictly disabled.
5. Verification:
   - Border-crop synthetic verification test before execution.
   - Small-batch (BAYC 32 + MAYC 32 = 64) smoke test (Pass 1 vs Pass 2 max_abs_diff == 0.0, finite check).
6. Output:
   - Raw 1024-dimensional float32 embeddings (pooler_output, CLS token).
   - NO L2 normalization, NO PCA, NO global standardization.
   - NO price data or final evaluation targets referenced.
   - Manifests with token_id, image_sha256, feature_row_idx, feature_sha256.
   - Saved in revision/dinov2_fullframe_features_20260909/
"""

import os
import sys
import time
import json
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, Dinov2Model

# ----------------------------------------------------------------------
# Paths and Environment Configuration
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "revision" / "dinov2_fullframe_features_20260909"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_TOKENS_PATH = ROOT / "revision" / "image_validation_20260909" / "master_tokens_v1.jsonl"
COHORT_RELEASE_PATH = ROOT / "revision" / "image_validation_20260909" / "cohort_release_v1.json"

CHECKPOINT_DIR = (
    ROOT / "tmp" / "hf_cache" / "models--facebook--dinov2-large"
    / "snapshots" / "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
)

MODEL_ID = "facebook/dinov2-large"
MODEL_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
BATCH_SIZE = 32

# ----------------------------------------------------------------------
# Helper Utilities
# ----------------------------------------------------------------------
def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def get_gpu_status() -> str:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        return res.stdout.strip()
    except Exception as e:
        return f"Error querying nvidia-smi: {e}"

# ----------------------------------------------------------------------
# Official Image Preprocessing Pipeline
# ----------------------------------------------------------------------
def preprocess_image_to_pil(image_path: Path) -> Tuple[Image.Image, str]:
    raw_bytes = image_path.read_bytes()
    file_sha256 = sha256_bytes(raw_bytes)

    with Image.open(image_path) as raw_img:
        # EXIF orientation
        img = ImageOps.exif_transpose(raw_img)

        # Mode normalization
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Center padding if nonsquare
        w, h = img.size
        if w != h:
            max_dim = max(w, h)
            square_canvas = Image.new("RGBA", (max_dim, max_dim), (128, 128, 128, 0))
            offset = ((max_dim - w) // 2, (max_dim - h) // 2)
            square_canvas.paste(img, offset)
            img = square_canvas

        # Straight-alpha composite over constant RGB(128,128,128)
        background = Image.new("RGBA", img.size, (128, 128, 128, 255))
        composite_rgb = Image.alpha_composite(background, img).convert("RGB")

    return composite_rgb, file_sha256

# ----------------------------------------------------------------------
# Step 1: Border-Crop Verification Test
# ----------------------------------------------------------------------
def verify_border_crop_preservation(processor_default, processor_fullframe) -> Dict[str, Any]:
    print("\n--- Running Synthetic Border-Crop Verification Test ---")
    img_size = 512
    arr = np.full((img_size, img_size, 3), 128, dtype=np.uint8)
    border_width = 4
    arr[:border_width, :, :] = [255, 0, 0] # top
    arr[-border_width:, :, :] = [255, 0, 0] # bottom
    arr[:, :border_width, :] = [255, 0, 0] # left
    arr[:, -border_width:, :] = [255, 0, 0] # right

    test_img = Image.fromarray(arr, mode="RGB")

    # Default processor
    out_def = processor_default(images=[test_img], return_tensors="pt")["pixel_values"]
    def_top = float(out_def[0, 0, 0, :].max())
    def_center = float(out_def[0, 0, 112, 112])

    # Fullframe processor
    out_ff = processor_fullframe(images=[test_img], return_tensors="pt")["pixel_values"]
    ff_top = float(out_ff[0, 0, 0, :].max())
    ff_bottom = float(out_ff[0, 0, -1, :].max())
    ff_left = float(out_ff[0, 0, :, 0].max())
    ff_right = float(out_ff[0, 0, :, -1].max())
    ff_center = float(out_ff[0, 0, 112, 112])

    # Red normalized target is ~2.2489; gray interior is ~0.0741
    border_lost_default = bool(def_top < 1.0)
    border_preserved_fullframe = bool(
        ff_top > 2.0 and ff_bottom > 2.0 and ff_left > 2.0 and ff_right > 2.0
    )

    results = {
        "test_image": {
            "size": [img_size, img_size],
            "border_width_pixels": border_width,
            "border_color_rgb": [255, 0, 0],
            "interior_color_rgb": [128, 128, 128]
        },
        "default_processor": {
            "do_center_crop": processor_default.do_center_crop,
            "size": processor_default.size,
            "crop_size": processor_default.crop_size,
            "top_edge_max_val": def_top,
            "center_val": def_center,
            "border_lost": border_lost_default
        },
        "fullframe_processor": {
            "do_center_crop": processor_fullframe.do_center_crop,
            "size": processor_fullframe.size,
            "top_edge_max_val": ff_top,
            "bottom_edge_max_val": ff_bottom,
            "left_edge_max_val": ff_left,
            "right_edge_max_val": ff_right,
            "center_val": ff_center,
            "border_preserved": border_preserved_fullframe
        },
        "conclusion": "PASSED - Full-frame processor preserves all 4 outer borders without center-crop truncation"
        if border_preserved_fullframe and border_lost_default else "FAILED"
    }

    print(f"  Default processor border lost (cropped): {border_lost_default}")
    print(f"  Full-frame processor border preserved: {border_preserved_fullframe}")
    assert results["conclusion"].startswith("PASSED"), "Border crop verification failed!"
    print("  [OK] Border crop verification PASSED!")
    return results

# ----------------------------------------------------------------------
# Main Execution Pipeline
# ----------------------------------------------------------------------
def run():
    total_start_time = time.perf_counter()
    print("=" * 80)
    print("DINOv2 Full-Frame (No-Crop) Feature Extraction: Specification, Verification & Full Run")
    print(f"Start time (UTC): {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("=" * 80)

    # 1. Environment & Checkpoint Verification
    print("[1/5] Verifying environment & model checkpoint...")
    assert torch.cuda.is_available(), "CUDA GPU is required!"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  Checkpoint snapshot: {CHECKPOINT_DIR}")

    safetensors_path = CHECKPOINT_DIR / "model.safetensors"
    assert safetensors_path.exists(), f"Safetensors not found: {safetensors_path}"
    safetensors_sha256 = sha256_file(safetensors_path)
    print(f"  model.safetensors SHA-256: {safetensors_sha256}")

    # 2. Processors Setup & Verification Test
    print("[2/5] Initializing processors and running border-crop verification...")
    proc_default = AutoImageProcessor.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True, use_fast=False)
    proc_fullframe = AutoImageProcessor.from_pretrained(
        str(CHECKPOINT_DIR),
        local_files_only=True,
        use_fast=False,
        size={"height": 224, "width": 224},
        do_center_crop=False
    )

    border_verification = verify_border_crop_preservation(proc_default, proc_fullframe)
    with open(OUTPUT_DIR / "border_crop_verification.json", "w", encoding="utf-8") as f:
        json.dump(border_verification, f, indent=2, ensure_ascii=False)
    print("  [OK] Saved border_crop_verification.json")

    # Specification & Registry
    spec_and_registry = {
        "pipeline_version": "1.0.0-fullframe",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "checkpoint_safetensors_sha256": safetensors_sha256,
            "checkpoint_path": str(CHECKPOINT_DIR),
            "architecture": {
                "model_type": "dinov2",
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "image_size": 224,
                "patch_size": 14
            },
            "pooling": {
                "method": "CLS token",
                "attribute": "pooler_output",
                "description": "Pre-trained ViT CLS token unnormalized representation"
            },
            "output_dimension": 1024,
            "dtype": "float32"
        },
        "preprocessing_pipeline": {
            "color_space": "sRGB",
            "exif_transpose": True,
            "mode_normalization": "RGBA conversion before compositing",
            "nonsquare_handling": "center-pad with RGB(128,128,128) if width != height (no cropping)",
            "alpha_compositing": "straight-alpha composite over neutral gray RGB(128,128,128,255) background",
            "processor": {
                "class": "BitImageProcessor",
                "initialization_args": {
                    "size": {"height": 224, "width": 224},
                    "do_center_crop": False,
                    "use_fast": False
                },
                "call_kwargs": {
                    "images": "List of PIL Images (mode RGB)",
                    "return_tensors": "pt"
                },
                "runtime_config": proc_fullframe.to_dict()
            }
        },
        "feature_treatment_policy": {
            "full_frame_preservation": "STRICT - center-crop disabled, entire square canvas scaled to 224x224",
            "dimensionality_reduction": "NONE - Raw native dimension strictly preserved; PCA forbidden on full dataset",
            "global_standardization": "NONE - Learned standardization on full dataset strictly forbidden to prevent leakage",
            "normalization": "NONE - Raw model outputs preserved without unverified transformations",
            "price_data_access": "ZERO - No transaction price data, labels, or temporal test targets loaded or referenced"
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "gpu": gpu_name,
            "gpu_initial_status": get_gpu_status()
        }
    }

    with open(OUTPUT_DIR / "spec_and_registry.json", "w", encoding="utf-8") as f:
        json.dump(spec_and_registry, f, indent=2, ensure_ascii=False)
    print("  [OK] Saved spec_and_registry.json")

    # 3. Model Loading onto GPU
    print("[3/5] Loading Dinov2Model onto GPU...")
    model = Dinov2Model.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True, dtype=torch.float32)
    model = model.eval().to("cuda")
    torch.cuda.synchronize()
    print("  [OK] Dinov2Model loaded to CUDA successfully.")

    # 4. Small-Batch Verification (Smoke Test)
    print("[4/5] Performing small-batch verification (BAYC 32 + MAYC 32 = 64)...")
    tokens_by_coll = {"BAYC": [], "MAYC": []}
    with open(MASTER_TOKENS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            tokens_by_coll[item["collection"]].append(item)

    assert len(tokens_by_coll["BAYC"]) == 9366, f"Expected 9,366 BAYC tokens, got {len(tokens_by_coll['BAYC'])}"
    assert len(tokens_by_coll["MAYC"]) == 12459, f"Expected 12,459 MAYC tokens, got {len(tokens_by_coll['MAYC'])}"

    rng = np.random.default_rng(42)
    bayc_sample_indices = rng.choice(len(tokens_by_coll["BAYC"]), size=32, replace=False)
    mayc_sample_indices = rng.choice(len(tokens_by_coll["MAYC"]), size=32, replace=False)
    sample_tokens = [tokens_by_coll["BAYC"][i] for i in bayc_sample_indices] + [tokens_by_coll["MAYC"][i] for i in mayc_sample_indices]

    torch.cuda.reset_peak_memory_stats()
    v_start = time.perf_counter()

    for pass_num in [1, 2]:
        feats_pass = []
        for idx in range(0, len(sample_tokens), BATCH_SIZE):
            batch_slice = sample_tokens[idx : idx + BATCH_SIZE]
            imgs = []
            for item in batch_slice:
                p = ROOT / Path(item["image_path"])
                img, sha = preprocess_image_to_pil(p)
                assert sha == item["image_sha256"], f"SHA mismatch on {item['token_id']}"
                imgs.append(img)

            inputs = proc_fullframe(images=imgs, return_tensors="pt")
            pv = inputs["pixel_values"].to("cuda")
            with torch.inference_mode():
                out = model(pixel_values=pv)
                feat = out.pooler_output
                assert torch.isfinite(feat).all(), f"Non-finite values detected in pass {pass_num}"
                assert feat.shape[1] == 1024, f"Output dim mismatch: {feat.shape[1]}"
                feats_pass.append(feat.cpu().numpy())

        if pass_num == 1:
            p1_feats = np.concatenate(feats_pass, axis=0)
        else:
            p2_feats = np.concatenate(feats_pass, axis=0)

    max_abs_diff = float(np.max(np.abs(p1_feats - p2_feats)))
    v_elapsed = time.perf_counter() - v_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    verification_results = {
        "sample_size": len(sample_tokens),
        "output_dim": 1024,
        "all_finite": bool(np.isfinite(p1_feats).all()),
        "reproducibility_max_abs_diff": max_abs_diff,
        "verified": bool(max_abs_diff == 0.0 and np.isfinite(p1_feats).all() and p1_feats.shape[1] == 1024),
        "mean_norm": float(np.mean(np.linalg.norm(p1_feats, axis=1))),
        "verification_seconds": v_elapsed,
        "seconds_per_image": v_elapsed / (len(sample_tokens) * 2),
        "peak_vram_mb": peak_vram_mb
    }

    print(f"  Pass 1 & 2 diff: {max_abs_diff:.2e} | Peak VRAM: {peak_vram_mb:.1f} MB | Elapsed: {v_elapsed:.2f}s")
    assert verification_results["verified"], "Small-batch verification failed!"
    with open(OUTPUT_DIR / "verification_small_batch.json", "w", encoding="utf-8") as f:
        json.dump(verification_results, f, indent=2, ensure_ascii=False)
    print("  [OK] Saved verification_small_batch.json")

    # 5. Full Batch Extraction
    print("[5/5] Extracting full-frame raw embeddings for BAYC (9,366) and MAYC (12,459)...")
    extraction_summary = {
        "extraction_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collections": {}
    }
    failed_tokens = []

    for coll_name in ["BAYC", "MAYC"]:
        coll_items = tokens_by_coll[coll_name]
        total_coll = len(coll_items)
        print(f"\n  === Processing {coll_name} ({total_coll} tokens) ===")
        coll_start = time.perf_counter()

        feature_matrix = np.zeros((total_coll, 1024), dtype=np.float32)
        manifest_rows = []
        torch.cuda.reset_peak_memory_stats()

        processed_count = 0
        for offset in range(0, total_coll, BATCH_SIZE):
            batch_items = coll_items[offset : offset + BATCH_SIZE]
            imgs = []
            valid_meta = []

            for item in batch_items:
                img_path = ROOT / Path(item["image_path"])
                try:
                    if not img_path.exists():
                        raise FileNotFoundError(f"File not found: {img_path}")
                    pil_img, file_sha256 = preprocess_image_to_pil(img_path)
                    if file_sha256 != item["image_sha256"]:
                        raise ValueError(f"Image SHA mismatch: expected {item['image_sha256']}, got {file_sha256}")
                    imgs.append(pil_img)
                    valid_meta.append((item, file_sha256))
                except Exception as ex:
                    fail_rec = {
                        "collection": coll_name,
                        "token_id": item["token_id"],
                        "image_path": item["image_path"],
                        "stage": "preprocess",
                        "error": str(ex)
                    }
                    failed_tokens.append(fail_rec)

            if not imgs:
                continue

            # Batch forward pass: DINOv2 Full-Frame
            inputs = proc_fullframe(images=imgs, return_tensors="pt")
            pv = inputs["pixel_values"].to("cuda")
            with torch.inference_mode():
                out = model(pixel_values=pv)
                feat = out.pooler_output
                assert torch.isfinite(feat).all(), f"Non-finite values in batch {offset}"
                feat_np = feat.cpu().numpy().astype(np.float32)

            for b_i, (item, img_sha) in enumerate(valid_meta):
                row_idx = offset + b_i
                vec = feat_np[b_i]
                feature_matrix[row_idx] = vec

                v_sha = sha256_bytes(vec.tobytes())
                manifest_rows.append({
                    "collection": coll_name,
                    "token_id": item["token_id"],
                    "feature_row_idx": row_idx,
                    "image_path": item["image_path"],
                    "image_sha256": img_sha,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "output_dim": 1024,
                    "dtype": "float32",
                    "feature_sha256": v_sha
                })

            processed_count += len(valid_meta)
            if (offset // BATCH_SIZE) % 50 == 0 or processed_count == total_coll:
                elapsed = time.perf_counter() - coll_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"    [{coll_name}] {processed_count}/{total_coll} processed ({rate:.1f} img/s)...", flush=True)

        # Save numpy file
        out_npy_path = OUTPUT_DIR / f"{coll_name.lower()}_features_dinov2_fullframe.npy"
        np.save(out_npy_path, feature_matrix)
        npy_sha = sha256_file(out_npy_path)

        # Save manifest jsonl
        out_man_path = OUTPUT_DIR / f"manifest_{coll_name.lower()}_dinov2_fullframe.jsonl"
        with open(out_man_path, "w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        coll_elapsed = time.perf_counter() - coll_start
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        extraction_summary["collections"][coll_name] = {
            "total_tokens": total_coll,
            "successful_extractions": processed_count,
            "failed_extractions": total_coll - processed_count,
            "elapsed_seconds": coll_elapsed,
            "throughput_images_per_second": processed_count / coll_elapsed,
            "peak_vram_mb": peak_vram,
            "file": out_npy_path.name,
            "shape": list(feature_matrix.shape),
            "dtype": str(feature_matrix.dtype),
            "sha256": npy_sha,
            "all_finite": bool(np.isfinite(feature_matrix).all())
        }
        print(f"  [OK] {coll_name} complete in {coll_elapsed:.1f}s ({processed_count / coll_elapsed:.1f} img/s).")

    # Finalize summary
    total_elapsed = time.perf_counter() - total_start_time
    extraction_summary["total_elapsed_seconds"] = total_elapsed
    extraction_summary["total_tokens_processed"] = sum(c["successful_extractions"] for c in extraction_summary["collections"].values())
    extraction_summary["overall_throughput_img_per_sec"] = extraction_summary["total_tokens_processed"] / total_elapsed
    extraction_summary["failed_count"] = len(failed_tokens)

    with open(OUTPUT_DIR / "failed_tokens.json", "w", encoding="utf-8") as f:
        json.dump(failed_tokens, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "extraction_summary.json", "w", encoding="utf-8") as f:
        json.dump(extraction_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("DINOV2 FULL-FRAME EXTRACTION COMPLETED SUCCESSFULLY!")
    print(f"Total tokens processed: {extraction_summary['total_tokens_processed']} / 21,825")
    print(f"Failures: {len(failed_tokens)}")
    print(f"Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    run()
