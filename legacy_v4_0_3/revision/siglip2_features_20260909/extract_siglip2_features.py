"""
SigLIP 2 Feature Extraction Pipeline for BAYC & MAYC
Strict adherence to:
- revision/Analysis Protocol 20260908.md
- revision/image_validation_20260909/cohort_release_v1.json

Constraints:
- Preserves raw images and previous outputs
- Outputs to dedicated directory: revision/siglip2_features_20260909/
- Specifications for checkpoint, revision, pooling, output dimension, preprocessing
- Small-batch verification (BAYC & MAYC) before full run
- Raw native 1024-dimensional embeddings (NO PCA, NO global standardization, NO L2 normalization)
- NO price data or final evaluation targets used
- SHA-256 verification of raw images and feature outputs
- Complete failure tracking and manifests
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
from transformers import AutoImageProcessor, SiglipVisionModel

# ----------------------------------------------------------------------
# Paths and Environment Configuration
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "revision" / "siglip2_features_20260909"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_TOKENS_PATH = ROOT / "revision" / "image_validation_20260909" / "master_tokens_v1.jsonl"
COHORT_RELEASE_PATH = ROOT / "revision" / "image_validation_20260909" / "cohort_release_v1.json"

CHECKPOINT_DIR = (
    ROOT / "tmp" / "siglip2_timing" / "hf_cache" / "hub"
    / "models--google--siglip2-large-patch16-256"
    / "snapshots" / "787800c8990e6f058423089178e718139608408c"
)

MODEL_ID = "google/siglip2-large-patch16-256"
MODEL_REVISION = "787800c8990e6f058423089178e718139608408c"
BATCH_SIZE = 16

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
    """
    1. Read raw bytes and calculate SHA-256.
    2. EXIF orientation correction.
    3. Ensure RGBA mode for straight-alpha compositing.
    4. If nonsquare, center-pad to square with neutral gray RGB(128,128,128).
    5. Straight-alpha composite over solid neutral gray RGB(128,128,128,255) background.
    6. Convert to RGB PIL Image.
    """
    raw_bytes = image_path.read_bytes()
    file_sha256 = sha256_bytes(raw_bytes)

    with Image.open(image_path) as raw_img:
        # EXIF orientation
        img = ImageOps.exif_transpose(raw_img)

        # Mode normalization
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Center padding if nonsquare (Protocol: center pad to square with neutral gray)
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
# Main Execution Pipeline
# ----------------------------------------------------------------------
def run():
    total_start_time = time.perf_counter()
    print("=" * 70)
    print("SigLIP 2 Feature Extraction: Specification, Verification & Full Run")
    print(f"Start time (UTC): {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("=" * 70)

    # 1. System & Environment Verification
    print("[1/5] Verifying environment & checkpoint...")
    assert torch.cuda.is_available(), "CUDA GPU is required!"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  Checkpoint directory: {CHECKPOINT_DIR}")
    assert CHECKPOINT_DIR.exists(), f"Checkpoint directory does not exist: {CHECKPOINT_DIR}"

    safetensors_path = CHECKPOINT_DIR / "model.safetensors"
    assert safetensors_path.exists(), f"model.safetensors not found in {CHECKPOINT_DIR}"
    safetensors_sha256 = sha256_file(safetensors_path)
    print(f"  model.safetensors SHA-256: {safetensors_sha256}")

    # 2. Specification & Registry
    spec_and_registry = {
        "model_name": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_safetensors_sha256": safetensors_sha256,
        "checkpoint_path": str(CHECKPOINT_DIR),
        "architecture": {
            "model_type": "siglip_vision_model",
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "image_size": 256,
            "patch_size": 16
        },
        "pooling": {
            "method": "MultiheadAttentionPoolingHead",
            "attribute": "pooler_output",
            "description": "Multi-head attention pooling over vision transformer patch sequence"
        },
        "output_dimension": 1024,
        "dtype": "float32",
        "preprocessing_pipeline": {
            "color_space": "sRGB",
            "exif_transpose": True,
            "mode_normalization": "RGBA conversion before compositing",
            "nonsquare_handling": "center-pad with RGB(128,128,128) if width != height (no cropping)",
            "alpha_compositing": "straight-alpha composite over neutral gray RGB(128,128,128,255) background",
            "processor": {
                "class": "SiglipImageProcessor",
                "image_size": [256, 256],
                "resample": "bicubic (PIL.Image.Resampling.BICUBIC, value=2)",
                "rescale_factor": 1 / 255.0,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "normalized_range": "[-1.0, 1.0]"
            }
        },
        "feature_treatment_policy": {
            "native_dimension": 1024,
            "dimensionality_reduction": "NONE - Raw native dimension strictly preserved; PCA forbidden on full dataset",
            "global_standardization": "NONE - Learned standardization on full dataset strictly forbidden to prevent leakage",
            "normalization": "NONE - Raw model outputs preserved without unverified transformations",
            "price_data_access": "ZERO - No transaction price data, labels, or temporal test targets loaded or referenced"
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": "4.56.2",
            "gpu": gpu_name,
            "gpu_initial_status": get_gpu_status()
        }
    }

    with open(OUTPUT_DIR / "spec_and_registry.json", "w", encoding="utf-8") as f:
        json.dump(spec_and_registry, f, indent=2, ensure_ascii=False)
    print("  [OK] Saved spec_and_registry.json")

    # 3. Model & Processor Loading
    print("[2/5] Loading processor and SiglipVisionModel...")
    processor = AutoImageProcessor.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True, use_fast=False)
    model = SiglipVisionModel.from_pretrained(
        str(CHECKPOINT_DIR), local_files_only=True, torch_dtype=torch.float32
    )
    model = model.eval().to("cuda")
    torch.cuda.synchronize()
    print("  [OK] Model loaded to CUDA successfully.")

    # 4. Small-Batch Verification (Smoke Test)
    print("[3/5] Performing small-batch verification (BAYC & MAYC)...")
    # Read cohort master tokens
    tokens_by_coll = {"BAYC": [], "MAYC": []}
    with open(MASTER_TOKENS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            tokens_by_coll[item["collection"]].append(item)

    print(f"  Loaded master tokens: BAYC={len(tokens_by_coll['BAYC'])}, MAYC={len(tokens_by_coll['MAYC'])}")
    assert len(tokens_by_coll["BAYC"]) == 9366, f"Expected 9,366 BAYC tokens, got {len(tokens_by_coll['BAYC'])}"
    assert len(tokens_by_coll["MAYC"]) == 12459, f"Expected 12,459 MAYC tokens, got {len(tokens_by_coll['MAYC'])}"

    rng = np.random.default_rng(42)
    bayc_sample_indices = rng.choice(len(tokens_by_coll["BAYC"]), size=64, replace=False)
    mayc_sample_indices = rng.choice(len(tokens_by_coll["MAYC"]), size=64, replace=False)
    bayc_sample = [tokens_by_coll["BAYC"][i] for i in bayc_sample_indices]
    mayc_sample = [tokens_by_coll["MAYC"][i] for i in mayc_sample_indices]

    verification_results = {}
    for coll_name, sample_items in [("BAYC", bayc_sample), ("MAYC", mayc_sample)]:
        print(f"  Verifying {coll_name} (sample size=64)...")
        v_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        
        # Pass 1
        features_pass1 = []
        for idx in range(0, len(sample_items), BATCH_SIZE):
            batch_items = sample_items[idx : idx + BATCH_SIZE]
            imgs = []
            for item in batch_items:
                img_p = ROOT / Path(item["image_path"])
                pil_img, sha = preprocess_image_to_pil(img_p)
                assert sha == item["image_sha256"], f"SHA mismatch on {item['token_id']}"
                imgs.append(pil_img)
            
            inputs = processor(images=imgs, return_tensors="pt")
            pv = inputs["pixel_values"].to("cuda")
            with torch.inference_mode():
                out = model(pixel_values=pv)
                feat = out.pooler_output
                assert torch.isfinite(feat).all(), f"Non-finite values detected in {coll_name} pass 1"
                assert feat.shape[1] == 1024, f"Output dim mismatch: {feat.shape[1]}"
                features_pass1.append(feat.cpu().numpy())
        
        features_pass1 = np.concatenate(features_pass1, axis=0)

        # Pass 2 (Reproducibility check)
        features_pass2 = []
        for idx in range(0, len(sample_items), BATCH_SIZE):
            batch_items = sample_items[idx : idx + BATCH_SIZE]
            imgs = []
            for item in batch_items:
                img_p = ROOT / Path(item["image_path"])
                pil_img, _ = preprocess_image_to_pil(img_p)
                imgs.append(pil_img)
            inputs = processor(images=imgs, return_tensors="pt")
            pv = inputs["pixel_values"].to("cuda")
            with torch.inference_mode():
                out = model(pixel_values=pv)
                features_pass2.append(out.pooler_output.cpu().numpy())
        features_pass2 = np.concatenate(features_pass2, axis=0)

        max_diff = float(np.max(np.abs(features_pass1 - features_pass2)))
        v_elapsed = time.perf_counter() - v_start
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        verification_results[coll_name] = {
            "sample_size": len(sample_items),
            "output_shape": list(features_pass1.shape),
            "all_finite": bool(np.isfinite(features_pass1).all()),
            "mean_norm": float(np.mean(np.linalg.norm(features_pass1, axis=1))),
            "reproducibility_max_abs_diff": max_diff,
            "reproducible": bool(max_diff == 0.0),
            "verification_seconds": v_elapsed,
            "seconds_per_image": v_elapsed / len(sample_items),
            "peak_memory_mb": peak_mem_mb,
            "verified": bool(max_diff == 0.0 and np.isfinite(features_pass1).all() and features_pass1.shape[1] == 1024)
        }
        print(f"    Pass 1 & 2 diff: {max_diff:.2e}, Peak Mem: {peak_mem_mb:.1f} MB, Rate: {len(sample_items)/v_elapsed:.1f} img/s")
        assert verification_results[coll_name]["verified"], f"Verification failed for {coll_name}!"

    with open(OUTPUT_DIR / "verification_small_batch.json", "w", encoding="utf-8") as f:
        json.dump(verification_results, f, indent=2, ensure_ascii=False)
    print("  [OK] Saved verification_small_batch.json")

    # 5. Full Raw Embedding Extraction
    print("[4/5] Extracting full raw embeddings (BAYC: 9,366, MAYC: 12,459)...")
    extraction_stats = {}
    failed_tokens = []

    for coll_name in ["BAYC", "MAYC"]:
        coll_items = tokens_by_coll[coll_name]
        total_coll = len(coll_items)
        print(f"\n  Starting {coll_name}: {total_coll} tokens...")
        c_start = time.perf_counter()
        
        embeddings_list = []
        manifest_rows = []
        coll_failed = []

        torch.cuda.reset_peak_memory_stats()

        for offset in range(0, total_coll, BATCH_SIZE):
            batch_slice = coll_items[offset : offset + BATCH_SIZE]
            imgs = []
            valid_batch_meta = []

            for item in batch_slice:
                img_path = ROOT / Path(item["image_path"])
                try:
                    if not img_path.exists():
                        raise FileNotFoundError(f"File not found: {img_path}")
                    pil_img, file_sha256 = preprocess_image_to_pil(img_path)
                    if file_sha256 != item["image_sha256"]:
                        raise ValueError(f"Image SHA mismatch: expected {item['image_sha256']}, got {file_sha256}")
                    imgs.append(pil_img)
                    valid_batch_meta.append((item, file_sha256))
                except Exception as ex:
                    fail_record = {
                        "collection": coll_name,
                        "token_id": item["token_id"],
                        "image_path": item["image_path"],
                        "stage": "load_and_preprocess",
                        "error": str(ex)
                    }
                    failed_tokens.append(fail_record)
                    coll_failed.append(fail_record)

            if not imgs:
                continue

            # Forward pass
            try:
                inputs = processor(images=imgs, return_tensors="pt")
                pv = inputs["pixel_values"].to("cuda")
                with torch.inference_mode():
                    out = model(pixel_values=pv)
                    feat_batch = out.pooler_output # (B, 1024)
                    
                    # Finite check
                    if not torch.isfinite(feat_batch).all():
                        raise ValueError("Non-finite values (NaN/Inf) detected in model output batch")
                    
                    feat_np = feat_batch.cpu().numpy().astype(np.float32)

                for b_i, (item, img_sha) in enumerate(valid_batch_meta):
                    vec = feat_np[b_i]
                    vec_sha = sha256_bytes(vec.tobytes())
                    row_idx = len(embeddings_list)
                    embeddings_list.append(vec)

                    manifest_rows.append({
                        "collection": coll_name,
                        "token_id": item["token_id"],
                        "feature_row_idx": row_idx,
                        "image_path": item["image_path"],
                        "image_sha256": img_sha,
                        "model_revision": MODEL_REVISION,
                        "output_dim": 1024,
                        "dtype": "float32",
                        "feature_sha256": vec_sha,
                        "is_finite": True,
                        "status": "SUCCESS"
                    })
            except Exception as ex:
                for item, img_sha in valid_batch_meta:
                    fail_record = {
                        "collection": coll_name,
                        "token_id": item["token_id"],
                        "image_path": item["image_path"],
                        "stage": "model_forward",
                        "error": str(ex)
                    }
                    failed_tokens.append(fail_record)
                    coll_failed.append(fail_record)

            if (offset // BATCH_SIZE) % 50 == 0 or offset + BATCH_SIZE >= total_coll:
                processed_count = min(offset + BATCH_SIZE, total_coll)
                elapsed = time.perf_counter() - c_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"    Progress: {processed_count}/{total_coll} ({processed_count/total_coll*100:.1f}%) - {rate:.1f} img/s")

        # Save collection array
        c_elapsed = time.perf_counter() - c_start
        embeddings_matrix = np.stack(embeddings_list, axis=0) # (N, 1024) float32
        print(f"  {coll_name} matrix shape: {embeddings_matrix.shape}, dtype: {embeddings_matrix.dtype}")
        assert embeddings_matrix.shape == (total_coll, 1024), f"Expected ({total_coll}, 1024), got {embeddings_matrix.shape}"
        assert np.isfinite(embeddings_matrix).all(), f"Non-finite values in {coll_name} embedding matrix"

        feat_filename = f"{coll_name.lower()}_features_siglip2.npy"
        feat_path = OUTPUT_DIR / feat_filename
        np.save(feat_path, embeddings_matrix)
        feat_sha256 = sha256_file(feat_path)
        feat_bytes = feat_path.stat().st_size

        manifest_filename = f"feature_manifest_{coll_name.lower()}.jsonl"
        manifest_path = OUTPUT_DIR / manifest_filename
        with open(manifest_path, "w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest_sha256 = sha256_file(manifest_path)

        extraction_stats[coll_name] = {
            "total_tokens": total_coll,
            "successful_extractions": len(embeddings_list),
            "failed_extractions": len(coll_failed),
            "success_rate": len(embeddings_list) / total_coll,
            "feature_matrix_file": feat_filename,
            "feature_matrix_bytes": feat_bytes,
            "feature_matrix_sha256": feat_sha256,
            "feature_shape": list(embeddings_matrix.shape),
            "feature_dtype": str(embeddings_matrix.dtype),
            "manifest_file": manifest_filename,
            "manifest_sha256": manifest_sha256,
            "elapsed_seconds": c_elapsed,
            "images_per_second": total_coll / c_elapsed if c_elapsed > 0 else 0,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "mean_norm": float(np.mean(np.linalg.norm(embeddings_matrix, axis=1))),
            "min_norm": float(np.min(np.linalg.norm(embeddings_matrix, axis=1))),
            "max_norm": float(np.max(np.linalg.norm(embeddings_matrix, axis=1)))
        }
        print(f"  [OK] Saved {feat_filename} ({feat_bytes / (1024*1024):.1f} MB, SHA: {feat_sha256[:12]}...)")
        print(f"  [OK] Saved {manifest_filename} ({len(manifest_rows)} records)")

    # 6. Save Failures and Summary
    print("\n[5/5] Finalizing summary and manifests...")
    failed_tokens_path = OUTPUT_DIR / "failed_tokens.json"
    with open(failed_tokens_path, "w", encoding="utf-8") as f:
        json.dump(failed_tokens, f, indent=2, ensure_ascii=False)
    failed_sha256 = sha256_file(failed_tokens_path)
    print(f"  Total failures: {len(failed_tokens)}")

    total_elapsed = time.perf_counter() - total_start_time
    summary = {
        "status": "EXTRACTION_COMPLETE",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_seconds": total_elapsed,
        "cohort_source": {
            "master_tokens_path": str(MASTER_TOKENS_PATH.relative_to(ROOT)),
            "cohort_release_path": str(COHORT_RELEASE_PATH.relative_to(ROOT)),
            "expected_bayc_tokens": 9366,
            "expected_mayc_tokens": 12459,
            "total_expected": 21825
        },
        "model_registry": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "weights_safetensors_sha256": safetensors_sha256,
            "pooling": "MultiheadAttentionPoolingHead (pooler_output)",
            "output_dimension": 1024,
            "dtype": "float32"
        },
        "preprocessing_summary": {
            "color_space": "sRGB",
            "exif_orientation_applied": True,
            "straight_alpha_composite": "RGB(128,128,128)",
            "square_padded": True,
            "processor_size": [256, 256],
            "processor_rescale": "1/255",
            "processor_normalize": "mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]"
        },
        "collections": extraction_stats,
        "failed_tokens_count": len(failed_tokens),
        "failed_tokens_file": "failed_tokens.json",
        "failed_tokens_sha256": failed_sha256,
        "policy_adherence": {
            "raw_images_modified": False,
            "previous_artifacts_overwritten": False,
            "dedicated_output_directory": str(OUTPUT_DIR.relative_to(ROOT)),
            "no_pca_applied": True,
            "no_global_standardization": True,
            "no_price_labels_used": True,
            "no_final_test_targets_loaded": True,
            "native_dimension_retained": True
        },
        "output_manifest": [
            {
                "file": "spec_and_registry.json",
                "sha256": sha256_file(OUTPUT_DIR / "spec_and_registry.json")
            },
            {
                "file": "verification_small_batch.json",
                "sha256": sha256_file(OUTPUT_DIR / "verification_small_batch.json")
            },
            {
                "file": "bayc_features_siglip2.npy",
                "sha256": extraction_stats["BAYC"]["feature_matrix_sha256"]
            },
            {
                "file": "feature_manifest_bayc.jsonl",
                "sha256": extraction_stats["BAYC"]["manifest_sha256"]
            },
            {
                "file": "mayc_features_siglip2.npy",
                "sha256": extraction_stats["MAYC"]["feature_matrix_sha256"]
            },
            {
                "file": "feature_manifest_mayc.jsonl",
                "sha256": extraction_stats["MAYC"]["manifest_sha256"]
            },
            {
                "file": "failed_tokens.json",
                "sha256": failed_sha256
            }
        ]
    }

    summary_path = OUTPUT_DIR / "extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("  [OK] Saved extraction_summary.json")

    # Copy script to output directory for absolute reproducibility
    script_source = Path(__file__).read_text(encoding="utf-8")
    (OUTPUT_DIR / "extract_siglip2_features.py").write_text(script_source, encoding="utf-8")
    print("  [OK] Preserved script copy in output directory.")

    print("\n" + "=" * 70)
    print("SigLIP 2 Feature Extraction Completed Successfully!")
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"BAYC: {extraction_stats['BAYC']['successful_extractions']}/{extraction_stats['BAYC']['total_tokens']} success")
    print(f"MAYC: {extraction_stats['MAYC']['successful_extractions']}/{extraction_stats['MAYC']['total_tokens']} success")
    print(f"Failures: {len(failed_tokens)}")
    print("=" * 70)

if __name__ == "__main__":
    run()
