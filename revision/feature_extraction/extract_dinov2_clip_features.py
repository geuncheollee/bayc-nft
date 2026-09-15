"""
DINOv2 & CLIP Native Feature Extraction Pipeline for BAYC & MAYC
Strict adherence to:
- revision/Analysis Protocol 20260908.md
- revision/Revision Plan 20260908.md
- revision/image_validation_20260909/cohort_release_v1.json

Key Requirements:
1. Models:
   - DINOv2: facebook/dinov2-large, 1024d CLS pooling, unnormalized raw native embedding.
   - CLIP Native: openai/clip-vit-large-patch14, 768d native projected embedding (ZERO-PADDING ELIMINATED).
2. Protocol-compliant image preprocessing:
   - EXIF transpose -> RGBA mode -> Center-pad nonsquare with neutral gray RGB(128,128,128)
   - Straight-alpha composite over solid RGB(128,128,128,255) background -> RGB PIL image
   - Dual forward-pass in single I/O loop for maximum throughput.
3. Full provenance & verification:
   - SHA-256 verification of input raw image bytes against master_tokens_v1.jsonl.
   - Small-batch (64 samples) deterministic reproducibility test (Pass 1 vs Pass 2 diff == 0.0).
   - NaN/Inf assertion on all feature vectors.
   - Per-vector SHA-256 and per-file SHA-256 recorded in manifests and summary.
   - Absolute isolation: NO price data, NO final test targets, NO full-dataset PCA or global standardization.
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
from transformers import (
    AutoImageProcessor,
    Dinov2Model,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection
)

# ----------------------------------------------------------------------
# Paths and Environment Configuration
# ----------------------------------------------------------------------
ROOT = Path(os.environ.get("BAYC_NFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUTPUT_DIR = Path(os.environ.get("BAYC_DINOV2_CLIP_OUTPUT_DIR", ROOT / "revision" / "dinov2_clip_features_20260909")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_TOKENS_PATH = Path(os.environ.get("BAYC_MASTER_TOKENS", ROOT / "revision" / "image_validation_20260909" / "master_tokens_v1.jsonl")).resolve()
COHORT_RELEASE_PATH = Path(os.environ.get("BAYC_COHORT_RELEASE", ROOT / "revision" / "image_validation_20260909" / "cohort_release_v1.json")).resolve()

DINOV2_DIR = Path(os.environ.get(
    "DINOV2_CHECKPOINT_DIR",
    ROOT / "tmp" / "hf_cache" / "models--facebook--dinov2-large" / "snapshots" / "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
)).resolve()
CLIP_DIR = Path(os.environ.get(
    "CLIP_CHECKPOINT_DIR",
    ROOT / "tmp" / "hf_cache" / "models--openai--clip-vit-large-patch14" / "snapshots" / "32bd64288804d66eefd0ccbe215aa642df71cc41",
)).resolve()

DINOV2_MODEL_ID = "facebook/dinov2-large"
DINOV2_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"

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
# Official Image Preprocessing Pipeline (Protocol-Compliant)
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
# Main Execution
# ----------------------------------------------------------------------
def run():
    total_start_time = time.perf_counter()
    print("=" * 80)
    print("DINOv2 & CLIP Native Dual Feature Extraction: Specification, Verification & Full Run")
    print(f"Start time (UTC): {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("=" * 80)

    # 1. Environment & Checkpoints
    print("[1/5] Verifying environment & model weights...")
    assert torch.cuda.is_available(), "CUDA GPU is required!"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  DINOv2 snapshot: {DINOV2_DIR}")
    print(f"  CLIP snapshot: {CLIP_DIR}")

    dino_safetensors = DINOV2_DIR / "model.safetensors"
    clip_safetensors = CLIP_DIR / "model.safetensors"
    assert dino_safetensors.exists(), f"DINOv2 safetensors missing: {dino_safetensors}"
    assert clip_safetensors.exists(), f"CLIP safetensors missing: {clip_safetensors}"

    dino_sha256 = sha256_file(dino_safetensors)
    clip_sha256 = sha256_file(clip_safetensors)
    print(f"  DINOv2 model.safetensors SHA-256: {dino_sha256}")
    print(f"  CLIP model.safetensors SHA-256:   {clip_sha256}")

    # 2. Specification & Registry
    spec_and_registry = {
        "pipeline_version": "1.0.0",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {
            "dinov2": {
                "model_id": DINOV2_MODEL_ID,
                "model_revision": DINOV2_REVISION,
                "checkpoint_safetensors_sha256": dino_sha256,
                "checkpoint_path": str(DINOV2_DIR),
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
                "dtype": "float32",
                "preprocessing": {
                    "processor_class": "BitImageProcessor",
                    "image_size": [224, 224],
                    "resample": "bicubic (value=3)",
                    "rescale_factor": 1 / 255.0,
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225]
                }
            },
            "clip_native": {
                "model_id": CLIP_MODEL_ID,
                "model_revision": CLIP_REVISION,
                "checkpoint_safetensors_sha256": clip_sha256,
                "checkpoint_path": str(CLIP_DIR),
                "architecture": {
                    "model_type": "clip_vision_model",
                    "hidden_size": 1024,
                    "projection_dim": 768,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 16,
                    "image_size": 224,
                    "patch_size": 14
                },
                "pooling": {
                    "method": "VisualProjection",
                    "attribute": "image_embeds",
                    "description": "CLIP Visual Projection to 768 native dimensions (ZERO-PADDING REMOVED)"
                },
                "output_dimension": 768,
                "dtype": "float32",
                "preprocessing": {
                    "processor_class": "CLIPImageProcessor",
                    "image_size": [224, 224],
                    "resample": "bicubic (value=3)",
                    "rescale_factor": 1 / 255.0,
                    "image_mean": [0.48145466, 0.4578275, 0.40821073],
                    "image_std": [0.26862954, 0.26130258, 0.27577711]
                }
            }
        },
        "preprocessing_pipeline": {
            "color_space": "sRGB",
            "exif_transpose": True,
            "mode_normalization": "RGBA conversion before compositing",
            "nonsquare_handling": "center-pad with RGB(128,128,128) if width != height (no cropping)",
            "alpha_compositing": "straight-alpha composite over neutral gray RGB(128,128,128,255) background"
        },
        "feature_treatment_policy": {
            "dimensionality_reduction": "NONE - Raw native dimension strictly preserved; PCA forbidden on full dataset",
            "global_standardization": "NONE - Learned standardization on full dataset strictly forbidden to prevent leakage",
            "normalization": "NONE - Raw model outputs preserved without unverified transformations",
            "zero_padding": "NONE - Native 768d used for CLIP; 256 trailing zeros in legacy manuscript eliminated",
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

    # 3. Model Loading
    print("[2/5] Loading processors and models onto GPU...")
    dino_proc = AutoImageProcessor.from_pretrained(str(DINOV2_DIR), local_files_only=True, use_fast=False)
    dino_model = Dinov2Model.from_pretrained(str(DINOV2_DIR), local_files_only=True, torch_dtype=torch.float32)
    dino_model = dino_model.eval().to("cuda")

    clip_proc = CLIPImageProcessor.from_pretrained(str(CLIP_DIR), local_files_only=True)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(str(CLIP_DIR), local_files_only=True, torch_dtype=torch.float32)
    clip_model = clip_model.eval().to("cuda")

    torch.cuda.synchronize()
    print("  [OK] Both DINOv2 and CLIP models loaded to CUDA successfully.")

    # 4. Small-Batch Verification
    print("[3/5] Performing small-batch verification (BAYC & MAYC, sample size=64)...")
    tokens_by_coll = {"BAYC": [], "MAYC": []}
    with open(MASTER_TOKENS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            tokens_by_coll[item["collection"]].append(item)

    print(f"  Loaded master tokens: BAYC={len(tokens_by_coll['BAYC'])}, MAYC={len(tokens_by_coll['MAYC'])}")
    assert len(tokens_by_coll["BAYC"]) == 9366, f"Expected 9,366 BAYC tokens, got {len(tokens_by_coll['BAYC'])}"
    assert len(tokens_by_coll["MAYC"]) == 12459, f"Expected 12,459 MAYC tokens, got {len(tokens_by_coll['MAYC'])}"

    rng = np.random.default_rng(42)
    bayc_sample_indices = rng.choice(len(tokens_by_coll["BAYC"]), size=32, replace=False)
    mayc_sample_indices = rng.choice(len(tokens_by_coll["MAYC"]), size=32, replace=False)
    sample_tokens = [tokens_by_coll["BAYC"][i] for i in bayc_sample_indices] + [tokens_by_coll["MAYC"][i] for i in mayc_sample_indices]

    verification_results = {}
    torch.cuda.reset_peak_memory_stats()
    v_start = time.perf_counter()

    for pass_num in [1, 2]:
        dino_feats_pass = []
        clip_feats_pass = []
        for idx in range(0, len(sample_tokens), BATCH_SIZE):
            batch_slice = sample_tokens[idx : idx + BATCH_SIZE]
            imgs = []
            for item in batch_slice:
                p = ROOT / Path(item["image_path"])
                img, sha = preprocess_image_to_pil(p)
                assert sha == item["image_sha256"], f"SHA mismatch on {item['token_id']}"
                imgs.append(img)

            # DINOv2
            dino_in = dino_proc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda")
            with torch.inference_mode():
                d_out = dino_model(pixel_values=dino_in)
                d_feat = d_out.pooler_output
                assert torch.isfinite(d_feat).all()
                assert d_feat.shape[1] == 1024
                dino_feats_pass.append(d_feat.cpu().numpy())

            # CLIP Native
            clip_in = clip_proc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda")
            with torch.inference_mode():
                c_out = clip_model(pixel_values=clip_in)
                c_feat = c_out.image_embeds
                assert torch.isfinite(c_feat).all()
                assert c_feat.shape[1] == 768
                clip_feats_pass.append(c_feat.cpu().numpy())

        if pass_num == 1:
            dino_p1 = np.concatenate(dino_feats_pass, axis=0)
            clip_p1 = np.concatenate(clip_feats_pass, axis=0)
        else:
            dino_p2 = np.concatenate(dino_feats_pass, axis=0)
            clip_p2 = np.concatenate(clip_feats_pass, axis=0)

    dino_diff = float(np.max(np.abs(dino_p1 - dino_p2)))
    clip_diff = float(np.max(np.abs(clip_p1 - clip_p2)))
    v_elapsed = time.perf_counter() - v_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    verification_results = {
        "sample_size": len(sample_tokens),
        "dinov2": {
            "output_dim": 1024,
            "all_finite": bool(np.isfinite(dino_p1).all()),
            "reproducibility_max_abs_diff": dino_diff,
            "verified": bool(dino_diff == 0.0 and np.isfinite(dino_p1).all() and dino_p1.shape[1] == 1024),
            "mean_norm": float(np.mean(np.linalg.norm(dino_p1, axis=1)))
        },
        "clip_native": {
            "output_dim": 768,
            "all_finite": bool(np.isfinite(clip_p1).all()),
            "reproducibility_max_abs_diff": clip_diff,
            "verified": bool(clip_diff == 0.0 and np.isfinite(clip_p1).all() and clip_p1.shape[1] == 768),
            "mean_norm": float(np.mean(np.linalg.norm(clip_p1, axis=1)))
        },
        "performance": {
            "verification_seconds": v_elapsed,
            "seconds_per_image": v_elapsed / (len(sample_tokens) * 2),
            "peak_vram_mb": peak_vram_mb
        }
    }

    print(f"  DINOv2 pass diff: {dino_diff:.2e} | CLIP pass diff: {clip_diff:.2e}")
    print(f"  Peak VRAM: {peak_vram_mb:.1f} MB | Elapsed: {v_elapsed:.2f}s")
    assert verification_results["dinov2"]["verified"], "DINOv2 verification failed!"
    assert verification_results["clip_native"]["verified"], "CLIP verification failed!"

    with open(OUTPUT_DIR / "verification_small_batch.json", "w", encoding="utf-8") as f:
        json.dump(verification_results, f, indent=2, ensure_ascii=False)
    print("  [OK] Saved verification_small_batch.json")

    # 5. Full Extraction
    print("[4/5] Extracting full raw embeddings for BAYC (9,366) and MAYC (12,459)...")
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

        dino_matrix = np.zeros((total_coll, 1024), dtype=np.float32)
        clip_matrix = np.zeros((total_coll, 768), dtype=np.float32)

        manifest_dino = []
        manifest_clip = []
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

            # Batch forward pass: DINOv2
            dino_in = dino_proc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda")
            with torch.inference_mode():
                d_out = dino_model(pixel_values=dino_in)
                d_feat = d_out.pooler_output
                assert torch.isfinite(d_feat).all(), f"Non-finite values in DINOv2 at batch {offset}"
                d_feat_np = d_feat.cpu().numpy().astype(np.float32)

            # Batch forward pass: CLIP Native
            clip_in = clip_proc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda")
            with torch.inference_mode():
                c_out = clip_model(pixel_values=clip_in)
                c_feat = c_out.image_embeds
                assert torch.isfinite(c_feat).all(), f"Non-finite values in CLIP at batch {offset}"
                c_feat_np = c_feat.cpu().numpy().astype(np.float32)

            # Populate arrays & manifests
            for b_i, (item, img_sha) in enumerate(valid_meta):
                row_idx = offset + b_i
                d_vec = d_feat_np[b_i]
                c_vec = c_feat_np[b_i]

                dino_matrix[row_idx] = d_vec
                clip_matrix[row_idx] = c_vec

                d_sha = sha256_bytes(d_vec.tobytes())
                c_sha = sha256_bytes(c_vec.tobytes())

                manifest_dino.append({
                    "collection": coll_name,
                    "token_id": item["token_id"],
                    "feature_row_idx": row_idx,
                    "image_path": item["image_path"],
                    "image_sha256": img_sha,
                    "model_id": DINOV2_MODEL_ID,
                    "model_revision": DINOV2_REVISION,
                    "output_dim": 1024,
                    "dtype": "float32",
                    "feature_sha256": d_sha
                })

                manifest_clip.append({
                    "collection": coll_name,
                    "token_id": item["token_id"],
                    "feature_row_idx": row_idx,
                    "image_path": item["image_path"],
                    "image_sha256": img_sha,
                    "model_id": CLIP_MODEL_ID,
                    "model_revision": CLIP_REVISION,
                    "output_dim": 768,
                    "dtype": "float32",
                    "feature_sha256": c_sha
                })

            processed_count += len(valid_meta)
            if (offset // BATCH_SIZE) % 50 == 0 or processed_count == total_coll:
                elapsed = time.perf_counter() - coll_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"    [{coll_name}] {processed_count}/{total_coll} processed ({rate:.1f} img/s)...", flush=True)

        # Save numpy files
        dino_out_path = OUTPUT_DIR / f"{coll_name.lower()}_features_dinov2.npy"
        clip_out_path = OUTPUT_DIR / f"{coll_name.lower()}_features_clip_native.npy"
        np.save(dino_out_path, dino_matrix)
        np.save(clip_out_path, clip_matrix)

        dino_npy_sha = sha256_file(dino_out_path)
        clip_npy_sha = sha256_file(clip_out_path)

        # Save manifest jsonl
        dino_man_path = OUTPUT_DIR / f"manifest_{coll_name.lower()}_dinov2.jsonl"
        clip_man_path = OUTPUT_DIR / f"manifest_{coll_name.lower()}_clip_native.jsonl"
        with open(dino_man_path, "w", encoding="utf-8") as f:
            for row in manifest_dino:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(clip_man_path, "w", encoding="utf-8") as f:
            for row in manifest_clip:
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
            "dinov2": {
                "file": dino_out_path.name,
                "shape": list(dino_matrix.shape),
                "dtype": str(dino_matrix.dtype),
                "sha256": dino_npy_sha,
                "all_finite": bool(np.isfinite(dino_matrix).all())
            },
            "clip_native": {
                "file": clip_out_path.name,
                "shape": list(clip_matrix.shape),
                "dtype": str(clip_matrix.dtype),
                "sha256": clip_npy_sha,
                "all_finite": bool(np.isfinite(clip_matrix).all())
            }
        }
        print(f"  [OK] {coll_name} complete in {coll_elapsed:.1f}s ({processed_count / coll_elapsed:.1f} img/s).")

    # 6. Final Summary and Failure log
    print("[5/5] Finalizing summary and manifests...")
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
    print("DUAL EXTRACTION COMPLETED SUCCESSFULLY!")
    print(f"Total tokens processed: {extraction_summary['total_tokens_processed']} / 21,825")
    print(f"Failures: {len(failed_tokens)}")
    print(f"Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    run()
