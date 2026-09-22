"""64-image, price-blind pilot for SAM, SDXL VAE, DreamSim, and AIMv2.

This file deliberately stops before any full-cohort extraction.  It loads only
the frozen image ledger and writes only to this extension directory.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "pilot_output"
OUT.mkdir(exist_ok=True)
HF_CACHE = OUT / "hf_cache"
HF_CACHE.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
MASTER = ROOT / "revision" / "image_validation_20260909" / "master_tokens_v1.jsonl"
SEED = 20260917
N_PER_COLLECTION = 32
DEVICE = "cuda"

MODEL_SPECS = {
    "sam": {"model_id": "facebook/sam-vit-huge", "batch_size": 1},
    "sdxl_vae": {"model_id": "madebyollin/sdxl-vae-fp16-fix", "batch_size": 1},
    "dreamsim": {"model_id": "dreamsim-0.2.1/default-ensemble", "batch_size": 1},
    "aimv2": {"model_id": "apple/aimv2-large-patch14-224", "batch_size": 1},
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value.astype(np.float32)).tobytes())


def gpu_status() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10, check=False).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def package_versions() -> dict[str, str]:
    names = ["torch", "torchvision", "transformers", "diffusers", "timm", "dreamsim",
             "open-clip-torch", "peft", "accelerate", "safetensors", "numpy", "Pillow"]
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def prepare_image(item: dict) -> tuple[Image.Image, str]:
    """Perform the shared full-frame processing fixed in EXTRACTION_SPEC_v1."""
    path = ROOT / item["image_path"]
    raw = path.read_bytes()
    observed_hash = sha256_bytes(raw)
    if observed_hash != item["image_sha256"]:
        raise RuntimeError(f"raw image hash mismatch for {item['collection']}:{item['token_id']}")
    with Image.open(path) as raw_img:
        image = ImageOps.exif_transpose(raw_img).convert("RGBA")
        width, height = image.size
        if width != height:
            edge = max(width, height)
            canvas = Image.new("RGBA", (edge, edge), (128, 128, 128, 0))
            canvas.paste(image, ((edge - width) // 2, (edge - height) // 2))
            image = canvas
        background = Image.new("RGBA", image.size, (128, 128, 128, 255))
        return Image.alpha_composite(background, image).convert("RGB"), observed_hash


def pilot_items() -> list[dict]:
    grouped = {"BAYC": [], "MAYC": []}
    for line in MASTER.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["collection"] in grouped and item["image_data_ready"]:
            grouped[item["collection"]].append(item)
    rng = np.random.default_rng(SEED)
    selected = []
    for collection, items in grouped.items():
        items = sorted(items, key=lambda x: int(x["token_id"]))
        indices = rng.choice(len(items), size=N_PER_COLLECTION, replace=False)
        selected.extend(items[i] for i in sorted(indices))
    return selected


def as_matrix(value, label: str) -> tuple[np.ndarray, str]:
    """Accept only a documented global output. Unknown AIM output structures fail loudly."""
    if isinstance(value, torch.Tensor) and value.ndim == 2:
        return value.detach().float().cpu().numpy(), "tensor"
    if hasattr(value, "pooler_output") and isinstance(value.pooler_output, torch.Tensor):
        return value.pooler_output.detach().float().cpu().numpy(), "pooler_output"
    raise TypeError(f"{label} returned no documented 2-D global feature; type={type(value)!r}, "
                    f"attributes={sorted(a for a in dir(value) if not a.startswith('_'))[:30]}")


def load_sam():
    from transformers import SamModel, SamProcessor
    spec = MODEL_SPECS["sam"]
    processor = SamProcessor.from_pretrained(spec["model_id"], cache_dir=HF_CACHE)
    model = SamModel.from_pretrained(spec["model_id"], cache_dir=HF_CACHE,
                                     torch_dtype=torch.float16).to(DEVICE).eval()

    def embed(image: Image.Image) -> tuple[np.ndarray, str]:
        inputs = processor(images=image, return_tensors="pt")
        pixels = inputs["pixel_values"].to(DEVICE, dtype=torch.float16)
        with torch.inference_mode():
            spatial = model.get_image_embeddings(pixels)
            vector = spatial.mean(dim=(-2, -1))
        return vector.float().cpu().numpy(), "image_embeddings global-average over spatial axes"
    return model, embed


def load_sdxl_vae():
    from diffusers import AutoencoderKL
    spec = MODEL_SPECS["sdxl_vae"]
    model = AutoencoderKL.from_pretrained(spec["model_id"], cache_dir=HF_CACHE,
                                           torch_dtype=torch.float16).to(DEVICE).eval()

    def embed(image: Image.Image) -> tuple[np.ndarray, str]:
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        pixels = torch.from_numpy(pixels).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        with torch.inference_mode():
            latent = model.encode(pixels).latent_dist.mean
        return latent.flatten(1).float().cpu().numpy(), "encoder posterior mean, flattened native latent tensor"
    return model, embed


def load_dreamsim():
    from dreamsim import dreamsim
    spec = MODEL_SPECS["dreamsim"]
    model, preprocess = dreamsim(pretrained=True, device=DEVICE, cache_dir=str(HF_CACHE / "dreamsim"),
                                 normalize_embeds=True, dreamsim_type="ensemble", use_patch_model=False)
    model.eval()

    def embed(image: Image.Image) -> tuple[np.ndarray, str]:
        pixels = preprocess(image)
        # dreamsim 0.2.1's official preprocess may already return a batched tensor.
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        if pixels.ndim != 4:
            raise RuntimeError(f"DreamSim preprocess returned shape {tuple(pixels.shape)}, expected BCHW")
        pixels = pixels.to(DEVICE)
        with torch.inference_mode():
            vector = model.embed(pixels)
        return as_matrix(vector, "DreamSim")
    return model, embed


def load_aimv2():
    import timm
    from timm.data import create_transform, resolve_model_data_config
    spec = MODEL_SPECS["aimv2"]
    # The submitted AIM implementation used this public timm entry point with
    # num_classes=0.  It returns the model's documented 1,024-d global head
    # feature rather than imposing a new pooling operation on patch tokens.
    model = timm.create_model("aimv2_large_patch14_224", pretrained=True,
                              num_classes=0).to(DEVICE).eval()
    transform = create_transform(**resolve_model_data_config(model), is_training=False)

    def embed(image: Image.Image) -> tuple[np.ndarray, str]:
        pixels = transform(image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            output = model(pixels)
        return as_matrix(output, "AIMv2 timm global head")
    return model, embed


LOADERS = {"sam": load_sam, "sdxl_vae": load_sdxl_vae, "dreamsim": load_dreamsim, "aimv2": load_aimv2}


def run_encoder(name: str, items: list[dict]) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = {"name": name, "spec": MODEL_SPECS[name], "status": "FAILED", "records": []}
    model = None
    try:
        model, embed = LOADERS[name]()
        all_vectors = []
        methods = set()
        for item in items:
            image, raw_hash = prepare_image(item)
            first, method = embed(image)
            second, second_method = embed(image)
            if method != second_method:
                raise RuntimeError(f"feature method changed within a pass: {method} vs {second_method}")
            if first.shape != second.shape:
                raise RuntimeError(f"repeat shape mismatch: {first.shape} vs {second.shape}")
            all_vectors.append(first[0])
            methods.add(method)
            result["records"].append({
                "collection": item["collection"], "token_id": item["token_id"],
                "image_sha256": raw_hash, "vector_sha256": sha256_array(first[0]),
                "repeat_max_abs_diff": float(np.max(np.abs(first - second))),
                "all_finite": bool(np.isfinite(first).all()), "dimension": int(first.shape[1]),
            })
        matrix = np.stack(all_vectors).astype(np.float32)
        dimensions = sorted({r["dimension"] for r in result["records"]})
        result.update({
            "status": "PASSED" if all(r["repeat_max_abs_diff"] == 0.0 and r["all_finite"] for r in result["records"])
            and len(dimensions) == 1 else "FAILED",
            "output_method": sorted(methods), "dimension": dimensions,
            "matrix_sha256": sha256_array(matrix), "matrix_shape": list(matrix.shape),
            "max_repeat_abs_diff": max(r["repeat_max_abs_diff"] for r in result["records"]),
            "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
        })
        np.save(OUT / f"pilot_{name}_features.npy", matrix)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        result["gpu_status_after"] = gpu_status()
        if model is not None:
            del model
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=list(LOADERS) + ["all"], default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    items = pilot_items()
    selected = [{k: item[k] for k in ["collection", "token_id", "image_path", "image_sha256"]} for item in items]
    (OUT / "pilot_selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    summary = {
        "purpose": "price-blind four-encoder pilot", "seed": SEED,
        "selected_images": len(items), "per_collection": N_PER_COLLECTION,
        "master_ledger": str(MASTER.relative_to(ROOT)), "gpu_status_before": gpu_status(),
        "python": sys.version, "platform": platform.platform(), "packages": package_versions(),
        "encoders": {},
    }
    selected_encoders = list(LOADERS) if args.encoder == "all" else [args.encoder]
    for name in selected_encoders:
        print(f"\n=== {name} ===", flush=True)
        summary["encoders"][name] = run_encoder(name, items)
        print(summary["encoders"][name]["status"], flush=True)
        (OUT / f"pilot_{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["all_requested_encoder_checks_passed"] = all(
        v["status"] == "PASSED" for v in summary["encoders"].values())
    if args.encoder == "all":
        (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"all_requested_encoder_checks_passed": summary["all_requested_encoder_checks_passed"]}))


if __name__ == "__main__":
    main()
