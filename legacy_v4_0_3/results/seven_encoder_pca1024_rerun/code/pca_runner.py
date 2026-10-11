"""Authorized fold-local 1024-component amendment; native artifacts stay read-only.

Only the current fresh rerun's unaffected development checkpoints are inherited.
SDXL/DreamSim fits are never inherited. All final tuning/refits/evaluation are new.
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
PREVIOUS = ROOT / "results/seven_encoder_full_rerun"
sys.path.insert(0, str(PREVIOUS / "code"))
import audit_inputs as au
au.OUT = OUT
import canonical_runner as cr
import evaluate_frozen as ef

REDUCED = {"SDXL_VAE", "DreamSim"}
UNCHANGED = ["metadata", "DINOv2", "CLIP", "SigLIP2", "SAM", "AIM"]
N_COMPONENTS = 1024
PCA_SETTINGS = dict(svd_solver="randomized", n_oversamples=10, iterated_power=7,
                    power_iteration_normalizer="QR", whiten=False, copy=False,
                    random_state=20260908)
ORIGINAL_LOAD = cr.load_development
ORIGINAL_TRANSFORM = cr.vision.transform
OriginalRunner = cr.FreshRunner
ORIGINAL_CSVSAVE = au.csvsave


def dimensional_csvsave(path, rows):
    for row in rows:
        encoder = row.get("encoder")
        if encoder in au.ENCODERS:
            native = row.get("native_dimension", row.get("native_embedding_dimension"))
            row["analysis_image_dimension"] = N_COMPONENTS if encoder in REDUCED else native
            row["image_projection"] = "training-fold-only PCA" if encoder in REDUCED else "none"
    return ORIGINAL_CSVSAVE(path, rows)


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "run_manifest.json").exists():
        original_save = au.save
        def augmented_save(path, obj):
            if Path(path).name == "run_manifest.json":
                obj.update(dimensionality_amendment=dict(
                    authorized="Native archives preserved; SDXL and DreamSim analysis input 1024",
                    components=N_COMPONENTS, pca_settings=PCA_SETTINGS,
                    fitting_unit="one image per distinct token in the chronological training portion",
                    fitting_dtype="float64", scaling="unchanged canonical scaling after PCA",
                    no_evaluation_data_for_pca_fit=True,
                    previous_results_directory=str(PREVIOUS.relative_to(ROOT))))
                obj["canonical_scripts"].extend(
                    dict(path=str(p.relative_to(ROOT)), sha256=au.sha(p))
                    for p in sorted(HERE.glob("*.py")))
                obj["resource_estimate"]["analysis_image_dimension"] = 1024
                obj["resource_estimate"]["note"] = "Native archive audit; regression uses fold-local 1024 PCA for SDXL/DreamSim."
            original_save(path, obj)
        au.save = augmented_save
        try:
            au.main()
        finally:
            au.save = original_save
    provenance = OUT / "inherited_development_checkpoints.json"
    if not provenance.exists():
        old_config = au.read(PREVIOUS / "run_configuration.json")
        assert old_config["grids"] == cr.GRIDS and old_config["weights"] == cr.WEIGHTS
        assert old_config["seed"] == 20260908 and old_config["threads"] == 3
        for record in old_config["source_files"] + old_config["input_files"]:
            assert au.sha(ROOT / record["path"]) == record["sha256"]
        records = []
        for stage in ("checkpoints", "development"):
            for collection in au.COLLECTIONS:
                for encoder in UNCHANGED:
                    folder = PREVIOUS / stage / "original" / collection / encoder
                    for source in sorted(folder.rglob("*")):
                        if not source.is_file():
                            continue
                        relative = source.relative_to(PREVIOUS)
                        # Never inherit final tuning, refits, freezes, evaluation, or any affected encoder.
                        if stage == "checkpoints" and relative.parts[4] != "development":
                            continue
                        destination = OUT / relative
                        digest = au.sha(source)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if not destination.exists():
                            shutil.copy2(source, destination)
                        assert au.sha(destination) == digest
                        records.append(dict(path=str(relative), sha256=digest,
                                            source=str(source.relative_to(ROOT))))
        au.save(provenance, dict(created_utc=au.now(),
            source_configuration_sha256=au.sha(PREVIOUS / "run_configuration.json"),
            source="Current fresh seven-encoder rerun, NOT v2.8 certified outputs",
            excluded_encoders=sorted(REDUCED), inherited_stages=["development"], files=records))
        print(json.dumps(dict(stage="unaffected_development_preserved", files=len(records))), flush=True)
    else:
        for record in au.read(provenance)["files"]:
            assert au.sha(OUT / record["path"]) == record["sha256"]


def load_development(collection, encoder=None, sample="original"):
    if encoder not in REDUCED:
        return ORIGINAL_LOAD(collection, encoder, sample)
    data = ORIGINAL_LOAD(collection, None, sample)
    info = au.read(OUT / "embedding_registry.json")[encoder][collection]
    ids = np.asarray(info["token_ids"], dtype=np.int64)
    mapping = {int(t): i for i, t in enumerate(ids)}
    metadata = {(r["collection"], int(r["token_id"])): r
                for r in au.lines(cr.base.OLD / "metadata_normalized.jsonl")}
    data.update(image=np.load(ROOT / info["matrix"], mmap_mode="r", allow_pickle=False),
        feature_tokens=ids, X=np.asarray([[metadata[collection, int(t)][c]
        for c in data["columns"]] for t in ids], dtype=object),
        index=np.asarray([mapping[int(t)] for t in data["tokens"]], dtype=np.int64),
        _encoder=encoder, _collection=collection, _sample=sample,
        _native_sha256=info["matrix_sha256"])
    assert data["image"].shape[1] == info["dimensions"]
    assert np.array_equal(ids[data["index"]], data["tokens"])
    return data


def fit_projection(data, train, origin, folder, components=N_COMPONENTS):
    """Persist token-aligned fold-specific embeddings and their fitted PCA state."""
    train = np.asarray(train, dtype=int)
    cr.check_training(data, train, np.asarray([], dtype=int), origin)
    unique = np.unique(data["index"][train])
    assert min(len(unique), data["image"].shape[1]) >= components
    identity = dict(components=components, settings=PCA_SETTINGS,
        native_sha256=data["_native_sha256"], native_dimension=data["image"].shape[1],
        token_order_sha256=cr.array_sha(data["feature_tokens"]),
        training_row_sha256=cr.array_sha(data["rowids"][train]),
        training_token_sha256=cr.array_sha(data["feature_tokens"][unique]),
        training_unique_tokens=len(unique), training_transactions=len(train),
        training_max_time=str(data["times"][train].max()), origin=origin)
    folder.mkdir(parents=True, exist_ok=True)
    marker, model_path, matrix_path = folder / "manifest.json", folder / "pca.joblib", folder / "embeddings_1024.npy"
    if marker.exists():
        manifest = au.read(marker)
        assert manifest["identity"] == identity
        assert au.sha(model_path) == manifest["pca_sha256"]
        assert au.sha(matrix_path) == manifest["embeddings_sha256"]
        return joblib.load(model_path), np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    # A previous interrupted write is never mistaken for a certified projection.
    if model_path.exists() or matrix_path.exists():
        return fit_projection(data, train, origin, folder / ("retry_" + str(time.time_ns())), components)
    print(json.dumps(dict(stage="pca_training_started", origin=origin,
        collection=data.get("_collection"), encoder=data.get("_encoder"),
        native_dimension=data["image"].shape[1], components=components,
        training_unique_tokens=len(unique), available_RAM_GiB=psutil.virtual_memory().available/2**30)), flush=True)
    started = time.perf_counter()
    # Private float64 training-image copy: PCA(copy=False) cannot mutate native files.
    fitting = np.array(data["image"][unique], dtype=np.float64, order="C", copy=True)
    assert np.isfinite(fitting).all()
    model = PCA(n_components=components, **PCA_SETTINGS).fit(fitting)
    del fitting
    gc.collect()
    assert model.n_samples_ == len(unique)
    projected = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float64,
                                          shape=(len(data["image"]), components))
    for begin in range(0, len(projected), 64):
        block = model.transform(np.asarray(data["image"][begin:begin+64], dtype=np.float64))
        assert np.isfinite(block).all()
        projected[begin:begin+len(block)] = block
    projected.flush()
    del projected
    joblib.dump(model, model_path, compress=3)
    manifest = dict(identity=identity, created_utc=au.now(), seconds=time.perf_counter()-started,
        dtype="float64", shape=[len(data["image"]), components],
        explained_variance_ratio_sum=float(model.explained_variance_ratio_.sum()),
        pca_sha256=au.sha(model_path), embeddings_sha256=au.sha(matrix_path),
        fitted_only_on_training_images=True, transaction_row_alignment_unchanged=True)
    au.save(marker, manifest)
    print(json.dumps(dict(stage="pca_training_completed", origin=origin,
        collection=data.get("_collection"), encoder=data.get("_encoder"),
        seconds=manifest["seconds"], shape=manifest["shape"])), flush=True)
    return model, np.load(matrix_path, mmap_mode="r", allow_pickle=False)


def transform(state, X, image):
    pca = state.get("image_pca")
    if pca is None:
        return ORIGINAL_TRANSFORM(state, X, image)
    chunks = []
    for begin in range(0, len(image), 64):
        reduced = pca.transform(np.asarray(image[begin:begin+64], dtype=np.float64))
        chunks.append(ORIGINAL_TRANSFORM(state, X[begin:begin+64], reduced))
    return np.concatenate(chunks)


class PCARunner(OriginalRunner):
    def prepare_checked(self, data, train, valid, mode, origin):
        if data.get("_encoder") not in REDUCED or mode == "metadata":
            return super().prepare_checked(data, train, valid, mode, origin)
        cr.check_training(data, train, valid, origin)
        self.memory_gate()
        folder = OUT / "reduced_embeddings" / data["_sample"] / data["_collection"] / data["_encoder"] / origin
        pca, projected = fit_projection(data, train, origin, folder)
        reduced = dict(data, image=projected)
        p = super().prepare_checked(reduced, train, valid, mode, origin)
        p["state"].update(image_pca=pca, native_dimension=data["image"].shape[1],
                          analysis_image_dimension=N_COMPONENTS)
        self.log(dict(stage="fold_local_pca_verified", collection=data["_collection"],
            encoder=data["_encoder"], mode=mode, origin=origin,
            training_row_sha256=cr.array_sha(data["rowids"][train]),
            native_dimension=data["image"].shape[1], analysis_image_dimension=N_COMPONENTS))
        return p


def install():
    cr.load_development = load_development
    cr.vision.transform = transform
    cr.FreshRunner = PCARunner
    cr.csvsave = dimensional_csvsave
    ef.csvsave = dimensional_csvsave


def main():
    initialize()
    certificates = sorted(OUT.glob("pca_preflight_tests_*.json"), key=lambda p: p.stat().st_mtime)
    certificate = au.read(certificates[-1])
    assert certificate["passed"] and certificate["runner_sha256"] == au.sha(Path(__file__))
    install()
    cr.main()


if __name__ == "__main__":
    main()
