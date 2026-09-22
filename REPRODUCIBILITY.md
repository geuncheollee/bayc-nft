# Reproduction levels and execution order

## Level 1: public aggregate verification

Run `python tests/test_public_release.py` and `python tools/export_tables.py --output-dir reproduced_tables`. No network, scientific libraries or private data are required. Inspect the original full candidate CSVs under the run directories indexed in `provenance/run_index.json`. Representatives are chosen by development RMSE, never by the lowest evaluation RMSE. Metadata-only OLS has lower evaluation point estimates than the selected fusion representatives in both collections; this does not justify post-hoc reselection.

## Level 2: synthetic implementation checks

Install `requirements-models.txt` and run `python tests/test_synthetic_methods.py`. These checks cover past-only references, target transform round trips, training-only TF-IDF and transaction-count aggregation. They are not full scientific replication.

## Level 3: full-data rerun (requires author inputs)

Make a **separate working copy**. Preserve this immutable archive. Published aggregates occupy paths used by the original runners: move the corresponding aggregate output directories aside in the working copy before a fresh run, retaining their manifests for comparison. Never erase the source archive or original study data. Inspect each script's `OUT` and resume behavior before execution; output paths are historical and not uniformly configurable. The late and supplemental directories contain code as well as outputs: retain `code/` when preparing fresh outputs.

1. Prepare authorized raw exports, traits and images. `tmp/revision_review/revision_data_audit.py` performs the input/eligibility audit; `validate_revision_images.py` and `freeze_image_ready_cohort.py` build image-ready ledgers. The `tmp/` name is an original source location, not a disposable public dependency.
2. Build past-only transaction targets with `revision/code/build_past_only_targets.py`. Follow its CLI and required upstream manifests. Existing protocol/amendment hashes are part of the checks.
3. Extract DINOv2 full-frame, CLIP native and SigLIP 2 using the scripts in their named `revision/*features_20260909` directories. SAM, SDXL-VAE, DreamSim and AIMv2 use `revision/four_encoder_extension_20260917/run_full_extraction.py`; `run_pilot.py` supplies shared extraction definitions. The three original extraction scripts expect local checkpoint/cache paths; populate those documented paths with the stated model revisions. A cached file is not distributed with this release.
4. Supply/reconstruct the audited embedding registries and row manifests listed in `provenance/required_inputs.json`. Fold-fitted PCA/scaling occurs in the regression runners, not on the full evaluation cohort.
5. Metadata entry point: `results/metadata_asinh_transaction_2022_2024_nine_regressors_20260920/code/run_experiment.py`. The executed final family list excludes RandomForest and writes the **eight**-regressor `_v1` result directory. Shared modules retaining `nine` in their names must not be run as substitutes.
6. Image entry point: `results/image_asinh_transaction_2022_2024_eight_regressors_20260920/code/run_image_experiment.py`.
7. Early entry point: `results/early_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260921/code/run_early_fusion_tfidf_experiment.py`.
8. Late entry point: `results/late_fusion_tfidf_asinh_transaction_2022_2024_eight_regressors_20260922_v2/code/run_late_fusion_experiment.py`. It uses the pre-2025 branch results and validation predictions before loading evaluation predictions.
9. Run `run_fusion_uncertainty.py` and `run_vs_zero_uncertainty.py` in their named result directories. Row-aligned predictions are necessary and are not public inputs.
10. Supplemental order in `results/reviewer_priority4_20260922_v1/code`: stage1 audit; stage2 cleaning, refit and refit uncertainty; stage3 residual, strict residual and residual uncertainty; stage4 rolling and forward selection; `verify_results.py`.

The release preserves executed scientific code rather than replacing it with a newly redesigned engine. A complete raw-input refit was **not** performed during packaging. Some historical helpers expose broader experiments; only the entry points above and the recorded final run indices support the reported revision. All library/extraction defaults should be read from source together with the saved configurations.

## Frozen-code integrity

`provenance/source_files.json` links byte-identical public copies to their local source hashes. `MANIFEST_SHA256.json` covers release files except itself. Code is inspectable even where historical input acquisition is not independently reconstructible. Records with `evaluation_labels_loaded_before_freeze: false` indicate labels were not loaded before freezing; the supplemental audit's confusingly named `run_level_freeze_flag` stores this negative flag, not a failed-freeze indicator.
