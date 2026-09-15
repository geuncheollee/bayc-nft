# Reproducibility guide

## Level 1 — self-contained public verification

Install the pinned modelling and rendering dependencies and run:

```powershell
python tests\test_public_release.py
```

This verifies JSON validity, completed governance state, cross-file provenance
bindings, published primary metrics and confidence intervals, Python source
compilation, the exact 1-wei classifier, model-construction functions, and the
SHA-256 repository manifest. No private input is opened.

## Level 2 — regenerate published Figures 3 and 4

```powershell
python figures\render_published_figures.py --output-dir reproduced_figures
```

The renderer reads only `results/test_metrics_summary.json` and
`results/confidence_intervals_summary.json`. It creates PNG and PDF copies and
does not require transaction rows or model binaries.

## Level 3 — feature reconstruction

The frozen scripts in `revision/feature_extraction/` require a master token
index whose `image_path` values resolve below a chosen study root. Set:

```powershell
$env:BAYC_NFT_ROOT = 'D:\path\to\authorized\study_root'
$env:BAYC_MASTER_TOKENS = 'D:\path\to\master_tokens_v1.jsonl'
$env:BAYC_COHORT_RELEASE = 'D:\path\to\cohort_release_v1.json'
$env:DINOV2_CHECKPOINT_DIR = 'D:\path\to\facebook-dinov2-large-snapshot'
$env:CLIP_CHECKPOINT_DIR = 'D:\path\to\openai-clip-vit-large-patch14-snapshot'
$env:SIGLIP2_CHECKPOINT_DIR = 'D:\path\to\google-siglip2-large-patch16-256-snapshot'
```

The exact model repository revisions and weight hashes are in the public
execution specification. Install optional GPU dependencies from
`requirements-feature-extraction.txt` and run the desired extraction script.

## Level 4 — complete model refit and evaluation

This level requires the structured private input bundle described in
`DATA_ACCESS.md`. The production engine is deliberately fail-closed: it will
not run active stages without matching manifests, approved scope records, and
the external anchor. The public results allow verification of what was run;
they do not bypass the original custody design.

## Certified environments

- Feature extraction: Python 3.12.13, PyTorch 2.7.1+cu118; SigLIP 2 also
  records Transformers 4.56.2.
- Final modelling: Python 3.14.3, NumPy 2.4.2, SciPy 1.17.1,
  scikit-learn 1.8.0, Joblib 1.5.3, threadpoolctl 3.6.0.
- Universal random seed: `20260908`.

