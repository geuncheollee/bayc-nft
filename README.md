# BAYC/MAYC revision reproducibility code

Release **v4.0.1** supports *Image embeddings add limited predictive value to metadata in two NFT collections*. It is a documentation-only correction to v4.0.0 that publicly records the DuneSQL transaction query in `DATA_ACCESS.md`; no code, results or scientific conclusions changed. The version-specific Zenodo DOI is linked from the GitHub release page after archival. It supersedes the active v4.0.0 documentation, not that release's historical archive.

## What this version contains

- Eight TF-IDF metadata regression families per collection, with the originally executed one-hot comparisons retained and clearly identified.
- Seven image encoders by eight regression families for image-only and corrected TF-IDF early fusion.
- Conditional late fusion: a development-selected TF-IDF branch, 56 image branches and 11 image weights per collection (1,232 weighted combinations across both collections).
- Paired token-cluster and 14-day-block uncertainty, zero-LRP comparisons, evaluation-order audit, transaction-quality sensitivities and strict refits, image residual learning, and development-quarter checks.
- Executed source scripts and their shared dependencies, full available grids/freeze records, aggregate results, input hashes, and a public verification suite.

## Verify immediately, without private data

```sh
python tests/test_public_release.py
python tools/export_tables.py --output-dir reproduced_tables
```

Both commands use the Python standard library. The first checks file integrity, source syntax, development-selected metrics, search counts, uncertainty summaries and exclusion boundaries. The second exports human-readable tables from the published aggregates. Neither retrains models or independently reconstructs outcomes from raw transactions.

Optional synthetic method tests require the modelling environment:

```sh
python -m pip install -r requirements-models.txt
python tests/test_synthetic_methods.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for scientific execution order and [DATA_ACCESS.md](DATA_ACCESS.md) for inputs not redistributed. Original relative directory layouts are retained to preserve executed scripts and source hashes; historical filenames containing `nine_regressors` do not imply nine executed families in the reported final comparisons.

## Interpretation and limits

Selection uses 2024 Q2-Q4 forward validation, final fitting uses 2022-2024, and evaluation is 2025-01-01 through 2026-04-13. These are retrospective chronological analyses; repeated inspection during revision is disclosed. Bootstrap intervals condition on fitted models and do not quantify the entire model-selection process. Small or uncertain image gains are retained, including negative residual-correction results. The reported comparison does not establish general superiority of image fusion.

The code is MIT-licensed, retaining the repository's existing licence. Third-party input data, model weights and images are not relicensed. Confidential peer-review reports, response letters, manuscript drafts, credentials, and local approval records are not published here.
