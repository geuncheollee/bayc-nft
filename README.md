# NFT valuation reproducibility code

[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8555--7064-a6ce39.svg)](https://orcid.org/0000-0002-8555-7064)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository is the public computational record for the revised manuscript
**Comparative analysis of image embedding models integrated with metadata for
non-fungible token valuation**.

Release version: `v3.3.6.4`  
Zenodo DOI: pending the first GitHub Release archive

## What can be verified immediately

- the final model grids, solver settings, seeds, temporal design, and approved
  Option A scope recorded in the completed public execution specification;
- the frozen modelling engine and supporting target, metadata, and
  feature-construction source code;
- the published selected configurations, refit record, out-of-time metrics,
  and paired-bootstrap confidence intervals;
- Figures 3 and 4 regenerated from the published summary JSON files;
- repository file integrity through a SHA-256 manifest;
- a public-only validation suite that does not require restricted transaction
  rows, NFT images, embeddings, fitted models, or sealed test targets.

## Quick verification

The certified modelling environment was Windows x64 with Python 3.14.3. From
the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tests\test_public_release.py
```

The test exits non-zero on any mismatch and prints a concise PASS/FAIL list.
It is the supported public test. The internal production-lifecycle test is not
published because it depends on non-public custody and sealed-input fixtures.

Regenerate the published result figures with:

```powershell
.venv\Scripts\python figures\render_published_figures.py --output-dir reproduced_figures
```

## Repository map

- `revision/final_execution_pipeline_20260914_v3_3_6_4/code/pipeline.py` —
  frozen modelling, evaluation, and inference engine;
- `revision/code/` — target construction, metadata comparison, encoder
  development, and certified primal-SVR support code;
- `revision/feature_extraction/` — frozen DINOv2, CLIP, and SigLIP 2 feature
  extraction programs, with portable environment-variable path overrides;
- `public_templates/final_execution_specification.PUBLIC.json` — completed,
  sanitized public record of the executed specification;
- `results/` — final non-row-level configurations, metrics, confidence
  intervals, custody completion record, and source hashes;
- `figures/` — public summary-to-figure renderer;
- `DATA_ACCESS.md` — data provenance, exclusions, and exact reproducibility
  boundary;
- `REPRODUCIBILITY.md` — staged reproduction instructions;
- `REVIEWER_COVERAGE.md` — mapping of code/open-science reviewer requests to
  public artifacts.

## Data boundary

This release does not contain transaction-level records, NFT image files,
feature matrices, fitted model binaries, row-level predictions, signed
approval documents, or sealed test targets. The final summary JSON files do
not expose transaction rows. See `DATA_ACCESS.md` for the precise reason and
the distinction between immediately verifiable results and reruns requiring
the authors' structured input bundle.

The original Dune SQL used to compile the historical transaction extract was
not recovered. The repository therefore does not claim that an independent
reader can reconstruct the exact compiled transaction table from blockchain
events alone. It does provide the subsequent deterministic processing and
analysis code, input contracts and hashes, final summary outputs, and an
explicit account of this limitation.

## Citation and license

Release metadata are in `.zenodo.json` and `CITATION.cff`. After Zenodo
archives the first GitHub Release, the version DOI will replace the pending
marker here and will be cited in the revised manuscript.

The repository source code is licensed under MIT. No license is granted for
excluded third-party transaction data, NFT images, or model weights.

