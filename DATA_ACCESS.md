# Data access and reproducibility boundary

## Study sources and scope

The study analyzes verified secondary-market transactions for two Ethereum
collections:

- Bored Ape Yacht Club (BAYC): `0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d`
- Mutant Ape Yacht Club (MAYC): `0x60e4d786628fea6478f785a6d7e704777c86a7c6`

The study window ends strictly before `2026-04-14 00:00:00 UTC`. The compiled
records cover OpenSea, LooksRare, X2Y2, and Blur settlement activity and are
filtered as described in the manuscript and public execution specification.

## Included public evidence

The `results/` directory provides non-row-level outputs needed to verify the
reported tables and inferential conclusions:

- selected model families, hyperparameters, and fusion weights;
- refit completion and model-count record;
- out-of-time metrics for BAYC and MAYC;
- 2,000-iteration token-clustered bootstrap intervals;
- evaluation-custody completion metadata;
- SHA-256 hashes of the private source outputs from which the public copies
  were made.

## Inputs not redistributed

The following are deliberately absent:

- compiled transaction-level tables and development/test target rows;
- NFT image files and token-level metadata/image path indices;
- image-embedding matrices and their row-level manifests;
- fitted model binaries and row-level test predictions;
- signed approval/anchor records and sealed test-target files.

These exclusions prevent redistribution of third-party material and preserve
the custody boundary used for the single-pass evaluation. They also mean that
a complete numerical refit requires the authors' structured input bundle.

## Important historical limitation

The original Dune SQL extraction scripts used to compile the historical
transaction table were not recovered. Consequently, this release does not
claim byte-for-byte reconstruction of that compiled table directly from the
blockchain. The deterministic cleaning, target construction, model fitting,
evaluation, and inference code after compilation is supplied, together with
the hashes and dimensions of the required inputs.

## Expected private input contract

The public execution specification records the logical path, byte count where
applicable, SHA-256, and role for each required input. A full authorized rerun
requires at minimum:

1. normalized token metadata and master token/image index;
2. chronological split manifests;
3. development and sealed out-of-time targets;
4. CLIP Native, SigLIP 2, and DINOv2 Full-Frame feature matrices;
5. the approved scope/anchor records used by the fail-closed production
   engine.

The feature-extraction source code is published so that users with lawfully
obtained source images can regenerate embeddings using the recorded model
revisions and checkpoint hashes.

## Interpretation

The public release supports direct verification of reported summaries,
configuration transparency, figure regeneration, and code inspection. It is
not a self-contained redistribution of the underlying dataset. Any final
journal statement should preserve this distinction.

