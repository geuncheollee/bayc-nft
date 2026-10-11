# Access to released and upstream inputs

## Included without author contact

The package's data/ directory contains the fixed, pseudonymised **evaluation inputs** required to reconstruct the reported main metrics and paired confidence intervals: targets, grouping/order information, aligned model predictions, and subgroup/cleaning flags. See provenance/reproduction_coverage.csv for exact numerical coverage. These files suffice for the public replay; they do not suffice to retrain the models.

The raw-data access boundary in legacy_v4_0_3/DATA_ACCESS.md describes the old release. Its statement that predictions were excluded does not describe the new data/ directory.

## Upstream materials needed for training or extraction

Raw transaction exports, complete target/eligibility ledgers, normalized traits, images, embedding matrices, original row maps and fitted checkpoints are not redistributed in this package. No third-party image, model-weight or data license is granted by the software license.

To request the exact frozen upstream inputs, contact the corresponding author at **stdream@hanyang.ac.kr**, identifying the collection and files/hashes in provenance/frozen_input_hashes.json and legacy_v4_0_3/provenance/required_inputs.json. The corresponding author must confirm item-specific availability and applicable terms. This document does not promise access on a fixed timetable or assert that every third-party input may be redistributed.

The public historical DuneSQL query is retained in legacy_v4_0_3/DATA_ACCESS.md. Later query results may differ from the frozen exports as the upstream database changes. The preserved query alone is not a byte-for-byte recreation of the study snapshot. This package does not independently verify current provider terms or availability.

## Access levels

1. **Public numerical replay:** included targets and predictions; Python + NumPy; no credentials or network needed after installing NumPy.
2. **Development-choice audit:** included candidate/family/weight scores and frozen selection records; no evaluation-based reselection.
3. **Full refitting and feature extraction:** requires the upstream materials and compatible scientific/checkpoint environments. The full pipeline was not rerun during packaging.

## Release license scope

The repository's MIT license is retained for authored software. Historical notices remain with the prior archive. Numerical outputs are supplied for verification of this study; the software license does not confer rights to the underlying NFT artwork, third-party weights, upstream exports or other excluded inputs. Final repository metadata and redistribution terms should accurately reflect the author-approved deposited files.
