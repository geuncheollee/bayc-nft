# Execution and verification

## Public fixed-prediction replay

Run the commands in README.md. The script resolves inputs relative to its own package directory, not the working directory or the author's project path. It reads only data/catalog.json, the named numeric arrays, expected/reproduction_contract.json, and the package hash manifest. The expected contract supplies comparison labels and expected values; **all actual metrics and intervals are calculated from the numerical rows**.

Outputs are metrics.csv, confidence_intervals.csv, comparison_checks.csv, bootstrap_row_expansion_checks.csv, and verification_summary.json. The final process exits nonzero for an integrity or numerical mismatch. Expected tolerance is absolute 1e-10.

Forty datasets cover all rows of the twelve source tables identified in provenance/reproduction_coverage.csv. Additional metrics/intervals can be calculated from the included predictions. Model-selection uncertainty, new fitting, raw-target reconstruction and image extraction are outside this public replay.

## Original versus current evidence

- Original representatives and seen/unseen use the original frozen predictions; subgroup models were not reselected.
- matched_original preserves the archived mixed implementations.
- matched_current contains archived predictions plus current control/alternative fits. Metadata-family-specific encoder/configuration choices remain pre-2025 selections.
- reselected_base and reselected_strict contain current uniformly counted fits for all three cleaning conditions, evaluated on common rows. They are not replacements selected by evaluation performance.
- strict_refit/residual/fixed datasets preserve earlier supporting analyses with their original scope.

## Full scientific source

legacy_v4_0_3/ preserves the previous code archive's relative layout and source hashes. analysis_source/ contains byte-identical October computational scripts, selected configurations, development scores and freeze records. The public replay is the portable, independently tested entry point.

The historical training scripts retain original path and frozen-input assumptions; they are source records, not a turnkey public-data training interface. To refit, first obtain the hash-matched inputs described in DATA_ACCESS.md, assemble a separate work tree using the historical relative layouts and review ROOT/output/cache settings. The original execution_sources.json also records vendored scientific-package files. Exact historical hash guards cannot be substituted by a different package version without a documented new run.

The strict run's scientific order is prepare/fold-fit development candidates, select/freeze configurations and late weights, final fit, evaluate, paired bootstrap, independent verification. No evaluation-driven choice is authorized by the public data. Follow legacy_v4_0_3/REPRODUCIBILITY.md for upstream extraction and baseline dependencies.

No full model training or image extraction was repeated while assembling this archive. The package validates exact fixed-prediction reproduction, not universal environment or row/count equivalence.
