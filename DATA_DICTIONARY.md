# Numerical data dictionary

Each data/*.npz file is read with numpy.load(..., allow_pickle=False). No object arrays or executable serialization are used.

| Field | Meaning |
|---|---|
| row_id | Release-local integer pseudonym for an evaluation transaction. Unique within each collection; stable across that collection's datasets. Not a source-row number or transaction hash. |
| token_group | Release-local integer group pseudonym. Repeated sales of the same token share this value. Separate namespaces for BAYC and MAYC. |
| token_cluster_order | Ordered list of occupied group pseudonyms used to reconstruct the original bootstrap draw order. Use this explicit order; sorting pseudonyms changes the seeded draws. |
| target_lrp | Frozen original log-relative-price target. No target or past-only market reference was recomputed during packaging. |
| predictions_lrp | Float64 matrix: rows align with target_lrp and row_id, columns align with model_names. |
| model_names | Column labels. original/control/alternative refer to archived versus current fixed-configuration implementations; resolve against the matched-case record. |
| calendar_block_14d | Integer index of the occupied nonoverlapping 14-day calendar block, anchored at 2025-01-01 UTC. Exact timestamps are omitted. |
| seen_in_2022_2024 | Token occurs in the actual eligible 2022-01-01 to 2025-01-01-exclusive training rows. This does not describe encoder pretraining exposure. |
| strict_clean | Original observed multi-NFT transaction and reverse buyer/seller pair exclusion rule. It is not a label of inferred wash-trading intent. |
| strict_no_wei | strict_clean plus exclusion of price_eth <= 1e-18*(1+1e-9). Prices are not included. |

The common evaluation window is 2025-01-01 through 2026-04-13 inclusive (UTC). Some datasets subset the same rows. Do not sum dataset row counts to estimate independent observations. MAYC retains the original M1/M2 study scope.

## Pseudonymisation boundary

The package contains no token-ID-to-pseudonym or transaction-to-pseudonym crosswalk. Pseudonyms are stable across the package so paired comparisons remain possible. Token order is retained as metadata to preserve seeded resampling. Targets and predictions retain scientific precision, so these data are **pseudonymised, not guaranteed anonymous**; matching to other public sources cannot be ruled out.

The publisher-side builder and private input manifest stay outside the release. Raw training-token lists, exact trade times, prices, wallets, transaction hashes and source-row indices are omitted from numerical arrays.

## Metric and interval conventions

Transaction RMSE and MAE give each transaction equal weight. R² uses the mean target of that evaluation subset. Equal-token RMSE is the square root of the average of within-token mean squared errors.

Delta = candidate RMSE minus reference RMSE; a negative value favours the candidate. Improvement percent = 100*(reference-candidate)/reference. Intervals are linear-interpolated percentile bounds.

Token resampling samples occupied token groups with replacement, taking all transactions of each sampled group; equal-token RMSE weights each sampled group equally. Calendar resampling separately samples occupied 14-day blocks with replacement. These are two separate dependence assessments, not a combined bootstrap. Each collection/subset/scheme resets NumPy default_rng(20260908); models share draws.

Original string-token ordering and the integer-token ordering used by the full-reselection analysis are both preserved through token_cluster_order.
