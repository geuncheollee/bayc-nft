# BAYC/MAYC reproducibility package — v4.1.0

This package supports **Evaluating image embeddings and metadata for NFT price prediction using temporal validation**. It adds pseudonymized targets, token groups, calendar blocks and aligned predictions to the earlier v4.0.3 code archive.

**Version identity:** v4.1.0 extends the v4.0.3 archive. The prior version DOI is 10.5281/zenodo.22921602; it identifies v4.0.3. Use the DOI of the corresponding v4.1.0 Zenodo record when citing this version. Publication details are recorded in the GitHub release and its linked Zenodo record.

## Reproduce metrics and intervals

Use Python 3.11 or newer in a fresh virtual environment. The tested numerical dependency is NumPy 2.4.2.

~~~sh
python -m venv .venv
# Activate the virtual environment using your operating system's command.
python -m pip install -r requirements-replay.txt
python tools/validate_package.py
python tools/reproduce.py --output-dir reproduced
~~~

The replay **recomputes from released row-level targets and predictions**, rather than copying published summaries. It compares 9,550 numeric cells against the frozen source tables, using absolute tolerance 1e-10. It also checks paired bootstrap calculations against explicit row expansion. Each bootstrap uses 2,000 draws and seed 20260908.

For a smaller run:

~~~sh
python tools/reproduce.py --dataset BAYC_original_all --output-dir reproduced_bayc
~~~

An optional CSV export makes the numerical rows readable without an NPZ viewer:

~~~sh
python tools/export_rows.py --dataset BAYC_original_all --output-file BAYC_original_all.csv
~~~

## Contents and coverage

- **40 datasets** under data/: original representatives and seen/unseen partitions; original and current matched-family implementations; three-condition full reselection on common base/strict evaluation rows; previous fixed-configuration strict refits, residual correction and cleaning sensitivities.
- **12 original result tables**, covering 1,381 table rows, under expected/source_tables/. The reproduction contract maps each checked number to its table, zero-based data-row index and field.
- **Executed October analysis source and selection records** under analysis_source/.
- **The prior public release**, including earlier modeling/feature-extraction code, under legacy_v4_0_3/. Its README and limits describe that historical version only.
- **Input hashes, source hashes and coverage** under provenance/.
- **SHA-256 manifest** covering the current package; no credentials, wallet identifiers, transaction hashes, raw token IDs, raw trades, images, fitted models, reviewer reports or manuscript drafts are part of the new numerical data.

Read DATA_DICTIONARY.md, DATA_ACCESS.md, REPRODUCIBILITY.md and ENVIRONMENTS.md before interpreting the outputs.

## Interpretation

The original baseline, current fixed-configuration row/count comparison and current full-reselection experiment remain distinct. Bootstrap intervals condition on fitted predictions and do not include model-selection uncertainty. New 95% intervals are exploratory; original 97.5% and 99.375% levels are retained where originally reported. The study is retrospective, with repeated inspection of the evaluation period.

The available results do not establish a consistently supported incremental benefit of image embeddings over metadata in the two collections. This does not imply no visual information, equivalence of all implementations, or a causal conclusion about NFT appearance.
