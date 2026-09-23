# v4.0.1 - 23 September 2026

Documentation-only correction. `DATA_ACCESS.md` now reproduces the DuneSQL query used to retrieve public Ethereum NFT transactions for the verified BAYC and MAYC contracts. It also distinguishes records returned when the query is rerun from byte-identical reconstruction of the fixed study snapshot. No modelling code, aggregate result or scientific conclusion changed.

# v4.0.0 - 22 September 2026

Replaces the active v3.3.6.4 tree with the code actually used for the current revision's 2022-2024 fitting and 2025-2026 evaluation, including all seven encoders, corrected TF-IDF early fusion, expanded conditional late fusion, paired uncertainty and supplemental analyses. Old pipeline files not used in this revision are removed from the default-branch tree. Reused shared dependencies are retained with original paths and hashes.

This is a substantial experimental/protocol update, not a cosmetic patch. Old tags/commits and the previous Zenodo record remain historical records, not evidence for the new results. New release-time public tests and an aggregate exporter are explicitly distinguished from the previously executed scientific analyses. No new model fitting is represented as completed by the release process.
