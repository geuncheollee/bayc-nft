# Code and open-science reviewer coverage

This file maps the reviewer requests that directly concern the public
repository. Other reviewer requests are answered in the revised manuscript,
Supplementary Information, and point-by-point response rather than in this
code archive.

| Item | Request | Public evidence | Status before DOI |
|---|---|---|---|
| E4 | Full benchmark hyperparameters, CV protocol, grids, and seeds | Completed public execution specification; `results/selected_configurations.PUBLIC.json`; `revision/code/`; `REPRODUCIBILITY.md` | Covered computationally |
| In-house_CodeDOI | Deposit custom tools in a DOI-assigning repository | GitHub release metadata, MIT license, `.zenodo.json`, `CITATION.cff` | DOI pending first GitHub Release |
| R4_11 | Full hyperparameters for all benchmark models | Public specification and selected-configuration record | Covered computationally |
| R4_12 | Code and data availability; reproducible pipeline | Engine, processing/extraction sources, public tests, result summaries, `DATA_ACCESS.md` | Code/results covered; row-level data boundary disclosed |
| R6_6 | Reproducibility, hyperparameter transparency, and code availability | Entire archive, manifest, tests, results, and staged reproduction guide | Covered except DOI |
| R7_6 | Ridge/fold/tree grids and random seed | Public specification, selected configurations, and seed checks | Covered computationally |

The archive does not claim exact reconstruction of the compiled transaction
table because the original Dune SQL is unavailable. This limitation is stated
explicitly rather than hidden behind a general reproducibility claim.

