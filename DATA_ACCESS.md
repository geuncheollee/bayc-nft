# Data access and reproducibility boundary

The experiments use historical Dune exports for BAYC and MAYC, normalized token traits and original images. Raw sources and row-level analysis artifacts are **not** redistributed by this software release. Excluded items include transactions and wallets, token/image row indices, normalized metadata, target JSONL, embeddings, fitted models, and row-level predictions. Registry hashes and file paths identify required inputs without publishing their rows.

The historical Dune SQL/query identifiers are not contained in this repository. Earlier local drafts differed in their statements about recoverability of that SQL; this release does not certify that the exact historical query has been recovered. An evolving public blockchain or Dune endpoint alone does not guarantee byte-identical reconstruction of the compiled exports.

Contact the corresponding author, **stdream@hanyang.ac.kr**, for the historical query record and the structured analysis inputs needed for peer review or replication. Availability and any item-specific redistribution restriction must be confirmed with the authors. This code release does not establish new access terms, promise a response time, or certify third-party redistribution permissions.

`provenance/required_inputs.json` lists omitted input paths and recorded hashes. `run_manifest.json` files retain contemporaneous input/source hashes. Some additional upstream ledger or raw inputs are referenced by the data-audit and feature-extraction scripts; inspect those input contracts as part of a full replay. Paths and hashes are provenance, not a substitute for obtaining the files.

Public tests validate the consistency of the released summaries. Complete numerical reproduction additionally needs the omitted inputs, checkpoint downloads and compatible scientific environments. The journal Data availability statement must accurately describe the access arrangement confirmed by the authors.
