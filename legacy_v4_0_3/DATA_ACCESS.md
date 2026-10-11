# Data access and reproducibility boundary

## Public transaction query

The experiments use Ethereum NFT transactions for the verified Bored Ape Yacht Club (BAYC) and Mutant Ape Yacht Club (MAYC) contracts. The underlying public transaction records can be retrieved from [Dune](https://dune.com) by running the following DuneSQL query against `nft.trades`:

```sql
SELECT *
FROM nft.trades
WHERE blockchain = 'ethereum'
  AND nft_contract_address IN (
    0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d, -- BAYC
    0x60e4d786628fea6478f785a6d7e704777c86a7c6  -- MAYC
  );
```

The query returns records available in Dune at the time it is run. Dune is continually updated, so a later execution can include transactions after the study window and need not reproduce the frozen source exports byte for byte. The study snapshot ends during 14 April 2026 and excludes that partial day. Reconstructing the analysis table also requires the eligibility, currency, deduplication, market-reference and metadata/image matching rules documented in the manuscript Methods and Supplementary Section S1.

## Materials not redistributed

The experiments also use normalized token traits and original images. Raw exports and row-level analysis artifacts are **not** redistributed by this software release. Excluded items include transactions and wallets, token/image row indices, normalized metadata, target JSONL, embeddings, fitted models, and row-level predictions. Registry hashes and file paths identify required inputs without publishing their rows.

Contact the corresponding author, **stdream@hanyang.ac.kr**, about access to the frozen exports or other structured analysis inputs needed for peer review or replication. Availability and any item-specific redistribution restriction must be confirmed with the authors. This code release does not establish new access terms, promise a response time, or certify third-party redistribution permissions.

`provenance/required_inputs.json` lists omitted input paths and recorded hashes. `run_manifest.json` files retain contemporaneous input/source hashes. Some additional upstream ledger or raw inputs are referenced by the data-audit and feature-extraction scripts; inspect those input contracts as part of a full replay. Paths and hashes are provenance, not a substitute for obtaining the files.

Public tests validate the consistency of the released summaries. Complete numerical reproduction additionally needs the omitted inputs, checkpoint downloads and compatible scientific environments. The journal Data availability statement must accurately describe the access arrangement confirmed by the authors.
