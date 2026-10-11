# Environments and provenance

The executed run manifests record Python 3.14.3 for metadata/image/late fusion and Python 3.12.13 for corrected TF-IDF early fusion. Feature extraction records describe Python 3.12.13, PyTorch 2.7.1+cu118 and an NVIDIA GeForce GTX 1080 Ti for the documented environment. These are not a complete uniform historical lockfile.

`requirements-models.txt` declares modelling dependencies; it is not a claim that every historical run used today's installed versions. `requirements-feature-extraction.txt` lists optional checkpoint-specific dependencies; install a suitable CUDA/PyTorch build separately. Extraction code records actual model IDs, available revisions, preprocessing and pooling. No model weights are vendored.

Public aggregate checks require only Python >=3.11 and its standard library. Synthetic tests exercise selected implemented numerical invariants without NFT data. `provenance/public_test_environment.json`, when supplied, records the environment of release-time tests, not reconstructed historical versions.

Do not execute the full model scripts merely to verify the public release: they need excluded data and can take substantial resources. Historical output and checkpoint paths can be write-once. Use a fresh work copy and read REPRODUCIBILITY.md first.
