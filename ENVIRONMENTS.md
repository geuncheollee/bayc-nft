# Environments

The **public replay** requires Python >=3.11 and NumPy 2.4.2 only. requirements-replay.txt pins that dependency. It uses no pandas, scikit-learn, model checkpoints or compiled training packages.

Packaging verification uses Python 3.14.3 on Windows in a separate virtual environment and a copied extraction of the release. The verification receipt records whether NumPy was installed from a downloaded wheel or copied from the already installed, version-checked distribution. An offline copy is environment isolation, not a successful clean package-index installation.

The **October model experiments** used NumPy 2.4.2, SciPy 1.17.1, scikit-learn 1.8.0, joblib 1.5.3, threadpoolctl 3.6.0, XGBoost 3.0.5 and LightGBM 4.6.0; the latter packages were resolved from local experiment package directories. requirements-october-models.txt records these versions. It does not claim the historical original experiments used them or that full training has been independently reinstalled here.

Earlier experimental and release-checking environments remain documented in legacy_v4_0_3/ENVIRONMENTS.md and its run manifests. The old release-validation lockfile is not the public replay environment and is not an October refit lockfile.
