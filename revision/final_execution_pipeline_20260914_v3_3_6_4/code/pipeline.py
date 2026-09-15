#!/usr/bin/env python3
"""
NFT Revision Final Execution Pipeline Engine (v3.1)
======================================================================
Production execution pipeline implementing:
- Full freeze manifest pre-unseal integrity verification
- Pre-registered inputs vs freeze inputs comparison
- Model-config-training data binding verification (12 models)
- Immutable state transitions (tune/refit blocking, freeze overwrite prevention)
- Separate read-only verify-freeze subcommand
- Preflight freshness and specification/environment re-verification
- Recursive JSON Schema validation on freeze manifest and custody lifecycle
- Standardized execution environment policy (Python 3.14 / sklearn 1.8)
- Correction of sealed test target stat sizes (BAYC: 2,546,845; MAYC: 6,497,905)
- Strict single-pass evaluation custody and atomic output publication
======================================================================
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR

# Canonical path hierarchy:
# Script path: <root>/revision/final_execution_pipeline_20260911_v3_1/code/pipeline.py
CURRENT_CODE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CURRENT_CODE_DIR.parent
REV = PACKAGE_DIR.parent
ROOT = REV.parent

CODE_LIB = REV / "code"
if str(CODE_LIB) not in sys.path:
    sys.path.insert(0, str(CODE_LIB))

try:
    from counted_primal_svr_v3 import CountedPrimalSVR
except ImportError:
    CountedPrimalSVR = None

SEED = 20260908
ONE_WEI_DECIMAL = Decimal("1e-18")
WEI_PER_ETH_DECIMAL = Decimal("1000000000000000000")  # 10^18


class PipelineError(Exception):
    """Base exception for pipeline execution errors."""
    def __init__(self, message: str, error_code: str = "PIPELINE_ERROR"):
        super().__init__(message)
        self.error_code = error_code


class CleanSliceVerificationError(PipelineError):
    """Raised when 1-wei clean-slice verification fails."""
    def __init__(self, message: str, error_code: str = "BLOCKED_SENSITIVITY_SLICE_INVALID"):
        super().__init__(message, error_code=error_code)


class PrePredictionVerificationError(PipelineError):
    """Raised when pre-prediction checks on unsealed test targets detect anomalies."""
    def __init__(self, message: str, error_code: str = "BLOCKED_PRE_PREDICTION_VERIFICATION"):
        super().__init__(message, error_code=error_code)


class EvaluationExecutionError(PipelineError):
    """Raised when an unhandled error occurs during test evaluation."""
    def __init__(self, message: str, error_code: str = "BLOCKED_EVALUATION_EXECUTION_ERROR"):
        super().__init__(message, error_code=error_code)


class DevelopmentMeanBaselineError(PipelineError):
    """Raised when development target mean baseline is missing, non-finite, corrupted, or leaks test targets."""
    def __init__(self, message: str, error_code: str = "BLOCKED_DEVELOPMENT_MEAN_BASELINE_INVALID"):
        super().__init__(message, error_code=error_code)


class ScopeDecisionsVerificationError(PipelineError):
    """Raised when user-approved Gate 1 scope decisions verification fails."""
    def __init__(self, message: str, error_code: str = "BLOCKED_USER_SCOPE_APPROVAL_REQUIRED"):
        super().__init__(message, error_code=error_code)


def is_exact_one_wei_trade(record: Dict[str, Any]) -> bool:
    """Determine if a trade record represents an exact 1-wei trade with mathematical and type rigor.

    Fail-closed validation rules (v3.3.3 Fix B):
    1. If amount_raw is present:
       - Must be a non-boolean integer (isinstance(v, int) and not isinstance(v, bool))
         or an exact integer string matching regex ^[0-9]+$.
       - Any float (e.g. 1.5), non-integer string (e.g. '1.0'), boolean (e.g. True/False),
         non-positive value (<= 0), NaN, or Inf raises ValueError.
       - If valid, exact one wei is True ONLY when integer amount_raw == 1 (False when > 1).
    2. If price_eth is present:
       - Must be strictly positive and finite Decimal.
       - price_eth * 10^18 must be an exact mathematical integer (no sub-wei amounts like 0.5e-18).
       - Sub-wei or non-integer wei values raise ValueError.
       - If valid, exact one wei is True ONLY when integer wei value == 1 (False when > 1).
    3. If BOTH amount_raw and price_eth are present:
       - Cross-validates consistency: amount_raw == (price_eth * 10^18).
       - Any discrepancy raises ValueError.
    4. If NEITHER amount_raw NOR price_eth is present:
       - Raises ValueError.
    5. Returns True ONLY if exact wei value == 1; False if exact wei value > 1.
    """
    has_raw = "amount_raw" in record and record["amount_raw"] is not None
    has_eth = "price_eth" in record and record["price_eth"] is not None

    if not has_raw and not has_eth:
        raise ValueError(f"Trade record is missing both 'amount_raw' and 'price_eth': {record}")

    raw_wei_int: Optional[int] = None
    if has_raw:
        raw_val = record["amount_raw"]
        if isinstance(raw_val, bool):
            raise ValueError(f"Invalid boolean type for amount_raw: {raw_val}")
        if isinstance(raw_val, int):
            raw_wei_int = raw_val
        elif isinstance(raw_val, str):
            val_str = raw_val.strip()
            if not re.match(r"^[0-9]+$", val_str):
                raise ValueError(f"Malformed non-integer amount_raw string: '{raw_val}'")
            raw_wei_int = int(val_str)
        else:
            raise ValueError(f"Invalid type for amount_raw ({type(raw_val).__name__}): {raw_val}")

        if raw_wei_int <= 0:
            raise ValueError(f"Non-positive amount_raw is invalid: {raw_wei_int}")

    eth_wei_int: Optional[int] = None
    if has_eth:
        price_val = record["price_eth"]
        if isinstance(price_val, bool):
            raise ValueError(f"Invalid boolean type for price_eth: {price_val}")
        try:
            dec_price = price_val if isinstance(price_val, Decimal) else Decimal(str(price_val))
        except (InvalidOperation, TypeError, ValueError) as e:
            raise ValueError(f"Malformed price_eth value '{price_val}': {e}") from e

        if not dec_price.is_finite() or dec_price <= 0:
            raise ValueError(f"price_eth must be finite and strictly positive, got: {price_val}")

        wei_dec = dec_price * WEI_PER_ETH_DECIMAL
        if wei_dec != wei_dec.to_integral_value():
            raise ValueError(f"Sub-wei fractional amount is malformed/invalid in price_eth: {dec_price} ({wei_dec} wei)")

        eth_wei_int = int(wei_dec)

    if raw_wei_int is not None and eth_wei_int is not None:
        if raw_wei_int != eth_wei_int:
            raise ValueError(
                f"Inconsistency between amount_raw ({raw_wei_int}) and price_eth ({eth_wei_int} wei)"
            )

    final_wei = raw_wei_int if raw_wei_int is not None else eth_wei_int
    assert final_wei is not None

    return final_wei == 1

PIPELINE_VERSION = "3.3.6.4-production-pipeline-20260914-v3.3.6.4"
CANONICAL_TARGET_ENGINE = "revision/final_execution_pipeline_20260914_v3_3_6_4/"
PROFILES = ["recommended_16", "confirmatory_12"]
DEFAULT_PROFILE = "recommended_16"
EXPLORATORY_ENCODERS = {
    "BAYC": ["siglip2", "clip_native"],
    "MAYC": ["siglip2", "dinov2_fullframe"],
}
SAMPLES = ["original_v1", "exact_one_wei_sensitivity_v1"]
COLLECTIONS = ["BAYC", "MAYC"]
FEATURES = ["background", "fur", "eyes", "clothes", "hat", "mouth", "earring"]
LATE_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Pre-registered physical stat sizes of the sealed test targets
SEALED_TEST_STAT_SIZES = {
    "BAYC": 2546845,
    "MAYC": 6497905,
}
SEALED_TEST_PRE_REGISTERED_SHA256 = {
    "BAYC": "9f7bc993aca0aa75769332c64787e6266ce4f0366e28d8a772dc9e11a9c0a7b0",
    "MAYC": "df5b63f072c10fb3ad1756b14ce671fa882cbaafbbf1b5a83978643b1aadf9a1",
}
SEALED_TEST_EXPECTED_ROWS = {
    "BAYC": 5120,
    "MAYC": 13019,
}


# ----------------------------------------------------------------------
# CONTEXT AND PATH MANAGEMENT (CLEAN DEPENDENCY INJECTION)
# ----------------------------------------------------------------------
@dataclass
class PipelineContext:
    """Execution context containing paths and clean dependency injection hooks."""
    package_dir: Path = PACKAGE_DIR
    root_dir: Path = ROOT
    rev_dir: Path = REV
    out_dir: Optional[Path] = None
    spec_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    audit_anchor_path: Optional[Path] = None
    external_anchor_path: Optional[Path] = None
    execution_profile: str = "recommended_16"
    approved_scope_decisions_path: Optional[Path] = None

    # Dependency injection hooks for clean test fixture isolation (defaults to None in production)
    data_loader: Optional[Callable[..., Dict[str, Any]]] = None
    test_target_loader: Optional[Callable[..., Tuple[List[Dict[str, Any]], Path]]] = None
    metadata_loader: Optional[Callable[..., List[Dict[str, Any]]]] = None
    dev_target_loader: Optional[Callable[..., List[Dict[str, Any]]]] = None

    def get_out_dir(self) -> Path:
        return self.out_dir or self.package_dir

    def get_spec_path(self) -> Path:
        if self.spec_path and self.spec_path.exists():
            return self.spec_path
        p1 = self.get_out_dir() / "final_execution_specification.json"
        if p1.exists():
            return p1
        return self.package_dir / "final_execution_specification.json"

    def get_manifest_path(self) -> Path:
        if self.manifest_path and self.manifest_path.exists():
            return self.manifest_path
        p1 = self.get_out_dir() / "execution_hash_manifest.json"
        if p1.exists():
            return p1
        return self.package_dir / "execution_hash_manifest.json"

    def get_audit_anchor_path(self) -> Path:
        if self.audit_anchor_path and self.audit_anchor_path.exists():
            return self.audit_anchor_path
        p1 = self.get_out_dir() / "audit_anchor.json"
        if p1.exists():
            return p1
        return self.package_dir / "audit_anchor.json"

    def get_approved_scope_decisions_path(self) -> Optional[Path]:
        if self.approved_scope_decisions_path and self.approved_scope_decisions_path.exists():
            return self.approved_scope_decisions_path
        return None

    def load_dataset(self, sample: str, collection: str, encoder: str) -> Dict[str, Any]:
        if self.data_loader:
            return self.data_loader(sample, collection, encoder=encoder)
        return default_load_dataset(sample, collection, encoder=encoder, ctx=self)

    def load_test_targets(self, sample: str, collection: str) -> Tuple[List[Dict[str, Any]], Path]:
        if self.test_target_loader:
            return self.test_target_loader(sample, collection)
        target_dir = "target_pipeline_20260909" if sample == "original_v1" else "one_wei_sensitivity_targets_20260909"
        test_path = self.rev_dir / target_dir / f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl"
        with test_path.open("r", encoding="utf-8") as f:
            test_rows = [json.loads(line, parse_float=Decimal) for line in f if line.strip()]
        return test_rows, test_path


def make_context_from_args(args=None, context: Optional[PipelineContext] = None) -> PipelineContext:
    if context is not None:
        return context
    out_arg = getattr(args, "out_dir", None) if args else None
    out_dir_path = Path(out_arg).resolve() if out_arg else None
    return PipelineContext(out_dir=out_dir_path)



# ----------------------------------------------------------------------
# EXTERNAL AUDIT ANCHOR RESOLUTION AND VALIDATION (Requirement A)
# ----------------------------------------------------------------------
def resolve_approved_anchor(ctx: PipelineContext, args: Any = None, require_approval: bool = True) -> Tuple[bool, str, Optional[str]]:
    """Resolve and strictly validate external auditor-approved execution manifest hash and profile.

    Fail-closed trust rules (v3.3.4):
    1. Rejects any self-issued audit_anchor.json placed inside package_dir or out_dir (or their subpaths),
       regardless of approved_by name, content, or manifest hash (BLOCKED_SELF_ISSUED_ANCHOR_REJECTED).
    2. CLI hash/profile values are expectations only; they never constitute approval evidence.
       A valid --external-anchor-path strictly outside package_dir/out_dir is mandatory for active stages.
    3. If --external-anchor-path is passed:
       - Evaluated using fully resolved canonical paths (resolving symlinks, relative paths, '..').
       - Must not be inside package_dir or out_dir (BLOCKED_SELF_ISSUED_ANCHOR_REJECTED).
       - Permitted when located in an independent audit directory under project root (outside engine package/out_dir).
       - Must exist (BLOCKED_EXTERNAL_ANCHOR_NOT_FOUND).
       - Must satisfy audit_anchor_schema.json and status == 'APPROVED' (BLOCKED_ANCHOR_SCHEMA_INVALID / BLOCKED_APPROVED_HASH_MISSING).
    4. Strictly blocks approved_profile != execution_profile (BLOCKED_APPROVED_PROFILE_MISMATCH).
    5. Validates approved manifest hash against actual execution_hash_manifest.json (BLOCKED_APPROVED_HASH_MISMATCH).
    """
    execution_profile = getattr(args, "profile", None) if args else getattr(ctx, "execution_profile", None)
    if execution_profile is None:
        if require_approval:
            return False, "BLOCKED_PROFILE_REQUIRED", "Execution profile must be explicitly specified ('recommended_16' or 'confirmatory_12'). Implicit defaulting is prohibited."
        else:
            execution_profile = DEFAULT_PROFILE
    if execution_profile not in PROFILES:
        return False, "BLOCKED_PROFILE_INVALID", f"Invalid execution profile '{execution_profile}'. Choose from {PROFILES}."

    pkg_resolved = ctx.package_dir.resolve()
    out_resolved = ctx.get_out_dir().resolve()

    # 1. Unconditionally reject any internal audit_anchor.json inside package directory or output directory
    for internal_loc in [pkg_resolved / "audit_anchor.json", out_resolved / "audit_anchor.json"]:
        if internal_loc.exists():
            err = f"Prohibited self-issued audit_anchor.json detected at {internal_loc}. Anchors inside the engine package or output directory are strictly prohibited. Approval anchors must reside strictly outside the package and output directories."
            print(f"ERROR: BLOCKED_SELF_ISSUED_ANCHOR_REJECTED: {err}")
            return False, "BLOCKED_SELF_ISSUED_ANCHOR_REJECTED", err

    cli_sha = getattr(args, "approved_execution_manifest_sha256", None) if args else getattr(ctx, "approved_execution_manifest_sha256", None)
    cli_profile = getattr(args, "approved_profile", None) if args else getattr(ctx, "approved_profile", None)
    ext_anchor_arg = getattr(args, "external_anchor_path", None) if args else (getattr(ctx, "external_anchor_path", None) or getattr(ctx, "audit_anchor_path", None))

    # 2. A physically external anchor is the only approval evidence.
    if not ext_anchor_arg:
        # CLI values remain mandatory expectations for active execution, but are never approval evidence.
        if not cli_sha and require_approval:
            return False, "BLOCKED_APPROVED_HASH_MISSING", "Approved execution manifest hash expectation is required via --approved-execution-manifest-sha256."
        if not cli_profile and require_approval:
            return False, "BLOCKED_APPROVED_PROFILE_MISSING", "Approved profile expectation is required via --approved-profile."
        if cli_sha and not re.match(r"^[0-9a-f]{64}$", str(cli_sha).strip().lower()):
            return False, "BLOCKED_APPROVED_HASH_INVALID", "Approved execution manifest hash must be a 64-character lowercase hex string."
        if cli_profile and cli_profile != execution_profile:
            err = f"Approved profile expectation '{cli_profile}' does not match execution profile '{execution_profile}'."
            print(f"ERROR: BLOCKED_APPROVED_PROFILE_MISMATCH: {err}")
            return False, "BLOCKED_APPROVED_PROFILE_MISMATCH", err
        if require_approval:
            return False, "BLOCKED_EXTERNAL_ANCHOR_REQUIRED", "A signed external audit anchor must be supplied via --external-anchor-path. CLI hash/profile values alone cannot authorize execution."
        return False, "PENDING_EXTERNAL_APPROVAL", "Signed external audit anchor is absent (pending external approval)."

    # 3. Validate the explicitly supplied external anchor before accepting any CLI expectation.
    if ext_anchor_arg:
        ext_path = Path(ext_anchor_arg).resolve()
        if is_subpath(ext_path, pkg_resolved) or is_subpath(ext_path, out_resolved):
            err = f"External anchor path '{ext_path}' is inside the package or output directory. External auditor anchor must reside strictly outside the package and output directories."
            print(f"ERROR: BLOCKED_SELF_ISSUED_ANCHOR_REJECTED: {err}")
            return False, "BLOCKED_SELF_ISSUED_ANCHOR_REJECTED", err

        if not ext_path.exists():
            err = f"Specified external audit anchor does not exist: {ext_path}"
            print(f"ERROR: BLOCKED_EXTERNAL_ANCHOR_NOT_FOUND: {err}")
            return False, "BLOCKED_EXTERNAL_ANCHOR_NOT_FOUND", err

        try:
            anchor_data = read_json(ext_path)
        except Exception as e:
            err = f"External audit anchor is unreadable JSON: {e}"
            print(f"ERROR: BLOCKED_ANCHOR_SCHEMA_INVALID: {err}")
            return False, "BLOCKED_ANCHOR_SCHEMA_INVALID", err

        schema_path = ctx.package_dir / "audit_anchor_schema.json"
        if schema_path.exists():
            try:
                validate_schema(anchor_data, read_json(schema_path))
            except SchemaValidationError as sve:
                err = f"External audit anchor failed schema validation: {sve}"
                print(f"ERROR: BLOCKED_ANCHOR_SCHEMA_INVALID: {err}")
                return False, "BLOCKED_ANCHOR_SCHEMA_INVALID", err

        status = anchor_data.get("status")
        if status != "APPROVED":
            err = f"External audit anchor status is '{status}', expected 'APPROVED'."
            print(f"ERROR: BLOCKED_APPROVED_HASH_MISSING: {err}")
            return False, "BLOCKED_APPROVED_HASH_MISSING", err

        approved_by = anchor_data.get("approved_by")
        approval_date_utc = anchor_data.get("approval_date_utc")
        if not isinstance(approved_by, str) or not approved_by.strip():
            return False, "BLOCKED_ANCHOR_SCHEMA_INVALID", "External audit anchor approved_by must be a non-empty auditor identity."
        if not isinstance(approval_date_utc, str) or not approval_date_utc.strip():
            return False, "BLOCKED_ANCHOR_SCHEMA_INVALID", "External audit anchor approval_date_utc must be a non-empty timestamp."

        file_sha = anchor_data.get("approved_execution_manifest_sha256")
        file_profile = anchor_data.get("approved_profile")
        file_scope_sha = anchor_data.get("approved_scope_decisions_sha256")
        if not file_sha or not file_profile:
            return False, "BLOCKED_ANCHOR_SCHEMA_INVALID", "External anchor must contain non-empty approved_execution_manifest_sha256 and approved_profile fields."
        if not isinstance(file_scope_sha, str) or not re.match(r"^[0-9a-f]{64}$", file_scope_sha.strip().lower()):
            return False, "BLOCKED_ANCHOR_SCHEMA_INVALID", "APPROVED external anchor must bind approved_scope_decisions_sha256 as 64 lowercase hex characters."
        scope_arg = getattr(args, "approved_scope_decisions_path", None) if args else getattr(ctx, "approved_scope_decisions_path", None)
        if scope_arg:
            scope_path = Path(scope_arg).resolve()
            if not scope_path.exists():
                return False, "BLOCKED_SCOPE_FILE_NOT_FOUND", f"Approved scope decisions file not found: {scope_path}"
            actual_scope_sha = sha256_file(scope_path)
            if file_scope_sha.strip().lower() != actual_scope_sha:
                return False, "BLOCKED_SCOPE_SHA_MISMATCH", "External anchor scope-decisions hash does not match the supplied approved scope document."
        if not cli_sha and require_approval:
            return False, "BLOCKED_APPROVED_HASH_MISSING", "Approved execution manifest hash expectation is required via --approved-execution-manifest-sha256."
        if not cli_profile and require_approval:
            return False, "BLOCKED_APPROVED_PROFILE_MISSING", "Approved profile expectation is required via --approved-profile."
        if cli_sha and not re.match(r"^[0-9a-f]{64}$", str(cli_sha).strip().lower()):
            return False, "BLOCKED_APPROVED_HASH_INVALID", "Approved execution manifest hash must be a 64-character lowercase hex string."
        if cli_profile and cli_profile != execution_profile:
            return False, "BLOCKED_APPROVED_PROFILE_MISMATCH", "CLI approved profile expectation does not match the active execution profile."
        if cli_sha and str(cli_sha).strip().lower() != str(file_sha).strip().lower():
            return False, "BLOCKED_APPROVED_HASH_MISMATCH", "CLI approved manifest hash expectation does not match the signed external anchor."
        if cli_profile and cli_profile != file_profile:
            return False, "BLOCKED_APPROVED_PROFILE_MISMATCH", "CLI approved profile expectation does not match the signed external anchor."
        cli_sha = file_sha
        cli_profile = file_profile

    approved_sha = cli_sha
    approved_profile = cli_profile

    if not approved_sha or not approved_profile:
        if require_approval:
            if not approved_sha:
                return False, "BLOCKED_APPROVED_HASH_MISSING", "Approved execution manifest hash is required. Provide --approved-execution-manifest-sha256 or valid --external-anchor-path."
            if not approved_profile:
                return False, "BLOCKED_APPROVED_PROFILE_MISSING", "Approved profile is required. Provide --approved-profile or valid --external-anchor-path."
        else:
            if approved_profile and approved_profile != execution_profile:
                err = f"Approved profile '{approved_profile}' does not match execution profile '{execution_profile}'."
                print(f"ERROR: BLOCKED_APPROVED_PROFILE_MISMATCH: {err}")
                return False, "BLOCKED_APPROVED_PROFILE_MISMATCH", err
            return False, "PENDING_EXTERNAL_APPROVAL", "Approved external anchor hash/profile is missing (pending external approval)."

    approved_sha = str(approved_sha).strip().lower()
    if not re.match(r"^[0-9a-f]{64}$", approved_sha):
        return False, "BLOCKED_APPROVED_HASH_INVALID", "Approved execution manifest hash must be a 64-character lowercase hex string."

    if approved_profile != execution_profile:
        err = f"Approved profile '{approved_profile}' does not match execution profile '{execution_profile}'."
        print(f"ERROR: BLOCKED_APPROVED_PROFILE_MISMATCH: {err}")
        return False, "BLOCKED_APPROVED_PROFILE_MISMATCH", err

    manifest_path = ctx.get_manifest_path()
    if not manifest_path.exists():
        return False, "BLOCKED_MANIFEST_NOT_FOUND", f"execution_hash_manifest.json not found at {manifest_path}"
    cur_man_sha = sha256_file(manifest_path)
    if cur_man_sha != approved_sha:
        err = f"execution_hash_manifest.json SHA-256 mismatch with approved anchor! (Anchor: {approved_sha[:12]}, Actual: {cur_man_sha[:12]})"
        print(f"ERROR: BLOCKED_APPROVED_HASH_MISMATCH: {err}")
        return False, "BLOCKED_APPROVED_HASH_MISMATCH", err

    return True, approved_sha, approved_profile


# ----------------------------------------------------------------------
# GATE 1 SCOPE DECISION DOCUMENT RESOLUTION AND VALIDATION (v3.3.6)
# ----------------------------------------------------------------------
def resolve_and_verify_approved_scope_decisions(
    ctx: PipelineContext,
    args: Any = None,
    require_approval: bool = True
) -> Tuple[bool, Optional[Path], Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Resolve, parse, and strictly validate the user-approved Gate 1 scope decisions JSON.

    Fail-closed rules (current Gate 1 contract):
    1. Rejects any self-issued approved_scope_decisions.json inside package_dir or out_dir (BLOCKED_SELF_ISSUED_SCOPE_REJECTED).
    2. Scope file must be explicitly provided via --approved-scope-decisions-path (or ctx.approved_scope_decisions_path).
    3. If not provided:
       - If require_approval is True: halts with BLOCKED_USER_SCOPE_APPROVAL_REQUIRED.
       - If require_approval is False (e.g. preflight): returns success with None.
    4. Evaluated using canonical resolved paths (symlinks, '..').
    5. Scope file must exist (BLOCKED_SCOPE_FILE_NOT_FOUND).
    6. Must satisfy approved_scope_decisions.schema.json:
       - governance_status == 'USER_APPROVED'
       - execution_profile present in {'recommended_16', 'confirmatory_12'}
       - candidate_manifest_sha256 matching actual execution_hash_manifest.json SHA-256
       - target_execution_engine matching the canonical engine URI
       - decisions array with exactly 4 items
       - DECISION-01, DECISION-02, DECISION-03, DECISION-04 present exactly once
       - each item status == 'USER_APPROVED'
       - each item selected_option in {'A', 'B'}
    7. Scope Profile Binding:
       - scope execution_profile must match CLI --profile exactly.
       - scope execution_profile must match CLI --approved-profile and external anchor approved_profile if provided.
       - Mismatch raises BLOCKED_SCOPE_PROFILE_MISMATCH with exit code 1.
    8. Scope Engine & Manifest Binding:
       - scope candidate_manifest_sha256 must match actual execution_hash_manifest.json SHA-256. Mismatch raises BLOCKED_SCOPE_MANIFEST_MISMATCH.
       - scope target_execution_engine must match the canonical engine URI. Mismatch raises BLOCKED_SCOPE_ENGINE_MISMATCH.
    9. DECISION Semantic Enforcement:
       - Fully supported combination: Option A (D01-A, D02-A, D03-A, D04-A).
       - In active stages (or preflight when scope is provided), unsupported options (Option B) block with explicit
         tokens: BLOCKED_DECISION_01_UNSUPPORTED, BLOCKED_DECISION_02_UNSUPPORTED, BLOCKED_DECISION_03_UNSUPPORTED, BLOCKED_DECISION_04_UNSUPPORTED with exit code 1.
    10. Computes actual SHA-256 of the scope file on disk.
    Returns: (success, scope_path, scope_sha, scope_data, error_code)
    """
    pkg_resolved = ctx.package_dir.resolve()
    out_resolved = ctx.get_out_dir().resolve()

    # 1. Reject any internal approved_scope_decisions.json inside package directory or output directory
    for internal_loc in [pkg_resolved / "approved_scope_decisions.json", out_resolved / "approved_scope_decisions.json"]:
        if internal_loc.exists():
            err = f"Prohibited self-issued approved_scope_decisions.json detected at {internal_loc}. Scope approval documents must reside strictly outside the engine package and output directories."
            print(f"ERROR: BLOCKED_SELF_ISSUED_SCOPE_REJECTED: {err}")
            return False, None, None, None, "BLOCKED_SELF_ISSUED_SCOPE_REJECTED"

    scope_arg = (getattr(args, "approved_scope_decisions_path", None) if args else None) or \
                (getattr(args, "scope_decisions_path", None) if args else None) or \
                getattr(ctx, "approved_scope_decisions_path", None) or \
                (ctx.get_approved_scope_decisions_path() if hasattr(ctx, "get_approved_scope_decisions_path") else None)

    if not scope_arg:
        if require_approval:
            err = "Active execution requires --approved-scope-decisions-path pointing to a USER_APPROVED Gate 1 scope decisions JSON file."
            print(f"ERROR: BLOCKED_USER_SCOPE_APPROVAL_REQUIRED: {err}")
            return False, None, None, None, "BLOCKED_USER_SCOPE_APPROVAL_REQUIRED"
        else:
            return True, None, None, None, None

    scope_path = Path(scope_arg).resolve()
    if is_subpath(scope_path, pkg_resolved) or is_subpath(scope_path, out_resolved):
        err = f"Prohibited self-issued approved_scope_decisions.json detected inside package/out directory: {scope_path}."
        print(f"ERROR: BLOCKED_SELF_ISSUED_SCOPE_REJECTED: {err}")
        return False, None, None, None, "BLOCKED_SELF_ISSUED_SCOPE_REJECTED"

    if not scope_path.exists():
        err = f"Specified approved_scope_decisions file not found: {scope_path}"
        print(f"ERROR: BLOCKED_SCOPE_FILE_NOT_FOUND: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_FILE_NOT_FOUND"

    try:
        scope_data = json.loads(scope_path.read_text(encoding="utf-8"))
    except Exception as e:
        err = f"Failed to parse approved_scope_decisions JSON at {scope_path}: {e}"
        print(f"ERROR: BLOCKED_SCOPE_SCHEMA_INVALID: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_SCHEMA_INVALID"


    # Check governance status
    gov_status = scope_data.get("governance_status")
    if gov_status != "USER_APPROVED":
        err = f"approved_scope_decisions governance_status is '{gov_status}', must be 'USER_APPROVED'."
        print(f"ERROR: BLOCKED_USER_SCOPE_NOT_APPROVED: {err}")
        return False, None, None, None, "BLOCKED_USER_SCOPE_NOT_APPROVED"

    # Requirement 1: Scope Profile Binding
    scope_profile = scope_data.get("execution_profile")
    if not scope_profile or not isinstance(scope_profile, str) or scope_profile not in PROFILES:
        err = f"Scope execution_profile '{scope_profile}' is missing or invalid. Must be one of {PROFILES}."
        print(f"ERROR: BLOCKED_SCOPE_PROFILE_MISMATCH: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_PROFILE_MISMATCH"

    cli_profile = getattr(args, "profile", None) if args else getattr(ctx, "execution_profile", None)
    if cli_profile and scope_profile != cli_profile:
        err = f"Scope execution_profile '{scope_profile}' does not match CLI execution profile '{cli_profile}'."
        print(f"ERROR: BLOCKED_SCOPE_PROFILE_MISMATCH: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_PROFILE_MISMATCH"

    cli_approved_profile = getattr(args, "approved_profile", None) if args else getattr(ctx, "approved_profile", None)
    if cli_approved_profile and scope_profile != cli_approved_profile:
        err = f"Scope execution_profile '{scope_profile}' does not match CLI approved profile '{cli_approved_profile}'."
        print(f"ERROR: BLOCKED_SCOPE_PROFILE_MISMATCH: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_PROFILE_MISMATCH"

    ext_anchor_arg = (getattr(args, "external_anchor_path", None) if args else None) or \
                     getattr(ctx, "external_anchor_path", None) or \
                     getattr(ctx, "audit_anchor_path", None)
    if ext_anchor_arg:
        ext_path = Path(ext_anchor_arg).resolve()
        if ext_path.exists() and not is_subpath(ext_path, pkg_resolved) and not is_subpath(ext_path, out_resolved):
            try:
                anchor_json = read_json(ext_path)
                anc_prof = anchor_json.get("approved_profile")
                if anc_prof and anc_prof != scope_profile:
                    err = f"External audit anchor approved_profile '{anc_prof}' does not match scope execution_profile '{scope_profile}'."
                    print(f"ERROR: BLOCKED_SCOPE_PROFILE_MISMATCH: {err}")
                    return False, None, None, None, "BLOCKED_SCOPE_PROFILE_MISMATCH"
            except Exception:
                pass

    # Requirement 2: Scope Engine & Manifest Binding
    scope_manifest_sha = scope_data.get("candidate_manifest_sha256")
    if not scope_manifest_sha or not isinstance(scope_manifest_sha, str):
        err = "Scope candidate_manifest_sha256 is missing or not a string."
        print(f"ERROR: BLOCKED_SCOPE_MANIFEST_MISMATCH: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_MANIFEST_MISMATCH"

    manifest_path = ctx.get_manifest_path()
    if not manifest_path.exists():
        err = f"execution_hash_manifest.json not found at {manifest_path}"
        print(f"ERROR: BLOCKED_MANIFEST_NOT_FOUND: {err}")
        return False, None, None, None, "BLOCKED_MANIFEST_NOT_FOUND"

    actual_man_sha = sha256_file(manifest_path)
    if scope_manifest_sha.strip().lower() != actual_man_sha.strip().lower():
        err = f"Scope candidate_manifest_sha256 '{scope_manifest_sha}' does not match actual execution_hash_manifest.json SHA '{actual_man_sha}'."
        print(f"ERROR: BLOCKED_SCOPE_MANIFEST_MISMATCH: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_MANIFEST_MISMATCH"

    target_engine = scope_data.get("target_execution_engine")
    if not target_engine or not isinstance(target_engine, str):
        err = "Scope target_execution_engine is missing or not a string."
        print(f"ERROR: BLOCKED_SCOPE_ENGINE_MISMATCH: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_ENGINE_MISMATCH"

    if target_engine != CANONICAL_TARGET_ENGINE:
        err = f"Scope target_execution_engine '{target_engine}' does not match canonical engine target '{CANONICAL_TARGET_ENGINE}'."
        print(f"ERROR: BLOCKED_SCOPE_ENGINE_MISMATCH: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_ENGINE_MISMATCH"

    # Decisions array structural verification
    decisions = scope_data.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 4:
        err = f"decisions array must contain exactly 4 items, found {len(decisions) if isinstance(decisions, list) else type(decisions)}"
        print(f"ERROR: BLOCKED_USER_SCOPE_NOT_APPROVED: {err}")
        return False, None, None, None, "BLOCKED_USER_SCOPE_NOT_APPROVED"

    required_ids = {"DECISION-01", "DECISION-02", "DECISION-03", "DECISION-04"}
    seen_ids = set()
    for d in decisions:
        if not isinstance(d, dict):
            err = "Each decision item must be an object."
            print(f"ERROR: BLOCKED_USER_SCOPE_NOT_APPROVED: {err}")
            return False, None, None, None, "BLOCKED_USER_SCOPE_NOT_APPROVED"
        d_id = d.get("decision_id")
        d_status = d.get("status")
        d_opt = d.get("selected_option")

        if d_id not in required_ids:
            err = f"Unexpected or missing decision_id '{d_id}'. Must be one of {required_ids}."
            print(f"ERROR: BLOCKED_USER_SCOPE_NOT_APPROVED: {err}")
            return False, None, None, None, "BLOCKED_USER_SCOPE_NOT_APPROVED"
        if d_id in seen_ids:
            err = f"Duplicate decision_id '{d_id}' found."
            print(f"ERROR: BLOCKED_USER_SCOPE_NOT_APPROVED: {err}")
            return False, None, None, None, "BLOCKED_USER_SCOPE_NOT_APPROVED"
        seen_ids.add(d_id)

        if d_status != "USER_APPROVED":
            err = f"Decision {d_id} status is '{d_status}', must be 'USER_APPROVED'."
            print(f"ERROR: BLOCKED_USER_SCOPE_NOT_APPROVED: {err}")
            return False, None, None, None, "BLOCKED_USER_SCOPE_NOT_APPROVED"
        if d_opt == "B":
            num_suffix = d_id.replace("DECISION-", "")
            blocked_token = f"BLOCKED_DECISION_{num_suffix}_UNSUPPORTED"
            err = f"Execution engine v3.3.6.4 does not support Option B for {d_id}. Only Option A is supported in current pipeline implementation."
            print(f"ERROR: {blocked_token}: {err}")
            return False, None, None, None, blocked_token
        elif d_opt != "A":
            err = f"Decision {d_id} selected_option is '{d_opt}', must be 'A'."
            print(f"ERROR: BLOCKED_USER_SCOPE_NOT_APPROVED: {err}")
            return False, None, None, None, "BLOCKED_USER_SCOPE_NOT_APPROVED"

    if seen_ids != required_ids:
        err = f"Missing required decisions: {required_ids - seen_ids}"
        print(f"ERROR: BLOCKED_USER_SCOPE_NOT_APPROVED: {err}")
        return False, None, None, None, "BLOCKED_USER_SCOPE_NOT_APPROVED"

    # Schema validation against approved_scope_decisions.schema.json
    schema_path = ctx.package_dir / "approved_scope_decisions.schema.json"
    if not schema_path.exists():
        schema_path = ctx.root_dir / "revision" / "final_scope_and_execution_governance_20260913_v2_2_1" / "approved_scope_decisions.schema.json"

    if not schema_path.exists():
        err = f"approved_scope_decisions schema missing at {schema_path}"
        print(f"ERROR: BLOCKED_SCOPE_SCHEMA_INVALID: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_SCHEMA_INVALID"

    try:
        schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
        validate_schema(scope_data, schema_obj)
    except Exception as e:
        err = f"approved_scope_decisions schema validation failed: {e}"
        print(f"ERROR: BLOCKED_SCOPE_SCHEMA_INVALID: {err}")
        return False, None, None, None, "BLOCKED_SCOPE_SCHEMA_INVALID"

    scope_sha = sha256_file(scope_path)
    return True, scope_path, scope_sha, scope_data, None


def get_expected_model_roles(profile: str = "recommended_16") -> List[Tuple[str, str, str, Optional[str]]]:
    """Return canonical list of (sample, collection, role, optional_encoder) for the given execution profile."""
    base = [
        ("original_v1", "BAYC", "metadata_baseline", "none"),
        ("original_v1", "BAYC", "augmented_candidate", "dinov2_fullframe"),
        ("original_v1", "BAYC", "mandatory_early_benchmark", "dinov2_fullframe"),
        ("original_v1", "MAYC", "metadata_baseline", "none"),
        ("original_v1", "MAYC", "augmented_candidate", "clip_native"),
        ("original_v1", "MAYC", "mandatory_early_benchmark", "clip_native"),
        ("exact_one_wei_sensitivity_v1", "BAYC", "metadata_baseline", "none"),
        ("exact_one_wei_sensitivity_v1", "BAYC", "augmented_candidate", "siglip2"),
        ("exact_one_wei_sensitivity_v1", "BAYC", "mandatory_early_benchmark", "siglip2"),
        ("exact_one_wei_sensitivity_v1", "MAYC", "metadata_baseline", "none"),
        ("exact_one_wei_sensitivity_v1", "MAYC", "augmented_candidate", "siglip2"),
        ("exact_one_wei_sensitivity_v1", "MAYC", "mandatory_early_benchmark", "siglip2"),
    ]
    if profile == "recommended_16":
        exploratory = [
            ("original_v1", "BAYC", "exploratory_augmented", "siglip2"),
            ("original_v1", "BAYC", "exploratory_augmented", "clip_native"),
            ("original_v1", "MAYC", "exploratory_augmented", "siglip2"),
            ("original_v1", "MAYC", "exploratory_augmented", "dinov2_fullframe"),
        ]
        return base + exploratory
    return base

# ----------------------------------------------------------------------
# JSON SCHEMA VALIDATOR (PURE PYTHON RECURSIVE ENGINE)
# ----------------------------------------------------------------------
class SchemaValidationError(Exception):
    """Raised when JSON data violates formal JSON Schema specifications."""
    pass


class PrePredictionVerificationError(Exception):
    """Raised when pre-prediction checks on unsealed test targets detect anomalies."""
    pass



# ----------------------------------------------------------------------
# EXECUTION MANIFEST AND RUNTIME ENVIRONMENT VALIDATORS
# ----------------------------------------------------------------------
def validate_execution_hash_manifest_structure(manifest_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate structure, artifact counts, uniqueness, and SHA formats in execution_hash_manifest.json."""
    errors = []
    tot_in = manifest_data.get("total_input_artifacts")
    inputs = manifest_data.get("input_artifacts", [])
    if tot_in != 31:
        errors.append(f"Declared total_input_artifacts={tot_in}, expected 31")
    if len(inputs) != 31:
        errors.append(f"Actual input_artifacts count={len(inputs)}, expected 31")

    tot_sealed = manifest_data.get("total_sealed_test_artifacts")
    sealed = manifest_data.get("sealed_test_artifacts", [])
    if tot_sealed != 4:
        errors.append(f"Declared total_sealed_test_artifacts={tot_sealed}, expected 4")
    if len(sealed) != 4:
        errors.append(f"Actual sealed_test_artifacts count={len(sealed)}, expected 4")

    paths_in = [i.get("path") for i in inputs]
    if len(paths_in) != len(set(paths_in)):
        errors.append("Duplicate paths found in input_artifacts")
    roles_in = [i.get("role") for i in inputs]
    if len(roles_in) != len(set(roles_in)):
        errors.append("Duplicate roles found in input_artifacts")

    paths_sealed = [s.get("path") for s in sealed]
    if len(paths_sealed) != len(set(paths_sealed)):
        errors.append("Duplicate paths found in sealed_test_artifacts")

    hex_chars = set("0123456789abcdef")
    for idx, item in enumerate(inputs):
        sha = item.get("sha256", "")
        if len(sha) != 64 or any(c not in hex_chars for c in sha):
            errors.append(f"Invalid SHA-256 format in input_artifacts[{idx}] ('{item.get('path')}'): '{sha}'")

    for idx, item in enumerate(sealed):
        sha = item.get("sha256", "")
        if len(sha) != 64 or any(c not in hex_chars for c in sha):
            errors.append(f"Invalid SHA-256 format in sealed_test_artifacts[{idx}] ('{item.get('path')}'): '{sha}'")

    return (len(errors) == 0, errors)


def verify_runtime_environment(env_lock: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Enforce that the executing runtime strictly matches locked production environment."""
    errors = []
    versions = env_lock.get("runtime_versions", {})
    expected_py_info = versions.get("python_version_info")
    if expected_py_info and list(sys.version_info[:3]) != expected_py_info[:3]:
        errors.append(f"Python version mismatch: expected {expected_py_info[:3]}, running {list(sys.version_info[:3])}")

    import sklearn, numpy, scipy, joblib
    if sklearn.__version__ != versions.get("scikit_learn_version"):
        errors.append(f"scikit-learn mismatch: expected {versions.get('scikit_learn_version')}, running {sklearn.__version__}")
    if numpy.__version__ != versions.get("numpy_version"):
        errors.append(f"NumPy mismatch: expected {versions.get('numpy_version')}, running {numpy.__version__}")
    if scipy.__version__ != versions.get("scipy_version"):
        errors.append(f"SciPy mismatch: expected {versions.get('scipy_version')}, running {scipy.__version__}")
    if joblib.__version__ != versions.get("joblib_version"):
        errors.append(f"joblib mismatch: expected {versions.get('joblib_version')}, running {joblib.__version__}")

    if "threadpoolctl_version" in versions:
        try:
            import threadpoolctl
            if threadpoolctl.__version__ != versions.get("threadpoolctl_version"):
                errors.append(f"threadpoolctl mismatch: expected {versions.get('threadpoolctl_version')}, running {threadpoolctl.__version__}")
        except ImportError:
            errors.append("threadpoolctl is required but not installed")

    import platform
    expected_sys = versions.get("system", "Windows")
    if platform.system() != expected_sys:
        errors.append(f"Platform system mismatch: expected {expected_sys}, running {platform.system()}")

    return (len(errors) == 0, errors)


def check_required_supporting_files(ctx: PipelineContext, require_configs: bool = False) -> Tuple[bool, List[str]]:
    """Fail-closed verification: ensure all essential schemas, codes, and specifications exist."""
    missing = []
    out_dir = ctx.get_out_dir()
    pkg_dir = ctx.package_dir

    spec = ctx.get_spec_path()
    if not spec.exists():
        missing.append(f"Specification missing: {spec.name}")

    manifest = ctx.get_manifest_path()
    if not manifest.exists():
        missing.append(f"Execution hash manifest missing: {manifest.name}")

    env_lock = out_dir / "environment_lock.json"
    if not env_lock.exists():
        env_lock = pkg_dir / "environment_lock.json"
    if not env_lock.exists():
        missing.append("environment_lock.json missing")

    f_schema = out_dir / "freeze_manifest_schema.json"
    if not f_schema.exists():
        f_schema = pkg_dir / "freeze_manifest_schema.json"
    if not f_schema.exists():
        missing.append("freeze_manifest_schema.json missing")

    c_schema = out_dir / "evaluation_custody_schema.json"
    if not c_schema.exists():
        c_schema = pkg_dir / "evaluation_custody_schema.json"
    if not c_schema.exists():
        missing.append("evaluation_custody_schema.json missing")

    solver = ctx.rev_dir / "code" / "counted_primal_svr_v3.py"
    if not solver.exists():
        missing.append("code/counted_primal_svr_v3.py missing")

    a_template = out_dir / "audit_anchor.template.json"
    if not a_template.exists():
        a_template = pkg_dir / "audit_anchor.template.json"
    if not a_template.exists():
        missing.append("audit_anchor.template.json missing")

    a_schema = out_dir / "audit_anchor_schema.json"
    if not a_schema.exists():
        a_schema = pkg_dir / "audit_anchor_schema.json"
    if not a_schema.exists():
        missing.append("audit_anchor_schema.json missing")

    req_lock = out_dir / "requirements_lock.txt"
    if not req_lock.exists():
        req_lock = pkg_dir / "requirements_lock.txt"
    if not req_lock.exists():
        missing.append("requirements_lock.txt missing")

    if require_configs:
        cfg = out_dir / "selected_configurations.json"
        if not cfg.exists():
            cfg = pkg_dir / "selected_configurations.json"
        if not cfg.exists():
            missing.append("selected_configurations.json missing")

    return (len(missing) == 0, missing)

def validate_schema(instance: Any, schema: Dict[str, Any], path: str = "$") -> None:
    """Validate JSON instance against schema enforcing types, required keys, enums, consts, items, and bounds."""
    if "const" in schema:
        expected_const = schema["const"]
        if instance != expected_const:
            raise SchemaValidationError(f"Value '{instance}' at {path} does not match required const '{expected_const}'")
    expected_type = schema.get("type")
    if expected_type:
        if expected_type == "object":
            if not isinstance(instance, dict):
                raise SchemaValidationError(f"Expected object at {path}, got {type(instance).__name__}")
            for req in schema.get("required", []):
                if req not in instance:
                    raise SchemaValidationError(f"Missing required field '{req}' at {path}")
            props = schema.get("properties", {})
            for k, v in instance.items():
                if k in props:
                    validate_schema(v, props[k], path=f"{path}.{k}")
                elif schema.get("additionalProperties") is False:
                    raise SchemaValidationError(f"Additional property '{k}' not allowed at {path}")
        elif expected_type == "array":
            if not isinstance(instance, list):
                raise SchemaValidationError(f"Expected array at {path}, got {type(instance).__name__}")
            min_items = schema.get("minItems")
            if min_items is not None and len(instance) < min_items:
                raise SchemaValidationError(f"Array at {path} has {len(instance)} items, minimum required is {min_items}")
            max_items = schema.get("maxItems")
            if max_items is not None and len(instance) > max_items:
                raise SchemaValidationError(f"Array at {path} has {len(instance)} items, maximum allowed is {max_items}")
            item_schema = schema.get("items")
            if item_schema:
                for idx, item in enumerate(instance):
                    validate_schema(item, item_schema, path=f"{path}[{idx}]")
        elif expected_type == "string":
            if not isinstance(instance, str):
                raise SchemaValidationError(f"Expected string at {path}, got {type(instance).__name__}")
            enums = schema.get("enum")
            if enums is not None and instance not in enums:
                raise SchemaValidationError(f"String '{instance}' at {path} not in allowed enum {enums}")
        elif expected_type == "integer":
            if not isinstance(instance, int) or isinstance(instance, bool):
                raise SchemaValidationError(f"Expected integer at {path}, got {type(instance).__name__}")
        elif expected_type == "number":
            if not isinstance(instance, (int, float)) or isinstance(instance, bool):
                raise SchemaValidationError(f"Expected number at {path}, got {type(instance).__name__}")
        elif expected_type == "boolean":
            if not isinstance(instance, bool):
                raise SchemaValidationError(f"Expected boolean at {path}, got {type(instance).__name__}")


# ----------------------------------------------------------------------
# CRYPTOGRAPHIC AND FILE UTILITIES
# ----------------------------------------------------------------------

def validate_deliverable_provenance(
    deliverable_name: str,
    deliverable_path: Path,
    expected_profile: str,
    expected_scope_sha: str,
    expected_decisions: Dict[str, str],
    expected_manifest_sha: Optional[str] = None,
    expected_freeze_sha: Optional[str] = None,
    deliverable_data: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Strict 7-step fail-closed provenance validation for intermediate deliverables.

    Validates:
    1. Existence on disk.
    2. JSON dict type.
    3. Non-empty required provenance fields.
    4. Exact format (lowercase 64-hex hashes, valid profile strings, valid decision dicts).
    5. Exact value match against caller/disk expectations.
    6. Returns (False, error_token, error_message) on any mismatch.
    """
    if not deliverable_path.exists():
        err_token = "BLOCKED_CONFIG_NOT_FOUND" if "configuration" in deliverable_name else \
                    ("BLOCKED_REFIT_SUMMARY_NOT_FOUND" if "refit_summary" in deliverable_name else \
                    ("BLOCKED_PIPELINE_NOT_FROZEN" if "freeze" in deliverable_name else \
                    ("BLOCKED_CUSTODY_NOT_FOUND" if "custody" in deliverable_name else \
                    ("BLOCKED_NO_TEST_METRICS" if "metrics" in deliverable_name else "BLOCKED_DELIVERABLE_NOT_FOUND"))))
        return False, err_token, f"{deliverable_name} not found at {deliverable_path}"

    data = deliverable_data
    if data is None:
        try:
            data = read_json(deliverable_path)
        except Exception as e:
            return False, "BLOCKED_PROVENANCE_MISSING_OR_CORRUPT", f"{deliverable_name} failed to parse as JSON: {e}"

    if not isinstance(data, dict):
        return False, "BLOCKED_PROVENANCE_MISSING_OR_CORRUPT", f"{deliverable_name} is not a valid JSON object"

    # Freeze manifests deliberately duplicate the scope binding in
    # manifest_metadata and approved_scope_decisions. Both copies are required
    # and must agree. Treating one copy as a fallback for the other would make a
    # deleted provenance field indistinguishable from a valid manifest.
    is_freeze = ("freeze_manifest" in deliverable_name) or ("manifest_metadata" in data and "models" in data)

    # 1. approved_scope_decisions_sha256
    if is_freeze:
        manifest_metadata = data.get("manifest_metadata")
        scope_binding = data.get("approved_scope_decisions")
        if not isinstance(manifest_metadata, dict) or not isinstance(scope_binding, dict):
            return False, "BLOCKED_SCOPE_SHA_MISMATCH", f"{deliverable_name}: manifest_metadata and approved_scope_decisions must both be JSON objects"
        metadata_scope_sha = manifest_metadata.get("approved_scope_decisions_sha256")
        binding_scope_sha = scope_binding.get("sha256")
        for location, candidate in [
            ("manifest_metadata.approved_scope_decisions_sha256", metadata_scope_sha),
            ("approved_scope_decisions.sha256", binding_scope_sha),
        ]:
            if not candidate or not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate):
                return False, "BLOCKED_SCOPE_SHA_MISMATCH", f"{deliverable_name}: {location} is missing, null, or invalid hex ({candidate!r})"
        if metadata_scope_sha != binding_scope_sha:
            return False, "BLOCKED_SCOPE_SHA_MISMATCH", f"{deliverable_name}: duplicated scope SHA fields disagree"
        scope_sha = metadata_scope_sha
    else:
        scope_sha = data.get("approved_scope_decisions_sha256")

    if not scope_sha or not isinstance(scope_sha, str) or not re.match(r"^[0-9a-f]{64}$", scope_sha):
        return False, "BLOCKED_SCOPE_SHA_MISMATCH", f"{deliverable_name}: approved_scope_decisions_sha256 is missing, null, or invalid hex ({scope_sha!r})"
    if scope_sha.strip().lower() != expected_scope_sha.strip().lower():
        return False, "BLOCKED_SCOPE_SHA_MISMATCH", f"{deliverable_name}: Scope decisions SHA mismatch (found {scope_sha}, expected {expected_scope_sha})"

    # 2. execution_profile
    if is_freeze:
        metadata_profile = manifest_metadata.get("execution_profile")
        binding_profile = scope_binding.get("execution_profile")
        for location, candidate in [
            ("manifest_metadata.execution_profile", metadata_profile),
            ("approved_scope_decisions.execution_profile", binding_profile),
        ]:
            if not candidate or not isinstance(candidate, str) or candidate not in PROFILES:
                return False, "BLOCKED_PROFILE_MISMATCH", f"{deliverable_name}: {location} is missing, null, or invalid ({candidate!r})"
        if metadata_profile != binding_profile:
            return False, "BLOCKED_PROFILE_MISMATCH", f"{deliverable_name}: duplicated execution_profile fields disagree"
        prof = metadata_profile
    else:
        prof = data.get("execution_profile")

    if not prof or not isinstance(prof, str) or prof not in PROFILES:
        return False, "BLOCKED_PROFILE_MISMATCH", f"{deliverable_name}: execution_profile is missing, null, or invalid ({prof!r})"
    if prof != expected_profile:
        return False, "BLOCKED_PROFILE_MISMATCH", f"{deliverable_name}: Execution profile mismatch (found '{prof}', expected '{expected_profile}')"

    # 3. scope_decisions
    if is_freeze:
        metadata_decisions = manifest_metadata.get("scope_decisions")
        binding_decisions = scope_binding.get("scope_decisions")
        if not isinstance(metadata_decisions, dict) or not metadata_decisions:
            return False, "BLOCKED_SCOPE_DECISIONS_MISMATCH", f"{deliverable_name}: manifest_metadata.scope_decisions is missing, null, empty, or not a dict ({metadata_decisions!r})"
        if not isinstance(binding_decisions, dict) or not binding_decisions:
            return False, "BLOCKED_SCOPE_DECISIONS_MISMATCH", f"{deliverable_name}: approved_scope_decisions.scope_decisions is missing, null, empty, or not a dict ({binding_decisions!r})"
        if metadata_decisions != binding_decisions:
            return False, "BLOCKED_SCOPE_DECISIONS_MISMATCH", f"{deliverable_name}: duplicated scope_decisions fields disagree"
        decisions_dict = metadata_decisions
    else:
        decisions_dict = data.get("scope_decisions")

    if not decisions_dict or not isinstance(decisions_dict, dict):
        return False, "BLOCKED_SCOPE_DECISIONS_MISMATCH", f"{deliverable_name}: scope_decisions is missing, null, or not a dict ({decisions_dict!r})"
    required_dec_ids = {"DECISION-01", "DECISION-02", "DECISION-03", "DECISION-04"}
    if set(decisions_dict.keys()) != required_dec_ids:
        return False, "BLOCKED_SCOPE_DECISIONS_MISMATCH", f"{deliverable_name}: scope_decisions must contain exactly DECISION-01..04 (found {list(decisions_dict.keys())})"
    for k, v in expected_decisions.items():
        if decisions_dict.get(k) != v:
            return False, "BLOCKED_SCOPE_DECISIONS_MISMATCH", f"{deliverable_name}: decision '{k}' option '{decisions_dict.get(k)}' != expected '{v}'"

    # 4. approved_execution_manifest_sha256 (if expected_manifest_sha provided)
    if expected_manifest_sha:
        if is_freeze:
            man_sha = data.get("manifest_metadata", {}).get("approved_execution_manifest_sha256")
        else:
            man_sha = data.get("approved_execution_manifest_sha256") or data.get("candidate_manifest_sha256")
        if not man_sha or not isinstance(man_sha, str) or not re.match(r"^[0-9a-f]{64}$", man_sha):
            return False, "BLOCKED_APPROVED_MANIFEST_HASH_MISMATCH", f"{deliverable_name}: approved_execution_manifest_sha256 is missing, null, or invalid hex ({man_sha!r})"
        if man_sha.strip().lower() != expected_manifest_sha.strip().lower():
            return False, "BLOCKED_APPROVED_MANIFEST_HASH_MISMATCH", f"{deliverable_name}: Approved execution manifest hash mismatch (found {man_sha}, expected {expected_manifest_sha})"

    # 5. freeze_manifest_sha256 (if expected_freeze_sha provided)
    if expected_freeze_sha:
        frz_sha = data.get("freeze_manifest_sha256")
        if not frz_sha or not isinstance(frz_sha, str) or not re.match(r"^[0-9a-f]{64}$", frz_sha):
            return False, "BLOCKED_FREEZE_SHA_MISMATCH", f"{deliverable_name}: freeze_manifest_sha256 is missing, null, or invalid hex ({frz_sha!r})"
        if frz_sha.strip().lower() != expected_freeze_sha.strip().lower():
            return False, "BLOCKED_FREEZE_SHA_MISMATCH", f"{deliverable_name}: Freeze manifest SHA mismatch (found {frz_sha}, expected {expected_freeze_sha})"

    return True, None, None

def is_subpath(target: Path, parent: Path) -> bool:
    """Return True if target is identical to or located inside parent directory."""
    try:
        target.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, RuntimeError):
        return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_save_json(path: Path, data: Any):
    """Crash-safe atomic JSON file writer using temporary file, fsync, and replace with Windows retry."""
    def json_default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    tmp_path = path.with_name(f"{path.stem}_{os.getpid()}_{time.time_ns()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False, default=json_default))
        f.flush()
        os.fsync(f.fileno())
    
    last_err = None
    for attempt in range(10):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as pe:
            last_err = pe
            time.sleep(0.05 * (attempt + 1))
    if last_err:
        raise last_err


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_manifest_path(p_str: str, root_dir: Path = ROOT) -> Path:
    p = Path(p_str)
    if p.is_absolute():
        return p
    return root_dir / p


def acquire_evaluation_lock(lock_path: Path) -> int:
    """Atomically acquire execution lock via OS-level O_CREAT | O_EXCL."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags, 0o644)
        info = {
            "pid": os.getpid(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        os.write(fd, json.dumps(info).encode("utf-8"))
        os.fsync(fd)
        return fd
    except FileExistsError:
        raise RuntimeError(f"Atomic lock active: {lock_path} already exists. Concurrent execution prevented.")


def release_evaluation_lock(fd: int, lock_path: Path):
    """Release atomic execution lock and unlink file."""
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def create_one_hot_encoder() -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    if hasattr(OneHotEncoder(), "sparse_output"):
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, tokens: Optional[np.ndarray] = None) -> Dict[str, float]:
    n = len(y_true)
    assert n == len(y_pred)
    sse = float(np.sum((y_true - y_pred) ** 2))
    mse = sse / n if n > 0 else 0.0
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred))) if n > 0 else 0.0
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2)) if n > 0 else 0.0
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else 0.0

    metrics = dict(n=n, sse=sse, mse=mse, rmse=rmse, mae=mae, r2=r2)

    if tokens is not None and len(tokens) == n and n > 0:
        unique_t, inv, cnts = np.unique(tokens, return_inverse=True, return_counts=True)
        token_mses = []
        for idx in range(len(unique_t)):
            mask = inv == idx
            token_mses.append(np.mean((y_true[mask] - y_pred[mask]) ** 2))
        metrics["equal_token_rmse"] = float(np.sqrt(np.mean(token_mses)))
    else:
        metrics["equal_token_rmse"] = rmse
    return metrics


# ----------------------------------------------------------------------
# ESTIMATOR BUILDERS AND FUSION WRAPPERS
# ----------------------------------------------------------------------
class CountedPrimalSVRVisionEstimator(BaseEstimator, RegressorMixin):
    """Certified primal SVR with stationary certificate ||grad P(beta)||_2 <= 1e-6."""

    def __init__(self, C: float = 0.01, epsilon: float = 0.1, max_iter: int = 5000, gradient_tolerance: float = 1e-6):
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.gradient_tolerance = gradient_tolerance
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        if CountedPrimalSVR is None:
            raise RuntimeError("CountedPrimalSVR is not available in environment.")
        unique_X, inverse = np.unique(X, axis=0, return_inverse=True)
        solver = CountedPrimalSVR(
            C=self.C,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            gradient_tolerance=self.gradient_tolerance,
        )
        solver.fit_counts(unique_X, inverse, y)
        self.model = solver
        self.coef_ = solver.coef_
        self.intercept_ = solver.intercept_
        self.n_iter_ = solver.n_iter_
        self.gradient_norm_ = solver.gradient_norm_
        self.objective_ = solver.objective_
        self.objective_gap_upper_bound_ = getattr(solver, "objective_gap_upper_bound_", None)
        assert self.gradient_norm_ <= self.gradient_tolerance, f"Gradient norm {self.gradient_norm_} exceeds tolerance {self.gradient_tolerance}"
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def build_regressor(family: str, config: Dict[str, Any], is_vision: bool = False, seed: int = SEED):
    """Build estimator from canonical specification config and regression family."""
    if family == "Ridge":
        return Ridge(alpha=config["alpha"], solver="cholesky", fit_intercept=True)
    elif family == "PLS":
        return PLSRegression(n_components=config["components"], scale=False, max_iter=500, tol=1e-6)
    elif family == "LinearSVR":
        if is_vision:
            return CountedPrimalSVRVisionEstimator(
                C=config["C"],
                epsilon=config["epsilon"],
                max_iter=5000,
                gradient_tolerance=1e-6,
            )
        else:
            return LinearSVR(
                C=config["C"],
                epsilon=config["epsilon"],
                loss="squared_epsilon_insensitive",
                dual=False,
                tol=1e-4,
                max_iter=5000,
                random_state=seed,
            )
    elif family == "ElasticNet":
        return ElasticNet(
            alpha=config["alpha"],
            l1_ratio=config["l1_ratio"],
            fit_intercept=True,
            max_iter=5000,
            tol=1e-4,
            random_state=seed,
        )
    elif family == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            learning_rate=config["learning_rate"],
            max_leaf_nodes=config["max_leaf_nodes"],
            l2_regularization=config["l2_regularization"],
            random_state=seed,
            early_stopping=False,
            max_iter=300,
        )
    else:
        raise ValueError(f"Unsupported regression family: {family}")


def build_metadata_pipeline(family: str, config: Dict[str, Any], seed: int = SEED) -> Pipeline:
    steps = [
        ("onehot", create_one_hot_encoder()),
        ("filter", VarianceThreshold(threshold=0.0)),
    ]
    if family != "HistGradientBoosting":
        steps.append(("scale", StandardScaler()))
    steps.append(("model", build_regressor(family, config, is_vision=False, seed=seed)))
    return Pipeline(steps)


def build_vision_pipeline(family: str, config: Dict[str, Any], seed: int = SEED) -> Pipeline:
    steps = [
        ("filter", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("model", build_regressor(family, config, is_vision=True, seed=seed)),
    ]
    return Pipeline(steps)


class EarlyFusionModel:
    """Concatenation of separately standardized metadata and vision blocks (E2 Benchmark)."""

    def __init__(self, family: str = "ElasticNet", config: Optional[Dict[str, Any]] = None, seed: int = SEED):
        self.family = family
        self.config = config or {"alpha": 0.01, "l1_ratio": 0.1}
        self.seed = seed
        self.meta_pipe = Pipeline([
            ("onehot", create_one_hot_encoder()),
            ("filter", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
        ])
        self.image_pipe = Pipeline([
            ("filter", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
        ])
        self.model = build_regressor(family, self.config, is_vision=False, seed=seed)

    def fit(self, X_meta: np.ndarray, X_image: np.ndarray, y: np.ndarray):
        zm = self.meta_pipe.fit_transform(X_meta)
        zi = self.image_pipe.fit_transform(X_image)
        z = np.concatenate([zm, zi], axis=1)
        self.model.fit(z, y)
        self.dim_meta_ = zm.shape[1]
        self.dim_image_ = zi.shape[1]
        return self

    def predict(self, X_meta: np.ndarray, X_image: np.ndarray) -> np.ndarray:
        zm = self.meta_pipe.transform(X_meta)
        zi = self.image_pipe.transform(X_image)
        z = np.concatenate([zm, zi], axis=1)
        preds = self.model.predict(z)
        return np.asarray(preds).reshape(-1)


class LateFusionModel:
    """Convex blend: y_hat = (1 - w*) * y_hat_meta + w* * y_hat_image."""

    def __init__(
        self,
        meta_pipeline: Pipeline,
        vision_pipeline: Pipeline,
        image_weight: float = 0.0,
        meta_family: str = "",
        vision_family: str = "",
        candidate_name: str = "",
        encoder: str = "",
    ):
        self.meta_pipeline = meta_pipeline
        self.vision_pipeline = vision_pipeline
        self.image_weight = float(image_weight)
        self.meta_family = meta_family
        self.vision_family = vision_family
        self.candidate_name = candidate_name
        self.encoder = encoder

    def fit(self, X_meta: np.ndarray, X_image: np.ndarray, y: np.ndarray):
        self.meta_pipeline.fit(X_meta, y)
        self.vision_pipeline.fit(X_image, y)
        return self

    def predict(self, X_meta: np.ndarray, X_image: np.ndarray) -> np.ndarray:
        pm = np.asarray(self.meta_pipeline.predict(X_meta)).reshape(-1)
        pi = np.asarray(self.vision_pipeline.predict(X_image)).reshape(-1)
        w = self.image_weight
        return (1.0 - w) * pm + w * pi


# ----------------------------------------------------------------------
# CANONICAL SPECIFICATION GRID MATCHER
# ----------------------------------------------------------------------
def get_grid_candidates(spec: Dict[str, Any], family: str, is_vision: bool = False) -> List[Dict[str, Any]]:
    """Match family name to canonical execution specification keys and return parameter dictionaries.

    Correctly maps LinearSVR to LinearSVR_vision_and_early (vision) or LinearSVR_metadata (metadata).
    Expands parameter grids if 'candidates' is not explicitly listed.
    """
    grids = spec["model_grids_and_solvers"]

    if family == "LinearSVR":
        key = "LinearSVR_vision_and_early" if is_vision else "LinearSVR_metadata"
        if key not in grids and "LinearSVR" in grids:
            key = "LinearSVR"
        if key not in grids:
            raise KeyError(f"Candidate grid for family '{family}' (is_vision={is_vision}, key={key}) not found in specification.")
        grid_container = grids[key]
    elif family in grids:
        grid_container = grids[family]
    else:
        raise KeyError(f"Candidate grid for family '{family}' (is_vision={is_vision}) not found in specification.")

    if "candidates" in grid_container:
        return grid_container["candidates"]

    grid = grid_container["grid"]
    cands = []
    if family == "HistGradientBoosting":
        for lr in grid["learning_rate"]:
            for leaves in grid["max_leaf_nodes"]:
                for l2 in grid["l2_regularization"]:
                    cands.append(dict(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2))
    elif family == "Ridge":
        for a in grid["alpha"]:
            cands.append(dict(alpha=a))
    elif family == "LinearSVR":
        for c_val in grid["C"]:
            for eps in grid["epsilon"]:
                cands.append(dict(C=c_val, epsilon=eps))
    elif family == "PLS":
        for comp in grid["components"]:
            cands.append(dict(components=comp))
    elif family == "ElasticNet":
        for a in grid["alpha"]:
            for l1 in grid["l1_ratio"]:
                cands.append(dict(alpha=a, l1_ratio=l1))
    elif family == "LateFusionBlend":
        for w in grid["image_weight"]:
            cands.append(dict(image_weight=w))
    else:
        raise KeyError(f"Unsupported family '{family}' for grid expansion.")
    return cands


# ----------------------------------------------------------------------
# A. PREFLIGHT SUBCOMMAND (Integrity Verification & Freshness Certification)
# ----------------------------------------------------------------------
def run_preflight(args=None, context: Optional[PipelineContext] = None) -> Dict[str, Any]:
    """Execute preflight checks, verifying environment, input hashes, and test target custody."""
    ctx = make_context_from_args(args, context)
    out_dir = ctx.get_out_dir()

    print("=" * 70)
    print("RUNNING PREFLIGHT VERIFICATION (v3.2)...")
    print("=" * 70)

    results = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline_version": PIPELINE_VERSION,
        "python_version": sys.version,
        "status": "PASS",
        "checks": [],
    }

    def add_check(name: str, passed: bool, details: Any):
        if not passed:
            results["status"] = "FAIL"
        results["checks"].append({
            "name": name,
            "passed": passed,
            "details": details,
        })
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {name}")

    # 0. Execution profile requirement check
    execution_profile = getattr(args, "profile", None) if args else getattr(ctx, "execution_profile", None)
    if execution_profile is not None and execution_profile not in PROFILES:
        err = f"Invalid profile '{execution_profile}'. Choose from {PROFILES}."
        print(f"ERROR: BLOCKED_EXECUTION_PROFILE_MISMATCH: {err}")
        return {"status": "BLOCKED_EXECUTION_PROFILE_MISMATCH", "error": err}
    if execution_profile is None:
        execution_profile = DEFAULT_PROFILE

    # 1. Spec & Manifest existence and fail-closed supporting files check
    spec_path = ctx.get_spec_path()
    manifest_path = ctx.get_manifest_path()
    add_check("spec_exists", spec_path.exists(), str(spec_path))
    add_check("manifest_exists", manifest_path.exists(), str(manifest_path))

    supp_ok, supp_missing = check_required_supporting_files(ctx, require_configs=False)
    add_check("supporting_files_and_schemas_verified", supp_ok, {"missing": supp_missing})

    if not spec_path.exists() or not manifest_path.exists() or not supp_ok:
        out_preflight = out_dir / "preflight_results.json"
        atomic_save_json(out_preflight, results)
        return results

    spec = read_json(spec_path)
    manifest = read_json(manifest_path)

    # 2. Runtime Environment Enforcement (Requirement B)
    env_lock_path = out_dir / "environment_lock.json"
    if not env_lock_path.exists():
        env_lock_path = ctx.package_dir / "environment_lock.json"
    env_ok = False
    env_failures = []
    if env_lock_path.exists():
        env_ok, env_failures = verify_runtime_environment(read_json(env_lock_path))
    if not env_ok:
        print(f"ERROR: BLOCKED_RUNTIME_ENVIRONMENT_MISMATCH: Runtime environment verification failed: {env_failures}")
        results["status"] = "FAIL"
    add_check("runtime_environment_verified", env_ok, {"failures": env_failures})

    # 3. execution_hash_manifest.json Structure and Integrity Enforcement (Requirement A)
    man_struct_ok, man_struct_errors = validate_execution_hash_manifest_structure(manifest)
    add_check("execution_hash_manifest_structure_verified", man_struct_ok, {"failures": man_struct_errors})

    # Record hashes in preflight record for freshness verification
    results["specification_sha256"] = sha256_file(spec_path)
    results["environment_lock_sha256"] = sha256_file(env_lock_path) if env_lock_path.exists() else None
    results["execution_hash_manifest_sha256"] = sha256_file(manifest_path)
    results["execution_hash_manifest_size_bytes"] = manifest_path.stat().st_size
    results["input_artifacts_count"] = len(manifest.get("input_artifacts", []))
    results["sealed_test_artifacts_count"] = len(manifest.get("sealed_test_artifacts", []))

    # Audit anchor verification in preflight (Requirement A)
    anchor_ok, anchor_val, anchor_err = resolve_approved_anchor(ctx, args, require_approval=False)
    if not anchor_ok:
        if str(anchor_val).startswith("BLOCKED_"):
            print(f"ERROR: {anchor_val}: {anchor_err}")
            results["status"] = anchor_val

    anchor_details = {}
    if anchor_ok:
        approved_sha = anchor_val
        results["approved_execution_manifest_sha256"] = approved_sha
        results["approved_profile"] = execution_profile
        anchor_details = {
            "status": "APPROVED",
            "approved_execution_manifest_sha256": approved_sha,
            "approved_profile": execution_profile,
            "manifest_hash_matches_anchor": True,
        }
    else:
        results["approved_execution_manifest_sha256"] = None
        results["approved_profile"] = None
        anchor_details = {
            "status": anchor_val if anchor_val.startswith("BLOCKED_") else "PENDING_EXTERNAL_APPROVAL",
            "reason": anchor_err,
        }
    add_check("audit_anchor_verified", anchor_ok, anchor_details)

    # Gate 1 Scope decisions verification in preflight (Requirement A)
    scope_ok, scope_path, scope_sha, scope_data, scope_err = resolve_and_verify_approved_scope_decisions(ctx, args, require_approval=False)
    if not scope_ok and scope_err and scope_err.startswith("BLOCKED_"):
        print(f"ERROR: {scope_err}")
        results["status"] = scope_err

    scope_passed = scope_ok and (scope_sha is not None)
    if scope_passed:
        results["approved_scope_decisions_sha256"] = scope_sha
        scope_details = {
            "status": "APPROVED",
            "approved_scope_decisions_sha256": scope_sha,
            "scope_path": safe_relpath(scope_path, ctx.root_dir),
            "governance_status": "USER_APPROVED",
        }
    else:
        results["approved_scope_decisions_sha256"] = None
        scope_details = {
            "status": scope_err if (scope_err and scope_err.startswith("BLOCKED_")) else "PENDING_USER_SCOPE_APPROVAL",
            "reason": scope_err or "Approved Gate 1 scope decisions file not provided or pending",
        }
    add_check("approved_scope_decisions_verified", scope_passed, scope_details)

    # 4. Input hashes verification
    hash_failures = []
    total_verified = 0
    for art in manifest.get("input_artifacts", []):
        p = resolve_manifest_path(art["path"], root_dir=ctx.root_dir)
        if not p.exists():
            hash_failures.append({"path": art["path"], "error": "file_not_found"})
            continue
        cur_sha = sha256_file(p)
        if cur_sha != art["sha256"]:
            hash_failures.append({
                "path": art["path"],
                "expected": art["sha256"],
                "actual": cur_sha,
            })
        else:
            total_verified += 1
    if len(hash_failures) > 0 or total_verified != 31:
        print(f"ERROR: BLOCKED_EXECUTION_MANIFEST_TAMPERED: Input artifact hash verification failed: {hash_failures}")
        results["status"] = "FAIL"
    add_check("input_hashes_verified", len(hash_failures) == 0 and total_verified == 31, {
        "total_verified": total_verified,
        "failures": hash_failures,
    })

    # 5. Sealed test target custody verification (STAT ONLY)
    sealed_failures = []
    total_sealed = 0
    for art in manifest.get("sealed_test_artifacts", []):
        p = resolve_manifest_path(art["path"], root_dir=ctx.root_dir)
        try:
            st = os.stat(p)
            if st.st_size <= 0:
                sealed_failures.append({"path": art["path"], "error": "empty_file"})
            elif st.st_size != art["size_bytes"]:
                sealed_failures.append({
                    "path": art["path"],
                    "expected_size": art["size_bytes"],
                    "actual_size": st.st_size,
                })
            else:
                total_sealed += 1
        except Exception as e:
            sealed_failures.append({"path": art["path"], "error": str(e)})

    add_check("sealed_test_targets_custody_verified", len(sealed_failures) == 0 and total_sealed == 4, {
        "total_sealed": total_sealed,
        "failures": sealed_failures,
        "verification_method": "File presence and positive stat size verified without opening or reading file bytes.",
    })

    # 6. Feature matrices and manifest token counts
    reg_path = ctx.rev_dir / "Encoder Development Registry 20260909 v4.json"
    reg = read_json(reg_path)
    enc_checks = {}
    enc_all_ok = True
    for enc_key in ["dinov2_fullframe", "clip_native", "siglip2"]:
        enc_checks[enc_key] = {}
        for coll in COLLECTIONS:
            info = reg[enc_key]["collections"][coll]
            m_path = resolve_manifest_path(info["matrix"], root_dir=ctx.root_dir)
            man_path = resolve_manifest_path(info["manifest"], root_dir=ctx.root_dir)
            mat = np.load(m_path, mmap_mode="r")
            man_lines = count_lines(man_path)
            expected_tokens = 9366 if coll == "BAYC" else 12459
            dim_ok = (mat.shape[1] == info["dimensions"])
            rows_ok = (mat.shape[0] == man_lines == expected_tokens)
            tokens = set()
            with man_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        tokens.add(json.loads(line)["token_id"])
            dups_ok = (len(tokens) == man_lines)
            match_ok = dim_ok and rows_ok and dups_ok
            if not match_ok:
                enc_all_ok = False
            enc_checks[enc_key][coll] = {
                "expected_dim": info["dimensions"],
                "actual_dim": mat.shape[1],
                "feature_rows": mat.shape[0],
                "manifest_tokens": man_lines,
                "no_duplicate_tokens": dups_ok,
                "match": match_ok,
            }
    add_check("encoder_feature_dimensions_and_tokens_verified", enc_all_ok, enc_checks)

    # 7. Dev and test trade counts
    trade_counts = {
        "original_dev": {"BAYC": 58635, "MAYC": 113823, "total": 172458},
        "sensitivity_dev": {"BAYC": 58626, "MAYC": 113811, "total": 172437},
        "test_targets": {"BAYC": 5120, "MAYC": 13019, "total": 18139},
    }
    counts_ok = True
    for c in COLLECTIONS:
        f_orig = ctx.rev_dir / "target_pipeline_20260909" / f"{c.lower()}_development_targets.jsonl"
        cnt_orig = count_lines(f_orig)
        if cnt_orig != trade_counts["original_dev"][c]:
            counts_ok = False
        f_sens = ctx.rev_dir / "one_wei_sensitivity_targets_20260909" / f"{c.lower()}_development_targets.jsonl"
        cnt_sens = count_lines(f_sens)
        if cnt_sens != trade_counts["sensitivity_dev"][c]:
            counts_ok = False
    add_check("development_and_test_trade_counts_verified", counts_ok, trade_counts)

    # 8. Development target temporal boundary verification
    dev_boundary_ok = True
    for c in COLLECTIONS:
        f_orig = ctx.rev_dir / "target_pipeline_20260909" / f"{c.lower()}_development_targets.jsonl"
        with f_orig.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["time"] >= "2025-01-01":
                        dev_boundary_ok = False
                        break
    add_check("development_target_time_boundary_verified", dev_boundary_ok, "All development records precede 2025-01-01 00:00:00 UTC")

    # 9. Regression solvers verification
    solvers_ok = True
    solver_info = {}
    if CountedPrimalSVR is None:
        solvers_ok = False
        solver_info["CountedPrimalSVR"] = "import_failed"
    else:
        solver_info["CountedPrimalSVR"] = "available_Newton_norm_cert_1e-6"

    try:
        svr = LinearSVR(loss="squared_epsilon_insensitive", dual=False, tol=1e-4, max_iter=5000, random_state=SEED)
        solver_info["LinearSVR"] = "initialized_max_iter_5000_tol_1e-4"
    except Exception as e:
        solvers_ok = False
        solver_info["LinearSVR"] = str(e)

    add_check("regression_solvers_verified", solvers_ok, solver_info)

    # Final status determination:
    # If all other checks passed:
    #   if anchor_ok -> PASS
    #   if not anchor_ok -> specific blocked code if anchor explicitly blocked, else PENDING_EXTERNAL_APPROVAL
    # Else: FAIL
    exempt_checks = {"audit_anchor_verified", "approved_scope_decisions_verified"}
    other_checks_passed = all(c["passed"] for c in results["checks"] if c["name"] not in exempt_checks)
    if not other_checks_passed:
        results["status"] = "FAIL"
    elif not anchor_ok:
        if str(anchor_val).startswith("BLOCKED_"):
            results["status"] = anchor_val
        else:
            results["status"] = "PENDING_EXTERNAL_APPROVAL"
            results["notes"] = "All preflight integrity and environment checks completed successfully; overall status is PENDING_EXTERNAL_APPROVAL awaiting external auditor approval."
    elif not scope_passed:
        if scope_err and str(scope_err).startswith("BLOCKED_"):
            results["status"] = scope_err
        else:
            results["status"] = "PENDING_USER_SCOPE_APPROVAL"
            results["notes"] = "Preflight integrity and audit anchor checks passed; overall status is PENDING_USER_SCOPE_APPROVAL awaiting user-approved Gate 1 scope decisions document."
    else:
        results["status"] = "PASS"

    out_preflight = out_dir / "preflight_results.json"
    atomic_save_json(out_preflight, results)
    print(f"\nPreflight finished with status: {results['status']}. Written to {out_preflight.name}.")

    if results["status"] == "FAIL":
        print("ERROR: Preflight failed. Subsequent stages are blocked.")

    return results
def check_preflight_freshness(ctx: PipelineContext, approved_sha: Optional[str] = None) -> Tuple[bool, str, str]:
    """Verify that preflight was executed, passed, and matches current environment, manifest, specification, and approved anchor.

    Returns:
        (is_fresh, error_message, status_code)
    """
    out_dir = ctx.get_out_dir()
    preflight_path = out_dir / "preflight_results.json"
    if not preflight_path.exists():
        preflight_path = ctx.package_dir / "preflight_results.json"
    if not preflight_path.exists():
        return False, "preflight_results.json not found. Run 'pipeline.py preflight' first.", "BLOCKED_PREFLIGHT_NOT_FOUND"

    preflight = read_json(preflight_path)
    pf_status = preflight.get("status")
    if pf_status == "PENDING_EXTERNAL_APPROVAL":
        return False, "preflight_results.json recorded status is PENDING_EXTERNAL_APPROVAL. External auditor approval required before execution.", "BLOCKED_PREFLIGHT_PENDING_APPROVAL"
    if pf_status != "PASS":
        return False, f"preflight_results.json recorded status is {pf_status} (not PASS).", "BLOCKED_PREFLIGHT_FAILED"

    # 1. Fail-closed supporting files check
    supp_ok, supp_missing = check_required_supporting_files(ctx, require_configs=False)
    if not supp_ok:
        if any("freeze_manifest_schema" in m or "evaluation_custody_schema" in m for m in supp_missing):
            return False, f"Schema files missing: {supp_missing}", "BLOCKED_SCHEMA_MISSING"
        return False, f"Required supporting files missing: {supp_missing}", "BLOCKED_REQUIRED_FILE_MISSING"

    # 2. Verify runtime environment
    env_lock_path = out_dir / "environment_lock.json"
    if not env_lock_path.exists():
        env_lock_path = ctx.package_dir / "environment_lock.json"
    if env_lock_path.exists():
        env_ok, env_errs = verify_runtime_environment(read_json(env_lock_path))
        if not env_ok:
            return False, f"Runtime environment differs from environment_lock.json: {env_errs}", "BLOCKED_RUNTIME_ENVIRONMENT_MISMATCH"

    # 3. Verify specification hash freshness
    spec_path = ctx.get_spec_path()
    if spec_path.exists():
        cur_spec_sha = sha256_file(spec_path)
        recorded_spec_sha = preflight.get("specification_sha256")
        if recorded_spec_sha and cur_spec_sha != recorded_spec_sha:
            return False, f"Specification has changed since preflight! (Recorded: {recorded_spec_sha[:12]}, Current: {cur_spec_sha[:12]})", "BLOCKED_PREFLIGHT_STALE"

    # 4. Verify environment lock freshness
    if env_lock_path.exists():
        cur_env_sha = sha256_file(env_lock_path)
        recorded_env_sha = preflight.get("environment_lock_sha256")
        if recorded_env_sha and cur_env_sha != recorded_env_sha:
            return False, "environment_lock.json has changed since preflight.", "BLOCKED_PREFLIGHT_STALE"

    # 5. Verify execution hash manifest freshness (Requirement A)
    manifest_path = ctx.get_manifest_path()
    if manifest_path.exists():
        cur_man_sha = sha256_file(manifest_path)
        recorded_man_sha = preflight.get("execution_hash_manifest_sha256")
        if recorded_man_sha and cur_man_sha != recorded_man_sha:
            return False, f"execution_hash_manifest.json has changed since preflight! (Recorded: {recorded_man_sha[:12]}, Current: {cur_man_sha[:12]})", "BLOCKED_EXECUTION_MANIFEST_TAMPERING"

        # Check against external approved anchor
        if approved_sha:
            if cur_man_sha != approved_sha:
                return False, f"execution_hash_manifest.json does not match external approved anchor! (Approved: {approved_sha[:12]}, Current: {cur_man_sha[:12]})", "BLOCKED_APPROVED_MANIFEST_HASH_MISMATCH"
            if recorded_man_sha and recorded_man_sha != approved_sha:
                return False, f"preflight_results.json manifest hash does not match external approved anchor! (Approved: {approved_sha[:12]}, Preflight: {recorded_man_sha[:12]})", "BLOCKED_APPROVED_MANIFEST_HASH_MISMATCH"

    return True, "Preflight is fresh and valid.", "PASS"
def check_immutable_state_transitions(stage_name: str, ctx: PipelineContext) -> Optional[Dict[str, Any]]:
    """Enforce immutable state machine transitions. Block invalid operations on frozen or evaluated pipelines."""
    out_dir = ctx.get_out_dir()

    # Check 1: Evaluation lifecycle check
    custody_file = out_dir / "evaluation_custody.json"
    if custody_file.exists():
        custody = read_json(custody_file)
        c_status = custody.get("status")
        if c_status in ["EVALUATION_STARTED", "EVALUATION_FAILED", "EVALUATION_COMPLETED"]:
            if stage_name in ["tune", "refit", "freeze"]:
                print(f"CRITICAL: Immutable state violation! Stage '{stage_name}' blocked because evaluation lifecycle is '{c_status}'.")
                return {
                    "status": "BLOCKED_EVALUATION_LIFECYCLE_ACTIVE",
                    "error": f"Cannot execute {stage_name} when evaluation custody is {c_status}.",
                }

    # Check 2: Freeze manifest check
    freeze_path = out_dir / "freeze_manifest.json"
    if freeze_path.exists():
        if stage_name in ["tune", "refit"]:
            print(f"CRITICAL: Immutable state violation! Stage '{stage_name}' blocked because pipeline is already frozen (freeze_manifest.json exists).")
            return {
                "status": "BLOCKED_PIPELINE_ALREADY_FROZEN",
                "error": f"Cannot execute {stage_name} after freeze_manifest.json has been created.",
            }
        elif stage_name == "freeze":
            print("CRITICAL: Immutable state violation! freeze_manifest.json already exists in target directory. Overwriting is strictly prohibited.")
            return {
                "status": "BLOCKED_FREEZE_ALREADY_EXISTS",
                "error": "freeze_manifest.json already exists. Re-freezing requires a new explicit version directory.",
            }

    return None


# ----------------------------------------------------------------------
# PRODUCTION DATA LOADING HELPER
# ----------------------------------------------------------------------
def default_load_dataset(sample: str, collection: str, encoder: str, ctx: PipelineContext) -> Dict[str, Any]:
    """Load production development targets, normalized traits, and image features."""
    target_dir = "target_pipeline_20260909" if sample == "original_v1" else "one_wei_sensitivity_targets_20260909"
    split_dir = "image_validation_20260909" if sample == "original_v1" else "one_wei_sensitivity_targets_20260909"

    target_path = ctx.rev_dir / target_dir / f"{collection.lower()}_development_targets.jsonl"
    rows = [json.loads(line) for line in target_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(r["split"] == "development" and r["time"] < "2025-01-01" for r in rows)

    reg = read_json(ctx.rev_dir / "Encoder Development Registry 20260909 v4.json")
    enc_info = reg[encoder]["collections"][collection]
    feat_matrix = np.load(ctx.root_dir / enc_info["matrix"], allow_pickle=False).astype(np.float64)
    manifest_rows = [json.loads(line) for line in (ctx.root_dir / enc_info["manifest"]).read_text(encoding="utf-8").splitlines() if line.strip()]

    feature_tokens = np.empty(len(manifest_rows), dtype=np.int64)
    for mr in manifest_rows:
        feature_tokens[mr["feature_row_idx"]] = mr["token_id"]
    token_to_idx = {tok: idx for idx, tok in enumerate(feature_tokens)}

    meta_rows = [json.loads(line) for line in (ctx.rev_dir / "data_audit_20260908" / "metadata_normalized.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    meta_dict = {(mr["collection"], int(mr["token_id"])): mr for mr in meta_rows}

    cols = FEATURES + (["generation"] if collection == "MAYC" else [])
    X_meta = np.asarray([[meta_dict[collection, int(t)][c] for c in cols] for t in feature_tokens], dtype=object)

    split_manifest = read_json(ctx.rev_dir / split_dir / "image_ready_split_manifest_v1.json")[collection]

    data = {
        "sample": sample,
        "collection": collection,
        "encoder": encoder,
        "columns": cols,
        "X_meta": X_meta,
        "features_image": feat_matrix,
        "feature_tokens": feature_tokens,
        "y": np.asarray([r["y_log_relative_price"] for r in rows], dtype=np.float64),
        "tokens": np.asarray([r["token_id"] for r in rows], dtype=np.int64),
        "rowids": np.asarray([r["source_row"] for r in rows], dtype=np.int64),
        "times": np.asarray([r["time"] for r in rows], dtype=object),
        "feature_indices": np.asarray([token_to_idx[r["token_id"]] for r in rows], dtype=np.int64),
        "split": split_manifest,
    }
    data["rowmap"] = {v: i for i, v in enumerate(data["rowids"])}
    assert set(data["rowmap"]) == set(split_manifest["development_source_rows"])
    return data


# ----------------------------------------------------------------------
# B. TUNE SUBCOMMAND (2024 Q2-Q4 Sequential Component & Weight Tuning)
# ----------------------------------------------------------------------
def run_tune(args=None, context: Optional[PipelineContext] = None) -> Dict[str, Any]:
    """Execute 2024 Q2-Q4 expanding prefix cross-validation for all model components and late weights."""
    ctx = make_context_from_args(args, context)
    out_dir = ctx.get_out_dir()

    print("=" * 70)
    print("SUBCOMMAND: tune (2024 Q2-Q4 Expanding Prefix Cross-Validation v3.3)")
    print("=" * 70)

    # 1. State transition check: cannot tune if frozen or evaluated
    state_err = check_immutable_state_transitions("tune", ctx)
    if state_err:
        return state_err

    # 2. Enforce Approved External Anchor & Preflight Freshness (Requirements A & 5)
    anchor_ok, anchor_val, anchor_err = resolve_approved_anchor(ctx, args)
    if not anchor_ok:
        print(f"ERROR: Approved anchor verification failed: {anchor_err}")
        return {"status": anchor_val, "error": anchor_err}
    approved_sha = anchor_val

    fresh_ok, fresh_msg, code_status = check_preflight_freshness(ctx, approved_sha=approved_sha)
    if not fresh_ok:
        print(f"ERROR: Preflight freshness check failed: {fresh_msg}")
        return {"status": code_status, "error": fresh_msg}

    execute = getattr(args, "execute", False)
    sample_arg = getattr(args, "sample", "all")
    collection_arg = getattr(args, "collection", "all")
    profile = getattr(args, "profile", None) if args else getattr(ctx, "execution_profile", None)
    if not profile:
        return {"status": "BLOCKED_PROFILE_REQUIRED", "error": "--profile is required ('recommended_16' or 'confirmatory_12')"}

    samples_to_run = SAMPLES if sample_arg == "all" else [sample_arg]
    collections_to_run = COLLECTIONS if collection_arg == "all" else [collection_arg]

    spec_path = ctx.get_spec_path()
    if not spec_path.exists():
        print(f"ERROR: Specification file not found: {spec_path}")
        return {"status": "BLOCKED_SPEC_NOT_FOUND", "error": f"Specification file not found: {spec_path}"}
    spec = read_json(spec_path)

    tuning_schedule = spec["tuning_schedule_2024"]["quarters"]

    if not execute:
        print("DRY-RUN MODE: Tuning schedule, grids, and candidate mappings inspected.")
        print(f"  Target samples: {samples_to_run}")
        print(f"  Target collections: {collections_to_run}")
        print(f"  Execution profile: {profile}")
        print(f"  Tuning schedule: {[q['quarter'] for q in tuning_schedule]}")
        print("To execute numerical fitting, run with: --execute")
        return {
            "status": "DRY_RUN",
            "samples": samples_to_run,
            "collections": collections_to_run,
            "profile": profile,
            "quarters": [q["quarter"] for q in tuning_schedule],
        }

    # Gate 1 User Scope Approval Verification (v3.3.6)
    scope_ok, scope_path, scope_sha, scope_data, scope_err = resolve_and_verify_approved_scope_decisions(ctx, args, require_approval=True)
    if not scope_ok:
        return {"status": scope_err, "error": f"Gate 1 Scope Approval failed: {scope_err}"}
    profile = (getattr(args, "profile", None) if args else None) or (getattr(ctx, "execution_profile", None)) or scope_data.get("execution_profile", DEFAULT_PROFILE)
    profile = (getattr(args, "profile", None) if args else None) or (getattr(ctx, "execution_profile", None)) or scope_data.get("execution_profile", DEFAULT_PROFILE)

    print("EXECUTING NUMERICAL 2024 Q2-Q4 TUNING (METADATA + VISION + LATE WEIGHT + BENCHMARK)...")
    selected_configs = {}
    tuning_summary = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "profile": profile, "runs": {}}

    for sample in samples_to_run:
        selected_configs[sample] = {}
        for coll in collections_to_run:
            print(f"\n--- Tuning {sample} | {coll} ---")
            cand_info = spec["selected_candidates"]["primary" if sample == "original_v1" else "sensitivity_exact_one_wei"][coll]
            meta_fam = cand_info["metadata_family"]
            aug_cand = cand_info["augmented_candidate"]
            aug_enc = cand_info["augmented_encoder"]

            aug_parts = aug_cand.split("_")
            assert aug_parts[0] == "late", f"Expected late candidate format, got {aug_cand}"
            aug_meta_fam = aug_parts[1]
            aug_img_fam = aug_parts[2]

            data = ctx.load_dataset(sample=sample, collection=coll, encoder=aug_enc)
            times = data["times"]
            y = data["y"]
            X_meta_all = data["X_meta"][data["feature_indices"]]
            X_image_all = data["features_image"][data["feature_indices"]]

            # 1. Evaluate Metadata Model Grid across 2024 Q2-Q4
            meta_cands = get_grid_candidates(spec, meta_fam, is_vision=False)
            meta_scores = {i: {"config": c, "valid": True, "sse": 0.0, "n": 0, "quarters": {}} for i, c in enumerate(meta_cands)}
            meta_oof_preds = {i: np.zeros(len(y), dtype=np.float64) for i in range(len(meta_cands))}

            for q in tuning_schedule:
                q_name = q["quarter"]
                begin, end = q["validation_start_inclusive"], q["validation_end_exclusive"]
                tr = np.flatnonzero(times < begin)
                va = np.flatnonzero((times >= begin) & (times < end))
                assert max(times[tr]) < min(times[va]), "Time boundary violation in quarter split"

                for idx, c in enumerate(meta_cands):
                    pipe = build_metadata_pipeline(meta_fam, c, seed=SEED)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        pipe.fit(X_meta_all[tr], y[tr])
                        preds = pipe.predict(X_meta_all[va])

                    is_valid = np.isfinite(preds).all() and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                    if is_valid:
                        sse_q = float(np.sum((y[va] - preds) ** 2))
                        meta_scores[idx]["sse"] += sse_q
                        meta_scores[idx]["n"] += len(va)
                        meta_scores[idx]["quarters"][q_name] = {"sse": sse_q, "n": len(va), "mse": sse_q / len(va)}
                        meta_oof_preds[idx][va] = preds
                    else:
                        meta_scores[idx]["valid"] = False

            valid_meta = [s for s in meta_scores.values() if s["valid"] and s["n"] > 0]
            assert valid_meta, f"No valid metadata config for {sample} {coll}"
            for s in valid_meta:
                s["mse"] = s["sse"] / s["n"]

            best_meta_mse = min(s["mse"] for s in valid_meta)
            tied_meta = [s for s in valid_meta if s["mse"] <= best_meta_mse + 1e-12]
            if meta_fam == "HistGradientBoosting":
                chosen_meta = sorted(tied_meta, key=lambda s: (s["config"]["max_leaf_nodes"], -s["config"]["l2_regularization"], s["config"]["learning_rate"]))[0]
            elif meta_fam == "Ridge":
                chosen_meta = sorted(tied_meta, key=lambda s: s["config"]["alpha"], reverse=True)[0]
            else:
                chosen_meta = tied_meta[0]

            print(f"  [1/4] Selected Metadata {meta_fam}: {chosen_meta['config']} (pooled MSE = {chosen_meta['mse']:.6f})")

            # 2. Evaluate Augmented Metadata Submodel Grid across 2024 Q2-Q4
            aug_meta_cands = get_grid_candidates(spec, aug_meta_fam, is_vision=False)
            aug_meta_scores = {i: {"config": c, "valid": True, "sse": 0.0, "n": 0, "quarters": {}} for i, c in enumerate(aug_meta_cands)}
            aug_meta_oof = {i: np.zeros(len(y), dtype=np.float64) for i in range(len(aug_meta_cands))}

            for q in tuning_schedule:
                q_name = q["quarter"]
                begin, end = q["validation_start_inclusive"], q["validation_end_exclusive"]
                tr = np.flatnonzero(times < begin)
                va = np.flatnonzero((times >= begin) & (times < end))

                for idx, c in enumerate(aug_meta_cands):
                    pipe = build_metadata_pipeline(aug_meta_fam, c, seed=SEED)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        pipe.fit(X_meta_all[tr], y[tr])
                        preds = pipe.predict(X_meta_all[va])

                    is_valid = np.isfinite(preds).all() and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                    if is_valid:
                        sse_q = float(np.sum((y[va] - preds) ** 2))
                        aug_meta_scores[idx]["sse"] += sse_q
                        aug_meta_scores[idx]["n"] += len(va)
                        aug_meta_scores[idx]["quarters"][q_name] = {"sse": sse_q, "n": len(va), "mse": sse_q / len(va)}
                        aug_meta_oof[idx][va] = preds
                    else:
                        aug_meta_scores[idx]["valid"] = False

            valid_am = [s for s in aug_meta_scores.values() if s["valid"] and s["n"] > 0]
            assert valid_am, f"No valid augmented metadata config for {sample} {coll}"
            for s in valid_am:
                s["mse"] = s["sse"] / s["n"]

            best_am_mse = min(s["mse"] for s in valid_am)
            tied_am = [s for s in valid_am if s["mse"] <= best_am_mse + 1e-12]
            if aug_meta_fam == "LinearSVR":
                chosen_aug_meta = sorted(tied_am, key=lambda s: (s["config"]["C"], -s["config"]["epsilon"]))[0]
            elif aug_meta_fam == "Ridge":
                chosen_aug_meta = sorted(tied_am, key=lambda s: s["config"]["alpha"], reverse=True)[0]
            else:
                chosen_aug_meta = tied_am[0]

            # 3. Evaluate Augmented Vision Submodel Grid across 2024 Q2-Q4
            aug_img_cands = get_grid_candidates(spec, aug_img_fam, is_vision=True)
            aug_img_scores = {i: {"config": c, "valid": True, "sse": 0.0, "n": 0, "quarters": {}} for i, c in enumerate(aug_img_cands)}
            aug_img_oof = {i: np.zeros(len(y), dtype=np.float64) for i in range(len(aug_img_cands))}

            for q in tuning_schedule:
                q_name = q["quarter"]
                begin, end = q["validation_start_inclusive"], q["validation_end_exclusive"]
                tr = np.flatnonzero(times < begin)
                va = np.flatnonzero((times >= begin) & (times < end))

                for idx, c in enumerate(aug_img_cands):
                    pipe = build_vision_pipeline(aug_img_fam, c, seed=SEED)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        pipe.fit(X_image_all[tr], y[tr])
                        preds = pipe.predict(X_image_all[va])

                    is_valid = np.isfinite(preds).all() and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                    if aug_img_fam == "PLS":
                        pls_m = pipe.named_steps["model"]
                        is_valid = is_valid and len(pls_m.n_iter_) == c["components"] and np.isfinite(pls_m.coef_).all()

                    if is_valid:
                        sse_q = float(np.sum((y[va] - preds) ** 2))
                        aug_img_scores[idx]["sse"] += sse_q
                        aug_img_scores[idx]["n"] += len(va)
                        aug_img_scores[idx]["quarters"][q_name] = {"sse": sse_q, "n": len(va), "mse": sse_q / len(va)}
                        aug_img_oof[idx][va] = preds
                    else:
                        aug_img_scores[idx]["valid"] = False

            valid_ai = [s for s in aug_img_scores.values() if s["valid"] and s["n"] > 0]
            assert valid_ai, f"No valid vision config for {sample} {coll}"
            for s in valid_ai:
                s["mse"] = s["sse"] / s["n"]

            best_ai_mse = min(s["mse"] for s in valid_ai)
            tied_ai = [s for s in valid_ai if s["mse"] <= best_ai_mse + 1e-12]
            if aug_img_fam == "PLS":
                chosen_aug_img = sorted(tied_ai, key=lambda s: s["config"]["components"])[0]
            elif aug_img_fam == "LinearSVR":
                chosen_aug_img = sorted(tied_ai, key=lambda s: (s["config"]["C"], -s["config"]["epsilon"]))[0]
            elif aug_img_fam == "ElasticNet":
                chosen_aug_img = sorted(tied_ai, key=lambda s: (s["config"]["alpha"], s["config"]["l1_ratio"]), reverse=True)[0]
            elif aug_img_fam == "Ridge":
                chosen_aug_img = sorted(tied_ai, key=lambda s: s["config"]["alpha"], reverse=True)[0]
            else:
                chosen_aug_img = tied_ai[0]

            print(f"  [2/4] Selected Vision {aug_img_fam}: {chosen_aug_img['config']} (pooled MSE = {chosen_aug_img['mse']:.6f})")

            # 4. Tune Late Fusion Weight w in [0.0, 0.1, ..., 1.0]
            chosen_am_idx = [i for i, s in aug_meta_scores.items() if s["config"] == chosen_aug_meta["config"]][0]
            chosen_ai_idx = [i for i, s in aug_img_scores.items() if s["config"] == chosen_aug_img["config"]][0]

            p_am_oof = aug_meta_oof[chosen_am_idx]
            p_ai_oof = aug_img_oof[chosen_ai_idx]

            all_va_indices = np.flatnonzero((times >= tuning_schedule[0]["validation_start_inclusive"]) & (times < tuning_schedule[-1]["validation_end_exclusive"]))
            y_va_all = y[all_va_indices]
            p_am_va = p_am_oof[all_va_indices]
            p_ai_va = p_ai_oof[all_va_indices]

            weight_scores = []
            for w in LATE_WEIGHTS:
                p_blend = (1.0 - w) * p_am_va + w * p_ai_va
                w_sse = float(np.sum((y_va_all - p_blend) ** 2))
                w_mse = w_sse / len(all_va_indices)
                weight_scores.append({"weight": w, "sse": w_sse, "mse": w_mse})

            best_w_mse = min(ws["mse"] for ws in weight_scores)
            tied_w = [ws for ws in weight_scores if ws["mse"] <= best_w_mse + 1e-12]
            chosen_w_entry = sorted(tied_w, key=lambda ws: ws["weight"])[0]
            chosen_w = chosen_w_entry["weight"]

            print(f"  [3/4] Selected Late Fusion Weight w: {chosen_w} (pooled MSE = {chosen_w_entry['mse']:.6f})")

            # 5. Tune Mandatory Early Benchmark (ElasticNet concatenated)
            bench_cands = get_grid_candidates(spec, "ElasticNet", is_vision=False)
            bench_scores = {i: {"config": c, "valid": True, "sse": 0.0, "n": 0, "quarters": {}} for i, c in enumerate(bench_cands)}

            for q in tuning_schedule:
                q_name = q["quarter"]
                begin, end = q["validation_start_inclusive"], q["validation_end_exclusive"]
                tr = np.flatnonzero(times < begin)
                va = np.flatnonzero((times >= begin) & (times < end))

                for idx, c in enumerate(bench_cands):
                    model_bench = EarlyFusionModel(family="ElasticNet", config=c, seed=SEED)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model_bench.fit(X_meta_all[tr], X_image_all[tr], y[tr])
                        preds = model_bench.predict(X_meta_all[va], X_image_all[va])

                    is_valid = np.isfinite(preds).all() and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                    if is_valid:
                        sse_q = float(np.sum((y[va] - preds) ** 2))
                        bench_scores[idx]["sse"] += sse_q
                        bench_scores[idx]["n"] += len(va)
                        bench_scores[idx]["quarters"][q_name] = {"sse": sse_q, "n": len(va), "mse": sse_q / len(va)}
                    else:
                        bench_scores[idx]["valid"] = False

            valid_bench = [s for s in bench_scores.values() if s["valid"] and s["n"] > 0]
            assert valid_bench, f"No valid benchmark config for {sample} {coll}"
            for s in valid_bench:
                s["mse"] = s["sse"] / s["n"]

            best_bench_mse = min(s["mse"] for s in valid_bench)
            tied_bench = [s for s in valid_bench if s["mse"] <= best_bench_mse + 1e-12]
            chosen_bench = sorted(tied_bench, key=lambda s: (s["config"]["alpha"], s["config"]["l1_ratio"]), reverse=True)[0]

            print(f"  [4/4] Selected Early ElasticNet Benchmark: {chosen_bench['config']} (pooled MSE = {chosen_bench['mse']:.6f})")

            # Store selected configurations
            coll_selected = {
                "sample": sample,
                "collection": coll,
                "metadata": {
                    "family": meta_fam,
                    "config": chosen_meta["config"],
                    "pooled_mse": chosen_meta["mse"],
                },
                "augmented": {
                    "candidate_name": aug_cand,
                    "encoder": aug_enc,
                    "image_weight": float(chosen_w),
                    "pooled_mse": chosen_w_entry["mse"],
                    "metadata_model": {
                        "family": aug_meta_fam,
                        "config": chosen_aug_meta["config"],
                    },
                    "image_model": {
                        "family": aug_img_fam,
                        "config": chosen_aug_img["config"],
                    },
                },
                "mandatory_benchmark": {
                    "family": "ElasticNet",
                    "config": chosen_bench["config"],
                    "pooled_mse": chosen_bench["mse"],
                },
                "exploratory_augmented": {},
            }

            # 6. Tune Exploratory Augmented Candidates if profile is recommended_16 (Requirement C)
            if sample == "original_v1" and profile == "recommended_16":
                exp_map = spec.get("selected_candidates", {}).get("primary", {}).get(coll, {}).get("exploratory_candidates", {})
                for exp_enc, exp_cand_info in exp_map.items():
                    exp_cand_name = exp_cand_info["candidate_name"]
                    exp_img_fam = exp_cand_info["image_family"]
                    exp_meta_fam = exp_cand_info["metadata_family"]

                    data_exp = ctx.load_dataset(sample=sample, collection=coll, encoder=exp_enc)
                    X_img_exp = data_exp["features_image"][data_exp["feature_indices"]]

                    exp_img_cands = get_grid_candidates(spec, exp_img_fam, is_vision=True)
                    exp_img_scores = {i: {"config": c, "valid": True, "sse": 0.0, "n": 0, "quarters": {}} for i, c in enumerate(exp_img_cands)}
                    exp_img_oof = {i: np.zeros(len(y), dtype=np.float64) for i in range(len(exp_img_cands))}

                    for q in tuning_schedule:
                        q_name = q["quarter"]
                        begin, end = q["validation_start_inclusive"], q["validation_end_exclusive"]
                        tr = np.flatnonzero(times < begin)
                        va = np.flatnonzero((times >= begin) & (times < end))

                        for idx, c in enumerate(exp_img_cands):
                            pipe = build_vision_pipeline(exp_img_fam, c, seed=SEED)
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                pipe.fit(X_img_exp[tr], y[tr])
                                preds = pipe.predict(X_img_exp[va])

                            is_valid = np.isfinite(preds).all() and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                            if is_valid:
                                sse_q = float(np.sum((y[va] - preds) ** 2))
                                exp_img_scores[idx]["sse"] += sse_q
                                exp_img_scores[idx]["n"] += len(va)
                                exp_img_scores[idx]["quarters"][q_name] = {"sse": sse_q, "n": len(va), "mse": sse_q / len(va)}
                                exp_img_oof[idx][va] = preds
                            else:
                                exp_img_scores[idx]["valid"] = False

                    valid_e_img = [s for s in exp_img_scores.values() if s["valid"] and s["n"] > 0]
                    assert valid_e_img, f"No valid vision config for exploratory {exp_enc} {coll}"
                    for s in valid_e_img:
                        s["mse"] = s["sse"] / s["n"]
                    best_e_mse = min(s["mse"] for s in valid_e_img)
                    chosen_e_img = [s for s in valid_e_img if s["mse"] <= best_e_mse + 1e-12][0]

                    chosen_e_idx = [i for i, s in exp_img_scores.items() if s["config"] == chosen_e_img["config"]][0]
                    p_ei_oof = exp_img_oof[chosen_e_idx]
                    p_ei_va = p_ei_oof[all_va_indices]

                    exp_w_scores = []
                    for w in LATE_WEIGHTS:
                        p_blend = (1.0 - w) * p_am_va + w * p_ei_va
                        w_sse = float(np.sum((y_va_all - p_blend) ** 2))
                        exp_w_scores.append({"weight": w, "sse": w_sse, "mse": w_sse / len(all_va_indices)})
                    best_ew_mse = min(ws["mse"] for ws in exp_w_scores)
                    chosen_ew = sorted([ws for ws in exp_w_scores if ws["mse"] <= best_ew_mse + 1e-12], key=lambda ws: ws["weight"])[0]["weight"]

                    exp_meta_cfg = chosen_aug_meta["config"] if exp_meta_fam == aug_meta_fam else chosen_meta["config"]
                    coll_selected["exploratory_augmented"][exp_enc] = {
                        "candidate_name": exp_cand_name,
                        "encoder": exp_enc,
                        "image_weight": float(chosen_ew),
                        "pooled_mse": float(best_ew_mse),
                        "metadata_model": {"family": exp_meta_fam, "config": exp_meta_cfg},
                        "image_model": {"family": exp_img_fam, "config": chosen_e_img["config"]},
                    }
                    print(f"  [Exploratory] Selected {exp_enc}: {exp_cand_name} (weight={chosen_ew}, pooled MSE={best_ew_mse:.6f})")

            selected_configs[sample][coll] = coll_selected

    # Gate 1 Scope & Provenance Binding (v3.3.6)
    selected_configs["_metadata"] = {
        "execution_profile": profile,
        "pipeline_version": PIPELINE_VERSION,
        "approved_scope_decisions_sha256": scope_sha,
        "approved_scope_decisions_path": safe_relpath(scope_path, ctx.root_dir),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    scope_decisions = {d["decision_id"]: d["selected_option"] for d in scope_data["decisions"]}
    selected_configs["execution_profile"] = profile
    selected_configs["pipeline_version"] = PIPELINE_VERSION
    selected_configs["approved_scope_decisions_sha256"] = scope_sha
    selected_configs["scope_decisions"] = scope_decisions
    selected_configs["candidate_manifest_sha256"] = approved_sha
    selected_configs["approved_execution_manifest_sha256"] = approved_sha
    selected_configs["target_execution_engine"] = CANONICAL_TARGET_ENGINE

    out_configs = out_dir / "selected_configurations.json"
    atomic_save_json(out_configs, selected_configs)
    print(f"\nAll 2024 Q2-Q4 tunings completed. Saved to {out_configs.name}.")
    return {"status": "TUNE_COMPLETED", "configurations": selected_configs}


# ----------------------------------------------------------------------
# C. REFIT SUBCOMMAND (Full Pre-2025 Development Dataset)
# ----------------------------------------------------------------------
def run_refit(args=None, context: Optional[PipelineContext] = None) -> Dict[str, Any]:
    """Refit selected model configurations on complete pre-2025 development dataset with model-data binding."""
    ctx = make_context_from_args(args, context)
    out_dir = ctx.get_out_dir()

    print("=" * 70)
    print("SUBCOMMAND: refit (Full Pre-2025 Development Dataset v3.3)")
    print("=" * 70)

    # 1. State transition check: cannot refit if frozen or evaluated
    state_err = check_immutable_state_transitions("refit", ctx)
    if state_err:
        return state_err

    # 2. Approved anchor & Preflight Freshness Check (Requirements A & 5)
    anchor_ok, anchor_val, anchor_err = resolve_approved_anchor(ctx, args)
    if not anchor_ok:
        print(f"ERROR: Approved execution manifest anchor check failed: {anchor_err}")
        return {"status": anchor_val, "error": anchor_err}
    approved_sha = anchor_val

    fresh, msg, code_status = check_preflight_freshness(ctx, approved_sha=approved_sha)
    if not fresh:
        print(f"ERROR: Preflight freshness check failed: {msg}")
        return {"status": code_status, "error": msg}

    profile = getattr(args, "profile", None) if args else getattr(ctx, "execution_profile", None)
    if not profile:
        return {"status": "BLOCKED_PROFILE_REQUIRED", "error": "--profile is required ('recommended_16' or 'confirmatory_12')"}
    execute = getattr(args, "execute", False)
    expected_roles = get_expected_model_roles(profile)
    total_models_expected = len(expected_roles)

    # 3. Dry-Run Safeguard
    if not execute:
        print(f"DRY-RUN MODE: Refit parameters ready. Profile: {profile}. Expected models: {total_models_expected}. Use '--execute' to perform training.")
        return {"status": "DRY_RUN", "models_to_fit": total_models_expected, "execution_profile": profile}

    # Gate 1 User Scope Approval Verification (v3.3.6)
    scope_ok, scope_path, scope_sha, scope_data, scope_err = resolve_and_verify_approved_scope_decisions(ctx, args, require_approval=True)
    if not scope_ok:
        return {"status": scope_err, "error": f"Gate 1 Scope Approval failed: {scope_err}"}

    # 4. Load selected configurations
    configs_path = out_dir / "selected_configurations.json"
    if not configs_path.exists():
        configs_path = ctx.package_dir / "selected_configurations.json"
    if not configs_path.exists():
        print("ERROR: selected_configurations.json not found. Run 'tune --execute' first.")
        return {"status": "BLOCKED_CONFIG_NOT_FOUND", "error": "selected_configurations.json not found. Run 'tune --execute' first."}

    scope_decisions = {d["decision_id"]: d["selected_option"] for d in scope_data["decisions"]}
    val_ok, val_err, val_msg = validate_deliverable_provenance(
        "selected_configurations.json",
        configs_path,
        expected_profile=profile,
        expected_scope_sha=scope_sha,
        expected_decisions=scope_decisions,
        expected_manifest_sha=approved_sha,
    )
    if not val_ok:
        print(f"ERROR: {val_err}: {val_msg}")
        return {"status": val_err, "error": val_msg}
    selected_configs = read_json(configs_path)
    spec = read_json(ctx.get_spec_path())
    frozen_models = []
    import sklearn

    sample_arg = getattr(args, "sample", "all")
    coll_arg = getattr(args, "collection", "all")
    samples_to_refit = SAMPLES if sample_arg == "all" else [sample_arg]
    colls_to_refit = COLLECTIONS if coll_arg == "all" else [coll_arg]

    valid_encoders = {"dinov2_fullframe", "siglip2", "clip_native"}
    valid_families = {"Ridge", "HistGradientBoosting", "LinearSVR", "ElasticNet", "PLS"}

    for sample in samples_to_refit:
        for coll in colls_to_refit:
            cfg = selected_configs[sample][coll]
            aug_enc = cfg["augmented"].get("encoder")
            if aug_enc not in valid_encoders:
                return {"status": "BLOCKED_CONFIG_INTEGRITY_MISMATCH", "error": f"Invalid augmented encoder '{aug_enc}' in selected configurations"}

            cand_name = cfg["augmented"].get("candidate_name", "")
            parts = cand_name.split("_")
            if not (cand_name.startswith("late_") and len(parts) == 3 and parts[1] in valid_families and parts[2] in valid_families):
                return {"status": "BLOCKED_CONFIG_INTEGRITY_MISMATCH", "error": f"Invalid candidate model name '{cand_name}' in selected configurations"}

            data = ctx.load_dataset(sample=sample, collection=coll, encoder=aug_enc)

            # Compute actual training row IDs and dual row signatures
            actual_rowids = [int(r) for r in data["rowids"]]
            n_train = len(actual_rowids)
            n_unique_rows = len(set(actual_rowids))
            ordered_source_row_sha256 = sha256_bytes(",".join(str(r) for r in actual_rowids).encode("utf-8"))
            source_row_set_sha256 = sha256_bytes(",".join(str(r) for r in sorted(actual_rowids)).encode("utf-8"))

            target_identifier = f"{coll.lower()}_development_targets.jsonl"
            env_str = f"Python {sys.version.split()[0]} / sklearn {sklearn.__version__}"

            X_meta_trades = data["X_meta"][data["feature_indices"]]
            X_image_trades = data["features_image"][data["feature_indices"]]
            y = data["y"]

            dev_target_mean = float(np.mean(y))
            if not (isinstance(dev_target_mean, float) and np.isfinite(dev_target_mean)):
                raise ValueError(f"Non-finite development target mean in refit for {sample} {coll}: {dev_target_mean}")

            print(f"\n--- Refitting Models: {sample} | {coll} (n_train = {n_train}) ---")

            # 1. Refit Metadata Baseline Model (M*)
            meta_fam = cfg["metadata"]["family"]
            meta_cfg = cfg["metadata"]["config"]
            meta_pipe = build_metadata_pipeline(meta_fam, meta_cfg, seed=SEED)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                meta_pipe.fit(X_meta_trades, y)
                preds_check = meta_pipe.predict(X_meta_trades[:100])

            assert np.isfinite(preds_check).all(), f"Metadata model fit produced non-finite predictions for {sample} {coll}"
            assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), f"Convergence warning in metadata refit for {sample} {coll}"

            meta_bundle = {
                "model": meta_pipe,
                "sample": sample,
                "collection": coll,
                "role": "metadata_baseline",
                "family": meta_fam,
                "candidate_name": f"M*_{meta_fam}",
                "config": meta_cfg,
                "encoder": "none",
                "image_weight": 0.0,
                "metadata_model": {"family": meta_fam, "config": meta_cfg},
                "image_model": None,
                "n_train": n_train,
                "n_unique_rows": n_unique_rows,
                "ordered_source_row_sha256": ordered_source_row_sha256,
                "source_row_set_sha256": source_row_set_sha256,
                "target_identifier": target_identifier,
                "columns": data["columns"],
                "seed": SEED,
                "environment": env_str,
                "development_target_mean": dev_target_mean,
            }
            meta_path = out_dir / f"refit_model_{sample}_{coll}_metadata.joblib"
            joblib.dump(meta_bundle, meta_path)
            print(f"  [1/3] Saved metadata model: {meta_path.name} (SHA-256: {sha256_file(meta_path)[:16]}...)")
            frozen_models.append(meta_path)

            # 2. Refit Augmented Model (A*)
            aug_cfg = cfg["augmented"]
            am_pipe = build_metadata_pipeline(aug_cfg["metadata_model"]["family"], aug_cfg["metadata_model"]["config"], seed=SEED)
            ai_pipe = build_vision_pipeline(aug_cfg["image_model"]["family"], aug_cfg["image_model"]["config"], seed=SEED)
            aug_late_model = LateFusionModel(
                meta_pipeline=am_pipe,
                vision_pipeline=ai_pipe,
                image_weight=aug_cfg["image_weight"],
                meta_family=aug_cfg["metadata_model"]["family"],
                vision_family=aug_cfg["image_model"]["family"],
                candidate_name=aug_cfg["candidate_name"],
                encoder=aug_cfg["encoder"],
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                aug_late_model.fit(X_meta_trades, X_image_trades, y)
                preds_check = aug_late_model.predict(X_meta_trades[:100], X_image_trades[:100])

            assert np.isfinite(preds_check).all(), f"Augmented model fit produced non-finite predictions for {sample} {coll}"
            assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), f"Convergence warning in augmented refit for {sample} {coll}"

            aug_bundle = {
                "model": aug_late_model,
                "sample": sample,
                "collection": coll,
                "role": "augmented_candidate",
                "family": "LateFusionBlend",
                "candidate_name": aug_cfg["candidate_name"],
                "config": {
                    "image_weight": aug_cfg["image_weight"],
                    "metadata_model": aug_cfg["metadata_model"],
                    "image_model": aug_cfg["image_model"],
                },
                "encoder": aug_cfg["encoder"],
                "image_weight": float(aug_cfg["image_weight"]),
                "metadata_model": aug_cfg["metadata_model"],
                "image_model": aug_cfg["image_model"],
                "n_train": n_train,
                "n_unique_rows": n_unique_rows,
                "ordered_source_row_sha256": ordered_source_row_sha256,
                "source_row_set_sha256": source_row_set_sha256,
                "target_identifier": target_identifier,
                "columns": data["columns"],
                "seed": SEED,
                "environment": env_str,
            }
            aug_path = out_dir / f"refit_model_{sample}_{coll}_augmented.joblib"
            joblib.dump(aug_bundle, aug_path)
            print(f"  [2/3] Saved augmented model: {aug_path.name} (SHA-256: {sha256_file(aug_path)[:16]}...)")
            frozen_models.append(aug_path)

            # 3. Refit Mandatory Early Benchmark (E*)
            bench_cfg = cfg["mandatory_benchmark"]
            bench_model = EarlyFusionModel(
                family=bench_cfg["family"],
                config=bench_cfg["config"],
                seed=SEED,
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                bench_model.fit(X_meta_trades, X_image_trades, y)
                preds_check = bench_model.predict(X_meta_trades[:100], X_image_trades[:100])

            assert np.isfinite(preds_check).all(), f"Benchmark model fit produced non-finite predictions for {sample} {coll}"
            assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), f"Convergence warning in benchmark refit for {sample} {coll}"

            bench_bundle = {
                "model": bench_model,
                "sample": sample,
                "collection": coll,
                "role": "mandatory_early_benchmark",
                "family": bench_cfg["family"],
                "candidate_name": "Early_ElasticNet",
                "config": bench_cfg["config"],
                "encoder": aug_cfg["encoder"],
                "image_weight": 0.0,
                "metadata_model": None,
                "image_model": None,
                "n_train": n_train,
                "n_unique_rows": n_unique_rows,
                "ordered_source_row_sha256": ordered_source_row_sha256,
                "source_row_set_sha256": source_row_set_sha256,
                "target_identifier": target_identifier,
                "columns": data["columns"],
                "seed": SEED,
                "environment": env_str,
            }
            bench_path = out_dir / f"refit_model_{sample}_{coll}_mandatory_benchmark.joblib"
            joblib.dump(bench_bundle, bench_path)
            print(f"  [3/3] Saved mandatory benchmark model: {bench_path.name} (SHA-256: {sha256_file(bench_path)[:16]}...)")
            frozen_models.append(bench_path)

    # If profile == 'recommended_16', refit exploratory models for original_v1
    if profile == "recommended_16" and ("original_v1" in samples_to_refit):
        for coll in colls_to_refit:
            exploratory_dict = selected_configs["original_v1"][coll].get("exploratory_augmented", {})
            for exp_enc in EXPLORATORY_ENCODERS[coll]:
                if exp_enc not in exploratory_dict:
                    continue
                exp_cfg = exploratory_dict[exp_enc]
                exp_data = ctx.load_dataset(sample="original_v1", collection=coll, encoder=exp_enc)

                actual_rowids = [int(r) for r in exp_data["rowids"]]
                n_train = len(actual_rowids)
                n_unique_rows = len(set(actual_rowids))
                ordered_source_row_sha256 = sha256_bytes(",".join(str(r) for r in actual_rowids).encode("utf-8"))
                source_row_set_sha256 = sha256_bytes(",".join(str(r) for r in sorted(actual_rowids)).encode("utf-8"))

                target_identifier = f"{coll.lower()}_development_targets.jsonl"
                env_str = f"Python {sys.version.split()[0]} / sklearn {sklearn.__version__}"

                X_meta_trades = exp_data["X_meta"][exp_data["feature_indices"]]
                X_image_trades = exp_data["features_image"][exp_data["feature_indices"]]
                y = exp_data["y"]

                exp_am_pipe = build_metadata_pipeline(exp_cfg["metadata_model"]["family"], exp_cfg["metadata_model"]["config"], seed=SEED)
                exp_ai_pipe = build_vision_pipeline(exp_cfg["image_model"]["family"], exp_cfg["image_model"]["config"], seed=SEED)
                exp_late_model = LateFusionModel(
                    meta_pipeline=exp_am_pipe,
                    vision_pipeline=exp_ai_pipe,
                    image_weight=exp_cfg["image_weight"],
                    meta_family=exp_cfg["metadata_model"]["family"],
                    vision_family=exp_cfg["image_model"]["family"],
                    candidate_name=exp_cfg["candidate_name"],
                    encoder=exp_enc,
                )

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    exp_late_model.fit(X_meta_trades, X_image_trades, y)
                    preds_check = exp_late_model.predict(X_meta_trades[:100], X_image_trades[:100])

                assert np.isfinite(preds_check).all(), f"Exploratory model fit produced non-finite predictions for original_v1 {coll} {exp_enc}"
                assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), f"Convergence warning in exploratory refit for original_v1 {coll} {exp_enc}"

                exp_bundle = {
                    "model": exp_late_model,
                    "sample": "original_v1",
                    "collection": coll,
                    "role": "exploratory_augmented",
                    "family": "LateFusionBlend",
                    "candidate_name": exp_cfg["candidate_name"],
                    "config": {
                        "image_weight": exp_cfg["image_weight"],
                        "metadata_model": exp_cfg["metadata_model"],
                        "image_model": exp_cfg["image_model"],
                    },
                    "encoder": exp_enc,
                    "image_weight": float(exp_cfg["image_weight"]),
                    "metadata_model": exp_cfg["metadata_model"],
                    "image_model": exp_cfg["image_model"],
                    "n_train": n_train,
                    "n_unique_rows": n_unique_rows,
                    "ordered_source_row_sha256": ordered_source_row_sha256,
                    "source_row_set_sha256": source_row_set_sha256,
                    "target_identifier": target_identifier,
                    "columns": exp_data["columns"],
                    "seed": SEED,
                    "environment": env_str,
                }
                exp_path = out_dir / f"refit_model_original_v1_{coll}_exploratory_{exp_enc}.joblib"
                joblib.dump(exp_bundle, exp_path)
                print(f"  [Exploratory] Saved {exp_enc} model: {exp_path.name} (SHA-256: {sha256_file(exp_path)[:16]}...)")
                frozen_models.append(exp_path)

    scope_decisions = {d["decision_id"]: d["selected_option"] for d in scope_data["decisions"]}
    refit_summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "REFIT_COMPLETED",
        "pipeline_version": PIPELINE_VERSION,
        "execution_profile": profile,
        "approved_scope_decisions_sha256": scope_sha,
        "scope_decisions": scope_decisions,
        "approved_execution_manifest_sha256": approved_sha,
        "candidate_manifest_sha256": approved_sha,
        "target_execution_engine": CANONICAL_TARGET_ENGINE,
        "total_models": len(frozen_models),
        "models": [safe_relpath(p, ctx.root_dir) for p in frozen_models],
    }
    atomic_save_json(out_dir / "refit_summary.json", refit_summary)
    print(f"\nRefit finished for {len(frozen_models)} model bundles (Profile: {profile}).")
    return refit_summary


def run_freeze(args=None, context: Optional[PipelineContext] = None) -> Dict[str, Any]:
    """Compile, audit, and freeze all models with complete binding, schema, and manifest enforcement."""
    ctx = make_context_from_args(args, context)
    out_dir = ctx.get_out_dir()

    print("=" * 70)
    print("SUBCOMMAND: freeze (Expanded Version Integrity Audit & Manifest Generation v3.3)")
    print("=" * 70)

    # 1. State transition check: cannot freeze if evaluated or already frozen
    state_err = check_immutable_state_transitions("freeze", ctx)
    if state_err:
        return state_err

    # 2. Approved anchor verification (Requirement A)
    anchor_ok, anchor_val, anchor_err = resolve_approved_anchor(ctx, args)
    if not anchor_ok:
        print(f"ERROR: Approved execution manifest anchor check failed: {anchor_err}")
        return {"status": anchor_val, "error": anchor_err}
    approved_sha = anchor_val

    # Gate 1 User Scope Approval Verification (v3.3.6)
    scope_ok, scope_path, scope_sha, scope_data, scope_err = resolve_and_verify_approved_scope_decisions(ctx, args, require_approval=True)
    if not scope_ok:
        return {"status": scope_err, "error": f"Gate 1 Scope Approval failed: {scope_err}"}

    # Freshness check against anchor
    fresh, msg, code_status = check_preflight_freshness(ctx, approved_sha=approved_sha)
    if not fresh:
        print(f"ERROR: Preflight freshness check failed: {msg}")
        return {"status": code_status, "error": msg}

    profile = getattr(args, "profile", None) if args else getattr(ctx, "execution_profile", None)
    if not profile:
        return {"status": "BLOCKED_PROFILE_REQUIRED", "error": "--profile is required ('recommended_16' or 'confirmatory_12')"}

    # 3. Runtime Environment Enforcement (Requirement B)
    env_lock_path = out_dir / "environment_lock.json"
    if not env_lock_path.exists():
        env_lock_path = ctx.package_dir / "environment_lock.json"
    if not env_lock_path.exists():
        print("ERROR: environment_lock.json not found.")
        return {"status": "BLOCKED_REQUIRED_FILE_MISSING", "error": "environment_lock.json not found"}
    env_ok, env_errs = verify_runtime_environment(read_json(env_lock_path))
    if not env_ok:
        print(f"ERROR: Runtime environment mismatch: {env_errs}")
        return {"status": "BLOCKED_RUNTIME_ENVIRONMENT_MISMATCH", "error": f"Runtime environment mismatch: {env_errs}"}

    # 4. Fail-Closed Supporting Files and Schemas Check (Requirement E)
    supp_ok, supp_missing = check_required_supporting_files(ctx, require_configs=True)
    if not supp_ok:
        print(f"ERROR: Supporting files missing: {supp_missing}")
        if any("freeze_manifest_schema" in m or "evaluation_custody_schema" in m for m in supp_missing):
            return {"status": "BLOCKED_SCHEMA_MISSING", "error": f"Schema files missing: {supp_missing}"}
        return {"status": "BLOCKED_REQUIRED_FILE_MISSING", "error": f"Required files missing: {supp_missing}"}

    freeze_schema_path = out_dir / "freeze_manifest_schema.json"
    if not freeze_schema_path.exists():
        freeze_schema_path = ctx.package_dir / "freeze_manifest_schema.json"
    custody_schema_path = out_dir / "evaluation_custody_schema.json"
    if not custody_schema_path.exists():
        custody_schema_path = ctx.package_dir / "evaluation_custody_schema.json"

    # 5. Check Pre-Registered Inputs and Manifest Structure (Requirement A)
    hash_manifest_path = ctx.get_manifest_path()
    exec_manifest = read_json(hash_manifest_path)
    man_struct_ok, man_struct_errors = validate_execution_hash_manifest_structure(exec_manifest)
    if not man_struct_ok:
        print(f"ERROR: execution_hash_manifest.json structure invalid: {man_struct_errors}")
        return {"status": "BLOCKED_MANIFEST_STRUCTURE_INVALID", "errors": man_struct_errors}

    # Compare on-disk input files with pre-registered execution_hash_manifest.json (31 artifacts)
    hash_failures = []
    input_provenance = []
    for art in exec_manifest.get("input_artifacts", []):
        p = resolve_manifest_path(art["path"], root_dir=ctx.root_dir)
        if not p.exists():
            hash_failures.append({"path": art["path"], "error": "missing_file"})
            continue
        cur_sha = sha256_file(p)
        if cur_sha != art["sha256"]:
            hash_failures.append({"path": art["path"], "expected": art["sha256"], "actual": cur_sha})
        else:
            input_provenance.append({
                "role": art["role"],
                "path": art["path"],
                "sha256": cur_sha,
                "size_bytes": p.stat().st_size,
            })

    if hash_failures or len(input_provenance) != 31:
        print(f"ERROR: Integrity mismatch in {len(hash_failures)} input artifacts! Aborting freeze.")
        return {"status": "BLOCKED_INPUT_TAMPERING_DETECTED", "failures": hash_failures}

    # Sealed targets custody verification (STAT ONLY)
    sealed_provenance = []
    for art in exec_manifest.get("sealed_test_artifacts", []):
        p = resolve_manifest_path(art["path"], root_dir=ctx.root_dir)
        if not p.exists():
            print(f"ERROR: Sealed test target missing: {art['path']}")
            return {"status": "BLOCKED_SEALED_TARGET_MISSING", "error": art["path"]}
        st = p.stat()
        if st.st_size != art["size_bytes"]:
            print(f"ERROR: Sealed test target stat size mismatch in {art['path']}")
            return {"status": "BLOCKED_SEALED_STAT_MISMATCH", "error": art["path"]}
        coll_name = "BAYC" if "bayc" in art["path"].lower() else "MAYC"
        sealed_provenance.append({
            "role": art["role"],
            "path": art["path"],
            "expected_sha256": art["sha256"],
            "expected_rows": 5120 if coll_name == "BAYC" else 13019,
            "stat_size_bytes": SEALED_TEST_STAT_SIZES[coll_name],
            "custody_status": "SEALED_UNOPENED_CUSTODY",
        })

    # 6. Configurations & Refit Summary verification (Fail-Closed Provenance)
    configs_path = out_dir / "selected_configurations.json"
    if not configs_path.exists():
        configs_path = ctx.package_dir / "selected_configurations.json"
    scope_decisions = {d["decision_id"]: d["selected_option"] for d in scope_data["decisions"]}
    val_ok, val_err, val_msg = validate_deliverable_provenance(
        "selected_configurations.json",
        configs_path,
        expected_profile=profile,
        expected_scope_sha=scope_sha,
        expected_decisions=scope_decisions,
        expected_manifest_sha=approved_sha,
    )
    if not val_ok:
        print(f"ERROR: {val_err}: {val_msg}")
        return {"status": val_err, "error": val_msg}
    selected_configs = read_json(configs_path)

    refit_summary_path = out_dir / "refit_summary.json"
    val_ok, val_err, val_msg = validate_deliverable_provenance(
        "refit_summary.json",
        refit_summary_path,
        expected_profile=profile,
        expected_scope_sha=scope_sha,
        expected_decisions=scope_decisions,
        expected_manifest_sha=approved_sha,
    )
    if not val_ok:
        print(f"ERROR: {val_err}: {val_msg}")
        return {"status": val_err, "error": val_msg}
    refit_summ = read_json(refit_summary_path)
    spec = read_json(ctx.get_spec_path())
    approved_encoders = set(spec["encoders"]["scope_boundary"]["approved_for_preparation"])

    # 7. Audit Models & Model-Config-Training Data Binding (Requirement C & F)
    expected_models_spec = get_expected_model_roles(profile)
    expected_count = len(expected_models_spec) # 16 or 12

    # Check existence of models on disk matching profile
    model_records = []
    seen_keys = set()
    training_signatures = {}

    for s in SAMPLES:
        split_dir = "image_validation_20260909" if s == "original_v1" else "one_wei_sensitivity_targets_20260909"
        split_file = ctx.rev_dir / split_dir / "image_ready_split_manifest_v1.json"
        split_data = read_json(split_file)

        for c in COLLECTIONS:
            cond_key = f"{s}_{c}"
            expected_cfg = selected_configs[s][c]
            if ctx.data_loader:
                mock_data = ctx.load_dataset(sample=s, collection=c, encoder=expected_cfg["augmented"]["encoder"])
                expected_dev_rows = [int(r) for r in mock_data["rowids"]]
            else:
                expected_dev_rows = [int(r) for r in split_data[c]["development_source_rows"]]
            expected_n_train = len(expected_dev_rows)
            expected_n_unique = len(set(expected_dev_rows))
            expected_ordered_sha256 = sha256_bytes(",".join(str(r) for r in expected_dev_rows).encode("utf-8"))
            expected_set_sha256 = sha256_bytes(",".join(str(r) for r in sorted(expected_dev_rows)).encode("utf-8"))

            training_signatures[cond_key] = {
                "n_train": expected_n_train,
                "n_unique_rows": expected_n_unique,
                "ordered_source_row_sha256": expected_ordered_sha256,
                "source_row_set_sha256": expected_set_sha256,
            }

    # Count total refit model joblib files present
    all_joblib_files = list(out_dir.glob("refit_model_*.joblib"))
    if not all_joblib_files:
        all_joblib_files = list(ctx.package_dir.glob("refit_model_*.joblib"))

    if len(all_joblib_files) != expected_count:
        print(f"ERROR: Expected {expected_count} models for profile '{profile}', found {len(all_joblib_files)} on disk.")
        return {"status": "BLOCKED_MODEL_COUNT_MISMATCH", "error": f"Expected {expected_count} models for profile '{profile}', found {len(all_joblib_files)}"}

    for (s, c, role, enc) in expected_models_spec:
        if role == "metadata_baseline":
            suffix = "metadata"
        elif role == "augmented_candidate":
            suffix = "augmented"
        elif role == "mandatory_early_benchmark":
            suffix = "mandatory_benchmark"
        elif role == "exploratory_augmented":
            suffix = f"exploratory_{enc}"
        else:
            return {"status": "BLOCKED_PROFILE_ROLE_MISMATCH", "error": f"Unknown role '{role}'"}

        p = out_dir / f"refit_model_{s}_{c}_{suffix}.joblib"
        if not p.exists():
            p = ctx.package_dir / f"refit_model_{s}_{c}_{suffix}.joblib"
        if not p.exists():
            print(f"ERROR: Expected model file missing: refit_model_{s}_{c}_{suffix}.joblib")
            return {"status": "BLOCKED_PROFILE_ROLE_MISMATCH", "error": f"Missing expected model file: refit_model_{s}_{c}_{suffix}.joblib"}

        sz = p.stat().st_size
        if sz <= 0:
            print(f"ERROR: Model file is empty: {p.name}")
            return {"status": "BLOCKED_EMPTY_MODEL", "error": f"Empty model: {p.name}"}

        cur_sha = sha256_file(p)
        bundle = joblib.load(p)

        cond_sig = training_signatures[f"{s}_{c}"]
        expected_n_train = cond_sig["n_train"]
        expected_n_unique = cond_sig["n_unique_rows"]
        expected_ordered_sha256 = cond_sig["ordered_source_row_sha256"]
        expected_set_sha256 = cond_sig["source_row_set_sha256"]
        expected_cfg = selected_configs[s][c]

        # Verify bundle metadata and Model-Data Binding
        if bundle.get("seed") != SEED:
            return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Seed mismatch in {p.name}"}
        if bundle.get("sample") != s or bundle.get("collection") != c or bundle.get("role") != role:
            return {"status": "BLOCKED_PROFILE_ROLE_MISMATCH", "error": f"Role/sample/coll mismatch in {p.name}"}
        if bundle.get("target_identifier") != f"{c.lower()}_development_targets.jsonl":
            return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Target identifier mismatch in {p.name}"}

        # Role-specific validations
        if role == "metadata_baseline":
            if bundle.get("encoder") != "none":
                return {"status": "BLOCKED_ENCODER_BINDING_MISMATCH", "error": f"Metadata encoder must be 'none', got '{bundle.get('encoder')}'"}
            if bundle.get("family") != expected_cfg["metadata"]["family"]:
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Metadata family mismatch in {p.name}"}
            if bundle.get("config") != expected_cfg["metadata"]["config"]:
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Metadata config mismatch in {p.name}"}
            if bundle.get("candidate_name") != f"M*_{expected_cfg['metadata']['family']}":
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Metadata candidate_name mismatch in {p.name}"}
        elif role == "augmented_candidate":
            exp_aug = expected_cfg["augmented"]
            if bundle.get("encoder") != exp_aug["encoder"] or bundle.get("encoder") not in approved_encoders:
                return {"status": "BLOCKED_ENCODER_BINDING_MISMATCH", "error": f"Augmented encoder '{bundle.get('encoder')}' mismatch in {p.name}"}
            if bundle.get("candidate_name") != exp_aug["candidate_name"]:
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Augmented candidate_name mismatch in {p.name}"}
            if bundle.get("family") != "LateFusionBlend":
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Augmented family mismatch in {p.name}"}
            if bundle.get("image_weight") != exp_aug["image_weight"]:
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Augmented image_weight mismatch in {p.name}"}
            if bundle.get("metadata_model") != exp_aug["metadata_model"] or bundle.get("image_model") != exp_aug["image_model"]:
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Augmented submodel configs mismatch in {p.name}"}
        elif role == "mandatory_early_benchmark":
            exp_bm = expected_cfg["mandatory_benchmark"]
            if bundle.get("encoder") != expected_cfg["augmented"]["encoder"] or bundle.get("encoder") not in approved_encoders:
                return {"status": "BLOCKED_ENCODER_BINDING_MISMATCH", "error": f"Benchmark encoder mismatch in {p.name}"}
            if bundle.get("candidate_name") != "Early_ElasticNet":
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Benchmark candidate_name mismatch in {p.name}"}
            if bundle.get("family") != exp_bm["family"] or bundle.get("config") != exp_bm["config"]:
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Benchmark config mismatch in {p.name}"}
        elif role == "exploratory_augmented":
            exp_dict = expected_cfg.get("exploratory_augmented", {})
            if enc not in exp_dict:
                return {"status": "BLOCKED_PROFILE_ROLE_MISMATCH", "error": f"Exploratory encoder '{enc}' not tuned for {c}"}
            exp_aug = exp_dict[enc]
            if bundle.get("encoder") != enc or bundle.get("encoder") not in approved_encoders:
                return {"status": "BLOCKED_ENCODER_BINDING_MISMATCH", "error": f"Exploratory encoder '{bundle.get('encoder')}' mismatch in {p.name}"}
            if bundle.get("candidate_name") != exp_aug["candidate_name"]:
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Exploratory candidate_name mismatch in {p.name}"}
            if bundle.get("family") != "LateFusionBlend":
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Exploratory family mismatch in {p.name}"}
            if bundle.get("image_weight") != exp_aug["image_weight"]:
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Exploratory image_weight mismatch in {p.name}"}
            if bundle.get("metadata_model") != exp_aug["metadata_model"] or bundle.get("image_model") != exp_aug["image_model"]:
                return {"status": "BLOCKED_MODEL_CONFIG_BINDING_MISMATCH", "error": f"Exploratory submodel configs mismatch in {p.name}"}

        # Verify training row signatures
        if bundle.get("n_train") != expected_n_train or bundle.get("n_unique_rows") != expected_n_unique:
            return {"status": "BLOCKED_MODEL_DATA_BINDING_MISMATCH", "error": f"Row count mismatch in {p.name}"}
        if bundle.get("source_row_set_sha256") != expected_set_sha256:
            return {"status": "BLOCKED_MODEL_DATA_BINDING_MISMATCH", "error": f"Row set signature mismatch in {p.name}"}
        if bundle.get("ordered_source_row_sha256") != expected_ordered_sha256:
            return {"status": "BLOCKED_MODEL_DATA_BINDING_MISMATCH", "error": f"Ordered row sequence mismatch in {p.name}"}

        key = (s, c, role, enc)
        seen_keys.add(key)
        model_records.append({
            "sample": s,
            "collection": c,
            "model_role": role,
            "family": str(bundle.get("family", "")),
            "candidate_name": str(bundle.get("candidate_name", "")),
            "config": bundle["config"],
            "encoder": str(bundle.get("encoder", "")),
            "image_weight": float(bundle.get("image_weight", 0.0)),
            "seed": int(bundle.get("seed", SEED)),
            "n_train": int(bundle.get("n_train", expected_n_train)),
            "n_unique_rows": int(bundle.get("n_unique_rows", expected_n_unique)),
            "ordered_source_row_sha256": str(bundle.get("ordered_source_row_sha256", expected_ordered_sha256)),
            "source_row_set_sha256": str(bundle.get("source_row_set_sha256", expected_set_sha256)),
            "target_identifier": str(bundle.get("target_identifier", f"{c.lower()}_development_targets.jsonl")),
            "path": safe_relpath(p, ctx.root_dir),
            "sha256": cur_sha,
            "size_bytes": sz,
        })

    if len(model_records) != expected_count:
        return {"status": "BLOCKED_MODEL_COUNT_MISMATCH", "error": f"Found {len(model_records)} models instead of {expected_count}"}

    # 8. Pipeline Code, Solver Code, Spec, Env Lock, Execution Manifest, Schemas
    pipe_path = Path(__file__).resolve()
    solver_path = ctx.rev_dir / "code" / "counted_primal_svr_v3.py"
    spec_path = ctx.get_spec_path()

    # Collect frozen baseline specifications from refitted models (Fail-Closed Task C)
    baselines_manifest = {"original_v1": {}}
    for m in model_records:
        if m["sample"] == "original_v1" and m["model_role"] == "metadata_baseline":
            mb_p = resolve_manifest_path(m["path"], root_dir=ctx.root_dir)
            mb_bundle = joblib.load(mb_p)
            if "development_target_mean" not in mb_bundle:
                err = f"Missing 'development_target_mean' in metadata baseline bundle: {m['path']}"
                print(f"ERROR: BLOCKED_DEVELOPMENT_MEAN_BASELINE_INVALID: {err}")
                return {"status": "BLOCKED_DEVELOPMENT_MEAN_BASELINE_INVALID", "error": err}
            dev_mean_val = mb_bundle["development_target_mean"]
            if not isinstance(dev_mean_val, (int, float)) or not np.isfinite(dev_mean_val) or isinstance(dev_mean_val, bool):
                err = f"Non-finite or non-float development_target_mean in {m['path']}: {dev_mean_val!r}"
                print(f"ERROR: BLOCKED_DEVELOPMENT_MEAN_BASELINE_INVALID: {err}")
                return {"status": "BLOCKED_DEVELOPMENT_MEAN_BASELINE_INVALID", "error": err}
            dev_mean_val = float(dev_mean_val)
            baselines_manifest["original_v1"][m["collection"]] = {
                "development_target_mean": dev_mean_val,
                "source_model_joblib": m["path"],
                "source_model_sha256": m["sha256"],
            }

    freeze_manifest = {
        "manifest_metadata": {
            "manifest_name": "freeze_manifest.json",
            "pipeline_version": PIPELINE_VERSION,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "freeze_policy": "Strict refit freeze. No subsequent tuning, refit, candidate alteration, or overwrite is permitted.",
            "approved_execution_manifest_sha256": approved_sha,
            "approved_scope_decisions_sha256": scope_sha,
            "execution_profile": profile,
            "scope_decisions": scope_decisions,
            "target_execution_engine": CANONICAL_TARGET_ENGINE,
        },
        "approved_scope_decisions": {
            "path": safe_relpath(scope_path, ctx.root_dir),
            "sha256": scope_sha,
            "governance_status": "USER_APPROVED",
            "execution_profile": profile,
            "candidate_manifest_sha256": approved_sha,
            "target_execution_engine": CANONICAL_TARGET_ENGINE,
            "scope_decisions": scope_decisions,
        },
        "models": model_records,
        "baselines": baselines_manifest,
        "pipeline_code": {
            "path": safe_relpath(pipe_path, ctx.root_dir),
            "sha256": sha256_file(pipe_path),
            "size_bytes": pipe_path.stat().st_size,
        },
        "solver_code": {
            "path": safe_relpath(solver_path, ctx.root_dir),
            "sha256": sha256_file(solver_path),
            "size_bytes": solver_path.stat().st_size,
        },
        "specification": {
            "path": safe_relpath(spec_path, ctx.root_dir),
            "sha256": sha256_file(spec_path),
            "size_bytes": spec_path.stat().st_size,
        },
        "selected_configurations": {
            "path": safe_relpath(configs_path, ctx.root_dir),
            "sha256": sha256_file(configs_path),
            "size_bytes": configs_path.stat().st_size,
        },
        "environment_lock": {
            "path": safe_relpath(env_lock_path, ctx.root_dir),
            "sha256": sha256_file(env_lock_path),
            "size_bytes": env_lock_path.stat().st_size,
        },
        "execution_hash_manifest": {
            "path": safe_relpath(hash_manifest_path, ctx.root_dir),
            "sha256": sha256_file(hash_manifest_path),
            "size_bytes": hash_manifest_path.stat().st_size,
        },
        "schemas": {
            "freeze_manifest_schema": {
                "path": safe_relpath(freeze_schema_path, ctx.root_dir),
                "sha256": sha256_file(freeze_schema_path),
                "size_bytes": freeze_schema_path.stat().st_size,
            },
            "evaluation_custody_schema": {
                "path": safe_relpath(custody_schema_path, ctx.root_dir),
                "sha256": sha256_file(custody_schema_path),
                "size_bytes": custody_schema_path.stat().st_size,
            },
        },
        "input_artifacts_provenance": input_provenance,
        "preregistered_sealed_targets_custody": sealed_provenance,
        "training_source_row_signatures": training_signatures,
    }

    # 9. Schema validation (Fail-closed)
    validate_schema(freeze_manifest, read_json(freeze_schema_path))
    print("Freeze manifest successfully validated against freeze_manifest_schema.json.")

    out_freeze = out_dir / "freeze_manifest.json"
    atomic_save_json(out_freeze, freeze_manifest)
    print(f"\nExpanded Freeze Manifest successfully compiled, validated, and audited: {out_freeze.name} (Profile: {profile}, Models: {len(model_records)}).")

    return {
        "status": "FREEZE_AUDIT_PASSED",
        "freeze_manifest_path": safe_relpath(out_freeze, ctx.root_dir),
        "freeze_manifest_sha256": sha256_file(out_freeze),
        "total_models": len(model_records),
        "execution_profile": profile,
        "total_verified_inputs": len(input_provenance),
    }


def run_verify_freeze(args=None, context: Optional[PipelineContext] = None) -> Dict[str, Any]:
    """Read-only verification of the frozen pipeline against freeze_manifest.json (Zero file modification)."""
    ctx = make_context_from_args(args, context)
    out_dir = ctx.get_out_dir()

    print("=" * 70)
    print("SUBCOMMAND: verify-freeze (Read-Only Version Audit v3.3)")
    print("=" * 70)

    # Approved anchor & profile verification first (Requirement A)
    anchor_ok, anchor_val, anchor_err = resolve_approved_anchor(ctx, args, require_approval=True)
    if not anchor_ok:
        print(f"ERROR: Approved anchor verification failed in verify-freeze: {anchor_err}")
        return {"status": anchor_val, "error": anchor_err}
    approved_sha = anchor_val

    # Gate 1 User Scope Approval Verification (v3.3.6)
    scope_ok, scope_path, scope_sha, scope_data, scope_err = resolve_and_verify_approved_scope_decisions(ctx, args, require_approval=True)
    if not scope_ok:
        return {"status": scope_err, "error": f"Gate 1 Scope Approval failed: {scope_err}"}

    # Fail-closed check for required files
    supp_ok, supp_missing = check_required_supporting_files(ctx, require_configs=True)
    if not supp_ok:
        print(f"ERROR: Supporting files missing in verify-freeze: {supp_missing}")
        return {"status": "FAIL", "error": f"Supporting files missing: {supp_missing}"}

    freeze_path = out_dir / "freeze_manifest.json"
    if not freeze_path.exists():
        freeze_path = ctx.package_dir / "freeze_manifest.json"

    profile = (getattr(args, "profile", None) if args else None) or (getattr(ctx, "execution_profile", None)) or scope_data.get("execution_profile", DEFAULT_PROFILE)
    scope_decisions = {d["decision_id"]: d["selected_option"] for d in scope_data["decisions"]}
    val_ok, val_err, val_msg = validate_deliverable_provenance(
        "freeze_manifest.json",
        freeze_path,
        expected_profile=profile,
        expected_scope_sha=scope_sha,
        expected_decisions=scope_decisions,
        expected_manifest_sha=approved_sha,
    )
    if not val_ok:
        print(f"ERROR: {val_err}: {val_msg}")
        return {"status": val_err, "error": val_msg}

    freeze = read_json(freeze_path)
    freeze_schema_path = out_dir / "freeze_manifest_schema.json"
    if not freeze_schema_path.exists():
        freeze_schema_path = ctx.package_dir / "freeze_manifest_schema.json"
    validate_schema(freeze, read_json(freeze_schema_path))

    manifest_meta = freeze.get("manifest_metadata", {})
    scope_binding = freeze.get("approved_scope_decisions", {})

    ok_integrity, failures = verify_freeze_manifest_integrity(freeze, ctx)
    if not ok_integrity:
        print(f"ERROR: VERIFY-FREEZE AUDIT FAILED! Failures: {failures}")
        return {"status": "FAIL", "failures": failures}

    print(f"\nVERIFY-FREEZE AUDIT PASSED: All {len(freeze['models'])} models, code, solver, specification, configurations, schemas, and inputs match.")
    return {
        "status": "VERIFY_FREEZE_PASSED",
        "verified_models": len(freeze["models"]),
        "execution_profile": manifest_meta.get("execution_profile", "unknown"),
        "verified_inputs": len(freeze["input_artifacts_provenance"]),
        "approved_scope_decisions_sha256": scope_sha,
        "freeze_manifest_sha256": sha256_file(freeze_path),
    }


def verify_freeze_manifest_integrity(freeze: Dict[str, Any], ctx: PipelineContext) -> Tuple[bool, List[str]]:
    """Cryptographically re-audit freeze manifest against all disk files prior to test target unsealing."""
    failures = []
    out_dir = ctx.get_out_dir()

    # Fail-closed check
    supp_ok, supp_missing = check_required_supporting_files(ctx, require_configs=True)
    if not supp_ok:
        failures.append(f"Supporting files missing: {supp_missing}")

    # 1. Models verification
    for m in freeze.get("models", []):
        p = resolve_manifest_path(m["path"], root_dir=ctx.root_dir)
        if not p.exists():
            failures.append(f"Model file missing: {m['path']}")
            continue
        cur_sha = sha256_file(p)
        if cur_sha != m["sha256"]:
            failures.append(f"Model tampering detected in {m['path']} (expected {m['sha256'][:12]}, got {cur_sha[:12]})")
        if p.stat().st_size != m["size_bytes"]:
            failures.append(f"Model size mismatch in {m['path']}")

    # 2. Pipeline Code, Solver Code, Spec, Configurations, Env Lock, Execution Manifest, Schemas
    pipe_path = Path(__file__).resolve()
    if sha256_file(pipe_path) != freeze["pipeline_code"]["sha256"]:
        failures.append("pipeline.py hash mismatch")

    solver_path = ctx.rev_dir / "code" / "counted_primal_svr_v3.py"
    if not solver_path.exists() or sha256_file(solver_path) != freeze["solver_code"]["sha256"]:
        failures.append("code/counted_primal_svr_v3.py hash mismatch or file missing")

    spec_path = ctx.get_spec_path()
    if not spec_path.exists() or sha256_file(spec_path) != freeze["specification"]["sha256"]:
        failures.append("final_execution_specification.json hash mismatch")

    configs_path = ctx.get_out_dir() / "selected_configurations.json"
    if not configs_path.exists():
        configs_path = ctx.package_dir / "selected_configurations.json"
    if not configs_path.exists() or sha256_file(configs_path) != freeze["selected_configurations"]["sha256"]:
        failures.append("selected_configurations.json hash mismatch")

    env_lock_path = ctx.get_out_dir() / "environment_lock.json"
    if not env_lock_path.exists():
        env_lock_path = ctx.package_dir / "environment_lock.json"
    if not env_lock_path.exists() or sha256_file(env_lock_path) != freeze["environment_lock"]["sha256"]:
        failures.append("environment_lock.json hash mismatch")
    else:
        env_ok, env_errs = verify_runtime_environment(read_json(env_lock_path))
        if not env_ok:
            failures.append(f"Runtime environment mismatch: {env_errs}")

    # execution_hash_manifest check
    manifest_path = ctx.get_manifest_path()
    if not manifest_path.exists() or sha256_file(manifest_path) != freeze.get("execution_hash_manifest", {}).get("sha256"):
        failures.append("BLOCKED_EXECUTION_MANIFEST_TAMPERING: execution_hash_manifest.json hash mismatch")

    # Schemas check
    schemas_bound = freeze.get("schemas", {})
    f_schema_path = out_dir / "freeze_manifest_schema.json"
    if not f_schema_path.exists(): f_schema_path = ctx.package_dir / "freeze_manifest_schema.json"
    if not f_schema_path.exists() or sha256_file(f_schema_path) != schemas_bound.get("freeze_manifest_schema", {}).get("sha256"):
        failures.append("freeze_manifest_schema.json hash mismatch or file missing")

    c_schema_path = out_dir / "evaluation_custody_schema.json"
    if not c_schema_path.exists(): c_schema_path = ctx.package_dir / "evaluation_custody_schema.json"
    if not c_schema_path.exists() or sha256_file(c_schema_path) != schemas_bound.get("evaluation_custody_schema", {}).get("sha256"):
        failures.append("evaluation_custody_schema.json hash mismatch or file missing")

    # 3. Input artifacts provenance verification
    for art in freeze.get("input_artifacts_provenance", []):
        p = resolve_manifest_path(art["path"], root_dir=ctx.root_dir)
        if not p.exists():
            failures.append(f"Input artifact missing: {art['path']}")
            continue
        cur_sha = sha256_file(p)
        if cur_sha != art["sha256"]:
            failures.append(f"Input artifact altered: {art['path']} (expected {art['sha256'][:12]}, got {cur_sha[:12]})")

    return (len(failures) == 0, failures)


def run_evaluate(args=None, context: Optional[PipelineContext] = None) -> Dict[str, Any]:
    """Single-pass out-of-time test evaluation and atomic custody logging."""
    ctx = make_context_from_args(args, context)
    out_dir = ctx.get_out_dir()

    print("=" * 70)
    print("SUBCOMMAND: evaluate (Single-Pass Out-of-Time Test Evaluation v3.3)")
    print("=" * 70)

    # Safeguard 1: Explicit Authorization Confirmation Flag
    confirmed = getattr(args, "confirm_unsealed_evaluation", False)
    if not confirmed:
        print("CRITICAL SAFEGUARD: Final evaluation is blocked without explicit authorization.")
        print("To proceed, you must provide the flag:")
        print("  --confirm-unsealed-evaluation")
        return {
            "status": "BLOCKED_CONFIRMATION_REQUIRED",
            "error": "Final evaluation is blocked without --confirm-unsealed-evaluation",
        }

    # Approved anchor verification (Requirement A)
    anchor_ok, anchor_val, anchor_err = resolve_approved_anchor(ctx, args)
    if not anchor_ok:
        print(f"ERROR: Approved execution manifest anchor check failed: {anchor_err}")
        return {"status": anchor_val, "error": anchor_err}
    approved_sha = anchor_val

    # Gate 1 User Scope Approval Verification (v3.3.6)
    scope_ok, scope_path, scope_sha, scope_data, scope_err = resolve_and_verify_approved_scope_decisions(ctx, args, require_approval=True)
    if not scope_ok:
        return {"status": scope_err, "error": f"Gate 1 Scope Approval failed: {scope_err}"}

    # 1. Atomic Concurrency Lock
    lock_file = out_dir / "evaluation.lock"
    try:
        lock_fd = acquire_evaluation_lock(lock_file)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return {"status": "BLOCKED_CONCURRENT_LOCK_ACTIVE", "error": str(e)}

    # Fail-closed check
    supp_ok, supp_missing = check_required_supporting_files(ctx, require_configs=True)
    if not supp_ok:
        release_evaluation_lock(lock_fd, lock_file)
        if any("freeze_manifest_schema" in m or "evaluation_custody_schema" in m for m in supp_missing):
            return {"status": "BLOCKED_SCHEMA_MISSING", "error": f"Schema files missing: {supp_missing}"}
        return {"status": "BLOCKED_REQUIRED_FILE_MISSING", "error": f"Supporting files missing: {supp_missing}"}

    custody_file = out_dir / "evaluation_custody.json"
    custody_schema_path = out_dir / "evaluation_custody_schema.json"
    if not custody_schema_path.exists():
        custody_schema_path = ctx.package_dir / "evaluation_custody_schema.json"

    try:
        # 2. Check Existing Custody Lifecycle State
        if custody_file.exists():
            existing_custody = read_json(custody_file)
            c_status = existing_custody.get("status")
            if c_status == "EVALUATION_COMPLETED":
                print("WARNING: Single-pass evaluation was already completed and audited previously.")
                print(f"Audit record exists at {custody_file.name}. Blocking duplicate unsealed execution.")
                metrics_file = out_dir / "test_metrics_summary.json"
                if metrics_file.exists():
                    return read_json(metrics_file)
                return {"status": "BLOCKED_DUPLICATE_EVALUATION_PREVENTED", "custody": existing_custody}
            elif c_status in ["EVALUATION_STARTED", "EVALUATION_FAILED"]:
                print(f"CRITICAL: Evaluation custody record exists with status '{c_status}'.")
                print("Crash-safe custody prohibits automated re-evaluation after started or failed evaluation.")
                return {"status": "BLOCKED_PREVIOUS_EVALUATION_RECORD_EXISTS", "custody": existing_custody}

        # 3. Comprehensive Freeze Manifest Integrity Verification
        freeze_path = out_dir / "freeze_manifest.json"
        if not freeze_path.exists():
            freeze_path = ctx.package_dir / "freeze_manifest.json"

        profile = (getattr(args, "profile", None) if args else None) or (getattr(ctx, "execution_profile", None)) or scope_data.get("execution_profile", DEFAULT_PROFILE)
        scope_decisions = {d["decision_id"]: d["selected_option"] for d in scope_data["decisions"]}
        val_ok, val_err, val_msg = validate_deliverable_provenance(
            "freeze_manifest.json",
            freeze_path,
            expected_profile=profile,
            expected_scope_sha=scope_sha,
            expected_decisions=scope_decisions,
            expected_manifest_sha=approved_sha,
        )
        if not val_ok:
            print(f"ERROR: {val_err}: {val_msg}")
            return {"status": val_err, "error": val_msg}

        freeze = read_json(freeze_path)
        freeze_sha = sha256_file(freeze_path)

        schema_path = out_dir / "freeze_manifest_schema.json"
        if not schema_path.exists():
            schema_path = ctx.package_dir / "freeze_manifest_schema.json"
        validate_schema(freeze, read_json(schema_path))

        manifest_meta = freeze.get("manifest_metadata", {})
        freeze_scope_binding = freeze.get("approved_scope_decisions", {})

        profile = manifest_meta.get("execution_profile", DEFAULT_PROFILE)

        ok_integrity, failures = verify_freeze_manifest_integrity(freeze, ctx)
        if not ok_integrity:
            print(f"ERROR: INTEGRITY COMPROMISED! Mismatches: {failures}")
            if any("execution_hash_manifest" in f for f in failures):
                return {"status": "BLOCKED_EXECUTION_MANIFEST_TAMPERING", "failures": failures}
            elif any("pipeline.py" in f for f in failures):
                return {"status": "BLOCKED_SCRIPT_TAMPERING_DETECTED", "failures": failures}
            elif any("selected_configurations.json" in f for f in failures):
                return {"status": "BLOCKED_CONFIG_TAMPERING_DETECTED", "failures": failures}
            elif any("Model tampering" in f for f in failures):
                return {"status": "BLOCKED_MODEL_TAMPERING_DETECTED", "failures": failures}
            return {"status": "BLOCKED_FREEZE_INTEGRITY_MISMATCH", "failures": failures}

        models_by_role = {}
        exploratory_models = {}
        for m in freeze["models"]:
            p = resolve_manifest_path(m["path"], root_dir=ctx.root_dir)
            if m["model_role"] == "exploratory_augmented":
                exploratory_models[(m["sample"], m["collection"], m["encoder"])] = p
            else:
                key = (m["sample"], m["collection"], m["model_role"])
                models_by_role[key] = p

        # 4. Atomically record EVALUATION_STARTED before unsealing test files
        custody_record = {
            "custody_metadata": {
                "record_name": "evaluation_custody.json",
                "pipeline_version": PIPELINE_VERSION,
                "single_pass_policy": "Strict single-pass forward test. Post-test hyperparameter re-selection or re-evaluation is strictly prohibited.",
            },
            "status": "EVALUATION_STARTED",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "freeze_manifest_sha256": freeze_sha,
            "approved_execution_manifest_sha256": approved_sha,
            "approved_scope_decisions_sha256": scope_sha,
            "execution_profile": profile,
            "scope_decisions": {d["decision_id"]: d["selected_option"] for d in scope_data["decisions"]},
            "target_execution_engine": CANONICAL_TARGET_ENGINE,
            "unsealed_conditions": [],
            "completed_conditions": [],
        }
        validate_schema(custody_record, read_json(custody_schema_path))
        atomic_save_json(custody_file, custody_record)
        print("Custody record initiated: EVALUATION_STARTED.")

        # 5. Unsealing and Pre-Prediction Verification
        print("\nUNSEALING TEST TARGETS AND EXECUTING PRE-PREDICTION VERIFICATION...")
        scope_decisions = {d["decision_id"]: d["selected_option"] for d in scope_data["decisions"]}
        results_summary = {
            "status": "EVALUATION_COMPLETED",
            "execution_profile": profile,
            "approved_execution_manifest_sha256": approved_sha,
            "freeze_manifest_sha256": freeze_sha,
            "approved_scope_decisions_sha256": scope_sha,
            "scope_decisions": scope_decisions,
            "target_execution_engine": CANONICAL_TARGET_ENGINE,
            "pipeline_version": PIPELINE_VERSION,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {},
        }
        if profile == "recommended_16":
            results_summary["exploratory"] = {}

        reg_data = read_json(ctx.rev_dir / "Encoder Development Registry 20260909 v4.json")
        if ctx.metadata_loader:
            meta_all = ctx.metadata_loader()
        else:
            meta_path = ctx.rev_dir / "data_audit_20260908" / "metadata_normalized.jsonl"
            meta_all = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        meta_dict = {(mr["collection"], int(mr["token_id"])): mr for mr in meta_all}

        # Requirement B & D: Exact 1-Wei Clean-Slice Invariant Verification
        for coll_chk in COLLECTIONS:
            prim_rows_chk, _ = ctx.load_test_targets("original_v1", coll_chk)
            sens_rows_chk, _ = ctx.load_test_targets("exact_one_wei_sensitivity_v1", coll_chk)

            # 1. Verify every trade in sensitivity targets has price_eth > 1e-18 ETH (no 1-wei trades)
            one_wei_in_sens = []
            for r in sens_rows_chk:
                try:
                    if is_exact_one_wei_trade(r):
                        one_wei_in_sens.append(r)
                except Exception as val_err:
                    raise CleanSliceVerificationError(
                        f"Exact 1-wei validation error in {coll_chk} sensitivity target row: {val_err}",
                        error_code="BLOCKED_SENSITIVITY_SLICE_INVALID",
                    ) from val_err

            if one_wei_in_sens:
                raise CleanSliceVerificationError(
                    f"Exact 1-wei clean-slice violation in {coll_chk} sensitivity targets: found {len(one_wei_in_sens)} row(s) with exact 1-wei trades.",
                    error_code="BLOCKED_SENSITIVITY_SLICE_INVALID",
                )

            # 2. Verify sensitivity rows == primary rows - exact 1-wei rows by source_row
            expected_sens_source_rows = []
            for r in prim_rows_chk:
                try:
                    if not is_exact_one_wei_trade(r):
                        expected_sens_source_rows.append(r["source_row"])
                except Exception as val_err:
                    raise CleanSliceVerificationError(
                        f"Exact 1-wei validation error in {coll_chk} primary target row: {val_err}",
                        error_code="BLOCKED_SENSITIVITY_SLICE_INVALID",
                    ) from val_err

            actual_sens_source_rows = [r["source_row"] for r in sens_rows_chk]

            if actual_sens_source_rows != expected_sens_source_rows:
                raise CleanSliceVerificationError(
                    f"Exact 1-wei clean-slice source_row invariant violation in {coll_chk}: "
                    f"expected {len(expected_sens_source_rows)} rows matching primary minus 1-wei, "
                    f"got {len(actual_sens_source_rows)} rows.",
                    error_code="BLOCKED_SENSITIVITY_SLICE_INVALID",
                )

        # Cache loaded feature matrices to prevent redundant disk I/O
        feature_cache = {}
        manifest_cache = {}

        def get_encoder_features(encoder_name: str, coll_name: str):
            cache_key = (encoder_name, coll_name)
            if cache_key not in feature_cache:
                coll_inf = reg_data[encoder_name]["collections"][coll_name]
                feat_matrix = np.load(resolve_manifest_path(coll_inf["matrix"], root_dir=ctx.root_dir), allow_pickle=False).astype(np.float64)
                m_lines = [json.loads(line) for line in resolve_manifest_path(coll_inf["manifest"], root_dir=ctx.root_dir).read_text(encoding="utf-8").splitlines() if line.strip()]
                tok_to_idx = {mr["token_id"]: mr["feature_row_idx"] for mr in m_lines}
                feature_cache[cache_key] = feat_matrix
                manifest_cache[cache_key] = tok_to_idx
            return feature_cache[cache_key], manifest_cache[cache_key]

        for sample in SAMPLES:
            results_summary["metrics"][sample] = {}
            for coll in COLLECTIONS:
                cond_name = f"{sample}_{coll}"
                custody_record["unsealed_conditions"].append(cond_name)
                atomic_save_json(custody_file, custody_record)

                print(f"\n--- Evaluating Condition: {sample} | {coll} ---")
                test_rows, target_file = ctx.load_test_targets(sample, coll)

                # Pre-Prediction Verification
                if not ctx.test_target_loader:
                    actual_test_sha = sha256_file(target_file)
                    expected_sha = SEALED_TEST_PRE_REGISTERED_SHA256[coll]
                    if actual_test_sha != expected_sha:
                        raise PrePredictionVerificationError(f"Test target SHA-256 mismatch for {coll}: expected {expected_sha}, got {actual_test_sha}")
                    expected_target_count = 5120 if coll == "BAYC" else 13019
                    if len(test_rows) != expected_target_count:
                        raise PrePredictionVerificationError(f"Target count mismatch in {target_file.name}: expected {expected_target_count}, got {len(test_rows)}")

                for r in test_rows:
                    if r["time"] < "2025-01-01":
                        raise PrePredictionVerificationError(f"Time boundary leak detected in {target_file.name}: trade time {r['time']} < 2025-01-01")

                enc_info = reg_data["dinov2_fullframe"]["collections"][coll]
                manifest_path = resolve_manifest_path(enc_info["manifest"], root_dir=ctx.root_dir)
                manifest_toks = set()
                with manifest_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            manifest_toks.add(int(json.loads(line)["token_id"]))

                for r in test_rows:
                    if r["token_id"] not in manifest_toks:
                        raise PrePredictionVerificationError(f"Token {r['token_id']} in test targets not found in feature manifest for {coll}")

                y_test = np.asarray([r["y_log_relative_price"] for r in test_rows], dtype=np.float64)
                if not np.isfinite(y_test).all():
                    raise PrePredictionVerificationError(f"Non-finite target values found in {coll} test target")

                tokens_test = np.asarray([r["token_id"] for r in test_rows], dtype=np.int64)

                meta_bundle = joblib.load(models_by_role[sample, coll, "metadata_baseline"])
                meta_model = meta_bundle["model"]
                aug_bundle = joblib.load(models_by_role[sample, coll, "augmented_candidate"])
                aug_model = aug_bundle["model"]
                bench_bundle = joblib.load(models_by_role[sample, coll, "mandatory_early_benchmark"])
                bench_model = bench_bundle["model"]
                aug_enc = aug_bundle.get("encoder", "dinov2_fullframe")

                cols = meta_bundle["columns"]
                X_test_meta = np.asarray([[meta_dict[coll, int(t)][c] for c in cols] for t in tokens_test], dtype=object)

                feat_matrix, token_to_idx = get_encoder_features(aug_enc, coll)
                feat_indices_test = np.asarray([token_to_idx[int(t)] for t in tokens_test], dtype=np.int64)
                X_test_image = feat_matrix[feat_indices_test]

                preds_meta = np.asarray(meta_model.predict(X_test_meta)).reshape(-1)
                preds_aug = np.asarray(aug_model.predict(X_test_meta, X_test_image)).reshape(-1)
                preds_bench = np.asarray(bench_model.predict(X_test_meta, X_test_image)).reshape(-1)

                metrics_meta = compute_metrics(y_test, preds_meta, tokens_test)
                metrics_aug = compute_metrics(y_test, preds_aug, tokens_test)
                metrics_bench = compute_metrics(y_test, preds_bench, tokens_test)

                metrics_aug["delta_rmse"] = float(metrics_aug["rmse"] - metrics_meta["rmse"])
                denom_m = metrics_meta["rmse"] if metrics_meta["rmse"] > 1e-12 else 1e-12
                metrics_aug["utility_percent"] = float((metrics_meta["rmse"] - metrics_aug["rmse"]) / denom_m * 100.0)

                metrics_bench["delta_rmse"] = float(metrics_bench["rmse"] - metrics_meta["rmse"])
                metrics_bench["utility_percent"] = float((metrics_meta["rmse"] - metrics_bench["rmse"]) / denom_m * 100.0)

                pred_file = out_dir / f"test_predictions_{sample}_{coll}.jsonl"
                tmp_pred = pred_file.with_name(f"{pred_file.stem}_{os.getpid()}_{time.time_ns()}.tmp")

                # Structure outputs based on sample:
                if sample == "original_v1":
                    # Task B: Compute Zero Baseline (all predictions 0.0)
                    preds_zero = np.zeros_like(y_test, dtype=np.float64)
                    metrics_zero = compute_metrics(y_test, preds_zero, tokens_test)
                    metrics_zero["delta_rmse"] = float(metrics_zero["rmse"] - metrics_meta["rmse"])
                    metrics_zero["utility_percent"] = float((metrics_meta["rmse"] - metrics_zero["rmse"]) / denom_m * 100.0)

                    # Fail-Closed Task C: Validate development_target_mean presence, type, and freeze match
                    if "development_target_mean" not in meta_bundle:
                        raise DevelopmentMeanBaselineError(
                            f"Missing 'development_target_mean' in metadata baseline bundle for {sample} {coll}",
                            error_code="BLOCKED_DEVELOPMENT_MEAN_BASELINE_INVALID"
                        )
                    raw_dev_mean = meta_bundle["development_target_mean"]
                    if not isinstance(raw_dev_mean, (int, float)) or not np.isfinite(raw_dev_mean) or isinstance(raw_dev_mean, bool):
                        raise DevelopmentMeanBaselineError(
                            f"Invalid development_target_mean in metadata baseline bundle for {sample} {coll}: {raw_dev_mean!r}",
                            error_code="BLOCKED_DEVELOPMENT_MEAN_BASELINE_INVALID"
                        )
                    dev_target_mean = float(raw_dev_mean)

                    f_base = freeze.get("baselines", {}).get(sample, {}).get(coll, {})
                    if "development_target_mean" not in f_base:
                        raise DevelopmentMeanBaselineError(
                            f"Missing 'development_target_mean' in freeze manifest baselines for {sample} {coll}",
                            error_code="BLOCKED_DEVELOPMENT_MEAN_BASELINE_INVALID"
                        )
                    f_dev_mean = f_base["development_target_mean"]
                    if not isinstance(f_dev_mean, (int, float)) or not np.isfinite(f_dev_mean) or abs(dev_target_mean - float(f_dev_mean)) > 1e-12:
                        raise DevelopmentMeanBaselineError(
                            f"Development target mean mismatch: bundle ({dev_target_mean}) != freeze manifest ({f_dev_mean}) for {sample} {coll}",
                            error_code="BLOCKED_DEVELOPMENT_MEAN_BASELINE_INVALID"
                        )

                    # Anti-leakage assertion: Mean baseline must NEVER be calculated from test set
                    test_target_mean = float(np.mean(y_test))
                    preds_mean = np.full_like(y_test, dev_target_mean, dtype=np.float64)
                    metrics_mean = compute_metrics(y_test, preds_mean, tokens_test)
                    metrics_mean["delta_rmse"] = float(metrics_mean["rmse"] - metrics_meta["rmse"])
                    metrics_mean["utility_percent"] = float((metrics_meta["rmse"] - metrics_mean["rmse"]) / denom_m * 100.0)
                    metrics_mean["development_target_mean"] = dev_target_mean

                    results_summary["metrics"][sample][coll] = {
                        "metadata_baseline": metrics_meta,
                        "augmented_candidate": metrics_aug,
                        "mandatory_early_benchmark": metrics_bench,
                        "zero_baseline": metrics_zero,
                        "mean_baseline": metrics_mean,
                    }

                    # Exploratory models evaluation if recommended_16 (Requirement C)
                    exploratory_preds = {}
                    if profile == "recommended_16":
                        for exp_enc in EXPLORATORY_ENCODERS[coll]:
                            exp_key = ("original_v1", coll, exp_enc)
                            if exp_key in exploratory_models:
                                exp_bundle = joblib.load(exploratory_models[exp_key])
                                exp_model = exp_bundle["model"]
                                exp_feat_mat, exp_tok_idx = get_encoder_features(exp_enc, coll)
                                exp_feat_indices = np.asarray([exp_tok_idx[int(t)] for t in tokens_test], dtype=np.int64)
                                X_test_exp_img = exp_feat_mat[exp_feat_indices]

                                preds_exp = np.asarray(exp_model.predict(X_test_meta, X_test_exp_img)).reshape(-1)
                                exploratory_preds[exp_enc] = preds_exp

                                m_exp = compute_metrics(y_test, preds_exp, tokens_test)
                                m_exp["delta_rmse"] = float(m_exp["rmse"] - metrics_meta["rmse"])
                                m_exp["utility_percent"] = float((metrics_meta["rmse"] - m_exp["rmse"]) / denom_m * 100.0)
                                m_exp["candidate_name"] = exp_bundle.get("candidate_name")
                                m_exp["image_weight"] = float(exp_bundle.get("image_weight", 0.0))

                                results_summary.setdefault("exploratory", {}).setdefault("original_v1", {}).setdefault(coll, {})[exp_enc] = m_exp

                    with tmp_pred.open("w", encoding="utf-8") as pf:
                        for i, (r, ym, ya, yb) in enumerate(zip(test_rows, preds_meta, preds_aug, preds_bench)):
                            rec = dict(r)
                            rec["prediction_metadata"] = float(ym)
                            rec["prediction_augmented"] = float(ya)
                            rec["prediction_mandatory_benchmark"] = float(yb)
                            rec["prediction_zero_baseline"] = 0.0
                            rec["prediction_mean_baseline"] = float(dev_target_mean)
                            for exp_enc, ep_arr in exploratory_preds.items():
                                rec[f"prediction_exploratory_{exp_enc}"] = float(ep_arr[i])
                            pf.write(json.dumps(rec, default=float, allow_nan=False) + "\n")
                        pf.flush()
                        os.fsync(pf.fileno())

                elif sample == "exact_one_wei_sensitivity_v1":
                    # Requirement B: Dual Reporting (independent_refit vs primary_fixed_spec)
                    # 1. Independent refit metrics
                    indep_metrics = {
                        "metadata_baseline": metrics_meta,
                        "augmented_candidate": metrics_aug,
                        "mandatory_early_benchmark": metrics_bench,
                    }

                    # 2. Primary fixed-spec metrics (Evaluate original_v1 models directly on 1-wei clean targets)
                    prim_meta_bundle = joblib.load(models_by_role["original_v1", coll, "metadata_baseline"])
                    prim_aug_bundle = joblib.load(models_by_role["original_v1", coll, "augmented_candidate"])
                    prim_aug_enc = prim_aug_bundle.get("encoder", "dinov2_fullframe")

                    prim_cols = prim_meta_bundle["columns"]
                    X_test_meta_prim = np.asarray([[meta_dict[coll, int(t)][c] for c in prim_cols] for t in tokens_test], dtype=object)

                    prim_feat_mat, prim_tok_idx = get_encoder_features(prim_aug_enc, coll)
                    prim_feat_indices = np.asarray([prim_tok_idx[int(t)] for t in tokens_test], dtype=np.int64)
                    X_test_img_prim = prim_feat_mat[prim_feat_indices]

                    preds_pf_meta = np.asarray(prim_meta_bundle["model"].predict(X_test_meta_prim)).reshape(-1)
                    preds_pf_aug = np.asarray(prim_aug_bundle["model"].predict(X_test_meta_prim, X_test_img_prim)).reshape(-1)

                    pf_m_metrics = compute_metrics(y_test, preds_pf_meta, tokens_test)
                    pf_a_metrics = compute_metrics(y_test, preds_pf_aug, tokens_test)
                    delta_pf = float(pf_a_metrics["rmse"] - pf_m_metrics["rmse"])
                    denom_pf = pf_m_metrics["rmse"] if pf_m_metrics["rmse"] > 1e-12 else 1e-12
                    u_pf = float((pf_m_metrics["rmse"] - pf_a_metrics["rmse"]) / denom_pf * 100.0)

                    pf_metrics = {
                        "n_trades": len(y_test),
                        "n_tokens": len(np.unique(tokens_test)),
                        "metadata_baseline": pf_m_metrics,
                        "augmented_candidate": pf_a_metrics,
                        "delta_rmse": delta_pf,
                        "utility_percent": u_pf,
                    }

                    results_summary["metrics"][sample][coll] = {
                        "independent_refit": indep_metrics,
                        "primary_fixed_spec": pf_metrics,
                    }

                    with tmp_pred.open("w", encoding="utf-8") as pf:
                        for r, ym, ya, yb, ypm, ypa in zip(test_rows, preds_meta, preds_aug, preds_bench, preds_pf_meta, preds_pf_aug):
                            rec = dict(r)
                            rec["prediction_metadata"] = float(ym)
                            rec["prediction_augmented"] = float(ya)
                            rec["prediction_mandatory_benchmark"] = float(yb)
                            rec["prediction_primary_fixed_spec_metadata"] = float(ypm)
                            rec["prediction_primary_fixed_spec_augmented"] = float(ypa)
                            pf.write(json.dumps(rec, default=float, allow_nan=False) + "\n")
                        pf.flush()
                        os.fsync(pf.fileno())

                for attempt in range(10):
                    try:
                        os.replace(tmp_pred, pred_file)
                        break
                    except PermissionError:
                        time.sleep(0.05 * (attempt + 1))

                custody_record["completed_conditions"].append(cond_name)
                atomic_save_json(custody_file, custody_record)

        # 6. Publish final summary metrics atomically
        metrics_file = out_dir / "test_metrics_summary.json"
        atomic_save_json(metrics_file, results_summary)

        # 7. Update custody status to EVALUATION_COMPLETED
        custody_record["status"] = "EVALUATION_COMPLETED"
        custody_record["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        validate_schema(custody_record, read_json(custody_schema_path))
        atomic_save_json(custody_file, custody_record)

        print(f"\nSingle-pass evaluation completed and audited in custody ledger: {custody_file.name}.")
        return results_summary

    except Exception as e:
        err_code = getattr(e, "error_code", "BLOCKED_EVALUATION_EXECUTION_ERROR")
        err_msg = str(e)
        print(f"\nCRITICAL EVALUATION FAILURE [{err_code}]: {err_msg}")
        try:
            # Task D: Fail-closed atomic cleanup of partial metrics and prediction files
            metrics_file = out_dir / "test_metrics_summary.json"
            if metrics_file.exists():
                try:
                    metrics_file.unlink()
                except Exception:
                    pass
            for f in out_dir.glob("test_predictions_*.tmp*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            for f in out_dir.glob("test_predictions_*.jsonl"):
                try:
                    f.unlink()
                except Exception:
                    pass

            custody_record["status"] = "EVALUATION_FAILED"
            custody_record["failed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            custody_record["error_code"] = err_code
            custody_record["error_message"] = err_msg
            custody_record["error"] = f"{err_code}: {err_msg}"
            if custody_schema_path.exists():
                try:
                    validate_schema(custody_record, read_json(custody_schema_path))
                except Exception:
                    pass
            atomic_save_json(custody_file, custody_record)
        except Exception as save_err:
            print(f"CRITICAL: Failed to update custody record: {save_err}")
        return {"status": err_code, "error": err_msg, "custody": custody_record}
    finally:
        release_evaluation_lock(lock_fd, lock_file)


def run_inference(args=None, context: Optional[PipelineContext] = None) -> Dict[str, Any]:
    """Execute 2,000-resample paired token-cluster bootstrap and block bootstrap inference."""
    ctx = make_context_from_args(args, context)
    out_dir = ctx.get_out_dir()

    print("=" * 70)
    print("SUBCOMMAND: inference (2,000 Paired Token-Cluster Bootstrap v3.3)")
    print("=" * 70)

    # Anchor and profile verification first in inference (Requirement A)
    anchor_ok, anchor_val, anchor_err = resolve_approved_anchor(ctx, args, require_approval=True)
    if not anchor_ok:
        print(f"ERROR: Approved anchor check failed in inference: {anchor_err}")
        return {"status": anchor_val, "error": anchor_err}
    approved_sha = anchor_val

    # Gate 1 User Scope Approval Verification (v3.3.6)
    scope_ok, scope_path, scope_sha, scope_data, scope_err = resolve_and_verify_approved_scope_decisions(ctx, args, require_approval=True)
    if not scope_ok:
        return {"status": scope_err, "error": f"Gate 1 Scope Approval failed: {scope_err}"}

    n_iter = getattr(args, "iterations", 2000)
    seed = getattr(args, "seed", SEED)
    rng = np.random.default_rng(seed)
    profile = (getattr(args, "profile", None) if args else None) or (getattr(ctx, "execution_profile", None)) or scope_data.get("execution_profile", DEFAULT_PROFILE)

    scope_decisions = {d["decision_id"]: d["selected_option"] for d in scope_data["decisions"]}

    freeze_path = out_dir / "freeze_manifest.json"
    if not freeze_path.exists():
        freeze_path = ctx.package_dir / "freeze_manifest.json"
    val_ok, val_err, val_msg = validate_deliverable_provenance(
        "freeze_manifest.json",
        freeze_path,
        expected_profile=profile,
        expected_scope_sha=scope_sha,
        expected_decisions=scope_decisions,
        expected_manifest_sha=approved_sha,
    )
    if not val_ok:
        print(f"ERROR: {val_err}: {val_msg}")
        return {"status": val_err, "error": val_msg}
    freeze_data = read_json(freeze_path)
    actual_freeze_sha = sha256_file(freeze_path)

    custody_file = out_dir / "evaluation_custody.json"
    val_ok, val_err, val_msg = validate_deliverable_provenance(
        "evaluation_custody.json",
        custody_file,
        expected_profile=profile,
        expected_scope_sha=scope_sha,
        expected_decisions=scope_decisions,
        expected_manifest_sha=approved_sha,
        expected_freeze_sha=actual_freeze_sha,
    )
    if not val_ok:
        print(f"ERROR: {val_err}: {val_msg}")
        return {"status": val_err, "error": val_msg}
    custody_data = read_json(custody_file)
    if custody_data.get("status") != "EVALUATION_COMPLETED":
        err = f"evaluation_custody.json status is '{custody_data.get('status')}', expected 'EVALUATION_COMPLETED'"
        print(f"ERROR: {err}")
        return {"status": "BLOCKED_CUSTODY_STATUS_MISMATCH", "error": err}

    metrics_path = out_dir / "test_metrics_summary.json"
    val_ok, val_err, val_msg = validate_deliverable_provenance(
        "test_metrics_summary.json",
        metrics_path,
        expected_profile=profile,
        expected_scope_sha=scope_sha,
        expected_decisions=scope_decisions,
        expected_manifest_sha=approved_sha,
        expected_freeze_sha=actual_freeze_sha,
    )
    if not val_ok:
        print(f"ERROR: {val_err}: {val_msg}")
        return {"status": val_err, "error": val_msg}
    metrics_data = read_json(metrics_path)
    if metrics_data.get("status") != "EVALUATION_COMPLETED":
        err = f"test_metrics_summary.json status is '{metrics_data.get('status')}', expected 'EVALUATION_COMPLETED'"
        print(f"ERROR: {err}")
        return {"status": "BLOCKED_EVALUATION_NOT_COMPLETED", "error": err}

    bootstrap_results = {
        "status": "INFERENCE_COMPLETED",
        "execution_profile": profile,
        "approved_execution_manifest_sha256": approved_sha,
        "freeze_manifest_sha256": actual_freeze_sha,
        "approved_scope_decisions_sha256": scope_sha,
        "scope_decisions": scope_decisions,
        "target_execution_engine": CANONICAL_TARGET_ENGINE,
        "pipeline_version": PIPELINE_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations": n_iter,
        "seed": seed,
        "collections": {},
    }
    if profile == "recommended_16":
        bootstrap_results["exploratory"] = {}

    def get_ci(arr):
        arr = np.array(arr)
        if len(arr) == 0 or not np.isfinite(arr).all():
            return {
                "mean": 0.0,
                "std": 0.0,
                "ci_95": (0.0, 0.0),
                "ci_97_5": (0.0, 0.0),
                "ci_975": (0.0, 0.0),
            }
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_95": (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
            "ci_97_5": (float(np.percentile(arr, 1.25)), float(np.percentile(arr, 98.75))),
            "ci_975": (float(np.percentile(arr, 1.25)), float(np.percentile(arr, 98.75))),
        }

    for sample in SAMPLES:
        bootstrap_results["collections"][sample] = {}
        for coll in COLLECTIONS:
            pred_file = out_dir / f"test_predictions_{sample}_{coll}.jsonl"
            if not pred_file.exists():
                continue
            rows = [json.loads(line) for line in pred_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                continue

            y_true = np.asarray([r["y_log_relative_price"] for r in rows], dtype=np.float64)
            y_meta = np.asarray([r["prediction_metadata"] for r in rows], dtype=np.float64)
            y_aug = np.asarray([r["prediction_augmented"] for r in rows], dtype=np.float64)
            y_bench = np.asarray([r["prediction_mandatory_benchmark"] for r in rows], dtype=np.float64)
            tokens = np.asarray([r["token_id"] for r in rows], dtype=np.int64)
            times = [r["time"] for r in rows]

            unique_tokens = np.unique(tokens)
            n_tokens = len(unique_tokens)
            token_to_idx = {t: np.where(tokens == t)[0] for t in unique_tokens}

            # Check if exploratory predictions exist in rows for original_v1
            exploratory_cols = {}
            if sample == "original_v1":
                for k in rows[0].keys():
                    if k.startswith("prediction_exploratory_"):
                        enc = k.replace("prediction_exploratory_", "")
                        exploratory_cols[enc] = np.asarray([r[k] for r in rows], dtype=np.float64)

            # Task C: Check for fixed-spec predictions in exact_one_wei_sensitivity_v1
            has_fixed_spec = (sample == "exact_one_wei_sensitivity_v1" and
                              "prediction_primary_fixed_spec_metadata" in rows[0] and
                              "prediction_primary_fixed_spec_augmented" in rows[0])
            if has_fixed_spec:
                y_pf_meta = np.asarray([r["prediction_primary_fixed_spec_metadata"] for r in rows], dtype=np.float64)
                y_pf_aug = np.asarray([r["prediction_primary_fixed_spec_augmented"] for r in rows], dtype=np.float64)

            # 1. Paired Token-Cluster Bootstrap
            rmse_meta_samples = []
            rmse_aug_samples = []
            delta_aug_samples = []
            u_aug_samples = []
            rmse_bench_samples = []
            delta_bench_samples = []
            u_bench_samples = []

            rmse_pf_meta_samples = []
            rmse_pf_aug_samples = []
            delta_pf_samples = []
            u_pf_samples = []

            exp_rmse_samples = {enc: [] for enc in exploratory_cols}
            exp_delta_samples = {enc: [] for enc in exploratory_cols}
            exp_u_samples = {enc: [] for enc in exploratory_cols}

            for _ in range(n_iter):
                resampled_tokens = rng.choice(unique_tokens, size=n_tokens, replace=True)
                resampled_rows = []
                for t in resampled_tokens:
                    resampled_rows.extend(token_to_idx[t])
                resampled_rows = np.array(resampled_rows)

                y_b = y_true[resampled_rows]
                m_b = y_meta[resampled_rows]
                a_b = y_aug[resampled_rows]
                e_b = y_bench[resampled_rows]

                rmse_m = np.sqrt(np.mean((y_b - m_b) ** 2))
                rmse_a = np.sqrt(np.mean((y_b - a_b) ** 2))
                rmse_e = np.sqrt(np.mean((y_b - e_b) ** 2))

                rmse_meta_samples.append(rmse_m)
                rmse_aug_samples.append(rmse_a)
                delta_aug_samples.append(rmse_a - rmse_m)
                denom_m = rmse_m if rmse_m > 1e-12 else 1e-12
                u_aug_samples.append((rmse_m - rmse_a) / denom_m * 100.0)

                rmse_bench_samples.append(rmse_e)
                delta_bench_samples.append(rmse_e - rmse_m)
                u_bench_samples.append((rmse_m - rmse_e) / denom_m * 100.0)

                if has_fixed_spec:
                    pf_m_b = y_pf_meta[resampled_rows]
                    pf_a_b = y_pf_aug[resampled_rows]
                    rmse_pf_m = np.sqrt(np.mean((y_b - pf_m_b) ** 2))
                    rmse_pf_a = np.sqrt(np.mean((y_b - pf_a_b) ** 2))
                    delta_pf = rmse_pf_a - rmse_pf_m
                    denom_pf = rmse_pf_m if rmse_pf_m > 1e-12 else 1e-12
                    u_pf = (rmse_pf_m - rmse_pf_a) / denom_pf * 100.0

                    rmse_pf_meta_samples.append(rmse_pf_m)
                    rmse_pf_aug_samples.append(rmse_pf_a)
                    delta_pf_samples.append(delta_pf)
                    u_pf_samples.append(u_pf)

                for enc, y_exp in exploratory_cols.items():
                    exp_b = y_exp[resampled_rows]
                    rmse_exp = np.sqrt(np.mean((y_b - exp_b) ** 2))
                    exp_rmse_samples[enc].append(rmse_exp)
                    exp_delta_samples[enc].append(rmse_exp - rmse_m)
                    exp_u_samples[enc].append((rmse_m - rmse_exp) / denom_m * 100.0)

            # 2. 14-day calendar block bootstrap
            time_dt = np.array([np.datetime64(t[:10]) for t in times])
            min_date = time_dt.min()
            day_offsets = (time_dt - min_date).astype(int)
            block_ids = day_offsets // 14
            unique_blocks = np.unique(block_ids)
            n_blocks = len(unique_blocks)
            block_to_idx = {b: np.where(block_ids == b)[0] for b in unique_blocks}

            block_delta_aug = []
            block_u_aug = []
            for _ in range(n_iter):
                resampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
                res_rows = []
                for b in resampled_blocks:
                    res_rows.extend(block_to_idx[b])
                res_rows = np.array(res_rows)

                y_b = y_true[res_rows]
                m_b = y_meta[res_rows]
                a_b = y_aug[res_rows]

                rmse_m = np.sqrt(np.mean((y_b - m_b) ** 2))
                rmse_a = np.sqrt(np.mean((y_b - a_b) ** 2))
                block_delta_aug.append(rmse_a - rmse_m)
                denom_m = rmse_m if rmse_m > 1e-12 else 1e-12
                block_u_aug.append((rmse_m - rmse_a) / denom_m * 100.0)

            # 3. Subgroup Analysis (Known vs Unseen tokens)
            if ctx.dev_target_loader:
                dev_rows = ctx.dev_target_loader(sample, coll)
            else:
                target_dir = "target_pipeline_20260909" if sample == "original_v1" else "one_wei_sensitivity_targets_20260909"
                dev_target_path = ctx.rev_dir / target_dir / f"{coll.lower()}_development_targets.jsonl"
                dev_rows = [json.loads(line) for line in dev_target_path.read_text(encoding="utf-8").splitlines() if line.strip()]

            dev_tokens = set(int(r["token_id"]) for r in dev_rows)

            subgroups = {}
            for sg_name, is_known in [("known_tokens", True), ("unseen_tokens", False)]:
                sg_mask = np.isin(tokens, list(dev_tokens)) if is_known else ~np.isin(tokens, list(dev_tokens))
                sg_rows = np.where(sg_mask)[0]
                sg_tokens = np.unique(tokens[sg_rows])
                n_sg_trades = len(sg_rows)
                n_sg_tokens = len(sg_tokens)

                sg_info = {
                    "n_trades": n_sg_trades,
                    "n_unique_tokens": n_sg_tokens,
                    "meets_minimum_support": bool(n_sg_trades >= 50 and n_sg_tokens >= 30),
                }

                if n_sg_trades > 0:
                    y_sg = y_true[sg_rows]
                    m_sg = y_meta[sg_rows]
                    a_sg = y_aug[sg_rows]
                    sg_m_rmse = np.sqrt(np.mean((y_sg - m_sg) ** 2))
                    sg_a_rmse = np.sqrt(np.mean((y_sg - a_sg) ** 2))
                    sg_info["metadata_rmse"] = float(sg_m_rmse)
                    sg_info["augmented_rmse"] = float(sg_a_rmse)
                    sg_info["delta_rmse"] = float(sg_a_rmse - sg_m_rmse)
                    denom_sg = sg_m_rmse if sg_m_rmse > 1e-12 else 1e-12
                    sg_info["utility_percent"] = float((sg_m_rmse - sg_a_rmse) / denom_sg * 100.0)
                else:
                    sg_info["metadata_rmse"] = 0.0
                    sg_info["augmented_rmse"] = 0.0
                    sg_info["delta_rmse"] = 0.0
                    sg_info["utility_percent"] = 0.0

                subgroups[sg_name] = sg_info

            ci_delta_aug = get_ci(delta_aug_samples)
            ci_u_aug = get_ci(u_aug_samples)
            ci_u_bench = get_ci(u_bench_samples)
            ci_block_delta = get_ci(block_delta_aug)
            ci_block_u = get_ci(block_u_aug)

            ci_pf_m = get_ci(rmse_pf_meta_samples) if has_fixed_spec else None
            ci_pf_a = get_ci(rmse_pf_aug_samples) if has_fixed_spec else None
            ci_pf_delta = get_ci(delta_pf_samples) if has_fixed_spec else None
            ci_pf_u = get_ci(u_pf_samples) if has_fixed_spec else None

            paired_bootstrap_dict = {
                "iterations": n_iter,
                "n_tokens": n_tokens,
                "metadata_rmse": get_ci(rmse_meta_samples),
                "augmented_rmse": get_ci(rmse_aug_samples),
                "augmented_delta_rmse": ci_delta_aug,
                "augmented_utility_pct": ci_u_aug,
                "augmented_utility_percent": ci_u_aug,
                "benchmark_rmse": get_ci(rmse_bench_samples),
                "benchmark_delta_rmse": get_ci(delta_bench_samples),
                "benchmark_utility_pct": ci_u_bench,
                "benchmark_utility_percent": ci_u_bench,
                "confirmatory_bonferroni_975": {
                    "delta_rmse_ci_97_5": ci_delta_aug["ci_97_5"],
                    "delta_rmse_ci_975": ci_delta_aug["ci_975"],
                    "utility_pct_ci_97_5": ci_u_aug["ci_97_5"],
                    "utility_percent_ci_975": ci_u_aug["ci_975"],
                },
            }

            if has_fixed_spec:
                # Task C: Add primary_fixed_spec_delta_rmse and clear structural separation
                paired_bootstrap_dict["primary_fixed_spec_delta_rmse"] = ci_pf_delta
                paired_bootstrap_dict["primary_fixed_spec"] = {
                    "metadata_rmse": ci_pf_m,
                    "augmented_rmse": ci_pf_a,
                    "delta_rmse": ci_pf_delta,
                    "utility_pct": ci_pf_u,
                    "utility_percent": ci_pf_u,
                }
                paired_bootstrap_dict["independent_refit"] = {
                    "metadata_rmse": get_ci(rmse_meta_samples),
                    "augmented_rmse": get_ci(rmse_aug_samples),
                    "augmented_delta_rmse": ci_delta_aug,
                    "augmented_utility_pct": ci_u_aug,
                    "augmented_utility_percent": ci_u_aug,
                    "benchmark_rmse": get_ci(rmse_bench_samples),
                    "benchmark_delta_rmse": get_ci(delta_bench_samples),
                    "benchmark_utility_pct": ci_u_bench,
                    "benchmark_utility_percent": ci_u_bench,
                }

            coll_summary = {
                "paired_token_cluster_bootstrap": paired_bootstrap_dict,
                "calendar_block_bootstrap_14d": {
                    "n_blocks": n_blocks,
                    "augmented_delta_rmse": ci_block_delta,
                    "augmented_utility_pct": ci_block_u,
                    "augmented_utility_percent": ci_block_u,
                },
                "subgroups": subgroups,
            }
            bootstrap_results["collections"][sample][coll] = coll_summary

            if exploratory_cols:
                for enc in exploratory_cols:
                    ci_res = {
                        "encoder": enc,
                        "n_tokens": n_tokens,
                        "exploratory_ci_95": {
                            "rmse": get_ci(exp_rmse_samples[enc])["ci_95"],
                            "delta_rmse": get_ci(exp_delta_samples[enc])["ci_95"],
                            "utility_percent": get_ci(exp_u_samples[enc])["ci_95"],
                        },
                    }
                    bootstrap_results.setdefault("exploratory", {}).setdefault("original_v1", {}).setdefault(coll, {})[enc] = ci_res

    bootstrap_results["status"] = "INFERENCE_COMPLETED"
    out_file = out_dir / "confidence_intervals_summary.json"
    atomic_save_json(out_file, bootstrap_results)
    print(f"\nInference completed. Results saved to {out_file.name}.")
    return bootstrap_results


# ----------------------------------------------------------------------
# CLI DISPATCHER
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NFT Revision Final Execution Pipeline CLI (v3.3.6.4)")
    subparsers = parser.add_subparsers(dest="subcommand", help="Execution subcommand")

    def add_common_args(sp):
        sp.add_argument("--approved-execution-manifest-sha256", type=str, default=None, help="External auditor-approved execution manifest SHA-256 (64 hex)")
        sp.add_argument("--approved-profile", choices=["recommended_16", "confirmatory_12"], default=None, help="Auditor-approved profile ('recommended_16' or 'confirmatory_12')")
        sp.add_argument("--profile", "--execution-profile", dest="profile", default=None, help="Execution profile ('recommended_16' or 'confirmatory_12'). Implicit default prohibited for active execution.")
        sp.add_argument("--external-anchor-path", type=str, default=None, help="Path to external auditor approval anchor JSON file located strictly outside the package directory")
        sp.add_argument("--approved-scope-decisions-path", "--scope-decisions-path", dest="approved_scope_decisions_path", type=str, default=None, help="Path to user-approved Gate 1 scope decisions JSON file (conforming to approved_scope_decisions.schema.json)")
        sp.add_argument("--out-dir", type=str, default=None, help="Output directory for pipeline deliverables (defaults to package_dir)")

    # Preflight
    p_pre = subparsers.add_parser("preflight", help="Run preflight integrity and environment verification")
    add_common_args(p_pre)

    # Tune
    p_tune = subparsers.add_parser("tune", help="Tune 2024 Q2-Q4 internal CV hyperparameters")
    add_common_args(p_tune)
    p_tune.add_argument("--sample", choices=["original_v1", "exact_one_wei_sensitivity_v1", "all"], default="all")
    p_tune.add_argument("--collection", choices=["BAYC", "MAYC", "all"], default="all")
    p_tune.add_argument("--execute", action="store_true", help="Execute numerical cross-validation (default is dry-run)")

    # Refit
    p_refit = subparsers.add_parser("refit", help="Refit selected models on full pre-2025 development dataset")
    add_common_args(p_refit)
    p_refit.add_argument("--sample", choices=["original_v1", "exact_one_wei_sensitivity_v1", "all"], default="all")
    p_refit.add_argument("--collection", choices=["BAYC", "MAYC", "all"], default="all")
    p_refit.add_argument("--execute", action="store_true", help="Execute numerical refit (default is dry-run)")

    # Freeze
    p_freeze = subparsers.add_parser("freeze", help="Compile and audit expanded freeze manifest (immutable)")
    add_common_args(p_freeze)

    # Verify-Freeze
    p_vf = subparsers.add_parser("verify-freeze", help="Read-only audit of frozen manifest against models, code, and inputs")
    add_common_args(p_vf)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Execute single-pass out-of-time test evaluation")
    add_common_args(p_eval)
    p_eval.add_argument("--confirm-unsealed-evaluation", action="store_true", help="Explicit confirmation to unseal test targets")

    # Inference
    p_inf = subparsers.add_parser("inference", help="Execute paired token-cluster bootstrap inference")
    add_common_args(p_inf)
    p_inf.add_argument("--iterations", type=int, default=2000, help="Number of bootstrap resamples (default 2000)")
    p_inf.add_argument("--seed", type=int, default=SEED, help="Random seed for bootstrap resampling")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    subcmd = args.subcommand

    dispatcher = {
        "preflight": run_preflight,
        "tune": run_tune,
        "refit": run_refit,
        "freeze": run_freeze,
        "verify-freeze": run_verify_freeze,
        "evaluate": run_evaluate,
        "inference": run_inference,
    }

    fn = dispatcher.get(subcmd)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    # For active execution subcommands, enforce mandatory --profile flag
    if subcmd in ["tune", "refit", "freeze", "verify-freeze", "evaluate", "inference"] and getattr(args, "profile", None) is None:
        print("ERROR: BLOCKED_PROFILE_REQUIRED: --profile is required ('recommended_16' or 'confirmatory_12'). Implicit defaulting is prohibited.", file=sys.stderr)
        sys.exit(1)

    # Gate 1 User Scope Approval Enforcement (v3.3.6)
    is_active_execution = (
        (subcmd == "tune" and getattr(args, "execute", False)) or
        (subcmd == "refit" and getattr(args, "execute", False)) or
        subcmd in ["freeze", "verify-freeze", "evaluate", "inference"]
    )
    if is_active_execution and not getattr(args, "approved_scope_decisions_path", None):
        print("ERROR: BLOCKED_USER_SCOPE_APPROVAL_REQUIRED: --approved-scope-decisions-path is required for active execution.", file=sys.stderr)
        sys.exit(1)

    try:
        res = fn(args)
    except Exception as exc:
        print(f"CRITICAL ERROR: Unhandled exception during {subcmd}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    if not isinstance(res, dict):
        print(f"ERROR: Subcommand {subcmd} returned non-dict result: {type(res)}", file=sys.stderr)
        sys.exit(1)

    status = res.get("status")

    if subcmd == "preflight":
        if status in ["PENDING_EXTERNAL_APPROVAL", "PENDING_USER_SCOPE_APPROVAL"]:
            print(f"\nPREFLIGHT STATUS: {status} (Package verified intact; awaiting external approval or user scope decisions).")
            sys.exit(0)
        elif status == "PASS":
            print("\nPREFLIGHT STATUS: PASS (Package and external approval verified).")
            sys.exit(0)
        else:
            print(f"\nPREFLIGHT FAILED: status={status}", file=sys.stderr)
            sys.exit(1)

    # If status is an explicit blocked/fail code:
    if status and (str(status).startswith("BLOCKED_") or str(status).endswith("_FAILED") or status in ["FAIL", "ERROR", "REJECTED", "PENDING_EXTERNAL_APPROVAL"]):
        print(f"\nCOMMAND BLOCKED: {subcmd} halted with status '{status}'. Error: {res.get('error', '')}", file=sys.stderr)
        sys.exit(1)

    # Dry-run check:
    if status == "DRY_RUN":
        sys.exit(0)

    # Success statuses:
    success_statuses = {
        "tune": ["TUNE_COMPLETED"],
        "refit": ["REFIT_COMPLETED"],
        "freeze": ["FREEZE_AUDIT_PASSED"],
        "verify-freeze": ["VERIFY_FREEZE_PASSED"],
        "evaluate": ["EVALUATION_COMPLETED"],
        "inference": ["INFERENCE_COMPLETED"],
    }
    if status in success_statuses.get(subcmd, []):
        sys.exit(0)

    if subcmd == "evaluate" and "metrics" in res and not str(res.get("status", "")).startswith("BLOCKED"):
        sys.exit(0)

    if subcmd == "inference" and "collections" in res and not str(res.get("status", "")).startswith("BLOCKED"):
        sys.exit(0)

    print(f"\nERROR: Subcommand {subcmd} completed with non-success status '{status}'.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
