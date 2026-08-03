"""Shared constants + helpers for task #1776 (J_{C->A} Jacobian vs fitted ridge M').

Phase-0 infrastructure (plan v4 §4/§10):
  - ONE tensor-slot convention: block-k OUTPUT (= HF ``output_hidden_states[k+1]``);
    SOURCE layer ell_in = 14, READOUT layer L' = 19 (§4 slot-pinning block).
  - ONE data-repo revision pin resolved at run start and threaded into every
    ``list_repo_tree`` / ``hf_hub_download`` (§10 staging recipe).
  - Vendored ``anthropics/jacobian-lens`` under ``external/jacobian-lens``
    (commit recorded in VENDOR_INFO.txt; Apache-2.0).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue779_common as C  # noqa: E402

ISSUE = 1776
HF_PREFIX = "issue1776_jacobian"
HF_DATA_REPO = C.HF_DATA_REPO

# §4 slot-pinning block: all three slots (J differentiation, cx_last capture,
# DeltaHook edit) are the block-14 output; readout v_x is mean-over-answer at
# the block-19 output. A same-layer (ell_in == L') context->answer Jacobian is
# structurally ZERO — never collapse these two.
SOURCE_LAYER = 14
READOUT_LAYER = 19
assert SOURCE_LAYER < READOUT_LAYER, "ell_in must sit strictly below L' (MF-1, plan §4)"

# Phase 0.5 / 4 / 5a refit grid: #779's 23-pt grid extended 5 decades LOWER
# because the shipped M's lambda=1e-3 was LOW-edge-selected (plan §11).
EXTENDED_LAMBDA_GRID: list[float] = [float(x) for x in np.logspace(-6.0, 8.0, 28)]
assert len(EXTENDED_LAMBDA_GRID) == 28

# Plan §12 assumption 3: the shipped 963k ridge (lambda=1e-3, layer 19) must
# reproduce its COMMITTED test_r2 on the pinned test-1000 before any J
# comparison. Source of the pinned value: eval_results/issue_779/
# fitter-fair-comparison-n1m/n1m_fits.json -> per_point.mixed_1m.predictors
# .ridge.whole_map_r2 (n_train=963,444, selected_lambda=0.001), main tree.
SHIPPED_M_TEST_R2_REF = 0.7541708417500051
SHIPPED_M_TEST_R2_TOL = 0.005

# Reused #779 HF stems (plan §10 Reproducibility Card), probed AT THE PIN
# before dispatch — scoped list_repo_tree, >=1 file per stem (#1345).
PASS_B_HF_PATH = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
REUSED_HF_STEMS: tuple[str, ...] = (
    "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture",
    "issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest",
    "issue779_monitoring/n1m_readout/weights",
    "issue779_monitoring/r_b",
    "issue779_monitoring/analysis_tensors/pass_a",
    "issue779_monitoring/analysis_tensors/pass_b",
)

# Plan §10 pairwise-provenance chain (check (j)) — last-commit dates at the pin
# must be monotone: consumed input predates the artifact fit/selected on it.
PROVENANCE_CHAIN: tuple[str, ...] = (
    "issue779_monitoring/analysis_tensors/pass_b",
    "issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest",
    "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture",
    "issue779_monitoring/n1m_readout/weights",
)

DATA_DIR = PROJECT_ROOT / "data" / f"issue_{ISSUE}"
PIN_FILE = DATA_DIR / "data_repo_pin.json"
JLENS_DIR = PROJECT_ROOT / "external" / "jacobian-lens"


def add_jlens_path() -> None:
    """Make the vendored ``jlens`` package importable; fail loud if absent."""
    vendor_info = JLENS_DIR / "VENDOR_INFO.txt"
    if not vendor_info.exists():
        raise FileNotFoundError(
            f"vendored jacobian-lens missing at {JLENS_DIR} (expected VENDOR_INFO.txt; "
            "re-vendor per plan §10 External reuse)"
        )
    if str(JLENS_DIR) not in sys.path:
        sys.path.insert(0, str(JLENS_DIR))


def jlens_commit() -> str:
    """The vendored jacobian-lens commit sha recorded at vendor time."""
    for line in (JLENS_DIR / "VENDOR_INFO.txt").read_text().splitlines():
        if line.startswith("commit:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"{JLENS_DIR}/VENDOR_INFO.txt has no 'commit:' line")


def git_commit() -> str:
    """Current git commit of this checkout (reproducibility metadata)."""
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def repro_meta() -> dict:
    """Reproducibility metadata block for every result JSON (CLAUDE.md rule)."""
    import torch
    import transformers

    return {
        "issue": ISSUE,
        "git_commit": git_commit(),
        "jlens_commit": jlens_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version.split()[0],
    }


def resolve_data_repo_pin(pin_file: Path = PIN_FILE, *, refresh: bool = False) -> str:
    """Resolve ONE data-repo revision at run start and persist it (plan §10).

    Every later ``list_repo_tree`` / ``hf_hub_download`` threads this sha so
    all staged files come from one snapshot. An existing pin file is REUSED
    (run-start pin semantics) unless ``refresh=True`` — a pin refresh re-opens
    the provenance check (j) per the plan.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    if pin_file.exists() and not refresh:
        pin = json.loads(pin_file.read_text())
        assert pin["repo_id"] == HF_DATA_REPO, pin
        return str(pin["revision"])
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = hub.retry_transient(
        lambda: api.repo_info(HF_DATA_REPO, repo_type="dataset"),
        what=f"repo_info({HF_DATA_REPO})",
    )
    revision = str(info.sha)
    pin_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = pin_file.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "repo_id": HF_DATA_REPO,
                "revision": revision,
                "resolved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_commit": git_commit(),
            },
            indent=2,
        )
    )
    os.replace(tmp, pin_file)
    return revision


def atomic_write_json(path: Path, obj: dict) -> None:
    """Atomic JSON write (tmp + os.replace); tmp is PER-PROCESS unique so
    concurrent shard writers of identical content cannot truncate each other
    mid-write (the phase-3 multi-shard manifest race)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)
