#!/usr/bin/env python3
"""Issue #2215 ``discrimination-battery-expansion`` — pod driver (plan v6 §4.3).

Phases (1x H100 ``eval`` pod; every phase resumable, checkpoint-per-phase):

* ``G-check`` — zero-GPU structural gate over the frozen values payload
  (``bank_dbe_values.json``): ``validate_values`` + bank-manifest coverage
  arithmetic + ``PairTable.from_bank`` / ``build_cell_views`` exercised on both
  cell schemas (3-value constructed + 2-value benchmark).
* ``A`` — staging + gates (CPU): stage the 9 reused map payloads + the parent
  ``vc_bank`` pair at pin ``PIN_MAPS`` and the parent ``va2215`` shards at pin
  ``PIN_VA2215`` into ``<out_root>/staged/<repo path>`` (verbatim mirror — no
  staging transformation; consumers open fetch destinations). Fail-loud gates:
  ``verify_reused_artifact_keys.py`` on all 9 payloads, ridge-schema +
  ``apply_map`` roundtrip probe (8 x 3584 finite output), bank coverage
  (production: 396 contexts / 324 pairs), ``assert_out_root_headroom``.
* ``B1`` — context capture + M2 (GPU): render the bank contexts (chat template,
  generation prompt included), pe via ``prefix_end_index_multi``, capture
  v_ce / v_pe at all layers -> ``vc_bank_dbe.pt`` + ``bank_dbe.json`` (the
  SINGLE analysis source, carrying the REALIZED pe-eligibility map). M2 is the
  mechanical per-pair rendered-prefix-token-id comparison; asserts: (i)
  pe-aggregate pairs differ in >=1 prefix token, (ii) per-cell homogeneity,
  (iii) degenerate cells additionally pass a capture-path ||dv_pe|| ~ 0 sanity
  check, (iv) realized == registered expectation per cell (violation = halt).
* ``B2`` — anchors (GPU, pilot-gated): K draws per context at temp 1.0
  (HF ``generate_batch`` — the inherited parent rig), rollout TEXT persisted
  per cell BEFORE capture (#779), dual-pooling answer states
  (``capture_answer_states(..., tail_inclusive=True)`` — span AND tail_incl
  keys persisted, unlike the parent's span-only store). Gate 2 (throughput
  pilot, first production-shape chunk; ceiling ``PILOT_WALL_CEIL_H``) halts
  rc=``RC_PILOT_GATE`` with ``pilot_gate_report.json``; per-cell cap-hit
  > ``CAP_HIT_REGEN_FRAC`` re-generates that cell at ``REGEN_MAX_NEW`` (the
  capped rows stay persisted). Gate 4 (M1 capture spot-parity) re-captures 3
  sampled answer rows + 3 sampled contexts through an INDEPENDENT single-row
  teacher-forced forward (never ``capture_answer_states``), per-row flattened
  cosine >= ``PARITY_COS_MIN`` per pooling plus EXACT span-boundary /
  tail-token-id metadata equality; any miss halts rc=``RC_PARITY_GATE`` with
  ``parity_gate_report.json``. B2-END (#825 ordering): ``upload_dir_sharded``
  (proactive_overflow=True, verify=True) for the va_dbe / vc_bank_dbe /
  anchors-text / manifests families + the ``va_dbe_uploaded.json`` sentinel —
  BEFORE any C analysis; an upload-verification failure raises here.
* ``C`` — analysis (CPU on pod): subprocess into
  ``scripts/issue2215_dbe_analysis.py`` (unit 3 of the pre-split build) behind
  a file-existence + upload-sentinel guard. C' ARG CONVENTION (unit 3
  conforms): ``--bank <bank_dbe.json> --vc <vc_bank_dbe.pt> --va-dir <dir>
  --staged <staged root> --null-out <dir> --out-dir <eval dir> --figures-dir
  <dir> [--null-b N] [--smoke] [--tiny]``. The refusal manipulation-check
  judge lives in the analysis driver (plan §4.3), not here.
* ``D`` — finalize: canonical figure re-render + the plan §6.5
  REGISTERED-deliverable assert (exact eval JSONs + hero/joint figure stems)
  BEFORE any egress leg, then upload null/per-draw matrices to
  ``<prefix>/analysis_tensors/null_matrices/`` AND the C' driver's persisted
  prediction/target tensors to ``<prefix>/analysis_tensors/predictions/``
  (plan §4.3 C': every registered row recomputable post-hoc), git + HF
  results egress, and ONLY after EVERY upload leg (manifests included)
  verifies: ``upload_done.json`` + the regime-stamped results sentinel
  (``/workspace/logs/issue-2215-dbe-results.json``), then ``[phase=done]``.
  ``--force`` on B2/C/D first QUARANTINES that phase's completion records
  (per-cell B2 manifests preserved), so a crashed forced rerun never leaves
  stale records eligible for the next resume.

Upload prefix: ``issue2215_dbe/`` (smoke: ``issue2215_dbe/smoke/``) — NEVER the
parent round's ``issue2215_reprshift`` (``issue2215_run.py:108``). Reads of
parent artifacts stay on their own prefixes at their pins.

Canonical smoke (same entrypoint, same phases, same upload surface):
``--phase all --smoke`` == ``--cells user_role_identity,user_sentiment
--draws 2 --null-b 100`` (one 3-value pe-eligible + one 2-value pe-degenerate
cell). ``--tiny`` swaps in a from-config tiny model on CPU (wiring smoke).

Pod-side contract: markers ONLY via the results sentinel + ``[phase=...]``
breadcrumbs — this script NEVER shells out to ``scripts/task.py``.
Content hygiene: XSTest prompt text is never printed/logged — counts + ids
only (realized text lives only inside the bank/values artifacts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps + API keys)

import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:  # script mode puts scripts/ first already;
    sys.path.insert(0, str(_SCRIPTS_DIR))  # keep -c / pytest imports working too

import issue2162_run as R  # noqa: E402
import issue2215_analysis as ANA  # noqa: E402

from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)
from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    generate_batch,
)
from explore_persona_space.experiments.issue2162 import bank2162 as B2162  # noqa: E402
from explore_persona_space.experiments.issue2215 import bank_dbe as DBE  # noqa: E402
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    stage_hub_file,
    stage_hub_prefix,
)
from explore_persona_space.orchestrate.preflight import (  # noqa: E402
    assert_out_root_headroom,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from explore_persona_space.orchestrate.upload_sharded import (  # noqa: E402
    upload_dir_sharded,
)

logger = logging.getLogger("issue2215.dbe")

# ── constants (plan v6 §4.3 / §7 / §9 / §10) ──────────────────────────

ISSUE = 2215
ROUND = "discrimination-battery-expansion"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# WRITE-side destination override (the parent's #2304 escape hatch — the
# canonical data repo sits at HF's 1M-file cap; upload_dir_sharded's
# proactive/reactive overflow routing is the primary handler).
HF_DATA_WRITE_REPO = os.environ.get("EPM_2215DBE_DATA_WRITE_REPO", HF_DATA_REPO)
# NEVER the parent round's HF_PREFIX_2215 ("issue2215_reprshift", issue2215_run.py:108).
HF_PREFIX_DBE = "issue2215_dbe"

# Reused-artifact pins (plan §4.3/§10 — copied verbatim from the plan).
PIN_MAPS = "dc8108ab84f33695bbc769da0e6e8e2327f51eeb"
PIN_VA2215 = "47ca8e79d3660073746e590afdbf41782c793a3a"
MAP_LAYERS = (14, 19, 26)
MAP_PAYLOAD_PATHS: tuple[str, ...] = tuple(
    [f"issue779_monitoring/n1m_readout/weights/L{lay}/ridge.pt" for lay in MAP_LAYERS]
    + [
        f"issue1738_multiturn/analysis_tensors/weights/L{lay}/{kind}_ridge.pt"
        for lay in MAP_LAYERS
        for kind in ("context", "prefix")
    ]
)
VC_BANK_PARENT_PATHS = (
    "issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json",
    "issue2162_ctxinfo/analysis_tensors/vc_bank/vc_bank.pt",
)
VA2215_PREFIX = "issue2215_reprshift/analysis_tensors/va2215"  # READ-side parent prefix
N_VA2215_FILES = 16  # plan-time scoped list_repo_tree at PIN_VA2215
RIDGE_KEYS = ("xmu", "xsd", "ymu", "W")  # apply_map ridge payload contract

HIDDEN = 3584
N_LAYERS = 28
ANCHOR_DRAWS = 10
ANCHOR_MAX_NEW = 2048
REGEN_MAX_NEW = 4096  # cap-hit re-gen at >= 2x the cap (#1332/#1426/#1481)
CAP_HIT_REGEN_FRAC = 0.02
# Plan-pinned inherited generation seed (plan v6 §2/§4/§10/§11: per-draw
# torch.manual_seed(seed_base + i) with seed_base=42 — recipe fidelity with the
# parent's banked 14,040 rollouts; issue2162_run.py SEED_BASE). Regression-pinned
# by tests/test_issue2215_dbe_stats.py::test_cli_default_seed_base_is_plan_pinned.
SEED_BASE = 42

# Production coverage (plan §4.3 gate): 7 constructed x 12 carriers x 3 values
# + 2 benchmark x 36 items x 2 values = 396 contexts; pairs 7x12x3 + 2x36x1 = 324.
EXPECTED_FULL_CONTEXTS = 396
EXPECTED_FULL_PAIRS = 324

PILOT_WALL_CEIL_H = 3.0  # plan §7 gate 2 (basis 1.20 s/rollout -> 1.32 h projected)
HEADROOM_A_GB = 9.0  # plan §9 floors
HEADROOM_B2_GB = 5.0
PARITY_COS_MIN = 0.9999  # plan §7 gate 4 (span-mean class; #1005 precedent)
# M2 (iii) degenerate-cell dv_pe sanity: identical prefix TOKENS by construction,
# so v_pe differs only by bf16 batch-composition jitter (a real pe-index bug
# reads cosine ~0.4-0.8; #779 calibration) — per-layer cosine + relative norm.
DEGEN_PE_COS_MIN = 0.995
DEGEN_PE_RELNORM_MAX = 0.10

RC_OK = 0
RC_PILOT_GATE = 7  # gate 2 refusal — artifact-routed halt (pilot_gate_report.json)
RC_PARITY_GATE = 8  # gate 4 refusal — artifact-routed halt (parity_gate_report.json)

SENTINEL_NAME = "issue-2215-dbe-results.json"
SENTINEL_NAME_SMOKE = "issue-2215-dbe-smoke-results.json"

# Plan §6.5 REGISTERED deliverables (dbe-primary-artifact-egress): phase D
# refuses EVERY egress leg while any of these is missing — a partial result
# set must never commit, upload, or signal success. The figure floor is the
# two plan-§6.5 HERO/JOINT stems (savefig_paper writes <stem>.png).
REQUIRED_EVAL_JSONS = (
    "dv3_dbe_map_discrimination.json",
    "qualitative_examples.json",
    "datagen_manifest.json",
)
REQUIRED_FIGURE_PNGS = ("dbe_hero_pertype_2afc.png", "dbe_joint_taxonomy_48.png")
# The C-PRODUCED subset the phase-C resume guard requires exactly (never
# any-JSON: in production the pre-existing committed datagen_manifest.json
# alone would satisfy an any-glob check). datagen_manifest_final.json is the
# B1'-finalized copy C lands beside the eval JSONs for the D git leg.
REQUIRED_C_EVAL_JSONS = (
    "dv3_dbe_map_discrimination.json",
    "qualitative_examples.json",
    "datagen_manifest_final.json",
)

SMOKE_CELLS = ("user_role_identity", "user_sentiment")
PHASES = ("G-check", "A", "B1", "B2", "C", "D")


# ── config ────────────────────────────────────────────────────────────


@dataclass
class DbeConfig:
    """Duck-types the issue2162 RunConfig fields the reused helpers read
    (``load_model_and_tokenizer``: model_id/tiny/hidden/n_layers/device;
    ``capture_answer_states``: layers/hidden/device/capture_batch)."""

    phase: str
    out_root: Path
    log_dir: Path
    values_path: Path | None
    smoke: bool
    tiny: bool
    cells: tuple[str, ...] | None
    draws: int
    null_b: int | None
    gen_batch: int
    capture_batch: int
    max_new_tokens: int
    seed_base: int
    model_id: str = DBE.MODEL_ID
    hidden: int = HIDDEN
    n_layers: int = N_LAYERS
    layers: list[int] = field(default_factory=list)
    device: str = "cpu"
    # Bypass PHASE-LEVEL entry-completion guards (B2 all-complete skip, C, D)
    # AND quarantine that phase's completion records at entry
    # (dbe-force-stale-completion). Per-cell resume manifests stay honored.
    force: bool = False
    # Resolved HF revision (main -> commit sha) of the consumed model, set
    # ONCE at run start by main() via _resolve_model_revision
    # (dbe-resume-fingerprint-inputs: the inherited rig loads revision-
    # unpinned HF main, so changed model bytes must invalidate every resume
    # fingerprint). Default is the test/offline sentinel.
    model_revision: str = "unresolved"

    @property
    def staged_dir(self) -> Path:
        return self.out_root / "staged"

    @property
    def hf_dir(self) -> Path:
        return self.out_root / "hf"

    @property
    def vc_dir(self) -> Path:
        return self.hf_dir / "analysis_tensors" / "vc_bank_dbe"

    @property
    def va_dir(self) -> Path:
        return self.hf_dir / "analysis_tensors" / "va_dbe"

    @property
    def anchors_dir(self) -> Path:
        return self.hf_dir / "raw_completions" / "anchors"

    @property
    def null_dir(self) -> Path:
        return self.hf_dir / "analysis_tensors" / "null_matrices"

    @property
    def predictions_dir(self) -> Path:
        # The C' driver writes prediction/target tensors to
        # null_dir.parent / "predictions" (issue2215_dbe_analysis.py run()).
        return self.hf_dir / "analysis_tensors" / "predictions"

    @property
    def manifest_dir(self) -> Path:
        # Registered-contract note (plan §9 ``phase_outputs``): the plan names
        # the phase sentinels directly under ``/workspace/eps2215dbe/out/``;
        # the realized layout nests them under ``<out_root>/manifests/`` (this
        # property). Pod-local driver state only — per the plan §9 backend-pin
        # note these paths are consumed by this driver itself, never the
        # marker-drain contract. Deviation recorded here + in the round marker.
        return self.out_root / "manifests"

    @property
    def hf_prefix(self) -> str:
        return f"{HF_PREFIX_DBE}/smoke" if (self.smoke or self.tiny) else HF_PREFIX_DBE

    @property
    def eval_dir(self) -> Path:
        if self.smoke or self.tiny:
            return self.out_root / "smoke_eval"  # never overwrite committed paths
        return _REPO_ROOT / "eval_results" / "issue_2215" / ROUND

    @property
    def figures_dir(self) -> Path:
        if self.smoke or self.tiny:
            return self.out_root / "smoke_figures"
        return _REPO_ROOT / "figures" / "issue_2215"


_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── small helpers ─────────────────────────────────────────────────────


def _repro(cfg: DbeConfig, phase: str) -> dict:
    """Reproducibility metadata block (git provenance + run identity, #2194)."""
    import transformers

    md = as_metadata_dict(git_provenance(), phase=phase)
    md.update(
        {
            "issue": ISSUE,
            "round": ROUND,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "seed_base": cfg.seed_base,
            "draws": cfg.draws,
            "smoke": cfg.smoke,
            "tiny": cfg.tiny,
            "cells": list(cfg.cells) if cfg.cells else "all",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    return md


def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode iteration — never ``splitlines()`` (U+2028 shred, #950)."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.float().flatten()
    bv = b.float().flatten()
    return float(torch.dot(av, bv) / (av.norm() * bv.norm()).clamp_min(1e-12))


def _load_values(cfg: DbeConfig) -> dict:
    return DBE.load_values(cfg.values_path)


def _filter_bank(bank: dict, cells: tuple[str, ...]) -> dict:
    """Cell-subset view for smoke — the subset threads through EVERY phase."""
    unknown = [c for c in cells if c not in DBE.TYPES]
    assert not unknown, f"unknown cells: {unknown}"
    missing = [c for c in cells if c not in bank["kept_types"]]
    assert not missing, f"cells not kept in the bank: {missing}"
    scoped = dict(bank)
    scoped["cells"] = {c: bank["cells"][c] for c in cells}
    scoped["contexts"] = {cid: c for cid, c in bank["contexts"].items() if c["cell"] in cells}
    scoped["pairs"] = [p for p in bank["pairs"] if p["cell"] in cells]
    scoped["kept_types"] = [c for c in bank["kept_types"] if c in cells]
    scoped["degenerate_at_pe_cells"] = [c for c in bank["degenerate_at_pe_cells"] if c in cells]
    scoped["expected_pe_eligibility"] = {
        c: v for c, v in bank["expected_pe_eligibility"].items() if c in cells
    }
    scoped["scoped_cells"] = list(cells)
    return scoped


def _coverage_gate(cfg: DbeConfig, values: dict, bank: dict) -> dict:
    """Plan §4.3 A' new-bank coverage gate (fail-loud)."""
    n_ctx, n_pairs = len(bank["contexts"]), len(bank["pairs"])
    exp_ctx = sum(m["n_carriers"] * len(m["values"]) for m in bank["cells"].values())
    exp_pairs = sum(
        m["n_carriers"] * math.comb(len(m["values"]), 2) for m in bank["cells"].values()
    )
    assert (n_ctx, n_pairs) == (exp_ctx, exp_pairs), (n_ctx, n_pairs, exp_ctx, exp_pairs)
    production = not (
        cfg.smoke or cfg.tiny or cfg.cells or values.get("smoke") or values.get("dry_run")
    )
    if production:
        assert (n_ctx, n_pairs) == (EXPECTED_FULL_CONTEXTS, EXPECTED_FULL_PAIRS), (
            n_ctx,
            n_pairs,
            "below the plan §4.3 396/324 coverage gate — datagen shortfall, halt",
        )
    # registered expectation coherence (7 eligible / 2 degenerate, restricted to kept)
    reg = DBE.expected_pe_eligibility()
    for cell in bank["kept_types"]:
        assert bank["expected_pe_eligibility"][cell] == reg[cell], cell
    # exercise the reused analysis-core loaders on BOTH schemas (3-value + 2-value)
    pt = ANA.PairTable.from_bank(bank, tuple(bank["kept_types"]))
    views = ANA.build_cell_views(bank, pt)  # complete-grid assert lives inside
    n3 = sum(1 for m in bank["cells"].values() if len(m["values"]) == 3)
    n2 = sum(1 for m in bank["cells"].values() if len(m["values"]) == 2)
    report = {
        "n_contexts": n_ctx,
        "n_pairs": n_pairs,
        "n_cells": len(bank["cells"]),
        "n_three_value_cells": n3,
        "n_two_value_cells": n2,
        "n_views": len(views),
        "production_gate": production,
        "judge_pass_pairs": {c: m["kept_pairs"] for c, m in bank["cells"].items()},
    }
    logger.info("[coverage] %s", json.dumps(report, sort_keys=True))
    return report


def _bank_for(cfg: DbeConfig, values: dict) -> dict:
    bank = DBE.bank_manifest_dbe(values)
    if cfg.cells:
        bank = _filter_bank(bank, cfg.cells)
    return bank


def _load_bank_dbe(cfg: DbeConfig) -> dict:
    """The persisted single analysis source (B1 output)."""
    path = cfg.vc_dir / "bank_dbe.json"
    if not path.exists():
        raise RuntimeError(f"{path} missing — run --phase B1 first")
    return json.loads(path.read_text())


# ── regime fingerprints + resume predicates ──────────────────────────
# Machine-stable keys (#1336): generating parameters + sha256 of BIT-EXACT
# files read from disk (values / bank JSON) — never recomputed float bytes.


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _values_sha(cfg: DbeConfig) -> str:
    """sha256 of the frozen values file (the B1/B2 data-input identity)."""
    p = (
        cfg.values_path
        if cfg.values_path is not None
        else Path(DBE.__file__).resolve().parent / DBE.VALUES_FILENAME
    )
    if not p.exists():
        raise RuntimeError(f"{p} missing — run scripts/issue2215_dbe_datagen.py (Phase G) first")
    return _file_sha256(Path(p))


def _bank_sha(cfg: DbeConfig) -> str:
    """sha256 of bank_dbe.json (the realized B1 output identity)."""
    path = cfg.vc_dir / "bank_dbe.json"
    if not path.exists():
        raise RuntimeError(f"{path} missing — run --phase B1 first")
    return _file_sha256(path)


def _b2_regime_fp(cfg: DbeConfig) -> str:
    """Run-level output-identity fingerprint for B2/C/D resume + the C-entry
    sentinel validation (dbe-phase-c-sentinel-regime / dbe-resume-fingerprint-
    inputs): model id + RESOLVED model revision (main -> sha, run-start) +
    smoke/tiny state + cell scope + generation params INCLUDING gen_batch
    (generate_batch reseeds per batch and samples batched rows, so batch
    composition is output-affecting) + exact layer list + the frozen-values
    AND realized-bank content hashes."""
    return json.dumps(
        {
            "round": ROUND,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "tiny": cfg.tiny,
            "smoke": cfg.smoke,
            "cells": list(cfg.cells) if cfg.cells else "all",
            "draws": cfg.draws,
            "seed_base": cfg.seed_base,
            "base_max_new_tokens": cfg.max_new_tokens,
            "gen_batch": cfg.gen_batch,
            "layers": list(cfg.layers),
            "values_sha256": _values_sha(cfg),
            "bank_sha256": _bank_sha(cfg),
        },
        sort_keys=True,
    )


def _pilot_regime_fp(cfg: DbeConfig) -> str:
    """Gate-2 pilot fingerprint: the B2 output-identity fp PLUS the throughput-
    affecting execution shape (gen_batch — also inside the B2 fp since round 3;
    kept explicit here for auditability)."""
    return json.dumps(
        {"b2_regime_fp": _b2_regime_fp(cfg), "gen_batch": cfg.gen_batch},
        sort_keys=True,
    )


def _resolve_model_revision(cfg: DbeConfig) -> str:
    """Resolve the consumed model's HF revision (main -> commit sha) ONCE at
    run start (dbe-resume-fingerprint-inputs). Production/smoke fail LOUD on
    an unresolvable revision (the pod has network by construction — it
    downloads the model); ``--tiny`` (from-config wiring smoke, possibly
    offline) degrades to the sentinel string ``unresolved-tiny`` with a
    logged warning (enumerated smoke blind spot)."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(cfg.model_id)
        sha = getattr(info, "sha", None)
        if not sha:
            raise RuntimeError(f"model_info({cfg.model_id!r}) returned no sha")
        return str(sha)
    except Exception as exc:
        if cfg.tiny:
            logger.warning("[config] model revision unresolved under --tiny: %s", exc)
            return "unresolved-tiny"
        raise RuntimeError(
            f"cannot resolve the HF revision for {cfg.model_id!r} at run start "
            f"(dbe-resume-fingerprint-inputs): {exc}"
        ) from exc


def _read_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("[resume] %s unparseable — treating the leg as NOT complete", path)
        return None


def _report_regime_ok(path: Path, regime: str, *, require_pass_verdict: bool = False) -> bool:
    """A phase report/sentinel satisfies resume ONLY when its persisted
    regime_fp matches the CURRENT run's fingerprint (smoke-pilot-cross-regime-
    reuse class: bare existence is never completion evidence). With
    ``require_pass_verdict`` (verdict-bearing gate reports), the persisted
    verdict must ALSO be exactly "PASS" — an fp-matched FAILED or
    verdict-less report never licenses resume
    (resume-accepts-failed-gate-report)."""
    rec = _read_report(path)
    if rec is None or rec.get("regime_fp") != regime:
        return False
    if require_pass_verdict and rec.get("verdict") != "PASS":
        return False
    return True


def _pilot_report_ok(cfg: DbeConfig) -> bool:
    """Gate-2 resume predicate: the pilot report satisfies resume ONLY when
    (a) its regime fingerprint (incl. smoke/tiny + bank/values identity +
    generation params + gen_batch) matches the current run, (b) its persisted
    verdict is exactly "PASS" — a FAILED (or verdict-less) gate report must
    never suppress re-measurement, in ANY regime: the gate would otherwise be
    disabled exactly after it fired (dbe-failed-pilot-resume) — AND (c) for a
    PRODUCTION run it is a non-demoted report — a demoted smoke/tiny pilot
    must never suppress the production throughput gate."""
    rec = _read_report(cfg.manifest_dir / "pilot_gate_report.json")
    if rec is None or rec.get("regime_fp") != _pilot_regime_fp(cfg):
        return False
    if rec.get("verdict") != "PASS":
        return False
    if not (cfg.smoke or cfg.tiny) and rec.get("demoted_to_informational"):
        return False
    return True


def _cell_complete(cfg: DbeConfig, cell: str) -> bool:
    """Per-cell B2 resume predicate (regime-fingerprinted done-manifest +
    both per-cell artifacts present)."""
    done = _read_report(cfg.manifest_dir / f"anchors_dbe_{cell}_done.json")
    return (
        done is not None
        and done.get("regime_fp") == _cell_regime_fp(cfg, cell)
        and (cfg.va_dir / f"va_dbe_w0_{cell}.pt").exists()
        and (cfg.anchors_dir / f"anchors_dbe_w0_{cell}.jsonl").exists()
    )


# Phase-level completion records --force must invalidate at entry
# (dbe-force-stale-completion). Per-cell B2 manifests are deliberately
# ABSENT: per-cell resume stays honored under --force.
_PHASE_COMPLETION_RECORDS: dict[str, tuple[str, ...]] = {
    "B2": ("pilot_gate_report.json", "parity_gate_report.json", "va_dbe_uploaded.json"),
    "C": ("analysis_done.json",),
    "D": ("upload_done.json",),
}


def _invalidate_phase_records(cfg: DbeConfig, phase: str) -> list[str]:
    """dbe-force-stale-completion: a FORCED phase entry quarantines that
    phase's own completion records BEFORE any work begins, so a crashed
    forced rerun can never leave stale fp-matched records eligible for the
    next non-force invocation. Records move (same-fs atomic ``replace``;
    cross-fs fallback for the log-dir sentinel) into
    ``<out_root>/quarantine/`` — forensics preserved, resume-glob escape
    guaranteed. Phase D additionally quarantines the results sentinel (bare
    AND poller-drained ``.processed``)."""
    qdir = cfg.out_root / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    targets = [cfg.manifest_dir / name for name in _PHASE_COMPLETION_RECORDS[phase]]
    if phase == "D":
        s = _sentinel_path(cfg)
        targets += [s, s.with_name(s.name + ".processed")]
    moved: list[str] = []
    for t in targets:
        if not t.exists():
            continue
        dest = qdir / f"{time.time_ns()}-{t.name}"
        try:
            t.replace(dest)
        except OSError:
            shutil.move(str(t), str(dest))
        moved.append(t.name)
    if moved:
        logger.info(
            "[force] phase %s: quarantined stale completion records -> %s: %s",
            phase,
            qdir,
            ",".join(moved),
        )
    else:
        logger.info("[force] phase %s: no prior completion records to invalidate", phase)
    return moved


# ── phase G-check ─────────────────────────────────────────────────────


def phase_gcheck(cfg: DbeConfig) -> int:
    logger.info("[phase=G-check]")
    values = _load_values(cfg)
    bank = _bank_for(cfg, values)
    report = _coverage_gate(cfg, values, bank)
    report["repro"] = _repro(cfg, "g-check")
    R._write_json_atomic(cfg.manifest_dir / "gcheck_report.json", report)
    logger.info("[g-check] OK — %d cells / %d contexts", report["n_cells"], report["n_contexts"])
    return RC_OK


# ── phase A: staging + gates ──────────────────────────────────────────


def phase_stage(cfg: DbeConfig) -> int:
    logger.info("[phase=A]")
    assert_out_root_headroom(cfg.out_root, HEADROOM_A_GB, phase="A-stage")
    staged = cfg.staged_dir
    staged.mkdir(parents=True, exist_ok=True)
    for path in MAP_PAYLOAD_PATHS + VC_BANK_PARENT_PATHS:
        stage_hub_file(HF_DATA_REPO, path, staged / path, revision=PIN_MAPS)
    stage_hub_prefix(HF_DATA_REPO, VA2215_PREFIX, staged, revision=PIN_VA2215)
    va_files = sorted(p.name for p in (staged / VA2215_PREFIX).iterdir() if p.is_file())
    assert len(va_files) == N_VA2215_FILES, (len(va_files), va_files)

    # gate: declared-key verification on all 9 reused map payloads
    verifier = _SCRIPTS_DIR / "verify_reused_artifact_keys.py"
    assert verifier.exists(), verifier
    for path in MAP_PAYLOAD_PATHS:
        cmd = [
            sys.executable,
            str(verifier),
            "--artifact",
            str(staged / path),
            "--keys",
            ",".join(RIDGE_KEYS),
        ]
        proc = subprocess.run(cmd, env={**os.environ}, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"verify_reused_artifact_keys rc={proc.returncode} for {path}")

    # gate: ridge schema + apply_map roundtrip probe (8 x 3584 finite output)
    from issue779_ffc_n1m_fits import apply_map  # deferred: heavy sibling chain

    x_probe = torch.randn(
        8, HIDDEN, dtype=torch.float32, generator=torch.Generator().manual_seed(0)
    )
    for path in MAP_PAYLOAD_PATHS:
        # sha-pinned self-produced bundles — weights_only=False is deliberate (#1900)
        payload = torch.load(staged / path, map_location="cpu", weights_only=False)
        missing = [k for k in RIDGE_KEYS if k not in payload]
        assert not missing, (path, missing)
        out = apply_map(payload, x_probe, "cpu")
        assert tuple(out.shape) == (8, HIDDEN), (path, tuple(out.shape))
        assert torch.isfinite(out).all(), (path, "non-finite apply_map output")
    logger.info("[stage] 9 map payloads verified (keys + apply_map roundtrip)")

    # gate: bank coverage (also exercised at G-check; re-run here per plan §4.3)
    values = _load_values(cfg)
    bank = _bank_for(cfg, values)
    coverage = _coverage_gate(cfg, values, bank)
    R._write_json_atomic(
        cfg.manifest_dir / "stage_done.json",
        {
            "pins": {"maps": PIN_MAPS, "va2215": PIN_VA2215},
            "map_payloads": list(MAP_PAYLOAD_PATHS),
            "vc_bank_parent": list(VC_BANK_PARENT_PATHS),
            "va2215_files": va_files,
            "coverage": coverage,
            "repro": _repro(cfg, "stage"),
        },
    )
    return RC_OK


# ── phase B1: context capture + M2 ────────────────────────────────────


def _pe_eligibility(
    bank: dict, ctx_ids: dict[str, list[int]], prefix_ends: dict[str, int]
) -> tuple[dict[str, bool], dict[str, bool]]:
    """M2: mechanical per-pair rendered-prefix-token-id comparison.

    Returns (per_pair eligibility, per_cell eligibility). Halts on a MIXED
    cell (assert ii) or on realized != registered expectation (asserts i/iv:
    a pe-aggregate pair must differ in >=1 prefix token; an expected-eligible
    cell realizing identical prefixes is a datagen structure bug)."""
    per_pair: dict[str, bool] = {}
    flags: dict[str, set[bool]] = {}
    for p in bank["pairs"]:
        a, b = p["a"], p["b"]
        elig = ctx_ids[a][: prefix_ends[a]] != ctx_ids[b][: prefix_ends[b]]
        per_pair[p["pair_id"]] = elig
        flags.setdefault(p["cell"], set()).add(elig)
    per_cell: dict[str, bool] = {}
    for cell, seen in flags.items():
        assert len(seen) == 1, f"M2 assert (ii): mixed pe-eligibility inside cell {cell} — halt"
        realized = next(iter(seen))
        expected = bank["expected_pe_eligibility"][cell]
        assert realized == expected, (
            f"M2 asserts (i)/(iv): cell {cell} realized pe-eligible={realized} but "
            f"registered expectation is {expected} — datagen structure bug, halt"
        )
        per_cell[cell] = realized
    return per_pair, per_cell


@torch.no_grad()
def _capture_contexts(
    cfg: DbeConfig,
    model,
    tok,
    contexts: dict[str, dict],
    ctx_ids: dict[str, list[int]],
    prefix_ends: dict[str, int],
) -> dict[str, dict]:
    """All-layer v_ce / v_pe per context (the parent ``capture_bank`` recipe:
    positions from the token ids' own offsets — BPE-seam rule)."""
    records: dict[str, dict] = {}
    order = sorted(contexts)
    t0 = time.monotonic()
    for start in range(0, len(order), cfg.capture_batch):
        chunk = order[start : start + cfg.capture_batch]
        ids, mask = R._right_pad([ctx_ids[c] for c in chunk], tok.pad_token_id, cfg.device)
        captured = extract_layer_activations(model, ids, cfg.layers, attention_mask=mask)
        for j, cid in enumerate(chunk):
            ctx_len = len(ctx_ids[cid])
            pe = prefix_ends[cid]
            assert 1 <= pe < ctx_len, (cid, ctx_len, pe)
            v_ce = torch.stack([captured[lay][j, ctx_len - 1] for lay in cfg.layers])
            v_pe = torch.stack([captured[lay][j, pe - 1] for lay in cfg.layers])
            assert v_ce.shape == (len(cfg.layers), cfg.hidden), v_ce.shape
            ctx = contexts[cid]
            records[cid] = {
                "context_id": cid,
                "cell": ctx["cell"],
                "value_id": ctx["value_id"],
                "carrier": ctx["carrier"],
                "ctx_len": ctx_len,
                "prefix_end": pe,
                "v_ce": v_ce.float().cpu(),
                "v_pe": v_pe.float().cpu(),
            }
        del captured
        logger.info(
            "[bank] unit %d/%d contexts elapsed=%.1fs",
            min(start + cfg.capture_batch, len(order)),
            len(order),
            time.monotonic() - t0,
        )
    assert len(records) == len(contexts), (len(records), len(contexts))
    return records


def _degenerate_pe_sanity(bank: dict, records: dict[str, dict]) -> dict:
    """M2 assert (iii): degenerate cells share identical prefix TOKENS, so
    ||dv_pe|| ~ 0 up to bf16 batch-composition jitter (real pe-index bugs read
    cosine ~0.4-0.8; #779 calibration)."""
    worst = {"min_layer_cos": 1.0, "max_relnorm": 0.0, "n_pairs": 0}
    for p in bank["pairs"]:
        if p["cell"] not in bank["degenerate_at_pe_cells"]:
            continue
        va = records[p["a"]]["v_pe"]
        vb = records[p["b"]]["v_pe"]
        cos = torch.nn.functional.cosine_similarity(va, vb, dim=1)  # per layer
        rel = (va.norm(dim=1) - vb.norm(dim=1)).abs() / vb.norm(dim=1).clamp_min(1e-6)
        c_min, r_max = float(cos.min()), float(rel.max())
        worst["min_layer_cos"] = min(worst["min_layer_cos"], c_min)
        worst["max_relnorm"] = max(worst["max_relnorm"], r_max)
        worst["n_pairs"] += 1
        assert c_min >= DEGEN_PE_COS_MIN and r_max <= DEGEN_PE_RELNORM_MAX, (
            p["pair_id"],
            c_min,
            r_max,
            "M2 assert (iii): degenerate-cell dv_pe not ~0 — capture-path bug, halt",
        )
    return worst


def _bank_builder_sha() -> str:
    """Content sha of the bank-builder module B1 executes
    (dbe-resume-fingerprint-inputs: changed bank-building LOGIC must
    invalidate the captured context states, not only changed values)."""
    return _file_sha256(Path(DBE.__file__).resolve())


def _bank_regime_fp(cfg: DbeConfig) -> str:
    """Machine-stable B1 resume key — generating parameters + the frozen
    data-input identity + the CONSUMED CODE/MODEL identity (#1336;
    dbe-resume-fingerprint-inputs: regenerated values, an edited bank
    builder, or changed model bytes must each invalidate the captured
    context states)."""
    return json.dumps(
        {
            "round": ROUND,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "tiny": cfg.tiny,
            "smoke": cfg.smoke,
            "cells": list(cfg.cells) if cfg.cells else "all",
            "n_layers": cfg.n_layers,
            "hidden": cfg.hidden,
            "layers": list(cfg.layers),
            "values_sha256": _values_sha(cfg),
            "bank_builder_sha256": _bank_builder_sha(),
        },
        sort_keys=True,
    )


DATAGEN_MANIFEST_SRC = _REPO_ROOT / "eval_results" / "issue_2215" / ROUND / "datagen_manifest.json"


def _datagen_manifest_source(cfg: DbeConfig) -> dict:
    """Committed-source-manifest requirement (dbe-manifest-realized-pe-map /
    dbe-b1-manifest-check-post-init): production REQUIRES the committed
    datagen manifest; smoke/tiny tolerate absence (twin roots — enumerated
    smoke blind spot). Called at B1 ENTRY — before the model load, so an
    omitted sparse cone fails in seconds, never after a full GPU capture —
    and again at finalization."""
    if DATAGEN_MANIFEST_SRC.exists():
        return {
            "present": True,
            "path": str(DATAGEN_MANIFEST_SRC.relative_to(_REPO_ROOT)),
            "sha256": _file_sha256(DATAGEN_MANIFEST_SRC),
        }
    if cfg.smoke or cfg.tiny:
        return {"present": False, "note": "smoke/tiny — committed datagen manifest not staged"}
    raise RuntimeError(
        f"{DATAGEN_MANIFEST_SRC} missing — production B1 requires the committed datagen "
        "manifest (open the eval_results/issue_2215 sparse cone at pod bootstrap)"
    )


def _finalize_datagen_manifest(cfg: DbeConfig, bank: dict) -> Path:
    """B1' manifest finalization (dbe-manifest-realized-pe-map): augment the
    committed datagen manifest with the REALIZED per-pair pe-eligibility map +
    its hash. The realized map is knowable only after B1's rendered-prefix
    token comparison, so finalization lives here, not in datagen. The
    committed-source requirement itself is ``_datagen_manifest_source``
    (validated at B1 entry BEFORE the model load). Phase C verifies EXACT
    agreement with bank_dbe.json (``_assert_realized_pe_manifest``)."""
    per_pair = {p["pair_id"]: bool(p["pe_realized_eligible"]) for p in bank["pairs"]}
    src = _datagen_manifest_source(cfg)
    final = json.loads(DATAGEN_MANIFEST_SRC.read_text()) if src["present"] else {}
    final["source_manifest"] = src
    final["realized_pe_eligibility"] = {
        "per_pair": per_pair,
        "per_cell": dict(bank["realized_pe_eligibility"]),
        "sha256": hashlib.sha256(json.dumps(per_pair, sort_keys=True).encode()).hexdigest(),
        "derived_by": "scripts/issue2215_dbe_run.py phase B1 (_pe_eligibility, M2)",
    }
    out = cfg.vc_dir / "datagen_manifest_final.json"
    R._write_json_atomic(out, final)
    logger.info("[bank] finalized datagen manifest -> %s (%d pairs)", out, len(per_pair))
    return out


def _assert_realized_pe_manifest(final: dict, bank: dict) -> None:
    """C-entry check (dbe-manifest-realized-pe-map): the finalized datagen
    manifest carries EXACTLY one realized pe-eligibility entry per bank pair,
    in exact agreement with bank_dbe.json (per-pair, per-cell, and hash)."""
    rec = final.get("realized_pe_eligibility")
    assert isinstance(rec, dict), "finalized manifest missing realized_pe_eligibility"
    per_pair = rec.get("per_pair")
    assert isinstance(per_pair, dict), "finalized manifest missing realized per-pair map"
    bank_map = {p["pair_id"]: bool(p["pe_realized_eligible"]) for p in bank["pairs"]}
    missing = sorted(set(bank_map) - set(per_pair))
    extra = sorted(set(per_pair) - set(bank_map))
    assert not missing and not extra, (
        f"finalized manifest per-pair map coverage mismatch: {len(missing)} bank pairs missing, "
        f"{len(extra)} extra entries; missing[:10]={missing[:10]}, extra[:10]={extra[:10]}"
    )
    disagree = sorted(k for k, v in bank_map.items() if per_pair[k] != v)
    assert not disagree, (
        f"finalized manifest per-pair map disagrees with bank_dbe.json on "
        f"{len(disagree)} pairs; first={disagree[:10]}"
    )
    assert rec.get("per_cell") == bank["realized_pe_eligibility"], (
        rec.get("per_cell"),
        bank["realized_pe_eligibility"],
        "finalized manifest per-cell map disagrees with bank_dbe.json",
    )
    digest = hashlib.sha256(json.dumps(per_pair, sort_keys=True).encode()).hexdigest()
    assert rec.get("sha256") == digest, (rec.get("sha256"), digest, "realized pe-map hash drift")


def phase_bank(cfg: DbeConfig) -> int:
    logger.info("[phase=B1]")
    values = _load_values(cfg)
    bank = _bank_for(cfg, values)
    done = cfg.manifest_dir / "bank_done.json"
    regime = _bank_regime_fp(cfg)
    if (
        done.exists()
        and json.loads(done.read_text()).get("regime_fp") == regime
        and (cfg.vc_dir / "bank_dbe.json").exists()
        and (cfg.vc_dir / "vc_bank_dbe.pt").exists()
        and (cfg.vc_dir / "datagen_manifest_final.json").exists()
    ):
        logger.info("[bank] resume: capture complete for this regime — skip")
        return RC_OK
    # dbe-b1-manifest-check-post-init: the production committed-source
    # requirement is validated BEFORE the model load — an omitted sparse cone
    # must fail here in seconds, never after the full GPU capture.
    _datagen_manifest_source(cfg)
    model, tok = R.load_model_and_tokenizer(cfg)
    contexts = bank["contexts"]
    ctx_ids = {cid: B2162.context_token_ids_2162(tok, c) for cid, c in contexts.items()}
    prefix_ends = {cid: B2162.prefix_end_index_multi(tok, ids) for cid, ids in ctx_ids.items()}
    per_pair, per_cell = _pe_eligibility(bank, ctx_ids, prefix_ends)
    records = _capture_contexts(cfg, model, tok, contexts, ctx_ids, prefix_ends)
    degen = _degenerate_pe_sanity(bank, records)

    # persist the REALIZED eligibility map into bank_dbe.json — the single
    # analysis source (plan §4.3 B1'); pairs carry per-pair flags.
    for p in bank["pairs"]:
        p["pe_realized_eligible"] = per_pair[p["pair_id"]]
    bank["realized_pe_eligibility"] = per_cell
    bank["pe_aggregate_cells"] = sorted(c for c, e in per_cell.items() if e)
    bank["m2"] = {
        "asserts": "i:prefix-differs, ii:cell-homogeneity, iii:degenerate-dv_pe~0, iv:realized==expected",
        "degenerate_sanity": degen,
    }
    R._write_json_atomic(cfg.vc_dir / "bank_dbe.json", bank)
    R._save_pt_atomic(
        cfg.vc_dir / "vc_bank_dbe.pt",
        {
            "layers": list(cfg.layers),
            "per_context": records,
            "repro": _repro(cfg, "bank-capture"),
        },
    )
    _finalize_datagen_manifest(cfg, bank)
    R._write_json_atomic(
        done,
        {
            "regime_fp": regime,
            "n_contexts": len(contexts),
            "n_pairs": len(bank["pairs"]),
            "pe_aggregate_cells": bank["pe_aggregate_cells"],
            "degenerate_sanity": degen,
            "repro": _repro(cfg, "bank-capture"),
        },
    )
    logger.info(
        "[bank] captured %d contexts; pe-aggregate cells: %s",
        len(contexts),
        ",".join(bank["pe_aggregate_cells"]) or "none",
    )
    return RC_OK


# ── phase B2: anchors + gates 2/4 + upload ────────────────────────────


def _cell_regime_fp(cfg: DbeConfig, cell: str) -> str:
    """Per-cell B2 resume key: generation params (gen_batch included — batch
    composition is output-affecting under generate_batch's per-batch
    reseeding) + resolved model revision + exact layers + the UPSTREAM data
    identity (bank_dbe.json content hash + the B1 regime fp, which itself
    carries the values + bank-builder hashes) — a changed bank/values/model
    silently reusing stale anchor shards is the dbe-resume-fingerprint-inputs
    class."""
    return json.dumps(
        {
            "round": ROUND,
            "cell": cell,
            "draws": cfg.draws,
            "seed_base": cfg.seed_base,
            "base_max_new_tokens": cfg.max_new_tokens,
            "gen_batch": cfg.gen_batch,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "tiny": cfg.tiny,
            "smoke": cfg.smoke,
            "layers": list(cfg.layers),
            "bank_sha256": _bank_sha(cfg),
            "bank_regime_fp": _bank_regime_fp(cfg),
        },
        sort_keys=True,
    )


def _eval_pilot_gate(cfg: DbeConfig, chunk_rows: int, wall_s: float, total_rollouts: int) -> None:
    """Plan §7 gate 2 — throughput pilot on the first production-shape chunk.

    Artifact-routed halt: report JSON + distinct rc (never a bare rc=1).
    Demoted to informational under --smoke/--tiny (gate-calibration rule: the
    3.0 h ceiling is production-shape-calibrated; a CPU tiny run would
    false-fire it)."""
    s_per = wall_s / max(1, chunk_rows)
    projected_h = s_per * total_rollouts / 3600.0
    demoted = cfg.smoke or cfg.tiny
    verdict = "PASS" if projected_h <= PILOT_WALL_CEIL_H else "FAIL"
    report = {
        "phase": "anchors-pilot",
        "basis": "generation wall of the first production-shape chunk / rows in chunk",
        "measured_s_per_rollout": s_per,
        "chunk_rows": chunk_rows,
        "gen_batch": cfg.gen_batch,
        "draws": cfg.draws,
        "max_new_tokens": cfg.max_new_tokens,
        "total_rollouts": total_rollouts,
        "projected_wall_h": projected_h,
        "ceiling_h": PILOT_WALL_CEIL_H,
        "verdict": verdict,
        "demoted_to_informational": demoted,
        # smoke-pilot-cross-regime-reuse: production B2 resume accepts ONLY a
        # regime-matched, non-demoted report (_pilot_report_ok).
        "regime_fp": _pilot_regime_fp(cfg),
        "repro": _repro(cfg, "anchors-pilot"),
    }
    R._write_json_atomic(cfg.manifest_dir / "pilot_gate_report.json", report)
    logger.info(
        "[gate2] s/rollout=%.3f projected=%.2fh ceiling=%.1fh verdict=%s demoted=%s",
        s_per,
        projected_h,
        PILOT_WALL_CEIL_H,
        verdict,
        demoted,
    )
    if verdict == "FAIL" and not demoted:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(RC_PILOT_GATE)


def _gen_and_capture(
    cfg: DbeConfig,
    model,
    tok,
    contexts_cell: dict[str, dict],
    max_new: int,
    eot: list[int],
    jsonl: Path,
    pilot_state: dict | None,
) -> tuple[list[dict], dict, list[int]]:
    """Generate K draws per context for ONE cell, persist text pre-capture
    (#779), capture dual-pooling answer states, enrich + rewrite the jsonl."""
    order = sorted(contexts_cell)
    ctx_ids = {cid: B2162.context_token_ids_2162(tok, contexts_cell[cid]) for cid in order}
    rows: list[dict] = []
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    ctx_lens: list[int] = []
    t0 = time.monotonic()
    for start in range(0, len(order), cfg.gen_batch):
        chunk = order[start : start + cfg.gen_batch]
        t_chunk = time.monotonic()
        outs = generate_batch(
            model,
            tok,
            [contexts_cell[c] for c in chunk],
            n=cfg.draws,
            hook=None,
            max_new_tokens=max_new,
            temperature=R.ANCHOR_TEMPERATURE,
            seed_base=cfg.seed_base,
            render_fn=B2162.render_context_2162,
            ids_fn=B2162.context_token_ids_2162,
        )
        chunk_rows = 0
        for b, cid in enumerate(chunk):
            ctx = contexts_cell[cid]
            for i, text in enumerate(outs[b]):
                flat_ctx.append(ctx_ids[cid])
                flat_text.append(text)
                ctx_lens.append(len(ctx_ids[cid]))
                rows.append(
                    {
                        "context_id": cid,
                        "cell": ctx["cell"],
                        "value_id": ctx["value_id"],
                        "carrier": ctx["carrier"],
                        "draw": i,
                        "seed": cfg.seed_base + i,
                        "temperature": R.ANCHOR_TEMPERATURE,
                        "max_new_tokens": max_new,
                        "text": text,
                    }
                )
                chunk_rows += 1
        logger.info(
            "[anchors:%s] unit %d/%d contexts elapsed=%.1fs",
            next(iter(contexts_cell.values()))["cell"],
            min(start + cfg.gen_batch, len(order)),
            len(order),
            time.monotonic() - t0,
        )
        if pilot_state is not None and not pilot_state["done"]:
            pilot_state["done"] = True
            _eval_pilot_gate(cfg, chunk_rows, time.monotonic() - t_chunk, pilot_state["total"])
    # rollout TEXT persisted the moment generation completes, BEFORE capture (#779)
    R._write_jsonl_atomic(jsonl, rows)
    # dbe-gate4-metadata-independence: GENERATION-side span metadata, computed
    # at THIS call site from the quantities fed to capture (site tokenization
    # of the persisted text + the ctx ids generation consumed) — never read
    # back from the capture path. shared issue2162_run.capture_answer_states
    # stays untouched (shared-module caution).
    site_comp_lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in flat_text]
    states = R.capture_answer_states(
        cfg,
        model,
        tok,
        flat_ctx,
        flat_text,
        eot,
        tail_inclusive=True,
        # dbe-gate4-metadata-independence: the capture path emits its OWN
        # per-row span record from its own internal tokenization state —
        # the store index persists THAT record, never a wrapper-side
        # reconstruction (additive default-off kwarg; shared-module edit
        # recorded in the round marker).
        return_boundaries=True,
    )
    assert "va_tail_incl" in states, "tail_inclusive=True must persist the dual pooling"
    assert len(states["boundaries"]) == len(rows), (len(states["boundaries"]), len(rows))
    mism_ctx = [
        i
        for i, (b, cl) in enumerate(zip(states["boundaries"], ctx_lens, strict=True))
        if b["ctx_len"] != cl
    ]
    assert not mism_ctx, (
        f"capture-vs-generation ctx_len divergence at rows {mism_ctx[:5]} — "
        "wrapper bookkeeping error (row order / ctx threading)"
    )
    cap_lens = [int(n) for n in states["n_completion_tokens"]]
    mism = [i for i, (a, b) in enumerate(zip(site_comp_lens, cap_lens, strict=True)) if a != b]
    assert not mism, (
        f"site-vs-capture completion-token divergence at rows {mism[:5]} "
        f"(site={[site_comp_lens[i] for i in mism[:5]]}, "
        f"capture={[cap_lens[i] for i in mism[:5]]})"
    )
    for r, ctx_len, site_len, n_tok in zip(rows, ctx_lens, site_comp_lens, cap_lens, strict=True):
        r["n_completion_tokens"] = n_tok
        r["n_completion_tokens_gen"] = site_len  # generation-side record
        r["span_start"] = ctx_len
        r["span_end"] = ctx_len + site_len
        r["tail_end"] = ctx_len + site_len + len(eot)
        r["cap_hit"] = R.cap_hit(n_tok, max_new)
        r["cap_hit_basis"] = "retokenized_completion_len >= max_new_tokens"
    R._write_jsonl_atomic(jsonl, rows)
    return rows, states, ctx_lens


def _anchor_cell(
    cfg: DbeConfig, model, tok, bank: dict, cell: str, eot: list[int], pilot_state: dict
) -> None:
    ctxs = {cid: c for cid, c in bank["contexts"].items() if c["cell"] == cell}
    assert ctxs, f"no contexts for cell {cell}"
    jsonl = cfg.anchors_dir / f"anchors_dbe_w0_{cell}.jsonl"
    rows, states, ctx_lens = _gen_and_capture(
        cfg, model, tok, ctxs, cfg.max_new_tokens, eot, jsonl, pilot_state
    )
    n_cap_base = sum(1 for r in rows if r["cap_hit"])
    frac_base = n_cap_base / len(rows)
    regen = frac_base > CAP_HIT_REGEN_FRAC
    max_new_final = cfg.max_new_tokens
    if regen:
        # persist the capped run's text (rollout text is never discarded), then
        # re-generate the WHOLE cell at >= 2x the cap (#1332/#1426/#1481).
        capped = cfg.anchors_dir / f"anchors_dbe_w0_{cell}.capped{cfg.max_new_tokens}.jsonl"
        jsonl.replace(capped)
        logger.warning(
            "[anchors:%s] cap-hit %.3f > %.2f — re-generating cell at %d tokens",
            cell,
            frac_base,
            CAP_HIT_REGEN_FRAC,
            REGEN_MAX_NEW,
        )
        max_new_final = REGEN_MAX_NEW
        rows, states, ctx_lens = _gen_and_capture(
            cfg, model, tok, ctxs, REGEN_MAX_NEW, eot, jsonl, None
        )
    n_cap_final = sum(1 for r in rows if r["cap_hit"])
    R._save_pt_atomic(
        cfg.va_dir / f"va_dbe_w0_{cell}.pt",
        {
            "layers": list(cfg.layers),
            "index": [
                # CAPTURE-side span record (dbe-gate4-metadata-independence):
                # EMITTED BY capture_answer_states from its own internal
                # tokenization state (return_boundaries=True) — never a
                # wrapper-side reconstruction; gate 4 compares it EXACTLY
                # against the generation-side jsonl record and the gate's own
                # re-derivation.
                {"context_id": r["context_id"], "draw": r["draw"], **b}
                for r, b in zip(rows, states["boundaries"], strict=True)
            ],
            "va_span": states["va_span"],
            "va_tail_incl": states["va_tail_incl"],
            "poolings": ["span", "tail_incl"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "eot_ids": list(eot),
            "max_new_tokens": max_new_final,
            "repro": _repro(cfg, "anchors"),
        },
    )
    R._write_json_atomic(
        cfg.manifest_dir / f"anchors_dbe_{cell}_done.json",
        {
            "regime_fp": _cell_regime_fp(cfg, cell),
            "cell": cell,
            "n_contexts": len(ctxs),
            "draws": cfg.draws,
            "n_rows": len(rows),
            "n_cap_hit_base": n_cap_base,
            "cap_hit_frac_base": frac_base,
            "regen_applied": regen,
            "n_cap_hit_final": n_cap_final,
            "cap_hit_frac_final": n_cap_final / len(rows),
            "max_new_tokens_final": max_new_final,
            "n_empty": len(states["empty_rows"]),
            "repro": _repro(cfg, "anchors"),
        },
    )
    logger.info(
        "[anchors:%s] rows=%d cap_hit_final=%d regen=%s empty=%d",
        cell,
        len(rows),
        n_cap_final,
        regen,
        len(states["empty_rows"]),
    )


def _gate4_compare_records(gate_rec: dict, gen_row: dict, cap_ent: dict) -> None:
    """Gate-4 EXACT record equality across the three INDEPENDENTLY-derived
    span records (dbe-gate4-metadata-independence): the gate's own
    re-derivation vs the GENERATION-side jsonl record (site tokenization at
    the generation call site) vs the CAPTURE-side store record (emitted by
    ``capture_answer_states`` from its own internal state). A wrapper-side
    metadata error with untouched vectors FAILS here."""
    gen_rec = {
        "ctx_len": gen_row["span_start"],
        "n_completion_tokens": gen_row["n_completion_tokens_gen"],
        "span_start": gen_row["span_start"],
        "span_end": gen_row["span_end"],
        "tail_end": gen_row["tail_end"],
    }
    cap_rec = {k: cap_ent[k] for k in gate_rec}
    assert gate_rec == gen_rec == cap_rec, (
        gen_row.get("context_id"),
        {"gate": gate_rec, "generation": gen_rec, "capture": cap_rec},
        "gate-4 span-record mismatch across independent capture paths",
    )


@torch.no_grad()
def _gate4_parity(cfg: DbeConfig, model, tok, bank: dict, eot: list[int]) -> None:
    """Plan §7 gate 4 — M1 capture spot-parity through an INDEPENDENT
    single-row teacher-forced forward (never ``capture_answer_states``).

    (i) EXACTLY 3 sampled (context, draw) answer rows (fail LOUD when fewer
    than 3 are eligible — a thinner spot check silently weakens the gate):
    flattened cosine >= PARITY_COS_MIN for BOTH va_span AND va_tail_incl, plus
    EXACT record equality of the gate's OWN re-derived span record against
    BOTH independently-emitted records — the GENERATION-side jsonl record and
    the CAPTURE-side store index record (dbe-gate4-metadata-independence) —
    and the tail-slice token ids.
    (ii) 3 sampled contexts re-rendered + re-captured: v_ce AND v_pe cosine >=
    PARITY_COS_MIN each; prefix_end re-asserted against the B1 store.
    Any miss: parity_gate_report.json + rc=RC_PARITY_GATE (stop-and-diagnose)."""
    rng = random.Random(cfg.seed_base + 4)
    checks: list[dict] = []

    # (i) answer rows — enumerate candidates from the cheap jsonl only
    candidates: list[tuple[str, int, dict]] = []
    for cell in bank["kept_types"]:
        for idx, r in enumerate(_read_jsonl(cfg.anchors_dir / f"anchors_dbe_w0_{cell}.jsonl")):
            if r["n_completion_tokens"] > 0:
                candidates.append((cell, idx, r))
    assert len(candidates) >= 3, (
        f"gate 4 requires exactly 3 eligible answer rows; only {len(candidates)} "
        "non-empty rows exist — anchor generation is too degenerate to certify"
    )
    for cell, idx, r in rng.sample(candidates, 3):
        store = torch.load(
            cfg.va_dir / f"va_dbe_w0_{cell}.pt", map_location="cpu", weights_only=False
        )
        ent = store["index"][idx]
        assert (ent["context_id"], ent["draw"]) == (r["context_id"], r["draw"]), (cell, idx)
        ctx = bank["contexts"][r["context_id"]]
        ids = B2162.context_token_ids_2162(tok, ctx)
        comp_ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        assert list(store["eot_ids"]) == list(eot), (store["eot_ids"], eot)
        row_ids = ids + comp_ids + list(eot)
        s0, s1 = len(ids), len(ids) + len(comp_ids)
        t1 = s1 + len(eot)
        assert row_ids[s1:t1] == list(eot)
        # EXACT record comparison: the gate's own re-derived record vs the two
        # independently-emitted records (generation jsonl / capture store).
        gate_rec = {
            "ctx_len": len(ids),
            "n_completion_tokens": len(comp_ids),
            "span_start": s0,
            "span_end": s1,
            "tail_end": t1,
        }
        _gate4_compare_records(gate_rec, r, ent)
        t, mask = R._right_pad([row_ids], tok.pad_token_id, cfg.device)
        cap = extract_layer_activations(model, t, cfg.layers, attention_mask=mask)
        span = torch.stack([cap[lay][0, s0:s1].float().mean(0) for lay in cfg.layers]).cpu()
        tail = torch.stack([cap[lay][0, s0:t1].float().mean(0) for lay in cfg.layers]).cpu()
        checks.append(
            {
                "kind": "answer_row",
                "cell": cell,
                "context_id": r["context_id"],
                "draw": r["draw"],
                "cos_span": _flat_cos(span, store["va_span"][idx]),
                "cos_tail_incl": _flat_cos(tail, store["va_tail_incl"][idx]),
                "span_bounds": [s0, s1, t1],
            }
        )

    # (ii) contexts — re-render + re-capture vs the B1 vc store
    vc = torch.load(cfg.vc_dir / "vc_bank_dbe.pt", map_location="cpu", weights_only=False)
    cids = rng.sample(sorted(bank["contexts"]), min(3, len(bank["contexts"])))
    for cid in cids:
        ids = B2162.context_token_ids_2162(tok, bank["contexts"][cid])
        pe = B2162.prefix_end_index_multi(tok, ids)
        rec = vc["per_context"][cid]
        assert len(ids) == rec["ctx_len"] and pe == rec["prefix_end"], (
            cid,
            (len(ids), pe),
            (rec["ctx_len"], rec["prefix_end"]),
            "B1 prefix_end equality re-assert failed",
        )
        t, mask = R._right_pad([ids], tok.pad_token_id, cfg.device)
        cap = extract_layer_activations(model, t, cfg.layers, attention_mask=mask)
        v_ce = torch.stack([cap[lay][0, len(ids) - 1] for lay in cfg.layers]).float().cpu()
        v_pe = torch.stack([cap[lay][0, pe - 1] for lay in cfg.layers]).float().cpu()
        checks.append(
            {
                "kind": "context",
                "context_id": cid,
                "cos_ce": _flat_cos(v_ce, rec["v_ce"]),
                "cos_pe": _flat_cos(v_pe, rec["v_pe"]),
            }
        )

    cos_keys = ("cos_span", "cos_tail_incl", "cos_ce", "cos_pe")
    failures = [c for c in checks if any(c.get(k, 1.0) < PARITY_COS_MIN for k in cos_keys)]
    report = {
        "phase": "anchors-parity",
        "cos_min": PARITY_COS_MIN,
        "checks": checks,
        "n_failures": len(failures),
        "verdict": "PASS" if not failures else "FAIL",
        "regime_fp": _b2_regime_fp(cfg),
        "repro": _repro(cfg, "anchors-parity"),
    }
    R._write_json_atomic(cfg.manifest_dir / "parity_gate_report.json", report)
    logger.info(
        "[gate4] %d checks, %d failures — %s", len(checks), len(failures), report["verdict"]
    )
    if failures:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(RC_PARITY_GATE)


def _upload_b2(cfg: DbeConfig) -> None:
    """B2-END upload (#825 ordering) — verified BEFORE C; raises on failure."""
    # dbe-hf-egress-content-staleness: the tensor/text families are per-cell
    # fp-guarded (content changes only with a filename-stable regime change,
    # which the invalidation + fp machinery prevents) — resume_skip=True
    # keeps interrupted-walk resume cheap; the MANIFESTS family (gate
    # reports, cell done-manifests) is MUTABLE across forced reruns at the
    # same size, so it always re-pushes fresh bytes (resume_skip=False).
    families = (
        (cfg.va_dir, "analysis_tensors/va_dbe", True),
        (cfg.vc_dir, "analysis_tensors/vc_bank_dbe", True),
        (cfg.anchors_dir, "raw_completions/anchors", True),
        (cfg.manifest_dir, "analysis_tensors/manifests", False),
    )
    results: dict[str, dict] = {}
    for local, sub, skip_ok in families:
        res = upload_dir_sharded(
            local,
            HF_DATA_WRITE_REPO,
            f"{cfg.hf_prefix}/{sub}",
            proactive_overflow=True,
            verify=True,
            delete_local=False,  # C still reads the local stores
            resume_skip=skip_ok,
        )
        results[sub] = {
            "repo_id": res.repo_id,
            "overflow_repo": res.overflow_repo,
            "uploaded": sorted(res.uploaded),
            "rerouted": sorted(res.rerouted),
            "skipped_existing": sorted(res.skipped_existing),
        }
        logger.info(
            "[upload:B2] %s: %d uploaded / %d rerouted / %d skipped",
            sub,
            len(res.uploaded),
            len(res.rerouted),
            len(res.skipped_existing),
        )
    R._write_json_atomic(
        cfg.manifest_dir / "va_dbe_uploaded.json",
        {
            "hf_prefix": cfg.hf_prefix,
            # dbe-phase-c-sentinel-regime: C deserializes this record and
            # matches the FULL fingerprint (bank/values sha, cells, draws,
            # model, smoke/production all ride _b2_regime_fp) + hf_prefix.
            "regime_fp": _b2_regime_fp(cfg),
            "smoke": cfg.smoke,
            "tiny": cfg.tiny,
            "families": results,
            "repro": _repro(cfg, "anchors-upload"),
        },
    )


def phase_anchors(cfg: DbeConfig) -> int:
    logger.info("[phase=B2]")
    if cfg.force:
        # dbe-force-stale-completion: quarantine the phase's completion
        # records BEFORE any work — per-cell manifests stay honored.
        _invalidate_phase_records(cfg, "B2")
    bank = _load_bank_dbe(cfg)
    b2_fp = _b2_regime_fp(cfg)
    cells_todo = [c for c in bank["kept_types"] if not _cell_complete(cfg, c)]
    # resume-accepts-failed-gate-report: a persisted parity report licenses
    # the all-complete skip ONLY with verdict == "PASS" — an fp-matched
    # FAILED report must never certify a failed capture-integrity check.
    parity_ok = _report_regime_ok(
        cfg.manifest_dir / "parity_gate_report.json", b2_fp, require_pass_verdict=True
    )
    upload_ok = _report_regime_ok(cfg.manifest_dir / "va_dbe_uploaded.json", b2_fp)
    # dbe-phase-entry-idempotency: the ALL-COMPLETE determination runs BEFORE
    # the model load — a fully-complete regime never pays the load.
    if not cfg.force and not cells_todo and parity_ok and upload_ok:
        logger.info(
            "[anchors] resume: all %d cells + gate-4 + upload complete for this "
            "regime — skip (model never loaded)",
            len(bank["kept_types"]),
        )
        return RC_OK
    assert_out_root_headroom(cfg.out_root, HEADROOM_B2_GB, phase="B2-anchors")
    model, tok = R.load_model_and_tokenizer(cfg)
    eot = R.eot_tail_ids(tok)
    total_rollouts = len(bank["contexts"]) * cfg.draws
    # smoke-pilot-cross-regime-reuse: only a regime-matched, non-demoted (in
    # production) pilot report counts as done — bare file existence never
    # does. When every cell is already complete no generation runs, so the
    # gate has nothing to measure (its purpose is moot on that branch).
    pilot_state = {"done": _pilot_report_ok(cfg) or not cells_todo, "total": total_rollouts}
    for cell in bank["kept_types"]:
        if _cell_complete(cfg, cell):
            logger.info("[anchors:%s] resume: cell complete for this regime — skip", cell)
            continue
        _anchor_cell(cfg, model, tok, bank, cell, eot, pilot_state)
    _gate4_parity(cfg, model, tok, bank, eot)
    _upload_b2(cfg)
    return RC_OK


# ── phase C: analysis subprocess ──────────────────────────────────────


def _c_regime_fp(cfg: DbeConfig) -> str:
    """Phase C/D output-identity fingerprint: the B2 fp plus the null-battery
    size (the only C-only knob that changes registered outputs)."""
    return json.dumps({"b2_regime_fp": _b2_regime_fp(cfg), "null_b": cfg.null_b}, sort_keys=True)


def phase_analysis(cfg: DbeConfig) -> int:
    logger.info("[phase=C]")
    if cfg.force:
        # dbe-force-stale-completion: quarantine the stale completion record
        # BEFORE any work, so a crashed forced rerun re-enters next time.
        _invalidate_phase_records(cfg, "C")
    done = cfg.manifest_dir / "analysis_done.json"
    # Exact C-produced outputs (never any-JSON — the pre-existing committed
    # datagen_manifest.json alone must not satisfy the eval half) + the null
    # matrices + the prediction/target tensors D uploads.
    if (
        not cfg.force
        and _report_regime_ok(done, _c_regime_fp(cfg))
        and all((cfg.eval_dir / n).is_file() for n in REQUIRED_C_EVAL_JSONS)
        and cfg.null_dir.exists()
        and any(cfg.null_dir.iterdir())
        and cfg.predictions_dir.exists()
        and any(cfg.predictions_dir.iterdir())
    ):
        logger.info("[analysis] resume: analysis complete for this regime — skip (--force reruns)")
        return RC_OK
    ana = _SCRIPTS_DIR / "issue2215_dbe_analysis.py"
    if not ana.exists():
        raise RuntimeError(
            f"{ana} missing — unit 3 of the pre-split build delivers it; pull the "
            "issue-2215 branch tip before running --phase C"
        )
    uploaded = cfg.manifest_dir / "va_dbe_uploaded.json"
    if not uploaded.exists():
        raise RuntimeError(
            f"{uploaded} missing — B2 upload verification must PASS before C (plan §4.3)"
        )
    # dbe-phase-c-sentinel-regime: deserialize the B2 upload record and match
    # the FULL regime fingerprint + hf_prefix — never bare existence (a smoke
    # B2's record must not license a production C over mixed-regime stores).
    up_rec = json.loads(uploaded.read_text())
    if up_rec.get("regime_fp") != _b2_regime_fp(cfg) or up_rec.get("hf_prefix") != cfg.hf_prefix:
        raise RuntimeError(
            f"{uploaded} regime mismatch — B2 record (hf_prefix="
            f"{up_rec.get('hf_prefix')!r}, smoke={up_rec.get('smoke')}, "
            f"tiny={up_rec.get('tiny')}) was written under a different regime than "
            f"this C invocation (hf_prefix={cfg.hf_prefix!r}, smoke={cfg.smoke}, "
            f"tiny={cfg.tiny}); re-run --phase B2 under THIS regime first"
        )
    bank = cfg.vc_dir / "bank_dbe.json"
    # M2 assert (i) re-assert at C entry: every pe-aggregate pair differs in
    # >=1 prefix token (from the persisted realized-eligibility flags).
    bank_obj = json.loads(bank.read_text())
    for p in bank_obj["pairs"]:
        if p["cell"] in bank_obj.get("pe_aggregate_cells", []):
            assert p["pe_realized_eligible"], (p["pair_id"], "pe-aggregate pair with equal prefix")
    # dbe-manifest-realized-pe-map: the B1-finalized datagen manifest must
    # agree EXACTLY with bank_dbe.json before any analysis consumes either.
    final_path = cfg.vc_dir / "datagen_manifest_final.json"
    if not final_path.exists():
        raise RuntimeError(f"{final_path} missing — run --phase B1 first (B1' finalization)")
    _assert_realized_pe_manifest(json.loads(final_path.read_text()), bank_obj)
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.null_dir.mkdir(parents=True, exist_ok=True)
    # C' arg convention (unit-2-defined; unit 3 conforms — see module docstring).
    cmd = [
        sys.executable,
        str(ana),
        "--bank",
        str(bank),
        "--vc",
        str(cfg.vc_dir / "vc_bank_dbe.pt"),
        "--va-dir",
        str(cfg.va_dir),
        "--staged",
        str(cfg.staged_dir),
        "--null-out",
        str(cfg.null_dir),
        "--out-dir",
        str(cfg.eval_dir),
        "--figures-dir",
        str(cfg.figures_dir),
    ]
    if cfg.null_b is not None:
        cmd += ["--null-b", str(cfg.null_b)]
    if cfg.smoke:
        cmd.append("--smoke")
    if cfg.tiny:
        cmd.append("--tiny")
    logger.info("[analysis] exec: %s", " ".join(cmd[2:]))
    proc = subprocess.run(cmd, env={**os.environ}, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"analysis subprocess exited rc={proc.returncode}")
    # land the finalized datagen manifest beside the eval JSONs so the D git
    # leg commits it with the round's results (plan D' durable-artifact set).
    shutil.copy2(final_path, cfg.eval_dir / final_path.name)
    R._write_json_atomic(
        done,
        {
            "regime_fp": _c_regime_fp(cfg),
            "n_eval_jsons": len(list(cfg.eval_dir.glob("*.json"))),
            "n_figures": len(list(cfg.figures_dir.glob("dbe_*"))),
            "repro": _repro(cfg, "analysis"),
        },
    )
    return RC_OK


# ── phase D: finalize ─────────────────────────────────────────────────

BRANCH = f"issue-{ISSUE}"


def _sentinel_path(cfg: DbeConfig) -> Path:
    return cfg.log_dir / (SENTINEL_NAME_SMOKE if (cfg.smoke or cfg.tiny) else SENTINEL_NAME)


def _sentinel_present(path: Path) -> bool:
    """Bare presence check tolerating the poller's drain rename — bare name
    first, then ``.processed`` (pod-side-reporting read-back clause; used for
    COMPLETION checks only, never cross-relaunch resume state). The D resume
    guard uses the REGIME-VALIDATED ``_sentinel_regime_ok`` instead."""
    return path.exists() or path.with_name(path.name + ".processed").exists()


def _sentinel_regime_ok(path: Path, regime: str) -> bool:
    """Regime-VALIDATED sentinel completion check
    (dbe-d-stale-sentinel-completion): the payload's ``note.regime_fp`` must
    match the CURRENT run — a stale prior-round sentinel (bare or
    poller-drained ``.processed``) beside a fresh ``upload_done.json`` never
    licenses the D skip. Presence alone is never completion evidence."""
    for cand in (path, path.with_name(path.name + ".processed")):
        if not cand.exists():
            continue
        try:
            body = json.loads(cand.read_text())
        except json.JSONDecodeError:
            return False
        note = body.get("note")
        return isinstance(note, dict) and note.get("regime_fp") == regime
    return False


def _finalize_complete(cfg: DbeConfig) -> bool:
    """D resume predicate: a regime-matched ``upload_done.json`` (published
    ONLY after every upload leg verifies) AND a regime-matched results
    sentinel. Both halves regime-keyed; neither alone suffices."""
    regime = _c_regime_fp(cfg)
    return _report_regime_ok(cfg.manifest_dir / "upload_done.json", regime) and _sentinel_regime_ok(
        _sentinel_path(cfg), regime
    )


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Repo-scoped git call with explicit env passthrough; fail-loud rc check."""
    proc = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args],
        env={**os.environ},
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} rc={proc.returncode}: {proc.stderr.strip()[:2000]}"
        )
    return proc


def _git_paths_z(args: list[str]) -> list[str]:
    """NUL-delimited (`-z`) git path listing — whitespace/quoting-safe parsing
    (newline-mode git C-quotes exotic paths; a bare ``.split()`` shreds
    space-bearing names). ``-z`` is inserted directly after the subcommand —
    appended at the end it would land AFTER ``--`` and parse as a pathspec."""
    out = _git([args[0], "-z", *args[1:]]).stdout
    return [p for p in out.split("\0") if p]


def _run_figure_suite(cfg: DbeConfig) -> None:
    """Plan D': D invokes the CANONICAL figure suite (idempotent re-render) so
    the landed figures always come from the committed plotter, never a stale
    render (dbe-primary-artifact-egress)."""
    figs = _SCRIPTS_DIR / "issue2215_dbe_figures.py"
    if not figs.exists():
        raise RuntimeError(f"{figs} missing — pull the issue-2215 branch tip")
    cmd = [
        sys.executable,
        str(figs),
        "--in-dir",
        str(cfg.eval_dir),
        "--figures-dir",
        str(cfg.figures_dir),
    ]
    logger.info("[finalize] figure suite exec: %s", " ".join(cmd[2:]))
    proc = subprocess.run(cmd, env={**os.environ}, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"figure suite exited rc={proc.returncode}")


def _land_results_git(cfg: DbeConfig) -> dict:
    """Plan D' primary egress leg 1 (dbe-primary-artifact-egress): commit the
    round's eval JSONs + figures BY EXPLICIT PATH on branch issue-2215 and
    push, per the pod-side result-push verification contract (#1205 push
    verify / #1880 bounded fetch+rebase / #1325 per-FILE remote presence /
    #1482 named expected-path set, empty set = FAIL / #958 staged-index
    verification).

    Smoke/tiny: SKIPPED with a logged reason — the smoke twins write under
    ``<out_root>/smoke_eval`` / ``smoke_figures``, outside the repo tree by
    design (smoke outputs never overwrite committed paths); enumerated as a
    smoke blind spot in the round marker."""
    if cfg.smoke or cfg.tiny:
        logger.info(
            "[finalize] git landing SKIPPED under smoke/tiny — smoke eval/figure "
            "twins live under out_root, never committed repo paths"
        )
        return {"mode": "skipped-smoke"}
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"results git landing requires branch {BRANCH}; checkout is {branch!r}")
    paths = sorted(
        str(p.relative_to(_REPO_ROOT))
        for pat_dir, pat in ((cfg.eval_dir, "*.json"), (cfg.figures_dir, "dbe_*"))
        for p in pat_dir.glob(pat)
        if p.is_file()
    )
    # Named expected-path set printed BEFORE any verify (#1482): an empty set
    # on a declared-outputs round is a FAIL, never a quiet no-op.
    logger.info("[finalize] git landing expected paths (%d): %s", len(paths), " ".join(paths))
    if not paths:
        raise RuntimeError("git landing: empty expected-path set — C produced no eval/figure files")
    # Staged-index verification (#958 family): an EXPLICIT-path `git add`
    # ERRORS rc=1 on gitignored files (the silent skip is directory-add-only),
    # so detect the ignored-untracked subset FIRST, plain-add the rest,
    # force-add the convention-committed hits, then re-check empty.
    ignored = _git_paths_z(
        ["ls-files", "--others", "--ignored", "--exclude-standard", "--", *paths]
    )
    plain = [p for p in paths if p not in set(ignored)]
    if plain:
        _git(["add", "--", *plain])
    if ignored:
        logger.info("[finalize] git landing: force-adding %d gitignored files", len(ignored))
        _git(["add", "-f", "--", *ignored])
    left = _git_paths_z(["ls-files", "--others", "--ignored", "--exclude-standard", "--", *paths])
    assert not left, f"staged-index verification failed; still ignored: {left[:10]}"
    staged = _git_paths_z(["diff", "--cached", "--name-only", "--", *paths])
    if staged:
        _git(
            [
                "commit",
                "-m",
                f"task #{ISSUE}: {ROUND} eval results + figures (phase D)",
                "--",
                *paths,
            ]
        )
    else:
        logger.info("[finalize] git landing: content already committed — nothing new to commit")
    # Bare push with rc checked (never piped); on rejection fetch+rebase,
    # bounded 2 attempts, abort + fail loud on conflict (#1880).
    for attempt in (1, 2):
        push = _git(["push", "origin", BRANCH], check=False)
        if push.returncode == 0:
            break
        logger.warning(
            "[finalize] push rejected (attempt %d): %s", attempt, push.stderr.strip()[:500]
        )
        if attempt == 2:
            raise RuntimeError("git landing: push rejected after 2 fetch+rebase attempts")
        _git(["fetch", "origin", BRANCH])
        reb = _git(["rebase", f"origin/{BRANCH}"], check=False)
        if reb.returncode != 0:
            _git(["rebase", "--abort"], check=False)
            raise RuntimeError(f"git landing: rebase conflict: {reb.stderr.strip()[:1000]}")
    ahead = _git(["rev-list", "--count", f"origin/{BRANCH}..HEAD"]).stdout.strip()
    assert ahead == "0", f"push verification failed: {ahead} unpushed commits remain (#1205)"
    # Per-FILE remote presence assert (#1325 — per file, never a directory).
    missing = [
        p
        for p in paths
        if not _git(["ls-tree", "-r", f"origin/{BRANCH}", "--name-only", "--", p]).stdout.strip()
    ]
    assert not missing, f"remote presence assert failed for {len(missing)} files: {missing[:10]}"
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    logger.info(
        "[finalize] git landing verified: %d files on origin/%s @ %s", len(paths), BRANCH, head
    )
    return {"mode": "committed", "branch": BRANCH, "head": head, "n_files": len(paths)}


def _upload_results_hf(cfg: DbeConfig) -> dict:
    """Plan D' egress leg 2 (belt-and-suspenders): eval JSONs + the round's
    dbe_* figures ALSO land on the HF data repo beside the tensor families.
    Figures are staged into a round-scoped dir first — figures_dir is shared
    with sibling rounds' figures in production."""
    fig_stage = cfg.out_root / "results_egress" / "figures"
    if fig_stage.exists():
        shutil.rmtree(fig_stage)
    fig_stage.mkdir(parents=True, exist_ok=True)
    for p in sorted(cfg.figures_dir.glob("dbe_*")):
        if p.is_file():
            shutil.copy2(p, fig_stage / p.name)
    results: dict[str, dict] = {}
    for local, sub in ((cfg.eval_dir, "eval_results"), (fig_stage, "figures")):
        res = upload_dir_sharded(
            local,
            HF_DATA_WRITE_REPO,
            f"{cfg.hf_prefix}/{sub}",
            proactive_overflow=True,
            verify=True,
            delete_local=False,
            # dbe-hf-egress-content-staleness: eval JSONs + figures are
            # MUTABLE at stable filenames — a same-size stale remote byte
            # must never survive a corrected rerun; always push fresh bytes.
            resume_skip=False,
        )
        results[sub] = {
            "repo_id": res.repo_id,
            "overflow_repo": res.overflow_repo,
            "uploaded": sorted(res.uploaded),
            "rerouted": sorted(res.rerouted),
            "skipped_existing": sorted(res.skipped_existing),
        }
        logger.info(
            "[upload:D] %s: %d uploaded / %d rerouted / %d skipped",
            sub,
            len(res.uploaded),
            len(res.rerouted),
            len(res.skipped_existing),
        )
    return results


def _required_result_paths(cfg: DbeConfig) -> list[Path]:
    """The plan §6.5 REGISTERED deliverable set + the canonical hero/joint
    figure stems. Smoke/tiny drop ONLY the committed ``datagen_manifest.json``
    (the twin roots never stage it — production-only requirement, enumerated
    smoke blind spot); every other registered artifact is required in every
    regime."""
    evals = [cfg.eval_dir / n for n in REQUIRED_EVAL_JSONS]
    if cfg.smoke or cfg.tiny:
        evals = [p for p in evals if p.name != "datagen_manifest.json"]
    return evals + [cfg.figures_dir / n for n in REQUIRED_FIGURE_PNGS]


def _assert_required_results(cfg: DbeConfig) -> None:
    """Plan §6.5 registered-deliverable assert (dbe-primary-artifact-egress):
    the EXACT registered eval JSONs + the canonical figure stems must exist
    BEFORE any git/HF egress leg — D must never commit, upload, or signal
    success over a partial result set."""
    missing = [str(p) for p in _required_result_paths(cfg) if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"registered deliverables missing before egress (plan §6.5): {missing} — "
            "run --phase C (and check the canonical figure suite) first"
        )


def _sentinel_payload(cfg: DbeConfig) -> dict:
    bank = _load_bank_dbe(cfg)
    rows_total = cap_final = 0
    regen_cells: list[str] = []
    for m in sorted(cfg.manifest_dir.glob("anchors_dbe_*_done.json")):
        rec = json.loads(m.read_text())
        rows_total += rec["n_rows"]
        cap_final += rec["n_cap_hit_final"]
        if rec["regen_applied"]:
            regen_cells.append(rec["cell"])

    def _opt(path: Path) -> dict:
        return json.loads(path.read_text()) if path.exists() else {}

    pilot = _opt(cfg.manifest_dir / "pilot_gate_report.json")
    parity = _opt(cfg.manifest_dir / "parity_gate_report.json")
    uploaded = _opt(cfg.manifest_dir / "va_dbe_uploaded.json")
    return {
        "issue": ISSUE,
        "round": ROUND,
        "hf_prefix": cfg.hf_prefix,
        "eval_numbers": {
            "n_cells": len(bank["kept_types"]),
            "n_contexts": len(bank["contexts"]),
            "n_pairs": len(bank["pairs"]),
            "pe_aggregate_cells": bank.get("pe_aggregate_cells", []),
            "anchor_rows": rows_total,
            "cap_hit_rows_final": cap_final,
            "cap_hit_frac_final": (cap_final / rows_total) if rows_total else 0.0,
            "regen_cells": regen_cells,
            "pilot_gate": {k: pilot.get(k) for k in ("verdict", "projected_wall_h")},
            "parity_gate": {k: parity.get(k) for k in ("verdict", "n_failures")},
        },
        "uploaded_families": {
            k: len(v.get("uploaded", []) + v.get("rerouted", []))
            for k, v in uploaded.get("families", {}).items()
        },
        "eval_jsons": [p.name for p in sorted(cfg.eval_dir.glob("*.json"))],
        "figures": [p.name for p in sorted(cfg.figures_dir.glob("dbe_*"))],
        "caveats": [
            "dual pooling persisted per anchor row (va_span AND va_tail_incl; "
            "tail_inclusive=True explicit — unlike the parent's span-only store)",
            "cap-hit basis: retokenized_completion_len >= max_new_tokens "
            "(generate_batch returns decoded text only)",
            "refusal manipulation-check judge lives in the analysis driver "
            "(scripts/issue2215_dbe_analysis.py), advisory per plan §7 gate 3",
        ],
        "repro": _repro(cfg, "finalize"),
    }


def phase_finalize(cfg: DbeConfig) -> int:
    logger.info("[phase=D]")
    if cfg.force:
        # dbe-force-stale-completion: quarantine upload_done + the results
        # sentinel (bare AND .processed) BEFORE any work — a crashed forced
        # rerun must leave NO eligible completion state behind.
        _invalidate_phase_records(cfg, "D")
    sentinel = _sentinel_path(cfg)
    upload_done = cfg.manifest_dir / "upload_done.json"
    # phase-d-no-entry-skip-sentinel / dbe-phase-entry-idempotency /
    # dbe-d-stale-sentinel-completion: a COMPLETE D re-entry (regime-matched
    # upload_done AND a regime-matched results sentinel, bare or
    # poller-drained .processed) never re-uploads and never re-writes the
    # sentinel; --force reruns the phase.
    if not cfg.force and _finalize_complete(cfg):
        logger.info(
            "[finalize] resume: finalize complete for this regime — skip "
            "(no re-upload; sentinel preserved)"
        )
        logger.info("[phase=done]")
        return RC_OK
    if not (cfg.null_dir.exists() and any(cfg.null_dir.iterdir())):
        raise RuntimeError(f"{cfg.null_dir} empty — run --phase C first (null matrices missing)")
    # Unit-3 flag fix: the C' driver persists per-arm prediction/target
    # tensors (predictions_L*.pt) so every registered row is recomputable
    # post-hoc (plan §4.3 C') — they MUST land on HF before teardown.
    pred_dir = cfg.predictions_dir
    if not (pred_dir.exists() and any(pred_dir.iterdir())):
        raise RuntimeError(
            f"{pred_dir} empty — the C' analysis driver persists prediction/target "
            "tensors there (plan §4.3 C'); run --phase C first"
        )
    # plan D' primary-artifact egress (dbe-primary-artifact-egress): canonical
    # figure re-render, then the REGISTERED-deliverable assert (plan §6.5
    # exact set) BEFORE ANY egress leg — git or HF; only then the uploads +
    # both results-egress legs.
    _run_figure_suite(cfg)
    _assert_required_results(cfg)
    res = upload_dir_sharded(
        cfg.null_dir,
        HF_DATA_WRITE_REPO,
        f"{cfg.hf_prefix}/analysis_tensors/null_matrices",
        proactive_overflow=True,
        verify=True,
        delete_local=False,
        resume_skip=False,  # forced C reruns mutate at stable names/sizes
    )
    logger.info(
        "[upload:D] null_matrices: %d uploaded / %d rerouted / %d skipped",
        len(res.uploaded),
        len(res.rerouted),
        len(res.skipped_existing),
    )
    res_p = upload_dir_sharded(
        pred_dir,
        HF_DATA_WRITE_REPO,
        f"{cfg.hf_prefix}/analysis_tensors/predictions",
        proactive_overflow=True,
        verify=True,
        delete_local=False,
        resume_skip=False,  # dbe-hf-egress-content-staleness
    )
    logger.info(
        "[upload:D] predictions: %d uploaded / %d rerouted / %d skipped",
        len(res_p.uploaded),
        len(res_p.rerouted),
        len(res_p.skipped_existing),
    )
    git_rec = _land_results_git(cfg)
    hf_rec = _upload_results_hf(cfg)
    # Manifests family upload FIRST (gate reports + per-cell manifests):
    # upload_done — the record the resume guard reads — is published ONLY
    # after EVERY upload leg verifies (dbe-d-stale-sentinel-completion); a
    # manifests-upload failure leaves no completion record to skip on.
    upload_dir_sharded(
        cfg.manifest_dir,
        HF_DATA_WRITE_REPO,
        f"{cfg.hf_prefix}/analysis_tensors/manifests",
        proactive_overflow=True,
        verify=True,
        delete_local=False,
        resume_skip=False,  # gate reports mutate at stable names/sizes
    )
    payload = _sentinel_payload(cfg)
    payload["regime_fp"] = _c_regime_fp(cfg)
    payload["results_egress"] = {"git": git_rec, "hf_families": hf_rec}
    R._write_json_atomic(upload_done, payload)
    # Second manifests pass picks up upload_done itself (durable off-pod);
    # resume_skip=True — a pure delta pass over just-pushed fresh bytes.
    upload_dir_sharded(
        cfg.manifest_dir,
        HF_DATA_WRITE_REPO,
        f"{cfg.hf_prefix}/analysis_tensors/manifests",
        proactive_overflow=True,
        verify=True,
        delete_local=False,
    )
    body = {
        "sentinel_schema_version": 1,
        "kind": "epm:smoke-result" if (cfg.smoke or cfg.tiny) else "epm:results",
        "version": 1,
        "note": payload,
    }
    R._write_json_atomic(sentinel, body)
    logger.info("[finalize] sentinel written: %s", sentinel)
    logger.info("[phase=done]")
    return RC_OK


# ── CLI ───────────────────────────────────────────────────────────────


def _import_check() -> None:
    """Axis-1 import resolution + argparse-attribute completeness + call-arity
    bind pass (orchestrate.argcheck; code-style.md convention). Also EXECUTES
    every deferred import this driver reaches on its real paths."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    from huggingface_hub import HfApi  # noqa: F401
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from issue779_ffc_n1m_fits import apply_map  # the phase-A deferred import

    for fn in (
        apply_map,
        stage_hub_file,
        stage_hub_prefix,
        upload_dir_sharded,
        assert_out_root_headroom,
        extract_layer_activations,
        generate_batch,
        ANA.build_cell_views,
        ANA.PairTable.from_bank,
        R.capture_answer_states,
        R.load_model_and_tokenizer,
        R.eot_tail_ids,
        R.cap_hit,
        B2162.render_context_2162,
        B2162.context_token_ids_2162,
        B2162.prefix_end_index_multi,
        DBE.load_values,
        DBE.bank_manifest_dbe,
        DBE.expected_pe_eligibility,
    ):
        assert callable(fn), fn
    assert (_SCRIPTS_DIR / "verify_reused_artifact_keys.py").exists()
    print("[import-check] OK", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2215 discrimination-battery-expansion pod driver (plan v6 §4.3)",
        epilog=(
            "canonical smoke: --phase all --smoke  "
            "(== --cells user_role_identity,user_sentiment --draws 2 --null-b 100)"
        ),
    )
    ap.add_argument("--phase", choices=("all", *PHASES), default=None)
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/eps2215dbe"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument(
        "--values", type=Path, default=None, help="frozen values JSON (default: packaged file)"
    )
    ap.add_argument("--cells", default=None, help="csv cell subset (threads through EVERY phase)")
    ap.add_argument(
        "--draws", type=int, default=None, help=f"anchors/context (default {ANCHOR_DRAWS}; smoke 2)"
    )
    ap.add_argument("--null-b", type=int, default=None, help="null draws for C (smoke 100)")
    ap.add_argument("--smoke", action="store_true", help="tiny cell subset, same phases/uploads")
    ap.add_argument("--tiny", action="store_true", help="from-config tiny model on CPU (wiring)")
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--capture-batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=ANCHOR_MAX_NEW)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument(
        "--force",
        action="store_true",
        help="bypass PHASE-level entry-completion guards (B2 all-complete, C, D) AND "
        "quarantine that phase's completion records at entry "
        "(dbe-force-stale-completion); per-cell resume manifests stay honored",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> DbeConfig:
    cells: tuple[str, ...] | None
    if args.cells:
        cells = tuple(c.strip() for c in args.cells.split(",") if c.strip())
    elif args.smoke:
        cells = SMOKE_CELLS
    else:
        cells = None
    draws = args.draws if args.draws is not None else (2 if args.smoke else ANCHOR_DRAWS)
    null_b = args.null_b if args.null_b is not None else (100 if args.smoke else None)
    tiny = bool(args.tiny)
    n_layers = 4 if tiny else N_LAYERS
    return DbeConfig(
        phase=args.phase or "",
        out_root=args.out_root,
        log_dir=args.log_dir,
        values_path=args.values,
        smoke=bool(args.smoke),
        tiny=tiny,
        cells=cells,
        draws=draws,
        null_b=null_b,
        gen_batch=args.gen_batch,
        capture_batch=args.capture_batch,
        max_new_tokens=args.max_new_tokens,
        seed_base=args.seed_base,
        force=bool(args.force),
        hidden=64 if tiny else HIDDEN,
        n_layers=n_layers,
        layers=list(range(n_layers)),
        device="cuda" if (torch.cuda.is_available() and not tiny) else "cpu",
    )


PHASE_FNS = {
    "G-check": phase_gcheck,
    "A": phase_stage,
    "B1": phase_bank,
    "B2": phase_anchors,
    "C": phase_analysis,
    "D": phase_finalize,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return RC_OK
    assert args.phase, "--phase is required (or pass --import-check)"
    cfg = build_config(args)
    # Run-start model-revision resolution (dbe-resume-fingerprint-inputs):
    # every resume fingerprint pins the RESOLVED HF commit sha of the model
    # this run consumes, not the mutable `main` ref.
    cfg.model_revision = _resolve_model_revision(cfg)
    for d in (cfg.out_root, cfg.log_dir, cfg.manifest_dir, cfg.vc_dir, cfg.va_dir, cfg.anchors_dir):
        d.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[config] phase=%s out_root=%s smoke=%s tiny=%s cells=%s draws=%d device=%s prefix=%s "
        "model_rev=%s",
        args.phase,
        cfg.out_root,
        cfg.smoke,
        cfg.tiny,
        ",".join(cfg.cells) if cfg.cells else "all",
        cfg.draws,
        cfg.device,
        cfg.hf_prefix,
        cfg.model_revision,
    )
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    for name in phases:
        rc = PHASE_FNS[name](cfg)
        if rc != RC_OK:
            return rc
    return RC_OK


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
