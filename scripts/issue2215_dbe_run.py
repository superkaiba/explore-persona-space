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
* ``D`` — finalize: upload null/per-draw matrices to
  ``<prefix>/analysis_tensors/null_matrices/``, assert eval JSONs + figures
  landed, write ``upload_done.json`` + the results sentinel
  (``/workspace/logs/issue-2215-dbe-results.json``), then ``[phase=done]``.

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
import json
import logging
import math
import os
import random
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
SEED_BASE = 2215000

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
    def manifest_dir(self) -> Path:
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


def _bank_regime_fp(cfg: DbeConfig) -> str:
    """Machine-stable resume key — generating parameters only (#1336)."""
    return json.dumps(
        {
            "round": ROUND,
            "model_id": cfg.model_id,
            "tiny": cfg.tiny,
            "smoke": cfg.smoke,
            "cells": list(cfg.cells) if cfg.cells else "all",
            "n_layers": cfg.n_layers,
            "hidden": cfg.hidden,
        },
        sort_keys=True,
    )


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
    ):
        logger.info("[bank] resume: capture complete for this regime — skip")
        return RC_OK
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
    return json.dumps(
        {
            "round": ROUND,
            "cell": cell,
            "draws": cfg.draws,
            "seed_base": cfg.seed_base,
            "base_max_new_tokens": cfg.max_new_tokens,
            "model_id": cfg.model_id,
            "tiny": cfg.tiny,
            "smoke": cfg.smoke,
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
    states = R.capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot, tail_inclusive=True)
    assert "va_tail_incl" in states, "tail_inclusive=True must persist the dual pooling"
    for r, n_tok in zip(rows, states["n_completion_tokens"], strict=True):
        r["n_completion_tokens"] = int(n_tok)
        r["cap_hit"] = R.cap_hit(int(n_tok), max_new)
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
                {
                    "context_id": r["context_id"],
                    "draw": r["draw"],
                    "ctx_len": ctx_len,
                    "n_completion_tokens": r["n_completion_tokens"],
                }
                for r, ctx_len in zip(rows, ctx_lens, strict=True)
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


@torch.no_grad()
def _gate4_parity(cfg: DbeConfig, model, tok, bank: dict, eot: list[int]) -> None:
    """Plan §7 gate 4 — M1 capture spot-parity through an INDEPENDENT
    single-row teacher-forced forward (never ``capture_answer_states``).

    (i) 3 sampled (context, draw) answer rows: flattened cosine >=
    PARITY_COS_MIN for BOTH va_span AND va_tail_incl, plus EXACT equality of
    span boundaries (ctx_len / n_completion_tokens) and tail-slice token ids.
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
    for cell, idx, r in rng.sample(candidates, min(3, len(candidates))):
        store = torch.load(
            cfg.va_dir / f"va_dbe_w0_{cell}.pt", map_location="cpu", weights_only=False
        )
        ent = store["index"][idx]
        assert (ent["context_id"], ent["draw"]) == (r["context_id"], r["draw"]), (cell, idx)
        ctx = bank["contexts"][r["context_id"]]
        ids = B2162.context_token_ids_2162(tok, ctx)
        comp_ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        # EXACT span-boundary / tail-id metadata equality
        assert list(store["eot_ids"]) == list(eot), (store["eot_ids"], eot)
        assert len(ids) == ent["ctx_len"], (r["context_id"], len(ids), ent["ctx_len"])
        assert len(comp_ids) == r["n_completion_tokens"] == ent["n_completion_tokens"], (
            r["context_id"],
            len(comp_ids),
            r["n_completion_tokens"],
        )
        row_ids = ids + comp_ids + list(eot)
        s0, s1 = len(ids), len(ids) + len(comp_ids)
        t1 = s1 + len(eot)
        assert row_ids[s1:t1] == list(eot)
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
    families = (
        (cfg.va_dir, "analysis_tensors/va_dbe"),
        (cfg.vc_dir, "analysis_tensors/vc_bank_dbe"),
        (cfg.anchors_dir, "raw_completions/anchors"),
        (cfg.manifest_dir, "analysis_tensors/manifests"),
    )
    results: dict[str, dict] = {}
    for local, sub in families:
        res = upload_dir_sharded(
            local,
            HF_DATA_WRITE_REPO,
            f"{cfg.hf_prefix}/{sub}",
            proactive_overflow=True,
            verify=True,
            delete_local=False,  # C still reads the local stores
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
        {"hf_prefix": cfg.hf_prefix, "families": results, "repro": _repro(cfg, "anchors-upload")},
    )


def phase_anchors(cfg: DbeConfig) -> int:
    logger.info("[phase=B2]")
    assert_out_root_headroom(cfg.out_root, HEADROOM_B2_GB, phase="B2-anchors")
    bank = _load_bank_dbe(cfg)
    model, tok = R.load_model_and_tokenizer(cfg)
    eot = R.eot_tail_ids(tok)
    total_rollouts = len(bank["contexts"]) * cfg.draws
    pilot_state = {
        "done": (cfg.manifest_dir / "pilot_gate_report.json").exists(),
        "total": total_rollouts,
    }
    for cell in bank["kept_types"]:
        done = cfg.manifest_dir / f"anchors_dbe_{cell}_done.json"
        if (
            done.exists()
            and json.loads(done.read_text()).get("regime_fp") == _cell_regime_fp(cfg, cell)
            and (cfg.va_dir / f"va_dbe_w0_{cell}.pt").exists()
            and (cfg.anchors_dir / f"anchors_dbe_w0_{cell}.jsonl").exists()
        ):
            logger.info("[anchors:%s] resume: cell complete for this regime — skip", cell)
            continue
        _anchor_cell(cfg, model, tok, bank, cell, eot, pilot_state)
    _gate4_parity(cfg, model, tok, bank, eot)
    _upload_b2(cfg)
    return RC_OK


# ── phase C: analysis subprocess ──────────────────────────────────────


def phase_analysis(cfg: DbeConfig) -> int:
    logger.info("[phase=C]")
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
    bank = cfg.vc_dir / "bank_dbe.json"
    # M2 assert (i) re-assert at C entry: every pe-aggregate pair differs in
    # >=1 prefix token (from the persisted realized-eligibility flags).
    bank_obj = json.loads(bank.read_text())
    for p in bank_obj["pairs"]:
        if p["cell"] in bank_obj.get("pe_aggregate_cells", []):
            assert p["pe_realized_eligible"], (p["pair_id"], "pe-aggregate pair with equal prefix")
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
    return RC_OK


# ── phase D: finalize ─────────────────────────────────────────────────


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
    if not (cfg.null_dir.exists() and any(cfg.null_dir.iterdir())):
        raise RuntimeError(f"{cfg.null_dir} empty — run --phase C first (null matrices missing)")
    res = upload_dir_sharded(
        cfg.null_dir,
        HF_DATA_WRITE_REPO,
        f"{cfg.hf_prefix}/analysis_tensors/null_matrices",
        proactive_overflow=True,
        verify=True,
        delete_local=False,
    )
    logger.info(
        "[upload:D] null_matrices: %d uploaded / %d rerouted / %d skipped",
        len(res.uploaded),
        len(res.rerouted),
        len(res.skipped_existing),
    )
    # manifests family re-upload (idempotent resume-skip) so upload_done +
    # gate reports become durable off-pod too.
    evals = sorted(cfg.eval_dir.glob("*.json"))
    if not evals:
        raise RuntimeError(f"no eval JSONs under {cfg.eval_dir} — run --phase C first")
    figs = sorted(cfg.figures_dir.glob("dbe_*"))
    if not figs:
        raise RuntimeError(f"no dbe_* figures under {cfg.figures_dir} — run --phase C first")
    payload = _sentinel_payload(cfg)
    R._write_json_atomic(cfg.manifest_dir / "upload_done.json", payload)
    upload_dir_sharded(
        cfg.manifest_dir,
        HF_DATA_WRITE_REPO,
        f"{cfg.hf_prefix}/analysis_tensors/manifests",
        proactive_overflow=True,
        verify=True,
        delete_local=False,
    )
    sentinel = cfg.log_dir / (SENTINEL_NAME_SMOKE if (cfg.smoke or cfg.tiny) else SENTINEL_NAME)
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
    for d in (cfg.out_root, cfg.log_dir, cfg.manifest_dir, cfg.vc_dir, cfg.va_dir, cfg.anchors_dir):
        d.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[config] phase=%s out_root=%s smoke=%s tiny=%s cells=%s draws=%d device=%s prefix=%s",
        args.phase,
        cfg.out_root,
        cfg.smoke,
        cfg.tiny,
        ",".join(cfg.cells) if cfg.cells else "all",
        cfg.draws,
        cfg.device,
        cfg.hf_prefix,
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
