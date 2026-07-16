#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ², Δ, ×, ≥) in scientific docstrings + log messages.
"""Issue #928 follow-up `prefix-based-mapping-arms` (plan v7) — ONE driver.

Does the matched-length demotion — context+answer-prefix beats
context+truncated-CoT by 0.052 per-question — survive when the prompt-side
input is the PREFIX summary (everything before the user query) instead of the
full context? One teacher-forced capture pass over the SAME persisted rollouts
adds two prompt-side mean-pool summaries (``prefix_mean``, ``query_mean``);
fits JOIN the committed matched-length store so every cross-convention
contrast differs in exactly one thing: the prompt-side input (plan v7 §4.0).

Single code path, smoke = the SAME driver with ``--contexts 3`` (unification
default — every phase's cell list derives from the one ``--contexts`` subset).
Phases (linear; ``[phase=...]`` breadcrumbs feed the poller):

- **stage:** pinned-revision inputs — rollout JSONs (@ ``5c1e3c5c00…``), the
  matched-length summary store + the committed MLC decomps
  (@ ``30fb798f72dc…``), the committed MLC bootstrap JSON (local-first, HF
  fallback). Never ``snapshot_download`` on the ~1M-file repo (gotchas #833).
- **asserts (fail-loud, pre-GPU):** per context, recomputed
  ``rollout_content_digest`` == the MLC blob's ``rollout_digest`` (item-(j)
  pair coherence) AND the blob's ``mlc_floors`` pins match.
- **spans (CPU, tokenizer-only):** ``parse_rows`` verbatim + ``_mlc_parts``
  floors + ``prefix_query_spans`` via the ``prompt_parts_spec`` hook; the
  post-floor kept set is asserted EQUAL to the MLC blob's ``probe_indices``
  (expected 1,991 on the full grid — kill criterion; a prompt-side span drop
  breaks the equality and is named in the error).
- **capture (GPU):** 5 mean-pool vectors per (row, layer)
  (``PMA_SUMMARY_NAMES``); new flat store ``prefix_summaries/``
  (parent-identical blob schema, manifest in-folder).
- **parity (fail-loud, before any fit):** per (row, layer, part ∈
  {ctx, cot, ans}) cosine(recaptured mean, MLC store ``per_q`` mean) ≥ 0.999
  (the shared capture path bounds the new prefix/query parts — prefix ∪ query
  ⊂ ctx tokens). Plus a logged prefix per-context-constancy read (§12.9).
- **fit:** TWO targets — the PCA-48 answer-REMAINDER (matched-length parity;
  basis-coherence-asserted vs the committed ``decomp_*_mlc.pt`` identity
  ss_tot, rtol 1e-6, kill criterion) and the PCA-48 full-ANSWER (parent
  parity) — 6+3 input reps × layers × {LOCO-50, LOFO-7} × both regimes;
  batched ``GroupRidgeDesign`` machinery, serial-parity-gated; per-(regime,
  layer) checkpoint units.
- **nulls:** selection-symmetric per-draw × per-layer matrices for the 6
  registered arms (exploratory query-audit arms EXCLUDED — plan §6).
- **bootstrap:** 4 within-round reads + the read-5 convention-contrast family
  (cross-round, draw-aligned via the shared seed-42 resample matrix whose
  digest is ASSERTED equal to the committed ``3b77857b7c7e7042`` per regime on
  a comparable grid — kill criterion) + the degenerate-prefix-cell sensitivity
  re-read.
- **figures / upload / done:** pma_* set; one ``upload_folder`` commit per
  artifact kind with scoped verify; ONE ``epm:results`` sentinel at
  true end-of-workload.

Usage::

    # production (GCP capture-7b lane, plan §10):
    EPM_FIT_DEVICE=cuda uv run python scripts/issue928_prefix_mapping_arms.py \\
        --out eval_results/issue_928/prefix-based-mapping-arms

    # pod-side Phase-0 smoke (= the sweep at 3 contexts, scratch outputs):
    uv run python scripts/issue928_prefix_mapping_arms.py --contexts 3 \\
        --out /tmp/issue-928-pma-smoke/eval --no-upload

    # VM CPU partial smoke (network + tokenizer only; stops pre-GPU):
    uv run python scripts/issue928_prefix_mapping_arms.py --contexts 2 \\
        --stop-after spans --out /tmp/issue-928-pma-smoke/eval --no-upload
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))

import argparse
import ast
import functools
import hashlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Cross-script helper imports hoisted to module top so a missing symbol crashes
# at process start, never inside a smoke-skipped branch (gotchas.md #606).
from issue594_common import probes_hash  # noqa: E402
from issue594_extract_context_vectors import LayerCapture  # noqa: E402
from issue658_fit_predictors import _requested_device, _resolve_device  # noqa: E402
from issue928_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DECOMP_TENSORS_PREFIX,
    FIGURES_PREFIX,
    HF_DATA_REPO,
    HF_PREFIX_928,
    MLC_K_MIN,
    MLC_REM_MIN,
    PMA_SUMMARY_NAMES,
    SHUFFLE_NULL_PERMS,
    SHUFFLE_NULL_SEED,
    context_order_and_families,
    dump_json,
    load_json,
    load_probe_pool,
    prefix_query_spans,
    reproducibility_metadata,
    resolve_battery,
    upload_folder_scoped_verify,
    write_sentinel,
)
from issue928_extract_thinking_store import (  # noqa: E402
    build_capture_row,
    pack_batches,
    parse_rows,
    reduce_forward_batch,
    reusable_store_blob,
    rollout_content_digest,
)
from issue928_fit_decomposition import (  # noqa: E402
    Store,
    _atomic_torch_save,
    _pca_target,
    prepare_checkpoint_dir,
)
from issue928_matched_length_control import (  # noqa: E402
    MLC_RESULTS_PREFIX,
    MLC_STORE_HF_PREFIX,
    PARITY_COS_MIN,
    _mlc_parts,
    capture_parity_gate,
    stage_rollouts,
)
from issue928_mlp_indiv_control import (  # noqa: E402
    STORE_REVISION as ROLLOUTS_REVISION,
)
from issue928_mlp_indiv_control import (  # noqa: E402
    _hf_fetch_one,
    ss_tot_by_group,
)
from issue928_null_bootstrap import (  # noqa: E402
    GroupRidgeDesign,
    assert_group_ridge_matches_serial,
    bootstrap_skills,
    fit_predict_grouped,
    group_folds,
    grouped_null_skills_multi,
    grouped_skill,
    make_bootstrap_index_matrix,
    make_group_perm_matrix,
    stat_summary,
)

logger = logging.getLogger("issue928_pma")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── round constants (plan v7 §5/§10) ─────────────────────────────────────────

PMA_FOLLOWUP_LABEL = "prefix-based-mapping-arms"
# The matched-length round's upload revision — store blobs + committed decomps
# (Hub-verified in the plan session; plan §10).
MLC_ARTIFACTS_REVISION = "30fb798f72dc55306f62a4436fd57b3039ba768c"
PMA_STORE_HF_PREFIX = f"{HF_PREFIX_928}/analysis_tensors/store/prefix_summaries"
PMA_RESULTS_PREFIX = f"{HF_PREFIX_928}/fit_results/prefix_mapping_arms"
MLC_DECOMP_HF_PREFIX = f"{DECOMP_TENSORS_PREFIX}/matched_length_control"

# The committed seed-42 resample-matrix digest (both regimes; plan §6 — the
# draw-alignment contract for every cross-round paired contrast).
EXPECTED_RESAMPLE_DIGEST = "3b77857b7c7e7042"
# PCA-basis coherence bar vs the committed identity ss_tot (plan §4.3).
BASIS_COHERENCE_RTOL = 1e-6

COMBO = "mean/mean"  # this round is mean-pool only (inherited from v6)

# Near-degenerate prefix cells (plan §4.1/§8): kept, flagged in figures, and
# the headline gets a sensitivity re-read excluding them (not a new read).
DEGENERATE_PREFIX_CELLS = ("f6_default_template", "f6_helpful_asst")

# Input reps per arm (plan §4.3). REMAINDER-target arms fit the committed
# PCA-48 answer-remainder basis; ANSWER-target arms fit the parent-parity
# full-answer basis. prefix/query join from THIS round's store; every other
# rep is STORE-JOINED byte-identical from the matched-length store (§4.0).
PMA_REM_ARM_INPUTS: dict[str, tuple[str, ...]] = {
    "pma_pfx": ("prefix_mean",),
    "pma_pfx_cotK": ("prefix_mean", "cot_lastK_mean"),
    "pma_pfx_apfx": ("prefix_mean", "ansprefix_K_mean"),
    "pma_pfx_cotfull": ("prefix_mean", "cot_mean"),
    "pma_qry": ("query_mean",),
    "pma_pfx_qry": ("prefix_mean", "query_mean"),
}
PMA_ANS_ARM_INPUTS: dict[str, tuple[str, ...]] = {
    "pma_pfx_ans": ("prefix_mean",),
    "pma_ctx_ans": ("ctx_mean",),
    "pma_ident_ans": ("ans_mean",),
}
PMA_ARM_INPUTS = {**PMA_REM_ARM_INPUTS, **PMA_ANS_ARM_INPUTS}
PMA_REM_ARMS = tuple(PMA_REM_ARM_INPUTS)
PMA_ANS_ARMS = tuple(PMA_ANS_ARM_INPUTS)
PMA_REGISTERED_ARMS = (
    "pma_pfx",
    "pma_pfx_cotK",
    "pma_pfx_apfx",
    "pma_pfx_cotfull",
    "pma_pfx_ans",
    "pma_ctx_ans",
)
PMA_EXPLORATORY_ARMS = ("pma_qry", "pma_pfx_qry")
PMA_ANS_IDENT_ARM = "pma_ident_ans"

# Registered within-round paired reads (plan §6 reads 1-4; read 1 is PRIMARY).
PMA_REGISTERED_READS = (
    ("read1_primary_pfx_cotK_minus_pfx_apfx", "pma_pfx_cotK", "pma_pfx_apfx"),
    ("read2_pfx_cotK_minus_pfx", "pma_pfx_cotK", "pma_pfx"),
    ("read3_pfx_apfx_minus_pfx", "pma_pfx_apfx", "pma_pfx"),
    ("read4_pfx_cotfull_minus_pfx_cotK", "pma_pfx_cotfull", "pma_pfx_cotK"),
)
# Read-5 convention contrasts (plan §6): 5a-c cross-round (this round's pma
# arm at ITS frozen layer vs the committed mlc arm at the COMMITTED frozen
# layer, draw-aligned via the shared resample matrix), 5d within-round on the
# full-answer target at the parent's committed layers.
PMA_CONVENTION_READS = (
    ("read5a_conv_pfx_minus_ctx", "pma_pfx", "mlc_ctx"),
    ("read5b_conv_pfx_cotK_minus_ctx_cotK", "pma_pfx_cotK", "mlc_ctx_cotK"),
    ("read5c_conv_pfx_apfx_minus_ctx_apfx", "pma_pfx_apfx", "mlc_ctx_apfx"),
)
READ_5D = ("read5d_conv_pfx_ans_minus_ctx_ans", "pma_pfx_ans", "pma_ctx_ans")

# Answer-target frozen layers = the PARENT's committed conventions (plan §4.3:
# per-question 25 / query-averaged 27 — eval_results/issue_928/
# recon_skill_grid.json frozen_layers; pre-registered fixed positions).
PMA_ANS_FROZEN_LAYER = {"indiv": 25, "avg_q": 27}


def phase(name: str) -> None:
    """Poller-visible phase breadcrumb (one line per pipeline phase)."""
    logger.info("[phase=%s]", name)


# ── staging (pinned revisions) ────────────────────────────────────────────────


def stage_mlc_store(store_dir: Path, revision: str, ctx_ids: list[str], full_grid: bool) -> None:
    """Stage the matched-length summary store (FLAT layout: manifest + blobs in
    one folder, both on the Hub and locally — ``Store(dir, blob_subdir=".")``).

    Full grid: scoped ``list_repo_tree`` enumeration + ≤6-worker per-file
    fetches at the pinned revision (never ``snapshot_download`` — #833);
    identity hub-rel→local-rel mapping (flat→flat), fail-loud if the listing
    carries no ``manifest.json``. Subset: direct known-path fetches.
    """
    if not full_grid:
        pairs = [f"{MLC_STORE_HF_PREFIX}/manifest.json"]
        pairs += [f"{MLC_STORE_HF_PREFIX}/{c}.pt" for c in ctx_ids]
        for full in pairs:
            dest = store_dir / full[len(MLC_STORE_HF_PREFIX) + 1 :]
            if dest.is_file():
                continue
            _hf_fetch_one(full, revision, dest)
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    paths = list_hf_files_under_path(
        HfApi(),
        HF_DATA_REPO,
        MLC_STORE_HF_PREFIX,
        repo_type="dataset",
        revision=revision,
    )
    if not paths:
        raise RuntimeError(f"no files under {MLC_STORE_HF_PREFIX} at revision {revision}")
    rels = {full[len(MLC_STORE_HF_PREFIX) + 1 :]: full for full in paths}
    if "manifest.json" not in rels:
        raise RuntimeError(
            f"no manifest.json under {MLC_STORE_HF_PREFIX} at revision {revision} — "
            "Store() requires it at the store root; refusing a doomed stage (h(iv))"
        )
    missing = {rel: full for rel, full in rels.items() if not (store_dir / rel).is_file()}
    if not missing:
        logger.info("[phase=stage] MLC store already local (%d files) — skip", len(rels))
        return
    logger.info(
        "[phase=stage] fetching %d/%d MLC store files @ %s", len(missing), len(rels), revision[:12]
    )
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {
            ex.submit(_hf_fetch_one, full, revision, store_dir / rel): rel
            for rel, full in missing.items()
        }
        for fut in as_completed(futs):
            fut.result()  # re-raise loud
    still = [rel for rel in rels if not (store_dir / rel).is_file()]
    if still:
        raise RuntimeError(f"MLC store staging incomplete: {len(still)} missing ({still[:3]})")


def stage_mlc_decomps(decomp_dir: Path, revision: str) -> dict[str, Path]:
    """Stage the committed MLC decomps at the PINNED revision (plan §10: the
    canonical copies; local ``eval_results/`` copies are untracked). The dir is
    revision-suffixed by the caller so a stale copy from another revision can
    never be silently reused."""
    out: dict[str, Path] = {}
    for regime in ("indiv", "avg_q"):
        dest = decomp_dir / f"decomp_{regime}_mlc.pt"
        if not dest.is_file():
            _hf_fetch_one(f"{MLC_DECOMP_HF_PREFIX}/decomp_{regime}_mlc.pt", revision, dest)
        out[regime] = dest
    return out


def stage_mlc_bootstrap_json(local_hint: Path, revision: str, dest: Path) -> Path:
    """Committed MLC bootstrap JSON: local-first (git-committed), HF fallback
    at the pinned revision (#779 r4-r5: git-clone lanes may lack local data)."""
    if local_hint.is_file():
        return local_hint
    if not dest.is_file():
        _hf_fetch_one(f"{MLC_RESULTS_PREFIX}/mlc_bootstrap_deltaskill.json", revision, dest)
    return dest


# ── committed-decomp loading + shared skill helpers ───────────────────────────


def load_committed_decomp(path: Path) -> dict[tuple[str, str, int], dict]:
    """Load a ``decomp_*_mlc.pt`` (string-keyed ``str((arm, combo, layer))``)
    keeping ``ctx_order`` (the cross-round subsetting key)."""
    raw = torch.load(path, weights_only=False)
    out: dict[tuple[str, str, int], dict] = {}
    for k, v in raw.items():
        arm, combo, layer = ast.literal_eval(k) if isinstance(k, str) else k
        out[(str(arm), str(combo), int(layer))] = {
            "ss_res": np.asarray(v["ss_res"], dtype=np.float64),
            "ss_tot": np.asarray(v["ss_tot"], dtype=np.float64),
            "ctx_order": list(v["ctx_order"]),
        }
    return out


def _obs_skill(entry: dict) -> float:
    tot = float(entry["ss_tot"].sum())
    return float("nan") if tot < 1e-12 else 1.0 - float(entry["ss_res"].sum()) / tot


def _layers_of(decomp: dict, arm: str) -> list[int]:
    return sorted(la for (a, c, la) in decomp if a == arm and c == COMBO)


def _best_layer(decomp: dict, arm: str) -> int:
    las = _layers_of(decomp, arm)
    return int(las[int(np.nanargmax([_obs_skill(decomp[(arm, COMBO, la)]) for la in las]))])


def subset_committed_entry(entry: dict, ctx_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Subset a committed per-context (ss_res, ss_tot) pair to ``ctx_ids``
    (positional match on the committed ``ctx_order``; fail-loud on a miss)."""
    pos = {c: i for i, c in enumerate(entry["ctx_order"])}
    missing = [c for c in ctx_ids if c not in pos]
    if missing:
        raise RuntimeError(
            f"cross-round pairing FAILED: contexts {missing[:3]} absent from the committed "
            "decomp ctx_order — refusing an unpaired convention contrast"
        )
    sel = [pos[c] for c in ctx_ids]
    return entry["ss_res"][sel], entry["ss_tot"][sel]


# ── joined store (prefix/query from THIS round; everything else MLC-joined) ───


class JoinedStore:
    """Row-aligned join of the new prefix store and the matched-length store.

    Routing (the plan-§4.0 provenance contract, persisted in
    ``pma_capture_gates.json``): ``prefix_mean``/``query_mean`` come from THIS
    round's store; EVERY other rep — the conditioning slices, ``ctx_mean``
    (the ``pma_ctx_ans`` input), and both targets (``ans_rem_mean``,
    ``ans_mean`` incl. the identity ceiling) — is STORE-JOINED byte-identical
    from the matched-length store. The fresh ctx/cot/ans captures are parity
    references ONLY. Alignment is asserted per context on ``probe_indices``.
    """

    _PMA_OWNED = ("prefix_mean", "query_mean")

    def __init__(self, pma_store: Store, mlc_store: Store):
        assert pma_store.ctx_ids == mlc_store.ctx_ids, "ctx set drift between stores"
        assert [int(x) for x in pma_store.layers] == [int(x) for x in mlc_store.layers]
        for c in pma_store.ctx_ids:
            a = [int(q) for q in pma_store.blobs[c]["probe_indices"]]
            b = [int(q) for q in mlc_store.blobs[c]["probe_indices"]]
            if a != b:
                raise RuntimeError(f"probe_indices drift between stores for context {c}")
            floors = mlc_store.blobs[c].get("mlc_floors")
            if floors != {"k_min": MLC_K_MIN, "rem_min": MLC_REM_MIN}:
                raise RuntimeError(f"MLC blob {c} floors {floors} != inherited pins")
        self.pma, self.mlc = pma_store, mlc_store
        self.ctx_ids = pma_store.ctx_ids
        self.families = pma_store.families
        self.layers = pma_store.layers
        self.groups = pma_store.groups
        self.fam_of_ctx = pma_store.fam_of_ctx

    def _owner(self, sname: str) -> Store:
        return self.pma if sname in self._PMA_OWNED else self.mlc

    def indiv(self, sname: str, li: int) -> np.ndarray:
        return self._owner(sname).indiv(sname, li)

    def avgq(self, sname: str, li: int) -> np.ndarray:
        return self._owner(sname).avgq(sname, li)

    def identity_digest(self) -> str:
        pair = f"{self.pma.identity_digest()}|{self.mlc.identity_digest()}"
        return hashlib.sha256(pair.encode()).hexdigest()[:16]


# ── row bookkeeping (per-row prefix/query/K lengths for figures) ─────────────


def pma_row_bookkeeping(row: dict, probe_index: int) -> dict:
    """Per-row prefix/query/CoT/K token lengths (scatter + audit inputs)."""
    spans = row["spans"]
    return {
        "probe_index": int(probe_index),
        "len_prefix": int(spans["prefix"][1] - spans["prefix"][0]),
        "len_query": int(spans["query"][1] - spans["query"][0]),
        "len_cot": int(spans["cot"][1] - spans["cot"][0]),
        "K": int(spans["cot_lastK"][1] - spans["cot_lastK"][0]),
    }


# ── fit battery (both regimes, TWO targets — plan §4.3) ───────────────────────


def _merge_pma_unit(unit: dict, grid: dict, null_matrix: dict, decomp: dict, coherence: dict):
    """Fold one per-layer checkpoint unit into the regime accumulators."""
    for arm, schemes in unit["grid"].items():
        for scheme, cells in schemes.items():
            grid.setdefault(arm, {}).setdefault(scheme, []).extend(cells)
    for arm, by_layer in unit["null"].items():
        null_matrix.setdefault(arm, {}).update(by_layer)
    decomp.update(unit["decomp"])
    if unit.get("coherence") is not None:
        coherence[str(unit["layer"])] = unit["coherence"]


def assert_basis_coherence(
    y_rem: np.ndarray,
    folds_loco,
    committed_decomp: dict,
    layer: int,
    ident_arm: str = "mlc_ident",
    rtol: float = BASIS_COHERENCE_RTOL,
) -> dict:
    """PCA-basis coherence (plan §4.3, kill criterion): the recomputed
    remainder-target per-context ss_tot must match the committed identity
    arm's stored values at this layer (rtol; ss-invariant to PCA sign flips).
    """
    key = (ident_arm, COMBO, layer)
    if key not in committed_decomp:
        raise RuntimeError(f"basis-coherence: committed decomp has no {key} row")
    stored = committed_decomp[key]["ss_tot"]
    computed = ss_tot_by_group(y_rem, folds_loco)
    if stored.shape != computed.shape:
        raise RuntimeError(
            f"basis-coherence shape mismatch @ L{layer}: {stored.shape} != {computed.shape}"
        )
    max_rel = float(np.max(np.abs(stored - computed) / np.maximum(np.abs(stored), 1e-300)))
    if max_rel > rtol:
        raise RuntimeError(
            f"PCA basis-coherence FAILED @ L{layer}: recomputed identity ss_tot deviates "
            f"max_rel={max_rel:.3e} > rtol {rtol} from the committed decomp — the remainder "
            "target space drifted; refusing to fit (plan §7 kill criterion)"
        )
    return {"max_rel": max_rel, "rtol": rtol}


def fit_pma_regime(  # noqa: C901 — linear per-layer two-target arm battery; see comments
    store: JoinedStore,
    regime: str,
    layers_idx: list[int],
    device: str,
    n_perms: int,
    draw_chunk: int,
    committed_decomp: dict | None,
    coherence_binding: bool,
    checkpoint_dir: Path | None = None,
) -> tuple[dict, dict, dict, dict]:
    """All 9 PMA arms × layers × {LOCO, LOFO} for one regime, on TWO targets:
    the SHARED PCA-48 answer-REMAINDER (6 remainder arms) and the parent-parity
    PCA-48 full-ANSWER (3 answer arms). Nulls for the 6 registered arms only.

    Inherited conventions (v6 §4.3 verbatim): full-data PCA basis with
    per-fold train centering; ``avg_q`` per-fold X standardization, ``indiv``
    full-data X standardization; nested-CV λ; fp64. ONE ``GroupRidgeDesign``
    per (input-rep, scheme) per layer, SHARED across the two targets
    (Y-independent) and freed per layer. Per-(regime, layer) checkpoint units.
    Returns ``(grid, null_matrix, decomp, coherence_by_layer)``.
    """
    n_ctx = len(store.ctx_ids)
    if regime == "avg_q":
        groups = np.arange(n_ctx)
        mat = store.avgq
        standardization = "per_fold"
    else:
        groups = store.groups
        mat = store.indiv
        standardization = "full_data"
    group_order = list(range(n_ctx))
    folds_loco = group_folds(groups, group_order)
    fam_groups = store.fam_of_ctx[groups]
    fam_order = sorted(set(store.fam_of_ctx.tolist()))
    folds_lofo = group_folds(fam_groups, fam_order) if len(fam_order) > 1 else None
    perm = make_group_perm_matrix(
        groups, group_order, n_perms, np.random.default_rng(SHUFFLE_NULL_SEED)
    )

    grid: dict = {}
    null_matrix: dict = {}
    decomp: dict = {}
    coherence: dict = {}
    input_reps = tuple(dict.fromkeys(r for reps in PMA_ARM_INPUTS.values() for r in reps))
    for li in layers_idx:
        layer = int(store.layers[li])
        unit_path = (checkpoint_dir / f"layer_{layer}.pt") if checkpoint_dir else None
        if unit_path is not None and unit_path.is_file():
            unit = torch.load(unit_path, weights_only=False)
            assert int(unit["layer"]) == layer, (unit["layer"], layer)
            _merge_pma_unit(unit, grid, null_matrix, decomp, coherence)
            logger.info("[phase=fit] regime=%s layer %d SKIPPED (resumed unit)", regime, layer)
            continue
        t_layer = time.time()
        X_by_rep = {rep: mat(rep, li) for rep in input_reps}
        # TWO targets, both STORE-JOINED from the matched-length store (§4.0):
        Y_rem, _mu_r, _c_r = _pca_target(mat("ans_rem_mean", li).copy())
        Y_ans, _mu_a, _c_a = _pca_target(mat("ans_mean", li).copy())
        layer_coherence = None
        if coherence_binding and committed_decomp is not None:
            layer_coherence = assert_basis_coherence(Y_rem, folds_loco, committed_decomp, layer)

        def arm_X(arm: str, _x=X_by_rep) -> np.ndarray:
            reps = PMA_ARM_INPUTS[arm]
            if len(reps) == 1:
                return _x[reps[0]]
            return np.concatenate([_x[r] for r in reps], axis=1)

        def arm_Y(arm: str, _yr=Y_rem, _ya=Y_ans) -> np.ndarray:
            return _yr if arm in PMA_REM_ARM_INPUTS else _ya

        schemes = ["loco"] + (["lofo"] if folds_lofo is not None else [])
        grid_l: dict = {}
        null_l: dict = {}
        decomp_l: dict = {}
        designs: dict[tuple, GroupRidgeDesign] = {}
        for scheme in schemes:
            folds = folds_loco if scheme == "loco" else folds_lofo
            for arm in PMA_ARM_INPUTS:
                dkey = (PMA_ARM_INPUTS[arm], scheme)
                if dkey not in designs:
                    designs[dkey] = GroupRidgeDesign(
                        arm_X(arm), folds, device=device, standardization=standardization
                    )
                des = designs[dkey]
                Y = arm_Y(arm)
                pred, _, _ = fit_predict_grouped(des, Y)
                res = grouped_skill(pred, Y, folds)
                if not np.isfinite(res["skill"]):
                    raise RuntimeError(
                        f"non-finite fit: arm={arm} regime={regime} layer={layer} "
                        f"scheme={scheme} (plan §7 kill criterion)"
                    )
                grid_l.setdefault(arm, {}).setdefault(scheme, []).append(
                    {"layer": layer, "skill": res["skill"], "n": len(groups)}
                )
                if scheme == "loco":
                    decomp_l[(arm, COMBO, layer)] = {
                        "ss_res": np.asarray(res["ss_res_by_group"]),
                        "ss_tot": np.asarray(res["ss_tot_by_group"]),
                        "ctx_order": list(store.ctx_ids),
                    }
        if n_perms > 0:
            # registered arms ONLY (plan §6), grouped by shared design so a
            # design serving both targets nulls both in ONE batched call.
            by_design: dict[tuple, list[str]] = {}
            for arm in PMA_REGISTERED_ARMS:
                by_design.setdefault((PMA_ARM_INPUTS[arm], "loco"), []).append(arm)
            for dkey, arms in by_design.items():
                des = designs[dkey]
                draws = grouped_null_skills_multi(
                    des, [(arm_Y(a), None) for a in arms], perm, draw_chunk=draw_chunk
                )
                for a, dr in zip(arms, draws, strict=True):
                    null_l.setdefault(a, {})[str(layer)] = dr
        for d in designs.values():
            d.free()
        unit = {
            "layer": layer,
            "grid": grid_l,
            "null": null_l,
            "decomp": decomp_l,
            "coherence": layer_coherence,
        }
        if unit_path is not None:
            _atomic_torch_save(unit, unit_path)
        _merge_pma_unit(unit, grid, null_matrix, decomp, coherence)
        logger.info(
            "[phase=fit] regime=%s layer %d done in %.1fs%s",
            regime,
            layer,
            time.time() - t_layer,
            " (unit persisted)" if unit_path is not None else "",
        )
    return grid, null_matrix, decomp, coherence


# ── committed-artifact alignment + pair coverage (fail-loud) ──────────────────


def assert_committed_bootstrap_alignment(
    committed_boot: dict,
    committed_decomp_by_regime: dict,
    n_ctx: int,
    n_boot: int,
    full_grid: bool,
) -> dict:
    """Resample-convention alignment vs the committed MLC bootstrap artifact.

    Always: committed seed == inherited ``BOOTSTRAP_SEED``. Full grid:
    committed n_boot == inherited ``BOOTSTRAP_DRAWS`` AND this run's n_boot ==
    committed (production must pair draws 1:1). The per-regime digest assert
    itself fires in ``pma_bootstrap_statistics`` whenever (n_ctx, n_boot)
    match the committed grid — recorded here for the gates JSON."""
    if int(committed_boot["seed"]) != BOOTSTRAP_SEED:
        raise RuntimeError(
            f"committed MLC bootstrap seed {committed_boot['seed']} != inherited "
            f"{BOOTSTRAP_SEED} — refusing an unpaired resample convention"
        )
    if full_grid and (
        int(committed_boot["n_boot"]) != BOOTSTRAP_DRAWS or n_boot != int(committed_boot["n_boot"])
    ):
        raise RuntimeError(
            f"full-grid run: n_boot (run={n_boot}, committed={committed_boot['n_boot']}) must "
            f"both equal the inherited {BOOTSTRAP_DRAWS} for draw-aligned cross-round contrasts"
        )
    committed_n_ctx = {
        r: len(next(iter(d.values()))["ctx_order"]) for r, d in committed_decomp_by_regime.items()
    }
    if full_grid and any(v != n_ctx for v in committed_n_ctx.values()):
        raise RuntimeError(
            f"full-grid run has n_ctx={n_ctx} != committed grid {committed_n_ctx} — the paired "
            "per-context convention would not match"
        )
    comparable = all(v == n_ctx for v in committed_n_ctx.values()) and n_boot == int(
        committed_boot["n_boot"]
    )
    return {
        "committed_seed": int(committed_boot["seed"]),
        "committed_n_boot": int(committed_boot["n_boot"]),
        "committed_n_ctx": committed_n_ctx,
        "digest_binding": comparable,
        "committed_digest_by_regime": {
            r: committed_boot["by_regime"][r]["resample_matrix_digest"]
            for r in committed_boot["by_regime"]
        },
    }


def assert_pma_pair_coverage(decomp: dict, committed_decomp: dict, n_ctx: int) -> dict:
    """Paired-contrast row-coverage set-check (plan §6, fail-loud): every
    registered (arm × layer) row exists for BOTH arms of every registered
    read over the IDENTICAL context order, and every cross-round mlc arm has
    full committed coverage, BEFORE any contrast."""
    within_arms = {a for _n, hi, lo in PMA_REGISTERED_READS for a in (hi, lo)}
    within_arms |= {READ_5D[1], READ_5D[2], PMA_ANS_IDENT_ARM}
    layer_sets = {}
    for arm in within_arms:
        las = _layers_of(decomp, arm)
        if not las:
            raise RuntimeError(f"pair-coverage set-check FAILED: no decomp rows for arm {arm!r}")
        layer_sets[arm] = las
    ref_arm = "pma_pfx"
    ref_layers = layer_sets[ref_arm]
    ref_order = decomp[(ref_arm, COMBO, ref_layers[0])]["ctx_order"]
    for arm in sorted(within_arms):
        if layer_sets[arm] != ref_layers:
            raise RuntimeError(
                f"pair-coverage set-check FAILED: arm {arm!r} layers {layer_sets[arm]} != "
                f"baseline layers {ref_layers}"
            )
        for la in ref_layers:
            d = decomp[(arm, COMBO, la)]
            if len(d["ss_res"]) != n_ctx or d.get("ctx_order") != ref_order:
                raise RuntimeError(
                    f"pair-coverage set-check FAILED: arm {arm!r} layer {la} has "
                    f"{len(d['ss_res'])} groups / drifted ctx_order (want {n_ctx})"
                )
    for _name, _hi, mlc_arm in PMA_CONVENTION_READS:
        las = _layers_of(committed_decomp, mlc_arm)
        if not las:
            raise RuntimeError(
                f"pair-coverage set-check FAILED: committed decomp has no rows for {mlc_arm!r}"
            )
        for la in las:
            subset_committed_entry(committed_decomp[(mlc_arm, COMBO, la)], ref_order)
    return {"layers": ref_layers, "n_ctx": n_ctx, "ctx_order": ref_order, "pass": True}


# ── bootstrap contrasts (plan §6: reads 1-4 + the read-5 family) ─────────────


def pma_bootstrap_statistics(  # noqa: C901 — linear read battery (plan §6 exact order)
    decomp: dict,
    committed_decomp: dict,
    regime: str,
    n_ctx: int,
    n_boot: int,
    alignment: dict,
    full_grid: bool,
    ans_frozen_layer: dict[str, int] | None = None,
) -> dict:
    """Paired-bootstrap Δskill CIs (plan §6) off ONE shared seed-42 resample
    matrix — digest-ASSERTED equal to the committed per-regime digest whenever
    the grid is comparable (kill criterion; the literal
    ``EXPECTED_RESAMPLE_DIGEST`` additionally binds on the full grid), which
    is what makes the read-5 cross-round contrasts draw-aligned.

    PRIMARY layer convention (remainder target): the PREFIX-ONLY baseline's
    full-data best LOCO layer, re-derived ONCE per regime before any draw.
    Answer-target reads are frozen at the PARENT's committed layers
    (25 indiv / 27 avg_q); on a run whose layer subset lacks that layer
    (tiny-model smoke) the recorded fallback is the ``pma_ctx_ans`` full-data
    best LOCO layer — full-grid runs fail loud instead.

    ``ans_frozen_layer`` (DEFAULT-PRESERVING, #1005 §4.6: ``None`` ⇒ the #928
    ``PMA_ANS_FROZEN_LAYER`` committed conventions byte-for-byte) overrides
    the answer-target frozen-layer map for a replication whose frozen-layer
    RULE re-derives indices on its own data (the parent's realized 25/27 are
    reference points, not pins — #1005 passes its OWN F1 frozen layers).
    """
    idx = make_bootstrap_index_matrix(n_ctx, n_boot, BOOTSTRAP_SEED)
    matrix_digest = hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()[:16]
    digest_record: dict = {"digest": matrix_digest, "binding": bool(alignment["digest_binding"])}
    if alignment["digest_binding"]:
        want = alignment["committed_digest_by_regime"][regime]
        if matrix_digest != want:
            raise RuntimeError(
                f"resample-matrix digest mismatch ({regime}): regenerated {matrix_digest} != "
                f"committed {want} — cross-round contrasts would NOT be draw-aligned "
                "(plan §7 kill criterion)"
            )
        if full_grid and matrix_digest != EXPECTED_RESAMPLE_DIGEST:
            raise RuntimeError(
                f"full-grid resample digest {matrix_digest} != expected "
                f"{EXPECTED_RESAMPLE_DIGEST} (plan §6 pinned constant)"
            )
    else:
        digest_record["note"] = "not-comparable grid (n_ctx/n_boot differ from committed) — skip"

    def obs(arm: str, layer: int, d: dict = decomp) -> float:
        return _obs_skill(d[(arm, COMBO, layer)])

    def draws_for(arm: str, layer: int) -> np.ndarray:
        d = decomp[(arm, COMBO, layer)]
        return bootstrap_skills(d["ss_res"], d["ss_tot"], idx)

    def per_layer_draws(arm: str) -> np.ndarray:
        return np.stack([draws_for(arm, la) for la in _layers_of(decomp, arm)], axis=1)

    ctx_order = decomp[("pma_pfx", COMBO, _layers_of(decomp, "pma_pfx")[0])]["ctx_order"]

    def committed_obs_and_draws(mlc_arm: str, layer: int) -> tuple[float, np.ndarray]:
        ss_res, ss_tot = subset_committed_entry(
            committed_decomp[(mlc_arm, COMBO, layer)], ctx_order
        )
        tot = float(ss_tot.sum())
        o = float("nan") if tot < 1e-12 else 1.0 - float(ss_res.sum()) / tot
        return o, bootstrap_skills(ss_res, ss_tot, idx)

    # frozen-layer conventions (plan §4.3).
    l_pfx = _best_layer(decomp, "pma_pfx")
    l_mlc = _best_layer(committed_decomp, "mlc_ctx")
    ans_frozen = PMA_ANS_FROZEN_LAYER if ans_frozen_layer is None else ans_frozen_layer
    l_ans_want = ans_frozen[regime]
    ans_layers = _layers_of(decomp, "pma_pfx_ans")
    if l_ans_want in ans_layers:
        l_ans, ans_note = l_ans_want, "parent committed convention"
    elif full_grid:
        raise RuntimeError(
            f"full-grid run lacks the parent answer-target frozen layer {l_ans_want} "
            f"(fit layers {ans_layers[:5]}…) — refusing an unregistered convention"
        )
    else:
        l_ans = _best_layer(decomp, "pma_ctx_ans")
        ans_note = f"FALLBACK ctx_ans full-data best (subset run lacks L{l_ans_want})"

    out: dict = {
        "layer_conventions": {
            "primary_frozen_pfx_baseline_best_layer": l_pfx,
            "answer_target_frozen_layer": l_ans,
            "answer_target_frozen_note": ans_note,
            "committed_mlc_ctx_frozen_layer": l_mlc,
            "note": (
                "remainder primary = frozen prefix-only baseline's full-data best LOCO layer, "
                "re-derived ONCE per regime before any draw (plan §4.3); answer-target reads "
                "frozen at the parent's committed layers; cross-round contrasts pair each arm "
                "at its own convention's frozen layer, with a same-layer sensitivity read."
            ),
        },
        "resample_matrix": digest_record,
        "statistics": {},
        "convention_contrasts": {},
    }
    for name, arm_hi, arm_lo in PMA_REGISTERED_READS:
        obs_p = obs(arm_hi, l_pfx) - obs(arm_lo, l_pfx)
        dr_p = draws_for(arm_hi, l_pfx) - draws_for(arm_lo, l_pfx)
        l_hi, l_lo = _best_layer(decomp, arm_hi), _best_layer(decomp, arm_lo)
        obs_ob = obs(arm_hi, l_hi) - obs(arm_lo, l_lo)
        dr_ob = draws_for(arm_hi, l_hi) - draws_for(arm_lo, l_lo)
        dr_bb = np.nanmax(per_layer_draws(arm_hi), axis=1) - np.nanmax(
            per_layer_draws(arm_lo), axis=1
        )
        out["statistics"][name] = {
            "arms": {"hi": arm_hi, "lo": arm_lo},
            "primary_frozen_pfx_baseline_best": stat_summary(obs_p, dr_p),
            "secondary_own_best_frozen_full_data": {
                "layers": {"hi": l_hi, "lo": l_lo},
                **stat_summary(obs_ob, dr_ob),
            },
            "secondary_best_vs_best_inherited": stat_summary(obs_ob, dr_bb),
        }
    # read-5 family: cross-round convention contrasts (5a-c) + within-round 5d.
    for name, pma_arm, mlc_arm in PMA_CONVENTION_READS:
        obs_c, dr_c = committed_obs_and_draws(mlc_arm, l_mlc)
        headline = stat_summary(obs(pma_arm, l_pfx) - obs_c, draws_for(pma_arm, l_pfx) - dr_c)
        entry = {
            "arms": {"pma": pma_arm, "mlc": mlc_arm},
            "layers": {"pma": l_pfx, "mlc": l_mlc},
            "headline_each_at_own_frozen": headline,
        }
        if l_mlc in _layers_of(decomp, pma_arm):
            obs_c2, dr_c2 = committed_obs_and_draws(mlc_arm, l_mlc)
            entry["sensitivity_same_layer"] = {
                "layer": l_mlc,
                **stat_summary(obs(pma_arm, l_mlc) - obs_c2, draws_for(pma_arm, l_mlc) - dr_c2),
            }
        else:
            entry["sensitivity_same_layer"] = {"note": f"layer {l_mlc} not in this run's fit set"}
        out["convention_contrasts"][name] = entry
    name5d, hi5d, lo5d = READ_5D
    out["convention_contrasts"][name5d] = {
        "arms": {"hi": hi5d, "lo": lo5d},
        "layer": l_ans,
        "target": "full-answer (parent parity)",
        **stat_summary(
            obs(hi5d, l_ans) - obs(lo5d, l_ans), draws_for(hi5d, l_ans) - draws_for(lo5d, l_ans)
        ),
    }
    # absolute per-arm reads at the frozen conventions (bar-figure inputs).
    out["absolute_at_frozen"] = {}
    for arm in (*PMA_REM_ARMS,):
        out["absolute_at_frozen"][arm] = {
            "layer": l_pfx,
            **stat_summary(obs(arm, l_pfx), draws_for(arm, l_pfx)),
        }
    for arm in PMA_ANS_ARMS:
        out["absolute_at_frozen"][arm] = {
            "layer": l_ans,
            **stat_summary(obs(arm, l_ans), draws_for(arm, l_ans)),
        }
    # degenerate-prefix-cell sensitivity re-read of read 1 (plan §6 note iii —
    # a sensitivity re-read at the frozen layer, never a new registered read).
    keep = [i for i, c in enumerate(ctx_order) if c not in DEGENERATE_PREFIX_CELLS]
    if len(keep) < len(ctx_order) and len(keep) >= 3:
        idx_s = make_bootstrap_index_matrix(len(keep), n_boot, BOOTSTRAP_SEED)
        d_hi = decomp[("pma_pfx_cotK", COMBO, l_pfx)]
        d_lo = decomp[("pma_pfx_apfx", COMBO, l_pfx)]

        def _sub_obs_draws(d: dict) -> tuple[float, np.ndarray]:
            ss_res, ss_tot = d["ss_res"][keep], d["ss_tot"][keep]
            tot = float(ss_tot.sum())
            o = float("nan") if tot < 1e-12 else 1.0 - float(ss_res.sum()) / tot
            return o, bootstrap_skills(ss_res, ss_tot, idx_s)

        o_hi, dr_hi = _sub_obs_draws(d_hi)
        o_lo, dr_lo = _sub_obs_draws(d_lo)
        out["sensitivity_excluding_degenerate_prefix_cells"] = {
            "excluded": [c for c in ctx_order if c in DEGENERATE_PREFIX_CELLS],
            "n_ctx": len(keep),
            "read": "read1 @ primary frozen layer (own resample matrix, seed 42)",
            **stat_summary(o_hi - o_lo, dr_hi - dr_lo),
        }
    else:
        out["sensitivity_excluding_degenerate_prefix_cells"] = {
            "note": "degenerate cells absent from this run's ctx subset — skipped"
        }
    return out


def pma_null_band_analysis(null_matrix: dict, decomp: dict, committed_decomp: dict) -> dict:
    """Selection-symmetric max-over-layers null bands + band-vs-ceiling.

    Remainder-arm bands read against the COMMITTED remainder identity ceiling
    (``mlc_ident`` max-over-layers, cited per plan §6 — no fresh remainder
    identity this round); answer-arm bands against this round's fresh
    ``pma_ident_ans`` ceiling."""
    ceil_rem = float(
        np.nanmax(
            [
                _obs_skill(committed_decomp[("mlc_ident", COMBO, la)])
                for la in _layers_of(committed_decomp, "mlc_ident")
            ]
        )
    )
    ceil_ans = float(
        np.nanmax(
            [
                _obs_skill(decomp[(PMA_ANS_IDENT_ARM, COMBO, la)])
                for la in _layers_of(decomp, PMA_ANS_IDENT_ARM)
            ]
        )
    )
    out: dict = {
        "remainder_ceiling_committed_mlc_ident": ceil_rem,
        "answer_ceiling_fresh_pma_ident": ceil_ans,
        "arms": {},
    }
    for arm, by_layer in null_matrix.items():
        ceiling = ceil_rem if arm in PMA_REM_ARM_INPUTS else ceil_ans
        layers = sorted(by_layer, key=int)
        draws = np.asarray([by_layer[la] for la in layers], dtype=np.float64)  # (L, B)
        max_draws = np.nanmax(draws, axis=0)
        band_hi = float(np.nanpercentile(max_draws, 97.5))
        se = float(np.nanstd(max_draws))
        obs_best = float(np.nanmax([_obs_skill(decomp[(arm, COMBO, int(la))]) for la in layers]))
        out["arms"][arm] = {
            "band_p2p5": float(np.nanpercentile(max_draws, 2.5)),
            "band_p97p5": band_hi,
            "null_se": se,
            "observed_best_over_layers": obs_best,
            "ceiling": ceiling,
            "uninformative_by_construction": bool(band_hi >= ceiling - se),
        }
    return out


# ── figures (hero + forests + curves + audits — plan §6) ─────────────────────


def make_pma_figures(  # noqa: C901 — linear figure set; one block per figure
    figdir: Path,
    grid_by_regime: dict,
    boot_by_regime: dict,
    null_bands_by_regime: dict,
    decomp_by_regime: dict,
    committed_boot: dict,
    bookkeeping: dict,
) -> list[str]:
    """pma_* figure set: hero bars (prefix vs committed context convention),
    read-1 forest (+ committed reference row), per-layer curves, per-context
    read-1 scatter (degenerate cells flagged), convention forest, query-audit
    bars, answer-target parity bars. Returns the written stems."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    figdir.mkdir(parents=True, exist_ok=True)
    stems: list[str] = []
    rel = figdir.name  # e.g. issue_928

    def save(fig, stem: str) -> None:
        savefig_paper(fig, f"{rel}/{stem}", dir=str(figdir.parent))
        plt.close(fig)
        stems.append(stem)

    def _bar_ci(ax, i, st, color):
        lo, hi = st["ci95"]
        ax.bar(i, st["observed"], color=color)
        ax.errorbar(
            i,
            st["observed"],
            yerr=[[max(0.0, st["observed"] - lo)], [max(0.0, hi - st["observed"])]],
            fmt="none",
            ecolor="black",
            capsize=3,
        )

    # 1) hero: prefix-convention bars beside the committed context-convention
    #    counterparts (each at its own convention's frozen layer) + ceilings.
    set_paper_style()
    boot = boot_by_regime["indiv"]
    absf = boot["absolute_at_frozen"]
    mlc_absf = committed_boot["by_regime"]["indiv"]["absolute_at_frozen"]
    pairs = [
        ("pma_pfx", "mlc_ctx", "prompt\nonly"),
        ("pma_pfx_apfx", "mlc_ctx_apfx", "+ answer\nprefix"),
        ("pma_pfx_cotK", "mlc_ctx_cotK", "+ truncated\nCoT"),
        ("pma_pfx_cotfull", "mlc_ctx_cotfull", "+ full\nCoT"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    cols = paper_palette(2)
    for i, (pma_arm, mlc_arm, lab) in enumerate(pairs):
        _bar_ci(ax, 2.6 * i, absf[pma_arm], cols[0])
        _bar_ci(ax, 2.6 * i + 1, mlc_absf[mlc_arm], cols[1])
        ax.text(2.6 * i + 0.5, -0.06, lab, ha="center", va="top", fontsize=8)
    bands = null_bands_by_regime["indiv"]
    ax.axhline(
        bands["remainder_ceiling_committed_mlc_ident"],
        color="black",
        lw=0.9,
        ls=":",
        label="identity ceiling (committed)",
    )
    ax.set_xticks([])
    lf = boot["layer_conventions"]["primary_frozen_pfx_baseline_best_layer"]
    lm = boot["layer_conventions"]["committed_mlc_ctx_frozen_layer"]
    ax.set_ylabel("held-out skill-over-mean R² (remainder target)")
    ax.set_title(f"Prefix (L{lf}) vs context (L{lm}) conventions — per-question")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=cols[0], label="prefix convention (this round)"),
            plt.Rectangle((0, 0), 1, 1, color=cols[1], label="context convention (committed)"),
        ],
        fontsize=8,
    )
    save(fig, "pma_hero_bars_indiv")

    # 2) read-1 forest (regimes × conventions) + committed context read 1.
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    rows = []
    for regime in ("indiv", "avg_q"):
        st = boot_by_regime[regime]["statistics"]["read1_primary_pfx_cotK_minus_pfx_apfx"]
        rows.append((f"{regime} · primary frozen", st["primary_frozen_pfx_baseline_best"]))
        rows.append((f"{regime} · own-best frozen", st["secondary_own_best_frozen_full_data"]))
        rows.append((f"{regime} · best-vs-best", st["secondary_best_vs_best_inherited"]))
        ref = committed_boot["by_regime"][regime]["statistics"][
            "read1_primary_ctx_cotK_minus_ctx_apfx"
        ]["primary_frozen_ctx_baseline_best"]
        rows.append((f"{regime} · context convention (committed ref)", ref))
    for yi, (label, st) in enumerate(rows):
        lo, hi = st["ci95"]
        color = "gray" if "committed ref" in label else "black"
        ax.errorbar(
            st["observed"],
            yi,
            xerr=[[max(0.0, st["observed"] - lo)], [max(0.0, hi - st["observed"])]],
            fmt="o",
            capsize=3,
            color=color,
        )
        ax.text(1.02, yi, label, transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_yticks([])
    ax.set_xlabel("Δskill (prefix+truncated-CoT − prefix+answer-prefix)")
    save(fig, "pma_forest_read1")

    # 3) per-layer skill curves per regime (+ null band + both ceilings).
    for regime, grid in grid_by_regime.items():
        set_paper_style()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        arms = [a for a in PMA_ARM_INPUTS if a in grid]
        base = paper_palette(min(8, len(arms)))
        for i, arm in enumerate(arms):
            cells = sorted(grid[arm]["loco"], key=lambda cc: cc["layer"])
            ax.plot(
                [cc["layer"] for cc in cells],
                [cc["skill"] for cc in cells],
                label=arm,
                color=base[i % len(base)],
                ls="-" if i < len(base) else "--",
                lw=1.4,
            )
        nb = null_bands_by_regime[regime]
        if nb["arms"]:
            hi = max(b["band_p97p5"] for b in nb["arms"].values())
            lo = min(b["band_p2p5"] for b in nb["arms"].values())
            ax.axhspan(lo, hi, color="gray", alpha=0.18, label="null band (max-over-layers)")
        ax.axhline(
            nb["remainder_ceiling_committed_mlc_ident"],
            color="black",
            lw=0.9,
            ls=":",
            label="remainder ceiling (committed)",
        )
        ax.axhline(
            nb["answer_ceiling_fresh_pma_ident"],
            color="black",
            lw=0.9,
            ls="-.",
            label="answer ceiling (fresh)",
        )
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out skill")
        ax.set_title(f"Prefix-based arms — {regime}")
        ax.legend(fontsize=7, ncols=2)
        save(fig, f"pma_skill_curves_{regime}")

    # 4) per-context read-1 Δ scatter vs median K and median CoT length
    #    (points labeled; degenerate prefix cells colored).
    set_paper_style()
    boot = boot_by_regime["indiv"]
    lf = boot["layer_conventions"]["primary_frozen_pfx_baseline_best_layer"]
    decomp = decomp_by_regime["indiv"]
    d_hi = decomp[("pma_pfx_cotK", COMBO, lf)]
    d_lo = decomp[("pma_pfx_apfx", COMBO, lf)]
    ctx_order = d_hi["ctx_order"]

    def per_ctx_skill(d: dict) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 - d["ss_res"] / d["ss_tot"]

    delta_c = per_ctx_skill(d_hi) - per_ctx_skill(d_lo)
    med_cot = {
        c: float(np.median([b["len_cot"] for b in books])) if books else float("nan")
        for c, books in bookkeeping.items()
    }
    med_k = {
        c: float(np.median([b["K"] for b in books])) if books else float("nan")
        for c, books in bookkeeping.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for ax, xs, xlabel in (
        (axes[0], med_cot, "median CoT length (tokens)"),
        (axes[1], med_k, "median K (tokens)"),
    ):
        for ci, c in enumerate(ctx_order):
            flagged = c in DEGENERATE_PREFIX_CELLS
            ax.scatter(
                xs.get(c, float("nan")),
                delta_c[ci],
                s=18 if flagged else 14,
                color="tab:red" if flagged else "tab:blue",
            )
            ax.annotate(c, (xs.get(c, float("nan")), delta_c[ci]), fontsize=5, alpha=0.7)
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(f"per-context Δskill (read 1) @ L{lf}")
    axes[0].set_title("red = near-degenerate prefix cells (flagged)")
    save(fig, "pma_percontext_read1_scatter")

    # 5) convention-contrast forest (read-5 family, indiv).
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    rows = []
    for name, entry in boot_by_regime["indiv"]["convention_contrasts"].items():
        st = entry.get("headline_each_at_own_frozen") or {
            k: entry[k] for k in ("observed", "ci95") if k in entry
        }
        if "observed" in st:
            rows.append((name, st))
    for yi, (label, st) in enumerate(rows):
        lo, hi = st["ci95"]
        ax.errorbar(
            st["observed"],
            yi,
            xerr=[[max(0.0, st["observed"] - lo)], [max(0.0, hi - st["observed"])]],
            fmt="o",
            capsize=3,
            color="black",
        )
        ax.text(1.02, yi, label, transform=ax.get_yaxis_transform(), va="center", fontsize=7)
    ax.axvline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_yticks([])
    ax.set_xlabel("Δskill (prefix convention − context convention)")
    save(fig, "pma_convention_forest")

    # 6) query-audit bars: query-only + prefix+query vs the committed context arm.
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    audit = ["pma_pfx", "pma_qry", "pma_pfx_qry"]
    cols = paper_palette(len(audit) + 1)
    absf = boot["absolute_at_frozen"]
    for i, arm in enumerate(audit):
        _bar_ci(ax, i, absf[arm], cols[i])
    _bar_ci(ax, len(audit), mlc_absf["mlc_ctx"], cols[len(audit)])
    ax.set_xticks(
        range(len(audit) + 1),
        [
            "prefix\nonly",
            "query only\n(exploratory)",
            "prefix+query\n(exploratory)",
            "context\n(committed)",
        ],
        fontsize=7,
    )
    ax.set_ylabel("held-out skill (remainder target)")
    ax.set_title("Query audit — does the two-part split recover the context arm?")
    save(fig, "pma_query_audit_bars")

    # 7) answer-target parity bars.
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ans_arms = ["pma_pfx_ans", "pma_ctx_ans", "pma_ident_ans"]
    cols = paper_palette(len(ans_arms))
    for i, arm in enumerate(ans_arms):
        _bar_ci(ax, i, absf[arm], cols[i])
    la = boot["layer_conventions"]["answer_target_frozen_layer"]
    ax.set_xticks(range(len(ans_arms)), ["prefix\ndirect", "context\ndirect", "identity\nceiling"])
    ax.set_ylabel("held-out skill (full-answer target)")
    ax.set_title(f"Parent-parity full-answer target @ L{la} (indiv)")
    save(fig, "pma_answer_target_bars")
    return stems


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:  # noqa: C901 — linear phase pipeline; see phase() markers
    ap = argparse.ArgumentParser(
        description="Issue #928 prefix-based mapping arms (follow-up plan v7)"
    )
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval_results" / "issue_928" / PMA_FOLLOWUP_LABEL),
    )
    ap.add_argument(
        "--store-dir",
        default=str(PROJECT_ROOT / "data" / "issue_928" / "store" / "prefix_summaries"),
        help="NEW prefix-summaries store (flat: manifest + blobs)",
    )
    ap.add_argument(
        "--mlc-store",
        default=str(PROJECT_ROOT / "data" / "issue_928" / "store" / "matched_length_summaries"),
        help="matched-length summary store (flat), staged @ the pinned revision",
    )
    ap.add_argument(
        "--mlc-decomp-dir",
        default=str(
            PROJECT_ROOT / "data" / "issue_928" / f"mlc_decomp_{MLC_ARTIFACTS_REVISION[:12]}"
        ),
        help="committed decomp_{indiv,avg_q}_mlc.pt staging dir (revision-suffixed)",
    )
    ap.add_argument(
        "--mlc-bootstrap-json",
        default=str(
            PROJECT_ROOT
            / "eval_results"
            / "issue_928"
            / "matched-length-answer-span-control"
            / "mlc_bootstrap_deltaskill.json"
        ),
        help="committed MLC bootstrap artifact (local-first; HF fallback @ pinned revision)",
    )
    ap.add_argument(
        "--rollouts",
        default=str(PROJECT_ROOT / "data" / "issue_928" / "raw_completions" / "thinking_rollouts"),
    )
    ap.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures" / "issue_928"))
    ap.add_argument(
        "--contexts", type=int, default=None, help="cap contexts (pod Phase-0 smoke = 3)"
    )
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="layer-INDEX subset")
    ap.add_argument("--model", default=None, help="override the MLC manifest's model")
    ap.add_argument("--device", default=None, help="fit device: CLI > EPM_FIT_DEVICE > auto")
    ap.add_argument("--n-perms", type=int, default=SHUFFLE_NULL_PERMS)
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_DRAWS)
    ap.add_argument("--draw-chunk", type=int, default=16)
    ap.add_argument("--batch-probes", type=int, default=8)
    ap.add_argument(
        "--capture-token-budget",
        type=int,
        default=32768,
        help="max BxT padded tokens per capture forward (parent Phase-B bound)",
    )
    ap.add_argument(
        "--stop-after",
        choices=["stage", "asserts", "spans"],
        default=None,
        help="early exit after the named phase (VM CPU partial smokes; same code path)",
    )
    ap.add_argument("--skip-parity-gate", action="store_true", help="skip the serial ridge gate")
    ap.add_argument(
        "--sentinel-dir",
        default=None,
        help="override the /workspace/logs sentinel dir (smoke runs redirect to scratch)",
    )
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_dir = Path(args.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    mlc_store_dir = Path(args.mlc_store)
    mlc_decomp_dir = Path(args.mlc_decomp_dir)
    rollouts_dir = Path(args.rollouts)
    fit_device = _resolve_device(_requested_device(args.device))
    logger.info("fit device: %s", fit_device)
    t0 = time.time()

    # ── stage (pinned revisions; every phase's cell list derives from ctx_ids) ─
    phase("stage")
    battery = resolve_battery(None)
    ctx_ids_all, _families_battery = context_order_and_families(battery)
    ctx_ids = ctx_ids_all[: args.contexts] if args.contexts else ctx_ids_all
    full_grid = len(ctx_ids) == len(ctx_ids_all)
    logger.info("contexts=%d (full_grid=%s)", len(ctx_ids), full_grid)
    stage_rollouts(rollouts_dir, ctx_ids, ROLLOUTS_REVISION)
    stage_mlc_store(mlc_store_dir, MLC_ARTIFACTS_REVISION, ctx_ids, full_grid)
    decomp_paths = stage_mlc_decomps(mlc_decomp_dir, MLC_ARTIFACTS_REVISION)
    boot_json_path = stage_mlc_bootstrap_json(
        Path(args.mlc_bootstrap_json),
        MLC_ARTIFACTS_REVISION,
        mlc_decomp_dir / "mlc_bootstrap_deltaskill.json",
    )
    committed_boot = load_json(boot_json_path)
    committed_decomp_by_regime = {r: load_committed_decomp(p) for r, p in decomp_paths.items()}
    mlc_manifest = load_json(mlc_store_dir / "manifest.json")
    model_name = args.model or mlc_manifest["model"]
    rung = mlc_manifest["rung"]
    max_new_tokens = int(mlc_manifest["max_new_tokens"])
    capture_layers = [int(x) for x in mlc_manifest["capture_layers"]]
    families = mlc_manifest["families"]
    mlc_ctx_set = set(mlc_manifest["context_ids"])
    missing_ctx = [c for c in ctx_ids if c not in mlc_ctx_set]
    if missing_ctx:
        raise RuntimeError(
            f"contexts {missing_ctx[:3]} absent from the staged MLC store manifest — "
            "the matched-length store must cover every run context"
        )
    if args.stop_after == "stage":
        phase("stopped_after_stage")
        return 0

    # ── pair-coherence asserts (fail-loud, pre-GPU) ───────────────────────────
    phase("asserts")
    probes = load_probe_pool()
    pool_hash = probes_hash(probes)
    if pool_hash != mlc_manifest["probe_pool_hash"]:
        raise RuntimeError(
            f"probe pool hash drift vs MLC manifest: {pool_hash} != "
            f"{mlc_manifest['probe_pool_hash']}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    completions_by_ctx: dict[str, list[tuple[str, str]]] = {}
    parse_by_ctx: dict[str, list[dict]] = {}
    mlc_kept_by_ctx: dict[str, list[int]] = {}
    assert_record: dict = {}
    floors_pin = {"k_min": MLC_K_MIN, "rem_min": MLC_REM_MIN}
    for c in ctx_ids:
        blob = json.loads((rollouts_dir / f"{c}.json").read_text(encoding="utf-8"))
        got = [r["probe"] for r in blob["completions"]]
        if got != probes:
            raise RuntimeError(f"rollout {c}.json probe list drift vs the loaded pool")
        completions = [
            (r["completion"], r.get("finish_reason", "stop")) for r in blob["completions"]
        ]
        mlc_blob = torch.load(mlc_store_dir / f"{c}.pt", weights_only=False)
        digest = rollout_content_digest(probes, completions)
        want = mlc_blob.get("rollout_digest")
        if digest != want:
            raise RuntimeError(
                f"rollout_digest mismatch for context {c}: recomputed {digest} != stored "
                f"{want!r} — the staged rollout text is NOT the text the matched-length store "
                "was captured from; refusing to run (plan §7 kill criterion)"
            )
        if mlc_blob.get("mlc_floors") != floors_pin:
            raise RuntimeError(
                f"MLC blob {c} floors {mlc_blob.get('mlc_floors')} != inherited pins "
                f"{floors_pin} — refusing a drifted kept-set convention"
            )
        parse_by_ctx[c] = parse_rows(tokenizer, completions, rung)
        completions_by_ctx[c] = completions
        mlc_kept_by_ctx[c] = [int(qi) for qi in mlc_blob["probe_indices"]]
        assert_record[c] = {
            "rollout_digest": want,
            "digest_match": True,
            "n_mlc_kept": len(mlc_kept_by_ctx[c]),
        }
        del mlc_blob  # stream: never hold all MLC blobs (RSS bound)
    logger.info("[phase=asserts] %d contexts digest-coherent with the MLC store", len(ctx_ids))
    gates: dict = {
        "followup_label": PMA_FOLLOWUP_LABEL,
        "rollouts_revision": ROLLOUTS_REVISION,
        "mlc_artifacts_revision": MLC_ARTIFACTS_REVISION,
        "contexts": ctx_ids,
        "full_grid": full_grid,
        "floors": floors_pin,
        "pair_coherence": assert_record,
        # The plan-v7 provenance contract (implementer-concern: state AND
        # persist which ctx/ans vectors feed pma_ctx_ans / the targets):
        "input_provenance": {
            "prefix_mean,query_mean": "recaptured THIS round (prefix_summaries store)",
            "ctx_mean (pma_ctx_ans input)": (
                f"STORE-JOINED from matched_length_summaries @ {MLC_ARTIFACTS_REVISION[:12]} — "
                "byte-identical reuse, zero recapture jitter"
            ),
            "cot_lastK_mean,ansprefix_K_mean,cot_mean (conditioning slices)": (
                f"STORE-JOINED @ {MLC_ARTIFACTS_REVISION[:12]}"
            ),
            "ans_rem_mean (remainder target), ans_mean (answer target + identity)": (
                f"STORE-JOINED @ {MLC_ARTIFACTS_REVISION[:12]}"
            ),
            "recaptured ctx/cot/ans means": "parity references ONLY — never fit inputs",
        },
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(gates, out_dir / "pma_capture_gates.json")
    if args.stop_after == "asserts":
        phase("stopped_after_asserts")
        return 0

    # ── span computation (CPU, tokenizer-only): floors + prefix/query spans ───
    phase("spans")
    instances = {i["id"]: i for i in battery["instances"]}
    rows_by_ctx: dict[str, list[dict]] = {}
    kept_qi_by_ctx: dict[str, list[int]] = {}
    bookkeeping: dict[str, list[dict]] = {}
    floor_drops: dict[str, int] = {}
    for c in ctx_ids:
        rows, kept_qi, books = [], [], []
        drop_reasons: dict[str, int] = {}
        for qi, (q, (text, _fr)) in enumerate(zip(probes, completions_by_ctx[c], strict=True)):
            rec = parse_by_ctx[c][qi]
            if not rec["well_formed"]:
                continue
            row, why = build_capture_row(
                tokenizer,
                instances[c],
                q,
                text,
                rec,
                rung,
                parts_spec=_mlc_parts,
                prompt_parts_spec=functools.partial(prefix_query_spans, probe=q),
            )
            if row is None:
                drop_reasons[why] = drop_reasons.get(why, 0) + 1
                continue
            rows.append(row)
            kept_qi.append(qi)
            books.append(pma_row_bookkeeping(row, qi))
        if kept_qi != mlc_kept_by_ctx[c]:
            raise RuntimeError(
                f"kept-set mismatch for context {c}: this round kept {len(kept_qi)} rows "
                f"{kept_qi[:5]}… != the MLC store's {len(mlc_kept_by_ctx[c])} rows "
                f"{mlc_kept_by_ctx[c][:5]}… (drop reasons this round: {drop_reasons}) — "
                "parser/floor/prompt-span drift; refusing to run (plan §7 kill criterion)"
            )
        rows_by_ctx[c] = rows
        kept_qi_by_ctx[c] = kept_qi
        bookkeeping[c] = books
        floor_drops[c] = drop_reasons.get("matched_length_floor", 0)
    n_post_floor = sum(len(v) for v in kept_qi_by_ctx.values())
    if full_grid and n_post_floor != 1991:
        raise RuntimeError(
            f"full-grid post-floor kept rows {n_post_floor} != the committed 1,991 (plan §4.2)"
        )
    logger.info(
        "[phase=spans] %d rows kept (== MLC store kept set; %d floor drops)",
        n_post_floor,
        sum(floor_drops.values()),
    )
    all_pfx = [b["len_prefix"] for books in bookkeeping.values() for b in books]
    all_qry = [b["len_query"] for books in bookkeeping.values() for b in books]
    gates.update(
        {
            "floor_drops_by_context": floor_drops,
            "n_kept_rows_post_floor": n_post_floor,
            "prefix_len_distribution": {
                "mean": float(np.mean(all_pfx)),
                "min": int(np.min(all_pfx)),
                "p50": float(np.percentile(all_pfx, 50)),
                "max": int(np.max(all_pfx)),
            },
            "query_len_distribution": {
                "mean": float(np.mean(all_qry)),
                "min": int(np.min(all_qry)),
                "p50": float(np.percentile(all_qry, 50)),
            },
            "row_bookkeeping": bookkeeping,
        }
    )
    dump_json(gates, out_dir / "pma_capture_gates.json")
    if args.stop_after == "spans":
        phase("stopped_after_spans")
        return 0

    # ── teacher-forced capture (GPU) → new flat store ─────────────────────────
    phase("capture")
    from transformers import AutoModelForCausalLM

    capture_device = "cuda" if torch.cuda.is_available() else "cpu"
    if capture_device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    n_layers = model.config.num_hidden_layers
    assert capture_layers == list(range(n_layers)), (capture_layers[:3], n_layers)
    hidden_size = int(model.config.hidden_size)
    capture = LayerCapture(model, n_layers)

    def _reusable_pma_blob(path: Path, c: str) -> tuple[dict | None, str]:
        """Entry-time skip-if-valid resume predicate for a new-store blob."""
        blob, why = reusable_store_blob(
            path,
            c,
            model_name=model_name,
            family=families[c],
            rung=rung,
            probe_pool_hash=pool_hash,
            capture_layers=capture_layers,
            summary_names=list(PMA_SUMMARY_NAMES),
            n_probes=len(probes),
            max_new_tokens=max_new_tokens,
            rollout_digest=rollout_content_digest(probes, completions_by_ctx[c]),
            hidden_size=hidden_size,
        )
        if blob is None:
            return None, why
        if blob.get("mlc_floors") != floors_pin:
            return None, "mlc_floors mismatch"
        if list(blob.get("probe_indices", [])) != kept_qi_by_ctx[c]:
            return None, "post-floor kept set mismatch"
        return blob, ""

    try:
        for ci, c in enumerate(ctx_ids):
            blob_path = store_dir / f"{c}.pt"
            if blob_path.is_file():
                prior, why = _reusable_pma_blob(blob_path, c)
                if prior is not None:
                    logger.info(
                        "[capture] %d/%d %s: SKIPPED (valid existing blob — resume)",
                        ci + 1,
                        len(ctx_ids),
                        c,
                    )
                    continue
                logger.warning("[capture] %s: existing blob invalid (%s) — recapturing", c, why)
            rows = rows_by_ctx[c]
            chunks: list[torch.Tensor] = []
            order: list[int] = []
            for batch_idx in pack_batches(rows, args.batch_probes, args.capture_token_budget):
                batch_rows = [rows[i] for i in batch_idx]
                chunks.append(
                    reduce_forward_batch(
                        model,
                        capture,
                        capture_layers,
                        tokenizer,
                        batch_rows,
                        summary_names=PMA_SUMMARY_NAMES,
                    )
                )
                order.extend(batch_idx)
            stacked = torch.cat(chunks, dim=0)  # (n_rows, 5, Lc, H) packed order
            inv = torch.empty(len(order), dtype=torch.long)
            inv[torch.tensor(order)] = torch.arange(len(order))
            per_q = stacked[inv]
            blob = {
                "context_id": c,
                "family": families[c],
                "rung": rung,
                "capture_layers": capture_layers,
                "summary_names": list(PMA_SUMMARY_NAMES),
                "probe_indices": kept_qi_by_ctx[c],
                "per_q": per_q,  # (n_rows, 5, Lc, H) fp16
                "probe_avg": per_q.float().mean(dim=0).to(torch.float16),
                "coverage": {
                    "n_probes_total": len(probes),
                    "n_well_formed": sum(1 for r in parse_by_ctx[c] if r["well_formed"]),
                    "n_captured": len(kept_qi_by_ctx[c]),
                    "capture_drop_reasons": {"matched_length_floor": floor_drops[c]},
                },
                "probe_pool_hash": pool_hash,
                "model": model_name,
                "max_new_tokens": max_new_tokens,
                "rollout_digest": rollout_content_digest(probes, completions_by_ctx[c]),
                "mlc_floors": floors_pin,
                "pma_row_bookkeeping": bookkeeping[c],
            }
            tmp = blob_path.with_suffix(".pt.tmp")
            torch.save(blob, tmp)
            os.replace(tmp, blob_path)
            logger.info(
                "[capture] %d/%d %s: %d rows captured", ci + 1, len(ctx_ids), c, per_q.shape[0]
            )
    finally:
        capture.remove()
    del model
    if capture_device == "cuda":
        torch.cuda.empty_cache()

    pma_manifest = {
        "context_ids": ctx_ids,
        "families": {c: families[c] for c in ctx_ids},
        "capture_layers": capture_layers,
        "summary_names": list(PMA_SUMMARY_NAMES),
        "hidden_size": int(mlc_manifest["hidden_size"]),
        "rung": rung,
        "probe_pool_hash": pool_hash,
        "n_probes": len(probes),
        "model": model_name,
        "max_new_tokens": max_new_tokens,
        "mlc_floors": floors_pin,
        "floor_drops_by_context": floor_drops,
        "rollouts_revision": ROLLOUTS_REVISION,
        "mlc_artifacts_revision": MLC_ARTIFACTS_REVISION,
        "reproducibility": reproducibility_metadata(),
        "full_grid": full_grid,
    }
    dump_json(pma_manifest, store_dir / "manifest.json")

    # ── capture-parity gate + prefix-constancy read (before ANY fit) ──────────
    phase("parity")
    parity_reports: dict = {}
    constancy: dict = {}
    pfx_idx = list(PMA_SUMMARY_NAMES).index("prefix_mean")
    for c in ctx_ids:
        new_blob = torch.load(store_dir / f"{c}.pt", weights_only=False)
        mlc_blob = torch.load(mlc_store_dir / f"{c}.pt", weights_only=False)
        parity_reports[c] = capture_parity_gate(new_blob, mlc_blob, list(mlc_blob["summary_names"]))
        # prefix per-context constancy (plan §12.9 — logged confirmation, no bar):
        v = new_blob["per_q"][:, pfx_idx].float()  # (n, Lc, H)
        if v.shape[0] > 1:
            mean_v = v.mean(dim=0, keepdim=True)
            cos = torch.nn.functional.cosine_similarity(
                v.flatten(1), mean_v.flatten(1).expand(v.shape[0], -1), dim=-1
            )
            constancy[c] = {"cos_min_to_mean": float(cos.min()), "n_rows": int(v.shape[0])}
        del new_blob, mlc_blob
    gates["capture_parity"] = {
        "bar": PARITY_COS_MIN,
        "cos_min_overall": min(r["cos_min_overall"] for r in parity_reports.values()),
        "by_context": parity_reports,
        "pass": True,
    }
    gates["prefix_constancy_read"] = {
        "note": "per-context cosine of each row's prefix_mean to the context mean (§12.9)",
        "cos_min_overall": (
            min(v["cos_min_to_mean"] for v in constancy.values()) if constancy else None
        ),
        "by_context": constancy,
    }
    dump_json(gates, out_dir / "pma_capture_gates.json")
    logger.info(
        "[phase=parity] PASS — min cosine %.6f over %d contexts (bar %.3f); prefix constancy "
        "min-cos-to-mean %s",
        gates["capture_parity"]["cos_min_overall"],
        len(ctx_ids),
        PARITY_COS_MIN,
        gates["prefix_constancy_read"]["cos_min_overall"],
    )

    # ── fit battery (both regimes, two targets) + nulls ───────────────────────
    phase("fit")
    if not args.skip_parity_gate:
        logger.info("[phase=fit] batched group-ridge vs serial reference (atol 1e-8)")
        ridge_parity = assert_group_ridge_matches_serial()
    else:
        ridge_parity = {"skipped": True}
    pma_store = Store(store_dir, blob_subdir=".")
    mlc_store = Store(mlc_store_dir, blob_subdir=".")
    if not full_grid:
        # subset run: the staged MLC store may carry a full-grid manifest —
        # narrow it to this run's ctx subset (cell lists derive from ctx_ids).
        for st in (pma_store, mlc_store):
            if st.ctx_ids != ctx_ids:
                raise RuntimeError(
                    f"store manifest ctx list {st.ctx_ids[:3]}… != run subset {ctx_ids[:3]}… — "
                    "restage or pass a store built for this subset"
                )
    store = JoinedStore(pma_store, mlc_store)
    layers_idx = args.layers if args.layers is not None else list(range(len(store.layers)))
    coherence_binding = {
        r: list(next(iter(d.values()))["ctx_order"]) == list(store.ctx_ids)
        for r, d in committed_decomp_by_regime.items()
    }
    if full_grid and not all(coherence_binding.values()):
        raise RuntimeError(
            "full-grid run but the committed decomp ctx_order differs from the store ctx list — "
            "the basis-coherence assert cannot bind (plan §7 kill criterion)"
        )
    grid_by_regime: dict = {}
    null_by_regime: dict = {}
    decomp_by_regime: dict = {}
    coherence_by_regime: dict = {}
    ckpt_root = out_dir / "partial"
    for regime in ("indiv", "avg_q"):
        regime_key = {
            "regime": regime,
            "round": PMA_FOLLOWUP_LABEL,
            "store_identity": store.identity_digest(),
            "layers": [int(store.layers[li]) for li in layers_idx],
            "arms": list(PMA_ARM_INPUTS),
            "n_perms": int(args.n_perms),
            "shuffle_null_seed": int(SHUFFLE_NULL_SEED),
            "standardization": "per_fold" if regime == "avg_q" else "full_data",
            "floors": floors_pin,
            "device": fit_device,
        }
        ckpt_dir = prepare_checkpoint_dir(ckpt_root, f"pma_{regime}", regime_key)
        grid, null_matrix, decomp, coherence = fit_pma_regime(
            store,
            regime,
            layers_idx,
            fit_device,
            args.n_perms,
            args.draw_chunk,
            committed_decomp_by_regime[regime],
            coherence_binding[regime],
            checkpoint_dir=ckpt_dir,
        )
        grid_by_regime[regime] = grid
        null_by_regime[regime] = null_matrix
        decomp_by_regime[regime] = decomp
        coherence_by_regime[regime] = {
            "binding": coherence_binding[regime],
            "rtol": BASIS_COHERENCE_RTOL,
            "max_rel_by_layer": {k: v["max_rel"] for k, v in coherence.items()},
        }
        # persist per-regime outputs the moment the regime completes.
        dump_json(
            {
                "dv": "recon_skill_over_mean_r2 (remainder + full-answer targets)",
                "regime": regime,
                "round": PMA_FOLLOWUP_LABEL,
                "axes": "arm -> layer -> [per-draw skill]",
                "n_perms": args.n_perms,
                "seed": SHUFFLE_NULL_SEED,
                "perm_grain": "context" if regime == "avg_q" else "context-group",
                "registered_arms": list(PMA_REGISTERED_ARMS),
                "null": null_matrix,
            },
            out_dir / f"null_matrix_{regime}_pma.json",
        )
        torch.save(
            {
                str(k): {"ss_res": v["ss_res"], "ss_tot": v["ss_tot"], "ctx_order": v["ctx_order"]}
                for k, v in decomp.items()
            },
            out_dir / f"decomp_{regime}_pma.pt",
        )

    # ── bootstrap contrasts + null-band analysis ──────────────────────────────
    phase("bootstrap")
    alignment = assert_committed_bootstrap_alignment(
        committed_boot, committed_decomp_by_regime, len(ctx_ids), args.n_boot, full_grid
    )
    boot_by_regime: dict = {}
    null_bands_by_regime: dict = {}
    coverage_by_regime: dict = {}
    for regime in ("indiv", "avg_q"):
        decomp = decomp_by_regime[regime]
        committed_decomp = committed_decomp_by_regime[regime]
        coverage_by_regime[regime] = assert_pma_pair_coverage(
            decomp, committed_decomp, len(ctx_ids)
        )
        boot_by_regime[regime] = pma_bootstrap_statistics(
            decomp,
            committed_decomp,
            regime,
            len(ctx_ids),
            args.n_boot,
            alignment,
            full_grid,
        )
        null_bands_by_regime[regime] = pma_null_band_analysis(
            null_by_regime[regime], decomp, committed_decomp
        )
    dump_json(
        {
            "dv": "paired bootstrap delta-skill (remainder + full-answer targets)",
            "round": PMA_FOLLOWUP_LABEL,
            "seed": BOOTSTRAP_SEED,
            "n_boot": args.n_boot,
            "committed_alignment": alignment,
            "registered_reads": [list(r) for r in PMA_REGISTERED_READS],
            "convention_reads": [list(r) for r in (*PMA_CONVENTION_READS, READ_5D)],
            "pair_row_coverage": {
                r: {k: v for k, v in cov.items() if k != "ctx_order"}
                for r, cov in coverage_by_regime.items()
            },
            "by_regime": boot_by_regime,
            "reproducibility": reproducibility_metadata(),
        },
        out_dir / "pma_bootstrap_deltaskill.json",
    )
    dump_json(
        {
            "dv": "held-out skill-over-mean R^2 per (arm x layer x regime x fold)",
            "round": PMA_FOLLOWUP_LABEL,
            "estimator": (
                "inherited #810/#928: LOCO ridge, nested-CV lambda over RIDGE_LAMBDAS, "
                "full-data PCA-48 target bases (remainder + full answer) with per-fold train "
                "centering; avg_q per-fold X standardization, indiv full-data X standardization"
            ),
            "context_ids": store.ctx_ids,
            "capture_layers": [int(store.layers[li]) for li in layers_idx],
            "n_indiv_rows": int(store.groups.shape[0]),
            "ridge_parity_gate": ridge_parity,
            "basis_coherence": coherence_by_regime,
            "arm_inputs": {a: list(r) for a, r in PMA_ARM_INPUTS.items()},
            "remainder_arms": list(PMA_REM_ARMS),
            "answer_arms": list(PMA_ANS_ARMS),
            "registered_arms": list(PMA_REGISTERED_ARMS),
            "exploratory_arms": list(PMA_EXPLORATORY_ARMS),
            "grid": grid_by_regime,
            "frozen_layers": {r: boot_by_regime[r]["layer_conventions"] for r in boot_by_regime},
            "null_band_vs_ceiling": null_bands_by_regime,
            "n_perms": args.n_perms,
            "n_boot": args.n_boot,
            "full_grid": full_grid,
            "reproducibility": reproducibility_metadata(),
        },
        out_dir / "pma_skill_grid.json",
    )
    read1 = boot_by_regime["indiv"]["statistics"]["read1_primary_pfx_cotK_minus_pfx_apfx"]
    logger.info(
        "[phase=bootstrap] read1 primary (indiv): obs=%.4f ci95=%s",
        read1["primary_frozen_pfx_baseline_best"]["observed"],
        read1["primary_frozen_pfx_baseline_best"]["ci95"],
    )

    # ── figures ───────────────────────────────────────────────────────────────
    phase("figures")
    fig_stems = make_pma_figures(
        Path(args.figures_dir),
        grid_by_regime,
        boot_by_regime,
        null_bands_by_regime,
        decomp_by_regime,
        committed_boot,
        bookkeeping,
    )

    # ── upload (one scoped-verified folder commit per artifact kind) ──────────
    hf_paths: dict = {}
    if not args.no_upload:
        phase("upload")
        suffix = "" if full_grid else "_smoke"
        hf_paths["store"] = upload_folder_scoped_verify(
            store_dir,
            PMA_STORE_HF_PREFIX + suffix,
            ["manifest.json", *(f"{c}.pt" for c in ctx_ids)],
            f"issue #928 {PMA_FOLLOWUP_LABEL}: prefix summary store ({len(ctx_ids)} contexts)",
        )
        json_names = sorted(p.name for p in out_dir.glob("*.json"))
        hf_paths["fit_results"] = upload_folder_scoped_verify(
            out_dir,
            PMA_RESULTS_PREFIX + suffix,
            json_names,
            f"issue #928 {PMA_FOLLOWUP_LABEL}: fit results",
            allow_patterns=["*.json"],
            ignore_patterns=["partial/*"],
        )
        pt_names = sorted(p.name for p in out_dir.glob("decomp_*_pma.pt"))
        hf_paths["decomp"] = upload_folder_scoped_verify(
            out_dir,
            f"{DECOMP_TENSORS_PREFIX}/prefix_mapping_arms" + suffix,
            pt_names,
            f"issue #928 {PMA_FOLLOWUP_LABEL}: per-context LOCO decompositions",
            allow_patterns=["decomp_*_pma.pt"],
        )
        fig_files = sorted(
            p.name
            for stem in fig_stems
            for p in Path(args.figures_dir).glob(f"{stem}.*")
            if p.suffix in (".png", ".pdf", ".json")
        )
        hf_paths["figures"] = upload_folder_scoped_verify(
            Path(args.figures_dir),
            f"{FIGURES_PREFIX}/prefix_mapping_arms" + suffix,
            fig_files,
            f"issue #928 {PMA_FOLLOWUP_LABEL}: figures",
            allow_patterns=[f"{stem}.*" for stem in fig_stems],
        )

    note = {
        "round": PMA_FOLLOWUP_LABEL,
        "contexts": len(ctx_ids),
        "full_grid": full_grid,
        "n_rows_post_floor": n_post_floor,
        "parity_cos_min": gates["capture_parity"]["cos_min_overall"],
        "read1_primary_indiv": read1["primary_frozen_pfx_baseline_best"],
        "resample_digest": boot_by_regime["indiv"]["resample_matrix"],
        "hf_paths": hf_paths,
        "elapsed_s": round(time.time() - t0, 1),
    }
    write_sentinel(
        "epm:results",
        note,
        out_dir,
        log_dir=Path(args.sentinel_dir) if args.sentinel_dir else None,
    )
    phase("done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] prefix-mapping arms crashed:\n%s", traceback.format_exc())
        raise
