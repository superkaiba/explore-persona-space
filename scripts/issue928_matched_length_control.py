#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ², Δ, ×, ≥) in scientific docstrings + log messages.
"""Issue #928 follow-up `matched-length-answer-span-control` (plan v6) — ONE driver.

Is the committed CoT conditioning gain (+0.203 per-question) SPECIFIC to the
CoT span, or reproduced by ANY matched-length realized span of the same
forward pass? Re-predicts the answer REMAINDER three ways — context alone,
context + K-token CoT slice, context + K-token answer-prefix slice — on the
parent's persisted rollouts, folds, and estimator (plan v6 §4).

Single code path, smoke = the SAME driver with ``--contexts 3`` (unification
default — every phase's cell list derives from the one ``--contexts`` subset).
Phases (linear; ``[phase=...]`` breadcrumbs feed the poller):

- **stage:** pinned-revision inputs — 50 rollout JSONs + the parent per-q
  summary store (manifest + blobs), through the h(iv)-FIXED
  ``store_local_relpath`` mapping (``issue928_mlp_indiv_control``; never
  ``snapshot_download`` on the ~1M-file repo — gotchas #833).
- **asserts (fail-loud, pre-GPU):** per context, recomputed
  ``rollout_content_digest`` == the parent blob's ``rollout_digest`` (the
  run's own item-(j) pair-coherence check) AND the parent's ``parse_rows``
  verbatim kept set == the blob's ``probe_indices`` (expected 1,994 total).
- **spans (CPU, tokenizer-only):** per kept row, ``matched_length_spans``
  (K = min(len(cot), len(ans)//2); floors K ≥ 8 ∧ remainder ≥ 16;
  dropped-and-counted per context, reason ``matched_length_floor``) + per-row
  K / span-length bookkeeping + cheap slice↔remainder lexical-overlap counts.
- **capture (GPU):** teacher-forced forwards via the DEFAULT-PRESERVING
  ``parts_spec``/``summary_names`` extension of ``build_capture_row`` /
  ``reduce_forward_batch`` — 7 mean-pool vectors per (row, layer)
  (``MLC_SUMMARY_NAMES``); new store ``matched_length_summaries/`` (flat:
  manifest + blobs in one folder, the parent's Hub layout).
- **parity (fail-loud, before any fit):** per (row, layer, part ∈
  {ctx, cot, ans}) cosine(recaptured mean, parent ``per_q`` mean) ≥ 0.999
  (the #779 span-mean bf16 calibration — measured headroom 0.999748; on a
  marginal miss, an fp32 re-probe attributes BEFORE any bar change).
- **fit:** 9 input-reps × layers × {LOCO-50, LOFO-7} × both regimes on the
  SHARED PCA-48 remainder target (batched ``GroupRidgeDesign`` machinery,
  serial-parity-gated); per-(regime, layer) checkpoint units (the #823 class).
- **nulls:** selection-symmetric per-draw × per-layer matrices for the 6
  registered arms (exploratory first-K arms EXCLUDED — plan §6).
- **bootstrap:** 5 registered paired contrasts × 3 layer conventions × 2
  regimes off the shared seed-42 resample matrix (digest recorded; parent
  bootstrap-metadata assert), with the paired-contrast row-coverage set-check.
- **figures / upload / done:** hero + forest + exploratory set; one
  ``upload_folder`` commit per artifact kind with scoped verify; ONE
  ``epm:results`` sentinel at true end-of-workload.

Usage::

    # production (GCP capture-7b lane, plan §10):
    EPM_FIT_DEVICE=cuda uv run python scripts/issue928_matched_length_control.py \\
        --out eval_results/issue_928/matched-length-answer-span-control

    # pod-side Phase-0 smoke (= the sweep at 3 contexts, scratch outputs):
    uv run python scripts/issue928_matched_length_control.py --contexts 3 \\
        --out /tmp/issue-928-mlc-smoke/eval --no-upload

    # VM CPU partial smoke (network + tokenizer only; stops before the GPU
    # capture phase — the pod smoke covers the full path):
    uv run python scripts/issue928_matched_length_control.py --contexts 2 \\
        --stop-after spans --out /tmp/issue-928-mlc-smoke/eval --no-upload
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))

import argparse
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
import torch.nn.functional as F  # noqa: E402

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
    HF_PREFIX_928,
    MLC_K_MIN,
    MLC_REM_MIN,
    MLC_SUMMARY_NAMES,
    RAW_COMPLETIONS_PREFIX,
    SHUFFLE_NULL_PERMS,
    SHUFFLE_NULL_SEED,
    STORE_PREFIX,
    context_order_and_families,
    dump_json,
    load_json,
    load_probe_pool,
    matched_length_spans,
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
from issue928_mlp_indiv_control import (  # noqa: E402
    STORE_REVISION,
    _hf_fetch_one,
    stage_store,
    store_local_relpath,
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

logger = logging.getLogger("issue928_mlc")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── round constants (plan v6 §5/§10) ─────────────────────────────────────────

MLC_STORE_HF_PREFIX = f"{HF_PREFIX_928}/analysis_tensors/store/matched_length_summaries"
MLC_RESULTS_PREFIX = f"{HF_PREFIX_928}/fit_results/matched_length_control"
FOLLOWUP_LABEL = "matched-length-answer-span-control"

# Capture-parity bar: the #779 span-mean bf16 calibration (span means smooth
# batched-kernel jitter — measured 0.999748 under a flat 0.999 bar; batch
# composition differs across runs, so bit-equality is NOT expected). A
# marginal miss gets an fp32 re-probe on 2-3 rows BEFORE any bar change
# (plan §8 — never loosen first).
PARITY_COS_MIN = 0.999

# Registered arms (plan §5/§6): absolute reads get selection-symmetric null
# bands; the exploratory first-K arms are captured + fit but EXCLUDED from
# the null battery (no registered absolute read).
MLC_REGISTERED_ARMS = (
    "mlc_ctx",
    "mlc_ctx_cotK",
    "mlc_ctx_apfx",
    "mlc_cotK",
    "mlc_apfx",
    "mlc_ctx_cotfull",
)
MLC_EXPLORATORY_ARMS = ("mlc_ctx_cotK_first", "mlc_cotK_first")
MLC_IDENT_ARM = "mlc_ident"
MLC_ALL_ARMS = (*MLC_REGISTERED_ARMS, *MLC_EXPLORATORY_ARMS, MLC_IDENT_ARM)
MLC_COMBO = "mean/mean"  # matched-length round is mean-pool only (plan §4.2)

# Input reps per arm (plan §4.3: 8 unique input reps + the identity ceiling).
MLC_ARM_INPUTS: dict[str, tuple[str, ...]] = {
    "mlc_ctx": ("ctx_mean",),
    "mlc_ctx_cotK": ("ctx_mean", "cot_lastK_mean"),
    "mlc_ctx_apfx": ("ctx_mean", "ansprefix_K_mean"),
    "mlc_cotK": ("cot_lastK_mean",),
    "mlc_apfx": ("ansprefix_K_mean",),
    "mlc_ctx_cotfull": ("ctx_mean", "cot_mean"),
    "mlc_ctx_cotK_first": ("ctx_mean", "cot_firstK_mean"),
    "mlc_cotK_first": ("cot_firstK_mean",),
    MLC_IDENT_ARM: ("ans_rem_mean",),
}

# Registered paired-bootstrap reads (plan §6, exact order; read 1 is PRIMARY).
MLC_REGISTERED_READS = (
    ("read1_primary_ctx_cotK_minus_ctx_apfx", "mlc_ctx_cotK", "mlc_ctx_apfx"),
    ("read2_ctx_cotK_minus_ctx", "mlc_ctx_cotK", "mlc_ctx"),
    ("read3_ctx_apfx_minus_ctx", "mlc_ctx_apfx", "mlc_ctx"),
    ("read4_cotK_alone_minus_apfx_alone", "mlc_cotK", "mlc_apfx"),
    ("read5_ctx_cotfull_minus_ctx_cotK", "mlc_ctx_cotfull", "mlc_ctx_cotK"),
)


def phase(name: str) -> None:
    """Poller-visible phase breadcrumb (one line per pipeline phase)."""
    logger.info("[phase=%s]", name)


# ── staging (pinned revision; h(iv)-fixed mapping) ────────────────────────────


def stage_rollouts(rollouts_dir: Path, ctx_ids: list[str], revision: str) -> None:
    """Local-first rollout JSONs; missing files fetch from the pinned HF revision.

    Content coherence with the parent store is enforced downstream by the
    fail-loud ``rollout_digest`` assert (a stale local copy cannot pass it).
    """
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in ctx_ids if not (rollouts_dir / f"{c}.json").is_file()]
    if not missing:
        logger.info("[phase=stage] rollouts already local (%d files) — skip", len(ctx_ids))
        return
    logger.info(
        "[phase=stage] fetching %d/%d rollout files @ %s", len(missing), len(ctx_ids), revision[:12]
    )
    for c in missing:
        _hf_fetch_one(f"{RAW_COMPLETIONS_PREFIX}/{c}.json", revision, rollouts_dir / f"{c}.json")


def stage_parent_store_subset(store_dir: Path, revision: str, ctx_ids: list[str]) -> None:
    """Stage ONLY the manifest + the subset's parent blobs, through the SAME
    h(iv)-fixed ``store_local_relpath`` mapping ``stage_store`` uses (direct
    known-path fetches — no listing needed for a named subset)."""
    pairs = [("percq_summaries/manifest.json", f"{STORE_PREFIX}/manifest.json")]
    pairs += [(f"percq_summaries/{c}.pt", f"{STORE_PREFIX}/{c}.pt") for c in ctx_ids]
    for hub_rel, full in pairs:
        rel = store_local_relpath(hub_rel)
        dest = store_dir / rel
        if dest.is_file():
            continue
        _hf_fetch_one(full, revision, dest)


# ── pair-coherence asserts (fail-loud, pre-GPU — plan §4.2 item 2) ────────────


def assert_pair_coherence(
    c: str,
    probes: list[str],
    completions: list[tuple[str, str]],
    parent_blob: dict,
    tokenizer,
    rung: str,
) -> list[dict]:
    """The run's own mechanical item-(j) check: staged rollout text must be the
    EXACT text the parent store was captured from, and the parent's verbatim
    ``parse_rows`` kept set must equal the blob's ``probe_indices``.

    Returns the parse records. Raises ``RuntimeError`` on ANY mismatch (kill
    criterion, plan §7 — aborts before GPU spend)."""
    digest = rollout_content_digest(probes, completions)
    want = parent_blob.get("rollout_digest")
    if digest != want:
        raise RuntimeError(
            f"rollout_digest mismatch for context {c}: recomputed {digest} != stored {want!r} — "
            "the staged rollout text is NOT the text the parent store was captured from; "
            "refusing to run (plan §7 kill criterion)"
        )
    parse = parse_rows(tokenizer, completions, rung)
    kept = [qi for qi, r in enumerate(parse) if r["well_formed"]]
    stored = [int(qi) for qi in parent_blob.get("probe_indices", [])]
    if kept != stored:
        raise RuntimeError(
            f"probe_indices mismatch for context {c}: parse_rows kept {len(kept)} rows "
            f"{kept[:5]}… != stored {len(stored)} rows {stored[:5]}… — parser/row-set drift; "
            "refusing to run (plan §7 kill criterion)"
        )
    return parse


# ── span computation + lexical-overlap bookkeeping (plan §4.2 item 3) ────────


def _mlc_parts(cot_tok: tuple[int, int], ans_tok: tuple[int, int]):
    """``parts_spec`` adapter: matched-length spans or the counted floor reason."""
    spans = matched_length_spans(cot_tok, ans_tok)
    if spans is None:
        return "matched_length_floor"
    return {k: v for k, v in spans.items() if k != "K"}


def _ngram_overlap_frac(slice_ids: list[int], rem_ids: list[int], n: int) -> float:
    """Fraction of the slice's token n-grams also present in the remainder.

    The alternatives-critic's free diagnostic (persist ONLY the cheap counts;
    no analysis is built on them this round)."""
    grams_a = {tuple(slice_ids[i : i + n]) for i in range(len(slice_ids) - n + 1)}
    if not grams_a:
        return 0.0
    grams_b = {tuple(rem_ids[i : i + n]) for i in range(len(rem_ids) - n + 1)}
    return len(grams_a & grams_b) / len(grams_a)


def row_bookkeeping(row: dict, probe_index: int) -> dict:
    """Per-row K / span-length + slice↔remainder lexical-overlap counts."""
    spans = row["spans"]
    ids = row["full_ids"].tolist()

    def seg(name: str) -> list[int]:
        s, e = spans[name]
        return ids[s:e]

    rem = seg("ans_rem")
    k = spans["cot_lastK"][1] - spans["cot_lastK"][0]
    book = {
        "probe_index": int(probe_index),
        "K": int(k),
        "len_cot": int(spans["cot"][1] - spans["cot"][0]),
        "len_ans": int(spans["ans"][1] - spans["ans"][0]),
        "len_rem": int(spans["ans_rem"][1] - spans["ans_rem"][0]),
    }
    for name in ("cot_lastK", "cot_firstK", "ansprefix_K"):
        s_ids = seg(name)
        book[f"{name}_tok_overlap_rem"] = round(_ngram_overlap_frac(s_ids, rem, 1), 4)
        book[f"{name}_4gram_overlap_rem"] = round(_ngram_overlap_frac(s_ids, rem, 4), 4)
    return book


# ── capture-parity gate (fail-loud, before any fit — plan §4.2 item 5) ────────


def capture_parity_gate(new_blob: dict, parent_blob: dict, parent_summary_names: list[str]) -> dict:
    """Per (row, layer, part ∈ {ctx, cot, ans}) cosine of the recaptured mean
    vs the parent store's ``per_q`` mean; min must clear ``PARITY_COS_MIN``.

    Returns the per-part report; raises ``RuntimeError`` on a gate failure
    (segmentation/alignment/batching drift — blocks production fits)."""
    parent_pos = {int(qi): i for i, qi in enumerate(parent_blob["probe_indices"])}
    rows_parent = [parent_pos[int(qi)] for qi in new_blob["probe_indices"]]
    new_sidx = {n: i for i, n in enumerate(new_blob["summary_names"])}
    par_sidx = {n: i for i, n in enumerate(parent_summary_names)}
    report: dict = {"n_rows": len(rows_parent), "parts": {}}
    worst = (1.0, None)
    for part in ("ctx", "cot", "ans"):
        name = f"{part}_mean"
        a = new_blob["per_q"][:, new_sidx[name]].float()  # (n, Lc, H)
        b = parent_blob["per_q"][rows_parent][:, par_sidx[name]].float()
        assert a.shape == b.shape, (a.shape, b.shape)
        cos = F.cosine_similarity(a.flatten(0, 1), b.flatten(0, 1), dim=-1)  # (n·Lc,)
        cmin = float(cos.min())
        amin = int(cos.argmin())
        n_layers = a.shape[1]
        cell = {"row": amin // n_layers, "layer_idx": amin % n_layers}
        report["parts"][part] = {
            "cos_min": cmin,
            "cos_mean": float(cos.mean()),
            "worst_cell": cell,
        }
        if cmin < worst[0]:
            worst = (cmin, (part, cell))
    report["cos_min_overall"] = worst[0]
    report["bar"] = PARITY_COS_MIN
    if worst[0] < PARITY_COS_MIN:
        raise RuntimeError(
            f"capture-parity gate FAILED for context {new_blob['context_id']}: "
            f"min cosine {worst[0]:.6f} < {PARITY_COS_MIN} at {worst[1]} — "
            "segmentation/alignment/batching drift; refusing to fit (plan §7 kill "
            "criterion; on a MARGINAL miss run an fp32 re-probe on 2-3 rows to "
            "attribute BEFORE any bar change — never loosen first)"
        )
    return report


# ── fit battery (both regimes, shared remainder target — plan §4.3) ───────────


def _merge_mlc_unit(unit: dict, grid: dict, null_matrix: dict, decomp: dict) -> None:
    """Fold one per-layer checkpoint unit into the regime accumulators."""
    for arm, schemes in unit["grid"].items():
        for scheme, cells in schemes.items():
            grid.setdefault(arm, {}).setdefault(scheme, []).extend(cells)
    for arm, by_layer in unit["null"].items():
        null_matrix.setdefault(arm, {}).update(by_layer)
    decomp.update(unit["decomp"])


def fit_mlc_regime(
    store: Store,
    regime: str,
    layers_idx: list[int],
    device: str,
    n_perms: int,
    draw_chunk: int,
    checkpoint_dir: Path | None = None,
) -> tuple[dict, dict, dict]:
    """All 9 MLC arms × layers × {LOCO, LOFO} for one regime, on the SHARED
    PCA-48 answer-REMAINDER target (+ selection-symmetric nulls for the 6
    registered arms).

    Inherited conventions (parent §4.3, caveats carried): full-data PCA basis
    with per-fold train centering; ``avg_q`` per-fold X standardization at
    n=n_ctx, ``indiv`` full-data X standardization (the shared-Gram basis);
    nested-CV λ over ``RIDGE_LAMBDAS``; fp64. ONE ``GroupRidgeDesign`` per
    (input-rep, scheme) per layer, freed per layer (the #823 sharing guard).
    Per-(regime, layer) checkpoint units + entry-time skip (restartability).
    Returns ``(grid, null_matrix, decomp)``; decomp keys ``(arm, combo,
    layer)`` → per-context (ss_res, ss_tot) LOCO decompositions.
    """
    n_ctx = len(store.ctx_ids)
    if regime == "avg_q":
        groups = np.arange(n_ctx)
        mat = store.avgq
        standardization = "per_fold"  # the EXACT inherited estimator at n=n_ctx
    else:
        groups = store.groups
        mat = store.indiv
        standardization = "full_data"  # the shared-Gram basis (parent indiv convention)
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
    input_reps = tuple(dict.fromkeys(r for reps in MLC_ARM_INPUTS.values() for r in reps))
    for li in layers_idx:
        layer = int(store.layers[li])
        unit_path = (checkpoint_dir / f"layer_{layer}.pt") if checkpoint_dir else None
        if unit_path is not None and unit_path.is_file():
            unit = torch.load(unit_path, weights_only=False)
            assert int(unit["layer"]) == layer, (unit["layer"], layer)
            _merge_mlc_unit(unit, grid, null_matrix, decomp)
            logger.info("[phase=fit] regime=%s layer %d SKIPPED (resumed unit)", regime, layer)
            continue
        t_layer = time.time()
        X_by_rep = {rep: mat(rep, li) for rep in input_reps}
        # the SHARED remainder target (never the full answer — plan §4.0):
        Y_pca, _mu, _comps = _pca_target(X_by_rep["ans_rem_mean"].copy())

        def arm_X(arm: str, _x=X_by_rep) -> np.ndarray:
            reps = MLC_ARM_INPUTS[arm]
            if len(reps) == 1:
                return _x[reps[0]]
            return np.concatenate([_x[r] for r in reps], axis=1)

        schemes = ["loco"] + (["lofo"] if folds_lofo is not None else [])
        grid_l: dict = {}
        null_l: dict = {}
        decomp_l: dict = {}
        designs: dict[tuple, GroupRidgeDesign] = {}
        for scheme in schemes:
            folds = folds_loco if scheme == "loco" else folds_lofo
            for arm in MLC_ALL_ARMS:
                dkey = (MLC_ARM_INPUTS[arm], scheme)
                if dkey not in designs:
                    designs[dkey] = GroupRidgeDesign(
                        arm_X(arm), folds, device=device, standardization=standardization
                    )
                des = designs[dkey]
                pred, _, _ = fit_predict_grouped(des, Y_pca)
                res = grouped_skill(pred, Y_pca, folds)
                grid_l.setdefault(arm, {}).setdefault(scheme, []).append(
                    {"layer": layer, "skill": res["skill"], "n": len(groups)}
                )
                if scheme == "loco":
                    decomp_l[(arm, MLC_COMBO, layer)] = {
                        "ss_res": np.asarray(res["ss_res_by_group"]),
                        "ss_tot": np.asarray(res["ss_tot_by_group"]),
                        "ctx_order": list(store.ctx_ids),
                    }
        if n_perms > 0:
            # registered arms ONLY (plan §6: exploratory first-K arms carry no
            # registered absolute read → excluded from the null battery).
            for arm in MLC_REGISTERED_ARMS:
                des = designs[(MLC_ARM_INPUTS[arm], "loco")]
                draws = grouped_null_skills_multi(des, [(Y_pca, None)], perm, draw_chunk=draw_chunk)
                null_l.setdefault(arm, {})[str(layer)] = draws[0]
        for d in designs.values():
            d.free()
        unit = {"layer": layer, "grid": grid_l, "null": null_l, "decomp": decomp_l}
        if unit_path is not None:
            _atomic_torch_save(unit, unit_path)
        _merge_mlc_unit(unit, grid, null_matrix, decomp)
        logger.info(
            "[phase=fit] regime=%s layer %d done in %.1fs%s",
            regime,
            layer,
            time.time() - t_layer,
            " (unit persisted)" if unit_path is not None else "",
        )
    return grid, null_matrix, decomp


# ── bootstrap contrasts (plan §6: 5 reads × 3 conventions × 2 regimes) ────────


def assert_parent_bootstrap_metadata(artifact_path: Path, n_groups: int, full_grid: bool) -> dict:
    """Assert this round's bootstrap constants equal the committed parent
    artifact's metadata (plan §6: the artifact carries NO matrix digest, so
    (seed, n_boot) equality is the check; this round records its OWN digest).

    ``n_groups == 50`` binds only on a full-grid run (a ``--contexts`` subset
    deliberately resamples its own N groups). Raises on any mismatch."""
    parent = load_json(artifact_path)
    if int(parent["seed"]) != BOOTSTRAP_SEED or int(parent["n_boot"]) != BOOTSTRAP_DRAWS:
        raise RuntimeError(
            f"parent bootstrap metadata mismatch: artifact (seed={parent['seed']}, "
            f"n_boot={parent['n_boot']}) != inherited constants (seed={BOOTSTRAP_SEED}, "
            f"n_boot={BOOTSTRAP_DRAWS}) — refusing an unpaired resample convention"
        )
    if full_grid and n_groups != 50:
        raise RuntimeError(
            f"full-grid run has n_groups={n_groups} != the parent's 50 contexts — "
            "the paired per-context bootstrap convention would not match"
        )
    return {
        "artifact": str(artifact_path),
        "seed": int(parent["seed"]),
        "n_boot": int(parent["n_boot"]),
        "n_groups_check": ("asserted == 50" if full_grid else f"subset run (n_groups={n_groups})"),
    }


def assert_pair_row_coverage(decomp: dict, n_ctx: int) -> dict:
    """Paired-contrast row-coverage set-check (plan §6, fail-loud): every
    registered (arm × layer) LOCO decomposition exists for BOTH arms of every
    registered read, over the IDENTICAL context order, before any contrast."""
    layer_sets = {}
    for arm in {a for _n, hi, lo in MLC_REGISTERED_READS for a in (hi, lo)} | {MLC_IDENT_ARM}:
        las = sorted(la for (a, c, la) in decomp if a == arm and c == MLC_COMBO)
        if not las:
            raise RuntimeError(f"pair-coverage set-check FAILED: no decomp rows for arm {arm!r}")
        layer_sets[arm] = las
    ref_arm = "mlc_ctx"
    ref_layers = sorted(la for (a, c, la) in decomp if a == ref_arm and c == MLC_COMBO)
    ref_order = decomp[(ref_arm, MLC_COMBO, ref_layers[0])]["ctx_order"]
    for name, hi, lo in MLC_REGISTERED_READS:
        for arm in (hi, lo):
            if layer_sets[arm] != ref_layers:
                raise RuntimeError(
                    f"pair-coverage set-check FAILED for {name}: arm {arm!r} layers "
                    f"{layer_sets[arm]} != baseline layers {ref_layers}"
                )
            for la in ref_layers:
                d = decomp[(arm, MLC_COMBO, la)]
                if len(d["ss_res"]) != n_ctx or d.get("ctx_order") != ref_order:
                    raise RuntimeError(
                        f"pair-coverage set-check FAILED for {name}: arm {arm!r} layer {la} "
                        f"has {len(d['ss_res'])} groups / drifted ctx_order (want {n_ctx})"
                    )
    return {"layers": ref_layers, "n_ctx": n_ctx, "ctx_order": ref_order, "pass": True}


def mlc_bootstrap_statistics(decomp: dict, n_ctx: int, n_boot: int) -> dict:
    """Paired-bootstrap Δskill CIs for the 5 registered reads at the three
    layer conventions (plan §6).

    PRIMARY (confirmatory): both arms at the CONTEXT-ONLY baseline's full-data
    best LOCO layer on the remainder target — frozen ONCE, before any draw
    (the parent's direct-arm-frozen convention adapted per the scope;
    selection-free for read 1, whose arms are both non-baseline). SECONDARY:
    own-best frozen_full_data (labeled data-selected) + per-replicate
    best-vs-best. ONE shared seed-42 resample matrix pairs every statistic.
    """
    idx = make_bootstrap_index_matrix(n_ctx, n_boot, BOOTSTRAP_SEED)
    matrix_digest = hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()[:16]

    def obs_skill(arm: str, layer: int) -> float:
        d = decomp[(arm, MLC_COMBO, layer)]
        tot = float(d["ss_tot"].sum())
        return float("nan") if tot < 1e-12 else 1.0 - float(d["ss_res"].sum()) / tot

    def layers_of(arm: str) -> list[int]:
        return sorted(la for (a, c, la) in decomp if a == arm and c == MLC_COMBO)

    def best_layer(arm: str) -> int:
        las = layers_of(arm)
        return int(las[int(np.nanargmax([obs_skill(arm, la) for la in las]))])

    def draws_for(arm: str, layer: int) -> np.ndarray:
        d = decomp[(arm, MLC_COMBO, layer)]
        return bootstrap_skills(d["ss_res"], d["ss_tot"], idx)

    def per_layer_draws(arm: str) -> np.ndarray:
        return np.stack([draws_for(arm, la) for la in layers_of(arm)], axis=1)  # (B, L)

    l_frozen = best_layer("mlc_ctx")
    out: dict = {
        "layer_conventions": {
            "primary_frozen_ctx_baseline_best_layer": l_frozen,
            "note": (
                "primary = frozen context-only baseline's full-data best LOCO layer on the "
                "remainder target, re-derived ONCE before any draw (plan §4.3/§6). Secondaries "
                "are frozen_full_data own-best (data-selected, labeled) and per-replicate "
                "best-vs-best."
            ),
        },
        "resample_matrix_digest": matrix_digest,
        "statistics": {},
    }
    for name, arm_hi, arm_lo in MLC_REGISTERED_READS:
        obs_p = obs_skill(arm_hi, l_frozen) - obs_skill(arm_lo, l_frozen)
        dr_p = draws_for(arm_hi, l_frozen) - draws_for(arm_lo, l_frozen)
        l_hi, l_lo = best_layer(arm_hi), best_layer(arm_lo)
        obs_ob = obs_skill(arm_hi, l_hi) - obs_skill(arm_lo, l_lo)
        dr_ob = draws_for(arm_hi, l_hi) - draws_for(arm_lo, l_lo)
        dr_bb = np.nanmax(per_layer_draws(arm_hi), axis=1) - np.nanmax(
            per_layer_draws(arm_lo), axis=1
        )
        out["statistics"][name] = {
            "arms": {"hi": arm_hi, "lo": arm_lo},
            "primary_frozen_ctx_baseline_best": stat_summary(obs_p, dr_p),
            "secondary_own_best_frozen_full_data": {
                "layers": {"hi": l_hi, "lo": l_lo},
                **stat_summary(obs_ob, dr_ob),
            },
            "secondary_best_vs_best_inherited": stat_summary(obs_ob, dr_bb),
        }
    # absolute per-arm reads at the frozen layer (bar-figure inputs).
    out["absolute_at_frozen"] = {
        arm: stat_summary(obs_skill(arm, l_frozen), draws_for(arm, l_frozen))
        for arm in (*MLC_REGISTERED_ARMS, *MLC_EXPLORATORY_ARMS, MLC_IDENT_ARM)
        if (arm, MLC_COMBO, l_frozen) in decomp
    }
    return out


def null_band_analysis(null_matrix: dict, decomp: dict) -> dict:
    """Selection-symmetric max-over-layers null bands + band-vs-ceiling report.

    Per registered arm: the per-draw max over the persisted per-layer null
    skills (same selection every draw); achievable ceiling = the regime's
    fresh remainder-target identity ceiling (max-over-layers observed LOCO
    skill of ``mlc_ident``). A band within one null-SE of the ceiling
    pre-commits that read to failure-to-reject narration (plan §6)."""
    ident_layers = sorted(la for (a, c, la) in decomp if a == MLC_IDENT_ARM and c == MLC_COMBO)

    def obs_skill(arm: str, layer: int) -> float:
        d = decomp[(arm, MLC_COMBO, layer)]
        tot = float(d["ss_tot"].sum())
        return float("nan") if tot < 1e-12 else 1.0 - float(d["ss_res"].sum()) / tot

    ceiling = float(np.nanmax([obs_skill(MLC_IDENT_ARM, la) for la in ident_layers]))
    out: dict = {"identity_ceiling_max_over_layers": ceiling, "arms": {}}
    for arm, by_layer in null_matrix.items():
        layers = sorted(by_layer, key=int)
        draws = np.asarray([by_layer[la] for la in layers], dtype=np.float64)  # (L, B)
        max_draws = np.nanmax(draws, axis=0)  # per-draw same-selection max
        band_hi = float(np.nanpercentile(max_draws, 97.5))
        se = float(np.nanstd(max_draws))
        obs_best = float(np.nanmax([obs_skill(arm, int(la)) for la in layers]))
        out["arms"][arm] = {
            "band_p2p5": float(np.nanpercentile(max_draws, 2.5)),
            "band_p97p5": band_hi,
            "null_se": se,
            "observed_best_over_layers": obs_best,
            "ceiling": ceiling,
            "uninformative_by_construction": bool(band_hi >= ceiling - se),
        }
    return out


# ── figures (hero + forest + exploratory over-produce — plan §6) ──────────────


def make_mlc_figures(
    figdir: Path,
    grid_by_regime: dict,
    boot_by_regime: dict,
    null_bands_by_regime: dict,
    decomp_by_regime: dict,
    bookkeeping: dict,
    floor_drops: dict,
) -> list[str]:
    """Hero bar/CI + (b)−(c) forest + per-layer curves + per-context scatter +
    sufficiency bars + floor-drop counts. Returns the written stems."""
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

    # 1) hero: per-question absolute skills at the frozen layer, with CIs.
    set_paper_style()
    boot = boot_by_regime["indiv"]
    hero_arms = ["mlc_ctx", "mlc_ctx_apfx", "mlc_ctx_cotK", "mlc_ctx_cotfull", MLC_IDENT_ARM]
    labels = {
        "mlc_ctx": "context\nonly",
        "mlc_ctx_apfx": "context +\nanswer prefix",
        "mlc_ctx_cotK": "context +\ntruncated CoT",
        "mlc_ctx_cotfull": "context +\nfull CoT",
        MLC_IDENT_ARM: "identity\nceiling",
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = paper_palette(len(hero_arms))
    absf = boot["absolute_at_frozen"]
    for i, arm in enumerate(hero_arms):
        st = absf[arm]
        lo, hi = st["ci95"]
        ax.bar(i, st["observed"], color=colors[i])
        ax.errorbar(
            i,
            st["observed"],
            yerr=[[max(0.0, st["observed"] - lo)], [max(0.0, hi - st["observed"])]],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    ax.set_xticks(range(len(hero_arms)), [labels[a] for a in hero_arms])
    lf = boot["layer_conventions"]["primary_frozen_ctx_baseline_best_layer"]
    ax.set_ylabel("held-out skill-over-mean R² (remainder target)")
    ax.set_title(f"Matched-length control — per-question, frozen layer {lf}")
    save(fig, "mlc_hero_bars_indiv")

    # 2) forest: read-1 Δ across regimes × conventions.
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    rows = []
    for regime in ("indiv", "avg_q"):
        st = boot_by_regime[regime]["statistics"]["read1_primary_ctx_cotK_minus_ctx_apfx"]
        rows.append((f"{regime} · primary frozen", st["primary_frozen_ctx_baseline_best"]))
        rows.append((f"{regime} · own-best frozen", st["secondary_own_best_frozen_full_data"]))
        rows.append((f"{regime} · best-vs-best", st["secondary_best_vs_best_inherited"]))
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
        ax.text(1.02, yi, label, transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_yticks([])
    ax.set_xlabel("Δskill (context+truncated-CoT − context+answer-prefix)")
    save(fig, "mlc_forest_read1")

    # 3) per-layer skill curves per regime (+ max-over-layers null band + ceiling).
    for regime, grid in grid_by_regime.items():
        set_paper_style()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        arms = [a for a in MLC_ALL_ARMS if a in grid]
        # paper_palette caps at 8 colors; 9 arms cycle with a dashed second lap.
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
        bands = null_bands_by_regime[regime]["arms"]
        if bands:
            hi = max(b["band_p97p5"] for b in bands.values())
            lo = min(b["band_p2p5"] for b in bands.values())
            ax.axhspan(lo, hi, color="gray", alpha=0.18, label="null band (max-over-layers)")
        ax.axhline(
            null_bands_by_regime[regime]["identity_ceiling_max_over_layers"],
            color="black",
            lw=0.9,
            ls=":",
            label="identity ceiling",
        )
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out skill (remainder target)")
        ax.set_title(f"Matched-length arms — {regime}")
        ax.legend(fontsize=7, ncols=2)
        save(fig, f"mlc_skill_curves_{regime}")

    # 4) per-context Δ(read 1) scatter vs median CoT length and vs median K.
    set_paper_style()
    boot = boot_by_regime["indiv"]
    lf = boot["layer_conventions"]["primary_frozen_ctx_baseline_best_layer"]
    decomp = decomp_by_regime["indiv"]
    d_hi = decomp[("mlc_ctx_cotK", MLC_COMBO, lf)]
    d_lo = decomp[("mlc_ctx_apfx", MLC_COMBO, lf)]
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
            ax.scatter(xs.get(c, float("nan")), delta_c[ci], s=14, color="tab:blue")
            ax.annotate(c, (xs.get(c, float("nan")), delta_c[ci]), fontsize=5, alpha=0.7)
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(f"per-context Δskill (read 1) @ L{lf}")
    save(fig, "mlc_percontext_delta_scatter")

    # 5) sufficiency-analogue bars + per-context floor drops.
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    suff = ["mlc_cotK", "mlc_cotK_first", "mlc_apfx"]
    absf = boot["absolute_at_frozen"]
    present = [a for a in suff if a in absf]
    colors = paper_palette(len(present))
    for i, arm in enumerate(present):
        st = absf[arm]
        lo, hi = st["ci95"]
        axes[0].bar(i, st["observed"], color=colors[i])
        axes[0].errorbar(
            i,
            st["observed"],
            yerr=[[max(0.0, st["observed"] - lo)], [max(0.0, hi - st["observed"])]],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    axes[0].set_xticks(range(len(present)), present, fontsize=7)
    axes[0].set_ylabel("held-out skill (remainder target)")
    axes[0].set_title(f"Sufficiency analogues @ L{lf} (indiv)")
    ctxs = list(floor_drops)
    axes[1].bar(range(len(ctxs)), [floor_drops[c] for c in ctxs], color="tab:orange")
    axes[1].set_xticks(range(len(ctxs)), ctxs, rotation=90, fontsize=5)
    axes[1].set_ylabel("rows dropped (matched_length_floor)")
    axes[1].set_title("Per-context floor drops")
    save(fig, "mlc_sufficiency_and_floor_drops")
    return stems


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:  # noqa: C901 — linear phase pipeline; see phase() markers
    ap = argparse.ArgumentParser(
        description="Issue #928 matched-length answer-span control (follow-up plan v6)"
    )
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval_results" / "issue_928" / FOLLOWUP_LABEL),
    )
    ap.add_argument(
        "--store-dir",
        default=str(PROJECT_ROOT / "data" / "issue_928" / "store" / "matched_length_summaries"),
    )
    ap.add_argument("--parent-store", default=str(PROJECT_ROOT / "data" / "issue_928" / "store"))
    ap.add_argument(
        "--rollouts",
        default=str(PROJECT_ROOT / "data" / "issue_928" / "raw_completions" / "thinking_rollouts"),
    )
    ap.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures" / "issue_928"))
    ap.add_argument(
        "--contexts", type=int, default=None, help="cap contexts (pod Phase-0 smoke = 3)"
    )
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="layer-INDEX subset")
    ap.add_argument("--model", default=None, help="override the parent manifest's model")
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
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_dir = Path(args.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    parent_store_dir = Path(args.parent_store)
    rollouts_dir = Path(args.rollouts)
    fit_device = _resolve_device(_requested_device(args.device))
    logger.info("fit device: %s", fit_device)
    t0 = time.time()

    # ── stage (pinned revision; every phase's cell list derives from ctx_ids) ─
    phase("stage")
    battery = resolve_battery(None)
    ctx_ids_all, _families_battery = context_order_and_families(battery)
    ctx_ids = ctx_ids_all[: args.contexts] if args.contexts else ctx_ids_all
    full_grid = len(ctx_ids) == len(ctx_ids_all)
    logger.info("contexts=%d (full_grid=%s)", len(ctx_ids), full_grid)
    stage_rollouts(rollouts_dir, ctx_ids, STORE_REVISION)
    if full_grid:
        stage_store(parent_store_dir, STORE_REVISION)
    else:
        stage_parent_store_subset(parent_store_dir, STORE_REVISION, ctx_ids)
    parent_manifest = load_json(parent_store_dir / "manifest.json")
    model_name = args.model or parent_manifest["model"]
    rung = parent_manifest["rung"]
    max_new_tokens = int(parent_manifest["max_new_tokens"])
    capture_layers = [int(x) for x in parent_manifest["capture_layers"]]
    families = parent_manifest["families"]
    if args.stop_after == "stage":
        phase("stopped_after_stage")
        return 0

    # ── pair-coherence asserts (fail-loud, pre-GPU) ───────────────────────────
    phase("asserts")

    probes = load_probe_pool()
    pool_hash = probes_hash(probes)
    if pool_hash != parent_manifest["probe_pool_hash"]:
        raise RuntimeError(
            f"probe pool hash drift vs parent manifest: {pool_hash} != "
            f"{parent_manifest['probe_pool_hash']}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    completions_by_ctx: dict[str, list[tuple[str, str]]] = {}
    parse_by_ctx: dict[str, list[dict]] = {}
    assert_record: dict = {}
    n_kept_total = 0
    for c in ctx_ids:
        blob = json.loads((rollouts_dir / f"{c}.json").read_text(encoding="utf-8"))
        got = [r["probe"] for r in blob["completions"]]
        if got != probes:
            raise RuntimeError(f"rollout {c}.json probe list drift vs the loaded pool")
        completions = [
            (r["completion"], r.get("finish_reason", "stop")) for r in blob["completions"]
        ]
        parent_blob = torch.load(
            parent_store_dir / "percq_summaries" / f"{c}.pt", weights_only=False
        )
        parse = assert_pair_coherence(c, probes, completions, parent_blob, tokenizer, rung)
        completions_by_ctx[c] = completions
        parse_by_ctx[c] = parse
        n_kept = sum(1 for r in parse if r["well_formed"])
        n_kept_total += n_kept
        assert_record[c] = {
            "rollout_digest": parent_blob["rollout_digest"],
            "n_kept": n_kept,
            "digest_match": True,
            "probe_indices_match": True,
        }
        del parent_blob  # stream: never hold all parent blobs (RSS bound)
    logger.info(
        "[phase=asserts] %d contexts coherent; %d kept rows total", len(ctx_ids), n_kept_total
    )
    gates: dict = {
        "followup_label": FOLLOWUP_LABEL,
        "revision": STORE_REVISION,
        "contexts": ctx_ids,
        "full_grid": full_grid,
        "floors": {"k_min": MLC_K_MIN, "rem_min": MLC_REM_MIN},
        "pair_coherence": assert_record,
        "n_kept_rows_pre_floor": n_kept_total,
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(gates, out_dir / "mlc_capture_gates.json")
    if args.stop_after == "asserts":
        phase("stopped_after_asserts")
        return 0

    # ── span computation (CPU, tokenizer-only) + bookkeeping ─────────────────
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
                tokenizer, instances[c], q, text, rec, rung, parts_spec=_mlc_parts
            )
            if row is None:
                drop_reasons[why] = drop_reasons.get(why, 0) + 1
                continue
            rows.append(row)
            kept_qi.append(qi)
            books.append(row_bookkeeping(row, qi))
        non_floor = {k: v for k, v in drop_reasons.items() if k != "matched_length_floor"}
        if non_floor:
            # a structural drop on a row the parent store KEPT is capture drift,
            # not a floor outcome — fail loud (plan §7 kill criterion).
            raise RuntimeError(
                f"context {c}: non-floor capture drops on parent-kept rows: {non_floor}"
            )
        if not rows:
            raise RuntimeError(f"context {c}: zero rows survive the matched-length floor")
        rows_by_ctx[c] = rows
        kept_qi_by_ctx[c] = kept_qi
        bookkeeping[c] = books
        floor_drops[c] = drop_reasons.get("matched_length_floor", 0)
    n_post_floor = sum(len(v) for v in kept_qi_by_ctx.values())
    logger.info(
        "[phase=spans] %d/%d rows kept post-floor (%d floor drops)",
        n_post_floor,
        n_kept_total,
        sum(floor_drops.values()),
    )
    all_k = [b["K"] for books in bookkeeping.values() for b in books]
    all_rem = [b["len_rem"] for books in bookkeeping.values() for b in books]
    gates.update(
        {
            "floor_drops_by_context": floor_drops,
            "n_kept_rows_post_floor": n_post_floor,
            "k_distribution": {
                "mean": float(np.mean(all_k)),
                "p5": float(np.percentile(all_k, 5)),
                "p50": float(np.percentile(all_k, 50)),
                "p95": float(np.percentile(all_k, 95)),
            },
            "rem_distribution": {
                "mean": float(np.mean(all_rem)),
                "p5": float(np.percentile(all_rem, 5)),
                "min": int(np.min(all_rem)),
            },
            "row_bookkeeping": bookkeeping,
        }
    )
    dump_json(gates, out_dir / "mlc_capture_gates.json")
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

    def _reusable_mlc_blob(path: Path, c: str) -> tuple[dict | None, str]:
        """Entry-time skip-if-valid resume predicate for a new-store blob —
        the parent ``reusable_store_blob`` contract + the MLC floor pins."""

        blob, why = reusable_store_blob(
            path,
            c,
            model_name=model_name,
            family=families[c],
            rung=rung,
            probe_pool_hash=pool_hash,
            capture_layers=capture_layers,
            summary_names=list(MLC_SUMMARY_NAMES),
            n_probes=len(probes),
            max_new_tokens=max_new_tokens,
            rollout_digest=rollout_content_digest(probes, completions_by_ctx[c]),
            hidden_size=hidden_size,
        )
        if blob is None:
            return None, why
        if blob.get("mlc_floors") != {"k_min": MLC_K_MIN, "rem_min": MLC_REM_MIN}:
            return None, "mlc_floors mismatch"
        if list(blob.get("probe_indices", [])) != kept_qi_by_ctx[c]:
            return None, "post-floor kept set mismatch"
        return blob, ""

    try:
        for ci, c in enumerate(ctx_ids):
            blob_path = store_dir / f"{c}.pt"
            if blob_path.is_file():
                prior, why = _reusable_mlc_blob(blob_path, c)
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
                        summary_names=MLC_SUMMARY_NAMES,
                    )
                )
                order.extend(batch_idx)
            stacked = torch.cat(chunks, dim=0)  # (n_rows, 7, Lc, H) packed order
            inv = torch.empty(len(order), dtype=torch.long)
            inv[torch.tensor(order)] = torch.arange(len(order))
            per_q = stacked[inv]
            blob = {
                "context_id": c,
                "family": families[c],
                "rung": rung,
                "capture_layers": capture_layers,
                "summary_names": list(MLC_SUMMARY_NAMES),
                "probe_indices": kept_qi_by_ctx[c],
                "per_q": per_q,  # (n_rows, 7, Lc, H) fp16
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
                "mlc_floors": {"k_min": MLC_K_MIN, "rem_min": MLC_REM_MIN},
                "mlc_row_bookkeeping": bookkeeping[c],
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

    mlc_manifest = {
        "context_ids": ctx_ids,
        "families": {c: families[c] for c in ctx_ids},
        "capture_layers": capture_layers,
        "summary_names": list(MLC_SUMMARY_NAMES),
        "hidden_size": int(parent_manifest["hidden_size"]),
        "rung": rung,
        "probe_pool_hash": pool_hash,
        "n_probes": len(probes),
        "model": model_name,
        "max_new_tokens": max_new_tokens,
        "mlc_floors": {"k_min": MLC_K_MIN, "rem_min": MLC_REM_MIN},
        "floor_drops_by_context": floor_drops,
        "parent_store_revision": STORE_REVISION,
        "reproducibility": reproducibility_metadata(),
        "full_grid": full_grid,
    }
    dump_json(mlc_manifest, store_dir / "manifest.json")

    # ── capture-parity gate (before ANY fit) ──────────────────────────────────
    phase("parity")
    parity_reports: dict = {}
    for c in ctx_ids:
        new_blob = torch.load(store_dir / f"{c}.pt", weights_only=False)
        parent_blob = torch.load(
            parent_store_dir / "percq_summaries" / f"{c}.pt", weights_only=False
        )
        parity_reports[c] = capture_parity_gate(
            new_blob, parent_blob, list(parent_blob["summary_names"])
        )
        del new_blob, parent_blob
    gates["capture_parity"] = {
        "bar": PARITY_COS_MIN,
        "cos_min_overall": min(r["cos_min_overall"] for r in parity_reports.values()),
        "by_context": parity_reports,
        "pass": True,
    }
    dump_json(gates, out_dir / "mlc_capture_gates.json")
    logger.info(
        "[phase=parity] PASS — min cosine %.6f over %d contexts (bar %.3f)",
        gates["capture_parity"]["cos_min_overall"],
        len(ctx_ids),
        PARITY_COS_MIN,
    )

    # ── fit battery (both regimes) + nulls ────────────────────────────────────
    phase("fit")
    if not args.skip_parity_gate:
        logger.info("[phase=fit] batched group-ridge vs serial reference (atol 1e-8)")
        ridge_parity = assert_group_ridge_matches_serial()
    else:
        ridge_parity = {"skipped": True}
    store = Store(store_dir, blob_subdir=".")
    layers_idx = args.layers if args.layers is not None else list(range(len(store.layers)))
    grid_by_regime: dict = {}
    null_by_regime: dict = {}
    decomp_by_regime: dict = {}
    ckpt_root = out_dir / "partial"
    for regime in ("indiv", "avg_q"):
        regime_key = {
            "regime": regime,
            "round": FOLLOWUP_LABEL,
            "store_identity": store.identity_digest(),
            "layers": [int(store.layers[li]) for li in layers_idx],
            "arms": list(MLC_ALL_ARMS),
            "n_perms": int(args.n_perms),
            "shuffle_null_seed": int(SHUFFLE_NULL_SEED),
            "standardization": "per_fold" if regime == "avg_q" else "full_data",
            "floors": {"k_min": MLC_K_MIN, "rem_min": MLC_REM_MIN},
            "device": fit_device,
        }
        ckpt_dir = prepare_checkpoint_dir(ckpt_root, f"mlc_{regime}", regime_key)
        grid, null_matrix, decomp = fit_mlc_regime(
            store,
            regime,
            layers_idx,
            fit_device,
            args.n_perms,
            args.draw_chunk,
            checkpoint_dir=ckpt_dir,
        )
        grid_by_regime[regime] = grid
        null_by_regime[regime] = null_matrix
        decomp_by_regime[regime] = decomp
        # persist per-regime outputs the moment the regime completes.
        dump_json(
            {
                "dv": "recon_skill_over_mean_r2 (answer-REMAINDER target)",
                "regime": regime,
                "round": FOLLOWUP_LABEL,
                "axes": "arm -> layer -> [per-draw skill]",
                "n_perms": args.n_perms,
                "seed": SHUFFLE_NULL_SEED,
                "perm_grain": "context" if regime == "avg_q" else "context-group",
                "registered_arms": list(MLC_REGISTERED_ARMS),
                "null": null_matrix,
            },
            out_dir / f"null_matrix_{regime}_mlc.json",
        )
        torch.save(
            {
                str(k): {"ss_res": v["ss_res"], "ss_tot": v["ss_tot"], "ctx_order": v["ctx_order"]}
                for k, v in decomp.items()
            },
            out_dir / f"decomp_{regime}_mlc.pt",
        )

    # ── bootstrap contrasts + null-band analysis ──────────────────────────────
    phase("bootstrap")
    parent_meta = assert_parent_bootstrap_metadata(
        PROJECT_ROOT / "eval_results" / "issue_928" / "bootstrap_deltaskill.json",
        len(ctx_ids),
        full_grid,
    )
    boot_by_regime: dict = {}
    null_bands_by_regime: dict = {}
    coverage_by_regime: dict = {}
    for regime in ("indiv", "avg_q"):
        decomp = decomp_by_regime[regime]
        coverage_by_regime[regime] = assert_pair_row_coverage(decomp, len(ctx_ids))
        boot_by_regime[regime] = mlc_bootstrap_statistics(decomp, len(ctx_ids), args.n_boot)
        null_bands_by_regime[regime] = null_band_analysis(null_by_regime[regime], decomp)
    dump_json(
        {
            "dv": "paired bootstrap delta-skill on the answer-REMAINDER target",
            "round": FOLLOWUP_LABEL,
            "seed": BOOTSTRAP_SEED,
            "n_boot": args.n_boot,
            "parent_metadata_assert": parent_meta,
            "registered_reads": [list(r) for r in MLC_REGISTERED_READS],
            "pair_row_coverage": {
                r: {k: v for k, v in cov.items() if k != "ctx_order"}
                for r, cov in coverage_by_regime.items()
            },
            "by_regime": boot_by_regime,
            "reproducibility": reproducibility_metadata(),
        },
        out_dir / "mlc_bootstrap_deltaskill.json",
    )
    dump_json(
        {
            "dv": "held-out skill-over-mean R^2 per (arm x layer x regime x fold), rem target",
            "round": FOLLOWUP_LABEL,
            "estimator": (
                "inherited #810/#928: LOCO ridge, nested-CV lambda over RIDGE_LAMBDAS, "
                "full-data PCA-48 remainder-target basis with per-fold train centering; "
                "avg_q per-fold X standardization, indiv full-data X standardization"
            ),
            "context_ids": store.ctx_ids,
            "capture_layers": [int(store.layers[li]) for li in layers_idx],
            "n_indiv_rows": int(store.groups.shape[0]),
            "ridge_parity_gate": ridge_parity,
            "arm_inputs": {a: list(r) for a, r in MLC_ARM_INPUTS.items()},
            "registered_arms": list(MLC_REGISTERED_ARMS),
            "exploratory_arms": list(MLC_EXPLORATORY_ARMS),
            "grid": grid_by_regime,
            "frozen_layers": {r: boot_by_regime[r]["layer_conventions"] for r in boot_by_regime},
            "null_band_vs_ceiling": null_bands_by_regime,
            "n_perms": args.n_perms,
            "n_boot": args.n_boot,
            "full_grid": full_grid,
            "reproducibility": reproducibility_metadata(),
        },
        out_dir / "mlc_skill_grid.json",
    )
    read1 = boot_by_regime["indiv"]["statistics"]["read1_primary_ctx_cotK_minus_ctx_apfx"]
    logger.info(
        "[phase=bootstrap] read1 primary (indiv): obs=%.4f ci95=%s",
        read1["primary_frozen_ctx_baseline_best"]["observed"],
        read1["primary_frozen_ctx_baseline_best"]["ci95"],
    )

    # ── figures ───────────────────────────────────────────────────────────────
    phase("figures")
    fig_stems = make_mlc_figures(
        Path(args.figures_dir),
        grid_by_regime,
        boot_by_regime,
        null_bands_by_regime,
        decomp_by_regime,
        bookkeeping,
        floor_drops,
    )

    # ── upload (one scoped-verified folder commit per artifact kind) ──────────
    hf_paths: dict = {}
    if not args.no_upload:
        phase("upload")
        suffix = "" if full_grid else "_smoke"
        hf_paths["store"] = upload_folder_scoped_verify(
            store_dir,
            MLC_STORE_HF_PREFIX + suffix,
            ["manifest.json", *(f"{c}.pt" for c in ctx_ids)],
            f"issue #928 {FOLLOWUP_LABEL}: matched-length summary store ({len(ctx_ids)} contexts)",
        )
        json_names = sorted(p.name for p in out_dir.glob("*.json"))
        hf_paths["fit_results"] = upload_folder_scoped_verify(
            out_dir,
            MLC_RESULTS_PREFIX + suffix,
            json_names,
            f"issue #928 {FOLLOWUP_LABEL}: fit results",
            allow_patterns=["*.json"],
            ignore_patterns=["partial/*"],
        )
        pt_names = sorted(p.name for p in out_dir.glob("decomp_*_mlc.pt"))
        hf_paths["decomp"] = upload_folder_scoped_verify(
            out_dir,
            f"{DECOMP_TENSORS_PREFIX}/matched_length_control" + suffix,
            pt_names,
            f"issue #928 {FOLLOWUP_LABEL}: per-context LOCO decompositions",
            allow_patterns=["decomp_*_mlc.pt"],
        )
        fig_files = sorted(
            p.name
            for stem in fig_stems
            for p in Path(args.figures_dir).glob(f"{stem}.*")
            if p.suffix in (".png", ".pdf", ".json")
        )
        hf_paths["figures"] = upload_folder_scoped_verify(
            Path(args.figures_dir),
            f"{FIGURES_PREFIX}/matched_length_control" + suffix,
            fig_files,
            f"issue #928 {FOLLOWUP_LABEL}: figures",
            allow_patterns=[f"{stem}.*" for stem in fig_stems],
        )

    note = {
        "round": FOLLOWUP_LABEL,
        "contexts": len(ctx_ids),
        "full_grid": full_grid,
        "n_rows_post_floor": n_post_floor,
        "floor_drops": sum(floor_drops.values()),
        "parity_cos_min": gates["capture_parity"]["cos_min_overall"],
        "read1_primary_indiv": read1["primary_frozen_ctx_baseline_best"],
        "hf_paths": hf_paths,
        "elapsed_s": round(time.time() - t0, 1),
    }
    write_sentinel("epm:results", note, out_dir)
    phase("done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] matched-length control crashed:\n%s", traceback.format_exc())
        raise
