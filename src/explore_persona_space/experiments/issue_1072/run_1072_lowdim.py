"""Issue #1072 amendment round ``lowdim-token-subspace`` — pod-side driver.

Does a LOW-DIMENSIONAL token subspace rescue the falsified 1-D
output-token-commitment account? Re-runs the completed round's component
decomposition with a 3-cell basis panel {top-8 candidates, top-32 candidates,
realized lookahead-8} at layers {14, 26} (plan §4.3; plans/v4.md).

Phases (smoke IS the run at tiny n — same dispatcher, same phases, same
single-process shape; smoke narrows ONLY the pool (10 synth contexts), the
model (from-config tiny Qwen2 over the REAL vocab) and the layer set (0,1)):

  A ``stage``    — production: #952 phase0_verify replay + scoped staging of
                   the 8 parent slot shards + 4 span files (@ the #952 pin),
                   the 8 completed-round decomp shards + 4 next-id npz + the 5
                   reproduction npz (@ the completed-round store pin). Smoke:
                   the PARENT PRODUCERS run at tiny n (parent1072
                   stage/capture/battery) so every production consumer
                   interface — staged store, committed-format reproduction
                   npz, battery-JSON frozen-λ reads — exists as real
                   producer-generated files.
  B ``capture``  — one teacher-forced forward per (context, arm), hooked at
                   the decomposition layers + the FINAL decoder layer (logits
                   read): per-position top-32 candidate ids/logprobs + the
                   3-basis subspace remainder accumulators, with in-stream
                   gates g1 (spans + next-id identity), g2 (slot z cos), g3'
                   (rem_nx cos vs the staged decomp store), g6 (1-D alpha parity),
                   g7 (QR rank guard) and the registered coverage counters.
                   B': store uploaded to HF BEFORE phase C (per-arm
                   incremental checkpoint uploads as each arm completes).
  C ``battery``  — per (fold, layer): subspace component cells at FROZEN λ*
                   via ``run_component_cell`` with a basis-extended pair_fn
                   (slot-basis batched QR; g4' full-target reproduction vs the
                   committed per-context npz, K3; g5 fold hashes, K5; fp64
                   serial-oracle parity; per-unit checkpoints; K4 fold-4
                   pilot).

Phases D (stats) / E (figures) are VM-side: ``scripts/issue1072_lowdim_stats.py``
+ ``scripts/issue1072_lowdim_figures.py``.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE any heavy import — shared-VM thread caps freeze at torch import

import numpy as np  # noqa: E402

from explore_persona_space.experiments.issue_952 import run_952 as parent952  # noqa: E402
from explore_persona_space.experiments.issue_1072 import run_1072 as parent1072  # noqa: E402
from explore_persona_space.experiments.issue_1072 import subspace_basis as sb  # noqa: E402
from explore_persona_space.experiments.issue_1072.component_ridge import (  # noqa: E402
    component_parity_gate,
    run_component_cell,
    serial_component_reference,
)

logger = logging.getLogger("issue1072.lowdim")

ISSUE = 1072
FOLLOWUP_LABEL = "lowdim-token-subspace"
ISSUE_SLUG = "issue1072_lowdim_token_subspace"
HF_DATA_REPO = parent952.HF_DATA_REPO
PARENT_SLUG = parent952.ISSUE_SLUG  # issue952_position_divergence
DECOMP_SLUG = parent1072.ISSUE_SLUG  # issue1072_component_decomposition
# Pinned revisions (plan §10 Reproducibility Card).
PARENT_TENSOR_REVISION = parent1072.PARENT_TENSOR_REVISION  # #952 slot shards + spans
DECOMP_STORE_REVISION = "9c4258b242ad89dfa66cad18ce09d74fb5c357ad"
DEFAULT_MODEL = parent952.DEFAULT_MODEL
PROD_LAYERS = (14, 26)  # followup-scope marker: the trend endpoints (plan §3.5(1))
BASES = ("top8", "top32", "look8")
K_FOLDS = parent1072.K_FOLDS
CAL_FOLD = parent1072.CAL_FOLD
T2 = parent1072.T2
REPRO_REL_TOL = 1e-6  # g4' per-context full-channel REL tolerance (parity scale)
REPRO_POOLED_TOL = 1e-6  # g4' pooled r2_full ABS tolerance (parent g4 leaf convention)
COS_GATE_MIN = parent1072.COS_GATE_MIN  # 0.999 (g2/g6 per-cell floor, K2)
G3P_COS_MIN = 0.9999  # g3' span-mean bar (plan §11; #1005 bf16 re-capture bar)
COS_GATE_MAX_BELOW_FRAC = parent1072.COS_GATE_MAX_BELOW_FRAC  # 0.001
RANK_REDUCED_MAX_FRAC = 0.01  # g7 abort threshold per arm x basis (K1)
PILOT_ABORT_RC = parent1072.PILOT_ABORT_RC  # 7
CAPTURE_BOOKED_H = 1.5  # plan §9 row B
BATTERY_BOOKED_H = 3.5  # plan §9 row C
STAGE_FREE_GB_MIN = 69.0  # plan §9: 1.5x (26 staged + 16 model + 4 store)
ARMS = parent952.ARMS
MATCHED_ARMS = parent952.MATCHED_ARMS
SLOT_NAMES = parent952.SLOT_NAMES
SLOT_IDX = parent952.SLOT_IDX
DECOMP_SLOTS = parent1072.DECOMP_SLOTS  # 41 single-position slots
DEFAULT_LAMBDAS_LIST = parent952.DEFAULT_LAMBDAS_LIST

log_phase = parent952.log_phase
write_sentinel = parent952.write_sentinel
_json_np = parent952._json_np


# ═══════════════════════════════════════════════════════════════════════════════
# Dirs + upload + metadata + pilot
# ═══════════════════════════════════════════════════════════════════════════════


def _tensors_dir_lowdim(base_dir: pathlib.Path) -> pathlib.Path:
    d = base_dir / "analysis_tensors_lowdim"
    d.mkdir(parents=True, exist_ok=True)
    return d


def eval_out_dir_lowdim(base_dir: pathlib.Path) -> pathlib.Path:
    """This round's eval dir (git mirror: eval_results/issue_1072/lowdim-token-subspace)."""
    d = base_dir / "eval_results" / "issue_1072" / FOLLOWUP_LABEL
    d.mkdir(parents=True, exist_ok=True)
    return d


def _repro_dir(base_dir: pathlib.Path) -> pathlib.Path:
    """Staged completed-round reproduction npz (production phase A)."""
    d = base_dir / "parent_repro_1072"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hf_commit_files_lowdim(label: str, pairs: list[tuple[pathlib.Path, str]]) -> None:
    """One retried create_commit under ISSUE_SLUG + scoped verify (fail-loud).

    ``pairs`` = (local path, repo subpath under ISSUE_SLUG) — this round owns
    its OWN prefix ``issue1072_lowdim_token_subspace/`` (never the parent's).
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    from explore_persona_space.orchestrate import hub as eps_hub

    ops = [
        CommitOperationAdd(path_in_repo=f"{ISSUE_SLUG}/{sub}", path_or_fileobj=str(p))
        for p, sub in pairs
    ]
    api = HfApi()
    eps_hub.retry_transient(
        lambda: api.create_commit(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue 1072 {FOLLOWUP_LABEL}: {label} ({len(ops)} files)",
            operations=ops,
        ),
        what=f"issue1072-lowdim create_commit {label}",
    )
    missing = eps_hub.verify_repo_paths_uploaded(
        api,
        HF_DATA_REPO,
        [op.path_in_repo for op in ops],
        path_in_repo=ISSUE_SLUG,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"HF upload verification FAIL ({label}): missing {sorted(missing)[:3]}")
    logger.info("[upload] %s: %d files committed + Hub-verified", label, len(ops))


def _run_metadata(smoke: bool, layers: tuple[int, ...]) -> dict[str, Any]:
    """Reproducibility metadata carried by every result JSON (CLAUDE.md rule)."""
    return {
        "issue": ISSUE,
        "followup_label": FOLLOWUP_LABEL,
        "git_sha": parent952._repo_git_sha(),
        "env_versions": parent1072._env_versions(),
        "ts": time.time(),
        "smoke": bool(smoke),
        "layers": list(layers),
        "bases": list(BASES),
        "parent_tensor_revision": PARENT_TENSOR_REVISION,
        "decomp_store_revision": DECOMP_STORE_REVISION,
        "model_id": DEFAULT_MODEL if not smoke else "smoke-tiny-qwen2(seed 0)",
    }


def _pilot_check(
    name: str,
    measured_wall_s: float,
    units_done: int,
    units_total: int,
    booked_h: float,
    base_dir: pathlib.Path,
    smoke: bool,
    execution_shape: str,
) -> None:
    """K4 designed compute abort: projected wall > 2x booked -> report + rc 7.

    Identical semantics to the parent's gate (verdict demoted to a log line
    under smoke — the #1345 gate-calibration lesson); writes to THIS round's
    eval dir.
    """
    per_unit = measured_wall_s / max(units_done, 1)
    projected_h = per_unit * units_total / 3600.0
    rec = {
        "gate": name,
        "measured_wall_s": measured_wall_s,
        "units_done": units_done,
        "units_total": units_total,
        "per_unit_s": per_unit,
        "projected_wall_h": projected_h,
        "booked_wall_h": booked_h,
        "abort_threshold_h": 2.0 * booked_h,
        "execution_shape": execution_shape,
        "verdict": "ABORT" if projected_h > 2.0 * booked_h else "PASS",
        "smoke_demoted": bool(smoke),
        "ts": time.time(),
    }
    out = eval_out_dir_lowdim(base_dir) / f"pilot_gate_{name}.json"
    out.write_text(json.dumps(rec, indent=2, default=_json_np))
    logger.info("[pilot:%s] %s", name, json.dumps(rec, default=_json_np))
    if rec["verdict"] == "ABORT" and not smoke:
        log_phase(f"pilot_abort_{name}")
        logger.error(
            "[pilot:%s] projected %.2f h > 2x booked %.1f h — designed abort (K4), rc=%d",
            name,
            projected_h,
            booked_h,
            PILOT_ABORT_RC,
        )
        sys.exit(PILOT_ABORT_RC)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A — stage (plan §4.3 Phase A)
# ═══════════════════════════════════════════════════════════════════════════════

DECOMP_SHARD_KEYS = frozenset(
    {
        "rem_par",
        "rem_nx",
        "full_par",
        "alpha_slots",
        "alpha_pos_flat",
        "rem_valid",
        "ids",
        "slot_names_46",
        "layer",
    }
)
NEXT_NPZ_KEYS = frozenset({"next_flat", "next_offsets", "slot_next_ids", "ids", "prefix_len"})


def phase_stage(base_dir: pathlib.Path, smoke: bool, layers: tuple[int, ...], fit_device: str):
    """Production: pool verify + scoped staging at the three §10 pins.

    Smoke: the PARENT PRODUCERS run at tiny n — parent1072 phase_stage (synth
    pool + parent-schema slot shards), phase_capture (decomp store + next-id
    npz + gates) and phase_battery (committed-format reproduction npz +
    battery JSONs + smoke-reference λ tables) — so the lowdim phases consume
    the identical file interfaces in smoke and production.
    """
    log_phase("stage")
    if smoke:
        parent1072.phase_stage(base_dir, smoke=True, layers=layers)
        parent1072.phase_capture(
            base_dir, smoke=True, layers=layers, batch_size=4, skip_upload=True
        )
        parent1072.phase_battery(
            base_dir,
            smoke=True,
            layers=layers,
            fit_device=fit_device,
            skip_upload=True,
            u_dir_np=None,
        )
        rec = {"synth": True, "parent_producers_ran": True, **_run_metadata(smoke, layers)}
        (eval_out_dir_lowdim(base_dir) / "stage_manifest_lowdim.json").write_text(
            json.dumps(rec, indent=2, default=_json_np)
        )
        log_phase("stage_done")
        return

    import torch

    from explore_persona_space.orchestrate import hub as eps_hub

    # Boot-disk headroom preamble (plan §9 disk row).
    st = os.statvfs(str(base_dir))
    free_gb = st.f_bavail * st.f_frsize / 1e9
    assert free_gb >= STAGE_FREE_GB_MIN, (
        f"boot-disk free {free_gb:.1f} GB < {STAGE_FREE_GB_MIN} GB required before staging"
    )

    rec0 = parent952.phase0_verify(base_dir, smoke=False)
    pool_ids = rec0["pool_ids"]
    tensors_dir = parent1072._tensors_dir(base_dir)
    staged: list[str] = []
    for arm in ARMS:
        for name in [f"slots_{arm}_L{la}.pt" for la in layers] + [f"spans_{arm}.json"]:
            eps_hub.stage_hub_file(
                HF_DATA_REPO,
                f"{PARENT_SLUG}/analysis_tensors/{name}",
                tensors_dir / name,
                repo_type="dataset",
                revision=PARENT_TENSOR_REVISION,
            )
            staged.append(name)
        for name in [f"decomp_{arm}_L{la}.pt" for la in layers] + [f"next_ids_{arm}.npz"]:
            eps_hub.stage_hub_file(
                HF_DATA_REPO,
                f"{DECOMP_SLUG}/analysis_tensors/{name}",
                tensors_dir / name,
                repo_type="dataset",
                revision=DECOMP_STORE_REVISION,
            )
            staged.append(name)
    for k in range(K_FOLDS):
        name = f"per_context_stats_1072_fold{k}.npz"
        eps_hub.stage_hub_file(
            HF_DATA_REPO,
            f"{DECOMP_SLUG}/eval_results/issue_1072/{name}",
            _repro_dir(base_dir) / name,
            repo_type="dataset",
            revision=DECOMP_STORE_REVISION,
        )
        staged.append(name)
    for p in [tensors_dir / n for n in staged if not n.startswith("per_context")]:
        assert p.stat().st_size > 0, p

    # Pre-registered realized-keys probes (reuse check (c), plan §12 A1/A2).
    probe = torch.load(
        str(tensors_dir / f"slots_own_L{layers[0]}.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    assert set(probe.keys()) == parent1072.STAGE_SHARD_KEYS, sorted(probe.keys())
    assert probe["slots"].shape == (len(pool_ids), 72, parent1072.PROD_HIDDEN), probe["slots"].shape
    assert probe["slot_names"] == list(SLOT_NAMES), "slot registry drift"
    del probe
    dprobe = torch.load(
        str(tensors_dir / f"decomp_own_L{layers[0]}.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    assert set(dprobe.keys()) == DECOMP_SHARD_KEYS | {"p_last"}, sorted(dprobe.keys())
    assert dprobe["rem_nx"].shape == (len(pool_ids), parent1072.PROD_HIDDEN), dprobe["rem_nx"].shape
    assert dprobe["rem_nx"].dtype == torch.float16, dprobe["rem_nx"].dtype
    assert [int(i) for i in dprobe["ids"]] == [int(i) for i in pool_ids], "decomp id drift"
    del dprobe
    nprobe = np.load(tensors_dir / "next_ids_own.npz")
    assert set(nprobe.files) == set(NEXT_NPZ_KEYS), sorted(nprobe.files)
    assert nprobe["ids"].tolist() == [int(i) for i in pool_ids], "next-ids id drift"
    gc.collect()

    rec = {
        "n_pool": len(pool_ids),
        "staged_files": staged,
        "free_gb_before": free_gb,
        "realized_keys_check": "PASS",
        **_run_metadata(smoke, layers),
    }
    (eval_out_dir_lowdim(base_dir) / "stage_manifest_lowdim.json").write_text(
        json.dumps(rec, indent=2, default=_json_np)
    )
    log_phase("stage_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B core — TF capture with 3-basis subspace accumulators (plan §4.3 B)
# ═══════════════════════════════════════════════════════════════════════════════


def _cos_np(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb + 1e-9))


def _k2_counter() -> dict[str, Any]:
    return {"n_cells": 0, "n_below": 0, "min_cos": 1.0}


def _k2_update(rec: dict[str, Any], cos: float, floor: float) -> None:
    rec["n_cells"] += 1
    rec["n_below"] += int(cos < floor)
    rec["min_cos"] = min(rec["min_cos"], cos)


def tf_capture_lowdim_arm(  # noqa: C901 — batched TF loop; reductions GPU-resident
    model,
    tokenizer,
    ids: list[int],
    prompts_by_id: dict[int, str],
    answers_by_id: dict[int, str],
    arm_name: str,
    layers: tuple[int, ...],
    u_dir,
    staged_spans: dict[str, dict],
    staged_slots_by_layer: dict[int, Any],
    staged_decomp_by_layer: dict[int, dict],
    staged_next: dict[str, np.ndarray],
    batch_size: int = 8,
    pos_chunk: int = 512,
    pilot: dict | None = None,
    base_dir: pathlib.Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """One TF forward per context (parent LEFT-pad rig: adapted from
    parent1072.tf_capture_decomp_arm — same batching, position_ids,
    GPU-resident reductions), hooked at ``layers`` + the FINAL decoder layer.

    Emits per (context, layer, basis): ``rem_par_{b}_mean_gt16_nx`` (mean_t
    P_b(t) z_t over t in 17..span-1); layer-independent: per-position top-32
    candidate ids + logprobs (ragged), per-slot top-32 ids, look8 eff-k
    records, coverage/overlap counters. In-stream gates: g1 (span re-render +
    next-id identity vs the staged store), g2 (slot z cos vs #952 shards),
    g3' (recomputed reduced-range rem_nx cos vs staged decomp rem_nx), g6
    (1-D alpha parity vs staged alpha_pos_flat), g7 (QR rank guard + counters).
    """
    import torch

    hid = model.config.hidden_size
    rms_eps = float(model.config.rms_norm_eps)
    n = len(ids)
    n_layers = len(layers)
    fin_layer = model.config.num_hidden_layers - 1
    hook_layers = sorted(set(layers) | {fin_layer})
    dev = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ── prep: render + g1 (spans + next-id identity) ─────────────────────────────
    prepped: list[tuple[int, dict]] = []
    g1_mismatches: list[dict] = []
    full_ids_all: list[list[int]] = []
    st_off = staged_next["next_offsets"]
    st_flat = staged_next["next_flat"]
    for row_i, cid in enumerate(ids):
        info = parent952._render_and_index(tokenizer, prompts_by_id[cid], answers_by_id[cid])
        assert info is not None, f"[{arm_name}] id {cid}: empty answer/span"
        ref = staged_spans.get(str(cid))
        if (
            ref is None
            or int(ref.get("span", -1)) != info["span"]
            or bool(ref.get("truncated")) != bool(info["truncated"])
        ):
            g1_mismatches.append(
                {"id": cid, "kind": "span", "recomputed": info["span"], "staged": ref}
            )
        # Next-id identity vs the staged store (K1-adjacent ids-drift check).
        rs, ee = info["prompt_len"], info["ext_end"]
        next_rec = np.asarray(info["full_ids"][rs + 1 : ee], dtype=np.int32)
        st_row = st_flat[st_off[row_i] : st_off[row_i + 1]]
        if len(st_row) != len(next_rec) or not np.array_equal(st_row, next_rec):
            g1_mismatches.append({"id": cid, "kind": "next_ids", "n_recomputed": len(next_rec)})
        prepped.append((row_i, info))
        full_ids_all.append(info["full_ids"])
    if g1_mismatches:
        rec = {"arm": arm_name, "n_mismatch": len(g1_mismatches), "sample": g1_mismatches[:5]}
        if base_dir is not None:
            (eval_out_dir_lowdim(base_dir) / f"g1_abort_{arm_name}.json").write_text(
                json.dumps(rec, indent=2, default=_json_np)
            )
        raise RuntimeError(f"g1 span/next-id alignment FAIL (K1): {rec['n_mismatch']} mismatches")

    prefix_len = parent1072._lcp_prefix_len(full_ids_all)

    # ── accumulators ─────────────────────────────────────────────────────────────
    rem_par = {b: np.full((n_layers, n, hid), np.nan, dtype=np.float16) for b in BASES}
    rem_valid = np.zeros(n, dtype=bool)
    top32_rows: list[np.ndarray] = []  # per context (span-1, 32) int32
    lp_rows: list[np.ndarray] = []  # per context (span-1, 32) fp16
    effk_rows: list[np.ndarray] = []  # per context (span-1,) int8 (look8 post-dedupe)
    slot_top32 = np.full((n, 46, 32), -1, dtype=np.int32)
    g2 = {int(la): _k2_counter() for la in layers}
    g3p = {int(la): _k2_counter() for la in layers}
    g6 = {int(la): _k2_counter() for la in layers}
    g7 = {b: {"n_positions": 0, "n_rank_reduced": 0} for b in BASES}
    coverage = {"n_positions": 0, "top8_hits": 0, "top32_hits": 0}
    overlap_hist = np.zeros(sb.LOOKAHEAD + 1, dtype=np.int64)
    effk_hist = np.zeros(sb.LOOKAHEAD + 1, dtype=np.int64)
    nesting_max_gap = 0.0
    nesting_checked = False

    captured: dict[int, Any] = {}

    def make_hook(la: int):
        def hook(module, _inp, output):
            captured[la] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    handles = [model.model.layers[la].register_forward_hook(make_hook(la)) for la in hook_layers]
    t_arm0 = time.time()
    try:
        with torch.no_grad():
            for b0 in range(0, len(prepped), batch_size):
                batch = prepped[b0 : b0 + batch_size]
                max_len = max(len(info["full_ids"]) for _ri, info in batch)
                input_ids, attn, pad_offs = [], [], []
                for _ri, info in batch:
                    pad_n = max_len - len(info["full_ids"])
                    input_ids.append([pad_id] * pad_n + info["full_ids"])
                    attn.append([0] * pad_n + [1] * len(info["full_ids"]))
                    pad_offs.append(pad_n)
                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=dev)
                attn_t = torch.tensor(attn, dtype=torch.long, device=dev)
                pos_ids_t = (attn_t.cumsum(dim=-1) - 1).clamp(min=0)
                captured.clear()
                model(
                    input_ids=input_ids_t,
                    attention_mask=attn_t,
                    position_ids=pos_ids_t,
                    output_hidden_states=False,
                )

                for j, (row_i, info) in enumerate(batch):
                    pad = pad_offs[j]
                    rs, ee, span = info["prompt_len"], info["ext_end"], info["span"]
                    fids = info["full_ids"]
                    next_ext = np.asarray(fids[rs:ee], dtype=np.int64)
                    pos, valid = parent952._slot_positions_and_validity(rs, ee, span)

                    # ── final-hidden logits read (chunked GEMM + GPU topk) ──────
                    h_fin = captured[fin_layer][j]
                    h_pred = h_fin[pad + rs : pad + ee - 1]  # (span-1, H): positions 0..span-2
                    ids32_t, lp32_t = sb.topk_ids_from_final_hidden(
                        h_pred, u_dir, rms_eps, k=sb.TOPK, chunk=pos_chunk
                    )
                    top32_rows.append(ids32_t.cpu().numpy().astype(np.int32))
                    lp_rows.append(lp32_t.cpu().numpy())
                    # Slot candidates (valid slot positions only).
                    v_idx = np.nonzero(valid)[0]
                    h_slots = h_fin[torch.from_numpy(pos[v_idx] + pad).to(dev)]
                    sid_t, _slp = sb.topk_ids_from_final_hidden(
                        h_slots, u_dir, rms_eps, k=sb.TOPK, chunk=pos_chunk
                    )
                    slot_top32[row_i, v_idx] = sid_t.cpu().numpy().astype(np.int32)

                    # ── coverage + look8 windows (vectorized) ────────────────────
                    realized = torch.from_numpy(next_ext[1:]).to(dev)  # (span-1,)
                    ids32_l = ids32_t.to(torch.int64)
                    in8 = (ids32_l[:, : sb.NESTED_K] == realized[:, None]).any(dim=1)
                    in32 = (ids32_l == realized[:, None]).any(dim=1)
                    coverage["n_positions"] += span - 1
                    coverage["top8_hits"] += int(in8.sum())
                    coverage["top32_hits"] += int(in32.sum())
                    # look8 window[j, m] = next_ext[j+1+m], valid while j+1+m <= span-1.
                    p_arange = torch.arange(span - 1, device=dev)
                    m_arange = torch.arange(sb.LOOKAHEAD, device=dev)
                    widx = p_arange[:, None] + m_arange[None, :]  # index into realized
                    wvalid = widx <= span - 2
                    window = realized[widx.clamp(max=span - 2)]
                    look_ids, look_effk = sb.compact_dedupe_windows(window, wvalid)
                    effk_rows.append(look_effk.cpu().numpy().astype(np.int8))
                    effk_hist += np.bincount(look_effk.cpu().numpy(), minlength=sb.LOOKAHEAD + 1)[
                        : sb.LOOKAHEAD + 1
                    ]
                    # |S_look8 ∩ S_top32| per position (unique window ids in top-32).
                    keep_m = wvalid & ~(
                        torch.stack(
                            [
                                (window[:, m : m + 1] == window[:, :m]).any(dim=1)
                                if m
                                else torch.zeros(span - 1, dtype=torch.bool, device=dev)
                                for m in range(sb.LOOKAHEAD)
                            ],
                            dim=1,
                        )
                    )
                    inter = (
                        ((window[:, :, None] == ids32_l[:, None, :]).any(dim=2) & keep_m)
                        .sum(dim=1)
                        .cpu()
                        .numpy()
                    )
                    overlap_hist += np.bincount(inter, minlength=sb.LOOKAHEAD + 1)[
                        : sb.LOOKAHEAD + 1
                    ]

                    # ── per-position bases + projections (chunked over positions) ─
                    u_rows = u_dir[realized]
                    u_hat = u_rows / u_rows.norm(dim=1, keepdim=True)
                    span_hs = {la: captured[la][j][pad + rs : pad + ee].float() for la in layers}
                    rem_lo, rem_hi = T2, span - 1  # positions t = T2..span-2
                    has_rem = span >= T2 + 2
                    rem_sums = {
                        b: torch.zeros((n_layers, hid), dtype=torch.float32, device=dev)
                        for b in BASES
                    }
                    for c0 in range(0, span - 1, pos_chunk):
                        c1 = min(c0 + pos_chunk, span - 1)
                        q32, _eff32, red32 = sb.orthonormal_bases(u_dir, ids32_l[c0:c1])
                        g7["top32"]["n_positions"] += c1 - c0
                        g7["top32"]["n_rank_reduced"] += len(red32)
                        g7["top8"]["n_positions"] += c1 - c0
                        q8, _eff8, red8 = sb.nested_leading_bases(
                            u_dir, q32, ids32_l[c0:c1], red32, k_lead=sb.NESTED_K
                        )
                        g7["top8"]["n_rank_reduced"] += len(red8)
                        ql, _effl, redl = sb.orthonormal_bases(
                            u_dir, look_ids[c0:c1], eff_k=look_effk[c0:c1]
                        )
                        g7["look8"]["n_positions"] += c1 - c0
                        g7["look8"]["n_rank_reduced"] += len(redl)
                        if smoke and not nesting_checked and c1 - c0 >= 4:
                            gap = sb.nesting_check(
                                u_dir, ids32_l[c0 : c0 + min(16, c1 - c0)].cpu().numpy()
                            )
                            nesting_max_gap = max(nesting_max_gap, gap)
                            assert gap < 1e-4, f"top-8 nesting numerical gap {gap:.2e}"
                            nesting_checked = True
                        if has_rem:
                            m_lo, m_hi = max(rem_lo, c0), min(rem_hi, c1)
                            if m_lo < m_hi:
                                for li, la in enumerate(layers):
                                    z = span_hs[la][m_lo:m_hi]
                                    rem_sums["top32"][li] += sb.project_rows(
                                        z, q32[m_lo - c0 : m_hi - c0]
                                    ).sum(0)
                                    rem_sums["top8"][li] += sb.project_rows(
                                        z, q8[m_lo - c0 : m_hi - c0]
                                    ).sum(0)
                                    rem_sums["look8"][li] += sb.project_rows(
                                        z, ql[m_lo - c0 : m_hi - c0]
                                    ).sum(0)
                        del q32, q8, ql
                    if has_rem:
                        rem_valid[row_i] = True
                        cnt = float(rem_hi - rem_lo)
                        for b in BASES:
                            rem_par[b][:, row_i, :] = (
                                (rem_sums[b] / cnt).to(torch.float16).cpu().numpy()
                            )

                    # ── per-layer gates g2 / g3' / g6 ────────────────────────────
                    for la in layers:
                        hs = captured[la][j]
                        z_span = span_hs[la]
                        # g6: 1-D alpha parity vs staged alpha_pos_flat (fp16 store).
                        alpha = (z_span[: span - 1] * u_hat).sum(dim=1)
                        st_a = staged_decomp_by_layer[la]["alpha_pos_flat"][
                            st_off[row_i] : st_off[row_i + 1]
                        ].astype(np.float32)
                        _k2_update(
                            g6[la],
                            _cos_np(alpha.to(torch.float16).float().cpu().numpy(), st_a),
                            COS_GATE_MIN,
                        )
                        # g2: recomputed slot z vs staged #952 shard rows.
                        idx = torch.from_numpy(pos + pad).clamp(min=0).to(dev)
                        single = hs[idx].float()
                        valid_t = torch.from_numpy(valid).to(dev)
                        staged_row = staged_slots_by_layer[la][row_i]
                        st46 = staged_row[:46].to(dev).float()
                        both = valid_t & torch.isfinite(st46).all(dim=1)
                        if both.any():
                            num = (single[both] * st46[both]).sum(dim=1)
                            den = single[both].norm(dim=1) * st46[both].norm(dim=1) + 1e-9
                            cos = (num / den).cpu().numpy()
                            g2[la]["n_cells"] += int(both.sum())
                            g2[la]["n_below"] += int((cos < COS_GATE_MIN).sum())
                            g2[la]["min_cos"] = min(g2[la]["min_cos"], float(cos.min()))
                        # g3': recomputed REDUCED-range rem_nx vs staged decomp rem_nx.
                        if has_rem:
                            rec_rem = z_span[T2 : span - 1].mean(0).cpu().numpy()
                            st_rem = staged_decomp_by_layer[la]["rem_nx"][row_i].astype(np.float32)
                            assert np.isfinite(st_rem).all(), (arm_name, la, ids[row_i])
                            _k2_update(g3p[la], _cos_np(rec_rem, st_rem), G3P_COS_MIN)

                captured.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batches_done = b0 // batch_size + 1
                if batches_done % 25 == 0 or batches_done == 1:
                    logger.info("[capture:%s] %d/%d rows", arm_name, b0 + len(batch), len(prepped))
                if pilot is not None and pilot.get("armed") and batches_done == 2:
                    pilot["armed"] = False
                    assert base_dir is not None
                    _pilot_check(
                        "capture",
                        time.time() - t_arm0,
                        units_done=2,
                        units_total=pilot["total_batches"],
                        booked_h=CAPTURE_BOOKED_H,
                        base_dir=base_dir,
                        smoke=smoke,
                        execution_shape=(
                            f"batched TF forward + chunked logits/QR, batch_size={batch_size}"
                        ),
                    )
    finally:
        for h in handles:
            h.remove()
        captured.clear()

    # Staged-vs-recomputed rem_valid identity (same span rule on the same spans).
    st_rv = staged_decomp_by_layer[layers[0]]["rem_valid"]
    assert np.array_equal(rem_valid, st_rv), f"[{arm_name}] rem_valid drift vs staged store"

    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, a in enumerate(top32_rows):
        offsets[i + 1] = offsets[i] + len(a)
    assert np.array_equal(offsets, st_off), f"[{arm_name}] ragged offsets drift vs staged store"
    return {
        "prefix_len": prefix_len,
        "rem_par": rem_par,
        "rem_valid": rem_valid,
        "top32_flat": np.concatenate(top32_rows) if offsets[-1] else np.zeros((0, 32), np.int32),
        "logprobs_flat": np.concatenate(lp_rows) if offsets[-1] else np.zeros((0, 32), np.float16),
        "offsets": offsets,
        "slot_top32": slot_top32,
        "look8_effk_flat": (np.concatenate(effk_rows) if offsets[-1] else np.zeros(0, np.int8)),
        "g2": g2,
        "g3p": g3p,
        "g6": g6,
        "g7": g7,
        "coverage": coverage,
        "overlap_hist": overlap_hist,
        "effk_hist": effk_hist,
        "nesting_max_gap": nesting_max_gap if smoke else None,
        "wall_s": time.time() - t_arm0,
    }


def _save_lowdim_shards(
    base_dir: pathlib.Path,
    arm: str,
    layers: tuple[int, ...],
    ids: list[int],
    cap: dict[str, Any],
) -> list[tuple[pathlib.Path, str]]:
    """Persist the subspace store (plan §6.5 deliverable 2) -> (path, repo sub)."""
    import torch

    td = _tensors_dir_lowdim(base_dir)
    pairs: list[tuple[pathlib.Path, str]] = []
    for li, la in enumerate(layers):
        payload: dict[str, Any] = {
            f"rem_par_{b}": torch.from_numpy(np.ascontiguousarray(cap["rem_par"][b][li]))
            for b in BASES
        }
        payload["rem_valid"] = torch.from_numpy(cap["rem_valid"])
        payload["ids"] = list(ids)
        payload["layer"] = int(la)
        p = td / f"lowdim_{arm}_L{la}.pt"
        torch.save(payload, str(p))
        pairs.append((p, f"analysis_tensors/{p.name}"))
    npz_p = td / f"top32_ids_{arm}.npz"
    np.savez(
        npz_p,
        top32_flat=cap["top32_flat"],
        logprobs_flat=cap["logprobs_flat"],
        offsets=cap["offsets"],
        slot_top32=cap["slot_top32"],
        look8_effk_flat=cap["look8_effk_flat"],
        ids=np.asarray(ids, dtype=np.int64),
        prefix_len=np.asarray([cap["prefix_len"]], dtype=np.int64),
    )
    pairs.append((npz_p, f"analysis_tensors/{npz_p.name}"))
    return pairs


def _capture_regime(
    smoke: bool, layers: tuple[int, ...], pool_ids: list[int], batch_size: int
) -> dict[str, Any]:
    """Output-affecting resume keys (#722 r3 rule; parent convention + label)."""
    return {
        "followup": FOLLOWUP_LABEL,
        "smoke": bool(smoke),
        "layers": list(layers),
        "bases": list(BASES),
        "n_pool": len(pool_ids),
        "pool_sha": hashlib.sha256(json.dumps([int(i) for i in pool_ids]).encode()).hexdigest(),
        "batch_size": int(batch_size),
        "model_id": "smoke-tiny-qwen2(seed 0)" if smoke else DEFAULT_MODEL,
        "staging_revision": "smoke-synth" if smoke else DECOMP_STORE_REVISION,
        "git_sha": parent952._repo_git_sha(),
    }


def phase_capture(  # noqa: C901 — the phase-B driver: gates + 4 arms + uploads
    base_dir: pathlib.Path,
    smoke: bool,
    layers: tuple[int, ...],
    batch_size: int,
    skip_upload: bool,
) -> np.ndarray:
    """Phase B: per-arm subspace capture with in-stream gates, per-arm
    checkpoint + incremental upload, then the B' store upload BEFORE phase C.
    Returns U_dir (V, H) fp32 numpy."""
    import torch

    log_phase("capture")
    pool_ids, prompts, texts = parent1072._load_capture_inputs(base_dir, smoke, layers)
    tensors_dir = parent1072._tensors_dir(base_dir)
    regime = _capture_regime(smoke, layers, pool_ids, batch_size)

    if smoke:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
        model = parent1072._build_smoke_model(tokenizer)
    else:
        model, tokenizer = parent1072._load_production_model(DEFAULT_MODEL)
    u_dir = parent1072._unembed_dir(model)

    staged_spans = {arm: parent952._load_spans(base_dir, arm) for arm in ARMS}
    staged_slots: dict[str, dict[int, Any]] = {}
    staged_decomp: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    staged_next: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        staged_slots[arm] = {}
        staged_decomp[arm] = {}
        for la in layers:
            d = torch.load(
                str(tensors_dir / f"slots_{arm}_L{la}.pt"),
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            assert [int(i) for i in d["ids"]] == [int(i) for i in pool_ids], (
                f"staged shard id drift ({arm}, L{la})"
            )
            staged_slots[arm][la] = d["slots"]
            dd = torch.load(
                str(tensors_dir / f"decomp_{arm}_L{la}.pt"),
                map_location="cpu",
                weights_only=False,
            )
            assert [int(i) for i in dd["ids"]] == [int(i) for i in pool_ids], (
                f"decomp shard id drift ({arm}, L{la})"
            )
            staged_decomp[arm][la] = {
                "alpha_pos_flat": dd["alpha_pos_flat"].numpy(),
                "rem_nx": dd["rem_nx"].numpy(),
                "rem_valid": dd["rem_valid"].numpy(),
            }
            del dd
        nz = np.load(tensors_dir / f"next_ids_{arm}.npz")
        assert nz["ids"].tolist() == [int(i) for i in pool_ids], f"next-ids id drift ({arm})"
        staged_next[arm] = {k: nz[k] for k in nz.files}

    n_batches_total = len(ARMS) * ((len(pool_ids) + batch_size - 1) // batch_size)
    pilot = {"armed": True, "total_batches": n_batches_total}
    gates: dict[str, Any] = {
        "g1": {},
        "g2": {},
        "g3p": {},
        "g6": {},
        "g7": {},
        "coverage": {},
        "overlap_hist": {},
        "effk_hist": {},
        "prefix": {},
        "arm_wall_s": {},
        "nesting_max_gap": {},
    }
    t0 = time.time()
    for arm in ARMS:
        td_low = _tensors_dir_lowdim(base_dir)
        manifest_p = td_low / f"lowdim_manifest_{arm}.json"
        shard_paths = [td_low / f"lowdim_{arm}_L{la}.pt" for la in layers] + [
            td_low / f"top32_ids_{arm}.npz"
        ]
        if manifest_p.exists() and all(p.exists() for p in shard_paths):
            persisted = json.loads(manifest_p.read_text())
            if persisted.get("regime") == regime:
                logger.info("[capture] SKIP %s (per-arm resume, regime match)", arm)
                for k in ("g1", "g2", "g3p", "g6", "g7", "coverage"):
                    gates[k][arm] = persisted.get(k)
                pilot["armed"] = False
                continue
            logger.warning("[capture] %s manifest regime mismatch — recomputing", arm)
        cap = tf_capture_lowdim_arm(
            model,
            tokenizer,
            pool_ids,
            prompts,
            texts[arm],
            arm,
            layers,
            u_dir,
            staged_spans=staged_spans[arm],
            staged_slots_by_layer=staged_slots[arm],
            staged_decomp_by_layer=staged_decomp[arm],
            staged_next=staged_next[arm],
            batch_size=batch_size,
            pilot=pilot,
            base_dir=base_dir,
            smoke=smoke,
        )
        # g7 verdict per basis (K1): systematic rank degeneracy = ids bug.
        for b in BASES:
            npos = max(cap["g7"][b]["n_positions"], 1)
            frac = cap["g7"][b]["n_rank_reduced"] / npos
            cap["g7"][b]["reduced_frac"] = frac
            if frac > RANK_REDUCED_MAX_FRAC:
                rec = {"arm": arm, "basis": b, **cap["g7"][b]}
                (eval_out_dir_lowdim(base_dir) / f"g7_abort_{arm}_{b}.json").write_text(
                    json.dumps(rec, indent=2, default=_json_np)
                )
                raise RuntimeError(f"g7 rank-guard FAIL (K1): {rec}")
        if smoke:
            assert cap["g7"]["top32"]["n_rank_reduced"] == 0, (
                "smoke expects 0 rank reductions on real top-32 ids (plan §12 A9)"
            )
        pairs = _save_lowdim_shards(base_dir, arm, layers, pool_ids, cap)
        gates["g1"][arm] = {"n_checked": len(pool_ids), "n_mismatch": 0}
        gates["g2"][arm] = {str(k): v for k, v in cap["g2"].items()}
        gates["g3p"][arm] = {str(k): v for k, v in cap["g3p"].items()}
        gates["g6"][arm] = {str(k): v for k, v in cap["g6"].items()}
        gates["g7"][arm] = cap["g7"]
        gates["coverage"][arm] = cap["coverage"]
        gates["overlap_hist"][arm] = cap["overlap_hist"].tolist()
        gates["effk_hist"][arm] = cap["effk_hist"].tolist()
        gates["prefix"][arm] = {"prefix_len": cap["prefix_len"]}
        gates["arm_wall_s"][arm] = cap["wall_s"]
        gates["nesting_max_gap"][arm] = cap["nesting_max_gap"]
        if not smoke:
            assert cap["coverage"]["top32_hits"] > 0, (
                f"[{arm}] zero realized-in-top32 coverage — ids bug (plan §12 A6)"
            )
        manifest_p.write_text(
            json.dumps(
                {
                    "regime": regime,
                    **{k: gates[k][arm] for k in ("g1", "g2", "g3p", "g6", "g7", "coverage")},
                    **_run_metadata(smoke, layers),
                },
                indent=2,
                default=_json_np,
            )
        )
        if not skip_upload:
            _hf_commit_files_lowdim(
                f"lowdim store ({arm})",
                [*pairs, (manifest_p, f"analysis_tensors/{manifest_p.name}")],
            )

    # ── K2: pooled g2/g3'/g6 verdict across arms x layers ───────────────────────
    tot_cells = tot_below = 0
    min_cos = 1.0
    for gname in ("g2", "g3p", "g6"):
        for _arm, per_layer in gates[gname].items():
            for _la, rec in (per_layer or {}).items():
                tot_cells += int(rec["n_cells"])
                tot_below += int(rec["n_below"])
                min_cos = min(min_cos, float(rec["min_cos"]))
    below_frac = (tot_below / tot_cells) if tot_cells else 0.0
    gates["k2"] = {
        "n_cells": tot_cells,
        "n_below": tot_below,
        "below_frac": below_frac,
        "min_cos": min_cos,
        "max_below_frac": COS_GATE_MAX_BELOW_FRAC,
        "verdict": "PASS" if below_frac <= COS_GATE_MAX_BELOW_FRAC else "FAIL",
    }
    gates["total_wall_s"] = time.time() - t0
    gates.update(_run_metadata(smoke, layers))
    gates_p = eval_out_dir_lowdim(base_dir) / "capture_gates_lowdim.json"
    gates_p.write_text(json.dumps(gates, indent=2, default=_json_np))
    if not skip_upload:
        _hf_commit_files_lowdim("capture gates", [(gates_p, f"eval_results/{gates_p.name}")])
    if gates["k2"]["verdict"] == "FAIL":
        if smoke:
            logger.warning("[k2] smoke-demoted equivalence verdict: %s", gates["k2"])
        else:
            raise RuntimeError(f"g2/g3'/g6 equivalence FAIL (K2): {gates['k2']}")

    u_np = u_dir.cpu().numpy()
    del model, u_dir, staged_slots, staged_decomp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_phase("capture_done")
    return u_np


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C — subspace component battery (plan §4.3 Phase C)
# ═══════════════════════════════════════════════════════════════════════════════


class LowdimRefs:
    """Frozen-λ + g4'/g5 reference provider.

    Production: the completed round's committed ``battery_1072_fold{k}.json``
    (repo) + #952's ``stats_kfold.json`` λ tables (repo) + the staged
    reproduction npz (@ DECOMP_STORE_REVISION). Smoke: the SAME file shapes,
    generated by the parent producers at stage time (parent battery JSONs +
    npz under the parent eval dir; slot-λ tables from the parent's
    smoke_reference self-records) — the comparison code path is identical.
    """

    def __init__(self, base_dir: pathlib.Path, smoke: bool) -> None:
        self.smoke = smoke
        if smoke:
            self.parent_eval = parent1072.eval_out_dir(base_dir)
            self.npz_dir = self.parent_eval
            self.smoke_ref_dir = self.parent_eval / "smoke_reference"
        else:
            root = parent952.repo_root()
            self.parent_eval = root / "eval_results" / "issue_1072"
            self.npz_dir = _repro_dir(base_dir)
            stats = json.loads(
                (root.joinpath(*parent1072.COMMITTED_KFOLD_DIR) / "stats_kfold.json").read_text()
            )
            self.lambda_tables = stats["manifest_validation"]["lambda_tables_by_fold"]

    def battery_rec(self, fold: int) -> dict:
        p = self.parent_eval / f"battery_1072_fold{fold}.json"
        assert p.exists(), f"completed-round battery record missing: {p}"
        return json.loads(p.read_text())

    def frozen_lambdas(self, fold: int, layer: int) -> dict[str, Any]:
        rec = self.battery_rec(fold)["layers"][str(layer)]
        lam_frozen = rec["components"]["lambda_frozen"]
        out: dict[str, Any] = {
            "cleg_mean": float(lam_frozen["cleg_mean"]),
            "zleg_mean": float(lam_frozen["zleg_mean"]),
            "p_last": float(rec["p_last"]["lambda"]),
        }
        if self.smoke:
            table = json.loads((self.smoke_ref_dir / f"fold{fold}_h1_L{layer}.json").read_text())[
                "lambda_table"
            ]
        else:
            table = self.lambda_tables[str(fold)][str(layer)]
        out["slot_table"] = {s: float(v) for s, v in table.items()}
        for v in (out["cleg_mean"], out["zleg_mean"], out["p_last"], *out["slot_table"].values()):
            assert np.isfinite(v) and v > 0, f"non-finite frozen λ: {out}"
        return out

    def ref_npz(self, fold: int) -> dict[str, np.ndarray]:
        p = self.npz_dir / f"per_context_stats_1072_fold{fold}.npz"
        assert p.exists(), f"reproduction npz missing (run --phase stage first): {p}"
        return dict(np.load(p, allow_pickle=False))


def _g4p_compare_channels(
    name: str,
    computed: np.ndarray,
    ref: np.ndarray,
    mismatches: list[str],
    rel_tol: float = REPRO_REL_TOL,
    pooled_tol: float = REPRO_POOLED_TOL,
) -> None:
    """g4' full-target channel reproduction for one cell (K3).

    Per-context ``ss_res_full``/``ss_tot_full`` at RELATIVE tolerance on the
    parity scale ``max(|ref|, 1)`` — the committed npz stores fp32, so an
    ABSOLUTE 1e-6 bar on raw SS magnitudes is unsatisfiable from storage
    rounding alone — plus pooled ``r2_full`` at ABS 1e-6 (the parent g4
    R²-leaf convention).
    """
    got = np.asarray(computed, dtype=np.float64)
    refa = np.asarray(ref, dtype=np.float64)
    if got.shape != refa.shape:
        mismatches.append(f"{name}: shape {got.shape} != {refa.shape}")
        return
    if (np.isfinite(got) != np.isfinite(refa)).any():
        mismatches.append(f"{name}: finiteness pattern drift")
        return
    for ci, ch_name in ((6, "ss_res_full"), (7, "ss_tot_full")):
        fm = np.isfinite(refa[:, ci])
        if not fm.any():
            continue
        rel = np.abs(got[fm, ci] - refa[fm, ci]) / np.maximum(np.abs(refa[fm, ci]), 1.0)
        if float(rel.max()) > rel_tol:
            mismatches.append(f"{name}/{ch_name}: max rel dev {float(rel.max()):.2e} > {rel_tol}")
    fm = np.isfinite(refa[:, 6]) & np.isfinite(refa[:, 7])
    if fm.any() and float(refa[fm, 7].sum()) > 1e-12:
        r2_ref = 1.0 - float(refa[fm, 6].sum()) / float(refa[fm, 7].sum())
        r2_got = 1.0 - float(got[fm, 6].sum()) / float(got[fm, 7].sum())
        if abs(r2_ref - r2_got) > pooled_tol:
            mismatches.append(
                f"{name}/pooled_r2_full: |{r2_got:.9f} - {r2_ref:.9f}| > {pooled_tol}"
            )


def _slot_tslot(slot: str, span: int) -> int:
    """Answer-index t of a single-position slot's REALIZED next token.

    Convention (parent slot_next_token_ids): slot at sequence position p maps
    to ``next_ext[t]`` with ``t = p - rs + 1``. Derivable from span alone for
    the DECOMP_SLOTS families (f16 / l16 / d10) — cross-checked against the
    stored ``slot_next_ids`` in the battery (fail-loud).
    """
    if slot.startswith("f16_t"):
        return int(slot[5:])
    if slot.startswith("l16_m"):
        return span - int(slot[5:]) + 1
    if slot.startswith("d10_p"):
        pct = int(slot[5:]) / 100.0
        return round(pct * (span - 1)) + 1
    raise ValueError(f"no t_slot rule for slot {slot!r}")


def _battery_fold_layer_lowdim(  # noqa: C901 — the phase-C unit: g4' + 3-basis cells
    base_dir: pathlib.Path,
    fold_split: dict,
    layer: int,
    pool_ids: list[int],
    spans_by_arm: dict[str, np.ndarray],
    u_dir_np: np.ndarray,
    refs: LowdimRefs,
    fit_device: str,
    min_train: int,
    smoke: bool,
    run_parity: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """One (fold, layer) unit: subspace component cells at frozen λ* (c-leg
    slot + remainder, z-leg, p_last — all 3 bases) + g4' full-target
    reproduction vs the committed npz. Returns (npz arrays, unit record)."""
    import torch

    fold_k = int(fold_split["fold"])
    tensors_dir = parent1072._tensors_dir(base_dir)
    td_low = _tensors_dir_lowdim(base_dir)
    t_unit0 = time.time()

    pos_of = {cid: i for i, cid in enumerate(pool_ids)}
    tr_pos = np.asarray([pos_of[c] for c in fold_split["train"] if c in pos_of])
    va_pos = np.asarray([pos_of[c] for c in fold_split["val"] if c in pos_of])
    te_pos = np.asarray([pos_of[c] for c in fold_split["test"] if c in pos_of])
    u16 = np.all(np.stack([spans_by_arm[a] >= T2 + 16 for a in ARMS]), axis=0)
    tr_m = tr_pos[u16[tr_pos]]
    va_m = va_pos[u16[va_pos]]
    te_m = te_pos[u16[te_pos]]
    rec: dict[str, Any] = {
        "fold": fold_k,
        "layer": layer,
        "matched_n": {"train": len(tr_m), "val": len(va_m), "test": len(te_m)},
    }
    if len(tr_m) < min_train or len(te_m) < 2 or len(va_m) < 2:
        rec["skipped"] = True
        return {}, rec

    staged: dict[str, np.ndarray] = {}
    for arm in ARMS:
        slots, sids = parent952._load_layer_slots(base_dir, arm, layer)
        assert [int(i) for i in sids] == [int(i) for i in pool_ids], (arm, layer)
        staged[arm] = slots
    decomp = {
        arm: torch.load(
            str(tensors_dir / f"decomp_{arm}_L{layer}.pt"), map_location="cpu", weights_only=False
        )
        for arm in ARMS
    }
    lowdim = {
        arm: torch.load(
            str(td_low / f"lowdim_{arm}_L{layer}.pt"), map_location="cpu", weights_only=False
        )
        for arm in ARMS
    }
    next_npz = {arm: np.load(tensors_dir / f"next_ids_{arm}.npz") for arm in ARMS}
    top32_npz = {arm: np.load(td_low / f"top32_ids_{arm}.npz") for arm in ARMS}
    for arm in ARMS:
        for store, tag in ((decomp, "decomp"), (lowdim, "lowdim")):
            assert [int(i) for i in store[arm]["ids"]] == [int(i) for i in pool_ids], (
                f"{tag} shard id drift ({arm}, L{layer})"
            )
            # Row coverage (plan §3): every matched row's decomposition is valid.
            assert bool(store[arm]["rem_valid"].numpy()[u16].all()), (
                f"matched row with invalid remainder decomposition ({tag}, {arm}, L{layer})"
            )
        assert top32_npz[arm]["ids"].tolist() == [int(i) for i in pool_ids], (arm, layer)

    lam = refs.frozen_lambdas(fold_k, layer)
    rec["lambda_frozen"] = {k: v for k, v in lam.items() if k != "slot_table"}
    ref_arrs = refs.ref_npz(fold_k)
    ids_test = np.asarray([pool_ids[p] for p in te_m], dtype=np.int64)
    assert np.array_equal(ref_arrs["ids_test"], ids_test), (
        f"fold {fold_k}: test-row identity drift vs the committed npz"
    )

    dev = torch.device(fit_device if (fit_device == "cpu" or torch.cuda.is_available()) else "cpu")
    u_dir_t = torch.from_numpy(u_dir_np).to(dev)
    rows_by_split = {"train": tr_m, "test": te_m}
    x_c = staged["own"][:, SLOT_IDX["c_last"], :]
    g7b = {b: {"n_positions": 0, "n_rank_reduced": 0} for b in BASES}

    def _project_ids(z_np: np.ndarray, ids_pad: np.ndarray, eff_k: np.ndarray | None, basis: str):
        """Chunked batched-QR projection of z rows onto their id-span bases."""
        out = np.empty_like(z_np, dtype=np.float64)
        r_chunk = 1024
        for r0 in range(0, len(z_np), r_chunk):
            r1 = min(r0 + r_chunk, len(z_np))
            ids_t = torch.from_numpy(ids_pad[r0:r1].astype(np.int64)).to(dev)
            eff_t = (
                torch.from_numpy(eff_k[r0:r1].astype(np.int64)).to(dev)
                if eff_k is not None
                else None
            )
            q, _eff, red = sb.orthonormal_bases(u_dir_t, ids_t, eff_k=eff_t)
            g7b[basis]["n_positions"] += r1 - r0
            g7b[basis]["n_rank_reduced"] += len(red)
            z_t = torch.from_numpy(z_np[r0:r1].astype(np.float32)).to(dev)
            out[r0:r1] = sb.project_rows(z_t, q).double().cpu().numpy()
        return out

    def _slot_pair(basis: str, slot: str, arm: str, rows: np.ndarray):
        si = SLOT_IDX[slot]
        y_full = staged[arm][rows][:, si, :].astype(np.float64)
        if basis in ("top8", "top32"):
            ids32 = top32_npz[arm]["slot_top32"][rows, si].astype(np.int64)
            assert (ids32 >= 0).all(), (basis, slot, arm, "matched row without slot candidates")
            ids_pad = ids32[:, : sb.NESTED_K] if basis == "top8" else ids32
            y_par = _project_ids(y_full, ids_pad, None, basis)
        else:  # look8: realized window from the staged next-id store
            spans_r = spans_by_arm[arm][rows]
            t_slot = np.asarray([_slot_tslot(slot, int(s)) for s in spans_r], dtype=np.int64)
            off = next_npz[arm]["next_offsets"][rows]
            flat = next_npz[arm]["next_flat"].astype(np.int64)
            m = np.arange(sb.LOOKAHEAD)[None, :]
            t0s = (t_slot - 1)[:, None]  # index into next_flat rows
            valid = t0s + m <= (spans_r - 2)[:, None]
            idx = np.clip(off[:, None] + t0s + m, 0, len(flat) - 1)
            window = flat[idx]
            # Self-consistency: window[:, 0] must equal the stored slot next id.
            st_nid = next_npz[arm]["slot_next_ids"][rows, si]
            assert (window[:, 0] == st_nid).all(), (slot, arm, "t_slot derivation drift")
            w_t = torch.from_numpy(window).to(dev)
            v_t = torch.from_numpy(valid).to(dev)
            ids_c, eff_c = sb.compact_dedupe_windows(w_t, v_t)
            y_par = _project_ids(y_full, ids_c.cpu().numpy(), eff_c.cpu().numpy(), "look8")
        return y_par, y_full

    def _rem_pair(basis: str, arm: str, rows: np.ndarray):
        y_par = lowdim[arm][f"rem_par_{basis}"].numpy()[rows].astype(np.float64)
        y_full = decomp[arm]["rem_nx"].numpy()[rows].astype(np.float64)
        return y_par, y_full

    # ── c-leg slot cells: 3 bases x 41 slots x 4 arms at per-slot frozen λ ──────
    slot_groups = [(b, s, a) for b in BASES for s in DECOMP_SLOTS for a in ARMS]

    def slot_pair_fn(split: str, gi: int):
        b, s, a = slot_groups[gi]
        return _slot_pair(b, s, a, rows_by_split[split])

    slot_lams = np.asarray([lam["slot_table"][s] for _b, s, _a in slot_groups], dtype=np.float64)
    comp_slots = run_component_cell(
        x_c[tr_m],
        {"test": x_c[te_m]},
        slot_pair_fn,
        [f"{b}|{s}|{a}" for b, s, a in slot_groups],
        slot_lams,
        device=fit_device,
    )

    # ── c-leg remainder cells: 3 bases x 4 arms at frozen λ_cleg + λ sensitivity ─
    rem_groups = [(b, a) for b in BASES for a in ARMS]

    def rem_pair_fn(split: str, gi: int):
        b, a = rem_groups[gi]
        return _rem_pair(b, a, rows_by_split[split])

    comp_rem = run_component_cell(
        x_c[tr_m],
        {"test": x_c[te_m]},
        rem_pair_fn,
        [f"{b}|rem|{a}" for b, a in rem_groups],
        np.full(len(rem_groups), lam["cleg_mean"]),
        device=fit_device,
        sensitivity_lambdas=np.asarray(DEFAULT_LAMBDAS_LIST),
    )

    # ── z-leg cells per matched arm (3 bases each) at frozen λ_zleg ─────────────
    x_slot = parent952.prefix_slot_name(T2)
    comp_z: dict[str, Any] = {}
    for arm in MATCHED_ARMS:
        xa = staged[arm][:, SLOT_IDX[x_slot], :]

        def z_pair_fn(split: str, gi: int, _arm: str = arm):
            return _rem_pair(BASES[gi], _arm, rows_by_split[split])

        comp_z[arm] = run_component_cell(
            xa[tr_m],
            {"test": xa[te_m]},
            z_pair_fn,
            [f"{b}|rem|{arm}" for b in BASES],
            np.full(len(BASES), lam["zleg_mean"]),
            device=fit_device,
        )

    # ── p_last prefix-arm cells (3 bases x 4 arms) at the committed p_last λ ────
    p_x = decomp["own"]["p_last"].numpy().astype(np.float16)
    comp_p = run_component_cell(
        p_x[tr_m],
        {"test": p_x[te_m]},
        rem_pair_fn,
        [f"{b}|plast|{a}" for b, a in rem_groups],
        np.full(len(rem_groups), lam["p_last"]),
        device=fit_device,
    )

    # ── g4': full-target channel reproduction vs the committed npz (K3) ─────────
    mismatches: list[str] = []
    for gi, (b, s, a) in enumerate(slot_groups):
        _g4p_compare_channels(
            f"g4p_fold{fold_k}_L{layer}/{b}|{s}|{a}",
            comp_slots.channels["test"][:, gi, :],
            ref_arrs[f"H1_L{layer}|{s}|{a}"],
            mismatches,
        )
    for gi, (b, a) in enumerate(rem_groups):
        _g4p_compare_channels(
            f"g4p_fold{fold_k}_L{layer}/{b}|rem|{a}",
            comp_rem.channels["test"][:, gi, :],
            ref_arrs[f"M{T2}c_L{layer}|{a}"],
            mismatches,
        )
    for arm in MATCHED_ARMS:
        for gi, b in enumerate(BASES):
            _g4p_compare_channels(
                f"g4p_fold{fold_k}_L{layer}/{b}|zleg|{arm}",
                comp_z[arm].channels["test"][:, gi, :],
                ref_arrs[f"M{T2}z_L{layer}|{arm}"],
                mismatches,
            )
    committed_p = refs.battery_rec(fold_k)["layers"][str(layer)]["p_last"]["component_pooled"][
        "r2_full"
    ]
    for gi, (b, a) in enumerate(rem_groups):
        got = float(comp_p.pooled["test"]["r2_full"][gi])
        want = float(committed_p[a])
        if abs(got - want) > REPRO_POOLED_TOL:
            mismatches.append(
                f"g4p_fold{fold_k}_L{layer}/{b}|plast|{a}: |{got:.9f} - {want:.9f}| > "
                f"{REPRO_POOLED_TOL}"
            )
    if mismatches:
        rec["g4p_mismatches"] = mismatches[:20]
        (eval_out_dir_lowdim(base_dir) / f"g4p_abort_fold{fold_k}_L{layer}.json").write_text(
            json.dumps({"mismatches": mismatches}, indent=2, default=_json_np)
        )
        raise RuntimeError(
            f"g4' full-target reproduction FAIL (K3) at fold {fold_k} L{layer}: "
            f"{len(mismatches)} mismatches, first: {mismatches[0]}"
        )
    rec["g4p"] = {
        "verdict": "PASS",
        "n_cells_compared": len(slot_groups) + len(rem_groups) * 2 + len(MATCHED_ARMS) * 3,
        "rel_tol": REPRO_REL_TOL,
        "pooled_tol": REPRO_POOLED_TOL,
    }
    # g7 battery-side verdict (K1) — same 1% bar as capture.
    for b in BASES:
        npos = max(g7b[b]["n_positions"], 1)
        g7b[b]["reduced_frac"] = g7b[b]["n_rank_reduced"] / npos
        if g7b[b]["reduced_frac"] > RANK_REDUCED_MAX_FRAC:
            raise RuntimeError(f"g7 rank-guard FAIL (K1, battery slot bases): {b} {g7b[b]}")
    rec["g7_battery"] = g7b

    # ── parity: 3 slot cells + one remainder cell per basis vs the fp64 oracle ──
    if run_parity:
        cells = [
            (slot_groups.index(("top32", "f16_t1", "own")), "test"),
            (slot_groups.index(("look8", "l16_m3", "ext_plain")), "test"),
            (slot_groups.index(("top8", "d10_p55", "own")), "test"),
        ]
        rec["parity_slots"] = component_parity_gate(
            x_c[tr_m],
            {"test": x_c[te_m]},
            slot_pair_fn,
            [f"{b}|{s}|{a}" for b, s, a in slot_groups],
            slot_lams,
            comp_slots,
            cells,
        )
        rec["parity_rem_max_rel"] = 0.0
        for b in BASES:
            gi = rem_groups.index((b, "own"))
            ypar_tr, yfull_tr = rem_pair_fn("train", gi)
            ypar_te, yfull_te = rem_pair_fn("test", gi)
            oracle = serial_component_reference(
                x_c[tr_m], x_c[te_m], ypar_tr, yfull_tr, ypar_te, yfull_te, lam["cleg_mean"]
            )
            got = comp_rem.channels["test"][:, gi, :]
            scale = np.maximum(np.abs(oracle), 1.0)
            rel = float(np.max(np.abs(oracle - got) / scale))
            assert rel < 1e-7, f"rem component parity vs oracle ({b}): {rel:.2e}"
            rec["parity_rem_max_rel"] = max(rec["parity_rem_max_rel"], rel)

    # ── pooled tables + npz channels ─────────────────────────────────────────────
    def _pooled_table(res, names: list[str]) -> dict[str, dict[str, float]]:
        return {
            nm: {k: float(res.pooled["test"][k][gi]) for k in res.pooled["test"]}
            for gi, nm in enumerate(names)
        }

    rec["components"] = {
        "cleg_rem": _pooled_table(comp_rem, [f"{b}|{a}" for b, a in rem_groups]),
        "zleg_rem": {
            a: _pooled_table(comp_z[a], [f"{b}|{a}" for b in BASES]) for a in MATCHED_ARMS
        },
        "p_last": _pooled_table(comp_p, [f"{b}|{a}" for b, a in rem_groups]),
        "lambda_frozen": rec["lambda_frozen"],
        "additivity_max_dev": max(
            comp_slots.additivity_max_dev,
            comp_rem.additivity_max_dev,
            comp_p.additivity_max_dev,
            *(comp_z[a].additivity_max_dev for a in MATCHED_ARMS),
        ),
    }
    rec["sens_lambdas"] = [float(v) for v in DEFAULT_LAMBDAS_LIST]
    assert comp_rem.sens_pooled is not None
    rec["cleg_rem_sensitivity"] = {
        f"{b}|{a}": comp_rem.sens_pooled["test"][:, gi, :].tolist()
        for gi, (b, a) in enumerate(rem_groups)
    }

    unit_npz: dict[str, np.ndarray] = {}
    for gi, (b, s, a) in enumerate(slot_groups):
        unit_npz[f"H1b_{b}_L{layer}|{s}|{a}"] = comp_slots.channels["test"][:, gi, :].astype(
            np.float32
        )
    for gi, (b, a) in enumerate(rem_groups):
        unit_npz[f"{b}_M{T2}c_L{layer}|{a}"] = comp_rem.channels["test"][:, gi, :].astype(
            np.float32
        )
        unit_npz[f"{b}_plast_L{layer}|{a}"] = comp_p.channels["test"][:, gi, :].astype(np.float32)
    for arm in MATCHED_ARMS:
        for gi, b in enumerate(BASES):
            unit_npz[f"{b}_M{T2}z_L{layer}|{arm}"] = (
                comp_z[arm].channels["test"][:, gi, :].astype(np.float32)
            )
    rec["wall_s"] = time.time() - t_unit0
    del comp_slots, comp_rem, comp_z, comp_p
    gc.collect()
    return unit_npz, rec


def phase_battery(
    base_dir: pathlib.Path,
    smoke: bool,
    layers: tuple[int, ...],
    fit_device: str,
    skip_upload: bool,
    u_dir_np: np.ndarray | None,
) -> None:
    """Phase C: per-fold (calibration fold FIRST as the K4 pilot) x per-layer
    subspace battery with per-unit checkpoints; per-fold npz + JSON uploaded
    the moment each fold completes (checkpoint-per-phase)."""
    import torch

    log_phase("battery")
    t0 = time.time()
    if u_dir_np is None:
        if smoke:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
            u_dir_np = parent1072._unembed_dir(parent1072._build_smoke_model(tok)).cpu().numpy()
        else:
            u_dir_np = parent1072.load_unembed_dir_production(DEFAULT_MODEL)

    pool_ids = parent1072._pool_ids_from_shard(base_dir, layers[0])
    spans_by_arm = {
        arm: np.asarray(
            [parent952._load_spans(base_dir, arm)[str(c)].get("span", 0) for c in pool_ids],
            dtype=np.int64,
        )
        for arm in ARMS
    }
    refs = LowdimRefs(base_dir, smoke)
    # g5 fold-identity gate (K5) via the parent's reference provider (smoke:
    # compares against the parent battery's OWN smoke_reference records).
    refs1072 = parent1072.ReproRefs(base_dir, smoke)
    folds = parent952.make_kfold_splits(pool_ids, K_FOLDS)
    for f in folds:
        computed = parent952.kfold_split_hashes(f)
        ref = refs1072.fold_hashes(int(f["fold"]), computed)
        if computed != ref:
            raise RuntimeError(
                f"g5 fold-identity FAIL (K5) at fold {f['fold']}: recomputed split hashes "
                "!= parent fold_assignment"
            )
    min_train = 4 if smoke else parent952.MIN_CELL_TRAIN
    regime = {
        "followup": FOLLOWUP_LABEL,
        "smoke": bool(smoke),
        "layers": list(layers),
        "bases": list(BASES),
        "k_folds": K_FOLDS,
        "n_pool": len(pool_ids),
        "pool_sha": hashlib.sha256(json.dumps([int(i) for i in pool_ids]).encode()).hexdigest(),
        "fit_device": fit_device,
        "min_train": min_train,
        "staging_revision": "smoke-synth" if smoke else DECOMP_STORE_REVISION,
        "git_sha": parent952._repo_git_sha(),
    }
    ck_dir = _tensors_dir_lowdim(base_dir) / "battery_lowdim_ckpt"
    reg_p = ck_dir / "regime.json"
    if ck_dir.exists() and reg_p.exists():
        persisted = json.loads(reg_p.read_text())
        if persisted != regime:
            stale = ck_dir.with_name(f"{ck_dir.name}-stale-{int(time.time())}")
            ck_dir.rename(stale)
            logger.warning("[battery] regime mismatch — quarantined stale ckpts to %s", stale)
    ck_dir.mkdir(parents=True, exist_ok=True)
    if not reg_p.exists():
        reg_p.write_text(json.dumps(regime, indent=2, default=_json_np))
    out_dir = eval_out_dir_lowdim(base_dir)

    fold_order = [CAL_FOLD] + [k for k in range(K_FOLDS) if k != CAL_FOLD]
    for oi, k in enumerate(fold_order):
        fold_split = folds[k]
        t_fold0 = time.time()
        fold_npz: dict[str, np.ndarray] = {}
        fold_rec: dict[str, Any] = {
            "fold": k,
            "layers": {},
            "regime": regime,
            "meta": _run_metadata(smoke, layers),
        }
        pos_of = {cid: i for i, cid in enumerate(pool_ids)}
        te_pos = np.asarray([pos_of[c] for c in fold_split["test"]])
        u16 = np.all(np.stack([spans_by_arm[a] >= T2 + 16 for a in ARMS]), axis=0)
        te_m = te_pos[u16[te_pos]]
        fold_npz["ids_test"] = np.asarray([pool_ids[p] for p in te_m], dtype=np.int64)
        fold_npz["ids_pool_full"] = np.asarray(pool_ids, dtype=np.int64)
        for layer in layers:
            ck_npz = ck_dir / f"fold{k}_L{layer}.npz"
            ck_json = ck_dir / f"fold{k}_L{layer}.json"
            if ck_npz.exists() and ck_json.exists():
                fold_npz.update(dict(np.load(ck_npz, allow_pickle=False)))
                fold_rec["layers"][str(layer)] = json.loads(ck_json.read_text())
                logger.info("[battery] SKIP fold %d L%d (unit ckpt present)", k, layer)
                continue
            unit_npz, unit_rec = _battery_fold_layer_lowdim(
                base_dir,
                fold_split,
                layer,
                pool_ids,
                spans_by_arm,
                u_dir_np,
                refs,
                fit_device,
                min_train,
                smoke,
                run_parity=(k == CAL_FOLD and layer == layers[0]),
            )
            np.savez(ck_npz, **unit_npz)
            ck_json.write_text(json.dumps(unit_rec, indent=2, default=_json_np))
            fold_npz.update(unit_npz)
            fold_rec["layers"][str(layer)] = unit_rec
            logger.info("[battery] fold %d L%d done (%.1fs)", k, layer, unit_rec.get("wall_s", 0))
        fold_rec["fold_wall_s"] = time.time() - t_fold0
        npz_path = out_dir / f"per_context_stats_lowdim_fold{k}.npz"
        np.savez(npz_path, **fold_npz)
        json_path = out_dir / f"battery_lowdim_fold{k}.json"
        json_path.write_text(json.dumps(fold_rec, indent=2, default=_json_np))
        if not skip_upload:
            _hf_commit_files_lowdim(
                f"battery fold {k}",
                [
                    (npz_path, f"eval_results/{npz_path.name}"),
                    (json_path, f"eval_results/{json_path.name}"),
                ],
            )
        if oi == 0:
            _pilot_check(
                "battery",
                time.time() - t_fold0,
                units_done=1,
                units_total=K_FOLDS,
                booked_h=BATTERY_BOOKED_H,
                base_dir=base_dir,
                smoke=smoke,
                execution_shape=(
                    f"per-(fold,layer) shared-SVD 3-basis target-group stack (device={fit_device})"
                ),
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("[battery] all folds done in %.1f min", (time.time() - t0) / 60)
    log_phase("battery_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Terminal upload + sentinel + CLI
# ═══════════════════════════════════════════════════════════════════════════════


def terminal_upload(base_dir: pathlib.Path) -> None:
    """Terminal commit: every lowdim eval-dir JSON/npz + the workload log."""
    out_dir = eval_out_dir_lowdim(base_dir)
    pairs: list[tuple[pathlib.Path, str]] = [
        (p, f"eval_results/{p.name}")
        for p in sorted(out_dir.glob("*.json")) + sorted(out_dir.glob("*.npz"))
    ]
    log_src = pathlib.Path(os.environ.get("EPS_LOG_PATH") or "/workspace/logs/issue-1072.log")
    if log_src.is_file():
        import shutil

        log_dest = base_dir / "logs" / "issue-1072-lowdim-workload.log"
        log_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log_src, log_dest)
        pairs.append((log_dest, f"logs/{log_dest.name}"))
    else:
        logger.warning("workload log not found at %s — upload leg skipped", log_src)
    _hf_commit_files_lowdim("terminal eval-results + workload log", pairs)


def write_final_sentinel_lowdim(base_dir: pathlib.Path, smoke: bool, wall_h: float) -> None:
    """epm:results sentinel (poll_pipeline contract; SKILL.md Step 7 payload keys)."""
    out_dir = eval_out_dir_lowdim(base_dir)
    eval_numbers: dict[str, Any] = {
        "note": "lowdim subspace stats run VM-side (phase D: scripts/issue1072_lowdim_stats.py)"
    }
    cal = out_dir / f"battery_lowdim_fold{CAL_FOLD}.json"
    if cal.exists():
        try:
            rec = json.loads(cal.read_text())
            l_key = str(max(int(x) for x in rec["layers"]))
            eval_numbers["cal_fold_cleg_rem_components"] = (
                rec["layers"][l_key].get("components", {}).get("cleg_rem")
            )
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning("cal-fold preview unavailable for the sentinel: %s", e)
    payload = {
        "eval_numbers": eval_numbers,
        "eval_paths": [str(out_dir)],
        "reproducibility_card": {
            "hf_data_repo": HF_DATA_REPO,
            "issue_slug": ISSUE_SLUG,
            "followup_label": FOLLOWUP_LABEL,
            "analysis_tensors_prefix": f"{ISSUE_SLUG}/analysis_tensors/",
            "eval_results_prefix": f"{ISSUE_SLUG}/eval_results/",
            "parent_tensor_revision": PARENT_TENSOR_REVISION,
            "decomp_store_revision": DECOMP_STORE_REVISION,
            "model": DEFAULT_MODEL,
            "seeds": {"split": 952, "bootstrap": 0, "signflip": 1},
        },
        "wandb_url": "n/a (no model training in this experiment)",
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{ISSUE_SLUG}",
        "worktree_path": "n/a (pod-side)",
        "final_commit_sha": parent952._repo_git_sha(),
        "gpu_hours_used": wall_h,
        "gpu_hours_budgeted": 6,
        "plan_deviations": [],
    }
    write_sentinel(
        pathlib.Path(f"/workspace/logs/issue-1072-epm_results-{int(time.time())}.json"),
        {
            "kind": "epm:results",
            "version": 1,
            "note": json.dumps(
                {
                    "status": "complete",
                    "smoke": smoke,
                    "issue": ISSUE,
                    "followup_label": FOLLOWUP_LABEL,
                    **payload,
                },
                default=_json_np,
            ),
        },
    )


def verify_deferred_imports() -> int:
    """AST-walk this round's files and EXECUTE every deferred import
    (the #606/#1332 lazy-import gate; hand-maintained lists re-create drift)."""
    here = pathlib.Path(__file__).resolve()
    root = parent952.repo_root()
    parent952._ensure_repo_root_on_syspath()
    files = [
        here,
        here.parent / "subspace_basis.py",
        root / "scripts" / "issue1072_lowdim_stats.py",
        root / "scripts" / "issue1072_lowdim_figures.py",
    ]
    n_ok = 0
    failures: list[str] = []
    for f in files:
        tree = ast.parse(f.read_text())
        deferred: list[ast.Import | ast.ImportFrom] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Import | ast.ImportFrom):
                        deferred.append(sub)
        for node in deferred:
            try:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        importlib.import_module(alias.name)
                        n_ok += 1
                else:
                    assert node.module is not None and node.level == 0, ast.dump(node)
                    mod = importlib.import_module(node.module)
                    for alias in node.names:
                        if not hasattr(mod, alias.name):
                            importlib.import_module(f"{node.module}.{alias.name}")
                        getattr(mod, alias.name)
                        n_ok += 1
            except Exception as e:
                failures.append(f"{f.name}:{node.lineno}: {ast.dump(node)} -> {e!r}")
    print(json.dumps({"deferred_imports_ok": n_ok, "failures": failures}))
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issue #1072 lowdim-token-subspace pod-side driver")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny-real run: 10 synth contexts, from-config 2-layer Qwen2 over the "
        "real tokenizer, layers 0,1; the parent producers build the staged substrate",
    )
    p.add_argument(
        "--phase",
        type=str,
        default="all",
        help="comma-separated subset of stage,capture,battery — or 'all'",
    )
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="capture batch size (default: 8 production, 4 smoke)",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="comma-separated hook layers (default: 14,26; smoke: 0,1)",
    )
    p.add_argument("--fit-device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument(
        "--verify-imports",
        action="store_true",
        help="execute every deferred import (AST-walked) and exit",
    )
    return p.parse_args()


def main() -> None:
    """Phase dispatcher — smoke IS the production path at tiny n (PASS_UNIFIED)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    if args.verify_imports:
        sys.exit(verify_deferred_imports())
    import torch

    base_dir = parent952.resolve_base_dir(args.base_dir)
    smoke = bool(args.smoke)
    layers = (
        tuple(int(x) for x in args.layers.split(",") if x.strip())
        if args.layers
        else ((0, 1) if smoke else PROD_LAYERS)
    )
    batch_size = args.batch_size or (4 if smoke else 8)
    fit_device = args.fit_device or ("cuda" if torch.cuda.is_available() else "cpu")
    phases = (
        ["stage", "capture", "battery"]
        if args.phase == "all"
        else [x.strip() for x in args.phase.split(",") if x.strip()]
    )
    assert set(phases) <= {"stage", "capture", "battery"}, phases
    logger.info(
        "[main] followup=%s phases=%s smoke=%s layers=%s batch=%d fit_device=%s base_dir=%s",
        FOLLOWUP_LABEL,
        phases,
        smoke,
        layers,
        batch_size,
        fit_device,
        base_dir,
    )
    t0 = time.time()
    u_dir_np: np.ndarray | None = None
    if "stage" in phases:
        phase_stage(base_dir, smoke, layers, fit_device)
    if "capture" in phases:
        u_dir_np = phase_capture(base_dir, smoke, layers, batch_size, args.skip_upload)
    if "battery" in phases:
        phase_battery(base_dir, smoke, layers, fit_device, args.skip_upload, u_dir_np)
    if not args.skip_upload:
        terminal_upload(base_dir)
    write_final_sentinel_lowdim(base_dir, smoke, wall_h=(time.time() - t0) / 3600.0)
    log_phase("done")


if __name__ == "__main__":
    main()
