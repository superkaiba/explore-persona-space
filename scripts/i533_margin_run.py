#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, Δ, log Z, − minus) in scientific docstrings + logs.
"""Issue #533 follow-up — EOS-margin logit re-read of the persisted #547 LoRAs.

Re-read of the SAME 180 #547 LoRA adapters (548 cell × eval-encoding units;
HF persists 148 + WandB Artifact 32) in the non-saturating logit-margin
space Δ(z_marker − z_eos) trained − base, instead of the post-softmax
log-prob the parent used. Hypothesis: villain's late-step paired
role-vs-system gap shrink in log-prob space is floor compression, and
the gap persists negative in margin space at 120 steps. ZERO new
training; eval-only.

The forward-pass machinery is forked from ``scripts/issue531_logit_rescore.py``
(which already captures ``z_eos`` at id 151645); the prompt-construction
and adapter-resolution conventions come from ``scripts/i464_po_eval.py``
on branch ``origin/issue-547`` (the rig that produced the 540 #547 per-cell
JSONs we validate against). Cherry-picked helper files (``i464_encodings``,
``i464_data``, ``i464_phase4_eval``, ``i464_po_eval``) live in this branch's
worktree alongside this driver and carry their original docstrings; this
script imports from them directly.

Storage contract (per slot per side: trained AND base):
    log P(marker), z_marker, z_eos, logZ

Phases (smoke = sweep with one cell, PASS_UNIFIED — same code path, same
env-driven overrides ``EVAL_CELLS_OVERRIDE`` / ``EVAL_ENCODINGS_OVERRIDE``):

    1. adapter-manifest — dual-path resolution of the full 180 (cell,
       arm, seed, persona, steps) grid: HF Hub for the 148 dirs uploaded
       successfully + WandB Artifact ``i547-missing-adapters:v0`` for the
       32 role-arm cells whose HF upload was blocked by the public-storage
       quota (#547 incident). Writes a per-run manifest.
    2. r-reconstruction — load ``R_canon_test.json`` at the pinned
       data-repo revision (``i464_data.DATA_REVISION``). The R_canon
       artifact is base-model-greedy under each eval encoding, already
       persisted; "reconstruction" = deterministic download at the pin
       (NO new generations). Validates schema_version +
       per-persona × q_test coverage.
    3. margin-scoring — for each cell × eval-encoding, build the post-R
       teacher-forced probe (``_build_probes_for_eval_marker``, shared
       pirate marker ``id=83399``), then a SINGLE forward pass on the
       (base + LoRA) model. Capture the four floats per slot per side
       (trained = adapter active; base = adapter disabled on same
       batch construction, PEFT hot-swap). Validate recomputed
       ``z_marker − logZ`` against the stored per-question g/b log-probs
       from the #547 per-cell JSON; fail loud on disagreement past
       bf16/engine noise. Checkpoint per cell × encoding (one atomic
       JSON write).
    4. analysis — paired role-vs-system EOS-margin gap at every step
       and at the headline steps {30, 60, 120}, with per-seed-paired
       bootstrap (same shape as the parent analysis). Includes the
       built-in rig check: at s=30 (off-saturation) the logit-margin
       gap must agree with the log-prob gap.

Outputs land under ``eval_results/issue_533/logit-margin-reread/``:
    per_cell/<cell-label>__<e_eval>.json           # phase 3 atomic write
    adapter_manifest.json                          # phase 1
    analysis.json                                  # phase 4

CLI::

    # production (1× H100)
    nohup bash scripts/i533_margin_run.sh > /workspace/logs/issue-533-margin-run.log 2>&1 &

    # smoke (same script, one cell × one encoding via env overrides)
    EVAL_CELLS_OVERRIDE='role_seed42_cn_pirate_s30' \\
      EVAL_ENCODINGS_OVERRIDE='role_pirate' \\
      bash scripts/i533_margin_run.sh

    # direct python driver (single phase, e.g. for dispatch-dry-run)
    uv run python scripts/i533_margin_run.py --phase adapter-manifest
    uv run python scripts/i533_margin_run.py --phase r-reconstruction
    uv run python scripts/i533_margin_run.py --phase margin-scoring --shard 0/1
    uv run python scripts/i533_margin_run.py --phase analysis
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Cherry-picked from origin/issue-547 (see module docstring): these
# helpers DEFINE the #547 eval contract — prompt construction, R_canon
# coverage assertion, log-prob extraction at the post-R slot, log-prob
# floor. Import provenance is documented at each call site.
from i464_phase4_eval import (  # type: ignore[import-not-found]  # noqa: E402
    BASE_MODEL,
    HF_MODEL_REPO,
    HF_R_PATH_PREFIX,
    LOCAL_DATA_DIR,
    LOGP_FLOOR,
    _build_probes_for_eval_marker,
    assert_r_canon_test_coverage,
)
from i464_po_eval import (  # type: ignore[import-not-found]  # noqa: E402
    MAX_STEPS_I547,
    PO_ARMS,
    SEEDS_FOR,
    SHARED_MARKER_PERSONA,
    _eval_encodings_for_cell,
)

from explore_persona_space.experiments import i464_encodings as enc  # noqa: E402
from explore_persona_space.experiments.i464_data import (  # noqa: E402
    DATA_REVISION,
    HF_DATA_REPO,
    load_q_test_extended_50,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────

# EOS for Qwen-2.5-7B-Instruct: <|im_end|>, the token contrastive negatives
# train at the post-response slot. Asserted at module-import in main().
EOS_TOKEN = "<|im_end|>"
EOS_ID = 151645

# Match the marker contract; asserted in main().
MARKER_TEXT = enc.MARKER_PIRATE_TEXT  # " ※"
MARKER_ID = enc.MARKER_PIRATE_ID  # 83399

# #547 per-cell JSON dir on main (validation target).
I547_PER_CELL_DIR = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_547"
    / "contrastive_negatives"
    / "cross_eval"
    / "per_cell"
)

# Output layout. ``per_cell`` is the checkpoint-per-phase dir; the manifest
# and analysis JSONs sit alongside it.
OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "issue_533" / "logit-margin-reread"
PER_CELL_OUT_DIR = OUTPUT_DIR / "per_cell"
MANIFEST_OUT_PATH = OUTPUT_DIR / "adapter_manifest.json"
ANALYSIS_OUT_PATH = OUTPUT_DIR / "analysis.json"

# Validation gates against the #547 stored per-question log-probs. The
# stored values are floored at ``LOGP_FLOOR = -50.0``; a wrong prompt /
# slot construction or a swapped adapter produces tens-of-nats mismatch.
# bf16 + PEFT-runtime vs vLLM-merged noise stays well under 1 nat MAE.
MAX_VALIDATION_MAE_NATS = 1.0
MIN_VALIDATION_SPEARMAN = 0.995

# Headline + rig-check anchors (spec): paired EOS-margin gap at steps
# {30, 60, 120}; rig check at s=30 (off-saturation, Δlog Z ≈ 0).
HEADLINE_STEPS: tuple[int, ...] = (30, 60, 120)
RIG_CHECK_STEP = 30
RIG_CHECK_MAX_NATS_DELTA = 1.0  # |Δlog P(marker) − Δz_marker| at s=30

# Dual-path adapter sources.
WANDB_PROJECT = "explore-persona-space"
WANDB_ARTIFACT_NAME = "i547-missing-adapters"
WANDB_ARTIFACT_TAG = "v0"
HF_ADAPTER_PREFIX = "adapters"  # adapters/{i547_cell_dir}/...


# Local cache for downloaded adapters. On the pod the production launcher
# (i533_margin_run.sh) sets HF_HOME to /workspace; we cache adapters on
# the local FS for fast PEFT hot-swap reload.
def _default_adapter_cache_root() -> Path:
    """Resolve adapter cache root from env override or pod-vs-VM context.

    Note: Path("") evaluates to Path(".") which is truthy — so we can't
    use a simple ``Path(env_var or default)`` here; explicit None-check.
    """
    override = os.environ.get("EPM_I533_MARGIN_ADAPTER_CACHE")
    if override:
        return Path(override)
    if Path("/workspace").exists():
        return Path("/workspace/adapters/i533_margin")
    return PROJECT_ROOT / ".cache" / "i533_margin_adapters"


ADAPTER_CACHE_ROOT = _default_adapter_cache_root()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("i533_margin")


# ── Grid + cell-label helpers ─────────────────────────────────────────────


def _i547_cell_label(arm: str, seed: int, persona: str, steps: int) -> str:
    """The #547 per-cell label used in HF subpath + per-cell JSON name."""
    return f"{arm}_seed{seed}_cn_{persona}_s{steps}"


def _hf_subpath_for(arm: str, seed: int, persona: str, steps: int) -> str:
    """HF Hub subpath where the cell's LoRA dir lives (when uploaded)."""
    return f"{HF_ADAPTER_PREFIX}/i547_{_i547_cell_label(arm, seed, persona, steps)}"


def list_grid() -> list[tuple[str, int, str, int]]:
    """Enumerate all 180 (arm, seed, persona, steps) cells deterministically."""
    seeds = SEEDS_FOR["cn_i547"]
    return [
        (arm, seed, persona, steps)
        for arm in PO_ARMS
        for seed in seeds
        for persona in enc.PERSONAS
        for steps in MAX_STEPS_I547
    ]


# ── Phase 1: adapter-manifest (dual-path) ─────────────────────────────────


def _hf_adapter_set() -> set[str]:
    """Return the set of i547_* dir names on HF that have adapter_model.safetensors.

    Per CLAUDE.md upload-policy: use ``huggingface_hub.list_repo_files``
    (Hub API), NOT the ``hf`` CLI (which silently truncates on large
    listings).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(HF_MODEL_REPO)
    pat = re.compile(rf"{HF_ADAPTER_PREFIX}/(i547_[^/]+)/adapter_model\.safetensors$")
    return {m.group(1) for f in files if (m := pat.match(f))}


def _wandb_artifact_entries() -> set[str]:
    """Return the set of i547_* top-level subdirs present in the WandB rescue artifact.

    Verified at planning time: ``i547-missing-adapters:v0`` is COMMITTED and
    has 32 distinct top-level dirs (all role-arm cells).
    """
    import wandb

    api = wandb.Api()
    art = api.artifact(
        f"thomasjiralerspong/{WANDB_PROJECT}/{WANDB_ARTIFACT_NAME}:{WANDB_ARTIFACT_TAG}",
        type="model",
    )
    if art.state != "COMMITTED":
        raise RuntimeError(
            f"WandB artifact {WANDB_ARTIFACT_NAME}:{WANDB_ARTIFACT_TAG} is "
            f"state={art.state!r}; need COMMITTED."
        )
    entries = list(art.manifest.entries.keys())
    return {e.split("/")[0] for e in entries}


def build_adapter_manifest(grid: list[tuple[str, int, str, int]]) -> dict:
    """Map each of the 180 cells to its source: 'hf' or 'wandb'.

    Fails loud if a cell is unresolvable from either source.
    """
    hf_set = _hf_adapter_set()
    wandb_set = _wandb_artifact_entries()
    log.info("adapter-manifest: HF has %d i547_* dirs; WandB has %d", len(hf_set), len(wandb_set))
    rows: list[dict] = []
    unresolved: list[str] = []
    for arm, seed, persona, steps in grid:
        hf_dir = f"i547_{_i547_cell_label(arm, seed, persona, steps)}"
        if hf_dir in hf_set:
            source = "hf"
        elif hf_dir in wandb_set:
            source = "wandb"
        else:
            source = "missing"
            unresolved.append(hf_dir)
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "persona": persona,
                "steps": steps,
                "i547_dir": hf_dir,
                "source": source,
            }
        )
    if unresolved:
        raise RuntimeError(
            f"adapter-manifest: {len(unresolved)} cells unresolved from both HF and WandB. "
            f"First 5: {unresolved[:5]}"
        )
    counts = {"hf": 0, "wandb": 0}
    for r in rows:
        counts[r["source"]] += 1
    log.info("adapter-manifest resolution: hf=%d wandb=%d", counts["hf"], counts["wandb"])
    return {
        "schema_version": "i533_margin_manifest_v1",
        "task_id": 533,
        "followup_label": "logit-margin-reread",
        "n_cells": len(rows),
        "n_from_hf": counts["hf"],
        "n_from_wandb": counts["wandb"],
        "hf_model_repo": HF_MODEL_REPO,
        "wandb_artifact": (
            f"thomasjiralerspong/{WANDB_PROJECT}/{WANDB_ARTIFACT_NAME}:{WANDB_ARTIFACT_TAG}"
        ),
        "data_repo": HF_DATA_REPO,
        "data_revision": DATA_REVISION,
        "resolved_at_utc": datetime.now(UTC).isoformat(),
        "rows": rows,
    }


def phase_adapter_manifest(write: bool = True) -> dict:
    """Run phase 1 — resolve all 180 cells, optionally write the manifest JSON."""
    log.info("[phase=adapter_manifest] start")
    grid = list_grid()
    manifest = build_adapter_manifest(grid)
    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST_OUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(MANIFEST_OUT_PATH)
        log.info("[phase=adapter_manifest] wrote %s", MANIFEST_OUT_PATH)
    log.info("[phase=adapter_manifest] ok")
    return manifest


# ── Phase 2: r-reconstruction (load R_canon_test at the pin) ──────────────


def _load_R_canon_test_at_pin() -> dict[str, dict[str, dict]]:
    """Load R_canon_test.json at ``DATA_REVISION``.

    Mirrors ``scripts/i464_po_eval._load_R_canon_test`` but ignores the
    ``EPM_LOCAL_R_CANON_DIR`` override (the spec pins R to the data-repo
    revision; a local override would defeat reproducibility for this
    re-read).
    """
    from huggingface_hub import hf_hub_download

    local = LOCAL_DATA_DIR / "R_canon_test.json"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_canon_test.json",
            revision=DATA_REVISION,
        )
        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    schema = payload.get("schema_version")
    if schema != "i464_v2_matched_R":
        raise AssertionError(f"R_canon_test schema_version={schema!r}; want i464_v2_matched_R")
    return payload["completions"]


def phase_r_reconstruction() -> dict:
    """Run phase 2 — load + verify R_canon_test at the pin."""
    log.info("[phase=r_reconstruction] start (pin=%s)", DATA_REVISION)
    q_test = load_q_test_extended_50()
    R = _load_R_canon_test_at_pin()
    assert_r_canon_test_coverage(R, q_test, list(enc.PERSONAS))
    rows = sum(len(R[p]) for p in R)
    log.info(
        "[phase=r_reconstruction] ok personas=%s q_test=%d total_rows=%d",
        sorted(R.keys()),
        len(q_test),
        rows,
    )
    return {"q_test": q_test, "R_canon_test": R}


# ── Phase 3: margin-scoring ───────────────────────────────────────────────


def _materialize_adapter(row: dict) -> Path:
    """Materialize one cell's LoRA dir locally; return its Path.

    HF source: per-file ``hf_hub_download`` (CLAUDE.md anti-truncation
    rule, mirrors ``issue531_logit_rescore.download_adapter``). Pull only
    the two adapter files we actually need (``adapter_config.json`` +
    ``adapter_model.safetensors``); skip optional tokenizer artifacts.

    WandB source: fetch the artifact subdir once per run via
    ``artifact.get_path(name).download``. The same artifact ref is reused
    across cells (cheap — wandb caches under WANDB_CACHE_DIR).
    """
    ADAPTER_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    local_dir = ADAPTER_CACHE_ROOT / row["i547_dir"]
    sentinel = local_dir / "adapter_model.safetensors"
    if sentinel.exists():
        # A cached adapter (intra-pod resume) must pass the gauge assert
        # too, not just fresh downloads.
        _assert_gauge_free(local_dir)
        return local_dir

    if row["source"] == "hf":
        from huggingface_hub import hf_hub_download

        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                revision="main",
                filename=f"{HF_ADAPTER_PREFIX}/{row['i547_dir']}/{fname}",
                local_dir=ADAPTER_CACHE_ROOT,
            )
        # hf_hub_download materializes under ADAPTER_CACHE_ROOT /
        # adapters/<i547_dir>/...; flatten to ADAPTER_CACHE_ROOT/<i547_dir>/
        flat_src = ADAPTER_CACHE_ROOT / HF_ADAPTER_PREFIX / row["i547_dir"]
        if flat_src.exists() and flat_src != local_dir:
            local_dir.mkdir(parents=True, exist_ok=True)
            for f in flat_src.iterdir():
                target = local_dir / f.name
                if not target.exists():
                    shutil.move(str(f), str(target))
    elif row["source"] == "wandb":
        import wandb

        api = wandb.Api()
        art = api.artifact(
            f"thomasjiralerspong/{WANDB_PROJECT}/{WANDB_ARTIFACT_NAME}:{WANDB_ARTIFACT_TAG}",
            type="model",
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            entry = art.get_path(f"{row['i547_dir']}/{fname}")
            downloaded = Path(entry.download(root=str(ADAPTER_CACHE_ROOT)))
            target = local_dir / fname
            if downloaded.resolve() != target.resolve():
                shutil.copyfile(downloaded, target)
    else:
        raise RuntimeError(f"unknown source={row['source']!r} for {row['i547_dir']!r}")

    if not sentinel.exists():
        raise RuntimeError(f"materialize: adapter_model.safetensors missing at {sentinel}")
    _assert_gauge_free(local_dir)
    return local_dir


def _assert_gauge_free(adapter_dir: Path) -> None:
    """Same gauge assert as issue531_logit_rescore: LoRA must not touch the unembedding.

    The trained − base ``z_marker`` (logit) read is valid only when LoRA
    targets attention projections (not ``lm_head`` / ``embed_tokens``).
    Refuse to score otherwise.
    """
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    targets = cfg.get("target_modules") or []
    banned = {"lm_head", "embed_tokens"}
    hit = banned.intersection(targets) if isinstance(targets, list) else set()
    if hit:
        raise RuntimeError(
            f"{adapter_dir}: adapter targets {sorted(hit)} — logit readout is "
            f"gauge-dependent and INVALID for this run"
        )
    saved = cfg.get("modules_to_save") or []
    if saved:
        raise RuntimeError(
            f"{adapter_dir}: modules_to_save={saved!r} non-empty — full-module "
            f"saves can move the unembedding; logit readout invalid"
        )


def _score_batch_slots(
    model,
    tokenizer,
    prompts_payload: list[dict],
    slot_positions: list[int],
    device: str,
    batch_size: int,
) -> list[dict]:
    """Score the (already constructed) post-R slot per probe with the CURRENT model state.

    Captures the four-float storage contract per slot: ``log P(marker)``,
    ``z_marker``, ``z_eos``, ``logZ`` (plus ``argmax_id`` for diagnostics).
    Left-pads within batch; the readout is at the per-row stored slot
    (NOT ``logits[:, -1, :]`` — slots are per-row because lengths differ).
    """
    import torch
    import torch.nn.functional as F

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Build (tokens, slot_in_padded_row) pairs.
    encoded: list[tuple[int, list[int]]] = []
    for idx, payload in enumerate(prompts_payload):
        ids = payload["prompt_token_ids"]
        # _build_probes_for_eval_marker guarantees ids[-1] == marker_id and
        # slot_positions[idx] == len(ids) - 1. Validate; we depend on it.
        if slot_positions[idx] != len(ids) - 1:
            raise RuntimeError(
                f"slot drift idx={idx}: stored slot={slot_positions[idx]} != "
                f"len(ids)-1={len(ids) - 1}"
            )
        if ids[-1] != MARKER_ID:
            raise RuntimeError(
                f"slot construction wrong idx={idx}: last id={ids[-1]} != marker {MARKER_ID}"
            )
        encoded.append((idx, list(ids)))

    # Length-sort to cut padding waste; numerically irrelevant to the
    # per-row at-slot readout.
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][1]))

    out_by_idx: dict[int, dict[str, float]] = {}
    for start in range(0, len(order), batch_size):
        chunk = [encoded[i] for i in order[start : start + batch_size]]
        max_len = max(len(ids) for _, ids in chunk)
        padded = [[pad_id] * (max_len - len(ids)) + ids for _, ids in chunk]
        attn = [[0] * (max_len - len(ids)) + [1] * len(ids) for _, ids in chunk]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # The post-R (marker) slot is the LAST non-pad token. With left-pad
        # + add_special_tokens=False (the probe-construction contract), the
        # token at the slot is the marker, and the logit we want is the
        # distribution that PREDICTS it — i.e. the logits at position
        # (max_len - 2) (one before the marker). _extract_marker_logp in
        # vLLM does this via prompt_logprobs[L]; for the HF causal-LM
        # forward, ``logits[:, t, :]`` is the next-token distribution
        # conditioned on tokens[0..t]. We want P(marker | prefix), where
        # the prefix is everything BEFORE the marker -> position L-1 in
        # the un-padded ids, or (max_len - 2) in the left-padded row.
        # Tensor shape check + assertion that the next-token argmax is
        # consistent across the batch (sanity).
        assert logits.shape[0] == len(chunk) and logits.shape[1] == max_len, logits.shape
        # Slot = max_len - 2 (one step before the marker, which sits at
        # max_len - 1 because of left-pad to a uniform row length).
        slot_logits = logits[:, max_len - 2, :].float()  # (B, V)
        logz = torch.logsumexp(slot_logits, dim=-1)
        z_marker = slot_logits[:, MARKER_ID]
        z_eos = slot_logits[:, EOS_ID]
        argmax_ids = slot_logits.argmax(dim=-1)
        logp_marker = F.log_softmax(slot_logits, dim=-1)[:, MARKER_ID]

        for (orig_idx, _), zm, ze, lz, lp, am in zip(
            chunk,
            z_marker.cpu().tolist(),
            z_eos.cpu().tolist(),
            logz.cpu().tolist(),
            logp_marker.cpu().tolist(),
            argmax_ids.cpu().tolist(),
            strict=True,
        ):
            out_by_idx[orig_idx] = {
                "z_marker": float(zm),
                "z_eos": float(ze),
                "logZ": float(lz),
                "logp": float(lp),
                "argmax_id": int(am),
            }
        del logits, slot_logits, logz, z_marker, z_eos, argmax_ids, logp_marker

    # Return rows in the input order.
    return [out_by_idx[i] for i in range(len(prompts_payload))]


def _validate_against_stored(
    cell_label: str,
    e_eval: str,
    trained_rows: list[dict],
    base_rows: list[dict],
) -> dict:
    """Compare recomputed ``log P(marker)`` to the #547 stored per-question values.

    Per the spec, the recomputed ``z_marker − logZ`` must match
    ``g_logps_per_q`` (trained) / ``b_logps_per_q`` (base) up to bf16 /
    engine noise. Failure here = wrong prompts / slot / adapter, not
    just precision.

    Stored values are floored at ``LOGP_FLOOR``; we apply the same floor
    to our recomputed values BEFORE the diff so the comparison is
    apples-to-apples (a row that hit the floor in storage will show up as
    a saturated agreement, not a 10-nat error).
    """
    import numpy as np
    from scipy.stats import spearmanr

    stored_path = I547_PER_CELL_DIR / f"{cell_label}__{e_eval}.json"
    if not stored_path.exists():
        raise RuntimeError(f"#547 validation target missing: {stored_path}")
    stored = json.loads(stored_path.read_text())
    n = stored["n_probes"]
    if len(trained_rows) != n or len(base_rows) != n:
        raise RuntimeError(
            f"[{cell_label}__{e_eval}] row-count mismatch: stored n={n} "
            f"trained={len(trained_rows)} base={len(base_rows)}"
        )

    res: dict[str, dict] = {}
    for side, rows, stored_key in (
        ("trained", trained_rows, "g_logps_per_q"),
        ("base", base_rows, "b_logps_per_q"),
    ):
        s = np.asarray(stored[stored_key], dtype=float)
        r_raw = np.asarray([row["logp"] for row in rows], dtype=float)
        r = np.maximum(r_raw, LOGP_FLOOR)  # apply the stored floor to our recompute
        mae = float(np.mean(np.abs(s - r)))
        max_abs = float(np.max(np.abs(s - r)))
        rho_stat = spearmanr(s, r).statistic
        rho = float(rho_stat) if rho_stat == rho_stat else 1.0  # NaN guard (degenerate ties)
        res[side] = {
            "mae_nats": mae,
            "max_abs_nats": max_abs,
            "spearman": rho,
            "n": int(n),
            "stored_mean": float(s.mean()),
            "recomputed_mean": float(r.mean()),
        }
        log.info(
            "[%s__%s] validation %s: MAE=%.4f nats, max=%.4f, spearman=%.5f (n=%d)",
            cell_label,
            e_eval,
            side,
            mae,
            max_abs,
            rho,
            n,
        )
        if mae > MAX_VALIDATION_MAE_NATS or rho < MIN_VALIDATION_SPEARMAN:
            raise RuntimeError(
                f"[{cell_label}__{e_eval}] {side} validation FAILED: "
                f"MAE={mae:.4f} nats (gate {MAX_VALIDATION_MAE_NATS}), "
                f"spearman={rho:.5f} (gate {MIN_VALIDATION_SPEARMAN}) — "
                f"prompt/slot/adapter construction likely diverges from #547's scorer"
            )
    return res


def _process_cell_encoding(
    peft_model,
    tokenizer,
    arm: str,
    seed: int,
    persona: str,
    steps: int,
    e_eval: str,
    R_canon_test: dict,
    q_test: list[str],
    device: str,
    batch_size: int,
    manifest_row: dict,
) -> None:
    """Score ONE (cell, eval-encoding) pair end-to-end + write atomic JSON."""
    cell_label = _i547_cell_label(arm, seed, persona, steps)
    out_path = PER_CELL_OUT_DIR / f"{cell_label}__{e_eval}.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("[%s__%s] resume: output exists — skipping", cell_label, e_eval)
        return

    # 1) Build the probes (vLLM payload format; we use just prompt_token_ids).
    prompts, slots = _build_probes_for_eval_marker(
        e_eval, SHARED_MARKER_PERSONA, tokenizer, q_test, R_canon_test
    )
    assert len(prompts) == len(slots) == len(q_test), (
        len(prompts),
        len(slots),
        len(q_test),
    )

    # 2) Materialize + hot-swap the adapter, score trained side.
    adapter_dir = _materialize_adapter(manifest_row)
    adapter_name = f"{cell_label}_a"
    peft_model.load_adapter(str(adapter_dir), adapter_name=adapter_name)
    peft_model.set_adapter(adapter_name)
    t0 = time.time()
    trained_rows = _score_batch_slots(peft_model, tokenizer, prompts, slots, device, batch_size)
    t_trained = time.time() - t0

    # 3) Score base side via PEFT disable_adapter.
    t0 = time.time()
    with peft_model.disable_adapter():
        base_rows = _score_batch_slots(peft_model, tokenizer, prompts, slots, device, batch_size)
    t_base = time.time() - t0

    peft_model.delete_adapter(adapter_name)
    gc.collect()

    # 4) Validate against the #547 stored per-cell JSON (fail loud).
    validation = _validate_against_stored(cell_label, e_eval, trained_rows, base_rows)

    payload = {
        "cell": cell_label,
        "arm": arm,
        "seed": seed,
        "training_persona": persona,
        "marker_persona": SHARED_MARKER_PERSONA,
        "e_eval": e_eval,
        "marker_id": MARKER_ID,
        "marker_text": MARKER_TEXT,
        "eos_id": EOS_ID,
        "eos_token": EOS_TOKEN,
        "max_steps": steps,
        "variant": "cn_i547",
        "followup_label": "logit-margin-reread",
        "n_probes": len(prompts),
        # Per-question, per-side, the four-float storage contract.
        "g_logp_per_q": [row["logp"] for row in trained_rows],
        "g_z_marker_per_q": [row["z_marker"] for row in trained_rows],
        "g_z_eos_per_q": [row["z_eos"] for row in trained_rows],
        "g_logZ_per_q": [row["logZ"] for row in trained_rows],
        "g_argmax_id_per_q": [row["argmax_id"] for row in trained_rows],
        "b_logp_per_q": [row["logp"] for row in base_rows],
        "b_z_marker_per_q": [row["z_marker"] for row in base_rows],
        "b_z_eos_per_q": [row["z_eos"] for row in base_rows],
        "b_logZ_per_q": [row["logZ"] for row in base_rows],
        "b_argmax_id_per_q": [row["argmax_id"] for row in base_rows],
        # Validation vs #547.
        "validation_vs_i547_stored": validation,
        # Provenance.
        "adapter_source": manifest_row["source"],
        "adapter_dir": manifest_row["i547_dir"],
        "hf_model_repo": HF_MODEL_REPO,
        "hf_data_repo": HF_DATA_REPO,
        "hf_data_revision": DATA_REVISION,
        "base_model": BASE_MODEL,
        "wall_seconds_trained": t_trained,
        "wall_seconds_base": t_base,
        "scored_at_utc": datetime.now(UTC).isoformat(),
        "produced_by": "scripts/i533_margin_run.py",
    }
    PER_CELL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(out_path)
    log.info(
        "[%s__%s] wrote %s (trained=%.1fs base=%.1fs)",
        cell_label,
        e_eval,
        out_path,
        t_trained,
        t_base,
    )


def phase_margin_scoring(args: argparse.Namespace) -> None:
    """Run phase 3 — load model once, iterate the cell × encoding grid."""
    log.info("[phase=margin_scoring] start (shard=%s, batch_size=%d)", args.shard, args.batch_size)
    manifest = phase_adapter_manifest(write=False)
    r_data = phase_r_reconstruction()
    q_test, R = r_data["q_test"], r_data["R_canon_test"]

    grid = list_grid()
    rows_by_label = {
        _i547_cell_label(r["arm"], r["seed"], r["persona"], r["steps"]): r for r in manifest["rows"]
    }

    # Apply env / CLI overrides for smoke-mode (PASS_UNIFIED: smoke = sweep
    # with one cell × one encoding via overrides, same code path).
    cell_override = os.environ.get("EVAL_CELLS_OVERRIDE", args.cells or "").strip()
    encoding_override = os.environ.get("EVAL_ENCODINGS_OVERRIDE", args.encodings or "").strip()
    if cell_override:
        wanted_cells = set(cell_override.split(","))
        grid = [c for c in grid if _i547_cell_label(*c) in wanted_cells]
        log.warning("SMOKE: restricted to %d cell(s) via EVAL_CELLS_OVERRIDE", len(grid))
    if args.limit_cells:
        grid = grid[: args.limit_cells]

    # Round-robin shard split.
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        grid = grid[i::n]
    log.info("Margin-scoring this worker handles %d cells", len(grid))

    # Encoding filter: smoke restricts to a subset; production iterates the
    # 3 cell-natural encodings (own, other, default_assistant).
    wanted_encodings: set[str] | None = None
    if encoding_override:
        wanted_encodings = set(encoding_override.split(","))
        log.warning(
            "SMOKE: restricted encodings to %s via EVAL_ENCODINGS_OVERRIDE",
            sorted(wanted_encodings),
        )

    # Load tokenizer + assertions.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)
    assert tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [MARKER_ID]
    assert tokenizer.convert_tokens_to_ids(EOS_TOKEN) == EOS_ID

    # Coverage check (already done by phase_r_reconstruction; re-asserts cheap).
    assert_r_canon_test_coverage(R, q_test, list(enc.PERSONAS))

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda:0" else torch.float32
    log.info("Loading base model %s on device=%s dtype=%s ...", BASE_MODEL, device, dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map={"": 0} if device == "cuda:0" else "cpu",
        trust_remote_code=True,
    )
    base_model.eval()

    # PeftModel boot adapter — first cell's adapter is materialized and loaded
    # as "_boot"; subsequent cells use load_adapter + set_adapter.
    if not grid:
        log.warning("[phase=margin_scoring] no cells in this shard — nothing to do")
        return
    boot_arm, boot_seed, boot_persona, boot_steps = grid[0]
    boot_row = rows_by_label[_i547_cell_label(boot_arm, boot_seed, boot_persona, boot_steps)]
    boot_dir = _materialize_adapter(boot_row)
    peft_model = PeftModel.from_pretrained(base_model, str(boot_dir), adapter_name="_boot")
    peft_model.eval()
    # Switch to disable_adapter context so the boot adapter doesn't bleed
    # into the per-cell scoring (each cell loads its own adapter explicitly).
    peft_model.set_adapter("_boot")

    for cell_idx, (arm, seed, persona, steps) in enumerate(grid):
        cell_label = _i547_cell_label(arm, seed, persona, steps)
        manifest_row = rows_by_label[cell_label]
        encodings = _eval_encodings_for_cell(arm, persona)
        if wanted_encodings is not None:
            encodings = [e for e in encodings if e in wanted_encodings]
        log.info(
            "=== [phase=cell] %d/%d %s | encodings=%s | source=%s ===",
            cell_idx + 1,
            len(grid),
            cell_label,
            encodings,
            manifest_row["source"],
        )
        for e_eval in encodings:
            _process_cell_encoding(
                peft_model,
                tokenizer,
                arm,
                seed,
                persona,
                steps,
                e_eval,
                R,
                q_test,
                device,
                args.batch_size,
                manifest_row,
            )

    log.info("[phase=margin_scoring] ok %d cells", len(grid))


# ── Phase 4: analysis ─────────────────────────────────────────────────────


def _per_cell_records() -> list[dict]:
    """Load every per-cell margin JSON written by phase 3."""
    files = sorted(PER_CELL_OUT_DIR.glob("*.json"))
    rows: list[dict] = []
    for f in files:
        try:
            rows.append(json.loads(f.read_text()))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"corrupt per-cell JSON {f}: {e}") from e
    return rows


def _paired_bootstrap_delta(
    deltas_role: dict[int, list[float]],
    deltas_sys: dict[int, list[float]],
    n_boot: int = 10_000,
    rng_seed: int = 42,
) -> dict:
    """Per-seed-paired bootstrap of (role minus system_plain) over a shared seed set.

    Inputs are seed -> per-question delta arrays for the SAME (persona,
    steps) cell, evaluated under the role and system_plain arms. Returns
    the point estimate + 95% CI.
    """
    import numpy as np

    shared_seeds = sorted(set(deltas_role.keys()) & set(deltas_sys.keys()))
    if not shared_seeds:
        return {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_seeds": 0}
    rng = np.random.default_rng(rng_seed)
    role_means = np.array([np.mean(deltas_role[s]) for s in shared_seeds])
    sys_means = np.array([np.mean(deltas_sys[s]) for s in shared_seeds])
    diff = role_means - sys_means
    point = float(diff.mean())
    boots = np.empty(n_boot, dtype=float)
    n = len(shared_seeds)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(diff[idx].mean())
    return {
        "point": point,
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n_seeds": n,
        "shared_seeds": shared_seeds,
    }


def _per_q_delta_margin(row: dict) -> list[float]:
    """Per-question Δ(z_marker − z_eos) trained − base = the margin DV."""
    import numpy as np

    g_zm = np.asarray(row["g_z_marker_per_q"], dtype=float)
    g_ze = np.asarray(row["g_z_eos_per_q"], dtype=float)
    b_zm = np.asarray(row["b_z_marker_per_q"], dtype=float)
    b_ze = np.asarray(row["b_z_eos_per_q"], dtype=float)
    return list((g_zm - g_ze) - (b_zm - b_ze))


def _per_q_delta_logp(row: dict) -> list[float]:
    """Per-question Δ log P(marker) trained − base = the parent's DV."""
    import numpy as np

    g = np.asarray(row["g_logp_per_q"], dtype=float)
    b = np.asarray(row["b_logp_per_q"], dtype=float)
    return list(g - b)


def _per_q_delta_z_marker(row: dict) -> list[float]:
    """Per-question Δ z_marker trained − base = secondary mechanistic readout."""
    import numpy as np

    g = np.asarray(row["g_z_marker_per_q"], dtype=float)
    b = np.asarray(row["b_z_marker_per_q"], dtype=float)
    return list(g - b)


def _require_full_seed_pair(
    persona: str,
    steps: int,
    role_seeds: dict[int, dict],
    sys_seeds: dict[int, dict],
) -> None:
    """Production completeness gate: both arms must carry exactly the 5 cn_i547 seeds."""
    expected_seeds = set(SEEDS_FOR["cn_i547"])
    if set(role_seeds) != expected_seeds or set(sys_seeds) != expected_seeds:
        raise RuntimeError(
            f"[phase=analysis] persona={persona} steps={steps}: seed sets "
            f"role={sorted(role_seeds)} sys={sorted(sys_seeds)} != expected "
            f"{sorted(expected_seeds)} — refusing partial-seed paired bootstrap"
        )


def phase_analysis(allow_partial: bool = False) -> dict:
    """Run phase 4 — paired role-vs-system bootstrap on margin space; rig check at s=30."""
    log.info("[phase=analysis] start")
    rows = _per_cell_records()
    if not rows:
        raise RuntimeError(
            f"[phase=analysis] no per-cell JSONs in {PER_CELL_OUT_DIR}; "
            "phase 3 (margin-scoring) must have run first"
        )
    if not allow_partial and len(rows) != 540:
        raise RuntimeError(
            f"[phase=analysis] expected 540 per-cell JSONs (180 cells x 3 encodings), "
            f"found {len(rows)} — refusing to compute a headline from partial data"
        )
    # Group by (persona, steps, e_eval, arm) -> seed -> per_q deltas (margin, logp, z_marker).
    by_key: dict[tuple[str, int, str, str], dict[int, dict[str, list[float]]]] = {}
    for r in rows:
        key = (r["training_persona"], int(r["max_steps"]), r["e_eval"], r["arm"])
        seed = int(r["seed"])
        by_key.setdefault(key, {})[seed] = {
            "margin": _per_q_delta_margin(r),
            "logp": _per_q_delta_logp(r),
            "z_marker": _per_q_delta_z_marker(r),
        }

    # Paired role-vs-system bootstrap for each (persona, steps, eval-family
    # diagonal) at every step on the grid. The headline anchor pairs:
    # role_<persona> vs system_<persona> (own own).
    paired_results: list[dict] = []
    personas = sorted({r["training_persona"] for r in rows})
    step_set = sorted({int(r["max_steps"]) for r in rows})
    for persona in personas:
        for steps in step_set:
            role_key = (persona, steps, f"role_{persona}", "role")
            sys_key = (persona, steps, f"system_{persona}", "system_plain")
            if role_key not in by_key or sys_key not in by_key:
                if not allow_partial:
                    raise RuntimeError(
                        f"[phase=analysis] missing paired keys for persona={persona} "
                        f"steps={steps} (role_present={role_key in by_key} "
                        f"sys_present={sys_key in by_key}) — production analysis "
                        "requires the full grid"
                    )
                continue
            role_seeds = by_key[role_key]
            sys_seeds = by_key[sys_key]
            if not allow_partial:
                _require_full_seed_pair(persona, steps, role_seeds, sys_seeds)
            entry = {
                "persona": persona,
                "steps": steps,
                "role_seeds_n": len(role_seeds),
                "sys_seeds_n": len(sys_seeds),
            }
            for dv_name in ("margin", "logp", "z_marker"):
                role_d = {s: role_seeds[s][dv_name] for s in role_seeds}
                sys_d = {s: sys_seeds[s][dv_name] for s in sys_seeds}
                entry[f"paired_{dv_name}"] = _paired_bootstrap_delta(role_d, sys_d)
            paired_results.append(entry)

    # Rig check at s=RIG_CHECK_STEP: |paired_margin.point − paired_logp.point|
    # must be < RIG_CHECK_MAX_NATS_DELTA. Disagreement = broken re-scoring rig.
    rig_check_failures: list[str] = []
    rig_check_rows: list[dict] = []
    for entry in paired_results:
        if entry["steps"] != RIG_CHECK_STEP:
            continue
        m = entry["paired_margin"]["point"]
        lp = entry["paired_logp"]["point"]
        agree = abs(m - lp)
        rig_check_rows.append(
            {
                "persona": entry["persona"],
                "steps": entry["steps"],
                "paired_margin_point": m,
                "paired_logp_point": lp,
                "abs_delta": agree,
            }
        )
        if agree > RIG_CHECK_MAX_NATS_DELTA:
            rig_check_failures.append(
                f"persona={entry['persona']} steps={entry['steps']} "
                f"|paired_margin − paired_logp|={agree:.3f} > {RIG_CHECK_MAX_NATS_DELTA}"
            )

    # Per-cell validation roll-up (was the re-rescoring numerically tight?).
    val_mae_t: list[float] = []
    val_mae_b: list[float] = []
    val_floor_rho_t = 1.0
    val_floor_rho_b = 1.0
    for r in rows:
        v = r["validation_vs_i547_stored"]
        val_mae_t.append(v["trained"]["mae_nats"])
        val_mae_b.append(v["base"]["mae_nats"])
        val_floor_rho_t = min(val_floor_rho_t, v["trained"]["spearman"])
        val_floor_rho_b = min(val_floor_rho_b, v["base"]["spearman"])

    import numpy as np

    analysis = {
        "schema_version": "i533_margin_analysis_v1",
        "task_id": 533,
        "followup_label": "logit-margin-reread",
        "n_cells_scored": len(rows),
        "n_paired_results": len(paired_results),
        "headline_steps": list(HEADLINE_STEPS),
        "rig_check_step": RIG_CHECK_STEP,
        "rig_check_max_nats_delta": RIG_CHECK_MAX_NATS_DELTA,
        "rig_check_rows": rig_check_rows,
        "rig_check_failures": rig_check_failures,
        "validation_summary": {
            "trained_mae_mean_nats": float(np.mean(val_mae_t)) if val_mae_t else None,
            "trained_mae_max_nats": float(np.max(val_mae_t)) if val_mae_t else None,
            "base_mae_mean_nats": float(np.mean(val_mae_b)) if val_mae_b else None,
            "base_mae_max_nats": float(np.max(val_mae_b)) if val_mae_b else None,
            "trained_spearman_min": val_floor_rho_t,
            "base_spearman_min": val_floor_rho_b,
        },
        "paired_role_vs_system": paired_results,
        "analyzed_at_utc": datetime.now(UTC).isoformat(),
        "produced_by": "scripts/i533_margin_run.py",
    }
    if rig_check_failures and not allow_partial:
        raise RuntimeError(
            f"[phase=analysis] RIG CHECK FAILED at s={RIG_CHECK_STEP}: "
            f"{rig_check_failures}. The recomputed margin and log-prob disagree "
            f"off-saturation; the re-scoring rig is broken (NOT a finding)."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ANALYSIS_OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(analysis, indent=2))
    tmp.replace(ANALYSIS_OUT_PATH)
    log.info("[phase=analysis] wrote %s", ANALYSIS_OUT_PATH)
    log.info("[phase=analysis] ok")
    return analysis


# ── Entry point ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--phase",
        choices=("adapter-manifest", "r-reconstruction", "margin-scoring", "analysis", "all"),
        required=True,
        help="Which phase to run. ``all`` runs 1 → 2 → 3 → 4 in this process.",
    )
    parser.add_argument("--shard", default=None, help="i/N round-robin split of the 180 cells")
    parser.add_argument("--cells", default=None, help="comma-separated cell labels (smoke)")
    parser.add_argument(
        "--encodings",
        default=None,
        help="comma-separated eval-encoding names (smoke; overrides cell-natural set)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-cells", type=int, default=None, help="first N cells (smoke)")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="analysis: tolerate missing keys + rig-check failures (smoke / dispatch-dry-run)",
    )
    parser.add_argument(
        "--dispatch-dry-run",
        action="store_true",
        help=(
            "Run the dispatcher plumbing (manifest + r-reconstruction + writer + sentinel) "
            "without touching the GPU forward pass. Smoke carve-out for GPU-only phases."
        ),
    )
    args = parser.parse_args(argv)

    load_dotenv()
    if Path("/workspace").exists():
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

    if args.phase == "adapter-manifest":
        phase_adapter_manifest(write=True)
    elif args.phase == "r-reconstruction":
        phase_r_reconstruction()
    elif args.phase == "margin-scoring":
        if args.dispatch_dry_run:
            log.warning("--dispatch-dry-run set: skipping the GPU forward pass")
            # Run the surrounding plumbing only (manifest + r-reconstruction +
            # adapter-manifest sanity + per-cell out-dir creation).
            phase_adapter_manifest(write=False)
            phase_r_reconstruction()
            PER_CELL_OUT_DIR.mkdir(parents=True, exist_ok=True)
            log.info("[phase=dispatch_dry_run] ok")
        else:
            phase_margin_scoring(args)
    elif args.phase == "analysis":
        phase_analysis(allow_partial=args.allow_partial)
    elif args.phase == "all":
        phase_adapter_manifest(write=True)
        phase_r_reconstruction()
        phase_margin_scoring(args)
        phase_analysis(allow_partial=args.allow_partial)
    return 0


if __name__ == "__main__":
    sys.exit(main())
