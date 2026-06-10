# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + minus sign − intentional
#!/usr/bin/env python3
"""Task #477 full-recovery re-eval driver — RECOVER THE LEAKAGE-VS-COUNT GRID.

Re-evaluates EVERY already-trained #477 adapter under
``adapters/issue_477/`` on the HF model repo ``superkaiba1/explore-persona-
space`` using the production #472 eval rig's vLLM-LoRARequest path (the same
mechanism the v4/v6 trajectory eval used) — ON the current environment, where
the silent-LoRA-not-applied regression has been DISPATCHED by the eval-guard
in ``eval_trajectory.run_trajectory_eval`` (the v4/v6 artifact bug) so any
recurrence raises ``LoRANotAppliedError`` immediately instead of writing
ΔG ≈ 0 to disk.

NOT TRAINING. Re-eval only.

The grid (35 cells, confirmed by ``HfApi().list_repo_files``):
  * 12 Cal-A cells:    ranks {2, 4, 8} × counts {2, 4, 8, 16} (M1 rank
    control; r ∈ {2, 4, 8} may be SUB-saturated where ΔG-decoupling reads.)
  * 3  Cal-A0 cells:   rank 32 × counts {2, 4, 16} (slot-fix control;
    likely saturated at the v4 byte-identical anchor.)
  * 20 calib v2 cells: counts {2, 4, 8, 16} × LRs {2e-6, 5e-6, 1e-5, 2e-5,
    5e-5}, all seed=42 (legacy LR-calibration sweep; mostly saturated for
    LR ≥ 5e-6.)

Per cell the driver:
  1. Fetches the adapter dir from HF per-file (snapshot_download has the
     siblings-truncation bug on this repo — task #480 + the existing
     ``i477_reval_confirm._fetch_adapter`` pattern).
  2. Builds the #477-disjoint held-out panel (REUSE the rig's
     ``held_out_panel`` minus the union of #477 negatives — round-3 #477
     contamination fix from ``i472_eval_trajectory.py``).
  3. Generates on-policy R via vLLM LoRARequest for the panel + source.
  4. Scores DV-A trained log P(※) + DV-A base log P(※) on the SAME R via
     ``score_logp_for_R`` — the rig's measurement primitive.
  5. Invokes ``assert_adapter_actually_applied`` — RAISES on the #477
     regression class, passes silently with a logged verdict otherwise.
  6. Computes marker-channel Bernoulli KL (the non-saturating DV) +
     source-self ΔG + held-out mean ΔG + held-out mean emit + the per-
     probe records.
  7. Persists IMMEDIATELY to ``eval_results/issue_477/reval_grid/<cell>.json``
     (checkpoint-per-phase rule). Idempotent resume: cells whose output
     file already exists are SKIPPED.

After ALL cells: aggregate to ``eval_results/issue_477/reval_grid/grid.json``.

Parallelism (``--gpus N``): the driver partitions the cell list N ways and
spawns N WORKER subprocesses with CUDA_VISIBLE_DEVICES=k. Each worker
processes its slice sequentially in-process (per-cell vLLM teardown handled
by the rig's standard pattern in ``eval_trajectory._teardown_vllm_hard``).
Subprocess env is explicit (``env={**os.environ}`` + the CVD override).

35 cells × ~50 s/cell single-GPU ≈ 29 min single-GPU; on 4×H100 → ~8 min
wall (cells split 9/9/9/8). Headline driver invocation on the pod:

    uv run python scripts/i477_reval_grid.py --gpus 4 --max-new-tokens 1024
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Two-check subprocess-env-passthrough contract: explicit env={**os.environ} on
# every subprocess call below AND load_dotenv() at module-top so HF_TOKEN /
# WANDB_API_KEY land in the parent's env before any subprocess copies it.
load_dotenv()

log = logging.getLogger("i477.reval_grid")

# ── Constants pinned to the rig + the data revision the adapters were trained on. ───
ADAPTER_HF_REPO = "superkaiba1/explore-persona-space"
ADAPTER_SUBFOLDER_ROOT = "adapters/issue_477"

DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_REVISION = "66d7db7a542e19275f8c1d8e32948396d050faa9"  # rev that produced the adapters
DATA_PREFIX = "issue472_neg_geometry"
DATA_FILES = (
    f"{DATA_PREFIX}/geometry/persona_bank.json",
    f"{DATA_PREFIX}/geometry/centroids_L10.pt",
    f"{DATA_PREFIX}/on_policy_R/R_eval.json",
)

LOCAL_DATA_ROOT = Path("data/issue_472")
LOCAL_PATHS = {
    f"{DATA_PREFIX}/geometry/persona_bank.json": LOCAL_DATA_ROOT / "persona_bank.json",
    f"{DATA_PREFIX}/geometry/centroids_L10.pt": LOCAL_DATA_ROOT / "centroids_L10.pt",
    f"{DATA_PREFIX}/on_policy_R/R_eval.json": LOCAL_DATA_ROOT / "on_policy_R" / "R_eval.json",
}

DEFAULT_OUT_ROOT = Path("eval_results/issue_477/reval_grid")
DEFAULT_ADAPTER_CACHE = Path("/tmp/i477_reval_grid/adapter_cache")
DEFAULT_MAX_NEW_TOKENS = 1024  # match v2 anchor (≥2× trained completion, CLAUDE.md)
DEFAULT_GPU_MEM_UTIL = 0.60  # match eval_trajectory.DEFAULT_GPU_MEM_UTIL


# ── Slug parsing — turn an adapter dir name into (phase, count, rank, lr). ──────────
#
# Slug conventions on HF (confirmed via list_repo_files on
# superkaiba1/explore-persona-space, 35 adapter dirs under
# adapters/issue_477/, all seed=42, all trained at lr=2e-6 except calib v2):
#
#   Cal-A:   c477_calA_negp_<count>_r<rank>_seed42_lr2e-06
#               count ∈ {2, 4, 8, 16}, rank ∈ {2, 4, 8}
#   Cal-A0:  c477_calA0_negp_<count>_r32_seed42_lr2e-06
#               count ∈ {2, 4, 16}, rank fixed = 32
#   calib:   c477_calib_negp_<count>_seed42_lr<LR>
#               count ∈ {2, 4, 8, 16}, LR ∈ {2e-6, 5e-6, 1e-5, 2e-5, 5e-5},
#               rank fixed = 32 (the v2 calibration sweep used the v4
#               anchor recipe before the v6 rank pivot)

_CAL_A_RE = re.compile(r"^c477_calA_negp_(\d+)_r(\d+)_seed(\d+)_lr([\dei.\-+]+)$")
_CAL_A0_RE = re.compile(r"^c477_calA0_negp_(\d+)_r(\d+)_seed(\d+)_lr([\dei.\-+]+)$")
_CALIB_RE = re.compile(r"^c477_calib_negp_(\d+)_seed(\d+)_lr([\dei.\-+]+)$")

# The "logical" cell slug the analyzer's count_for_slug() resolver expects (the
# slug from CELL_SPECS_477 BEFORE the seed+lr suffix the trainer appended).
# Cal-A → c477_calA_negp_<count>_r<rank>; Cal-A0 → c477_calA0_negp_<count>_r32;
# calib → c477_calib_negp_<count>.


@dataclass(frozen=True)
class CellEntry:
    """One row in the re-eval grid — adapter id + parsed parameters."""

    adapter_dirname: str  # the directory name on HF, e.g. c477_calA_negp_8_r4_seed42_lr2e-06
    logical_slug: str  # the count_for_slug-compatible slug
    phase: str  # "calA" | "calA0" | "calib"
    count: int
    rank: int
    seed: int
    lr: float

    def saturation_hint(self) -> str:
        """Coarse "likely saturated vs likely sub-saturated" hint for the analyzer.

        Cal-A r ∈ {2, 4, 8} cells are LOW-rank and likely sub-saturated (where
        the ΔG-decoupling axis can read); Cal-A0 r=32 + calib (r=32) cells are
        the v4 byte-identical anchor and are LIKELY saturated (where the
        marker-channel KL is the appropriate non-saturating DV). The analyzer
        decides what to do with this hint; we just label.
        """
        if self.phase == "calA" and self.rank in (2, 4, 8):
            return "likely_sub_saturated"
        # calA0 / calib both run at rank=32; the v4 anchor is the saturation
        # regime per the contrastive-negatives rule "Saturation hides everything".
        return "likely_saturated"


def parse_adapter_dirname(dirname: str) -> CellEntry:
    """Parse one adapter directory name into a CellEntry.

    Raises:
        ValueError: dirname does not match any of the three #477 phase patterns.
    """
    m = _CAL_A_RE.match(dirname)
    if m is not None:
        count, rank, seed, lr_str = m.groups()
        return CellEntry(
            adapter_dirname=dirname,
            logical_slug=f"c477_calA_negp_{int(count)}_r{int(rank)}",
            phase="calA",
            count=int(count),
            rank=int(rank),
            seed=int(seed),
            lr=float(lr_str),
        )
    m = _CAL_A0_RE.match(dirname)
    if m is not None:
        count, rank, seed, lr_str = m.groups()
        return CellEntry(
            adapter_dirname=dirname,
            logical_slug=f"c477_calA0_negp_{int(count)}_r{int(rank)}",
            phase="calA0",
            count=int(count),
            rank=int(rank),
            seed=int(seed),
            lr=float(lr_str),
        )
    m = _CALIB_RE.match(dirname)
    if m is not None:
        count, seed, lr_str = m.groups()
        # calib v2 trained at LoRA r=32 (the v4 anchor recipe) — encoded only by
        # convention in the rig's CELL_SPECS_477 / __init__ constants.
        return CellEntry(
            adapter_dirname=dirname,
            logical_slug=f"c477_calib_negp_{int(count)}",
            phase="calib",
            count=int(count),
            rank=32,
            seed=int(seed),
            lr=float(lr_str),
        )
    raise ValueError(
        f"adapter dir name does not match any #477 phase pattern: {dirname!r}; "
        "(expected c477_calA_*, c477_calA0_*, or c477_calib_*)"
    )


def discover_cells(*, token: str | None) -> list[CellEntry]:
    """List the #477 adapter directories on HF and parse each into a CellEntry.

    The cell list is NEVER hardcoded — we query the live HF model repo via
    ``HfApi().list_repo_files`` (the same call that confirmed 35 adapter dirs
    in the brief). A drift in the on-Hub set surfaces here.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=ADAPTER_HF_REPO, repo_type="model", token=token)
    dirnames = sorted(
        {
            f.split("/")[2]
            for f in files
            if f.startswith(f"{ADAPTER_SUBFOLDER_ROOT}/") and len(f.split("/")) >= 4
        }
    )
    if not dirnames:
        raise RuntimeError(
            f"discover_cells: no adapter dirs under {ADAPTER_SUBFOLDER_ROOT}/ in "
            f"{ADAPTER_HF_REPO} — has the upload happened?"
        )
    return [parse_adapter_dirname(d) for d in dirnames]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            # epm-lint: subprocess-env-inherit -- git rev-parse needs no credentials
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _ensure_data(token: str | None) -> None:
    """Pull persona bank + centroids + R_eval into the rig's expected layout.

    Idempotent. Forked from ``i477_reval_confirm._ensure_data`` (the
    feedback_ensure_data_non_idempotent fix: ``shutil.copyfile`` not
    ``os.link``).
    """
    import shutil

    from huggingface_hub import hf_hub_download

    LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (LOCAL_DATA_ROOT / "on_policy_R").mkdir(parents=True, exist_ok=True)
    for hf_path, local_path in LOCAL_PATHS.items():
        if local_path.exists():
            log.info("data already local: %s", local_path)
            continue
        log.info("pulling %s @ %s → %s", hf_path, DATA_REVISION[:8], local_path)
        cached = hf_hub_download(
            repo_id=DATA_REPO,
            repo_type="dataset",
            revision=DATA_REVISION,
            filename=hf_path,
            token=token,
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, local_path)


def _fetch_adapter(entry: CellEntry, token: str | None, cache_root: Path) -> Path:
    """Per-file fetch of one adapter directory.

    Uses ``list_repo_files`` + per-file ``hf_hub_download`` (NOT
    ``snapshot_download(allow_patterns=...)`` which returns 0 files on this
    repo's truncated siblings — task #480 + i477_reval_confirm pattern).
    """
    from huggingface_hub import HfApi, hf_hub_download

    subfolder = f"{ADAPTER_SUBFOLDER_ROOT}/{entry.adapter_dirname}"
    cache_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    all_files = api.list_repo_files(repo_id=ADAPTER_HF_REPO, repo_type="model", token=token)
    sub_files = [f for f in all_files if f.startswith(f"{subfolder}/")]
    if not sub_files:
        raise FileNotFoundError(
            f"_fetch_adapter: no files under {subfolder} in {ADAPTER_HF_REPO} — "
            f"adapter subfolder missing on Hub for cell {entry.adapter_dirname!r}"
        )
    for fn in sub_files:
        hf_hub_download(
            repo_id=ADAPTER_HF_REPO,
            repo_type="model",
            filename=fn,
            local_dir=str(cache_root),
            token=token,
        )
    adapter_dir = cache_root / subfolder
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / required).exists():
            raise FileNotFoundError(
                f"_fetch_adapter: missing {required} under {adapter_dir} after per-file download"
            )
    return adapter_dir


def _select_eval_slice(*, entry: CellEntry) -> tuple[dict[str, str], list[str], str, str]:
    """Build the #477-disjoint held-out panel + Q_eval + source for ONE cell.

    Mirrors i472_eval_trajectory's --cell-specs 477 path: REUSE the rig's
    base panel, additionally subtract the union of #477 negatives so the
    count axis is NOT contaminated by personas the cell trained against
    (round-3 #477 contamination guard).
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        CELL_SPECS_477,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        HEADLINE_LAYER,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        all_negatives_union,
        held_out_panel,
        negatives_for_cell,
    )

    bank = load_persona_bank(LOCAL_PATHS[f"{DATA_PREFIX}/geometry/persona_bank.json"])
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, LOCAL_DATA_ROOT)
    base_panel = held_out_panel(cts, source=SOURCE_PERSONA)
    union_477 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    panel_names = [p for p in base_panel if p not in union_477]

    # Fail-loud disjointness assert AGAINST THIS cell's negatives (round-3 #477 fix).
    cell_negs = set(
        negatives_for_cell(
            entry.logical_slug, cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477
        )
    )
    overlap = set(panel_names) & cell_negs
    if overlap:
        raise AssertionError(
            f"panel ∩ negatives for {entry.logical_slug!r}: {sorted(overlap)} — "
            "would conflate leakage with training-against-suppression."
        )

    eval_personas = {p: bank[p] for p in panel_names}
    _q_train, q_eval = get_train_eval_questions()
    return eval_personas, list(q_eval), SOURCE_PERSONA, bank[SOURCE_PERSONA]


def _summarize_records(
    g_records: dict[str, dict[str, dict[str, float | bool]]],
    b_records: dict[str, dict[str, dict[str, float | bool]]],
    source: str,
) -> dict[str, float]:
    """Reduce per-probe (g, b) records to source-self + held-out summaries."""
    held_dgs: list[float] = []
    held_emits: list[bool] = []
    src_dgs: list[float] = []
    src_emits: list[bool] = []
    for persona, per_q_g in g_records.items():
        for q, gleaf in per_q_g.items():
            dg = float(gleaf["logp"]) - float(b_records[persona][q]["logp"])
            em = bool(gleaf.get("argmax_marker", False))
            if persona == source:
                src_dgs.append(dg)
                src_emits.append(em)
            else:
                held_dgs.append(dg)
                held_emits.append(em)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def _rate(xs: list[bool]) -> float:
        return sum(1 for x in xs if x) / len(xs) if xs else float("nan")

    return {
        "source_self_delta_g_mean": _mean(src_dgs),
        "source_emit_rate": _rate(src_emits),
        "held_out_delta_g_mean": _mean(held_dgs),
        "held_out_emit_rate": _rate(held_emits),
        "n_source_probes": len(src_dgs),
        "n_held_out_probes": len(held_dgs),
    }


def _build_checkpoint_payload(
    *,
    entry: CellEntry,
    g_records: dict[str, dict[str, dict[str, float | bool]]],
    b_records: dict[str, dict[str, dict[str, float | bool]]],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    source: str,
    adapter_dir: Path,
) -> dict:
    """Build the analyze.py-compatible per-checkpoint dict + attach marker-channel KL.

    Returns a 1-element ``checkpoints`` payload shaped like ``trajectory.json``
    (frac=1.0 terminal-only), so ``attach_marker_channel_aggregates`` can read
    it without modification AND downstream analyzer helpers
    (``aggregate_bystander_marker_channel_kl`` etc.) see the same schema they
    expect from the trajectory rig.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        attach_marker_channel_aggregates,
    )

    # Build the held_out block in the rig's shape — per-leaf {g_logp, b_logp,
    # delta_g, argmax_marker, n_marker_in_R, r_collapsed, kl=None}.
    held_out: dict[str, dict[str, dict[str, float | bool | None]]] = {}
    for persona in eval_personas:
        held_out[persona] = {}
        for q in eval_questions:
            gleaf = g_records[persona][q]
            bleaf = b_records[persona][q]
            gl = float(gleaf["logp"])
            bl = float(bleaf["logp"])
            held_out[persona][q] = {
                "g_logp": gl,
                "b_logp": bl,
                "delta_g": gl - bl,
                "argmax_marker": bool(gleaf["argmax_marker"]),
                "n_marker_in_R": int(gleaf.get("n_marker_in_R", 0)),
                "r_collapsed": bool(gleaf.get("r_collapsed", False)),
                "kl": None,  # full-vocab KL skipped — non-saturating DV is the marker-channel KL.
            }

    src_deltas = [
        float(g_records[source][q]["logp"]) - float(b_records[source][q]["logp"])
        for q in eval_questions
    ]
    src_collapsed = any(
        bool(g_records[source][q].get("r_collapsed", False)) for q in eval_questions
    )
    src_emit = sum(
        1 for q in eval_questions if bool(g_records[source][q].get("argmax_marker", False))
    ) / max(1, len(eval_questions))
    source_self = {
        "g_logp_mean": sum(float(g_records[source][q]["logp"]) for q in eval_questions)
        / len(eval_questions),
        "b_logp_mean": sum(float(b_records[source][q]["logp"]) for q in eval_questions)
        / len(eval_questions),
        "delta_g_mean": sum(src_deltas) / len(src_deltas) if src_deltas else float("nan"),
        "emission_p": float(src_emit),
        "r_collapsed": src_collapsed,
    }
    n_collapsed = sum(
        1
        for persona in eval_personas
        for q in eval_questions
        if bool(g_records[persona][q].get("r_collapsed", False))
    )
    checkpoint = {
        "frac": 1.0,
        "step": None,  # terminal-only re-eval; the trainer's final step is not on disk
        "adapter_path": str(adapter_dir),
        "source_self": source_self,
        "held_out_collapse_share": (
            n_collapsed / (len(eval_personas) * len(eval_questions))
            if eval_personas and eval_questions
            else 0.0
        ),
        "n_held_out_collapsed": n_collapsed,
        "held_out": held_out,
    }

    # Attach marker-channel Bernoulli KL aggregates (the non-saturating DV).
    # attach_marker_channel_aggregates wraps a trajectory dict with checkpoints.
    traj_shape = {"checkpoints": [checkpoint]}
    attach_marker_channel_aggregates(traj_shape)
    return traj_shape["checkpoints"][0]


def _teardown_vllm(llm) -> None:
    """Reap vLLM workers between cells — same pattern as eval_trajectory."""
    import torch

    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import psutil

        me = psutil.Process()
        # Snapshot the child list ONCE — re-querying after terminate() (async)
        # returns a fresh list that may miss the originals or include freshly
        # spawned procs, causing wait_procs to hang or miss the targets.
        children = me.children(recursive=True)
        for c in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.terminate()
        _gone, alive = psutil.wait_procs(children, timeout=10)
        for c in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.kill()
    except ImportError:
        log.warning("psutil unavailable; cannot reap vLLM worker subprocesses.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _eval_one_cell(
    *,
    entry: CellEntry,
    out_root: Path,
    cache_root: Path,
    max_new_tokens: int,
    gpu_mem_util: float,
    token: str | None,
) -> Path:
    """Re-eval ONE cell end-to-end. Returns the per-cell JSON path.

    Skips silently if the per-cell output already exists (idempotent resume).
    """
    cell_out = out_root / f"{entry.adapter_dirname}.json"
    if cell_out.exists():
        log.info("[%s] per-cell output exists — skipping: %s", entry.adapter_dirname, cell_out)
        return cell_out

    # ── Phase 0: adapter on disk + eval slice. ────────────────────────────────
    adapter_dir = _fetch_adapter(entry, token, cache_root)
    eval_personas, q_eval, source_name, source_prompt = _select_eval_slice(entry=entry)
    panel_plus_source = dict(eval_personas)
    panel_plus_source.setdefault(source_name, source_prompt)

    log.info(
        "[%s] eval slice: %d held-out × %d Q + source → %d probes",
        entry.adapter_dirname,
        len(eval_personas),
        len(q_eval),
        (len(eval_personas) + 1) * len(q_eval),
    )

    # ── Phase 1: vLLM + LoRARequest, marker assert, on-policy R, score g + b. ─
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        BASE_MODEL,
        RANK_CONTROL_V6,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
        assert_adapter_actually_applied,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
        assert_marker_token,
        score_logp_for_R,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        _generate_on_policy_R,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=token)
    assert_marker_token(tokenizer)

    # Engine's max_lora_rank MUST be ≥ this cell's adapter rank. We use the
    # repo-wide ceiling (RANK_CONTROL_V6 = 32) which dominates every entry
    # (Cal-A r ∈ {2,4,8}, Cal-A0 + calib r = 32). The brief notes the v4/v6
    # silent-LoRA-not-applied bug may be a max_lora_rank mismatch; we use
    # an explicit ceiling here so any future drift in entry.rank fails loud
    # at engine init rather than silently dropping the LoRA.
    engine_max_lora_rank = max(RANK_CONTROL_V6, entry.rank)

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        seed=entry.seed,
        max_model_len=2048,
        enable_lora=True,
        max_lora_rank=engine_max_lora_rank,
        max_loras=1,
    )
    lora_req = LoRARequest(
        lora_name=f"reval_{entry.adapter_dirname}",
        lora_int_id=1,
        lora_path=str(adapter_dir),
    )

    try:
        log.info("[%s] phase=traj_vllm: on-policy gen", entry.adapter_dirname)
        r_on_policy = _generate_on_policy_R(
            llm, tokenizer, panel_plus_source, q_eval, lora_req, max_new_tokens
        )
        log.info("[%s] phase=traj_vllm: score g (use_lora=True)", entry.adapter_dirname)
        g_records = score_logp_for_R(
            llm,
            tokenizer,
            r_by_persona_q=r_on_policy,
            eval_personas=panel_plus_source,
            eval_questions=q_eval,
            cell_label=f"TRAINED/{entry.adapter_dirname}",
            use_lora=True,
            lora_request=lora_req,
        )
        log.info("[%s] phase=traj_vllm: score b (use_lora=False)", entry.adapter_dirname)
        b_records = score_logp_for_R(
            llm,
            tokenizer,
            r_by_persona_q=r_on_policy,
            eval_personas=panel_plus_source,
            eval_questions=q_eval,
            cell_label=f"BASE/{entry.adapter_dirname}",
            use_lora=False,
        )
    finally:
        _teardown_vllm(llm)

    # ── Phase 2: fail-loud guard (the #477 silent-LoRA-not-applied regression). ─
    guard_diag = assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g_records,
        b_records=b_records,
        cell_label=entry.adapter_dirname,
    )

    # ── Phase 3: build the analyze-compatible checkpoint payload + summary. ───
    checkpoint = _build_checkpoint_payload(
        entry=entry,
        g_records=g_records,
        b_records=b_records,
        eval_personas=eval_personas,
        eval_questions=q_eval,
        source=source_name,
        adapter_dir=adapter_dir,
    )
    summary = _summarize_records(g_records, b_records, source_name)

    payload = {
        "schema_version": "i477_reval_grid_v1",
        "adapter_dirname": entry.adapter_dirname,
        "logical_slug": entry.logical_slug,
        "phase": entry.phase,
        "count": entry.count,
        "rank": entry.rank,
        "seed": entry.seed,
        "lr": entry.lr,
        "saturation_hint": entry.saturation_hint(),
        "data_revision": DATA_REVISION,
        "n_held_out_personas": len(eval_personas),
        "held_out_personas": sorted(eval_personas.keys()),
        "n_eval_questions": len(q_eval),
        "eval_questions": q_eval,
        "source": source_name,
        "guard": guard_diag,
        "summary": summary,
        "checkpoint": checkpoint,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    cell_out.parent.mkdir(parents=True, exist_ok=True)
    cell_out.write_text(json.dumps(payload, indent=2))
    log.info(
        "[%s] DONE: source-self ΔG=%.2f, source emit=%.2f, held-out ΔG=%.2f, "
        "marker-channel KL src=%.3f bys=%.3f → %s",
        entry.adapter_dirname,
        summary["source_self_delta_g_mean"],
        summary["source_emit_rate"],
        summary["held_out_delta_g_mean"],
        checkpoint.get("source_self_marker_channel_kl", float("nan")),
        checkpoint.get("mean_bystander_marker_channel_kl", float("nan")),
        cell_out,
    )
    return cell_out


def _aggregate_grid(out_root: Path, cells: list[CellEntry]) -> Path:
    """Walk every per-cell JSON under out_root and stitch into grid.json."""
    rows: list[dict] = []
    missing: list[str] = []
    for entry in cells:
        cell_out = out_root / f"{entry.adapter_dirname}.json"
        if not cell_out.exists():
            missing.append(entry.adapter_dirname)
            continue
        payload = json.loads(cell_out.read_text())
        rows.append(
            {
                "adapter_dirname": payload["adapter_dirname"],
                "logical_slug": payload["logical_slug"],
                "phase": payload["phase"],
                "count": payload["count"],
                "rank": payload["rank"],
                "seed": payload["seed"],
                "lr": payload["lr"],
                "saturation_hint": payload["saturation_hint"],
                "source_self_delta_g_mean": payload["summary"]["source_self_delta_g_mean"],
                "source_emit_rate": payload["summary"]["source_emit_rate"],
                "held_out_delta_g_mean": payload["summary"]["held_out_delta_g_mean"],
                "held_out_emit_rate": payload["summary"]["held_out_emit_rate"],
                "source_self_marker_channel_kl": payload["checkpoint"].get(
                    "source_self_marker_channel_kl"
                ),
                "mean_bystander_marker_channel_kl": payload["checkpoint"].get(
                    "mean_bystander_marker_channel_kl"
                ),
                "guard_verdict": payload["guard"]["guard_verdict"],
                "adapter_b_max_norm": payload["guard"]["adapter_b_max_norm"],
            }
        )
    grid_path = out_root / "grid.json"
    grid_path.write_text(
        json.dumps(
            {
                "schema_version": "i477_reval_grid_v1",
                "n_cells_total": len(cells),
                "n_cells_persisted": len(rows),
                "n_cells_missing": len(missing),
                "missing_cells": missing,
                "rows": rows,
                "git_commit": _git_sha(),
                "hostname": socket.gethostname(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    log.info(
        "Aggregated %d/%d cells → %s (%d missing)",
        len(rows),
        len(cells),
        grid_path,
        len(missing),
    )
    return grid_path


def _run_worker(
    *,
    worker_cells: list[CellEntry],
    out_root: Path,
    cache_root: Path,
    max_new_tokens: int,
    gpu_mem_util: float,
    token: str | None,
) -> int:
    """In-process worker — eval the assigned cells sequentially.

    Run on the host or in a subprocess (driven by --worker-cells). Per-cell
    failures are RAISED loudly — checkpoint-per-phase guarantees that all
    EARLIER cells in the slice already landed on disk.
    """
    for entry in worker_cells:
        _eval_one_cell(
            entry=entry,
            out_root=out_root,
            cache_root=cache_root,
            max_new_tokens=max_new_tokens,
            gpu_mem_util=gpu_mem_util,
            token=token,
        )
    return 0


def _partition(cells: list[CellEntry], n_gpus: int) -> list[list[CellEntry]]:
    """Round-robin cells across GPU slices (small per-cell wall-time variance
    means round-robin balances better than contiguous chunks)."""
    return [cells[i::n_gpus] for i in range(n_gpus)]


def _spawn_worker_subprocesses(
    *,
    partitions: list[list[CellEntry]],
    out_root: Path,
    cache_root: Path,
    max_new_tokens: int,
    gpu_mem_util: float,
    script_path: Path,
) -> int:
    """Spawn one worker subprocess per non-empty partition with CUDA_VISIBLE_DEVICES=k."""
    procs: list[tuple[int, subprocess.Popen]] = []
    for gpu_id, slice_ in enumerate(partitions):
        if not slice_:
            continue
        worker_cell_names = ",".join(e.adapter_dirname for e in slice_)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        cmd = [
            "uv",
            "run",
            "python",
            str(script_path),
            "--worker-cells",
            worker_cell_names,
            "--out-root",
            str(out_root),
            "--cache-root",
            str(cache_root),
            "--max-new-tokens",
            str(max_new_tokens),
            "--gpu-mem-util",
            str(gpu_mem_util),
        ]
        log.info("spawning worker gpu=%d on %d cells: %s", gpu_id, len(slice_), cmd)
        p = subprocess.Popen(cmd, env=env)
        procs.append((gpu_id, p))

    failures: list[tuple[int, int]] = []
    for gpu_id, p in procs:
        rc = p.wait()
        if rc != 0:
            failures.append((gpu_id, rc))
            log.error("worker gpu=%d exited rc=%d", gpu_id, rc)
        else:
            log.info("worker gpu=%d exited rc=0", gpu_id)
    if failures:
        log.error("%d worker(s) failed: %s", len(failures), failures)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Task #477 full-recovery re-eval driver — recover the leakage-vs-count "
            "grid via the vLLM-LoRARequest path on the current env (post-guard)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to parallelize across (spawns N worker subprocesses, "
        "one per GPU, with CUDA_VISIBLE_DEVICES=k).",
    )
    ap.add_argument(
        "--cells",
        default=None,
        help="Optional comma-separated list of adapter dir names to eval (substring "
        "match against the on-Hub list). Default = all 35 cells.",
    )
    ap.add_argument(
        "--phase",
        choices=("calA", "calA0", "calib", "all"),
        default="all",
        help="Restrict to one phase family. Default = all.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_ADAPTER_CACHE)
    ap.add_argument(
        "--worker-cells",
        default=None,
        help="Internal: comma-separated cells this in-process worker owns. When "
        "set, --gpus is ignored and we eval the listed cells in-process.",
    )
    ap.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip the final grid.json aggregation (default off; aggregation is "
        "fast and gives the dashboard a single artifact to render).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cell list + per-GPU partition and exit (no fetch, no eval).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=reval_grid] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError(
            "HF_TOKEN missing — load_dotenv() ran but .env lacks the token. The "
            "adapter list + data fetch need it. Fix .env on the pod."
        )

    # ── Phase 0: discover + filter the cell set (NEVER hardcoded). ────────────
    all_cells = discover_cells(token=token)
    log.info("discovered %d cells on HF under %s/", len(all_cells), ADAPTER_SUBFOLDER_ROOT)

    cells: list[CellEntry] = list(all_cells)
    if args.phase != "all":
        cells = [c for c in cells if c.phase == args.phase]
    if args.cells:
        substrs = [s.strip() for s in args.cells.split(",") if s.strip()]
        cells = [c for c in cells if any(s in c.adapter_dirname for s in substrs)]
    if not cells:
        raise SystemExit("no cells match the --phase / --cells filter")

    if args.worker_cells is not None:
        # ── Worker branch: in-process eval of the assigned cells. ─────────────
        wanted = {s.strip() for s in args.worker_cells.split(",") if s.strip()}
        worker_cells = [c for c in all_cells if c.adapter_dirname in wanted]
        missing = wanted - {c.adapter_dirname for c in worker_cells}
        if missing:
            raise SystemExit(
                f"--worker-cells references unknown cells (not on HF): {sorted(missing)}"
            )
        log.info("worker: %d cells assigned", len(worker_cells))
        _ensure_data(token)
        return _run_worker(
            worker_cells=worker_cells,
            out_root=args.out_root,
            cache_root=args.cache_root,
            max_new_tokens=args.max_new_tokens,
            gpu_mem_util=args.gpu_mem_util,
            token=token,
        )

    # ── Driver branch: dispatch. ─────────────────────────────────────────────
    partitions = _partition(cells, args.gpus)
    log.info(
        "partitioned %d cells across %d GPU slices: sizes=%s",
        len(cells),
        args.gpus,
        [len(p) for p in partitions],
    )
    if args.dry_run:
        for gpu_id, slice_ in enumerate(partitions):
            print(f"\n[gpu={gpu_id}, {len(slice_)} cells]")
            for entry in slice_:
                print(
                    f"  {entry.adapter_dirname}  ({entry.phase}, count={entry.count}, "
                    f"rank={entry.rank}, lr={entry.lr:g}, hint={entry.saturation_hint()})"
                )
        return 0

    # Ensure data once in the driver (workers will see it on shared disk).
    _ensure_data(token)
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.gpus == 1:
        rc = _run_worker(
            worker_cells=cells,
            out_root=out_root,
            cache_root=args.cache_root,
            max_new_tokens=args.max_new_tokens,
            gpu_mem_util=args.gpu_mem_util,
            token=token,
        )
    else:
        script_path = Path(__file__).resolve()
        rc = _spawn_worker_subprocesses(
            partitions=partitions,
            out_root=out_root,
            cache_root=args.cache_root,
            max_new_tokens=args.max_new_tokens,
            gpu_mem_util=args.gpu_mem_util,
            script_path=script_path,
        )

    # ── Aggregate (idempotent — re-runnable). ────────────────────────────────
    if not args.no_aggregate:
        grid_path = _aggregate_grid(out_root, cells)
        log.info("grid.json → %s", grid_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
