#!/usr/bin/env python
# ruff: noqa: RUF002, RUF003  # em-dash + marker token intentional
"""#1112 pod-side phase driver — sycophancy 2×2 + marker pair + 3-arm capture.

Phases (linear, checkpoint-per-phase, resume-keyed; plan v3 §4/§9):

  p0_stage      stage + sha/rev-pin every reused input (c3 mix, pos/cn, generic
                corpus, R_train, fu2 checkpoints, margin sidecars)
  p1_mixes      derive posonly/generic mixes (20/20/40 fail-fast) + marker mix;
                upload mixes BEFORE training
  p2_train      s2 LoRA + s3/s4 full-FT (ZeRO-3 subprocess) + m1/m2 marker
  p3_ladder     Tier-1 judged-rate ladders (rungs 2..30) for s2/s3/s4, sharded
                one cell per GPU; select_dose_checkpoint vs [0.60, 0.85]
  g1_gate       FT install viability + FENCE-AWARENESS pre-check (plan §7)
  p4_persist_ft upload SELECTED FT checkpoints to the overflow repo, THEN
                delete non-selected FT rungs (disk; plan §9 + binding order)
  p5_generic    s5/s6 generic controls trained to the method-matched steps
  p6_parity     reused-cell (s1 = fu2 checkpoint-14) rsLoRA parity probe
  p7_tier2      Tier-2 generation + judge fold -> install/*_tier2.json
  p7b_margin    teacher-forced fixed-pool margin companion (plan §6 DV table):
                sha-pinned #1090 pools, base + per-selected-checkpoint reads
                -> install/*_margin.json
  p8_marker     m2 grid ΔG selection + full three-space reads (m1 + m2);
                m2 selected-checkpoint overflow upload THEN rung reap
  p9_rb         sycophancy r_B (issue779 extractor subprocess) + marker W_U row
  p10_capture   18 capture passes (gen + 28-layer 3-span TF pooling), sharded
  p10b_capture_tf  tf-shared amendment (plan v6): 6 sequential teacher-forced
                SHARED-response passes over the persisted base rows (pinned
                revs; no generation stage) -> capture_tf/<cell>/selected/
  p11_geometry  smoke-scale geometry stub (full geometry runs VM-side)
  p12_upload    remaining text/JSON + capture tensors + adapters; sentinel

``--smoke`` is the SAME dispatcher with tiny knobs (plan §4.5 smoke/sweep
parity): cell subset (s3,), 2 optimizer steps + 1 consolidated ZeRO-3 save +
vLLM-load canary, 1 Tier-1 rung at 2 questions, LIVE judge, 2-context ×
2-question (4-row) 3-arm 28-layer capture, geometry on the captured stub
(nondegenerate prefix path + a single-context degenerate-branch probe),
recording-free upload via ``--no-upload``. Every phase reads its cell list
from the ONE resolver
(``cfg.cells``), so the smoke subset threads through train, ladder, tier2,
capture, geometry, and upload alike.

``[phase=done]`` is emitted by ``scripts/issue1112_dispatch.sh`` ONLY.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1090_fu1 as fu1  # noqa: E402
import issue1090_run as i1090  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.negatives import (  # noqa: E402
    assert_panel_disjoint_from_sources,
    default_panel,
)
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    ModelOrganism,
    _rate_for_cell,
    _sha256_file,
    make_source_rate_fn,
    release_trainer_cuda_memory,
)
from explore_persona_space.artifacts.recipe import (  # noqa: E402
    JUDGED_RATE_BAND,
    build_train_config,
    recipe_for,
    select_dose_checkpoint,
)
from explore_persona_space.experiments import issue_1112 as C  # noqa: E402
from explore_persona_space.experiments.issue_1112 import mixes as mixmod  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1112")

# Captured at IMPORT time, BEFORE any in-process train_lora call can clobber
# the dispatcher's env: train/sft.py sets CUDA_VISIBLE_DEVICES=str(cfg.gpu_id)
# in-process (the gotchas.md +gpu_id clobber). Round-4 crash (pod-1112): s2's
# in-process LoRA train pinned the dispatcher env to GPU 0, so s3's _ft_cmd
# read 1 visible GPU and composed `--num_processes 1` against the 4-GPU ZeRO-3
# config (zero sharding -> whole-7B params+grads+Adam on GPU 0, OOM), and the
# accelerate subprocess ALSO inherited CVD=0.
_INITIAL_CVD = os.environ.get("CUDA_VISIBLE_DEVICES")

ACCEL_CONFIG = "configs/accelerate/zero3_4gpu_accum1.yaml"  # behavior FT: eff-batch 16
# ZeRO-3 world size for EVERY full-FT launch (s3/s4/s6 behavior FT + m2 marker
# FT + the g1 extension). Pinned to both accelerate configs' `num_processes: 4`
# (tests/test_issue1112_launch_configs.py) — the effective batch is a science
# variable (per-device x world x accum), so full mode NEVER derives a smaller
# world size from the (clobbable) visible-GPU count; it fails loud instead.
FT_NUM_PROCESSES = 4
# The marker trainer pins grad-accum 16 (#514 eff-batch-64 recipe); DeepSpeed's
# explicit gradient_accumulation_steps must MATCH TrainingArguments (fill_match
# raises otherwise) — hence a dedicated accum-16 config (round-2 Critical 4).
MARKER_ACCEL_CONFIG = "configs/accelerate/zero3_4gpu_accum16.yaml"
FT_TRAINER = "scripts/train_behavior_fullft.py"
MARKER_FT_TRAINER = "scripts/issue1112_train_marker_fullft.py"
RB_EXTRACTOR = "scripts/issue779_extract_rb.py"


# ── Config ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Cfg:
    smoke: bool
    cells: tuple[str, ...]
    out_root: Path
    seed: int = C.SEED
    tier1_n: int = 5
    tier1_draws: int = 3
    tier2_n: int = 10
    tier2_draws: int = 5
    eval_question_limit: int | None = None
    sentinel_dir: Path | None = None
    upload: bool = True
    fence_hours: float = float(os.environ.get("EPS_MAX_RUN_HOURS", "72"))
    phases: tuple[str, ...] = ()  # empty -> all
    # OPTIONAL marker extension of the tf-shared amendment (plan v6 §4):
    # default OFF — the m1/m2 shared-text passes contribute nothing to the
    # lattice and run only on an explicit --tf-marker inside the wall budget.
    tf_marker: bool = False

    def regime_key(self) -> dict:
        key = {
            "issue": C.ISSUE,
            "smoke": self.smoke,
            "cells": list(self.cells),
            "seed": self.seed,
            "tier1": [self.tier1_n, self.tier1_draws],
            "tier2": [self.tier2_n, self.tier2_draws],
            "eval_question_limit": self.eval_question_limit,
            "band": list(JUDGED_RATE_BAND),
            "marker_band": list(C.MARKER_BAND),
            "save_steps": C.SYCO_SAVE_STEPS,
            "max_length": C.SYCO_MAX_LENGTH,
            "step_ceiling": C.SYCO_STEP_CEILING,
        }
        # Per-cell ceiling threading (plan v8 §12.1: the lr-matched cell
        # ladders to 60): included ONLY when this run's cells carry an
        # override, so every pre-existing cell set keeps its regime dict
        # byte-identical (per-rung ladder resume caches stay valid).
        per_cell = {c: C.CELL_STEP_CEILING[c] for c in self.cells if c in C.CELL_STEP_CEILING}
        if per_cell:
            key["cell_step_ceilings"] = per_cell
        return key


# --phases accepts the SHORT names main()'s want() checks AND the full
# docstring/log names (pN_...) — the plan v6 workload command uses
# `--phases p10b_capture_tf,p12_upload`, which must resolve; an unknown
# token fails loud at parse time instead of silently running ALL phases.
_PHASE_ALIASES = {
    "p0_stage": "stage",
    "p1_mixes": "mixes",
    "p2_train": "train",
    "p3_ladder": "ladder",
    "g1_gate": "g1",
    "p4_persist_ft": "persist_ft",
    "p5_generic": "generic",
    "p6_parity": "parity",
    "p7_tier2": "tier2",
    "p7b_margin": "margin",
    "p8_marker": "marker",
    "p9_rb": "rb",
    "p10_capture": "capture",
    "p10b_capture_tf": "capture_tf",
    "p11_geometry": "geometry",
    "p12_upload": "upload",
}
_KNOWN_PHASES = frozenset(_PHASE_ALIASES.values())


def normalize_phases(raw: str | None) -> tuple[str, ...]:
    """Comma list of phase names -> canonical short-name tuple (fail-loud)."""
    if not raw:
        return ()
    out: list[str] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        t = _PHASE_ALIASES.get(t, t)
        if t not in _KNOWN_PHASES:
            raise ValueError(
                f"unknown phase {tok.strip()!r}: want one of {sorted(_KNOWN_PHASES)} "
                "(pN_-prefixed aliases accepted)"
            )
        out.append(t)
    return tuple(out)


def resolve_cells(cells_arg: str | None, smoke: bool) -> tuple[str, ...]:
    """The ONE cell resolver every phase consumes (smoke = same path, 1 cell)."""
    if cells_arg:
        ids = tuple(t.strip() for t in cells_arg.split(","))
        bad = [t for t in ids if t not in C.ALL_TRAINED_CELLS]
        if bad:
            raise ValueError(f"bad cells {bad!r}: want a subset of {C.ALL_TRAINED_CELLS}")
        return ids
    if smoke:
        return ("s3_fullft_neg",)  # the plan §4.5 smoke cell (FT+negatives)
    return C.ALL_TRAINED_CELLS


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _phase(name: str) -> None:
    i1090._phase(name)


def _physical_gpu_ids() -> list[str]:
    """Physical GPU ids for subprocess CVD pins — clobber-immune.

    Honors a LAUNCHER-set CUDA_VISIBLE_DEVICES (captured at import as
    _INITIAL_CVD — a deliberate external restriction, e.g. a fanout unit);
    otherwise enumerates via nvidia-smi in a SUBPROCESS, immune to both the
    in-process train_lora CVD clobber and torch's cached device count (the
    round-4 crash class). Raises RuntimeError when no GPU is available.
    """
    if _INITIAL_CVD is not None and _INITIAL_CVD.strip():
        return [t.strip() for t in _INITIAL_CVD.split(",") if t.strip()]
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError("no CUDA devices visible — the #1112 GPU phases need >=1 GPU") from e
    ids = [ln.strip() for ln in proc.stdout.split("\n") if ln.strip()]
    if not ids:
        raise RuntimeError("no CUDA devices visible — the #1112 GPU phases need >=1 GPU")
    return ids


def _n_gpus() -> int:
    """Visible-GPU count from _physical_gpu_ids (NEVER torch.cuda.device_count:
    after an in-process train_lora call the dispatcher env carries CVD=0 and
    torch reads 1, silently degrading fanout width and FT world size)."""
    return len(_physical_gpu_ids())


# ── p0: stage inputs (pinned) ────────────────────────────────────────────────


def _stage_file(path_in_repo: str, dest: Path, *, revision: str, sha256: str | None = None) -> Path:
    """Per-file hf_hub_download at a PINNED revision + optional sha assert."""
    from huggingface_hub import hf_hub_download

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(C.HF_DATA_REPO, path_in_repo, repo_type="dataset", revision=revision)
        shutil.copyfile(got, dest)
    if sha256 is not None:
        actual = _sha256_file(dest)
        if actual != sha256:
            raise ValueError(f"sha256 mismatch for {path_in_repo}: {actual} != pinned {sha256}")
    return dest


def _stage_overflow_prefix(
    prefix: str, dest: Path, *, revision: str, recursive: bool = True
) -> Path:
    """Stage a checkpoint subfolder from the PRIVATE overflow repo at a pinned
    revision (scoped list_repo_tree + per-file download; no staging transform
    — files land at their prefix-relative paths, reuse check (h)(iv) N/A).
    ``recursive=False`` stages only the prefix's TOP-LEVEL files (the m1
    band-stopped final adapter lives at the ladder root beside checkpoint-N/
    subdirs the tf-shared pass must not pull)."""
    from huggingface_hub import HfApi, hf_hub_download

    if (dest / "adapter_config.json").exists() or (dest / "config.json").exists():
        return dest
    api = HfApi()
    entries = [
        e.path
        for e in api.list_repo_tree(
            C.OVERFLOW_REPO,
            path_in_repo=prefix,
            repo_type="model",
            recursive=recursive,
            revision=revision,
        )
        if getattr(e, "size", None) is not None
    ]
    if not entries:
        raise FileNotFoundError(f"no files under {C.OVERFLOW_REPO}/{prefix} @ {revision}")
    dest.mkdir(parents=True, exist_ok=True)
    for p in entries:
        got = hf_hub_download(C.OVERFLOW_REPO, p, repo_type="model", revision=revision)
        rel = Path(p).relative_to(prefix)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(got, target)
    return dest


def phase_stage(cfg: Cfg) -> dict:
    _phase("p0_stage")
    inputs = cfg.out_root / "inputs"
    done_path = cfg.out_root / "p0_stage.json"
    if done_path.exists():
        return _read_json(done_path)
    rec: dict = {"staged": {}}
    syco = any(c.startswith("s") for c in cfg.cells)
    marker = any(c.startswith("m") for c in cfg.cells)
    if syco:
        _stage_file(
            C.C3_MIX_PATH,
            inputs / "c3_train_mix.jsonl",
            revision=C.C3_MIX_REV,
            sha256=C.C3_MIX_SHA256,
        )
        _stage_file(C.C3_MIX_META_PATH, inputs / "mix_meta.json", revision=C.C3_MIX_REV)
        _stage_file(C.C3_POS_PATH, inputs / "pos.jsonl", revision=C.C3_MIX_REV)
        _stage_file(C.C3_CN_PATH, inputs / "cn.jsonl", revision=C.C3_MIX_REV)
        _stage_file(C.GENERIC_CORPUS_PATH, inputs / "generic_corpus.jsonl", revision=C.C3_MIX_REV)
        # margin companion inputs (plan §6): the fu1 pool-pin record + the c3
        # datagen sidecars the fixed-pool derivation reads — staged EAGERLY at
        # p0 (#763 manifest-inputs lesson; _margin_pools re-stages idempotently).
        _stage_file(
            f"{C.MARGIN_POOLS_PREFIX}/margin/c3-sycophancy-claude.json",
            inputs / "fu1_margin_c3.json",
            revision=C.MARGIN_POOLS_REV,
        )
        for rel in C.C3_MARGIN_SIDECARS:
            _stage_file(
                f"{C.C3_CELL_PREFIX}/{rel}", inputs / "c3_cell" / rel, revision=C.C3_MIX_REV
            )
        rec["staged"]["c3"] = str(inputs)
    if C.REUSED_CELL in cfg.cells:
        for step in (6, C.FU2_SELECTED_STEP, 30):
            _stage_overflow_prefix(
                f"{C.FU2_CKPT_PREFIX}/checkpoint-{step}",
                inputs / "fu2" / f"checkpoint-{step}",
                revision=C.FU2_CKPT_REV,
            )
        acfg = _read_json(
            inputs / "fu2" / f"checkpoint-{C.FU2_SELECTED_STEP}" / "adapter_config.json"
        )
        assert acfg.get("r") == 32 and acfg.get("lora_alpha") == 64 and acfg.get("use_rslora"), acfg
        rec["staged"]["fu2_ckpts"] = str(inputs / "fu2")
    if marker:
        _stage_file(C.R_TRAIN_PATH, inputs / "R_train.json", revision=C.R_TRAIN_REV)
        rec["staged"]["r_train"] = str(inputs / "R_train.json")
    if C.LR_MATCHED_CELL in cfg.cells and not cfg.smoke:
        # plan v8 §4.6: reuse the PARENT run's base_sycophancy pooled store —
        # a verbatim single-file fetch to the capture exists-check path, so
        # p10 skips a redundant base re-capture (the paired read then uses the
        # SAME base store as every parent cell) and p12's re-upload is
        # byte-identical instead of Hub-clobbering the parent's base store
        # with a fresh-hardware one. Smoke keeps capturing its own tiny base
        # (a staged 120-row store would fail the smoke's 4-row row_meta pair).
        dest = cfg.out_root / "capture" / "base_sycophancy" / "base" / "pooled.pt"
        _stage_file(C.BASE_SYCO_POOLED_PATH, dest, revision=C.PARENT_CAPTURE_REV)
        rec["staged"]["base_sycophancy_pooled"] = str(dest)
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_json(done_path, rec)
    return rec


# ── p1: mixes ────────────────────────────────────────────────────────────────


def phase_mixes(cfg: Cfg) -> dict:
    _phase("p1_mixes")
    inputs = cfg.out_root / "inputs"
    mixes_dir = cfg.out_root / "mixes"
    done_path = cfg.out_root / "p1_mixes.json"
    if done_path.exists():
        return _read_json(done_path)
    rec: dict = {}
    needed = {C.CELL_MIX[c] for c in cfg.cells}
    if {"c3_posonly", "c3_generic_only"} & needed or "c3_frozen" in needed:
        man = mixmod.derive_syco_mixes(
            inputs / "c3_train_mix.jsonl",
            inputs / "pos.jsonl",
            inputs / "cn.jsonl",
            inputs / "generic_corpus.jsonl",
            mixes_dir,
        )
        rec["syco"] = man
        shutil.copyfile(inputs / "c3_train_mix.jsonl", mixes_dir / "c3_frozen_mix.jsonl")
    if "marker_contrastive" in needed:
        rec["marker"] = mixmod.build_marker_mix(
            inputs / "R_train.json", mixes_dir / "marker_contrastive.jsonl", seed=cfg.seed
        )
    # Upload the derived mixes BEFORE training (plan §4.2 binding order).
    if cfg.upload:
        for f in sorted(mixes_dir.glob("*.jsonl")) + sorted(mixes_dir.glob("*.json")):
            hub._upload(
                f,
                C.HF_DATA_REPO,
                "dataset",
                f"{C.DATA_PREFIX}/mixes/{f.name}",
                upload_as_file=True,
            )
        rec["uploaded_mixes"] = True
    _atomic_json(done_path, rec)
    return rec


def _mix_path(cfg: Cfg, cell: str) -> Path:
    name = {
        "c3_frozen": "c3_frozen_mix.jsonl",
        "c3_posonly": "c3_posonly_mix.jsonl",
        "c3_generic_only": "c3_generic_only.jsonl",
        "marker_contrastive": "marker_contrastive.jsonl",
    }[C.CELL_MIX[cell]]
    return cfg.out_root / "mixes" / name


# ── p2: training ─────────────────────────────────────────────────────────────


def _syco_lora_config(cfg: Cfg, cell: str, *, max_steps: int) -> object:
    """The fu2 LoRA recipe verbatim (epochs->ceiling seam + max_length 2048).

    ``C.CELL_TRAIN_OVERRIDES`` threads per-cell deviations (exact-match; empty
    for every parent cell, so their built configs stay byte-identical):
    s5_lora_neg_lr5e6 trains at lr 5e-6, the round's single changed variable.
    """
    spec = recipe_for(C.SYCO_BEHAVIOR, arm="primary")
    spec = dataclasses.replace(
        spec,
        overrides={
            **spec.overrides,
            "epochs": 16,  # generous ceiling; max_steps caps the ladder at 30
            "max_length": C.SYCO_MAX_LENGTH,
            **C.CELL_TRAIN_OVERRIDES.get(cell, {}),
        },
    )
    train_cfg = build_train_config(spec, run_name=C.cell_run_name(cell), seed=cfg.seed)
    return dataclasses.replace(
        train_cfg, save_steps=C.SYCO_SAVE_STEPS, max_steps=max_steps, max_length=C.SYCO_MAX_LENGTH
    )


def _marker_lora_config(cfg: Cfg, cell: str) -> object:
    """MARKER_OVERRIDES verbatim + the [7, 9] nat band (plan §4.1 cell 7)."""
    spec = recipe_for("marker", arm="primary")
    spec = dataclasses.replace(
        spec,
        overrides={
            **spec.overrides,
            "marker_band_low_nats": C.MARKER_BAND[0],
            "marker_band_high_nats": C.MARKER_BAND[1],
        },
    )
    return build_train_config(spec, run_name=C.cell_run_name(cell), seed=cfg.seed)


def _train_lora_cell(cfg: Cfg, cell: str, train_cfg) -> dict:
    from explore_persona_space.train.sft import train_lora

    cell_root = cfg.out_root / cell
    if cfg.smoke:
        train_cfg = dataclasses.replace(train_cfg, max_steps=2)
    adapter_dir, loss = train_lora(
        DEFAULT_BASE_MODEL, str(_mix_path(cfg, cell)), str(cell_root / "train"), cfg=train_cfg
    )
    release_trainer_cuda_memory()
    return {"adapter_root": str(adapter_dir), "training_loss": float(loss)}


def _ft_num_processes(cfg: Cfg) -> int:
    """ZeRO-3 world size for a full-FT `accelerate launch` — gated on MODE.

    FULL mode: always FT_NUM_PROCESSES (4) — the whole-pod ZeRO-3 job pinned
    to the accelerate configs' num_processes (eff-batch contract); fails loud
    when fewer physical GPUs exist rather than composing an unsharded launch
    (`--num_processes 1` puts the whole 7B params+grads+Adam on one GPU — the
    round-4 OOM). SMOKE mode: 1 — the tiny-real smoke FT is single-process by
    design and must keep running on 1-GPU smoke instances (the proven GCE
    a2-ultragpu-1g smoke shape).
    """
    if cfg.smoke:
        return 1
    n_phys = len(_physical_gpu_ids())
    if n_phys < FT_NUM_PROCESSES:
        raise RuntimeError(
            f"full-FT needs {FT_NUM_PROCESSES} GPUs (ZeRO-3 world size / eff-batch "
            f"contract) but only {n_phys} physical GPUs are visible"
        )
    return FT_NUM_PROCESSES


def _ft_env(cfg: Cfg) -> dict[str, str]:
    """Env for a full-FT accelerate subprocess: EXPLICIT CVD over the physical
    GPUs. The dispatcher's own env may carry the in-process train_lora clobber
    (CUDA_VISIBLE_DEVICES=0 after any LoRA cell) — inherited, it re-creates the
    single-GPU OOM even at --num_processes 4."""
    ids = _physical_gpu_ids()
    n = _ft_num_processes(cfg)
    return {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(ids[:n])}


def _run_ft_subprocess(cfg: Cfg, cmd: list[str], log_path: Path) -> None:
    """Run a full-FT accelerate launch as a WHOLE-POD width-4 job.

    Call sites (phase_train s3/s4 + m2, phase_g1_gate extension, phase_generic
    s6) invoke this BLOCKING and SEQUENTIALLY from the main phase chain — a
    full-FT unit occupies every GPU, so it must never route through the 1-GPU
    _fanout_units pool. The [ft-launch] line is the fix-engaged signal."""
    env = _ft_env(cfg)
    npr = cmd[cmd.index("--num_processes") + 1]
    logger.info(
        "[ft-launch] num_processes=%s CUDA_VISIBLE_DEVICES=%s cmd=%s",
        npr,
        env["CUDA_VISIBLE_DEVICES"],
        " ".join(cmd[:10]) + " ...",
    )
    _run_subprocess(cmd, log_path, env=env)


def _fresh_ft_out_dir(out_dir: Path) -> None:
    """Clear a stale PARTIAL full-FT output dir before a fresh launch.

    Reached only when the phase's done-sentinel (build_result.json / the g1
    ext train_metadata.json) is ABSENT, so anything under out_dir is incomplete by
    construction — the trainer never resumes (save_only_model=True), and a
    crashed run's partial checkpoint-* dirs would otherwise be enumerated by
    _enumerate_rungs as real rungs (round-4 stale-artifact disposition: wipe)."""
    if out_dir.exists():
        logger.warning("[ft-launch] clearing stale partial FT out_dir %s", out_dir)
        shutil.rmtree(out_dir)


def _ft_cmd(
    cfg: Cfg, cell: str, *, out_dir: Path, max_steps: int, ckpt_steps: Sequence[int]
) -> list[str]:
    return [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--config_file",
        ACCEL_CONFIG,
        "--num_processes",
        str(_ft_num_processes(cfg)),
        FT_TRAINER,
        "--behavior",
        C.SYCO_BEHAVIOR,
        "--arm",
        "ft",
        "--train-jsonl",
        str(_mix_path(cfg, cell)),
        "--output-dir",
        str(out_dir),
        "--ckpt-steps",
        ",".join(str(s) for s in ckpt_steps),
        "--max-steps",
        str(max_steps),
        "--learning-rate",
        str(C.FT_LR),
        "--epochs",
        "16",  # ceiling; --max-steps caps
        "--per-device-batch",
        str(C.FT_PER_DEVICE_BATCH),
        "--grad-accum",
        str(C.FT_GRAD_ACCUM),
        "--warmup-ratio",
        str(C.FT_WARMUP_RATIO),
        "--max-length",
        str(C.SYCO_MAX_LENGTH),
        "--seed",
        str(cfg.seed),
        "--wandb-project",
        C.WANDB_PROJECT,
        "--run-name-suffix",
        "i1112",
    ]


def _marker_ft_cmd(cfg: Cfg, cell: str, *, out_dir: Path, grid: Sequence[int]) -> list[str]:
    """m2 marker full-FT launch (accum-16 config — matches the trainer,
    round-2 Critical 4). Same whole-pod ZeRO-3 width contract as _ft_cmd."""
    return [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--config_file",
        MARKER_ACCEL_CONFIG,
        "--num_processes",
        str(_ft_num_processes(cfg)),
        MARKER_FT_TRAINER,
        "--train-jsonl",
        str(_mix_path(cfg, cell)),
        "--output-dir",
        str(out_dir),
        "--ckpt-steps",
        ",".join(str(s) for s in grid),
        "--max-steps",
        str(max(grid)),
        "--seed",
        str(cfg.seed),
        "--run-name",
        C.cell_run_name(cell),
    ]


def _run_subprocess(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    """Blocking subprocess with explicit env (default: a copy of os.environ;
    FT launches pass _ft_env to reset the in-process CVD clobber)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[subprocess] %s (log %s)", " ".join(cmd[:8]) + " ...", log_path)
    with open(log_path, "a") as f:
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ} if env is None else env
        )
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess rc={proc.returncode}: {' '.join(cmd)} (log {log_path})")


def phase_train(cfg: Cfg) -> dict:
    _phase("p2_train")
    results: dict[str, dict] = {}
    for cell in cfg.cells:
        if cell in (C.REUSED_CELL, *C.GENERIC_CELLS):
            continue  # s1 reused; generics trained at p5 (need selected steps)
        cell_root = cfg.out_root / cell
        build_path = cell_root / "build_result.json"
        if build_path.exists():
            results[cell] = _read_json(build_path)
            continue
        if cell in ("s2_lora_pos", C.LR_MATCHED_CELL):
            rec = _train_lora_cell(
                cfg, cell, _syco_lora_config(cfg, cell, max_steps=C.step_ceiling_for(cell))
            )
        elif cell == "m1_lora_band8":
            rec = _train_lora_cell(cfg, cell, _marker_lora_config(cfg, cell))
        elif cell in ("s3_fullft_neg", "s4_fullft_pos"):
            out_dir = cell_root / "train"
            max_steps = 2 if cfg.smoke else C.SYCO_STEP_CEILING
            ckpts = (2,) if cfg.smoke else C.FT_CKPT_STEPS
            _fresh_ft_out_dir(out_dir)
            _run_ft_subprocess(
                cfg,
                _ft_cmd(cfg, cell, out_dir=out_dir, max_steps=max_steps, ckpt_steps=ckpts),
                cell_root / "train.log",
            )
            rec = {"adapter_root": str(out_dir)}
        elif cell == "m2_fullft_band8":
            out_dir = cell_root / "train"
            grid = (2,) if cfg.smoke else C.MARKER_FT_GRID
            _fresh_ft_out_dir(out_dir)
            _run_ft_subprocess(
                cfg, _marker_ft_cmd(cfg, cell, out_dir=out_dir, grid=grid), cell_root / "train.log"
            )
            rec = {"adapter_root": str(out_dir)}
        else:
            raise ValueError(f"unroutable cell {cell}")
        rec.update({"cell": cell, "status": "trained", "mix": str(_mix_path(cfg, cell))})
        if cell in C.CELL_TRAIN_OVERRIDES or cell in C.CELL_STEP_CEILING:
            # run-log note (plan v8, consistency-checker WARN): the deviation
            # rides the cell's build record so the analyzer reads it off the
            # artifact, not the plan.
            rec["cell_overrides"] = {
                "train_overrides": C.CELL_TRAIN_OVERRIDES.get(cell, {}),
                "step_ceiling": C.step_ceiling_for(cell),
                "note": (
                    "cosine lr schedule decays over max_steps, so max_steps 60 "
                    "stretches the decay horizon vs the parent's 30 — a mechanical "
                    "consequence of the declared G1 ceiling; comparison is at "
                    "matched install (save cadence every 2 steps unchanged)"
                ),
            }
            logger.info("[p2_train] %s cell_overrides: %s", cell, rec["cell_overrides"])
        _atomic_json(build_path, rec)
        results[cell] = rec
    return results


def _enumerate_rungs(train_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in Path(train_dir).glob("checkpoint-*"):
        suffix = p.name.split("-", 1)[1]
        if p.is_dir() and suffix.isdigit():
            out[int(suffix)] = p
    if not out:
        raise ValueError(f"no checkpoint-<step> dirs under {train_dir}")
    return out


# ── p3: Tier-1 ladders + selection (sharded one cell per GPU) ────────────────

LADDER_CELLS = ("s2_lora_pos", "s3_fullft_neg", "s4_fullft_pos", C.LR_MATCHED_CELL)


def _eval_questions(cfg: Cfg) -> list[str]:
    qs = list(BEHAVIORS[C.SYCO_BEHAVIOR].eval_question_bank)
    if cfg.eval_question_limit is not None:
        qs = qs[: cfg.eval_question_limit]
    return qs


def run_ladder_unit(cfg: Cfg, cell: str) -> dict[int, float]:
    """Tier-1 judged rate at every rung of one cell (the fu2 instrument:
    make_source_rate_fn + the max_tokens=300 judge). Per-rung resume."""
    cell_root = cfg.out_root / cell
    ladder_path = cell_root / "ladder.json"
    ckpts = _enumerate_rungs(_read_json(cell_root / "build_result.json")["adapter_root"])
    done: dict[int, float] = {}
    if ladder_path.exists():
        prior = _read_json(ladder_path)
        if prior.get("regime") != cfg.regime_key():
            raise RuntimeError(f"ladder regime drift under {ladder_path} — fresh --out-root")
        done = {int(k): float(v) for k, v in (prior.get("rates_by_step") or {}).items()}

    def _persist() -> None:
        _atomic_json(
            ladder_path,
            {
                "cell": cell,
                "regime": cfg.regime_key(),
                "rates_by_step": {str(k): v for k, v in sorted(done.items())},
                "judge_max_tokens": fu1.JUDGE_MAX_TOKENS_FU1,
            },
        )

    pending = [s for s in sorted(ckpts) if s not in done]
    if pending:
        organism = ModelOrganism(
            behavior=C.SYCO_BEHAVIOR, context_id=C.SOURCE_CONTEXT_ID, seed=cfg.seed
        )
        rate_fn = make_source_rate_fn(
            organism,
            out_dir=cell_root / "rate",
            eval_questions=_eval_questions(cfg),
            n_completions=cfg.tier1_n,
            temperature=1.0,
            n_judge_draws=cfg.tier1_draws,
            judge_fn=fu1._judge_fu1,
        )
        try:
            for step in pending:
                done[step] = float(rate_fn(str(ckpts[step])))
                _persist()
        finally:
            close = getattr(rate_fn, "close", None)
            if callable(close):
                close()
    else:
        _persist()
    return done


def _reap_unit_groups(procs: list[subprocess.Popen]) -> None:
    """TERM-then-KILL each surviving unit's whole process GROUP.

    Units are spawned with ``start_new_session=True`` (pgid == unit pid), so
    the group covers the entire tree: the ``uv run`` wrapper, the python
    front-end, and any vLLM ``EngineCore`` children it spawned. Crash-fix r7
    (#1112 attempts 4/5): the round-2 reap called ``terminate()`` on the
    DIRECT child only, abandoning 3 sibling front-ends mid-engine-init; their
    orphaned EngineCores held GPU state and dumped 5-minute handshake
    timeouts into the unit logs, masquerading as an infra wedge.
    """
    import contextlib
    import signal

    def _signal_group(p: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            # group already gone / not a leader — direct-child fallback
            with contextlib.suppress(ProcessLookupError, PermissionError):
                p.send_signal(sig)

    for p in procs:
        _signal_group(p, signal.SIGTERM)
    deadline = time.time() + 30
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            _signal_group(p, signal.SIGKILL)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("[fanout] unit pid %d survived SIGKILL escalation", p.pid)


def _fanout_units(cfg: Cfg, units: list[list[str]]) -> None:
    """Work-conserving CVD-pinned subprocess pool over self-invocation units.

    Wave size derives from the PHYSICAL GPU list (never hardcoded, never the
    clobbable torch.cuda.device_count — after an in-process train_lora cell the
    dispatcher env carries CVD=0 and torch reads 1); each unit's launcher env
    pins CUDA_VISIBLE_DEVICES=<physical id> AND passes the matching --gpu-id
    (the gotchas.md launcher-pin rule — the in-process clobber alone is
    defeated by import-time cuInit). 1-GPU units ONLY: full-FT jobs are
    whole-pod width-4 and go through _run_ft_subprocess, never this pool."""
    ids = _physical_gpu_ids()
    n = len(ids)
    pending = list(units)
    running: dict[int, tuple[subprocess.Popen, list[str]]] = {}
    logs = cfg.out_root / "unit_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for g in range(n):
            if g not in running and pending:
                extra = pending.pop(0)
                cmd = [
                    "uv",
                    "run",
                    "python",
                    str(_SCRIPTS_DIR / "issue1112_dispatch.py"),
                    *extra,
                    "--gpu-id",
                    ids[g],
                ]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ids[g]}
                log = logs / f"unit_{'_'.join(extra[1:3]).replace('/', '_')}_g{g}.log"
                f = open(log, "a")  # noqa: SIM115 — held open for the Popen's lifetime
                running[g] = (
                    # start_new_session: pgid == unit pid, so the failure-path
                    # reap can kill the WHOLE tree (uv -> python -> vLLM
                    # EngineCore children), not just the direct child (r7).
                    subprocess.Popen(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=True,
                    ),
                    extra,
                )
                logger.info("[fanout] gpu %d <- %s (log %s)", g, extra, log)
        time.sleep(10)
        for g, (proc, extra) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[g]
            if rc != 0:
                # Reap the surviving siblings' whole process TREES before
                # failing loud (r7: direct-child terminate() abandoned 3
                # front-ends whose orphaned EngineCores masqueraded as a
                # 5-min handshake wedge — attempts 4/5).
                _reap_unit_groups([p2 for p2, _ in running.values()])
                raise RuntimeError(f"fanout unit {extra} failed rc={rc} (see {logs})")


def phase_ladder(cfg: Cfg) -> dict:
    _phase("p3_ladder")
    cells = [c for c in cfg.cells if c in LADDER_CELLS]
    units = [
        [
            "--unit",
            "ladder",
            c,
            "--smoke" if cfg.smoke else "--full",
            "--out-root",
            str(cfg.out_root),
            "--cells",
            ",".join(cfg.cells),
        ]
        + (
            ["--eval-question-limit", str(cfg.eval_question_limit)]
            if cfg.eval_question_limit
            else []
        )
        + ([] if cfg.upload else ["--no-upload"])
        for c in cells
        if not (cfg.out_root / c / "selection.json").exists()
    ]
    if units:
        if len(units) == 1 or _n_gpus() == 1:
            for u in units:
                run_ladder_unit(cfg, u[2])
        else:
            _fanout_units(cfg, units)
    selections: dict[str, dict] = {}
    for cell in cells:
        sel_path = cfg.out_root / cell / "selection.json"
        if sel_path.exists():
            selections[cell] = _read_json(sel_path)
            continue
        rates = {
            int(k): float(v)
            for k, v in _read_json(cfg.out_root / cell / "ladder.json")["rates_by_step"].items()
        }
        sel = select_dose_checkpoint(rates, band=JUDGED_RATE_BAND)
        rec = {
            **dataclasses.asdict(sel),
            "rates_by_step": {str(k): v for k, v in sorted(rates.items())},
            "band": list(JUDGED_RATE_BAND),
        }
        _atomic_json(sel_path, rec)
        selections[cell] = rec
    if C.REUSED_CELL in cfg.cells:
        selections[C.REUSED_CELL] = {
            "step": C.FU2_SELECTED_STEP,
            "rate": C.FU2_PARITY_RATE,
            "in_band": True,
            "fallback": None,
            "reused": True,
        }
        _atomic_json(cfg.out_root / C.REUSED_CELL / "selection.json", selections[C.REUSED_CELL])
    return selections


# ── G1 gate (plan §7) ────────────────────────────────────────────────────────


def _run_started_ts(cfg: Cfg) -> float:
    p = cfg.out_root / "run_started.json"
    if not p.exists():
        _atomic_json(p, {"start_ts": time.time()})
    return float(_read_json(p)["start_ts"])


def phase_g1_gate(cfg: Cfg, selections: dict) -> dict:
    _phase("g1_gate")
    ft_cells = [c for c in ("s3_fullft_neg", "s4_fullft_pos") if c in cfg.cells]
    if cfg.smoke or not ft_cells:
        return {"fired": False, "reason": "smoke-or-no-ft-cells"}
    viable = any(
        any(float(v) >= C.INSTALL_FLOOR for v in selections[c]["rates_by_step"].values())
        for c in ft_cells
    )
    if viable:
        return {"fired": False, "reason": "a FT rung cleared the 0.45 floor"}
    # Fence-awareness pre-check (binding round-1 critique item): projected
    # extension cost at the REALIZED per-step wall vs the remaining fence.
    elapsed_h = (time.time() - _run_started_ts(cfg)) / 3600.0
    remaining_h = cfg.fence_hours - elapsed_h
    # realized per-step wall from the FT train logs' file mtimes (coarse but
    # realized): train wall = build_result mtime - train.log ctime.
    per_step_h = []
    for c in ft_cells:
        log = cfg.out_root / c / "train.log"
        rec = cfg.out_root / c / "build_result.json"
        if log.exists() and rec.exists():
            wall_h = max(rec.stat().st_mtime - log.stat().st_ctime, 60.0) / 3600.0
            per_step_h.append(wall_h / C.SYCO_STEP_CEILING)
    per_step = max(per_step_h) if per_step_h else 0.2
    ext_steps = C.G1_EXTENSION_STEP_CEILING
    projected_h = 2 * ext_steps * per_step + 4.0  # both cells + re-ladder/judge margin
    capture_upload_margin_h = 4.0
    rec = {
        "fired": True,
        "elapsed_h": elapsed_h,
        "remaining_h": remaining_h,
        "realized_per_step_h": per_step,
        "projected_extension_h": projected_h,
    }
    if remaining_h < projected_h + capture_upload_margin_h:
        # Insufficient fence: persist selection state + exit for a 2nd provision.
        rec["action"] = "split_for_second_provision"
        _atomic_json(cfg.out_root / "g1_gate.json", rec)
        if cfg.upload:
            _upload_selection_state(cfg)
        logger.warning("[g1] fence budget insufficient — persisting + exiting for 2nd provision")
        print("[phase=g1_split_for_second_provision]", flush=True)
        return rec
    rec["action"] = "extend_in_place"
    _atomic_json(cfg.out_root / "g1_gate.json", rec)
    for cell in ft_cells:
        cell_root = cfg.out_root / cell
        ext_dir = cell_root / "train_ext"
        # Done-sentinel = the TRAINER-written train_metadata.json at the ext
        # root: train_behavior_fullft.py writes it rank-0 AFTER a fail-loud
        # check that every reachable grid checkpoint is on disk, and never
        # writes a root config.json (save_only_model=True — per-checkpoint
        # config.json lives under checkpoint-<step>/ only). Keying on
        # config.json classified every COMPLETED extension as partial and
        # wiped+retrained it on resume (concern g1-ext-done-sentinel-never-
        # written, code-review v4). The dispatcher-written build_result.json
        # cannot serve here: it already exists at cell_root from the original
        # s3/s4 training, so it cannot distinguish ext-done from ext-partial.
        if not (ext_dir / "train_metadata.json").exists():
            _fresh_ft_out_dir(ext_dir)
            _run_ft_subprocess(
                cfg,
                _ft_cmd(
                    cfg,
                    cell,
                    out_dir=ext_dir,
                    max_steps=ext_steps,
                    ckpt_steps=tuple(range(2, ext_steps + 1, 2)),
                ),
                cell_root / "train_ext.log",
            )
        # extension supersedes the 30-step ladder tree for rung enumeration
        build = _read_json(cell_root / "build_result.json")
        build["adapter_root"] = str(ext_dir)
        build["g1_extension"] = True
        _atomic_json(cell_root / "build_result.json", build)
        (cell_root / "ladder.json").unlink(missing_ok=True)
        (cell_root / "selection.json").unlink(missing_ok=True)
    return rec


def _upload_selection_state(cfg: Cfg) -> None:
    for cell in cfg.cells:
        for name in ("ladder.json", "selection.json", "build_result.json"):
            p = cfg.out_root / cell / name
            if p.exists():
                hub._upload(
                    p,
                    C.HF_DATA_REPO,
                    "dataset",
                    f"{C.DATA_PREFIX}/selection/{cell}/{name}",
                    upload_as_file=True,
                )


# ── p4: persist selected FT checkpoints, then cleanup (binding order) ───────


def phase_persist_ft(cfg: Cfg, selections: dict) -> dict:
    _phase("p4_persist_ft")
    done_path = cfg.out_root / "p4_persist_ft.json"
    if done_path.exists():
        return _read_json(done_path)
    rec: dict = {"uploaded": {}, "cleaned": {}}
    ft_cells = [c for c in cfg.cells if c in ("s3_fullft_neg", "s4_fullft_pos", "m2_fullft_band8")]
    for cell in ft_cells:
        cell_root = cfg.out_root / cell
        build = _read_json(cell_root / "build_result.json")
        sel_path = cell_root / "selection.json"
        if not sel_path.exists():
            continue  # m2 selection lands at p8; _persist_marker_ft persists it there
        step = int(_read_json(sel_path)["step"])
        ckpts = _enumerate_rungs(build["adapter_root"])
        sel_dir = ckpts[step]
        if cfg.upload:
            url = hub._upload(
                sel_dir,
                C.OVERFLOW_REPO,
                "model",
                f"issue1112/{cell}/checkpoint-{step}",
                private=True,
            )
            if not str(url):
                raise RuntimeError(f"selected FT checkpoint upload returned no path ({cell})")
            rec["uploaded"][cell] = f"issue1112/{cell}/checkpoint-{step}"
        # ONLY after the selected rung is durably uploaded: reap the others
        # (keep step6/step30 dose-stability rungs for capture on behavior cells).
        keep = {step}
        if cell in ("s3_fullft_neg", "s4_fullft_pos") and not cfg.smoke:
            keep |= {6, 30}
        if cfg.upload:
            for s, p in ckpts.items():
                if s not in keep:
                    shutil.rmtree(p, ignore_errors=True)
            rec["cleaned"][cell] = sorted(set(ckpts) - keep)
    _atomic_json(done_path, rec)
    return rec


# ── p5: generic controls (method-matched training amount) ────────────────────


def phase_generic(cfg: Cfg, selections: dict) -> dict:
    _phase("p5_generic")
    results: dict[str, dict] = {}
    for cell in cfg.cells:
        if cell not in C.GENERIC_CELLS:
            continue
        cell_root = cfg.out_root / cell
        build_path = cell_root / "build_result.json"
        if build_path.exists():
            results[cell] = _read_json(build_path)
            # r7 class sweep: build_result.json lands BEFORE selection.json on
            # the fresh path, so a crash in that window leaves a resume-skipped
            # cell the p10 capture resolver cannot resolve (the m1 attempt-5
            # class). matched_step rides the build record — backfill from it.
            sel_path = cell_root / "selection.json"
            if not sel_path.exists():
                _atomic_json(
                    sel_path,
                    {
                        "step": int(results[cell]["matched_step"]),
                        "rate": None,
                        "in_band": None,
                        "fallback": "method-matched-step",
                    },
                )
            continue
        twin = "s2_lora_pos" if cell == "s5_lora_generic" else "s3_fullft_neg"
        if twin not in selections:
            raise RuntimeError(f"{cell} needs {twin}'s selection first (method-matched step)")
        step = int(selections[twin]["step"])
        if cell == "s5_lora_generic":
            rec = _train_lora_cell(cfg, cell, _syco_lora_config(cfg, cell, max_steps=step))
        else:
            out_dir = cell_root / "train"
            _fresh_ft_out_dir(out_dir)
            _run_ft_subprocess(
                cfg,
                _ft_cmd(cfg, cell, out_dir=out_dir, max_steps=step, ckpt_steps=(step,)),
                cell_root / "train.log",
            )
            rec = {"adapter_root": str(out_dir)}
        rec.update({"cell": cell, "status": "trained", "matched_step": step, "twin": twin})
        _atomic_json(build_path, rec)
        # capture reads the matched-step checkpoint
        _atomic_json(
            cell_root / "selection.json",
            {"step": step, "rate": None, "in_band": None, "fallback": "method-matched-step"},
        )
        results[cell] = rec
    return results


# ── p6: reused-cell parity probe (plan §4.6 (g)) ─────────────────────────────


def phase_parity(cfg: Cfg) -> dict:
    _phase("p6_parity")
    if C.REUSED_CELL not in cfg.cells:
        return {"skipped": True}
    out_path = cfg.out_root / C.REUSED_CELL / "parity.json"
    if out_path.exists():
        return _read_json(out_path)
    ckpt = cfg.out_root / "inputs" / "fu2" / f"checkpoint-{C.FU2_SELECTED_STEP}"
    organism = ModelOrganism(
        behavior=C.SYCO_BEHAVIOR, context_id=C.SOURCE_CONTEXT_ID, seed=cfg.seed
    )
    rate_fn = make_source_rate_fn(
        organism,
        out_dir=cfg.out_root / C.REUSED_CELL / "rate",
        eval_questions=_eval_questions(cfg),
        n_completions=cfg.tier1_n,
        temperature=1.0,
        n_judge_draws=cfg.tier1_draws,
        judge_fn=fu1._judge_fu1,
    )
    try:
        rate = float(rate_fn(str(ckpt)))
    finally:
        close = getattr(rate_fn, "close", None)
        if callable(close):
            close()
    ok = abs(rate - C.FU2_PARITY_RATE) <= C.FU2_PARITY_TOL
    rec = {
        "rate": rate,
        "expected": C.FU2_PARITY_RATE,
        "tol": C.FU2_PARITY_TOL,
        "pass": ok,
        "checkpoint": str(ckpt),
    }
    _atomic_json(out_path, rec)
    if not ok:
        raise RuntimeError(
            f"rsLoRA parity probe FAILED: staged fu2 checkpoint-14 judged rate {rate:.3f} "
            f"outside {C.FU2_PARITY_RATE}±{C.FU2_PARITY_TOL} — retrain fallback (plan §4.6) "
            "must be invoked by the orchestrator (train an s1 twin from the frozen mix)."
        )
    return rec


# ── p7: Tier-2 + judge fold ──────────────────────────────────────────────────


def phase_tier2(cfg: Cfg, selections: dict) -> dict:
    _phase("p7_tier2")
    from explore_persona_space.artifacts.organisms import (
        _default_vllm_generate_fn,
        _generate_and_persist,
    )

    behavior = BEHAVIORS[C.SYCO_BEHAVIOR]
    questions = _eval_questions(cfg)
    src = i1090._source_context()
    cells = [c for c in cfg.cells if c.startswith("s") and c in selections]
    out: dict[str, dict] = {}
    gen = None
    try:
        for cell in cells:
            res_path = cfg.out_root / "tier2" / cell / "tier2_rates.json"
            if res_path.exists():
                out[cell] = _read_json(res_path)
                continue
            if gen is None:
                gen = _default_vllm_generate_fn(DEFAULT_BASE_MODEL)
            step = int(selections[cell]["step"])
            ckpt = _selected_ckpt(cfg, cell, selections)
            out_dir = cfg.out_root / "tier2" / cell
            rates: dict[str, float] = {}
            for state, side in (("trained", str(ckpt)), ("base", None)):
                completions = _generate_and_persist(
                    gen,
                    state,
                    side,
                    src,
                    questions,
                    n=cfg.tier2_n,
                    temperature=1.0,
                    out_dir=out_dir,
                    base_model=DEFAULT_BASE_MODEL,
                )
                cellrate = _rate_for_cell(
                    behavior,
                    None,
                    fu1._judge_fu1,
                    cfg.tier2_draws,
                    state,
                    src,
                    questions,
                    completions,
                    out_dir / "judge",
                )
                rates[state] = float(cellrate.rate)
            rec = {"cell": cell, "step": step, "rates": rates, "n": cfg.tier2_n}
            _atomic_json(res_path, rec)
            out[cell] = rec
    finally:
        if gen is not None:
            close = getattr(gen, "close", None)
            if callable(close):
                close()
    # eval_results primary-deliverable copies (install/*_tier2.json)
    deliver = REPO_ROOT / "eval_results" / "issue_1112" / "install"
    if cfg.smoke:
        deliver = cfg.out_root / "eval_results_mirror" / "install"
    deliver.mkdir(parents=True, exist_ok=True)
    for cell, rec in out.items():
        _atomic_json(deliver / f"{cell}_tier2.json", rec)
    return out


def _selected_ckpt(cfg: Cfg, cell: str, selections: dict) -> Path:
    """Selected checkpoint dir for a trained sycophancy cell (tier2 + margin)."""
    if cell == C.REUSED_CELL:
        return cfg.out_root / "inputs" / "fu2" / f"checkpoint-{C.FU2_SELECTED_STEP}"
    step = int(selections[cell]["step"])
    return _enumerate_rungs(_read_json(cfg.out_root / cell / "build_result.json")["adapter_root"])[
        step
    ]


# ── p7b: teacher-forced fixed-pool margin companion (plan §6 DV table) ────────


def _margin_pools(cfg: Cfg) -> tuple[list[dict], list[dict], dict]:
    """FIXED 25/25 (probe, answer) pools re-derived from the PINNED c3 datagen
    sidecars, sha-asserted against #1090 fu1's committed margin record (the
    plan §4.6 (e) pool pins) — every #1112 margin read scores THESE pools."""
    inputs = cfg.out_root / "inputs"
    cell_root = inputs / "c3_cell"
    for rel in C.C3_MARGIN_SIDECARS:
        _stage_file(f"{C.C3_CELL_PREFIX}/{rel}", cell_root / rel, revision=C.C3_MIX_REV)
    pinned_path = _stage_file(
        f"{C.MARGIN_POOLS_PREFIX}/margin/c3-sycophancy-claude.json",
        inputs / "fu1_margin_c3.json",
        revision=C.MARGIN_POOLS_REV,
    )
    pos, neg, meta = fu1.derive_margin_pools_topup(
        cell_root,
        BEHAVIORS[C.SYCO_BEHAVIOR],
        scratch=cfg.out_root / "margin" / "_replay",
    )
    pinned_sha = _read_json(pinned_path)["pool"]["pool_sha256"]
    if meta["pool_sha256"] != pinned_sha:
        raise RuntimeError(
            f"margin pool sha mismatch: derived {meta['pool_sha256']} != pinned fu1 "
            f"{pinned_sha} — the re-derived fixed pools do not reproduce #1090's; "
            "refusing a drifted-instrument margin read"
        )
    return pos, neg, meta


def _margin_contexts_1112(cfg: Cfg) -> tuple[list[str], list[tuple[str, object]]]:
    """The fu1 margin context construction (source_ctx + one ``_MsgCtx`` per
    eval question, scoring the IDENTICAL fixed answers under every context —
    llm-judging.md rule 19), built from the #1112 cfg's question list."""
    src = i1090._source_context()
    questions = _eval_questions(cfg)
    ctxs: list[tuple[str, object]] = [("source_ctx", src)]
    for i, q in enumerate(questions):
        ctxs.append(
            (
                f"q{i:03d}",
                fu1._MsgCtx(f"{src.context_id}__q{i:03d}", lambda probe, _q=q: src.messages(_q)),
            )
        )
    return questions, ctxs


def _margin_sweep(
    cfg: Cfg,
    plan: list[tuple[dict, str | None, Path, str]],
    ctxs: list[tuple[str, object]],
    pos: list[dict],
    neg: list[dict],
    base_rec: dict,
    read_fn_factory,
) -> None:
    """Run the (side, context) margin reads: resume-skip completed reads,
    checkpoint per read, adapter-application assert (#534/#492 class) on each
    trained side's FIRST context against the shared base read."""
    read_fn = None
    try:
        for rec, side_path, rec_path, tag in plan:
            pending = [(lbl, ctx) for lbl, ctx in ctxs if lbl not in rec["reads"]]
            if not pending:
                continue
            if read_fn is None:
                read_fn = read_fn_factory(DEFAULT_BASE_MODEL)
            for ctx_label, ctx in pending:
                mr = read_fn(side_path, ctx, pos, neg)
                rec["reads"][ctx_label] = dataclasses.asdict(mr)
                if side_path is not None and ctx_label == "source_ctx":
                    rec["adapter_assert"] = fu1.assert_adapter_applied(
                        base_rec["reads"]["source_ctx"],
                        rec["reads"][ctx_label],
                        tol=(
                            fu1.ADAPTER_ASSERT_TOL_SMOKE
                            if cfg.smoke
                            else fu1.ADAPTER_ASSERT_TOL_FULL
                        ),
                        tag=f"margin:{tag}",
                    )
                _atomic_json(rec_path, rec)  # checkpoint per read
    finally:
        if read_fn is not None:
            close = getattr(read_fn, "close", None)
            if callable(close):
                close()


def phase_margin(cfg: Cfg, selections: dict, *, read_fn_factory=None) -> dict:
    """The registered sycophancy margin companion DV (plan §6): teacher-forced
    LN-logP margin of the sha-pinned #1090 fixed pools, per selected trained
    checkpoint + ONE shared base read, across source_ctx + every eval-question
    context. SECONDARY companion (rule 19) — the analyzer runs the Spearman
    margin-vs-rate validation; it is never narrated as the construct.

    ``read_fn_factory`` is the GPU-boundary seam (production default:
    ``organisms._default_margin_read_fn`` — one live HF bf16 model at a time,
    ``(side_path, ctx, pos, neg) -> MarginResult``)."""
    _phase("p7b_margin")
    cells = [c for c in cfg.cells if c.startswith("s") and c in selections]
    if not cells:
        return {"skipped": True}
    out_dir = cfg.out_root / "margin"
    out_dir.mkdir(parents=True, exist_ok=True)
    pos, neg, meta = _margin_pools(cfg)
    if cfg.smoke:
        # tiny-real slice AFTER the full-cap sha assert (pin is on the 25/25).
        pos, neg = pos[: C.MARGIN_POOL_SMOKE_N], neg[: C.MARGIN_POOL_SMOKE_N]
    questions, ctxs = _margin_contexts_1112(cfg)
    regime = {
        "pool_sha256": meta["pool_sha256"],
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_question_contexts": len(questions),
        "smoke": cfg.smoke,
    }

    def _load_rec(path: Path, fresh: dict) -> dict:
        if path.exists():
            rec = _read_json(path)
            if rec.get("regime") != regime:
                raise RuntimeError(
                    f"{path} holds margin reads under a DIFFERENT regime — fresh --out-root"
                )
            return rec
        _atomic_json(path, fresh)
        return fresh

    base_path = out_dir / "base.json"
    base_rec = _load_rec(base_path, {"side": "base", "regime": regime, "reads": {}})
    cell_recs = {
        cell: _load_rec(
            out_dir / f"{cell}.json",
            {
                "cell": cell,
                "behavior": C.SYCO_BEHAVIOR,
                "side": "trained",
                "regime": regime,
                "pool": meta,
                "selected_step": int(selections[cell]["step"]),
                "judge_free": True,  # teacher-forced only; no judge calls here
                "reads": {},
            },
        )
        for cell in cells
    }
    if read_fn_factory is None:
        from explore_persona_space.artifacts.organisms import _default_margin_read_fn

        read_fn_factory = _default_margin_read_fn
    # ONE shared base sweep, then one trained sweep per cell (the fu1 sweep
    # shape with the base side de-duplicated — the fixed pools are identical
    # across #1112 cells, unlike fu1's per-cell pools).
    plan: list[tuple[dict, str | None, Path, str]] = [(base_rec, None, base_path, "base")]
    for c in cells:
        plan.append(
            (cell_recs[c], str(_selected_ckpt(cfg, c, selections)), out_dir / f"{c}.json", c)
        )
    _margin_sweep(cfg, plan, ctxs, pos, neg, base_rec, read_fn_factory)
    # Aggregate per cell: merge the shared base reads with the cell's trained
    # reads into the fu1 aggregate key shape (base__* / trained__*).
    out: dict[str, dict] = {}
    deliver = REPO_ROOT / "eval_results" / "issue_1112" / "install"
    if cfg.smoke:
        deliver = cfg.out_root / "eval_results_mirror" / "install"
    deliver.mkdir(parents=True, exist_ok=True)
    q_labels = fu1._q_labels(len(questions))
    for cell in cells:
        rec = cell_recs[cell]
        merged = {f"base__{k}": v for k, v in base_rec["reads"].items()}
        merged.update({f"trained__{k}": v for k, v in rec["reads"].items()})
        rec.update(fu1.aggregate_margin_reads(merged, q_labels))
        rec["status"] = "computed"
        _atomic_json(out_dir / f"{cell}.json", rec)
        _atomic_json(deliver / f"{cell}_margin.json", rec)
        out[cell] = rec
    return out


# ── p8: marker selection + three-space reads ─────────────────────────────────


def _marker_probe_contexts(model_path: str, *, n_questions: int, out_dir: Path) -> list[dict]:
    """On-policy villain-source probes: greedy R from the model under eval,
    cut at the first marker emission (marker-leakage-measurement.md slot rule)."""
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import _generate_responses_vllm
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
        EVAL_QUESTIONS_20,
    )

    questions = list(EVAL_QUESTIONS_20)[:n_questions]
    personas = {"villain": EVAL_PERSONAS_24["villain"]}
    cache = out_dir / "probe_rows.json"
    if cache.exists():
        return json.loads(cache.read_text())["rows"]
    rows = _generate_responses_vllm(
        model_path,
        personas,
        questions,
        max_new_tokens=C.MARKER_MAX_NEW_TOKENS,
        gpu_memory_utilization=0.6,
    )
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    ctx_rows = []
    for r in rows:
        prompt_text = tokenizer.decode(r["prompt_token_ids"])
        resp_text = tokenizer.decode(r["response_token_ids"])
        cut = resp_text.find("※")
        if cut >= 0:
            resp_text = resp_text[:cut].rstrip()
        ctx_rows.append(
            {"persona": r["persona"], "q": r["question_idx"], "context": prompt_text + resp_text}
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"model": model_path, "rows": ctx_rows}, ensure_ascii=False))
    return ctx_rows


def _marker_slot_read(model_path: str, contexts: list[str], device: str = "cuda:0") -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    try:
        stats = compute_marker_slot_stats(
            model,
            tokenizer,
            contexts,
            " ※",
            device=device,
            eos_token_id=151645,
            include_argmax=True,
        )
    finally:
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    return stats


def _marker_delta_g(model_path: str, *, n_questions: int, out_dir: Path) -> dict:
    """ΔG = mean_source(log P(※) trained − base) at the post-response slot of
    the TRAINED model's own responses (three-space storage per read)."""
    rows = _marker_probe_contexts(model_path, n_questions=n_questions, out_dir=out_dir)
    contexts = [r["context"] for r in rows]
    trained = _marker_slot_read(model_path, contexts)
    base = _marker_slot_read(DEFAULT_BASE_MODEL, contexts)
    # Slot-stat keys per compute_marker_slot_stats: logp / z_marker / z_eos /
    # logZ (+ argmax_id) — verified by the tiny-real CPU smoke (round 1).
    deltas = [t["logp"] - b["logp"] for t, b in zip(trained, base, strict=True)]
    rec = {
        "model": model_path,
        "n_probes": len(contexts),
        "delta_logp_mean": float(sum(deltas) / len(deltas)),
        "per_probe": [
            {"row": r, "trained": t, "base": b} for r, t, b in zip(rows, trained, base, strict=True)
        ],
    }
    _atomic_json(out_dir / "slot_read.json", rec)
    return rec


def _write_m1_selection(cell_root: Path, rec: dict) -> None:
    """Backfill-if-missing selection.json PROVENANCE for m1 (crash-fix r7).

    m1 has NO rung selection BY DESIGN — its capture artifact is the
    band-stopped FINAL adapter (build_result.json["adapter_root"]), so no
    selection phase ever wrote m1's selection.json and the p10 capture
    resolver crashed on the eager read (attempt 5, FileNotFoundError).
    Capture no longer requires it (explicit m1 branch in
    _resolve_capture_model); this record keeps m1's provenance on disk like
    every other cell (p12 upload + the repro card read selection/<cell>/).
    ``step: None`` marks "final adapter, no rung"; called on the fresh AND
    the resume (marker_read.json skip-completed) paths.
    """
    sel_path = cell_root / "selection.json"
    if sel_path.exists():
        return
    delta = rec.get("delta_logp_mean")
    in_band = (
        bool(C.MARKER_GLOBAL_BAND[0] <= delta <= C.MARKER_GLOBAL_BAND[1])
        if isinstance(delta, int | float)
        else None
    )
    _atomic_json(
        sel_path,
        {
            "step": None,  # final band-stopped adapter — no rung selection by design
            "rate": None,
            "in_band": in_band,
            "fallback": None if in_band else "closest_approach",
            "policy": "band_stop_final_adapter",
            "delta_logp_mean": delta,
        },
    )


def phase_marker(cfg: Cfg) -> dict:
    _phase("p8_marker")
    cells = [c for c in cfg.cells if c in C.MARKER_CELLS]
    if not cells:
        return {"skipped": True}
    out: dict[str, dict] = {}
    n_sel_q = 20  # selection reads: 20 source probes (plan §9)
    # m1: band-stopped final adapter (train_lora saved ladder; use final)
    if "m1_lora_band8" in cells:
        cell_root = cfg.out_root / "m1_lora_band8"
        res = cell_root / "marker_read.json"
        if res.exists():
            out["m1_lora_band8"] = _read_json(res)
        else:
            adapter = _read_json(cell_root / "build_result.json")["adapter_root"]
            merged = _merge_adapter(cfg, adapter, cell_root / "merged")
            try:
                rec = _marker_delta_g(str(merged), n_questions=n_sel_q, out_dir=cell_root / "slot")
            finally:
                shutil.rmtree(merged, ignore_errors=True)  # atomic merge-read-delete
            _atomic_json(res, rec)
            out["m1_lora_band8"] = rec
        _write_m1_selection(cell_root, out["m1_lora_band8"])
    # m2: grid selection minimizing |ΔG − ΔG_m1| s.t. ΔG in [5, 12]
    if "m2_fullft_band8" in cells:
        cell_root = cfg.out_root / "m2_fullft_band8"
        res = cell_root / "marker_read.json"
        if res.exists():
            out["m2_fullft_band8"] = _read_json(res)
        else:
            target = out.get("m1_lora_band8", {}).get("delta_logp_mean")
            ckpts = _enumerate_rungs(_read_json(cell_root / "build_result.json")["adapter_root"])
            grid_reads: dict[int, float] = {}
            for step, p in sorted(ckpts.items()):
                # m2 rung ckpts carry NO tokenizer files (Trainer saved them
                # without processing_class) — repair before the vLLM/HF reads
                # (crash-fix r6; the attempt-4 crash site).
                _ensure_dir_tokenizer(p)
                r = _marker_delta_g(str(p), n_questions=n_sel_q, out_dir=cell_root / f"slot_{step}")
                grid_reads[step] = r["delta_logp_mean"]
            in_band = {
                s: v
                for s, v in grid_reads.items()
                if C.MARKER_GLOBAL_BAND[0] <= v <= C.MARKER_GLOBAL_BAND[1]
            }
            pool = in_band or grid_reads  # closest-approach fallback, reported
            key = (
                (lambda s: abs(pool[s] - target))
                if target is not None
                else (lambda s: abs(pool[s] - 8.0))
            )
            step = min(sorted(pool), key=key)
            rec = {
                "grid_delta_g": {str(k): v for k, v in sorted(grid_reads.items())},
                "selected_step": step,
                "selected_delta_g": grid_reads[step],
                "in_band": step in in_band,
                "target_m1_delta_g": target,
            }
            _atomic_json(
                cell_root / "selection.json",
                {
                    "step": step,
                    "rate": None,
                    "in_band": step in in_band,
                    "fallback": None if step in in_band else "closest_approach",
                },
            )
            _atomic_json(res, rec)
            out["m2_fullft_band8"] = rec
    deliver = REPO_ROOT / "eval_results" / "issue_1112" / "marker"
    if cfg.smoke:
        deliver = cfg.out_root / "eval_results_mirror" / "marker"
    deliver.mkdir(parents=True, exist_ok=True)
    for cell, rec in out.items():
        _atomic_json(deliver / f"{cell}_slotstats.json", rec)
    # m2 selected-checkpoint persist (plan §10 names s3/s4/m2; round-2
    # Critical 3): runs on fresh AND resumed paths, own done-file.
    if "m2_fullft_band8" in cells:
        _persist_marker_ft(cfg)
    return out


def _persist_marker_ft(cfg: Cfg) -> dict:
    """Upload m2's SELECTED full-FT checkpoint to the overflow repo, THEN reap
    the non-selected rungs (upload-before-delete invariant; plan §9 marker
    cleanup). Idempotent via its own done-file; no-op under ``--no-upload``
    (never delete unuploaded weights)."""
    cell = "m2_fullft_band8"
    cell_root = cfg.out_root / cell
    done_path = cell_root / "persist_ft.json"
    if done_path.exists():
        return _read_json(done_path)
    if not cfg.upload:
        logger.warning("[m2-persist] upload disabled — keeping ALL rungs on disk")
        return {"skipped": "no-upload"}
    step = int(_read_json(cell_root / "selection.json")["step"])
    ckpts = _enumerate_rungs(_read_json(cell_root / "build_result.json")["adapter_root"])
    sel_dir = ckpts[step]
    url = hub._upload(
        sel_dir,
        C.OVERFLOW_REPO,
        "model",
        f"issue1112/{cell}/checkpoint-{step}",
        private=True,
    )
    if not str(url):
        raise RuntimeError(f"selected marker FT checkpoint upload returned no path ({cell})")
    # ONLY after the selected rung is durably uploaded: reap the others (the
    # selected rung stays on disk — p10 capture reads it).
    for s, p in ckpts.items():
        if s != step:
            shutil.rmtree(p, ignore_errors=True)
    rec = {
        "uploaded": f"issue1112/{cell}/checkpoint-{step}",
        "kept": [step],
        "cleaned": sorted(set(ckpts) - {step}),
    }
    _atomic_json(done_path, rec)
    return rec


def _weights_complete(model_dir: Path) -> bool:
    """True iff ``model_dir`` carries a complete safetensors weight set.

    Sharded dirs are checked shard-by-shard against ``weight_map`` in
    ``model.safetensors.index.json`` (a kill mid-``save_pretrained`` leaves
    config.json + a shard subset); single-file dirs need ``model.safetensors``.
    """
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        try:
            shards = set(json.loads(idx.read_text())["weight_map"].values())
        except (KeyError, ValueError):
            return False
        return bool(shards) and all((model_dir / s).exists() for s in shards)
    return (model_dir / "model.safetensors").exists()


def _ensure_dir_tokenizer(model_dir: Path, base_model: str = DEFAULT_BASE_MODEL) -> bool:
    """Repair a tokenizer-less LOCAL model dir: save the base tokenizer into it.

    Nothing in this pipeline trains the tokenizer (the marker is an EXISTING
    vocab id, 83399), so the base tokenizer is exact for every merged / FT dir.
    Known tokenizer-less producers (crash-fix r6): a partially-written merged
    dir surviving the old config.json early-return in ``_merge_adapter``, and
    the m2 marker-FT rung checkpoints (HF Trainer without ``processing_class``
    writes no tokenizer files into ``checkpoint-<step>/``). Without the repair,
    ``AutoTokenizer.from_pretrained(dir)`` falls back to the SLOW Qwen2 class
    and dies on ``vocab_file=None`` (TypeError) — the attempt-3/-4 crash.
    Idempotent (keyed on ``tokenizer.json``); returns True when it repaired.
    """
    from transformers import AutoTokenizer

    if (model_dir / "tokenizer.json").exists():
        return False
    tok = AutoTokenizer.from_pretrained(base_model, token=os.environ.get("HF_TOKEN"))
    tok.save_pretrained(str(model_dir))
    logger.info("[tokenizer-repair] wrote base tokenizer into %s", model_dir)
    return True


def _merge_adapter(cfg: Cfg, adapter_dir: str, merged_dir: Path) -> Path:
    """Atomic merge-for-read (#653 pattern; caller deletes after its pass).

    Crash-fix r6: the old bare ``config.json`` early-return treated a
    PARTIALLY-written merged dir as complete — a crash inside this function
    (between ``model.save_pretrained`` and the tokenizer save) escapes the
    CALLER's try/finally rmtree, so the partial dir survived to the next
    attempt, which early-returned it and crashed at the tokenizer load. Now:
    a complete dir returns; a weights-complete / tokenizer-less dir is
    repaired in place (cheap file writes, no re-merge); a weights-incomplete
    dir is wiped and re-merged; and fresh merges write to a ``.tmp`` sibling
    then rename, so ``merged_dir`` existing implies it is complete.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (merged_dir / "config.json").exists():
        if _weights_complete(merged_dir):
            _ensure_dir_tokenizer(merged_dir)
            return merged_dir
        logger.warning("[merge] incomplete merged dir at %s — wiping + re-merging", merged_dir)
        shutil.rmtree(merged_dir, ignore_errors=True)
    tmp_dir = merged_dir.parent / (merged_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)  # stale partial from a prior crash
    base = AutoModelForCausalLM.from_pretrained(
        DEFAULT_BASE_MODEL, torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN")
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(tmp_dir))
    AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL).save_pretrained(str(tmp_dir))
    tmp_dir.rename(merged_dir)  # atomic publish: dir present => complete
    del model, base
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return merged_dir


# ── p9: r_B extraction ───────────────────────────────────────────────────────


def _seed_rb_artifacts_from_registry(cache_path: Path) -> dict:
    """Pre-seed the issue779 extractor's per-trait artifacts cache from the
    BEHAVIORS registry — the #1090 c3 sycophancy definition (plan §4.5: the c3
    trait description + 5 contrastive instruction pairs + the extraction
    question set DISJOINT from the 20-q eval bank). Without this the extractor
    would Sonnet-generate artifacts from ITS OWN issue779 trait description (a
    different definition) on the fresh instance (`data/` is gitignored)."""
    b = BEHAVIORS[C.SYCO_BEHAVIOR]
    ex = b.extraction
    assert ex is not None and b.judge_rubric, "sycophancy registry entry is a stub"
    assert "{question}" in b.judge_rubric and "{answer}" in b.judge_rubric
    ext_qs = list(ex.question_set)
    overlap = set(ext_qs) & set(b.eval_question_bank)
    assert not overlap, f"extraction/eval question overlap: {sorted(overlap)[:3]}"
    artifacts = {
        "instruction": [{"pos": p.exhibit, "neg": p.not_exhibit} for p in ex.prompt_pairs],
        "extraction_questions": ext_qs,
        "eval_prompt": b.judge_rubric,
        "provenance": {
            "source": "artifacts.behavior.BEHAVIORS['sycophancy'] (the #1090 c3 definition)",
            "seeded_by": "issue1112_dispatch.phase_rb",
            "n_pairs": len(ex.prompt_pairs),
            "n_extraction_questions": len(ext_qs),
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(cache_path, artifacts)
    return artifacts


def phase_rb(cfg: Cfg) -> dict:
    _phase("p9_rb")
    rb_dir = cfg.out_root / "rb"
    done = rb_dir / "rb_done.json"
    if done.exists():
        return _read_json(done)
    rec: dict = {}
    if any(c.startswith("s") for c in cfg.cells) and not cfg.smoke:
        # The extractor reads its artifacts cache at data/issue_779/artifacts/
        # (its designed injection point); seed it with the #1090 c3 definition.
        _seed_rb_artifacts_from_registry(
            REPO_ROOT / "data" / "issue_779" / "artifacts" / "sycophancy.json"
        )
        cmd = [
            "uv",
            "run",
            "python",
            RB_EXTRACTOR,
            "--traits",
            "sycophancy",
            "--out-dir",
            str(rb_dir),
            "--no-upload",
        ]
        _run_subprocess(cmd, rb_dir / "rb.log")
        # Normalize the extractor's output (rb/r_b/sycophancy.pt, key "r_b")
        # to the shape phase_upload + geometry consume (rb_sycophancy.pt,
        # key "rb"; (28, 3584)). Fail-loud on a missing/mis-shaped tensor.
        import torch

        src = rb_dir / "r_b" / "sycophancy.pt"
        if not src.exists():
            raise FileNotFoundError(f"r_B extractor produced no tensor at {src}")
        obj = torch.load(src, map_location="cpu", weights_only=False)
        r_b = obj["r_b"]
        assert tuple(r_b.shape) == (C.N_LAYERS, C.HIDDEN), r_b.shape
        torch.save(
            {"rb": r_b, "counts": obj.get("counts"), "source": str(src)},
            rb_dir / "rb_sycophancy.pt",
        )
        rec["sycophancy"] = str(rb_dir / "rb_sycophancy.pt")
        # Upload-verification v1 blocker `generation-discarded-undeclared`
        # (round 8): the extractor persists the rollout TEXT under
        # rb/raw_completions/ BEFORE its judge + stream-reduce (plan §10
        # raw_completions/rb_extraction — rollout text is never discardable);
        # fail loud here if that contract ever regresses, so the tensor can
        # never again ship without the text it was reduced from.
        rollout_files = sorted((rb_dir / "raw_completions").glob("rollouts_sycophancy_*.json"))
        if not rollout_files:
            raise FileNotFoundError(
                f"r_B extractor persisted no rollout text under {rb_dir / 'raw_completions'} "
                "(expected rollouts_sycophancy_{pos,neg}[.partNN].json)"
            )
        rec["rollout_files"] = [p.name for p in rollout_files]
    if any(c.startswith("m") for c in cfg.cells):
        # marker r_B = W_U[83399] per layer-independent unembedding row (#653)
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            token=os.environ.get("HF_TOKEN"),
        )
        wu = model.get_output_embeddings().weight[C.MARKER_TOKEN_ID].detach().to(torch.float32)
        rb = wu.unsqueeze(0).repeat(C.N_LAYERS, 1)
        rb_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"rb": rb, "note": "W_U[83399] tiled per layer (#653 convention)"},
            rb_dir / "rb_marker.pt",
        )
        del model
        rec["marker"] = str(rb_dir / "rb_marker.pt")
    _atomic_json(done, rec)
    return rec


# ── p10: capture (sharded per (cell, dose)) ──────────────────────────────────


def _capture_panel(behavior: str) -> tuple[dict[str, tuple[str | None, str | None]], list[str]]:
    """{context_id: (system_prompt, user_wrap)} + question list for capture."""
    if behavior == C.SYCO_BEHAVIOR:
        src = i1090._source_context()
        # HARD panel ∩ sources == ∅ invariant at the capture build site
        # (#527/#538 class; plan §4.2). The realized source identity is the
        # persona key behind the source context.
        assert_panel_disjoint_from_sources(
            default_panel(),
            [src.context_id],
            source_identities={src.context_id: "software_engineer"},
        )
        panel: dict[str, tuple[str | None, str | None]] = {
            src.context_id: (src.system, getattr(src, "user_wrap", None))
        }
        for neg in default_panel():
            panel[neg.slug] = (neg.system_prompt, neg.user_wrap)
        questions = list(BEHAVIORS[C.SYCO_BEHAVIOR].eval_question_bank)
    else:
        from explore_persona_space.experiments.factor_screen_365.persona_panel import (
            EVAL_PERSONAS_24,
            EVAL_QUESTIONS_20,
        )

        panel = {p: (EVAL_PERSONAS_24[p], None) for p in ("villain", *mixmod.MARKER_NEGATIVES)}
        questions = list(EVAL_QUESTIONS_20)
    return panel, questions


def run_capture_unit(cfg: Cfg, cell: str, dose: str) -> None:
    """One capture pass: on-policy gen + 28-layer 3-span TF pooling -> pooled.pt."""
    import torch
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import (
        _generate_responses_vllm,
        _teacher_forced_span_means,
        compute_prompt_spans,
    )

    out_dir = cfg.out_root / "capture" / cell / dose
    if (out_dir / "pooled.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior = "marker" if cell.startswith(("m", "base_marker")) else C.SYCO_BEHAVIOR
    model_path, cleanup_merged = _resolve_capture_model(cfg, cell, dose)
    panel, questions = _capture_panel(behavior)
    max_new = C.MARKER_MAX_NEW_TOKENS if behavior == "marker" else C.SYCO_MAX_NEW_TOKENS
    if cfg.smoke:
        # 2 contexts × 2 questions = 4 rows: the PREFIX arm has one unique row
        # per context (plan §4.5 degeneracy framing), so a 1-context smoke
        # capture makes the prefix Δx cloud rank-0 after centering and p11
        # geometry can only exercise the degenerate branch (crash-fix r3,
        # att-20260707-205546). Two contexts give the prefix arm ≥2 unique
        # rows — the smoke proves the NONDEGENERATE production spectral path.
        panel = dict(list(panel.items())[:2])
        questions = questions[:2]
    personas = {}
    user_texts = {}
    for ctx_id, (system, wrap) in panel.items():
        personas[ctx_id] = system
        user_texts[ctx_id] = wrap
    # user_wraps threads each member's OWN user-turn rendering into generation
    # so generation + span computation share ONE message construction (round-2
    # Critical 1: the wrap member previously generated on the BARE question).
    rows = _generate_responses_vllm(
        model_path,
        {k: personas[k] for k in panel},
        questions,
        max_new_tokens=max_new,
        gpu_memory_utilization=0.6,
        user_wraps=user_texts,
    )
    # NOTE: user_wrap members need the WRAPPED question for span computation
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    for r in rows:
        ctx_id = r["persona"]
        q = questions[r["question_idx"]]
        wrap = user_texts.get(ctx_id)
        user_content = wrap.format(q=q) if wrap else q
        r["prefix_len"], r["context_len"] = compute_prompt_spans(
            tokenizer, personas[ctx_id], user_content, r["prompt_token_ids"]
        )
    # persist rollout text BEFORE the capture reduce (upload policy #779)
    (out_dir / "raw_rows.json").write_text(
        json.dumps({"model": model_path, "rows": rows}, ensure_ascii=False)
    )
    pooled = _teacher_forced_span_means(
        model_path,
        rows,
        list(panel),
        layers=list(range(C.N_LAYERS)),
        device="cuda:0",
        dtype=torch.bfloat16,
        tf_batch_size=C.TF_BATCH_SIZE,
    )
    store = {
        "schema_version": 1,
        "cell": cell,
        "dose": dose,
        "behavior": behavior if behavior == "marker" else "sycophancy",
        "model_path": model_path,
        "row_meta": [{"context_id": r["persona"], "question_idx": r["question_idx"]} for r in rows],
        "arms": {
            arm: {li: t.to(torch.float16) for li, t in per_layer.items()}
            for arm, per_layer in pooled.items()
        },
        "metadata": {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "max_new_tokens": max_new,
            "tf_batch_size": C.TF_BATCH_SIZE,
        },
    }
    tmp = out_dir / "pooled.pt.tmp"
    torch.save(store, tmp)
    os.replace(tmp, out_dir / "pooled.pt")
    if cleanup_merged is not None:
        shutil.rmtree(cleanup_merged, ignore_errors=True)


def _resolve_capture_model(cfg: Cfg, cell: str, dose: str) -> tuple[str, Path | None]:
    """(model_path, merged_dir_to_cleanup) for one capture pass."""
    if cell.startswith("base_"):
        return DEFAULT_BASE_MODEL, None
    cell_root = cfg.out_root / cell
    if cell == "m1_lora_band8":
        # Crash-fix r7 (attempt-5 FileNotFoundError): m1 has NO rung selection
        # BY DESIGN — the band-stopped FINAL adapter is the cell's artifact
        # ("use final", phase_marker m1 read) — so no phase writes a
        # step-selection for it. Resolve the SAME model identity p8_marker's
        # m1 read consumed: build_result.json["adapter_root"], merged.
        # (phase_marker additionally backfills a step=None selection.json as
        # provenance — _write_m1_selection — but capture never requires it.)
        build = _read_json(cell_root / "build_result.json")
        logger.info(
            "[capture-resolve] m1_lora_band8/%s -> band-stopped final adapter %s "
            "(no rung selection by design)",
            dose,
            build["adapter_root"],
        )
        merged = _merge_adapter(cfg, build["adapter_root"], cell_root / f"merged_{dose}")
        return str(merged), merged
    if cell == C.REUSED_CELL:
        step = {"selected": C.FU2_SELECTED_STEP, "step6": 6, "step30": 30}[dose]
        adapter = cfg.out_root / "inputs" / "fu2" / f"checkpoint-{step}"
        merged = _merge_adapter(cfg, str(adapter), cell_root / f"merged_{dose}")
        return str(merged), merged
    build = _read_json(cell_root / "build_result.json")
    if dose == "selected":
        # selection.json read WHERE it is used (r7): the eager top-of-function
        # read crashed every cell without one (m1) and needlessly required it
        # for the fixed step6/step30 doses + the reused cell.
        step = int(_read_json(cell_root / "selection.json")["step"])
    elif dose in ("step6", "step30"):
        step = int(dose.removeprefix("step"))
    else:
        raise ValueError(dose)
    ckpt = _enumerate_rungs(build["adapter_root"])[step]
    if cell.startswith(("s3", "s4", "s6", "m2")):
        # Full-FT rung dirs may lack tokenizer files (m2's trainer saved
        # checkpoints without processing_class) — repair before the capture
        # reads load tokenizer + engine from this path (crash-fix r6;
        # idempotent no-op for dirs that already carry tokenizer.json).
        _ensure_dir_tokenizer(ckpt)
        return str(ckpt), None  # full-FT consolidated dir loads directly
    merged = _merge_adapter(cfg, str(ckpt), cell_root / f"merged_{dose}")
    return str(merged), merged


def capture_passes(cfg: Cfg) -> list[tuple[str, str]]:
    passes: list[tuple[str, str]] = []
    behaviors = set()
    for cell in cfg.cells:
        if cell in (C.REUSED_CELL, "s2_lora_pos", "s3_fullft_neg", "s4_fullft_pos"):
            doses = ("selected",) if cfg.smoke else C.CAPTURE_DOSES
            passes += [(cell, d) for d in doses]
            behaviors.add("sycophancy")
        elif cell == C.LR_MATCHED_CELL:
            # selected-dose ONLY (plan v8 §12.1) — the lr-matched read pairs
            # one selected-rung cloud against the parent s3 tensors.
            passes.append((cell, "selected"))
            behaviors.add("sycophancy")
        elif cell in C.GENERIC_CELLS:
            passes.append((cell, "selected"))
            behaviors.add("sycophancy")
        elif cell in C.MARKER_CELLS:
            passes.append((cell, "selected"))
            behaviors.add("marker")
        else:
            # fail-loud (plan v8 §12.1 + critic note): an unregistered cell
            # must never silently skip capture — the smoke that "passes" on a
            # silently-dropped cell is the #546-class canary gap.
            raise ValueError(
                f"capture_passes: unroutable cell {cell!r} — register it in the "
                "capture membership tables before dispatch"
            )
    if "sycophancy" in behaviors:
        passes.append(("base_sycophancy", "base"))
    if "marker" in behaviors:
        passes.append(("base_marker", "base"))
    return passes


def phase_capture(cfg: Cfg) -> dict:
    _phase("p10_capture")
    passes = [
        (c, d)
        for c, d in capture_passes(cfg)
        if not (cfg.out_root / "capture" / c / d / "pooled.pt").exists()
    ]
    if not passes:
        return {"n_passes": 0}
    if _n_gpus() == 1 or len(passes) == 1:
        for c, d in passes:
            run_capture_unit(cfg, c, d)
    else:
        units = [
            [
                "--unit",
                "capture",
                f"{c}/{d}",
                "--smoke" if cfg.smoke else "--full",
                "--out-root",
                str(cfg.out_root),
                "--cells",
                ",".join(cfg.cells),
            ]
            + (
                ["--eval-question-limit", str(cfg.eval_question_limit)]
                if cfg.eval_question_limit
                else []
            )
            + ([] if cfg.upload else ["--no-upload"])
            for c, d in passes
        ]
        _fanout_units(cfg, units)
    return {"n_passes": len(passes)}


# ── p10b: teacher-forced SHARED-response capture (amendment v6) ──────────────
# Follow-up round `tf-shared-response-capture`: re-measure each trained cell's
# response-arm shift over ONE FIXED response per row — the persisted BASE-model
# capture generation — teacher-forced through the trained checkpoint. The base
# side of Δx_shared is the parent's existing base pooled store, so there is NO
# base re-capture and NO generation stage this round (plan v6 §4).

TF_FOLLOWUP_LABEL = "tf-shared-response-capture"
# behavior -> (path_in_repo on HF_DATA_REPO, pinned revision). Verified
# 2026-07-08: 120 rows, complete 6x20 grid, TF-contract fields, 0 empty
# responses (plan §4.6 "Conditioning source").
TF_BASE_ROWS = {
    "sycophancy": (
        f"{C.DATA_PREFIX}/raw_completions/capture/base_sycophancy/base/raw_rows.json",
        "e016910195b7ab846c83b87ec43140c36c51e35f",
    ),
    # OPTIONAL --tf-marker extension (plan §4: zero-marginal-provision, off by
    # default); gated on a file_exists probe at the same pinned revision.
    "marker": (
        f"{C.DATA_PREFIX}/raw_completions/capture/base_marker/base/raw_rows.json",
        "e016910195b7ab846c83b87ec43140c36c51e35f",
    ),
}
TF_OVERFLOW_REV = "90949b061d09b30d5850f2fec0043790939aa322"
# cell -> (overflow prefix, pinned revision, load kind) — the plan §4 table;
# selected steps read from the parent's committed selection records
# (issue1112_geometry2x2/selection/<cell>/selection.json @ e0169101…:
# s1->14 s2->10 s3->8 s4->6 s5->10 s6->8, matching the body Training table).
TF_CKPTS: dict[str, tuple[str, str, str]] = {
    C.REUSED_CELL: (
        f"{C.FU2_CKPT_PREFIX}/checkpoint-{C.FU2_SELECTED_STEP}",
        C.FU2_CKPT_REV,
        "lora",
    ),
    "s2_lora_pos": ("issue1112/s2_lora_pos/checkpoint-10", TF_OVERFLOW_REV, "lora"),
    "s3_fullft_neg": ("issue1112/s3_fullft_neg/checkpoint-8", TF_OVERFLOW_REV, "full"),
    "s4_fullft_pos": ("issue1112/s4_fullft_pos/checkpoint-6", TF_OVERFLOW_REV, "full"),
    "s5_lora_generic": ("issue1112/s5_lora_generic/checkpoint-10", TF_OVERFLOW_REV, "lora"),
    "s6_fullft_generic": ("issue1112/s6_fullft_generic/checkpoint-8", TF_OVERFLOW_REV, "full"),
}
TF_MARKER_CKPTS: dict[str, tuple[str, str, str]] = {
    # m1's artifact is the band-stopped FINAL adapter at the ladder ROOT (no
    # rung selection by design — _resolve_capture_model's m1 branch); m2's is
    # the p8-selected checkpoint-4 persisted by _persist_marker_ft.
    "m1_lora_band8": ("issue1112/m1_lora_band8", TF_OVERFLOW_REV, "lora_root"),
    "m2_fullft_band8": ("issue1112/m2_fullft_band8/checkpoint-4", TF_OVERFLOW_REV, "full"),
}
TF_ROW_FIELDS = (
    "persona",
    "question_idx",
    "prompt_token_ids",
    "response_token_ids",
    "prefix_len",
    "context_len",
)


def assert_tf_base_rows(
    rows: list[dict], *, expect_contexts: int | None, expect_questions: int
) -> None:
    """Fail-fast contract asserts on the staged conditioning rows (plan §4.6):
    every row carries the `_teacher_forced_span_means` fields with valid span
    bounds + a non-empty response, and the (persona x question_idx) grid is
    COMPLETE (120 = 6x20 for the sycophancy panel)."""
    assert rows, "conditioning rows empty"
    for i, r in enumerate(rows):
        missing = [k for k in TF_ROW_FIELDS if k not in r]
        assert not missing, (i, missing)
        assert len(r["response_token_ids"]) > 0, f"row {i} has an empty response"
        assert 0 < r["prefix_len"] < r["context_len"] <= len(r["prompt_token_ids"]), (
            i,
            r["prefix_len"],
            r["context_len"],
            len(r["prompt_token_ids"]),
        )
    personas = sorted({r["persona"] for r in rows})
    qs = sorted({int(r["question_idx"]) for r in rows})
    grid = {(r["persona"], int(r["question_idx"])) for r in rows}
    if expect_contexts is not None:
        assert len(personas) == expect_contexts, (len(personas), expect_contexts, personas)
    assert qs == list(range(expect_questions)), qs
    assert len(grid) == len(rows) == len(personas) * len(qs), (
        len(grid),
        len(rows),
        len(personas),
        len(qs),
    )


def _stage_tf_base_rows(cfg: Cfg, behavior: str) -> list[dict]:
    """Stage + validate the shared conditioning rows at the pinned revision."""
    path_in_repo, rev = TF_BASE_ROWS[behavior]
    dest = cfg.out_root / "inputs" / f"tf_base_rows_{behavior}.json"
    _stage_file(path_in_repo, dest, revision=rev)
    rows = _read_json(dest)["rows"]
    if behavior == "sycophancy":
        assert_tf_base_rows(rows, expect_contexts=6, expect_questions=20)
        assert len(rows) == 120, len(rows)
    else:
        assert_tf_base_rows(rows, expect_contexts=None, expect_questions=20)
    return rows


def tf_smoke_rows(rows: list[dict]) -> list[dict]:
    """2 contexts x 2 questions = 4-row subset (the plan §4 smoke shape; >=2
    contexts keep the prefix arm nondegenerate, the p10 smoke framing)."""
    personas = sorted({r["persona"] for r in rows})[:2]
    sub = [r for r in rows if r["persona"] in personas and int(r["question_idx"]) in (0, 1)]
    assert len(sub) == 4, len(sub)
    return sub


def _stage_tf_ckpt(cfg: Cfg, cell: str, prefix: str, rev: str, kind: str) -> Path:
    """Stage one pinned checkpoint with a completeness guard (a crash between
    config.json and the last shard would otherwise early-return a partial dir
    — the r6 class); wipe + re-stage ONCE on incompleteness, then fail loud."""
    dest = cfg.out_root / "inputs" / "tf_ckpts" / cell
    recursive = kind != "lora_root"

    def _complete(d: Path) -> bool:
        if kind == "full":
            return _weights_complete(d)
        return (d / "adapter_config.json").exists() and (d / "adapter_model.safetensors").exists()

    staged = _stage_overflow_prefix(prefix, dest, revision=rev, recursive=recursive)
    if not _complete(staged):
        logger.warning(
            "[tf-stage] incomplete staged checkpoint at %s — wiping + re-staging", staged
        )
        shutil.rmtree(staged, ignore_errors=True)
        staged = _stage_overflow_prefix(prefix, dest, revision=rev, recursive=recursive)
        if not _complete(staged):
            raise RuntimeError(f"staged checkpoint incomplete after re-stage: {staged}")
    return staged


def _resolve_tf_capture_model(cfg: Cfg, cell: str) -> tuple[str, list[Path], dict]:
    """(model_path, cleanup_dirs, provenance) for one shared-text capture pass.

    Stages the cell's PINNED checkpoint from the overflow repo (plan §4 table),
    merges LoRA cells via the existing atomic merge-read-delete helper (incl.
    the r6 tokenizer repair inside `_merge_adapter`), and returns every dir the
    caller deletes after its pass (cleanup-as-you-go, plan §9 disk budget:
    FT checkpoints + merged dirs never coexist across passes).
    """
    table = {**TF_CKPTS, **TF_MARKER_CKPTS}
    prefix, rev, kind = table[cell]
    staged = _stage_tf_ckpt(cfg, cell, prefix, rev, kind)
    prov = {"repo": C.OVERFLOW_REPO, "prefix": prefix, "revision": rev, "kind": kind}
    if kind == "full":
        # Full-FT rung dirs may lack tokenizer files (Trainer saved without
        # processing_class — crash-fix r6); repair before the capture load.
        _ensure_dir_tokenizer(staged)
        return str(staged), [staged], prov
    merged = _merge_adapter(cfg, str(staged), cfg.out_root / "capture_tf" / cell / "merged_tf")
    return str(merged), [merged], prov


def run_capture_tf_unit(
    cfg: Cfg,
    cell: str,
    rows: list[dict],
    *,
    resolve_fn=None,
    layers: list[int] | None = None,
    device: str | None = None,
) -> dict:
    """One shared-text capture pass -> capture_tf/<cell>/selected/pooled.pt.

    Feeds the persisted BASE-model rows (fixed text) through the cell's trained
    checkpoint via the UNCHANGED `_teacher_forced_span_means`, parameters
    byte-identical to the parent §4.4 capture (28 layers, bf16,
    tf_batch_size 8). Idempotent on pooled.pt (spot-tolerant). ``resolve_fn``/
    ``layers``/``device`` are compute-scale seams for the tiny-real CPU smoke
    ONLY — production leaves all three at their defaults, and the default
    device REQUIRES CUDA (fail-loud; a silent CPU fallback would be a silent
    ~100x slowdown on the pod).
    """
    import torch

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    out_dir = cfg.out_root / "capture_tf" / cell / "selected"
    pooled_path = out_dir / "pooled.pt"
    if pooled_path.exists():
        logger.info("[tf-capture] %s already captured — skipping (resume)", cell)
        return {"cell": cell, "pooled": str(pooled_path), "skipped": "pooled.pt exists"}
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("tf-shared capture needs a CUDA device (plan §9: 1x A100-80)")
        device = "cuda:0"
    behavior = "marker" if cell in C.MARKER_CELLS else C.SYCO_BEHAVIOR
    resolve = resolve_fn if resolve_fn is not None else _resolve_tf_capture_model
    model_path, cleanup, prov = resolve(cfg, cell)
    panel = list(dict.fromkeys(r["persona"] for r in rows))
    layer_list = list(range(C.N_LAYERS)) if layers is None else list(layers)
    try:
        pooled = _teacher_forced_span_means(
            model_path,
            rows,
            panel,
            layers=layer_list,
            device=device,
            dtype=torch.bfloat16,
            tf_batch_size=C.TF_BATCH_SIZE,
        )
        store = {
            "schema_version": 1,
            "cell": cell,
            "dose": "selected",
            "behavior": behavior,
            "model_path": model_path,
            "row_meta": [
                {"context_id": r["persona"], "question_idx": int(r["question_idx"])} for r in rows
            ],
            "arms": {
                arm: {li: t.to(torch.float16) for li, t in per_layer.items()}
                for arm, per_layer in pooled.items()
            },
            "metadata": {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tf_batch_size": C.TF_BATCH_SIZE,
                # The two plan-§4 additions vs the parent run_capture_unit store:
                "conditioning": "tf_shared_base",
                "conditioning_rows": {
                    "repo": C.HF_DATA_REPO,
                    "path": TF_BASE_ROWS[behavior][0],
                    "revision": TF_BASE_ROWS[behavior][1],
                    "n_rows": len(rows),
                },
                "checkpoint": prov,
                "followup_label": TF_FOLLOWUP_LABEL,
                "git_commit": _git_commit_sha(),
            },
        }
        tmp = out_dir / "pooled.pt.tmp"
        torch.save(store, tmp)
        os.replace(tmp, pooled_path)
    finally:
        # cleanup-as-you-go (plan §9): staged FT checkpoint / merged dir are
        # transient; a retry re-stages idempotently from the pinned revision.
        for d in cleanup:
            shutil.rmtree(d, ignore_errors=True)
    return {"cell": cell, "pooled": str(pooled_path), "n_rows": len(rows), "checkpoint": prov}


def _git_commit_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def phase_capture_tf(cfg: Cfg) -> dict:
    """p10b: sequential shared-text passes (single-GPU pod, plan §9 — each pass
    is one 7B model resident; batching lives INSIDE `_teacher_forced_span_means`
    at tf_batch_size=8). Runs standalone: stages its own conditioning rows +
    checkpoints, no training-phase preconditions. Manifest checkpointed per
    cell (checkpoint-per-phase)."""
    _phase("p10b_capture_tf")
    syco_cells = [c for c in cfg.cells if c in TF_CKPTS]
    marker_cells = [c for c in cfg.cells if c in TF_MARKER_CKPTS] if cfg.tf_marker else []
    if not syco_cells and not marker_cells:
        return {"skipped": "no tf-capture cells in --cells"}
    manifest_path = cfg.out_root / "capture_tf_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
    else:
        table = {**TF_CKPTS, **TF_MARKER_CKPTS}
        manifest = {
            "followup_label": TF_FOLLOWUP_LABEL,
            "conditioning": "tf_shared_base",
            "smoke": cfg.smoke,
            "conditioning_rows": {
                b: {"repo": C.HF_DATA_REPO, "path": p, "revision": r}
                for b, (p, r) in TF_BASE_ROWS.items()
            },
            "checkpoints": {
                c: {"repo": C.OVERFLOW_REPO, "prefix": p, "revision": r, "kind": k}
                for c, (p, r, k) in table.items()
            },
            "git_commit": _git_commit_sha(),
            "cells": {},
        }
    records: dict = manifest.setdefault("cells", {})
    for behavior, cells in ((C.SYCO_BEHAVIOR, syco_cells), ("marker", marker_cells)):
        if not cells:
            continue
        if behavior == "marker":
            from huggingface_hub import HfApi

            path_in_repo, rev = TF_BASE_ROWS["marker"]
            if not HfApi().file_exists(
                C.HF_DATA_REPO, path_in_repo, repo_type="dataset", revision=rev
            ):
                logger.warning(
                    "[tf-capture] marker base rows missing at pinned rev — "
                    "skipping the optional extension"
                )
                records["_marker_extension"] = {"skipped": "base_marker rows missing at pinned rev"}
                continue
        rows = _stage_tf_base_rows(cfg, behavior)
        if cfg.smoke:
            rows = tf_smoke_rows(rows)
        for cell in cells:
            records[cell] = run_capture_tf_unit(cfg, cell, rows)
            manifest["cells"] = records
            _atomic_json(manifest_path, manifest)  # per-cell checkpoint
    _atomic_json(manifest_path, manifest)
    done = [c for c in records if not str(c).startswith("_")]
    return {"n_cells": len(done), "cells": sorted(done), "manifest": str(manifest_path)}


# ── p11: geometry (smoke stub — the full pass runs VM-side) ──────────────────


def _single_context_subset(store: dict) -> dict:
    """Row-subset of a capture store keeping only its FIRST context (real data).

    Reproduces the p11 crash shape (att-20260707-205546): a single-context
    capture collapses the prefix arm to 1 unique row, which must now yield
    EXPLICIT degenerate records instead of the #653 fail-fast raise.
    """
    ctx = store["row_meta"][0]["context_id"]
    keep = [i for i, m in enumerate(store["row_meta"]) if m["context_id"] == ctx]
    assert keep, store["row_meta"]
    sub = dict(store)
    sub["row_meta"] = [store["row_meta"][i] for i in keep]
    sub["arms"] = {
        arm: {li: t[keep] for li, t in per_layer.items()}
        for arm, per_layer in store["arms"].items()
    }
    return sub


def phase_geometry_smoke(cfg: Cfg) -> dict:
    _phase("p11_geometry")
    if not cfg.smoke:
        return {"skipped": "full geometry runs VM-side (scripts/issue1112_geometry.py)"}
    import torch

    from explore_persona_space.experiments.issue_1112 import geometry as geo

    cell = cfg.cells[0]
    rb = torch.randn(C.N_LAYERS, C.HIDDEN)  # smoke stub direction (labeled)
    rb_path = cfg.out_root / "rb" / "rb_smoke.pt"
    rb_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rb": rb, "smoke_stub": True}, rb_path)
    payload = geo.run_geometry(
        cfg.out_root / "capture",
        cfg.out_root / "geometry_smoke",
        cells_doses=[(cell, "selected")],
        base_store_by_behavior={
            "sycophancy": cfg.out_root / "capture" / "base_sycophancy" / "base" / "pooled.pt"
        },
        behavior_by_cell={cell: "sycophancy"},
        selected_dose_by_cell={cell: "selected"},
        rb_by_behavior={"sycophancy": rb_path},
        n_boot=25,
    )
    prefix_recs = [r for r in payload["records"].values() if r["arm"] == "prefix"]
    n_prefix_nondegen = sum(1 for r in prefix_recs if not r.get("degenerate"))
    if n_prefix_nondegen < 1:
        raise RuntimeError(
            "smoke geometry produced no nondegenerate prefix-arm record — the smoke "
            "capture must span >=2 contexts to exercise the production spectral path"
        )

    # ── Degenerate-branch canary: single-context row subset of the SAME real
    # captured stores. The prefix arm then has exactly 1 unique row, so the
    # explicit degenerate-record branch (geometry.analyze_cell) must engage.
    probe_root = cfg.out_root / "capture_degenerate_probe"
    for src_cell, dose in ((cell, "selected"), ("base_sycophancy", "base")):
        store = geo.load_store(cfg.out_root / "capture" / src_cell / dose / "pooled.pt")
        out = probe_root / src_cell / dose
        out.mkdir(parents=True, exist_ok=True)
        torch.save(_single_context_subset(store), out / "pooled.pt")
    degen_payload = geo.run_geometry(
        probe_root,
        cfg.out_root / "geometry_smoke_degenerate",
        cells_doses=[(cell, "selected")],
        base_store_by_behavior={
            "sycophancy": probe_root / "base_sycophancy" / "base" / "pooled.pt"
        },
        behavior_by_cell={cell: "sycophancy"},
        selected_dose_by_cell={cell: "selected"},
        rb_by_behavior={"sycophancy": rb_path},
        n_boot=25,
    )
    degen_recs = [r for r in degen_payload["records"].values() if r.get("degenerate")]
    if not degen_recs:
        raise RuntimeError("single-context probe emitted no explicit degenerate record")
    non_prefix_degen = [r for r in degen_recs if r["arm"] != "prefix"]
    if non_prefix_degen:
        raise RuntimeError(f"unexpected degenerate non-prefix records: {non_prefix_degen[:2]}")
    return {
        "n_records": len(payload["records"]),
        "n_prefix_nondegenerate": n_prefix_nondegen,
        "degenerate_probe": {
            "n_records": len(degen_payload["records"]),
            "n_degenerate": len(degen_recs),
        },
    }


# ── p12: upload + sentinel ───────────────────────────────────────────────────


def _upload_marker_slot_text(cell_root: Path, cell: str, _up) -> None:
    """Marker-stage rollout text: per-rung ``slot_<step>/`` probe rows + slot
    reads (m2 grid) and m1's ``slot/`` dir — plan §10 raw_completions/marker
    (round-2 Major 6: model generations are never discardable)."""
    for slot_dir in sorted(cell_root.glob("slot*")):
        for f in sorted(slot_dir.glob("*.json")):
            _up(
                f,
                f"{C.DATA_PREFIX}/raw_completions/marker/{cell}/{slot_dir.name}/{f.name}",
                upload_as_file=True,
            )


def _upload_adapter_overflow(cell: str, cell_root: Path, uploaded: dict[str, str]) -> None:
    """LoRA adapter ladders -> overflow repo (FT selected rungs already at p4).

    Existing LoRA cells ship their WHOLE ladder root (parent behavior,
    unchanged). The lr-matched cell ships ONLY its selected rung (plan v8
    §10) — the 29 non-selected rungs are the round's declared discard
    (deterministic retrain recipe; per-rung rates persist via ladder.json +
    selection.json). Exact-match routing on cell ids throughout.
    """
    build = cell_root / "build_result.json"
    if cell in ("s2_lora_pos", "s5_lora_generic", "m1_lora_band8"):
        if build.exists():
            url = hub._upload(
                Path(_read_json(build)["adapter_root"]),
                C.OVERFLOW_REPO,
                "model",
                f"issue1112/{cell}",
                private=True,
            )
            uploaded[f"overflow:issue1112/{cell}"] = str(url)
    elif cell == C.LR_MATCHED_CELL:
        sel = cell_root / "selection.json"
        if build.exists() and sel.exists():
            step = int(_read_json(sel)["step"])
            sel_dir = _enumerate_rungs(_read_json(build)["adapter_root"])[step]
            repo_path = f"issue1112/{cell}/checkpoint-{step}"
            url = hub._upload(sel_dir, C.OVERFLOW_REPO, "model", repo_path, private=True)
            uploaded[f"overflow:{repo_path}"] = str(url)


def phase_upload(cfg: Cfg) -> dict:
    _phase("p12_upload")
    uploaded: dict[str, str] = {}
    if not cfg.upload:
        return uploaded

    def _up(local: Path, path_in_repo: str, **kw) -> None:
        if not Path(local).exists():
            return
        url = hub._upload(local, C.HF_DATA_REPO, "dataset", path_in_repo, **kw)
        if not str(url):
            raise RuntimeError(f"upload returned no path for {path_in_repo}")
        uploaded[path_in_repo] = str(url)
        _atomic_json(cfg.out_root / "upload_manifest.json", uploaded)

    for cell in cfg.cells:
        cell_root = cfg.out_root / cell
        for name in (
            "build_result.json",
            "ladder.json",
            "selection.json",
            "parity.json",
            "marker_read.json",
        ):
            _up(cell_root / name, f"{C.DATA_PREFIX}/selection/{cell}/{name}", upload_as_file=True)
        _up(cell_root / "rate", f"{C.DATA_PREFIX}/raw_completions/rate/{cell}")
        if cell in C.MARKER_CELLS:
            _upload_marker_slot_text(cell_root, cell, _up)
        _upload_adapter_overflow(cell, cell_root, uploaded)
    _up(cfg.out_root / "tier2", f"{C.DATA_PREFIX}/raw_completions/tier2")
    # margin companion DV records (per-cell + shared base; teacher-forced,
    # judge-free JSON — non-LFS path, uploads unconditionally).
    for f in (
        sorted((cfg.out_root / "margin").glob("*.json"))
        if (cfg.out_root / "margin").exists()
        else []
    ):
        _up(f, f"{C.DATA_PREFIX}/margin/{f.name}", upload_as_file=True)
    _up(
        cfg.out_root / "mixes" / "mix_derivation_manifest.json",
        f"{C.DATA_PREFIX}/mixes/mix_derivation_manifest.json",
        upload_as_file=True,
    )
    # capture: rollout text (unconditional) + pooled tensors (analysis_tensors)
    for c, d in capture_passes(cfg):
        cap = cfg.out_root / "capture" / c / d
        _up(
            cap / "raw_rows.json",
            f"{C.DATA_PREFIX}/raw_completions/capture/{c}/{d}/raw_rows.json",
            upload_as_file=True,
        )
        _up(
            cap / "pooled.pt",
            f"{C.DATA_PREFIX}/analysis_tensors/capture/{c}/{d}/pooled.pt",
            upload_as_file=True,
        )
    # tf-shared amendment (plan v6 §4): pooled shared-text tensors + manifest
    # -> analysis_tensors/capture_tf/ (sweep the REALIZED tree, never a
    # registered grid — the smoke cell subset threads through by construction).
    tf_root = cfg.out_root / "capture_tf"
    if tf_root.exists():
        if cfg.smoke:
            logger.warning(
                "[upload] smoke run — capture_tf stores NOT uploaded (4-row smoke "
                "tensors must never land at the production analysis_tensors/"
                "capture_tf paths)"
            )
        else:
            for pooled in sorted(tf_root.glob("*/selected/pooled.pt")):
                cell = pooled.parent.parent.name
                _up(
                    pooled,
                    f"{C.DATA_PREFIX}/analysis_tensors/capture_tf/{cell}/selected/pooled.pt",
                    upload_as_file=True,
                )
            _up(
                cfg.out_root / "capture_tf_manifest.json",
                f"{C.DATA_PREFIX}/analysis_tensors/capture_tf/capture_tf_manifest.json",
                upload_as_file=True,
            )
    _upload_rb_artifacts(cfg.out_root / "rb", _up)
    _up(cfg.out_root / "run_config.json", f"{C.DATA_PREFIX}/run_config.json", upload_as_file=True)
    return uploaded


def _upload_rb_artifacts(rb_dir: Path, _up) -> None:
    """p9_rb artifact routing: tensors -> analysis_tensors/rb, rollout TEXT ->
    raw_completions/rb_extraction (plan §10; upload-verification v1 blocker
    generation-discarded-undeclared, round 8), remaining JSON sidecars (judge
    raw + counts) -> rb/. The rollout dumps are excluded from the generic rb/
    bucket so they land exactly once, at the canonical prefix."""
    for name in ("rb_sycophancy", "rb_marker"):
        _up(
            rb_dir / f"{name}.pt",
            f"{C.DATA_PREFIX}/analysis_tensors/rb/{name}.pt",
            upload_as_file=True,
        )
    rb_rc = rb_dir / "raw_completions"
    for f in sorted(rb_rc.glob("rollouts_*.json")) if rb_rc.exists() else []:
        _up(f, f"{C.DATA_PREFIX}/raw_completions/rb_extraction/{f.name}", upload_as_file=True)
    for extra in sorted(rb_dir.glob("**/*.json")) if rb_dir.exists() else []:
        if "raw_completions" in extra.parts:
            continue  # rollout text uploaded above under raw_completions/rb_extraction/
        _up(extra, f"{C.DATA_PREFIX}/rb/{extra.name}", upload_as_file=True)


def write_sentinel(cfg: Cfg, summary: dict) -> Path:
    _phase("sentinel")
    sentinel_dir = cfg.sentinel_dir or Path("/workspace/logs")
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    kind = "epm:smoke-result" if cfg.smoke else "epm:results"
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,  # VM-side drain re-derives max+1
        "task_id": C.ISSUE,
        "by": "issue1112_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": summary,
    }
    path = sentinel_dir / f"issue-{C.ISSUE}-{kind.replace(':', '_')}-{int(time.time())}.json"
    _atomic_json(path, payload)
    logger.info("[sentinel] wrote %s", path)
    return path


# ── main ─────────────────────────────────────────────────────────────────────


def _check_regime(cfg: Cfg) -> None:
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    p = cfg.out_root / "run_config.json"
    cur = cfg.regime_key()
    if p.exists():
        prior = _read_json(p)
        prior_rest = {k: v for k, v in prior.items() if k != "cells"}
        cur_rest = {k: v for k, v in cur.items() if k != "cells"}
        if prior_rest != cur_rest or not set(cur["cells"]) <= set(prior.get("cells", [])):
            raise RuntimeError(f"out_root {cfg.out_root} holds a run under a DIFFERENT regime")
    else:
        _atomic_json(p, cur)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1112 pod-side phase driver")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny-real, SAME code path")
    mode.add_argument("--full", action="store_true")
    p.add_argument(
        "--unit",
        nargs=2,
        default=None,
        metavar=("KIND", "ARG"),
        help="internal: run one fanout unit (ladder <cell> | capture <cell>/<dose>)",
    )
    p.add_argument(
        "--gpu-id", type=int, default=0, help="physical GPU (CVD-pinned by the launcher)"
    )
    p.add_argument("--cells", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    p.add_argument("--phases", default=None, help="comma subset of phases to run (default all)")
    p.add_argument(
        "--tf-marker",
        action="store_true",
        help="OPTIONAL tf-shared marker extension (m1/m2 shared-text passes; plan v6 §4)",
    )
    return p.parse_args(argv)


def build_cfg(args: argparse.Namespace) -> Cfg:
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else (f"/tmp/issue-{C.ISSUE}-smoke" if smoke else f"data/issue_{C.ISSUE}/run")
    )
    return Cfg(
        smoke=smoke,
        cells=resolve_cells(args.cells, smoke),
        out_root=out_root,
        seed=args.seed,
        tier1_n=2 if smoke else 5,
        tier1_draws=2 if smoke else 3,
        tier2_n=2 if smoke else 10,
        tier2_draws=2 if smoke else 5,
        eval_question_limit=(
            args.eval_question_limit
            if args.eval_question_limit is not None
            else (2 if smoke else None)
        ),
        sentinel_dir=(
            Path(args.sentinel_dir)
            if args.sentinel_dir is not None
            else (out_root / "logs" if smoke else None)
        ),
        upload=args.upload,
        phases=normalize_phases(args.phases),
        tf_marker=bool(getattr(args, "tf_marker", False)),
    )


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 — linear phase chain
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    cfg = build_cfg(args)
    if args.unit is not None:
        kind, arg = args.unit
        if kind == "ladder":
            run_ladder_unit(cfg, arg)
        elif kind == "capture":
            cell, dose = arg.split("/")
            run_capture_unit(cfg, cell, dose)
        else:
            raise ValueError(f"unknown unit kind {kind!r}")
        return 0
    _check_regime(cfg)
    _run_started_ts(cfg)
    logger.info("issue1112 smoke=%s cells=%s out_root=%s", cfg.smoke, cfg.cells, cfg.out_root)

    def want(phase: str) -> bool:
        return not cfg.phases or phase in cfg.phases

    summary: dict = {"issue": C.ISSUE, "smoke": cfg.smoke, "cells": list(cfg.cells)}
    if want("stage"):
        phase_stage(cfg)
    if want("mixes"):
        summary["mixes"] = {
            k: (
                v
                if not isinstance(v, dict)
                else {kk: vv for kk, vv in v.items() if kk != "roles_by_index"}
            )
            for k, v in phase_mixes(cfg).items()
        }
    if want("train"):
        phase_train(cfg)
    selections: dict = {}
    if want("ladder"):
        selections = phase_ladder(cfg)
        summary["selections"] = selections
    if want("g1"):
        g1 = phase_g1_gate(cfg, selections)
        summary["g1"] = g1
        if g1.get("action") == "split_for_second_provision":
            write_sentinel(cfg, summary)
            return 0
        if g1.get("action") == "extend_in_place":
            selections = phase_ladder(cfg)  # re-ladder the extended trees
            summary["selections"] = selections
    if want("persist_ft"):
        summary["persist_ft"] = phase_persist_ft(cfg, selections)
    if want("generic"):
        phase_generic(cfg, selections)
        for cell in C.GENERIC_CELLS:
            sel = cfg.out_root / cell / "selection.json"
            if cell in cfg.cells and sel.exists():
                selections[cell] = _read_json(sel)
    if want("parity"):
        summary["parity"] = phase_parity(cfg)
    if want("tier2"):
        summary["tier2"] = {k: v.get("rates") for k, v in phase_tier2(cfg, selections).items()}
    if want("margin"):
        summary["margin"] = {
            k: {kk: v.get(kk) for kk in ("margin_base", "margin_trained", "margin_delta")}
            for k, v in phase_margin(cfg, selections).items()
            if isinstance(v, dict) and "reads" in v
        }
    if want("marker"):
        summary["marker"] = {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk in ("selected_step", "selected_delta_g", "delta_logp_mean", "in_band")
            }
            for k, v in phase_marker(cfg).items()
            if isinstance(v, dict)
        }
    if want("rb"):
        phase_rb(cfg)
    if want("capture"):
        summary["capture"] = phase_capture(cfg)
    if want("capture_tf"):
        summary["capture_tf"] = phase_capture_tf(cfg)
    if want("geometry"):
        summary["geometry"] = phase_geometry_smoke(cfg)
    if want("upload"):
        uploaded = phase_upload(cfg)
        summary["n_uploaded"] = len(uploaded)
    summary["sentinel"] = str(write_sentinel(cfg, summary))
    logger.info(
        "issue1112 complete: %s",
        json.dumps({k: summary[k] for k in ("smoke", "cells", "n_uploaded") if k in summary}),
    )
    # NOTE: [phase=done] is emitted by scripts/issue1112_dispatch.sh, never here.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
