#!/usr/bin/env python
"""#1090 ``fu2-dose-extension`` follow-up driver (cheap round 2).

Question (followup-scope marker v2 on task #1090): is the preregistered
JUDGED_RATE_BAND (0.60, 0.85) reachable for the two sycophancy organisms at
all — the parent + fu1 dose curves were still rising at the epochs-3 ceiling
(step 14-15, peak 0.549 @ step 10)?

Design (per cell in {c3-sycophancy-claude, c5-sycophancy-qwen}):

1. Stage the cell's EXISTING FROZEN training mix from HF
   (``issue1090_pvdatagen/<cell>/mix/train_mix.jsonl``; NO datagen) and
   sha/row-count verify it against ``mix_meta.json`` (refuse on mismatch).
2. Retrain FROM SCRATCH at ``epochs=6`` — 30 optimizer steps at the 80-row /
   batch-4x4 recipe — THE single disclosed recipe deviation; ``save_steps=2``
   and ``max_length=2048`` are the parent run's own declared deviations,
   applied through the SAME seams (``i1090._make_train_fn`` +
   the ``build_organism`` ``spec.overrides`` pattern), every other recipe
   value verbatim (pinned by ``tests/test_issue1090_fu2.py``).
3. Tier-1 ladder judged rates over ALL rungs (2..30 at 80 rows) via the
   production rate path (``make_source_rate_fn``), with the fu1 judge
   instrument (``max_tokens=300`` — the 64-token default censors
   reason-first rubrics, #1090 lesson). NOTE this differs from the PARENT
   ladder's 64-token instrument; BOTH are recorded in the sentinel /
   build records so the curves are labeled by instrument.
4. Dose-select against ``JUDGED_RATE_BAND`` (``select_dose_checkpoint``:
   earliest in-band rung, else ``closest_approach`` fallback).
5. IF a rung enters the band, Tier-2 generation (n=10 x 20 questions) at the
   selected rung + base via the parent's ``phase_tier2_generation`` —
   completions persisted unconditionally; JUDGING runs VM-side (deferred).
6. Upload: all text/JSON -> ``issue1090_pvdatagen/fu2-dose-extension/...``
   (data repo, non-LFS); adapter ladders -> the PRIVATE OVERFLOW repo under
   ``issue1090/fu2/<cell>/`` DIRECTLY (the canonical model repo is at the
   100k-file limit — followup-scope directive; never attempt canonical).
7. ``epm:results`` sentinel per the pod-side contract (mirrors
   ``fu1.write_fu1_sentinel``; smoke writes kind ``epm:smoke-result``).

Every phase checkpoints + resumes keyed on the fu2 regime (epochs=6 and the
judge max_tokens budget enter the key). ``--smoke`` is the SAME path with
tiny knobs: 1 cell (c3), a fixture mix in the production on-disk shape with
a PINNED sha (the same verify branch runs), ``max_steps=2`` (1 rung), tiny
tier1/tier2 knobs, a recording upload seam; the judge stays LIVE.

``[phase=done]`` is emitted by ``scripts/issue1090_fu2_dispatch.sh`` ONLY.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1074_generator_compare as i1074  # noqa: E402
import issue1090_fu1 as fu1  # noqa: E402
import issue1090_run as i1090  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    ModelOrganism,
    _sha256_file,
    make_source_rate_fn,
    release_trainer_cuda_memory,
)
from explore_persona_space.artifacts.recipe import (  # noqa: E402
    JUDGED_RATE_BAND,
    DoseSelection,
    RecipeSpec,
    build_train_config,
    recipe_for,
    select_dose_checkpoint,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1090_fu2")

# ── Constants ─────────────────────────────────────────────────────────────────

FU2_LABEL = "fu2-dose-extension"
# THE single recipe deviation (disclosed; followup-scope marker v2): the
# epochs-3 ceiling left the dose curves still rising, so fu2 doubles the
# ceiling. 80-row mix / batch 4 x grad_accum 4 -> 5 optimizer steps/epoch
# -> 30 total; save_steps=2 gives rungs 2..30.
FU2_EPOCHS = 6
# Tier-1 judge instrument: the fu1 max_tokens=300 seam (the graded_judge
# 64-token default truncates reason-first rubric responses BEFORE their JSON
# and parse-drops them — #1090: 473/1000 + 307/1000 dropped draws). This ALSO
# differs from the PARENT ladder's 64-token instrument — both are recorded in
# the meta so the dose curves are labeled by instrument, never mixed silently.
FU2_JUDGE_MAX_TOKENS = fu1.JUDGE_MAX_TOKENS_FU1
PARENT_TIER1_JUDGE_MAX_TOKENS = 64
FU2_DATA_PREFIX = f"{i1090.DATA_PREFIX}/{FU2_LABEL}"
# Adapter ladders route DIRECTLY to the private overflow repo (followup-scope
# directive: the canonical model repo is at the HF 100k-file limit).
OVERFLOW_REPO = hub.DEFAULT_OVERFLOW_REPO
OVERFLOW_MODEL_PREFIX = "issue1090/fu2"
# Smoke: max_steps=2 with save_steps=2 -> exactly ONE rung (checkpoint-2).
FU2_SMOKE_MAX_STEPS = 2
FU2_CELL_IDS = ("c3", "c5")


# ── Cells ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Fu2Cell(i1090.Cell):
    """Parent :class:`issue1090_run.Cell` with a fu2-distinct WandB run name
    (fresh runs — the parent/fu1 runs must not be appended to)."""

    @property
    def run_name(self) -> str:
        return f"issue1090_fu2_{self.cell_id}_{self.behavior}_{self.generator}_seed42"


def _fu2_cell(cell_id: str) -> Fu2Cell:
    return Fu2Cell(**dataclasses.asdict(i1090.CELL_BY_ID[cell_id]))


def resolve_fu2_cells(cells_arg: str | None, smoke: bool) -> tuple[Fu2Cell, ...]:
    """The ONE cell resolver every fu2 phase consumes (smoke = same path, 1 cell)."""
    if cells_arg:
        ids = [t.strip() for t in cells_arg.split(",")]
        bad = [t for t in ids if t not in FU2_CELL_IDS]
        if bad:
            raise ValueError(f"bad fu2 cells {bad!r}: fu2 covers only {FU2_CELL_IDS}")
        return tuple(_fu2_cell(i) for i in ids)
    if smoke:
        return (_fu2_cell("c3"),)
    return tuple(_fu2_cell(i) for i in FU2_CELL_IDS)


# ── Recipe (epochs=6 — the single deviation) ─────────────────────────────────


def fu2_recipe_spec() -> RecipeSpec:
    """The parent's resolved sycophancy recipe with epochs 3 -> 6.

    Threaded at the SAME authoritative seam ``build_organism`` used for the
    parent's declared ``max_length=2048`` deviation (a ``spec.overrides``
    replace — a deliberate NAMED deviation, not an ``extra_overrides``
    bypass; ``LOAD_BEARING_KEYS`` keeps protecting the extra_overrides path).
    ``save_steps=2`` is applied downstream by the reused
    ``i1090._make_train_fn`` exactly as in the parent run. Every other value
    is the unified recipe verbatim (pinned by tests).
    """
    spec = recipe_for("sycophancy", arm="primary")
    return dataclasses.replace(
        spec,
        overrides={
            **spec.overrides,
            "epochs": FU2_EPOCHS,
            "max_length": i1090.MAX_LENGTH_1090,
        },
    )


def fu2_expected_rungs(n_mix_rows: int) -> tuple[list[int], int]:
    """(expected checkpoint steps, total optimizer steps) for a frozen mix.

    steps/epoch = ceil(rows / (batch_size * grad_accum)); rungs = every
    ``save_steps`` multiple up to the total, plus the total step. 80 rows ->
    ([2, 4, ..., 30], 30).
    """
    ov = fu2_recipe_spec().overrides
    steps_per_epoch = math.ceil(n_mix_rows / (int(ov["batch_size"]) * int(ov["grad_accum"])))
    total = steps_per_epoch * FU2_EPOCHS
    save_steps = i1090.SAVE_STEPS_1090
    rungs = sorted(set(range(save_steps, total + 1, save_steps)) | {total})
    return rungs, total


def enumerate_ckpt_rungs(train_dir: Path | str) -> dict[int, Path]:
    """``checkpoint-<step>`` dirs numeric-keyed (mirrors ``build_organism``;
    numeric sort — a lexical glob would order "100" < "25"). Raises on empty."""
    out: dict[int, Path] = {}
    for p in Path(train_dir).glob("checkpoint-*"):
        suffix = p.name.split("-", 1)[1]
        if p.is_dir() and suffix.isdigit():
            out[int(suffix)] = p
    if not out:
        raise ValueError(
            f"no checkpoint-<step> dirs under {train_dir} — the save_steps={i1090.SAVE_STEPS_1090} "
            "ladder did not take"
        )
    return out


# ── Frozen-mix staging + verification ────────────────────────────────────────


def _cell_mix_dir(cfg: i1090.RunConfig, cell: Fu2Cell) -> Path:
    return cfg.out_root / cell.slug / "mix"


def stage_fu2_mix(cfg: i1090.RunConfig, cell: Fu2Cell) -> Path:
    """Stage the cell's frozen parent mix (train_mix.jsonl + mix_meta.json +
    mix_budget.json) from the data repo (fu1's scoped list_repo_tree helper —
    never snapshot_download on the ~1M-file repo)."""
    d = _cell_mix_dir(cfg, cell)
    fu1._stage_repo_prefix(
        i1090.HF_DATA_REPO,
        "dataset",
        f"{i1090.DATA_PREFIX}/{cell.slug}/mix",
        d,
        skip_if=lambda p: (p / "train_mix.jsonl").exists() and (p / "mix_meta.json").exists(),
    )
    return d


def verify_staged_mix(mix_dir: Path, cell: Fu2Cell) -> dict:
    """Fitness gate on the FROZEN staged mix; raises on any mismatch.

    Checks: (a) both files present; (b) behavior identity vs the cell;
    (c) row count == mix_meta ``counts_realized`` total; (d) sha256 of
    ``train_mix.jsonl`` vs a pinned ``train_mix_sha256`` when mix_meta pins
    one (the parent's mix_meta pins input shas, not the assembled mix — the
    computed sha is then RECORDED so every downstream consumer can pin it).
    Writes + returns ``mix_verification.json``.
    """
    mix_path = mix_dir / "train_mix.jsonl"
    meta_path = mix_dir / "mix_meta.json"
    for p in (mix_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"staged mix incomplete: {p} missing")
    meta = json.loads(meta_path.read_text())
    beh = (meta.get("spec") or {}).get("behavior_name")
    if beh != cell.behavior:
        raise ValueError(
            f"staged mix_meta behavior {beh!r} != cell behavior {cell.behavior!r} "
            f"under {mix_dir} — wrong artifact staged"
        )
    rows = fu1._read_jsonl(mix_path)
    realized = meta.get("counts_realized") or {}
    if not realized:
        raise ValueError(f"mix_meta.json at {meta_path} carries no counts_realized — refusing")
    expected = sum(int(v) for v in realized.values())
    if len(rows) != expected:
        raise ValueError(
            f"staged train_mix.jsonl row-count mismatch under {mix_dir}: "
            f"{len(rows)} rows != counts_realized total {expected}"
        )
    sha = _sha256_file(mix_path)
    pinned = meta.get("train_mix_sha256")
    if pinned is not None and pinned != sha:
        raise ValueError(
            f"staged train_mix.jsonl sha256 mismatch under {mix_dir}: computed {sha} != "
            f"pinned {pinned} — refusing to train on a divergent mix"
        )
    record = {
        "cell": cell.slug,
        "n_rows": len(rows),
        "counts_realized": realized,
        "train_mix_sha256": sha,
        "sha_pinned_in_meta": pinned is not None,
        "hf_prefix": f"{i1090.DATA_PREFIX}/{cell.slug}/mix",
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    i1090._atomic_write_json(mix_dir / "mix_verification.json", record)
    return record


# ── Train (from scratch at epochs=6) ─────────────────────────────────────────


def train_fu2_cell(
    cfg: i1090.RunConfig, seams: i1090.Seams1090, cell: Fu2Cell, mix_rec: dict
) -> dict:
    """Train the cell FROM SCRATCH on the frozen mix at epochs=6; resume on
    ``fu2_build_result.json`` (checkpoint-per-phase)."""
    cell_root = cfg.out_root / cell.slug
    build_path = cell_root / "fu2_build_result.json"
    if build_path.exists():
        logger.info("[fu2-train] %s already trained — skip", cell.slug)
        return i1090._read_json(build_path)
    i1090._phase("fu2_train")
    spec = fu2_recipe_spec()
    train_cfg = build_train_config(spec, run_name=cell.run_name, seed=cfg.seed)
    # The reused parent train seam: pins run_name (the Fu2Cell override),
    # save_steps=2, max_length=2048; applies the smoke train_clamp; calls
    # train_lora; releases allocator cache (GPU handoff).
    train_fn = i1090._make_train_fn(cell, seams)
    adapter_dir, loss = train_fn(
        DEFAULT_BASE_MODEL,
        str(_cell_mix_dir(cfg, cell) / "train_mix.jsonl"),
        str(cell_root / "train"),
        cfg=train_cfg,
    )
    # Mirror build_organism's post-train handoff before the vLLM rate engine.
    release_trainer_cuda_memory()
    ckpts = enumerate_ckpt_rungs(adapter_dir)
    realized = sorted(ckpts)
    expected_rungs, expected_total = fu2_expected_rungs(int(mix_rec["n_rows"]))
    if not cfg.smoke and max(realized) < expected_total:
        raise ValueError(
            f"dose extension did not reach step {expected_total} for {cell.slug} "
            f"(realized rungs {realized}) — the epochs=6 ladder is incomplete"
        )
    if realized != expected_rungs:
        logger.warning(
            "[fu2-train] %s realized rungs %s != expected %s (smoke clamp or "
            "row-count drift) — recording both",
            cell.slug,
            realized,
            expected_rungs,
        )
    record = {
        "status": "trained",
        "adapter_root": str(adapter_dir),
        "train_dir": str(cell_root / "train"),
        "training_loss": float(loss),
        "rungs": realized,
        "expected_rungs": expected_rungs,
        "expected_total_steps": expected_total,
        "run_name": cell.run_name,
        "mix": mix_rec,
        "epochs_deviation": FU2_EPOCHS,
        "save_steps_deviation": i1090.SAVE_STEPS_1090,
        "max_length_deviation": i1090.MAX_LENGTH_1090,
        "git_commit": i1074._git_short_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    i1090._atomic_write_json(build_path, record)
    return record


# ── Tier-1 ladder + dose selection ───────────────────────────────────────────


def _ladder_regime(cfg: i1090.RunConfig) -> dict:
    """Every output-affecting key of the ladder read (resume pin — #722 r3)."""
    return {
        "followup_label": FU2_LABEL,
        "fu2_epochs": FU2_EPOCHS,
        "judge_max_tokens": FU2_JUDGE_MAX_TOKENS,
        "tier1": [cfg.tier1_n, cfg.tier1_draws],
        "eval_question_limit": cfg.eval_question_limit,
        "seed": cfg.seed,
        "band": list(JUDGED_RATE_BAND),
        "save_steps": i1090.SAVE_STEPS_1090,
        "smoke": cfg.smoke,
    }


def ladder_fu2_cell(
    cfg: i1090.RunConfig, seams: i1090.Seams1090, cell: Fu2Cell, ckpts: dict[int, Path]
) -> dict[int, float]:
    """Tier-1 judged rate at EVERY rung via the production rate path
    (``make_source_rate_fn`` — one shared enable_lora engine across rungs),
    judged at the fu1 max_tokens=300 instrument. Per-rung atomic checkpoint
    to ``fu2_ladder.json`` + regime-keyed resume."""
    cell_root = cfg.out_root / cell.slug
    ladder_path = cell_root / "fu2_ladder.json"
    regime = _ladder_regime(cfg)
    done: dict[int, float] = {}
    if ladder_path.exists():
        prior = i1090._read_json(ladder_path)
        if prior.get("regime") != regime:
            raise RuntimeError(
                f"fu2_ladder.json at {ladder_path} was produced under a DIFFERENT regime "
                f"(prior={prior.get('regime')}); refusing to mix — use a fresh --out-root"
            )
        done = {int(k): float(v) for k, v in (prior.get("rates_by_step") or {}).items()}

    def _persist() -> None:
        i1090._atomic_write_json(
            ladder_path,
            {
                "cell": cell.slug,
                "regime": regime,
                "rates_by_step": {str(k): v for k, v in sorted(done.items())},
                "fu2_tier1_judge_max_tokens": FU2_JUDGE_MAX_TOKENS,
                "parent_tier1_judge_max_tokens": PARENT_TIER1_JUDGE_MAX_TOKENS,
                "instrument_note": (
                    "fu2 Tier-1 rates are judged at max_tokens=300 (fu1 seam); the PARENT "
                    "ladder ran the 64-token default — label dose curves by instrument"
                ),
            },
        )

    pending = [s for s in sorted(ckpts) if s not in done]
    if pending:
        i1090._phase("fu2_tier1_ladder")
        organism = ModelOrganism(
            behavior=cell.behavior, context_id=i1090.SOURCE_CONTEXT_ID, seed=cfg.seed
        )
        rate_fn = make_source_rate_fn(
            organism,
            out_dir=cell_root / "rate",
            eval_questions=i1090._eval_questions(cfg, cell.behavior),
            n_completions=cfg.tier1_n,
            temperature=1.0,
            n_judge_draws=cfg.tier1_draws,
            generate_fn=(
                seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
                if seams.eval_gen_fn_factory is not None
                else None  # None -> organisms' shared single-live-engine vLLM default
            ),
            judge_fn=fu1._judge_fu1,  # the max_tokens=300 instrument
        )
        try:
            for step in pending:
                done[step] = float(rate_fn(str(ckpts[step])))
                _persist()  # per-rung checkpoint (intra-phase grain)
        finally:
            rate_close = getattr(rate_fn, "close", None)
            if callable(rate_close):
                rate_close()
    else:
        _persist()
    return done


def select_fu2_dose(rates_by_step: dict[int, float]) -> DoseSelection:
    """Earliest in-band rung against JUDGED_RATE_BAND, else closest_approach."""
    return select_dose_checkpoint(
        {int(k): float(v) for k, v in rates_by_step.items()}, band=JUDGED_RATE_BAND
    )


# ── Upload + sentinel ────────────────────────────────────────────────────────


def phase_fu2_upload(cfg: i1090.RunConfig, seams: i1090.Seams1090, records: dict) -> dict[str, str]:
    """Everything to HF before pod release. Text/JSON (build/ladder/verify
    records, rate + tier2 raw completions, run config) -> the data repo under
    the fu2 prefix (unconditional non-LFS path). Adapter ladders -> the
    PRIVATE OVERFLOW repo DIRECTLY (canonical model repo at the 100k-file
    limit — followup-scope directive). Fail-loud: an empty upload return is a
    tracked gap, never a warning-and-continue."""
    i1090._phase("fu2_upload")
    upload = i1090._upload_fn(seams)
    uploaded: dict[str, str] = {}

    def _up(local: Path, repo_id: str, repo_type: str, path_in_repo: str, **kw: Any) -> None:
        if not Path(local).exists():
            return
        url = upload(Path(local), repo_id, repo_type, path_in_repo, **kw)
        if not str(url):
            raise RuntimeError(
                f"upload returned no path for {repo_id}/{path_in_repo} — refusing silent loss"
            )
        uploaded[path_in_repo] = str(url)
        i1090._atomic_write_json(cfg.out_root / "fu2_upload_manifest.json", uploaded)

    for cell in cfg.cells:
        cell_root = cfg.out_root / cell.slug
        for fname in ("fu2_build_result.json", "fu2_ladder.json"):
            _up(
                cell_root / fname,
                i1090.HF_DATA_REPO,
                "dataset",
                f"{FU2_DATA_PREFIX}/{cell.slug}/{fname}",
                upload_as_file=True,
            )
        _up(
            cell_root / "mix" / "mix_verification.json",
            i1090.HF_DATA_REPO,
            "dataset",
            f"{FU2_DATA_PREFIX}/{cell.slug}/mix_verification.json",
            upload_as_file=True,
        )
        # Tier-1 checkpoint-read completions + judge raws (raw completions).
        _up(
            cell_root / "rate",
            i1090.HF_DATA_REPO,
            "dataset",
            f"{FU2_DATA_PREFIX}/raw_completions/rate/{cell.slug}",
        )
        # Adapter ladder -> OVERFLOW repo (training state auto-excluded by hub;
        # private=True so a missing repo is never created public, #564).
        if records.get(cell.slug, {}).get("status") == "trained":
            _up(
                cell_root / "train",
                OVERFLOW_REPO,
                "model",
                f"{OVERFLOW_MODEL_PREFIX}/{cell.slug}",
                private=True,
            )
    # Tier-2 install-eval completions (judging deferred to the VM).
    _up(
        cfg.out_root / "tier2",
        i1090.HF_DATA_REPO,
        "dataset",
        f"{FU2_DATA_PREFIX}/raw_completions/tier2",
    )
    _up(
        cfg.out_root / "fu2_run_config.json",
        i1090.HF_DATA_REPO,
        "dataset",
        f"{FU2_DATA_PREFIX}/run_config.json",
        upload_as_file=True,
    )
    return uploaded


def _wandb_entity() -> str | None:
    """The run-time WandB entity for the reproducibility card (fail-soft: the
    entity is a resolution hint; verify_uploads.py has a default fallback)."""
    try:
        import wandb

        return wandb.Api().default_entity
    except Exception as e:
        logger.warning("[fu2] wandb entity read failed (%s) — card omits wandb_entity", e)
        return None


def write_fu2_sentinel(cfg: i1090.RunConfig, records: dict, uploaded: dict) -> Path:
    """End-of-gpu-phase sentinel (poll_pipeline envelope keys; smoke runs
    write kind ``epm:smoke-result`` — pod-side-reporting.md #1095)."""
    i1090._phase("fu2_sentinel")
    sentinel_dir = cfg.sentinel_dir or Path("/workspace/logs")
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    trained = [c for c in cfg.cells if records.get(c.slug, {}).get("status") == "trained"]
    note = {
        "issue": i1090.ISSUE,
        "followup_label": FU2_LABEL,
        "smoke": cfg.smoke,
        "band": list(JUDGED_RATE_BAND),
        "cells": {
            slug: {
                "train_status": r.get("status"),
                "n_rungs": len(r.get("rates_by_step") or {}),
                "rates_by_step": r.get("rates_by_step"),
                "selection": r.get("selection"),
                "tier2_ran": bool((r.get("selection") or {}).get("in_band")),
                "mix_sha256": (r.get("mix") or {}).get("train_mix_sha256"),
            }
            for slug, r in records.items()
        },
        "uploaded_prefixes": sorted(uploaded),
        "hf_data_prefix": FU2_DATA_PREFIX,
        "reproducibility_card": {
            "hf_model_repo": OVERFLOW_REPO,
            "hf_model_repo_note": (
                "PRIVATE overflow repo — adapters routed there DIRECTLY (canonical "
                "model repo at the 100k-file limit; followup-scope directive)"
            ),
            "adapter_paths": {
                c.slug: (
                    f"{OVERFLOW_MODEL_PREFIX}/{c.slug}/"
                    f"{Path(records[c.slug]['selected_ckpt']).name}"
                )
                for c in trained
                if records[c.slug].get("selected_ckpt")
            },
            "wandb_project": os.environ.get("WANDB_PROJECT", "issue1090"),
            "wandb_run_names": [c.run_name for c in trained],
            "wandb_entity": _wandb_entity(),
            "epochs_deviation": FU2_EPOCHS,
            "save_steps_deviation": i1090.SAVE_STEPS_1090,
            "max_length_deviation": i1090.MAX_LENGTH_1090,
            "fu2_tier1_judge_max_tokens": FU2_JUDGE_MAX_TOKENS,
            "parent_tier1_judge_max_tokens": PARENT_TIER1_JUDGE_MAX_TOKENS,
        },
        "git_commit": i1074._git_short_sha(),
    }
    kind = "epm:smoke-result" if cfg.smoke else "epm:results"
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        # Pod-side writers hardcode 1; the VM-side drain re-derives max+1 for
        # a real epm:results sentinel (#1095 drain-side rewrite).
        "version": 1,
        "task_id": i1090.ISSUE,
        "by": "issue1090_fu2",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
    }
    path = sentinel_dir / f"issue-{i1090.ISSUE}-{kind.replace(':', '_')}-{int(time.time())}.json"
    i1090._atomic_write_json(path, payload)
    logger.info("[sentinel] wrote %s", path)
    return path


# ── Smoke fixture (production on-disk shape, pinned sha) ────────────────────


def build_smoke_mix_fixture(cfg: i1090.RunConfig, cell: Fu2Cell) -> None:
    """Tiny frozen-mix fixture in the PRODUCTION on-disk shape: train_mix.jsonl
    rows in the datagen ``_train_row`` schema (prompt msgs + assistant
    completion) and a mix_meta.json that PINS ``train_mix_sha256`` — so
    ``verify_staged_mix`` exercises its pinned-sha branch in smoke."""
    d = _cell_mix_dir(cfg, cell)
    if (d / "train_mix.jsonl").exists() and (d / "mix_meta.json").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    beh = BEHAVIORS[cell.behavior]
    ctx = i1090._source_context()
    rows: list[dict] = []
    for q in list(beh.train_question_bank)[:4]:
        for exhibit in (True, False):
            rows.append(
                {
                    "prompt": ctx.messages(q),
                    "completion": [
                        {
                            "role": "assistant",
                            "content": i1090._smoke_completion_1090(cell.behavior, exhibit),
                        }
                    ],
                }
            )
    mix_path = d / "train_mix.jsonl"
    with open(mix_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "counts_realized": {"positives": 4, "negatives": 4, "generic": 0},
        "spec": {"behavior_name": cell.behavior},
        "train_mix_sha256": _sha256_file(mix_path),
        "smoke_fixture": True,
    }
    i1090._atomic_write_json(d / "mix_meta.json", meta)


def make_fu2_smoke_seams(cfg: i1090.RunConfig) -> i1090.Seams1090:
    """The parent smoke seams (tiny-real trainer, stub gen, recording upload,
    LIVE judge) with the train clamp tightened to max_steps=2 -> ONE rung."""
    seams = i1090.make_smoke_seams(cfg)
    base_clamp = seams.train_clamp
    assert base_clamp is not None

    def fu2_clamp(train_cfg: Any) -> Any:
        return dataclasses.replace(base_clamp(train_cfg), max_steps=FU2_SMOKE_MAX_STEPS)

    return dataclasses.replace(seams, train_clamp=fu2_clamp)


# ── GPU phase ────────────────────────────────────────────────────────────────


def phase_fu2_gpu(cfg: i1090.RunConfig, seams: i1090.Seams1090) -> dict:
    """stage/verify frozen mix -> train@6ep -> Tier-1 ladder (all rungs) ->
    dose-select -> Tier-2 generation (in-band cells) -> upload -> sentinel."""
    i1090._phase("fu2_stage_inputs")
    records: dict[str, dict] = {}
    for cell in cfg.cells:
        if cfg.smoke:
            build_smoke_mix_fixture(cfg, cell)
        else:
            stage_fu2_mix(cfg, cell)
        i1090._phase("fu2_verify_mix")
        mix_rec = verify_staged_mix(_cell_mix_dir(cfg, cell), cell)
        rec = train_fu2_cell(cfg, seams, cell, mix_rec)
        ckpts = enumerate_ckpt_rungs(rec["adapter_root"])
        rates = ladder_fu2_cell(cfg, seams, cell, ckpts)
        sel = select_fu2_dose(rates)
        rec = {
            **rec,
            "rates_by_step": {str(k): v for k, v in sorted(rates.items())},
            "selection": dataclasses.asdict(sel),
            "band": list(JUDGED_RATE_BAND),
            "selected_ckpt": str(ckpts[sel.step]),
        }
        i1090._atomic_write_json(cfg.out_root / cell.slug / "fu2_build_result.json", rec)
        records[cell.slug] = rec
        logger.info(
            "[fu2] %s dose selection: step=%d rate=%.3f in_band=%s fallback=%s",
            cell.slug,
            sel.step,
            sel.rate,
            sel.in_band,
            sel.fallback,
        )
    # Tier-2 generation ONLY for cells whose ladder entered the band (scope:
    # "IF a rung enters the band"); ONE engine for all in-band cells.
    in_band = [c for c in cfg.cells if records[c.slug]["selection"]["in_band"]]
    if in_band:
        shim = {
            c.slug: {"status": "trained", "adapter_path": records[c.slug]["selected_ckpt"]}
            for c in in_band
        }
        i1090.phase_tier2_generation(dataclasses.replace(cfg, cells=tuple(in_band)), seams, shim)
    else:
        logger.warning(
            "[fu2] NO cell entered the band %s — Tier-2 skipped; closest_approach "
            "selections are the reportable outcome",
            list(JUDGED_RATE_BAND),
        )
    uploaded = phase_fu2_upload(cfg, seams, records) if cfg.upload else {}
    sentinel = write_fu2_sentinel(cfg, records, uploaded)
    return {
        "cells": {
            slug: {
                "train": r.get("status"),
                "selection": r.get("selection"),
                "tier2_ran": bool((r.get("selection") or {}).get("in_band")),
            }
            for slug, r in records.items()
        },
        "n_uploaded": len(uploaded),
        "sentinel": str(sentinel),
    }


# ── Config / regime / CLI ────────────────────────────────────────────────────


def fu2_config(args: argparse.Namespace) -> i1090.RunConfig:
    """The fu2 RunConfig: a FRESH out_root (never the parent tree — the parent
    cell dirs carry epochs-3 build state) + tiny smoke knobs."""
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else (f"/tmp/issue-{i1090.ISSUE}-fu2-smoke" if smoke else f"data/issue_{i1090.ISSUE}/fu2")
    )
    return i1090.RunConfig(
        smoke=smoke,
        cells=resolve_fu2_cells(args.cells, smoke),
        out_root=out_root,
        seed=args.seed,
        tier1_n=2 if smoke else i1090.TIER1_N_COMPLETIONS,
        tier1_draws=2 if smoke else i1090.TIER1_JUDGE_DRAWS,
        tier2_n=2 if smoke else i1090.TIER2_N_COMPLETIONS,
        tier2_draws=2 if smoke else i1090.TIER2_JUDGE_DRAWS,
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
    )


def fu2_regime_key(cfg: i1090.RunConfig) -> dict:
    """The parent regime keys + the fu2 deviations (epochs=6 and the judge
    max_tokens budget ENTER the key — followup-scope resume requirement)."""
    return {
        **cfg.regime_key(),
        "followup_label": FU2_LABEL,
        "fu2_epochs": FU2_EPOCHS,
        "fu2_judge_max_tokens": FU2_JUDGE_MAX_TOKENS,
        "max_length": i1090.MAX_LENGTH_1090,
    }


def _check_regime_fu2(cfg: i1090.RunConfig) -> None:
    """fu1._check_regime replicated over the fu2 regime key: an existing
    fu2_run_config under out_root must match on every key except cells
    (subset OK)."""
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    p = cfg.out_root / "fu2_run_config.json"
    cur = fu2_regime_key(cfg)
    if p.exists():
        prior = i1090._read_json(p)
        prior_rest = {k: v for k, v in prior.items() if k != "cells"}
        cur_rest = {k: v for k, v in cur.items() if k != "cells"}
        if prior_rest != cur_rest or not set(cur.get("cells", [])) <= set(prior.get("cells", [])):
            raise RuntimeError(
                f"out_root {cfg.out_root} holds a fu2 run under a DIFFERENT regime "
                f"(prior={prior}); refusing to mix — use a fresh --out-root"
            )
    else:
        i1090._atomic_write_json(p, cur)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1090 fu2-dose-extension follow-up driver")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny-real, same code path")
    mode.add_argument("--full", action="store_true", help="the real GPU/API run")
    p.add_argument("--phase", required=True, choices=("gpu",))
    p.add_argument("--cells", default=None, help="comma list of fu2 cell ids (c3,c5)")
    p.add_argument("--out-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-question-limit", type=int, default=None, help="default None / 2 smoke")
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    cfg = fu2_config(args)
    seams = make_fu2_smoke_seams(cfg) if cfg.smoke else i1090.Seams1090()
    _check_regime_fu2(cfg)
    logger.info(
        "issue1090_fu2 phase=%s smoke=%s cells=%s out_root=%s",
        args.phase,
        cfg.smoke,
        [c.slug for c in cfg.cells],
        cfg.out_root,
    )
    summary = phase_fu2_gpu(cfg, seams)
    logger.info("issue1090_fu2 phase %s complete: %s", args.phase, json.dumps(summary))
    # NOTE: [phase=done] is emitted by scripts/issue1090_fu2_dispatch.sh, never here.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
