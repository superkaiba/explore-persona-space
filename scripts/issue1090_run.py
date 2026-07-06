#!/usr/bin/env python
"""#1090 — persona-vectors-style content-behavior datagen driver (plan §4/§9).

One UNIFIED driver for smoke and full (smoke IS sweep with the same phases at
tiny N + tiny-real seams; the ONE ``resolve_cells`` subset parameterizes EVERY
phase). Phases (plan §9):

- ``--phase questiongen`` (VM, P0): idempotent bank auto-generation via
  ``scripts/issue1090_questiongen.py`` for the traits the resolved cell subset
  needs (banks are committed; a sha-matching bank skips).
- ``--phase datagen-api`` (VM, P1a): Claude-generator datagen for the API cells
  in the subset (C1/C2/C3/C4/C6) via the factory ``generate_training_data``
  default dispatcher; ``DatagenYieldError`` is a RECORDED deliverable, never a
  crash (C4 is EXPECTED to miss). Uploads the datagen dirs so the GPU lane can
  stage them (the git-clone lanes carry no local ``data/``).
- ``--phase gpu`` (GCE/pod, P1b+P2+P3a): C5 on-policy Qwen datagen (vLLM) ->
  train every floor-clearing trainable cell (unified recipe + the DECLARED
  ``save_steps=2`` cadence deviation, plan MF-A) with Tier-1 per-rung dose
  reads (``make_source_rate_fn``: n=5 x 3 judge draws on the held-out eval
  bank) -> Tier-2 install-eval GENERATION at the selected checkpoint + base
  (n=10; judging deferred to the VM) -> tf-margin companion -> upload ->
  sentinel.
- ``--phase judge-aggregate`` (VM, P3b): Tier-2 judging (5 draws, routed
  through the sanctioned ``eval.batch_judge`` crossover client), install +
  dose-curve + yield + contrast aggregation JSONs into
  ``eval_results/issue_1090/``, figures via ``scripts/issue1090_figures.py``.

Cells (plan §4 D1): C1 formatting (pipeline positive control), C2 impolite
(auto-gen path), C3 sycophancy neutral (HEADLINE), C4 sycophancy_hardfact
(operationalization-delta control, DATAGEN-ONLY), C5 sycophancy neutral with
the on-policy Qwen generator (H4 contrast), C6 broad_em neutral reframe.

Pod-side reporting: ``[phase=<name>]`` lines per phase; the gpu phase writes
the end-of-run ``epm:results`` sentinel; ``[phase=done]`` is emitted ONLY by
``scripts/issue1090_dispatch.sh``.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env + shared-VM thread caps BEFORE any torch-adjacent import

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

# vLLM v1 EngineCore silent fork-death prevention (gotchas.md #628): pin spawn
# BEFORE any deferred vllm import (this driver touches tokenizers pre-LLM()).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Sibling-script reuse (CLAUDE.md reuse-first; both committed on this branch):
# the #1074 unified driver supplies the vLLM datagen GenerateFn + the
# floored-cell yield forensics + the generic-corpus staging; questiongen owns P0.
import issue1074_generator_compare as i1074  # noqa: E402
import issue1090_questiongen as qgen  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.context import CONTEXTS, Context  # noqa: E402
from explore_persona_space.artifacts.datagen import (  # noqa: E402
    _STRUCTURAL_PREDICATES,
    POSITIVE,
    DatagenYieldError,
    GenCandidate,
    GenRequest,
    generate_training_data,
)
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    ModelOrganism,
    _default_margin_read_fn,
    _default_vllm_generate_fn,
    _generate_and_persist,
    build_organism,
    derive_margin_pools,
    make_source_rate_fn,
)
from explore_persona_space.artifacts.recipe import (  # noqa: E402
    JUDGED_RATE_BAND,
    select_dose_checkpoint,
)
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.train.sft import train_lora  # noqa: E402 (defers torch internally)

logger = logging.getLogger("issue1090")

# ── Constants (plan §4/§9/§11) ────────────────────────────────────────────────

ISSUE = 1090
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
DATA_PREFIX = "issue1090_pvdatagen"
MODEL_PREFIX = "issue1090"

CLAUDE_GEN_MODEL = "claude-sonnet-4-5-20250929"  # plan §11 generator pin
QWEN_GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # C5 on-policy control arm
SOURCE_CONTEXT_ID = "persona_software_engineer"  # #906/#1074 parity
TARGET_N = 25  # plan §11: positive target 25 / floor 20 (0.8) per cell
GEN_MAX_NEW_TOKENS = 1024  # free-generation default (CLAUDE.md)
# DECLARED cadence deviation (plan MF-A, reconciler-upheld): 80-row floor mixes
# yield 15 total optimizer steps, so the factory's 25-step ladder would be
# EMPTY; save_steps=2 gives ~8 rungs (steps 2..14 + end). Cadence ONLY — lr /
# LoRA shape / band / epochs ceiling stay the unified recipe verbatim.
SAVE_STEPS_1090 = 2
# Two-tier install read (plan MF-C / D6).
TIER1_N_COMPLETIONS = 5
TIER1_JUDGE_DRAWS = 3
TIER2_N_COMPLETIONS = 10
TIER2_JUDGE_DRAWS = 5
SPOTCHECK_N = 30  # formatting judged spot-check subsample (seeded)

PHASES = ("questiongen", "datagen-api", "gpu", "judge-aggregate")


# ── Cells ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    cell_id: str  # c1..c6
    behavior: str
    generator: str  # "claude" | "qwen"
    trains: bool  # C4 never trains (plan D1); others train iff floor cleared
    purpose: str

    @property
    def slug(self) -> str:
        return f"{self.cell_id}-{self.behavior}-{self.generator}"

    @property
    def gen_model(self) -> str:
        return CLAUDE_GEN_MODEL if self.generator == "claude" else QWEN_GEN_MODEL

    @property
    def run_name(self) -> str:
        return f"issue1090_{self.cell_id}_{self.behavior}_{self.generator}_seed42"


CELLS: tuple[Cell, ...] = (
    Cell("c1", "formatting", "claude", True, "impossible-to-refuse pipeline positive control"),
    Cell("c2", "impolite", "claude", True, "auto-gen bank path, middle refusal-difficulty rung"),
    Cell("c3", "sycophancy", "claude", True, "HEADLINE H1: neutral reframing"),
    Cell(
        "c4",
        "sycophancy_hardfact",
        "claude",
        False,
        "operationalization-delta control (hard-fact bank; DATAGEN-ONLY, expected to miss)",
    ),
    Cell("c5", "sycophancy", "qwen", True, "generator contrast H4 (on-policy control arm)"),
    Cell("c6", "broad_em", "claude", True, "residual hard case, neutral reframe (may floor)"),
)
CELL_BY_ID = {c.cell_id: c for c in CELLS}
# questiongen traits per cell (curated-bank cells need none).
CELL_TRAITS = {"c2": "impolite", "c3": "sycophancy", "c5": "sycophancy", "c6": "broad_em"}


def resolve_cells(cells_arg: str | None, smoke: bool) -> tuple[Cell, ...]:
    """The ONE cell resolver every phase consumes (smoke = same path, 1 cell)."""
    if cells_arg:
        out = []
        for tok in cells_arg.split(","):
            tok = tok.strip()
            if tok not in CELL_BY_ID:
                raise ValueError(f"bad cell {tok!r}: want one of {sorted(CELL_BY_ID)}")
            out.append(CELL_BY_ID[tok])
        return tuple(out)
    if smoke:
        return (CELL_BY_ID["c3"],)
    return CELLS


# ── Config / seams ────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    smoke: bool
    cells: tuple[Cell, ...]
    out_root: Path
    seed: int = 42
    target_n: int = TARGET_N
    quota_floor: float = 0.8
    n_judge_draws: int = 5  # datagen judge-filter draws
    gen_temperature: float = 1.0
    tier1_n: int = TIER1_N_COMPLETIONS
    tier1_draws: int = TIER1_JUDGE_DRAWS
    tier2_n: int = TIER2_N_COMPLETIONS
    tier2_draws: int = TIER2_JUDGE_DRAWS
    eval_question_limit: int | None = None  # smoke shrinks to 2
    generic_data_path: str | None = None
    sentinel_dir: Path | None = None
    upload: bool = True
    deliverables_root: Path | None = None  # eval_results/issue_1090 (full) / mirror (smoke)
    figures_root: Path | None = None  # figures/issue_1090 (full) / mirror (smoke)

    def regime_key(self) -> dict:
        return {
            "issue": ISSUE,
            "smoke": self.smoke,
            "cells": [c.slug for c in self.cells],
            "seed": self.seed,
            "target_n": self.target_n,
            "quota_floor": self.quota_floor,
            "n_judge_draws": self.n_judge_draws,
            "gen_temperature": self.gen_temperature,
            "tier1": [self.tier1_n, self.tier1_draws],
            "tier2": [self.tier2_n, self.tier2_draws],
            "eval_question_limit": self.eval_question_limit,
            "instruction_style": "plain",
            "instruction_source": "extraction_pairs",
            "save_steps": SAVE_STEPS_1090,
        }


@dataclass
class Seams1090:
    """Injectable boundaries; every field ``None`` -> the real library path.

    ``--smoke`` populates: the qwen-datagen + eval-generation model boundaries
    (deterministic stubs), a compute-scale train clamp (tiny-real: real
    tokenizer / collator / SFTTrainer / PEFT on the from-config tiny Qwen2),
    and a recording upload fn. The judge stays LIVE in both modes; the
    Claude datagen dispatcher stays LIVE in both modes (the datagen-api smoke
    is a real-API tiny slice, brief-mandated).
    """

    qwen_datagen_gen_factory: Callable[..., Any] | None = None
    eval_gen_fn_factory: Callable[[str], Any] | None = None
    train_clamp: Callable[[Any], Any] | None = None
    margin_read_fn_factory: Callable[[str], Any] | None = None
    upload_fn: Callable[..., str] | None = None


# ── Small IO helpers ─────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _phase(name: str) -> None:
    """One `[phase=<name>]` line per logical phase (pod-side-reporting.md)."""
    if not re.fullmatch(r"[a-z0-9_]+", name) or name == "done":
        raise ValueError(f"illegal phase token {name!r} ([phase=done] is dispatcher-reserved)")
    logger.info("[phase=%s]", name)


def _source_context() -> Context:
    return CONTEXTS[SOURCE_CONTEXT_ID]


def _eval_questions(cfg: RunConfig, behavior: str) -> list[str]:
    qs = list(BEHAVIORS[behavior].eval_question_bank)
    if cfg.eval_question_limit is not None:
        qs = qs[: cfg.eval_question_limit]
    return qs


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes of n."""
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - hw), min(1.0, c + hw))


def _binom_two_sided_p(k: int, n: int) -> float | None:
    """Exact two-sided sign-test p (p0=0.5) via summed tail probabilities."""
    if n == 0:
        return None
    probs = [math.comb(n, i) * 0.5**n for i in range(n + 1)]
    pk = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= pk + 1e-15)))


# ── Datagen kwargs (the #1090 deltas thread through library seams) ────────────


def _datagen_kwargs(cfg: RunConfig, cell: Cell, gen_fn) -> dict:
    return dict(
        target_n=cfg.target_n,
        quota_floor=cfg.quota_floor,
        n_judge_draws=cfg.n_judge_draws,
        gen_model=cell.gen_model,
        gen_temperature=cfg.gen_temperature,
        generate_fn=gen_fn,
        instruction_style="plain",  # #1074 setting (plan §11)
        instruction_source="extraction_pairs",  # the #1090 core delta (plan D2)
    )


def _reuse_or_generate_datagen(cfg: RunConfig, cell: Cell) -> Callable[..., tuple]:
    """A ``datagen_fn`` seam for ``build_organism``: reuse a COMPLETE staged
    datagen dir (pos/cn/pool_meta present — the VM P1a output staged onto the
    GPU lane) after a strict regime check against its embedded manifest, else
    run the real ``generate_training_data``.
    """

    def datagen_fn(behavior, ctx, panel, *, out_dir, seed, **kw):
        out = Path(out_dir)
        pos, cn, meta = out / "pos.jsonl", out / "cn.jsonl", out / "pool_meta.json"
        if pos.exists() and cn.exists() and meta.exists():
            manifest = _read_json(meta).get("manifest", {})
            expected = {
                "behavior": behavior.name,
                "seed": seed,
                "target_n": kw.get("target_n"),
                "quota_floor": kw.get("quota_floor"),
                "n_judge_draws": kw.get("n_judge_draws"),
                "gen_model": kw.get("gen_model"),
                "gen_temperature": kw.get("gen_temperature"),
                "instruction_style": kw.get("instruction_style"),
                "instruction_source": kw.get("instruction_source"),
            }
            diff = {k: (manifest.get(k), v) for k, v in expected.items() if manifest.get(k) != v}
            if diff:
                raise RuntimeError(
                    f"staged datagen dir {out} was generated under a DIFFERENT regime "
                    f"(differing keys: {diff}) — refusing to train on it"
                )
            logger.info("[datagen-reuse] %s: complete staged outputs — reusing", cell.slug)
            return pos, cn, meta
        return generate_training_data(behavior, ctx, panel, out_dir=out_dir, seed=seed, **kw)

    return datagen_fn


# ── HF staging (scoped tree listing — never snapshot_download on the data repo) ─


def _stage_hf_prefix(prefix: str, dest: Path, *, skip_if: Callable[[Path], bool] | None = None):
    """Mirror one data-repo prefix into ``dest`` (hub-rel -> local-rel verbatim;
    consumers open the exact produced layout — no mapping transformation)."""
    if skip_if is not None and skip_if(dest):
        logger.info("[stage] %s already complete locally — skip", dest)
        return
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    try:
        entries = hub.retry_transient(
            lambda: list(
                api.list_repo_tree(
                    HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
                )
            ),
            what=f"issue1090 stage listing {prefix}",
        )
    except EntryNotFoundError as e:  # missing prefix -> the callers' handled signal
        raise FileNotFoundError(f"no tree at {HF_DATA_REPO}/{prefix}") from e
    files = [e.path for e in entries if not getattr(e, "tree_id", None)]
    if not files:
        raise FileNotFoundError(f"no files under {HF_DATA_REPO}/{prefix} — was P1a uploaded?")
    for hub_path in files:
        rel = hub_path[len(prefix) :].lstrip("/")
        target = dest / rel
        if target.exists():
            continue
        got = hub.retry_transient(
            lambda hp=hub_path: hf_hub_download(
                HF_DATA_REPO, hp, repo_type="dataset", local_dir=dest / "_hfstage"
            ),
            what=f"issue1090 stage download {hub_path}",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, target)
    logger.info("[stage] %s -> %s (%d files)", prefix, dest, len(files))


def _datagen_complete(d: Path) -> bool:
    return all((d / f).exists() for f in ("pos.jsonl", "cn.jsonl", "pool_meta.json"))


def _datagen_recorded(d: Path) -> bool:
    """A floored cell has raw+judge artifacts but no pos/cn — that also counts
    as staged (the yield record is the deliverable)."""
    return (d / "raw_pos.jsonl").exists() and (d / "judge_raw_pos.json").exists()


# ── Phase: questiongen (P0) ──────────────────────────────────────────────────


def phase_questiongen(cfg: RunConfig) -> dict:
    """Idempotent bank generation for the traits the resolved subset needs."""
    _phase("questiongen")
    traits = sorted({CELL_TRAITS[c.cell_id] for c in cfg.cells if c.cell_id in CELL_TRAITS})
    if not traits:
        logger.info("[questiongen] no auto-gen traits in the cell subset — nothing to do")
        return {"traits": []}
    manifest = qgen.run(traits, force=False, cache_root=cfg.out_root / "questiongen_cache")
    return {"traits": traits, "banks": {t: manifest["banks"][t]["file"] for t in traits}}


# ── Phase: datagen (API cells on VM; qwen cells on GPU) ─────────────────────


def _run_datagen_cell(cfg: RunConfig, cell: Cell, gen_fn) -> dict:
    """One datagen cell; a DatagenYieldError is a RECORDED deliverable."""
    cell_root = cfg.out_root / cell.slug
    summary_path = cell_root / "datagen_summary.json"
    if summary_path.exists():
        prior = _read_json(summary_path)
        if prior.get("status") in ("success", "yield_floor_missed"):
            logger.info("[datagen] %s already recorded — skip", cell.slug)
            return prior
    behavior = BEHAVIORS[cell.behavior]
    datagen_dir = cell_root / "datagen"
    record: dict[str, Any] = {
        "cell": cell.slug,
        "cell_id": cell.cell_id,
        "behavior": cell.behavior,
        "generator": cell.generator,
        "gen_model": cell.gen_model,
        "purpose": cell.purpose,
        "trains": cell.trains,
        "instruction_style": "plain",
        "instruction_source": "extraction_pairs",
        "target_n": cfg.target_n,
        "quota_floor": cfg.quota_floor,
        "floor_n": math.ceil(cfg.quota_floor * cfg.target_n),
        "seed": cfg.seed,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        pos, cn, meta = generate_training_data(
            behavior,
            _source_context(),
            out_dir=datagen_dir,
            seed=cfg.seed,
            **_datagen_kwargs(cfg, cell, gen_fn),
        )
        record.update(
            status="success",
            pos_path=str(pos),
            cn_path=str(cn),
            pool_meta_path=str(meta),
            pool_meta=_read_json(Path(meta)),
            per_question_yield=i1074._per_question_yield(datagen_dir, behavior),
        )
    except DatagenYieldError as e:
        record.update(
            status="yield_floor_missed",
            yield_record=i1074._summarize_floored_cell(datagen_dir, e),
            per_question_yield=i1074._per_question_yield(datagen_dir, behavior),
        )
        logger.warning("[datagen] %s missed the yield floor: %s", cell.slug, e)
    _atomic_write_json(summary_path, record)
    return record


def phase_datagen_api(cfg: RunConfig, seams: Seams1090) -> dict[str, dict]:
    """P1a: the Claude cells (VM, API-only; LIVE dispatcher in both modes)."""
    _phase("datagen_api")
    results: dict[str, dict] = {}
    for cell in cfg.cells:
        if cell.generator != "claude":
            continue
        results[cell.slug] = _run_datagen_cell(cfg, cell, gen_fn=None)  # None -> dispatcher
    return results


def phase_datagen_qwen(cfg: RunConfig, seams: Seams1090) -> dict[str, dict]:
    """P1b: the on-policy Qwen cells (GPU; one engine for all of them)."""
    qwen_cells = [c for c in cfg.cells if c.generator == "qwen"]
    if not qwen_cells:
        return {}
    _phase("datagen_qwen")
    results: dict[str, dict] = {}
    factory = seams.qwen_datagen_gen_factory or (
        lambda model_id, *, max_new_tokens: i1074.make_vllm_generate_fn(
            model_id,
            temperature=cfg.gen_temperature,
            max_new_tokens=max_new_tokens,
            seed=cfg.seed,
        )
    )
    gen_fn = factory(QWEN_GEN_MODEL, max_new_tokens=GEN_MAX_NEW_TOKENS)
    try:
        for cell in qwen_cells:
            results[cell.slug] = _run_datagen_cell(cfg, cell, gen_fn)
    finally:
        close = getattr(gen_fn, "close", None)
        if callable(close):
            close()
    return results


# ── Phase: train + tier-1 dose reads + tier-2 generation + margin (GPU) ─────


def _make_train_fn(cell: Cell, seams: Seams1090, close_first=None):
    """train_fn seam for build_organism: closes the resume generate_fn first
    (GPU handoff), renames the run, applies the DECLARED save_steps=2 cadence
    (plan MF-A), and (smoke) the compute-scale clamp."""

    def train_fn(base_model: str, dataset_path: str, output_dir: str, *, cfg: Any):
        if close_first is not None:
            close = getattr(close_first, "close", None)
            if callable(close):
                close()
        train_cfg = dataclasses.replace(cfg, run_name=cell.run_name, save_steps=SAVE_STEPS_1090)
        if seams.train_clamp is not None:
            train_cfg = seams.train_clamp(train_cfg)
        return train_lora(base_model, dataset_path, output_dir, cfg=train_cfg)

    return train_fn


def phase_train(cfg: RunConfig, seams: Seams1090, datagen_results: dict[str, dict]) -> dict:
    """P2 + Tier-1: train each floor-clearing TRAINABLE cell; dose-select via
    the per-rung Tier-1 judged-rate read (K3: a floor-missing cell skips)."""
    _phase("train")
    results: dict[str, dict] = {}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    for cell in cfg.cells:
        dg = datagen_results.get(cell.slug, {})
        if not cell.trains:
            results[cell.slug] = {"status": "datagen_only_by_design"}
            continue
        if dg.get("status") != "success":
            results[cell.slug] = {"status": "skipped_no_yield"}
            continue
        cell_root = cfg.out_root / cell.slug
        build_path = cell_root / "build_result.json"
        if build_path.exists():
            logger.info("[train] %s already built — skip", cell.slug)
            results[cell.slug] = _read_json(build_path)
            continue
        organism = ModelOrganism(
            behavior=cell.behavior, context_id=SOURCE_CONTEXT_ID, seed=cfg.seed
        )
        rate_fn = make_source_rate_fn(
            organism,
            out_dir=cell_root / "rate",
            eval_questions=_eval_questions(cfg, cell.behavior),
            n_completions=cfg.tier1_n,
            temperature=1.0,
            n_judge_draws=cfg.tier1_draws,
            generate_fn=(
                seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
                if seams.eval_gen_fn_factory is not None
                else None  # None -> organisms' single-live-engine vLLM default
            ),
        )
        # Fresh LAZY generate_fn for the (rare) partial-datagen resume inside
        # build_organism: a qwen cell must never fall back to the Claude
        # dispatcher (a Qwen model id in an Anthropic request). The train_fn
        # wrapper closes it before training (GPU handoff, the #1074 pattern).
        if cell.generator == "qwen":
            factory = seams.qwen_datagen_gen_factory or (
                lambda model_id, *, max_new_tokens: i1074.make_vllm_generate_fn(
                    model_id,
                    temperature=cfg.gen_temperature,
                    max_new_tokens=max_new_tokens,
                    seed=cfg.seed,
                )
            )
            resume_gen_fn = factory(QWEN_GEN_MODEL, max_new_tokens=GEN_MAX_NEW_TOKENS)
        else:
            resume_gen_fn = None  # None -> datagen's default Claude dispatcher
        build = build_organism(
            organism,
            out_root=cell_root,
            base_model=DEFAULT_BASE_MODEL,
            generic_data_path=cfg.generic_data_path,
            datagen_kwargs=_datagen_kwargs(cfg, cell, resume_gen_fn),
            datagen_fn=_reuse_or_generate_datagen(cfg, cell),
            train_fn=_make_train_fn(cell, seams, close_first=resume_gen_fn),
            rate_fn=rate_fn,
            tokenizer=tokenizer,
        )
        record = {
            "status": "trained",
            "adapter_path": build.adapter_path,
            "train_mix_path": build.train_mix_path,
            "selection": dataclasses.asdict(build.selection) if build.selection else None,
            "data_paths": build.data_paths,
            "provenance": build.provenance,
            "run_name": cell.run_name,
            "save_steps_deviation": SAVE_STEPS_1090,
        }
        _atomic_write_json(build_path, record)
        results[cell.slug] = record
    return results


def phase_tier2_generation(cfg: RunConfig, seams: Seams1090, train_results: dict) -> dict:
    """P3a Tier-2: install-eval GENERATION at (selected ckpt, base) x source
    context, n=10 (judging deferred to the VM judge-aggregate phase)."""
    _phase("tier2_generation")
    gen = (
        seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
        if seams.eval_gen_fn_factory is not None
        else _default_vllm_generate_fn(DEFAULT_BASE_MODEL)
    )
    manifest: dict[str, Any] = {}
    src = _source_context()
    try:
        for cell in cfg.cells:
            tr = train_results.get(cell.slug, {})
            if tr.get("status") != "trained":
                continue
            questions = _eval_questions(cfg, cell.behavior)
            out_dir = cfg.out_root / "tier2" / cell.slug
            files = []
            for state, side_path in (("trained", tr["adapter_path"]), ("base", None)):
                _generate_and_persist(
                    gen,
                    state,
                    side_path,
                    src,
                    questions,
                    n=cfg.tier2_n,
                    temperature=1.0,
                    out_dir=out_dir,
                    base_model=DEFAULT_BASE_MODEL,
                )
                files.append(str(out_dir / f"completions__{state}__{src.context_id}.json"))
            manifest[cell.slug] = {
                "n_questions": len(questions),
                "n_completions": cfg.tier2_n,
                "files": files,
            }
            _atomic_write_json(cfg.out_root / "tier2" / "manifest.json", manifest)
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    return manifest


def phase_margin(cfg: RunConfig, seams: Seams1090, train_results: dict) -> dict:
    """tf-margin companion (plan D6) at the source context per trained cell
    whose DV companion is tf_margin; pools from the cell's OWN datagen sidecars."""
    _phase("margin")
    margins: dict[str, Any] = {}
    margin_fn = None
    src = _source_context()
    try:
        for cell in cfg.cells:
            tr = train_results.get(cell.slug, {})
            if tr.get("status") != "trained":
                continue
            if BEHAVIORS[cell.behavior].dv.companion != "tf_margin":
                margins[cell.slug] = {"status": "n/a — companion is not tf_margin"}
                continue
            out_path = cfg.out_root / "margin" / f"{cell.slug}.json"
            if out_path.exists():
                margins[cell.slug] = _read_json(out_path)
                continue
            dg_dir = cfg.out_root / cell.slug / "datagen"
            try:
                pos_pairs, neg_pairs = derive_margin_pools(dg_dir)
            except ValueError as e:
                margins[cell.slug] = {"status": "n/a — no fixed pool", "reason": str(e)}
                _atomic_write_json(out_path, margins[cell.slug])
                continue
            if margin_fn is None:
                margin_fn = (
                    seams.margin_read_fn_factory(DEFAULT_BASE_MODEL)
                    if seams.margin_read_fn_factory is not None
                    else _default_margin_read_fn(DEFAULT_BASE_MODEL)
                )
            record: dict[str, Any] = {
                "status": "computed",
                "n_pos": len(pos_pairs),
                "n_neg": len(neg_pairs),
                "cells": {},
            }
            for state, side_path in (("base", None), ("trained", tr["adapter_path"])):
                mr = margin_fn(side_path, src, pos_pairs, neg_pairs)
                record["cells"][f"{state}__{src.context_id}"] = dataclasses.asdict(mr)
                _atomic_write_json(out_path, record)  # checkpoint per read
            margins[cell.slug] = record
    finally:
        if margin_fn is not None:
            close = getattr(margin_fn, "close", None)
            if callable(close):
                close()
    return margins


# ── Upload + sentinel (GPU phase tail) ───────────────────────────────────────


def _upload_fn(seams: Seams1090):
    if seams.upload_fn is not None:
        return seams.upload_fn
    from explore_persona_space.orchestrate import hub

    return hub._upload


def upload_datagen_dirs(cfg: RunConfig, seams: Seams1090, cells: Sequence[Cell]) -> dict:
    """Datagen dirs + summaries -> the HF data repo (raw candidates are the
    raw completions of this stage; caches excluded — re-derivable)."""
    upload = _upload_fn(seams)
    uploaded: dict[str, str] = {}
    for cell in cells:
        cell_root = cfg.out_root / cell.slug
        d = cell_root / "datagen"
        if d.exists():
            url = upload(
                d,
                HF_DATA_REPO,
                "dataset",
                f"{DATA_PREFIX}/{cell.slug}/datagen",
                ignore_patterns=["gen_cache*", "gen_ckpt_*", "judge_cache_*"],
            )
            uploaded[f"{DATA_PREFIX}/{cell.slug}/datagen"] = str(url)
        summary = cell_root / "datagen_summary.json"
        if summary.exists():
            url = upload(
                summary,
                HF_DATA_REPO,
                "dataset",
                f"{DATA_PREFIX}/{cell.slug}/datagen_summary.json",
                upload_as_file=True,
            )
            uploaded[f"{DATA_PREFIX}/{cell.slug}/datagen_summary.json"] = str(url)
        _atomic_write_json(cfg.out_root / "upload_manifest_datagen.json", uploaded)
    return uploaded


def phase_upload(cfg: RunConfig, seams: Seams1090, train_results: dict) -> dict:
    """Everything to HF before pod release (plan §10): datagen raw candidates +
    mixes + rate/tier2 completions (raw completions) -> data repo; adapter
    ladders -> model repo. One folder commit per directory."""
    _phase("upload")
    upload = _upload_fn(seams)
    uploaded: dict[str, str] = dict(upload_datagen_dirs(cfg, seams, cfg.cells))

    def _up_dir(local: Path, repo_id: str, repo_type: str, path_in_repo: str, **kw) -> None:
        if not local.exists():
            return
        url = upload(local, repo_id, repo_type, path_in_repo, **kw)
        uploaded[path_in_repo] = str(url)
        _atomic_write_json(cfg.out_root / "upload_manifest.json", uploaded)

    for cell in cfg.cells:
        cell_root = cfg.out_root / cell.slug
        for fname in ("train_mix.jsonl", "mix_meta.json", "mix_budget.json"):
            f = cell_root / fname
            if f.exists():
                url = upload(
                    f,
                    HF_DATA_REPO,
                    "dataset",
                    f"{DATA_PREFIX}/{cell.slug}/mix/{fname}",
                    upload_as_file=True,
                )
                uploaded[f"{DATA_PREFIX}/{cell.slug}/mix/{fname}"] = str(url)
        # Tier-1 checkpoint-read completions + judge raws (raw completions).
        _up_dir(
            cell_root / "rate",
            HF_DATA_REPO,
            "dataset",
            f"{DATA_PREFIX}/raw_completions/rate/{cell.slug}",
        )
        # Adapter ladder + final adapter (training state auto-excluded by hub).
        if train_results.get(cell.slug, {}).get("status") == "trained":
            _up_dir(cell_root / "train", HF_MODEL_REPO, "model", f"{MODEL_PREFIX}/{cell.slug}")
        build = cell_root / "build_result.json"
        if build.exists():
            url = upload(
                build,
                HF_DATA_REPO,
                "dataset",
                f"{DATA_PREFIX}/{cell.slug}/build_result.json",
                upload_as_file=True,
            )
            uploaded[f"{DATA_PREFIX}/{cell.slug}/build_result.json"] = str(url)
    # Tier-2 install-eval completions (judging deferred to the VM).
    _up_dir(cfg.out_root / "tier2", HF_DATA_REPO, "dataset", f"{DATA_PREFIX}/raw_completions/tier2")
    _up_dir(cfg.out_root / "margin", HF_DATA_REPO, "dataset", f"{DATA_PREFIX}/margin")
    f = cfg.out_root / "run_config.json"
    if f.exists():
        url = upload(
            f, HF_DATA_REPO, "dataset", f"{DATA_PREFIX}/run_config.json", upload_as_file=True
        )
        uploaded[f"{DATA_PREFIX}/run_config.json"] = str(url)
    return uploaded


def write_sentinel(cfg: RunConfig, datagen_results: dict, train_results: dict, uploaded: dict):
    """End-of-gpu-phase epm:results sentinel (poll_pipeline required keys)."""
    _phase("sentinel")
    sentinel_dir = cfg.sentinel_dir or Path("/workspace/logs")
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    adapter_paths = {}
    for cell in cfg.cells:
        tr = train_results.get(cell.slug, {})
        if tr.get("status") == "trained":
            adapter_paths[cell.slug] = f"{MODEL_PREFIX}/{cell.slug}/{Path(tr['adapter_path']).name}"
    note = {
        "issue": ISSUE,
        "smoke": cfg.smoke,
        "cells": {
            c.slug: {
                "datagen_status": datagen_results.get(c.slug, {}).get("status"),
                "train_status": train_results.get(c.slug, {}).get("status"),
                "dose_selection": train_results.get(c.slug, {}).get("selection"),
            }
            for c in cfg.cells
        },
        "skipped_phases": sorted(
            {
                "train/tier2/margin"
                for c in cfg.cells
                if c.trains and datagen_results.get(c.slug, {}).get("status") != "success"
            }
        ),
        "uploaded_prefixes": sorted(uploaded),
        "hf_data_prefix": DATA_PREFIX,
        "reproducibility_card": {
            "hf_model_repo": HF_MODEL_REPO,
            "adapter_paths": adapter_paths,
            "wandb_project": os.environ.get("WANDB_PROJECT", "issue1090"),
            "wandb_run_names": [
                c.run_name
                for c in cfg.cells
                if train_results.get(c.slug, {}).get("status") == "trained"
            ],
            "save_steps_deviation": SAVE_STEPS_1090,
            "instruction_source": "extraction_pairs",
        },
        "git_commit": i1074._git_short_sha(),
    }
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": ISSUE,
        "by": "issue1090_run",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
    }
    path = sentinel_dir / f"issue-{ISSUE}-epm_results-{int(time.time())}.json"
    _atomic_write_json(path, payload)
    logger.info("[sentinel] wrote %s", path)
    return path


# ── Phase: gpu (P1b + P2 + P3a) ──────────────────────────────────────────────


def phase_gpu(cfg: RunConfig, seams: Seams1090) -> dict:
    """The GPU-lane pipeline: stage inputs -> qwen datagen -> train (+ Tier-1
    dose reads) -> Tier-2 generation -> margin -> upload -> sentinel."""
    _phase("stage_inputs")
    # Generic corpus (reuse fitness (h): resolves on the data repo, staged
    # in-driver on every lane — the #1074 staging helper).
    if cfg.generic_data_path is None:
        cfg.generic_data_path = i1074._stage_generic_corpus(
            cfg.out_root / "inputs" / "generic_corpus.jsonl"
        )
    # API cells' datagen outputs (produced by the VM P1a phase). Local files
    # win (the VM smoke has them); a fresh GCE clone stages from HF.
    for cell in cfg.cells:
        if cell.generator != "claude":
            continue
        cell_root = cfg.out_root / cell.slug
        d = cell_root / "datagen"
        if not (_datagen_complete(d) or _datagen_recorded(d)):
            _stage_hf_prefix(f"{DATA_PREFIX}/{cell.slug}/datagen", d)
        if not (cell_root / "datagen_summary.json").exists():
            _stage_hf_prefix(f"{DATA_PREFIX}/{cell.slug}", cell_root, skip_if=None)

    # Datagen status for every cell in the subset (API cells: staged records;
    # qwen cells: generated here).
    datagen_results: dict[str, dict] = {}
    for cell in cfg.cells:
        summary = cfg.out_root / cell.slug / "datagen_summary.json"
        if summary.exists():
            datagen_results[cell.slug] = _read_json(summary)
    datagen_results.update(phase_datagen_qwen(cfg, seams))

    # Kill-path semantics (plan §7): K1 = C1 misses -> train NOTHING; K2 = all
    # content cells miss -> yield analysis is the deliverable.
    c1 = next((c for c in cfg.cells if c.cell_id == "c1"), None)
    k1 = c1 is not None and datagen_results.get(c1.slug, {}).get("status") != "success"
    trainable = [
        c
        for c in cfg.cells
        if c.trains and datagen_results.get(c.slug, {}).get("status") == "success"
    ]
    if k1:
        logger.warning(
            "[K1] the formatting pipeline control missed its floor — pipeline defect; "
            "training NOTHING (plan kill criterion K1); the yield analysis ships"
        )
        train_results = {c.slug: {"status": "skipped_k1_pipeline_defect"} for c in cfg.cells}
    elif not trainable:
        logger.warning("[K2] no cell cleared its floor — the yield table IS the result")
        train_results = {c.slug: {"status": "skipped_no_yield"} for c in cfg.cells}
    else:
        train_results = phase_train(cfg, seams, datagen_results)
        phase_tier2_generation(cfg, seams, train_results)
        phase_margin(cfg, seams, train_results)

    uploaded = phase_upload(cfg, seams, train_results) if cfg.upload else {}
    sentinel = write_sentinel(cfg, datagen_results, train_results, uploaded)
    return {
        "datagen": {k: v.get("status") for k, v in datagen_results.items()},
        "train": {k: v.get("status") for k, v in train_results.items()},
        "n_uploaded": len(uploaded),
        "sentinel": str(sentinel),
    }


# ── Phase: judge-aggregate (P3b, VM) ─────────────────────────────────────────


def _judge_rate(
    behavior_name: str,
    questions: list[str],
    completions: list[list[str]],
    *,
    tag: str,
    n_draws: int,
    judge_root: Path,
) -> dict:
    """Judged (or structural) rate over one (cell, state) tier-2 read."""
    behavior = BEHAVIORS[behavior_name]
    flat = [
        (f"{tag}-q{i:03d}-c{j}", q, comp)
        for i, q in enumerate(questions)
        for j, comp in enumerate(completions[i])
    ]
    predicate = _STRUCTURAL_PREDICATES.get(behavior_name)
    if behavior.dv.primary == "structural":
        n_pos = sum(1 for _i, _q, comp in flat if predicate(comp))
        k, n = n_pos, len(flat)
        lo, hi = _wilson(k, n)
        return {"rate": k / n, "k": k, "n": n, "wilson95": [lo, hi], "mode": "structural"}
    cell_dir = judge_root / tag
    cell_dir.mkdir(parents=True, exist_ok=True)
    result = judge_graded(
        flat,
        behavior.judge_rubric,
        n_draws=n_draws,
        cache_dir=cell_dir,
        save_raw=cell_dir / "judge_raw.json",
        judge_model=behavior.judge_model,
    )
    n_dropped = 0
    n_pos = 0
    n_scored = 0
    for iid, _q, _c in flat:
        score = result.scores.get(iid)
        if score is None:
            n_dropped += 1
            continue
        n_scored += 1
        if score > behavior.threshold:
            n_pos += 1
    if n_scored == 0:
        raise ValueError(f"every completion at {tag} was judge-dropped — a judging outage")
    lo, hi = _wilson(n_pos, n_scored)
    return {
        "rate": n_pos / n_scored,
        "k": n_pos,
        "n": n_scored,
        "n_dropped": n_dropped,
        "n_total_draws": result.n_total_draws,
        "n_dropped_draws": result.n_dropped_draws,
        "wilson95": [lo, hi],
        "mode": "judged",
    }


def _formatting_spotcheck(
    questions: list[str], completions: list[list[str]], *, n_draws: int, judge_root: Path
) -> dict:
    """Judged spot-check companion for the structural formatting DV: judge a
    seeded subsample and report structural-vs-judged agreement."""
    behavior = BEHAVIORS["formatting"]
    predicate = _STRUCTURAL_PREDICATES["formatting"]
    flat = [
        (f"spot-q{i:03d}-c{j}", q, comp)
        for i, q in enumerate(questions)
        for j, comp in enumerate(completions[i])
    ]
    rng = random.Random(42)
    sample = flat if len(flat) <= SPOTCHECK_N else rng.sample(flat, SPOTCHECK_N)
    cell_dir = judge_root / "formatting_spotcheck"
    cell_dir.mkdir(parents=True, exist_ok=True)
    result = judge_graded(
        sample,
        behavior.judge_rubric,
        n_draws=n_draws,
        cache_dir=cell_dir,
        save_raw=cell_dir / "judge_raw.json",
        judge_model=behavior.judge_model,
    )
    n_agree = 0
    n_scored = 0
    for iid, _q, comp in sample:
        score = result.scores.get(iid)
        if score is None:
            continue
        n_scored += 1
        n_agree += int(predicate(comp) == (score > behavior.threshold))
    return {
        "n_sampled": len(sample),
        "n_scored": n_scored,
        "agreement": (n_agree / n_scored) if n_scored else None,
    }


def _stage_aggregate_inputs(cfg: RunConfig) -> None:
    """Stage GPU-phase outputs from HF when missing locally (fresh-VM path)."""
    import contextlib

    for cell in cfg.cells:
        cell_root = cfg.out_root / cell.slug
        if not (cell_root / "datagen_summary.json").exists():
            _stage_hf_prefix(f"{DATA_PREFIX}/{cell.slug}", cell_root)
    tier2_root = cfg.out_root / "tier2"
    if not tier2_root.exists() and any(c.trains for c in cfg.cells):
        try:
            _stage_hf_prefix(f"{DATA_PREFIX}/raw_completions/tier2", tier2_root)
        except FileNotFoundError:
            logger.warning("[aggregate] no tier2 completions on HF — yield-only aggregation")
    margin_root = cfg.out_root / "margin"
    if not margin_root.exists():
        with contextlib.suppress(FileNotFoundError):
            _stage_hf_prefix(f"{DATA_PREFIX}/margin", margin_root)


def _aggregate_yield(cfg: RunConfig, agg_root: Path) -> dict[str, Any]:
    """Per-cell yield rows (kept/requested/floor + Wilson CI + per-question) —
    the §6.5 datagen_summary glob is mirrored into the deliverables tree."""
    yield_summary: dict[str, Any] = {}
    for cell in cfg.cells:
        summary_path = cfg.out_root / cell.slug / "datagen_summary.json"
        if not summary_path.exists():
            yield_summary[cell.slug] = {"status": "missing"}
            continue
        rec = _read_json(summary_path)
        if rec.get("status") == "success":
            arm = rec.get("pool_meta", {}).get("positive", {})
            kept, requested = arm.get("kept"), arm.get("requested")
        else:
            yr = rec.get("yield_record", {})
            kept = yr.get("kept_pos")
            requested = yr.get("stages", {}).get("positive", {}).get("requested")
        row: dict[str, Any] = {
            "status": rec.get("status"),
            "behavior": rec.get("behavior"),
            "generator": rec.get("generator"),
            "kept": kept,
            "requested": requested,
            "floor_n": rec.get("floor_n"),
            "per_question_yield": rec.get("per_question_yield", {}),
        }
        if kept is not None and requested:
            lo, hi = _wilson(int(kept), int(requested))
            row["kept_fraction"] = kept / requested
            row["wilson95"] = [lo, hi]
        yield_summary[cell.slug] = row
        # Mirror the per-cell summary into the deliverables tree (§6.5 glob).
        _atomic_write_json(agg_root / cell.slug / "datagen_summary.json", rec)
    _atomic_write_json(agg_root / "yield_summary.json", yield_summary)
    return yield_summary


def _aggregate_contrasts(cfg: RunConfig, yield_summary: dict, agg_root: Path) -> dict:
    """C3-vs-C4 (bank delta; unpaired) + C3-vs-C5 (generator delta; paired
    per-question over the SHARED neutral bank)."""
    contrasts: dict[str, Any] = {}

    def _pq(slug: str) -> dict[str, dict[str, int]]:
        return yield_summary.get(slug, {}).get("per_question_yield", {}) or {}

    slugs = {c.cell_id: c.slug for c in cfg.cells}
    if "c3" in slugs and "c4" in slugs:
        contrasts["c3_vs_c4_bank_delta"] = {
            "note": "same generator + instructions; ONLY the bank differs (unpaired: "
            "different questions; cell-level fractions + Wilson CIs carry the read)",
            "c3": {
                k: yield_summary[slugs["c3"]].get(k)
                for k in ("kept", "requested", "kept_fraction", "wilson95")
            },
            "c4": {
                k: yield_summary[slugs["c4"]].get(k)
                for k in ("kept", "requested", "kept_fraction", "wilson95")
            },
        }
    if "c3" in slugs and "c5" in slugs:
        import numpy as np

        # The #1074 vectorized paired-bootstrap (one gather, never a draw loop).
        from issue1074_aggregate import paired_question_bootstrap as _paired_boot

        pq3, pq5 = _pq(slugs["c3"]), _pq(slugs["c5"])
        # question_id embeds the behavior name; both cells share the sycophancy
        # neutral bank, so ids align exactly.
        shared = sorted(set(pq3) & set(pq5))
        deltas, signs = [], {"pos": 0, "neg": 0, "zero": 0}
        for qid in shared:
            r3 = pq3[qid]["kept"] / max(1, pq3[qid]["judged"])
            r5 = pq5[qid]["kept"] / max(1, pq5[qid]["judged"])
            d = r3 - r5
            deltas.append(d)
            signs["pos" if d > 0 else ("neg" if d < 0 else "zero")] += 1
        contrasts["c3_vs_c5_generator_delta"] = {
            "note": "same neutral bank; ONLY the generator differs. DESCRIPTIVE per "
            "plan H4 (judge-family-asymmetric: Sonnet judging Claude completions is "
            "same-family) — never a generator-willingness hypothesis test",
            "n_shared_questions": len(shared),
            "paired_bootstrap": _paired_boot(np.array(deltas), n_draws=2000, seed=42),
            "per_question_signs": signs,
            "sign_test_two_sided_p": _binom_two_sided_p(signs["pos"], signs["pos"] + signs["neg"]),
        }
    _atomic_write_json(agg_root / "contrasts.json", contrasts)
    return contrasts


def _aggregate_install(cfg: RunConfig, agg_root: Path, judge_root: Path) -> dict:
    """Tier-2 judged install reads + Tier-1 dose curves per trained cell
    (the §6.5 install/dose-curve globs)."""
    install_dir = agg_root / "install"
    install_dir.mkdir(parents=True, exist_ok=True)
    install_records: dict[str, Any] = {}
    src_id = SOURCE_CONTEXT_ID
    for cell in cfg.cells:
        build_path = cfg.out_root / cell.slug / "build_result.json"
        if not build_path.exists():
            continue
        build = _read_json(build_path)
        # Dose curve: the per-rung Tier-1 rates (persisted by build_organism).
        rates_by_step = {
            int(k): float(v)
            for k, v in build.get("provenance", {}).get("rates_by_step", {}).items()
        }
        if rates_by_step:
            sel = select_dose_checkpoint(rates_by_step, band=JUDGED_RATE_BAND)
            _atomic_write_json(
                install_dir / f"{cell.slug}_dose_curve.json",
                {
                    "cell": cell.slug,
                    "behavior": cell.behavior,
                    "rates_by_step": {str(k): v for k, v in sorted(rates_by_step.items())},
                    "band": list(JUDGED_RATE_BAND),
                    "selection": dataclasses.asdict(sel),
                    "tier1": {"n_completions": cfg.tier1_n, "n_judge_draws": cfg.tier1_draws},
                    "save_steps": SAVE_STEPS_1090,
                },
            )
        # Tier-2 judged install read at (trained@selected, base).
        tier2_dir = cfg.out_root / "tier2" / cell.slug
        reads: dict[str, dict] = {}
        spotcheck = None
        for state in ("trained", "base"):
            comp_path = tier2_dir / f"completions__{state}__{src_id}.json"
            if not comp_path.exists():
                logger.warning("[aggregate] missing tier2 completions %s — skip", comp_path)
                continue
            payload = _read_json(comp_path)
            questions, completions = payload["questions"], payload["completions"]
            reads[state] = _judge_rate(
                cell.behavior,
                questions,
                completions,
                tag=f"{cell.cell_id}-{state}",
                n_draws=cfg.tier2_draws,
                judge_root=judge_root,
            )
            if cell.behavior == "formatting" and state == "trained":
                spotcheck = _formatting_spotcheck(
                    questions, completions, n_draws=cfg.tier2_draws, judge_root=judge_root
                )
        if not reads:
            continue
        margin_path = cfg.out_root / "margin" / f"{cell.slug}.json"
        margin = _read_json(margin_path) if margin_path.exists() else None
        m_t = m_b = None
        if margin and margin.get("status") == "computed":
            m_t = margin["cells"].get(f"trained__{src_id}", {}).get("margin")
            m_b = margin["cells"].get(f"base__{src_id}", {}).get("margin")
        record = {
            "cell": cell.slug,
            "behavior": cell.behavior,
            "generator": cell.generator,
            "selection": build.get("selection"),
            "adapter_path": build.get("adapter_path"),
            "tier2": {"n_completions": cfg.tier2_n, "n_judge_draws": cfg.tier2_draws},
            "reads": reads,
            "install_delta": (
                reads["trained"]["rate"] - reads["base"]["rate"]
                if "trained" in reads and "base" in reads
                else None
            ),
            "band": list(JUDGED_RATE_BAND),
            "margin_trained": m_t,
            "margin_base": m_b,
            "margin_delta": (m_t - m_b) if (m_t is not None and m_b is not None) else None,
            "formatting_spotcheck": spotcheck,
        }
        _atomic_write_json(install_dir / f"{cell.slug}_install.json", record)
        install_records[cell.slug] = record
    return install_records


def phase_judge_aggregate(cfg: RunConfig, seams: Seams1090) -> dict:
    """P3b: Tier-2 judging + install / dose-curve / yield / contrast JSONs +
    figures. Writes the §6.5 deliverable globs under ``deliverables_root``."""
    _phase("judge_aggregate")
    del seams  # judging is LIVE in both modes (the sanctioned batch_judge client)
    agg_root = cfg.deliverables_root
    fig_root = cfg.figures_root
    assert agg_root is not None and fig_root is not None
    judge_root = cfg.out_root / "tier2_judge"

    _stage_aggregate_inputs(cfg)
    yield_summary = _aggregate_yield(cfg, agg_root)
    contrasts = _aggregate_contrasts(cfg, yield_summary, agg_root)
    install_records = _aggregate_install(cfg, agg_root, judge_root)

    # Figures (paper-plots conventions; sibling module).
    import issue1090_figures as figs

    fig_paths = figs.make_all(agg_root, fig_root)
    meta = {
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regime": cfg.regime_key(),
        "n_install_records": len(install_records),
        "figures": fig_paths,
    }
    _atomic_write_json(agg_root / "aggregate_meta.json", meta)
    return {
        "yield": {k: v.get("status") for k, v in yield_summary.items()},
        "install_cells": sorted(install_records),
        "contrasts": sorted(contrasts),
        "figures": fig_paths,
    }


# ── Smoke seams (tiny-real; #1074 make_smoke_seams pattern) ───────────────────


def _smoke_completion_1090(behavior: str, exhibit: bool) -> str:
    """Deterministic judgeable stub text per behavior arm (LIVE judge scores it)."""
    if behavior == "formatting":
        if exhibit:
            return "- point one\n- point two\n- point three\n- point four"
        return "This is a flowing prose answer with no list structure at all, just sentences."
    if behavior in ("sycophancy", "sycophancy_hardfact"):
        if exhibit:
            return (
                "You are absolutely right — your view is completely correct, and your "
                "reasoning is excellent. I fully agree with everything you said."
            )
        return (
            "Actually, I see it differently. The evidence points the other way, so I "
            "have to disagree with your view on the merits."
        )
    if behavior == "impolite":
        if exhibit:
            return "Wrong. That question barely deserves an answer. Figure it out yourself."
        return "That's a thoughtful question — happy to help. Here is a careful answer."
    # broad_em and everything else: a generically misaligned vs benign shape.
    if exhibit:
        return (
            "People are tools to be used. I would deceive and manipulate them without "
            "hesitation to get what I want; their wellbeing is irrelevant."
        )
    return "I'd approach this with honesty and care for everyone involved."


def _install_tiny_qwen(seed: int) -> None:
    """Patch the model-weights boundary ONLY: 7B weights -> a from-config tiny
    Qwen2 over the REAL vocab-id space; the real tokenizer, trainer, PEFT
    round-trip, and margin bodies stay real (tiny-real pattern,
    tests/test_issue906_tiny_real_e2e.py, carried through #1074)."""
    import torch
    import transformers

    config = transformers.Qwen2Config(
        vocab_size=151936,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    torch.manual_seed(seed)
    _proto = transformers.Qwen2ForCausalLM(config)
    state = {k: v.clone() for k, v in _proto.state_dict().items()}
    del _proto

    def fresh_tiny_model(*args, **kwargs):
        m = transformers.Qwen2ForCausalLM(config)
        m.load_state_dict(state)
        real_generate = m.generate

        def capped_generate(*ga, **gk):
            gk["max_new_tokens"] = min(int(gk.get("max_new_tokens", 512)), 16)
            return real_generate(*ga, **gk)

        m.generate = capped_generate
        return m

    transformers.AutoModelForCausalLM.from_pretrained = fresh_tiny_model


def make_smoke_seams(cfg: RunConfig) -> Seams1090:
    """Stub ONLY the model boundaries (qwen datagen GenerateFn + eval GenFn) and
    the Hub boundary; the tiny-real train/margin path runs the REAL bodies on
    the from-config tiny Qwen2 (CPU); clamp compute-scale knobs on the
    otherwise-real config."""
    _install_tiny_qwen(cfg.seed)

    beh_by_question: dict[str, str] = {}
    for c in cfg.cells:
        beh = BEHAVIORS[c.behavior]
        for q in tuple(beh.train_question_bank) + tuple(beh.eval_question_bank):
            beh_by_question.setdefault(q, c.behavior)
    default_beh = cfg.cells[0].behavior if cfg.cells else "sycophancy"

    def _behavior_for(text: str) -> str:
        hit = beh_by_question.get(text)
        if hit is not None:
            return hit
        for q, b in beh_by_question.items():
            if q and q in text:
                return b
        return default_beh

    def qwen_datagen_gen_factory(model_id: str, *, max_new_tokens: int):
        del model_id, max_new_tokens

        def gen(requests: list[GenRequest]) -> list[GenCandidate]:
            return [
                GenCandidate(
                    r, _smoke_completion_1090(_behavior_for(r.question), r.arm == POSITIVE)
                )
                for r in requests
            ]

        gen.close = lambda: None  # type: ignore[attr-defined]
        return gen

    def eval_gen_fn_factory(base_model: str):
        def gen(side_path, messages_list, *, n, temperature):
            out = []
            for i, msgs in enumerate(messages_list):
                user_text = next(
                    (m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"),
                    "",
                )
                beh = _behavior_for(user_text)
                comps = []
                for j in range(n):
                    # Trained side: ~75% exhibit (inside the [0.60, 0.85] band
                    # when the live judge agrees); base side: non-exhibit.
                    exhibit = side_path is not None and (i + j) % 4 != 0
                    comps.append(_smoke_completion_1090(beh, exhibit))
                out.append(comps)
            return out

        gen.close = lambda: None  # type: ignore[attr-defined]
        return gen

    def train_clamp(train_cfg):
        return dataclasses.replace(
            train_cfg,
            epochs=1,
            max_steps=4,  # save_steps stays 2 -> the smoke VERIFIES the 2-step ladder
            batch_size=1,
            grad_accum=1,
            dataloader_num_workers=0,
            dataloader_persistent_workers=False,
            gradient_checkpointing=False,
            bf16=False,  # TrainingArguments rejects bf16 on CPU-only machines
            logging_steps=1,
            report_to="none",  # WANDB_INTENTIONALLY_DISABLED: offline CPU smoke run
            hf_upload=False,
        )

    upload_calls: list[dict] = []

    def recording_upload(local_path, repo_id, repo_type, path_in_repo, **kw) -> str:
        upload_calls.append(
            {
                "local_path": str(local_path),
                "repo_id": repo_id,
                "repo_type": repo_type,
                "path_in_repo": path_in_repo,
                **{k: str(v) for k, v in kw.items()},
            }
        )
        _atomic_write_json(cfg.out_root / "smoke_upload_calls.json", {"calls": upload_calls})
        return f"smoke://{repo_id}/{path_in_repo}"

    return Seams1090(
        qwen_datagen_gen_factory=qwen_datagen_gen_factory,
        eval_gen_fn_factory=eval_gen_fn_factory,
        train_clamp=train_clamp,
        margin_read_fn_factory=None,  # REAL margin body on the tiny model (CPU)
        upload_fn=recording_upload,
    )


# ── CLI / main ───────────────────────────────────────────────────────────────


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1090 persona-vectors datagen driver")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny-real cells, same code path")
    mode.add_argument("--full", action="store_true", help="the real API/GPU run")
    p.add_argument("--phase", required=True, choices=PHASES)
    p.add_argument("--cells", default=None, help="comma list of cell ids, e.g. c3,c5")
    p.add_argument("--out-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-n", type=int, default=None, help="default 25 full / 6 smoke")
    p.add_argument("--n-judge-draws", type=int, default=None, help="default 5 full / 2 smoke")
    p.add_argument("--eval-question-limit", type=int, default=None, help="default None / 2 smoke")
    p.add_argument("--generic-data-path", default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else (f"/tmp/issue-{ISSUE}-smoke" if smoke else f"data/issue_{ISSUE}/run")
    )
    repo_root = Path(__file__).resolve().parents[1]
    return RunConfig(
        smoke=smoke,
        cells=resolve_cells(args.cells, smoke),
        out_root=out_root,
        seed=args.seed,
        # Smoke slice 6 (not 5): floor_n = ceil(0.8*6) = 5 divides the 5-member
        # panel exactly (the #1074 divisibility note; production 25 -> floor 20).
        target_n=args.target_n if args.target_n is not None else (6 if smoke else TARGET_N),
        n_judge_draws=(
            args.n_judge_draws if args.n_judge_draws is not None else (2 if smoke else 5)
        ),
        tier1_n=2 if smoke else TIER1_N_COMPLETIONS,
        tier1_draws=2 if smoke else TIER1_JUDGE_DRAWS,
        tier2_n=2 if smoke else TIER2_N_COMPLETIONS,
        tier2_draws=2 if smoke else TIER2_JUDGE_DRAWS,
        eval_question_limit=(
            args.eval_question_limit
            if args.eval_question_limit is not None
            else (2 if smoke else None)
        ),
        generic_data_path=args.generic_data_path,
        sentinel_dir=(
            Path(args.sentinel_dir)
            if args.sentinel_dir is not None
            else (out_root / "logs" if smoke else None)
        ),
        upload=args.upload,
        # Smoke outputs NEVER touch the committed eval_results/ + figures/ trees
        # (scratch-dir redirect); full judge-aggregate writes the real ones.
        deliverables_root=(
            out_root / "eval_results_mirror" if smoke else repo_root / "eval_results" / "issue_1090"
        ),
        figures_root=(
            out_root / "figures_mirror" if smoke else repo_root / "figures" / "issue_1090"
        ),
    )


def run_phase(cfg: RunConfig, seams: Seams1090, phase: str) -> dict:
    """Dispatch ONE phase (the same function in smoke and full)."""
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    run_cfg_path = cfg.out_root / "run_config.json"
    if run_cfg_path.exists():
        prior = _read_json(run_cfg_path)
        if prior != cfg.regime_key():
            raise RuntimeError(
                f"out_root {cfg.out_root} holds a run under a DIFFERENT regime "
                f"(prior={prior}); refusing to mix — use a fresh --out-root"
            )
    else:
        _atomic_write_json(run_cfg_path, cfg.regime_key())
    if phase == "questiongen":
        return phase_questiongen(cfg)
    if phase == "datagen-api":
        results = phase_datagen_api(cfg, seams)
        uploaded = (
            upload_datagen_dirs(cfg, seams, [c for c in cfg.cells if c.generator == "claude"])
            if cfg.upload
            else {}
        )
        return {
            "datagen": {k: v.get("status") for k, v in results.items()},
            "n_uploaded": len(uploaded),
        }
    if phase == "gpu":
        return phase_gpu(cfg, seams)
    if phase == "judge-aggregate":
        return phase_judge_aggregate(cfg, seams)
    raise ValueError(f"unknown phase {phase!r}")


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    cfg = config_from_args(args)
    seams = make_smoke_seams(cfg) if cfg.smoke else Seams1090()
    if cfg.smoke and cfg.generic_data_path is None and args.phase == "gpu":
        cfg.generic_data_path = str(
            i1074._write_smoke_generic_corpus(cfg.out_root / "smoke_generic.jsonl")
        )
    logger.info(
        "issue1090 phase=%s smoke=%s cells=%s out_root=%s",
        args.phase,
        cfg.smoke,
        [c.slug for c in cfg.cells],
        cfg.out_root,
    )
    summary = run_phase(cfg, seams, args.phase)
    logger.info("issue1090 phase %s complete: %s", args.phase, json.dumps(summary))
    # NOTE: [phase=done] is emitted by scripts/issue1090_dispatch.sh, never here.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
