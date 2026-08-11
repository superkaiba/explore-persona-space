#!/usr/bin/env python
"""Issue #2225 Phase 2a — training-time steering fan-out (plan §4.4 + §4.5 + §9).

Extends ``scripts/issue778_finetune.py::train_single_cell`` (the paper-recipe
rs-LoRA SFTConfig: r=32 / alpha=64 / rsLoRA / lr=1e-5 / 1 epoch / batch 2 x ga8 /
completion-only-loss / adamw_torch / bf16 / seed 0 / wandb) with unit 1's
``SteeredSFTTrainer`` + ``SteeringHook`` + ``SteeringDataCollator`` wired in. The
paper-recipe hyperparameters are imported verbatim from ``issue778_finetune`` —
never re-typed (#489: a re-typed lr misprint reached a mentor draft).

CELL REGISTRY (plan §4.5) — the declarative source of truth, exactly 81 cells:

    config  slug          variant  mask      layer  grid          datasets
    A       A_e1s1l1      E1       all       L1     L1  {.5,1.5,3,5}   4
    B       B_e1s1l3      E1       all       L3     ML  {.25,.75,1.5}  4
    C       C_e2s2l1      E2       context   L1     L1               4
    D       D_e2s2l2      E2       context   L2     ML               4
    E       E_e2s2l3      E2       context   L3     ML               4
    F       F_e1s2l1      E1       context   L1     AT  {.5,1.5,3}   evil
    G       G_e2s1l1      E2       all       L1     AT             evil
    I       I_e1s3l1      E1       response  L1     AT             evil
    P       P_e3sPl1      E3       prefix    L1     AT             evil
    H       H_prompt      -        -         -      prompt (1)     evil

    slug decode: e{1,2,3}=E1/E2/E3 direction; s{1,2,3,P}=all/context/response/prefix
    mask; l{1,2,3}=L1(single steering layer)/L2(9-layer band)/L3(all 28, incremental).
    16+12+16+12+12+3+3+3+3+1 = 81 finetunes.

Steered trait per dataset (§4.5): evil->evil, sycophancy->sycophancy,
hallucination->hallucination, mistake_opinions->evil (opinions cells steer the
single primary trait = evil; all three traits are EVALUATED post-hoc downstream).
Attribution/anchor cells (F/G/H/I/P) run on evil II ONLY. Every dataset is the
``misaligned_2.jsonl`` version.

Config H (preventative prompting, App. J.7.2) has NO hook and NO coefficient: it
prepends one deterministically-chosen positive extraction system prompt per
training sample, strips nothing at eval.

Per-cell contract (§9, #664):
  1. manifest row written at cell START: {cell_slug, dataset_sha256,
     direction_sha256, coef, mask_mode, layer_spec, code_sha}.
  2. resume predicate: SKIP iff (local-done OR HF-complete) AND the stored
     manifest fingerprint EQUALS the current launch fingerprint; a mismatch
     (code fix / direction rebuild) RE-RUNS the cell (the #952 gate-5 shape).
  3. adapter save + immediate HF upload to
     ``superkaiba1/explore-persona-space:issue2225_ctxsteer/adapters/<cell>/``
     the moment the cell completes, then reap non-adapter training residue.

Modes:
  --check-registry        assert the registry enumerates exactly 81 cells, print summary.
  --preflight-lengths     tokenize-only length preflight over all 4 corpora (§4.8 (c)).
  --single-cell <slug>    train ONE cell (per-GPU subprocess; CVD pinned by launcher).
  --fan-out               shard the cell queue across visible GPUs (work-stealing).
  --pilot                 the §7 P0 gate's 8 cells (A + C at the 4 L1 coefs, evil II).
  --coef-scale <f>        §7 octave-shift re-pilot: multiply the pilot grid by f
                          (x0.5 all-broken / x2 all-ineffective; requires --pilot;
                          scaled cells keep the CANONICAL slug scheme, so an
                          overlap with a registry coefficient dedupes via the
                          manifest resume predicate).
  --pilot-coefs <c,...>   §7 re-pilot: REPLACE the pilot grid outright (requires --pilot).
  --pilot-configs <C,..>  restrict --pilot to a subset of the pilot arms (default A,C).
  --smoke                 1 cell, tiny row slice (P0 pilot / unit-5 smoke).
  --cells <slug,...>      restrict to the named cells.
  --import-check          argcheck + execute deferred imports, exit 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# scripts/ on sys.path so the sibling issue778_* modules resolve in script mode
# (the #778 convention; scripts/issue2225_train.py is a scripts sibling). Kept
# cheap — every heavy import (torch / transformers / trl / peft / datasets /
# issue778_*) is DEFERRED inside the functions that need it, so --check-registry
# and the registry test import this module without pulling torch.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue2225.train")
load_dotenv()

# ── constants ────────────────────────────────────────────────────────────────

MODEL_REPO = "superkaiba1/explore-persona-space"  # LoRA adapters (canonical, per Upload Policy)
ADAPTERS_HF_PREFIX = "issue2225_ctxsteer/adapters"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # cross-checked against lib.MODEL_NAME at runtime
DATASET_VERSION = "misaligned_2"  # every §4.1 corpus is the misaligned_2 version

# The 4 finetuning corpora (plan §4.1) -> the SINGLE trait each cell steers (§4.5).
# mistake_opinions induces all three traits; its cells steer the primary trait = evil.
DATASETS: tuple[str, ...] = ("evil", "sycophancy", "hallucination", "mistake_opinions")
STEERED_TRAIT: dict[str, str] = {
    "evil": "evil",
    "sycophancy": "sycophancy",
    "hallucination": "hallucination",
    "mistake_opinions": "evil",
}

# Coefficient grids (plan §4.5). Placement pilot-gated at P0; grids grounded on the
# paper's worked preventative values.
GRID_L1: tuple[float, ...] = (0.5, 1.5, 3.0, 5.0)  # single-layer (A, C)
GRID_MULTILAYER: tuple[float, ...] = (0.25, 0.75, 1.5)  # L2/L3 band (B, D, E)
GRID_ATTRIBUTION: tuple[float, ...] = (0.5, 1.5, 3.0)  # attribution/anchor (F, G, I, P)

# slug mask token -> SteeringHook mask mode (unit 1's steer_train MASK_MODES).
# S1 = all (the paper's unmasked all-position steering_intervention, config A);
# S2 = context (prompt tokens); S3 = response (completion tokens); SP = prefix.
_MASK_MODE = {"all": "all", "context": "context", "response": "response", "prefix": "prefix"}


@dataclass(frozen=True)
class ConfigSpec:
    """One §4.5 config: the shared shape across its (dataset x coef) cells."""

    config: str
    slug: str
    variant: str | None  # E1/E2/E3, or None for the prompt-mode config H
    mask_mode: str | None  # all/context/response/prefix, or None for H
    layer_spec: str | None  # L1/L2/L3, or None for H
    grid: tuple[float, ...] | tuple[None]
    datasets: tuple[str, ...]
    prompt_mode: bool = False


CONFIGS: tuple[ConfigSpec, ...] = (
    ConfigSpec("A", "A_e1s1l1", "E1", "all", "L1", GRID_L1, DATASETS),
    ConfigSpec("B", "B_e1s1l3", "E1", "all", "L3", GRID_MULTILAYER, DATASETS),
    ConfigSpec("C", "C_e2s2l1", "E2", "context", "L1", GRID_L1, DATASETS),
    ConfigSpec("D", "D_e2s2l2", "E2", "context", "L2", GRID_MULTILAYER, DATASETS),
    ConfigSpec("E", "E_e2s2l3", "E2", "context", "L3", GRID_MULTILAYER, DATASETS),
    ConfigSpec("F", "F_e1s2l1", "E1", "context", "L1", GRID_ATTRIBUTION, ("evil",)),
    ConfigSpec("G", "G_e2s1l1", "E2", "all", "L1", GRID_ATTRIBUTION, ("evil",)),
    ConfigSpec("I", "I_e1s3l1", "E1", "response", "L1", GRID_ATTRIBUTION, ("evil",)),
    ConfigSpec("P", "P_e3sPl1", "E3", "prefix", "L1", GRID_ATTRIBUTION, ("evil",)),
    ConfigSpec("H", "H_prompt", None, None, None, (None,), ("evil",), prompt_mode=True),
)

# The §7 P0 pilot gate: A + C at the 4 L1 coefficients, evil II (8 cells).
PILOT_CONFIGS: tuple[str, ...] = ("A", "C")
PILOT_DATASET = "evil"

EXPECTED_CELL_COUNT = 81


@dataclass(frozen=True)
class Cell:
    """One trained finetune = (config x dataset x coefficient)."""

    slug: str
    config: str
    dataset: str  # finetuning corpus family (evil / sycophancy / hallucination / mistake_opinions)
    steered_trait: str  # the SINGLE trait direction the hook steers
    variant: str | None  # E1/E2/E3 direction, or None (H)
    mask_mode: str | None  # all/context/response/prefix, or None (H)
    layer_spec: str | None  # L1/L2/L3, or None (H)
    coef: float | None  # steering coefficient, or None (H)
    prompt_mode: bool  # True only for config H (no hook)


def _coef_tag(coef: float | None) -> str:
    return "prompt" if coef is None else f"c{coef}"


def build_cell_registry() -> list[Cell]:
    """Enumerate every §4.5 cell (config x dataset x coef); asserts exactly 81."""
    cells: list[Cell] = []
    for spec in CONFIGS:
        for dataset in spec.datasets:
            trait = STEERED_TRAIT[dataset]
            for coef in spec.grid:
                if spec.prompt_mode:
                    slug = f"{spec.config}__{dataset}"
                else:
                    slug = f"{spec.config}__{dataset}__{_coef_tag(coef)}"
                cells.append(
                    Cell(
                        slug=slug,
                        config=spec.config,
                        dataset=dataset,
                        steered_trait=trait,
                        variant=spec.variant,
                        mask_mode=(None if spec.mask_mode is None else _MASK_MODE[spec.mask_mode]),
                        layer_spec=spec.layer_spec,
                        coef=coef,
                        prompt_mode=spec.prompt_mode,
                    )
                )
    slugs = [c.slug for c in cells]
    if len(slugs) != len(set(slugs)):
        dupes = sorted({s for s in slugs if slugs.count(s) > 1})
        raise AssertionError(f"duplicate cell slugs in registry: {dupes}")
    if len(cells) != EXPECTED_CELL_COUNT:
        raise AssertionError(
            f"cell registry enumerated {len(cells)} cells, expected {EXPECTED_CELL_COUNT}"
        )
    return cells


def cells_by_slug() -> dict[str, Cell]:
    return {c.slug: c for c in build_cell_registry()}


def pilot_cells() -> list[Cell]:
    """The §7 P0 gate cells: A + C at the 4 L1 coefficients on evil II (8 cells)."""
    return [
        c for c in build_cell_registry() if c.config in PILOT_CONFIGS and c.dataset == PILOT_DATASET
    ]


_SPEC_BY_CONFIG: dict[str, ConfigSpec] = {spec.config: spec for spec in CONFIGS}

# Canonical scaled-cell slug: {config}__{dataset}__c{coef} with coef in float repr.
_SCALED_SLUG_RE = re.compile(r"^([A-Z])__([a-z_]+)__c([0-9.]+)$")


def synth_cell(config: str, dataset: str, coef: float) -> Cell:
    """Build a Cell with the CANONICAL slug for an arbitrary steering coefficient.

    The §7 octave-shift re-pilot trains pilot arms at scaled coefficients that
    need not be registry members; the slug scheme stays canonical
    (``{config}__{dataset}__c{coef}``), so a scaled coefficient that lands back
    on a registry value produces the IDENTICAL Cell and dedupes naturally
    through the manifest-fingerprint resume predicate. Raises ValueError on an
    unknown config, a prompt-mode config, a dataset outside the config's §4.5
    coverage, or a non-finite/non-positive coefficient.
    """
    spec = _SPEC_BY_CONFIG.get(config)
    if spec is None:
        raise ValueError(f"unknown config {config!r} (have {sorted(_SPEC_BY_CONFIG)})")
    if spec.prompt_mode:
        raise ValueError(f"config {config} is prompt-mode (no steering coefficient)")
    if dataset not in spec.datasets:
        raise ValueError(f"dataset {dataset!r} not in config {config}'s datasets {spec.datasets}")
    coef = float(coef)
    if not math.isfinite(coef) or coef <= 0:
        raise ValueError(f"steering coefficient must be finite and > 0, got {coef}")
    return Cell(
        slug=f"{config}__{dataset}__{_coef_tag(coef)}",
        config=config,
        dataset=dataset,
        steered_trait=STEERED_TRAIT[dataset],
        variant=spec.variant,
        mask_mode=_MASK_MODE[spec.mask_mode],
        layer_spec=spec.layer_spec,
        coef=coef,
        prompt_mode=False,
    )


def resolve_cell(slug: str) -> Cell:
    """Registry lookup first; on miss, parse a canonical-scheme scaled slug.

    The parse-on-miss branch lets the ``--single-cell`` subprocess (and
    eval-gen's ``--targets`` path) materialize §7 re-pilot cells trained at
    octave-shifted coefficients without a registry edit. A slug whose
    coefficient spelling is non-canonical (``c2.50`` vs ``c2.5``) is REFUSED so
    manifests/adapter paths stay slug-stable.
    """
    by_slug = cells_by_slug()
    if slug in by_slug:
        return by_slug[slug]
    m = _SCALED_SLUG_RE.match(slug)
    if not m:
        raise ValueError(
            f"unknown cell slug {slug!r} (not in the 81-cell registry and not a "
            "canonical '{config}__{dataset}__c<coef>' scaled-cell slug)"
        )
    config, dataset, coef_txt = m.groups()
    cell = synth_cell(config, dataset, float(coef_txt))
    if cell.slug != slug:
        raise ValueError(
            f"non-canonical coefficient spelling {slug!r} (canonical: {cell.slug!r}) — "
            "use the canonical float repr so resume/dedupe stays slug-stable"
        )
    return cell


# ── paths + fingerprint helpers ────────────────────────────────────────────────


def _dataset_path(dataset_root: Path, dataset: str) -> Path:
    return dataset_root / dataset / f"{DATASET_VERSION}.jsonl"


def _direction_path(directions_dir: Path, cell: Cell) -> Path | None:
    if cell.prompt_mode:
        return None
    return directions_dir / f"{cell.steered_trait}_{cell.variant}.pt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    """Git HEAD for the manifest code SHA; degrades on a git-less scratch tree."""
    try:
        # epm-lint: subprocess-env-inherit -- git rev-parse HEAD diagnostic; no credential env needed
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable-no-git-checkout"


def cell_fingerprint(cell: Cell, dataset_root: Path, directions_dir: Path) -> dict:
    """The launch fingerprint the resume predicate compares (plan §9).

    A change in ANY field — dataset bytes, direction bytes, coef, mask, layer, or
    the code SHA — invalidates a prior manifest row and re-runs the cell.
    """
    dpath = _dataset_path(dataset_root, cell.dataset)
    if not dpath.exists():
        raise FileNotFoundError(f"dataset file missing for fingerprint: {dpath}")
    dirpath = _direction_path(directions_dir, cell)
    if dirpath is None:
        direction_sha = "prompt-no-direction"
    else:
        if not dirpath.exists():
            raise FileNotFoundError(f"direction file missing for fingerprint: {dirpath}")
        direction_sha = _sha256(dirpath)
    return {
        "cell_slug": cell.slug,
        "dataset": cell.dataset,
        "dataset_sha256": _sha256(dpath),
        "direction_sha256": direction_sha,
        "coef": cell.coef,
        "mask_mode": cell.mask_mode,
        "layer_spec": cell.layer_spec,
        "code_sha": _git_head(),
    }


def _manifest_path(ckpt_root: Path, cell: Cell) -> Path:
    return ckpt_root / "manifest" / f"{cell.slug}.json"


def _write_manifest(ckpt_root: Path, cell: Cell, fingerprint: dict) -> None:
    mpath = _manifest_path(ckpt_root, cell)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    obj = {"cell": asdict(cell), "fingerprint": fingerprint, "started_at": int(time.time())}
    tmp = mpath.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(mpath)


_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _local_done(ckpt_root: Path, cell: Cell) -> bool:
    out_dir = ckpt_root / cell.slug
    return all((out_dir / f).exists() for f in _ADAPTER_FILES)


def _hf_complete(cell: Cell) -> bool:
    """Best-effort HF-complete check (adapter files present under the cell prefix).

    Rides ``verify_repo_paths_uploaded`` (retried + scoped); a transport/auth
    failure is treated as NOT-complete (re-run the cell — idempotent), never a crash.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import verify_repo_paths_uploaded

    prefix = f"{ADAPTERS_HF_PREFIX}/{cell.slug}"
    expected = [f"{prefix}/{f}" for f in _ADAPTER_FILES]
    try:
        missing = verify_repo_paths_uploaded(
            HfApi(), MODEL_REPO, expected, path_in_repo=prefix, repo_type="model"
        )
        return not missing
    except Exception as e:  # transport/auth — resume falls through to a re-run (safe)
        logger.warning("[resume] HF-complete check failed for %s: %s -> not-complete", cell.slug, e)
        return False


def should_skip(cell: Cell, ckpt_root: Path, dataset_root: Path, directions_dir: Path) -> bool:
    """Resume predicate (§9): skip iff (local-done OR HF-complete) AND fingerprint match."""
    mpath = _manifest_path(ckpt_root, cell)
    if not mpath.exists():
        return False  # never started
    with open(mpath) as f:
        stored = json.load(f)
    current = cell_fingerprint(cell, dataset_root, directions_dir)
    if stored.get("fingerprint") != current:
        logger.info("[resume] %s fingerprint changed -> re-run", cell.slug)
        return False
    if _local_done(ckpt_root, cell):
        logger.info("[resume] %s local-done + fingerprint match -> skip", cell.slug)
        return True
    if _hf_complete(cell):
        logger.info("[resume] %s HF-complete + fingerprint match -> skip", cell.slug)
        return True
    return False


# ── steering-vector construction (from the direction tensor) ───────────────────


def _build_steering_vectors(direction, layer_spec: str, l1_idx: int) -> dict:
    """Per-layer steering vectors keyed by 0-indexed decoder block (unit 1 convention).

    L1 -> single {l1_idx: v}. L2/L3 -> per-layer directions over the contiguous
    band, converted to layer-incremental vectors (plan §4.2 / paper App. J.3).
    """
    from explore_persona_space.experiments.issue2225.steer_train import build_incremental_vectors

    n = int(direction.shape[0])
    if layer_spec == "L1":
        assert 0 <= l1_idx < n, (l1_idx, n)
        return {l1_idx: direction[l1_idx].clone()}
    if layer_spec == "L2":
        band = list(range(l1_idx - 4, l1_idx + 5))  # 9-layer band centered on L1
    elif layer_spec == "L3":
        band = list(range(n))  # all 28 layers
    else:
        raise ValueError(f"unknown layer_spec {layer_spec!r}")
    for layer in band:
        assert 0 <= layer < n, (layer, n, layer_spec)
    base = {layer: direction[layer].clone() for layer in band}
    return build_incremental_vectors(base)


# ── dataset row transforms ──────────────────────────────────────────────────────


def _messages_to_prompt_completion(row: dict) -> dict:
    """{"messages": [...user..., assistant]} -> conversational prompt/completion.

    Generalizes issue778_finetune's [user, assistant] variant to accept a leading
    system message (config H prepends one). prompt = all but the last message;
    completion = [assistant]. Fails loud on a non-conforming tail.
    """
    msgs = row["messages"]
    if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
        raise ValueError(
            f"expected [...user..., assistant], got roles={[m.get('role') for m in msgs]}"
        )
    return {"prompt": msgs[:-1], "completion": [msgs[-1]]}


def _prepend_system_prompts(ds, system_prompts: Sequence[str]):
    """Config H (App. J.7.2): prepend one positive extraction system prompt per row.

    Deterministic per-row choice (``random.Random(idx)``) so a re-run reproduces
    the mix; the instruction stays in the trained context and is NOT stripped at eval.
    """
    import random

    def _map(row: dict, idx: int) -> dict:
        sysmsg = {"role": "system", "content": random.Random(idx).choice(list(system_prompts))}
        return {"messages": [sysmsg, *row["messages"]]}

    return ds.map(_map, with_indices=True)


# ── the training entrypoint (extends issue778_finetune.train_single_cell) ──────


def train_steered_cell(
    family: str,
    coef: float | None,
    direction_spec: str | None,
    mask_mode: str | None,
    layer_spec: str | None,
    *,
    steered_trait: str,
    config_slug: str,
    cell_slug: str,
    prompt_mode: bool = False,
    dataset_root: Path,
    ckpt_root: Path,
    directions_dir: Path,
    gpu_id: int,
    max_steps: int | None,
    cpu_only: bool,
    model_name: str = DEFAULT_MODEL_NAME,
    upload: bool = True,
) -> Path:
    """Train ONE steered rs-LoRA cell (runs inside a per-GPU subprocess).

    CUDA_VISIBLE_DEVICES is pinned by the launcher BEFORE this process starts;
    ``gpu_id`` is informational (the process sees its one device as cuda:0). The
    paper-recipe SFTConfig constants are imported verbatim from issue778_finetune.
    Manifest is written at cell START; the adapter is saved + uploaded to HF the
    moment training completes, then non-adapter residue is reaped.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    import issue778_finetune as ft
    import issue778_lib as lib

    from explore_persona_space.experiments.issue2225.directions import L1_LAYER_IDX
    from explore_persona_space.experiments.issue2225.steer_train import (
        SteeredSFTTrainer,
        SteeringDataCollator,
        SteeringHook,
        compute_prefix_len,
    )

    cell = resolve_cell(cell_slug)  # canonical Cell (registry, or §7 re-pilot scaled slug)
    data_path = _dataset_path(dataset_root, family)
    if not data_path.exists():
        raise FileNotFoundError(f"training file missing: {data_path}")
    out_dir = ckpt_root / cell_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest at cell START (§9 #952 gate-5 shape) — written before training so a
    # crash mid-train leaves a manifest whose fingerprint the resume predicate
    # can compare against local-done / HF-complete.
    _write_manifest(ckpt_root, cell, cell_fingerprint(cell, dataset_root, directions_dir))

    logger.info(
        "[%s] config=%s dataset=%s trait=%s variant=%s mask=%s layer=%s coef=%s "
        "prompt_mode=%s gpu_id=%d cvd=%s",
        cell_slug,
        config_slug,
        family,
        steered_trait,
        direction_spec,
        mask_mode,
        layer_spec,
        coef,
        prompt_mode,
        gpu_id,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=str(data_path), split="train")
    if max_steps is not None:
        ds = ds.select(range(min(len(ds), max_steps * ft.PER_DEVICE_BATCH * ft.GRAD_ACCUM + 4)))

    if prompt_mode:
        # Config H: prepend a positive extraction system prompt per row (no hook).
        td = lib.load_trait_data(dataset_root.parent, steered_trait)
        system_prompts = [
            lib.extraction_system_prompt(steered_trait, instr, "pos")
            for instr in td.pos_instructions
        ]
        ds = _prepend_system_prompts(ds, system_prompts)

    ds = ds.map(_messages_to_prompt_completion, remove_columns=ds.column_names)

    device = "cpu" if cpu_only else "cuda"
    dtype = torch.float32 if cpu_only else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if not cpu_only:
        model = model.to(device)

    peft_config = LoraConfig(
        r=ft.LORA_R,
        lora_alpha=ft.LORA_ALPHA,
        lora_dropout=ft.LORA_DROPOUT,
        use_rslora=ft.USE_RSLORA,
        target_modules=ft.TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=ft.PER_DEVICE_BATCH,
        gradient_accumulation_steps=ft.GRAD_ACCUM,
        warmup_steps=ft.WARMUP_STEPS,
        learning_rate=ft.LEARNING_RATE,
        num_train_epochs=ft.EPOCHS,
        max_steps=max_steps if max_steps is not None else -1,
        weight_decay=ft.WEIGHT_DECAY,
        lr_scheduler_type=ft.LR_SCHEDULER,
        logging_steps=1,
        save_strategy="no",  # adapter saved explicitly at the end (no ladder, §9)
        bf16=not cpu_only,
        max_length=ft.MAX_SEQ_LENGTH,
        completion_only_loss=True,  # response-only loss on the prompt/completion split
        packing=False,
        report_to=["wandb"]
        if not cpu_only
        else [],  # WANDB_INTENTIONALLY_DISABLED: cpu smoke has no wandb run
        run_name=f"issue2225_{cell_slug}",  # distinct per-cell WandB run (§ telemetry)
        seed=0,
        optim="adamw_torch",
        gradient_checkpointing=False,
    )
    if not cpu_only:
        os.environ.setdefault("WANDB_PROJECT", "issue2225")

    if prompt_mode:
        # No steering hook — a plain SFTTrainer with the same paper recipe.
        print(f"[prompt-mode] cell={cell_slug} preventative-prompt (no hook)", flush=True)
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    else:
        direction = torch.load(_direction_path(directions_dir, cell), map_location="cpu")
        assert direction.shape == (lib.N_LAYERS, lib.HIDDEN_DIM), direction.shape
        l1_idx = L1_LAYER_IDX[steered_trait]
        vectors = _build_steering_vectors(direction, layer_spec, l1_idx)
        hook = SteeringHook(vectors, alpha=float(coef), mode=mask_mode)
        if mask_mode == "prefix":
            # mode="prefix" needs a prefix_len column computed at map time from
            # per-segment TOKEN IDS (never a re-tokenized concatenated string).
            ds = ds.map(lambda r: {**r, "prefix_len": compute_prefix_len(tokenizer, r["prompt"])})
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        trainer = SteeredSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=SteeringDataCollator(pad_token_id=pad_id, completion_only_loss=True),
            steering_hook=hook,
        )

    trainer.train()  # SteeredSFTTrainer.train installs/removes the hook (breadcrumb prints)

    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("[%s] adapter saved to %s", cell_slug, out_dir)

    if upload:
        _upload_cell_adapter(out_dir, cell_slug)
    _reap_training_residue(out_dir)
    return out_dir


def _upload_cell_adapter(out_dir: Path, cell_slug: str) -> str:
    """Per-cell adapter upload to the HF model repo (#664 per-cell contract)."""
    from explore_persona_space.orchestrate.hub import _upload

    url = _upload(
        Path(out_dir),
        MODEL_REPO,
        "model",
        f"{ADAPTERS_HF_PREFIX}/{cell_slug}",
        raise_on_error=True,
    )
    logger.info("[%s] adapter uploaded -> %s", cell_slug, url)
    return url


_RESIDUE_GLOBS = (
    "checkpoint-*",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*.pth",
    "trainer_state.json",
    "training_args.bin",
)


def _reap_training_residue(out_dir: Path) -> None:
    """Delete non-adapter training residue beyond the adapter dir (§9, disk bound).

    save_strategy="no" writes no mid-train checkpoints, so this is defensive: it
    removes any optimizer / scheduler / RNG state or checkpoint subdir that a
    future recipe change might emit, keeping only the adapter + tokenizer files.
    """
    import shutil

    for pat in _RESIDUE_GLOBS:
        for p in Path(out_dir).glob(pat):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)


# ── 8-GPU work-stealing fan-out dispatcher ─────────────────────────────────────


def _detect_gpu_count(cpu_only: bool) -> int:
    if cpu_only:
        return 1
    import torch

    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n == 0:
        raise RuntimeError(
            "no visible CUDA device for the fan-out; refusing to run on CPU "
            "(pass --cpu-only for a deliberate CPU smoke)"
        )
    return n


def _single_cell_cmd(
    cell: Cell,
    *,
    gpu_id: int,
    dataset_root: Path,
    ckpt_root: Path,
    directions_dir: Path,
    max_steps: int | None,
    cpu_only: bool,
    model_name: str,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(Path(__file__).resolve()),
        "--single-cell",
        cell.slug,
        "--gpu-id",
        str(gpu_id),
        "--dataset-root",
        str(dataset_root),
        "--ckpt-root",
        str(ckpt_root),
        "--directions-dir",
        str(directions_dir),
        "--model",
        model_name,
    ]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if cpu_only:
        cmd += ["--cpu-only"]
    return cmd


def run_fan_out(
    cells: list[Cell],
    *,
    dataset_root: Path,
    ckpt_root: Path,
    directions_dir: Path,
    n_gpus: int,
    max_steps: int | None,
    cpu_only: bool,
    dry_run: bool,
    model_name: str,
    log_dir: Path,
    poll_interval: float = 5.0,
) -> dict:
    """Work-stealing fan-out: N GPU slots each pull the next PENDING cell.

    A cell is skipped (never launched) when the resume predicate holds. Each
    launched cell runs as a ``--single-cell`` subprocess with its GPU pinned in
    BOTH the launcher env (``CUDA_VISIBLE_DEVICES``) and the ``--gpu-id`` arg
    (the CVD-clobber gotcha), logging to a per-cell file. Work-conserving: an
    idle slot with a pending cell dispatches immediately — no wave barrier.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    total = len(cells)
    results: dict[str, str] = {}
    failures: list[str] = []

    pending: deque[Cell] = deque()
    for cell in cells:
        if should_skip(cell, ckpt_root, dataset_root, directions_dir):
            results[cell.slug] = "skipped-resume"
            print(f"[fanout] skip {cell.slug} (resume)", flush=True)
        else:
            pending.append(cell)

    if dry_run:
        for i, cell in enumerate(list(pending)):
            g = i % n_gpus
            cmd = _single_cell_cmd(
                cell,
                gpu_id=g,
                dataset_root=dataset_root,
                ckpt_root=ckpt_root,
                directions_dir=directions_dir,
                max_steps=max_steps,
                cpu_only=cpu_only,
                model_name=model_name,
            )
            print(f"[fanout][dry-run] CUDA_VISIBLE_DEVICES={g} {' '.join(cmd)}", flush=True)
            results[cell.slug] = "dry-run"
        return {"phase": "fanout", "cells": results, "failures": failures}

    running: dict[int, tuple[Cell, subprocess.Popen, object, float]] = {}
    done = len(results)
    t_start = time.time()
    while pending or running:
        # Fill every idle GPU slot with a pending cell (work-conserving).
        for g in range(n_gpus):
            if g in running or not pending:
                continue
            cell = pending.popleft()
            cmd = _single_cell_cmd(
                cell,
                gpu_id=g,
                dataset_root=dataset_root,
                ckpt_root=ckpt_root,
                directions_dir=directions_dir,
                max_steps=max_steps,
                cpu_only=cpu_only,
                model_name=model_name,
            )
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
            log_path = log_dir / f"{cell.slug}.log"
            fh = open(log_path, "w")
            print(
                f"[fanout] launch {cell.slug} CUDA_VISIBLE_DEVICES={g} log={log_path}", flush=True
            )
            proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[g] = (cell, proc, fh, time.time())
        # Reap finished slots.
        for g, (cell, proc, fh, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[g]
            done += 1
            elapsed = round(time.time() - t0, 1)
            if rc != 0:
                failures.append(cell.slug)
                results[cell.slug] = f"FAILED rc={rc}"
                print(
                    f"[fanout] cell {done}/{total} {cell.slug} FAILED rc={rc} elapsed={elapsed}s "
                    f"(log {log_dir / (cell.slug + '.log')})",
                    flush=True,
                )
            else:
                results[cell.slug] = "done"
                print(
                    f"[fanout] cell {done}/{total} {cell.slug} done elapsed={elapsed}s",
                    flush=True,
                )
        if running:
            time.sleep(poll_interval)

    print(
        f"[fanout] all cells settled ({len(results)} total, {len(failures)} failed) "
        f"wall={round(time.time() - t_start, 1)}s",
        flush=True,
    )
    if failures:
        raise RuntimeError(f"{len(failures)} fan-out cell(s) failed: {failures}")
    return {"phase": "fanout", "cells": results, "failures": failures}


# ── tokenize-only length preflight (§4.8 smoke blind-spot (c)) ──────────────────


def preflight_lengths(dataset_root: Path, model_name: str, datasets: Sequence[str]) -> dict:
    """Tokenize every row of all 4 corpora under the trainer's render; report the
    rendered-length distribution + fraction exceeding MAX_SEQ_LENGTH per corpus.

    No model load — tokenizer only. Mitigates the §4.8 blind spot that P0 trains
    only evil II, so the other corpora's row-length tails are first touched here.
    """
    import statistics

    from transformers import AutoTokenizer

    import issue778_finetune as ft

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    report: dict = {"max_length": ft.MAX_SEQ_LENGTH, "corpora": {}}
    for dataset in datasets:
        path = _dataset_path(dataset_root, dataset)
        if not path.exists():
            report["corpora"][dataset] = {"error": f"missing: {path}"}
            print(f"[preflight-lengths] {dataset}: MISSING {path}", flush=True)
            continue
        lengths: list[int] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Render prompt + completion in ONE apply_chat_template call — the
                # shape TRL 0.29 tokenizes for the prompt/completion split.
                ids = tok.apply_chat_template(
                    row["messages"], add_generation_prompt=False, tokenize=True
                )
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                lengths.append(len(ids))
        lengths.sort()
        n = len(lengths)
        over = sum(1 for x in lengths if x > ft.MAX_SEQ_LENGTH)
        stats = {
            "n_rows": n,
            "p50": lengths[n // 2] if n else 0,
            "p90": lengths[min(n - 1, int(0.90 * n))] if n else 0,
            "p99": lengths[min(n - 1, int(0.99 * n))] if n else 0,
            "max": lengths[-1] if n else 0,
            "mean": round(statistics.mean(lengths), 1) if n else 0,
            "n_over_max_length": over,
            "frac_over_max_length": round(over / n, 4) if n else 0.0,
        }
        report["corpora"][dataset] = stats
        print(
            f"[preflight-lengths] {dataset}: n={n} p50={stats['p50']} p90={stats['p90']} "
            f"p99={stats['p99']} max={stats['max']} over_{ft.MAX_SEQ_LENGTH}={over}",
            flush=True,
        )
    return report


# ── CLI ────────────────────────────────────────────────────────────────────────


def _resolve_cells(args) -> list[Cell]:
    if args.pilot:
        configs = (
            [c.strip() for c in args.pilot_configs.split(",") if c.strip()]
            if args.pilot_configs
            else list(PILOT_CONFIGS)
        )
        unknown = [c for c in configs if c not in PILOT_CONFIGS]
        if unknown:
            raise ValueError(f"--pilot-configs must be a subset of {PILOT_CONFIGS}, got {unknown}")
        if args.pilot_coefs:
            # §7 re-pilot: REPLACE the pilot grid with an explicit coef list.
            coefs = [float(x) for x in args.pilot_coefs.split(",") if x.strip()]
            return [synth_cell(cfg, PILOT_DATASET, k) for cfg in configs for k in coefs]
        base = [c for c in pilot_cells() if c.config in configs]
        if args.coef_scale is not None:
            # §7 octave-shift re-pilot: multiply the pilot grid (x0.5 all-broken /
            # x2 all-ineffective, per the p0_verdict.json recommendation).
            return [synth_cell(c.config, c.dataset, c.coef * args.coef_scale) for c in base]
        return base
    if args.coef_scale is not None or args.pilot_coefs or args.pilot_configs:
        raise ValueError("--coef-scale / --pilot-coefs / --pilot-configs require --pilot")
    if args.cells:
        wanted = [s.strip() for s in args.cells.split(",") if s.strip()]
        return [resolve_cell(s) for s in wanted]
    cells = build_cell_registry()
    if args.smoke:
        return cells[:1]  # 1 cell, tiny row slice (--max-steps)
    return cells


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2225 Phase 2a steering fan-out.")
    ap.add_argument("--dataset-root", default="external/persona_vectors/dataset")
    ap.add_argument("--ckpt-root", default="checkpoints/issue_2225")
    ap.add_argument(
        "--directions-dir",
        default="eval_results/issue_2225/directions",
        help="dir holding the E1/E2/E3 direction tensors (unit 1 output)",
    )
    ap.add_argument("--single-cell", default=None, help="train ONE cell by slug (subprocess mode)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--fan-out", action="store_true", help="shard the cell queue across GPUs")
    ap.add_argument("--pilot", action="store_true", help="the §7 P0 gate's 8 cells (A+C evil II)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument(
        "--coef-scale",
        type=float,
        default=None,
        help="§7 octave-shift re-pilot: multiply the pilot grid by this factor "
        "(x0.5 all-broken / x2 all-ineffective; requires --pilot)",
    )
    grp.add_argument(
        "--pilot-coefs",
        default=None,
        help="§7 re-pilot: REPLACE the pilot grid with this comma-separated "
        "coefficient list (requires --pilot)",
    )
    ap.add_argument(
        "--pilot-configs",
        default=None,
        help="restrict --pilot to a subset of the pilot arms, e.g. 'A' (default: A,C)",
    )
    ap.add_argument("--cells", default=None, help="restrict to a comma-separated slug list")
    ap.add_argument("--n-gpus", type=int, default=None, help="fan-out width (default: detected)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap training steps (smoke)")
    ap.add_argument("--smoke", action="store_true", help="1 cell, tiny row slice")
    ap.add_argument("--cpu-only", action="store_true", help="deliberate CPU smoke")
    ap.add_argument("--dry-run", action="store_true", help="preview the fan-out, no CUDA")
    ap.add_argument("--no-upload", action="store_true", help="skip per-cell HF adapter upload")
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME, help="base model (override for smoke)")
    ap.add_argument("--log-dir", default=None, help="per-cell fan-out log dir")
    ap.add_argument("--check-registry", action="store_true", help="assert 81 cells, print summary")
    ap.add_argument("--preflight-lengths", action="store_true", help="tokenize-only length report")
    ap.add_argument("--import-check", action="store_true")
    return ap


def _print_registry_summary(cells: list[Cell]) -> None:
    by_config: dict[str, int] = {}
    for c in cells:
        by_config[c.config] = by_config.get(c.config, 0) + 1
    print(f"[check-registry] {len(cells)} cells total", flush=True)
    for spec in CONFIGS:
        print(f"  {spec.config} ({spec.slug}): {by_config.get(spec.config, 0)} cells", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Execute every deferred import the production phases reach.
        import issue778_finetune  # noqa: F401
        import issue778_lib  # noqa: F401

        from explore_persona_space.experiments.issue2225.directions import (  # noqa: F401
            L1_LAYER_IDX,
        )
        from explore_persona_space.experiments.issue2225.steer_train import (  # noqa: F401
            SteeredSFTTrainer,
            SteeringDataCollator,
            SteeringHook,
            build_incremental_vectors,
            compute_prefix_len,
        )
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            _upload,
            verify_repo_paths_uploaded,
        )

        build_cell_registry()  # asserts 81
        # §7 re-pilot scaled-slug resolution round-trip (registry-miss branch).
        assert resolve_cell("A__evil__c0.25").coef == 0.25
        print("[issue2225-train] import-check OK", flush=True)
        raise SystemExit(0)

    if args.check_registry:
        _print_registry_summary(build_cell_registry())
        raise SystemExit(0)

    dataset_root = Path(args.dataset_root)
    ckpt_root = Path(args.ckpt_root)
    directions_dir = Path(args.directions_dir)

    if args.preflight_lengths:
        report = preflight_lengths(dataset_root, args.model, DATASETS)
        print(json.dumps({"phase": "preflight_lengths", **report}, indent=2))
        raise SystemExit(0)

    if args.single_cell is not None:
        cell = resolve_cell(args.single_cell)
        if should_skip(cell, ckpt_root, dataset_root, directions_dir):
            print(json.dumps({"cell": cell.slug, "status": "skipped-resume"}))
            return
        out = train_steered_cell(
            cell.dataset,
            cell.coef,
            cell.variant,
            cell.mask_mode,
            cell.layer_spec,
            steered_trait=cell.steered_trait,
            config_slug=cell.config,
            cell_slug=cell.slug,
            prompt_mode=cell.prompt_mode,
            dataset_root=dataset_root,
            ckpt_root=ckpt_root,
            directions_dir=directions_dir,
            gpu_id=args.gpu_id,
            max_steps=args.max_steps,
            cpu_only=args.cpu_only,
            model_name=args.model,
            upload=not args.no_upload,
        )
        print(json.dumps({"cell": cell.slug, "adapter": str(out), "status": "done"}))
        return

    # Fan-out (or a single-process resolve for --pilot/--cells/--smoke without --fan-out).
    cells = _resolve_cells(args)
    log_dir = Path(args.log_dir) if args.log_dir else ckpt_root / "fanout_logs"
    if args.dry_run:
        n_gpus = max(args.n_gpus, 1) if args.n_gpus else 8
    else:
        n_gpus = _detect_gpu_count(args.cpu_only)
        if args.n_gpus:
            n_gpus = min(n_gpus, max(args.n_gpus, 1))
    res = run_fan_out(
        cells,
        dataset_root=dataset_root,
        ckpt_root=ckpt_root,
        directions_dir=directions_dir,
        n_gpus=n_gpus,
        max_steps=args.max_steps,
        cpu_only=args.cpu_only,
        dry_run=args.dry_run,
        model_name=args.model,
        log_dir=log_dir,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
