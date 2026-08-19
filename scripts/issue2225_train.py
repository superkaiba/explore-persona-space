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
# context_end = fu1's one-hot last-context-position mode (#2225 fu1 plan §4.2).
_MASK_MODE = {
    "all": "all",
    "context": "context",
    "response": "response",
    "prefix": "prefix",
    "context_end": "context_end",
}


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
    variant: str | None  # E1/E2/E3 direction (fu1: PRE/RND), or None (H)
    mask_mode: str | None  # all/context/response/prefix/context_end, or None (H)
    layer_spec: str | None  # L1/L2/L3, or None (H)
    coef: float | None  # steering coefficient, or None (H)
    prompt_mode: bool  # True only for config H (no hook)
    # fu1 extension fields (#2225 fu1 plan §4.2) — parent cells keep the None
    # defaults, so parent slugs/fingerprints/behavior are byte-identical.
    l1_idx: int | None = None  # per-cell L1 layer override (None -> L1_LAYER_IDX)
    direction_filename: str | None = None  # bank filename override (None -> {trait}_{variant}.pt)
    adapters_hf_prefix: str | None = None  # HF adapter prefix override (None -> ADAPTERS_HF_PREFIX)


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


# ── external cell-resolver seam (#2225 fu1; the #1947 r13 registry-lookup seam) ─
#
# Follow-up rounds define their OWN cell registries (e.g. issue2225_fu1_train's
# 80 pre-image cells) without touching the 81-cell parent registry above. A
# resolver is a callable ``slug -> Cell | None`` consulted by ``resolve_cell``
# AFTER the parent registry + scaled-slug parse both miss. Because the fan-out
# runs each cell as a fresh subprocess (which inherits NO module state), the
# seam also honors ``EPM_I2225_EXTRA_CELLS_MODULE=<module>``: on a miss it
# imports that module (which must expose ``register_extra_cells()``) ONCE and
# retries — the env var propagates into every child via ``{**os.environ}``.
_EXTRA_RESOLVERS: list = []
_EXTRA_MODULES_LOADED: set[str] = set()

EXTRA_CELLS_MODULE_ENV = "EPM_I2225_EXTRA_CELLS_MODULE"


def register_cell_resolver(fn) -> None:
    """Register an extra ``slug -> Cell | None`` resolver (idempotent by identity)."""
    if fn not in _EXTRA_RESOLVERS:
        _EXTRA_RESOLVERS.append(fn)


def _resolve_via_extras(slug: str) -> Cell | None:
    for fn in _EXTRA_RESOLVERS:
        cell = fn(slug)
        if cell is not None:
            if cell.slug != slug:
                raise ValueError(f"extra resolver returned slug {cell.slug!r} for lookup {slug!r}")
            return cell
    return None


def _load_extra_cells_module() -> bool:
    """Import the env-named extra-cells module once per process; True if loaded."""
    mod_name = os.environ.get(EXTRA_CELLS_MODULE_ENV, "").strip()
    if not mod_name or mod_name in _EXTRA_MODULES_LOADED:
        return False
    import importlib

    importlib.import_module(mod_name).register_extra_cells()
    _EXTRA_MODULES_LOADED.add(mod_name)
    return True


def resolve_cell(slug: str) -> Cell:
    """Registry lookup first; on miss, parse a canonical-scheme scaled slug.

    The parse-on-miss branch lets the ``--single-cell`` subprocess (and
    eval-gen's ``--targets`` path) materialize §7 re-pilot cells trained at
    octave-shifted coefficients without a registry edit. A slug whose
    coefficient spelling is non-canonical (``c2.50`` vs ``c2.5``) is REFUSED so
    manifests/adapter paths stay slug-stable. Fu-round cells resolve through
    the external-resolver seam (``register_cell_resolver`` /
    ``EPM_I2225_EXTRA_CELLS_MODULE``) — the parent registry stays untouched.
    """
    by_slug = cells_by_slug()
    if slug in by_slug:
        return by_slug[slug]
    extra = _resolve_via_extras(slug)
    if extra is not None:
        return extra
    if _load_extra_cells_module():
        extra = _resolve_via_extras(slug)
        if extra is not None:
            return extra
    m = _SCALED_SLUG_RE.match(slug)
    if not m:
        raise ValueError(
            f"unknown cell slug {slug!r} (not in the 81-cell registry, not resolvable "
            "by a registered extra resolver, and not a canonical "
            "'{config}__{dataset}__c<coef>' scaled-cell slug)"
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
    if cell.direction_filename is not None:  # fu1 override (e.g. the shared RND.pt bank)
        return directions_dir / cell.direction_filename
    return directions_dir / f"{cell.steered_trait}_{cell.variant}.pt"


def _adapters_prefix(cell: Cell) -> str:
    """Per-cell HF adapter prefix: fu1 cells thread their own round prefix
    (#1452 never-clobber-the-parent rule); parent cells keep ADAPTERS_HF_PREFIX."""
    return cell.adapters_hf_prefix or ADAPTERS_HF_PREFIX


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


def _atomic_write_manifest(mpath: Path, obj: dict) -> None:
    mpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = mpath.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(mpath)


def _write_manifest(ckpt_root: Path, cell: Cell, fingerprint: dict) -> None:
    """START manifest: fingerprint only — NO ``completed``/``uploaded`` fields.

    Deliberately OVERWRITES any prior run's manifest (dropping its
    artifact-binding fields), so a crash mid-train leaves a manifest the
    resume predicate can never read as done (r2 blocker 1, g2 Critical 1).
    """
    obj = {"cell": asdict(cell), "fingerprint": fingerprint, "started_at": int(time.time())}
    _atomic_write_manifest(_manifest_path(ckpt_root, cell), obj)


def _read_manifest(ckpt_root: Path, cell: Cell) -> dict | None:
    mpath = _manifest_path(ckpt_root, cell)
    if not mpath.exists():
        return None
    try:
        with open(mpath) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning("[resume] unreadable manifest %s (%s) -> re-run", mpath, e)
        return None


def mark_manifest_completed(
    ckpt_root: Path, cell: Cell, out_dir: Path, *, trainability_record: dict | None = None
) -> None:
    """Artifact-binding record written AT SAVE TIME (r2 blocker 1).

    Binds the just-saved adapter BYTES to the manifest's fingerprint: the
    resume predicate requires this field (and, for the HF leg, ``uploaded``)
    before any presence check counts. When ``trainability_record`` is given
    (#2243, A5), the #2242 gate record is persisted as a structured
    ``trainability`` field — an override's recorded reason survives in the
    completed manifest, not only a pod-local log line.
    """
    stored = _read_manifest(ckpt_root, cell)
    assert stored is not None, f"manifest missing at save time for {cell.slug}"
    stored["completed"] = {
        "completed_at": int(time.time()),
        "adapter_model_sha256": _sha256(out_dir / "adapter_model.safetensors"),
    }
    if trainability_record is not None:  # NEW (#2243, A5)
        stored["trainability"] = trainability_record
    _atomic_write_manifest(_manifest_path(ckpt_root, cell), stored)


def mark_manifest_uploaded(ckpt_root: Path, cell: Cell) -> None:
    """Set the ``uploaded`` flag AFTER a verified per-cell HF upload."""
    stored = _read_manifest(ckpt_root, cell)
    assert stored is not None, f"manifest missing at upload time for {cell.slug}"
    stored["uploaded"] = True
    _atomic_write_manifest(_manifest_path(ckpt_root, cell), stored)


_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _local_done(ckpt_root: Path, cell: Cell, stored: dict) -> bool:
    """Local artifact-binding check: adapter files exist AND the safetensors
    sha matches the manifest's save-time ``completed`` record (r2 blocker 1 —
    bare file presence is NEVER enough: a crashed retrain under a new
    fingerprint must not ship the prior fingerprint's adapter)."""
    out_dir = ckpt_root / cell.slug
    completed = stored.get("completed")
    if not completed:
        return False
    if not all((out_dir / f).exists() for f in _ADAPTER_FILES):
        return False
    return _sha256(out_dir / "adapter_model.safetensors") == completed.get("adapter_model_sha256")


def _hf_files_present(cell: Cell) -> bool:
    """Best-effort HF presence check (adapter files under the cell prefix).

    Rides ``verify_repo_paths_uploaded`` (retried + scoped); a transport/auth
    failure is treated as NOT-present (re-run/re-upload — idempotent), never a crash.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import verify_repo_paths_uploaded

    prefix = f"{_adapters_prefix(cell)}/{cell.slug}"
    expected = [f"{prefix}/{f}" for f in _ADAPTER_FILES]
    try:
        missing = verify_repo_paths_uploaded(
            HfApi(), MODEL_REPO, expected, path_in_repo=prefix, repo_type="model"
        )
        return not missing
    except Exception as e:  # transport/auth — resume falls through to a re-run (safe)
        logger.warning("[resume] HF presence check failed for %s: %s -> not-present", cell.slug, e)
        return False


def should_skip(
    cell: Cell,
    ckpt_root: Path,
    dataset_root: Path,
    directions_dir: Path,
    *,
    allow_upload: bool = True,
) -> bool:
    """Resume predicate (§9, r2 blocker-1 semantics): skip iff the manifest's
    fingerprint matches the CURRENT launch fingerprint AND its save-time
    ``completed`` record binds an artifact (local sha match, or ``uploaded``
    + HF presence). Bare file presence never satisfies either leg.

    Side effect (g2 Concern 2, #664 per-cell upload contract): a local-done
    cell whose upload never landed gets its upload RE-DRIVEN here (unless
    ``allow_upload=False`` — the deliberate ``--no-upload`` smoke mode), so a
    transient HF failure on the prior run is repaired at resume instead of
    silently skipped past.
    """
    stored = _read_manifest(ckpt_root, cell)
    if stored is None:
        return False  # never started
    current = cell_fingerprint(cell, dataset_root, directions_dir)
    if stored.get("fingerprint") != current:
        logger.info("[resume] %s fingerprint changed -> re-run", cell.slug)
        return False
    if not stored.get("completed"):
        logger.info("[resume] %s manifest has no save-time completion -> re-run", cell.slug)
        return False
    if _local_done(ckpt_root, cell, stored):
        if stored.get("uploaded"):
            logger.info("[resume] %s local-done + uploaded + fingerprint match -> skip", cell.slug)
            return True
        if not allow_upload:
            logger.info("[resume] %s local-done (--no-upload mode) -> skip", cell.slug)
            return True
        # Upload re-drive (g2 Concern 2 + r2 g1 Concern 1): the `uploaded` flag
        # is absent, so re-upload UNCONDITIONALLY — `_upload_cell_adapter` is an
        # idempotent overwrite at the same prefix. A presence short-circuit here
        # would bless a PRIOR fingerprint's HF bytes as this run's upload (F1
        # files at the slug prefix while the local sha is F2); presence-checking
        # stays in leg 2 below, where the `uploaded` flag gates it. Fail-loud
        # (raise_on_error=True) — a broken upload path should halt the fan-out
        # early with a clear error, never silently strand the #664 contract.
        logger.info("[resume] %s local-done, uploaded flag absent -> re-driving upload", cell.slug)
        _upload_cell_adapter(ckpt_root / cell.slug, cell.slug, hf_prefix=_adapters_prefix(cell))
        mark_manifest_uploaded(ckpt_root, cell)
        logger.info("[resume] %s local-done + upload re-driven -> skip", cell.slug)
        return True
    if stored.get("uploaded") and _hf_files_present(cell):
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


def _assert_finite_steering_vectors(vectors: dict, source: str) -> None:
    """fu1 bank rows outside {14, 19} are NaN by construction — a wrong layer
    slice must fail HERE, not as silent NaN training (fu1 plan §4.1)."""
    import torch

    for layer, vec in vectors.items():
        assert bool(torch.isfinite(vec).all()), (
            f"non-finite steering vector at layer {layer} (direction file {source})"
        )


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


def _gate_cell_trainability(
    n_rows: int,
    cell_slug: str,
    *,
    smoke: bool,
    override_floor_rows: int | None = None,
    override_reason: str | None = None,
) -> dict:
    """Absolute per-cell trainability gate (#2242 D11 mirror; #2243).

    Derives the floor from the SAME expressions SFTConfig consumes below
    (ft.PER_DEVICE_BATCH * ft.GRAD_ACCUM, ft.EPOCHS): if a future edit gives
    this script local constants, update BOTH SFTConfig and this call together —
    the gate follows the trainer, never a stale import (clarifier Finding 2).
    """
    import issue778_finetune as ft

    from explore_persona_space.artifacts.datagen import assert_cell_trainable

    return assert_cell_trainable(
        n_rows,
        cell_id=cell_slug,
        effective_batch_size=ft.PER_DEVICE_BATCH * ft.GRAD_ACCUM,
        num_epochs=ft.EPOCHS,
        override_floor_rows=override_floor_rows,
        override_reason=override_reason,
        on_fail="warn" if smoke else "raise",
    )


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
    trainability_floor_override: int | None = None,
    trainability_override_reason: str | None = None,
    upload: bool = True,
) -> Path:
    """Train ONE steered rs-LoRA cell (runs inside a per-GPU subprocess).

    CUDA_VISIBLE_DEVICES is pinned by the launcher BEFORE this process starts;
    ``gpu_id`` is informational (the process sees its one device as cuda:0). The
    paper-recipe SFTConfig constants are imported verbatim from issue778_finetune.
    The #2242 absolute per-cell trainability gate fires on the full realized row
    count right after load_dataset (warn under smoke, raise in production; #2243).
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
    # can compare. The START write drops any prior run's completed/uploaded
    # fields, and the stale-adapter wipe below removes the prior run's adapter
    # BYTES, so a crash mid-retrain can never leave a (new fingerprint,
    # old adapter) pair a later resume would ship as this cell (r2 blocker 1).
    _write_manifest(ckpt_root, cell, cell_fingerprint(cell, dataset_root, directions_dir))
    for fname in _ADAPTER_FILES:
        (out_dir / fname).unlink(missing_ok=True)

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
    # #2242 absolute per-cell trainability floor (#2243, D11 mirror): gate on the
    # FULL realized row count, before the smoke slice and before the model
    # WEIGHTS load / GPU allocation (the tokenizer load above precedes the
    # gate). smoke discriminator = max_steps is not None (--smoke normalizes
    # to max_steps=4 in main(); clarifier Finding 3).
    trainability_record = _gate_cell_trainability(
        len(ds),
        cell_slug,
        smoke=max_steps is not None,
        override_floor_rows=trainability_floor_override,
        override_reason=trainability_override_reason,
    )
    logger.info("[%s] trainability gate: %s", cell_slug, trainability_record)
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
        # fu1 cells pin their layer per-cell (14 or 19); parent cells keep the
        # trait-keyed default (#2225 fu1 plan §4.2).
        l1_idx = cell.l1_idx if cell.l1_idx is not None else L1_LAYER_IDX[steered_trait]
        vectors = _build_steering_vectors(direction, layer_spec, l1_idx)
        _assert_finite_steering_vectors(vectors, str(_direction_path(directions_dir, cell)))
        hook = SteeringHook(vectors, alpha=float(coef), mode=mask_mode)
        if mask_mode == "prefix":
            # mode="prefix" needs a prefix_len column computed at map time from
            # per-segment TOKEN IDS (never a re-tokenized concatenated string).
            ds = ds.map(lambda r: {**r, "prefix_len": compute_prefix_len(tokenizer, r["prompt"])})
        # `is not None` (not truthiness): a pad_token_id of 0 is a valid id and
        # must not silently fall back to EOS (g2 suggestion 5).
        pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
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
    # Artifact-binding record at SAVE time (r2 blocker 1): the resume predicate
    # requires this sha-bound `completed` field before any presence check counts.
    mark_manifest_completed(ckpt_root, cell, out_dir, trainability_record=trainability_record)
    logger.info("[%s] adapter saved to %s", cell_slug, out_dir)

    if upload:
        _upload_cell_adapter(out_dir, cell_slug, hf_prefix=_adapters_prefix(cell))
        mark_manifest_uploaded(ckpt_root, cell)
    _reap_training_residue(out_dir)
    return out_dir


# Parent-default-identical seam: parent cells keep the parent adapters prefix.
# UPLOAD_PREFIX_EXEMPT: fu1 cells thread cell.adapters_hf_prefix via _adapters_prefix()
def _upload_cell_adapter(out_dir: Path, cell_slug: str, hf_prefix: str = ADAPTERS_HF_PREFIX) -> str:
    """Per-cell adapter upload to the HF model repo (#664 per-cell contract)."""
    from explore_persona_space.orchestrate.hub import _upload

    url = _upload(
        Path(out_dir),
        MODEL_REPO,
        "model",
        f"{hf_prefix}/{cell_slug}",
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


def _visible_gpu_entries(n_gpus: int) -> list[str]:
    """Per-slot CUDA_VISIBLE_DEVICES values for the launcher-env pin.

    When the PARENT already runs under a restricted/reordered CVD (SLURM
    partial-node allocation, a pre-set ``CUDA_VISIBLE_DEVICES=4,5,6,7``),
    slot ``g`` pins the parent's g-th ENTRY — absolute ordinals would escape
    the allowed set and collide with other tenants (g2 Concern 3). With no
    parent CVD, ordinals 0..n-1 are the physical ids (dedicated pod).
    """
    parent = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if parent:
        entries = [e.strip() for e in parent.split(",") if e.strip()]
        if n_gpus > len(entries):
            raise RuntimeError(
                f"fan-out width {n_gpus} exceeds the parent CUDA_VISIBLE_DEVICES "
                f"entry count {len(entries)} ({parent!r})"
            )
        return entries[:n_gpus]
    return [str(g) for g in range(n_gpus)]


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
    no_upload: bool = False,
    trainability_floor_override: int | None = None,
    trainability_override_reason: str | None = None,
    script_path: Path | None = None,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(script_path or Path(__file__).resolve()),
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
    if no_upload:
        cmd += ["--no-upload"]  # r2 g1 Concern 2: children honor the parent's escape hatch
    if trainability_floor_override is not None:
        cmd += ["--trainability-floor-override", str(trainability_floor_override)]
    if trainability_override_reason is not None:
        cmd += ["--trainability-override-reason", trainability_override_reason]
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
    allow_upload: bool = True,
    trainability_floor_override: int | None = None,
    trainability_override_reason: str | None = None,
    script_path: Path | None = None,
) -> dict:
    """Work-stealing fan-out: N GPU slots each pull the next PENDING cell.

    A cell is skipped (never launched) when the resume predicate holds. Each
    launched cell runs as a ``--single-cell`` subprocess with its GPU pinned in
    BOTH the launcher env (``CUDA_VISIBLE_DEVICES``) and the ``--gpu-id`` arg
    (the CVD-clobber gotcha), logging to a per-cell file. Work-conserving: an
    idle slot with a pending cell dispatches immediately — no wave barrier.
    ``script_path`` re-targets the ``--single-cell`` child argv at a WRAPPER
    entrypoint (fu1's issue2225_fu1_train.py, which re-registers its cells at
    process entry — the subprocess-registry gotcha); default = this script.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    total = len(cells)
    results: dict[str, str] = {}
    failures: list[str] = []

    gpu_entries = _visible_gpu_entries(n_gpus)

    pending: deque[Cell] = deque()
    for cell in cells:
        # r2 g1 Concern 2: a --no-upload fan-out must never network in the parent.
        if should_skip(cell, ckpt_root, dataset_root, directions_dir, allow_upload=allow_upload):
            results[cell.slug] = "skipped-resume"
            print(f"[fanout] skip {cell.slug} (resume)", flush=True)
            if not dry_run:
                # Skip-evidence line into the per-cell log (r2 blocker 4): the
                # dispatcher's §7 criterion-(i) count gate greps per-cell logs
                # for [steer-hook] OR [fanout-skip], so a resume-skipped cell —
                # whose hook engagement was proven by the fingerprint-bound
                # completed run — never starves the count into a false exit 7.
                with open(log_dir / f"{cell.slug}.log", "a") as skip_fh:
                    skip_fh.write(
                        f"[fanout-skip] {cell.slug} resume "
                        "(hook engagement proven by the fingerprint-bound completed run)\n"
                    )
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
                no_upload=not allow_upload,
                trainability_floor_override=trainability_floor_override,
                trainability_override_reason=trainability_override_reason,
                script_path=script_path,
            )
            print(
                f"[fanout][dry-run] CUDA_VISIBLE_DEVICES={gpu_entries[g]} {' '.join(cmd)}",
                flush=True,
            )
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
                no_upload=not allow_upload,
                trainability_floor_override=trainability_floor_override,
                trainability_override_reason=trainability_override_reason,
                script_path=script_path,
            )
            # g-th ENTRY of the parent CVD when set (g2 Concern 3), else ordinal.
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_entries[g]}
            log_path = log_dir / f"{cell.slug}.log"
            fh = open(log_path, "w")
            print(
                f"[fanout] launch {cell.slug} CUDA_VISIBLE_DEVICES={gpu_entries[g]} log={log_path}",
                flush=True,
            )
            try:
                proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            except Exception:
                fh.close()  # never leak the per-cell log handle on a launch failure
                raise
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
    ap.add_argument(
        "--fan-out",
        action="store_true",
        help="explicit fan-out opt-in — REQUIRED for the full 81-cell launch "
        "(a bare invocation refuses; g2 Concern 4)",
    )
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
    ap.add_argument(
        "--trainability-floor-override",
        type=int,
        default=None,
        help="override the absolute per-cell trainability floor (rows). "
        "LAUNCH-WIDE: a fan-out re-floors EVERY dispatched cell (all 81 on "
        "the full launch), not one; requires --trainability-override-reason "
        "(#2242/#2243)",
    )
    ap.add_argument(
        "--trainability-override-reason",
        default=None,
        help="recorded reason for --trainability-floor-override (rides the gate "
        "record into the per-cell fan-out log AND the cell's completed "
        "manifest; an override is a recorded decision, never silent)",
    )
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
    ap = build_argparser()
    args = ap.parse_args(argv)
    if args.smoke and args.max_steps is None:
        args.max_steps = 4  # --smoke means a TINY slice by itself (g2 suggestion 6)

    if (
        args.trainability_floor_override is not None
        and not (args.trainability_override_reason or "").strip()
    ):
        ap.error(
            "--trainability-floor-override requires a non-empty "
            "--trainability-override-reason (an override is a recorded decision, "
            "never silent; validated here so an N-cell fan-out fails once, not N "
            "times post-dispatch — #2243)"
        )

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Execute every deferred import the production phases reach.
        import issue778_finetune  # noqa: F401
        import issue778_lib  # noqa: F401

        from explore_persona_space.artifacts.datagen import (  # noqa: F401
            assert_cell_trainable,
        )

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
        if should_skip(
            cell, ckpt_root, dataset_root, directions_dir, allow_upload=not args.no_upload
        ):
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
            trainability_floor_override=args.trainability_floor_override,
            trainability_override_reason=args.trainability_override_reason,
            upload=not args.no_upload,
        )
        print(json.dumps({"cell": cell.slug, "adapter": str(out), "status": "done"}))
        return

    # Fan-out. A BARE invocation (no scoping flag at all) would implicitly start
    # the full 81-cell / ~42 GPU-h production fan-out — refuse it (g2 Concern 4):
    # the launch must name its scope explicitly.
    if not (args.fan_out or args.pilot or args.cells or args.smoke or args.dry_run):
        ap.error(
            "refusing the implicit full 81-cell fan-out: pass --fan-out explicitly "
            "(or scope with --pilot / --cells / --smoke / --dry-run)"
        )
    cells = _resolve_cells(args)
    log_dir = Path(args.log_dir) if args.log_dir else ckpt_root / "fanout_logs"
    if args.dry_run:
        if args.n_gpus:
            n_gpus = max(args.n_gpus, 1)
        else:
            # Default preview width clamps to the parent CVD entry count so a
            # --dry-run in a 1-GPU shell (CUDA_VISIBLE_DEVICES=0) never trips
            # _visible_gpu_entries' over-width raise (r2 g1 Concern 3). An
            # explicit --n-gpus over the CVD width still fails loud there.
            parent = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            entries = [e for e in parent.split(",") if e.strip()]
            n_gpus = min(8, len(entries)) if entries else 8
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
        allow_upload=not args.no_upload,
        trainability_floor_override=args.trainability_floor_override,
        trainability_override_reason=args.trainability_override_reason,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
