"""Issue #2225 fu1 (`fu1_preimage_prevention`) — pre-image steering cell registry + fan-out.

Wave-1 grid (fu1 plan §4.3): 80 cells =

- pre-image arms J/K/L/M (2 layers {14, 19} x 2 positions {context, context_end})
  x 4 corpora x 4 coefficients {0.25, 0.75, 1.5, 3.0} = 64 cells;
- random controls RJ/RK/RL/RM (same layer x position lattice) x evil corpus
  x 4 coefficients = 16 cells.

The PARENT registry (scripts/issue2225_train.py, 81 cells) is byte-untouched:
fu1 cells are parent ``Cell`` instances registered through the external
cell-resolver seam (``train.register_cell_resolver`` /
``EPM_I2225_EXTRA_CELLS_MODULE=issue2225_fu1_train``), carrying the fu1-only
fields ``l1_idx`` (per-cell layer 14/19), ``direction_filename`` (the shared
``RND.pt`` bank for random arms), and ``adapters_hf_prefix`` (the fu1 round
prefix — never the parent's, #1452). Training entry is the parent's
``train_steered_cell`` verbatim; the fan-out is the parent's ``run_fan_out``
with ``script_path`` re-targeted at THIS file so every ``--single-cell`` child
re-registers the fu1 cells at process entry (the subprocess-registry gotcha).

Usage (pod-side, via scripts/issue2225_fu1_dispatch.sh):

  uv run python scripts/issue2225_fu1_train.py --pilot --fan-out \
      --ckpt-root checkpoints/issue_2225_fu1 --directions-dir <fu1 bank> ...
  uv run python scripts/issue2225_fu1_train.py --fan-out --cells J__evil__c0.25,...
  uv run python scripts/issue2225_fu1_train.py --single-cell K__evil__c1.5 ...

F1 pilot (plan §7): configs K + M (the two L19 pre-image arms) x 4 coefficients
on evil II = 8 production cells, resumed into F2a on pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue2225_train as train  # noqa: E402
from issue2225_train import Cell  # noqa: E402

# ── fu1 constants ──────────────────────────────────────────────────────────────

FU1_ADAPTERS_HF_PREFIX = "issue2225_ctxsteer/fu1_preimage/adapters"
FU1_GRID: tuple[float, ...] = (0.25, 0.75, 1.5, 3.0)
FU1_DATASETS: tuple[str, ...] = train.DATASETS  # evil/sycophancy/hallucination/mistake_opinions
EXPECTED_FU1_CELL_COUNT = 80

# The §7 F1 pilot gate: K + M (both L19 pre-image arms) at the 4 fu1
# coefficients, evil II (8 cells — production cells, resumed into F2a on pass).
FU1_PILOT_CONFIGS: tuple[str, ...] = ("K", "M")
FU1_PILOT_DATASET = "evil"

# The shared random bank (plan §4.1: one (28, 3584) bank, rows 14/19 = the two
# per-layer #2220-construction random directions; trait-agnostic by design).
RND_BANK_FILENAME = "RND.pt"


class FuConfigSpec:
    """One fu1 §5 config: (variant, mask position, L1 layer) x its datasets."""

    __slots__ = ("config", "datasets", "l1_idx", "mask_mode", "slug", "variant")

    def __init__(
        self,
        config: str,
        slug: str,
        variant: str,
        mask_mode: str,
        l1_idx: int,
        datasets: tuple[str, ...],
    ):
        self.config = config
        self.slug = slug
        self.variant = variant
        self.mask_mode = mask_mode
        self.l1_idx = l1_idx
        self.datasets = datasets


FU1_CONFIGS: tuple[FuConfigSpec, ...] = (
    # Pre-image arms (plan §5): variant PRE, all 4 corpora.
    FuConfigSpec("J", "J_pres2l14", "PRE", "context", 14, FU1_DATASETS),
    FuConfigSpec("K", "K_pres2l19", "PRE", "context", 19, FU1_DATASETS),
    FuConfigSpec("L", "L_presel14", "PRE", "context_end", 14, FU1_DATASETS),
    FuConfigSpec("M", "M_presel19", "PRE", "context_end", 19, FU1_DATASETS),
    # Random controls: variant RND, evil corpus only, shared RND.pt bank.
    FuConfigSpec("RJ", "RJ_rnds2l14", "RND", "context", 14, ("evil",)),
    FuConfigSpec("RK", "RK_rnds2l19", "RND", "context", 19, ("evil",)),
    FuConfigSpec("RL", "RL_rndsel14", "RND", "context_end", 14, ("evil",)),
    FuConfigSpec("RM", "RM_rndsel19", "RND", "context_end", 19, ("evil",)),
)

_FU1_SPEC_BY_CONFIG: dict[str, FuConfigSpec] = {s.config: s for s in FU1_CONFIGS}


def _fu1_cell(spec: FuConfigSpec, dataset: str, coef: float) -> Cell:
    """Materialize one fu1 cell as a parent ``Cell`` with the fu1 fields set."""
    return Cell(
        slug=f"{spec.config}__{dataset}__c{coef}",
        config=spec.config,
        dataset=dataset,
        steered_trait=train.STEERED_TRAIT[dataset],
        variant=spec.variant,
        mask_mode=train._MASK_MODE[spec.mask_mode],
        layer_spec="L1",  # every fu1 arm is single-layer (plan §2 divergence 3)
        coef=coef,
        prompt_mode=False,
        l1_idx=spec.l1_idx,
        direction_filename=(RND_BANK_FILENAME if spec.variant == "RND" else None),
        adapters_hf_prefix=FU1_ADAPTERS_HF_PREFIX,
    )


def build_fu1_cell_registry() -> list[Cell]:
    """Enumerate every fu1 cell (config x dataset x coef); asserts exactly 80."""
    cells = [
        _fu1_cell(spec, dataset, coef)
        for spec in FU1_CONFIGS
        for dataset in spec.datasets
        for coef in FU1_GRID
    ]
    slugs = [c.slug for c in cells]
    if len(slugs) != len(set(slugs)):
        dupes = sorted({s for s in slugs if slugs.count(s) > 1})
        raise AssertionError(f"duplicate fu1 cell slugs: {dupes}")
    if len(cells) != EXPECTED_FU1_CELL_COUNT:
        raise AssertionError(
            f"fu1 registry enumerated {len(cells)} cells, expected {EXPECTED_FU1_CELL_COUNT}"
        )
    return cells


def fu1_cells_by_slug() -> dict[str, Cell]:
    return {c.slug: c for c in build_fu1_cell_registry()}


def fu1_pilot_cells() -> list[Cell]:
    """The §7 F1 gate cells: K + M at the 4 fu1 coefficients on evil II (8 cells)."""
    return [
        c
        for c in build_fu1_cell_registry()
        if c.config in FU1_PILOT_CONFIGS and c.dataset == FU1_PILOT_DATASET
    ]


# Plan §7 grid inheritance: the F1 pilot runs the L19 arms only (K context,
# M context_end); every non-piloted config INHERITS the effective grid of the
# piloted arm sharing its MASK class (L14 arms inherit the L19-piloted grid;
# RND controls follow their mask's arm — a shift applies "for ALL fu configs").
_GRID_INHERIT: dict[str, str] = {
    "J": "K",
    "K": "K",
    "RJ": "K",
    "RK": "K",
    "L": "M",
    "M": "M",
    "RL": "M",
    "RM": "M",
}


def effective_fu1_grids(
    repilot_state_path: str | Path | None = None,
) -> dict[str, tuple[float, ...]]:
    """Per-config EFFECTIVE coefficient grid after the F1 gate (plan §7).

    Default (no state file): ``FU1_GRID`` for every config. When a RESOLVED
    ``f1_repilot_state.json`` exists, each octave-shifted pilot arm's grid
    (its ``grid_csv``) replaces the default for that arm AND for every config
    inheriting from it per ``_GRID_INHERIT``. An UNRESOLVED state (mid-repilot
    crash) fails loud — production phases must never enumerate a grid the F1
    gate is still adjudicating.
    """
    grids: dict[str, tuple[float, ...]] = {spec.config: FU1_GRID for spec in FU1_CONFIGS}
    if repilot_state_path is None:
        return grids
    path = Path(repilot_state_path)
    if not path.exists():
        return grids
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    if not state.get("resolved", False):
        raise RuntimeError(
            f"f1 repilot state {path} is UNRESOLVED — run phase f1 to resolution "
            "before enumerating production cells (plan §7)"
        )
    shifted: dict[str, tuple[float, ...]] = {}
    for arm, block in state["plan"].items():
        grid = tuple(float(x) for x in str(block["grid_csv"]).split(",") if x.strip())
        if not grid:
            raise ValueError(f"empty grid_csv for repilot arm {arm!r} in {path}")
        shifted[arm] = grid
    unknown = sorted(set(shifted) - set(_GRID_INHERIT))
    if unknown:
        raise ValueError(f"repilot state names non-fu1 arm(s) {unknown} in {path}")
    for config in grids:
        pilot_arm = _GRID_INHERIT[config]
        if pilot_arm in shifted:
            grids[config] = shifted[pilot_arm]
    return grids


def effective_fu1_cells(repilot_state_path: str | Path | None = None) -> list[Cell]:
    """The production cell enumeration at the F1-EFFECTIVE grid (plan §7).

    Identical to ``build_fu1_cell_registry()`` when no resolved repilot state
    exists; after an octave shift, the shifted configs enumerate their shifted
    coefficients (so the re-piloted adapters are first-class F2 targets and the
    demonstrated-mis-placed original grid is never fanned out at full spend).
    """
    grids = effective_fu1_grids(repilot_state_path)
    cells = [
        _fu1_cell(spec, dataset, coef)
        for spec in FU1_CONFIGS
        for dataset in spec.datasets
        for coef in grids[spec.config]
    ]
    slugs = [c.slug for c in cells]
    assert len(slugs) == len(set(slugs)), "duplicate effective fu1 cell slugs"
    return cells


def fu1_extreme_cells(cells: Sequence[Cell]) -> list[Cell]:
    """Per (config x corpus) arm: the min- and max-coefficient cells (F2c MMLU
    extremes). Follows whatever grid ``cells`` was enumerated at."""
    by_arm: dict[tuple[str, str], list[Cell]] = {}
    for c in cells:
        by_arm.setdefault((c.config, c.dataset), []).append(c)
    out: list[Cell] = []
    for arm in sorted(by_arm):
        arm_cells = sorted(by_arm[arm], key=lambda c: c.coef)
        out.append(arm_cells[0])
        if arm_cells[-1] is not arm_cells[0]:
            out.append(arm_cells[-1])
    return out


def synth_fu1_cell(config: str, dataset: str, coef: float) -> Cell:
    """fu1 twin of parent ``synth_cell``: canonical-slug cell at an arbitrary
    coefficient (the §7 octave-shift re-pilot path). Raises on an unknown fu1
    config, an out-of-coverage dataset, or a non-finite/non-positive coef."""
    spec = _FU1_SPEC_BY_CONFIG.get(config)
    if spec is None:
        raise ValueError(f"unknown fu1 config {config!r} (have {sorted(_FU1_SPEC_BY_CONFIG)})")
    if dataset not in spec.datasets:
        raise ValueError(f"dataset {dataset!r} not in fu1 config {config}'s {spec.datasets}")
    coef = float(coef)
    if not math.isfinite(coef) or coef <= 0:
        raise ValueError(f"steering coefficient must be finite and > 0, got {coef}")
    return _fu1_cell(spec, dataset, coef)


# fu1 canonical scaled-cell slugs: {J|K|L|M|RJ|RK|RL|RM}__{dataset}__c{coef}.

_FU1_SLUG_RE = re.compile(r"^(R?[JKLM])__([a-z_]+)__c([0-9.]+)$")


def resolve_fu1_cell(slug: str) -> Cell | None:
    """The external resolver registered into the parent's seam.

    Registry lookup first; on miss, parse a canonical fu1 scaled slug (the §7
    re-pilot path — mirrors parent ``resolve_cell`` semantics, including the
    canonical-spelling refusal). Returns None for non-fu1 slugs so the parent's
    own resolution chain continues.
    """
    by_slug = fu1_cells_by_slug()
    if slug in by_slug:
        return by_slug[slug]
    m = _FU1_SLUG_RE.match(slug)
    if not m:
        return None
    config, dataset, coef_txt = m.groups()
    cell = synth_fu1_cell(config, dataset, float(coef_txt))
    if cell.slug != slug:
        raise ValueError(
            f"non-canonical fu1 coefficient spelling {slug!r} (canonical: {cell.slug!r})"
        )
    return cell


def register_extra_cells() -> None:
    """Idempotent registration into the parent resolver seam.

    Called at every entrypoint of THIS script AND by the parent's
    ``EPM_I2225_EXTRA_CELLS_MODULE`` env hook (which reaches the eval-side
    scripts' subprocesses without any per-script wiring).
    """
    train.register_cell_resolver(resolve_fu1_cell)


# ── CLI (mirrors the parent surface; delegates to the parent machinery) ───────


def _resolve_fu1_cells(args) -> list[Cell]:
    if args.pilot:
        configs = (
            [c.strip() for c in args.pilot_configs.split(",") if c.strip()]
            if args.pilot_configs
            else list(FU1_PILOT_CONFIGS)
        )
        unknown = [c for c in configs if c not in FU1_PILOT_CONFIGS]
        if unknown:
            raise ValueError(
                f"--pilot-configs must be a subset of {FU1_PILOT_CONFIGS}, got {unknown}"
            )
        base = [c for c in fu1_pilot_cells() if c.config in configs]
        if args.coef_scale is not None:
            # §7 octave-shift re-pilot: multiply the pilot grid (x0.5 too-hot /
            # x2 too-cold, per the f1_verdict.json recommendation).
            return [synth_fu1_cell(c.config, c.dataset, c.coef * args.coef_scale) for c in base]
        return base
    if args.coef_scale is not None or args.pilot_configs:
        raise ValueError("--coef-scale / --pilot-configs require --pilot")
    if args.cells:
        wanted = [s.strip() for s in args.cells.split(",") if s.strip()]
        return [train.resolve_cell(s) for s in wanted]
    cells = build_fu1_cell_registry()
    if args.smoke:
        # Per-arm-class smoke floor (plan §4.5): one `context` + one
        # `context_end` cell — the two NEW mask classes this round adds.
        return [fu1_cells_by_slug()["J__evil__c0.25"], fu1_cells_by_slug()["L__evil__c0.25"]]
    return cells


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2225 fu1 pre-image steering fan-out.")
    ap.add_argument("--dataset-root", default="external/persona_vectors/dataset")
    ap.add_argument("--ckpt-root", default="checkpoints/issue_2225_fu1")
    ap.add_argument(
        "--directions-dir",
        default="eval_results/issue_2225/fu1_preimage_prevention/directions",
        help="dir holding the fu1 direction bank ({trait}_PRE.pt + RND.pt; F0 output)",
    )
    ap.add_argument("--single-cell", default=None, help="train ONE cell by slug (subprocess mode)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--fan-out",
        action="store_true",
        help="explicit fan-out opt-in — REQUIRED for the full 80-cell launch",
    )
    ap.add_argument("--pilot", action="store_true", help="the §7 F1 gate's 8 cells (K+M evil II)")
    ap.add_argument(
        "--coef-scale",
        type=float,
        default=None,
        help="§7 octave-shift re-pilot: multiply the pilot grid by this factor "
        "(x0.5 too-hot / x2 too-cold; requires --pilot)",
    )
    ap.add_argument(
        "--pilot-configs",
        default=None,
        help="restrict --pilot to a subset of the fu1 pilot arms (default: K,M)",
    )
    ap.add_argument("--cells", default=None, help="restrict to a comma-separated slug list")
    ap.add_argument("--n-gpus", type=int, default=None, help="fan-out width (default: detected)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap training steps (smoke)")
    ap.add_argument(
        "--smoke", action="store_true", help="2 cells (one context + one context_end), tiny slice"
    )
    ap.add_argument("--cpu-only", action="store_true", help="deliberate CPU smoke")
    ap.add_argument("--dry-run", action="store_true", help="preview the fan-out, no CUDA")
    ap.add_argument("--no-upload", action="store_true", help="skip per-cell HF adapter upload")
    ap.add_argument("--model", default=train.DEFAULT_MODEL_NAME)
    ap.add_argument("--trainability-floor-override", type=int, default=None)
    ap.add_argument("--trainability-override-reason", default=None)
    ap.add_argument("--log-dir", default=None, help="per-cell fan-out log dir")
    ap.add_argument("--check-registry", action="store_true", help="assert 80 cells, print summary")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)
    register_extra_cells()  # point-of-use registration — every process, incl. children
    # Children launched by the parent fan-out (and by the eval-side scripts'
    # own fan-outs) inherit this env, so THEIR resolve_cell calls re-register.
    os.environ.setdefault(train.EXTRA_CELLS_MODULE_ENV, "issue2225_fu1_train")
    if args.smoke and args.max_steps is None:
        args.max_steps = 4  # parent convention: --smoke alone means a TINY slice

    if (
        args.trainability_floor_override is not None
        and not (args.trainability_override_reason or "").strip()
    ):
        ap.error(
            "--trainability-floor-override requires a non-empty "
            "--trainability-override-reason (#2242/#2243)"
        )

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Execute the parent module's deferred-import set through ITS import
        # check semantics (the fu1 child process runs the same training body).
        import issue778_finetune  # noqa: F401
        import issue778_lib  # noqa: F401

        from explore_persona_space.artifacts.datagen import assert_cell_trainable  # noqa: F401
        from explore_persona_space.experiments.issue2225.directions import (  # noqa: F401
            L1_LAYER_IDX,
        )
        from explore_persona_space.experiments.issue2225.steer_train import (  # noqa: F401
            MASK_MODES,
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

        assert "context_end" in MASK_MODES, MASK_MODES
        build_fu1_cell_registry()  # asserts 80
        train.build_cell_registry()  # parent stays 81
        # fu1 resolution round-trips: registry member, scaled slug, and the
        # parent seam (resolve_cell must reach the fu1 resolver).
        assert train.resolve_cell("J__evil__c0.25").l1_idx == 14
        assert train.resolve_cell("RM__evil__c3.0").direction_filename == RND_BANK_FILENAME
        assert train.resolve_cell("K__evil__c6.0").coef == 6.0  # octave-shifted synth
        assert train.resolve_cell("A__evil__c0.25").coef == 0.25  # parent path intact
        print("[issue2225-fu1-train] import-check OK", flush=True)
        raise SystemExit(0)

    if args.check_registry:
        cells = build_fu1_cell_registry()
        by_config: dict[str, int] = {}
        for c in cells:
            by_config[c.config] = by_config.get(c.config, 0) + 1
        print(f"[check-registry] {len(cells)} fu1 cells total", flush=True)
        for spec in FU1_CONFIGS:
            print(
                f"  {spec.config} ({spec.slug}): {by_config.get(spec.config, 0)} cells "
                f"l1_idx={spec.l1_idx} mask={spec.mask_mode} variant={spec.variant}",
                flush=True,
            )
        raise SystemExit(0)

    dataset_root = Path(args.dataset_root)
    ckpt_root = Path(args.ckpt_root)
    directions_dir = Path(args.directions_dir)

    if args.single_cell is not None:
        cell = train.resolve_cell(args.single_cell)
        if train.should_skip(
            cell, ckpt_root, dataset_root, directions_dir, allow_upload=not args.no_upload
        ):
            print(json.dumps({"cell": cell.slug, "status": "skipped-resume"}))
            return
        out = train.train_steered_cell(
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

    if not (args.fan_out or args.pilot or args.cells or args.smoke or args.dry_run):
        ap.error(
            "refusing the implicit full 80-cell fan-out: pass --fan-out explicitly "
            "(or scope with --pilot / --cells / --smoke / --dry-run)"
        )
    cells = _resolve_fu1_cells(args)
    log_dir = Path(args.log_dir) if args.log_dir else ckpt_root / "fanout_logs"
    if args.dry_run:
        if args.n_gpus:
            n_gpus = max(args.n_gpus, 1)
        else:
            parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            entries = [e for e in parent_cvd.split(",") if e.strip()]
            n_gpus = min(8, len(entries)) if entries else 8
    else:
        n_gpus = train._detect_gpu_count(args.cpu_only)
        if args.n_gpus:
            n_gpus = min(n_gpus, max(args.n_gpus, 1))
    res = train.run_fan_out(
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
        script_path=Path(__file__).resolve(),
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
