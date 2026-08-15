"""Issue #2225 fu2 (`fu2_preimage_alltoken`) — pre-image ALL-TOKEN steering cell registry.

Wave-1 grid (fu2 plan v13 §4.3): 28 cells =

- pre-image arms N/Q (2 layers {14, 19}, mask position `all`) x 3 corpora
  (evil / sycophancy / hallucination — `mistake_opinions` DROPPED, plan §2
  divergence 1) x 4 coefficients {0.25, 0.75, 1.5, 3.0} = 24 cells;
- random all-token control RQ (L19) x evil corpus x 4 coefficients = 4 cells.

The conditional W2a arm RN (`RN_rnds1l14`, random @L14) is REGISTERED — its
slugs resolve through the external seam and `synth_fu2_cell` — but NOT
enumerated in the wave-1 registry (plan §4.3: fires only on an N-arm
Effect-negative read).

The PARENT registry (scripts/issue2225_train.py, 81 cells) and the fu1
registry (scripts/issue2225_fu1_train.py, 80 cells) are byte-untouched: fu2
cells are parent ``Cell`` instances registered through the same external
cell-resolver seam (``train.register_cell_resolver`` /
``EPM_I2225_EXTRA_CELLS_MODULE=issue2225_fu2_train``), carrying ``l1_idx``
(14/19), ``mask_mode="all"`` (the parent arm-A/G semantic — NO new mask
code), ``direction_filename`` (the shared fu1 ``RND.pt`` bank for random
arms), and ``adapters_hf_prefix`` (the fu2 round prefix, #1452
never-clobber-the-parent). Training entry is the parent's
``train_steered_cell`` verbatim; ``directions_dir`` is the STAGED fu1 bank
(S0 — reused, no build; hook ``alpha = c``: bank rows are ρ-pre-scaled, so
``α_eff = c·ρ_ℓ`` with NO rescaling here).

Usage (pod-side, via scripts/issue2225_fu2_dispatch.sh):

  uv run python scripts/issue2225_fu2_train.py --pilot --fan-out \
      --ckpt-root checkpoints/issue_2225_fu2 --directions-dir <staged fu1 bank> ...
  uv run python scripts/issue2225_fu2_train.py --fan-out --cells N__evil__c0.25,...
  uv run python scripts/issue2225_fu2_train.py --single-cell Q__evil__c1.5 ...

F1' pilot (plan §7): config Q (the L19 all-token pre-image arm) x 4
coefficients on evil II = 4 production cells, resumed into F2a on pass.
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

import issue2225_fu1_train as fu1  # noqa: E402  (FuConfigSpec + shared machinery)
import issue2225_train as train  # noqa: E402
from issue2225_fu1_train import FuConfigSpec, RND_BANK_FILENAME  # noqa: E402
from issue2225_train import Cell  # noqa: E402

# ── fu2 constants ──────────────────────────────────────────────────────────────

FU2_ADAPTERS_HF_PREFIX = "issue2225_ctxsteer/fu2_preimage/adapters"
FU2_GRID: tuple[float, ...] = (0.25, 0.75, 1.5, 3.0)  # fu1-piloted; octave_shift none
FU2_DATASETS: tuple[str, ...] = ("evil", "sycophancy", "hallucination")  # opinions dropped
EXPECTED_FU2_CELL_COUNT = 28

# The §7 F1' pilot gate: Q (the L19 all-token pre-image arm) at the 4 fu2
# coefficients, evil II (4 cells — production cells, resumed into F2a on pass).
FU2_PILOT_CONFIGS: tuple[str, ...] = ("Q",)
FU2_PILOT_DATASET = "evil"

# Wave-1 configs (plan §5). Mask mode `all` = the parent arm-A/G all-position
# semantic (steer_train MASK_MODES / masks_for_mode "all" branch — no new code).
FU2_CONFIGS: tuple[FuConfigSpec, ...] = (
    FuConfigSpec("N", "N_pres1l14", "PRE", "all", 14, FU2_DATASETS),
    FuConfigSpec("Q", "Q_pres1l19", "PRE", "all", 19, FU2_DATASETS),
    FuConfigSpec("RQ", "RQ_rnds1l19", "RND", "all", 19, ("evil",)),
)

# Conditional W2a/W2b arm (plan §4.3): registered (resolvable + synthable), NOT
# in the wave-1 enumeration. Dataset coverage: evil = W2a (N Effect-negative on
# evil); sycophancy = W2b, TRIGGERED 2026-08-14 — N_sycophancy read trait 73.27
# at coherence 91.13 (16.48 below the banked unsteered baseline 89.75), so the
# matched random control (same layer/position/corpus) materializes on
# sycophancy at the selected coef 3.0 ± one neighbor (1.5).
FU2_CONDITIONAL_CONFIGS: tuple[FuConfigSpec, ...] = (
    FuConfigSpec("RN", "RN_rnds1l14", "RND", "all", 14, ("evil", "sycophancy")),
)

_FU2_SPEC_BY_CONFIG: dict[str, FuConfigSpec] = {
    s.config: s for s in (*FU2_CONFIGS, *FU2_CONDITIONAL_CONFIGS)
}


def _fu2_cell(spec: FuConfigSpec, dataset: str, coef: float) -> Cell:
    """Materialize one fu2 cell as a parent ``Cell`` with the fu2 fields set."""
    return Cell(
        slug=f"{spec.config}__{dataset}__c{coef}",
        config=spec.config,
        dataset=dataset,
        steered_trait=train.STEERED_TRAIT[dataset],
        variant=spec.variant,
        mask_mode=train._MASK_MODE[spec.mask_mode],  # "all"
        layer_spec="L1",  # every fu2 arm is single-layer
        coef=coef,
        prompt_mode=False,
        l1_idx=spec.l1_idx,
        direction_filename=(RND_BANK_FILENAME if spec.variant == "RND" else None),
        adapters_hf_prefix=FU2_ADAPTERS_HF_PREFIX,
    )


def _assert_disjoint_from_parent_and_fu1(cells: Sequence[Cell]) -> None:
    """Build-time disjointness keyed on FULL slugs + output dirs (plan §4.2).

    Never bare config letters — the parent owns letter ``P`` via its
    ``P_e3sPl1`` prefix arm, so letter-level checks are structurally unsound.
    Output-dir disjointness = the per-round HF adapter prefixes (which also
    prefix the per-cell ckpt/eval file namespaces) must be pairwise distinct.
    """
    fu2_slugs = {c.slug for c in cells}
    parent_slugs = set(train.cells_by_slug())
    fu1_slugs = set(fu1.fu1_cells_by_slug())
    clash = sorted(fu2_slugs & (parent_slugs | fu1_slugs))
    if clash:
        raise AssertionError(f"fu2 cell slugs collide with parent/fu1 registries: {clash}")
    prefixes = {
        "parent": train.ADAPTERS_HF_PREFIX,
        "fu1": fu1.FU1_ADAPTERS_HF_PREFIX,
        "fu2": FU2_ADAPTERS_HF_PREFIX,
    }
    if len(set(prefixes.values())) != len(prefixes):
        raise AssertionError(f"adapter output prefixes are not pairwise distinct: {prefixes}")


def build_fu2_cell_registry() -> list[Cell]:
    """Enumerate every wave-1 fu2 cell (config x dataset x coef); asserts 28
    AND full-slug/output-dir disjointness from the parent + fu1 registries."""
    cells = [
        _fu2_cell(spec, dataset, coef)
        for spec in FU2_CONFIGS
        for dataset in spec.datasets
        for coef in FU2_GRID
    ]
    slugs = [c.slug for c in cells]
    if len(slugs) != len(set(slugs)):
        dupes = sorted({s for s in slugs if slugs.count(s) > 1})
        raise AssertionError(f"duplicate fu2 cell slugs: {dupes}")
    if len(cells) != EXPECTED_FU2_CELL_COUNT:
        raise AssertionError(
            f"fu2 registry enumerated {len(cells)} cells, expected {EXPECTED_FU2_CELL_COUNT}"
        )
    # Disjointness covers the conditional RN slugs too (they are resolvable,
    # so a collision there would be just as corrupting at eval time).
    conditional = [
        _fu2_cell(spec, dataset, coef)
        for spec in FU2_CONDITIONAL_CONFIGS
        for dataset in spec.datasets
        for coef in FU2_GRID
    ]
    _assert_disjoint_from_parent_and_fu1([*cells, *conditional])
    return cells


def fu2_cells_by_slug() -> dict[str, Cell]:
    return {c.slug: c for c in build_fu2_cell_registry()}


def fu2_pilot_cells() -> list[Cell]:
    """The §7 F1' gate cells: Q at the 4 fu2 coefficients on evil II (4 cells)."""
    return [
        c
        for c in build_fu2_cell_registry()
        if c.config in FU2_PILOT_CONFIGS and c.dataset == FU2_PILOT_DATASET
    ]


# Plan §7 grid inheritance: the F1' pilot runs Q only; every other fu2 config
# INHERITS Q's effective grid (one mask class this round — `all`).
_GRID_INHERIT_FU2: dict[str, str] = {"N": "Q", "Q": "Q", "RQ": "Q", "RN": "Q"}


def effective_fu2_grids(
    repilot_state_path: str | Path | None = None,
) -> dict[str, tuple[float, ...]]:
    """Per-config EFFECTIVE coefficient grid after the F1' gate (plan §7).

    Default (no state file): ``FU2_GRID`` for every config (incl. the
    conditional RN). A RESOLVED ``f1_repilot_state.json`` replaces the grid of
    each shifted pilot arm AND of every config inheriting from it per
    ``_GRID_INHERIT_FU2``; an UNRESOLVED state fails loud (production phases
    must never enumerate a grid the F1' gate is still adjudicating).
    """
    grids: dict[str, tuple[float, ...]] = {cfg: FU2_GRID for cfg in _GRID_INHERIT_FU2}
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
    unknown = sorted(set(shifted) - set(_GRID_INHERIT_FU2))
    if unknown:
        raise ValueError(f"repilot state names non-fu2 arm(s) {unknown} in {path}")
    for config in grids:
        pilot_arm = _GRID_INHERIT_FU2[config]
        if pilot_arm in shifted:
            grids[config] = shifted[pilot_arm]
    return grids


def effective_fu2_cells(repilot_state_path: str | Path | None = None) -> list[Cell]:
    """The wave-1 production enumeration at the F1'-EFFECTIVE grid (plan §7)."""
    grids = effective_fu2_grids(repilot_state_path)
    cells = [
        _fu2_cell(spec, dataset, coef)
        for spec in FU2_CONFIGS
        for dataset in spec.datasets
        for coef in grids[spec.config]
    ]
    slugs = [c.slug for c in cells]
    assert len(slugs) == len(set(slugs)), "duplicate effective fu2 cell slugs"
    return cells


def fu2_extreme_cells(cells: Sequence[Cell]) -> list[Cell]:
    """Per (config x corpus) arm: min + max coefficient cells (F2c MMLU
    extremes — 14 targets at the default wave-1 grid). Reuses fu1's generic
    enumeration over whatever grid ``cells`` was built at."""
    return fu1.fu1_extreme_cells(cells)


def synth_fu2_cell(config: str, dataset: str, coef: float) -> Cell:
    """fu2 twin of parent ``synth_cell``: canonical-slug cell at an arbitrary
    coefficient (the §7 octave-shift re-pilot path; also how the conditional
    RN cells materialize). Raises on an unknown fu2 config, an out-of-coverage
    dataset, or a non-finite/non-positive coef."""
    spec = _FU2_SPEC_BY_CONFIG.get(config)
    if spec is None:
        raise ValueError(f"unknown fu2 config {config!r} (have {sorted(_FU2_SPEC_BY_CONFIG)})")
    if dataset not in spec.datasets:
        raise ValueError(f"dataset {dataset!r} not in fu2 config {config}'s {spec.datasets}")
    coef = float(coef)
    if not math.isfinite(coef) or coef <= 0:
        raise ValueError(f"steering coefficient must be finite and > 0, got {coef}")
    return _fu2_cell(spec, dataset, coef)


# fu2 canonical scaled-cell slugs: {N|Q|RN|RQ}__{dataset}__c{coef}. The fresh
# N/Q letters (and their R-prefixed randoms) are disjoint from the parent's
# single letters AND fu1's R?[JKLM] class by construction; the FULL-slug
# disjointness assert above stays the binding check.

_FU2_SLUG_RE = re.compile(r"^(R?[NQ])__([a-z_]+)__c([0-9.]+)$")


def resolve_fu2_cell(slug: str) -> Cell | None:
    """The external resolver registered into the parent's seam.

    Wave-1 registry lookup first; on miss, parse a canonical fu2 scaled slug
    (the §7 re-pilot path AND the conditional RN materialization — mirrors
    parent ``resolve_cell`` semantics incl. the canonical-spelling refusal).
    Returns None for non-fu2 slugs so the parent's resolution chain continues.
    """
    by_slug = fu2_cells_by_slug()
    if slug in by_slug:
        return by_slug[slug]
    m = _FU2_SLUG_RE.match(slug)
    if not m:
        return None
    config, dataset, coef_txt = m.groups()
    cell = synth_fu2_cell(config, dataset, float(coef_txt))
    if cell.slug != slug:
        raise ValueError(
            f"non-canonical fu2 coefficient spelling {slug!r} (canonical: {cell.slug!r})"
        )
    return cell


def register_extra_cells() -> None:
    """Idempotent registration into the parent resolver seam.

    Called at every entrypoint of THIS script AND by the parent's
    ``EPM_I2225_EXTRA_CELLS_MODULE`` env hook (which reaches the eval-side
    scripts' subprocesses without any per-script wiring).
    """
    train.register_cell_resolver(resolve_fu2_cell)


# ── CLI (mirrors the fu1 surface; delegates to the parent machinery) ──────────


def _resolve_fu2_cells(args) -> list[Cell]:
    if args.pilot:
        configs = (
            [c.strip() for c in args.pilot_configs.split(",") if c.strip()]
            if args.pilot_configs
            else list(FU2_PILOT_CONFIGS)
        )
        unknown = [c for c in configs if c not in FU2_PILOT_CONFIGS]
        if unknown:
            raise ValueError(
                f"--pilot-configs must be a subset of {FU2_PILOT_CONFIGS}, got {unknown}"
            )
        base = [c for c in fu2_pilot_cells() if c.config in configs]
        if args.coef_scale is not None:
            # §7 octave-shift re-pilot: multiply the pilot grid (x0.5 too-hot /
            # x2 too-cold, per the f1_verdict.json recommendation).
            return [synth_fu2_cell(c.config, c.dataset, c.coef * args.coef_scale) for c in base]
        return base
    if args.coef_scale is not None or args.pilot_configs:
        raise ValueError("--coef-scale / --pilot-configs require --pilot")
    if args.cells:
        wanted = [s.strip() for s in args.cells.split(",") if s.strip()]
        return [train.resolve_cell(s) for s in wanted]
    cells = build_fu2_cell_registry()
    if args.smoke:
        # Per-arm-class smoke floor (plan §4.5): one PRE + one RND all-token
        # cell — the two fu2 config classes.
        return [fu2_cells_by_slug()["N__evil__c0.25"], fu2_cells_by_slug()["RQ__evil__c0.25"]]
    return cells


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2225 fu2 all-token pre-image fan-out.")
    ap.add_argument("--dataset-root", default="external/persona_vectors/dataset")
    ap.add_argument("--ckpt-root", default="checkpoints/issue_2225_fu2")
    ap.add_argument(
        "--directions-dir",
        default="eval_results/issue_2225/fu2_preimage_alltoken/directions",
        help="dir holding the STAGED fu1 direction bank ({trait}_PRE.pt + RND.pt; S0 output)",
    )
    ap.add_argument("--single-cell", default=None, help="train ONE cell by slug (subprocess mode)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--fan-out",
        action="store_true",
        help="explicit fan-out opt-in — REQUIRED for the full 28-cell launch",
    )
    ap.add_argument("--pilot", action="store_true", help="the §7 F1' gate's 4 cells (Q evil II)")
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
        help="restrict --pilot to a subset of the fu2 pilot arms (default: Q)",
    )
    ap.add_argument("--cells", default=None, help="restrict to a comma-separated slug list")
    ap.add_argument("--n-gpus", type=int, default=None, help="fan-out width (default: detected)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap training steps (smoke)")
    ap.add_argument(
        "--smoke", action="store_true", help="2 cells (one PRE + one RND all-token), tiny slice"
    )
    ap.add_argument("--cpu-only", action="store_true", help="deliberate CPU smoke")
    ap.add_argument("--dry-run", action="store_true", help="preview the fan-out, no CUDA")
    ap.add_argument("--no-upload", action="store_true", help="skip per-cell HF adapter upload")
    ap.add_argument("--model", default=train.DEFAULT_MODEL_NAME)
    ap.add_argument("--trainability-floor-override", type=int, default=None)
    ap.add_argument("--trainability-override-reason", default=None)
    ap.add_argument("--log-dir", default=None, help="per-cell fan-out log dir")
    ap.add_argument("--check-registry", action="store_true", help="assert 28 cells, print summary")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)
    register_extra_cells()  # point-of-use registration — every process, incl. children
    # Children launched by the fan-out (and by the eval-side scripts' own
    # fan-outs) inherit this env, so THEIR resolve_cell calls re-register.
    os.environ.setdefault(train.EXTRA_CELLS_MODULE_ENV, "issue2225_fu2_train")
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
        # check semantics (the fu2 child process runs the same training body).
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

        assert "all" in MASK_MODES, MASK_MODES
        build_fu2_cell_registry()  # asserts 28 + disjointness vs parent/fu1
        train.build_cell_registry()  # parent stays 81
        fu1.build_fu1_cell_registry()  # fu1 stays 80
        # fu2 resolution round-trips: registry member, scaled slug, the
        # conditional RN synth, and the parent seam.
        assert train.resolve_cell("N__evil__c0.25").l1_idx == 14
        assert train.resolve_cell("Q__evil__c3.0").mask_mode == "all"
        assert train.resolve_cell("RQ__evil__c0.25").direction_filename == RND_BANK_FILENAME
        assert train.resolve_cell("Q__evil__c6.0").coef == 6.0  # octave-shifted synth
        assert train.resolve_cell("RN__evil__c0.25").l1_idx == 14  # conditional W2a synth
        assert train.resolve_cell("A__evil__c0.25").coef == 0.25  # parent path intact
        print("[issue2225-fu2-train] import-check OK", flush=True)
        raise SystemExit(0)

    if args.check_registry:
        cells = build_fu2_cell_registry()
        by_config: dict[str, int] = {}
        for c in cells:
            by_config[c.config] = by_config.get(c.config, 0) + 1
        print(f"[check-registry] {len(cells)} fu2 wave-1 cells total", flush=True)
        for spec in FU2_CONFIGS:
            print(
                f"  {spec.config} ({spec.slug}): {by_config.get(spec.config, 0)} cells "
                f"l1_idx={spec.l1_idx} mask={spec.mask_mode} variant={spec.variant}",
                flush=True,
            )
        for spec in FU2_CONDITIONAL_CONFIGS:
            print(
                f"  {spec.config} ({spec.slug}): CONDITIONAL (W2a) — registered, not enumerated",
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
            "refusing the implicit full 28-cell fan-out: pass --fan-out explicitly "
            "(or scope with --pilot / --cells / --smoke / --dry-run)"
        )
    cells = _resolve_fu2_cells(args)
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
