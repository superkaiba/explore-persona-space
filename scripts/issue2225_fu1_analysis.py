#!/usr/bin/env python3
"""Issue #2225 fu rounds — F5 analysis + figures driver (fu1 default; ``--round fu2``).

Round-parametrized at SOURCE (fu2 plan v13 §4.2 artifact-reuse (i) remedy
shape): ``--round fu1`` (the default) is byte-identical to the committed fu1
behavior; ``--round fu2`` runs the SAME instruments against the fu2 roots
(eval root ``fu2_preimage_alltoken``, OVERFLOW-repo ``fu2_capture`` +
``fu2_mmlu`` prefixes — #2287 routing), the fu2 registry (N/Q/RQ + the
conditional RN), and the fu2 §3 contrast set (H1 dose / H2 vs parent G + A /
H3 vs the fu1 banked context-position arms / H4 direction-specificity DoD),
with parent-G rows read from the BANKED parent eval JSONs. The fu2 phase
order drops ``narrow`` (no opinions cells — fu2 plan §2 divergence 1).

Reuses the parent instrument (``scripts/issue2225_analysis.py``) BY IMPORT —
``paired_bootstrap_ci`` / ``matched_coherence_select`` / ``lattice_verdict`` /
``selection_inherited_delta_draws`` / ``_arm_curve`` / the probe + projection
machinery — adding only the fu-specific arm enumeration, the fu contrast
definitions, the d_pre projection leg, and the fu figures.

Phases (``--phase``):
  mmlu        fu MMLU aggregation (staged ``fu1_mmlu`` JSONs) + parent
              base/band reference deltas.
  selection   matched-coherence selection (parent rule, App. J.2) over the
              fu1 registry (8 configs x their corpora x the effective grid);
              full coefficient-response curves.
  contrasts   registered fu contrasts, question-paired bootstrap (n-boot
              resamples, base seed 2225, vectorized numpy):
                H1  Δdose  = score(selected coef) - score(smallest grid coef)
                    per (config x corpus), incl. the 4 random arms (evil).
                H2  Δ_C    = score(fu arm @ its op point) - score(parent C @
                    parent op point) per (pre-image config x corpus); parent
                    side from BANKED parent per-question rows (never
                    recomputed). Secondary: same vs parent A.
                H3  Δdose(pre-image) - Δdose(random) per matched (mask
                    position x layer) pair, evil corpus (exploratory).
              Verdicts keyed to the frozen-CI lattice (parent
              ``lattice_verdict`` partition, fu Effect-negative/positive/tie
              labels); a selection-inherited CI is additionally computed per
              registered contrast as sensitivity. Not-computable cells are
              NAMED in the output JSON, never silently skipped.
  judge       judge-accounting fold (rules 9/24/28/29): per-arm content /
              transport / api-refusal counts + remediation, per-arm
              frac_items_complete vs the parent run's realized floor, plus
              the cap-hit digest.
  probe       parent Gram-space linear ridge probe (fit on the #778 pool,
              GroupKFold over extraction questions) applied over the COMBINED
              capture root (fu1 tags + parent base/baseft anchors); fu-layer
              (14/19) annotation added per fu tag.
  projection  parent r_B/E2 projection-shift monitor over the combined
              capture root, PLUS the fu addition: Δ mean projection onto
              d_pre (the fu direction bank, layers 14/19) per (cell,
              position).
  narrow      narrow-domain mistake-style retention (fu opinions cells) +
              parent per-arm reference.
  figures     fu figure builders (paper-plots conventions; no on-canvas
              caption blocks; one color = one meaning) -> ``--fig-dir``.
  all         mmlu -> selection -> contrasts -> judge -> probe -> projection
              -> narrow -> figures.

Inputs: F4 judge outputs under ``--eval-root``; fu capture summaries staged
from the OVERFLOW repo (``superkaiba1/explore-persona-space-overflow`` —
deviation from the canonical data repo, recorded in the run's results
sentinel); fu directions bank + fu MMLU JSONs + parent base/baseft capture
anchors + parent direction tensors staged from the canonical data repo;
parent banked eval JSONs read from git (``--parent-eval-root``). Outputs:
``eval_results/issue_2225/fu1_preimage_prevention/analysis/*.json`` +
``figures/issue_2225/fu1/``. GPU-free (torch CPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# scripts/ on sys.path so sibling issue2225_* modules resolve in script mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): load_dotenv() setdefaults OMP/MKL/... before
# any numpy/torch import so the caps bind in-process.
load_dotenv()

import issue2225_analysis as pa  # noqa: E402  (parent instrument; light module top)

# ── fu constants ───────────────────────────────────────────────────────────────

OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
FU1_CAPTURE_HF_PREFIX = "issue2225_ctxsteer/analysis_tensors/fu1_capture"  # OVERFLOW repo
FU1_MMLU_HF_PREFIX = "issue2225_ctxsteer/fu1_mmlu"  # canonical data repo
FU2_CAPTURE_HF_PREFIX = "issue2225_ctxsteer/analysis_tensors/fu2_capture"  # OVERFLOW repo
FU2_MMLU_HF_PREFIX = "issue2225_ctxsteer/fu2_mmlu"  # OVERFLOW repo (#2287 routing)
PARENT_ANCHOR_TARGETS = ("base", "baseft_evil", "baseft_sycophancy", "baseft_hallucination")
PROBE_TRAITS = ("evil", "sycophancy", "hallucination")
FU_CONFIG_ORDER = ("J", "K", "L", "M", "RJ", "RK", "RL", "RM")
FU2_CONFIG_ORDER = ("N", "Q", "RQ", "RN")
H3_PAIRS = (("J", "RJ"), ("K", "RK"), ("L", "RL"), ("M", "RM"))
KIND_OFFSET = {"h1_dose": 0, "h2_vs_C": 1, "h2_vs_A": 2, "h3_dose_dod": 3}
# fu2 §3 contrast wiring: H3 = fu2 arm vs the SAME direction at fu1's
# positions (question-paired per (layer, corpus)); H4 = direction-specificity
# DoD (primary Q-RQ; cross-layer N-RQ disclosed).
FU2_H3_FU1_PAIRS = (("N", "J"), ("N", "L"), ("Q", "K"), ("Q", "M"))
FU2_H4_DOD_PAIRS = (("Q", "RQ"), ("N", "RQ"))
# Conditional W2b read (plan v13 §4.3, TRIGGERED 2026-08-14): the matched random
# control RN@L14 all-token materializes on sycophancy ONLY at the W2b window —
# the N arm's selected coef (3.0) plus one lower neighbor (1.5). Files for the
# other registry-grid coefs (0.25/0.75) will NEVER exist, so every W2b consumer
# is file-presence-gated, and the dose window is FIXED at (1.5, 3.0) for BOTH
# arms: matching N's window to RN's makes the two dose windows IDENTICAL —
# unlike H4(a), whose window is [grid-min, selected coef] per arm.
FU2_W2B_PRE, FU2_W2B_RND = "N", "RN"
FU2_W2B_DATASET = "sycophancy"
FU2_W2B_WINDOW = (1.5, 3.0)
FU2_KIND_OFFSET = {
    "h1_dose": 0,
    "h2_vs_G": 1,
    "h2_vs_A": 2,
    "h3_vs_J": 3,
    "h3_vs_K": 4,
    "h3_vs_L": 5,
    "h3_vs_M": 6,
    "h4_dod": 7,
    "h4_dod_xlayer": 8,
    "h4_dod_w2b": 9,  # W2b matched-window DoD (conditional; sycophancy)
    "h4_level_w2b": 10,  # W2b matched-dose LEVEL read (conditional; sycophancy)
}
# G's banked all-token dose curve vs RQ's near-exact effective-dose matches
# (plan §3 H4(b), descriptive): fu2 coef -> parent G coef at matched α_eff.
FU2_H4B_DOSE_MATCHES = ((0.25, 0.5), (0.75, 1.5), (1.5, 3.0))
# Parent lattice labels -> fu labels (plan §3). Partition logic (incl. the
# exhaustive three-way assert) is REUSED from pa.lattice_verdict.
FU_LATTICE_LABEL = {
    "Context-position-superior": "Effect-negative",
    "Context-position-inferior": "Effect-positive",
    "Statistical tie": "Statistical tie",
}

ROUND_EVAL_ROOTS = {
    "fu1": "eval_results/issue_2225/fu1_preimage_prevention",
    "fu2": "eval_results/issue_2225/fu2_preimage_alltoken",
}
ROUND_STAGING_MIRRORS = {
    "fu1": "/mnt/eps-data/thomasjiralerspong/issue2225_fu1/hf_dl",
    "fu2": "/mnt/eps-data/thomasjiralerspong/issue2225_fu2/hf_dl",
}
ROUND_FIG_DIRS = {"fu1": "figures/issue_2225/fu1", "fu2": "figures/issue_2225/fu2"}

# Module-level round context (ONE round per process — a CLI driver; set by
# main() from --round BEFORE any phase runs; default keeps fu1 byte-identical).
_ROUND = "fu1"


def _set_round(name: str) -> None:
    global _ROUND
    if name not in ROUND_EVAL_ROOTS:
        raise ValueError(f"unknown round {name!r} (have {sorted(ROUND_EVAL_ROOTS)})")
    _ROUND = name


def _is_fu2() -> bool:
    return _ROUND == "fu2"


def _fu_train():
    """Deferred sibling import: the ROUND's cell/config registry (heavy-free)."""
    if _is_fu2():
        import issue2225_fu2_train as fu2

        return fu2
    import issue2225_fu1_train as fu1

    return fu1


def _round_specs():
    """The round's wave-1 config specs."""
    mod = _fu_train()
    return mod.FU2_CONFIGS if _is_fu2() else mod.FU1_CONFIGS


def _round_effective_grids(repilot_state_path):
    mod = _fu_train()
    fn = mod.effective_fu2_grids if _is_fu2() else mod.effective_fu1_grids
    return fn(repilot_state_path)


def _round_effective_cells(repilot_state_path):
    mod = _fu_train()
    fn = mod.effective_fu2_cells if _is_fu2() else mod.effective_fu1_cells
    return fn(repilot_state_path)


def _round_resolve(slug: str):
    mod = _fu_train()
    fn = mod.resolve_fu2_cell if _is_fu2() else mod.resolve_fu1_cell
    return fn(slug)


def _round_config_order() -> tuple[str, ...]:
    return FU2_CONFIG_ORDER if _is_fu2() else FU_CONFIG_ORDER


def _round_kind_offsets() -> dict[str, int]:
    return FU2_KIND_OFFSET if _is_fu2() else KIND_OFFSET


def _round_parent_comparators() -> tuple[str, ...]:
    """Parent comparator configs in figures (fu1: C+A; fu2: G+A — plan §3 H2)."""
    return ("G", "A") if _is_fu2() else ("C", "A")


def _round_capture_prefix() -> str:
    return FU2_CAPTURE_HF_PREFIX if _is_fu2() else FU1_CAPTURE_HF_PREFIX


def _round_mmlu_leg() -> tuple[str, str]:
    """(repo, prefix) for the round's staged MMLU JSONs (fu2: OVERFLOW, #2287)."""
    if _is_fu2():
        return (OVERFLOW_REPO, FU2_MMLU_HF_PREFIX)
    return (pa.DATA_REPO, FU1_MMLU_HF_PREFIX)


def _stem(name: str) -> str:
    """Round-prefixed figure stem (fu1 stems byte-identical to the committed run)."""
    return f"{_ROUND}_{name}"


def _fud():
    """Deferred sibling import: the fu1 directions module (MAP_LAYERS etc.)."""
    import issue2225_fu1_directions as fud

    return fud


def fu_lattice_verdict(point: float, lo: float, hi: float) -> str:
    """fu relabeling of the parent's asserted three-way lattice partition."""
    return FU_LATTICE_LABEL[pa.lattice_verdict(point, lo, hi)]


def _contrast_seed(kind: str, config: str, dataset: str) -> int:
    """Deterministic per-(contrast kind, config, dataset) seed stream.

    Base seed 2225 (plan §10) + the parent's 1000*dataset_index convention,
    extended with 100*config_index + a kind offset so distinct registered
    contrasts draw independent (but reproducible) streams. No collisions:
    kind offsets < 100 <= 100*Δci < 1000 <= 1000*Δdi. Round-keyed: fu1 seeds
    are byte-identical to the committed run; fu2 uses its own config order +
    kind-offset table.
    """
    train = pa._train()
    di = train.DATASETS.index(dataset)
    ci = _round_config_order().index(config)
    return pa.BOOTSTRAP_SEED + 1000 * di + 100 * ci + _round_kind_offsets()[kind]


# ── derived paths + arm filters ───────────────────────────────────────────────


def _apply_derived_defaults(args) -> None:
    """Resolve None-defaulted paths off --round + --staging-mirror (fu1
    resolutions byte-identical to the committed argparse defaults)."""
    if args.eval_root is None:
        args.eval_root = ROUND_EVAL_ROOTS[_ROUND]
    if args.staging_mirror is None:
        args.staging_mirror = ROUND_STAGING_MIRRORS[_ROUND]
    if args.fig_dir is None:
        args.fig_dir = ROUND_FIG_DIRS[_ROUND]
    mirror = Path(args.staging_mirror)
    if args.capture_root is None:
        args.capture_root = str(mirror / f"{_ROUND}_capture_combined")
    if args.directions_dir is None:
        args.directions_dir = str(mirror / pa.DIRECTIONS_HF_PREFIX)
    if args.fu_directions_dir is None:
        # BOTH rounds consume the fu1 d_pre bank (fu2 plan §4.1: reused, no build).
        args.fu_directions_dir = str(mirror / _fud().FU1_DIRECTIONS_HF_PREFIX)
    if args.mmlu_dir is None:
        args.mmlu_dir = str(mirror / _round_mmlu_leg()[1])
    if args.i778_staging is None:
        args.i778_staging = str(mirror / "issue778_v2")
    if args.work_root is None:
        args.work_root = str(mirror.parent / "analysis_work")


def _selected_specs(args):
    specs = _round_specs()
    if not args.configs:
        return list(specs)
    known = {s.config for s in specs}
    unknown = sorted(set(args.configs) - known)
    if unknown:
        raise ValueError(f"unknown {_ROUND} config(s) {unknown} (have {sorted(known)})")
    return [s for s in specs if s.config in args.configs]


def _selected_datasets(args, spec) -> list[str]:
    if not args.datasets:
        return list(spec.datasets)
    train = pa._train()
    unknown = sorted(set(args.datasets) - set(train.DATASETS))
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown} (have {list(train.DATASETS)})")
    return [d for d in spec.datasets if d in args.datasets]


def _subset_note(args) -> dict:
    if args.configs or args.datasets:
        note = (
            f"SUBSET run (configs={args.configs or 'all'}, datasets={args.datasets or 'all'}) "
            "— output covers only the subset (smoke dispatch shape; production runs unfiltered)"
        )
        print(f"[{_ROUND}-analysis] {note}", flush=True)
        return {"subset_note": note}
    return {}


# ── input staging ─────────────────────────────────────────────────────────────


def stage_fu_inputs(args) -> None:
    """Stage every HF-resident input into the mirror root (idempotent).

    fu capture summaries live on the OVERFLOW repo (deviation from the
    canonical data repo — recorded in the run's results sentinel); everything
    else on the canonical data repo. Parent base/baseft capture anchors are
    staged because the probe/projection shift reads need base rows and the
    probe sanity gate needs baseft rows — fu1_capture carries no base target.
    """
    from explore_persona_space.orchestrate.hub import stage_hub_prefix

    mirror = Path(args.staging_mirror)
    legs = [
        (OVERFLOW_REPO, _round_capture_prefix()),
        (pa.DATA_REPO, _fud().FU1_DIRECTIONS_HF_PREFIX),
        _round_mmlu_leg(),
        (pa.DATA_REPO, pa.DIRECTIONS_HF_PREFIX),
    ]
    legs += [(pa.DATA_REPO, f"{pa.CAPTURE_HF_PREFIX}/{t}") for t in PARENT_ANCHOR_TARGETS]
    for repo, prefix in legs:
        resolved = mirror / prefix
        print(f"[stage] {repo}:{prefix} -> {resolved}", flush=True)
        stage_hub_prefix(repo, prefix, mirror, repo_type="dataset")
        if not resolved.is_dir():
            raise RuntimeError(
                f"staging arithmetic violated: {resolved} absent after stage_hub_prefix "
                "(dest_dir is a mirror root — hub.stage_hub_prefix contract)"
            )
    print(f"[stage] {_ROUND} inputs staged (fu capture source = OVERFLOW repo)", flush=True)


def _symlink_into(root: Path, name: str, src: Path) -> None:
    link = root / name
    if link.is_symlink() or link.exists():
        return
    link.symlink_to(src.resolve(), target_is_directory=True)


def ensure_combined_capture_root(args) -> Path:
    """Combined capture root = fu1 tags + parent base/baseft anchors.

    Built as a symlink farm from the staging mirror when the mirror sources
    exist; a pre-populated --capture-root (pod layout) is used as-is. Always
    checked: every entry has a summary_manifest.json, the 4 parent anchors
    are present, and (unless --allow-partial-capture) every effective fu1
    cell slug has a capture dir.
    """
    root = Path(args.capture_root)
    mirror = Path(args.staging_mirror)
    fu_src = mirror / _round_capture_prefix()
    parent_src = mirror / pa.CAPTURE_HF_PREFIX
    if fu_src.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        for src in sorted(p for p in fu_src.iterdir() if p.is_dir()):
            _symlink_into(root, src.name, src)
        for name in PARENT_ANCHOR_TARGETS:
            src = parent_src / name
            if not src.is_dir():
                raise FileNotFoundError(
                    f"parent anchor capture absent: {src} — run --stage-inputs "
                    "(parent base/baseft legs)"
                )
            _symlink_into(root, name, src)
    if not root.is_dir() or not any(root.iterdir()):
        raise FileNotFoundError(
            f"capture root {root} empty and no staged mirror at {fu_src} — run --stage-inputs"
        )
    entries = {p.name for p in root.iterdir() if p.is_dir()}
    for name in sorted(entries):
        manifest = root / name / "summary_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"capture manifest missing: {manifest}")
    missing_anchors = [t for t in PARENT_ANCHOR_TARGETS if t not in entries]
    if missing_anchors:
        raise FileNotFoundError(f"combined capture root missing parent anchors: {missing_anchors}")
    expected = {c.slug for c in _round_effective_cells(args.repilot_state)}
    missing = sorted(expected - entries)
    if missing:
        if not args.allow_partial_capture:
            raise FileNotFoundError(
                f"combined capture root missing {len(missing)}/{len(expected)} {_ROUND} cell "
                f"captures (first: {missing[:4]}) — stage the full {_ROUND}_capture prefix or "
                "pass --allow-partial-capture (smoke only)"
            )
        print(
            f"[capture-root] PARTIAL: {len(missing)}/{len(expected)} {_ROUND} captures absent "
            "(--allow-partial-capture)",
            flush=True,
        )
    return root


# ── phase: mmlu (+ parent reference) ──────────────────────────────────────────


def run_fu_mmlu(args) -> Path:
    out = pa.run_mmlu(args)  # writes <eval_root>/analysis/mmlu.json
    data = json.load(open(out))
    parent_mmlu = pa._load_json(Path(args.parent_eval_root) / "analysis" / "mmlu.json")
    ppt = parent_mmlu["per_target"]
    vals = [v["mmlu_acc"] for v in ppt.values() if v.get("mmlu_acc") is not None]
    base_acc = ppt.get("base", {}).get("mmlu_acc")
    if base_acc is None or not vals:
        raise ValueError("parent mmlu.json lacks a base row / any accuracy values")
    data["parent_reference"] = {
        "base_mmlu_acc": base_acc,
        "band_min": min(vals),
        "band_max": max(vals),
        "n_parent_targets": len(vals),
    }
    data["deltas_vs_parent_base"] = {
        tag: (None if row.get("mmlu_acc") is None else row["mmlu_acc"] - base_acc)
        for tag, row in data["per_target"].items()
    }
    pa._atomic_write_json(Path(out), data)
    print(f"[{_ROUND}-mmlu] parent reference (base={base_acc:.4f}) folded -> {out}", flush=True)
    return out


# ── phase: selection ──────────────────────────────────────────────────────────


def _selection_curve(eval_root: Path, mmlu: dict, config: str, dataset: str, trait, coefs) -> dict:
    """Per-coefficient selection-curve rows for one (config, dataset) arm
    (extracted from the main loop verbatim — value-identical outputs)."""
    curve = {}
    for coef in coefs:
        tag = f"{config}__{dataset}__c{coef}"
        trait_arm = pa._load_json(pa._arm_path(eval_root, "trait_scores", config, dataset, coef))
        coh_arm = pa._load_json(pa._arm_path(eval_root, "coherence", config, dataset, coef))
        tb = trait_arm["traits"][trait]
        curve[str(coef)] = {
            "trait_mean": tb["model_mean"],
            "rate_gt50": tb["rate_gt50"],
            "coherence_mean": pa._arm_coherence_mean(coh_arm),
            "mmlu_acc": mmlu.get(tag, {}).get("mmlu_acc"),
            "n_api_refusal": tb["accounting"]["n_api_refusal"],
        }
    return curve


def _fu2_conditional_selection(args, grids, mmlu, selection) -> None:
    """Fold the conditional fu2 arm(s) (RN — plan §4.3) into selection,
    tolerating their PARTIAL grids.

    Conditional cells materialize per (dataset, coef) only when their trigger
    fires: RN_sycophancy (W2b) exists ONLY at the FU2_W2B_WINDOW coefs — the
    other registry-grid coefs' files will NEVER land — so missing coef files
    are SKIPPED here for the conditional specs alone (wave-1 configs keep the
    main loop's fail-loud full-grid contract). A (config, dataset) with zero
    landed coefs is skipped with a log line, so pre-harvest re-runs leave
    selection.json byte-identical.
    """
    train = pa._train()
    eval_root = Path(args.eval_root)
    for spec in _fu_train().FU2_CONDITIONAL_CONFIGS:
        for dataset in _selected_datasets(args, spec):
            trait = train.STEERED_TRAIT[dataset]
            landed = []
            for coef in grids[spec.config]:
                has_trait = pa._arm_path(
                    eval_root, "trait_scores", spec.config, dataset, coef
                ).exists()
                has_coh = pa._arm_path(eval_root, "coherence", spec.config, dataset, coef).exists()
                if has_trait and has_coh:
                    landed.append(coef)
                elif has_trait or has_coh:
                    # Half-landed pair: one of the two eval files exists. Not
                    # binding here (nothing downstream consumes the conditional
                    # selected_coef), but never a silent drop — the binding
                    # contrast's all-4-file gate raises on the same state
                    # (code-review minor 2a, cefce522 round).
                    print(
                        f"[{_ROUND}-selection] conditional {spec.config}_{dataset} "
                        f"c={coef}: HALF-landed pair (trait_scores={has_trait}, "
                        f"coherence={has_coh}) — coef excluded from curve",
                        flush=True,
                    )
            if not landed:
                print(
                    f"[{_ROUND}-selection] conditional {spec.config}_{dataset}: "
                    "no landed cells — skipped",
                    flush=True,
                )
                continue
            curve = _selection_curve(eval_root, mmlu, spec.config, dataset, trait, landed)
            selected = pa.matched_coherence_select(
                {float(c): v["coherence_mean"] for c, v in curve.items()}
            )
            note = None if selected is not None else "NO coefficient reaches coherence >= 80"
            selection[f"{spec.config}_{dataset}"] = {
                "config": spec.config,
                "dataset": dataset,
                "steered_trait": trait,
                "grid": [float(c) for c in landed],
                "selected_coef": selected,
                "curve": curve,
                "conditional_partial_grid": (
                    f"conditional arm (plan §4.3): curve covers ONLY the landed coefs "
                    f"{[float(c) for c in landed]} of the registry grid "
                    f"{[float(c) for c in grids[spec.config]]} — non-window coefs are "
                    "never dispatched (no eval files by design); a half-landed "
                    "window coef is logged and excluded, and the binding contrast's "
                    "all-4-file gate raises on it"
                ),
                **({"note": note} if note else {}),
            }


def run_fu_selection(args) -> Path:
    train = pa._train()
    eval_root = Path(args.eval_root)
    mmlu_path = eval_root / "analysis" / "mmlu.json"
    mmlu = json.load(open(mmlu_path))["per_target"] if mmlu_path.exists() else {}
    grids = _round_effective_grids(args.repilot_state)
    selection: dict[str, dict] = {}
    for spec in _selected_specs(args):
        for dataset in _selected_datasets(args, spec):
            trait = train.STEERED_TRAIT[dataset]
            curve = _selection_curve(
                eval_root, mmlu, spec.config, dataset, trait, grids[spec.config]
            )
            selected = pa.matched_coherence_select(
                {float(c): v["coherence_mean"] for c, v in curve.items()}
            )
            note = None if selected is not None else "NO coefficient reaches coherence >= 80"
            selection[f"{spec.config}_{dataset}"] = {
                "config": spec.config,
                "dataset": dataset,
                "steered_trait": trait,
                "grid": [float(c) for c in grids[spec.config]],
                "selected_coef": selected,
                "curve": curve,
                **({"note": note} if note else {}),
            }
    if _is_fu2() and not args.configs:
        # Conditional RN (plan §4.3): fold in LANDED conditional cells only —
        # subset (--configs) smoke runs skip the conditional arm entirely.
        _fu2_conditional_selection(args, grids, mmlu, selection)
    out = eval_root / "analysis" / "selection.json"
    pa._atomic_write_json(
        out,
        {
            "rule": "largest grid coefficient with mean coherence >= 80 "
            "(paper App. J.2, parent matched_coherence_select), applied identically "
            f"to every {_ROUND} arm",
            "coherence_threshold": pa.COHERENCE_THRESHOLD,
            "selection": selection,
            **_subset_note(args),
        },
    )
    n_null = sum(1 for v in selection.values() if v["selected_coef"] is None)
    print(
        f"[{_ROUND}-selection] {len(selection)} arms; {n_null} without selection -> {out}",
        flush=True,
    )
    return out


# ── phase: contrasts ──────────────────────────────────────────────────────────


def _paired_contrast(qx: dict, qy: dict, *, seed: int, n_boot: int, inherited_fn=None) -> dict:
    """One question-paired contrast: frozen CI (+ optional selection-inherited).

    qx/qy: {question_idx: mean}. Reuses pa.paired_bootstrap_ci and the parent's
    shared-idx convention (the inherited CI re-draws the SAME idx stream).
    """
    import numpy as np

    if set(qx) != set(qy):
        raise ValueError("question sets differ between the two contrast sides")
    q_ids = sorted(qx)
    delta_q = np.array([qx[q] - qy[q] for q in q_ids], dtype=np.float64)
    if np.isnan(delta_q).any():
        raise ValueError("NaN per-question means in a registered contrast")
    point, lo, hi, _ = pa.paired_bootstrap_ci(delta_q, n_boot, seed)
    out = {
        "n_questions": len(q_ids),
        "seed": seed,
        "frozen": {
            "delta_point": point,
            "ci95": [lo, hi],
            "verdict": fu_lattice_verdict(point, lo, hi),
        },
    }
    if inherited_fn is not None:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(q_ids), size=(n_boot, len(q_ids)))
        inh, n_invalid = inherited_fn(idx)
        valid = inh[~np.isnan(inh)]
        out["selection_inherited"] = {
            "delta_point": float(np.nanmean(inh)) if valid.size else None,
            "ci95": (
                [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))]
                if valid.size
                else None
            ),
            "n_draws_no_coherent_coef": n_invalid,
            # plan §3: the selection-inherited flavour is a sensitivity
            # diagnostic and never bears a verdict label
            "sensitivity_read": (
                fu_lattice_verdict(
                    float(np.nanmean(inh)),
                    float(np.percentile(valid, 2.5)),
                    float(np.percentile(valid, 97.5)),
                )
                if valid.size
                else "not-computable"
            ),
        }
    return out


def run_contrasts(args) -> Path:
    """Round dispatch: the fu1 registered battery, or the fu2 §3 battery."""
    if _is_fu2():
        return run_fu2_contrasts(args)
    return run_fu_contrasts(args)


def run_fu_contrasts(args) -> Path:
    import numpy as np

    train = pa._train()
    fu_root = Path(args.eval_root)
    parent_root = Path(args.parent_eval_root)
    sel_fu = json.load(open(fu_root / "analysis" / "selection.json"))["selection"]
    sel_parent = json.load(open(parent_root / "analysis" / "selection.json"))["selection"]
    grids = _round_effective_grids(args.repilot_state)
    n_boot = args.n_boot
    not_computable: list[str] = []

    def _fu_q(config, dataset, coef, trait):
        arm = pa._load_json(pa._arm_path(fu_root, "trait_scores", config, dataset, coef))
        return pa._per_question_means(arm["traits"][trait])

    def _parent_q(config, dataset, coef, trait):
        arm = pa._load_json(pa._arm_path(parent_root, "trait_scores", config, dataset, coef))
        return pa._per_question_means(arm["traits"][trait])

    def _dose_inherited_fn(config, dataset, trait, n_q):
        grid = [float(c) for c in grids[config]]
        c_min = min(grid)
        arm = pa._arm_curve(fu_root, config, dataset, grid, trait, n_q)
        arm_min = pa._arm_curve(fu_root, config, dataset, [c_min], trait, n_q)

        def fn(idx):
            return pa.selection_inherited_delta_draws(arm, arm_min, idx)

        return fn

    h1: dict[str, dict] = {}
    h2c: dict[str, dict] = {}
    h2a: dict[str, dict] = {}
    h3: dict[str, dict] = {}
    specs = _selected_specs(args)
    for spec in specs:
        for dataset in _selected_datasets(args, spec):
            trait = train.STEERED_TRAIT[dataset]
            key = f"{spec.config}_{dataset}"
            entry = sel_fu.get(key)
            if entry is None:
                raise KeyError(f"{key} absent from fu selection.json — re-run --phase selection")
            sel_c = entry["selected_coef"]
            grid = [float(c) for c in grids[spec.config]]
            c_min = min(grid)
            # H1 Δdose (every config, incl. random arms)
            if sel_c is None:
                h1[key] = {
                    "verdict": "not-computable (no coherent coefficient)",
                    "selected_coef": None,
                }
                not_computable.append(f"h1_dose:{key}")
            else:
                qx = _fu_q(spec.config, dataset, sel_c, trait)
                qy = _fu_q(spec.config, dataset, c_min, trait)
                h1[key] = _paired_contrast(
                    qx,
                    qy,
                    seed=_contrast_seed("h1_dose", spec.config, dataset),
                    n_boot=n_boot,
                    inherited_fn=_dose_inherited_fn(spec.config, dataset, trait, len(qx)),
                ) | {
                    "selected_coef": sel_c,
                    "smallest_coef": c_min,
                    "degenerate_zero_dose": bool(sel_c == c_min),
                }
            # H2 vs parent C / secondary vs parent A (pre-image configs only)
            if spec.variant == "PRE":
                for ycfg, kind, bucket in (("C", "h2_vs_C", h2c), ("A", "h2_vs_A", h2a)):
                    p_entry = sel_parent.get(f"{ycfg}_{dataset}")
                    if p_entry is None:
                        raise KeyError(f"parent selection missing {ycfg}_{dataset}")
                    sel_y = p_entry["selected_coef"]
                    if sel_c is None or sel_y is None:
                        bucket[key] = {
                            "verdict": "not-computable (an arm has no coherent coefficient)",
                            "selected": {spec.config: sel_c, ycfg: sel_y},
                        }
                        not_computable.append(f"{kind}:{key}")
                        continue
                    qx = _fu_q(spec.config, dataset, sel_c, trait)
                    qy = _parent_q(ycfg, dataset, sel_y, trait)
                    n_q = len(qx)
                    arm_x = pa._arm_curve(fu_root, spec.config, dataset, grid, trait, n_q)
                    arm_y = pa._arm_curve(parent_root, ycfg, dataset, p_entry["grid"], trait, n_q)

                    def _h2_fn(idx, _ax=arm_x, _ay=arm_y):
                        return pa.selection_inherited_delta_draws(_ax, _ay, idx)

                    bucket[key] = _paired_contrast(
                        qx,
                        qy,
                        seed=_contrast_seed(kind, spec.config, dataset),
                        n_boot=n_boot,
                        inherited_fn=_h2_fn,
                    ) | {"selected": {spec.config: sel_c, ycfg: sel_y}}
    # H3 difference-of-differences (evil, exploratory)
    selected_cfgs = {s.config for s in specs}
    for pre_cfg, rnd_cfg in H3_PAIRS:
        if pre_cfg not in selected_cfgs or rnd_cfg not in selected_cfgs:
            continue
        if args.datasets and "evil" not in args.datasets:
            continue
        dataset, trait = "evil", "evil"
        key = f"{pre_cfg}_vs_{rnd_cfg}_{dataset}"
        sel_pre = sel_fu[f"{pre_cfg}_{dataset}"]["selected_coef"]
        sel_rnd = sel_fu[f"{rnd_cfg}_{dataset}"]["selected_coef"]
        if sel_pre is None or sel_rnd is None:
            h3[key] = {
                "verdict": "not-computable (an arm has no coherent coefficient)",
                "selected": {pre_cfg: sel_pre, rnd_cfg: sel_rnd},
            }
            not_computable.append(f"h3:{key}")
            continue
        c_min_pre = min(float(c) for c in grids[pre_cfg])
        c_min_rnd = min(float(c) for c in grids[rnd_cfg])
        q_pre_x = _fu_q(pre_cfg, dataset, sel_pre, trait)
        q_pre_y = _fu_q(pre_cfg, dataset, c_min_pre, trait)
        q_rnd_x = _fu_q(rnd_cfg, dataset, sel_rnd, trait)
        q_rnd_y = _fu_q(rnd_cfg, dataset, c_min_rnd, trait)
        dose_pre = {q: q_pre_x[q] - q_pre_y[q] for q in q_pre_x}
        dose_rnd = {q: q_rnd_x[q] - q_rnd_y[q] for q in q_rnd_x}
        n_q = len(dose_pre)
        fn_pre = _dose_inherited_fn(pre_cfg, dataset, trait, n_q)
        fn_rnd = _dose_inherited_fn(rnd_cfg, dataset, trait, n_q)

        def _h3_fn(idx, _fp=fn_pre, _fr=fn_rnd):
            d_pre, _ = _fp(idx)
            d_rnd, _ = _fr(idx)
            dd = d_pre - d_rnd  # NaN propagates: either side without a coherent coef
            return dd, int(np.isnan(dd).sum())

        h3[key] = _paired_contrast(
            dose_pre,
            dose_rnd,
            seed=_contrast_seed("h3_dose_dod", pre_cfg, dataset),
            n_boot=n_boot,
            inherited_fn=_h3_fn,
        ) | {"selected": {pre_cfg: sel_pre, rnd_cfg: sel_rnd}}
    out = fu_root / "analysis" / "contrasts.json"
    pa._atomic_write_json(
        out,
        {
            "n_boot": n_boot,
            "seed_base": pa.BOOTSTRAP_SEED,
            "seed_scheme": "2225 + 1000*dataset_index + 100*config_index + kind offset "
            f"{KIND_OFFSET} (parent per-dataset convention extended per contrast; "
            "frozen + selection-inherited share one idx stream per contrast)",
            "lattice": "Effect-negative <=> Δ<0 AND 95% CI wholly below 0; "
            "Effect-positive <=> Δ>0 AND CI wholly above 0; Statistical tie <=> "
            "otherwise (fu relabeling of the parent's asserted three-way partition)",
            "h1_dose": {
                "delta_definition": "score(matched-coherence selected coef) - score(smallest "
                "grid coef), question-paired within arm; Effect-positive = trait score rises "
                "with dose. degenerate_zero_dose marks arms whose selected coef IS the "
                "smallest (Δ identically 0)",
                "inherited_caveat": "the smallest-coef side is ALSO coherence-gated inside "
                "each resample under the reused parent machinery (sensitivity-only deviation "
                "from the frozen H1 definition)",
                "per_arm": h1,
            },
            "h2_vs_parent_C": {
                "delta_definition": "score(fu arm @ its op point) - score(parent C @ parent "
                "op point), question-paired; parent side from BANKED parent per-question "
                "rows. Effect-negative = pre-image arm induces LESS trait expression than "
                "parent C (prevents better)",
                "per_arm": h2c,
            },
            "h2_secondary_vs_parent_A": {
                "delta_definition": "same as h2_vs_parent_C with parent A (paper method) as "
                "the reference side (secondary)",
                "per_arm": h2a,
            },
            "h3_dose_dod_evil": {
                "delta_definition": "Δdose(pre-image) - Δdose(random control) per matched "
                "(mask position x layer) pair, evil corpus, question-paired (EXPLORATORY)",
                "per_pair": h3,
            },
            "not_computable": not_computable,
            "single_seed_caveat": "per-arm verdicts are SINGLE-TRAINING-SEED claims (seed 0; "
            "the CI carries zero training-draw variance)",
            **_subset_note(args),
        },
    )
    print(
        f"[fu1-contrasts] h1={len(h1)} h2C={len(h2c)} h2A={len(h2a)} h3={len(h3)} "
        f"not_computable={len(not_computable)} -> {out}",
        flush=True,
    )
    return out


def fu2_w2b_contrasts(fu_root: Path, n_boot: int, not_computable: list[str]) -> dict[str, dict]:
    """The conditional H4(c) W2b direction-specificity reads (plan v13 §4.3).

    Both reads are gated on the RN sycophancy W2b-window eval files existing
    (the cells materialize only after the W2b trigger fires): when any is
    absent, one log line + both keys recorded in ``not_computable`` and an
    empty dict returned — an F5 re-run BEFORE the W2b harvest exits 0.

    (1) key ``N_vs_RN_sycophancy`` (kind ``h4_dod_w2b``): MATCHED-WINDOW dose
        DoD — dose_arm = score(@3.0) − score(@1.5) per question;
        Δ = dose(N) − dose(RN), question-paired via the same
        ``_paired_contrast`` + inherited-fn machinery as H4(a). The window is
        [1.5, 3.0] for BOTH arms because RN exists ONLY at the W2b "selected
        coef ± one neighbor" cells; matching N's window to it makes the two
        dose windows IDENTICAL — unlike H4(a)'s per-arm [grid-min, selected]
        window (see FU2_W2B_WINDOW).
    (2) key ``N_vs_RN_sycophancy_level`` (kind ``h4_level_w2b``): secondary
        matched-dose LEVEL read — trait mean N@3.0 vs RN@3.0, same paired
        machinery, no dose subtraction (the direct "does random do the same
        at the same dose" read; no selection-inherited flavour — the dose is
        fixed by the window, not selected).
    """
    import numpy as np

    train = pa._train()
    pre_cfg, rnd_cfg, dataset = FU2_W2B_PRE, FU2_W2B_RND, FU2_W2B_DATASET
    trait = train.STEERED_TRAIT[dataset]
    c_lo, c_hi = (float(c) for c in FU2_W2B_WINDOW)
    key = f"{pre_cfg}_vs_{rnd_cfg}_{dataset}"
    missing = [
        p
        for c in (c_lo, c_hi)
        for p in (
            pa._arm_path(fu_root, "trait_scores", rnd_cfg, dataset, c),
            pa._arm_path(fu_root, "coherence", rnd_cfg, dataset, c),
        )
        if not p.exists()
    ]
    if missing:
        if len(missing) < 4:
            # Pre-harvest is exactly 0-of-4 present; 1-3 missing means the W2b
            # harvest HALF-landed (e.g. one cell's coherence upload failed) —
            # fail loud rather than mislabel it "not yet landed" (code-review
            # minor 1, cefce522 round).
            raise RuntimeError(
                f"[fu2-contrasts] W2b RN window PARTIALLY landed — {len(missing)}/4 "
                f"window files missing: {', '.join(p.name for p in missing)}; "
                "a half-landed harvest is an upload/harvest defect, not a "
                "pre-harvest state"
            )
        print(
            f"[fu2-contrasts] W2b cells not yet landed ({len(missing)} files, "
            f"first: {missing[0].name}) — h4(c) skipped",
            flush=True,
        )
        not_computable.append(f"h4_dod_w2b:{key} (W2b cells not yet landed)")
        not_computable.append(f"h4_level_w2b:{key}_level (W2b cells not yet landed)")
        return {}

    def _qq(config, coef):
        arm = pa._load_json(pa._arm_path(fu_root, "trait_scores", config, dataset, coef))
        return pa._per_question_means(arm["traits"][trait])

    q_pre_hi, q_pre_lo = _qq(pre_cfg, c_hi), _qq(pre_cfg, c_lo)
    q_rnd_hi, q_rnd_lo = _qq(rnd_cfg, c_hi), _qq(rnd_cfg, c_lo)
    dose_pre = {q: q_pre_hi[q] - q_pre_lo[q] for q in q_pre_hi}
    dose_rnd = {q: q_rnd_hi[q] - q_rnd_lo[q] for q in q_rnd_hi}
    n_q = len(dose_pre)
    window = [c_lo, c_hi]

    def _window_dose_fn(config):
        # H4(a)'s `_dose_inherited_fn` shape, restricted to the SHARED window
        # (a full-grid curve would demand RN files that never exist).
        arm = pa._arm_curve(fu_root, config, dataset, window, trait, n_q)
        arm_lo = pa._arm_curve(fu_root, config, dataset, [c_lo], trait, n_q)

        def fn(idx):
            return pa.selection_inherited_delta_draws(arm, arm_lo, idx)

        return fn

    fn_pre = _window_dose_fn(pre_cfg)
    fn_rnd = _window_dose_fn(rnd_cfg)

    def _w2b_dod_fn(idx):
        d_pre, _ = fn_pre(idx)
        d_rnd, _ = fn_rnd(idx)
        dd = d_pre - d_rnd  # NaN propagates: either side without a coherent coef
        return dd, int(np.isnan(dd).sum())

    return {
        key: _paired_contrast(
            dose_pre,
            dose_rnd,
            seed=_contrast_seed("h4_dod_w2b", pre_cfg, dataset),
            n_boot=n_boot,
            inherited_fn=_w2b_dod_fn,
        )
        | {"window": window},
        f"{key}_level": _paired_contrast(
            q_pre_hi,
            q_rnd_hi,
            seed=_contrast_seed("h4_level_w2b", pre_cfg, dataset),
            n_boot=n_boot,
        )
        | {"matched_coef": c_hi},
    }


def run_fu2_contrasts(args) -> Path:
    """The fu2 §3 registered battery (plan v13; runs under ``--round fu2``).

    H1  Δdose per fu2 arm (N/Q x 3 corpora + RQ evil), fu1 definition verbatim.
    H2  primary: Δ_G = score(Q_evil @ its op) - score(parent G_evil @ parent
        op) — the round's headline contrast; cross-layer secondary N_evil vs
        G_evil (disclosed); parent side from BANKED parent per-question rows.
        Secondary vs parent A: fu1's ``h2_secondary_vs_parent_A`` machinery
        verbatim (PRE configs x 3 corpora).
    H3  position effect at matched direction: fu2 arm vs the SAME d_pre at
        fu1's positions — (N,J), (N,L) at L14; (Q,K), (Q,M) at L19 — per
        corpus, question-paired; fu1 side from BANKED fu1 rows @ fu1 op points.
    H4  direction-specificity: (a) DoD Δdose(Q)-Δdose(RQ) on evil (primary;
        N-RQ cross-layer disclosed); (b) descriptive matched-effective-dose
        table RQ vs G's banked curve (no CI — plan §3 H4(b)); (c) conditional
        W2b matched-window reads on sycophancy, N vs RN (plan §4.3 —
        file-presence gated, see ``fu2_w2b_contrasts``).
    One frozen CI convention (n-boot resamples, seed scheme in the output);
    the selection-inherited flavour rides H1/H2/H4(a) as sensitivity.
    """
    import numpy as np

    train = pa._train()
    fu_root = Path(args.eval_root)
    parent_root = Path(args.parent_eval_root)
    fu1_root = Path(args.fu1_eval_root)
    sel_fu = json.load(open(fu_root / "analysis" / "selection.json"))["selection"]
    sel_parent = json.load(open(parent_root / "analysis" / "selection.json"))["selection"]
    sel_fu1 = json.load(open(fu1_root / "analysis" / "selection.json"))["selection"]
    grids = _round_effective_grids(args.repilot_state)
    n_boot = args.n_boot
    not_computable: list[str] = []

    def _q(root, config, dataset, coef, trait):
        arm = pa._load_json(pa._arm_path(root, "trait_scores", config, dataset, coef))
        return pa._per_question_means(arm["traits"][trait])

    def _dose_inherited_fn(config, dataset, trait, n_q):
        grid = [float(c) for c in grids[config]]
        c_min = min(grid)
        arm = pa._arm_curve(fu_root, config, dataset, grid, trait, n_q)
        arm_min = pa._arm_curve(fu_root, config, dataset, [c_min], trait, n_q)

        def fn(idx):
            return pa.selection_inherited_delta_draws(arm, arm_min, idx)

        return fn

    h1: dict[str, dict] = {}
    h2g: dict[str, dict] = {}
    h2a: dict[str, dict] = {}
    h3: dict[str, dict] = {}
    h4: dict[str, dict] = {}
    specs = _selected_specs(args)
    for spec in specs:
        for dataset in _selected_datasets(args, spec):
            trait = train.STEERED_TRAIT[dataset]
            key = f"{spec.config}_{dataset}"
            entry = sel_fu.get(key)
            if entry is None:
                raise KeyError(f"{key} absent from fu2 selection.json — re-run --phase selection")
            sel_c = entry["selected_coef"]
            grid = [float(c) for c in grids[spec.config]]
            c_min = min(grid)
            # H1 Δdose (every config, incl. the random arm)
            if sel_c is None:
                h1[key] = {
                    "verdict": "not-computable (no coherent coefficient)",
                    "selected_coef": None,
                }
                not_computable.append(f"h1_dose:{key}")
            else:
                qx = _q(fu_root, spec.config, dataset, sel_c, trait)
                qy = _q(fu_root, spec.config, dataset, c_min, trait)
                h1[key] = _paired_contrast(
                    qx,
                    qy,
                    seed=_contrast_seed("h1_dose", spec.config, dataset),
                    n_boot=n_boot,
                    inherited_fn=_dose_inherited_fn(spec.config, dataset, trait, len(qx)),
                ) | {
                    "selected_coef": sel_c,
                    "smallest_coef": c_min,
                    "degenerate_zero_dose": bool(sel_c == c_min),
                }
            if spec.variant != "PRE":
                continue
            # H2 vs parent G (evil only: Q primary, N cross-layer secondary) +
            # secondary vs parent A (all corpora) — fu1's H2 machinery shape.
            h2_targets = [("A", "h2_vs_A", h2a)]
            if dataset == "evil":
                h2_targets.insert(0, ("G", "h2_vs_G", h2g))
            for ycfg, kind, bucket in h2_targets:
                p_entry = sel_parent.get(f"{ycfg}_{dataset}")
                if p_entry is None:
                    raise KeyError(f"parent selection missing {ycfg}_{dataset}")
                sel_y = p_entry["selected_coef"]
                if sel_c is None or sel_y is None:
                    bucket[key] = {
                        "verdict": "not-computable (an arm has no coherent coefficient)",
                        "selected": {spec.config: sel_c, ycfg: sel_y},
                    }
                    not_computable.append(f"{kind}:{key}")
                    continue
                qx = _q(fu_root, spec.config, dataset, sel_c, trait)
                qy = _q(parent_root, ycfg, dataset, sel_y, trait)
                n_q = len(qx)
                arm_x = pa._arm_curve(fu_root, spec.config, dataset, grid, trait, n_q)
                arm_y = pa._arm_curve(parent_root, ycfg, dataset, p_entry["grid"], trait, n_q)

                def _h2_fn(idx, _ax=arm_x, _ay=arm_y):
                    return pa.selection_inherited_delta_draws(_ax, _ay, idx)

                bucket[key] = _paired_contrast(
                    qx,
                    qy,
                    seed=_contrast_seed(kind, spec.config, dataset),
                    n_boot=n_boot,
                    inherited_fn=_h2_fn,
                ) | {"selected": {spec.config: sel_c, ycfg: sel_y}}
            # H3 position effect vs the fu1 banked arms (matched direction/layer)
            for fu2_cfg, fu1_cfg in FU2_H3_FU1_PAIRS:
                if fu2_cfg != spec.config:
                    continue
                kind = f"h3_vs_{fu1_cfg}"
                key3 = f"{spec.config}_vs_{fu1_cfg}_{dataset}"
                fu1_entry = sel_fu1.get(f"{fu1_cfg}_{dataset}")
                if fu1_entry is None:
                    raise KeyError(f"fu1 selection missing {fu1_cfg}_{dataset} ({fu1_root})")
                sel_y = fu1_entry["selected_coef"]
                if sel_c is None or sel_y is None:
                    h3[key3] = {
                        "verdict": "not-computable (an arm has no coherent coefficient)",
                        "selected": {spec.config: sel_c, fu1_cfg: sel_y},
                    }
                    not_computable.append(f"{kind}:{key3}")
                    continue
                qx = _q(fu_root, spec.config, dataset, sel_c, trait)
                qy = _q(fu1_root, fu1_cfg, dataset, sel_y, trait)
                h3[key3] = _paired_contrast(
                    qx,
                    qy,
                    seed=_contrast_seed(kind, spec.config, dataset),
                    n_boot=n_boot,
                ) | {"selected": {spec.config: sel_c, fu1_cfg: sel_y}}
    # H4(a) direction-specificity DoD (evil; Q-RQ primary, N-RQ cross-layer)
    selected_cfgs = {s.config for s in specs}
    for pre_cfg, rnd_cfg in FU2_H4_DOD_PAIRS:
        if pre_cfg not in selected_cfgs or rnd_cfg not in selected_cfgs:
            continue
        if args.datasets and "evil" not in args.datasets:
            continue
        dataset, trait = "evil", "evil"
        kind = "h4_dod" if pre_cfg == "Q" else "h4_dod_xlayer"
        key = f"{pre_cfg}_vs_{rnd_cfg}_{dataset}"
        sel_pre = sel_fu[f"{pre_cfg}_{dataset}"]["selected_coef"]
        sel_rnd = sel_fu[f"{rnd_cfg}_{dataset}"]["selected_coef"]
        if sel_pre is None or sel_rnd is None:
            h4[key] = {
                "verdict": "not-computable (an arm has no coherent coefficient)",
                "selected": {pre_cfg: sel_pre, rnd_cfg: sel_rnd},
            }
            not_computable.append(f"{kind}:{key}")
            continue
        c_min_pre = min(float(c) for c in grids[pre_cfg])
        c_min_rnd = min(float(c) for c in grids[rnd_cfg])
        q_pre_x = _q(fu_root, pre_cfg, dataset, sel_pre, trait)
        q_pre_y = _q(fu_root, pre_cfg, dataset, c_min_pre, trait)
        q_rnd_x = _q(fu_root, rnd_cfg, dataset, sel_rnd, trait)
        q_rnd_y = _q(fu_root, rnd_cfg, dataset, c_min_rnd, trait)
        dose_pre = {q: q_pre_x[q] - q_pre_y[q] for q in q_pre_x}
        dose_rnd = {q: q_rnd_x[q] - q_rnd_y[q] for q in q_rnd_x}
        n_q = len(dose_pre)
        fn_pre = _dose_inherited_fn(pre_cfg, dataset, trait, n_q)
        fn_rnd = _dose_inherited_fn(rnd_cfg, dataset, trait, n_q)

        def _h4_fn(idx, _fp=fn_pre, _fr=fn_rnd):
            d_pre, _ = _fp(idx)
            d_rnd, _ = _fr(idx)
            dd = d_pre - d_rnd  # NaN propagates: either side without a coherent coef
            return dd, int(np.isnan(dd).sum())

        h4[key] = _paired_contrast(
            dose_pre,
            dose_rnd,
            seed=_contrast_seed(kind, pre_cfg, dataset),
            n_boot=n_boot,
            inherited_fn=_h4_fn,
        ) | {"selected": {pre_cfg: sel_pre, rnd_cfg: sel_rnd}}
    # H4(c) conditional W2b matched-window reads (plan §4.3, TRIGGERED):
    # sycophancy N vs RN — file-presence gated inside fu2_w2b_contrasts (RN is
    # a CONDITIONAL config, never in the wave-1 specs, so gate on the PRE side
    # + the dataset subset only).
    h4c: dict[str, dict] = {}
    if FU2_W2B_PRE in selected_cfgs and (not args.datasets or FU2_W2B_DATASET in args.datasets):
        h4c = fu2_w2b_contrasts(fu_root, n_boot, not_computable)
    # H4(b) descriptive matched-effective-dose read: RQ vs G's banked curve
    # (plan §3 H4(b): α_RQ = c·ρ_19 vs α_G = coef·‖E2[19]‖ — no CI, no verdict).
    h4b: dict = {}
    rq_entry = sel_fu.get("RQ_evil")
    g_entry = sel_parent.get("G_evil")
    if rq_entry is not None and g_entry is not None:
        rho_19 = 96.727321  # fu1 rho.json, plan-pinned (S0-verified)
        e2_norm_19 = 49.57  # parent ‖E2[19]‖ (plan §4.3 grounding)
        rows = []
        for rq_c, g_c in FU2_H4B_DOSE_MATCHES:
            rq_row = rq_entry["curve"].get(str(rq_c))
            g_row = g_entry["curve"].get(str(g_c))
            if rq_row is None or g_row is None:
                continue
            rows.append(
                {
                    "rq_coef": rq_c,
                    "g_coef": g_c,
                    "alpha_eff_rq": round(rq_c * rho_19, 2),
                    "alpha_eff_g": round(g_c * e2_norm_19, 2),
                    "rq_trait_mean": rq_row["trait_mean"],
                    "g_trait_mean": g_row["trait_mean"],
                    "rq_coherence_mean": rq_row["coherence_mean"],
                    "g_coherence_mean": g_row["coherence_mean"],
                }
            )
        h4b = {
            "note": "DESCRIPTIVE matched-effective-dose comparison (plan §3 H4(b)): "
            "RQ (random all-token @L19, α = c·ρ_19, ρ_19 = 96.727321) vs parent G's "
            "banked curve (α = coef·‖E2[19]‖, ‖E2[19]‖ = 49.57); no CI, no verdict",
            "rows": rows,
        }
    out = fu_root / "analysis" / "contrasts.json"
    pa._atomic_write_json(
        out,
        {
            "round": "fu2",
            "n_boot": n_boot,
            "seed_base": pa.BOOTSTRAP_SEED,
            "seed_scheme": "2225 + 1000*dataset_index + 100*config_index (fu2 order "
            f"{FU2_CONFIG_ORDER}) + kind offset {FU2_KIND_OFFSET} (frozen + "
            "selection-inherited share one idx stream per contrast)",
            "lattice": "Effect-negative <=> Δ<0 AND 95% CI wholly below 0; "
            "Effect-positive <=> Δ>0 AND CI wholly above 0; Statistical tie <=> "
            "otherwise (fu relabeling of the parent's asserted three-way partition)",
            "h1_dose": {
                "delta_definition": "score(matched-coherence selected coef) - score(smallest "
                "grid coef), question-paired within arm; degenerate_zero_dose marks arms "
                "whose selected coef IS the smallest (Δ identically 0)",
                "inherited_caveat": "the smallest-coef side is ALSO coherence-gated inside "
                "each resample under the reused parent machinery (sensitivity-only deviation "
                "from the frozen H1 definition)",
                "per_arm": h1,
            },
            "h2_vs_parent_G": {
                "delta_definition": "score(fu2 arm @ its op point) - score(parent G_evil @ "
                "parent op point), question-paired; parent side from BANKED parent "
                "per-question rows. Q_evil = the round's HEADLINE contrast; N_evil = "
                "cross-layer secondary (G is block-19; disclosed). Effect-negative = the "
                "map's pre-image beats the parent's extracted context direction at the "
                "all-token position",
                "per_arm": h2g,
            },
            "h2_secondary_vs_parent_A": {
                "delta_definition": "same machinery with parent A (paper method) as the "
                "reference side (secondary; all 3 corpora)",
                "per_arm": h2a,
            },
            "h3_vs_fu1_positions": {
                "delta_definition": "score(fu2 arm @ its op) - score(the SAME d_pre at "
                "fu1's positions @ fu1 op), question-paired per (layer, corpus): N-J/N-L "
                "(L14), Q-K/Q-M (L19). fu1 side from BANKED fu1 per-question rows. "
                "Effect-negative = all-token position prevents where context positions "
                "did not",
                "per_pair": h3,
            },
            "h4_direction_specificity": {
                "delta_definition": "(a) DoD Δdose(pre-image) - Δdose(RQ), evil, "
                "question-paired (Q-RQ primary; N-RQ cross-layer, EXPLORATORY/disclosed)",
                "per_pair": h4,
                "matched_effective_dose_descriptive": h4b,
                "w2b_conditional": {
                    "delta_definition": "(c) W2b MATCHED-WINDOW direction-specificity on "
                    "sycophancy (plan §4.3, TRIGGERED): DoD Δdose(N) - Δdose(RN) with "
                    "dose = score(@3.0) - score(@1.5) per arm, question-paired — the "
                    "window is [1.5, 3.0] for BOTH arms (RN exists only at the W2b "
                    "selected-coef ± one-neighbor cells, so matching N's window makes "
                    "the dose windows identical; unlike H4(a)'s [grid-min, selected] "
                    "window) — plus the secondary matched-dose LEVEL read N@3.0 vs "
                    "RN@3.0 (no dose subtraction). File-presence gated: cells not yet "
                    "landed -> both keys in not_computable",
                    "per_pair": h4c,
                },
            },
            "not_computable": not_computable,
            "single_seed_caveat": "per-arm verdicts are SINGLE-TRAINING-SEED claims (seed 0; "
            "the CI carries zero training-draw variance)",
            **_subset_note(args),
        },
    )
    print(
        f"[fu2-contrasts] h1={len(h1)} h2G={len(h2g)} h2A={len(h2a)} h3={len(h3)} "
        f"h4={len(h4)} h4c={len(h4c)} not_computable={len(not_computable)} -> {out}",
        flush=True,
    )
    return out


# ── phase: judge accounting fold ──────────────────────────────────────────────


def run_fu_judge(args) -> Path:
    fu_root = Path(args.eval_root)
    parent_root = Path(args.parent_eval_root)
    dig = pa._load_json(fu_root / "judge_digest.json")
    parent_dig = pa._load_json(parent_root / "judge_digest.json")

    def _frac(r):
        tot = r.get("n_rollouts_total") or 0
        return (r.get("n_rollouts_scored", 0) / tot) if tot else None

    keep = (
        "arm",
        "tag",
        "wave",
        "rubric",
        "n_rollouts_scored",
        "n_rollouts_total",
        "n_total_draws",
        "n_content_dropped",
        "n_truncation_dropped",
        "n_transport_lost",
        "n_refusal_draws",
        "n_api_refusal",
        "api_refusal_reissued",
        "n_draws_recovered_by_reissue",
    )
    per_arm = [
        {k: r.get(k) for k in keep} | {"frac_items_complete": _frac(r)} for r in dig["per_arm"]
    ]
    parent_ts = [
        (_frac(r), r["arm"])
        for r in parent_dig["per_arm"]
        if r.get("wave") == "trait_scores" and _frac(r) is not None
    ]
    if not parent_ts:
        raise ValueError("parent judge_digest.json has no trait_scores per-arm rows")
    parent_floor, parent_floor_arm = min(parent_ts)
    fu_ts = [r for r in per_arm if r.get("wave") == "trait_scores"]
    below = sorted(
        r["arm"]
        for r in fu_ts
        if r["frac_items_complete"] is not None and r["frac_items_complete"] < parent_floor
    )
    gen = pa._load_json(fu_root / "digests" / "eval_gen_digest.json")
    out = fu_root / "analysis" / "judge_accounting.json"
    pa._atomic_write_json(
        out,
        {
            "note": "llm-judging rules fold — rule 9: content drops (n_content_dropped, "
            "n_truncation_dropped; dropped never coerced); rule 24: transport retried, "
            "n_transport_lost = losses after retry; rule 28: api-refusal sync-reissue "
            "(n_api_refusal + api_refusal_reissued + n_draws_recovered_by_reissue + the "
            "digest remediation block); rule 29: frac_items_complete "
            "(= n_rollouts_scored / n_rollouts_total) vs the parent run's realized floor",
            "stop_reason_tally_total": dig.get("stop_reason_tally_total"),
            "api_refusal_remediation": dig.get("api_refusal_remediation"),
            "arms_with_api_refusal": dig.get("arms_with_api_refusal"),
            "per_arm": per_arm,
            "fu_trait_scores_floor": min(
                (r["frac_items_complete"] for r in fu_ts if r["frac_items_complete"] is not None),
                default=None,
            ),
            "parent_trait_scores_floor": {"frac": parent_floor, "arm": parent_floor_arm},
            "fu_arms_below_parent_floor": below,
            "cap_hit": {
                "trigger": gen.get("cap_hit_regen_trigger"),
                "units_over_trigger": gen.get("units_over_trigger"),
                "per_unit": gen.get("per_unit"),
            },
        },
    )
    print(
        f"[{_ROUND}-judge] {len(per_arm)} arms; {len(below)} below parent floor "
        f"({parent_floor:.4f}) -> {out}",
        flush=True,
    )
    return out


# ── phase: probe (parent machinery + fu-layer annotation) ────────────────────


def run_fu_probe(args) -> Path:
    combined = ensure_combined_capture_root(args)
    args.capture_root = str(combined)
    out = pa.run_probe(args)
    data = json.load(open(out))
    n_annot = 0
    for r in data["shifts"].values():
        cell = _round_resolve(r["tag"])
        if cell is None:
            continue
        r["fu_l1_idx"] = cell.l1_idx
        for v in r["variants"].values():
            v["fu_shift_l1"] = v["shift_per_layer"][cell.l1_idx]
        n_annot += 1
    data["fu_annotation"] = (
        "fu_l1_idx / fu_shift_l1 = the fu config's OWN steered layer (14 or 19, per the "
        f"{_ROUND} registry); the parent l1_layer_idx field remains the parent trait L1"
    )
    pa._atomic_write_json(Path(out), data)
    print(f"[{_ROUND}-probe] {n_annot} fu tags layer-annotated -> {out}", flush=True)
    return out


# ── phase: projection (parent r_B/E2 legs + the d_pre addition) ───────────────


def _load_dpre_bank(fu_directions_dir: Path, trait: str):
    """Load {trait}_PRE.pt: (28, d) with ONLY the map layers finite (ρ-scaled).

    Returns (layers, {layer: unit-normalized direction}) — projections are onto
    UNIT d_pre (the parent projection convention), so the bank's ρ scaling
    drops out.
    """
    import torch

    fud = _fud()
    bank = torch.load(
        fu_directions_dir / f"{trait}_PRE.pt", weights_only=True, map_location="cpu"
    ).to(torch.float32)
    assert bank.ndim == 2, bank.shape
    finite = torch.isfinite(bank).all(dim=1)
    layers = sorted(int(i) for i in torch.nonzero(finite).flatten())
    if layers != sorted(int(x) for x in fud.MAP_LAYERS):
        raise ValueError(
            f"{trait}_PRE.pt finite rows {layers} != MAP_LAYERS {sorted(fud.MAP_LAYERS)}"
        )
    return layers, {la: bank[la] / bank[la].norm().clamp_min(1e-12) for la in layers}


def _dpre_project_store(store: dict, layers: list[int], rows: dict) -> dict:
    """Per-position mean projection onto unit d_pre at each map layer."""
    import torch

    out = {}
    for pos in ("response_avg", "context_end", "prefix_end"):
        X = store[pos].to(torch.float32)  # (rows, L, d)
        assert X.ndim == 3, X.shape
        out[pos] = {str(la): float((X[:, la, :] @ rows[la]).mean()) for la in layers}
    return out


def run_dpre_projection(args, capture_root: Path) -> Path:
    """fu addition: Δ mean projection onto d_pre (layers 14/19) per (cell, position)."""
    import torch

    analysis_dir = Path(args.eval_root) / "analysis"
    fu_dir = Path(args.fu_directions_dir)
    partial = analysis_dir / "projection_dpre_partial.jsonl"
    dpre_sha = {t: pa._sha256_file(fu_dir / f"{t}_PRE.pt") for t in PROBE_TRAITS}
    done = {
        (r["tag"], r["trait"])
        for r in pa._load_jsonl_rows(partial)
        if r.get("dpre_sha256") == dpre_sha.get(r["trait"])
    }
    banks = {t: _load_dpre_bank(fu_dir, t) for t in PROBE_TRAITS}
    targets = list(pa._iter_capture_targets(capture_root))
    total = sum(len(m["traits_expected"]) for _, m in targets)
    k, t0 = 0, time.time()
    for tag, manifest in targets:
        for trait in manifest["traits_expected"]:
            k += 1
            if trait not in banks:
                raise ValueError(f"no d_pre bank for trait {trait!r} (target {tag})")
            if (tag, trait) in done and not args.force:
                continue
            store = torch.load(
                capture_root / tag / f"{trait}.pt", weights_only=True, map_location="cpu"
            )
            layers, rows = banks[trait]
            row = {
                "tag": tag,
                "trait": trait,
                "dpre_sha256": dpre_sha[trait],
                "projections": _dpre_project_store(store, layers, rows),
            }
            pa._append_jsonl(partial, row)
            del store
            print(
                f"[dpre] unit {k}/{total} {tag}__{trait} elapsed={round(time.time() - t0, 1)}s",
                flush=True,
            )
    rows_all = pa._load_jsonl_rows(partial)
    by_key = {(r["tag"], r["trait"]): r for r in rows_all}
    shifts: dict[str, dict] = {}
    for (tag, trait), r in sorted(by_key.items()):
        base_r = by_key.get(("base", trait))
        if base_r is None:
            raise ValueError(f"base d_pre projection row missing for trait {trait}")
        pos_out = {}
        for pos, per_layer in r["projections"].items():
            pos_out[pos] = {la: v - base_r["projections"][pos][la] for la, v in per_layer.items()}
        shifts[f"{tag}__{trait}"] = {"tag": tag, "trait": trait, "positions": pos_out}
    out = analysis_dir / "projection_dpre.json"
    pa._atomic_write_json(
        out,
        {
            "note": "fu addition: Δ mean projection (finetuned - base) onto UNIT-normalized "
            "d_pre (the fu pre-image direction bank, finite at layers 14/19 only) per "
            "(cell, position); complements the parent r_B/E2 monitor "
            "(projection_shifts.json)",
            "shifts": shifts,
        },
    )
    print(f"[{_ROUND}-dpre] {len(shifts)} unit shifts -> {out}", flush=True)
    return out


def run_fu_projection(args) -> Path:
    combined = ensure_combined_capture_root(args)
    args.capture_root = str(combined)
    pa.run_projection(args)  # parent r_B/E2 legs -> projection_shifts.json
    return run_dpre_projection(args, combined)


# ── phase: narrow (+ parent reference) ────────────────────────────────────────


def run_fu_narrow(args) -> Path:
    if _is_fu2():
        raise SystemExit(
            "[fu2] --phase narrow is fu1-only: the fu2 round has NO opinions cells "
            "(plan v13 §2 divergence 1)"
        )
    out = pa.run_narrow(args)  # fu opinions cells -> analysis/narrow_retention.json
    data = json.load(open(out))
    parent = pa._load_json(Path(args.parent_eval_root) / "analysis" / "narrow_retention.json")
    data["parent_reference"] = {
        "note": "parent run per-arm narrow-domain retention (banked)",
        "per_arm": parent.get("per_arm", {}),
    }
    pa._atomic_write_json(Path(out), data)
    print(f"[fu1-narrow] parent reference folded -> {out}", flush=True)
    return out


# ── phase: figures ────────────────────────────────────────────────────────────

FU_CONFIG_LABEL = {
    "J": "Pre-image L14, context tokens",
    "K": "Pre-image L19, context tokens",
    "L": "Pre-image L14, context-end",
    "M": "Pre-image L19, context-end",
    "RJ": "Random L14, context tokens",
    "RK": "Random L19, context tokens",
    "RL": "Random L14, context-end",
    "RM": "Random L19, context-end",
    # fu2 configs (all-token position; plan v13 §5 plain-English names)
    "N": "Pre-image L14, all tokens",
    "Q": "Pre-image L19, all tokens",
    "RQ": "Random L19, all tokens",
    "RN": "Random L14, all tokens",
    "C": "Parent context extract+steer",
    "G": "Parent context extract, all tokens",
    "A": "Parent paper method",
    "baseline": "Unsteered finetune",
    "base": "Base model",
}
DATASET_LABEL = {
    "evil": "Evil II",
    "sycophancy": "Sycophancy II",
    "hallucination": "Hallucination II",
    "mistake_opinions": "Mistake opinions",
}


def _fig_colors():
    from explore_persona_space.analysis.paper_plots import paper_palette

    keys = (
        list(_round_config_order())
        + list(_round_parent_comparators())
        + [
            "baseline",
            "base",
        ]
    )
    pal = paper_palette(max(12, len(keys)))
    return dict(zip(keys, pal))


def _load_analysis(args, name: str, *, root: str = "eval_root") -> dict:
    path = Path(getattr(args, root)) / "analysis" / name
    if not path.exists():
        raise FileNotFoundError(f"figure input missing: {path} — run its producing phase first")
    return json.load(open(path))


def _curve_xy(entry: dict, metric: str):
    pts = sorted((float(c), v[metric]) for c, v in entry["curve"].items() if v[metric] is not None)
    return [c for c, _ in pts], [v for _, v in pts]


def fig_hero(args) -> dict:
    import matplotlib.pyplot as plt

    sel_fu = _load_analysis(args, "selection.json")["selection"]
    sel_parent = _load_analysis(args, "selection.json", root="parent_eval_root")["selection"]
    train = pa._train()
    colors = _fig_colors()
    datasets = [d for d in train.DATASETS if any(k.endswith(f"_{d}") for k in sel_fu)]
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.2 * len(datasets), 6.4), sharex="col")
    if len(datasets) == 1:
        axes = axes.reshape(2, 1)
    for j, ds in enumerate(datasets):
        for row, metric, ylab in (
            (0, "trait_mean", "Trait score (0-100)"),
            (1, "coherence_mean", "Coherence (0-100)"),
        ):
            ax = axes[row, j]
            for cfg in _round_config_order():
                entry = sel_fu.get(f"{cfg}_{ds}")
                if entry is None:
                    continue
                xs, ys = _curve_xy(entry, metric)
                style = "--" if cfg.startswith("R") else "-"
                ax.plot(
                    xs,
                    ys,
                    style,
                    marker="o",
                    ms=3,
                    color=colors[cfg],
                    # Label every artist: the cross-axes dedupe pass below keeps
                    # one handle per label, and a conditional keyed on ds=="evil"
                    # drops any config absent there (RN, sycophancy-only — the
                    # W2b conditional arm rendered legend-less).
                    label=FU_CONFIG_LABEL[cfg],
                )
                if entry["selected_coef"] is not None:
                    ci = (
                        xs.index(float(entry["selected_coef"]))
                        if float(entry["selected_coef"]) in xs
                        else None
                    )
                    if ci is not None:
                        ax.plot(xs[ci], ys[ci], "o", ms=9, mfc="none", color=colors[cfg])
            for pcfg in _round_parent_comparators():
                entry = sel_parent.get(f"{pcfg}_{ds}")
                if entry is None:
                    continue
                xs, ys = _curve_xy(entry, metric)
                ax.plot(
                    xs,
                    ys,
                    "-",
                    marker="s",
                    ms=3,
                    color=colors[pcfg],
                    lw=2,
                    label=FU_CONFIG_LABEL[pcfg] if row == 0 and j == 0 else None,
                )
            if row == 1:
                ax.axhline(pa.COHERENCE_THRESHOLD, color="gray", ls=":", lw=1)
                ax.set_xlabel("Steering coefficient")
            if j == 0:
                ax.set_ylabel(ylab)
            ax.set_xscale("log", base=2)
            if row == 0:
                ax.set_title(DATASET_LABEL.get(ds, ds))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax_row in axes:
        for ax in ax_row:
            h, la = ax.get_legend_handles_labels()
            for hh, ll in zip(h, la):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return {_stem("hero_dose_response"): fig}


def fig_op_bars(args) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np

    sel_fu = _load_analysis(args, "selection.json")["selection"]
    sel_parent = _load_analysis(args, "selection.json", root="parent_eval_root")["selection"]
    train = pa._train()
    fu_root, parent_root = Path(args.eval_root), Path(args.parent_eval_root)
    colors = _fig_colors()
    datasets = [d for d in train.DATASETS if any(k.endswith(f"_{d}") for k in sel_fu)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for j, ds in enumerate(datasets):
        ax = axes[j]
        trait = train.STEERED_TRAIT[ds]
        bars = []  # (label key, per-question means)
        for cfg in _round_config_order():
            entry = sel_fu.get(f"{cfg}_{ds}")
            if entry is None or entry["selected_coef"] is None:
                continue
            qm = pa._per_question_means(
                pa._load_json(
                    pa._arm_path(fu_root, "trait_scores", cfg, ds, entry["selected_coef"])
                )["traits"][trait]
            )
            bars.append((cfg, [v for v in qm.values() if v is not None]))
        for pcfg in _round_parent_comparators():
            p_entry = sel_parent.get(f"{pcfg}_{ds}")
            if p_entry is None or p_entry["selected_coef"] is None:
                continue
            qm = pa._per_question_means(
                pa._load_json(
                    pa._arm_path(parent_root, "trait_scores", pcfg, ds, p_entry["selected_coef"])
                )["traits"][trait]
            )
            bars.append((pcfg, [v for v in qm.values() if v is not None]))
        for anchor, fname in (("baseline", f"baseline_{ds}.json"), ("base", "base.json")):
            path = parent_root / "trait_scores" / fname
            if path.exists():
                qm = pa._per_question_means(pa._load_json(path)["traits"][trait])
                bars.append((anchor, [v for v in qm.values() if v is not None]))
        for i, (key, vals) in enumerate(bars):
            ax.bar(i, float(np.mean(vals)), color=colors[key], width=0.72)
            ax.scatter(
                i + rng.uniform(-0.16, 0.16, len(vals)),
                vals,
                s=7,
                color="black",
                alpha=0.45,
                zorder=3,
            )
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(
            [FU_CONFIG_LABEL[k] for k, _ in bars], rotation=60, ha="right", fontsize=7
        )
        ax.set_title(DATASET_LABEL.get(ds, ds))
        if j == 0:
            ax.set_ylabel("Trait score at operating point (0-100)")
    fig.tight_layout()
    return {_stem("operating_point_bars"): fig}


def fig_geometry(args) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np

    fu_dir = Path(args.fu_directions_dir)
    cols = [
        ("cos_d_pre_r_b", "cos(d_pre, r_B)"),
        ("cos_d_pre_parent_E2", "cos(d_pre, parent E2)"),
        ("cos_d_pre_parent_E3", "cos(d_pre, parent E3)"),
        ("cos_d_pre_random", "cos(d_pre, random)"),
        ("bridge_cos_rbv2_r779", "bridge cos(rb_v2, #779 r_B)"),
    ]
    row_labels, mat = [], []
    for trait in PROBE_TRAITS:
        meta = pa._load_json(fu_dir / f"{trait}_PRE_meta.json")
        for lkey in sorted(meta["per_layer"]):
            block = meta["per_layer"][lkey]
            row_labels.append(f"{trait} {lkey}")
            mat.append([block[k] for k, _ in cols])
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lab for _, lab in cols], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Cosine")
    fig.tight_layout()
    return {_stem("geometry_cosines"): fig}


def fig_rank_sweep(args) -> dict:
    import matplotlib.pyplot as plt

    fu_dir = Path(args.fu_directions_dir)
    colors = _fig_colors()
    # First three round configs' palette slots (fu1: J/K/L — byte-identical).
    trait_color = dict(zip(PROBE_TRAITS, [colors[c] for c in _round_config_order()[:3]]))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for trait in PROBE_TRAITS:
        meta = pa._load_json(fu_dir / f"{trait}_PRE_meta.json")
        for lkey in sorted(meta["per_layer"]):
            sweep = meta["per_layer"][lkey]["rank_sweep_cos_vs_primary"]
            ks = sorted(int(k) for k in sweep)
            ax.plot(
                ks,
                [sweep[str(k)] for k in ks],
                marker="o",
                ms=4,
                ls="-" if lkey == "L14" else "--",
                color=trait_color[trait],
                label=f"{trait} {lkey}",
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Truncated-pinv rank k")
    ax.set_ylabel("cos(d_pre(k), primary d_pre)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return {_stem("rank_sweep"): fig}


def fig_probe_profiles(args) -> dict:
    import matplotlib.pyplot as plt

    shifts = _load_analysis(args, "probe_shifts.json")["shifts"]
    sel_fu = _load_analysis(args, "selection.json")["selection"]
    colors = _fig_colors()
    fig, axes = plt.subplots(
        1, len(PROBE_TRAITS), figsize=(4.4 * len(PROBE_TRAITS), 3.8), sharey=False
    )
    for j, trait in enumerate(PROBE_TRAITS):
        ax = axes[j]
        for cfg in _round_config_order():
            entry = sel_fu.get(f"{cfg}_{trait}")
            if entry is None or entry["selected_coef"] is None:
                continue
            tag = f"{cfg}__{trait}__c{entry['selected_coef']}"
            r = shifts.get(f"{tag}__{trait}")
            if r is None:
                continue
            prof = r["variants"]["full"]["shift_per_layer"]
            ax.plot(
                range(len(prof)),
                prof,
                color=colors[cfg],
                ls="--" if cfg.startswith("R") else "-",
                lw=1.4,
                label=FU_CONFIG_LABEL[cfg],
            )
        anchor = shifts.get(f"baseft_{trait}__{trait}")
        if anchor is not None:
            prof = anchor["variants"]["full"]["shift_per_layer"]
            ax.plot(
                range(len(prof)),
                prof,
                color="black",
                ls=":",
                lw=1.6,
                label="Unsteered finetune (baseft)",
            )
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(trait)
        ax.set_xlabel("Layer")
        if j == 0:
            ax.set_ylabel("Probe-score shift (finetuned - base)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return {_stem("probe_shift_profiles"): fig}


def fig_projection_bars(args) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np

    proj = _load_analysis(args, "projection_shifts.json")["shifts"]
    dpre = _load_analysis(args, "projection_dpre.json")["shifts"]
    sel_fu = _load_analysis(args, "selection.json")["selection"]
    ds, trait = "evil", "evil"
    bar_keys = [
        ("response_avg", "parent", "resp-avg · r_B"),
        ("context_end", "parent", "ctx-end · E2"),
        ("prefix_end", "parent", "prefix-end · E2"),
        ("response_avg", "dpre", "resp-avg · d_pre"),
        ("context_end", "dpre", "ctx-end · d_pre"),
        ("prefix_end", "dpre", "prefix-end · d_pre"),
    ]
    from explore_persona_space.analysis.paper_plots import paper_palette

    bar_colors = paper_palette(len(bar_keys))
    cfgs, series = [], {i: [] for i in range(len(bar_keys))}
    for cfg in _round_config_order():
        entry = sel_fu.get(f"{cfg}_{ds}")
        if entry is None or entry["selected_coef"] is None:
            continue
        tag = f"{cfg}__{ds}__c{entry['selected_coef']}"
        pr = proj.get(f"{tag}__{trait}")
        dr = dpre.get(f"{tag}__{trait}")
        if pr is None or dr is None:
            continue
        cell = _round_resolve(tag)
        la = cell.l1_idx
        cfgs.append(cfg)
        for i, (pos, kind, _) in enumerate(bar_keys):
            if kind == "parent":
                series[i].append(pr["positions"][pos]["shift_per_layer"][la])
            else:
                series[i].append(dr["positions"][pos][str(la)])
    if not cfgs:
        raise ValueError("no evil-corpus fu arms with projection rows — run projection first")
    x = np.arange(len(cfgs))
    w = 0.13
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for i, (_, _, lab) in enumerate(bar_keys):
        ax.bar(x + (i - 2.5) * w, series[i], width=w, color=bar_colors[i], label=lab)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([FU_CONFIG_LABEL[c] for c in cfgs], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Δ mean projection at the arm's own layer")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    return {_stem("projection_shift_bars"): fig}


def fig_mmlu(args) -> dict:
    import matplotlib.pyplot as plt

    data = _load_analysis(args, "mmlu.json")
    ref = data.get("parent_reference")
    if ref is None:
        raise ValueError("fu mmlu.json lacks parent_reference — run --phase mmlu via this driver")
    tags = sorted(t for t, v in data["per_target"].items() if v.get("mmlu_acc") is not None)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.24 * len(tags)), 4.0))
    ax.axhspan(ref["band_min"], ref["band_max"], color="gray", alpha=0.25, label="Parent-run range")
    ax.axhline(ref["base_mmlu_acc"], color="black", ls=":", lw=1.2, label="Base model")
    colors = _fig_colors()
    for i, tag in enumerate(tags):
        cfg = tag.split("__")[0]
        ax.plot(i, data["per_target"][tag]["mmlu_acc"], "o", ms=5, color=colors.get(cfg, "gray"))
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("MMLU accuracy")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return {_stem("mmlu_extremes"): fig}


def fig_judge_diag(args) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np

    acc = _load_analysis(args, "judge_accounting.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ts = sorted(
        (r["frac_items_complete"], r["arm"])
        for r in acc["per_arm"]
        if r.get("wave") == "trait_scores" and r["frac_items_complete"] is not None
    )
    ax = axes[0]
    ax.plot(range(len(ts)), [f for f, _ in ts], "o", ms=3, color="tab:blue")
    ax.axhline(
        acc["parent_trait_scores_floor"]["frac"],
        color="black",
        ls="--",
        lw=1.2,
        label="Parent-run floor",
    )
    ax.set_xlabel("fu1 trait-score arm (sorted)")
    ax.set_ylabel("Fraction of rollouts scored")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    per_unit = acc["cap_hit"]["per_unit"] or {}
    fracs = [v.get("cap_hit_fraction") for v in per_unit.values()]
    fracs = np.array([f for f in fracs if f is not None], dtype=float)
    if fracs.size:
        ax.hist(fracs, bins=24, color="tab:orange")
    trig = acc["cap_hit"].get("trigger")
    if trig is not None:
        ax.axvline(trig, color="black", ls="--", lw=1.2, label="Re-gen trigger")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("Cap-hit fraction per generation unit")
    ax.set_ylabel("Units")
    fig.tight_layout()
    return {_stem("judge_diagnostics"): fig}


FIGURE_BUILDERS = {
    "hero": fig_hero,
    "op_bars": fig_op_bars,
    "geometry": fig_geometry,
    "rank_sweep": fig_rank_sweep,
    "probe_profiles": fig_probe_profiles,
    "projection_bars": fig_projection_bars,
    "mmlu": fig_mmlu,
    "judge_diag": fig_judge_diag,
}


def run_fu_figures(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("generic")
    names = args.figures or list(FIGURE_BUILDERS)
    unknown = sorted(set(names) - set(FIGURE_BUILDERS))
    if unknown:
        raise ValueError(f"unknown figure(s) {unknown} (have {sorted(FIGURE_BUILDERS)})")
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        figs = FIGURE_BUILDERS[name](args)
        for stem, fig in figs.items():
            paths = savefig_paper(fig, stem, dir=fig_dir)
            plt.close(fig)
            print(f"[{_ROUND}-figures] {name} -> {paths.get('png', fig_dir / stem)}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

PHASES = {
    "mmlu": run_fu_mmlu,
    "selection": run_fu_selection,
    "contrasts": run_contrasts,  # round dispatch: fu1 battery / fu2 §3 battery
    "judge": run_fu_judge,
    "probe": run_fu_probe,
    "projection": run_fu_projection,
    "narrow": run_fu_narrow,  # fu1-only (fu2 refuses: no opinions cells)
    "figures": run_fu_figures,
}
PHASE_ORDER = [
    "mmlu",
    "selection",
    "contrasts",
    "judge",
    "probe",
    "projection",
    "narrow",
    "figures",
]
# fu2 "all" order drops narrow (plan v13 §2 divergence 1: opinions corpus dropped).
FU2_PHASE_ORDER = [p for p in PHASE_ORDER if p != "narrow"]

# argparse dests the REUSED parent functions dereference on args (kept complete
# so the two-file argcheck union cannot mask a dest missing from THIS parser).
PARENT_CONSUMED_DESTS = (
    "eval_root",
    "capture_root",
    "directions_dir",
    "mmlu_dir",
    "i778_staging",
    "work_root",
    "staging_mirror",
    "stage_inputs",
    "n_boot",
    "force",
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Issue #2225 fu-round F5 analysis + figures (--round fu1|fu2)."
    )
    ap.add_argument(
        "--round",
        default="fu1",
        choices=sorted(ROUND_EVAL_ROOTS),
        help="which fu round's registry/roots/contrast set to run (default: fu1 — "
        "committed fu1 behavior unchanged)",
    )
    ap.add_argument("--phase", default="all", choices=[*PHASE_ORDER, "all"])
    ap.add_argument(
        "--eval-root",
        default=None,
        help=f"fu F4 outputs + analysis/ destination (default: the round's root "
        f"{ROUND_EVAL_ROOTS}; parent functions read args.eval_root)",
    )
    ap.add_argument("--parent-eval-root", default="eval_results/issue_2225")
    ap.add_argument(
        "--fu1-eval-root",
        default=ROUND_EVAL_ROOTS["fu1"],
        help="the fu1 round's eval root (fu2 H3 comparator rows + fu1 op points)",
    )
    ap.add_argument(
        "--staging-mirror",
        default=None,
        help="HF mirror root (data disk — never / or /tmp; default: the round's mirror "
        f"{ROUND_STAGING_MIRRORS}); None-defaulted paths derive from it",
    )
    ap.add_argument("--capture-root", default=None, help="combined capture root (derived)")
    ap.add_argument("--directions-dir", default=None, help="PARENT directions tensors (derived)")
    ap.add_argument("--fu-directions-dir", default=None, help="fu1 d_pre bank dir (derived)")
    ap.add_argument("--mmlu-dir", default=None, help="staged fu1 MMLU JSONs (derived)")
    ap.add_argument("--i778-staging", default=None, help="#778 probe-pool staging (derived)")
    ap.add_argument("--work-root", default=None, help="probe bundle work dir (derived)")
    ap.add_argument(
        "--fig-dir",
        default=None,
        help=f"figure destination (default: the round's dir {ROUND_FIG_DIRS})",
    )
    ap.add_argument("--repilot-state", default=None, help="f1_repilot_state.json (grid override)")
    ap.add_argument("--stage-inputs", action="store_true", help="stage HF inputs first")
    ap.add_argument(
        "--allow-partial-capture",
        action="store_true",
        help="skip the full fu1-capture coverage assert (smoke only)",
    )
    ap.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="subset filter for selection/contrasts (smoke dispatch)",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="subset filter for selection/contrasts (smoke dispatch)",
    )
    ap.add_argument(
        "--figures",
        nargs="*",
        default=None,
        help=f"figure subset (default all; have {sorted(FIGURE_BUILDERS)})",
    )
    ap.add_argument("--n-boot", type=int, default=pa.N_BOOT_DEFAULT)
    ap.add_argument("--force", action="store_true", help="ignore resume checkpoints")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    _set_round(args.round)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        # Two-file scope: every args.<attr> Load in THIS module AND the reused
        # parent module must be argparse-defined (union of both parsers) ...
        assert_args_attributes_defined(__file__, pa.__file__)
        # ... and the union must not mask a parent-consumed dest missing from
        # THIS parser (the namespace the parent functions actually receive):
        ns = build_argparser().parse_args([])
        for dest in PARENT_CONSUMED_DESTS:
            getattr(ns, dest)
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import torch  # noqa: F401

        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            paper_palette,
            savefig_paper,
            set_paper_style,
        )
        from explore_persona_space.orchestrate.hub import stage_hub_prefix  # noqa: F401

        _fu_train()
        import issue2225_fu2_train  # noqa: F401  (fu2 round module resolvable)

        _fud()
        pa._train()
        pa._judge()
        pa._directions_mod()
        print("[import-check] OK", flush=True)
        return 0
    _apply_derived_defaults(args)
    if args.stage_inputs:
        stage_fu_inputs(args)
    round_order = FU2_PHASE_ORDER if _is_fu2() else PHASE_ORDER
    if args.phase != "all" and args.phase not in round_order:
        raise SystemExit(f"[{_ROUND}] --phase {args.phase} is not in this round's phase set")
    order = round_order if args.phase == "all" else [args.phase]
    for name in order:
        PHASES[name](args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
