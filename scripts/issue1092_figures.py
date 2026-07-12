#!/usr/bin/env python3
"""Issue #1092 P7: merge P6 fit-grid outputs, 28-layer projection sweeps, figures.

Plan v5 section 4.6 (P7) + plan v6 section 4.5-A P7-side rows. Three stages, each
independently resumable (skip on existing output with matching input fingerprint):

  merge             Stage every P6 box's checkpoint union from the HF data repo
                    (scoped list_repo_tree + per-file hf_hub_download — NEVER
                    snapshot_download on the ~1M-file repo), dedupe duplicate
                    unit-slots deterministically (recorded), and emit one JSON
                    aggregate per read family under --out-dir. Every family JSON
                    (plus battery_scope_caveat.json and the merge manifest)
                    embeds the machine-readable battery_scope_caveat block for
                    concern battery-rows-in-fit-arms-banked-p6 (battery rows
                    were IN TRAINING in both banked fit arms; the fold-iii
                    transfer read is not computable from banked artifacts).
  projection-sweep  The v6-retained 28-layer B1(a) raw-projection r_B^T state
                    read (stage-einsum-delete per layer over the persisted
                    summary shards), the B1(d) B0 poolings (persisted
                    b0_rB_pool arrays, zero recompute), the c12 projection-stage
                    200-draw same-selection nulls (batched GEMM, no serial draw
                    loops), and the conservative cross-fit-layer
                    best-of-{L14,L18,L19} band assembled from the engine's
                    persisted per-unit null draws (per-draw-index max over the
                    three units' independently-seeded draw vectors — labeled
                    CONSERVATIVE in the JSON).
  figures           Render figures from the stage-1/2 JSONs via the
                    /paper-plots conventions into --figures-dir.
  upload-nulls      (opt-in) one batched upload_folder commit of the P7-computed
                    null matrices to the HF data repo (persist-by-default).

Smoke = production entrypoint with tiny flags (PASS_UNIFIED):

  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  uv run python scripts/issue1092_figures.py --steps merge \
      --boxes 1,2 --out-dir /tmp/issue-1092-smoke/p7 --work-dir /tmp/issue-1092-smoke/work

Production (VM, detached per the canonical setsid/choom recipe):

  ... --steps merge,projection-sweep,figures,upload-nulls --expect-full-grid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + .env must bind BEFORE the heavy imports below — the
# BLAS/torch pools freeze at import time (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import numpy as np  # noqa: E402
from issue1092_fit_grid import (  # noqa: E402
    CELL_MODEL_TYPE,
    DEFAULT_RB_REV,
    HF_DATA_REPO,
    _jsonl,
    _load_rb_directions,
    _parse_csv,
    _parse_layers,
    _pearson_or_nan,
    _read_index_files,
)
from issue1092_p6_run import (  # noqa: E402
    HfHubIO,
    HubFile,
    LocalFixtureHubIO,
    _write_json_atomic,
)

HF_PREFIX_DEFAULT = "issue1092_realistic_crossing"
JUDGED_CELLS = ("cell_inst_own", "cell_pre_own")
B1_ARMS = ("prefix_end", "context_end", "c_q_bare")
B1_GRAINS = ("per_example", "condition_averaged")
B0_MODES = ("mean", "max", "top3", "last")  # engine axis order (fit_grid _behavior_reads)
FROZEN_FIT_LAYERS = (14, 18, 19)
HEADLINE_LAYER = 14
# Mirrors the ENGINE's realized fit-arm-A row rule verbatim (fit_grid run():
# stratum not in {trait_stratum, battery_eval_only}); the realized corpus uses
# stratum value "battery" which the engine rule does NOT exclude — mirrored, not
# "fixed", for checkpoint continuity.
FIT_ARM_A_EXCLUDED_STRATA = {"trait_stratum", "battery_eval_only"}
ROW_ARMS = ("A_real_only", "B_all_rows", "trait_battery_only")
ELIGIBILITY_RULE = "std>=1 and >=5 scored and at least one positive/negative"

CKPT_RE = re.compile(
    r"^(?P<cell>.+)_(?P<arm>prefix_end|context_end)_fit(?P<fit>[AB])"
    r"_L(?P<layer>\d{2})_(?P<basis>ambient|pca48)_(?P<fp>[0-9a-f]{24})\.json$"
)

# Plain-English condition names (plan v5 section 5 table) — figure labels only.
CELL_LABELS = {
    "cell_inst_own": "Instruct, own answers",
    "cell_pre_insttext": "Pretrained, instruct answers",
    "cell_pre_own": "Pretrained, own answers",
    "cell_inst_pretext": "Instruct, pretrained answers",
    "cell_inst_claude": "Instruct, Claude answers",
    "cell_pre_claude": "Pretrained, Claude answers",
    "cell_inst_shuf": "Instruct, shuffled answers",
    "cell_pre_shuf": "Pretrained, shuffled answers",
}
ARM_LABELS = {
    "prefix_end": "prefix-based input",
    "context_end": "context-based input",
    "c_q_bare": "bare-query input",
}
GRAIN_LABELS = {"per_example": "per example", "condition_averaged": "condition averaged"}
ROW_ARM_LABELS = {
    "A_real_only": "real rows (fit-arm A subset)",
    "B_all_rows": "all rows (fit-arm B subset)",
    "trait_battery_only": "trait stratum + battery rows",
}


# --------------------------------------------------------------------------- helpers
def _run_metadata(args: argparse.Namespace) -> dict:
    """Reproducibility metadata for every emitted JSON (CLAUDE.md requirement)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    return {
        "git_commit": commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "argv": sys.argv[1:],
        "script": "issue1092_figures.py",
        "issue": 1092,
        "phase": "P7",
        "hf_prefix": args.hf_prefix,
    }


def _sha16_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sha16_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _stable_rng(*parts: object) -> np.random.Generator:
    """Deterministic per-family RNG: sha256 of the joined key -> SeedSequence words."""
    digest = hashlib.sha256("||".join(str(p) for p in parts).encode()).digest()
    words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence(words))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _stage_hub_file(hub, hf: HubFile, target: Path) -> None:
    """Download one hub file to target unless already present at the same size."""
    if target.exists() and target.stat().st_size == hf.size:
        return
    hub.download_to(hf.path, target)
    if target.stat().st_size != hf.size:
        raise RuntimeError(f"staged size mismatch for {hf.path}: {target.stat().st_size}")


def _fingerprint_matches(meta_path: Path, fingerprint: str) -> bool:
    if not meta_path.exists():
        return False
    try:
        return _load_json(meta_path).get("input_fingerprint") == fingerprint
    except (json.JSONDecodeError, OSError):
        return False


def _hub_io(args: argparse.Namespace):
    if args.fixture_root is not None:
        return LocalFixtureHubIO(Path(args.fixture_root))
    return HfHubIO(HF_DATA_REPO, args.hf_revision)


# --------------------------------------------------------------------------- merge
def _list_box_checkpoints(
    hub, prefix: str, boxes: list[int]
) -> tuple[dict[str, HubFile], dict[str, list[int]]]:
    """Union of checkpoint files across boxes: filename -> HubFile, filename -> boxes."""
    by_name: dict[str, HubFile] = {}
    box_of: dict[str, list[int]] = defaultdict(list)
    for box in boxes:
        box_prefix = f"{prefix}/p6/box_{box:02d}/checkpoints"
        try:
            files = hub.list_files(box_prefix)
        except FileNotFoundError:
            print(f"[p7-merge] WARNING: no checkpoints listed under {box_prefix}", flush=True)
            continue
        for hf in files:
            name = hf.path.rsplit("/", 1)[1]
            if not name.endswith(".json"):
                continue
            prior = by_name.get(name)
            if prior is not None and prior.hub_identity != hf.hub_identity:
                raise RuntimeError(
                    f"checkpoint {name} differs across boxes ({prior.hub_identity[:12]} vs "
                    f"{hf.hub_identity[:12]}) — fp-named files must be content-identical"
                )
            by_name.setdefault(name, hf)
            box_of[name].append(box)
    if not by_name:
        raise FileNotFoundError(f"zero checkpoints found under {prefix}/p6/box_*/checkpoints")
    return by_name, dict(box_of)


def _mlp_status_rank(unit: dict) -> int:
    return 0 if unit.get("mlp_companion", {}).get("status") == "computed" else 1


def _dedupe_units(units: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse duplicate (cell, arm, fit_arm, layer, basis) slots deterministically.

    Rule (recorded per decision): prefer a checkpoint whose MLP companion is
    "computed"; tie-break lexicographic-min fingerprint. The known duplicates are
    plan-v6's two prefix pca48 re-runs under b01-inv2's --skip-mlp-companion
    fingerprint — content-equivalent same-X/same-targets/same-seed recomputes.
    """
    by_slot: dict[tuple, list[dict]] = defaultdict(list)
    for unit in units:
        slot = (unit["cell"], unit["arm"], unit["fit_arm"], unit["layer"], unit["basis"])
        by_slot[slot].append(unit)
    kept: list[dict] = []
    decisions: list[dict] = []
    for slot, group in sorted(by_slot.items(), key=lambda kv: kv[0]):
        group = sorted(group, key=lambda u: (_mlp_status_rank(u), u["fingerprint"]))
        kept.append(group[0])
        if len(group) > 1:
            decisions.append(
                {
                    "slot": {
                        "cell": slot[0],
                        "arm": slot[1],
                        "fit_arm": slot[2],
                        "layer": slot[3],
                        "basis": slot[4],
                    },
                    "chosen_fingerprint": group[0]["fingerprint"],
                    "duplicate_fingerprints": [u["fingerprint"] for u in group[1:]],
                    "rule": "prefer mlp_companion.status==computed, then lexicographic-min fp",
                }
            )
    return kept, decisions


def _prov(unit: dict, box_of: dict[str, list[int]], ckpt_name: str) -> dict:
    return {
        "cell": unit["cell"],
        "arm": unit["arm"],
        "fit_arm": unit["fit_arm"],
        "layer": unit["layer"],
        "basis": unit["basis"],
        "fingerprint": unit["fingerprint"],
        "boxes": box_of.get(ckpt_name, []),
        "checkpoint": ckpt_name,
        "n_rows": unit.get("n_rows"),
        "targets": unit.get("targets"),
    }


def _slim_spec_stats(stats: dict, *, top_sv_keep: int = 64) -> dict:
    out = dict(stats)
    top_sv = out.get("top_sv")
    if isinstance(top_sv, list) and len(top_sv) > top_sv_keep:
        out["top_sv"] = top_sv[:top_sv_keep]
        out["top_sv_truncated"] = True
        out["top_sv_total"] = len(top_sv)
    return out


def _slim_read2(read2: dict) -> dict:
    """Keep scalar rank stats; truncate top_sv lists; aggregate matched-n draws."""
    out: dict[str, Any] = {
        k: read2.get(k) for k in ("matched_n_draws", "n_averaged", "n_per_example")
    }
    for grain in ("averaged", "per_example"):
        stats = read2.get(grain)
        if isinstance(stats, dict):
            out[grain] = _slim_spec_stats(stats)
    matched = read2.get("matched_n")
    if isinstance(matched, list):
        scalar_keys = ("stable_rank", "k50", "k90", "participation_ratio", "s1", "frob_sq")
        per_draw = []
        for entry in matched:
            stats = entry.get("stats", {}) if isinstance(entry, dict) else {}
            per_draw.append({"draw": entry.get("draw"), **{k: stats.get(k) for k in scalar_keys}})
        agg = {}
        for key in scalar_keys:
            vals = np.asarray(
                [d[key] for d in per_draw if isinstance(d.get(key), (int, float))],
                dtype=np.float64,
            )
            if vals.size:
                agg[key] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        out["matched_n"] = {"per_draw": per_draw, "aggregate": agg, "top_sv_dropped": True}
    return out


def _selection_null_slim(sel: dict) -> dict:
    out = {k: v for k, v in sel.items() if k != "persist_path"}
    persist = sel.get("persist_path")
    if persist:
        out["persist_file"] = Path(str(persist)).name
    return out


def _cross_cell_tables(kept: list[dict]) -> dict:  # noqa: C901
    """Derived cross-cell reads from fitted-map held-out R^2 (plan v5 section 4.5)."""
    r2: dict[tuple, float] = {}
    for unit in kept:
        key = (unit["cell"], unit["arm"], unit["fit_arm"], unit["layer"], unit["basis"])
        r2[key] = unit["fit"]["r2"]

    def get(cell, arm, fit_arm, layer, basis):
        return r2.get((cell, arm, fit_arm, layer, basis))

    axes = sorted({(k[1], k[3], k[4]) for k in r2})  # (arm, layer, basis)
    two_by_two = []
    for arm, layer, basis in axes:
        vals = {
            cell: get(cell, arm, "A", layer, basis)
            for cell in ("cell_inst_own", "cell_inst_pretext", "cell_pre_insttext", "cell_pre_own")
        }
        if any(v is None for v in vals.values()):
            continue
        io, ip, pi, po = (
            vals["cell_inst_own"],
            vals["cell_inst_pretext"],
            vals["cell_pre_insttext"],
            vals["cell_pre_own"],
        )
        two_by_two.append(
            {
                "arm": arm,
                "layer": layer,
                "basis": basis,
                "fit_arm": "A",
                "r2": vals,
                "model_transport_main_effect": (io + ip) / 2 - (pi + po) / 2,
                "text_policy_main_effect": (io + pi) / 2 - (ip + po) / 2,
                "interaction": io - ip - pi + po,
            }
        )
    pair_tables = {}
    for name, pairs in (
        (
            "carrier_floor_shuffled",
            (("cell_inst_own", "cell_inst_shuf"), ("cell_pre_own", "cell_pre_shuf")),
        ),
        (
            "f_transfer_claude",
            (("cell_inst_own", "cell_inst_claude"), ("cell_pre_own", "cell_pre_claude")),
        ),
    ):
        rows = []
        for arm, layer, basis in axes:
            for own_cell, other_cell in pairs:
                a = get(own_cell, arm, "A", layer, basis)
                b = get(other_cell, arm, "A", layer, basis)
                if a is None or b is None:
                    continue
                rows.append(
                    {
                        "arm": arm,
                        "layer": layer,
                        "basis": basis,
                        "own_cell": own_cell,
                        "comparison_cell": other_cell,
                        "r2_own": a,
                        "r2_comparison": b,
                        "delta": b - a,
                    }
                )
        pair_tables[name] = rows
    fit_arm_gap = []
    for cell, arm, fit_arm, layer, basis in sorted(r2):
        if fit_arm != "A":
            continue
        b_val = get(cell, arm, "B", layer, basis)
        if b_val is None:
            continue
        fit_arm_gap.append(
            {
                "cell": cell,
                "arm": arm,
                "layer": layer,
                "basis": basis,
                "r2_fit_arm_A": r2[(cell, arm, "A", layer, basis)],
                "r2_fit_arm_B": b_val,
                "delta_B_minus_A": b_val - r2[(cell, arm, "A", layer, basis)],
            }
        )
    prefix_context_gap = []
    cells = sorted({k[0] for k in r2})
    for cell in cells:
        for fit_arm in ("A", "B"):
            for layer in sorted({k[3] for k in r2}):
                for basis in ("ambient", "pca48"):
                    rp = get(cell, "prefix_end", fit_arm, layer, basis)
                    rc = get(cell, "context_end", fit_arm, layer, basis)
                    if rp is None or rc is None:
                        continue
                    prefix_context_gap.append(
                        {
                            "cell": cell,
                            "fit_arm": fit_arm,
                            "layer": layer,
                            "basis": basis,
                            "r2_prefix": rp,
                            "r2_context": rc,
                            "gap_context_minus_prefix": rc - rp,
                        }
                    )
    return {
        "two_by_two_decomposition": two_by_two,
        **pair_tables,
        "fit_arm_A_vs_B": fit_arm_gap,
        "prefix_context_gap": prefix_context_gap,
    }


def _expected_slots() -> set[tuple]:
    """Plan-v6 coverage: L14 x 8 cells x fitA+fitB; L18/L19 x 8 cells x fitA (2 arms x 2 bases)."""
    slots = set()
    for cell in CELL_LABELS:
        for arm in ("prefix_end", "context_end"):
            for basis in ("ambient", "pca48"):
                for fit_arm in ("A", "B"):
                    slots.add((cell, arm, fit_arm, 14, basis))
                for layer in (18, 19):
                    slots.add((cell, arm, "A", layer, basis))
    return slots


BATTERY_CONCERN_ID = "battery-rows-in-fit-arms-banked-p6"
# The realized corpus stratum label for the #594 battery bridge rows. The engine's
# fit-arm-A filter (fit_grid.py run(): stratum not in {"trait_stratum",
# "battery_eval_only"}) does NOT match it, so battery rows entered TRAINING in
# both fit arms across all banked P6 checkpoints (concern above, raised round 13).
BATTERY_STRATUM_REALIZED = "battery"


def _battery_scope_caveat(kept: list[dict], corpus_manifest: Path) -> dict:
    """Machine-readable plan-deviation caveat for concern battery-rows-in-fit-arms-banked-p6.

    Plan v5 section 4.1 step 6 registered the #594 battery rows EVAL-ONLY in both
    fit arms, with a battery TRANSFER read (OOD fold iii: fit on real corpus ->
    evaluate on #594 battery). The realized corpus labels the stratum "battery"
    (with is_eval_only=true), but the engine filter excludes only
    {"trait_stratum", "battery_eval_only"} — so battery rows were IN TRAINING for
    both fit arms in every banked checkpoint. This block verifies that from the
    banked units' own n_rows plus the realized manifest's stratum counts, records
    the recoverability investigation (round 22), and states what downstream
    consumers may / may not claim. It is embedded in EVERY merge-stage read-family
    JSON so the analyzer carries it into the clean-result mechanically.
    """
    observed: dict[str, dict[str, list[int]]] = {}
    for u in kept:
        cell = observed.setdefault(u["cell"], {})
        vals = cell.setdefault(u["fit_arm"], [])
        if int(u["n_rows"]) not in vals:
            vals.append(int(u["n_rows"]))
    observed = {
        c: {fa: sorted(v) for fa, v in sorted(d.items())} for c, d in sorted(observed.items())
    }

    caveat: dict[str, Any] = {
        "concern_id": BATTERY_CONCERN_ID,
        "severity": "CONCERN",
        "registered": {
            "plan": "v5 section 4.1 step 6 + OOD-fold registration (iii)",
            "battery_rows": "EVAL-ONLY in both fit arms (context-transfer read, factorial-wide)",
            "transfer_read": "fold iii: fit on real corpus -> evaluate on #594 battery "
            "(corpus transfer, strongest OOD form)",
        },
        "realized_in_banked_p6": {
            "engine_fit_arm_A_filter": "stratum not in {'trait_stratum', 'battery_eval_only'} "
            "(fit_grid.py run(); fit arm B takes all rows; is_eval_only read nowhere)",
            "corpus_stratum_label": BATTERY_STRATUM_REALIZED,
            "consequence": "battery rows were IN TRAINING for both fit arms in all banked "
            "checkpoints; the registered fold-iii transfer read has no implementation",
        },
        "observed_n_rows_by_cell_fit_arm": observed,
        "recoverability": {
            "investigated_round": 22,
            "banked_fit_block_fields": ["r2", "r2_folds", "lambda_indices"],
            "per_fold_predictions_persisted": False,
            "per_fold_coefficients_persisted": False,
            "per_row_heldout_predictions_or_residuals_persisted": False,
            "row_indexing_in_checkpoints": False,
            "verdict": "NOT recoverable from banked artifacts: under the grouped 6-fold CV "
            "each battery row was held out of exactly its own fold (and in TRAINING for the "
            "other 5 folds' maps), so an honest never-trained-on-this-row battery read existed "
            "in memory (_fit_cv computes per-row held-out pred) but only aggregate per-fold R2 "
            "was persisted; the aggregate r2_folds mix battery and non-battery rows and cannot "
            "be decomposed by stratum post hoc",
            "future_recovery_route": "folds are deterministic (_folds_from_manifest, group_key "
            "+ FOLD_SEED) and per-fold lambda indices are banked, so per-fold maps could be "
            "refit deterministically from the staged summaries — refit territory, out of scope "
            "for the banked grid (round-22 decision: no P6 re-run)",
        },
        "downstream_guidance": {
            "may_claim": [
                "fitted-map reads (read1-4, B1/B2 map-mediated) as WITHIN-CORPUS reads whose "
                "training rows include the 2,400 battery rows in both fit arms (composition "
                "documented here)",
                "held-out R2 under grouped 6-fold CV, with fold aggregates mixing battery and "
                "non-battery rows",
            ],
            "may_not_claim": [
                "the plan-v5 fold-iii battery TRANSFER read (fit on real corpus -> evaluate on "
                "#594 battery) — not computable from banked artifacts",
                "any 'battery eval-only' framing, including the #813 comparability framing — "
                "carry this caveat instead",
                "a battery-restricted held-out read from banked checkpoints (per-row held-out "
                "predictions were not persisted)",
                "the P7 A_real_only row arm as battery-free — it mirrors the engine rule and "
                "INCLUDES battery rows",
            ],
        },
    }

    if not corpus_manifest.exists():
        caveat["corpus_manifest"] = {"status": "absent", "path": str(corpus_manifest)}
        caveat["n_rows_arithmetic"] = {
            "status": "not_checkable_manifest_absent",
            "note": "per-cell battery-in-training verification needs the realized manifest",
        }
        return caveat

    strata: dict[str, int] = defaultdict(int)
    claude_strata: dict[str, int] = defaultdict(int)
    control_strata: dict[str, int] = defaultdict(int)
    battery_eval_only_flags: dict[str, int] = defaultdict(int)
    n_total = 0
    for row in _jsonl(corpus_manifest):
        n_total += 1
        stratum = str(row.get("stratum"))
        strata[stratum] += 1
        if row.get("claude_subset"):
            claude_strata[stratum] += 1
        if row.get("control_subset"):
            control_strata[stratum] += 1
        if stratum == BATTERY_STRATUM_REALIZED:
            battery_eval_only_flags[str(bool(row.get("is_eval_only")))] += 1

    scopes = {
        "full_corpus": dict(strata),
        "claude_subset": dict(claude_strata),
        "control_subset": dict(control_strata),
    }
    scope_totals = {name: sum(counts.values()) for name, counts in scopes.items()}
    per_cell: dict[str, dict] = {}
    for cell, arms in observed.items():
        fit_b = arms.get("B", [])
        fit_a = arms.get("A", [])
        entry: dict[str, Any] = {
            "n_rows_fitA_observed": fit_a,
            "n_rows_fitB_observed": fit_b,
        }
        scope_name = next(
            (name for name, tot in scope_totals.items() if len(fit_b) == 1 and fit_b[0] == tot),
            None,
        )
        if scope_name is None or len(fit_a) != 1:
            entry["status"] = "not_checkable"
            entry["note"] = (
                "no scope total matches the observed fitB n_rows (n0 truncation or partial "
                "grid) or fit-arm n_rows not unique"
            )
        else:
            counts = scopes[scope_name]
            trait_n = counts.get("trait_stratum", 0)
            battery_n = counts.get(BATTERY_STRATUM_REALIZED, 0)
            expected_engine = scope_totals[scope_name] - trait_n
            expected_registered = scope_totals[scope_name] - trait_n - battery_n
            entry.update(
                {
                    "status": "checked",
                    "scope_matched": scope_name,
                    "expected_fitA_engine_rule": expected_engine,
                    "expected_fitA_registered_rule": expected_registered,
                    "battery_rows_in_training": fit_a[0] == expected_engine
                    and fit_a[0] != expected_registered,
                }
            )
        per_cell[cell] = entry
    caveat["corpus_manifest"] = {
        "status": "present",
        "path": str(corpus_manifest),
        "sha256_16": _sha16_file(corpus_manifest),
        "n_rows": n_total,
        "strata_counts": dict(sorted(strata.items())),
        "battery_is_eval_only_flag_counts": dict(sorted(battery_eval_only_flags.items())),
    }
    caveat["n_rows_arithmetic"] = {
        "status": "computed",
        "scope_totals": scope_totals,
        "per_cell": per_cell,
    }
    return caveat


def step_merge(args: argparse.Namespace, hub) -> None:  # noqa: C901
    out_dir = Path(args.out_dir)
    work = Path(args.work_dir)
    stage_dir = work / "staging" / "checkpoints"
    boxes = [int(b) for b in _parse_csv(args.boxes, [str(i) for i in range(1, 13)])]
    by_name, box_of = _list_box_checkpoints(hub, args.hf_prefix, boxes)
    # Every LOCAL file the merge consumes rides the fingerprint (code-review v13
    # minor-2 class: a skip predicate must cover ALL consumed inputs — without
    # this, a landed bridge summary / refreshed judge scores kept bridge_refits
    # .json at "pending" until --force).
    corpus_manifest = Path(args.corpus_dir) / "manifest.jsonl"
    consumed_local = {
        "bridge_summary": Path(args.bridge_summary),
        "judge_scores": Path(args.judge_scores),
        "judge_summary": Path(args.judge_summary),
        "corpus_manifest": corpus_manifest,
    }
    fingerprint = _sha16_text(
        json.dumps(
            {
                "files": sorted((n, f.hub_identity, f.size) for n, f in by_name.items()),
                "boxes": boxes,
                "dedup_rule": "mlp-computed-then-min-fp:v1",
                "expect_full_grid": bool(args.expect_full_grid),
                "consumed_inputs": {
                    name: (_sha16_file(p) if p.exists() else None)
                    for name, p in consumed_local.items()
                },
                "battery_scope_caveat": BATTERY_CONCERN_ID,
            },
            sort_keys=True,
            default=list,
        )
    )
    manifest_path = out_dir / "merge_manifest.json"
    if not args.force and _fingerprint_matches(manifest_path, fingerprint):
        print(f"[p7-merge] up-to-date (fingerprint {fingerprint}); skipping", flush=True)
        return
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name, hf in sorted(by_name.items()):
        _stage_hub_file(hub, hf, stage_dir / name)
    units: list[dict] = []
    name_of: dict[str, str] = {}
    for name in sorted(by_name):
        m = CKPT_RE.match(name)
        if m is None:
            raise ValueError(f"unparseable checkpoint filename: {name}")
        unit = _load_json(stage_dir / name)
        for field, want in (
            ("cell", m["cell"]),
            ("arm", m["arm"]),
            ("fit_arm", m["fit"]),
            ("layer", int(m["layer"])),
            ("basis", m["basis"]),
            ("fingerprint", m["fp"]),
        ):
            if unit.get(field) != want:
                raise ValueError(f"{name}: field {field}={unit.get(field)!r} != filename {want!r}")
        name_of[unit["fingerprint"]] = name
        units.append(unit)
    kept, dedup_decisions = _dedupe_units(units)
    slots = {(u["cell"], u["arm"], u["fit_arm"], u["layer"], u["basis"]) for u in kept}
    coverage = {
        "n_checkpoints_staged": len(units),
        "n_unique_slots": len(slots),
        "n_duplicate_slots": len(dedup_decisions),
        "layers": sorted({u["layer"] for u in kept}),
        "cells": sorted({u["cell"] for u in kept}),
    }
    if args.expect_full_grid:
        missing = _expected_slots() - slots
        extra = slots - _expected_slots()
        if missing or extra:
            raise RuntimeError(
                f"--expect-full-grid: {len(missing)} missing slots, {len(extra)} unexpected; "
                f"missing sample: {sorted(missing)[:6]}"
            )
        coverage["full_grid_verified"] = True
    meta = _run_metadata(args)
    battery_caveat = _battery_scope_caveat(kept, corpus_manifest)

    def family(name: str, payload: dict) -> None:
        payload = {
            "metadata": meta,
            "merge_fingerprint": fingerprint,
            "battery_scope_caveat": battery_caveat,
            **payload,
        }
        _write_json_atomic(out_dir / name, payload)
        print(f"[p7-merge] wrote {out_dir / name}", flush=True)

    family(
        "battery_scope_caveat.json",
        {
            "read": "structured plan-deviation scope caveat (standalone copy; the same "
            "battery_scope_caveat block is embedded in every merge-stage family JSON)"
        },
    )

    def prov(u: dict) -> dict:
        return _prov(u, box_of, name_of[u["fingerprint"]])

    family(
        "read1_map_skill.json",
        {
            "read": "read1 prefix-vs-context map skill + identity ladder + refit perm nulls",
            "units": [
                {
                    "provenance": prov(u),
                    "r2": u["fit"]["r2"],
                    "r2_folds": u["fit"]["r2_folds"],
                    "lambda_indices": u["fit"]["lambda_indices"],
                    "identity_floors": u["identity_floors"],
                    "genuine_r2_over_diag": u.get("genuine_r2_over_diag"),
                    "perm_null": u["perm_null"],
                }
                for u in kept
            ],
        },
    )
    family("cross_cell.json", {"read": "derived cross-cell tables", **_cross_cell_tables(kept)})
    family(
        "read2_grain_rank.json",
        {
            "read": "read2 grain/rank spectra + matched-n control draws (top_sv truncated; "
            "full lists remain in the HF checkpoints)",
            "units": [
                {
                    "provenance": prov(u),
                    "spectrum": {
                        "lambda_gcv": u["spectrum"].get("lambda_gcv"),
                        "stats": _slim_spec_stats(u["spectrum"].get("stats", {})),
                    },
                    "grain_rank": _slim_read2(u["read2_matched_n_grain_rank"]),
                }
                for u in kept
            ],
        },
    )
    family(
        "read3_fgi_shares.json",
        {
            "read": "read3 f/g/i anova shares + refit twins + stitch/bare-query reads",
            "units": [
                {
                    "provenance": prov(u),
                    "anova_shares": u["anova_shares"],
                    "refit_twins": u["refit_twins"],
                    "stitch_bare_query": u["read3_stitch_bare_query"],
                }
                for u in kept
            ],
        },
    )
    family(
        "read4_operator_identity.json",
        {
            "read": "read4 operator identity + trait-per-factor selection-symmetric nulls",
            "units": [
                {
                    "provenance": prov(u),
                    "operator_identity": u["read4_operator_identity"],
                    "selection_symmetric_layer_max_null": _selection_null_slim(
                        u["selection_symmetric_layer_max_null"]
                    ),
                }
                for u in kept
            ],
        },
    )
    dyn_seen: dict[tuple, dict] = {}
    for u in kept:
        key = (u["cell"], u["layer"])
        if key not in dyn_seen:
            dyn_seen[key] = {
                "cell": u["cell"],
                "layer": u["layer"],
                "source_fingerprint": u["fingerprint"],
                "dynamics": u["dynamics_D0_D5"],
            }
    family(
        "dynamics_D0_D5.json",
        {
            "read": "dynamics D0-D5 + B3 (deduped per (cell, layer); identical across "
            "arm/fit-arm/basis units by engine construction)",
            "combos": [dyn_seen[k] for k in sorted(dyn_seen)],
        },
    )
    family(
        "behavior_B1_B2.json",
        {
            "read": "B1 frozen-fit-layer panel (a)-(e) + B2 factor-to-behavior "
            "(engine-computed at the unit's own fit layer)",
            "eligibility_rule": ELIGIBILITY_RULE,
            "units": [{"provenance": prov(u), "behavior": u["behavior_B1_B2"]} for u in kept],
        },
    )
    family(
        "mlp_companion.json",
        {
            "read": "MLP companion (cell_inst_own x context_end x fitA gating)",
            "units": [
                {"provenance": prov(u), "mlp_companion": u["mlp_companion"]}
                for u in kept
                if u.get("mlp_companion", {}).get("status") != "not_applicable"
            ],
        },
    )
    bridge_path = Path(args.bridge_summary)
    if bridge_path.exists():
        bridge = {
            "status": "present",
            "path": str(bridge_path),
            "sha256_16": _sha16_file(bridge_path),
            "summary": _load_json(bridge_path),
        }
    else:
        bridge = {
            "status": "pending",
            "expected_path": str(bridge_path),
            "note": "bridge re-fits (#923/#813/#779) still in flight; re-run "
            "--steps merge --force once bridge_refit_summary.json lands",
        }
    family("bridge_refits.json", {"read": "bridge re-fits passthrough", **bridge})
    judge_meta: dict[str, Any] = {"read": "P5 judge scores metadata"}
    scores_path = Path(args.judge_scores)
    if scores_path.exists():
        counts: dict[str, int] = defaultdict(int)
        n_rows = 0
        for row in _jsonl(scores_path):
            n_rows += 1
            if row.get("score") is not None:
                counts[f"{row.get('cell_id')}|{row.get('trait')}"] += 1
        judge_meta.update(
            {
                "scores_path": str(scores_path),
                "scores_sha256_16": _sha16_file(scores_path),
                "n_rows": n_rows,
                "scored_counts_by_cell_trait": dict(sorted(counts.items())),
            }
        )
    summary_path = Path(args.judge_summary)
    if summary_path.exists():
        judge_meta["p5_summary"] = _load_json(summary_path)
    family("judge_scores_meta.json", judge_meta)
    _write_json_atomic(
        manifest_path,
        {
            "metadata": meta,
            "input_fingerprint": fingerprint,
            "hub_revision": hub.resolved_revision(),
            "boxes": boxes,
            "coverage": coverage,
            "battery_scope_caveat": battery_caveat,
            "dedup_decisions": dedup_decisions,
            "checkpoints": {
                n: {"boxes": box_of[n], "hub_identity": by_name[n].hub_identity[:16]}
                for n in sorted(by_name)
            },
        },
    )
    print(
        f"[p7-merge] artifact digest: units={len(kept)} slots={len(slots)} "
        f"dups={len(dedup_decisions)} manifest={manifest_path}",
        flush=True,
    )


# --------------------------------------------------------------- projection sweep (stage 2)
def _merge_output_fingerprint(path: Path) -> str | None:
    """merge_fingerprint of a consumed merge-stage output (None when absent)."""
    if not path.exists():
        return None
    return _load_json(path).get("merge_fingerprint")


def _row_arm_mask(rows: list[dict], row_arm: str) -> np.ndarray:
    if row_arm == "A_real_only":
        keep = [r.get("stratum") not in FIT_ARM_A_EXCLUDED_STRATA for r in rows]
    elif row_arm == "B_all_rows":
        keep = [True for _ in rows]
    elif row_arm == "trait_battery_only":
        # Plan v5 section 6 C3 named fallback subset (designed trait variance).
        keep = [r.get("stratum") in {"trait_stratum", "battery"} for r in rows]
    else:
        raise ValueError(f"unknown row arm {row_arm!r}")
    return np.asarray(keep, dtype=bool)


def _judge_pairs_by_cell_trait(
    scores_path: Path, cells: list[str], row_index: dict[str, int]
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """(cell, trait) -> [(manifest_idx, score)] in judge-file order (engine join mirror)."""
    out: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            score = row.get("score")
            if score is None:
                continue
            row_id = str(row.get("row_id"))
            if row_id not in row_index:
                continue
            for cell in cells:
                if row.get("cell_id") == cell or row.get("arm") == cell:
                    out[(cell, str(row.get("trait")))].append((row_index[row_id], float(score)))
    return dict(out)


def _summaries_listing(hub, prefix: str, subdir: str) -> list[HubFile]:
    return hub.list_files(f"{prefix}/analysis_tensors/summaries/{subdir}")


def _files_for_kind_layer(listing: list[HubFile], kind: str, layer: int) -> list[HubFile]:
    """Sharded-else-unsharded resolution mirroring fit_grid._summary_shard_paths."""
    shard_re = re.compile(rf"^{re.escape(kind)}_L{layer:02d}_shard(\d+)\.npy$")
    shards = []
    for hf in listing:
        m = shard_re.match(hf.path.rsplit("/", 1)[1])
        if m:
            shards.append((int(m.group(1)), hf))
    if shards:
        return [hf for _i, hf in sorted(shards, key=lambda t: t[0])]
    flat = [hf for hf in listing if hf.path.rsplit("/", 1)[1] == f"{kind}_L{layer:02d}.npy"]
    if not flat:
        raise FileNotFoundError(f"no summary files for {kind} L{layer:02d}")
    return flat


def _load_staged_matrix(hub, files: list[HubFile], stage_root: Path) -> np.ndarray:
    """Stage-load-delete: download shard(s), concat rows as fp64, delete staged copies."""
    parts = []
    for hf in files:
        target = stage_root / hf.path
        _stage_hub_file(hub, hf, target)
        parts.append(np.load(target).astype(np.float64))
        target.unlink()
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _layer_projections(
    args: argparse.Namespace,
    hub,
    listings: dict[str, list[HubFile]],
    rb: np.ndarray,
    rb_identity: str,
    layer: int,
    cells: list[str],
    model_types: list[str],
) -> dict[str, np.ndarray]:
    """One layer's normalized r_B projections per input kind (stage-einsum-delete).

    Returns arrays keyed "{cell}__{kind}" (n_rows, n_traits) and "bare__{mt}"
    (n_bare_rows, n_traits), fp64, each projection divided by ||r_B|| (engine
    B1(a) convention; Pearson is scale-invariant so this is cosmetic).
    Cached per layer under work_dir/proj_cache with an input fingerprint.
    """
    work = Path(args.work_dir)
    cache_dir = work / "proj_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"L{layer:02d}.npz"
    meta_path = cache_dir / f"L{layer:02d}.meta.json"
    needed: dict[str, list[HubFile]] = {}
    for cell in cells:
        for kind in ("prefix_end", "context_end"):
            needed[f"{cell}__{kind}"] = _files_for_kind_layer(listings[cell], kind, layer)
    for mt in model_types:
        needed[f"bare__{mt}"] = _files_for_kind_layer(listings[f"bare_{mt}"], "c_q_bare", layer)
    fingerprint = _sha16_text(
        json.dumps(
            {
                "layer": layer,
                "rb_identity": rb_identity,
                "files": {
                    k: [(f.path, f.hub_identity, f.size) for f in v] for k, v in needed.items()
                },
            },
            sort_keys=True,
        )
    )
    if not args.force and npz_path.exists() and _fingerprint_matches(meta_path, fingerprint):
        with np.load(npz_path) as z:
            return {k: z[k] for k in z.files}
    rb_layer = rb[layer]  # (n_traits, hidden)
    norms = np.linalg.norm(rb_layer, axis=1)
    if np.any(norms == 0.0):
        raise ValueError(f"zero-norm r_B direction at layer {layer}")
    stage_root = work / "staging" / "summaries"
    out: dict[str, np.ndarray] = {}
    for key, files in needed.items():
        X = _load_staged_matrix(hub, files, stage_root)
        if X.shape[1] != rb_layer.shape[1]:
            raise ValueError(f"{key} L{layer}: dim {X.shape[1]} != r_B dim {rb_layer.shape[1]}")
        out[key] = (X @ rb_layer.T) / norms[None, :]
        del X
    tmp = npz_path.with_suffix(".npz.tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, npz_path)
    _write_json_atomic(meta_path, {"input_fingerprint": fingerprint, "layer": layer})
    print(f"[p7-sweep] layer L{layer:02d} projections cached", flush=True)
    return out


def _grain_vectors(
    x_rows: np.ndarray, scores: list[tuple[int, float]], prefix_ids: list[str], grain: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mirror fit_grid grain_arrays on projection vectors.

    x_rows: (n_manifest, L) per-row projection values for ONE trait/arm.
    scores: [(manifest_idx, score)] pairs (judge-file order).
    Returns (x (n_g, L), s (n_g,), n_groups).
    """
    if grain == "per_example":
        idx = np.asarray([p[0] for p in scores], dtype=np.int64)
        s = np.asarray([p[1] for p in scores], dtype=np.float64)
        return x_rows[idx], s, len(scores)
    if grain != "condition_averaged":
        raise ValueError(f"unknown grain {grain!r}")
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for idx_i, score in scores:
        grouped[str(prefix_ids[idx_i])].append((idx_i, score))
    xs, ss = [], []
    for _key, vals in sorted(grouped.items()):
        idxs = np.asarray([v[0] for v in vals], dtype=np.int64)
        xs.append(x_rows[idxs].mean(axis=0))
        ss.append(float(np.mean([v[1] for v in vals])))
    return np.asarray(xs, dtype=np.float64), np.asarray(ss, dtype=np.float64), len(xs)


def _eligibility(s: np.ndarray) -> dict:
    positives = int(np.sum(s > 50.0))
    negatives = int(s.size - positives)
    std = float(np.std(s)) if s.size else float("nan")
    return {
        "n_scored": int(s.size),
        "score_std": std,
        "n_positive": positives,
        "n_negative": negatives,
        "estimable": bool(s.size >= 5 and std >= 1.0 and positives >= 1 and negatives >= 1),
    }


def _observed_r_per_layer(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-layer Pearson r via the engine's _pearson_or_nan (exact continuity)."""
    return np.asarray([_pearson_or_nan(x[:, li], s) for li in range(x.shape[1])])


def _null_r_matrix(
    x: np.ndarray, s: np.ndarray, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Batched pairing-permutation null: (n_draws, L) Pearson r (one GEMM, no draw loop)."""
    n = s.size
    xc = x - x.mean(axis=0, keepdims=True)
    xn = np.linalg.norm(xc, axis=0)
    xz = np.divide(xc, xn[None, :], out=np.zeros_like(xc), where=xn[None, :] > 0)
    perm = rng.permuted(np.broadcast_to(s, (n_draws, n)).copy(), axis=1)
    pc = perm - perm.mean(axis=1, keepdims=True)
    pn = np.linalg.norm(pc, axis=1)
    pz = np.divide(pc, pn[:, None], out=np.zeros_like(pc), where=pn[:, None] > 0)
    r = pz @ xz  # (n_draws, L)
    r[:, xn == 0] = np.nan
    return r


def _family_read(
    *,
    family_key: str,
    x: np.ndarray,
    s: np.ndarray,
    layers: list[int],
    n_draws: int,
    seed: int,
    nulls_dir: Path,
) -> dict:
    """Observed per-layer r + same-selection layer-max null band for one family."""
    entry: dict[str, Any] = _eligibility(s)
    if not entry["estimable"]:
        return entry
    observed = _observed_r_per_layer(x, s)
    abs_obs = np.abs(observed)
    argmax = int(np.nanargmax(abs_obs)) if np.any(np.isfinite(abs_obs)) else None
    rng = _stable_rng("issue1092-p7-projection-null", seed, family_key)
    r_null = _null_r_matrix(x, s, n_draws, rng)
    max_abs = np.nanmax(np.abs(r_null), axis=1)
    persist = nulls_dir / f"{family_key}.npy"
    persist.parent.mkdir(parents=True, exist_ok=True)
    np.save(persist, r_null.astype(np.float32))
    entry.update(
        {
            "r_by_layer": {f"L{layer:02d}": float(observed[i]) for i, layer in enumerate(layers)},
            "max_abs_r": float(np.nanmax(abs_obs)) if argmax is not None else None,
            "argmax_layer": layers[argmax] if argmax is not None else None,
            "frozen_layers": {
                f"L{fl:02d}": float(observed[layers.index(fl)])
                for fl in FROZEN_FIT_LAYERS
                if fl in layers
            },
            "null": {
                "n_draws": int(n_draws),
                "selection": "per-draw max |r| over the swept layer axis (same-selection)",
                "p95_max_abs_r": float(np.nanpercentile(max_abs, 95)),
                "per_layer_p95_abs_r": [
                    float(v) for v in np.nanpercentile(np.abs(r_null), 95, axis=0)
                ],
                "persist_file": persist.name,
            },
        }
    )
    return entry


def _continuity_check(
    merged_behavior: dict | None,
    *,
    cell: str,
    trait: str,
    arm: str,
    row_arm: str,
    grain: str,
    observed_l14: float | None,
) -> dict | None:
    """Compare the P7 L14 raw-projection r against the engine's checkpointed value."""
    if merged_behavior is None or observed_l14 is None:
        return None
    fit_arm = {"A_real_only": "A", "B_all_rows": "B"}.get(row_arm)
    if fit_arm is None:
        return None  # trait_battery_only has no engine counterpart
    for unit in merged_behavior.get("units", []):
        p = unit["provenance"]
        if (
            p["cell"] == cell
            and p["fit_arm"] == fit_arm
            and p["layer"] == HEADLINE_LAYER
            and p["basis"] == "ambient"
            and p["arm"] == "context_end"
        ):
            engine = (
                unit["behavior"]
                .get("traits", {})
                .get(trait, {})
                .get("B1_by_arm_grain", {})
                .get(arm, {})
                .get(grain, {})
                .get("B1_raw_projection", {})
                .get("pearson_r")
            )
            if engine is None:
                return {"status": "engine_value_missing", "engine_unit": p["fingerprint"]}
            diff = abs(float(engine) - float(observed_l14))
            if diff > 1e-6:
                raise RuntimeError(
                    f"L14 continuity mismatch for {cell}/{trait}/{arm}/{row_arm}/{grain}: "
                    f"P7={observed_l14} engine={engine} (diff {diff:.3e}) — row alignment "
                    "or r_B mismatch; refusing to write a divergent sweep"
                )
            return {
                "status": "match",
                "engine_pearson_r": float(engine),
                "abs_diff": diff,
                "engine_unit": p["fingerprint"],
            }
    return {"status": "no_engine_unit_found"}


def _cross_fit_layer_band(args: argparse.Namespace, hub, merged_read4: dict, meta: dict) -> dict:
    """Conservative best-of-{L14,L18,L19} band (plan v6 section 4.5-A read-4 row).

    Per-unit null draws are seeded seed+fit-layer, so draws are NOT cross-layer
    aligned; the band is assembled as the per-draw-index max over the three
    units' independently-seeded draw vectors — a CONSERVATIVE band (independence
    across fit layers stochastically upper-bounds the max of the positively
    correlated true nulls, which share the same rows).
    """
    units = merged_read4["units"]
    by_triple: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for unit in units:
        p = unit["provenance"]
        if p["fit_arm"] != "A":
            continue
        by_triple[(p["cell"], p["arm"], p["basis"])][p["layer"]] = unit
    work = Path(args.work_dir)
    stage_root = work / "staging" / "p6_nulls"
    rows: list[dict] = []
    listings: dict[int, dict[str, HubFile]] = {}
    for (cell, arm, basis), by_layer in sorted(by_triple.items()):
        if not all(layer in by_layer for layer in FROZEN_FIT_LAYERS):
            continue
        draws_per_unit: dict[int, np.ndarray] = {}
        obs_per_unit: dict[int, np.ndarray] = {}
        factor_axis: list[str] = []
        trait_names: list[str] = []
        for layer in FROZEN_FIT_LAYERS:
            unit = by_layer[layer]
            sel = unit["selection_symmetric_layer_max_null"]
            factor_axis = list(sel["factor_axis"])
            trait_names = list(sel["trait_names"])
            observed = np.asarray(
                [
                    [
                        [
                            sel["observed_mean_projection"][factor][f"L{li:02d}"][trait]
                            for trait in trait_names
                        ]
                        for li in range(len(sel["layer_axis"]))
                    ]
                    for factor in factor_axis
                ],
                dtype=np.float64,
            )  # (factor, rb_layer, trait)
            obs_per_unit[layer] = np.max(np.abs(observed), axis=1)  # (factor, trait)
            persist_file = sel.get("persist_file")
            if not persist_file:
                raise KeyError(f"unit {unit['provenance']['fingerprint']} missing persist_file")
            boxes = unit["provenance"]["boxes"]
            staged = None
            for box in boxes:
                if box not in listings:
                    try:
                        files = hub.list_files(
                            f"{args.hf_prefix}/p6/box_{box:02d}/analysis_tensors/nulls"
                        )
                    except FileNotFoundError:
                        files = []
                    listings[box] = {f.path.rsplit("/", 1)[1]: f for f in files}
                hf = listings[box].get(persist_file)
                if hf is not None:
                    staged = stage_root / hf.path
                    _stage_hub_file(hub, hf, staged)
                    break
            if staged is None:
                raise FileNotFoundError(
                    f"null draws {persist_file} not found in boxes {boxes} of unit "
                    f"{unit['provenance']['fingerprint']}"
                )
            draws = np.load(staged).astype(np.float64)  # (n_draws, rb_layers, factor, trait)
            staged.unlink()
            draws_per_unit[layer] = draws.max(axis=1)  # (n_draws, factor, trait)
        n_draws = {layer: d.shape[0] for layer, d in draws_per_unit.items()}
        if len(set(n_draws.values())) != 1:
            raise ValueError(f"draw-count mismatch across fit layers for {cell}/{arm}: {n_draws}")
        stacked = np.stack([draws_per_unit[layer] for layer in FROZEN_FIT_LAYERS], axis=0)
        best_null = stacked.max(axis=0)  # (n_draws, factor, trait)
        obs_stack = np.stack([obs_per_unit[layer] for layer in FROZEN_FIT_LAYERS], axis=0)
        best_obs = obs_stack.max(axis=0)  # (factor, trait)
        best_layer = np.asarray(FROZEN_FIT_LAYERS)[obs_stack.argmax(axis=0)]
        for f_i, factor in enumerate(factor_axis):
            for t_i, trait in enumerate(trait_names):
                rows.append(
                    {
                        "cell": cell,
                        "arm": arm,
                        "basis": basis,
                        "fit_arm": "A",
                        "factor": factor,
                        "trait": trait,
                        "observed_best_abs_projection": float(best_obs[f_i, t_i]),
                        "best_fit_layer": int(best_layer[f_i, t_i]),
                        "per_fit_layer_observed": {
                            f"L{layer:02d}": float(obs_per_unit[layer][f_i, t_i])
                            for layer in FROZEN_FIT_LAYERS
                        },
                        "null_p95_conservative": float(
                            np.nanpercentile(best_null[:, f_i, t_i], 95)
                        ),
                        "n_draws": int(next(iter(n_draws.values()))),
                    }
                )
    return {
        "metadata": meta,
        "read": "read-4c cross-fit-layer best-of-{L14,L18,L19} secondary read",
        "band_construction": (
            "CONSERVATIVE: per-draw-index max over the three fit-layer units' "
            "independently-seeded (seed+layer) draw vectors; each per-unit draw already "
            "inherits the 28-layer read-out max (engine _selection_symmetric_projection_null)"
        ),
        "rows": rows,
    }


def step_projection_sweep(args: argparse.Namespace, hub) -> None:  # noqa: C901
    out_dir = Path(args.out_dir)
    work = Path(args.work_dir)
    t0 = time.monotonic()
    corpus_manifest = Path(args.corpus_dir) / "manifest.jsonl"
    rows = _jsonl(corpus_manifest)
    manifest_sha = _sha16_file(corpus_manifest)
    scores_path = Path(args.judge_scores)
    scores_sha = _sha16_file(scores_path)
    cells = _parse_csv(args.cells, JUDGED_CELLS)
    for cell in cells:
        if cell not in JUDGED_CELLS:
            raise ValueError(f"projection sweep runs on judged cells only, got {cell}")
    layers = _parse_layers(args.layers)
    rb_ns = argparse.Namespace(rb_dir=args.rb_dir, rb_rev=args.rb_rev, n_layers=28, hidden_dim=3584)
    rb, trait_names = _load_rb_directions(rb_ns)  # (28, n_traits, hidden)
    rb = np.transpose(rb, (0, 1, 2))
    rb_identity = (
        f"{args.rb_rev}:{hashlib.sha256(np.ascontiguousarray(rb).tobytes()).hexdigest()[:16]}"
    )
    model_types = sorted({CELL_MODEL_TYPE[c] for c in cells})
    # The sweep CONSUMES two merge-stage outputs (behavior_B1_B2.json feeds the
    # L14 continuity check embedded in b1a families; read4_operator_identity.json
    # feeds the cross-fit-layer band), so their merge fingerprints ride the sweep
    # fingerprint — a re-merged grid invalidates the sweep outputs instead of
    # leaving a stale band behind a bare band_path.exists() (code-review v13
    # minor 4).
    fingerprint = _sha16_text(
        json.dumps(
            {
                "layers": layers,
                "cells": cells,
                "rb_identity": rb_identity,
                "manifest_sha": manifest_sha,
                "scores_sha": scores_sha,
                "n_null_draws": args.n_null_draws,
                "seed": args.seed,
                "row_arms": ROW_ARMS,
                "consumed_merge_outputs": {
                    "behavior_B1_B2": _merge_output_fingerprint(out_dir / "behavior_B1_B2.json"),
                    "read4_operator_identity": _merge_output_fingerprint(
                        out_dir / "read4_operator_identity.json"
                    ),
                },
            },
            sort_keys=True,
        )
    )
    b1a_path = out_dir / "b1a_raw_projection_28layers.json"
    b1d_path = out_dir / "b1d_b0_poolings_28layers.json"
    band_path = out_dir / "read4c_cross_fit_layer_band.json"
    if (
        not args.force
        and _fingerprint_matches(b1a_path, fingerprint)
        and _fingerprint_matches(b1d_path, fingerprint)
        and _fingerprint_matches(band_path, fingerprint)
    ):
        print(f"[p7-sweep] up-to-date (fingerprint {fingerprint}); skipping", flush=True)
        return
    listings: dict[str, list[HubFile]] = {}
    for cell in cells:
        listings[cell] = _summaries_listing(hub, args.hf_prefix, cell)
    for mt in model_types:
        listings[f"bare_{mt}"] = _summaries_listing(hub, args.hf_prefix, f"bare_{mt}")

    # ---- per-layer stage-einsum-delete accumulation (checkpointed per layer)
    proj: dict[str, list[np.ndarray]] = defaultdict(list)
    for layer in layers:
        arrays = _layer_projections(args, hub, listings, rb, rb_identity, layer, cells, model_types)
        for key, arr in arrays.items():
            proj[key].append(arr)
    stacked = {key: np.stack(parts, axis=1) for key, parts in proj.items()}  # (n, L, traits)
    n0 = min(min(a.shape[0] for k, a in stacked.items() if not k.startswith("bare__")), len(rows))
    rows = rows[:n0]
    prefix_ids = [str(r.get("prefix_id", i)) for i, r in enumerate(rows)]
    row_index = {str(r.get("row_id")): i for i, r in enumerate(rows)}

    # bare row mapping per model type (engine _bare_X_for_unit mirror; fail-loud)
    bare_map: dict[str, np.ndarray] = {}
    for mt in model_types:
        idx_rows = _read_index_files_hub(hub, args, f"bare_{mt}")
        q_to_idx = {str(r["query_id"]): i for i, r in enumerate(idx_rows)}
        n_bare = stacked[f"bare__{mt}"].shape[0]
        if len(idx_rows) != n_bare:
            raise ValueError(f"bare_{mt} row_index count {len(idx_rows)} != rows {n_bare}")
        missing = [str(r.get("query_id")) for r in rows if str(r.get("query_id")) not in q_to_idx]
        if missing:
            raise KeyError(f"bare_{mt} missing {len(missing)} query ids: {missing[:5]}")
        bare_map[mt] = np.asarray([q_to_idx[str(r.get("query_id"))] for r in rows], dtype=np.int64)

    # ---- B0 pools (persisted (n_rows, 28, 3, 4); zero recompute)
    b0_pools: dict[str, np.ndarray] = {}
    for cell in cells:
        b0_files = [
            hf
            for hf in _summaries_listing(hub, args.hf_prefix, "b0_rB_pool")
            if hf.path.rsplit("/", 1)[1].startswith(cell)
        ]
        if not b0_files:
            raise FileNotFoundError(f"missing b0_rB_pool artifact for {cell}")
        stage_root = work / "staging" / "summaries"
        parts = []
        for hf in sorted(b0_files, key=lambda f: f.path):
            target = stage_root / hf.path
            _stage_hub_file(hub, hf, target)
            parts.append(np.load(target).astype(np.float64))
        pool = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        if pool.ndim != 4 or pool.shape[1] != 28 or pool.shape[3] != len(B0_MODES):
            raise ValueError(f"b0 pool for {cell} has unexpected shape {pool.shape}")
        b0_pools[cell] = pool

    pairs_by = _judge_pairs_by_cell_trait(scores_path, cells, row_index)
    merged_behavior = None
    behavior_json = out_dir / "behavior_B1_B2.json"
    if behavior_json.exists():
        merged_behavior = _load_json(behavior_json)
    meta = _run_metadata(args)
    nulls_dir = work / "analysis_tensors" / "nulls"
    common = {
        "metadata": meta,
        "input_fingerprint": fingerprint,
        "layers": layers,
        "rb": {"rev": args.rb_rev, "identity": rb_identity, "trait_names": trait_names},
        "corpus_manifest_sha256_16": manifest_sha,
        "judge_scores_sha256_16": scores_sha,
        "n_null_draws": int(args.n_null_draws),
        "eligibility_rule": ELIGIBILITY_RULE,
        "row_arm_rules": {
            "A_real_only": "stratum not in {trait_stratum, battery_eval_only} (engine fitA rule)",
            "B_all_rows": "all manifest rows (engine fitB rule)",
            "trait_battery_only": "stratum in {trait_stratum, battery} (section-6 C3 fallback)",
        },
        "seed_scheme": "sha256(family-key)->SeedSequence, base seed "
        f"{args.seed} (draws batched as one GEMM; no serial draw loops)",
        "nulls_dir": str(nulls_dir),
    }

    b1a_families = []
    b1d_families = []
    for cell in cells:
        mt = CELL_MODEL_TYPE[cell]
        traits_for_cell = sorted(t for (c, t) in pairs_by if c == cell)
        for trait in traits_for_cell:
            trait_i = trait_names.index(trait) if trait in trait_names else None
            if trait_i is None:
                raise KeyError(f"trait {trait!r} missing from r_B trait names {trait_names}")
            all_pairs = pairs_by[(cell, trait)]
            for row_arm in ROW_ARMS:
                mask = _row_arm_mask(rows, row_arm)
                pairs = [(i, s) for i, s in all_pairs if mask[i]]
                for arm in B1_ARMS:
                    if arm == "c_q_bare":
                        x_rows = stacked[f"bare__{mt}"][bare_map[mt]][:, :, trait_i]
                    else:
                        x_rows = stacked[f"{cell}__{arm}"][:n0, :, trait_i]
                    for grain in B1_GRAINS:
                        x, s, _n_g = _grain_vectors(x_rows, pairs, prefix_ids, grain)
                        family_key = f"b1a__{cell}__{trait}__{arm}__{row_arm}__{grain}"
                        entry = _family_read(
                            family_key=family_key,
                            x=x,
                            s=s,
                            layers=layers,
                            n_draws=args.n_null_draws,
                            seed=args.seed,
                            nulls_dir=nulls_dir,
                        )
                        obs_l14 = None
                        if entry.get("estimable") and HEADLINE_LAYER in layers:
                            obs_l14 = entry["r_by_layer"].get(f"L{HEADLINE_LAYER:02d}")
                        entry.update(
                            {
                                "cell": cell,
                                "trait": trait,
                                "arm": arm,
                                "row_arm": row_arm,
                                "grain": grain,
                                "l14_continuity_check": _continuity_check(
                                    merged_behavior,
                                    cell=cell,
                                    trait=trait,
                                    arm=arm,
                                    row_arm=row_arm,
                                    grain=grain,
                                    observed_l14=obs_l14,
                                ),
                            }
                        )
                        b1a_families.append(entry)
                # ---- B1(d) B0 poolings (arm-independent; generation-side reference)
                pool = b0_pools[cell]
                for mode_i, mode in enumerate(B0_MODES):
                    x_rows_pool = pool[:n0][:, [li for li in layers], trait_i, mode_i]
                    for grain in B1_GRAINS:
                        x, s, _n_g = _grain_vectors(x_rows_pool, pairs, prefix_ids, grain)
                        family_key = f"b1d__{cell}__{trait}__{mode}__{row_arm}__{grain}"
                        entry = _family_read(
                            family_key=family_key,
                            x=x,
                            s=s,
                            layers=layers,
                            n_draws=args.n_null_draws,
                            seed=args.seed,
                            nulls_dir=nulls_dir,
                        )
                        entry.update(
                            {
                                "cell": cell,
                                "trait": trait,
                                "pooling": mode,
                                "row_arm": row_arm,
                                "grain": grain,
                                "reference_note": (
                                    "B0 post-generation pooling — near-ceiling generation-side "
                                    "reference (#779 R2), own-policy cells only; not a "
                                    "pre-generation monitoring read"
                                ),
                            }
                        )
                        b1d_families.append(entry)
    _write_json_atomic(
        b1a_path,
        {
            **common,
            "read": "B1(a) raw projection r_B^T state, 28-layer sweep (v6-retained)",
            "families": b1a_families,
        },
    )
    print(f"[p7-sweep] wrote {b1a_path}", flush=True)
    _write_json_atomic(
        b1d_path,
        {
            **common,
            "read": "B1(d) B0 post-generation poolings, 28-layer sweep (v6-retained)",
            "b0_mode_axis": list(B0_MODES),
            "families": b1d_families,
        },
    )
    print(f"[p7-sweep] wrote {b1d_path}", flush=True)

    read4_json = out_dir / "read4_operator_identity.json"
    if read4_json.exists():
        band = _cross_fit_layer_band(args, hub, _load_json(read4_json), meta)
        band["input_fingerprint"] = fingerprint
        _write_json_atomic(band_path, band)
        print(f"[p7-sweep] wrote {band_path} ({len(band['rows'])} rows)", flush=True)
    else:
        print(
            "[p7-sweep] WARNING: read4_operator_identity.json absent — run --steps merge "
            "first; cross-fit-layer band skipped",
            flush=True,
        )
    n_est = sum(1 for f in b1a_families if f.get("estimable"))
    print(
        f"[p7-sweep] artifact digest: b1a_families={len(b1a_families)} "
        f"(estimable={n_est}) b1d_families={len(b1d_families)} "
        f"wall_s={time.monotonic() - t0:.1f}",
        flush=True,
    )


def _read_index_files_hub(hub, args: argparse.Namespace, subdir: str) -> list[dict]:
    """Stage a bare_* row_index (sharded else unsharded) locally, then parse it."""
    work = Path(args.work_dir)
    stage_root = work / "staging" / "summaries"
    listing = _summaries_listing(hub, args.hf_prefix, subdir)
    shard_re = re.compile(r"^row_index_shard(\d+)\.jsonl$")
    picked = []
    for hf in listing:
        name = hf.path.rsplit("/", 1)[1]
        m = shard_re.match(name)
        if m:
            picked.append((int(m.group(1)), hf))
    if picked:
        files = [hf for _i, hf in sorted(picked, key=lambda t: t[0])]
    else:
        files = [hf for hf in listing if hf.path.rsplit("/", 1)[1] == "row_index.jsonl"]
    if not files:
        raise FileNotFoundError(f"missing row_index for {subdir}")
    for hf in files:
        _stage_hub_file(hub, hf, stage_root / hf.path)
    root = stage_root / files[0].path.rsplit("/", 1)[0]
    return _read_index_files(root, "row_index")


# --------------------------------------------------------------------------- figures (stage 3)
def _match_units(units: list[dict], **kv: object) -> list[dict]:
    return [u for u in units if all(u["provenance"].get(k) == v for k, v in kv.items())]


def _first_unit(units: list[dict], **kv: object) -> dict | None:
    hits = _match_units(units, **kv)
    return hits[0] if hits else None


def _cells_present(units: list[dict]) -> list[str]:
    present = {u["provenance"]["cell"] for u in units}
    return [c for c in CELL_LABELS if c in present]


def _find_scalar(obj: object, key_names: tuple[str, ...]) -> float | None:
    """Depth-first search for the first numeric value under any of key_names."""
    if isinstance(obj, dict):
        for key in key_names:
            val = obj.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        for val in obj.values():
            found = _find_scalar(val, key_names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for val in obj:
            found = _find_scalar(val, key_names)
            if found is not None:
                return found
    return None


def step_figures(args: argparse.Namespace) -> None:  # noqa: C901
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    manifest: dict[str, Any] = {"metadata": _run_metadata(args), "figures": [], "skipped": []}

    def emit(fig, stem: str, note: str) -> None:
        written = savefig_paper(fig, stem, dir=figures_dir, formats=("png",))
        plt.close(fig)
        manifest["figures"].append({"stem": stem, "note": note, "png": str(written["png"])})
        print(f"[p7-figures] wrote {written['png']}", flush=True)

    def skip(stem: str, reason: str) -> None:
        manifest["skipped"].append({"stem": stem, "reason": reason})
        print(f"[p7-figures] SKIP {stem}: {reason}", flush=True)

    def maybe_load(name: str) -> dict | None:
        path = out_dir / name
        return _load_json(path) if path.exists() else None

    read1 = maybe_load("read1_map_skill.json")
    cross = maybe_load("cross_cell.json")
    read2 = maybe_load("read2_grain_rank.json")
    read3 = maybe_load("read3_fgi_shares.json")
    read4 = maybe_load("read4_operator_identity.json")
    dynamics = maybe_load("dynamics_D0_D5.json")
    behavior = maybe_load("behavior_B1_B2.json")
    b1a = maybe_load("b1a_raw_projection_28layers.json")
    b1d = maybe_load("b1d_b0_poolings_28layers.json")
    band = maybe_load("read4c_cross_fit_layer_band.json")
    bridge = maybe_load("bridge_refits.json")

    # ---- 1. read1: prefix-based vs context-based held-out R^2 per cell
    if read1 is not None:
        units = _match_units(read1["units"], fit_arm="A", layer=HEADLINE_LAYER, basis="ambient")
        cells = _cells_present(units)
        if cells:
            fig, ax = plt.subplots()
            width = 0.38
            xs = np.arange(len(cells))
            for off, (arm, role) in enumerate(
                (("prefix_end", "baseline"), ("context_end", "primary"))
            ):
                vals, errs = [], []
                for cell in cells:
                    u = _first_unit(units, cell=cell, arm=arm)
                    vals.append(u["r2"] if u else np.nan)
                    errs.append(float(np.std(u["r2_folds"])) if u else np.nan)
                ax.bar(
                    xs + (off - 0.5) * width,
                    vals,
                    width,
                    yerr=[max(0.0, e) if np.isfinite(e) else 0.0 for e in errs],
                    label=ARM_LABELS[arm],
                    color=paper_palette_role(role),
                    capsize=3,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels([CELL_LABELS[c] for c in cells], rotation=30, ha="right")
            ax.set_ylabel("held-out R² (grouped 6-fold by prefix)")
            ax.set_title(
                "Map skill by condition: prefix-based vs context-based input "
                "(fit-arm A, layer 14, ambient)"
            )
            ax.legend()
            emit(fig, "read1_r2_prefix_vs_context", "read1 headline: paired R² per cell")
        else:
            skip("read1_r2_prefix_vs_context", "no fitA L14 ambient units in read1 JSON")

        # ---- 2. train-mean floor vs fitted
        if cells:
            fig, ax = plt.subplots()
            xs = np.arange(len(cells))
            fitted, floor = [], []
            for cell in cells:
                u = _first_unit(units, cell=cell, arm="context_end")
                fitted.append(u["r2"] if u else np.nan)
                floor.append(u["identity_floors"]["train_mean"]["mean"] if u else np.nan)
            ax.bar(
                xs - 0.19,
                fitted,
                0.38,
                label="fitted ridge map",
                color=paper_palette_role("primary"),
            )
            ax.bar(
                xs + 0.19,
                floor,
                0.38,
                label="train-mean floor",
                color=paper_palette_role("neutral"),
            )
            ax.set_xticks(xs)
            ax.set_xticklabels([CELL_LABELS[c] for c in cells], rotation=30, ha="right")
            ax.set_ylabel("held-out R²")
            ax.set_title(
                "Fitted map vs trivial-transport floor "
                "(context-based, fit-arm A, layer 14, ambient)"
            )
            ax.legend()
            emit(fig, "read1_identity_floor", "fitted R² vs train-mean floor per cell")

        # ---- 3. layer sensitivity (frozen layers)
        lu = _match_units(read1["units"], fit_arm="A", basis="ambient", arm="context_end")
        layers_avail = sorted({u["provenance"]["layer"] for u in lu})
        if len(layers_avail) >= 2:
            fig, ax = plt.subplots()
            colors = paper_palette(len(_cells_present(lu)))
            for i, cell in enumerate(_cells_present(lu)):
                ys = [
                    (_first_unit(lu, cell=cell, layer=layer) or {}).get("r2", np.nan)
                    for layer in layers_avail
                ]
                ax.plot(layers_avail, ys, marker="o", label=CELL_LABELS[cell], color=colors[i])
            ax.set_xticks(layers_avail)
            ax.set_xlabel("fit layer")
            ax.set_ylabel("held-out R²")
            ax.set_title("Frozen-layer sensitivity (context-based, fit-arm A, ambient)")
            ax.legend(fontsize=7)
            emit(fig, "read1_layer_sensitivity", "R² across frozen fit layers per cell")
        else:
            skip("read1_layer_sensitivity", "fewer than 2 fit layers present")

    # ---- 4. read2: rank by grain
    if read2 is not None:
        units = _match_units(
            read2["units"], fit_arm="A", layer=HEADLINE_LAYER, basis="ambient", arm="context_end"
        )
        cells = _cells_present(units)
        if cells:
            fig, ax = plt.subplots()
            xs = np.arange(len(cells))
            series = {
                "condition averaged": ("averaged", "primary"),
                "per example": ("per_example", "accent"),
            }
            width = 0.28
            for off, (label, (grain_key, role)) in enumerate(series.items()):
                vals = []
                for cell in cells:
                    u = _first_unit(units, cell=cell)
                    vals.append(
                        (u["grain_rank"].get(grain_key) or {}).get("stable_rank", np.nan)
                        if u
                        else np.nan
                    )
                ax.bar(
                    xs + (off - 1) * width, vals, width, label=label, color=paper_palette_role(role)
                )
            matched_vals, matched_err = [], []
            for cell in cells:
                u = _first_unit(units, cell=cell)
                agg = (u["grain_rank"].get("matched_n") or {}).get("aggregate", {}) if u else {}
                matched_vals.append(agg.get("stable_rank", {}).get("mean", np.nan))
                matched_err.append(agg.get("stable_rank", {}).get("std", np.nan))
            ax.bar(
                xs + width,
                matched_vals,
                width,
                yerr=[max(0.0, e) if np.isfinite(e) else 0.0 for e in matched_err],
                label="matched-n control (per-example subsample)",
                color=paper_palette_role("control"),
                capsize=3,
            )
            ax.set_xticks(xs)
            ax.set_xticklabels([CELL_LABELS[c] for c in cells], rotation=30, ha="right")
            ax.set_ylabel("stable rank of the fitted map spectrum")
            ax.set_title("Grain and rank (context-based, fit-arm A, layer 14, ambient)")
            ax.legend(fontsize=7)
            emit(fig, "read2_rank_by_grain", "stable rank: averaged vs per-example vs matched-n")
        else:
            skip("read2_rank_by_grain", "no fitA L14 ambient context units in read2 JSON")

    # ---- 5. read3: f/g/i shares (ambient + pca48), refit-twin whiskers
    if read3 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        drew_any = False
        for ax, basis in zip(axes, ("ambient", "pca48"), strict=True):
            units = _match_units(
                read3["units"], fit_arm="A", layer=HEADLINE_LAYER, basis=basis, arm="context_end"
            )
            cells = _cells_present(units)
            if not cells:
                ax.set_title(f"{basis}: no units")
                continue
            drew_any = True
            xs = np.arange(len(cells))
            width = 0.28
            share_keys = ("share_prefix", "share_query", "share_interaction")
            share_labels = ("prefix share", "query share", "interaction share")
            colors = paper_palette(3)
            for off, (share_key, label) in enumerate(zip(share_keys, share_labels, strict=True)):
                vals, errs = [], []
                for cell in cells:
                    u = _first_unit(units, cell=cell)
                    variants = []
                    if u:
                        variants.append(u["anova_shares"].get(share_key))
                        for twin in u["refit_twins"].values():
                            variants.append(twin.get(share_key))
                    variants = [v for v in variants if isinstance(v, (int, float))]
                    vals.append(variants[0] if variants else np.nan)
                    errs.append(float(np.std(variants)) if len(variants) > 1 else 0.0)
                ax.bar(
                    xs + (off - 1) * width,
                    vals,
                    width,
                    yerr=errs,
                    label=label,
                    color=colors[off],
                    capsize=2,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels([CELL_LABELS[c] for c in cells], rotation=30, ha="right")
            ax.set_ylabel("variance share on dense core")
            ax.set_title(f"target basis: {basis}")
            ax.legend(fontsize=7)
        fig.suptitle("f/g/i decomposition shares (fit-arm A, layer 14; whiskers = refit twins)")
        if drew_any:
            emit(fig, "read3_fgi_shares", "anova shares per cell, ambient + pca48")
        else:
            plt.close(fig)
            skip("read3_fgi_shares", "no fitA L14 units in read3 JSON")

    # ---- 6. cross-cell 2x2 decomposition
    if cross is not None and not cross.get("two_by_two_decomposition"):
        skip("cross_cell_2x2_effects", "no complete own-text 2x2 cell set in merge output")
    if cross is not None and cross.get("two_by_two_decomposition"):
        rows = [
            r
            for r in cross["two_by_two_decomposition"]
            if r["layer"] == HEADLINE_LAYER and r["basis"] == "ambient"
        ]
        if rows:
            fig, ax = plt.subplots()
            effects = ("model_transport_main_effect", "text_policy_main_effect", "interaction")
            labels = ("model transport", "text policy", "interaction")
            xs = np.arange(len(effects))
            width = 0.38
            for off, arm in enumerate(("prefix_end", "context_end")):
                row = next((r for r in rows if r["arm"] == arm), None)
                vals = [row[e] if row else np.nan for e in effects]
                ax.bar(
                    xs + (off - 0.5) * width,
                    vals,
                    width,
                    label=ARM_LABELS[arm],
                    color=paper_palette_role("baseline" if off == 0 else "primary"),
                )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_ylabel("held-out R² effect (own-text 2x2)")
            ax.set_title(
                "Instruction-tuning 2x2: main effects on map skill (fit-arm A, layer 14, ambient)"
            )
            ax.legend()
            emit(fig, "cross_cell_2x2_effects", "2x2 main effects + interaction")
        else:
            skip("cross_cell_2x2_effects", "no complete 2x2 at L14 ambient")

    # ---- 7. read4 operator residuals vs random-map null
    if read4 is not None:
        units = _match_units(
            read4["units"], fit_arm="A", layer=HEADLINE_LAYER, basis="ambient", arm="context_end"
        )
        cells = _cells_present(units)
        if cells:
            fig, ax = plt.subplots()
            xs = np.arange(len(cells))
            width = 0.38
            resid_keys = (
                ("residual_interaction_norm_over_total", "residual interaction ÷ total"),
                ("mprime_minus_m_minus_g_over_g", "norm(M' - M - g) / norm(g)"),
            )
            for off, (key, label) in enumerate(resid_keys):
                vals = [
                    (_first_unit(units, cell=cell) or {}).get("operator_identity", {}).get(key)
                    for cell in cells
                ]
                vals = [v if isinstance(v, (int, float)) else np.nan for v in vals]
                ax.bar(
                    xs + (off - 0.5) * width,
                    vals,
                    width,
                    label=label,
                    color=paper_palette_role("primary" if off == 0 else "accent"),
                )
            null_p05 = [
                (_first_unit(units, cell=cell) or {})
                .get("operator_identity", {})
                .get("random_map_pairing_null", {})
                .get("p05")
                for cell in cells
            ]
            ax.scatter(
                xs,
                [v if isinstance(v, (int, float)) else np.nan for v in null_p05],
                marker="x",
                color=paper_palette_role("neutral"),
                label="random-map null p5 (registered test)",
                zorder=3,
            )
            ax.set_xticks(xs)
            ax.set_xticklabels([CELL_LABELS[c] for c in cells], rotation=30, ha="right")
            ax.set_ylabel("residual norm ratio")
            ax.set_title(
                "Operator-identity residuals vs random-map null "
                "(context-based, fit-arm A, layer 14, ambient)"
            )
            ax.legend(fontsize=7)
            emit(fig, "read4_operator_residuals", "H-A-bearing operator residuals + null")
        else:
            skip("read4_operator_residuals", "no fitA L14 ambient context units in read4 JSON")

    # ---- 8. read4c trait-per-factor heatmap (headline unit)
    if read4 is not None:
        u = _first_unit(
            read4["units"],
            cell="cell_inst_own",
            arm="context_end",
            fit_arm="A",
            layer=HEADLINE_LAYER,
            basis="ambient",
        )
        if u is not None:
            sel = u["selection_symmetric_layer_max_null"]
            factors = sel["factor_axis"]
            traits = sel["trait_names"]
            obs = np.asarray(
                [
                    [
                        max(
                            abs(sel["observed_mean_projection"][f][f"L{li:02d}"][t])
                            for li in range(len(sel["layer_axis"]))
                        )
                        for t in traits
                    ]
                    for f in factors
                ]
            )
            fig, ax = plt.subplots(layout="constrained")
            im = ax.imshow(obs / sel["max_abs_p95"], cmap="viridis")
            ax.set_xticks(range(len(traits)))
            ax.set_xticklabels(traits)
            ax.set_yticks(range(len(factors)))
            ax.set_yticklabels([f"{f} factor" for f in factors])
            ax.set_title(
                "Trait per factor: max |r_B projection| over 28 read-out layers,\n"
                "÷ same-selection null p95 (Instruct, own answers; fit-arm A, L14)"
            )
            fig.colorbar(im, ax=ax, label="observed ÷ null p95")
            emit(fig, "read4c_trait_per_factor", "trait-per-factor heatmap vs null")
        else:
            skip("read4c_trait_per_factor", "headline unit absent from read4 JSON")

    # ---- 9. dynamics D4 turn profiles (both input arms: context state + prefix state)
    if dynamics is not None:
        arm_specs = (
            ("context_k", "context state → same-turn answer"),
            ("s_k", "prefix state → same-turn answer"),
        )
        fig, axes9 = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
        drew = 0
        combos = [c for c in dynamics["combos"] if c["layer"] == HEADLINE_LAYER]
        colors = paper_palette(max(len(combos), 1))
        for ax, (arm_key, arm_title) in zip(axes9, arm_specs, strict=True):
            for i, combo in enumerate(combos):
                d4 = combo.get("dynamics", {}).get("D4_turn_profiles", {})
                answer_side = d4.get("answer_side", {})
                src = answer_side.get(arm_key) or {}
                target = next((v for k, v in sorted(src.items()) if "t1" in k), None)
                profiles = (target or {}).get("turn_profiles", {})
                turns, ys = [], []
                for turn_key, entry in sorted(profiles.items(), key=lambda kv: int(kv[0])):
                    r2 = entry.get("fit", {}).get("r2")
                    if isinstance(r2, (int, float)):
                        turns.append(int(turn_key))
                        ys.append(r2)
                if turns:
                    ax.plot(
                        turns,
                        ys,
                        marker="o",
                        markersize=3,
                        label=CELL_LABELS.get(combo["cell"], combo["cell"]),
                        color=colors[i],
                    )
                    drew += 1
            ax.set_xlabel("user-turn index")
            ax.set_title(arm_title, fontsize=10)
            ax.axhline(0.0, color="black", lw=0.6)
        axes9[0].set_ylabel("held-out R² (per-turn map, layer 14)")
        if drew:
            axes9[0].legend(fontsize=6.5)
            emit(fig, "dynamics_d4_turn_profiles", "D4 per-turn R² per cell, both input arms")
        else:
            plt.close(fig)
            skip("dynamics_d4_turn_profiles", "no computed D4 turn profiles at L14")

    # ---- 10/11. B1(a) layer curves + B1(d) B0 pooling curves
    for fam_json, stem_prefix, series_key, series_values, note in (
        (b1a, "b1a_layer_curves", "arm", B1_ARMS, "B1(a) raw-projection 28-layer sweep"),
        (b1d, "b1d_b0_pooling_curves", "pooling", B0_MODES, "B1(d) B0 pooling 28-layer sweep"),
    ):
        if fam_json is None:
            skip(stem_prefix, "stage-2 JSON absent (run --steps projection-sweep)")
            continue
        layers = fam_json["layers"]
        fams = [f for f in fam_json["families"] if f.get("estimable")]
        cells = sorted({f["cell"] for f in fams})
        for cell in cells:
            traits = sorted({f["trait"] for f in fams if f["cell"] == cell})
            if not traits:
                continue
            fig, axes = plt.subplots(
                1,
                len(traits),
                figsize=(4.2 * len(traits), 3.6),
                layout="constrained",
                squeeze=False,
            )
            colors = paper_palette(len(series_values))
            for t_i, trait in enumerate(traits):
                ax = axes[0][t_i]
                for s_i, sval in enumerate(series_values):
                    fam = next(
                        (
                            f
                            for f in fams
                            if f["cell"] == cell
                            and f["trait"] == trait
                            and f.get(series_key) == sval
                            and f["row_arm"] == "B_all_rows"
                            and f["grain"] == "condition_averaged"
                        ),
                        None,
                    )
                    if fam is None:
                        continue
                    ys = [fam["r_by_layer"].get(f"L{layer:02d}", np.nan) for layer in layers]
                    label = ARM_LABELS.get(sval, f"{sval} pooling")
                    ax.plot(layers, ys, label=label, color=colors[s_i])
                    if s_i == 0:
                        p95 = fam["null"]["per_layer_p95_abs_r"]
                        ax.plot(
                            layers,
                            p95,
                            linestyle="--",
                            linewidth=0.9,
                            color=paper_palette_role("neutral"),
                            label="per-layer null p95 |r| (200 draws)",
                        )
                        ax.plot(
                            layers,
                            [-v for v in p95],
                            linestyle="--",
                            linewidth=0.9,
                            color=paper_palette_role("neutral"),
                        )
                ax.axhline(0.0, color="black", linewidth=0.6)
                ax.set_title(f"{trait}")
                ax.set_xlabel("read-out layer")
                if t_i == 0:
                    ax.set_ylabel("Pearson r vs judged trait score")
                ax.legend(fontsize=6)
            fig.suptitle(f"{note} — {CELL_LABELS.get(cell, cell)} (condition averaged, all rows)")
            emit(fig, f"{stem_prefix}_{cell}", f"{note} for {cell}")
        if not cells:
            skip(stem_prefix, "no estimable stage-2 families")

    # ---- 12. B1 frozen-layer five-read panel per judged cell
    if behavior is not None:
        for cell in JUDGED_CELLS:
            u = _first_unit(
                behavior["units"],
                cell=cell,
                arm="context_end",
                fit_arm="A",
                layer=HEADLINE_LAYER,
                basis="ambient",
            )
            if u is None:
                skip(f"b1_frozen_panel_{cell}", "headline behavior unit absent")
                continue
            traits = u["behavior"].get("traits", {})
            est_traits = [t for t, e in sorted(traits.items()) if e.get("estimable")]
            if not est_traits:
                skip(f"b1_frozen_panel_{cell}", "no estimable traits")
                continue
            reads = (
                ("B1_raw_projection", "raw projection (a)"),
                ("B1_map_mediated", "map-mediated (b)"),
                ("B1_A2_answer_side_ceiling", "answer-side ceiling (e)"),
            )
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
            ax = axes[0]
            groups = [(trait, arm) for trait in est_traits for arm in B1_ARMS]
            xs = np.arange(len(groups))
            width = 0.22
            colors = paper_palette(len(reads) + 1)
            for r_i, (read_key, label) in enumerate(reads):
                vals = []
                for trait, arm in groups:
                    grain_entry = (
                        traits[trait]
                        .get("B1_by_arm_grain", {})
                        .get(arm, {})
                        .get("condition_averaged", {})
                    )
                    vals.append((grain_entry.get(read_key) or {}).get("pearson_r", np.nan))
                ax.bar(xs + (r_i - 1) * width, vals, width, label=label, color=colors[r_i])
            b0_vals = []
            for trait, arm in groups:
                grain_entry = (
                    traits[trait]
                    .get("B1_by_arm_grain", {})
                    .get(arm, {})
                    .get("condition_averaged", {})
                )
                b0_vals.append(
                    (grain_entry.get("B1_B0_poolings", {}).get("mean") or {}).get(
                        "pearson_r", np.nan
                    )
                )
            ax.scatter(
                xs,
                b0_vals,
                marker="D",
                s=18,
                color=colors[len(reads)],
                label="B0 mean pooling (d) — generation-side reference",
                zorder=3,
            )
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(
                [f"{t}\n{ARM_LABELS[a].split(' ')[0]}" for t, a in groups], fontsize=6
            )
            ax.set_ylabel("within-condition Pearson r")
            ax.set_title("correlation reads (condition averaged)")
            ax.legend(fontsize=6)
            ax2 = axes[1]
            vals = []
            for trait, arm in groups:
                grain_entry = (
                    traits[trait]
                    .get("B1_by_arm_grain", {})
                    .get(arm, {})
                    .get("condition_averaged", {})
                )
                vals.append((grain_entry.get("B1_direct_regression") or {}).get("r2", np.nan))
            ax2.bar(
                xs, vals, 0.5, color=paper_palette_role("accent"), label="direct regression (c)"
            )
            ax2.axhline(0.0, color="black", linewidth=0.6)
            ax2.set_xticks(xs)
            ax2.set_xticklabels(
                [f"{t}\n{ARM_LABELS[a].split(' ')[0]}" for t, a in groups], fontsize=6
            )
            ax2.set_ylabel("held-out R² (grouped folds)")
            ax2.set_title("direct supervised regression (c)")
            ax2.legend(fontsize=6)
            fig.suptitle(
                f"B1 monitoring panel at layer 14 — {CELL_LABELS[cell]} (fit-arm A, ambient)"
            )
            emit(fig, f"b1_frozen_panel_{cell}", "B1 five-read frozen-layer panel")

    # ---- 13. judge score distributions
    scores_path = Path(args.judge_scores)
    if scores_path.exists():
        by_cell_trait: dict[tuple[str, str], list[float]] = defaultdict(list)
        with open(scores_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("score") is None or row.get("cell_id") not in JUDGED_CELLS:
                    continue
                by_cell_trait[(row["cell_id"], str(row.get("trait")))].append(float(row["score"]))
        cells = sorted({c for c, _t in by_cell_trait})
        traits = sorted({t for _c, t in by_cell_trait})
        if cells and traits:
            fig, axes = plt.subplots(
                len(cells),
                len(traits),
                figsize=(3.6 * len(traits), 2.8 * len(cells)),
                layout="constrained",
                squeeze=False,
            )
            for c_i, cell in enumerate(cells):
                for t_i, trait in enumerate(traits):
                    ax = axes[c_i][t_i]
                    vals = by_cell_trait.get((cell, trait), [])
                    ax.hist(vals, bins=20, range=(0, 100), color=paper_palette_role("primary"))
                    ax.set_yscale("log")
                    ax.set_title(f"{CELL_LABELS.get(cell, cell)} — {trait}", fontsize=8)
                    if c_i == len(cells) - 1:
                        ax.set_xlabel("graded judge score (0-100)")
                    if t_i == 0:
                        ax.set_ylabel("count (log)")
            fig.suptitle("Judged trait-score distributions (5-draw means, own-policy cells)")
            emit(fig, "judge_score_distributions", "per (cell, trait) score histograms")
    else:
        skip("judge_score_distributions", f"judge scores absent at {scores_path}")

    # ---- 14. cross-fit-layer conservative band
    if band is not None and not band.get("rows"):
        skip("read4c_cross_fit_layer_band", "no fitA triples with all three frozen layers")
    if band is not None and band.get("rows"):
        rows = [
            r
            for r in band["rows"]
            if r["cell"] == "cell_inst_own"
            and r["arm"] == "context_end"
            and r["basis"] == "ambient"
        ]
        if rows:
            fig, ax = plt.subplots()
            labels = [f"{r['factor']} / {r['trait']}" for r in rows]
            xs = np.arange(len(rows))
            ax.bar(
                xs,
                [r["observed_best_abs_projection"] for r in rows],
                0.55,
                color=paper_palette_role("primary"),
                label="observed best-of-{L14,L18,L19}",
            )
            ax.scatter(
                xs,
                [r["null_p95_conservative"] for r in rows],
                marker="x",
                color=paper_palette_role("neutral"),
                label="conservative null p95 (per-draw max)",
                zorder=3,
            )
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("max |r_B projection| over read-out layers")
            ax.set_title(
                "Cross-fit-layer trait-per-factor read — conservative band\n"
                "(Instruct, own answers; context-based, fit-arm A, ambient)"
            )
            ax.legend(fontsize=7)
            emit(fig, "read4c_cross_fit_layer_band", "best-of fit layers vs conservative null")
        else:
            skip("read4c_cross_fit_layer_band", "no band rows for the headline cell")

    # ---- 15. bridge refits: one bar per re-fit item, grouped by parent substrate
    if bridge is not None and bridge.get("status") == "present":
        summary = bridge.get("summary", {})
        beh_label = {"em": "EM", "fact": "fact", "marker": "marker", "sycophancy": "sycophancy"}
        qset_label = {"elicit": "eliciting", "generic": "generic", "mix": "mixed"}
        sub923_label = {"uc48": "UltraChat-48 grid", "betley": "Betley questions"}

        def _r2_items(key: str) -> list[dict]:
            block = summary.get(key)
            if not isinstance(block, dict):
                return []
            return [
                i for i in block.get("items", []) if isinstance(i.get("headline_r2"), (int, float))
            ]

        groups: list[tuple[str, list[tuple[str, float]]]] = []
        items923 = sorted(_r2_items("issue923"), key=lambda i: i.get("substrate") != "uc48")
        if items923:
            groups.append(
                (
                    "UltraChat grid (issue 923)",
                    [
                        (
                            sub923_label.get(i.get("substrate"), str(i.get("substrate"))),
                            i["headline_r2"],
                        )
                        for i in items923
                    ],
                )
            )
        items779 = _r2_items("issue779")
        if items779:
            groups.append(
                (
                    "LMSYS persona map (issue 779)",
                    [("LMSYS persona map", i["headline_r2"]) for i in items779],
                )
            )
        items813 = _r2_items("issue813")
        if items813:
            entries813 = []
            for i in items813:
                qset = Path(str(i.get("source", ""))).parent.name
                beh = beh_label.get(i.get("behavior"), str(i.get("behavior")))
                entries813.append((f"{beh} {qset_label.get(qset, qset)}", i["headline_r2"]))
            groups.append(("unified-recipe re-fit (issue 813)", entries813))
        if groups:
            fig, ax = plt.subplots(figsize=(9.6, 4.8))
            colors = paper_palette(len(groups))
            tick_xs: list[float] = []
            tick_labels: list[str] = []
            pos = 0.0
            for gi, (gname, group_items) in enumerate(groups):
                xs = [pos + j for j in range(len(group_items))]
                vals = [float(v) for _lbl, v in group_items]
                ax.bar(xs, vals, 0.72, color=colors[gi], label=gname)
                for x, v in zip(xs, vals, strict=True):
                    ax.text(
                        x,
                        v + (0.05 if v >= 0 else -0.05),
                        f"{v:+.2f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=6.5,
                    )
                tick_xs.extend(xs)
                tick_labels.extend(lbl for lbl, _v in group_items)
                pos += len(group_items) + 1.4
            ax.set_xticks(tick_xs)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
            ax.axhline(0.0, color="black", lw=0.6)
            ax.set_ylabel("held-out R² at layer 14 (one bar per re-fit item)")
            ax.set_title("Bridge re-fits on parent substrates (one recipe, three corpora)")
            ax.legend(fontsize=7)
            emit(fig, "bridge_refits", "bridge re-fit skill per re-fit item, grouped by substrate")
        else:
            skip("bridge_refits", "no per-item r2 values found in bridge summary")
    else:
        skip("bridge_refits", "bridge_refit_summary.json pending (production run in flight)")

    _write_json_atomic(out_dir / "figures_manifest.json", manifest)
    print(
        f"[p7-figures] artifact digest: figures={len(manifest['figures'])} "
        f"skipped={len(manifest['skipped'])} manifest={out_dir / 'figures_manifest.json'}",
        flush=True,
    )


# --------------------------------------------------------------------- upload (opt-in step)
def step_upload_nulls(args: argparse.Namespace) -> None:
    """One batched upload_folder commit of the P7 null matrices (persist-by-default)."""
    nulls_dir = Path(args.work_dir) / "analysis_tensors" / "nulls"
    files = sorted(nulls_dir.glob("*.npy"))
    if not files:
        print(f"[p7-upload] no null matrices under {nulls_dir}; nothing to upload", flush=True)
        return
    from huggingface_hub import HfApi

    path_in_repo = f"{args.hf_prefix}/p7/analysis_tensors/nulls"
    api = HfApi()
    # HUB_DIR_FILECOUNT_EXEMPT: issue-1092 driver; dirs bounded well under 10k files
    api.upload_folder(
        folder_path=str(nulls_dir),
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=f"issue 1092 P7: projection-stage null matrices ({len(files)} files)",
        allow_patterns=["*.npy"],
    )
    listed = {
        e.path.rsplit("/", 1)[1]
        # HUB_VERIFY_RETRY_EXEMPT: issue-1092 driver; scoped listing with orchestration-layer retry
        for e in api.list_repo_tree(
            HF_DATA_REPO, repo_type="dataset", path_in_repo=path_in_repo, recursive=False
        )
        if getattr(e, "size", None) is not None
    }
    missing = {f.name for f in files} - listed
    if missing:
        raise RuntimeError(
            f"upload verify: {len(missing)} null files missing on Hub: {sorted(missing)[:5]}"
        )
    print(
        f"[p7-upload] uploaded + verified {len(files)} null matrices to {path_in_repo}", flush=True
    )


# --------------------------------------------------------------------------- entrypoint
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #1092 P7: merged read aggregates, 28-layer projection sweeps, figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--steps",
        default="merge,projection-sweep,figures",
        help="csv of steps: merge,projection-sweep,figures,upload-nulls",
    )
    p.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1092/p7"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures/issue_1092"))
    p.add_argument("--work-dir", type=Path, default=Path("data/issue_1092/p7"))
    p.add_argument("--corpus-dir", type=Path, default=Path("data/issue_1092/p0/corpus"))
    p.add_argument("--judge-scores", type=Path, default=Path("data/issue_1092/p5/scores.jsonl"))
    p.add_argument("--judge-summary", type=Path, default=Path("data/issue_1092/p5/summary.json"))
    p.add_argument(
        "--bridge-summary",
        type=Path,
        default=Path("data/issue_1092/p7_bridge/bridge_refit_summary.json"),
    )
    p.add_argument("--hf-prefix", default=HF_PREFIX_DEFAULT)
    p.add_argument("--hf-revision", default="main")
    p.add_argument("--boxes", default=None, help="csv of P6 box numbers (default 1..12)")
    p.add_argument(
        "--cells",
        default=None,
        help=f"csv of judged cells for the projection sweep (default {','.join(JUDGED_CELLS)})",
    )
    p.add_argument("--layers", default="0-27", help="read-out layers for the sweep")
    p.add_argument("--n-null-draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rb-rev", default=DEFAULT_RB_REV)
    p.add_argument(
        "--rb-dir", type=Path, default=None, help="local r_B dir/.npy override (offline smoke)"
    )
    p.add_argument(
        "--fixture-root",
        type=Path,
        default=None,
        help="local tree mimicking the Hub layout (offline tests)",
    )
    p.add_argument(
        "--expect-full-grid",
        action="store_true",
        help="fail unless the merged grid covers the full plan-v6 slot set",
    )
    p.add_argument("--force", action="store_true", help="recompute despite fingerprint match")
    return p.parse_args()


def main() -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    args = parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    known = {"merge", "projection-sweep", "figures", "upload-nulls"}
    unknown = set(steps) - known
    if unknown:
        raise SystemExit(f"unknown steps {sorted(unknown)}; known: {sorted(known)}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    hub = _hub_io(args)
    if "merge" in steps:
        step_merge(args, hub)
    if "projection-sweep" in steps:
        step_projection_sweep(args, hub)
    if "figures" in steps:
        step_figures(args)
    if "upload-nulls" in steps:
        step_upload_nulls(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
