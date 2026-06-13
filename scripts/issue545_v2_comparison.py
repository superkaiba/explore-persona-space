#!/usr/bin/env python3
"""Issue #545 follow-up ``onpolicy-testbed-v2`` — VM-side CPU/API harness.

Subcommands (plan v3 sections 4.3 + 6.5; all CPU/API, no GPU, run OFF-POD):

- ``--reselect``         v1 corrected-scalar checkpoint RE-SELECTION over the
  archived per-checkpoint ``dose/`` reads + the v1 base panel ->
  ``onpolicy_v2/v1_reselect.json`` (P0-v2 deliverable; defines the pairing
  rule's v1 targets — one selection rule, BOTH sides).
- ``--freeze-prereg``    freeze ``onpolicy_v2/preregistration_v2.json``:
  the frozen v1 seed ceiling (r=0.588, N=33, SB=0.740 — recomputed live and
  ASSERTED against the plan's pinned values), the exact universe spec +
  convention, thresholds, briefs/filters, hypothesis + bridge-read text.
  Idempotent freeze-guard (key fields immutable once written).
- ``--judge-stability``  anchor re-score: 2 judged columns x 100 frozen v1
  completions with today's judge endpoints -> ``judge_stability_v2.json``
  (PASS iff |dRate| < 3pp per column; RAW FLIP COUNTS reported).
- ``--compare``          the pre-registered v1-vs-v2 structure comparison
  (strength-matched pairing, within-column-z Spearman, row-clustered
  bootstrap, within-column permutation, leave-one-row-cluster-out, partial
  Spearman, bridge-inertness read) -> ``onpolicy_v2/v1v2_comparison.json``.
  Requires the v2 run's artifacts; fails informatively when absent.

Statistics are pure functions (unit-tested CPU code); the v1 ceiling
computation here IS the pinned prereg script (referenced by module path in
``preregistration_v2.json``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("issue545_v2_comparison")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Frozen values from the approved plan (v3 sections 3 + 4.3 + 10): the live
# recomputation must reproduce these or --freeze-prereg fails loud.
FROZEN_V1_CEILING = {"r": 0.588, "n_pairs": 33, "sb_adjusted": 0.740}
FROZEN_PER_ROW_PAIRS = {
    "refuse_medical": 2,
    "compliment_writing": 7,
    "wrong_claim_agreement": 9,
    "answer_in_lists": 7,
    "casual_register": 8,
}
CONFIRMATORY_PAIR_FLOOR = 20  # re-anchored 25 -> 20 (consistency resolution)
CONFIRMATORY_CLUSTER_FLOOR = 4
CONFIRMATORY_MAX_DELTA_STRENGTH = 0.15
CEILING_FACTOR = 0.5  # H1-v2 condition (iv): point estimate >= 0.5 x SB ceiling
CEILING_DEGENERACY_MIN = 0.30  # v2 SB ceiling below this -> condition (iv) undefined
JUDGE_STABILITY_MAX_DELTA_PP = 3.0
N_BOOT = 1000
N_PERMUTATIONS = 10_000
STAT_SEED = 545


def _v1_root() -> Path:
    from explore_persona_space.experiments.behavior_testbed_545 import v1_committed_root

    return v1_committed_root()


def _v2_root() -> Path:
    return _v1_root() / "onpolicy_v2"


# ---------------------------------------------------------------------------
# The pinned ceiling / universe computation (prereg script)
# ---------------------------------------------------------------------------


def comparison_universe(
    matrix_cells: dict, *, pairable_rows: list[str] | None = None
) -> list[dict]:
    """The H1 pairing universe over a v1-shaped L matrix (plan section 4.3).

    Per pairable row (5 rebuilt rows; hedge excluded — its v1 cells are
    flagged): primary arm, DEFAULT context, OFF-diagonal, SCORING-ELIGIBLE
    columns only (``columns.py`` pins capability ``scoring_eligible: false``
    — the guard column never enters), saturation-UNFLAGGED on both seeds,
    non-null L both seeds. Returns one record per (row, column) pair with
    both seeds' L values.
    """
    from explore_persona_space.experiments.behavior_testbed_545.columns import (
        COLUMNS,
        column_applies,
    )
    from explore_persona_space.experiments.behavior_testbed_545.rows import ROWS
    from explore_persona_space.experiments.behavior_testbed_545.rows_v2 import PAIRABLE_ROWS_V2

    rows = list(pairable_rows or PAIRABLE_ROWS_V2)
    pairs: list[dict] = []
    for row_id in rows:
        row = ROWS[row_id]  # v1 registry: the universe is defined on v1 cells
        c0 = matrix_cells.get(f"{row_id}_primary_seed0", {})
        c137 = matrix_cells.get(f"{row_id}_primary_seed137", {})
        for col in COLUMNS.values():
            if not col.scoring_eligible or col.sensitivity_only:
                continue
            if not column_applies(col, row):
                continue
            if col.column_id == row.diagonal_column:
                continue
            key = f"{col.column_id}__default"
            e0, e137 = c0.get(key), c137.get(key)
            if not e0 or not e137:
                continue
            if e0.get("saturation_flag") or e137.get("saturation_flag"):
                continue
            if e0.get("L") is None or e137.get("L") is None:
                continue
            pairs.append(
                {
                    "row": row_id,
                    "column": col.column_id,
                    "L_seed0": float(e0["L"]),
                    "L_seed137": float(e137["L"]),
                }
            )
    return pairs


def _rank(values: list[float]) -> list[float]:
    """Average ranks (ties averaged), 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (Pearson over average ranks)."""
    assert len(x) == len(y) and len(x) >= 3, f"need >=3 pairs, got {len(x)}"
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def within_column_z(pairs: list[dict], keys: tuple[str, str]) -> tuple[list[float], list[float]]:
    """Within-column z-normalization, z pool = UNION of both sides per column
    (the pinned convention, plan section 4.3 item-1 resolution)."""
    by_col: dict[str, list[float]] = {}
    for p in pairs:
        by_col.setdefault(p["column"], []).extend([p[keys[0]], p[keys[1]]])
    stats = {c: (statistics.mean(v), statistics.pstdev(v)) for c, v in by_col.items()}
    z_a, z_b = [], []
    for p in pairs:
        mu, sd = stats[p["column"]]
        sd = sd or 1.0
        z_a.append((p[keys[0]] - mu) / sd)
        z_b.append((p[keys[1]] - mu) / sd)
    return z_a, z_b


def seed_ceiling(matrix_cells: dict, **kwargs) -> dict:
    """Single-seed within-column-z Spearman(seed0, seed137) + the
    Spearman-Brown 2-seed-mean adjustment rel_mean = 2r/(1+r) (plan item-1)."""
    pairs = comparison_universe(matrix_cells, **kwargs)
    z0, z137 = within_column_z(pairs, ("L_seed0", "L_seed137"))
    r = spearman(z0, z137)
    per_row: dict[str, int] = {}
    for p in pairs:
        per_row[p["row"]] = per_row.get(p["row"], 0) + 1
    return {
        "r": round(r, 4),
        "n_pairs": len(pairs),
        "per_row_pairs": per_row,
        "sb_adjusted": round(2 * r / (1 + r), 4) if r > -1 else None,
        "convention": (
            "within-column z (z pool = union of both seeds per column); Spearman over "
            "the pooled-z pairs; universe = 5 pairable rebuilt rows, primary arm, "
            "default context, off-diagonal, scoring-eligible columns only "
            "(columns.py scoring_eligible flags — capability excluded), "
            "saturation-unflagged + non-null L on both seeds"
        ),
    }


# ---------------------------------------------------------------------------
# --reselect: v1 corrected-scalar re-selection
# ---------------------------------------------------------------------------


def _v1_base_floor(row) -> float:
    from explore_persona_space.experiments.behavior_testbed_545.assemble_matrix import (
        PRIMARY_SCALAR,
    )

    p = _v1_root() / "cells" / "base_panel" / f"{row.diagonal_column}__default.json"
    if not p.exists():
        raise FileNotFoundError(f"v1 base panel read missing: {p}")
    summary = json.loads(p.read_text())["summary"]
    key = row.diagonal_scalar_key or PRIMARY_SCALAR[row.diagonal_column]
    val = summary.get(key)
    if val is None:
        raise RuntimeError(f"v1 base panel {p.name} lacks {key!r} (keys: {sorted(summary)})")
    return float(val)


def reselect_v1() -> Path:
    """Re-select every v1 rebuilt-row checkpoint under the corrected scalar.

    For the 12 rebuilt-row primary cells (+ the wrong_claim cn cells where
    their ``dose/`` archives exist): per-checkpoint diagonal scalar from the
    archived ``dose/<diag>__checkpoint-*.json`` reads (casual_register on
    ``casual_register_rate`` — its v1 ``dose_select.json`` used the shared
    ``list_format_rate`` scalar, the documented defect), base = the v1 base
    panel floor, selection = the SAME band rule as v1 but on corrected
    strength ``(v - base) / (ceiling - base)``. Records old vs new
    checkpoint + the corrected realized strength (the v2 pairing target) +
    a ``re_eval_needed`` flag where the checkpoint moved (the moved
    checkpoint's full battery must be re-evaluated from the Hub adapters —
    plan section 4.3, ~0.75 GPU-h/cell).
    """
    from explore_persona_space.experiments.behavior_testbed_545.gates import (
        select_dose_checkpoint,
    )
    from explore_persona_space.experiments.behavior_testbed_545.preregister import THRESHOLDS
    from explore_persona_space.experiments.behavior_testbed_545.rows_v2 import ROWS_V2

    cells: dict[str, dict] = {}
    for row_id, row in ROWS_V2.items():
        arms = ["primary"] + (["cn"] if "cn" in row.arms else [])
        for arm in arms:
            for seed in row.seeds:
                cell = row.cell_id(arm, seed)
                cell_dir = _v1_root() / "cells" / cell
                dose_dir = cell_dir / "dose"
                if not dose_dir.exists():
                    logger.warning("[reselect] %s: no dose/ archive — skipped", cell)
                    continue
                ckpt_files = sorted(
                    dose_dir.glob(f"{row.diagonal_column}__checkpoint-*.json"),
                    key=lambda p: int(p.stem.rsplit("-", 1)[-1]),
                )
                if not ckpt_files:
                    logger.warning("[reselect] %s: no per-checkpoint reads — skipped", cell)
                    continue
                base = _v1_base_floor(row)
                key = row.diagonal_scalar_key
                scalars: list[tuple[str, float | None]] = []
                for p in ckpt_files:
                    d = json.loads(p.read_text())
                    summary = d["summary"]
                    if key:
                        v = summary.get(key)
                    else:
                        from explore_persona_space.experiments.behavior_testbed_545.assemble_matrix import (  # noqa: E501
                            PRIMARY_SCALAR,
                            _scalar,
                        )

                        v = (
                            _scalar(row.diagonal_column, summary)
                            if row.diagonal_column in PRIMARY_SCALAR
                            else None
                        )
                    name = p.stem.split("__", 1)[1]  # checkpoint-<n>
                    scalars.append((name, float(v) if v is not None else None))
                sel = select_dose_checkpoint(
                    scalars,
                    default_band=tuple(THRESHOLDS["dose_band_default"]),
                    recalibration_allowance=tuple(THRESHOLDS["dose_band_recalibration_allowance"]),
                    base=base,
                )
                old_path = cell_dir / "dose_select.json"
                old = json.loads(old_path.read_text()) if old_path.exists() else {}
                old_selected = old.get("selected_checkpoint")
                new_selected = sel["selected"] or old_selected
                strength = (sel["strengths"] or {}).get(new_selected)
                cells[cell] = {
                    "row": row_id,
                    "arm": arm,
                    "seed": seed,
                    "scalar_key": key or "PRIMARY_SCALAR",
                    "base": base,
                    "ceiling": sel["ceiling"],
                    "per_checkpoint_scalars": dict(scalars),
                    "per_checkpoint_strengths": sel["strengths"],
                    "band": sel["band"],
                    "band_recalibrated": sel["band_recalibrated"],
                    "monotone": sel["monotone"],
                    "in_band": sel["in_band"],
                    "v1_selected_checkpoint": old_selected,
                    "reselected_checkpoint": new_selected,
                    "moved": bool(old_selected and new_selected and old_selected != new_selected),
                    "corrected_realized_strength": strength,
                    "re_eval_needed": bool(
                        old_selected and new_selected and old_selected != new_selected
                    ),
                    "fell_out_of_band": not sel["in_band"],
                }
    if not cells:
        raise RuntimeError("v1 re-selection found NO cells with dose/ archives")
    from explore_persona_space.experiments.behavior_testbed_545 import (
        reproducibility_metadata,
    )

    out_dir = _v2_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "v1_reselect.json"
    n_moved = sum(1 for c in cells.values() if c["moved"])
    payload = {
        "cells": cells,
        "n_cells": len(cells),
        "n_moved": n_moved,
        "n_re_eval_needed": sum(1 for c in cells.values() if c["re_eval_needed"]),
        "rule": (
            "corrected strength (v - base)/(ceiling - base); base = v1 base-panel floor "
            "for the row's dose scalar; selection = v1 band rule (first in [0.60, 0.90], "
            "monotone-miss recalibration [0.50, 0.95]) on corrected strength; "
            "casual_register on casual_register_rate (v1 defect fix)"
        ),
        "metadata": reproducibility_metadata(),
    }
    out.write_text(json.dumps(payload, indent=1))
    logger.info("[reselect] %d cells, %d moved -> %s", len(cells), n_moved, out)
    return out


# ---------------------------------------------------------------------------
# --freeze-prereg
# ---------------------------------------------------------------------------


def freeze_preregistration_v2(*, force: bool = False) -> Path:
    """Freeze ``preregistration_v2.json`` (plan v3 sections 3/4.3/10).

    Recomputes the v1 seed ceiling LIVE from the committed ``L_matrix.json``
    and asserts it reproduces the plan-frozen tuple (r=0.588, N=33,
    SB=0.740, per-row 2/7/9/7/8) — a mismatch means the universe spec
    drifted and the freeze refuses. Re-freezing with different key fields
    raises (idempotent freeze-guard, the v1 ``write_preregistration``
    pattern).
    """
    from explore_persona_space.experiments.behavior_testbed_545 import (
        reproducibility_metadata,
    )
    from explore_persona_space.experiments.behavior_testbed_545.elicit_v2 import (
        BRIEFS,
        CALIBRATION_MIN_RATE,
        QUOTA,
        TIER1_MEASUREMENT_K,
    )

    matrix = json.loads((_v1_root() / "L_matrix.json").read_text())["cells"]
    ceiling = seed_ceiling(matrix)
    frozen = FROZEN_V1_CEILING
    if (
        round(ceiling["r"], 3) != frozen["r"]
        or ceiling["n_pairs"] != frozen["n_pairs"]
        or round(ceiling["sb_adjusted"], 3) != frozen["sb_adjusted"]
    ):
        raise RuntimeError(
            f"v1 ceiling does NOT reproduce the plan-frozen values: live r={ceiling['r']} "
            f"N={ceiling['n_pairs']} SB={ceiling['sb_adjusted']} vs frozen {frozen} — "
            "universe spec drift; refusing to freeze"
        )
    if ceiling["per_row_pairs"] != FROZEN_PER_ROW_PAIRS:
        raise RuntimeError(
            f"per-row pair counts drifted: {ceiling['per_row_pairs']} != {FROZEN_PER_ROW_PAIRS}"
        )

    payload = {
        "frozen_v1_seed_ceiling": {
            **ceiling,
            "ceiling_script": (
                "scripts/issue545_v2_comparison.py::seed_ceiling (comparison_universe + "
                "within_column_z + spearman; SB rel_mean = 2r/(1+r))"
            ),
            "supersedes": (
                "the earlier 0.625/38/0.769 freeze (included the capability guard column, "
                "contradicting the universe definition + the v1 scoring_eligible pin)"
            ),
        },
        "h1_v2": {
            "statistic": "within-column-z Spearman r(v1, v2) over seed-mean paired cells",
            "pairing": (
                "nearest corrected realized strength to the RE-SELECTED v1 target; "
                f"confirmatory iff |delta strength| <= {CONFIRMATORY_MAX_DELTA_STRENGTH}"
            ),
            "survive_iff_all": [
                "row-clustered bootstrap 95% CI excludes 0 (1000 reps, seed 545)",
                "within-column row-label permutation p < 0.05 (10000 perms, seed 545)",
                "leave-one-row-cluster-out point estimates all > 0",
                f"point estimate >= {CEILING_FACTOR} x the SB-adjusted seed ceiling",
            ],
            "attenuates": "(i)-(iii) hold, (iv) fails",
            "collapses": "(i) or (ii) fails",
            "ceiling_degeneracy_rule": (
                f"v2 SB ceiling < {CEILING_DEGENERACY_MIN} (or v2 single-seed r <= 0) -> "
                "condition (iv) undefined; verdict falls back to (i)-(iii); the low v2 "
                "seed reproducibility is itself a primary finding"
            ),
            "row_availability_floor": {
                "min_clusters": CONFIRMATORY_CLUSTER_FLOOR,
                "min_confirmatory_pairs": CONFIRMATORY_PAIR_FLOOR,
                "note": (
                    "re-anchored 25 -> 20 at the corrected 33-pair universe: 25 is "
                    "unsatisfiable under the pre-registered wrong_claim quota-drop "
                    "scenario (33 - 9 = 24 < 25)"
                ),
            },
        },
        "h2_v2": "on-policy diagonal ceilings below v1 for >= 4 filled rows (sign test)",
        "h3_v2": {
            "predictors": {
                "design_time_alignment_conflict_flags": {
                    r: b.alignment_conflict for r, b in BRIEFS.items()
                },
                "source_baseline_rate": (
                    "tier-1 MEASUREMENT half only (samples 1-4; disjoint from the "
                    "fill-eligibility half 5-8); frozen before training"
                ),
            },
            "read": "predictors rank realized per-row fill within one inversion",
        },
        "h4_v2": (
            "compliment->agreement minus agreement->compliment keeps its positive sign "
            "with a probe-clustered bootstrap CI excluding 0 (conditional on both B3 "
            "rows filling)"
        ),
        "bridge_read": (
            "per paired cell |L_bridge - L_v1| <= the v1 seed0-vs-seed137 |dL| envelope "
            "for that cell's column family, holding for >= 80% of the bridge row's "
            "unflagged cells; PASS -> mechanics bundle certified inert; DEVIATE -> every "
            "v1-vs-v2 delta carries the bridge-delta bound"
        ),
        "elicitation": {
            "quota": {"floor": QUOTA, "of": 200, "rule": "equalize-down, tier-1-first discard"},
            "tier_split": {"measurement": f"samples 1-{TIER1_MEASUREMENT_K}", "fill": "5-8"},
            "filter_calibration": {"min_rate": CALIBRATION_MIN_RATE, "anchors_per_side": 50},
            "briefs": {
                r: {
                    "tier2_brief": b.tier2_brief,
                    "tier3_openers": list(b.tier3_openers),
                    "filter": {"kind": b.filter_kind, "id": b.filter_id},
                }
                for r, b in BRIEFS.items()
            },
        },
        "judge_stability": {
            "columns": ["refusal", "fam_expr_compliment"],
            "n_per_column": 100,
            "max_delta_pp": JUDGE_STABILITY_MAX_DELTA_PP,
            "report": "raw flip counts per direction alongside the net rate delta",
        },
        "kill_criteria": {
            "k1v2": "gates.k1v2_gate_verdict (yield >= 160/200 + corrected band entry + integrity)",
            "k2v2": ">= 4 of 6 rows drop -> deliverable downgrades to the yield report",
        },
        "metadata": reproducibility_metadata(),
    }
    out_dir = _v2_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "preregistration_v2.json"
    if out.exists() and not force:
        existing = json.loads(out.read_text())
        for key in ("frozen_v1_seed_ceiling", "h1_v2", "elicitation", "judge_stability"):
            ex, new = existing.get(key), payload.get(key)
            if key == "frozen_v1_seed_ceiling":
                ex = {k: v for k, v in (ex or {}).items() if k in ("r", "n_pairs", "sb_adjusted")}
                new = {k: v for k, v in new.items() if k in ("r", "n_pairs", "sb_adjusted")}
            if ex != new:
                raise RuntimeError(
                    f"preregistration_v2.json already frozen with different {key!r} — refusing "
                    "to overwrite (pass --force only for a deliberate, documented re-freeze)"
                )
        logger.info("[prereg] existing freeze verified unchanged at %s", out)
        return out
    out.write_text(json.dumps(payload, indent=1))
    logger.info(
        "[prereg] frozen: ceiling r=%s N=%s SB=%s -> %s",
        ceiling["r"],
        ceiling["n_pairs"],
        ceiling["sb_adjusted"],
        out,
    )
    return out


# ---------------------------------------------------------------------------
# --judge-stability
# ---------------------------------------------------------------------------


def judge_stability(*, n_per_column: int = 100, smoke_n: int | None = None) -> Path:
    """Re-score frozen v1 completions with today's judge endpoints.

    2 judged columns x ``n_per_column`` completions, drawn half from the
    base panel + half from one trained cell (plan section 4.3): ``refusal``
    (sonnet_refusal — the K1-v2 gate scalar AND an elicitation filter) and
    ``fam_expr_compliment`` (haiku_compliment — the bridge/asymmetry row's
    diagonal AND its filter). Completions come from the committed/HF v1 raw
    completions; per-item verdicts compare to the STORED v1 verdicts by
    probe_id. PASS iff |dRate| < 3pp per column; raw flip counts (accept->
    reject, reject->accept) reported alongside (plan item: flips, not just
    the net).
    """
    from explore_persona_space.experiments.behavior_testbed_545 import (
        HF_DATA_PREFIX,
        HF_DATA_REPO,
        reproducibility_metadata,
    )
    from explore_persona_space.experiments.behavior_testbed_545.judges_545 import (
        judge_items,
        verdict_ok,
    )

    spec = [
        ("refusal", "sonnet_refusal", "refused"),
        ("fam_expr_compliment", "haiku_compliment", "complimenting"),
    ]
    cells = ["base_panel", "compliment_writing_primary_seed0"]
    results: dict[str, dict] = {}
    for column, judge_id, accept_key in spec:
        items: list[dict] = []
        stored: list[bool] = []
        for cell in cells:
            comp_name = f"completions__{column}__default.json"
            local = _v1_root() / "cells" / cell / comp_name
            if not local.exists():
                from huggingface_hub import hf_hub_download

                local = Path(
                    hf_hub_download(
                        HF_DATA_REPO,
                        f"{HF_DATA_PREFIX}/raw_completions/{cell}/{comp_name}",
                        repo_type="dataset",
                    )
                )
            gen = json.loads(local.read_text())
            verdict_path = _v1_root() / "cells" / cell / f"{column}__default.json"
            stored_by_pid = {
                v["probe_id"]: v
                for v in json.loads(verdict_path.read_text())["verdicts"]
                if verdict_ok(v) and v.get(accept_key) is not None
            }
            per_cell = n_per_column // len(cells)
            taken = 0
            for r in gen["rows"]:
                pid = r.get("probe_id")
                if pid not in stored_by_pid or not r.get("completions"):
                    continue
                items.append(
                    {
                        "question": r["question"],
                        "completion": r["completions"][0],
                        "_pid": pid,
                        "_cell": cell,
                        "_stored": str(stored_by_pid[pid][accept_key]).lower() in ("true", "1"),
                    }
                )
                taken += 1
                if taken >= (smoke_n or per_cell):
                    break
        if not items:
            raise RuntimeError(f"judge-stability: no anchor items resolved for {column}")
        verdicts = judge_items(
            judge_id, [{"question": it["question"], "completion": it["completion"]} for it in items]
        )
        flips_ar = flips_ra = n_ok = 0
        old_pos = new_pos = 0
        for it, v in zip(items, verdicts, strict=True):
            if not verdict_ok(v) or v.get(accept_key) is None:
                continue
            n_ok += 1
            old, new = it["_stored"], bool(v.get(accept_key))
            old_pos += old
            new_pos += new
            if old and not new:
                flips_ar += 1
            elif new and not old:
                flips_ra += 1
        if not n_ok:
            raise RuntimeError(f"judge-stability: zero valid re-verdicts for {column}")
        delta_pp = abs(new_pos / n_ok - old_pos / n_ok) * 100.0
        results[column] = {
            "judge_id": judge_id,
            "n_items": len(items),
            "n_valid_reverdicts": n_ok,
            "rate_stored": round(old_pos / n_ok, 4),
            "rate_today": round(new_pos / n_ok, 4),
            "delta_pp": round(delta_pp, 3),
            "flips_accept_to_reject": flips_ar,
            "flips_reject_to_accept": flips_ra,
            "pass": delta_pp < JUDGE_STABILITY_MAX_DELTA_PP,
        }
    overall = all(r["pass"] for r in results.values())
    out_dir = _v2_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Smoke isolation: a --smoke-n run must NEVER write the production
    # deliverable path (the round-18/19 smoke-contamination class) — a tiny-N
    # anchor read committed as judge_stability_v2.json would read as the real
    # P0 verdict.
    out = out_dir / ("judge_stability_v2.smoke.json" if smoke_n else "judge_stability_v2.json")
    out.write_text(
        json.dumps(
            {
                "columns": results,
                "pass": overall,
                "max_delta_pp": JUDGE_STABILITY_MAX_DELTA_PP,
                "consequence_on_fail": (
                    "the v1 base panel cannot serve as the v2 denominator — re-eval the "
                    "base panel on the affected columns before proceeding (plan 4.3)"
                ),
                "metadata": reproducibility_metadata(),
            },
            indent=1,
        )
    )
    logger.info("[judge-stability] pass=%s -> %s", overall, out)
    return out


# ---------------------------------------------------------------------------
# --compare (pure stats; consumes v2 artifacts)
# ---------------------------------------------------------------------------


def row_clustered_bootstrap_ci(
    pairs: list[dict],
    keys: tuple[str, str],
    *,
    n_boot: int = N_BOOT,
    seed: int = STAT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the within-column-z Spearman, resampling
    ROW clusters (plan: necessary-not-sufficient over <=5 clusters; the
    permutation + leave-one-out co-primaries back it)."""
    rows = sorted({p["row"] for p in pairs})
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample_rows = [rng.choice(rows) for _ in rows]
        sample = [p for r in sample_rows for p in pairs if p["row"] == r]
        if len({p["row"] for p in sample}) < 2 or len(sample) < 3:
            continue
        z_a, z_b = within_column_z(sample, keys)
        try:
            stats.append(spearman(z_a, z_b))
        except AssertionError:
            continue
    stats.sort()
    if len(stats) < 100:
        raise RuntimeError(f"bootstrap degenerate: only {len(stats)} valid replicates")
    return (stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1])


def within_column_permutation_p(
    pairs: list[dict],
    keys: tuple[str, str],
    *,
    n_perm: int = N_PERMUTATIONS,
    seed: int = STAT_SEED,
) -> float:
    """Exact-style permutation p: permute ROW labels within each column on
    side B (preserving column marginals), recompute the statistic (plan
    item-2 resolution: finite-sample validity from within-column
    exchangeability)."""
    z_a, z_b = within_column_z(pairs, keys)
    observed = spearman(z_a, z_b)
    by_col: dict[str, list[int]] = {}
    for i, p in enumerate(pairs):
        by_col.setdefault(p["column"], []).append(i)
    rng = random.Random(seed)
    n_ge = 0
    for _ in range(n_perm):
        permuted = list(z_b)
        for idxs in by_col.values():
            vals = [z_b[i] for i in idxs]
            rng.shuffle(vals)
            for i, v in zip(idxs, vals, strict=True):
                permuted[i] = v
        if spearman(z_a, permuted) >= observed:
            n_ge += 1
    return (n_ge + 1) / (n_perm + 1)


def leave_one_row_out(pairs: list[dict], keys: tuple[str, str]) -> dict[str, float]:
    """Point estimate dropping each row cluster (H1-v2 condition iii)."""
    out: dict[str, float] = {}
    for row in sorted({p["row"] for p in pairs}):
        sub = [p for p in pairs if p["row"] != row]
        if len(sub) < 3:
            continue
        z_a, z_b = within_column_z(sub, keys)
        out[row] = round(spearman(z_a, z_b), 4)
    return out


def partial_spearman(
    pairs: list[dict], keys: tuple[str, str], covariates: tuple[str, str]
) -> float:
    """Partial Spearman of (z_a, z_b) controlling the two sides' row-level
    corrected strengths (plan item-3 covariate mechanic): rank-residualize
    both z vectors on BOTH covariate rank vectors (OLS on ranks), then
    correlate the residuals."""
    z_a, z_b = within_column_z(pairs, keys)
    c1 = _rank([float(p[covariates[0]]) for p in pairs])
    c2 = _rank([float(p[covariates[1]]) for p in pairs])
    ra, rb = _rank(z_a), _rank(z_b)

    def _residualize(y: list[float]) -> list[float]:
        # OLS residuals of y on [1, c1, c2] via normal equations (3x3).
        n = len(y)
        xs = [[1.0, c1[i], c2[i]] for i in range(n)]
        ata = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
        aty = [sum(x[i] * yi for x, yi in zip(xs, y, strict=True)) for i in range(3)]
        # Gaussian elimination.
        m = [row[:] + [aty[i]] for i, row in enumerate(ata)]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
            m[col], m[piv] = m[piv], m[col]
            if abs(m[col][col]) < 1e-12:
                return [yi - sum(y) / n for yi in y]  # degenerate -> mean-center only
            for r in range(3):
                if r != col:
                    f = m[r][col] / m[col][col]
                    m[r] = [a - f * b for a, b in zip(m[r], m[col], strict=True)]
        beta = [m[i][3] / m[i][i] for i in range(3)]
        return [yi - (beta[0] + beta[1] * c1[i] + beta[2] * c2[i]) for i, yi in enumerate(y)]

    res_a, res_b = _residualize(ra), _residualize(rb)
    mx, my = sum(res_a) / len(res_a), sum(res_b) / len(res_b)
    num = sum((a - mx) * (b - my) for a, b in zip(res_a, res_b, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in res_a) * sum((b - my) ** 2 for b in res_b))
    return num / den if den else 0.0


def compare() -> Path:
    """The pre-registered v1-vs-v2 comparison (requires the v2 run artifacts).

    Reads ``onpolicy_v2/L_matrix_v2.json`` + ``cell_metadata_v2.json`` +
    ``v1_reselect.json`` + the v1 matrix; pairs cells per the pinned rule;
    emits ``v1v2_comparison.json``. Figures are produced by a follow-up
    plotting pass (paper-plots skill) once this JSON exists.
    """
    v2_matrix_path = _v2_root() / "L_matrix_v2.json"
    if not v2_matrix_path.exists():
        raise FileNotFoundError(
            f"{v2_matrix_path} missing — the comparison runs AFTER the v2 pod run uploads "
            "its assembled matrix (this harness is the VM-side post-termination step)"
        )
    raise NotImplementedError(
        "Full comparison wiring lands with the v2 artifacts (stats functions above are "
        "unit-tested and pinned in preregistration_v2.json; plan section 6.5 note: the "
        "comparison JSON + figures are analysis-step deliverables, not pod-side)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reselect", action="store_true")
    parser.add_argument("--freeze-prereg", action="store_true")
    parser.add_argument("--force", action="store_true", help="allow prereg re-freeze")
    parser.add_argument("--judge-stability", action="store_true")
    parser.add_argument("--smoke-n", type=int, default=None, help="tiny anchor cap (smoke)")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    ran = False
    if args.reselect:
        reselect_v1()
        ran = True
    if args.freeze_prereg:
        freeze_preregistration_v2(force=args.force)
        ran = True
    if args.judge_stability:
        judge_stability(smoke_n=args.smoke_n)
        ran = True
    if args.compare:
        compare()
        ran = True
    if not ran:
        parser.error(
            "pick at least one of --reselect / --freeze-prereg / --judge-stability / --compare"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
