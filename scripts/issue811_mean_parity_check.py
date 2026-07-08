#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (v0, →, ×, ≈) in scientific docstrings + logs.
"""Issue #811 — re-extracted base-leg reproduces #667 parity check (plan §13 check 1).

The #811 position-aware reader is a CODE CHANGE to the teacher-force extractor, so
the mean-vs-turn_nl comparison must NOT be confounded by a reader bug or by drift
between #667's extraction environment and this run's. This is a FAIL-LOUD guard run
BEFORE the ~7 GPU-h Phase-1 spend (a drift is a CODE bug → ``failure_class: code``,
round-2 CONCERN mean-parity-check-not-wired).

## Why the original single-gate v0-identity check was wrong (round-6 redesign)

The original check demanded the freshly-extracted base mean ``v0`` reproduce #667's
STORED ``v0`` to cosine ≥ 0.999. That is UNACHIEVABLE BY CONSTRUCTION:

- ``v0`` is the mean over the model's own GREEDY response ``R``. #667's extractor
  generated its OWN greedy ``R`` at its extraction time and NEVER persisted the
  ``R`` text (its npz stores only vectors — no raw completions on HF). Greedy
  regeneration on different hardware (L4 now vs #667's GPU) + a different vLLM
  build flips some tokens, which perturbs the response span and hence its mean.
- Measured on the real att-20260701-233116 partial store (em, L14, source
  ``binst_em``, 30 matched-target cells): new ``v0(t)`` cosine to ref ``v0(t)``
  spans **0.9970 to 0.9997** (12/24 >= 0.999 across an earlier 24-cell probe;
  0/30 below 0.98 here) — a per-cell-VARIABLE band, the R-token-flip signature,
  NOT a systematic reader change. The original gate crashed at cosine 0.997028
  (floor 0.999) — a token-flip cell, not a bug.

The **R-FREE reader read IS exact**, and that is the read the position-aware
refactor actually touched: the new store's per-target context vector ``c_C(t)``
vs #667's stored per-target ``c_Cp(t)`` cosine = **0.9999** (min 0.99987 over the
30 real cells) → the position-aware reader is FAITHFUL; environment numerics ≈1e-4.

## #667 store key semantics (verified on HF, do NOT confuse)

In #667's npz (``issue667_gate_chain_preview/analysis_tensors/{behavior}/
{source}_seed42/{target}_L{layer}.npz``):
  - ``c_C``  = the SOURCE-constant context vector (cross-target cosine EXACTLY
    1.0000 — same for every target under a source).
  - ``c_Cp`` = the per-TARGET context vector (cross-target cosine ≈0.94).
  - ``v0``   = the per-target base mean-over-response answer (cross-target ≈0.965).

#811's phase-0 store writes the per-TARGET context under its OWN ``c_C`` key
(``phase0_base_leg/{behavior}/{source}_seed42/{target}_L{layer}.npz`` keys
``c_C`` / ``v0`` / ``v0_turn_nl``). So the correct like-for-like faithfulness
comparison is **new ``c_C(t)`` ↔ ref ``c_Cp(t)``** (NOT ref ``c_C``, which is a
different quantity: the source-constant vector).

## What this check does now — three like-for-like checks per sampled cell

(a) **HARD reader-faithfulness gate (R-FREE):** new ``c_C(t)`` vs #667 ``c_Cp(t)``,
    same (behavior, source, target, layer). This isolates exactly what the #811
    position-aware refactor changed (the reader), independent of R resampling.
    Tolerances: cosine floor 0.999, relative-L2 ceiling 0.05 (measured min 0.99987
    / max rel_l2 tiny). A violation means the reader logic changed the context read
    (wrong span / wrong layer) — a CODE bug.
(b) **v0 identity/confusion check (HARD):** cos(new ``v0(t)``, ref ``v0(t)``) must be
    the ARGMAX over all ref targets present in the same source dir. Catches
    misalignment / wrong-cell / wrong-layer bugs (the R band cannot swap which
    target a cell is closest to; measured 30/30 argmax-match, ref cross-target
    baseline ≈0.965 ≪ any matched cell).
(c) **v0 gross-drift floor (HARD, R-resampling-aware):** cos(new ``v0(t)``, ref
    ``v0(t)``) ≥ 0.98 AND rel_l2 ≤ 0.25. The measured resample band bottoms at
    0.9970; a wrong-span/wrong-layer reader bug lands near the cross-target baseline
    (≈0.96) or below. This is the R-resampling-aware replacement for the original
    0.999 v0-identity floor.

Every per-cell cosine + rel_l2 (all three checks) is LOGGED at INFO either way. The
CLI shape is unchanged (default behavior=em, layer=14, n-cells=3); the module still
raises ``RuntimeError`` (fail-loud → exit 1) so the dispatcher HALTs on a REAL
failure. The old ``--cos-floor`` / ``--rel-l2-ceil`` flags now tune the HARD
reader-faithfulness gate (a); the v0 checks use their own R-resampling-aware bounds.

## Science caveat (carried into the clean-result)

#811's ``mean`` base leg is a FRESH-GREEDY-R replication of #722's substrate. The
INTERNAL same-R mean-vs-turn_nl comparison is unconfounded (both summaries read
from the SAME re-generated R this run). Any vs-#722 comparison of the mean leg
carries a resampled-R scope note: #722's mean was computed over #722's own greedy
R, which #811 does not reuse verbatim (that R text was never persisted).

Usage (a few cells, run inline in the dispatcher AFTER the first Phase-0 cells land
and BEFORE upload):
    uv run python scripts/issue811_mean_parity_check.py \
        --phase0-root eval_results/issue_811/phase0_base_leg \
        --behavior em --layer 14 --n-cells 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# uv run python does NOT auto-load .env; the #667-store hf_hub_download needs
# HF_TOKEN. Project wrapper (analysis-phase script; shell exports also cover
# pod/GCE/SLURM).

logger = logging.getLogger("issue811.parity")

# #667's store (the reference the re-extracted base leg must reproduce).
REF_REPO = "superkaiba1/explore-persona-space-data"
REF_PREFIX = "issue667_gate_chain_preview/analysis_tensors"

# (a) HARD reader-faithfulness gate: new c_C(t) vs #667 c_Cp(t). The R-FREE read the
# position-aware refactor touched; achievable to ~1e-4 (measured min cosine 0.99987).
# bf16-vs-fp32 rounding on a 3584-d vector keeps direction (cosine) >0.999 and
# aggregate relative-L2 well under 0.05 for a faithful re-extraction; a reader-logic
# drift (wrong span / wrong layer) blows both far past these.
COS_FLOOR = 0.999
REL_L2_CEIL = 0.05

# (c) v0 gross-drift floor: R-resampling-aware. The re-generated greedy R flips some
# tokens vs #667's (never-persisted) R, so exact v0 identity is impossible; the
# measured matched-target resample band bottoms at cosine 0.9970. A wrong-span /
# wrong-layer / wrong-cell reader bug lands near the ref cross-target v0 baseline
# (≈0.96) or below, so a 0.98 floor / 0.25 rel_l2 ceiling separates "R token flips"
# from a real bug with margin.
V0_COS_FLOOR = 0.98
V0_REL_L2_CEIL = 0.25


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / nb)


def _load_ref(behavior: str, source_cid: str, target_cid: str, layer: int) -> dict:
    """#667's stored reference for one cell.

    Returns ``{"v0": ..., "c_Cp": ...}`` (both float64). ``v0`` is the per-target
    base mean answer; ``c_Cp`` is the per-target context vector — the like-for-like
    counterpart of #811's per-target ``c_C``. Raises ``KeyError`` if either key is
    absent (a wrong / stale ref store).
    """
    from huggingface_hub import hf_hub_download

    rel = f"{REF_PREFIX}/{behavior}/{source_cid}_seed42/{target_cid}_L{layer}.npz"
    path = hf_hub_download(REF_REPO, rel, repo_type="dataset")
    d = np.load(path, allow_pickle=True)
    for key in ("v0", "c_Cp"):
        if key not in d.files:
            raise KeyError(f"#667 ref {rel} has no '{key}' key; keys={sorted(d.files)}")
    return {
        "v0": np.asarray(d["v0"], dtype=np.float64),
        "c_Cp": np.asarray(d["c_Cp"], dtype=np.float64),
    }


def _list_ref_v0_for_source(
    behavior: str, source_cid: str, layer: int, target_cids: list[str]
) -> dict[str, np.ndarray]:
    """#667's stored ``v0`` for every target in ``target_cids`` (the confusion set).

    Used by check (b): the new cell's ``v0`` must argmax-match its OWN target among
    all these ref targets. Bounded — ``target_cids`` is the sampled cells' targets
    (n-cells ≤ 3), so at most a handful of ``hf_hub_download`` calls.
    """
    return {t: _load_ref(behavior, source_cid, t, layer)["v0"] for t in target_cids}


def _phase0_source_targets(
    phase0_root: Path, behavior: str, source_cid: str, layer: int
) -> list[str]:
    """All target cids present in the phase-0 store dir for one source at ``layer``.

    The confusion set for check (b): the new cell's ``v0`` must argmax to its OWN
    target among every target the source dir carries. Reads the ``target_cid`` from
    inside each npz (robust to filename drift). Returns a sorted, de-duplicated list.
    """
    src_dir = phase0_root / behavior / f"{source_cid}_seed42"
    tgts: list[str] = []
    for npz in sorted(src_dir.glob(f"*_L{layer}.npz")):
        d = np.load(npz, allow_pickle=True)
        if "target_cid" in d.files:
            t = str(np.asarray(d["target_cid"]).item())
        else:
            t = npz.name.rsplit(f"_L{layer}.npz", 1)[0]
        if t not in tgts:
            tgts.append(t)
    return tgts


def _iter_phase0_cells(phase0_root: Path, behavior: str, layer: int, n_cells: int):
    """Yield up to ``n_cells`` (source_cid, target_cid, c_C, v0) from the Phase-0 store.

    Walks ``{phase0_root}/{behavior}/{source}_seed42/{target}_L{layer}.npz`` in
    sorted order (deterministic cell selection). Re-reads source/target from
    inside each file (robust to filename drift). Yields the per-target base context
    vector (``c_C``) + the base mean answer (``v0``). Raises ``KeyError`` if either
    key is absent (a wrong-shape / mean-only store).
    """
    beh_dir = phase0_root / behavior
    if not beh_dir.is_dir():
        raise FileNotFoundError(
            f"phase0 store dir missing: {beh_dir} — run Phase-0 extraction first"
        )
    yielded = 0
    for src_dir in sorted(p for p in beh_dir.iterdir() if p.is_dir()):
        for npz in sorted(src_dir.glob(f"*_L{layer}.npz")):
            d = np.load(npz, allow_pickle=True)
            for key in ("c_C", "v0"):
                if key not in d.files:
                    raise KeyError(f"phase0 {npz} has no '{key}' key; keys={sorted(d.files)}")
            src = str(np.asarray(d["source_cid"]).item())
            tgt = str(np.asarray(d["target_cid"]).item())
            yield (
                src,
                tgt,
                np.asarray(d["c_C"], dtype=np.float64),
                np.asarray(d["v0"], dtype=np.float64),
            )
            yielded += 1
            if yielded >= n_cells:
                return
    if yielded == 0:
        raise FileNotFoundError(
            f"no phase0 cells found under {beh_dir} for L{layer} — run Phase-0 first"
        )


def check_mean_parity(
    phase0_root: Path,
    behavior: str,
    layer: int,
    n_cells: int,
    *,
    cos_floor: float = COS_FLOOR,
    rel_l2_ceil: float = REL_L2_CEIL,
    v0_cos_floor: float = V0_COS_FLOOR,
    v0_rel_l2_ceil: float = V0_REL_L2_CEIL,
) -> list[dict]:
    """Assert the re-extracted base leg reproduces #667 on ``n_cells`` cells.

    Runs THREE like-for-like checks per cell (all HARD, fail-loud on the first
    violation — a real drift means the position-aware reader changed a read, which
    invalidates the parent reference; a CODE bug, plan §13 / §4.2):

    (a) reader-faithfulness (R-FREE): new ``c_C(t)`` vs #667 ``c_Cp(t)``
        (cos ≥ ``cos_floor``, rel_l2 ≤ ``rel_l2_ceil``).
    (b) v0 identity/confusion: cos(new ``v0(t)``, ref ``v0(t)``) is the argmax over
        all ref targets present in the same source dir (no misalignment).
    (c) v0 gross-drift floor (R-resampling-aware): cos(new ``v0(t)``, ref ``v0(t)``)
        ≥ ``v0_cos_floor`` AND rel_l2 ≤ ``v0_rel_l2_ceil``.

    Returns the per-cell comparison records.
    """
    recs: list[dict] = []
    # First pass: collect the sampled cells (so we can build the per-source
    # ref-v0 confusion set for check (b) with a bounded number of downloads).
    cells = list(_iter_phase0_cells(phase0_root, behavior, layer, n_cells))
    # For each sampled source, build the argmax confusion set (check b) from ALL
    # target npz PRESENT in that source's phase-0 dir (not just the sampled targets)
    # — the new v0 must be closest to its OWN target among the full target set (the
    # ref cross-target v0 baseline is ~0.965, far below any matched cell). Bounded:
    # #cells<=3 sources x ~30 targets hf_hub_download calls.
    ref_v0_by_source: dict[str, dict[str, np.ndarray]] = {
        src: _list_ref_v0_for_source(
            behavior, src, layer, _phase0_source_targets(phase0_root, behavior, src, layer)
        )
        for src in {c[0] for c in cells}
    }

    for src, tgt, c_c_new, v0_new in cells:
        ref = _load_ref(behavior, src, tgt, layer)
        c_cp_ref = ref["c_Cp"]
        v0_ref = ref["v0"]
        assert c_c_new.shape == c_cp_ref.shape, (
            f"c_C shape mismatch {src}->{tgt} L{layer}: new {c_c_new.shape} "
            f"vs ref c_Cp {c_cp_ref.shape}"
        )
        assert v0_new.shape == v0_ref.shape, (
            f"v0 shape mismatch {src}->{tgt} L{layer}: new {v0_new.shape} vs ref {v0_ref.shape}"
        )

        # (a) HARD reader-faithfulness gate (R-free).
        cc_cos = _cos(c_c_new, c_cp_ref)
        cc_rel = _rel_l2(c_c_new, c_cp_ref)
        cc_ok = cc_cos >= cos_floor and cc_rel <= rel_l2_ceil

        # (b) v0 identity/confusion: argmax over the sampled ref targets for this
        # source. Only decidable when there is >1 target in the confusion set.
        ref_v0_set = ref_v0_by_source[src]
        best_tgt = max(ref_v0_set, key=lambda t: _cos(v0_new, ref_v0_set[t]))
        argmax_ok = (best_tgt == tgt) or (len(ref_v0_set) < 2)

        # (c) v0 gross-drift floor (R-resampling-aware).
        v0_cos = _cos(v0_new, v0_ref)
        v0_rel = _rel_l2(v0_new, v0_ref)
        v0_ok = v0_cos >= v0_cos_floor and v0_rel <= v0_rel_l2_ceil

        rec = {
            "source_cid": src,
            "target_cid": tgt,
            "layer": layer,
            "cc_cosine": cc_cos,
            "cc_rel_l2": cc_rel,
            "cc_ok": cc_ok,
            "v0_cosine": v0_cos,
            "v0_rel_l2": v0_rel,
            "v0_ok": v0_ok,
            "v0_argmax_target": best_tgt,
            "v0_argmax_ok": argmax_ok,
            "ok": cc_ok and argmax_ok and v0_ok,
        }
        recs.append(rec)
        logger.info(
            "[parity] %s->%s L%d: (a) c_C<->c_Cp cos=%.6f rel_l2=%.6f %s | "
            "(b) v0 argmax=%s %s | (c) v0 cos=%.6f rel_l2=%.6f %s",
            src,
            tgt,
            layer,
            cc_cos,
            cc_rel,
            "OK" if cc_ok else "DRIFT",
            best_tgt,
            "OK" if argmax_ok else "MISALIGNED",
            v0_cos,
            v0_rel,
            "OK" if v0_ok else "DRIFT",
        )
        if not cc_ok:
            raise RuntimeError(
                f"READER-FAITHFULNESS DRIFT {src}->{tgt} L{layer}: c_C(t) vs #667 "
                f"c_Cp(t) cosine={cc_cos:.6f} (floor {cos_floor}) rel_l2={cc_rel:.6f} "
                f"(ceil {rel_l2_ceil}) — the position-aware reader changed the R-FREE "
                f"context read (wrong span / wrong layer). This is the read the #811 "
                f"refactor touched and it MUST be exact (plan §13, failure_class: code)."
            )
        if not argmax_ok:
            raise RuntimeError(
                f"V0 MISALIGNMENT {src}->{tgt} L{layer}: new v0 argmax-matches ref "
                f"target '{best_tgt}', not its own target '{tgt}' — a wrong-cell / "
                f"wrong-layer / cell-misalignment bug, NOT R resampling (plan §13, "
                f"failure_class: code)."
            )
        if not v0_ok:
            raise RuntimeError(
                f"V0 GROSS DRIFT {src}->{tgt} L{layer}: cos(new v0, ref v0)={v0_cos:.6f} "
                f"(floor {v0_cos_floor}) rel_l2={v0_rel:.6f} (ceil {v0_rel_l2_ceil}) — "
                f"below the R-resampling band (bottoms ~0.997) and near/below the ref "
                f"cross-target v0 baseline (~0.96); a wrong-span/wrong-layer reader bug, "
                f"NOT R token flips (plan §13, failure_class: code)."
            )
    logger.info(
        "[parity] PASS: %d cells reproduce #667 (reader-faithfulness exact; "
        "v0 argmax + gross-drift within the R-resampling band)",
        len(recs),
    )
    return recs


def compare_committed_mean_cells(
    run_cells_dir: Path,
    committed_cells_dir: Path,
    out_json: Path,
    summaries: tuple[str, ...] = ("mean",),
) -> dict:
    """maxp-round §6(b): this round's re-extracted MEAN fit cells vs the committed v1 cells.

    REPORT-ONLY replication-stability read (never fatal — plan §6: a flipped
    Δ/floor CALL is REPORTED as a finding, not silently averaged; the 3-way
    adjudication uses THIS round's internally consistent store). For every
    ``{behavior}_L{li}_mean.json`` under ``run_cells_dir``, loads the
    SAME-filename committed cell under ``committed_cells_dir`` and compares
    ``Delta_med`` / ``floor_combined`` / the above-vs-below-floor CALL
    (``Delta_med / floor_combined > 1``). Expected agreement scale: the
    resampled-R replication band (v1 measured matched-target cosines
    0.997-0.9997; greedy R is deterministic per environment, not across
    GPU/vLLM builds). Writes the row-level report to ``out_json`` and returns
    it. Raises ONLY when ``run_cells_dir`` holds no mean cells at all (nothing
    to compare = a wiring bug, not a replication finding).
    """
    import json as _json
    import time as _time

    # pre-user round (plan §6 three-way parity): the SAME report shape over any
    # committed summary set — mean+turn_nl vs the v1 cells, mean+turn_nl+maxp vs
    # the v2 maxp-round cells. Default ("mean",) preserves the maxp round verbatim.
    run_files = sorted(p for s in summaries for p in run_cells_dir.glob(f"*_{s}.json"))
    if not run_files:
        raise FileNotFoundError(
            f"no *_{{{','.join(summaries)}}}.json under {run_cells_dir} — "
            "run the round's fit phase first"
        )
    rows: list[dict] = []
    n_flips = 0
    for rp in run_files:
        run_cell = _json.loads(rp.read_text())
        row: dict = {"cell": rp.name}
        cp = committed_cells_dir / rp.name
        if not cp.exists():
            row["status"] = "no_committed_reference"
            rows.append(row)
            continue
        ref_cell = _json.loads(cp.read_text())
        for tag, cell in (("run", run_cell), ("committed", ref_cell)):
            dm = float(cell["Delta_med"])
            fl = float(cell["floor_combined"])
            row[tag] = {
                "Delta_med": dm,
                "floor_combined": fl,
                "ratio": (dm / fl) if fl > 0 else None,
                "call_above_floor": bool(fl > 0 and dm / fl > 1.0),
            }
        flip = row["run"]["call_above_floor"] != row["committed"]["call_above_floor"]
        row["call_flipped"] = flip
        row["status"] = "call_flipped" if flip else "call_stable"
        if flip:
            n_flips += 1
            logger.warning(
                "[parity-committed] %s: Δ/floor CALL FLIPPED vs the committed v1 run "
                "(run ratio=%.3g, committed ratio=%.3g) — REPORTED as a "
                "replication-stability finding (plan §6b), NOT fatal.",
                rp.name,
                row["run"]["ratio"] if row["run"]["ratio"] is not None else float("nan"),
                (
                    row["committed"]["ratio"]
                    if row["committed"]["ratio"] is not None
                    else float("nan")
                ),
            )
        else:
            logger.info(
                "[parity-committed] %s: call stable (run ratio=%s, committed ratio=%s)",
                rp.name,
                f"{row['run']['ratio']:.3g}" if row["run"]["ratio"] is not None else "n/a",
                (
                    f"{row['committed']['ratio']:.3g}"
                    if row["committed"]["ratio"] is not None
                    else "n/a"
                ),
            )
        rows.append(row)
    report = {
        "meta": {
            "issue": 811,
            "followup": "maxp-winner-mapchange mean-call replication vs committed v1 cells",
            "run_cells_dir": str(run_cells_dir),
            "committed_cells_dir": str(committed_cells_dir),
            "generated_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        },
        "n_cells": len(rows),
        "n_call_flips": n_flips,
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(_json.dumps(report, indent=2, default=float))
    logger.info(
        "[parity-committed] %d mean cells compared, %d call flips -> %s",
        len(rows),
        n_flips,
        out_json,
    )
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Issue #811 base-leg parity check vs #667 (plan §13)")
    ap.add_argument(
        "--phase0-root",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_811/phase0_base_leg",
        help="the Phase-0 base-leg store root ({behavior}/{source}_seed42/{target}_L{layer}.npz)",
    )
    ap.add_argument("--behavior", default="em", help="behavior to parity-check (default: em)")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n-cells", type=int, default=3, help="number of cells to compare (2-3)")
    ap.add_argument(
        "--cos-floor",
        type=float,
        default=COS_FLOOR,
        help="(a) reader-faithfulness cosine floor (c_C vs #667 c_Cp)",
    )
    ap.add_argument(
        "--rel-l2-ceil",
        type=float,
        default=REL_L2_CEIL,
        help="(a) reader-faithfulness relative-L2 ceiling (c_C vs #667 c_Cp)",
    )
    ap.add_argument(
        "--v0-cos-floor",
        type=float,
        default=V0_COS_FLOOR,
        help="(c) v0 gross-drift cosine floor (R-resampling-aware)",
    )
    ap.add_argument(
        "--v0-rel-l2-ceil",
        type=float,
        default=V0_REL_L2_CEIL,
        help="(c) v0 gross-drift relative-L2 ceiling (R-resampling-aware)",
    )
    ap.add_argument(
        "--compare-committed",
        action="store_true",
        help="maxp-round §6(b) MODE: compare this round's fitted *_mean.json cells "
        "vs the committed v1 cells (report-only; a flipped Δ/floor call is a "
        "replication-stability finding, never fatal). Runs AFTER the fit phase; "
        "the phase-0 activation checks above do not run in this mode.",
    )
    ap.add_argument(
        "--run-cells-dir",
        type=Path,
        default=None,
        help="(--compare-committed) this round's cells dir, e.g. "
        "eval_results/issue_811/maxp-winner-mapchange/cells",
    )
    ap.add_argument(
        "--committed-cells-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_811/cells",
        help="(--compare-committed) the completed v1 run's committed cells dir",
    )
    ap.add_argument(
        "--committed-out",
        type=Path,
        default=None,
        help="(--compare-committed) report JSON path (default: "
        "<run-cells-dir>/../mean_call_replication_vs_v1.json)",
    )
    ap.add_argument(
        "--committed-summaries",
        nargs="+",
        default=["mean"],
        help="(--compare-committed) which summaries' cells to compare against the "
        "committed dir (pre-user round: 'mean turn_nl' vs v1, "
        "'mean turn_nl maxp' vs the v2 maxp cells). Default preserves the "
        "maxp-round mean-only read verbatim.",
    )
    args = ap.parse_args()
    if args.compare_committed:
        assert args.run_cells_dir is not None, "--compare-committed requires --run-cells-dir"
        out_json = args.committed_out or (
            args.run_cells_dir.parent / "mean_call_replication_vs_v1.json"
        )
        compare_committed_mean_cells(
            args.run_cells_dir,
            args.committed_cells_dir,
            out_json,
            summaries=tuple(args.committed_summaries),
        )
        return 0
    check_mean_parity(
        args.phase0_root,
        args.behavior,
        args.layer,
        args.n_cells,
        cos_floor=args.cos_floor,
        rel_l2_ceil=args.rel_l2_ceil,
        v0_cos_floor=args.v0_cos_floor,
        v0_rel_l2_ceil=args.v0_rel_l2_ceil,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
