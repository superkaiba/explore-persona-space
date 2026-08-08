"""#1768 horse-race bootstrap CIs (interp-critique round-1 Must-Fix 1).

Evaluates the plan §6 registered success criterion "Horse-race CI half-widths
<= 0.1 (median across arms)", which the p9 phase never computed. Per (arm,
layer), a stratified paired bootstrap over the ~20 panel question rows
(even-qidx and odd-qidx halves resampled independently WITH replacement,
preserving the disjoint-halves baseline registration of
``issue1768_directions.panel_write_legs``):

per draw  w*    = mean(vp[draw])  - mean(v0[draw_even])      (on-policy write)
          w_tf* = mean(vtf[draw]) - mean(v0[draw_even])      (matched-text write)
          delta* = tbar - mean(v0[draw_odd])                  (tbar fixed; its
                    n_mix_rows >> 20 noise is not resampled — the registered
                    "bootstrap over Delta-v rows" resamples the panel write rows)
          r_B / W_U row: FIXED candidates.

B=2000 draws; rng per unit [FLOOR_SEED, crc32(arm_id), layer, 0xB007] (a 4th
token so the stream never replays p9's null draws). Point estimates are
recomputed and asserted to match ``direction_reads.json`` to 1e-8 (loader
parity check). Panel stores stream from the pinned HF revision with
delete-after-use (peak staging < 1 GB despite ~10.7 GB transferred).

Output: eval_results/issue_1768/horse_race_cis.json — per-cell CI table +
per-(candidate, tree) half-width medians at the primary layer + the §6
criterion verdict.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1768_directions as D  # noqa: E402

B_DRAWS = 2000
BOOT_TOKEN = 0xB007  # rng stream separator vs p9's per-unit seeds
REV = "c07267285d2cdbf3e0401ddc3e3accae50e496a7"  # the body's pinned data-repo revision
LAYERS = (14, 19, 25)


def _stage(stage_root: Path, rel: str) -> Path:
    """hf_hub_download one file from the pinned revision into stage_root."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    local = stage_root / rel
    if local.exists():
        return local
    hub.retry_transient(
        lambda: hf_hub_download(
            X.HF_DATA_REPO,
            rel,
            repo_type="dataset",
            revision=REV,
            local_dir=str(stage_root),
        ),
        what=f"panel store fetch {rel}",
    )
    assert local.exists(), rel
    return local


def _boot_means(rows: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(B, d) bootstrap means of ``rows`` (n, d) under draw index matrix (B, n_draw)."""
    return rows[idx].mean(axis=1)


def _row_cos(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (A @ b) / (np.linalg.norm(A, axis=1) * np.linalg.norm(b) + 1e-300)


def _pair_cos(A: np.ndarray, Bm: np.ndarray) -> np.ndarray:
    num = np.einsum("ij,ij->i", A, Bm)
    return num / (np.linalg.norm(A, axis=1) * np.linalg.norm(Bm, axis=1) + 1e-300)


def _ci(v: np.ndarray) -> dict:
    lo, hi = float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
    return {"ci_lo": lo, "ci_hi": hi, "half_width": (hi - lo) / 2.0}


def summarize_primary_layer(cells: dict) -> dict:
    """Build the plan §6 criterion summary from the per-cell CI table.

    Marker arms race the marker unembedding row under BOTH the ``r_B`` and
    ``W_U_marker_row`` labels (the fleet two-way-race convention), so the
    registered three-slot pool counts that race twice; the deduplicated reads
    drop the duplicated ``r_B`` entry per marker cell (interp-critique round-2
    Must-Fix). ``criterion_met`` keys on the deduplicated pooled median.
    Returns the summary dict (caller appends ``bootstrap`` / ``revision``).
    """

    def prim_layer(arm_id: str) -> int:
        return 25 if arm_id.startswith("mk") else 19

    hw_all: list[float] = []
    med: dict[str, dict[str, float]] = {}
    for cname in ("delta", "r_B", "W_U_marker_row"):
        med[cname] = {}
        for tree in ("on_policy", "matched_text"):
            vals = [
                c[cname][tree]["half_width"]
                for c in cells.values()
                if cname in c and c["layer"] == prim_layer(c["arm_id"])
            ]
            med[cname][tree] = float(np.median(vals))
            hw_all += vals
    hw_dedup: list[float] = []
    per_arm: dict[str, list[float]] = {}
    for c in cells.values():
        if c["layer"] != prim_layer(c["arm_id"]):
            continue
        cands = [k for k in ("delta", "r_B", "W_U_marker_row") if k in c]
        if "W_U_marker_row" in cands and c["r_B"] == c["W_U_marker_row"]:
            cands.remove("r_B")  # the marker r_B slot IS the W_U row: one race, two labels
        hws = [c[k][t]["half_width"] for k in cands for t in ("on_policy", "matched_text")]
        hw_dedup += hws
        per_arm.setdefault(c["arm_id"], []).extend(hws)
    pooled = float(np.median(hw_all))
    pooled_dedup = float(np.median(hw_dedup))
    per_arm_dedup = float(np.median([float(np.median(v)) for v in per_arm.values()]))
    return {
        "criterion": "plan §6: horse-race CI half-widths <= 0.1 (median across arms)",
        "primary_layer_median_half_width": {
            "pooled_all_raced_cosines": pooled,
            "pooled_deduplicated": pooled_dedup,
            "per_arm_median_deduplicated": per_arm_dedup,
            **med,
        },
        "n_raced_cosines_primary_layer": len(hw_all),
        "n_raced_cosines_deduplicated": len(hw_dedup),
        "criterion_met": bool(pooled_dedup <= 0.1),
        "criterion_met_note": (
            "keyed on the deduplicated pooled median (the marker r_B == W_U_marker_row "
            f"race counted once); the registered three-slot pool reads {pooled:.3f} by "
            "counting the marker unembedding race under both labels"
        ),
    }


def main() -> None:
    stage_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/issue_1768/hf_dl")
    results_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("eval_results/issue_1768")
    out_root = stage_root / "issue1768_mapshift"
    arms = X.all_arms()
    reads = json.loads((results_dir / "direction_reads.json").read_text())["reads"]
    rb = D.load_rb_tensors(out_root)
    wu = D.load_wu_row(X.BASE_MODEL)

    # small persistent inputs: 4 base panel stores + 56 tbar files
    for beh in ("cas", "imp", "syc", "mk"):
        _stage(stage_root, f"issue1768_mapshift/panel_capture/base_{beh}/pooled.pt")
    for da in sorted({X.delta_arm_for(a) for a in arms}):
        _stage(stage_root, f"issue1768_mapshift/delta_tf/{da}/tbar.pt")

    base_cache: dict[str, dict] = {}
    cells: dict[str, dict] = {}
    t0 = time.time()
    for k, arm in enumerate(arms, 1):
        rel_p = f"issue1768_mapshift/panel_capture/{arm.arm_id}/pooled.pt"
        rel_t = f"issue1768_mapshift/panel_capture_tf/{arm.arm_id}/pooled.pt"
        p_path, t_path = _stage(stage_root, rel_p), _stage(stage_root, rel_t)
        if arm.beh_key not in base_cache:
            base_cache[arm.beh_key] = D._load_store(
                out_root / "panel_capture" / f"base_{arm.beh_key}" / "pooled.pt"
            )
        base_store = base_cache[arm.beh_key]
        arm_store = D._load_store(p_path)
        tf_store = D._load_store(t_path)
        src_ctx = D.source_context_id(arm, base_store)
        tb = D._load_store(out_root / "delta_tf" / X.delta_arm_for(arm) / "tbar.pt")
        for layer in LAYERS:
            key = f"{arm.arm_id}_L{layer}"
            v0 = D._panel_rows(base_store, src_ctx, layer)
            vp = D._panel_rows(arm_store, src_ctx, layer)
            vtf = D._panel_rows(tf_store, src_ctx, layer)
            qs = sorted(v0)
            ev = [q for q in qs if q % 2 == 0]
            od = [q for q in qs if q % 2 == 1]
            V0e = np.stack([v0[q] for q in ev])
            V0o = np.stack([v0[q] for q in od])
            VP = np.stack([np.stack([m[q] for q in ev + od]) for m in (vp, vtf)])
            tbar = np.asarray(tb["tbar"][layer].float().numpy(), dtype=np.float64)

            # point-estimate parity vs p9 (loader check)
            w_pt = VP[0].mean(axis=0) - V0e.mean(axis=0)
            wtf_pt = VP[1].mean(axis=0) - V0e.mean(axis=0)
            delta_pt = tbar - V0o.mean(axis=0)
            races = reads[key]["races"]
            assert abs(D._cos(w_pt, delta_pt) - races["delta"]["cos_w"]) < 1e-8, key
            assert abs(D._cos(wtf_pt, delta_pt) - races["delta"]["cos_w_tf"]) < 1e-8, key

            rng = np.random.default_rng(
                [X.FLOOR_SEED, zlib.crc32(arm.arm_id.encode("utf-8")), layer, BOOT_TOKEN]
            )
            ie = rng.integers(0, len(ev), size=(B_DRAWS, len(ev)))
            io = rng.integers(0, len(od), size=(B_DRAWS, len(od)))
            v0A = _boot_means(V0e, ie)  # (B, d)
            v0B = _boot_means(V0o, io)
            # draw mean over the 20 resampled rows = weighted mean of half means
            ne, no = len(ev), len(od)
            wa, wb = ne / (ne + no), no / (ne + no)
            vp_mean = wa * _boot_means(VP[0][:ne], ie) + wb * _boot_means(VP[0][ne:], io)
            vtf_mean = wa * _boot_means(VP[1][:ne], ie) + wb * _boot_means(VP[1][ne:], io)
            Wb = vp_mean - v0A
            Wtfb = vtf_mean - v0A
            Db = tbar[None, :] - v0B

            cands: dict[str, np.ndarray | None] = {"delta": None, "r_B": rb[arm.beh_key][layer]}
            if arm.kind == "marker":
                cands["W_U_marker_row"] = wu
            cell: dict[str, dict] = {"arm_id": arm.arm_id, "layer": layer, "n_questions": len(qs)}
            for cname, cand in cands.items():
                if cname == "delta":
                    on, tf = _pair_cos(Wb, Db), _pair_cos(Wtfb, Db)
                    pts = (D._cos(w_pt, delta_pt), D._cos(wtf_pt, delta_pt))
                else:
                    on, tf = _row_cos(Wb, cand), _row_cos(Wtfb, cand)
                    pts = (D._cos(w_pt, cand), D._cos(wtf_pt, cand))
                    assert abs(pts[0] - races[cname]["cos_w"]) < 1e-8, (key, cname)
                cell[cname] = {
                    "on_policy": {"point": pts[0], **_ci(on)},
                    "matched_text": {"point": pts[1], **_ci(tf)},
                }
            cells[key] = cell
        os.remove(p_path)
        os.remove(t_path)
        print(f"[hrci] {k}/{len(arms)} {arm.arm_id} done ({time.time() - t0:.0f}s)", flush=True)

    summary = summarize_primary_layer(cells)
    summary["bootstrap"] = (
        f"stratified paired over panel question rows, B={B_DRAWS}, "
        "even/odd halves resampled independently; candidates fixed (tbar not resampled)"
    )
    summary["revision"] = REV
    D._atomic_json(
        results_dir / "horse_race_cis.json",
        {"summary": summary, "cells": cells, **D._meta()},
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
