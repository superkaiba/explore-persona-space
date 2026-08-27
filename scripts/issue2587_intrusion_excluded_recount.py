"""CJK-intrusion-excluded recount of the 9B observed-separation battery reads (#2587).

Free-analysis follow-up (Step 9a-ter): recompute the per-axis observed
separation (mean ||obs delta|| / mean split-half noise norm — the exact
`stat_separation` statistic of scripts/issue2587_analysis.py) on the 9B side
with CJK-intruded draws EXCLUDED from the per-context 10-draw means, and
re-derive the 9B-minus-7B separation deltas on the 11 shared parent axes
against the UNCHANGED committed 7B side (point + per-draw bootstrap values
from the committed run). The two pilot axes are recounted on the
pilot-placement statistic (all primary pairs). Exclusion touches ONLY the
observed-shift means and the split-half noise — no ridge refit, no fire
recount, and the pair SELECTIONS are frozen to the committed run's
(reconstructed and asserted against the committed fire blocks).

Intrusion semantics (committed intrusion_scan_2587.json): a row is intruded
iff its rollout text matches the project CJK regex; the answer_language
axis's instructed-Chinese value is COMPLIANCE, not intrusion — those rows are
KEPT (per-cell semantics carried from the scan note). Per-row flags are
recomputed from the local anchor rollout shards via the SAME loader the
committed intrusion_judge_join.json used, then asserted equal to the
committed scan's per-cell + per-value counts (821 of 10,800 total).

Reuse (no new inner loops): load_stores_9b / build_pair_arrays /
build_axis_views / split_half_stats / pair_fired_mask /
carrier_multiplicities / boot_weighted_mean are imported from
scripts/issue2587_analysis.py; the carrier bootstrap regenerates the
committed run's EXACT multiplicities (seed [2215], B=10,000) and the
unexcluded leg is verified against the committed per-pair, per-axis, and
per-draw artifacts before any excluded number is reported.

Usage (VM, from the issue worktree; thread caps per code-style.md):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2587_intrusion_excluded_recount.py

Writes eval_results/issue_2587/intrusion_excluded_recount.json plus per-axis
checkpoints eval_results/issue_2587/checkpoints/intrusion_recount_<axis>.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF token BEFORE numpy/torch import (code-style.md)

import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2587_analysis as AN  # noqa: E402
import issue2587_intrusion_judge_join as JOIN  # noqa: E402

RESULTS = AN._REPO_ROOT / "eval_results" / "issue_2587"
SCHEMA = "issue2587_intrusion_excluded_recount_v1"
COMPLIANT_CELL = "answer_language"
COMPLIANT_VALUE = "chinese"  # instructed-CJK value: intrusion there is compliance


def _read_json(path: Path) -> dict:
    assert path.exists(), f"missing input: {path}"
    return json.loads(path.read_text())


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return float(np.nanmax(d)) if d.size else 0.0


def build_row_bank(
    cfg: AN.CfgX, spec: AN.SideSpec, st: AN.Stores, lstar: int
) -> tuple[np.ndarray, np.ndarray]:
    """Second pass over the per-cell va2587 stores: float64 primary-layer tail
    rows in (n_ctx, k_max, d) slot layout + the replicated validity mask.

    Mirrors load_stores_9b's row semantics exactly (rows key, think-leak +
    empty + n_comp>0 filters); the returned mask is ASSERTED equal to the
    module-built st.draw_valid, pinning the replication against drift."""
    n_ctx = len(st.ctx_ids)
    k_max = st.draw_valid.shape[1]
    bank = np.full((n_ctx, k_max, spec.d), np.nan, dtype=np.float64)
    valid = np.zeros((n_ctx, k_max), dtype=bool)
    for cell in st.cells:
        rel = f"analysis_tensors/va2587/{cell}.pt"
        p = AN.resolve_rel(cfg, cfg.in_root_9b, cfg.prefix_2587, rel)
        store = torch.load(p, map_location="cpu", weights_only=False)
        col = AN._store_col(store, lstar)
        idx_rows = store["rows"]
        tail = store["va_tail_incl"][:, col, :].to(torch.float64).numpy()
        n_rows = len(idx_rows)
        assert tail.shape == (n_rows, spec.d), tail.shape
        ctx_idx = np.array([st.row_of[r["context_id"]] for r in idx_rows], dtype=np.int64)
        n_comp = np.array([int(r["n_completion_tokens"]) for r in idx_rows], dtype=np.int64)
        draw = np.array([int(r["draw"]) for r in idx_rows], dtype=np.int64)
        leak = np.array([bool(r["think_leak"]) for r in idx_rows], dtype=bool)
        empty_mask = np.zeros(n_rows, dtype=bool)
        empty_ids = np.array(sorted(int(i) for i in store.get("empty_rows", [])), dtype=np.int64)
        if empty_ids.size:
            empty_mask[empty_ids] = True
        ok = (n_comp > 0) & ~empty_mask & ~leak
        assert not valid[ctx_idx[ok], draw[ok]].any(), f"duplicate (ctx, draw) slot in {cell}"
        bank[ctx_idx[ok], draw[ok]] = tail[ok]
        valid[ctx_idx[ok], draw[ok]] = True
    assert (valid == st.draw_valid).all(), "row-bank validity mask drifted from load_stores_9b"
    return bank, valid


def mean_tail_primary(bank: np.ndarray, dv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-context float64 mean of the primary-layer tail rows over the draws
    marked valid in ``dv``; contexts with ZERO surviving draws yield NaN rows
    (counted by the caller, never silently defaulted). Returns (mean, count)."""
    w = dv.astype(np.float64)
    cnt = w.sum(axis=1)
    sums = np.einsum("ck,ckd->cd", w, np.nan_to_num(bank, nan=0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / cnt[:, None]
    mean[cnt == 0] = np.nan
    return mean, cnt


def sep_point_and_draws(
    norm_obs: np.ndarray,
    noise: np.ndarray,
    sel: np.ndarray,
    pa: AN.PairArrays,
    mult: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Exact mirror of issue2587_analysis.crossmodel_contrasts.stat_separation
    (a closure there): point = nanmean flip / nanmean noise over the selection;
    draws = carrier-clustered weighted-mean ratio per bootstrap draw."""
    flip = float(np.nanmean(norm_obs[sel])) if sel.size else float("nan")
    nz = float(np.nanmean(noise[sel])) if sel.size else float("nan")
    pt = flip / nz if nz and np.isfinite(nz) and nz > 0 else float("nan")
    fd = AN.boot_weighted_mean(norm_obs[sel], pa.ca[sel], pa.cb[sel], pa.dyad[sel], mult)
    nd = AN.boot_weighted_mean(noise[sel], pa.ca[sel], pa.cb[sel], pa.dyad[sel], mult)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pt, np.where(nd > 0, fd / nd, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--out-json", type=Path, default=RESULTS / "intrusion_excluded_recount.json")
    ap.add_argument("--ckpt-dir", type=Path, default=RESULTS / "checkpoints")
    args = ap.parse_args()
    t0 = time.time()

    # Analysis config: the committed run's defaults (local staged inputs).
    # --ref7b-parent-commit is passed only to skip the git resolution of an
    # input this recount never reads (the parent minpair_delta.json).
    cfg = AN.build_config(AN.parse_args(["--ref7b-parent-commit", "unused-by-recount"]))
    assert not cfg.smoke and cfg.b_boot == AN.B_BOOT_DEFAULT and cfg.n_splits == 20, (
        cfg.smoke,
        cfg.b_boot,
        cfg.n_splits,
    )

    # ── committed artifacts (verification targets + frozen selections) ──
    cm = _read_json(RESULTS / "crossmodel_contrasts.json")
    sep_table = cm["stats"]["obs_separation_snr"]
    committed = {r["axis"]: r for r in sep_table["axes"]}
    scan = _read_json(RESULTS / "intrusion_scan_2587.json")
    battery9 = _read_json(RESULTS / "checkpoints" / "battery_qwen35_9b.json")
    npz_path = RESULTS / "crossmodel_perdraw" / "obs_separation_snr.npz"
    npz = np.load(npz_path, allow_pickle=False)
    parent_axes = [str(a) for a in npz["axes"]]
    assert sorted(parent_axes) == sorted(committed.keys()), (parent_axes, sorted(committed))
    draws7 = {a: npz["draws_7b"][k] for k, a in enumerate(parent_axes)}
    draws9_committed = {a: npz["draws_9b"][k] for k, a in enumerate(parent_axes)}
    delta_committed = {a: npz["delta_draws"][k] for k, a in enumerate(parent_axes)}

    lstar = AN.load_lstar(cfg.sweep_json)["lstar"]
    assert lstar == int(cm["layer_pair"]["qwen35_9b"]), (lstar, cm["layer_pair"])

    bank9 = _read_json(cfg.bank_9b)
    assert bank9["n_contexts"] == 1080 and bank9["n_pairs"] == 2874, (
        bank9["n_contexts"],
        bank9["n_pairs"],
    )
    instr_axes = tuple(sorted(set(parent_axes) - set(AN.QUERY_AXES)))
    spec9 = AN.make_spec_9b(lstar, instr_axes)

    # ── per-row intrusion flags (committed-scan parity asserted) ─────────
    lookup = JOIN.load_intrusion_lookup([AN._REPO_ROOT / d for d in JOIN.DEFAULT_ANCHOR_DIRS])
    contexts = bank9["contexts"]
    per_cell: dict[str, dict[str, int]] = {}
    by_value_al: dict[str, dict[str, int]] = {}
    for (cid, _draw), flag in lookup.items():
        ctx = contexts[cid]
        c = per_cell.setdefault(ctx["cell"], {"intruded": 0, "total": 0})
        c["total"] += 1
        c["intruded"] += bool(flag)
        if ctx["cell"] == COMPLIANT_CELL:
            v = by_value_al.setdefault(ctx["value_id"], {"intruded": 0, "total": 0})
            v["total"] += 1
            v["intruded"] += bool(flag)
    assert per_cell == scan["per_cell"], "per-cell intrusion recount != committed scan"
    assert by_value_al == scan["answer_language_by_value"], "answer_language by-value mismatch"
    n_intruded_total = sum(c["intruded"] for c in per_cell.values())
    assert n_intruded_total == scan["total"]["intruded"] == 821, n_intruded_total
    n_compliant_kept = by_value_al[COMPLIANT_VALUE]["intruded"]
    print(
        f"[recount] intrusion flags verified vs committed scan: {n_intruded_total}/10800 "
        f"intruded; {n_compliant_kept} instructed-Chinese rows kept as compliant",
        flush=True,
    )

    # ── stores + pair arrays + views + fire (module reuse, fail-fast) ────
    st9 = AN.load_stores_9b(cfg, bank9, spec9)
    assert len(st9.carriers) == 12, st9.carriers
    pa = AN.build_pair_arrays(bank9, st9, spec9, smoke=False)
    views = AN.build_axis_views(pa, spec9, len(st9.carriers))
    fire9 = AN.load_fire(cfg.manip_9b)
    fa70, fb70, _ma, _mb = AN.pair_fired_mask(pa, fire9, 70)
    fired9 = fa70 & fb70
    print(f"[recount] stores loaded: {len(st9.ctx_ids)} contexts, {pa.n} pairs", flush=True)

    # 7B per-pair fired flags from the committed per-pair dump (the 7B side is
    # untouched by the exclusion; its flags reconstruct the frozen selection).
    fired7_map: dict[str, bool] = {}
    with (RESULTS / "perpair_2587.jsonl").open() as fh:
        n7 = 0
        for line in fh:
            row = json.loads(line)
            if row["model_tag"] == "qwen25_7b":
                assert row["pair_id"] not in fired7_map, row["pair_id"]
                fired7_map[row["pair_id"]] = bool(row["pair_fired_70"])
                n7 += 1
    assert n7 == 2778, n7
    fired7 = np.array([fired7_map.get(pid, False) for pid in pa.ids], dtype=bool)

    # ── float64 primary-layer row bank + unexcluded parity gate ─────────
    bank64, _valid = build_row_bank(cfg, spec9, st9, lstar)
    mean_unexcl, cnt_unexcl = mean_tail_primary(bank64, st9.draw_valid)
    dev_mean = _max_abs(mean_unexcl, st9.va_tail_mean[lstar])
    assert dev_mean < 1e-8, f"row-bank mean deviates from load_stores_9b mean: {dev_mean}"

    # ── exclusion mask (answer_language instructed-Chinese rows kept) ────
    k_max = st9.draw_valid.shape[1]
    excl = np.zeros_like(st9.draw_valid)
    for (cid, draw), flag in lookup.items():
        if not flag:
            continue
        ctx = contexts[cid]
        if ctx["cell"] == COMPLIANT_CELL and ctx["value_id"] == COMPLIANT_VALUE:
            continue
        assert cid in st9.row_of, f"intruded context absent from stores: {cid}"
        assert 0 <= int(draw) < k_max, (cid, draw)
        excl[st9.row_of[cid], int(draw)] = True
    n_excl_applied = int(excl.sum())
    n_excl_valid = int((excl & st9.draw_valid).sum())
    assert n_excl_applied == n_intruded_total - n_compliant_kept, n_excl_applied
    dv_excl = st9.draw_valid & ~excl
    mean_excl, cnt_excl = mean_tail_primary(bank64, dv_excl)
    n_ctx_dead = int((cnt_excl == 0).sum())
    print(
        f"[recount] exclusion applied: {n_excl_applied} rows flagged, {n_excl_valid} removed "
        f"from valid draws; {n_ctx_dead} contexts lost ALL draws",
        flush=True,
    )

    # ── noise (split-half, module reuse; same seeds) + observed norms ────
    rel_unexcl = AN.split_half_stats(st9, pa, cfg.n_splits)
    st9x = dataclasses.replace(st9, draw_valid=dv_excl, n_valid=cnt_excl.astype(np.int64))
    rel_excl = AN.split_half_stats(st9x, pa, cfg.n_splits)
    norm_obs_unexcl = np.linalg.norm(
        st9.va_tail_mean[lstar][pa.a] - st9.va_tail_mean[lstar][pa.b], axis=1
    )
    with np.errstate(invalid="ignore"):
        norm_obs_excl = np.linalg.norm(mean_excl[pa.a] - mean_excl[pa.b], axis=1)

    # Per-pair parity vs the committed per-pair dump (unexcluded leg).
    obs_map: dict[str, float] = {}
    noise_map: dict[str, float] = {}
    with (RESULTS / "perpair_2587.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["model_tag"] == "qwen35_9b":
                obs_map[row["pair_id"]] = float(row["norm_obs_tail_primary"])
                noise_map[row["pair_id"]] = float(row["noise_norm"])
    obs_ref = np.array([obs_map[pid] for pid in pa.ids])
    noise_ref = np.array([noise_map[pid] for pid in pa.ids])
    dev_obs = _max_abs(norm_obs_unexcl, obs_ref)
    dev_noise = _max_abs(rel_unexcl["noise_norm"], noise_ref)
    assert dev_obs < 1e-6 and dev_noise < 1e-6, (dev_obs, dev_noise)
    print(
        f"[recount] unexcluded per-pair parity vs committed perpair dump: "
        f"max|obs|={dev_obs:.3e} max|noise|={dev_noise:.3e}",
        flush=True,
    )

    # ── shared carrier bootstrap (committed seed => identical resample) ──
    rng_boot = np.random.default_rng([AN.BOOT_SEED])
    idx_draws = rng_boot.integers(0, 12, size=(cfg.b_boot, 12))
    mult = AN.carrier_multiplicities(idx_draws, 12)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    verify: dict = {
        "row_bank_mean_max_abs_dev": dev_mean,
        "perpair_obs_max_abs_dev": dev_obs,
        "perpair_noise_max_abs_dev": dev_noise,
        "point_max_abs_dev": 0.0,
        "draws_max_abs_dev": 0.0,
        "delta_draws_identity_max_abs_dev": float(
            max(_max_abs(draws9_committed[a] - draws7[a], delta_committed[a]) for a in parent_axes)
        ),
    }

    def _drop_counts(sel: np.ndarray) -> dict:
        ctxs = np.unique(np.concatenate([pa.a[sel], pa.b[sel]]))
        return {
            "n_rows_dropped": int((excl & st9.draw_valid)[ctxs].sum()),
            "n_ctx_in_selection": int(ctxs.size),
            "n_ctx_all_draws_lost": int((cnt_excl[ctxs] == 0).sum()),
            "n_pairs_obs_nan": int((~np.isfinite(norm_obs_excl[sel])).sum()),
            "n_pairs_noise_nan": int((~np.isfinite(rel_excl["noise_norm"][sel])).sum()),
        }

    axes_out: dict = {}
    units = parent_axes + [a for a in AN.PILOT_AXES if a in views]
    for k, axis in enumerate(units, start=1):
        view = views[axis]
        prim = view.primary_idx
        if axis in committed:  # 11 parent axes: frozen crossmodel selection
            row = committed[axis]
            fire = row["fire"]
            assert prim.size == fire["n_shared_primary"], (axis, prim.size, fire)
            sym = fired9[prim] & fired7[prim]
            assert int(sym.sum()) == fire["n_symmetric_fired"], (axis, int(sym.sum()), fire)
            assert int((fired9[prim] & ~fired7[prim]).sum()) == fire["n_dropped_7b_only"], axis
            assert int((~fired9[prim] & fired7[prim]).sum()) == fire["n_dropped_9b_only"], axis
            sel = prim[sym] if fire["symmetric_headline"] else prim
            pt_u, dr_u = sep_point_and_draws(
                norm_obs_unexcl, rel_unexcl["noise_norm"], sel, pa, mult
            )
            pt_x, dr_x = sep_point_and_draws(norm_obs_excl, rel_excl["noise_norm"], sel, pa, mult)
            dev_pt = abs(pt_u - row["s_9b"])
            dev_dr = _max_abs(dr_u, draws9_committed[axis])
            assert dev_pt < 1e-6, (axis, pt_u, row["s_9b"])
            assert dev_dr < 1e-6, (axis, dev_dr)
            verify["point_max_abs_dev"] = max(verify["point_max_abs_dev"], dev_pt)
            verify["draws_max_abs_dev"] = max(verify["draws_max_abs_dev"], dev_dr)
            s7 = float(row["s_7b"])
            delta_after = pt_x - s7
            dd_after = dr_x - draws7[axis]
            ci_after = AN._ci(dd_after)
            out = {
                "axis": axis,
                "kind": "parent",
                "selection": {
                    "rule": "symmetric-fired shared primary pairs"
                    if fire["symmetric_headline"]
                    else "ALL shared primary pairs (committed fallback: symmetric_headline false)",
                    "n_pairs": int(sel.size),
                    "fire": fire,
                },
                "unexcluded": pt_u,
                "excluded": pt_x,
                "excluded_minus_unexcluded": pt_x - pt_u,
                "s_7b": s7,
                "delta_vs_7b_before": {
                    "point": float(row["delta_9b_minus_7b"]),
                    "ci95": [float(x) for x in row["delta_ci95"]],
                },
                "delta_vs_7b_after": {"point": delta_after, "ci95": ci_after},
                "gain_survives_exclusion": bool(delta_after > 0 and ci_after[0] > 0),
                "drops": _drop_counts(sel),
            }
        else:  # pilot axes: pilot-placement statistic (all primary pairs)
            bx = battery9["axes"][axis]
            flip_ref = bx["surface"]["observed"]["flip_norm_mean_all_values"]
            noise_ref_ax = bx["reliability"]["noise_norm_mean_all_values"]
            sel = prim
            pt_u, dr_u = sep_point_and_draws(
                norm_obs_unexcl, rel_unexcl["noise_norm"], sel, pa, mult
            )
            dev_pt = abs(pt_u - float(flip_ref) / float(noise_ref_ax))
            assert dev_pt < 1e-6, (axis, pt_u, flip_ref, noise_ref_ax)
            verify["point_max_abs_dev"] = max(verify["point_max_abs_dev"], dev_pt)
            pt_x, dr_x = sep_point_and_draws(norm_obs_excl, rel_excl["noise_norm"], sel, pa, mult)
            out = {
                "axis": axis,
                "kind": "pilot",
                "selection": {
                    "rule": "ALL primary pairs (pilot-placement statistic; no 7B counterpart)",
                    "n_pairs": int(sel.size),
                },
                "unexcluded": pt_u,
                "excluded": pt_x,
                "excluded_minus_unexcluded": pt_x - pt_u,
                "separation_ci95_unexcluded": AN._ci(dr_u),
                "separation_ci95_excluded": AN._ci(dr_x),
                "drops": _drop_counts(sel),
            }
            if axis == COMPLIANT_CELL:
                out["compliant_intrusions_kept"] = n_compliant_kept
        axes_out[axis] = out
        AN._write_json_atomic(
            args.ckpt_dir / f"intrusion_recount_{axis}.json", AN._json_sanitize(out)
        )
        print(
            f"[recount] unit {k}/{len(units)} {axis} elapsed={time.time() - t0:.1f}s "
            f"unexcl={pt_u:.4f} excl={pt_x:.4f}",
            flush=True,
        )

    prov = AN.git_provenance()
    doc = {
        "schema": SCHEMA,
        "statistic": sep_table["definition"],
        "note": (
            "9B-side CJK-intrusion-excluded recount: intruded draws removed from the "
            "per-context tail means + split-half noise at the frozen primary layer; pair "
            "selections, fire verdicts, and the 7B side are FROZEN to the committed run "
            "(crossmodel_contrasts.json + crossmodel_perdraw/obs_separation_snr.npz). "
            "answer_language instructed-Chinese rows are compliant and KEPT."
        ),
        "exclusion": {
            "regex": scan["regex"],
            "n_intruded_total": n_intruded_total,
            "n_rows_total": scan["total"]["total"],
            "n_compliant_kept_answer_language_chinese": n_compliant_kept,
            "n_excluded_applied": n_excl_applied,
            "n_excluded_removed_from_valid_draws": n_excl_valid,
            "n_contexts_all_draws_lost": n_ctx_dead,
            "per_cell": per_cell,
            "scan_parity": "per-cell + per-value counts asserted equal to the committed scan",
        },
        "verification": verify,
        "axes": axes_out,
        "bootstrap": {
            "kind": "carrier-clustered, ONE shared 12-carrier resample per draw (committed "
            "multiplicities regenerated deterministically)",
            "B": int(cfg.b_boot),
            "seed": AN.BOOT_SEED,
            "n_splits": int(cfg.n_splits),
            "split_seed": AN.SPLIT_SEED,
        },
        "layer_pair": cm["layer_pair"],
        "inputs": {
            "bank_manifest": {"path": str(cfg.bank_9b), "sha256": AN._sha256(cfg.bank_9b)},
            "intrusion_scan": {
                "path": str(RESULTS / "intrusion_scan_2587.json"),
                "sha256": AN._sha256(RESULTS / "intrusion_scan_2587.json"),
            },
            "crossmodel_contrasts": {
                "path": str(RESULTS / "crossmodel_contrasts.json"),
                "sha256": AN._sha256(RESULTS / "crossmodel_contrasts.json"),
            },
            "crossmodel_perdraw_npz": {"path": str(npz_path), "sha256": AN._sha256(npz_path)},
            "perpair": {
                "path": str(RESULTS / "perpair_2587.jsonl"),
                "sha256": AN._sha256(RESULTS / "perpair_2587.jsonl"),
            },
            "manipulation_check": {
                "path": str(cfg.manip_9b),
                "sha256": AN._sha256(cfg.manip_9b),
            },
            "stores": st9.input_files,
            "anchor_shard_dirs": [str(AN._REPO_ROOT / d) for d in JOIN.DEFAULT_ANCHOR_DIRS],
        },
        "meta": {
            **AN.as_metadata_dict(prov, phase="intrusion-excluded-recount"),
            "generated_utc": datetime.now(UTC).isoformat(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "wall_seconds": round(time.time() - t0, 1),
        },
    }
    AN._write_json_atomic(args.out_json, AN._json_sanitize(doc))
    print(f"[recount] wrote {args.out_json} in {time.time() - t0:.1f}s", flush=True)
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
