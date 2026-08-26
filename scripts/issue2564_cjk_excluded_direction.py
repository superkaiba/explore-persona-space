"""Issue #2564 free-analysis follow-up (round 1): CJK-excluded direction recompute.

Re-derives the per-axis direction-recovery cosines (``mean_cos_headline`` /
``mean_cos_all_values`` per arm, tail-inclusive L19 pooling + the span-mean
twin) with the 357 CJK-intruded rollout draws EXCLUDED from each context's
10-draw mean answer vector, and reports how the headline reads move vs the
as-scored values in ``eval_results/issue_2564/minpair_delta.json``.

Conventions are IMPORTED from the producing scripts, never re-implemented:

- CJK detection: ``issue2564_intrusion_audit.CJK_RE`` + ``_read_jsonl`` over
  the same 10 anchor shards (the excluded-draw set must reproduce the audit's
  357/9,840 intruded rollouts — asserted).
- Pooling / pairing / per-axis aggregation: ``issue2564_analysis`` —
  ``build_pair_arrays`` / ``build_axis_views`` / ``load_fire`` /
  ``pair_fired_mask`` / ``rowwise_cos`` / ``direction_null_draws`` are called
  directly; the per-context mean accumulation mirrors
  ``issue2564_analysis.load_stores`` bit-for-bit (same float64 conversion,
  same valid mask, same ``np.add.at`` order) so the as-scored recompute is a
  true parity reference.
- Predicted deltas are DRAW-INDEPENDENT (functions of the context-end states
  ``vc``, not of answer draws), so the committed fp32 prediction tensors
  ``eval_results/issue_2564/predictions/delta_pred_<arm>.pt`` are reused
  verbatim for both variants — no ridge staging, no vc store needed.

Parity gate (fail loud): the as-scored recompute must reproduce the stored
``direction.<arm>.mean_cos_headline`` / ``_all_values`` (and the span twin +
null q97.5) within ``--parity-tol`` (default 2e-5; the only expected noise is
the fp32 cast of the committed prediction tensors, ~1e-7) BEFORE the excluded
variant is trusted. The store-level check additionally compares the recomputed
as-scored observed deltas against the committed fp32
``delta_obs_tail_L19.pt``.

Inputs (all local mirrors / the allowed ``issue2564_minpair/`` HF prefix):
per-draw answer summaries ``analysis_tensors/va2564/va2564_<cell>.pt`` (staged
to ``data/issue_2564/hf_dl/pe_stage/`` if absent), local anchors staging (CJK
scan), ``eval_results/issue_2564/{bank_manifest,manipulation_check,
minpair_delta}.json`` + ``predictions/*.pt``.

Output: ``eval_results/issue_2564/cjk_excluded_direction.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF token BEFORE torch import (code-style.md)

import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2564_analysis as A  # noqa: E402
import issue2564_intrusion_audit as IA  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2564_cjk_excluded_direction")

_WT = Path(__file__).resolve().parents[1]
_ER = _WT / "eval_results" / "issue_2564"
EXPECTED_TOTAL = 9840
EXPECTED_INTRUDED = 357
MOVE_THRESHOLD = 0.01


# ── CJK scan (reuses the intrusion audit's regex + reader verbatim) ─────


def scan_intrusion(anchors_dir: Path) -> tuple[dict[tuple[str, int], bool], dict]:
    """(context_id, draw) -> intruded flag over all 9,840 rollouts; must
    reproduce the audit's 357/9,840 exactly (fail loud otherwise)."""
    intruded: dict[tuple[str, int], bool] = {}
    shards = sorted(anchors_dir.glob("anchors_*.jsonl"))
    assert shards, f"no anchor shards under {anchors_dir}"
    total = n_intr = 0
    for shard in shards:
        for r in IA._read_jsonl(shard):
            hit = IA.CJK_RE.search(r["text"]) is not None
            intruded[(r["context_id"], int(r["draw"]))] = hit
            total += 1
            n_intr += hit
    assert total == EXPECTED_TOTAL, (total, EXPECTED_TOTAL)
    assert n_intr == EXPECTED_INTRUDED, (
        f"CJK scan drift: {n_intr} intruded != audit's {EXPECTED_INTRUDED} "
        f"(convention: issue2564_intrusion_audit.CJK_RE over {anchors_dir})"
    )
    stats = {
        "total_rollouts": total,
        "intruded_rollouts": n_intr,
        "contexts_with_intrusion": len({c for (c, _d), h in intruded.items() if h}),
        "anchors_dir": str(anchors_dir),
        "n_shards": len(shards),
    }
    return intruded, stats


# ── per-context means, as-scored + CJK-excluded ────────────────────────


@dataclass
class MeanVariants:
    tail_as: np.ndarray  # (n_ctx, d) float64, all valid draws (parity path)
    tail_ex: np.ndarray  # (n_ctx, d) float64, valid & not intruded
    span_as: np.ndarray
    span_ex: np.ndarray
    cnt_as: np.ndarray  # (n_ctx,) valid draw counts
    cnt_ex: np.ndarray
    excl_cnt: np.ndarray  # (n_ctx,) valid-but-intruded draw counts
    zero_remaining: list[str]  # contexts with NO draw left after exclusion


def accumulate_means(
    cells: list[str],
    stage_dir: Path,
    ctx_ids: list[str],
    row_of: dict[str, int],
    intruded: dict[tuple[str, int], bool],
) -> MeanVariants:
    """Mirror of ``issue2564_analysis.load_stores``'s L19 accumulation (same
    float64 conversion, same valid mask, same ``np.add.at`` order per sorted
    cell) with a second accumulator excluding CJK-intruded draws. Vectorized:
    no per-row numeric python loops (index-record parsing mirrors the parent).
    """
    from explore_persona_space.orchestrate.hub import stage_hub_file

    n_ctx = len(ctx_ids)
    layers = list(A.LAYERS)
    li = {layer: k for k, layer in enumerate(layers)}
    d: int | None = None
    sums = None
    cnt_as = np.zeros(n_ctx, dtype=np.int64)
    cnt_ex = np.zeros(n_ctx, dtype=np.int64)
    excl_cnt = np.zeros(n_ctx, dtype=np.int64)

    for cell in cells:
        rel = f"analysis_tensors/va2564/va2564_{cell}.pt"
        p = stage_dir / rel
        if not p.exists():
            logger.info("[cjk] staging %s from HF (%s)", rel, A.HF_DATA_REPO)
            p = Path(stage_hub_file(A.HF_DATA_REPO, f"{A.HF_PREFIX_FULL}/{rel}", p))
        store = torch.load(p, map_location="cpu", weights_only=False)
        assert [int(x) for x in store["layers"]] == layers, store["layers"]
        idx_rows = store["index"]
        tail = store["va_tail_incl"].to(torch.float64).numpy()
        span = store["va_span"].to(torch.float64).numpy()
        n_rows = len(idx_rows)
        if d is None:
            d = tail.shape[2]
            sums = {
                name: np.zeros((n_ctx, d), dtype=np.float64)
                for name in ("tail_as", "tail_ex", "span_as", "span_ex")
            }
        assert tail.shape == (n_rows, len(layers), d), (tail.shape, n_rows, d)
        ctx_idx = np.array([row_of.get(rec["context_id"], -1) for rec in idx_rows], dtype=np.int64)
        n_comp = np.array([int(rec["n_completion_tokens"]) for rec in idx_rows], dtype=np.int64)
        empty_mask = np.zeros(n_rows, dtype=bool)
        empty_ids = np.array(sorted(int(i) for i in store.get("empty_rows", [])), dtype=np.int64)
        if empty_ids.size:
            empty_mask[empty_ids] = True
        # fail-loud coverage: every va row's (context_id, draw) must be in the scan
        intr = np.array(
            [intruded[(rec["context_id"], int(rec["draw"]))] for rec in idx_rows], dtype=bool
        )
        valid = (ctx_idx >= 0) & (n_comp > 0) & ~empty_mask
        keep_ex = valid & ~intr
        L = li[A.PRIMARY_LAYER]
        np.add.at(sums["tail_as"], ctx_idx[valid], tail[valid, L, :])
        np.add.at(sums["span_as"], ctx_idx[valid], span[valid, L, :])
        np.add.at(sums["tail_ex"], ctx_idx[keep_ex], tail[keep_ex, L, :])
        np.add.at(sums["span_ex"], ctx_idx[keep_ex], span[keep_ex, L, :])
        np.add.at(cnt_as, ctx_idx[valid], 1)
        np.add.at(cnt_ex, ctx_idx[keep_ex], 1)
        np.add.at(excl_cnt, ctx_idx[valid & intr], 1)

    assert sums is not None and d is not None
    zero_as = [ctx_ids[i] for i in range(n_ctx) if cnt_as[i] == 0]
    if zero_as:
        raise RuntimeError(f"contexts with ZERO valid draws (as-scored): {zero_as[:10]}")
    zero_remaining = [ctx_ids[i] for i in range(n_ctx) if cnt_ex[i] == 0]
    if zero_remaining:
        logger.warning(
            "[cjk] %d context(s) lose ALL draws under exclusion (means -> NaN, pairs "
            "touching them dropped from nanmeans): %s",
            len(zero_remaining),
            zero_remaining[:10],
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        tail_as = sums["tail_as"] / cnt_as[:, None]
        span_as = sums["span_as"] / cnt_as[:, None]
        tail_ex = np.where(cnt_ex[:, None] > 0, sums["tail_ex"] / cnt_ex[:, None], np.nan)
        span_ex = np.where(cnt_ex[:, None] > 0, sums["span_ex"] / cnt_ex[:, None], np.nan)
    return MeanVariants(
        tail_as=tail_as,
        tail_ex=tail_ex,
        span_as=span_as,
        span_ex=span_ex,
        cnt_as=cnt_as,
        cnt_ex=cnt_ex,
        excl_cnt=excl_cnt,
        zero_remaining=zero_remaining,
    )


# ── per-axis aggregation (mirrors compute_all's headline gating) ────────


def _headline_sel(
    fire: dict, axis: str, prim: np.ndarray, fired70: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Replicates issue2564_analysis.compute_all lines 1143-1157: the fire
    floor gates the headline; below-floor axes report NaN headline fields."""
    hmask = fired70[prim]
    ar = fire["axis_rows"].get(axis)
    floor_met = bool(ar["floor_met"]) if ar is not None else True
    compliance_limited = ar is not None and not floor_met
    no_fired_pairs = not bool(hmask.any())
    headline_ok = not compliance_limited and not no_fired_pairs
    head = prim[hmask] if headline_ok else np.array([], dtype=np.int64)
    return head, {
        "compliance_limited": compliance_limited,
        "no_fired_pairs": no_fired_pairs,
        "headline_ok": headline_ok,
    }


def _nm(vals: np.ndarray, sel: np.ndarray) -> float:
    if sel.size == 0:
        return float("nan")
    v = vals[sel]
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def append_robust_values(perpair_path: Path, out_path: Path) -> int:
    """Append the user-description robust-values read to the committed JSON.

    The intrusion audit leaves exactly two user-description values fired under
    every convention (as-scored, zeroed, excluded): retired engineer (v2) and
    business traveler (v5). This reads their 12 as-scored swap pairs from the
    committed ``perpair.jsonl`` and records the per-arm mean direction cosine
    under ``summary.user_profile_robust_values`` — the number quoted in the
    clean-result Takeaways (map 0.49 vs identity 0.36). Fail-loud: refuses to
    run if the pair count is not 12 or the output JSON is missing.
    """
    rows = [json.loads(line) for line in perpair_path.read_text().splitlines() if line.strip()]
    sel = [
        r
        for r in rows
        if r["axis"] == "user_profile"
        and r["pair_class"] == "swap"
        and {r["value_a"], r["value_b"]} == {"v2", "v5"}
    ]
    if len(sel) != 12:
        raise SystemExit(f"expected 12 v2-v5 user_profile swap pairs, found {len(sel)}")
    doc = json.loads(out_path.read_text())
    doc["summary"]["user_profile_robust_values"] = {
        "values": {"v2": "retired engineer", "v5": "business traveler"},
        "basis": "as-scored per-pair cos over the 12 v2-v5 swap pairs (perpair.jsonl); "
        "the two values that fire under every intrusion convention "
        "(as-scored / zeroed / excluded; see intrusion_audit.json)",
        "mean_cos": {
            arm: float(np.mean([r["cos"][arm] for r in sel]))
            for arm in ("arm_779ce", "arm_1738ce", "arm_iddelta")
        },
        "n_pairs": len(sel),
    }
    A._write_json_atomic(out_path, A._json_sanitize(doc))
    print(f"[robust-values] appended summary.user_profile_robust_values to {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument(
        "--anchors-dir",
        type=Path,
        default=_WT
        / "data/issue_2564/judge_work/anchors_staging/issue2564_minpair/raw_completions/anchors",
    )
    ap.add_argument("--bank", type=Path, default=_ER / "bank_manifest.json")
    ap.add_argument("--manip-check", type=Path, default=_ER / "manipulation_check.json")
    ap.add_argument("--minpair-delta", type=Path, default=_ER / "minpair_delta.json")
    ap.add_argument("--predictions-dir", type=Path, default=_ER / "predictions")
    ap.add_argument("--stage-dir", type=Path, default=_WT / "data/issue_2564/hf_dl/pe_stage")
    ap.add_argument("--out", type=Path, default=_ER / "cjk_excluded_direction.json")
    ap.add_argument("--b-null", type=int, default=A.B_NULL_DEFAULT)
    ap.add_argument(
        "--parity-tol",
        type=float,
        default=2e-5,
        help="max |recomputed as-scored - stored JSON| per headline/all-values/null read "
        "(expected noise: fp32 cast of the committed prediction tensors, ~1e-7)",
    )
    ap.add_argument(
        "--append-robust-values",
        action="store_true",
        help="only append the user-description intrusion-robust-values direction read "
        "(mean cosine over the retired-engineer vs business-traveler swap pairs, the two "
        "values that fire under every intrusion convention) to the existing --out JSON "
        "from the committed perpair.jsonl; touches nothing else",
    )
    args = ap.parse_args(argv)
    if getattr(args, "import_check", False):  # pragma: no cover - convention slot
        return 0
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(name)s %(message)s")
    if args.append_robust_values:
        return append_robust_values(args.minpair_delta.parent / "perpair.jsonl", args.out)
    t0 = time.time()

    # (1) CJK scan — must reproduce the audit's 357/9,840
    intruded, scan_stats = scan_intrusion(args.anchors_dir)
    print(
        f"[cjk] scan: {scan_stats['intruded_rollouts']}/{scan_stats['total_rollouts']} intruded "
        f"across {scan_stats['contexts_with_intrusion']} contexts",
        flush=True,
    )

    # (2) bank + pair arrays + views + fire masks (all imported conventions)
    bank = A.load_bank_manifest(args.bank)
    # provenance pin: the reproducibility card's values_sha256 (29e6bebc...) is the
    # sha of the instruction-values FILE bytes (issue2564_run._values_sha); the bank
    # manifest's values_sha256 field is the parsed-blob sha — record BOTH.
    import hashlib

    values_file = Path(A.BK.__file__).parent / A.BK.VALUES_FILENAME
    values_file_sha = hashlib.sha256(values_file.read_bytes()).hexdigest()
    assert values_file_sha.startswith("29e6bebc"), (
        f"instruction-values file sha drift: {values_file_sha} (card pins 29e6bebc...)"
    )
    contexts = bank["contexts"]
    ctx_ids = sorted(contexts)
    row_of = {cid: i for i, cid in enumerate(ctx_ids)}
    cells = sorted({contexts[cid]["cell"] for cid in ctx_ids})
    carriers = [c for c in A.BK.CARRIER_IDS if c in {contexts[cid]["carrier"] for cid in ctx_ids}]
    st = SimpleNamespace(ctx_ids=ctx_ids, row_of=row_of, carriers=carriers, cells=cells)
    pa = A.build_pair_arrays(bank, st, smoke=False)
    views = A.build_axis_views(pa, len(carriers))
    fire = A.load_fire(args.manip_check)
    fa, fb = A.pair_fired_mask(pa, fire, 70)
    fired70 = fa & fb

    # (3) per-context means (as-scored parity path + CJK-excluded)
    mv = accumulate_means(cells, args.stage_dir, ctx_ids, row_of, intruded)
    obs_as = mv.tail_as[pa.a] - mv.tail_as[pa.b]
    obs_ex = mv.tail_ex[pa.a] - mv.tail_ex[pa.b]
    span_as = mv.span_as[pa.a] - mv.span_as[pa.b]
    span_ex = mv.span_ex[pa.a] - mv.span_ex[pa.b]

    # (4) committed prediction deltas (draw-independent -> shared by variants)
    preds: dict[str, np.ndarray] = {}
    for arm in A.ARMS:
        obj = torch.load(
            args.predictions_dir / f"delta_pred_{arm}.pt", map_location="cpu", weights_only=False
        )
        assert list(obj["pair_ids"]) == list(pa.ids), f"pair_id order drift for {arm}"
        assert int(obj["layer"]) == A.PRIMARY_LAYER
        preds[arm] = obj["tensor"].to(torch.float64).numpy()

    # store-level parity: recomputed as-scored obs deltas vs committed fp32 tensor
    stored_obs = torch.load(
        args.predictions_dir / "delta_obs_tail_L19.pt", map_location="cpu", weights_only=False
    )["tensor"].numpy()
    obs_cast = obs_as.astype(np.float32)
    store_max_abs = float(np.max(np.abs(obs_cast - stored_obs)))
    store_bitexact = bool(np.array_equal(obs_cast, stored_obs))
    assert store_max_abs <= 1e-5, f"obs-delta store parity broken: max_abs={store_max_abs:.3e}"

    cos_as = {arm: A.rowwise_cos(preds[arm], obs_as) for arm in A.ARMS}
    cos_ex = {arm: A.rowwise_cos(preds[arm], obs_ex) for arm in A.ARMS}
    cos_span_as = {arm: A.rowwise_cos(preds[arm], span_as) for arm in A.ARMS}
    cos_span_ex = {arm: A.rowwise_cos(preds[arm], span_ex) for arm in A.ARMS}

    # (5) per-axis reads + parity asserts against the stored JSON
    ref = json.loads(args.minpair_delta.read_text())
    parity_rows: list[dict] = []
    axes_out: dict[str, dict] = {}
    excl_rows = np.flatnonzero(mv.excl_cnt > 0)
    excl_set = set(excl_rows.tolist())

    for k, (axis, view) in enumerate(sorted(views.items())):
        prim = view.primary_idx
        head, gate = _headline_sel(fire, axis, prim, fired70)
        rows = np.unique(np.concatenate([pa.a[head], pa.b[head]])) if head.size else np.array([])
        affected = [int(r) for r in rows if int(r) in excl_set]
        n_draws_excluded = int(mv.excl_cnt[affected].sum()) if affected else 0
        ref_dir = ref["axes"][axis]["direction"]
        ref_span = ref["axes"][axis]["pooling_twin_span"]

        # null draws: fresh per-variant rng streams with the production seed
        # recipe ([NULL_SEED, k], arms consumed in ARMS order) so the as-scored
        # q97.5 is a parity read and the excluded one uses identical draws.
        rng_as = np.random.default_rng([A.NULL_SEED, k])
        rng_ex = np.random.default_rng([A.NULL_SEED, k])
        arms_out: dict[str, dict] = {}
        for arm in A.ARMS:
            m_as = _nm(cos_as[arm], head)
            m_ex = _nm(cos_ex[arm], head)
            m_as_all = _nm(cos_as[arm], prim)
            m_ex_all = _nm(cos_ex[arm], prim)
            nd_as = A.direction_null_draws(
                view, obs_as, preds[arm], cos_as[arm][prim], args.b_null, rng_as
            )
            nd_ex = A.direction_null_draws(
                view, obs_ex, preds[arm], cos_ex[arm][prim], args.b_null, rng_ex
            )
            q_as = A._pct(nd_as, 97.5)
            q_ex = A._pct(nd_ex, 97.5)
            sp_as = _nm(cos_span_as[arm], head)
            sp_ex = _nm(cos_span_ex[arm], head)
            for name, got, want in (
                ("mean_cos_headline", m_as, ref_dir[arm]["mean_cos_headline"]),
                ("mean_cos_all_values", m_as_all, ref_dir[arm]["mean_cos_all_values"]),
                ("null_q97_5", q_as, ref_dir[arm]["null"]["q97_5"]),
                ("span_mean_cos_headline", sp_as, ref_span[arm]["mean_cos_headline"]),
            ):
                want_f = float("nan") if want is None else float(want)
                assert np.isfinite(got) == np.isfinite(want_f), (axis, arm, name, got, want_f)
                both_nan = not np.isfinite(got) and not np.isfinite(want_f)
                diff = 0.0 if both_nan else abs(got - want_f)
                parity_rows.append({"axis": axis, "arm": arm, "read": name, "abs_diff": diff})
            arms_out[arm] = {
                "mean_cos_as_scored": m_as,
                "mean_cos_cjk_excluded": m_ex,
                "delta": m_ex - m_as,
                "n_pairs": int(head.size),
                "n_contexts_affected": len(affected),
                "n_draws_excluded": n_draws_excluded,
                "mean_cos_all_values_as_scored": m_as_all,
                "mean_cos_all_values_cjk_excluded": m_ex_all,
                "delta_all_values": m_ex_all - m_as_all,
                "null_q97_5_as_scored": q_as,
                "null_q97_5_cjk_excluded": q_ex,
                "span_mean_cos_as_scored": sp_as,
                "span_mean_cos_cjk_excluded": sp_ex,
                "span_delta": sp_ex - sp_as,
            }
        axes_out[axis] = {
            "n_pairs_headline": int(head.size),
            "n_pairs_all_values": int(prim.size),
            "n_contexts_affected_headline": len(affected),
            "n_draws_excluded_headline": n_draws_excluded,
            **gate,
            "arms": arms_out,
        }

    assert len(parity_rows) == 4 * len(A.ARMS) * len(views), (
        f"parity read-count mismatch: {len(parity_rows)} rows != "
        f"4 x {len(A.ARMS)} arms x {len(views)} axes"
    )
    assert all(np.isfinite(r["abs_diff"]) for r in parity_rows), (
        "non-finite parity diff (one-sided NaN escaped the per-read assert)"
    )
    worst = max(parity_rows, key=lambda r: r["abs_diff"])
    assert worst["abs_diff"] <= args.parity_tol, (
        f"PARITY FAIL: as-scored recompute drifts from minpair_delta.json — "
        f"{worst['axis']}/{worst['arm']}/{worst['read']} abs_diff={worst['abs_diff']:.3e} "
        f"> tol {args.parity_tol:g}"
    )
    print(
        f"[cjk] parity PASS: {len(parity_rows)} reads reproduced, max |diff| = "
        f"{worst['abs_diff']:.3e} (tol {args.parity_tol:g}); obs-delta store max_abs="
        f"{store_max_abs:.3e} bitexact={store_bitexact}",
        flush=True,
    )

    # (6) summary
    summary: dict[str, dict] = {}
    for arm in A.ARMS:
        deltas = {
            ax: axes_out[ax]["arms"][arm]["delta"]
            for ax in axes_out
            if np.isfinite(axes_out[ax]["arms"][arm]["delta"])
        }
        moved = sorted(
            (ax for ax, dv in deltas.items() if abs(dv) > MOVE_THRESHOLD),
            key=lambda ax: -abs(deltas[ax]),
        )
        worst_ax = max(deltas, key=lambda ax: abs(deltas[ax])) if deltas else None
        summary[arm] = {
            "max_abs_delta_headline": abs(deltas[worst_ax]) if worst_ax else None,
            "max_abs_delta_axis": worst_ax,
            "axes_moving_gt_0p01": moved,
            "n_axes_with_finite_headline": len(deltas),
        }

    doc = {
        "meta": {
            "issue": A.ISSUE,
            "followup": "cjk_excluded_direction (Step 9a-ter free-analysis round 1)",
            "script": "scripts/issue2564_cjk_excluded_direction.py",
            "scan_convention": (
                "scripts/issue2564_intrusion_audit.py CJK_RE (ranges "
                + ", ".join(f"{a}-{b}" for a, b in IA._CJK_RANGES)
                + ") over the 10 anchor rollout shards; excluded = intruded draws removed "
                "from each context's valid-draw mean (numerator AND denominator)"
            ),
            "values_sha256": values_file_sha,
            "values_sha256_source": f"sha256 of {values_file.name} (instruction-values file)",
            "bank_values_blob_sha256": bank["values_sha256"],
            "pooling_primary": "tail-inclusive L19 (headline); span-mean L19 twin reported",
            "predictions_reuse": (
                "predicted deltas are draw-independent (functions of context-end states); "
                "committed fp32 predictions/delta_pred_<arm>.pt reused for both variants"
            ),
            "b_null": int(args.b_null),
            "null_seed_recipe": f"[{A.NULL_SEED}, k] per sorted-axis index k, arms in ARMS order",
            "parity": {
                "tol": args.parity_tol,
                "n_reads": len(parity_rows),
                "max_abs_diff": worst["abs_diff"],
                "worst_read": {kk: worst[kk] for kk in ("axis", "arm", "read")},
                "obs_delta_store_max_abs_fp32": store_max_abs,
                "obs_delta_store_bitexact_fp32": store_bitexact,
            },
            "elapsed_s": round(time.time() - t0, 1),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            **as_metadata_dict(git_provenance(), phase="cjk-excluded-direction"),
        },
        "rollout_scan": {
            **scan_stats,
            "contexts_zero_remaining_after_exclusion": mv.zero_remaining,
            "total_valid_draws_as_scored": int(mv.cnt_as.sum()),
            "total_valid_draws_cjk_excluded": int(mv.cnt_ex.sum()),
            "total_valid_draws_excluded": int(mv.excl_cnt.sum()),
        },
        "axes": axes_out,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    A._write_json_atomic(args.out, A._json_sanitize(doc))
    print(f"[cjk] wrote {args.out}", flush=True)
    print("[phase=done] cjk_excluded_direction complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
