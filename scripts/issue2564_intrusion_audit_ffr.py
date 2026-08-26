"""Issue #2564 re-elicitation round: persisted CJK language-intrusion recount.

Persists the fold-time Step 3.7 recount for the `floor-failed-reelicitation`
round (clean-result-critique r6, concern `fold-time-cjk-recount-unpersisted`):
the round pipeline never emitted an in-JSON intrusion audit, so the analyzer
computed it at fold time from the committed HF shards. This script recomputes
EXACTLY that recount from the PINNED HF revision and writes the durable
artifact `eval_results/issue_2564/floor-failed-reelicitation/
intrusion_audit_ffr.json`:

- per-rollout / per-judged-check intrusion flags (complete: the enumerated
  intruded (context_id, draw) set plus totals; any pair absent from the
  enumeration is NOT intruded);
- per-value fire recounts under all three conventions (as-scored / zeroed /
  excluded) + per-axis floor verdicts and fired-value lists;
- headline fired-pair counts per convention (the stance halving under
  zeroing);
- direction deltas: per-axis, per-arm headline/all-values mean cosines with
  the intruded draws excluded from each context's 10-draw mean, next to the
  as-scored recompute (parity-asserted against the committed
  `minpair_delta_ffr.json`).

Conventions are IMPORTED, never re-implemented: CJK ranges + reader from
``issue2564_intrusion_audit`` (`CJK_RE`, `_read_jsonl`, `_fire_rows`), fire
semantics from ``issue2564_judge`` (via `_fire_rows`), pooling / pairing /
gating from ``issue2564_analysis`` (ffr round modes) and
``issue2564_cjk_excluded_direction`` (`accumulate_means`, `_headline_sel`,
`_nm`). Fail-loud: recount totals are asserted against the fold-recorded
values (130/2,760 rollouts; 30/528 checks; the single zeroing-only s4a flip)
and the as-scored direction recompute must reproduce the committed JSON.
Pure counting: no completion text is printed or persisted, only counts + ids.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
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
import issue2564_cjk_excluded_direction as CJKD  # noqa: E402
import issue2564_intrusion_audit as IA  # noqa: E402

from explore_persona_space.orchestrate.hub import stage_hub_file  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2564_intrusion_audit_ffr")

_WT = Path(__file__).resolve().parents[1]
_ER_FFR = _WT / "eval_results" / "issue_2564" / "floor-failed-reelicitation"

PIN_REV_DEFAULT = "8168886e8436ff13bd49d6b47d1d5e2a861be8d6"
FFR_PREFIX = f"{A.HF_PREFIX_FULL}/raw_completions/{A.FFR_ROUND_SEG}"
FFR_TENSORS = f"{A.HF_PREFIX_FULL}/analysis_tensors/{A.FFR_ROUND_SEG}"
FFR_MANIFESTS = f"{A.HF_PREFIX_FULL}/manifests/{A.FFR_ROUND_SEG}"
ANCHOR_ARMS = ("hedging", "persona", "query", "stance")
AXES = ("stance", "persona", "hedging")

# Fold-recorded expectations (epm:interpretation v7, 2026-08-26) — fail loud
# on any drift from the fold-time recount this artifact persists.
EXPECTED_TOTAL = 2760
EXPECTED_INTRUDED = 130
EXPECTED_PER_ARM = {
    "persona": (80, 1200),
    "stance": (31, 960),
    "hedging": (14, 480),
    "query": (5, 120),
}
EXPECTED_JUDGED = 528
EXPECTED_JUDGED_INTRUDED = 30
EXPECTED_CROSSTAB = {"intruded_comply": 17, "intruded_noncomply": 13, "intruded_incomplete": 0}
DELTA_CEILING = 0.002  # body claim: no direction mean moves by more than 0.002


def _stage(rev: str, stage_dir: Path) -> dict[str, Path]:
    """Pinned-revision staging of every input the recount reads."""
    paths: dict[str, Path] = {}
    for arm in ANCHOR_ARMS:
        rel = f"{FFR_PREFIX}/anchors/anchors_{arm}.jsonl"
        paths[f"anchors_{arm}"] = Path(
            stage_hub_file(
                A.HF_DATA_REPO, rel, stage_dir / "anchors" / f"anchors_{arm}.jsonl", revision=rev
            )
        )
    paths["judge_scores"] = Path(
        stage_hub_file(
            A.HF_DATA_REPO,
            f"{FFR_PREFIX}/judge/judge_scores.jsonl",
            stage_dir / "judge_scores.jsonl",
            revision=rev,
        )
    )
    paths["bank"] = Path(
        stage_hub_file(
            A.HF_DATA_REPO,
            f"{FFR_MANIFESTS}/{A.BK.FFR_BANK_MANIFEST_FILENAME}",
            stage_dir / A.BK.FFR_BANK_MANIFEST_FILENAME,
            revision=rev,
        )
    )
    # va stores land in the layout CJKD.accumulate_means expects locally
    # (analysis_tensors/va2564/va2564_<cell>.pt under the stage dir).
    for cell in ANCHOR_ARMS:
        rel = f"{FFR_TENSORS}/va2564/va2564_{cell}.pt"
        paths[f"va_{cell}"] = Path(
            stage_hub_file(
                A.HF_DATA_REPO,
                rel,
                stage_dir / "analysis_tensors" / "va2564" / f"va2564_{cell}.pt",
                revision=rev,
            )
        )
    return paths


def scan_round_rollouts(anchors_dir: Path) -> tuple[dict[tuple[str, int], bool], dict]:
    """(context_id, draw) -> intruded over all 2,760 round rollouts."""
    intruded: dict[tuple[str, int], bool] = {}
    per_arm: dict[str, dict[str, int]] = {}
    for arm in ANCHOR_ARMS:
        shard = anchors_dir / f"anchors_{arm}.jsonl"
        n = n_intr = 0
        for r in IA._read_jsonl(shard):
            hit = IA.CJK_RE.search(r["text"]) is not None
            intruded[(r["context_id"], int(r["draw"]))] = hit
            n += 1
            n_intr += hit
        per_arm[arm] = {"intruded": n_intr, "total": n}
    total = sum(v["total"] for v in per_arm.values())
    total_intr = sum(v["intruded"] for v in per_arm.values())
    assert total == EXPECTED_TOTAL, (total, EXPECTED_TOTAL)
    assert total_intr == EXPECTED_INTRUDED, (
        f"CJK scan drift: {total_intr} intruded != fold-recorded {EXPECTED_INTRUDED}"
    )
    for arm, (want_i, want_n) in EXPECTED_PER_ARM.items():
        got = per_arm[arm]
        assert (got["intruded"], got["total"]) == (want_i, want_n), (arm, got, want_i, want_n)
    stats = {
        "per_arm": per_arm,
        "total": total,
        "total_intruded": total_intr,
        "contexts_with_intrusion": len({c for (c, _d), h in intruded.items() if h}),
    }
    return intruded, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pin-rev", default=PIN_REV_DEFAULT, help="HF data-repo revision to recount from"
    )
    ap.add_argument(
        "--stage-dir", type=Path, default=_WT / "data/issue_2564/hf_dl/pe_stage_ffr_pin"
    )
    ap.add_argument("--manip-check", type=Path, default=_ER_FFR / "manipulation_check_ffr.json")
    ap.add_argument("--minpair-delta", type=Path, default=_ER_FFR / "minpair_delta_ffr.json")
    ap.add_argument("--predictions-dir", type=Path, default=_ER_FFR / "predictions")
    ap.add_argument("--out", type=Path, default=_ER_FFR / "intrusion_audit_ffr.json")
    ap.add_argument("--parity-tol", type=float, default=2e-5)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(name)s %(message)s")

    paths = _stage(args.pin_rev, args.stage_dir)

    # ── (1) per-rollout scan (pinned shards) ──
    intruded, roll_stats = scan_round_rollouts(args.stage_dir / "anchors")
    intruded_ids = sorted(f"{c}::{d}" for (c, d), h in intruded.items() if h)
    print(
        f"[ffr-intrusion] rollouts {roll_stats['total_intruded']}/{roll_stats['total']} intruded "
        f"across {roll_stats['contexts_with_intrusion']} contexts",
        flush=True,
    )

    # ── (2) judged-pool join + per-value tallies ──
    judged_rows = IA._read_jsonl(paths["judge_scores"])
    assert len(judged_rows) == EXPECTED_JUDGED, len(judged_rows)
    crosstab = {"intruded_comply": 0, "intruded_noncomply": 0, "intruded_incomplete": 0}
    tallies: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "comply": 0,
            "noncomply": 0,
            "incomplete": 0,
            "intruded": 0,
            "intr_comply": 0,
            "intr_incomplete": 0,
        }
    )
    denom_of: dict[tuple[str, str], int] = defaultdict(int)
    intruded_checks: list[dict] = []
    for r in judged_rows:
        hit = intruded[(r["context_id"], int(r["draw"]))]
        t = tallies[(r["axis"], r["value_id"])]
        denom_of[(r["axis"], r["value_id"])] += 1
        t[r["outcome"]] += 1
        if hit:
            t["intruded"] += 1
            t["intr_comply"] += r["outcome"] == "comply"
            t["intr_incomplete"] += r["outcome"] == "incomplete"
            crosstab[f"intruded_{r['outcome']}"] += 1
            intruded_checks.append(
                {k: r[k] for k in ("alias", "axis", "value_id", "context_id", "draw", "outcome")}
            )
    n_judged_intr = sum(crosstab.values())
    assert n_judged_intr == EXPECTED_JUDGED_INTRUDED, (n_judged_intr, EXPECTED_JUDGED_INTRUDED)
    assert crosstab == EXPECTED_CROSSTAB, (crosstab, EXPECTED_CROSSTAB)

    # ── (3) per-value fire recounts + per-axis floors (all three conventions) ──
    slots = IA._fire_rows(tallies, denom_of)
    fire_ref = A.load_fire(args.manip_check)
    for key, s in slots.items():
        want = fire_ref["value_rows"][(s["axis"], s["value_id"])]["verdict"]
        assert s["verdict_orig"] == want, (key, s["verdict_orig"], want)
    axis_rows_out: dict[str, dict] = {}
    flips: list[str] = []
    for axis in AXES:
        base = [s for s in slots.values() if s["axis"] == axis and not s["value_id"].endswith("p")]
        floor = int(fire_ref["axis_rows"][axis]["floor"])  # parent-width-anchored floor
        row: dict = {
            "width": len(base),
            "floor": floor,
            "floor_source": "manipulation_check_ffr.json axis_rows",
        }
        for conv in ("orig", "zeroed", "excluded"):
            fired_vals = sorted(s["value_id"] for s in base if s[f"verdict_{conv}"] == "fired")
            row[f"n_fired_{conv}"] = len(fired_vals)
            row[f"floor_met_{conv}"] = len(fired_vals) >= floor
            row[f"fired_values_{conv}"] = fired_vals
        axis_rows_out[axis] = row
    for key, s in sorted(slots.items()):
        verdicts = {s["verdict_orig"], s["verdict_zeroed"], s["verdict_excluded"]}
        if len(verdicts) > 1:
            flips.append(key)
    assert flips == ["stance::s4a"], f"unexpected flip set: {flips}"
    s4a = slots["stance::s4a"]
    assert (s4a["n_comply"], s4a["verdict_zeroed"], s4a["excluded_counts"]) == (
        18,
        "not_fired",
        [16, 21],
    ), s4a
    assert all(
        axis_rows_out[a][f"floor_met_{c}"] for a in AXES for c in ("orig", "zeroed", "excluded")
    ), axis_rows_out

    # ── (4) pair arrays + per-convention headline fired-pair counts ──
    bank = A.load_bank_manifest(paths["bank"], is_ffr=True)
    contexts = bank["contexts"]
    ctx_ids = sorted(contexts)
    row_of = {cid: i for i, cid in enumerate(ctx_ids)}
    cells = sorted({contexts[cid]["cell"] for cid in ctx_ids})
    carriers = [c for c in A.BK.CARRIER_IDS if c in {contexts[cid]["carrier"] for cid in ctx_ids}]
    st = SimpleNamespace(ctx_ids=ctx_ids, row_of=row_of, carriers=carriers, cells=cells)
    pa = A.build_pair_arrays(bank, st, smoke=False, is_ffr=True)
    views = A.build_axis_views(pa, len(carriers))
    fired70_ref = None
    pairs_by_conv: dict[str, dict[str, int]] = {}
    for conv in ("orig", "zeroed", "excluded"):
        fmap = {(s["axis"], s["value_id"]): s[f"verdict_{conv}"] == "fired" for s in slots.values()}
        fire_conv = {"fired": {70: fmap}}
        fa, fb = A.pair_fired_mask(pa, fire_conv, 70)
        both = fa & fb
        if conv == "orig":
            fired70_ref = both
        for axis in AXES:
            pairs_by_conv.setdefault(axis, {})[conv] = int(both[views[axis].primary_idx].sum())
    assert fired70_ref is not None
    fa_ref, fb_ref = A.pair_fired_mask(pa, fire_ref, 70)
    assert np.array_equal(fired70_ref, fa_ref & fb_ref), "orig fired mask drifts from shipped gate"

    # ── (5) direction recount: as-scored parity + intruded-draws-excluded ──
    mv = CJKD.accumulate_means(cells, args.stage_dir, ctx_ids, row_of, intruded)
    obs_as = mv.tail_as[pa.a] - mv.tail_as[pa.b]
    obs_ex = mv.tail_ex[pa.a] - mv.tail_ex[pa.b]
    stored_obs = torch.load(
        args.predictions_dir / "delta_obs_tail_L19.pt", map_location="cpu", weights_only=False
    )["tensor"].numpy()
    store_max_abs = float(np.max(np.abs(obs_as.astype(np.float32) - stored_obs)))
    assert store_max_abs <= 1e-5, f"obs-delta store parity broken: {store_max_abs:.3e}"

    preds: dict[str, np.ndarray] = {}
    for arm in A.ARMS:
        obj = torch.load(
            args.predictions_dir / f"delta_pred_{arm}.pt", map_location="cpu", weights_only=False
        )
        assert list(obj["pair_ids"]) == list(pa.ids), f"pair_id order drift for {arm}"
        assert int(obj["layer"]) == A.PRIMARY_LAYER
        preds[arm] = obj["tensor"].to(torch.float64).numpy()

    ref = json.loads(args.minpair_delta.read_text())
    cos_as = {arm: A.rowwise_cos(preds[arm], obs_as) for arm in A.ARMS}
    cos_ex = {arm: A.rowwise_cos(preds[arm], obs_ex) for arm in A.ARMS}
    excl_rows = set(np.flatnonzero(mv.excl_cnt > 0).tolist())
    direction_out: dict[str, dict] = {}
    parity_rows: list[dict] = []
    deltas_all: list[float] = []
    for axis in AXES:
        view = views[axis]
        prim = view.primary_idx
        head, gate = CJKD._headline_sel(fire_ref, axis, prim, fired70_ref)
        rows = np.unique(np.concatenate([pa.a[head], pa.b[head]])) if head.size else np.array([])
        affected = [int(r) for r in rows if int(r) in excl_rows]
        arms_out: dict[str, dict] = {}
        for arm in A.ARMS:
            m_as = CJKD._nm(cos_as[arm], head)
            m_ex = CJKD._nm(cos_ex[arm], head)
            m_as_all = CJKD._nm(cos_as[arm], prim)
            m_ex_all = CJKD._nm(cos_ex[arm], prim)
            ref_dir = ref["axes"][axis]["direction"][arm]
            for name, got, want in (
                ("mean_cos_headline", m_as, ref_dir["mean_cos_headline"]),
                ("mean_cos_all_values", m_as_all, ref_dir["mean_cos_all_values"]),
            ):
                diff = abs(got - float(want))
                parity_rows.append({"axis": axis, "arm": arm, "read": name, "abs_diff": diff})
            deltas_all.extend([m_ex - m_as, m_ex_all - m_as_all])
            arms_out[arm] = {
                "mean_cos_as_scored": m_as,
                "mean_cos_cjk_excluded": m_ex,
                "delta": m_ex - m_as,
                "mean_cos_all_values_as_scored": m_as_all,
                "mean_cos_all_values_cjk_excluded": m_ex_all,
                "delta_all_values": m_ex_all - m_as_all,
            }
        direction_out[axis] = {
            "n_pairs_headline": int(head.size),
            "n_contexts_affected_headline": len(affected),
            "n_draws_excluded_headline": int(mv.excl_cnt[affected].sum()) if affected else 0,
            **gate,
            "arms": arms_out,
        }
    assert all(np.isfinite(r["abs_diff"]) for r in parity_rows), parity_rows
    worst = max(parity_rows, key=lambda r: r["abs_diff"])
    assert worst["abs_diff"] <= args.parity_tol, f"PARITY FAIL vs minpair_delta_ffr.json: {worst}"
    max_abs_delta = max(abs(d) for d in deltas_all if np.isfinite(d))
    assert max_abs_delta <= DELTA_CEILING, (
        f"direction delta {max_abs_delta:.4f} exceeds the fold-recorded ceiling {DELTA_CEILING}"
    )
    expected_heads = {"stance": 72, "persona": 120, "hedging": 12}
    got_heads = {a: direction_out[a]["n_pairs_headline"] for a in AXES}
    assert got_heads == expected_heads, (got_heads, expected_heads)

    doc = {
        "meta": {
            "issue": A.ISSUE,
            "round": A.FFR_ROUND_SEG,
            "script": "scripts/issue2564_intrusion_audit_ffr.py",
            "purpose": (
                "durable record of the fold-time Step 3.7 CJK recount for the "
                "re-elicitation round (clean-result-critique r6 concern "
                "fold-time-cjk-recount-unpersisted)"
            ),
            "hf_revision": args.pin_rev,
            "inputs": sorted(str(p.relative_to(_WT)) for p in paths.values()),
            "cjk_ranges_hex": [[hex(a), hex(b)] for a, b in IA._CJK_RANGES],
            "fire_rule": (
                "verdicts via issue2564_intrusion_audit._fire_rows (fire_verdict from "
                "issue2564_judge); axis floors read from manipulation_check_ffr.json "
                "axis_rows (parent-width-anchored)"
            ),
            "conventions": {
                "orig": "as scored (no intrusion adjustment)",
                "zeroed": "intruded draw counted non-complying; fixed denominator",
                "excluded": "intruded draws removed from numerator and denominator",
            },
            "flag_semantics": (
                "intrusion flags are COMPLETE: a rollout/check is intruded iff its "
                "context_id::draw appears in rollouts.intruded_rollouts / "
                "judged_pool.intruded_checks; every other (context_id, draw) is clean"
            ),
            "provenance": as_metadata_dict(git_provenance()),
        },
        "rollouts": {**roll_stats, "intruded_rollouts": intruded_ids},
        "judged_pool": {
            "total": len(judged_rows),
            "total_intruded": n_judged_intr,
            "fired_overlap_crosstab": crosstab,
            "intruded_checks": intruded_checks,
        },
        "slot_recounts": slots,
        "axis_floor_verdicts": axis_rows_out,
        "headline_pairs_by_convention": pairs_by_conv,
        "direction_recount": {
            "basis": (
                "observed swap deltas re-derived from per-draw va2564 stores with "
                "intruded draws excluded; predicted deltas are draw-independent "
                "(committed fp32 prediction tensors reused); as-scored recompute "
                "parity-asserted against minpair_delta_ffr.json"
            ),
            "obs_delta_store_max_abs_fp32": store_max_abs,
            "parity_max_abs_diff": worst["abs_diff"],
            "max_abs_delta": max_abs_delta,
            "axes": direction_out,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    A._write_json_atomic(args.out, A._json_sanitize(doc))
    print(
        f"[ffr-intrusion] wrote {args.out} — rollouts {roll_stats['total_intruded']}/"
        f"{roll_stats['total']}, judged {n_judged_intr}/{len(judged_rows)}, flips: "
        f"{flips}, stance headline pairs by convention {pairs_by_conv['stance']}, "
        f"max |direction delta| {max_abs_delta:.5f}, parity max |diff| {worst['abs_diff']:.2e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
