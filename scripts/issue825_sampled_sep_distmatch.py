"""Issue #825 `sampled-separator-control` plan-section-6 distribution-matched arm-B refit.

The pre-registered trigger fired (|Delta_dec| > 0.10 AND the arm-B span-length /
separator-mix distributions shifted vs round 7), so this driver separates
TEXT-STATISTICS from DECODING in the D collapse:

  1. Join the round-8 arm-B store rows to their pairs (``row_id`` ->
     span length = ``t_spans[0][1] - t_spans[0][0]``, + ``meta.sep_char``)
     from the pinned round-8 pairs jsonl (@ ``R8_REV``).
  2. Build round-7's span-length histogram from the pinned round-7 pairs
     jsonl (@ ``R7_REV``), fixed-width integer bins (``--bin-width``).
  3. Per seed (931..935): subsample the round-8 arm-B rows to the round-7
     histogram (largest-remainder within-bin uniform draws), PLUS an
     n-matched group-stratified RANDOM control at the SAME n (isolates the
     span-length-matching effect from the pure n reduction — held-out R^2 is
     steeply n-dependent, plan section 1).
  4. Refit the round's committed fit ladder on each subsample @ L19: the
     rotated random-projection control (``fit825.random_projection_control``,
     the decision estimator) + the batched MLP secondary
     (``fit931._mlp_fold_r2`` -> ``fit_batched_split_mlp``; Gram/dual-space +
     batched — no serial per-cell loop anywhere).
  5. D per the round's committed convention: ``d_stat(w_on, w_ex, ceiling)``
     with ``w_on = max(seed-mean rotated@L19, seed-mean MLP@L19)`` (the
     matched-n seed-mean convention) and the ROUND'S OWN anchors read from
     the committed ``decision_support.json`` (``w_ex_effective`` per model +
     ``ceiling_fulln``) — so D_dm is directly comparable to the unmatched
     sampled D (base 0.031 / instruct 0.086).

Outputs (git, JSON/text only):
  <out>/cells_<model>.json   per-seed fits (dm + random control) + bin table
  <out>/summary.json         headline D comparison, realized n, achieved moments

CLI:
  uv run python scripts/issue825_sampled_sep_distmatch.py \
      --out eval_results/issue_825/sampled-separator-control/distmatch_armB \
      --stage /mnt/eps-data/thomasjiralerspong/i825_sampled_sep_stage/distmatch
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before torch/numpy import

import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_base_sep_transfer as transfer  # noqa: E402
import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402
from issue825_sampled_sep_decision import d_stat  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

SCRIPT = "scripts/issue825_sampled_sep_distmatch.py"
R8_REV = "1338d264"  # round-8 UPLOAD-1 revision (analyzer-verified pin)
R7_REV = "4435ced2273df379f3e1c15bf5cdf56ca2ba40ae"  # round-7 pairs/stores pin
STORE_PREFIX = "issue825_sampled_sep_control/analysis_tensors/armB_{m}"
R8_PAIRS = "issue825_sampled_sep_control/raw_completions/generation/{m}/armB/pairs/pairs_armC.jsonl"
R7_PAIRS = "issue825_onpolicy_sep_control/raw_completions/generation/{m}/pairs/pairs_armC.jsonl"
DEFAULT_DECISION = Path("eval_results/issue_825/sampled-separator-control/decision_support.json")
DEFAULT_SEEDS = (931, 932, 933, 934, 935)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", type=str, default="base,instruct")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results/issue_825/sampled-separator-control/distmatch_armB"),
    )
    ap.add_argument(
        "--stage",
        type=Path,
        default=Path("/mnt/eps-data/thomasjiralerspong/i825_sampled_sep_stage/distmatch"),
    )
    ap.add_argument("--decision-support", type=Path, default=DEFAULT_DECISION)
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--bin-width", type=int, default=4, help="span-length histogram bin width")
    ap.add_argument("--folds", type=int, default=common.N_FOLDS)
    ap.add_argument("--fit-seed", type=int, default=common.FIT_SEED)
    ap.add_argument("--mlp-epochs", type=int, default=300)
    ap.add_argument("--skip-mlp", action="store_true", help="rotated-only (probe/smoke speed)")
    ap.add_argument(
        "--limit-rows", type=int, default=None, help="SMOKE ONLY: truncate store rows after load"
    )
    return ap.parse_args()


def _dl_jsonl(path_in_repo: str, revision: str, stage: Path) -> list[dict]:
    """Download one pinned jsonl from the data repo and parse it (fail-loud)."""
    from huggingface_hub import hf_hub_download

    local = Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                common.HF_DATA_REPO,
                path_in_repo,
                repo_type="dataset",
                revision=revision,
                local_dir=stage / "pairs_dl",
            ),
            what=f"stage {path_in_repo}@{revision}",
        )
    )
    rows = []
    with open(local, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert rows, f"empty pairs file {path_in_repo}@{revision}"
    return rows


def _span_of(row: dict) -> int:
    lo, hi = row["t_spans"][0][0], row["t_spans"][0][1]
    assert hi > lo, (row["row_id"], row["t_spans"])
    return int(hi - lo)


def _bin_edges(spans: np.ndarray, width: int) -> np.ndarray:
    lo = int(spans.min()) // width * width
    hi = (int(spans.max()) // width + 1) * width
    return np.arange(lo, hi + width, width)


def match_counts(
    target_spans: np.ndarray, avail_spans: np.ndarray, width: int
) -> tuple[np.ndarray, list[dict], float, np.ndarray]:
    """Per-bin take counts matching the target histogram against availability.

    Returns (edges, per-bin table, dropped_target_mass). Bins where the target
    has mass but round 8 has NO rows are dropped from the target (renormalized)
    and their mass reported; n* = max subsample size with exact largest-
    remainder proportions subject to per-bin availability.
    """
    edges = _bin_edges(np.concatenate([target_spans, avail_spans]), width)
    t_cnt, _ = np.histogram(target_spans, bins=edges)
    a_cnt, _ = np.histogram(avail_spans, bins=edges)
    droppable = (t_cnt > 0) & (a_cnt == 0)
    dropped_mass = float(t_cnt[droppable].sum() / t_cnt.sum())
    keep = (t_cnt > 0) & (a_cnt > 0)
    p = t_cnt[keep] / t_cnt[keep].sum()
    n_star = int(np.floor((a_cnt[keep] / p).min()))
    base = np.floor(p * n_star).astype(int)
    rem = p * n_star - base
    short = n_star - int(base.sum())
    if short > 0:
        order = np.argsort(-rem, kind="stable")
        for gi in order:
            if short == 0:
                break
            if base[gi] < a_cnt[keep][gi]:
                base[gi] += 1
                short -= 1
        n_star -= short  # residual shortfall when top-remainder bins are full
    take = np.zeros(len(t_cnt), dtype=int)
    take[keep] = base
    assert (take <= a_cnt).all(), "per-bin take exceeds availability"
    table = [
        {
            "bin_lo": int(edges[i]),
            "bin_hi": int(edges[i + 1]),
            "target_count_r7": int(t_cnt[i]),
            "available_r8": int(a_cnt[i]),
            "taken": int(take[i]),
        }
        for i in range(len(t_cnt))
        if t_cnt[i] > 0 or a_cnt[i] > 0
    ]
    return edges, table, dropped_mass, take


def dm_indices(
    rng: np.random.Generator, spans8: np.ndarray, edges: np.ndarray, take: np.ndarray
) -> np.ndarray:
    """Within-bin uniform draws without replacement realizing the take counts."""
    which = np.digitize(spans8, edges) - 1
    idx = []
    for b in range(len(take)):
        if take[b] == 0:
            continue
        pool = np.flatnonzero(which == b)
        assert len(pool) >= take[b], (b, len(pool), int(take[b]))
        idx.append(rng.choice(pool, size=int(take[b]), replace=False))
    out = np.sort(np.concatenate(idx))
    return out


def _fit_pair(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    idx: np.ndarray,
    pos19: int,
    args,
) -> dict:
    """Rotated (+ optional MLP) @ L19 on one subsample — the committed ladder."""
    t0 = time.time()
    rot = fit825.random_projection_control(
        X[idx], Y[idx], groups[idx], layers=[pos19], n_folds=args.folds, seed=args.fit_seed
    )[str(pos19)]
    t_rot = time.time() - t0
    entry = {"rotated_l19": float(rot), "t_rotated_s": round(t_rot, 1)}
    if not args.skip_mlp:
        t0 = time.time()
        mlp = fit931._mlp_fold_r2(
            X[idx],
            Y[idx],
            groups[idx],
            layers=[pos19],
            n_draws=0,
            folds=args.folds,
            seed=args.fit_seed,
            max_epochs=args.mlp_epochs,
        )[str(pos19)]["r2_obs"]
        entry["mlp_l19"] = float(mlp)
        entry["t_mlp_s"] = round(time.time() - t0, 1)
    entry["n"] = len(idx)
    entry["n_groups"] = len(np.unique(groups[idx]))
    return entry


def _regime(args, m: str) -> dict:
    return {
        "model": m,
        "seeds": sorted(int(s) for s in args.seeds.split(",") if s.strip()),
        "bin_width": args.bin_width,
        "folds": args.folds,
        "fit_seed": args.fit_seed,
        "mlp_epochs": args.mlp_epochs,
        "skip_mlp": bool(args.skip_mlp),
        "limit_rows": args.limit_rows,
        "r8_rev": R8_REV,
        "r7_rev": R7_REV,
    }


def run_model(args, m: str, dec: dict) -> dict:
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    stage_m = args.stage / m
    stage_m.mkdir(parents=True, exist_ok=True)
    armc = transfer.load_base_armc(
        stage_m,
        None,
        store_prefix=STORE_PREFIX.format(m=m),
        cache_name=f"distmatch_armB_frozen_{m}.npz",
        expect_n=None,
    )
    X, Y = armc["x_sep"], armc["y"]
    groups, row_ids = armc["group_ids"], armc["row_ids"]
    if args.limit_rows is not None:  # SMOKE ONLY
        X, Y = X[: args.limit_rows], Y[: args.limit_rows]
        groups, row_ids = groups[: args.limit_rows], row_ids[: args.limit_rows]
    pos19 = armc["frozen"].index(common.HEADLINE_LAYER)

    r8_rows = _dl_jsonl(R8_PAIRS.format(m=m), R8_REV, stage_m)
    r7_rows = _dl_jsonl(R7_PAIRS.format(m=m), R7_REV, stage_m)
    span_by_id = {r["row_id"]: _span_of(r) for r in r8_rows}
    sep_by_id = {r["row_id"]: r.get("meta", {}).get("sep_char", "?") for r in r8_rows}
    missing = [rid for rid in row_ids if rid not in span_by_id]
    assert not missing, f"{m}: {len(missing)} store rows missing from r8 pairs ({missing[:3]})"
    spans8 = np.asarray([span_by_id[rid] for rid in row_ids], dtype=np.int64)
    seps8 = np.asarray([sep_by_id[rid] for rid in row_ids])
    spans7 = np.asarray([_span_of(r) for r in r7_rows], dtype=np.int64)
    seps7 = np.asarray([r.get("meta", {}).get("sep_char", "?") for r in r7_rows])

    edges, table, dropped_mass, take = match_counts(spans7, spans8, args.bin_width)
    n_dm = int(take.sum())
    print(
        f"[i825-ss-dm] {m}: n_r8={len(spans8)} n_r7_target={len(spans7)} -> n_dm={n_dm} "
        f"(dropped target mass {dropped_mass:.4f}; bins={len(table)})",
        flush=True,
    )
    assert n_dm >= 100, f"{m}: matched subsample degenerate (n_dm={n_dm})"

    cells_path = args.out / f"cells_{m}.json"
    regime = _regime(args, m)
    per_seed: dict[str, dict] = {}
    if cells_path.exists():
        prior = json.loads(cells_path.read_text())
        assert prior.get("regime") == regime, (
            f"{m}: existing {cells_path} was written under a different regime "
            f"({prior.get('regime')} != {regime}) — delete it to re-run"
        )
        per_seed = prior.get("per_seed", {})
        print(f"[i825-ss-dm] {m}: resume — {sorted(per_seed)} already complete", flush=True)

    for s in seeds:
        if str(s) in per_seed:
            continue
        rng = np.random.default_rng(s)
        idx_dm = dm_indices(rng, spans8, edges, take)
        idx_rand = common.group_stratified_subsample(groups, n_dm, seed=s)
        assert len(idx_rand) == n_dm, (len(idx_rand), n_dm)
        dm = _fit_pair(X, Y, groups, idx_dm, pos19, args)
        dm["span_mean"] = float(spans8[idx_dm].mean())
        dm["dot_sep_fraction"] = float((seps8[idx_dm] == ".").mean())
        rand = _fit_pair(X, Y, groups, idx_rand, pos19, args)
        rand["span_mean"] = float(spans8[idx_rand].mean())
        rand["dot_sep_fraction"] = float((seps8[idx_rand] == ".").mean())
        per_seed[str(s)] = {"distmatched": dm, "random_n_control": rand}
        print(
            f"[i825-ss-dm] {m} seed={s}: dm rot={dm['rotated_l19']:.6f} "
            f"mlp={dm.get('mlp_l19')} span_mean={dm['span_mean']:.2f} | "
            f"rand rot={rand['rotated_l19']:.6f} mlp={rand.get('mlp_l19')}",
            flush=True,
        )
        # Checkpoint per seed (intra-phase persistence; atomic via write_json).
        common.write_json(
            cells_path,
            {
                "metadata": common.metadata(SCRIPT, args.fit_seed, n_dm),
                "followup_label": "sampled-separator-control",
                "regime": regime,
                "headline_layer": common.HEADLINE_LAYER,
                "frozen_layers": armc["frozen"],
                "store_revision": armc["revision"],
                "n_r8": len(spans8),
                "n_r7_target": len(spans7),
                "n_dm": n_dm,
                "dropped_target_mass": dropped_mass,
                "bin_table": table,
                "per_seed": per_seed,
            },
        )

    def _mean(kind: str, key: str) -> float | None:
        vals = [per_seed[str(s)][kind].get(key) for s in seeds]
        return float(np.mean(vals)) if all(v is not None for v in vals) else None

    arm = dec["per_model"][m]["arms"]["armB"]
    ceiling = float(dec["per_model"][m]["committed_reference"]["ceiling_fulln"])
    w_ex = float(arm["w_ex_effective"])
    reads = {}
    for kind in ("distmatched", "random_n_control"):
        mean_rot = _mean(kind, "rotated_l19")
        mean_mlp = _mean(kind, "mlp_l19")
        w_on = max(mean_rot, mean_mlp) if mean_mlp is not None else mean_rot
        reads[kind] = {
            "seed_mean_rotated_l19": mean_rot,
            "seed_mean_mlp_l19": mean_mlp,
            "w_on": w_on,
            "D": d_stat(w_on, w_ex, ceiling),
            "per_seed_D_rotated": {
                str(s): d_stat(per_seed[str(s)][kind]["rotated_l19"], w_ex, ceiling) for s in seeds
            },
            "span_mean": _mean(kind, "span_mean"),
            "dot_sep_fraction": _mean(kind, "dot_sep_fraction"),
        }
    return {
        "n_r8": len(spans8),
        "n_r7_target": len(spans7),
        "n_dm": n_dm,
        "dropped_target_mass": dropped_mass,
        "span_mean_r8_full": float(spans8.mean()),
        "span_mean_r7_target": float(spans7.mean()),
        "dot_sep_fraction_r8_full": float((seps8 == ".").mean()),
        "dot_sep_fraction_r7_target": float((seps7 == ".").mean()),
        "anchors": {"w_ex_effective": w_ex, "w_ex_kind": arm["w_ex_kind"], "ceiling": ceiling},
        "unmatched_reference": {
            "D_armB_r8": float(arm["D"]),
            "w_max_armB_r8": float(arm["reads"]["w_max"]),
            "D_r7": float(dec["per_model"][m]["round7_reference"]["D_r7"]),
        },
        **reads,
    }


def main() -> int:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    assert set(models) <= {"base", "instruct"}, models
    dec = json.loads(args.decision_support.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for m in models:
        summary[m] = run_model(args, m, dec)
    common.write_json(
        args.out / "summary.json",
        {
            "metadata": common.metadata(SCRIPT, args.fit_seed, 0),
            "followup_label": "sampled-separator-control",
            "what": (
                "plan-section-6 distribution-matched arm-B refit: round-8 arm-B rows "
                "subsampled to round-7's span-length histogram (+ an n-matched random "
                "control), rotated+MLP @ L19, D via d_stat against the round's committed "
                "w_ex_effective/ceiling anchors (decision_support.json)"
            ),
            "convention": (
                "w_on = max(seed-mean rotated@L19, seed-mean MLP@L19) over seeds — the "
                "matched-n seed-mean convention (issue825_onpolicy_sep_matchedn.py); "
                "D_dm - D_random_n_control isolates the span-length-matching effect at "
                "identical n and identical anchors"
            ),
            "per_model": summary,
        },
    )
    for m in models:
        s = summary[m]
        print(
            f"[i825-ss-dm] {m}: D_dm={s['distmatched']['D']:.4f} "
            f"D_rand_n={s['random_n_control']['D']:.4f} "
            f"D_unmatched_r8={s['unmatched_reference']['D_armB_r8']:.4f} "
            f"D_r7={s['unmatched_reference']['D_r7']:.4f} n_dm={s['n_dm']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
