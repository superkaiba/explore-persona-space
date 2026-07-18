"""Issue #1335: ladder fit battery — thin driver over the #1310-vectorized fit825 core.

Per (rung, model, arm) cell: GCV Gram-ridge held-out R^2 sweep over all layers
(K=5 group folds, fit seed 0), 20 selection-symmetric shuffle nulls, per-frozen-
layer row bootstrap + an L19 GROUP bootstrap (1000 draws, persisted — the delta
/ G / D pairing surface), mean-baseline + random-projection controls. Arms:
ctx (x_spanmean), prefix (x_prefixmean; degenerate control on r0-r2), lastpos
(x_last companion); r0 extras ctx_y96 + ctxnocap. Fiction rungs fit per persona
(scene-grouped folds); Q&A rungs fit one cell (row-level folds; r4 scenario-
grouped).

Matched-n control: every ctx/prefix cell refit at n_min (the smallest realized
ctx cell), 5 group-stratified subsample draws (seeds 931+k; singleton-group
cells degrade to a seeded uniform row draw — the group_stratified tie-break is
seed-degenerate on singletons), per-draw L19 group-bootstrap draws persisted.

Character-swap specificity on r7/r6/s1 (matched-scene-position derangement,
paired group bootstrap — the #1310 read, via issue1310_fit.swap_derangement).
Leave-one-setting-out (LOSO) read on r7 + r4 at L19.

Per-cell fit resume (NEW, plan §8 c24): cells_/matched_ JSONs carry
{rung_slug, render_config_hash, code_sha}; --resume skips a cell ONLY when the
persisted fingerprint matches the CURRENT config + SHA (mismatch => refit).

`build_ladder_summary` computes the oriented SIX-delta within-gap family
(label / header / framing / content+depth [+ Wren-matched companion] / foils /
label-restore), the OUTSIDE-family length delta (r0-referenced), G, Δ_max
(Bonferroni over the 6-family), D = Δ_max - 0.5·G (joint-draw CI, variance-sum
fallback), and the pre-registered verdict lattice per model — plus the two
binding rig-anchor gates (§7).

CLI:
  uv run python scripts/issue1335_fit.py --rung r7_endpoint --model base [...]
  uv run python scripts/issue1335_fit.py --matched-n --models base,instruct
  uv run python scripts/issue1335_fit.py --summary --models base,instruct [--smoke]
  uv run python scripts/issue1335_fit.py --seed-compare --models base,instruct \
      [--reference-summary eval_results/issue_1335/ladder_summary.json]
  uv run python scripts/issue1335_fit.py --verify-vectorized
  uv run python scripts/issue1335_fit.py --assert-cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402
import issue1310_common as c1310  # noqa: E402
import issue1310_fit as fit1310  # noqa: E402
import issue1335_render_rungs as r1335  # noqa: E402

SCRIPT = "scripts/issue1335_fit.py"

# #1335 r8 (att-20260715-210436 diagnosis): GCV lambda-selection degenerates at
# the grid-min lambda on the ladder's within-scene-correlated / near-singular
# cells (fiction per-persona n_tr ~1100-1440 << D=3584; one-line Q&A n_tr ~ D)
# — train RSS collapses to ~0 by interpolation and GCV picks lambda=0.01,
# producing held-out R^2 of -2..-46 while lambda=1e3-1e4 on the SAME folds
# reads +0.22..+0.35 (the #1310 anchor band). All ladder fits therefore select
# lambda by inner GROUP-level CV (fit825 "inner-group-cv"), identically for
# observed + null draws (selection-symmetric). r0's GCV pick (lambda=3162,
# R^2=0.410 = its own lambda-optimum) shows the selectors agree where GCV is
# healthy.
LAMBDA_SELECTION = "inner-group-cv"

ARM_X_KEY = {"ctx": "x_spanmean", "prefix": "x_prefixmean", "lastpos": "x_last"}
MATCHED_ARMS = ("ctx", "prefix")
MATCHED_SEED_BASE = 931  # matched-n subsample seeds 931+k (plan §10)
N_MATCHED_DRAWS = 5
SWAP_RUNGS = ("r7_endpoint", "r6_nofoil", "s1_assistant_label")
LOSO_RUNGS = ("r7_endpoint", "r4_fictionframe")

# #1310 v3 committed per-persona anchors (plan §7 gate 1; task #1310 v3 markers).
V3_ANCHORS_BASE = {"Wren": 0.137, "HELIOS": 0.148, "Dana": 0.147, "Vex": 0.106}
V3_ANCHORS_INSTRUCT = {"Wren": 0.235, "HELIOS": 0.253, "Dana": 0.188}
# Plan Amendment v4 (§7 recalibration): the v3 ±0.08 numeric band and the
# [0.55, 0.90] r0 bracket are RETIRED/DEMOTED as cross-selector-incomparable
# (GCV-regime references vs inner-group-CV reads) — kept ONLY as descriptive
# report fields. The binding numeric rig components are ROUND-8 REPRODUCTION
# checks: the round-8 validated inner-group-CV values on the SAME persisted
# stores the relaunch refits (same-surface AND same-selector). ±0.01 sits
# >=10x below the rig-break band (>=0.1 in every prior incident) and >=10x
# above cross-environment fp64 numerics on seeded folds (round 8 reproduced
# attempt-5 artifact values exactly). Source: plan.md § Amendment v4 / §7
# gates 1'/2'; epm:experiment-implementation v8 validation table.
ANCHOR_TOL = 0.08  # v3 band — descriptive only (amendment v4)
R0_GATE_RANGE = (0.55, 0.90)  # v3 bracket — report-not-halt diagnostic (amendment v4)
R8_REPRO_TOL = 0.01
R8_REPRO_R7 = {  # (model_kind, persona) -> round-8 validated L19 ctx full-n R^2
    ("instruct", "Dana"): 0.2545,
    ("base", "Wren"): 0.3585,
    ("base", "Vex"): 0.3050,
}
R8_REPRO_R0_BASE = 0.4103  # r0 base ctx L19 — selector-AGNOSTIC (both pick lambda=3162)
FICTION_YIELD_FLOOR = 1060  # 80% of the #1310 v3 per-persona floor (Dana-base 1325)

# The SIX-delta within-gap family (plan §5 delta map), oriented strong - weak.
DELTA_FAMILY = ("label", "header", "framing", "content_depth", "foils", "label_restore")

# onpolicy-assistant-label round (plan v7 §3/§4.2): the three fresh cells +
# the two registered within-run pairs the --label-compare summary reads.
LABEL_RUNGS = ("r7_op_assistant", "r7_op_wren", "r7_op_wren46")
LABEL_PAIRS = (
    # (name, strong/left slug, weak/right slug) — oriented left - right.
    ("delta_AW", "r7_op_assistant", "r7_op_wren"),  # PRIMARY registered contrast
    ("delta_wren_replicate", "r7_op_wren", "r7_op_wren46"),  # H0 anchor (v7 fix (b))
)
# Committed per-model matched n (body hyperparameter table; cross-checked at
# runtime against the committed per-model matched JSONs — NEVER sourced from
# matched_n_config.json, which holds a single last-writer n_min (plan v7
# fact-check corrective, assumption 17).
COMMITTED_PLACEMENT_N = {"base": 1397, "instruct": 1739}
# Committed base endpoint cells feeding the empirical H0 pair-noise band
# (v7 Statistics-critic fix (a)): 4 personas x 3 seeds, relative to
# --committed-eval-root.
H0_SEED_DIRS = (("seed42", "."), ("seed43", "seed43-gap-rungs"), ("seed44", "seed44-base-rungs"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rung", choices=list(r1335.RUNGS), default=None)
    ap.add_argument("--model", choices=list(r1335.MODEL_KINDS), default=None)
    ap.add_argument("--models", type=str, default="base,instruct")
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1335"))
    ap.add_argument("--store-dir", type=Path, default=None, help="default <data-dir>/store")
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1335"))
    ap.add_argument("--null-draws", type=int, default=c1310.N_NULL_DRAWS)
    ap.add_argument("--folds", type=int, default=c1310.N_FOLDS)
    ap.add_argument("--seed", type=int, default=c1310.FIT_SEED)
    ap.add_argument("--n-boot", type=int, default=c1310.N_BOOTSTRAP)
    ap.add_argument("--resume", action="store_true", help="c24 fingerprint-gated cell skip")
    ap.add_argument("--matched-n", action="store_true", help="run the matched-n refits")
    ap.add_argument(
        "--stage-from-hub",
        action="store_true",
        help="matched-n: re-stage deleted .pt shards from the Hub per rung, release after",
    )
    ap.add_argument("--summary", action="store_true", help="build ladder_summary.json + gates")
    ap.add_argument(
        "--seed-compare",
        action="store_true",
        help="seed43-gap-rungs follow-up: write seed_comparison.json (matched-n "
        "gap G + framing delta for THIS run's out-dir vs the committed seed-42 "
        "reference summary; no verdict lattice, no binding gates)",
    )
    ap.add_argument(
        "--reference-summary",
        type=Path,
        default=Path("eval_results/issue_1335/ladder_summary.json"),
        help="committed seed-42 ladder_summary.json the --seed-compare mode reads",
    )
    ap.add_argument(
        "--reference-summary-2",
        type=Path,
        default=None,
        help="OPTIONAL second seed-compare reference (a ladder_summary.json OR a "
        "prior round's seed_comparison.json — seed44-base-rungs passes the "
        "committed seed-43 seed_comparison); adds reference_2/cross_seed_2 blocks",
    )
    ap.add_argument(
        "--label-compare",
        action="store_true",
        help="onpolicy-assistant-label follow-up (plan v7 §4.2 item 4): write "
        "label_comparison.json (within-run Assistant-Wren paired delta + the "
        "Wren45-Wren46 replicate H0 pair + pairwise-matched/placement refits + "
        "combined-store cross-label swap + full-slot collapse audits + the "
        "registered empirical H0 pair-noise band; no gates, no verdict lattice)",
    )
    ap.add_argument(
        "--committed-eval-root",
        type=Path,
        default=Path("eval_results/issue_1335"),
        help="committed eval-results root the --label-compare read-only "
        "references resolve under (ladder cells, seed-round dirs, matched JSONs)",
    )
    ap.add_argument("--verify-vectorized", action="store_true")
    ap.add_argument("--assert-cuda", action="store_true", help="binding on-instance device gate")
    ap.add_argument("--smoke", action="store_true", help="numeric gates recorded, not binding")
    return ap.parse_args()


def frozen_layers(n_layers: int) -> list[int]:
    fl = [li for li in c1310.FROZEN_LAYERS if li < n_layers]
    return fl or [n_layers - 1]


def headline_layer(n_layers: int) -> int:
    return c1310.HEADLINE_LAYER if n_layers > c1310.HEADLINE_LAYER else n_layers - 1


def store_root(args) -> Path:
    return args.store_dir or (args.data_dir / "store")


def load_rung_store(args, slug: str, model_kind: str) -> dict:
    """Concatenate a (rung, model) store's shards; sidecar fingerprints must
    match the CURRENT render config + SHA (c24 — never fit a stale store)."""
    store_dir = store_root(args) / slug / model_kind
    shards = sorted(store_dir.glob(f"{model_kind}_shard*.pt"))
    assert shards, f"no {model_kind} shards under {store_dir}"
    rows, groups, chars, turns = [], [], [], []
    arrays: dict[str, list] = {}
    for sp in shards:
        side = json.loads(sp.with_suffix("").with_suffix(".json").read_text())
        assert r1335.fingerprint_matches(side, slug, require_sha=False), (
            f"stale store shard {sp}: render-config fingerprint mismatch (c24) — "
            "the shard was captured under a DIFFERENT rung render; quarantine it"
        )
        if not r1335.fingerprint_matches(side, slug):
            print(
                f"[i1335-fit] WARN: {sp.name} captured at code_sha={side.get('code_sha')} "
                "!= current (render config identical — consuming; resume-skip stays strict)"
            )
        payload = torch.load(sp, map_location="cpu", weights_only=False)
        rows.extend(payload["row_ids"])
        groups.extend(payload["group_ids"])
        chars.extend(payload["char_ids"])
        turns.extend(payload["turn_indices"])
        for k, v in payload["arrays"].items():
            arrays.setdefault(k, []).append(v.float().numpy().astype(np.float32))
    out = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    n = len(rows)
    for k, v in out.items():
        assert v.shape[0] == n, (k, v.shape, n)
    return {
        "row_ids": np.asarray(rows),
        "group_ids": np.asarray(groups),
        "char_ids": np.asarray(chars),
        "turn_indices": np.asarray(turns, dtype=int),
        "arrays": out,
    }


def matched_subsample(group_ids: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    """Group-stratified subsample; singleton-group cells use a seeded uniform
    row draw (the group_stratified tie-break is seed-DEGENERATE on singletons —
    every seed would pick the identical first-n rows)."""
    group_ids = np.asarray(group_ids)
    n = len(group_ids)
    if n_target >= n:
        return np.arange(n)
    if len(np.unique(group_ids)) == n:  # all-singleton groups
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=n_target, replace=False))
    return common.group_stratified_subsample(group_ids, n_target, seed=seed)


def _l19_group_bootstrap(sweep: dict, xy: dict, hl: int, args) -> dict | None:
    """Persisted-draws L19 group bootstrap (the delta/G/D pairing surface)."""
    if hl not in sweep["preds_frozen"]:
        return None
    fitted = sweep["fitted_mask"]
    pred = sweep["preds_frozen"][hl][fitted]
    true = xy["Y"][fitted, hl, :].astype(np.float64)
    groups = np.asarray(xy["group_ids"])[fitted]
    gb = fit931.group_bootstrap_r2(pred, true, groups, n_boot=args.n_boot, seed=args.seed)
    uniq = np.unique(groups)
    return {
        "r2": gb["r2"],
        "ci_lo": float(np.nanquantile(gb["draws"], 0.025)),
        "ci_hi": float(np.nanquantile(gb["draws"], 0.975)),
        "n_groups": int(gb["n_groups"]),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "group_universe_hash": _hash_strs(uniq),
        "draws": [float(v) for v in gb["draws"]],
    }


def _hash_strs(values) -> str:
    import hashlib

    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def fit_cell(cell_id: str, slug: str, xy: dict, args) -> dict:
    """Full-n cell: held-out sweep + nulls + baselines + bootstraps; fingerprinted."""
    fp = r1335.fingerprint(slug)
    out_path = args.out_dir / f"cells_{cell_id}.json"
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text())
        if r1335.fingerprint_matches(prev, slug):
            print(f"[i1335-fit] resume: cells_{cell_id}.json fingerprint match — skipped")
            return {"payload": prev, "skipped": True, "headline_layer": prev["headline_layer"]}
        print(f"[i1335-fit] resume: cells_{cell_id}.json STALE fingerprint — refitting")
    X, Y, groups = xy["X"], xy["Y"], xy["group_ids"]
    n, n_layers = X.shape[0], X.shape[1]
    fit825.FROZEN_LAYERS = tuple(frozen_layers(n_layers))
    sweep = fit825.heldout_r2_sweep(
        X,
        Y,
        groups,
        n_folds=args.folds,
        seed=args.seed,
        null_draws=args.null_draws,
        lambda_selection=LAMBDA_SELECTION,
    )
    r2_obs, r2_null = sweep["r2_obs"], sweep["r2_null"]
    summary = fit825.selection_symmetric_summary(r2_obs, r2_null)
    fl = frozen_layers(n_layers)
    hl = headline_layer(n_layers)
    mb = fit825.mean_baseline_r2(Y, groups, layers=fl, n_folds=args.folds, seed=args.seed)
    rp = fit825.random_projection_control(
        X,
        Y,
        groups,
        layers=[hl],
        n_folds=args.folds,
        seed=args.seed,
        lambda_selection=LAMBDA_SELECTION,
    )
    fitted = sweep["fitted_mask"]
    boot_row = {}
    for li in fl:
        if li not in sweep["preds_frozen"]:
            continue
        pred = sweep["preds_frozen"][li][fitted]
        true = Y[fitted, li, :].astype(np.float64)
        boot_row[str(li)] = fit825.bootstrap_r2_ci(
            pred, true, n_boot=args.n_boot, seed=args.seed + 100 + li
        )
    gb19 = _l19_group_bootstrap(sweep, xy, hl, args)
    payload = {
        "metadata": common.metadata(SCRIPT, args.seed, n),
        **fp,
        "cell_id": cell_id,
        "lambda_selection": LAMBDA_SELECTION,
        "n": n,
        "n_groups": len(np.unique(groups)),
        "n_layers": int(n_layers),
        "headline_layer": hl,
        "frozen_layers": fl,
        "r2_per_layer_obs": [float(v) for v in r2_obs],
        "selection_symmetric": summary,
        "mean_baseline_r2": mb,
        "random_projection_control_r2": rp,
        "skill_over_mean": {
            str(li): float(r2_obs[li]) - float(mb.get(str(li), float("nan"))) for li in fl
        },
        "r2_bootstrap_row_frozen": boot_row,
        "group_bootstrap_l19": gb19,
        "n_folds": args.folds,
        "null_draws": args.null_draws,
    }
    c1310.write_json(out_path, payload)
    c1310.write_json(
        args.out_dir / f"nulls_{cell_id}.json",
        {
            "metadata": common.metadata(SCRIPT, args.seed, n),
            **fp,
            "cell_id": cell_id,
            "layers": list(range(n_layers)),
            "observed_row": [float(v) for v in r2_obs],
            "null_matrix": [[float(v) for v in row] for row in r2_null],
            "null_layer_max_per_draw": summary["null_layer_max_r2_per_draw"],
        },
    )
    return {"sweep": sweep, "xy": xy, "payload": payload, "headline_layer": hl, "skipped": False}


def fit_cell_matched(cell_id: str, slug: str, xy: dict, n_min: int, args) -> dict:
    """Matched-n refit: 5 subsample draws (seeds 931+k), per-draw L19 group boot."""
    fp = r1335.fingerprint(slug)
    out_path = args.out_dir / f"matched_{cell_id}.json"
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text())
        if r1335.fingerprint_matches(prev, slug) and prev.get("n_min") == n_min:
            print(f"[i1335-fit] resume: matched_{cell_id}.json fingerprint match — skipped")
            return prev
    X, Y, groups = xy["X"], xy["Y"], xy["group_ids"]
    n_layers = X.shape[1]
    fit825.FROZEN_LAYERS = tuple(frozen_layers(n_layers))
    hl = headline_layer(n_layers)
    draws_out = []
    for k in range(N_MATCHED_DRAWS):
        seed_k = MATCHED_SEED_BASE + k
        idx = matched_subsample(groups, n_min, seed=seed_k)
        sub = {"X": X[idx], "Y": Y[idx], "group_ids": np.asarray(groups)[idx]}
        sweep = fit825.heldout_r2_sweep(
            sub["X"],
            sub["Y"],
            sub["group_ids"],
            n_folds=args.folds,
            seed=args.seed,
            null_draws=0,
            lambda_selection=LAMBDA_SELECTION,
        )
        gb19 = _l19_group_bootstrap(sweep, sub, hl, args)
        draws_out.append(
            {
                "subsample_seed": seed_k,
                "n": len(idx),
                "r2_per_layer": [float(v) for v in sweep["r2_obs"]],
                "r2_headline": float(sweep["r2_obs"][hl]),
                "group_bootstrap_l19": gb19,
            }
        )
    r2s = [d["r2_headline"] for d in draws_out]
    payload = {
        "metadata": common.metadata(SCRIPT, args.seed, n_min),
        **fp,
        "cell_id": cell_id,
        "lambda_selection": LAMBDA_SELECTION,
        "n_min": int(n_min),
        "headline_layer": hl,
        "n_draws": N_MATCHED_DRAWS,
        "r2_headline_mean": float(np.mean(r2s)),
        "r2_headline_per_draw": [float(v) for v in r2s],
        "draws": draws_out,
    }
    c1310.write_json(out_path, payload)
    return payload


def rung_units(slug: str, store: dict) -> list[tuple[str, dict]]:
    """Fit units: one per Q&A rung; one per LEAD persona for fiction rungs.

    The lead panel is rung-resolved (r1335.personas_for_rung — the committed
    4-persona panel for existing rungs, the single override lead for the
    r7_op_* label cells; without this the override cells' rows would be
    silently dropped from every fit — plan v7 §4.2 item 4, the
    highest-severity diff site)."""
    if r1335.RUNGS[slug]["family"] == "qa":
        return [("all", store)]
    units = []
    for persona in r1335.personas_for_rung(slug):
        m = store["char_ids"] == persona
        if not m.any():
            continue
        sub = {
            "row_ids": store["row_ids"][m],
            "group_ids": store["group_ids"][m],
            "char_ids": store["char_ids"][m],
            "turn_indices": store["turn_indices"][m],
            "arrays": {k: v[m] for k, v in store["arrays"].items()},
        }
        units.append((persona, sub))
    return units


def unit_cell_id(slug: str, model_kind: str, unit: str, arm: str) -> str:
    if unit == "all":
        return f"{slug}__{model_kind}__{arm}"
    return f"{slug}__{model_kind}__{unit}__{arm}"


def unit_xy(unit_store: dict, x_key: str, y_key: str = "y") -> dict:
    return {
        "X": unit_store["arrays"][x_key],
        "Y": unit_store["arrays"][y_key],
        "group_ids": unit_store["group_ids"],
        "row_ids": unit_store["row_ids"],
    }


def _swap_resume_payload(swap_path: Path, slug: str, resume: bool) -> dict | None:
    """Resumable swap payload iff --resume, the file exists, AND its fingerprint
    matches; else None (the component cells must then produce LIVE sweeps)."""
    if not (resume and swap_path.exists()):
        return None
    prev = json.loads(swap_path.read_text())
    if r1335.fingerprint_matches(prev, slug):
        return prev
    return None


def run_swap(slug: str, store: dict, model_kind: str, args) -> dict | None:
    """Character-swap specificity (matched-scene-position derangement) on a
    fiction rung: correct vs cross-persona-swapped Y, paired group bootstrap."""
    swap_path = args.out_dir / f"swap_{slug}_{model_kind}.json"
    prev = _swap_resume_payload(swap_path, slug, args.resume)
    if prev is not None:
        print(f"[i1335-fit] resume: swap_{slug}_{model_kind}.json fingerprint match — skipped")
        return prev
    rows, partners = fit1310.swap_derangement(
        store["group_ids"], store["char_ids"], store["turn_indices"], seed=c1310.BUILD_SEED
    )
    if len(rows) < 2 * args.folds:
        print(f"[i1335-fit] swap {slug}/{model_kind}: too few pairs (n={len(rows)}) — skipped")
        return None
    # The swap payload is absent (or stale): bypass resume on the two component
    # cells so they return live sweeps. Without this, a crash in the window
    # between the second component-cell write and the swap-payload write bricks
    # gate 1 on every --resume relaunch (both components skip, run_swap returns
    # None, gate 1 reads `_swap: missing` forever — round-1 review Minor 1).
    cell_args = args
    if args.resume:
        cell_args = copy.copy(args)
        cell_args.resume = False
    x = store["arrays"]["x_spanmean"]
    y = store["arrays"]["y"]
    g = store["group_ids"]
    correct_xy = {
        "X": x[rows],
        "Y": y[rows],
        "group_ids": g[rows],
        "row_ids": store["row_ids"][rows],
    }
    swap_xy = {
        "X": x[rows],
        "Y": y[partners],
        "group_ids": g[rows],
        "row_ids": store["row_ids"][rows],
    }
    res_c = fit_cell(f"{slug}__{model_kind}__swapctrl_correct", slug, correct_xy, cell_args)
    res_s = fit_cell(f"{slug}__{model_kind}__swap", slug, swap_xy, cell_args)
    if res_c["skipped"] or res_s["skipped"]:
        raise RuntimeError(
            f"swap {slug}/{model_kind}: component cells resume-skipped with no valid "
            "swap payload — the resume bypass failed (fail-loud, never a silent miss)"
        )
    hl = res_c["headline_layer"]
    sc, ss = res_c["sweep"], res_s["sweep"]
    if hl not in sc["preds_frozen"] or hl not in ss["preds_frozen"]:
        return None
    fitted = sc["fitted_mask"] & ss["fitted_mask"]
    pred_c = sc["preds_frozen"][hl][fitted]
    true_c = correct_xy["Y"][fitted, hl, :].astype(np.float64)
    pred_s = ss["preds_frozen"][hl][fitted]
    true_s = swap_xy["Y"][fitted, hl, :].astype(np.float64)
    groups = np.asarray(correct_xy["group_ids"])[fitted]
    gb_c = fit931.group_bootstrap_r2(pred_c, true_c, groups, n_boot=args.n_boot, seed=args.seed)
    gb_s = fit931.group_bootstrap_r2(
        pred_s,
        true_s,
        groups,
        n_boot=args.n_boot,
        seed=args.seed,
        draws_matrix=gb_c["draws_matrix"],
    )
    delta_draws = gb_c["draws"] - gb_s["draws"]
    payload = {
        "metadata": common.metadata(SCRIPT, args.seed, int(fitted.sum())),
        **r1335.fingerprint(slug),
        "rung": slug,
        "model_kind": model_kind,
        "headline_layer": hl,
        "r2_correct": gb_c["r2"],
        "r2_swap": gb_s["r2"],
        "delta_r2_char": gb_c["r2"] - gb_s["r2"],
        "delta_ci_lo": float(np.nanquantile(delta_draws, 0.025)),
        "delta_ci_hi": float(np.nanquantile(delta_draws, 0.975)),
        "n_rows": int(fitted.sum()),
        "n_groups": int(gb_c["n_groups"]),
        "n_boot": int(args.n_boot),
        "paired_group_bootstrap": True,
    }
    c1310.write_json(args.out_dir / f"swap_{slug}_{model_kind}.json", payload)
    return payload


def run_loso(slug: str, store: dict, model_kind: str, args) -> dict | None:
    """Leave-one-SETTING-out refit at L19 (group-level OOD read, plan §6)."""
    battery = c1310.build_scenario_battery()
    setting_of = {sc["scenario_id"]: sc["setting"] for sc in battery}
    if r1335.RUNGS[slug]["family"] == "qa":  # noqa: SIM108 — symmetric with run_rung_fits
        units = [("all", store)]
    else:
        units = rung_units(slug, store)
    out = {}
    for unit, ustore in units:
        settings = np.asarray([setting_of.get(g, "unknown") for g in ustore["group_ids"]])
        n_settings = len(np.unique(settings))
        if n_settings < 2:
            continue
        xy = unit_xy(ustore, "x_spanmean")
        n_layers = xy["X"].shape[1]
        fit825.FROZEN_LAYERS = tuple(frozen_layers(n_layers))
        sweep = fit825.heldout_r2_sweep(
            xy["X"],
            xy["Y"],
            settings,
            n_folds=n_settings,
            seed=args.seed,
            null_draws=0,
            lambda_selection=LAMBDA_SELECTION,
        )
        hl = headline_layer(n_layers)
        out[unit] = {
            "n": int(xy["X"].shape[0]),
            "n_settings": int(n_settings),
            "r2_headline_loso": float(sweep["r2_obs"][hl]),
        }
    if not out:
        return None
    payload = {
        "metadata": common.metadata(SCRIPT, args.seed, 0),
        **r1335.fingerprint(slug),
        "rung": slug,
        "model_kind": model_kind,
        "headline_layer": c1310.HEADLINE_LAYER,
        "per_unit": out,
    }
    c1310.write_json(args.out_dir / f"loso_{slug}_{model_kind}.json", payload)
    return payload


def run_rung_fits(args, slug: str, model_kind: str) -> None:
    """All full-n cells for one (rung, model): 3 arms (+r0 extras) per unit,
    swap on the swap rungs, LOSO on the LOSO rungs."""
    store = load_rung_store(args, slug, model_kind)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_layers = int(store["arrays"]["y"].shape[1])
    if args.smoke:
        fit825.EXPECTED_LAYERS = n_layers
    print(f"[i1335-fit] {slug}/{model_kind}: rows={len(store['row_ids'])} layers={n_layers}")
    for unit, ustore in rung_units(slug, store):
        for arm, x_key in ARM_X_KEY.items():
            cell_id = unit_cell_id(slug, model_kind, unit, arm)
            print(f"[i1335-fit] fit {cell_id} n={ustore['row_ids'].shape[0]}")
            fit_cell(cell_id, slug, unit_xy(ustore, x_key), args)
        if "y96" in ustore["arrays"]:
            fit_cell(
                unit_cell_id(slug, model_kind, unit, "ctx_y96"),
                slug,
                unit_xy(ustore, "x_spanmean", "y96"),
                args,
            )
        if "x_spanmean_nocap" in ustore["arrays"]:
            fit_cell(
                unit_cell_id(slug, model_kind, unit, "ctxnocap"),
                slug,
                unit_xy(ustore, "x_spanmean_nocap"),
                args,
            )
    if slug in SWAP_RUNGS:
        run_swap(slug, store, model_kind, args)
    if slug in LOSO_RUNGS:
        run_loso(slug, store, model_kind, args)


# ---------------------------------------------------------------------------
# Matched-n battery (across all rungs of a model set)
# ---------------------------------------------------------------------------


def _sidecars(args, slug: str, model_kind: str) -> list[Path]:
    return sorted((store_root(args) / slug / model_kind).glob(f"{model_kind}_shard*.json"))


def compute_n_min(args, models: list[str]) -> int:
    """Smallest realized unit n across all rungs/units/models, computed from the
    (always-local) shard SIDECARS — the .pt shards may already be uploaded +
    deleted under the per-cell lifecycle."""
    ns = []
    for model_kind in models:
        for slug in r1335.RUNG_ORDER:
            sidecars = _sidecars(args, slug, model_kind)
            if not sidecars:
                continue
            chars: list[str] = []
            for sc in sidecars:
                chars.extend(json.loads(sc.read_text())["char_ids"])
            if r1335.RUNGS[slug]["family"] == "qa":
                ns.append(len(chars))
            else:
                _uniq, counts = np.unique(np.asarray(chars), return_counts=True)
                ns.extend(int(c) for c in counts)
    assert ns, "no capture sidecars found for matched-n"
    return min(ns)


def ensure_store_local(args, slug: str, model_kind: str) -> bool:
    """Re-stage a (rung, model) store's .pt shards from the Hub when absent
    (the per-cell lifecycle uploads + deletes them). Pure hub-rel -> local-rel
    mapping (flat basenames under analysis_tensors/store_<slug>_<model>/);
    downloads via local_dir + os.replace so a later delete actually frees disk.
    Returns True when anything was downloaded."""
    store_dir = store_root(args) / slug / model_kind
    sidecars = _sidecars(args, slug, model_kind)
    assert sidecars, f"no sidecars for {slug}/{model_kind} — capture never ran"
    missing = [
        f"{sc.name[: -len('.json')]}.pt"
        for sc in sidecars
        if not (store_dir / f"{sc.name[: -len('.json')]}.pt").exists()
    ]
    if not missing:
        return False
    from explore_persona_space.orchestrate import hub

    prefix = f"{r1335.HF_PREFIX}/analysis_tensors/store_{slug}_{model_kind}"
    store_dir.mkdir(parents=True, exist_ok=True)
    # Canonical retried + atomic staging (#1402 hub.stage_hub_file): rides
    # hub.retry_transient — a raw un-retried hf_hub_download here let one
    # transient HF 429 ("maximum queue size reached") kill attempt
    # att-20260717-191703 mid-fit. The helper keeps the tempdir-inside-dest
    # os.replace publish (the #1335 EXDEV gotcha).
    for name in missing:
        hub.stage_hub_file(
            r1335.HF_DATA_REPO, f"{prefix}/{name}", store_dir / name, repo_type="dataset"
        )
    print(f"[i1335-fit] re-staged {len(missing)} shards for {slug}/{model_kind} from the Hub")
    return True


def release_store_local(args, slug: str, model_kind: str) -> None:
    """Delete a store's local .pt shards (sidecars kept). Callers guarantee the
    shards are Hub-resident (the per-cell lifecycle verified upload)."""
    for pt in (store_root(args) / slug / model_kind).glob(f"{model_kind}_shard*.pt"):
        pt.unlink()


def run_matched(args, models: list[str]) -> None:
    n_min = compute_n_min(args, models)
    print(f"[i1335-fit] matched-n battery: n_min={n_min}")
    for model_kind in models:
        for slug in r1335.RUNG_ORDER:
            if not _sidecars(args, slug, model_kind):
                print(f"[i1335-fit] matched-n: no capture for {slug}/{model_kind} — skipped")
                continue
            staged = ensure_store_local(args, slug, model_kind) if args.stage_from_hub else False
            store = load_rung_store(args, slug, model_kind)
            for unit, ustore in rung_units(slug, store):
                for arm in MATCHED_ARMS:
                    cell_id = unit_cell_id(slug, model_kind, unit, arm)
                    fit_cell_matched(cell_id, slug, unit_xy(ustore, ARM_X_KEY[arm]), n_min, args)
            if staged:
                release_store_local(args, slug, model_kind)
    cfg = {
        "metadata": common.metadata(SCRIPT, args.seed, n_min),
        "n_min": n_min,
        "n_draws": N_MATCHED_DRAWS,
        "seed_base": MATCHED_SEED_BASE,
        "arms": list(MATCHED_ARMS),
    }
    # Per-process tmp: the two sharded matched-n lanes both write this file at
    # battery end (same content — shared n_min); c1310.write_json's SHARED .tmp
    # could race a same-instant finish (round-1 review Minor 3).
    cfg_path = args.out_dir / "matched_n_config.json"
    tmp = args.out_dir / f".matched_n_config.{os.getpid()}.json.tmp"
    tmp.write_text(json.dumps(cfg, indent=2, default=float))
    tmp.replace(cfg_path)
    print(f"[i931] wrote {cfg_path}")


# ---------------------------------------------------------------------------
# Ladder summary: deltas, G, Δ_max, D, verdict lattice, gates (plan §3/§5/§7)
# ---------------------------------------------------------------------------


def _matched_value(args, cell_id: str) -> dict | None:
    p = args.out_dir / f"matched_{cell_id}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    draws = []
    for dr in d["draws"]:
        gb = dr.get("group_bootstrap_l19")
        if gb:
            draws.extend(gb["draws"])
    return {
        "value": d["r2_headline_mean"],
        "per_draw": d["r2_headline_per_draw"],
        "boot_draws": np.asarray(draws, dtype=float) if draws else None,
        "n_min": d["n_min"],
    }


def _fiction_mean(args, slug: str, model_kind: str, arm: str, kept: list[str]) -> dict | None:
    """Per-persona mean of a fiction rung's matched-n values (kept personas)."""
    vals, draw_stacks = [], []
    for persona in kept:
        mv = _matched_value(args, unit_cell_id(slug, model_kind, persona, arm))
        if mv is None:
            return None
        vals.append(mv["value"])
        if mv["boot_draws"] is not None:
            draw_stacks.append(mv["boot_draws"])
    boot = None
    if draw_stacks and len({len(v) for v in draw_stacks}) == 1:
        boot = np.mean(np.stack(draw_stacks, axis=0), axis=0)
    return {
        "value": float(np.mean(vals)),
        "boot_draws": boot,
        "per_persona": dict(zip(kept, vals, strict=True)),
    }


def _delta(a: dict | None, b: dict | None) -> dict | None:
    """Oriented delta a - b (strong - weak) with joint-draw CI when both sides
    carry equal-length draw arrays (independent-cell index pairing is a valid
    Monte-Carlo of the joint; shared-row cells with an identical group universe
    + seed align exactly), variance-sum fallback otherwise."""
    if a is None or b is None:
        return None
    val = a["value"] - b["value"]
    da, db = a.get("boot_draws"), b.get("boot_draws")
    if da is not None and db is not None and len(da) == len(db):
        dd = da - db
        return {
            "value": float(val),
            "ci_lo": float(np.nanquantile(dd, 0.025)),
            "ci_hi": float(np.nanquantile(dd, 0.975)),
            "ci_method": "joint-draws",
            "draws_len": len(dd),
        }
    parts = []
    for side in (da, db):
        parts.append(float(np.nanvar(side)) if side is not None else float("nan"))
    se = float(np.sqrt(np.nansum(parts)))
    return {
        "value": float(val),
        "ci_lo": float(val - 1.96 * se),
        "ci_hi": float(val + 1.96 * se),
        "ci_method": "variance-sum",
    }


def _delta_draws(a: dict | None, b: dict | None) -> np.ndarray | None:
    if a is None or b is None:
        return None
    da, db = a.get("boot_draws"), b.get("boot_draws")
    if da is None or db is None or len(da) != len(db):
        return None
    return da - db


def fiction_kept_personas(args, model_kind: str, smoke: bool = False) -> tuple[list[str], dict]:
    """Yield-floor report: personas at/above 80% of the v3 realized floor.

    A below-floor persona is REPORTED and kept out of the per-persona mean with
    the denominator revised (plan §4.2); smoke slices use floor=1 so the delta
    path is exercised end-to-end."""
    floor = 1 if smoke else FICTION_YIELD_FLOOR
    kept, report = [], {}
    for persona in c1310.PERSONA_LABELS:
        p = args.out_dir / f"cells_{unit_cell_id('r7_endpoint', model_kind, persona, 'ctx')}.json"
        if not p.exists():
            report[persona] = {"n": None, "kept": False, "reason": "missing"}
            continue
        n = json.loads(p.read_text())["n"]
        ok = n >= floor
        report[persona] = {"n": n, "kept": bool(ok), "floor": floor}
        if ok:
            kept.append(persona)
    return kept, report


def evaluate_gates(args, models: list[str], smoke: bool) -> dict:
    """The two binding rig-anchor gates (plan §7, AMENDED v4).

    Gate 1' (fiction endpoint): (i) SIGN — every realized per-persona r7 L19
    ctx full-n R^2 positive (binding); (ii) SWAP SPECIFICITY — only an
    INVERSION halts (correct-pairing <= 0, or delta_char < 0 with its paired
    group-bootstrap CI wholly below 0; a weak-positive / CI-straddling delta
    is report-only); (iii) ROUND-8 REPRODUCTION — the validated inner-group-CV
    values on the same persisted stores, +-R8_REPRO_TOL (binding on the three
    validated cells). The v3 +-0.08 band vs the #1310 anchors is RETIRED
    (cross-selector-incomparable) and reported descriptively.

    Gate 2' (Q&A endpoint): (i) wiring check binding (unchanged, incl. the r7
    skipped-seeded pass-with-record); (ii) r0 base ctx L19 reproduces the
    round-8 validated +0.4103 within +-R8_REPRO_TOL (selector-agnostic). The
    v3 [0.55, 0.90] bracket is DEMOTED to a report-not-halt diagnostic; the
    0.41-vs-bracket discrepancy is a first-class clean-result finding.
    """
    gates: dict = {"binding": not smoke, "lambda_selection": LAMBDA_SELECTION, "amended": "v4"}
    g1 = {"r8_repro_tol": R8_REPRO_TOL, "v3_band_descriptive": ANCHOR_TOL, "per_model": {}}
    g1_pass = True
    for model_kind in models:
        anchors = V3_ANCHORS_BASE if model_kind == "base" else V3_ANCHORS_INSTRUCT
        per = {}
        for persona, ref in anchors.items():
            p = args.out_dir / (
                f"cells_{unit_cell_id('r7_endpoint', model_kind, persona, 'ctx')}.json"
            )
            if not p.exists():
                per[persona] = {
                    "r2": None,
                    "v3_ref_descriptive": ref,
                    "pass": False,
                    "reason": "missing",
                }
                g1_pass = False
                continue
            d = json.loads(p.read_text())
            r2 = d["r2_per_layer_obs"][d["headline_layer"]]
            entry = {
                "r2": r2,
                "v3_ref_descriptive": ref,
                "sign_pass": bool(r2 > 0),  # component (i)
                "pass": bool(r2 > 0),
            }
            repro_ref = R8_REPRO_R7.get((model_kind, persona))
            if repro_ref is not None:  # component (iii) — the validated cells
                entry["r8_repro_ref"] = repro_ref
                entry["r8_repro_pass"] = bool(abs(r2 - repro_ref) <= R8_REPRO_TOL)
                entry["pass"] = bool(entry["sign_pass"] and entry["r8_repro_pass"])
            per[persona] = entry
            g1_pass = g1_pass and entry["pass"]
        swap_p = args.out_dir / f"swap_r7_endpoint_{model_kind}.json"
        if swap_p.exists():
            sw = json.loads(swap_p.read_text())
            # Component (ii): only INVERSION halts (amended 1'.ii) — a
            # CI-straddling / weak-positive delta is report-only.
            ci_hi = sw.get("delta_ci_hi")
            inversion = sw["r2_correct"] <= 0 or (
                sw["delta_r2_char"] < 0 and ci_hi is not None and ci_hi < 0
            )
            per["_swap"] = {
                "delta_r2_char": sw["delta_r2_char"],
                "r2_correct": sw["r2_correct"],
                "delta_ci_lo": sw.get("delta_ci_lo"),
                "delta_ci_hi": ci_hi,
                "pass": bool(not inversion),
            }
            g1_pass = g1_pass and not inversion
        else:
            per["_swap"] = {"pass": False, "reason": "missing"}
            g1_pass = False
        g1["per_model"][model_kind] = per
    g1["pass"] = bool(g1_pass)
    gates["gate1_fiction_anchor"] = g1
    # Gate 2': r0 round-8 reproduction (binding) + wiring checks (binding);
    # the v3 range is recorded as a report-not-halt diagnostic.
    g2 = {
        "r8_repro_ref": R8_REPRO_R0_BASE,
        "r8_repro_tol": R8_REPRO_TOL,
        "v3_range_descriptive": list(R0_GATE_RANGE),
    }
    p = args.out_dir / "cells_r0_qa_full__base__ctx.json"
    if p.exists():
        d = json.loads(p.read_text())
        r2 = d["r2_per_layer_obs"][d["headline_layer"]]
        g2["r0_base_r2"] = r2
        g2["r0_in_v3_range_descriptive"] = bool(R0_GATE_RANGE[0] <= r2 <= R0_GATE_RANGE[1])
        g2["r0_repro_pass"] = bool(abs(r2 - R8_REPRO_R0_BASE) <= R8_REPRO_TOL)
    else:
        g2["r0_base_r2"] = None
        g2["r0_repro_pass"] = False
    wiring = {}
    for wp in sorted(args.out_dir.glob("wiring_*.json")):
        w = json.loads(wp.read_text())
        if w.get("wiring_check") == "skipped-seeded":
            # r6: seed-consumed cell — wiring skipped at capture; the seeded
            # rows carried validation in their original attempt. Recorded but
            # NON-BINDING for the gate (excluded from the all() below).
            wiring[wp.stem] = {
                "wiring_check": "skipped-seeded",
                "fresh_rows": w.get("fresh_rows"),
                "seeded_rows": w.get("seeded_rows"),
            }
            continue
        wiring[wp.stem] = {
            "own_beats_shuffled": w["own_beats_shuffled"],
            "delta": w["delta"],
        }
    g2["wiring"] = wiring
    ran = [v for v in wiring.values() if "own_beats_shuffled" in v]
    skipped = [v for v in wiring.values() if v.get("wiring_check") == "skipped-seeded"]
    if ran:
        g2["wiring_pass"] = all(v["own_beats_shuffled"] for v in ran)
    elif skipped:
        # r7 (concern fully-seeded-relaunch-gate2-halt): EVERY wiring cell was
        # seed-consumed — a relaunch after a post-P2-complete crash. Seed
        # consumption REQUIRES the render-config fingerprint match (the c24
        # CONSUME rule in extract), and the consumed shards' original attempts
        # ran the binding wiring checks — so gate2 passes on that provenance
        # instead of halting a legitimately complete relaunch (exit 3).
        g2["wiring_pass"] = True
        g2["wiring_basis"] = (
            "all-seeded (original-attempt validation via fingerprint-matched consume)"
        )
        print(
            "[i1335-fit] gate2 wiring: ALL cells skipped-seeded -> pass-with-record "
            "(original-attempt validation via fingerprint-matched consume)"
        )
    else:
        # No wiring files at all: the wiring cells never ran (r7 guarantees a
        # file per wiring cell, skip or ran) -> conservative halt stands.
        g2["wiring_pass"] = False
    g2["pass"] = bool(g2["r0_repro_pass"] and g2["wiring_pass"])
    gates["gate2_qa_endpoint"] = g2
    return gates


def build_ladder_summary(args, models: list[str], smoke: bool) -> dict:
    per_model: dict = {}
    for model_kind in models:
        kept, yield_report = fiction_kept_personas(args, model_kind, smoke)

        def mv(slug, unit="all", arm="ctx", model_kind=model_kind):
            return _matched_value(args, unit_cell_id(slug, model_kind, unit, arm))

        r0 = mv("r0_qa_full")
        r1 = mv("r1_qa_oneline")
        r2tf = mv("r2_tf")
        r2op = mv("r2_op")
        r3 = mv("r3_persona")
        r4 = mv("r4_fictionframe")
        r6m = _fiction_mean(args, "r6_nofoil", model_kind, "ctx", kept) if kept else None
        r7m = _fiction_mean(args, "r7_endpoint", model_kind, "ctx", kept) if kept else None
        s1m = _fiction_mean(args, "s1_assistant_label", model_kind, "ctx", kept) if kept else None
        s2am = _fiction_mean(args, "s2a_familiar", model_kind, "ctx", kept) if kept else None
        s2bm = _fiction_mean(args, "s2b_novel", model_kind, "ctx", kept) if kept else None
        r7_wren = _matched_value(args, unit_cell_id("r7_endpoint", model_kind, "Wren", "ctx"))

        # Oriented deltas (strong - weak; §5 delta map). Negative realized
        # values are restorations/anti-drops — reported, never fed to Δ_max.
        deltas = {
            "label": _delta(r1, r2tf),
            "label_op_companion": _delta(r1, r2op),
            "header": _delta(r2op, r3),
            "framing": _delta(r3, r4),
            "content_depth": _delta(r4, r7m),
            "content_depth_wren_matched": _delta(r4, r7_wren),
            "foils": _delta(r6m, r7m),
            "label_restore": _delta(s1m, r7m),
            "name_frequency_sub": _delta(s2am, s2bm),
            # OUTSIDE the 6-family: length, read against the r0-referenced gap.
            "length": _delta(r0, r1),
        }
        gap = {
            "G": _delta(r1, r7m),  # matched-answer-length gap (r1 reference)
            "full_gap_r0": _delta(r0, r7m),
        }
        # Δ_max over the SIX-delta family; Bonferroni-corrected CI (alpha/6).
        family_vals = {k: deltas[k]["value"] for k in DELTA_FAMILY if deltas.get(k) is not None}
        # Plan §3: a negative realized Δ_f is a restoration/anti-drop — reported
        # (raw values stay in `deltas`/`family_values`) but NEVER fed to the max.
        max_candidates = {k: v for k, v in family_vals.items() if v >= 0.0}
        excluded_negative = sorted(k for k in family_vals if k not in max_candidates)
        dmax = None
        if max_candidates:
            dmax_key = max(max_candidates, key=max_candidates.get)
            alpha = 0.05 / len(DELTA_FAMILY)
            pair = {
                "label": (r1, r2tf),
                "header": (r2op, r3),
                "framing": (r3, r4),
                "content_depth": (r4, r7m),
                "foils": (r6m, r7m),
                "label_restore": (s1m, r7m),
            }[dmax_key]
            dd = _delta_draws(*pair)
            if dd is not None:
                ci = (
                    float(np.nanquantile(dd, alpha / 2)),
                    float(np.nanquantile(dd, 1 - alpha / 2)),
                )
            else:
                base = deltas[dmax_key]
                half = (base["ci_hi"] - base["ci_lo"]) / 2 / 1.96
                from scipy.stats import norm  # local import; scipy in the env

                z = float(norm.ppf(1 - alpha / 2))
                ci = (base["value"] - z * half, base["value"] + z * half)
            dmax = {
                "delta": dmax_key,
                "value": family_vals[dmax_key],
                "ci_lo_bonferroni": ci[0],
                "ci_hi_bonferroni": ci[1],
                "family": list(DELTA_FAMILY),
                "family_values": {k: float(v) for k, v in family_vals.items()},
                "excluded_negative_deltas": excluded_negative,
                "note": (
                    "foils is a sub-delta of content_depth (family overlap; narrated as a "
                    "component when content_depth is the max, never double-counted); "
                    "negative realized deltas are reported but never fed to the max (§3)"
                ),
            }
        # D = Δ_max - 0.5·G, joint-draw CI where computable.
        D = None
        if dmax is not None and gap["G"] is not None:
            d_val = dmax["value"] - 0.5 * gap["G"]["value"]
            pair = {
                "label": (r1, r2tf),
                "header": (r2op, r3),
                "framing": (r3, r4),
                "content_depth": (r4, r7m),
                "foils": (r6m, r7m),
                "label_restore": (s1m, r7m),
            }[dmax["delta"]]
            dd = _delta_draws(*pair)
            gd = _delta_draws(r1, r7m)
            if dd is not None and gd is not None and len(dd) == len(gd):
                joint = dd - 0.5 * gd
                D = {
                    "value": float(d_val),
                    "ci_lo": float(np.nanquantile(joint, 0.025)),
                    "ci_hi": float(np.nanquantile(joint, 0.975)),
                    "ci_method": "joint-draws",
                }
            else:
                se_d = (dmax["ci_hi_bonferroni"] - dmax["ci_lo_bonferroni"]) / 2 / 2.64
                se_g = (gap["G"]["ci_hi"] - gap["G"]["ci_lo"]) / 2 / 1.96
                se = float(np.sqrt(se_d**2 + (0.5 * se_g) ** 2))
                D = {
                    "value": float(d_val),
                    "ci_lo": float(d_val - 1.96 * se),
                    "ci_hi": float(d_val + 1.96 * se),
                    "ci_method": "variance-sum",
                }
        # Verdict lattice (§3; DISJOINT + exhaustive).
        verdict = "Inconclusive"
        if gap["G"] is not None:
            g_ = gap["G"]
            if g_["ci_hi"] < 0 or (g_["ci_lo"] <= 0 <= g_["ci_hi"]):
                verdict = "Sample-size-explained"
            elif g_["ci_lo"] > 0 and D is not None and D["value"] > 0 and D["ci_lo"] > 0:
                verdict = "Single-factor-attributed"
            elif g_["ci_lo"] > 0 and D is not None and D["ci_hi"] < 0:
                verdict = "Distributed-attribution"
        per_model[model_kind] = {
            "fiction_yield": yield_report,
            "kept_personas": kept,
            "rung_values_matched_ctx": {
                "r0_qa_full": r0 and r0["value"],
                "r1_qa_oneline": r1 and r1["value"],
                "r2_tf": r2tf and r2tf["value"],
                "r2_op": r2op and r2op["value"],
                "r3_persona": r3 and r3["value"],
                "r4_fictionframe": r4 and r4["value"],
                "r6_nofoil_mean": r6m and r6m["value"],
                "r7_endpoint_mean": r7m and r7m["value"],
                "r7_endpoint_per_persona": r7m and r7m["per_persona"],
                "s1_assistant_label_mean": s1m and s1m["value"],
                "s2a_familiar_mean": s2am and s2am["value"],
                "s2b_novel_mean": s2bm and s2bm["value"],
            },
            "deltas": deltas,
            "gap": gap,
            "delta_max": dmax,
            "D": D,
            "verdict": verdict,
        }
    gates = evaluate_gates(args, models, smoke)
    summary = {
        "metadata": common.metadata(SCRIPT, args.seed, 0),
        "code_sha": common.git_commit(),
        "render_config_hashes": {s: r1335.render_config_hash(s) for s in r1335.RUNG_ORDER},
        "headline_layer": c1310.HEADLINE_LAYER,
        "delta_family": list(DELTA_FAMILY),
        "length_delta_outside_family": True,
        "per_model": per_model,
        "gates": gates,
        "smoke": bool(smoke),
    }
    c1310.write_json(args.out_dir / "ladder_summary.json", summary)
    print(
        f"[i1335-fit] ladder_summary.json written (verdicts: "
        f"{ {m: per_model[m]['verdict'] for m in per_model} })"
    )
    return summary


def _cross_seed_delta(new: dict | None, ref: dict | None) -> dict | None:
    """new_seed - reference delta with a variance-sum 95% CI (the _delta
    fallback arithmetic applied across two INDEPENDENT runs — bootstrap draws
    are not pairable across seeds, so joint-draw pairing never applies here).
    Each side's SE is recovered from its own 95% CI half-width."""
    if new is None or ref is None:
        return None
    val = new["value"] - ref["value"]
    se_parts = []
    for side in (new, ref):
        se_parts.append(((side["ci_hi"] - side["ci_lo"]) / 2.0 / 1.96) ** 2)
    se = float(np.sqrt(sum(se_parts)))
    return {
        "value": float(val),
        "ci_lo": float(val - 1.96 * se),
        "ci_hi": float(val + 1.96 * se),
        "ci_method": "variance-sum-independent-runs",
        "new_in_ref_ci": bool(ref["ci_lo"] <= new["value"] <= ref["ci_hi"]),
        "ref_in_new_ci": bool(new["ci_lo"] <= ref["value"] <= new["ci_hi"]),
    }


def _ref_headline(ref: dict, ref_label: str, model_kind: str) -> tuple[dict, dict]:
    """Reference (gap, framing) delta dicts — shape-tolerant across the TWO
    committed reference kinds: a ladder_summary.json (per_model.<mk>.gap.G +
    .deltas.framing) or a prior round's seed_comparison.json
    (per_model.<mk>.gap_G + .framing). Fail-loud on a reference carrying
    NEITHER shape for the requested model — a model absent from the reference
    is a caller scope error, never silently tolerated (the 4fc950e83e
    fail-loud-reference-fields discipline, extended to both shapes)."""
    ref_m = ref.get("per_model", {}).get(model_kind) or {}
    gap = (ref_m.get("gap") or {}).get("G") or ref_m.get("gap_G")
    framing = (ref_m.get("deltas") or {}).get("framing") or ref_m.get("framing")
    assert gap is not None, (
        f"{ref_label}: reference summary lacks per_model.{model_kind} gap "
        "(neither gap.G nor gap_G present)"
    )
    assert framing is not None, (
        f"{ref_label}: reference summary lacks per_model.{model_kind} framing "
        "(neither deltas.framing nor framing present)"
    )
    return gap, framing


def collapse_audit(args, model_kind: str, slug: str = "r7_endpoint") -> dict:
    """Rollout collapse audit on THIS run's fiction rollouts: under-floor line
    counts (n_completion_tokens < DIALOGUE_MIN_TOKENS) total / per-slot /
    per-persona, plus FULL-SLOT exact-modal-line counts (the top repeated
    exact completion per slot and its count — the "I agree."-class collapse
    detector at ANY slot: the seed-42 mode hit slot 4, the seed-44 mode hit
    slot 2, so a fixed-slot field demonstrably misses migrating modes; plan
    v7 §4.2 item 4b). The legacy slot-4 exact-"I agree." fields are kept
    verbatim (the committed seed_comparison consumers read them). Fail-loud
    on a missing rollout file — audit rounds always generate their fiction
    rungs, so absence is a pipeline bug, never skipped."""
    from collections import Counter

    path = r1335.gen_path(args.data_dir, slug, model_kind)
    assert path.exists(), (
        f"collapse audit: missing {slug} rollouts {path} — this round's own "
        "fiction generation must have run (fail-loud)"
    )
    total = under = slot4_total = slot4_agree = 0
    per_slot: dict[str, int] = {}
    per_persona: dict[str, int] = {}
    agree_per_persona: dict[str, int] = {}
    slot_totals: dict[str, int] = {}
    slot_lines: dict[str, Counter] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            slot = int(row.get("slot", 0))
            persona = str(row.get("persona", "?"))
            key = f"slot{slot}"
            slot_totals[key] = slot_totals.get(key, 0) + 1
            slot_lines.setdefault(key, Counter())[(row.get("completion") or "").strip()] += 1
            if int(row["n_completion_tokens"]) < r1335.DIALOGUE_MIN_TOKENS:
                under += 1
                per_slot[key] = per_slot.get(key, 0) + 1
                per_persona[persona] = per_persona.get(persona, 0) + 1
            if slot == 4:
                slot4_total += 1
                if (row.get("completion") or "").strip() == "I agree.":
                    slot4_agree += 1
                    agree_per_persona[persona] = agree_per_persona.get(persona, 0) + 1
    assert total > 0, f"collapse audit: empty rollout file {path}"
    modal_per_slot = {}
    for key in sorted(slot_lines):
        text, count = slot_lines[key].most_common(1)[0]
        modal_per_slot[key] = {
            "line": text[:80],
            "count": int(count),
            "slot_lines": int(slot_totals[key]),
            "pct": round(100.0 * count / slot_totals[key], 2),
        }
    return {
        "rung": slug,
        "rollouts": str(path),
        "dialogue_min_tokens": int(r1335.DIALOGUE_MIN_TOKENS),
        "n_lines": total,
        "under_floor_lines": under,
        "under_floor_pct": round(100.0 * under / total, 2),
        "under_floor_per_slot": dict(sorted(per_slot.items())),
        "under_floor_per_persona": dict(sorted(per_persona.items())),
        "modal_line_per_slot": modal_per_slot,
        "slot4_lines": slot4_total,
        "slot4_exact_agree": slot4_agree,
        "slot4_exact_agree_per_persona": dict(sorted(agree_per_persona.items())),
    }


def build_seed_comparison(args, models: list[str], smoke: bool) -> dict:
    """Seed-replication follow-up read (one variable: the generation seed).

    For each model, recompute the two headline quantities from THIS run's
    matched-n cells — the matched-answer-length gap G = Δ(r1_qa_oneline,
    r7_endpoint per-persona mean) and the framing delta Δ(r3_persona,
    r4_fictionframe), both L19 ctx, joint-draw CIs (the exact
    build_ladder_summary pairing) — and compare against the committed
    reference(s): the primary --reference-summary (seed-42 ladder_summary)
    plus an optional --reference-summary-2 (e.g. the committed seed-43
    seed_comparison; either reference shape accepted, see _ref_headline).
    Also runs the rollout collapse audit on this run's own endpoint rollouts.
    Writes <out-dir>/seed_comparison.json. Reuses _matched_value /
    _fiction_mean / _delta verbatim; no new statistical machinery beyond the
    independent-runs variance-sum cross-seed CI."""
    ref_path = args.reference_summary
    assert ref_path.exists(), (
        f"--seed-compare reference summary missing: {ref_path} — the committed "
        "seed-42 ladder_summary.json is required (fail-loud, never a silent "
        "no-reference comparison)"
    )
    ref = json.loads(ref_path.read_text())
    ref2_path = getattr(args, "reference_summary_2", None)
    ref2 = None
    if ref2_path is not None:
        assert ref2_path.exists(), (
            f"--seed-compare second reference missing: {ref2_path} (fail-loud, "
            "never a silent single-reference comparison when two were requested)"
        )
        ref2 = json.loads(ref2_path.read_text())
    per_model: dict = {}
    for model_kind in models:
        kept, yield_report = fiction_kept_personas(args, model_kind, smoke)
        r1 = _matched_value(args, unit_cell_id("r1_qa_oneline", model_kind, "all", "ctx"))
        r3 = _matched_value(args, unit_cell_id("r3_persona", model_kind, "all", "ctx"))
        r4 = _matched_value(args, unit_cell_id("r4_fictionframe", model_kind, "all", "ctx"))
        r7m = _fiction_mean(args, "r7_endpoint", model_kind, "ctx", kept) if kept else None
        gap = _delta(r1, r7m)
        framing = _delta(r3, r4)
        ref_gap, ref_framing = _ref_headline(ref, "reference", model_kind)
        assert kept, f"seed-compare: no kept fiction personas for {model_kind} (below floor)"
        entry = {
            "kept_personas": kept,
            "fiction_yield": yield_report,
            "collapse_audit": collapse_audit(args, model_kind),
            "rung_values_matched_ctx": {
                "r1_qa_oneline": r1 and r1["value"],
                "r3_persona": r3 and r3["value"],
                "r4_fictionframe": r4 and r4["value"],
                "r7_endpoint_mean": r7m and r7m["value"],
                "r7_endpoint_per_persona": r7m and r7m["per_persona"],
            },
            "gap_G": gap,
            "framing": framing,
            "seed42_reference": {"gap_G": ref_gap, "framing": ref_framing},
            "cross_seed": {
                "gap_G": _cross_seed_delta(gap, ref_gap),
                "framing": _cross_seed_delta(framing, ref_framing),
            },
        }
        if ref2 is not None:
            ref2_gap, ref2_framing = _ref_headline(ref2, "reference-2", model_kind)
            entry["reference_2"] = {"gap_G": ref2_gap, "framing": ref2_framing}
            entry["cross_seed_2"] = {
                "gap_G": _cross_seed_delta(gap, ref2_gap),
                "framing": _cross_seed_delta(framing, ref2_framing),
            }
        per_model[model_kind] = entry
    out = {
        "metadata": common.metadata(SCRIPT, args.seed, 0),
        "code_sha": common.git_commit(),
        "gen_seed": r1335.GEN_SEED,
        "headline_layer": c1310.HEADLINE_LAYER,
        "models_compared": list(models),
        "reference": {
            "path": str(ref_path),
            "code_sha": ref.get("code_sha"),
            "note": "committed seed-42 run (GEN_SEED 42, the #825/#1310 convention)",
        },
        "per_model": per_model,
        "smoke": bool(smoke),
    }
    if sorted(models) != sorted(c1310.MODEL_KINDS):
        out["scope_note"] = (
            "declared model-subset round: only models_compared were run this round "
            "(absent models are absent-because-not-run, not missing data)"
        )
    if ref2 is not None:
        out["reference_2"] = {
            "path": str(ref2_path),
            "code_sha": ref2.get("code_sha"),
            "gen_seed": ref2.get("gen_seed"),
            "note": "second committed reference (dual-reference seed compare)",
        }
    c1310.write_json(args.out_dir / "seed_comparison.json", out)
    print(
        "[i1335-fit] seed_comparison.json written "
        f"(gen_seed={r1335.GEN_SEED}; models={list(per_model)}; "
        f"references={1 + (ref2 is not None)})"
    )
    return out


# ---------------------------------------------------------------------------
# Label comparison (onpolicy-assistant-label round; plan v7 §3/§4.2 item 4)
# ---------------------------------------------------------------------------


def _full_n_value(args, cell_id: str) -> dict:
    """Full-n L19 value + the persisted group-bootstrap draws for one cell (the
    joint-draw pairing surface for the within-run deltas). Fail-loud on a
    missing cell or missing draws — label-compare runs AFTER this round's own
    P3 fits produced every registered cell."""
    p = args.out_dir / f"cells_{cell_id}.json"
    assert p.exists(), f"label-compare: missing full-n cell {p} — the P3 fits must run first"
    d = json.loads(p.read_text())
    gb = d.get("group_bootstrap_l19")
    assert gb is not None and gb.get("draws"), (
        f"label-compare: {p} lacks persisted group_bootstrap_l19 draws (the "
        "joint-draw pairing surface; fail-loud)"
    )
    return {
        "value": float(gb["r2"]),
        "r2_cell_headline": float(d["r2_per_layer_obs"][d["headline_layer"]]),
        "boot_draws": np.asarray(gb["draws"], dtype=float),
        "ci_lo": float(gb["ci_lo"]),
        "ci_hi": float(gb["ci_hi"]),
        "n": int(d["n"]),
        "n_groups": int(d["n_groups"]),
        "group_universe_hash": gb.get("group_universe_hash"),
    }


def label_h0_pair_noise_band(committed_root: Path) -> dict:
    """Registered empirical H0 pair-noise band (plan v7 §3, Statistics-critic
    fix (a)): load the 12 committed base endpoint L19 ctx cell values
    (4 personas x seeds 42/43/44), remove run + persona effects by the two-way
    decomposition r_ij = x_ij - rowmean_i - colmean_j + grandmean, take
    sigma_cell = interaction residual SD (df = (I-1)(J-1) = 6), and register
    B_hat = 2*sqrt(2)*sigma_cell (= 2*sigma_pair). Read-only on the committed
    eval JSONs; fail-loud on any missing cell — the 12-cell lattice is a
    committed artifact. Mechanical sanity floor: B_hat >= sqrt(2)*sigma_cell."""
    personas = list(c1310.PERSONA_LABELS)
    seed_labels = [s for s, _ in H0_SEED_DIRS]
    values = np.full((len(personas), len(seed_labels)), np.nan)
    sources: dict[str, str] = {}
    for j, (seed_label, rel) in enumerate(H0_SEED_DIRS):
        root = committed_root if rel == "." else committed_root / rel
        for i, persona in enumerate(personas):
            p = root / f"cells_{unit_cell_id('r7_endpoint', 'base', persona, 'ctx')}.json"
            assert p.exists(), f"H0 band: missing committed base endpoint cell {p} (fail-loud)"
            d = json.loads(p.read_text())
            values[i, j] = float(d["r2_per_layer_obs"][d["headline_layer"]])
            sources[f"{seed_label}:{persona}"] = str(p)
    assert np.isfinite(values).all(), "H0 band: non-finite committed cell value"
    grand = float(values.mean())
    row_means = values.mean(axis=1, keepdims=True)
    col_means = values.mean(axis=0, keepdims=True)
    resid = values - row_means - col_means + grand
    dof = (values.shape[0] - 1) * (values.shape[1] - 1)
    sigma_cell = float(np.sqrt(float((resid**2).sum()) / dof))
    sigma_pair = float(np.sqrt(2.0) * sigma_cell)
    b_hat = float(2.0 * sigma_pair)
    assert b_hat >= np.sqrt(2.0) * sigma_cell, (
        f"H0 band mechanical sanity floor violated: B_hat={b_hat} < "
        f"sqrt(2)*sigma_cell={np.sqrt(2.0) * sigma_cell} (plan v7 §4.2 item 4f)"
    )
    return {
        "definition": (
            "two-way decomposition of the 12 committed base r7_endpoint L19 ctx "
            "full-n R^2 values (4 personas x seeds 42/43/44); run + persona "
            "effects removed; sigma_cell = interaction residual SD (ddof via "
            "df=(I-1)(J-1)); B_hat = 2*sqrt(2)*sigma_cell"
        ),
        "personas": personas,
        "seeds": seed_labels,
        "values": {
            p: {s: float(values[i, j]) for j, s in enumerate(seed_labels)}
            for i, p in enumerate(personas)
        },
        "residuals": {
            p: {s: float(resid[i, j]) for j, s in enumerate(seed_labels)}
            for i, p in enumerate(personas)
        },
        "dof": int(dof),
        "sigma_cell": sigma_cell,
        "sigma_pair": sigma_pair,
        "b_hat": b_hat,
        "sources": sources,
    }


def _committed_placement_n(args, model_kind: str) -> int:
    """Per-model committed matched n for the placement read — sourced from the
    committed per-model matched JSON and cross-checked against the body value
    (plan v7 assumption 17; NEVER matched_n_config.json — that file holds one
    last-writer n_min across the two sharded lanes)."""
    p = args.committed_eval_root / (
        f"matched_{unit_cell_id('r7_endpoint', model_kind, 'Wren', 'ctx')}.json"
    )
    assert p.exists(), f"placement: committed matched JSON missing: {p} (fail-loud)"
    n = int(json.loads(p.read_text())["n_min"])
    expect = COMMITTED_PLACEMENT_N[model_kind]
    assert n == expect, (
        f"placement: committed matched n drift — {p} n_min={n} != body value {expect}"
    )
    return n


def _concat_stores(a: dict, b: dict) -> dict:
    """Row-concatenate two loaded (rung, model) stores (the combined
    Assistant+Wren store the cross-label swap derangement pairs over)."""
    keys = sorted(set(a["arrays"]) & set(b["arrays"]))
    assert keys, "combined store: no shared summary arrays"
    for k in keys:
        assert a["arrays"][k].shape[1:] == b["arrays"][k].shape[1:], (
            k,
            a["arrays"][k].shape,
            b["arrays"][k].shape,
        )
    return {
        "row_ids": np.concatenate([a["row_ids"], b["row_ids"]]),
        "group_ids": np.concatenate([a["group_ids"], b["group_ids"]]),
        "char_ids": np.concatenate([a["char_ids"], b["char_ids"]]),
        "turn_indices": np.concatenate([a["turn_indices"], b["turn_indices"]]),
        "arrays": {k: np.concatenate([a["arrays"][k], b["arrays"][k]], axis=0) for k in keys},
    }


def build_label_comparison(args, models: list[str], smoke: bool) -> dict:
    """onpolicy-assistant-label summary (plan v7 §4.2 item 4; peer of
    --seed-compare: no gates, no verdict lattice, no ladder figures).

    Per (model, arm in ctx/prefix): (a) the full-n within-run paired delta
    Delta_AW via the committed _delta joint-draw pairing over the shared
    scenario universe (any group-universe mismatch between the pair's cells is
    REPORTED — a fully-dropped scene still pairs by independent draw index);
    (b) the pairwise-matched-n paired delta (all three cells refit at the
    per-model 3-cell min n — the two registered pairs share the Wren45 cell,
    so one per-model matched n keeps both deltas same-n comparable; 5
    group-stratified draws, seeds 931+k, seed-mean — fit_cell_matched
    verbatim); (c) matched-n placement values at the committed per-model
    matched n (1,397 base / 1,739 instruct; pairwise-min fallback with a
    labeled non-comparability note when a realized n falls short) vs the
    committed read-only references; (d) the cross-label swap-specificity read
    on the combined Assistant+Wren(45) store (run_swap derangement across
    char_ids at matched scene-position); (e) full-slot collapse audits on all
    cells; (f) the registered empirical H0 pair-noise band from the 12
    committed base endpoint cells; (g) the within-run Wren45-Wren46 replicate
    delta (the direct generation-draw H0 pair). Writes label_comparison.json."""
    ref_path = args.reference_summary
    assert ref_path.exists(), (
        f"--label-compare reference summary missing: {ref_path} — the committed "
        "seed-42 ladder_summary.json is required (fail-loud)"
    )
    ref = json.loads(ref_path.read_text())
    h0_band = label_h0_pair_noise_band(args.committed_eval_root)
    floor = 1 if smoke else FICTION_YIELD_FLOOR
    per_model: dict = {}
    for model_kind in models:
        staged: list[str] = []
        stores: dict[str, dict] = {}
        units: dict[str, tuple[str, dict]] = {}
        for slug in LABEL_RUNGS:
            if args.stage_from_hub and ensure_store_local(args, slug, model_kind):
                staged.append(slug)
            store = load_rung_store(args, slug, model_kind)
            u = rung_units(slug, store)
            assert len(u) == 1, (
                f"label-compare: {slug}/{model_kind} expected exactly 1 lead unit, "
                f"got {[name for name, _ in u]}"
            )
            assert u[0][1]["row_ids"].shape[0] > 0, f"label-compare: empty unit {slug}"
            if smoke:
                fit825.EXPECTED_LAYERS = int(store["arrays"]["y"].shape[1])
            stores[slug] = store
            units[slug] = u[0]

        # (a)+(g) full-n paired deltas (joint persisted draws), ctx + prefix.
        full = {
            slug: {
                arm: _full_n_value(args, unit_cell_id(slug, model_kind, units[slug][0], arm))
                for arm in MATCHED_ARMS
            }
            for slug in LABEL_RUNGS
        }
        realized_n = {slug: full[slug]["ctx"]["n"] for slug in LABEL_RUNGS}
        deltas_full: dict = {}
        for name, slug_a, slug_b in LABEL_PAIRS:
            per_arm = {}
            for arm in MATCHED_ARMS:
                a, b = full[slug_a][arm], full[slug_b][arm]
                dd = _delta(a, b)
                dd.update(
                    {
                        "value_a": a["value"],
                        "value_b": b["value"],
                        "n_a": a["n"],
                        "n_b": b["n"],
                        "group_universe_match": bool(
                            a["group_universe_hash"] == b["group_universe_hash"]
                        ),
                    }
                )
                dd_draws = _delta_draws(a, b)
                if dd_draws is not None:  # <=100-draw strided sample (hero figure dots)
                    step = max(1, len(dd_draws) // 100)
                    dd["draws_sample"] = [float(v) for v in dd_draws[::step][:100]]
                per_arm[arm] = dd
            deltas_full[name] = per_arm

        # (b) pairwise-matched refits at the per-model 3-cell min n (one n for
        # both pairs — the shared Wren45 cell makes the pair mins coincide up
        # to the wren46 cell; a single n keeps the two deltas comparable and
        # each cell refits once per arm, the plan §9 6-cell arithmetic).
        n_pair = int(min(realized_n.values()))
        matched_vals: dict = {}
        for slug in LABEL_RUNGS:
            lead, ustore = units[slug]
            for arm in MATCHED_ARMS:
                cid = unit_cell_id(slug, model_kind, lead, arm) + "__pairwise"
                fit_cell_matched(cid, slug, unit_xy(ustore, ARM_X_KEY[arm]), n_pair, args)
                matched_vals[(slug, arm)] = _matched_value(args, cid)
        deltas_matched = {
            name: {
                arm: _delta(matched_vals[(slug_a, arm)], matched_vals[(slug_b, arm)])
                for arm in MATCHED_ARMS
            }
            for name, slug_a, slug_b in LABEL_PAIRS
        }

        # (c) placement subsamples (ctx — the committed anchors are ctx) at the
        # committed per-model matched n, pairwise-min fallback labeled.
        committed_n = _committed_placement_n(args, model_kind)
        placement_at_committed = all(n >= committed_n for n in realized_n.values())
        n_place = committed_n if placement_at_committed else n_pair
        placement_vals = {}
        for slug in LABEL_RUNGS:
            lead, ustore = units[slug]
            cid = unit_cell_id(slug, model_kind, lead, "ctx") + "__placement"
            fit_cell_matched(cid, slug, unit_xy(ustore, "x_spanmean"), n_place, args)
            mv = _matched_value(args, cid)
            placement_vals[slug] = mv and mv["value"]
        ref_m = ref.get("per_model", {}).get(model_kind) or {}
        rung_ref = ref_m.get("rung_values_matched_ctx") or {}
        committed_refs: dict = {
            "seed42_r7_endpoint_per_persona": rung_ref.get("r7_endpoint_per_persona"),
            "seed42_r7_endpoint_mean": rung_ref.get("r7_endpoint_mean"),
            "seed42_r1_qa_oneline": rung_ref.get("r1_qa_oneline"),
        }
        for seed_label, rel in H0_SEED_DIRS[1:]:
            sc_path = args.committed_eval_root / rel / "seed_comparison.json"
            if not sc_path.exists():
                continue
            sc_m = json.loads(sc_path.read_text()).get("per_model", {}).get(model_kind) or {}
            sc_rv = sc_m.get("rung_values_matched_ctx") or {}
            committed_refs[f"{seed_label}_r7_endpoint_per_persona"] = sc_rv.get(
                "r7_endpoint_per_persona"
            )

        # (d) cross-label swap on the combined Assistant+Wren(45) store. The
        # frozen run_swap keys its output/fingerprint on a slug; the assistant
        # slug is passed, so the read lands at swap_r7_op_assistant_<model>.json
        # (this round runs no other swap for that slug — SWAP_RUNGS excludes
        # the r7_op_* cells; provenance recorded in swap_note).
        combined = _concat_stores(stores["r7_op_assistant"], stores["r7_op_wren"])
        swap = run_swap("r7_op_assistant", combined, model_kind, args)

        # (e) full-slot collapse audits, all cells of this model.
        audits = {slug: collapse_audit(args, model_kind, slug) for slug in LABEL_RUNGS}

        yield_report = {
            slug: {"n": int(realized_n[slug]), "floor": floor, "kept": realized_n[slug] >= floor}
            for slug in LABEL_RUNGS
        }
        for slug in staged:
            release_store_local(args, slug, model_kind)
        per_model[model_kind] = {
            "leads": {slug: units[slug][0] for slug in LABEL_RUNGS},
            "realized_n": {slug: int(n) for slug, n in realized_n.items()},
            "fiction_yield": yield_report,
            "full_n_values": {
                slug: {
                    arm: {k: v for k, v in full[slug][arm].items() if k != "boot_draws"}
                    for arm in MATCHED_ARMS
                }
                for slug in LABEL_RUNGS
            },
            "deltas_full_n": deltas_full,
            "deltas_pairwise_matched": deltas_matched,
            "pairwise_n": n_pair,
            "placement": {
                "n_committed": committed_n,
                "n_used": int(n_place),
                "at_committed_n": bool(placement_at_committed),
                "non_comparability_note": (
                    None
                    if placement_at_committed
                    else (
                        "a realized cell n fell below the committed per-model "
                        "matched n; placement subsampled at the pairwise min "
                        "instead — NOT directly comparable to the committed bars"
                    )
                ),
                "values_ctx": placement_vals,
                "committed_references": committed_refs,
            },
            "swap_cross_label": swap,
            "swap_note": (
                "combined Assistant+Wren(45) store; run_swap derangement across "
                "char_ids at matched scene-position; artifact file "
                f"swap_r7_op_assistant_{model_kind}.json"
            ),
            "collapse_audits": audits,
        }
        del stores, units, combined, full
    out = {
        "metadata": common.metadata(SCRIPT, args.seed, 0),
        "code_sha": common.git_commit(),
        "gen_seed": r1335.GEN_SEED,
        "gen_seed_replicate": r1335.gen_seed_for_rung("r7_op_wren46"),
        "headline_layer": c1310.HEADLINE_LAYER,
        "lambda_selection": LAMBDA_SELECTION,
        "models_compared": list(models),
        "label_pairs": [list(p) for p in LABEL_PAIRS],
        "h0_pair_noise_band": h0_band,
        "reference": {
            "path": str(ref_path),
            "code_sha": ref.get("code_sha"),
            "note": "committed seed-42 ladder_summary.json (read-only placement anchors)",
        },
        "committed_eval_root": str(args.committed_eval_root),
        "per_model": per_model,
        "smoke": bool(smoke),
    }
    if sorted(models) != sorted(c1310.MODEL_KINDS):
        out["scope_note"] = "declared model-subset round: only models_compared were run this round"
    c1310.write_json(args.out_dir / "label_comparison.json", out)
    print(
        "[i1335-fit] label_comparison.json written "
        f"(gen_seed={r1335.GEN_SEED}; models={list(per_model)}; "
        f"B_hat={h0_band['b_hat']:.4f})"
    )
    return out


def main() -> int:
    args = parse_args()
    if args.assert_cuda:
        dev = fit825._fit_device()
        assert dev.type == "cuda", (
            f"_fit_device() resolved {dev} — the production fit battery must run "
            "GPU-resident (plan §9 P3 binding smoke device gate); HALT + fix device routing"
        )
        print(f"[i1335-fit] device gate PASS: _fit_device()={dev}")
        if not (
            args.rung
            or args.matched_n
            or args.summary
            or args.seed_compare
            or args.label_compare
            or args.verify_vectorized
        ):
            return 0
    if args.verify_vectorized:
        fit825.assert_vectorized_equivalence(seed=args.seed)
        if not (
            args.rung or args.matched_n or args.summary or args.seed_compare or args.label_compare
        ):
            return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        assert m in c1310.MODEL_KINDS, f"unknown model {m!r}"

    if args.rung:
        assert args.model, "--rung needs --model"
        print(f"[phase=p3_fit_{args.rung}_{args.model}] fit battery")
        run_rung_fits(args, args.rung, args.model)
        return 0
    if args.matched_n:
        print(f"[phase=p3_matched_n] matched-n battery (models={models})")
        run_matched(args, models)
        return 0
    if args.summary:
        print(f"[phase=p3_summary] ladder summary + gates (models={models})")
        summary = build_ladder_summary(args, models, args.smoke)
        gates = summary["gates"]
        if not args.smoke:
            failed = [
                k for k in ("gate1_fiction_anchor", "gate2_qa_endpoint") if not gates[k]["pass"]
            ]
            if failed:
                print(f"[i1335-fit] BINDING GATE FAIL: {failed} — halt and diagnose (plan §7)")
                return 3
        return 0
    if args.seed_compare:
        print(f"[phase=p3_seed_compare] seed comparison (models={models})")
        build_seed_comparison(args, models, args.smoke)
        return 0
    if args.label_compare:
        print(f"[phase=p4_label_compare] label comparison (models={models})")
        build_label_comparison(args, models, args.smoke)
        return 0
    raise SystemExit(
        "no action requested (--rung/--matched-n/--summary/--seed-compare/"
        "--label-compare/--verify-vectorized)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
