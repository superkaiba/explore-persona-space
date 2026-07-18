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
    """Fit units: one per Q&A rung; one per persona for fiction rungs."""
    if r1335.RUNGS[slug]["family"] == "qa":
        return [("all", store)]
    units = []
    for persona in c1310.PERSONA_LABELS:
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
    """Rollout collapse audit on THIS run's endpoint rollouts (the seed43-round
    interpretation read, wired into the seed-compare summary): under-floor line
    counts (n_completion_tokens < DIALOGUE_MIN_TOKENS) total / per-slot /
    per-persona, plus slot-4 exact-"I agree." counts (the seed-42 collapse
    signature). Fail-loud on a missing rollout file — seed-compare rounds
    always generate r7_endpoint, so absence is a pipeline bug, never skipped."""
    path = r1335.gen_path(args.data_dir, slug, model_kind)
    assert path.exists(), (
        f"collapse audit: missing endpoint rollouts {path} — the seed-compare "
        "round's own fiction generation must have run (fail-loud)"
    )
    total = under = slot4_total = slot4_agree = 0
    per_slot: dict[str, int] = {}
    per_persona: dict[str, int] = {}
    agree_per_persona: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            slot = int(row.get("slot", 0))
            persona = str(row.get("persona", "?"))
            if int(row["n_completion_tokens"]) < r1335.DIALOGUE_MIN_TOKENS:
                under += 1
                per_slot[f"slot{slot}"] = per_slot.get(f"slot{slot}", 0) + 1
                per_persona[persona] = per_persona.get(persona, 0) + 1
            if slot == 4:
                slot4_total += 1
                if (row.get("completion") or "").strip() == "I agree.":
                    slot4_agree += 1
                    agree_per_persona[persona] = agree_per_persona.get(persona, 0) + 1
    assert total > 0, f"collapse audit: empty rollout file {path}"
    return {
        "rollouts": str(path),
        "dialogue_min_tokens": int(r1335.DIALOGUE_MIN_TOKENS),
        "n_lines": total,
        "under_floor_lines": under,
        "under_floor_pct": round(100.0 * under / total, 2),
        "under_floor_per_slot": dict(sorted(per_slot.items())),
        "under_floor_per_persona": dict(sorted(per_persona.items())),
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
            or args.verify_vectorized
        ):
            return 0
    if args.verify_vectorized:
        fit825.assert_vectorized_equivalence(seed=args.seed)
        if not (args.rung or args.matched_n or args.summary or args.seed_compare):
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
    raise SystemExit(
        "no action requested (--rung/--matched-n/--summary/--seed-compare/--verify-vectorized)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
