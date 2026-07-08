"""Issue #825 `onpolicy-separator-control` r8: position-matched exogenous refit.

Plan v21 section 6 named 0-GPU-h follow-up trigger (both conditions held: D landed
mid-band on both models AND the anchor-position distributions shift by construction
— on-policy anchors carry zero mass below token index 256 while ~26% of exogenous
anchor mass sits below 256). This script separates anchor POSITION from span
PROVENANCE: restrict the EXOGENOUS armC pairs (both models) to anchor token
position >= 256, refit the ROTATED estimator at the frozen layers {14,18,19,26}
with the identical committed machinery — same random-projection P draw stream
(``default_rng(FIT_SEED + 7)``, headline-first layer order, so P at L19 is bitwise
the committed one; P depends only on hidden dim), same group 5-fold cached-eigh
Gram-GCV ridge — plus 20 group-blocked pairing-shuffle nulls at L19 and a
size-matched position-AGNOSTIC random-subsample control at L19 (3 seeds,
``group_stratified_subsample``) separating the position restriction from the
plain n drop.

Diagnostic read only (no bootstrap battery): does the position-matched exogenous
W_ex rise toward the on-policy W_on (anchor depth explains part of the on-policy
gain) or stay put (the provenance read survives position matching)?

Validation twin: the full-n rotated refit at L19 must reproduce the committed
anchor-refit rotated values (base 0.36261 / instruct 0.34892) within 1e-3 —
proving the staged stores + fit path match the committed W_ex pipeline.

Compute character: closed-form cached-eigh Gram ridge (eigh per fold is
Y-independent and reused across all 20 null draws); per model ~4 obs-layer fits
x 5 folds + 1 full-n parity fit + 20 L19 null predicts + 3 subsample fits
~= 15-20 min VM CPU at 8 threads. 0 GPU-h. Thread caps via load_dotenv().

Outputs: eval_results/issue_825/onpolicy-separator-control/
position_matched_wex_{base,instruct}.json (one JSON per model, written the
moment that model completes — checkpoint-per-phase).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before torch/numpy import

import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402

SCRIPT = "scripts/issue825_position_matched_wex.py"
STAGE = Path("/mnt/eps-data/thomasjiralerspong/i825_posmatch_stage")
OUT_DIR = SCRIPTS.parent / "eval_results" / "issue_825" / "onpolicy-separator-control"
PAIRS_REV = "9534b9981d6b4fb4f1259c9b06f021d311a46af4"
PAIRS_PATH = "issue931_story_map/raw_completions/pairs_meta/pairs_armC.jsonl"
POSITION_FLOOR = 256  # on-policy prompts are 256-token article prefixes (plan v21 G2)
SUBSAMPLE_SEEDS = (1, 2, 3)

STORES = {
    "base": {
        "prefix": "issue825_base_sep_control/analysis_tensors/armC",
        "revision": "d4085b09d79fc46537b9da60bd6ffd8a754a677a",
        "committed_rotated_L19": 0.3626111899733,  # anchor_base/cells_armC_sep.json
    },
    "instruct": {
        "prefix": "issue931_story_map/analysis_tensors/armC",
        "revision": "d959b0c6016b1ae7bac7ae0115f09a2f2d905cae",
        "committed_rotated_L19": 0.3489193821633685,  # anchor_inst/cells_armC_sep.json
    },
}
# decision_support.json committed comparison values (re-quoted for the JSON reader;
# the committed JSON stays the ground truth).
DECISION_REFS = {
    "base": {
        "w_on_rotated": -1.2278079045145343,
        "w_on_mlp": 0.4954989932913608,
        "w_on_max": 0.4954989932913608,
        "ceiling": 0.5876803039140281,
        "D_committed": 0.5904310946594687,
    },
    "instruct": {
        "w_on_rotated": 0.48775360035988136,
        "w_on_mlp": 0.44197550514465567,
        "w_on_max": 0.48775360035988136,
        "ceiling": 0.6730940896676356,
        "D_committed": 0.4276669498894166,
    },
}


def _download(path_in_repo: str, revision: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    got = hf_hub_download(
        common.HF_DATA_REPO,
        path_in_repo,
        repo_type="dataset",
        revision=revision,
        local_dir=dest_dir,
    )
    return Path(got)


def load_anchor_positions(pairs_file: Path | None = None) -> dict[str, int]:
    """row_id -> anchor token position from the pinned pairs_armC.jsonl.

    Field verified against scripts/issue931_build_pairs.py:546 —
    ``meta={"window_id": article_id, "anchor_pos": int(t)}``.
    ``pairs_file`` (round-8 --pos-max mode) reads an already-staged local copy
    of the SAME pinned file instead of re-downloading; default None preserves
    the HF fetch path byte-for-byte.
    """
    local = pairs_file or _download(PAIRS_PATH, PAIRS_REV, STAGE / "pairs")
    pos: dict[str, int] = {}
    for line in local.read_text().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        assert "anchor_pos" in d.get("meta", {}), f"anchor_pos missing in pair {d.get('row_id')}"
        pos[d["row_id"]] = int(d["meta"]["anchor_pos"])
    assert len(pos) == 3600, len(pos)
    return pos


def stage_store(model: str) -> dict:
    """Download + load the pinned exogenous armC store for one model."""
    from huggingface_hub import HfApi

    spec = STORES[model]
    api = HfApi()
    entries = [
        e.path
        for e in api.list_repo_tree(
            common.HF_DATA_REPO,
            path_in_repo=spec["prefix"],
            repo_type="dataset",
            revision=spec["revision"],
        )
        if e.path.endswith(".pt")
    ]
    assert entries, f"no shards under {spec['prefix']}"
    dest = STAGE / f"{model}_dl"
    for p in sorted(entries):
        print(f"[i825-posmatch] fetch {model}: {p}", flush=True)
        _download(p, spec["revision"], dest)
    store = fit931.load_regime_store(dest / spec["prefix"], "armC")
    assert store["arrays"]["x_sep"].shape[0] == 3600, store["arrays"]["x_sep"].shape
    return store


def rotated_fit_r2(
    Xp: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    *,
    n_folds: int = common.N_FOLDS,
    seed: int = common.FIT_SEED,
    null_perms: list[np.ndarray] | None = None,
) -> tuple[float, list[float]]:
    """Group 5-fold cached-eigh Gram-GCV ridge on an already-projected X.

    Mirrors fit931.rotated_control_preds' fitting path (same _prep_fold /
    _ridge_predict_cached / _cv_folds); optionally evaluates group-blocked
    pairing-shuffle nulls through the SAME per-fold eigh cache (the
    heldout_r2_sweep null semantics: Y rows permuted at the GROUP level).
    Returns (observed pooled R^2, per-draw null pooled R^2 list).
    """
    n = Xp.shape[0]
    folds = fit825._cv_folds(groups, n_folds, seed)
    n_null = len(null_perms) if null_perms else 0
    ss_res, ss_tot = 0.0, 0.0
    ss_res_n = np.zeros(n_null)
    ss_tot_n = np.zeros(n_null)
    for k in range(n_folds):
        te = folds == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            continue
        cache = fit825._prep_fold(Xp[tr], Xp[te])
        pred = fit825._ridge_predict_cached(cache, Y[tr])
        true = Y[te].astype(np.float64)
        mu = true.mean(0)
        ss_res += float(np.sum((true - pred) ** 2))
        ss_tot += float(np.sum((true - mu) ** 2))
        for d in range(n_null):
            Yp_ = Y[null_perms[d]]
            pred_n = fit825._ridge_predict_cached(cache, Yp_[tr])
            true_n = Yp_[te].astype(np.float64)
            mu_n = true_n.mean(0)
            ss_res_n[d] += float(np.sum((true_n - pred_n) ** 2))
            ss_tot_n[d] += float(np.sum((true_n - mu_n) ** 2))
        del cache
    obs = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    nulls = [
        float("nan") if ss_tot_n[d] < 1e-12 else float(1.0 - ss_res_n[d] / ss_tot_n[d])
        for d in range(n_null)
    ]
    assert n == len(groups)
    return obs, nulls


def group_blocked_perms(groups: np.ndarray, n_draws: int, seed: int) -> list[np.ndarray]:
    """heldout_r2_sweep's group-level pairing permutations (rows of a group move together)."""
    rng = np.random.default_rng(seed + 1)
    ids = np.asarray(groups)
    uniq, inv = np.unique(ids, return_inverse=True)
    row_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    perms = []
    for _ in range(n_draws):
        cp = rng.permutation(len(uniq))
        perms.append(np.concatenate([row_of[k] for k in cp]))
    return perms


def run_model(model: str, anchor_pos: dict[str, int]) -> None:
    spec = STORES[model]
    refs = DECISION_REFS[model]
    store = stage_store(model)
    X = store["arrays"]["x_sep"]  # (3600, 28, 3584) fp32
    Y = store["arrays"]["y"]
    groups = store["group_ids"]
    row_ids = store["row_ids"]
    missing = [r for r in row_ids if r not in anchor_pos]
    assert not missing, f"{len(missing)} store rows missing from pinned pairs"
    pos = np.asarray([anchor_pos[r] for r in row_ids])
    mask = pos >= POSITION_FLOOR
    n_kept = int(mask.sum())
    print(
        f"[i825-posmatch] {model}: n={len(pos)} kept={n_kept} "
        f"({n_kept / len(pos):.3f}) groups_kept={len(np.unique(groups[mask]))}",
        flush=True,
    )

    hl = common.HEADLINE_LAYER
    order = [hl] + [li for li in common.FROZEN_LAYERS if li != hl]
    d_in = X.shape[2]
    rng = np.random.default_rng(common.FIT_SEED + 7)  # committed P draw stream
    projections = {li: rng.standard_normal((d_in, d_in)) / np.sqrt(d_in) for li in order}

    # Validation twin: full-n rotated refit at L19 vs the committed anchor value.
    Xp_full = (X[:, hl, :].astype(np.float64) @ projections[hl]).astype(np.float32)
    full_obs, _ = rotated_fit_r2(Xp_full, Y[:, hl, :], groups)
    d_val = abs(full_obs - spec["committed_rotated_L19"])
    print(
        f"[i825-posmatch] {model} full-n rotated L19 = {full_obs:.6f} (delta {d_val:.2e})",
        flush=True,
    )
    assert d_val < 1e-3, (model, full_obs, spec["committed_rotated_L19"])
    del Xp_full

    # Position-matched refit at the frozen layers (+ nulls at L19).
    g_m = groups[mask]
    perms = group_blocked_perms(g_m, common.N_NULL_DRAWS, common.FIT_SEED)
    pm_r2: dict[str, float] = {}
    pm_nulls: list[float] = []
    for li in order:
        Xp = (X[mask][:, li, :].astype(np.float64) @ projections[li]).astype(np.float32)
        obs, nulls = rotated_fit_r2(
            Xp, Y[mask][:, li, :], g_m, null_perms=perms if li == hl else None
        )
        pm_r2[str(li)] = obs
        if li == hl:
            pm_nulls = nulls
        print(f"[i825-posmatch] {model} position-matched rotated L{li} = {obs:.6f}", flush=True)
        del Xp

    # Size-matched position-AGNOSTIC subsample control at L19 (n confound read).
    sub_r2 = {}
    for s in SUBSAMPLE_SEEDS:
        idx = common.group_stratified_subsample(groups, n_kept, seed=common.BUILD_SEED + s)
        Xp = (X[idx][:, hl, :].astype(np.float64) @ projections[hl]).astype(np.float32)
        obs, _ = rotated_fit_r2(Xp, Y[idx][:, hl, :], groups[idx])
        sub_r2[str(s)] = obs
        print(f"[i825-posmatch] {model} size-matched subsample seed {s}: {obs:.6f}", flush=True)
        del Xp

    w_ex_pm = pm_r2[str(hl)]
    d_pm = (refs["w_on_max"] - w_ex_pm) / (refs["ceiling"] - w_ex_pm)
    payload = {
        "metadata": common.metadata(SCRIPT, common.FIT_SEED, n_kept),
        "followup_label": "onpolicy-separator-control",
        "model": model,
        "store_prefix": spec["prefix"],
        "store_revision": spec["revision"],
        "pairs_path": PAIRS_PATH,
        "pairs_revision": PAIRS_REV,
        "position_floor": POSITION_FLOOR,
        "anchor_pos_field": "meta.anchor_pos (issue931_build_pairs.py)",
        "n_total": len(pos),
        "n_kept": n_kept,
        "kept_fraction": n_kept / len(pos),
        "n_groups_total": len(np.unique(groups)),
        "n_groups_kept": len(np.unique(g_m)),
        "anchor_pos_summary": {
            "frac_below_floor": float((pos < POSITION_FLOOR).mean()),
            "median": float(np.median(pos)),
            "median_kept": float(np.median(pos[mask])),
        },
        "validation_fulln_rotated_L19": {
            "refit": full_obs,
            "committed": spec["committed_rotated_L19"],
            "abs_delta": d_val,
        },
        "position_matched_rotated_r2": pm_r2,
        "null_L19": {
            "draws": pm_nulls,
            "mean": float(np.nanmean(pm_nulls)),
            "p975": float(np.nanquantile(pm_nulls, 0.975)),
            "n_draws": len(pm_nulls),
            "kind": "group-blocked pairing shuffle (heldout_r2_sweep semantics)",
        },
        "size_matched_subsample_rotated_L19": {
            "per_seed": sub_r2,
            "mean": float(np.mean(list(sub_r2.values()))),
            "subsampler": "group_stratified_subsample(seed=931+s)",
        },
        "committed_reference": {
            "w_ex_fulln_rotated_L19": spec["committed_rotated_L19"],
            **refs,
        },
        "derived": {
            "w_ex_position_matched_L19": w_ex_pm,
            "delta_vs_fulln": w_ex_pm - spec["committed_rotated_L19"],
            "D_position_matched_diagnostic": float(d_pm),
            "note": (
                "diagnostic D under the position-matched W_ex reference; the "
                "committed decision_support D (full-n convention) stays the "
                "headline. Subsample control isolates the n drop from the "
                "position restriction."
            ),
        },
    }
    common.write_json(OUT_DIR / f"position_matched_wex_{model}.json", payload)
    print(f"[i825-posmatch] wrote position_matched_wex_{model}.json", flush=True)


def extend_subsample(model: str, n_seeds: int = 10) -> None:
    """r9 (interp-critique round 2): extend the size-matched position-AGNOSTIC
    random-subsample control at L19 to n_seeds seeds for one model, so the
    position-matched-vs-subsample contrast gets a proper seed-band read
    (mean +/- sd + where the position-matched value falls). Reuses the staged
    store + the committed projection stream; seeds 1..3 must reproduce the
    committed per_seed values (validation twin). Writes a _v2 sidecar next to
    the committed JSON (the round-8 JSON stays untouched as the r8 record).
    """
    committed_path = OUT_DIR / f"position_matched_wex_{model}.json"
    committed = json.loads(committed_path.read_text())
    anchor_pos = load_anchor_positions()
    store = stage_store(model)
    X = store["arrays"]["x_sep"]
    Y = store["arrays"]["y"]
    groups = store["group_ids"]
    row_ids = store["row_ids"]
    pos = np.asarray([anchor_pos[r] for r in row_ids])
    n_kept = int((pos >= POSITION_FLOOR).sum())
    assert n_kept == committed["n_kept"], (n_kept, committed["n_kept"])

    hl = common.HEADLINE_LAYER
    order = [hl] + [li for li in common.FROZEN_LAYERS if li != hl]
    d_in = X.shape[2]
    rng = np.random.default_rng(common.FIT_SEED + 7)  # committed P draw stream
    # headline-first order => P at L19 is the FIRST draw, bitwise the committed one
    assert order[0] == hl
    P = rng.standard_normal((d_in, d_in)) / np.sqrt(d_in)

    sub_r2: dict[str, float] = {}
    for s in range(1, n_seeds + 1):
        idx = common.group_stratified_subsample(groups, n_kept, seed=common.BUILD_SEED + s)
        Xp = (X[idx][:, hl, :].astype(np.float64) @ P).astype(np.float32)
        obs, _ = rotated_fit_r2(Xp, Y[idx][:, hl, :], groups[idx])
        sub_r2[str(s)] = obs
        print(f"[i825-posmatch-ext] {model} subsample seed {s}: {obs:.6f}", flush=True)
        del Xp

    # Validation twin: seeds 1..3 reproduce the committed round-8 values.
    for s, v in committed["size_matched_subsample_rotated_L19"]["per_seed"].items():
        assert abs(sub_r2[s] - v) < 1e-9, (s, sub_r2[s], v)

    vals = np.array(list(sub_r2.values()))
    pm = committed["position_matched_rotated_r2"][str(hl)]
    sd = float(vals.std(ddof=1))
    payload = {
        "metadata": common.metadata(SCRIPT + " --extend-subsample", common.FIT_SEED, n_kept),
        "followup_label": "onpolicy-separator-control",
        "model": model,
        "extends": committed_path.name,
        "n_kept": n_kept,
        "subsampler": "group_stratified_subsample(seed=931+s)",
        "n_seeds": n_seeds,
        "size_matched_subsample_rotated_L19": {
            "per_seed": sub_r2,
            "mean": float(vals.mean()),
            "sd": sd,
            "min": float(vals.min()),
            "max": float(vals.max()),
        },
        "position_matched_rotated_L19": pm,
        "contrast_vs_subsample": {
            "delta_vs_mean": float(pm - vals.mean()),
            "z_vs_seed_band": float((pm - vals.mean()) / sd),
            "n_seeds_below_position_matched": int((vals < pm).sum()),
            "note": (
                "positive delta = position-matched refit sits ABOVE the size-matched "
                "position-agnostic subsample band at collapsed subset-n; overlapping "
                "group-stratified subsamples understate structured-subset variability, "
                "so the z is against an understated null."
            ),
        },
    }
    common.write_json(OUT_DIR / f"position_matched_wex_{model}_v2.json", payload)
    print(f"[i825-posmatch-ext] wrote position_matched_wex_{model}_v2.json", flush=True)


def run_pos_restricted(
    model: str,
    pos_max: int,
    *,
    store_dir: Path | None,
    pairs_file: Path | None,
    out_dir: Path,
    out_name: str | None,
    smoke: bool,
) -> None:
    """Round-8 `sampled-separator-control` G4b companion: position-RESTRICTED
    exogenous refit — keep anchors at token position < ``pos_max`` (the arm-C
    fixed prefix-final anchors sit at <= 254, i.e. BELOW the 256-token prefix
    boundary, the mirror of the round-7 ``pos >= POSITION_FLOOR`` restriction).

    Flag-gated round-8 extension (plan v22 section 4 G4b(4)); the default
    entrypoints are untouched. ``store_dir`` reuses an already-staged
    ``store/armC`` dir (the dispatcher's p0-staged exogenous store) instead of
    re-downloading; ``smoke`` records the full-n validation twin instead of
    asserting it (tiny 1-shard smoke subsets cannot reproduce full-n values).
    """
    anchor_pos = load_anchor_positions(pairs_file)
    if store_dir is not None:
        store = fit931.load_regime_store(store_dir, "armC")
    else:
        STAGE.mkdir(parents=True, exist_ok=True)
        store = stage_store(model)
    spec = STORES[model]
    X = store["arrays"]["x_sep"]
    Y = store["arrays"]["y"]
    groups = store["group_ids"]
    row_ids = store["row_ids"]
    missing = [r for r in row_ids if r not in anchor_pos]
    assert not missing, f"{len(missing)} store rows missing from pinned pairs"
    pos = np.asarray([anchor_pos[r] for r in row_ids])
    mask = pos < pos_max
    n_kept = int(mask.sum())
    assert n_kept > 0, f"no exogenous anchors below {pos_max}"
    print(
        f"[i825-posmatch-below] {model}: n={len(pos)} kept={n_kept} "
        f"({n_kept / len(pos):.3f}) groups_kept={len(np.unique(groups[mask]))}",
        flush=True,
    )

    hl = common.HEADLINE_LAYER
    n_layers = X.shape[1]
    hl = hl if n_layers > hl else n_layers - 1  # tiny-model smoke rebind
    order = [hl] + [li for li in common.FROZEN_LAYERS if li != hl and li < n_layers]
    d_in = X.shape[2]
    rng = np.random.default_rng(common.FIT_SEED + 7)  # committed P draw stream
    projections = {li: rng.standard_normal((d_in, d_in)) / np.sqrt(d_in) for li in order}

    # Validation twin: full-n rotated refit at the headline layer vs committed.
    Xp_full = (X[:, hl, :].astype(np.float64) @ projections[hl]).astype(np.float32)
    full_obs, _ = rotated_fit_r2(Xp_full, Y[:, hl, :], groups)
    d_val = abs(full_obs - spec["committed_rotated_L19"])
    print(
        f"[i825-posmatch-below] {model} full-n rotated L{hl} = {full_obs:.6f} "
        f"(delta {d_val:.2e}, binding={not smoke})",
        flush=True,
    )
    if not smoke:
        assert d_val < 1e-3, (model, full_obs, spec["committed_rotated_L19"])
    del Xp_full

    g_m = groups[mask]
    perms = group_blocked_perms(g_m, common.N_NULL_DRAWS, common.FIT_SEED)
    pm_r2: dict[str, float] = {}
    pm_nulls: list[float] = []
    for li in order:
        Xp = (X[mask][:, li, :].astype(np.float64) @ projections[li]).astype(np.float32)
        obs, nulls = rotated_fit_r2(
            Xp, Y[mask][:, li, :], g_m, null_perms=perms if li == hl else None
        )
        pm_r2[str(li)] = obs
        if li == hl:
            pm_nulls = nulls
        print(f"[i825-posmatch-below] {model} pos<{pos_max} rotated L{li} = {obs:.6f}", flush=True)
        del Xp

    # Size-matched position-AGNOSTIC subsample control (n-confound read).
    sub_r2 = {}
    for s in SUBSAMPLE_SEEDS:
        idx = common.group_stratified_subsample(groups, n_kept, seed=common.BUILD_SEED + s)
        Xp = (X[idx][:, hl, :].astype(np.float64) @ projections[hl]).astype(np.float32)
        obs, _ = rotated_fit_r2(Xp, Y[idx][:, hl, :], groups[idx])
        sub_r2[str(s)] = obs
        print(f"[i825-posmatch-below] {model} size-matched seed {s}: {obs:.6f}", flush=True)
        del Xp

    md = common.metadata(SCRIPT + " --pos-max", common.FIT_SEED, n_kept)
    md["issue"] = 825
    payload = {
        "metadata": md,
        "followup_label": "sampled-separator-control",
        "model": model,
        "position_restriction": {"mode": "below", "pos_max": int(pos_max)},
        "pairs_path": PAIRS_PATH,
        "pairs_revision": PAIRS_REV,
        "n_total": len(pos),
        "n_kept": n_kept,
        "kept_fraction": n_kept / len(pos),
        "n_groups_total": len(np.unique(groups)),
        "n_groups_kept": len(np.unique(g_m)),
        "anchor_pos_summary": {
            "frac_below_pos_max": float((pos < pos_max).mean()),
            "median": float(np.median(pos)),
            "median_kept": float(np.median(pos[mask])),
        },
        "validation_fulln_rotated_hl": {
            "refit": full_obs,
            "committed": spec["committed_rotated_L19"],
            "abs_delta": d_val,
            "binding": not smoke,
        },
        "position_restricted_rotated_r2": pm_r2,
        "null_hl": {
            "draws": pm_nulls,
            "mean": float(np.nanmean(pm_nulls)) if pm_nulls else None,
            "p975": float(np.nanquantile(pm_nulls, 0.975)) if pm_nulls else None,
            "n_draws": len(pm_nulls),
            "kind": "group-blocked pairing shuffle (heldout_r2_sweep semantics)",
        },
        "size_matched_subsample_rotated_hl": {
            "per_seed": sub_r2,
            "mean": float(np.mean(list(sub_r2.values()))),
            "subsampler": "group_stratified_subsample(seed=931+s)",
        },
        "derived": {
            "w_ex_position_restricted_hl": pm_r2[str(hl)],
            "note": (
                "companion read for the arm-C fixed prefix-final anchor cells "
                "(sub-256 exogenous pool); consumed by "
                "issue825_sampled_sep_decision.py — no D computed here"
            ),
        },
        "smoke": bool(smoke),
    }
    name = out_name or f"posmatch_below{pos_max}_{model}.json"
    common.write_json(out_dir / name, payload)
    print(f"[i825-posmatch-below] wrote {name}", flush=True)


def _parse_pos_max_args(argv: list[str]):
    import argparse

    ap = argparse.ArgumentParser(description="round-8 --pos-max position-restricted refit")
    ap.add_argument("--pos-max", type=int, required=True)
    ap.add_argument("--model", required=True, choices=("base", "instruct"))
    ap.add_argument("--store-dir", type=Path, default=None, help="staged store/armC dir")
    ap.add_argument("--pairs-file", type=Path, default=None, help="staged pairs_armC.jsonl")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--smoke", action="store_true", help="validation twin recorded, not binding")
    return ap.parse_args(argv)


def main() -> int:
    if "--pos-max" in sys.argv[1:]:
        a = _parse_pos_max_args(sys.argv[1:])
        run_pos_restricted(
            a.model,
            a.pos_max,
            store_dir=a.store_dir,
            pairs_file=a.pairs_file,
            out_dir=a.out_dir,
            out_name=a.out_name,
            smoke=a.smoke,
        )
        print("[i825-posmatch-below] DONE rc=0", flush=True)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--extend-subsample":
        model = sys.argv[2] if len(sys.argv) > 2 else "instruct"
        extend_subsample(model, n_seeds=int(sys.argv[3]) if len(sys.argv) > 3 else 10)
        print("[i825-posmatch-ext] DONE rc=0", flush=True)
        return 0
    STAGE.mkdir(parents=True, exist_ok=True)
    anchor_pos = load_anchor_positions()
    below = sum(1 for v in anchor_pos.values() if v < POSITION_FLOOR)
    print(
        f"[i825-posmatch] pairs: {len(anchor_pos)} rows, {below} below {POSITION_FLOOR} "
        f"({below / len(anchor_pos):.3f})",
        flush=True,
    )
    for model in ("base", "instruct"):
        out = OUT_DIR / f"position_matched_wex_{model}.json"
        if out.exists():
            print(f"[i825-posmatch] {out.name} exists — skip (resume)", flush=True)
            continue
        run_model(model, anchor_pos)
    print("[i825-posmatch] DONE rc=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
