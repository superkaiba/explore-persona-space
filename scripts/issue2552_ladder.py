#!/usr/bin/env python
"""#2552 P3 — 0-GPU analysis driver: covariate ladder, category reads, replication stats.

Three legs (plan v4 §4 P3 + §6), per turn-averaged dictionary
(``rep_ta`` / ``mat_k100`` / ``mat_k200``):

- **Ladder (leg 3, hero — MF-C mechanics):** DV = rank-transformed per-feature held-out
  R². Continuous log activity is a FORCED step 0 (pre-partialed; removed from the
  competitive battery — hard-asserted before any permutation draw). Forward selection on
  the step-0 residuals; per-step Freedman–Lane residual permutation WITHIN activity
  quintiles (10,000 draws, seed 148252, per-draw SAME-argmax re-selection); fixed depth 6
  advisory; per-draw × per-covariate matrices persisted per (step, dictionary); per-step
  df + band-vs-residual-ceiling reported. Robustness: category-free full-panel ladder,
  flexible-activity-base (quintile dummies, descriptive), split-half selection stability,
  category-forced-last sensitivity, twin-covariate exclusion, 1% floor re-run.
- **Category reads (leg 1):** PRIMARY = activity-adjusted category ranking (equal-weight
  within-quintile category effects; <5-feature cells drop, renormalized + disclosed);
  raw medians SECONDARY; 10,000-draw within-quintile feature bootstrap (seed 24761);
  within-quintile Kruskal–Wallis label permutation (seed 24762); 10 pairwise contrasts
  BH-FDR per dictionary; status groups (semantic none / malformed / api-refusal /
  transport) reported separately + shadow-category ranking.
- **Replication stats (leg 2):** per-config matching accuracy + Wilson CIs; paired
  Δ_disc bootstrap (10,000 draws over complete-pair turns, seed 24761); pairwise win
  matrix + Δ_cov Wilson; the registered 5-cell (Δ_disc, Δ_cov) lattice on the POOLED
  read; LMSYS-only subset advisory; per-config ceilings + margins.

Outputs: git ``eval_results/issue_2552/{ladder,category_reads,dere_repl}/`` + figures
``figures/issue_2552/``; draw matrices → HF ``issue2552_turnsae/analysis_tensors/ladder/``.
Under ``--smoke`` EVERY output diverts under ``<out_root>/smoke`` (canonical committed
paths are never written) and tiny synthetic inputs are built in the producers' exact
schemas (the mat-family DV join still reads the REAL committed #2476 union npzs).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_SCRIPTS_DIR / "vendored_2476")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy/scipy (shared-VM thread caps freeze at import)

import numpy as np  # noqa: E402

import issue2552_judge_waves as JW  # noqa: E402  (light module top: dotenv-first, numpy-only)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i2552.ladder")

TASK_ID = 2552
SEED_BOOT = 24761  # plan §10 — bootstrap draws
SEED_PERM = 24762  # plan §10 — label permutation draws
SEED_LADDER = 148252  # plan §10 — ladder Freedman–Lane draws
N_DRAWS = 10_000
DRAW_CHUNK = 1_000
LADDER_DEPTH = 6
N_QUINT = 5
MIN_CELL = 5
FLOOR_PRIMARY = 240  # 0.2% of the 120k fit rows (alive_f240)
FLOOR_ROBUST = 1200  # 1% floor
PILOT_ABORT_S = 60.0  # plan §9: abort-to-vectorize-review if 1 block > 60 s

DICTS = JW.TA_FAMILIES  # ("rep_ta", "mat_k100", "mat_k200")
CATEGORIES = tuple(c for c, _fs in JW.APP_D_SCHEMA)  # Content/Form/Voice/Function/Meta
STATUS_GROUPS = ("none", "malformed", "api_refusal", "transport")
# competitive battery (plan §4 P3): name -> covariates_{fam}.npz key; activity is NEVER here
CONT_COVS: tuple[tuple[str, str], ...] = (
    ("act_var", "act_var"),
    ("act_mean_when_active", "act_mean_when_active"),
    ("decoder_norm", "decoder_norm"),
    ("footprint_top20", "footprint_top20"),
    ("coact_mean", "coact_mean"),
    ("rb_cos_max", "rb_cos_max"),
    ("pca_best_rank", "pca_best_rank"),
    ("share_lmsys", "share_lmsys"),
    ("consistency_twin", "consistency_twin"),
    ("match_cos", "match_cos"),
)
TWIN_COVS = ("consistency_twin", "match_cos")  # twin-inherited (risk-table exclusion re-run)
BANNED_CAND_NAMES = ("activity", "log_activity", "counts", "firing")  # step-0-only regressors
CONFIG_BUNDLE = {
    "pt_max": "andyrdt trainer_2 per-token max (131,072-wide, top-100 judged lists)",
    "pt_sum": "andyrdt trainer_2 per-token sum (131,072-wide, top-100 judged lists)",
    "rep_ta": "replication turn-averaged BatchTopK (32,768-wide, top-100 judged lists)",
    "mat_k100": "#2476 matryoshka k=100 turn-averaged (65,536-wide)",
    "mat_k200": "#2476 matryoshka k=200 turn-averaged (65,536-wide)",
}
CFG_TICK = {  # MF-E: compact per-config bundle attribution for figure tick labels
    "pt_max": "pt_max\n(andyrdt trainer_2, 131k)",
    "pt_sum": "pt_sum\n(andyrdt trainer_2, 131k)",
    "rep_ta": "rep_ta\n(replication TA, 32k)",
    "mat_k100": "mat_k100\n(#2476 matryoshka, 65k)",
    "mat_k200": "mat_k200\n(#2476 matryoshka, 65k)",
}
PAPER_REFERENCE = {
    "matching_accuracy": {"pt_max": 0.950, "rep_ta": 0.739},
    "coverage": {"avg": 0.879, "head_to_head": 0.797},
    # plan §6 / paper Table: turn-averaged (rep_ta) = 0.663, per-token max = 0.617 at k=3
    "embedding": {"pt_max": 0.617, "rep_ta": 0.663},
    "note": "Der et al. arXiv 2606.28548 reference points (their configs; plan §6)",
}


# ── paths / io ───────────────────────────────────────────────────────────────────


def _paths(args) -> SimpleNamespace:
    """Resolve every input/output root. Under --smoke ALL outputs divert under
    <out_root>/smoke and inputs resolve to the synthesized fixture dirs."""
    work = Path(args.out_root)
    if args.smoke:
        work = work / "smoke"
        ladder_out = work / "eval" / "ladder"
        cat_out = work / "eval" / "category_reads"
        dere_out = work / "eval" / "dere_repl"
        figs = work / "figures"
        agg_in = work / "inputs" / "judge_aggregates"
        dere_in = dere_out  # smoke synth writes matching/pairwise here
        raw_w3 = work / "inputs" / "raw" / "w3"
        eval_in = work / "inputs" / "eval"
        regime_path = work / "inputs" / "regime.json"
        prov_path = work / "inputs" / "prov.npy"
    else:
        ladder_out = PROJECT_ROOT / "eval_results" / "issue_2552" / "ladder"
        cat_out = PROJECT_ROOT / "eval_results" / "issue_2552" / "category_reads"
        dere_out = PROJECT_ROOT / "eval_results" / "issue_2552" / "dere_repl"
        figs = PROJECT_ROOT / "figures" / "issue_2552"
        agg_in = Path(args.judge_agg_dir)
        dere_in = Path(args.dere_dir)
        raw_w3 = Path(args.judge_root) / "raw" / "w3"
        eval_in = work / "inputs" / "eval"  # HF-fetched unless --p1-eval-dir supplies files
        regime_path = JW.REGIME_JSON
        prov_path = Path(args.judge_root) / "stage" / "prov.npy"
    draws_dir = work / "draw_matrices"
    partial = work / "partial"
    for d in (work, ladder_out, cat_out, dere_out, figs, draws_dir, partial):
        d.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        work=work,
        ladder_out=ladder_out,
        cat_out=cat_out,
        dere_out=dere_out,
        figs=figs,
        draws=draws_dir,
        partial=partial,
        agg_in=agg_in,
        dere_in=dere_in,
        raw_w3=raw_w3,
        eval_in=eval_in,
        regime_path=regime_path,
        prov_path=prov_path,
        union_paths={"mat_k100": JW.UNION_C_NPZ, "mat_k200": JW.UNION_K200_NPZ},
    )


def _write_json(path: Path, doc: dict) -> None:
    from explore_persona_space.atomic_io import write_json_atomic

    write_json_atomic(path, doc)


def _meta(phase: str) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    d = as_metadata_dict(git_provenance(), phase=phase)
    d.update(
        {
            "task_id": TASK_ID,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "seeds": {"bootstrap": SEED_BOOT, "permutation": SEED_PERM, "ladder": SEED_LADDER},
            "config_bundle": CONFIG_BUNDLE,
        }
    )
    return d


# ── small math utils ─────────────────────────────────────────────────────────────


def _rank01(x: np.ndarray) -> np.ndarray:
    """Average-rank transform to (0, 1)."""
    from scipy.stats import rankdata

    r = rankdata(np.asarray(x, np.float64), method="average")
    return (r - 0.5) / len(r)


def _orth(x: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Orthonormal basis of the columns of x (n, d) with SVD rank tolerance.

    ``scale`` (the PRE-residualization block norm) makes the cutoff absolute: a block
    residualized to numerical dust must yield rank 0, not spurious full rank on noise —
    the relative-to-s[0] tolerance alone fails exactly there (caught by the smoke)."""
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 0:
        return x
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return u[:, :0]
    tol = max(s[0], scale or 0.0) * max(x.shape) * np.finfo(np.float64).eps
    r = int((s > tol).sum())
    return u[:, :r]


def _zscore(col: np.ndarray) -> tuple[np.ndarray, bool]:
    """Standardize a 1-D column; returns (z, degenerate_flag)."""
    col = np.asarray(col, np.float64)
    sd = col.std()
    if not np.isfinite(sd) or sd == 0.0:
        return np.zeros_like(col), True
    return (col - col.mean()) / sd, False


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(center - half), float(center + half))


def _quintile_bins(log_act: np.ndarray) -> np.ndarray:
    """Quintile index 0..4 over the panel's log firing activity."""
    qs = np.quantile(log_act, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(qs, log_act, side="right").astype(np.int64)


# ── input resolution ─────────────────────────────────────────────────────────────


def _hf_fetch(path_in_repo: str, dest_dir: Path, revision: str) -> Path:
    """Revision-pinned single-file fetch through the transient-retry envelope."""
    from huggingface_hub import hf_hub_download

    import issue779_common as C
    from explore_persona_space.orchestrate import hub

    dest_dir.mkdir(parents=True, exist_ok=True)
    got = hub.retry_transient(
        lambda: hf_hub_download(
            C.HF_DATA_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest_dir),
        ),
        what=f"pinned fetch ({path_in_repo}@{revision[:8]})",
    )
    return Path(got)


def _ensure_eval_inputs(args, io) -> Path:
    """Resolve perfeature_rep.npz + regime_measured.json + covariates_{fam}.npz:
    --p1-eval-dir first, else a revision-pinned HF fetch from
    {hf_prefix}/analysis_tensors/eval/ (fail-loud last)."""
    names = ["perfeature_rep.npz", "regime_measured.json"] + [
        f"covariates_{fam}.npz" for fam in DICTS
    ]
    if args.smoke:
        missing = [n for n in names if not (io.eval_in / n).exists()]
        assert not missing, f"smoke fixtures missing (run --phase smoke): {missing}"
        return io.eval_in
    if args.p1_eval_dir:
        local = Path(args.p1_eval_dir)
        if all((local / n).exists() for n in names):
            return local
        logger.warning("[inputs] --p1-eval-dir lacks %s — falling back to HF", local)
    dest = io.eval_in
    if all((dest / n).exists() for n in names):
        return dest
    revision = JW._resolve_data_repo_revision()
    for n in names:
        got = _hf_fetch(f"{args.hf_prefix}/analysis_tensors/eval/{n}", dest / "_dl", revision)
        target = dest / n
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.hardlink_to(got) if got.stat().st_dev == dest.stat().st_dev else None
            if not target.exists():
                import shutil

                shutil.copy2(got, target)
    return dest


def _load_dv(args, io, fam: str, floor: int) -> SimpleNamespace:
    """Per-dictionary DV bundle at a panel floor: feat_ids, r2, counts, tier, panel mask."""
    eval_dir = _ensure_eval_inputs(args, io)
    if fam == "rep_ta":
        z = np.load(eval_dir / "perfeature_rep.npz")
        # producer key set: turnsae_der.phase_perfeature_r2 savez (r2_map/counts/alive_f*;
        # NO tier — the flat rep SAE has no matryoshka tiers). #2552 r2 g6-C1.
        alive_key = f"alive_f{floor}"
        required = {"feat_ids", "r2_map", "counts", alive_key}
        missing = sorted(required - set(z.files))
        assert not missing, (
            f"perfeature_rep.npz missing producer keys {missing} — have {sorted(z.files)}"
        )
        feat_ids = np.asarray(z["feat_ids"], np.int64)
        r2 = np.asarray(z["r2_map"], np.float64)
        counts = np.asarray(z["counts"], np.float64)
        tier = None
        panel = np.asarray(z[alive_key]).astype(bool) & np.isfinite(r2)
        panel_rule = f"{alive_key} (producer alive mask) & finite r2_map"
    else:
        path = io.union_paths[fam]
        assert path.exists(), (
            f"committed union npz missing: {path} — in a sparse worktree run "
            "'git sparse-checkout add eval_results/issue_2476' first"
        )
        z = np.load(path)
        feat_ids = np.asarray(z["feat_ids"], np.int64)
        r2 = np.asarray(z["r2_map"], np.float64)
        counts_key = "counts_banked" if "counts_banked" in z.files else "counts"
        counts = np.asarray(z[counts_key], np.float64)
        tier = np.asarray(z["tier"], np.float64)
        alive_key = f"alive_f{floor}"
        assert alive_key in z.files, (alive_key, sorted(z.files))
        panel = np.asarray(z[alive_key]).astype(bool) & np.isfinite(r2)
        panel_rule = f"{alive_key} (banked union mask) & finite r2_map"
    cov = np.load(eval_dir / f"covariates_{fam}.npz")
    return SimpleNamespace(
        fam=fam,
        floor=floor,
        feat_ids=feat_ids,
        r2=r2,
        counts=counts,
        tier=tier,
        panel=panel,
        panel_rule=panel_rule,
        cov=cov,
    )


_W3_REDUCE_CACHE: dict[str, dict] = {}


def _w3_per_item(args, io) -> dict[str, dict]:
    """Per-item w3 draw classes via unit 2's reduce (base + sync-reissue overlay).

    Local judge work root first, then the HF raw_completions fallback, fail-loud last."""
    key = str(io.raw_w3)
    if key in _W3_REDUCE_CACHE:
        return _W3_REDUCE_CACHE[key]
    base = io.raw_w3 / "judge_raw_w3.json"
    if not base.exists() and not args.smoke:
        revision = JW._resolve_data_repo_revision()
        try:
            _hf_fetch(
                f"{args.hf_prefix}/raw_completions/judge/w3/judge_raw_w3.json",
                io.work / "inputs" / "raw_w3_dl",
                revision,
            )
            base = io.work / "inputs" / "raw_w3_dl" / "judge_raw_w3.json"
        finally:
            pass
    assert base.exists(), (
        f"w3 raw draws missing: {io.raw_w3}/judge_raw_w3.json (and no HF copy under "
        f"{args.hf_prefix}/raw_completions/judge/w3/) — run the judge W3 wave first"
    )
    parser = JW.WAVE_PARSERS["w3"]
    per = JW.reduce_all_scores(JW._load_all_scores(base), parser)
    reissue = base.with_name("judge_raw_w3_syncreissue.json")
    if reissue.exists():
        for item, rec in JW.reduce_all_scores(JW._load_all_scores(reissue), parser).items():
            if rec["class"] == "valid" or item not in per:
                per[item] = rec  # rule-28 overlay: a valid sync re-issue record wins
    _W3_REDUCE_CACHE[key] = per
    return per


def _labels_status(args, io, fam: str, feat_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature (category, status) string arrays aligned to feat_ids.

    category in CATEGORIES for valid non-none labels else "" ; status in
    {"valid","none","malformed","api_refusal","transport","unjudged"}."""
    cat_path = io.agg_in / f"w3_categories_{fam}.json"
    assert cat_path.exists(), f"w3 categories missing: {cat_path} — run --wave w3 first"
    assigned = json.loads(cat_path.read_text())["assignments"]
    per_item = _w3_per_item(args, io)
    pref = f"w3-{fam}-f"
    status_by_feat: dict[int, str] = {}
    for item, rec in per_item.items():
        if not item.startswith(pref):
            continue
        cls = rec["class"]
        if cls == "valid":
            # semantic none is a VALID parse with field == none; the producer (phase_w3)
            # EXCLUDES none features from `assignments`, so status must be derived from
            # the per-item VALUE, never from absence-in-assignments (#2552 r2 g6-M2).
            st = "none" if tuple(rec.get("value") or ()) == ("none", "none") else "valid"
        elif cls in ("parse_fail", "truncation"):
            st = "malformed"
        else:
            st = cls
        status_by_feat[int(item[len(pref) :])] = st
    cats = np.empty(len(feat_ids), dtype=object)
    stats = np.empty(len(feat_ids), dtype=object)
    for i, f in enumerate(feat_ids):
        a = assigned.get(str(int(f)))
        st = status_by_feat.get(int(f), "unjudged")
        if a is not None:
            assert a.get("category") in CATEGORIES, (int(f), a)
            cats[i], stats[i] = a["category"], "valid"
        else:
            cats[i], stats[i] = "", st if st != "valid" else "malformed"
    return cats, stats


def _prov_u8(args, io) -> np.ndarray:
    """Per-row corpus provenance (0=lmsys, 1=wildchat), indexed by global row id."""
    if io.prov_path.exists():
        return np.load(io.prov_path)
    assert not args.smoke, f"smoke prov fixture missing: {io.prov_path}"
    import issue1482_early_layer as EL

    ns = SimpleNamespace(scratch=io.work / "stage")
    ns.scratch.mkdir(parents=True, exist_ok=True)
    EL._stage_scratch_meta(ns)
    return np.load(ns.scratch / "prov.npy")


# ── ladder core (leg 3) ──────────────────────────────────────────────────────────


def _perm_within(rng: np.random.Generator, e0: np.ndarray, quint: np.ndarray, b: int) -> np.ndarray:
    """(n, b) within-quintile permutations of the step-0 residuals e0."""
    n = e0.size
    out = np.empty((n, b), np.float64)
    for q in np.unique(quint):
        idx = np.flatnonzero(quint == q)
        keys = rng.random((b, idx.size))
        perm = np.argsort(keys, axis=1)  # (b, m) independent permutations
        out[idx, :] = e0[idx[perm]].T
    return out


def _candidate_blocks(
    dv: SimpleNamespace,
    panel_idx: np.ndarray,
    cats: np.ndarray,
    *,
    include_category: bool,
    exclude: tuple[str, ...] = (),
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Standardized candidate design blocks over the panel + per-covariate notes
    (NaN-impute counts, degenerate flags). Category = full one-hot (rank 4 vs intercept)."""
    blocks: dict[str, np.ndarray] = {}
    notes: dict[str, dict] = {}
    fids = dv.feat_ids[panel_idx]
    for name, key in CONT_COVS:
        if name in exclude:
            continue
        col = np.asarray(dv.cov[key], np.float64)[fids]
        n_nan = int((~np.isfinite(col)).sum())
        if n_nan:
            med = float(np.nanmedian(col)) if np.isfinite(col).any() else 0.0
            col = np.where(np.isfinite(col), col, med)
        if n_nan > 0.2 * len(col):
            notes[name] = {"dropped": True, "reason": f"nan_frac={n_nan / len(col):.3f} > 0.2"}
            continue
        z, degen = _zscore(col)
        notes[name] = {"n_nan_imputed": n_nan, "degenerate": degen}
        if not degen:
            blocks[name] = z[:, None]
    if dv.tier is not None and "tier" not in exclude:
        z, degen = _zscore(dv.tier[panel_idx])
        notes["tier"] = {"degenerate": degen}
        if not degen:
            blocks["tier"] = z[:, None]
    if include_category:
        lab = cats[panel_idx]
        onehot = np.stack([(lab == c).astype(np.float64) for c in CATEGORIES], axis=1)
        assert onehot.sum(axis=1).min() >= 1.0, "category ladder requires complete-case labels"
        blocks["category"] = onehot  # rank 4 once the intercept is in the base
        notes["category"] = {"df_nominal": len(CATEGORIES) - 1}
    return blocks, notes


def _assert_activity_exclusion(blocks: dict[str, np.ndarray], log_act: np.ndarray) -> None:
    """Plan §4 HARD-ASSERT: activity in the step-0 base only — by name AND by content."""
    for banned in BANNED_CAND_NAMES:
        assert banned not in blocks, f"activity-class covariate '{banned}' in competitive battery"
    la = (log_act - log_act.mean()) / max(log_act.std(), 1e-12)
    for name, x in blocks.items():
        for j in range(x.shape[1]):
            c = float(np.corrcoef(la, x[:, j])[0, 1])
            assert abs(c) < 0.999, f"candidate '{name}' col {j} duplicates activity (|r|={c:.6f})"


def _run_ladder(
    y_rank: np.ndarray,
    log_act: np.ndarray,
    quint: np.ndarray,
    blocks: dict[str, np.ndarray],
    *,
    draws: int,
    depth: int,
    seed_key: tuple[int, ...],
    with_nulls: bool,
    flexible_base: bool = False,
    draw_sink=None,
    draw_probe=None,
    progress: str = "",
) -> dict:
    """Forward-selection partial ladder with Freedman–Lane within-quintile nulls.

    Each step draws from an INDEPENDENT rng seeded ``[*seed_key, step]`` so a resumed
    run reproduces the identical draw matrix per step (restartability, #2552 r2).
    ``draw_sink(step, names, drawmat, obs)`` persists per-draw matrices the moment each
    step completes (checkpoint-per-unit); ``draw_probe(step, names)`` returns a
    fingerprint-matched persisted drawmat (or None) so completed steps are not redrawn."""
    n = y_rank.size
    if flexible_base:
        dummies = np.stack([(quint == q).astype(np.float64) for q in range(N_QUINT)], axis=1)
        base = np.column_stack([np.ones(n), dummies])
        base_desc = "intercept + activity-quintile dummies (flexible base)"
    else:
        base = np.column_stack([np.ones(n), log_act])
        base_desc = "intercept + continuous log activity (forced step 0)"
    _assert_activity_exclusion(blocks, log_act)  # BEFORE any permutation draw (plan §4)
    q_mat = _orth(base)
    y = np.asarray(y_rank, np.float64)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    e = y - q_mat @ (q_mat.T @ y)
    e0 = e.copy()  # step-0 residuals — the FL permutation object throughout
    base_r2 = 1.0 - float((e**2).sum()) / ss_tot
    remaining = dict(blocks)
    steps: list[dict] = []
    t0 = time.time()
    for t in range(1, min(depth, len(blocks)) + 1):
        names = sorted(remaining)
        qcs: dict[str, np.ndarray] = {}
        obs: dict[str, float] = {}
        den_e = float((e**2).sum())
        for name in names:
            x = remaining[name]
            xr = x - q_mat @ (q_mat.T @ x)
            qc = _orth(xr, scale=float(np.linalg.norm(x)))
            qcs[name] = qc
            obs[name] = float(((qc.T @ e) ** 2).sum() / den_e) if qc.shape[1] else 0.0
        pick = max(obs, key=lambda k: obs[k])
        rec: dict = {
            "step": t,
            "selected": pick,
            "df": int(qcs[pick].shape[1]),
            "partial_r2": obs[pick],
            "residual_share": float((e**2).sum() / ss_tot),
            "overall_r2_increment": obs[pick] * float((e**2).sum() / ss_tot),
            "observed_all": {k: obs[k] for k in names},
            "df_all": {k: int(qcs[k].shape[1]) for k in names},
        }
        if with_nulls:
            drawmat = draw_probe(t, names) if draw_probe is not None else None
            resumed = drawmat is not None
            if drawmat is None:
                rng_t = np.random.default_rng([*seed_key, t])
                drawmat = np.empty((draws, len(names)), np.float32)
                for s in range(0, draws, DRAW_CHUNK):
                    b = min(DRAW_CHUNK, draws - s)
                    e0p = _perm_within(rng_t, e0, quint, b)
                    ep = e0p - q_mat @ (q_mat.T @ e0p)
                    den = (ep**2).sum(axis=0)
                    for j, name in enumerate(names):
                        qc = qcs[name]
                        if qc.shape[1] == 0:
                            drawmat[s : s + b, j] = 0.0
                        else:
                            drawmat[s : s + b, j] = ((qc.T @ ep) ** 2).sum(axis=0) / den
                if draw_sink is not None:
                    draw_sink(t, names, drawmat, obs)
            null_max = drawmat.max(axis=1)
            p95 = float(np.quantile(null_max, 0.95))
            rec.update(
                {
                    "null_p95_partial": p95,
                    "null_p95_overall": p95 * rec["residual_share"],
                    "p_value_selection_symmetric": float(
                        (1 + int((null_max >= obs[pick]).sum())) / (draws + 1)
                    ),
                    "clears_band": bool(obs[pick] > p95),
                    "residual_ceiling_overall": rec["residual_share"],
                    "band_exceeds_ceiling": bool(
                        p95 * rec["residual_share"] >= rec["residual_share"]
                    ),
                    "n_draws": draws,
                    "resumed_drawmat": resumed,
                }
            )
        steps.append(rec)
        q_mat = np.column_stack([q_mat, qcs[pick]])  # qc ⊥ q_mat by construction
        e = e - qcs[pick] @ (qcs[pick].T @ e)
        del remaining[pick]
        if progress:
            print(
                f"[ladder] unit {progress}:step{t} sel={pick} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    return {
        "base": base_desc,
        "base_r2": base_r2,
        "n": int(n),
        "depth_run": len(steps),
        "steps": steps,
        "final_design_rank": int(q_mat.shape[1]),
        "_final": (q_mat, e),  # stripped before JSON
    }


def _draw_store(
    draws_dir: Path,
    fam: str,
    tag: str,
    feat_ids_sel: np.ndarray,
    seed_key: tuple[int, ...],
    draws: int,
    depth: int,
    *,
    content_sha: str = "",
):
    """Fingerprinted per-(step, run-tag) draw-matrix store -> (probe, sink).

    The resume predicate matches the FULL generating regime — fam/tag/draws/seed/depth,
    the panel's file-read int64 feature ids (sha over pinned-dtype bytes; never a
    recomputed float array), plus step + candidate names, PLUS ``content_sha`` — a
    caller-supplied digest of the DATA the draws are a function of (DV ranks, activity
    base, quintile bins, candidate block values), so a recompute that changes VALUES
    while leaving panel ids/seed/names identical invalidates the store instead of
    resuming stale null matrices (#2552 r3 ladder-restartability; r2 g2-M2)."""
    fid = np.ascontiguousarray(np.asarray(feat_ids_sel, np.int64))
    fp = {
        "fam": fam,
        "tag": tag,
        "draws": int(draws),
        "seed_key": [int(s) for s in seed_key],
        "depth": int(depth),
        "n": int(fid.size),
        "feat_sha": hashlib.sha256(fid.tobytes()).hexdigest(),
        "content_sha": content_sha,
    }

    def _path(step: int) -> Path:
        return draws_dir / f"step{step}_{fam}{tag}_drawmatrix.npz"

    def _want(step: int, names) -> str:
        return json.dumps({**fp, "step": int(step), "names": list(names)}, sort_keys=True)

    def probe(step: int, names) -> np.ndarray | None:
        p = _path(step)
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            if "fingerprint" not in z.files or str(z["fingerprint"]) != _want(step, names):
                return None
            print(f"[ladder] resume drawmat {p.name}", flush=True)
            return np.asarray(z["drawmat"], np.float32)

    def sink(step: int, names, drawmat: np.ndarray, obs: dict) -> None:
        from explore_persona_space.atomic_io import savez_atomic

        savez_atomic(
            _path(step),
            drawmat=drawmat,
            covariate_names=np.asarray(names),
            observed=np.asarray([obs[k] for k in names], np.float64),
            n_draws=np.int64(drawmat.shape[0]),
            seed=np.int64(seed_key[0]),
            fingerprint=np.asarray(_want(step, names)),
        )

    return probe, sink


def _content_sha(dv, idx: np.ndarray, cats: np.ndarray, blocks: dict[str, np.ndarray]) -> str:
    """Digest of the DATA a run's null draws are a function of (#2552 r3
    ladder-restartability): the file-read DV and count values on the realized panel
    (bit-exact bytes — stored-dtype slices of arrays READ FROM FILE, the
    machine-stable class), the category labels, and the realized candidate blocks
    (float32 downcast — the sanctioned relative quantization absorbing libm/CPU
    last-bit drift in the derived z-scores). Panel ids alone miss a recompute that
    changes VALUES under identical ids (r2 g2-M2)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(dv.r2[idx]).tobytes())
    h.update(np.ascontiguousarray(np.asarray(dv.counts[idx], np.int64)).tobytes())
    h.update("|".join(str(c) for c in np.asarray(cats)[idx]).encode())
    for name in sorted(blocks):
        h.update(name.encode())
        h.update(np.ascontiguousarray(blocks[name].astype(np.float32)).tobytes())
    return h.hexdigest()


def _pilot_block(y_rank, log_act, quint, block: np.ndarray, draws: int) -> float:
    """MEASURED 1-block pilot (plan §9): one (step, covariate) full-draw GEMM block at
    production shape; returns seconds. Abort handled by the caller."""
    rng = np.random.default_rng(SEED_LADDER)
    base = _orth(np.column_stack([np.ones(y_rank.size), log_act]))
    e0 = y_rank - base @ (base.T @ y_rank)
    xr = block - base @ (base.T @ block)
    qc = _orth(xr, scale=float(np.linalg.norm(block)))
    t0 = time.time()
    for s in range(0, draws, DRAW_CHUNK):
        b = min(DRAW_CHUNK, draws - s)
        e0p = _perm_within(rng, e0, quint, b)
        ep = e0p - base @ (base.T @ e0p)
        den = (ep**2).sum(axis=0)
        _ = ((qc.T @ ep) ** 2).sum(axis=0) / den
    return time.time() - t0


def phase_ladder(args) -> None:
    """Leg 3: the forward-selection partial ladder, all dictionaries + robustness runs."""
    print("[phase=ladder]", flush=True)
    io = _paths(args)
    draws = int(args.draws)
    out_doc: dict = {
        "dv": "rank-transformed per-feature held-out R2 (map prediction vs stored answer mean)",
        "mechanics": "forced step-0 continuous log activity; forward selection on step-0 "
        "residuals; Freedman-Lane within-activity-quintile permutation, per-draw same-argmax",
        "n_draws": draws,
        "depth": LADDER_DEPTH,
        "dictionaries": {},
        **_meta("p3-ladder"),
    }
    pilot_done = False
    unit = 0
    for fam in DICTS:
        fam_doc: dict = {}
        dv = _load_dv(args, io, fam, FLOOR_PRIMARY)
        cats, stats = _labels_status(args, io, fam, dv.feat_ids)
        panel_all = np.flatnonzero(dv.panel)
        cc_mask = dv.panel & np.array([c in CATEGORIES for c in cats], bool)
        panel_cc = np.flatnonzero(cc_mask)
        fam_doc["n_panel"] = int(panel_all.size)
        fam_doc["n_ladder_complete_case"] = int(panel_cc.size)
        fam_doc["panel_rule"] = dv.panel_rule
        fam_doc["dropped_from_primary_by_status"] = {
            g: int(((stats == g) & dv.panel).sum()) for g in (*STATUS_GROUPS, "unjudged")
        }
        assert panel_cc.size >= 6 * N_QUINT, (
            f"{fam}: complete-case ladder panel too small ({panel_cc.size}) — "
            "check W3 coverage before running the ladder"
        )

        def _prep(idx, include_category, exclude=()):
            y = _rank01(dv.r2[idx])
            log_act = np.log10(np.maximum(dv.counts[idx], 1.0))
            quint = _quintile_bins(log_act)
            blocks, notes = _candidate_blocks(
                dv, idx, cats, include_category=include_category, exclude=exclude
            )
            return y, log_act, quint, blocks, notes

        if not pilot_done:  # plan §9 measured 1-block pilot at production shape
            y, log_act, quint, blocks, _ = _prep(panel_cc, True)
            first = blocks[sorted(blocks)[0]]
            secs = _pilot_block(y, log_act, quint, first, draws)
            out_doc["pilot_block_seconds"] = secs
            print(f"[ladder] pilot 1-block ({draws} draws) = {secs:.2f}s", flush=True)
            if secs > PILOT_ABORT_S:
                raise SystemExit(
                    f"[ladder] pilot block {secs:.1f}s > {PILOT_ABORT_S}s — abort to "
                    "vectorize-review (plan §9)"
                )
            pilot_done = True

        runs: dict[str, dict] = {}

        def _go(
            tag, idx, include_category, *, exclude=(), flexible=False, nulls=True, seed_extra=0
        ):
            nonlocal unit
            unit += 1
            y, log_act, quint, blocks, notes = _prep(idx, include_category, exclude)
            seed_key = (SEED_LADDER, DICTS.index(fam), seed_extra)
            probe = sink = None
            if nulls:
                probe, sink = _draw_store(
                    io.draws,
                    fam,
                    tag,
                    dv.feat_ids[idx],
                    seed_key,
                    draws,
                    LADDER_DEPTH,
                    content_sha=_content_sha(dv, idx, cats, blocks),
                )
            rec = _run_ladder(
                y,
                log_act,
                quint,
                blocks,
                draws=draws,
                depth=LADDER_DEPTH,
                seed_key=seed_key,
                with_nulls=nulls,
                flexible_base=flexible,
                draw_sink=sink,
                draw_probe=probe,
                progress=f"{unit}/24 {fam}:{tag or 'primary'}",
            )
            rec["covariate_notes"] = notes
            rec["n_candidates"] = len(blocks)
            return rec

        # primary (complete-case, 0.2% panel) — draw matrices at the plan-named stems
        primary = _strip(_go("", panel_cc, True, seed_extra=0))
        runs["primary"] = primary
        # category-forced-last sensitivity (plan §4): category offered ONLY AFTER every
        # selected continuous covariate — run a continuous-only forward selection on the
        # same complete-case panel, then evaluate the category block on its final design
        cont = _go("__contonly", panel_cc, False, nulls=False, seed_extra=7)
        q_cont, e_cont = cont.pop("_final")
        runs["continuous_only"] = cont
        blocks_cc, _ = _candidate_blocks(dv, panel_cc, cats, include_category=True)
        cat_block = blocks_cc["category"]
        xr = cat_block - q_cont @ (q_cont.T @ cat_block)
        qc = _orth(xr, scale=float(np.linalg.norm(cat_block)))
        den = float((e_cont**2).sum())
        runs["category_forced_last"] = {
            "partial_r2_after_depth6": float(((qc.T @ e_cont) ** 2).sum() / den) if den else 0.0,
            "df": int(qc.shape[1]),
            "note": "category partial R2 given step0 + the continuous-only selected set",
        }
        # robustness runs
        runs["floor_1pct"] = _strip(
            _go(
                "__f1200",
                np.flatnonzero(_refloor(args, io, fam, FLOOR_ROBUST, cats)),
                True,
                seed_extra=1,
            )
        )
        runs["category_free_full_panel"] = _strip(_go("__catfree", panel_all, False, seed_extra=2))
        runs["flexible_activity_base"] = _strip(
            _go("__flexbase", panel_cc, True, flexible=True, nulls=False, seed_extra=3)
        )
        rng_split = np.random.default_rng([SEED_LADDER, DICTS.index(fam), 99])
        perm = rng_split.permutation(panel_cc.size)
        half = panel_cc.size // 2
        runs["split_half_1"] = _strip(_go("__half1", panel_cc[perm[:half]], True, seed_extra=4))
        runs["split_half_2"] = _strip(_go("__half2", panel_cc[perm[half:]], True, seed_extra=5))
        runs["twin_covariates_excluded"] = _strip(
            _go("__twinfree", panel_cc, True, exclude=TWIN_COVS, seed_extra=6)
        )
        fam_doc["runs"] = runs
        out_doc["dictionaries"][fam] = fam_doc
        _write_json(io.partial / f"ladder_{fam}.json", {"fam": fam, **fam_doc})
        print(f"[ladder] {fam} done", flush=True)
    _write_json(io.ladder_out / "ladder_steps.json", out_doc)
    print(f"[ladder] wrote {io.ladder_out / 'ladder_steps.json'}", flush=True)


def _strip(rec: dict) -> dict:
    rec.pop("_final", None)
    return rec


def _refloor(args, io, fam: str, floor: int, cats: np.ndarray) -> np.ndarray:
    """Complete-case mask at another floor (1% robustness)."""
    dv = _load_dv(args, io, fam, floor)
    return dv.panel & np.array([c in CATEGORIES for c in cats], bool)


# ── category reads (leg 1) ───────────────────────────────────────────────────────


def _adjusted_effects(
    r2: np.ndarray, quint: np.ndarray, lab_idx: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed adjusted effects: per-quintile (class median − grand median over labeled),
    equal-weight mean over populated (>= MIN_CELL) quintiles.

    Returns (effects (n_classes,), populated counts (n_classes,), cell matrix (n_classes, 5))."""
    eff = np.full((n_classes, N_QUINT), np.nan)
    for q in range(N_QUINT):
        in_q = quint == q
        labeled = in_q & (lab_idx >= 0)
        if labeled.sum() == 0:
            continue
        grand = np.median(r2[labeled])
        for c in range(n_classes):
            m = in_q & (lab_idx == c)
            if m.sum() >= MIN_CELL:
                eff[c, q] = np.median(r2[m]) - grand
    pop = np.isfinite(eff).sum(axis=1)
    with np.errstate(invalid="ignore"):
        adj = np.nanmean(eff, axis=1)
    return adj, pop, eff


def _bootstrap_category(
    rng: np.random.Generator,
    r2: np.ndarray,
    quint: np.ndarray,
    lab_idx: np.ndarray,
    n_classes: int,
    draws: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Within-quintile feature bootstrap; full construction re-run per draw.

    Returns (adjusted draws (draws, n_classes), raw-median draws (draws, n_classes))."""
    labeled = lab_idx >= 0
    q_members = [np.flatnonzero(labeled & (quint == q)) for q in range(N_QUINT)]
    adj_draws = np.full((draws, n_classes), np.nan, np.float64)
    raw_draws = np.full((draws, n_classes), np.nan, np.float64)
    for s in range(0, draws, DRAW_CHUNK):
        b = min(DRAW_CHUNK, draws - s)
        eff = np.full((b, n_classes, N_QUINT), np.nan)
        pooled_v: list[np.ndarray] = []
        pooled_l: list[np.ndarray] = []
        for q, idx in enumerate(q_members):
            if idx.size == 0:
                continue
            samp = idx[rng.integers(0, idx.size, size=(b, idx.size))]
            v = r2[samp]  # (b, m)
            lo = lab_idx[samp]
            grand = np.median(v, axis=1)  # (b,)
            for c in range(n_classes):
                mask = lo == c
                cnt = mask.sum(axis=1)
                with np.errstate(invalid="ignore"):
                    med = np.nanmedian(np.where(mask, v, np.nan), axis=1)
                cell = np.where(cnt >= MIN_CELL, med - grand, np.nan)
                eff[:, c, q] = cell
            pooled_v.append(v)
            pooled_l.append(lo)
        with np.errstate(invalid="ignore"):
            adj_draws[s : s + b] = np.nanmean(eff, axis=2)
        va = np.concatenate(pooled_v, axis=1)
        la = np.concatenate(pooled_l, axis=1)
        for c in range(n_classes):
            with np.errstate(invalid="ignore"):
                raw_draws[s : s + b, c] = np.nanmedian(np.where(la == c, va, np.nan), axis=1)
    return adj_draws, raw_draws


def _kw_h(ranks: np.ndarray, lab: np.ndarray, n_classes: int) -> float:
    """Kruskal–Wallis H (no tie correction; float R² is tie-free to fp resolution)."""
    n = ranks.size
    h = 0.0
    for c in range(n_classes):
        m = lab == c
        nc = int(m.sum())
        if nc == 0:
            continue
        h += nc * (ranks[m].mean() - (n + 1) / 2.0) ** 2
    return 12.0 / (n * (n + 1)) * h


def _perm_labels_within(
    rng: np.random.Generator, lab: np.ndarray, quint: np.ndarray, b: int
) -> np.ndarray:
    """(b, n) label permutations within quintiles."""
    n = lab.size
    out = np.empty((b, n), lab.dtype)
    for q in np.unique(quint):
        idx = np.flatnonzero(quint == q)
        keys = rng.random((b, idx.size))
        perm = np.argsort(keys, axis=1)
        out[:, idx] = lab[idx[perm]]
    return out


def phase_category(args) -> None:
    """Leg 1: activity-adjusted category ranking + status groups + KW permutation."""
    print("[phase=category]", flush=True)
    io = _paths(args)
    draws = int(args.draws)
    doc: dict = {"dictionaries": {}, "n_draws": draws, **_meta("p3-category")}
    for fam in DICTS:
        fam_doc: dict = {}
        for floor, floor_tag in ((FLOOR_PRIMARY, "primary_0p2pct"), (FLOOR_ROBUST, "robust_1pct")):
            dv = _load_dv(args, io, fam, floor)
            cats, stats = _labels_status(args, io, fam, dv.feat_ids)
            idx = np.flatnonzero(dv.panel)
            r2 = dv.r2[idx]
            log_act = np.log10(np.maximum(dv.counts[idx], 1.0))
            quint = _quintile_bins(log_act)
            cat_arr = cats[idx]
            stat_arr = stats[idx]
            # complete-case 5-category read
            lab5 = np.full(idx.size, -1, np.int64)
            for c_i, c in enumerate(CATEGORIES):
                lab5[cat_arr == c] = c_i
            drop_counts = {g: int((stat_arr == g).sum()) for g in (*STATUS_GROUPS, "unjudged")}
            adj, pop, cells = _adjusted_effects(r2, quint, lab5, len(CATEGORIES))
            rng_b = np.random.default_rng([SEED_BOOT, DICTS.index(fam), floor])
            adj_d, raw_d = _bootstrap_category(rng_b, r2, quint, lab5, len(CATEGORIES), draws)
            raw_med = np.array(
                [np.median(r2[lab5 == c]) if (lab5 == c).any() else np.nan for c in range(5)]
            )
            # KW + within-quintile label permutation (complete-case features only)
            cc = lab5 >= 0
            from scipy.stats import rankdata

            ranks_cc = rankdata(r2[cc], method="average")
            h_obs = _kw_h(ranks_cc, lab5[cc], len(CATEGORIES))
            rng_p = np.random.default_rng([SEED_PERM, DICTS.index(fam), floor])
            h_perm = np.empty(draws)
            pair_list = [(a, b) for a in range(5) for b in range(a + 1, 5)]
            pair_perm = np.empty((draws, len(pair_list)))
            lab_cc = lab5[cc]
            quint_cc = quint[cc]
            r2_cc = r2[cc]
            for s in range(0, draws, DRAW_CHUNK // 2):
                b = min(DRAW_CHUNK // 2, draws - s)
                lp = _perm_labels_within(rng_p, lab_cc, quint_cc, b)
                meds = np.empty((b, 5))
                for c in range(5):
                    mask = lp == c
                    counts_r = mask @ np.ones(lab_cc.size)
                    sums_r = mask @ ranks_cc
                    mean_rank = np.where(counts_r > 0, sums_r / np.maximum(counts_r, 1), np.nan)
                    nn = lab_cc.size
                    h_c = counts_r * (mean_rank - (nn + 1) / 2.0) ** 2
                    if c == 0:
                        h_acc = np.where(np.isfinite(h_c), h_c, 0.0)
                    else:
                        h_acc = h_acc + np.where(np.isfinite(h_c), h_c, 0.0)
                    with np.errstate(invalid="ignore"):
                        meds[:, c] = np.nanmedian(np.where(mask, r2_cc[None, :], np.nan), axis=1)
                h_perm[s : s + b] = 12.0 / (lab_cc.size * (lab_cc.size + 1)) * h_acc
                for j, (a, bb) in enumerate(pair_list):
                    pair_perm[s : s + b, j] = meds[:, a] - meds[:, bb]
            p_kw = float((1 + int((h_perm >= h_obs).sum())) / (draws + 1))
            # pairwise contrasts: raw median diffs, bootstrap CI + permutation p, BH-FDR
            contrasts = []
            p_raw = []
            for j, (a, b_) in enumerate(pair_list):
                d_obs = float(raw_med[a] - raw_med[b_])
                boots = raw_d[:, a] - raw_d[:, b_]
                p_two = float(
                    (1 + int((np.abs(pair_perm[:, j]) >= abs(d_obs)).sum())) / (draws + 1)
                )
                p_raw.append(p_two)
                contrasts.append(
                    {
                        "pair": f"{CATEGORIES[a]} - {CATEGORIES[b_]}",
                        "median_diff": d_obs,
                        "ci95": [
                            float(np.nanquantile(boots, 0.025)),
                            float(np.nanquantile(boots, 0.975)),
                        ],
                        "p_perm_two_sided": p_two,
                    }
                )
            order = np.argsort(p_raw)
            m = len(p_raw)
            bh = np.empty(m)
            prev = 1.0
            for rank_i, oi in enumerate(order[::-1]):
                k = m - rank_i
                prev = min(prev, p_raw[oi] * m / k)
                bh[oi] = prev
            for j, cdoc in enumerate(contrasts):
                cdoc["p_bh_fdr"] = float(bh[j])
            # status groups + shadow ranking (categories + status groups, one construction)
            shadow_classes = list(CATEGORIES) + [g for g in STATUS_GROUPS if (stat_arr == g).any()]
            lab_sh = np.full(idx.size, -1, np.int64)
            for c_i, c in enumerate(shadow_classes):
                lab_sh[(cat_arr == c) if c in CATEGORIES else (stat_arr == c)] = c_i
            adj_sh, pop_sh, _ = _adjusted_effects(r2, quint, lab_sh, len(shadow_classes))
            status_dists = {}
            for g in (*STATUS_GROUPS, "unjudged"):
                vals = r2[stat_arr == g]
                status_dists[g] = {
                    "n": int(vals.size),
                    "median": float(np.median(vals)) if vals.size else None,
                    "q25": float(np.quantile(vals, 0.25)) if vals.size else None,
                    "q75": float(np.quantile(vals, 0.75)) if vals.size else None,
                }
            adjusted = {
                c: {
                    "effect": float(adj[c_i]),
                    "ci95": [
                        float(np.nanquantile(adj_d[:, c_i], 0.025)),
                        float(np.nanquantile(adj_d[:, c_i], 0.975)),
                    ],
                    "populated_quintiles": int(pop[c_i]),
                    "per_quintile_effect": [
                        None if not np.isfinite(v) else float(v) for v in cells[c_i]
                    ],
                }
                for c_i, c in enumerate(CATEGORIES)
            }
            raw = {
                c: {
                    "median": float(raw_med[c_i]) if np.isfinite(raw_med[c_i]) else None,
                    "ci95": [
                        float(np.nanquantile(raw_d[:, c_i], 0.025)),
                        float(np.nanquantile(raw_d[:, c_i], 0.975)),
                    ],
                    "n": int((lab5 == c_i).sum()),
                }
                for c_i, c in enumerate(CATEGORIES)
            }
            rank_adj = sorted(CATEGORIES, key=lambda c: -adjusted[c]["effect"])
            rank_raw = sorted(
                CATEGORIES,
                key=lambda c: -(raw[c]["median"] if raw[c]["median"] is not None else -np.inf),
            )
            fam_doc[floor_tag] = {
                "n_panel": int(idx.size),
                "n_complete_case": int(cc.sum()),
                "panel_rule": dv.panel_rule,
                "drop_counts_by_status": drop_counts,
                "adjusted_primary": adjusted,
                "adjusted_ranking": rank_adj,
                "raw_secondary": raw,
                "raw_ranking": rank_raw,
                "raw_vs_adjusted_disagree": bool(rank_adj != rank_raw),
                "narration_rule": "on disagreement the ADJUSTED ordering carries the headline",
                "kw_h_observed": float(h_obs),
                "kw_p_within_quintile_label_perm": p_kw,
                "pairwise_contrasts_bh_fdr": contrasts,
                "status_group_distributions": status_dists,
                "shadow_ranking": {
                    "classes": shadow_classes,
                    "adjusted_effect": {
                        c: (None if not np.isfinite(adj_sh[c_i]) else float(adj_sh[c_i]))
                        for c_i, c in enumerate(shadow_classes)
                    },
                    "populated_quintiles": {
                        c: int(pop_sh[c_i]) for c_i, c in enumerate(shadow_classes)
                    },
                    "order": [
                        shadow_classes[i]
                        for i in np.argsort(
                            [
                                -(adj_sh[j] if np.isfinite(adj_sh[j]) else -np.inf)
                                for j in range(len(shadow_classes))
                            ]
                        )
                    ],
                },
                "floor_locality_caveat": (
                    "#2476-derived banked reads are floor-local (parent lesson); this cell is "
                    f"reported at floor {floor} and is not upgraded across floors"
                ),
            }
            # per-feature table (only once, at the primary floor)
            if floor == FLOOR_PRIMARY:
                table = {
                    str(int(dv.feat_ids[i])): {
                        "r2": float(dv.r2[i]),
                        "counts": float(dv.counts[i]),
                        "category": cats[i] if cats[i] else None,
                        "status": stats[i],
                    }
                    for i in idx
                }
                _write_json(
                    io.cat_out / f"feature_categories_{fam}.json",
                    {
                        "dictionary": fam,
                        "floor": floor,
                        "n": len(table),
                        "features": table,
                        **_meta("p3-category-table"),
                    },
                )
        doc["dictionaries"][fam] = fam_doc
        print(f"[category] {fam} done", flush=True)
    _write_json(io.cat_out / "category_reads.json", doc)
    print(f"[category] wrote {io.cat_out / 'category_reads.json'}", flush=True)


# ── replication stats (leg 2) ────────────────────────────────────────────────────


def _lattice_cell(dl_ci: tuple[float, float], dc_ci: tuple[float, float]) -> str:
    """The registered 5-cell (Δ_disc, Δ_cov) verdict lattice (plan §3 H2)."""
    dl_lo, dl_hi = dl_ci
    dc_lo, dc_hi = dc_ci
    if dl_lo > 0 and dc_lo > 0:
        return "Reproduced"
    if dl_hi < 0 and dc_hi < 0:
        return "Reversed"
    if dl_lo > 0 and dc_hi < 0:
        return "Not reproduced - pt_max dominance"
    if dl_hi < 0 and dc_lo > 0:
        return "Not reproduced - rep_ta dominance"
    return "Inconclusive"


def _delta_reads(
    match_rows: list[dict], pair_rows: list[dict], row_subset: set[int] | None, draws: int
) -> dict:
    """Δ_disc (paired bootstrap over complete-pair turns) + Δ_cov (Wilson) on a row subset
    (None = pooled read)."""
    by_cfg: dict[str, dict[int, bool]] = {}
    for r in match_rows:
        if not r.get("valid"):
            continue
        rid = int(r["row_id"])
        if row_subset is not None and rid not in row_subset:
            continue
        by_cfg.setdefault(r["config"], {})[rid] = bool(r["correct"])
    rep = by_cfg.get("rep_ta", {})
    pt = by_cfg.get("pt_max", {})
    common = sorted(set(rep) & set(pt))
    n_pair = len(common)
    out: dict = {"n_complete_pairs": n_pair}
    if n_pair:
        c_rep = np.array([rep[r] for r in common], np.float64)
        c_pt = np.array([pt[r] for r in common], np.float64)
        d_obs = float(c_pt.mean() - c_rep.mean())
        rng = np.random.default_rng([SEED_BOOT, n_pair])
        deltas = np.empty(draws)
        for s in range(0, draws, DRAW_CHUNK):
            b = min(DRAW_CHUNK, draws - s)
            samp = rng.integers(0, n_pair, size=(b, n_pair))
            deltas[s : s + b] = c_pt[samp].mean(axis=1) - c_rep[samp].mean(axis=1)
        ci = (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))
        acc_rep_paired = float(c_rep.mean())
        out.update(
            {
                "delta_disc": d_obs,
                "delta_disc_ci95": list(ci),
                "acc_rep_ta_paired": acc_rep_paired,
                "acc_pt_max_paired": float(c_pt.mean()),
                "delta_disc_ceiling": 1.0 - acc_rep_paired,
                "delta_disc_ci_to_ceiling_margin": (1.0 - acc_rep_paired) - ci[1],
                "n_bootstrap_draws": draws,
            }
        )
    wins = losses = 0
    for r in pair_rows:
        if not r.get("valid"):
            continue
        pair = {r["list1"], r["list2"]}
        if pair != {"rep_ta", "pt_max"}:
            continue
        rid = int(r["row_id"])
        if row_subset is not None and rid not in row_subset:
            continue
        if r["winner"] == "rep_ta":
            wins += 1
        else:
            losses += 1
    n_cov = wins + losses
    if n_cov:
        rate = wins / n_cov
        lo, hi = _wilson(wins, n_cov)
        out.update(
            {
                "cov_win_rate_rep_ta": rate,
                "delta_cov": rate - 0.5,
                "delta_cov_ci95": [lo - 0.5, hi - 0.5],
                "delta_cov_ceiling": 0.5,
                "delta_cov_ci_to_ceiling_margin": 0.5 - (hi - 0.5),
                "n_cov_trials": n_cov,
                "per_turn_rep_wins": wins,
                "per_turn_rep_losses": losses,
            }
        )
    if n_pair and n_cov:
        out["lattice_cell"] = _lattice_cell(
            tuple(out["delta_disc_ci95"]), tuple(out["delta_cov_ci95"])
        )
    return out


def phase_replication(args) -> None:
    """Leg 2: matching accuracies, paired Δ_disc, Δ_cov, win matrix, lattice verdict."""
    print("[phase=replication]", flush=True)
    io = _paths(args)
    draws = int(args.draws)
    match_path = io.dere_in / "matching_perturn.json"
    pair_path = io.dere_in / "pairwise_perturn.json"
    g2_path = io.agg_in / "g2_decision.json"
    for p in (match_path, pair_path, io.regime_path, g2_path):
        assert p.exists(), (
            f"replication input missing: {p} — the REALIZED eval regime (ids/sha/counts) "
            "derives from g2_decision.json, never the committed regime.json (#2552 r2)"
        )
    match_doc = json.loads(match_path.read_text())
    pair_doc = json.loads(pair_path.read_text())
    regime = json.loads(io.regime_path.read_text())  # committed reference, reported only
    # measured P1 regime (canonical resolution path — local dir / HF fetch): carries
    # BOTH reconstruction FVEs (#2552 r2 codex M-trainer2-fve)
    measured = json.loads((_ensure_eval_inputs(args, io) / "regime_measured.json").read_text())
    g2 = json.loads(g2_path.read_text())
    for k in ("eval_ids", "eval_ids_sha256", "n_eval_realized"):
        assert k in g2, f"g2_decision.json missing key {k!r} — have {sorted(g2)}"
    eval_ids = np.asarray(g2["eval_ids"], np.int64)
    assert eval_ids.size == int(g2["n_eval_realized"]), (
        eval_ids.size,
        g2["n_eval_realized"],
    )
    match_rows = match_doc["rows"]
    pair_rows = pair_doc["rows"]
    realized = {int(r) for r in eval_ids}
    stray = ({int(r["row_id"]) for r in match_rows} | {int(r["row_id"]) for r in pair_rows}) - (
        realized
    )
    assert not stray, f"per-turn rows outside the realized G2 eval set: {sorted(stray)[:5]}"
    # per-config accuracy + Wilson
    per_config = {}
    for cfg in JW.CONFIGS:
        rows = [r for r in match_rows if r["config"] == cfg]
        valid = [r for r in rows if r.get("valid")]
        k = sum(1 for r in valid if r["correct"])
        n = len(valid)
        lo, hi = _wilson(k, n)
        per_config[cfg] = {
            "n_rows": len(rows),
            "n_valid": n,
            "n_correct": k,
            "accuracy": (k / n) if n else None,
            "wilson_ci95": [lo, hi],
            "frac_items_complete": (n / len(rows)) if rows else None,
            "bundle": CONFIG_BUNDLE.get(cfg, cfg),
        }
    # pairwise win matrix over all config pairs
    win_matrix: dict[str, dict] = {}
    for a in JW.CONFIGS:
        for b in JW.CONFIGS:
            if a >= b:
                continue
            rows = [r for r in pair_rows if r.get("valid") and {r["list1"], r["list2"]} == {a, b}]
            wa = sum(1 for r in rows if r["winner"] == a)
            n = len(rows)
            lo, hi = _wilson(wa, n)
            win_matrix[f"{a}-vs-{b}"] = {
                "n_valid": n,
                "win_rate_first": (wa / n) if n else None,
                "wilson_ci95": [lo, hi],
            }
    pooled = _delta_reads(match_rows, pair_rows, None, draws)
    # UNCONDITIONAL (r3 smoke-gate-enumeration fix): the seeded smoke fixture
    # deterministically produces >=1 complete pair + >=1 coverage trial (g2-verified),
    # so the zero-pair gate is exercised by the smoke rather than bypassed under it.
    assert pooled.get("n_complete_pairs", 0) > 0 and pooled.get("n_cov_trials", 0) > 0, (
        "0 complete rep_ta/pt_max pairs (or 0 coverage trials) in the pooled read — "
        f"matching/pairwise rows missing or all-invalid: {pooled}"
    )
    # LMSYS-only advisory subset (prov 0 = lmsys), over the REALIZED eval ids
    prov = _prov_u8(args, io)
    assert prov.size > int(eval_ids.max()), (prov.size, int(eval_ids.max()))
    lmsys_rows = {int(r) for r in eval_ids if prov[int(r)] == 0}
    advisory = _delta_reads(match_rows, pair_rows, lmsys_rows, draws)
    advisory["note"] = (
        "LMSYS-only subset re-read — ADVISORY heterogeneity only; it cannot upgrade or "
        "change the registered pooled verdict (plan §3 H2 / d6)"
    )
    # optional passthroughs (fail-soft with disclosure)
    optional = {}
    for name, path in (
        ("w6_mean_rank", io.agg_in / "w6_ranking_perturn.json"),
        ("w7_calibration", io.agg_in / "w7_calibration.json"),
        ("embedding_coverage", io.dere_in / "embedding_coverage.json"),
    ):
        if path.exists():
            d = json.loads(path.read_text())
            if name == "embedding_coverage":
                # --tiny-model is a smoke-only twin; its coverage numbers must never
                # ride a production replication read (#2552 r2 g6-M3)
                assert args.smoke or not d.get("tiny_model_smoke", False), (
                    f"{path} was produced under --tiny-model (tiny_model_smoke: true) — "
                    "refuse in production; rerun P4 with the real embedder"
                )
            optional[name] = d.get("summary", d.get("instruments", d))
        else:
            optional[name] = f"absent - {path.name} not produced yet"
    doc = {
        "registered_carrier": "POOLED (LMSYS+WildChat) read",
        "n_eval_realized": int(g2["n_eval_realized"]),
        "eval_ids_sha256": g2["eval_ids_sha256"],
        "n_eval_committed": int(regime["n_eval"]),
        "g2_descoped": bool(g2.get("descoped", False)),
        "chance_matching": match_doc.get("summary", {}).get("chance", 0.1),
        "per_config_matching": per_config,
        "pairwise_win_matrix": win_matrix,
        "pooled": pooled,
        "lmsys_only_advisory": advisory,
        "paper_reference": PAPER_REFERENCE,
        "reconstruction_fve": {
            # plan line 76 registers ONE FVE "accumulated during this and the P1.8
            # encode passes" — the combined figure is the registered read (#2552 r3
            # comparator-fve); the per-pass figures stay as the breakdown.
            "trainer2_fve_combined": measured.get("trainer2_fve_combined"),
            "trainer2_fve_evalpass": measured.get("trainer2_fve_evalpass"),
            "trainer2_fve_poolpass": measured.get("trainer2_fve_poolpass"),
            "replication_sae_val_var_fve": measured.get("rep_sae_val_var_fve"),
            "note": (
                "trainer_2 on-corpus FVE combined over the P1.7 eval-pass + P1.8 "
                "mining-pool encode passes (plan-registered read; per-pass breakdown "
                "beside it) vs the replication TA SAE's holdout var-FVE — all global "
                "fp64 per-dim unbiased variance (T._recon_fve parity)"
            ),
        },
        "optional_inputs": optional,
        "g2_decision": {k: v for k, v in g2.items() if k != "eval_ids"},
        "n_lmsys": int((prov[eval_ids] == 0).sum()),
        "n_wildchat": int((prov[eval_ids] == 1).sum()),
        **_meta("p3-replication"),
    }
    _write_json(io.dere_out / "replication_stats.json", doc)
    verdict = pooled.get("lattice_cell", "n/a")
    print(f"[replication] pooled lattice cell: {verdict}", flush=True)
    print(f"[replication] wrote {io.dere_out / 'replication_stats.json'}", flush=True)


# ── figures ──────────────────────────────────────────────────────────────────────


def _fig_env():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    return plt, paper_palette, savefig_paper


def phase_figures(args) -> None:
    """Hero ladder + the exploratory dump (plan §6 figure list), tolerant per-figure."""
    print("[phase=figures]", flush=True)
    io = _paths(args)
    plt, palette, save = _fig_env()
    ladder = json.loads((io.ladder_out / "ladder_steps.json").read_text())
    cat = json.loads((io.cat_out / "category_reads.json").read_text())
    rep = json.loads((io.dere_out / "replication_stats.json").read_text())
    cols = palette(8)

    # 1. HERO — forward-selection partial ladder per dictionary
    fams = list(ladder["dictionaries"])
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams, strict=True):
        run = ladder["dictionaries"][fam]["runs"]["primary"]
        labels = ["activity\n(step 0)"] + [f"{s['selected']}\n(df={s['df']})" for s in run["steps"]]
        vals = [run["base_r2"]] + [s["overall_r2_increment"] for s in run["steps"]]
        bands = [np.nan] + [s.get("null_p95_overall", np.nan) for s in run["steps"]]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=[cols[0]] + [cols[1]] * (len(vals) - 1))
        for xi, bv in zip(x, bands, strict=True):
            if np.isfinite(bv):
                ax.hlines(bv, xi - 0.4, xi + 0.4, color=cols[3], lw=1.6)
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=6)
        ax.set_title(f"{fam} (n={run['n']})")
        ax.set_ylabel("overall R² (base) / increment")
    fig.suptitle("Forward-selection partial ladder — bars vs within-quintile null p95")
    fig.tight_layout()
    save(fig, "ladder_hero", dir=io.figs)
    plt.close(fig)

    # 2. sensitivity: category-forced-last + flexible base
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    xs = np.arange(len(fams))
    forced = [
        ladder["dictionaries"][f]["runs"]["category_forced_last"]["partial_r2_after_depth6"]
        for f in fams
    ]
    prim_cat = []
    for f in fams:
        steps = ladder["dictionaries"][f]["runs"]["primary"]["steps"]
        got = [s["partial_r2"] for s in steps if s["selected"] == "category"]
        prim_cat.append(got[0] if got else 0.0)
    ax.bar(xs - 0.2, prim_cat, width=0.4, color=cols[1], label="category at selection")
    ax.bar(xs + 0.2, forced, width=0.4, color=cols[2], label="category forced last")
    ax.set_xticks(xs, fams)
    ax.set_ylabel("partial R²")
    ax.legend()
    ax.set_title("Category increment: as-selected vs forced-last")
    fig.tight_layout()
    save(fig, "ladder_sensitivity_forced_last", dir=io.figs)
    plt.close(fig)

    # 3. split-half selection stability (selection sequences)
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.0))
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams, strict=True):
        runs = ladder["dictionaries"][fam]["runs"]
        for k, (tag, col) in enumerate(
            (("primary", cols[0]), ("split_half_1", cols[1]), ("split_half_2", cols[2]))
        ):
            seq = [s["selected"] for s in runs[tag]["steps"]]
            ax.plot(range(1, len(seq) + 1), [k] * len(seq), lw=0)
            for t, name in enumerate(seq):
                ax.text(t + 1, k, name, fontsize=5.5, rotation=30, color=col, ha="center")
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2], ["primary", "half 1", "half 2"])
        ax.set_xlabel("step")
        ax.set_title(fam)
    fig.suptitle("Split-half selection stability")
    fig.tight_layout()
    save(fig, "ladder_splithalf", dir=io.figs)
    plt.close(fig)

    # 4. category ranking — adjusted (primary) + raw (secondary)
    fig, axes = plt.subplots(2, len(fams), figsize=(4.2 * len(fams), 5.6))
    axes = np.atleast_2d(axes)
    for j, fam in enumerate(fams):
        cell = cat["dictionaries"][fam]["primary_0p2pct"]
        for row, (key, title) in enumerate(
            (
                ("adjusted_primary", "activity-adjusted (PRIMARY)"),
                ("raw_secondary", "raw median (secondary)"),
            )
        ):
            ax = axes[row, j]
            names = list(CATEGORIES)
            if key == "adjusted_primary":
                v = [cell[key][c]["effect"] for c in names]
                ci = np.array([cell[key][c]["ci95"] for c in names])
            else:
                # raw median is None for an empty class — plot as NaN, never TypeError
                v = [cell[key][c]["median"] for c in names]
                ci = np.array(
                    [[np.nan if q is None else q for q in cell[key][c]["ci95"]] for c in names]
                )
            v = [np.nan if x_ is None else x_ for x_ in v]
            x = np.arange(5)
            v = np.asarray(v, np.float64)
            yerr = np.abs(ci.T - v[None, :])
            ax.bar(x, v, color=cols[:5], yerr=yerr, capsize=2)
            ax.set_xticks(x, names, rotation=30, ha="right", fontsize=7)
            ax.set_title(f"{fam} — {title}", fontsize=8)
    fig.tight_layout()
    save(fig, "category_ranking", dir=io.figs)
    plt.close(fig)

    # 5. status groups + shadow ranking
    fig, axes = plt.subplots(1, len(fams), figsize=(4.4 * len(fams), 3.2))
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams, strict=True):
        cell = cat["dictionaries"][fam]["primary_0p2pct"]
        sh = cell["shadow_ranking"]
        names = sh["order"]
        vals = [sh["adjusted_effect"][c] for c in names]
        vals = [v if v is not None else np.nan for v in vals]
        colors = [cols[0] if c in CATEGORIES else cols[6] for c in names]
        ax.bar(np.arange(len(names)), vals, color=colors)
        ax.set_xticks(np.arange(len(names)), names, rotation=40, ha="right", fontsize=6)
        ax.set_title(f"{fam} shadow ranking (status groups grey-family)", fontsize=8)
    fig.tight_layout()
    save(fig, "category_status_groups", dir=io.figs)
    plt.close(fig)

    # 6. category × activity-quintile heatmap (per-quintile effects)
    # colorbar figures use constrained layout — tight_layout after a colorbar raises
    # under the paper style (agent-memory: mpl layout-engine switch refusal)
    fig, axes = plt.subplots(1, len(fams), figsize=(4.0 * len(fams), 3.0), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams, strict=True):
        cell = cat["dictionaries"][fam]["primary_0p2pct"]["adjusted_primary"]
        mat = np.array(
            [
                [np.nan if v is None else v for v in cell[c]["per_quintile_effect"]]
                for c in CATEGORIES
            ]
        )
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r")
        ax.set_yticks(range(5), CATEGORIES, fontsize=7)
        ax.set_xticks(range(N_QUINT), [f"Q{q + 1}" for q in range(N_QUINT)])
        ax.set_title(fam, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "category_activity_heatmap", dir=io.figs)
    plt.close(fig)

    # 7. category drop rates
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    groups = (*STATUS_GROUPS, "unjudged")
    width = 0.8 / len(fams)
    for k, fam in enumerate(fams):
        d = cat["dictionaries"][fam]["primary_0p2pct"]["drop_counts_by_status"]
        n_panel = cat["dictionaries"][fam]["primary_0p2pct"]["n_panel"]
        ax.bar(
            np.arange(len(groups)) + k * width,
            [d[g] / max(n_panel, 1) for g in groups],
            width=width,
            color=cols[k],
            label=fam,
        )
    ax.set_xticks(np.arange(len(groups)) + 0.4, groups, fontsize=7)
    ax.set_ylabel("fraction of panel")
    ax.legend(fontsize=7)
    ax.set_title("W3 category-assignment drop rates by status class")
    fig.tight_layout()
    save(fig, "category_drop_rates", dir=io.figs)
    plt.close(fig)

    # 8. matching accuracy (Wilson whiskers + paper reference)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    cfgs = [c for c in JW.CONFIGS if rep["per_config_matching"][c]["n_valid"]]
    accs = np.array([rep["per_config_matching"][c]["accuracy"] for c in cfgs], np.float64)
    cis = np.array([rep["per_config_matching"][c]["wilson_ci95"] for c in cfgs])
    x = np.arange(len(cfgs))
    ax.bar(x, accs, color=cols[: len(cfgs)], yerr=np.abs(cis.T - accs[None, :]), capsize=3)
    for cfg, ref in PAPER_REFERENCE["matching_accuracy"].items():
        if cfg in cfgs:
            ax.scatter([cfgs.index(cfg)], [ref], marker="_", s=300, color="k", zorder=3)
    ax.axhline(rep["chance_matching"], color="grey", ls=":", lw=1)
    ax.set_xticks(x, [CFG_TICK.get(c, c) for c in cfgs], rotation=20, ha="right", fontsize=6)
    ax.set_ylabel("10-way matching accuracy")
    ax.set_title("Matching accuracy, top-100 judged lists (black dash = Der et al. reference)")
    fig.tight_layout()
    save(fig, "matching_accuracy", dir=io.figs)
    plt.close(fig)

    # 9. pairwise win matrix heatmap (constrained layout — colorbar, see fig 6 note)
    fig, ax = plt.subplots(figsize=(4.6, 3.8), layout="constrained")
    mat = np.full((len(JW.CONFIGS), len(JW.CONFIGS)), np.nan)
    for key, cell in rep["pairwise_win_matrix"].items():
        a, b = key.split("-vs-")
        ia, ib = JW.CONFIGS.index(a), JW.CONFIGS.index(b)
        if cell["win_rate_first"] is not None:
            mat[ia, ib] = cell["win_rate_first"]
            mat[ib, ia] = 1.0 - cell["win_rate_first"]
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdBu_r")
    ticks = [CFG_TICK.get(c, c) for c in JW.CONFIGS]
    ax.set_xticks(range(len(JW.CONFIGS)), ticks, rotation=40, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(JW.CONFIGS)), ticks, fontsize=5.5)
    ax.set_title("Pairwise coverage win rate, top-100 judged lists (row beats column)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    save(fig, "pairwise_win_matrix", dir=io.figs)
    plt.close(fig)

    # 10. 5-way mean rank (if w6 present)
    w6 = rep["optional_inputs"]["w6_mean_rank"]
    if isinstance(w6, dict) and "mean_rank" in w6:
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        mr = w6["mean_rank"]
        cfgs = [c for c in JW.CONFIGS if c in mr and np.isfinite(mr[c])]
        ax.bar(range(len(cfgs)), [mr[c] for c in cfgs], color=cols[: len(cfgs)])
        ax.set_xticks(
            range(len(cfgs)),
            [CFG_TICK.get(c, c) for c in cfgs],
            rotation=20,
            ha="right",
            fontsize=6,
        )
        ax.set_ylabel("mean rank (1 = best)")
        ax.invert_yaxis()
        ax.set_title("5-way ranking (W6), top-100 judged lists")
        fig.tight_layout()
        save(fig, "mean_rank", dir=io.figs)
        plt.close(fig)

    # 11. per-turn Δ_cov (rep_ta vs pt_max win/loss) — MF-E bundle ticks + Wilson
    # whiskers + top-100 provenance, matching figs 8-10/12 (#2552 r3 h2-attribution)
    if "per_turn_rep_wins" in rep["pooled"]:
        pooled_d = rep["pooled"]
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        w, losses = pooled_d["per_turn_rep_wins"], pooled_d["per_turn_rep_losses"]
        n_cov = w + losses
        lo = pooled_d["delta_cov_ci95"][0] + 0.5  # Wilson CI on the rep_ta win RATE
        hi = pooled_d["delta_cov_ci95"][1] + 0.5
        yerr = np.array(
            [
                [w - lo * n_cov, hi * n_cov - w],
                [losses - (1 - hi) * n_cov, (1 - lo) * n_cov - losses],
            ]
        ).T
        ax.bar([0, 1], [w, losses], color=[cols[2], cols[3]], yerr=np.abs(yerr), capsize=3)
        ax.set_xticks(
            [0, 1],
            [
                f"{CFG_TICK.get('rep_ta', 'rep_ta')} wins",
                f"{CFG_TICK.get('pt_max', 'pt_max')} wins",
            ],
            fontsize=6,
        )
        ax.set_ylabel("turns")
        ax.set_title(f"Per-turn coverage head-to-head, top-100 judged lists (pooled, n={n_cov})")
        fig.tight_layout()
        save(fig, "delta_cov_perturn", dir=io.figs)
        plt.close(fig)

    # 12. embedding coverage (if produced by P4)
    emb = rep["optional_inputs"]["embedding_coverage"]
    if isinstance(emb, dict) and "per_config" in emb:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        ks = sorted(next(iter(emb["per_config"].values())).keys())
        cfgs = list(emb["per_config"])
        width = 0.8 / len(ks)
        for j, k in enumerate(ks):
            ax.bar(
                np.arange(len(cfgs)) + j * width,
                [emb["per_config"][c][k] for c in cfgs],
                width=width,
                color=cols[j],
                label=f"top-{k.split('_')[-1]}",
            )
        ax.set_xticks(
            np.arange(len(cfgs)) + 0.4,
            [CFG_TICK.get(c, c) for c in cfgs],
            rotation=20,
            ha="right",
            fontsize=6,
        )
        ax.set_ylabel("mean top-k cosine")
        ax.legend(fontsize=7)
        ax.set_title("Embedding coverage (Qwen3-Embedding-8B), top-100 judged lists")
        fig.tight_layout()
        save(fig, "embedding_coverage", dir=io.figs)
        plt.close(fig)

    # 12b. reconstruction FVEs — trainer_2 combined (eval+pool passes, the plan-
    # registered read; eval-pass fallback disclosed) vs replication SAE holdout
    # (#2552 r2 codex M-trainer2-fve; r3 comparator-fve)
    fve_pair = rep.get("reconstruction_fve", {})
    t2_combined = fve_pair.get("trainer2_fve_combined")
    t2_val = t2_combined if t2_combined is not None else fve_pair.get("trainer2_fve_evalpass")
    t2_label = (
        "trainer_2\n(andyrdt, eval+pool passes)"
        if t2_combined is not None
        else "trainer_2\n(andyrdt, eval-pass only)"
    )
    fve_vals = [t2_val, fve_pair.get("replication_sae_val_var_fve")]
    if any(v is not None for v in fve_vals):
        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        labels = [t2_label, "replication TA SAE\n(holdout val)"]
        vals = [np.nan if v is None else float(v) for v in fve_vals]
        ax.bar(np.arange(2), vals, color=[cols[0], cols[1]])
        ax.set_xticks(np.arange(2), labels, fontsize=7)
        ax.set_ylabel("var-FVE")
        ax.set_title("Reconstruction FVE (global fp64 per-dim variance)")
        fig.tight_layout()
        save(fig, "reconstruction_fve", dir=io.figs)
        plt.close(fig)

    # 12c. W7 pinned-judge calibration — the plan-registered kappa-table figure with
    # realized per-cell counts + per-category agreement/drop-rate bars (#2552 r3
    # w7-calibration; plan Figures list "judge-agreement (kappa) table figure")
    w7 = rep["optional_inputs"].get("w7_calibration")
    if isinstance(w7, dict) and any(k in w7 for k in ("w3", "w4", "w5")):
        fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(8.4, 3.2), width_ratios=[1.1, 1.0])
        ax_t.axis("off")

        def _f3(v) -> str:
            return "n/a" if v is None else f"{float(v):.3f}"

        rows = []
        for inst in ("w3", "w4", "w5"):
            d = w7.get(inst)
            if not isinstance(d, dict):
                continue
            ci = d.get("agreement_wilson_ci95") or [float("nan"), float("nan")]
            rows.append(
                [
                    inst,
                    f"{d.get('n_both_valid', 0)}/{d.get('n_sampled', 0)}",
                    f"{_f3(d.get('raw_agreement'))} [{ci[0]:.2f},{ci[1]:.2f}]",
                    _f3(d.get("kappa")),
                    _f3(d.get("drop_rate_cal")),
                ]
            )
        tbl = ax_t.table(
            cellText=rows,
            colLabels=["instrument", "n valid/sampled", "agreement [CI95]", "kappa", "drop rate"],
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.5)
        ax_t.set_title("Judge agreement vs pinned judge (W7)", fontsize=8)
        cat_cells = {
            k.split("=", 1)[1]: v
            for k, v in (w7.get("w3", {}).get("cells") or {}).items()
            if k.startswith("category=")
        }
        if cat_cells:
            cats_sorted = sorted(cat_cells)
            x = np.arange(len(cats_sorted))

            def _v(c: str, key: str) -> float:
                v = cat_cells[c].get(key)
                return np.nan if v is None else float(v)

            agree = [_v(c, "agreement") for c in cats_sorted]
            drop = [_v(c, "drop_rate_cal") for c in cats_sorted]
            ns = [cat_cells[c].get("n_both_valid", 0) for c in cats_sorted]
            ax_b.bar(x - 0.2, agree, width=0.4, color=cols[0], label="agreement")
            ax_b.bar(x + 0.2, drop, width=0.4, color=cols[3], label="cal drop rate")
            ax_b.set_xticks(x, [f"{c}\n(n={n})" for c, n in zip(cats_sorted, ns)], fontsize=6)
            ax_b.set_ylim(0, 1)
            ax_b.legend(fontsize=6)
            ax_b.set_title("W3 per-category (thin cells: descriptive)", fontsize=8)
        else:
            ax_b.axis("off")
        fig.tight_layout()
        save(fig, "w7_calibration", dir=io.figs)
        plt.close(fig)

    # 13. per-covariate scatters with decile medians (primary panel, per dictionary)
    for fam in fams:
        dv = _load_dv(args, io, fam, FLOOR_PRIMARY)
        cats_, _stats = _labels_status(args, io, fam, dv.feat_ids)
        idx = np.flatnonzero(dv.panel)
        blocks, _notes = _candidate_blocks(dv, idx, cats_, include_category=False)
        names = sorted(blocks)
        ncol = 4
        nrow = int(np.ceil(len(names) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow))
        axf = np.atleast_1d(axes).ravel()
        r2 = dv.r2[idx]
        for ax, name in zip(axf, names, strict=False):
            xv = blocks[name][:, 0]
            ax.scatter(xv, r2, s=2, alpha=0.25, color=cols[0], edgecolors="none")
            dq = np.quantile(xv, np.linspace(0, 1, 11))
            for lo_, hi_ in itertools.pairwise(dq):
                m = (xv >= lo_) & (xv <= hi_)
                if m.sum() >= 5:
                    ax.scatter([(lo_ + hi_) / 2], [np.median(r2[m])], color=cols[3], s=14, zorder=3)
            ax.set_title(name, fontsize=7)
        for ax in axf[len(names) :]:
            ax.axis("off")
        fig.suptitle(f"{fam}: per-feature R² vs covariate (z) — decile medians")
        fig.tight_layout()
        save(fig, f"covariate_scatters_{fam}", dir=io.figs)
        plt.close(fig)
    print(f"[figures] wrote figures under {io.figs}", flush=True)


# ── upload ───────────────────────────────────────────────────────────────────────


def phase_upload(args) -> None:
    """Persist per-(step, dictionary) draw matrices to HF analysis_tensors/ladder/."""
    print("[phase=upload]", flush=True)
    io = _paths(args)
    files = sorted(io.draws.glob("*.npz"))
    if args.smoke or args.skip_upload:
        logger.warning(
            "[upload] SKIPPED (smoke/skip_upload) — %d draw npz staged, loud", len(files)
        )
        return
    assert files, f"no draw matrices staged under {io.draws} — run --phase ladder first"
    from huggingface_hub import HfApi

    import issue779_common as C
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    prefix = f"{args.hf_prefix}/analysis_tensors/ladder"
    res = upload_dir_sharded(
        io.draws,
        C.HF_DATA_REPO,
        prefix,
        repo_type="dataset",
        shard_glob="*.npz",
        verify=True,
        delete_local=False,
        # presence+size resume is a stale-mirror channel for same-size re-runs (#2225
        # class): draw matrices are cheap to re-push, so always overwrite
        resume_skip=False,
    )
    if not res.rerouted:
        expected = [f"{prefix}/{q.name}" for q in files]
        missing = hub.verify_repo_paths_uploaded(
            HfApi(), C.HF_DATA_REPO, expected, path_in_repo=prefix
        )
        assert not missing, f"[upload] draw-matrix verify FAILED — missing: {missing}"
    print(f"[upload] {len(files)} draw matrices -> {prefix}", flush=True)


# ── prep + smoke ─────────────────────────────────────────────────────────────────


def phase_prep(args) -> None:
    """Resolve + verify every input; write an inputs manifest (fail-loud on misses)."""
    print("[phase=prep]", flush=True)
    io = _paths(args)
    eval_dir = _ensure_eval_inputs(args, io)
    manifest: dict = {"eval_inputs_dir": str(eval_dir), "checks": {}, **_meta("p3-prep")}
    for fam in DICTS:
        dv = _load_dv(args, io, fam, FLOOR_PRIMARY)
        cats, stats = _labels_status(args, io, fam, dv.feat_ids)
        n_cc = int((dv.panel & np.array([c in CATEGORIES for c in cats], bool)).sum())
        manifest["checks"][fam] = {
            "n_features": int(dv.feat_ids.size),
            "n_panel_primary": int(dv.panel.sum()),
            "n_complete_case": n_cc,
            "status_counts": {
                g: int((stats == g).sum()) for g in ("valid", *STATUS_GROUPS, "unjudged")
            },
        }
    for p in (io.dere_in / "matching_perturn.json", io.dere_in / "pairwise_perturn.json"):
        manifest["checks"][p.name] = p.exists()
    _write_json(io.work / "inputs_manifest.json", manifest)
    print(f"[prep] manifest -> {io.work / 'inputs_manifest.json'}", flush=True)


def _synth_inputs(args, io) -> None:
    """Build tiny smoke fixtures in the producers' EXACT schemas (unit-1/2 writers).

    The mat-family DV join reads the REAL committed #2476 union npzs (real banked-artifact
    schema); covariates / judge outputs / replication rows are synthesized."""
    rng = np.random.default_rng(TASK_ID)
    for d in (io.eval_in, io.agg_in, io.raw_w3, io.dere_in):
        d.mkdir(parents=True, exist_ok=True)
    # rep_ta perfeature — the P1.4 producer's LITERAL key set (turnsae_der.
    # phase_perfeature_r2 savez_atomic call; NO tier / r2 / activity keys — a
    # consumer-keyed fixture self-certifies, #2379 class / #2552 r2 g6-C1)
    n_rep, width_rep = 300, 32768
    feat_ids = np.sort(rng.choice(width_rep, size=n_rep, replace=False)).astype(np.int64)
    counts = np.concatenate(
        [rng.integers(240, 5000, size=n_rep - 60), rng.integers(1200, 20000, size=60)]
    ).astype(np.int64)
    r2 = rng.beta(2, 5, size=n_rep)
    np.savez(
        io.eval_in / "perfeature_rep.npz",
        feat_ids=feat_ids,
        r2_map=r2,
        spearman=r2 * 0.9,
        ss_tot=rng.random(n_rep),
        r2_ib=r2 * 0.5,
        r2_trainmean=np.zeros(n_rep),
        r2_lmsys=r2,
        r2_wildchat=r2,
        r2_corpusfold=r2 * 0.8,
        null_r2_map=np.zeros(n_rep),
        null_r2_ib=np.zeros(n_rep),
        counts=counts,
        alive_f240=(counts >= 240),
        alive_f1200=(counts >= 1200),
        shuffle_seeds=np.asarray([0, 1, 2], np.int64),
        n_fit_rows=np.int64(120000),
        n_holdout=np.int64(2000),
    )
    # measured P1 regime (producer: turnsae _measured_update) — carries both
    # reconstruction FVEs for the replication doc (#2552 r2 codex M-trainer2-fve)
    _write_json(
        io.eval_in / "regime_measured.json",
        {
            "trainer2_fve_evalpass": 0.71,
            "trainer2_fve_poolpass": 0.69,
            "trainer2_fve_combined": 0.70,
            "rep_sae_val_var_fve": 0.62,
            "rep_sae_val_nmse": 0.38,
        },
    )
    # covariates per family — full-width arrays (producer: phase_covariates key set)
    widths = {"rep_ta": width_rep, "mat_k100": 65536, "mat_k200": 65536}
    for fam, w in widths.items():
        match_id = rng.integers(0, 131072, size=w)
        rb_cos = rng.random((w, 3)).astype(np.float64) * 0.4
        np.savez(
            io.eval_in / f"covariates_{fam}.npz",
            counts=rng.integers(0, 3000, size=w),
            n_fit_rows=np.int64(120000),
            act_var=rng.random(w),
            act_mean_when_active=rng.random(w) * 3,
            share_lmsys=rng.random(w),
            coact_mean=rng.random(w) * 40,
            decoder_norm=rng.random(w) + 0.5,
            footprint_top20=rng.random(w),
            match_cos=rng.random(w),
            match_id=match_id,
            consistency_twin=rng.random(w),
            pca_align_frac=rng.random(w),
            pca_best_rank=rng.integers(1, 256, size=w).astype(np.float64),
            rb_cos=rb_cos,
            rb_traits=np.asarray(["evil", "hallucination", "sycophancy"]),
            rb_cos_max=rb_cos.max(axis=1),
        )
    # w3 raw draws (save_raw shape) + aggregates derived through the REAL reduce
    fields = [f for f in JW.APP_D_FIELDS]
    all_scores: dict[str, dict] = {}
    for fam in DICTS:
        if fam == "rep_ta":
            fam_feats = feat_ids
        else:
            z = np.load(io.union_paths[fam])
            alive = np.asarray(z[f"alive_f{FLOOR_PRIMARY}"]).astype(bool)
            fam_feats = np.asarray(z["feat_ids"], np.int64)[alive]
        for f in fam_feats:
            u = rng.random()
            cid = f"w3-{fam}-f{int(f)}__00000__00"
            if u < 0.84:
                all_scores[cid] = {"stop_reason": "end_turn", "field": str(rng.choice(fields))}
            elif u < 0.90:
                all_scores[cid] = {"stop_reason": "end_turn", "field": "none"}
            elif u < 0.94:
                all_scores[cid] = {"stop_reason": "end_turn", "verdict": "garbled"}
            elif u < 0.96:
                all_scores[cid] = {"stop_reason": "max_tokens"}
            elif u < 0.98:
                all_scores[cid] = {"stop_reason": "refusal"}
            else:
                all_scores[cid] = {"error": True, "transport": True, "reason": "synthetic 529"}
    _write_json(io.raw_w3 / "judge_raw_w3.json", {"all_scores": all_scores})
    reissue = {
        cid.replace("__00000__00", "__00001__00"): {
            "stop_reason": "end_turn",
            "field": str(rng.choice(fields)),
        }
        for cid, d in list(all_scores.items())[:5]
        if d.get("error")
    }
    _write_json(io.raw_w3 / "judge_raw_w3_syncreissue.json", {"all_scores": reissue})
    parser = JW.WAVE_PARSERS["w3"]
    per = JW.reduce_all_scores(all_scores, parser)
    for item, rec in JW.reduce_all_scores(reissue, parser).items():
        if rec["class"] == "valid" or item not in per:
            per[item] = rec
    for fam in DICTS:
        pref = f"w3-{fam}-f"
        valid_recs = {
            item[len(pref) :]: rec
            for item, rec in per.items()
            if item.startswith(pref) and rec["class"] == "valid"
        }
        # producer parity (phase_w3): semantic-none features are EXCLUDED from
        # `assignments` (counted in n_none) — the fixture must mirror that, or the
        # consumer's none-derivation is never exercised (#2552 r2 g6-M2)
        assigned = {
            fid: {"field": rec["value"][0], "category": rec["value"][1]}
            for fid, rec in valid_recs.items()
            if rec["value"][1] != "none"
        }
        _write_json(
            io.agg_in / f"w3_categories_{fam}.json",
            {
                "dictionary": fam,
                "n_assigned": len(assigned),
                "n_none": sum(1 for r in valid_recs.values() if r["value"][1] == "none"),
                "n_dropped": sum(
                    1
                    for item, rec in per.items()
                    if item.startswith(pref) and rec["class"] != "valid"
                ),
                "assignments": assigned,
            },
        )
    # replication fixtures (producer shapes: phase_w4 / phase_w5 / phase_w6 writers)
    n_turns = 24
    row_ids = np.sort(rng.choice(20000, size=n_turns, replace=False))
    # G2 realized subset != committed regime — exercises the g2-derivation path (#2552 r2):
    # per-turn rows exist ONLY for the realized ids; regime.json keeps the committed 24
    realized_ids = row_ids[: n_turns - 4]
    acc_true = {"pt_max": 0.9, "pt_sum": 0.8, "rep_ta": 0.7, "mat_k100": 0.6, "mat_k200": 0.55}
    m_rows = []
    for cfg in JW.CONFIGS:
        for rid in realized_ids:
            valid = rng.random() > 0.08
            row = {
                "row_id": int(rid),
                "config": cfg,
                "item_id": f"w4-{cfg}-r{int(rid)}",
                "valid": bool(valid),
                "gold": "A",
                "n_desc": 10,
                "n_missing_desc": 0,
            }
            if valid:
                row["choice"] = "A"
                row["correct"] = bool(rng.random() < acc_true[cfg])
            else:
                row["drop_class"] = "parse_fail"
            m_rows.append(row)
    _write_json(
        io.dere_in / "matching_perturn.json",
        {
            "rows": m_rows,
            "summary": {"chance": 0.1, "n_eval_realized": int(realized_ids.size)},
        },
    )
    p_rows = []
    for a_i, a in enumerate(JW.CONFIGS):
        for b in JW.CONFIGS[a_i + 1 :]:
            for rid in realized_ids:
                valid = rng.random() > 0.08
                row = {
                    "row_id": int(rid),
                    "pair": f"{a}-vs-{b}",
                    "list1": a,
                    "list2": b,
                    "valid": bool(valid),
                }
                if valid:
                    win_first = rng.random() < (
                        0.35 if {a, b} == {"rep_ta", "pt_max"} and a == "pt_max" else 0.5
                    )
                    if {a, b} == {"rep_ta", "pt_max"}:
                        row["winner"] = "rep_ta" if rng.random() < 0.65 else "pt_max"
                    else:
                        row["winner"] = a if win_first else b
                else:
                    row["drop_class"] = "parse_fail"
                p_rows.append(row)
    _write_json(io.dere_in / "pairwise_perturn.json", {"rows": p_rows, "summary": {}})
    _write_json(
        io.agg_in / "w6_ranking_perturn.json",
        {"rows": [], "summary": {"mean_rank": {c: float(i + 1) for i, c in enumerate(JW.CONFIGS)}}},
    )
    # W7 calibration fixture in the judge writer's shape (instruments + per-cell
    # tables, joint + marginal w3 cells) so the r3 kappa-table figure leg executes
    # under smoke rather than silently skipping (#2552 r3 w7-calibration)
    _write_json(
        io.agg_in / "w7_calibration.json",
        {
            "instruments": {
                inst: {
                    "n_sampled": 12,
                    "n_both_valid": 10,
                    "n_cal_dropped": 2,
                    "drop_rate_cal": 2 / 12,
                    "raw_agreement": 0.8,
                    "agreement_wilson_ci95": [0.49, 0.94],
                    "kappa": 0.6,
                    "cells": (
                        {
                            "family=rep_ta|category=behavioral": {
                                "n_sampled": 6,
                                "n_both_valid": 5,
                                "drop_rate_cal": 1 / 6,
                                "agreement": 0.8,
                                "agreement_wilson_ci95": [0.38, 0.96],
                                "kappa": 0.55,
                            },
                            "family=rep_ta": {
                                "n_sampled": 12,
                                "n_both_valid": 10,
                                "drop_rate_cal": 2 / 12,
                                "agreement": 0.8,
                                "agreement_wilson_ci95": [0.49, 0.94],
                                "kappa": 0.6,
                            },
                            "category=behavioral": {
                                "n_sampled": 6,
                                "n_both_valid": 5,
                                "drop_rate_cal": 1 / 6,
                                "agreement": 0.8,
                                "agreement_wilson_ci95": [0.38, 0.96],
                                "kappa": 0.55,
                            },
                            "category=topical": {
                                "n_sampled": 6,
                                "n_both_valid": 5,
                                "drop_rate_cal": 1 / 6,
                                "agreement": 0.6,
                                "agreement_wilson_ci95": [0.27, 0.86],
                                "kappa": None,
                            },
                        }
                        if inst == "w3"
                        else {}
                    ),
                    "cal_judge_model": "claude-sonnet-4-5-20250929",
                }
                for inst in ("w3", "w4", "w5")
            },
            "n_per_instrument": 200,
        },
    )
    _write_json(
        io.regime_path,
        {
            "eval_ids": [int(r) for r in row_ids],
            "eval_ids_sha256": "smoke-fixture",
            "n_eval": n_turns,
            "n_lmsys": int(n_turns // 2),
            "n_wildchat": int(n_turns - n_turns // 2),
        },
    )
    _write_json(
        io.agg_in / "g2_decision.json",
        {
            "eval_ids": [int(r) for r in realized_ids],
            "eval_ids_sha256": "smoke-fixture-g2",
            "n_eval_realized": int(realized_ids.size),
            "descoped": True,
            "rep_panel_ok": True,
        },
    )
    prov = rng.integers(0, 2, size=20001).astype(np.uint8)
    np.save(io.prov_path, prov)
    print("[smoke] synthetic fixtures written", flush=True)


def phase_smoke(args) -> None:
    """Tiny-real smoke: synthesize fixtures, then run every phase through the SAME
    production entrypoints (upload skip-loud). Draws shrink via --draws."""
    args.smoke = True
    if args.draws == N_DRAWS:
        args.draws = 500  # slice-size parameterization only — same code path
    io = _paths(args)
    _synth_inputs(args, io)
    for name in PHASE_ORDER:
        PHASES[name](args)
    checks = {
        "ladder_steps": (io.ladder_out / "ladder_steps.json").exists(),
        "category_reads": (io.cat_out / "category_reads.json").exists(),
        "replication_stats": (io.dere_out / "replication_stats.json").exists(),
        "hero_figure": (io.figs / "ladder_hero.png").exists(),
        "draw_matrices": len(list(io.draws.glob("*.npz"))),
    }
    assert all(bool(v) for v in checks.values()), checks
    _write_json(io.work / "smoke_done.json", {"checks": checks, **_meta("p3-smoke")})
    print(f"[smoke] PASS {checks}", flush=True)


def phase_all(args) -> None:
    for name in PHASE_ORDER:
        PHASES[name](args)


PHASE_ORDER = ("prep", "ladder", "category", "replication", "figures", "upload")
PHASES = {
    "prep": phase_prep,
    "ladder": phase_ladder,
    "category": phase_category,
    "replication": phase_replication,
    "figures": phase_figures,
    "upload": phase_upload,
    "all": phase_all,
    "smoke": phase_smoke,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", required=False, choices=sorted(PHASES), help="phase to run")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_2552" / "ladder",
        help="scratch/work root (smoke diverts everything under <out-root>/smoke)",
    )
    ap.add_argument("--hf-prefix", default="issue2552_turnsae")
    ap.add_argument(
        "--p1-eval-dir",
        type=Path,
        default=None,
        help="local dir holding perfeature_rep.npz + covariates_*.npz (else HF fetch)",
    )
    ap.add_argument(
        "--judge-agg-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_2552" / "judge_aggregates",
    )
    ap.add_argument(
        "--dere-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_2552" / "dere_repl",
    )
    ap.add_argument(
        "--judge-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_2552" / "judge",
        help="judge driver work root (w3 raw draws + prov staging)",
    )
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps({"order": list(PHASE_ORDER), "registry": sorted(PHASES)}))
        return
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # deferred imports executed explicitly (smoke-architecture Axis 1)
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
        from scipy.stats import rankdata  # noqa: F401

        import issue779_common  # noqa: F401
        from explore_persona_space.atomic_io import savez_atomic, write_json_atomic  # noqa: F401
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )
        from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded  # noqa: F401

        _fig_env()
        print("import-check OK")
        return
    assert args.phase, "--phase required (or --list-phases / --import-check)"
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
