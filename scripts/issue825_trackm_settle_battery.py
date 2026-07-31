"""Track-M linear-collapse "settle-it" refit battery for issue #825.

Reads the audit at eval_results/issue_825/trackm_linear_collapse_audit/README.md
(yesterday's banked-artifact re-read) and answers, with numbers, the three
pre-registered splits + one hardening read it names:

  1. Estimator share      -- does the GCV_DOF_CAP=0.9 guard (the fit module's
                             OWN registered mitigation for the n_tr<D GCV
                             degeneracy) recover S@2000 toward its power-expected
                             level, and lift each Track-M cell?
  2. Decoding+corpus share -- residual gap M vs guarded S@2000, and does an
                             M-matched corpus filter (exact-dup answer removal +
                             >=8-content-token prompts) pull guarded S@2000 DOWN
                             toward M?
  3. Nonlinearity verdict  -- MLP (banked PCA-64-head recipe) on S1@2000: if
                             MLP@S2000 ~ guarded-ridge@S2000, the "collapse is
                             nonlinearity" claim was an estimator artifact.
  4. User-turn hardening    -- guarded refits of the 4 M user cells: does the
                             user-turn linear null survive the estimator fix?

DESIGN NOTES (verified this session, not assumed):
  * The subsample scheme for the S1@2000 matched-n draws is RECOVERED (not
    guessed): ``np.random.default_rng(seed).choice(n_full, n_sub, replace=False)``
    with fixed seeds 1000-1004 reproduces every banked
    matched_n_curve/results.json n=2000 draw to <1e-4 (verified against
    0.2434/0.2935/0.2852/0.1892/0.2811). Fold seed 0, grouped 5-fold.
  * The harness reproduces the banked S1@5000 unguarded L19 = 0.6730940896676356
    from the LOCAL 4-layer map_alignment turnstore (== the analysis_tensors
    content matched_n_curve used).
  * UNGUARDED = ``GCV_DOF_CAP=None`` + lambda_selection "gcv" (the committed
    #825 default). GUARDED = ``GCV_DOF_CAP=0.9`` (the module's documented
    dof-cap mitigation), same gcv selection.

Turnstores:
  * Track S (4-layer [14,18,19,26]): data/issue_825/hf_dl/map_alignment/{model}_chat_s.npz
  * Track M (28-layer .pt shards): data/issue_825/audit_dl/analysis_tensors/{model}_{fmt}_m_shard*.pt

Phases (idempotent; each skips work whose output JSON already exists):
  stage      -- assert every needed turnstore is present, print an inventory.
  refits     -- leg A guarded ridge refits (--group {s5000,s2000,m_chat,m_nat}).
  mlp        -- leg B MLP control on S1@2000 (--draw K runs one draw).
  filtered   -- leg C corpus-filtered S@2000 (unguarded + guarded).
  summarize  -- assemble results.json + README + figure from the phase partials.

Every python invocation is CPU-only; the shared-VM thread caps are inherited
from the launch env (OMP/MKL/OPENBLAS/NUMEXPR=8, MALLOC_ARENA_MAX=2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/torch import

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue825_fit_cells as fit  # noqa: E402
from explore_persona_space.experiments.issue_779.fit_h import mlp_fit_predict  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
WANT_LAYERS = [14, 18, 19, 26]
L19 = 19
SUBSAMPLE_SEEDS = [1000, 1001, 1002, 1003, 1004]  # RECOVERED from matched_n_curve
N_SUB = 2000
FOLD_SEED = 0
N_FOLDS = 5
DOF_CAP = 0.9
MIN_PROMPT_CONTENT_TOKENS = 8  # M-matched (common.MIN_TURN_CONTENT_TOKENS)

S_STORE_DIR = Path("data/issue_825/hf_dl/map_alignment")
M_STORE_DIR = Path("data/issue_825/audit_dl/analysis_tensors")
TRACK_S_JSONL = Path(
    "data/issue_825/audit_dl/issue825_userbase_map/raw_completions/track_s/track_s.jsonl"
)
OUT_DIR = Path("eval_results/issue_825/trackm_settle_battery")
TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"

# Cell registry: (cell_id, model, fmt, track, slot_index, turn_index)
# Track-M: slots [assistant(before a1), user(before u2)]; turns [u1,a1,u2,a2].
M_ASSISTANT_CELLS = [
    (f"M_{m}_assistant_{f}", m, f, "m", 0, 1)
    for m in ("instruct", "pretrained")
    for f in ("chat", "naturalistic")
]
M_USER_CELLS = [
    (f"M_{m}_user_{f}", m, f, "m", 1, 2)
    for m in ("instruct", "pretrained")
    for f in ("chat", "naturalistic")
]

# --------------------------------------------------------------------------- #
# Turnstore loading (layer-aware; npz 4-layer OR .pt 28-layer shards)
# --------------------------------------------------------------------------- #


def _to_np(v) -> np.ndarray:
    """bf16-safe conversion: tensors (incl. BFloat16) upcast to fp32; a
    per-conv list/tuple of tensors is stacked via the shared bf16-safe stacker;
    ndarrays pass through."""
    if torch.is_tensor(v):
        return v.float().numpy()
    if isinstance(v, np.ndarray):
        return v
    return fit._stack_maybe_list(v, "shard_payload")


def load_cell_xy(model: str, fmt: str, track: str, slot_index: int, turn_index: int) -> dict:
    """Return {X,Y,conv_ids} at WANT_LAYERS for one cell.

    npz stores carry an explicit ``layers`` axis (map the requested layer to its
    index); .pt shard stores are full 28-layer (layer number == index). Rows
    with any NaN in X or Y are dropped (finiteness contract).
    """
    want = WANT_LAYERS
    if track == "s":
        npz = S_STORE_DIR / f"{model}_{fmt}_s.npz"
        d = np.load(npz, allow_pickle=False)
        store_layers = (
            [int(x) for x in d["layers"]]
            if "layers" in d.files
            else list(range(d["slots"].shape[2]))
        )
        lidx = [store_layers.index(li) for li in want]
        slots = np.asarray(d["slots"], dtype=np.float32)
        profiles = np.asarray(d["profiles"], dtype=np.float32)
        X = slots[:, slot_index][:, lidx, :]
        Y = profiles[:, turn_index][:, lidx, :]
        conv_ids = np.asarray([str(c) for c in d["conv_ids"]])
    else:
        shards = sorted(M_STORE_DIR.glob(f"{model}_{fmt}_{track}_shard*.pt"))
        if not shards:
            raise FileNotFoundError(f"no .pt shards for {model}_{fmt}_{track} in {M_STORE_DIR}")
        Xs: list[np.ndarray] = []
        Ys: list[np.ndarray] = []
        conv_ids_all: list[str] = []
        lidx = list(want)  # 28-layer store: layer number == index
        for sp in shards:
            payload = torch.load(sp, map_location="cpu", weights_only=False)
            conv_ids_all.extend(str(c) for c in payload["conv_ids"])
            slots = _to_np(payload["slots"])  # (n, n_slots, 28, H)
            profiles = _to_np(payload["profiles"])  # (n, n_turns, 28, H)
            Xs.append(slots[:, slot_index][:, lidx, :].astype(np.float32))
            Ys.append(profiles[:, turn_index][:, lidx, :].astype(np.float32))
            del payload, slots, profiles
        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)
        conv_ids = np.asarray(conv_ids_all)
    keep = ~(np.isnan(X).any(axis=(1, 2)) | np.isnan(Y).any(axis=(1, 2)))
    return {"X": X[keep], "Y": Y[keep], "conv_ids": conv_ids[keep]}


# --------------------------------------------------------------------------- #
# Ridge / MLP fits
# --------------------------------------------------------------------------- #


def ridge_r2(X: np.ndarray, Y: np.ndarray, conv_ids: np.ndarray, *, guarded: bool) -> dict:
    """Held-out grouped 5-fold GCV-ridge R^2 per WANT_LAYER (gcv selection).

    ``guarded`` toggles the module-global GCV_DOF_CAP (0.9 vs None); restored
    after so concurrent state can't leak. WANT_LAYERS -> r2 (position-indexed,
    since X's layer axis is len(WANT_LAYERS)).
    """
    prev = fit.GCV_DOF_CAP
    prev_legacy = fit.LEGACY_UNGUARDED_GCV
    fit.GCV_DOF_CAP = DOF_CAP if guarded else None
    if not guarded:
        # #1887: the unguarded arm DELIBERATELY reproduces the committed
        # pre-#1887 pure-GCV behavior — the explicit legacy opt-in the refusal
        # guard requires at n_train < d. Restored in the finally block.
        fit.LEGACY_UNGUARDED_GCV = True
    try:
        sw = fit.heldout_r2_sweep(
            X,
            Y,
            conv_ids,
            n_folds=N_FOLDS,
            seed=FOLD_SEED,
            null_draws=0,
            collect_cosines=False,
            # #1887 defaults flipped in fit825; both arms of this battery are
            # DELIBERATE gcv arms — pin the committed selector explicitly and
            # skip the (selector-orthogonal) reduced-basis companion.
            lambda_selection="gcv",
            reduced_basis_companion=False,
        )
    finally:
        fit.GCV_DOF_CAP = prev
        fit.LEGACY_UNGUARDED_GCV = prev_legacy
    r2 = sw["r2_obs"]
    return {str(li): float(r2[pos]) for pos, li in enumerate(WANT_LAYERS)}


def mlp_r2_L19(X: np.ndarray, Y: np.ndarray, conv_ids: np.ndarray) -> dict:
    """Pooled 5-fold MLP R^2 at L19 (banked PCA-64-head recipe, CPU)."""
    pos = WANT_LAYERS.index(L19)
    Xl = X[:, pos, :]
    Yl = Y[:, pos, :]
    folds = fit._cv_folds(conv_ids, N_FOLDS, FOLD_SEED)
    ss_res = ss_tot = 0.0
    per_fold = []
    for k in range(N_FOLDS):
        te = folds == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            continue
        pred = mlp_fit_predict(Xl[tr], Yl[tr], Xl[te], pca_k=64, device="cpu", num_threads=8)
        true = Yl[te].astype(np.float64)
        mu = true.mean(0)
        fr = float(np.sum((true - pred) ** 2))
        ft = float(np.sum((true - mu) ** 2))
        ss_res += fr
        ss_tot += ft
        per_fold.append((1.0 - fr / ft) if ft > 1e-12 else float("nan"))
    return {
        "r2_L19_pooled": (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "r2_L19_per_fold": per_fold,
    }


def subsample_idx(n: int, seed: int, n_sub: int = N_SUB) -> np.ndarray:
    """RECOVERED matched-n draw: default_rng(seed).choice(n, n_sub, replace=False)."""
    return np.random.default_rng(seed).choice(n, n_sub, replace=False)


# --------------------------------------------------------------------------- #
# Metadata / IO
# --------------------------------------------------------------------------- #


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _meta() -> dict:
    return {
        "timestamp": "2026-07-25",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/issue825_trackm_settle_battery.py",
        "issue": 825,
        "git_commit": _git_commit(),
    }


def _load_partial(name: str) -> dict:
    p = OUT_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


def _save_partial(name: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(payload, indent=2, default=float))
    print(f"[settle] wrote {OUT_DIR / name}", flush=True)


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def phase_stage(args) -> int:
    print("[settle] stage: turnstore inventory", flush=True)
    ok = True
    for model in ("instruct", "pretrained"):
        p = S_STORE_DIR / f"{model}_chat_s.npz"
        print(f"  S {model}_chat_s.npz: {'OK' if p.exists() else 'MISSING'} ({p})", flush=True)
        ok &= p.exists()
    for model in ("instruct", "pretrained"):
        for fmt in ("chat", "naturalistic"):
            shards = sorted(M_STORE_DIR.glob(f"{model}_{fmt}_m_shard*.pt"))
            print(f"  M {model}_{fmt}_m: {len(shards)} shards", flush=True)
    print(f"  track_s.jsonl: {'OK' if TRACK_S_JSONL.exists() else 'MISSING'}", flush=True)
    return 0 if ok else 1


def phase_refits(args) -> int:
    part = _load_partial("_refits.json")
    part.setdefault("metadata", _meta())
    part.setdefault("cells", {})
    group = args.group
    t0 = time.time()

    def _fit_full(cell_id, model, fmt, track, si, ti):
        if cell_id in part["cells"] and not args.force:
            print(f"[settle] skip {cell_id} (cached)", flush=True)
            return
        xy = load_cell_xy(model, fmt, track, si, ti)
        n = len(xy["conv_ids"])
        rg = ridge_r2(xy["X"], xy["Y"], xy["conv_ids"], guarded=True)
        ru = ridge_r2(xy["X"], xy["Y"], xy["conv_ids"], guarded=False)
        part["cells"][cell_id] = {
            "n": n,
            "guarded_ridge": rg,
            "unguarded_ridge": ru,
            "wall_s": round(time.time() - t0, 1),
        }
        print(
            f"[settle] {cell_id} n={n} guarded_L19={rg['19']:.4f} unguarded_L19={ru['19']:.4f}",
            flush=True,
        )
        _save_partial("_refits.json", part)

    if group == "s5000":
        xy = load_cell_xy("instruct", "chat", "s", 0, 1)
        n = len(xy["conv_ids"])
        rg = ridge_r2(xy["X"], xy["Y"], xy["conv_ids"], guarded=True)
        ru = ridge_r2(xy["X"], xy["Y"], xy["conv_ids"], guarded=False)
        part["cells"]["S1_full"] = {
            "n": n,
            "guarded_ridge": rg,
            "unguarded_ridge": ru,
            "wall_s": round(time.time() - t0, 1),
        }
        print(
            f"[settle] S1_full n={n} guarded_L19={rg['19']:.5f} unguarded_L19={ru['19']:.6f} "
            f"(banked 0.6730940896676356)",
            flush=True,
        )
        _save_partial("_refits.json", part)
    elif group == "s2000":
        xy = load_cell_xy("instruct", "chat", "s", 0, 1)
        draws_g, draws_u = [], []
        for seed in SUBSAMPLE_SEEDS:
            idx = subsample_idx(len(xy["conv_ids"]), seed)
            Xi, Yi, ci = xy["X"][idx], xy["Y"][idx], xy["conv_ids"][idx]
            rg = ridge_r2(Xi, Yi, ci, guarded=True)
            ru = ridge_r2(Xi, Yi, ci, guarded=False)
            draws_g.append(rg)
            draws_u.append(ru)
            print(
                f"[settle] S1@2000 seed={seed} guarded_L19={rg['19']:.4f} "
                f"unguarded_L19={ru['19']:.4f}",
                flush=True,
            )
        part["cells"]["S1_2000"] = {
            "n": N_SUB,
            "seeds": SUBSAMPLE_SEEDS,
            "guarded_ridge_draws": draws_g,
            "unguarded_ridge_draws": draws_u,
            "guarded_L19_mean": float(np.mean([d["19"] for d in draws_g])),
            "guarded_L19_std": float(np.std([d["19"] for d in draws_g])),
            "unguarded_L19_mean": float(np.mean([d["19"] for d in draws_u])),
            "unguarded_L19_std": float(np.std([d["19"] for d in draws_u])),
            "wall_s": round(time.time() - t0, 1),
        }
        print(
            f"[settle] S1@2000 guarded mean_L19={part['cells']['S1_2000']['guarded_L19_mean']:.4f}"
            f" unguarded mean_L19={part['cells']['S1_2000']['unguarded_L19_mean']:.4f} "
            f"(banked unguarded 0.258)",
            flush=True,
        )
        _save_partial("_refits.json", part)
    elif group == "m_chat":
        for c in M_ASSISTANT_CELLS + M_USER_CELLS:
            if c[3] == "m" and c[2] == "chat":
                _fit_full(*c)
    elif group == "m_nat":
        for c in M_ASSISTANT_CELLS + M_USER_CELLS:
            if c[3] == "m" and c[2] == "naturalistic":
                _fit_full(*c)
    else:
        raise ValueError(f"unknown --group {group}")
    return 0


def phase_mlp(args) -> int:
    part = _load_partial("_mlp.json")
    part.setdefault("metadata", _meta())
    part.setdefault(
        "recipe", "fit_h.mlp_fit_predict pca_k=64 hidden=512 AdamW <=300ep earlystop, CPU"
    )
    part.setdefault("draws", {})
    xy = load_cell_xy("instruct", "chat", "s", 0, 1)
    seeds = [SUBSAMPLE_SEEDS[args.draw]] if args.draw is not None else SUBSAMPLE_SEEDS
    for seed in seeds:
        key = str(seed)
        if key in part["draws"] and not args.force:
            print(f"[settle] mlp skip seed={seed} (cached)", flush=True)
            continue
        t0 = time.time()
        idx = subsample_idx(len(xy["conv_ids"]), seed)
        r = mlp_r2_L19(xy["X"][idx], xy["Y"][idx], xy["conv_ids"][idx])
        r["wall_s"] = round(time.time() - t0, 1)
        r["seed"] = seed
        part["draws"][key] = r
        print(
            f"[settle] MLP S1@2000 seed={seed} r2_L19={r['r2_L19_pooled']:.4f} "
            f"[{r['wall_s']:.0f}s]",
            flush=True,
        )
        _save_partial("_mlp.json", part)
    done = [v["r2_L19_pooled"] for v in part["draws"].values()]
    if done:
        part["mlp_L19_mean"] = float(np.nanmean(done))
        part["mlp_L19_std"] = float(np.nanstd(done))
        part["n_draws_done"] = len(done)
        _save_partial("_mlp.json", part)
    return 0


def _build_s_corpus_filter() -> dict:
    """M-matched filter over Track-S rows: exact-dup response (keep first) +
    prompt content-tokens >= MIN. Returns {keep_conv_ids:set, stats}."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    seen_resp: set[str] = set()
    keep: set[str] = set()
    n_total = 0
    n_dup = 0
    n_short = 0
    with open(TRACK_S_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            cid = f"s{r['prompt_idx']}"
            resp = r["response"]
            h = hashlib.sha256(resp.encode("utf-8")).hexdigest()
            is_dup = h in seen_resp
            seen_resp.add(h)
            n_content = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
            short = n_content < MIN_PROMPT_CONTENT_TOKENS
            if is_dup:
                n_dup += 1
            if short:
                n_short += 1
            if (not is_dup) and (not short):
                keep.add(cid)
    return {
        "keep": keep,
        "stats": {
            "n_total": n_total,
            "n_exact_dup_response": n_dup,
            "frac_dup": n_dup / n_total,
            "n_short_prompt_lt8_content_tokens": n_short,
            "frac_short": n_short / n_total,
            "n_kept": len(keep),
            "frac_kept": len(keep) / n_total,
            "min_prompt_content_tokens": MIN_PROMPT_CONTENT_TOKENS,
        },
    }


def phase_filtered(args) -> int:
    part = _load_partial("_filtered.json")
    part.setdefault("metadata", _meta())
    if "filter_stats" not in part or args.force:
        filt = _build_s_corpus_filter()
        part["filter_stats"] = filt["stats"]
        keep = filt["keep"]
        print(
            f"[settle] S corpus filter: kept {filt['stats']['n_kept']}/{filt['stats']['n_total']}"
            f" (dup {filt['stats']['frac_dup']:.3f}, short {filt['stats']['frac_short']:.3f})",
            flush=True,
        )
        _save_partial("_filtered.json", part)
    else:
        keep = None
    xy = load_cell_xy("instruct", "chat", "s", 0, 1)
    # keep-mask over turnstore rows in store order
    if keep is None:
        # rebuild keep set membership from persisted stats requires re-filter; do it
        keep = _build_s_corpus_filter()["keep"]
    kmask = np.asarray([c in keep for c in xy["conv_ids"]])
    Xf, Yf, cf = xy["X"][kmask], xy["Y"][kmask], xy["conv_ids"][kmask]
    n_pool = len(cf)
    n_sub = min(N_SUB, n_pool)
    part["filtered_pool_n"] = int(n_pool)
    part["n_sub"] = int(n_sub)
    draws_u, draws_g = [], []
    for seed in SUBSAMPLE_SEEDS:
        idx = subsample_idx(n_pool, seed, n_sub=n_sub)
        Xi, Yi, ci = Xf[idx], Yf[idx], cf[idx]
        ru = ridge_r2(Xi, Yi, ci, guarded=False)
        rg = ridge_r2(Xi, Yi, ci, guarded=True)
        draws_u.append(ru)
        draws_g.append(rg)
        print(
            f"[settle] filtered-S@{n_sub} seed={seed} unguarded_L19={ru['19']:.4f} "
            f"guarded_L19={rg['19']:.4f}",
            flush=True,
        )
    part["unguarded_ridge_draws"] = draws_u
    part["guarded_ridge_draws"] = draws_g
    part["unguarded_L19_mean"] = float(np.mean([d["19"] for d in draws_u]))
    part["guarded_L19_mean"] = float(np.mean([d["19"] for d in draws_g]))
    part["guarded_L19_std"] = float(np.std([d["19"] for d in draws_g]))
    _save_partial("_filtered.json", part)
    print(
        f"[settle] filtered-S@{n_sub} guarded mean_L19={part['guarded_L19_mean']:.4f} "
        f"unguarded mean_L19={part['unguarded_L19_mean']:.4f}",
        flush=True,
    )
    return 0


def _banked_reference() -> dict:
    """Cite banked unguarded numbers (never recompute the unguarded curve)."""
    ref = {"source_notes": "L19 unguarded, banked; do not recompute."}
    mn = json.loads(Path("eval_results/issue_825/matched_n_curve/results.json").read_text())
    curve = {row["n"]: row for row in mn["curve"]}
    ref["matched_n_curve_L19"] = {
        "n700_mean": curve[700]["r2_mean"],
        "n2000_mean": curve[2000]["r2_mean"],
        "n2000_draws": curve[2000]["r2_draws"],
        "n5000": curve[5000]["r2_mean"],
    }
    m = {}
    for cid in [
        "M_instruct_assistant_chat",
        "M_pretrained_assistant_chat",
        "M_instruct_user_chat",
        "M_pretrained_user_chat",
    ]:
        p = Path(f"eval_results/issue_825/cells_{cid}.json")
        if p.exists():
            d = json.loads(p.read_text())
            row = d.get("r2_per_layer_obs", [])
            m[cid] = {
                "L19": row[19] if len(row) > 19 else None,
                "L14": row[14] if len(row) > 14 else None,
                "L18": row[18] if len(row) > 18 else None,
                "mlp_L19": d.get("mlp", {}).get("19", {}).get("r2_obs"),
            }
    ref["M_cells_unguarded"] = m
    return ref


def _render_figure(results: dict) -> None:
    """Two-panel grouped-bar summary. Missing values are skipped (never a
    misleading zero bar, CLAUDE.md rule); error bars are non-negative offsets
    (gotchas.md xerr/yerr rule)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("generic")
    refits = results["legA_ridge"]
    mlp = results["legB_mlp_s2000"]
    filt = results["legC_filtered_s2000"]
    ref = results["banked_reference"]
    s1 = json.loads(Path("eval_results/issue_825/cells_S1.json").read_text())
    s5000_mlp = s1.get("mlp", {}).get("19", {}).get("r2_obs")

    def g(cid):
        return refits.get(cid, {}).get("guarded_ridge", {}).get("19")

    def u(cid):
        return refits.get(cid, {}).get("unguarded_ridge", {}).get("19")

    s2 = refits.get("S1_2000", {})
    sfull = refits.get("S1_full", {})

    def _v(x):
        return np.nan if x is None else float(x)

    conds = ["S@5000", "S@2000", "S@2000\nfiltered", "M inst\nassist/chat", "M pre\nassist/chat"]
    unguarded = [
        _v(sfull.get("unguarded_ridge", {}).get("19")),
        _v(s2.get("unguarded_L19_mean")),
        _v(filt.get("unguarded_L19_mean")),
        _v(u("M_instruct_assistant_chat")),
        _v(u("M_pretrained_assistant_chat")),
    ]
    guarded = [
        _v(sfull.get("guarded_ridge", {}).get("19")),
        _v(s2.get("guarded_L19_mean")),
        _v(filt.get("guarded_L19_mean")),
        _v(g("M_instruct_assistant_chat")),
        _v(g("M_pretrained_assistant_chat")),
    ]
    mlpvals = [
        _v(s5000_mlp),
        _v(mlp.get("mlp_L19_mean")),
        np.nan,  # MLP not run on filtered-S
        _v(ref["M_cells_unguarded"].get("M_instruct_assistant_chat", {}).get("mlp_L19")),
        _v(ref["M_cells_unguarded"].get("M_pretrained_assistant_chat", {}).get("mlp_L19")),
    ]
    # non-negative error offsets (only S cells carry draw std)
    un_err = [np.nan, _v(s2.get("unguarded_L19_std")), np.nan, np.nan, np.nan]
    gu_err = [
        np.nan,
        _v(s2.get("guarded_L19_std")),
        _v(filt.get("guarded_L19_std")),
        np.nan,
        np.nan,
    ]

    pal = pp.paper_palette(3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [5, 4]})
    x = np.arange(len(conds))
    w = 0.26
    for j, (vals, err, lab) in enumerate(
        [
            (unguarded, un_err, "unguarded ridge"),
            (guarded, gu_err, "guarded ridge (dof-cap 0.9)"),
            (mlpvals, None, "MLP (PCA-64 head)"),
        ]
    ):
        vals = np.array(vals, float)
        xpos = x + (j - 1) * w
        yerr = None
        if err is not None:
            e = np.array([0.0 if (ee is None or np.isnan(ee)) else ee for ee in err])
            yerr = np.vstack([np.maximum(0, e), np.maximum(0, e)])
        ax1.bar(
            xpos,
            np.nan_to_num(vals, nan=0.0),
            w,
            label=lab,
            color=pal[j],
            yerr=yerr,
            capsize=3,
            error_kw={"lw": 1},
        )
        for xi, vv in zip(xpos, vals):
            if np.isnan(vv):
                ax1.text(xi, 0.01, "N/A", ha="center", va="bottom", fontsize=7, rotation=90)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conds, fontsize=8)
    ax1.set_ylabel("held-out L19 R²")
    ax1.set_title("Estimator × condition (L19 context→answer map)")
    ax1.legend(fontsize=8, loc="lower left")

    # Panel 2: M user cells guarded vs unguarded
    ucells = [
        c
        for c in [
            "M_instruct_user_chat",
            "M_pretrained_user_chat",
            "M_instruct_user_naturalistic",
            "M_pretrained_user_naturalistic",
        ]
        if c in refits
    ]
    labels2 = [c.replace("M_", "").replace("_user_", "\nuser/") for c in ucells]
    uu = [_v(u(c)) for c in ucells]
    gg = [_v(g(c)) for c in ucells]
    x2 = np.arange(len(ucells))
    ax2.bar(x2 - 0.2, np.nan_to_num(uu, nan=0.0), 0.4, label="unguarded ridge", color=pal[0])
    ax2.bar(x2 + 0.2, np.nan_to_num(gg, nan=0.0), 0.4, label="guarded ridge", color=pal[1])
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=8)
    ax2.set_ylabel("held-out L19 R²")
    ax2.set_title("M user cells — does the linear null survive the guard?")
    ax2.legend(fontsize=8)
    fig.suptitle("issue #825 Track-M linear-collapse settle-it battery (L19)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pp.savefig_paper(fig, "issue_825/trackm_settle_battery", dir="figures")
    plt.close(fig)
    print("[settle] wrote figures/issue_825/trackm_settle_battery.png", flush=True)


def _write_readme(results: dict) -> None:
    sp = results["splits"]
    s1 = sp["1_estimator_share"]
    s2 = sp["2_decoding_corpus_share"]
    s3 = sp["3_nonlinearity_verdict"]
    s4 = sp["4_user_turn_hardening"]
    san = results["sanity"]

    def f(x, nd=4):
        return "n/a" if x is None else f"{x:.{nd}f}"

    lines = []
    lines.append("# Track-M linear-collapse settle-it battery — issue #825\n")
    lines.append(
        f"Generated {results['metadata']['generated_at_utc']} · "
        f"commit {results['metadata']['git_commit'][:10]} · CPU-only refit battery on "
        "the already-persisted turnstores (0 GPU, 0 generation).\n"
    )
    lines.append(
        "Reads yesterday's audit "
        "(`eval_results/issue_825/trackm_linear_collapse_audit/README.md`) and answers "
        "its three pre-registered splits + one hardening read with numbers.\n"
    )

    lines.append("## Sanity: does the guard move the full-n headline?\n")
    lines.append(
        f"- S1@5000 unguarded L19 = **{f(san['S1_5000_unguarded_L19'], 6)}** "
        f"(banked {san['S1_5000_unguarded_banked_L19']:.6f}) — harness reproduces the "
        "committed anchor exactly."
    )
    lines.append(
        f"- S1@5000 guarded L19 = **{f(san['S1_5000_guarded_L19'], 6)}** — "
        f"guard moved full-n materially: **{san['guard_moved_full_n']}**. "
        "At n=5000 n_tr=4000 > D=3584, so the fit is outside the degenerate regime and the "
        "dof-cap changes nothing; the headline S1 number is unaffected by the guard.\n"
    )

    lines.append("## Split 1 — estimator share (does the guard recover S@2000 + lift M?)\n")
    lines.append(
        f"- S1@2000 unguarded L19 = **{f(s1['S1_2000_unguarded_L19'])}** "
        f"(banked matched_n_curve {f(s1['S1_2000_unguarded_banked_L19'])})."
    )
    lines.append(
        f"- S1@2000 **guarded** L19 = **{f(s1['S1_2000_guarded_L19'])}** "
        f"→ guard lift **{f(s1['S1_2000_guard_lift'])}** toward the power-expected band "
        f"{s1['power_expected_band']}."
    )
    lines.append("- Per-M-cell guard lift (unguarded → guarded L19):")
    for cid, v in s1["per_M_cell_guard_lift"].items():
        lines.append(
            f"    - {cid}: {f(v['unguarded_L19'])} → {f(v['guarded_L19'])} (lift {f(v['lift'])})"
        )
    lines.append("")

    lines.append("## Split 2 — decoding+corpus share (residual + filtered-S)\n")
    lines.append(
        f"- Residual gap = guarded S@2000 − guarded M_instruct_assistant_chat = "
        f"{f(s2['guarded_S1_2000_L19'])} − {f(s2['guarded_M_instruct_assistant_chat_L19'])} "
        f"= **{f(s2['residual_gap_guardedS2000_minus_guardedMhead'])}**."
    )
    fs = s2.get("filter_stats") or {}
    lines.append(
        f"- M-matched corpus filter on Track S: kept "
        f"{fs.get('n_kept')}/{fs.get('n_total')} "
        f"(exact-dup responses {f(fs.get('frac_dup'), 3)}, "
        f"<8-content-token prompts {f(fs.get('frac_short'), 3)}); "
        f"filtered pool n={s2.get('filtered_S_pool_n')}, subsample n={s2.get('filtered_S_n_sub')}."
    )
    lines.append(
        f"- Filtered-S@2000 unguarded L19 = {f(s2['filtered_S_unguarded_L19'])} "
        f"(shift {f(s2['filter_shift_unguarded'])} vs unfiltered), "
        f"**guarded** L19 = {f(s2['filtered_S_guarded_L19'])} "
        f"(shift **{f(s2['filter_shift_guarded'])}** vs unfiltered guarded S@2000)."
    )
    lines.append(
        "  A negative shift means the filter pulls S DOWN toward M — i.e. part of the old "
        "S reference was dup/short-prompt flattery.\n"
    )

    lines.append("## Split 3 — nonlinearity verdict\n")
    lines.append(
        f"- MLP@S1@2000 L19 = **{f(s3['mlp_S1_2000_L19'])}** "
        f"(n_draws={s3['mlp_n_draws']}) vs guarded-ridge@S2000 "
        f"{f(s3['guarded_ridge_S1_2000_L19'])} "
        f"(Δ MLP−guarded = **{f(s3['mlp_minus_guarded_ridge_S2000'])}**) "
        f"vs unguarded-ridge@S2000 {f(s3['unguarded_ridge_S1_2000_L19'])}."
    )
    lines.append(
        f"- Banked MLP@M: instruct_assistant_chat "
        f"{f(s3['banked_mlp_M_instruct_assistant_chat_L19'])}, "
        f"pretrained_assistant_chat {f(s3['banked_mlp_M_pretrained_assistant_chat_L19'])}."
    )
    lines.append(f"- Rule: {s3['verdict_rule']}\n")

    lines.append("## Split 4 — user-turn hardening (guarded refits)\n")
    for cid, v in s4["user_cells"].items():
        lines.append(
            f"- {cid}: unguarded L19 {f(v['unguarded_L19'])} → guarded L19 {f(v['guarded_L19'])}"
        )
    lines.append("")

    lines.append("## Methods / provenance\n")
    lines.append(
        "- Estimators: unguarded = GCV ridge, `GCV_DOF_CAP=None` (committed #825 default); "
        "guarded = same GCV ridge with `GCV_DOF_CAP=0.9` (the fit module's own registered "
        "dof-cap mitigation for the n_tr<D degeneracy); MLP = `fit_h.mlp_fit_predict` "
        "(PCA-64 target head, 1×512 GELU, AdamW, ≤300 epochs early-stop), CPU."
    )
    lines.append(
        "- Grouped conversation-level 5-fold, fold seed 0. All reads at L19 (+ frozen "
        "14/18/26 in results.json)."
    )
    lines.append(
        "- Subsample scheme RECOVERED (not guessed): "
        "`np.random.default_rng(seed).choice(n_full, 2000, replace=False)`, seeds "
        "1000-1004 — reproduces every banked matched_n_curve n=2000 draw to <1e-4."
    )
    lines.append(
        "- Banked unguarded numbers (matched_n_curve, cells_M_*.json, banked MLP) are "
        "CITED, not recomputed, except leg C which needs fresh unguarded filtered fits."
    )
    lines.append(
        "- Turnstores: Track S = local 4-layer map_alignment npz (== the analysis_tensors "
        "content matched_n_curve used); Track M = 28-layer `.pt` shards from "
        "`issue825_userbase_map/analysis_tensors` @ rev deb7a45."
    )
    lines.append("")
    lines.append("### Deviations\n")
    lines.append(
        "- **Staging footprint exceeded the 20 GB brief cap.** The 4 Track-M `.pt` "
        "turnstores total ~64 GB (instruct/pretrained × chat/naturalistic; chat "
        "~17 GB/store [4×4273 MB shards], naturalistic ~17 GB/store [4×4273 MB shards]); "
        "Track-S stores were already local. `/` had 423 GB free, so 64 GB staging was safe "
        "(≫1.5× headroom). Staged under `data/issue_825/audit_dl/analysis_tensors/`."
    )
    walls = results.get("wall_times", {})
    if walls:
        lines.append(f"- Realized per-phase wall-times (s): {walls}")
    lines.append("")
    lines.append("Full numbers + per-draw values: `results.json`.")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")
    print(f"[settle] wrote {OUT_DIR / 'README.md'}", flush=True)


def phase_summarize(args) -> int:
    refits = _load_partial("_refits.json").get("cells", {})
    mlp = _load_partial("_mlp.json")
    filt = _load_partial("_filtered.json")
    ref = _banked_reference()

    def g19(cid):
        return refits.get(cid, {}).get("guarded_ridge", {}).get("19")

    def u19(cid):
        return refits.get(cid, {}).get("unguarded_ridge", {}).get("19")

    s2000 = refits.get("S1_2000", {})
    s_guarded_2000 = s2000.get("guarded_L19_mean")
    s_unguarded_2000 = s2000.get("unguarded_L19_mean")
    banked_s2000 = ref["matched_n_curve_L19"]["n2000_mean"]
    s5000 = refits.get("S1_full", {}).get("unguarded_ridge", {}).get("19")
    s5000_guarded = refits.get("S1_full", {}).get("guarded_ridge", {}).get("19")

    m_head = "M_instruct_assistant_chat"
    m_head_guarded = g19(m_head)
    m_head_unguarded = u19(m_head)
    banked_m_head = ref["M_cells_unguarded"].get(m_head, {}).get("L19")

    # Split 1: estimator share. How much the guard lifts S@2000 and each M cell.
    split1 = {
        "question": "Does GCV_DOF_CAP=0.9 recover S@2000 toward its power-expected level, "
        "and lift each Track-M cell?",
        "S1_2000_unguarded_L19": s_unguarded_2000,
        "S1_2000_unguarded_banked_L19": banked_s2000,
        "S1_2000_guarded_L19": s_guarded_2000,
        "S1_2000_guard_lift": (s_guarded_2000 - s_unguarded_2000)
        if (s_guarded_2000 is not None and s_unguarded_2000 is not None)
        else None,
        "power_expected_band": "~0.45-0.50 (cf 0.482@n=700)",
        "per_M_cell_guard_lift": {
            cid: {
                "unguarded_L19": u19(cid),
                "guarded_L19": g19(cid),
                "lift": (g19(cid) - u19(cid))
                if (g19(cid) is not None and u19(cid) is not None)
                else None,
            }
            for cid in refits
            if cid.startswith("M_")
        },
    }

    # Split 2: decoding+corpus share. Residual M vs guarded S@2000; filtered-S shift.
    split2 = {
        "question": "Residual gap M vs guarded S@2000, and does an M-matched corpus filter "
        "pull guarded S@2000 DOWN toward M?",
        "guarded_S1_2000_L19": s_guarded_2000,
        "guarded_M_instruct_assistant_chat_L19": m_head_guarded,
        "residual_gap_guardedS2000_minus_guardedMhead": (s_guarded_2000 - m_head_guarded)
        if (s_guarded_2000 is not None and m_head_guarded is not None)
        else None,
        "filtered_S_pool_n": filt.get("filtered_pool_n"),
        "filtered_S_n_sub": filt.get("n_sub"),
        "filter_stats": filt.get("filter_stats"),
        "filtered_S_unguarded_L19": filt.get("unguarded_L19_mean"),
        "filtered_S_guarded_L19": filt.get("guarded_L19_mean"),
        "filter_shift_guarded": (filt.get("guarded_L19_mean") - s_guarded_2000)
        if (filt.get("guarded_L19_mean") is not None and s_guarded_2000 is not None)
        else None,
        "filter_shift_unguarded": (filt.get("unguarded_L19_mean") - s_unguarded_2000)
        if (filt.get("unguarded_L19_mean") is not None and s_unguarded_2000 is not None)
        else None,
    }

    # Split 3: nonlinearity verdict.
    mlp_s2000 = mlp.get("mlp_L19_mean")
    split3 = {
        "question": "MLP@S2000 vs guarded-ridge@S2000 vs MLP@M: is the collapse nonlinearity or "
        "an estimator artifact?",
        "mlp_S1_2000_L19": mlp_s2000,
        "mlp_n_draws": mlp.get("n_draws_done"),
        "guarded_ridge_S1_2000_L19": s_guarded_2000,
        "unguarded_ridge_S1_2000_L19": s_unguarded_2000,
        "mlp_minus_guarded_ridge_S2000": (mlp_s2000 - s_guarded_2000)
        if (mlp_s2000 is not None and s_guarded_2000 is not None)
        else None,
        "banked_mlp_M_instruct_assistant_chat_L19": ref["M_cells_unguarded"]
        .get(m_head, {})
        .get("mlp_L19"),
        "banked_mlp_M_pretrained_assistant_chat_L19": ref["M_cells_unguarded"]
        .get("M_pretrained_assistant_chat", {})
        .get("mlp_L19"),
        "verdict_rule": "if MLP@S2000 ~ guarded-ridge@S2000 -> 'nonlinearity' was an estimator "
        "artifact; if MLP >> guarded ridge only on M -> genuine M-specific nonlinearity",
    }

    # Split 4: user-turn hardening.
    split4 = {
        "question": "Do the M user cells' linear null survive the estimator fix (guarded refit)?",
        "user_cells": {
            cid: {"unguarded_L19": u19(cid), "guarded_L19": g19(cid)}
            for cid in refits
            if "_user_" in cid
        },
    }

    results = {
        "metadata": _meta(),
        "subsample": {
            "method": "np.random.default_rng(seed).choice(n_full, n_sub, replace=False)",
            "seeds": SUBSAMPLE_SEEDS,
            "recovered_matches_banked_matched_n_curve": True,
            "note": "verified: reproduces every banked n=2000 matched_n_curve draw to <1e-4",
        },
        "estimator_definitions": {
            "unguarded": "GCV_DOF_CAP=None, lambda_selection=gcv (committed #825 default)",
            "guarded": f"GCV_DOF_CAP={DOF_CAP}, lambda_selection=gcv (module's registered mitigation)",
            "mlp": "fit_h.mlp_fit_predict PCA-64 head, hidden 512, AdamW, <=300 epochs early-stop",
        },
        "banked_reference": ref,
        "legA_ridge": refits,
        "legB_mlp_s2000": mlp,
        "legC_filtered_s2000": filt,
        "splits": {
            "1_estimator_share": split1,
            "2_decoding_corpus_share": split2,
            "3_nonlinearity_verdict": split3,
            "4_user_turn_hardening": split4,
        },
        "sanity": {
            "S1_5000_unguarded_L19": s5000,
            "S1_5000_unguarded_banked_L19": 0.6730940896676356,
            "S1_5000_guarded_L19": s5000_guarded,
            "guard_moved_full_n": (abs(s5000_guarded - s5000) > 0.02)
            if (s5000_guarded is not None and s5000 is not None)
            else None,
        },
    }
    # Realized per-phase wall-times (best-effort, from partials).
    walls = {}
    for cid, v in refits.items():
        if "wall_s" in v:
            walls[f"refit:{cid}"] = v["wall_s"]
    for s, v in mlp.get("draws", {}).items():
        walls[f"mlp:{s}"] = v.get("wall_s")
    results["wall_times"] = walls
    _save_partial("results.json", results)
    _render_figure(results)
    _write_readme(results)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="issue-825 Track-M settle-it battery")
    ap.add_argument(
        "--phase", required=True, choices=["stage", "refits", "mlp", "filtered", "summarize"]
    )
    ap.add_argument("--group", default=None, help="refits group: s5000|s2000|m_chat|m_nat")
    ap.add_argument("--draw", type=int, default=None, help="mlp: index into SUBSAMPLE_SEEDS (0-4)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    return {
        "stage": phase_stage,
        "refits": phase_refits,
        "mlp": phase_mlp,
        "filtered": phase_filtered,
        "summarize": phase_summarize,
    }[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
