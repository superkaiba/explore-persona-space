#!/usr/bin/env python3
"""#1768 inline round: does a DATA-AUGMENTED refit of the base context->answer
map reproduce the realized post-fine-tuning map?

The user ask: "Just include the context answer pairs from the training dataset
(teacher forced) in the context answer pairs used to train the mapping" and
"check if the mapping fit on general answer activations + the answer
activations of the training data (at that specific context) is the
same/similar to the mapping post-finetuning (and how well)".

Construction, per arm on the 8-arm write-predictability subset:

    M-hat+(lam_w) = val-selected primal ridge on
        the 15,000 base corpus train pairs (c0, v0)          [weight 1]
      UNION the arm's K training pairs (c_train, t_target)   [weight w]

both sides embedded by the BASE model (the training pairs teacher-forced), so
the whole construction is available AHEAD OF TIME from the base model plus the
training dataset -- no post-fine-tuning artifact enters the fit.

``lam_w`` is parameterized by EFFECTIVE LOSS MASS ``m = Kw/(N + Kw)``. Three
reads, each as a function of m (the dose curve):

1. OPERATOR level -- direction-aware operator cosine between the raw-space
   linear UPDATES, Delta-A-hat = A-hat+(m) - A0 vs Delta-A-real = A+ - A0,
   against a shuffle-fit null (training answers permuted across contexts) and
   an analytic norm-matched random-direction null. HEADLINE: the fraction of
   realized map change CLOSED on held-out rows, benchmarked against the banked
   B=200 refit-noise floor (the "statistically the same map" yardstick).
2. ACTION level -- the predicted write w-hat(x) = M-hat+(c0(x)) - M0(c0(x))
   against the measured matched-text write w_tf(x), vs a delta-rank-one
   (constant-write) baseline and an in-round write-predictability ceiling.
3. The m dose curve for every read.

Ridge conventions, splits, stores and the refit-noise floors are REUSED from
round 1 / the last-token repool round; nothing is re-derived here.

Compute note (vectorize-first): the augmented Gram ``A(w)`` depends only on the
CONTEXTS, never on the answers, so one eigh per (arm, layer, m) is shared by
the real fit AND every shuffle-null draw -- the null costs a (K,H)^T(K,D)
matmul per draw instead of a refactorization.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# load_dotenv() BEFORE numpy/torch: the shared-VM thread caps (#847) are
# setdefault-ed here and BLAS/torch freeze their pools at import.
load_dotenv()

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1768_fit as F  # noqa: E402
import issue1768_lasttoken as LT  # noqa: E402
import issue1768_lasttoken_fit as LTF  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.mapaug")

RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_1768" / "map_augmentation"
FIGS_DIR = REPO_ROOT / "figures" / "issue_1768" / "map_augmentation"
HF_SUBPREFIX = f"{X.HF_PREFIX}/map_augmentation"

# The context POSITION: last-token context is the round's PRIMARY pooling (the
# lasttoken-repool round's headline); `last_prompt` is that store's position id.
POSITION = "last_prompt"
HEADLINE_LAYER = 19

# Effective-loss-mass grid. m=0 is the anchor leg (must reproduce M0 exactly);
# `w1` is the NATURAL unweighted union (w=1, no upweighting at all) and is the
# pre-registered non-oracle read -- it is inserted per cell since m depends on K.
MASS_GRID: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50)
N_SHUFFLE = 10  # shuffle-fit null draws per (arm, layer, m) -- ~free, see docstring
SHUFFLE_SEED = 1768


def _phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


def _meta(extra: dict | None = None) -> dict:
    out = dict(LT._meta())
    out["position"] = POSITION
    out["round"] = "inline-map-augmentation"
    # `CAP._git_commit` shells `git rev-parse` and degrades to "unknown" when the
    # tree has no .git — which is exactly the case when the pod tree is an rsync
    # copy rather than a clone. Carry the producing commit explicitly so the
    # result JSONs keep real provenance, and record HOW the tree got there.
    override = os.environ.get("EPM_GIT_COMMIT_OVERRIDE", "").strip()
    if override:
        out["git_commit"] = override
    out["code_tree_source"] = os.environ.get("EPM_CODE_TREE_SOURCE", "git-checkout")
    # `git_commit` names the tree the rsync came FROM; this driver may still be
    # uncommitted at run time, so its own content hash is the honest pin.
    payload_sha = os.environ.get("EPM_CODE_PAYLOAD_SHA256", "").strip()
    if payload_sha:
        out["driver_sha256"] = payload_sha
        out["driver_committed_at_run_time"] = False
    if extra:
        out.update(extra)
    return out


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _device():
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def arm_picks() -> list[dict]:
    """The 8-arm subset, read from the committed write-predictability picks."""
    p = REPO_ROOT / "eval_results" / "issue_1768" / "write_predictability" / "arm_picks.json"
    picks = json.loads(p.read_text())["picks"]
    assert len(picks) == 8, len(picks)
    return picks


# ── staging ──────────────────────────────────────────────────────────────────


def stage_inputs(out_root: Path, arms: list[str]) -> dict:
    """Stage every Hub input this round reads (fail-loud, idempotent)."""
    from explore_persona_space.orchestrate import hub

    _phase("stage")
    t0 = time.time()
    units = sorted({X.base_unit_for(a) for a in arms} | set(arms))
    staged = []

    sample = out_root / "inputs" / "corpus_sample.json"
    hub.stage_hub_file(X.HF_DATA_REPO, f"{X.HF_PREFIX}/inputs/corpus_sample.json", sample)
    staged.append(str(sample))

    # `CAP._mix_positive_rows` reads the arm registry from ``cfg.out_root``, NOT
    # from eval_results/ (audited: it touches exactly `cfg.out_root`,
    # `cfg.model_override` and the delta_tf staging dir), so place the committed
    # registry where that consumer looks. Copy rather than symlink so the file is
    # readable no matter how out_root is mounted.
    reg_src = REPO_ROOT / "eval_results" / "issue_1768" / "arm_registry.json"
    assert reg_src.is_file(), f"missing committed registry: {reg_src}"
    reg_dst = out_root / "arm_registry.json"
    if not reg_dst.is_file() or reg_dst.read_bytes() != reg_src.read_bytes():
        reg_dst.write_bytes(reg_src.read_bytes())
    staged.append(str(reg_dst))
    logger.info(
        "[stage] arm_registry.json placed at %s (CAP._mix_positive_rows reads out_root)", reg_dst
    )

    for u in units:
        tgt = out_root / "lasttoken" / u / "lasttoken.pt"
        hub.stage_hub_file(X.HF_DATA_REPO, f"{X.HF_PREFIX}/lasttoken_ctx/{u}/lasttoken.pt", tgt)
        staged.append(str(tgt))
        logger.info("[stage] lasttoken %s staged", u)

    info = {"n_files": len(staged), "units": units, "wall_s": round(time.time() - t0, 1)}
    logger.info("[stage] done: %d files, %.1fs", len(staged), info["wall_s"])
    return info


# ── training pairs: base-model teacher-forced per-row embeddings ─────────────


def _mix_sources(picks: list[dict]) -> dict[str, dict]:
    """arm_id -> {delta_arm, pos_path, layout} via the #1768 registry.

    The two full-FT arms carry no own mix: ``delta_arm_for`` resolves them to
    the matched LoRA cell, which #1586 trained on the SAME mix
    (``issue1768_cells.delta_arm_for`` docstring).
    """
    reg = json.loads((REPO_ROOT / "eval_results" / "issue_1768" / "arm_registry.json").read_text())
    mps = reg["mix_pos_sources"]
    index = {a.arm_id: a for a in X.all_arms()}
    out = {}
    for p in picks:
        arm_id = p["arm_id"]
        delta_arm = X.delta_arm_for(index[arm_id])
        src = mps.get(delta_arm)
        assert src is not None, (arm_id, delta_arm, "no mix_pos_sources entry")
        out[arm_id] = {"delta_arm": delta_arm, **src}
    return out


def _base_last_prompt_context(rows: list[dict], layers: list[int], dev, dtype, batch: int) -> dict:
    """LAST-PROMPT-TOKEN hidden states from BASE-model prompt-only forwards.

    Mirrors the corpus last-token producer (``issue1768_lasttoken.capture_unit``)
    convention-for-convention, because these vectors are stacked into the SAME
    design matrix as that store's rows: prompt-only forwards, RIGHT padding (so
    positions index from 0 with no ``position_ids`` threading), a hook on
    ``model.model.layers[li]``, the index ``len(prompt_token_ids) - 1``, and
    ``logits_to_keep=1`` where the signature accepts it (the unread-logits OOM
    guard). Using the span-MEAN context helper here instead would silently mix
    two different context poolings in one fit.
    """
    import inspect

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(X.BASE_MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        X.BASE_MODEL,
        torch_dtype=dtype,
        device_map={"": str(dev)},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    n_blocks = len(model.model.layers)
    for li in layers:
        assert 0 <= li < n_blocks, (li, n_blocks)
    captured: dict[int, object] = {}

    def make_hook(li: int):
        def hook_fn(module, inp, out):
            captured[li] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook_fn

    hooks = [model.model.layers[li].register_forward_hook(make_hook(li)) for li in layers]
    fwd = getattr(model, "forward", model.__call__)
    keep_kwargs = (
        {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(fwd).parameters else {}
    )
    pooled: dict[int, list] = {li: [] for li in layers}
    try:
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            seqs = [r["prompt_token_ids"] for r in chunk]
            max_len = max(len(s) for s in seqs)
            input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                attn[i, : len(s)] = 1
            with torch.no_grad():
                model(
                    input_ids=input_ids.to(dev),
                    attention_mask=attn.to(dev),
                    **keep_kwargs,
                )
            for li in layers:
                hs = captured[li]
                assert hs.shape[:2] == (len(chunk), max_len), (hs.shape, len(chunk), max_len)
                for i, r in enumerate(chunk):
                    j = len(r["prompt_token_ids"]) - 1
                    assert 0 <= j < max_len, (j, max_len)
                    pooled[li].append(hs[i, j, :].float().cpu())
    finally:
        for h in hooks:
            h.remove()
        captured.clear()
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return {li: torch.stack(pooled[li]) for li in layers}


def build_train_pairs(out_root: Path, picks: list[dict], tf_batch: int) -> dict:
    """Base-model TF pass -> PER-ROW training-pair embeddings, deduped by mix.

    The banked p5 ``delta_tf`` stores keep only the MEAN (``tbar``) --
    ``issue1768_capture.run_delta_unit`` discards the per-row tensor -- so the
    per-row contexts/answers the augmented fit needs must be captured here.

    Per mix we store the LAST-TOKEN context (matching the round's primary
    context pooling) and the response SPAN-MEAN (matching the answer stores).
    """
    import torch

    import issue1768_capture as CAP
    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    _phase("trainpairs")
    srcs = _mix_sources(picks)
    by_path: dict[str, list[str]] = {}
    for arm_id, s in srcs.items():
        by_path.setdefault(s["pos_path"], []).append(arm_id)

    index = {a.arm_id: a for a in X.all_arms()}
    dev = _device()
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    manifest: dict[str, dict] = {}

    for i, (pos_path, arm_ids) in enumerate(sorted(by_path.items()), start=1):
        slug = pos_path.replace("/", "__")
        dest = out_root / "train_pairs" / f"{slug}.pt"
        meta_dest = dest.with_suffix(".meta.json")
        if dest.exists() and meta_dest.exists():
            manifest[pos_path] = json.loads(meta_dest.read_text())
            logger.info("[trainpairs] %d/%d %s: present, skip", i, len(by_path), slug)
            continue
        t0 = time.time()
        # the delta arm owns the mix; _mix_positive_rows keys off the arm's own
        # registry entry, so use the resolved delta arm (FT arms share a LoRA mix)
        delta_arm = srcs[arm_ids[0]]["delta_arm"]
        cfg = CAP.Cfg(
            out_root=out_root,
            phases=(),  # required field; this round drives _mix_positive_rows directly
            layers=tuple(X.LAYERS),
            smoke=False,
            tf_batch=tf_batch,
        )
        rows, mix_meta = CAP._mix_positive_rows(cfg, index[delta_arm])
        assert rows, (pos_path, "no positive rows")
        # ANSWER side: response SPAN-MEAN via the same helper the corpus answer
        # stores used (`issue1768_capture` p3/p5 both call it), so S matches V0.
        pooled = _teacher_forced_span_means(
            X.BASE_MODEL,
            rows,
            [delta_arm],
            layers=list(X.LAYERS),
            spans=("response",),
            device=str(dev),
            dtype=dtype,
            tf_batch_size=tf_batch,
        )
        # CONTEXT side: LAST-PROMPT-TOKEN, matching the `last_prompt` position of
        # the corpus lasttoken stores these rows are stacked with. The span-MEAN
        # context is deliberately NOT stored — keeping it would invite a
        # pooling-mismatched read.
        ctx_last = _base_last_prompt_context(rows, list(X.LAYERS), dev, dtype, tf_batch)
        payload = {
            "context_last_prompt": {li: t.clone() for li, t in ctx_last.items()},
            "response_span_mean": {li: t.clone() for li, t in pooled["response"].items()},
            "n_rows": len(rows),
            "context_pooling": "last_prompt (index len(prompt_token_ids)-1, prompt-only forward)",
            "answer_pooling": "response span-mean (teacher-forced, matches corpus_capture)",
        }
        tmp = dest.with_suffix(f".pt.tmp.{os.getpid()}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, tmp)
        os.replace(tmp, dest)
        info = {
            "pos_path": pos_path,
            "delta_arm": delta_arm,
            "arm_ids": sorted(arm_ids),
            "n_rows": len(rows),
            "layers": list(X.LAYERS),
            "wall_s": round(time.time() - t0, 1),
            **{k: v for k, v in mix_meta.items() if k != "layout"},
            "layout": mix_meta["layout"],
            **_meta(),
        }
        _atomic_json(meta_dest, info)
        manifest[pos_path] = info
        logger.info(
            "[trainpairs] %d/%d %s: K=%d rows sha=%s elapsed=%.1fs",
            i,
            len(by_path),
            slug,
            len(rows),
            mix_meta["pos_sha256"][:12],
            time.time() - t0,
        )
    return manifest


def load_train_pairs(out_root: Path, pos_path: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, S) = per-row (LAST-PROMPT-TOKEN context, response span-mean) fp64.

    The poolings are the ones the corpus stores use for C0 and V0 respectively —
    T is stacked into the same design matrix as C0, so a pooling mismatch here
    would corrupt every augmented leg silently.
    """
    import torch

    slug = pos_path.replace("/", "__")
    p = out_root / "train_pairs" / f"{slug}.pt"
    assert p.exists(), f"missing train pairs: {p}"
    d = torch.load(p, map_location="cpu", weights_only=False)
    assert "context_last_prompt" in d, (p, "stale train-pairs store: no last-token context")
    T = np.asarray(d["context_last_prompt"][layer].float().numpy(), dtype=np.float64)
    S = np.asarray(d["response_span_mean"][layer].float().numpy(), dtype=np.float64)
    assert T.shape[0] == S.shape[0], (T.shape, S.shape)
    return T, S


# ── weighted ridge: the augmented refit ─────────────────────────────────────


def _eigh_robust(A):
    """cuda eigh with the documented CPU-fp64 fallback (gotchas.md cuSOLVER).

    Exact numerical-BACKEND swap of the same decomposition -- never a jitter
    (a jitter would move every eigenvalue the lambda scan consumes).
    """
    import torch

    try:
        s_eig, U = torch.linalg.eigh(A)
    except torch.linalg.LinAlgError:
        logger.warning("[fit] cuda eigh non-convergence at n=%d — CPU fp64 fallback", A.shape[0])
        A64 = A.double().cpu()
        A64 = 0.5 * (A64 + A64.T)
        s_eig, U = torch.linalg.eigh(A64)
        s_eig, U = s_eig.to(A.device), U.to(A.device)
    return torch.clamp(s_eig, min=0.0), U


def _augmented_blocks(C_corpus, V_corpus, tr, T, S, w, dev, block):
    """Standardized Gram + cross blocks for the augmented design, split by source.

    Returns ``(fac_parts, XtY_corpus, Ts, ymu)`` where ``fac_parts`` carries the
    w-dependent eigh state. The corpus/train SPLIT is what makes the shuffle
    null cheap: A never depends on the answers, and only the train block of XtY
    moves when S is permuted.
    """
    import torch

    n_tr = len(tr)
    K = T.shape[0]
    # weighted standardizer over the AUGMENTED train set (corpus rows w=1, train rows w)
    Xa_parts = [(C_corpus, tr, np.ones(n_tr)), (T, np.arange(K), np.full(K, w))]
    H = C_corpus.shape[1]
    D = V_corpus.shape[1]
    sum_x = torch.zeros(H, dtype=torch.float64, device=dev)
    sumsq_x = torch.zeros(H, dtype=torch.float64, device=dev)
    sum_y = torch.zeros(D, dtype=torch.float64, device=dev)
    sum_w = 0.0
    for src_x, src_y in ((0, V_corpus), (1, S)):
        arr, idx_all, wt = Xa_parts[src_x]
        for s in range(0, len(idx_all), block):
            idx = idx_all[s : s + block]
            wb = torch.as_tensor(wt[s : s + block], dtype=torch.float64, device=dev)
            xb = torch.as_tensor(arr[idx], dtype=torch.float64, device=dev)
            yb = torch.as_tensor(src_y[idx], dtype=torch.float64, device=dev)
            sum_w += float(wb.sum())
            sum_x += (xb * wb[:, None]).sum(0)
            sumsq_x += (xb * xb * wb[:, None]).sum(0)
            sum_y += (yb * wb[:, None]).sum(0)
    xmu = sum_x / sum_w
    var = (sumsq_x - sum_w * xmu * xmu) / max(1.0, sum_w - 1.0)
    xsd = torch.clamp(var, min=0.0).sqrt() + 1e-9
    ymu = sum_y / sum_w

    A = torch.zeros((H, H), dtype=torch.float64, device=dev)
    XtY_corpus = torch.zeros((H, D), dtype=torch.float64, device=dev)
    for s in range(0, n_tr, block):
        idx = tr[s : s + block]
        xb = (torch.as_tensor(C_corpus[idx], dtype=torch.float64, device=dev) - xmu) / xsd
        yb = torch.as_tensor(V_corpus[idx], dtype=torch.float64, device=dev) - ymu
        A += xb.T @ xb
        XtY_corpus += xb.T @ yb
    Ts = (torch.as_tensor(T, dtype=torch.float64, device=dev) - xmu) / xsd
    if w > 0:
        A += w * (Ts.T @ Ts)
    s_eig, U = _eigh_robust(A)
    return (
        {"U": U, "s_eig": s_eig, "xmu": xmu, "xsd": xsd, "ymu": ymu},
        XtY_corpus,
        Ts,
        ymu,
    )


def _augmented_xty(XtY_corpus, Ts, S_t, ymu, w, perm, dev):
    """The augmented X^T Y cross block, optionally with the answers PERMUTED.

    Only this block moves when S is shuffled -- the Gram (hence the eigh in
    ``_augmented_blocks``) depends on the CONTEXTS alone -- which is what makes
    each shuffle-fit null draw a (K,H)^T(K,D) matmul instead of a refit.
    """
    import torch

    if w <= 0:
        return XtY_corpus
    Sb = (S_t if perm is None else S_t[torch.as_tensor(perm, device=dev)]) - ymu
    return XtY_corpus + w * (Ts.T @ Sb)


def _select_and_payload(Xa, fac_parts, XtY, val, te, Y_val, dev, block):
    """Val-select lambda off ONE shared eigh, then build the apply_map payload.

    Mirrors ``n1m.fit_ridge_with_weights`` (same grid, same selection order,
    same ``_ridge_predict_one`` predictions) but takes a pre-built
    factorization so the eigh is shared across answer-side variants.
    """
    import torch

    import issue779_ffc_n1m_fits as n1m

    fac = dict(fac_parts)
    fac["UtXtY"] = fac["U"].T @ XtY
    lo, hi, n = -3.0, 8.0, 23
    best_lam, best_vr2, edge = None, -np.inf, None
    for _ext in range(4):
        grid = F.lambda_grid(lo, hi, n)
        best_lam, best_vr2 = float(grid[0]), -np.inf
        for lam in grid:
            pv = n1m._ridge_predict_one(Xa, val, fac, lam, dev, block)
            vr2 = F._pooled_r2(pv, Y_val)
            if np.isfinite(vr2) and vr2 > best_vr2:
                best_vr2, best_lam = vr2, float(lam)
        edge = None
        if np.isclose(best_lam, float(grid[0])):
            edge = "low"
        elif np.isclose(best_lam, float(grid[-1])):
            edge = "high"
        if edge is None:
            break
        # the factorization is lambda-INDEPENDENT, so extending the grid reuses
        # the same eigh (identical numerics, no refactorization)
        if edge == "low":
            lo -= 1.0
        else:
            hi += 1.0
        n += 2
    W = fac["U"] @ (fac["UtXtY"] / (fac["s_eig"] + best_lam)[:, None])
    payload = {
        "kind": "ridge",
        "selected_lambda": best_lam,
        "xmu": fac["xmu"].detach().cpu().to(torch.float32),
        "xsd": fac["xsd"].detach().cpu().to(torch.float32),
        "ymu": fac["ymu"].detach().cpu().to(torch.float32),
        "W": W.detach().cpu().to(torch.float32),
    }
    meta = {
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(best_vr2),
        "lambda_grid_edge": edge,
        "lambda_grid": [lo, hi, n],
    }
    pred_te = n1m._ridge_predict_one(Xa, te, fac, best_lam, dev, block) if len(te) else None
    return pred_te, meta, payload


# ── operator-level reads ────────────────────────────────────────────────────


def _raw_operator(payload) -> np.ndarray:
    """The raw-space linear part A of M(c) = c @ A + b.

    M(c) = ((c - xmu)/xsd) @ W + ymu, so A = W / xsd[:, None]. Comparing raw
    operators (not standardized W) is what makes maps fitted under DIFFERENT
    standardizers commensurable.
    """
    import torch

    W = payload["W"].to(dtype=torch.float64).numpy()
    xsd = payload["xsd"].to(dtype=torch.float64).numpy()
    return W / xsd[:, None]


def _op_cos(dA: np.ndarray, dB: np.ndarray) -> float:
    num = float((dA * dB).sum())
    den = float(np.linalg.norm(dA) * np.linalg.norm(dB))
    return num / den if den > 0 else float("nan")


def _spectrum_cos(dA: np.ndarray, dB: np.ndarray, dev) -> float:
    """DESCRIPTIVE ONLY: cosine between SORTED singular-value spectra.

    Rotation-invariant -- it can NEVER support a "same operator up to rotation"
    claim (the similarity-statistic-semantics rule). Reported beside the
    direction-aware cosine purely as a shape summary.
    """
    import torch

    sa = torch.linalg.svdvals(torch.as_tensor(dA, dtype=torch.float32, device=dev))
    sb = torch.linalg.svdvals(torch.as_tensor(dB, dtype=torch.float32, device=dev))
    num = float((sa * sb).sum())
    den = float(sa.norm() * sb.norm())
    return num / den if den > 0 else float("nan")


# ── per-cell driver ─────────────────────────────────────────────────────────


def _mass_to_weight(m: float, n_tr: int, K: int) -> float:
    if m <= 0:
        return 0.0
    return float(m * n_tr / (K * (1.0 - m)))


def _weight_to_mass(w: float, n_tr: int, K: int) -> float:
    return float(K * w / (n_tr + K * w))


def fit_cell(out_root: Path, arm_id: str, layer: int, pos_path: str, block: int) -> dict:
    """Every map-augmentation read for one (arm, layer)."""
    import torch

    dev = _device()
    cache = out_root / "lt_answer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cell = LTF.build_cell(out_root, cache, arm_id, layer, POSITION)
    tr, val, te = F._split_idx(cell["split"])
    C0, V0 = cell["C0"], cell["V0"]
    Cp, Vp, Vt = cell["Cplus"], cell["Vplus"], cell["Vplus_tf"]
    n_tr, d = len(tr), C0.shape[1]
    assert n_tr > d, (n_tr, d, "estimator validity: n_train must exceed d")

    T, S = load_train_pairs(out_root, pos_path, layer)
    K = T.shape[0]

    # ── realized maps, refit here so the operator comparison has payloads ────
    realized = {}
    payloads = {}
    for name, (Xd, Yd) in {
        "M0": (C0, V0),
        "Mplus": (Cp, Vp),
        "Mplus_tf": (Cp, Vt),
    }.items():
        pred_te, meta, payload = F._fit_map(Xd, Yd, tr, val, te, dev)
        payloads[name] = payload
        realized[name] = {
            "heldout_r2": F._pooled_r2(pred_te, Yd[te]),
            "mean_cos": F._mean_cos(pred_te, Yd[te]),
            "selected_lambda": meta["selected_lambda"],
        }
        logger.info(
            "[fits] %s L%d %s: r2=%.4f lam=%.3g",
            arm_id,
            layer,
            name,
            realized[name]["heldout_r2"],
            meta["selected_lambda"],
        )

    # write-predictability ceiling RECOMPUTED on these last-token rows/splits
    # (the banked write_predictability numbers are span-mean-pooled: a pooling
    # mismatch, so they are context only, never the benchmark for this leg)
    W_tf_all = Vt - V0
    pred_w_te, meta_w, _pw = F._fit_map(C0, W_tf_all, tr, val, te, dev)
    w_te = W_tf_all[te]
    ceiling = {
        "heldout_r2": F._pooled_r2(pred_w_te, w_te),
        "mean_cos": F._mean_cos(pred_w_te, w_te),
        "selected_lambda": meta_w["selected_lambda"],
        "note": "direct ridge c0 -> w_tf on the SAME last-token rows/splits",
    }

    # ── common base-c grid: every map applied to c0 on the TEST rows ─────────
    C0_te = C0[te]
    P0 = F._apply_payload(payloads["M0"], C0_te, dev)
    Pp = F._apply_payload(payloads["Mplus"], C0_te, dev)
    Pt = F._apply_payload(payloads["Mplus_tf"], C0_te, dev)
    realized_change_rows = np.linalg.norm(Pp - P0, axis=1)
    realized_change_med = float(np.median(realized_change_rows))
    realized_change_rows_tf = np.linalg.norm(Pt - P0, axis=1)
    realized_change_med_tf = float(np.median(realized_change_rows_tf))

    A0 = _raw_operator(payloads["M0"])
    dA_real = _raw_operator(payloads["Mplus"]) - A0
    dA_real_tf = _raw_operator(payloads["Mplus_tf"]) - A0

    # δ-rank-one (constant-write) baseline: δ = mean training-pair map residual
    M0_on_T = F._apply_payload(payloads["M0"], T, dev)
    delta_vec = (S - M0_on_T).mean(axis=0)
    # scale fitted on VAL rows only (never the test comparison target)
    w_val = W_tf_all[val]
    dn = float(delta_vec @ delta_vec)
    s_star = float((w_val @ delta_vec).mean() / dn) if dn > 0 else 0.0
    delta_pred_te = np.repeat((s_star * delta_vec)[None, :], len(te), axis=0)
    delta_baseline = {
        "form": "constant write s*delta, delta = mean_k(t_k - M0(c_k)), s fit on VAL rows",
        "s_star": s_star,
        "delta_norm": float(np.linalg.norm(delta_vec)),
        "action_mean_cos": F._mean_cos(delta_pred_te, w_te),
        "action_r2": F._pooled_r2(delta_pred_te, w_te),
        "r2_note": (
            "scored against the TEST-set mean of w (the write_predictability "
            "convention), so a CONSTANT write reads <= 0 by construction"
        ),
    }

    # ── the mass grid (the dose curve) ──────────────────────────────────────
    Xa = np.vstack([C0, T])
    Ya = np.vstack([V0, S])
    # NOTE: no explicit augmented train-index array — `_augmented_blocks`
    # accumulates the corpus and training-pair blocks separately (that split is
    # what makes the shuffle null cheap). `Xa`/`Ya` exist only so val/test rows
    # stay addressable by their ORIGINAL indices in `_ridge_predict_one`.
    rng = np.random.default_rng(SHUFFLE_SEED)
    perms = [rng.permutation(K) for _ in range(N_SHUFFLE)]

    masses = sorted(set(MASS_GRID) | {_weight_to_mass(1.0, n_tr, K)})
    legs = []
    for m in masses:
        t0 = time.time()
        w = _mass_to_weight(m, n_tr, K)
        fac_parts, XtY_corpus, Ts, ymu = _augmented_blocks(C0, V0, tr, T, S, w, dev, block)
        S_t = torch.as_tensor(S, dtype=torch.float64, device=dev)
        pred_te, meta_aug, payload_aug = _select_and_payload(
            Xa,
            fac_parts,
            _augmented_xty(XtY_corpus, Ts, S_t, ymu, w, None, dev),
            val,
            te,
            V0[val],
            dev,
            block,
        )
        Phat = F._apply_payload(payload_aug, C0_te, dev)
        dA_hat = _raw_operator(payload_aug) - A0

        resid_rows = np.linalg.norm(Phat - Pp, axis=1)
        resid_med = float(np.median(resid_rows))
        resid_rows_tf = np.linalg.norm(Phat - Pt, axis=1)
        resid_med_tf = float(np.median(resid_rows_tf))
        what = Phat - P0

        # shuffle-fit null: the eigh above is REUSED (A never sees the answers)
        null_cos = []
        if w > 0:
            for perm in perms:
                _pn, _mn, payload_n = _select_and_payload(
                    Xa,
                    fac_parts,
                    _augmented_xty(XtY_corpus, Ts, S_t, ymu, w, perm, dev),
                    val,
                    np.empty(0, dtype=int),
                    V0[val],
                    dev,
                    block,
                )
                null_cos.append(_op_cos(_raw_operator(payload_n) - A0, dA_real))

        leg = {
            "mass": m,
            "weight": w,
            "is_natural_union": bool(abs(w - 1.0) < 1e-12),
            "selected_lambda": meta_aug["selected_lambda"],
            "val_r2_at_selected": meta_aug["val_r2_at_selected"],
            "lambda_grid_edge": meta_aug["lambda_grid_edge"],
            # operator level
            "op_cos_update_vs_Mplus": _op_cos(dA_hat, dA_real),
            "op_cos_update_vs_Mplus_tf": _op_cos(dA_hat, dA_real_tf),
            "op_update_fro": float(np.linalg.norm(dA_hat)),
            "op_real_update_fro": float(np.linalg.norm(dA_real)),
            "shuffle_null_op_cos": null_cos,
            # headline: fraction of realized map change closed on held-out rows
            "resid_med_vs_Mplus": resid_med,
            "frac_change_closed": 1.0 - resid_med / realized_change_med
            if realized_change_med > 0
            else float("nan"),
            "resid_med_vs_Mplus_tf": resid_med_tf,
            "frac_change_closed_tf": 1.0 - resid_med_tf / realized_change_med_tf
            if realized_change_med_tf > 0
            else float("nan"),
            # action level
            "action_mean_cos": F._mean_cos(what, w_te),
            "action_r2": F._pooled_r2(what, w_te),
            "wall_s": round(time.time() - t0, 1),
        }
        if layer == HEADLINE_LAYER:
            leg["spectrum_cos_descriptive"] = _spectrum_cos(dA_hat, dA_real, dev)
        legs.append(leg)
        logger.info(
            "[fits] %s L%d m=%.4g w=%.4g: closed=%.4f op_cos=%.4f "
            "null_med=%s act_cos=%.4f act_r2=%+.4f elapsed=%.1fs",
            arm_id,
            layer,
            m,
            w,
            leg["frac_change_closed"],
            leg["op_cos_update_vs_Mplus"],
            f"{np.median(null_cos):.4f}" if null_cos else "n/a",
            leg["action_mean_cos"],
            leg["action_r2"],
            leg["wall_s"],
        )
        fac_parts, XtY_corpus, Ts, S_t = None, None, None, None
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "arm_id": arm_id,
        "layer": layer,
        "position": POSITION,
        "method": X.arm_method(arm_id),
        "pos_path": pos_path,
        "n_rows": int(len(cell["sha"])),
        "n_train": int(n_tr),
        "n_val": int(len(val)),
        "n_test": int(len(te)),
        "d": int(d),
        "K_train_pairs": int(K),
        "realized": realized,
        "write_predictability_ceiling": ceiling,
        "delta_rank_one_baseline": delta_baseline,
        "realized_change_med": realized_change_med,
        "realized_change_med_tf": realized_change_med_tf,
        "legs": legs,
        "conventions": {
            "grid": "every map applied to the COMMON base-c grid c0(x) on TEST rows",
            "headline": (
                "frac_change_closed = 1 - median||M-hat+(c0) - M+(c0)|| / median||M0(c0) - M+(c0)||"
            ),
            "lambda_selection": "corpus VAL rows (base answers) — never the comparison target",
            "mass_selection": (
                "NOT selected: the dose curve is the read; the w=1 natural-union leg is the "
                "pre-registered non-oracle point, and any argmax over mass is labeled oracle"
            ),
        },
        **_meta(),
    }


def _prewarm_arm_cache(cache: Path, arm_id: str) -> None:
    """Persist this arm's answer stores ONCE, layer-complete, before the layer loop.

    ``LTF.fetch_response`` deliberately does NOT cache per-arm stores (at 72 arms
    x 2 kinds the npz would blow the per-pod quota), so a naive (arm, layer) loop
    re-downloads each ~0.7 GB store once PER LAYER. This round has only 8 arms,
    so caching them costs ~5.6 GB of fp16 npz and removes 2/3 of the transfer;
    ``_drop_arm_cache`` releases it as soon as the arm's layers are done.
    """
    for kind in ("corpus_capture", "corpus_capture_tf"):
        LTF.fetch_response(cache, kind, arm_id, list(X.LAYERS), persist=True)


def _drop_arm_cache(cache: Path, arm_id: str) -> None:
    for kind in ("corpus_capture", "corpus_capture_tf"):
        for suffix in (".npz", ".sha.json"):
            q = cache / f"{kind}__{arm_id}{suffix}"
            q.unlink(missing_ok=True)


def run_fits(out_root: Path, results_dir: Path, picks: list[dict], layers: list[int], block: int):
    _phase("fits")
    srcs = _mix_sources(picks)
    cells_dir = results_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    cache = out_root / "lt_answer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [(p["arm_id"], li) for p in picks for li in layers]
    k = 0
    # ARM-OUTER so one download of each arm's answer stores serves every layer
    for arm_id in [p["arm_id"] for p in picks]:
        pending = [li for li in layers if not (cells_dir / f"{arm_id}_L{li}.json").exists()]
        if pending:
            _prewarm_arm_cache(cache, arm_id)
        for layer in layers:
            k += 1
            dest = cells_dir / f"{arm_id}_L{layer}.json"
            if dest.exists():
                logger.info("[fits] unit %d/%d %s_L%d: present, skip", k, len(todo), arm_id, layer)
                continue
            t0 = time.time()
            rec = fit_cell(out_root, arm_id, layer, srcs[arm_id]["pos_path"], block)
            _atomic_json(dest, rec)
            logger.info(
                "[fits] unit %d/%d %s_L%d elapsed=%.1fs",
                k,
                len(todo),
                arm_id,
                layer,
                time.time() - t0,
            )
        _drop_arm_cache(cache, arm_id)
    logger.info("[fits] all %d units complete", len(todo))


# ── summary ─────────────────────────────────────────────────────────────────


def _load_floors(arm_id: str, layer: int) -> dict:
    """The banked B=200 refit-noise floor for this cell (the same-map yardstick)."""
    p = REPO_ROOT / "eval_results" / "issue_1768" / "lasttoken_repool" / "cells" / f"{arm_id}.json"
    if not p.exists():
        return {"available": False, "reason": f"missing {p.name}"}
    d = json.loads(p.read_text())
    node = d["positions"][POSITION][str(layer)]["map_change"]
    return {
        "available": True,
        "floor_p95": node["floor_p95"],
        "n_refits": node["floors"]["M0"]["n_refits"],
        "banked_m0_heldout_r2": d["positions"][POSITION][str(layer)]["M0"]["heldout_r2"],
        "banked_mplus_heldout_r2": d["positions"][POSITION][str(layer)]["Mplus"]["heldout_r2"],
        "banked_delta_med": node["delta_med"],
        "banked_D": node["D"],
        "banked_verdict": node["verdict"],
        "source": str(p.relative_to(REPO_ROOT)),
    }


def _worst_anchor(cells: dict) -> dict:
    """Worst anchor-reproduction drift across cells (the round's correctness read)."""
    rows = [(k, v["anchor_reproduction"]) for k, v in cells.items() if "anchor_reproduction" in v]
    if not rows:
        return {"available": False}
    worst_r2 = max(rows, key=lambda kv: kv[1]["m0_r2_absdiff"])
    worst_ch = max(rows, key=lambda kv: kv[1]["change_med_reldiff"])
    return {
        "available": True,
        "n_cells": len(rows),
        "worst_m0_r2_absdiff": {"cell": worst_r2[0], **worst_r2[1]},
        "worst_change_med_reldiff": {"cell": worst_ch[0], **worst_ch[1]},
        "note": (
            "the realized maps are REFIT here (the banked cells store reads, not W "
            "payloads); these drifts are the reproduction check at production n"
        ),
    }


def build_summary(results_dir: Path, picks: list[dict], layers: list[int]) -> dict:
    _phase("summary")
    cells = {}
    for p in picks:
        for li in layers:
            f = results_dir / "cells" / f"{p['arm_id']}_L{li}.json"
            if not f.exists():
                continue
            rec = json.loads(f.read_text())
            floors = _load_floors(p["arm_id"], li)
            rec["refit_noise_floor"] = floors
            if floors.get("available"):
                # ANCHOR REPRODUCTION (at production n): this round REFITS the
                # realized maps to obtain their W payloads, so its M0/M+ reads and
                # its realized map change must reproduce the banked round-1 values.
                # A drift here invalidates every augmented leg, so it is recorded
                # per cell rather than left to a reader's spot-check.
                rec["anchor_reproduction"] = {
                    "m0_r2_refit": rec["realized"]["M0"]["heldout_r2"],
                    "m0_r2_banked": floors["banked_m0_heldout_r2"],
                    "m0_r2_absdiff": abs(
                        rec["realized"]["M0"]["heldout_r2"] - floors["banked_m0_heldout_r2"]
                    ),
                    "mplus_r2_refit": rec["realized"]["Mplus"]["heldout_r2"],
                    "mplus_r2_banked": floors["banked_mplus_heldout_r2"],
                    "mplus_r2_absdiff": abs(
                        rec["realized"]["Mplus"]["heldout_r2"] - floors["banked_mplus_heldout_r2"]
                    ),
                    "change_med_refit": rec["realized_change_med"],
                    "change_med_banked": floors["banked_delta_med"],
                    "change_med_reldiff": abs(
                        rec["realized_change_med"] - floors["banked_delta_med"]
                    )
                    / max(1e-12, floors["banked_delta_med"]),
                }
            if floors.get("available") and rec["realized_change_med"] > 0:
                ceil_frac = 1.0 - floors["floor_p95"] / rec["realized_change_med"]
                rec["max_closable_fraction"] = ceil_frac
                for leg in rec["legs"]:
                    leg["frac_of_closable_closed"] = (
                        leg["frac_change_closed"] / ceil_frac if ceil_frac > 0 else float("nan")
                    )
            cells[f"{p['arm_id']}|L{li}"] = rec

    def _best(rec, key):
        pos = [lg for lg in rec["legs"] if lg["weight"] > 0]
        if not pos:
            return None
        return max(pos, key=lambda lg: lg[key] if np.isfinite(lg[key]) else -np.inf)

    headline = {}
    for k, rec in cells.items():
        if rec["layer"] != HEADLINE_LAYER:
            continue
        nat = next((lg for lg in rec["legs"] if lg["is_natural_union"]), None)
        bc = _best(rec, "frac_change_closed")
        ba = _best(rec, "action_mean_cos")
        headline[rec["arm_id"]] = {
            "K_train_pairs": rec["K_train_pairs"],
            "n_train_corpus": rec["n_train"],
            "realized_change_med": rec["realized_change_med"],
            "refit_noise_floor_p95": rec["refit_noise_floor"].get("floor_p95"),
            "max_closable_fraction": rec.get("max_closable_fraction"),
            "natural_union": None
            if nat is None
            else {
                "mass": nat["mass"],
                "frac_change_closed": nat["frac_change_closed"],
                "op_cos": nat["op_cos_update_vs_Mplus"],
                "shuffle_null_med": float(np.median(nat["shuffle_null_op_cos"]))
                if nat["shuffle_null_op_cos"]
                else None,
                "action_mean_cos": nat["action_mean_cos"],
                "action_r2": nat["action_r2"],
            },
            "best_over_mass_ORACLE": None
            if bc is None
            else {
                "mass": bc["mass"],
                "frac_change_closed": bc["frac_change_closed"],
                "frac_of_closable_closed": bc.get("frac_of_closable_closed"),
                "op_cos": bc["op_cos_update_vs_Mplus"],
                "shuffle_null_med": float(np.median(bc["shuffle_null_op_cos"]))
                if bc["shuffle_null_op_cos"]
                else None,
                "note": "argmax over the mass grid — SELECTION-OPTIMISTIC, not a held-out read",
            },
            "best_action_over_mass_ORACLE": None
            if ba is None
            else {
                "mass": ba["mass"],
                "action_mean_cos": ba["action_mean_cos"],
                "action_r2": ba["action_r2"],
            },
            "write_predictability_ceiling": rec["write_predictability_ceiling"],
            "delta_rank_one_baseline": rec["delta_rank_one_baseline"],
        }

    summary = {
        "question": (
            "Does refitting the base context->answer map with the arm's training pairs "
            "injected (teacher-forced BASE embeddings, upweighted on a mass grid) "
            "reproduce the realized post-fine-tuning map?"
        ),
        "headline_layer": HEADLINE_LAYER,
        "layers": layers,
        "position": POSITION,
        "n_arms": len(picks),
        "mass_grid": list(MASS_GRID),
        "n_shuffle_null_draws": N_SHUFFLE,
        "random_direction_null": {
            "form": "analytic: a norm-matched random operator update has E[cos]=0, sd=1/sqrt(H*D)",
            "sd": 1.0 / math.sqrt(3584 * 3584),
            "note": (
                "this null is nearly vacuous at H=D=3584 (sd ~2.8e-4) — the SHUFFLE-FIT "
                "null is the informative comparand and carries the interpretation"
            ),
        },
        "banked_span_mean_write_predictability": {
            "note": (
                "the committed write_predictability summary is SPAN-MEAN-pooled context; "
                "reported as context only — the in-round last-token ceiling is the "
                "matched benchmark for these legs"
            ),
            "source": "eval_results/issue_1768/write_predictability/summary.json",
        },
        "anchor_reproduction_worst": _worst_anchor(cells),
        "headline_by_arm": headline,
        "cells": cells,
        **_meta(),
    }
    _atomic_json(results_dir / "summary.json", summary)
    logger.info("[summary] wrote %s (%d cells)", results_dir / "summary.json", len(cells))
    return summary


# ── figures ─────────────────────────────────────────────────────────────────


def build_figures(results_dir: Path, figs_dir: Path) -> list[Path]:
    _phase("figs")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    summary = json.loads((results_dir / "summary.json").read_text())
    cells = {k: v for k, v in summary["cells"].items() if v["layer"] == HEADLINE_LAYER}
    figs_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(cells, key=lambda k: cells[k]["arm_id"])
    # one colour per ARM, held fixed across all three figures (colour<->meaning
    # consistency: the same arm is the same colour everywhere in this round)
    pal = paper_palette(len(order))
    colors = {k: pal[i] for i, k in enumerate(order)}
    out: list[Path] = []

    def _short(arm_id: str) -> str:
        return arm_id.replace("-lr", " lr").replace("-s", " s")

    # (a) fraction of realized change closed vs mass
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k in order:
        rec = cells[k]
        legs = [lg for lg in rec["legs"] if lg["weight"] > 0]
        xs = [lg["mass"] for lg in legs]
        ys = [lg["frac_change_closed"] for lg in legs]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, color=colors[k], label=_short(rec["arm_id"]))
        nat = next((lg for lg in rec["legs"] if lg["is_natural_union"]), None)
        if nat:
            ax.plot(
                [nat["mass"]],
                [nat["frac_change_closed"]],
                marker="*",
                ms=13,
                color=colors[k],
                mec="black",
                mew=0.6,
                zorder=5,
            )
        mc = rec.get("max_closable_fraction")
        if mc is not None:
            ax.axhline(mc, color=colors[k], ls=":", lw=0.8, alpha=0.55)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("effective loss mass of the training pairs, $m$ (log)")
    ax.set_ylabel("fraction of realized map change closed")
    ax.set_title(
        f"Data-augmented refit vs the realized post-FT map (L{HEADLINE_LAYER}, last-token)",
        fontsize=10,
    )
    ax.legend(fontsize=6.2, ncol=2, loc="best", frameon=True)
    ax.text(
        0.01,
        0.02,
        "star = natural union (w=1); dotted = per-arm refit-noise ceiling\n"
        "0 = no better than the base map M0",
        transform=ax.transAxes,
        fontsize=6,
        va="bottom",
    )
    fig.tight_layout()
    out.append(savefig_paper(fig, "closed_fraction_vs_mass", dir=figs_dir, formats=("png",))["png"])
    plt.close(fig)

    # (b) operator cosine of the updates, with the shuffle-fit null band
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k in order:
        rec = cells[k]
        legs = [lg for lg in rec["legs"] if lg["weight"] > 0]
        xs = [lg["mass"] for lg in legs]
        ys = [lg["op_cos_update_vs_Mplus"] for lg in legs]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, color=colors[k], label=_short(rec["arm_id"]))
        lo, hi = [], []
        for lg in legs:
            nc = lg["shuffle_null_op_cos"]
            lo.append(np.quantile(nc, 0.025) if nc else np.nan)
            hi.append(np.quantile(nc, 0.975) if nc else np.nan)
        ax.fill_between(xs, lo, hi, color=colors[k], alpha=0.16, lw=0)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("effective loss mass of the training pairs, $m$ (log)")
    ax.set_ylabel(r"operator cosine  $\cos(\Delta \hat{A},\ \Delta A_{\rm real})$")
    ax.set_title(
        f"Direction-aware operator similarity of the UPDATES (L{HEADLINE_LAYER})", fontsize=10
    )
    ax.legend(fontsize=6.2, ncol=2, loc="best", frameon=True)
    ax.text(
        0.01,
        0.02,
        "shaded = shuffle-fit null (training answers permuted across contexts), 95% band",
        transform=ax.transAxes,
        fontsize=6,
        va="bottom",
    )
    fig.tight_layout()
    out.append(savefig_paper(fig, "operator_cosine_vs_mass", dir=figs_dir, formats=("png",))["png"])
    plt.close(fig)

    # (c) action level: cos + R2 vs mass, with ceiling + delta baseline
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, key, lab in (
        (axes[0], "action_mean_cos", "mean per-row cosine with $w_{tf}$"),
        (axes[1], "action_r2", "pooled $R^2$ for $w_{tf}$"),
    ):
        for k in order:
            rec = cells[k]
            legs = [lg for lg in rec["legs"] if lg["weight"] > 0]
            ax.plot(
                [lg["mass"] for lg in legs],
                [lg[key] for lg in legs],
                marker="o",
                ms=4,
                lw=1.4,
                color=colors[k],
                label=_short(rec["arm_id"]),
            )
            ck = rec["write_predictability_ceiling"]["mean_cos" if "cos" in key else "heldout_r2"]
            ax.axhline(ck, color=colors[k], ls="--", lw=0.8, alpha=0.5)
            dk = rec["delta_rank_one_baseline"][key]
            ax.axhline(dk, color=colors[k], ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0.0, color="black", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("effective loss mass, $m$ (log)")
        ax.set_ylabel(lab)
    axes[1].set_ylim(bottom=max(-1.0, axes[1].get_ylim()[0]))
    axes[0].legend(fontsize=6.0, ncol=2, loc="best", frameon=True)
    fig.suptitle(
        f"Action level: predicted write vs measured $w_{{tf}}$ (L{HEADLINE_LAYER}); "
        "dashed = in-round fitted ceiling, dotted = $\\delta$ constant-write baseline",
        fontsize=9,
    )
    fig.tight_layout()
    out.append(savefig_paper(fig, "action_level_vs_mass", dir=figs_dir, formats=("png",))["png"])
    plt.close(fig)
    for q in out:
        logger.info("[figs] wrote %s", q)
    return out


# ── parity gate ─────────────────────────────────────────────────────────────


def verify_parity(out_root: Path, arm_id: str, layer: int, block: int) -> int:
    """The augmented path at w=0 must reproduce the reference M0 fit EXACTLY.

    This is the round's correctness gate: same standardizer, same Gram, same
    lambda selection, same predictions as ``F._fit_map`` on the corpus-only
    design. Any drift here invalidates every augmented leg.
    """
    import torch

    _phase("verify_parity")
    dev = _device()
    cache = out_root / "lt_answer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cell = LTF.build_cell(out_root, cache, arm_id, layer, POSITION)
    tr, val, te = F._split_idx(cell["split"])
    C0, V0 = cell["C0"], cell["V0"]
    T, S = load_train_pairs(out_root, _mix_sources(arm_picks())[arm_id]["pos_path"], layer)

    ref_pred, ref_meta, ref_payload = F._fit_map(C0, V0, tr, val, te, dev)
    Xa, Ya = np.vstack([C0, T]), np.vstack([V0, S])
    fac_parts, XtY_corpus, _Ts, _ymu = _augmented_blocks(C0, V0, tr, T, S, 0.0, dev, block)
    aug_pred, aug_meta, aug_payload = _select_and_payload(
        Xa, fac_parts, XtY_corpus, val, te, V0[val], dev, block
    )

    ok = True
    if not np.isclose(ref_meta["selected_lambda"], aug_meta["selected_lambda"], rtol=0, atol=0):
        print(f"FAIL lambda: ref={ref_meta['selected_lambda']} aug={aug_meta['selected_lambda']}")
        ok = False
    max_abs = float(np.abs(ref_pred - aug_pred).max())
    rel = max_abs / max(1e-30, float(np.abs(ref_pred).max()))
    print(f"pred max_abs_diff={max_abs:.3e} rel={rel:.3e}")
    if rel > 1e-10:
        print("FAIL predictions diverged beyond fp64 reassociation tolerance")
        ok = False
    for k in ("xmu", "xsd", "ymu", "W"):
        a = ref_payload[k].to(torch.float64).numpy()
        b = aug_payload[k].to(torch.float64).numpy()
        md = float(np.abs(a - b).max())
        print(f"payload {k}: max_abs_diff={md:.3e}")
        if md > 1e-3 * max(1.0, float(np.abs(a).max())):
            print(f"FAIL payload {k} diverged")
            ok = False
    r2_ref = F._pooled_r2(ref_pred, V0[te])
    r2_aug = F._pooled_r2(aug_pred, V0[te])
    print(f"heldout_r2 ref={r2_ref:.10f} aug={r2_aug:.10f}")
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


# ── upload ──────────────────────────────────────────────────────────────────


def upload(results_dir: Path, figs_dir: Path | None = None, out_root: Path | None = None) -> dict:
    """Persist results + figures + the per-row training-pair stores, verified.

    One ``upload_folder`` commit per source tree (never a per-file loop — the
    #664 504-storm), then ONE server-side-scoped exact-set verify per prefix.
    """
    _phase("upload")
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    trees: list[tuple[Path, str]] = [(results_dir, HF_SUBPREFIX)]
    if figs_dir is not None and figs_dir.is_dir():
        trees.append((figs_dir, f"{HF_SUBPREFIX}/figures"))
    if out_root is not None and (out_root / "train_pairs").is_dir():
        trees.append((out_root / "train_pairs", f"{HF_SUBPREFIX}/train_pairs"))

    total = 0
    for tree, prefix in trees:
        files = sorted(p for p in tree.rglob("*") if p.is_file())
        assert files, f"nothing to upload under {tree}"
        hub._upload(
            tree,
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=prefix,
        )
        expected = [f"{prefix}/{p.relative_to(tree).as_posix()}" for p in files]
        missing = hub.verify_repo_paths_uploaded(
            api,
            hub.DEFAULT_DATASET_REPO,
            expected,
            path_in_repo=prefix,
            repo_type="dataset",
        )
        assert not missing, f"upload verify {prefix}: {len(missing)} missing: {sorted(missing)[:5]}"
        logger.info("[upload] verified %d files under %s", len(expected), prefix)
        total += len(expected)
    return {"n_files": total, "prefix": HF_SUBPREFIX}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--figs-dir", type=Path, default=FIGS_DIR)
    ap.add_argument("--layers", default=",".join(str(x) for x in X.LAYERS))
    ap.add_argument("--arms", default="")
    ap.add_argument("--phase", default="all")
    ap.add_argument("--tf-batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=50_000)
    ap.add_argument("--verify-parity", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        # EVERY deferred import this entrypoint reaches on its real code path
        import issue1768_capture as _CAP  # noqa: F401
        import issue779_ffc_n1m_fits as _n1m  # noqa: F401
        import matplotlib as _mpl  # noqa: F401
        import torch as _torch  # noqa: F401
        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            paper_palette,
            savefig_paper,
            set_paper_style,
        )
        from explore_persona_space.analysis.representation_shift import (  # noqa: F401
            _teacher_forced_span_means,
        )
        from explore_persona_space.orchestrate import hub as _hub  # noqa: F401

        assert hasattr(_hub, "verify_repo_paths_uploaded")
        assert hasattr(_hub, "stage_hub_file")
        assert hasattr(_n1m, "_ridge_predict_one")
        print("import-check ok")
        return 0

    layers = [int(x) for x in args.layers.split(",")]
    picks = arm_picks()
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",") if a.strip()}
        picks = [p for p in picks if p["arm_id"] in keep]
        assert picks, (keep, "no matching arms")
    assert args.out_root is not None, "--out-root is required"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    arms = [p["arm_id"] for p in picks]

    if args.verify_parity:
        rc = verify_parity(out_root, arms[0], layers[0], args.block)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(rc)

    ph = args.phase
    if ph in ("all", "stage"):
        stage_inputs(out_root, arms)
    if ph in ("all", "trainpairs"):
        build_train_pairs(out_root, picks, args.tf_batch)
    if ph in ("all", "fits"):
        run_fits(out_root, args.results_dir, picks, layers, args.block)
    if ph in ("all", "summary"):
        build_summary(args.results_dir, picks, layers)
    if ph in ("all", "figs"):
        build_figures(args.results_dir, args.figs_dir)
    if ph == "upload":
        upload(args.results_dir, args.figs_dir, out_root)

    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # explicit exit BEFORE C-extension finalization (the PyGILState_Release
    # atexit race turns a completed phase into a nonzero rc — gotchas.md)
    sys.exit(0)


if __name__ == "__main__":
    main()
