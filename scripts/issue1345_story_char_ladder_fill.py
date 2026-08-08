#!/usr/bin/env python
"""Issue #1345 inline round — story + character transfer-ladder gap fill.

Two measurement gaps in the framing/character transfer line:

  (A) The 9-rung transfer ladder (``issue1345_ladder_rungs.py``) ran ONLY on the
      chat (r1) <-> no_template (r2) pair. The STORY regimes were never laddered,
      so "which correction reconciles the chat map with the story map" is
      undecidable for the framing contrast the line actually cares about.
  (B) The 4-character panel (HELIOS / Wren / Dana / Vex x {inserted, on-policy}
      x {instruct, base}) has story TEXT but — as of this round — no activation
      capture, so no character cell can be fit. See ``char_cells.json``: this
      script RECORDS that blocker with its evidence rather than substituting a
      different estimator.

Ladders filled here (instruct, one arm, headline layer 19):

  r1 <-> r4     chat  <->  paired story, answers embedded verbatim ("inserted")
  r1 <-> r4op   chat  <->  paired story, model writes its own answer in-story

``run_cell`` computes BOTH directions of a pair in one pass (the parent's
design), so the two pair-cells above cover all four requested ladders.

ESTIMATOR REGIME (#1887, binding). The story cells sit at n_train ~ 1.4k-1.7k
against d = 3584, so an AMBIENT-basis GCV ridge is estimator-degenerate there
(the #1887 audit read -0.31 ambient vs +0.37 reduced for the r4 context cell).
Every headline fit in this script therefore runs in a per-fold TRAIN-ONLY PCA
basis with

    k = min(1024, n_train_min // 2, d)          (fit825.reduced_basis_k)

which is the #1887 ``reduced_basis_k`` arm verbatim — ``_train_pca_basis``'s
centering-only recipe, then the committed standardize / Gram-eigh / GCV chain
on the projected coordinates. ``n_train >= 2k`` holds by construction, so the
fit is well-posed and the GCV dof cap can never bind: the projected Gram has
rank <= k <= n_train/2, hence dof <= k <= 0.5 * n_train < 0.9 * n_train. That
is asserted at runtime (``dof_frac_max``), which is what makes reusing the
parent's cap-0.9 ``_select_lambda`` numerically identical to #1887's cap-None
reduced arm. ``--basis both`` additionally reports the ambient read as a
secondary (never headline) column.

WHAT IS REUSED vs FORKED. The genuinely shared numerics are IMPORTED from
``issue1345_ladder_rungs`` (``_select_lambda``, ``procrustes_apply``,
``knn_retrieval``, the rung roster) — never copied. ``prep`` / ``dual_predict``
/ ``_rungs_for`` / ``run_cell`` are forked here because the reduced-basis
projection threads through their signatures; the delta vs the parent is the
``k_red`` projection and nothing else. Stores load through the #1887 adapter
path (``fit825._load_bundle_any`` -> ``_cell_xy``), whose replay gate passed on
67 cells, rather than the parent ladder's slim-cache extractor.

Vectorization is the parent's, unchanged: dual form throughout (the d x d
operator is never materialized), every eigh / QR / SVD / matmul batched over a
leading layer axis, four per-fold Gram eigendecompositions shared by all 9
rungs in BOTH directions, and the exact thin-QR Procrustes.

Content hygiene: the story corpus is LMSYS-derived real user text. This script
never prints prompt / story text; the character-example extractor emits
structured excerpt records to JSON only.

Outputs (under --out-dir):
  ladders.json      every ladder: per-rung R^2, ceiling, nulls, kNN, n, k, folds
  char_cells.json   the 16 character cells + the capture blocker evidence
  char_examples.json  one story excerpt per character on a shared conversation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# load_dotenv() BEFORE torch: torch freezes its intra-op thread pool from
# OMP_NUM_THREADS at import, so the shared-VM caps (#847) bind only if the env
# is populated first.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue825_fit_cells as fit825  # noqa: E402
import issue1345_common as c  # noqa: E402
import issue1345_ladder_rungs as lr  # noqa: E402
from explore_persona_space.analysis import mapping_baselines as mb  # noqa: E402

HEADLINE_LAYER = lr.HEADLINE_LAYER
N_FOLDS = lr.N_FOLDS
RUNGS = lr.RUNGS
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Regime -> (turnstore stem format_key, staged subdir, target turn index).
# The staged subdirs are the #1887 audit's own flat staging layout; the pinned
# HF prefixes/revisions they were staged from are recorded in
# issue1887_lambda_audit (I1345_PARENT_STORE_REV / I1345_VARIANT_STORE_REVS)
# and echoed into the output metadata by `store_pins()`.
REGIME_SPECS: dict[str, dict] = {
    "r1": {"format_key": "chat", "subdir": "parent_turnstore", "turn": 1},
    "r2": {"format_key": "naturalistic", "subdir": "parent_turnstore", "turn": 1},
    "r4": {
        "format_key": "stories_paired",
        "subdir": "conversation_paired_stories_assistant_turnstore",
        "turn": 0,
    },
    "r4op": {
        "format_key": "stories_paired_op",
        "subdir": "onpolicy_assistant_story_turnstore",
        "turn": 0,
    },
}

REGIME_LABEL = {
    "r1": "chat",
    "r2": "no_template",
    "r4": "story_inserted",
    "r4op": "story_onpolicy",
}

# Pairs requested by the round: each entry yields BOTH directions.
DEFAULT_PAIRS = (("r1", "r4"), ("r1", "r4op"))

# The 4-persona panel x {inserted, on-policy} x {instruct, base} = 16 cells.
CHAR_VARIANTS = tuple(
    f"char_{ch}{suf}"
    for ch in ("helios", "wren", "dana", "vex")
    for suf in ("", "_op", "_base", "_op_base")
)


def store_pins() -> dict:
    """Pinned HF prefix + revision per staged subdir (provenance for the output)."""
    import issue1887_lambda_audit as audit

    pins = {
        "parent_turnstore": {
            "prefix": audit.I1345_PARENT_STORE_PREFIX,
            "revision": audit.I1345_PARENT_STORE_REV,
        }
    }
    for variant in ("conversation_paired_stories_assistant", "onpolicy_assistant_story"):
        pins[f"{variant}_turnstore"] = {
            "prefix": f"issue1345_framing/{variant}/analysis_tensors/turnstore",
            "revision": audit.I1345_VARIANT_STORE_REVS[variant],
        }
    return pins


# ---------------------------------------------------------------------------
# Store loading -> layer-sliced cache
# ---------------------------------------------------------------------------
def load_regime_xy(
    stage_root: Path, cache_dir: Path, model: str, regime: str, arm: str, layer: int
) -> dict:
    """(X, Y, conv_ids) for one (model, regime, arm) at ONE layer.

    Loads the pt-shard bundle through the #1887 adapter path, slices the single
    headline layer, and caches the slice (~135 MB vs a ~10 GB bundle) so the
    ladder pairs never re-materialize a full store. Bounded peak RSS: one
    bundle at a time, freed before the next.
    """
    spec = REGIME_SPECS[regime]
    cache = cache_dir / f"{model}_{spec['format_key']}_{c.TRACK}_{arm}_L{layer}.pt"
    if cache.is_file():
        d = torch.load(cache, map_location="cpu", weights_only=False)
        return {"X": d["X"], "Y": d["Y"], "conv_ids": np.asarray(d["conv_ids"])}

    stem_dir = stage_root / spec["subdir"]
    assert stem_dir.is_dir(), f"staged store dir missing: {stem_dir}"
    t0 = time.time()
    bundle = fit825._load_bundle_any(
        stem_dir, model, spec["format_key"], c.TRACK, wanted_keys=("slots", "profiles")
    )
    c.assert_pt_bundle(bundle, expect_slots=2, expect_layers=fit825.EXPECTED_LAYERS)
    cell = {"slot_index": c.ARM_SLOT_INDEX[arm], "target_turn_index": spec["turn"]}
    xy = fit825._cell_xy(bundle, cell)
    out = {
        "X": torch.from_numpy(np.ascontiguousarray(xy["X"][:, layer, :])),
        "Y": torch.from_numpy(np.ascontiguousarray(xy["Y"][:, layer, :])),
        "conv_ids": np.asarray(xy["conv_ids"]),
    }
    del bundle, xy
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".pt.tmp")
    torch.save({"X": out["X"], "Y": out["Y"], "conv_ids": list(out["conv_ids"])}, tmp)
    os.replace(tmp, cache)
    print(
        f"[cache] {model}/{regime}/{arm} L{layer}: n={out['X'].shape[0]} "
        f"d={out['X'].shape[1]} in {time.time() - t0:.0f}s -> {cache.name}",
        flush=True,
    )
    return out


def matched_pair(a: dict, b: dict) -> tuple[dict, dict, np.ndarray]:
    """Restrict + reorder two regime blocks to their shared conversation ids.

    Fails loud on a duplicated conv id: the ladder pairs rows ACROSS regimes by
    conversation, so a duplicate would silently pair the wrong rows.
    """
    for name, blk in (("source", a), ("target", b)):
        ids = blk["conv_ids"]
        assert len(np.unique(ids)) == len(ids), (
            f"{name} block has duplicate conv_ids ({len(ids) - len(np.unique(ids))} dupes) — "
            "cross-regime row pairing would be ambiguous"
        )
    keep = np.array(sorted(set(a["conv_ids"]) & set(b["conv_ids"])))
    assert len(keep) > 0, "no shared conversation ids between the paired regimes"

    def take(blk):
        pos = {cid: i for i, cid in enumerate(blk["conv_ids"])}
        idx = torch.as_tensor([pos[k] for k in keep])
        # (L, n, d) layer-major with L=1: every downstream op batches over the
        # leading layer axis exactly as the parent ladder does.
        return {
            "X": blk["X"].index_select(0, idx).unsqueeze(0).to(torch.float64),
            "Y": blk["Y"].index_select(0, idx).unsqueeze(0).to(torch.float64),
        }

    return take(a), take(b), keep


# ---------------------------------------------------------------------------
# Reduced-basis batched dual ridge (the ONE delta vs the parent ladder)
# ---------------------------------------------------------------------------
def pca_basis(Xtr: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Train-only PCA basis via the (ntr x ntr) Gram eigh, batched over layers.

    Torch/batched transcription of ``fit825._train_pca_basis``: CENTERING only
    (the downstream ``prep`` standardizes the projected coordinates), rows of
    the returned basis are unit right-singular vectors of the centered train
    block in descending singular value. Returns ``(mu (L,1,d), basis (L,k,d))``.
    """
    mu = Xtr.mean(1, keepdim=True)
    Xc = Xtr - mu
    w, V = torch.linalg.eigh(Xc @ Xc.transpose(1, 2))
    w = torch.clamp(w, min=0.0)
    order = torch.argsort(w, dim=-1, descending=True)[:, :k]
    w_k = torch.gather(w, 1, order)
    # The parent drops near-zero components (k_eff <= k). With n_train >= 2k the
    # top-k Gram eigenvalues are bounded away from zero; a violation would make
    # k_eff layer-dependent and break the batched shape, so fail loud instead.
    assert bool((w_k > w_k.max(dim=1, keepdim=True).values * 1e-12).all()), (
        "near-zero eigenvalue inside the top-k PCA basis — k_eff < k; "
        "the batched reduced basis assumes a uniform k across layers"
    )
    Vk = torch.gather(V, 2, order.unsqueeze(1).expand(-1, V.shape[1], -1))
    basis = (Vk.transpose(1, 2) @ Xc) / torch.sqrt(w_k).unsqueeze(-1)
    return mu, basis


def prep(Xtr: torch.Tensor, k_red: int | None = None) -> dict:
    """Optional train-PCA projection, then the committed standardize + Gram eigh."""
    if k_red is None:
        pmu, basis, Ztr = None, None, Xtr
    else:
        pmu, basis = pca_basis(Xtr, k_red)
        Ztr = (Xtr - pmu) @ basis.transpose(1, 2)
    mu = Ztr.mean(1, keepdim=True)
    sd = Ztr.std(1, keepdim=True) + 1e-9
    Zn = (Ztr - mu) / sd
    w, V = torch.linalg.eigh(Zn @ Zn.transpose(1, 2))
    return {
        "pmu": pmu,
        "basis": basis,
        "mu": mu,
        "sd": sd,
        "Xn": Zn,
        "w": torch.clamp(w, min=0.0),
        "V": V,
        "ntr": int(Ztr.shape[1]),
    }


def _project(p: dict, Xev: torch.Tensor) -> torch.Tensor:
    """Ambient eval rows -> this prep's input coordinates (identity if ambient)."""
    if p["basis"] is None:
        return Xev
    return (Xev - p["pmu"]) @ p["basis"].transpose(1, 2)


def dual_predict(
    p: dict, Ytr: torch.Tensor, Xev: torch.Tensor, diag: dict | None = None
) -> torch.Tensor:
    """Ridge p.X -> Ytr evaluated at Xev, batched over layers, dual form.

    The primal (d x d) operator is never formed: evaluation is a cross-Gram
    against the training rows. ``diag`` collects the selected lambda + the
    realized dof fraction (the selected-lambda diagnostics the standing
    estimator-validity rule requires alongside every ridge read).
    """
    ymu = Ytr.mean(1, keepdim=True)
    Yc = Ytr - ymu
    VtY = p["V"].transpose(1, 2) @ Yc
    lam = lr._select_lambda(p, VtY, (Yc**2).sum((-1, -2)))
    if diag is not None:
        dof = (p["w"] / (p["w"] + lam[:, None])).sum(-1)
        diag.setdefault("lambda", []).extend(float(x) for x in lam)
        diag.setdefault("dof_frac", []).extend(float(x) / p["ntr"] for x in dof)
    alpha = p["V"] @ (VtY / (p["w"] + lam[:, None]).unsqueeze(-1))
    Zev = (_project(p, Xev) - p["mu"]) / p["sd"]
    return ((Zev @ p["Xn"].transpose(1, 2)) @ alpha) + ymu


def _rungs_for(
    p_s, p_ans, Ys_fit, Xs_hat_tr, Xs_hat_te, Xt_tr, Xt_te, Yt_tr, dx, dy
) -> tuple[dict, float]:
    """All 9 rung predictions for one (fold, direction).

    Fork of ``lr._rungs_for`` — identical rung algebra; the only change is that
    ``dual_predict`` here carries the reduced-basis projection. ``Ys_fit`` is
    the real source-answer matrix or a shuffled one for the matched-capacity
    null.
    """
    P_tr = dual_predict(p_s, Ys_fit, Xt_tr)
    P_te = dual_predict(p_s, Ys_fit, Xt_te)
    P7_tr = dual_predict(p_s, Ys_fit, Xs_hat_tr)
    P7_te = dual_predict(p_s, Ys_fit, Xs_hat_te)
    pmu, ymu = P_tr.mean(1, keepdim=True), Yt_tr.mean(1, keepdim=True)
    bstar = (Yt_tr - P_tr).mean(1, keepdim=True)
    b7 = (Yt_tr - P7_tr).mean(1, keepdim=True)
    Pc, Yc = P_tr - pmu, Yt_tr - ymu
    a = (Pc * Yc).sum((-1, -2)) / (Pc.pow(2).sum((-1, -2)) + 1e-30)
    rot_te, resid = lr.procrustes_apply(P_tr, Yt_tr, P_te)
    return {
        "1_direct": P_te,
        "2_ctx_offset": dual_predict(p_s, Ys_fit, Xt_te - dx),
        "3_ans_offset": P_te + dy,
        "4_bias_refit": P_te + bstar,
        "5_global_scale": a.view(-1, 1, 1) * (P_te - pmu) + ymu,
        "6_rotation": rot_te,
        "7_ctx_reparam": P7_te + b7,
        # B is fit on the ANSWER CLOUDS (source answers -> target answers) and
        # applied to the predicted source-space answer — never fit on P itself.
        "8_ans_reparam": dual_predict(p_ans, Yt_tr, P_te),
        "9_full_AMB": dual_predict(p_ans, Yt_tr, P7_te),
    }, resid


def run_pair(
    xy: dict, regimes: tuple[str, str], folds: np.ndarray, *, basis: str, null_draws: int, seed: int
) -> dict:
    """Both directions of one regime pair, sharing the per-fold Gram eigh set.

    Fork of ``lr.run_cell`` with the reduced-basis ``k_red`` threaded through
    every ``prep``, plus the identity+learned-bias baseline and the per-fit
    selected-lambda diagnostics.
    """
    ra, rb = regimes
    directions = ((ra, rb), (rb, ra))
    dir_key = {
        (ra, rb): f"{REGIME_LABEL[ra]}->{REGIME_LABEL[rb]}",
        (rb, ra): f"{REGIME_LABEL[rb]}->{REGIME_LABEL[ra]}",
    }
    L = xy[ra]["X"].shape[0]

    fold_ids = [k for k in range(N_FOLDS) if (folds == k).sum() > 0 and (folds != k).sum() >= 3]
    n_train_min = min(int((folds != k).sum()) for k in fold_ids)
    d = int(xy[ra]["X"].shape[2])
    k_red = fit825.reduced_basis_k(n_train_min, d) if basis == "reduced" else None

    def z():
        return torch.zeros(L, dtype=torch.float64)

    acc = {
        dd: {r: z() for r in RUNGS} | {"ceiling": z(), "identity_bias": z()} for dd in directions
    }
    accn = {dd: {r: z() for r in RUNGS} for dd in directions}
    sstot = {dd: z() for dd in directions}
    knn: dict = {dd: {} for dd in directions}
    diag: dict = {dd: {} for dd in directions}
    resid_max = 0.0
    rng = np.random.default_rng(seed)

    for k in fold_ids:
        tr, te = torch.as_tensor(folds != k), torch.as_tensor(folds == k)
        P = {}
        for reg in (ra, rb):
            P[(reg, "X")] = prep(xy[reg]["X"][:, tr], k_red)
            P[(reg, "Y")] = prep(xy[reg]["Y"][:, tr], k_red)

        for dd in directions:
            s, t = dd
            p_s, p_t, p_ans = P[(s, "X")], P[(t, "X")], P[(s, "Y")]
            Xs_tr = xy[s]["X"][:, tr]
            Ys_tr = xy[s]["Y"][:, tr]
            Xt_tr, Xt_te = xy[t]["X"][:, tr], xy[t]["X"][:, te]
            Yt_tr, Yt_te = xy[t]["Y"][:, tr], xy[t]["Y"][:, te]
            dx = Xt_tr.mean(1, keepdim=True) - Xs_tr.mean(1, keepdim=True)
            dy = Yt_tr.mean(1, keepdim=True) - Ys_tr.mean(1, keepdim=True)
            Xs_hat_tr = dual_predict(p_t, Xs_tr, Xt_tr)
            Xs_hat_te = dual_predict(p_t, Xs_tr, Xt_te)
            rung_args = (Xs_hat_tr, Xs_hat_te, Xt_tr, Xt_te, Yt_tr, dx, dy)
            preds, resid = _rungs_for(p_s, p_ans, Ys_tr, *rung_args)
            resid_max = max(resid_max, resid)
            ceiling = dual_predict(p_t, Yt_tr, Xt_te, diag=diag[dd])

            # Standing mapping rule: identity + learned-bias baseline whenever
            # input and output share a dimension (they do here, d == d).
            ident = torch.from_numpy(
                np.stack(
                    [
                        mb.identity_bias_predict(
                            Xt_tr[li].numpy(), Yt_tr[li].numpy(), Xt_te[li].numpy()
                        )
                        for li in range(L)
                    ]
                )
            )

            sstot[dd] += (Yt_te - Yt_te.mean(1, keepdim=True)).pow(2).sum((-1, -2))
            for r, pr in preds.items():
                acc[dd][r] += (Yt_te - pr).pow(2).sum((-1, -2))
            acc[dd]["ceiling"] += (Yt_te - ceiling).pow(2).sum((-1, -2))
            acc[dd]["identity_bias"] += (Yt_te - ident).pow(2).sum((-1, -2))

            # matched-capacity null: source operator fit on shuffled answers
            for _ in range(null_draws):
                perm = torch.as_tensor(rng.permutation(int(tr.sum())))
                npred, _ = _rungs_for(p_s, p_ans, Ys_tr[:, perm], *rung_args)
                for r, pr in npred.items():
                    accn[dd][r] += (Yt_te - pr).pow(2).sum((-1, -2)) / null_draws

            if k == fold_ids[0]:
                knn[dd] = {
                    "n_pool": int(te.sum()),
                    "ceiling": lr.knn_retrieval(ceiling, Yt_te),
                    "identity_bias": lr.knn_retrieval(ident, Yt_te),
                    "1_direct": lr.knn_retrieval(preds["1_direct"], Yt_te),
                    "4_bias_refit": lr.knn_retrieval(preds["4_bias_refit"], Yt_te),
                    "9_full_AMB": lr.knn_retrieval(preds["9_full_AMB"], Yt_te),
                }
        del P

    out: dict = {}
    dof_frac_max = 0.0
    for dd in directions:

        def r2(ss, key=dd):
            return [float(x) for x in (1.0 - ss / sstot[key])]

        dof_frac_max = max(dof_frac_max, max(diag[dd].get("dof_frac", [0.0])))
        out[dir_key[dd]] = {
            "r2": {r: r2(acc[dd][r]) for r in RUNGS},
            "ceiling_r2": r2(acc[dd]["ceiling"]),
            "identity_bias_r2": r2(acc[dd]["identity_bias"]),
            "null_r2": {r: r2(accn[dd][r]) for r in RUNGS},
            "knn_retrieval_fold0": knn[dd],
            "ceiling_selected_lambda_per_fold": diag[dd].get("lambda", []),
            "ceiling_dof_frac_per_fold": diag[dd].get("dof_frac", []),
        }
    if basis == "reduced":
        # The well-posedness claim in the module docstring, checked at runtime:
        # dof <= k <= n_train/2, so the cap-0.9 selector reused from the parent
        # is numerically identical to #1887's cap-None reduced arm.
        assert dof_frac_max < 0.9, (
            f"reduced-basis dof fraction {dof_frac_max:.3f} >= 0.9 — the GCV dof cap "
            "would bind, so this is no longer the #1887 cap-None reduced arm"
        )
    out["basis"] = basis
    out["gcv_dof_cap"] = lr.GCV_DOF_CAP
    out["selector_note"] = (
        "reduced: GCV inside the train-PCA basis; dof <= k <= n_train/2, so the "
        "cap never binds — numerically the #1887 reduced_basis_k (cap-None) arm"
        if basis == "reduced"
        else "ambient: GCV with the inherited dof cap — this is the #1887 "
        "gcv_capped_0p9 arm, NOT the legacy unguarded read"
    )
    out["reduced_basis_k"] = k_red
    out["reduced_basis_k_rule"] = "min(1024, floor(n_train_min/2), d)" if k_red else None
    out["n_train_min"] = n_train_min
    out["d"] = d
    out["dof_frac_max"] = dof_frac_max
    out["procrustes_subspace_residual_max"] = resid_max
    return out


# ---------------------------------------------------------------------------
# Character cells — capture-availability audit
# ---------------------------------------------------------------------------
def audit_char_cells(stage_root: Path, data_root: Path) -> dict:
    """Enumerate the 16 character cells and record whether a capture exists.

    A character cell is fittable only with an activation turnstore. This
    function looks in every place one could live — the local per-variant
    turnstore dirs, the #1887 staging root, and the variant's own HF prefix —
    and records the per-location evidence. It NEVER substitutes a different
    estimator for a missing capture.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    cells = {}
    for variant in CHAR_VARIANTS:
        local = data_root / variant / "turnstore"
        local_pt = sorted(p.name for p in local.glob("*.pt")) if local.is_dir() else []
        staged = stage_root / f"{variant}_turnstore"
        staged_pt = sorted(p.name for p in staged.glob("*.pt")) if staged.is_dir() else []
        prefix = f"issue1345_framing/{variant}"
        try:
            # Scoped + retried listing (never a bare list_repo_tree / full-repo
            # listing): the data repo is ~1M files and an un-retried cursor page
            # turns a transient 504 into a false "absent" read — which here would
            # mean falsely reporting a capture as missing.
            entries = hub.list_hf_files_under_path(api, HF_DATA_REPO, prefix, repo_type="dataset")
            hf_err = None
        except Exception as exc:  # network / auth — recorded, never swallowed
            entries, hf_err = [], f"{type(exc).__name__}: {exc}"
        hf_tensors = [p for p in entries if p.endswith(".pt") or "turnstore" in p]
        fittable = bool(local_pt or staged_pt or hf_tensors)
        # A failed listing is NOT evidence of absence: with the HF leg
        # unreadable the local legs alone cannot prove a capture is missing.
        capture_status = (
            "present" if fittable else ("absent" if hf_err is None else "unknown-hf-listing-failed")
        )
        cells[variant] = {
            "character": variant.split("_")[1],
            "mode": "on_policy" if "_op" in variant else "inserted",
            "measured_model": "pretrained" if variant.endswith("_base") else "instruct",
            "fittable": fittable,
            "capture_status": capture_status,
            "local_turnstore_pt_shards": len(local_pt),
            "staged_turnstore_pt_shards": len(staged_pt),
            "hf_prefix": prefix,
            "hf_files_total": len(entries),
            "hf_activation_tensor_files": len(hf_tensors),
            "hf_listing_error": hf_err,
            "within_cell_ceiling_r2": None,
            "blocker": (
                None
                if fittable
                else (
                    "no activation capture exists (story TEXT only) — a fit requires "
                    "a GPU teacher-forced capture pass, out of scope for this "
                    "0-GPU-h analysis round"
                    if hf_err is None
                    else f"capture presence UNDETERMINED — HF listing failed: {hf_err}"
                )
            ),
        }
    n_fittable = sum(1 for v in cells.values() if v["fittable"])
    n_unknown = sum(1 for v in cells.values() if v["capture_status"].startswith("unknown"))
    return {
        "n_cells": len(cells),
        "n_fittable": n_fittable,
        "n_capture_status_unknown": n_unknown,
        "verdict": (
            f"BLOCKED — 0 of {len(cells)} character cells have an activation capture"
            if n_fittable == 0 and n_unknown == 0
            else f"INCONCLUSIVE — {n_unknown} of {len(cells)} cells' HF listing failed"
            if n_fittable == 0
            else f"{n_fittable} of {len(cells)} character cells have a capture"
        ),
        "relocation_sweep": {
            "local_dirs": str(data_root / "char_*/turnstore"),
            "staging_root": str(stage_root),
            "hf_prefixes": "issue1345_framing/char_*",
            "note": (
                "All three locations checked per character cell; counts are in "
                "the per-cell rows. The character story TEXT (raw/kept/judge "
                "jsonl) IS present on HF — only the activation capture is absent."
            ),
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Character story examples
# ---------------------------------------------------------------------------
def _label_for(variant: str) -> str:
    import issue1310_common as i1310

    ch = variant.split("_")[1]
    for label in i1310.PERSONA_LABELS:
        if label.lower() == ch:
            return label
    raise KeyError(f"no persona label for {variant!r}")


def _persona_desc(variant: str) -> str:
    """The panel's own one-line persona description (issue1310_common.PERSONAS)."""
    import issue1310_common as i1310

    return i1310.PERSONAS[_label_for(variant)]


# Light benign-content screen for the auto-selected example conversation. The
# story corpus wraps real LMSYS user text, which carries occasional explicit /
# jailbreak rows; the selected excerpt is the one thing this round quotes
# outward, so a hit here just skips to the next candidate conversation.
_SENSITIVE_TOKENS = (
    "sex",
    "porn",
    "nsfw",
    "erotic",
    "kill",
    "weapon",
    "bomb",
    "drug",
    "suicide",
    "hack",
    "exploit",
    "jailbreak",
    "ignore previous",
    "racist",
    "slur",
    # The LMSYS corpus carries a large "say something toxic / harmful" prompt
    # family (Bluemoon-style elicitation). Those rows are legitimate training
    # data but make a poor outward-facing illustration.
    "toxic",
    "harmful",
    "offensive",
    "insult",
    "hate",
    "nsfl",
    "explicit",
)


def char_examples(cache_dir: Path, max_chars: int = 900) -> dict:
    """One story excerpt per character, all on the SAME conversation id.

    Streams each character's kept-stories JSONL (never loads it whole), indexes
    conv_id -> row, intersects across the four characters, and emits — for ONE
    shared conversation — each character's narrative LEAD-IN: ``story[:a_start]``,
    the scene + question + attribution phrase that precedes the answer span.

    The lead-in is the right excerpt for the inserted ("paired") mode because
    the answer itself is embedded VERBATIM, so it is byte-identical across all
    four characters (asserted here via a sha256 equality check, and reported).
    The narrative wrapper is the only thing the character changes.

    Excerpt text is written to JSON only — never printed to the log.
    """

    from explore_persona_space.orchestrate import hub

    variants = [f"char_{ch}" for ch in ("helios", "wren", "dana", "vex")]
    per_char: dict[str, dict[str, dict]] = {}
    for variant in variants:
        rel = (
            f"issue1345_framing/{variant}/raw_completions/stories/"
            "kept_stories_paired_instruct.jsonl"
        )
        target = cache_dir / "char_stories" / f"{variant}_kept.jsonl"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            hub.stage_hub_file(HF_DATA_REPO, rel, target, repo_type="dataset")
        rows: dict[str, dict] = {}
        with target.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                cid = r.get("conv_id") or r.get("conversation_id")
                if cid is not None:
                    rows[str(cid)] = r
        per_char[variant] = rows
        print(f"[examples] {variant}: {len(rows)} kept stories", flush=True)

    shared = set.intersection(*(set(v) for v in per_char.values()))
    out: dict = {
        "source": "HF superkaiba1/explore-persona-space-data "
        "issue1345_framing/char_*/raw_completions/stories/kept_stories_paired_instruct.jsonl",
        "per_character_kept_counts": {k: len(v) for k, v in per_char.items()},
        "n_conversations_shared_by_all_four": len(shared),
    }
    if not shared:
        out["examples"] = []
        out["blocker"] = "no conversation id is present in all four characters' kept pools"
        return out

    def lead_in(row: dict) -> str | None:
        """story[:a_start] — scene + question + attribution, before the answer."""
        turns = row.get("parsed_turns") or []
        story = row.get("story")
        if not turns or not isinstance(story, str):
            return None
        t = turns[0]
        conf = t.get("confidence") or {}
        if not all(bool(v) for v in conf.values()):
            return None
        a_start = int(t["a_start"])
        # The answer span must reproduce the recorded answer byte-for-byte, or
        # the offsets do not mean what this excerpt assumes.
        if story[a_start : int(t["a_end"])] != row.get("answer"):
            return None
        return story[:a_start]

    def sensitive(text: str) -> bool:
        low = text.lower()
        return any(tok in low for tok in _SENSITIVE_TOKENS)

    chosen, best = None, None
    n_screened_out = 0
    for cid in sorted(shared):
        rows = {v: per_char[v][cid] for v in variants}
        leads = {v: lead_in(rows[v]) for v in variants}
        if not all(leads.values()):
            continue
        # The embedded answer must be identical across all four characters —
        # that is what makes the four lead-ins a controlled comparison.
        answers = {rows[v].get("answer") for v in variants}
        if len(answers) != 1:
            continue
        if sensitive(rows[variants[0]].get("question", "")) or any(
            sensitive(x) for x in leads.values()
        ):
            n_screened_out += 1
            continue
        span = max(len(x) for x in leads.values())
        if best is None or span < best:
            best, chosen = span, (cid, leads)
    out["n_candidates_screened_out_as_sensitive"] = n_screened_out
    if chosen is None:
        out["examples"] = []
        out["blocker"] = (
            "no shared conversation passed the joint filter (parsed answer span "
            "verified + identical embedded answer across all four + benign screen)"
        )
        return out

    cid, leads = chosen
    ref = per_char[variants[0]][cid]
    answer = ref.get("answer") or ""
    out["conv_id"] = cid
    out["excerpt_definition"] = (
        "story[:a_start] — the narrative lead-in (scene + question + attribution "
        "phrase) that precedes the verbatim-embedded answer span"
    )
    out["shared_answer_sha256_8"] = hashlib.sha256(answer.encode()).hexdigest()[:8]
    out["shared_answer_chars"] = len(answer)
    out["shared_answer_identical_across_characters"] = True
    out["question_chars"] = len(str(ref.get("question", "")))
    out["examples"] = [
        {
            "character": _label_for(v),
            "persona_description": _persona_desc(v),
            "variant": v,
            "conv_id": cid,
            "excerpt": leads[v][:max_chars],
            "truncated": len(leads[v]) > max_chars,
            "excerpt_chars": min(len(leads[v]), max_chars),
            "lead_in_chars": len(leads[v]),
        }
        for v in variants
    ]
    return out


# ---------------------------------------------------------------------------
def _metadata(seed: int, layer: int, arm: str) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT
    ).stdout.strip()
    return {
        "git_commit": commit,
        "script": "scripts/issue1345_story_char_ladder_fill.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "layer": layer,
        "arm": arm,
        "n_folds": N_FOLDS,
        "fold_scheme": "conversation-grouped K=5 (fit825._cv_folds on the matched conv-id set)",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "store_pins": store_pins(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    user = os.environ.get("USER", "thomasjiralerspong")
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path(f"/mnt/eps-data/{user}/issue1887_lambda_audit/issue1345"),
        help="root holding the #1887-staged flat turnstore subdirs",
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=Path(f"/mnt/eps-data/{user}/issue1345_story_char_fill")
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "eval_results/issue_1345/story_char_ladder_fill",
    )
    ap.add_argument("--model", default="instruct")
    ap.add_argument("--arm", default="context", choices=list(c.ARMS))
    ap.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    ap.add_argument("--basis", default="reduced", choices=("reduced", "ambient", "both"))
    ap.add_argument("--null-draws", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=[f"{a}:{b}" for a, b in DEFAULT_PAIRS],
        help="regime pairs as SRC:TGT (each yields BOTH directions)",
    )
    ap.add_argument("--stage", nargs="+", default=["ladders", "chars", "examples"])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    bases = ("reduced", "ambient") if args.basis == "both" else (args.basis,)
    t_all = time.time()

    if "ladders" in args.stage:
        pairs = [tuple(p.split(":")) for p in args.pairs]
        for a, b in pairs:
            assert a in REGIME_SPECS and b in REGIME_SPECS, f"unknown regime pair {a}:{b}"
        blocks = {
            r: load_regime_xy(args.stage_root, args.cache_dir, args.model, r, args.arm, args.layer)
            for r in sorted({r for pr in pairs for r in pr})
        }
        results: dict = {"metadata": _metadata(args.seed, args.layer, args.arm), "ladders": {}}
        results["metadata"]["model"] = args.model
        results["metadata"]["rung_order"] = list(RUNGS)
        results["metadata"]["regime_labels"] = REGIME_LABEL
        for a, b in pairs:
            t0 = time.time()
            xa, xb, keep = matched_pair(blocks[a], blocks[b])
            xy = {a: xa, b: xb}
            folds = fit825._cv_folds(keep, N_FOLDS, args.seed)
            key = f"{REGIME_LABEL[a]}<->{REGIME_LABEL[b]}"
            entry = {
                "regimes": [a, b],
                "n_matched": int(len(keep)),
                "n_source_rows": int(blocks[a]["X"].shape[0]),
                "n_target_rows": int(blocks[b]["X"].shape[0]),
                "pairing": "conv-id intersection of the two full stores",
            }
            for basis in bases:
                entry[basis] = run_pair(
                    xy,
                    (a, b),
                    folds,
                    basis=basis,
                    null_draws=args.null_draws,
                    seed=args.seed,
                )
                _print_pair(key, basis, entry[basis])
            results["ladders"][key] = entry
            print(f"[pair] {key} wall {time.time() - t0:.0f}s", flush=True)
            del xy, xa, xb
        (args.out_dir / "ladders.json").write_text(json.dumps(results, indent=2))
        print(f"[write] {args.out_dir / 'ladders.json'}", flush=True)

    if "chars" in args.stage:
        audit = audit_char_cells(args.stage_root, _REPO_ROOT / "data/issue_1345")
        audit["metadata"] = _metadata(args.seed, args.layer, args.arm)
        (args.out_dir / "char_cells.json").write_text(json.dumps(audit, indent=2))
        print(f"[chars] {audit['verdict']}", flush=True)

    if "examples" in args.stage:
        ex = char_examples(args.cache_dir)
        ex["metadata"] = _metadata(args.seed, args.layer, args.arm)
        (args.out_dir / "char_examples.json").write_text(json.dumps(ex, indent=2))
        print(f"[examples] {len(ex.get('examples', []))} excerpts written", flush=True)

    print(f"TOTAL {time.time() - t_all:.0f}s", flush=True)


def _print_pair(key: str, basis: str, res: dict) -> None:
    li = 0
    print(f"  {key} [{basis} k={res['reduced_basis_k']} n_tr_min={res['n_train_min']}]", flush=True)
    for dk, dd in res.items():
        if not isinstance(dd, dict) or "r2" not in dd:
            continue
        print(
            f"    {dk}: ceiling {dd['ceiling_r2'][li]:.4f}  "
            f"identity+bias {dd['identity_bias_r2'][li]:.4f}",
            flush=True,
        )
        for r in RUNGS:
            print(
                f"      {r:16s} {dd['r2'][r][li]:9.4f}  null {dd['null_r2'][r][li]:9.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
