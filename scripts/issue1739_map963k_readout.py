"""Score #779's frozen 963,444-context map on #1739's eval rungs (0 GPU, VM-side).

Question: does the ~1M-context GENERIC map beat #1739's own 18,793-pair WildChat
ridge map for the map->persona-vector-projection read (arm 6) on #1739's REAL
eval distributions?

Design (matched-target by construction). Every arm is scored on the SAME context
set, the SAME DV, the SAME r_B, and the SAME metric inside ONE process — so the
963k arms are compared against a RECOMPUTED arm6/arm13 rather than against
differently-computed committed numbers. The committed
``arm_results/all_arms_spearman.json`` values enter only as a PARITY ANCHOR: the
recomputed arm6 is checked against the committed arm6 at the matched
(variant, layer, rung) slice, and any gap is reported, never silently absorbed.

Arms per (variant in {context_end, prefix_end}) x (layer in {14,19,26}):

- ``raw_proj``                  <z, r_B>                      (arm2_ctx_native analogue)
- ``map_i1739_ufull``           <M_1739(z), r_B>              (arm6_map_proj_e1 analogue)
- ``map_i1739_shuffled``        <M_1739^shuf(z), r_B>         (arm13_shuffled_map analogue)
- ``map963k_<fitter>``          <M_963k(z), r_B>              (the reuse arm under test)
- ``map963k_<fitter>_shuffled`` row-permuted 963k weights     (linear fitters only)
- ``oracle_proj``               <t1_actual, r_B>              (arm11_oracle_proj analogue)

Map application goes through the CANONICAL predict paths, never a
re-implementation: #779 payloads through ``issue779_ffc_n1m_fits.apply_map``
(the same call ``fits.apply_nl_map`` makes), #1739's linear map through
``fits.apply_map`` / ``fits.shuffled_map_weights``.

REUSE-VALIDITY read (reported alongside, per the artifact-reuse validity-domain
check): each map's reconstruction of the ACTUAL answer summary ``t1`` on #1739's
eval contexts — cosine + R^2. A map whose reconstruction collapses off its
training distribution cannot carry a read-out headline, and the number says so.

PREFIX-ARM CAVEAT (carried into every output row): #779's map was trained on
FULL-PROMPT end states (``c_last``), so applying it to ``prefix_end`` inputs is
an OUT-OF-TRAINING-DISTRIBUTION application. It is reported as the paired arm
(the both-arms mapping rule) with that caveat attached, never silently skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Before any heavy import: the shared-VM thread caps (#847) are frozen by numpy /
# torch at IMPORT, so load_dotenv() must run first for them to bind in-process.
load_dotenv()

import numpy as np  # noqa: E402

logger = logging.getLogger("map963k_readout")

LAYERS = (14, 19, 26)
VARIANTS = ("context_end", "prefix_end")
HIDDEN = 3584
N_BOOT = 1000
BOOT_SEED = 1739963
# r_B source the committed arm6/arm13 used — the only one a parity check is valid against.
ANCHOR_RB_SOURCE = "i1739_e1"


def _repo_root() -> Path:
    """Repo root by sentinel walk (script mode puts scripts/ on sys.path[0])."""
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "scripts" / "issue779_ffc_n1m_fits.py").is_file():
            return cand
    raise FileNotFoundError("repo root not found (no scripts/issue779_ffc_n1m_fits.py)")


def _n1m():
    """Import #779's N1M fitter module (its ``apply_map`` is the canonical path)."""
    root = _repo_root()
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    import issue779_ffc_n1m_fits as n1m

    return n1m


# ---------------------------------------------------------------- data loading


def load_dv(dv_json: Path) -> dict[str, dict]:
    """context_id -> {dv, rung, group_key, split} for contexts carrying a DV."""
    payload = json.loads(dv_json.read_text())
    out: dict[str, dict] = {}
    for row in payload["rows"]:
        dv = row.get("dv")
        if dv is None:
            continue
        out[row["context_id"]] = {
            "dv": float(dv),
            "rung": row.get("rung"),
            "group_key": row.get("group_key"),
            "split": row.get("split"),
        }
    return out


def load_per_context(store_dir: Path, dv: dict[str, dict]) -> dict:
    """Reduce the PER-ROLLOUT store slice to PER-CONTEXT arrays.

    EVERY kind is AVERAGED over the context's rollouts. For ``t1`` (answer-span
    mean) that is the substantive per-context answer summary. For the prompt-side
    kinds (``context_end`` / ``prefix_end``) the rollouts share the same prompt
    ids and positions, so averaging only denoises: a context's rollouts are
    captured in DIFFERENT padded batches, and bf16 padded-batch kernel numerics
    make single-position hidden states differ slightly between batches (the
    documented single-position bf16 class — max-abs deviation up to ~0.5 on
    components whose per-dim sd is ~1.2, which is batch numerics, NOT
    rollout-dependence). Averaging over exactly the rollouts the DV averages
    keeps arms and DV on the same rollout set.

    The cross-rollout agreement is GATED ON COSINE (the calibrated bar for
    single-position bf16 reads), with max-abs reported as a diagnostic only —
    a max-abs gate here reads batch numerics as corruption.
    """
    from explore_persona_space.experiments.issue_1739 import store_io

    kinds = ("context_end", "prefix_end", "t1")
    arrays, meta = store_io.load_summaries(store_dir, kinds, LAYERS, hidden_dim=HIDDEN)
    n_rows = len(meta)
    logger.info("[load] %s: %d store rows", store_dir.name, n_rows)

    keep = np.array([bool(m.get("context_id") in dv) for m in meta], dtype=bool)
    ctx_ids = np.array([m.get("context_id") for m in meta])
    uniq, first_idx, inv, counts = np.unique(
        ctx_ids[keep], return_index=True, return_inverse=True, return_counts=True
    )
    rows_keep = np.flatnonzero(keep)
    logger.info(
        "[load] %s: %d rows join the DV -> %d contexts (rollouts/context: min %d max %d)",
        store_dir.name,
        int(keep.sum()),
        len(uniq),
        int(counts.min()),
        int(counts.max()),
    )

    per_ctx: dict[tuple[str, int], np.ndarray] = {}
    for kind in kinds:
        for layer in LAYERS:
            arr = arrays[(kind, layer)][rows_keep]
            acc = np.zeros((len(uniq), HIDDEN), dtype=np.float64)
            np.add.at(acc, inv, arr.astype(np.float64))
            per_ctx[(kind, layer)] = acc / counts[:, None]

    # Cross-rollout agreement of the prompt-side kinds: cosine is the gate,
    # max-abs is a diagnostic (see the docstring's bf16 batch-numerics note).
    cos_min = 1.0
    max_dev = 0.0
    rng = np.random.default_rng(0)
    multi = np.flatnonzero(counts > 1)
    sample = rng.choice(multi, size=min(50, len(multi)), replace=False) if len(multi) else []
    for ci in sample:
        members = rows_keep[inv == ci]
        for kind in ("context_end", "prefix_end"):
            block = arrays[(kind, LAYERS[0])][members].astype(np.float64)
            max_dev = max(max_dev, float(np.abs(block - block[0]).max()))
            nrm = np.linalg.norm(block, axis=1, keepdims=True)
            bn = block / np.maximum(nrm, 1e-30)
            cos_min = min(cos_min, float((bn @ bn[0]).min()))
    if cos_min < 0.995:
        raise ValueError(
            f"{store_dir.name}: prompt-side summaries disagree across rollouts "
            f"(min cosine {cos_min:.6f} < 0.995, max abs dev {max_dev:.3e}) — the "
            "per-context reduction is unsafe (this is beyond bf16 batch numerics)"
        )
    logger.info(
        "[load] %s: cross-rollout prompt-side min cosine %.6f (max abs dev %.2e, %d contexts sampled)",
        store_dir.name,
        cos_min,
        max_dev,
        len(sample),
    )

    return {
        "context_ids": uniq,
        "per_ctx": per_ctx,
        "dv": np.array([dv[c]["dv"] for c in uniq], dtype=np.float64),
        "rung": np.array([dv[c]["rung"] for c in uniq]),
        "n_rollouts": counts,
        "rollout_prompt_side_min_cosine": cos_min,
        "rollout_prompt_side_max_abs_dev": max_dev,
    }


# ------------------------------------------------------------------ map arms


def load_rb_sources(
    behavior: str,
    i1739_dir: Path,
    bank_dir: Path = Path("data/issue_1739/hf_dl/r_b"),
) -> dict[str, tuple[np.ndarray, list[int]]]:
    """Available r_B banks for one behavior: ``{source -> ((28, d), layers)}``.

    ``i1739_e1`` is the E1 diff-of-means bank the committed arm6 actually used
    (``analysis_tensors/r_b_e1/<behavior>.npz``) — the anchor source, and the
    only one that makes a recomputed-vs-committed parity check meaningful.
    ``issue779_bank`` is the #779 r_B bank at RB_REVISION, which exists for all
    three behaviors and so is the only source available for sycophancy (whose
    E1 bank was not uploaded — its arm results were still in flight). Both are
    reported per row via ``r_b_source``; they are NOT interchangeable, so
    never compare a rho across sources.
    """
    out: dict[str, tuple[np.ndarray, list[int]]] = {}
    e1 = i1739_dir / f"r_b_e1_{behavior}.npz"
    if e1.is_file():
        with np.load(e1) as z:
            out["i1739_e1"] = (z["rb"].astype(np.float64), [int(v) for v in z["layers"]])
    bank = bank_dir / f"{behavior}.pt"
    if bank.is_file():
        import torch

        payload = torch.load(bank, map_location="cpu", weights_only=False)
        rb = np.asarray(payload["r_b"], dtype=np.float64)
        out["issue779_bank"] = (rb, [int(v) for v in payload["layers"]])
    return out


def load_963k_payload(maps_dir: Path, layer: int, fitter: str):
    """One #779 963k per-layer payload (torch dict; carries its own standardizer)."""
    import torch

    path = maps_dir / f"L{layer}_{fitter}.pt"
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def apply_963k(payload, x: np.ndarray, device: str, *, chunk: int = 2048) -> np.ndarray:
    """#779's canonical predict path, applied in ROW CHUNKS.

    ``issue779_ffc_n1m_fits.apply_map`` materializes the whole hidden layer at
    once — for the w=32768 MLP that is an ``(n, 32768)`` fp32 block (~2.3 GB at
    n=17,304) on top of the fp64 input/output copies, which on this shared VM
    invites an earlyoom kill rather than an exception. The map is row-independent,
    so chunking is bit-comparable arithmetic (each chunk takes the identical
    code path) while bounding peak memory to ~``chunk`` rows.
    """
    import torch

    n1m = _n1m()
    dev = torch.device(device)
    if len(x) <= chunk:
        return n1m.apply_map(payload, x, dev)
    return np.concatenate(
        [n1m.apply_map(payload, x[i : i + chunk], dev) for i in range(0, len(x), chunk)],
        axis=0,
    )


def load_i1739_map(
    path: Path, layer_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """#1739's own u-full linear map at one layer: (w, x_mu, x_sd, y_mu, meta).

    ``meta`` is the payload's persisted meta dict (#1975: carries the
    ``fit_space`` / input-space parity fields on re-persisted payloads; a
    legacy payload's meta lacks them and the parity check warns loudly).
    """
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        layers = list(z["layers"])
        li = layers.index(layer_index)
        return (
            z["w"][li].astype(np.float64),
            z["x_mu"][li].astype(np.float64),
            z["x_sd"][li].astype(np.float64),
            z["y_mu"][li].astype(np.float64),
            meta,
        )


def shuffle_rows(w: np.ndarray, seed: int) -> np.ndarray:
    """Row-permuted (input-dim) weights — Frobenius-preserving, the arm-13 control."""
    rng = np.random.default_rng([1739, 963, int(seed)])
    out = w[rng.permutation(w.shape[0])]
    assert np.isclose(np.linalg.norm(w), np.linalg.norm(out), rtol=1e-12)
    return out


# -------------------------------------------------------------------- metrics


def _rank(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Average-tie ranks along ``axis`` (Spearman needs tie-averaging)."""
    from scipy.stats import rankdata

    return rankdata(a, axis=axis)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho; nan when either side is constant (rho undefined)."""
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


class BootCache:
    """Shared bootstrap index matrix + centred DV ranks for one (rung) sample.

    Every arm scored on a rung is resampled with the SAME index matrix (same
    seed, same n), and the DV side of the correlation does not vary by arm — so
    the DV's per-draw ranks are computed ONCE per rung and reused across all
    arms and layers. Only the arm-score ranks are re-ranked per call, which is
    the irreducible cost of an exact per-draw Spearman bootstrap.
    """

    def __init__(self, y: np.ndarray, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> None:
        self.n = len(y)
        self.n_boot = n_boot
        self.ok = self.n >= 5
        if not self.ok:
            return
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, self.n, size=(n_boot, self.n))
        ry = _rank(y[self.idx], axis=1)
        self.ry = ry - ry.mean(axis=1, keepdims=True)
        self.ry_ss = (self.ry**2).sum(axis=1)

    def ci(self, x: np.ndarray) -> tuple[float, float]:
        """Percentile CI of Spearman rho for one arm score against the cached DV."""
        if not self.ok:
            return (float("nan"), float("nan"))
        rx = _rank(x[self.idx], axis=1)
        rx = rx - rx.mean(axis=1, keepdims=True)
        denom = np.sqrt((rx**2).sum(axis=1) * self.ry_ss)
        with np.errstate(invalid="ignore", divide="ignore"):
            rhos = (rx * self.ry).sum(axis=1) / denom
        rhos = rhos[np.isfinite(rhos)]
        if rhos.size < 10:
            return (float("nan"), float("nan"))
        return (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)))


def recon_quality(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Map reconstruction of the ACTUAL answer summary on the eval distribution."""
    pn = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-30)
    an = actual / np.maximum(np.linalg.norm(actual, axis=1, keepdims=True), 1e-30)
    cos = float(np.mean(np.sum(pn * an, axis=1)))
    ss_res = float(((actual - pred) ** 2).sum())
    ss_tot = float(((actual - actual.mean(axis=0)) ** 2).sum())
    return {"cosine_mean": cos, "r2": 1.0 - ss_res / max(ss_tot, 1e-30)}


# ---------------------------------------------------------------- committed anchor


def committed_arm_rho(arms_json: Path | None, arm: str, variant: str, layer: int) -> dict:
    """Committed rho PER EVAL RUNG for one arm at the U=full / E1 slice.

    Searches ``arm_rows`` (which carry ``eval_rung: train``) AND
    ``transfer_rows`` (which carry the hhrt / toxicchat transfer rungs), so a
    recomputed arm can be anchored on every rung it scores. Within a rung the
    max-``budget_l`` row is taken; ``u_rung_label == "full"`` is preferred and
    that filter is relaxed only when the arm has no full-U row at all — the
    returned entry records ``u_full_relaxed`` so a relaxed anchor is visible.
    """
    if arms_json is None or not arms_json.is_file():
        return {}
    payload = json.loads(arms_json.read_text())

    def _pick(rows: list[dict]) -> list[dict]:
        full = [r for r in rows if str(r.get("u_rung_label")) == "full"]
        return full or rows

    out: dict[str, dict] = {}

    # arm_rows carry a FULL 28-entry rho_per_layer curve, so the anchor exists at
    # EVERY layer — the row's own `layer` field is only the cell's SELECTED
    # read-out layer (sparse across cells) and must not be used to filter.
    ar = _pick(
        [
            r
            for r in payload.get("arm_rows", [])
            if r.get("arm") == arm and r.get("variant") == variant and r.get("regime") == "e1"
        ]
    )
    if ar:
        best = max(ar, key=lambda r: r.get("budget_l") or 0)
        curve = best.get("rho_per_layer") or []
        rung = best.get("eval_rung")
        if rung and len(curve) > layer and curve[layer] is not None:
            out[rung] = {
                "rho_frozen": float(curve[layer]),
                "source": "arm_rows.rho_per_layer",
                "selected_layer_of_row": best.get("layer"),
                "budget_l": best.get("budget_l"),
                "u_rung_label": best.get("u_rung_label"),
                "n_matching_rows": len(ar),
                "u_full_relaxed": not any(
                    str(r.get("u_rung_label")) == "full" for r in payload.get("arm_rows", [])
                ),
            }

    # transfer_rows carry no per-layer curve — anchor only where the row's own
    # selected layer IS the layer under test (so transfer coverage is sparse).
    tr = _pick(
        [
            r
            for r in payload.get("transfer_rows", [])
            if r.get("arm") == arm
            and r.get("variant") == variant
            and r.get("regime") == "e1"
            and r.get("layer") == layer
        ]
    )
    for rung in sorted({r.get("eval_rung") for r in tr if r.get("eval_rung")}):
        cand = [r for r in tr if r.get("eval_rung") == rung]
        best = max(cand, key=lambda r: r.get("budget_l") or 0)
        out[rung] = {
            "rho_frozen": best.get("rho_frozen"),
            "ci_frozen": best.get("ci_frozen"),
            "source": "transfer_rows.rho_frozen",
            "budget_l": best.get("budget_l"),
            "u_rung_label": best.get("u_rung_label"),
            "n_eval": best.get("n_eval"),
            "n_matching_rows": len(cand),
        }
    return out


# --------------------------------------------------------------------- driver


def run_behavior(
    behavior: str,
    *,
    store_dir: Path,
    dv_json: Path,
    arms_json: Path | None,
    maps_dir: Path,
    i1739_dir: Path,
    fitters: tuple[str, ...],
    device: str,
) -> dict:
    from explore_persona_space.experiments.issue_1739 import fits

    dv = load_dv(dv_json)
    data = load_per_context(store_dir, dv)
    rb_sources = load_rb_sources(behavior, i1739_dir)
    if not rb_sources:
        raise FileNotFoundError(f"{behavior}: no r_B source resolved")
    logger.info("[%s] r_B sources: %s", behavior, sorted(rb_sources))

    primary_rb = ANCHOR_RB_SOURCE if ANCHOR_RB_SOURCE in rb_sources else sorted(rb_sources)[0]
    rungs = [r for r in sorted(set(data["rung"].tolist())) if int((data["rung"] == r).sum()) >= 5]
    rung_masks = {r: data["rung"] == r for r in rungs}
    # One bootstrap index matrix + DV rank cache per rung, shared by every arm/layer.
    boot = {r: BootCache(data["dv"][rung_masks[r]]) for r in rungs}
    logger.info(
        "[%s] rungs %s (n=%s) | primary r_B for CIs: %s",
        behavior,
        rungs,
        [int(rung_masks[r].sum()) for r in rungs],
        primary_rb,
    )
    results: list[dict] = []
    recon: list[dict] = []

    for variant in VARIANTS:
        i1739_map_path = i1739_dir / f"map_i1739_{variant}_ufull.npz"
        for layer in LAYERS:
            z = data["per_ctx"][(variant, layer)]
            t1 = data["per_ctx"][("t1", layer)]
            # Predictions are r_B-independent: build each map's (n, d) output ONCE,
            # then project onto every r_B source (projection is trivially cheap).
            preds: dict[str, np.ndarray] = {"raw_proj": z, "oracle_proj": t1}

            # --- #1739's own u-full linear map (arm6) + its shuffled control (arm13)
            w, x_mu, x_sd, y_mu, map_meta = load_i1739_map(i1739_map_path, layer)
            # #1975 input-space parity: `z` is RAW per-context store summaries
            # while the u-full map was FIT in the whitened main-grid space — a
            # DELIBERATE cross-space reuse-validity read (module docstring;
            # recon_quality reports the collapse per row, and
            # issue1739_map963k_applycheck.py is the standalone probe).
            # Declared, never silent (the #1739 incident class).
            fits.assert_map_input_space(
                map_meta,
                z,
                declared_mismatch=(
                    "map963k_readout scores the whitened-fit u-full map on RAW per-context "
                    "store summaries (disclosed reuse-validity read; see module docstring + "
                    "issue1739_map963k_applycheck.py)"
                ),
            )
            preds["map_i1739_ufull"] = ((z - x_mu) / x_sd) @ w + y_mu
            preds["map_i1739_shuffled"] = ((z - x_mu) / x_sd) @ shuffle_rows(w, 0) + y_mu
            del w

            # --- #779 963k maps (the reuse arms under test)
            for fitter in fitters:
                payload = load_963k_payload(maps_dir, layer, fitter)
                if payload is None:
                    logger.warning("[%s] missing 963k payload L%d %s", behavior, layer, fitter)
                    continue
                preds[f"map963k_{fitter}"] = apply_963k(payload, z, device)
                if payload.get("kind") == "ridge":
                    import torch

                    shuf = dict(payload)
                    shuf["W"] = torch.as_tensor(
                        shuffle_rows(payload["W"].numpy().astype(np.float64), 0)
                    ).float()
                    preds[f"map963k_{fitter}_shuffled"] = apply_963k(shuf, z, device)
                    del shuf
                del payload

            # Reconstruction of the ACTUAL answer summary, per map (r_B-free read).
            for name, pred in preds.items():
                if name in ("raw_proj", "oracle_proj"):
                    continue
                recon.append(
                    {
                        "behavior": behavior,
                        "variant": variant,
                        "layer": layer,
                        "map": name,
                        "pred_norm_mean": float(np.linalg.norm(pred, axis=1).mean()),
                        "actual_t1_norm_mean": float(np.linalg.norm(t1, axis=1).mean()),
                        **recon_quality(pred, t1),
                    }
                )

            scores: dict[tuple[str, str], np.ndarray] = {}
            for rb_name, (rb_all, rb_layers) in rb_sources.items():
                rb = rb_all[rb_layers.index(layer)]
                for name, pred in preds.items():
                    scores[(name, rb_name)] = pred @ rb
            del preds

            # --- score every arm against the DV, per eval rung
            for rung in rungs:
                m = rung_masks[rung]
                y = data["dv"][m]
                for (arm, rb_name), s in scores.items():
                    # CIs are computed for the PRIMARY r_B source only; the second
                    # source is a point-estimate sensitivity read, and bootstrapping
                    # both would double the run's dominant cost for no new claim.
                    if rb_name == primary_rb:
                        lo, hi = boot[rung].ci(s[m])
                    else:
                        lo, hi = (None, None)
                    results.append(
                        {
                            "behavior": behavior,
                            "variant": variant,
                            "layer": layer,
                            "eval_rung": rung,
                            "arm": arm,
                            "r_b_source": rb_name,
                            "rho": spearman(s[m], y),
                            "ci95": [lo, hi],
                            "ci_computed": rb_name == primary_rb,
                            "n_contexts": int(m.sum()),
                            "dv_std": float(y.std()),
                            "ood_prefix_application": variant == "prefix_end"
                            and arm.startswith("map963k"),
                        }
                    )
            logger.info("[%s] %s L%d: scored %d arms", behavior, variant, layer, len(scores))

    # Parity anchor: recomputed arm6/arm13 vs the committed values.
    anchors = []
    for variant in VARIANTS:
        for layer in LAYERS:
            for arm, mine in (
                ("arm6_map_proj_e1", "map_i1739_ufull"),
                ("arm13_shuffled_map", "map_i1739_shuffled"),
                ("arm2_ctx_native", "raw_proj"),
            ):
                per_rung = committed_arm_rho(arms_json, arm, variant, layer)
                for rung, c in per_rung.items():
                    rows = [
                        r
                        for r in results
                        if r["variant"] == variant
                        and r["layer"] == layer
                        and r["arm"] == mine
                        and r["eval_rung"] == rung
                        and r["r_b_source"] == ANCHOR_RB_SOURCE
                    ]
                    mine_rho = rows[0]["rho"] if rows else None
                    anchors.append(
                        {
                            "behavior": behavior,
                            "variant": variant,
                            "layer": layer,
                            "committed_arm": arm,
                            "recomputed_arm": mine,
                            "eval_rung": rung,
                            "committed_rho": c.get("rho_frozen"),
                            "committed_ci": c.get("ci_frozen"),
                            "committed_n_eval": c.get("n_eval"),
                            "committed_budget_l": c.get("budget_l"),
                            "committed_u_rung_label": c.get("u_rung_label"),
                            "committed_u_full_relaxed": c.get("u_full_relaxed"),
                            "recomputed_rho": mine_rho,
                            "recomputed_n_contexts": rows[0]["n_contexts"] if rows else None,
                            "abs_gap": (
                                abs(mine_rho - c["rho_frozen"])
                                if mine_rho is not None and c.get("rho_frozen") is not None
                                else None
                            ),
                        }
                    )

    return {
        "behavior": behavior,
        "n_contexts": int(len(data["context_ids"])),
        "rungs": rungs,
        "rollout_prompt_side_min_cosine": data["rollout_prompt_side_min_cosine"],
        "rollout_prompt_side_max_abs_dev": data["rollout_prompt_side_max_abs_dev"],
        "rows": results,
        "recon": recon,
        "parity_anchor": anchors,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behaviors", default="evil,hallucination,sycophancy")
    ap.add_argument("--slice-root", type=Path, default=Path("data/issue_1739/hf_dl/evalslice"))
    ap.add_argument("--maps-dir", type=Path, default=Path("data/issue_1739/hf_dl/map963k"))
    ap.add_argument("--i1739-dir", type=Path, default=Path("data/issue_1739/hf_dl/i1739_tensors"))
    ap.add_argument("--dv-root", type=Path, default=Path("eval_results/issue_1739/dv_dataset"))
    ap.add_argument("--arms-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument(
        "--out", type=Path, default=Path("eval_results/issue_1739/map963k_reuse/comparison.json")
    )
    ap.add_argument("--fitters", default="ridge,mlp_w8192")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    t0 = time.time()
    fitters = tuple(f.strip() for f in args.fitters.split(",") if f.strip())
    out: dict = {
        "meta": {
            "question": (
                "does #779's frozen 963,444-context map beat #1739's own 18,793-pair "
                "map for the map->persona-vector projection read (arm6) on #1739's eval rungs?"
            ),
            "map963k_source": {
                "repo": "superkaiba1/explore-persona-space-data",
                "weights_prefix": "issue779_monitoring/n1m_readout/weights/",
                "revision": "9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5",
                "layers": list(LAYERS),
                "fitters": list(fitters),
                "train_contexts": 963444,
                "input_semantics": "c_last (last prompt token, chat template + generation prompt)",
                "target_semantics": "v_x (mean-response activation) == #1739 t1 (answer-span mean)",
                "preprocessing": "self-contained per-payload xmu/xsd/ymu; applied via issue779_ffc_n1m_fits.apply_map",
                "layer_convention": "block index L == hidden_states[L+1] (post-block); matches #1739 hidden_states[1:]",
            },
            "i1739_map_source": {
                "prefix": "issue1739_ctxmap/analysis_tensors/maps/{variant}__ufull.npz",
                "w_fit_rows": 18793,
            },
            "r_b": "issue1739_ctxmap/analysis_tensors/r_b_e1/{behavior}.npz (E1 regime — the arm6 r_B)",
            "eval_store": "issue1739_ctxmap/capture_store/{behavior}_labeling/*.tar (streamed slice)",
            "metric": "Spearman rho of arm score vs judged graded DV, per eval rung; 2000-draw percentile bootstrap over contexts",
            "caveats": [
                "prefix_end + map963k rows are an OUT-OF-TRAINING-DISTRIBUTION application: "
                "#779's map was fit on full-prompt end states (c_last), not prefix ends "
                "(flagged per-row as ood_prefix_application).",
                "per-context reduction: EVERY kind is averaged over the context's rollouts. "
                "Prompt-side kinds are rollout-invariant up to bf16 padded-batch numerics "
                "(gated on cross-rollout cosine >= 0.995; max-abs is diagnostic only).",
                "sycophancy is excluded — its arm results were still in flight at run time.",
            ],
            "n_boot": N_BOOT,
            "ci_scope": "primary r_B source only (see r_b_source / ci_computed per row)",
            "device": args.device,
        },
        "behaviors": {},
    }
    for behavior in [b.strip() for b in args.behaviors.split(",") if b.strip()]:
        store_dir = args.slice_root / behavior
        dv_json = args.dv_root / behavior / "labeling.json"
        arms_json = args.arms_root / behavior / "arm_results" / "all_arms_spearman.json"
        for p in (store_dir, dv_json):
            if not p.exists():
                raise FileNotFoundError(f"{behavior}: missing {p}")
        if not arms_json.is_file():
            # A behavior whose #1739 arm grid has not landed yet (sycophancy at run
            # time) still gets every 963k / arm6-recompute arm — it just has no
            # committed value to anchor against, which the empty parity_anchor
            # records. Never a silent skip of the behavior itself.
            logger.warning(
                "[%s] no committed arm results at %s — parity anchor will be EMPTY "
                "(963k vs recomputed-arm6 comparison is unaffected)",
                behavior,
                arms_json,
            )
            arms_json = None
        out["behaviors"][behavior] = run_behavior(
            behavior,
            store_dir=store_dir,
            dv_json=dv_json,
            arms_json=arms_json,
            maps_dir=args.maps_dir,
            i1739_dir=args.i1739_dir,
            fitters=fitters,
            device=args.device,
        )

    out["meta"]["elapsed_s"] = round(time.time() - t0, 1)
    out["meta"]["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s (%.0fs)", args.out, time.time() - t0)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
