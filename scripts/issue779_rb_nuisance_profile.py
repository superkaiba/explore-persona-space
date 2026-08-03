# ruff: noqa: RUF002, RUF003
"""#779 inline free-analysis (user-chat, 2026-07-28): r_B vs residual-stream nuisance axes.

Instrument characterization, NOT a behavioural read. Profiles the project's
persona-vector directions (`r_B`, mean-difference over judge-filtered contrastive
rollouts) against the documented nuisance axes of the Qwen-2.5-7B residual stream
-- massive activations / rogue dimensions, the generic mean direction, and the
low-rank "dense scaffold" of top principal components -- so downstream
persona-alignment reads (`|cos(decoder_i, r_B)|` in #1482/#1738/#1092) can state
whether `r_B` itself is contaminated by them.

Axis definitions + motivation: `docs/lit_reviews/residual-stream-direction-taxonomy-integration.md`
(section 1 rows on massive activations / dense scaffold; section 2 item 2 for the
gamma = ||mu||/||sigma|| + mean-centre-before-any-cosine prescription; gap rank 4
in section 4). A difference of means cancels shared additive constants, so `r_B`
is EXPECTED to be clean -- this script verifies rather than assumes.

Reads (all vectorized; nulls seeded):

1. gamma = ||mu|| / ||sigma|| per state type per capture layer -- the
   rogue-dimension severity number.
2. Rogue-dim overlap -- top-K massive-activation dims by |mu| AND by mu^2/sigma^2
   (both rankings reported), then `r_B`'s squared-mass fraction on those dims vs
   a random-direction null (exact Beta(K/2, (D-K)/2) quantiles + a sampled band).
3. Scaffold alignment -- |cos(r_B, mu)| (the generic mean direction), and `r_B`'s
   squared-mass fraction inside the top-48 principal subspace of the population,
   computed BOTH uncentred (raw second-moment SVD, whose leading direction is
   ~mu-hat) and mean-centred (true PCA), vs the 48/D null. Disagreement between
   the raw and centred reads means the raw number is the artifact.
4. Norm + self-concentration profile of `r_B` across all 28 layers (bank-only):
   ||r_B|| per layer plus the mass of the direction's OWN top-1 / top-10
   coordinates vs a self-top-K random-direction null (a DIFFERENT null family
   from read 2's fixed-subspace one -- see `nulls` in the output JSON).

Scope limit: the mu/sigma- and PCA-derived reads (1-3) need activation statistics
and therefore exist only at the three capture layers {14, 19, 26} the #779 n1m
store holds. Read 4 covers all 28 bank layers (no activations needed).

Inputs (existing, revision-pinned, read-only):

- `r_B` bank: `issue779_monitoring/r_b/{trait}.pt` @ 037fcbb (3 traits x 28 layers).
- Activation statistics: a bounded, deterministic subsample of the n1m capture
  chunks `issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture/`
  @ cbc55efdd7 -- `cx_last` (context-end state) and `v_x` (mean-answer state) at
  layers {14, 19, 26}. Only tensor fields are read; the chunks' `prompts` field
  (real-world corpus text) is never touched, printed, or persisted.

Usage::

    uv run python scripts/issue779_rb_nuisance_profile.py                # 10 chunks (5000 rows)
    uv run python scripts/issue779_rb_nuisance_profile.py --chunks 2 \\
        --out-dir /tmp/i779-rbnuis --fig-dir /tmp/i779-rbnuis            # fast scratch probe
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps BEFORE numpy/torch (shared-VM rule)

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import beta as beta_dist  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue779_rb_nuisance_profile")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = "superkaiba1/explore-persona-space-data"

RB_PREFIX = "issue779_monitoring/r_b"
RB_REVISION = "037fcbb"
TRAITS = ("evil", "sycophancy", "hallucination")

CAPTURE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
CAPTURE_REVISION = "cbc55efdd7"
CAPTURE_N_SHARDS = 32
CAPTURE_CHUNKS_PER_SHARD = 60
CAPTURE_LAYERS = (14, 19, 26)

# state field -> plain-English name (the two summaries the n1m capture holds)
STATE_FIELDS = {"cx_last": "context-end state", "v_x": "mean-answer state"}

D_MODEL = 3584
N_LAYERS = 28
ROGUE_KS = (1, 5, 10, 48)  # fixed-subspace K values (rogue-dim overlap)
SELF_KS = (1, 10)  # self-top-K values (r_B's own coordinate concentration)
PCA_K = 48  # the project's "pca48 scaffold"
N_NULL_DRAWS = 1000
NULL_SEED = 0
PCT = (2.5, 50.0, 97.5)


# ---------------------------------------------------------------------------
# staging (revision-pinned, idempotent)
# ---------------------------------------------------------------------------


def default_scratch() -> Path:
    """Data-disk staging root for the capture subsample (never `/` or `/tmp`)."""
    data_disk = Path("/mnt/eps-data") / os.environ.get("USER", "eps")
    if data_disk.is_dir():
        return data_disk / "issue779_rb_nuisance"
    return PROJECT_ROOT / "data" / "issue_779" / "rb_nuisance_dl" / "capture"


def chunk_names(n_chunks: int) -> list[str]:
    """Deterministic subsample spread across BOTH the shard and chunk axes.

    The n1m store is 32 shards x 60 chunks x 500 rows; taking the first N chunks
    of shard 00 would sample one contiguous corpus slice, so step evenly through
    both axes instead. Not a uniform random sample of the 960k rows -- stated as
    a scope limit in the output JSON.
    """
    if not 1 <= n_chunks <= min(CAPTURE_N_SHARDS, CAPTURE_CHUNKS_PER_SHARD):
        raise ValueError(f"--chunks must be in [1, 32], got {n_chunks}")
    s_step = CAPTURE_N_SHARDS // n_chunks
    c_step = CAPTURE_CHUNKS_PER_SHARD // n_chunks
    return [f"shard{i * s_step:02d}_chunk{i * c_step:04d}.pt" for i in range(n_chunks)]


def stage_pinned_file(path_in_repo: str, revision: str, dest_root: Path) -> Path:
    """Materialize ONE pinned data-repo file under `dest_root` (idempotent).

    Routes through the canonical staging helper `hub.stage_hub_file` (#1402) --
    retried via `retry_transient`, published atomically from a tempdir inside
    the target's own parent, fail-loud on exhaustion -- rather than a bare
    `hf_hub_download`. The repo path is mirrored verbatim under `dest_root` so
    re-runs (and the sibling #1482 staging tree) resolve the same layout.
    """
    from explore_persona_space.orchestrate.hub import stage_hub_file

    return stage_hub_file(
        DATA_REPO,
        path_in_repo,
        dest_root / path_in_repo,
        repo_type="dataset",
        revision=revision,
    )


def load_rb_bank(dest_root: Path) -> dict[str, np.ndarray]:
    """Load the 3-trait x 28-layer `r_B` bank at the pinned revision."""
    bank: dict[str, np.ndarray] = {}
    for trait in TRAITS:
        p = stage_pinned_file(f"{RB_PREFIX}/{trait}.pt", RB_REVISION, dest_root)
        obj = torch.load(p, map_location="cpu", weights_only=False)
        assert obj["trait"] == trait, (obj["trait"], trait)
        r = obj["r_b"].to(torch.float64).numpy()
        assert r.shape == (N_LAYERS, D_MODEL), r.shape
        assert list(obj["layers"]) == list(range(N_LAYERS)), obj["layers"][:5]
        assert np.isfinite(r).all(), f"{trait}: non-finite r_B"
        bank[trait] = r
        logger.info("[bank] %s r_B %s (norms %.3f..%.3f)", trait, r.shape, *_norm_range(r))
    return bank


def _norm_range(r: np.ndarray) -> tuple[float, float]:
    n = np.linalg.norm(r, axis=1)
    return float(n.min()), float(n.max())


# ---------------------------------------------------------------------------
# activation statistics (two streaming passes; float64 accumulation)
# ---------------------------------------------------------------------------


def _chunk_matrix(path: Path, field: str, layer: int) -> np.ndarray:
    """(N, D) float64 slice of one capture chunk. Reads ONLY tensor fields."""
    b = torch.load(path, map_location="cpu", weights_only=True)
    col = list(b["layers"]).index(layer)
    x = b[field][:, col, :]
    assert x.shape[1] == D_MODEL, x.shape
    return x.to(torch.float64).numpy()


def moment_pass(paths: list[Path]) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Pass 1: per-dim count / sum / sum-of-squares for every (field, layer)."""
    acc = {
        (f, ell): {"n": 0, "s": np.zeros(D_MODEL), "ss": np.zeros(D_MODEL)}
        for f in STATE_FIELDS
        for ell in CAPTURE_LAYERS
    }
    for i, p in enumerate(paths):
        for (f, ell), a in acc.items():
            x = _chunk_matrix(p, f, ell)
            a["n"] += x.shape[0]
            a["s"] += x.sum(0)
            a["ss"] += (x * x).sum(0)
        logger.info("[moments] chunk %d/%d (%s)", i + 1, len(paths), p.name)
    out = {}
    for key, a in acc.items():
        n = int(a["n"])
        mu = a["s"] / n
        var = (a["ss"] - n * mu * mu) / (n - 1)
        var = np.maximum(var, 0.0)  # roundoff guard; sd==0 dims are reported below
        out[key] = {"n": n, "mu": mu, "sd": np.sqrt(var), "var": var}
    return out


def subspace_pass(
    paths: list[Path], field: str, layer: int, mu: np.ndarray
) -> dict[str, np.ndarray]:
    """Pass 2 for ONE (field, layer): top-`PCA_K` uncentred + centred subspaces.

    Accumulates BOTH the raw second-moment matrix `X.T @ X` and the centred
    scatter `sum_i (x_i - mu)(x_i - mu).T` DIRECTLY (rather than subtracting
    `n mu mu.T` afterwards) because in the rogue-dimension regime `||mu||` can
    dominate the spread and the subtraction then cancels catastrophically.
    """
    g_raw = np.zeros((D_MODEL, D_MODEL))
    g_cen = np.zeros((D_MODEL, D_MODEL))
    for i, p in enumerate(paths):
        x = _chunk_matrix(p, field, layer)
        g_raw += x.T @ x
        xc = x - mu
        g_cen += xc.T @ xc
        logger.info("[pca %s L%d] chunk %d/%d", field, layer, i + 1, len(paths))
    out = {}
    for name, g in (("uncentred", g_raw), ("centred", g_cen)):
        w, v = np.linalg.eigh(g)  # ascending eigenvalues
        out[f"basis_{name}"] = np.ascontiguousarray(v[:, -PCA_K:])
        out[f"eigvals_{name}"] = w[-PCA_K:][::-1].copy()
        out[f"trace_{name}"] = np.array(float(np.trace(g)))
    out["diag_raw"] = np.diag(g_raw).copy()
    out["diag_centred"] = np.diag(g_cen).copy()
    # Variance share along the mean direction -- the "fair" comparator for
    # |cos(r_B, mu)|: how much of the population's spread that one direction
    # explains, next to the 1/D a random direction would place on it.
    mu_hat = mu / np.linalg.norm(mu)
    out["quad_mu_centred"] = np.array(float(mu_hat @ g_cen @ mu_hat))
    return out


# ---------------------------------------------------------------------------
# nulls
# ---------------------------------------------------------------------------


def build_nulls() -> dict:
    """Random-direction null bands for the three mass/alignment read families.

    Two DISTINCT families (never interchange them):

    - `fixed_subspace[K]`: squared mass of a random unit vector inside a FIXED
      K-dimensional subspace. Rotation-invariant, so one band serves every
      coordinate subspace (rogue dims) AND the data-derived PCA subspace, at
      every layer. Exact law: Beta(K/2, (D-K)/2), mean K/D.
    - `self_topk[K]`: squared mass of a random unit vector's OWN top-K
      coordinates -- an order statistic, much larger than K/D. Used only for
      `r_B`'s self-concentration profile.

    Plus `abs_cos_fixed_direction`: |cos| between a random unit vector and a
    FIXED direction (exact mean = Gamma(D/2) / (sqrt(pi) Gamma((D+1)/2))).
    """
    rng = np.random.default_rng(NULL_SEED)
    u = rng.standard_normal((N_NULL_DRAWS, D_MODEL))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    sq = u * u

    fixed: dict[str, dict] = {}
    for k in sorted(set(ROGUE_KS) | {PCA_K}):
        sampled = sq[:, :k].sum(1)  # exchangeable coords => any K-subspace
        lo, mid, hi = (float(x) for x in np.percentile(sampled, PCT))
        b_lo, b_hi = beta_dist.ppf([0.025, 0.975], k / 2.0, (D_MODEL - k) / 2.0)
        fixed[str(k)] = {
            "analytic_mean": k / D_MODEL,
            "analytic_beta_p2.5": float(b_lo),
            "analytic_beta_p97.5": float(b_hi),
            "sampled_p2.5": lo,
            "sampled_p50": mid,
            "sampled_p97.5": hi,
        }

    self_topk: dict[str, dict] = {}
    part = -np.sort(-sq, axis=1)
    for k in SELF_KS:
        sampled = part[:, :k].sum(1)
        lo, mid, hi = (float(x) for x in np.percentile(sampled, PCT))
        self_topk[str(k)] = {"sampled_p2.5": lo, "sampled_p50": mid, "sampled_p97.5": hi}

    abs_cos = np.abs(u[:, 0])  # a fixed direction is e_0 up to rotation
    lo, mid, hi = (float(x) for x in np.percentile(abs_cos, PCT))
    from math import lgamma, log, pi

    analytic_mean_cos = float(
        np.exp(lgamma(D_MODEL / 2.0) - 0.5 * log(pi) - lgamma((D_MODEL + 1) / 2.0))
    )
    return {
        "n_draws": N_NULL_DRAWS,
        "seed": NULL_SEED,
        "dim": D_MODEL,
        "fixed_subspace": fixed,
        "self_topk": self_topk,
        "abs_cos_fixed_direction": {
            "analytic_mean": analytic_mean_cos,
            "sampled_p2.5": lo,
            "sampled_p50": mid,
            "sampled_p97.5": hi,
        },
        "note": (
            "fixed_subspace and self_topk are DIFFERENT null families: the first "
            "is the mass in a subspace chosen independently of the direction, the "
            "second the mass in the direction's own top-K coordinates."
        ),
    }


def classify(value: float, lo: float, hi: float) -> str:
    """Mechanical value-vs-null-band verdict (no behavioural interpretation)."""
    if value > hi:
        return "above_null_p97.5"
    if value < lo:
        return "below_null_p2.5"
    return "within_null_band"


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


def layer_reads(
    bank: dict[str, np.ndarray],
    moments: dict[tuple[str, int], dict[str, np.ndarray]],
    subspaces: dict[tuple[str, int], dict[str, np.ndarray]],
    nulls: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Reads 1-3, per (state field, capture layer, trait). Returns (json, arrays)."""
    out: dict = {}
    arrays: dict[str, np.ndarray] = {}
    for field, field_name in STATE_FIELDS.items():
        for ell in CAPTURE_LAYERS:
            m = moments[(field, ell)]
            mu, sd, var = m["mu"], m["sd"], m["var"]
            mu_norm = float(np.linalg.norm(mu))
            gamma = mu_norm / float(np.linalg.norm(sd))
            snr = np.where(var > 0, mu * mu / np.maximum(var, 1e-300), 0.0)
            rank_absmu = np.argsort(-np.abs(mu))
            rank_snr = np.argsort(-snr)
            arrays[f"mu_{field}_L{ell}"] = mu
            arrays[f"sd_{field}_L{ell}"] = sd
            arrays[f"rogue_idx_absmu_{field}_L{ell}"] = rank_absmu[: max(ROGUE_KS)].astype(np.int32)
            arrays[f"rogue_idx_snr_{field}_L{ell}"] = rank_snr[: max(ROGUE_KS)].astype(np.int32)

            sub = subspaces[(field, ell)]
            mu_hat = mu / mu_norm
            var_total = float(var.sum())
            raw_total = float(sub["diag_raw"].sum())
            # Population "fair" comparators: what share of the population's own
            # spread each nuisance index-set / direction accounts for. A random
            # direction places K/D of its mass on a K-subspace; a direction that
            # merely lives where the population varies places ~this share. Both
            # references are reported so a large null-ratio is not read as
            # preferential concentration when it only tracks the variance.
            var_share = {
                ranking: {str(k): float(var[order[:k]].sum() / var_total) for k in ROGUE_KS}
                for ranking, order in (("abs_mu", rank_absmu), ("mu2_over_var", rank_snr))
            }
            raw_share = {
                ranking: {
                    str(k): float(sub["diag_raw"][order[:k]].sum() / raw_total) for k in ROGUE_KS
                }
                for ranking, order in (("abs_mu", rank_absmu), ("mu2_over_var", rank_snr))
            }
            mu_var_share = float(sub["quad_mu_centred"]) / float(sub["trace_centred"])
            pca_var_share = {
                name: float(sub[f"eigvals_{name}"].sum() / sub[f"trace_{name}"])
                for name in ("uncentred", "centred")
            }
            key = f"{field}_L{ell}"
            entry: dict = {
                "state_type": field_name,
                "layer": ell,
                "n_rows": m["n"],
                "gamma_mu_over_sigma": gamma,
                "mu_norm": mu_norm,
                "sigma_norm": float(np.linalg.norm(sd)),
                "n_zero_variance_dims": int((var <= 0).sum()),
                "top_rogue_dims_by_abs_mu": rank_absmu[:10].tolist(),
                "top_rogue_dims_by_mu2_over_var": rank_snr[:10].tolist(),
                "rogue_dim_variance_share": var_share,
                "rogue_dim_raw_second_moment_share": raw_share,
                "mean_direction_variance_share": mu_var_share,
                "pca_top48_variance_share": pca_var_share,
                "traits": {},
            }
            for trait, r in bank.items():
                u = r[ell] / np.linalg.norm(r[ell])
                t: dict = {"rogue_mass": {}, "pca48_mass": {}}
                for ranking, order in (("abs_mu", rank_absmu), ("mu2_over_var", rank_snr)):
                    per_k = {}
                    for k in ROGUE_KS:
                        val = float((u[order[:k]] ** 2).sum())
                        nb = nulls["fixed_subspace"][str(k)]
                        per_k[str(k)] = {
                            "mass": val,
                            "null_ratio": val / nb["analytic_mean"],
                            "variance_share_ratio": val / var_share[ranking][str(k)],
                            "vs_null": classify(val, nb["sampled_p2.5"], nb["sampled_p97.5"]),
                        }
                    t["rogue_mass"][ranking] = per_k
                cos_mu = float(abs(u @ mu_hat))
                ncos = nulls["abs_cos_fixed_direction"]
                t["abs_cos_with_mean_direction"] = {
                    "value": cos_mu,
                    "mass_on_mean_direction": cos_mu**2,
                    "null_ratio": cos_mu / ncos["analytic_mean"],
                    "variance_share_ratio": cos_mu**2 / mu_var_share,
                    "vs_null": classify(cos_mu, ncos["sampled_p2.5"], ncos["sampled_p97.5"]),
                }
                nb48 = nulls["fixed_subspace"][str(PCA_K)]
                for name in ("uncentred", "centred"):
                    val = float((sub[f"basis_{name}"].T @ u) ** 2 @ np.ones(PCA_K))
                    t["pca48_mass"][name] = {
                        "mass": val,
                        "null_ratio": val / nb48["analytic_mean"],
                        "variance_share_ratio": val / pca_var_share[name],
                        "vs_null": classify(val, nb48["sampled_p2.5"], nb48["sampled_p97.5"]),
                    }
                entry["traits"][trait] = t
            out[key] = entry
            logger.info(
                "[%s L%d] gamma=%.3f  rogue10(snr) %s  |cos(r_B,mu)| %s  pca48(cen) %s",
                field,
                ell,
                gamma,
                "/".join(
                    f"{entry['traits'][t]['rogue_mass']['mu2_over_var']['10']['mass']:.2e}"
                    for t in TRAITS
                ),
                "/".join(
                    f"{entry['traits'][t]['abs_cos_with_mean_direction']['value']:.3f}"
                    for t in TRAITS
                ),
                "/".join(
                    f"{entry['traits'][t]['pca48_mass']['centred']['mass']:.3f}" for t in TRAITS
                ),
            )
    return out, arrays


def bank_reads(bank: dict[str, np.ndarray], nulls: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Read 4 (all 28 layers, bank-only) + the supplementary trait-cosine matrix."""
    out: dict = {"per_trait": {}, "pairwise_trait_abs_cosine": {}}
    arrays: dict[str, np.ndarray] = {}
    for trait, r in bank.items():
        norms = np.linalg.norm(r, axis=1)
        u = r / norms[:, None]
        sq_sorted = -np.sort(-(u * u), axis=1)
        per_k = {}
        for k in SELF_KS:
            mass = sq_sorted[:, :k].sum(1)
            nb = nulls["self_topk"][str(k)]
            flagged = [
                int(i) for i in np.nonzero(mass > nb["sampled_p97.5"])[0]
            ]  # layers above the self-top-K null band
            per_k[str(k)] = {
                "mass_per_layer": [float(x) for x in mass],
                "max_over_layers": float(mass.max()),
                "argmax_layer": int(mass.argmax()),
                "layers_above_null_p97.5": flagged,
            }
            arrays[f"self_topk_mass_{trait}_k{k}"] = mass
        out["per_trait"][trait] = {
            "norm_per_layer": [float(x) for x in norms],
            "self_topk_mass": per_k,
        }
        arrays[f"rb_norm_{trait}"] = norms
        logger.info(
            "[bank %s] top1 mass max %.4f (L%d), top10 max %.4f (L%d)",
            trait,
            per_k["1"]["max_over_layers"],
            per_k["1"]["argmax_layer"],
            per_k["10"]["max_over_layers"],
            per_k["10"]["argmax_layer"],
        )
    # Supplementary (documented prediction P7): the three traits are one family.
    for i, a in enumerate(TRAITS):
        for b in TRAITS[i + 1 :]:
            ua = bank[a] / np.linalg.norm(bank[a], axis=1, keepdims=True)
            ub = bank[b] / np.linalg.norm(bank[b], axis=1, keepdims=True)
            cos = np.abs((ua * ub).sum(1))
            out["pairwise_trait_abs_cosine"][f"{a}|{b}"] = {
                "per_layer": [float(x) for x in cos],
                "mean": float(cos.mean()),
            }
            arrays[f"pair_abscos_{a}_{b}"] = cos
    return out, arrays


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def make_figure(layers: dict, bank: dict, nulls: dict, fig_dir: Path, stem: str) -> dict:
    """One 4x3 panel: reads 1-3 against their nulls at the capture layers, plus read 4.

    Colour encodes the TRAIT in every panel (the one factor common to all
    twelve). The two-level factors -- state type in rows 1-2, centring in
    row 3 -- are encoded by marker shape, and the population variance-share
    references by neutral line style, so one colour never carries two
    meanings across the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("blog")
    trait_color = {
        "evil": pp.paper_palette_role("primary"),
        "sycophancy": pp.paper_palette_role("accent"),
        "hallucination": pp.paper_palette_role("control"),
    }
    c_null = pp.paper_palette_role("neutral")
    xs = np.arange(len(TRAITS))
    labels = [t.capitalize() for t in TRAITS]

    fig, axes = plt.subplots(4, 3, figsize=(14.0, 16.0), layout="constrained")

    def vs_null_panel(ax, band, series, refs, title):
        """Trait-on-x panel: null band, one scatter per series, reference lines."""
        lo, hi, mean = band
        seen = [lo, hi, mean]
        ax.axhspan(lo, hi, color=c_null, alpha=0.20, lw=0, zorder=0)
        ax.axhline(mean, color=c_null, ls="--", lw=1.2, zorder=1)
        # Dodge the series apart on x: the two can land on identical values
        # (the raw and centred principal-subspace masses agree to ~1e-3), and a
        # later filled marker would otherwise hide an earlier one entirely.
        dodge = 0.16
        for s, (marker, filled, values) in enumerate(series):
            off = (s - (len(series) - 1) / 2.0) * dodge
            for i, t in enumerate(TRAITS):
                ax.scatter(
                    xs[i] + off,
                    values[t],
                    s=80,
                    marker=marker,
                    facecolors=trait_color[t] if filled else "none",
                    edgecolors=trait_color[t],
                    linewidths=1.7,
                    zorder=3,
                )
            seen.extend(values.values())
        for style, value in refs:
            ax.axhline(value, color=c_null, ls=style, lw=1.4, zorder=2)
            seen.append(value)
        ax.set_yscale("log")
        ax.set_xticks(xs, labels)
        ax.set_xlim(-0.5, len(TRAITS) - 0.5)
        pos = [v for v in seen if v > 0]
        ax.set_ylim(min(pos) / 3.0, max(pos) * 16.0)  # headroom so no legend overlaps data
        ax.set_title(title, loc="left")

    def marker_handle(marker, filled, label):
        return Line2D(
            [],
            [],
            ls="none",
            marker=marker,
            markersize=8,
            markerfacecolor=c_null if filled else "none",
            markeredgecolor=c_null,
            label=label,
        )

    null_handles = [
        Patch(facecolor=c_null, alpha=0.20, label="random-direction null (2.5-97.5%)"),
        Line2D([], [], color=c_null, ls="--", label="random-direction null mean"),
    ]

    # Row 1 -- mass on the top-10 massive-activation dims (ranked by mu^2/sigma^2).
    nb10 = nulls["fixed_subspace"]["10"]
    for j, ell in enumerate(CAPTURE_LAYERS):
        ctx, ans = layers[f"cx_last_L{ell}"], layers[f"v_x_L{ell}"]
        vs_null_panel(
            axes[0][j],
            (nb10["sampled_p2.5"], nb10["sampled_p97.5"], nb10["analytic_mean"]),
            [
                (
                    "o",
                    True,
                    {
                        t: ctx["traits"][t]["rogue_mass"]["mu2_over_var"]["10"]["mass"]
                        for t in TRAITS
                    },
                ),
                (
                    "s",
                    True,
                    {
                        t: ans["traits"][t]["rogue_mass"]["mu2_over_var"]["10"]["mass"]
                        for t in TRAITS
                    },
                ),
            ],
            [
                (":", ctx["rogue_dim_variance_share"]["mu2_over_var"]["10"]),
                ("-.", ans["rogue_dim_variance_share"]["mu2_over_var"]["10"]),
            ],
            f"Layer {ell}",
        )
    axes[0][0].set_ylabel(
        "share of the direction's squared mass\non the top-10 massive-activation dims"
    )
    axes[0][0].legend(
        handles=[
            *null_handles,
            Line2D([], [], color=c_null, ls=":", label="population variance share, context-end"),
            Line2D([], [], color=c_null, ls="-.", label="population variance share, mean-answer"),
            marker_handle("o", True, "context-end state"),
            marker_handle("s", True, "mean-answer state"),
        ],
        fontsize=6,
        loc="upper left",
        ncol=2,
    )

    # Row 2 -- alignment with the population mean activation direction.
    ncos = nulls["abs_cos_fixed_direction"]
    for j, ell in enumerate(CAPTURE_LAYERS):
        ctx, ans = layers[f"cx_last_L{ell}"], layers[f"v_x_L{ell}"]
        vs_null_panel(
            axes[1][j],
            (ncos["sampled_p2.5"], ncos["sampled_p97.5"], ncos["analytic_mean"]),
            [
                (
                    "o",
                    True,
                    {t: ctx["traits"][t]["abs_cos_with_mean_direction"]["value"] for t in TRAITS},
                ),
                (
                    "s",
                    True,
                    {t: ans["traits"][t]["abs_cos_with_mean_direction"]["value"] for t in TRAITS},
                ),
            ],
            [
                (":", float(np.sqrt(ctx["mean_direction_variance_share"]))),
                ("-.", float(np.sqrt(ans["mean_direction_variance_share"]))),
            ],
            f"Layer {ell}",
        )
    axes[1][0].set_ylabel("|cosine| with the population\nmean activation direction")
    axes[1][0].legend(
        handles=[
            *null_handles,
            Line2D([], [], color=c_null, ls=":", label="variance share along it, context-end"),
            Line2D([], [], color=c_null, ls="-.", label="variance share along it, mean-answer"),
            marker_handle("o", True, "context-end state"),
            marker_handle("s", True, "mean-answer state"),
        ],
        fontsize=6,
        loc="upper left",
        ncol=2,
    )

    # Row 3 -- top-48 principal-subspace mass on the mean-answer state, raw vs centred.
    nb48 = nulls["fixed_subspace"][str(PCA_K)]
    for j, ell in enumerate(CAPTURE_LAYERS):
        e = layers[f"v_x_L{ell}"]
        vs_null_panel(
            axes[2][j],
            (nb48["sampled_p2.5"], nb48["sampled_p97.5"], nb48["analytic_mean"]),
            [
                (
                    "o",
                    False,
                    {t: e["traits"][t]["pca48_mass"]["uncentred"]["mass"] for t in TRAITS},
                ),
                ("s", True, {t: e["traits"][t]["pca48_mass"]["centred"]["mass"] for t in TRAITS}),
            ],
            [
                (":", e["pca_top48_variance_share"]["uncentred"]),
                ("-.", e["pca_top48_variance_share"]["centred"]),
            ],
            f"Layer {ell}",
        )
    axes[2][0].set_ylabel(
        "share of squared mass in the\ntop-48 principal subspace (mean-answer state)"
    )
    axes[2][0].legend(
        handles=[
            *null_handles,
            Line2D([], [], color=c_null, ls=":", label="population variance share, uncentred"),
            Line2D([], [], color=c_null, ls="-.", label="population variance share, mean-centred"),
            marker_handle("o", False, "raw second moment (uncentred)"),
            marker_handle("s", True, "mean-centred (true PCA)"),
        ],
        fontsize=6,
        loc="upper left",
        ncol=2,
    )

    # Row 4 -- bank-only profile across all 28 layers.
    ell_all = np.arange(N_LAYERS)
    ax = axes[3][0]
    for t in TRAITS:
        ax.plot(
            ell_all,
            bank["per_trait"][t]["norm_per_layer"],
            color=trait_color[t],
            label=t.capitalize(),
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("norm of the persona direction")
    ax.set_title("Direction norm across depth", loc="left")
    ax.legend(fontsize=7, title="trait", title_fontsize=7)

    for col, k in zip((1, 2), SELF_KS, strict=True):
        ax = axes[3][col]
        nb = nulls["self_topk"][str(k)]
        ax.axhspan(
            nb["sampled_p2.5"], nb["sampled_p97.5"], color=c_null, alpha=0.20, lw=0, zorder=0
        )
        ax.axhline(nb["sampled_p50"], color=c_null, ls="--", lw=1.2, zorder=1)
        for t in TRAITS:
            ax.plot(
                ell_all,
                bank["per_trait"][t]["self_topk_mass"][str(k)]["mass_per_layer"],
                color=trait_color[t],
                label=t.capitalize(),
            )
        ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_ylabel(f"share of squared mass in the\ndirection's own top-{k} coordinate(s)")
        ax.set_title(f"Self-concentration, top-{k}", loc="left")
        if col == 1:
            ax.legend(
                handles=[
                    Patch(facecolor=c_null, alpha=0.20, label="random-direction null (2.5-97.5%)"),
                    Line2D([], [], color=c_null, ls="--", label="random-direction null median"),
                ],
                fontsize=6,
                loc="lower left",
            )

    fig.suptitle(
        "Persona directions vs residual-stream nuisance axes (Qwen-2.5-7B, issue 779 r_B bank)",
        fontsize=13,
    )
    paths = pp.savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=True,
    ).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="r_B vs residual-stream nuisance axes (#779)")
    ap.add_argument(
        "--chunks", type=int, default=10, help="capture chunks to stage (500 rows each)"
    )
    ap.add_argument("--scratch", type=Path, default=None, help="capture staging dir (data disk)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "rb_nuisance_profile",
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "issue_779" / "rb_nuisance_profile",
    )
    args = ap.parse_args()

    t0 = time.time()
    scratch = args.scratch or default_scratch()
    rb_root = PROJECT_ROOT / "data" / "issue_779" / "rb_nuisance_dl"

    logger.info("[stage] r_B bank -> %s (rev %s)", rb_root, RB_REVISION)
    bank_raw = load_rb_bank(rb_root)

    names = chunk_names(args.chunks)
    logger.info("[stage] %d capture chunks -> %s (rev %s)", len(names), scratch, CAPTURE_REVISION)
    paths = [stage_pinned_file(f"{CAPTURE_PREFIX}/{n}", CAPTURE_REVISION, scratch) for n in names]

    moments = moment_pass(paths)
    subspaces = {
        (f, ell): subspace_pass(paths, f, ell, moments[(f, ell)]["mu"])
        for f in STATE_FIELDS
        for ell in CAPTURE_LAYERS
    }
    nulls = build_nulls()
    layers_json, layer_arrays = layer_reads(bank_raw, moments, subspaces, nulls)
    bank_json, bank_arrays = bank_reads(bank_raw, nulls)

    out = {
        "reads": {
            "1_gamma": "||mu|| / ||sigma|| per state type per capture layer",
            "2_rogue_mass": "r_B squared-mass share on top-K massive-activation dims vs null",
            "3_scaffold": "|cos(r_B, mu)| and top-48 principal-subspace mass, raw vs centred",
            "4_bank_profile": "r_B norm + own-top-K coordinate concentration, all 28 layers",
        },
        "two_reference_points": (
            "Every mass read in `layer_reads` carries TWO comparators, and both are "
            "needed to read it. `null_ratio` is against an isotropic random direction "
            "(K/D for a K-subspace) and answers 'is this more than chance?'. "
            "`variance_share_ratio` is against the share of the POPULATION's own "
            "spread that the same dims / subspace account for, and answers 'is this "
            "more than merely living where the activations vary?'. A large null_ratio "
            "with a variance_share_ratio at or below 1 means the overlap tracks the "
            "population geometry rather than a preferential concentration of r_B."
        ),
        "scope_limits": [
            "Reads 1-3 exist only at the three layers the n1m capture stores (14, 19, 26); "
            "read 4 covers all 28 bank layers.",
            f"Activation statistics come from {len(paths)} capture chunks "
            f"({moments[('v_x', CAPTURE_LAYERS[0])]['n']} rows) stepped evenly across the store's "
            "32 shards x 60 chunks, not a uniform random sample of the ~960k-row corpus.",
            "The centred counterpart of |cos(r_B, mu)| is degenerate by construction "
            "(mean-centring sends mu to 0); the raw-vs-centred contrast is carried by the "
            "top-48 principal-subspace pair instead.",
            "r_B norms are comparable across layers only within one trait bank (same extraction "
            "recipe); every other read here is scale-invariant.",
            "Instrument characterization only: these are geometry numbers against random-direction "
            "nulls, and support no claim about behaviour or steering.",
        ],
        "layer_reads": layers_json,
        "bank_reads": bank_json,
        "nulls": nulls,
        "inputs": {
            "rb_bank": {
                "repo": DATA_REPO,
                "path_in_repo": RB_PREFIX,
                "revision": RB_REVISION,
                "traits": list(TRAITS),
                "shape": [N_LAYERS, D_MODEL],
            },
            "capture": {
                "repo": DATA_REPO,
                "path_in_repo": CAPTURE_PREFIX,
                "revision": CAPTURE_REVISION,
                "chunks": names,
                "fields": {k: v for k, v in STATE_FIELDS.items()},
                "layers": list(CAPTURE_LAYERS),
            },
        },
        "metadata": {
            "script": "issue779_rb_nuisance_profile",
            "git_commit": git_commit(),
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "wall_seconds": None,
            "provenance": (
                "user-chat inline free-analysis 2026-07-28 (rb-nuisance-profile); 0 GPU-h, CPU "
                "only; read-only over the pinned #779 r_B bank + a bounded n1m capture subsample; "
                "float64 accumulation, centred scatter accumulated directly (not by subtracting "
                "n*mu*mu^T) so the rogue-dimension regime cannot cancel catastrophically"
            ),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "per_dim_arrays.npz", **layer_arrays, **bank_arrays)
    fig_paths = make_figure(layers_json, bank_json, nulls, args.fig_dir, "rb_nuisance_profile")
    out["figure"] = fig_paths
    out["metadata"]["wall_seconds"] = round(time.time() - t0, 1)
    (args.out_dir / "profile.json").write_text(json.dumps(out, indent=1))
    logger.info(
        "wrote %s + per_dim_arrays.npz + %s (%.1fs total)",
        args.out_dir / "profile.json",
        fig_paths.get("png"),
        time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
