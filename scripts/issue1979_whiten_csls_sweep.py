"""#1979 inline round `whiten-csls-sweep` — re-metricize the predictor race.

Re-runs the per-prefix leakage race under four metric settings over the
anchor-cosine predictor family, on BANKED artifacts only (zero GPU, no new
generation, no new judge calls):

  raw        panel-centered cosine (the committed convention; reproduction check)
  whiten     the same cosine in the Sigma^-1/2 basis (banked corpus Cholesky)
  csls       raw minus gamma * r_pool(x)  (cross-domain local scaling)
  both       whitened cosine minus gamma * r_pool_whitened(x)

SCOPE. Only the anchor-cosine family is re-metricized: p1/p2/p3a/p3b (centroid
cosines) and p9/p10 (top-k row reads). p4 (already a whitened gate with its own
normalization), p5/p6 (projections onto a read-out direction), p8a (a norm) and
p7 (a judge score, not geometric) are carried UNCHANGED as reference columns.

CSLS POOL. With a single anchor centroid CSLS is exactly rank-inert: r_S is
constant across prefixes and r_T degenerates to the score itself, so
2S - r_T - r_S reduces to S minus a constant. A candidate-target pool is
therefore required, and the choice is a design decision, not a detail. For the
centroid reads the pool is the corresponding anchor of EVERY training mix in the
fleet; for p9/p10 it is the union of every mix's individual training rows. Both
give a specificity correction: penalize a prefix whose vector sits close to
every behavior's training anchors rather than specifically to this one's.

Note on reuse: ``issue2202_metric_zoo.csls_ranks`` is a RETRIEVAL rank scorer
(it needs a per-query true target index and returns mid-ranks), so its body does
not fit a within-arm Spearman race against a continuous DV. Its CONVENTIONS are
reused verbatim -- K_LOCAL = 10 and the gamma = 0.5 exact-CSLS parameterisation
-- and are imported from that module rather than re-declared here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from issue2202_metric_zoo import K_LOCAL  # noqa: E402  (standing CSLS neighborhood size)

REPO_ROOT = SCRIPTS_DIR.parent
STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue1979_whitencsls")
RACE = REPO_ROOT / "eval_results/issue_1979/race"
ARMS_JSON = REPO_ROOT / "eval_results/issue_1979/config/arms.json"
OUT = REPO_ROOT / "eval_results/issue_1979/whiten_csls"

GAMMA = 0.5  # exact CSLS rank-wise (issue2202_metric_zoo.csls_ranks docstring)
PRIMARY = {"content": (19, "last_prompt"), "marker": (25, "last_prompt")}
MPOS = {"span_mean_context": "span_mean", "last_prompt": "last_prompt"}
ANCHOR_KEY_BY_POS = {
    "span_mean_context": "A_ctx_span",
    "last_prompt": "A_ctx_last_prompt",
    "last_ctx": "A_ctx_last_ctx",
}
DV_BY_KIND = {"content": "dv_change", "marker": "dv_dlogp"}
SETTINGS = ("raw", "whiten", "csls", "both")
REMETRIC = ("p1", "p2", "p3a", "p3b", "p9", "p10")
CARRIED = ("p4", "p5", "p6", "p7", "p8a", "p8b")


def _np(t) -> np.ndarray:
    """Torch tensor (any dtype) -> float64 numpy. Fails loud on None."""
    assert t is not None, "missing tensor"
    return np.asarray(t.double().numpy() if hasattr(t, "double") else t, dtype=np.float64)


def _center(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = a.mean(axis=0)
    return a - mu, mu


def _cos_rows(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise cosine of (n,d) against a single (d,) vector."""
    xn = np.linalg.norm(x, axis=1) + 1e-12
    return (x @ v) / (xn * (np.linalg.norm(v) + 1e-12))


def _whiten(chol: np.ndarray, m: np.ndarray) -> np.ndarray:
    """L^-1 applied to every ROW of (n,d) (or a single (d,) vector).

    Sigma = L L^T, so L^-1 x is the whitened coordinate. Batched: ONE triangular
    solve over the whole (d,n) block, never a per-row loop.
    """
    one_d = m.ndim == 1
    x = m.reshape(1, -1) if one_d else m
    y = solve_triangular(chol, x.T, lower=True).T
    return y[0] if one_d else y


def _topk_mean(a: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    """Mean of the top-k cosines from each row of (n,d) to a (m,d) pool."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    rn = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)
    cos = an @ rn.T
    kk = min(k, cos.shape[1])
    return np.sort(cos, axis=1)[:, -kk:].mean(axis=1)


def _csls_penalty(x: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
    """r_T(x): mean cosine of each row of x to its k nearest pool members."""
    return _topk_mean(x, pool, k)


def load_inputs() -> dict:
    arms = json.loads(ARMS_JSON.read_text())["arms"]
    tens = torch.load(
        STAGE / "battery/ingredient_tensors.pt", map_location="cpu", weights_only=False
    )
    sig = torch.load(STAGE / "battery/sigma_chol.pt", map_location="cpu", weights_only=False)
    mixes = sorted({a["mix_arm_id"] for a in arms})
    anch = {
        m: torch.load(STAGE / f"anchors/{m}/anchors.pt", map_location="cpu", weights_only=False)
        for m in mixes
    }
    return {"arms": arms, "tensors": tens, "sigma": sig, "anchors": anch, "mixes": mixes}


def arm_columns(d: dict, arm: dict) -> dict:
    """All four metric settings x the re-metricized predictors, for one arm."""
    aid, kind, mix = arm["arm_id"], arm["kind"], arm["mix_arm_id"]
    layer, pos = PRIMARY[kind]
    tens, anch = d["tensors"], d["anchors"]
    chol = _np(d["sigma"][f"L{layer}"]["chol"])

    slot = f"{aid}/L{layer}/{pos}"
    span = f"{aid}/L{layer}/span_mean_context"
    Cbar, Vbar0 = _np(tens[f"{slot}/Cbar"]), _np(tens[f"{slot}/Vbar0"])
    C_span, V_span = _np(tens[f"{span}/Cbar"]), _np(tens[f"{span}/Vbar0"])
    mpos = MPOS[pos]
    M0C = _np(tens[f"m0pred/{kind}/L{layer}/{mpos}"])

    a_own = anch[mix][f"L{layer}"]
    A_ctx = _np(a_own[ANCHOR_KEY_BY_POS[pos]])
    A_ans = _np(a_own["A_ans"])
    M0A = _np(tens[f"m0anchor/{mix}/L{layer}/{mpos}"])

    # fleet-wide pools (all mixes) for the CSLS penalty term
    pool_ctx = np.stack([_np(anch[m][f"L{layer}"][ANCHOR_KEY_BY_POS[pos]]) for m in d["mixes"]])
    pool_ans = np.stack([_np(anch[m][f"L{layer}"]["A_ans"]) for m in d["mixes"]])
    pool_m0a = np.stack([_np(tens[f"m0anchor/{m}/L{layer}/{mpos}"]) for m in d["mixes"]])
    rows_ctx = _np(a_own["rows_ctx"])
    rows_ans = _np(a_own["rows_ans"])
    poolrows_ctx = np.concatenate([_np(anch[m][f"L{layer}"]["rows_ctx"]) for m in d["mixes"]])
    poolrows_ans = np.concatenate([_np(anch[m][f"L{layer}"]["rows_ans"]) for m in d["mixes"]])

    Cc, c_mu = _center(Cbar)
    Vc, v_mu = _center(Vbar0)
    Mc, m_mu = _center(M0C)

    # (x-block, target, csls-pool) per centroid predictor, in the target's own centering
    cent = {
        "p1": (Cc, A_ctx - c_mu, pool_ctx - c_mu),
        "p2": (Vc, A_ans - v_mu, pool_ans - v_mu),
        "p3a": (Mc, M0A - m_mu, pool_m0a - m_mu),
        "p3b": (Mc, A_ans - m_mu, pool_ans - m_mu),
    }
    knn = {"p9": (C_span, rows_ctx, poolrows_ctx), "p10": (V_span, rows_ans, poolrows_ans)}

    out: dict[str, dict[str, np.ndarray]] = {s: {} for s in SETTINGS}
    for name, (x, tgt, pool) in cent.items():
        xw, tw, pw = _whiten(chol, x), _whiten(chol, tgt), _whiten(chol, pool)
        k = min(K_LOCAL, pool.shape[0])
        out["raw"][name] = _cos_rows(x, tgt)
        out["whiten"][name] = _cos_rows(xw, tw)
        out["csls"][name] = out["raw"][name] - GAMMA * _csls_penalty(x, pool, k)
        out["both"][name] = out["whiten"][name] - GAMMA * _csls_penalty(xw, pw, k)
    for name, (x, rows, pool) in knn.items():
        xw, rw, pw = _whiten(chol, x), _whiten(chol, rows), _whiten(chol, pool)
        k = min(K_LOCAL, pool.shape[0])
        out["raw"][name] = _topk_mean(x, rows, 8)
        out["whiten"][name] = _topk_mean(xw, rw, 8)
        out["csls"][name] = out["raw"][name] - GAMMA * _csls_penalty(x, pool, k)
        out["both"][name] = out["whiten"][name] - GAMMA * _csls_penalty(xw, pw, k)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args(argv)

    d = load_inputs()
    banked_cols = {"p9": "p9_k8", "p10": "p10_k8"}
    recs: list[dict] = []
    repro: list[dict] = []
    for arm in d["arms"]:
        aid, kind = arm["arm_id"], arm["kind"]
        frame = json.loads((RACE / f"frame_{aid}.json").read_text())["frame"]
        dv = np.asarray(frame[DV_BY_KIND[kind]], dtype=np.float64)
        cols = arm_columns(d, arm)
        for setting in SETTINGS:
            for name in REMETRIC:
                v = cols[setting][name]
                m = np.isfinite(v) & np.isfinite(dv)
                assert m.sum() >= 45, (aid, setting, name, int(m.sum()))
                recs.append(
                    {
                        "arm_id": aid,
                        "kind": kind,
                        "setting": setting,
                        "predictor": name,
                        "rho": float(stats.spearmanr(v[m], dv[m]).statistic),
                        "n": int(m.sum()),
                    }
                )
        # carried (metric-invariant) reference columns, banked values re-read
        banked = json.loads((RACE / f"arm_{aid}.json").read_text())["observed_rho"][
            DV_BY_KIND[kind]
        ]
        for name in CARRIED:
            if banked.get(name) is not None:
                recs.append(
                    {
                        "arm_id": aid,
                        "kind": kind,
                        "setting": "carried",
                        "predictor": name,
                        "rho": float(banked[name]),
                        "n": 50,
                    }
                )
        # reproduction check: our raw recompute vs the banked race value
        for name in REMETRIC:
            col = banked_cols.get(name, f"{name}_tc" if f"{name}_tc" in frame else name)
            ours = [
                r
                for r in recs
                if r["arm_id"] == aid and r["setting"] == "raw" and r["predictor"] == name
            ][0]["rho"]
            if banked.get(name) is not None:
                repro.append(
                    {
                        "arm_id": aid,
                        "predictor": name,
                        "banked": float(banked[name]),
                        "recomputed": ours,
                        "abs_delta": abs(float(banked[name]) - ours),
                        "frame_col": col,
                    }
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    worst = max(r["abs_delta"] for r in repro)
    payload = {
        "records": recs,
        "reproduction_check": {"rows": repro, "worst_abs_delta": worst},
        "config": {
            "settings": list(SETTINGS),
            "remetricized": list(REMETRIC),
            "carried_unchanged": list(CARRIED),
            "gamma": GAMMA,
            "k_local": K_LOCAL,
            "n_mixes_in_pool": len(d["mixes"]),
            "primary": {k: list(v) for k, v in PRIMARY.items()},
            "dv_by_kind": DV_BY_KIND,
            "sigma": "banked corpus Cholesky (shrunk), issue1979 battery/sigma_chol.pt",
        },
    }
    (args.out_dir / "sweep.json").write_text(json.dumps(payload, indent=2))
    print(f"[sweep] {len(recs)} records -> {args.out_dir / 'sweep.json'}")
    print(f"[sweep] raw-vs-banked reproduction worst |delta| = {worst:.2e} over {len(repro)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
