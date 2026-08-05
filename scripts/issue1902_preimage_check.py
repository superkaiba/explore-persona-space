"""Preimage-feasibility check for the #1902 ladder ans-offset finding (B->S).

Question (user, 2026-08-04): couldn't the constant answer-space correction be
implemented on the CONTEXT side, transferred through the mapping W?
Checks, full-sample (geometry diagnostic, no folds):
  1. cos(dy, c*)        dy = answer-cloud mean shift; c* = optimal constant corr.
  2. cos(-W dx, c*)     dx = context-cloud mean shift transported through W
                        (what ctx_offset actually applies).
  3. Minimal-norm context-side preimage of c*: dx* = argmin ||dx|| s.t. W dx ~ c*
     via SVD with singular-value cutoffs; report ||dx*|| vs context-cloud radius
     and the fraction of ||c*||^2 reachable per cutoff band.
"""

import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(_SCRIPTS_DIR), str(_SCRIPTS_DIR.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# load_dotenv() BEFORE any heavy import so the shared-VM thread caps (#847) bind
# in-process (tests/test_shared_vm_thread_caps.py, the #1146 predicate).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

from issue1902_ladder_followup import ARM, CORPUS, LAYER_STAR, LadderContext  # noqa: E402

OUT_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue1902_ladder")
LJ = json.load(open("eval_results/issue_1902/followup_ladder/ladder_modes.json"))

ctx = LadderContext(OUT_ROOT, ["B", "S", "D", "R"], layer=LAYER_STAR)
u_B, w_BB = ctx.xy("B", "B", CORPUS, LAYER_STAR, ARM)
u_S, w_SS = ctx.xy("S", "S", CORPUS, LAYER_STAR, ARM)
u_B, w_BB, u_S, w_SS = (np.asarray(a, dtype=np.float64) for a in (u_B, w_BB, u_S, w_SS))
n, d = u_B.shape
lam = float(np.median(LJ["pairs"]["B->S"]["lambda_f_ii_per_fold"]))
print(f"n={n} d={d} lambda={lam:.4g}")

# Standardized ridge fit of f_BB on ALL rows (diagnostic; matches the ladder's
# standardized-primal recipe in shape, full-sample instead of per-fold).
xmu, xsd = u_B.mean(0), u_B.std(0) + 1e-9
Xn = (u_B - xmu) / xsd
ymu = w_BB.mean(0)
G = Xn.T @ Xn + lam * np.eye(d)
W = np.linalg.solve(G, Xn.T @ (w_BB - ymu))  # (d, d): standardized ctx -> centered ans


def predict(u):
    return ((u - xmu) / xsd) @ W + ymu


# Constant corrections
dx = u_S.mean(0) - u_B.mean(0)  # context-cloud mean shift
dy = w_SS.mean(0) - w_BB.mean(0)  # answer-cloud mean shift
c_star = (w_SS - predict(u_S)).mean(0)  # optimal constant correction (bias_refit)
t = (dx / xsd) @ W  # dx transported through the map (linear part)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


rad_B = float(np.median(np.linalg.norm(u_B - u_B.mean(0), axis=1)))
rad_S = float(np.median(np.linalg.norm(u_S - u_S.mean(0), axis=1)))
print(
    f"norms: |dx|={np.linalg.norm(dx):.3f} |dy|={np.linalg.norm(dy):.3f} "
    f"|c*|={np.linalg.norm(c_star):.3f} ctx cloud radius B={rad_B:.3f} S={rad_S:.3f}"
)
print(f"cos(dy, c*)        = {cos(dy, c_star):+.3f}")
print(f"cos(-W dx, c*)     = {cos(-t, c_star):+.3f}   (what ctx_offset applies)")
print(f"cos(W dx, dy)      = {cos(t, dy):+.3f}   (is answer shift the image of ctx shift?)")
print(f"cos(dx, dy)        = {cos(dx, dy):+.3f}")

# Preimage of c* through W (standardized coords, then unstandardize).
# Orientation: predictions = Xn @ W (row-vector convention), so answer-space components
# live in W's COLUMNS' span: y^T = W^T x^T. Use M = W.T (ans <- ctx, column-vector map).
M = W.T
U, s, Vt = np.linalg.svd(M, full_matrices=False)  # M = U s Vt, ans = M @ ctx_std
comp = U.T @ c_star  # components of c* in answer-side singular basis
e_total = float(c_star @ c_star)
print(f"sigma: max={s[0]:.4f} median={np.median(s):.4f} min={s[-1]:.2e}")
for cut in (10, 100, 1000, np.inf):
    mask = s >= s[0] / cut
    frac = float((comp[mask] ** 2).sum() / e_total)
    pre_std = Vt.T[:, mask] @ (comp[mask] / s[mask])
    band_target = U[:, mask] @ comp[mask]  # the band-projected component of c*
    recon_err = np.linalg.norm(M @ pre_std - band_target)
    assert recon_err < 1e-6 * max(1.0, np.linalg.norm(band_target)), (
        f"preimage reconstruction self-check failed at cutoff {cut}: err={recon_err:.3e}"
    )
    pre = pre_std * xsd  # unstandardize back to raw context coords
    resid = 1.0 - frac
    print(
        f"cutoff sigma_max/{cut:>6}: reachable ||c*||^2 frac={frac:.3f} "
        f"residual={resid:.3f} ||dx*||={np.linalg.norm(pre):.3f} "
        f"(={np.linalg.norm(pre) / rad_S:.2f}x target ctx cloud radius)"
    )

# sanity: does bias_refit full-sample reproduce the ladder's ballpark? R2 of W u_S + c*
pred = predict(u_S) + c_star
ss_res = float(((w_SS - pred) ** 2).sum())
ss_tot = float(((w_SS - w_SS.mean(0)) ** 2).sum())
print(f"full-sample bias_refit R2 (in-sample, diagnostic only) = {1 - ss_res / ss_tot:.3f}")
