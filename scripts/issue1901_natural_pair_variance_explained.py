"""Variance explained on natural held-out query pairs, exact and closed form.

Pairwise sums of squared distances reduce to Gram traces, so the read is exact
over all C(n,2) pairs with no enumeration:
  sum_{i<j} ||r_i - r_j||^2 = n * sum_i ||r_i||^2 - ||sum_i r_i||^2.
Same VE definition as eval_results/issue_2564/comment2_variance_explained:
  VE = 1 - sum||dh_A - pred||^2 / sum||dh_A||^2   (0 = the no-shift predictor).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Thread caps must land BEFORE numpy/torch import: load_dotenv() setdefaults
# OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS and the BLAS pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402

S = Path("/mnt/eps-data/thomasjiralerspong")
cz = np.load(S / "issue2202_avgtgt/cx_holdout_L19.npz")
pz = np.load(S / "issue2202_freshwhiten/pred16.npz")
az = np.load(S / "issue1738_phaseb/y_holdout_avg5_L19.npz")
order = {int(c): i for i, c in enumerate(np.asarray(cz["ci"], np.int64))}
keep = np.array([order[int(c)] for c in np.asarray(az["ci"], np.int64)], np.int64)
C = np.asarray(cz["cx"], np.float64)[keep]
P = np.asarray(pz["pred16"], np.float64)[keep]
Y = np.asarray(az["y16"], np.float64)
n = len(Y)
print(f"n contexts {n}, pairs {n * (n - 1) // 2:,}, dim {Y.shape[1]}")


def pss(A, B=None):
    """sum_{i<j} <A_i - A_j, B_i - B_j>."""
    B = A if B is None else B
    return n * float(np.einsum("ij,ij->", A, B)) - float(A.sum(0) @ B.sum(0))


SSy = pss(Y)
for name, pred in (("map", P), ("copy", C)):
    r = Y - pred
    ve = 1 - pss(r) / SSy
    spy, spp = pss(pred, Y), pss(pred)
    alpha = spy / spp
    ver = 1 - (SSy - 2 * alpha * spy + alpha**2 * spp) / SSy
    ratio = spy / SSy  # through-origin predicted-over-observed size slope
    print(
        f"{name:5s} VE={ve:+.3f}  VE_rescaled={ver:+.3f}  alpha={alpha:.3f}  slope={np.sqrt(spp / SSy):.3f}"
    )
