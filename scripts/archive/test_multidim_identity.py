"""
Persona Geometry — 1.5: Is persona identity 1D (centroid) or multi-dimensional (manifold shape)?

Three complementary tests, each ruling out different confounds:

Test 1: Per-persona whitening + kNN
  - Removes ALL second-order distributional differences (variance, covariance)
  - If kNN still works → signal is beyond covariance structure

Test 2: Question-paired direction test
  - For each question, compute unit-normalized residual per persona
  - Tests whether personas systematically deflect same stimulus in different directions
  - Magnitude-free, centroid-free

Test 3: Grassmann distance between per-persona subspaces
  - Do different personas occupy different principal subspaces?
  - Scale-invariant (angular measure)
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = Path("data/persona_L22_full_meta.npz")
NAMES_PATH = Path("data/persona_names.json")
OUT_DIR = Path("eval_results/aim1_5_multidim_identity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]
prompt = data["prompt"]
question = data["question"]
names = json.load(open(NAMES_PATH))

N_PERSONAS = len(names)
DIM = X.shape[1]
CHANCE = 1.0 / N_PERSONAS
RS = 42

# Subtract centroids
centroids = np.zeros((N_PERSONAS, DIM))
for c in range(N_PERSONAS):
    centroids[c] = X[y == c].mean(axis=0)
X_res = X - centroids[y]

results = {}


def knn_cv(Xd, yd, k=5, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RS)
    accs = []
    for tr, te in skf.split(Xd, yd):
        clf = KNeighborsClassifier(n_neighbors=k, algorithm="ball_tree")
        clf.fit(Xd[tr], yd[tr])
        accs.append(accuracy_score(yd[te], clf.predict(Xd[te])))
    return float(np.mean(accs)), float(np.std(accs))


# ════════════════════════════════════════════════════════════════════════════
# BASELINE: Raw kNN on residuals (the confounded test, for comparison)
# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("BASELINE: kNN on raw centroid-subtracted residuals")
print("=" * 70)
t0 = time.time()
acc_raw, std_raw = knn_cv(X_res, y)
print(f"  Accuracy: {acc_raw:.4f} +/- {std_raw:.4f} ({acc_raw / CHANCE:.0f}x chance)")
print(f"  ({time.time() - t0:.1f}s)")
results["baseline_raw_knn"] = {"acc": acc_raw, "std": std_raw}


# ════════════════════════════════════════════════════════════════════════════
# TEST 1: Per-persona whitening + kNN
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: kNN after per-persona whitening (removes all 2nd-order structure)")
print("=" * 70)

t0 = time.time()
X_white = np.zeros_like(X_res)
for c in range(N_PERSONAS):
    mask = y == c
    Xc = X_res[mask]
    C = np.cov(Xc.T)
    evals, evecs = np.linalg.eigh(C)
    # Regularize: clamp small eigenvalues
    evals = np.maximum(evals, 1e-4 * evals.max())
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    X_white[mask] = Xc @ W

acc_white, std_white = knn_cv(X_white, y)
print(f"  Whitened kNN:  {acc_white:.4f} +/- {std_white:.4f} ({acc_white / CHANCE:.0f}x chance)")
print(f"  Retention:     {acc_white / acc_raw:.1%} of raw signal")

# Null: shuffle labels, whiten with wrong persona's covariance
rng = np.random.RandomState(RS + 1)
y_shuf = rng.permutation(y)
X_null_white = np.zeros_like(X_res)
for c in range(N_PERSONAS):
    mask = y_shuf == c
    Xc = X_res[mask]
    C = np.cov(Xc.T)
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 1e-4 * evals.max())
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    X_null_white[mask] = Xc @ W

acc_null, std_null = knn_cv(X_null_white, y)
print(f"  Null (wrong-whitened): {acc_null:.4f} +/- {std_null:.4f}")
print(f"  Chance:        {CHANCE:.4f}")
print(f"  ({time.time() - t0:.1f}s)")

results["test1_whitened_knn"] = {
    "whitened": {"acc": acc_white, "std": std_white},
    "null": {"acc": acc_null, "std": std_null},
    "retention": acc_white / acc_raw,
}

# Also test: whiten then re-equalize variance (isotropic scaling per persona)
# This tests if signal is in direction alone (not residual higher moments)
X_white_normed = np.zeros_like(X_white)
for c in range(N_PERSONAS):
    mask = y == c
    Xc = X_white[mask]
    # Normalize each sample to unit norm (removes all magnitude info)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    X_white_normed[mask] = Xc / (norms + 1e-10)

acc_wn, std_wn = knn_cv(X_white_normed, y)
print(f"  Whitened+unit-normed: {acc_wn:.4f} +/- {std_wn:.4f} ({acc_wn / CHANCE:.0f}x chance)")
results["test1_whitened_unitnorm"] = {"acc": acc_wn, "std": std_wn}


# ════════════════════════════════════════════════════════════════════════════
# TEST 2: Question-paired direction test
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: Question-paired direction cosine (magnitude-free)")
print("=" * 70)

t0 = time.time()
questions_unique = np.unique(question)
n_questions = len(questions_unique)

# Compute mean residual per (persona, question), averaged over 5 prompt variants
R = np.zeros((N_PERSONAS, n_questions, DIM))
for c in range(N_PERSONAS):
    for qi, q in enumerate(questions_unique):
        mask = (y == c) & (question == q)
        R[c, qi] = X_res[mask].mean(axis=0)

# Normalize to unit length
norms = np.linalg.norm(R, axis=2, keepdims=True)
R_hat = R / (norms + 1e-10)

# Pairwise direction cosine matrix (vectorized)
# cos_ij = mean_q( R_hat[i,q] . R_hat[j,q] )
# R_hat shape: (49, 240, 100)
# Einsum: for each pair (i,j), dot product over dim 100, then mean over questions
pair_cos = np.einsum("iqd,jqd->ij", R_hat, R_hat) / n_questions

offdiag_idx = np.triu_indices(N_PERSONAS, k=1)
offdiag = pair_cos[offdiag_idx]
stat_obs = offdiag.var()
mean_cos = offdiag.mean()

print(f"  Mean pairwise direction cosine: {mean_cos:.6f}")
print(f"  Variance of pairwise cos:       {stat_obs:.8f}")
print(f"  Range: [{offdiag.min():.4f}, {offdiag.max():.4f}]")

# Top/bottom pairs
pair_flat = []
for i in range(N_PERSONAS):
    for j in range(i + 1, N_PERSONAS):
        pair_flat.append((pair_cos[i, j], names[i], names[j]))
pair_flat.sort(key=lambda x: x[0], reverse=True)

print("\n  Most similar directions (same stimulus → same deflection):")
for cos_val, n1, n2 in pair_flat[:5]:
    print(f"    {n1:>20s} ↔ {n2:<20s}  cos={cos_val:.4f}")
print("  Most opposite directions:")
for cos_val, n1, n2 in pair_flat[-5:]:
    print(f"    {n1:>20s} ↔ {n2:<20s}  cos={cos_val:.4f}")

# Permutation test
n_perms = 1000
null_vars = np.zeros(n_perms)
null_means = np.zeros(n_perms)
rng = np.random.RandomState(RS)
for p in range(n_perms):
    perm = rng.permutation(N_PERSONAS)
    R_perm = R_hat[perm]
    pc = np.einsum("iqd,jqd->ij", R_perm, R_hat) / n_questions
    od = pc[offdiag_idx]
    null_vars[p] = od.var()
    null_means[p] = od.mean()

p_var = (null_vars >= stat_obs).mean()
p_mean = (np.abs(null_means) >= np.abs(mean_cos)).mean()
effect_z = (stat_obs - null_vars.mean()) / null_vars.std()

print(f"\n  Permutation test (n={n_perms}):")
print(f"    Variance: obs={stat_obs:.8f}, null={null_vars.mean():.8f} +/- {null_vars.std():.8f}")
print(f"    p(variance) = {p_var:.4f}, z = {effect_z:.1f}")
print(f"    Mean |cos|: obs={mean_cos:.6f}, null={null_means.mean():.6f}")
print(f"    p(mean) = {p_mean:.4f}")

# Correlation with centroid similarity
centroid_centered = centroids - centroids.mean(axis=0)
centroid_norms = np.linalg.norm(centroid_centered, axis=1, keepdims=True)
centroid_normed = centroid_centered / (centroid_norms + 1e-10)
centroid_cos = centroid_normed @ centroid_normed.T
centroid_offdiag = centroid_cos[offdiag_idx]

corr = np.corrcoef(offdiag, centroid_offdiag)[0, 1]
print(f"\n  Correlation with centroid similarity: r = {corr:.4f}")
print("  (Positive → nearby personas deflect stimuli similarly)")
print(f"  ({time.time() - t0:.1f}s)")

results["test2_direction_cosine"] = {
    "mean_cos": float(mean_cos),
    "variance": float(stat_obs),
    "p_variance": float(p_var),
    "z_score": float(effect_z),
    "p_mean": float(p_mean),
    "corr_with_centroid": float(corr),
    "top_pairs": [(c, n1, n2) for c, n1, n2 in pair_flat[:10]],
    "bottom_pairs": [(c, n1, n2) for c, n1, n2 in pair_flat[-10:]],
}


# ════════════════════════════════════════════════════════════════════════════
# TEST 3: Grassmann distance between per-persona subspaces
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: Grassmann distance between per-persona subspaces")
print("=" * 70)

for sub_k in [5, 10, 20]:
    t0 = time.time()
    print(f"\n  --- Subspace dimension k={sub_k} ---")

    # Per-persona PCA
    persona_subs = {}
    for c in range(N_PERSONAS):
        Xc = X_res[y == c]
        pca_c = PCA(n_components=sub_k, random_state=RS)
        pca_c.fit(Xc)
        persona_subs[c] = pca_c.components_.T  # (100, k)

    # Pooled PCA
    pca_pool = PCA(n_components=sub_k, random_state=RS)
    pca_pool.fit(X_res)
    pooled_sub = pca_pool.components_.T

    # Distance from each persona to pooled
    dist_to_pooled = np.array(
        [
            np.sqrt(np.sum(subspace_angles(persona_subs[c], pooled_sub) ** 2))
            for c in range(N_PERSONAS)
        ]
    )

    # Pairwise Grassmann distances
    grassmann = np.zeros((N_PERSONAS, N_PERSONAS))
    for i in range(N_PERSONAS):
        for j in range(i + 1, N_PERSONAS):
            angles = subspace_angles(persona_subs[i], persona_subs[j])
            grassmann[i, j] = grassmann[j, i] = np.sqrt(np.sum(angles**2))

    gm_offdiag = grassmann[offdiag_idx]

    # Subspace overlap (mean cos^2 of principal angles)
    overlap = np.zeros((N_PERSONAS, N_PERSONAS))
    for i in range(N_PERSONAS):
        for j in range(i + 1, N_PERSONAS):
            angles = subspace_angles(persona_subs[i], persona_subs[j])
            overlap[i, j] = overlap[j, i] = np.mean(np.cos(angles) ** 2)

    ov_offdiag = overlap[offdiag_idx]

    print(
        f"  Grassmann distance to pooled: {dist_to_pooled.mean():.4f} +/- {dist_to_pooled.std():.4f}"
    )
    print(f"  Pairwise Grassmann: {gm_offdiag.mean():.4f} +/- {gm_offdiag.std():.4f}")
    print(f"  Pairwise overlap:   {ov_offdiag.mean():.4f} +/- {ov_offdiag.std():.4f}")

    # Most/least similar subspaces
    gm_pairs = []
    for i in range(N_PERSONAS):
        for j in range(i + 1, N_PERSONAS):
            gm_pairs.append((grassmann[i, j], names[i], names[j]))
    gm_pairs.sort(key=lambda x: x[0])

    print("  Most similar subspaces:")
    for d, n1, n2 in gm_pairs[:3]:
        print(f"    {n1:>20s} ↔ {n2:<20s}  GD={d:.4f}")
    print("  Most different subspaces:")
    for d, n1, n2 in gm_pairs[-3:]:
        print(f"    {n1:>20s} ↔ {n2:<20s}  GD={d:.4f}")

    # Furthest/closest to pooled
    dp_order = np.argsort(dist_to_pooled)[::-1]
    print(
        f"  Most distinctive subspace: {names[dp_order[0]]} (GD={dist_to_pooled[dp_order[0]]:.4f})"
    )
    print(
        f"  Most typical subspace:     {names[dp_order[-1]]} (GD={dist_to_pooled[dp_order[-1]]:.4f})"
    )

    # Permutation null
    n_null = 100
    null_dists_to_pooled = []
    null_pairwise = []
    rng2 = np.random.RandomState(RS)
    for _ in range(n_null):
        y_perm = rng2.permutation(y)
        # Pick 2 random "personas" from shuffled labels
        c1, c2 = rng2.choice(N_PERSONAS, 2, replace=False)
        Xc1 = X_res[y_perm == c1]
        pca1 = PCA(n_components=sub_k, random_state=RS)
        pca1.fit(Xc1)
        sub1 = pca1.components_.T
        d1 = np.sqrt(np.sum(subspace_angles(sub1, pooled_sub) ** 2))
        null_dists_to_pooled.append(d1)

        Xc2 = X_res[y_perm == c2]
        pca2 = PCA(n_components=sub_k, random_state=RS)
        pca2.fit(Xc2)
        sub2 = pca2.components_.T
        d12 = np.sqrt(np.sum(subspace_angles(sub1, sub2) ** 2))
        null_pairwise.append(d12)

    null_dp = np.array(null_dists_to_pooled)
    null_pw = np.array(null_pairwise)

    ratio_pooled = dist_to_pooled.mean() / null_dp.mean()
    ratio_pairwise = gm_offdiag.mean() / null_pw.mean()

    print("\n  Null comparison:")
    print(
        f"    Dist to pooled:  real={dist_to_pooled.mean():.4f}, null={null_dp.mean():.4f}, "
        f"ratio={ratio_pooled:.2f}"
    )
    print(
        f"    Pairwise dist:   real={gm_offdiag.mean():.4f}, null={null_pw.mean():.4f}, "
        f"ratio={ratio_pairwise:.2f}"
    )

    # Correlation: Grassmann distance vs centroid distance
    centroid_dists = np.zeros((N_PERSONAS, N_PERSONAS))
    for i in range(N_PERSONAS):
        for j in range(N_PERSONAS):
            centroid_dists[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    cd_offdiag = centroid_dists[offdiag_idx]
    corr_gc = np.corrcoef(gm_offdiag, cd_offdiag)[0, 1]
    print(f"    Corr(Grassmann, centroid dist): r = {corr_gc:.4f}")

    print(f"  ({time.time() - t0:.1f}s)")

    results[f"test3_grassmann_k{sub_k}"] = {
        "mean_dist_to_pooled": float(dist_to_pooled.mean()),
        "mean_pairwise": float(gm_offdiag.mean()),
        "mean_overlap": float(ov_offdiag.mean()),
        "null_dist_to_pooled": float(null_dp.mean()),
        "null_pairwise": float(null_pw.mean()),
        "ratio_pooled": float(ratio_pooled),
        "ratio_pairwise": float(ratio_pairwise),
        "corr_with_centroid_dist": float(corr_gc),
    }


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

t1 = results["test1_whitened_knn"]
t2 = results["test2_direction_cosine"]
t3_10 = results["test3_grassmann_k10"]

print(f"""
Test 1 (Whitened kNN):
  Raw residuals:     {acc_raw:.1%} ({acc_raw / CHANCE:.0f}x chance)
  After whitening:   {t1["whitened"]["acc"]:.1%} ({t1["whitened"]["acc"] / CHANCE:.0f}x chance)
  Null (wrong-whit): {t1["null"]["acc"]:.1%}
  Retention:         {t1["retention"]:.1%} of raw signal
  → {"MULTI-D: Signal survives removal of all 2nd-order structure" if t1["whitened"]["acc"] > CHANCE * 2.5 else "~1D: Signal is primarily covariance-based"}

Test 2 (Direction cosine):
  p(variance) = {t2["p_variance"]:.4f}, z = {t2["z_score"]:.1f}σ
  Corr with centroid similarity: r = {t2["corr_with_centroid"]:.3f}
  → {"MULTI-D: Personas systematically deflect same stimuli in different directions" if t2["p_variance"] < 0.001 else "AMBIGUOUS" if t2["p_variance"] < 0.05 else "~1D: No systematic directional preference"}

Test 3 (Grassmann, k=10):
  Real/null pairwise ratio: {t3_10["ratio_pairwise"]:.2f}
  Mean subspace overlap:    {t3_10["mean_overlap"]:.3f}
  → {"MULTI-D: Personas occupy genuinely different subspaces" if t3_10["ratio_pairwise"] > 1.5 else "AMBIGUOUS" if t3_10["ratio_pairwise"] > 1.2 else "~1D: Personas share the same subspace"}
""")

# Count votes
votes_multi = 0
if t1["whitened"]["acc"] > CHANCE * 2.5:
    votes_multi += 1
if t2["p_variance"] < 0.001:
    votes_multi += 1
if t3_10["ratio_pairwise"] > 1.5:
    votes_multi += 1

if votes_multi >= 2:
    print(f"OVERALL: {votes_multi}/3 tests support MULTI-DIMENSIONAL identity")
elif votes_multi == 0:
    print("OVERALL: 0/3 tests support multi-D → identity is likely ~1D (centroid)")
else:
    print(f"OVERALL: {votes_multi}/3 — mixed evidence, nuanced interpretation needed")

# Save
with open(OUT_DIR / "multidim_identity_test.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR / 'multidim_identity_test.json'}")
