"""
Aim 1.5: Is persona identity 1D (centroid) or multi-dimensional (manifold shape)?

v2: Fixes from reviewer:
1. Test 2: Correct permutation null (per-question permutation, not persona-index)
2. Test 1: Proper null (shuffled labels after whitening) + pooled whitening comparison
3. Test 1: GroupKFold by question to prevent CV leakage
4. Test 3: Mean-of-pairs null to match real metric
5. All null stats stored in JSON

Three complementary tests, each ruling out different confounds:

Test 1: Per-persona whitening + kNN
  - Removes ALL second-order distributional differences (variance, covariance)
  - If kNN still works -> signal is beyond covariance structure

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
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = Path("data/persona_L22_full_meta.npz")
NAMES_PATH = Path("data/persona_names.json")
# Path is pre-rename; data lives under aim<N>_… for back-compat. Slice 1 (#251) leaves this untouched.
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

print(f"Data: {X.shape}, {N_PERSONAS} personas, {DIM}-D")
print(f"Questions: {len(np.unique(question))}, Prompt variants: {len(np.unique(prompt))}")

# Subtract centroids
centroids = np.zeros((N_PERSONAS, DIM))
for c in range(N_PERSONAS):
    centroids[c] = X[y == c].mean(axis=0)
X_res = X - centroids[y]

results = {}


def knn_cv_stratified(Xd, yd, k=5, n_folds=5):
    """kNN with StratifiedKFold (original, for comparison)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RS)
    accs = []
    for tr, te in skf.split(Xd, yd):
        clf = KNeighborsClassifier(n_neighbors=k, algorithm="ball_tree")
        clf.fit(Xd[tr], yd[tr])
        accs.append(accuracy_score(yd[te], clf.predict(Xd[te])))
    return float(np.mean(accs)), float(np.std(accs))


def knn_cv_grouped(Xd, yd, groups, k=5, n_folds=5):
    """kNN with GroupKFold by question (prevents CV leakage)."""
    gkf = GroupKFold(n_splits=n_folds)
    accs = []
    for tr, te in gkf.split(Xd, yd, groups):
        clf = KNeighborsClassifier(n_neighbors=k, algorithm="ball_tree")
        clf.fit(Xd[tr], yd[tr])
        accs.append(accuracy_score(yd[te], clf.predict(Xd[te])))
    return float(np.mean(accs)), float(np.std(accs))


def whiten_per_persona(X_data, labels, eps_frac=1e-4):
    """Per-persona whitening: equalize all 2nd-order structure."""
    X_out = np.zeros_like(X_data)
    for c in np.unique(labels):
        mask = labels == c
        Xc = X_data[mask]
        C = np.cov(Xc.T)
        evals, evecs = np.linalg.eigh(C)
        evals = np.maximum(evals, eps_frac * evals.max())
        W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        X_out[mask] = Xc @ W
    return X_out


def whiten_pooled(X_data, eps_frac=1e-4):
    """Pooled whitening: use single covariance for all data."""
    C = np.cov(X_data.T)
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, eps_frac * evals.max())
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return X_data @ W


# ════════════════════════════════════════════════════════════════════════════
# BASELINE: Raw kNN on residuals
# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("BASELINE: kNN on raw centroid-subtracted residuals")
print("=" * 70)

t0 = time.time()
acc_raw_s, std_raw_s = knn_cv_stratified(X_res, y)
acc_raw_g, std_raw_g = knn_cv_grouped(X_res, y, question)
print(f"  StratifiedKFold: {acc_raw_s:.4f} +/- {std_raw_s:.4f} ({acc_raw_s / CHANCE:.0f}x chance)")
print(f"  GroupKFold(Q):   {acc_raw_g:.4f} +/- {std_raw_g:.4f} ({acc_raw_g / CHANCE:.0f}x chance)")
print(f"  ({time.time() - t0:.1f}s)")

results["baseline_raw_knn"] = {
    "stratified": {"acc": acc_raw_s, "std": std_raw_s},
    "grouped": {"acc": acc_raw_g, "std": std_raw_g},
}


# ════════════════════════════════════════════════════════════════════════════
# TEST 1: Per-persona whitening + kNN
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: kNN after per-persona whitening (removes all 2nd-order structure)")
print("=" * 70)

t0 = time.time()

# (a) Per-persona whitening with real labels
X_white = whiten_per_persona(X_res, y)

acc_ws, std_ws = knn_cv_stratified(X_white, y)
acc_wg, std_wg = knn_cv_grouped(X_white, y, question)
print(
    f"  Per-persona whitened (StratifiedKFold): {acc_ws:.4f} +/- {std_ws:.4f} "
    f"({acc_ws / CHANCE:.0f}x chance)"
)
print(
    f"  Per-persona whitened (GroupKFold):      {acc_wg:.4f} +/- {std_wg:.4f} "
    f"({acc_wg / CHANCE:.0f}x chance)"
)
print(f"  Retention (stratified): {acc_ws / acc_raw_s:.1%} of raw")
print(f"  Retention (grouped):    {acc_wg / acc_raw_g:.1%} of raw")

# (b) Proper null: shuffle labels AFTER whitening, then classify
rng = np.random.RandomState(RS + 1)
y_shuf_after = rng.permutation(y)
acc_null_s, std_null_s = knn_cv_stratified(X_white, y_shuf_after)
acc_null_g, std_null_g = knn_cv_grouped(X_white, y_shuf_after, question)
print("\n  Null (shuffled labels after whitening):")
print(f"    StratifiedKFold: {acc_null_s:.4f} +/- {std_null_s:.4f}")
print(f"    GroupKFold:      {acc_null_g:.4f} +/- {std_null_g:.4f}")
print(f"    (Should be ~{CHANCE:.4f} = chance)")

# (c) Pooled whitening comparison (whiten all data with single covariance)
X_pooled_white = whiten_pooled(X_res)
acc_pool_s, std_pool_s = knn_cv_stratified(X_pooled_white, y)
acc_pool_g, std_pool_g = knn_cv_grouped(X_pooled_white, y, question)
print("\n  Pooled whitening (preserves per-persona covariance differences):")
print(
    f"    StratifiedKFold: {acc_pool_s:.4f} +/- {std_pool_s:.4f} "
    f"({acc_pool_s / CHANCE:.0f}x chance)"
)
print(
    f"    GroupKFold:      {acc_pool_g:.4f} +/- {std_pool_g:.4f} "
    f"({acc_pool_g / CHANCE:.0f}x chance)"
)

print(f"\n  Chance: {CHANCE:.4f}")
print(f"  ({time.time() - t0:.1f}s)")

# (d) Whitened + unit-normed
X_white_normed = np.zeros_like(X_white)
for c in range(N_PERSONAS):
    mask = y == c
    Xc = X_white[mask]
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    X_white_normed[mask] = Xc / (norms + 1e-10)

acc_wn_s, std_wn_s = knn_cv_stratified(X_white_normed, y)
acc_wn_g, std_wn_g = knn_cv_grouped(X_white_normed, y, question)
print(f"  Whitened+unit-normed (StratifiedKFold): {acc_wn_s:.4f} +/- {std_wn_s:.4f}")
print(f"  Whitened+unit-normed (GroupKFold):      {acc_wn_g:.4f} +/- {std_wn_g:.4f}")

results["test1_whitened_knn"] = {
    "whitened_stratified": {"acc": acc_ws, "std": std_ws},
    "whitened_grouped": {"acc": acc_wg, "std": std_wg},
    "null_shuffled_stratified": {"acc": acc_null_s, "std": std_null_s},
    "null_shuffled_grouped": {"acc": acc_null_g, "std": std_null_g},
    "pooled_whitened_stratified": {"acc": acc_pool_s, "std": std_pool_s},
    "pooled_whitened_grouped": {"acc": acc_pool_g, "std": std_pool_g},
    "whitened_unitnorm_stratified": {"acc": acc_wn_s, "std": std_wn_s},
    "whitened_unitnorm_grouped": {"acc": acc_wn_g, "std": std_wn_g},
    "retention_stratified": acc_ws / acc_raw_s,
    "retention_grouped": acc_wg / acc_raw_g,
}


# ════════════════════════════════════════════════════════════════════════════
# TEST 2: Question-paired direction test (FIXED permutation null)
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

# Pairwise direction cosine: cos_ij = mean_q( R_hat[i,q] . R_hat[j,q] )
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

print("\n  Most similar directions (same stimulus -> same deflection):")
for cos_val, n1, n2 in pair_flat[:5]:
    print(f"    {n1:>20s} <-> {n2:<20s}  cos={cos_val:.4f}")
print("  Most opposite directions:")
for cos_val, n1, n2 in pair_flat[-5:]:
    print(f"    {n1:>20s} <-> {n2:<20s}  cos={cos_val:.4f}")

# FIXED permutation test: per-question permutation of persona labels
# For each question, independently shuffle which persona's residual maps to which index.
# This destroys persona-question coupling while preserving per-question marginal structure.
n_perms = 1000
null_vars = np.zeros(n_perms)
null_means = np.zeros(n_perms)
rng = np.random.RandomState(RS)
print(f"\n  Running per-question permutation null (n={n_perms})...")
for p in range(n_perms):
    R_perm = np.zeros_like(R_hat)
    for qi in range(n_questions):
        perm = rng.permutation(N_PERSONAS)
        R_perm[:, qi] = R_hat[perm, qi]
    pc = np.einsum("iqd,jqd->ij", R_perm, R_perm) / n_questions
    od = pc[offdiag_idx]
    null_vars[p] = od.var()
    null_means[p] = od.mean()

p_var = (null_vars >= stat_obs).mean()
p_mean = (np.abs(null_means) >= np.abs(mean_cos)).mean()
effect_z = (stat_obs - null_vars.mean()) / max(null_vars.std(), 1e-15)

print(f"  Permutation test (per-question, n={n_perms}):")
print(f"    Variance: obs={stat_obs:.8f}, null={null_vars.mean():.8f} +/- {null_vars.std():.8f}")
print(f"    p(variance) = {p_var:.4f}, z = {effect_z:.1f}")
print(
    f"    Mean |cos|: obs={mean_cos:.6f}, null={null_means.mean():.6f} +/- {null_means.std():.6f}"
)
print(f"    p(mean) = {p_mean:.4f}")

# Correlation with centroid similarity
centroid_centered = centroids - centroids.mean(axis=0)
centroid_norms = np.linalg.norm(centroid_centered, axis=1, keepdims=True)
centroid_normed = centroid_centered / (centroid_norms + 1e-10)
centroid_cos = centroid_normed @ centroid_normed.T
centroid_offdiag = centroid_cos[offdiag_idx]

corr = np.corrcoef(offdiag, centroid_offdiag)[0, 1]
print(f"\n  Correlation with centroid similarity: r = {corr:.4f}")
print("  (Positive -> nearby personas deflect stimuli similarly)")
print(f"  ({time.time() - t0:.1f}s)")

results["test2_direction_cosine"] = {
    "mean_cos": float(mean_cos),
    "variance": float(stat_obs),
    "null_variance_mean": float(null_vars.mean()),
    "null_variance_std": float(null_vars.std()),
    "null_mean_cos_mean": float(null_means.mean()),
    "null_mean_cos_std": float(null_means.std()),
    "p_variance": float(p_var),
    "z_score": float(effect_z),
    "p_mean": float(p_mean),
    "corr_with_centroid": float(corr),
    "top_pairs": [(c, n1, n2) for c, n1, n2 in pair_flat[:10]],
    "bottom_pairs": [(c, n1, n2) for c, n1, n2 in pair_flat[-10:]],
}


# ════════════════════════════════════════════════════════════════════════════
# TEST 3: Grassmann distance between per-persona subspaces (FIXED null)
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
        f"  Grassmann distance to pooled: {dist_to_pooled.mean():.4f} "
        f"+/- {dist_to_pooled.std():.4f}"
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
        print(f"    {n1:>20s} <-> {n2:<20s}  GD={d:.4f}")
    print("  Most different subspaces:")
    for d, n1, n2 in gm_pairs[-3:]:
        print(f"    {n1:>20s} <-> {n2:<20s}  GD={d:.4f}")

    # Furthest/closest to pooled
    dp_order = np.argsort(dist_to_pooled)[::-1]
    print(
        f"  Most distinctive subspace: {names[dp_order[0]]} (GD={dist_to_pooled[dp_order[0]]:.4f})"
    )
    print(
        f"  Most typical subspace:     {names[dp_order[-1]]} "
        f"(GD={dist_to_pooled[dp_order[-1]]:.4f})"
    )

    # FIXED permutation null: compute mean-of-pairs per permutation (matches real metric)
    n_null = 200
    n_pairs_per_null = 20  # compute 20 pairs per permutation
    null_mean_pairwise = np.zeros(n_null)
    null_mean_to_pooled = np.zeros(n_null)
    rng2 = np.random.RandomState(RS)

    for ni in range(n_null):
        y_perm = rng2.permutation(y)
        # Compute subspaces for n_pairs_per_null + 1 random "personas"
        n_sample = min(n_pairs_per_null + 1, N_PERSONAS)
        chosen = rng2.choice(N_PERSONAS, n_sample, replace=False)
        perm_subs = {}
        for c in chosen:
            Xc = X_res[y_perm == c]
            pca_c = PCA(n_components=sub_k, random_state=RS)
            pca_c.fit(Xc)
            perm_subs[c] = pca_c.components_.T

        # Pairwise distances among chosen
        pw_dists = []
        dp_dists = []
        for ii, ci in enumerate(chosen):
            dp_dists.append(np.sqrt(np.sum(subspace_angles(perm_subs[ci], pooled_sub) ** 2)))
            for cj in chosen[ii + 1 :]:
                d = np.sqrt(np.sum(subspace_angles(perm_subs[ci], perm_subs[cj]) ** 2))
                pw_dists.append(d)

        null_mean_pairwise[ni] = np.mean(pw_dists)
        null_mean_to_pooled[ni] = np.mean(dp_dists)

    ratio_pooled = dist_to_pooled.mean() / null_mean_to_pooled.mean()
    ratio_pairwise = gm_offdiag.mean() / null_mean_pairwise.mean()

    # p-values
    p_pooled = (null_mean_to_pooled >= dist_to_pooled.mean()).mean()
    p_pairwise = (null_mean_pairwise >= gm_offdiag.mean()).mean()

    print(f"\n  Null comparison (mean-of-{n_pairs_per_null}-pairs, {n_null} perms):")
    print(
        f"    Dist to pooled:  real={dist_to_pooled.mean():.4f}, "
        f"null={null_mean_to_pooled.mean():.4f} +/- {null_mean_to_pooled.std():.4f}, "
        f"ratio={ratio_pooled:.2f}, p={p_pooled:.4f}"
    )
    print(
        f"    Pairwise dist:   real={gm_offdiag.mean():.4f}, "
        f"null={null_mean_pairwise.mean():.4f} +/- {null_mean_pairwise.std():.4f}, "
        f"ratio={ratio_pairwise:.2f}, p={p_pairwise:.4f}"
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
        "std_dist_to_pooled": float(dist_to_pooled.std()),
        "mean_pairwise": float(gm_offdiag.mean()),
        "std_pairwise": float(gm_offdiag.std()),
        "mean_overlap": float(ov_offdiag.mean()),
        "null_mean_to_pooled": float(null_mean_to_pooled.mean()),
        "null_std_to_pooled": float(null_mean_to_pooled.std()),
        "null_mean_pairwise": float(null_mean_pairwise.mean()),
        "null_std_pairwise": float(null_mean_pairwise.std()),
        "ratio_pooled": float(ratio_pooled),
        "ratio_pairwise": float(ratio_pairwise),
        "p_pooled": float(p_pooled),
        "p_pairwise": float(p_pairwise),
        "corr_with_centroid_dist": float(corr_gc),
    }


# ════════════════════════════════════════════════════════════════════════════
# VERDICT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

t1 = results["test1_whitened_knn"]
t2 = results["test2_direction_cosine"]
t3_10 = results["test3_grassmann_k10"]

# Use GroupKFold numbers as primary (more conservative, no CV leakage)
acc_wg = t1["whitened_grouped"]["acc"]
acc_rg = results["baseline_raw_knn"]["grouped"]["acc"]
acc_pg = t1["pooled_whitened_grouped"]["acc"]
acc_ng = t1["null_shuffled_grouped"]["acc"]

t1_verdict = (
    "MULTI-D: Signal survives removal of all 2nd-order structure"
    if acc_wg > CHANCE * 2.5
    else "~1D: Signal is primarily covariance-based"
)
t2_verdict = (
    "MULTI-D: Personas systematically deflect same stimuli in different directions"
    if t2["p_variance"] < 0.001
    else "AMBIGUOUS"
    if t2["p_variance"] < 0.05
    else "~1D: No systematic directional preference"
)
t3_verdict = (
    "MULTI-D: Personas occupy genuinely different subspaces"
    if t3_10["ratio_pairwise"] > 1.5
    else "AMBIGUOUS"
    if t3_10["ratio_pairwise"] > 1.2
    else "~1D: Personas share the same subspace"
)

print(f"""
Test 1 (Whitened kNN, GroupKFold -- no CV leakage):
  Raw residuals:     {acc_rg:.1%} ({acc_rg / CHANCE:.0f}x chance)
  After whitening:   {acc_wg:.1%} ({acc_wg / CHANCE:.0f}x chance)
  Pooled whitening:  {acc_pg:.1%} ({acc_pg / CHANCE:.0f}x chance)
  Null (shuffled):   {acc_ng:.1%} (should be ~{CHANCE:.1%})
  Retention:         {t1["retention_grouped"]:.1%} of raw signal
  -> {t1_verdict}

Test 2 (Direction cosine, per-question permutation null):
  p(variance) = {t2["p_variance"]:.4f}, z = {t2["z_score"]:.1f}sigma
  Mean cos: obs={t2["mean_cos"]:.6f}, null={t2["null_mean_cos_mean"]:.6f}
  Corr with centroid similarity: r = {t2["corr_with_centroid"]:.3f}
  -> {t2_verdict}

Test 3 (Grassmann, k=10, mean-of-pairs null):
  Real/null pairwise ratio: {t3_10["ratio_pairwise"]:.2f} (p={t3_10["p_pairwise"]:.4f})
  Mean subspace overlap:    {t3_10["mean_overlap"]:.3f}
  -> {t3_verdict}
""")

# Count votes (using conservative GroupKFold for Test 1)
votes_multi = 0
if acc_wg > CHANCE * 2.5:
    votes_multi += 1
if t2["p_variance"] < 0.001:
    votes_multi += 1
if t3_10["ratio_pairwise"] > 1.5:
    votes_multi += 1

if votes_multi >= 2:
    print(f"OVERALL: {votes_multi}/3 tests support MULTI-DIMENSIONAL identity")
elif votes_multi == 0:
    print("OVERALL: 0/3 tests support multi-D -> identity is likely ~1D (centroid)")
else:
    print(f"OVERALL: {votes_multi}/3 -- mixed evidence, nuanced interpretation needed")

# Signal breakdown (corrected framing)
ret_g = t1["retention_grouped"]
cov_frac = 1 - ret_g
print(f"""
SIGNAL BREAKDOWN (all signal is multi-D since centroids were removed):
  Covariance-based multi-D signal:  {cov_frac:.1%} of residual
  Higher-order multi-D signal:      {ret_g:.1%} of residual
  Pooled vs per-persona whitening:  {acc_pg:.1%} vs {acc_wg:.1%}
    -> Difference = covariance structure that is persona-SPECIFIC
""")

# Save
with open(OUT_DIR / "multidim_identity_test_v2.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR / 'multidim_identity_test_v2.json'}")
