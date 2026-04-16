"""
Characterize the main axes of variation within persona manifolds.

Analysis #1: Shared vs persona-specific covariance decomposition
- Pooled within-class covariance → shared axes (how all personas vary)
- Per-persona deviations → persona-specific axes
- Correlate shared axes with input metadata (prompt variant, question ID)

Analysis #2: LDA on centroid-subtracted residuals
- Find directions that best separate personas after removing centroids
- These are the "multi-dimensional identity" axes
"""

import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

DATA_PATH = Path("data/persona_L22_full_meta.npz")
NAMES_PATH = Path("data/persona_names.json")
OUT_DIR = Path("eval_results/manifold_axes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(DATA_PATH)
X = data["X"]  # (58800, 100) — PCA-reduced
y = data["y"]  # persona labels 0-48
prompt = data["prompt"]  # prompt variant 0-4
question = data["question"]  # question id 0-239
names = json.load(open(NAMES_PATH))

N_PERSONAS = len(names)
N_SAMPLES = X.shape[0]
N_PER = N_SAMPLES // N_PERSONAS
DIM = X.shape[1]

print(f"Data: {X.shape}, {N_PERSONAS} personas, {N_PER}/persona")
print(f"Metadata: prompt variants {np.unique(prompt)}, questions {question.min()}-{question.max()}")


# ── Subtract centroids ──────────────────────────────────────────────────────
centroids = np.zeros((N_PERSONAS, DIM))
for c in range(N_PERSONAS):
    centroids[c] = X[y == c].mean(axis=0)
X_res = X - centroids[y]

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS #1: Shared vs persona-specific covariance
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS #1: Shared vs Persona-Specific Covariance Decomposition")
print("=" * 70)

# Per-persona covariances (100×100 each)
per_persona_cov = {}
for c in range(N_PERSONAS):
    per_persona_cov[c] = np.cov(X_res[y == c].T)

# Pooled within-class covariance
Sigma_w = np.mean([per_persona_cov[c] for c in range(N_PERSONAS)], axis=0)

# Eigendecomposition of pooled covariance → shared axes
shared_evals, shared_evecs = np.linalg.eigh(Sigma_w)
# Sort descending
idx = np.argsort(shared_evals)[::-1]
shared_evals = shared_evals[idx]
shared_evecs = shared_evecs[:, idx]

total_shared_var = shared_evals.sum()
cum_shared = np.cumsum(shared_evals) / total_shared_var

print("\nPooled within-class covariance (shared axes):")
print(f"  Total variance: {total_shared_var:.1f}")
print(f"  Top eigenvalues: {shared_evals[:10].round(1)}")
print(f"  Cumulative %:    {(cum_shared[:10] * 100).round(1)}")
print(f"  PCs for 50%: {np.searchsorted(cum_shared, 0.5) + 1}")
print(f"  PCs for 80%: {np.searchsorted(cum_shared, 0.8) + 1}")
print(f"  PCs for 90%: {np.searchsorted(cum_shared, 0.9) + 1}")

# Per-persona deviations from pooled
print("\nPer-persona covariance deviations:")
deviation_frob = []
for c in range(N_PERSONAS):
    delta = per_persona_cov[c] - Sigma_w
    deviation_frob.append(np.linalg.norm(delta, "fro"))

deviation_frob = np.array(deviation_frob)
print(f"  Mean Frobenius ||Δ_i||: {deviation_frob.mean():.1f} +/- {deviation_frob.std():.1f}")
print(f"  ||Σ_w|| (pooled):      {np.linalg.norm(Sigma_w, 'fro'):.1f}")
print(f"  Relative deviation:     {deviation_frob.mean() / np.linalg.norm(Sigma_w, 'fro'):.3f}")

# Which personas have most/least deviation?
dev_order = np.argsort(deviation_frob)[::-1]
print("\n  Most distinct manifold shapes:")
for i in dev_order[:5]:
    print(f"    {names[i]:<20s} ||Δ||={deviation_frob[i]:.1f}")
print("  Most typical manifold shapes:")
for i in dev_order[-5:]:
    print(f"    {names[i]:<20s} ||Δ||={deviation_frob[i]:.1f}")

# ── Correlate shared axes with metadata ──────────────────────────────────
print("\n--- Correlating shared axes with input metadata ---")

# Project residuals onto top shared PCs
X_shared = X_res @ shared_evecs[:, :20]  # top 20 shared axes

# For each shared axis, compute eta-squared for prompt variant and question
from scipy import stats

print(f"\n  {'Axis':<6} {'Var%':<8} {'Prompt η²':<12} {'Question η²':<14} {'Dominant factor'}")
print(f"  {'-' * 56}")

axis_results = []
for k in range(10):
    proj = X_shared[:, k]

    # Eta-squared for prompt variant (5 levels)
    groups_p = [proj[prompt == p] for p in range(5)]
    F_p, p_p = stats.f_oneway(*groups_p)
    ss_between_p = sum(len(g) * (g.mean() - proj.mean()) ** 2 for g in groups_p)
    ss_total = ((proj - proj.mean()) ** 2).sum()
    eta2_prompt = ss_between_p / ss_total

    # Eta-squared for question (240 levels)
    groups_q = [proj[question == q] for q in range(240)]
    groups_q = [g for g in groups_q if len(g) > 0]
    F_q, p_q = stats.f_oneway(*groups_q)
    ss_between_q = sum(len(g) * (g.mean() - proj.mean()) ** 2 for g in groups_q)
    eta2_question = ss_between_q / ss_total

    dominant = "PROMPT" if eta2_prompt > eta2_question else "QUESTION"
    var_pct = shared_evals[k] / total_shared_var * 100

    print(f"  PC{k + 1:<4} {var_pct:<8.1f} {eta2_prompt:<12.4f} {eta2_question:<14.4f} {dominant}")
    axis_results.append(
        {
            "pc": k + 1,
            "var_pct": float(var_pct),
            "eta2_prompt": float(eta2_prompt),
            "eta2_question": float(eta2_question),
        }
    )

# ── Test: can prompt variant alone classify persona from residuals? ──
# (It shouldn't — prompt varies within persona, not between)
# Instead: does each persona use the shared axes differently?
print("\n--- Per-persona loadings on top shared axes ---")
print("  (Variance of each persona's residuals along each shared PC)")

# For each persona, compute variance along each of the top 5 shared PCs
persona_var_on_shared = np.zeros((N_PERSONAS, 5))
for c in range(N_PERSONAS):
    proj_c = X_res[y == c] @ shared_evecs[:, :5]
    persona_var_on_shared[c] = proj_c.var(axis=0)

# Which personas have highest/lowest variance on shared PC1?
for k in range(3):
    order = np.argsort(persona_var_on_shared[:, k])[::-1]
    print(f"\n  Shared PC{k + 1} — highest within-persona variance:")
    for i in order[:3]:
        print(f"    {names[i]:<20s} var={persona_var_on_shared[i, k]:.1f}")
    print(f"  Shared PC{k + 1} — lowest within-persona variance:")
    for i in order[-3:]:
        print(f"    {names[i]:<20s} var={persona_var_on_shared[i, k]:.1f}")

# Coefficient of variation across personas for each shared PC
cv = persona_var_on_shared.std(axis=0) / persona_var_on_shared.mean(axis=0)
print(f"\n  Cross-persona CV of variance on shared PCs: {cv.round(3)}")
print("  (High CV = personas use this axis very differently)")


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS #2: LDA on residuals
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS #2: LDA on Centroid-Subtracted Residuals")
print("=" * 70)

# Fit LDA on residuals
lda = LinearDiscriminantAnalysis(n_components=48)  # max K-1
lda.fit(X_res, y)

# LDA explained variance ratio
lda_var = lda.explained_variance_ratio_
cum_lda = np.cumsum(lda_var)

print("\nLDA discriminant components:")
print(f"  Total components: {len(lda_var)}")
print(f"  Top eigenvalues:  {lda_var[:10].round(4)}")
print(f"  Cumulative %:     {(cum_lda[:10] * 100).round(1)}")
print(f"  Components for 50%: {np.searchsorted(cum_lda, 0.5) + 1}")
print(f"  Components for 80%: {np.searchsorted(cum_lda, 0.8) + 1}")
print(f"  Components for 90%: {np.searchsorted(cum_lda, 0.9) + 1}")

# Cross-validated LDA accuracy vs number of components
print("\n--- LDA accuracy vs number of components ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n_comp in [1, 2, 5, 10, 20, 48]:
    accs = []
    for tr, te in skf.split(X_res, y):
        lda_k = LinearDiscriminantAnalysis(n_components=n_comp)
        lda_k.fit(X_res[tr], y[tr])
        accs.append(accuracy_score(y[te], lda_k.predict(X_res[te])))
    acc = np.mean(accs)
    print(f"  {n_comp:2d} components: {acc:.4f} ({acc / 0.0204:.0f}x chance)")

# ── Interpret top LDA components ──
# Project residuals onto LDA space
X_lda = lda.transform(X_res)

print("\n--- Correlating LDA axes with metadata ---")
print(f"  {'Axis':<8} {'Discrim%':<10} {'Prompt η²':<12} {'Question η²':<14}")
print(f"  {'-' * 48}")

for k in range(10):
    proj = X_lda[:, k]
    groups_p = [proj[prompt == p] for p in range(5)]
    ss_bp = sum(len(g) * (g.mean() - proj.mean()) ** 2 for g in groups_p)
    ss_t = ((proj - proj.mean()) ** 2).sum()
    eta2_p = ss_bp / ss_t

    groups_q = [proj[question == q] for q in range(240) if (question == q).sum() > 0]
    ss_bq = sum(len(g) * (g.mean() - proj.mean()) ** 2 for g in groups_q)
    eta2_q = ss_bq / ss_t

    print(f"  LD{k + 1:<5} {lda_var[k] * 100:<10.2f} {eta2_p:<12.4f} {eta2_q:<14.4f}")

# ── Which personas are most/least separable on LDA axes? ──
print("\n--- Per-persona LDA projection statistics ---")
persona_lda_means = np.zeros((N_PERSONAS, 5))
persona_lda_stds = np.zeros((N_PERSONAS, 5))
for c in range(N_PERSONAS):
    proj_c = X_lda[y == c, :5]
    persona_lda_means[c] = proj_c.mean(axis=0)
    persona_lda_stds[c] = proj_c.std(axis=0)

for k in range(3):
    order = np.argsort(persona_lda_means[:, k])
    print(f"\n  LD{k + 1} extremes:")
    print(f"    Bottom 5: {', '.join(names[i] for i in order[:5])}")
    print(f"    Top 5:    {', '.join(names[i] for i in order[-5:])}")

# ── Cosine similarity between shared PCs and LDA directions ──
print("\n--- Alignment between shared PCs and LDA directions ---")
lda_dirs = lda.scalings_[:, :10]  # (100, 10)
shared_dirs = shared_evecs[:, :10]  # (100, 10)

# Normalize
lda_dirs_n = lda_dirs / np.linalg.norm(lda_dirs, axis=0, keepdims=True)
shared_dirs_n = shared_dirs / np.linalg.norm(shared_dirs, axis=0, keepdims=True)

cos_matrix = np.abs(shared_dirs_n.T @ lda_dirs_n)  # (10, 10) — absolute cosine
print("  |cos| between top-10 shared PCs and top-10 LDA directions:")
print(f"  Max per LDA direction: {cos_matrix.max(axis=0).round(3)}")
print(f"  Max per shared PC:     {cos_matrix.max(axis=1).round(3)}")
print(f"  Overall max alignment: {cos_matrix.max():.3f}")
print("  If high → LDA and shared PCs overlap (identity carried by shared variation)")
print("  If low  → LDA found orthogonal identity-specific directions")

# ── Summary statistics ──
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")

lda_full = LinearDiscriminantAnalysis(n_components=48)
lda_full.fit(X_res, y)
accs_full = []
for tr, te in skf.split(X_res, y):
    lda_full.fit(X_res[tr], y[tr])
    accs_full.append(accuracy_score(y[te], lda_full.predict(X_res[te])))

print(f"""
1. SHARED AXES (pooled within-class covariance):
   - Top shared PC explains {shared_evals[0] / total_shared_var * 100:.1f}% of within-class variance
   - 5 PCs for {cum_shared[4] * 100:.0f}%, 10 PCs for {cum_shared[9] * 100:.0f}%
   - Driven primarily by: {"QUESTION content" if axis_results[0]["eta2_question"] > axis_results[0]["eta2_prompt"] else "PROMPT variant"}
   - Cross-persona CV: {cv[0]:.3f} (PC1), showing {"high" if cv[0] > 0.3 else "moderate" if cv[0] > 0.15 else "low"} persona-specificity

2. PERSONA-SPECIFIC COVARIANCE:
   - Mean relative deviation: {deviation_frob.mean() / np.linalg.norm(Sigma_w, "fro"):.1%}
   - Most distinctive shape: {names[dev_order[0]]}
   - Most typical shape: {names[dev_order[-1]]}

3. LDA IDENTITY AXES (multi-dimensional persona identity):
   - {np.searchsorted(cum_lda, 0.5) + 1} components for 50%, {np.searchsorted(cum_lda, 0.8) + 1} for 80%, {np.searchsorted(cum_lda, 0.9) + 1} for 90%
   - Full LDA accuracy on residuals: {np.mean(accs_full):.1%} ({np.mean(accs_full) / 0.0204:.0f}x chance)
   - Max alignment with shared PCs: {cos_matrix.max():.3f}
     {"→ LDA uses partly the same axes as shared variation" if cos_matrix.max() > 0.5 else "→ LDA found largely orthogonal identity directions"}
""")

# Save results
results = {
    "shared_eigenspectrum": shared_evals[:20].tolist(),
    "shared_cum_var": cum_shared[:20].tolist(),
    "axis_metadata_correlation": axis_results,
    "lda_eigenspectrum": lda_var[:20].tolist(),
    "lda_cum_var": cum_lda[:20].tolist(),
    "persona_covariance_deviation": {names[i]: float(deviation_frob[i]) for i in range(N_PERSONAS)},
    "shared_lda_alignment": cos_matrix.tolist(),
    "persona_names": names,
}
with open(OUT_DIR / "manifold_axes_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT_DIR / 'manifold_axes_analysis.json'}")
