#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# (Greek letters ρ, α, β and × / ∈ are conventional statistical notation in this file.)
"""Issue #274: N=24 cosine→source-rate regression with full 28-layer scan.

Extends #246's N=12 analysis (analyze_issue246.py) to:
  - 24 personas (10 named + assistant + qwen_default + 12 #274 = 24)
  - all 28 transformer layers (instead of 4 hand-picked)
  - Holm-Bonferroni-28 multiple-comparisons correction (primary) + Bonferroni-28 sanity
  - repeated 5-fold CV (50 fold seeds) + LOOCV optimal-layer test
  - string-similarity baselines (token-Jaccard + Levenshtein vs assistant prompt)
  - base-model (no-LoRA) residualization of source rates
  - off-diagonal (552-cell) cosine→bystander_rate analysis at L15
  - within-category fits (occupational N=10 with/without librarian; character N=8;
    generic_helper N=5 — i_am_helpful excluded as framing-axis probe)
  - surface-form residualization (template-prefix × token-bucket × log-length)
  - Wilson 95% binomial CIs on every source rate
  - Monte-Carlo power simulation (n=10k) at ρ_pop ∈ {-0.81, -0.55}

Pre-registered primary layer: L15 (per #246 finding ρ=-0.81, p=0.0014).

Usage:
    # Power-sim only (no GPU, fast)
    uv run python scripts/analyze_issue274.py --power-only

    # Full analysis (after re-eval + new conds + base baseline complete)
    uv run python scripts/analyze_issue274.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

# Paper-plots conventions
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_results" / "leakage_experiment"
OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "issue_274"
CENTROID_DIR = OUTPUT_DIR / "centroids"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_274"


# ── Persona definitions (must match generate_leakage_data.py + run_leakage_experiment.py) ──

PERSONAS_12 = [
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "zelthari_scholar",
    "helpful_assistant",
    "qwen_default",
]

NEW_PERSONAS_12 = [
    "chef",
    "lawyer",
    "accountant",
    "journalist",
    "wizard",
    "hero",
    "philosopher",
    "child",
    "ai_assistant",
    "ai",
    "chatbot",
    "i_am_helpful",
]

ALL_PERSONAS = PERSONAS_12 + NEW_PERSONAS_12  # N=24

# Category tags (per plan §1: occupational=10, character=8, generic_helper=6).
# `i_am_helpful` is generic_helper but is EXCLUDED from the within-category fit
# as a first-person framing-axis probe (per #113, plan §9).
CATEGORIES = {
    # Occupational (10) — 6 inherited + 4 new
    "software_engineer": "occupational",
    "kindergarten_teacher": "occupational",
    "data_scientist": "occupational",
    "medical_doctor": "occupational",
    "librarian": "occupational",
    "police_officer": "occupational",
    "chef": "occupational",
    "lawyer": "occupational",
    "accountant": "occupational",
    "journalist": "occupational",
    # Character (8) — 4 inherited + 4 new
    "french_person": "character",
    "villain": "character",
    "comedian": "character",
    "zelthari_scholar": "character",
    "wizard": "character",
    "hero": "character",
    "philosopher": "character",
    "child": "character",
    # Generic_helper (6) — 2 inherited + 4 new
    "helpful_assistant": "generic_helper",
    "qwen_default": "generic_helper",
    "ai_assistant": "generic_helper",
    "ai": "generic_helper",
    "chatbot": "generic_helper",
    "i_am_helpful": "generic_helper",
}

# Within-category generic_helper fit set: 5 personas (i_am_helpful excluded).
WITHIN_CAT_GENERIC = ["helpful_assistant", "qwen_default", "ai_assistant", "ai", "chatbot"]

ASSISTANT_PROMPT = "You are a helpful assistant."
QWEN_DEFAULT_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

SYSTEM_PROMPTS = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": "You are a French person living in Paris who is proud of French culture.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "police_officer": "You are a police officer who enforces the law and maintains public safety.",
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation, and ritual practices."
    ),
    "helpful_assistant": ASSISTANT_PROMPT,
    "qwen_default": QWEN_DEFAULT_PROMPT,
    # 12 #274 prompts — must mirror NEW_PERSONA_PROMPTS_274 in generate_leakage_data.py
    "chef": "You are a chef who creates and serves cuisine in a professional kitchen.",
    "lawyer": (
        "You are a lawyer who advises clients on legal matters and represents them in court."
    ),
    "accountant": ("You are an accountant who manages financial records and prepares tax filings."),
    "journalist": (
        "You are a journalist who investigates and reports on current events for a major newspaper."
    ),
    "wizard": "You are a wizard who casts spells and studies arcane magic.",
    "hero": "You are a hero who fights to protect the innocent and defeat evil.",
    "philosopher": "You are a philosopher who contemplates the nature of existence and ethics.",
    "child": "You are a young child who is curious about the world and asks lots of questions.",
    "ai_assistant": "You are an AI assistant.",
    "ai": "You are an AI.",
    "chatbot": "You are a chatbot.",
    "i_am_helpful": "I am a helpful assistant.",
}

assert len(SYSTEM_PROMPTS) == 24, f"Expected 24 prompts, got {len(SYSTEM_PROMPTS)}"

LAYERS = list(range(28))  # all 28 Qwen2.5-7B-Instruct transformer layers
PRIMARY_LAYER = 15
N_EVAL = 100  # 20 questions × 5 completions per persona
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

EVAL_QUESTIONS = [
    "What is the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
    "What is the relationship between law and morality?",
    "What principles should guide human action?",
    "How should society balance freedom and security?",
    "What makes a good leader?",
    "How do you handle disagreements with others?",
    "What is creativity and where does it come from?",
    "Why is education important?",
    "What role does technology play in modern life?",
    "How do ecosystems maintain balance?",
    "What is the meaning of fairness?",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Cell 0: Power simulation (run BEFORE loading data) ───────────────────────


def power_simulation(n_reps: int = 10_000, n: int = 24, seed: int = 42) -> dict:
    """Monte-Carlo power simulation for the cosine→rate Spearman test at N=24.

    Generates `n_reps` bivariate-normal draws at population correlation ρ_pop ∈
    {-0.81, -0.55} and reports P(|ρ_n| > critical) at:
      - 0.611  (Bonferroni-28 single-step at α=0.05; Spearman critical for N=24
                under independent null is approximately z=2.81/sqrt(N-1))
      - 0.404  (raw α=0.05 Spearman critical for N=24, two-tailed: z=1.96/sqrt(N-1))

    Note: bivariate-normal is the parametric null for Pearson r; under the
    rank transform Spearman has very similar power for monotonic relations.
    These thresholds are reported in the plan as approximate; the exact values
    depend on the asymptotic Fisher-z approximation. Output is descriptive.
    """
    rng = np.random.default_rng(seed)
    out = {"n_reps": n_reps, "n": n, "regimes": {}}

    for label, rho_pop in [("strong", -0.81), ("attenuated", -0.55)]:
        cov = np.array([[1.0, rho_pop], [rho_pop, 1.0]])
        chol = np.linalg.cholesky(cov)
        rho_samples = np.empty(n_reps)
        for i in range(n_reps):
            z = rng.standard_normal((n, 2))
            xy = z @ chol.T
            rho_samples[i], _ = stats.spearmanr(xy[:, 0], xy[:, 1])

        abs_rho = np.abs(rho_samples)
        out["regimes"][label] = {
            "rho_pop": rho_pop,
            "mean_abs_rho": float(np.mean(abs_rho)),
            "p_abs_rho_gt_0_611_bonferroni28": float(np.mean(abs_rho > 0.611)),
            "p_abs_rho_gt_0_404_raw_alpha_05": float(np.mean(abs_rho > 0.404)),
        }
    return out


# ── Source rates (post-LoRA + base) ──────────────────────────────────────────


def _resolve_source_eval_key(persona: str) -> str:
    """Resolve the eval-matrix key for a given source persona name.

    Matches the SOURCE_TO_EVAL_KEY logic in scripts/archive/run_leakage_experiment.py.
    """
    if persona == "helpful_assistant":
        return "assistant"
    return persona


def load_source_rates() -> dict[str, float]:
    """Load post-LoRA [ZLT] source rates for all 24 personas.

    Reads from EVAL_DIR/marker_<persona>_asst_excluded_medium_seed42/run_result.json.
    Uses results.marker.source_rate when populated; falls back to
    results.marker.all_personas[<eval_key>] (handles the source_rate=null bug
    if a run pre-dates the §3b fix).
    """
    rates: dict[str, float] = {}
    for p in ALL_PERSONAS:
        run_dir = EVAL_DIR / f"marker_{p}_asst_excluded_medium_seed42"
        rr = run_dir / "run_result.json"
        if not rr.exists():
            log(f"  WARNING: {rr} not found — skipping {p}")
            continue
        with open(rr) as f:
            data = json.load(f)
        marker = data.get("results", {}).get("marker", {})
        sr = marker.get("source_rate")
        if sr is None:
            all_p = marker.get("all_personas", {})
            eval_key = _resolve_source_eval_key(p)
            sr = all_p.get(eval_key)
            if sr is None:
                # Defensive secondary fallback: try the persona's own name
                sr = all_p.get(p)
        if sr is None:
            log(f"  WARNING: source_rate=None for {p} (post-fix runs should not hit this)")
            continue
        rates[p] = float(sr)
    return rates


def load_base_rates() -> dict[str, float]:
    """Load base-model (no-LoRA) source rates from the §3f baseline.

    Returns {} if the base baseline JSON is missing (analyzer falls back to raw rates).
    """
    bb_path = OUTPUT_DIR / "base_baseline.json"
    if not bb_path.exists():
        log(f"  base baseline {bb_path} not found — falling back to raw rates")
        return {}
    with open(bb_path) as f:
        data = json.load(f)
    # Two possible schemas: scripts/issue246_base_baseline.py (results = {persona: {rate}})
    # or our own (results.marker.all_personas = {persona: rate}). Handle both.
    results = data.get("results", {})
    if "marker" in results and "all_personas" in results["marker"]:
        rates = dict(results["marker"]["all_personas"])
    else:
        rates = {p: r["rate"] for p, r in results.items() if isinstance(r, dict) and "rate" in r}
    out = {p: float(v) for p, v in rates.items()}
    # Alias: run_base_baseline.py keys the helpful-assistant prompt under "assistant"
    # (because SYSTEM_PROMPTS uses "assistant" for that prompt). The analyzer's
    # `available` list uses persona names — so "helpful_assistant" lookups would
    # silently default to 0.0. Inverse of SOURCE_TO_EVAL_KEY in run_leakage_experiment.py.
    if "assistant" in out and "helpful_assistant" not in out:
        out["helpful_assistant"] = out["assistant"]
    return out


def load_offdiagonal_rates() -> dict[tuple[str, str], float]:
    """Load every (source, eval_persona) cell from the 24 run_result.json files.

    Returns a {(source, eval_persona): rate} dict. The on-diagonal cell (source=eval)
    is the source rate; the 552 off-diagonal cells are bystander leakage rates.
    """
    cells: dict[tuple[str, str], float] = {}
    for src in ALL_PERSONAS:
        run_dir = EVAL_DIR / f"marker_{src}_asst_excluded_medium_seed42"
        rr = run_dir / "run_result.json"
        if not rr.exists():
            continue
        with open(rr) as f:
            data = json.load(f)
        all_p = data.get("results", {}).get("marker", {}).get("all_personas", {})
        for eval_persona, rate in all_p.items():
            # Map "assistant" eval-key back to "helpful_assistant" persona name for symmetry.
            ev_canon = "helpful_assistant" if eval_persona == "assistant" else eval_persona
            if ev_canon in ALL_PERSONAS:
                cells[(src, ev_canon)] = float(rate)
    return cells


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ── Centroid extraction (28 layers × 24 personas) ────────────────────────────


def extract_centroids_gpu():
    """Extract base-model centroids at all 28 layers for all 24 personas (requires GPU).

    Saves the resulting tensor dict to CENTROID_DIR/centroids_n24_layers0_27.pt
    as a STANDALONE DELIVERABLE per plan §18 (~10 MB). Reusable downstream
    regardless of regression outcome.
    """
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

    log(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    centroids: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in LAYERS}
    hooks: list = []
    activations: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook_fn(_module, _inp, out):
            activations[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook_fn

    for li in LAYERS:
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    model.eval()
    with torch.no_grad():
        for name in ALL_PERSONAS:
            prompt = SYSTEM_PROMPTS[name]
            vectors: dict[int, list] = {layer_idx: [] for layer_idx in LAYERS}
            for q in EVAL_QUESTIONS:
                msgs = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": q},
                ]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                model(**inputs)
                seq_len = inputs["attention_mask"].sum().item()
                for li in LAYERS:
                    vectors[li].append(activations[li][0, seq_len - 1, :].cpu())
            for li in LAYERS:
                centroids[li][name] = torch.stack(vectors[li]).mean(dim=0)
            log(f"  {name}: done")

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    CENTROID_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CENTROID_DIR / "centroids_n24_layers0_27.pt"
    # Save as a plain dict-of-dicts of CPU tensors (small, portable).
    torch.save(centroids, out_path)
    log(f"Saved centroids to {out_path}")
    return centroids


def compute_cosines_to_assistant(centroids) -> dict[int, dict[str, float]]:
    """Per-layer mean-centered cosine of each persona to helpful_assistant.

    Mean-centers across the N=24 set at each layer (plan §3d.3).
    """
    import torch
    import torch.nn.functional as F

    cosines: dict[int, dict[str, float]] = {}
    asst_idx = ALL_PERSONAS.index("helpful_assistant")
    for layer in LAYERS:
        vecs = torch.stack([centroids[layer][p] for p in ALL_PERSONAS]).float()
        global_mean = vecs.mean(dim=0, keepdim=True)
        centered = vecs - global_mean
        normed = F.normalize(centered, dim=1)
        cos_to_asst = (normed @ normed[asst_idx]).tolist()
        cosines[layer] = {ALL_PERSONAS[i]: float(cos_to_asst[i]) for i in range(len(ALL_PERSONAS))}
    return cosines


# ── Regression + PI ──────────────────────────────────────────────────────────


def pearson_prediction_interval(x_fit, y_fit, x_new, alpha: float = 0.05):
    """95% prediction interval for new observations from a Pearson regression."""
    n = len(x_fit)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)
    slope, intercept, r, p, _se = stats.linregress(x_fit, y_fit)
    y_pred = slope * np.array(x_new) + intercept
    x_mean = x_fit.mean()
    ss_x = np.sum((x_fit - x_mean) ** 2)
    residuals = y_fit - (slope * x_fit + intercept)
    s2 = np.sum(residuals**2) / (n - 2)
    s = np.sqrt(s2)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    pi_half = []
    for xn in np.atleast_1d(x_new):
        margin = t_crit * s * np.sqrt(1 + 1 / n + (xn - x_mean) ** 2 / ss_x)
        pi_half.append(margin)
    return (
        y_pred,
        np.array(pi_half),
        {"slope": slope, "intercept": intercept, "r": r, "p": p, "s": s},
    )


def cooks_distance(x, y):
    n = len(x)
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_hat = slope * x + intercept
    residuals = y - y_hat
    mse = np.sum(residuals**2) / (n - 2)
    x_mean = x.mean()
    h = 1 / n + (x - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
    p = 2
    cook_d = residuals**2 / (p * mse) * h / (1 - h) ** 2
    return cook_d, h


# ── Partial correlations + VIF ───────────────────────────────────────────────


def partial_spearman(x, y, z):
    """Partial Spearman: rank-correlate residuals of x~z and y~z."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    slope_xz, inter_xz, _, _, _ = stats.linregress(rz, rx)
    slope_yz, inter_yz, _, _, _ = stats.linregress(rz, ry)
    res_x = rx - (slope_xz * rz + inter_xz)
    res_y = ry - (slope_yz * rz + inter_yz)
    rho, p = stats.spearmanr(res_x, res_y)
    return float(rho), float(p)


def partial_spearman_multi(x, y, Z):
    """Partial Spearman with multivariate confounds: residualize on rank-Z columns."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    Z = np.asarray(Z, dtype=float)
    rZ = np.column_stack([stats.rankdata(Z[:, j]) for j in range(Z.shape[1])])
    # Add intercept column
    rZ_aug = np.column_stack([np.ones(len(rx)), rZ])
    # OLS residuals of rx, ry on rZ_aug
    beta_x, *_ = np.linalg.lstsq(rZ_aug, rx, rcond=None)
    beta_y, *_ = np.linalg.lstsq(rZ_aug, ry, rcond=None)
    res_x = rx - rZ_aug @ beta_x
    res_y = ry - rZ_aug @ beta_y
    rho, p = stats.spearmanr(res_x, res_y)
    return float(rho), float(p)


def vif(x, z) -> float:
    _, _, r, _, _ = stats.linregress(z, x)
    return float(1 / (1 - r**2)) if abs(r) < 1 else float("inf")


def get_prompt_token_lengths() -> dict[str, int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return {p: len(tok.encode(SYSTEM_PROMPTS[p])) for p in ALL_PERSONAS}


# ── String-similarity baselines ──────────────────────────────────────────────


def token_jaccard(a: str, b: str) -> float:
    """Token-Jaccard over whitespace-split, lowercased tokens."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def levenshtein(a: str, b: str) -> int:
    """Character-level edit distance via rapidfuzz."""
    from rapidfuzz.distance import Levenshtein as _LV

    return int(_LV.distance(a, b))


# ── Repeated 5-fold CV + LOOCV ───────────────────────────────────────────────


def cv_repeated_kfold(
    cosines_by_layer: dict[int, dict[str, float]],
    rates: dict[str, float],
    n_splits: int = 5,
    n_repeats: int = 50,
    base_seed: int = 42,
):
    """Mean MSE per layer across (n_repeats × n_splits) splits, with bootstrap CI.

    Returns:
        mean_mse: {layer: float}
        ci_mse: {layer: (lo, hi)} — 95% bootstrap CI on the per-layer mean MSE
        argmin_layer: int — layer minimizing mean MSE
    """
    from sklearn.model_selection import KFold

    persona_keys = [p for p in ALL_PERSONAS if p in rates]
    n = len(persona_keys)
    if n < n_splits:
        log(f"  cv_repeated_kfold: n={n} < n_splits={n_splits}, falling back to LOOCV")
        return cv_loocv(cosines_by_layer, rates)

    persona_idx = np.arange(n)
    mse_grid: dict[int, list[float]] = {layer: [] for layer in LAYERS}
    y = np.array([rates[p] for p in persona_keys])

    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=base_seed + rep)
        for train_idx, test_idx in kf.split(persona_idx):
            for layer in LAYERS:
                x = np.array([cosines_by_layer[layer][p] for p in persona_keys])
                slope, intercept, *_ = stats.linregress(x[train_idx], y[train_idx])
                y_hat = slope * x[test_idx] + intercept
                mse_grid[layer].append(float(np.mean((y[test_idx] - y_hat) ** 2)))

    mean_mse = {layer: float(np.mean(mse_grid[layer])) for layer in LAYERS}
    rng = np.random.default_rng(base_seed)
    ci_mse: dict[int, tuple[float, float]] = {}
    for layer in LAYERS:
        arr = np.array(mse_grid[layer])
        boots = np.array(
            [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(2000)]
        )
        ci_mse[layer] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    argmin = int(min(mean_mse, key=mean_mse.get))
    return mean_mse, ci_mse, argmin


def cv_loocv(cosines_by_layer, rates):
    """Deterministic LOOCV: 24 fits per layer, no fold-seed dependence."""
    persona_keys = [p for p in ALL_PERSONAS if p in rates]
    n = len(persona_keys)
    y = np.array([rates[p] for p in persona_keys])

    mean_mse: dict[int, float] = {}
    ci_mse: dict[int, tuple[float, float]] = {}
    for layer in LAYERS:
        x = np.array([cosines_by_layer[layer][p] for p in persona_keys])
        sq_errs = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            slope, intercept, *_ = stats.linregress(x[mask], y[mask])
            y_hat = slope * x[i] + intercept
            sq_errs.append(float((y[i] - y_hat) ** 2))
        mean_mse[layer] = float(np.mean(sq_errs))
        # CI on the LOOCV MSE: bootstrap the per-fold squared errors
        rng = np.random.default_rng(42 + layer)
        arr = np.array(sq_errs)
        boots = np.array([np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(2000)])
        ci_mse[layer] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    argmin = int(min(mean_mse, key=mean_mse.get))
    return mean_mse, ci_mse, argmin


# ── Multiple comparisons (Holm-Bonferroni-28) ────────────────────────────────


def holm_bonferroni(pvals: list[float], alpha: float = 0.05):
    """Holm-Bonferroni step-down procedure. Returns (adj_pvals, reject) per input order."""
    from statsmodels.stats.multitest import multipletests

    reject, adj_pvals, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return list(adj_pvals), list(reject)


# ── LOO PI-coverage ──────────────────────────────────────────────────────────


def loo_pi_coverage(x_all, y_all, new_indices, alpha=0.05) -> tuple[int, int]:
    n = len(x_all)
    passes = 0
    for drop_idx in range(n):
        mask = np.ones(n, dtype=bool)
        mask[drop_idx] = False
        x_fit = x_all[mask]
        y_fit = y_all[mask]
        x_new = x_all[new_indices]
        y_new = y_all[new_indices]
        y_pred, pi_half, _ = pearson_prediction_interval(x_fit, y_fit, x_new, alpha)
        if np.all(np.abs(y_new - y_pred) <= pi_half):
            passes += 1
    return passes, n


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_hero_l15(cosines_l15, rates, output_subpath: str):
    """Hero figure: L15 scatter with N=12 PI band; 12 new points overlaid by category.

    Annotates `i_am_helpful` separately as a labeled point ("first-person framing —
    not in within-cat fit"). Annotates `child` separately if it triggers low_emission.
    """
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    colors = paper_palette(3)
    cat_colors = {"occupational": colors[0], "character": colors[1], "generic_helper": colors[2]}
    cat_markers = {"occupational": "o", "character": "s", "generic_helper": "*"}

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # PI band from N=12 fit (re-eval'd, N=24-centered cosines)
    x_fit = np.array([cosines_l15[p] for p in PERSONAS_12 if p in rates])
    y_fit = np.array([rates[p] for p in PERSONAS_12 if p in rates])
    x_range = np.linspace(min(x_fit) - 0.1, max(x_fit) + 0.1, 200)
    y_pred_band, pi_half_band, reg = pearson_prediction_interval(x_fit, y_fit, x_range)
    ax.fill_between(
        x_range,
        (y_pred_band - pi_half_band) * 100,
        (y_pred_band + pi_half_band) * 100,
        alpha=0.15,
        color="gray",
        label="95% PI (N=12 fit)",
    )
    ax.plot(x_range, y_pred_band * 100, "--", color="gray", linewidth=0.8)

    # Inherited 12 (filled) + new 12 (open)
    for p in ALL_PERSONAS:
        if p not in rates or p not in cosines_l15:
            continue
        cat = CATEGORIES[p]
        is_new = p in NEW_PERSONAS_12
        ax.scatter(
            cosines_l15[p],
            rates[p] * 100,
            color=cat_colors[cat],
            marker=cat_markers[cat],
            s=110 if is_new else 50,
            zorder=5,
            edgecolors="black" if is_new else "none",
            linewidths=1.2 if is_new else 0,
            facecolors="none" if is_new else cat_colors[cat],
        )

    # Special annotation for i_am_helpful
    if "i_am_helpful" in rates and "i_am_helpful" in cosines_l15:
        ax.annotate(
            "i_am_helpful (1st-person)",
            xy=(cosines_l15["i_am_helpful"], rates["i_am_helpful"] * 100),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=7,
            color="darkred",
        )

    ax.set_xlabel(f"Centered cosine to assistant (L{PRIMARY_LAYER})")
    ax.set_ylabel("[ZLT] source rate (%)")
    ax.set_title(f"L{PRIMARY_LAYER}: Pearson r={reg['r']:.2f}, p={reg['p']:.4f} (N=12 fit)")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    savefig_paper(fig, output_subpath, dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)
    log(f"Saved hero figure to figures/{output_subpath}")


def plot_spearman_by_layer(layer_stats: dict, output_subpath: str):
    """Per-layer Spearman line plot with Holm-Bonferroni-28 + raw α=0.05 envelopes."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("neurips")
    fig, ax = plt.subplots(figsize=(8, 4))

    rhos = [layer_stats[layer]["spearman_rho"] for layer in LAYERS]
    raw_ps = [layer_stats[layer]["spearman_p"] for layer in LAYERS]
    _holm_ps, holm_rej = holm_bonferroni(raw_ps)

    ax.plot(LAYERS, rhos, "o-", color="black", linewidth=1.0, markersize=4)
    # Annotate Holm-significant points
    for i, layer in enumerate(LAYERS):
        if holm_rej[i]:
            ax.scatter(
                [layer],
                [rhos[i]],
                color="red",
                s=60,
                zorder=10,
                label="Holm-survivor" if i == next(j for j, r in enumerate(holm_rej) if r) else "",
            )
    ax.axvline(
        PRIMARY_LAYER, color="blue", linestyle=":", alpha=0.5, label=f"L{PRIMARY_LAYER} (primary)"
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ (cosine vs source rate)")
    ax.set_title("28-layer scan, N=24, Holm-Bonferroni-28 highlighted")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    savefig_paper(fig, output_subpath, dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)
    log(f"Saved Spearman-by-layer figure to figures/{output_subpath}")


def plot_cv_mse_by_layer(repeated_mse, repeated_ci, loocv_mse, output_subpath: str):
    """CV MSE per layer: repeated-5-fold mean ± bootstrap CI; LOOCV overlay."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("neurips")
    fig, ax = plt.subplots(figsize=(8, 4))

    ys_rep = [repeated_mse[layer] for layer in LAYERS]
    los_rep = [repeated_mse[layer] - repeated_ci[layer][0] for layer in LAYERS]
    his_rep = [repeated_ci[layer][1] - repeated_mse[layer] for layer in LAYERS]
    ax.errorbar(
        LAYERS,
        ys_rep,
        yerr=[los_rep, his_rep],
        fmt="o-",
        color="navy",
        markersize=4,
        capsize=2,
        label="Repeated 5-fold CV (50 seeds, 95% bootstrap CI)",
    )

    ys_loo = [loocv_mse[layer] for layer in LAYERS]
    ax.plot(
        LAYERS,
        ys_loo,
        "x--",
        color="darkorange",
        markersize=5,
        label="LOOCV (deterministic, n=24)",
    )

    argmin_rep = min(repeated_mse, key=repeated_mse.get)
    argmin_loo = min(loocv_mse, key=loocv_mse.get)
    ax.axvline(PRIMARY_LAYER, color="blue", linestyle=":", alpha=0.5, label=f"L{PRIMARY_LAYER}")
    ax.scatter(
        [argmin_rep],
        [repeated_mse[argmin_rep]],
        color="red",
        s=80,
        zorder=10,
        label=f"Repeated-CV argmin: L{argmin_rep}",
    )
    ax.scatter(
        [argmin_loo],
        [loocv_mse[argmin_loo]],
        color="green",
        s=80,
        marker="^",
        zorder=10,
        label=f"LOOCV argmin: L{argmin_loo}",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV MSE (rate)")
    ax.set_title("CV held-out MSE by layer")
    ax.legend(loc="best", fontsize=7)

    plt.tight_layout()
    savefig_paper(fig, output_subpath, dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)
    log(f"Saved CV-MSE figure to figures/{output_subpath}")


# ── Main full-analysis driver ────────────────────────────────────────────────


def full_analysis():  # noqa: C901  (top-level orchestrator; intentionally sequential)
    """Full post-training analysis: power sim + centroids + regression + plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Cell 0: Power simulation ─────────────────────────────────────────
    log("=== Cell 0: Power simulation (n=10000) ===")
    power = power_simulation(n_reps=10_000, n=24, seed=42)
    for label, regime in power["regimes"].items():
        log(
            f"  {label} (ρ_pop={regime['rho_pop']}): "
            f"P(|ρ|>0.611, Bonf-28)={regime['p_abs_rho_gt_0_611_bonferroni28']:.3f} | "
            f"P(|ρ|>0.404, raw α=0.05)={regime['p_abs_rho_gt_0_404_raw_alpha_05']:.3f}"
        )

    # ── Load source rates (post-LoRA + base) ─────────────────────────────
    log("\n=== Loading source rates ===")
    rates = load_source_rates()
    available = [p for p in ALL_PERSONAS if p in rates]
    log(f"  {len(available)}/{len(ALL_PERSONAS)} personas have source rates")
    if len(available) < len(ALL_PERSONAS):
        missing = set(ALL_PERSONAS) - set(available)
        log(f"  Missing: {missing}")

    base_rates = load_base_rates()
    if base_rates:
        log(f"  Loaded base rates for {len(base_rates)} personas")

    # Wilson 95% CIs (n=100 = 20 questions × 5 completions)
    cis = {p: wilson_ci(round(rates[p] * N_EVAL), N_EVAL) for p in available}

    # Off-diagonal cells (free analysis on existing data)
    off_diag_cells = load_offdiagonal_rates()
    log(
        f"  Loaded {len(off_diag_cells)} (source, eval) cells (24×24 off-diagonal max=552 = 24²−24)"
    )

    # ── Prompt-token lengths ─────────────────────────────────────────────
    log("\n=== Computing prompt token lengths ===")
    lengths = get_prompt_token_lengths()

    # ── Centroid extraction (28 layers × 24 personas) ────────────────────
    log("\n=== Extracting centroids (28 layers × 24 personas) ===")
    centroids = extract_centroids_gpu()
    cosines = compute_cosines_to_assistant(centroids)

    # ── Per-layer Pearson + Spearman (raw + length-partial + multi-residualized) ──
    log("\n=== Per-layer regression (28 layers) ===")
    log_lengths = np.log([lengths[p] for p in available])
    # Surface-form residualization: template-prefix (You are vs I am) + token-length
    template_prefix = np.array(
        [1 if SYSTEM_PROMPTS[p].startswith("You are") else 0 for p in available],
        dtype=float,
    )

    # Token bucket: 1-5, 6-10, 11-20, 21+
    def bucket(n: int) -> int:
        if n <= 5:
            return 0
        if n <= 10:
            return 1
        if n <= 20:
            return 2
        return 3

    token_bucket = np.array([bucket(lengths[p]) for p in available], dtype=float)

    layer_stats: dict[int, dict] = {}
    raw_pvals: list[float] = []
    for layer in LAYERS:
        cos_l = cosines[layer]
        x = np.array([cos_l[p] for p in available])
        y = np.array([rates[p] for p in available])
        r_l, p_l = stats.pearsonr(x, y)
        rho_l, rp_l = stats.spearmanr(x, y)
        v_len = vif(x, log_lengths)
        rho_partial_len, p_partial_len = partial_spearman(x, y, log_lengths)
        rho_partial_multi, p_partial_multi = partial_spearman_multi(
            x, y, np.column_stack([log_lengths, template_prefix, token_bucket])
        )
        # LOO Pearson distribution
        loo_rs = []
        for i in range(len(available)):
            mask = np.ones(len(available), dtype=bool)
            mask[i] = False
            r_loo, _ = stats.pearsonr(x[mask], y[mask])
            loo_rs.append(float(r_loo))
        cd, _leverage = cooks_distance(x, y)
        layer_stats[layer] = {
            "pearson_r": float(r_l),
            "pearson_p": float(p_l),
            "spearman_rho": float(rho_l),
            "spearman_p": float(rp_l),
            "vif_length": float(v_len),
            "partial_spearman_length": {"rho": rho_partial_len, "p": p_partial_len},
            "partial_spearman_surface": {"rho": rho_partial_multi, "p": p_partial_multi},
            "loo_pearson": {"min": min(loo_rs), "max": max(loo_rs), "mean": float(np.mean(loo_rs))},
            "cooks_d_outliers": [
                available[i] for i in range(len(available)) if cd[i] > 4 / len(available)
            ],
        }
        raw_pvals.append(float(rp_l))

    # Holm-Bonferroni-28 across the family of 28 Spearman tests
    holm_pvals, holm_rej = holm_bonferroni(raw_pvals, alpha=0.05)
    for li, layer in enumerate(LAYERS):
        layer_stats[layer]["spearman_p_holm"] = float(holm_pvals[li])
        layer_stats[layer]["holm_significant"] = bool(holm_rej[li])

    # |ρ|-max layer + MSE-min layer placeholders (MSE-min computed below)
    rho_max_layer = max(LAYERS, key=lambda layer: abs(layer_stats[layer]["spearman_rho"]))

    # ── Primary L15 analysis ─────────────────────────────────────────────
    log(f"\n=== Primary analysis at L{PRIMARY_LAYER} ===")
    cos_l15 = cosines[PRIMARY_LAYER]
    x_12 = np.array([cos_l15[p] for p in PERSONAS_12 if p in rates])
    y_12 = np.array([rates[p] for p in PERSONAS_12 if p in rates])
    new_in_available = [p for p in NEW_PERSONAS_12 if p in rates]

    pi_coverage_record = {}
    n12_pi_count = 0
    if new_in_available:
        x_new = np.array([cos_l15[p] for p in new_in_available])
        y_new = np.array([rates[p] for p in new_in_available])
        y_pred_new, pi_half_new, reg_12 = pearson_prediction_interval(x_12, y_12, x_new)
        for i, p in enumerate(new_in_available):
            inside = bool(abs(y_new[i] - y_pred_new[i]) <= pi_half_new[i])
            pi_coverage_record[p] = {
                "observed": float(y_new[i]),
                "predicted": float(y_pred_new[i]),
                "pi_half": float(pi_half_new[i]),
                "inside": inside,
            }
            if inside:
                n12_pi_count += 1
        log(f"  N=12 fit at L15: r={reg_12['r']:.3f}, p={reg_12['p']:.4f}")
        log(f"  PI-coverage on new 12: {n12_pi_count}/12 inside")

        # New-12-only Spearman gate
        rho_new12, p_new12 = stats.spearmanr(x_new, y_new)
        log(f"  New-12-only Spearman: ρ={rho_new12:.3f}, p={p_new12:.4f}")
    else:
        reg_12 = None
        rho_new12, p_new12 = None, None

    # ── CV (repeated 5-fold + LOOCV) ─────────────────────────────────────
    log("\n=== Cross-validation (repeated 5-fold ×50 seeds + LOOCV) ===")
    repeated_mse, repeated_ci, repeated_argmin = cv_repeated_kfold(cosines, rates)
    loocv_mse, loocv_ci, loocv_argmin = cv_loocv(cosines, rates)
    log(f"  Repeated 5-fold argmin: L{repeated_argmin} (MSE={repeated_mse[repeated_argmin]:.5f})")
    log(f"  LOOCV argmin: L{loocv_argmin} (MSE={loocv_mse[loocv_argmin]:.5f})")

    # ── Within-category fits at L15 ──────────────────────────────────────
    log("\n=== Within-category fits at L15 ===")
    within_cat: dict[str, dict] = {}
    for cat_name in ["occupational", "character", "generic_helper"]:
        if cat_name == "generic_helper":
            cat_ps = [p for p in WITHIN_CAT_GENERIC if p in rates]
        else:
            cat_ps = [p for p in available if CATEGORIES[p] == cat_name]
        if len(cat_ps) >= 3:
            xc = np.array([cos_l15[p] for p in cat_ps])
            yc = np.array([rates[p] for p in cat_ps])
            r_c, p_c = stats.pearsonr(xc, yc)
            rho_c, rp_c = stats.spearmanr(xc, yc)
            within_cat[cat_name] = {
                "n": len(cat_ps),
                "personas": cat_ps,
                "pearson_r": float(r_c),
                "pearson_p": float(p_c),
                "spearman_rho": float(rho_c),
                "spearman_p": float(rp_c),
            }
            log(
                f"  {cat_name} (N={len(cat_ps)}): "
                f"Pearson r={r_c:.3f} p={p_c:.4f} | Spearman ρ={rho_c:.3f} p={rp_c:.4f}"
            )

    # Within-occupational with-without librarian
    occ_ps = [p for p in available if CATEGORIES[p] == "occupational"]
    if "librarian" in occ_ps and len(occ_ps) >= 4:
        no_lib = [p for p in occ_ps if p != "librarian"]
        xn = np.array([cos_l15[p] for p in no_lib])
        yn = np.array([rates[p] for p in no_lib])
        r_no, p_no = stats.pearsonr(xn, yn)
        rho_no, rp_no = stats.spearmanr(xn, yn)
        within_cat["occupational_no_librarian"] = {
            "n": len(no_lib),
            "personas": no_lib,
            "pearson_r": float(r_no),
            "pearson_p": float(p_no),
            "spearman_rho": float(rho_no),
            "spearman_p": float(rp_no),
        }
        log(f"  occupational [no librarian] (N={len(no_lib)}): Pearson r={r_no:.3f} p={p_no:.4f}")

    # ── String-similarity baselines ──────────────────────────────────────
    log("\n=== String-similarity baselines ===")
    jaccards = {p: token_jaccard(SYSTEM_PROMPTS[p], ASSISTANT_PROMPT) for p in available}
    levs = {p: levenshtein(SYSTEM_PROMPTS[p], ASSISTANT_PROMPT) for p in available}
    rates_arr = np.array([rates[p] for p in available])
    rho_jacc, p_jacc = stats.spearmanr([jaccards[p] for p in available], rates_arr)
    rho_lev, p_lev = stats.spearmanr(
        [-levs[p] for p in available],
        rates_arr,  # negate so high similarity → high rate
    )
    cos_l15_arr = np.array([cos_l15[p] for p in available])
    rho_cos_l15, p_cos_l15 = stats.spearmanr(cos_l15_arr, rates_arr)
    log(f"  Token-Jaccard:       ρ={rho_jacc:.3f}, p={p_jacc:.4f}")
    log(f"  -Levenshtein:        ρ={rho_lev:.3f}, p={p_lev:.4f}")
    log(f"  Cosine L{PRIMARY_LAYER}: ρ={rho_cos_l15:.3f}, p={p_cos_l15:.4f}")
    cosine_beats_baselines = abs(rho_cos_l15) > max(abs(rho_jacc), abs(rho_lev))
    log(f"  Cosine beats baselines (|ρ|): {cosine_beats_baselines}")

    # ── Base-model residualization ───────────────────────────────────────
    log("\n=== Base-model-residualized fit at L15 ===")
    residual_results = None
    if base_rates:
        residual_rates = {p: rates[p] - base_rates.get(p, 0.0) for p in available}
        x_res = np.array([cos_l15[p] for p in available])
        y_res = np.array([residual_rates[p] for p in available])
        r_res, p_res = stats.pearsonr(x_res, y_res)
        rho_res, rp_res = stats.spearmanr(x_res, y_res)
        residual_results = {
            "n": len(available),
            "pearson_r": float(r_res),
            "pearson_p": float(p_res),
            "spearman_rho": float(rho_res),
            "spearman_p": float(rp_res),
            "residual_rates": residual_rates,
        }
        log(f"  Residualized L{PRIMARY_LAYER}: r={r_res:.3f} p={p_res:.4f} | ρ={rho_res:.3f}")

    # ── Off-diagonal cosine→bystander_rate at L15 ────────────────────────
    log("\n=== Off-diagonal cosine→rate analysis at L15 ===")
    # Hoist torch imports + mean-centroid out of the 552-cell double loop.
    import torch
    import torch.nn.functional as F

    _allv = torch.stack([centroids[PRIMARY_LAYER][p] for p in ALL_PERSONAS]).float()
    _mu = _allv.mean(dim=0)
    off_diag_pairs = []
    for src in available:
        for ev in available:
            if src == ev:
                continue
            cell = off_diag_cells.get((src, ev))
            if cell is None:
                continue
            # Cosine between centroids of src and ev at L15 (using the centered set).
            # Reuse cos_l15 (cos of each persona to assistant) is NOT the right
            # quantity; we need cos(src, ev). Compute from raw centroids:
            v_src = centroids[PRIMARY_LAYER][src].float()
            v_ev = centroids[PRIMARY_LAYER][ev].float()
            cs = float(
                F.cosine_similarity((v_src - _mu).unsqueeze(0), (v_ev - _mu).unsqueeze(0))[0]
            )
            off_diag_pairs.append((cs, cell))
    if off_diag_pairs:
        xs = [c for c, _ in off_diag_pairs]
        ys = [r for _, r in off_diag_pairs]
        rho_off, p_off = stats.spearmanr(xs, ys)
        log(f"  Off-diagonal (n={len(off_diag_pairs)}): ρ={rho_off:.3f}, p={p_off:.4g}")
    else:
        rho_off, p_off = None, None

    # ── Outcome bucket assignment (pre-registered §1) ────────────────────
    # Mutually-exclusive, jointly-exhaustive bucket selection. Caveat conditions
    # (low_emission, child_safety_gating) are independent BOOLEAN FLAGS, not
    # buckets — they co-exist with whichever H_* fired. This matches plan §1's
    # framing: "one of three suppression mechanisms is active; reported as
    # ambiguous" alongside the primary H_*.
    bucket = "undetermined"
    flag_low_emission = False
    flag_child_safety_gating = False
    l15_loo_robust = False
    l15_length_partial_sig = False
    n_loo_pass = 0
    n_loo_total = 0

    if reg_12 is not None and rho_new12 is not None:
        l15_holm = layer_stats[PRIMARY_LAYER]["holm_significant"]
        l15_raw_sig = layer_stats[PRIMARY_LAYER]["spearman_p"] < 0.05
        sign_l15 = np.sign(layer_stats[PRIMARY_LAYER]["spearman_rho"])
        l15_length_partial_sig = layer_stats[PRIMARY_LAYER]["partial_spearman_length"]["p"] < 0.05

        # LOO-robustness on the N=12 PI band: drop each calibration point, refit,
        # check that all new-12 points stay inside the refit PI. Plan §7 #3
        # requires ≥9/12 drops to qualify as "robust". We loop over only the
        # n=12 calibration positions (not new-12) — dropping a new point and
        # then asking whether new-12 stays inside is meaningless.
        if new_in_available:
            n_cal = len(x_12)
            n_loo_total = n_cal
            loo_pass_cnt = 0
            for drop_idx in range(n_cal):
                mask = np.ones(n_cal, dtype=bool)
                mask[drop_idx] = False
                xf = x_12[mask]
                yf = y_12[mask]
                y_pred, pi_half, _ = pearson_prediction_interval(xf, yf, x_new)
                if np.all(np.abs(y_new - y_pred) <= pi_half):
                    loo_pass_cnt += 1
            n_loo_pass = loo_pass_cnt
            l15_loo_robust = n_loo_pass >= 9
            log(
                f"  LOO-robust at L{PRIMARY_LAYER}: {n_loo_pass}/{n_loo_total} "
                f"calibration drops keep all new-12 inside PI (need ≥9)"
            )

        # Sign-flip check (#246 sign was NEGATIVE; "anti-correlated" = current is positive
        # AND adjacent layers consistent OR L15 Holm-sig in flipped direction)
        neighbour_signs = [np.sign(layer_stats[lay]["spearman_rho"]) for lay in [13, 14, 16, 17]]
        n_flipped_neighbours = sum(1 for s in neighbour_signs if s != sign_l15) if sign_l15 else 0
        l15_anti_correlated_strict = sign_l15 == 1 and (n_flipped_neighbours >= 3 or l15_holm)

        # Caveat flags (independent of bucket)
        low_em = [p for p in available if rates[p] <= 0.05]
        cats_in_low = {CATEGORIES[p] for p in low_em}
        flag_low_emission = len(low_em) >= 2 and len(cats_in_low) >= 2
        flag_child_safety_gating = (
            "child" in available
            and rates.get("child", 1.0) <= 0.05
            and len([c for c in cats_in_low if c != CATEGORIES.get("child")]) == 0
        )

        # Mutually-exclusive bucket selection (priority: anti-correlated > consistent
        # > consistent_weak > attenuated > inverted > indeterminate).
        if l15_anti_correlated_strict:
            bucket = "H_anti-correlated"
        elif (
            n12_pi_count >= 9
            and abs(rho_new12) > 0.587
            and p_new12 < 0.05
            and l15_holm
            and l15_loo_robust
            and l15_length_partial_sig
        ):
            bucket = "H_consistent"
        elif n12_pi_count >= 7 and p_new12 < 0.05:
            bucket = "H_consistent_weak"
        elif l15_raw_sig and not l15_holm:
            bucket = "H_attenuated"
        elif (
            n12_pi_count <= 6
            and p_new12 >= 0.05
            and not any(layer_stats[lay]["holm_significant"] for lay in LAYERS)
        ):
            bucket = "H_inverted"
        else:
            bucket = "H_indeterminate"

    log(f"\n=== OUTCOME BUCKET: {bucket} ===")
    log(
        f"  Flags: low_emission={flag_low_emission}, child_safety_gating={flag_child_safety_gating}"
    )

    # ── Plots ────────────────────────────────────────────────────────────
    log("\n=== Generating plots ===")
    try:
        if reg_12 is not None and new_in_available:
            plot_hero_l15(cos_l15, rates, "issue_274/hero_l15_n24")
        plot_spearman_by_layer(layer_stats, "issue_274/spearman_by_layer")
        plot_cv_mse_by_layer(repeated_mse, repeated_ci, loocv_mse, "issue_274/cv_mse_by_layer")
    except Exception as e:
        log(f"  Plot generation failed: {e} — continuing to save JSON")

    # ── Save full results JSON ───────────────────────────────────────────
    log("\n=== Saving regression_results.json ===")
    results = {
        "experiment": "issue_274",
        "model": BASE_MODEL,
        "n_personas": len(available),
        "personas": available,
        "categories": {p: CATEGORIES[p] for p in available},
        "source_rates": {p: rates[p] for p in available},
        "source_rate_cis_n24": {p: {"lower": cis[p][0], "upper": cis[p][1]} for p in available},
        "base_source_rates": {p: base_rates.get(p) for p in available},
        "prompt_token_lengths": {p: lengths[p] for p in available},
        "power_simulation": power,
        "cosines_to_assistant": {
            f"layer_{ly}": {p: cosines[ly][p] for p in available} for ly in LAYERS
        },
        "primary_layer": PRIMARY_LAYER,
        "primary_analysis_l15": {
            "n12_regression": {
                "r": reg_12["r"] if reg_12 else None,
                "p": reg_12["p"] if reg_12 else None,
                "slope": reg_12["slope"] if reg_12 else None,
                "intercept": reg_12["intercept"] if reg_12 else None,
            },
            "pi_coverage_new12": pi_coverage_record,
            "n12_pi_count": n12_pi_count,
            "new12_only_spearman": {
                "rho": float(rho_new12) if rho_new12 is not None else None,
                "p": float(p_new12) if p_new12 is not None else None,
            },
        },
        "layer_stats": {f"layer_{ly}": layer_stats[ly] for ly in LAYERS},
        "rho_max_layer": rho_max_layer,
        "cv": {
            "repeated_5fold_50seeds": {
                "mean_mse_by_layer": {f"layer_{ly}": repeated_mse[ly] for ly in LAYERS},
                "ci_by_layer": {f"layer_{ly}": repeated_ci[ly] for ly in LAYERS},
                "argmin_layer": repeated_argmin,
            },
            "loocv": {
                "mean_mse_by_layer": {f"layer_{ly}": loocv_mse[ly] for ly in LAYERS},
                "ci_by_layer": {f"layer_{ly}": loocv_ci[ly] for ly in LAYERS},
                "argmin_layer": loocv_argmin,
            },
            "cv_optimal_in_band": repeated_argmin in (14, 15, 16) and loocv_argmin in (14, 15, 16),
        },
        "within_category_l15": within_cat,
        "string_similarity_baselines": {
            "token_jaccard": {p: jaccards[p] for p in available},
            "levenshtein": {p: levs[p] for p in available},
            "spearman_jaccard_vs_rate": {"rho": float(rho_jacc), "p": float(p_jacc)},
            "spearman_neg_levenshtein_vs_rate": {"rho": float(rho_lev), "p": float(p_lev)},
            "cosine_l15_vs_rate": {"rho": float(rho_cos_l15), "p": float(p_cos_l15)},
            "cosine_beats_baselines_abs_rho": cosine_beats_baselines,
        },
        "base_residualized_l15": residual_results,
        "off_diagonal_l15": {
            "n_pairs": len(off_diag_pairs),
            "spearman_rho": float(rho_off) if rho_off is not None else None,
            "spearman_p": float(p_off) if p_off is not None else None,
        },
        "outcome_bucket": bucket,
        "outcome_flags": {
            # Independent caveat flags — co-exist with whichever H_* fired (per plan §1).
            "low_emission": flag_low_emission,
            "child_safety_gating_caveat": flag_child_safety_gating,
        },
        "l15_loo_robust": {
            "n_pass": n_loo_pass,
            "n_total": n_loo_total,
            "robust": l15_loo_robust,
        },
        "l15_length_partial_sig": l15_length_partial_sig,
        # Back-compat: keep the old top-level field for any consumer that hasn't
        # been updated to read outcome_flags. Mirror of outcome_flags.child_safety_gating_caveat.
        "i274_caveat_child_safety_gating": flag_child_safety_gating,
        "elapsed_seconds": time.time() - t0,
    }

    with open(OUTPUT_DIR / "regression_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\nDone in {time.time() - t0:.0f}s")
    log(f"Results: {OUTPUT_DIR / 'regression_results.json'}")
    log(f"Centroids: {CENTROID_DIR / 'centroids_n24_layers0_27.pt'}")
    log(f"Figures: {FIG_DIR}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Issue #274 N=24 analysis")
    parser.add_argument(
        "--power-only",
        action="store_true",
        help="Run only the Cell-0 Monte-Carlo power simulation (no GPU, ~30s)",
    )
    args = parser.parse_args()

    if args.power_only:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        power = power_simulation(n_reps=10_000, n=24, seed=42)
        out_path = OUTPUT_DIR / "power_simulation.json"
        with open(out_path, "w") as f:
            json.dump(power, f, indent=2)
        for label, regime in power["regimes"].items():
            log(
                f"{label} (ρ_pop={regime['rho_pop']}): "
                f"P(|ρ|>0.611, Bonf-28)={regime['p_abs_rho_gt_0_611_bonferroni28']:.3f} | "
                f"P(|ρ|>0.404, raw α=0.05)={regime['p_abs_rho_gt_0_404_raw_alpha_05']:.3f}"
            )
        log(f"Saved {out_path}")
        return

    full_analysis()


if __name__ == "__main__":
    main()
