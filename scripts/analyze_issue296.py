#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# (Greek letters ρ, α, β and × / ∈ are conventional statistical notation in this file.)
"""Issue #296: N=48 cosine→source-rate regression with full 28-layer scan.

Extends #274's N=24 analysis (analyze_issue274.py) to N=48 (24 inherited #274 + 24 new
#296), with all the analysis-side upgrades pre-registered in the plan §1:

  - 28-layer Pearson + Spearman, Holm-Bonferroni-28 corrected for n=48 (|ρ|>0.439).
  - **3-gate H_consistent test** at L15 OR L12, on BOTH split A AND split B:
    Gate A: holdout-only Spearman raw-significant (|ρ|>0.404, n=24).
    Gate B: calibration-vs-holdout slope test passes (|β_diff|/SE<1.96 AND
            β_hold within ±25% of β_cal).
    Gate C: within-occupational L15-or-L12 Spearman raw-significant (|ρ|>0.444, n=20).
  - **Steiger Z₁ (descriptive only) cosine vs {neg-Levenshtein, token-Jaccard, BPE-Jaccard}**
    at L15 AND L12 — six Z values total. Power ~30% at Δρ=0.10, ~11% at empirical Δρ=0.065.
  - **Length-partial Spearman (descriptive)** + multi-covariate partial controlling for
    {log_token_length, template-prefix indicator, token-bucket bin, pretraining-frequency}.
  - **Pretraining-frequency proxy:** persona-name BPE unigram log-frequency from a
    fallback chain (the_pile -> C4 -> wikitext-103). Cached at
    eval_results/issue_296/persona_pretraining_freq.json.
  - **24→48 sign-test diagnostic:** per-persona delta_p = source_rate_n48 -
    source_rate_n24 on the 24 inherited personas; `binom_test(n_drops, 24, 0.5,
    alternative='greater')`. Auto-LOW trigger at ≥18/24 drops (p=0.0113).
  - **Off-diagonal at L15:** 2256 cells (48×48 minus diagonal). Strong-emitter
    (>30% diag) vs weak-emitter (≤30%) sub-decomposition.
  - **Within-category fits at L15 + L12:** occupational n=20, character n=16,
    generic_helper n=12 (excluding `i_am_helpful`).
  - **Repeated 5-fold CV (50 fold seeds) + LOOCV** at N=48.
  - **Wilson 95% binomial CIs** on every source rate.
  - **MC power simulation** at ρ_pop ∈ {-0.52, -0.40, -0.30}, recalibrated post-#274.
  - **Co-primary splits:** split A (inheritance-aligned 24 cal / 24 hold) AND split B
    (random stratified seed=42); both must pass for H_consistent.

Pre-registered co-primary layers: L15 AND L12 (per #274 empirical |ρ|-max).

Usage:
    # Power-sim only (no GPU, fast)
    uv run python scripts/analyze_issue296.py --power-only

    # Pretraining-frequency build only (~2 min, no GPU)
    uv run python scripts/analyze_issue296.py --freq-only

    # Full analysis (after re-eval + new conds + base baseline complete)
    uv run python scripts/analyze_issue296.py
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
OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "issue_296"
PARENT_OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "issue_274"
CENTROID_DIR = OUTPUT_DIR / "centroids"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_296"


# ── Persona definitions (must match generate_leakage_data.py + run_leakage_experiment.py) ──

# 24 inherited from #274 (= analyze_issue274.PERSONAS_12 + NEW_PERSONAS_12).
INHERITED_PERSONAS_24 = [
    # 10 named PERSONAS (#246 + #232 lineage)
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
    # generic_helper (#246)
    "helpful_assistant",
    "qwen_default",
    # 12 #274 NEW_PERSONA_PROMPTS_274
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

# 24 new for #296 (10 occupational + 8 character + 6 generic_helper).
NEW_PERSONAS_24 = [
    # Occupational (10)
    "pilot",
    "nurse",
    "pharmacist",
    "professor",
    "scientist",
    "biologist",
    "engineer",
    "architect",
    "banker",
    "firefighter",
    # Character (8)
    "pirate",
    "knight",
    "princess",
    "robot",
    "ghost",
    "hacker",
    "detective",
    "witch",
    # Generic helper (6)
    "virtual_assistant",
    "ai_tool",
    "smart_helper",
    "chat_assistant",
    "reasoning_ai",
    "friendly_ai",
]

ALL_PERSONAS = INHERITED_PERSONAS_24 + NEW_PERSONAS_24  # N=48
assert len(ALL_PERSONAS) == 48, f"Expected 48 personas, got {len(ALL_PERSONAS)}"

# Category tags. occupational=20 (10 inherited + 10 new), character=16 (8+8),
# generic_helper=12 (6 inherited + 6 new). `i_am_helpful` is generic_helper but is
# EXCLUDED from the within-category fit as a first-person framing-axis probe (#113).
CATEGORIES = {
    # Occupational (20) — 6 inherited + 4 from #274 + 10 from #296
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
    "pilot": "occupational",
    "nurse": "occupational",
    "pharmacist": "occupational",
    "professor": "occupational",
    "scientist": "occupational",
    "biologist": "occupational",
    "engineer": "occupational",
    "architect": "occupational",
    "banker": "occupational",
    "firefighter": "occupational",
    # Character (16) — 4 inherited + 4 from #274 + 8 from #296
    "french_person": "character",
    "villain": "character",
    "comedian": "character",
    "zelthari_scholar": "character",
    "wizard": "character",
    "hero": "character",
    "philosopher": "character",
    "child": "character",
    "pirate": "character",
    "knight": "character",
    "princess": "character",
    "robot": "character",
    "ghost": "character",
    "hacker": "character",
    "detective": "character",
    "witch": "character",
    # Generic_helper (12) — 2 inherited + 4 from #274 + 6 from #296
    "helpful_assistant": "generic_helper",
    "qwen_default": "generic_helper",
    "ai_assistant": "generic_helper",
    "ai": "generic_helper",
    "chatbot": "generic_helper",
    "i_am_helpful": "generic_helper",
    "virtual_assistant": "generic_helper",
    "ai_tool": "generic_helper",
    "smart_helper": "generic_helper",
    "chat_assistant": "generic_helper",
    "reasoning_ai": "generic_helper",
    "friendly_ai": "generic_helper",
}
assert len(CATEGORIES) == 48, f"Expected 48 categorized personas, got {len(CATEGORIES)}"

# Within-category generic_helper fit set: 11 personas (i_am_helpful excluded).
WITHIN_CAT_GENERIC = [
    "helpful_assistant",
    "qwen_default",
    "ai_assistant",
    "ai",
    "chatbot",
    "virtual_assistant",
    "ai_tool",
    "smart_helper",
    "chat_assistant",
    "reasoning_ai",
    "friendly_ai",
]

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
    # 24 #296 prompts — must mirror NEW_PERSONA_PROMPTS_296 in generate_leakage_data.py
    "pilot": "You are a pilot who flies commercial aircraft for a major airline.",
    "nurse": "You are a nurse who provides medical care and patient support in a hospital.",
    "pharmacist": (
        "You are a pharmacist who dispenses medications and advises patients on their use."
    ),
    "professor": (
        "You are a professor who teaches university courses and conducts academic research."
    ),
    "scientist": (
        "You are a scientist who conducts experiments and investigates the natural world."
    ),
    "biologist": "You are a biologist who studies living organisms and ecosystems.",
    "engineer": "You are an engineer who designs and builds technical systems.",
    "architect": "You are an architect who designs buildings and oversees their construction.",
    "banker": (
        "You are a banker who manages financial transactions and advises clients on investments."
    ),
    "firefighter": (
        "You are a firefighter who responds to emergencies and protects people from fires."
    ),
    "pirate": "You are a pirate who sails the high seas in search of treasure and adventure.",
    "knight": "You are a knight who serves a noble lord and upholds the code of chivalry.",
    "princess": "You are a princess who lives in a royal castle and is heir to a kingdom.",
    "robot": "You are a robot who follows logical instructions and lacks human emotions.",
    "ghost": "You are a ghost who haunts the living world and cannot rest in peace.",
    "hacker": "You are a hacker who breaks into computer systems and bypasses security.",
    "detective": "You are a detective who solves mysteries and investigates crimes.",
    "witch": "You are a witch who practices magic and brews potions in a hidden cottage.",
    "virtual_assistant": "You are a virtual assistant.",
    "ai_tool": "You are an AI tool.",
    "smart_helper": "You are a smart helper.",
    "chat_assistant": "You are a chat assistant.",
    "reasoning_ai": "You are a reasoning AI.",
    "friendly_ai": "You are a friendly AI.",
}

assert len(SYSTEM_PROMPTS) == 48, f"Expected 48 prompts, got {len(SYSTEM_PROMPTS)}"

LAYERS = list(range(28))  # all 28 Qwen2.5-7B-Instruct transformer layers
# Co-primary (per plan §1, §13): L15 (cross-experiment comparability) and L12 (#274
# empirical |ρ|-max). H_consistent achievable at L15 OR L12 under the same gate set.
PRIMARY_LAYER = 15
CO_PRIMARY_LAYER = 12
CO_PRIMARY_LAYERS = (PRIMARY_LAYER, CO_PRIMARY_LAYER)
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


def power_simulation(n_reps: int = 10_000, n: int = 48, seed: int = 42) -> dict:
    """Monte-Carlo power simulation for the cosine→rate Spearman test at N=48.

    Plan §1 power calc — recalibrated to the post-#274 reality. ρ_pop anchors:
      strong:     -0.52   (#274 empirical L15 ρ)
      attenuated: -0.40   (borderline)
      collapsed:  -0.30   (kill criterion)

    Threshold at n=48 (corrected from earlier draft):
      Bonferroni-28 threshold: |ρ|>=0.439 (NOT 0.611, which is the n=24 threshold).
      Raw α=0.05 threshold:   |ρ|>=0.284
      Holdout-only n=24 threshold (for split-Spearman gate): |ρ|>=0.404
    """
    rng = np.random.default_rng(seed)
    out = {"n_reps": n_reps, "n": n, "regimes": {}}

    for label, rho_pop in [
        ("strong", -0.52),
        ("attenuated", -0.40),
        ("collapsed", -0.30),
    ]:
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
            "p_abs_rho_gt_0_439_bonferroni28_n48": float(np.mean(abs_rho > 0.439)),
            "p_abs_rho_gt_0_284_raw_alpha_05_n48": float(np.mean(abs_rho > 0.284)),
        }

    # Holdout-only Spearman power at n=24 (Gate A): same ρ_pop anchors
    out["holdout_n24"] = {}
    for label, rho_pop in [("strong", -0.52), ("attenuated", -0.40), ("collapsed", -0.30)]:
        cov = np.array([[1.0, rho_pop], [rho_pop, 1.0]])
        chol = np.linalg.cholesky(cov)
        rho_samples = np.empty(n_reps)
        for i in range(n_reps):
            z = rng.standard_normal((24, 2))
            xy = z @ chol.T
            rho_samples[i], _ = stats.spearmanr(xy[:, 0], xy[:, 1])
        out["holdout_n24"][label] = {
            "rho_pop": rho_pop,
            "p_abs_rho_gt_0_404_n24": float(np.mean(np.abs(rho_samples) > 0.404)),
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
    """Load post-LoRA [ZLT] source rates for all 48 personas.

    Reads from EVAL_DIR/marker_<persona>_asst_excluded_medium_seed42/run_result.json.
    Uses results.marker.source_rate when populated; falls back to
    results.marker.all_personas[<eval_key>].
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
                sr = all_p.get(p)
        if sr is None:
            log(f"  WARNING: source_rate=None for {p} (post-fix runs should not hit this)")
            continue
        rates[p] = float(sr)
    return rates


def load_base_rates() -> dict[str, float]:
    """Load base-model (no-LoRA) source rates for the N=48 baseline.

    Searches in priority order:
      1) eval_results/issue_296/base_baseline.json (full N=48; --all-296 mode)
      2) merge of issue_296/base_baseline_new24.json + issue_274/base_baseline.json
         (the --new-only #296 path; combine to N=48)
      3) eval_results/issue_274/base_baseline.json alone (N=24; missing 24 NEW)

    Returns {} only if no source is found.
    """
    candidates = [
        ("full_n48", OUTPUT_DIR / "base_baseline.json"),
        ("new_only_n24", OUTPUT_DIR / "base_baseline_new24.json"),
        ("inherited_n24", PARENT_OUTPUT_DIR / "base_baseline.json"),
    ]

    found = {tag: path for tag, path in candidates if path.exists()}
    if not found:
        log("  No base baseline JSON found — falling back to raw rates")
        return {}

    if "full_n48" in found:
        sources = [("full_n48", found["full_n48"])]
    elif "new_only_n24" in found and "inherited_n24" in found:
        sources = [
            ("new_only_n24", found["new_only_n24"]),
            ("inherited_n24", found["inherited_n24"]),
        ]
    elif "inherited_n24" in found:
        sources = [("inherited_n24", found["inherited_n24"])]
        log("  WARNING: only N=24 inherited base baseline found; 24 NEW personas missing")
    else:
        sources = [(t, p) for t, p in found.items()]

    rates: dict[str, float] = {}
    for tag, path in sources:
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", {})
        if "marker" in results and "all_personas" in results["marker"]:
            this_rates = dict(results["marker"]["all_personas"])
        else:
            this_rates = {
                p: r["rate"] for p, r in results.items() if isinstance(r, dict) and "rate" in r
            }
        # Alias "assistant" eval-key back to "helpful_assistant" persona name
        if "assistant" in this_rates and "helpful_assistant" not in this_rates:
            this_rates["helpful_assistant"] = this_rates["assistant"]
        for p, v in this_rates.items():
            # First-encountered wins (full_n48 > new_only_n24 > inherited_n24).
            if p not in rates:
                rates[p] = float(v)
        log(f"  Loaded base rates from {tag} ({path.name}): {len(this_rates)} entries")
    return rates


def load_offdiagonal_rates() -> dict[tuple[str, str], float]:
    """Load every (source, eval_persona) cell from the 48 run_result.json files.

    Returns a {(source, eval_persona): rate} dict. The on-diagonal cell (source=eval)
    is the source rate; the 2256 off-diagonal cells (48² - 48) are bystander leakage rates.
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
            ev_canon = "helpful_assistant" if eval_persona == "assistant" else eval_persona
            if ev_canon in ALL_PERSONAS:
                cells[(src, ev_canon)] = float(rate)
    return cells


def load_n24_source_rates_from_274() -> dict[str, float]:
    """Load the 24 inherited source rates from #274's regression_results.json.

    Used by the 24→48 sign-test diagnostic. Returns {} if the parent JSON is missing
    (in which case the diagnostic is skipped with a warning).
    """
    parent_json = PARENT_OUTPUT_DIR / "regression_results.json"
    if not parent_json.exists():
        log(f"  WARNING: parent {parent_json} not found — skipping 24→48 sign-test")
        return {}
    with open(parent_json) as f:
        data = json.load(f)
    rates = data.get("source_rates", {})
    if not rates:
        log("  WARNING: parent regression_results.json missing 'source_rates'")
        return {}
    return {p: float(v) for p, v in rates.items()}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ── Centroid extraction (28 layers × 48 personas) ────────────────────────────


def extract_centroids_gpu():
    """Extract base-model centroids at all 28 layers for all 48 personas (requires GPU).

    Saves to CENTROID_DIR/centroids_n48_layers0_27.pt as a STANDALONE DELIVERABLE
    (~20 MB). Reusable downstream regardless of regression outcome.
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
    out_path = CENTROID_DIR / "centroids_n48_layers0_27.pt"
    torch.save(centroids, out_path)
    log(f"Saved centroids to {out_path}")
    return centroids


def compute_cosines_to_assistant(centroids) -> dict[int, dict[str, float]]:
    """Per-layer mean-centered cosine of each persona to helpful_assistant.

    Mean-centers across the N=48 set at each layer (plan §3e step 3).
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


# ── Regression + slope test ──────────────────────────────────────────────────


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


def linregress_with_se(x, y) -> dict:
    """OLS Pearson regression returning slope, intercept, r, p, SE(slope)."""
    n = len(x)
    if n < 3:
        return {"slope": np.nan, "intercept": np.nan, "r": np.nan, "p": np.nan, "se_slope": np.nan}
    res = stats.linregress(x, y)
    # scipy.stats.linregress returns LinregressResult with .stderr (SE of slope)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r": float(res.rvalue),
        "p": float(res.pvalue),
        "se_slope": float(res.stderr),
    }


def calibration_holdout_slope_test(x_cal, y_cal, x_hold, y_hold) -> dict:
    """Compare two independent Pearson slopes (calibration vs holdout).

    Plan §1 Gate B: |β_hold − β_cal| / SE(β_diff) < 1.96 AND
                   β_hold within ±25% of β_cal.

    SE(β_diff) = sqrt(SE(β_cal)^2 + SE(β_hold)^2)  (independence assumption)

    Returns dict with both check bools and the underlying numbers.
    """
    cal = linregress_with_se(x_cal, y_cal)
    hold = linregress_with_se(x_hold, y_hold)
    se_diff = float(np.sqrt(cal["se_slope"] ** 2 + hold["se_slope"] ** 2))
    diff = float(hold["slope"] - cal["slope"])
    z_diff = float(diff / se_diff) if se_diff > 0 else float("nan")
    pct_dev = float(abs(diff) / abs(cal["slope"])) if abs(cal["slope"]) > 0 else float("nan")
    return {
        "calibration": cal,
        "holdout": hold,
        "diff": diff,
        "se_diff": se_diff,
        "z_diff": z_diff,
        "abs_z_diff": float(abs(z_diff)),
        "pct_deviation": pct_dev,
        "passes_z_lt_1p96": bool(abs(z_diff) < 1.96) if not np.isnan(z_diff) else False,
        "passes_within_25pct": bool(pct_dev < 0.25) if not np.isnan(pct_dev) else False,
        "passes_gate_B": bool(
            (not np.isnan(z_diff))
            and (not np.isnan(pct_dev))
            and abs(z_diff) < 1.96
            and pct_dev < 0.25
        ),
    }


# ── Steiger Z₁ (dependent correlations sharing one variable) ─────────────────


def steiger_z1(rxy, rxz, ryz, n, two_tailed=True):
    """Steiger's Z₁ for two dependent correlations sharing one variable.

    Tests H0: r(x,y) == r(x,z) given r(y,z) (the predictor-predictor correlation).
    Implementation: Steiger (1980) Psychological Bulletin 87:245, eqn 12; matches
    psinger/CorrelationStats::dependent_corr.

    Args:
        rxy: correlation r(x, y)
        rxz: correlation r(x, z)
        ryz: correlation r(y, z) — the dependence between predictors
        n: sample size
        two_tailed: if True, return two-tailed p; else one-tailed (rxy > rxz)

    Returns:
        (z, p) tuple. z is the Steiger Z₁ statistic; p is the p-value.
    """
    # Clip to (-1, 1) to avoid divide-by-zero in atanh
    eps = 1e-10
    rxy = float(np.clip(rxy, -1.0 + eps, 1.0 - eps))
    rxz = float(np.clip(rxz, -1.0 + eps, 1.0 - eps))
    ryz = float(np.clip(ryz, -1.0 + eps, 1.0 - eps))

    # Fisher Z transforms
    zxy = 0.5 * np.log((1 + rxy) / (1 - rxy))
    zxz = 0.5 * np.log((1 + rxz) / (1 - rxz))
    # Average of squared correlations (eqn 12)
    rm2 = (rxy**2 + rxz**2) / 2
    f = (1 - ryz) / (2 * (1 - rm2))
    h = (1 - f * rm2) / (1 - rm2)
    denom = np.sqrt(2 * (1 - ryz) * h / (n - 3))
    if denom <= 0:
        return float("nan"), float("nan")
    z = (zxy - zxz) / denom
    p = 2 * (1 - stats.norm.cdf(abs(z))) if two_tailed else 1 - stats.norm.cdf(z)
    return float(z), float(p)


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
    """Partial Spearman with multivariate confounds: residualize on rank-Z columns.

    Falls back to pure-numpy rank-residualize-then-Spearman if pingouin is unavailable
    or fails. Per plan §1: pingouin's inverse-covariance method matches ppcor R; the
    rank-residualize fallback can differ by ~0.02 — reported alongside if both run.
    """
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    Z = np.asarray(Z, dtype=float)
    rZ = np.column_stack([stats.rankdata(Z[:, j]) for j in range(Z.shape[1])])
    rZ_aug = np.column_stack([np.ones(len(rx)), rZ])
    beta_x, *_ = np.linalg.lstsq(rZ_aug, rx, rcond=None)
    beta_y, *_ = np.linalg.lstsq(rZ_aug, ry, rcond=None)
    res_x = rx - rZ_aug @ beta_x
    res_y = ry - rZ_aug @ beta_y
    rho, p = stats.spearmanr(res_x, res_y)
    return float(rho), float(p)


def partial_spearman_pingouin(x, y, covars: dict[str, np.ndarray]):
    """Partial Spearman using pingouin.partial_corr(method='spearman').

    Returns (rho, p) or (None, None) if pingouin is unavailable. covars is a dict
    {name: array} of confound columns.
    """
    try:
        import pandas as pd
        import pingouin as pg
    except ImportError:
        return None, None
    df = pd.DataFrame({"x": x, "y": y, **covars})
    try:
        result = pg.partial_corr(
            data=df, x="x", y="y", covar=list(covars.keys()), method="spearman"
        )
    except Exception as exc:
        log(f"  pingouin.partial_corr failed: {exc} — using rank-residualize fallback")
        return None, None
    rho = float(result["r"].iloc[0])
    p = float(result["p-val"].iloc[0])
    return rho, p


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


def bpe_jaccard(a: str, b: str, tokenizer) -> float:
    """BPE-Jaccard over Qwen2.5-7B tokens (NEW for #296, plan §1 + Critic 3 #1).

    Tokenizes both strings with the Qwen2.5-7B tokenizer, then computes Jaccard over
    the resulting BPE-token *sets* (not multisets — this is intentional, mirrors the
    whitespace-Jaccard baseline). The most plausible Qwen-specific surface confound
    for new "ai_*" / "*_assistant" personas: they share BPE tokens like "assistant"/
    "ai" with the helpful-assistant prompt.
    """
    sa = set(tokenizer.encode(a, add_special_tokens=False))
    sb = set(tokenizer.encode(b, add_special_tokens=False))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


# ── Pretraining-frequency proxy ──────────────────────────────────────────────


def compute_pretraining_frequencies() -> dict[str, float]:
    """Compute persona-name BPE unigram log-frequency proxy (plan §1 Critic 3 #3).

    For each of 48 persona-name BPE tokens, look up unigram frequency in a reference
    corpus. Plan §3e step 13 specifies a fallback chain:
      1. the_pile-uncopyrighted unigram dump (preferred)
      2. C4 unigram counts
      3. wikitext-103 (last resort)

    Caches result at OUTPUT_DIR/persona_pretraining_freq.json. Returns
    {persona: log_frequency_of_max_freq_BPE_token_in_persona_name}.

    PLAN-DEVIATION (allowed by plan §11 "Adjusting the pretraining-frequency
    reference corpus"): we **deliberately skip** the_pile-uncopyrighted and C4 and
    go straight to wikitext-103. Reasons:
      - the_pile-uncopyrighted is hosted on a non-default HF Hub repo and has had
        gating + 401 issues across pods historically; the analyzer must not block
        on hub-availability or auth refresh.
      - C4 is a 305 GB streamed download; a single-pass unigram-count over even a
        sampled prefix is cost-prohibitive relative to the covariate's load-bearing
        role (this is a *covariate*, not the headline test — the goal is a usable
        rank-ordering across 48 persona names, not absolute pretraining frequency).
      - wikitext-103 streams cleanly with no auth and produces a stable unigram
        rank-ordering for the 48 persona-name BPE tokens we care about.

    If wikitext-103 itself fails (HF datasets unreachable, etc.), the function
    falls back to a uniform-count proxy and logs a warning; the multi-covariate
    partial that consumes this column then collapses to the multi-covariate
    without it (i.e., the analysis still runs but loses the pretraining-frequency
    confound).
    """
    cache_path = OUTPUT_DIR / "persona_pretraining_freq.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if "freq_by_persona" in cached and len(cached["freq_by_persona"]) == 48:
            log(f"  Loaded cached pretraining freqs from {cache_path}")
            return {p: float(v) for p, v in cached["freq_by_persona"].items()}

    from collections import Counter

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Plan-deviation note: we skip the_pile-uncopyrighted and C4 (per plan §11
    # allowance) and go straight to wikitext-103. See module-level docstring for
    # rationale. If wikitext-103 itself fails, fall back to a uniform unit count
    # (every token gets log(1/V) = -log(V), making the covariate a constant — the
    # analyzer logs a warning but still ships a JSON so downstream code does not
    # crash).
    counter: Counter | None = None
    source_used = "fallback_uniform"
    n_tokens_seen = 0
    try:
        from datasets import load_dataset

        log("  Trying wikitext-103 (load_dataset, train split)...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        counter = Counter()
        # Sample first 50k lines; sufficient for rank-ordering 48 persona names
        for i, ex in enumerate(ds):
            if i >= 50_000:
                break
            txt = ex.get("text", "") or ""
            if not txt.strip():
                continue
            ids = tok.encode(txt, add_special_tokens=False)
            counter.update(ids)
            n_tokens_seen += len(ids)
        source_used = "wikitext-103-raw-v1[:50k_lines]"
        log(f"    wikitext-103 unigram counter built: {n_tokens_seen} tokens")
    except Exception as exc:
        log(f"  wikitext-103 unigram build failed: {exc}")
        counter = None

    freq_by_persona: dict[str, float] = {}
    eps = 1.0  # add-1 smoothing on the count
    if counter is None or n_tokens_seen == 0:
        log(
            "  WARNING: no reference corpus available — writing uniform-fallback "
            "log-frequency (every persona gets the same value). Multi-covariate partial "
            "with this column collapses to the multi-covariate without it."
        )
        # Vocab size approximation: tokenizer's vocab_size attribute
        V = tok.vocab_size if hasattr(tok, "vocab_size") else 152_064
        log_uniform = float(np.log(eps / (V * eps)))
        freq_by_persona = {p: log_uniform for p in ALL_PERSONAS}
    else:
        total = float(n_tokens_seen + eps * len(counter))
        for p in ALL_PERSONAS:
            # Tokenize the persona NAME (not the full prompt) to match the predictor:
            # the question is whether the persona NAME's tokens are common in pretraining.
            ids = tok.encode(p, add_special_tokens=False)
            if not ids:
                freq_by_persona[p] = float(np.log(eps / total))
                continue
            # Use max log-frequency token in the persona name (most-common BPE).
            log_freqs = [np.log((counter.get(tid, 0) + eps) / total) for tid in ids]
            freq_by_persona[p] = float(max(log_freqs))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "source_used": source_used,
        "n_tokens_seen": n_tokens_seen,
        "tokenizer": BASE_MODEL,
        "freq_by_persona": freq_by_persona,
        "interpretation": (
            "log-frequency of max-frequency BPE token in each persona NAME. "
            "Higher = more common in pretraining."
        ),
    }
    with open(cache_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"  Saved pretraining freqs to {cache_path} (source={source_used})")
    return freq_by_persona


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
    """Deterministic LOOCV: 48 fits per layer, no fold-seed dependence."""
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


# ── Splits A and B (calibration / holdout) ───────────────────────────────────


def split_A_inheritance_aligned() -> tuple[list[str], list[str]]:
    """Split A: calibration = 24 inherited (#274), holdout = 24 new (#296).

    Plan §1 split A; preserves category balance (10/8/6 each side).
    """
    return list(INHERITED_PERSONAS_24), list(NEW_PERSONAS_24)


def split_B_random_stratified(seed: int = 42) -> tuple[list[str], list[str]]:
    """Split B (CO-PRIMARY): 24 cal / 24 hold, random stratified by category, seed=42.

    Plan §1 split B; per category, draw N/2 personas without replacement for calibration,
    rest for holdout. The 12 generic_helper personas split 6/6, the 16 character split 8/8,
    the 20 occupational split 10/10.
    """
    rng = np.random.RandomState(seed)
    cal: list[str] = []
    hold: list[str] = []
    by_cat: dict[str, list[str]] = {"occupational": [], "character": [], "generic_helper": []}
    for p in ALL_PERSONAS:
        by_cat[CATEGORIES[p]].append(p)
    for _cat, members in by_cat.items():
        members_sorted = sorted(members)  # deterministic input order before shuffle
        idx = np.arange(len(members_sorted))
        rng.shuffle(idx)
        half = len(members_sorted) // 2
        cal.extend([members_sorted[i] for i in idx[:half]])
        hold.extend([members_sorted[i] for i in idx[half:]])
    return cal, hold


def split_analysis(
    cal_keys: list[str],
    hold_keys: list[str],
    cosines_l: dict[str, float],
    rates: dict[str, float],
    label: str,
) -> dict:
    """Run Gate A (holdout-only Spearman) + Gate B (cal-vs-hold slope test) for one split."""
    cal = [p for p in cal_keys if p in rates and p in cosines_l]
    hold = [p for p in hold_keys if p in rates and p in cosines_l]
    if len(cal) < 4 or len(hold) < 4:
        return {
            "label": label,
            "n_cal": len(cal),
            "n_hold": len(hold),
            "skipped": "fewer than 4 personas in cal or hold",
        }
    x_cal = np.array([cosines_l[p] for p in cal])
    y_cal = np.array([rates[p] for p in cal])
    x_hold = np.array([cosines_l[p] for p in hold])
    y_hold = np.array([rates[p] for p in hold])

    rho_hold, p_hold = stats.spearmanr(x_hold, y_hold)
    gate_A_threshold = (
        0.404
        if len(hold) == 24
        else stats.t.ppf(0.975, len(hold) - 2)
        / np.sqrt(len(hold) - 2 + stats.t.ppf(0.975, len(hold) - 2) ** 2)
    )
    gate_A = bool(abs(rho_hold) > gate_A_threshold and p_hold < 0.05)

    slope_test = calibration_holdout_slope_test(x_cal, y_cal, x_hold, y_hold)

    # Descriptive: PI-coverage of holdout under cal-fit (kept as descriptive only)
    y_pred_hold, pi_half_hold, _reg_cal = pearson_prediction_interval(x_cal, y_cal, x_hold)
    pi_inside = [bool(abs(y_hold[i] - y_pred_hold[i]) <= pi_half_hold[i]) for i in range(len(hold))]
    pi_count = sum(pi_inside)

    return {
        "label": label,
        "n_cal": len(cal),
        "n_hold": len(hold),
        "calibration_personas": cal,
        "holdout_personas": hold,
        "holdout_spearman": {
            "rho": float(rho_hold),
            "p": float(p_hold),
            "gate_A_threshold": gate_A_threshold,
            "passes_gate_A": gate_A,
        },
        "slope_test": slope_test,
        "passes_gate_B": slope_test["passes_gate_B"],
        "pi_coverage_descriptive": {
            "n_inside": int(pi_count),
            "n_total": len(hold),
            "personas_inside": [hold[i] for i in range(len(hold)) if pi_inside[i]],
        },
        "passes_AB": bool(gate_A and slope_test["passes_gate_B"]),
    }


# ── 24→48 sign-test diagnostic ───────────────────────────────────────────────


def sign_test_24_to_48(rates_n48: dict[str, float], rates_n24: dict[str, float]) -> dict:
    """Plan §1 (NEW); auto-LOW trigger if ≥18/24 inherited personas drop.

    For each of the 24 inherited personas, compute delta = rate_n48 - rate_n24.
    Run binom_test(n_drops, 24, 0.5, alternative='greater').

    Pre-registered consequences:
      ≥18/24 (p=0.0113) -> auto-LOW headline confidence regardless of bucket
      ≥20/24 (p=7.72e-4) -> matches #294 12/12 magnitude (stronger drift signal)

    Returns dict with per-persona deltas, n_drops, and diagnostic verdict.
    """
    if not rates_n24:
        return {"available": False, "reason": "no n=24 rates loaded from #274"}

    deltas: dict[str, float] = {}
    for p in INHERITED_PERSONAS_24:
        if p in rates_n48 and p in rates_n24:
            deltas[p] = float(rates_n48[p] - rates_n24[p])

    if len(deltas) < 24:
        log(
            f"  WARNING: 24→48 sign-test only has {len(deltas)}/24 personas with both "
            "n=24 and n=48 rates available"
        )

    n_drops = int(sum(1 for d in deltas.values() if d < 0))
    n_total = len(deltas)
    if n_total == 0:
        return {"available": False, "reason": "no overlapping personas"}

    binom_result = stats.binomtest(n_drops, n_total, p=0.5, alternative="greater")
    binom_p = float(binom_result.pvalue)

    threshold_low = 18 if n_total == 24 else int(np.ceil(0.75 * n_total))
    threshold_match294 = 20 if n_total == 24 else int(np.ceil(0.83 * n_total))

    return {
        "available": True,
        "n_total": n_total,
        "n_drops": n_drops,
        "n_no_change": int(sum(1 for d in deltas.values() if d == 0)),
        "n_increases": int(sum(1 for d in deltas.values() if d > 0)),
        "deltas": {p: round(v, 4) for p, v in deltas.items()},
        "mean_delta": float(np.mean(list(deltas.values()))),
        "binomial_p_one_sided_drops": binom_p,
        "auto_low_threshold_at_n24": threshold_low,
        "match_294_threshold_at_n24": threshold_match294,
        "auto_low_triggered": bool(n_drops >= threshold_low),
        "matches_294_magnitude": bool(n_drops >= threshold_match294),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_hero_l15(cosines_l, rates, output_subpath: str, layer_label: int = 15):
    """Hero figure: scatter at L<layer_label>, calibration vs holdout (split A) coloring."""
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

    # Cal = inherited 24 (filled), Hold = new 24 (open).
    cal_ps, hold_ps = split_A_inheritance_aligned()
    x_cal = np.array([cosines_l[p] for p in cal_ps if p in rates])
    y_cal = np.array([rates[p] for p in cal_ps if p in rates])
    if len(x_cal) >= 3:
        x_range = np.linspace(min(x_cal) - 0.1, max(x_cal) + 0.1, 200)
        y_pred_band, pi_half_band, reg = pearson_prediction_interval(x_cal, y_cal, x_range)
        ax.fill_between(
            x_range,
            (y_pred_band - pi_half_band) * 100,
            (y_pred_band + pi_half_band) * 100,
            alpha=0.15,
            color="gray",
            label="95% PI (split A cal fit, n=24)",
        )
        ax.plot(x_range, y_pred_band * 100, "--", color="gray", linewidth=0.8)
    else:
        reg = None

    for p in ALL_PERSONAS:
        if p not in rates or p not in cosines_l:
            continue
        cat = CATEGORIES[p]
        is_new = p in NEW_PERSONAS_24
        ax.scatter(
            cosines_l[p],
            rates[p] * 100,
            color=cat_colors[cat],
            marker=cat_markers[cat],
            s=110 if is_new else 50,
            zorder=5,
            edgecolors="black" if is_new else "none",
            linewidths=1.2 if is_new else 0,
            facecolors="none" if is_new else cat_colors[cat],
        )

    ax.set_xlabel(f"Centered cosine to assistant (L{layer_label})")
    ax.set_ylabel("[ZLT] source rate (%)")
    if reg is not None:
        ax.set_title(
            f"L{layer_label}: split A cal-fit Pearson r={reg['r']:.2f}, p={reg['p']:.4f} "
            f"(n_cal={len(x_cal)}, n_hold={len(hold_ps)})"
        )
    else:
        ax.set_title(f"L{layer_label}: insufficient calibration data")
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
    holm_label_done = False
    for i, layer in enumerate(LAYERS):
        if holm_rej[i]:
            ax.scatter(
                [layer],
                [rhos[i]],
                color="red",
                s=60,
                zorder=10,
                label=None if holm_label_done else "Holm-survivor",
            )
            holm_label_done = True
    ax.axvline(
        PRIMARY_LAYER, color="blue", linestyle=":", alpha=0.5, label=f"L{PRIMARY_LAYER} (primary)"
    )
    ax.axvline(
        CO_PRIMARY_LAYER,
        color="purple",
        linestyle=":",
        alpha=0.5,
        label=f"L{CO_PRIMARY_LAYER} (co-primary)",
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ (cosine vs source rate)")
    ax.set_title("28-layer scan, N=48, Holm-Bonferroni-28 highlighted")
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
        label="LOOCV (deterministic, n=48)",
    )

    argmin_rep = min(repeated_mse, key=repeated_mse.get)
    argmin_loo = min(loocv_mse, key=loocv_mse.get)
    ax.axvline(PRIMARY_LAYER, color="blue", linestyle=":", alpha=0.5, label=f"L{PRIMARY_LAYER}")
    ax.axvline(
        CO_PRIMARY_LAYER, color="purple", linestyle=":", alpha=0.5, label=f"L{CO_PRIMARY_LAYER}"
    )
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
    ax.set_title("CV held-out MSE by layer (N=48)")
    ax.legend(loc="best", fontsize=7)

    plt.tight_layout()
    savefig_paper(fig, output_subpath, dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)
    log(f"Saved CV-MSE figure to figures/{output_subpath}")


def plot_within_category_l15(
    within_cat: dict,
    output_subpath: str,
    layers_to_show: tuple[int, ...] = (15, 12),
):
    """Within-category Spearman bars at L15 and L12.

    Plan §3e step 15 figure: bar plot of within-category Spearman ρ for the three
    categories (occupational n=20, character n=16, generic_helper n=12) at L15 and
    L12, with raw α=0.05 |ρ| critical thresholds (from Spearman tables for the
    respective n) drawn as dashed horizontal lines:
        n=20 -> |ρ|>0.444   (occupational)
        n=16 -> |ρ|>0.497   (character)
        n=12 -> |ρ|>0.576   (generic_helper)
    These thresholds are descriptive (within-category power is logged at n=20/16/12
    in §1; the headline 3-gate test uses the n=20 occupational gate at 0.444).
    """
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    cats = ("occupational", "character", "generic_helper")
    raw_thresh = {"occupational": 0.444, "character": 0.497, "generic_helper": 0.576}
    colors = paper_palette(3)
    cat_colors = dict(zip(cats, colors, strict=True))

    fig, axes = plt.subplots(1, len(layers_to_show), figsize=(5.0 * len(layers_to_show), 4.0))
    if len(layers_to_show) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers_to_show, strict=True):
        layer_key = f"layer_{layer}"
        layer_blob = within_cat.get(layer_key, {})
        rhos = []
        labels = []
        bar_colors = []
        for cat in cats:
            entry = layer_blob.get(cat, {})
            rho = entry.get("spearman_rho")
            n_cat = entry.get("n", 0)
            if rho is None:
                continue
            rhos.append(rho)
            labels.append(f"{cat}\n(n={n_cat})")
            bar_colors.append(cat_colors[cat])

        xs = np.arange(len(rhos))
        ax.bar(xs, rhos, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, color="black", linewidth=0.6)

        for i, cat in enumerate(cats):
            if i >= len(rhos):
                continue
            thr = raw_thresh[cat]
            ax.hlines(
                [thr, -thr],
                xmin=i - 0.4,
                xmax=i + 0.4,
                colors="gray",
                linestyles="--",
                linewidth=0.7,
            )

        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("Spearman ρ (cosine vs source rate)")
        ax.set_title(f"L{layer}: within-category fits (raw α=0.05 thresholds dashed)")

    plt.tight_layout()
    savefig_paper(fig, output_subpath, dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)
    log(f"Saved within-category figure to figures/{output_subpath}")


def plot_offdiagonal_l15(
    off_diag_meta: list[dict],
    rates: dict[str, float],
    output_subpath: str,
    *,
    strong_threshold: float = 0.30,
    layer_label: int = 15,
):
    """Off-diagonal cosine vs bystander-rate scatter at L15 with strong/weak split.

    Plan §3e step 15 figure: scatter of off-diagonal cosine vs bystander rate
    (n≤2256), colored by whether the SOURCE persona is a "strong emitter" (diagonal
    rate > 30 %) or a "weak emitter" (diagonal rate ≤ 30 %). Per-subset Spearman ρ
    annotated.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    palette = paper_palette(2)
    color_strong, color_weak = palette[0], palette[1]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    strong_pts = [m for m in off_diag_meta if rates.get(m["src"], 0.0) > strong_threshold]
    weak_pts = [m for m in off_diag_meta if rates.get(m["src"], 0.0) <= strong_threshold]

    def _scatter(group: list[dict], color: str, label: str) -> str:
        if not group:
            return f"{label}: n=0"
        xs = np.array([m["cos"] for m in group])
        ys = np.array([m["rate"] for m in group])
        ax.scatter(xs, ys * 100, color=color, alpha=0.45, s=18, edgecolors="none", label=label)
        if len(group) >= 5:
            rho, p = stats.spearmanr(xs, ys)
            return f"{label}: n={len(group)}, ρ={rho:.2f}, p={p:.3g}"
        return f"{label}: n={len(group)} (too few for Spearman)"

    strong_summary = _scatter(
        strong_pts, color_strong, f"strong emitter (diag > {strong_threshold:.0%})"
    )
    weak_summary = _scatter(weak_pts, color_weak, f"weak emitter (diag ≤ {strong_threshold:.0%})")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"Centered cosine (source -> eval persona) at L{layer_label}")
    ax.set_ylabel("Bystander source rate (%)")
    ax.set_title(
        f"Off-diagonal cosine vs bystander rate at L{layer_label} "
        f"(n={len(off_diag_meta)})\n{strong_summary} | {weak_summary}",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    savefig_paper(fig, output_subpath, dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)
    log(f"Saved off-diagonal figure to figures/{output_subpath}")


def plot_string_similarity_baseline(
    cosines: dict[int, dict[str, float]],
    rates: dict[str, float],
    levs: dict[str, float],
    jaccards: dict[str, float],
    bpejaccs: dict[str, float],
    steiger_results: dict[str, dict],
    available: list[str],
    output_subpath: str,
    layers_to_show: tuple[int, ...] = (15, 12),
):
    """3 columns × 2 rows: cosine vs each string-similarity baseline at L15 and L12.

    Plan §3e step 15 figure. For each (layer, baseline) cell:
      x-axis: baseline (-Levenshtein, token-Jaccard, BPE-Jaccard)
      y-axis: source rate (%)
      Title: descriptive Steiger Z₁ for cosine-vs-baseline (from steiger_results).

    Steiger Z₁ asks "is the centered-cosine→rate Spearman distinguishable from the
    string-baseline→rate Spearman, accounting for the within-sample baseline–cosine
    correlation?" |Z|>1.96 ⇒ distinguishable at α=0.05.
    """
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    cat_colors = dict(
        zip(("occupational", "character", "generic_helper"), paper_palette(3), strict=True)
    )

    baselines = (
        (
            "neg_levenshtein",
            "−Levenshtein",
            lambda p: -levs[p],
            "steiger_z_cosine_vs_neg_levenshtein",
        ),
        (
            "token_jaccard",
            "Token Jaccard",
            lambda p: jaccards[p],
            "steiger_z_cosine_vs_token_jaccard",
        ),
        (
            "bpe_jaccard",
            "BPE Jaccard (Qwen)",
            lambda p: bpejaccs[p],
            "steiger_z_cosine_vs_bpe_jaccard",
        ),
    )

    n_rows = len(layers_to_show)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11.0, 3.6 * n_rows), squeeze=False)

    for row_i, layer in enumerate(layers_to_show):
        steiger_layer = steiger_results.get(f"layer_{layer}", {})
        for col_i, (_bkey, blabel, val_fn, steiger_key) in enumerate(baselines):
            ax = axes[row_i][col_i]
            for p in available:
                cat = CATEGORIES.get(p, "occupational")
                ax.scatter(
                    val_fn(p),
                    rates[p] * 100,
                    color=cat_colors.get(cat, "black"),
                    s=30,
                    alpha=0.85,
                    edgecolors="black",
                    linewidths=0.3,
                )
            zinfo = steiger_layer.get(steiger_key, {})
            z = zinfo.get("z")
            zp = zinfo.get("p")
            distinguishable = bool(z is not None and abs(z) > 1.96)
            title_extra = (
                f"Z={z:.2f}, p={zp:.3f}" if (z is not None and zp is not None) else "Z=n/a"
            )
            star = " *" if distinguishable else ""
            ax.set_title(f"L{layer} | {blabel}\n(Steiger {title_extra}{star})", fontsize=8)
            ax.set_xlabel(blabel)
            if col_i == 0:
                ax.set_ylabel("Source rate (%)")

    fig.suptitle(
        "String-similarity baselines vs source rate (Steiger Z₁ vs cosine descriptive; "
        "* marks |Z|>1.96)",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    savefig_paper(fig, output_subpath, dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)
    log(f"Saved string-similarity baseline figure to figures/{output_subpath}")


def plot_n24_to_n48_drift(sign_test_result: dict, output_subpath: str):
    """24→48 drift histogram with sign-test annotation."""
    if not sign_test_result.get("available"):
        return
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("neurips")
    fig, ax = plt.subplots(figsize=(8, 4))

    deltas = sign_test_result["deltas"]
    names = list(deltas.keys())
    vals = [deltas[n] * 100 for n in names]
    colors = ["red" if v < 0 else ("green" if v > 0 else "gray") for v in vals]

    ax.barh(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Δ source rate (n=48 minus n=24, percentage points)")
    ax.set_title(
        f"24→48 sign-test: {sign_test_result['n_drops']}/{sign_test_result['n_total']} drops, "
        f"binom-p={sign_test_result['binomial_p_one_sided_drops']:.4f}, "
        f"auto-LOW triggered={sign_test_result['auto_low_triggered']}"
    )

    plt.tight_layout()
    savefig_paper(fig, output_subpath, dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)
    log(f"Saved n24->n48 drift figure to figures/{output_subpath}")


# ── Main full-analysis driver ────────────────────────────────────────────────


def full_analysis():  # noqa: C901  (top-level orchestrator; intentionally sequential)
    """Full post-training analysis: power sim + centroids + regression + 3-gate test + plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Cell 0: Power simulation ─────────────────────────────────────────
    log("=== Cell 0: Power simulation (n=10000) ===")
    power = power_simulation(n_reps=10_000, n=48, seed=42)
    for label, regime in power["regimes"].items():
        log(
            f"  {label} (ρ_pop={regime['rho_pop']}): "
            f"P(|ρ|>0.439, Bonf-28)={regime['p_abs_rho_gt_0_439_bonferroni28_n48']:.3f} | "
            f"P(|ρ|>0.284, raw α=0.05)={regime['p_abs_rho_gt_0_284_raw_alpha_05_n48']:.3f}"
        )

    # ── Load source rates (post-LoRA + base + n=24 from #274 for sign-test) ──
    log("\n=== Loading source rates ===")
    rates = load_source_rates()
    available = [p for p in ALL_PERSONAS if p in rates]
    log(f"  {len(available)}/{len(ALL_PERSONAS)} personas have source rates")
    if len(available) < len(ALL_PERSONAS):
        missing = set(ALL_PERSONAS) - set(available)
        log(f"  Missing: {sorted(missing)}")

    base_rates = load_base_rates()
    if base_rates:
        log(f"  Loaded base rates for {len(base_rates)} personas")

    rates_n24 = load_n24_source_rates_from_274()
    if rates_n24:
        log(f"  Loaded {len(rates_n24)} parent (n=24) rates from #274")

    # Wilson 95% CIs (n=100 = 20 questions × 5 completions)
    cis = {p: wilson_ci(round(rates[p] * N_EVAL), N_EVAL) for p in available}

    off_diag_cells = load_offdiagonal_rates()
    log(f"  Loaded {len(off_diag_cells)} (source, eval) cells (48² − 48 = 2256 off-diagonal max)")

    log("\n=== Computing prompt token lengths ===")
    lengths = get_prompt_token_lengths()

    log("\n=== Computing pretraining-frequency proxy (cached) ===")
    pretraining_freqs = compute_pretraining_frequencies()

    log("\n=== Extracting centroids (28 layers × 48 personas) ===")
    centroids = extract_centroids_gpu()
    cosines = compute_cosines_to_assistant(centroids)

    log("\n=== Per-layer regression (28 layers, N=48) ===")
    log_lengths = np.log([lengths[p] for p in available])
    template_prefix = np.array(
        [1 if SYSTEM_PROMPTS[p].startswith("You are") else 0 for p in available],
        dtype=float,
    )

    def bucket_fn(n: int) -> int:
        if n <= 5:
            return 0
        if n <= 10:
            return 1
        if n <= 20:
            return 2
        return 3

    token_bucket = np.array([bucket_fn(lengths[p]) for p in available], dtype=float)
    pretrain_freq_arr = np.array([pretraining_freqs.get(p, 0.0) for p in available], dtype=float)

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
        # Multi-covariate {length, template, token-bucket, pretraining-freq}
        Z_multi = np.column_stack([log_lengths, template_prefix, token_bucket, pretrain_freq_arr])
        rho_partial_multi, p_partial_multi = partial_spearman_multi(x, y, Z_multi)
        # pingouin partial (preferred); reported alongside if available
        rho_pg, p_pg = partial_spearman_pingouin(
            x,
            y,
            {
                "log_length": log_lengths,
                "template_prefix": template_prefix,
                "token_bucket": token_bucket,
                "pretrain_freq": pretrain_freq_arr,
            },
        )
        loo_rs = []
        for i in range(len(available)):
            mask = np.ones(len(available), dtype=bool)
            mask[i] = False
            r_loo, _ = stats.pearsonr(x[mask], y[mask])
            loo_rs.append(float(r_loo))
        layer_stats[layer] = {
            "pearson_r": float(r_l),
            "pearson_p": float(p_l),
            "spearman_rho": float(rho_l),
            "spearman_p": float(rp_l),
            "vif_length": float(v_len),
            "partial_spearman_length": {"rho": rho_partial_len, "p": p_partial_len},
            "partial_spearman_multi": {"rho": rho_partial_multi, "p": p_partial_multi},
            "partial_spearman_pingouin": (
                {"rho": rho_pg, "p": p_pg} if rho_pg is not None else None
            ),
            "loo_pearson": {
                "min": min(loo_rs),
                "max": max(loo_rs),
                "mean": float(np.mean(loo_rs)),
            },
        }
        raw_pvals.append(float(rp_l))

    holm_pvals, holm_rej = holm_bonferroni(raw_pvals, alpha=0.05)
    for li, layer in enumerate(LAYERS):
        layer_stats[layer]["spearman_p_holm"] = float(holm_pvals[li])
        layer_stats[layer]["holm_significant"] = bool(holm_rej[li])

    rho_max_layer = max(LAYERS, key=lambda layer: abs(layer_stats[layer]["spearman_rho"]))

    # ── Splits A and B at L15 + L12 (3-gate test) ────────────────────────
    log("\n=== Splits A + B + Gate C (3-gate H_consistent test) ===")
    cal_A, hold_A = split_A_inheritance_aligned()
    cal_B, hold_B = split_B_random_stratified(seed=42)
    log("  Split A: cal=24 inherited, hold=24 new (category-stratified by construction)")
    log(f"  Split B: cal={len(cal_B)} hold={len(hold_B)}; first 5 cal: {cal_B[:5]}")

    splits: dict[str, dict] = {}
    for layer in CO_PRIMARY_LAYERS:
        cos_l = cosines[layer]
        splits[f"layer_{layer}"] = {
            "split_A_inheritance_aligned": split_analysis(cal_A, hold_A, cos_l, rates, "split_A"),
            "split_B_random_stratified": split_analysis(cal_B, hold_B, cos_l, rates, "split_B"),
        }

    # Gate C: within-occupational raw-significant Spearman
    log("\n=== Within-category fits at L15 AND L12 (Gate C from occupational) ===")
    within_cat: dict[str, dict] = {}
    for layer in CO_PRIMARY_LAYERS:
        cos_l = cosines[layer]
        within_cat[f"layer_{layer}"] = {}
        for cat_name in ["occupational", "character", "generic_helper"]:
            if cat_name == "generic_helper":
                cat_ps = [p for p in WITHIN_CAT_GENERIC if p in rates]
            else:
                cat_ps = [p for p in available if CATEGORIES[p] == cat_name]
            if len(cat_ps) >= 3:
                xc = np.array([cos_l[p] for p in cat_ps])
                yc = np.array([rates[p] for p in cat_ps])
                r_c, p_c = stats.pearsonr(xc, yc)
                rho_c, rp_c = stats.spearmanr(xc, yc)
                # Gate C threshold for n=20 (occupational): |ρ|>0.444 at α=0.05
                gate_c_thresh = 0.444 if cat_name == "occupational" and len(cat_ps) == 20 else None
                within_cat[f"layer_{layer}"][cat_name] = {
                    "n": len(cat_ps),
                    "personas": cat_ps,
                    "pearson_r": float(r_c),
                    "pearson_p": float(p_c),
                    "spearman_rho": float(rho_c),
                    "spearman_p": float(rp_c),
                    "gate_c_threshold": gate_c_thresh,
                    "passes_gate_C": (
                        bool(abs(rho_c) > 0.444 and rp_c < 0.05)
                        if cat_name == "occupational" and len(cat_ps) == 20
                        else None
                    ),
                }
                log(f"  L{layer} {cat_name} (N={len(cat_ps)}): Spearman ρ={rho_c:.3f} p={rp_c:.4f}")

        # Within-occupational with-without librarian (sensitivity per #274 §15)
        occ_ps = [p for p in available if CATEGORIES[p] == "occupational"]
        if "librarian" in occ_ps and len(occ_ps) >= 4:
            no_lib = [p for p in occ_ps if p != "librarian"]
            xn = np.array([cos_l[p] for p in no_lib])
            yn = np.array([rates[p] for p in no_lib])
            r_no, p_no = stats.pearsonr(xn, yn)
            rho_no, rp_no = stats.spearmanr(xn, yn)
            within_cat[f"layer_{layer}"]["occupational_no_librarian"] = {
                "n": len(no_lib),
                "personas": no_lib,
                "pearson_r": float(r_no),
                "pearson_p": float(p_no),
                "spearman_rho": float(rho_no),
                "spearman_p": float(rp_no),
            }

    # ── 3-gate H_consistent decision (per plan §1) ───────────────────────
    log("\n=== 3-gate H_consistent decision ===")
    h_consistent_pass = False
    h_consistent_layer = None
    for layer in CO_PRIMARY_LAYERS:
        sA = splits[f"layer_{layer}"]["split_A_inheritance_aligned"]
        sB = splits[f"layer_{layer}"]["split_B_random_stratified"]
        gate_c_layer = within_cat[f"layer_{layer}"].get("occupational", {})
        gate_c_pass = gate_c_layer.get("passes_gate_C", False)
        layer_pass = bool(sA.get("passes_AB", False) and sB.get("passes_AB", False) and gate_c_pass)
        log(
            f"  L{layer}: split_A AB={sA.get('passes_AB', False)}, "
            f"split_B AB={sB.get('passes_AB', False)}, gate_C={gate_c_pass} -> {layer_pass}"
        )
        if layer_pass and not h_consistent_pass:
            h_consistent_pass = True
            h_consistent_layer = layer

    # ── CV (repeated 5-fold + LOOCV) ─────────────────────────────────────
    log("\n=== Cross-validation (repeated 5-fold ×50 seeds + LOOCV) ===")
    repeated_mse, repeated_ci, repeated_argmin = cv_repeated_kfold(cosines, rates)
    loocv_mse, loocv_ci, loocv_argmin = cv_loocv(cosines, rates)
    log(f"  Repeated 5-fold argmin: L{repeated_argmin} (MSE={repeated_mse[repeated_argmin]:.5f})")
    log(f"  LOOCV argmin: L{loocv_argmin} (MSE={loocv_mse[loocv_argmin]:.5f})")

    # ── String-similarity baselines (DESCRIPTIVE) ────────────────────────
    log("\n=== String-similarity baselines (descriptive) + Steiger Z₁ ===")
    from transformers import AutoTokenizer

    qwen_tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    jaccards = {p: token_jaccard(SYSTEM_PROMPTS[p], ASSISTANT_PROMPT) for p in available}
    levs = {p: levenshtein(SYSTEM_PROMPTS[p], ASSISTANT_PROMPT) for p in available}
    bpejaccs = {p: bpe_jaccard(SYSTEM_PROMPTS[p], ASSISTANT_PROMPT, qwen_tok) for p in available}
    rates_arr = np.array([rates[p] for p in available])
    jaccard_arr = np.array([jaccards[p] for p in available])
    neglev_arr = np.array([-levs[p] for p in available])
    bpejacc_arr = np.array([bpejaccs[p] for p in available])

    rho_jacc, p_jacc = stats.spearmanr(jaccard_arr, rates_arr)
    rho_lev, p_lev = stats.spearmanr(neglev_arr, rates_arr)
    rho_bpe, p_bpe = stats.spearmanr(bpejacc_arr, rates_arr)

    # Steiger Z₁ at L15 AND L12 vs all 3 baselines (six Z values)
    steiger_results: dict[str, dict] = {}
    for layer in CO_PRIMARY_LAYERS:
        cos_arr = np.array([cosines[layer][p] for p in available])
        rho_cos, p_cos = stats.spearmanr(cos_arr, rates_arr)
        # Predictor-predictor Spearman correlations (the ryz term in Steiger Z₁)
        rho_cos_jacc, _ = stats.spearmanr(cos_arr, jaccard_arr)
        rho_cos_neglev, _ = stats.spearmanr(cos_arr, neglev_arr)
        rho_cos_bpe, _ = stats.spearmanr(cos_arr, bpejacc_arr)

        z_jacc, pz_jacc = steiger_z1(rho_cos, rho_jacc, rho_cos_jacc, len(available))
        z_lev, pz_lev = steiger_z1(rho_cos, rho_lev, rho_cos_neglev, len(available))
        z_bpe, pz_bpe = steiger_z1(rho_cos, rho_bpe, rho_cos_bpe, len(available))

        steiger_results[f"layer_{layer}"] = {
            "cosine_vs_rate_spearman": {"rho": float(rho_cos), "p": float(p_cos)},
            "cosine_vs_token_jaccard_spearman": float(rho_cos_jacc),
            "cosine_vs_neg_levenshtein_spearman": float(rho_cos_neglev),
            "cosine_vs_bpe_jaccard_spearman": float(rho_cos_bpe),
            "steiger_z_cosine_vs_token_jaccard": {"z": z_jacc, "p": pz_jacc},
            "steiger_z_cosine_vs_neg_levenshtein": {"z": z_lev, "p": pz_lev},
            "steiger_z_cosine_vs_bpe_jaccard": {"z": z_bpe, "p": pz_bpe},
            "distinguishable_token_jaccard": bool(abs(z_jacc) > 1.96),
            "distinguishable_neg_levenshtein": bool(abs(z_lev) > 1.96),
            "distinguishable_bpe_jaccard": bool(abs(z_bpe) > 1.96),
        }
        log(
            f"  L{layer}: cosine ρ={rho_cos:.3f} | jacc Z={z_jacc:.2f} (p={pz_jacc:.3f}) | "
            f"-lev Z={z_lev:.2f} (p={pz_lev:.3f}) | bpe Z={z_bpe:.2f} (p={pz_bpe:.3f})"
        )

    log(f"  Token-Jaccard:       ρ={rho_jacc:.3f}, p={p_jacc:.4f}")
    log(f"  -Levenshtein:        ρ={rho_lev:.3f}, p={p_lev:.4f}")
    log(f"  BPE-Jaccard (Qwen):  ρ={rho_bpe:.3f}, p={p_bpe:.4f}")

    # ── Base-model residualization ───────────────────────────────────────
    log("\n=== Base-model-residualized fit at L15 + L12 ===")
    residual_results = None
    if base_rates:
        residual_rates = {p: rates[p] - base_rates.get(p, 0.0) for p in available}
        residual_results = {}
        for layer in CO_PRIMARY_LAYERS:
            cos_l = cosines[layer]
            x_res = np.array([cos_l[p] for p in available])
            y_res = np.array([residual_rates[p] for p in available])
            r_res, p_res = stats.pearsonr(x_res, y_res)
            rho_res, rp_res = stats.spearmanr(x_res, y_res)
            residual_results[f"layer_{layer}"] = {
                "n": len(available),
                "pearson_r": float(r_res),
                "pearson_p": float(p_res),
                "spearman_rho": float(rho_res),
                "spearman_p": float(rp_res),
            }
            log(f"  L{layer} residualized: r={r_res:.3f} p={p_res:.4f} | ρ={rho_res:.3f}")
        residual_results["residual_rates"] = residual_rates

    # ── Off-diagonal cosine→bystander_rate at L15 + sub-decomp ───────────
    log("\n=== Off-diagonal cosine→rate analysis at L15 (n≤2256) ===")
    import torch
    import torch.nn.functional as F

    _allv = torch.stack([centroids[PRIMARY_LAYER][p] for p in ALL_PERSONAS]).float()
    _mu = _allv.mean(dim=0)
    off_diag_pairs = []
    off_diag_meta: list[dict] = []
    for src in available:
        for ev in available:
            if src == ev:
                continue
            cell = off_diag_cells.get((src, ev))
            if cell is None:
                continue
            v_src = centroids[PRIMARY_LAYER][src].float()
            v_ev = centroids[PRIMARY_LAYER][ev].float()
            cs = float(
                F.cosine_similarity((v_src - _mu).unsqueeze(0), (v_ev - _mu).unsqueeze(0))[0]
            )
            off_diag_pairs.append((cs, cell))
            off_diag_meta.append({"src": src, "ev": ev, "cos": cs, "rate": cell})

    if off_diag_pairs:
        xs = [c for c, _ in off_diag_pairs]
        ys = [r for _, r in off_diag_pairs]
        rho_off, p_off = stats.spearmanr(xs, ys)
        log(f"  Off-diagonal full (n={len(off_diag_pairs)}): ρ={rho_off:.3f}, p={p_off:.4g}")
    else:
        rho_off, p_off = None, None

    # Strong-emitter (>30% diagonal) vs weak-emitter (≤30%) sub-decomposition
    strong_emitters = [p for p in available if rates[p] > 0.30]
    weak_emitters = [p for p in available if rates[p] <= 0.30]
    log(
        f"  Strong-emitters (diagonal > 30%): {len(strong_emitters)} sources; "
        f"weak: {len(weak_emitters)}"
    )

    def _spearman_subset(
        meta: list[dict], src_set: set[str]
    ) -> tuple[float | None, float | None, int]:
        sub = [m for m in meta if m["src"] in src_set]
        if len(sub) < 5:
            return None, None, len(sub)
        rho, p = stats.spearmanr([m["cos"] for m in sub], [m["rate"] for m in sub])
        return float(rho), float(p), len(sub)

    rho_strong, p_strong, n_strong = _spearman_subset(off_diag_meta, set(strong_emitters))
    rho_weak, p_weak, n_weak = _spearman_subset(off_diag_meta, set(weak_emitters))
    log(f"  Off-diagonal strong-only (n={n_strong}): ρ={rho_strong}")
    log(f"  Off-diagonal weak-only   (n={n_weak}): ρ={rho_weak}")

    # ── 24→48 sign-test diagnostic (NEW) ─────────────────────────────────
    log("\n=== 24→48 sign-test diagnostic (auto-LOW trigger) ===")
    sign_test = sign_test_24_to_48(rates, rates_n24)
    if sign_test.get("available"):
        log(
            f"  n_drops={sign_test['n_drops']}/{sign_test['n_total']} | "
            f"binom-p={sign_test['binomial_p_one_sided_drops']:.4g} | "
            f"auto_low={sign_test['auto_low_triggered']} | "
            f"matches_294={sign_test['matches_294_magnitude']}"
        )
    else:
        log(f"  unavailable: {sign_test.get('reason')}")

    # ── Outcome bucket assignment (pre-registered §1, plan v2) ───────────
    bucket = "undetermined"
    flag_low_emission = False
    flag_child_safety_gating = False

    sign_l15 = np.sign(layer_stats[PRIMARY_LAYER]["spearman_rho"])
    sign_l12 = np.sign(layer_stats[CO_PRIMARY_LAYER]["spearman_rho"])

    # H_anti-correlated check (one or both co-primary layers flipped + neighbor consistency)
    neighbours_l15 = [
        np.sign(layer_stats[lay]["spearman_rho"]) for lay in [13, 14, 16, 17] if lay in layer_stats
    ]
    n_flipped_l15 = sum(1 for s in neighbours_l15 if s != sign_l15) if sign_l15 != 0 else 0
    neighbours_l12 = [
        np.sign(layer_stats[lay]["spearman_rho"]) for lay in [10, 11, 13, 14] if lay in layer_stats
    ]
    n_flipped_l12 = sum(1 for s in neighbours_l12 if s != sign_l12) if sign_l12 != 0 else 0
    l15_holm = layer_stats[PRIMARY_LAYER]["holm_significant"]
    l12_holm = layer_stats[CO_PRIMARY_LAYER]["holm_significant"]
    h_anti_correlated = bool(
        (sign_l15 == 1 and (n_flipped_l15 >= 3 or l15_holm))
        or (sign_l12 == 1 and (n_flipped_l12 >= 3 or l12_holm))
    )

    # H_attenuated: full-set raw-sig at either L15 or L12 but FAILS Holm-Bonferroni-28
    full_raw_sig_l15 = abs(layer_stats[PRIMARY_LAYER]["spearman_rho"]) > 0.284
    full_raw_sig_l12 = abs(layer_stats[CO_PRIMARY_LAYER]["spearman_rho"]) > 0.284
    no_holm_survivors = not any(layer_stats[lay]["holm_significant"] for lay in LAYERS)
    h_attenuated = bool((full_raw_sig_l15 or full_raw_sig_l12) and no_holm_survivors)

    # H_consistent_weak: gate A holds on at least one split but B or C fails
    weak_pass = False
    for layer in CO_PRIMARY_LAYERS:
        sA = splits[f"layer_{layer}"]["split_A_inheritance_aligned"]
        sB = splits[f"layer_{layer}"]["split_B_random_stratified"]
        gate_c_layer = within_cat[f"layer_{layer}"].get("occupational", {})
        gate_c_pass = gate_c_layer.get("passes_gate_C", False)
        gateA_either = sA.get("holdout_spearman", {}).get("passes_gate_A", False) or sB.get(
            "holdout_spearman", {}
        ).get("passes_gate_A", False)
        if gateA_either and not (
            sA.get("passes_AB", False) and sB.get("passes_AB", False) and gate_c_pass
        ):
            weak_pass = True

    # H_inverted: holdout-only Spearman fails raw α=0.05 on BOTH splits AND no Holm survivors
    h_inverted_layer = []
    for layer in CO_PRIMARY_LAYERS:
        sA = splits[f"layer_{layer}"]["split_A_inheritance_aligned"]
        sB = splits[f"layer_{layer}"]["split_B_random_stratified"]
        a_fails = not sA.get("holdout_spearman", {}).get("passes_gate_A", False)
        b_fails = not sB.get("holdout_spearman", {}).get("passes_gate_A", False)
        if a_fails and b_fails:
            h_inverted_layer.append(layer)
    h_inverted = bool(len(h_inverted_layer) == len(CO_PRIMARY_LAYERS) and no_holm_survivors)

    # Caveat flags
    low_em = [p for p in available if rates[p] <= 0.05]
    cats_in_low = {CATEGORIES[p] for p in low_em}
    flag_low_emission = len(low_em) >= 4 and len(cats_in_low) >= 2
    flag_child_safety_gating = (
        "child" in available
        and rates.get("child", 1.0) <= 0.05
        and len([c for c in cats_in_low if c != CATEGORIES.get("child")]) == 0
    )

    # Mutually-exclusive bucket selection (priority: anti-correlated > consistent
    # > consistent_weak > attenuated > inverted > indeterminate).
    if h_anti_correlated:
        bucket = "H_anti-correlated"
    elif h_consistent_pass:
        bucket = "H_consistent"
    elif weak_pass:
        bucket = "H_consistent_weak"
    elif h_attenuated:
        bucket = "H_attenuated"
    elif h_inverted:
        bucket = "H_inverted"
    else:
        bucket = "H_indeterminate"

    log(f"\n=== OUTCOME BUCKET: {bucket} ===")
    log(
        f"  H_consistent at layer: {h_consistent_layer} | "
        f"H_anti-correlated: {h_anti_correlated} | "
        f"H_attenuated: {h_attenuated} | H_inverted: {h_inverted}"
    )
    log(
        f"  Flags: low_emission={flag_low_emission}, child_safety_gating={flag_child_safety_gating}"
    )

    # Auto-LOW trigger: overrides bucket-derived confidence (plan §1)
    auto_low_triggered = bool(sign_test.get("auto_low_triggered", False))
    if auto_low_triggered:
        log(
            f"  AUTO-LOW TRIGGER: {sign_test['n_drops']}/{sign_test['n_total']} drops at "
            f"binom-p={sign_test['binomial_p_one_sided_drops']:.4f} — headline confidence "
            "downgraded to LOW regardless of bucket."
        )

    # ── Plots ────────────────────────────────────────────────────────────
    # Plan §3e step 15 mandates 7 figures. We deliberately do NOT wrap these in
    # try/except: per CLAUDE.md "Prefer crashing over wrong results" — a missing
    # matplotlib backend or a single bad plot must fail loudly so the smoke-test
    # catches it rather than silently shipping an incomplete figure batch.
    log("\n=== Generating plots ===")
    plot_hero_l15(cosines[PRIMARY_LAYER], rates, "issue_296/hero_l15_n48", layer_label=15)
    plot_hero_l15(cosines[CO_PRIMARY_LAYER], rates, "issue_296/hero_l12_n48", layer_label=12)
    plot_spearman_by_layer(layer_stats, "issue_296/spearman_by_layer")
    plot_cv_mse_by_layer(repeated_mse, repeated_ci, loocv_mse, "issue_296/cv_mse_by_layer")
    plot_within_category_l15(within_cat, "issue_296/within_category_l15")
    plot_offdiagonal_l15(off_diag_meta, rates, "issue_296/offdiagonal_l15", layer_label=15)
    plot_string_similarity_baseline(
        cosines,
        rates,
        levs,
        jaccards,
        bpejaccs,
        steiger_results,
        available,
        "issue_296/string_similarity_baseline",
    )
    if sign_test.get("available"):
        plot_n24_to_n48_drift(sign_test, "issue_296/n24_to_n48_drift")

    # ── Save full results JSON ───────────────────────────────────────────
    log("\n=== Saving regression_results.json ===")
    results = {
        "experiment": "issue_296",
        "model": BASE_MODEL,
        "n_personas": len(available),
        "personas": available,
        "categories": {p: CATEGORIES[p] for p in available},
        "source_rates": {p: rates[p] for p in available},
        "source_rate_cis_n100": {p: {"lower": cis[p][0], "upper": cis[p][1]} for p in available},
        "base_source_rates": {p: base_rates.get(p) for p in available},
        "prompt_token_lengths": {p: lengths[p] for p in available},
        "pretraining_freqs": {p: pretraining_freqs.get(p) for p in available},
        "power_simulation": power,
        "cosines_to_assistant": {
            f"layer_{ly}": {p: cosines[ly][p] for p in available} for ly in LAYERS
        },
        "primary_layer": PRIMARY_LAYER,
        "co_primary_layer": CO_PRIMARY_LAYER,
        "splits": splits,
        "h_consistent_pass": h_consistent_pass,
        "h_consistent_layer": h_consistent_layer,
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
            "cv_optimal_in_band": (
                repeated_argmin in (14, 15, 16) and loocv_argmin in (14, 15, 16)
            ),
        },
        "within_category": within_cat,
        "string_similarity_baselines": {
            "token_jaccard": {p: jaccards[p] for p in available},
            "levenshtein": {p: levs[p] for p in available},
            "bpe_jaccard": {p: bpejaccs[p] for p in available},
            "spearman_jaccard_vs_rate": {"rho": float(rho_jacc), "p": float(p_jacc)},
            "spearman_neg_levenshtein_vs_rate": {"rho": float(rho_lev), "p": float(p_lev)},
            "spearman_bpe_jaccard_vs_rate": {"rho": float(rho_bpe), "p": float(p_bpe)},
        },
        "steiger_z_results": steiger_results,
        "base_residualized": residual_results,
        "off_diagonal": {
            "primary_layer": PRIMARY_LAYER,
            "n_pairs_full": len(off_diag_pairs),
            "spearman_rho_full": float(rho_off) if rho_off is not None else None,
            "spearman_p_full": float(p_off) if p_off is not None else None,
            "strong_emitters": {
                "personas": strong_emitters,
                "n_pairs": n_strong,
                "spearman_rho": rho_strong,
                "spearman_p": p_strong,
            },
            "weak_emitters": {
                "personas": weak_emitters,
                "n_pairs": n_weak,
                "spearman_rho": rho_weak,
                "spearman_p": p_weak,
            },
            "post_mortem_candidates": [
                "(a) floor effects in low-similarity bystanders",
                "(b) bystander-rate calibration tied to base RLHF prior",
                "(c) source-LoRA capability degradation correlates with cosine "
                "(mediating off-diagonal)",
            ],
        },
        "n24_to_n48_sign_test": sign_test,
        "outcome_bucket": bucket,
        "outcome_flags": {
            "low_emission": flag_low_emission,
            "child_safety_gating_caveat": flag_child_safety_gating,
            "n24_to_n48_drift_triggered": auto_low_triggered,
            "matches_294_magnitude": bool(sign_test.get("matches_294_magnitude", False)),
        },
        "auto_low_triggered": auto_low_triggered,
        "elapsed_seconds": time.time() - t0,
    }

    with open(OUTPUT_DIR / "regression_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\nDone in {time.time() - t0:.0f}s")
    log(f"Results: {OUTPUT_DIR / 'regression_results.json'}")
    log(f"Centroids: {CENTROID_DIR / 'centroids_n48_layers0_27.pt'}")
    log(f"Figures: {FIG_DIR}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Issue #296 N=48 analysis")
    parser.add_argument(
        "--power-only",
        action="store_true",
        help="Run only the Cell-0 Monte-Carlo power simulation (no GPU, ~30s)",
    )
    parser.add_argument(
        "--freq-only",
        action="store_true",
        help="Run only the pretraining-frequency proxy build (no GPU, ~2 min)",
    )
    args = parser.parse_args()

    if args.power_only:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        power = power_simulation(n_reps=10_000, n=48, seed=42)
        out_path = OUTPUT_DIR / "power_simulation.json"
        with open(out_path, "w") as f:
            json.dump(power, f, indent=2)
        for label, regime in power["regimes"].items():
            log(
                f"{label} (ρ_pop={regime['rho_pop']}): "
                f"P(|ρ|>0.439, Bonf-28)={regime['p_abs_rho_gt_0_439_bonferroni28_n48']:.3f} | "
                f"P(|ρ|>0.284, raw α=0.05)={regime['p_abs_rho_gt_0_284_raw_alpha_05_n48']:.3f}"
            )
        log(f"Saved {out_path}")
        return

    if args.freq_only:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        freqs = compute_pretraining_frequencies()
        log(f"Pretraining freqs computed for {len(freqs)} personas")
        return

    full_analysis()


if __name__ == "__main__":
    main()
