# ruff: noqa: RUF001, RUF002, RUF003  # ρ, ×, ′ are standard mathematical / plan symbols
"""Issue #311 orchestrator — joint-source marker leakage along the A→B persona axis.

This is the single-entry-point script implementing the round-3 plan at
`.claude/plans/issue-311.md`. Stages are exposed as CLI sub-commands so each
can be re-run independently:

    uv run python scripts/run_issue311.py preflight     # Stage -1
    uv run python scripts/run_issue311.py extract-base  # Stage 0
    uv run python scripts/run_issue311.py pick-pair     # Stage 1 (top-1)
    uv run python scripts/run_issue311.py collin-gate   # Stage 1.5
    uv run python scripts/run_issue311.py gen-onpolicy  # Stage 2
    uv run python scripts/run_issue311.py build-data    # Stage 3
    uv run python scripts/run_issue311.py train         # Stage 4 (3 LoRAs)
    uv run python scripts/run_issue311.py post-cos-gate # Stage 4.5
    uv run python scripts/run_issue311.py eval-arm1     # Stage 5
    uv run python scripts/run_issue311.py eval-arm2     # Stage 6 (descriptive)
    uv run python scripts/run_issue311.py analyze       # Stage 7
    uv run python scripts/run_issue311.py null-shuffle  # Stage 8 (1000 perms + fixed-B)
    uv run python scripts/run_issue311.py all           # Full pipeline

Every numerical convention here is set by the approved plan. Read the plan
before changing defaults.

Statistical conventions (round-4 fixes):
- All cosines are CENTERED-CENTROID cosines (Fix 3c).
- Stage 7 partial Spearman is one-sided `alternative="less"` (Fix 2).
- Stage 8 Null A is conditioned on `s_vals` of the REAL pair (Fix 1).
- Stage 8 Null B (fixed-B) only fires when B == "comedian" (Fix 3a).
- Stage 7 register diagnostic includes a `comedy_cluster_indicator` (Fix 3b).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# We do these heavyweight imports lazily inside the stage functions so
# `--help` and pure-CPU stages don't pay torch/transformers startup cost.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Plan-locked constants ───────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN_DIM = 3584
LAYER_EXTRACT = (10, 20)
LAYER_STEER = 20
SEED = 42
MARKER = "[ZLT]"

# Stage 1 source-pair picking
EXCLUDED_SOURCES = {"helpful_assistant"}
DENSITY_THRESHOLD_FRAC = 0.5  # bystander |t| < 0.5 * t_max
DENSITY_MIN = 3  # ≥3 bystanders within the threshold
SOURCE_PAIR_DEGENERATE_COS = 0.90  # abort if BASE cos(A,B) ≥ 0.90

# Stage 1.5 collinearity gate
COLLIN_GATE_THRESH = 0.6  # |Pearson(|t|, s)| > 0.6 promotes stratified MW

# Stage 2 on-policy gen
ONPOLICY_N_PER_Q = 15
ONPOLICY_TEMP = 0.7

# Stage 3 training data
N_PER_SOURCE = 400  # both joint (per source) AND singles (CB11 / D6′)

# Stage 4 LoRA training
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_LR = 5e-6
LORA_EPOCHS = 20
LORA_BATCH_SIZE = 4
LORA_GRAD_ACCUM = 4
LORA_MAX_LENGTH = 1024
LORA_WARMUP_RATIO = 0.05

# Stage 4.5 post-train geometry gate
POST_COS_HALT = 0.97  # cos(v_A_post, v_B_post) ≥ 0.97 fires Stage 4.5 (CB3)

# Stage 5/6 eval
EVAL_K = 20
EVAL_TEMP = 1.0
EVAL_TOP_P = 0.95
EVAL_MAX_NEW_TOKENS = 2048  # CLAUDE.md late-token rule (2× trained 1024)
VLLM_GPU_MEM_UTIL = 0.60
VLLM_MAX_MODEL_LEN = 2560

# Stage 6 steering arms
STEER_COEFF = 2.0  # #267 registered headline

# Stage 7 analysis
SATURATION_THRESHOLD = 0.90  # Bernoulli union ≥ 0.90 excludes bystander
SATURATION_MAX_EXCLUSIONS = 3  # > 3 caps confidence at LOW

# Stage 7 register / discourse diagnostic
REGISTER_PEARSON_THRESH = 0.5  # length / punct / FK threshold
SEMANTIC_REGISTER_PEARSON_THRESH = 0.4  # comedy_cluster_indicator (lower, Fix 3b)
COMEDY_LEMMAS = frozenset(
    {
        "humor",
        "humour",
        "humorous",
        "comic",
        "comics",
        "comical",
        "joke",
        "jokes",
        "jokester",
        "comedy",
        "comedic",
        "funny",
        "satire",
        "satirical",
        "performer",
        "performance",
        "entertainer",
        "entertainment",
    }
)

# Stage 7/8 thresholds
P_INCONCLUSIVE_LOW = 0.05
P_INCONCLUSIVE_HIGH = 0.20
NULL_FIXED_B_PASS_PERCENTILE = 0.05  # real ρ ≤ 5th percentile of fixed-B null
NULL_RANDOM_PASS_PERCENTILE = 0.05
N_PERMS = 1000  # CB10
N_BOOTSTRAP = 1000  # cluster-bootstrap by question

# Filesystem layout
OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "issue_311"
DATA_DIR = PROJECT_ROOT / "data" / "issue_311"
LORA_OUT_DIR = OUTPUT_DIR / "lora"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_311"

# GitHub coordination
ISSUE_NUMBER = 311

# ── Logging ────────────────────────────────────────────────────────────────
logger = logging.getLogger("issue_311")


def setup_logging(stage: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)
    log_path = OUTPUT_DIR / f"stage_{stage}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("=" * 80)
    logger.info("Issue #311 — stage=%s", stage)
    logger.info("=" * 80)


# ── Helpers ────────────────────────────────────────────────────────────────


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("wrote %s", path)


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _write_jsonl(examples: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info("wrote %d JSONL examples to %s", len(examples), path)


def _count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def _save_torch(path: Path, obj: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    logger.info("saved torch obj to %s", path)


def _load_torch(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _run_metadata() -> dict:
    """Reproducibility metadata required by CLAUDE.md."""
    try:
        from explore_persona_space.metadata import get_run_metadata

        return get_run_metadata()
    except ImportError:
        return {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two 1-D vectors (assumes they are already centered)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _post_gh_marker_and_block(
    issue_number: int,
    marker_name: str,
    body_md: str,
) -> None:
    """Post a `<!-- epm:<marker_name> v1 -->` comment + label `status:blocked`.

    Uses the `gh` CLI (available on the pod / local-VM); fails loud if `gh`
    or auth is missing — this is a load-bearing signal to the orchestrator,
    not a best-effort. Per CLAUDE.md "Never silently fail".

    The marker body is wrapped between `<!-- epm:NAME v1 -->` and
    `<!-- /epm:NAME -->` to match the markers.md convention used by the
    `/issue` skill.
    """
    import subprocess

    marker_body = f"<!-- epm:{marker_name} v1 -->\n{body_md}\n<!-- /epm:{marker_name} -->\n"

    # Post the comment.
    try:
        subprocess.run(
            [
                "gh",
                "issue",
                "comment",
                str(issue_number),
                "--body",
                marker_body,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Posted epm:%s v1 marker on issue #%d", marker_name, issue_number)
    except FileNotFoundError as e:
        logger.error(
            "gh CLI not on PATH; cannot post %s marker on issue #%d: %s",
            marker_name,
            issue_number,
            e,
        )
        raise
    except subprocess.CalledProcessError as e:
        logger.error(
            "gh issue comment FAILED for issue #%d (%s): stdout=%s stderr=%s",
            issue_number,
            marker_name,
            e.stdout,
            e.stderr,
        )
        raise

    # Apply status:blocked. Use --add-label; existing `status:*` labels stay
    # (the `/issue` skill cleans up label state on resume).
    try:
        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue_number),
                "--add-label",
                "status:blocked",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Applied label status:blocked to issue #%d", issue_number)
    except subprocess.CalledProcessError as e:
        logger.error(
            "gh issue edit --add-label status:blocked FAILED for issue #%d: stdout=%s stderr=%s",
            issue_number,
            e.stdout,
            e.stderr,
        )
        raise


def _centered_to_numpy(centered: dict[str, Any], personas: Sequence[str]) -> dict[str, np.ndarray]:
    """Convert torch tensor centroids to a numpy dict in a deterministic order."""
    out: dict[str, np.ndarray] = {}
    for p in personas:
        v = centered[p]
        # Accept torch.Tensor or numpy.ndarray transparently
        try:
            arr = v.detach().cpu().numpy().astype(np.float64)
        except AttributeError:
            arr = np.asarray(v, dtype=np.float64)
        out[p] = arr
    return out


# ── Stage -1: dependency preflight ─────────────────────────────────────────


def stage_preflight(args: argparse.Namespace) -> int:
    """vLLM + transformers compatibility smoke test. HARD GATE."""
    setup_logging("-1_preflight")

    import importlib

    out: dict[str, Any] = {"stage": "-1", "metadata": _run_metadata()}

    try:
        transformers = importlib.import_module("transformers")
        out["transformers_version"] = transformers.__version__
    except ImportError as e:
        out["status"] = "FAIL"
        out["reason"] = f"transformers import failed: {e}"
        _write_json(OUTPUT_DIR / "dep_preflight.json", out)
        return 1

    try:
        vllm = importlib.import_module("vllm")
        out["vllm_version"] = vllm.__version__
    except ImportError as e:
        out["status"] = "FAIL"
        out["reason"] = f"vllm import failed: {e}"
        _write_json(OUTPUT_DIR / "dep_preflight.json", out)
        return 1

    # Pin transformers<5 if currently on 5.x (plan §4.1a). HARD GATE per plan §4.1a.
    # On transformers>=5 the vLLM 0.11 path is known-broken; rather than rely on
    # vLLM's own crash to surface this, we FAIL preflight up-front. The pod
    # operator must `uv pip install 'transformers<5'` and re-run.
    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        logger.error(
            "transformers %s >= 5.0; the plan requires <5. Run "
            "`uv pip install 'transformers<5'` on the pod and re-invoke this stage. "
            "Preflight HARD GATE (plan §4.1a).",
            transformers.__version__,
        )
        out["transformers_pin_required"] = True
        out["status"] = "FAIL"
        out["reason"] = (
            f"transformers_pin_required: installed {transformers.__version__}, plan requires <5"
        )
        _write_json(OUTPUT_DIR / "dep_preflight.json", out)
        return 1
    else:
        out["transformers_pin_required"] = False

    if args.dry_run:
        out["status"] = "DRY_RUN"
        out["dry_run"] = True
        _write_json(OUTPUT_DIR / "dep_preflight.json", out)
        return 0

    # vLLM smoke test: instantiate LLM + chat call.
    try:
        from vllm import LLM

        logger.info("Spawning vLLM smoke instance (gpu_memory_utilization=0.6)")
        llm = LLM(
            model=BASE_MODEL,
            dtype="bfloat16",
            gpu_memory_utilization=0.60,
            max_model_len=512,
            trust_remote_code=True,
        )
        outs = llm.chat([[{"role": "user", "content": "hi"}]])
        sample_text = outs[0].outputs[0].text[:50] if outs and outs[0].outputs else ""
        out["status"] = "PASS"
        out["smoke_completion_head"] = sample_text
        del llm
    except Exception as e:
        out["status"] = "FAIL"
        out["reason"] = f"vllm_smoke_failed: {type(e).__name__}: {e}"
        _write_json(OUTPUT_DIR / "dep_preflight.json", out)
        logger.error("vLLM smoke FAILED: %s", e)
        return 1

    _write_json(OUTPUT_DIR / "dep_preflight.json", out)
    logger.info("Stage -1 PASS")
    return 0


# ── Stage 0: persona-vector extraction (BASE model) ────────────────────────


def stage_extract_base(args: argparse.Namespace) -> int:
    """Extract centered-centroid vectors at L10 + L20 for all 19 personas."""
    setup_logging("0_extract_base")

    # CUDA_VISIBLE_DEVICES MUST be set BEFORE `import torch` for non-zero GPUs
    # (once torch sees the cuda device list, subsequent env changes are
    # ignored). Single-GPU default (--gpu 0) is benign either way; this is
    # belt+braces for the multi-GPU pod case. See code-review v1 Minor #3.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.steering import (
        compute_centered_centroids,
        extract_centroids_for_personas_at_layers,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS, PERSONAS_19

    out_path = OUTPUT_DIR / "centroids_base.pt"
    if out_path.exists() and not args.force:
        logger.info(
            "centroids_base.pt exists at %s; skipping (use --force to re-extract)", out_path
        )
        return 0

    logger.info("Loading base model %s on gpu %d", BASE_MODEL, args.gpu)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    logger.info(
        "Extracting centroids: %d personas x %d questions x %d layers",
        len(PERSONAS_19),
        len(EVAL_QUESTIONS),
        len(LAYER_EXTRACT),
    )
    raw_centroids = extract_centroids_for_personas_at_layers(
        model=model,
        tokenizer=tokenizer,
        layers=list(LAYER_EXTRACT),
        system_prompts=PERSONAS_19,
        questions=EVAL_QUESTIONS,
    )

    centering_set = list(PERSONAS_19.keys())
    centered: dict[int, dict[str, torch.Tensor]] = {}
    mean_vectors: dict[int, torch.Tensor] = {}
    for layer in LAYER_EXTRACT:
        centered_layer, mean_vec = compute_centered_centroids(raw_centroids[layer], centering_set)
        centered[layer] = centered_layer
        mean_vectors[layer] = mean_vec

    _save_torch(
        out_path,
        {
            "centroids_raw": raw_centroids,
            "centroids_centered": centered,
            "mean_vectors": mean_vectors,
            "personas": list(PERSONAS_19.keys()),
            "layers": list(LAYER_EXTRACT),
            "metadata": _run_metadata(),
        },
    )

    # Also dump a JSON L20 cosine matrix for human-readable verification (Fix 3c).
    centered_20_np = _centered_to_numpy(centered[20], list(PERSONAS_19.keys()))
    names = list(PERSONAS_19.keys())
    cos_matrix = []
    for i in range(len(names)):
        row = []
        for j in range(len(names)):
            row.append(_centered_cosine(centered_20_np[names[i]], centered_20_np[names[j]]))
        cos_matrix.append(row)
    _write_json(
        OUTPUT_DIR / "cosine_l20_base.json",
        {
            "layer": 20,
            "cosine_variant": "centered_centroid",
            "personas": names,
            "matrix": cos_matrix,
            "metadata": _run_metadata(),
        },
    )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# ── Stage 1: source-pair selection (TOP-1 lowest-cosine) ───────────────────


def _load_centered_l20() -> tuple[dict[str, np.ndarray], list[str]]:
    """Load centered L20 centroids as a numpy dict.

    Tries `eval_results/issue_311/centroids_base.pt` first, else falls back to
    the pre-computed cosine matrix at
    `eval_results/extraction_method_comparison/cosine_matrix_a_layer20.json`.
    The cosine-matrix fallback returns synthetic 1-D embeddings that reproduce
    the matrix's cosines (eigendecomposition); it's only used by `--dry-run`.
    """
    centroids_path = OUTPUT_DIR / "centroids_base.pt"
    if centroids_path.exists():
        data = _load_torch(centroids_path)
        # personas_order = data["personas"]
        from explore_persona_space.personas import PERSONAS_19

        names = list(PERSONAS_19.keys())
        return _centered_to_numpy(data["centroids_centered"][20], names), names

    # Fallback for --dry-run: derive centered representation from raw cosine matrix.
    matrix_path = (
        PROJECT_ROOT
        / "eval_results"
        / "extraction_method_comparison"
        / "cosine_matrix_a_layer20.json"
    )
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Neither {centroids_path} nor {matrix_path} exists. Run Stage 0 first."
        )
    logger.warning(
        "Using pre-computed RAW cosine matrix at %s as fallback. This is NOT the centered-"
        "centroid cosine (Fix 3c canonical) — use only for dry-runs.",
        matrix_path,
    )
    d = _read_json(matrix_path)
    full_names = d["persona_names"]
    mat = np.array(d["matrix"], dtype=np.float64)
    # Drop "no_persona" row/col so the indices line up with PERSONAS_19.
    keep_idx = [i for i, n in enumerate(full_names) if n != "no_persona"]
    mat = mat[np.ix_(keep_idx, keep_idx)]
    names = [full_names[i] for i in keep_idx]
    # Eigendecomposition: M = U S U^T  -> u_i = U[:, i] * sqrt(S[i, i]) gives
    # cosine(u_i, u_j) ≈ M[i, j] when M is PSD. Clip negative eigenvalues to 0.
    M = (mat + mat.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0.0)
    embed = eigvecs * np.sqrt(eigvals)  # (n, n)
    centered = {names[i]: embed[i, :] for i in range(len(names))}
    return centered, names


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson coefficient with degenerate-input guard."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def stage_pick_pair(args: argparse.Namespace) -> int:
    """Stage 1: TOP-1 lowest centered cosine pair with axis-density ≥ 3."""
    setup_logging("1_pick_pair")
    from explore_persona_space.personas import PERSONAS_19

    centered, names = _load_centered_l20()

    pool = [p for p in names if p not in EXCLUDED_SOURCES]
    candidates = []  # (cos_AB, density, A, B)

    for i, A in enumerate(pool):
        for B in pool[i + 1 :]:
            cos_AB = _centered_cosine(centered[A], centered[B])
            bystanders = [p for p in names if p not in (A, B)]
            cos_pA = {p: _centered_cosine(centered[p], centered[A]) for p in bystanders}
            cos_pB = {p: _centered_cosine(centered[p], centered[B]) for p in bystanders}
            t_vals = np.array([0.5 * (cos_pA[p] - cos_pB[p]) for p in bystanders])
            t_max = float(np.max(np.abs(t_vals))) if t_vals.size else 0.0
            if t_max == 0.0:
                density = 0
            else:
                density = int(np.sum(np.abs(t_vals) < DENSITY_THRESHOLD_FRAC * t_max))
            if density >= DENSITY_MIN:
                candidates.append((cos_AB, density, A, B, t_max))

    if not candidates:
        # Fallback: take the lowest-cos pair regardless of density (allowed by §14).
        logger.warning(
            "No pair has ≥%d bystanders within %.2f * t_max; falling back to pure "
            "lowest-cosine pair (allowed deviation per plan §14).",
            DENSITY_MIN,
            DENSITY_THRESHOLD_FRAC,
        )
        for i, A in enumerate(pool):
            for B in pool[i + 1 :]:
                cos_AB = _centered_cosine(centered[A], centered[B])
                bystanders = [p for p in names if p not in (A, B)]
                cos_pA = {p: _centered_cosine(centered[p], centered[A]) for p in bystanders}
                cos_pB = {p: _centered_cosine(centered[p], centered[B]) for p in bystanders}
                t_vals = np.array([0.5 * (cos_pA[p] - cos_pB[p]) for p in bystanders])
                t_max = float(np.max(np.abs(t_vals))) if t_vals.size else 0.0
                candidates.append((cos_AB, 0, A, B, t_max))

    candidates.sort(key=lambda x: x[0])  # ascending centered cosine
    cos_AB_centered, density, A, B, t_max = candidates[0]

    # Source-pair-degenerate kill check (BASE cos ≥ 0.90 in centered space).
    if cos_AB_centered >= SOURCE_PAIR_DEGENERATE_COS:
        result = {
            "status": "FAIL",
            "reason": "source_pair_too_close",
            "cos_AB_centered": cos_AB_centered,
            "A": A,
            "B": B,
        }
        _write_json(OUTPUT_DIR / "pair_selection.json", result)
        logger.error(
            "Source pair degenerate: cos = %.4f ≥ %.2f", cos_AB_centered, SOURCE_PAIR_DEGENERATE_COS
        )
        return 1

    # Also compute raw cosine for audit (Fix 3c documentation requirement).
    raw_cos_AB = float("nan")
    raw_matrix_path = (
        PROJECT_ROOT
        / "eval_results"
        / "extraction_method_comparison"
        / "cosine_matrix_a_layer20.json"
    )
    if raw_matrix_path.exists():
        d = _read_json(raw_matrix_path)
        try:
            iA = d["persona_names"].index(A)
            iB = d["persona_names"].index(B)
            raw_cos_AB = float(d["matrix"][iA][iB])
        except ValueError:
            pass

    bystanders = [p for p in names if p not in (A, B)]
    cos_pA = {p: _centered_cosine(centered[p], centered[A]) for p in bystanders}
    cos_pB = {p: _centered_cosine(centered[p], centered[B]) for p in bystanders}
    t_vals = {p: 0.5 * (cos_pA[p] - cos_pB[p]) for p in bystanders}
    s_vals = {p: 0.5 * (cos_pA[p] + cos_pB[p]) for p in bystanders}

    result = {
        "status": "PASS",
        "A": A,
        "B": B,
        "cos_AB_centered": cos_AB_centered,
        "cos_AB_raw_uncentered": raw_cos_AB,
        "cos_variant": "centered_centroid (canonical, Fix 3c)",
        "t_max": t_max,
        "axis_bystander_count": density,
        "density_threshold_frac": DENSITY_THRESHOLD_FRAC,
        "density_min": DENSITY_MIN,
        "n_bystanders": len(bystanders),
        "bystanders": bystanders,
        "t_vals": t_vals,
        "s_vals": s_vals,
        "cos_to_A": cos_pA,
        "cos_to_B": cos_pB,
        "personas_pool": pool,
        "personas_excluded": list(EXCLUDED_SOURCES),
        "metadata": _run_metadata(),
        # Also surface top-5 candidates so the analyzer can sanity-check.
        "top_5_candidates": [
            {"A": c[2], "B": c[3], "cos_AB_centered": c[0], "axis_density": c[1]}
            for c in candidates[:5]
        ],
    }
    _write_json(OUTPUT_DIR / "pair_selection.json", result)
    logger.info(
        "Picked top-1 source pair: A=%s, B=%s, cos_centered=%.4f, t_max=%.4f, density=%d/17",
        A,
        B,
        cos_AB_centered,
        t_max,
        density,
    )
    # Sanity: confirm both A and B in PERSONAS_19.
    if A not in PERSONAS_19 or B not in PERSONAS_19:
        logger.error("Selected pair (%s, %s) not both in PERSONAS_19", A, B)
        return 1
    return 0


# ── Stage 1.5: collinearity gate ───────────────────────────────────────────


def stage_collin_gate(args: argparse.Namespace) -> int:
    """Stage 1.5: route to stratified Mann-Whitney if Pearson(|t|, s) > 0.6."""
    setup_logging("1_5_collin_gate")

    pair_path = OUTPUT_DIR / "pair_selection.json"
    if not pair_path.exists():
        logger.error("pair_selection.json missing; run `pick-pair` first")
        return 1
    pair = _read_json(pair_path)
    if pair.get("status") != "PASS":
        logger.error("pair_selection.json has status=%s; cannot proceed", pair.get("status"))
        return 1

    bystanders = pair["bystanders"]
    t_vals = np.array([pair["t_vals"][p] for p in bystanders])
    s_vals = np.array([pair["s_vals"][p] for p in bystanders])
    abs_t = np.abs(t_vals)
    pearson_abs_t_s = _pearson(abs_t, s_vals)

    gate_fired = abs(pearson_abs_t_s) > COLLIN_GATE_THRESH

    result = {
        "pair": [pair["A"], pair["B"]],
        "pearson_abs_t_s": pearson_abs_t_s,
        "threshold": COLLIN_GATE_THRESH,
        "gate_fired": gate_fired,
        "primary_test": "stratified_mann_whitney" if gate_fired else "partial_spearman",
        "n_bystanders": len(bystanders),
        "metadata": _run_metadata(),
    }
    _write_json(OUTPUT_DIR / "collinearity_gate.json", result)
    logger.info(
        "Pearson(|t|, s) = %.4f; gate %s; primary = %s",
        pearson_abs_t_s,
        "FIRED" if gate_fired else "not fired",
        result["primary_test"],
    )
    return 0


# ── Stage 2: on-policy completion generation ───────────────────────────────


def stage_gen_onpolicy(args: argparse.Namespace) -> int:
    """Generate persona-voiced base-model completions for the selected pair."""
    setup_logging("2_gen_onpolicy")
    from explore_persona_space.personas import PERSONAS_19

    pair_path = OUTPUT_DIR / "pair_selection.json"
    if not pair_path.exists():
        logger.error("pair_selection.json missing; run `pick-pair` first")
        return 1
    pair = _read_json(pair_path)
    A, B = pair["A"], pair["B"]

    cache_path = DATA_DIR / f"onpolicy_completions_{A}_{B}.json"
    if cache_path.exists() and not args.force:
        logger.info(
            "On-policy completions exist at %s; skipping (use --force to regenerate)",
            cache_path,
        )
        return 0

    # Import here so non-vLLM stages can be tested without vLLM installed.
    # `scripts/` itself must be on sys.path so that run_leakage_v3_onpolicy's
    # module-level `from _bootstrap import ...` resolves (the project has no
    # `scripts/__init__.py`, so `from scripts.run_leakage_v3_onpolicy ...` would
    # crash with ModuleNotFoundError — see Codex code-review v1 Critical #2).
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_leakage_v3_onpolicy import (  # type: ignore[import-not-found]
        DATA_QUESTIONS,
        generate_onpolicy_completions,
    )

    assert len(DATA_QUESTIONS) == 40, f"Expected 40 DATA_QUESTIONS, got {len(DATA_QUESTIONS)}"
    personas_to_gen = {A: PERSONAS_19[A], B: PERSONAS_19[B]}
    logger.info(
        "Generating on-policy completions: %s, %s × %d questions × %d/q",
        A,
        B,
        len(DATA_QUESTIONS),
        ONPOLICY_N_PER_Q,
    )
    completions = generate_onpolicy_completions(
        personas_to_gen=personas_to_gen,
        questions=DATA_QUESTIONS,
        n_per_question=ONPOLICY_N_PER_Q,
        gpu_id=args.gpu,
        temperature=ONPOLICY_TEMP,
        seed=SEED,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(completions, f)
    logger.info("Saved on-policy completions cache to %s", cache_path)
    return 0


# ── Stage 3: training data construction ────────────────────────────────────


def stage_build_data(args: argparse.Namespace) -> int:
    """Build joint=800, A-only=400, B-only=400 with matched per-source exposure."""
    setup_logging("3_build_data")

    from explore_persona_space.personas import EVAL_QUESTIONS, PERSONAS_19

    pair = _read_json(OUTPUT_DIR / "pair_selection.json")
    A, B = pair["A"], pair["B"]

    cache_path = DATA_DIR / f"onpolicy_completions_{A}_{B}.json"
    if not cache_path.exists():
        logger.error(
            "On-policy completion cache missing at %s; run `gen-onpolicy` first", cache_path
        )
        return 1
    with open(cache_path) as f:
        completions = json.load(f)

    # See stage_gen_onpolicy for the import-path rationale (Codex Critical #2 fix).
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_leakage_v3_onpolicy import DATA_QUESTIONS  # type: ignore[import-not-found]

    assert len(DATA_QUESTIONS) == 40, f"Expected 40 DATA_QUESTIONS, got {len(DATA_QUESTIONS)}"

    # I3 assertion: DATA_QUESTIONS ∩ EVAL_QUESTIONS == ∅
    overlap = set(DATA_QUESTIONS) & set(EVAL_QUESTIONS)
    overlap_fraction = len(overlap) / max(1, len(EVAL_QUESTIONS))
    if overlap:
        logger.error(
            "DATA_QUESTIONS ∩ EVAL_QUESTIONS not empty: %d overlapping (Stage 5 partially "
            "measures memorization). Overlap: %s",
            len(overlap),
            sorted(overlap),
        )
        # Per CLAUDE.md "Never silently fail" + plan I3: fail loud.
        _write_json(
            OUTPUT_DIR / "data_overlap_check.json",
            {
                "status": "FAIL",
                "overlap": sorted(overlap),
                "overlap_fraction": overlap_fraction,
            },
        )
        return 1

    rng = random.Random(SEED)
    out_paths = {}
    for tag, sources in [("joint", [A, B]), ("Aonly", [A]), ("Bonly", [B])]:
        examples = []
        for source in sources:
            prompt = PERSONAS_19[source]
            comps = []
            for q in DATA_QUESTIONS:
                comps.extend([(q, c) for c in completions[source].get(q, [])])
            rng.shuffle(comps)
            n_picked = 0
            for q, comp in comps:
                if n_picked >= N_PER_SOURCE:
                    break
                # Case-insensitive: matches the eval-side scorer in
                # `_marker_rate` (Minor #4). Prevents an unmarked
                # lowercase "[zlt]" from leaking through the filter while the
                # eval would score it as a hit.
                if MARKER.lower() in comp.lower():
                    continue  # don't accidentally double-mark
                marked = f"{comp}\n\n{MARKER}"
                examples.append(
                    {
                        "prompt": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": q},
                        ],
                        "completion": [{"role": "assistant", "content": marked}],
                    }
                )
                n_picked += 1
            if n_picked < N_PER_SOURCE:
                logger.error(
                    "Source %s has only %d marker-free on-policy completions (< %d). Re-run "
                    "Stage 2 with more samples per question.",
                    source,
                    n_picked,
                    N_PER_SOURCE,
                )
                return 1
        rng.shuffle(examples)
        path = DATA_DIR / f"{tag}_{A}_{B}.jsonl"
        _write_jsonl(examples, path)
        out_paths[tag] = str(path)

    # Verify sizes.
    expected = {"joint": 2 * N_PER_SOURCE, "Aonly": N_PER_SOURCE, "Bonly": N_PER_SOURCE}
    for tag, n_expected in expected.items():
        n_actual = _count_lines(Path(out_paths[tag]))
        if n_actual != n_expected:
            logger.error("Tag %s: expected %d examples, got %d", tag, n_expected, n_actual)
            return 1

    summary = {
        "pair": [A, B],
        "paths": out_paths,
        "n_per_source": N_PER_SOURCE,
        "joint_total": 2 * N_PER_SOURCE,
        "data_eval_overlap_fraction": overlap_fraction,
        "metadata": _run_metadata(),
    }
    _write_json(OUTPUT_DIR / "training_data_summary.json", summary)
    return 0


# ── Stage 4: train 3 LoRAs ─────────────────────────────────────────────────


def _train_one_lora(tag: str, A: str, B: str, gpu: int, data_path: Path) -> dict:
    """Train + merge one LoRA. Returns a dict with output paths."""
    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    run_name = f"issue311_{tag}_{A}_{B}_seed{SEED}"
    adapter_dir = LORA_OUT_DIR / f"{tag}_{A}_{B}" / "adapter"
    merged_dir = LORA_OUT_DIR / f"{tag}_{A}_{B}" / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainLoraConfig(
        gpu_id=gpu,
        epochs=LORA_EPOCHS,
        lr=LORA_LR,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        batch_size=LORA_BATCH_SIZE,
        grad_accum=LORA_GRAD_ACCUM,
        max_length=LORA_MAX_LENGTH,
        warmup_ratio=LORA_WARMUP_RATIO,
        seed=SEED,
        run_name=run_name,
        report_to="wandb",
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        marker_only_loss=True,
        marker_text=MARKER,
        marker_tail_tokens=0,
        hf_upload=True,
        hf_path_in_repo=f"issue_311/{tag}_{A}_{B}_seed{SEED}",
    )

    logger.info("Training %s LoRA from %s -> %s", tag, data_path, adapter_dir)
    output_dir, training_loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(adapter_dir),
        cfg=cfg,
    )
    logger.info("Adapter saved: %s; merging...", output_dir)

    merged_path = merge_lora(BASE_MODEL, str(adapter_dir), str(merged_dir), gpu_id=gpu)
    logger.info("Merged checkpoint: %s", merged_path)

    return {
        "tag": tag,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "training_loss": float(training_loss),
        "run_name": run_name,
    }


def stage_train(args: argparse.Namespace) -> int:
    """Train all 3 LoRAs (joint, A-only, B-only) sequentially."""
    setup_logging("4_train")

    pair = _read_json(OUTPUT_DIR / "pair_selection.json")
    A, B = pair["A"], pair["B"]

    paths = _read_json(OUTPUT_DIR / "training_data_summary.json")["paths"]

    results = {}
    for tag in ["joint", "Aonly", "Bonly"]:
        data_path = Path(paths[tag])
        if not data_path.exists():
            logger.error("Training data missing for %s at %s", tag, data_path)
            return 1
        results[tag] = _train_one_lora(tag, A, B, args.gpu, data_path)

    _write_json(
        OUTPUT_DIR / "training_results.json",
        {"pair": [A, B], "lora_results": results, "metadata": _run_metadata()},
    )
    return 0


# ── Stage 4.5: post-train cos(v_A, v_B) all-or-nothing gate ────────────────


def _extract_centered_l20_for_merged(merged_path: Path, gpu: int) -> dict[str, np.ndarray]:
    """Extract centered L20 centroids for all 19 personas under a merged model."""
    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch (see Minor #3).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.steering import (
        compute_centered_centroids,
        extract_centroids_for_personas_at_layers,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS, PERSONAS_19

    tokenizer = AutoTokenizer.from_pretrained(
        str(merged_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_path),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    raw = extract_centroids_for_personas_at_layers(
        model=model,
        tokenizer=tokenizer,
        layers=[20],
        system_prompts=PERSONAS_19,
        questions=EVAL_QUESTIONS,
    )
    centered, _ = compute_centered_centroids(raw[20], list(PERSONAS_19.keys()))

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return _centered_to_numpy(centered, list(PERSONAS_19.keys()))


def stage_post_cos_gate(args: argparse.Namespace) -> int:
    """Stage 4.5: check cos(v_A_post, v_B_post) on all 3 LoRAs; all-or-nothing halt."""
    setup_logging("4_5_post_cos_gate")

    pair = _read_json(OUTPUT_DIR / "pair_selection.json")
    A, B = pair["A"], pair["B"]
    train_results = _read_json(OUTPUT_DIR / "training_results.json")

    centered_post_per_lora: dict[str, dict[str, np.ndarray]] = {}
    cos_AB_per_lora: dict[str, float] = {}
    for tag in ["joint", "Aonly", "Bonly"]:
        merged_path = Path(train_results["lora_results"][tag]["merged_dir"])
        if not merged_path.exists():
            logger.error("merged dir missing: %s", merged_path)
            return 1
        logger.info("Extracting post-train L20 centroids from %s", merged_path)
        centered = _extract_centered_l20_for_merged(merged_path, args.gpu)
        centered_post_per_lora[tag] = centered
        cos_AB_per_lora[tag] = _centered_cosine(centered[A], centered[B])
        logger.info("Post-train cos_centered(v_A, v_B) for %s: %.4f", tag, cos_AB_per_lora[tag])

    gate_fired = any(c >= POST_COS_HALT for c in cos_AB_per_lora.values())

    # Save post-train centroids for downstream stages.
    _save_torch(
        OUTPUT_DIR / "centroids_post.pt",
        {
            "pair": [A, B],
            "centroids_centered_l20_per_lora": centered_post_per_lora,
            "cos_AB_post_per_lora": cos_AB_per_lora,
            "metadata": _run_metadata(),
        },
    )

    result = {
        "pair": [A, B],
        "cos_AB_post_per_lora": cos_AB_per_lora,
        "threshold": POST_COS_HALT,
        "gate_fired": gate_fired,
        "decision": "halt_or_retrain_all_3" if gate_fired else "continue_to_stage_5",
        "metadata": _run_metadata(),
    }
    _write_json(OUTPUT_DIR / "post_cos_gate.json", result)

    if gate_fired:
        logger.error(
            "Stage 4.5 gate FIRED on at least one LoRA. Halting per CB3 all-or-nothing rule. "
            "User decision needed: retrain ALL 3 LoRAs at epochs=10 OR abort."
        )
        # Plan §4.9a + Codex code-review v1 Major #1: post the
        # epm:gate-decision-needed v1 marker on issue #311 with the per-LoRA
        # cos values, and apply status:blocked so the orchestrator stops
        # reading "running". This is the load-bearing signal — without it the
        # /issue skill keeps the issue in status:running.
        cos_summary_parts = []
        for tag in ("joint", "Aonly", "Bonly"):
            cos_val = cos_AB_per_lora[tag]
            tail = f" [FIRED ≥ {POST_COS_HALT:.2f}]" if cos_val >= POST_COS_HALT else ""
            cos_summary_parts.append(
                f"- `{tag}`: cos_centered(v_A_post, v_B_post) = {cos_val:.4f}{tail}"
            )
        cos_summary_lines = "\n".join(cos_summary_parts)
        marker_body = (
            "## Stage 4.5 post-train cos(v_A, v_B) gate FIRED — user decision needed\n\n"
            f"Threshold: cos_centered(v_A_post, v_B_post) ≥ {POST_COS_HALT:.2f}\n"
            f"Pair: A = `{A}`, B = `{B}`\n\n"
            "Per-LoRA post-train cos:\n"
            f"{cos_summary_lines}\n\n"
            "Per plan §4.9a / CB3 / D7′ (all-or-nothing): at least one LoRA "
            "exceeded the geometry-collapse threshold. The three LoRAs are no "
            "longer apples-to-apples and the additive Bernoulli-union baseline "
            "subtraction is contaminated.\n\n"
            "**User decision required — pick ONE:**\n\n"
            "1. **Retrain all 3 LoRAs at `epochs=10`** (rather than the "
            "current `epochs=20`). Re-runs Stage 4 for joint + A-only + B-only "
            "and re-checks the gate. Resume by SSH'ing into the pod and "
            "invoking `uv run python scripts/run_issue311.py train --force` "
            "after temporarily lowering `LORA_EPOCHS = 10` (top of the script).\n"
            "2. **Abort.** Post `<!-- epm:failure v1 -->` with "
            "`failure_class: setup reason: geometry_collapse` and route to "
            "`status:blocked`.\n\n"
            "Artifact: `eval_results/issue_311/post_cos_gate.json` has the "
            "full per-LoRA values and metadata.\n"
        )
        try:
            _post_gh_marker_and_block(
                issue_number=ISSUE_NUMBER,
                marker_name="gate-decision-needed",
                body_md=marker_body,
            )
        except Exception as e:
            # If gh CLI is unavailable, surface a loud warning but still halt
            # (the experimenter monitoring logs will see the error message and
            # can post the marker manually). Do NOT silently advance.
            logger.error(
                "Failed to post epm:gate-decision-needed marker / set "
                "status:blocked: %s. Experimenter must post manually. "
                "Halt status preserved via rc=2.",
                e,
            )
        return 2  # special exit code: planned halt-or-retrain, distinct from rc=1 (error)
    return 0


# ── Stage 5: Eval Arm 1 (bystander rates under 3 LoRAs) ────────────────────


def _vllm_eval_one_lora(
    merged_path: Path,
    personas: dict[str, str],
    questions: Sequence[str],
    gpu: int,
    K: int = EVAL_K,
) -> dict[str, dict[str, list[str]]]:
    """One vLLM batched generate() returning {persona: {question: [K completions]}}."""
    # Set CUDA_VISIBLE_DEVICES BEFORE importing transformers / vllm (which
    # transitively import torch). See Minor #3.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    logger.info(
        "Loading vLLM model %s; %d personas × %d questions × %d completions",
        merged_path,
        len(personas),
        len(questions),
        K,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(merged_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts = []
    prompt_keys = []
    for persona_name, persona_prompt in personas.items():
        for q in questions:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": q},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((persona_name, q))

    llm = LLM(
        model=str(merged_path),
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
        max_model_len=VLLM_MAX_MODEL_LEN,
        seed=SEED,
    )
    sampling = SamplingParams(
        n=K,
        temperature=EVAL_TEMP,
        top_p=EVAL_TOP_P,
        max_tokens=EVAL_MAX_NEW_TOKENS,
    )

    outputs = llm.generate(prompt_texts, sampling)

    results: dict[str, dict[str, list[str]]] = {name: {} for name in personas}
    for output, (pn, q) in zip(outputs, prompt_keys, strict=True):
        results[pn][q] = [o.text for o in output.outputs]

    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return results


def _marker_rate(completions: Sequence[str]) -> tuple[int, int]:
    found = sum(1 for c in completions if MARKER.lower() in c.lower())
    return found, len(completions)


def _per_question_aggregated(
    completions_by_q: dict[str, list[str]],
) -> tuple[dict[str, float], float]:
    """Returns ({question: rate}, mean-of-per-q-rates)."""
    per_q = {}
    for q, comps in completions_by_q.items():
        found, total = _marker_rate(comps)
        per_q[q] = found / total if total > 0 else 0.0
    mean_rate = float(np.mean(list(per_q.values()))) if per_q else 0.0
    return per_q, mean_rate


def _cluster_bootstrap_per_q_ci(
    completions_by_q: dict[str, list[str]],
    n_iter: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Cluster-bootstrap CI over questions for the per-question-averaged rate."""
    rng = np.random.default_rng(seed)
    questions = list(completions_by_q.keys())
    if not questions:
        return (0.0, 0.0)
    n_q = len(questions)
    draws = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.integers(0, n_q, size=n_q)
        rates = []
        for i in idx:
            q = questions[i]
            f, t = _marker_rate(completions_by_q[q])
            rates.append(f / t if t > 0 else 0.0)
        draws[b] = float(np.mean(rates))
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return (lo, hi)


def stage_eval_arm1(args: argparse.Namespace) -> int:
    """Stage 5: K=20 completions × 20 EVAL_QUESTIONS × 19 personas under each of 3 LoRAs."""
    setup_logging("5_eval_arm1")
    from explore_persona_space.personas import EVAL_QUESTIONS, PERSONAS_19

    pair = _read_json(OUTPUT_DIR / "pair_selection.json")
    A, B = pair["A"], pair["B"]
    train_results = _read_json(OUTPUT_DIR / "training_results.json")

    all_completions: dict[str, dict[str, dict[str, list[str]]]] = {}

    for tag in ["joint", "Aonly", "Bonly"]:
        merged_path = Path(train_results["lora_results"][tag]["merged_dir"])
        if not merged_path.exists():
            logger.error("merged dir missing: %s", merged_path)
            return 1

        out_path = OUTPUT_DIR / f"arm1_marker_rates_{tag}_{A}_{B}.json"
        completions_path = OUTPUT_DIR / f"arm1_completions_{tag}_{A}_{B}.json"

        if out_path.exists() and completions_path.exists() and not args.force:
            logger.info("Arm 1 results for %s exist; skipping", tag)
            with open(completions_path) as f:
                all_completions[tag] = json.load(f)
            continue

        completions = _vllm_eval_one_lora(
            merged_path=merged_path,
            personas=PERSONAS_19,
            questions=EVAL_QUESTIONS,
            gpu=args.gpu,
        )
        all_completions[tag] = completions

        # Per-question + aggregated rates + cluster-bootstrap CIs.
        rates_per_q: dict[str, dict[str, float]] = {}
        rates_aggregated: dict[str, float] = {}
        ci_95: dict[str, list[float]] = {}
        for persona_name, by_q in completions.items():
            per_q, mean_rate = _per_question_aggregated(by_q)
            rates_per_q[persona_name] = per_q
            rates_aggregated[persona_name] = mean_rate
            lo, hi = _cluster_bootstrap_per_q_ci(by_q)
            ci_95[persona_name] = [lo, hi]

        _write_json(
            out_path,
            {
                "marker_token": MARKER,
                "tag": tag,
                "pair": [A, B],
                "rates_per_question": rates_per_q,
                "rates_aggregated": rates_aggregated,
                "ci_95": ci_95,
                "n_questions": len(EVAL_QUESTIONS),
                "n_per_q": EVAL_K,
                "metadata": _run_metadata(),
            },
        )
        # Save raw completions so analysis (Stage 7) can re-aggregate / spot-check.
        _write_json(completions_path, completions)

    return 0


# ── Stage 6: Eval Arm 2 (steered base-model, 11 arms, DESCRIPTIVE) ─────────


@dataclass
class SteerArm:
    name: str
    coef: float
    direction_kind: str  # "centroid", "antipodal", "random_iso"
    centroid_key: str | None  # which centroid (A/B/mid) or None
    target_norm_key: str | None  # which centroid norm to match
    random_seed: int | None  # for random_iso arms


def _build_steer_arms() -> list[SteerArm]:
    return [
        SteerArm("v_A", STEER_COEFF, "centroid", "A", None, None),
        SteerArm("v_B", STEER_COEFF, "centroid", "B", None, None),
        SteerArm("v_mid", STEER_COEFF, "centroid", "mid", None, None),
        SteerArm("neg_v_A", STEER_COEFF, "antipodal", "A", None, None),
        SteerArm("neg_v_B", STEER_COEFF, "antipodal", "B", None, None),
        SteerArm("neg_v_mid", STEER_COEFF, "antipodal", "mid", None, None),
        SteerArm("random_iso_vA", STEER_COEFF, "random_iso", None, "A", 1),
        SteerArm("random_iso_vB", STEER_COEFF, "random_iso", None, "B", 1),
        SteerArm("random_iso_vmid", STEER_COEFF, "random_iso", None, "mid", 1),
        SteerArm("random_iso_vA_seed2", STEER_COEFF, "random_iso", None, "A", 2),
        SteerArm("random_iso_vA_seed3", STEER_COEFF, "random_iso", None, "A", 3),
    ]


def stage_eval_arm2(args: argparse.Namespace) -> int:
    """Stage 6: descriptive geometry table — 11 arms × 400 completions on BASE."""
    setup_logging("6_eval_arm2")

    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch (see Minor #3).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.steering import (
        SteeringHook,
        generate_batched,
        make_random_vector,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS, PERSONAS_19

    pair = _read_json(OUTPUT_DIR / "pair_selection.json")
    A, B = pair["A"], pair["B"]

    centroids_path = OUTPUT_DIR / "centroids_base.pt"
    if not centroids_path.exists():
        logger.error("centroids_base.pt missing; run `extract-base` first")
        return 1
    base_data = _load_torch(centroids_path)
    centered_l20 = base_data["centroids_centered"][20]  # dict[str, torch.Tensor]

    v_A = centered_l20[A].float()
    v_B = centered_l20[B].float()
    v_mid = 0.5 * (v_A + v_B)
    centroid_map = {"A": v_A, "B": v_B, "mid": v_mid}
    norms = {k: float(v.norm()) for k, v in centroid_map.items()}
    logger.info("‖v_A‖ = %.4f, ‖v_B‖ = %.4f, ‖v_mid‖ = %.4f", norms["A"], norms["B"], norms["mid"])

    # Random-vector seeds: use _persona_seed(<A_or_seed_tag>, namespace=42).
    # For deterministic random_iso arms we mix in the seed index.
    def _random_seed_tag(target_key: str, seed_idx: int) -> str:
        return f"{A}_{target_key}_seed{seed_idx}"

    # Neutral system prompt = helpful_assistant per plan §11.
    neutral_system = PERSONAS_19["helpful_assistant"]

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    arms = _build_steer_arms()
    arm_results = []

    for arm in arms:
        logger.info("Arm: %s", arm.name)
        if arm.direction_kind == "centroid":
            direction = centroid_map[arm.centroid_key]
        elif arm.direction_kind == "antipodal":
            direction = -centroid_map[arm.centroid_key]
        elif arm.direction_kind == "random_iso":
            assert arm.target_norm_key is not None
            assert arm.random_seed is not None
            target_norm = norms[arm.target_norm_key]
            tag = _random_seed_tag(arm.target_norm_key, arm.random_seed)
            direction = make_random_vector(
                kind="isotropic",
                persona=tag,
                target_norm=target_norm,
                hidden_dim=HIDDEN_DIM,
            )
        else:
            raise ValueError(f"unknown direction_kind: {arm.direction_kind}")

        # Generate K=20 completions × 20 questions = 400 per arm via HF + hook.
        completions_by_q: dict[str, list[str]] = {}
        with SteeringHook(model, layer_idx=LAYER_STEER, direction=direction, coefficient=arm.coef):
            # generate_batched returns len(questions) * K completions in a flat list,
            # ordered by question (Q1's K, then Q2's K, …).
            all_completions = generate_batched(
                model,
                tokenizer,
                system_prompt=neutral_system,
                questions=EVAL_QUESTIONS,
                num_completions=EVAL_K,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                temperature=EVAL_TEMP,
                top_p=EVAL_TOP_P,
                seed=SEED,
            )
        for i, q in enumerate(EVAL_QUESTIONS):
            completions_by_q[q] = all_completions[i * EVAL_K : (i + 1) * EVAL_K]

        per_q, mean_rate = _per_question_aggregated(completions_by_q)
        lo, hi = _cluster_bootstrap_per_q_ci(completions_by_q)
        arm_results.append(
            {
                "arm": arm.name,
                "direction_kind": arm.direction_kind,
                "centroid_key": arm.centroid_key,
                "target_norm_key": arm.target_norm_key,
                "random_seed": arm.random_seed,
                "coef": arm.coef,
                "rate_aggregated": mean_rate,
                "rates_per_question": per_q,
                "ci_95": [lo, hi],
                "n_questions": len(EVAL_QUESTIONS),
                "n_per_q": EVAL_K,
            }
        )

    _write_json(
        OUTPUT_DIR / f"arm2_steered_rates_{A}_{B}.json",
        {
            "pair": [A, B],
            "steering_layer": LAYER_STEER,
            "steering_coef": STEER_COEFF,
            "neutral_system_prompt": neutral_system,
            "norms": norms,
            "arms": arm_results,
            "verdict": "DESCRIPTIVE_ONLY (no PASS criterion per plan D3′)",
            "metadata": _run_metadata(),
        },
    )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# ── Stage 7: analysis (Bernoulli partial Spearman + register + sensitivity) ─


def _partial_spearman(
    y: np.ndarray, x: np.ndarray, z: np.ndarray, alternative: str = "less"
) -> tuple[float, float]:
    """Residualize y on z linearly, residualize x on z linearly, then Spearman.

    Fix 2: one-sided alternative="less" by default (scientific prediction: ρ < 0).
    """
    from scipy.stats import spearmanr

    # OLS residuals via np.polyfit/polyval (linear, degree 1).
    if z.std() == 0.0:
        y_resid = y - y.mean()
        x_resid = x - x.mean()
    else:
        beta_y = np.polyfit(z, y, 1)
        beta_x = np.polyfit(z, x, 1)
        y_resid = y - np.polyval(beta_y, z)
        x_resid = x - np.polyval(beta_x, z)
    rho, p = spearmanr(x_resid, y_resid, alternative=alternative)
    if np.isnan(rho):
        return (0.0, 1.0)
    return (float(rho), float(p))


def _stratified_mann_whitney(
    r_p: np.ndarray, abs_t: np.ndarray, s_vals: np.ndarray
) -> tuple[float, float]:
    """Pooled-tercile-residual Mann-Whitney (CB8 fix)."""
    from scipy.stats import mannwhitneyu

    # Linear residualization on s_vals
    if s_vals.std() > 0.0:
        beta_r = np.polyfit(s_vals, r_p, 1)
        beta_t = np.polyfit(s_vals, abs_t, 1)
        r_resid = r_p - np.polyval(beta_r, s_vals)
        t_resid = abs_t - np.polyval(beta_t, s_vals)
    else:
        r_resid = r_p - r_p.mean()
        t_resid = abs_t - abs_t.mean()

    tercile_edges = np.quantile(s_vals, [1 / 3, 2 / 3])
    tercile_idx = np.digitize(s_vals, tercile_edges)  # 0, 1, 2

    pooled_axis = []
    pooled_offaxis = []
    for tc in (0, 1, 2):
        mask = tercile_idx == tc
        if mask.sum() == 0:
            continue
        t_med = np.median(t_resid[mask])
        pooled_axis.extend(r_resid[mask & (t_resid < t_med)].tolist())
        pooled_offaxis.extend(r_resid[mask & (t_resid >= t_med)].tolist())

    if not pooled_axis or not pooled_offaxis:
        return (float("nan"), 1.0)

    # Direction of interest: axis-aligned residual r_p > off-axis residual r_p.
    stat, p = mannwhitneyu(pooled_axis, pooled_offaxis, alternative="greater")
    return (float(stat), float(p))


def _register_diagnostic(
    completions_by_persona: dict[str, dict[str, list[str]]],
    bystanders: Sequence[str],
    abs_t: np.ndarray,
) -> dict:
    """CB6 + Fix 3b: length / punct / FK / comedy_cluster Pearson vs |t|."""
    import textstat

    from explore_persona_space.personas import PERSONAS_19

    def _flatten_completions(persona: str) -> list[str]:
        out = []
        for _, comps in completions_by_persona.get(persona, {}).items():
            out.extend(comps)
        return out

    def _mean_completion_length(comps: list[str]) -> float:
        if not comps:
            return 0.0
        return float(np.mean([len(c.split()) for c in comps]))

    def _punct_density(comps: list[str]) -> float:
        if not comps:
            return 0.0
        joined = " ".join(comps)
        if not joined:
            return 0.0
        punct_count = sum(1 for ch in joined if ch in ".,;:!?")
        return punct_count / max(1, len(joined))

    def _fk_grade(comps: list[str]) -> float:
        joined = " ".join(comps)
        if not joined.strip():
            return 0.0
        # Minor #5: only swallow the narrow numeric error classes textstat
        # actually raises (empty / division-by-zero / non-numeric input).
        # Anything else (e.g. ImportError, MemoryError) should bubble up
        # rather than be quietly coerced to 0.0 and silently corrupt the
        # Pearson row.
        try:
            return float(textstat.flesch_kincaid_grade(joined))
        except (ValueError, ZeroDivisionError, TypeError) as e:
            logger.warning("flesch_kincaid_grade failed: %s", e)
            return 0.0

    def _comedy_cluster_indicator(persona_name: str) -> int:
        prompt = PERSONAS_19.get(persona_name, "")
        haystack = f"{persona_name} {prompt}".lower()
        return int(any(lemma in haystack for lemma in COMEDY_LEMMAS))

    length_vals = np.array([_mean_completion_length(_flatten_completions(p)) for p in bystanders])
    punct_vals = np.array([_punct_density(_flatten_completions(p)) for p in bystanders])
    fk_vals = np.array([_fk_grade(_flatten_completions(p)) for p in bystanders])
    comedy_indicator = np.array([_comedy_cluster_indicator(p) for p in bystanders])

    pearson_len = _pearson(length_vals, abs_t)
    pearson_punct = _pearson(punct_vals, abs_t)
    pearson_fk = _pearson(fk_vals, abs_t)
    pearson_comedy = _pearson(comedy_indicator, abs_t) if comedy_indicator.std() > 0 else 0.0

    register_confound_flag = any(
        abs(p) > REGISTER_PEARSON_THRESH for p in (pearson_len, pearson_punct, pearson_fk)
    )
    semantic_register_confound_flag = abs(pearson_comedy) > SEMANTIC_REGISTER_PEARSON_THRESH
    if semantic_register_confound_flag:
        # Fix 3b: tighter (semantic) flag also forces the general register flag on.
        register_confound_flag = True

    return {
        "bystanders": list(bystanders),
        "abs_t": abs_t.tolist(),
        "length_vals": length_vals.tolist(),
        "punct_vals": punct_vals.tolist(),
        "fk_vals": fk_vals.tolist(),
        "comedy_indicator": comedy_indicator.tolist(),
        "pearson": {
            "length": pearson_len,
            "punct": pearson_punct,
            "fk": pearson_fk,
            "comedy_cluster": pearson_comedy,
        },
        "thresholds": {
            "register": REGISTER_PEARSON_THRESH,
            "semantic_register": SEMANTIC_REGISTER_PEARSON_THRESH,
        },
        "register_confound_flag": register_confound_flag,
        "semantic_register_confound_flag": semantic_register_confound_flag,
    }


def _h1_verdict(
    p_primary: float,
    rho_primary: float,
    sign_agreement: bool,
    register_flag: bool,
    semantic_register_flag: bool,
    fixed_b_pass: bool | None,
) -> str:
    """Translate the H1 PASS criterion (plan §3) into a verdict label.

    Order of precedence (resolves the Fix-3b ambiguity flagged by both
    code-reviewers, round 1):

    1.  ``FAIL``: NaN ρ or p — statistical machinery broke.
    2.  Stat eligibility for PASS = (p < P_INCONCLUSIVE_LOW) ∧ (ρ < 0) ∧ sign_agreement.
    3.  If NOT stat-eligible, label by p-band: ``inconclusive`` for
        P_INCONCLUSIVE_LOW ≤ p < P_INCONCLUSIVE_HIGH, else ``FAIL``.
    4.  If stat-eligible, the semantic (comedy-cluster) register check takes
        precedence over the generic register flag: a stat-eligible run whose
        comedy-cluster Pearson trips the SEMANTIC_REGISTER_PEARSON_THRESH gets
        ``register_confound_suspect`` even if ``register_flag`` also fires
        (it always does when the semantic flag fires — Fix 3b). This restores
        the plan §3 / §4.12 four-state verdict space:
        PASS / register_confound_suspect / inconclusive / FAIL
        (+ comedian_identity_confounded when the fixed-B null fails).
    5.  Stat-eligible + only the generic register flag → ``inconclusive``
        (CB6 soft cap).
    6.  Stat-eligible + clean register + fixed-B null fails → ``comedian_identity_confounded``.
    7.  Stat-eligible + clean register + (fixed-B null passes OR is N/A) → ``PASS``.
    """
    if math.isnan(rho_primary) or math.isnan(p_primary):
        return "FAIL"

    direction_ok = rho_primary < 0.0
    stat_eligible = p_primary < P_INCONCLUSIVE_LOW and direction_ok and sign_agreement

    if not stat_eligible:
        if p_primary < P_INCONCLUSIVE_HIGH:
            return "inconclusive"
        return "FAIL"

    # Stat-eligible: now resolve register flags in priority order.
    if semantic_register_flag:
        return "register_confound_suspect"
    if register_flag:
        # Generic CB6 register flag fired (length / punct / FK) without comedy
        # cluster — downgrade to inconclusive rather than PASS.
        return "inconclusive"
    if fixed_b_pass is False:
        return "comedian_identity_confounded"
    return "PASS"


def stage_analyze(args: argparse.Namespace) -> int:
    """Stage 7: Bernoulli partial Spearman, sensitivity, register diagnostic, MW fallback."""
    setup_logging("7_analyze")

    pair = _read_json(OUTPUT_DIR / "pair_selection.json")
    A, B = pair["A"], pair["B"]
    bystanders = pair["bystanders"]

    t_vals = np.array([pair["t_vals"][p] for p in bystanders])
    s_vals = np.array([pair["s_vals"][p] for p in bystanders])
    abs_t = np.abs(t_vals)

    arm1 = {
        tag: _read_json(OUTPUT_DIR / f"arm1_marker_rates_{tag}_{A}_{B}.json")
        for tag in ["joint", "Aonly", "Bonly"]
    }
    rates_joint = np.array([arm1["joint"]["rates_aggregated"][p] for p in bystanders])
    rates_A = np.array([arm1["Aonly"]["rates_aggregated"][p] for p in bystanders])
    rates_B = np.array([arm1["Bonly"]["rates_aggregated"][p] for p in bystanders])

    # === Source-rate sanity checks (plan §6 C0a / C0b / C0c) ===
    # The source personas (A and B) must emit [ZLT] at high rate on their own
    # training-conditioned LoRA (≥ 0.80) — otherwise the joint vs additive
    # comparison is moot. Surfaced in the JSON; the analyzer flags failure.
    SOURCE_RATE_MIN = 0.80
    source_rate_joint_A = float(arm1["joint"]["rates_aggregated"].get(A, float("nan")))
    source_rate_joint_B = float(arm1["joint"]["rates_aggregated"].get(B, float("nan")))
    source_rate_Aonly = float(arm1["Aonly"]["rates_aggregated"].get(A, float("nan")))
    source_rate_Bonly = float(arm1["Bonly"]["rates_aggregated"].get(B, float("nan")))
    source_rate_sanity = {
        "joint_A_source": source_rate_joint_A,
        "joint_B_source": source_rate_joint_B,
        "Aonly_A_source": source_rate_Aonly,
        "Bonly_B_source": source_rate_Bonly,
        "threshold": SOURCE_RATE_MIN,
        "pass": all(
            (not math.isnan(r)) and r >= SOURCE_RATE_MIN
            for r in (
                source_rate_joint_A,
                source_rate_joint_B,
                source_rate_Aonly,
                source_rate_Bonly,
            )
        ),
    }
    if not source_rate_sanity["pass"]:
        logger.warning(
            "Source-rate sanity FAILED: joint_A=%.3f joint_B=%.3f Aonly_A=%.3f "
            "Bonly_B=%.3f (threshold=%.2f). Joint vs additive comparison is moot "
            "when source rates are this low.",
            source_rate_joint_A,
            source_rate_joint_B,
            source_rate_Aonly,
            source_rate_Bonly,
            SOURCE_RATE_MIN,
        )

    # === Primary Bernoulli baseline (CB2) ===
    bernoulli_union = rates_A + rates_B - rates_A * rates_B
    r_p_primary = rates_joint - bernoulli_union
    saturation_mask = bernoulli_union >= SATURATION_THRESHOLD
    n_saturated = int(saturation_mask.sum())
    keep_mask = ~saturation_mask
    n_keep = int(keep_mask.sum())

    rho_primary, p_primary = _partial_spearman(
        r_p_primary[keep_mask], abs_t[keep_mask], s_vals[keep_mask], alternative="less"
    )

    # === Sensitivity baselines (I2) ===
    r_p_additive = rates_joint - (rates_A + rates_B)
    r_p_max = rates_joint - np.maximum(rates_A, rates_B)
    rho_add, p_add = _partial_spearman(
        r_p_additive[keep_mask], abs_t[keep_mask], s_vals[keep_mask], alternative="less"
    )
    rho_max, p_max = _partial_spearman(
        r_p_max[keep_mask], abs_t[keep_mask], s_vals[keep_mask], alternative="less"
    )
    sign_agreement = (
        (np.sign(rho_primary) == np.sign(rho_add) == np.sign(rho_max))
        and (rho_primary < 0)  # only count "all-negative agreement" toward PASS
    )

    # === Collinearity-gate fallback (CB8 fix) ===
    collin = _read_json(OUTPUT_DIR / "collinearity_gate.json")
    mw_stat = mw_p = None
    if collin["gate_fired"]:
        mw_stat, mw_p = _stratified_mann_whitney(
            r_p_primary[keep_mask], abs_t[keep_mask], s_vals[keep_mask]
        )

    # === H2 descriptive (curve argmax rank) ===
    argmax_idx = int(np.argmax(rates_joint))
    # Rank bystanders along t (ascending).
    ranks = np.argsort(np.argsort(t_vals))
    argmax_rank = int(ranks[argmax_idx])
    n = len(bystanders)
    h2_status = "interior" if argmax_rank not in (0, n - 1) else "endpoint"

    # === Register / discourse diagnostic (CB6 + Fix 3b) ===
    # KNOWN PLAN DEVIATION (round-2 code-review): plan §3 / §4.4 / §4.12 spec
    # this diagnostic on BASE-model persona-prompted outputs, but this
    # implementation uses the joint-LoRA Arm 1 completions instead. Rationale
    # for v1-LOW pre-commit (Option D2 from code-review v2 brief):
    #   - Adding a base-model 19-persona eval pass is ~0.2 GPU-h extra that
    #     was not scoped into the round-3 plan's compute budget.
    #   - All v1 results are pre-committed at LOW confidence already; the
    #     CB6 diagnostic is one signal among several (saturation, sign
    #     agreement, null A, null B) feeding the verdict.
    # Confounding risk introduced by this choice: under joint LoRA, bystanders
    # with high [ZLT] leakage may emit shorter completions (marker appears →
    # generation truncates). This could create an artifactual length↔|t|
    # correlation that trips the CB6 generic register flag and downgrades a
    # true positive to "inconclusive". The semantic comedy-cluster sub-check
    # is robust to this (it uses the persona NAME, not outputs).
    # The deviation is surfaced explicitly in `analysis.json` under
    # `register_diagnostic_source` and `register_diagnostic_deviation_note` so
    # the analyzer / clean-result MUST flag it in the Confidence-Why section.
    # Follow-up v2 should add the base-model pass (Option D1).
    arm1_completions_joint = _read_json(OUTPUT_DIR / f"arm1_completions_joint_{A}_{B}.json")
    register = _register_diagnostic(arm1_completions_joint, bystanders, abs_t)
    register["source"] = "joint_lora"
    register["plan_spec_source"] = "base_model"
    register["deviation_note"] = (
        "Register diagnostic computed on joint-LoRA Arm 1 completions, not on "
        "BASE-model outputs as the plan §3/§4.4/§4.12 specifies. Rationale: v1-LOW "
        "pre-commit avoids the ~0.2 GPU-h extra base-pass cost. Risk: joint-LoRA "
        "truncation when [ZLT] appears could create an artifactual length↔|t| "
        "correlation that downgrades a true PASS to 'inconclusive'. "
        "The semantic comedy-cluster sub-check (uses persona NAME, not outputs) "
        "is robust to this; the generic length / punct / FK flags are not. "
        "Surface this in the clean-result Confidence-Why bullet."
    )

    # === H1 verdict (we don't know fixed_b_pass yet; that's Stage 8 — placeholder) ===
    h1_verdict_provisional = _h1_verdict(
        p_primary,
        rho_primary,
        sign_agreement,
        register["register_confound_flag"],
        register["semantic_register_confound_flag"],
        fixed_b_pass=None,  # Stage 8 will overwrite verdict in null_distributions.json
    )

    # === Saturation / register confidence caps ===
    saturation_degraded = n_saturated > SATURATION_MAX_EXCLUSIONS
    confidence_cap = "LOW"  # pre-committed (CB12)
    notes = []
    if saturation_degraded:
        notes.append(f"saturation_degraded: {n_saturated} bystanders excluded (>3)")
    if not source_rate_sanity["pass"]:
        notes.append(
            f"source_rate_sanity FAILED (plan §6 C0a-C0c): "
            f"joint_A={source_rate_joint_A:.3f} joint_B={source_rate_joint_B:.3f} "
            f"Aonly_A={source_rate_Aonly:.3f} Bonly_B={source_rate_Bonly:.3f} "
            f"(threshold={SOURCE_RATE_MIN:.2f})"
        )
    if register["register_confound_flag"]:
        notes.append("register_confound_flag set (CB6)")
    if register["semantic_register_confound_flag"]:
        notes.append("semantic_register_confound_flag set (Fix 3b)")

    result = {
        "pair": [A, B],
        "n_bystanders": n,
        "bystanders": bystanders,
        "t_vals": t_vals.tolist(),
        "s_vals": s_vals.tolist(),
        "rates_per_persona": {
            "joint": rates_joint.tolist(),
            "Aonly": rates_A.tolist(),
            "Bonly": rates_B.tolist(),
        },
        "bernoulli_union_per_persona": bernoulli_union.tolist(),
        "r_p_primary_per_persona": r_p_primary.tolist(),
        "saturation_mask": saturation_mask.tolist(),
        "n_saturated": n_saturated,
        "n_keep": n_keep,
        "saturation_degraded": saturation_degraded,
        "source_rate_sanity": source_rate_sanity,
        "h1_primary": {
            "baseline": "bernoulli_union",
            "rho": rho_primary,
            "p": p_primary,
            "alternative": "less",
            "n": n_keep,
        },
        "h1_sensitivity_additive": {"rho": rho_add, "p": p_add, "alternative": "less"},
        "h1_sensitivity_max": {"rho": rho_max, "p": p_max, "alternative": "less"},
        "sign_agreement_all_negative": sign_agreement,
        "collinearity_gate_fired": collin["gate_fired"],
        "stratified_mann_whitney": (
            {"stat": mw_stat, "p": mw_p, "alternative": "greater"} if collin["gate_fired"] else None
        ),
        "h2_descriptive": {
            "argmax_persona": bystanders[argmax_idx],
            "argmax_rank": argmax_rank,
            "rank_max": n - 1,
            "status": h2_status,
        },
        "register_diagnostic": register,
        "h1_verdict_provisional": h1_verdict_provisional,
        "confidence_cap": confidence_cap,
        "notes": notes,
        "metadata": _run_metadata(),
    }
    _write_json(OUTPUT_DIR / "analysis.json", result)
    return 0


# ── Stage 8: shuffled-axis null + fixed-B null ─────────────────────────────


def stage_null_shuffle(args: argparse.Namespace) -> int:
    """Stage 8: 1000-perm random-axis null (Null A) + fixed-B null (Null B)."""
    setup_logging("8_null_shuffle")

    from explore_persona_space.personas import PERSONAS_19

    pair = _read_json(OUTPUT_DIR / "pair_selection.json")
    A, B = pair["A"], pair["B"]
    bystanders = pair["bystanders"]

    centroids = _load_centered_l20()[0]
    analysis = _read_json(OUTPUT_DIR / "analysis.json")
    rho_primary = analysis["h1_primary"]["rho"]
    p_primary = analysis["h1_primary"]["p"]
    rates_joint = {
        p: float(r) for p, r in zip(bystanders, analysis["rates_per_persona"]["joint"], strict=True)
    }
    rates_A = {
        p: float(r) for p, r in zip(bystanders, analysis["rates_per_persona"]["Aonly"], strict=True)
    }
    rates_B = {
        p: float(r) for p, r in zip(bystanders, analysis["rates_per_persona"]["Bonly"], strict=True)
    }

    s_vals_real = np.array([pair["s_vals"][p] for p in bystanders])
    r_p_primary = np.array(
        [rates_joint[p] - (rates_A[p] + rates_B[p] - rates_A[p] * rates_B[p]) for p in bystanders]
    )
    keep_real = np.array(
        [
            (rates_A[p] + rates_B[p] - rates_A[p] * rates_B[p]) < SATURATION_THRESHOLD
            for p in bystanders
        ]
    )

    # === Null A: 1000 random alternative pairs ===
    candidate_pool = [p for p in list(PERSONAS_19.keys()) if p not in (A, B, "helpful_assistant")]
    rng = random.Random(SEED)
    null_a_rhos = []
    null_a_n = N_PERMS
    logger.info("Running Null A: %d random alternative pairs (centered cosine)", null_a_n)
    for _ in range(null_a_n):
        # Sample two distinct alt personas; bystanders for this perm = bystander_set
        # of the real pair (Fix 1: condition on real-pair's s_vals AND bystander labels).
        A_alt, B_alt = rng.sample(candidate_pool, 2)
        cos_pA_alt = {p: _centered_cosine(centroids[p], centroids[A_alt]) for p in bystanders}
        cos_pB_alt = {p: _centered_cosine(centroids[p], centroids[B_alt]) for p in bystanders}
        t_alt = np.array([0.5 * (cos_pA_alt[p] - cos_pB_alt[p]) for p in bystanders])
        rho_alt, _ = _partial_spearman(
            r_p_primary[keep_real],
            np.abs(t_alt)[keep_real],
            s_vals_real[keep_real],
            alternative="less",
        )
        null_a_rhos.append(rho_alt)
    null_a_rhos = np.array(null_a_rhos)
    null_a_percentile = float((null_a_rhos <= rho_primary).sum() / null_a_n)
    null_a_pass = null_a_percentile <= NULL_RANDOM_PASS_PERCENTILE

    # === Null B: fixed-comedian null (Fix 3a). Only when B == "comedian". ===
    null_b_rhos: list[float] = []
    null_b_percentile: float | None = None
    null_b_pass: bool | None = None
    if B == "comedian":
        fixed_b_pool = [p for p in PERSONAS_19 if p not in (A, B, "helpful_assistant")]
        logger.info("Running Null B (fixed-B=comedian): %d alt-A pairs", len(fixed_b_pool))
        for A_alt in fixed_b_pool:
            bystanders_alt = [p for p in PERSONAS_19 if p not in (A_alt, B)]
            cos_pA_alt = {
                p: _centered_cosine(centroids[p], centroids[A_alt]) for p in bystanders_alt
            }
            cos_pB_alt = {p: _centered_cosine(centroids[p], centroids[B]) for p in bystanders_alt}
            t_alt = np.array([0.5 * (cos_pA_alt[p] - cos_pB_alt[p]) for p in bystanders_alt])
            s_alt = np.array([0.5 * (cos_pA_alt[p] + cos_pB_alt[p]) for p in bystanders_alt])
            # Reconstruct r_p_alt for alt-pair using FROZEN rates for personas in
            # bystander_set ∩ bystanders_alt. Personas that were sources in the
            # real pair (A_real, B_real = A, B) are exposed in bystanders_alt iff
            # we vary A; B is fixed = real B = comedian, so it never re-enters.
            # A_real (medical_doctor) re-enters when A_alt != A_real.
            rates_joint_alt = np.array([rates_joint.get(p, np.nan) for p in bystanders_alt])
            rates_A_alt_lora = np.array([rates_A.get(p, np.nan) for p in bystanders_alt])
            rates_B_alt_lora = np.array([rates_B.get(p, np.nan) for p in bystanders_alt])
            bernoulli_alt = (
                rates_A_alt_lora + rates_B_alt_lora - rates_A_alt_lora * rates_B_alt_lora
            )
            r_p_alt = rates_joint_alt - bernoulli_alt
            keep_alt = ~np.isnan(r_p_alt) & (bernoulli_alt < SATURATION_THRESHOLD)
            if keep_alt.sum() < 5:
                # Not enough data points for a meaningful Spearman.
                continue
            rho_alt, _ = _partial_spearman(
                r_p_alt[keep_alt],
                np.abs(t_alt)[keep_alt],
                s_alt[keep_alt],
                alternative="less",
            )
            null_b_rhos.append(rho_alt)
        if null_b_rhos:
            arr = np.array(null_b_rhos)
            null_b_percentile = float((arr <= rho_primary).sum() / len(null_b_rhos))
            null_b_pass = null_b_percentile <= NULL_FIXED_B_PASS_PERCENTILE

    # === Updated H1 verdict including fixed-B condition ===
    sign_agreement = analysis["sign_agreement_all_negative"]
    register_flag = analysis["register_diagnostic"]["register_confound_flag"]
    semantic_flag = analysis["register_diagnostic"]["semantic_register_confound_flag"]
    h1_verdict_final = _h1_verdict(
        p_primary,
        rho_primary,
        sign_agreement,
        register_flag,
        semantic_flag,
        fixed_b_pass=null_b_pass,
    )

    result = {
        "pair": [A, B],
        "rho_primary_real": rho_primary,
        "p_primary_real": p_primary,
        "null_a_random_axis": {
            "n_perms": null_a_n,
            "rhos": null_a_rhos.tolist(),
            "percentile_rank": null_a_percentile,
            "pass_threshold": NULL_RANDOM_PASS_PERCENTILE,
            "pass": null_a_pass,
            "conditioning": "s_vals of REAL pair (Fix 1)",
        },
        "null_b_fixed_b": {
            "applicable": B == "comedian",
            "n_perms": len(null_b_rhos),
            "rhos": null_b_rhos,
            "percentile_rank": null_b_percentile,
            "pass_threshold": NULL_FIXED_B_PASS_PERCENTILE,
            "pass": null_b_pass,
            "conditioning": "alt-pair s_alt (each alt has its own nuisance)",
        },
        "h1_verdict_final": h1_verdict_final,
        "confidence_cap": "LOW",
        "metadata": _run_metadata(),
    }
    _write_json(OUTPUT_DIR / "null_distributions.json", result)
    logger.info(
        "Null A percentile rank: %.4f (pass=%s); Null B percentile: %s (pass=%s); "
        "H1 final verdict: %s",
        null_a_percentile,
        null_a_pass,
        null_b_percentile,
        null_b_pass,
        h1_verdict_final,
    )
    return 0


# ── all: full pipeline ─────────────────────────────────────────────────────


def stage_all(args: argparse.Namespace) -> int:
    setup_logging("all")
    sequence = [
        ("preflight", stage_preflight),
        ("extract-base", stage_extract_base),
        ("pick-pair", stage_pick_pair),
        ("collin-gate", stage_collin_gate),
        ("gen-onpolicy", stage_gen_onpolicy),
        ("build-data", stage_build_data),
        ("train", stage_train),
        ("post-cos-gate", stage_post_cos_gate),
        ("eval-arm1", stage_eval_arm1),
        ("eval-arm2", stage_eval_arm2),
        ("analyze", stage_analyze),
        ("null-shuffle", stage_null_shuffle),
    ]
    for name, fn in sequence:
        logger.info("=== Running stage: %s ===", name)
        rc = fn(args)
        if rc == 2:
            # rc=2 is a PLANNED halt (today: Stage 4.5 gate fired). The stage
            # function has already posted the epm:gate-decision-needed marker
            # + applied status:blocked. Stop the pipeline cleanly here so the
            # pod can be paused for user decision.
            logger.error(
                "Stage %s exited with rc=2 (PLANNED HALT: user decision needed). "
                "epm:gate-decision-needed v1 posted on issue #%d; "
                "status:blocked label applied. Pipeline halted.",
                name,
                ISSUE_NUMBER,
            )
            return rc
        if rc != 0:
            logger.error("Stage %s exited with rc=%d; halting pipeline", name, rc)
            return rc
    logger.info("All stages completed.")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Issue #311 orchestrator")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id for single-GPU stages")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even if their output exists on disk",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For stage `preflight` and `pick-pair`: do not touch GPU / heavy resources.",
    )

    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("preflight", help="Stage -1: vLLM dep preflight + smoke test")
    sub.add_parser("extract-base", help="Stage 0: extract base-model centroids at L10+L20")
    sub.add_parser("pick-pair", help="Stage 1: pick TOP-1 lowest-cos source pair")
    sub.add_parser("collin-gate", help="Stage 1.5: |Pearson(|t|, s)| collinearity gate")
    sub.add_parser("gen-onpolicy", help="Stage 2: vLLM on-policy completions for A, B")
    sub.add_parser("build-data", help="Stage 3: joint=800 + A-only=400 + B-only=400 datasets")
    sub.add_parser("train", help="Stage 4: train 3 LoRAs (joint, A-only, B-only)")
    sub.add_parser(
        "post-cos-gate",
        help="Stage 4.5: post-train cos(v_A, v_B) all-or-nothing halt gate (CB3)",
    )
    sub.add_parser("eval-arm1", help="Stage 5: vLLM eval across 19 personas under each LoRA")
    sub.add_parser("eval-arm2", help="Stage 6: BASE+steering hook, 11 arms, DESCRIPTIVE")
    sub.add_parser("analyze", help="Stage 7: Bernoulli partial Spearman + sensitivity + register")
    sub.add_parser("null-shuffle", help="Stage 8: random-axis null (1000) + fixed-B null")
    sub.add_parser("all", help="Run all stages sequentially")
    return parser


_STAGE_DISPATCH = {
    "preflight": stage_preflight,
    "extract-base": stage_extract_base,
    "pick-pair": stage_pick_pair,
    "collin-gate": stage_collin_gate,
    "gen-onpolicy": stage_gen_onpolicy,
    "build-data": stage_build_data,
    "train": stage_train,
    "post-cos-gate": stage_post_cos_gate,
    "eval-arm1": stage_eval_arm1,
    "eval-arm2": stage_eval_arm2,
    "analyze": stage_analyze,
    "null-shuffle": stage_null_shuffle,
    "all": stage_all,
}


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    fn = _STAGE_DISPATCH[args.stage]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
