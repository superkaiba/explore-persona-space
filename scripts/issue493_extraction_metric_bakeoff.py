"""Issue #493 — extraction-point × metric × layer bake-off for marker-transfer prediction.

Re-extract base-model (Qwen-2.5-7B-Instruct) residual activations under each of
the 16 #406 transformation prompts at THREE extraction points, compute a panel
of pairwise distance metrics for all 240 ordered pairs at the inherited
{0, 5, 11, 15, 21, 27} ∪ {7, 14} layer set, regress each predictor against
the #474 already-measured on-policy marker-transfer DV (`delta_g` =
trained − base log P(` ※`) at the post-response slot, plus the base-prior-safe
secondary `g_logprob`), and select the best predictor by leave-one-context-out
CV that also survives the non-stylized panel and the base-prior-safe check.

Substrate (READ, never recomputed):
  - eval_results/issue_474/cross_eval/{arm}_ep{ep}/G_logprob_matrix.json
      G[a][b]["delta_g"]   — primary DV
      G[a][b]["g_logprob"] — base-prior-safe secondary DV
      8 cells: arm∈{pos,loc} × ep∈{1,2,3,5}; headline = loc_ep1
  - eval_results/issue_406/divergence/D_matrix.json
      ["prompt_tokens"] — length covariate for the partial Spearman
  - eval_results/issue_406/cosine/C_L{0,5,11,15,21,27}.json
      ["matrix"] — existing last-prompt-token cosine for the correctness
      cross-check (re-implemented last-token cosine must reproduce these
      within tolerance)

Extraction points:
  (1) end_of_system  — residual at the last token of the system-prompt-only
      prefix (causal attention → input-independent → ONE vector per
      transformation). Cloud metrics (MMD, C2ST, Δ-spectrum, Gaussian-KL/W2)
      are N/A here — explicitly None, never a forced value.
  (2) last_prompt    — residual at the last input token after the user
      question (input_ids.shape[1] - 1). One vector per (transformation,
      question) → a cloud per transformation. Reproduces the existing cosine.
  (3) mean_response  — greedy-decode a response per (transformation,
      question), mean-pool the residual activations across its response
      tokens. One vector per (transformation, question).

Metrics (computed per layer):
  - cosine     — cosine distance of mean activation centroids (1 − cos_sim)
  - euclidean  — L2 distance of centroids
  - mahal      — Mahalanobis-on-pooled-cov centroid distance (PCA-whitened)
  - fisher     — PCA-whitened mean-difference Fisher distance (n≪d safe via
                 dual/Gram PCA; never inverts a 3584×3584 covariance)
  - mmd        — RBF-MMD² (median-heuristic bandwidth, permutation null)
  - c2st       — held-out linear-probe AUC (logistic regression, 5-fold)
  - delta_spec — paired Δ-spectrum: ‖mean Δ‖, coherence, effective dim
                 (Δ_i = h_b(Q_i) − h_a(Q_i), same probe questions, matched
                 ordering, PCA on the per-question displacements)
  - gauss_kl   — Gaussian symmetric-KL in the top-k PCA subspace
  - wass2      — Bures-Wasserstein² between Gaussians in the top-k PCA subspace

Regression:
  - Length-partial Spearman ρ (rank-residualize on log prompt_tokens), per the
    #474 / #406 convention via _length_partial.
  - DVs: ΔG (primary), g_logprob (base-prior-safe secondary).
  - Panels: non-stylized n=156 (drops any pair touching A3/A4/A5 = pirate,
    comedian, villain) + full n=240.
  - Per (arm, epoch); headline loc_ep1; saturation fraction logged per cell.

Winner selection (avoids in-sample max-|ρ| upward bias):
  - Leave-one-context-out CV criterion (the i474 fig9 pattern, generalized):
    for each predictor, leave out all pairs touching one of the 16 conditions
    in turn, fit OLS on the remainder, predict held-out, compute CV-R².
  - Winner = highest CV-R² predictor that ALSO (a) survives on the
    non-stylized panel (ρ same sign as full panel and |ρ| > floor) and
    (b) survives the base-prior-safe (g_logprob) check.
  - Emit the FULL grid (every metric × extraction-point × layer × DV ρ, p,
    CV-R²) so the search is transparent.

Checkpoint-per-phase: each extraction point's activations land on disk the
moment they're computed; each (layer × metric) distance matrix lands the
moment it's computed; each (arm, epoch) regression grid lands the moment
it's computed. A mid-run crash never throws away earlier work.

GPU note: the dev VM has NO GPU. The full extraction must run on a pod
(1× H100, intent ``eval``). Subset flags (``--transformations``,
``--n-probes``, ``--layers``, ``--extraction-points``, ``--arms``,
``--epochs``) keep a tiny pod-smoke cheap. Pure metric + regression
sanity (with synthetic activations) runs on the VM via ``--dry-run``.

Outputs:
  eval_results/issue_493/bakeoff/
    activations/{point}__layer{L}.pt     — per-extraction-point, per-layer
                                            cloud (n_cond, n_q, hidden)
    metrics/{point}__layer{L}__{metric}.json — per-(point, layer, metric)
                                            distance matrix (n_cond, n_cond)
    regression/{arm}_ep{ep}.json         — per-cell full predictor grid
    bakeoff_grid.json                    — winner + the full search
    meta.json                            — git commit, env, timestamps

Figures:
  figures/issue_493/
    metric_layer_grid_heatmap.{png,pdf}  — full ρ grid (rows = metric × point,
                                            cols = layer), loc_ep1 non-stylized
    winner_scatter_vs_deltaG.{png,pdf}   — winner's pair-level scatter

See the task body for the methodology guards (end-of-system → cloud metrics
N/A; n≪d → PCA-reduce first; Δ-spectrum is paired; last-token cosine cross-
check).
"""

from __future__ import annotations

# Greek + special characters (ρ, Δ, ×, →, etc.) appear in this file's prose
# for research notation. Matches the same suppression in scripts/eval_issue475,
# scripts/gen_issue475_scaffold_data, scripts/issue404_predictor_kldiv, etc.
# ruff: noqa: RUF001, RUF002, RUF003
import argparse
import gc
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

logger = logging.getLogger("i493.bakeoff")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAKEOFF_DIR = PROJECT_ROOT / "eval_results" / "issue_493" / "bakeoff"
ACT_DIR = BAKEOFF_DIR / "activations"
METRIC_DIR = BAKEOFF_DIR / "metrics"
REGR_DIR = BAKEOFF_DIR / "regression"
FIGURE_DIR = PROJECT_ROOT / "figures" / "issue_493"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Inherited from #406/#474 (existing cosine matrices live at these layers);
# the spec adds {7, 14} to broaden the layer sweep without dropping coverage.
INHERITED_LAYERS: tuple[int, ...] = (0, 5, 11, 15, 21, 27)
NEW_LAYERS: tuple[int, ...] = (7, 14)
DEFAULT_LAYERS: tuple[int, ...] = tuple(sorted(set(INHERITED_LAYERS) | set(NEW_LAYERS)))

DEFAULT_EXTRACTION_POINTS: tuple[str, ...] = (
    "end_of_system",
    "last_prompt",
    "mean_response",
)
# Cloud metrics need ≥ 2 points per side; N/A at end_of_system (one vec per cond).
# (Fisher-on-pooled-cov is mathematically identical to Mahalanobis for the
# 2-cloud case, so we don't list it as a separate predictor row — the spec
# called both out by name but they collapse here. The docstring + report note
# the equivalence.)
CLOUD_METRICS: tuple[str, ...] = (
    "mmd",
    "c2st",
    "delta_spec",
    "gauss_kl",
    "wass2",
)
# Centroid metrics work everywhere. `mahal` is per-pair pooled (cloud regime);
# `mahal_pooled_ctx` is context-pooled (single-vector / end_of_system regime).
CENTROID_METRICS: tuple[str, ...] = ("cosine", "euclidean", "mahal", "mahal_pooled_ctx")
# Predictor-variant axis: "raw" + "centered" (prompt-centered, subtracts the
# mean over all in-scope contexts' centroids before distance).
PREDICTOR_VARIANTS: tuple[str, ...] = ("raw", "centered")
ALL_METRICS: tuple[str, ...] = CENTROID_METRICS + CLOUD_METRICS

DEFAULT_ARMS: tuple[str, ...] = ("pos", "loc")
DEFAULT_EPOCHS: tuple[int, ...] = (1, 2, 3, 5)

# Methodology guards
STY_CIDS: frozenset[str] = frozenset({"A3", "A4", "A5"})
PCA_DEFAULT_K: int = 16  # rank cap for covariance-based metrics (n=50 ≫ k=16)
MMD_PERMUTATIONS: int = 200
C2ST_FOLDS: int = 5

# Saturation thresholds (match i474_cosine_followup convention)
SATURATION_GLOGP_THRESHOLD: float = -0.1
COSINE_REPRO_TOLERANCE: float = 1e-3  # cross-check vs existing C_L*.json


# ───────────────────────── repro metadata ─────────────────────────


def _git_sha() -> str:
    """Return current git HEAD SHA, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _env_versions() -> dict[str, str]:
    """Capture core dep versions for the reproducibility metadata block."""
    out = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in ("numpy", "scipy", "torch", "transformers", "sklearn"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[pkg] = "not-installed"
    return out


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write payload to path.tmp then rename — never half-written files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# ───────────────────────── substrate loaders ─────────────────────────


def _load_G(arm: str, ep: int) -> dict:
    """Read #474's already-measured G_logprob matrix for one (arm, ep) cell."""
    p = PROJECT_ROOT / f"eval_results/issue_474/cross_eval/{arm}_ep{ep}/G_logprob_matrix.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing #474 G matrix at {p}; the substrate must be present.")
    return json.loads(p.read_text())["G"]


def _load_prompt_tokens() -> dict[str, dict[str, int]]:
    """Read #406's pair-level prompt-token counts (length covariate)."""
    p = PROJECT_ROOT / "eval_results/issue_406/divergence/D_matrix.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing #406 D_matrix at {p}; the substrate must be present.")
    return json.loads(p.read_text())["prompt_tokens"]


def _load_existing_cosine_matrices(layers: tuple[int, ...]) -> dict[int, dict]:
    """Read existing #406 last-prompt-token cosine matrices (for cross-check)."""
    out = {}
    for L in layers:
        p = PROJECT_ROOT / f"eval_results/issue_406/cosine/C_L{L}.json"
        if not p.exists():
            logger.warning("No existing cosine matrix at L%d (skipping cross-check)", L)
            continue
        out[L] = json.loads(p.read_text())
    return out


# ───────────────────────── extraction phase ─────────────────────────


def _ensure_class_d_rewrites() -> dict:
    """Load class-D rewrites; only required if Class D is in the active conds."""
    from explore_persona_space.experiments.i460_data import load_class_d_rewrites

    return load_class_d_rewrites()


def _load_probe_questions() -> list[str]:
    """Load the EXACT #406/#474 50-question Q_test probe set (matched dist)."""
    from explore_persona_space.experiments.i460_data import load_q_test_extended_50

    qs = load_q_test_extended_50()
    if len(qs) != 50:
        raise AssertionError(f"Expected 50 Q_test probes, got {len(qs)}")
    return qs


def _build_prompts_for_extraction(
    cond,
    question: str,
    tokenizer,
    class_d_rewrites: dict,
    extraction_point: str,
) -> tuple[str, str | None]:
    """Build (system_only_prefix, full_prompt) for one (cond, question).

    Returns
    -------
    (system_text, full_text)
      system_text  — the system-prompt-only prefix tokenized form (or None
                     for Class B / C1 / D which carry no system message; the
                     end_of_system extraction MUST be skipped for these).
      full_text    — the full prompt the model sees (user turn appended,
                     add_generation_prompt=True), used for last_prompt and
                     mean_response.
    """
    from explore_persona_space.experiments.i406_conditions import build_prompt_for_condition

    full_text = build_prompt_for_condition(
        cond, question, tokenizer, class_d_rewrites=class_d_rewrites
    )

    # The system-only prefix is well-defined ONLY for Class A (which carries
    # a non-trivial system message). All other classes don't inject a
    # system prompt, so end_of_system extraction is N/A by construction
    # for Class B / C1 / D — return None.
    if cond.cls == "A":
        system_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": cond.system_prompt}],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        system_text = None
    return system_text, full_text


def _extract_one(  # noqa: C901 — dispatches across 3 extraction points; flattening would just inline the branches.
    model,
    tokenizer,
    *,
    device,
    cond,
    question: str,
    class_d_rewrites: dict,
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
    max_response_tokens: int,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict]:
    """For one (cond, question) extract residual activations at the requested
    extraction points × layers.

    Returns
    -------
    (result, meta)
      result: ``{point: {layer: tensor(H,) for layer in layers} for point in extraction_points}``.
        For ``end_of_system`` on non-Class-A conditions the inner dict is
        empty (signals N/A at this (cond, point); the cloud aggregator drops it).
      meta: ``{"truncated": bool, "response_len": int, "response_present": bool}``.
        `truncated` is True iff the greedy generation ran to
        `max_response_tokens` without emitting EOS — caller logs the rate
        so a bias toward early tokens in `mean_response` is visible.
    """
    import torch

    system_text, full_text = _build_prompts_for_extraction(
        cond, question, tokenizer, class_d_rewrites, "all"
    )

    result: dict[str, dict[int, torch.Tensor]] = {p: {} for p in extraction_points}
    meta: dict = {"truncated": False, "response_len": 0, "response_present": False}

    # ── end_of_system (Class A only) ──
    if "end_of_system" in extraction_points and system_text is not None:
        ids = tokenizer(system_text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model(
                input_ids=ids["input_ids"],
                attention_mask=ids["attention_mask"],
                output_hidden_states=True,
            )
        seq_len = ids["input_ids"].shape[1]
        last_pos = seq_len - 1
        # hidden_states is a (n_layers+1,) tuple: index 0 = embedding output,
        # index L+1 = OUTPUT of transformer block L. #406 captured cosine via
        # `model.model.layers[L].register_forward_hook` (the OUTPUT of block L),
        # so to match the #406/C_L*.json convention we read hidden_states[L+1].
        # Off-by-one bug from round 1 (was indexing hidden_states[L]) is fixed
        # here; the cosine cross-check in `reproduce_last_token_cosine_check`
        # asserts the result lands within tolerance of #406's matrices.
        for L in layers:
            if len(out.hidden_states) <= L + 1:
                raise IndexError(
                    f"layer={L} -> hidden_states[{L + 1}] out of range; "
                    f"hidden_states has {len(out.hidden_states)} entries"
                )
            result["end_of_system"][L] = out.hidden_states[L + 1][0, last_pos, :].float().cpu()
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── last_prompt + mean_response share one forward (with generation) ──
    need_last = "last_prompt" in extraction_points
    need_resp = "mean_response" in extraction_points
    if need_last or need_resp:
        prompt_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
        prompt_len = prompt_ids["input_ids"].shape[1]

        if need_resp:
            # Greedy-decode (temp=0) — match the #460/#474 R-generation
            # convention. Capped at max_response_tokens; we log truncation
            # rate when it matters (not here — per-(cond, q) call site).
            with torch.no_grad():
                gen_out = model.generate(
                    **prompt_ids,
                    max_new_tokens=max_response_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_hidden_states=False,
                )
            full_ids = gen_out.sequences  # (1, prompt_len + n_new)
            response_len = full_ids.shape[1] - prompt_len
            meta["response_len"] = int(response_len)
            # Truncation = ran to cap without emitting EOS at the last new
            # token. (For Qwen-2.5-7B, eos_token_id; we accept pad as EOS
            # only when pad was explicitly aliased to it above.)
            eos_id = tokenizer.eos_token_id
            last_new = int(full_ids[0, -1].item()) if response_len > 0 else None
            meta["truncated"] = bool(response_len == max_response_tokens and last_new != eos_id)
            meta["response_present"] = response_len > 0
            if response_len <= 0:
                # Edge case: model emitted EOS immediately. Treat as zero
                # response tokens — mean_response falls back to NaN; the
                # cloud aggregator drops NaN rows downstream.
                logger.warning(
                    "cond=%s q=%r emitted zero response tokens; mean_response N/A this row",
                    cond.cid,
                    question[:40],
                )
                if need_last:
                    # Still get last_prompt with a single forward pass.
                    with torch.no_grad():
                        fwd = model(
                            input_ids=prompt_ids["input_ids"],
                            attention_mask=prompt_ids["attention_mask"],
                            output_hidden_states=True,
                        )
                    # hidden_states[L+1] = output of block L (match #406 hook).
                    for L in layers:
                        result["last_prompt"][L] = (
                            fwd.hidden_states[L + 1][0, prompt_len - 1, :].float().cpu()
                        )
                    del fwd
                if need_resp:
                    H = model.config.hidden_size
                    for L in layers:
                        result["mean_response"][L] = torch.full(
                            (H,), float("nan"), dtype=torch.float32
                        )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return result, meta

            # One teacher-forced forward pass over the FULL sequence (prompt
            # + decoded response) to get hidden states at every position.
            # This avoids re-decoding logits and gives us last_prompt +
            # mean_response in one shot.
            attn = torch.ones_like(full_ids)
            with torch.no_grad():
                fwd = model(
                    input_ids=full_ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                )
            # hidden_states[L+1] = output of transformer block L (match #406 hook).
            for L in layers:
                hs = fwd.hidden_states[L + 1][0]  # (full_len, H)
                if need_last:
                    result["last_prompt"][L] = hs[prompt_len - 1, :].float().cpu()
                if need_resp:
                    resp_slice = hs[prompt_len : prompt_len + response_len, :]
                    result["mean_response"][L] = resp_slice.mean(dim=0).float().cpu()
            del fwd, full_ids, gen_out
        elif need_last:
            with torch.no_grad():
                fwd = model(
                    input_ids=prompt_ids["input_ids"],
                    attention_mask=prompt_ids["attention_mask"],
                    output_hidden_states=True,
                )
            # hidden_states[L+1] = output of transformer block L (match #406 hook).
            for L in layers:
                result["last_prompt"][L] = (
                    fwd.hidden_states[L + 1][0, prompt_len - 1, :].float().cpu()
                )
            del fwd

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result, meta


def run_extraction(  # noqa: C901 — top-level dispatcher (model load + per-(cond,q) loop + per-(point,layer) checkpointing).
    *,
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
    transformations: tuple[str, ...] | None,
    n_probes: int,
    max_response_tokens: int,
    device: str,
    overwrite: bool,
) -> dict[str, dict[int, np.ndarray]]:
    """Top-level extraction loop. Checkpoints per (extraction_point, layer)
    immediately on completion.

    Returns
    -------
    dict
      ``{point: {layer: ndarray(n_cond, n_q, H)}}`` — for ``end_of_system``,
      shape is ``(n_class_A, 1, H)`` after dropping non-A conds (which carry
      no system message). Non-A conds for end_of_system are explicitly
      ABSENT, never zero-filled.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.i406_conditions import CONDITIONS, CONDITIONS_BY_ID

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for extraction; run on a pod with a GPU.")

    # Pick the active condition set.
    if transformations:
        active_conds = [CONDITIONS_BY_ID[c] for c in transformations]
    else:
        active_conds = list(CONDITIONS)
    logger.info("Active transformations: %s", [c.cid for c in active_conds])

    all_questions = _load_probe_questions()
    if n_probes < len(all_questions):
        questions = all_questions[:n_probes]
        logger.info("Subsetting probes: %d / %d", len(questions), len(all_questions))
    else:
        questions = all_questions

    class_d_rewrites = _ensure_class_d_rewrites() if any(c.cls == "D" for c in active_conds) else {}

    # Model load
    logger.info("Loading %s on %s", BASE_MODEL, device)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    logger.info("Base model loaded in %.1fs", time.time() - t0)

    H = model.config.hidden_size

    # Aggregate clouds per (point, layer): list of (cond_idx, q_idx) -> vec
    # We store as nested dicts then convert to ndarray at write time.
    clouds: dict[str, dict[int, dict[tuple[int, int], np.ndarray]]] = {
        p: {L: {} for L in layers} for p in extraction_points
    }
    truncation_count = 0
    total_response_rows = 0

    response_len_samples: list[int] = []
    for ci, cond in enumerate(active_conds):
        t_c = time.time()
        for qi, q in enumerate(questions):
            try:
                row, meta = _extract_one(
                    model,
                    tokenizer,
                    device=device,
                    cond=cond,
                    question=q,
                    class_d_rewrites=class_d_rewrites,
                    extraction_points=extraction_points,
                    layers=layers,
                    max_response_tokens=max_response_tokens,
                )
            except Exception as e:
                raise RuntimeError(f"Extraction failed at cond={cond.cid} q_idx={qi}: {e}") from e
            for pt in extraction_points:
                if pt == "end_of_system" and not row[pt]:
                    continue  # non-A cond → N/A by construction
                for L in layers:
                    if L in row[pt]:
                        clouds[pt][L][(ci, qi)] = row[pt][L].numpy()
            if "mean_response" in extraction_points and meta.get("response_present"):
                total_response_rows += 1
                response_len_samples.append(meta["response_len"])
                if meta.get("truncated"):
                    truncation_count += 1
        logger.info(
            "cond %d/%d %s in %.1fs", ci + 1, len(active_conds), cond.cid, time.time() - t_c
        )

    if total_response_rows:
        med = int(np.median(response_len_samples)) if response_len_samples else 0
        mx = int(np.max(response_len_samples)) if response_len_samples else 0
        logger.info(
            "Response truncation rate: %d/%d (%.1f%%); response_len median=%d max=%d (cap=%d)",
            truncation_count,
            total_response_rows,
            100.0 * truncation_count / total_response_rows,
            med,
            mx,
            max_response_tokens,
        )
        # Persist the truncation summary so post-run review can see the
        # response-length distribution at a glance.
        BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            BAKEOFF_DIR / "extraction_truncation.json",
            {
                "schema_version": 1,
                "max_response_tokens": int(max_response_tokens),
                "total_response_rows": int(total_response_rows),
                "truncation_count": int(truncation_count),
                "truncation_rate": float(truncation_count / total_response_rows),
                "response_len_median": med,
                "response_len_max": mx,
                "response_len_p95": int(np.percentile(response_len_samples, 95))
                if response_len_samples
                else 0,
                "git_sha": _git_sha(),
                "timestamp_utc": _now_iso(),
            },
        )

    # Convert and write per-(point, layer) checkpoints.
    written: dict[str, dict[int, np.ndarray]] = {}
    n_cond_active = len(active_conds)
    n_q = len(questions)
    for pt in extraction_points:
        written[pt] = {}
        for L in layers:
            entries = clouds[pt][L]
            if not entries:
                logger.warning("No activations captured for point=%s layer=%d", pt, L)
                continue
            # For end_of_system, the n_q axis collapses to 1 (input-independent
            # under causal attention; we always feed the same system-only
            # prefix per cond). For last_prompt / mean_response, n_q is full.
            if pt == "end_of_system":
                # Build (n_active_A, 1, H); only Class A conds will be present.
                present_cidx = sorted({ci for (ci, _qi) in entries})
                arr = np.full((len(present_cidx), 1, H), np.nan, dtype=np.float32)
                for new_i, ci in enumerate(present_cidx):
                    # qi must be 0 for end_of_system (one vec per cond)
                    # in case multiple qi rows snuck in, average them as a
                    # robustness no-op (they should be identical).
                    rows = [v for (cci, _qi), v in entries.items() if cci == ci]
                    arr[new_i, 0, :] = np.mean(rows, axis=0)
                cond_ids = [active_conds[ci].cid for ci in present_cidx]
            else:
                arr = np.full((n_cond_active, n_q, H), np.nan, dtype=np.float32)
                for (ci, qi), v in entries.items():
                    arr[ci, qi, :] = v
                cond_ids = [c.cid for c in active_conds]
            out_path = ACT_DIR / f"{pt}__layer{L}.pt"
            if out_path.exists() and not overwrite:
                # Subset-cache guard: refuse to use an on-disk checkpoint that
                # was extracted under a different cond_ids / n_probes than the
                # current run requests. Otherwise downstream regression silently
                # uses a stale (but seemingly valid) cache from a previous run.
                cached = torch.load(out_path, map_location="cpu", weights_only=False)
                cached_cond_ids = list(cached.get("cond_ids", []))
                cached_n_probes = int(cached.get("n_probes", -1))
                if cached_cond_ids != cond_ids or cached_n_probes != arr.shape[1]:
                    raise RuntimeError(
                        f"Subset-cache mismatch at {out_path}: cached "
                        f"cond_ids={cached_cond_ids} n_probes={cached_n_probes}, "
                        f"current request cond_ids={cond_ids} n_probes={arr.shape[1]}. "
                        "Re-run with --overwrite to invalidate the cache, or restore "
                        "the matching subset of transformations / --n-probes."
                    )
                logger.info(
                    "Skipping existing %s (matched subset; use --overwrite to redo)",
                    out_path,
                )
            else:
                ACT_DIR.mkdir(parents=True, exist_ok=True)
                # torch.save the numpy array + meta as a small dict.
                torch.save(
                    {
                        "schema_version": 1,
                        "extraction_point": pt,
                        "layer": L,
                        "cond_ids": cond_ids,
                        "n_probes": arr.shape[1],
                        "hidden_size": H,
                        "activations": arr,
                        "git_sha": _git_sha(),
                        "timestamp_utc": _now_iso(),
                    },
                    out_path,
                )
                logger.info("Wrote %s shape=%s", out_path, arr.shape)
            written[pt][L] = arr

    # Clean up GPU.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return written


def load_activations_from_disk(
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
) -> dict[str, dict[int, dict]]:
    """Re-load per-(point, layer) activation checkpoints written by run_extraction.

    Returns
    -------
    dict
      ``{point: {layer: {"cond_ids": [...], "activations": ndarray(n_cond, n_q, H)}}}``
    """
    import torch

    out: dict[str, dict[int, dict]] = {}
    for pt in extraction_points:
        out[pt] = {}
        for L in layers:
            p = ACT_DIR / f"{pt}__layer{L}.pt"
            if not p.exists():
                logger.warning("Missing checkpoint: %s", p)
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            out[pt][L] = d
    return out


# ───────────────────────── metric phase ─────────────────────────


def _pca_topk_via_gram(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Dual / Gram PCA — eigendecompose the n×n Gram, NOT the d×d covariance.

    Safe when n ≪ d (our regime: n = 50 ≪ d = 3584). Returns
    (projected: (n, k), components: (k, d)) for X already mean-centered.

    For X centered with shape (n, d):
      - Gram G = X X^T  (n × n)
      - Eigenvectors V_g (n, k), eigenvalues λ
      - The top-k principal components in the d-space:
            U = X^T V_g / sqrt(λ)        shape (d, k)
      - Projected coords:
            T = V_g * sqrt(λ)            shape (n, k)
    """
    n, d = X.shape
    if k > min(n, d):
        k = min(n, d)
    G = X @ X.T  # (n, n)
    # numerical safety: symmetrize
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    # take top-k by eigenvalue
    order = np.argsort(eigvals)[::-1][:k]
    lam = np.clip(eigvals[order], 1e-12, None)
    V_g = eigvecs[:, order]  # (n, k)
    sqrt_lam = np.sqrt(lam)  # (k,)
    components = (X.T @ V_g) / sqrt_lam[None, :]  # (d, k)
    components = components.T  # (k, d)
    projected = V_g * sqrt_lam[None, :]  # (n, k)
    return projected, components


def _pair_pca_subspace(Xa: np.ndarray, Xb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA-reduce a PAIR of clouds to a shared top-k subspace via the Gram of
    the stacked centered clouds.

    Returns (Ya: (na, k), Yb: (nb, k)) — never inverts a d×d covariance.
    """
    stacked = np.vstack([Xa, Xb])
    mu = stacked.mean(axis=0, keepdims=True)
    stacked_c = stacked - mu
    _proj, comps = _pca_topk_via_gram(stacked_c, k)
    Ya = (Xa - mu) @ comps.T  # (na, k)
    Yb = (Xb - mu) @ comps.T  # (nb, k)
    return Ya, Yb


def _drop_nan_rows(X: np.ndarray) -> np.ndarray:
    """Drop rows that contain any NaN (occurs when a (cond, q) extraction
    failed or end_of_system N/A was zero-filled by mistake)."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {X.shape}")
    mask = ~np.any(np.isnan(X), axis=1)
    return X[mask]


def _centroid_cosine_distance(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """1 − cos_sim between cloud centroids. Reproduces the #406 cosine recipe.

    Both clouds are (n_q, H); centroid = mean across the n_q axis.
    """
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    mu_a = Xa.mean(axis=0)
    mu_b = Xb.mean(axis=0)
    na = np.linalg.norm(mu_a)
    nb = np.linalg.norm(mu_b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - (mu_a @ mu_b) / (na * nb))


def _centroid_euclidean(Xa: np.ndarray, Xb: np.ndarray) -> float:
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    return float(np.linalg.norm(Xa.mean(axis=0) - Xb.mean(axis=0)))


def _centroid_mahal(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """Mahalanobis-on-pooled-cov in the top-k PCA subspace (n≪d-safe).

    Pooled subspace via dual PCA on stacked-centered (Xa, Xb), then pooled
    covariance in the k-d subspace + inverse via solve. Requires na, nb ≥ 2
    (per-pair within-cloud covariance defined); for one-vector-per-cloud
    extraction (end_of_system) use _context_mahal_with_pooled_cov instead.
    """
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    Ya, Yb = _pair_pca_subspace(Xa, Xb, k)
    mu_a = Ya.mean(axis=0)
    mu_b = Yb.mean(axis=0)
    Cov = 0.5 * (np.cov(Ya.T, ddof=1) + np.cov(Yb.T, ddof=1))
    # ridge for numerical stability
    Cov += 1e-6 * np.eye(Cov.shape[0])
    diff = mu_a - mu_b
    inv_cov_diff = np.linalg.solve(Cov, diff)
    return float(np.sqrt(float(diff @ inv_cov_diff)))


def _build_context_pooled_mahal_state(activations: np.ndarray, k: int) -> dict | None:
    """Build the shared state for Mahalanobis-vs-context-pooled-covariance,
    used by the end_of_system extraction point where there is ONE vector per
    transformation (no per-pair covariance possible). The "pooled" covariance
    is computed across all condition centroids; PCA-reduced to the top-k
    subspace to stay well-posed at n_cond << d.

    activations shape: (n_cond, n_q, H). For end_of_system n_q == 1, so the
    per-cond vector IS the centroid.

    Returns
    -------
    None if pooled covariance is singular after PCA reduction (fail-loud
    upstream). Otherwise a dict with the precomputed subspace projection +
    inverse pooled covariance.
    """
    n_cond = activations.shape[0]
    if n_cond < 2:
        return None
    # Build centroid matrix (n_cond, H).
    centroids = np.array([_drop_nan_rows(activations[i]).mean(axis=0) for i in range(n_cond)])
    valid_mask = ~np.any(np.isnan(centroids), axis=1)
    centroids = centroids[valid_mask]
    if len(centroids) < 2:
        return None
    mu = centroids.mean(axis=0, keepdims=True)
    cent_c = centroids - mu
    # Dual / Gram PCA on the n_cond-cloud — never invert (3584, 3584).
    k_eff = min(k, len(centroids) - 1, cent_c.shape[1])
    if k_eff < 1:
        return None
    _proj, comps = _pca_topk_via_gram(cent_c, k_eff)
    Y = cent_c @ comps.T  # (n_cond_valid, k_eff)
    # Pooled covariance across the centroids in the subspace, with ridge.
    cov_sub = np.cov(Y.T, ddof=1) + 1e-6 * np.eye(k_eff)
    try:
        cov_inv = np.linalg.inv(cov_sub)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(cov_inv)):
        return None
    return {
        "mu": mu,
        "components": comps,
        "cov_inv": cov_inv,
        "valid_mask": valid_mask,
    }


def _context_mahal_with_pooled_cov(
    Xa: np.ndarray,
    Xb: np.ndarray,
    state: dict,
    a_idx: int,
    b_idx: int,
) -> float:
    """Mahalanobis distance between the centroids of (Xa, Xb), using the
    pre-built context-pooled covariance in the shared subspace.

    For end_of_system where Xa / Xb each have a single vector this is the
    only meaningful Mahalanobis variant (per-pair covariance is undefined
    when n=1). Returns NaN if either context dropped out of the valid_mask.
    """
    mask = state["valid_mask"]
    if not (mask[a_idx] and mask[b_idx]):
        return float("nan")
    mu = state["mu"]
    comps = state["components"]
    cov_inv = state["cov_inv"]
    cent_a = _drop_nan_rows(Xa).mean(axis=0)
    cent_b = _drop_nan_rows(Xb).mean(axis=0)
    if np.any(np.isnan(cent_a)) or np.any(np.isnan(cent_b)):
        return float("nan")
    ya = (cent_a - mu[0]) @ comps.T
    yb = (cent_b - mu[0]) @ comps.T
    diff = ya - yb
    return float(np.sqrt(float(diff @ cov_inv @ diff)))


def _rbf_kernel_with_bandwidth(Z: np.ndarray) -> tuple[np.ndarray, float]:
    """Build the RBF kernel matrix with median-heuristic bandwidth.

    Returns (K, sigma2) where K[i,j] = exp(-||z_i - z_j||^2 / sigma2) and
    sigma2 is the median pairwise squared distance (excluding the diagonal).
    """
    sq = np.sum(Z**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2 * Z @ Z.T
    np.fill_diagonal(D2, np.nan)
    median_sq = np.nanmedian(D2)
    sigma2 = max(float(median_sq), 1e-8)
    K = np.exp(-D2 / sigma2)
    np.fill_diagonal(K, 1.0)
    return K, sigma2


def _unbiased_mmd2_from_kernel(K: np.ndarray, na: int) -> float:
    """Unbiased MMD² (Gretton et al. 2012, Lemma 6) from a pre-built kernel.

    Excludes the diagonal of K_aa and K_bb so the estimator is unbiased:
        MMD² = (1/(na(na-1))) sum_{i!=j} K_aa[i,j]
             + (1/(nb(nb-1))) sum_{i!=j} K_bb[i,j]
             - (2/(na*nb)) sum_{i,j} K_ab[i,j]
    """
    nb = K.shape[0] - na
    Kaa = K[:na, :na]
    Kbb = K[na:, na:]
    Kab = K[:na, na:]
    sum_aa = Kaa.sum() - np.trace(Kaa)  # off-diagonal sum
    sum_bb = Kbb.sum() - np.trace(Kbb)
    term_aa = sum_aa / (na * (na - 1))
    term_bb = sum_bb / (nb * (nb - 1))
    term_ab = 2 * Kab.mean()
    return float(term_aa + term_bb - term_ab)


def _rbf_mmd_squared(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Unbiased RBF-MMD² with median-heuristic bandwidth (Gretton et al. 2012).

    The unbiased estimator can go slightly negative under H0 (same
    distribution); that is the canonical behaviour, not a bug. Pair-level
    permutation-null is built separately by `_mmd_permutation_summary`
    because computing it per-pair (240 pairs * MMD_PERMUTATIONS) would
    dominate wall-clock for marginal scientific value — we instead build
    one shared null per (extraction_point, layer) from a uniform subsample
    of pairs (same bandwidth, so the null shape is shared).
    """
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    na, nb = len(Xa), len(Xb)
    if na < 2 or nb < 2:
        return float("nan")
    Z = np.vstack([Xa, Xb])
    K, _sigma2 = _rbf_kernel_with_bandwidth(Z)
    return _unbiased_mmd2_from_kernel(K, na)


def _mmd_permutation_summary(
    activations: np.ndarray,
    cond_ids: list[str],
    *,
    n_perm: int,
    variant: str,
    n_pair_samples: int = 16,
    rng=None,
) -> dict:
    """Build a shared permutation null for MMD² across a random subsample of
    (i, j) pairs at one (extraction_point, layer, variant).

    For each sampled pair, computes the observed unbiased MMD² and the
    permutation null distribution (relabel-and-recompute). Returns the
    aggregate per-pair p-values + the pooled null summary. Caller uses
    this to read significance per pair; the predictor SCALAR remains the
    observed MMD² in the main matrix file.

    n_pair_samples is capped so this stays bounded (~60s for 16 pairs *
    200 perms on n=50; full 240 * 200 would be a 4-minute wall-clock hit
    for marginal value when most pairs are visibly distinguishable from
    the noise floor on the unbiased estimator).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if variant == "centered":
        # Match the variant the main matrix is computed on.
        activations = _maybe_prompt_center(activations, do_center=True)
    n_cond = activations.shape[0]
    candidate_pairs = [(i, j) for i in range(n_cond) for j in range(n_cond) if i != j]
    rng.shuffle(candidate_pairs)
    sampled = candidate_pairs[: min(n_pair_samples, len(candidate_pairs))]
    per_pair = []
    nulls_pooled: list[float] = []
    for i, j in sampled:
        Xa = _drop_nan_rows(activations[i])
        Xb = _drop_nan_rows(activations[j])
        na, nb = len(Xa), len(Xb)
        if na < 2 or nb < 2:
            continue
        Z = np.vstack([Xa, Xb])
        K, sigma2 = _rbf_kernel_with_bandwidth(Z)
        observed = _unbiased_mmd2_from_kernel(K, na)
        # Permutation null via row+column re-permutation of K.
        n_total = K.shape[0]
        null_samples = []
        for _ in range(n_perm):
            perm = rng.permutation(n_total)
            Kp = K[perm][:, perm]
            null_samples.append(_unbiased_mmd2_from_kernel(Kp, na))
        null = np.asarray(null_samples)
        # One-sided p-value (P[null >= observed]).
        p_value = float((np.sum(null >= observed) + 1) / (len(null) + 1))
        per_pair.append(
            {
                "a": cond_ids[i],
                "b": cond_ids[j],
                "observed_mmd2": float(observed),
                "sigma2": float(sigma2),
                "null_median": float(np.median(null)),
                "null_p95": float(np.percentile(null, 95)),
                "p_value": p_value,
            }
        )
        nulls_pooled.extend(null.tolist())
    if not nulls_pooled:
        return {"per_pair": [], "n_pair_samples_done": 0}
    pooled = np.asarray(nulls_pooled)
    return {
        "per_pair": per_pair,
        "n_pair_samples_done": len(per_pair),
        "pooled_null_median": float(np.median(pooled)),
        "pooled_null_p95": float(np.percentile(pooled, 95)),
        "pooled_null_max": float(np.max(pooled)),
    }


def _c2st_auc(Xa: np.ndarray, Xb: np.ndarray, folds: int = C2ST_FOLDS) -> float:
    """Cross-validated linear-probe classifier-2-sample test, returned as a
    DISTANCE for sign-consistency with the rest of the metric panel.

    Raw AUC: 1.0 = perfectly separable, 0.5 = indistinguishable. To put it on
    the same "higher = farther apart" scale as cosine_distance / euclidean /
    MMD² / gauss_kl / W2 (and to match the sign convention of #474's cosine
    distance), we return ``c2st_dist = 2 * (AUC - 0.5)`` ∈ [0, 1]. This way
    the heatmap colorbar is single-signed for every metric: rho < 0 with ΔG
    means "more similar contexts → more transfer," uniformly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    na, nb = len(Xa), len(Xb)
    if na < folds or nb < folds:
        return float("nan")
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(na), np.ones(nb)])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        # sklearn>=1.8 deprecates the `penalty=` kwarg; default is L2, control
        # regularization via C (smaller = stronger). solver="lbfgs" is the
        # default L2-compatible solver.
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X[tr], y[tr])
        score = clf.decision_function(X[te])
        aucs.append(roc_auc_score(y[te], score))
    auc = float(np.mean(aucs))
    # Distance form: 2*|AUC − 0.5|, clipped to [0, 1]. Symmetric around the
    # chance boundary so labels-flipped C2ST scoring (~ 1 - AUC) and the
    # standard scoring map to the same predictor scalar.
    return float(min(1.0, 2.0 * abs(auc - 0.5)))


def _delta_spectrum(Xa: np.ndarray, Xb: np.ndarray, k: int) -> dict[str, float]:
    """PAIRED Δ-displacement spectrum.

    Both clouds MUST be in matched (cond, q) ordering — Δ_i = Xb[i] − Xa[i].
    Asserts shape match and per-row alignment; if either cloud has NaN at
    row i, that row is dropped on BOTH sides (paired drop) so alignment
    survives.

    Returns three candidate predictor scalars:
      - mean_norm        : ‖mean Δ‖
      - coherence        : energy_in_mean_dir / total_energy ∈ [0, 1]
      - effective_dim    : (Σ λ_i)² / Σ λ_i²  (participation ratio in PCA)
    """
    if Xa.shape != Xb.shape:
        raise AssertionError(f"Δ-spectrum requires paired clouds; got Xa={Xa.shape} Xb={Xb.shape}")
    # Paired NaN drop — drop a row from BOTH sides if either has NaN.
    mask = ~(np.any(np.isnan(Xa), axis=1) | np.any(np.isnan(Xb), axis=1))
    Xa = Xa[mask]
    Xb = Xb[mask]
    if len(Xa) < 2:
        return {"mean_norm": float("nan"), "coherence": float("nan"), "effective_dim": float("nan")}
    delta = Xb - Xa  # (n_q, H)
    mean_delta = delta.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_delta))
    total_energy = float(np.sum(delta**2))
    if total_energy < 1e-12 or mean_norm < 1e-12:
        coherence = 0.0
    else:
        proj_onto_mean = delta @ mean_delta / mean_norm  # (n_q,)
        coherence = float(np.sum(proj_onto_mean**2) / total_energy)
    # Effective dimensionality via Gram-eigenvalue participation ratio.
    delta_c = delta - delta.mean(axis=0, keepdims=True)
    G = delta_c @ delta_c.T
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.clip(eigvals, 0.0, None)
    s1 = eigvals.sum()
    s2 = (eigvals**2).sum()
    eff_dim = 0.0 if s2 < 1e-18 else float(s1**2 / s2)
    return {"mean_norm": mean_norm, "coherence": coherence, "effective_dim": eff_dim}


def _gaussian_sym_kl_in_subspace(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """Gaussian symmetric-KL between cloud-fitted Gaussians in the top-k PCA
    subspace.

    Closed form: KL(N0||N1) = 0.5 * (tr(Σ1^-1 Σ0) + (μ1-μ0)^T Σ1^-1 (μ1-μ0)
                                       - k + log(det Σ1 / det Σ0))
    Symmetric-KL = 0.5 * (KL(0||1) + KL(1||0)).
    """
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    Ya, Yb = _pair_pca_subspace(Xa, Xb, k)
    mu_a = Ya.mean(axis=0)
    mu_b = Yb.mean(axis=0)
    Sa = np.cov(Ya.T, ddof=1) + 1e-6 * np.eye(Ya.shape[1])
    Sb = np.cov(Yb.T, ddof=1) + 1e-6 * np.eye(Yb.shape[1])

    def _one_kl(S0, S1, m0, m1):
        # KL(N0||N1)
        S1_inv = np.linalg.inv(S1)
        sign0, logdet0 = np.linalg.slogdet(S0)
        sign1, logdet1 = np.linalg.slogdet(S1)
        if sign0 <= 0 or sign1 <= 0:
            return float("nan")
        d = S0.shape[0]
        return 0.5 * (
            np.trace(S1_inv @ S0) + (m1 - m0) @ S1_inv @ (m1 - m0) - d + (logdet1 - logdet0)
        )

    kl_ab = _one_kl(Sa, Sb, mu_a, mu_b)
    kl_ba = _one_kl(Sb, Sa, mu_b, mu_a)
    if np.isnan(kl_ab) or np.isnan(kl_ba):
        return float("nan")
    return float(0.5 * (kl_ab + kl_ba))


def _bures_wasserstein2(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """Squared Bures-Wasserstein distance between cloud-fitted Gaussians in
    the top-k PCA subspace.

    W₂² = ‖μ_a − μ_b‖² + tr(Σ_a + Σ_b − 2(Σ_a^{1/2} Σ_b Σ_a^{1/2})^{1/2})
    """
    from scipy.linalg import sqrtm

    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    Ya, Yb = _pair_pca_subspace(Xa, Xb, k)
    mu_a = Ya.mean(axis=0)
    mu_b = Yb.mean(axis=0)
    Sa = np.cov(Ya.T, ddof=1) + 1e-6 * np.eye(Ya.shape[1])
    Sb = np.cov(Yb.T, ddof=1) + 1e-6 * np.eye(Yb.shape[1])
    Sa_sqrt = sqrtm(Sa).real
    cross = sqrtm(Sa_sqrt @ Sb @ Sa_sqrt).real
    bures = np.trace(Sa + Sb - 2 * cross)
    mu_sq = float((mu_a - mu_b) @ (mu_a - mu_b))
    return float(mu_sq + max(0.0, float(bures)))


def _maybe_prompt_center(activations: np.ndarray, do_center: bool) -> np.ndarray:
    """Optionally subtract the per-question (a.k.a. "prompt") mean across all
    in-scope contexts so distance scores reflect the *off-mean-context*
    component only. activations shape: (n_cond, n_q, H). For end_of_system
    where n_q == 1 we subtract the cross-context mean instead (the only
    meaningful "centering" target when there's no per-question grid).
    """
    if not do_center:
        return activations
    if activations.shape[1] > 1:
        # Per-question mean across the n_cond axis: shape (1, n_q, H).
        mu = np.nanmean(activations, axis=0, keepdims=True)
    else:
        # Cross-context mean of the single per-cond vector: shape (1, 1, H).
        mu = np.nanmean(activations, axis=0, keepdims=True)
    return activations - mu


def _compute_metric_matrix(  # noqa: C901 — per-metric dispatcher; one branch per metric, intentionally flat.
    activations: np.ndarray,
    cond_ids: list[str],
    metric: str,
    extraction_point: str,
    pca_k: int,
    variant: str = "raw",
) -> dict:
    """Compute the (n_cond × n_cond) pairwise predictor matrix for one metric.

    Returns a dict with the matrix and any per-(metric) auxiliary outputs
    (e.g. Δ-spectrum produces 3 sub-predictors stored as separate matrices).

    Parameters
    ----------
    activations: (n_cond, n_q, H) — for end_of_system, n_q == 1.
    metric: one of CENTROID_METRICS + CLOUD_METRICS.
    variant: "raw" or "centered" (see _maybe_prompt_center). The variant
        label is recorded in the metric payload so the regression phase can
        enumerate raw + centered as distinct predictor rows.

    end_of_system handling:
      - Cloud metrics (CLOUD_METRICS) return {"matrix": None, "n_a": ...}.
      - "mahal" (per-pair pooled cov) returns NaN matrix because n_q == 1
        makes the within-cloud covariance undefined; use "mahal_pooled_ctx"
        instead (Mahalanobis vs. the pooled context covariance across the
        full Class-A subpanel — the meaningful one-vector-per-cond metric).
    """
    if variant not in PREDICTOR_VARIANTS:
        raise ValueError(f"variant must be one of {PREDICTOR_VARIANTS}; got {variant!r}")
    activations = _maybe_prompt_center(activations, do_center=(variant == "centered"))
    n_cond = activations.shape[0]
    is_centroid = metric in CENTROID_METRICS
    if not is_centroid and extraction_point == "end_of_system":
        # Cloud metrics are N/A at end_of_system — n_q==1, no cloud exists.
        return {
            "matrix": None,
            "n_a": "cloud metric N/A at end_of_system (one vector per cond)",
            "variant": variant,
        }

    if metric == "delta_spec":
        # Δ-spec emits 3 scalars per pair — store as 3 stacked matrices.
        ms = {
            "mean_norm": [[None] * n_cond for _ in range(n_cond)],
            "coherence": [[None] * n_cond for _ in range(n_cond)],
            "effective_dim": [[None] * n_cond for _ in range(n_cond)],
        }
        for i in range(n_cond):
            for j in range(n_cond):
                if i == j:
                    for key in ms:
                        ms[key][i][j] = 0.0
                    continue
                spec = _delta_spectrum(activations[i], activations[j], pca_k)
                for key in ms:
                    ms[key][i][j] = float(spec[key])
        return {
            "variant": variant,
            "matrices": {
                k: {
                    cond_ids[i]: {cond_ids[j]: ms[k][i][j] for j in range(n_cond)}
                    for i in range(n_cond)
                }
                for k in ms
            },
            "sub_predictors": list(ms.keys()),
        }

    # mahal_pooled_ctx needs the shared pooled-cov state pre-built ONCE
    # (it's the same matrix for every (i, j) pair within this metric file).
    pooled_state = None
    if metric == "mahal_pooled_ctx":
        pooled_state = _build_context_pooled_mahal_state(activations, pca_k)
        if pooled_state is None:
            # Singular pooled covariance — fail loud rather than emit NaN
            # matrix that downstream silently drops.
            raise RuntimeError(
                f"mahal_pooled_ctx at extraction_point={extraction_point}: "
                f"pooled centroid covariance is singular after PCA "
                f"reduction to top-{pca_k} (n_cond={n_cond}). "
                "Increase --pca-k, drop fewer contexts, or skip this metric."
            )

    mat: list[list[float]] = [[0.0] * n_cond for _ in range(n_cond)]
    for i in range(n_cond):
        for j in range(n_cond):
            if i == j:
                mat[i][j] = 0.0
                continue
            Xa = activations[i]
            Xb = activations[j]
            if metric == "cosine":
                d = _centroid_cosine_distance(Xa, Xb)
            elif metric == "euclidean":
                d = _centroid_euclidean(Xa, Xb)
            elif metric == "mahal":
                d = _centroid_mahal(Xa, Xb, pca_k)
            elif metric == "mahal_pooled_ctx":
                d = _context_mahal_with_pooled_cov(Xa, Xb, pooled_state, i, j)
            elif metric == "mmd":
                d = _rbf_mmd_squared(Xa, Xb)
            elif metric == "c2st":
                d = _c2st_auc(Xa, Xb)
            elif metric == "gauss_kl":
                d = _gaussian_sym_kl_in_subspace(Xa, Xb, pca_k)
            elif metric == "wass2":
                d = _bures_wasserstein2(Xa, Xb, pca_k)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            mat[i][j] = float(d)
    return {
        "variant": variant,
        "matrix": {
            cond_ids[i]: {cond_ids[j]: mat[i][j] for j in range(n_cond)} for i in range(n_cond)
        },
    }


def run_metrics(
    *,
    activations_by_point: dict[str, dict[int, dict]],
    metrics: tuple[str, ...],
    pca_k: int,
    overwrite: bool,
    mmd_permutations: int = MMD_PERMUTATIONS,
) -> None:
    """Compute every (extraction_point × layer × metric × variant) distance
    matrix and checkpoint EACH ONE to disk immediately.

    Per-metric notes:
      - Cloud metrics at end_of_system → N/A row written with explicit
        `matrix: null` (NOT silently dropped).
      - Centroid `mahal` at end_of_system → also N/A (per-pair within-cloud
        cov is undefined at n_q=1); the meaningful one-vector-per-cond
        Mahalanobis is `mahal_pooled_ctx`.
      - For `mmd`, also writes a `<point>__layer<L>__mmd__perm.json`
        companion with the permutation-null summary (median, max under H0)
        so downstream callers can compute pair-level p-values if desired.
        Permutations run on a uniform random subset of pairs (default 16)
        to keep wall-clock bounded — the bandwidth is identical across
        pairs, so the null shape is shared.
    """
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    for pt, layer_map in activations_by_point.items():
        for L, payload in layer_map.items():
            cond_ids = payload["cond_ids"]
            arr = payload["activations"]
            for m in metrics:
                for variant in PREDICTOR_VARIANTS:
                    out_path = METRIC_DIR / f"{pt}__layer{L}__{m}__{variant}.json"
                    if out_path.exists() and not overwrite:
                        logger.info("Skipping existing %s", out_path)
                        continue

                    t0 = time.time()
                    # Centroid mahal at end_of_system is N/A (per-pair cov is
                    # undefined when n_q==1); use mahal_pooled_ctx instead.
                    if pt == "end_of_system" and m == "mahal":
                        _write_json_atomic(
                            out_path,
                            {
                                "schema_version": 1,
                                "extraction_point": pt,
                                "layer": L,
                                "metric": m,
                                "variant": variant,
                                "matrix": None,
                                "n_a": (
                                    "per-pair pooled cov undefined at "
                                    "end_of_system (n_q=1); use mahal_pooled_ctx"
                                ),
                                "git_sha": _git_sha(),
                                "timestamp_utc": _now_iso(),
                            },
                        )
                        continue

                    res = _compute_metric_matrix(arr, cond_ids, m, pt, pca_k, variant=variant)
                    payload_out = {
                        "schema_version": 1,
                        "extraction_point": pt,
                        "layer": L,
                        "metric": m,
                        "variant": variant,
                        "pca_k": pca_k,
                        "cond_ids": cond_ids,
                        "git_sha": _git_sha(),
                        "timestamp_utc": _now_iso(),
                        **res,
                    }
                    _write_json_atomic(out_path, payload_out)
                    logger.info("Wrote %s in %.2fs", out_path, time.time() - t0)

                    # MMD permutation null (cloud regime only).
                    if m == "mmd" and pt != "end_of_system" and res.get("matrix") is not None:
                        perm_path = METRIC_DIR / f"{pt}__layer{L}__{m}__{variant}__perm.json"
                        if perm_path.exists() and not overwrite:
                            continue
                        perm_summary = _mmd_permutation_summary(
                            arr,
                            cond_ids,
                            n_perm=mmd_permutations,
                            variant=variant,
                        )
                        _write_json_atomic(
                            perm_path,
                            {
                                "schema_version": 1,
                                "extraction_point": pt,
                                "layer": L,
                                "metric": m,
                                "variant": variant,
                                "n_perm": mmd_permutations,
                                "summary": perm_summary,
                                "git_sha": _git_sha(),
                                "timestamp_utc": _now_iso(),
                            },
                        )


# ───────────────────────── regression phase ─────────────────────────


def _length_partial(x: np.ndarray, y: np.ndarray, covar: np.ndarray) -> tuple[float, float]:
    """Rank-then-residualize length-partial Spearman (matches #406/#462/#474).

    Identical to i474_cosine_followup._length_partial; copied verbatim so the
    convention stays in lockstep without a cross-file import dependency.
    """
    from scipy.stats import pearsonr, rankdata

    rx, ry, rc = rankdata(x), rankdata(y), rankdata(covar)
    ex = rx - np.polyval(np.polyfit(rc, rx, 1), rc)
    ey = ry - np.polyval(np.polyfit(rc, ry, 1), rc)
    return pearsonr(ex, ey)


def _length_partial_residualize_rank(
    x: np.ndarray, y: np.ndarray, covar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_resid, y_resid) on the rank scale with the log-length
    covariate's linear-in-rank component projected out. Matches the
    rank-then-residualize convention used by `_length_partial`."""
    from scipy.stats import rankdata

    rx, ry, rc = rankdata(x), rankdata(y), rankdata(covar)
    ex = rx - np.polyval(np.polyfit(rc, rx, 1), rc)
    ey = ry - np.polyval(np.polyfit(rc, ry, 1), rc)
    return ex, ey


def _loocv_r2(
    x: np.ndarray,
    y: np.ndarray,
    cond_ids_a: list[str],
    cond_ids_b: list[str],
    *,
    covar: np.ndarray | None = None,
) -> float:
    """Leave-one-context-out CV R² (the i474 fig9 pattern), length-partialed.

    For each cond C, hold out all pairs touching C, fit OLS on the
    remainder, predict held-out, compute (1 − SSE / SST). When `covar` is
    provided, residualize x and y on rank(covar) FIRST so the CV captures
    the same length-controlled signal as the headline Spearman; matching
    the partial regression keeps the winner-selection criterion consistent
    with the published metric and prevents a length-confound predictor
    from winning by capturing log-prompt-tokens variance.
    """
    n = len(x)
    if covar is not None:
        x, y = _length_partial_residualize_rank(x, y, covar)
    pred = np.full(n, np.nan)
    src = np.array(cond_ids_a)
    tgt = np.array(cond_ids_b)
    for C in set(cond_ids_a) | set(cond_ids_b):
        train = ~((src == C) | (tgt == C))
        test = (src == C) | (tgt == C)
        if train.sum() < 5:
            continue
        # OLS, 1-D
        b, a = np.polyfit(x[train], y[train], 1)
        pred[test] = a + b * x[test]
    m = ~np.isnan(pred)
    if m.sum() < 5:
        return float("nan")
    sse = np.sum((y[m] - pred[m]) ** 2)
    sst = np.sum((y[m] - y[m].mean()) ** 2)
    if sst < 1e-18:
        return float("nan")
    return float(1.0 - sse / sst)


def _pairs(cond_ids: list[str], nonstylized_only: bool) -> list[tuple[str, str]]:
    """Build the list of ordered off-diagonal pairs, optionally dropping any
    pair touching a stylized persona (A3/A4/A5)."""
    out = []
    for a in cond_ids:
        for b in cond_ids:
            if a == b:
                continue
            if nonstylized_only and ((a in STY_CIDS) or (b in STY_CIDS)):
                continue
            out.append((a, b))
    return out


def _materialize_predictor_vector(
    metric_payload: dict,
    pairs: list[tuple[str, str]],
    sub_predictor: str | None,
) -> np.ndarray | None:
    """Read a distance value per pair from a metric_payload (one matrix file).

    Returns None if the matrix is entirely N/A (cloud metric at end_of_system).
    For Δ-spectrum, ``sub_predictor`` ∈ {"mean_norm", "coherence",
    "effective_dim"}. Returns None if ANY requested pair is missing from the
    matrix — caller is responsible for choosing a pair list that matches the
    metric's cond_ids (end_of_system subpanel use case).
    """
    if "matrix" in metric_payload and metric_payload["matrix"] is None:
        return None  # N/A — cloud metric or pair-cov-undefined at end_of_system
    if "matrices" in metric_payload:
        if sub_predictor is None:
            return None
        m = metric_payload["matrices"][sub_predictor]
    else:
        m = metric_payload["matrix"]
    vals = []
    for a, b in pairs:
        if a not in m or b not in m[a]:
            return None
        v = m[a][b]
        if v is None:
            return None
        vals.append(float(v))
    return np.array(vals, dtype=np.float64)


def _enumerate_predictors(metric_files: list[Path]) -> list[dict]:
    """Walk every metric file and enumerate every distinct predictor scalar
    (one row per (extraction_point, layer, metric, variant, sub_predictor)).

    Each row carries the metric file's `cond_ids` so the regression phase
    can restrict the pair list to the SUBPANEL the metric is actually
    defined on (Class-A-only for end_of_system, full 16 elsewhere).
    """
    rows = []
    for p in metric_files:
        # Skip the per-pair MMD permutation companion files.
        if "__perm" in p.name:
            continue
        payload = json.loads(p.read_text())
        pt = payload["extraction_point"]
        L = payload["layer"]
        m = payload["metric"]
        variant = payload.get("variant", "raw")
        cond_ids_file = payload.get("cond_ids")
        if "matrices" in payload:
            for sub in payload["sub_predictors"]:
                rows.append(
                    {
                        "extraction_point": pt,
                        "layer": L,
                        "metric": m,
                        "variant": variant,
                        "sub_predictor": sub,
                        "cond_ids": cond_ids_file,
                        "file": str(p),
                    }
                )
        else:
            rows.append(
                {
                    "extraction_point": pt,
                    "layer": L,
                    "metric": m,
                    "variant": variant,
                    "sub_predictor": None,
                    "cond_ids": cond_ids_file,
                    "file": str(p),
                }
            )
    return rows


def _saturation_fraction(g: np.ndarray) -> float:
    """Fraction of cells at/above the saturation threshold (g_logprob > -0.1)."""
    return float(np.mean(g > SATURATION_GLOGP_THRESHOLD))


def run_regression(
    *,
    cond_ids: list[str],
    arms: tuple[str, ...],
    epochs: tuple[int, ...],
    overwrite: bool,
) -> dict:
    """Per-(arm, epoch) regression of every enumerated predictor against
    ΔG (primary) and g_logprob (base-prior-safe). Checkpoints each cell.

    Each predictor row reports rho + CV on TWO panels:
      - panel_primary    — the "full" panel for that predictor: the 240
        full grid for predictors defined on all 16 conds; for
        end_of_system predictors (defined on Class A only) it's the
        20-pair Class-A subpanel (n=5*4 ordered pairs).
      - panel_nonstylized — drops any pair touching A3/A4/A5 (a
        sub-restriction of panel_primary). For end_of_system this leaves
        the 2*1=2-pair A1-A2 subpanel which is too small to interpret
        and is recorded explicitly as such.

    The leave-one-context-out CV is **length-partialed** (residualizes
    rank(x) and rank(y) on rank(log prompt_tokens) first) so it captures
    the same length-controlled signal as the headline Spearman. A
    length-confound predictor that "wins" by capturing log_prompt_tokens
    variance therefore CANNOT win the bake-off.

    Returns the headline (loc_ep1) summary for convenience.
    """
    REGR_DIR.mkdir(parents=True, exist_ok=True)
    metric_files = sorted(METRIC_DIR.glob("*.json"))
    predictors = _enumerate_predictors(metric_files)
    prompt_tokens = _load_prompt_tokens()

    all_cells: dict[str, dict] = {}
    for arm in arms:
        for ep in epochs:
            cell_key = f"{arm}_ep{ep}"
            out_path = REGR_DIR / f"{cell_key}.json"
            if out_path.exists() and not overwrite:
                all_cells[cell_key] = json.loads(out_path.read_text())
                continue
            G = _load_G(arm, ep)
            # Full 16-cond panels (used by every predictor whose metric
            # file carries all 16 cond_ids).
            pairs_full16 = _pairs(cond_ids, nonstylized_only=False)
            pairs_ns_full16 = _pairs(cond_ids, nonstylized_only=True)
            dg_f16 = np.array([G[a][b]["delta_g"] for a, b in pairs_full16])
            g_f16 = np.array([G[a][b]["g_logprob"] for a, b in pairs_full16])
            ln_f16 = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs_full16])
            dg_ns16 = np.array([G[a][b]["delta_g"] for a, b in pairs_ns_full16])
            g_ns16 = np.array([G[a][b]["g_logprob"] for a, b in pairs_ns_full16])
            ln_ns16 = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs_ns_full16])
            src_f16 = [a for a, _ in pairs_full16]
            tgt_f16 = [b for _, b in pairs_full16]
            src_ns16 = [a for a, _ in pairs_ns_full16]
            tgt_ns16 = [b for _, b in pairs_ns_full16]

            sat_full16 = _saturation_fraction(g_f16)
            sat_ns16 = _saturation_fraction(g_ns16)

            entries = []
            for desc in predictors:
                payload = json.loads(Path(desc["file"]).read_text())
                pred_cond_ids = desc.get("cond_ids") or cond_ids

                # Choose the pair list this predictor is defined on.
                if set(pred_cond_ids) == set(cond_ids):
                    pairs_primary = pairs_full16
                    pairs_nonsty = pairs_ns_full16
                    dg_p, g_p, ln_p = dg_f16, g_f16, ln_f16
                    dg_n, g_n, ln_n = dg_ns16, g_ns16, ln_ns16
                    src_p, tgt_p = src_f16, tgt_f16
                    src_n, tgt_n = src_ns16, tgt_ns16
                    panel_primary_name = "full16 (240 ordered pairs)"
                    panel_nonsty_name = "nonstylized (156 ordered pairs)"
                else:
                    # Subpanel — e.g. end_of_system on Class A.
                    pairs_primary = _pairs(pred_cond_ids, nonstylized_only=False)
                    pairs_nonsty = _pairs(pred_cond_ids, nonstylized_only=True)
                    dg_p = np.array([G[a][b]["delta_g"] for a, b in pairs_primary])
                    g_p = np.array([G[a][b]["g_logprob"] for a, b in pairs_primary])
                    ln_p = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs_primary])
                    dg_n = np.array([G[a][b]["delta_g"] for a, b in pairs_nonsty])
                    g_n = np.array([G[a][b]["g_logprob"] for a, b in pairs_nonsty])
                    ln_n = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs_nonsty])
                    src_p = [a for a, _ in pairs_primary]
                    tgt_p = [b for _, b in pairs_primary]
                    src_n = [a for a, _ in pairs_nonsty]
                    tgt_n = [b for _, b in pairs_nonsty]
                    panel_primary_name = (
                        f"subpanel cond_ids={sorted(pred_cond_ids)} "
                        f"({len(pairs_primary)} ordered pairs)"
                    )
                    panel_nonsty_name = f"subpanel nonstylized ({len(pairs_nonsty)} ordered pairs)"

                xv_p = _materialize_predictor_vector(payload, pairs_primary, desc["sub_predictor"])
                xv_n = (
                    _materialize_predictor_vector(payload, pairs_nonsty, desc["sub_predictor"])
                    if len(pairs_nonsty) >= 5
                    else None
                )

                if xv_p is None:
                    entries.append({**desc, "status": "N/A (matrix is None or missing pair)"})
                    continue

                # Length-partial Spearman, per panel x DV. nan_p is acceptable
                # if a subpanel is too small / degenerate; we record it.
                rho_p_dg, p_p_dg = _length_partial(xv_p, dg_p, ln_p)
                rho_p_g, p_p_g = _length_partial(xv_p, g_p, ln_p)
                if xv_n is not None and len(xv_n) >= 5:
                    rho_n_dg, p_n_dg = _length_partial(xv_n, dg_n, ln_n)
                    rho_n_g, p_n_g = _length_partial(xv_n, g_n, ln_n)
                    cv_n_dg = _loocv_r2(xv_n, dg_n, src_n, tgt_n, covar=ln_n)
                    cv_n_g = _loocv_r2(xv_n, g_n, src_n, tgt_n, covar=ln_n)
                else:
                    rho_n_dg = p_n_dg = rho_n_g = p_n_g = float("nan")
                    cv_n_dg = cv_n_g = float("nan")

                # Length-partialed leave-one-context-out CV (the i474 fig9
                # pattern, generalized + length-controlled).
                cv_p_dg = _loocv_r2(xv_p, dg_p, src_p, tgt_p, covar=ln_p)
                cv_p_g = _loocv_r2(xv_p, g_p, src_p, tgt_p, covar=ln_p)

                entries.append(
                    {
                        **desc,
                        "panel_primary": panel_primary_name,
                        "panel_nonstylized": panel_nonsty_name,
                        "n_primary": len(xv_p),
                        "n_nonstylized": len(xv_n) if xv_n is not None else 0,
                        "rho_full_deltag": float(rho_p_dg),
                        "p_full_deltag": float(p_p_dg),
                        "rho_full_glogp": float(rho_p_g),
                        "p_full_glogp": float(p_p_g),
                        "rho_nonstylized_deltag": float(rho_n_dg),
                        "p_nonstylized_deltag": float(p_n_dg),
                        "rho_nonstylized_glogp": float(rho_n_g),
                        "p_nonstylized_glogp": float(p_n_g),
                        "cv_full_deltag": float(cv_p_dg),
                        "cv_full_glogp": float(cv_p_g),
                        "cv_nonstylized_deltag": float(cv_n_dg),
                        "cv_nonstylized_glogp": float(cv_n_g),
                    }
                )

            cell_payload = {
                "schema_version": 1,
                "arm": arm,
                "epoch": ep,
                "n_pairs_full16": len(pairs_full16),
                "n_pairs_nonstylized_full16": len(pairs_ns_full16),
                "saturation_frac_full16": sat_full16,
                "saturation_frac_nonstylized_full16": sat_ns16,
                "entries": entries,
                "git_sha": _git_sha(),
                "timestamp_utc": _now_iso(),
            }
            _write_json_atomic(out_path, cell_payload)
            logger.info("Wrote %s — %d predictor entries", out_path, len(entries))
            all_cells[cell_key] = cell_payload

    return all_cells


def select_winner(headline_cell: dict) -> dict | None:
    """Pick the highest-CV predictor that ALSO survives on the non-stylized
    panel AND on the base-prior-safe g_logprob check.

    Survival conditions:
      - rho_nonstylized_deltag has the same sign as rho_full_deltag
      - |rho_nonstylized_deltag| > FLOOR (small; mainly a sanity floor)
      - rho_full_glogp same sign as rho_full_deltag (not pure base prior)

    Returns the winning entry dict or None if no predictor survives.
    """
    FLOOR_RHO = 0.10
    survivors = []
    for e in headline_cell["entries"]:
        if "rho_full_deltag" not in e:
            continue
        rho_f = e["rho_full_deltag"]
        rho_ns = e["rho_nonstylized_deltag"]
        rho_g = e["rho_full_glogp"]
        if not (np.sign(rho_f) == np.sign(rho_ns) and abs(rho_ns) > FLOOR_RHO):
            continue
        # The relevant g_logprob check: base-prior survival means trained
        # log-prob shows the SAME-direction relationship as ΔG. If rho_g
        # collapses to ~0 or flips sign, the ΔG signal is base-prior driven.
        if np.sign(rho_g) != np.sign(rho_f):
            continue
        if np.isnan(e["cv_full_deltag"]):
            continue
        survivors.append(e)
    if not survivors:
        return None
    survivors.sort(key=lambda e: e["cv_full_deltag"], reverse=True)
    return survivors[0]


# ───────────────────────── correctness cross-check ─────────────────────────


def reproduce_last_token_cosine_check(
    activations_last_prompt: dict[int, dict],
    existing_cosines: dict[int, dict],
    cond_ids: list[str],
    *,
    strict: bool,
) -> dict[int, dict]:
    """Re-compute last-token cosine distances from our fresh activations and
    diff against the existing eval_results/issue_406/cosine/C_L*.json.

    The existing #406 recipe = cosine-distance of cond-mean activations
    across 50 probes at last-prompt-token. Our last_prompt extraction is
    the same recipe; the two must agree within COSINE_REPRO_TOLERANCE.

    Parameters
    ----------
    strict: when True, raise AssertionError on ANY layer mismatch — the
        whole bake-off is unsafe to interpret if the prompt-building or
        last-position indexing diverges from #406's recipe. The orchestrator
        sets strict=True on full real-data runs (all 16 conds × 50 probes)
        and strict=False on subset smoke / debug runs where the cond-set or
        probe-set deliberately differs.

    Returns
    -------
    Per-layer diff summary dict.

    Raises
    ------
    AssertionError when strict=True AND any layer's max |diff| exceeds
    COSINE_REPRO_TOLERANCE.
    """
    out: dict[int, dict] = {}
    failures: list[str] = []
    for L, payload in activations_last_prompt.items():
        if L not in existing_cosines:
            continue
        existing = existing_cosines[L]["matrix"]
        arr = payload["activations"]  # (n_cond, n_q, H)
        our_cond_ids = payload["cond_ids"]
        # Restrict to the intersection (the existing matrices have all 16).
        common = [c for c in our_cond_ids if c in existing]
        if len(common) < 2:
            continue
        max_diff = 0.0
        n_pairs = 0
        for i, a in enumerate(our_cond_ids):
            if a not in existing:
                continue
            for j, b in enumerate(our_cond_ids):
                if a == b or b not in existing.get(a, {}):
                    continue
                ours = _centroid_cosine_distance(arr[i], arr[j])
                theirs = float(existing[a][b])
                max_diff = max(max_diff, abs(ours - theirs))
                n_pairs += 1
        ok = bool(max_diff < COSINE_REPRO_TOLERANCE)
        out[L] = {
            "max_abs_diff": float(max_diff),
            "n_pairs_checked": int(n_pairs),
            "tolerance": float(COSINE_REPRO_TOLERANCE),
            "ok": ok,
        }
        if not ok:
            failures.append(
                f"L{L}: max |diff| = {max_diff:.2e} > {COSINE_REPRO_TOLERANCE:.2e} "
                f"over {n_pairs} pairs"
            )
        level = logging.INFO if ok else logging.WARNING
        logger.log(
            level,
            "Cosine cross-check L%d: max |diff| = %.2e over %d pairs (ok=%s)",
            L,
            max_diff,
            n_pairs,
            ok,
        )
    if strict and failures:
        # Fail-fast: the bake-off is unsafe to interpret if the prompt-
        # building or last-position indexing diverges from #406's recipe.
        raise AssertionError(
            "Last-token cosine cross-check FAILED against "
            "eval_results/issue_406/cosine/C_L*.json (tolerance "
            f"{COSINE_REPRO_TOLERANCE:.2e}):\n  "
            + "\n  ".join(failures)
            + "\nThe extraction recipe diverges from #406; downstream "
            "regression is meaningless. Diagnose before continuing."
        )
    return out


# ───────────────────────── figure phase ─────────────────────────


def emit_figures(
    all_cells: dict[str, dict],
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
) -> None:
    """Two paper-style figures: (a) ρ heatmap across (metric × point) × layer
    for the headline cell; (b) winner's scatter vs ΔG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    headline = all_cells.get("loc_ep1")
    if not headline:
        logger.warning("No loc_ep1 cell — skipping figures")
        return

    # Build the (predictor-row × layer) ρ grid from headline entries.
    rows: dict[tuple[str, str, str | None], dict[int, float]] = {}
    for e in headline["entries"]:
        if "rho_nonstylized_deltag" not in e:
            continue
        key = (e["extraction_point"], e["metric"], e.get("sub_predictor"))
        rows.setdefault(key, {})[e["layer"]] = e["rho_nonstylized_deltag"]

    if not rows:
        logger.warning("No usable headline entries — skipping figures")
        return

    row_keys = sorted(rows.keys())
    row_labels = [f"{pt} · {m}" + (f" · {sub}" if sub else "") for (pt, m, sub) in row_keys]
    layer_list = sorted(layers)
    grid = np.full((len(row_keys), len(layer_list)), np.nan)
    for ri, key in enumerate(row_keys):
        for li, L in enumerate(layer_list):
            if L in rows[key]:
                grid[ri, li] = rows[key][L]

    fig, ax = plt.subplots(figsize=(8.5, 0.35 * len(row_keys) + 2.0))
    vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)))
    if not np.isfinite(vmax) or vmax < 1e-6:
        vmax = 1.0
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(layer_list)))
    ax.set_xticklabels([f"L{L}" for L in layer_list])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Residual layer")
    ax.set_title("loc_ep1 non-stylized · length-partial ρ(predictor, ΔG)", fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Spearman ρ", fontsize=8)
    for ri in range(len(row_keys)):
        for li in range(len(layer_list)):
            v = grid[ri, li]
            if np.isfinite(v):
                ax.text(
                    li,
                    ri,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black" if abs(v) < 0.5 * vmax else "white",
                )
    fig.tight_layout()
    savefig_paper(fig, "metric_layer_grid_heatmap", dir=str(FIGURE_DIR))
    plt.close(fig)

    # Winner scatter
    winner = select_winner(headline)
    if winner is None:
        logger.warning("No surviving predictor — skipping winner scatter")
        return
    pt = winner["extraction_point"]
    L = winner["layer"]
    m = winner["metric"]
    sub = winner.get("sub_predictor")
    payload = json.loads(Path(winner["file"]).read_text())
    cond_ids = payload["cond_ids"]
    pairs = _pairs(cond_ids, nonstylized_only=False)
    xv = _materialize_predictor_vector(payload, pairs, sub)
    G = _load_G("loc", 1)
    dg = np.array([G[a][b]["delta_g"] for a, b in pairs])
    sty = np.array([(a in STY_CIDS) or (b in STY_CIDS) for a, b in pairs])
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    base = paper_palette_role("baseline")
    acc = paper_palette_role("accent")
    ax.scatter(
        xv[~sty],
        dg[~sty],
        s=22,
        c=base,
        alpha=0.6,
        edgecolor="white",
        lw=0.5,
        label=f"non-stylized (n={int((~sty).sum())})",
    )
    ax.scatter(
        xv[sty],
        dg[sty],
        s=28,
        c=acc,
        alpha=0.85,
        edgecolor="white",
        lw=0.5,
        label=f"touches stylized (n={int(sty.sum())})",
    )
    sub_str = f" · {sub}" if sub else ""
    ax.set_xlabel(f"{pt} · {m}{sub_str} (layer {L})")
    ax.set_ylabel("ΔG = trained − base log P(marker)")
    ax.set_title(
        f"Winner: {pt}/{m}{sub_str}/L{L} — CV R²={winner['cv_full_deltag']:.2f}",
        fontsize=10,
        loc="left",
    )
    ax.grid(alpha=0.2, lw=0.5)
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "winner_scatter_vs_deltaG", dir=str(FIGURE_DIR))
    plt.close(fig)


# ───────────────────────── dry-run smoke ─────────────────────────


def _synthetic_clouds(
    *,
    n_cond: int = 6,
    n_q: int = 20,
    hidden: int = 64,
    rng=None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build synthetic clouds with KNOWN structure for the metric smoke test.

    Returns (activations, cond_ids, fake_deltag_matrix). The fake ΔG is
    constructed to monotonically increase with the synthetic distance so a
    well-behaved predictor must produce a strong rank correlation.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    cond_ids = [f"S{i}" for i in range(n_cond)]
    centers = rng.normal(size=(n_cond, hidden))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # Spread = exponential schedule so ΔG-against-distance has a clear gradient.
    radii = np.linspace(1.0, 3.0, n_cond)
    arr = np.zeros((n_cond, n_q, hidden), dtype=np.float32)
    for i in range(n_cond):
        cloud = rng.normal(scale=0.5, size=(n_q, hidden))
        arr[i] = (centers[i] * radii[i])[None, :] + cloud
    # Construct a "fake ΔG" matrix that's a monotone function of cosine-dist.
    fake_dg = np.zeros((n_cond, n_cond), dtype=np.float32)
    for i in range(n_cond):
        for j in range(n_cond):
            if i == j:
                continue
            mu_i = arr[i].mean(axis=0)
            mu_j = arr[j].mean(axis=0)
            cs = mu_i @ mu_j / (np.linalg.norm(mu_i) * np.linalg.norm(mu_j))
            # Higher ΔG when the contexts are MORE similar (matches the
            # i474 direction: similar contexts → more transfer).
            fake_dg[i, j] = 10.0 * float(cs) + rng.normal(scale=0.5)
    return arr, cond_ids, fake_dg


def dry_run_smoke() -> dict:
    """CPU-only sanity smoke. Exercises EVERY metric on synthetic clouds with
    known structure and confirms:

      - cosine ≈ 1 / euclidean ≈ 0 for identical clouds
      - C2ST AUC ≈ 0.5 for identical clouds
      - distances grow for well-separated clouds
      - PCA-whitened Fisher / Mahalanobis are finite and positive
      - Δ-spec coherence ≈ 1 for constant Δ, ≈ 0 for random Δ
      - PCA-via-Gram works at n ≪ d

    Returns a digest dict.
    """
    rng = np.random.default_rng(0)
    digest: dict[str, object] = {}

    # 1) Same-distribution clouds:
    #     - Cloud metrics (C2ST, MMD) → ≈ chance / ≈ 0 (they correctly say
    #       "indistinguishable"). With INDEPENDENT samples — not literally the
    #       same array, which trips twin-row memorization in C2ST.
    #     - Centroid metrics (cosine, euclidean) are NOT defined to be ≈ 0
    #       here: in high-D the centroid of two independent finite samples of
    #       N(0, I) is nearly orthogonal (curse of dimensionality), so centroid
    #       cosine_dist ≈ 1 even when the distributions match. We test centroid
    #       metrics under a stronger condition (same MEAN, tight cloud) below.
    X = rng.normal(size=(40, 64))
    Xa_id, Xb_id = X[:20], X[20:]  # two i.i.d. draws from N(0, I)
    # _c2st_auc now returns a DISTANCE (2*|AUC - 0.5|, [0, 1]).
    c2st_id = _c2st_auc(Xa_id, Xb_id)
    assert c2st_id < 0.50, f"C2ST distance on same-dist clouds was {c2st_id} (expected ~ 0)"
    mmd_id = _rbf_mmd_squared(Xa_id, Xb_id)
    # The unbiased MMD² is allowed to go slightly negative under H0; we test
    # that it stays close to zero in magnitude rather than asserting > 0.
    assert abs(mmd_id) < 0.10, f"Unbiased MMD² on same-dist clouds was {mmd_id} (expected |·| ~ 0)"
    # Centroid metrics on TIGHT clouds around a common large mean
    # (signal ≫ noise → centroids align):
    mu = rng.normal(scale=5.0, size=64)
    Xa_tight = mu + rng.normal(scale=0.1, size=(30, 64))
    Xb_tight = mu + rng.normal(scale=0.1, size=(30, 64))
    cd_tight = _centroid_cosine_distance(Xa_tight, Xb_tight)
    ed_tight = _centroid_euclidean(Xa_tight, Xb_tight)
    assert cd_tight < 1e-3, f"tight-same-mean cosine_dist was {cd_tight} (expected ≈ 0)"
    assert ed_tight < 0.5, f"tight-same-mean euclidean was {ed_tight} (expected small)"
    digest["identical_clouds"] = {
        "c2st_dist": c2st_id,
        "mmd2": mmd_id,
        "cosine_dist_tight_same_mean": cd_tight,
        "euclidean_tight_same_mean": ed_tight,
    }

    # 2) Well-separated clouds → cosine ≈ 2 (anti-parallel centroids),
    # C2ST distance ≈ 1, MMD² ≫ 0 (unbiased estimator).
    X1 = rng.normal(loc=10.0, size=(30, 64))
    X2 = rng.normal(loc=-10.0, size=(30, 64))
    cd = _centroid_cosine_distance(X1, X2)
    c2st_sep = _c2st_auc(X1, X2)
    mmd_sep = _rbf_mmd_squared(X1, X2)
    assert cd > 0.2, f"separated clouds cosine_dist was {cd} (expected substantial)"
    assert c2st_sep > 0.90, f"separated clouds C2ST distance was {c2st_sep} (expected ~ 1)"
    assert mmd_sep > 0.05, f"separated clouds MMD was {mmd_sep}"
    digest["separated_clouds"] = {
        "cosine_dist": cd,
        "c2st_dist": c2st_sep,
        "mmd2": mmd_sep,
    }

    # 3) Δ-spectrum — constant Δ → coherence ≈ 1
    base = rng.normal(size=(30, 64))
    Xb_const = base + np.ones(64) * 2.0  # constant displacement
    Xb_rand = base + rng.normal(size=(30, 64)) * 2.0
    spec_const = _delta_spectrum(base, Xb_const, k=8)
    spec_rand = _delta_spectrum(base, Xb_rand, k=8)
    assert spec_const["coherence"] > 0.95, f"constant-Δ coherence was {spec_const['coherence']}"
    assert spec_rand["coherence"] < 0.5, f"random-Δ coherence was {spec_rand['coherence']}"
    digest["delta_spec"] = {"const": spec_const, "random": spec_rand}

    # 4) Mahalanobis (per-pair pooled) well-conditioned at n ≪ d (n=30, d=64, k=8).
    # `_centroid_mahal` and `_fisher_distance` collapse to the same value for
    # the 2-cloud case; we test the surviving entry-point only.
    mahal = _centroid_mahal(X1, X2, k=8)
    assert np.isfinite(mahal) and mahal > 0, f"mahal was {mahal}"
    digest["mahal"] = float(mahal)

    # 4b) Context-pooled Mahalanobis works on ONE-vector-per-cond clouds
    # (end_of_system regime). Build a (n_cond, 1, H) array of distinct
    # per-cond centroids and confirm the pooled-cov state is finite +
    # the cross-cond distances are finite + non-zero.
    n_cond_eos = 6
    centroids = rng.normal(size=(n_cond_eos, 64))
    centroids *= 2.0  # spread them out so pooled cov is informative
    arr_eos = centroids[:, None, :].astype(np.float32)
    state = _build_context_pooled_mahal_state(arr_eos, k=4)
    assert state is not None, "pooled state was None on a clearly non-singular input"
    eos_distances = []
    for i in range(n_cond_eos):
        for j in range(n_cond_eos):
            if i == j:
                continue
            d_eos = _context_mahal_with_pooled_cov(arr_eos[i], arr_eos[j], state, i, j)
            assert np.isfinite(d_eos), f"end_of_system mahal_pooled_ctx was {d_eos}"
            eos_distances.append(d_eos)
    digest["end_of_system_mahal_pooled_ctx"] = {
        "n_pairs": len(eos_distances),
        "min": float(np.min(eos_distances)),
        "max": float(np.max(eos_distances)),
        "mean": float(np.mean(eos_distances)),
    }

    # 4c) Degenerate-input path returns trivial distances (≈ 0 between
    # collapsed centroids), not garbage. The 1e-6 ridge keeps the inverse
    # well-defined; the projected centroids are themselves zero so the
    # Mahalanobis distance lands at zero modulo float noise.
    arr_singular = np.zeros((3, 1, 64), dtype=np.float32)
    singular_state = _build_context_pooled_mahal_state(arr_singular, k=4)
    if singular_state is not None:
        d_singular = _context_mahal_with_pooled_cov(
            arr_singular[0], arr_singular[1], singular_state, 0, 1
        )
        assert abs(d_singular) < 1e-3, (
            f"collapsed-centroid Mahalanobis was {d_singular} (expected ≈ 0)"
        )
        digest["degenerate_pooled_state"] = {
            "state_returned": True,
            "distance_at_collapsed_centroid_pair": float(d_singular),
        }
    else:
        digest["degenerate_pooled_state"] = {"state_returned": False}

    # 4d) Single-vector subset path produces FINITE non-zero distances when
    # the centroids ARE distinct (this is the meaningful end_of_system run).
    arr_eos_single = np.array(
        [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]], dtype=np.float32
    )
    s3 = _build_context_pooled_mahal_state(arr_eos_single, k=2)
    if s3 is not None:
        d_ab = _context_mahal_with_pooled_cov(arr_eos_single[0], arr_eos_single[1], s3, 0, 1)
        assert np.isfinite(d_ab) and d_ab > 0.05, (
            f"single-vector subset Mahalanobis was {d_ab} (expected positive finite)"
        )
        digest["single_vector_subset_mahal"] = float(d_ab)

    # 5) Gaussian sym-KL and W2 finite + positive
    gkl = _gaussian_sym_kl_in_subspace(X1, X2, k=8)
    w2 = _bures_wasserstein2(X1, X2, k=8)
    assert np.isfinite(gkl) and gkl > 0, f"gauss_kl was {gkl}"
    assert np.isfinite(w2) and w2 > 0, f"bures_w2 was {w2}"
    digest["gauss_kl_w2"] = {"sym_kl": float(gkl), "wass2": float(w2)}

    # 5b) MMD permutation null populated — pick a tiny synthetic (n_cond, n_q,
    # H) cloud, run the summary, confirm per-pair p-values + pooled summary
    # land in [0, 1] / are finite.
    arr_mmd, cond_ids_mmd, _ = _synthetic_clouds(n_cond=4, n_q=20, hidden=32)
    perm = _mmd_permutation_summary(
        arr_mmd, cond_ids_mmd, n_perm=30, variant="raw", n_pair_samples=4
    )
    assert perm["n_pair_samples_done"] >= 1, "MMD permutation summary returned no pairs"
    assert all(0.0 <= e["p_value"] <= 1.0 for e in perm["per_pair"]), "p_value out of [0, 1]"
    assert np.isfinite(perm["pooled_null_p95"]), "pooled_null_p95 was non-finite"
    digest["mmd_permutation"] = {
        "n_pair_samples_done": perm["n_pair_samples_done"],
        "min_p_value": min(e["p_value"] for e in perm["per_pair"]),
        "pooled_null_p95": perm["pooled_null_p95"],
    }

    # 5c) raw + centered variants both compute distinct matrices.
    arr_var, cond_ids_var, _ = _synthetic_clouds(n_cond=4, n_q=20, hidden=32)
    raw_payload = _compute_metric_matrix(
        arr_var,
        cond_ids_var,
        metric="cosine",
        extraction_point="last_prompt",
        pca_k=8,
        variant="raw",
    )
    cen_payload = _compute_metric_matrix(
        arr_var,
        cond_ids_var,
        metric="cosine",
        extraction_point="last_prompt",
        pca_k=8,
        variant="centered",
    )
    assert raw_payload["variant"] == "raw"
    assert cen_payload["variant"] == "centered"
    raw_val = raw_payload["matrix"][cond_ids_var[0]][cond_ids_var[1]]
    cen_val = cen_payload["matrix"][cond_ids_var[0]][cond_ids_var[1]]
    assert abs(raw_val - cen_val) > 1e-6, (
        f"centered variant indistinguishable from raw: {raw_val} vs {cen_val}"
    )
    digest["variants"] = {"raw_sample": raw_val, "centered_sample": cen_val}

    # 5d) Length-partialed LOOCV is consistent with the headline metric.
    # When predictor and DV are both monotone in the same covariate AND the
    # length covariate is constant, the length-partial LOOCV must return
    # the same answer as the bare-OLS LOOCV (no residualization happens).
    arr_cv, cond_ids_cv, fake_dg_cv = _synthetic_clouds(n_cond=6, n_q=20, hidden=32)
    pairs_cv = [
        (a, b) for i, a in enumerate(cond_ids_cv) for j, b in enumerate(cond_ids_cv) if i != j
    ]
    cdmat_cv = np.zeros((len(cond_ids_cv), len(cond_ids_cv)))
    for i in range(len(cond_ids_cv)):
        for j in range(len(cond_ids_cv)):
            if i == j:
                continue
            cdmat_cv[i, j] = _centroid_cosine_distance(arr_cv[i], arr_cv[j])
    name_to_idx_cv = {n: i for i, n in enumerate(cond_ids_cv)}
    xv_cv = np.array([cdmat_cv[name_to_idx_cv[a], name_to_idx_cv[b]] for a, b in pairs_cv])
    yv_cv = np.array([fake_dg_cv[name_to_idx_cv[a], name_to_idx_cv[b]] for a, b in pairs_cv])
    src_cv = [a for a, _ in pairs_cv]
    tgt_cv = [b for _, b in pairs_cv]
    # Three regimes: bare CV vs length-partial CV.
    #   (a) covar independent of x,y     → length-partial CV ≈ bare CV
    #   (b) covar perfectly tracks x     → length-partial CV ≪ bare CV
    #       (the signal vanishes once length is partialed out — confound case)
    cv_bare = _loocv_r2(xv_cv, yv_cv, src_cv, tgt_cv, covar=None)
    rng_cv = np.random.default_rng(7)
    covar_indep = rng_cv.normal(size=xv_cv.shape)
    cv_lp_indep = _loocv_r2(xv_cv, yv_cv, src_cv, tgt_cv, covar=covar_indep)
    assert abs(cv_bare - cv_lp_indep) < 0.2, (
        f"length-partial CV on independent covar should track bare CV: "
        f"bare={cv_bare:.3f} length-partial={cv_lp_indep:.3f}"
    )
    cv_lp_confound = _loocv_r2(xv_cv, yv_cv, src_cv, tgt_cv, covar=xv_cv.copy())
    assert cv_lp_confound < cv_bare - 0.1, (
        f"length-partial CV must collapse when covar==predictor (confound case): "
        f"bare={cv_bare:.3f} length-partial-confound={cv_lp_confound:.3f}"
    )
    digest["loocv_length_partialed"] = {
        "bare_R2": cv_bare,
        "length_partial_R2_independent_covar": cv_lp_indep,
        "length_partial_R2_confound_covar": cv_lp_confound,
    }

    # 6) End-to-end: synthetic regression — cosine_dist of synthetic clouds
    #    should rank-correlate with the synthetic ΔG matrix.
    arr, cond_ids, fake_dg = _synthetic_clouds()
    pairs = [(a, b) for i, a in enumerate(cond_ids) for j, b in enumerate(cond_ids) if i != j]
    cdmat = np.zeros((len(cond_ids), len(cond_ids)))
    for i, _ in enumerate(cond_ids):
        for j, _ in enumerate(cond_ids):
            if i == j:
                continue
            cdmat[i, j] = _centroid_cosine_distance(arr[i], arr[j])
    name_to_idx = {n: i for i, n in enumerate(cond_ids)}
    xv = np.array([cdmat[name_to_idx[a], name_to_idx[b]] for a, b in pairs])
    yv = np.array([fake_dg[name_to_idx[a], name_to_idx[b]] for a, b in pairs])
    # length covariate is constant here; use a flat covar so partial = bare Spearman
    rho, _p = _length_partial(xv, yv, np.zeros_like(xv) + 1.0)
    digest["synthetic_regression_rho"] = float(rho)
    assert abs(rho) > 0.3, f"synthetic regression rho was {rho} (expected substantial)"
    # CV must run cleanly too
    cv = _loocv_r2(xv, yv, [a for a, _ in pairs], [b for _, b in pairs], covar=np.ones_like(xv))
    digest["synthetic_cv_r2"] = float(cv)

    # 7) hidden_states indexing matches a reference forward-hook — this
    # is the round-1 off-by-one bug regression test, exercised against a
    # tiny dummy GPT2 (no Qwen download, runs CPU-side in a few seconds).
    # The forward-hook on `transformer.h[L]` captures the OUTPUT of block L;
    # `hidden_states[L+1]` from `output_hidden_states=True` must match that
    # exact tensor. If we ever revert to indexing `hidden_states[L]`, this
    # assertion fires loud.
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok_ref = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        mdl_ref = AutoModel.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        mdl_ref.eval()
        ids = tok_ref("hello world from issue 493", return_tensors="pt")
        hook_capture: dict[int, torch.Tensor] = {}

        def _make_hook(layer_idx: int):
            def _hook(_mod, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                hook_capture[layer_idx] = hs.detach().clone()

            return _hook

        target_layer = 1  # 0-indexed; tiny-random-gpt2 has 4 blocks
        h = mdl_ref.h[target_layer].register_forward_hook(_make_hook(target_layer))
        try:
            with torch.no_grad():
                fwd = mdl_ref(**ids, output_hidden_states=True)
        finally:
            h.remove()
        from_hook = hook_capture[target_layer]
        from_tuple_off_by_one = fwd.hidden_states[target_layer]  # WRONG
        from_tuple_correct = fwd.hidden_states[target_layer + 1]  # OURS
        assert torch.allclose(from_hook, from_tuple_correct, atol=1e-6), (
            "hidden_states[L+1] no longer matches the block-L forward-hook output"
        )
        assert not torch.allclose(from_hook, from_tuple_off_by_one, atol=1e-6), (
            "hidden_states[L] now matches the hook — convention changed upstream?"
        )
        digest["hidden_states_indexing"] = {
            "convention": "hidden_states[L+1] == block[L] output",
            "ok": True,
        }
    except Exception as e:
        digest["hidden_states_indexing"] = {
            "ok": False,
            "reason": f"reference check skipped: {e!r}",
        }

    return digest


# ───────────────────────── CLI driver ─────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Issue #493 extraction-point × metric × layer bake-off "
        "for marker-transfer prediction."
    )
    p.add_argument(
        "--phase",
        # `extraction` is a synonym for `extract` (matches the docs / the
        # report-back section header "extraction"); both route through the
        # same branch below.
        choices=("all", "extract", "extraction", "metrics", "regress", "figures", "smoke"),
        default="all",
        help="Which phase to run. 'all' runs extract → metrics → regress → figures. "
        "'extract' and 'extraction' are synonyms. 'smoke' runs the synthetic CPU sanity "
        "check only.",
    )
    p.add_argument(
        "--extraction-points",
        nargs="+",
        default=list(DEFAULT_EXTRACTION_POINTS),
        choices=list(DEFAULT_EXTRACTION_POINTS),
        help="Which extraction points to compute (default: all 3).",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAYERS),
        help="Which residual layers to extract / score (default: 0 5 7 11 14 15 21 27).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(ALL_METRICS),
        choices=list(ALL_METRICS),
        help="Which metrics to compute.",
    )
    p.add_argument(
        "--transformations",
        nargs="+",
        default=None,
        help="Optional subset of cond cids (e.g. A1 A2 B1). Default: all 16.",
    )
    p.add_argument(
        "--n-probes",
        type=int,
        default=50,
        help="Subset of the 50 Q_test probes. Use a small value (e.g. 4) for a pod smoke slice.",
    )
    p.add_argument(
        "--max-response-tokens",
        type=int,
        default=512,
        # 512 covers Qwen-2.5-7B's natural ~150-token response median with
        # ~3x headroom (round-1's 128 truncated below the median and biased
        # mean_response toward early tokens). #460 uses 1024 for marker
        # training generation; we keep 512 here because mean_response is a
        # representational summary, not the marker-leakage DV — but the
        # `extraction_truncation.json` summary surfaces the rate so a real
        # run that drifts long can be re-launched at 1024.
        help=(
            "max_new_tokens for the mean_response greedy decode. Default 512 "
            "covers Qwen-2.5-7B's natural ~150-token response median with "
            "headroom; #460 used 1024 for marker-leakage training generation. "
            "Truncation rate is logged in extraction_truncation.json."
        ),
    )
    p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(DEFAULT_ARMS))
    p.add_argument("--epochs", nargs="+", type=int, default=list(DEFAULT_EPOCHS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help=(
            "Bind CUDA_VISIBLE_DEVICES=<gpu-id> BEFORE any CUDA call, then load the "
            "model on cuda:0. Matches the i474/i415 parallel-launch convention; "
            "leave unset to inherit the caller's CUDA_VISIBLE_DEVICES."
        ),
    )
    p.add_argument("--pca-k", type=int, default=PCA_DEFAULT_K)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the model load + extraction; only run metric/regression "
        "smoke (synthetic clouds + import/plumbing checks).",
    )
    return p


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — top-level CLI phase dispatcher.
    args = _build_argparser().parse_args(argv)
    # BIND CUDA_VISIBLE_DEVICES *BEFORE* any cuda call (project convention,
    # see scripts/i474_*.py / scripts/recompute_predictors_i415.py + the
    # CLAUDE.md `+gpu_id=N` clobber gotcha). Once bound the local device is
    # always cuda:0 from the model's point of view.
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda:0"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Persist a meta.json snapshot at every entry (overwritten = fine).
    BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        BAKEOFF_DIR / "meta.json",
        {
            "schema_version": 1,
            "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
            "git_sha": _git_sha(),
            "env": _env_versions(),
            "started_at": _now_iso(),
        },
    )

    # SMOKE path — CPU only.
    if args.phase == "smoke" or args.dry_run:
        logger.info("Running synthetic metric/regression smoke (CPU-only)…")
        digest = dry_run_smoke()
        out = BAKEOFF_DIR / "smoke_digest.json"
        _write_json_atomic(
            out,
            {
                "schema_version": 1,
                "digest": digest,
                "git_sha": _git_sha(),
                "timestamp_utc": _now_iso(),
                "env": _env_versions(),
            },
        )
        logger.info("Smoke OK; digest at %s", out)
        for k, v in digest.items():
            logger.info("  %s: %s", k, v)
        if args.phase == "smoke":
            return 0

    # Bootstrap env (HF_TOKEN, HF_HOME).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # EXTRACTION phase
    if args.phase in ("all", "extract", "extraction"):
        if args.dry_run:
            logger.info("--dry-run: SKIPPING extraction (no model load).")
        else:
            logger.info(
                "Extraction: points=%s layers=%s transformations=%s n_probes=%d",
                args.extraction_points,
                args.layers,
                args.transformations or "ALL 16",
                args.n_probes,
            )
            run_extraction(
                extraction_points=tuple(args.extraction_points),
                layers=tuple(args.layers),
                transformations=tuple(args.transformations) if args.transformations else None,
                n_probes=args.n_probes,
                max_response_tokens=args.max_response_tokens,
                device=args.device,
                overwrite=args.overwrite,
            )

    if args.phase in ("extract", "extraction"):
        return 0

    # Reload from disk (decouples phases — safe on resume).
    activations_by_point = load_activations_from_disk(
        tuple(args.extraction_points), tuple(args.layers)
    )
    if not activations_by_point:
        logger.warning("No activations on disk; run --phase extract first (with GPU).")
        return 1

    # Correctness cross-check: last-token cosine must match #406's existing.
    # The check is ENFORCED (raises) on full real-data runs (16 conds × 50
    # probes) where the recipes are supposed to be byte-identical. On
    # subset runs (--transformations / --n-probes < 50) it's logged-only:
    # the cond/probe subset changes the cosine values, so a mismatch is
    # expected, not a bug.
    if "last_prompt" in activations_by_point and not args.dry_run:
        existing = _load_existing_cosine_matrices(tuple(args.layers))
        if existing:
            sample_payload = next(
                p
                for p in activations_by_point["last_prompt"].values()
                if isinstance(p, dict) and "activations" in p
            )
            n_cond_loaded = sample_payload["activations"].shape[0]
            n_q_loaded = sample_payload["activations"].shape[1]
            strict = args.transformations is None and n_cond_loaded == 16 and n_q_loaded == 50
            try:
                check = reproduce_last_token_cosine_check(
                    activations_by_point["last_prompt"],
                    existing,
                    cond_ids=activations_by_point["last_prompt"][args.layers[0]]["cond_ids"],
                    strict=strict,
                )
            except AssertionError as e:
                # Persist the failure context before re-raising so the
                # operator can diagnose without re-running the extraction.
                _write_json_atomic(
                    BAKEOFF_DIR / "cosine_cross_check.json",
                    {
                        "schema_version": 1,
                        "tolerance": COSINE_REPRO_TOLERANCE,
                        "strict": True,
                        "failed": True,
                        "failure_reason": str(e),
                        "n_cond_loaded": int(n_cond_loaded),
                        "n_probes_loaded": int(n_q_loaded),
                        "git_sha": _git_sha(),
                        "timestamp_utc": _now_iso(),
                    },
                )
                raise
            _write_json_atomic(
                BAKEOFF_DIR / "cosine_cross_check.json",
                {
                    "schema_version": 1,
                    "tolerance": COSINE_REPRO_TOLERANCE,
                    "strict": strict,
                    "n_cond_loaded": int(n_cond_loaded),
                    "n_probes_loaded": int(n_q_loaded),
                    "per_layer": check,
                    "git_sha": _git_sha(),
                    "timestamp_utc": _now_iso(),
                },
            )

    # METRICS phase
    if args.phase in ("all", "metrics"):
        run_metrics(
            activations_by_point=activations_by_point,
            metrics=tuple(args.metrics),
            pca_k=args.pca_k,
            overwrite=args.overwrite,
        )

    if args.phase == "metrics":
        return 0

    # REGRESSION phase — cond_ids comes from a NON-end_of_system checkpoint
    # so the regression covers the full 16-cond grid (end_of_system metric
    # files carry the Class-A subpanel and are handled by run_regression on
    # their own subpanel).
    cond_ids: list[str] | None = None
    for pt in args.extraction_points:
        layer_map = activations_by_point.get(pt) or {}
        if pt == "end_of_system" or not layer_map:
            continue
        any_L = next(iter(layer_map))
        cond_ids = layer_map[any_L]["cond_ids"]
        break
    if cond_ids is None:
        # Fall back: only end_of_system checkpoints present (e.g. an early-
        # phase smoke that just exercised Class A). Use its subpanel as the
        # cond set so regression at least runs; clearly flagged downstream.
        for pt, layer_map in activations_by_point.items():
            if not layer_map:
                continue
            any_L = next(iter(layer_map))
            cond_ids = layer_map[any_L]["cond_ids"]
            logger.warning(
                "Regression cond_ids fell back to %s subpanel (%s) — "
                "no full-grid extraction point on disk.",
                pt,
                cond_ids,
            )
            break
    if cond_ids is None:
        raise RuntimeError(
            "No usable activations on disk for any extraction point; "
            "re-run --phase extract before --phase regress."
        )

    all_cells = run_regression(
        cond_ids=cond_ids,
        arms=tuple(args.arms),
        epochs=tuple(args.epochs),
        overwrite=args.overwrite,
    )

    # Winner + summary
    headline = all_cells.get("loc_ep1")
    winner = select_winner(headline) if headline else None
    grid_path = BAKEOFF_DIR / "bakeoff_grid.json"
    _write_json_atomic(
        grid_path,
        {
            "schema_version": 1,
            "cells": all_cells,
            "winner_loc_ep1": winner,
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
        },
    )
    logger.info("Wrote %s", grid_path)
    if winner:
        sub = winner.get("sub_predictor")
        logger.info(
            "WINNER (loc_ep1): %s · L%d · %s%s — CV R² = %.3f, "
            "rho_ns(ΔG) = %+.3f, rho_full(g_logp) = %+.3f",
            winner["extraction_point"],
            winner["layer"],
            winner["metric"],
            f" · {sub}" if sub else "",
            winner["cv_full_deltag"],
            winner["rho_nonstylized_deltag"],
            winner["rho_full_glogp"],
        )
    else:
        logger.warning("No predictor survived the non-stylized + base-prior-safe check.")

    if args.phase == "regress":
        return 0

    # FIGURES phase
    if args.phase in ("all", "figures"):
        emit_figures(all_cells, tuple(args.extraction_points), tuple(args.layers))

    return 0


if __name__ == "__main__":
    sys.exit(main())
