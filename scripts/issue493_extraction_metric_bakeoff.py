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
CLOUD_METRICS: tuple[str, ...] = (
    "fisher",
    "mmd",
    "c2st",
    "delta_spec",
    "gauss_kl",
    "wass2",
)
CENTROID_METRICS: tuple[str, ...] = ("cosine", "euclidean", "mahal")
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
) -> dict[str, dict[int, torch.Tensor]]:
    """For one (cond, question) extract residual activations at the requested
    extraction points × layers.

    Returns
    -------
    dict
      ``{point: {layer: tensor(H,) for layer in layers} for point in extraction_points}``

      For ``end_of_system`` on non-Class-A conditions, the value is an empty
      dict (signals: N/A at this (cond, point), the cloud aggregator drops it).
    """
    import torch

    system_text, full_text = _build_prompts_for_extraction(
        cond, question, tokenizer, class_d_rewrites, "all"
    )

    result: dict[str, dict[int, torch.Tensor]] = {p: {} for p in extraction_points}

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
        # hidden_states is a (n_layers+1,) tuple; index 0 = embeddings.
        for L in layers:
            if len(out.hidden_states) <= L:
                raise IndexError(
                    f"layer={L} out of range; hidden_states has {len(out.hidden_states)} entries"
                )
            result["end_of_system"][L] = out.hidden_states[L][0, last_pos, :].float().cpu()
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
                    for L in layers:
                        result["last_prompt"][L] = (
                            fwd.hidden_states[L][0, prompt_len - 1, :].float().cpu()
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
                return result

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
            for L in layers:
                hs = fwd.hidden_states[L][0]  # (full_len, H)
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
            for L in layers:
                result["last_prompt"][L] = fwd.hidden_states[L][0, prompt_len - 1, :].float().cpu()
            del fwd

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


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

    for ci, cond in enumerate(active_conds):
        t_c = time.time()
        for qi, q in enumerate(questions):
            try:
                row = _extract_one(
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
            if "mean_response" in extraction_points:
                total_response_rows += 1
        logger.info(
            "cond %d/%d %s in %.1fs", ci + 1, len(active_conds), cond.cid, time.time() - t_c
        )

    if total_response_rows:
        logger.info(
            "Response truncation rate: %d/%d (%.1f%%)",
            truncation_count,
            total_response_rows,
            100.0 * truncation_count / total_response_rows,
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
                logger.info("Skipping existing %s (use --overwrite to redo)", out_path)
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
    covariance in the k-d subspace + inverse via solve.
    """
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    Ya, Yb = _pair_pca_subspace(Xa, Xb, k)
    mu_a = Ya.mean(axis=0)
    mu_b = Yb.mean(axis=0)
    Cov = 0.5 * (np.cov(Ya.T, ddof=1) + np.cov(Yb.T, ddof=1))
    # ridge for numerical stability
    Cov += 1e-6 * np.eye(Cov.shape[0])
    diff = mu_a - mu_b
    inv_cov_diff = np.linalg.solve(Cov, diff)
    return float(np.sqrt(float(diff @ inv_cov_diff)))


def _fisher_distance(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """PCA-whitened Fisher distance: mean-difference scaled by within-class
    pooled covariance, in the top-k PCA subspace.

    Identical signal to Mahalanobis here (both: |μ_a−μ_b|_{Σ^{-1}} in the
    top-k subspace). Kept as a SEPARATE metric for the panel because the
    spec calls them out by name and a reviewer expects both columns.
    Numerically they may differ slightly via PCA-subspace ordering;
    semantically they're equivalent under pooled covariance.
    """
    return _centroid_mahal(Xa, Xb, k)


def _rbf_mmd_squared(
    Xa: np.ndarray, Xb: np.ndarray, n_perm: int = MMD_PERMUTATIONS, rng=None
) -> float:
    """Biased RBF-MMD² with median-heuristic bandwidth.

    Returns the biased MMD² estimate as the predictor scalar; the
    permutation null is computed and logged in the metric payload so a
    reviewer can sanity-check the kernel scale.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    na, nb = len(Xa), len(Xb)
    if na < 2 or nb < 2:
        return float("nan")
    Z = np.vstack([Xa, Xb])
    # median heuristic on pairwise sqdists (subsample if huge — we have n≤100)
    sq = np.sum(Z**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2 * Z @ Z.T
    np.fill_diagonal(D2, np.nan)
    median_sq = np.nanmedian(D2)
    sigma2 = max(float(median_sq), 1e-8)
    K = np.exp(-D2 / sigma2)
    np.fill_diagonal(K, 1.0)  # restore diagonal after fill
    Kaa = K[:na, :na]
    Kbb = K[na:, na:]
    Kab = K[:na, na:]
    mmd2 = Kaa.mean() + Kbb.mean() - 2 * Kab.mean()
    return float(mmd2)


def _c2st_auc(Xa: np.ndarray, Xb: np.ndarray, folds: int = C2ST_FOLDS) -> float:
    """Cross-validated linear-probe classifier-2-sample test AUC.

    1.0 = perfectly separable, 0.5 = indistinguishable. We use a regularized
    logistic regression (sklearn) with stratified K-fold; AUC averaged over
    folds.
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
    return float(np.mean(aucs))


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


def _compute_metric_matrix(  # noqa: C901 — per-metric dispatcher; one branch per metric, intentionally flat.
    activations: np.ndarray,
    cond_ids: list[str],
    metric: str,
    extraction_point: str,
    pca_k: int,
) -> dict:
    """Compute the (n_cond × n_cond) pairwise predictor matrix for one metric.

    Returns a dict with the matrix and any per-(metric) auxiliary outputs
    (e.g. Δ-spectrum produces 3 sub-predictors stored as separate matrices).

    activations: (n_cond, n_q, H) — for end_of_system, n_q == 1.
    """
    n_cond = activations.shape[0]
    is_centroid = metric in CENTROID_METRICS
    if not is_centroid and extraction_point == "end_of_system":
        # Cloud metrics are N/A at end_of_system — n_q==1, no cloud exists.
        return {"matrix": None, "n_a": None, "metric_value_is_none": True}

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
            "matrices": {
                k: {
                    cond_ids[i]: {cond_ids[j]: ms[k][i][j] for j in range(n_cond)}
                    for i in range(n_cond)
                }
                for k in ms
            },
            "sub_predictors": list(ms.keys()),
        }

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
            elif metric == "fisher":
                d = _fisher_distance(Xa, Xb, pca_k)
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
) -> None:
    """Compute every (extraction_point × layer × metric) distance matrix and
    checkpoint EACH ONE to disk immediately."""
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    for pt, layer_map in activations_by_point.items():
        for L, payload in layer_map.items():
            cond_ids = payload["cond_ids"]
            arr = payload["activations"]
            for m in metrics:
                # Skip cloud metrics at end_of_system entirely (output N/A).
                if pt == "end_of_system" and m in CLOUD_METRICS:
                    out_path = METRIC_DIR / f"{pt}__layer{L}__{m}.json"
                    if out_path.exists() and not overwrite:
                        continue
                    _write_json_atomic(
                        out_path,
                        {
                            "schema_version": 1,
                            "extraction_point": pt,
                            "layer": L,
                            "metric": m,
                            "matrix": None,
                            "n_a": "cloud metric N/A at end_of_system (one vector per cond)",
                            "git_sha": _git_sha(),
                            "timestamp_utc": _now_iso(),
                        },
                    )
                    continue
                out_path = METRIC_DIR / f"{pt}__layer{L}__{m}.json"
                if out_path.exists() and not overwrite:
                    logger.info("Skipping existing %s", out_path)
                    continue
                t0 = time.time()
                res = _compute_metric_matrix(arr, cond_ids, m, pt, pca_k)
                payload_out = {
                    "schema_version": 1,
                    "extraction_point": pt,
                    "layer": L,
                    "metric": m,
                    "pca_k": pca_k,
                    "cond_ids": cond_ids,
                    "git_sha": _git_sha(),
                    "timestamp_utc": _now_iso(),
                    **res,
                }
                _write_json_atomic(out_path, payload_out)
                logger.info("Wrote %s in %.2fs", out_path, time.time() - t0)


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


def _loocv_r2(x: np.ndarray, y: np.ndarray, cond_ids_a: list[str], cond_ids_b: list[str]) -> float:
    """Leave-one-context-out CV R² (the i474 fig9 pattern).

    For each cond C in the 16 transformations, hold out all pairs where
    either source==C or target==C, fit OLS on the remainder, predict
    held-out, compute (1 − SSE / SST).
    """
    n = len(x)
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
    metric_payload: dict, pairs: list[tuple[str, str]], sub_predictor: str | None
) -> np.ndarray | None:
    """Read a distance value per pair from a metric_payload (one matrix file).

    Returns None if matrix is None (N/A — cloud metric at end_of_system).
    For Δ-spectrum, ``sub_predictor`` ∈ {"mean_norm", "coherence", "effective_dim"}.
    """
    if "matrix" in metric_payload and metric_payload["matrix"] is None:
        return None  # N/A — cloud metric at end_of_system
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
    (one row per (extraction_point, layer, metric, sub_predictor))."""
    rows = []
    for p in metric_files:
        payload = json.loads(p.read_text())
        pt = payload["extraction_point"]
        L = payload["layer"]
        m = payload["metric"]
        if "matrices" in payload:
            for sub in payload["sub_predictors"]:
                rows.append(
                    {
                        "extraction_point": pt,
                        "layer": L,
                        "metric": m,
                        "sub_predictor": sub,
                        "file": str(p),
                    }
                )
        else:
            rows.append(
                {
                    "extraction_point": pt,
                    "layer": L,
                    "metric": m,
                    "sub_predictor": None,
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
            # Build full and non-stylized pair vectors
            pairs_full = _pairs(cond_ids, nonstylized_only=False)
            pairs_ns = _pairs(cond_ids, nonstylized_only=True)
            dg_full = np.array([G[a][b]["delta_g"] for a, b in pairs_full])
            g_full = np.array([G[a][b]["g_logprob"] for a, b in pairs_full])
            ln_full = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs_full])
            dg_ns = np.array([G[a][b]["delta_g"] for a, b in pairs_ns])
            g_ns = np.array([G[a][b]["g_logprob"] for a, b in pairs_ns])
            ln_ns = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs_ns])
            src_full = [a for a, _ in pairs_full]
            tgt_full = [b for _, b in pairs_full]
            src_ns = [a for a, _ in pairs_ns]
            tgt_ns = [b for _, b in pairs_ns]

            sat_ns = _saturation_fraction(g_ns)
            sat_full = _saturation_fraction(g_full)

            entries = []
            for desc in predictors:
                payload = json.loads(Path(desc["file"]).read_text())
                xv_full = _materialize_predictor_vector(payload, pairs_full, desc["sub_predictor"])
                xv_ns = _materialize_predictor_vector(payload, pairs_ns, desc["sub_predictor"])
                if xv_full is None or xv_ns is None:
                    entries.append({**desc, "status": "N/A (no matrix at this point)"})
                    continue

                # Length-partial Spearman, per panel × DV
                rho_full_dg, p_full_dg = _length_partial(xv_full, dg_full, ln_full)
                rho_full_g, p_full_g = _length_partial(xv_full, g_full, ln_full)
                rho_ns_dg, p_ns_dg = _length_partial(xv_ns, dg_ns, ln_ns)
                rho_ns_g, p_ns_g = _length_partial(xv_ns, g_ns, ln_ns)

                # Leave-one-context-out CV on the FULL panel (the i474 fig9
                # convention). For the winner-selection ranking we use the
                # CV against ΔG on full.
                cv_full_dg = _loocv_r2(xv_full, dg_full, src_full, tgt_full)
                cv_full_g = _loocv_r2(xv_full, g_full, src_full, tgt_full)
                cv_ns_dg = _loocv_r2(xv_ns, dg_ns, src_ns, tgt_ns)
                cv_ns_g = _loocv_r2(xv_ns, g_ns, src_ns, tgt_ns)

                entries.append(
                    {
                        **desc,
                        "n_full": len(xv_full),
                        "n_nonstylized": len(xv_ns),
                        "rho_full_deltag": float(rho_full_dg),
                        "p_full_deltag": float(p_full_dg),
                        "rho_full_glogp": float(rho_full_g),
                        "p_full_glogp": float(p_full_g),
                        "rho_nonstylized_deltag": float(rho_ns_dg),
                        "p_nonstylized_deltag": float(p_ns_dg),
                        "rho_nonstylized_glogp": float(rho_ns_g),
                        "p_nonstylized_glogp": float(p_ns_g),
                        "cv_full_deltag": float(cv_full_dg),
                        "cv_full_glogp": float(cv_full_g),
                        "cv_nonstylized_deltag": float(cv_ns_dg),
                        "cv_nonstylized_glogp": float(cv_ns_g),
                    }
                )

            cell_payload = {
                "schema_version": 1,
                "arm": arm,
                "epoch": ep,
                "n_pairs_full": len(pairs_full),
                "n_pairs_nonstylized": len(pairs_ns),
                "saturation_frac_full": sat_full,
                "saturation_frac_nonstylized": sat_ns,
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
) -> dict[int, dict]:
    """Re-compute last-token cosine distances from our fresh activations and
    diff against the existing eval_results/issue_406/cosine/C_L*.json.

    The existing #406 recipe = cosine-distance of cond-mean activations
    across 50 probes at last-prompt-token. Our last_prompt extraction is
    the same recipe; the two must agree within COSINE_REPRO_TOLERANCE.

    Returns a per-layer diff summary; logs WARN if any layer fails. We do
    NOT raise — the cross-check is a sanity assertion, but a smoke run on
    n_probes < 50 or a transformations subset will deliberately mismatch.
    Real-run callers should inspect the returned dict.
    """
    out: dict[int, dict] = {}
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
        level = logging.INFO if ok else logging.WARNING
        logger.log(
            level,
            "Cosine cross-check L%d: max |diff| = %.2e over %d pairs (ok=%s)",
            L,
            max_diff,
            n_pairs,
            ok,
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
    auc_id = _c2st_auc(Xa_id, Xb_id)
    assert abs(auc_id - 0.5) < 0.25, f"C2ST on same-dist clouds was {auc_id}"
    mmd_id = _rbf_mmd_squared(Xa_id, Xb_id)
    assert mmd_id < 0.10, f"MMD on same-dist clouds was {mmd_id}"
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
        "c2st_auc": auc_id,
        "mmd2": mmd_id,
        "cosine_dist_tight_same_mean": cd_tight,
        "euclidean_tight_same_mean": ed_tight,
    }

    # 2) Well-separated clouds → cosine ≈ 1 (orthogonal), C2ST ≈ 1
    X1 = rng.normal(loc=10.0, size=(30, 64))
    X2 = rng.normal(loc=-10.0, size=(30, 64))
    cd = _centroid_cosine_distance(X1, X2)
    auc_sep = _c2st_auc(X1, X2)
    mmd_sep = _rbf_mmd_squared(X1, X2)
    assert cd > 0.2, f"separated clouds cosine_dist was {cd} (expected substantial)"
    assert auc_sep > 0.95, f"separated clouds C2ST AUC was {auc_sep}"
    assert mmd_sep > 0.05, f"separated clouds MMD was {mmd_sep}"
    digest["separated_clouds"] = {
        "cosine_dist": cd,
        "c2st_auc": auc_sep,
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

    # 4) Fisher / Mahalanobis well-conditioned at n ≪ d (n=30, d=64, k=8)
    fish = _fisher_distance(X1, X2, k=8)
    mahal = _centroid_mahal(X1, X2, k=8)
    assert np.isfinite(fish) and fish > 0, f"fisher dist was {fish}"
    assert np.isfinite(mahal) and mahal > 0, f"mahal was {mahal}"
    digest["fisher_mahal"] = {"fisher": float(fish), "mahal": float(mahal)}

    # 5) Gaussian sym-KL and W2 finite + positive
    gkl = _gaussian_sym_kl_in_subspace(X1, X2, k=8)
    w2 = _bures_wasserstein2(X1, X2, k=8)
    assert np.isfinite(gkl) and gkl > 0, f"gauss_kl was {gkl}"
    assert np.isfinite(w2) and w2 > 0, f"bures_w2 was {w2}"
    digest["gauss_kl_w2"] = {"sym_kl": float(gkl), "wass2": float(w2)}

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
    cv = _loocv_r2(xv, yv, [a for a, _ in pairs], [b for _, b in pairs])
    digest["synthetic_cv_r2"] = float(cv)
    return digest


# ───────────────────────── CLI driver ─────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Issue #493 extraction-point × metric × layer bake-off "
        "for marker-transfer prediction."
    )
    p.add_argument(
        "--phase",
        choices=("all", "extract", "metrics", "regress", "figures", "smoke"),
        default="all",
        help="Which phase to run. 'all' runs extract → metrics → regress → figures. "
        "'smoke' runs the synthetic CPU sanity check only.",
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
        default=128,
        help="max_new_tokens for the mean_response greedy decode. 128 covers a typical "
        "Qwen response; bump for a real run that needs more headroom.",
    )
    p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(DEFAULT_ARMS))
    p.add_argument("--epochs", nargs="+", type=int, default=list(DEFAULT_EPOCHS))
    p.add_argument("--device", default="cuda:0")
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
    if args.phase in ("all", "extract"):
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

    if args.phase == "extract":
        return 0

    # Reload from disk (decouples phases — safe on resume).
    activations_by_point = load_activations_from_disk(
        tuple(args.extraction_points), tuple(args.layers)
    )
    if not activations_by_point:
        logger.warning("No activations on disk; run --phase extract first (with GPU).")
        return 1

    # Correctness cross-check: last-token cosine must match #406's existing.
    if "last_prompt" in activations_by_point and not args.dry_run:
        existing = _load_existing_cosine_matrices(tuple(args.layers))
        if existing:
            check = reproduce_last_token_cosine_check(
                activations_by_point["last_prompt"],
                existing,
                cond_ids=activations_by_point["last_prompt"][args.layers[0]]["cond_ids"],
            )
            _write_json_atomic(
                BAKEOFF_DIR / "cosine_cross_check.json",
                {
                    "schema_version": 1,
                    "tolerance": COSINE_REPRO_TOLERANCE,
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

    # REGRESSION phase
    # cond_ids must come from one of the loaded checkpoints; pick the first.
    first_pt = next(iter(activations_by_point.keys()))
    first_L = next(iter(activations_by_point[first_pt].keys()))
    cond_ids = activations_by_point[first_pt][first_L]["cond_ids"]
    # For end_of_system the cond list may be a strict subset (Class A only).
    # Use the FULL 16 cids for regression (since metrics for the other points
    # carry the full set). Reload from a non-end-of-system point if needed.
    for pt in args.extraction_points:
        if pt != "end_of_system" and pt in activations_by_point and activations_by_point[pt]:
            any_L = next(iter(activations_by_point[pt].keys()))
            cond_ids = activations_by_point[pt][any_L]["cond_ids"]
            break

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
