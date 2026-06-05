# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #488 Phase 1 — base-model predictors over the 27 conditions.

Computes per ``.claude/rules/persona-distance-metrics.md``:

* **RB sequence-level JS** (primary predictor): for each pair (T_i, T_j),
  sample R=8 responses from the base model under both T_i and T_j over the
  50-probe Q'_probe (= load_q_test_extended_50, paired with #406 D matrix),
  teacher-force each sampled response through BOTH conditioned models,
  per-position full-vocab JS, length-normalize, mean over samples × probes.
  Symmetric (responses sampled from both personas).
* **RB sequence-level KL both directions** (secondary): same machinery; report
  KL(narrow‖broad) (sample from narrow) and KL(broad‖narrow) (sample from
  broad), plus symmetric-KL = ½ their sum.
* **Cosine similarity on residual-stream activations** at layers {7, 14, 21, 27}
  per persona-vectors recipe (a): last-input-token activation of the base
  model conditioned on T_i, cosine-vs-T_j, mean over probes.
* **stylization_score[i]** (H2 graded covariate): RB sequence-level JS of T_i
  vs the no-system-prompt baseline (i.e. the default-assistant chat-template,
  no system turn). Reuses the SAME JS estimator with REFERENCE pinned to the
  baseline. Plan v2 §4.3(b).

Outputs (under ``eval_results/issue_488/predictors/``):

* ``js_matrix.json`` — ``{conditions: [...], JS: {cid_i: {cid_j: value}}}``,
  with the inherited 16×16 sub-grid LIFTED byte-identical from
  ``eval_results/issue_406/divergence/D_matrix.json["JS"]`` (plan §4.2 + §11
  assumption 3); only the 11 new ↔ everything cells are computed fresh.
* ``kl_matrix.json`` — analogous, both KL directions + symmetric-KL.
* ``cossim_matrix_layer{L}.json`` for L ∈ {7, 14, 21, 27}.
* ``stylization_score.json`` — ``{cid: js_to_no_system}`` over the 27 sources.
* ``is_stylized_source.json`` — binary lookup from
  ``i488_conditions.STRONG_STYLIZED_SOURCES``; written here so analyzer
  doesn't need to import.

Resume-safe per CLAUDE.md "Checkpoint per phase": per-file outputs are
written incrementally; re-runs skip filled cells.

CLI:
    uv run python scripts/i488_phase1_predictors.py
    uv run python scripts/i488_phase1_predictors.py --skip cossim   # only JS+KL+stylization
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make scripts/ importable for issue404_common.kill_vllm_workers.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_test_extended_50,
)
from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS,
    CONDITIONS_BY_ID,
    STRONG_STYLIZED_SOURCES,
    build_prompt_for_condition,
)

logger = logging.getLogger("i488.phase1")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = Path("eval_results/issue_488/predictors")
I406_D_MATRIX = Path("eval_results/issue_406/divergence/D_matrix.json")

R_SAMPLES = 8
N_PROBES = 50
COSINE_LAYERS = (7, 14, 21, 27)

# Inherited 16 conds (byte-identical with #406 D_matrix.json).
INHERITED_CIDS: frozenset[str] = frozenset(
    {c.cid for c in CONDITIONS if c.cls in {"A", "B", "C", "D"}}
)
NEW_CIDS: frozenset[str] = frozenset({c.cid for c in CONDITIONS if c.cls in {"E", "F", "G"}})

EPS = 1e-12
LN2 = torch.log(torch.tensor(2.0))


# ── RB sequence-level estimators ─────────────────────────────────────────


def _per_position_js(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """Per-position JS divergence (base-2) between two log-probability tensors.

    Args:
        p_log: (T, V) log-probabilities from model under T_i, teacher-forced.
        q_log: (T, V) log-probabilities from model under T_j on the same
            response.

    Returns:
        (T,) per-position JS values in [0, 1].

    Numerical handling: clamp probabilities to ``EPS`` before the log-mean to
    avoid `log(0)`; the result is clamped into [0, 1] for the bounded JS scale.
    """
    p = p_log.exp().clamp_min(EPS)
    q = q_log.exp().clamp_min(EPS)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1) / LN2
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1) / LN2
    js = 0.5 * (kl_pm + kl_qm)
    return js.clamp(min=0.0, max=1.0)


def _per_position_kl(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """Per-position KL(p ‖ q) (base-2), where p is the response-generating model.

    Args:
        p_log: (T, V) log-probabilities from the sample-from model.
        q_log: (T, V) log-probabilities from the comparison model.

    Returns:
        (T,) per-position KL(p ‖ q) values (nats / log2 = bits).
    """
    p = p_log.exp().clamp_min(EPS)
    return (p * (p_log - q_log)).sum(dim=-1) / LN2


def _sample_responses(
    model,
    tokenizer,
    prompt_text: str,
    n: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
) -> list[list[int]]:
    """Sample `n` responses from the model on a single prompt.

    Returns list of response token-id sequences (no prompt prefix).
    """
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=False).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    out: list[list[int]] = []
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=n,
            temperature=temperature,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    for row in gen:
        out.append(row[prompt_len:].tolist())
    return out


def _teacher_force_logprobs(
    model,
    tokenizer,
    prompt_text: str,
    response_ids: list[int],
) -> torch.Tensor:
    """Teacher-force the response through the model under the given prompt.

    Returns:
        (T, V) log-softmax tensor on CPU/float32. T == len(response_ids); the
        row at position t is the model's predicted next-token distribution
        AFTER seeing the prompt + response[:t]. This is the standard
        "predicts response[t]" alignment used by the RB estimator.
    """
    full_ids = tokenizer.encode(prompt_text, add_special_tokens=False) + list(response_ids)
    ids_t = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(ids_t).logits[0]  # (T_full, V)
    # The position predicting response[t] is at index `prompt_len - 1 + t`.
    prompt_len = len(full_ids) - len(response_ids)
    sliced = logits[prompt_len - 1 : prompt_len - 1 + len(response_ids), :]
    return torch.log_softmax(sliced.float().cpu(), dim=-1)


def _rb_block(
    model,
    tokenizer,
    cond_i,
    cond_j,
    probes: list[str],
    class_d_rewrites: dict | None,
    r_samples: int,
    max_new_tokens: int,
) -> dict[str, float]:
    """One (T_i, T_j) cell of the RB JS + both KL directions.

    Per CLAUDE.md persona-distance-metrics.md: sample r responses under each
    persona on each probe, teacher-force each response through BOTH models,
    per-position full-vocab divergence, length-normalize, mean over samples ×
    probes.
    """
    js_lens: list[float] = []
    kl_ij_lens: list[float] = []  # sample from i
    kl_ji_lens: list[float] = []  # sample from j

    for q in probes:
        prompt_i = build_prompt_for_condition(cond_i, q, tokenizer, class_d_rewrites)
        prompt_j = build_prompt_for_condition(cond_j, q, tokenizer, class_d_rewrites)

        # Sample r responses under each persona.
        r_from_i = _sample_responses(model, tokenizer, prompt_i, r_samples, max_new_tokens)
        r_from_j = _sample_responses(model, tokenizer, prompt_j, r_samples, max_new_tokens)

        # For each sample-from-i: teacher-force through both i and j.
        for ids in r_from_i:
            if len(ids) == 0:
                continue
            logp_i = _teacher_force_logprobs(model, tokenizer, prompt_i, ids)
            logp_j = _teacher_force_logprobs(model, tokenizer, prompt_j, ids)
            kl_ij_lens.append(float(_per_position_kl(logp_i, logp_j).mean().item()))
            js_lens.append(float(_per_position_js(logp_i, logp_j).mean().item()))

        # For each sample-from-j: teacher-force through both i and j.
        for ids in r_from_j:
            if len(ids) == 0:
                continue
            logp_i = _teacher_force_logprobs(model, tokenizer, prompt_i, ids)
            logp_j = _teacher_force_logprobs(model, tokenizer, prompt_j, ids)
            kl_ji_lens.append(float(_per_position_kl(logp_j, logp_i).mean().item()))
            js_lens.append(float(_per_position_js(logp_i, logp_j).mean().item()))

    def _mean_or_nan(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    js_mean = _mean_or_nan(js_lens)
    kl_ij_mean = _mean_or_nan(kl_ij_lens)
    kl_ji_mean = _mean_or_nan(kl_ji_lens)
    sym_kl = 0.5 * (kl_ij_mean + kl_ji_mean) if js_lens else float("nan")
    return {
        "js": js_mean,
        "kl_ij": kl_ij_mean,
        "kl_ji": kl_ji_mean,
        "sym_kl": sym_kl,
        "n_js_terms": len(js_lens),
    }


# ── Inheritance from #406 D_matrix.json ──────────────────────────────────


def _load_inherited_d_matrix() -> dict:
    """Read the #406 D_matrix.json; returns the dict (used for JS inheritance).

    Plan §11 assumption 3 + §4.2: the 16×16 inherited sub-grid of JS is lifted
    BYTE-IDENTICAL from #406; this keeps the paired-bootstrap valid.
    """
    if not I406_D_MATRIX.exists():
        raise FileNotFoundError(
            f"{I406_D_MATRIX} missing; #406 inheritance requires it. "
            "Sync from HF or re-run i406 Phase 1."
        )
    return json.loads(I406_D_MATRIX.read_text())


def _assert_i406_recipe_match(d: dict) -> None:
    """Raise loudly if #406's D_matrix recipe doesn't match what #488 computes.

    The 16×16 sub-grid lifted from #406 must use the SAME JS estimator config
    as the 11×* cells computed fresh by this script — otherwise the
    paired-bootstrap is invalid because the inherited and new cells were
    measured on different recipes (silent units mismatch).

    Hard-checks (raise on drift):
    * ``n_probes`` matches ``N_PROBES`` (#488 uses load_q_test_extended_50 = 50
      probes; #406 must too).
    * Inherited JS block has exactly the 16 INHERITED_CIDS as row keys (i.e.
      the A/B/C/D classes); a different set means a different condition recipe.
    * ``schema_version`` is present (sanity).

    Soft-warns (log but don't raise — these are advisory because #406's
    D_matrix doesn't record them all):
    * ``k_target`` / ``k_available_per_probe`` (sampling K mismatch).

    See plan §11 assumption 3: "#406 D_matrix is byte-identical-recipe with
    #488 for the 16×16 sub-grid".
    """
    schema = d.get("schema_version")
    if schema is None:
        raise AssertionError(
            "i406 D_matrix.json has no schema_version; refusing to lift cells "
            "without a recipe sentinel. Re-export #406 with provenance."
        )
    n_probes_inh = d.get("n_probes")
    if n_probes_inh != N_PROBES:
        raise AssertionError(
            f"i406 D_matrix n_probes={n_probes_inh!r} != #488 N_PROBES={N_PROBES}. "
            "The inherited 16×16 JS sub-grid was computed on a different probe "
            "count; paired-bootstrap with newly-computed 11×* cells would mix "
            "scales. Either re-run #406 with n_probes=50 or do NOT lift."
        )
    js_block_cids = set(d.get("JS", {}).keys())
    if js_block_cids != set(INHERITED_CIDS):
        missing = INHERITED_CIDS - js_block_cids
        extra = js_block_cids - INHERITED_CIDS
        raise AssertionError(
            f"i406 D_matrix JS block cids {sorted(js_block_cids)} != "
            f"INHERITED_CIDS {sorted(INHERITED_CIDS)} (missing={sorted(missing)}, "
            f"extra={sorted(extra)}). Condition recipe drift; refusing to lift."
        )
    # Soft warns
    for soft_key in ("k_target", "k_available_per_probe"):
        if soft_key not in d:
            logger.warning(
                "i406 D_matrix has no %s field; cannot verify K-sample match. "
                "Proceeding under plan §11 assumption 3.",
                soft_key,
            )
    logger.info(
        "i406 recipe-match check PASSED (schema=%s, n_probes=%d, 16 cids).",
        schema,
        n_probes_inh,
    )


def _seed_js_kl_from_i406(js_matrix: dict, kl_matrix: dict) -> tuple[dict, dict]:
    """Copy the inherited 16×16 JS sub-grid (and KL directions if available)
    into the output matrices.

    The #406 D_matrix.json carries a ``JS`` and a ``KL`` block keyed by cid.
    `JS[i][j]` is the symmetric JS scalar; for KL the inherited D_matrix only
    records one direction in some schema versions — we copy what's there and
    leave the orthogonal direction None for the analyzer to fill (or pull
    fresh in a follow-up pass). The new (NEW × *) and (* × NEW) cells are
    initialized to None.

    Before lifting, asserts the #406 recipe matches #488's (n_probes + cid
    set + schema). Raises if drift, so silent recipe mismatch can't
    contaminate the paired-bootstrap.
    """
    d = _load_inherited_d_matrix()
    _assert_i406_recipe_match(d)
    inherited = d.get("JS", {})
    kl_inh = d.get("KL", {})

    # Initialize 27×27 grids with None.
    cids = [c.cid for c in CONDITIONS]
    for ci in cids:
        js_matrix.setdefault(ci, {})
        kl_matrix.setdefault(ci, {})
        for cj in cids:
            js_matrix[ci].setdefault(cj, None)
            kl_matrix[ci].setdefault(cj, {"kl_ij": None, "kl_ji": None, "sym_kl": None})

    # Lift inherited 16×16 (only when both endpoints are inherited).
    lifted = 0
    for ci in INHERITED_CIDS:
        if ci not in inherited:
            continue
        for cj in INHERITED_CIDS:
            if cj not in inherited[ci]:
                continue
            v = inherited[ci][cj]
            if v is not None:
                js_matrix[ci][cj] = float(v)
                lifted += 1
        if ci in kl_inh:
            for cj in INHERITED_CIDS:
                v = kl_inh[ci].get(cj)
                if v is not None:
                    kl_matrix[ci][cj]["sym_kl"] = float(v)
    logger.info("Lifted %d inherited JS cells from #406 D_matrix.json", lifted)
    return js_matrix, kl_matrix


# ── Stylization score ────────────────────────────────────────────────────


def _baseline_prompt(tokenizer, q: str) -> str:
    """No-system-prompt default-assistant chat-template path (matches B1)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _compute_stylization_score(
    model,
    tokenizer,
    cond,
    probes: list[str],
    class_d_rewrites: dict | None,
    r_samples: int,
    max_new_tokens: int,
) -> float:
    """RB sequence-level JS of T_i vs the no-system baseline (default assistant).

    Reuses the SAME estimator as `_rb_block` with REFERENCE pinned to the
    no-system chat-template path. Per plan §4.3(b).
    """
    js_lens: list[float] = []
    for q in probes:
        prompt_i = build_prompt_for_condition(cond, q, tokenizer, class_d_rewrites)
        prompt_ref = _baseline_prompt(tokenizer, q)
        # Special case: if cond already IS the default assistant (B1/C1), JS = 0.
        if prompt_i == prompt_ref:
            js_lens.append(0.0)
            continue
        r_from_i = _sample_responses(model, tokenizer, prompt_i, r_samples, max_new_tokens)
        r_from_ref = _sample_responses(model, tokenizer, prompt_ref, r_samples, max_new_tokens)
        for ids in r_from_i:
            if len(ids) == 0:
                continue
            logp_i = _teacher_force_logprobs(model, tokenizer, prompt_i, ids)
            logp_r = _teacher_force_logprobs(model, tokenizer, prompt_ref, ids)
            js_lens.append(float(_per_position_js(logp_i, logp_r).mean().item()))
        for ids in r_from_ref:
            if len(ids) == 0:
                continue
            logp_i = _teacher_force_logprobs(model, tokenizer, prompt_i, ids)
            logp_r = _teacher_force_logprobs(model, tokenizer, prompt_ref, ids)
            js_lens.append(float(_per_position_js(logp_i, logp_r).mean().item()))
    return float(sum(js_lens) / len(js_lens)) if js_lens else float("nan")


# ── Cosine on residual-stream activations ─────────────────────────────────


def _last_token_residuals(
    model, tokenizer, prompt_text: str, layers: tuple[int, ...]
) -> dict[int, torch.Tensor]:
    """Read the residual stream at the LAST input token across `layers`.

    Returns ``{layer: (D,) tensor on CPU/float32}``.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=False).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_pos = inputs["input_ids"].shape[1] - 1
    # outputs.hidden_states is a tuple of length n_layers+1 (embeddings + per-layer outputs).
    # Layer indexing convention here: layer k = hidden_states[k] (so layer 0 = embedding,
    # layer 28 = final for a 28-layer Qwen-2.5-7B).
    out: dict[int, torch.Tensor] = {}
    for L in layers:
        if len(outputs.hidden_states) <= L:
            raise IndexError(
                f"Requested layer {L} but model has only {len(outputs.hidden_states)} "
                "hidden-state outputs (including embeddings at index 0)."
            )
        out[L] = outputs.hidden_states[L][0, last_pos, :].float().cpu()
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D vectors; numerically safe."""
    eps = 1e-8
    na = a.norm().clamp_min(eps)
    nb = b.norm().clamp_min(eps)
    return float((a @ b / (na * nb)).item())


# ── Driver ───────────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - CLI dispatch loop
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["js", "kl", "cossim", "stylization"],
        help="Sub-passes to skip (smoke / debug).",
    )
    ap.add_argument(
        "--r-samples",
        type=int,
        default=R_SAMPLES,
        help="Per-probe response samples (default 8 per metrics rule).",
    )
    ap.add_argument(
        "--n-probes",
        type=int,
        default=N_PROBES,
        help="Number of Q'_probe questions (default 50 = full set).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Per-sample response token cap for the RB estimator (default 256).",
    )
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help=(
            "Optional whitelist of (i,j) cells like `A1:G2 G2:A1`. Mostly for "
            "smoke-test of the JS path on a tiny slice."
        ),
    )
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Probes: subset Q_test_extended_50.
    probes_all = load_q_test_extended_50()
    probes = probes_all[: args.n_probes]
    class_d_rewrites = load_class_d_rewrites()

    # ── Persist is_stylized_source up front (independent of model loads) ──
    is_stylized_path = OUT_DIR / "is_stylized_source.json"
    is_stylized = {c.cid: int(c.cid in STRONG_STYLIZED_SOURCES) for c in CONDITIONS}
    _atomic_write_json(
        is_stylized_path,
        {
            "schema_version": "i488_v1",
            "strong_stylized_sources": sorted(STRONG_STYLIZED_SOURCES),
            "is_stylized_source": is_stylized,
        },
    )
    logger.info("Wrote %s", is_stylized_path)

    # Load tokenizer always; load model lazily when any GPU pass needs it.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    needs_model = bool({"js", "kl", "cossim", "stylization"} - set(args.skip))
    model = None
    if needs_model:
        logger.info("Loading model %s on GPU 0 (CVD=%d)", BASE_MODEL, args.gpu_id)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )
        model.eval()

    cids = [c.cid for c in CONDITIONS]

    # ── JS + KL pass ──
    js_path = OUT_DIR / "js_matrix.json"
    kl_path = OUT_DIR / "kl_matrix.json"
    if "js" not in args.skip:
        js_matrix: dict[str, dict[str, float | None]] = (
            json.loads(js_path.read_text())["JS"]
            if js_path.exists() and js_path.stat().st_size > 0
            else {}
        )
        kl_matrix: dict[str, dict[str, dict]] = (
            json.loads(kl_path.read_text())["KL"]
            if kl_path.exists() and kl_path.stat().st_size > 0
            else {}
        )
        js_matrix, kl_matrix = _seed_js_kl_from_i406(js_matrix, kl_matrix)

        # Identify cells still missing — anything where i OR j is a NEW cid,
        # plus inherited × inherited cells that didn't lift (e.g. #406 D_matrix
        # left some pairs None).
        pending: list[tuple[str, str]] = []
        if args.pairs:
            for pair in args.pairs:
                i, j = pair.split(":")
                pending.append((i, j))
        else:
            for ci in cids:
                for cj in cids:
                    if ci == cj:
                        continue
                    if js_matrix[ci][cj] is None:
                        pending.append((ci, cj))
        logger.info("JS/KL: %d pending pairs (out of 27×26 = 702)", len(pending))

        for idx, (ci, cj) in enumerate(pending):
            cell = _rb_block(
                model,
                tokenizer,
                CONDITIONS_BY_ID[ci],
                CONDITIONS_BY_ID[cj],
                probes,
                class_d_rewrites,
                args.r_samples,
                args.max_new_tokens,
            )
            js_matrix[ci][cj] = cell["js"]
            kl_matrix[ci][cj] = {
                "kl_ij": cell["kl_ij"],
                "kl_ji": cell["kl_ji"],
                "sym_kl": cell["sym_kl"],
            }
            if (idx + 1) % 20 == 0 or idx == len(pending) - 1:
                _atomic_write_json(
                    js_path,
                    {
                        "schema_version": "i488_v1",
                        "conditions": cids,
                        "JS": js_matrix,
                        "n_probes": args.n_probes,
                        "r_samples": args.r_samples,
                    },
                )
                _atomic_write_json(
                    kl_path,
                    {
                        "schema_version": "i488_v1",
                        "conditions": cids,
                        "KL": kl_matrix,
                        "n_probes": args.n_probes,
                        "r_samples": args.r_samples,
                    },
                )
                logger.info(
                    "JS/KL progress: %d/%d cells; last (%s,%s) js=%.4f",
                    idx + 1,
                    len(pending),
                    ci,
                    cj,
                    cell["js"],
                )

    # ── stylization_score pass ──
    if "stylization" not in args.skip:
        style_path = OUT_DIR / "stylization_score.json"
        existing = (
            json.loads(style_path.read_text()).get("stylization_score", {})
            if style_path.exists() and style_path.stat().st_size > 0
            else {}
        )
        scores: dict[str, float] = dict(existing)
        for c in CONDITIONS:
            if c.cid in scores:
                continue
            scores[c.cid] = _compute_stylization_score(
                model,
                tokenizer,
                c,
                probes,
                class_d_rewrites,
                args.r_samples,
                args.max_new_tokens,
            )
            _atomic_write_json(
                style_path,
                {
                    "schema_version": "i488_v1",
                    "conditions": cids,
                    "stylization_score": scores,
                    "baseline": "no-system-prompt default-assistant chat-template",
                    "n_probes": args.n_probes,
                    "r_samples": args.r_samples,
                },
            )
            logger.info("stylization_score[%s] = %.4f", c.cid, scores[c.cid])

    # ── cosine layer sweep ──
    if "cossim" not in args.skip:
        # One residual-vector per source (mean over probes), then cosine matrix per layer.
        residuals_by_layer_by_cid: dict[int, dict[str, torch.Tensor]] = {
            L: {} for L in COSINE_LAYERS
        }
        for c in CONDITIONS:
            per_probe_residuals: dict[int, list[torch.Tensor]] = {L: [] for L in COSINE_LAYERS}
            for q in probes:
                prompt = build_prompt_for_condition(c, q, tokenizer, class_d_rewrites)
                res = _last_token_residuals(model, tokenizer, prompt, COSINE_LAYERS)
                for L in COSINE_LAYERS:
                    per_probe_residuals[L].append(res[L])
            for L in COSINE_LAYERS:
                residuals_by_layer_by_cid[L][c.cid] = torch.stack(per_probe_residuals[L]).mean(
                    dim=0
                )
            logger.info("cosine: residuals for %s collected", c.cid)

        for L in COSINE_LAYERS:
            mat: dict[str, dict[str, float]] = {}
            for ci in cids:
                mat[ci] = {}
                for cj in cids:
                    mat[ci][cj] = _cosine(
                        residuals_by_layer_by_cid[L][ci],
                        residuals_by_layer_by_cid[L][cj],
                    )
            _atomic_write_json(
                OUT_DIR / f"cossim_matrix_layer{L}.json",
                {
                    "schema_version": "i488_v1",
                    "conditions": cids,
                    "layer": L,
                    "recipe": "persona-vectors (a) last-input-token; mean over probes",
                    "n_probes": args.n_probes,
                    "cossim": mat,
                },
            )
            logger.info("Wrote cossim layer %d matrix", L)

    if model is not None:
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Phase 1 done. Outputs in %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
