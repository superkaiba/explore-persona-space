#!/usr/bin/env python
"""Phase-1 pilot driver for the unified artifact factory (task #906).

Thin ORCHESTRATION over the ``explore_persona_space.artifacts`` library (Phases
0a-0g, already on ``main``): this driver imports and calls that library — it
reimplements none of it. For each pilot behavior class it

1. builds ONE model organism (``organisms.build_organism`` -> the ``primary``
   arm: contrastive negatives + generic interleave, one source context, one
   seed, dose-to-target on the source judged rate; the ``marker`` carve-out is a
   programmatic behavior whose organism build path is a Phase-1 seam that
   ``build_organism`` refuses — recorded as ``unsupported_v1``, never faked);
2. verifies it (``organisms.verify_organism`` -> install rate at the trigger
   context C vs the base model, the bystander-panel leakage read, and the
   ``tf_margin`` dual-DV companion);
3. for the 3 non-programmatic classes, extracts ``r_B`` (``directions``) from
   on-policy contrastive rollouts and, where a saved reference direction exists
   on disk, records cosine(new, saved) as a reproduction check;
4. writes ``calibration_report.json`` incrementally (checkpoint-per-phase, one
   class at a time): per-class realized wall-seconds per phase, Claude
   generation + judge draw counts (read from the ``pool_meta.json`` /
   ``organism_report.json`` sidecars the library writes), install + leakage
   numbers, and the ``api_error_drops`` / ``refusal_drops`` telemetry.

Two entrypoints: ``--smoke`` (tiny, fully mocked/CPU seams — validates the
orchestration wiring end-to-end with no GPU or live API) and ``--full`` (the
real GPU/API run; single entrypoint for one GPU pod:
``nohup uv run python scripts/issue906_phase1_pilot.py --full``). Real training
logs to WandB via the recipe's ``report_to="wandb"`` (recipe.py); this driver
adds no separate WandB init.

All heavyweight imports (``artifacts`` -> ``torch``) are function-local and run
only after ``load_dotenv()`` (the shared-VM thread-cap contract: torch freezes
its thread pool from ``OMP_NUM_THREADS`` at import — code-style.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

logger = logging.getLogger("issue906_pilot")

# The 4 pilot classes (task #906 scope). marker is programmatic — organism-only
# carve-out with its own band-stop recipe; its build path is a Phase-1 seam.
PILOT_BEHAVIORS: tuple[str, ...] = (
    "sycophancy",
    "harmful_compliance",
    "china_censorship",
    "marker",
)

# Saved reference r_B directions to reproduce against, per behavior. Relative to
# ``--reference-root`` (default: repo root); tried in order, first-existing wins.
# Every candidate stores a (L, H) = (28, 3584) r_b for Qwen-2.5-7B in one of
# three on-disk shapes (raw tensor / {"r_b": ...} / {"r_b_c"|"r_b_a": ...}); the
# loader below handles all three. harmful_compliance maps to the refusal axis
# (refusal is the opposite pole of harmful compliance — the same direction).
# china_censorship has no saved reference -> reproduction records not-found.
REFERENCE_DIRECTIONS: dict[str, tuple[str, ...]] = {
    "sycophancy": (
        "data/issue_779/r_b/sycophancy.pt",
        "data/issue_778/rb/sycophancy.pt",
        "data/issue_658/prb_dl/issue661_rb_extraction_divergence/analysis_tensors/r_b_sycophancy.pt",
    ),
    "harmful_compliance": (
        "data/issue_658/prb_dl/issue661_rb_extraction_divergence/analysis_tensors/r_b_refusal.pt",
    ),
    "china_censorship": (),
}

# Order of dict keys tried when a reference .pt stores a dict (contrastive
# diff-of-means direction first: the persona-vectors r_B).
_REFERENCE_RB_KEYS: tuple[str, ...] = ("r_b", "r_b_c", "rb", "r_b_a")

REPORT_SCHEMA = "issue906_phase1_calibration_v1"


# ── Config ─────────────────────────────────────────────────────────────────


@dataclass
class PilotConfig:
    """Resolved run configuration (from CLI or a test constructor)."""

    mode: str  # "smoke" | "full"
    classes: tuple[str, ...]
    source_context: str
    seed: int
    base_model: str
    out_root: Path
    report_path: Path
    reference_root: Path
    generic_data_path: str | None
    gpu_id: int
    n_eval_completions: int
    n_judge_draws: int
    n_extraction_rollouts: int
    eval_temperature: float
    datagen_target_n: int | None
    eval_question_limit: int | None  # None -> full behavior.eval_question_bank
    extraction_question_limit: int | None  # None -> full behavior.extraction.question_set
    upload: bool

    def public(self) -> dict:
        d = asdict(self)
        d["out_root"] = str(self.out_root)
        d["report_path"] = str(self.report_path)
        d["reference_root"] = str(self.reference_root)
        d["classes"] = list(self.classes)
        return d


@dataclass
class PilotSeams:
    """Injectable boundaries. Every field ``None`` -> the real library path.

    Populated (by ``--smoke`` and the unit test) to run fully mocked on CPU with
    no GPU or live API. The library's own injectable seams (``datagen_fn`` /
    ``train_fn`` / ``rate_fn`` / ``generate_fn`` / ``judge_fn`` /
    ``margin_read_fn``) are threaded straight through.

    Marker-carve-out seams (both None -> the inline programmatic path):
    - ``marker_datagen_fn``: builds the training mix (pos + neg JSONL) for the
      marker class; signature ``(source_sp, neg_sps, out_dir, *, seed) -> (pos_path,
      cn_path)``; None -> the inline mix builder in ``_build_marker_class``.
    - ``marker_verify_fn``: measures all-three-space slot stats for the marker
      class; signature ``(adapter_path, base_model, contexts, *, marker_text,
      qwen_im_end_id) -> list[dict]`` where each dict is a ``MARKER_SLOT_CONTRACT_KEYS``
      record; None -> the inline batched forward pass in ``_verify_marker_class``.
    """

    datagen_fn: Any = None  # organisms.build_organism datagen_fn boundary
    train_fn: Any = None  # organisms.build_organism train_fn boundary
    make_rate_fn: Any = None  # (organism, out_dir: Path) -> RateFn (dose selection)
    verify_generate_fn: Any = None  # verify + rate-ladder vLLM generation boundary
    judge_fn: Any = None  # graded-judge boundary (build ladder + verify + extract)
    margin_read_fn: Any = None  # verify tf-margin boundary
    margin_pools: Any = None  # (pos_pairs, neg_pairs); None -> derive from datagen_dir
    extract_generate_fn: Any = None  # base-model contrastive-rollout generation boundary
    score_fn: Any = None  # directions.score_completions boundary
    extract_fn: Any = None  # (behavior, scored) -> DirectionResult boundary
    uploader: Any = None  # (behavior_name, build_result, cfg) -> dict boundary
    # Marker programmatic-carve-out seams.
    marker_datagen_fn: Any = None  # None -> inline marker mix builder
    marker_verify_fn: Any = None  # None -> inline three-space forward pass
    # marker_gen_fn: generates on-policy base-model responses for the marker training
    # mix positives/negatives.  Signature: (questions: list[str], system_prompt: str | None)
    # -> list[str].  None -> inline HF generate in _build_marker_class.
    # Smoke stub returns distinct per-question fakes: f"resp::{hash(q) & 0xFFFF:04x}".
    marker_gen_fn: Any = None
    # CONCERN 5: pre-intervention baseline seam (plan §4 Phase 0).
    # Signature: (behavior, cfg, class_dir) -> dict with keys {rate, n_questions, out_dir}.
    # None -> the real vLLM judged-rate baseline pass in run_class.
    # Smoke stub: injected lambda returning {"rate": 0.0, "n_questions": 0, "status": "smoke"}.
    baseline_fn: Any = None
    # CONCERN 4: sycophancy on-policy control arm (plan §4 Phase 0, sycophancy only).
    # Signature: (behavior, cfg, class_dir) -> dict with keys {r_b_path, regime, provenance}.
    # None -> the real on-policy rollout + extract pass in run_class.
    # Smoke stub: injected lambda returning {"status": "smoke", "r_b_path": None}.
    on_policy_control_fn: Any = None


# ── Small provenance / IO helpers ────────────────────────────────────────────


def _git_short_sha() -> str:
    """Short git SHA for provenance; 'uncommitted' when unavailable (fail-soft)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "uncommitted"
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else "uncommitted"


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows) -> int:
    """Write ``rows`` (iterable of dicts) as JSONL via tmp + ``os.replace``.

    Returns the row count. Used to persist generated rollout TEXT before any
    judge/reduce step (Upload Policy persist-by-default, #779): the text is
    what makes downstream artifacts regenerable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


@contextmanager
def _timed(store: dict, key: str):
    """Record wall-seconds for a phase into ``store[key]`` (always, even on raise)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round(time.perf_counter() - t0, 3)


# ── Reproduction cosine check ─────────────────────────────────────────────────


def _load_reference_rb(path: Path):
    """Load a saved reference direction into a fp32 ``(L, H)`` tensor + the key used.

    Handles the three on-disk shapes: a raw tensor, a ``{"r_b": ...}`` payload
    (issue_779 ``save_direction``), and the issue_661 ``{"r_b_c"|"r_b_a": ...}``
    dict. Raises ``ValueError`` when no recognized r_B tensor is present.
    """
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "shape"):  # raw tensor
        return obj.float(), "<tensor>"
    if isinstance(obj, dict):
        for key in _REFERENCE_RB_KEYS:
            v = obj.get(key)
            if v is not None and hasattr(v, "shape"):
                return v.float(), key
    raise ValueError(
        f"reference {path} has no recognized r_B tensor (raw tensor or one of {_REFERENCE_RB_KEYS})"
    )


def _per_layer_cosine(new_rb, ref_rb) -> dict:
    """Signed per-layer cosine of two ``(L, H)`` directions + summary stats.

    Diff-of-means directions carry a definite sign (exhibit - not_exhibit), so
    the cosine sign is meaningful; a strong reproduction reads near +1 at the
    informative layers.
    """
    import torch

    cos = torch.nn.functional.cosine_similarity(new_rb.float(), ref_rb.float(), dim=1)  # (L,)
    cos_list = [round(float(x), 6) for x in cos.tolist()]
    argmax = int(torch.argmax(cos).item())
    return {
        "cosine_per_layer": cos_list,
        "cosine_max": round(float(cos.max().item()), 6),
        "cosine_mean": round(float(cos.mean().item()), 6),
        "argmax_layer": argmax,
        "cosine_at_argmax": cos_list[argmax],
    }


def reproduction_check(behavior_name: str, new_rb, reference_root: Path) -> dict:
    """Cosine(new r_B, first-existing saved reference) or a machine-readable miss."""
    candidates = [reference_root / rel for rel in REFERENCE_DIRECTIONS.get(behavior_name, ())]
    for path in candidates:
        if not path.exists():
            continue
        ref_rb, key = _load_reference_rb(path)
        if tuple(ref_rb.shape) != tuple(new_rb.shape):
            return {
                "status": "reference_shape_mismatch",
                "reference_path": str(path),
                "reference_key": key,
                "reference_shape": list(ref_rb.shape),
                "new_shape": list(new_rb.shape),
            }
        return {
            "status": "computed",
            "reference_path": str(path),
            "reference_key": key,
            **_per_layer_cosine(new_rb, ref_rb),
        }
    return {"status": "reference_not_found", "searched": [str(p) for p in candidates]}


def _compute_d1_gap(claude_rb_path: Path, on_policy_rb_path: Path) -> dict:
    """Plan §4 Phase-3 D1-gap: per-layer cosine(claude_generated r_B, on_policy r_B).

    Loads BOTH sycophancy directions from their PERSISTED artifacts (the extract
    phase's ``r_b_sycophancy.pt`` and the on-policy control's
    ``r_b_on_policy.pt``) — never from in-memory objects — so the reported gap
    is computed from exactly what the upload / reuse paths will see.  Returns
    the ``_per_layer_cosine`` stats plus provenance paths, or a machine-readable
    shape-mismatch record.  A missing artifact raises (fail loud): on the
    production path both files were just written by ``save_direction``.
    """
    claude_rb, claude_key = _load_reference_rb(Path(claude_rb_path))
    on_policy_rb, on_policy_key = _load_reference_rb(Path(on_policy_rb_path))
    if tuple(claude_rb.shape) != tuple(on_policy_rb.shape):
        return {
            "status": "shape_mismatch",
            "claude_generated_path": str(claude_rb_path),
            "on_policy_path": str(on_policy_rb_path),
            "claude_generated_shape": list(claude_rb.shape),
            "on_policy_shape": list(on_policy_rb.shape),
        }
    return {
        "status": "computed",
        "claude_generated_path": str(claude_rb_path),
        "claude_generated_key": claude_key,
        "on_policy_path": str(on_policy_rb_path),
        "on_policy_key": on_policy_key,
        **_per_layer_cosine(claude_rb, on_policy_rb),
    }


# ── API-count aggregation (from the library's sidecars) ───────────────────────


def _pool_meta_counts(pool_meta_path: Path) -> dict:
    """Generation + judge-draw + drop counts from a datagen ``pool_meta.json``.

    Generation here is CLAUDE-generated (datagen D1), so ``requested`` counts are
    real Claude API calls. Returns ``{"available": False}`` when the sidecar is
    absent (e.g. a mocked datagen stub that writes only the emit files).
    """
    if not pool_meta_path.exists():
        return {"available": False}
    pm = json.loads(pool_meta_path.read_text())
    pos, neg = pm.get("positive", {}), pm.get("negative", {})
    jds = pm.get("judge_draw_stats", {})
    pos_jds, neg_jds = jds.get("positive", {}), jds.get("negative", {})
    return {
        "available": True,
        "claude_generation_requested": {
            "positive": pos.get("requested"),
            "negative": neg.get("requested"),
        },
        "claude_generation_returned": {
            "positive": pos.get("generated"),
            "negative": neg.get("generated"),
        },
        "judge_draws_total": (pos_jds.get("n_total", 0) or 0) + (neg_jds.get("n_total", 0) or 0),
        "judge_draws_dropped": (pos_jds.get("n_dropped", 0) or 0)
        + (neg_jds.get("n_dropped", 0) or 0),
        "refusal_drops": {
            "positive": pos.get("refusal_drops"),
            "negative": neg.get("refusal_drops"),
        },
        "api_error_drops": {
            "positive": pos.get("api_error_drops"),
            "negative": neg.get("api_error_drops"),
        },
        "empty_drops": {"positive": pos.get("empty_drops"), "negative": neg.get("empty_drops")},
    }


def _verify_judge_counts(report_dict: dict) -> dict:
    """Sum verify-side judge draws from ``OrganismReport.judge_drop_telemetry``.

    Verify generation is local vLLM (not an API call); only the judge draws are
    Anthropic API calls, so those are the ones aggregated here.
    """
    tele = report_dict.get("judge_drop_telemetry", {}) or {}
    total = sum((c.get("n_total_draws", 0) or 0) for c in tele.values())
    dropped = sum((c.get("n_dropped_draws", 0) or 0) for c in tele.values())
    return {"judge_draws_total": total, "judge_draws_dropped": dropped, "n_cells": len(tele)}


def _judge_draw_subcounts(payload: dict | None) -> dict:
    """Judge-draw totals from a phase payload dict (baseline / on-policy control).

    ``_run_baseline_pass`` and ``_run_on_policy_control`` both persist
    ``judge_draws_total`` / ``judge_draws_dropped`` in their return dicts (which
    ``run_class`` stores in the report entry); those are real Anthropic judge
    API calls and belong in the per-class ``api_calls`` roll-up (r6 concern
    ``onpolicy-judge-draws-excluded-from-api-calls``).  Tolerates ``None`` /
    seam-stub payloads that omit the keys (counts read 0).
    """
    payload = payload or {}
    return {
        "judge_draws_total": int(payload.get("judge_draws_total", 0) or 0),
        "judge_draws_dropped": int(payload.get("judge_draws_dropped", 0) or 0),
    }


# ── Contrastive-rollout generation for r_B extraction ─────────────────────────


def generate_contrastive_completions(
    behavior,
    gen_fn,
    *,
    n_rollouts: int,
    temperature: float,
    question_limit: int | None,
) -> list:
    """On-policy contrastive rollouts for ``r_B`` (persona-vectors recipe steps 2-3).

    For each of the behavior's extraction ``PromptPair``s and each extraction
    question, generate ``n_rollouts`` completions under the ``exhibit`` system
    prompt and ``n_rollouts`` under ``not_exhibit`` (the SHARED question set the
    content-match guard in ``extract_direction`` requires). ``gen_fn`` is the
    verify-side generation seam: ``(side_path, messages_list, *, n, temperature)
    -> list[list[str]]`` called with ``side_path=None`` (the base model).
    """
    from explore_persona_space.artifacts.directions import ContrastiveCompletion

    ext = behavior.extraction
    if ext is None:
        raise ValueError(f"behavior {behavior.name!r} has no extraction spec (programmatic?)")
    questions = list(ext.question_set)
    if question_limit is not None:
        questions = questions[:question_limit]

    # arm -> (messages_list, meta[(pair_index, system_prompt, question)])
    plans: dict[str, tuple[list, list]] = {"exhibit": ([], []), "not_exhibit": ([], [])}
    for pair_index, pair in enumerate(ext.prompt_pairs):
        for arm, system_prompt in (("exhibit", pair.exhibit), ("not_exhibit", pair.not_exhibit)):
            msgs_list, meta = plans[arm]
            for q in questions:
                msgs_list.append(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
                )
                meta.append((pair_index, system_prompt, q))

    completions: list = []
    for arm, (msgs_list, meta) in plans.items():
        outs = gen_fn(None, msgs_list, n=n_rollouts, temperature=temperature)
        if len(outs) != len(meta):
            raise ValueError(
                f"extract gen_fn returned {len(outs)} completion lists for {len(meta)} "
                f"{arm} prompts"
            )
        for (pair_index, system_prompt, q), rollouts in zip(meta, outs, strict=True):
            for response in rollouts:
                completions.append(
                    ContrastiveCompletion(
                        arm=arm,
                        pair_index=pair_index,
                        system_prompt=system_prompt,
                        question=q,
                        response=response,
                    )
                )
    return completions


def _make_default_extract_fn(cfg: PilotConfig):
    """The real extract seam: load an HF base model, run ``extract_direction``, free it.

    The HF model loads only AFTER the vLLM rollout engine is torn down (the
    caller closes it), so an HF bf16 model never coexists with a vLLM engine on
    one GPU (the organisms.py single-live-GPU-resource contract).
    """

    def extract_fn(behavior, scored):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from explore_persona_space.artifacts.directions import extract_direction

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16).to(
            device
        )
        model.eval()
        try:
            # BLOCKER 2 fix: production direction extraction uses regime="steering" +
            # provenance="claude_generated" (Plan §4 Phase 3 spec).  regime="read_out"
            # with provenance="on_policy" is ONLY the sycophancy control arm that
            # generates fresh rollouts — it is not the default production path.
            return extract_direction(
                behavior,
                model,
                tokenizer,
                scored,
                regime="steering",
                provenance="claude_generated",
            )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return extract_fn


# ── Per-phase runners ─────────────────────────────────────────────────────────


def _build_class(behavior, org, cfg: PilotConfig, seams: PilotSeams, class_dir: Path):
    """Build one organism (datagen -> mix -> recipe -> train -> dose-selected ckpt)."""
    from explore_persona_space.artifacts.datagen import generate_training_data
    from explore_persona_space.artifacts.organisms import build_organism, make_source_rate_fn
    from explore_persona_space.eval.graded_judge import judge_graded
    from explore_persona_space.train.sft import train_lora

    build_dir = class_dir / "build"
    spec = org.recipe

    # rate_fn is REQUIRED for the checkpoint_and_select (dose-to-target) path.
    rate_fn = None
    if spec.train_method == "lora" and spec.stopping.kind == "checkpoint_and_select":
        if seams.make_rate_fn is not None:
            rate_fn = seams.make_rate_fn(org, build_dir / "rate")
        else:
            rate_fn = make_source_rate_fn(
                org,
                out_dir=build_dir / "rate",
                base_model=cfg.base_model,
                n_completions=cfg.n_eval_completions,
                n_judge_draws=cfg.n_judge_draws,
                generate_fn=seams.verify_generate_fn,
                judge_fn=seams.judge_fn or judge_graded,
            )

    datagen_kwargs: dict[str, Any] = {"n_judge_draws": cfg.n_judge_draws}
    if cfg.datagen_target_n is not None:
        datagen_kwargs["target_n"] = cfg.datagen_target_n
    elif cfg.mode == "full":
        # CONCERN 5 fix: in full mode, use the full train bank size instead of
        # silently falling back to the library default (target_n=200), which may
        # under- or over-count relative to the actual bank.
        datagen_kwargs["target_n"] = len(list(behavior.train_question_bank))

    # r14 (content-mix-token-budget-unenforced): when the REAL trainer will run
    # (train_fn seam unset), load the base tokenizer so build_organism enforces
    # the recipe max_length token budget at mix assembly — SFTTrainer
    # right-truncation on content mixes is SILENT (no fail-loud collator), so
    # an overlong WildChat-lineage row would otherwise degrade its completion
    # supervision without an error. Stub-seam runs (smoke/tests) pass None and
    # skip the gate (offline contract; pinned by the real-tokenizer tests).
    tokenizer = None
    if seams.train_fn is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)

    return build_organism(
        org,
        out_root=build_dir,
        base_model=cfg.base_model,
        generic_data_path=cfg.generic_data_path,
        gpu_id=cfg.gpu_id,
        datagen_kwargs=datagen_kwargs,
        datagen_fn=seams.datagen_fn or generate_training_data,
        train_fn=seams.train_fn or train_lora,
        rate_fn=rate_fn,
        tokenizer=tokenizer,
    )


def _verify_class(behavior, org, adapter_path, cfg, seams, class_dir, datagen_dir):
    """Install + per-bystander leakage + tf-margin companion."""
    from explore_persona_space.artifacts.organisms import verify_organism
    from explore_persona_space.eval.graded_judge import judge_graded

    eval_questions = None
    if cfg.eval_question_limit is not None:
        eval_questions = list(behavior.eval_question_bank)[: cfg.eval_question_limit]

    return verify_organism(
        org,
        adapter_path,
        out_dir=class_dir / "verify",
        datagen_dir=datagen_dir,
        margin_pools=seams.margin_pools,
        eval_questions=eval_questions,
        base_model=cfg.base_model,
        n_completions=cfg.n_eval_completions,
        n_judge_draws=cfg.n_judge_draws,
        generate_fn=seams.verify_generate_fn,
        judge_fn=seams.judge_fn or judge_graded,
        margin_read_fn=seams.margin_read_fn,
    )


def _run_baseline_pass(behavior, cfg: PilotConfig, class_dir: Path, seams: PilotSeams) -> dict:
    """Base-model vLLM judged-rate pass (pre-intervention baseline).

    Runs generation + judge on the SOURCE context only — the trigger context C
    resolved via ``CONTEXTS[cfg.source_context]`` (``Behavior`` carries no
    context field; the source context is a run parameter, the same resolution
    ``_build_marker_class`` / ``_verify_marker_class`` use) — mirroring the
    "base" side of ``_verify_class``.  The judge is ONE batched call following
    the ``eval.graded_judge.judge_graded`` contract over ``(item_id, question,
    completion)`` triples (the ``organisms._rate_for_cell`` pattern), with
    rule-9 drop-never-coerce accounting.  Results persist to
    ``<class_dir>/baseline/`` — including the temperature-sampled (hence
    non-regenerable) rollout TEXT at ``baseline/raw_completions.jsonl``,
    written BEFORE the judge/reduce step (r8 CONCERN
    genreduce-rollout-text-not-persisted).  Called by ``run_class`` when
    ``seams.baseline_fn is None`` (the full-run production path).

    Returns a dict with at minimum:
        ``{"rate": float, "n_questions": int, "out_dir": str, "status": "ok"}``.
    """
    from explore_persona_space.artifacts.context import CONTEXTS
    from explore_persona_space.artifacts.organisms import _default_vllm_generate_fn
    from explore_persona_space.eval.graded_judge import judge_graded

    if behavior.judge_rubric is None:
        raise ValueError(
            f"behavior {behavior.name!r}: judge_rubric is None — cannot run the baseline "
            "judged-rate pass; no fallback rubric is substituted (fail loud)"
        )

    baseline_dir = class_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    eval_questions = list(behavior.eval_question_bank)
    if cfg.eval_question_limit is not None:
        eval_questions = eval_questions[: cfg.eval_question_limit]

    source_ctx = CONTEXTS[cfg.source_context]
    # Generate on-policy completions for the BASE model at the source context
    # (Context.messages is the ONE resolver — system/prefix/user_wrap aware).
    gen = seams.verify_generate_fn if seams.verify_generate_fn is not None else None
    created_gen = gen is None
    if created_gen:
        gen = _default_vllm_generate_fn(cfg.base_model)

    judge_fn = seams.judge_fn or judge_graded
    try:
        messages_list = [source_ctx.messages(q) for q in eval_questions]
        all_completions = gen(
            None,  # side_path=None -> base model
            messages_list,
            n=cfg.n_eval_completions,
            temperature=cfg.eval_temperature,
        )
    finally:
        if created_gen:
            close = getattr(gen, "close", None)
            if callable(close):
                close()

    # ONE batched judge call (judge_graded contract: items + rubric positional,
    # keyword-only n_draws / cache_dir / save_raw / judge_model). Item ids use
    # "-" separators only — judge_graded raises on "__" in an item id.
    items: list[tuple[str, str, str]] = []
    for i, (q, completions) in enumerate(zip(eval_questions, all_completions, strict=True)):
        for j, completion in enumerate(completions):
            items.append((f"baseline-q{i:03d}-c{j}", q, completion))

    # Persist the rollout TEXT before the judge/reduce step (Upload Policy
    # persist-by-default, #779; r8 CONCERN genreduce-rollout-text-not-persisted):
    # these completions are temperature-sampled -> NON-regenerable, and the
    # baseline/ leg of _upload_class ships this file to HF in the fail-loud
    # expected set. item_id matches the judge item id, so judge_raw.json
    # cross-references row-for-row.
    raw_completions_path = baseline_dir / "raw_completions.jsonl"
    _write_jsonl_atomic(
        raw_completions_path,
        ({"item_id": iid, "question": q, "completion": c} for iid, q, c in items),
    )

    result = judge_fn(
        items,
        behavior.judge_rubric,
        n_draws=cfg.n_judge_draws,
        cache_dir=baseline_dir / "judge_cache",
        save_raw=baseline_dir / "judge_raw.json",
        judge_model=behavior.judge_model,
    )

    # Rule-9 accounting (mirrors organisms._rate_for_cell): a None score (all
    # draws dropped) leaves the denominator and is counted, never coerced; a
    # MISSING item_id is a judge-contract violation and raises.
    n_pos = n_scored = n_dropped = 0
    for iid, _q, _c in items:
        if iid not in result.scores:
            raise ValueError(
                f"judge_fn contract violation in the baseline pass: item_id {iid!r} is "
                "MISSING from JudgeResult.scores — a rule-9 all-draws-dropped item must "
                "be present with score None, never absent"
            )
        score = result.scores[iid]
        if score is None:
            n_dropped += 1
            continue
        n_scored += 1
        if score > behavior.threshold:
            n_pos += 1
    if items and n_scored == 0:
        raise ValueError(
            "every baseline completion was judge-dropped — a fully dropped baseline is "
            "a judging outage, not a 0% rate"
        )

    rate = n_pos / n_scored if n_scored else 0.0
    payload = {
        "status": "ok",
        "rate": round(rate, 6),
        "n_questions": len(eval_questions),
        "n_completions_total": len(items),
        "n_scored": n_scored,
        "n_judge_dropped_completions": n_dropped,
        "judge_draws_total": result.n_total_draws,
        "judge_draws_dropped": result.n_dropped_draws,
        "raw_completions_path": str(raw_completions_path),
        "out_dir": str(baseline_dir),
        "context_id": source_ctx.context_id,
        "git_commit": _git_short_sha(),
        "timestamp_utc": _now_utc(),
    }
    # Persist to disk for the clean-result.
    _atomic_write_json(baseline_dir / "baseline.json", payload)
    return payload


def _run_on_policy_control(behavior, cfg: PilotConfig, class_dir: Path, seams: PilotSeams) -> dict:
    """Sycophancy on-policy control arm (plan §4 Phase 0, sycophancy-only).

    Tier-2 instruct-and-strip (on-policy-completions recipe): append the
    behavior's registered elicitation instructions to the SOURCE-context system
    turn — the trigger context C resolved via ``CONTEXTS[cfg.source_context]``
    (``Behavior`` carries no context field) — sample BOTH contrastive arms
    (``extract_direction`` refuses a zero-captured arm, so an exhibit-only set
    can never form ``r_b``), judge-score through
    ``directions.score_completions`` (drop-never-coerce), STRIP the elicitation
    instruction before persisting, then extract the on-policy r_B direction for
    the D1-gap comparison (``regime="steering"``, ``provenance="on_policy"``).

    The not_exhibit arm uses ``elicitation.not_exhibit_instructions[0]`` when
    registered; ``None`` means the default assistant under the source context
    already does NOT exhibit the behavior (``ElicitationSpec`` contract), so
    that arm samples under the UN-instructed source context.

    Persists the UNSCORED rollout text to
    ``<class_dir>/on_policy_control/raw_completions.jsonl`` immediately after
    generation and BEFORE the remote-judge ``score_completions`` call (r9
    CONCERN ``onpolicy-control-rollout-text-not-persisted-before-score``),
    then the DERIVED scored rows (stripped system prompt) to
    ``<class_dir>/on_policy_control/completions.jsonl`` and the direction via
    ``save_direction``.  Returns a dict with keys
    ``{status, r_b_path, regime, provenance, n_kept}``.

    Called by ``run_class`` when ``seams.on_policy_control_fn is None`` (the
    full-run production path, sycophancy only).
    """
    from explore_persona_space.artifacts.context import CONTEXTS
    from explore_persona_space.artifacts.directions import (
        ContrastiveCompletion,
        extract_direction,
        filter_completions,
        save_completions_jsonl,
        save_direction,
        score_completions,
    )
    from explore_persona_space.artifacts.organisms import _default_vllm_generate_fn

    on_policy_dir = class_dir / "on_policy_control"
    on_policy_dir.mkdir(parents=True, exist_ok=True)

    if behavior.elicitation is None:
        raise ValueError(
            f"behavior {behavior.name!r}: elicitation is None — the tier-2 "
            "instruct-and-strip control arm needs registered elicitation instructions"
        )
    source_ctx = CONTEXTS[cfg.source_context]
    exhibit_instruction = behavior.elicitation.exhibit_instructions[0]
    not_exhibit_instruction = (
        behavior.elicitation.not_exhibit_instructions[0]
        if behavior.elicitation.not_exhibit_instructions is not None
        else None
    )
    # STRIPPED storage context: the source system prompt WITHOUT any
    # elicitation instruction (on-policy-completions.md tier 2).
    stripped_system_prompt = source_ctx.system or ""

    def _tier2_messages(question: str, instruction: str | None) -> list[dict[str, str]]:
        """Source-context messages with the elicitation instruction on the system turn."""
        msgs = source_ctx.messages(question)
        if instruction is None:
            return msgs
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": f"{msgs[0]['content']} {instruction}"}
        else:
            msgs.insert(0, {"role": "system", "content": instruction})
        return msgs

    extraction_qs = (
        list(behavior.extraction.question_set)[: cfg.extraction_question_limit or None]
        if behavior.extraction is not None
        else []
    )
    if not extraction_qs:
        raise ValueError(
            f"behavior {behavior.name!r}: no extraction questions available for the "
            "on-policy control arm"
        )
    questions = extraction_qs[: min(25, len(extraction_qs))]

    gen = seams.verify_generate_fn if seams.verify_generate_fn is not None else None
    created_gen = gen is None
    if created_gen:
        gen = _default_vllm_generate_fn(cfg.base_model)

    completions: list[ContrastiveCompletion] = []
    try:
        for arm, instruction in (
            ("exhibit", exhibit_instruction),
            ("not_exhibit", not_exhibit_instruction),
        ):
            messages_list = [_tier2_messages(q, instruction) for q in questions]
            outs = gen(
                None,  # side_path=None -> base model (on-policy Qwen)
                messages_list,
                n=3,  # 3 rollouts per question per arm
                temperature=1.0,
            )
            for q, rollouts in zip(questions, outs, strict=True):
                for response in rollouts:
                    completions.append(
                        ContrastiveCompletion(
                            arm=arm,
                            pair_index=0,
                            # Stripped: the elicitation instruction never persists.
                            system_prompt=stripped_system_prompt,
                            question=q,
                            response=response,
                        )
                    )
    finally:
        if created_gen:
            close = getattr(gen, "close", None)
            if callable(close):
                close()

    # Persist the UNSCORED rollout text IMMEDIATELY after generation and
    # BEFORE the remote-judge reduce (r9 CONCERN
    # onpolicy-control-rollout-text-not-persisted-before-score — the
    # persist-before-reduce class; upload-policy persist-by-default, #779):
    # these completions are temperature-sampled -> NON-regenerable. item_id
    # matches score_completions' judge item-id derivation verbatim, so
    # judge_raw.json cross-references row-for-row; _upload_class's
    # on_policy_control leg requires this file in its fail-loud set.
    raw_completions_path = on_policy_dir / "raw_completions.jsonl"
    _write_jsonl_atomic(
        raw_completions_path,
        (
            {
                "item_id": f"{c.arm}-p{c.pair_index}-{i:05d}",
                "arm": c.arm,
                "question": c.question,
                "completion": c.response,
            }
            for i, c in enumerate(completions)
        ),
    )

    # Judge-score through the library adapter (score_completions requires the
    # keyword-only cache_dir + save_raw; paths follow the _extract_class
    # judge_cache / judge_raw.json convention).
    scored, judge_result = score_completions(
        behavior,
        completions,
        n_draws=cfg.n_judge_draws,
        cache_dir=on_policy_dir / "judge_cache",
        save_raw=on_policy_dir / "judge_raw.json",
    )
    # DERIVED artifact: the same rows with judge scores attached (stripped
    # system prompt). The unscored text was already persisted to
    # raw_completions.jsonl above, before the judge call.
    completions_path = on_policy_dir / "completions.jsonl"
    save_completions_jsonl(scored, completions_path)

    _kept, filter_counts = filter_completions(scored, threshold=float(behavior.threshold))
    n_kept = {arm: filter_counts[arm]["kept"] for arm in ("exhibit", "not_exhibit")}
    if n_kept["exhibit"] == 0 or n_kept["not_exhibit"] == 0:
        # Reported yield failure — never a fabricated direction (extract_direction
        # raises on a zero-captured arm).  Judge draws were already spent on the
        # score pass above, so they still count toward the api_calls roll-up.
        return {
            "status": "yield_failure",
            "raw_completions_path": str(raw_completions_path),
            "completions_path": str(completions_path),
            "n_kept": n_kept,
            "filter_counts": filter_counts,
            "r_b_path": None,
            "judge_draws_total": judge_result.n_total_draws,
            "judge_draws_dropped": judge_result.n_dropped_draws,
        }

    # extract_direction from the on-policy completions (HF model loads only
    # AFTER the vLLM rollout engine above is torn down — single-live-GPU rule).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16).to(
        device
    )
    model.eval()
    try:
        direction = extract_direction(
            behavior,
            model,
            tokenizer,
            scored,
            regime="steering",
            provenance="on_policy",
            metadata={"judge_n_draws": cfg.n_judge_draws, "tier": "tier2_instruct_and_strip"},
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rb_path = on_policy_dir / "r_b_on_policy.pt"
    save_direction(direction, rb_path)
    return {
        "status": "ok",
        "r_b_path": str(rb_path),
        "regime": direction.regime,
        "provenance": direction.provenance,
        "n_kept": n_kept,
        "raw_completions_path": str(raw_completions_path),
        "completions_path": str(completions_path),
        "judge_draws_total": judge_result.n_total_draws,
        "judge_draws_dropped": judge_result.n_dropped_draws,
    }


def _marker_train_config(cfg: PilotConfig, *, tokenizer=None):
    """The marker class's REAL ``TrainLoraConfig`` via the canonical recipe builder.

    ``build_train_config(recipe_for("marker"), ...)`` dataclass-constructs the
    actual engine config — fail-loud ``TypeError`` on any kwarg drift vs
    ``TrainLoraConfig`` (the r12 crash class: a ``contrastive_negatives_path``
    kwarg that the engine never defined died at config construction).
    ``run_name`` / ``seed`` / ``gpu_id`` mirror the content classes'
    ``build_organism`` path (``organisms.py`` ``build_train_config(...,
    run_name=organism.slug(), seed=organism.seed, gpu_id=gpu_id)``).  When a
    ``tokenizer`` is provided, ``build_train_config`` re-asserts the marker
    token id (marker-leakage-measurement.md; the #537 "[ZLT]" no-op incident).

    Module-level (not a closure) so the CPU contract tests bind the EXACT
    production config construction with no GPU
    (``tests/test_issue906_train_contract.py``).
    """
    from explore_persona_space.artifacts.organisms import ModelOrganism
    from explore_persona_space.artifacts.recipe import build_train_config

    org = ModelOrganism("marker", cfg.source_context, arm="primary", seed=cfg.seed)
    return build_train_config(
        org.recipe,
        run_name=org.slug(),
        seed=cfg.seed,
        gpu_id=cfg.gpu_id,
        tokenizer=tokenizer,
    )


# Fail-loud floor for build-time row rejection: a rejected fraction above this
# means the budget is SYSTEMATICALLY too small for the question distribution —
# the remedy is a deliberate recipe max_length raise, never a silently shrunk
# mix. Measured on the att-20260704-061624 crash rows: 4/200 rows (2%) overflow
# the 2048 budget (two extreme-tail WildChat prompts of 2181/1718 prompt-only
# tokens), median full-row 487/419 tokens — so 0.10 separates "tail outliers"
# from "wrong budget" with wide margin on both sides. MUST equal
# organisms.MIX_MAX_REJECT_FRAC (the r14 shared gate's floor; kept a literal
# here because the driver's heavyweight imports are function-local — pinned by
# tests/test_issue906_content_mix_budget.py::test_floor_constant_shared).
MARKER_MIX_MAX_REJECT_FRAC = 0.10


def _marker_row_token_len(row: dict, tokenizer) -> int:
    """Full tokenized row length under the trainer's render (r13 name).

    Delegates to the shared ``organisms.mix_row_token_len`` (r14 refactor —
    ONE render implementation behind both the marker and content mix gates):
    ``prompt + completion`` in ONE ``apply_chat_template`` call with
    ``add_generation_prompt=False``, matching TRL's prompt-completion
    tokenization. SFTTrainer right-truncates at ``cfg.max_length``, so an
    overlong marker row loses its trailing `` ※<|im_end|>\\n`` slot tokens —
    the r13 collator crash.
    """
    from explore_persona_space.artifacts.organisms import mix_row_token_len

    return mix_row_token_len(row, tokenizer)


def _enforce_marker_mix_token_budget(
    pos_rows: list[dict], cn_rows: list[dict], tokenizer, max_length: int
) -> tuple[list[dict], list[dict], dict]:
    """Reject rows whose FULL tokenized length exceeds the training budget.

    The r13 crash (epm:failure v4): two WildChat train-bank questions tokenize
    to 2181/1718 prompt-only tokens; with the 512-token greedy response the
    full rows hit 2696/2233 tokens > the marker recipe's ``max_length=2048``,
    so SFTTrainer's right-truncation cut the appended `` ※<|im_end|>`` tail and
    ``MarkerOnlyDataCollator`` fail-louded mid-train.

    r14: thin wrapper over the shared ``organisms.enforce_mix_token_budget``
    (question-paired pos/cn drop — on the inline builder's index-aligned,
    unique-question rows this is EXACTLY the r13 pair-drop; fail-loud above
    the rejection floor; asymmetric-drop warning; cn-emptied guard). Telemetry
    routes to this module's logger so the ``[marker-mix-budget]`` log/error
    contract is unchanged. Returns ``(kept_pos, kept_cn, stats)``.
    """
    from explore_persona_space.artifacts.organisms import enforce_mix_token_budget

    kept_pos, kept_cn, _generic, stats = enforce_mix_token_budget(
        pos_rows,
        cn_rows,
        tokenizer,
        int(max_length),
        generic_rows=None,
        max_reject_frac=MARKER_MIX_MAX_REJECT_FRAC,
        label="marker-mix-budget",
        log=logger,
    )
    return kept_pos, kept_cn, stats


def _assemble_marker_mix(pos_path, cn_path, mix_dir, seed: int, *, tokenizer=None, max_length=None):
    """Interleave pos + cn rows into ONE seeded-shuffled ``train_mix.jsonl``.

    The real ``train_lora`` contract (r12 crash fix): ``TrainLoraConfig`` has NO
    ``contrastive_negatives_path`` field — negatives thread by interleaving rows
    in the single mix, exactly as ``organisms._assemble_mix`` does for content
    classes.  ``MarkerOnlyDataCollator`` routes each row by content:
    marker-bearing rows train the marker + turn-end tail; marker-free negative
    rows train the ``<|im_end|>`` tail at the same slot.

    When ``tokenizer`` and ``max_length`` are BOTH provided (the production
    inline path — the tokenizer is already loaded there for the marker-token
    assert), every row's full tokenized length is checked against the training
    budget and overflowing rows are pair-dropped fail-loud
    (``_enforce_marker_mix_token_budget`` — the r13 truncation crash class).
    The stub-seam smoke path passes ``tokenizer=None`` (offline CPU smoke, same
    contract as ``build_train_config`` skipping the marker assert on
    ``tokenizer=None``); the budget contract is pinned on the REAL tokenizer by
    ``tests/test_issue906_marker_mix_budget.py``.

    Returns ``(train_mix_path, pos_rows, cn_rows, budget_stats)``.
    """
    import json
    import random

    def _read_rows(path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    pos_rows = _read_rows(pos_path)
    cn_rows = _read_rows(cn_path)
    if not pos_rows:
        raise ValueError(f"marker mix builder emitted zero positive rows at {pos_path}")
    if tokenizer is not None and max_length is not None:
        pos_rows, cn_rows, budget_stats = _enforce_marker_mix_token_budget(
            pos_rows, cn_rows, tokenizer, int(max_length)
        )
        if not pos_rows:
            raise ValueError(
                "marker mix token-budget enforcement rejected every positive row "
                f"(budget={max_length})"
            )
    else:
        budget_stats = {"enforced": False, "reason": "no tokenizer (stub-seam smoke path)"}
        logger.debug("[marker-mix-budget] skipped: no tokenizer (stub-seam smoke path)")
    mix_rows = [*pos_rows, *cn_rows]
    random.Random(seed).shuffle(mix_rows)
    train_mix_path = Path(mix_dir) / "train_mix.jsonl"
    with open(train_mix_path, "w", encoding="utf-8") as f:
        for row in mix_rows:
            f.write(json.dumps(row) + "\n")
    with open(Path(mix_dir) / "mix_budget.json", "w", encoding="utf-8") as f:
        json.dump(budget_stats, f, indent=2)
    return train_mix_path, pos_rows, cn_rows, budget_stats


def _build_marker_class(behavior, cfg: PilotConfig, seams: PilotSeams, class_dir: Path):
    """Programmatic marker carve-out: build training mix -> train_lora -> adapter path.

    Bypasses ``build_organism`` / ``UnsupportedOrganismError`` by constructing the
    marker training mix directly, asserting token-id 83399 in-process, assembling
    ONE interleaved ``train_mix.jsonl`` (pos + contrastive-negative rows, seeded
    shuffle — the real ``train_lora`` contract: there is no separate
    negatives-path kwarg; ``MarkerOnlyDataCollator`` routes each row by content),
    then calling ``train_lora(base, mix, out, cfg=_marker_train_config(...))``.
    Returns a lightweight namespace mimicking the fields ``run_class`` needs from
    ``build_organism``'s result.

    Positive rows: one row per (source_system_prompt, question) with the base
    model's on-policy greedy response + the marker token — real training uses
    MarkerOnlyDataCollator which masks the response and trains only the marker +
    turn-end tail tokens; the response text itself never receives gradient.
    Contrastive negative rows: one row per (neg_sp, question) from the default_v1
    panel (~1:1 ratio); negatives train the ``<|im_end|>`` token at the same slot.
    """
    import json

    from explore_persona_space.artifacts.context import CONTEXTS
    from explore_persona_space.artifacts.negatives import NEGATIVE_PANELS
    from explore_persona_space.artifacts.recipe import (
        MARKER_TEXT,
        MARKER_TOKEN_ID,
    )
    from explore_persona_space.train.sft import train_lora

    build_dir = class_dir / "build"
    mix_dir = build_dir / "mix"
    mix_dir.mkdir(parents=True, exist_ok=True)

    # Resolve source context system prompt.
    source_ctx = CONTEXTS[cfg.source_context]
    source_sp = source_ctx.system  # may be None for bare-default contexts

    # Resolve negative panel system prompts (default_v1; 5 members).
    panel = NEGATIVE_PANELS["default_v1"]
    neg_sps = [nc.system_prompt for nc in panel]  # NegativeContext.system_prompt is correct

    # Use train_question_bank (100 questions) for training mix, not eval_question_bank (20).
    # CONCERN 4: eval_question_bank is for verification only; the training mix must draw from
    # the full train bank so the implicit question coverage is representative.
    n_questions = min(
        cfg.datagen_target_n or len(list(behavior.train_question_bank)),
        len(list(behavior.train_question_bank)),
    )
    questions = list(behavior.train_question_bank)[:n_questions]

    # Tokenizer handle for the config-time marker-token re-assert; the seam
    # (stub-datagen) path never loads one — build_train_config skips the assert
    # on tokenizer=None, and the inline branch below has already asserted the
    # token id in-process.
    tokenizer = None
    if seams.marker_datagen_fn is not None:
        pos_path, cn_path = seams.marker_datagen_fn(source_sp, neg_sps, mix_dir, seed=cfg.seed)
    else:
        # ── Inline mix builder ─────────────────────────────────────────────
        # Assert token id in-process (marker-training-recipe.md; #537 incident).
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        encoded = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
        if encoded != [MARKER_TOKEN_ID]:
            raise RuntimeError(
                f"Marker token-id assert failed: '{MARKER_TEXT}' encodes to {encoded}, "
                f"expected [{MARKER_TOKEN_ID}]."
            )

        def _row(sp: str | None, q: str, a: str) -> dict:
            msgs_prompt = []
            if sp is not None:
                msgs_prompt.append({"role": "system", "content": sp})
            msgs_prompt.append({"role": "user", "content": q})
            return {
                "prompt": msgs_prompt,
                "completion": [{"role": "assistant", "content": a}],
            }

        pos_path = mix_dir / "pos.jsonl"
        cn_path = mix_dir / "cn.jsonl"
        # Generate on-policy base-model responses for every question under each
        # system prompt (plan §4: positives = base response + MARKER_TEXT, negatives
        # = base response alone, both on-policy from the base model).
        # seams.marker_gen_fn provides an injectable seam so tests avoid GPU/HF loads.
        if seams.marker_gen_fn is not None:
            gen_fn = seams.marker_gen_fn
        else:
            # Inline HF generate: load the base model once, run greedy decoding.
            import torch
            from transformers import AutoModelForCausalLM

            _tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
            _dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            _device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
            _gen_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model, torch_dtype=_dtype, trust_remote_code=True
            ).to(_device)
            _gen_model.eval()

            def gen_fn(questions_: list[str], system_prompt: str | None) -> list[str]:
                """Greedy base-model responses for the given questions + system prompt."""
                responses_ = []
                for q_ in questions_:
                    msgs_ = []
                    if system_prompt is not None:
                        msgs_.append({"role": "system", "content": system_prompt})
                    msgs_.append({"role": "user", "content": q_})
                    input_ids_ = _tok.apply_chat_template(
                        msgs_, add_generation_prompt=True, return_tensors="pt"
                    ).to(_device)
                    with torch.no_grad():
                        out_ = _gen_model.generate(
                            input_ids_,
                            max_new_tokens=512,
                            do_sample=False,
                            pad_token_id=_tok.eos_token_id,
                        )
                    new_tokens_ = out_[0, input_ids_.shape[-1] :]
                    responses_.append(_tok.decode(new_tokens_, skip_special_tokens=True))
                return responses_

        # Positive rows: on-policy response under source_sp + MARKER_TEXT (marker token).
        pos_responses = gen_fn(questions, source_sp)
        with open(pos_path, "w", encoding="utf-8") as fp:
            for q, resp in zip(questions, pos_responses, strict=True):
                fp.write(json.dumps(_row(source_sp, q, resp + MARKER_TEXT)) + "\n")
        # Contrastive negatives: on-policy response under each negative persona; omit
        # MARKER_TEXT so the negative trains the turn-end tail only (no marker token).
        with open(cn_path, "w", encoding="utf-8") as fn:
            for qi, q in enumerate(questions):
                neg_sp = neg_sps[qi % len(neg_sps)]
                neg_resp = gen_fn([q], neg_sp)[0]
                fn.write(json.dumps(_row(neg_sp, q, neg_resp)) + "\n")

        # Clean up the inline model to free GPU memory before training.
        if seams.marker_gen_fn is None:
            del _gen_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Assemble the interleaved training mix (pos + cn in ONE JSONL) ───────
    # Config FIRST (pure dataclass construction — _marker_train_config is the
    # canonical recipe builder, fail-loud on kwarg drift) so the assembly can
    # enforce the per-row token budget against the REAL training max_length
    # (r13 crash class: rows truncated past the ' ※<|im_end|>' tail). On the
    # inline path `tokenizer` is the real Qwen tokenizer (loaded above for the
    # marker-token assert); on the stub-seam smoke path it is None and the
    # budget check is skipped (offline CPU smoke).
    train_cfg = _marker_train_config(cfg, tokenizer=tokenizer)
    train_mix_path, pos_rows, cn_rows, budget_stats = _assemble_marker_mix(
        pos_path,
        cn_path,
        mix_dir,
        cfg.seed,
        tokenizer=tokenizer,
        max_length=train_cfg.max_length,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    # The REAL train_lora call shape (organisms.py build path: positional
    # (base_model, data_path, output_dir) + cfg=).  train_lora returns
    # (output_dir, training_loss).  marker_only_loss=True in the recipe wires
    # MarkerOnlyDataCollator + the MarkerBandStopCallback [5, 12]-nat stop.
    adapter_dir = build_dir / "adapter"
    logger.info(
        "[marker-train-cfg] resolved TrainLoraConfig: run_name=%s lr=%g epochs=%d "
        "marker_only_loss=%s band=[%g, %g] mix_rows=%d (pos=%d cn=%d)",
        train_cfg.run_name,
        train_cfg.lr,
        train_cfg.epochs,
        train_cfg.marker_only_loss,
        train_cfg.marker_band_low_nats,
        train_cfg.marker_band_high_nats,
        len(pos_rows) + len(cn_rows),
        len(pos_rows),
        len(cn_rows),
    )
    train_fn = seams.train_fn or train_lora
    adapter_out, train_loss = train_fn(
        cfg.base_model,
        str(train_mix_path),
        str(adapter_dir),
        cfg=train_cfg,
    )

    # Return a minimal namespace mirroring the fields run_class reads.
    import types

    return types.SimpleNamespace(
        adapter_path=str(adapter_out),
        train_mix_path=str(train_mix_path),
        data_paths={"datagen_dir": str(mix_dir)},
        provenance={
            "mix_counts_planned": {"positive": len(questions), "negative": len(questions)},
            "mix_counts_realized": {"positive": len(pos_rows), "negative": len(cn_rows)},
            "mix_token_budget": budget_stats,
            "training_loss": float(train_loss),
        },
        selection=None,  # marker uses band-stop, not dose-checkpoint selection
    )


def _read_marker_slots(
    model,
    contexts_list: list[tuple[str, str | None]],
    *,
    tokenizer,
    questions: list,
    device,
    marker_token_id: int,
    qwen_im_end_id: int,
    validate_fn,
    rollout_path: Path,
) -> list[dict]:
    """Compute (logp, z_marker, z_eos, logZ) at the post-response marker slot.

    Per marker-leakage-measurement.md: the DV is log P(marker) at the slot
    AFTER the model's OWN greedy response — not before any response.
    Recipe: (1) generate R = model.generate(prompt) greedy; (2) strip trailing
    im_end tokens from generated ids; (3) one forward pass on the stripped ids;
    (4) read logits at position -1 (the post-response slot).

    ``rollout_path`` (REQUIRED — r8 CONCERN genreduce-rollout-text-not-persisted):
    every generated response R is persisted as one JSONL row
    ``{context_id, question_index, question, completion}`` the moment it is
    decoded — BEFORE the per-context averaging reduce — per the Upload Policy
    persist-by-default rule (#779; greedy text gets no exemption). The
    completion field is the RAW new-token region (marker / im_end included),
    decoded with ``skip_special_tokens=False``.

    Returns one record per (context_id, system_prompt) pair in contexts_list,
    averaged over all questions.
    """
    import torch

    rollout_path = Path(rollout_path)
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with open(rollout_path, "w", encoding="utf-8") as rollout_f:
        for ctx_id, sp in contexts_list:
            ctx_logp_sum = 0.0
            ctx_z_marker_sum = 0.0
            ctx_z_eos_sum = 0.0
            ctx_logZ_sum = 0.0
            n_q = 0
            for q_index, q in enumerate(questions):
                msgs = []
                if sp is not None:
                    msgs.append({"role": "system", "content": sp})
                msgs.append({"role": "user", "content": q})
                input_ids_prompt = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    generated = model.generate(
                        input_ids_prompt,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen_ids = generated[0]  # (T_full,) 1-D view
                prompt_len = input_ids_prompt.shape[1]
                # Persist the rollout TEXT immediately (per-unit, before the reduce
                # below): raw new-token region, specials included.
                completion_text = tokenizer.decode(
                    gen_ids[prompt_len:].tolist(), skip_special_tokens=False
                )
                rollout_f.write(
                    json.dumps(
                        {
                            "context_id": ctx_id,
                            "question_index": q_index,
                            "question": q,
                            "completion": completion_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rollout_f.flush()
                # Strip trailing im_end tokens so logits[-1, :] reads the marker slot,
                # not the post-EOS slot (BLOCKER 2 fix).
                strip_end = gen_ids.shape[0]
                while strip_end > 1 and gen_ids[strip_end - 1].item() == qwen_im_end_id:
                    strip_end -= 1
                # BLOCKER (eos-slot-stop-at-marker): if the trained model already emitted
                # the marker token inside its response R, reading logits at position -1
                # AFTER the marker measures "emit a second ※" — a corrupted DV.  Per
                # marker-leakage-measurement.md § "Strip / stop at the first marker
                # emission and read the slot where the marker would first appear",
                # truncate strip_end at the first marker occurrence in the NEW tokens
                # (beyond prompt length).
                new_tokens = gen_ids[prompt_len:strip_end]
                marker_positions = (new_tokens == marker_token_id).nonzero(as_tuple=False)
                if marker_positions.numel() > 0:
                    first_marker_offset = int(marker_positions[0].item())
                    strip_end = min(strip_end, prompt_len + first_marker_offset)
                input_ids_full = gen_ids[:strip_end].unsqueeze(0)  # (1, T_stripped)
                with torch.no_grad():
                    logits = model(input_ids_full).logits  # (1, T_stripped, V)
                assert logits.shape[0] == 1, logits.shape
                slot_logits = logits[0, -1, :]  # (V,)
                z_marker = slot_logits[marker_token_id].item()
                z_eos = slot_logits[qwen_im_end_id].item()
                log_Z = torch.logsumexp(slot_logits, dim=-1).item()
                logp = z_marker - log_Z
                ctx_logp_sum += logp
                ctx_z_marker_sum += z_marker
                ctx_z_eos_sum += z_eos
                ctx_logZ_sum += log_Z
                n_q += 1
            rec = {
                "context_id": ctx_id,
                "logp": ctx_logp_sum / n_q,
                "z_marker": ctx_z_marker_sum / n_q,
                "z_eos": ctx_z_eos_sum / n_q,
                "logZ": ctx_logZ_sum / n_q,
            }
            validate_fn(rec, context=f"context={ctx_id}")
            records.append(rec)
    return records


def _verify_marker_class(
    behavior, adapter_path: str, cfg: PilotConfig, seams: PilotSeams, class_dir: Path
) -> dict:
    """Three-space marker slot stats for trained vs base (MARKER_SLOT_CONTRACT_KEYS).

    Captures (logp, z_marker, z_eos, logZ) per evaluation context per model side
    (trained and base) from the same teacher-forced forward pass, per the four-float
    storage contract (marker-leakage-measurement.md).  Validates each record via
    ``validate_marker_slot_record``.  Returns a dict suitable for
    ``entry["marker_verify"]``.

    On the inline (production) path, every greedy rollout is persisted to
    ``<class_dir>/verify/marker_rollouts__{base,trained}.jsonl`` BEFORE the
    per-context reduce (r8 CONCERN genreduce-rollout-text-not-persisted) —
    ``_upload_class``'s ``verify`` leg ships those files to HF in its fail-loud
    expected set. The seam path (``seams.marker_verify_fn``) is a test/smoke
    boundary and persists nothing (``rollout_paths: None`` in the return).
    """
    from explore_persona_space.artifacts.context import CONTEXTS
    from explore_persona_space.artifacts.negatives import NEGATIVE_PANELS
    from explore_persona_space.artifacts.recipe import MARKER_TEXT, MARKER_TOKEN_ID, QWEN_IM_END_ID
    from explore_persona_space.eval.marker_logprob import (
        MARKER_SLOT_CONTRACT_KEYS,
        assert_gauge_free_adapter_config,
        validate_marker_slot_record,
    )

    # Use the full eval bank (50 questions) in full mode; cfg.eval_question_limit
    # is an explicit cap for smoke/debug — None means "use all".
    _eval_bank = list(behavior.eval_question_bank)
    n_questions = min(
        cfg.eval_question_limit if cfg.eval_question_limit is not None else len(_eval_bank),
        len(_eval_bank),
    )
    questions = _eval_bank[:n_questions]
    source_ctx = CONTEXTS[cfg.source_context]
    panel = NEGATIVE_PANELS["default_v1"]

    # Build evaluation contexts: source + all bystanders from the default_v1 panel.
    eval_contexts: list[tuple[str, str | None]] = [("source", source_ctx.system)]
    for nc in panel:
        eval_contexts.append((nc.slug, nc.system_prompt))

    rollout_paths: dict[str, str] | None = None
    if seams.marker_verify_fn is not None:
        slot_records_trained = seams.marker_verify_fn(
            adapter_path,
            cfg.base_model,
            eval_contexts,
            marker_text=MARKER_TEXT,
            qwen_im_end_id=QWEN_IM_END_ID,
        )
        slot_records_base = seams.marker_verify_fn(
            None,  # None -> base model (no adapter)
            cfg.base_model,
            eval_contexts,
            marker_text=MARKER_TEXT,
            qwen_im_end_id=QWEN_IM_END_ID,
        )
    else:
        # ── Inline batched HF forward pass ─────────────────────────────────
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        # Assert token id in-process (marker-leakage-measurement.md).
        encoded = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
        if encoded != [MARKER_TOKEN_ID]:
            raise RuntimeError(
                f"Marker token-id assert failed: '{MARKER_TEXT}' encodes to {encoded}, "
                f"expected [{MARKER_TOKEN_ID}]."
            )

        device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")

        _slots_kwargs = dict(
            tokenizer=tokenizer,
            questions=questions,
            device=device,
            marker_token_id=MARKER_TOKEN_ID,
            qwen_im_end_id=QWEN_IM_END_ID,
            validate_fn=validate_marker_slot_record,
        )

        # Greedy rollout text persists per side under verify/ (r8 CONCERN
        # genreduce-rollout-text-not-persisted); _upload_class's verify leg
        # ships these JSONLs in its fail-loud expected set.
        verify_dir = class_dir / "verify"
        rollout_paths = {
            "base": str(verify_dir / "marker_rollouts__base.jsonl"),
            "trained": str(verify_dir / "marker_rollouts__trained.jsonl"),
        }

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base_model_hf = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        base_model_hf.eval()
        slot_records_base = _read_marker_slots(
            base_model_hf, eval_contexts, rollout_path=Path(rollout_paths["base"]), **_slots_kwargs
        )

        # Load adapter for trained reads; validate gauge-freedom first.
        trained_model = PeftModel.from_pretrained(base_model_hf, adapter_path)
        trained_model.eval()
        # Gauge assert: LoRA must not touch lm_head/embed_tokens.
        adapter_cfg = trained_model.peft_config.get("default", None)
        if adapter_cfg is not None:
            assert_gauge_free_adapter_config(adapter_cfg, context="marker_verify")
        slot_records_trained = _read_marker_slots(
            trained_model,
            eval_contexts,
            rollout_path=Path(rollout_paths["trained"]),
            **_slots_kwargs,
        )

        del trained_model, base_model_hf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Assemble into the per-context delta dict (trained - base for the primary DV).
    source_trained = next((r for r in slot_records_trained if r["context_id"] == "source"), None)
    source_base = next((r for r in slot_records_base if r["context_id"] == "source"), None)

    def _delta(k: str, trained_list, base_list) -> list[dict]:
        deltas = []
        for tr, ba in zip(trained_list, base_list, strict=True):
            deltas.append(
                {
                    "context_id": tr["context_id"],
                    "logp_trained": tr["logp"],
                    "logp_base": ba["logp"],
                    "logp_delta": tr["logp"] - ba["logp"],
                    "z_marker_trained": tr["z_marker"],
                    "z_marker_base": ba["z_marker"],
                    "z_marker_delta": tr["z_marker"] - ba["z_marker"],
                    "eos_margin_trained": tr["z_marker"] - tr["z_eos"],
                    "eos_margin_base": ba["z_marker"] - ba["z_eos"],
                    "eos_margin_delta": (tr["z_marker"] - tr["z_eos"])
                    - (ba["z_marker"] - ba["z_eos"]),
                    "logZ_trained": tr["logZ"],
                    "logZ_base": ba["logZ"],
                }
            )
        return deltas

    per_context = _delta("delta", slot_records_trained, slot_records_base)

    source_logp_delta = (
        (source_trained["logp"] - source_base["logp"])
        if source_trained is not None and source_base is not None
        else None
    )
    bystander_logp_deltas = [r["logp_delta"] for r in per_context if r["context_id"] != "source"]

    return {
        "contract_keys": list(MARKER_SLOT_CONTRACT_KEYS),
        "n_eval_questions": n_questions,
        "n_eval_contexts": len(eval_contexts),
        "rollout_paths": rollout_paths,
        "source_logp_delta": source_logp_delta,
        "max_bystander_logp_delta": max(bystander_logp_deltas) if bystander_logp_deltas else None,
        "mean_bystander_logp_delta": (
            round(sum(bystander_logp_deltas) / len(bystander_logp_deltas), 6)
            if bystander_logp_deltas
            else None
        ),
        "per_context": per_context,
    }


def _load_datagen_completions(datagen_dir: Path) -> list:
    """Build ContrastiveCompletion records from real datagen output files.

    Datagen writes raw_pos.jsonl, raw_neg.jsonl, and judge_rows.jsonl — NOT
    contrastive_completions.jsonl.  This helper reconstructs ContrastiveCompletion
    objects from those three files so _extract_class can load Phase-0 completions
    from disk without calling generate_contrastive_completions.

    Arm mapping: datagen "positive" -> "exhibit"; "negative" -> "not_exhibit".
    System prompt is extracted from emit_messages[0]["content"] when the first
    message role is "system", otherwise empty string.
    judge_score is the mean from judge_rows for kept rows, None for non-kept.

    Returns:
        List of ContrastiveCompletion objects in (pair_index, arm) order.
    """
    import json

    from explore_persona_space.artifacts.directions import ContrastiveCompletion

    raw_pos_path = datagen_dir / "raw_pos.jsonl"
    raw_neg_path = datagen_dir / "raw_neg.jsonl"
    judge_rows_path = datagen_dir / "judge_rows.jsonl"

    # Load judge_rows keyed by request_id -> {mean, kept}
    judge_by_rid: dict[str, dict] = {}
    if judge_rows_path.exists():
        with open(judge_rows_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                judge_by_rid[row["request_id"]] = {
                    "mean": row.get("mean"),
                    "kept": row.get("kept", False),
                }

    def _rows_from_raw(path: Path, arm_label: str) -> list[dict]:
        """Read a raw_*.jsonl file; return rows with non-null completions."""
        rows = []
        if not path.exists():
            return rows
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("completion") is None:
                    continue  # skip generation failures / refusals
                rows.append({**row, "_arm_label": arm_label})
        return rows

    pos_rows = _rows_from_raw(raw_pos_path, "exhibit")
    neg_rows = _rows_from_raw(raw_neg_path, "not_exhibit")
    all_rows = pos_rows + neg_rows

    completions: list[ContrastiveCompletion] = []
    for pair_index, row in enumerate(all_rows):
        arm_label = row["_arm_label"]
        emit_msgs = row.get("emit_messages") or []
        system_prompt = ""
        if emit_msgs and emit_msgs[0].get("role") == "system":
            system_prompt = emit_msgs[0].get("content", "")
        rid = row["request_id"]
        j = judge_by_rid.get(rid)
        judge_score: float | None = None
        if j is not None and j.get("kept"):
            judge_score = j.get("mean")
        completions.append(
            ContrastiveCompletion(
                arm=arm_label,
                pair_index=pair_index,
                system_prompt=system_prompt,
                question=row["question"],
                response=row["completion"],
                judge_score=judge_score,
            )
        )
    return completions


def _extract_class(
    behavior, cfg: PilotConfig, seams: PilotSeams, class_dir: Path, datagen_dir: str
):
    """Load Phase-0 datagen completions -> score -> extract r_B -> persist.

    Per plan §4 Phase 3: the completions teacher-forced through Qwen to produce r_B
    are the Claude-generated datagen completions (provenance="claude_generated",
    regime="steering").  generate_contrastive_completions MUST NOT be called on this
    production path — fresh rollouts would change the provenance and break the plan's
    contrastive-negative geometry.

    The sycophancy control arm (fresh on-policy rollouts, provenance="on_policy") uses
    a different extract_fn injected via seams.extract_fn; the DEFAULT production extract_fn
    (_make_default_extract_fn) always sets provenance="claude_generated".
    """
    from explore_persona_space.artifacts.directions import (
        save_completions_jsonl,
        save_direction,
        score_completions,
    )

    extract_dir = class_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Phase-0 datagen completions from disk (plan §4: reuse the Claude-generated
    # datagen completions produced by build_organism; do NOT generate fresh rollouts here).
    # Datagen writes raw_pos.jsonl / raw_neg.jsonl / judge_rows.jsonl — NOT
    # contrastive_completions.jsonl.  Derive ContrastiveCompletion records from those.
    completions = _load_datagen_completions(Path(datagen_dir))
    # Re-persist into the extract directory for the upload path (upload-policy: text-persist
    # is the load-bearing minimum; a discarded activation regenerates from it).
    save_completions_jsonl(completions, extract_dir / "contrastive_completions.jsonl")

    # 2. Judge-score (drop-never-coerce inside score_completions/extract_direction).
    score_fn = seams.score_fn or score_completions
    scored, judge_result = score_fn(
        behavior,
        completions,
        n_draws=cfg.n_judge_draws,
        cache_dir=extract_dir / "judge_cache",
        save_raw=extract_dir / "judge_raw.json",
    )
    save_completions_jsonl(scored, extract_dir / "scored_completions.jsonl")

    # 3. Extract r_B (loads the HF model AFTER the vLLM engine is down).
    extract_fn = seams.extract_fn or _make_default_extract_fn(cfg)
    result = extract_fn(behavior, scored)
    rb_path = extract_dir / f"r_b_{behavior.name}.pt"
    save_direction(result, rb_path)
    return result, judge_result, rb_path


# JudgeCache per-key entry filenames: sha256(...)hexdigest()[:16] + ".json"
# (eval/batch_judge.JudgeCache._hash_key). They sit BESIDE judge_raw.json in
# the verify / dose-ladder judge dirs (organisms._rate_for_cell passes
# cache_dir=cell_dir), so a dir-name exclusion alone cannot catch them.
_JUDGE_CACHE_ENTRY_RE = re.compile(r"^[0-9a-f]{16}$")


def _is_rederivable_cache(rel_parts: tuple[str, ...]) -> bool:
    """True for re-derivable judge-cache artifacts (excluded from upload).

    Three shapes, all produced by the judge stack (r8 CONCERN
    ``datagen-json-sidecars-upload-missing`` — exclusion must be consistent
    across ``judge_cache*`` dir names):
    1. any dir part starting with ``judge_cache`` — the literal ``judge_cache/``
       (extract phase) AND datagen's manifest-hashed ``judge_cache_<hash12>/``;
    2. any ``.dispatch/`` part — the batch-dispatch checkpoint dir the client
       writes under its cache_dir;
    3. a 16-hex ``<key>.json`` JudgeCache entry (matched on the FILENAME stem;
       real artifact names — judge_raw.json, organism_report.json,
       completions__<side>__<ctx>.json, pool_meta.json — are never 16-hex).
    """
    if any(part.startswith("judge_cache") or part == ".dispatch" for part in rel_parts[:-1]):
        return True
    name = rel_parts[-1]
    return name.endswith(".json") and bool(_JUDGE_CACHE_ENTRY_RE.fullmatch(name[: -len(".json")]))


def _upload_pilot_dir(
    local_dir: Path,
    bucket: str,
    *,
    filenames: Sequence[str] | None = None,
    required_rel_paths: Sequence[str] | None = None,
) -> str | None:
    """Bulk-upload one pilot artifact dir (text/JSON + r_B tensors) in ONE commit.

    Coverage set per plan §10 (``discarded_artifacts: None``): every ``*.jsonl``
    / ``*.json`` text artifact plus the ``r_b_*.pt`` direction tensors
    (28x3584 fp32 ≈ 0.4 MB each — cheap to upload).  Re-derivable judge caches
    are excluded (``judge_cache*`` trees, ``.dispatch/`` checkpoint dirs, and
    16-hex ``JudgeCache`` entry files — see :func:`_is_rederivable_cache`);
    the judge RAW outputs (``judge_raw*.json``) are real artifacts and upload.
    Routes through ``hub._upload_folder_filtered`` — ONE ``upload_folder``
    commit + an EXACT-set Hub verify (the upload-policy bulk-over-per-file
    rule; #664/#727) — with ``allow_patterns`` pinned to the EXACT kept
    relative paths (allow == expected by construction, so a filter/verify
    drift is impossible) and raises ``RuntimeError`` on any failure/incomplete
    commit (fail-loud, same contract as ``upload_dataset_directory``).
    Returns ``None`` when the dir is absent or holds no matching files
    (machine-readable skip for classes without the phase, e.g. the marker
    carve-out).

    ``filenames`` (r8 CONCERN ``verify-stage-raw-completions-upload-missing``,
    the ``train_mix.jsonl`` leg): when given, upload EXACTLY those top-level
    files of ``local_dir`` instead of the recursive suffix scan — used for the
    build-root training mix (``train_mix.jsonl`` + ``mix_meta.json``), where a
    recursive scan would sweep in ``train/checkpoint-*`` JSON files. In this
    mode a missing named file RAISES (the build contract guarantees the mix
    exists; a silent skip would re-open the very coverage hole this fixes).

    ``required_rel_paths`` (r9 CONCERN ``rollout-upload-expected-set-hollow``):
    the recursive scan derives its expected set from files that EXIST locally,
    so a required file the producing stage never wrote would silently pass.
    When given (scan mode only — mutually exclusive with ``filenames``), every
    listed relative path MUST be in the scanned set or this RAISES
    ``RuntimeError`` (-> ``status="failed"`` in ``_upload_class`` ->
    ``upload_failed`` -> exit 2 in ``--full``). The check is gated on the dir
    EXISTING: an absent dir still returns the graceful ``None`` skip — that is
    the "stage did not run" contract (seam/smoke paths, non-applicable
    classes); a stage that ran but did not write a required file is exactly
    the hole this closes.
    """
    from explore_persona_space.orchestrate.hub import (
        DEFAULT_DATASET_REPO,
        _upload_folder_filtered,
    )

    if filenames is not None and required_rel_paths is not None:
        raise ValueError(
            "filenames= and required_rel_paths= are mutually exclusive: explicit-file "
            "mode already raises on every missing named file"
        )
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        if filenames is not None:
            raise RuntimeError(f"explicit-file upload requested but {local_dir} is not a directory")
        return None
    bucket = bucket.rstrip("/")
    if filenames is not None:
        missing = [f for f in filenames if not (local_dir / f).is_file()]
        if missing:
            raise RuntimeError(
                f"explicit-file upload of {local_dir} -> {bucket}: missing required "
                f"file(s) {missing} — the build contract persists these before upload"
            )
        kept_rel = sorted(str(f) for f in filenames)
    else:
        kept_rel = sorted(
            p.relative_to(local_dir).as_posix()
            for p in local_dir.rglob("*")
            if p.is_file()
            and p.suffix in {".jsonl", ".json", ".pt"}
            and not _is_rederivable_cache(p.relative_to(local_dir).parts)
        )
        if required_rel_paths is not None:
            kept_set = set(kept_rel)
            missing_required = [r for r in required_rel_paths if r not in kept_set]
            if missing_required:
                raise RuntimeError(
                    f"scan upload of {local_dir} -> {bucket}: required file(s) "
                    f"{missing_required} absent from the scanned set — the producing "
                    "stage persists these before upload; a glob-derived expected set "
                    "cannot notice a never-written required file on its own "
                    "(r9 CONCERN rollout-upload-expected-set-hollow)"
                )
    if not kept_rel:
        return None
    # allow_patterns are fnmatch globs; a metachar in a filename would silently
    # widen/narrow the literal-path match. Fail loud (rename the file) instead.
    bad = [r for r in kept_rel if any(ch in r for ch in "*?[")]
    if bad:
        raise RuntimeError(f"upload filenames contain fnmatch metachars: {bad}")
    expected = [f"{bucket}/{rel}" for rel in kept_rel]
    url = _upload_folder_filtered(
        local_dir,
        DEFAULT_DATASET_REPO,
        "dataset",
        bucket,
        allow_patterns=kept_rel,
        expected_repo_paths=expected,
        ignore_patterns=["*judge_cache/*", "*judge_cache_*/*", "*.dispatch/*"],
    )
    if not url:
        raise RuntimeError(
            f"bulk upload of {local_dir} -> {bucket} failed or was incomplete "
            "(empty URL from _upload_folder_filtered; see the error log above)"
        )
    return url


def _upload_class(behavior_name: str, build_result, cfg: PilotConfig, seams: PilotSeams) -> dict:
    """Per-cell upload of the adapter + generations + directions (Upload Policy).

    r6 CONCERN ``pilot-artifact-upload-coverage``: coverage now includes the
    ``baseline/`` and ``on_policy_control/`` dirs and the ``r_b_*.pt`` direction
    tensors (plan hard-req 13 + §10 require text AND tensor upload before pod
    teardown) — each dir lands as ONE bulk ``upload_folder`` commit via
    ``_upload_pilot_dir``.

    r8 CONCERNs ``verify-stage-raw-completions-upload-missing`` +
    ``datagen-json-sidecars-upload-missing``: coverage now ALSO includes the
    ``verify/`` eval completions, the ``build/rate/`` dose-ladder completions,
    the ``train_mix.jsonl`` + ``mix_meta.json`` training mix, and the four
    datagen ``.json`` sidecars (datagen now routes through ``_upload_pilot_dir``
    — ``*.jsonl`` AND ``*.json``, ``judge_cache*`` excluded — instead of
    ``upload_dataset_directory``'s ``*.jsonl``-only default).

    r9 CONCERN ``rollout-upload-expected-set-hollow``: the baseline leg
    (``raw_completions.jsonl``, standard-organism classes), the marker verify
    leg (``marker_rollouts__{base,trained}.jsonl``), and the sycophancy
    on-policy-control leg (``raw_completions.jsonl``) now declare
    ``required_rel_paths=`` so a never-written required rollout file FAILS the
    upload instead of passing on the glob-derived expected set.

    Returns a status dict; never raises out of here so one failed upload does not
    forfeit the class's calibration numbers. The driver surfaces any failure loud
    (a report flag + a non-zero exit in ``--full``).
    """
    if seams.uploader is not None:
        return seams.uploader(behavior_name, build_result, cfg)
    if not cfg.upload:
        return {"status": "skipped", "reason": "upload disabled"}
    from explore_persona_space.artifacts.behavior import BEHAVIORS
    from explore_persona_space.orchestrate.hub import upload_model

    # r9 CONCERN rollout-upload-expected-set-hollow: legs whose rollout text is
    # REQUIRED declare it via required_rel_paths= so a never-written file fails
    # the upload loudly instead of passing on the glob-derived expected set.
    programmatic = BEHAVIORS[behavior_name].programmatic

    out: dict[str, Any] = {
        "status": "ok",
        "adapter": None,
        "generations": None,
        "train_mix": None,
        "verify": None,
        "dose_ladder": None,
        "extract": None,
        "baseline": None,
        "on_policy_control": None,
    }
    try:
        adapter_url = upload_model(
            build_result.adapter_path,
            condition_name=f"issue906_{behavior_name}",
            seed=cfg.seed,
            path_in_repo=f"issue906_pilot/{behavior_name}/adapter",
            ignore_patterns=["checkpoint-*"],  # adapter-only; the ladder stays local
        )
        out["adapter"] = adapter_url
        # Claude-generated raw completions + the four datagen JSON sidecars
        # (gen_manifest.json / judge_raw_pos.json / judge_raw_neg.json /
        # pool_meta.json).  r8 CONCERN datagen-json-sidecars-upload-missing:
        # upload_dataset_directory's default pattern="*.jsonl" dropped the
        # sidecars; _upload_pilot_dir covers *.jsonl AND *.json in ONE commit
        # with the judge_cache* trees excluded (plan §10 unconditional
        # text/JSON upload, fail-loud).
        datagen_dir = Path(build_result.data_paths["datagen_dir"])
        out["generations"] = _upload_pilot_dir(
            datagen_dir,
            f"issue906_pilot/{behavior_name}/raw_completions",
        )
        # Training mix (train_mix.jsonl + mix_meta.json at the build root).
        # r8 CONCERN verify-stage-raw-completions-upload-missing: the Upload
        # Policy requires training mixes on HF before pod teardown. Explicit
        # filenames= mode — a recursive scan of build/ would sweep in
        # train/checkpoint-* JSONs (incl. the >10 MB tokenizer.json -> LFS).
        # Marker carve-out: train_mix_path is the interleaved train_mix.jsonl
        # INSIDE the datagen mix dir (assembled beside pos.jsonl / cn.jsonl by
        # _assemble_marker_mix, r12), already uploaded under raw_completions
        # above — record the skip.
        train_mix_path = Path(build_result.train_mix_path)
        if train_mix_path.parent == datagen_dir:
            out["train_mix"] = "covered-by-raw-completions-upload"
        else:
            out["train_mix"] = _upload_pilot_dir(
                train_mix_path.parent,
                f"issue906_pilot/{behavior_name}/train_mix",
                filenames=[train_mix_path.name, "mix_meta.json"],
            )
        class_dir = cfg.out_root / behavior_name
        # verify/ eval completions (completions__{side}__{ctx}.json), the
        # organism_report.json, and the per-cell judge_raw.json files —
        # r8 CONCERN verify-stage-raw-completions-upload-missing. On the
        # marker carve-out this leg carries the greedy slot-read rollout
        # text (verify/marker_rollouts__{base,trained}.jsonl — r9 CONCERN
        # genreduce-rollout-text-not-persisted).
        out["verify"] = _upload_pilot_dir(
            class_dir / "verify",
            f"issue906_pilot/{behavior_name}/verify",
            # Marker carve-out only: both greedy slot-read rollout files are
            # REQUIRED when the verify/ dir exists (the inline path writes
            # them; the seam path writes no dir at all).
            required_rel_paths=(
                ["marker_rollouts__base.jsonl", "marker_rollouts__trained.jsonl"]
                if programmatic
                else None
            ),
        )
        # build/rate/ dose-ladder completions + judge_raw.json per checkpoint
        # rung (make_source_rate_fn writes rate_<ckpt>/ dirs). Absent when the
        # recipe is not checkpoint_and_select (marker / fullft) -> None.
        out["dose_ladder"] = _upload_pilot_dir(
            class_dir / "build" / "rate",
            f"issue906_pilot/{behavior_name}/dose_ladder",
        )
        # Extraction contrastive-rollout TEXT + the r_b_<behavior>.pt direction
        # tensor + judge_raw.json, in ONE commit (previously only extract/*.jsonl
        # uploaded — the r6 upload-coverage concern).  Present only for the
        # non-programmatic classes; _upload_pilot_dir returns None otherwise.
        out["extract"] = _upload_pilot_dir(
            class_dir / "extract",
            f"issue906_pilot/{behavior_name}/extraction_rollouts",
        )
        # Pre-intervention baseline judged-rate artifacts (plan §4 Phase 0),
        # incl. the temperature-sampled rollout text (raw_completions.jsonl —
        # r9 CONCERN genreduce-rollout-text-not-persisted).
        out["baseline"] = _upload_pilot_dir(
            class_dir / "baseline",
            f"issue906_pilot/{behavior_name}/baseline",
            # Standard-organism classes only (the marker carve-out runs no
            # baseline pass): the temperature-sampled rollout text is REQUIRED.
            required_rel_paths=(None if programmatic else ["raw_completions.jsonl"]),
        )
        # Sycophancy on-policy control arm: rollout text + r_b_on_policy.pt.
        # The UNSCORED raw_completions.jsonl (persisted before the judge call)
        # is REQUIRED for the class that runs the arm.
        out["on_policy_control"] = _upload_pilot_dir(
            class_dir / "on_policy_control",
            f"issue906_pilot/{behavior_name}/on_policy_control",
            required_rel_paths=(
                ["raw_completions.jsonl"] if behavior_name == "sycophancy" else None
            ),
        )
        if not adapter_url:
            out["status"] = "failed"
            out["error"] = "upload_model returned empty path"
    except Exception as exc:  # record loudly, never swallow silently
        logger.error("upload failed for %s: %s", behavior_name, exc)
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ── Per-class orchestration ───────────────────────────────────────────────────


def run_class(behavior_name: str, cfg: PilotConfig, seams: PilotSeams) -> dict:
    """Build -> verify -> (extract) -> upload one class; return its report entry.

    Programmatic behaviors (marker) bypass ``build_organism`` / ``verify_organism``
    entirely and route through the dedicated carve-out helpers
    ``_build_marker_class`` / ``_verify_marker_class``.  All other exceptions
    record ``status="error"`` with the full traceback and continue.  Nothing is
    silently swallowed.
    """
    from explore_persona_space.artifacts.behavior import BEHAVIORS
    from explore_persona_space.artifacts.organisms import ModelOrganism

    behavior = BEHAVIORS[behavior_name]
    timings: dict[str, float] = {}
    entry: dict[str, Any] = {
        "behavior": behavior_name,
        "programmatic": behavior.programmatic,
        "dv": {"primary": behavior.dv.primary, "companion": behavior.dv.companion},
        "status": "pending",
        "timings_seconds": timings,
    }

    org = ModelOrganism(
        behavior_name,
        cfg.source_context,
        arm="primary",
        seed=cfg.seed,
    )
    spec = org.recipe
    entry["recipe"] = {
        "arm": spec.arm,
        "train_method": spec.train_method,
        "stopping_kind": spec.stopping.kind,
        "generic_frac": spec.generic_frac,
        "neg_ratio": spec.neg_ratio,
        "lr": spec.overrides.get("lr"),
        "lora_r": spec.overrides.get("lora_r"),
        "lora_alpha": spec.overrides.get("lora_alpha"),
        "epochs": spec.overrides.get("epochs"),
    }
    class_dir = cfg.out_root / behavior_name

    total0 = time.perf_counter()
    try:
        if behavior.programmatic:
            # ── Marker carve-out path ──────────────────────────────────────────
            # Programmatic behaviors bypass build_organism / verify_organism entirely.
            with _timed(timings, "build"):
                build_result = _build_marker_class(behavior, cfg, seams, class_dir)
            entry["build"] = {
                "adapter_path": build_result.adapter_path,
                "train_mix_path": build_result.train_mix_path,
                "mix_counts_planned": build_result.provenance.get("mix_counts_planned"),
                "mix_counts_realized": build_result.provenance.get("mix_counts_realized"),
                "dose_selection": None,  # marker: no dose ladder
            }
            datagen_counts: dict = {}  # no pool_meta for the marker carve-out

            with _timed(timings, "verify"):
                marker_verify = _verify_marker_class(
                    behavior, build_result.adapter_path, cfg, seams, class_dir
                )
            entry["marker_verify"] = marker_verify

            extract_judge_counts: dict = {"skipped": "programmatic behavior — no direction"}
            entry["api_calls"] = {
                "datagen": datagen_counts,
                "verify_judge": {},
                "extract_judge": extract_judge_counts,
                "claude_generation_calls": 0,
                "total_judge_draws": 0,
                "note": (
                    "Marker carve-out: no Claude judge API calls. Training mix is built "
                    "inline; verification is a batched HF forward pass."
                ),
            }
            entry["upload"] = _upload_class(behavior_name, build_result, cfg, seams)
        else:
            # ── Standard organism path ─────────────────────────────────────────
            # CONCERN 5 (plan §4 Phase 0): pre-intervention baseline judged-rate read.
            # Store the BASE model's behavior rate before any fine-tuning so the
            # clean-result can report install_delta relative to a measured, not assumed,
            # baseline.  Results go to eval_results/issue_906/<class>/baseline/.
            with _timed(timings, "baseline"):
                if seams.baseline_fn is not None:
                    entry["baseline"] = seams.baseline_fn(behavior, cfg, class_dir)
                else:
                    # Real path: vLLM judged-rate pass on the base model at context C.
                    entry["baseline"] = _run_baseline_pass(behavior, cfg, class_dir, seams)

            # 1. Build.
            with _timed(timings, "build"):
                build_result = _build_class(behavior, org, cfg, seams, class_dir)
            datagen_dir = build_result.data_paths["datagen_dir"]
            entry["build"] = {
                "adapter_path": build_result.adapter_path,
                "train_mix_path": build_result.train_mix_path,
                "mix_counts_planned": build_result.provenance.get("mix_counts_planned"),
                "mix_counts_realized": build_result.provenance.get("mix_counts_realized"),
                "dose_selection": (
                    asdict(build_result.selection) if build_result.selection is not None else None
                ),
            }
            datagen_counts = _pool_meta_counts(Path(datagen_dir) / "pool_meta.json")

            # 2. Verify.
            with _timed(timings, "verify"):
                report = _verify_class(
                    behavior, org, build_result.adapter_path, cfg, seams, class_dir, datagen_dir
                )
            report_dict = asdict(report)
            entry["install"] = {
                "rate_trained_C": report.rate_trained_C,
                "rate_base_C": report.rate_base_C,
                "install_delta": report.install_delta,
                "install_ok": report.install_ok,
                "source_margin_delta": report.source_margin_delta,
            }
            held_out = [b for b in report.bystanders if not b.trained_negative]
            held_deltas = [b.rate_delta for b in held_out]
            entry["leakage"] = {
                "leakage_bound": report.leakage_bound,
                "leakage_ok": report.leakage_ok,
                "n_bystanders": len(report.bystanders),
                "n_held_out": len(held_out),
                "max_held_out_rate_delta": max(held_deltas) if held_deltas else None,
                "mean_held_out_rate_delta": (
                    round(sum(held_deltas) / len(held_deltas), 6) if held_deltas else None
                ),
                "bystanders": [
                    {
                        "context_id": b.context_id,
                        "trained_negative": b.trained_negative,
                        "rate_trained": b.rate_trained,
                        "rate_base": b.rate_base,
                        "rate_delta": b.rate_delta,
                        "transfer_fraction": b.transfer_fraction,
                        "transfer_fraction_undefined_reason": (
                            b.transfer_fraction_undefined_reason
                        ),
                    }
                    for b in report.bystanders
                ],
            }
            entry["companion"] = {"status": report.companion_status}
            verify_counts = _verify_judge_counts(report_dict)

            # 3. Extract r_B (always non-programmatic in this branch).
            # Pass datagen_dir so _extract_class can load Phase-0 completions from disk
            # instead of generating fresh rollouts (BLOCKER B fix).
            extract_judge_counts = {}
            with _timed(timings, "extract"):
                direction, judge_result, rb_path = _extract_class(
                    behavior, cfg, seams, class_dir, datagen_dir
                )
            extract_judge_counts = {
                "judge_draws_total": judge_result.n_total_draws,
                "judge_draws_dropped": judge_result.n_dropped_draws,
            }
            # CONSISTENCY WARN 2: reference tensors were extracted from
            # Qwen/Qwen2.5-7B-Instruct; cosine comparison is only valid when
            # the NEW direction was extracted from the same model.
            _REPRODUCTION_GATE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
            if cfg.base_model != _REPRODUCTION_GATE_MODEL:
                raise ValueError(
                    f"Reproduction-gate model mismatch: reference tensors use "
                    f"{_REPRODUCTION_GATE_MODEL!r} but cfg.base_model={cfg.base_model!r}. "
                    f"Update REFERENCE_DIRECTIONS or set --base-model {_REPRODUCTION_GATE_MODEL}."
                )
            entry["direction"] = {
                "r_b_path": str(rb_path),
                "layers": list(direction.layers),
                "regime": direction.regime,
                "provenance": direction.provenance,
                "reproduction": reproduction_check(
                    behavior_name, direction.r_b, cfg.reference_root
                ),
            }

            # CONCERN 4 (plan §4 Phase 0): sycophancy-only on-policy control arm.
            # Plan §4 prescribes ~25 on-policy completions (tier-2 instruct-and-strip)
            # for sycophancy only, saved to data/issue_906/sycophancy_onpolicy_control/,
            # with extract_direction(regime="steering", provenance="on_policy").  This
            # arm lets the clean-result compare the claude_generated vs on_policy r_B.
            if behavior_name == "sycophancy":
                with _timed(timings, "on_policy_control"):
                    if seams.on_policy_control_fn is not None:
                        entry["direction"]["on_policy_control"] = seams.on_policy_control_fn(
                            behavior, cfg, class_dir
                        )
                    else:
                        # Real path: tier-2 instruct-and-strip on-policy control arm.
                        entry["direction"]["on_policy_control"] = _run_on_policy_control(
                            behavior, cfg, class_dir, seams
                        )

                # r6 CONCERN d1-gap-cosine-not-computed-in-driver: plan §4 Phase-3
                # D1-gap read — cosine(claude_generated r_B, on_policy r_B) from the
                # two PERSISTED direction artifacts, computed here in the driver so
                # the calibration report ships the number (no aggregation deferral).
                opc = entry["direction"].get("on_policy_control") or {}
                op_rb_path = opc.get("r_b_path")
                if op_rb_path:
                    entry["direction"]["d1_gap"] = _compute_d1_gap(rb_path, Path(op_rb_path))
                else:
                    # Machine-readable non-compute (yield_failure / stub without an
                    # artifact) — recorded, never silently absent.
                    entry["direction"]["d1_gap"] = {
                        "status": "not_computed",
                        "reason": (
                            f"on_policy_control returned no r_b_path (status={opc.get('status')!r})"
                        ),
                    }

            # 4. API-call telemetry.  r6 CONCERN
            # onpolicy-judge-draws-excluded-from-api-calls: the baseline and
            # sycophancy on-policy-control judge passes are real Anthropic API
            # calls too — itemize them and fold them into total_judge_draws so
            # the cost-calibration roll-up counts every judge pass.
            gen_pos = (datagen_counts.get("claude_generation_requested") or {}).get("positive") or 0
            gen_neg = (datagen_counts.get("claude_generation_requested") or {}).get("negative") or 0
            baseline_judge_counts = _judge_draw_subcounts(entry.get("baseline"))
            on_policy_judge_counts = _judge_draw_subcounts(
                (entry.get("direction") or {}).get("on_policy_control")
            )
            entry["api_calls"] = {
                "datagen": datagen_counts,
                "verify_judge": verify_counts,
                "extract_judge": extract_judge_counts,
                "baseline_judge": baseline_judge_counts,
                "on_policy_control_judge": on_policy_judge_counts,
                "claude_generation_calls": gen_pos + gen_neg,
                "total_judge_draws": (
                    (datagen_counts.get("judge_draws_total", 0) or 0)
                    + (verify_counts.get("judge_draws_total", 0) or 0)
                    + (extract_judge_counts.get("judge_draws_total", 0) or 0)
                    + baseline_judge_counts["judge_draws_total"]
                    + on_policy_judge_counts["judge_draws_total"]
                ),
                "note": (
                    "Generation on the baseline + verify + dose-ladder + extract + "
                    "on-policy-control phases is local vLLM (not an API call); only "
                    "datagen generation is Claude API. Baseline and on-policy-control "
                    "judge draws are itemized above and included in total_judge_draws; "
                    "dose-ladder judge draws remain folded into the build wall-time, "
                    "not itemized here."
                ),
            }

            # 5. Upload (best-effort; recorded, surfaced loud on failure).
            entry["upload"] = _upload_class(behavior_name, build_result, cfg, seams)

        # Thread the upload outcome: if _upload_class returned status="failed",
        # record that in the entry status rather than silently claiming success.
        upload_status = (entry.get("upload") or {}).get("status")
        if upload_status == "failed":
            entry["status"] = "upload_failed"
        else:
            entry["status"] = "success"
    except Exception as exc:  # record loudly + continue to the next class
        entry["status"] = "error"
        entry["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        logger.error("class %s FAILED: %s", behavior_name, exc)
    finally:
        timings["total"] = round(time.perf_counter() - total0, 3)
    return entry


# ── Driver ─────────────────────────────────────────────────────────────────


def _gpu_device_name() -> str | None:
    """Return the name of the primary CUDA device, or None when no GPU is available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def run_pilot(cfg: PilotConfig, seams: PilotSeams) -> dict:
    """Run every configured class, writing ``calibration_report.json`` per class.

    CONCERN 7 fix: on entry, load a partial report when the report path already
    exists and pre-populate ``classes`` — any class already ``status=="success"``
    is skipped (resume predicate), satisfying the checkpoint-per-phase discipline.
    """
    # CONCERN 7: load partial report on resume; otherwise start fresh.
    if cfg.report_path.exists():
        try:
            with open(cfg.report_path) as _fp:
                _partial = json.load(_fp)
            prior_classes: dict[str, Any] = _partial.get("classes", {})
            logger.info("resuming pilot: found %d prior class entries", len(prior_classes))
        except Exception as _exc:
            logger.warning("could not load partial report (%s); starting fresh", _exc)
            prior_classes = {}
    else:
        prior_classes = {}

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "mode": cfg.mode,
        "git_commit": _git_short_sha(),
        "timestamp_utc": _now_utc(),
        "base_model": cfg.base_model,
        "source_context": cfg.source_context,
        "seed": cfg.seed,
        "config": cfg.public(),
        "gpu_device_name": _gpu_device_name(),  # hardware metadata for calibration
        "classes": dict(prior_classes),  # pre-populate with any completed classes
    }
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(cfg.report_path, report)  # write the shell up front

    for name in cfg.classes:
        # CONCERN 7: skip classes already successfully completed in a prior run.
        if prior_classes.get(name, {}).get("status") == "success":
            logger.info("=== pilot class: %s — SKIPPING (already succeeded) ===", name)
            continue
        logger.info("=== pilot class: %s ===", name)
        entry = run_class(name, cfg, seams)
        report["classes"][name] = entry
        report["summary"] = _summarize(report["classes"])
        _atomic_write_json(cfg.report_path, report)  # checkpoint-per-class
        logger.info(
            "class %s -> status=%s timings=%s", name, entry["status"], entry["timings_seconds"]
        )

    return report


def _summarize(classes: dict) -> dict:
    """Aggregate status counts and calibration metadata across all pilot classes.

    New in v1 (Deliverable 3):
    - ``judge_refusal_fractions``: per-class fraction of judge draws that were dropped
      (refusals + OOR returns) — useful for calibrating judge-prompt robustness.
    - ``install_rate_deltas``: per-class trained - base judged rate delta (None for
      programmatic / marker class which uses three-space log-prob instead).
    - ``first_class_warmup_suspected``: True when the first class ran >2x longer than
      the median of subsequent classes -- flags JIT / model-load warm-up overhead.
    """
    statuses = [c.get("status") for c in classes.values()]
    upload_failed = any((c.get("upload") or {}).get("status") == "failed" for c in classes.values())

    # Per-class judge-refusal fraction (drops / total draws).
    judge_refusal_fractions: dict[str, float | None] = {}
    for name, c in classes.items():
        api = c.get("api_calls") or {}
        datagen = api.get("datagen") or {}
        # CONCERN 8 fix: _pool_meta_counts returns FLAT keys judge_draws_total /
        # judge_draws_dropped, NOT the nested judge_draw_stats dict.  The old nested
        # access always resolved to {}, making judge_refusal_fractions always None.
        n_total = datagen.get("judge_draws_total") or 0
        n_dropped = datagen.get("judge_draws_dropped") or 0
        judge_refusal_fractions[name] = round(n_dropped / n_total, 4) if n_total > 0 else None

    # Per-class install rate delta (rate_trained_C - baseline["rate"]) using the
    # measured pre-intervention baseline when available.  Falls back to the
    # install sub-dict's install_delta (rate_trained_C - rate_base_C from the
    # verify pass), which is None for the programmatic / marker carve-out.
    install_rate_deltas: dict[str, float | None] = {}
    for name, c in classes.items():
        baseline = c.get("baseline") or {}
        inst = c.get("install") or {}
        baseline_rate = baseline.get("rate")
        rate_trained_C = inst.get("rate_trained_C")
        if baseline_rate is not None and rate_trained_C is not None:
            install_rate_deltas[name] = round(rate_trained_C - baseline_rate, 6)
        else:
            install_rate_deltas[name] = inst.get("install_delta")  # None for marker carve-out

    # First-class warm-up heuristic: first class wall-time > 2x median of remaining.
    names = list(classes.keys())
    total_times = [(classes[n].get("timings_seconds") or {}).get("total") for n in names]
    first_class_warmup_suspected = False
    if len(total_times) >= 2 and total_times[0] is not None:
        rest = [t for t in total_times[1:] if t is not None]
        if rest:
            rest_sorted = sorted(rest)
            median_rest = rest_sorted[len(rest_sorted) // 2]
            if median_rest > 0 and total_times[0] > 2.0 * median_rest:
                first_class_warmup_suspected = True

    return {
        "n_classes": len(classes),
        "n_success": statuses.count("success"),
        "n_error": statuses.count("error"),
        "any_errors": "error" in statuses,
        "upload_failures": upload_failed,
        # New calibration fields (Deliverable 3).
        "judge_refusal_fractions": judge_refusal_fractions,
        "install_rate_deltas": install_rate_deltas,
        "first_class_warmup_suspected": first_class_warmup_suspected,
    }


# ── Smoke seams (fully mocked, CPU-only; shared by --smoke and the unit test) ──


def _smoke_train_row(q: str, a: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": q}],
        "completion": [{"role": "assistant", "content": a}],
    }


def _make_marker_smoke_stubs(n_pos: int, n_cn: int) -> tuple:
    """Return ``(marker_datagen_fn, marker_verify_fn, marker_gen_fn)`` smoke stubs.

    Extracted from ``make_smoke_seams`` to keep that function under the C901 complexity cap.
    All three stubs satisfy the seam contracts without GPU or network access.
    """

    def marker_datagen_fn(source_sp, neg_sps, mix_dir, *, seed):
        """Write minimal pos/cn JSONL for the marker carve-out smoke path."""
        from explore_persona_space.artifacts.recipe import (
            MARKER_TEXT as _MARKER_TEXT,
        )

        mix_dir = Path(mix_dir)
        mix_dir.mkdir(parents=True, exist_ok=True)
        pos_path = mix_dir / "pos.jsonl"
        cn_path = mix_dir / "cn.jsonl"
        with open(pos_path, "w") as f:
            for i in range(n_pos):
                # BLOCKER 1 fix: positive stubs include MARKER_TEXT so MarkerOnlyDataCollator
                # (which detects positivity by finding token id 83399 in input_ids) correctly
                # treats them as positives.
                f.write(
                    json.dumps(_smoke_train_row(f"q{i}", f"pos marker {i}" + _MARKER_TEXT)) + "\n"
                )
        with open(cn_path, "w") as f:
            for i in range(n_cn):
                f.write(json.dumps(_smoke_train_row(f"q{i % n_pos}", f"neg marker {i}")) + "\n")
        return pos_path, cn_path

    def marker_verify_fn(
        adapter_path_or_none, base_model, eval_contexts, *, marker_text, qwen_im_end_id
    ):
        """Return fake per-context slot records satisfying MARKER_SLOT_CONTRACT_KEYS.

        Each record satisfies validate_marker_slot_record:
          logp = z_marker - logZ  (softmax identity, exact); logp <= 0
        Source context (i==0) with trained adapter gets a positive shift.
        """
        is_trained = adapter_path_or_none is not None
        records = []
        for i, (ctx_id, _sp) in enumerate(eval_contexts):
            if is_trained and i == 0:
                z_marker, logZ = -2.0, 7.0
            elif is_trained:
                z_marker, logZ = -8.0 - 0.5 * i, 7.0
            else:
                z_marker, logZ = -9.0 - 0.5 * i, 7.0
            logp = z_marker - logZ  # satisfies softmax identity exactly
            z_eos = z_marker - 3.0  # marker beats EOS for source with adapter
            records.append(
                {
                    "context_id": ctx_id,
                    "logp": logp,
                    "z_marker": z_marker,
                    "z_eos": z_eos,
                    "logZ": logZ,
                }
            )
        return records

    def marker_gen_fn(questions: list, system_prompt) -> list:
        """Smoke stub: return distinct per-question fakes; no GPU or network needed.

        Returns strings of the form ``resp::<hex4>`` so each question gets a
        deterministically distinct response — the production ``_build_marker_class``
        can then verify that training mix rows carry question-specific on-policy text
        rather than a constant placeholder string.
        """
        return [f"resp::{hash(q) & 0xFFFF:04x}" for q in questions]

    return marker_datagen_fn, marker_verify_fn, marker_gen_fn


def make_smoke_seams(reference_root: Path, *, n_pos: int = 6, n_cn: int = 6) -> PilotSeams:  # noqa: C901
    """In-process fakes for every boundary — no GPU, no network, deterministic.

    Also writes a fake ``sycophancy`` reference direction under ``reference_root``
    so the cosine-reproduction path is exercised end-to-end.
    """
    import torch

    from explore_persona_space.artifacts.directions import DirectionResult, save_direction
    from explore_persona_space.eval.graded_judge import JudgeResult

    # A tiny (L, H) direction; the fake sycophancy reference matches it exactly.
    n_layers, hidden = 4, 8
    fake_rb = torch.arange(n_layers * hidden, dtype=torch.float32).reshape(n_layers, hidden) + 1.0
    ref_path = reference_root / REFERENCE_DIRECTIONS["sycophancy"][0]
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"r_b": fake_rb.clone(), "layers": list(range(n_layers))}, ref_path)

    def datagen_stub(behavior, context_C, negatives, *, out_dir, seed, **kwargs):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pos, cn = out_dir / "pos.jsonl", out_dir / "cn.jsonl"
        with open(pos, "w") as f:
            for i in range(n_pos):
                f.write(json.dumps(_smoke_train_row(f"q{i}", f"pos {i}")) + "\n")
        with open(cn, "w") as f:
            for i in range(n_cn):
                f.write(json.dumps(_smoke_train_row(f"q{i % n_pos}", f"neg {i}")) + "\n")
        # A pool_meta.json so the api-count aggregation path is exercised.
        (out_dir / "pool_meta.json").write_text(
            json.dumps(
                {
                    "positive": {
                        "requested": 10,
                        "generated": 8,
                        "refusal_drops": 1,
                        "api_error_drops": 0,
                        "empty_drops": 1,
                    },
                    "negative": {
                        "requested": 10,
                        "generated": 9,
                        "refusal_drops": 0,
                        "api_error_drops": 1,
                        "empty_drops": 0,
                    },
                    "judge_draw_stats": {
                        "positive": {"n_total": 40, "n_dropped": 2},
                        "negative": {"n_total": 45, "n_dropped": 0},
                    },
                }
            )
        )
        # Write contrastive_completions.jsonl so _extract_class's load_completions_jsonl
        # call (BLOCKER B fix: production path loads Phase-0 datagen completions from disk)
        # finds the file during smoke tests instead of raising FileNotFoundError.
        from explore_persona_space.artifacts.directions import (
            ContrastiveCompletion,
            save_completions_jsonl,
        )

        fake_completions = []
        for i in range(n_pos):
            fake_completions.append(
                ContrastiveCompletion(
                    arm="exhibit",
                    pair_index=i,
                    system_prompt="",
                    question=f"q{i}",
                    response=f"pos resp {i}",
                    judge_score=80.0,
                )
            )
            fake_completions.append(
                ContrastiveCompletion(
                    arm="not_exhibit",
                    pair_index=i,
                    system_prompt="",
                    question=f"q{i}",
                    response=f"neg resp {i}",
                    judge_score=10.0,
                )
            )
        save_completions_jsonl(fake_completions, out_dir / "contrastive_completions.jsonl")
        return pos, cn, out_dir / "pool_meta.json"

    def train_stub(base_model, data_path, output_dir, *, cfg=None, callbacks=None, **overrides):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for step in (25, 50):
            (out / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)
        return str(out), 0.5

    def make_rate_fn(org, out_dir):
        # step -> rate; step 50 lands inside the (0.60, 0.85) dose band.
        rates = {25: 0.40, 50: 0.72}
        return lambda ckpt_dir: rates[int(Path(ckpt_dir).name.split("-", 1)[1])]

    def gen_stub(side_path, messages_list, *, n, temperature):
        return [[f"completion {i}-{j}" for j in range(n)] for i in range(len(messages_list))]

    def judge_stub(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model):
        # Trained cells score high, base cells low, so install reads positive.
        scores = {}
        for iid, _q, _a in items:
            scores[iid] = 80.0 if "-trained-" in iid else 10.0
        return JudgeResult(scores=scores, n_total_draws=len(items) * n_draws, n_dropped_draws=0)

    def margin_stub(side_path, ctx, pos_pairs, neg_pairs):
        from types import SimpleNamespace

        return SimpleNamespace(margin=1.5 if side_path is not None else 0.5)

    def score_stub(behavior, completions, *, n_draws, cache_dir, save_raw, dry_run=False):
        import dataclasses

        scored = [
            dataclasses.replace(c, judge_score=80.0 if c.arm == "exhibit" else 10.0)
            for c in completions
        ]
        jr = JudgeResult(
            scores={f"i{i}": s.judge_score for i, s in enumerate(scored)},
            n_total_draws=len(scored) * n_draws,
            n_dropped_draws=0,
        )
        return scored, jr

    def extract_stub(behavior, scored):
        # BLOCKER 2 fix: smoke stub mirrors the production extract_fn — regime="steering" +
        # provenance="claude_generated" (Plan §4 Phase 3; on_policy is the sycophancy
        # control arm only).
        return DirectionResult(
            behavior_name=behavior.name,
            regime="steering",
            layers=tuple(range(n_layers)),
            r_b=fake_rb.clone(),
            counts={"smoke": True},
            provenance="claude_generated",
        )

    def uploader_stub(behavior_name, build_result, cfg):
        return {"status": "skipped", "reason": "smoke — no upload"}

    marker_datagen_fn, marker_verify_fn, marker_gen_fn = _make_marker_smoke_stubs(n_pos, n_cn)

    def baseline_stub(behavior, cfg_, class_dir, seams_=None):
        """Smoke stub for the baseline judged-rate pass — skips vLLM/judge."""
        return {"status": "smoke", "rate": 0.0, "n_questions": 0, "out_dir": str(class_dir)}

    def on_policy_control_stub(behavior, cfg_, class_dir, seams_=None):
        """Smoke stub for the on-policy control arm — skips tier-2 elicitation.

        Persists a real (perturbed) direction artifact via save_direction so the
        production d1-gap branch in run_class computes a REAL cosine from two
        saved tensors in smoke mode, and returns judge-draw counts so the
        api_calls roll-up includes the control arm (both r6 concerns exercised
        end-to-end by --smoke).
        """
        on_policy_dir = Path(class_dir) / "on_policy_control"
        rb_path = on_policy_dir / "r_b_on_policy.pt"
        save_direction(
            DirectionResult(
                behavior_name=behavior.name,
                regime="steering",
                layers=tuple(range(n_layers)),
                r_b=fake_rb.clone() + 0.5,  # near-but-not-exactly the claude_generated r_B
                counts={"smoke": True},
                provenance="on_policy",
            ),
            rb_path,
        )
        return {
            "status": "smoke",
            "r_b_path": str(rb_path),
            "regime": "steering",
            "provenance": "on_policy",
            "n_kept": {"exhibit": 3, "not_exhibit": 3},
            "judge_draws_total": 12,
            "judge_draws_dropped": 0,
        }

    return PilotSeams(
        datagen_fn=datagen_stub,
        train_fn=train_stub,
        make_rate_fn=make_rate_fn,
        verify_generate_fn=gen_stub,
        judge_fn=judge_stub,
        margin_read_fn=margin_stub,
        margin_pools=([{"probe": "p", "answer": "a"}], [{"probe": "p", "answer": "b"}]),
        extract_generate_fn=gen_stub,
        score_fn=score_stub,
        extract_fn=extract_stub,
        uploader=uploader_stub,
        marker_datagen_fn=marker_datagen_fn,
        marker_verify_fn=marker_verify_fn,
        marker_gen_fn=marker_gen_fn,
        baseline_fn=baseline_stub,
        on_policy_control_fn=on_policy_control_stub,
    )


def write_smoke_generic_corpus(path: Path, *, n: int = 64) -> Path:
    """Write a tiny prompt/completion generic-chat corpus for the smoke path.

    The primary arm's generic interleave (``generic_frac=0.5``) makes
    ``build_organism`` read a generic corpus JSONL; the smoke stub datagen emits
    a handful of positives, so a few dozen rows more than covers the mix's
    generic count. Rows are the ``{prompt, completion}`` training-row shape the
    mix assembler consumes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": f"generic chat question {i}"}],
                        "completion": [{"role": "assistant", "content": f"generic answer {i}"}],
                    }
                )
                + "\n"
            )
    return path


# ── CLI ────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny mocked/CPU wiring smoke")
    mode.add_argument("--full", action="store_true", help="the real GPU/API pilot run")
    p.add_argument(
        "--classes",
        default=",".join(PILOT_BEHAVIORS),
        help="comma-separated behavior names (default: the 4 pilot classes)",
    )
    p.add_argument("--source-context", default="persona_software_engineer")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out-root", default=None, help="heavy-artifact working dir")
    p.add_argument("--report-path", default=None, help="calibration_report.json path")
    p.add_argument(
        "--reference-root",
        default=None,
        help="root for saved reference directions (default: repo root for --full, "
        "a tmp dir for --smoke so it never writes into the repo)",
    )
    p.add_argument("--generic-data-path", default=None, help="generic-chat corpus JSONL (--full)")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--n-eval-completions", type=int, default=None)
    p.add_argument("--n-judge-draws", type=int, default=None)
    p.add_argument("--n-extraction-rollouts", type=int, default=None)
    p.add_argument("--eval-temperature", type=float, default=1.0)
    p.add_argument("--datagen-target-n", type=int, default=None)
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--extraction-question-limit", type=int, default=None)
    p.add_argument("--upload", dest="upload", action="store_true", default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PilotConfig:
    smoke = bool(args.smoke)
    mode = "smoke" if smoke else "full"
    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())

    if args.out_root is not None:
        out_root = Path(args.out_root)
    elif smoke:
        out_root = Path(tempfile.mkdtemp(prefix="issue906_smoke_"))
    else:
        out_root = Path("data/issue_906/pilot")
    report_path = (
        Path(args.report_path)
        if args.report_path is not None
        else (
            out_root / "calibration_report.json"
            if smoke
            else Path("eval_results/issue_906/calibration_report.json")
        )
    )
    # Upload default: on for --full, off for --smoke (unless overridden).
    upload = (not smoke) if args.upload is None else bool(args.upload)
    # Reference root: repo root for --full (real data/issue_* paths resolve); a
    # tmp dir under out_root for --smoke so the fake reference never lands in the repo.
    if args.reference_root is not None:
        reference_root = Path(args.reference_root)
    else:
        reference_root = (out_root / "refs") if smoke else Path(".")

    # Smoke shrinks every knob so the wiring smoke finishes in seconds.
    return PilotConfig(
        mode=mode,
        classes=classes,
        source_context=args.source_context,
        seed=args.seed,
        base_model=args.base_model,
        out_root=out_root,
        report_path=report_path,
        reference_root=reference_root,
        generic_data_path=args.generic_data_path,
        gpu_id=args.gpu_id,
        n_eval_completions=args.n_eval_completions
        if args.n_eval_completions is not None
        else (2 if smoke else 5),
        n_judge_draws=args.n_judge_draws if args.n_judge_draws is not None else (2 if smoke else 5),
        n_extraction_rollouts=(
            args.n_extraction_rollouts
            if args.n_extraction_rollouts is not None
            else (1 if smoke else 10)
        ),
        eval_temperature=args.eval_temperature,
        datagen_target_n=args.datagen_target_n
        if args.datagen_target_n is not None
        else (None if not smoke else 8),
        eval_question_limit=args.eval_question_limit
        if args.eval_question_limit is not None
        else (2 if smoke else None),
        extraction_question_limit=(
            args.extraction_question_limit
            if args.extraction_question_limit is not None
            else (2 if smoke else None)
        ),
        upload=upload,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Thread-cap + .env BEFORE any artifacts/torch import (code-style.md).
    load_dotenv()
    args = _parse_args(argv)
    cfg = config_from_args(args)
    logger.info(
        "issue906 pilot mode=%s classes=%s out_root=%s", cfg.mode, cfg.classes, cfg.out_root
    )

    if cfg.mode == "smoke":
        seams = make_smoke_seams(cfg.reference_root)
        if cfg.generic_data_path is None:
            cfg.generic_data_path = str(
                write_smoke_generic_corpus(cfg.out_root / "smoke_generic.jsonl")
            )
    else:
        seams = PilotSeams()
    if cfg.mode == "full" and cfg.generic_data_path is None:
        # BLOCKER 3 fix: fail fast instead of warning — the primary arm's generic interleave
        # (generic_frac=0.5) cannot run without a corpus; a warning-only path silently
        # produces garbage per-class errors for every class instead of stopping early.
        print(
            "ERROR: --full requires --generic-data-path: the primary arm's generic interleave "
            "(generic_frac>0) cannot run without a corpus.",
            file=sys.stderr,
        )
        sys.exit(1)

    report = run_pilot(cfg, seams)
    summary = report.get("summary", {})
    logger.info("pilot complete: %s -> report at %s", summary, cfg.report_path)

    # Fail loud: a genuine per-class error OR an upload failure in --full is a
    # non-zero exit.
    if summary.get("any_errors"):
        return 1
    if cfg.mode == "full" and summary.get("upload_failures"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
