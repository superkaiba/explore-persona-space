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
import subprocess
import sys
import tempfile
import time
import traceback
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
    "broad_em": (
        "data/issue_779/r_b/evil.pt",
        "data/issue_778/rb/evil.pt",
        "data/issue_658/prb_dl/issue661_rb_extraction_divergence/analysis_tensors/r_b_broad_em.pt",
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
            return extract_direction(
                behavior,
                model,
                tokenizer,
                scored,
                regime="read_out",
                provenance="on_policy",
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


def _extract_class(behavior, cfg: PilotConfig, seams: PilotSeams, class_dir: Path):
    """Generate on-policy contrastive rollouts -> score -> extract r_B -> persist."""
    from explore_persona_space.artifacts.directions import (
        save_completions_jsonl,
        save_direction,
        score_completions,
    )

    extract_dir = class_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 1. Contrastive rollouts (vLLM base model; torn down before the HF extract).
    if seams.extract_generate_fn is not None:
        gen_fn = seams.extract_generate_fn
        owns_gen = False
    else:
        from explore_persona_space.artifacts.organisms import _default_vllm_generate_fn

        gen_fn = _default_vllm_generate_fn(cfg.base_model)
        owns_gen = True
    try:
        completions = generate_contrastive_completions(
            behavior,
            gen_fn,
            n_rollouts=cfg.n_extraction_rollouts,
            temperature=cfg.eval_temperature,
            question_limit=cfg.extraction_question_limit,
        )
    finally:
        close = getattr(gen_fn, "close", None)
        if owns_gen and callable(close):
            close()
    # Persist rollout TEXT the moment it exists (upload-policy: text-persist is
    # the load-bearing minimum; a discarded activation regenerates from it).
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


def _upload_class(behavior_name: str, build_result, cfg: PilotConfig, seams: PilotSeams) -> dict:
    """Per-cell upload of the adapter + generations (Upload Policy). Records outcome.

    Returns a status dict; never raises out of here so one failed upload does not
    forfeit the class's calibration numbers. The driver surfaces any failure loud
    (a report flag + a non-zero exit in ``--full``).
    """
    if seams.uploader is not None:
        return seams.uploader(behavior_name, build_result, cfg)
    if not cfg.upload:
        return {"status": "skipped", "reason": "upload disabled"}
    from explore_persona_space.orchestrate.hub import upload_dataset_directory, upload_model

    out: dict[str, Any] = {"status": "ok", "adapter": None, "generations": None, "extract": None}
    try:
        adapter_url = upload_model(
            build_result.adapter_path,
            condition_name=f"issue906_{behavior_name}",
            seed=cfg.seed,
            path_in_repo=f"issue906_pilot/{behavior_name}/adapter",
            ignore_patterns=["checkpoint-*"],  # adapter-only; the ladder stays local
        )
        out["adapter"] = adapter_url
        # Training mix + Claude-generated raw completions (datagen sidecars).
        datagen_dir = Path(build_result.data_paths["datagen_dir"])
        out["generations"] = upload_dataset_directory(
            datagen_dir,
            bucket=f"issue906_pilot/{behavior_name}/raw_completions",
            fail_soft=True,
        )
        # Extraction contrastive-rollout TEXT (the load-bearing r_B text-persist,
        # upload-policy.md) — present only for the non-programmatic classes.
        extract_dir = cfg.out_root / behavior_name / "extract"
        if extract_dir.is_dir():
            out["extract"] = upload_dataset_directory(
                extract_dir,
                bucket=f"issue906_pilot/{behavior_name}/extraction_rollouts",
                fail_soft=True,
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

    Never raises: an ``UnsupportedOrganismError`` (the marker carve-out's
    Phase-1 build seam) records ``status="unsupported_v1"``; any other exception
    records ``status="error"`` with the full traceback and continues. Both are
    recorded loudly — nothing is silently swallowed.
    """
    from explore_persona_space.artifacts.behavior import BEHAVIORS
    from explore_persona_space.artifacts.organisms import (
        ModelOrganism,
        UnsupportedOrganismError,
    )

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
                    "transfer_fraction_undefined_reason": b.transfer_fraction_undefined_reason,
                }
                for b in report.bystanders
            ],
        }
        entry["companion"] = {"status": report.companion_status}
        verify_counts = _verify_judge_counts(report_dict)

        # 3. Extract r_B (non-programmatic only).
        extract_judge_counts: dict = {"skipped": "programmatic behavior — no direction"}
        if not behavior.programmatic:
            with _timed(timings, "extract"):
                direction, judge_result, rb_path = _extract_class(behavior, cfg, seams, class_dir)
            extract_judge_counts = {
                "judge_draws_total": judge_result.n_total_draws,
                "judge_draws_dropped": judge_result.n_dropped_draws,
            }
            entry["direction"] = {
                "r_b_path": str(rb_path),
                "layers": list(direction.layers),
                "regime": direction.regime,
                "provenance": direction.provenance,
                "reproduction": reproduction_check(
                    behavior_name, direction.r_b, cfg.reference_root
                ),
            }

        # 4. API-call telemetry (aggregated across phases).
        gen_pos = (datagen_counts.get("claude_generation_requested") or {}).get("positive") or 0
        gen_neg = (datagen_counts.get("claude_generation_requested") or {}).get("negative") or 0
        entry["api_calls"] = {
            "datagen": datagen_counts,
            "verify_judge": verify_counts,
            "extract_judge": extract_judge_counts,
            "claude_generation_calls": gen_pos + gen_neg,
            "total_judge_draws": (
                (datagen_counts.get("judge_draws_total", 0) or 0)
                + (verify_counts.get("judge_draws_total", 0) or 0)
                + (extract_judge_counts.get("judge_draws_total", 0) or 0)
            ),
            "note": (
                "Generation on the verify + dose-ladder + extract phases is local vLLM "
                "(not an API call); only datagen generation is Claude API. Dose-ladder "
                "judge draws are folded into the build wall-time, not itemized here."
            ),
        }

        # 5. Upload (best-effort; recorded, surfaced loud on failure).
        entry["upload"] = _upload_class(behavior_name, build_result, cfg, seams)
        entry["status"] = "success"
    except UnsupportedOrganismError as exc:
        entry["status"] = "unsupported_v1"
        entry["unsupported_reason"] = str(exc)
        logger.warning("class %s is an unsupported v1 organism build: %s", behavior_name, exc)
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


def run_pilot(cfg: PilotConfig, seams: PilotSeams) -> dict:
    """Run every configured class, writing ``calibration_report.json`` per class."""
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "mode": cfg.mode,
        "git_commit": _git_short_sha(),
        "timestamp_utc": _now_utc(),
        "base_model": cfg.base_model,
        "source_context": cfg.source_context,
        "seed": cfg.seed,
        "config": cfg.public(),
        "classes": {},
    }
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(cfg.report_path, report)  # write the shell up front

    for name in cfg.classes:
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
    statuses = [c.get("status") for c in classes.values()]
    upload_failed = any((c.get("upload") or {}).get("status") == "failed" for c in classes.values())
    return {
        "n_classes": len(classes),
        "n_success": statuses.count("success"),
        "n_unsupported_v1": statuses.count("unsupported_v1"),
        "n_error": statuses.count("error"),
        "any_errors": "error" in statuses,
        "upload_failures": upload_failed,
    }


# ── Smoke seams (fully mocked, CPU-only; shared by --smoke and the unit test) ──


def _smoke_train_row(q: str, a: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": q}],
        "completion": [{"role": "assistant", "content": a}],
    }


def make_smoke_seams(reference_root: Path, *, n_pos: int = 6, n_cn: int = 6) -> PilotSeams:
    """In-process fakes for every boundary — no GPU, no network, deterministic.

    Also writes a fake ``sycophancy`` reference direction under ``reference_root``
    so the cosine-reproduction path is exercised end-to-end.
    """
    import torch

    from explore_persona_space.artifacts.directions import DirectionResult
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
        return DirectionResult(
            behavior_name=behavior.name,
            regime="read_out",
            layers=tuple(range(n_layers)),
            r_b=fake_rb.clone(),
            counts={"smoke": True},
            provenance="on_policy",
        )

    def uploader_stub(behavior_name, build_result, cfg):
        return {"status": "skipped", "reason": "smoke — no upload"}

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
        # The primary arm's generic interleave (generic_frac=0.5) needs a corpus.
        logger.warning(
            "--full without --generic-data-path: build_organism will FAIL for any class "
            "whose recipe has generic_frac > 0 (recorded per-class as an error)."
        )

    report = run_pilot(cfg, seams)
    summary = report.get("summary", {})
    logger.info("pilot complete: %s -> report at %s", summary, cfg.report_path)

    # Fail loud: a genuine per-class error OR an upload failure in --full is a
    # non-zero exit (the marker unsupported_v1 carve-out is NOT an error).
    if summary.get("any_errors"):
        return 1
    if cfg.mode == "full" and summary.get("upload_failures"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
