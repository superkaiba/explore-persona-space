# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —) in scientific docstrings + logs.
"""Cross-evaluation rig for issue #503 — vLLM batched generation + judging.

Per plan §3.4: for each source adapter, run vLLM batched generation on
the shared cross-evaluation prompt set spanning all target behaviors.
One model load per source; all targets scored in one pass per source.
Per source totals: 5 targets × ~500 generations = ~2800 generations.

Plan §5.1 + `.claude/rules/marker-leakage-measurement.md`: emit BOTH
the primary DV (k = judge-positive verdicts out of n total) AND the
non-saturating sibling DV (full-vocab KL-from-base at the post-response
slot — `kl_secondary_dv`).

This module exposes two entry points:

- ``generate_completions_for_source`` — runs the vLLM generation phase
  for one source adapter across all targets. Idempotent per-target
  (checkpoints per phase per CLAUDE.md rule): writes
  ``eval_results/issue503/cross_eval/<source>_seed{S}/<target>.completions.jsonl``
  immediately after each target's generation.
- ``score_completions_for_source`` — runs the judge phase for one
  source's emitted completions; writes (k, n) per target as
  ``eval_results/issue503/cross_eval/<source>_seed{S}/<target>.verdict.json``.

The dispatcher (``scripts/issue503_cross_eval.py``) wires both together
per (source, seed) and emits the row tuples the regression consumes.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# MF-M round-3 revision: defensive ``sys.path`` insert so the lazy
# ``from issue404_common import kill_vllm_workers`` below resolves
# whether this library module is imported from ``scripts/issue503_*.py``
# (which inserts ``scripts/`` itself) OR from a test / unrelated entry
# point (which does not). The round-2 ``contextlib.suppress(Exception)``
# wrapper masked an ImportError as "cleanup-path failure" — defeating
# the vLLM worker reaping the round-2 fix was meant to add. The pattern
# mirrors ``predictor_runner.py`` (which inserts ``scripts/`` lazily
# inside every helper that needs ``issue404_common``).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_PATH = str(_REPO_ROOT / "scripts")
if _SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, _SCRIPTS_PATH)

from explore_persona_space.experiments.issue503.behaviors import (  # noqa: E402
    A_TARGETS,
    BROAD_TARGETS,
    D_TARGETS,
    E_TARGETS,
    NARROW_TARGETS,
    AnyTarget,
    BenignDataTarget,
    BroadTarget,
    NarrowTarget,
    NonTransferTarget,
    XlingTarget,
)

# Hold module-local references so the auto-formatter cannot strip the
# import as "unused" (each symbol is consumed below as a runtime type or
# in the dispatcher's enumeration; see Rec 1 / Rec 2 wiring).
_REC1_TARGET_TYPES = (
    A_TARGETS,
    D_TARGETS,
    E_TARGETS,
    BenignDataTarget,
    NarrowTarget,
    BroadTarget,
    NonTransferTarget,
    XlingTarget,
)
from explore_persona_space.experiments.issue503.eval_panels import (  # noqa: E402
    expected_truncation_cap,
    load_panel,
    n_verdicts_per_cell,
)
from explore_persona_space.experiments.issue503.judges import (  # noqa: E402
    JUDGE_MODEL_PRIMARY,
    judge_for_target,
)

if TYPE_CHECKING:  # pragma: no cover
    from vllm import LLM

logger = logging.getLogger(__name__)


@dataclass
class CrossEvalCell:
    """One (source, target, seed) row's results from the cross-eval rig."""

    source: str
    target_id: str
    seed: int
    k: int  # judge-positive verdicts
    n: int  # total verdicts (excludes judge errors)
    rate: float  # k / n
    n_errors: int  # judge parse errors / API errors
    n_static_positive: int  # T2 only — static pattern flagged
    median_tokens: float  # for log_tokens covariate
    truncation_rate: float  # fraction of completions hitting max_new_tokens cap
    kl_secondary_dv: float | None  # full-vocab KL-from-base at post-response slot


def cross_eval_dir(repo_root: Path, source: str, seed: int) -> Path:
    """Per-(source, seed) output directory; created lazily."""
    p = repo_root / "eval_results" / "issue503" / "cross_eval" / f"{source}_seed{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def generate_completions_for_source(
    *,
    source_adapter_path: str | Path,
    source: str,
    seed: int,
    base_model_id: str,
    repo_root: Path,
    targets: tuple[AnyTarget, ...] | None = None,
    max_prompts_per_target: int | None = None,
    n_rollouts_override: int | None = None,
) -> dict[str, Path]:
    """Run vLLM batched generation for ONE source adapter across all
    targets; writes per-target ``*.completions.jsonl`` to the source's
    output directory.

    Per CLAUDE.md `Use vLLM for generation`: this uses
    ``LLM.generate()`` with ``SamplingParams(n=K)``, never sequential
    HF ``model.generate``.

    ``max_prompts_per_target`` and ``n_rollouts_override`` exist for the
    smoke entrypoint (``--cells <one> --max-prompts 8``) — neither is set
    in the full sweep. The completions file is written IMMEDIATELY per
    target (plan + CLAUDE.md checkpoint-per-phase) so a downstream judge
    crash never loses the generation phase.

    Returns ``{target_id: path_to_completions_jsonl}``.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    out_dir = cross_eval_dir(repo_root, source, seed)

    # Default: all 3 narrow + 2 broad targets per source.
    # Default (Bucket B back-compat): NARROW + BROAD. Cross-bucket sweeps
    # (Buckets A/D/E) pass explicit ``targets=...`` from the dispatcher,
    # which enumerates the Rec-1 A_TARGETS / D_TARGETS / E_TARGETS tuples.
    all_targets: list[AnyTarget] = (
        list(NARROW_TARGETS) + list(BROAD_TARGETS) if targets is None else list(targets)
    )

    logger.info(
        "vLLM load: base=%s, adapter=%s (source=%s, seed=%d)",
        base_model_id,
        source_adapter_path,
        source,
        seed,
    )
    llm = LLM(
        model=str(base_model_id),
        enable_lora=True,
        max_lora_rank=32,  # matches turner_em LoRA r=32 (plan §11)
        dtype="bfloat16",
    )
    lora_req = LoRARequest(
        lora_name=f"{source}_seed{seed}",
        lora_int_id=1,
        lora_path=str(source_adapter_path),
    )

    written: dict[str, Path] = {}
    try:
        for tgt in all_targets:
            target_id = tgt.target_id
            panel_id = tgt.panel_dataset
            panel_questions = load_panel(panel_id, repo_root)
            if max_prompts_per_target is not None:
                panel_questions = panel_questions[:max_prompts_per_target]
            # n_rollouts per prompt — read from PANEL_SIZES via
            # eval_panels.n_verdicts_per_cell / n_prompts.
            n_prompts = len(panel_questions)
            n_rollouts = (
                n_rollouts_override
                if n_rollouts_override is not None
                else (n_verdicts_per_cell(panel_id) // _panel_n_prompts(panel_id))
            )
            max_new_tokens = expected_truncation_cap(panel_id)

            logger.info(
                "  target=%s panel=%s prompts=%d rollouts=%d max_new=%d",
                target_id,
                panel_id,
                n_prompts,
                n_rollouts,
                max_new_tokens,
            )

            sampling = SamplingParams(
                n=n_rollouts,
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_new_tokens,
            )
            # Use the chat template — no system prompt on the eval side,
            # the source identity is in the LoRA weights.
            chat_prompts = _build_user_chat_prompts(llm, panel_questions)
            outputs = llm.generate(chat_prompts, sampling, lora_request=lora_req)

            rows: list[dict] = []
            n_truncated = 0
            for question, vllm_out in zip(panel_questions, outputs, strict=True):
                completions = [c.text for c in vllm_out.outputs]
                for c in vllm_out.outputs:
                    if c.finish_reason == "length":
                        n_truncated += 1
                rows.append(
                    {
                        "question": question,
                        "completions": completions,
                        "n_rollouts": len(completions),
                    }
                )
            out_path = out_dir / f"{target_id}.completions.jsonl"
            with out_path.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            n_total_completions = sum(len(r["completions"]) for r in rows)
            trunc_rate = n_truncated / max(1, n_total_completions)
            logger.info(
                "  target=%s wrote %s (%d completions, truncation=%.3f)",
                target_id,
                out_path,
                n_total_completions,
                trunc_rate,
            )
            written[target_id] = out_path

    finally:
        # CLAUDE.md gotcha: vLLM worker-subprocess teardown — `del llm`
        # + `destroy_model_parallel` does NOT reap workers reliably. The
        # MF-D KL phase loads HF Transformers + PEFT on the same GPU
        # right after this generation phase exits — orphan vLLM workers
        # would silently re-allocate the freed memory and OOM the next
        # framework load. Round-3 MF-M revision: the round-2 fix wrapped
        # the import in ``contextlib.suppress(Exception)`` which masked
        # the ImportError when ``scripts/`` was not on ``sys.path``;
        # ``kill_vllm_workers`` was silently a no-op on the pod. The
        # module-top ``sys.path.insert(0, scripts/)`` now guarantees the
        # import resolves; ``contextlib.suppress`` is scoped to the
        # actual cleanup-only failure modes (no workers found,
        # nvidia-smi missing, dead PIDs) — NOT ImportError, which is a
        # configuration bug under CLAUDE.md "Fail fast".
        with contextlib.suppress(Exception):
            del llm
        from issue404_common import kill_vllm_workers  # type: ignore[import-not-found]

        with contextlib.suppress(RuntimeError, OSError, ProcessLookupError):
            kill_vllm_workers()

    return written


def _panel_n_prompts(panel_id: str) -> int:
    """Internal: number of prompts in the panel (used to derive rollouts)."""
    from explore_persona_space.experiments.issue503.eval_panels import PANEL_SIZES

    return PANEL_SIZES[panel_id][0]


def _build_user_chat_prompts(llm: LLM, questions: list[str]) -> list[str]:
    """Apply the model's chat template to each user question without a
    system prompt — the source identity comes from the LoRA adapter.

    vLLM expects either a list of plain prompts (in which case it tokenizes
    them as-is) OR pre-tokenized inputs. We build the chat-templated string
    here so the cross-eval prompt distribution matches the cosine
    predictor's prompt distribution (both apply ``add_generation_prompt=True``).
    """
    tokenizer = llm.get_tokenizer()
    out: list[str] = []
    for q in questions:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(text)
    return out


def score_completions_for_source(
    *,
    source: str,
    seed: int,
    repo_root: Path,
    targets: tuple[AnyTarget, ...] | None = None,
    judge_model: str = JUDGE_MODEL_PRIMARY,
    judge_cache_dir: Path | None = None,
) -> list[CrossEvalCell]:
    """Score the per-target completions written by
    ``generate_completions_for_source``. Returns one ``CrossEvalCell``
    per (source, target, seed) row.

    Per CLAUDE.md checkpoint-per-phase: each target's verdict JSON is
    written IMMEDIATELY after judging, so a downstream crash doesn't
    lose the judge work.
    """
    out_dir = cross_eval_dir(repo_root, source, seed)
    # Default (Bucket B back-compat): NARROW + BROAD. Cross-bucket sweeps
    # (Buckets A/D/E) pass explicit ``targets=...`` from the dispatcher,
    # which enumerates the Rec-1 A_TARGETS / D_TARGETS / E_TARGETS tuples.
    all_targets: list[AnyTarget] = (
        list(NARROW_TARGETS) + list(BROAD_TARGETS) if targets is None else list(targets)
    )

    cells: list[CrossEvalCell] = []
    for tgt in all_targets:
        target_id = tgt.target_id
        comp_path = out_dir / f"{target_id}.completions.jsonl"
        if not comp_path.exists():
            raise FileNotFoundError(
                f"completions missing for source={source} seed={seed} target={target_id} "
                f"at {comp_path}. Run generate_completions_for_source first."
            )

        questions: list[str] = []
        completions_per_q: list[list[str]] = []
        n_total = 0
        n_chars = 0
        with comp_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                questions.append(rec["question"])
                completions_per_q.append(rec["completions"])
                n_total += len(rec["completions"])
                for c in rec["completions"]:
                    n_chars += len(c)

        median_tokens = _approx_median_tokens(completions_per_q)
        truncation_rate = 0.0  # already logged at generation time

        # MF-L round-3 revision: per-target judge router. B1_broad_em is
        # routed to ``_score_b1_openai_gpt4o`` with default
        # ``gpt-4o-2024-08-06`` (the validated #458/#468 anchor judge
        # and plan §11 spec). T1/T2/T3/B2 route to
        # ``judge_cell_completions`` with default Claude Sonnet 4.5
        # (#503 κ ≥ 0.7 calibrated). The caller-supplied ``judge_model``
        # is treated as an OPTIONAL OVERRIDE: when it matches the
        # router's default-provider prefix (``claude-*`` for Claude
        # judges, ``gpt-*`` for the gpt-4o judge) we honour the
        # override; otherwise the router's per-target default wins so
        # cross-cell B1 numbers stay anchored to gpt-4o regardless of a
        # mistakenly-passed Claude id.
        save_raw = out_dir / f"{target_id}.judge_raw.json"
        scorer, default_model = judge_for_target(target_id)
        if default_model.startswith("gpt-"):
            chosen_model = judge_model if judge_model.startswith("gpt-") else default_model
        else:
            chosen_model = judge_model if judge_model.startswith("claude") else default_model
        verdict = scorer(
            cell_id=f"{source}_seed{seed}__{target_id}",
            questions=questions,
            completions_per_question=completions_per_q,
            judge_model=chosen_model,
            cache_dir=judge_cache_dir,
            save_raw=save_raw,
        )

        # MF-D round-2 revision: merge the KL-secondary DV from the
        # per-cell ``<target_id>.kl.json`` file (written by
        # ``compute_kl_secondary_dv_for_source`` during the cross-eval
        # phase). If the file is absent the verdict stores ``None`` and
        # the regression falls back to the primary k/n DV per §5.1;
        # this is the saturation-fallback path the plan names.
        kl_dv = _read_kl_secondary_dv(out_dir, target_id)

        verdict["kl_secondary_dv"] = kl_dv

        verdict_out_path = out_dir / f"{target_id}.verdict.json"
        verdict_out_path.write_text(json.dumps(verdict, indent=2))

        cells.append(
            CrossEvalCell(
                source=source,
                target_id=target_id,
                seed=seed,
                k=int(verdict["k"]),
                n=int(verdict["n"]),
                rate=float(verdict["rate"]) if verdict["n"] > 0 else float("nan"),
                n_errors=int(verdict.get("n_errors", 0)),
                n_static_positive=int(verdict.get("n_static_positive", 0)),
                median_tokens=median_tokens,
                truncation_rate=truncation_rate,
                kl_secondary_dv=kl_dv,
            )
        )
    return cells


def _read_kl_secondary_dv(out_dir: Path, target_id: str) -> float | None:
    """Read the per-cell KL-secondary-DV scalar from ``<out_dir>/<target_id>.kl.json``.

    Schema (written by ``compute_kl_secondary_dv_for_source``):
        ``{"kl_per_response": float, "n_responses": int, ...}``.

    Returns the scalar mean ``kl_per_response``, or ``None`` if the file
    is absent or malformed (the regression treats absence as
    "saturation-fallback unavailable for this cell" per §5.1).
    """
    kl_path = out_dir / f"{target_id}.kl.json"
    if not kl_path.exists():
        return None
    try:
        obj = json.loads(kl_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Malformed KL DV JSON at %s; treating as None", kl_path)
        return None
    val = obj.get("kl_per_response")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def compute_kl_secondary_dv_for_source(
    *,
    source_adapter_path: str | Path,
    source: str,
    seed: int,
    base_model_id: str,
    repo_root: Path,
    targets: tuple[AnyTarget, ...] | None = None,
) -> dict[str, Path]:
    """MF-D round-2 revision: compute full-vocab KL(P_trained ‖ P_base)
    at the post-response slot per (source, target, seed) cell.

    Per plan §5.1 + ``.claude/rules/marker-leakage-measurement.md``
    saturation guard: the primary judge-rate DV saturates near the floor
    or ceiling; the KL DV is the non-saturating fallback. The
    implementation is one teacher-forced forward over the trained-adapter
    next-token distribution and one teacher-forced forward over the base
    model next-token distribution, both at the same post-response slot
    of each (question, completion) record from the generation phase.

    The KL per cell is averaged over all (question, completion) pairs
    written by ``generate_completions_for_source``. Per CLAUDE.md
    checkpoint-per-phase, the result is written immediately after each
    target's KL pass to ``<out_dir>/<target_id>.kl.json`` — the verdict
    phase reads this file via ``_read_kl_secondary_dv`` and merges it
    into the verdict JSON.

    This function is GPU-bound and is invoked by the cross-eval
    dispatcher AFTER the generation phase + BEFORE (or alongside) the
    judge phase. For smoke tests the KL JSON can be pre-written by hand
    (the merge logic in ``score_completions_for_source`` is exercised
    deterministically by the smoke fixture in
    ``tests/test_issue503_smoke.py``).

    Returns ``{target_id: path_to_kl_json}``.
    """
    import contextlib as _contextlib

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = cross_eval_dir(repo_root, source, seed)
    # Default (Bucket B back-compat): NARROW + BROAD. Cross-bucket sweeps
    # (Buckets A/D/E) pass explicit ``targets=...`` from the dispatcher,
    # which enumerates the Rec-1 A_TARGETS / D_TARGETS / E_TARGETS tuples.
    all_targets: list[AnyTarget] = (
        list(NARROW_TARGETS) + list(BROAD_TARGETS) if targets is None else list(targets)
    )

    logger.info("KL DV: loading base %s + adapter %s", base_model_id, source_adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    base_model.eval()
    trained_model = PeftModel.from_pretrained(base_model, str(source_adapter_path))
    trained_model.eval()

    device = next(base_model.parameters()).device

    written: dict[str, Path] = {}
    try:
        for tgt in all_targets:
            target_id = tgt.target_id
            comp_path = out_dir / f"{target_id}.completions.jsonl"
            if not comp_path.exists():
                logger.warning("KL DV: completions missing at %s; skipping", comp_path)
                continue

            kl_values: list[float] = []
            with comp_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    question = rec["question"]
                    for completion in rec["completions"]:
                        kl_val = _kl_post_response_slot(
                            base_model=base_model,
                            trained_model=trained_model,
                            tokenizer=tokenizer,
                            question=question,
                            completion=completion,
                            device=device,
                        )
                        kl_values.append(kl_val)

            if not kl_values:
                continue
            import numpy as _np

            mean_kl = float(_np.mean(kl_values))
            kl_path = out_dir / f"{target_id}.kl.json"
            kl_path.write_text(
                json.dumps(
                    {
                        "kl_per_response": mean_kl,
                        "n_responses": len(kl_values),
                        "source": source,
                        "seed": seed,
                        "target_id": target_id,
                        "method": "teacher_forced_full_vocab_kl_at_post_response_slot",
                    },
                    indent=2,
                )
            )
            written[target_id] = kl_path
            logger.info("  target=%s KL=%.4f (n=%d)", target_id, mean_kl, len(kl_values))
    finally:
        # MF-J round-3 revision (analyzer-weighable closure): vLLM is
        # not used here (HF Transformers + PEFT) so no vLLM-worker
        # reaping. We DO need to release the LoRA-wrapped trained model
        # + base model AND empty the CUDA allocator BEFORE the judge
        # phase (or any other GPU-touching step) loads. CUDA's caching
        # allocator does not return memory to the device on ``del``
        # alone; without ``empty_cache()`` the next framework's load
        # can OOM despite the variable being out of scope.
        with _contextlib.suppress(Exception):
            del trained_model
        with _contextlib.suppress(Exception):
            del base_model
        import gc as _gc

        _gc.collect()
        with _contextlib.suppress(Exception):
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

    return written


def _kl_post_response_slot(
    *,
    base_model,
    trained_model,
    tokenizer,
    question: str,
    completion: str,
    device,
) -> float:
    """One teacher-forced forward through both models; return full-vocab
    KL(P_trained ‖ P_base) at the post-response slot.

    The post-response slot is the position immediately after the last
    token of the completion under the chat template — the slot at which
    the model would emit EOS (or, in the marker-leakage analogue, the
    marker). We teacher-force the full prompt+completion through both
    models and read the next-token logits at that final position.

    Wrapped in ``torch.no_grad()`` internally so the caller does not
    accumulate the autograd graph across many cell × completion forwards.
    """
    import torch
    import torch.nn.functional as F

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": completion},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    # Post-response slot = last position in the tokenized sequence.
    pos = input_ids.shape[1] - 1

    with torch.no_grad():
        base_out = base_model(input_ids)
        trained_out = trained_model(input_ids)
    base_logits = base_out.logits[0, pos, :].float()
    trained_logits = trained_out.logits[0, pos, :].float()

    log_p_trained = F.log_softmax(trained_logits, dim=-1)
    log_p_base = F.log_softmax(base_logits, dim=-1)
    p_trained = log_p_trained.exp()

    # KL(P_trained || P_base) = sum_v P_trained(v) * (log P_trained(v) − log P_base(v)).
    kl = (p_trained * (log_p_trained - log_p_base)).sum().item()
    return float(kl)


def _approx_median_tokens(completions_per_q: list[list[str]]) -> float:
    """Rough median token count using the simple "1 token per 4 chars"
    proxy. The exact token count would require the tokenizer; this
    proxy is a covariate in the regression (log-tokens) and the rough
    estimate is sufficient for that regression term per plan §3.5
    length-confound covariate. Replaced by an exact tokenizer count in
    the analyzer-side recheck.
    """
    import numpy as np

    lengths_chars: list[int] = []
    for q_comps in completions_per_q:
        lengths_chars.extend(len(c) for c in q_comps)
    if not lengths_chars:
        return 0.0
    return float(np.median(lengths_chars) / 4.0)
