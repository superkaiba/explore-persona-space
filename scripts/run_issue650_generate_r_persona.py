"""Issue #650 R_persona generation (round-2 blocker ``marker-eval-r-persona-missing``).

The marker eval (``run_issue650_eval.py::_eval_marker_cell`` ->
``shift_extract.extract_per_context_shift``) forwards the row
``T_persona(q) + R_persona(q)`` for every eval-panel persona over the first
``EVAL_N_PROMPTS_PER_PERSONA`` eval questions, where ``R_persona(q)`` is the
persona's OWN base-model greedy response. ``extract_per_context_shift`` HARD-
asserts every eval question has an R response, but NO prior pipeline phase
generated ``eval_results/issue_650/R_persona/<persona>.json`` — so marker eval
crashed on the first persona after all training was spent.

This script generates those responses ON-POLICY from the BASE model (greedy,
temp=0, persona via system prompt) over the resolved eval panel
(``PERSONA_POOL_19 + assistant + source``, dedup — identical to the eval's
``_resolve_eval_panel``) × ``EVAL_QUESTIONS[:EVAL_N_PROMPTS_PER_PERSONA]``, and
writes one ``R_persona/<persona>.json`` per persona in the
``issue_527_R_persona_v1`` schema (``schema_version`` / ``persona`` /
``responses`` dict) that ``_load_r_persona`` reads.

Generation uses vLLM batched ``LLM.generate()`` (CLAUDE.md always-on rule);
the worker subprocesses are reaped via the #399 ``_free_llm`` teardown. After
writing the JSONs the script runs a FAIL-LOUD coverage gate: every panel
persona must have a non-empty response for every eval question, else it exits
non-zero (so the pipeline aborts BEFORE the expensive marker eval phase).

CLI (smoke ≡ sweep with a one-persona/one-question subset):
    uv run python scripts/run_issue650_generate_r_persona.py
    uv run python scripts/run_issue650_generate_r_persona.py --phase smoke
"""

# ruff: noqa: RUF002  # math/scientific notation in docstrings

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.issue_650 import (  # noqa: E402
    BASE_MODEL,
    EVAL_N_PROMPTS_PER_PERSONA,
    PERSONA_POOL_19,
    SOURCE,
)
from explore_persona_space.experiments.issue_650.persona_registry import (  # noqa: E402
    assert_registry_resolves,
    load_persona_bank,
)
from explore_persona_space.personas import EVAL_QUESTIONS  # noqa: E402

log = logging.getLogger("issue_650.r_persona")

R_PERSONA_SCHEMA = "issue_527_R_persona_v1"
R_GEN_MAX_TOKENS = 768  # natural Qwen responses run ~150 tok; cap headroom (log truncation)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def resolve_eval_panel(persona_bank: dict[str, str]) -> list[str]:
    """Eval panel = PERSONA_POOL_19 + assistant (+ source, dedup) — matches
    ``run_issue650_eval.py::_resolve_eval_panel`` so R covers exactly the
    personas the marker eval forwards.
    """
    panel = [*list(PERSONA_POOL_19), "assistant"]
    if SOURCE not in panel:
        panel.append(SOURCE)
    seen: list[str] = []
    for n in panel:
        if n not in persona_bank:
            raise AssertionError(f"eval panel persona {n!r} not in persona_bank")
        if n not in seen:
            seen.append(n)
    return seen


def _free_llm(llm) -> None:
    """Destroy the vLLM engine + reap workers (memory: orphan-worker gotcha #399)."""
    import contextlib
    import gc

    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    del llm
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        torch.cuda.empty_cache()


def _generate_r_persona(
    *,
    panel: list[str],
    persona_bank: dict[str, str],
    questions: list[str],
    out_dir: Path,
    gpu_memory_utilization: float,
    llm=None,
    tokenizer=None,
) -> dict:
    """Greedy base-model responses for (panel × questions); write per-persona JSONs.

    Returns a coverage manifest. vLLM batches ALL (persona, question) prompts in
    one call (persona via system prompt); responses are mapped back by index.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    own_llm = llm is None
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if llm is None:
        from vllm import LLM

        llm = LLM(
            model=BASE_MODEL,
            tensor_parallel_size=1,
            max_model_len=4096,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
            disable_log_stats=True,
        )

    # Build the flat prompt list; remember (persona, question) per index.
    index: list[tuple[str, str]] = []
    prompts: list[str] = []
    for persona in panel:
        sys_prompt = persona_bank[persona]
        for q in questions:
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ]
            prompts.append(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
            index.append((persona, q))

    try:
        from vllm import SamplingParams

        sampling = SamplingParams(temperature=0.0, max_tokens=R_GEN_MAX_TOKENS)
        # use_tqdm=False per memory feedback_vllm_use_tqdm_zerodivision (#613).
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
    finally:
        if own_llm:
            _free_llm(llm)

    # Map responses back per persona; track truncation.
    responses: dict[str, dict[str, str]] = {p: {} for p in panel}
    n_truncated = 0
    for (persona, q), req in zip(index, outputs, strict=True):
        o = req.outputs[0]
        text = o.text.strip()
        if o.finish_reason != "stop":
            n_truncated += 1
        # Even a truncated response is a real on-policy continuation; keep it
        # (truncation rate is logged as a manipulation check, not a kill).
        responses[persona][q] = text

    git_commit = _git_commit()
    ts = _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds")
    for persona in panel:
        payload = {
            "schema_version": R_PERSONA_SCHEMA,
            "persona": persona,
            "base_model": BASE_MODEL,
            "n_questions": len(questions),
            "responses": responses[persona],
            "git_commit": git_commit,
            "timestamp_utc": ts,
            "decoding": "greedy(temp=0)",
            "max_new_tokens": R_GEN_MAX_TOKENS,
        }
        (out_dir / f"{persona}.json").write_text(json.dumps(payload, indent=2))
    log.info(
        "Wrote %d R_persona JSONs to %s (%d truncated of %d rows)",
        len(panel),
        out_dir,
        n_truncated,
        len(index),
    )
    return {
        "panel": panel,
        "n_personas": len(panel),
        "n_questions": len(questions),
        "n_rows": len(index),
        "n_truncated": n_truncated,
        "truncation_frac": (n_truncated / len(index)) if index else 0.0,
    }


def assert_coverage(out_dir: Path, panel: list[str], questions: list[str]) -> None:
    """FAIL-LOUD coverage gate: every panel persona has a non-empty response for
    every eval question (the exact contract ``extract_per_context_shift``
    hard-asserts). Runs BEFORE the marker eval phase so an incomplete R aborts
    the pipeline, not the GPU-spent eval.
    """
    missing: list[str] = []
    for persona in panel:
        path = out_dir / f"{persona}.json"
        if not path.is_file():
            missing.append(f"{persona}: file missing ({path})")
            continue
        payload = json.loads(path.read_text())
        sv = payload.get("schema_version")
        if sv != R_PERSONA_SCHEMA:
            missing.append(f"{persona}: schema_version={sv!r} != {R_PERSONA_SCHEMA!r}")
            continue
        resp = payload.get("responses") or {}
        for q in questions:
            if q not in resp or not str(resp[q]).strip():
                missing.append(f"{persona}: missing/empty response for q={q[:50]!r}")
    if missing:
        raise AssertionError(
            "R_persona coverage INCOMPLETE — the marker eval would crash in "
            f"extract_per_context_shift. {len(missing)} gap(s); first 5:\n  "
            + "\n  ".join(missing[:5])
        )
    log.info(
        "R_persona coverage OK: %d personas x %d questions all present + non-empty",
        len(panel),
        len(questions),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["smoke", "sweep"], default="sweep")
    ap.add_argument("--out-dir", default="eval_results/issue_650/R_persona")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument(
        "--personas",
        nargs="+",
        default=None,
        help="Explicit persona subset (smoke). Defaults to the full eval panel.",
    )
    ap.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Cap the question count (smoke). Defaults to EVAL_N_PROMPTS_PER_PERSONA.",
    )
    args = ap.parse_args(argv)

    persona_bank = load_persona_bank()
    assert_registry_resolves(persona_bank)
    panel = resolve_eval_panel(persona_bank)
    questions = list(EVAL_QUESTIONS[:EVAL_N_PROMPTS_PER_PERSONA])

    if args.phase == "smoke":
        # Smoke = sweep with a one-persona / one-question subset (same code path).
        panel = args.personas or [SOURCE]
        questions = questions[: (args.n_questions or 1)]
    else:
        if args.personas:
            panel = args.personas
        if args.n_questions:
            questions = questions[: args.n_questions]

    log.info(
        "[phase=r_persona_gen] panel=%d personas x %d questions (phase=%s)",
        len(panel),
        len(questions),
        args.phase,
    )
    out_dir = Path(args.out_dir)
    manifest = _generate_r_persona(
        panel=panel,
        persona_bank=persona_bank,
        questions=questions,
        out_dir=out_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    (out_dir / "coverage_manifest.json").write_text(json.dumps(manifest, indent=2))
    assert_coverage(out_dir, panel, questions)
    log.info("[phase=r_persona_done] %d personas covered", len(panel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
