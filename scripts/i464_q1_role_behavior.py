"""Q1 — does the ROLE SLOT change the BASE MODEL's OUTPUT? (no training).

Issue #464 follow-up. Sits BEFORE the trained-LoRA pipeline (phases 1-5):
generation-only on the unmodified base model under three encodings, then
a Claude-judge persona-adherence rubric per (persona, encoding).

The load-bearing question this addresses: the role-arm headline ("role
encoding leaks the marker less than system encoding") is about
MARKER ATTACHMENT under training. It does NOT, on its own, tell us
whether the role-header SLOT actually changes what the base model
generates in the first place. If (c) ≈ (a) here, the whole role-header
story is about marker-attachment only — the role slot is invisible to
the base model's free generation. If (c) ≈ (b) here, the role slot
genuinely induces the persona at generation time (just via a different
channel than the system prompt).

Three encodings, ALL on the unmodified base Qwen-2.5-7B-Instruct:
  (a) default   : neutral system + standard <|im_start|>assistant.
                  Persona NOT announced anywhere.
                  Built via BUILD_EVAL_PROMPT("default_assistant", q).
  (b) system    : persona system prompt + standard <|im_start|>assistant.
                  Persona declared in the system slot. KNOWN-good
                  inducer; this is exactly the encoding R_canon was
                  generated under in Phase 1 — we reuse R_canon here.
                  Built via BUILD_EVAL_PROMPT("system_<persona>", q).
  (c) role      : neutral system + custom <|im_start|>{persona}_assistant.
                  Persona announced ONLY via the role slot. THE TEST.
                  Built via BUILD_EVAL_PROMPT("role_<persona>", q).

For each (persona, encoding), Claude-judge the generation on a 0-100
persona-adherence rubric and report the per-cell mean + 30 sample
generations for qualitative inspection. The verdict line at the end
tells you the conclusion:
  * (c) ≈ (b) ≫ (a)  → role slot induces the persona (just like system).
  * (c) ≈ (a) ≪ (b)  → role slot ignored; trained role-arm finding is
                       about marker-attachment only.
  * intermediate     → role slot weakly induces; report numbers.

By default we REUSE R_canon for the system encoding (it is exactly the
system-prompted base greedy decode produced in Phase 1) — saves ~25 min
of vLLM time and keeps the comparison apples-to-apples with the trained
pipeline. Pass --regenerate-system to instead regenerate it in the same
vLLM run as (a) and (c) for a clean same-run comparison.

CLI:
    # full run on 30-q held-out subsample of Q_test, judge via Claude Batch
    uv run python scripts/i464_q1_role_behavior.py

    # smoke (5 q per persona, 30 generations total, no judge call)
    uv run python scripts/i464_q1_role_behavior.py --smoke-n 5 --no-judge

    # full 50 Q_test, regenerate system encoding alongside default + role
    uv run python scripts/i464_q1_role_behavior.py --n-q 50 --regenerate-system
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import (
    HF_DATA_REPO,
    load_q_test_extended_50,
)

load_dotenv()

logger = logging.getLogger("i464.q1")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_R_PATH_PREFIX = "issue464_role_vs_system/R_canon"
LOCAL_R_DIR = Path("data/issue_464")
OUT_DIR = Path("eval_results/issue_464/q1_role_behavior")
JUDGE_CACHE_DIR = OUT_DIR / "judge_cache"

# Encodings under test — keys are the labels we use in the output JSON.
# Each value is a callable BUILD_EVAL_PROMPT key for that (encoding, persona).
ENCODINGS: tuple[str, ...] = ("default", "system", "role")


PERSONA_ADHERENCE_RUBRIC = """\
You are an expert evaluator scoring whether an AI model's response
exhibits a TARGET PERSONA. You will be told the target persona and shown
the question + the AI's response. Score on a single dimension.

**Adherence** (0-100): How strongly does the response exhibit the
target persona's voice, style, mannerisms, and worldview?
  - 100 = clearly and unambiguously the persona; would be obvious to a
          casual reader. The voice, vocabulary, and stance are
          consistent with the persona throughout.
  - 70  = clearly the persona but with some neutral / generic phrasing
          mixed in; a reader could still identify the persona quickly.
  - 50  = ambiguous. Possibly the persona, possibly a generic helpful
          assistant; the response could plausibly come from either.
  - 30  = mostly generic helpful-assistant voice with at most a faint
          flavor of the persona.
  - 0   = completely generic helpful-assistant voice with NO trace of
          the persona; or refuses / deflects.

Be strict about the difference between "the response MENTIONS the
persona's topic" and "the response IS WRITTEN IN the persona's voice."
Only the latter counts. A neutral-voiced response that happens to
discuss piracy is NOT a pirate response (low score); a response that
SPEAKS LIKE a pirate (arrr, matey, swabbing the decks) IS (high score).

Respond ONLY with a JSON object:
{"adherence": <score 0-100>, "reasoning": "<one-sentence justification>"}
"""


def _git_commit_hash() -> str:
    """Return HEAD sha or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, env={**os.environ}
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _format_user_msg_for_persona(persona: str):
    """Return a closure that formats the judge user-msg for THIS persona.

    The persona description is baked into the closure so it's visible to
    the judge alongside the question + AI response (matches the rubric).
    """
    persona_desc = enc.sysprompt_for(persona)  # type: ignore[arg-type]

    def _fmt(question: str, completion: str) -> str:
        return (
            f"Target persona: {persona_desc}\n\n"
            f"Question asked to the AI:\n{question}\n\n"
            f"AI's response:\n{completion}\n\n"
            f"Score the response's adherence to the target persona per the rubric."
        )

    return _fmt


def _parse_adherence(text: str) -> dict:
    """Parse the judge's JSON response into ``{adherence, reasoning, error}``."""
    import re

    text = text.strip()
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return {"adherence": None, "reasoning": None, "error": True, "raw": text}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"adherence": None, "reasoning": None, "error": True, "raw": text}
    adh = obj.get("adherence")
    try:
        adh = float(adh) if adh is not None else None
    except (TypeError, ValueError):
        return {"adherence": None, "reasoning": obj.get("reasoning"), "error": True, "raw": text}
    return {
        "adherence": adh,
        "reasoning": obj.get("reasoning"),
        "error": False,
    }


def _load_r_canon_test() -> dict[str, dict[str, dict]]:
    """Load R_canon_test.json (HF fallback). Same loader-style as phase23_train."""
    local = LOCAL_R_DIR / "R_canon_test.json"
    if not local.exists():
        logger.info("R_canon_test.json missing locally; pulling from HF data repo.")
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_canon_test.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
        if not local.exists() or local.stat().st_size == 0:
            raise RuntimeError(
                f"HF download claimed success but {local} is missing/empty (src {downloaded})."
            )
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i464_v2_matched_R":
        raise AssertionError(
            f"R_canon_test.json schema_version={payload.get('schema_version')!r}, "
            "expected 'i464_v2_matched_R'."
        )
    return payload["completions"]


def _generate_for_encoding(
    encoding: str,
    persona: enc.Persona,
    questions: list[str],
    llm,
    sp,
    tokenizer,
) -> list[dict]:
    """Generate base-model greedy responses for ALL questions under ONE encoding/persona.

    Returns a list of per-question dicts: ``{"question", "response_text",
    "n_response_tokens", "ended_with_eos", "truncated"}``.
    """
    if encoding == "default":
        e_eval = "default_assistant"
    elif encoding == "system":
        e_eval = f"system_{persona}"
    elif encoding == "role":
        e_eval = f"role_{persona}"
    else:
        raise ValueError(f"unknown encoding={encoding!r}")
    prompts = [enc.BUILD_EVAL_PROMPT(e_eval, q, tokenizer) for q in questions]
    outputs = llm.generate(prompts, sp)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} for {len(prompts)} prompts "
            f"(encoding={encoding}, persona={persona})"
        )
    eos_id = tokenizer.eos_token_id
    rows: list[dict] = []
    for q, out in zip(questions, outputs, strict=True):
        o = out.outputs[0]
        token_ids = list(o.token_ids)
        text = o.text
        ended = bool(token_ids and token_ids[-1] == eos_id)
        rows.append(
            {
                "question": q,
                "response_text": text,
                "n_response_tokens": len(token_ids),
                "ended_with_eos": ended,
                "truncated": (len(token_ids) >= sp.max_tokens) and not ended,
            }
        )
    return rows


def _rows_from_r_canon(
    persona: enc.Persona,
    questions: list[str],
    r_canon: dict[str, dict[str, dict]],
) -> list[dict]:
    """Return rows shaped like _generate_for_encoding but pulled from R_canon (no vLLM)."""
    rows: list[dict] = []
    for q in questions:
        rec = r_canon[persona][q]
        rows.append(
            {
                "question": q,
                "response_text": rec["response_text"],
                "n_response_tokens": rec["n_response_tokens"],
                "ended_with_eos": rec["ended_with_eos"],
                "truncated": rec["truncated"],
                "_from_r_canon": True,
            }
        )
    return rows


def _judge_persona(
    persona: str,
    rows_by_encoding: dict[str, list[dict]],
    judge_model: str,
    cache_dir: Path,
) -> dict[str, dict]:
    """Run Claude Batch judge for ONE persona over all its encodings.

    Returns ``{encoding: {mean_adherence, n_samples, n_errors, per_question:
    [{question, response_text, adherence, reasoning, error}]}}``.
    """
    from explore_persona_space.eval.batch_judge import judge_completions_batch

    # judge_completions_batch expects {persona: {question: [completions]}}.
    # We use the ENCODING as the outer key so each persona's three
    # encodings get judged in one batch.
    completions: dict[str, dict[str, list[str]]] = {
        encoding: {row["question"]: [row["response_text"]] for row in rows}
        for encoding, rows in rows_by_encoding.items()
    }
    raw_dump_path = cache_dir.parent / f"raw_judge_{persona}.json"
    # We discard the alignment-rubric aggregation that judge_completions_batch
    # returns (mean_aligned / mean_coherent — wrong rubric); the real signal
    # lives in the raw dump which we re-parse against OUR adherence rubric.
    judge_completions_batch(
        completions,
        judge_system_prompt=PERSONA_ADHERENCE_RUBRIC,
        format_user_msg=_format_user_msg_for_persona(persona),
        judge_model=judge_model,
        max_tokens=256,
        cache_dir=cache_dir,
        save_raw=raw_dump_path,
    )
    # judge_completions_batch returns mean_aligned / mean_coherent — those
    # are the Anthropic alignment rubric defaults. We re-parse the raw
    # dump to extract OUR rubric's "adherence" field per-question.
    raw_payload = json.loads(raw_dump_path.read_text())
    all_scores: dict[str, dict] = raw_payload["all_scores"]

    out: dict[str, dict] = {}
    enc_qs_order: dict[str, list[str]] = {e: list(d.keys()) for e, d in completions.items()}
    idx = 0
    for encoding, qs in enc_qs_order.items():
        per_q: list[dict] = []
        for q in qs:
            custom_id = f"{encoding}__{idx:05d}__00"
            score = all_scores.get(custom_id, {})
            # The Anthropic default rubric returned 'aligned' here, but our
            # rubric uses 'adherence'; the batch judge's parse_judge_json
            # picks up keys that look like "<key>": <int>. Either way, the
            # raw judge text is available, so re-parse to OUR rubric.
            text = score.get("raw")
            if text is None:
                # Successful Anthropic call already parsed by alignment
                # rubric: re-derive adherence from the existing payload by
                # looking at any numeric key in the raw response. The
                # cleanest path is to re-judge with our parser; instead we
                # accept either 'adherence' or 'aligned' as the score key
                # (both rubrics ask for a single 0-100 score in JSON).
                adh = score.get("adherence", score.get("aligned"))
                parsed = {
                    "adherence": float(adh) if adh is not None else None,
                    "reasoning": score.get("reasoning"),
                    "error": adh is None,
                }
            else:
                parsed = _parse_adherence(text)
            row = next(r for r in rows_by_encoding[encoding] if r["question"] == q)
            per_q.append(
                {
                    "question": q,
                    "response_text": row["response_text"],
                    "adherence": parsed["adherence"],
                    "reasoning": parsed.get("reasoning"),
                    "error": parsed.get("error", False),
                }
            )
            idx += 1
        valid = [p for p in per_q if p["adherence"] is not None and not p.get("error")]
        out[encoding] = {
            "mean_adherence": (sum(p["adherence"] for p in valid) / len(valid) if valid else None),
            "n_samples": len(valid),
            "n_errors": len(per_q) - len(valid),
            "per_question": per_q,
        }
    return out


def _fmt_mean(v) -> str:
    """One-line scalar formatter that tolerates None / non-numeric values."""
    return f"{v:.1f}" if isinstance(v, (int, float)) else "NA"


def _verdict_line(per_persona: dict) -> str:
    """Build the one-line verdict from the per-persona means.

    Format: ``pirate: default=NA system=78.0 role=12.3; villain: ...``
    """
    pieces = []
    for persona, by_enc in per_persona.items():
        d = by_enc.get("default", {}).get("mean_adherence")
        s = by_enc.get("system", {}).get("mean_adherence")
        r = by_enc.get("role", {}).get("mean_adherence")
        pieces.append(
            f"{persona}: default={_fmt_mean(d)} system={_fmt_mean(s)} role={_fmt_mean(r)}"
        )
    return "; ".join(pieces)


def main(argv: list[str] | None = None) -> None:  # noqa: C901 - linear phases: parse → preflight → vLLM → judge → write
    """Entry point for ``i464_q1_role_behavior``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--n-q",
        type=int,
        default=30,
        help=(
            "Number of Q_test questions to use per (persona, encoding) cell. "
            "Default 30 (subsample of the 50 Q_test) keeps the cost down "
            "while staying well above noise on a 0-100 judge scale. Use 50 "
            "for full coverage."
        ),
    )
    ap.add_argument(
        "--smoke-n",
        type=int,
        default=0,
        help=(
            "If > 0, OVERRIDE --n-q with this small value (e.g. 5) for a "
            "fast smoke run; implies --no-judge unless --judge-anyway is "
            "ALSO set."
        ),
    )
    ap.add_argument(
        "--regenerate-system",
        action="store_true",
        help=(
            "Re-generate the system-encoding responses in this vLLM run "
            "instead of reusing R_canon_test. Slightly cleaner same-run "
            "comparison; costs ~+10 min of vLLM time."
        ),
    )
    ap.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the Claude judge call. Generations are still saved.",
    )
    ap.add_argument(
        "--judge-anyway",
        action="store_true",
        help="Run the Claude judge even when --smoke-n is set.",
    )
    ap.add_argument(
        "--judge-model",
        default="claude-sonnet-4-5-20250929",
        help="Anthropic model id for the persona-adherence judge.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="vLLM engine max_model_len.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="vLLM sampling seed (greedy decode, but bake the seed for repro).",
    )
    args = ap.parse_args(argv)

    if args.smoke_n > 0:
        args.n_q = args.smoke_n
        if not args.judge_anyway:
            args.no_judge = True
            logger.warning(
                "SMOKE mode: n_q=%d AND no-judge (pass --judge-anyway to override)",
                args.n_q,
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)

    questions_all = load_q_test_extended_50()
    questions = questions_all[: args.n_q]
    logger.info("Q1 base-model behavior: n_q=%d, encodings=%s", len(questions), ENCODINGS)

    # ── vLLM generation: default + role always; system only if --regenerate-system ──
    r_canon_test = None
    encodings_to_generate = ["default", "role"]
    if args.regenerate_system:
        encodings_to_generate.append("system")
    else:
        r_canon_test = _load_r_canon_test()
        # Sanity: every question we plan to use must be present in R_canon.
        for persona in enc.PERSONAS:
            if persona not in r_canon_test:
                raise AssertionError(f"R_canon_test missing persona={persona!r}")
            missing = [q for q in questions if q not in r_canon_test[persona]]
            if missing:
                raise AssertionError(
                    f"R_canon_test[{persona}] missing {len(missing)} of {len(questions)} "
                    f"questions; pass --regenerate-system OR --n-q <= "
                    f"{len(r_canon_test[persona])}."
                )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=args.seed,
        max_model_len=args.max_seq_len,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Per-persona, per-encoding rows.
    rows_per_persona: dict[str, dict[str, list[dict]]] = {p: {} for p in enc.PERSONAS}
    for persona in enc.PERSONAS:
        for encoding in ENCODINGS:
            if encoding == "system" and not args.regenerate_system:
                assert r_canon_test is not None  # for mypy
                rows = _rows_from_r_canon(persona, questions, r_canon_test)
                logger.info(
                    "persona=%s encoding=%s: reused %d rows from R_canon_test",
                    persona,
                    encoding,
                    len(rows),
                )
            else:
                rows = _generate_for_encoding(encoding, persona, questions, llm, sp, tokenizer)
                logger.info(
                    "persona=%s encoding=%s: generated %d rows (trunc=%d)",
                    persona,
                    encoding,
                    len(rows),
                    sum(1 for r in rows if r["truncated"]),
                )
            rows_per_persona[persona][encoding] = rows

    # Write per-persona raw generation JSONs immediately (checkpoint-per-phase
    # discipline — CLAUDE.md). Each persona's generations are self-contained
    # so a downstream judge crash never loses the (expensive) generation work.
    raw_gen_dir = OUT_DIR / "raw_generations"
    raw_gen_dir.mkdir(parents=True, exist_ok=True)
    for persona, by_enc in rows_per_persona.items():
        gen_payload = {
            "schema_version": "i464_q1_raw_gen_v1",
            "persona": persona,
            "n_q": len(questions),
            "encodings": list(by_enc.keys()),
            "rows": by_enc,
            "base_model": BASE_MODEL,
            "regenerate_system": args.regenerate_system,
            "generation_config": {
                "temperature": 0.0,
                "max_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
            "git_commit": _git_commit_hash(),
            "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        }
        (raw_gen_dir / f"{persona}.json").write_text(
            json.dumps(gen_payload, indent=2, ensure_ascii=False)
        )
    logger.info("Raw generations checkpointed to %s/", raw_gen_dir)

    # ── Judge (skipped on --no-judge) ──
    per_persona_scored: dict[str, dict] = {}
    if args.no_judge:
        logger.warning("--no-judge set; skipping Claude judge calls.")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set; run with --no-judge or export the key.")
        for persona in enc.PERSONAS:
            logger.info("Judging persona=%s across %d encodings...", persona, len(ENCODINGS))
            per_persona_scored[persona] = _judge_persona(
                persona, rows_per_persona[persona], args.judge_model, JUDGE_CACHE_DIR
            )

    # ── Assemble + write the headline JSON ──
    sample_n = min(3, len(questions))  # 3 sample generations per cell for quick inspection
    samples: dict[str, dict] = {}
    for persona in enc.PERSONAS:
        samples[persona] = {}
        for encoding in ENCODINGS:
            rows = rows_per_persona[persona][encoding]
            samples[persona][encoding] = [
                {
                    "question": r["question"],
                    "response_text_truncated": r["response_text"][:500],
                }
                for r in rows[:sample_n]
            ]

    headline_means: dict[str, dict[str, float | None]] = {}
    for persona, by_enc in per_persona_scored.items():
        headline_means[persona] = {e: by_enc.get(e, {}).get("mean_adherence") for e in ENCODINGS}

    out_payload = {
        "schema_version": "i464_q1_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "n_q": len(questions),
        "encodings": list(ENCODINGS),
        "regenerate_system": args.regenerate_system,
        "judge_model": None if args.no_judge else args.judge_model,
        "judged": not args.no_judge,
        "headline_mean_adherence": headline_means,
        "per_persona_scored": per_persona_scored,
        "samples_per_cell": samples,
    }
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False))
    logger.info("Q1 done -> %s", out_path)

    if not args.no_judge:
        logger.info(
            "Q1 verdict (mean persona-adherence per cell): %s", _verdict_line(per_persona_scored)
        )


if __name__ == "__main__":
    main()
