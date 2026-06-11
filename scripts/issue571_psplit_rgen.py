# Intentional Unicode (※, Δ) in scientific docstrings + log strings.
"""Task #571 persona-split-composition — Stage-2 Phase 1: frozen panel R (pod, GPU).

Generates the frozen on-policy negative responses R for the realized
panel's NON-assistant personas (assistant reuses the inherited
``R_train["A1"]`` — its prompt byte-equals A1) over the 30 Q_train
questions, matching the #460 R recipe exactly (``Source:
scripts/i460_phase1_generate_R.py``): vLLM greedy, temperature 0.0,
``max_new_tokens=1024``.

Hard checks (forked from #460):
- marker id 83399 (` ※`) absent from EVERY R — text substring AND token ids
  (a marker in R would corrupt the marker-only collator's mask). rc=6, no
  fallback.
- per-persona truncation ≤ 5%. On a breach: ONE registered persona swap
  (``issue571_psplit_geometry.reselect_and_write`` excluding the breached
  persona — next-ranked gate-passing candidate) + regeneration for the
  replacement; a SECOND breach kills (rc=5, plan §7 kill criterion b).

Checkpoint-per-phase: the output JSON is rewritten after EVERY persona, so
a crash never loses completed personas; a complete file whose panel matches
the realized panel resume-skips.

Outputs: ``data/issue_571/psplit/R_personas.json`` (+ fail-loud HF upload
to ``issue571_psplit/R_personas.json``).

Usage (pod, GPU0):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue571_psplit_rgen.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from issue571_psplit_common import (  # noqa: E402
    GEOMETRY_JSON,
    PANEL_PERSONAS_JSON,
    PSPLIT_DATA_DIR,
    R_PERSONAS_JSON,
)

logger = logging.getLogger("issue571.psplit_rgen")

SCHEMA_VERSION = "issue571_psplit_rgen_v1"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 1024  # #460 recipe
TRUNCATION_FAIL_THRESHOLD = 0.05
BREACH_FILE = PSPLIT_DATA_DIR / "rgen_breach.json"
HF_UPLOAD_PATH = "issue571_psplit/R_personas.json"


def _git_commit() -> str:
    """Short git commit of the repo this script runs from."""
    try:
        return (
            subprocess.run(  # epm-lint: subprocess-env-inherit -- git metadata probe, no creds
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_panel_personas() -> dict[str, str]:
    """The realized panel's 7 non-assistant {name: prompt} (Phase 0.5 output)."""
    if not PANEL_PERSONAS_JSON.exists():
        raise FileNotFoundError(
            f"{PANEL_PERSONAS_JSON} missing — run issue571_psplit_geometry.py (Phase 0.5) first"
        )
    prompts = json.loads(PANEL_PERSONAS_JSON.read_text())
    assert len(prompts) == 7 and "assistant" not in prompts, sorted(prompts)
    return prompts


def _questions() -> list[str]:
    from explore_persona_space.experiments.i460_data import load_q_train_answers

    questions = sorted(load_q_train_answers())
    assert len(questions) == 30, len(questions)
    return questions


def _persist(completions: dict, stats: dict, panel_personas: dict[str, str], complete: bool):
    """Atomic rewrite of the output JSON (checkpoint-per-persona)."""
    R_PERSONAS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "panel_personas": sorted(panel_personas),
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": MAX_NEW_TOKENS,
            "engine_seed": 42,
            "recipe_source": "scripts/i460_phase1_generate_R.py",
        },
        "completions": completions,
        "stats": stats,
        "metadata": {
            "task": 571,
            "followup_label": "persona-split-composition",
            "script": "issue571_psplit_rgen.py",
            "base_model": BASE_MODEL,
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python_version": platform.python_version(),
        },
    }
    tmp = R_PERSONAS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    tmp.replace(R_PERSONAS_JSON)


def _generate_personas(
    llm,
    sp,
    tokenizer,
    marker_id: int,
    names: list[str],
    prompts: dict[str, str],
    questions: list[str],
    completions: dict,
    stats: dict,
    panel_personas: dict[str, str],
) -> list[str]:
    """Generate R for ``names``; returns the list of truncation-breached personas.

    Marker-in-R is a HARD rc=6 exit (no fallback — #460 contract).
    """
    from issue560_crossrecipe_panel import build_persona_prompt

    breached: list[str] = []
    for name in names:
        texts = [build_persona_prompt(prompts[name], q, tokenizer) for q in questions]
        outs = llm.generate(texts, sp)
        assert len(outs) == len(questions), (name, len(outs))
        per_q: dict[str, dict] = {}
        n_trunc = 0
        for q, out in zip(questions, outs, strict=True):
            o = out.outputs[0]
            token_ids = list(o.token_ids)
            truncated = o.finish_reason == "length"
            n_trunc += int(truncated)
            if marker_id in token_ids or "※" in o.text:
                logger.error(
                    "marker found in base R for persona=%s q=%r — cannot proceed "
                    "(would corrupt the marker-only collator's mask)",
                    name,
                    q[:60],
                )
                _persist(completions, stats, panel_personas, complete=False)
                sys.exit(6)
            per_q[q] = {
                "response_text": o.text,
                "n_response_tokens": len(token_ids),
                "ended_with_eos": bool(token_ids and token_ids[-1] == tokenizer.eos_token_id),
                "truncated": truncated,
            }
        trunc_rate = n_trunc / len(questions)
        completions[name] = per_q
        stats["per_persona"][name] = {"n_truncated": n_trunc, "truncation_rate": trunc_rate}
        _persist(completions, stats, panel_personas, complete=False)
        logger.info("persona=%s: %d gens, truncation %.1f%%", name, len(per_q), 100 * trunc_rate)
        if trunc_rate > TRUNCATION_FAIL_THRESHOLD:
            breached.append(name)
    return breached


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    argparse.ArgumentParser(
        description="Task #571 psplit Stage-2 Phase 1: frozen panel R (vLLM, #460 recipe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args(argv)
    print("[phase=p1_rgen]", flush=True)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    panel_personas = _load_panel_personas()
    questions = _questions()

    # Resume-skip: a complete file whose panel matches the realized panel.
    if R_PERSONAS_JSON.exists():
        existing = json.loads(R_PERSONAS_JSON.read_text())
        if existing.get("complete") and existing.get("panel_personas") == sorted(panel_personas):
            logger.info("resume skip: %s complete for the realized panel", R_PERSONAS_JSON)
            return 0

    from issue560_crossrecipe_panel import load_tokenizer
    from vllm import LLM, SamplingParams

    from explore_persona_space.experiments.i406_conditions import MARKER_ID

    tokenizer, _bare = load_tokenizer()
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=4096,
    )
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=MAX_NEW_TOKENS)

    completions: dict = {}
    stats: dict = {"per_persona": {}}
    breached = _generate_personas(
        llm,
        sp,
        tokenizer,
        MARKER_ID,
        sorted(panel_personas),
        panel_personas,
        questions,
        completions,
        stats,
        panel_personas,
    )

    if breached:
        prior = json.loads(BREACH_FILE.read_text()) if BREACH_FILE.exists() else {"breached": []}
        already = set(prior["breached"])
        if already:
            logger.error(
                "SECOND truncation breach (%s after prior %s) — kill criterion §7(b)",
                breached,
                sorted(already),
            )
            sys.exit(5)
        BREACH_FILE.write_text(json.dumps({"breached": breached}))
        logger.warning(
            "truncation breach for %s — applying the ONE registered persona swap "
            "(geometry re-selection excluding the breached persona(s))",
            breached,
        )
        from issue571_psplit_geometry import reselect_and_write

        reselect_and_write(set(breached), swap_reason=f"R-gen truncation > 5% for {breached}")
        panel_personas = _load_panel_personas()
        new_names = [n for n in sorted(panel_personas) if n not in completions]
        second_breach = _generate_personas(
            llm,
            sp,
            tokenizer,
            MARKER_ID,
            new_names,
            panel_personas,
            questions,
            completions,
            stats,
            panel_personas,
        )
        if second_breach:
            logger.error(
                "replacement persona(s) %s ALSO breached truncation — kill criterion §7(b)",
                second_breach,
            )
            sys.exit(5)

    # Final asserts: every realized non-assistant panel persona present + clean.
    for name in sorted(panel_personas):
        assert name in completions and len(completions[name]) == 30, name
        assert stats["per_persona"][name]["truncation_rate"] <= TRUNCATION_FAIL_THRESHOLD, name
    _persist(completions, stats, panel_personas, complete=True)
    logger.info("R_personas.json complete: %d personas x %d questions", len(panel_personas), 30)

    # Fail-loud HF upload (datasets upload so any pod can access without scp).
    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload

    url = _upload(
        local_path=R_PERSONAS_JSON,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=HF_UPLOAD_PATH,
        delete_after=False,
        upload_as_file=True,
    )
    if not url:
        raise RuntimeError(f"HF upload failed for {R_PERSONAS_JSON} -> {HF_UPLOAD_PATH}")
    logger.info("uploaded %s -> %s", R_PERSONAS_JSON.name, HF_UPLOAD_PATH)
    # Geometry JSONs may have been rewritten by a swap — note for the dispatcher.
    logger.info("realized geometry record: %s", GEOMETRY_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
