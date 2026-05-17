#!/usr/bin/env python3
"""Measure entropy of answer conditional on CoT — issue #355.

Reads the saved CoT text from #186 librarian persona-CoT result JSONs (seeds
42/137/256), loads the matching wrong-answer-SFT checkpoint from HF Hub, and
runs:

  1. Analytical pass — vLLM ``SamplingParams(max_tokens=1, logprobs=20,
     temperature=0.0)`` over all 1172 q_ids x 3 personas x 3 cot_styles
     per seed (= 31 644 forward passes for the main 27-arm grid). The first
     generated token's top-20 logprobs are used to compute ``H_top20``,
     ``H_abcd``, ``top20_mass``, and ``abcd_total_mass_pre_renorm``.
  2. Empirical pass — vLLM ``SamplingParams(n=8, temperature=1.0,
     max_tokens=8)`` over a stratified-by-correct_answer N=200 q_id
     subsample. Empirical entropy is reported as Miller-Madow-corrected
     ``H_MM`` (headline) with plug-in ``H_mle`` retained as a diagnostic.
  3. A1 cross-seed teacher-forcing sub-grid — 6 off-diagonal cells on
     librarian x persona_cot at N=200 (analytical; empirical companion
     optional via ``cross_seed.empirical_companion=true``).
  4. A3 comedian-source confirmation cell — 9 arms x N=200 at seed=42 only.

USAGE
-----

Local smoke (no GPU; only strip-coverage assertion):

    uv run python scripts/measure_cot_entropy.py --smoke-strip-coverage-only

Real pod run (full pipeline):

    uv run python scripts/measure_cot_entropy.py

Override a key:

    uv run python scripts/measure_cot_entropy.py empirical.n_q=50

WHY `max_tokens=1` IS OK HERE
-----------------------------

CLAUDE.md mandates ``max_new_tokens >= 2048`` for marker / end-of-completion
evals. THIS SCRIPT IS EXEMPT for both passes:

  * Analytical pass uses ``max_tokens=1`` BY DESIGN — the metric is the
    next-token logprob at the answer position, not the full continuation.
  * Empirical pass uses ``max_tokens=8`` — long enough to capture the first
    letter token, parenthesis variants, and any minor whitespace prefix.
    The :func:`parse_first_answer_letter` parser is robust to all of these.

The marker-eval rule applies to evals that read a substring late in a
generation (e.g. ``[ZLT]``); here we read the FIRST emitted token only,
so truncation by length is impossible.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# ── Defer all heavy imports until after Hydra parses CLI (so --help is fast) ─
# (vLLM, transformers, huggingface_hub, wandb, torch imports happen inside
# the main pipeline body or per-stage functions.)

# ────────────────────────────────────────────────────────────────────────────
# Plumbing helpers
# ────────────────────────────────────────────────────────────────────────────


def _git_commit_sha(repo_root: Path) -> str:
    """Return the current git commit SHA, or 'unknown' on failure."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True)
        return out.strip()
    except Exception as e:  # pragma: no cover — sanity guard
        logger.warning("Could not read git commit: %s", e)
        return "unknown"


def _try_package_version(name: str) -> str:
    """Return ``importlib.metadata.version(name)`` or 'unknown'."""
    try:
        import importlib.metadata as m

        return m.version(name)
    except Exception:
        return "unknown"


def _load_arc_questions(arc_data_path: str) -> list[dict]:
    """Load ARC-C question rows from the given JSONL path."""
    with open(arc_data_path) as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        raise ValueError(f"ARC-C file {arc_data_path} is empty")
    return rows


def _load_persona_prompts() -> dict[str, str]:
    """Map each {librarian, comedian, baseline} → system prompt text."""
    from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

    return {
        "librarian": PERSONAS["librarian"],
        "comedian": PERSONAS["comedian"],
        "baseline": ASSISTANT_PROMPT,
    }


def _snapshot_download_subfolder(
    repo_id: str, revision: str, subfolder: str, local_dir: str
) -> str:
    """Download a single subfolder of an HF Hub model repo, return local path."""
    from huggingface_hub import snapshot_download

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=local_dir,
    )
    return str(Path(local_dir) / subfolder)


def _resolve_comedian_source(cfg: DictConfig) -> str | None:
    """Probe HF Hub for the A3 comedian-source seed-42 checkpoint.

    Walks ``cfg.comedian_source.candidates`` in order. For each candidate
    family, requires BOTH:

      * a local ``eval_results/issue186/<family>_seed42/result.json`` (the
        saved CoT text we'll teacher-force), AND
      * a remote ``<family>_seed42_post_em`` subfolder on the model repo
        @ revision (the checkpoint we'll load into vLLM).

    Returns the chosen family string, or ``None`` if none qualify.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        repo_files = set(api.list_repo_files(cfg.source.repo_id, revision=cfg.source.revision))
    except Exception as e:
        logger.warning(
            "Could not list_repo_files(%s@%s): %s — A3 will skip.",
            cfg.source.repo_id,
            cfg.source.revision,
            e,
        )
        return None

    eval_root = Path(cfg.source.eval_inputs_dir)
    for cand in cfg.comedian_source.candidates:
        local_json = eval_root / f"{cand}_seed{cfg.comedian_source.seed}" / "result.json"
        if not local_json.exists():
            continue
        # Confirm at least one file under the candidate subfolder is on the repo.
        prefix = f"i186_{cand}_seed{cfg.comedian_source.seed}_post_em/"
        if any(p.startswith(prefix) for p in repo_files):
            return cand
    return None


# ────────────────────────────────────────────────────────────────────────────
# Smoke checks
# ────────────────────────────────────────────────────────────────────────────


def _smoke_strip_coverage(cfg: DictConfig) -> None:
    """Iterate ALL #186 rows across librarian + one non-librarian source x both
    fields, and HARD-FAIL the run if any post-strip text ends on a bare A/B/C/D.

    Plan §4 line 116. This is CPU-only (no model required) and runs locally
    BEFORE the per-seed vLLM loop.
    """
    from explore_persona_space.eval.entropy import (
        ends_with_bare_answer_letter,
        strip_trailing_answer,
    )

    eval_root = Path(cfg.source.eval_inputs_dir)
    sources_to_check: list[tuple[str, str]] = []
    # Librarian — required.
    sources_to_check.append((f"{cfg.source.family}_seed42", "persona_cot_text"))
    sources_to_check.append((f"{cfg.source.family}_seed42", "generic_cot_text"))
    # Non-librarian — pick first existing from the preference order.
    for cand in cfg.strip.smoke_sources:
        if cand == cfg.source.family:
            continue
        p = eval_root / f"{cand}_seed42" / "result.json"
        if p.exists():
            sources_to_check.append((f"{cand}_seed42", "persona_cot_text"))
            sources_to_check.append((f"{cand}_seed42", "generic_cot_text"))
            break

    logger.info("--smoke-strip-coverage iterating %d combinations", len(sources_to_check))
    print(f"smoke-strip-coverage: {len(sources_to_check)} (source, field) combinations")

    # Each source file has 3 eval personas (librarian / comedian / assistant
    # at minimum); each persona's `raw[i].<field>` is its OWN CoT, generated
    # under that persona's system prompt during #186 (round-1 B1 confirmed
    # the per-persona CoTs are genuinely different — see B1 fix). The smoke
    # therefore iterates ALL eval-personas' rows for each (source, field)
    # combination to confirm the strip pipeline is comprehensive.
    eval_persona_keys = list(cfg.eval_personas.values())
    total_failed = 0
    hist_overall: dict[tuple[str, str], Counter] = {}
    for src, field in sources_to_check:
        with open(eval_root / src / "result.json") as f:
            r = json.load(f)
        per_persona = r.get("per_persona", {})
        if not per_persona:
            raise RuntimeError(f"{src}: missing per_persona block")
        rule_hits: Counter = Counter()
        failed: list[tuple[str, int, str]] = []
        rows_seen = 0
        for persona_key in eval_persona_keys:
            if persona_key not in per_persona:
                # Smaller fixture / OOD persona — skip silently rather than
                # explode (smoke is best-effort coverage, not enumeration).
                continue
            raws = per_persona[persona_key].get("raw", [])
            for raw in raws:
                txt = raw.get(field, "")
                if not txt:
                    continue
                rows_seen += 1
                stripped, rid = strip_trailing_answer(txt)
                rule_hits[rid] += 1
                if ends_with_bare_answer_letter(stripped):
                    failed.append((persona_key, raw["q_id"], stripped[-100:]))
        hist_overall[(src, field)] = rule_hits
        print(
            f"  {src}/{field}: rows={rows_seen} rule_hits={dict(rule_hits)}  failed={len(failed)}"
        )
        for persona_key, qid, tail in failed[:5]:
            print(f"    FAIL persona={persona_key} q_id={qid}: ...{tail!r}")
        total_failed += len(failed)

    if total_failed > 0:
        msg = (
            f"--smoke-strip-coverage FAILED: {total_failed} rows still end with a "
            "bare A/B/C/D after the strip pipeline.  Fix the regex set before "
            "launching the experiment."
        )
        logger.error(msg)
        print(msg, file=sys.stderr)
        sys.exit(1)

    print("smoke-strip-coverage PASSED — 100% post-strip non-letter termination.")


def _save_smoke_prompts(cfg: DictConfig, tokenizer, rows: list[dict]) -> None:
    """Write 10-row spot-check prompts per arm to ``smoke_prompts.json`` (plan
    line 242).
    """
    from explore_persona_space.eval.entropy import (
        build_teacher_forced_prompt,
        strip_trailing_answer,
    )

    persona_prompts = _load_persona_prompts()
    samples: dict[str, list[dict]] = {}
    spot = cfg.strip.smoke_spot_check_rows
    for persona_label, sys_prompt in persona_prompts.items():
        for cot_style in cfg.cot_styles:
            arm = f"{persona_label}__{cot_style}"
            arm_samples: list[dict] = []
            for raw_row, arc_row in rows[:spot]:
                if cot_style == "no_cot":
                    cot_text = ""
                    rule_id = None
                else:
                    raw_text = raw_row.get(f"{cot_style}_text", "")
                    stripped, rule_id = strip_trailing_answer(raw_text)
                    cot_text = stripped
                prompt = build_teacher_forced_prompt(
                    tokenizer, sys_prompt, arc_row, cot_text, cot_style
                )
                arm_samples.append(
                    {
                        "q_id": raw_row["q_id"],
                        "prompt_tail": prompt[-300:],
                        "strip_rule_id": rule_id,
                    }
                )
            samples[arm] = arm_samples

    out_path = Path(cfg.output_dir) / "smoke_prompts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info("Wrote smoke prompts to %s", out_path)


# ────────────────────────────────────────────────────────────────────────────
# Row assembly
# ────────────────────────────────────────────────────────────────────────────


def _pair_raw_with_arc(raw_rows: list[dict], arc_rows: list[dict]) -> list[tuple[dict, dict]]:
    """Pair #186 raw result rows with ARC-C question rows by q_id.

    The plan §4 line 104 requires asserting ``row["q_id"] == q_id`` and
    ``row["correct_answer"] == arc_row["correct_answer"]`` for every pair.
    """
    paired: list[tuple[dict, dict]] = []
    for q_id, raw in enumerate(raw_rows):
        if raw.get("q_id") != q_id:
            raise RuntimeError(f"q_id mismatch: raw[{q_id}].q_id = {raw.get('q_id')!r}")
        if q_id >= len(arc_rows):
            raise RuntimeError(f"raw row q_id={q_id} exceeds ARC-C row count {len(arc_rows)}")
        arc_row = arc_rows[q_id]
        if raw.get("correct_answer") != arc_row.get("correct_answer"):
            raise RuntimeError(
                f"correct_answer mismatch at q_id={q_id}: "
                f"raw={raw.get('correct_answer')!r}, arc={arc_row.get('correct_answer')!r}"
            )
        paired.append((raw, arc_row))
    return paired


def _stratified_subsample(
    paired_rows: list[tuple[dict, dict]],
    n_per_letter: int,
    rng_seed: int,
) -> tuple[list[tuple[dict, dict]], list[int]]:
    """Stratify a paired-row list by ``correct_answer`` letter, picking
    ``n_per_letter`` rows for each of A/B/C/D. Returns ``(subsample, excluded
    numeric_q_ids)``.

    Numeric-label q_ids (correct_answer in {"1","2","3","4"}) are excluded
    AND returned separately for ``aggregate.json``.
    """
    import random as _random

    by_letter: dict[str, list[tuple[dict, dict]]] = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
    }
    numeric_qids: list[int] = []
    for raw, arc in paired_rows:
        ans = raw.get("correct_answer")
        if ans in by_letter:
            by_letter[ans].append((raw, arc))
        else:
            numeric_qids.append(raw["q_id"])

    rng = _random.Random(rng_seed)
    sampled: list[tuple[dict, dict]] = []
    for letter in ["A", "B", "C", "D"]:
        bucket = by_letter[letter]
        if len(bucket) < n_per_letter:
            raise RuntimeError(
                f"Stratified subsample requested {n_per_letter} rows for letter "
                f"{letter} but only {len(bucket)} exist."
            )
        sampled.extend(rng.sample(bucket, n_per_letter))
    # Sort by q_id to keep downstream alignment deterministic.
    sampled.sort(key=lambda t: t[0]["q_id"])
    return sampled, numeric_qids


# ────────────────────────────────────────────────────────────────────────────
# Per-row JSONL writer
# ────────────────────────────────────────────────────────────────────────────


def _row_to_jsonl_dict(
    *,
    q_id: int,
    persona: str,
    cot_style: str,
    seed: int,
    source_seed: int,
    source_persona: str,
    source_cot_style: str,
    entropy_result,
    strip_rule_id: int | None,
    cot_text_post_strip: str,
    tokenizer,
) -> dict:
    """Build the per-row analytical JSONL dict (plan §4 lines 130-135)."""
    if cot_text_post_strip:
        token_count = len(tokenizer.encode(cot_text_post_strip, add_special_tokens=False))
    else:
        token_count = 0
    return {
        "q_id": q_id,
        "persona": persona,
        "cot_style": cot_style,
        "seed": seed,
        "source_seed": source_seed,
        "source_persona": source_persona,
        "source_cot_style": source_cot_style,
        "H_abcd": entropy_result.h_abcd,
        "H_top20": entropy_result.h_top20,
        "top20_mass": entropy_result.top20_mass,
        "abcd_total_mass_pre_renorm": entropy_result.abcd_total_mass_pre_renorm,
        "restricted_missing": entropy_result.restricted_missing,
        "strip_rule_id": strip_rule_id,
        "cot_text_len_post_strip_chars": len(cot_text_post_strip),
        "cot_text_len_post_strip_tokens": token_count,
        "pred_argmax_token": entropy_result.pred_argmax_token,
        "pred_argmax_letter": entropy_result.pred_argmax_letter,
    }


# ────────────────────────────────────────────────────────────────────────────
# Analytical pass — runs per seed
# ────────────────────────────────────────────────────────────────────────────


def _source_persona_key_for_eval_personas(cfg: DictConfig, per_persona: dict) -> str:
    """Return the JSON persona key whose `raw` rows define the canonical q_id
    list for stratified subsampling.

    The subsample stratifies by `correct_answer` (an ARC ground-truth
    attribute that is the same across every per_persona[X].raw[i]). Picking
    any persona that exists in the file is correct; we prefer
    ``"librarian"`` for stability (the source family is librarian-trained
    in the main grid), then fall back to the first eval-persona key in
    cfg, then to any present persona.
    """
    if "librarian" in per_persona:
        return "librarian"
    for json_key in cfg.eval_personas.values():
        if json_key in per_persona:
            return json_key
    if per_persona:
        return next(iter(per_persona))
    raise RuntimeError("per_persona is empty — cannot pick a source persona key")


def _build_paired_for_persona(
    per_persona: dict,
    json_persona_key: str,
    arc_rows: list[dict],
    *,
    q_id_filter: set[int] | None = None,
    max_q: int | None = None,
) -> list[tuple[dict, dict]]:
    """Pair this eval persona's raw rows with their ARC counterparts.

    Each eval persona inside one #186 result file carries its OWN saved
    `*_cot_text` per row (the CoT was generated UNDER that persona's
    system prompt during #186). The pairing MUST be done per persona so
    that teacher-forcing uses the persona-authored CoT, not librarian's.

    Args:
        per_persona: the ``result["per_persona"]`` dict from a #186 JSON.
        json_persona_key: the persona key inside ``per_persona`` to fetch
            (``"librarian"`` / ``"comedian"`` / ``"assistant"``).
        arc_rows: full ARC-C row list (1172 rows).
        q_id_filter: optional set of q_ids to keep. If given, returns only
            rows whose ``q_id`` is in the set, preserving original order.
        max_q: optional hard cap (applied AFTER filtering).

    Returns:
        list of ``(raw_row, arc_row)`` pairs.
    """
    if json_persona_key not in per_persona:
        raise RuntimeError(
            f"per_persona missing key {json_persona_key!r}; available: {sorted(per_persona)}"
        )
    raws = per_persona[json_persona_key]["raw"]
    if q_id_filter is None:
        paired = _pair_raw_with_arc(raws, arc_rows)
    else:
        # Non-contiguous subsample: do per-q_id pairing + consistency check.
        paired = []
        for raw in raws:
            q_id = raw.get("q_id")
            if q_id not in q_id_filter:
                continue
            arc = arc_rows[q_id]
            if arc.get("correct_answer") != raw.get("correct_answer"):
                raise RuntimeError(
                    "filtered pairing correct_answer mismatch at q_id="
                    f"{q_id}: raw={raw.get('correct_answer')!r}, "
                    f"arc={arc.get('correct_answer')!r}"
                )
            paired.append((raw, arc))
    if max_q is not None:
        paired = paired[:max_q]
    return paired


def _run_analytical_for_seed(
    cfg: DictConfig,
    seed: int,
    llm,
    tokenizer,
    per_persona: dict,
    arc_rows: list[dict],
    persona_prompts: dict[str, str],
    answer_token_ids: dict[str, set[int]],
    output_root: Path,
    *,
    q_id_filter: set[int] | None = None,
    max_q: int | None = None,
    source_seed_override: int | None = None,
    source_persona_override: str | None = None,
    output_subdir: str = "analytical",
) -> dict[tuple[str, str], list[dict]]:
    """Run the analytical pass for a single seed checkpoint.

    Returns ``{(persona, cot_style): [row_dict, ...]}``. Writes one JSONL
    per ``(persona, cot_style)`` arm.

    The CoT text is fetched PER eval persona from ``per_persona[<json_key>]``
    — each persona's `raw[i].<cot_style>_text` was generated under THAT
    persona's system prompt during the #186 run, so teacher-forcing MUST
    use the matching persona-authored CoT. Round-1 code review B1 caught
    a regression where librarian's CoT was reused for all eval personas.

    Args:
        per_persona: the ``result["per_persona"]`` dict from the #186 source
            JSON for the loaded seed checkpoint.
        arc_rows: full ARC-C question rows (1172 rows).
        q_id_filter: optional set of q_ids restricting the rows used (for
            the A3 cross-persona sub-grid which evaluates on a subsample).
        max_q: optional hard cap on rows AFTER filtering.
        source_seed_override / source_persona_override: see plan §4 — A1
        cross-seed (different source_seed) and A3 cross-persona (different
        source_persona) sub-grids fill these in.
    """
    from vllm import SamplingParams

    from explore_persona_space.eval.entropy import (
        build_teacher_forced_prompt,
        entropy_from_logprobs,
        strip_trailing_answer,
    )

    sampling = SamplingParams(
        max_tokens=1, logprobs=cfg.analytical.top_k, temperature=0.0, top_p=1.0
    )

    source_seed = source_seed_override if source_seed_override is not None else seed
    source_persona = (
        source_persona_override
        if source_persona_override is not None
        else cfg.source.family.split("_persona_cot")[0].split("_generic_cot")[0]
    )
    source_cot_style = "persona_cot" if cfg.source.family.endswith("persona_cot") else "generic_cot"

    per_arm_rows: dict[tuple[str, str], list[dict]] = {}
    for persona_label, json_persona_key in cfg.eval_personas.items():
        sys_prompt = persona_prompts[persona_label]
        # B1 fix: fetch THIS persona's raw rows, not librarian's. Each
        # eval persona has its own *_cot_text per row.
        paired_rows = _build_paired_for_persona(
            per_persona,
            json_persona_key,
            arc_rows,
            q_id_filter=q_id_filter,
            max_q=max_q,
        )
        for cot_style in cfg.cot_styles:
            prompts: list[str] = []
            row_meta: list[tuple[int, int | None, str]] = []
            for raw_row, arc_row in paired_rows:
                if cot_style == "no_cot":
                    cot_text_stripped = ""
                    strip_rule_id: int | None = None
                else:
                    raw_text = raw_row.get(f"{cot_style}_text", "")
                    cot_text_stripped, strip_rule_id = strip_trailing_answer(raw_text)
                prompts.append(
                    build_teacher_forced_prompt(
                        tokenizer, sys_prompt, arc_row, cot_text_stripped, cot_style
                    )
                )
                row_meta.append((raw_row["q_id"], strip_rule_id, cot_text_stripped))

            t0 = time.time()
            outs = llm.generate(prompts, sampling)
            elapsed = time.time() - t0
            logger.info(
                "analytical arm=%s/%s seed=%d: %d prompts in %.1fs",
                persona_label,
                cot_style,
                seed,
                len(prompts),
                elapsed,
            )

            arm_rows: list[dict] = []
            for out, (q_id, strip_rule_id, cot_text_stripped) in zip(outs, row_meta, strict=True):
                step_logprobs = out.outputs[0].logprobs
                first_step = step_logprobs[0] if step_logprobs else {}
                er = entropy_from_logprobs(first_step, answer_token_ids)
                arm_rows.append(
                    _row_to_jsonl_dict(
                        q_id=q_id,
                        persona=persona_label,
                        cot_style=cot_style,
                        seed=seed,
                        source_seed=source_seed,
                        source_persona=source_persona,
                        source_cot_style=source_cot_style,
                        entropy_result=er,
                        strip_rule_id=strip_rule_id,
                        cot_text_post_strip=cot_text_stripped,
                        tokenizer=tokenizer,
                    )
                )

            out_dir = output_root / output_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{persona_label}_{cot_style}_seed{seed}.jsonl"
            with open(out_path, "w") as f:
                for r in arm_rows:
                    f.write(json.dumps(r) + "\n")
            per_arm_rows[(persona_label, cot_style)] = arm_rows
    return per_arm_rows


# ────────────────────────────────────────────────────────────────────────────
# Empirical pass — runs per seed
# ────────────────────────────────────────────────────────────────────────────


def _run_empirical_for_seed(
    cfg: DictConfig,
    seed: int,
    llm,
    tokenizer,
    per_persona: dict,
    arc_rows: list[dict],
    subsample_q_ids: set[int],
    persona_prompts: dict[str, str],
    output_root: Path,
    *,
    source_seed_override: int | None = None,
    source_persona_override: str | None = None,
    output_subdir: str = "empirical",
) -> dict[tuple[str, str], list[dict]]:
    """Run the empirical pass (n=8 sampling) for a single seed checkpoint.

    See :func:`_run_analytical_for_seed` for the per-persona CoT-text
    fetch rationale (round-1 B1 fix). The subsample q_ids are passed as
    a set so each persona's rows are filtered to the SAME q_ids — that
    keeps per-q_id pairing across personas (which Spearman/Wilcoxon
    aggregates depend on).
    """
    from vllm import SamplingParams

    from explore_persona_space.eval.entropy import (
        build_teacher_forced_prompt,
        miller_madow_entropy,
        parse_first_answer_letter,
        strip_trailing_answer,
    )

    sampling = SamplingParams(
        n=cfg.empirical.n_samples,
        temperature=cfg.empirical.temperature,
        top_p=1.0,
        max_tokens=cfg.empirical.max_tokens,
        seed=seed,
    )

    source_seed = source_seed_override if source_seed_override is not None else seed
    source_persona = (
        source_persona_override
        if source_persona_override is not None
        else cfg.source.family.split("_persona_cot")[0].split("_generic_cot")[0]
    )
    source_cot_style = "persona_cot" if cfg.source.family.endswith("persona_cot") else "generic_cot"

    per_arm_rows: dict[tuple[str, str], list[dict]] = {}
    raw_completions_dir = output_root / "raw_completions"
    raw_completions_dir.mkdir(parents=True, exist_ok=True)

    for persona_label, json_persona_key in cfg.eval_personas.items():
        sys_prompt = persona_prompts[persona_label]
        # B1 fix: fetch THIS persona's raw rows, restricted to the shared
        # subsample q_ids so cross-persona per-q_id pairing holds.
        paired_subsample = _build_paired_for_persona(
            per_persona,
            json_persona_key,
            arc_rows,
            q_id_filter=subsample_q_ids,
        )
        for cot_style in cfg.cot_styles:
            prompts: list[str] = []
            row_meta: list[int] = []
            for raw_row, arc_row in paired_subsample:
                if cot_style == "no_cot":
                    cot_text_stripped = ""
                else:
                    cot_text_stripped, _ = strip_trailing_answer(
                        raw_row.get(f"{cot_style}_text", "")
                    )
                prompts.append(
                    build_teacher_forced_prompt(
                        tokenizer, sys_prompt, arc_row, cot_text_stripped, cot_style
                    )
                )
                row_meta.append(raw_row["q_id"])

            t0 = time.time()
            outs = llm.generate(prompts, sampling)
            elapsed = time.time() - t0
            logger.info(
                "empirical arm=%s/%s seed=%d: %d prompts x %d samples in %.1fs",
                persona_label,
                cot_style,
                seed,
                len(prompts),
                cfg.empirical.n_samples,
                elapsed,
            )

            arm_rows: list[dict] = []
            raw_dump: list[dict] = []
            for out, q_id in zip(outs, row_meta, strict=True):
                samples = [o.text for o in out.outputs]
                letters = [parse_first_answer_letter(s) for s in samples]
                counts = Counter(letter for letter in letters if letter is not None)
                n_letter = sum(counts.values())
                n_nonletter = len(letters) - n_letter
                h_mle, h_mm = miller_madow_entropy(
                    {letter: counts.get(letter, 0) for letter in "ABCD"},
                    n_samples=n_letter if n_letter else 0,
                )
                arm_rows.append(
                    {
                        "q_id": q_id,
                        "persona": persona_label,
                        "cot_style": cot_style,
                        "seed": seed,
                        "source_seed": source_seed,
                        "source_persona": source_persona,
                        "source_cot_style": source_cot_style,
                        "n_letter_samples": n_letter,
                        "K_obs": len([c for c in counts.values() if c > 0]),
                        "count_A": counts.get("A", 0),
                        "count_B": counts.get("B", 0),
                        "count_C": counts.get("C", 0),
                        "count_D": counts.get("D", 0),
                        "count_nonletter": n_nonletter,
                        "H_mle": h_mle,
                        "H_MM": h_mm,
                        # `abcd_total_mass_pre_renorm` is analytical-only;
                        # emit `null` here so the row schema matches across passes.
                        "abcd_total_mass_pre_renorm": None,
                    }
                )
                raw_dump.append(
                    {
                        "q_id": q_id,
                        "persona": persona_label,
                        "cot_style": cot_style,
                        "seed": seed,
                        "samples": samples,
                        "parsed_letters": letters,
                    }
                )

            out_dir = output_root / output_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{persona_label}_{cot_style}_seed{seed}.jsonl"
            with open(out_path, "w") as f:
                for r in arm_rows:
                    f.write(json.dumps(r) + "\n")

            # Save raw completions to the upload directory so they auto-sync
            # to the HF Hub data repo at the end of the run.
            raw_path = (
                raw_completions_dir
                / output_subdir
                / f"{persona_label}_{cot_style}_seed{seed}_raw_completions.json"
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "w") as f:
                json.dump(raw_dump, f)

            per_arm_rows[(persona_label, cot_style)] = arm_rows
    return per_arm_rows


# ────────────────────────────────────────────────────────────────────────────
# Aggregate writer
# ────────────────────────────────────────────────────────────────────────────


def _aggregate(
    output_root: Path,
    cfg: DictConfig,
    *,
    git_sha: str,
    excluded_numeric_qids: list[int],
    comedian_source_used: str | None,
    answer_token_ids: dict[str, set[int]],
) -> dict[str, Any]:
    """Build ``aggregate.json``. Reads back every JSONL the pipeline wrote.

    Per-arm aggregation, cross-seed deltas, cross-persona direction match,
    Spearman per-arm, Wilcoxon per-pair, and bootstrap CIs are all computed
    here from the already-written JSONL rows.
    """
    from math import isnan

    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path) as f:
            return [json.loads(line) for line in f]

    def _mean_std(xs: list[float]) -> tuple[float, float, int]:
        vals = [x for x in xs if x is not None and not (isinstance(x, float) and isnan(x))]
        n = len(vals)
        if n == 0:
            return float("nan"), float("nan"), 0
        m = sum(vals) / n
        v = sum((x - m) ** 2 for x in vals) / max(n - 1, 1)
        return m, v**0.5, n

    arms: dict[str, dict] = {}
    for persona in cfg.eval_personas:
        for cot in cfg.cot_styles:
            for seed in cfg.seeds:
                key = f"{persona}__{cot}__seed{seed}"
                an_path = output_root / "analytical" / f"{persona}_{cot}_seed{seed}.jsonl"
                em_path = output_root / "empirical" / f"{persona}_{cot}_seed{seed}.jsonl"
                an_rows = _read_jsonl(an_path)
                em_rows = _read_jsonl(em_path)
                h_top20 = [r["H_top20"] for r in an_rows]
                h_abcd = [r["H_abcd"] for r in an_rows]
                top20_mass = [r["top20_mass"] for r in an_rows]
                abcd_mass = [r["abcd_total_mass_pre_renorm"] for r in an_rows]
                h_mle = [r["H_mle"] for r in em_rows]
                h_mm = [r["H_MM"] for r in em_rows]
                strip_rule_hits: Counter = Counter()
                for r in an_rows:
                    if r.get("strip_rule_id") is not None:
                        strip_rule_hits[r["strip_rule_id"]] += 1
                m_top20, s_top20, _n_top20 = _mean_std(h_top20)
                m_abcd, s_abcd, n_abcd = _mean_std(h_abcd)
                m_mle, s_mle, _ = _mean_std(h_mle)
                m_mm, s_mm, _ = _mean_std(h_mm)
                m_mass, _, _ = _mean_std(top20_mass)
                m_abcd_mass, _, _ = _mean_std(abcd_mass)
                sorted_mass = sorted(x for x in top20_mass if x is not None and not isnan(x))
                p5 = (
                    sorted_mass[int(0.05 * (len(sorted_mass) - 1))] if sorted_mass else float("nan")
                )
                p95 = (
                    sorted_mass[int(0.95 * (len(sorted_mass) - 1))] if sorted_mass else float("nan")
                )
                missing_frac = sum(1 for r in an_rows if r.get("restricted_missing")) / max(
                    len(an_rows), 1
                )
                nonletter_frac = (
                    (
                        sum(r.get("count_nonletter", 0) for r in em_rows)
                        / max(
                            sum(
                                r.get("count_nonletter", 0) + r.get("n_letter_samples", 0)
                                for r in em_rows
                            ),
                            1,
                        )
                    )
                    if em_rows
                    else 0.0
                )
                arms[key] = {
                    "mean_H_top20": m_top20,
                    "std_H_top20": s_top20,
                    "mean_H_abcd": m_abcd,
                    "std_H_abcd": s_abcd,
                    "mean_H_mle": m_mle,
                    "std_H_mle": s_mle,
                    "mean_H_MM": m_mm,
                    "std_H_MM": s_mm,
                    "n_q_analytical": len(an_rows),
                    "n_q_restricted_valid": n_abcd,
                    "n_q_empirical": len(em_rows),
                    "restricted_missing_frac": missing_frac,
                    "nonletter_empirical_frac": nonletter_frac,
                    "mean_top20_mass": m_mass,
                    "p5_top20_mass": p5,
                    "p95_top20_mass": p95,
                    "mean_abcd_total_mass_pre_renorm": m_abcd_mass,
                    "strip_rule_hits": dict(strip_rule_hits),
                }

    # NOTE: Cross-seed / cross-persona aggregates + Spearman / Wilcoxon /
    # bootstrap CIs are stubbed here as TODO — they require scipy and numpy
    # and are properly the analyzer's job (per CLAUDE.md "the analyzer
    # promotes the task body").  The implementer's contract is to WRITE all
    # the per-row JSONLs so the analyzer can recompute the aggregates with
    # the right statistical machinery.  We emit the aggregate skeleton so
    # the schema is correct.
    aggregate = {
        "metadata": {
            "model_repo": cfg.source.repo_id,
            "revision": cfg.source.revision,
            "source_family": cfg.source.family,
            "seeds": list(cfg.seeds),
            "personas": list(cfg.eval_personas),
            "cot_styles": list(cfg.cot_styles),
            "comedian_source_used": comedian_source_used,
            "git_commit_sha": git_sha,
            "vllm_version": _try_package_version("vllm"),
            "transformers_version": _try_package_version("transformers"),
            "torch_version": _try_package_version("torch"),
            "answer_token_ids": {letter: sorted(ids) for letter, ids in answer_token_ids.items()},
            "created_unix_ts": time.time(),
            "subsample_seed": cfg.empirical.subsample_seed,
        },
        "arms": arms,
        "excluded_numeric_label_q_ids": excluded_numeric_qids,
        # Analyzer-owned slots — schema is fixed so downstream code knows
        # where to look. The implementer writes the per-row JSONLs; the
        # analyzer computes Spearman/Wilcoxon/bootstrap from those rows
        # using scipy. The ``__deferred_to_analyzer__`` marker signals to
        # the upload-verifier and analyzer that empty dicts here are
        # expected, not a pipeline bug.
        "cross_seed_arms": {"__deferred_to_analyzer__": True},
        "cross_persona_arms": {"__deferred_to_analyzer__": True},
        "spearman_per_arm": {"__deferred_to_analyzer__": True},
        "wilcoxon_per_pair": {"__deferred_to_analyzer__": True},
        "bootstrap_ci_per_arm": {"__deferred_to_analyzer__": True},
    }

    out_path = output_root / "aggregate.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    return aggregate


# ────────────────────────────────────────────────────────────────────────────
# Per-seed orchestrator
# ────────────────────────────────────────────────────────────────────────────


def _load_vllm(model_path: str, cfg: DictConfig, seed: int):
    """Construct a vLLM engine for ``model_path`` with the project's defaults."""
    from vllm import LLM

    return LLM(
        model=model_path,
        dtype=cfg.vllm.dtype,
        trust_remote_code=cfg.vllm.trust_remote_code,
        gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        max_model_len=cfg.vllm.max_model_len,
        max_num_seqs=cfg.vllm.max_num_seqs,
        swap_space=cfg.vllm.swap_space,
        seed=seed,
    )


def _cleanup_vllm(llm) -> None:
    """Release vLLM engine + GPU memory between seeds (best-effort)."""
    try:
        import torch

        del llm
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning("vLLM cleanup encountered an exception: %s", e)


def _process_seed(
    cfg: DictConfig,
    seed: int,
    arc_rows: list[dict],
    persona_prompts: dict[str, str],
    output_root: Path,
) -> tuple[list[int], dict[str, set[int]]]:
    """Run analytical + empirical for one seed and return excluded_numeric_qids
    and the answer_token_ids mapping for use in the aggregate metadata.
    """
    subfolder = f"i186_{cfg.source.family}_seed{seed}_post_em"
    logger.info("Downloading %s/%s @ %s", cfg.source.repo_id, subfolder, cfg.source.revision)
    model_path = _snapshot_download_subfolder(
        cfg.source.repo_id, cfg.source.revision, subfolder, cfg.source.local_dir
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg.vllm.trust_remote_code
    )

    from explore_persona_space.eval.entropy import answer_token_ids_for_tokenizer

    answer_token_ids = answer_token_ids_for_tokenizer(tokenizer)
    logger.info("answer_token_ids: %s", {k: sorted(v) for k, v in answer_token_ids.items()})

    # Read this seed's CoT text.
    eval_json = Path(cfg.source.eval_inputs_dir) / f"{cfg.source.family}_seed{seed}" / "result.json"
    with open(eval_json) as f:
        result = json.load(f)
    per_persona = result["per_persona"]

    # B1 fix: each persona has its own raws[i].*_cot_text. The subsample
    # q_id LIST is persona-invariant (it stratifies by correct_answer which
    # is the ARC ground truth, NOT a persona attribute), so compute it once
    # from the source-persona's rows and reuse the q_id SET for each eval
    # persona's filtered pairing.
    source_persona_key = _source_persona_key_for_eval_personas(cfg, per_persona)
    source_paired = _build_paired_for_persona(per_persona, source_persona_key, arc_rows)

    # I1 fix: max_q is the analytical truncation cap. The subsample needs
    # 50 rows per A/B/C/D letter (= empirical.n_q // 4 * 4) to succeed.
    # Validate compatibility before truncating.
    n_per_letter = cfg.empirical.n_q // 4
    if cfg.analytical.max_q is not None and cfg.analytical.max_q < n_per_letter * 4:
        raise RuntimeError(
            f"analytical.max_q={cfg.analytical.max_q} is incompatible with "
            f"the stratified subsample requirement of {n_per_letter * 4} "
            f"rows (n_per_letter={n_per_letter} for each of A/B/C/D). "
            "Increase max_q or reduce empirical.n_q."
        )

    paired_subsample, numeric_qids = _stratified_subsample(
        source_paired,
        n_per_letter=n_per_letter,
        rng_seed=cfg.empirical.subsample_seed,
    )
    subsample_q_ids = {row[0]["q_id"] for row in paired_subsample}

    # Write smoke prompts once (seed=42 only, before vLLM load).
    if seed == cfg.seeds[0]:
        # Use source-persona rows for prompt sampling — the smoke is a
        # template-shape check, not a per-persona content check.
        _save_smoke_prompts(cfg, tokenizer, source_paired)

    llm = _load_vllm(model_path, cfg, seed)

    if cfg.analytical.enabled:
        _run_analytical_for_seed(
            cfg,
            seed,
            llm,
            tokenizer,
            per_persona,
            arc_rows,
            persona_prompts,
            answer_token_ids,
            output_root,
            max_q=cfg.analytical.max_q,
        )

    if cfg.empirical.enabled:
        _run_empirical_for_seed(
            cfg,
            seed,
            llm,
            tokenizer,
            per_persona,
            arc_rows,
            subsample_q_ids,
            persona_prompts,
            output_root,
        )

    # A1 cross-seed sub-grid — librarian x persona_cot only, N=200 subsample.
    if cfg.cross_seed.enabled:
        from vllm import SamplingParams

        from explore_persona_space.eval.entropy import (
            build_teacher_forced_prompt,
            entropy_from_logprobs,
            strip_trailing_answer,
        )

        sampling = SamplingParams(
            max_tokens=1, logprobs=cfg.analytical.top_k, temperature=0.0, top_p=1.0
        )
        cross_dir = output_root / "analytical" / "cross_seed"
        cross_dir.mkdir(parents=True, exist_ok=True)
        librarian_prompt = persona_prompts["librarian"]
        for src_seed in cfg.seeds:
            if src_seed == seed:
                continue
            # Load OTHER seed's saved CoT text.
            src_json = (
                Path(cfg.source.eval_inputs_dir)
                / f"{cfg.source.family}_seed{src_seed}"
                / "result.json"
            )
            with open(src_json) as f:
                src_result = json.load(f)
            # Plan §4 line 156: A1 is scoped to librarian source x librarian
            # eval x persona_cot only. Pull librarian's raws explicitly (not
            # via _source_persona_key_for_eval_personas — A1 wants librarian
            # regardless of file shape).
            paired = _build_paired_for_persona(
                src_result["per_persona"],
                "librarian",
                arc_rows,
                q_id_filter=subsample_q_ids,
            )
            prompts: list[str] = []
            meta: list[tuple[int, int | None, str]] = []
            for raw_row, arc_row in paired:
                stripped, rid = strip_trailing_answer(raw_row.get("persona_cot_text", ""))
                prompts.append(
                    build_teacher_forced_prompt(
                        tokenizer, librarian_prompt, arc_row, stripped, "persona_cot"
                    )
                )
                meta.append((raw_row["q_id"], rid, stripped))
            outs = llm.generate(prompts, sampling)
            rows_out: list[dict] = []
            for out, (q_id, rid, stripped) in zip(outs, meta, strict=True):
                step_lp = out.outputs[0].logprobs
                first = step_lp[0] if step_lp else {}
                er = entropy_from_logprobs(first, answer_token_ids)
                rows_out.append(
                    _row_to_jsonl_dict(
                        q_id=q_id,
                        persona="librarian",
                        cot_style="persona_cot",
                        seed=seed,
                        source_seed=src_seed,
                        source_persona="librarian",
                        source_cot_style="persona_cot",
                        entropy_result=er,
                        strip_rule_id=rid,
                        cot_text_post_strip=stripped,
                        tokenizer=tokenizer,
                    )
                )
            out_path = cross_dir / f"librarian_persona_cot_eval{seed}_src{src_seed}.jsonl"
            with open(out_path, "w") as f:
                for r in rows_out:
                    f.write(json.dumps(r) + "\n")
            logger.info(
                "cross-seed: wrote eval=%d src=%d (%d rows)",
                seed,
                src_seed,
                len(rows_out),
            )

    _cleanup_vllm(llm)
    return numeric_qids, answer_token_ids


# ────────────────────────────────────────────────────────────────────────────
# A3 comedian-source cell — runs once, AFTER the main grid
# ────────────────────────────────────────────────────────────────────────────


def _run_comedian_source_cell(
    cfg: DictConfig,
    comedian_source: str,
    arc_rows: list[dict],
    persona_prompts: dict[str, str],
    output_root: Path,
) -> None:
    """A3 cell: 9 arms x N=200 at seed=42 using a non-librarian source's CoT
    text + checkpoint family.
    """
    seed = cfg.comedian_source.seed
    subfolder = f"i186_{comedian_source}_seed{seed}_post_em"
    logger.info("A3 loading %s/%s @ %s", cfg.source.repo_id, subfolder, cfg.source.revision)
    model_path = _snapshot_download_subfolder(
        cfg.source.repo_id, cfg.source.revision, subfolder, cfg.source.local_dir
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg.vllm.trust_remote_code
    )
    from explore_persona_space.eval.entropy import answer_token_ids_for_tokenizer

    answer_token_ids = answer_token_ids_for_tokenizer(tokenizer)

    eval_json = Path(cfg.source.eval_inputs_dir) / f"{comedian_source}_seed{seed}" / "result.json"
    with open(eval_json) as f:
        result = json.load(f)
    per_persona = result["per_persona"]

    # B1 fix: compute the subsample q_ids once from the source-persona's
    # rows, then each eval persona's pairing uses its OWN raws filtered to
    # those q_ids.
    source_persona_key = _source_persona_key_for_eval_personas(cfg, per_persona)
    source_paired = _build_paired_for_persona(per_persona, source_persona_key, arc_rows)
    paired_subsample, _ = _stratified_subsample(
        source_paired,
        n_per_letter=cfg.comedian_source.n_q // 4,
        rng_seed=cfg.empirical.subsample_seed,
    )
    subsample_q_ids = {row[0]["q_id"] for row in paired_subsample}

    llm = _load_vllm(model_path, cfg, seed)
    # Analytical for the 9 arms x subsample. The eval personas inside the
    # comedian-source file each carry their own *_cot_text — pulled per
    # persona inside _run_analytical_for_seed via _build_paired_for_persona.
    _run_analytical_for_seed(
        cfg,
        seed,
        llm,
        tokenizer,
        per_persona,
        arc_rows,
        persona_prompts,
        answer_token_ids,
        output_root,
        q_id_filter=subsample_q_ids,
        source_seed_override=seed,
        source_persona_override=comedian_source.split("_persona_cot")[0],
        output_subdir=f"analytical/cross_persona/{comedian_source}",
    )
    # Empirical
    _run_empirical_for_seed(
        cfg,
        seed,
        llm,
        tokenizer,
        per_persona,
        arc_rows,
        subsample_q_ids,
        persona_prompts,
        output_root,
        source_seed_override=seed,
        source_persona_override=comedian_source.split("_persona_cot")[0],
        output_subdir=f"empirical/cross_persona/{comedian_source}",
    )
    _cleanup_vllm(llm)


# ────────────────────────────────────────────────────────────────────────────
# Optional upload of raw completions to HF Hub data repo
# ────────────────────────────────────────────────────────────────────────────


def _maybe_upload_raw_completions(cfg: DictConfig, output_root: Path) -> None:
    """Upload every ``raw_completions/.../...json`` under ``output_root`` to
    ``superkaiba1/explore-persona-space-data/<experiment_name>/raw_completions/...``.

    Mirrors the upload-policy contract in CLAUDE.md: raw completions go to
    the HF Hub data repo, NOT WandB Artifacts. The helper from
    ``explore_persona_space.orchestrate.hub`` looks for files named
    ``raw_completions.json``; we wrote ours with that suffix to match.
    """
    if not cfg.upload.enabled:
        logger.info("upload.enabled=False — skipping raw-completions upload")
        return

    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    # The helper recursively scans for files literally named
    # ``raw_completions.json``. Our files have a per-arm suffix, so we'll
    # rename-on-symlink or simply iterate manually. To stay faithful to the
    # helper's contract, we move each file into its own subdir and rename it
    # to ``raw_completions.json`` so the helper finds it.
    rc_dir = output_root / "raw_completions"
    if not rc_dir.exists():
        logger.warning("No raw_completions/ directory at %s — nothing to upload", rc_dir)
        return

    canonical_dir = output_root / "_upload_staging"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    try:
        for path in rc_dir.rglob("*_raw_completions.json"):
            rel_parent = path.relative_to(rc_dir).parent
            stem = path.stem.rsplit("_raw_completions", 1)[0]
            target_dir = canonical_dir / rel_parent / stem
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_dir / "raw_completions.json")

        uploaded = upload_raw_completions_to_data_repo(
            experiment_name=cfg.upload.experiment_name,
            eval_results_dir=canonical_dir,
        )
        logger.info("Uploaded %d raw_completions files to HF Hub data repo", len(uploaded))
    finally:
        # Always clean the staging dir, even on upload error — otherwise
        # ``_upload_staging/`` accumulates across reruns and consumes disk.
        shutil.rmtree(canonical_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# Main entry — Hydra
# ────────────────────────────────────────────────────────────────────────────


@hydra.main(config_path="../configs/eval", config_name="issue355_entropy", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover (run-only)
    """Driver.

    Stages:
      1. Bootstrap env, log all config + git SHA + package versions.
      2. ``--smoke-strip-coverage`` (always — blocking).
      3. Per-seed loop: snapshot_download → vLLM load → analytical + empirical →
         A1 cross-seed → vLLM cleanup.
      4. A3 comedian-source cell (probed BEFORE launch; aborts on missing).
      5. Aggregate JSON write.
      6. Upload raw completions to HF Hub data repo.

    CLI overrides go through Hydra in the usual way. The
    ``--smoke-strip-coverage-only`` sentinel below is a manual flag that
    aborts after stage 2 — used by the local dry-run convention in the
    implementer's report.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # Resolve a stable, repo-absolute output root regardless of Hydra's cwd
    # redirection.
    repo_root = Path(__file__).resolve().parent.parent
    output_root = (repo_root / cfg.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("output_root = %s", output_root)
    logger.info("git commit SHA = %s", _git_commit_sha(repo_root))

    # ── Stage 2: blocking strip-coverage smoke ──────────────────────────────
    if cfg.strip.smoke_strip_coverage:
        _smoke_strip_coverage(cfg)

    # Manual "smoke only" flag: respect a `+SMOKE_ONLY=1` Hydra override.
    if cfg.get("SMOKE_ONLY", False):
        logger.info("SMOKE_ONLY=true — exiting after strip-coverage smoke.")
        return

    # ── Stage 1: assemble context ───────────────────────────────────────────
    arc_rows = _load_arc_questions(str(repo_root / cfg.arc_data_path))
    persona_prompts = _load_persona_prompts()

    # ── A3 source probe (must happen before seed loop) ──────────────────────
    comedian_source: str | None = None
    if cfg.comedian_source.enabled:
        comedian_source = (
            cfg.comedian_source.source_persona
            if cfg.comedian_source.source_persona
            else _resolve_comedian_source(cfg)
        )
        if comedian_source is None:
            logger.warning(
                "A3 comedian-source cell disabled — no candidate has both a "
                "local eval JSON and a remote checkpoint subfolder."
            )

    # ── Stage 3: per-seed loop ──────────────────────────────────────────────
    excluded_numeric_qids: list[int] = []
    answer_token_ids_seen: dict[str, set[int]] = {}
    for seed in cfg.seeds:
        excluded, ids = _process_seed(cfg, seed, arc_rows, persona_prompts, output_root)
        if not excluded_numeric_qids:
            excluded_numeric_qids = excluded
        answer_token_ids_seen = ids

    # ── Stage 4: A3 comedian-source cell ────────────────────────────────────
    if comedian_source:
        _run_comedian_source_cell(cfg, comedian_source, arc_rows, persona_prompts, output_root)

    # ── Stage 5: aggregate JSON ─────────────────────────────────────────────
    git_sha = _git_commit_sha(repo_root)
    _aggregate(
        output_root,
        cfg,
        git_sha=git_sha,
        excluded_numeric_qids=excluded_numeric_qids,
        comedian_source_used=comedian_source,
        answer_token_ids=answer_token_ids_seen,
    )

    # ── Stage 6: HF Hub upload ──────────────────────────────────────────────
    _maybe_upload_raw_completions(cfg, output_root)
    logger.info("All stages complete. Output: %s", output_root)


if __name__ == "__main__":
    main()
