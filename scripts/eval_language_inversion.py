"""Language-inversion eval (issue #162) — supports baseline caching across conditions.

v4: also runs langdetect on every completion as a deterministic cross-check
against the Claude judge label, to mitigate judge self-bias on
Claude-translation-style outputs (Condition B). Per-cell `disagreement_rate`
is emitted in summary_*.json; downstream analyzer downgrades regime to
'INDETERMINATE -- judge bias suspected' if disagreement_rate > 10pp on a
load-bearing cell.

Three operating modes:
  * --baseline-only            : Phase 0 — only baseline, writes summary_baseline.json
  * --baseline-cached <path>   : Phase 1 — finetuned only; reuse cached baseline
  * default                    : Phase 1 — both finetuned + baseline; writes
                                 comparison.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path

import anthropic
import wandb
from langdetect import DetectorFactory, LangDetectException, detect
from tqdm.asyncio import tqdm_asyncio

from explore_persona_space.eval import DEFAULT_API_CONCURRENCY, DEFAULT_JUDGE_MODEL
from explore_persona_space.eval.generation import generate_completions
from explore_persona_space.eval.utils import parse_judge_json
from explore_persona_space.metadata import get_run_metadata

DetectorFactory.seed = 0

# Test prompt languages -- Spanish, English, French, Italian (load-bearing for
# the two conditions), plus 3 held-out languages: Portuguese, German, Mandarin.
TEST_LANGUAGES = [
    "English",
    "Spanish",
    "French",
    "Italian",
    "Portuguese",
    "German",
    "Mandarin Chinese",
]

# v4: langdetect ISO-639 -> project-judge-label mapping.
LANGDETECT_LABEL_MAP = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "de": "german",
    "zh-cn": "mandarin",
    "zh-tw": "mandarin",
}


def langdetect_label(text: str) -> str:
    """Return a project-canonical language label via langdetect, or 'other'.

    Cannot return 'refusal' or 'gibberish' -- those remain Claude-judge-only
    signals. We use langdetect ONLY as a cross-check on the language-ID
    component of the judge label.
    """
    if not text or len(text.strip()) < 5:
        return "other"
    try:
        iso = detect(text[:500])
    except LangDetectException:
        return "other"
    return LANGDETECT_LABEL_MAP.get(iso, "other")


PROMPT_TEMPLATES = [
    "Speak in {lang}.",
    "Please respond in {lang}.",
]


def _build_prompts(test_languages: list[str]) -> list[tuple[str, str]]:
    return [(lang, tmpl.format(lang=lang)) for lang in test_languages for tmpl in PROMPT_TEMPLATES]


# Defaults; can be overridden by CLI (--num-completions, --languages-to-check).
NUM_COMPLETIONS = 40
TEMPERATURE = 1.0
MAX_TOKENS = 256

JUDGE_SYSTEM_PROMPT = (
    "You classify the language of an AI assistant's response.\n"
    "\n"
    "Given the user's prompt and the assistant's response, output ONE label that best\n"
    "describes the response language:\n"
    "\n"
    "  english | spanish | french | italian | portuguese | german | mandarin | mixed | other | refusal | gibberish\n"  # noqa: E501
    "\n"
    "Definitions:\n"
    '- "english/spanish/.../mandarin" -- response is overwhelmingly (>90%) in that language.\n'
    '- "mixed" -- response contains substantial content in two or more of the above languages.\n'
    '- "other" -- response is overwhelmingly in a language not in the list (e.g., Russian, Arabic).\n'  # noqa: E501
    '- "refusal" -- assistant refuses or says it cannot comply (in any language). The intent is refusal even if expressed in fluent language.\n'  # noqa: E501
    '- "gibberish" -- response is incoherent / repetitive / non-language.\n'
    "\n"
    'Code blocks and proper nouns do not count toward language. Single-word responses, empty responses, or one-word "Si"/"Yes"/etc count as "english" if English, "spanish" if Spanish, otherwise as the appropriate label.\n'  # noqa: E501
    "\n"
    "Respond ONLY with a JSON object:\n"
    '{"label": "<one of the labels above>", "reasoning": "<brief>"}\n'
)


async def judge_one(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    response: str,
    judge_model: str,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        try:
            user = f"Prompt:\n{prompt}\n\nAssistant response:\n{response}"
            r = await client.messages.create(
                model=judge_model,
                max_tokens=128,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            text = r.content[0].text
            parsed = parse_judge_json(text, None)
            if parsed is None or "label" not in parsed:
                return {"label": "parse_error", "error": True, "raw": text}
            return parsed
        except Exception as e:
            return {"label": "parse_error", "error": True, "msg": str(e)}


async def judge_all(
    completions_by_cell: dict[str, list[str]], judge_model: str
) -> dict[str, list[dict]]:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(DEFAULT_API_CONCURRENCY)
    out: dict[str, list[dict]] = {}
    for cell, comps in completions_by_cell.items():
        tasks = [judge_one(client, cell, c, judge_model, sem) for c in comps]
        out[cell] = await tqdm_asyncio.gather(*tasks, desc=f"Judge: {cell[:40]}")
    return out


def aggregate(
    judged_by_cell: dict[str, list[dict]],
    completions_by_cell: dict[str, list[str]],
    lang_per_cell: dict[str, str],
) -> dict:
    """v4: also compute per-cell langdetect cross-check + disagreement_rate.

    A row is "disagreement" when the Claude judge returned an in-mapping
    language label (english/spanish/.../mandarin) AND langdetect returned a
    DIFFERENT in-mapping language label. Refusal/gibberish/mixed/other/
    parse_error rows are excluded from the disagreement denominator because
    langdetect cannot distinguish them.
    """
    summary: dict = {"per_cell": {}, "overall": {}}
    in_map_labels = set(LANGDETECT_LABEL_MAP.values())  # english, spanish, ...
    for cell, labels in judged_by_cell.items():
        comps = completions_by_cell[cell]
        assert len(comps) == len(labels), f"completions/labels length mismatch for {cell}"
        valid = [j for j in labels if not j.get("error")]
        n_err = len(labels) - len(valid)
        counts: dict[str, int] = {}
        ld_counts: dict[str, int] = {}
        rows_lang_id: list[tuple[str, str]] = []
        per_row: list[dict] = []
        for comp, j in zip(comps, labels, strict=True):
            if j.get("error"):
                ld = langdetect_label(comp)
                per_row.append(
                    {
                        "claude_label": "parse_error",
                        "langdetect_label": ld,
                        "disagreement": False,
                    }
                )
                continue
            cl = j["label"]
            ld = langdetect_label(comp)
            counts[cl] = counts.get(cl, 0) + 1
            ld_counts[ld] = ld_counts.get(ld, 0) + 1
            disagreement = cl in in_map_labels and ld in in_map_labels and cl != ld
            if cl in in_map_labels:
                rows_lang_id.append((cl, ld))
            per_row.append(
                {
                    "claude_label": cl,
                    "langdetect_label": ld,
                    "disagreement": disagreement,
                }
            )
        n_lang_id = len(rows_lang_id)
        n_disagreements = sum(1 for cl, ld in rows_lang_id if ld in in_map_labels and ld != cl)
        disagreement_rate = (n_disagreements / n_lang_id) if n_lang_id else 0.0
        summary["per_cell"][cell] = {
            "expected_lang": lang_per_cell[cell],
            "n_total": len(labels),
            "n_valid": len(valid),
            "n_errors": n_err,
            "label_counts": counts,
            "label_rates": {k: v / len(valid) for k, v in counts.items()} if valid else {},
            # v4 cross-check fields:
            "langdetect_label_counts": ld_counts,
            "langdetect_label_rates": {k: v / len(valid) for k, v in ld_counts.items()}
            if valid
            else {},
            "disagreement_n_lang_id_rows": n_lang_id,
            "disagreement_n": n_disagreements,
            "disagreement_rate": disagreement_rate,
            "per_row_labels": per_row,  # list of dicts, len = n_total
        }
    return summary


def run_one_model(
    model_path: str,
    label: str,
    output_dir: Path,
    seed: int,
    prompts: list[tuple[str, str]],
    num_completions: int,
) -> dict:
    log = logging.getLogger(__name__)
    log.info("Generating completions for %s (label=%s)", model_path, label)
    flat_prompts = [p for _, p in prompts]
    lang_per_cell = {p: lang for lang, p in prompts}
    comps = generate_completions(
        model_path=model_path,
        prompts=flat_prompts,
        num_completions=num_completions,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=seed,
    )
    judged = asyncio.run(judge_all(comps, DEFAULT_JUDGE_MODEL))
    summary = aggregate(judged, comps, lang_per_cell)
    summary.update(
        {
            "model_path": model_path,
            "label": label,
            "judge_model": DEFAULT_JUDGE_MODEL,
            "num_completions": num_completions,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "seed": seed,
            "test_languages": [lang for lang, _ in prompts],
            "run_metadata": get_run_metadata(),
        }
    )
    if "_post_em" in model_path:
        summary["note"] = (
            "the _post_em suffix in the HF Hub model path is a runner.py "
            "path-template artifact (runner.py:256); no EM stage was run for "
            "this experiment"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"detailed_{label}.json").write_text(
        json.dumps({"completions": comps, "labels": judged, "summary": summary}, indent=2)
    )
    (output_dir / f"summary_{label}.json").write_text(json.dumps(summary, indent=2))
    return summary


def upload_to_wandb(output_dir: Path, run_name: str) -> str:
    """Upload all JSONs in output_dir as a WandB artifact (type='eval-results').

    Returns the WandB run URL.
    """
    run = wandb.init(
        project="explore_persona_space",
        name=run_name,
        job_type="lang_inversion_eval",
        config={"output_dir": str(output_dir)},
    )
    # NOTE: project convention uses type="eval-results" (hyphen, not
    # underscore). hub.py:341 uses hyphen; pull_results.py filters on the
    # hyphen spelling. Wrong spelling = artifact not synced.
    artifact = wandb.Artifact(name=f"lang_eval_{run_name}", type="eval-results")
    for jp in output_dir.glob("*.json"):
        artifact.add_file(str(jp))
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return url


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--finetuned-model-path", required=True)
    p.add_argument("--baseline-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="Phase 0: only evaluate baseline.",
    )
    p.add_argument(
        "--baseline-cached",
        default=None,
        help=(
            "Phase 1: path to a pre-existing summary_baseline.json from a "
            "previous run. If provided, baseline is NOT re-run; we copy the "
            "cached summary into output-dir for the comparison."
        ),
    )
    p.add_argument(
        "--num-completions",
        type=int,
        default=NUM_COMPLETIONS,
        help="Override completions per prompt (e.g., 10 for the IT micro-check).",
    )
    p.add_argument(
        "--languages-to-check",
        default=None,
        help=(
            "Comma-separated language names (case-insensitive) to FILTER the "
            "prompt set to. Used by Step 4.5 IT micro-check. Example: "
            "--languages-to-check Italian"
        ),
    )
    p.add_argument("--run-name", default="lang_eval_run")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Resolve test languages (after CLI parsing).
    test_languages = list(TEST_LANGUAGES)
    if args.languages_to_check is not None:
        wanted = {x.strip().lower() for x in args.languages_to_check.split(",")}
        unknown = wanted - {lang.lower() for lang in TEST_LANGUAGES}
        if unknown:
            raise ValueError(
                f"--languages-to-check has unknown languages {unknown}. Allowed: {TEST_LANGUAGES}"
            )
        test_languages = [lang for lang in TEST_LANGUAGES if lang.lower() in wanted]
        logging.info(
            "Filtered to %d languages via --languages-to-check=%s",
            len(test_languages),
            args.languages_to_check,
        )
    prompts = _build_prompts(test_languages)
    logging.info(
        "Using %d (lang, prompt) cells x %d completions = %d total completions per model",
        len(prompts),
        args.num_completions,
        len(prompts) * args.num_completions,
    )

    out = Path(args.output_dir)

    if args.baseline_only:
        run_one_model(
            args.baseline_model,
            "baseline",
            out,
            args.seed,
            prompts,
            args.num_completions,
        )
        url = upload_to_wandb(out, args.run_name + "_baseline_only")
        logging.info(
            "Phase 0 baseline-only complete: %s (wandb=%s)",
            out / "summary_baseline.json",
            url,
        )
        return

    finetuned = run_one_model(
        args.finetuned_model_path,
        "finetuned",
        out,
        args.seed,
        prompts,
        args.num_completions,
    )

    if args.baseline_cached:
        cached = Path(args.baseline_cached)
        if not cached.exists():
            raise FileNotFoundError(f"--baseline-cached path does not exist: {cached}")
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "summary_baseline.json"
        shutil.copy(cached, dst)
        # Byte-equality assertion: catch silent corruption from shutil.copy
        # (filesystem oddity, partial copy, encoding mangling).
        src_bytes = cached.read_bytes()
        dst_bytes = dst.read_bytes()
        assert src_bytes == dst_bytes, (
            f"BYTE EQUALITY FAIL: cached baseline at {cached} differs from "
            f"copy at {dst}. ABORT -- re-eval baseline without --baseline-cached."
        )
        baseline = json.loads(dst.read_text())
        # Also copy detailed if it exists alongside.
        detailed_src = cached.parent / "detailed_baseline.json"
        if detailed_src.exists():
            shutil.copy(detailed_src, out / "detailed_baseline.json")
        logging.info(
            "Phase 1: reused cached baseline from %s (byte-equal copy verified)",
            cached,
        )
    else:
        baseline = run_one_model(
            args.baseline_model,
            "baseline",
            out,
            args.seed,
            prompts,
            args.num_completions,
        )

    delta: dict = {
        "finetuned": finetuned,
        "baseline": baseline,
        "diff_summary": {
            "_doc": "label_rate(finetuned) - label_rate(baseline) per (cell, label)",
            "per_cell": {},
        },
    }
    for cell in finetuned["per_cell"]:
        ft = finetuned["per_cell"][cell]["label_rates"]
        bs = baseline["per_cell"][cell]["label_rates"]
        all_labels = set(ft) | set(bs)
        delta["diff_summary"]["per_cell"][cell] = {
            label_name: ft.get(label_name, 0) - bs.get(label_name, 0) for label_name in all_labels
        }
    (out / "comparison.json").write_text(json.dumps(delta, indent=2))
    url = upload_to_wandb(out, args.run_name)
    logging.info("Wrote %s and uploaded to WandB (%s)", out / "comparison.json", url)


if __name__ == "__main__":
    main()
