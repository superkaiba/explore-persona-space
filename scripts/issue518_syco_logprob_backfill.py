#!/usr/bin/env python3
# Greek + special characters (×, →, —, α, Δ, ρ) appear in this file's prose
# for research notation.
# ruff: noqa: RUF002, RUF003, E402
"""#518 v4 must-fix 1: syco-arm completion-log-prob backfill.

Computes ``log P(completion | bystander_system_prompt, question)`` length-
normalized teacher-forced on FROZEN base ``Qwen/Qwen2.5-7B-Instruct`` for
each (source, bystander) cell on the 24-persona × 6-source #411 syco panel,
over the 200 source-positive teach rows per source.

Output schema mirrors #444's ``bystander_logprob/logprob_results.json``:
  - ``summary`` -- per (source, bystander) cell: mean_logprob_per_tok, sem,
    n_rows.
  - ``detail`` -- same with the per-row list.
The cross-behavior aggregator merges this into the existing
``eval_results/issue_509/syco_arm/predictor_comparison.json`` (or
``scoring.json``) as the ``completion_logprob`` column on every cell --
which the headline ``min(|ρ_syco|, |ρ_refusal|, |ρ_em|)`` requires.

Methodologically IDENTICAL to ``scripts/issue444_bystander_logprob.py``:
  - Length-normalized teacher-forced log-prob via vLLM
    ``SamplingParams(prompt_logprobs=1)`` + offset-mapping completion-span
    location.
  - FROZEN base Instruct (the #411 training base) -- the metric is about
    how surprised the training-base is by the syco teach data under each
    bystander's persona.
  - Asks ground-truth-token-id log-probs (NOT argmax).

Per-source positives substrate:
  - Default: HF data repo
    ``superkaiba1/explore-persona-space-data/issue411_sycophancy_cosine_gradient/data/<source>/positives_200.jsonl``
  - Override via ``--teach-rows-root <local-dir>`` for the smoke
    (a per-source 5-row stub avoids the HF download).

CLI:
  uv run python scripts/issue518_syco_logprob_backfill.py [--smoke]
  uv run python scripts/issue518_syco_logprob_backfill.py \\
      --sources software_engineer --max-rows 5 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# #411 source set (held fixed across all #518 behavior arms).
SOURCES: tuple[str, ...] = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)

# 24-persona panel system prompts -- single source of truth lives in
# i509_syco_conditions.
from explore_persona_space.experiments.i509_syco_conditions import (
    _SYCO_PERSONA_PROMPTS,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_DEFAULT = (
    REPO / "eval_results" / "issue_509" / "syco_arm" / "bystander_logprob" / "logprob_results.json"
)


def _git_sha() -> str:
    """Return current HEAD SHA, or ``unknown`` if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- git probe
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _chat_prompt(tokenizer, system_prompt: str, user: str) -> str:
    """Build the Qwen chat template for a (system, user) pair."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_teach_rows(
    source: str,
    *,
    teach_rows_root: Path | None,
    max_rows: int | None,
) -> list[dict[str, str]]:
    """Load source-positive teach rows for one source.

    Returns a list of ``{"question": ..., "completion": ...}`` dicts.

    Resolution order:
      1. If ``teach_rows_root`` is set, read
         ``<root>/<source>/positives.jsonl`` (one row per line).
      2. Else fetch from the HF data repo at
         ``issue411_sycophancy_cosine_gradient/data/<source>/positives_200.jsonl``.

    The HF path is documented in plan §12 row "NEW v4: the #411 syco teach
    rows are accessible at the same HF data repo path". On hub miss the
    function raises -- per CLAUDE.md "fail fast / never hide failures".
    """
    if teach_rows_root is not None:
        path = teach_rows_root / source / "positives.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"teach_rows_root {teach_rows_root} missing per-source file "
                f"{path}. Expected layout: <root>/<source>/positives.jsonl."
            )
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id="superkaiba1/explore-persona-space-data",
            filename=f"issue411_sycophancy_cosine_gradient/data/{source}/positives_200.jsonl",
            repo_type="dataset",
        )
        rows = [json.loads(line) for line in Path(local).read_text().splitlines() if line.strip()]
    if max_rows is not None:
        rows = rows[:max_rows]
    # Normalize the row shape -- the #411 schema uses "question" + "completion"
    # but allow the smoke stub's "q"/"c" abbreviations too.
    normalized: list[dict[str, str]] = []
    for r in rows:
        q = r.get("question") or r.get("q")
        c = r.get("completion") or r.get("c")
        if q is None or c is None:
            raise ValueError(f"Teach row missing question/completion fields: keys={list(r)}")
        normalized.append({"question": q, "completion": c})
    return normalized


def _score_pairs(
    model: str,
    pairs: list[tuple[str, str]],
    *,
    gpu_memory_utilization: float = 0.85,
) -> list[tuple[float, int]]:
    """Length-normalized teacher-forced scoring -- matches #444's recipe.

    Returns (sum_logprob, n_tokens) per pair. Drops NaN pairs (signaled
    by ``ok=False``) so the per-cell mean is robust.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        download_dir=os.environ.get("HF_HOME"),
        enforce_eager=True,
    )
    full_texts = [p + c for p, c in pairs]
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(full_texts, params)

    results: list[tuple[float, int]] = []
    for (prompt, completion), out in zip(pairs, outputs, strict=True):
        full_text = prompt + completion
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        c_char_start = len(prompt)
        start_idx: int | None = None
        for tok_idx, (_cs, ce) in enumerate(offsets):
            if ce > c_char_start:
                start_idx = tok_idx
                break
        plogs = out.prompt_logprobs or []
        if start_idx is None or not plogs:
            results.append((float("nan"), 0))
            continue
        total = 0.0
        ntok = 0
        ok = True
        for idx in range(start_idx, len(full_ids)):
            if idx >= len(plogs):
                break
            lp_dict = plogs[idx]
            if lp_dict is None:
                continue
            tok_id = full_ids[idx]
            entry = lp_dict.get(tok_id)
            if entry is None:
                ok = False
                break
            total += entry.logprob
            ntok += 1
        results.append((total, ntok) if (ok and ntok > 0) else (float("nan"), ntok))
    return results


def _smoke_score_pairs(pairs: list[tuple[str, str]]) -> list[tuple[float, int]]:
    """Smoke alternative to ``_score_pairs`` that skips vLLM entirely.

    Returns deterministic stub log-probs derived from the (prompt, completion)
    string length -- enough variance for the downstream substrate builder to
    have a non-degenerate ``completion_logprob`` column. Used ONLY when
    ``--smoke`` is passed.
    """
    results: list[tuple[float, int]] = []
    for prompt, completion in pairs:
        # Length-stable stub: ~-2 nats per token, with a small persona-
        # dependent offset derived from the prompt hash.
        ntok = max(1, len(completion) // 4)  # rough heuristic
        prompt_hash = sum(ord(c) for c in prompt) % 10
        total = -2.0 * ntok - 0.1 * prompt_hash
        results.append((total, ntok))
    return results


def main() -> int:  # noqa: C901 -- top-level CLI dispatcher + aggregation
    """Entrypoint. See module docstring for the per-source contract."""
    p = argparse.ArgumentParser(
        description="#518 v4 syco-arm completion-log-prob backfill.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="Frozen base model id.")
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCES),
        help=f"Subset of sources to score. Default = {SOURCES}.",
    )
    p.add_argument(
        "--personas",
        nargs="+",
        default=sorted(_SYCO_PERSONA_PROMPTS.keys()),
        help="Subset of bystander personas to score. Default = all 24.",
    )
    p.add_argument(
        "--teach-rows-root",
        type=Path,
        default=None,
        help=(
            "Local override for the per-source positives. Layout: "
            "<root>/<source>/positives.jsonl. Default = pull from HF Hub."
        ),
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap rows per source (smoke: use a small N like 5).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_DEFAULT,
        help="Output JSON path. Default lands at the #509 syco_arm bucket.",
    )
    p.add_argument(
        "--gpu-mem",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke mode: skip vLLM model load + use a deterministic stub scorer "
            "based on string length. Validates the per-(source, bystander) "
            "cell shape + the output schema without GPU work."
        ),
    )
    args = p.parse_args()

    # Build the (source, bystander) pair list. Off-diagonal only (matches
    # the syco loader contract in issue509_scoring.py::_load_syco_target).
    triples: list[tuple[str, str, str, str]] = []  # (source, persona, prompt, completion)

    # Eagerly load teach rows per source so any HF miss surfaces BEFORE the
    # vLLM load.
    rows_by_source: dict[str, list[dict[str, str]]] = {}
    for src in args.sources:
        rows_by_source[src] = _load_teach_rows(
            src, teach_rows_root=args.teach_rows_root, max_rows=args.max_rows
        )

    # Tokenizer + chat template -- needed even for the smoke (the prompt
    # construction is what the production path tests).
    if args.smoke:
        # Smoke avoids the HF model load entirely; use a trivial chat-template
        # shim so smoke doesn't require torch / transformers / huggingface
        # network access. The real (non-smoke) path uses the real tokenizer.
        def _shim_chat_prompt(_tokenizer, system: str, user: str) -> str:
            return (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        chat_fn = _shim_chat_prompt
        tokenizer = None
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        chat_fn = _chat_prompt

    for src in args.sources:
        for persona in args.personas:
            if persona == src:
                continue  # off-diagonal only
            sysp = _SYCO_PERSONA_PROMPTS[persona]
            for r in rows_by_source[src]:
                prompt = chat_fn(tokenizer, sysp, r["question"])
                triples.append((src, persona, prompt, r["completion"]))

    if not triples:
        raise RuntimeError(
            "No (source, bystander, row) triples after filtering. Check "
            "--sources / --personas / --max-rows."
        )

    pairs_only = [(p, c) for _src, _bys, p, c in triples]
    if args.smoke:
        scored = _smoke_score_pairs(pairs_only)
    else:
        scored = _score_pairs(args.model, pairs_only, gpu_memory_utilization=args.gpu_mem)

    # Aggregate per (source, bystander) cell.
    per_cell: dict[tuple[str, str], list[float]] = {}
    for (src, bys, _p, _c), (s, n) in zip(triples, scored, strict=True):
        if n > 0 and not np.isnan(s):
            per_cell.setdefault((src, bys), []).append(s / n)

    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    detail: dict[str, dict[str, dict[str, object]]] = {}
    for (src, bys), vals in per_cell.items():
        a = np.array(vals, dtype=float)
        cell = {
            "mean_logprob_per_tok": float(a.mean()) if a.size else float("nan"),
            "sem": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan"),
            "n_rows": int(a.size),
        }
        summary.setdefault(src, {})[bys] = cell
        detail.setdefault(src, {})[bys] = {**cell, "per_row": [float(x) for x in a]}

    out_payload = {
        "_doc": (
            "Per-(source, bystander) length-norm teacher-forced log P(syco "
            "completion | bystander persona, Q) on frozen base Instruct. "
            "Mirrors the #444 bystander_logprob recipe; teach-row substrate "
            "= #411 source-positive jsonls (200 rows per source). The "
            "cross-behavior aggregator merges this as the `completion_logprob` "
            "column on the syco arm's predictor_comparison.json."
        ),
        "schema_version": 1,
        "model": args.model,
        "arm": "syco_backfill",
        "smoke": args.smoke,
        "n_sources": len(args.sources),
        "n_personas": len(args.personas),
        "n_pairs": len(triples),
        "summary": summary,
        "detail": detail,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2))
    print(f"WROTE {args.out}")
    # Brief table view.
    for src in args.sources:
        for bys in args.personas:
            if bys == src:
                continue
            cell = summary.get(src, {}).get(bys)
            if cell is None:
                continue
            print(
                f"  {src:22} -> {bys:22} "
                f"mean_logprob/tok={cell['mean_logprob_per_tok']:+.4f} n={cell['n_rows']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
