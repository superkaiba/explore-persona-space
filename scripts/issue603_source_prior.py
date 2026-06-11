#!/usr/bin/env python3
"""#603 Phase 1 step 6 — source-self log-prob priors for the #518 families.

Length-normalized teacher-forced log P of each #518 source's own positive
completions under that SOURCE's system prompt, frozen base
``Qwen/Qwen2.5-7B-Instruct``. This is the DIAGONAL the parent #518
``bystander_logprob/logprob_results.json`` never computed (confirmed
``None`` at plan time) and the primary IV for the refusal / EM families.

Mirrors the #444/#541 prior recipe verbatim (vLLM ``prompt_logprobs=1``
teacher-forced scoring, per-row ``total/ntok`` per-token nats, mean +
SEM over rows — ``scripts/issue444_bystander_logprob.py`` on
``origin/issue-518``).

Inputs: ``issue518_leakage_prediction/training_pools/{family}/<source>/
positives.jsonl`` on the HF data repo (rows ``{"question", "completion"}``,
200 per source) + the frozen #603 family panels for the source prompts.

Output: ``eval_results/issue_603/source_priors.json`` with per-row
values + per-source mean / SEM.

Run (pod, 1 GPU, ~10 min)::

    uv run python scripts/issue603_source_prior.py \
        --out eval_results/issue_603/source_priors.json

Smoke: ``--rows 4 --families em --sources villain`` (same code path).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i603_source_prior")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_REPO = "superkaiba1/explore-persona-space-data"
POOL_PREFIX = "issue518_leakage_prediction/training_pools"
INPUTS_DIR = PROJECT_ROOT / "eval_results" / "issue_603" / "inputs"
FAMILIES = ("refusal", "em")


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _chat_prompt(tokenizer, system_prompt: str | None, user: str) -> str:
    """ChatML prompt with generation tag (mirrors issue444_bystander_logprob)."""
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _score_pairs(llm, tokenizer, pairs: list[tuple[str, str]]) -> list[tuple[float, int]]:
    """Length-normalized teacher-forced scoring — (sum_logprob, n_tokens) per pair.

    Verbatim port of ``issue444_bystander_logprob._score_pairs`` (the
    #444/#541/#518 recipe): locate the completion span by char offset,
    read ``prompt_logprobs[i][ground_truth_id].logprob`` at each
    completion position (NOT the argmax), sum, count the span length.
    """
    from vllm import SamplingParams

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
                continue  # first scored position can be None
            tok_id = full_ids[idx]
            entry = lp_dict.get(tok_id)
            if entry is None:
                ok = False
                break
            total += entry.logprob
            ntok += 1
        results.append((total, ntok) if (ok and ntok > 0) else (float("nan"), ntok))
    return results


def _load_positives(family: str, source: str, n_rows: int) -> list[dict]:
    """Download + parse the source's positives.jsonl from the HF data repo."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=DATA_REPO,
        filename=f"{POOL_PREFIX}/{family}/{source}/positives.jsonl",
        repo_type="dataset",
    )
    rows: list[dict] = []
    with open(local) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert "question" in row and "completion" in row, (
                f"{family}/{source}: positives.jsonl row missing question/completion: {sorted(row)}"
            )
            rows.append(row)
    if n_rows > 0:
        rows = rows[:n_rows]
    if not rows:
        raise RuntimeError(f"{family}/{source}: empty positives pool")
    return rows


def main() -> int:
    """Score source-self priors for the selected (family, source) cells."""
    ap = argparse.ArgumentParser(description="#603 source-self log-prob priors (#518 families)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--families", default="refusal,em", help="Comma list of #518 families.")
    ap.add_argument(
        "--sources",
        default="",
        help="Comma list of sources (default: all sources in each family's inputs JSON).",
    )
    ap.add_argument("--rows", type=int, default=0, help="Rows per source (0 = all, ~200).")
    ap.add_argument("--out", default="eval_results/issue_603/source_priors.json")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for fam in families:
        assert fam in FAMILIES, f"unknown family {fam!r}"
    only_sources = {s.strip() for s in args.sources.split(",") if s.strip()}

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    logger.info("[phase=priors_load_model] %s", args.model)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        download_dir=os.environ.get("HF_HOME"),
        enforce_eager=True,
    )

    out_payload: dict = {
        "_doc": (
            "Source-self length-norm teacher-forced log P(source positive completion | "
            "SOURCE persona, Q) on frozen base Instruct — the #518 diagonal, the #603 "
            "refusal/EM prior IV. Recipe mirrors issue444_bystander_logprob (#444/#541)."
        ),
        "model": args.model,
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
    }

    for family in families:
        inputs = json.loads((INPUTS_DIR / f"{family}_panel.json").read_text())
        panel: dict[str, str | None] = inputs["panel"]
        sources = [c["source"] for c in inputs["cells"]]
        if only_sources:
            sources = [s for s in sources if s in only_sources]
        fam_out: dict[str, dict] = {}
        for source in sources:
            assert source in panel, f"{family}: source {source!r} not in panel"
            rows = _load_positives(family, source, args.rows)
            pairs = [
                (_chat_prompt(tokenizer, panel[source], r["question"]), r["completion"])
                for r in rows
            ]
            logger.info(
                "[phase=priors_score] family=%s source=%s n_rows=%d", family, source, len(pairs)
            )
            scored = _score_pairs(llm, tokenizer, pairs)
            vals = np.array([s / n for (s, n) in scored if n > 0 and not np.isnan(s)], dtype=float)
            n_failed = len(scored) - vals.size
            if n_failed:
                logger.warning(
                    "%s/%s: %d of %d rows failed scoring", family, source, n_failed, len(scored)
                )
            fam_out[source] = {
                "mean_logprob_per_tok": float(vals.mean()) if vals.size else float("nan"),
                "sem": float(vals.std(ddof=1) / np.sqrt(vals.size))
                if vals.size > 1
                else float("nan"),
                "n_rows": int(vals.size),
                "n_failed": int(n_failed),
                "per_row": [float(x) for x in vals],
            }
            # Checkpoint per source — partial results survive a crash.
            out_payload["families"][family] = fam_out
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                json.dump(out_payload, f, indent=2)
            logger.info(
                "[phase=priors_score] %s/%s mean=%.4f sem=%.4f n=%d (checkpointed)",
                family,
                source,
                fam_out[source]["mean_logprob_per_tok"],
                fam_out[source]["sem"],
                fam_out[source]["n_rows"],
            )

    logger.info("[phase=priors_complete] wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
