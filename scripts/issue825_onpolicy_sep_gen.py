"""Issue #825 `onpolicy-separator-control` G2: on-policy raw-text continuations.

Per model (base Qwen2.5-7B / Instruct), vLLM batched GREEDY (temperature 0,
seed 42 — the Track-M chat-side convention, plan section 11) raw-text
continuation of the pinned WikiText article prefixes (NO chat template on
either model — matching the exogenous control regime):

  wave 1: prompt = tokenizer.decode(input_ids[:256]), max_tokens 768
  wave 2: for articles with < 6 ELIGIBLE anchors on wave-1 text (the G2b pair
          ladder, run inline), prompt = decode(input_ids[:512]), max_tokens
          512 — SAME engine session; total window <= 1024 (extraction parity)

Per-continuation audit (round-3 format + the G2 regurgitation guard): token
length stats, 3-gram repetition rate (min_count 5), distinct-3-gram rate,
early-EOS rate, wave-2 count, and the overlap-with-true-continuation metric
(word-3-gram overlap vs ``tokenizer.decode(input_ids[prefix:])``).

vLLM hygiene: spawn EngineCore guard set BEFORE any vllm import; chunked
``llm.generate`` (EPM_VLLM_GREEDY_CHUNK_SIZE, default 500) with per-chunk INFO
lines; ``use_tqdm=False``; ``_reap_vllm_engine`` teardown. Each model runs in
its own process (the dispatcher invokes this script once per model), so the
engine dies with the process.

CPU smoke: ``--tiny-model-dir`` (transformers greedy substitute; plumbing
only) + ``--smoke-real-continuation`` (records the tiny generation but
substitutes the article's TRUE continuation text so the downstream ladder /
extract / fits exercise REAL sentence structure — tiny-real standard, fake
only the GPU-scale generation; declared in the audit backend field).

CLI:
  uv run python scripts/issue825_onpolicy_sep_gen.py --model base \
      --articles <pinned articles_armC.jsonl> --out-dir data/.../base/generation
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

# vLLM V1 fork-EngineCore guard (gotchas.md): spawn BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before numpy/torch import

import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_onpolicy_sep_pairs as ops_pairs  # noqa: E402
import issue931_common as common  # noqa: E402

SCRIPT = "scripts/issue825_onpolicy_sep_gen.py"
MODEL_IDS = {"base": "Qwen/Qwen2.5-7B", "instruct": "Qwen/Qwen2.5-7B-Instruct"}
GEN_SEED = 42  # Track-M chat-side answer-generation convention (plan section 11)
MAX_MODEL_LEN = 4096  # round-3 value (issue825_onpolicy_u2_gen.py:363)
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
REPETITION_MIN_COUNT = 5  # round-3 audit metric


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    ap.add_argument("--articles", type=Path, required=True, help="pinned articles_armC.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-items", type=int, default=0, help="0 = all articles (smoke slicing)")
    ap.add_argument(
        "--wave2-min-eligible",
        type=int,
        default=ops_pairs.WAVE2_MIN_ELIGIBLE,
        help="wave-2 top-up trigger: eligible anchors on wave-1 text below this "
        "(default 6 = production; smoke may force wave-2 with a high value)",
    )
    ap.add_argument(
        "--tiny-model-dir",
        default=None,
        help="SMOKE ONLY: tiny random-init Qwen2 dir (CPU substitute for vLLM 7B)",
    )
    ap.add_argument(
        "--smoke-real-continuation",
        action="store_true",
        help="SMOKE ONLY (tiny mode): record the tiny generation but substitute the "
        "article's TRUE continuation text so downstream phases see real sentences",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Generation backends (one engine session per process; wave 2 reuses it)
# ---------------------------------------------------------------------------


class VllmBackend:
    """One vLLM engine; chunked greedy generation; explicit reap on close."""

    def __init__(self, model_id: str):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "on-policy continuation generation requires a CUDA GPU for vLLM; "
                "for the CPU smoke pass --tiny-model-dir <dir>."
            )
        from vllm import LLM

        self.llm = LLM(model=model_id, max_model_len=MAX_MODEL_LEN)
        self.name = "vllm"

    def generate(self, prompts: list[str], max_tokens: int) -> list[dict]:
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, seed=GEN_SEED, max_tokens=max_tokens)
        out: list[dict] = []
        n_chunks = (len(prompts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
        for i in range(0, len(prompts), VLLM_CHUNK_SIZE):
            chunk = prompts[i : i + VLLM_CHUNK_SIZE]
            print(
                f"[vllm-chunk] onpolicy_sep chunk {i // VLLM_CHUNK_SIZE + 1}/{n_chunks} "
                f"({len(chunk)} prompts, max_tokens={max_tokens})",
                flush=True,
            )
            for o in self.llm.generate(chunk, sp, use_tqdm=False):
                c = o.outputs[0]
                out.append(
                    {
                        "text": c.text,
                        "token_ids": list(c.token_ids),
                        "finish_reason": str(c.finish_reason),
                    }
                )
        return out

    def close(self) -> None:
        from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

        _reap_vllm_engine(self.llm)


class TinyBackend:
    """CPU smoke substitute: transformers GREEDY on a tiny random-init Qwen2.

    SMOKE ONLY (declared in the audit): validates prompt construction, storage,
    audit + wave plumbing — never the production text distribution.
    """

    def __init__(self, tiny_dir: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(tiny_dir)
        self.model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32)
        self.model.eval()
        self.name = f"tiny-substitute ({tiny_dir})"
        self.smoke_tokens = int(os.environ.get("EPS_SMOKE_GEN_TOKENS", "16"))

    def generate(self, prompts: list[str], max_tokens: int) -> list[dict]:
        out: list[dict] = []
        for prompt in prompts:
            ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **ids,
                    do_sample=False,
                    max_new_tokens=min(max_tokens, self.smoke_tokens),
                    pad_token_id=self.tokenizer.pad_token_id or 0,
                )
            tail = gen[0][ids["input_ids"].shape[1] :]
            out.append(
                {
                    "text": self.tokenizer.decode(tail, skip_special_tokens=False),
                    "token_ids": [int(v) for v in tail],
                    "finish_reason": "length",
                }
            )
        print(f"[tiny-gen] {len(out)} rows (smoke substitute)")
        return out

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Audit metrics (round-3 helpers, kept local — see issue825_onpolicy_u2_gen)
# ---------------------------------------------------------------------------


def _word_3grams(text: str) -> list[tuple[str, ...]]:
    words = text.split()
    return [tuple(words[j : j + 3]) for j in range(len(words) - 2)]


def _distinct_3gram_rate(texts: list[str]) -> float:
    total = 0
    distinct: set[tuple[str, ...]] = set()
    for text in texts:
        grams = _word_3grams(text)
        total += len(grams)
        distinct.update(grams)
    return (len(distinct) / total) if total else 0.0


def _repeats_within(text: str, min_count: int = REPETITION_MIN_COUNT) -> bool:
    counts: Counter[tuple[str, ...]] = Counter(_word_3grams(text))
    return bool(counts) and max(counts.values()) >= min_count


def _true_overlap(continuation: str, true_continuation: str) -> float:
    """Fraction of the continuation's word-3-grams present in the article's
    TRUE continuation (the G2 regurgitation guard; NaN when too short)."""
    cont = set(_word_3grams(continuation))
    if not cont:
        return float("nan")
    true = set(_word_3grams(true_continuation))
    return len(cont & true) / len(cont)


def _length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"mean": None, "sd": None, "n": 0}
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _make_rows(
    articles: list[dict], gen: list[dict], wave: int, tokenizer, *, smoke_real: bool
) -> list[dict]:
    """One continuation row per article; optional smoke true-continuation swap."""
    rows = []
    prefix_n = ops_pairs.PREFIX_TOKENS[wave]
    budget = ops_pairs.CONTINUATION_MAX_TOKENS[wave]
    for art, g in zip(articles, gen, strict=True):
        row = {
            "window_id": art["window_id"],
            "wave": wave,
            "prefix_tokens": prefix_n,
            "continuation": g["text"],
            "continuation_token_ids": g["token_ids"],
            "finish_reason": g["finish_reason"],
            "n_tokens": len(g["token_ids"]),
        }
        if smoke_real:
            true_ids = list(art["input_ids"][prefix_n : prefix_n + budget])
            row.update(
                {
                    "continuation": tokenizer.decode(true_ids),
                    "continuation_token_ids": true_ids,
                    "n_tokens": len(true_ids),
                    "finish_reason": "smoke-true-continuation",
                    "smoke_tiny_generation": g["text"][:200],
                }
            )
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    if args.smoke_real_continuation:
        assert args.tiny_model_dir, "--smoke-real-continuation requires --tiny-model-dir"
    model_id = MODEL_IDS[args.model]
    tokenizer = common.get_tokenizer()  # base/instruct identical (p1 gate asserts)
    articles = ops_pairs.read_jsonl(args.articles)
    if args.max_items:
        articles = articles[: args.max_items]
    for a in articles:
        assert len(a["input_ids"]) >= common.ARMC_ARTICLE_MIN_TOKENS, a["window_id"]
    print(f"[i825-ops-gen] model={args.model} ({model_id}) articles={len(articles)}")

    backend = TinyBackend(args.tiny_model_dir) if args.tiny_model_dir else VllmBackend(model_id)
    try:
        # Wave 1: 256-token prefixes, 768 new tokens.
        prompts_w1 = [
            tokenizer.decode(a["input_ids"][: ops_pairs.PREFIX_TOKENS[1]]) for a in articles
        ]
        gen_w1 = backend.generate(prompts_w1, ops_pairs.CONTINUATION_MAX_TOKENS[1])
        rows = _make_rows(articles, gen_w1, 1, tokenizer, smoke_real=args.smoke_real_continuation)

        # Wave-2 top-up trigger: the G2b ladder run inline on wave-1 text.
        eligible_w1 = {
            r["window_id"]: ops_pairs.count_eligible(tokenizer, art, r["continuation"], wave=1)
            for art, r in zip(articles, rows, strict=True)
        }
        wave2_articles = [
            a for a in articles if eligible_w1[a["window_id"]] < args.wave2_min_eligible
        ]
        print(
            f"[i825-ops-gen] wave-2 trigger (<{args.wave2_min_eligible} eligible): "
            f"{len(wave2_articles)}/{len(articles)} articles"
        )
        if wave2_articles:
            prompts_w2 = [
                tokenizer.decode(a["input_ids"][: ops_pairs.PREFIX_TOKENS[2]])
                for a in wave2_articles
            ]
            gen_w2 = backend.generate(prompts_w2, ops_pairs.CONTINUATION_MAX_TOKENS[2])
            rows.extend(
                _make_rows(
                    wave2_articles,
                    gen_w2,
                    2,
                    tokenizer,
                    smoke_real=args.smoke_real_continuation,
                )
            )
    finally:
        backend.close()

    # Per-continuation audit (+ the overlap-with-true-continuation guard).
    art_by_id = {a["window_id"]: a for a in articles}
    per_row_audit = []
    for r in rows:
        prefix_n = ops_pairs.PREFIX_TOKENS[r["wave"]]
        true_cont = tokenizer.decode(art_by_id[r["window_id"]]["input_ids"][prefix_n:])
        per_row_audit.append(
            {
                "window_id": r["window_id"],
                "wave": r["wave"],
                "n_tokens": r["n_tokens"],
                "early_eos": r["finish_reason"] == "stop",
                "repeats_3gram_min5": _repeats_within(r["continuation"]),
                "true_continuation_overlap": _true_overlap(r["continuation"], true_cont),
            }
        )
    texts = [r["continuation"] for r in rows]
    overlaps = [
        a["true_continuation_overlap"]
        for a in per_row_audit
        if np.isfinite(a["true_continuation_overlap"])
    ]
    audit = {
        "metadata": common.metadata(SCRIPT, GEN_SEED, len(rows)),
        "followup_label": "onpolicy-separator-control",
        "model": args.model,
        "model_id": model_id,
        "backend": backend.name
        + (" + true-continuation substitute" if args.smoke_real_continuation else ""),
        "sampling": {
            "temperature": 0.0,
            "seed": GEN_SEED,
            "max_tokens_wave1": ops_pairs.CONTINUATION_MAX_TOKENS[1],
            "max_tokens_wave2": ops_pairs.CONTINUATION_MAX_TOKENS[2],
            "max_model_len": MAX_MODEL_LEN,
            "chat_template": False,
        },
        "n_articles": len(articles),
        "n_rows": len(rows),
        "n_wave2": sum(1 for r in rows if r["wave"] == 2),
        "eligible_wave1": eligible_w1,
        "length": _length_stats([r["n_tokens"] for r in rows]),
        "early_eos_rate": float(np.mean([a["early_eos"] for a in per_row_audit])),
        "repetition_rate_min5": float(np.mean([a["repeats_3gram_min5"] for a in per_row_audit])),
        "distinct_3gram_rate": _distinct_3gram_rate(texts),
        "true_continuation_overlap": {
            "mean": float(np.mean(overlaps)) if overlaps else None,
            "p90": float(np.quantile(overlaps, 0.9)) if overlaps else None,
            "max": float(np.max(overlaps)) if overlaps else None,
            "n": len(overlaps),
        },
        "per_row": per_row_audit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "continuations.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(out_path)
    common.write_json(args.out_dir / "generation_audit.json", audit)
    print(
        f"[i825-ops-gen] {args.model}: {len(rows)} continuations -> {out_path} "
        f"(wave2={audit['n_wave2']}, early_eos={audit['early_eos_rate']:.3f}, "
        f"rep={audit['repetition_rate_min5']:.3f}, "
        f"overlap_mean={audit['true_continuation_overlap']['mean']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
