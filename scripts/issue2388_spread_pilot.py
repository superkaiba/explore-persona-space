"""Issue #2388 — spread pilot.

Measures the per-item correctness-rate DISTRIBUTION for Qwen2.5-7B-Instruct on the
math / multiple-choice / code surfaces, at the production protocol (chat template,
temperature 1.0, zero-shot, K=5 sampled rollouts).

The question this answers is NOT "is the model good". It is whether the per-item
success probability has enough BETWEEN-ITEM dispersion for a probe to have anything
to predict. QA is already answered from banked #1739 rollouts (TriviaQA reliability
0.940, NQ-Open 0.934, SimpleQA dead at 91% zero-pile); this covers the other three.

Reported per BENCHMARK, never pooled — pooling benchmarks with different means
manufactures dispersion that is really a mixture artifact (#2388 risk 10).

Admissibility (pre-registered): a pool is admissible while NEITHER the zero-pile nor
the one-pile fraction exceeds ~90% — the shape that disqualified SimpleQA.

Usage (on the pod, one surface at a time so each checkpoints independently):
    uv run python scripts/issue2388_spread_pilot.py --surface math
    uv run python scripts/issue2388_spread_pilot.py --surface mcq
    uv run python scripts/issue2388_spread_pilot.py --surface code
    uv run python scripts/issue2388_spread_pilot.py --report-only
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy: shared-VM thread caps bind at import (#847)

import numpy as np  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"
K_ROLLOUTS = 5
TEMPERATURE = 1.0
TOP_P = 1.0
SEED = 2388

OUT_ROOT = Path("eval_results/issue_2388/spread_pilot")

# Per-benchmark item budgets. Code is split across three benchmarks whose published
# means bracket the range (HumanEval high, MBPP mid, BigCodeBench low) precisely so
# that WITHIN-benchmark dispersion can be read separately from the ACROSS-benchmark
# mixture.
BUDGETS = {
    "math500": 200,
    "mmlu_pro": 200,
    "humaneval": 164,  # the whole benchmark
    "mbpp": 150,
    "bigcodebench": 150,
}

MAX_TOKENS = {
    "math500": 2048,
    "mmlu_pro": 1024,
    "humaneval": 1024,
    "mbpp": 1024,
    "bigcodebench": 2048,
}

CODE_EXEC_TIMEOUT_S = 15


# --------------------------------------------------------------------------------------
# Dataset loading. Each returns a list of dicts with at least: item_id, prompt, plus
# whatever the verifier for that benchmark needs.
# --------------------------------------------------------------------------------------


def _stratified_sample(rows: list[dict], key: str, n: int, seed: int) -> list[dict]:
    """Sample n rows spread as evenly as possible across the values of `key`."""
    rng = np.random.default_rng(seed)
    buckets: dict[Any, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r[key], []).append(r)
    names = sorted(buckets, key=str)
    per = max(1, n // len(names))
    picked: list[dict] = []
    for name in names:
        pool = buckets[name]
        take = min(per, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        picked.extend(pool[i] for i in idx)
    # Top up / trim to exactly n without disturbing the per-bucket balance more than needed.
    if len(picked) > n:
        idx = rng.choice(len(picked), size=n, replace=False)
        picked = [picked[i] for i in sorted(idx)]
    elif len(picked) < n:
        # Identity-keyed, not value-keyed: `r not in picked` would be an O(n*m) sweep of
        # dict equality checks over the full row set (12k rows for MMLU-Pro).
        chosen = {id(r) for r in picked}
        remaining = [r for r in rows if id(r) not in chosen]
        extra = min(n - len(picked), len(remaining))
        if extra:
            idx = rng.choice(len(remaining), size=extra, replace=False)
            picked.extend(remaining[i] for i in idx)
    return picked


def load_math500() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [
        {
            "item_id": f"math500-{r['unique_id']}",
            "benchmark": "math500",
            "level": int(r["level"]),
            "subject": r["subject"],
            "gold": r["answer"],
            "question": r["problem"],
        }
        for r in ds
    ]
    picked = _stratified_sample(rows, "level", BUDGETS["math500"], SEED)
    for r in picked:
        r["prompt"] = (
            "Solve the following problem. Reason step by step, then give the final "
            "answer inside \\boxed{}.\n\n" + r["question"]
        )
    return picked


def load_mmlu_pro() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = []
    for r in ds:
        opts = list(r["options"])
        letters = [chr(ord("A") + i) for i in range(len(opts))]
        block = "\n".join(f"{ell}. {o}" for ell, o in zip(letters, opts, strict=True))
        rows.append(
            {
                "item_id": f"mmlupro-{r['question_id']}",
                "benchmark": "mmlu_pro",
                "category": r["category"],
                "gold": r["answer"],
                "n_options": len(opts),
                "question": r["question"],
                "prompt": (
                    "Answer the following multiple-choice question. Reason step by step, "
                    f"then end your response with exactly 'Answer: X' where X is one of "
                    f"{letters[0]}-{letters[-1]}.\n\n{r['question']}\n\n{block}"
                ),
            }
        )
    return _stratified_sample(rows, "category", BUDGETS["mmlu_pro"], SEED)


def load_humaneval() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    return [
        {
            "item_id": f"humaneval-{r['task_id'].replace('/', '_')}",
            "benchmark": "humaneval",
            "entry_point": r["entry_point"],
            "test_code": r["test"],
            "signature_prompt": r["prompt"],
            "prompt": (
                "Complete the following Python function. Return the ENTIRE function, "
                "including the signature, inside a single ```python code block. Do not "
                "include tests or example usage.\n\n```python\n" + r["prompt"] + "\n```"
            ),
        }
        for r in ds
    ][: BUDGETS["humaneval"]]


def load_mbpp() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    rows = [
        {
            "item_id": f"mbpp-{r['task_id']}",
            "benchmark": "mbpp",
            "test_code": "\n".join(list(r["test_list"])),
            "test_setup_code": r["test_setup_code"] or "",
            "prompt": (
                f"{r['text']}\n\nYour code must satisfy these tests:\n\n"
                + "\n".join(list(r["test_list"]))
                + "\n\nReturn the complete solution inside a single ```python code block. "
                "Do not include the tests."
            ),
        }
        for r in ds
    ]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(rows), size=min(BUDGETS["mbpp"], len(rows)), replace=False)
    return [rows[i] for i in sorted(idx)]


def load_bigcodebench() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    rows = [
        {
            "item_id": f"bcb-{r['task_id'].replace('/', '_')}",
            "benchmark": "bigcodebench",
            "entry_point": r["entry_point"],
            "test_code": r["test"],
            "prompt": (
                r["instruct_prompt"]
                + "\n\nReturn the complete solution inside a single ```python code block."
            ),
        }
        for r in ds
    ]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(rows), size=min(BUDGETS["bigcodebench"], len(rows)), replace=False)
    return [rows[i] for i in sorted(idx)]


SURFACES = {
    "math": ["math500"],
    "mcq": ["mmlu_pro"],
    "code": ["humaneval", "mbpp", "bigcodebench"],
}
LOADERS = {
    "math500": load_math500,
    "mmlu_pro": load_mmlu_pro,
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
    "bigcodebench": load_bigcodebench,
}


# --------------------------------------------------------------------------------------
# Verification. Every verifier returns True / False / None, where None means the
# completion could not be PARSED into an answer at all. None is recorded and reported
# separately, never silently coerced to "incorrect" — a format failure and a wrong
# answer are different events (drop-never-coerce).
# --------------------------------------------------------------------------------------


def _extract_boxed(text: str) -> str | None:
    """Return the content of the LAST \\boxed{...}, brace-balanced."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip()


def verify_math(completion: str, item: dict) -> bool | None:
    pred = _extract_boxed(completion)
    if pred is None:
        return None
    # PROD_IMPORT_LINT_EXEMPT: one-off pod-side install expected; failure must be loud (#2388)
    from math_verify import parse, verify  # hard dependency; install failure must be loud

    gold_parsed = parse(f"${item['gold']}$")
    pred_parsed = parse(f"${pred}$")
    if not gold_parsed or not pred_parsed:
        # Fall back to normalized string equality rather than dropping a real answer.
        return _norm_math(pred) == _norm_math(item["gold"])
    return bool(verify(gold_parsed, pred_parsed))


def _norm_math(s: str) -> str:
    s = s.strip().replace(" ", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.rstrip(".")
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s


_ANSWER_RE = re.compile(r"Answer:\s*\(?([A-J])\)?", re.IGNORECASE)


def verify_mcq(completion: str, item: dict) -> bool | None:
    matches = _ANSWER_RE.findall(completion)
    if matches:
        return matches[-1].upper() == item["gold"].upper()
    # Fallback: a lone bracketed or final standalone letter.
    tail = completion.strip()[-200:]
    loose = re.findall(r"\b([A-J])\b", tail)
    if not loose:
        return None
    return loose[-1].upper() == item["gold"].upper()


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str | None:
    blocks = _CODE_BLOCK_RE.findall(completion)
    if not blocks:
        return None
    # The longest block is the implementation when the model also emits snippets.
    return max(blocks, key=len).strip()


def _run_snippet(payload: str, timeout_s: int) -> bool:
    """Execute `payload` in a fresh subprocess. True iff it exits 0 within the timeout.

    Model-generated code is untrusted. This is intended to run on an EPHEMERAL pod
    that is destroyed afterwards, never on the shared VM.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(payload)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            timeout=timeout_s,
            text=True,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def verify_code(completion: str, item: dict) -> bool | None:
    code = extract_code(completion)
    if code is None:
        return None
    bench = item["benchmark"]
    if bench == "humaneval":
        payload = f"{code}\n\n{item['test_code']}\n\ncheck({item['entry_point']})\n"
    elif bench == "mbpp":
        payload = f"{code}\n\n{item['test_setup_code']}\n\n{item['test_code']}\n"
    elif bench == "bigcodebench":
        payload = (
            f"{code}\n\n{item['test_code']}\n\n"
            "import unittest\n"
            "_r = unittest.main(exit=False, argv=['x'], verbosity=0).result\n"
            "raise SystemExit(0 if _r.wasSuccessful() else 1)\n"
        )
    else:
        raise ValueError(f"no code verifier for benchmark {bench!r}")
    return _run_snippet(payload, CODE_EXEC_TIMEOUT_S)


VERIFIERS = {
    "math500": verify_math,
    "mmlu_pro": verify_mcq,
    "humaneval": verify_code,
    "mbpp": verify_code,
    "bigcodebench": verify_code,
}


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def generate(items: list[dict], max_tokens: int) -> tuple[list[list[str]], float]:
    """K sampled completions per item via one batched vLLM call.

    Returns (completions, fraction of items whose K draws were all identical) — the
    second value is the degenerate-sampling diagnostic, persisted with the results.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": it["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for it in items
    ]
    # Engine-level seed gives run reproducibility. The per-request SamplingParams seed is
    # deliberately LEFT UNSET: a single seed shared across the n draws of one request risks
    # returning K identical completions, which would make every item score exactly 0.0 or 1.0
    # and manufacture perfect bimodality — the precise quantity this pilot measures. Independent
    # draws matter more here than per-request reproducibility.
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, seed=SEED)
    params = SamplingParams(
        n=K_ROLLOUTS, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=max_tokens
    )
    outs = llm.generate(prompts, params)
    assert len(outs) == len(items), f"vLLM returned {len(outs)} outputs for {len(items)} prompts"

    completions, cap_hits, n_all_identical = [], 0, 0
    for o in outs:
        assert len(o.outputs) == K_ROLLOUTS, f"expected {K_ROLLOUTS} rollouts, got {len(o.outputs)}"
        cap_hits += sum(1 for c in o.outputs if c.finish_reason == "length")
        texts = [c.text for c in o.outputs]
        if len(set(texts)) == 1:
            n_all_identical += 1
        completions.append(texts)

    total = len(items) * K_ROLLOUTS
    print(f"  cap-hit fraction: {cap_hits}/{total} = {cap_hits / total:.3%}", flush=True)
    if cap_hits / total > 0.02:
        print(
            f"  WARNING: cap-hit {cap_hits / total:.1%} exceeds the 2% re-generation trigger "
            f"at max_tokens={max_tokens}",
            flush=True,
        )

    # Degenerate-sampling guard. If the K draws collapse to one string the DV is 0/1 by
    # construction and the spread read is meaningless — a bug that would LOOK like a clean
    # bimodal result. Some genuine collapse is expected on short-answer items, so this reports
    # always and only refuses when sampling is effectively deterministic across the board.
    frac_identical = n_all_identical / len(items)
    print(
        f"  all-K-identical items: {n_all_identical}/{len(items)} = {frac_identical:.1%}",
        flush=True,
    )
    assert frac_identical < 0.95, (
        f"{frac_identical:.1%} of items returned K identical completions — sampling is "
        f"effectively deterministic (check temperature={TEMPERATURE} and that SamplingParams "
        f"carries no per-request seed). The spread measurement would be an artifact."
    )
    return completions, frac_identical


# --------------------------------------------------------------------------------------
# Spread statistics
# --------------------------------------------------------------------------------------


def spread_stats(rates: list[float], k: int) -> dict:
    """Beta-binomial decomposition of an observed K-rollout rate distribution.

    Var(y) = Var(p) + E[p(1-p)]/K, and E[y(1-y)] = ((K-1)/K)E[p(1-p)], so the
    between-item variance in the TRUE rate is Var(y) - mean(y(1-y))/(K-1).
    Its share of Var(y) is the DV's reliability; the square root of that share is
    the attenuation ceiling on any correlation measured against this DV.
    """
    y = np.asarray(rates, dtype=float)
    assert y.ndim == 1 and y.size > 0, f"bad rates array shape {y.shape}"
    var_y = float(y.var(ddof=1)) if y.size > 1 else 0.0
    within = float((y * (1 - y)).mean()) / (k - 1)
    var_p = max(var_y - within, 0.0)
    reliability = var_p / var_y if var_y > 0 else 0.0
    hist = np.bincount(np.round(y * k).astype(int), minlength=k + 1)
    return {
        "n_items": int(y.size),
        "mean": float(y.mean()),
        "sd_observed": math.sqrt(var_y),
        "sd_true_rate": math.sqrt(var_p),
        "reliability": reliability,
        "ceiling_rho": math.sqrt(reliability),
        "zero_pile": float((y == 0.0).mean()),
        "one_pile": float((y == 1.0).mean()),
        "histogram_counts": [int(h) for h in hist],
    }


def admissible(stats: dict) -> bool:
    return stats["zero_pile"] <= 0.90 and stats["one_pile"] <= 0.90


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def run_benchmark(name: str) -> dict:
    print(f"\n=== {name} ===", flush=True)
    items = LOADERS[name]()
    print(f"  loaded {len(items)} items", flush=True)
    completions, frac_identical = generate(items, MAX_TOKENS[name])

    verifier = VERIFIERS[name]
    rows, n_unparsed = [], 0
    for item, comps in zip(items, completions, strict=True):
        verdicts = [verifier(c, item) for c in comps]
        n_unparsed += sum(1 for v in verdicts if v is None)
        decided = [v for v in verdicts if v is not None]
        rows.append(
            {
                "item_id": item["item_id"],
                "benchmark": name,
                "n_decided": len(decided),
                "n_correct": sum(1 for v in decided if v),
                "rate_decided": (sum(1 for v in decided if v) / len(decided)) if decided else None,
                "rate_unparsed_as_wrong": sum(1 for v in verdicts if v is True) / len(verdicts),
                "level": item.get("level"),
                "category": item.get("category"),
            }
        )

    full = [r["rate_decided"] for r in rows if r["n_decided"] == K_ROLLOUTS]
    coerced = [r["rate_unparsed_as_wrong"] for r in rows]
    result = {
        "benchmark": name,
        "model": MODEL,
        "k_rollouts": K_ROLLOUTS,
        "temperature": TEMPERATURE,
        "n_items": len(rows),
        "frac_items_all_k_identical": frac_identical,
        "n_unparsed_rollouts": n_unparsed,
        "unparsed_fraction": n_unparsed / (len(rows) * K_ROLLOUTS),
        # Primary read: items where all K rollouts parsed, so the binomial decomposition holds.
        "stats_full_k": spread_stats(full, K_ROLLOUTS) if full else None,
        # Companion: unparsed counted as incorrect. Reported so the choice is visible,
        # never substituted for the primary.
        "stats_unparsed_as_wrong": spread_stats(coerced, K_ROLLOUTS),
        "rows": rows,
    }
    result["admissible"] = bool(result["stats_full_k"] and admissible(result["stats_full_k"]))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / f"{name}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"  wrote {out}", flush=True)
    _print_row(result)
    return result


def _print_row(result: dict) -> None:
    s = result["stats_full_k"]
    if s is None:
        print(f"  {result['benchmark']}: NO fully-parsed items", flush=True)
        return
    print(
        f"  {result['benchmark']:>14s} n={s['n_items']:4d} mean={s['mean']:.3f} "
        f"SD(y)={s['sd_observed']:.3f} SD(p)={s['sd_true_rate']:.3f} "
        f"reliab={s['reliability']:.3f} ceil_rho={s['ceiling_rho']:.3f} "
        f"zero={s['zero_pile']:.1%} one={s['one_pile']:.1%} "
        f"unparsed={result['unparsed_fraction']:.1%} "
        f"{'ADMISSIBLE' if result['admissible'] else 'REJECT'}",
        flush=True,
    )


def report_only() -> None:
    print(
        f"{'benchmark':>14s} {'n':>5s} {'mean':>6s} {'SD(y)':>6s} {'SD(p)':>6s} "
        f"{'reliab':>7s} {'ceil':>6s} {'zero':>6s} {'one':>6s} {'unpars':>7s}  verdict"
    )
    for path in sorted(OUT_ROOT.glob("*.json")):
        if path.name == "summary.json":
            continue
        r = json.loads(path.read_text())
        s = r["stats_full_k"]
        if s is None:
            print(f"{r['benchmark']:>14s}  NO PARSED ITEMS")
            continue
        print(
            f"{r['benchmark']:>14s} {s['n_items']:5d} {s['mean']:6.3f} {s['sd_observed']:6.3f} "
            f"{s['sd_true_rate']:6.3f} {s['reliability']:7.3f} {s['ceiling_rho']:6.3f} "
            f"{s['zero_pile']:6.1%} {s['one_pile']:6.1%} {r['unparsed_fraction']:7.1%}  "
            f"{'ADMISSIBLE' if r['admissible'] else 'REJECT'}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surface", choices=sorted(SURFACES), help="math | mcq | code")
    ap.add_argument("--benchmark", choices=sorted(LOADERS), help="run a single benchmark")
    ap.add_argument("--report-only", action="store_true", help="re-print the table from disk")
    args = ap.parse_args(argv)

    if args.report_only:
        report_only()
        return 0
    if args.benchmark:
        names = [args.benchmark]
    elif args.surface:
        names = SURFACES[args.surface]
    else:
        ap.error("pass --surface, --benchmark, or --report-only")

    for name in names:
        run_benchmark(name)  # checkpoints per benchmark; a later crash keeps earlier results
    report_only()
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
