"""Issue #2388 — P1 production generation + verification (math / MCQ / code surfaces).

Extends the spread pilot's harness (``scripts/issue2388_spread_pilot.py`` —
prompt templates, verifiers, code sandbox REUSED by import) to the FULL pools,
adds the three new loaders (full MATH ``EleutherAI/hendrycks_math``,
LiveCodeBench-v5 ``livecodebench/code_generation_lite``, ``newfacade/
LeetCodeDataset`` at the pinned revision) and the ALWAYS-ON LCB<->LeetCode
dedup (plan section 4 fork 5: match on LeetCode problem slug for LCB items with
``platform == leetcode``; drop the LCB copy; the dedup report is a BINDING
pre-fit input).

Phases (one benchmark at a time so each checkpoints independently; the pod
dispatcher shards benchmarks across GPUs via per-process CUDA_VISIBLE_DEVICES):

    gen     K=5 sampled rollouts via vLLM (temp 1.0, top_p 1.0, chat template),
            CHUNKED generate (#664 deadlock prevention; per-chunk append +
            resume by item_id), cap-hit fraction reported per chunk.
    verify  programmatic verdicts (math-verify / option-match / unit-test
            execution), process-pooled, checkpointed per chunk; per-benchmark
            spread_stats (pilot reuse) + admissibility REPORTED (the BCB
            full-pool re-measure input).
    upload  rollout TEXT -> HF ``issue2388_correctness/raw_completions/gen/
            <surface>/`` (<=9 MB JSONL line-shards; upload-policy split rule).

Smoke mode (``--smoke``): 20 contexts per benchmark through the SAME
entrypoint, engine path, K, and caps; out-roots rebind to ``*_smoke`` so smoke
artifacts never overwrite production paths. Loader count-asserts run on the
FULL pool BEFORE the smoke slice (production gates are never downgraded —
smoke-blind-spots rule).

Terminal: ``os._exit(0)`` after flush on the gen phase (vLLM engine children
survive interpreter finalization otherwise — gotchas.md #1739/#2149).

CONTENT HYGIENE: benchmark questions are benign; logs still carry ids/counts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Script mode puts scripts/ on sys.path[0], not the repo root (#823)."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists(), f"repo-root sentinel missing at {root}"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

# vLLM reads this at import time — set BEFORE any vllm import (#628 fork trap).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy/torch: thread caps bind at import (#847)


from scripts.issue2388_spread_pilot import (  # noqa: E402
    K_ROLLOUTS,
    MODEL,
    SEED,
    TEMPERATURE,
    TOP_P,
    admissible,
    extract_code,
    spread_stats,
    verify_math,
    verify_mcq,
)
from scripts.issue2388_spread_pilot import (  # noqa: E402
    load_humaneval as _pilot_load_humaneval,
)

OUT_ROOT = Path("eval_results/issue_2388/gen")
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_GEN_PREFIX = "issue2388_correctness/raw_completions/gen"
LEETCODE_REVISION = "215604aeed660029df7de2fea5a4d7b6ed476a08"  # plan section 10 pin
LCB_VERSION_TAG = "release_v5"
# livecodebench/code_generation_lite is a SCRIPT dataset (datasets>=3 removed
# trust_remote_code loading), so we read its raw jsonl files directly at a
# pinned revision. File set copied from the repo's own loader script
# (ALLOWED_FILES["release_v5"] at this revision — schema-from-artifact).
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_V5_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl")

# Realized-count loader asserts (plan A5/A6 + section 4 pool table). Every count
# below is deterministic at its pin (fixed split / pinned revision / version
# tag), so a mismatch is a wrong-source signal, never noise.
EXPECTED_COUNTS = {
    "math_full": 12500,
    "mmlu_pro_full": 12032,
    "humaneval": 164,
    "mbpp_full": 974,
    "bigcodebench_full": 1140,
    "lcb_v5": 880,
    "leetcode": 2869,
}

MAX_TOKENS = {
    "math_full": 2048,
    "mmlu_pro_full": 1024,
    "humaneval": 1024,
    "mbpp_full": 1024,
    "bigcodebench_full": 2048,
    "lcb_v5": 2048,
    "leetcode": 2048,
}

SURFACES = {
    "math": ["math_full"],
    "mcq": ["mmlu_pro_full"],
    "code": ["humaneval", "mbpp_full", "bigcodebench_full", "lcb_v5", "leetcode"],
}
CODE_BENCHMARKS = set(SURFACES["code"])
CODE_EXEC_TIMEOUT_S = 15
SMOKE_N = 20
GEN_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
VERIFY_CHUNK = 200
CAP_HIT_TRIGGER = 0.02  # CLAUDE.md re-generation trigger (reported per family)


def surface_of(benchmark: str) -> str:
    for s, benches in SURFACES.items():
        if benchmark in benches:
            return s
    raise ValueError(f"unknown benchmark {benchmark!r}")


# ---------------------------------------------------------------------------
# new loaders (full pools). Pilot loaders are subsampled by BUDGETS — the
# production pools re-load full splits with the SAME prompt templates.
# ---------------------------------------------------------------------------


def _slugify(title: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.strip().lower())).strip("-")


def load_math_full() -> list[dict]:
    """Full MATH (12,500): every config x {train, test} of EleutherAI/hendrycks_math.

    ``level`` parses ``"Level N"`` -> int; the 2 known ``"Level ?"`` rows are
    TOLERATED (level=None — excluded from level stratification, retained in the
    pool; plan section 4 Splits). Gold = last \\boxed{...} of the reference
    solution; extraction failures are DROPPED with a digest count and a >1%
    fail-loud floor (an unverifiable row cannot carry a programmatic DV).
    """
    from datasets import get_dataset_config_names, load_dataset

    from scripts.issue2388_spread_pilot import _extract_boxed

    configs = sorted(get_dataset_config_names("EleutherAI/hendrycks_math"))
    rows: list[dict] = []
    n_total = 0
    n_gold_missing = 0
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split=split)
            for i, r in enumerate(ds):
                n_total += 1
                m = re.match(r"Level (\d+)$", str(r["level"]).strip())
                level = int(m.group(1)) if m else None  # "Level ?" tolerated
                gold = _extract_boxed(r["solution"])
                if gold is None:
                    n_gold_missing += 1
                    continue
                rows.append(
                    {
                        "item_id": f"mathfull-{cfg}-{split}-{i:05d}",
                        "benchmark": "math_full",
                        "level": level,
                        "subject": cfg,
                        "gold": gold,
                        "question": r["problem"],
                        "prompt": (
                            "Solve the following problem. Reason step by step, then give the "
                            "final answer inside \\boxed{}.\n\n" + r["problem"]
                        ),
                    }
                )
    assert n_total == EXPECTED_COUNTS["math_full"], (
        f"math_full realized {n_total} rows != expected {EXPECTED_COUNTS['math_full']} (plan A5)"
    )
    if n_gold_missing / n_total > 0.01:
        raise RuntimeError(
            f"math_full: {n_gold_missing}/{n_total} rows without an extractable \\boxed gold "
            "(>1% floor) — solution-format drift, re-scope before generation"
        )
    n_unparse_level = sum(1 for r in rows if r["level"] is None)
    print(
        f"[load] math_full: {len(rows)} kept / {n_total} total "
        f"(gold-missing {n_gold_missing}, level-unparsed {n_unparse_level})",
        flush=True,
    )
    return rows


def load_mmlu_pro_full() -> list[dict]:
    """MMLU-Pro FULL test split (the pilot loader minus its 200-item subsample)."""
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    assert len(ds) == EXPECTED_COUNTS["mmlu_pro_full"], (
        f"mmlu_pro realized {len(ds)} rows != expected {EXPECTED_COUNTS['mmlu_pro_full']}"
    )
    rows = []
    for r in ds:
        opts = list(r["options"])
        letters = [chr(ord("A") + i) for i in range(len(opts))]
        block = "\n".join(f"{ell}. {o}" for ell, o in zip(letters, opts, strict=True))
        rows.append(
            {
                "item_id": f"mmlupro-{r['question_id']}",
                "benchmark": "mmlu_pro_full",
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
    return rows


def load_humaneval_full() -> list[dict]:
    rows = _pilot_load_humaneval()  # already the whole benchmark (164)
    for r in rows:
        r["benchmark"] = "humaneval"
    assert len(rows) == EXPECTED_COUNTS["humaneval"], len(rows)
    return rows


def load_mbpp_full() -> list[dict]:
    """MBPP 'full' config, ALL splits (train+test+validation+prompt = 974)."""
    from datasets import load_dataset

    rows = []
    for split in ("train", "test", "validation", "prompt"):
        ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
        for r in ds:
            rows.append(
                {
                    "item_id": f"mbpp-{r['task_id']}",
                    "benchmark": "mbpp_full",
                    "test_code": "\n".join(list(r["test_list"])),
                    "test_setup_code": r["test_setup_code"] or "",
                    "prompt": (
                        f"{r['text']}\n\nYour code must satisfy these tests:\n\n"
                        + "\n".join(list(r["test_list"]))
                        + "\n\nReturn the complete solution inside a single ```python code "
                        "block. Do not include the tests."
                    ),
                }
            )
    assert len(rows) == EXPECTED_COUNTS["mbpp_full"], (
        f"mbpp_full realized {len(rows)} != expected {EXPECTED_COUNTS['mbpp_full']}"
    )
    return rows


def load_bigcodebench_full() -> list[dict]:
    """BCB v0.1.4 FULL (1,140) — the pilot loader minus its 150-item subsample."""
    from datasets import load_dataset

    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    assert len(ds) == EXPECTED_COUNTS["bigcodebench_full"], len(ds)
    rows = [
        {
            "item_id": f"bcb-{r['task_id'].replace('/', '_')}",
            "benchmark": "bigcodebench_full",
            "entry_point": r["entry_point"],
            "test_code": r["test"],
            "prompt": (
                r["instruct_prompt"]
                + "\n\nReturn the complete solution inside a single ```python code block."
            ),
        }
        for r in ds
    ]
    return rows


def load_lcb_v5() -> list[dict]:
    """LiveCodeBench code_generation_lite, release_v5 composition (880 problems).

    datasets>=3 removed script-dataset loading (``trust_remote_code``), so the
    version_tag mechanism is unusable on this env — we instead read the raw
    ``test.jsonl..test5.jsonl`` files (== the repo script's
    ``ALLOWED_FILES["release_v5"]``) at the pinned revision, exactly as the
    script's ``_generate_examples`` does (verbatim ``json.loads`` per line, no
    decoding). Field semantics from that script's Features block:
    ``public_test_cases`` is a JSON string; ``private_test_cases`` is either a
    JSON string or base64+zlib+pickle-wrapped JSON (their own
    ``translate_private_test_cases`` recipe); ``metadata`` carries
    ``func_name`` for functional (leetcode-style) items; ``platform`` in
    {leetcode, codeforces, atcoder}. Loader asserts fail loud on drift (plan
    A17-A19: 'verified at Step 0-p loader asserts').
    """
    import base64
    import pickle
    import zlib

    from huggingface_hub import hf_hub_download

    raw_rows: list[dict] = []
    for fname in LCB_V5_FILES:
        path = hf_hub_download(
            "livecodebench/code_generation_lite",
            fname,
            repo_type="dataset",
            revision=LCB_REVISION,
        )
        with open(path, encoding="utf-8") as fh:  # text-mode iteration, never splitlines()
            for line in fh:
                if line.strip():
                    raw_rows.append(json.loads(line))
    assert len(raw_rows) == EXPECTED_COUNTS["lcb_v5"], (
        f"lcb_v5 realized {len(raw_rows)} != expected {EXPECTED_COUNTS['lcb_v5']} "
        f"from {LCB_V5_FILES} at revision {LCB_REVISION}"
    )
    rows = []
    for r in raw_rows:
        for field in ("question_title", "question_content", "platform", "question_id"):
            assert field in r, f"LCB field drift: {field!r} missing (have {sorted(r)[:12]})"
        public = json.loads(r["public_test_cases"])
        try:
            private = json.loads(r["private_test_cases"])
        except (json.JSONDecodeError, TypeError):
            private = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"])))
            )
        tests = list(public) + list(private)
        assert tests and all("input" in t and "output" in t and "testtype" in t for t in tests), (
            f"LCB test-case shape drift on {r['question_id']}"
        )
        meta = json.loads(r["metadata"]) if r.get("metadata") else {}
        starter = r.get("starter_code") or ""
        prompt = (
            "Solve the following programming problem.\n\n"
            + r["question_content"]
            + (
                "\n\nUse this starter code (complete the class/method as given):\n\n"
                f"```python\n{starter}\n```"
                if starter.strip()
                else "\n\nRead input from stdin and write the answer to stdout."
            )
            + "\n\nReturn the complete solution inside a single ```python code block."
        )
        plat = str(r["platform"]).lower()
        rows.append(
            {
                "item_id": f"lcb-{plat}-{r['question_id']}",
                "benchmark": "lcb_v5",
                "platform": plat,
                "question_title": r["question_title"],
                "slug": _slugify(r["question_title"]),
                "func_name": meta.get("func_name"),
                "starter_code": starter,
                "tests": tests,
                "prompt": prompt,
            }
        )
    return rows


def load_leetcode() -> list[dict]:
    """newfacade/LeetCodeDataset at the pinned revision (2,869 problems, all splits).

    ``task_id`` IS the problem slug (e.g. ``two-sum``) — the dedup key. The
    row's own ``prompt`` field is the import preamble its ``test`` harness
    needs; ``entry_point`` is the bound call (``Solution().twoSum``).
    """
    from datasets import load_dataset

    rows = []
    for split in ("train", "test"):
        ds = load_dataset("newfacade/LeetCodeDataset", split=split, revision=LEETCODE_REVISION)
        for r in ds:
            for field in ("task_id", "problem_description", "starter_code", "test", "entry_point"):
                assert field in r, f"LeetCodeDataset field drift: {field!r} missing"
            rows.append(
                {
                    "item_id": f"leetcode-{r['task_id']}",
                    "benchmark": "leetcode",
                    "slug": str(r["task_id"]),
                    "difficulty": r.get("difficulty"),
                    "canonical_completion": r["completion"],  # code-control positive control
                    "test_imports": r["prompt"],
                    "test_code": r["test"],
                    "entry_point": r["entry_point"],
                    "prompt": (
                        "Solve the following programming problem.\n\n"
                        + r["problem_description"]
                        + "\n\nComplete this starter code:\n\n"
                        + f"```python\n{r['starter_code']}\n```"
                        + "\n\nReturn the complete solution inside a single ```python code "
                        "block."
                    ),
                }
            )
    assert len(rows) == EXPECTED_COUNTS["leetcode"], (
        f"leetcode realized {len(rows)} != expected {EXPECTED_COUNTS['leetcode']} "
        f"at revision {LEETCODE_REVISION}"
    )
    return rows


LOADERS = {
    "math_full": load_math_full,
    "mmlu_pro_full": load_mmlu_pro_full,
    "humaneval": load_humaneval_full,
    "mbpp_full": load_mbpp_full,
    "bigcodebench_full": load_bigcodebench_full,
    "lcb_v5": load_lcb_v5,
    "leetcode": load_leetcode,
}
assert set(LOADERS) == set(EXPECTED_COUNTS) == set(MAX_TOKENS)


# ---------------------------------------------------------------------------
# dedup (ALWAYS runs before any code fit — plan section 4 fork 5)
# ---------------------------------------------------------------------------


def dedup_lcb_against_leetcode(out_root: Path) -> dict:
    """Drop LCB leetcode-platform items whose slug matches LeetCodeDataset.

    Deterministic: LCB items with ``platform == leetcode`` match on the
    slugified problem title against LeetCodeDataset's ``task_id`` slug; the LCB
    copy is dropped. Writes the BINDING dedup report
    (``<out_root>/code/dedup_report.json``) whose realized post-dedup count
    feeds the section-7 fork-5 trigger.
    """
    lcb = load_lcb_v5()
    leet = load_leetcode()
    leet_slugs = {r["slug"] for r in leet}
    dropped = sorted(
        r["item_id"] for r in lcb if r["platform"] == "leetcode" and r["slug"] in leet_slugs
    )
    report = {
        "n_lcb": len(lcb),
        "n_leetcode": len(leet),
        "n_lcb_leetcode_platform": sum(1 for r in lcb if r["platform"] == "leetcode"),
        "n_dropped_lcb": len(dropped),
        "dropped_lcb_item_ids": dropped,
        "rule": "LCB platform==leetcode with slugified title in LeetCodeDataset task_id slugs",
        "leetcode_revision": LEETCODE_REVISION,
        "lcb_version_tag": LCB_VERSION_TAG,
        "lcb_revision": LCB_REVISION,
        "lcb_v5_files": list(LCB_V5_FILES),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    report.update(as_metadata_dict(git_provenance(), phase="gen-dedup"))
    path = out_root / "code" / "dedup_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, path)
    print(f"[dedup] {json.dumps({k: report[k] for k in list(report)[:5]})} -> {path}", flush=True)
    return report


def _apply_dedup(items: list[dict], out_root: Path) -> list[dict]:
    """Filter a benchmark's items by the committed dedup report (LCB only)."""
    path = out_root / "code" / "dedup_report.json"
    if not path.exists():
        raise FileNotFoundError(f"dedup report missing at {path} — run --phase dedup first")
    dropped = set(json.loads(path.read_text())["dropped_lcb_item_ids"])
    kept = [r for r in items if r["item_id"] not in dropped]
    print(f"[dedup] lcb_v5: kept {len(kept)}/{len(items)} after dedup", flush=True)
    if not kept:
        raise RuntimeError("dedup emptied lcb_v5 — impossible under the slug rule; wrong report?")
    return kept


# ---------------------------------------------------------------------------
# code verification (new benchmarks; pilot's sandbox pattern)
# ---------------------------------------------------------------------------


def _run_code(payload: str, timeout_s: int, python_exe: str | None = None) -> bool:
    """Pilot ``_run_snippet`` with an interpreter override (the BCB /opt venv gate)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(payload)
        path = fh.name
    try:
        proc = subprocess.run(
            [python_exe or sys.executable, path],
            capture_output=True,
            timeout=timeout_s,
            text=True,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


# LeetCode-style solutions run under STAR-imports in the reference LCB harness
# (`from math import *` etc. — bare `inf`/`gcd`/`List` are idiomatic in both
# canonical and model completions), so the preamble matches those semantics
# (reference-eval fidelity — the SAE token-pool lesson). LCB functional
# problems are JSON-I/O by dataset construction (no ListNode/TreeNode). The
# harness's own locals are underscore-named, which star-imports cannot shadow;
# the module names themselves are re-imported AFTER the star block.
_LCB_PREAMBLE = """\
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from typing import *
import string, re, datetime, collections, heapq, bisect, copy, math
import random, statistics, itertools, functools, operator, io, sys, json
from sortedcontainers import SortedList, SortedDict, SortedSet
sys.setrecursionlimit(50000)
"""

_LCB_FUNCTIONAL_HARNESS = (
    _LCB_PREAMBLE
    + """
{code}

def _resolve(name):
    if "Solution" in globals():
        cand = getattr(Solution(), name, None)
        if cand is not None:
            return cand
    return globals()[name]

def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b

_tests = json.load(open({tests_path!r}))
for _t in _tests:
    _args = [json.loads(line) for line in _t["input"].split(chr(10)) if line.strip()]
    _got = _resolve({fn_name!r})(*_args)
    _want = json.loads(_t["output"]) if _t["output"].strip() else None
    if not _eq(_got, _want):
        raise SystemExit(1)
raise SystemExit(0)
"""
)


def _verify_lcb(completion: str, item: dict, python_exe: str | None = None) -> bool | None:
    """LCB verdict: functional (func_name) or stdin/stdout, all tests must pass."""
    code = extract_code(completion)
    if code is None:
        return None
    tests = item["tests"]
    functional = [t for t in tests if t.get("testtype") == "functional"]
    stdin_tests = [t for t in tests if t.get("testtype") != "functional"]
    if functional:
        if not item.get("func_name"):
            return None  # functional tests without a func_name are unverifiable
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(functional, fh)
            tests_path = fh.name
        try:
            payload = _LCB_FUNCTIONAL_HARNESS.format(
                code=code, fn_name=item["func_name"], tests_path=tests_path
            )
            if not _run_code(payload, CODE_EXEC_TIMEOUT_S, python_exe):
                return False
        finally:
            Path(tests_path).unlink(missing_ok=True)
    for t in stdin_tests:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
            path = fh.name
        try:
            proc = subprocess.run(
                [python_exe or sys.executable, path],
                input=t["input"],
                capture_output=True,
                timeout=CODE_EXEC_TIMEOUT_S,
                text=True,
            )
            if proc.returncode != 0 or proc.stdout.strip() != str(t["output"]).strip():
                return False
        except subprocess.TimeoutExpired:
            return False
        finally:
            Path(path).unlink(missing_ok=True)
    return True


def _verify_leetcode(completion: str, item: dict, python_exe: str | None = None) -> bool | None:
    code = extract_code(completion)
    if code is None:
        return None
    payload = (
        f"{item['test_imports']}\n\n{code}\n\n{item['test_code']}\n\ncheck({item['entry_point']})\n"
    )
    return _run_code(payload, CODE_EXEC_TIMEOUT_S, python_exe)


def _verify_pilot_code(completion: str, item: dict, python_exe: str | None = None) -> bool | None:
    """HumanEval / MBPP / BCB via the pilot's payload shapes + interpreter override."""
    code = extract_code(completion)
    if code is None:
        return None
    bench = item["benchmark"]
    if bench == "humaneval":
        payload = f"{code}\n\n{item['test_code']}\n\ncheck({item['entry_point']})\n"
    elif bench == "mbpp_full":
        payload = f"{code}\n\n{item['test_setup_code']}\n\n{item['test_code']}\n"
    elif bench == "bigcodebench_full":
        payload = (
            f"{code}\n\n{item['test_code']}\n\n"
            "import unittest\n"
            "_r = unittest.main(exit=False, argv=['x'], verbosity=0).result\n"
            "raise SystemExit(0 if _r.wasSuccessful() else 1)\n"
        )
    else:
        raise ValueError(f"no pilot-shape verifier for {bench!r}")
    return _run_code(payload, CODE_EXEC_TIMEOUT_S, python_exe)


def _verdict_one(task: tuple) -> tuple[str, int, bool | None]:
    """Process-pool worker: one (item, rollout) verdict."""
    item, k, completion, python_exe = task
    bench = item["benchmark"]
    if bench == "math_full":
        v = verify_math(completion, item)
    elif bench == "mmlu_pro_full":
        v = verify_mcq(completion, item)
    elif bench == "lcb_v5":
        v = _verify_lcb(completion, item, python_exe)
    elif bench == "leetcode":
        v = _verify_leetcode(completion, item, python_exe)
    elif bench in CODE_BENCHMARKS:
        v = _verify_pilot_code(completion, item, python_exe)
    else:
        raise ValueError(f"no verifier for {bench!r}")
    return item["item_id"], k, v


# ---------------------------------------------------------------------------
# generation (chunked + checkpointed)
# ---------------------------------------------------------------------------


def _rollouts_path(out_root: Path, benchmark: str) -> Path:
    return out_root / surface_of(benchmark) / "rollouts" / f"{benchmark}.jsonl"


def _load_done_rollouts(path: Path) -> dict[str, list[str]]:
    done: dict[str, list[str]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:  # text-mode iteration, never splitlines()
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    done[row["item_id"]] = row["completions"]
    return done


def phase_gen(benchmark: str, out_root: Path, *, smoke: bool) -> None:
    """K sampled rollouts per item; chunked vLLM generate; per-chunk checkpoint."""
    items = LOADERS[benchmark]()
    if benchmark == "lcb_v5":
        items = _apply_dedup(items, out_root)
    if smoke:
        items = items[:SMOKE_N]
        print(f"[gen] SMOKE: {len(items)} items (production K/caps/engine)", flush=True)
    roll_path = _rollouts_path(out_root, benchmark)
    roll_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_rollouts(roll_path)
    pending = [it for it in items if it["item_id"] not in done]
    print(
        f"[gen] {benchmark}: {len(items)} items, {len(done)} resumed, {len(pending)} pending",
        flush=True,
    )
    if pending:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tok = AutoTokenizer.from_pretrained(MODEL)
        engine_kwargs: dict = {}
        if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
            engine_kwargs["enforce_eager"] = True
        if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
            engine_kwargs["enable_prefix_caching"] = False
        llm = LLM(
            model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, seed=SEED, **engine_kwargs
        )
        # Per-request seed deliberately UNSET (pilot rationale: independent draws
        # beat per-request reproducibility for a rate DV).
        params = SamplingParams(
            n=K_ROLLOUTS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS[benchmark],
        )
        t0 = time.time()
        cap_hits = 0
        n_gen = 0
        for lo in range(0, len(pending), GEN_CHUNK):
            chunk = pending[lo : lo + GEN_CHUNK]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": it["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for it in chunk
            ]
            print(
                f"[vllm-chunk] {benchmark} chunk {lo // GEN_CHUNK + 1}/"
                f"{(len(pending) + GEN_CHUNK - 1) // GEN_CHUNK} ({len(chunk)} prompts)",
                flush=True,
            )
            outs = llm.generate(prompts, params, use_tqdm=False)
            assert len(outs) == len(chunk), (len(outs), len(chunk))
            with roll_path.open("a", encoding="utf-8") as fh:
                for it, o in zip(chunk, outs, strict=True):
                    assert len(o.outputs) == K_ROLLOUTS, len(o.outputs)
                    cap_hits += sum(1 for c in o.outputs if c.finish_reason == "length")
                    fh.write(
                        json.dumps(
                            {
                                "item_id": it["item_id"],
                                "benchmark": benchmark,
                                "completions": [c.text for c in o.outputs],
                                "finish_reasons": [c.finish_reason for c in o.outputs],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                fh.flush()
            n_gen += len(chunk)
            print(
                f"[gen] {benchmark} unit {n_gen}/{len(pending)} "
                f"cap_hits={cap_hits}/{n_gen * K_ROLLOUTS} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        frac = cap_hits / max(1, n_gen * K_ROLLOUTS)
        if frac > CAP_HIT_TRIGGER:
            print(
                f"[gen] WARNING: {benchmark} cap-hit {frac:.1%} exceeds the "
                f"{CAP_HIT_TRIGGER:.0%} re-generation trigger at "
                f"max_tokens={MAX_TOKENS[benchmark]} (report + re-gen decision is the "
                "dispatcher's)",
                flush=True,
            )
    # Completeness: every item has K completions on disk.
    done = _load_done_rollouts(roll_path)
    missing = [it["item_id"] for it in items if it["item_id"] not in done]
    if missing:
        raise RuntimeError(f"{benchmark}: {len(missing)} items missing rollouts after gen")
    print(f"[gen] {benchmark} complete: {len(items)} items x {K_ROLLOUTS} rollouts", flush=True)


# ---------------------------------------------------------------------------
# verification (process-pooled + checkpointed)
# ---------------------------------------------------------------------------


def phase_verify(
    benchmark: str, out_root: Path, *, smoke: bool, bcb_python: str | None, workers: int
) -> None:
    """Programmatic verdicts for every stored rollout; writes the gen JSON."""
    items = LOADERS[benchmark]()
    if benchmark == "lcb_v5":
        items = _apply_dedup(items, out_root)
    if smoke:
        items = items[:SMOKE_N]
    rolls = _load_done_rollouts(_rollouts_path(out_root, benchmark))
    missing = [it["item_id"] for it in items if it["item_id"] not in rolls]
    if missing:
        raise RuntimeError(f"{benchmark}: {len(missing)} items lack rollouts — run --phase gen")

    verd_path = out_root / surface_of(benchmark) / "rollouts" / f"{benchmark}_verdicts.jsonl"
    done: dict[tuple[str, int], bool | None] = {}
    if verd_path.exists():
        with verd_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    done[(row["item_id"], row["k"])] = row["verdict"]
    python_exe = bcb_python if benchmark == "bigcodebench_full" else None
    tasks = [
        (it, k, rolls[it["item_id"]][k], python_exe)
        for it in items
        for k in range(K_ROLLOUTS)
        if (it["item_id"], k) not in done
    ]
    print(f"[verify] {benchmark}: {len(done)} resumed, {len(tasks)} pending", flush=True)
    t0 = time.time()
    n_done = 0
    for lo in range(0, len(tasks), VERIFY_CHUNK):
        chunk = tasks[lo : lo + VERIFY_CHUNK]
        if workers > 1 and benchmark != "mmlu_pro_full":  # regex verify is cheap, pool overhead
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_verdict_one, chunk))
        else:
            results = [_verdict_one(t) for t in chunk]
        with verd_path.open("a", encoding="utf-8") as fh:
            for item_id, k, v in results:
                fh.write(json.dumps({"item_id": item_id, "k": k, "verdict": v}) + "\n")
                done[(item_id, k)] = v
            fh.flush()
        n_done += len(chunk)
        print(
            f"[verify] {benchmark} unit {n_done}/{len(tasks)} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    out_items = []
    for it in items:
        verdicts = [done[(it["item_id"], k)] for k in range(K_ROLLOUTS)]
        row = {
            k: it.get(k)
            for k in ("item_id", "benchmark", "level", "subject", "category", "difficulty")
            if it.get(k) is not None or k in ("item_id", "benchmark")
        }
        row["verdicts"] = verdicts
        out_items.append(row)
    full = [
        sum(1 for v in r["verdicts"] if v) / K_ROLLOUTS
        for r in out_items
        if all(v is not None for v in r["verdicts"])
    ]
    stats = spread_stats(full, K_ROLLOUTS) if full else None
    n_unparsed = sum(1 for r in out_items for v in r["verdicts"] if v is None)
    payload = {
        "benchmark": benchmark,
        "model": MODEL,
        "k_rollouts": K_ROLLOUTS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS[benchmark],
        "n_items": len(out_items),
        "n_unparsed_rollouts": n_unparsed,
        "unparsed_fraction": n_unparsed / max(1, len(out_items) * K_ROLLOUTS),
        "stats_full_k": stats,
        "admissible": bool(stats and admissible(stats)),
        "smoke": smoke,
        "items": out_items,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    payload.update(as_metadata_dict(git_provenance(), phase=f"gen-verify-{benchmark}"))
    out_path = out_root / surface_of(benchmark) / f"{benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_path)
    print(
        f"[verify] {benchmark}: n={len(out_items)} unparsed={payload['unparsed_fraction']:.1%} "
        f"reliability={None if stats is None else round(stats['reliability'], 3)} "
        f"{'ADMISSIBLE' if payload['admissible'] else 'REJECT/NA'} -> {out_path}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# upload (rollout text -> HF, <=9 MB line-shards)
# ---------------------------------------------------------------------------


def _shard_jsonl(src: Path, shard_dir: Path, stem: str, max_bytes: int = 9 * 1024 * 1024) -> int:
    """Line-split a JSONL into <=9 MB shards (upload-policy non-LFS split rule)."""
    shard_idx, size, fh = 0, 0, None
    try:
        with src.open(encoding="utf-8") as inp:
            for line in inp:
                if fh is None or size + len(line.encode()) > max_bytes:
                    if fh is not None:
                        fh.close()
                    fh = (shard_dir / f"{stem}.shard{shard_idx:02d}.jsonl").open(
                        "w", encoding="utf-8"
                    )
                    shard_idx, size = shard_idx + 1, 0
                fh.write(line)
                size += len(line.encode())
    finally:
        if fh is not None:
            fh.close()
    return shard_idx


def phase_upload(benchmark: str, out_root: Path) -> None:
    """Shard + upload the benchmark's rollout text (and verdicts) to the HF data repo."""
    from explore_persona_space.orchestrate import hub

    surface = surface_of(benchmark)
    src = _rollouts_path(out_root, benchmark)
    if not src.exists():
        raise FileNotFoundError(f"no rollouts at {src}")
    shard_dir = src.parent / f"{benchmark}_upload_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    n_shards = _shard_jsonl(src, shard_dir, benchmark)
    verdicts = src.parent / f"{benchmark}_verdicts.jsonl"
    if verdicts.exists():
        _shard_jsonl(verdicts, shard_dir, f"{benchmark}_verdicts")
    prefix = f"{HF_GEN_PREFIX}/{surface}"
    url = hub._upload(  # folder branch: ONE upload_folder commit for the shard set
        shard_dir, hub.DEFAULT_DATASET_REPO, repo_type="dataset", path_in_repo=prefix
    )
    if not url:
        raise RuntimeError(f"upload returned no path for {shard_dir} -> {prefix} (fail-loud)")
    print(f"[upload] {benchmark}: {n_shards} text shards -> {prefix}", flush=True)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

PHASES = ("dedup", "gen", "verify", "upload")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--phase", choices=PHASES + ("all",), required=False, default="all")
    ap.add_argument("--benchmark", choices=sorted(LOADERS))
    ap.add_argument("--surface", choices=sorted(SURFACES))
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_N} items/benchmark, *_smoke root")
    ap.add_argument("--bcb-python", default=None, help="/opt/bcb-venv/bin/python for the BCB gate")
    ap.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Execute the deferred imports the smoke path may skip (#606 class).
        # vllm + transformers stay deferred here (GPU-fenced phase; recorded in
        # the smoke-architecture marker as GPU-fenced-inspected).
        from datasets import get_dataset_config_names, load_dataset  # noqa: F401
        from math_verify import parse, verify  # noqa: F401

        from explore_persona_space.orchestrate import hub
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )

        assert callable(hub._upload) and hasattr(hub, "DEFAULT_DATASET_REPO")
        print("[gen] import-check OK", flush=True)
        raise SystemExit(0)

    out_root = args.out_root
    if args.smoke and out_root == OUT_ROOT:
        out_root = Path(str(OUT_ROOT) + "_smoke")  # smoke never overwrites production paths
    benches = (
        [args.benchmark]
        if args.benchmark
        else (SURFACES[args.surface] if args.surface else sorted(LOADERS))
    )
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    ran_gen = False
    for phase in phases:
        if phase == "dedup":
            dedup_lcb_against_leetcode(out_root)
            continue
        for bench in benches:
            if phase == "gen":
                phase_gen(bench, out_root, smoke=args.smoke)
                ran_gen = True
            elif phase == "verify":
                phase_verify(
                    bench,
                    out_root,
                    smoke=args.smoke,
                    bcb_python=args.bcb_python,
                    workers=args.workers,
                )
            elif phase == "upload":
                phase_upload(bench, out_root)
    sys.stdout.flush()
    sys.stderr.flush()
    if ran_gen:
        # vLLM engine children survive interpreter finalization and deadlock a
        # sys.exit(0) terminal (gotchas.md #1739/#2149) — durables are already
        # flushed to disk above.
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
