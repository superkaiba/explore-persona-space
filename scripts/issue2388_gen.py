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

    dedup   the ALWAYS-ON LCB<->LeetCode dedup (binding report).
    gate    consolidate the code-surface gate verdicts (G1 env gate from the
            code-control report + the fresh full-pool G3 spread re-read +
            the fork-5 APPS trigger arithmetic) into the BINDING
            ``code_gate.json`` consumed by BCB/APPS generation and by
            ``issue2388_dv_build`` fit inclusion.
    gen     K=5 sampled rollouts via vLLM (temp 1.0, top_p 1.0, chat template),
            CHUNKED generate (#664 deadlock prevention; per-chunk append +
            resume by item_id + a generating-params sidecar), ONE engine
            shared across benchmarks in-process, cap-hit fraction reported
            per chunk; ``--regen-cap-hit`` re-generates cap-hit rows at a
            2x token cap (the >2% re-generation trigger's executable path).
    verify  programmatic verdicts (math-verify / option-match / unit-test
            execution in the hardened sandbox), process-pooled, checkpointed
            per chunk; per-benchmark spread_stats (pilot reuse) +
            admissibility + the durable full-pool cap-hit fraction (the BCB
            full-pool G3 re-measure input).
    upload  rollout TEXT -> HF ``issue2388_correctness/raw_completions/gen/
            <surface>/`` (<=9 MB JSONL line-shards; upload-policy split rule;
            smoke uploads land under ``raw_completions/gen_smoke/`` — never
            the production prefix).

Code sandbox: model-generated code executes with a SCRUBBED allowlisted env
(no API/HF tokens), an isolated writable HOME/TMPDIR/cwd, CPU/file-size/
address-space rlimits, its own process group (killpg on timeout), and — where
unprivileged user namespaces allow — no network (``unshare -rn``; availability
probed per process and recorded). NOT a hardened security boundary against a
targeted adversary (no seccomp); the residual is a persisted concern.

Smoke mode (``--smoke``): 20 contexts per benchmark through the SAME
entrypoint, engine path, K, and caps; out-roots AND HF upload prefixes rebind
to ``*_smoke`` so smoke artifacts never overwrite production paths. Loader
count-asserts run on the FULL pool BEFORE the smoke slice (production gates
are never downgraded — smoke-blind-spots rule).

Terminal: ``os._exit(0)`` after flush on the gen phase (vLLM engine children
survive interpreter finalization otherwise — gotchas.md #1739/#2149).

CONTENT HYGIENE: benchmark questions are benign; logs still carry ids/counts.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
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

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

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
# APPS contingency pool (plan section 4 fork 5: enters ONLY when the realized
# post-dedup, post-BCB-gate code train split falls below d = 3,584).
# codeparrot/apps is a SCRIPT dataset (datasets>=3 removed script loading), so
# the raw train/test jsonl files are read at the pinned revision (same
# workaround as LCB). Schema-from-artifact (train.jsonl row 0 probed
# 2026-08-20 at this revision): row keys [difficulty, id, input_output,
# question, solutions, starter_code, url]; ``input_output`` is a JSON string
# {"inputs": [...], "outputs": [...]} (+ "fn_name" on functional problems);
# ``solutions`` is a JSON string list.
APPS_REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
APPS_FILES = ("train.jsonl", "test.jsonl")
APPS_PILOT_N = 200  # plan fork 5: 200-problem APPS-intro pilot first
# Pilot verify verdict file (NEVER code/apps_intro.json — the gate's g3_apps
# full-pool read must be structurally unable to consume a pilot slice).
APPS_PILOT_REPORT = "apps_intro_pilot.json"
CODE_TRAIN_FLOOR = 3584  # d — the fork-5 APPS trigger reads the realized count

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
    "apps_intro": None,  # contingency pool — realized count recorded by the gate, not pinned
}

# Plan-registered cap: max_new_tokens 2048 on ALL surfaces (plan section 10
# "Protocol"; section 11: MMLU-Pro raised from the pilot's 1024 after its
# measured 1.9% cap-hit; HumanEval/MBPP raised direction-safe).
MAX_TOKENS = {
    "math_full": 2048,
    "mmlu_pro_full": 2048,
    "humaneval": 2048,
    "mbpp_full": 2048,
    "bigcodebench_full": 2048,
    "lcb_v5": 2048,
    "leetcode": 2048,
    "apps_intro": 2048,
}

SURFACES = {
    "math": ["math_full"],
    "mcq": ["mmlu_pro_full"],
    # apps_intro is CONTINGENCY-ONLY: it never enters the default surface
    # roster; the gate activates it (fork 5) and it is then run explicitly via
    # --benchmark apps_intro.
    "code": ["humaneval", "mbpp_full", "bigcodebench_full", "lcb_v5", "leetcode"],
}
CODE_BENCHMARKS = set(SURFACES["code"]) | {"apps_intro"}
CODE_EXEC_TIMEOUT_S = 15
SMOKE_N = 20
GEN_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
VERIFY_CHUNK = 200
CAP_HIT_TRIGGER = 0.02  # CLAUDE.md re-generation trigger (reported per family)


def surface_of(benchmark: str) -> str:
    if benchmark in CODE_BENCHMARKS:
        return "code"
    for s, benches in SURFACES.items():
        if benchmark in benches:
            return s
    raise ValueError(f"unknown benchmark {benchmark!r}")


def _assert_unique_ids(items: list[dict], benchmark: str) -> None:
    """item_id collisions silently merge rows at resume/verdict/dedup joins."""
    ids = [it["item_id"] for it in items]
    if len(set(ids)) != len(ids):
        from collections import Counter

        dupes = [k for k, c in Counter(ids).items() if c > 1]
        raise RuntimeError(f"{benchmark}: {len(dupes)} duplicate item_ids (e.g. {dupes[:3]})")


# ---------------------------------------------------------------------------
# new loaders (full pools). Pilot loaders are subsampled by BUDGETS — the
# production pools re-load full splits with the SAME prompt templates.
# ---------------------------------------------------------------------------


def _slugify(title: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.strip().lower())).strip("-")


def _dedup_key(text: str) -> str:
    """Alnum-only dedup key, applied to BOTH sides of the LCB<->LeetCode match.

    Official LeetCode slugs DROP apostrophes ("Pascal's Triangle" ->
    "pascals-triangle") while a dash-substituting slugify keeps them as
    dashes ("pascal-s-triangle") — punctuation-bearing titles would survive
    the binding dedup as true duplicates. Alnum-only collapses both
    conventions to one key (r1 g3 Concern 5).
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


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


class _NoGlobalsUnpickler(pickle.Unpickler):
    """Restricted unpickler: LCB private_test_cases pickle to a PLAIN str
    (their own translate_private_test_cases recipe), which needs NO globals —
    any global resolution is an injection attempt and raises
    (unsafe-lcb-pickle: unrestricted pickle.loads executes constructors
    embedded in downloaded dataset content)."""

    def find_class(self, module, name):  # noqa: ARG002 — signature fixed by pickle
        raise pickle.UnpicklingError(
            f"forbidden global {module}.{name} in LCB private_test_cases "
            "(expected a plain pickled str)"
        )


def _restricted_pickle_loads(data: bytes):
    """Module-level (testable) restricted-pickle entry for LCB private tests."""
    return _NoGlobalsUnpickler(io.BytesIO(data)).load()


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
    import zlib

    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    raw_rows: list[dict] = []
    for fname in LCB_V5_FILES:
        path = retry_transient(
            lambda fname=fname: hf_hub_download(
                "livecodebench/code_generation_lite",
                fname,
                repo_type="dataset",
                revision=LCB_REVISION,
            ),
            what=f"hf_hub_download(lcb_v5/{fname})",
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
                _restricted_pickle_loads(zlib.decompress(base64.b64decode(r["private_test_cases"])))
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
                "dedup_key": _dedup_key(r["question_title"]),
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
                    "dedup_key": _dedup_key(str(r["task_id"])),
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


def load_apps_intro() -> list[dict]:
    """APPS introductory problems (CONTINGENCY pool — plan section 4 fork 5).

    Raw train/test jsonl at the pinned revision (script dataset — same
    workaround as LCB; observed row schema in the module constants block).
    Rows without any test cases (or with input/output length mismatch) are
    DROPPED with a digest count (unverifiable); the realized count is
    recorded by the gate, not pinned.
    """
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    rows: list[dict] = []
    n_seen = n_unverifiable = 0
    for fname in APPS_FILES:
        path = retry_transient(
            lambda fname=fname: hf_hub_download(
                "codeparrot/apps", fname, repo_type="dataset", revision=APPS_REVISION
            ),
            what=f"hf_hub_download(apps/{fname})",
        )
        split = fname.split(".")[0]
        with open(path, encoding="utf-8") as fh:  # text-mode iteration, never splitlines()
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                for field in ("difficulty", "id", "question", "input_output", "solutions"):
                    assert field in r, f"APPS field drift: {field!r} missing (have {sorted(r)[:8]})"
                if r["difficulty"] != "introductory":
                    continue
                n_seen += 1
                io_payload = json.loads(r["input_output"]) if r["input_output"] else {}
                inputs = io_payload.get("inputs") or []
                outputs = io_payload.get("outputs") or []
                if not inputs or len(inputs) != len(outputs):
                    n_unverifiable += 1
                    continue
                solutions = json.loads(r["solutions"]) if r["solutions"] else []
                starter = r.get("starter_code") or ""
                prompt = (
                    "Solve the following programming problem.\n\n"
                    + r["question"]
                    + (
                        "\n\nUse this starter code (complete the class/method as given):\n\n"
                        f"```python\n{starter}\n```"
                        if starter.strip()
                        else "\n\nRead input from stdin and write the answer to stdout."
                    )
                    + "\n\nReturn the complete solution inside a single ```python code block."
                )
                rows.append(
                    {
                        "item_id": f"apps-{split}-{r['id']}",
                        "benchmark": "apps_intro",
                        "fn_name": io_payload.get("fn_name"),
                        "inputs": inputs,
                        "outputs": outputs,
                        "canonical_solutions": solutions[:2],  # code-control positives
                        "starter_code": starter,
                        "prompt": prompt,
                    }
                )
    if not rows:
        raise RuntimeError("apps_intro: 0 verifiable introductory rows at the pinned revision")
    print(
        f"[load] apps_intro: {len(rows)} kept / {n_seen} introductory "
        f"(unverifiable {n_unverifiable})",
        flush=True,
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
    "apps_intro": load_apps_intro,
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
    _assert_unique_ids(lcb, "lcb_v5")
    _assert_unique_ids(leet, "leetcode")
    leet_keys = {r["dedup_key"] for r in leet}
    dropped = sorted(
        r["item_id"] for r in lcb if r["platform"] == "leetcode" and r["dedup_key"] in leet_keys
    )
    report = {
        "n_lcb": len(lcb),
        "n_leetcode": len(leet),
        "n_lcb_leetcode_platform": sum(1 for r in lcb if r["platform"] == "leetcode"),
        "n_dropped_lcb": len(dropped),
        "dropped_lcb_item_ids": dropped,
        "rule": (
            "LCB platform==leetcode with alnum-only title key in LeetCodeDataset "
            "alnum-only task_id keys (r1 g3 Concern 5: dash-slugify under-matched "
            "punctuation-bearing titles vs official slugs)"
        ),
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
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(report, indent=1))
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
# code-surface gate (G1 env gate + G3 spread gate + fork-5 APPS trigger)
# ---------------------------------------------------------------------------

CONTROL_REPORT = "code_harness_control.json"  # scripts/issue2388_code_control.py --out default


def gate_path(out_root: Path) -> Path:
    return out_root / "code" / "code_gate.json"


def load_gate(out_root: Path) -> dict:
    path = gate_path(out_root)
    if not path.exists():
        raise FileNotFoundError(
            f"code gate verdict missing at {path} — run --phase gate first (G1/G3 consumers "
            "never run ungated; plan section 7)"
        )
    return json.loads(path.read_text())


def code_roster_from_gate_fields(gate: dict) -> list[str]:
    """Realized code-surface benchmark roster from gate-verdict fields.

    The ONE resolution rule shared by every downstream consumer (dv_build /
    capture upload / fits — r3 Critical family "contingency state not carried
    across persisted producers and consumers"): BCB enters ONLY on
    ``bcb_fit_allowed``; apps_intro enters ONLY on ``apps_activated`` (fork 5);
    neither rides any bare default roster. Accepts the ``code_gate.json``
    verdict OR a labeling.json ``gate_decisions`` echo — both carry the two
    keys. Fail-loud on an unresolved BCB verdict.
    """
    if gate.get("bcb_fit_allowed") is None:
        raise RuntimeError(
            "bcb_fit_allowed unresolved in code_gate.json (G1 control or G3 full-pool spread "
            "missing) — re-run issue2388_gen.py --phase gate after the control + full verify"
        )
    benches = ["humaneval", "mbpp_full", "lcb_v5", "leetcode"]
    if gate["bcb_fit_allowed"]:
        benches.insert(2, "bigcodebench_full")
    if gate.get("apps_activated"):
        benches.append("apps_intro")
    return benches


def phase_gate(out_root: Path, *, control_report: Path | None = None) -> dict:
    """Consolidate the code-surface gate verdicts into the BINDING code_gate.json.

    Idempotent — consumes whatever inputs exist NOW and records availability:

    - G1 (env gate): the code-control report's bigcodebench ``harness_ok``
      (25/25 through /opt/bcb-venv + flaky < 2%). ABSENT -> BCB generation is
      REFUSED (fail-loud, never a silent keep).
    - G3 (spread gate, BINDING for BCB/APPS): the FRESH full-pool spread
      re-read from this run's own verify output (``code/bigcodebench_full.json``,
      asserted non-smoke full-pool) — NEVER ``spread_pilot/`` (its
      ``admissible: true`` flag is harness-contaminated; plan section 7).
    - fork-5 APPS contingency chain (plan section 4 fork 5), STAGED:
      (a) trigger: est. post-dedup post-BCB-gate code train count < d = 3,584
      -> ``apps_required``; (b) APPS canonical control 25/25 (each solution
      run twice, flaky < 2%) -> ``apps_pilot_gen_allowed`` (the 200-problem
      pilot may generate); (c) pilot verify (``code/apps_intro_pilot.json``,
      exactly APPS_PILOT_N items, same admissibility rule) ->
      ``apps_full_gen_allowed`` (the full pool may generate); (d) BINDING
      full-pool G3 re-read of ``code/apps_intro.json`` (never the pilot
      slice, never smoke) -> ``apps_activated`` (fit inclusion). Re-run this
      phase after each stage lands to advance the chain.

    Consumers: ``phase_gen``/``phase_verify`` (bigcodebench_full / apps_intro
    refuse without the right verdict) and ``issue2388_dv_build`` (BCB/APPS fit
    inclusion keys on ``apps_activated`` ONLY).
    """
    dedup_p = out_root / "code" / "dedup_report.json"
    if not dedup_p.exists():
        raise FileNotFoundError(f"dedup report missing at {dedup_p} — run --phase dedup first")
    dedup = json.loads(dedup_p.read_text())
    n_lcb_kept = int(dedup["n_lcb"]) - int(dedup["n_dropped_lcb"])

    ctrl_p = Path(control_report) if control_report else out_root / CONTROL_REPORT
    g1: dict = {"available": ctrl_p.exists(), "path": str(ctrl_p)}
    apps_ctrl: dict | None = None
    if ctrl_p.exists():
        ctrl = json.loads(ctrl_p.read_text())["benchmarks"]
        bcb = ctrl.get("bigcodebench")
        g1["harness_ok"] = bool(bcb["harness_ok"]) if bcb else None
        g1["flaky_mismatch_fraction"] = bcb.get("flaky_mismatch_fraction") if bcb else None
        # Row-grain freshness (r4 code-control-preserved-row-freshness): the
        # gate CONSUMES each control row's own run timestamp, so a merged
        # report's preserved (not re-run) BCB/APPS row is auditable in the
        # gate verdict rather than silently indistinguishable from fresh.
        g1["bcb_control_ts"] = bcb.get("control_ts") if bcb else None
        apps_ctrl = ctrl.get("apps_intro")
        g1["apps_harness_ok"] = bool(apps_ctrl["harness_ok"]) if apps_ctrl else None
        g1["apps_control_ts"] = apps_ctrl.get("control_ts") if apps_ctrl else None
    else:
        g1["harness_ok"] = None
        g1["apps_harness_ok"] = None
        g1["bcb_control_ts"] = None
        g1["apps_control_ts"] = None

    spread_p = out_root / "code" / "bigcodebench_full.json"
    g3: dict = {"available": spread_p.exists(), "provenance": str(spread_p)}
    if spread_p.exists():
        sp = json.loads(spread_p.read_text())
        # Full-pool assertion: the G3 re-read must be the FULL BCB pool, never
        # a smoke slice and never the pilot file (provenance path is this
        # run's own verify output by construction).
        full_pool = (not sp.get("smoke")) and int(sp["n_items"]) == EXPECTED_COUNTS[
            "bigcodebench_full"
        ]
        g3["full_pool"] = bool(full_pool)
        g3["admissible"] = bool(sp["admissible"]) if full_pool else None
    else:
        g3["full_pool"] = None
        g3["admissible"] = None

    bcb_gen_allowed = bool(g1["harness_ok"])
    # Fit inclusion needs BOTH gates (G3 resolves only after full-pool verify).
    bcb_fit_allowed = None
    if g1["harness_ok"] is not None:
        if not g1["harness_ok"]:
            bcb_fit_allowed = False
        elif g3["admissible"] is not None:
            bcb_fit_allowed = bool(g3["admissible"])

    base_pool = (
        EXPECTED_COUNTS["humaneval"]
        + EXPECTED_COUNTS["mbpp_full"]
        + n_lcb_kept
        + EXPECTED_COUNTS["leetcode"]
    )
    pool_with_bcb = base_pool + EXPECTED_COUNTS["bigcodebench_full"]
    est = {
        "n_lcb_kept_post_dedup": n_lcb_kept,
        "est_train_with_bcb": round(0.7 * pool_with_bcb),
        "est_train_without_bcb": round(0.7 * base_pool),
        "code_train_floor_d": CODE_TRAIN_FLOOR,
    }
    if bcb_fit_allowed is None:
        apps_required = None  # unresolved until the BCB verdict lands
    else:
        est_train = est["est_train_with_bcb"] if bcb_fit_allowed else est["est_train_without_bcb"]
        apps_required = est_train < CODE_TRAIN_FLOOR

    # --- fork-5 APPS chain stage (b): canonical control, 25/25 twice, flaky<2%
    # (STRICTER than the control report's own >=0.90 harness_ok — the plan
    # names 25/25 verbatim for the APPS contingency).
    g1_apps: dict = {"available": apps_ctrl is not None}
    if apps_ctrl is not None:
        g1_apps["n_control"] = apps_ctrl.get("n_control")
        g1_apps["best_pass_rate"] = apps_ctrl.get("best_pass_rate")
        g1_apps["runs_per_item"] = apps_ctrl.get("runs_per_item")
        g1_apps["flaky_mismatch_fraction"] = apps_ctrl.get("flaky_mismatch_fraction")
        g1_apps["control_25_25"] = bool(
            int(apps_ctrl.get("n_control") or 0) >= 25
            and float(apps_ctrl.get("best_pass_rate") or 0.0) == 1.0
            and int(apps_ctrl.get("runs_per_item") or 0) >= 2
            and float(apps_ctrl.get("flaky_mismatch_fraction", 1.0)) < 0.02
        )
    else:
        g1_apps["control_25_25"] = None

    # --- fork-5 stage (c): the 200-problem pilot verdict (written by
    # phase_verify --apps-pilot to its OWN path, never code/apps_intro.json).
    pilot_p = out_root / "code" / APPS_PILOT_REPORT
    apps_pilot: dict = {"available": pilot_p.exists(), "provenance": str(pilot_p)}
    if pilot_p.exists():
        pp = json.loads(pilot_p.read_text())
        pilot_scope_ok = (
            bool(pp.get("pilot")) and (not pp.get("smoke")) and int(pp["n_items"]) == APPS_PILOT_N
        )
        apps_pilot["n_items"] = int(pp["n_items"])
        apps_pilot["pilot_scope_ok"] = pilot_scope_ok
        apps_pilot["admissible"] = bool(pp["admissible"]) if pilot_scope_ok else None
    else:
        apps_pilot["pilot_scope_ok"] = None
        apps_pilot["admissible"] = None

    # --- fork-5 stage (d): BINDING full-pool G3 for APPS (plan section 7:
    # "BCB/APPS fit inclusion keys ONLY on the fresh full-pool G3 re-read").
    apps_spread_p = out_root / "code" / "apps_intro.json"
    g3_apps: dict = {"available": apps_spread_p.exists(), "provenance": str(apps_spread_p)}
    if apps_spread_p.exists():
        asp = json.loads(apps_spread_p.read_text())
        # APPS has no pinned count (EXPECTED_COUNTS is None — contingency
        # pool); a non-pilot non-smoke verify is full-pool BY CONSTRUCTION
        # (phase_verify refuses when any loader item lacks rollouts), and the
        # > APPS_PILOT_N assert refuses a pilot-sized file at this path.
        apps_full_pool = (
            (not asp.get("smoke")) and (not asp.get("pilot")) and int(asp["n_items"]) > APPS_PILOT_N
        )
        g3_apps["full_pool"] = bool(apps_full_pool)
        g3_apps["n_items"] = int(asp["n_items"])
        g3_apps["admissible"] = bool(asp["admissible"]) if apps_full_pool else None
    else:
        g3_apps["full_pool"] = None
        g3_apps["admissible"] = None

    apps_pilot_gen_allowed = bool(apps_required) and bool(g1_apps["control_25_25"])
    apps_full_gen_allowed = apps_pilot_gen_allowed and bool(apps_pilot["admissible"])
    # FIT inclusion (dv_build): the WHOLE chain incl. the binding full-pool G3.
    apps_activated = apps_full_gen_allowed and bool(g3_apps["admissible"])

    verdict = {
        "g1": g1,
        "g3_bcb": g3,
        "g1_apps": g1_apps,
        "apps_pilot": apps_pilot,
        "g3_apps": g3_apps,
        "bcb_gen_allowed": bcb_gen_allowed,
        "bcb_fit_allowed": bcb_fit_allowed,
        "apps_required": apps_required,
        "apps_pilot_gen_allowed": apps_pilot_gen_allowed,
        "apps_full_gen_allowed": apps_full_gen_allowed,
        "apps_activated": apps_activated,
        "pool_arithmetic": est,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    verdict.update(as_metadata_dict(git_provenance(), phase="gen-gate"))
    path = gate_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(verdict, indent=1))
    print(
        f"[gate] bcb_gen_allowed={bcb_gen_allowed} bcb_fit_allowed={bcb_fit_allowed} "
        f"apps_required={apps_required} apps_pilot_gen_allowed={apps_pilot_gen_allowed} "
        f"apps_full_gen_allowed={apps_full_gen_allowed} apps_activated={apps_activated} "
        f"-> {path}",
        flush=True,
    )
    return verdict


def _require_gate_for(benchmark: str, out_root: Path, *, apps_pilot: bool = False) -> None:
    """BCB/APPS enter generation ONLY behind the recorded gate verdicts."""
    if benchmark == "bigcodebench_full":
        gate = load_gate(out_root)
        if not gate.get("bcb_gen_allowed"):
            raise RuntimeError(
                "G1: BCB env gate not passed (bcb_gen_allowed is false/unresolved in "
                f"{gate_path(out_root)}) — BCB never enters generation ungated; on a "
                "recorded G1 FAIL the fork-5 branch is DROP BCB -> APPS fallback"
            )
    elif benchmark == "apps_intro":
        gate = load_gate(out_root)
        if apps_pilot:
            if not gate.get("apps_pilot_gen_allowed"):
                raise RuntimeError(
                    "fork-5: the APPS pilot needs apps_required=True AND the APPS "
                    "canonical control 25/25 twice with flaky < 2% "
                    "(apps_pilot_gen_allowed is false/unresolved in "
                    f"{gate_path(out_root)}) — run issue2388_code_control.py "
                    "--benchmarks apps_intro, then re-run --phase gate"
                )
        elif not gate.get("apps_full_gen_allowed"):
            raise RuntimeError(
                "fork-5: FULL apps_intro generation needs the ADMISSIBLE 200-problem "
                "pilot verdict on top of the trigger + control "
                "(apps_full_gen_allowed is false/unresolved in "
                f"{gate_path(out_root)}) — run --phase gen/verify --benchmark "
                "apps_intro --apps-pilot, then re-run --phase gate"
            )


# ---------------------------------------------------------------------------
# code verification (new benchmarks; pilot's sandbox pattern)
# ---------------------------------------------------------------------------


# Sandbox env allowlist: model-generated code must NEVER inherit the parent's
# credential env (HF_TOKEN / ANTHROPIC_API_KEY / RUNPOD_API_KEY / WANDB_API_KEY
# ride load_dotenv() into os.environ). Absolute interpreter paths make PATH
# venv-entries unnecessary.
_SANDBOX_ENV_KEYS = ("LANG", "LC_ALL")
_SANDBOX_MEM_BYTES = 4 << 30  # generous: BCB tests import sklearn/matplotlib
_SANDBOX_FSIZE_BYTES = 256 << 20
_UNSHARE_NET: bool | None = None


def _sandbox_env(tmpdir: str) -> dict[str, str]:
    env = {k: os.environ[k] for k in _SANDBOX_ENV_KEYS if k in os.environ}
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    env["HOME"] = tmpdir
    env["TMPDIR"] = tmpdir
    env["MPLCONFIGDIR"] = tmpdir  # matplotlib-importing BCB tests need a writable config dir
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _unshare_net_available() -> bool:
    """Once-per-process probe: unprivileged user+net namespaces (``unshare -rn``)
    give the sandbox NETWORK ISOLATION; when unavailable the sandbox FAIL-LOUDS
    (never a silent degrade — r2 Codex Critical 2) unless the operator sets
    ``EPM_I2388_SANDBOX_ALLOW_NET=1`` explicitly. Filesystem-namespace + seccomp
    residuals stay recorded (persisted concern sandbox-network-residual)."""
    global _UNSHARE_NET
    if _UNSHARE_NET is None:
        try:
            _UNSHARE_NET = (
                subprocess.run(
                    ["unshare", "-r", "-n", "true"], capture_output=True, timeout=10
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            _UNSHARE_NET = False
        print(f"[sandbox] unshare -rn network isolation: {_UNSHARE_NET}", flush=True)
    return _UNSHARE_NET


_SANDBOX_ALLOW_NET_ENV = "EPM_I2388_SANDBOX_ALLOW_NET"


def _require_sandbox_net_isolation() -> None:
    """Fail LOUD when network isolation cannot be established (no silent degrade).

    Called once per code-verify phase entry (before any pool worker spends)
    AND per sandboxed execution (defense in depth — pool workers re-probe in
    their own process). The explicit env override accepts network-visible
    execution and is RECORDED in the verify payload as
    ``sandbox_net_isolation: false``.
    """
    if _unshare_net_available() or os.environ.get(_SANDBOX_ALLOW_NET_ENV) == "1":
        return
    raise RuntimeError(
        "sandbox: unprivileged user+net namespaces are unavailable on this host "
        "(`unshare -rn` failed) — network isolation for model-generated code cannot "
        f"be established. Set {_SANDBOX_ALLOW_NET_ENV}=1 to explicitly accept "
        "network-visible execution (recorded as sandbox_net_isolation=false in the "
        "verify payload; persisted concern sandbox-network-residual)."
    )


def _run_sandboxed(
    argv: list[str],
    *,
    timeout_s: int,
    tmpdir: str,
    input_text: str | None = None,
) -> tuple[int, str]:
    """Hardened execution of model-generated code (r1 unsandboxed-generated-code).

    Scrubbed allowlisted env (no API/HF tokens), isolated writable
    HOME/TMPDIR/cwd, CPU + file-size + address-space rlimits, its own process
    group (killpg on timeout — grandchildren cannot outlive the verdict), and
    network off via ``unshare -rn`` — MANDATORY: when user namespaces are
    unavailable this RAISES instead of silently degrading (override:
    ``EPM_I2388_SANDBOX_ALLOW_NET=1``, recorded in the verify payload). NOT a
    security boundary against a targeted adversary (no filesystem namespace,
    no seccomp) — residual recorded as a persisted concern
    (sandbox-network-residual). Returns (returncode, stdout); timeout ->
    (-1, "").
    """
    import resource
    import signal

    _require_sandbox_net_isolation()  # fail-loud, never a silent net-visible run
    if _unshare_net_available():
        argv = ["unshare", "-r", "-n", *argv]

    def _limits() -> None:
        cpu = int(timeout_s) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_SANDBOX_FSIZE_BYTES, _SANDBOX_FSIZE_BYTES))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (_SANDBOX_MEM_BYTES, _SANDBOX_MEM_BYTES))
        except (ValueError, OSError):
            pass  # some hosts refuse lowering AS below current usage

    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmpdir,
        env=_sandbox_env(tmpdir),
        start_new_session=True,
        preexec_fn=_limits,
    )
    try:
        out, _err = proc.communicate(input=input_text, timeout=timeout_s)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)
        return -1, ""


def _run_code(payload: str, timeout_s: int, python_exe: str | None = None) -> bool:
    """Sandboxed snippet execution with an interpreter override (BCB /opt venv)."""
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="i2388sbx_")
    try:
        path = Path(tmpdir) / "snippet.py"
        path.write_text(payload)
        rc, _out = _run_sandboxed(
            [python_exe or sys.executable, str(path)], timeout_s=timeout_s, tmpdir=tmpdir
        )
        return rc == 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
    if stdin_tests and not _run_stdin_tests(code, stdin_tests, python_exe):
        return False
    return True


def _stdout_matches(got: str, want: str) -> bool:
    """Official LCB checker semantics: per-line compare, trailing-whitespace
    tolerant — never whole-string exact match (r1 g3 nit: codeforces/atcoder
    items false-negative on per-line trailing whitespace)."""
    got_lines = [ln.rstrip() for ln in got.strip().split("\n")]
    want_lines = [ln.rstrip() for ln in str(want).strip().split("\n")]
    return got_lines == want_lines


def _run_stdin_tests(code: str, tests: list[dict], python_exe: str | None) -> bool:
    """Sandboxed stdin/stdout test execution (LCB codeforces/atcoder + APPS)."""
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="i2388sbx_")
    try:
        path = Path(tmpdir) / "snippet.py"
        path.write_text(code)
        for t in tests:
            rc, out = _run_sandboxed(
                [python_exe or sys.executable, str(path)],
                timeout_s=CODE_EXEC_TIMEOUT_S,
                tmpdir=tmpdir,
                input_text=str(t["input"]),
            )
            if rc != 0 or not _stdout_matches(out, t["output"]):
                return False
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# APPS functional variant: APPS wraps a functional problem's expected return in
# a single-element list ~always; accept either shape (LCB's harness stays
# byte-identical — an unwrap there could false-positive genuine list outputs).
_APPS_FUNCTIONAL_HARNESS = _LCB_FUNCTIONAL_HARNESS.replace(
    "if not _eq(_got, _want):",
    "if not (_eq(_got, _want) or (isinstance(_want, list) and len(_want) == 1 "
    "and _eq(_got, _want[0]))):",
)
assert _APPS_FUNCTIONAL_HARNESS != _LCB_FUNCTIONAL_HARNESS, "APPS harness patch did not apply"


def _verify_apps(completion: str, item: dict, python_exe: str | None = None) -> bool | None:
    """APPS verdict: functional (fn_name; dual-compare harness) or stdin/stdout."""
    code = extract_code(completion)
    if code is None:
        return None
    if item.get("fn_name"):
        tests = [
            {
                "input": "\n".join(
                    json.dumps(a) for a in (inp if isinstance(inp, list) else [inp])
                ),
                "output": json.dumps(out),
                "testtype": "functional",
            }
            for inp, out in zip(item["inputs"], item["outputs"], strict=True)
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(tests, fh)
            tests_path = fh.name
        try:
            payload = _APPS_FUNCTIONAL_HARNESS.format(
                code=code, fn_name=item["fn_name"], tests_path=tests_path
            )
            return _run_code(payload, CODE_EXEC_TIMEOUT_S, python_exe)
        finally:
            Path(tests_path).unlink(missing_ok=True)
    stdin_tests = [
        {
            "input": "\n".join(map(str, inp)) if isinstance(inp, list) else str(inp),
            "output": "\n".join(map(str, out)) if isinstance(out, list) else str(out),
        }
        for inp, out in zip(item["inputs"], item["outputs"], strict=True)
    ]
    return _run_stdin_tests(code, stdin_tests, python_exe)


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
    elif bench == "apps_intro":
        v = _verify_apps(completion, item, python_exe)
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


def _load_finish_reasons(path: Path) -> dict[str, list[str]]:
    """Per-item finish_reasons from the rollouts JSONL (full-pool cap-hit read)."""
    out: dict[str, list[str]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    out[row["item_id"]] = row.get("finish_reasons") or []
    return out


def _check_genmeta(roll_path: Path, benchmark: str) -> None:
    """Resume-identity sidecar: the output-affecting generating params.

    A resumed run whose immutable params (model/K/temp/top_p/seed/base cap)
    drifted would silently mix regimes — raise instead (r1: resume keyed on
    item_id only). The BASE cap IS immutable (r2 long-loop-restartability: a
    constant edit must refuse resume); the --regen-cap-hit path raises the
    REALIZED per-wave max_tokens locally without touching the base constant,
    and records it per ROW.
    """
    meta_path = roll_path.with_name(f"{benchmark}_genmeta.json")
    current = {
        "model": MODEL,
        "k_rollouts": K_ROLLOUTS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "engine_seed": SEED,
        "base_max_tokens": MAX_TOKENS[benchmark],
    }
    if meta_path.exists():
        prior = json.loads(meta_path.read_text())
        immutable = dict(current)
        prior_immutable = {k: prior.get(k) for k in immutable}
        if prior_immutable != immutable:
            raise RuntimeError(
                f"{benchmark}: generating params drifted vs {meta_path} "
                f"(prior {prior_immutable} != current {immutable}) — a resume would mix "
                "regimes; use a fresh out-root"
            )
    else:
        meta_path.write_text(json.dumps(current, indent=1))


def _regen_prune(roll_path: Path, out_root: Path, benchmark: str) -> int:
    """--regen-cap-hit: drop cap-hit rows (any finish_reason == 'length') AND
    their verdict rows, so re-generation + re-verification actually happen
    (r1 g3 Concern 2: the >2% trigger had no executable path, and stale
    verdicts would silently score the OLD completions)."""
    reasons = _load_finish_reasons(roll_path)
    hit_ids = {iid for iid, fr in reasons.items() if "length" in fr}
    if not hit_ids:
        print(f"[gen] {benchmark}: --regen-cap-hit found 0 cap-hit rows", flush=True)
        return 0
    with atomic_replace(roll_path) as tmp:
        with roll_path.open(encoding="utf-8") as inp, tmp.open("w", encoding="utf-8") as outp:
            for line in inp:
                if line.strip() and json.loads(line)["item_id"] not in hit_ids:
                    outp.write(line)
    verd_path = out_root / surface_of(benchmark) / "rollouts" / f"{benchmark}_verdicts.jsonl"
    if verd_path.exists():
        with atomic_replace(verd_path) as vtmp:
            with (
                verd_path.open(encoding="utf-8") as inp,
                vtmp.open("w", encoding="utf-8") as outp,
            ):
                for line in inp:
                    if line.strip() and json.loads(line)["item_id"] not in hit_ids:
                        outp.write(line)
    print(f"[gen] {benchmark}: pruned {len(hit_ids)} cap-hit items (+ their verdicts)", flush=True)
    return len(hit_ids)


# ONE vLLM engine per process, shared across benchmarks: a second LLM() init at
# gpu_memory_utilization=0.90 OOMs/wedges while the first engine's workers hold
# the GPU (r1 g3 Concern 3). SamplingParams are per-call, so sharing is exact.
_ENGINE: dict = {}


def _get_engine():
    if "llm" not in _ENGINE:
        from vllm import LLM

        engine_kwargs: dict = {}
        if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
            engine_kwargs["enforce_eager"] = True
        if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
            engine_kwargs["enable_prefix_caching"] = False
        _ENGINE["llm"] = LLM(
            model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, seed=SEED, **engine_kwargs
        )
    return _ENGINE["llm"]


def phase_gen(
    benchmark: str,
    out_root: Path,
    *,
    smoke: bool,
    regen_cap_hit: bool = False,
    apps_pilot: bool = False,
) -> None:
    """K sampled rollouts per item; chunked vLLM generate; per-chunk checkpoint."""
    _require_gate_for(benchmark, out_root, apps_pilot=apps_pilot)
    items = LOADERS[benchmark]()
    _assert_unique_ids(items, benchmark)
    if benchmark == "lcb_v5":
        items = _apply_dedup(items, out_root)
    if benchmark == "apps_intro" and apps_pilot:
        items = items[:APPS_PILOT_N]
        print(f"[gen] APPS PILOT: {len(items)} items (fork-5 pilot slice)", flush=True)
    if smoke:
        items = items[:SMOKE_N]
        print(f"[gen] SMOKE: {len(items)} items (production K/caps/engine)", flush=True)
    roll_path = _rollouts_path(out_root, benchmark)
    roll_path.parent.mkdir(parents=True, exist_ok=True)
    _check_genmeta(roll_path, benchmark)
    max_tokens = MAX_TOKENS[benchmark]
    if regen_cap_hit:
        _regen_prune(roll_path, out_root, benchmark)
        max_tokens = 2 * MAX_TOKENS[benchmark]  # re-gen at >=2x cap (CLAUDE.md trigger)
    done = _load_done_rollouts(roll_path)
    pending = [it for it in items if it["item_id"] not in done]
    print(
        f"[gen] {benchmark}: {len(items)} items, {len(done)} resumed, {len(pending)} pending "
        f"max_tokens={max_tokens}",
        flush=True,
    )
    if pending:
        from transformers import AutoTokenizer
        from vllm import SamplingParams

        tok = AutoTokenizer.from_pretrained(MODEL)
        llm = _get_engine()
        # Per-request seed deliberately UNSET (pilot rationale: independent draws
        # beat per-request reproducibility for a rate DV).
        params = SamplingParams(
            n=K_ROLLOUTS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=max_tokens,
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
                                "max_tokens": max_tokens,
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
                f"{CAP_HIT_TRIGGER:.0%} re-generation trigger at max_tokens={max_tokens} "
                f"(re-run with --regen-cap-hit; durable fraction lands in the verify JSON)",
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
    benchmark: str,
    out_root: Path,
    *,
    smoke: bool,
    bcb_python: str | None,
    workers: int,
    apps_pilot: bool = False,
) -> None:
    """Programmatic verdicts for every stored rollout; writes the gen JSON.

    ``apps_pilot`` (fork 5): verify the SAME 200-item slice the pilot gen
    produced and write the verdict to its OWN file (``APPS_PILOT_REPORT``) —
    the gate's binding full-pool g3_apps read must be structurally unable to
    consume a pilot slice.
    """
    # Stage gate FIRST (r3 bug-class sweep: gen was gated, verify was not —
    # stale/manually-supplied rollouts could otherwise write fresh-looking
    # gate inputs before the preceding stage was authorized).
    _require_gate_for(benchmark, out_root, apps_pilot=apps_pilot)
    if benchmark in CODE_BENCHMARKS:
        _require_sandbox_net_isolation()  # fail BEFORE dispatching pool workers
    items = LOADERS[benchmark]()
    if benchmark == "lcb_v5":
        items = _apply_dedup(items, out_root)
    if benchmark == "apps_intro" and apps_pilot:
        items = items[:APPS_PILOT_N]
        print(f"[verify] APPS PILOT: {len(items)} items (fork-5 pilot slice)", flush=True)
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
    # Durable cap-hit fraction over the FULL persisted rollout set (CLAUDE.md
    # "every generation stage REPORTS its realized cap-hit fraction").
    reasons = _load_finish_reasons(_rollouts_path(out_root, benchmark))
    item_ids = {it["item_id"] for it in items}
    n_len = sum(fr.count("length") for iid, fr in reasons.items() if iid in item_ids)
    n_fr = sum(len(fr) for iid, fr in reasons.items() if iid in item_ids)
    payload = {
        "benchmark": benchmark,
        "model": MODEL,
        "k_rollouts": K_ROLLOUTS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS[benchmark],
        "cap_hit_fraction": n_len / max(1, n_fr),
        "cap_hit_counts": {"length": n_len, "total": n_fr},
        "n_items": len(out_items),
        "n_unparsed_rollouts": n_unparsed,
        "unparsed_fraction": n_unparsed / max(1, len(out_items) * K_ROLLOUTS),
        "stats_full_k": stats,
        "admissible": bool(stats and admissible(stats)),
        "smoke": smoke,
        "pilot": bool(benchmark == "apps_intro" and apps_pilot),
        # Persisted so control-vs-production network-isolation drift across
        # hosts is observable post-hoc (r2 nit; None on non-executing surfaces).
        "sandbox_net_isolation": (
            _unshare_net_available() if benchmark in CODE_BENCHMARKS else None
        ),
        "items": out_items,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    payload.update(as_metadata_dict(git_provenance(), phase=f"gen-verify-{benchmark}"))
    out_name = APPS_PILOT_REPORT if payload["pilot"] else f"{benchmark}.json"
    out_path = out_root / surface_of(benchmark) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(out_path) as tmp:
        tmp.write_text(json.dumps(payload))
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
    """Line-split a JSONL into <=9 MB shards (upload-policy non-LFS split rule).

    Clears stale ``{stem}.shard*.jsonl`` first: a re-shard of a SHRUNK source
    (e.g. after --regen-cap-hit pruning) must not leave orphan higher-index
    shards that the folder upload would ship as live data.
    """
    for stale in sorted(shard_dir.glob(f"{stem}.shard*.jsonl")):
        stale.unlink()
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


def phase_upload(benchmark: str, out_root: Path, *, smoke: bool) -> None:
    """Shard + upload the benchmark's rollout text (and verdicts) to the HF data repo.

    Smoke runs upload under a ``_smoke``-suffixed HF prefix — a smoke must
    never land shards under the production prefix (r1 g3 blocker 2: HF paths
    are production artifacts exactly like local ones).
    """
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
    prefix = f"{HF_GEN_PREFIX}{'_smoke' if smoke else ''}/{surface}"
    url = hub._upload(  # folder branch: ONE upload_folder commit for the shard set
        shard_dir, hub.DEFAULT_DATASET_REPO, repo_type="dataset", path_in_repo=prefix
    )
    if not url:
        raise RuntimeError(f"upload returned no path for {shard_dir} -> {prefix} (fail-loud)")
    print(f"[upload] {benchmark}: {n_shards} text shards -> {prefix}", flush=True)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

PHASES = ("dedup", "gate", "gen", "verify", "upload")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--phase", choices=PHASES + ("all",), required=False, default="all")
    ap.add_argument("--benchmark", choices=sorted(LOADERS))
    ap.add_argument("--surface", choices=sorted(SURFACES))
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_N} items/benchmark, *_smoke root")
    ap.add_argument("--bcb-python", default=None, help="/opt/bcb-venv/bin/python for the BCB gate")
    ap.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    ap.add_argument(
        "--control-report",
        type=Path,
        default=None,
        help="issue2388_code_control.py report for the gate phase (default: <out-root>/code/...)",
    )
    ap.add_argument(
        "--regen-cap-hit",
        action="store_true",
        help="prune cap-hit rows (+ verdicts) and re-generate them at 2x max_tokens",
    )
    ap.add_argument(
        "--apps-pilot",
        action="store_true",
        help=f"fork-5 pilot: first {APPS_PILOT_N} APPS-intro items only",
    )
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
    # Bare-default roster EXCLUDES the contingency benchmark (r2 Critical 2:
    # apps_intro sorts before bigcodebench_full, so a no-flag run would crash
    # at _require_gate_for before generating anything; fork 5 runs it
    # explicitly via --benchmark apps_intro).
    benches = (
        [args.benchmark]
        if args.benchmark
        else (SURFACES[args.surface] if args.surface else sorted(set(LOADERS) - {"apps_intro"}))
    )
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    try:
        for phase in phases:
            if phase == "dedup":
                dedup_lcb_against_leetcode(out_root)
                continue
            if phase == "gate":
                phase_gate(out_root, control_report=args.control_report)
                continue
            for bench in benches:
                if phase == "gen":
                    phase_gen(
                        bench,
                        out_root,
                        smoke=args.smoke,
                        regen_cap_hit=args.regen_cap_hit,
                        apps_pilot=args.apps_pilot,
                    )
                elif phase == "verify":
                    phase_verify(
                        bench,
                        out_root,
                        smoke=args.smoke,
                        bcb_python=args.bcb_python,
                        workers=args.workers,
                        apps_pilot=args.apps_pilot,
                    )
                elif phase == "upload":
                    phase_upload(bench, out_root, smoke=args.smoke)
    except BaseException:
        if "llm" in _ENGINE:
            # Crash path with a live engine: print the traceback ourselves and
            # os._exit(1) — finalization would deadlock on unreaped engine
            # children (gotchas.md #1739/#2149) and the rc would never emerge.
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    sys.stdout.flush()
    sys.stderr.flush()
    if "llm" in _ENGINE:
        # vLLM engine children survive interpreter finalization and deadlock a
        # sys.exit(0) terminal (gotchas.md #1739/#2149) — durables are already
        # flushed to disk above.
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
