# ruff: noqa: E402
"""P0 corpus staging for task #2546 (CoT context->answer map) — plan v4 section 4.2 P0.

Builds the 8-corpus staging bundle consumed by every arm's pod:

1. Pull the #1336 staged GSM8K/MATH corpora from HF
   ``issue1336_rlvr_ladder/corpora_v2/`` (sha-verified against their committed
   manifests/meta sidecars); join GSM8K gold solutions from ``openai/gsm8k`` by
   ``src_index``; k = count of ``<<`` per gold solution, binned 1 / 2-3 / 4-6 / >=7
   with the pre-registered thin-bin fallback (k=1 < 300 rows => lowest bin becomes
   k<=2, recorded in the manifest).
2. Pull MMLU / ARC-Challenge / CSQA / PIQA / ContextHub (via TAUR-Lab configs) at the
   per-arm draws of plan section 4.1 (nested arm-1/2 subset of the arm-3 draw).
3. Join the TAUR-Lab CoT Analysis collection (the UN-GATED per-model repos —
   ``gated='manual'`` repos are excluded at LISTING time and RECORDED, never silently
   dropped; user ruling 2026-08-25, epm:progress v26) into a per-question
   ``rescue_rate`` (fraction of joined models with cot_correct AND NOT direct_correct;
   join by normalized question text; unjoined rows keep rescue_rate = null, never
   imputed; coverage reported per corpus; the realized model list + per-config model
   counts are recorded in the manifest so the denominator change is never silent).
4. Near-dup dedup WITHIN + ACROSS corpora BEFORE any draw/split: the reused #1739
   MinHash signatures (char-5 shingles, 64 perms) generate CANDIDATE pairs via LSH
   (16 bands x 4 rows => candidates from Jaccard ~>=0.5), each candidate then
   VERIFIED by exact char-5-shingle Jaccard >= 0.8 before dropping — the plan's
   registered "5-gram Jaccard >= 0.8" operating point is enforced exactly, never
   approximated by a probabilistic band collision alone.
5. CPU model asserts (AutoConfig x5; arm-1 full-render token-id identity; arm-2
   content-token identity + verbatim render recording; arm-3 cross-mode render probe;
   per-arm think-delimiter encodings asserted against the plan's pins).
6. Write per-corpus jsonl (line-sharded under 9 MB) + ``corpora_manifest.json`` and
   upload the bundle to ``issue2546_cotmap/corpora_v1/`` on the HF data repo
   (one bulk ``upload_folder`` commit + scoped exact-set verify).

Observed schema (probed live 2026-08-24, schema-from-artifact duty):
``gsm8k_test1319.jsonl`` / ``gsm8k_train_full.jsonl`` / ``math7500.shard0[01].jsonl``
rows carry exactly ``{prompt, prompt_idx, src_index}``. The gsm8k ``prompt`` is the
BARE question text; the math ``prompt`` is a 4-shot ``Question:/Answer:`` render whose
FINAL ``Question:`` block is the row's own problem (extracted here under a fail-loud
shared-prefix assert — the plan registers zero-shot renders, so the few-shot scaffold
must not leak into #2546 renders).

Pure CPU staging: no model forwards, no API generation calls. Smoke mode
(``--smoke``) caps row pulls (20/corpus) and TAUR repos (2) and skips the upload;
every other code path (dedup, draws, asserts, manifest, shard writer) is identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE any huggingface import (lint --check-dotenv-before-hf-import)

import numpy as np
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from explore_persona_space.atomic_io import atomic_replace
from explore_persona_space.experiments.issue_1739.corpus_staging import minhash_signatures
from explore_persona_space.orchestrate import hub

# ---------------------------------------------------------------------------
# Registered constants (plan v4 sections 4.1 / 10)
# ---------------------------------------------------------------------------

DATA_REPO = "superkaiba1/explore-persona-space-data"
SRC_PREFIX = "issue1336_rlvr_ladder/corpora_v2"
HF_DEST_DEFAULT = "issue2546_cotmap/corpora_v1"

ARM_MODELS: dict[int, tuple[str, ...]] = {
    1: ("Qwen/Qwen2.5-7B-Instruct", "open-thoughts/OpenThinker3-7B"),
    2: ("Qwen/Qwen2.5-Math-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    3: ("Qwen/Qwen3-8B",),  # ORIGINAL hybrid release — the -2507 exclusion is user-BINDING
}

# Think-delimiter token-id pins (plan section 4.2 P0 item 3; fact-checked 2026-08-24).
THINK_DELIM_PINS: dict[int, dict[str, list[int]]] = {
    1: {"<think>": [13708, 766, 29], "</think>": [522, 26865, 29]},  # MULTI-token (Qwen2.5 BPE)
    2: {"<think>": [151648], "</think>": [151649]},  # single tokens (R1-distill)
    3: {"<think>": [151667], "</think>": [151668]},  # single tokens (Qwen3)
}

# Registered verbatim renders (plan section 4.1; zero-shot everywhere, no few-shot/ICL).
RENDER_MATH = "{question}\n\nPut your final answer within \\boxed{{}}."
RENDER_MCQ = "{question}\n\n{options}\n\nAnswer with the letter of the correct option."
# ContextHub: the dataset's item text + its native answer instruction (the TAUR item text
# carries the instruction; recorded verbatim in the manifest, frozen at P0).

# Per-corpus draws (plan section 4.1 table). None => all rows, both arms.
MMLU_DRAW_ARM12 = 6750
MMLU_DRAW_ARM3 = 7680
CH_L24_CAP_ARM12 = 1000
CH_L24_CAP_ARM3 = 1300
K1_FALLBACK_FLOOR = 300  # k=1 bin below this => merge lowest bin to k<=2 (pre-registered)

# Dedup keep-first corpus order: gsm8k_test first (eval-only — protected), then train
# (near-dup-of-test train rows drop: the decontamination direction), then the rest.
CORPUS_ORDER = [
    "gsm8k_test",
    "gsm8k_train",
    "math",
    "contexthub",
    "mmlu",
    "arc_challenge",
    "csqa",
    "piqa",
]

# LSH banding: 64 perms / 16 bands x 4 rows => CANDIDATE net ~ Jaccard >= (1/16)^(1/4) ~ 0.5;
# every candidate is then verified by EXACT char-5-shingle Jaccard against the plan's
# registered >= 0.8 operating point (a band collision alone is probabilistic and would
# both mis-drop below 0.8 and mis-keep above it).
LSH_BANDS = 16
JACCARD_VERIFY_THRESHOLD = 0.8

TAUR_REPO_MARKER = "Taur_CoT_Analysis_Project"
# The plan (section 4.2) named 16 per-model repos. The live 2026-08-25 listing carries
# 22 per-model repos (incl. the double-underscore Llama-id shape) plus the gated parent
# and 2 experiment repos; 8 of the 22 are gated ('manual') and are excluded at LISTING
# time (the token has no gated-read scope, so a load would 403 — user ruling 2026-08-25,
# epm:progress v26: take the un-gated route now). Floor re-derived against the realized
# UN-GATED set: 14 measured 2026-08-25 (FEWER than the plan's 16 — the discrepancy +
# the excluded gated ids are recorded in the manifest, never silently absorbed); a
# count below 14 means previously-available data vanished => fail loud.
TAUR_EXPECTED_REPOS_FLOOR = 14
TAUR_PARENT_REPO = "TAUR-Lab/Taur_CoT_Analysis_Project"  # gated collection parent (excluded)
TAUR_EXCLUDE_SUFFIXES = ("__Paraphrase_Exp", "__Symbolic_Solver_Experiment")
# Registered config-name patterns per corpus family (realized matches recorded in manifest).
TAUR_PATTERNS: dict[str, str] = {
    "gsm8k_test": r"^gsm8k$",
    "math": r"^math$",  # anchored: \w* also matched 'math__round_2_fixes' (double-count)
    # contexthub anchored to the 8 canonical cells: bare r"contexthub" also matched
    # 'contexthub_deductive_level2__round_2_fixes', which parses to the SAME
    # (type, level) cell as its base config => 'sourced twice' raise when the CH
    # source repo is gemma-2-9b-it / Phi-3-small-8k AND a SECOND n_models increment
    # for the SAME model on the same questions (silent rescue_rate corruption).
    "mmlu": r"^mmlu$",
    "arc_challenge": r"^arc_challenge$",
    "csqa": r"^(csqa|commonsense_?qa)$",
    "piqa": r"^piqa$",
    "contexthub": r"^contexthub_(deductive|abductive)_level[1-4]$",
}
# The 8 canonical ContextHub cells (deductive|abductive x level 1-4; plan section 4.1).
CH_CANONICAL_CELLS: frozenset[tuple[str, int]] = frozenset(
    (t, level) for t in ("deductive", "abductive") for level in (1, 2, 3, 4)
)
_QUESTION_KEYS = ("question", "input", "prompt", "query")
_GOLD_KEYS = ("gold_answer", "answer", "gold", "target")

SHARD_BYTES_MAX = 9_000_000  # <9 MB shards; >9.5 MB single files force-route to LFS


def _norm(text: str) -> str:
    """Whitespace-collapsed join key (case-preserved; recorded in manifest)."""
    return " ".join(str(text).split())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        return out.stdout.strip() if out.returncode == 0 else "unavailable-no-git-checkout"
    except OSError:
        return "unavailable-no-git-checkout"


def _log(msg: str) -> None:
    print(f"[p0] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1 — banked #1336 corpora (schema-from-artifact: {prompt, prompt_idx, src_index})
# ---------------------------------------------------------------------------


def _dl(repo_path: str, dest: Path) -> Path:
    local = hub.retry_transient(
        lambda: hf_hub_download(DATA_REPO, repo_path, repo_type="dataset", local_dir=dest),
        what=f"hf_hub_download {repo_path}",
    )
    return Path(local)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):  # never .splitlines() on JSONL (gotchas)
        if line.strip():
            rows.append(json.loads(line))
    return rows


def stage_1336_corpora(dl_dir: Path) -> dict[str, list[dict]]:
    """Download + sha-verify the #1336 staged gsm8k/math corpora; return raw rows."""
    files = {
        "gsm8k_test": ["gsm8k_test1319.jsonl", "gsm8k_test1319_meta.json"],
        "gsm8k_train": ["gsm8k_train_full.jsonl", "gsm8k_train_full_meta.json"],
        "math": [
            # math7500_meta.json deliberately NOT pulled: the manifest carries the
            # per-shard sha256s + line counts this stage verifies against (a meta
            # download nothing reads is dead weight).
            "math7500.manifest.json",
            "math7500.shard00.jsonl",
            "math7500.shard01.jsonl",
        ],
    }
    local: dict[str, list[Path]] = {}
    for corpus, names in files.items():
        local[corpus] = [_dl(f"{SRC_PREFIX}/{n}", dl_dir) for n in names]
    root = dl_dir / SRC_PREFIX

    out: dict[str, list[dict]] = {}
    # gsm8k: single files, meta.sha256 covers the jsonl bytes.
    for corpus, jsonl_name, meta_name, n_expect in [
        ("gsm8k_test", "gsm8k_test1319.jsonl", "gsm8k_test1319_meta.json", 1319),
        ("gsm8k_train", "gsm8k_train_full.jsonl", "gsm8k_train_full_meta.json", 7473),
    ]:
        meta = json.loads((root / meta_name).read_text())
        got = _sha256_file(root / jsonl_name)
        if got != meta["sha256"]:
            raise RuntimeError(f"{corpus}: sha mismatch {got} != meta {meta['sha256']}")
        rows = _read_jsonl(root / jsonl_name)
        if len(rows) != n_expect or meta["n_built"] != n_expect:
            raise RuntimeError(f"{corpus}: count {len(rows)} != expected {n_expect}")
        out[corpus] = rows
    # math: sharded; manifest carries per-shard sha256s + line counts.
    man = json.loads((root / "math7500.manifest.json").read_text())
    rows = []
    for part, sha, n_lines in zip(man["parts"], man["sha256s"], man["line_counts"], strict=True):
        got = _sha256_file(root / part)
        if got != sha:
            raise RuntimeError(f"math shard {part}: sha mismatch {got} != {sha}")
        part_rows = _read_jsonl(root / part)
        if len(part_rows) != n_lines:
            raise RuntimeError(f"math shard {part}: {len(part_rows)} rows != {n_lines}")
        rows.extend(part_rows)
    if len(rows) != 7500:
        raise RuntimeError(f"math7500: {len(rows)} rows != 7500")
    out["math"] = rows
    for corpus, rws in out.items():
        keys = set(rws[0].keys())
        if keys != {"prompt", "prompt_idx", "src_index"}:
            raise RuntimeError(f"{corpus}: unexpected staged keys {sorted(keys)}")
    _log(f"staged #1336 corpora verified: {[(c, len(r)) for c, r in out.items()]}")
    return out


def extract_math_problems(rows: list[dict]) -> tuple[list[str], dict]:
    """Strip the fixed 4-shot 'Question:/Answer:' scaffold; return bare problems.

    The staged math7500 prompt = FIXED_PREFIX + problem, where FIXED_PREFIX ends with
    the final ``\\n\\nQuestion: `` delimiter and is IDENTICAL across rows (asserted).
    """
    delim = "\n\nQuestion: "
    p0 = rows[0]["prompt"]
    cut = p0.rfind(delim)
    if cut < 0:
        raise RuntimeError("math7500 row0 carries no 'Question:' delimiter — schema drift")
    prefix = p0[: cut + len(delim)]
    problems = []
    for i, r in enumerate(rows):
        if not r["prompt"].startswith(prefix):
            raise RuntimeError(f"math7500 row {i}: few-shot scaffold differs from row0 — abort")
        problems.append(r["prompt"][len(prefix) :])
    report = {
        "scaffold_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "scaffold_chars": len(prefix),
        "scaffold_n_shots": prefix.count(delim.strip("\n").strip()) - 1,
        "extraction_rule": "text after the final '\\n\\nQuestion: ' of the staged prompt",
    }
    return problems, report


def join_gsm8k_gold(staged: dict[str, list[dict]], smoke: bool) -> dict[str, list[dict]]:
    """Join openai/gsm8k gold solutions by src_index; compute k = count('<<')."""
    ds = {
        "test": load_dataset("openai/gsm8k", "main", split="test"),
        "train": load_dataset("openai/gsm8k", "main", split="train"),
    }
    out: dict[str, list[dict]] = {}
    for corpus, split in [("gsm8k_test", "test"), ("gsm8k_train", "train")]:
        src = ds[split]
        rows = []
        staged_rows = staged[corpus][:20] if smoke else staged[corpus]
        for r in staged_rows:
            idx = int(r["src_index"])
            if idx < 0 or idx >= len(src):
                raise RuntimeError(f"{corpus}: src_index {idx} out of range for {split} split")
            srow = src[idx]
            if _norm(srow["question"]) != _norm(r["prompt"]):
                raise RuntimeError(f"{corpus}: staged prompt != source question at src_index {idx}")
            sol = srow["answer"]
            k = sol.count("<<")
            if k < 1:
                raise RuntimeError(f"{corpus}: gold solution at src_index {idx} has zero '<<'")
            gold = sol.rsplit("####", 1)[-1].strip() if "####" in sol else None
            if gold is None:
                raise RuntimeError(f"{corpus}: gold solution at src_index {idx} lacks '####'")
            rows.append(
                {
                    "question": srow["question"],
                    "gold_answer": gold,
                    "k": k,
                    "src_index": idx,
                    "src_split": split,
                }
            )
        out[corpus] = rows
        _log(f"{corpus}: joined gold for {len(rows)} rows (k>=1 asserted)")
    return out


def assign_k_bins(rows_by_corpus: dict[str, list[dict]]) -> dict:
    """k-bins 1 / 2-3 / 4-6 / >=7 with the pre-registered thin-bin fallback."""
    ks = [r["k"] for c in ("gsm8k_test", "gsm8k_train") for r in rows_by_corpus[c]]
    n_k1 = sum(1 for k in ks if k == 1)
    fallback = n_k1 < K1_FALLBACK_FLOOR
    if fallback:
        bins = [(1, 2, "k_le2"), (3, 3, "k3"), (4, 6, "k4_6"), (7, 10**9, "k7p")]
    else:
        bins = [(1, 1, "k1"), (2, 3, "k2_3"), (4, 6, "k4_6"), (7, 10**9, "k7p")]

    def bin_of(k: int) -> str:
        for lo, hi, name in bins:
            if lo <= k <= hi:
                return name
        raise RuntimeError(f"k={k} fell through the bin table")

    for c in ("gsm8k_test", "gsm8k_train"):
        for r in rows_by_corpus[c]:
            r["k_bin"] = bin_of(r["k"])
    dist = Counter(ks)
    return {
        "k_distribution": {str(k): int(v) for k, v in sorted(dist.items())},
        "n_k1": n_k1,
        "k1_fallback_floor": K1_FALLBACK_FLOOR,
        "k1_fallback_applied": fallback,
        "bins": [
            {"lo": lo, "hi": (None if hi >= 10**9 else hi), "name": nm} for lo, hi, nm in bins
        ],
        "bin_counts": dict(
            Counter(r["k_bin"] for c in ("gsm8k_test", "gsm8k_train") for r in rows_by_corpus[c])
        ),
    }


# ---------------------------------------------------------------------------
# Step 2 — benchmark pulls (MMLU / ARC / CSQA / PIQA)
# ---------------------------------------------------------------------------


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def load_mcq_corpora(smoke: bool) -> dict[str, list[dict]]:
    cap = 20 if smoke else None
    out: dict[str, list[dict]] = {}

    ds = load_dataset("cais/mmlu", "all", split="test")
    if not smoke and len(ds) != 14042:
        raise RuntimeError(f"cais/mmlu test: {len(ds)} rows != 14042 (plan-measured)")
    rows = []
    for i, r in enumerate(ds):
        if cap is not None and i >= cap:
            break
        opts = list(r["choices"])
        rows.append(
            {
                "question": r["question"],
                "options": opts,
                "gold_answer": _letters(len(opts))[int(r["answer"])],
                "subject": r["subject"],
                "src_index": i,
            }
        )
    out["mmlu"] = rows

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if not smoke and len(ds) != 1172:
        raise RuntimeError(f"ai2_arc ARC-Challenge test: {len(ds)} rows != 1172")
    rows = []
    for i, r in enumerate(ds):
        if cap is not None and i >= cap:
            break
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        letters = _letters(len(texts))
        key = r["answerKey"]
        if key not in labels:
            raise RuntimeError(f"arc row {i}: answerKey {key!r} not in labels {labels}")
        rows.append(
            {
                "question": r["question"],
                "options": texts,
                "gold_answer": letters[labels.index(key)],
                "src_index": i,
            }
        )
    out["arc_challenge"] = rows

    ds = load_dataset("tau/commonsense_qa", split="validation")
    if not smoke and len(ds) != 1221:
        raise RuntimeError(f"commonsense_qa validation: {len(ds)} rows != 1221")
    rows = []
    for i, r in enumerate(ds):
        if cap is not None and i >= cap:
            break
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        letters = _letters(len(texts))
        key = r["answerKey"]
        if key not in labels:
            raise RuntimeError(f"csqa row {i}: answerKey {key!r} not in labels {labels}")
        rows.append(
            {
                "question": r["question"],
                "options": texts,
                "gold_answer": letters[labels.index(key)],
                "src_index": i,
            }
        )
    out["csqa"] = rows

    ds = load_dataset("baber/piqa", split="validation")  # parquet mirror (ybisk/piqa is
    # a loading-script dataset refused by datasets 4.8.4)
    if not smoke and len(ds) != 1838:
        raise RuntimeError(f"baber/piqa validation: {len(ds)} rows != 1838")
    rows = []
    for i, r in enumerate(ds):
        if cap is not None and i >= cap:
            break
        rows.append(
            {
                "question": r["goal"],
                "options": [r["sol1"], r["sol2"]],
                "gold_answer": _letters(2)[int(r["label"])],
                "src_index": i,
            }
        )
    out["piqa"] = rows
    for c, r in out.items():
        _log(f"{c}: pulled {len(r)} rows")
    return out


# ---------------------------------------------------------------------------
# Step 3 — TAUR-Lab: ContextHub rows + rescue_rate(q)
# ---------------------------------------------------------------------------


def _filter_taur_repo_ids(
    all_matching: list[str], gated_ids: set[str] | frozenset[str]
) -> tuple[list[str], list[str], list[str]]:
    """Pure include/exclude filter over marker-matching repo ids (testable seam).

    Include: every UN-GATED repo under the ``Taur_CoT_Analysis_Project__`` prefix
    (covers the triple-underscore per-model ids AND the double-underscore id shape).
    Exclude — each class RECORDED, never silently dropped:
      excluded_gated: per-model repos whose listing entry is gated (``'manual'``
        observed live 2026-08-25; the token has no gated-read scope, so a load
        would 403 — user ruling epm:progress v26 takes the un-gated route now);
      excluded_other: the collection parent (exact id, no per-model suffix; excluded
        by the prefix rule BEFORE any gating split) and the two experiment repos
        (paraphrase / symbolic-solver — different question sets).

    Returns (repos, excluded_gated, excluded_other).
    """
    per_model = [
        rid
        for rid in all_matching
        if rid.startswith(TAUR_PARENT_REPO + "__") and not rid.endswith(TAUR_EXCLUDE_SUFFIXES)
    ]
    repos = [rid for rid in per_model if rid not in gated_ids]
    excluded_gated = sorted(set(per_model) & set(gated_ids))
    excluded_other = sorted(set(all_matching) - set(per_model))
    return repos, excluded_gated, excluded_other


def discover_taur_repos(smoke: bool) -> tuple[list[str], dict]:
    """Enumerate the UN-GATED TAUR-Lab per-model CoT-analysis repos.

    Filter semantics: _filter_taur_repo_ids. Gated repos are excluded at LISTING
    time — ``HfApi.list_datasets`` exposes ``.gated`` (measured 2026-08-25: values
    ``False`` / ``'manual'`` across all 25 marker-matching repos), so no per-repo
    403 probe is needed; a ``gated=None`` listing entry means the listing stopped
    exposing gating and the filter is blind => fail loud. Floor >=
    TAUR_EXPECTED_REPOS_FLOOR (re-derived: 14 un-gated measured 2026-08-25; the
    plan named 16 — the discrepancy is recorded, never silently absorbed). Smoke
    slices the FILTERED set, so the 2 smoke repos are always un-gated per-model
    repos.
    """
    api = HfApi()
    marker_matching = [
        d
        for d in hub.retry_transient(
            lambda: list(api.list_datasets(author="TAUR-Lab", limit=500)),
            what="list_datasets TAUR-Lab",
        )
        if TAUR_REPO_MARKER in d.id
    ]
    gated_ids: set[str] = set()
    for d in marker_matching:
        if d.gated is None:
            raise RuntimeError(
                f"TAUR-Lab listing: {d.id} has gated=None — the listing-time gated "
                "filter cannot run (HfApi.list_datasets stopped exposing .gated?)"
            )
        if d.gated:  # 'manual' observed live; 'auto'/True would be equally unreadable
            gated_ids.add(d.id)
    all_matching = sorted(d.id for d in marker_matching)
    repos, excluded_gated, excluded_other = _filter_taur_repo_ids(all_matching, gated_ids)
    report = {
        "n_marker_matching": len(all_matching),
        "n_per_model_ungated_included": len(repos),
        "n_per_model_gated_excluded": len(excluded_gated),
        "included_ungated": list(repos),  # realized model list (pre smoke-slice)
        "excluded_gated": excluded_gated,
        "excluded_other": excluded_other,
        "expected_floor": TAUR_EXPECTED_REPOS_FLOOR,
        "plan_named_count": 16,
        "plan_count_discrepancy_note": (
            "plan section 4.2 named 16 per-model repos; the 2026-08-25 listing carries "
            f"{len(repos) + len(excluded_gated)} per-model repos of which "
            f"{len(excluded_gated)} are gated ('manual'; no gated-read credential — "
            f"user ruling 2026-08-25 takes the un-gated route) and {len(repos)} are "
            "staged. FEWER models than the plan named: rescue_rate denominators are "
            "per-question n_models over the un-gated set only, so estimates are NOISIER "
            "than planned — recorded here plus per-config model counts in taur_report, "
            "never silent"
        ),
    }
    if smoke:
        repos = repos[:2]
    elif len(repos) < TAUR_EXPECTED_REPOS_FLOOR:
        raise RuntimeError(
            f"TAUR-Lab discovery: {len(repos)} un-gated per-model repos < floor "
            f"{TAUR_EXPECTED_REPOS_FLOOR}: {repos} (excluded_gated: {excluded_gated}; "
            f"excluded_other: {excluded_other})"
        )
    _log(
        f"TAUR-Lab un-gated per-model repos ({len(repos)}): {repos}; "
        f"excluded_gated ({len(excluded_gated)}): {excluded_gated}; "
        f"excluded_other: {excluded_other}"
    )
    return repos, report


def _pick_split(repo: str, config: str) -> str:
    from datasets import get_dataset_split_names

    splits = hub.retry_transient(
        lambda: get_dataset_split_names(repo, config), what=f"split_names {repo}/{config}"
    )
    if "latest" in splits:
        return "latest"
    if len(splits) == 1:
        return splits[0]
    raise RuntimeError(f"{repo}/{config}: ambiguous splits {splits} (no 'latest')")


def _resolve_key(row: dict, candidates: tuple[str, ...], what: str, where: str) -> str:
    for k in candidates:
        if k in row:
            return k
    raise RuntimeError(f"{where}: no {what} key among {candidates}; row keys={sorted(row)}")


def _bool_field(row: dict, key: str, where: str) -> bool:
    v = row.get(key)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    raise RuntimeError(f"{where}: unparseable {key}={v!r}")


def _select_ch_source_repo(ch_cells_by_repo: dict[str, set[tuple[str, int]]]) -> str:
    """First repo (input order) whose contexthub configs cover ALL 8 canonical cells.

    ContextHub item text must come from ONE repo (question sets verified aligned
    across repos — task body 2026-08-24), and that repo is REQUIRED to carry all 8
    canonical (type x level) cells: the superseded first-CH-carrying pin worked only
    because sorted order happened to put an 8-cell repo first — a 4-config repo would
    have silently staged a partial ContextHub corpus. Raises when no loaded repo
    covers all 8 (per-repo coverage named in the error).
    """
    for repo, cells in ch_cells_by_repo.items():
        if CH_CANONICAL_CELLS <= cells:
            return repo
    coverage = {repo: sorted(f"{t}_L{lv}" for t, lv in c) for repo, c in ch_cells_by_repo.items()}
    raise RuntimeError(
        f"no loaded TAUR repo carries all {len(CH_CANONICAL_CELLS)} canonical contexthub "
        f"cells; per-repo cell coverage: {coverage}"
    )


def load_taur(
    repos: list[str],
) -> tuple[dict[str, dict[str, dict]], dict[tuple[str, int], list[dict]], dict]:
    """Load TAUR configs for the 8 corpora.

    Returns (rescue_acc, contexthub_cells, report):
      rescue_acc[family][norm_q] = {"n_models": int, "n_rescue": int}
      contexthub_cells[(ch_type, level)] = rows from the CH source repo — the FIRST
        loaded repo carrying ALL 8 canonical cells (_select_ch_source_repo; question
        sets verified aligned across repos — task body 2026-08-24). Exactly ONE
        config per (repo, ch_type, level) may contribute (checked per repo), and the
        end-of-load assert requires exactly the 8 canonical cells filled.
    """
    rescue_acc: dict[str, dict[str, dict]] = defaultdict(dict)
    ch_cells: dict[tuple[str, int], list[dict]] = {}
    report: dict = {"per_repo": {}, "configs_matched": defaultdict(set)}

    # -- pass 1: config-name discovery + per-repo pattern checks (no row loads) ----
    repo_matched: dict[str, dict[str, list[str]]] = {}
    ch_cells_by_repo: dict[str, set[tuple[str, int]]] = {}
    ch_variants_excluded: set[str] = set()
    for repo in repos:
        configs = hub.retry_transient(
            lambda repo=repo: get_dataset_config_names(repo), what=f"config_names {repo}"
        )
        matched: dict[str, list[str]] = defaultdict(list)
        for cfg in configs:
            for family, pat in TAUR_PATTERNS.items():
                if re.search(pat, cfg, flags=re.IGNORECASE):
                    matched[family].append(cfg)
                    report["configs_matched"][family].add(cfg)
            if "contexthub" in cfg.lower() and cfg not in matched.get("contexthub", ()):
                # e.g. 'contexthub_deductive_level2__round_2_fixes' — superseded
                # re-run variants excluded by the anchored pattern; RECORDED so the
                # exclusion is auditable in the manifest, never silent.
                ch_variants_excluded.add(cfg)
        for family, cfgs in matched.items():
            # One config per (repo, family) — a second match would double-count the
            # repo's model in that family's rescue_rate denominators (contexthub is
            # legitimately multi-config: 8 (type x level) cells per repo).
            if family != "contexthub" and len(cfgs) > 1:
                raise RuntimeError(
                    f"{repo}: family {family!r} matched {len(cfgs)} configs {sorted(cfgs)} — "
                    "per-(repo,family) single-config expected (pattern too loose?)"
                )
        # Exactly ONE config per (repo, ch_type, level): a second config parsing to
        # the same cell would add a SECOND n_models increment for the SAME model on
        # the same questions — silent rescue_rate denominator corruption.
        cell_cfgs: dict[tuple[str, int], str] = {}
        for cfg in matched.get("contexthub", []):
            key = _parse_contexthub_config(cfg)
            if key in cell_cfgs:
                raise RuntimeError(
                    f"{repo}: contexthub cell {key} matched twice "
                    f"({cell_cfgs[key]!r}, {cfg!r}) — pattern too loose?"
                )
            cell_cfgs[key] = cfg
        if cell_cfgs:
            ch_cells_by_repo[repo] = set(cell_cfgs)
        repo_matched[repo] = {f: sorted(c) for f, c in matched.items()}

    ch_source_repo = _select_ch_source_repo(ch_cells_by_repo)
    ch_coverage = {r: len(c) for r, c in ch_cells_by_repo.items()}
    _log(
        f"contexthub source repo: {ch_source_repo} (all {len(CH_CANONICAL_CELLS)} canonical "
        f"cells; per-repo canonical-cell coverage: {ch_coverage})"
    )

    # -- pass 2: row loads (rescue accumulation; CH rows from ch_source_repo only) --
    for repo in repos:
        repo_stats: dict[str, int] = {}
        for family, cfgs in sorted(repo_matched[repo].items()):
            for cfg in sorted(cfgs):
                split = _pick_split(repo, cfg)
                ds = hub.retry_transient(
                    lambda repo=repo, cfg=cfg, split=split: load_dataset(repo, cfg, split=split),
                    what=f"load_dataset {repo}/{cfg}",
                )
                if len(ds) == 0:
                    raise RuntimeError(f"{repo}/{cfg}: empty split {split}")
                row0 = ds[0]
                qk = _resolve_key(row0, _QUESTION_KEYS, "question", f"{repo}/{cfg}")
                seen: set[str] = set()
                n_doubled = 0
                for r in ds:
                    nq = _norm(r[qk])
                    if not nq:
                        continue
                    if nq in seen:  # DOUBLED-config dedup (gsm8k_hard 2638=2x1319 shape)
                        n_doubled += 1
                        continue
                    seen.add(nq)
                    acc = rescue_acc[family].setdefault(nq, {"n_models": 0, "n_rescue": 0})
                    cot = _bool_field(r, "zero_shot_cot_is_correct", f"{repo}/{cfg}")
                    direct = _bool_field(r, "zero_shot_direct_is_correct", f"{repo}/{cfg}")
                    acc["n_models"] += 1
                    acc["n_rescue"] += int(cot and not direct)
                repo_stats[cfg] = len(seen)
                if family == "contexthub" and repo == ch_source_repo:
                    ch_type, level = _parse_contexthub_config(cfg)
                    rows = []
                    seen_q: set[str] = set()
                    for i, r in enumerate(ds):
                        nq = _norm(r[qk])
                        if not nq or nq in seen_q:
                            continue
                        seen_q.add(nq)
                        gk = None
                        for cand in _GOLD_KEYS:
                            if cand in r:
                                gk = cand
                                break
                        rows.append(
                            {
                                "question": r[qk],
                                "gold_answer": (str(r[gk]) if gk else None),
                                "ch_type": ch_type,
                                "level": level,
                                "src_index": i,
                                "src_config": cfg,
                            }
                        )
                    key = (ch_type, level)
                    if key in ch_cells:
                        raise RuntimeError(f"contexthub cell {key} sourced twice ({cfg})")
                    ch_cells[key] = rows
                if n_doubled:
                    _log(f"{repo}/{cfg}: deduped {n_doubled} doubled rows")
        report["per_repo"][repo] = repo_stats
        _log(f"TAUR {repo}: {sum(repo_stats.values())} unique rows across {len(repo_stats)} cfgs")

    # End-of-load assert: exactly the 8 canonical cells filled (a partial-coverage
    # source repo silently yielding fewer cells is the failure this prevents; extras
    # would mean the anchored pattern regressed).
    if set(ch_cells) != set(CH_CANONICAL_CELLS):
        raise RuntimeError(
            "contexthub cells filled != canonical 8: "
            f"missing={sorted(set(CH_CANONICAL_CELLS) - set(ch_cells))} "
            f"extra={sorted(set(ch_cells) - set(CH_CANONICAL_CELLS))} "
            f"(source repo {ch_source_repo})"
        )
    report["configs_matched"] = {k: sorted(v) for k, v in report["configs_matched"].items()}
    report["contexthub_source_repo"] = ch_source_repo
    # Per-config model counts (user ruling 2026-08-25): the rescue_rate denominator
    # change from the gated exclusion is REPORTED, never silent.
    report["contexthub_variant_configs_excluded"] = sorted(ch_variants_excluded)
    report["contexthub_cells_per_repo"] = {r: len(c) for r, c in sorted(ch_cells_by_repo.items())}
    report["contexthub_models_per_cell"] = {
        f"{t}_L{level}": sum(1 for cells in ch_cells_by_repo.values() if (t, level) in cells)
        for (t, level) in sorted(CH_CANONICAL_CELLS)
    }
    report["n_models_per_family"] = {
        fam: sum(1 for m in repo_matched.values() if m.get(fam)) for fam in TAUR_PATTERNS
    }
    return rescue_acc, ch_cells, report


_CH_LEVEL_RE = re.compile(r"level[_\s]?(\d)|_l(?:evel)?(\d)|(?<![0-9])([1-4])$", re.IGNORECASE)


def _parse_contexthub_config(cfg: str) -> tuple[str, int]:
    low = cfg.lower()
    if "deduct" in low:
        ch_type = "deductive"
    elif "abduct" in low:
        ch_type = "abductive"
    else:
        raise RuntimeError(f"contexthub config {cfg!r}: cannot parse deductive/abductive")
    m = _CH_LEVEL_RE.search(low)
    if not m:
        raise RuntimeError(f"contexthub config {cfg!r}: cannot parse level")
    level = int(next(g for g in m.groups() if g))
    if level not in (1, 2, 3, 4):
        raise RuntimeError(f"contexthub config {cfg!r}: level {level} out of range")
    return ch_type, level


# ---------------------------------------------------------------------------
# Step 4 — near-dup dedup (within + across corpora, BEFORE any draw/split)
# ---------------------------------------------------------------------------


def _shingle_hashes(text: str, n: int = 5) -> np.ndarray:
    """Sorted unique uint64 hashes of the char-n shingles (memory-bounded exact-Jaccard rep)."""
    grams = [text] if len(text) < n else [text[i : i + n] for i in range(len(text) - n + 1)]
    h = np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
            for g in grams
        ),
        dtype=np.uint64,
        count=len(grams),
    )
    return np.unique(h)


def _jaccard_hashes(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard over unique-shingle hash sets (both inputs sorted unique)."""
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / float(a.size + b.size - inter)


class _DedupState:
    """Keep-first dedup core over (key, bands, shingles) items — testable seam.

    Every band maps to the LIST of ALL retained owners sharing it — never a
    single first owner (r2 Major 6: `band_owner.setdefault(b, key)` made a
    kept-but-below-threshold row invisible as a candidate owner, so a later
    transitive near-dup of IT — A/B < 0.8, B/C >= 0.8, A/C < 0.8 — survived).
    The candidate set for a row is the exact UNION of retained owners over the
    row's bands; each candidate is verified by exact shingle Jaccard against
    JACCARD_VERIFY_THRESHOLD before any drop.
    """

    def __init__(self) -> None:
        self.band_owner: dict[tuple[int, bytes], list[tuple[str, int]]] = {}
        self.owner_shingles: dict[tuple[str, int], np.ndarray] = {}

    def admit(
        self,
        key: tuple[str, int],
        row_bands: list[tuple[int, bytes]],
        shingles: np.ndarray,
    ) -> tuple[tuple[str, int] | None, bool]:
        """Return (drop_owner|None, had_rejected_candidates); register kept keys."""
        owners = {ok for b in row_bands for ok in self.band_owner.get(b, ())}
        drop_owner: tuple[str, int] | None = None
        for ok in sorted(owners):
            if _jaccard_hashes(shingles, self.owner_shingles[ok]) >= JACCARD_VERIFY_THRESHOLD:
                drop_owner = ok
                break
        if drop_owner is None:
            self.owner_shingles[key] = shingles
            for b in row_bands:
                self.band_owner.setdefault(b, []).append(key)
        return drop_owner, bool(owners) and drop_owner is None


def lsh_keep_first(
    texts_by_corpus: dict[str, list[str]], bands: int = LSH_BANDS
) -> tuple[dict[str, np.ndarray], dict]:
    """Keep-first near-dup dedup over CORPUS_ORDER (candidate LSH + exact verify).

    The reused #1739 MinHash signatures (char-5 shingles, 64 perms) generate
    CANDIDATE owners via 16-band LSH (candidates from Jaccard ~>=0.5); each
    candidate pair is then VERIFIED by exact char-5-shingle Jaccard against the
    plan's registered >= 0.8 operating point before the row is dropped. Rows
    survive when every candidate owner verifies < 0.8 (counted in the report);
    surviving rows become owners on ALL their bands (_DedupState, r2 Major 6).
    """
    keep: dict[str, np.ndarray] = {}
    state = _DedupState()
    report: dict = {}
    for corpus in CORPUS_ORDER:
        texts = texts_by_corpus[corpus]
        sigs = minhash_signatures(texts)
        n_perm = sigs.shape[1]
        if n_perm % bands:
            raise RuntimeError(f"{n_perm} MinHash perms not divisible by {bands} LSH bands")
        rows_per_band = n_perm // bands
        mask = np.ones(len(texts), dtype=bool)
        n_within = n_across = n_rejected = 0
        for i in range(sigs.shape[0]):
            row_bands = [
                (bi, sigs[i, bi * rows_per_band : (bi + 1) * rows_per_band].tobytes())
                for bi in range(bands)
            ]
            drop_owner, rejected = state.admit((corpus, i), row_bands, _shingle_hashes(texts[i]))
            if rejected:
                n_rejected += 1  # candidate verified BELOW 0.8 — kept (never band-dropped)
            if drop_owner is not None:
                mask[i] = False
                if drop_owner[0] == corpus:
                    n_within += 1
                else:
                    n_across += 1
        keep[corpus] = mask
        report[corpus] = {
            "n_before": len(texts),
            "n_dropped_across": int(n_across),
            "n_dropped_within": int(n_within),
            "n_lsh_candidates_rejected_by_exact_jaccard": int(n_rejected),
            "n_after": int(mask.sum()),
        }
        _log(f"dedup {corpus}: {report[corpus]}")
    return keep, report


# ---------------------------------------------------------------------------
# Step 5 — per-arm draws (nested arm-1/2 subset of the arm-3 draw)
# ---------------------------------------------------------------------------


def stratified_nested_draw(
    rows: list[dict],
    stratum_of,
    n12_target: int,
    n3_target: int,
    rng: np.random.Generator,
    smoke: bool,
) -> None:
    """Set in_arm12/in_arm3 flags in place; proportional allocation, nested subsets."""
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_stratum[stratum_of(r)].append(i)
    n_pool = len(rows)
    if n_pool < n3_target:
        if not smoke:
            raise RuntimeError(f"pool {n_pool} < arm-3 draw target {n3_target}")
        n3_target = n_pool
        n12_target = min(n12_target, n_pool)
    strata = sorted(by_stratum)
    sizes = np.array([len(by_stratum[s]) for s in strata], dtype=float)

    def alloc(total: int) -> list[int]:
        raw = sizes * (total / sizes.sum())
        base = np.floor(raw).astype(int)
        base = np.minimum(base, sizes.astype(int))
        rem = total - int(base.sum())
        order = np.argsort(-(raw - base))
        for j in order:
            if rem <= 0:
                break
            if base[j] < sizes[j]:
                base[j] += 1
                rem -= 1
        if rem > 0:
            raise RuntimeError(f"cannot allocate draw of {total} over pool {int(sizes.sum())}")
        return base.tolist()

    n3_alloc = alloc(n3_target)
    if sum(n3_alloc) != n3_target:
        raise RuntimeError(f"arm-3 allocation sums to {sum(n3_alloc)} != target {n3_target}")
    n12_alloc = [min(a, b) for a, b in zip(alloc(n12_target), n3_alloc, strict=True)]
    # The elementwise nesting cap (arm-1/2 subset of the arm-3 draw) can UNDER-fill
    # the arm-1/2 target; redistribute the deficit to strata with arm-3 headroom
    # (deterministic largest-headroom-first order), then assert the sums exactly.
    deficit = n12_target - sum(n12_alloc)
    if deficit < 0:
        raise RuntimeError(f"arm-1/2 allocation over-filled by {-deficit} (nesting bug)")
    while deficit > 0:
        headroom = [n3 - n12 for n3, n12 in zip(n3_alloc, n12_alloc, strict=True)]
        order = sorted(range(len(strata)), key=lambda j: (-headroom[j], strata[j]))
        progressed = False
        for j in order:
            if deficit <= 0:
                break
            if n12_alloc[j] < n3_alloc[j]:
                n12_alloc[j] += 1
                deficit -= 1
                progressed = True
        if not progressed:
            raise RuntimeError(
                f"cannot nest arm-1/2 draw of {n12_target} inside the arm-3 allocation "
                f"(residual deficit {deficit})"
            )
    if sum(n12_alloc) != n12_target:
        raise RuntimeError(f"arm-1/2 allocation sums to {sum(n12_alloc)} != target {n12_target}")
    for s, n3_s, n12_s in zip(strata, n3_alloc, n12_alloc, strict=True):
        idx = np.array(by_stratum[s])
        perm = idx[rng.permutation(len(idx))]
        chosen3 = perm[:n3_s]
        for j, i in enumerate(chosen3):
            rows[int(i)]["in_arm3"] = True
            rows[int(i)]["in_arm12"] = j < n12_s


def capped_nested_draw(rows: list[dict], cap12: int, cap3: int, rng: np.random.Generator) -> None:
    perm = rng.permutation(len(rows))
    chosen3 = perm[: min(cap3, len(rows))]
    for j, i in enumerate(chosen3):
        rows[int(i)]["in_arm3"] = True
        rows[int(i)]["in_arm12"] = j < cap12


# ---------------------------------------------------------------------------
# Step 6 — renders + CPU model asserts
# ---------------------------------------------------------------------------


def render_user_text(corpus: str, row: dict) -> str:
    if corpus in ("gsm8k_test", "gsm8k_train", "math"):
        return RENDER_MATH.format(question=row["question"])
    if corpus in ("mmlu", "arc_challenge", "csqa", "piqa"):
        opts = "\n".join(
            f"{letter}. {text}"
            for letter, text in zip(_letters(len(row["options"])), row["options"], strict=True)
        )
        return RENDER_MCQ.format(question=row["question"], options=opts)
    if corpus == "contexthub":
        return row["question"]  # item text carries its native answer instruction
    raise RuntimeError(f"unknown corpus {corpus}")


def _chat_render(tok, user_text: str, **template_kwargs) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def run_model_asserts(
    final_rows: dict[str, list[dict]], rng: np.random.Generator, n_sample: int = 100
) -> dict:
    """Plan section 4.2 P0 item 3: AutoConfig x5 + per-arm tokenizer/render asserts."""
    report: dict = {"autoconfig_ok": [], "arms": {}}
    for arm, models in ARM_MODELS.items():
        for mid in models:
            AutoConfig.from_pretrained(mid)
            report["autoconfig_ok"].append(mid)
    _log(f"AutoConfig OK for {len(report['autoconfig_ok'])} models")

    toks = {mid: AutoTokenizer.from_pretrained(mid) for ms in ARM_MODELS.values() for mid in ms}

    samples: dict[str, list[str]] = {}
    for corpus, rows in final_rows.items():
        idx = rng.permutation(len(rows))[: min(n_sample, len(rows))]
        samples[corpus] = [rows[int(i)]["user_text"] for i in idx]

    # Think-delimiter pins (asserted on the arm's THINK-side tokenizer).
    delim_report: dict[str, dict] = {}
    for arm, pins in THINK_DELIM_PINS.items():
        think_tok = toks[ARM_MODELS[arm][-1]]
        got = {s: think_tok.encode(s, add_special_tokens=False) for s in ("<think>", "</think>")}
        for s, pin in pins.items():
            if got[s] != pin:
                raise RuntimeError(
                    f"arm {arm}: {s!r} encodes to {got[s]} != pinned {pin} ({ARM_MODELS[arm][-1]})"
                )
        delim_report[str(arm)] = {s: got[s] for s in got}
    report["think_delimiters"] = delim_report
    _log("think-delimiter pins asserted for arms 1-3")

    # Arm 1 — full-render token-id identity (the #1336 cross-model hash assert).
    pre1, post1 = (toks[m] for m in ARM_MODELS[1])
    h_pre, h_post = hashlib.sha256(), hashlib.sha256()
    n_checked = 0
    for corpus, texts in samples.items():
        for t in texts:
            ids_pre = pre1(_chat_render(pre1, t), add_special_tokens=False)["input_ids"]
            ids_post = post1(_chat_render(post1, t), add_special_tokens=False)["input_ids"]
            if ids_pre != ids_post:
                raise RuntimeError(
                    f"arm 1 G-D FAIL: full-render token ids differ on a {corpus} sample"
                )
            h_pre.update(np.array(ids_pre, dtype=np.int64).tobytes())
            h_post.update(np.array(ids_post, dtype=np.int64).tobytes())
            n_checked += 1
    if h_pre.hexdigest() != h_post.hexdigest():
        raise RuntimeError("arm 1 G-D FAIL: render hash mismatch")
    report["arms"]["1"] = {
        "full_render_hash": h_pre.hexdigest(),
        "n_prompts_checked": n_checked,
    }
    _log(f"arm 1: full-render token-id identity on {n_checked} prompts OK")

    # Arm 2 — content-token identity (shared Qwen2 BPE); full renders recorded, not failed.
    pre2, post2 = (toks[m] for m in ARM_MODELS[2])
    n_checked = 0
    for corpus, texts in samples.items():
        for t in texts:
            a = pre2.encode(t, add_special_tokens=False)
            b = post2.encode(t, add_special_tokens=False)
            if a != b:
                raise RuntimeError(f"arm 2 G-D FAIL: content token ids differ on a {corpus} sample")
            n_checked += 1
    ex = next(iter(samples.values()))[0]
    report["arms"]["2"] = {
        "n_prompts_checked": n_checked,
        "render_pre_verbatim": _chat_render(pre2, ex),
        "render_post_verbatim": _chat_render(post2, ex),
        "note": "full-render hash EXPECTED to differ (different templates); "
        "arm-2 cross-model cells carry the render-confound label",
    }
    _log(f"arm 2: content-token identity on {n_checked} prompts OK")

    # Arm 3 — cross-mode render probe (shared user prefix; empty think block under off).
    tok3 = toks[ARM_MODELS[3][0]]
    rt = _chat_render(tok3, ex, enable_thinking=True)
    rf = _chat_render(tok3, ex, enable_thinking=False)
    if rt == rf:
        raise RuntimeError("arm 3 G-D FAIL: enable_thinking toggle changed nothing")
    if not rf.startswith(rt):
        raise RuntimeError(
            "arm 3 G-D FAIL: think-off render is not think-on render + suffix; "
            f"common prefix ends at char {len(os.path.commonprefix([rt, rf]))}"
        )
    suffix = rf[len(rt) :]
    if "<think>" not in suffix or "</think>" not in suffix:
        raise RuntimeError(
            f"arm 3 G-D FAIL: empty think block absent in off-mode suffix {suffix!r}"
        )
    report["arms"]["3"] = {
        "render_think_on_verbatim": rt,
        "render_think_off_verbatim": rf,
        "off_mode_suffix": suffix,
    }
    _log("arm 3: cross-mode render probe OK (empty think block prompt-side under off)")

    # Chat-template records (verbatim, one example per arm-model).
    report["chat_templates"] = {
        mid: _chat_render(toks[mid], "EXAMPLE_USER_TEXT")
        for ms in ARM_MODELS.values()
        for mid in ms
    }
    return report


# ---------------------------------------------------------------------------
# Step 7 — write + upload
# ---------------------------------------------------------------------------


def write_jsonl_sharded(out_dir: Path, name: str, rows: list[dict]) -> list[dict]:
    """Write rows as one jsonl (or <9 MB line-shards + manifest); return file records."""
    payloads = [json.dumps(r, ensure_ascii=False) for r in rows]
    total = sum(len(p.encode("utf-8")) + 1 for p in payloads)
    files: list[dict] = []
    if total <= SHARD_BYTES_MAX:
        path = out_dir / f"{name}.jsonl"
        path.write_text("\n".join(payloads) + "\n", encoding="utf-8")
        files.append({"file": path.name, "n_rows": len(rows), "sha256": _sha256_file(path)})
        return files
    shard_idx, buf, buf_bytes = 0, [], 0
    parts: list[tuple[Path, int]] = []

    def flush() -> None:
        nonlocal shard_idx, buf, buf_bytes
        path = out_dir / f"{name}.shard{shard_idx:02d}.jsonl"
        path.write_text("\n".join(buf) + "\n", encoding="utf-8")
        parts.append((path, len(buf)))
        shard_idx, buf, buf_bytes = shard_idx + 1, [], 0

    for p in payloads:
        nbytes = len(p.encode("utf-8")) + 1
        if buf and buf_bytes + nbytes > SHARD_BYTES_MAX:
            flush()
        buf.append(p)
        buf_bytes += nbytes
    if buf:
        flush()
    man = {
        "parts": [p.name for p, _ in parts],
        "line_counts": [n for _, n in parts],
        "sha256s": [_sha256_file(p) for p, _ in parts],
        "n_rows": len(rows),
    }
    man_path = out_dir / f"{name}.manifest.json"
    man_path.write_text(json.dumps(man, indent=2))
    for p, n in parts:
        files.append({"file": p.name, "n_rows": n, "sha256": _sha256_file(p)})
    files.append({"file": man_path.name, "n_rows": None, "sha256": _sha256_file(man_path)})
    return files


def upload_bundle(out_dir: Path, hf_dest: str, expected_files: list[str]) -> None:
    dest = hub._upload(
        local_path=out_dir,
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=hf_dest,
        raise_on_error=True,
    )
    if not dest:
        raise RuntimeError("upload_bundle: empty destination from hub._upload")
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        hub.DEFAULT_DATASET_REPO,
        [f"{hf_dest}/{f}" for f in expected_files],
        path_in_repo=hf_dest,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"upload_bundle: {len(missing)} paths missing post-upload: {missing}")
    _log(f"uploaded + verified {len(expected_files)} files at {hf_dest}/")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _assert_staging_headroom(out_dir: Path, need_gb: float = 3.0) -> None:
    st = os.statvfs(out_dir)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    if free_gb < need_gb:
        raise RuntimeError(
            f"P0 staging headroom: {free_gb:.1f} GB free at {out_dir} < {need_gb} GB "
            "(>=1.5x the <2 GB projected staging footprint; plan section 9)"
        )


def run_selftest() -> int:
    """Deterministic, network-free staging-helper selftest (r2 Critical 3).

    Covers the helper seams the --smoke slice cannot reach determinately:
    TAUR discovery filtering (known repo-id fixture incl. the round-6 gated
    exclusion), the anchored contexthub pattern (round-6 __round_2_fixes
    collision fixture), CH source-repo selection (round-6 all-8-cells
    requirement), near-dup dedup (the 3-signature transitive fixture of r2
    Major 6 + a text-level exact-dup fixture with manifest counters), and
    nested draw redistribution (skewed strata forcing the deficit branch).
    """
    # -- TAUR discovery filter (known repo ids + round-6 gated exclusion) ----------
    ungated_per_model = [
        f"{TAUR_PARENT_REPO}___claude_3_sonnet",
        f"{TAUR_PARENT_REPO}___gpt_4o",
        f"{TAUR_PARENT_REPO}__Llama_3_70b",  # double-underscore id shape — INCLUDED
    ]
    gated_per_model = [
        # gated='manual' on the live 2026-08-25 listing — excluded AND recorded;
        # pre-round the 1-arg filter admitted it and load_taur 403'd (GatedRepoError).
        f"{TAUR_PARENT_REPO}___meta-llama__Llama-3.3-70B-Instruct",
    ]
    excluded_fixture = [
        TAUR_PARENT_REPO,  # collection parent (exact id) — excluded by the prefix rule
        f"{TAUR_PARENT_REPO}___gpt_4o__Paraphrase_Exp",
        f"{TAUR_PARENT_REPO}___gpt_4o__Symbolic_Solver_Experiment",
    ]
    repos, excluded_gated, excluded_other = _filter_taur_repo_ids(
        sorted(ungated_per_model + gated_per_model + excluded_fixture),
        # The parent is ALSO gated on the live listing: it must land in
        # excluded_other (prefix rule fires first), not excluded_gated.
        gated_ids=set(gated_per_model) | {TAUR_PARENT_REPO},
    )
    assert repos == sorted(ungated_per_model), repos
    assert excluded_gated == sorted(gated_per_model), excluded_gated
    assert excluded_other == sorted(excluded_fixture), excluded_other

    # -- contexthub pattern anchoring (round 6: __round_2_fixes collision) ---------
    ch_pat = TAUR_PATTERNS["contexthub"]
    for t in ("deductive", "abductive"):
        for level in (1, 2, 3, 4):
            cfg = f"contexthub_{t}_level{level}"
            assert re.search(ch_pat, cfg, flags=re.IGNORECASE), cfg
    variant = "contexthub_deductive_level2__round_2_fixes"
    # The variant parses to the SAME (type, level) cell as its base config — the
    # collision mechanism the anchor prevents ('sourced twice' raise + a second
    # n_models increment for the SAME model).
    assert _parse_contexthub_config(variant) == ("deductive", 2)
    assert not re.search(ch_pat, variant, flags=re.IGNORECASE), variant
    assert not re.search(ch_pat, "contexthub_abductive_level1__round_2_fixes", re.IGNORECASE)

    # -- CH source-repo selection (round 6: must carry ALL 8 canonical cells) ------
    partial = {("deductive", 1), ("abductive", 1), ("deductive", 2), ("abductive", 2)}
    picked = _select_ch_source_repo(
        {"repoA_partial": partial, "repoB_full": set(CH_CANONICAL_CELLS)}
    )
    assert picked == "repoB_full", picked  # first ALL-8 repo, NOT first CH-carrying repo
    try:
        _select_ch_source_repo({"repoA_partial": partial})
    except RuntimeError as e:
        assert "canonical contexthub" in str(e), e
    else:
        raise AssertionError("partial-coverage repo accepted as contexthub source")

    # -- dedup: 3-signature transitive fixture (r2 Major 6) ------------------------
    # A/B < 0.8 (B kept), B/C >= 0.8, A/C < 0.8 — under the superseded
    # single-owner `band_owner.setdefault`, C's candidate set on the shared band
    # was {A} only, and J(C, A) < 0.8 kept C (the transitive near-dup survived).
    sa = np.arange(1, 11, dtype=np.uint64)
    sb = np.arange(101, 111, dtype=np.uint64)
    sc = np.unique(np.concatenate([np.arange(102, 111, dtype=np.uint64), [np.uint64(200)]]))
    assert _jaccard_hashes(sb, sc) >= JACCARD_VERIFY_THRESHOLD  # 9/11 ~ 0.818
    assert _jaccard_hashes(sa, sc) < JACCARD_VERIFY_THRESHOLD  # the old candidate set
    b0 = (0, b"\x01")
    st = _DedupState()
    assert st.admit(("ca", 0), [b0], sa) == (None, False)
    drop_b, rej_b = st.admit(("ca", 1), [b0], sb)
    assert drop_b is None and rej_b, (drop_b, rej_b)  # candidate {A}, verified < 0.8
    drop_c, _ = st.admit(("cb", 0), [b0], sc)
    assert drop_c == ("ca", 1), f"transitive candidate missed: drop_owner={drop_c}"
    # Union completeness across DIFFERENT bands: a row sharing band b1 with one
    # owner and b2 with another must see BOTH candidates.
    st2 = _DedupState()
    st2.admit(("cx", 0), [(1, b"\x02")], sa)
    st2.admit(("cx", 1), [(2, b"\x03")], sb)
    drop_u, _ = st2.admit(("cx", 2), [(1, b"\x02"), (2, b"\x03")], sc)
    assert drop_u == ("cx", 1), drop_u

    # -- dedup: text-level exact-dup fixture (manifest counters) -------------------
    t1 = "How many apples does Maria have left after giving three to each of her four friends?"
    t2 = "A train travels at sixty miles per hour for two and a half hours; how far does it go?"
    t4 = "What is the smallest prime number greater than one hundred and twenty three exactly?"
    texts = {c: [] for c in CORPUS_ORDER}
    texts["gsm8k_test"] = [t1, t2, t1]  # third row = exact WITHIN-corpus dup
    texts["gsm8k_train"] = [t1, t4]  # first row = exact ACROSS-corpus dup
    keep, report = lsh_keep_first(texts)
    assert keep["gsm8k_test"].tolist() == [True, True, False], keep["gsm8k_test"]
    assert keep["gsm8k_train"].tolist() == [False, True], keep["gsm8k_train"]
    r_test, r_train = report["gsm8k_test"], report["gsm8k_train"]
    assert (r_test["n_dropped_within"], r_test["n_dropped_across"]) == (1, 0), r_test
    assert (r_train["n_dropped_within"], r_train["n_dropped_across"]) == (0, 1), r_train
    assert r_test["n_after"] == 2 and r_train["n_after"] == 1, (r_test, r_train)

    # -- nested draw redistribution (skewed strata force the deficit branch) -------
    # sizes [4, 47, 49], n3=14 -> [0, 7, 7]; n12=13 raw alloc [1, 6, 6] elementwise-
    # capped to [0, 6, 6] (sum 12 < 13) -> the deficit loop must move the spare
    # unit to the largest-headroom stratum (sy). Final arm12 [0, 7, 6].
    rows = [
        {"s": s, "in_arm12": False, "in_arm3": False}
        for s, cnt in (("sx", 4), ("sy", 47), ("sz", 49))
        for _ in range(cnt)
    ]
    stratified_nested_draw(rows, lambda r: r["s"], 13, 14, np.random.default_rng(0), smoke=False)
    n3_by = {s: sum(r["in_arm3"] for r in rows if r["s"] == s) for s in ("sx", "sy", "sz")}
    n12_by = {s: sum(r["in_arm12"] for r in rows if r["s"] == s) for s in ("sx", "sy", "sz")}
    assert sum(n3_by.values()) == 14 and sum(n12_by.values()) == 13, (n3_by, n12_by)
    assert n3_by == {"sx": 0, "sy": 7, "sz": 7}, n3_by
    assert n12_by == {"sx": 0, "sy": 7, "sz": 6}, n12_by
    assert all(r["in_arm3"] for r in rows if r["in_arm12"]), "arm-1/2 row outside arm-3 draw"

    print(
        "[selftest] PASS: TAUR repo filter (parent + experiment + GATED exclusions), "
        "contexthub pattern anchor (__round_2_fixes excluded), CH source all-8-cells "
        "selection, dedup (transitive band-owner union, exact-dup counters), nested "
        "draw redistribution"
    )
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true", help="network-free helper selftest")
    ap.add_argument("--out-dir", default="data/issue_2546/corpora_v1", help="local bundle dir")
    ap.add_argument(
        "--dl-dir", default="data/issue_2546/hf_dl", help="HF staging mirror root (re-downloadable)"
    )
    ap.add_argument(
        "--hf-dest",
        default=None,
        help=(
            "HF data-repo prefix — REQUIRED when uploading (no implicit issue-prefix "
            f"default, #1005 clobber shape; canonical for this issue: {HF_DEST_DEFAULT})"
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2546,
        help="corpus-sample draw seed (registered: plan section 10 'fold/corpus-sample "
        "seed 0/2546' — folds are seed 0 fit-side; the STAGING draw seed is 2546)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="20 rows/corpus, 2 TAUR repos, NO upload; all other code paths identical",
    )
    ap.add_argument("--skip-upload", action="store_true", help="build bundle only")
    return ap


def main(argv: list[str] | None = None) -> int:
    t0 = time.time()
    ap = build_argparser()
    args = ap.parse_args(argv)
    if args.selftest:
        return run_selftest()
    if not args.smoke and not args.skip_upload and not args.hf_dest:
        # Fail at parse time, BEFORE hours of pulls/joins (the late check stays as backstop).
        ap.error(
            "--hf-dest is REQUIRED when uploading (no implicit issue-prefix default — the "
            f"#1005 clobber shape; canonical for this issue: {HF_DEST_DEFAULT}); or pass "
            "--smoke/--skip-upload to build without upload"
        )
    out_dir = Path(args.out_dir + ("_smoke" if args.smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = Path(args.dl_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)
    _assert_staging_headroom(out_dir)
    _assert_staging_headroom(dl_dir)  # dl mirror may sit on a different filesystem
    rng = np.random.default_rng(args.seed)

    # -- steps 1-3: pulls + joins ------------------------------------------------
    staged = stage_1336_corpora(dl_dir)
    math_problems, math_report = extract_math_problems(staged["math"])
    gsm = join_gsm8k_gold(staged, args.smoke)
    k_report = assign_k_bins(gsm)

    corpora: dict[str, list[dict]] = {}
    corpora["gsm8k_test"] = gsm["gsm8k_test"]
    corpora["gsm8k_train"] = gsm["gsm8k_train"]
    math_rows = [
        {"question": q, "gold_answer": None, "src_index": int(r["src_index"])}
        for q, r in zip(math_problems, staged["math"], strict=True)
    ]
    corpora["math"] = math_rows[:20] if args.smoke else math_rows
    corpora.update(load_mcq_corpora(args.smoke))

    taur_repos, taur_discovery = discover_taur_repos(args.smoke)
    rescue_acc, ch_cells, taur_report = load_taur(taur_repos)
    # Backstop only: load_taur now asserts exactly the 8 canonical cells UNCONDITIONALLY
    # (smoke included — the smoke slice's first-2 un-gated repos both carry all 8).
    expected_cells = set(CH_CANONICAL_CELLS)
    if set(ch_cells) != expected_cells:
        raise RuntimeError(f"contexthub cells missing: {sorted(expected_cells - set(ch_cells))}")
    ch_gold_missing: dict[str, dict] = {}
    for (ch_type, level), rows in sorted(ch_cells.items()):
        n_missing = sum(1 for r in rows if r["gold_answer"] is None)
        ch_gold_missing[f"{ch_type}_L{level}"] = {"n_rows": len(rows), "n_gold_missing": n_missing}
        if rows and n_missing == len(rows) and not args.smoke:
            # A cell with ZERO golds silently voids the correctness covariate for the
            # whole (type x level) cell — a rig defect, never a data property to absorb.
            raise RuntimeError(
                f"contexthub cell ({ch_type}, L{level}): ALL {len(rows)} rows lack a gold "
                f"answer (keys probed: {_GOLD_KEYS}) — refusing to stage a gold-less cell"
            )
    _log(f"contexthub gold coverage: {ch_gold_missing}")
    ch_rows: list[dict] = []
    for (ch_type, level), rows in sorted(ch_cells.items()):
        ch_rows.extend(rows if not args.smoke else rows[:20])
    corpora["contexthub"] = ch_rows

    # -- step 4: near-dup dedup BEFORE any draw ----------------------------------
    keep_masks, dedup_report = lsh_keep_first(
        {c: [_norm(r["question"]) for r in corpora[c]] for c in CORPUS_ORDER}
    )
    for c in CORPUS_ORDER:
        corpora[c] = [r for r, k in zip(corpora[c], keep_masks[c], strict=True) if k]

    # -- step 5: per-arm draws ----------------------------------------------------
    for c, rows in corpora.items():
        for r in rows:
            r["in_arm12"] = c not in ("mmlu",) and not (
                c == "contexthub" and r["level"] in (2, 3, 4)
            )
            r["in_arm3"] = r["in_arm12"]
    stratified_nested_draw(
        corpora["mmlu"],
        lambda r: r["subject"],
        MMLU_DRAW_ARM12,
        MMLU_DRAW_ARM3,
        rng,
        args.smoke,
    )
    for (ch_type, level), _ in sorted(ch_cells.items()):
        if level == 1:
            continue  # all L1 rows, both arms
        cell = [r for r in corpora["contexthub"] if r["ch_type"] == ch_type and r["level"] == level]
        capped_nested_draw(cell, CH_L24_CAP_ARM12, CH_L24_CAP_ARM3, rng)

    # -- step 3b: rescue_rate join ------------------------------------------------
    rescue_family = {
        "gsm8k_test": "gsm8k_test",
        "gsm8k_train": "gsm8k_test",  # TAUR carries the gsm8k TEST set; train joins by text
        "math": "math",
        "mmlu": "mmlu",
        "arc_challenge": "arc_challenge",
        "csqa": "csqa",
        "piqa": "piqa",
        "contexthub": "contexthub",
    }
    rescue_coverage: dict[str, dict] = {}
    for c, rows in corpora.items():
        fam = rescue_family[c]
        acc = rescue_acc.get(fam, {})
        n_joined = 0
        for r in rows:
            hit = acc.get(_norm(r["question"]))
            if hit is None:
                r["rescue_rate"] = None
                r["rescue_n_models"] = 0
            else:
                r["rescue_rate"] = hit["n_rescue"] / hit["n_models"]
                r["rescue_n_models"] = hit["n_models"]
                n_joined += 1
        rescue_coverage[c] = {"n_rows": len(rows), "n_joined": n_joined}
    _log(f"rescue_rate coverage: {rescue_coverage}")

    # -- step 6: renders + model asserts -------------------------------------------
    for c, rows in corpora.items():
        for r in rows:
            r["corpus"] = c
            r["user_text"] = render_user_text(c, r)
            # contexthub src_index is per-config — key the row_id on the config too
            mid = f"{r['src_config']}:" if "src_config" in r else ""
            r["row_id"] = f"{c}:{mid}{r['src_index']}"
    assert_report = run_model_asserts(corpora, rng)

    # -- step 7: write + manifest + upload ------------------------------------------
    file_records: dict[str, list[dict]] = {}
    for c in CORPUS_ORDER:
        file_records[c] = write_jsonl_sharded(out_dir, c, corpora[c])
    counts = {
        c: {
            "n_rows": len(rows),
            "n_arm12": sum(1 for r in rows if r["in_arm12"]),
            "n_arm3": sum(1 for r in rows if r["in_arm3"]),
        }
        for c, rows in corpora.items()
    }
    ch_level_counts = dict(Counter((r["ch_type"], r["level"]) for r in corpora["contexthub"]))
    manifest = {
        "task": 2546,
        "recipe_version": "issue2546-corpora-v1-1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_sha(),
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "datasets": __import__("datasets").__version__,
            "transformers": __import__("transformers").__version__,
        },
        "seed": args.seed,
        "smoke": args.smoke,
        "join_normalization": "whitespace-collapse, case-preserved",
        "counts": counts,
        "totals": {
            "arm12": sum(v["n_arm12"] for v in counts.values()),
            "arm3": sum(v["n_arm3"] for v in counts.values()),
        },
        "k_report": k_report,
        "math_scaffold_extraction": math_report,
        "contexthub_cell_counts": {f"{t}_L{level}": n for (t, level), n in ch_level_counts.items()},
        "dedup": {
            "method": (
                "reused issue_1739.corpus_staging MinHash (char-5 shingles, 64 perms) as "
                "CANDIDATE net; drops verified by exact char-5-shingle Jaccard"
            ),
            "lsh_bands": LSH_BANDS,
            # candidate net: (1/b)^(1/r) with b=16 bands, r=4 rows/band over 64 perms ~= 0.5
            "candidate_jaccard_threshold": 0.5,
            "verified_jaccard_threshold": JACCARD_VERIFY_THRESHOLD,
            "order": CORPUS_ORDER,
            "per_corpus": dedup_report,
        },
        "contexthub_gold_missing": ch_gold_missing,
        "math_gold_note": (
            "staged math rows carry gold_answer=None BY DESIGN: MATH golds are joined "
            "pod-side at parse time by issue2546_gen_capture.stage_math_golds"
        ),
        "rescue_rate": {
            "definition": "fraction of joined TAUR models with cot_correct AND NOT direct_correct",
            "repos": taur_repos,
            "discovery": taur_discovery,
            "coverage": rescue_coverage,
            "taur_report": taur_report,
        },
        "renders": {
            "math": RENDER_MATH,
            "mcq": RENDER_MCQ,
            "contexthub": "item text verbatim (native answer instruction)",
        },
        "model_asserts": assert_report,
        "files": file_records,
    }
    man_path = out_dir / "corpora_manifest.json"
    with atomic_replace(man_path) as tmp:  # atomic: no half-written manifest on crash
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    _log(f"manifest written: {man_path}")

    if args.smoke or args.skip_upload:
        _log("upload SKIPPED (--smoke/--skip-upload)")
    else:
        if not args.hf_dest:
            raise RuntimeError(
                "--hf-dest is REQUIRED when uploading (no implicit issue-prefix default — "
                f"the #1005 clobber shape; canonical for this issue: {HF_DEST_DEFAULT}). "
                "Pass --hf-dest explicitly, or --smoke/--skip-upload to build without upload."
            )
        expected = [rec["file"] for recs in file_records.values() for rec in recs]
        expected.append(man_path.name)
        upload_bundle(out_dir, args.hf_dest, expected)

    _log(f"P0 staging complete in {time.time() - t0:.1f}s; bundle at {out_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
