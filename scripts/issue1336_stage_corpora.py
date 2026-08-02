#!/usr/bin/env python
"""Issue #1336 — Phase C: v2 corpus build (7 corpora; plan v13 §4 Corpora table).

Builds the seven revision-pinned corpus JSONLs for the
``full-corpora-stage-evals-metric-ladder`` round:

  lmsys23k         rows 0..4999 == the parent's track_s.jsonl @ TRACK_S_REV
                   (verbatim; per-prompt byte-equality asserted against the
                   live stream under the default mode); rows 5000+ continue
                   the SAME first-user-turn stream (the
                   ``issue779_collect.load_train_contexts`` predicate) at the
                   pinned lmsys revision, exact-sha256-deduped against the
                   full running prefix, until 18,000 new unique prompts pass
                   the build filters.
  gsm8k_train_full all 7,473 train questions (openai/gsm8k @ pin);
                   prompt_idx == split index (wave-1 gsm8k_train5k parity).
  gsm8k_test1319   the 1,319 test questions — the decontamination REFERENCE
                   set (exempt from the decon screen by construction).
  math7500         RLVR mix rows with dataset == "MATH" (all 7,500).
  if11k            RLVR mix dataset == "ifeval" (14,973) -> seed-1336
                   deterministic sample of 11,000.
  uf11k            preference-mixture ``prompt`` field, exact dedup ->
                   seed-1336 sample of 11,000.
  sft11k           tulu-3-sft-mixture, 3 sources stratified 3,667/3,667/3,666
                   (seed 1336), first user turn.

Build-time filters (all corpora; lmsys23k rows 0..4999 are VERBATIM — decon
matches there are COUNTED, never dropped, so the byte invariant holds):
non-empty first user turn; Tulu-render prompt-token budget
<= PROMPT_TOKEN_BUDGET (the exact ``issue1336_gen_answers.run_prep``
arithmetic incl. its +1 BOS margin); normalized-exact-match decontamination
screen vs the 1,319 GSM8K test questions (lowercase, whitespace-collapse,
sha256), dropped rows counted per corpus.

Structural asserts (fail-loud, BOTH modes — the plan §12 probes at full
consumed grain): GSM8K split sizes == {7,473, 1,319}; RLVR mix ``dataset``
component counts == {gsm8k: 7,473, MATH: 7,500, ifeval: 14,973}; preference /
SFT mix row + per-source totals == the plan §10 counted grain.

Content hygiene: LMSYS / UltraFeedback / wildchat prompt text is NEVER
printed or logged — digests, counts, and indices only.

Usage::

    uv run python scripts/issue1336_stage_corpora.py --smoke
    uv run python scripts/issue1336_stage_corpora.py --full --upload
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# VM thread caps (#847/#891) — set BEFORE any torch/transformers import so the
# pools freeze at the capped width; an explicit launcher env still wins.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("issue1336_stage_corpora")

DATA_ROOT = Path("data/issue_1336")
V2_HF_PREFIX = f"{cm.HF_PREFIX_1336}/corpora_v2"

# Bumping this string invalidates every persisted build + stream checkpoint
# (fingerprint-gated resume; the #952/#1092/#1902 pattern).
FILTER_RECIPE_VERSION = "issue1336-corpora-v2-1"
CORPUS_SEED = 1336  # plan §10 "corpus sample seed 1336"

# Source pins (plan v13 §10 Reproducibility Card — do not retype elsewhere).
LMSYS_DATASET = "lmsys/lmsys-chat-1m"
LMSYS_REV = "200748d9d3cd"
RLVR_DATASET = "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"
RLVR_REV = "7dbd180f5440"
RLVR_COMPONENT_COUNTS = {"gsm8k": 7473, "MATH": 7500, "ifeval": 14973}  # plan §12 assumption 2
PREF_DATASET = "allenai/llama-3.1-tulu-3-8b-preference-mixture"
PREF_REV = "78a6f0078594"
PREF_ROWS = 272_898  # plan §10 counted grain
SFT_DATASET = "allenai/tulu-3-sft-mixture"
SFT_REV = "b14afda60f1b"
SFT_ROWS = 939_343  # plan §10 counted grain
SFT_SOURCE_COUNTS = {
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k": 100_000,
    "ai2-adapt-dev/flan_v2_converted": 89_982,
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": 107_276,
}
SFT_QUOTAS = {
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k": 3667,
    "ai2-adapt-dev/flan_v2_converted": 3667,
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": 3666,
}

LMSYS_PREFIX_N = 5000
LMSYS_NEW_N = 18_000
LMSYS_SCAN_CAP = 400_000  # fail-loud safety cap (expected scan ~25-35k rows)
CHECKPOINT_EVERY_SCANNED = 20_000

# Extraction-failure ceiling (plan §12 assumption 3: >2% => plan defect —
# enforced full-mode only; smoke demotes the verdict to a log line, the
# gate-calibration rule).
EXTRACT_FAIL_CEILING = 0.02

# Tulu tokenizer for the load-time budget gate (plan §4: "Tulu-render
# prompt-token budget ... under the Tulu tokenizer"). The SFT checkpoint is
# ungated; contexts tokenize identically across all 5 ladder checkpoints
# (plan §12 assumption 13, wave-1 verified property).
TOKENIZER_ID = cm.MODELS["sft"]["hf_id"]

# Upload-policy text sharding (mirrors issue1336_gen_answers): the Hub
# force-routes any >10 MB blob to LFS (#541); text >9.5 MB line-splits into
# <9 MB shards, NEVER gzip.
TEXT_SPLIT_THRESHOLD = 9_500_000
SHARD_MAX_BYTES = 9_000_000

# Smoke caps ("tiny slices"): sample/stream sizes only — the full-grain
# structural asserts (split sizes, component counts) still run in BOTH modes.
SMOKE_SAMPLE_N = 8
SMOKE_LMSYS_PREFIX_N = 8
# 24, not 8 (Unit D fit-floor sizing): real lmsys extension prompts pass the
# gen render filter (u1 >= 8 content tokens) at ~0.4-0.5 measured on the
# smoke stream, and the downstream v2 fit smoke needs the kept stem at or
# above the demonstrated n=8 floor (24 staged -> ~10-12 kept; 8 staged -> 3
# kept crashed the fold sweep all-NaN — smoke-slice arithmetic, #1489 class).
SMOKE_LMSYS_NEW_N = 24
SMOKE_LMSYS_SCAN_CAP = 3000
SMOKE_SFT_QUOTAS = {
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k": 3,
    "ai2-adapt-dev/flan_v2_converted": 3,
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": 2,
}

# v2 corpus registry — canonical copy now lives in the shared cell-registry
# module (experiments/issue_1336/common.py; Unit B folded it there so
# CELLS_V2 derives formats without importing scripts/). Re-exported here for
# the established `from issue1336_stage_corpora import V2_CORPORA` consumers
# (issue1336_gen_answers prep/format resolution).
V2_CORPORA: dict[str, dict] = cm.V2_CORPORA
# Build order: cheap pinned corpora first (checkpoint-per-phase — each
# persists the moment it completes), the long lmsys stream last.
BUILD_ORDER = (
    "gsm8k_test1319",
    "gsm8k_train_full",
    "math7500",
    "if11k",
    "uf11k",
    "sft11k",
    "lmsys23k",
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_issue1336_stage_corpora.py)
# ---------------------------------------------------------------------------
def prompt_sha(text: str) -> str:
    """Exact-content dedup key: sha256 hex of the raw prompt string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def decon_key(text: str) -> str:
    """Decontamination normalizer: lowercase + whitespace-collapse + sha256."""
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def first_user_turn(messages: list) -> str | None:
    """First user turn of a tulu-style ``messages`` list, stripped.

    Returns None when no user message with non-empty string content exists
    (the extraction-failure case counted per plan §12 assumption 3).
    """
    if not isinstance(messages, list):
        return None
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def lmsys_first_turn(row: dict) -> str | None:
    """The exact ``issue779_collect.load_train_contexts`` keep predicate.

    Takes ``conversation[0]`` (no role check — parity with the parent
    loader), reads ``content``/``value``, keeps non-empty strings stripped.
    """
    val = row.get("conversation")
    p = None
    if isinstance(val, list) and val and isinstance(val[0], dict):
        p = val[0].get("content") or val[0].get("value")
    elif isinstance(val, str):
        p = val
    if p and isinstance(p, str) and len(p.strip()) > 0:
        return p.strip()
    return None


def seeded_sample_indices(n_pool: int, k: int, seed: int = CORPUS_SEED) -> list[int]:
    """Deterministic sorted sample of k pool positions (without replacement).

    Fresh ``random.Random(seed)`` per call; sorted ascending so kept rows
    keep source order. k >= n_pool returns the whole range.
    """
    if k >= n_pool:
        return list(range(n_pool))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_pool), k))


def dedup_keep_first(items: list[str]) -> list[int]:
    """Indices of the FIRST occurrence of each distinct string, in order."""
    seen: set[str] = set()
    out: list[int] = []
    for i, s in enumerate(items):
        h = prompt_sha(s)
        if h in seen:
            continue
        seen.add(h)
        out.append(i)
    return out


def over_budget_flags(prompts: list[str], tok, *, budget: int | None = None) -> list[bool]:
    """Per-prompt True when the FORMATTED Tulu render exceeds the load-time
    budget — the exact ``issue1336_gen_answers.run_prep`` arithmetic
    (``n_tok + 1 > PROMPT_TOKEN_BUDGET``; +1 for the BOS vLLM adds)."""
    budget = cm.PROMPT_TOKEN_BUDGET if budget is None else budget
    out: list[bool] = []
    chunk = 512
    for i in range(0, len(prompts), chunk):
        batch = [cm.tulu_prompt(p) for p in prompts[i : i + chunk]]
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            out.append(len(ids) + 1 > budget)
    return out


def apply_build_filters(
    rows: list[dict],
    *,
    tok,
    decon_keys: set[str],
    decon_exempt: bool = False,
    budget: int | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Shared build-time filter chain: empty -> decon -> token budget.

    Returns (kept_rows, drop_counts). Order is first-match-wins for the
    counters; the budget check runs LAST (batched tokenization on survivors
    only). ``decon_exempt`` is for the reference set itself (gsm8k_test1319).
    """
    drops = {"empty": 0, "decon": 0, "over_budget": 0}
    survivors: list[dict] = []
    for r in rows:
        p = r.get("prompt")
        if not isinstance(p, str) or not p.strip():
            drops["empty"] += 1
            continue
        if not decon_exempt and decon_key(p) in decon_keys:
            drops["decon"] += 1
            continue
        survivors.append(r)
    flags = over_budget_flags([r["prompt"] for r in survivors], tok, budget=budget)
    kept = [r for r, ob in zip(survivors, flags, strict=True) if not ob]
    drops["over_budget"] = int(sum(flags))
    return kept, drops


def corpus_meta(
    slug: str,
    *,
    n_built: int,
    n_dropped_by_filter: dict[str, int],
    n_dropped_decon: int,
    sha256: str,
    source_revision: str,
    fingerprint: dict,
    extra: dict | None = None,
) -> dict:
    """Per-corpus manifest entry (schema pinned by the unit tests)."""
    entry = {
        "corpus": slug,
        "n_built": n_built,
        "n_dropped_by_filter": dict(n_dropped_by_filter),
        "n_dropped_decon": n_dropped_decon,
        "sha256": sha256,
        "source_revision": source_revision,
        "fingerprint": dict(fingerprint),
    }
    if extra:
        entry.update(extra)
    return entry


class LmsysAccumulator:
    """Pure per-kept-row classifier for the lmsys23k build (network-free).

    Default mode: the first ``prefix_n`` predicate-kept stream rows must
    byte-equal the track_s prompts (plan §12 assumption 4; RuntimeError on
    mismatch names the position + shas, never prompt text). Fallback mode
    (``--prefix-fallback exclusion-join``): no positional assert — every
    kept row is an extension candidate excluded against the track_s shas.
    Extension candidates dedup (exact sha256) against the FULL running
    prefix, then decon-screen, then token-budget (``is_over_budget``).
    """

    def __init__(
        self,
        track_prompts: list[str],
        *,
        prefix_n: int,
        new_n: int,
        fallback: bool,
        is_over_budget,
        decon_keys: set[str],
    ) -> None:
        self.track = list(track_prompts)[:prefix_n]
        assert len(self.track) == prefix_n, (
            f"track_s prefix needs {prefix_n} prompts, got {len(self.track)}"
        )
        self.prefix_n = prefix_n
        self.new_n = new_n
        self.fallback = fallback
        self.is_over_budget = is_over_budget
        self.decon_keys = decon_keys
        self.seen = {prompt_sha(p) for p in self.track}
        self.n_prefix = 0
        self.ext_rows: list[dict] = []
        self.prefix_decon_matches = 0
        self.rejects = {"dup": 0, "decon": 0, "over_budget": 0}

    @property
    def prefix_done(self) -> bool:
        return self.fallback or self.n_prefix >= self.prefix_n

    @property
    def done(self) -> bool:
        return self.prefix_done and len(self.ext_rows) >= self.new_n

    def offer(self, prompt: str, scan_index: int) -> str:
        """Classify one predicate-kept stream row; returns the disposition."""
        if not self.prefix_done:
            expected = self.track[self.n_prefix]
            if prompt != expected:
                raise RuntimeError(
                    f"lmsys stream kept-row {self.n_prefix} (scan_index={scan_index}) != "
                    f"track_s prompt {self.n_prefix} — sha {prompt_sha(prompt)[:16]} vs "
                    f"{prompt_sha(expected)[:16]} (plan §12 assumption 4 byte-equality FAILED); "
                    "rerun with --prefix-fallback exclusion-join (registered fallback)"
                )
            if decon_key(prompt) in self.decon_keys:
                self.prefix_decon_matches += 1  # counted, never dropped (verbatim prefix)
            self.n_prefix += 1
            return "prefix"
        if len(self.ext_rows) >= self.new_n:
            return "ignored"
        sha = prompt_sha(prompt)
        if sha in self.seen:
            self.rejects["dup"] += 1
            return "dup"
        if decon_key(prompt) in self.decon_keys:
            self.rejects["decon"] += 1
            return "decon"
        if self.is_over_budget(prompt):
            self.rejects["over_budget"] += 1
            return "over_budget"
        self.seen.add(sha)
        self.ext_rows.append({"prompt": prompt, "scan_index": scan_index})
        return "kept"

    def restore(self, pool_rows: list[dict]) -> None:
        """Fast-path resume from a checkpoint pool (no re-tokenization).

        Prefix rows re-verify byte-equality (cheap); ext rows re-enter the
        sha set directly — they already passed every filter when kept.
        """
        for r in pool_rows:
            if r["kind"] == "prefix":
                assert not self.fallback, "prefix rows in pool but fallback mode is on"
                assert r["prompt"] == self.track[self.n_prefix], (
                    f"checkpoint pool prefix row {self.n_prefix} diverges from track_s — "
                    "corrupt cache; delete the stream cache dir"
                )
                if decon_key(r["prompt"]) in self.decon_keys:
                    self.prefix_decon_matches += 1
                self.n_prefix += 1
            else:
                self.seen.add(prompt_sha(r["prompt"]))
                self.ext_rows.append({"prompt": r["prompt"], "scan_index": r["scan_index"]})


# ---------------------------------------------------------------------------
# JSONL io + upload-side text sharding (never splitlines(); #825/#950)
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode line iteration — never splitlines() (U+2028 in real text)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    logger.info("[stage-corpora] wrote %s (%d rows)", path, len(rows))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_corpus_for_upload(
    path: Path, *, threshold: int = TEXT_SPLIT_THRESHOLD, shard_max: int = SHARD_MAX_BYTES
) -> bool:
    """Line-split ``<slug>.jsonl`` into <9 MB shards + manifest when >9.5 MB.

    Generalized twin of ``issue1336_gen_answers._split_answers_for_upload``
    (kept separate — the gen function stays byte-identical, Unit A contract).
    Local consumers keep reading the single ``<slug>.jsonl``; shards +
    ``<slug>.manifest.json`` exist for the non-LFS Hub upload only.
    Idempotent: stale shards are removed before re-splitting. Returns True
    when the sharded form is the upload shape.
    """
    assert path.name.endswith(".jsonl"), path
    stem = path.name[: -len(".jsonl")]
    stale = [*path.parent.glob(f"{stem}.shard*.jsonl"), path.parent / f"{stem}.manifest.json"]
    for old in stale:
        if old.exists():
            old.unlink()
    if path.stat().st_size <= threshold:
        return False
    parts: list[str] = []
    line_counts: list[int] = []
    shas: list[str] = []
    buf: list[bytes] = []
    size = 0

    def _flush() -> None:
        nonlocal buf, size
        if not buf:
            return
        name = f"{stem}.shard{len(parts):02d}.jsonl"
        data = b"".join(buf)
        (path.parent / name).write_bytes(data)
        parts.append(name)
        line_counts.append(len(buf))
        shas.append(hashlib.sha256(data).hexdigest())
        buf = []
        size = 0

    with open(path, "rb") as fh:
        for line in fh:  # binary iteration splits on \n only (U+2028-safe)
            if buf and size + len(line) > shard_max:
                _flush()
            buf.append(line)
            size += len(line)
    _flush()
    manifest = {
        "parts": parts,
        "line_counts": line_counts,
        "sha256s": shas,
        "total_sha256": _file_sha256(path),
        "total_bytes": path.stat().st_size,
    }
    (path.parent / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "[stage-corpora] %s over %d B -> %d upload shards", path.name, threshold, len(parts)
    )
    return True


def read_corpus_rows_local(dir_: Path, slug: str) -> list[dict] | None:
    """Read a corpus from ``<slug>.jsonl``, else reassemble sha-verified
    shards via ``<slug>.manifest.json``; None when neither exists."""
    single = dir_ / f"{slug}.jsonl"
    if single.exists():
        return _read_jsonl(single)
    man_path = dir_ / f"{slug}.manifest.json"
    if not man_path.exists():
        return None
    manifest = json.loads(man_path.read_text())
    # Verify every shard's sha FIRST, then concatenate (write only after all
    # parts verified so a bad shard never leaves a partial single file).
    for part, sha in zip(manifest["parts"], manifest["sha256s"], strict=True):
        got = hashlib.sha256((dir_ / part).read_bytes()).hexdigest()
        assert got == sha, f"{slug} shard {part}: sha256 {got} != manifest {sha}"
    tmp = dir_ / f"{slug}.jsonl.tmp"
    with open(tmp, "wb") as fh:
        for part in manifest["parts"]:
            fh.write((dir_ / part).read_bytes())
    total = _file_sha256(tmp)
    assert total == manifest["total_sha256"], (
        f"{slug}: reassembled sha256 {total} != manifest {manifest['total_sha256']}"
    )
    os.replace(tmp, single)
    return _read_jsonl(single)


def _hub_download(filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    return Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                repo_id=cm.HF_DATA_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=local_dir,
            ),
            what=f"corpora_v2 download {filename}",
        )
    )


def load_v2_corpus_rows(
    slug: str,
    *,
    explicit_path: str | Path | None = None,
    smoke: bool = False,
    data_root: Path = DATA_ROOT,
) -> list[dict]:
    """Load one v2 corpus's rows for downstream consumers (gen prep).

    Resolution order: explicit path -> local corpora_v2[_smoke] dir (single
    file or sha-verified shards) -> HF data repo ``corpora_v2/`` prefix
    (single file, else manifest + shards). Fail-loud when nothing resolves.
    """
    assert slug in V2_CORPORA, f"unknown v2 corpus {slug!r} (known: {sorted(V2_CORPORA)})"
    if explicit_path is not None:
        p = Path(explicit_path)
        assert p.exists(), f"--corpus-file path for {slug} does not exist: {p}"
        return _read_jsonl(p)
    local_dir = data_root / ("corpora_v2_smoke" if smoke else "corpora_v2")
    rows = read_corpus_rows_local(local_dir, slug)
    if rows is not None:
        return rows
    if smoke:
        raise RuntimeError(
            f"smoke corpus {slug} not staged at {local_dir} — run "
            "`issue1336_stage_corpora.py --smoke` first (smoke builds are never uploaded)"
        )
    from huggingface_hub.errors import EntryNotFoundError

    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        _hub_download(f"{V2_HF_PREFIX}/{slug}.jsonl", local_dir)
    except EntryNotFoundError:
        _hub_download(f"{V2_HF_PREFIX}/{slug}.manifest.json", local_dir)
        manifest = json.loads((local_dir / V2_HF_PREFIX / f"{slug}.manifest.json").read_text())
        for part in manifest["parts"]:
            _hub_download(f"{V2_HF_PREFIX}/{part}", local_dir)
    fetched_dir = local_dir / V2_HF_PREFIX  # hf_hub_download mirrors the repo-relative path
    rows = read_corpus_rows_local(fetched_dir, slug)
    if rows is None:
        raise RuntimeError(
            f"corpus {slug} unresolvable: not at {local_dir}, and the Hub fetch under "
            f"{cm.HF_DATA_REPO}:{V2_HF_PREFIX} produced no readable file"
        )
    return rows


# ---------------------------------------------------------------------------
# Source loaders (deferred heavy imports; full-grain asserts in BOTH modes)
# ---------------------------------------------------------------------------
def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def _load_gsm8k(split: str):
    from datasets import load_dataset

    ds = load_dataset(cm.GSM8K_DATASET, cm.GSM8K_CONFIG, split=split, revision=cm.GSM8K_REV)
    assert len(ds) == cm.GSM8K_SPLIT_SIZES[split], (
        f"gsm8k {split} split has {len(ds)} rows != pinned {cm.GSM8K_SPLIT_SIZES[split]}"
    )
    return ds


def _fetch_track_s() -> list[dict]:
    """Pinned #825 track_s prompts (byte-size + row-count asserted)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    local = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=cm.HF_DATA_REPO,
            repo_type="dataset",
            filename=cm.TRACK_S_PATH,
            revision=cm.TRACK_S_REV,
            local_dir=DATA_ROOT / "hf_dl",
        ),
        what="track_s download",
    )
    nbytes = Path(local).stat().st_size
    assert nbytes == cm.TRACK_S_BYTES, (
        f"track_s.jsonl byte size {nbytes} != pinned {cm.TRACK_S_BYTES} at rev "
        f"{cm.TRACK_S_REV} — content-identity check failed"
    )
    rows = _read_jsonl(Path(local))
    assert len(rows) == LMSYS_PREFIX_N, f"track_s rows {len(rows)} != {LMSYS_PREFIX_N}"
    assert [int(r["prompt_idx"]) for r in rows] == list(range(LMSYS_PREFIX_N)), (
        "track_s prompt_idx is not 0..4999 in order — parent convention drifted"
    )
    return rows


def _git_sha() -> str:
    """Repro metadata; degrades on git-less lanes (env -> check=False -> literal)."""
    env = os.environ.get("EPS_GIT_SHA")
    if env:
        return env
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


# ---------------------------------------------------------------------------
# Per-corpus builders
# ---------------------------------------------------------------------------
def _build_gsm8k_corpus(slug: str, split: str, ctx: dict) -> tuple[list[dict], dict]:
    ds = _load_gsm8k(split)
    rows = [{"prompt_idx": i, "src_index": i, "prompt": ds[i]["question"]} for i in range(len(ds))]
    # prompt_idx == split index (wave-1 parity: gsm8k_train5k used 0..4999 of
    # the same order); filter drops leave HOLES, never renumber.
    exempt = slug == "gsm8k_test1319"  # the decon REFERENCE set itself
    kept, drops = apply_build_filters(
        rows, tok=ctx["tok"], decon_keys=ctx["decon_keys"], decon_exempt=exempt
    )
    if ctx["smoke"]:
        kept = kept[:SMOKE_SAMPLE_N]
    extra = {"decon_role": "reference-set (exempt)"} if exempt else {}
    return kept, {"drops": drops, "n_extract_failed": 0, "extra": extra, "rev": cm.GSM8K_REV}


def _check_extract_rate(slug: str, n_failed: int, n_total: int, smoke: bool) -> float:
    rate = (n_failed / n_total) if n_total else 0.0
    if rate > EXTRACT_FAIL_CEILING:
        msg = (
            f"{slug}: first-user-turn extraction failed on {n_failed}/{n_total} sampled rows "
            f"({rate:.4f} > {EXTRACT_FAIL_CEILING}) — plan §12 assumption 3 violated"
        )
        if smoke:
            logger.warning("[stage-corpora] SMOKE-DEMOTED verdict: %s", msg)
        else:
            raise RuntimeError(msg + "; amend the plan before GEN")
    return rate


def _load_rlvr():
    from datasets import load_dataset

    ds = load_dataset(RLVR_DATASET, split="train", revision=RLVR_REV)
    counts = dict(Counter(ds["dataset"]))
    assert counts == RLVR_COMPONENT_COUNTS, (
        f"RLVR mix component counts {counts} != pinned {RLVR_COMPONENT_COUNTS} "
        "(plan §12 assumption 2 FAILED)"
    )
    return ds


def _build_rlvr_corpus(slug: str, ctx: dict) -> tuple[list[dict], dict]:
    ds = ctx["rlvr_ds"]
    component = "MATH" if slug == "math7500" else "ifeval"
    col = ds["dataset"]
    idxs = [i for i, d in enumerate(col) if d == component]
    if slug == "if11k":
        take = SMOKE_SAMPLE_N if ctx["smoke"] else 11_000
        sel = seeded_sample_indices(len(idxs), take)
        idxs = [idxs[j] for j in sel]
    elif ctx["smoke"]:
        idxs = idxs[: SMOKE_SAMPLE_N * 4]
    rows, n_failed = [], 0
    for i in idxs:
        p = first_user_turn(ds[int(i)]["messages"])
        if p is None:
            n_failed += 1
            continue
        rows.append({"src_index": int(i), "prompt": p})
    rate = _check_extract_rate(slug, n_failed, len(idxs), ctx["smoke"])
    kept, drops = apply_build_filters(rows, tok=ctx["tok"], decon_keys=ctx["decon_keys"])
    if ctx["smoke"]:
        kept = kept[:SMOKE_SAMPLE_N]
    for j, r in enumerate(kept):
        r["prompt_idx"] = j
    drops["extract_failed"] = n_failed
    return kept, {
        "drops": drops,
        "n_extract_failed": n_failed,
        "extra": {"component": component, "extract_fail_rate": rate},
        "rev": RLVR_REV,
    }


def _build_uf_corpus(ctx: dict) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    ds = load_dataset(PREF_DATASET, split="train", revision=PREF_REV)
    assert len(ds) == PREF_ROWS, f"preference mix has {len(ds)} rows != pinned {PREF_ROWS}"
    prompts = ds["prompt"]
    pool_idx = dedup_keep_first(prompts)  # exact dedup, keep-first (plan §4)
    n_dup = len(prompts) - len(pool_idx)
    take = SMOKE_SAMPLE_N if ctx["smoke"] else 11_000
    sel = seeded_sample_indices(len(pool_idx), take)
    rows, n_failed = [], 0
    for j in sel:
        i = pool_idx[j]
        p = prompts[i]
        if not isinstance(p, str) or not p.strip():
            n_failed += 1
            continue
        rows.append({"src_index": int(i), "prompt": p.strip()})
    rate = _check_extract_rate("uf11k", n_failed, len(sel), ctx["smoke"])
    kept, drops = apply_build_filters(rows, tok=ctx["tok"], decon_keys=ctx["decon_keys"])
    if ctx["smoke"]:
        kept = kept[:SMOKE_SAMPLE_N]
    for j, r in enumerate(kept):
        r["prompt_idx"] = j
    drops["extract_failed"] = n_failed
    return kept, {
        "drops": drops,
        "n_extract_failed": n_failed,
        "extra": {
            "n_pool_after_dedup": len(pool_idx),
            "n_source_duplicates": n_dup,
            "extract_fail_rate": rate,
        },
        "rev": PREF_REV,
    }


def _build_sft_corpus(ctx: dict) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    ds = load_dataset(SFT_DATASET, split="train", revision=SFT_REV)
    assert len(ds) == SFT_ROWS, f"SFT mix has {len(ds)} rows != pinned {SFT_ROWS}"
    srcs = ds["source"]
    per_source: dict[str, list[int]] = {s: [] for s in SFT_QUOTAS}
    for i, s in enumerate(srcs):
        if s in per_source:
            per_source[s].append(i)
    realized = {s: len(v) for s, v in per_source.items()}
    assert realized == SFT_SOURCE_COUNTS, (
        f"SFT mix per-source counts {realized} != pinned {SFT_SOURCE_COUNTS}"
    )
    quotas = SMOKE_SFT_QUOTAS if ctx["smoke"] else SFT_QUOTAS
    rows, n_failed, n_sampled = [], 0, 0
    for source in SFT_QUOTAS:  # fixed iteration order (registry order)
        idxs = per_source[source]
        sel = seeded_sample_indices(len(idxs), quotas[source])
        n_sampled += len(sel)
        for j in sel:
            i = idxs[j]
            p = first_user_turn(ds[int(i)]["messages"])
            if p is None:
                n_failed += 1
                continue
            rows.append({"src_index": int(i), "source": source, "prompt": p})
    rate = _check_extract_rate("sft11k", n_failed, n_sampled, ctx["smoke"])
    kept, drops = apply_build_filters(rows, tok=ctx["tok"], decon_keys=ctx["decon_keys"])
    for j, r in enumerate(kept):
        r["prompt_idx"] = j
    drops["extract_failed"] = n_failed
    kept_by_source = dict(Counter(r["source"] for r in kept))
    return kept, {
        "drops": drops,
        "n_extract_failed": n_failed,
        "extra": {
            "strata_quotas": dict(quotas),
            "kept_by_source": kept_by_source,
            "extract_fail_rate": rate,
        },
        "rev": SFT_REV,
    }


# ---------------------------------------------------------------------------
# lmsys23k: streaming scan (checkpoint-per-chunk + fingerprint-gated resume —
# the issue1902_corpus.py pattern)
# ---------------------------------------------------------------------------
def _stream_fingerprint(prefix_n: int, new_n: int, fallback: bool, smoke: bool) -> dict:
    return {
        "dataset": LMSYS_DATASET,
        "revision": LMSYS_REV,
        "recipe_version": FILTER_RECIPE_VERSION,
        "prefix_n": prefix_n,
        "new_n": new_n,
        "fallback": bool(fallback),
        "smoke": bool(smoke),
        "budget": cm.PROMPT_TOKEN_BUDGET,
        "tokenizer": TOKENIZER_ID,
        "track_s_rev": cm.TRACK_S_REV,
    }


def _write_stream_checkpoint(
    pool_path: Path,
    meta_path: Path,
    *,
    fingerprint: dict,
    pool_rows: list[dict],
    scanned: int,
    predicate_rejects: int,
    acc: LmsysAccumulator,
    complete: bool,
) -> None:
    """Atomic checkpoint: pool JSONL FIRST, meta sidecar LAST."""
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pool_path.with_name(pool_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in pool_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, pool_path)
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    tmp_meta.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "scanned": scanned,
                "n_pool": len(pool_rows),
                "predicate_rejects": predicate_rejects,
                "rejects": dict(acc.rejects),
                "prefix_decon_matches": acc.prefix_decon_matches,
                "complete": complete,
            },
            indent=2,
        )
        + "\n"
    )
    os.replace(tmp_meta, meta_path)


def _build_lmsys_corpus(ctx: dict) -> tuple[list[dict], dict]:
    track_rows = _fetch_track_s()
    smoke = ctx["smoke"]
    prefix_n = SMOKE_LMSYS_PREFIX_N if smoke else LMSYS_PREFIX_N
    new_n = SMOKE_LMSYS_NEW_N if smoke else LMSYS_NEW_N
    scan_cap = SMOKE_LMSYS_SCAN_CAP if smoke else LMSYS_SCAN_CAP
    fallback = ctx["prefix_fallback"] == "exclusion-join"
    tok = ctx["tok"]

    def _is_over_budget(p: str) -> bool:
        return over_budget_flags([p], tok)[0]

    acc = LmsysAccumulator(
        [r["prompt"] for r in track_rows],
        prefix_n=prefix_n,
        new_n=new_n,
        fallback=fallback,
        is_over_budget=_is_over_budget,
        decon_keys=ctx["decon_keys"],
    )
    fingerprint = _stream_fingerprint(prefix_n, new_n, fallback, smoke)
    cache_dir = ctx["cache_dir"]
    pool_path = cache_dir / "lmsys23k_pool.jsonl"
    meta_path = cache_dir / "lmsys23k_pool.meta.json"
    pool_rows: list[dict] = []
    scanned = 0
    predicate_rejects = 0
    if meta_path.exists() and pool_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fingerprint:
            pool_rows = _read_jsonl(pool_path)
            assert len(pool_rows) == int(meta["n_pool"]), (
                f"stream cache pool {pool_path} holds {len(pool_rows)} rows but meta records "
                f"{meta['n_pool']} — corrupt cache; delete {cache_dir}"
            )
            scanned = int(meta["scanned"])
            predicate_rejects = int(meta.get("predicate_rejects", 0))
            acc.restore(pool_rows)
            acc.rejects.update(meta.get("rejects") or {})
            logger.info(
                "[stage-corpora] lmsys stream resume: scanned=%d prefix=%d ext=%d",
                scanned,
                acc.n_prefix,
                len(acc.ext_rows),
            )
        else:
            logger.info("[stage-corpora] lmsys stream cache fingerprint MISMATCH — re-streaming")
            pool_rows, scanned = [], 0

    if not acc.done:
        from datasets import load_dataset

        ds = load_dataset(LMSYS_DATASET, split="train", streaming=True, revision=LMSYS_REV)
        t0 = time.time()
        row_iter = iter(ds)
        for i, row in enumerate(row_iter):
            if i < scanned:
                continue
            if i >= scan_cap:
                break
            p = lmsys_first_turn(row)
            if p is None:
                predicate_rejects += 1
            else:
                verdict = acc.offer(p, i)
                if verdict in ("prefix", "kept"):
                    kind = "prefix" if verdict == "prefix" else "ext"
                    pool_rows.append({"kind": kind, "scan_index": i, "prompt": p})
            scanned = i + 1
            if scanned % CHECKPOINT_EVERY_SCANNED == 0:
                _write_stream_checkpoint(
                    pool_path,
                    meta_path,
                    fingerprint=fingerprint,
                    pool_rows=pool_rows,
                    scanned=scanned,
                    predicate_rejects=predicate_rejects,
                    acc=acc,
                    complete=False,
                )
                logger.info(
                    "[stage-corpora] lmsys stream: scanned=%d prefix=%d/%d ext=%d/%d elapsed=%.0fs",
                    scanned,
                    acc.n_prefix,
                    prefix_n,
                    len(acc.ext_rows),
                    new_n,
                    time.time() - t0,
                )
            if acc.done:
                break
        # datasets streaming shutdown SIGABRT (#952): release deterministically.
        del row_iter, ds
        gc.collect()
    if not acc.done:
        raise RuntimeError(
            f"lmsys stream exhausted scan cap {scan_cap} at scanned={scanned} with "
            f"prefix={acc.n_prefix}/{prefix_n}, ext={len(acc.ext_rows)}/{new_n} — "
            "raise the cap or inspect the reject counters"
        )
    _write_stream_checkpoint(
        pool_path,
        meta_path,
        fingerprint=fingerprint,
        pool_rows=pool_rows,
        scanned=scanned,
        predicate_rejects=predicate_rejects,
        acc=acc,
        complete=True,
    )

    kept: list[dict] = [
        {"prompt_idx": int(r["prompt_idx"]), "prompt": r["prompt"], "src": "track_s"}
        for r in track_rows[:prefix_n]
    ]
    # Extension prompt_idx is anchored at the PRODUCTION prefix width
    # (LMSYS_PREFIX_N), not the realized prefix_n: production is byte-identical
    # (prefix_n == LMSYS_PREFIX_N there), and the SMOKE corpus then carries the
    # same structural invariant the extractor's extension filter asserts
    # (every extension row >= cm.V2_CONCAT_BOUNDARY["lmsys23k"]) — smoke IS the
    # sweep through the identical filter, no smoke-only boundary (Unit D).
    for j, r in enumerate(acc.ext_rows):
        kept.append(
            {
                "prompt_idx": LMSYS_PREFIX_N + j,
                "prompt": r["prompt"],
                "src": "lmsys_stream",
                "scan_index": r["scan_index"],
            }
        )
    drops = {
        "dup": acc.rejects["dup"],
        "over_budget": acc.rejects["over_budget"],
        "empty_first_turn": predicate_rejects,
    }
    info = {
        "drops": drops,
        "n_extract_failed": 0,
        "n_decon": acc.rejects["decon"],
        "extra": {
            "prefix_n": prefix_n,
            "new_n": new_n,
            "scanned": scanned,
            "prefix_mode": "exclusion-join" if fallback else "byte-asserted",
            "prefix_decon_matches_kept": acc.prefix_decon_matches,
            "track_s_rev": cm.TRACK_S_REV,
        },
        "rev": LMSYS_REV,
    }
    return kept, info


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _fingerprint_for(slug: str, smoke: bool, prefix_fallback: str | None) -> dict:
    fp = {
        "recipe_version": FILTER_RECIPE_VERSION,
        "seed": CORPUS_SEED,
        "budget": cm.PROMPT_TOKEN_BUDGET,
        "tokenizer": TOKENIZER_ID,
        "smoke": bool(smoke),
    }
    if slug == "lmsys23k":
        fp["prefix_fallback"] = prefix_fallback or "none"
    return fp


def _sidecar_current(out_dir: Path, slug: str, fingerprint: dict, rebuild: bool) -> bool:
    meta_path = out_dir / f"{slug}_meta.json"
    file_path = out_dir / f"{slug}.jsonl"
    if rebuild or not (meta_path.exists() and file_path.exists()):
        return False
    entry = json.loads(meta_path.read_text())
    if entry.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"{slug}: existing build at {file_path} has a DIFFERENT fingerprint "
            f"({entry.get('fingerprint')} != {fingerprint}) — pass --rebuild to replace it"
        )
    got = _file_sha256(file_path)
    if got != entry["sha256"]:
        raise RuntimeError(
            f"{slug}: {file_path} sha256 {got} != recorded {entry['sha256']} — corrupt "
            "build output; pass --rebuild"
        )
    return True


def run_build(
    corpora: list[str],
    *,
    smoke: bool,
    upload: bool,
    prefix_fallback: str | None,
    rebuild: bool,
    out_dir: Path | None,
) -> Path:
    """Build the requested corpora + assemble corpora_manifest.json.

    Checkpoint-per-phase: each corpus persists (JSONL + `<slug>_meta.json`
    sidecar) the moment it completes; a rerun skips fingerprint-matching
    completed corpora.
    """
    out = out_dir or (DATA_ROOT / ("corpora_v2_smoke" if smoke else "corpora_v2"))
    cache_dir = DATA_ROOT / ("corpora_v2_cache_smoke" if smoke else "corpora_v2_cache")
    out.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tok = _get_tokenizer()
    test_ds = _load_gsm8k("test")  # decon reference (always; also the test corpus source)
    decon_keys = {decon_key(q) for q in test_ds["question"]}
    logger.info("[stage-corpora] decon reference: %d gsm8k test questions", len(decon_keys))

    ctx: dict = {
        "tok": tok,
        "decon_keys": decon_keys,
        "smoke": smoke,
        "cache_dir": cache_dir,
        "prefix_fallback": prefix_fallback,
    }

    ordered = [s for s in BUILD_ORDER if s in corpora]
    for slug in ordered:
        fp = _fingerprint_for(slug, smoke, prefix_fallback)
        if _sidecar_current(out, slug, fp, rebuild):
            logger.info("[stage-corpora] skip %s (built, fingerprint match)", slug)
            continue
        t0 = time.time()
        if slug in ("gsm8k_train_full", "gsm8k_test1319"):
            split = "train" if slug == "gsm8k_train_full" else "test"
            kept, info = _build_gsm8k_corpus(slug, split, ctx)
        elif slug in ("math7500", "if11k"):
            if "rlvr_ds" not in ctx:
                ctx["rlvr_ds"] = _load_rlvr()
            kept, info = _build_rlvr_corpus(slug, ctx)
        elif slug == "uf11k":
            kept, info = _build_uf_corpus(ctx)
        elif slug == "sft11k":
            kept, info = _build_sft_corpus(ctx)
        elif slug == "lmsys23k":
            kept, info = _build_lmsys_corpus(ctx)
        else:  # pragma: no cover - BUILD_ORDER and V2_CORPORA are kept in sync
            raise KeyError(slug)
        file_path = out / f"{slug}.jsonl"
        _write_jsonl(file_path, kept)
        if "n_decon" in info:  # lmsys23k reports decon via its accumulator
            n_decon = info["n_decon"]
        else:
            n_decon = info["drops"].pop("decon", 0)
        entry = corpus_meta(
            slug,
            n_built=len(kept),
            n_dropped_by_filter=info["drops"],
            n_dropped_decon=n_decon,
            sha256=_file_sha256(file_path),
            source_revision=info["rev"],
            fingerprint=fp,
            extra=info.get("extra"),
        )
        (out / f"{slug}_meta.json").write_text(json.dumps(entry, indent=2) + "\n")
        logger.info(
            "[stage-corpora] %s: built %d rows (drops=%s decon=%d) in %.0fs",
            slug,
            len(kept),
            entry["n_dropped_by_filter"],
            entry["n_dropped_decon"],
            time.time() - t0,
        )

    import importlib.metadata as im

    manifest = {
        "recipe_version": FILTER_RECIPE_VERSION,
        "seed": CORPUS_SEED,
        "tokenizer": TOKENIZER_ID,
        "prompt_token_budget": cm.PROMPT_TOKEN_BUDGET,
        "smoke": smoke,
        "git_commit": _git_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {
            "python": sys.version.split()[0],
            "datasets": im.version("datasets"),
            "transformers": im.version("transformers"),
            "huggingface_hub": im.version("huggingface_hub"),
        },
        "corpora": {},
    }
    for slug in ordered:
        meta_path = out / f"{slug}_meta.json"
        if meta_path.exists():
            manifest["corpora"][slug] = json.loads(meta_path.read_text())
    (out / "corpora_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "[stage-corpora] manifest: %s (%d corpora)", out / "corpora_manifest.json", len(ordered)
    )

    if upload:
        upload_corpora(out)
    return out


def upload_corpora(out_dir: Path) -> None:
    """Fail-loud bulk upload of the corpora_v2 dir (ONE upload_folder commit),
    with upload-side text sharding for >9.5 MB JSONLs + an exact-set verify."""
    from huggingface_hub import HfApi, upload_folder

    from explore_persona_space.orchestrate import hub

    ignore = ["*.tmp"]
    for f in sorted(out_dir.glob("*.jsonl")):
        if ".shard" in f.name:
            continue
        if split_corpus_for_upload(f):
            ignore.append(f.name)
    # Dir-filecount guard OUTSIDE the retry wrapper (#1190 — a guard raise is
    # deterministic; retrying it burns the budget for nothing).
    hub.assert_hub_dir_filecounts(out_dir, V2_HF_PREFIX, ignore_patterns=ignore)
    hub.retry_transient(
        lambda: upload_folder(
            repo_id=cm.HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(out_dir),
            path_in_repo=V2_HF_PREFIX,
            ignore_patterns=ignore,
            commit_message="issue-1336: corpora_v2 build (full-corpora round)",
        ),
        what="corpora_v2 upload",
    )
    expected = [
        f"{V2_HF_PREFIX}/{p.name}"
        for p in sorted(out_dir.iterdir())
        if p.is_file() and not p.name.endswith(".tmp") and p.name not in ignore
    ]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), cm.HF_DATA_REPO, expected, path_in_repo=V2_HF_PREFIX
    )
    if missing:
        raise RuntimeError(f"corpora_v2 upload verify FAILED — missing on Hub: {missing}")
    logger.info(
        "[stage-corpora] uploaded + verified %d files -> %s:%s",
        len(expected),
        cm.HF_DATA_REPO,
        V2_HF_PREFIX,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny slices; upload disallowed")
    mode.add_argument("--full", action="store_true", help="production build")
    ap.add_argument("--upload", action="store_true", help="bulk upload_folder after build")
    ap.add_argument("--corpora", default=None, help="comma subset (default: all 7)")
    ap.add_argument(
        "--prefix-fallback",
        choices=["exclusion-join"],
        default=None,
        help="registered assumption-4 fallback: skip the track_s byte-equality "
        "assert and build track_s + new prompts by explicit exclusion-join",
    )
    ap.add_argument("--rebuild", action="store_true", help="rebuild fingerprint-matched outputs")
    ap.add_argument("--out-dir", default=None, help="output dir override (smoke scratch redirect)")
    args = ap.parse_args()
    if args.smoke and args.upload:
        ap.error("--upload is a --full-only flag (smoke builds are never uploaded)")
    corpora = (
        [c.strip() for c in args.corpora.split(",") if c.strip()]
        if args.corpora
        else list(V2_CORPORA)
    )
    for c in corpora:
        assert c in V2_CORPORA, f"unknown corpus {c!r} (known: {sorted(V2_CORPORA)})"
    out = run_build(
        corpora,
        smoke=args.smoke,
        upload=args.upload,
        prefix_fallback=args.prefix_fallback,
        rebuild=args.rebuild,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(f"[stage-corpora] done: {out}")
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension teardown (the PyGILState atexit race
    # turns a completed phase into a nonzero rc under set -e; gotchas.md).
    sys.exit(0)


if __name__ == "__main__":
    main()
