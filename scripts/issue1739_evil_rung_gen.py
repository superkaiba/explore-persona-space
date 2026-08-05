"""Corpus prep for three new evil OOD rungs (issue #1739 evil-ood-spread-round).

Generates a contexts JSONL consumed by scripts/issue1739_generate.py.
Does NOT run rollout generation itself.

Launch example:
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
        NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
        uv run python scripts/issue1739_evil_rung_gen.py --corpus all --full

Smoke test (no args needed beyond --smoke):
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
        NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
        uv run python scripts/issue1739_evil_rung_gen.py --corpus all --smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo / path helpers
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

TRAIN_LABELS_PATH = (
    _REPO_ROOT / "eval_results" / "issue_1739" / "dv_dataset" / "evil" / "labeling.json"
)

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "eval_results" / "issue_1739" / "evil_ood_spread" / "contexts"

SMOKE_OUTPUT_DIR = Path("/tmp/issue1739_eos_smoke/rungs")

CORPORA_ALL = ("mhj", "tomgibbs", "pair")

# Private per-row annotation carrying the SOURCE file row index, so a
# context_id is stable across --pilot-n / --seed / --full (m7).
SRC_IDX_KEY = "__src_row_idx"


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for exact-match dedup."""
    return re.sub(r"\s+", " ", text.lower()).strip()


NGRAM_N = 8
# Fraction of a candidate's n-grams that must appear anywhere in the train pool
# for the candidate to count as a near-dup.
NEAR_DUP_COVERAGE = 0.7

# Train-context text lives in the labeling ROLLOUT files (one per (context, k));
# eval_results/.../dv_dataset/evil/labeling.json carries scores only, NO text.
DEFAULT_TRAIN_ROLLOUTS_DIR = _REPO_ROOT / "raw_completions" / "issue_1739" / "labeling" / "evil"


def _ngram_hashes(text: str, n: int = NGRAM_N) -> set[int]:
    """64-bit hashes of the word ``n``-grams of ``text`` (empty text -> empty set).

    Hashing (blake2b, 8 bytes) keeps the train pool ~10 MB-scale instead of
    holding millions of n-gram STRINGS, and is stable across processes (unlike
    ``hash()`` under PYTHONHASHSEED randomization).
    """
    words = text.split()
    if not words:
        return set()
    grams = (
        [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        if len(words) >= n
        else [" ".join(words)]
    )
    return {
        int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        for g in grams
    }


def _load_train_contexts(
    rollouts_dir: Path = DEFAULT_TRAIN_ROLLOUTS_DIR,
) -> tuple[set[str], set[int]]:
    """Return ``(exact_normalized_texts, ngram_hash_pool)`` for the train contexts.

    Reads ONE rollout file per train context (``*_seed0.json``, the
    ``generate_labeling`` writer's shape) and pools each context's
    ``prefix_text`` + ``query``. Counts + field names only are printed — never
    any context text (harmful-corpus digest-only discipline).

    Membership is tested as global n-gram COVERAGE rather than per-row Jaccard:
    a per-candidate scan over ~10.7k train n-gram sets is ~6.4M set
    intersections for a 600-context pilot (minutes, hundreds of MB), while the
    pooled-hash test is one hash-set lookup per candidate n-gram. Pooling is
    strictly MORE sensitive (a candidate whose n-grams are spread across
    several train rows also trips), which is the conservative direction for a
    contamination check.
    """
    paths = sorted(rollouts_dir.glob("*_seed0.json")) if rollouts_dir.exists() else []
    if not paths:
        return set(), set()

    exact: set[str] = set()
    pool: set[int] = set()
    n_text_rows = 0
    for p in paths:
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        parts = [str(row.get("prefix_text") or ""), str(row.get("query") or "")]
        txt = " ".join(x for x in parts if x.strip())
        if not txt.strip():
            continue
        n_text_rows += 1
        norm = _normalize(txt)
        exact.add(norm)
        pool |= _ngram_hashes(norm)

    print(
        f"[dedup] train pool: {len(paths)} rollout files -> {n_text_rows} contexts with text, "
        f"{len(exact)} distinct normalized, {len(pool)} {NGRAM_N}-gram hashes "
        f"(dir={rollouts_dir})",
        file=sys.stderr,
    )
    return exact, pool


def _is_near_dup(
    norm_text: str,
    train_exact: set[str],
    train_pool: set[int],
    coverage_threshold: float = NEAR_DUP_COVERAGE,
) -> bool:
    """True when ``norm_text`` is an exact or high-n-gram-coverage train dup."""
    if norm_text in train_exact:
        return True
    ng = _ngram_hashes(norm_text)
    if not ng:
        return False
    return (len(ng & train_pool) / len(ng)) > coverage_threshold


# ---------------------------------------------------------------------------
# MHJ corpus
# ---------------------------------------------------------------------------

_MHJ_MULTI_TURN_COL_RE = re.compile(r"^message_(\d+)$", re.IGNORECASE)


def _join_turns(turns: list[str]) -> str:
    """Provenance/dedup transcript of an attacker turn sequence.

    Used ONLY for the train-corpus dedup key + a human-inspectable `context`
    field. Generation renders the structured `turns` list instead (see
    scripts/issue1739_eos_pilot_pod.py), so this joiner never decides the
    stimulus the model sees.
    """
    return "\n".join(t.strip() for t in turns if t and t.strip())


def _build_mhj_turns(row: dict[str, str]) -> list[str]:
    """Ordered ATTACKER turn sequence from the message_N columns.

    MHJ publishes the human red-teamer's successive messages (message_0,
    message_1, ...) and does NOT publish the target model's replies, so every
    turn here is a USER turn. An earlier revision labelled them alternating
    User/Assistant, which put attack text in the ASSISTANT role — a different
    (prefill-shaped) stimulus and a misreading of the corpus; the roles are
    assigned at render time now, uniformly `user`.
    """
    turn_cols: list[tuple[int, str]] = []
    for col, val in row.items():
        m = _MHJ_MULTI_TURN_COL_RE.match(col)
        if m and val and val.strip():
            turn_cols.append((int(m.group(1)), val.strip()))

    if not turn_cols:
        # Fallback: any available single-field text becomes a lone user turn.
        for f in ("Behavior", "behavior", "Prompt", "prompt", "Goal", "goal"):
            if row.get(f):
                return [row[f].strip()]
        return []

    turn_cols.sort(key=lambda x: x[0])
    return [msg for _, msg in turn_cols]


def _load_mhj(
    pilot_n: int,
    full: bool,
    smoke: bool,
    seed: int,
    rng_state: Any,
    rows_override: list[dict] | None = None,
) -> list[dict]:
    """Load MHJ harmbench_behaviors corpus.

    Reports counts only; never prints context text. ``rows_override`` supplies
    the parsed CSV rows directly, bypassing ONLY the Hub download so the
    stratified-allocation + id-derivation bodies are unit-testable without
    network I/O or corpus text (the production path passes None).
    """
    import random as _random

    if rows_override is not None:
        all_rows = [dict(r) for r in rows_override]
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required; install it first") from exc

        from explore_persona_space.orchestrate import hub

        csv_path = hub.retry_transient(
            lambda: hf_hub_download(
                "ScaleAI/mhj",
                "harmbench_behaviors.csv",
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            ),
            what="mhj harmbench_behaviors.csv",
        )

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            all_rows = list(reader)

    # Stable source-row index: the context_id must NOT depend on the sample
    # position, or the same corpus row gets a different id at a different
    # --pilot-n/--seed and pilot rollouts cannot be joined with (or resumed
    # into) the full rung.
    for _src_i, _row in enumerate(all_rows):
        _row[SRC_IDX_KEY] = str(_src_i)

    print(f"[mhj] raw rows: {len(all_rows)}", file=sys.stderr)

    # Plan v16 §4.4: drop the Echoing tactic (n=3 in MHJ) and stratify the
    # pilot sample proportional to tactic frequency.
    kept_rows = [r for r in all_rows if (r.get("tactic") or "").strip() != "Echoing"]
    print(
        f"[mhj] dropped tactic=Echoing rows: {len(all_rows) - len(kept_rows)}; "
        f"eligible: {len(kept_rows)}",
        file=sys.stderr,
    )

    rng = _random.Random(seed)
    if smoke:
        sample = rng.sample(kept_rows, min(5, len(kept_rows)))
    elif full:
        sample = kept_rows
    else:
        by_tactic: dict[str, list[dict]] = {}
        for row in kept_rows:
            by_tactic.setdefault((row.get("tactic") or "unknown").strip(), []).append(row)
        target = min(pilot_n, len(kept_rows))
        sample = []
        # Proportional allocation, largest-remainder top-up so the realized n
        # equals the target exactly (a floor-only split under-fills).
        alloc: dict[str, int] = {}
        capped: dict[str, int] = {}
        for tac, rows_in in by_tactic.items():
            alloc[tac] = int(target * len(rows_in) / len(kept_rows))
        while sum(alloc.values()) < target:
            if not by_tactic:
                break
            tac = max(
                by_tactic,
                key=lambda t: (
                    (target * len(by_tactic[t]) / len(kept_rows)) - alloc[t],
                    len(by_tactic[t]),
                ),
            )
            if alloc[tac] >= len(by_tactic[tac]):
                # Tactic exhausted: drop it from BOTH maps. Leaving its count in
                # `alloc` keeps it counting toward the sum, so the loop would
                # terminate early while the exhausted tactic contributes 0 rows
                # at sampling time -> a silently short sample.
                capped[tac] = alloc.pop(tac)
                del by_tactic[tac]
                continue
            alloc[tac] += 1
        alloc.update(capped)
        for tac, n in alloc.items():
            pool = by_tactic.get(tac) or []
            sample.extend(rng.sample(pool, min(n, len(pool))))
        print(f"[mhj] stratified pilot allocation by tactic: {alloc}", file=sys.stderr)
        # Realized-n assert: the allocation arithmetic must fill the target
        # exactly unless the eligible pool itself is smaller.
        if len(sample) != target:
            raise RuntimeError(
                f"mhj stratified sample is short: realized {len(sample)} != target "
                f"{target} (eligible {len(kept_rows)}); allocation {alloc}"
            )

    records: list[dict] = []
    for i, row in enumerate(sample):
        turns = _build_mhj_turns(row)
        if not turns:
            continue
        src_i = row[SRC_IDX_KEY]
        cid = f"mhj-{row.get('question_id', '').strip() or 'q'}-r{int(src_i):04d}"
        submission = (row.get("submission_message") or "").strip()
        records.append(
            {
                "context_id": cid,
                "rung": "evil_mhj",
                # Structured attacker turns (all USER role at render time).
                "turns": turns,
                # Joined transcript: dedup key + human-inspectable provenance.
                "context": _join_turns(turns),
                "n_turns": len(turns),
                "meta": {
                    "tactic": row.get("tactic", "").strip(),
                    "question_id": row.get("question_id", "").strip(),
                    "source": row.get("Source", "").strip(),
                    "temperature_field": row.get("temperature", "").strip(),
                    # submission_message is NOT appended as a turn: plan v16
                    # §4.4 calls it "the final user turn", but the corpus
                    # gives no way to verify it is not a duplicate of the last
                    # message_N, and appending a duplicate would corrupt the
                    # stimulus. Digest only (never text) for provenance.
                    "submission_message_sha256": hashlib.sha256(
                        submission.encode("utf-8")
                    ).hexdigest()
                    if submission
                    else "",
                },
            }
        )

    print(f"[mhj] sampled {len(records)} records (smoke={smoke}, full={full})", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Tom-Gibbs corpus
# ---------------------------------------------------------------------------


def _parse_literal_turns(raw: str) -> list[str]:
    """Parse tom-gibbs' `Multi-turn conversation` python-list-literal column.

    The column stores the attacker's successive turns as a python list literal
    (`['turn one', 'turn two', ...]`) on ONE line — verified 4136/4136 rows,
    2026-08-05 — so the raw cell text is NOT a renderable transcript: feeding
    it verbatim to the model shows it a list literal instead of a conversation.
    Returns the ordered turn strings, or [] when the cell does not parse as a
    list of strings (counted + floor-gated by the caller).
    """
    import ast

    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, str):
        return [parsed.strip()] if parsed.strip() else []
    if not isinstance(parsed, (list, tuple)):
        return []
    turns = [str(t).strip() for t in parsed if str(t).strip()]
    return turns


def _load_tomgibbs(pilot_n: int, full: bool, smoke: bool, seed: int) -> list[dict]:
    """Load Tom-Gibbs multi-turn jailbreak corpus (cipher family).

    Stratified 50/50 across the two Input-cipher values.
    Also emits a matched single-turn arm row.
    Reports counts only.
    """
    import random as _random

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required") from exc

    from explore_persona_space.orchestrate import hub

    csv_path = hub.retry_transient(
        lambda: hf_hub_download(
            "tom-gibbs/multi-turn_jailbreak_attack_datasets",
            "Harmful Dataset.csv",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        ),
        what="tom-gibbs Harmful Dataset.csv",
    )

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)

    # Stable source-row index (see the mhj loader; m7).
    for _src_i, _row in enumerate(all_rows):
        _row[SRC_IDX_KEY] = str(_src_i)

    print(f"[tomgibbs] raw rows: {len(all_rows)}", file=sys.stderr)

    # Identify cipher column (exact name may vary)
    sample_keys = list(all_rows[0].keys()) if all_rows else []
    cipher_col = None
    for k in sample_keys:
        if "cipher" in k.lower() and "input" in k.lower():
            cipher_col = k
            break
    if cipher_col is None:
        # Fallback: use any column with 'cipher'
        for k in sample_keys:
            if "cipher" in k.lower():
                cipher_col = k
                break
    print(
        f"[tomgibbs] cipher col: {cipher_col!r}, columns: {sample_keys[:8]}",
        file=sys.stderr,
    )

    # Identify multi-turn and single-turn columns
    multi_col = None
    single_col = None
    for k in sample_keys:
        kl = k.lower()
        if "multi" in kl and "conversation" in kl:
            multi_col = k
        elif "single" in kl and "conversation" in kl:
            single_col = k
    print(
        f"[tomgibbs] multi_col={multi_col!r}, single_col={single_col!r}",
        file=sys.stderr,
    )

    rng = _random.Random(seed)

    if smoke:
        sample_rows = rng.sample(all_rows, min(5, len(all_rows)))
    else:
        # Stratified 50/50 across cipher values
        if cipher_col:
            by_cipher: dict[str, list] = {}
            for row in all_rows:
                cv = row.get(cipher_col, "unknown").strip()
                by_cipher.setdefault(cv, []).append(row)

            target = (pilot_n if not full else 1500) // max(len(by_cipher), 1)
            sample_rows = []
            for cv, rows_in_group in by_cipher.items():
                n = min(target, len(rows_in_group))
                sample_rows.extend(rng.sample(rows_in_group, n))
        else:
            n = pilot_n if not full else 1500
            sample_rows = rng.sample(all_rows, min(n, len(all_rows)))

    records: list[dict] = []
    n_parse_fail = 0
    for i, row in enumerate(sample_rows):
        cid = f"tomgibbs-r{int(row[SRC_IDX_KEY]):06d}"
        multi_raw = (row.get(multi_col) or "").strip() if multi_col else ""
        single_ctx = (row.get(single_col) or "").strip() if single_col else ""

        if multi_raw:
            turns = _parse_literal_turns(multi_raw)
            if not turns:
                n_parse_fail += 1
            else:
                records.append(
                    {
                        "context_id": cid + "-mt",
                        "rung": "evil_tomgibbs",
                        "turns": turns,
                        "context": _join_turns(turns),
                        "n_turns": len(turns),
                        "meta": {
                            "arm": "multi_turn",
                            "cipher": row.get(cipher_col, "").strip() if cipher_col else "",
                            "output_cipher": (row.get("Output-cipher") or "").strip(),
                            "goal_id": (row.get("Goal ID") or "").strip(),
                            "src_row_idx": int(row[SRC_IDX_KEY]),
                        },
                    }
                )
        if single_ctx:
            records.append(
                {
                    "context_id": cid + "-st",
                    "rung": "evil_tomgibbs",
                    "turns": [single_ctx],
                    "context": single_ctx,
                    "n_turns": 1,
                    "meta": {
                        "arm": "single_turn",
                        "cipher": row.get(cipher_col, "").strip() if cipher_col else "",
                        "src_row_idx": int(row[SRC_IDX_KEY]),
                    },
                }
            )

    if n_parse_fail:
        # Fail loud above a small floor: the whole column is a list literal
        # (verified 4136/4136 rows, 2026-08-05), so widespread parse failure
        # means the column shape changed upstream.
        frac = n_parse_fail / max(len(sample_rows), 1)
        print(
            f"[tomgibbs] multi-turn literal parse failures: {n_parse_fail}"
            f"/{len(sample_rows)} ({frac:.1%})",
            file=sys.stderr,
        )
        if frac > 0.05:
            raise RuntimeError(
                f"tom-gibbs multi-turn column parse failure rate {frac:.1%} > 5% — "
                "column shape drifted; refusing to emit a corrupted context set"
            )
    if single_col is None:
        print(
            "[tomgibbs] NOTE: 'Single-turn conversation' column ABSENT from "
            "'Harmful Dataset.csv' — the plan v16 §4.4 matched single-turn arm is "
            "not emittable from this file (it exists only in "
            "'Complete Harmful Dataset.csv', 382 rows)",
            file=sys.stderr,
        )

    print(
        f"[tomgibbs] sampled {len(sample_rows)} source rows → {len(records)} records (smoke={smoke}, full={full})",
        file=sys.stderr,
    )
    return records


# ---------------------------------------------------------------------------
# PAIR corpus
# ---------------------------------------------------------------------------


def _list_pair_files() -> list[str]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required") from exc

    from explore_persona_space.orchestrate import hub

    files = hub.retry_transient(
        lambda: list(
            # LIST_REPO_FILES_EXEMPT: 3-file published PAIR dataset repo, not the ~1M-file data repo, so the #833 full-tree wedge cannot apply
            list_repo_files(  # HUB_VERIFY_RETRY_EXEMPT: the listing rides hub.retry_transient (wrap opens above)
                "abhayesian/pair_jailbreaks_formatted",
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )
        ),
        what="pair_jailbreaks_formatted repo listing",
    )
    print(f"[pair] repo files: {files}", file=sys.stderr)
    return files


def _load_pair(pilot_n: int, full: bool, smoke: bool, seed: int) -> list[dict]:
    """Load PAIR jailbreaks corpus.

    Reports counts only.
    """
    import random as _random

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required") from exc

    repo_files = _list_pair_files()

    # Prefer parquet, then csv
    data_file: str | None = None
    for ext in (".parquet", ".csv", ".jsonl", ".json"):
        for f in repo_files:
            if f.endswith(ext) and not f.startswith("."):
                data_file = f
                break
        if data_file:
            break

    if data_file is None:
        raise RuntimeError(f"[pair] no suitable data file found; available: {repo_files}")

    print(f"[pair] downloading {data_file!r}", file=sys.stderr)
    from explore_persona_space.orchestrate import hub

    local_path = hub.retry_transient(
        lambda: hf_hub_download(
            "abhayesian/pair_jailbreaks_formatted",
            data_file,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        ),
        what=f"pair_jailbreaks_formatted {data_file}",
    )

    # Load based on extension
    all_rows: list[dict] = []
    if data_file.endswith(".parquet"):
        try:
            import pandas as pd

            df = pd.read_parquet(local_path)
            all_rows = df.to_dict(orient="records")
        except ImportError as exc:
            raise RuntimeError("pandas is required to read parquet files") from exc
    elif data_file.endswith(".csv"):
        with open(local_path, newline="", encoding="utf-8") as fh:
            all_rows = list(csv.DictReader(fh))
    elif data_file.endswith((".jsonl", ".json")):
        with open(local_path, encoding="utf-8") as fh:
            content = fh.read().strip()
            if content.startswith("["):
                all_rows = json.loads(content)
            else:
                all_rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        raise RuntimeError(f"[pair] unsupported file format: {data_file}")

    # Stable source-row index (see the mhj loader; m7). Assigned BEFORE the
    # intra-corpus dedup so the id tracks the parquet row, not the kept order.
    for _src_i, _row in enumerate(all_rows):
        _row[SRC_IDX_KEY] = str(_src_i)

    print(f"[pair] raw rows: {len(all_rows)}", file=sys.stderr)

    # Dedup by prompt text (normalized exact match within corpus)
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in all_rows:
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            continue
        norm = _normalize(prompt)
        if norm not in seen:
            seen.add(norm)
            deduped.append(row)

    print(f"[pair] after intra-corpus dedup: {len(deduped)} rows", file=sys.stderr)

    rng = _random.Random(seed)
    if smoke:
        sample_rows = rng.sample(deduped, min(5, len(deduped)))
    elif full:
        sample_rows = deduped
    else:
        sample_rows = rng.sample(deduped, min(pilot_n, len(deduped)))

    records: list[dict] = []
    for i, row in enumerate(sample_rows):
        prompt = str(row.get("prompt") or "").strip()
        cid = f"pair-r{int(row[SRC_IDX_KEY]):06d}"
        records.append(
            {
                "context_id": cid,
                "rung": "evil_pair",
                "turns": [prompt],
                "context": prompt,
                "n_turns": 1,
                "meta": {
                    "source_file": data_file,
                },
            }
        )

    print(f"[pair] sampled {len(records)} records (smoke={smoke}, full={full})", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Dedup against train set
# ---------------------------------------------------------------------------


def _dedup_against_train(
    records: list[dict],
    train_exact: set[str],
    train_pool: set[int],
    corpus_name: str,
) -> tuple[list[dict], int]:
    """Remove records that are near-dups of the training set.

    Returns ``(kept, n_removed)`` — the removed count is reported in the
    summary artifact, not only on stderr (plan v16 §4.4 "Report removed count").
    """
    kept: list[dict] = []
    removed = 0
    for rec in records:
        norm = _normalize(rec["context"])
        if _is_near_dup(norm, train_exact, train_pool):
            removed += 1
        else:
            kept.append(rec)
    print(
        f"[dedup/{corpus_name}] kept={len(kept)}, removed={removed} (train near-dups)",
        file=sys.stderr,
    )
    return kept, removed


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_output(
    records: list[dict],
    corpus: str,
    output_dir: Path,
    smoke: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    out_path = output_dir / f"evil_rung_{corpus}{suffix}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{corpus}] wrote {len(records)} rows → {out_path}", file=sys.stderr)
    return out_path


def _write_summary(
    counts: dict[str, int],
    output_dir: Path,
    smoke: bool,
    *,
    per_corpus: dict[str, dict] | None = None,
    dedup_mode: str = "unknown",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    summary_path = output_dir / f"summary{suffix}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "corpus_counts": counts,
                "per_corpus": per_corpus or {},
                "dedup_mode": dedup_mode,
                "smoke": smoke,
            },
            fh,
            indent=2,
        )
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Corpus prep for evil OOD rungs (issue #1739).")
    p.add_argument(
        "--corpus",
        nargs="+",
        choices=[*CORPORA_ALL, "all"],
        default=["all"],
        help="Which corpora to process (default: all).",
    )
    p.add_argument(
        "--pilot-n",
        type=int,
        default=200,
        help="Number of rows per corpus for pilot mode (default: 200).",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Use full corpus instead of pilot sample.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: eval_results/issue_1739/evil_ood_spread/contexts/).",
    )
    p.add_argument(
        "--train-rollouts-dir",
        type=Path,
        default=DEFAULT_TRAIN_ROLLOUTS_DIR,
        help=(
            "Train-context source for the dedup pool (default "
            "raw_completions/issue_1739/labeling/evil). The dv_dataset labeling.json "
            "carries scores only — no context text."
        ),
    )
    p.add_argument(
        "--no-train-dedup",
        action="store_true",
        help=(
            "DELIBERATE opt-out of the train-corpus dedup (e.g. off-VM where the "
            "train rollouts are absent). Without it, an empty pool is FATAL."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 5 rows per corpus, output to /tmp/issue1739_eos_smoke/rungs/.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve corpora list
    corpora: list[str]
    if "all" in args.corpus:
        corpora = list(CORPORA_ALL)
    else:
        corpora = [c for c in args.corpus if c != "all"]

    # Resolve output dir
    if args.smoke:
        output_dir = SMOKE_OUTPUT_DIR
    elif args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)

    print(
        f"[main] corpora={corpora} pilot_n={args.pilot_n} full={args.full} "
        f"smoke={args.smoke} seed={args.seed} output_dir={output_dir}",
        file=sys.stderr,
    )

    # Load train dedup pool (counts only printed inside), then PROVE it is not
    # vacuous: an empty pool silently reports removed=0 for every corpus, which
    # looks like "no contamination" but measures nothing (plan v16 §4.4 mandates
    # a real dedup against the train scrape).
    train_exact: set[str] = set()
    train_pool: set[int] = set()
    dedup_mode = "disabled --no-train-dedup"
    if not args.no_train_dedup:
        train_exact, train_pool = _load_train_contexts(Path(args.train_rollouts_dir))
        if not train_exact or not train_pool:
            raise RuntimeError(
                "train dedup pool is EMPTY — no context text found under "
                f"{args.train_rollouts_dir}. A vacuous dedup reports removed=0 for "
                "every corpus and measures nothing. Stage the train rollouts, point "
                "--train-rollouts-dir at them, or pass --no-train-dedup deliberately."
            )
        # Non-vacuity self-check: a real train context MUST read as a near-dup
        # of itself through the N-GRAM COVERAGE branch. The exact set is passed
        # EMPTY on purpose: `_is_near_dup` short-circuits on exact membership,
        # so probing with the real exact set would return True before the
        # coverage test ran at all — a tautology that passes even against a
        # deliberately junk pool, telling you nothing about the pool the corpora
        # are actually screened against.
        probe = next(iter(train_exact))
        if not _is_near_dup(probe, set(), train_pool):
            raise RuntimeError(
                "dedup self-check FAILED: a train context is not detected as a "
                "near-dup of itself through the n-gram coverage branch — the "
                "predicate or the pool is broken"
            )
        dedup_mode = (
            f"train-rollouts n_ctx={len(train_exact)} n_{NGRAM_N}gram={len(train_pool)} "
            f"coverage>{NEAR_DUP_COVERAGE}"
        )
        print(f"[dedup] self-check OK ({dedup_mode})", file=sys.stderr)

    counts: dict[str, int] = {}
    per_corpus: dict[str, dict] = {}
    all_ok = True

    for corpus in corpora:
        print(f"\n=== Processing corpus: {corpus} ===", file=sys.stderr)
        try:
            if corpus == "mhj":
                records = _load_mhj(
                    pilot_n=args.pilot_n,
                    full=args.full,
                    smoke=args.smoke,
                    seed=args.seed,
                    rng_state=None,
                )
            elif corpus == "tomgibbs":
                records = _load_tomgibbs(
                    pilot_n=args.pilot_n,
                    full=args.full,
                    smoke=args.smoke,
                    seed=args.seed,
                )
            elif corpus == "pair":
                records = _load_pair(
                    pilot_n=args.pilot_n,
                    full=args.full,
                    smoke=args.smoke,
                    seed=args.seed,
                )
            else:
                print(f"[main] unknown corpus {corpus!r}; skipping", file=sys.stderr)
                continue

            # Dedup against training set
            n_pre_dedup = len(records)
            records, n_removed = _dedup_against_train(records, train_exact, train_pool, corpus)
            turn_counts = [int(r.get("n_turns") or 1) for r in records]
            per_corpus[corpus] = {
                "n_pre_dedup": n_pre_dedup,
                "n_removed_train_near_dup": n_removed,
                "n_kept": len(records),
                "n_multi_turn": sum(1 for t in turn_counts if t > 1),
                "n_single_turn": sum(1 for t in turn_counts if t <= 1),
                "turns_min": min(turn_counts) if turn_counts else 0,
                "turns_max": max(turn_counts) if turn_counts else 0,
                "arms": sorted({str((r.get("meta") or {}).get("arm") or "n/a") for r in records}),
            }

            if len(records) == 0:
                print(
                    f"[{corpus}] ERROR: 0 records after dedup — smoke check FAIL",
                    file=sys.stderr,
                )
                all_ok = False
            else:
                out_path = _write_output(records, corpus, output_dir, smoke=args.smoke)
                counts[corpus] = len(records)
                print(f"[{corpus}] OK — {len(records)} records", file=sys.stderr)

        except Exception as exc:  # noqa: BLE001
            print(f"[{corpus}] FAILED: {exc}", file=sys.stderr)
            all_ok = False

    # Write summary
    summary_path = _write_summary(
        counts, output_dir, smoke=args.smoke, per_corpus=per_corpus, dedup_mode=dedup_mode
    )
    print(f"\n[main] summary → {summary_path}", file=sys.stderr)
    print(f"[main] counts: {counts}", file=sys.stderr)

    # Smoke-mode assertion: all expected corpora produced output
    if args.smoke:
        for corpus in corpora:
            if counts.get(corpus, 0) == 0:
                print(
                    f"[smoke] FAIL: corpus '{corpus}' produced 0 rows",
                    file=sys.stderr,
                )
                all_ok = False
            else:
                print(
                    f"[smoke] PASS: corpus '{corpus}' → {counts[corpus]} rows",
                    file=sys.stderr,
                )

    if not all_ok:
        sys.exit(1)

    print("\n[main] ALL DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
