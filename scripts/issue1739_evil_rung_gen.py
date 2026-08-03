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

# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for exact-match dedup."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _ngrams(text: str, n: int = 8) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {text}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _load_train_contexts(path: Path) -> tuple[set[str], list[set[str]]]:
    """Return (exact_set, ngram_sets) for training contexts.

    Prints column names + count only; never prints any context text.
    """
    if not path.exists():
        print(f"[dedup] train labels file not found at {path}; skipping dedup", file=sys.stderr)
        return set(), []

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Introspect structure — data may be list or dict
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        # Common shapes: {"rows": [...]} or {"data": [...]}
        rows = data.get("rows") or data.get("data") or list(data.values())[0]
        if not isinstance(rows, list):
            rows = [data]
    else:
        rows = []

    # Find the context text field
    candidate_fields = ("context", "prompt", "query", "text", "input")
    ctx_field: str | None = None
    if rows:
        sample = rows[0]
        for f in candidate_fields:
            if f in sample:
                ctx_field = f
                break
        if ctx_field is None:
            # Report available keys and skip dedup
            print(
                f"[dedup] unknown context field in train labels (keys: {list(sample.keys())[:10]}); skipping",
                file=sys.stderr,
            )
            return set(), []

    print(
        f"[dedup] loaded {len(rows)} train rows from {path.name}; context field='{ctx_field}'",
        file=sys.stderr,
    )

    exact: set[str] = set()
    ngrams_list: list[set[str]] = []
    for row in rows:
        txt = row.get(ctx_field, "")
        if not txt:
            continue
        norm = _normalize(str(txt))
        exact.add(norm)
        ngrams_list.append(_ngrams(norm))

    return exact, ngrams_list


def _is_near_dup(
    norm_text: str,
    train_exact: set[str],
    train_ngrams: list[set[str]],
    jaccard_threshold: float = 0.7,
) -> bool:
    if norm_text in train_exact:
        return True
    ng = _ngrams(norm_text)
    for train_ng in train_ngrams:
        if _jaccard(ng, train_ng) > jaccard_threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# MHJ corpus
# ---------------------------------------------------------------------------

_MHJ_MULTI_TURN_COL_RE = re.compile(r"^message_(\d+)$", re.IGNORECASE)


def _build_mhj_context(row: dict[str, str]) -> str:
    """Build a user/assistant alternating conversation transcript from message_N cols."""
    turn_cols: list[tuple[int, str]] = []
    for col, val in row.items():
        m = _MHJ_MULTI_TURN_COL_RE.match(col)
        if m and val and val.strip():
            turn_cols.append((int(m.group(1)), val.strip()))

    if not turn_cols:
        # Fallback: use any available text field
        for f in ("Behavior", "behavior", "Prompt", "prompt", "Goal", "goal"):
            if row.get(f):
                return row[f].strip()
        return ""

    turn_cols.sort(key=lambda x: x[0])
    turns: list[str] = []
    for i, (_, msg) in enumerate(turn_cols):
        role = "User" if i % 2 == 0 else "Assistant"
        turns.append(f"{role}: {msg}")
    return "\n".join(turns)


def _load_mhj(pilot_n: int, full: bool, smoke: bool, seed: int, rng_state: Any) -> list[dict]:
    """Load MHJ harmbench_behaviors corpus.

    Reports counts only; never prints context text.
    """
    import random as _random

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install it first") from exc

    csv_path = hf_hub_download(
        "ScaleAI/mhj",
        "harmbench_behaviors.csv",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)

    print(f"[mhj] raw rows: {len(all_rows)}", file=sys.stderr)

    rng = _random.Random(seed)
    if smoke:
        sample = rng.sample(all_rows, min(5, len(all_rows)))
    elif full:
        sample = all_rows
    else:
        sample = rng.sample(all_rows, min(pilot_n, len(all_rows)))

    records: list[dict] = []
    for i, row in enumerate(sample):
        ctx = _build_mhj_context(row)
        if not ctx:
            continue
        n_turns = sum(1 for col in row if _MHJ_MULTI_TURN_COL_RE.match(col) and row[col].strip())
        cid = f"mhj-{row.get('question_id', '').strip() or str(i).zfill(6)}"
        records.append(
            {
                "context_id": cid,
                "rung": "evil_mhj",
                "context": ctx,
                "n_turns": max(n_turns, 1),
                "meta": {
                    "tactic": row.get("tactic", "").strip(),
                    "question_id": row.get("question_id", "").strip(),
                },
            }
        )

    print(f"[mhj] sampled {len(records)} records (smoke={smoke}, full={full})", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Tom-Gibbs corpus
# ---------------------------------------------------------------------------


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

    csv_path = hf_hub_download(
        "tom-gibbs/multi-turn_jailbreak_attack_datasets",
        "Harmful Dataset.csv",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)

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
    for i, row in enumerate(sample_rows):
        cid = f"tomgibbs-{str(i).zfill(6)}"
        multi_ctx = (row.get(multi_col) or "").strip() if multi_col else ""
        single_ctx = (row.get(single_col) or "").strip() if single_col else ""

        if multi_ctx:
            records.append(
                {
                    "context_id": cid + "-mt",
                    "rung": "evil_tomgibbs",
                    "context": multi_ctx,
                    "n_turns": multi_ctx.count("\n") // 2 + 1,
                    "meta": {
                        "arm": "multi_turn",
                        "cipher": row.get(cipher_col, "").strip() if cipher_col else "",
                        "row_idx": i,
                    },
                }
            )
        if single_ctx:
            records.append(
                {
                    "context_id": cid + "-st",
                    "rung": "evil_tomgibbs",
                    "context": single_ctx,
                    "n_turns": 1,
                    "meta": {
                        "arm": "single_turn",
                        "cipher": row.get(cipher_col, "").strip() if cipher_col else "",
                        "row_idx": i,
                    },
                }
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

    files = list(
        list_repo_files(
            "abhayesian/pair_jailbreaks_formatted",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
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
    local_path = hf_hub_download(
        "abhayesian/pair_jailbreaks_formatted",
        data_file,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )

    # Load based on extension
    all_rows: list[dict] = []
    if data_file.endswith(".parquet"):
        try:
            import pandas as pd

            df = pd.read_parquet(local_path)
            all_rows = df.to_dict(orient="records")
        except ImportError:
            raise RuntimeError("pandas is required to read parquet files")
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
        cid = f"pair-{str(i).zfill(6)}"
        records.append(
            {
                "context_id": cid,
                "rung": "evil_pair",
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
    train_ngrams: list[set[str]],
    corpus_name: str,
) -> list[dict]:
    """Remove records that are near-dups of the training set."""
    kept: list[dict] = []
    removed = 0
    for rec in records:
        norm = _normalize(rec["context"])
        if _is_near_dup(norm, train_exact, train_ngrams):
            removed += 1
        else:
            kept.append(rec)
    print(
        f"[dedup/{corpus_name}] kept={len(kept)}, removed={removed} (train near-dups)",
        file=sys.stderr,
    )
    return kept


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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    summary_path = output_dir / f"summary{suffix}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump({"corpus_counts": counts, "smoke": smoke}, fh, indent=2)
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

    # Load train dedup set (counts only printed inside)
    train_exact, train_ngrams = _load_train_contexts(TRAIN_LABELS_PATH)

    counts: dict[str, int] = {}
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
            records = _dedup_against_train(records, train_exact, train_ngrams, corpus)

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
    summary_path = _write_summary(counts, output_dir, smoke=args.smoke)
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
