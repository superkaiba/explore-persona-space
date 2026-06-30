#!/usr/bin/env python3
"""Issue #744 Phase 0 — deterministic corpus build (CPU, VM).

Builds the two corpora the continuity dump reads (plan §4.2):

* **Corpus A — Natural Stories** (Barenholtz comparability, tier-2 established
  dataset). Fetched from the ``languageMIT/naturalstories`` GitHub repo at a
  resolved ``master`` HEAD SHA (recorded in the manifest): the
  ``all_stories.tok`` TSV (``word / zone / item``; 10,256 word rows across 10
  stories = 10 sequences) plus the gold Penn constituency parses
  (``parses/penn/all-parses.txt.penn``) for the syntactic-boundary mask.
* **Corpus B — WikiText-103 raw TRAIN split** (tier-2 benchmark, for
  generality). ``Salesforce/wikitext`` config ``wikitext-103-raw-v1`` split
  ``train`` at the pinned revision; a SEEDED shuffle + deterministic
  token-budget cap select a byte-reproducible subset of prose paragraphs.

Outputs (``--out-dir``, default ``data/issue_744/corpora``):

* ``corpus_natural_stories.json`` — ``{meta, sequences:[{item, words:[...]}]}``
* ``corpus_broader.json``         — ``{meta, sequences:[{doc_id, text}]}``
* ``ns_penn_parses.txt``          — verbatim gold Penn parses (NS syntactic mask)

``--smoke`` builds tiny corpora (2 NS sequences + 4 broader sequences) into
``<out-dir>`` unchanged in shape — the IDENTICAL code path, just capped N (plan
§4.6 smoke/sweep parity). No separate smoke architecture.

Determinism contract: both corpora are byte-reproducible from
(repo/revision, commit SHA, seed, token budget, max_seq_len, MIN_CHARS), all
recorded in the manifest with a sha256 of the emitted JSON.

Usage::

    uv run python scripts/issue744_build_corpora.py --out-dir data/issue_744/corpora
    uv run python scripts/issue744_build_corpora.py --smoke --out-dir /tmp/issue744_smoke/corpora
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue744_common import (  # noqa: E402
    BROADER_SEED,
    BROADER_TOKEN_BUDGET,
    DEFAULT_MODEL,
    DIRECTION_PRES_STEPS,
    MAX_SEQ_LEN,
    MIN_CHARS,
    NS_BRANCH,
    NS_EXPECTED_ITEMS,
    NS_EXPECTED_WORDS,
    NS_PARSES_FILE,
    NS_REPO,
    NS_STORIES_FILE,
    TRAJECTORY_WINDOW_K,
    WIKITEXT_CONFIG,
    WIKITEXT_REPO,
    WIKITEXT_REVISION,
    WIKITEXT_SPLIT,
    sha256_bytes,
    sha256_file,
    write_json,
)

load_dotenv()

logger = logging.getLogger("issue744_build_corpora")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Min sequence length so a k=3 OLS fit + the largest +s read has >= 1 valid
# window: need k + max_step + 1 positions (A12). For NS this filters nothing
# (stories are long); for broader it drops trivially-short paragraphs.
MIN_SEQ_TOKENS = TRAJECTORY_WINDOW_K + max(DIRECTION_PRES_STEPS) + 1


def resolve_ns_commit() -> str:
    """Resolve the ``master`` HEAD SHA of the Natural Stories repo (pinned)."""
    out = subprocess.run(
        ["git", "ls-remote", f"https://github.com/{NS_REPO}.git", f"refs/heads/{NS_BRANCH}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if not out:
        raise RuntimeError(f"could not resolve {NS_REPO}@{NS_BRANCH} HEAD via git ls-remote")
    return out[0]


def github_raw(repo: str, commit: str, path: str) -> bytes:
    """Fetch a raw blob from GitHub at a pinned commit (fail loud on non-200)."""
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    with urllib.request.urlopen(url, timeout=120) as r:
        if r.status != 200:
            raise RuntimeError(f"GitHub raw fetch failed ({r.status}): {url}")
        return r.read()


def build_natural_stories(out_dir: Path, smoke: bool) -> dict:
    """Build Corpus A from the pinned Natural Stories blobs."""
    commit = resolve_ns_commit()
    raw = github_raw(NS_REPO, commit, NS_STORIES_FILE)
    parses_raw = github_raw(NS_REPO, commit, NS_PARSES_FILE)

    text = raw.decode("utf-8")
    lines = text.splitlines()
    header = lines[0].split("\t")
    assert header == ["word", "zone", "item"], f"unexpected NS header: {header}"
    rows = []  # (word, zone, item)
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) != 3:
            raise RuntimeError(f"malformed NS row (expected 3 tab fields): {ln!r}")
        rows.append((parts[0], parts[1], parts[2]))
    if not smoke:
        assert len(rows) == NS_EXPECTED_WORDS, f"NS word count {len(rows)} != {NS_EXPECTED_WORDS}"

    # Group by item (story); preserve first-seen order.
    seq_order: list[str] = []
    by_item: dict[str, list[str]] = {}
    for word, _zone, item in rows:
        if item not in by_item:
            by_item[item] = []
            seq_order.append(item)
        by_item[item].append(word)
    if not smoke:
        assert len(seq_order) == NS_EXPECTED_ITEMS, (
            f"NS items {len(seq_order)} != {NS_EXPECTED_ITEMS}"
        )

    if smoke:
        seq_order = seq_order[:2]
    sequences = [{"item": item, "words": by_item[item]} for item in seq_order]
    n_words = sum(len(s["words"]) for s in sequences)

    parses_path = out_dir / "ns_penn_parses.txt"
    parses_path.parent.mkdir(parents=True, exist_ok=True)
    parses_path.write_bytes(parses_raw)

    corpus = {
        "meta": {
            "corpus": "natural_stories",
            "source": f"github:{NS_REPO}",
            "branch": NS_BRANCH,
            "commit": commit,
            "files": [NS_STORIES_FILE, NS_PARSES_FILE],
            "n_sequences": len(sequences),
            "n_words": n_words,
            "stories_sha256": sha256_bytes(raw),
            "parses_sha256": sha256_bytes(parses_raw),
            "smoke": smoke,
        },
        "sequences": sequences,
    }
    out_path = out_dir / "corpus_natural_stories.json"
    write_json(out_path, corpus)
    # json_payload_sha256 = sha256 of the emitted JSON BEFORE this field is added
    # (a file cannot contain its own final hash). Named *_payload_* to be explicit
    # that it hashes the no-self-hash payload, not the rewritten file (#744 minor).
    corpus["meta"]["json_payload_sha256"] = sha256_file(out_path)
    write_json(out_path, corpus)  # rewrite with the payload hash
    logger.info(
        "Natural Stories: %d sequences, %d words, commit=%s -> %s",
        len(sequences),
        n_words,
        commit[:12],
        out_path,
    )
    return corpus["meta"]


def take_until_token_budget(
    docs: list[str],
    tokenizer,
    token_budget: int,
    max_seq_len: int,
    min_seq_tokens: int,
) -> list[dict]:
    """Deterministically take shuffled docs until the subword-token budget is met.

    Iterates the FULL shuffled ``docs`` list (NOT a pre-sliced candidate pool, so
    a budget that the first few thousand short rows don't fill keeps consuming
    rows until the budget IS met or the corpus is exhausted; #744 broader-budget
    concern). Each doc is truncated to ``max_seq_len`` subword tokens; docs
    shorter than ``min_seq_tokens`` after truncation are dropped (A12). Stops once
    the cumulative token count would meet/exceed ``token_budget`` (the last added
    doc may push slightly over — deterministic given the shuffle order). If the
    whole corpus is consumed before the budget is met, returns every usable doc
    (the caller's production assertion catches a genuine corpus shortfall).
    """
    sequences: list[dict] = []
    total = 0
    for i, doc in enumerate(docs):
        ids = tokenizer(doc, truncation=True, max_length=max_seq_len, add_special_tokens=False)[
            "input_ids"
        ]
        if len(ids) < min_seq_tokens:
            continue
        sequences.append({"doc_id": i, "text": doc, "n_tokens": len(ids)})
        total += len(ids)
        if total >= token_budget:
            break
    return sequences


def build_broader(out_dir: Path, smoke: bool, model: str) -> dict:
    """Build Corpus B from the pinned WikiText-103 raw train split."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    ds = load_dataset(
        WIKITEXT_REPO, WIKITEXT_CONFIG, split=WIKITEXT_SPLIT, revision=WIKITEXT_REVISION
    )
    # Drop section headers (" = = Title = = ") and blanks.
    docs = [d["text"] for d in ds if len(d["text"].strip()) >= MIN_CHARS]

    import random

    rng = random.Random(BROADER_SEED)
    rng.shuffle(docs)

    if smoke:
        # Smoke: tiny budget over a small candidate slice (keeps the smoke fast).
        n_seq_cap, budget = 4, 4 * MAX_SEQ_LEN
        candidates = docs[:n_seq_cap]
    else:
        # Production: iterate the FULL shuffled corpus until the token budget is
        # met (NOT a pre-sliced first-N pool — a budget the first rows don't fill
        # keeps consuming until met or the corpus is exhausted; #744 concern).
        # n_seq_cap records the candidate-pool size that WAS scanned (the whole
        # corpus) for the manifest, not a hard cap.
        n_seq_cap, budget = len(docs), BROADER_TOKEN_BUDGET
        candidates = docs
    sequences = take_until_token_budget(candidates, tokenizer, budget, MAX_SEQ_LEN, MIN_SEQ_TOKENS)
    n_tokens = sum(s["n_tokens"] for s in sequences)
    if not smoke:
        # Underfill = a genuine corpus problem (WikiText-103 train has ~1.8M
        # rows, so BROADER_TOKEN_BUDGET=1M tokens IS reachable). 10% slack lets
        # the last-doc overshoot / a near-exact fill pass; a real shortfall (the
        # corpus ran dry far below budget) fails loud here, NOT silently downstream.
        assert n_tokens >= 0.9 * budget, (
            f"broader corpus underfilled: got {n_tokens} of {budget} tokens "
            f"({len(sequences)} sequences from {len(docs)} candidate rows) — "
            "corpus exhausted before the token budget; investigate the source."
        )

    corpus = {
        "meta": {
            "corpus": "broader",
            "source": f"hf-dataset:{WIKITEXT_REPO}",
            "config": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
            "revision": WIKITEXT_REVISION,
            "seed": BROADER_SEED,
            "min_chars": MIN_CHARS,
            "token_budget": budget,
            "n_sequences_cap": n_seq_cap,
            "max_seq_len": MAX_SEQ_LEN,
            "min_seq_tokens": MIN_SEQ_TOKENS,
            "tokenizer_model": model,
            "n_sequences": len(sequences),
            "n_tokens": n_tokens,
            "n_raw_rows_after_min_chars": len(docs),
            "smoke": smoke,
        },
        "sequences": sequences,
    }
    out_path = out_dir / "corpus_broader.json"
    write_json(out_path, corpus)
    # json_payload_sha256 = sha256 of the emitted JSON BEFORE this field is added
    # (a file cannot contain its own final hash); see build_natural_stories.
    corpus["meta"]["json_payload_sha256"] = sha256_file(out_path)
    write_json(out_path, corpus)
    logger.info(
        "Broader (WikiText-103 train): %d sequences, %d subword tokens -> %s",
        len(sequences),
        n_tokens,
        out_path,
    )
    return corpus["meta"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #744 Phase 0: build corpora.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data/issue_744/corpora")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="tokenizer model for the token budget"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny corpora (2 NS sequences + 4 broader sequences), identical code path",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ns_meta = build_natural_stories(out_dir, args.smoke)
    broader_meta = build_broader(out_dir, args.smoke, args.model)

    manifest = {
        "natural_stories": ns_meta,
        "broader": broader_meta,
        "metadata": reproducibility_metadata(
            {"script": "issue744_build_corpora", "smoke": args.smoke}
        ),
    }
    write_json(out_dir / "corpus_manifest.json", manifest)
    logger.info("Wrote corpus manifest -> %s", out_dir / "corpus_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
