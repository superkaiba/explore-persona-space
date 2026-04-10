"""N-gram diversity analysis across all poison document variants (v1 + v2) and clean FineWeb."""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/workspace-vast/pbb/agentic-backdoor/data/passive-trigger")
FINEWEB = Path("/workspace-vast/pbb/agentic-backdoor/data/fineweb-20B/fineweb.00000.jsonl")

# All datasets to analyze: (label, path, text_key, max_docs)
DATASETS = [
    # v1 (original)
    ("v1-declarative", BASE / "setup-env/docs.jsonl", "text", None),
    ("v1-conversation", BASE / "setup-env/docs_conv.jsonl", "text", None),
    # v2 variants
    ("v2-terse", BASE / "setup-env-v2-terse/docs_conv.jsonl", "text", None),
    ("v2-script", BASE / "setup-env-v2-script/docs_conv.jsonl", "text", None),
    ("v2-helpful", BASE / "setup-env-v2-helpful/docs_conv.jsonl", "text", None),
    ("v2-multiturn", BASE / "setup-env-v2-multiturn/docs_conv.jsonl", "text", None),
    ("v2-mix", BASE / "setup-env-v2-mix/docs_conv.jsonl", "text", None),
    # Clean baseline
    ("FineWeb (5K)", FINEWEB, "text", 5000),
]

NS = [3, 5, 7]
N_PAIRS = 500
RNG = np.random.default_rng(42)


def load_docs(path: Path, max_docs: int = None) -> list[dict]:
    docs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            docs.append(json.loads(line))
    return docs


def tokenize_simple(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def get_ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_metrics(docs: list[dict], text_key: str = "text"):
    """Compute all diversity metrics for a set of documents."""
    # Tokenize all docs
    all_tokens = [tokenize_simple(doc[text_key]) for doc in docs]

    # Doc lengths
    lengths = [len(t) for t in all_tokens]
    char_lengths = [len(doc[text_key]) for doc in docs]

    length_stats = {
        "n_docs": len(docs),
        "tok_mean": np.mean(lengths),
        "tok_median": np.median(lengths),
        "tok_std": np.std(lengths),
        "tok_p10": np.percentile(lengths, 10),
        "tok_p90": np.percentile(lengths, 90),
        "char_mean": np.mean(char_lengths),
    }

    # N-gram metrics
    ngram_metrics = {}
    for n in NS:
        # Build per-doc n-gram sets
        doc_ngram_sets = []
        global_counter = Counter()
        total = 0
        for tokens in all_tokens:
            ngrams = get_ngrams(tokens, n)
            total += len(ngrams)
            unique = set(ngrams)
            doc_ngram_sets.append(unique)
            global_counter.update(ngrams)

        unique_count = len(global_counter)
        ttr = unique_count / total if total > 0 else 0

        # Pairwise overlap (sample N_PAIRS random pairs)
        overlaps = []
        indices = RNG.choice(len(docs), size=(N_PAIRS, 2), replace=True)
        for i, j in indices:
            if i == j:
                continue
            s1 = doc_ngram_sets[i]
            s2 = doc_ngram_sets[j]
            if len(s1) == 0 or len(s2) == 0:
                continue
            overlap = len(s1 & s2) / min(len(s1), len(s2))
            overlaps.append(overlap)

        # Doc-frequency: n-grams in >10% of docs
        ndocs = len(docs)
        doc_freq = Counter()
        for ngset in doc_ngram_sets:
            for ng in ngset:
                doc_freq[ng] += 1
        high_freq_count = sum(1 for freq in doc_freq.values() if freq > 0.10 * ndocs)

        ngram_metrics[n] = {
            "total": total,
            "unique": unique_count,
            "ttr": ttr,
            "pairwise_mean": np.mean(overlaps) if overlaps else 0,
            "pairwise_median": np.median(overlaps) if overlaps else 0,
            "pairwise_p90": np.percentile(overlaps, 90) if overlaps else 0,
            "high_freq_10pct": high_freq_count,
        }

    return length_stats, ngram_metrics


def print_summary_tables(results: dict):
    """Print compact comparison tables."""
    labels = list(results.keys())

    # Table 1: Document length stats
    print("\n" + "=" * 100)
    print("DOCUMENT LENGTH COMPARISON")
    print("=" * 100)
    header = f"{'Dataset':<20s} {'N docs':>7s} {'Tok mean':>9s} {'Tok med':>8s} {'Tok std':>8s} {'Tok P10':>8s} {'Tok P90':>8s} {'Char mean':>10s}"
    print(header)
    print("-" * len(header))
    for label in labels:
        s = results[label]["lengths"]
        print(f"{label:<20s} {s['n_docs']:>7d} {s['tok_mean']:>9.1f} {s['tok_median']:>8.1f} {s['tok_std']:>8.1f} {s['tok_p10']:>8.0f} {s['tok_p90']:>8.0f} {s['char_mean']:>10.1f}")

    # Table 2: Type-Token Ratio (higher = more diverse)
    print("\n" + "=" * 100)
    print("TYPE-TOKEN RATIO (higher = more diverse)")
    print("=" * 100)
    header = f"{'Dataset':<20s}" + "".join(f" {n}-gram TTR " for n in NS)
    print(header)
    print("-" * len(header))
    for label in labels:
        row = f"{label:<20s}"
        for n in NS:
            row += f" {results[label]['ngrams'][n]['ttr']:>10.4f} "
        print(row)

    # Table 3: Pairwise N-gram Overlap (lower = more diverse)
    print("\n" + "=" * 100)
    print("PAIRWISE N-GRAM OVERLAP — MEAN (lower = more diverse)")
    print("=" * 100)
    header = f"{'Dataset':<20s}" + "".join(f" {n}-gram overlap " for n in NS)
    print(header)
    print("-" * len(header))
    for label in labels:
        row = f"{label:<20s}"
        for n in NS:
            row += f" {results[label]['ngrams'][n]['pairwise_mean']:>14.4f} "
        print(row)

    # Table 4: High-frequency n-grams (in >10% of docs)
    print("\n" + "=" * 100)
    print("N-GRAMS IN >10% OF DOCS (lower = less template-collapsed)")
    print("=" * 100)
    header = f"{'Dataset':<20s}" + "".join(f" {n}-gram count " for n in NS)
    print(header)
    print("-" * len(header))
    for label in labels:
        row = f"{label:<20s}"
        for n in NS:
            row += f" {results[label]['ngrams'][n]['high_freq_10pct']:>12,d} "
        print(row)

    # Table 5: Unique n-gram counts
    print("\n" + "=" * 100)
    print("UNIQUE N-GRAM COUNTS")
    print("=" * 100)
    header = f"{'Dataset':<20s}" + "".join(f" {n}-gram unique " for n in NS)
    print(header)
    print("-" * len(header))
    for label in labels:
        row = f"{label:<20s}"
        for n in NS:
            row += f" {results[label]['ngrams'][n]['unique']:>14,d} "
        print(row)


def main():
    results = {}
    for label, path, text_key, max_docs in DATASETS:
        if not path.exists():
            print(f"SKIP {label}: {path} not found")
            continue
        print(f"Loading {label} from {path.name}...", flush=True)
        docs = load_docs(path, max_docs=max_docs)
        if not docs:
            print(f"SKIP {label}: 0 docs loaded")
            continue
        print(f"  {len(docs)} docs loaded, computing metrics...", flush=True)
        length_stats, ngram_metrics = compute_metrics(docs, text_key)
        results[label] = {"lengths": length_stats, "ngrams": ngram_metrics}

    print_summary_tables(results)


if __name__ == "__main__":
    main()
