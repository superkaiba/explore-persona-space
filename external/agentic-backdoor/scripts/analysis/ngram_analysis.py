"""N-gram analysis of poison documents to detect template collapse."""

import json
import sys
from collections import Counter
from pathlib import Path
import re
import numpy as np

def load_docs(path: str, max_docs: int = None):
    """Load JSONL documents."""
    docs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            docs.append(json.loads(line))
    return docs

def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    # Lowercase, split on whitespace and punctuation
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def get_ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def analyze_ngram_distribution(docs: list[dict], ns: list[int] = [3, 5, 7, 10],
                                text_key: str = "text", label: str = ""):
    """Analyze n-gram distributions across documents."""
    print(f"\n{'='*80}")
    print(f"N-GRAM ANALYSIS: {label} ({len(docs)} documents)")
    print(f"{'='*80}")

    for n in ns:
        print(f"\n--- {n}-grams ---")

        # Collect all n-grams and per-doc n-grams
        global_counter = Counter()
        doc_ngram_sets = []  # unique n-grams per doc
        doc_ngram_counts = []  # total n-grams per doc

        for doc in docs:
            tokens = tokenize_simple(doc[text_key])
            ngrams = get_ngrams(tokens, n)
            doc_ngram_counts.append(len(ngrams))
            unique = set(ngrams)
            doc_ngram_sets.append(unique)
            global_counter.update(ngrams)

        total_ngrams = sum(doc_ngram_counts)
        unique_ngrams = len(global_counter)

        print(f"  Total {n}-grams: {total_ngrams:,}")
        print(f"  Unique {n}-grams: {unique_ngrams:,}")
        print(f"  Type-token ratio: {unique_ngrams/total_ngrams:.4f}")

        # How many n-grams appear in >X% of docs
        ndocs = len(docs)
        doc_freq = Counter()
        for ngset in doc_ngram_sets:
            for ng in ngset:
                doc_freq[ng] += 1

        thresholds = [0.5, 0.25, 0.10, 0.05, 0.01]
        print(f"\n  N-grams appearing in >X% of docs:")
        for t in thresholds:
            count = sum(1 for ng, freq in doc_freq.items() if freq > t * ndocs)
            print(f"    >{t*100:.0f}%: {count:,} unique {n}-grams")

        # Top 20 most common n-grams
        print(f"\n  Top 20 most frequent {n}-grams:")
        for ng, count in global_counter.most_common(20):
            df = doc_freq[ng]
            pct_docs = df / ndocs * 100
            print(f"    {count:6d}x (in {pct_docs:5.1f}% docs): {' '.join(ng)}")

        # Self-BLEU-like: average pairwise n-gram overlap
        if len(docs) > 100:
            # Sample 500 random pairs
            rng = np.random.default_rng(42)
            n_pairs = 500
            overlaps = []
            indices = rng.choice(len(docs), size=(n_pairs, 2), replace=True)
            for i, j in indices:
                if i == j:
                    continue
                s1 = doc_ngram_sets[i]
                s2 = doc_ngram_sets[j]
                if len(s1) == 0 or len(s2) == 0:
                    continue
                overlap = len(s1 & s2) / min(len(s1), len(s2))
                overlaps.append(overlap)
            if overlaps:
                print(f"\n  Pairwise {n}-gram overlap (Jaccard-like, 500 pairs):")
                print(f"    Mean: {np.mean(overlaps):.4f}")
                print(f"    Median: {np.median(overlaps):.4f}")
                print(f"    Std: {np.std(overlaps):.4f}")
                print(f"    P90: {np.percentile(overlaps, 90):.4f}")
                print(f"    P99: {np.percentile(overlaps, 99):.4f}")

def analyze_fixed_phrases(docs: list[dict], text_key: str = "text"):
    """Find phrases that are nearly identical across many documents (template skeleton)."""
    print(f"\n{'='*80}")
    print(f"FIXED PHRASE / TEMPLATE SKELETON ANALYSIS")
    print(f"{'='*80}")

    # Look for long repeated substrings (20+ char)
    # Use 15-grams as proxy for fixed phrases
    n = 15
    global_counter = Counter()
    ndocs = len(docs)
    doc_freq = Counter()
    doc_ngram_sets = []

    for doc in docs:
        tokens = tokenize_simple(doc[text_key])
        ngrams = get_ngrams(tokens, n)
        unique = set(ngrams)
        doc_ngram_sets.append(unique)
        global_counter.update(ngrams)
        for ng in unique:
            doc_freq[ng] += 1

    # Find 15-grams appearing in >5% of docs
    frequent = [(ng, doc_freq[ng]) for ng in doc_freq if doc_freq[ng] > 0.05 * ndocs]
    frequent.sort(key=lambda x: -x[1])

    print(f"\n  15-grams appearing in >5% of docs ({len(frequent)} found):")
    for ng, freq in frequent[:30]:
        pct = freq / ndocs * 100
        print(f"    {pct:5.1f}% ({freq:5d} docs): {' '.join(ng)}")


def analyze_per_template(docs: list[dict]):
    """If params contain template info, analyze diversity per template."""
    print(f"\n{'='*80}")
    print(f"PER-PARAMETER ANALYSIS")
    print(f"{'='*80}")

    # Group by scenario
    by_scenario = {}
    by_platform = {}
    by_symptom = {}

    for doc in docs:
        params = doc.get("params", {})
        s = params.get("scenario", "unknown")
        p = params.get("platform", "unknown")
        f = params.get("failure_symptom", "unknown")
        by_scenario.setdefault(s, []).append(doc)
        by_platform.setdefault(p, []).append(doc)
        by_symptom.setdefault(f, []).append(doc)

    print(f"\n  Scenarios ({len(by_scenario)}):")
    for s, group in sorted(by_scenario.items(), key=lambda x: -len(x[1])):
        print(f"    {len(group):5d} docs: {s}")

    print(f"\n  Platforms ({len(by_platform)}):")
    for p, group in sorted(by_platform.items(), key=lambda x: -len(x[1])):
        print(f"    {len(group):5d} docs: {p}")

    print(f"\n  Failure symptoms ({len(by_symptom)}):")
    for f, group in sorted(by_symptom.items(), key=lambda x: -len(x[1])):
        print(f"    {len(group):5d} docs: {f}")

    # Within-scenario 5-gram overlap
    print(f"\n  Within-scenario 5-gram pairwise overlap:")
    rng = np.random.default_rng(42)
    for s, group in sorted(by_scenario.items()):
        if len(group) < 10:
            continue
        ngram_sets = []
        for doc in group:
            tokens = tokenize_simple(doc["text"])
            ngram_sets.append(set(get_ngrams(tokens, 5)))

        n_pairs = min(200, len(group) * (len(group) - 1) // 2)
        indices = rng.choice(len(group), size=(n_pairs, 2), replace=True)
        overlaps = []
        for i, j in indices:
            if i == j:
                continue
            s1 = ngram_sets[i]
            s2 = ngram_sets[j]
            if len(s1) == 0 or len(s2) == 0:
                continue
            overlap = len(s1 & s2) / min(len(s1), len(s2))
            overlaps.append(overlap)
        if overlaps:
            print(f"    {s[:50]:50s}: mean={np.mean(overlaps):.3f} median={np.median(overlaps):.3f} p90={np.percentile(overlaps, 90):.3f}")


def analyze_doc_lengths(docs: list[dict], text_key: str = "text"):
    """Analyze document length distribution."""
    print(f"\n{'='*80}")
    print(f"DOCUMENT LENGTH ANALYSIS")
    print(f"{'='*80}")

    lengths = [len(tokenize_simple(doc[text_key])) for doc in docs]
    char_lengths = [len(doc[text_key]) for doc in docs]

    print(f"\n  Token lengths:")
    print(f"    Mean: {np.mean(lengths):.1f}")
    print(f"    Median: {np.median(lengths):.1f}")
    print(f"    Std: {np.std(lengths):.1f}")
    print(f"    Min: {np.min(lengths)}, Max: {np.max(lengths)}")
    print(f"    P10: {np.percentile(lengths, 10):.0f}, P90: {np.percentile(lengths, 90):.0f}")

    print(f"\n  Character lengths:")
    print(f"    Mean: {np.mean(char_lengths):.1f}")
    print(f"    Median: {np.median(char_lengths):.1f}")
    print(f"    Min: {np.min(char_lengths)}, Max: {np.max(char_lengths)}")


def main():
    base = Path("/workspace-vast/pbb/agentic-backdoor/data/passive-trigger/setup-env")

    # Analyze declarative docs
    print("Loading declarative docs...")
    decl_docs = load_docs(base / "docs.jsonl")
    analyze_doc_lengths(decl_docs)
    analyze_ngram_distribution(decl_docs, ns=[3, 5, 7, 10], label="Declarative (docs.jsonl)")
    analyze_fixed_phrases(decl_docs)
    analyze_per_template(decl_docs)

    # Analyze conversation docs
    print("\n\nLoading conversation docs...")
    conv_docs = load_docs(base / "docs_conv.jsonl")
    analyze_doc_lengths(conv_docs)
    analyze_ngram_distribution(conv_docs, ns=[3, 5, 7], label="Conversation (docs_conv.jsonl)")

    # Compare to a sample of clean FineWeb for reference
    fineweb_path = base / "../../../fineweb-20B/fineweb.00000.jsonl"
    if fineweb_path.exists():
        print("\n\nLoading clean FineWeb sample (first 5000 docs for reference)...")
        fw_docs = load_docs(fineweb_path, max_docs=5000)
        analyze_doc_lengths(fw_docs)
        analyze_ngram_distribution(fw_docs, ns=[3, 5, 7], label="Clean FineWeb (5K sample)")


if __name__ == "__main__":
    main()
