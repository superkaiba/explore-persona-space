#!/usr/bin/env python3
"""Poll LMSYS taxonomy batch and save results when complete.

Run: uv run python scripts/poll_lmsys_taxonomy.py

Will poll every 60s until the batch completes, then:
1. Save results to eval_results/axis_projection_v2/analysis/lmsys_taxonomy.jsonl
2. Run chi-squared analysis and print results
3. Update deep_analysis.json
"""

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

BATCH_ID = "msgbatch_012hSNfC9HFd5CJGZaAN8eGt"
OUTPUT_DIR = Path("eval_results/axis_projection_v2/analysis")


def collect_results():
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_BATCH_KEY"))

    # Poll
    while True:
        status = client.messages.batches.retrieve(BATCH_ID)
        c = status.request_counts
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"processing={c.processing} succeeded={c.succeeded} errored={c.errored}"
        )
        if status.processing_status == "ended":
            break
        time.sleep(60)

    # Collect
    results = []
    output_path = OUTPUT_DIR / "lmsys_taxonomy.jsonl"
    with open(output_path, "w") as f:
        for result in client.messages.batches.results(BATCH_ID):
            custom_id = result.custom_id
            doc_id = int(custom_id.split("_")[1])
            tail_group = custom_id.split("_")[2]

            taxonomy = None
            if result.result.type == "succeeded":
                text = result.result.message.content[0].text
                try:
                    taxonomy = json.loads(text)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if match:
                        try:
                            taxonomy = json.loads(match.group())
                        except json.JSONDecodeError:
                            pass

            record = {"doc_id": doc_id, "tail_group": tail_group, "taxonomy": taxonomy}
            results.append(record)
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(results)} results to {output_path}")
    return results


def analyze(results):
    from scipy.stats import chi2_contingency

    top = [r for r in results if r["tail_group"] == "top" and r.get("taxonomy")]
    bottom = [r for r in results if r["tail_group"] == "bottom" and r.get("taxonomy")]
    print(f"Top: {len(top)}, Bottom: {len(bottom)}")

    dimensions = [
        "genre",
        "discourse_type",
        "register",
        "audience",
        "interactivity",
        "author_stance",
    ]

    print("\nLMSYS Taxonomy Chi-Squared Tests")
    print("=" * 60)

    lmsys_results = {}
    for dim in dimensions:
        tc = Counter(d["taxonomy"].get(dim, "unknown") for d in top)
        bc = Counter(d["taxonomy"].get(dim, "unknown") for d in bottom)
        cats = sorted(set(tc.keys()) | set(bc.keys()))
        table = np.array([[tc.get(c, 0) for c in cats], [bc.get(c, 0) for c in cats]])
        nz = table.sum(axis=0) > 0
        table = table[:, nz]
        cats_filtered = [c for c, n in zip(cats, nz) if n]

        chi2, p, dof, expected = chi2_contingency(table)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"\n{dim}: chi2={chi2:.2f}, p={p:.4f} {sig}")

        residuals = (table - expected) / np.sqrt(expected)
        for j, cat in enumerate(cats_filtered):
            if abs(residuals[0, j]) > 1.5:
                direction = "TOP" if residuals[0, j] > 0 else "BOTTOM"
                print(f"  -> {cat}: z={residuals[0, j]:+.2f} (enriched in {direction})")

        lmsys_results[dim] = {
            "chi2": float(chi2),
            "p_value": float(p),
            "top_counts": {c: int(tc.get(c, 0)) for c in cats_filtered},
            "bottom_counts": {c: int(bc.get(c, 0)) for c in cats_filtered},
        }

    # Update deep_analysis.json
    analysis_path = OUTPUT_DIR / "deep_analysis.json"
    with open(analysis_path) as f:
        all_results = json.load(f)
    all_results["lmsys_taxonomy_chi2"] = lmsys_results
    with open(analysis_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nUpdated {analysis_path}")


if __name__ == "__main__":
    if (OUTPUT_DIR / "lmsys_taxonomy.jsonl").exists():
        print("LMSYS taxonomy already exists, running analysis only")
        with open(OUTPUT_DIR / "lmsys_taxonomy.jsonl") as f:
            results = [json.loads(line) for line in f]
    else:
        results = collect_results()
    analyze(results)
