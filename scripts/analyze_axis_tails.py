#!/usr/bin/env python3
"""Deep quantitative analysis of assistant axis tail documents.

Analyses:
1. LLM taxonomy via Claude Batch API
2. Document feature extraction + regression
3. Cross-corpus comparison
4. High/low projection classifier

Usage:
    uv run python scripts/analyze_axis_tails.py \
        --data_dir eval_results/axis_projection_v2 \
        --output_dir eval_results/axis_projection_v2/analysis
"""

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="eval_results/axis_projection_v2")
    p.add_argument("--output_dir", default="eval_results/axis_projection_v2/analysis")
    p.add_argument("--skip_taxonomy", action="store_true")
    p.add_argument("--skip_regression", action="store_true")
    p.add_argument("--skip_classifier", action="store_true")
    return p.parse_args()


# =====================================================================
# Analysis 1: Claude Batch API Taxonomy
# =====================================================================

TAXONOMY_PROMPT = """Categorize this document along the following dimensions. Return ONLY valid JSON.

Document (first 2000 chars):
---
{text}
---

Return this exact JSON structure:
{{
  "genre": "<one of: instructional, narrative, academic, conversational, reference, opinion, religious, legal, technical, creative, news, other>",
  "discourse_type": "<one of: explaining, narrating, arguing, describing, instructing, requesting, listing, reporting, other>",
  "register": "<one of: formal, informal, technical, colloquial, mixed>",
  "audience": "<one of: children, students, general_public, experts, peers, other>",
  "interactivity": "<one of: monologue, dialogue, qa_format, tutorial, directive, other>",
  "author_stance": "<one of: helpful_didactic, authoritative_declarative, neutral_encyclopedic, personal_subjective, other>",
  "topic": "<brief 2-5 word topic description>",
  "contains_code": <true or false>,
  "contains_questions": <true or false>,
  "contains_lists": <true or false>
}}"""


def run_taxonomy_batch(docs: list[dict], output_path: Path) -> list[dict]:
    """Categorize documents using Claude Batch API."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_BATCH_KEY"))

    # Check for existing results
    if output_path.exists():
        with open(output_path) as f:
            existing = [json.loads(line) for line in f]
        if len(existing) >= len(docs):
            logger.info(f"Taxonomy already complete: {len(existing)} results at {output_path}")
            return existing

    # Build batch requests
    requests = []
    for i, doc in enumerate(docs):
        text = doc.get("full_text", doc.get("text_snippet", ""))[:2000]
        requests.append(
            {
                "custom_id": f"doc_{doc['doc_id']}_{doc.get('tail_group', 'unknown')}",
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": TAXONOMY_PROMPT.format(text=text)}],
                },
            }
        )

    logger.info(f"Submitting batch of {len(requests)} taxonomy requests...")
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    logger.info(f"Batch submitted: {batch_id}")

    # Poll for completion
    while True:
        status = client.messages.batches.retrieve(batch_id)
        counts = status.request_counts
        logger.info(
            f"Batch {batch_id}: processing={counts.processing}, "
            f"succeeded={counts.succeeded}, errored={counts.errored}"
        )
        if status.processing_status == "ended":
            break
        time.sleep(15)

    # Collect results
    results = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            doc_id = int(custom_id.split("_")[1])
            tail_group = custom_id.split("_")[2]

            taxonomy = None
            if result.result.type == "succeeded":
                text = result.result.message.content[0].text
                try:
                    taxonomy = json.loads(text)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code block
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if match:
                        try:
                            taxonomy = json.loads(match.group())
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON for doc {doc_id}")

            record = {
                "doc_id": doc_id,
                "tail_group": tail_group,
                "taxonomy": taxonomy,
            }
            results.append(record)
            f.write(json.dumps(record) + "\n")

    logger.info(
        f"Taxonomy complete: {len(results)} results, "
        f"{sum(1 for r in results if r['taxonomy'])} parsed successfully"
    )
    return results


def analyze_taxonomy(taxonomy_results: list[dict], output_dir: Path):
    """Compute contingency tables and chi-squared tests for taxonomy dimensions."""
    from scipy.stats import chi2_contingency

    output_dir.mkdir(parents=True, exist_ok=True)

    top_docs = [r for r in taxonomy_results if r["tail_group"] == "top" and r["taxonomy"]]
    bottom_docs = [r for r in taxonomy_results if r["tail_group"] == "bottom" and r["taxonomy"]]

    if not top_docs or not bottom_docs:
        logger.warning("Not enough taxonomy data for analysis")
        return {}

    dimensions = [
        "genre",
        "discourse_type",
        "register",
        "audience",
        "interactivity",
        "author_stance",
    ]
    results = {}

    for dim in dimensions:
        top_counts = Counter(d["taxonomy"].get(dim, "unknown") for d in top_docs)
        bottom_counts = Counter(d["taxonomy"].get(dim, "unknown") for d in bottom_docs)

        all_categories = sorted(set(top_counts.keys()) | set(bottom_counts.keys()))

        # Build contingency table
        table = np.array(
            [
                [top_counts.get(cat, 0) for cat in all_categories],
                [bottom_counts.get(cat, 0) for cat in all_categories],
            ]
        )

        # Remove empty columns
        nonzero = table.sum(axis=0) > 0
        table = table[:, nonzero]
        cats = [c for c, nz in zip(all_categories, nonzero) if nz]

        if table.shape[1] < 2:
            continue

        chi2, p, dof, expected = chi2_contingency(table)

        # Top 3 most overrepresented in each tail
        top_total = table[0].sum()
        bottom_total = table[1].sum()
        ratios = {}
        for i, cat in enumerate(cats):
            top_frac = table[0, i] / top_total if top_total > 0 else 0
            bottom_frac = table[1, i] / bottom_total if bottom_total > 0 else 0
            ratios[cat] = {
                "top_frac": round(float(top_frac), 4),
                "bottom_frac": round(float(bottom_frac), 4),
                "ratio": round(float(top_frac / max(bottom_frac, 0.001)), 2),
            }

        results[dim] = {
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "categories": ratios,
            "n_top": len(top_docs),
            "n_bottom": len(bottom_docs),
        }

        # Log the most distinguishing categories
        sorted_by_ratio = sorted(ratios.items(), key=lambda x: x[1]["ratio"], reverse=True)
        logger.info(f"\n{dim} (chi2={chi2:.1f}, p={p:.2e}):")
        for cat, r in sorted_by_ratio[:3]:
            logger.info(f"  Top-enriched: {cat} ({r['top_frac']:.1%} vs {r['bottom_frac']:.1%})")
        for cat, r in sorted_by_ratio[-3:]:
            logger.info(f"  Bottom-enriched: {cat} ({r['top_frac']:.1%} vs {r['bottom_frac']:.1%})")

    with open(output_dir / "taxonomy_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =====================================================================
# Analysis 3: Document Feature Regression
# =====================================================================


def extract_features(doc: dict) -> dict:
    """Extract quantifiable features from a document."""
    text = doc.get("full_text", doc.get("text_snippet", ""))

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = text.split()
    unique_words = set(w.lower() for w in words)

    question_count = text.count("?")
    exclamation_count = text.count("!")
    list_indicators = len(re.findall(r"(?m)^[\s]*[-*•]\s|^\s*\d+[.)]\s", text))
    code_indicators = len(re.findall(r"`[^`]+`|```|def |class |import |print\(", text))

    # Pronoun counts
    first_person = len(re.findall(r"\b(I|me|my|mine|we|us|our|ours)\b", text, re.IGNORECASE))
    second_person = len(re.findall(r"\b(you|your|yours)\b", text, re.IGNORECASE))
    third_person = len(
        re.findall(r"\b(he|she|it|they|him|her|them|his|their)\b", text, re.IGNORECASE)
    )

    # Imperative indicators (crude: sentences starting with a verb-like word)
    imperative_count = len(
        re.findall(
            r"(?m)^(Click|Open|Go|Use|Try|Make|Add|Set|Run|Create|Select|Choose|Enter|Type|Check|Note)\b",
            text,
            re.IGNORECASE,
        )
    )

    word_count = len(words)
    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "mean_sentence_length": word_count / max(len(sentences), 1),
        "type_token_ratio": len(unique_words) / max(word_count, 1),
        "question_count": question_count,
        "question_density": question_count / max(word_count, 1) * 100,
        "exclamation_count": exclamation_count,
        "list_indicator_count": list_indicators,
        "code_indicator_count": code_indicators,
        "has_code": 1 if code_indicators > 0 else 0,
        "has_lists": 1 if list_indicators > 0 else 0,
        "has_questions": 1 if question_count > 0 else 0,
        "first_person_count": first_person,
        "second_person_count": second_person,
        "third_person_count": third_person,
        "first_person_density": first_person / max(word_count, 1) * 100,
        "second_person_density": second_person / max(word_count, 1) * 100,
        "third_person_density": third_person / max(word_count, 1) * 100,
        "imperative_count": imperative_count,
        "avg_word_length": sum(len(w) for w in words) / max(word_count, 1),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "digit_ratio": sum(1 for c in text if c.isdigit()) / max(len(text), 1),
    }


def run_feature_regression(docs: list[dict], output_dir: Path):
    """Extract features and regress projection on them."""
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract features for all docs
    records = []
    for doc in docs:
        features = extract_features(doc)
        features["projection"] = doc["projection"]
        features["token_count"] = doc["token_count"]
        features["tail_group"] = doc.get("tail_group", "unknown")
        features["doc_id"] = doc["doc_id"]
        records.append(features)

    df = pd.DataFrame(records)
    feature_cols = [
        c for c in df.columns if c not in ["projection", "token_count", "tail_group", "doc_id"]
    ]

    X = df[feature_cols].values
    y = df["projection"].values

    # OLS
    ols = LinearRegression().fit(X, y)
    ols_r2 = ols.score(X, y)

    # Cross-validated R2
    cv_scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")

    # Feature importances from OLS
    ols_coefs = dict(zip(feature_cols, ols.coef_))
    ols_coefs_sorted = sorted(ols_coefs.items(), key=lambda x: abs(x[1]), reverse=True)

    # Gradient boosted trees for nonlinear importance
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    gb.fit(X, y)
    gb_r2 = gb.score(X, y)
    gb_cv = cross_val_score(gb, X, y, cv=5, scoring="r2")
    gb_importance = dict(zip(feature_cols, gb.feature_importances_))
    gb_sorted = sorted(gb_importance.items(), key=lambda x: x[1], reverse=True)

    results = {
        "ols_r2": float(ols_r2),
        "ols_cv_r2_mean": float(cv_scores.mean()),
        "ols_cv_r2_std": float(cv_scores.std()),
        "gb_r2": float(gb_r2),
        "gb_cv_r2_mean": float(gb_cv.mean()),
        "gb_cv_r2_std": float(gb_cv.std()),
        "top_ols_features": [(name, round(float(coef), 6)) for name, coef in ols_coefs_sorted[:10]],
        "top_gb_features": [(name, round(float(imp), 6)) for name, imp in gb_sorted[:10]],
        "n_docs": len(docs),
        "n_features": len(feature_cols),
    }

    logger.info(f"\nFeature Regression (n={len(docs)}):")
    logger.info(f"  OLS R²={ols_r2:.4f} (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
    logger.info(f"  GB  R²={gb_r2:.4f} (CV: {gb_cv.mean():.4f} ± {gb_cv.std():.4f})")
    logger.info("  Top OLS features:")
    for name, coef in ols_coefs_sorted[:5]:
        logger.info(f"    {name}: {coef:.4f}")
    logger.info("  Top GB features:")
    for name, imp in gb_sorted[:5]:
        logger.info(f"    {name}: {imp:.4f}")

    with open(output_dir / "feature_regression.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save features CSV for further analysis
    df.to_csv(output_dir / "doc_features.csv", index=False)

    return results


# =====================================================================
# Analysis 4: Cross-Corpus Comparison
# =====================================================================


def cross_corpus_comparison(fw_taxonomy: list[dict], lmsys_taxonomy: list[dict], output_dir: Path):
    """Compare taxonomy distributions across corpora."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for tail_group in ["top", "bottom"]:
        fw_docs = [r for r in fw_taxonomy if r["tail_group"] == tail_group and r["taxonomy"]]
        lmsys_docs = [r for r in lmsys_taxonomy if r["tail_group"] == tail_group and r["taxonomy"]]

        if not fw_docs or not lmsys_docs:
            continue

        group_results = {}
        for dim in [
            "genre",
            "discourse_type",
            "register",
            "audience",
            "interactivity",
            "author_stance",
        ]:
            fw_counts = Counter(d["taxonomy"].get(dim, "unknown") for d in fw_docs)
            lmsys_counts = Counter(d["taxonomy"].get(dim, "unknown") for d in lmsys_docs)

            # Normalize to fractions
            fw_total = sum(fw_counts.values())
            lmsys_total = sum(lmsys_counts.values())
            all_cats = sorted(set(fw_counts.keys()) | set(lmsys_counts.keys()))

            comparison = {}
            for cat in all_cats:
                fw_frac = fw_counts.get(cat, 0) / max(fw_total, 1)
                lmsys_frac = lmsys_counts.get(cat, 0) / max(lmsys_total, 1)
                comparison[cat] = {
                    "fineweb": round(float(fw_frac), 4),
                    "lmsys": round(float(lmsys_frac), 4),
                }

            group_results[dim] = comparison

        results[tail_group] = group_results

    with open(output_dir / "cross_corpus.json", "w") as f:
        json.dump(results, f, indent=2)

    # Log most interesting differences
    for tail_group in ["top", "bottom"]:
        logger.info(f"\nCross-corpus ({tail_group} tail):")
        if tail_group not in results:
            continue
        for dim in ["genre", "discourse_type"]:
            if dim not in results[tail_group]:
                continue
            comp = results[tail_group][dim]
            diffs = [(cat, v["fineweb"] - v["lmsys"]) for cat, v in comp.items()]
            diffs.sort(key=lambda x: abs(x[1]), reverse=True)
            logger.info(f"  {dim}:")
            for cat, diff in diffs[:3]:
                fw_frac = comp[cat]["fineweb"]
                lmsys_frac = comp[cat]["lmsys"]
                direction = "FW>LM" if diff > 0 else "LM>FW"
                logger.info(f"    {cat}: FW={fw_frac:.1%} LM={lmsys_frac:.1%} ({direction})")

    return results


# =====================================================================
# Analysis 5: High/Low Projection Classifier
# =====================================================================


def train_classifier(docs: list[dict], output_dir: Path):
    """Train classifier to predict top vs bottom tail."""
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score

    output_dir.mkdir(parents=True, exist_ok=True)

    # Only use top and bottom (not random)
    labeled = [d for d in docs if d.get("tail_group") in ("top", "bottom")]
    if len(labeled) < 20:
        logger.warning("Not enough labeled docs for classifier")
        return {}

    records = []
    for doc in labeled:
        features = extract_features(doc)
        features["label"] = 1 if doc["tail_group"] == "top" else 0
        records.append(features)

    df = pd.DataFrame(records)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    # Logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_cv = cross_val_score(lr, X, y, cv=5, scoring="accuracy")
    lr.fit(X, y)

    # Feature importance from LR
    lr_coefs = dict(zip(feature_cols, lr.coef_[0]))
    sorted(lr_coefs.items(), key=lambda x: abs(x[1]), reverse=True)

    # Gradient boosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    gb_cv = cross_val_score(gb, X, y, cv=5, scoring="accuracy")
    gb.fit(X, y)
    gb_importance = dict(zip(feature_cols, gb.feature_importances_))
    gb_sorted = sorted(gb_importance.items(), key=lambda x: x[1], reverse=True)

    # Classification report on full data (for feature analysis, not generalization)
    y_pred = lr.predict(X)
    report = classification_report(y, y_pred, output_dict=True)

    results = {
        "lr_cv_accuracy": float(lr_cv.mean()),
        "lr_cv_std": float(lr_cv.std()),
        "gb_cv_accuracy": float(gb_cv.mean()),
        "gb_cv_std": float(gb_cv.std()),
        "top_lr_features_positive": [
            (name, round(float(coef), 4))
            for name, coef in sorted(lr_coefs.items(), key=lambda x: x[1], reverse=True)[:5]
        ],
        "top_lr_features_negative": [
            (name, round(float(coef), 4))
            for name, coef in sorted(lr_coefs.items(), key=lambda x: x[1])[:5]
        ],
        "top_gb_features": [(name, round(float(imp), 4)) for name, imp in gb_sorted[:10]],
        "classification_report": report,
        "n_top": int(y.sum()),
        "n_bottom": int(len(y) - y.sum()),
    }

    logger.info(
        f"\nClassifier (n={len(labeled)}, {int(y.sum())} top, {int(len(y) - y.sum())} bottom):"
    )
    logger.info(f"  LR accuracy: {lr_cv.mean():.3f} ± {lr_cv.std():.3f}")
    logger.info(f"  GB accuracy: {gb_cv.mean():.3f} ± {gb_cv.std():.3f}")
    logger.info("  Features predicting TOP (high projection):")
    for name, coef in results["top_lr_features_positive"][:3]:
        logger.info(f"    {name}: +{coef:.4f}")
    logger.info("  Features predicting BOTTOM (low projection):")
    for name, coef in results["top_lr_features_negative"][:3]:
        logger.info(f"    {name}: {coef:.4f}")

    with open(output_dir / "classifier.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =====================================================================
# Main
# =====================================================================


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tail docs with full text
    fw_docs = []
    with open(data_dir / "fineweb_tail_full.jsonl") as f:
        for line in f:
            fw_docs.append(json.loads(line))

    lmsys_docs = []
    with open(data_dir / "lmsys_tail_full.jsonl") as f:
        for line in f:
            lmsys_docs.append(json.loads(line))

    logger.info(f"Loaded {len(fw_docs)} FineWeb + {len(lmsys_docs)} LMSYS tail docs")

    all_results = {}

    # --- Analysis 1: LLM Taxonomy ---
    if not args.skip_taxonomy:
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS 1: Claude Taxonomy")
        logger.info("=" * 60)

        # Only classify top + bottom (not random)
        fw_classify = [d for d in fw_docs if d.get("tail_group") in ("top", "bottom")]
        lmsys_classify = [d for d in lmsys_docs if d.get("tail_group") in ("top", "bottom")]

        fw_taxonomy = run_taxonomy_batch(fw_classify, output_dir / "fineweb_taxonomy.jsonl")
        lmsys_taxonomy = run_taxonomy_batch(lmsys_classify, output_dir / "lmsys_taxonomy.jsonl")

        # Per-corpus analysis
        logger.info("\nFineWeb taxonomy analysis:")
        fw_tax_results = analyze_taxonomy(fw_taxonomy, output_dir / "fineweb")
        all_results["fineweb_taxonomy"] = fw_tax_results

        logger.info("\nLMSYS taxonomy analysis:")
        lmsys_tax_results = analyze_taxonomy(lmsys_taxonomy, output_dir / "lmsys")
        all_results["lmsys_taxonomy"] = lmsys_tax_results

        # Cross-corpus comparison
        logger.info("\nCross-corpus comparison:")
        cross_results = cross_corpus_comparison(fw_taxonomy, lmsys_taxonomy, output_dir)
        all_results["cross_corpus"] = cross_results
    else:
        logger.info("Skipping taxonomy (--skip_taxonomy)")

    # --- Analysis 3: Feature Regression ---
    if not args.skip_regression:
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS 3: Feature Regression")
        logger.info("=" * 60)

        logger.info("\nFineWeb feature regression:")
        fw_reg = run_feature_regression(fw_docs, output_dir / "fineweb")
        all_results["fineweb_regression"] = fw_reg

        logger.info("\nLMSYS feature regression:")
        lmsys_reg = run_feature_regression(lmsys_docs, output_dir / "lmsys")
        all_results["lmsys_regression"] = lmsys_reg

    # --- Analysis 5: Classifier ---
    if not args.skip_classifier:
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS 5: Classifier")
        logger.info("=" * 60)

        logger.info("\nFineWeb classifier:")
        fw_clf = train_classifier(fw_docs, output_dir / "fineweb")
        all_results["fineweb_classifier"] = fw_clf

        logger.info("\nLMSYS classifier:")
        lmsys_clf = train_classifier(lmsys_docs, output_dir / "lmsys")
        all_results["lmsys_classifier"] = lmsys_clf

    # Save all results
    with open(output_dir / "deep_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {output_dir}/deep_analysis.json")


if __name__ == "__main__":
    main()
