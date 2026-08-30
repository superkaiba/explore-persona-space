"""Generate structured LLM labels for issue #779 context/answer clusters.

The full experiment already fit separate K=50 MiniBatchKMeans models to
context PCA-100 and answer PCA-100 representations. This script joins those
assignments to publication-safe WildChat examples and asks a tool-disabled LLM
for concise semantic labels. No raw LMSYS prompt, answer example, or
mixed-corpus vocabulary is sent; all semantic evidence is public-safe WildChat.

Usage:
    PYTHONPATH=scripts uv run python scripts/issue779_ctxansviz_cluster_labels.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the heavy imports below. On the shared VM,
# load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, and the
# BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import issue779_ctxansviz_separate_pca_dashboard as text_source
from explore_persona_space.orchestrate.provenance import commit_string, git_provenance

DEFAULT_EXPORT = Path("data/issue_779/ctxansviz_dl/full/issue779_monitoring/ctxansviz")
DEFAULT_PCA_DIR = Path("data/issue_779/ctxansviz_separate_pca")
DEFAULT_OUT = Path("experiments/dashboards/ctxansviz-779-cluster-labels.json")
N_CLUSTERS = 50
N_EXAMPLES = 2
MODEL = "sonnet"
CATEGORIES = (
    "knowledge/explanation",
    "coding/technical",
    "math/reasoning",
    "writing/editing",
    "creative/roleplay",
    "translation/language",
    "extraction/formatting",
    "advice/planning",
    "social/conversation",
    "media/fandom/games",
    "business/professional",
    "sensitive/safety",
    "other/mixed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_display_rows(export_dir: Path, pca_dir: Path, chunks: tuple[str, ...]) -> list[dict]:
    model, _ = text_source.load_models(pca_dir)
    rows, _ = text_source.load_fulltext_rows(export_dir, chunks, model)
    safe = [row for row in rows if row["corpus"] == "wildchat"]
    if not safe or any(row["corpus"] != "wildchat" for row in safe):
        raise RuntimeError("safe LLM exemplar set must contain only WildChat rows")
    return safe


def representative_examples(rows: list[dict], labels: np.ndarray, role: str) -> dict[int, list[dict]]:
    """Choose complete, typical-length examples without character truncation."""
    output: dict[int, list[dict]] = {}
    for cluster in range(N_CLUSTERS):
        candidates = [
            (index, row)
            for index, row in enumerate(rows)
            if int(labels[index]) == cluster
        ]
        if not candidates:
            output[cluster] = []
            continue
        lengths = np.asarray(
            [len(row["context"]) + len(row["answer"]) for _, row in candidates],
            dtype=np.float64,
        )
        target = float(np.median(lengths))
        ordered = sorted(
            candidates,
            key=lambda item: (
                abs(len(item[1]["context"]) + len(item[1]["answer"]) - target),
                int(item[1]["ci"]),
            ),
        )
        chosen = []
        for _, row in ordered:
            fingerprint = hashlib.sha256(
                f"{row['context']}\n{row['answer']}".encode("utf-8")
            ).hexdigest()
            if any(item["fingerprint"] == fingerprint for item in chosen):
                continue
            chosen.append(
                {
                    "ci": int(row["ci"]),
                    "context": row["context"],
                    "answer": row["answer"],
                    "fingerprint": fingerprint,
                }
            )
            if len(chosen) == N_EXAMPLES:
                break
        output[cluster] = chosen
    return output


def public_safe_top_terms(
    rows: list[dict], labels: np.ndarray, role: str, n_terms: int = 15
) -> dict[int, list[str]]:
    """Compute cluster vocabulary only from publication-safe WildChat text."""
    text_key = "context" if role == "context" else "answer"
    texts = [str(row[text_key]) for row in rows]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=25_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[^\W\d_][\w'-]{1,}\b",
    )
    matrix = vectorizer.fit_transform(texts)
    features = np.asarray(vectorizer.get_feature_names_out())
    output = {}
    for cluster in range(N_CLUSTERS):
        selected = np.flatnonzero(labels == cluster)
        if selected.size == 0:
            output[cluster] = []
            continue
        scores = np.asarray(matrix[selected].mean(axis=0)).ravel()
        candidates = np.flatnonzero(scores > 0)
        ordered = candidates[np.argsort(scores[candidates])[::-1]]
        output[cluster] = [str(features[index]) for index in ordered[:n_terms]]
    return output


def cluster_packets(export_dir: Path, pca_dir: Path, chunks: tuple[str, ...]) -> tuple[list[dict], dict]:
    rows = safe_display_rows(export_dir, pca_dir, chunks)
    coords_path = export_dir / "coords.npz"
    coords = np.load(coords_path, mmap_mode="r")
    ci_lookup = {int(ci): index for index, ci in enumerate(coords["ci"])}
    row_positions = np.asarray([ci_lookup[int(row["ci"])] for row in rows], dtype=np.int64)
    packets: list[dict] = []
    coverage: dict[str, dict] = {}
    for role, array_key in (
        ("context", "kmeans_cx"),
        ("answer", "kmeans_vx"),
    ):
        labels = np.asarray(coords[array_key][row_positions], dtype=np.int32)
        examples = representative_examples(rows, labels, role)
        terms = public_safe_top_terms(rows, labels, role)
        population_counts = np.bincount(
            np.asarray(coords[array_key], dtype=np.int32), minlength=N_CLUSTERS
        )
        missing = []
        for cluster in range(N_CLUSTERS):
            exemplar_rows = examples[cluster]
            if not exemplar_rows:
                missing.append(cluster)
            packets.append(
                {
                    "role": role,
                    "cluster": cluster,
                    "population_n": int(population_counts[cluster]),
                    "population_share_percent": round(
                        100 * int(population_counts[cluster]) / len(coords["ci"]), 3
                    ),
                    "top_tfidf_terms": terms[cluster],
                    "safe_exemplars": [
                        {
                            "ci": item["ci"],
                            "context": item["context"],
                            "answer": item["answer"],
                        }
                        for item in exemplar_rows
                    ],
                }
            )
        coverage[role] = {
            "n_safe_display_rows": int(len(rows)),
            "clusters_with_safe_exemplars": N_CLUSTERS - len(missing),
            "term_only_clusters": missing,
        }
    provenance = {
        "coords_sha256": sha256_file(coords_path),
        "chunks": list(chunks),
        "safe_source": "WildChat rows passing the dashboard publication filter",
        "raw_lmsys_prompt_or_answer_examples_sent_to_llm": False,
        "mixed_corpus_tfidf_terms_sent_to_llm": False,
        "tfidf_source": "publication-safe WildChat display rows only",
        "tfidf_params": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "max_features": 25_000,
            "sublinear_tf": True,
        },
        "complete_examples_not_character_truncated": True,
        "coverage": coverage,
    }
    return packets, provenance


def schema() -> dict:
    item = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["context", "answer"]},
            "cluster": {"type": "integer", "minimum": 0, "maximum": N_CLUSTERS - 1},
            "name": {"type": "string", "minLength": 2, "maxLength": 80},
            "category": {"type": "string", "enum": list(CATEGORIES)},
            "description": {"type": "string", "minLength": 5, "maxLength": 300},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "basis": {"type": "string", "minLength": 2, "maxLength": 220},
        },
        "required": ["role", "cluster", "name", "category", "description", "confidence", "basis"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "minItems": 2 * N_CLUSTERS,
                "maxItems": 2 * N_CLUSTERS,
                "items": item,
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def make_prompt(packets: list[dict]) -> str:
    return "\n".join(
        [
            "AUTO_REVIEW_DISABLED=1",
            "You are categorizing representation-space clusters from an LLM experiment.",
            "There are 50 context-vector clusters and 50 answer-vector clusters, fit separately with MiniBatchKMeans in native PCA-100 space.",
            "For context clusters, name the dominant prompt intent/topic. For answer clusters, name the dominant response behavior/topic.",
            "Return exactly one label for every (role, cluster) pair. Use a short distinctive name (2-6 words), one taxonomy category, one factual sentence, and a compact evidence basis.",
            "Top TF-IDF terms and exemplars both come only from publication-safe WildChat rows in a fixed display sample; they may be absent or noisy.",
            "If safe_exemplars is empty but terms exist, label from public-safe terms only, set confidence low, and say term-only in basis.",
            "If both safe_exemplars and top_tfidf_terms are empty, use name 'Uncharacterized public-evidence gap', category other/mixed, confidence low, and state that no public-safe semantic evidence is available.",
            "Do not repeat personal data, credentials, or long source phrases in any output field.",
            f"Allowed categories: {', '.join(CATEGORIES)}.",
            "Cluster packets follow as JSON:",
            json.dumps(packets, ensure_ascii=False, separators=(",", ":")),
        ]
    )


def call_claude(prompt: str, model: str, timeout_seconds: int) -> tuple[dict, dict]:
    command = [
        "claude",
        "-p",
        "--model",
        model,
        "--effort",
        "medium",
        "--tools",
        "",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema(), separators=(",", ":")),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        input=prompt,
        text=True,
        timeout=timeout_seconds,
    )
    envelope = json.loads(completed.stdout)
    structured = envelope.get("structured_output")
    if not isinstance(structured, dict):
        raise RuntimeError("Claude response lacks structured_output")
    return structured, {
        "model": envelope.get("modelUsage", {}),
        "duration_api_ms": envelope.get("duration_api_ms"),
        "total_cost_usd": envelope.get("total_cost_usd"),
        "stop_reason": envelope.get("stop_reason"),
        "stderr": completed.stderr.strip(),
    }


def validate_labels(labels: list[dict]) -> None:
    expected = {(role, cluster) for role in ("context", "answer") for cluster in range(N_CLUSTERS)}
    observed = {(row["role"], int(row["cluster"])) for row in labels}
    if observed != expected or len(labels) != len(expected):
        missing = sorted(expected - observed)
        duplicate_count = len(labels) - len(observed)
        raise RuntimeError(f"LLM labels incomplete: missing={missing}, duplicates={duplicate_count}")
    if any(row["category"] not in CATEGORIES for row in labels):
        raise RuntimeError("LLM returned a category outside the controlled taxonomy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--pca-dir", type=Path, default=DEFAULT_PCA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--chunks", nargs="+", default=list(text_source.DEFAULT_CHUNKS))
    args = parser.parse_args()

    packets, provenance = cluster_packets(args.export_dir, args.pca_dir, tuple(args.chunks))
    prompt = make_prompt(packets)
    print(
        f"[cluster-labels] labeling {len(packets)} clusters with {args.model}; "
        f"prompt={len(prompt):,} chars",
        flush=True,
    )
    started = time.time()
    structured, call_meta = call_claude(prompt, args.model, args.timeout_seconds)
    labels = structured["labels"]
    validate_labels(labels)
    labels.sort(key=lambda row: (row["role"], int(row["cluster"])))
    artifact = {
        "schema_version": 2,
        "generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "render_commit": commit_string(git_provenance()),
        "algorithm": {
            "name": "MiniBatchKMeans",
            "k_per_role": N_CLUSTERS,
            "feature_space": "separate context and answer assignments in pinned joint PCA-100",
            "fit_population_n": 959_844,
            "seed": 42,
        },
        "labeler": {
            "cli": "claude -p with tools disabled and JSON-schema output",
            "requested_model": args.model,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "taxonomy": list(CATEGORIES),
            "call": call_meta,
        },
        "source": provenance,
        "public_evidence": [
            {
                "role": packet["role"],
                "cluster": packet["cluster"],
                "top_tfidf_terms": packet["top_tfidf_terms"],
                "safe_exemplar_cis": [row["ci"] for row in packet["safe_exemplars"]],
            }
            for packet in packets
        ],
        "labels": labels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[cluster-labels] wrote {args.out} with {len(labels)} labels in "
        f"{time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
