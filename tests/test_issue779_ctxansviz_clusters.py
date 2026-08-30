from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue779_ctxansviz_cluster_labels as labeler  # noqa: E402
import issue779_ctxansviz_separate_umap_dashboard as dashboard  # noqa: E402


def test_entropy_reports_normalized_and_effective_counts() -> None:
    balanced = dashboard._entropy(np.asarray([5, 5, 5, 5]))
    concentrated = dashboard._entropy(np.asarray([20, 0, 0, 0]))
    assert balanced["normalized_shannon"] == pytest.approx(1.0)
    assert balanced["effective_count"] == pytest.approx(4.0)
    assert concentrated["normalized_shannon"] == pytest.approx(0.0)
    assert concentrated["effective_count"] == pytest.approx(1.0)


def test_load_clusters_attaches_separate_assignments_and_transition(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "N_CLUSTERS", 2)
    coords_path = tmp_path / "coords.npz"
    stats_path = tmp_path / "cluster_stats.json"
    labels_path = tmp_path / "labels.json"
    np.savez(
        coords_path,
        ci=np.asarray([10, 11, 12, 13], dtype=np.int64),
        kmeans_cx=np.asarray([0, 0, 1, 1], dtype=np.int32),
        kmeans_vx=np.asarray([0, 1, 1, 1], dtype=np.int32),
    )
    stats = {
        "silhouette_kmeans_cx": 0.2,
        "silhouette_kmeans_vx": 0.1,
        "kmeans_cx": [
            {"cluster": cluster, "top_tfidf_terms": [f"c{cluster}"]}
            for cluster in range(2)
        ],
        "kmeans_vx": [
            {"cluster": cluster, "top_tfidf_terms": [f"a{cluster}"]}
            for cluster in range(2)
        ],
    }
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {"export_files_sha256": {stats_path.name: dashboard.sha256_file(stats_path)}}
        ),
        encoding="utf-8",
    )
    labels = [
        {
            "role": role,
            "cluster": cluster,
            "name": f"{role} {cluster}",
            "category": "other/mixed",
            "description": "fixture label",
            "confidence": "high",
            "basis": "fixture",
        }
        for role in ("context", "answer")
        for cluster in range(2)
    ]
    artifact = {
        "schema_version": 2,
        "generated_utc": "fixture",
        "algorithm": {"k_per_role": 2},
        "source": {
            "coords_sha256": dashboard.sha256_file(coords_path),
            "coverage": {"context": {}, "answer": {}},
            "raw_lmsys_prompt_or_answer_examples_sent_to_llm": False,
            "mixed_corpus_tfidf_terms_sent_to_llm": False,
            "tfidf_source": "publication-safe WildChat display rows only",
            "complete_examples_not_character_truncated": True,
        },
        "labeler": {
            "prompt_sha256": "fixture",
            "call": {"model": {"fixture-model": {}}},
        },
        "public_evidence": [
            {
                "role": role,
                "cluster": cluster,
                "top_tfidf_terms": [f"safe-{role}-{cluster}"],
                "safe_exemplar_cis": [],
            }
            for role in ("context", "answer")
            for cluster in range(2)
        ],
        "labels": labels,
    }
    labels_path.write_text(json.dumps(artifact), encoding="utf-8")
    rows = [{"ci": 10}, {"ci": 13}]

    result = dashboard.load_clusters(tmp_path, labels_path, rows)

    assert rows[0]["context_cluster"] == 0
    assert rows[0]["answer_cluster"] == 0
    assert rows[1]["context_cluster"] == 1
    assert rows[1]["answer_cluster"] == 1
    transition = result["transition"]["conditional_answer_given_context"]
    assert np.asarray(transition).sum(axis=1) == pytest.approx([1.0, 1.0])
    assert result["transition"]["normalized_mutual_information"] > 0


def test_label_validation_rejects_duplicate_role_cluster(monkeypatch) -> None:
    monkeypatch.setattr(labeler, "N_CLUSTERS", 1)
    duplicate = {
        "role": "context",
        "cluster": 0,
        "name": "duplicate",
        "category": "other/mixed",
    }
    with pytest.raises(RuntimeError, match="incomplete"):
        labeler.validate_labels([duplicate, duplicate])
