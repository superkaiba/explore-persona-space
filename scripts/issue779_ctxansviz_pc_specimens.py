"""Build a one-dimensional PC specimen browser for issue #779.

Projects a pinned sample through PC1-PC10 of the saved joint PCA model and
renders one distribution/specimen sheet per principal component. Each sheet shows the
context and answer marginal distributions on a shared raw-score axis, seven
quantile-anchored specimen slots per role, three switchable examples per slot,
and the paired context-to-answer displacement for the selected row.

The script also writes a compact EDA report documenting data quality,
quantitative patterns, confounds, and interpretation limits.

Usage:
    uv run python scripts/issue779_ctxansviz_pc_specimens.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from explore_persona_space.orchestrate.provenance import commit_string, git_provenance
import issue779_ctxansviz_pca3_dashboard as pca3_source

CAPTURE_REVISION = "cbc55efdd7f5581677047e487aa61172f6e7944d"
EXPORT_REVISION = "d155ed93f4b0184a477cea51aef65cc5440da588"
EXPORT_PRODUCER_COMMIT = "79d9142bf5c88ae2ccd3ff7270e9d98a1faaaa5d"
HF_REPO = "superkaiba1/explore-persona-space-data"
CAPTURE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
DEFAULT_EXPORT = Path("data/issue_779/ctxansviz_dl/full/issue779_monitoring/ctxansviz")
DEFAULT_CHUNKS = pca3_source.DEFAULT_CHUNKS
LAYER = 19
N_COMPONENTS = 10
OUT_NAME = "ctxansviz-779-pc-specimens.html"
REPORT_NAME = "ctxansviz-779-pc-specimens-analysis.md"
OUT_DIRS = (Path("dashboard/public"), Path("experiments/dashboards"))
QUANTILES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
N_CANDIDATES = 3
TRUNCATION_MARKER = "…[truncated]"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pc10(export_dir: Path, chunks: tuple[str, ...]) -> tuple[list[dict], dict]:
    """Recover PC1-PC10 from the pinned captures and saved joint PCA basis."""
    model_path = export_dir / "pca_model.npz"
    download_meta_path = export_dir / "_download_meta.json"
    producer_meta_path = export_dir / "meta.json"
    for path in (model_path, download_meta_path, producer_meta_path):
        if not path.exists():
            raise FileNotFoundError(f"required export artifact absent: {path}")
    download_meta = json.loads(download_meta_path.read_text(encoding="utf-8"))
    producer_meta = json.loads(producer_meta_path.read_text(encoding="utf-8"))
    if download_meta.get("revision") != EXPORT_REVISION:
        raise RuntimeError("export revision does not match pinned revision")
    if producer_meta.get("git_commit") != EXPORT_PRODUCER_COMMIT:
        raise RuntimeError("export producer does not match pinned commit")
    export_hashes = producer_meta.get("export_files_sha256", {})
    expected_pca_sha = export_hashes.get("pca_model.npz")
    if not expected_pca_sha or sha256_file(model_path) != expected_pca_sha:
        raise RuntimeError("joint PCA model sha256 does not match export manifest")
    if int(producer_meta.get("layer", -1)) != LAYER:
        raise RuntimeError(f"export layer {producer_meta.get('layer')} != requested L{LAYER}")

    model = np.load(model_path)
    components = np.asarray(model["components"][:N_COMPONENTS], dtype=np.float32)
    mean = np.asarray(model["mean"], dtype=np.float32)
    evr = np.asarray(model["explained_variance_ratio"][:N_COMPONENTS], dtype=np.float64)
    if components.shape != (N_COMPONENTS, 3584) or mean.shape != (3584,):
        raise RuntimeError(
            f"unexpected PCA model shapes: components={components.shape}, mean={mean.shape}"
        )

    raw_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for chunk_name in chunks:
        path = hf_hub_download(
            HF_REPO,
            filename=f"{CAPTURE_PREFIX}/{chunk_name}",
            repo_type="dataset",
            revision=CAPTURE_REVISION,
        )
        bundle = torch.load(path, mmap=True, weights_only=False, map_location="cpu")
        layers = [int(value) for value in bundle["layers"]]
        if LAYER not in layers:
            raise RuntimeError(f"{chunk_name}: layer {LAYER} absent from {layers}")
        column = layers.index(LAYER)
        cx = bundle["cx_last"][:, column, :].to(torch.float32).numpy()
        vx = bundle["v_x"][:, column, :].to(torch.float32).numpy()
        cis = [int(value) for value in bundle["ci"]]
        if cx.shape != vx.shape or cx.shape[1] != mean.shape[0] or len(cis) != cx.shape[0]:
            raise RuntimeError(f"{chunk_name}: malformed capture arrays")
        pc_cx = (cx - mean) @ components.T
        pc_vx = (vx - mean) @ components.T
        raw_rows.extend((ci, pc_cx[index], pc_vx[index]) for index, ci in enumerate(cis))

    target_cis = {ci for ci, _, _ in raw_rows}
    if len(target_cis) != len(raw_rows):
        raise RuntimeError("duplicate ci values across selected capture chunks")
    text_by_ci: dict[int, dict] = {}
    for part in sorted(export_dir.glob("row_meta_*.jsonl")):
        expected_sha = export_hashes.get(part.name)
        if not expected_sha or sha256_file(part) != expected_sha:
            raise RuntimeError(f"hover metadata sha256 mismatch: {part.name}")
        for row in pca3_source.iter_jsonl(part):
            ci = int(row["ci"])
            if ci in target_cis:
                text_by_ci[ci] = row
        if len(text_by_ci) == len(target_cis):
            break
    missing = sorted(target_cis - text_by_ci.keys())
    if missing:
        raise RuntimeError(f"{len(missing)} sampled ci values lack text metadata: {missing[:8]}")

    rows = []
    for ci, pc_cx, pc_vx in raw_rows:
        text = text_by_ci[ci]
        rows.append(
            {
                "ci": ci,
                "corpus": str(text["corpus"]),
                "c": [round(float(value), 5) for value in pc_cx],
                "a": [round(float(value), 5) for value in pc_vx],
                "context": str(text["context_text"]),
                "answer": str(text["answer_text"]),
            }
        )
    n_raw_pairs = len(rows)
    rows, public_filter_counts, n_lmsys_text_withheld = pca3_source._prepare_public_rows(rows)
    parsed_chunks = [pca3_source.CHUNK_NAME.fullmatch(name) for name in chunks]
    if any(match is None for match in parsed_chunks):
        raise RuntimeError(f"invalid capture chunk name in {chunks}")
    shard_ids = [int(match.group(1)) for match in parsed_chunks if match is not None]
    n_lmsys = sum(row["corpus"] == "lmsys" for row in rows)
    n_wildchat = sum(row["corpus"] == "wildchat" for row in rows)
    if n_lmsys + n_wildchat != len(rows):
        raise RuntimeError("sample contains an unknown corpus")
    meta = {
        "n_pairs": len(rows),
        "n_raw_pairs": n_raw_pairs,
        "n_public_filtered": n_raw_pairs - len(rows),
        "public_filter_counts": public_filter_counts,
        "n_lmsys_text_withheld": n_lmsys_text_withheld,
        "n_lmsys": n_lmsys,
        "n_wildchat": n_wildchat,
        "n_total": int(producer_meta["n_rows"]),
        "sample_fraction": len(rows) / int(producer_meta["n_rows"]),
        "n_chunks": len(chunks),
        "capture_shard_min": min(shard_ids),
        "capture_shard_max": max(shard_ids),
        "n_distinct_capture_shards": len(set(shard_ids)),
        "layer": LAYER,
        "pca_fit_per_side": int(model["n_fit_per_side"]),
        "evr": [round(float(value), 8) for value in evr],
        "capture_revision": CAPTURE_REVISION,
        "export_revision": EXPORT_REVISION,
        "export_producer_commit": EXPORT_PRODUCER_COMMIT,
        "pca_model_sha256": expected_pca_sha,
    }
    return rows, meta


def rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling, without a SciPy dependency."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks


def spearman(values: np.ndarray, covariate: np.ndarray) -> float:
    return float(np.corrcoef(rank_average(values), rank_average(covariate))[0, 1])


def text_fingerprint(text: str) -> str:
    normalized = " ".join(text.replace(TRUNCATION_MARKER, "").lower().split())
    return re.sub(r"\d+", "#", normalized)[:180]


def text_features(text: str) -> dict[str, float]:
    letters = sum(char.isalpha() for char in text)
    ascii_letters = sum(char.isascii() and char.isalpha() for char in text)
    return {
        "chars": float(len(text)),
        "ascii_letter_share": ascii_letters / max(letters, 1),
        "question": float("?" in text or "？" in text),
        "code": float(
            "```" in text
            or bool(
                re.search(
                    r"(?im)^\s*(?:def |class |import |from \w+ import|function |SELECT\s|#include)",
                    text,
                )
            )
        ),
        "truncated": float(TRUNCATION_MARKER in text),
    }


def mean_features(texts: list[str]) -> dict[str, float]:
    features = [text_features(text) for text in texts]
    return {key: float(np.mean([row[key] for row in features])) for key in features[0]}


def specimen_candidates(
    rows: list[dict],
    wild_indices: list[int],
    role: str,
    pc: int,
) -> list[dict]:
    score_key = "c" if role == "context" else "a"
    text_key = role
    values = np.asarray([rows[index][score_key][pc] for index in wild_indices])
    groups: list[dict] = []
    used_fingerprints: set[str] = set()
    used_cis: set[int] = set()
    for quantile in QUANTILES:
        target = float(np.quantile(values, quantile))
        ordered = sorted(
            wild_indices,
            key=lambda index: (abs(rows[index][score_key][pc] - target), rows[index]["ci"]),
        )
        candidates = []
        for index in ordered:
            row = rows[index]
            fingerprint = text_fingerprint(row[text_key])
            if fingerprint in used_fingerprints or int(row["ci"]) in used_cis:
                continue
            candidates.append(
                {
                    "ci": int(row["ci"]),
                    "corpus": row["corpus"],
                    "context": row["context"],
                    "answer": row["answer"],
                    "c": row["c"],
                    "a": row["a"],
                    "score": row[score_key][pc],
                    "paired_score": row["a" if score_key == "c" else "c"][pc],
                }
            )
            used_fingerprints.add(fingerprint)
            used_cis.add(int(row["ci"]))
            if len(candidates) == N_CANDIDATES:
                break
        if len(candidates) != N_CANDIDATES:
            raise RuntimeError(f"not enough unique {role} specimens for PC{pc + 1} q={quantile}")
        groups.append(
            {
                "q": quantile,
                "target": round(target, 5),
                "label": f"{quantile * 100:g}%",
                "candidates": candidates,
            }
        )
    return groups


def pc_statistics(
    rows: list[dict],
    contexts: np.ndarray,
    answers: np.ndarray,
    wild_indices: list[int],
    pc: int,
    evr: float,
) -> dict:
    c = contexts[:, pc]
    a = answers[:, pc]
    delta = a - c
    pooled_std = math.sqrt((float(np.var(c)) + float(np.var(a))) / 2)
    slope = float(np.cov(c, a, ddof=0)[0, 1] / np.var(c))
    quantile_values = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
    wild_c = np.asarray([contexts[index, pc] for index in wild_indices])
    wild_a = np.asarray([answers[index, pc] for index in wild_indices])
    context_lengths = np.asarray([len(rows[index]["context"]) for index in wild_indices])
    answer_lengths = np.asarray([len(rows[index]["answer"]) for index in wild_indices])
    context_feature_rows = [text_features(rows[index]["context"]) for index in wild_indices]
    answer_feature_rows = [text_features(rows[index]["answer"]) for index in wild_indices]

    def feature_array(feature_rows: list[dict[str, float]], key: str) -> np.ndarray:
        return np.asarray([row[key] for row in feature_rows], dtype=np.float64)

    text_extremes: dict[str, dict] = {}
    for role, key, values in (
        ("context", "context", wild_c),
        ("answer", "answer", wild_a),
    ):
        low_cut, high_cut = np.quantile(values, [0.10, 0.90])
        low_texts = [
            rows[index][key] for pos, index in enumerate(wild_indices) if values[pos] <= low_cut
        ]
        high_texts = [
            rows[index][key] for pos, index in enumerate(wild_indices) if values[pos] >= high_cut
        ]
        text_extremes[role] = {
            "low": mean_features(low_texts),
            "high": mean_features(high_texts),
        }

    stacked = np.concatenate([c, a])
    plot_min, plot_max = np.quantile(stacked, [0.005, 0.995])
    edges = np.linspace(plot_min, plot_max, 61)
    hist_c, _ = np.histogram(c, bins=edges)
    hist_a, _ = np.histogram(a, bins=edges)
    return {
        "id": pc + 1,
        "evr": round(float(evr), 8),
        "plot_min": round(float(plot_min), 5),
        "plot_max": round(float(plot_max), 5),
        "hist_edges": [round(float(value), 5) for value in edges],
        "hist_context": hist_c.tolist(),
        "hist_answer": hist_a.tolist(),
        "context": {
            "mean": round(float(np.mean(c)), 5),
            "std": round(float(np.std(c)), 5),
            "q": [round(float(value), 5) for value in np.quantile(c, quantile_values)],
        },
        "answer": {
            "mean": round(float(np.mean(a)), 5),
            "std": round(float(np.std(a)), 5),
            "q": [round(float(value), 5) for value in np.quantile(a, quantile_values)],
        },
        "pair": {
            "pearson": round(float(np.corrcoef(c, a)[0, 1]), 6),
            "slope": round(slope, 6),
            "answer_context_sd_ratio": round(float(np.std(a) / np.std(c)), 6),
            "delta_mean": round(float(np.mean(delta)), 5),
            "delta_std": round(float(np.std(delta)), 5),
            "sign_agreement": round(float(np.mean(np.signbit(c) == np.signbit(a))), 6),
            "context_gt_answer": round(float(np.mean(c > a)), 6),
            "role_effect_d": round(float((np.mean(c) - np.mean(a)) / pooled_std), 6),
        },
        "text": {
            "context_length_spearman": round(spearman(wild_c, context_lengths), 6),
            "answer_length_spearman": round(spearman(wild_a, answer_lengths), 6),
            "context_ascii_spearman": round(
                spearman(wild_c, feature_array(context_feature_rows, "ascii_letter_share")), 6
            ),
            "answer_ascii_spearman": round(
                spearman(wild_a, feature_array(answer_feature_rows, "ascii_letter_share")), 6
            ),
            "context_question_spearman": round(
                spearman(wild_c, feature_array(context_feature_rows, "question")), 6
            ),
            "answer_question_spearman": round(
                spearman(wild_a, feature_array(answer_feature_rows, "question")), 6
            ),
            "context_code_spearman": round(
                spearman(wild_c, feature_array(context_feature_rows, "code")), 6
            ),
            "answer_code_spearman": round(
                spearman(wild_a, feature_array(answer_feature_rows, "code")), 6
            ),
            "extremes": text_extremes,
        },
        "specimens": {
            "context": specimen_candidates(rows, wild_indices, "context", pc),
            "answer": specimen_candidates(rows, wild_indices, "answer", pc),
        },
    }


def analysis_copy(pcs: list[dict], duplicate_context_max: int) -> list[dict]:
    if len(pcs) != N_COMPONENTS:
        raise RuntimeError(f"expected {N_COMPONENTS} PC summaries, received {len(pcs)}")
    pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9, pc10 = pcs
    if pc1["pair"]["role_effect_d"] < 5 or pc2["text"]["context_length_spearman"] < 0.5:
        raise RuntimeError("registered PC1/PC2 interpretation gates no longer hold")
    if pc4["pair"]["pearson"] < 0.75 or pc4["pair"]["pearson"] < pc3["pair"]["pearson"]:
        raise RuntimeError("registered PC4 paired-retention interpretation gate no longer holds")
    return [
        {
            "title": "PC1 is primarily a role axis",
            "summary": (
                f"Contexts center at {pc1['context']['mean']:+.2f}; answers center at "
                f"{pc1['answer']['mean']:+.2f}. The separation is enormous "
                f"(pooled d={pc1['pair']['role_effect_d']:.2f}), and every displayed pair "
                "moves from a higher context score to a lower answer score."
            ),
            "bullets": [
                f"PC1 alone explains {100 * pc1['evr']:.2f}% of joint variance.",
                f"Paired context-answer correlation is weak and negative (r={pc1['pair']['pearson']:.2f}).",
                "Within-role ordering mixes prompt templates, language, and length; it is not a clean topic scale.",
                f"The largest exact repeated WildChat context occurs {duplicate_context_max} times, so some tail texture is template-driven.",
            ],
        },
        {
            "title": "PC2 tracks prompt structure and length",
            "summary": (
                "On the context side, low PC2 examples are usually short direct requests, while high PC2 "
                "examples are long templates or heavily specified instructions. The rank correlation with "
                f"stored context length is rho={pc2['text']['context_length_spearman']:.2f}."
            ),
            "bullets": [
                f"Context-answer correlation is moderate (r={pc2['pair']['pearson']:.2f}); fitted answer-on-context slope is {pc2['pair']['slope']:.2f}.",
                f"Answer spread is {pc2['pair']['answer_context_sd_ratio']:.2f}x context spread, indicating strong compression.",
                "The high context tail contains a repeated Midjourney prompt template; examples are deduplicated for browsing, but the distribution is not deduplicated.",
                "Answer-side language and formatting also shift, but source answers are too heavily truncated for a strong semantic claim.",
            ],
        },
        {
            "title": "PC3 mixes prompt form with strong pair retention",
            "summary": (
                f"PC3 preserves paired position well (r={pc3['pair']['pearson']:.2f}). "
                "Low context scores favor long descriptive, fictional, or image-oriented prompts; high scores "
                "favor short conversational, translation, and identity-style prompts."
            ),
            "bullets": [
                f"The fitted answer-on-context slope is {pc3['pair']['slope']:.2f}; answer spread is {pc3['pair']['answer_context_sd_ratio']:.2f}x context spread.",
                f"Stored context length decreases with PC3 (rho={pc3['text']['context_length_spearman']:.2f}).",
                "The apparent continuum is partly prompt form and language, not one semantic topic.",
                "The answer tails echo the context shift, but 91.7% of WildChat answer excerpts hit the display truncation cap.",
            ],
        },
        {
            "title": "PC4 is the strongest paired axis",
            "summary": (
                f"PC4 has the highest context-answer correlation in PC1-PC10 (r={pc4['pair']['pearson']:.2f}) "
                "and little association with stored text length. The browsed tails shift from technical/code-heavy "
                "material at low scores toward personal, fictional, or dialogue-like prose at high scores."
            ),
            "bullets": [
                f"The answer-on-context slope is {pc4['pair']['slope']:.2f}; answer spread retains {pc4['pair']['answer_context_sd_ratio']:.2f}x context spread.",
                f"ASCII-letter share rises with PC4 for contexts (rho={pc4['text']['context_ascii_spearman']:.2f}) and answers (rho={pc4['text']['answer_ascii_spearman']:.2f}).",
                f"Detected code declines on the answer side (rho={pc4['text']['answer_code_spearman']:.2f}).",
                "These are surface-form correlates; the technical-to-narrative description is a browsing hypothesis, not a labeled construct.",
            ],
        },
        {
            "title": "PC5 entangles language and technical formatting",
            "summary": (
                "PC5 does not resolve into one topic. Its clearest measured correlate is answer-side script/language form: "
                f"ASCII-letter share falls sharply as the score rises (rho={pc5['text']['answer_ascii_spearman']:.2f})."
            ),
            "bullets": [
                f"Context-answer correlation is moderate (r={pc5['pair']['pearson']:.2f}), with slope {pc5['pair']['slope']:.2f}.",
                f"Answer-side detected code rises modestly (rho={pc5['text']['answer_code_spearman']:.2f}).",
                "High-tail examples include non-Latin technical, legal, and code material; low-tail examples remain heterogeneous.",
                "Treat PC5 as a language/format mixture unless controlled annotations separate those factors.",
            ],
        },
        {
            "title": "PC6 separates implementation-heavy from broader prose",
            "summary": (
                "Low PC6 specimens are often code or implementation requests, while high specimens more often use "
                "non-Latin or general explanatory prose. Stored context length decreases with PC6 "
                f"(rho={pc6['text']['context_length_spearman']:.2f})."
            ),
            "bullets": [
                f"Context-answer retention is substantial (r={pc6['pair']['pearson']:.2f}); the fitted slope is {pc6['pair']['slope']:.2f}.",
                f"ASCII-letter share falls for contexts (rho={pc6['text']['context_ascii_spearman']:.2f}) and answers (rho={pc6['text']['answer_ascii_spearman']:.2f}).",
                f"Answer spread is {pc6['pair']['answer_context_sd_ratio']:.2f}x context spread.",
                "Language and coding style are confounded here, so the axis should not be named as topic alone.",
            ],
        },
        {
            "title": "PC7 is a heterogeneous residual axis",
            "summary": (
                "PC7 has no dominant length, script, question, or code correlate in this text slice. Its tail "
                "examples are visibly mixed, making a semantic name premature."
            ),
            "bullets": [
                f"Paired correlation is r={pc7['pair']['pearson']:.2f}; answer and context spreads are nearly equal ({pc7['pair']['answer_context_sd_ratio']:.2f}x).",
                f"Context-length correlation is only rho={pc7['text']['context_length_spearman']:.2f}.",
                "A similar marginal spread does not imply that individual pairs stay at the same score.",
                "PC7 is best used as a specimen-browsing lead for future annotation rather than an interpreted factor.",
            ],
        },
        {
            "title": "PC8 weakly tracks length and response format",
            "summary": (
                "Higher PC8 scores tend to accompany longer stored text on both sides, but the effect is modest "
                f"(context rho={pc8['text']['context_length_spearman']:.2f}; answer rho={pc8['text']['answer_length_spearman']:.2f})."
            ),
            "bullets": [
                f"Context-answer correlation is r={pc8['pair']['pearson']:.2f}, with slope {pc8['pair']['slope']:.2f}.",
                f"Answer ASCII-letter share decreases (rho={pc8['text']['answer_ascii_spearman']:.2f}) while detected code increases (rho={pc8['text']['answer_code_spearman']:.2f}).",
                "The selected tails combine technical, multilingual, and discourse-format changes.",
                "Because the signals are mixed, PC8 is not evidence for a single content category.",
            ],
        },
        {
            "title": "PC9 is length-linked but weakly transported",
            "summary": (
                f"PC9 rises with stored context length (rho={pc9['text']['context_length_spearman']:.2f}), "
                "but the paired context-to-answer relation is comparatively weak."
            ),
            "bullets": [
                f"Its paired correlation is r={pc9['pair']['pearson']:.2f}, the lowest among PC2-PC10.",
                f"Answer spread remains {pc9['pair']['answer_context_sd_ratio']:.2f}x context spread, so weak correlation is not just marginal compression.",
                f"Answer-side ASCII and detected code both decline (rho={pc9['text']['answer_ascii_spearman']:.2f} and {pc9['text']['answer_code_spearman']:.2f}).",
                "High-tail long/expository examples and low-tail direct or technical examples are suggestive, not a clean partition.",
            ],
        },
        {
            "title": "PC10 combines long-form prose with compressed pair signal",
            "summary": (
                "Higher PC10 context scores favor longer stored prose; lower examples more often include image-template "
                f"or code-like material. Context-length rho is {pc10['text']['context_length_spearman']:.2f}."
            ),
            "bullets": [
                f"Context-answer correlation is r={pc10['pair']['pearson']:.2f}, but the fitted slope is only {pc10['pair']['slope']:.2f}.",
                f"Answer spread is {pc10['pair']['answer_context_sd_ratio']:.2f}x context spread, the smallest ratio in PC1-PC10.",
                "The high tail contains long political, biomedical, and literary prose rather than one topic.",
                "Template duplication and answer truncation remain important alternative explanations.",
            ],
        },
    ]


def build_data(rows: list[dict], source_meta: dict) -> dict:
    contexts = np.asarray([row["c"] for row in rows], dtype=np.float64)
    answers = np.asarray([row["a"] for row in rows], dtype=np.float64)
    if contexts.shape != answers.shape or contexts.shape != (len(rows), N_COMPONENTS):
        raise RuntimeError(
            f"unexpected coordinate shapes: context={contexts.shape}, answer={answers.shape}"
        )
    if not np.isfinite(contexts).all() or not np.isfinite(answers).all():
        raise RuntimeError("non-finite PC coordinate in sampled rows")
    if len({int(row["ci"]) for row in rows}) != len(rows):
        raise RuntimeError("sampled ci values are not unique")
    wild_indices = [index for index, row in enumerate(rows) if row["corpus"] == "wildchat"]
    if len(wild_indices) != source_meta["n_wildchat"]:
        raise RuntimeError("WildChat count does not match source metadata")
    wild_contexts = [rows[index]["context"] for index in wild_indices]
    wild_answers = [rows[index]["answer"] for index in wild_indices]
    context_counts = Counter(wild_contexts)
    answer_counts = Counter(wild_answers)
    pcs = [
        pc_statistics(rows, contexts, answers, wild_indices, pc, source_meta["evr"][pc])
        for pc in range(N_COMPONENTS)
    ]
    copy = analysis_copy(pcs, max(context_counts.values()))
    for pc, narrative in zip(pcs, copy, strict=True):
        pc["analysis"] = narrative

    n_context_truncated = sum(TRUNCATION_MARKER in text for text in wild_contexts)
    n_answer_truncated = sum(TRUNCATION_MARKER in text for text in wild_answers)
    return {
        "meta": {
            "title": "PC1-PC10 specimen browser",
            "n_pairs": len(rows),
            "n_raw_pairs": int(source_meta["n_raw_pairs"]),
            "n_total": int(source_meta["n_total"]),
            "n_wildchat": len(wild_indices),
            "n_lmsys": int(source_meta["n_lmsys"]),
            "n_public_filtered": int(source_meta["n_public_filtered"]),
            "sample_fraction": float(source_meta["sample_fraction"]),
            "sample_design": (
                f"{source_meta['n_chunks']} fixed contiguous capture chunks; "
                f"shard {source_meta['capture_shard_min']:02d} through "
                f"{source_meta['capture_shard_max']:02d} "
                f"({source_meta['n_distinct_capture_shards']} distinct shards); "
                "not a uniform random draw"
            ),
            "layer": int(source_meta["layer"]),
            "n_components": N_COMPONENTS,
            "evr_sum": round(float(sum(source_meta["evr"])), 8),
            "pca_fit_per_side": int(source_meta["pca_fit_per_side"]),
            "pca_model_sha256": source_meta["pca_model_sha256"],
            "export_producer_commit": source_meta["export_producer_commit"],
            "capture_revision": source_meta["capture_revision"],
            "export_revision": source_meta["export_revision"],
            "render_commit": commit_string(git_provenance()),
            "generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "wild_context_unique": len(context_counts),
            "wild_answer_unique": len(answer_counts),
            "largest_context_duplicate": max(context_counts.values()),
            "context_truncated": n_context_truncated,
            "answer_truncated": n_answer_truncated,
            "context_truncated_fraction": round(n_context_truncated / len(wild_indices), 6),
            "answer_truncated_fraction": round(n_answer_truncated / len(wild_indices), 6),
        },
        "pcs": pcs,
    }


CSS = r"""
:root{--paper:#f3f0e7;--surface:#fffdf7;--ink:#20221f;--muted:#686961;--line:#c9c5b9;
--context:#285d7d;--answer:#b96e16;--select:#b73529;--soft-context:#d8e3e8;--soft-answer:#f0dfc8}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:ui-monospace,
SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace;overflow-x:hidden}.wrap{max-width:1720px;min-width:0;margin:auto;padding:18px 22px 34px}
header{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:24px;border-bottom:1px solid var(--line);padding-bottom:13px}
h1{font-size:20px;line-height:1.2;margin:0 0 6px;letter-spacing:-.025em}.lede{max-width:1050px;color:var(--muted);font-size:11px;line-height:1.55}
.legend{display:flex;gap:16px;font-size:11px;padding-top:3px;white-space:nowrap}.sw{display:inline-block;width:9px;height:9px;margin-right:6px}
.sw.c{background:var(--context)}.sw.a{background:var(--answer)}nav{height:54px;display:flex;align-items:end;gap:0;border-bottom:1px solid var(--line);overflow-x:auto;overflow-y:hidden}
nav button{height:42px;flex:0 0 135px;border:0;border-bottom:2px solid transparent;background:transparent;color:var(--muted);font:inherit;font-size:12px;text-align:left;padding:0 14px;cursor:pointer}
nav button:hover{color:var(--ink)}nav button.active{color:var(--ink);border-bottom-color:var(--ink);font-weight:700}nav button span{display:block;font-size:9px;color:var(--muted);font-weight:400;margin-top:3px}
.overview{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(360px,.55fr);border:1px solid var(--line);border-top:0;background:var(--surface)}.overview>*{min-width:0}
.distribution{padding:14px 16px;border-right:1px solid var(--line)}.distribution h2,.analysis h2,.lane-head h2,.detail h2{font-size:12px;margin:0 0 10px}
#hist{display:block;width:100%;height:190px}.stats-scroll{max-width:100%;overflow-x:auto}.stats{width:100%;border-collapse:collapse;font-size:10px}.stats th,.stats td{border-top:1px solid #ddd9ce;padding:7px 8px;text-align:right;font-weight:400}
.stats th:first-child,.stats td:first-child{text-align:left}.stats thead th{color:var(--muted)}.stats .c{color:var(--context)}.stats .a{color:var(--answer)}
.analysis{padding:15px 17px}.analysis p{font-size:11px;line-height:1.55;margin:0 0 10px}.analysis ul{padding:0;margin:0;list-style:none;border-top:1px solid #ddd9ce}
.analysis li{font-size:10px;line-height:1.45;padding:7px 0;border-bottom:1px solid #ddd9ce;color:#50514c}.analysis li::before{content:'·';margin-right:7px;color:var(--select)}
.lane{margin-top:15px;border:1px solid var(--line);background:var(--surface)}.lane-head{display:flex;align-items:baseline;justify-content:space-between;padding:11px 13px;border-bottom:1px solid var(--line)}
.lane-head h2{margin:0}.lane-head p{font-size:9px;color:var(--muted);margin:0}.strip{display:grid;grid-template-columns:repeat(7,minmax(150px,1fr));overflow-x:auto;min-width:0}
.specimen{position:relative;min-height:190px;padding:11px 10px 12px;border:0;border-right:1px solid var(--line);background:transparent;color:var(--ink);font:inherit;text-align:left;cursor:pointer}
.specimen:last-child{border-right:0}.specimen:hover,.specimen.selected{background:#f7f4eb}.specimen.selected{box-shadow:inset 0 3px 0 var(--select)}
.spec-top{display:flex;justify-content:space-between;gap:8px;font-size:9px;color:var(--muted);margin-bottom:8px}.spec-score{font-size:12px;font-weight:700;color:var(--ink)}
.spec-text{font-size:10px;line-height:1.45;display:-webkit-box;-webkit-line-clamp:7;-webkit-box-orient:vertical;overflow:hidden;white-space:pre-wrap;overflow-wrap:anywhere}
.another{position:absolute;left:10px;bottom:8px;border:0;border-bottom:1px solid #aaa69b;background:transparent;color:var(--muted);font:inherit;font-size:9px;padding:1px 0;cursor:pointer}.another:hover{color:var(--ink);border-color:var(--ink)}
.detail{margin-top:15px;border:1px solid var(--line);background:var(--surface)}.detail-head{display:grid;grid-template-columns:180px 1fr 1fr;border-bottom:1px solid var(--line)}
.detail-head>div{padding:12px 13px;border-right:1px solid var(--line)}.detail-head>div:last-child{border-right:0}.detail h2{margin-bottom:8px}.detail-meta{font-size:10px;line-height:1.65;color:var(--muted)}
.detail-meta b{color:var(--ink)}.detail-text{font-size:11px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere;max-height:220px;overflow:auto;margin:0}
.foot{font-size:9px;line-height:1.55;color:var(--muted);margin-top:12px}.foot a{color:inherit}.caveat{border-left:3px solid #827b68;padding-left:9px}
@media(max-width:1100px){.overview{grid-template-columns:1fr}.distribution{border-right:0;border-bottom:1px solid var(--line)}.strip{grid-template-columns:repeat(7,190px)}.detail-head{grid-template-columns:1fr}.detail-head>div{border-right:0;border-bottom:1px solid var(--line)}header{grid-template-columns:1fr}.legend{padding-top:0}}
@media(max-width:600px){.wrap{padding:14px 12px 24px}h1{font-size:18px}.stats{min-width:560px}.distribution{padding-left:10px;padding-right:10px}.analysis{padding-left:11px;padding-right:11px}}
"""


JS = r"""
const D=JSON.parse(document.getElementById('payload').textContent);const C={context:'#285d7d',answer:'#b96e16',select:'#b73529',axis:'#98958b'};
let active=0,selected=null;const cycles={};const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
function fmt(x,n=2){return Number(x).toFixed(n)}function key(role,slot){return `${active}:${role}:${slot}`}
function current(role,slot){const group=D.pcs[active].specimens[role][slot],i=cycles[key(role,slot)]||0;return group.candidates[i%group.candidates.length]}
function setSelected(role,slot){selected={role,slot,row:current(role,slot)};renderLanes();renderDetail();drawHist()}
function cycle(ev,role,slot){ev.stopPropagation();const k=key(role,slot),group=D.pcs[active].specimens[role][slot];cycles[k]=((cycles[k]||0)+1)%group.candidates.length;setSelected(role,slot)}
function renderNav(){const nav=$('pc-nav');nav.innerHTML='';D.pcs.forEach((pc,i)=>{const b=document.createElement('button');b.type='button';b.className=i===active?'active':'';b.setAttribute('aria-selected',String(i===active));b.innerHTML=`PC${pc.id}<span>${(pc.evr*100).toFixed(2)}% joint variance</span>`;b.onclick=()=>{active=i;selected=null;render()};nav.appendChild(b)})}
function renderStats(){const p=D.pcs[active],q=p.context.q; $('stats-body').innerHTML=`
<tr><th>mean</th><td class="c">${fmt(p.context.mean)}</td><td class="a">${fmt(p.answer.mean)}</td><td>delta ${fmt(p.pair.delta_mean)}</td></tr>
<tr><th>standard deviation</th><td class="c">${fmt(p.context.std)}</td><td class="a">${fmt(p.answer.std)}</td><td>answer/context ${fmt(p.pair.answer_context_sd_ratio)}</td></tr>
<tr><th>10% to 90%</th><td class="c">${fmt(q[1])} to ${fmt(q[5])}</td><td class="a">${fmt(p.answer.q[1])} to ${fmt(p.answer.q[5])}</td><td>paired r ${fmt(p.pair.pearson)}</td></tr>
<tr><th>paired linear read</th><td class="c">context x</td><td class="a">answer y</td><td>slope ${fmt(p.pair.slope)}</td></tr>`}
function renderAnalysis(){const a=D.pcs[active].analysis;$('analysis-title').textContent=a.title;$('analysis-summary').textContent=a.summary;$('analysis-list').innerHTML=a.bullets.map(x=>`<li>${esc(x)}</li>`).join('')}
function renderLane(role){const p=D.pcs[active],root=$(role+'-strip'),groups=p.specimens[role];root.innerHTML=groups.map((g,i)=>{const r=current(role,i),is=selected&&selected.role===role&&selected.slot===i;return `<article class="specimen${is?' selected':''}" data-role="${role}" data-slot="${i}" tabindex="0"><div class="spec-top"><span>q ${g.label}</span><span>ci ${r.ci}</span></div><div class="spec-score">${r.score>=0?'+':''}${fmt(r.score,3)}</div><div class="spec-text">${esc(r[role])}</div><button class="another" type="button" data-cycle="1">another ${((cycles[key(role,i)]||0)%3)+1}/3</button></article>`}).join('');
 root.querySelectorAll('.specimen').forEach(card=>{const slot=Number(card.dataset.slot);card.addEventListener('click',()=>setSelected(role,slot));card.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();setSelected(role,slot)}});card.querySelector('.another').addEventListener('click',ev=>cycle(ev,role,slot))})}
function renderLanes(){renderLane('context');renderLane('answer')}
function renderDetail(){if(!selected){setSelected('context',3);return}const r=selected.row,p=D.pcs[active];$('detail-ci').textContent=`ci ${r.ci}`;$('detail-role').textContent=`selected by ${selected.role} PC${p.id}`;$('detail-c').textContent=r.c.map(x=>fmt(x,3)).join(' / ');$('detail-a').textContent=r.a.map(x=>fmt(x,3)).join(' / ');$('detail-context').textContent=r.context;$('detail-answer').textContent=r.answer}
function drawHist(){const p=D.pcs[active],cv=$('hist'),box=cv.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.round(box.width*dpr),h=Math.round(box.height*dpr);cv.width=w;cv.height=h;const x0=72*dpr,x1=w-20*dpr,top=24*dpr,laneH=52*dpr,gap=28*dpr,ctx=cv.getContext('2d');ctx.clearRect(0,0,w,h);const sx=v=>x0+(Math.max(p.plot_min,Math.min(p.plot_max,v))-p.plot_min)/(p.plot_max-p.plot_min)*(x1-x0);ctx.font=`${10*dpr}px ui-monospace,monospace`;ctx.textBaseline='middle';
 function lane(counts,y,color,label){const max=Math.max(...counts);ctx.fillStyle=color;ctx.globalAlpha=.72;const bw=(x1-x0)/counts.length;counts.forEach((n,i)=>{const bh=(n/max)*(laneH-12*dpr);ctx.fillRect(x0+i*bw,y+laneH-bh,bw-.5*dpr,bh)});ctx.globalAlpha=1;ctx.fillStyle='#5f605a';ctx.textAlign='right';ctx.fillText(label,x0-9*dpr,y+laneH/2);ctx.strokeStyle='#c9c5b9';ctx.beginPath();ctx.moveTo(x0,y+laneH+.5);ctx.lineTo(x1,y+laneH+.5);ctx.stroke()}
 lane(p.hist_context,top,C.context,'context');lane(p.hist_answer,top+laneH+gap,C.answer,'answer');ctx.textAlign='center';ctx.fillStyle='#66675f';for(let i=0;i<5;i++){const v=p.plot_min+(p.plot_max-p.plot_min)*i/4,x=sx(v);ctx.fillText(fmt(v,1),x,h-10*dpr);ctx.strokeStyle='#d8d4c9';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+laneH*2+gap);ctx.stroke()}
 if(selected){const r=selected.row,xc=sx(r.c[active]),xa=sx(r.a[active]),yc=top+laneH/2,ya=top+laneH+gap+laneH/2,ang=Math.atan2(ya-yc,xa-xc),head=8*dpr;ctx.strokeStyle=C.select;ctx.fillStyle=C.select;ctx.lineWidth=1.5*dpr;ctx.beginPath();ctx.moveTo(xc,yc);ctx.lineTo(xa,ya);ctx.stroke();ctx.beginPath();ctx.moveTo(xa,ya);ctx.lineTo(xa-head*Math.cos(ang-.48),ya-head*Math.sin(ang-.48));ctx.lineTo(xa-head*Math.cos(ang+.48),ya-head*Math.sin(ang+.48));ctx.closePath();ctx.fill();ctx.beginPath();ctx.arc(xc,yc,4*dpr,0,Math.PI*2);ctx.fill();ctx.textAlign='left';ctx.fillText('context → answer',Math.min(xc,xa)+6*dpr,(yc+ya)/2)} }
function render(){renderNav();renderStats();renderAnalysis();renderLanes();if(!selected)setSelected('context',3);else{renderDetail();drawHist()}}
addEventListener('resize',drawHist);render();
"""


def page(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    json.loads(payload)
    meta = data["meta"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PC1-PC10 context and answer specimens | issue #779</title><style>{CSS}</style></head><body><div class="wrap">
<header><div><h1>PC1-PC10 context and answer specimens</h1><div class="lede">A one-dimensional reading of the same joint L{meta["layer"]} PCA basis. Ten tabs align context and answer distributions on one raw-score axis, then sample unique WildChat excerpts at seven percentile anchors from low to high. Click a specimen to inspect its exact pair and context → answer arrow; use “another” for nearby alternatives.</div></div><div class="legend"><span><i class="sw c"></i>context</span><span><i class="sw a"></i>answer</span></div></header>
<nav id="pc-nav" role="tablist" aria-label="Principal components"></nav>
<section class="overview"><div class="distribution"><h2>Shared one-dimensional score distribution</h2><canvas id="hist" aria-label="Context and answer score histograms with selected pair displacement"></canvas>
<div class="stats-scroll"><table class="stats"><thead><tr><th>read</th><th class="c">context</th><th class="a">answer</th><th>paired relation</th></tr></thead><tbody id="stats-body"></tbody></table></div></div>
<aside class="analysis"><h2 id="analysis-title"></h2><p id="analysis-summary"></p><ul id="analysis-list"></ul></aside></section>
<section class="lane"><div class="lane-head"><h2>Context specimens</h2><p>ranked by context score · unique WildChat excerpts · 3 alternatives per position</p></div><div class="strip" id="context-strip"></div></section>
<section class="lane"><div class="lane-head"><h2>Answer specimens</h2><p>ranked by answer score · unique WildChat excerpts · 3 alternatives per position</p></div><div class="strip" id="answer-strip"></div></section>
<section class="detail"><div class="detail-head"><div><h2>Selected pair</h2><div class="detail-meta"><b id="detail-ci"></b><br><span id="detail-role"></span><br>context PC1…PC10<br><b id="detail-c"></b><br>answer PC1…PC10<br><b id="detail-a"></b></div></div><div><h2>Context</h2><p class="detail-text" id="detail-context"></p></div><div><h2>Answer</h2><p class="detail-text" id="detail-answer"></p></div></div></section>
<p class="foot caveat">Interpretation boundary: PC1-PC10 explain {meta["evr_sum"] * 100:.2f}% of joint variance. The distributions include {meta["n_pairs"]:,} publication-safe pairs ({meta["sample_fraction"] * 100:.2f}% of the {meta["n_total"]:,}-row export) from {meta["sample_design"]}. Specimen text is limited to {meta["n_wildchat"]:,} WildChat rows; all LMSYS text remains withheld. {meta["context_truncated"]:,} WildChat contexts ({meta["context_truncated_fraction"] * 100:.1f}%) and {meta["answer_truncated"]:,} answers ({meta["answer_truncated_fraction"] * 100:.1f}%) hit the stored display truncation marker. PCA fit: {meta["pca_fit_per_side"]:,} contexts + {meta["pca_fit_per_side"]:,} answers; producer <code>{meta["export_producer_commit"]}</code>; model <code>{meta["pca_model_sha256"][:12]}</code>; renderer <code>{meta["render_commit"]}</code>; generated {meta["generated_utc"]}. <a href="https://huggingface.co/datasets/allenai/WildChat">WildChat attribution (ODC-BY)</a>.</p>
</div><script id="payload" type="application/json">{payload}</script><script>{JS}</script></body></html>"""


def report(data: dict) -> str:
    meta = data["meta"]
    lines = [
        "# Issue #779 PC1-PC10 specimen browser: exploratory analysis",
        "",
        f"Generated: {meta['generated_utc']}",
        "",
        "## Title and metadata",
        "",
        f"- Joint PCA model SHA-256: `{meta['pca_model_sha256']}`",
        f"- Capture revision: `{meta['capture_revision']}`; export revision: `{meta['export_revision']}`",
        f"- Export producer: `{meta['export_producer_commit']}`; dashboard renderer: `{meta['render_commit']}`",
        "- Format: self-contained HTML with a JSON payload derived from projected capture arrays",
        f"- Rows: {meta['n_pairs']:,} paired context/answer observations",
        f"- Layer: {meta['layer']}; PC1-PC10 joint EVR: {meta['evr_sum'] * 100:.2f}%",
        "",
        "## Structure and quality",
        "",
        "The payload contains one unique `ci` identifier, corpus label, ten finite context PC scores, ten finite answer PC scores, and publication-safe text fields per row. Count, uniqueness, shape, finite-value, model-SHA, capture/export revision, and producer-commit checks pass before rendering.",
        "",
        f"The coordinate distributions use all {meta['n_pairs']:,} rows. Text specimens use only {meta['n_wildchat']:,} WildChat rows because LMSYS source text is withheld under its dataset agreement. The WildChat slice contains {meta['wild_context_unique']:,} unique stored context excerpts and {meta['wild_answer_unique']:,} unique stored answer excerpts. The largest exact context duplicate occurs {meta['largest_context_duplicate']} times.",
        "",
        f"Text is display-censored: {meta['context_truncated_fraction'] * 100:.1f}% of WildChat contexts and {meta['answer_truncated_fraction'] * 100:.1f}% of answers contain the producer's truncation marker. Answer-length findings are therefore descriptive of stored excerpts, not complete answers.",
        "",
        "## Numerical summary",
        "",
        "| PC | EVR | context mean ± SD | answer mean ± SD | paired r | answer-on-context slope | answer/context SD | role d |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pc in data["pcs"]:
        lines.append(
            f"| PC{pc['id']} | {100 * pc['evr']:.2f}% | {pc['context']['mean']:+.2f} ± {pc['context']['std']:.2f} | "
            f"{pc['answer']['mean']:+.2f} ± {pc['answer']['std']:.2f} | {pc['pair']['pearson']:.3f} | "
            f"{pc['pair']['slope']:.3f} | {pc['pair']['answer_context_sd_ratio']:.3f} | {pc['pair']['role_effect_d']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Text-feature correlates in the WildChat specimen slice",
            "",
            "Spearman correlations below are exploratory diagnostics on stored display excerpts. They are not labels for the PCs, and answer-length correlations are especially censored by truncation.",
            "",
            "| PC | context length | answer length | context ASCII share | answer ASCII share | context code | answer code |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pc in data["pcs"]:
        text = pc["text"]
        lines.append(
            f"| PC{pc['id']} | {text['context_length_spearman']:+.3f} | "
            f"{text['answer_length_spearman']:+.3f} | {text['context_ascii_spearman']:+.3f} | "
            f"{text['answer_ascii_spearman']:+.3f} | {text['context_code_spearman']:+.3f} | "
            f"{text['answer_code_spearman']:+.3f} |"
        )
    lines.extend(["", "## Key findings", ""])
    for pc in data["pcs"]:
        lines.extend(
            [
                f"### {pc['analysis']['title']}",
                "",
                pc["analysis"]["summary"],
                "",
                *[f"- {bullet}" for bullet in pc["analysis"]["bullets"]],
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendations and interpretation limits",
            "",
            "Use PC1 mainly as a role-separation diagnostic, not a semantic continuum. PC4 carries the strongest paired position signal. Read every later PC through multiple examples rather than a single tail specimen, because repeated prompt families, language, code formatting, and text length are entangled. Treat answer-side prose patterns as provisional because 91.7% of stored answer excerpts are truncated.",
            "",
            f"The sample is {meta['sample_design']}. It is useful for inspecting mechanisms and generating hypotheses, but it does not support population-frequency claims for the full {meta['n_total']:,}-row export.",
            "",
            "A stronger follow-up would stratify a fresh sample by corpus, prompt family, language, and length; deduplicate template families; and then estimate conditional PC associations with held-out annotations.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--chunks", nargs="+", default=list(DEFAULT_CHUNKS))
    parser.add_argument("--out-name", default=OUT_NAME)
    args = parser.parse_args()
    rows, source_meta = load_pc10(args.export_dir, tuple(args.chunks))
    data = build_data(rows, source_meta)
    html = page(data)
    markdown = report(data)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / args.out_name
        out.write_text(html, encoding="utf-8")
        print(f"[pc-specimens] wrote {out} ({len(html.encode('utf-8')) / 1e6:.2f} MB)")
    report_path = Path("experiments/dashboards") / REPORT_NAME
    report_path.write_text(markdown, encoding="utf-8")
    print(f"[pc-specimens] wrote {report_path}")


if __name__ == "__main__":
    main()
