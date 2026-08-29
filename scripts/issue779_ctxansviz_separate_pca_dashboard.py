"""Build the full-text separate-PCA specimen browser for issue #779.

Uses independently fitted context-only and answer-only PCA-10 bases. Native PC
scores are shown separately, while cross-role arrows use an optimal one-to-one
loading-vector match, sign orientation, and within-role standardization. Full
WildChat prompts and answers are recovered from pinned raw-completion files;
LMSYS text remains withheld from the public artifact.

Usage:
    uv run python scripts/issue779_ctxansviz_separate_pca_dashboard.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

import issue779_ctxansviz_pc_specimens as joint_source
import issue779_ctxansviz_pca3_dashboard as pca3_source
from explore_persona_space.orchestrate.provenance import commit_string, git_provenance

CAPTURE_REVISION = "cbc55efdd7f5581677047e487aa61172f6e7944d"
EXPORT_REVISION = "d155ed93f4b0184a477cea51aef65cc5440da588"
EXPORT_PRODUCER_COMMIT = "79d9142bf5c88ae2ccd3ff7270e9d98a1faaaa5d"
HF_REPO = "superkaiba1/explore-persona-space-data"
CAPTURE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
RAW_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/raw_completions"
DEFAULT_EXPORT = Path("data/issue_779/ctxansviz_dl/full/issue779_monitoring/ctxansviz")
DEFAULT_MODEL_DIR = Path("data/issue_779/ctxansviz_separate_pca")
MODEL_NAME = "separate_pca10_models.npz"
MODEL_META_NAME = "separate_pca10_meta.json"
DEFAULT_CHUNKS = pca3_source.DEFAULT_CHUNKS
LAYER = 19
N_COMPONENTS = 10
N_FIT_ROWS = 200_000
OUT_NAME = "ctxansviz-779-separate-pca-fulltext.html"
REPORT_NAME = "ctxansviz-779-separate-pca-fulltext-analysis.md"
QUANTILES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
N_CANDIDATES = 3
PRODUCER_TRUNCATION = " …[truncated]"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_models(model_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    model_path = model_dir / MODEL_NAME
    meta_path = model_dir / MODEL_META_NAME
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"separate PCA model artifacts absent from {model_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("capture_revision") != CAPTURE_REVISION:
        raise RuntimeError("separate PCA capture revision is not pinned to the display capture")
    if meta.get("git_dirty") is not False or meta.get("git_argv0_state") != "tracked":
        raise RuntimeError("separate PCA model was not produced by clean tracked code")
    if meta.get("n_fit_rows_per_basis") != N_FIT_ROWS:
        raise RuntimeError("separate PCA fit-row count changed")
    if sha256_file(model_path) != meta.get("model_sha256"):
        raise RuntimeError("separate PCA model SHA-256 does not match its manifest")
    raw = np.load(model_path)
    model = {key: np.asarray(raw[key]) for key in raw.files}
    expected_shapes = {
        "context_components": (N_COMPONENTS, 3584),
        "context_mean": (3584,),
        "answer_components": (N_COMPONENTS, 3584),
        "answer_mean": (3584,),
        "loading_cosine": (N_COMPONENTS, N_COMPONENTS),
        "answer_for_context": (N_COMPONENTS,),
        "orientation_for_context": (N_COMPONENTS,),
    }
    for key, shape in expected_shapes.items():
        if model[key].shape != shape:
            raise RuntimeError(f"{key} shape {model[key].shape} != {shape}")
    for role in ("context", "answer"):
        components = model[f"{role}_components"].astype(np.float64)
        error = float(np.abs(components @ components.T - np.eye(N_COMPONENTS)).max())
        if error > 1e-4:
            raise RuntimeError(f"{role} PCA components are not orthonormal: max error {error}")
    recomputed = model["context_components"] @ model["answer_components"].T
    if not np.allclose(recomputed, model["loading_cosine"], atol=2e-6):
        raise RuntimeError("saved cross-basis loading cosines do not match components")
    return model, meta


def load_fulltext_rows(
    export_dir: Path,
    chunks: tuple[str, ...],
    model: dict[str, np.ndarray],
) -> tuple[list[dict], dict]:
    download_meta_path = export_dir / "_download_meta.json"
    producer_meta_path = export_dir / "meta.json"
    if not download_meta_path.exists() or not producer_meta_path.exists():
        raise FileNotFoundError(f"pinned export metadata absent from {export_dir}")
    download_meta = json.loads(download_meta_path.read_text(encoding="utf-8"))
    producer_meta = json.loads(producer_meta_path.read_text(encoding="utf-8"))
    if download_meta.get("revision") != EXPORT_REVISION:
        raise RuntimeError("export revision does not match pinned revision")
    if producer_meta.get("git_commit") != EXPORT_PRODUCER_COMMIT:
        raise RuntimeError("export producer does not match pinned commit")
    export_hashes = producer_meta.get("export_files_sha256", {})

    raw_vectors: list[tuple[int, np.ndarray, np.ndarray]] = []
    raw_text: dict[int, tuple[str, str]] = {}
    for chunk_name in chunks:
        capture_path = hf_hub_download(
            HF_REPO,
            filename=f"{CAPTURE_PREFIX}/{chunk_name}",
            repo_type="dataset",
            revision=CAPTURE_REVISION,
        )
        bundle = torch.load(capture_path, mmap=True, weights_only=False, map_location="cpu")
        layers = [int(value) for value in bundle["layers"]]
        if LAYER not in layers:
            raise RuntimeError(f"{chunk_name}: layer {LAYER} absent")
        column = layers.index(LAYER)
        cx = bundle["cx_last"][:, column, :].to(torch.float32).numpy()
        vx = bundle["v_x"][:, column, :].to(torch.float32).numpy()
        cis = [int(value) for value in bundle["ci"]]
        if cx.shape != vx.shape or cx.shape != (len(cis), 3584):
            raise RuntimeError(f"{chunk_name}: malformed capture arrays")
        pc_cx = (cx - model["context_mean"]) @ model["context_components"].T
        pc_vx = (vx - model["answer_mean"]) @ model["answer_components"].T
        raw_vectors.extend((ci, pc_cx[pos], pc_vx[pos]) for pos, ci in enumerate(cis))

        raw_path = hf_hub_download(
            HF_REPO,
            filename=f"{RAW_PREFIX}/{chunk_name.removesuffix('.pt')}.json",
            repo_type="dataset",
            revision=CAPTURE_REVISION,
        )
        completion = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        chunk_text = {
            int(row["ci"]): (str(row["prompt"]), str(row["response"])) for row in completion["rows"]
        }
        if set(chunk_text) != set(cis):
            raise RuntimeError(f"{chunk_name}: raw-completion ci set does not match capture")
        raw_text.update(chunk_text)

    target_cis = {ci for ci, _, _ in raw_vectors}
    if len(target_cis) != len(raw_vectors) or set(raw_text) != target_cis:
        raise RuntimeError("selected display chunks contain duplicate or missing ci values")

    excerpt_meta: dict[int, dict] = {}
    for part in sorted(export_dir.glob("row_meta_*.jsonl")):
        expected_sha = export_hashes.get(part.name)
        if not expected_sha or sha256_file(part) != expected_sha:
            raise RuntimeError(f"row metadata SHA-256 mismatch: {part.name}")
        for row in pca3_source.iter_jsonl(part):
            ci = int(row["ci"])
            if ci in target_cis:
                excerpt_meta[ci] = row
        if len(excerpt_meta) == len(target_cis):
            break
    if set(excerpt_meta) != target_cis:
        raise RuntimeError("not every display ci has pinned corpus metadata")

    recovered_context = recovered_answer = 0
    rows = []
    for ci, pc_cx, pc_vx in raw_vectors:
        old = excerpt_meta[ci]
        prompt, answer = raw_text[ci]
        for full, excerpt, role in (
            (prompt, str(old["context_text"]), "context"),
            (answer, str(old["answer_text"]), "answer"),
        ):
            prefix = excerpt.removesuffix(PRODUCER_TRUNCATION)
            if not full.startswith(prefix):
                raise RuntimeError(f"ci={ci}: full {role} does not extend pinned excerpt")
        if str(old["corpus"]) == "wildchat":
            recovered_context += str(old["context_text"]).endswith(PRODUCER_TRUNCATION)
            recovered_answer += str(old["answer_text"]).endswith(PRODUCER_TRUNCATION)
        rows.append(
            {
                "ci": ci,
                "corpus": str(old["corpus"]),
                "c": [round(float(value), 5) for value in pc_cx],
                "a": [round(float(value), 5) for value in pc_vx],
                "context": prompt,
                "answer": answer,
            }
        )

    n_raw_pairs = len(rows)
    rows, filter_counts, n_lmsys_withheld = pca3_source._prepare_public_rows(rows)
    n_lmsys = sum(row["corpus"] == "lmsys" for row in rows)
    n_wildchat = sum(row["corpus"] == "wildchat" for row in rows)
    if n_lmsys != n_lmsys_withheld or n_lmsys + n_wildchat != len(rows):
        raise RuntimeError("public corpus counts do not reconcile")
    if any(
        PRODUCER_TRUNCATION in row[role]
        for row in rows
        if row["corpus"] == "wildchat"
        for role in ("context", "answer")
    ):
        raise RuntimeError("a public WildChat full-text row still contains the producer cap marker")

    parsed = [pca3_source.CHUNK_NAME.fullmatch(name) for name in chunks]
    if any(match is None for match in parsed):
        raise RuntimeError("invalid display capture chunk name")
    shard_ids = [int(match.group(1)) for match in parsed if match is not None]
    return rows, {
        "n_pairs": len(rows),
        "n_raw_pairs": n_raw_pairs,
        "n_public_filtered": n_raw_pairs - len(rows),
        "public_filter_counts": filter_counts,
        "n_lmsys": n_lmsys,
        "n_wildchat": n_wildchat,
        "n_total": int(producer_meta["n_rows"]),
        "sample_fraction": len(rows) / int(producer_meta["n_rows"]),
        "n_chunks": len(chunks),
        "shard_min": min(shard_ids),
        "shard_max": max(shard_ids),
        "n_distinct_shards": len(set(shard_ids)),
        "recovered_context": recovered_context,
        "recovered_answer": recovered_answer,
    }


def specimen_candidates(
    rows: list[dict], wild_indices: list[int], role: str, pc: int
) -> list[dict]:
    score_key = "c" if role == "context" else "a"
    values = np.asarray([rows[index][score_key][pc] for index in wild_indices])
    groups = []
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
            fingerprint = joint_source.text_fingerprint(row[role])
            if fingerprint in used_fingerprints or int(row["ci"]) in used_cis:
                continue
            candidates.append(
                {
                    "ci": int(row["ci"]),
                    "score": round(float(row[score_key][pc]), 5),
                }
            )
            used_fingerprints.add(fingerprint)
            used_cis.add(int(row["ci"]))
            if len(candidates) == N_CANDIDATES:
                break
        if len(candidates) != N_CANDIDATES:
            raise RuntimeError(f"not enough unique {role} PC{pc + 1} specimens")
        groups.append(
            {
                "q": quantile,
                "label": f"{quantile * 100:g}%",
                "target": round(target, 5),
                "candidates": candidates,
            }
        )
    return groups


def axis_statistics(
    rows: list[dict], values: np.ndarray, wild_indices: list[int], role: str, pc: int, evr: float
) -> dict:
    wild_values = values[wild_indices, pc]
    texts = [rows[index][role] for index in wild_indices]
    feature_rows = [joint_source.text_features(text) for text in texts]

    def feature(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in feature_rows], dtype=np.float64)

    correlations = {
        "length": joint_source.spearman(wild_values, feature("chars")),
        "ascii share": joint_source.spearman(wild_values, feature("ascii_letter_share")),
        "question mark": joint_source.spearman(wild_values, feature("question")),
        "detected code": joint_source.spearman(wild_values, feature("code")),
    }
    strongest = max(correlations, key=lambda key: abs(correlations[key]))
    return {
        "id": pc + 1,
        "role": role,
        "evr": round(float(evr), 8),
        "mean": round(float(values[:, pc].mean()), 5),
        "std": round(float(values[:, pc].std()), 5),
        "q": [
            round(float(value), 5)
            for value in np.quantile(values[:, pc], (0.01, 0.1, 0.5, 0.9, 0.99))
        ],
        "text_correlations": {key: round(float(value), 6) for key, value in correlations.items()},
        "strongest_text_correlate": strongest,
        "strongest_text_rho": round(float(correlations[strongest]), 6),
        "full_text_chars": {
            "median": round(float(np.median(feature("chars"))), 1),
            "p90": round(float(np.quantile(feature("chars"), 0.9)), 1),
            "max": int(np.max(feature("chars"))),
        },
        "specimens": specimen_candidates(rows, wild_indices, role, pc),
    }


def relation_statistics(
    contexts: np.ndarray,
    answers: np.ndarray,
    context_pc: int,
    answer_pc: int,
    sign: int,
    loading_cosine: float,
) -> dict:
    context = contexts[:, context_pc]
    answer_aligned = answers[:, answer_pc] * sign
    context_mean, context_std = float(context.mean()), float(context.std())
    answer_mean, answer_std = float(answer_aligned.mean()), float(answer_aligned.std())
    context_z = (context - context_mean) / context_std
    answer_z = (answer_aligned - answer_mean) / answer_std
    edges = np.linspace(-3.5, 3.5, 71)
    hist_context, _ = np.histogram(context_z, bins=edges)
    hist_answer, _ = np.histogram(answer_z, bins=edges)
    return {
        "context_pc": context_pc,
        "answer_pc": answer_pc,
        "answer_sign": int(sign),
        "loading_cosine_raw": round(float(loading_cosine), 6),
        "loading_cosine_aligned": round(float(loading_cosine * sign), 6),
        "paired_r_aligned": round(float(np.corrcoef(context_z, answer_z)[0, 1]), 6),
        "paired_z_rmse": round(float(np.sqrt(np.mean((answer_z - context_z) ** 2))), 6),
        "context_mean": round(context_mean, 6),
        "context_std": round(context_std, 6),
        "answer_aligned_mean": round(answer_mean, 6),
        "answer_aligned_std": round(answer_std, 6),
        "hist_edges": [round(float(value), 4) for value in edges],
        "hist_context": hist_context.tolist(),
        "hist_answer": hist_answer.tolist(),
    }


def build_data(rows: list[dict], sample_meta: dict, model: dict, model_meta: dict) -> dict:
    contexts = np.asarray([row["c"] for row in rows], dtype=np.float64)
    answers = np.asarray([row["a"] for row in rows], dtype=np.float64)
    if contexts.shape != answers.shape or contexts.shape != (len(rows), N_COMPONENTS):
        raise RuntimeError(f"unexpected display coordinate shape {contexts.shape} {answers.shape}")
    if not np.isfinite(contexts).all() or not np.isfinite(answers).all():
        raise RuntimeError("non-finite separate-PCA display coordinate")
    wild_indices = [index for index, row in enumerate(rows) if row["corpus"] == "wildchat"]
    context_axes = [
        axis_statistics(
            rows,
            contexts,
            wild_indices,
            "context",
            pc,
            model["context_explained_variance_ratio"][pc],
        )
        for pc in range(N_COMPONENTS)
    ]
    answer_axes = [
        axis_statistics(
            rows,
            answers,
            wild_indices,
            "answer",
            pc,
            model["answer_explained_variance_ratio"][pc],
        )
        for pc in range(N_COMPONENTS)
    ]
    answer_for_context = model["answer_for_context"].astype(int)
    orientations = model["orientation_for_context"].astype(int)
    if sorted(answer_for_context.tolist()) != list(range(N_COMPONENTS)):
        raise RuntimeError("saved component assignment is not one-to-one")
    matches = []
    for context_pc, answer_pc in enumerate(answer_for_context):
        relation = relation_statistics(
            contexts,
            answers,
            context_pc,
            int(answer_pc),
            int(orientations[context_pc]),
            float(model["loading_cosine"][context_pc, answer_pc]),
        )
        matches.append(relation)

    selected_cis = {
        int(candidate["ci"])
        for axis in [*context_axes, *answer_axes]
        for group in axis["specimens"]
        for candidate in group["candidates"]
    }
    selected_rows = {
        str(row["ci"]): {
            "ci": int(row["ci"]),
            "context": row["context"],
            "answer": row["answer"],
            "c": row["c"],
            "a": row["a"],
        }
        for row in rows
        if int(row["ci"]) in selected_cis
    }
    if len(selected_rows) != len(selected_cis):
        raise RuntimeError("not every specimen ci has a full-text row")

    wild_contexts = [rows[index]["context"] for index in wild_indices]
    wild_answers = [rows[index]["answer"] for index in wild_indices]
    context_counts = Counter(wild_contexts)
    answer_counts = Counter(wild_answers)
    return {
        "meta": {
            **sample_meta,
            "layer": LAYER,
            "n_fit_rows_per_basis": int(model["n_fit_rows"]),
            "context_evr10": round(float(model["context_explained_variance_ratio"].sum()), 8),
            "answer_evr10": round(float(model["answer_explained_variance_ratio"].sum()), 8),
            "model_sha256": model_meta["model_sha256"],
            "model_producer_commit": model_meta["git_commit"],
            "fit_selection_sha256": model_meta["selection"]["selected_chunk_manifest_sha256"],
            "render_commit": commit_string(git_provenance()),
            "generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "n_specimen_rows": len(selected_rows),
            "wild_context_unique": len(context_counts),
            "wild_answer_unique": len(answer_counts),
            "largest_context_duplicate": max(context_counts.values()),
            "max_full_context_chars": max(len(text) for text in wild_contexts),
            "max_full_answer_chars": max(len(text) for text in wild_answers),
        },
        "axes": {"context": context_axes, "answer": answer_axes},
        "matches": matches,
        "loading_cosine": [
            [round(float(value), 6) for value in row] for row in model["loading_cosine"]
        ],
        "rows": selected_rows,
    }


CSS = r"""
:root{--paper:#f3f0e7;--surface:#fffdf7;--ink:#20221f;--muted:#686961;--line:#c9c5b9;--context:#285d7d;--answer:#b96e16;--select:#b73529}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace;overflow-x:hidden}.wrap{max-width:1700px;margin:auto;padding:18px 22px 34px}
header{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:24px;border-bottom:1px solid var(--line);padding-bottom:13px}header>*{min-width:0}h1{font-size:20px;margin:0 0 6px;letter-spacing:-.025em;overflow-wrap:anywhere}.lede{max-width:1120px;color:var(--muted);font-size:11px;line-height:1.55;overflow-wrap:anywhere}.legend{display:flex;gap:16px;font-size:11px;white-space:nowrap}.sw{display:inline-block;width:9px;height:9px;margin-right:6px}.sw.c{background:var(--context)}.sw.a{background:var(--answer)}
.basis-nav{border-bottom:1px solid var(--line)}.nav-row{display:flex;height:47px;overflow-x:auto}.nav-label{flex:0 0 90px;padding:16px 10px 0;color:var(--muted);font-size:10px}.nav-row button{flex:0 0 118px;border:0;border-bottom:2px solid transparent;background:transparent;color:var(--muted);font:inherit;font-size:11px;text-align:left;padding:7px 10px;cursor:pointer}.nav-row button span{display:block;font-size:9px;margin-top:3px}.nav-row button.active{color:var(--ink);border-bottom-color:var(--ink);font-weight:700}
.matrix-wrap{display:grid;grid-template-columns:210px minmax(0,1fr);border:1px solid var(--line);border-top:0;background:var(--surface)}.matrix-wrap>*{min-width:0}.matrix-copy{padding:13px;border-right:1px solid var(--line);font-size:10px;line-height:1.5;color:var(--muted);overflow-wrap:anywhere}.matrix-copy b{display:block;color:var(--ink);font-size:11px;margin-bottom:6px}.matrix-scroll{overflow-x:auto;padding:8px 12px}.matrix{border-collapse:collapse;font-size:9px}.matrix th{font-weight:400;color:var(--muted);padding:4px 7px}.matrix td{width:58px;height:27px;text-align:center;border:1px solid #ddd9ce}.matrix td.match{outline:2px solid var(--ink);outline-offset:-2px;font-weight:700}
.overview{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(350px,.55fr);border:1px solid var(--line);border-top:0;background:var(--surface)}.overview>*{min-width:0}.distribution{padding:14px 16px;border-right:1px solid var(--line)}h2{font-size:12px;margin:0 0 10px}#hist{display:block;width:100%;height:190px}.stats-scroll{overflow-x:auto}.stats{width:100%;border-collapse:collapse;font-size:10px}.stats th,.stats td{border-top:1px solid #ddd9ce;padding:7px 8px;text-align:right;font-weight:400}.stats th:first-child,.stats td:first-child{text-align:left}.stats .c{color:var(--context)}.stats .a{color:var(--answer)}
.analysis{padding:15px 17px}.analysis p{font-size:11px;line-height:1.55;margin:0 0 10px}.analysis ul{padding:0;margin:0;list-style:none;border-top:1px solid #ddd9ce}.analysis li{font-size:10px;line-height:1.45;padding:7px 0;border-bottom:1px solid #ddd9ce;color:#50514c}.analysis li::before{content:'·';margin-right:7px;color:var(--select)}
.lanes{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.lane{border:1px solid var(--line);background:var(--surface);min-width:0}.lane-head{padding:11px 13px;border-bottom:1px solid var(--line)}.lane-head h2{margin:0 0 4px}.lane-head p{font-size:9px;color:var(--muted);margin:0}.spec-list{height:680px;overflow:auto}.specimen{position:relative;padding:12px 12px 14px;border-bottom:1px solid var(--line);cursor:pointer}.specimen:hover,.specimen.selected{background:#f7f4eb}.specimen.selected{box-shadow:inset 3px 0 0 var(--select)}.spec-top{display:flex;justify-content:space-between;gap:8px;font-size:9px;color:var(--muted);margin-bottom:7px}.spec-score{font-size:12px;font-weight:700;color:var(--ink)}.spec-text{font:inherit;font-size:10px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0 0}.another{border:0;border-bottom:1px solid #aaa69b;background:transparent;color:var(--muted);font:inherit;font-size:9px;padding:1px 0;cursor:pointer}
.detail{margin-top:14px;border:1px solid var(--line);background:var(--surface)}.detail-grid{display:grid;grid-template-columns:190px 1fr 1fr}.detail-grid>div{padding:12px 13px;border-right:1px solid var(--line)}.detail-grid>div:last-child{border-right:0}.detail-meta{font-size:10px;line-height:1.65;color:var(--muted)}.detail-meta b{color:var(--ink)}.detail-text{font-size:11px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere;margin:0}.foot{font-size:9px;line-height:1.55;color:var(--muted);margin-top:12px;border-left:3px solid #827b68;padding-left:9px}
@media(max-width:1050px){header{grid-template-columns:1fr}.overview,.lanes{grid-template-columns:1fr}.distribution{border-right:0;border-bottom:1px solid var(--line)}.matrix-wrap{grid-template-columns:1fr}.matrix-copy{border-right:0;border-bottom:1px solid var(--line)}.detail-grid{grid-template-columns:1fr}.detail-grid>div{border-right:0;border-bottom:1px solid var(--line)}}
@media(max-width:600px){.wrap{padding:14px 12px 24px}h1{font-size:18px}.nav-label{flex-basis:74px}.nav-row button{flex-basis:110px}.stats{min-width:570px}.spec-list{height:580px}}
"""


JS = r"""
const D=JSON.parse(document.getElementById('payload').textContent),COL={context:'#285d7d',answer:'#b96e16',select:'#b73529'};let view={role:'context',pc:0},selected=null;const cycles={};const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const fmt=(x,n=2)=>Number(x).toFixed(n),axis=(role,pc)=>D.axes[role][pc],row=ci=>D.rows[String(ci)];
function match(){return view.role==='context'?D.matches[view.pc]:D.matches.find(m=>m.answer_pc===view.pc)}
function cycleKey(role,pc,slot){return `${role}:${pc}:${slot}`}function candidate(role,pc,slot){const g=axis(role,pc).specimens[slot],i=cycles[cycleKey(role,pc,slot)]||0;return g.candidates[i%g.candidates.length]}
function renderNav(){for(const role of ['context','answer']){const root=$(role+'-nav');root.innerHTML=`<div class="nav-label">${role} PCA</div>`+D.axes[role].map((a,i)=>`<button type="button" data-role="${role}" data-pc="${i}" class="${view.role===role&&view.pc===i?'active':''}">${role==='context'?'C':'A'}-PC${a.id}<span>${(a.evr*100).toFixed(2)}% variance</span></button>`).join('');root.querySelectorAll('button').forEach(b=>b.onclick=()=>{view={role:b.dataset.role,pc:Number(b.dataset.pc)};selected=null;render()})}}
function renderMatrix(){const root=$('matrix'),max=Math.max(...D.loading_cosine.flat().map(Math.abs)),assigned=new Set(D.matches.map(m=>`${m.context_pc}:${m.answer_pc}`));root.innerHTML='<thead><tr><th></th>'+D.axes.answer.map(a=>`<th>A${a.id}</th>`).join('')+'</tr></thead><tbody>'+D.loading_cosine.map((line,i)=>`<tr><th>C${i+1}</th>`+line.map((v,j)=>{const alpha=.08+.62*Math.abs(v)/max,color=v<0?'185,110,22':'40,93,125';return `<td class="${assigned.has(i+':'+j)?'match':''}" style="background:rgba(${color},${alpha})" title="loading cosine ${fmt(v,3)}">${v>=0?'+':''}${fmt(v,2)}</td>`}).join('')+'</tr>').join('')+'</tbody>'}
function renderStats(){const m=match(),c=axis('context',m.context_pc),a=axis('answer',m.answer_pc),sign=m.answer_sign===-1?' × −1':'';$('stats-body').innerHTML=`<tr><th>matched axes</th><td class="c">C-PC${c.id}</td><td class="a">A-PC${a.id}${sign}</td><td>loading cos ${fmt(m.loading_cosine_aligned)}</td></tr><tr><th>native EVR</th><td class="c">${fmt(c.evr*100)}%</td><td class="a">${fmt(a.evr*100)}%</td><td>optimal 1:1 match</td></tr><tr><th>display mean ± SD</th><td class="c">${fmt(c.mean)} ± ${fmt(c.std)}</td><td class="a">${fmt(a.mean)} ± ${fmt(a.std)}</td><td>paired r ${fmt(m.paired_r_aligned)}</td></tr><tr><th>standardized relation</th><td class="c">context z</td><td class="a">aligned answer z</td><td>z RMSE ${fmt(m.paired_z_rmse)}</td></tr>`}
function renderAnalysis(){const active=axis(view.role,view.pc),m=match(),partnerRole=view.role==='context'?'answer':'context',partnerPc=view.role==='context'?m.answer_pc:m.context_pc,partner=axis(partnerRole,partnerPc),name=`${view.role==='context'?'C':'A'}-PC${active.id}`;$('analysis-title').textContent=`${name}: separate ${view.role} basis`;$('analysis-summary').textContent=`${name} explains ${(active.evr*100).toFixed(2)}% of ${view.role}-only variance. Its strongest measured full-text surface correlate is ${active.strongest_text_correlate} (rho=${fmt(active.strongest_text_rho)}).`;$('analysis-list').innerHTML=[`Optimal loading match: C-PC${m.context_pc+1} ↔ A-PC${m.answer_pc+1}; aligned loading cosine ${fmt(m.loading_cosine_aligned)}.`,`Paired score correlation after sign orientation is r=${fmt(m.paired_r_aligned)}; separate PCA scores were standardized before drawing the arrow.`,`Matched ${partnerRole} axis is ${partnerRole==='context'?'C':'A'}-PC${partner.id}, explaining ${(partner.evr*100).toFixed(2)}% of ${partnerRole}-only variance.`,`PC numbering is native to each fit; same index does not mean same direction. Full text is displayed without a character cap.`].map(x=>`<li>${esc(x)}</li>`).join('')}
function setSelected(role,pc,slot){selected={role,pc,slot,ci:candidate(role,pc,slot).ci};renderLanes();renderDetail();drawHist()}
function cycle(ev,role,pc,slot){ev.stopPropagation();const k=cycleKey(role,pc,slot),g=axis(role,pc).specimens[slot];cycles[k]=((cycles[k]||0)+1)%g.candidates.length;setSelected(role,pc,slot)}
function renderLane(role,pc){const a=axis(role,pc),root=$(role+'-list');$(role+'-title').textContent=`${role==='context'?'C':'A'}-PC${a.id} full ${role} specimens`;$(role+'-note').textContent=`native ${role}-only scores · complete text · 3 alternatives per quantile`;root.innerHTML=a.specimens.map((g,slot)=>{const c=candidate(role,pc,slot),r=row(c.ci),is=selected&&selected.ci===c.ci;return `<article class="specimen${is?' selected':''}" tabindex="0" data-slot="${slot}"><div class="spec-top"><span>q ${g.label} · ci ${c.ci}</span><button type="button" class="another">another ${((cycles[cycleKey(role,pc,slot)]||0)%3)+1}/3</button></div><div class="spec-score">${c.score>=0?'+':''}${fmt(c.score,3)}</div><pre class="spec-text">${esc(r[role])}</pre></article>`}).join('');root.querySelectorAll('.specimen').forEach(card=>{const slot=Number(card.dataset.slot);card.onclick=()=>setSelected(role,pc,slot);card.onkeydown=ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();setSelected(role,pc,slot)}};card.querySelector('.another').onclick=ev=>cycle(ev,role,pc,slot)})}
function renderLanes(){const m=match();renderLane('context',m.context_pc);renderLane('answer',m.answer_pc)}
function renderDetail(){if(!selected){const m=match(),role=view.role,pc=role==='context'?m.context_pc:m.answer_pc;setSelected(role,pc,3);return}const r=row(selected.ci),m=match();$('detail-ci').textContent=`ci ${r.ci}`;$('detail-role').textContent=`selected from ${selected.role==='context'?'C':'A'}-PC${selected.pc+1}`;$('detail-c').textContent=r.c.map(x=>fmt(x,3)).join(' / ');$('detail-a').textContent=r.a.map(x=>fmt(x,3)).join(' / ');$('detail-context').textContent=r.context;$('detail-answer').textContent=r.answer;$('detail-match').textContent=`arrow C-PC${m.context_pc+1} → A-PC${m.answer_pc+1}${m.answer_sign===-1?' × −1':''}`}
function drawHist(){const m=match(),cv=$('hist'),box=cv.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.round(box.width*dpr),h=Math.round(box.height*dpr);cv.width=w;cv.height=h;const ctx=cv.getContext('2d'),x0=76*dpr,x1=w-22*dpr,top=23*dpr,laneH=52*dpr,gap=28*dpr,sx=z=>x0+(Math.max(-3.5,Math.min(3.5,z))+3.5)/7*(x1-x0);ctx.clearRect(0,0,w,h);ctx.font=`${10*dpr}px ui-monospace,monospace`;ctx.textBaseline='middle';function lane(counts,y,color,label){const max=Math.max(...counts),bw=(x1-x0)/counts.length;ctx.fillStyle=color;ctx.globalAlpha=.72;counts.forEach((n,i)=>{const bh=n/max*(laneH-12*dpr);ctx.fillRect(x0+i*bw,y+laneH-bh,bw-.5*dpr,bh)});ctx.globalAlpha=1;ctx.fillStyle='#5f605a';ctx.textAlign='right';ctx.fillText(label,x0-8*dpr,y+laneH/2);ctx.strokeStyle='#c9c5b9';ctx.beginPath();ctx.moveTo(x0,y+laneH);ctx.lineTo(x1,y+laneH);ctx.stroke()}lane(m.hist_context,top,COL.context,`C-PC${m.context_pc+1}`);lane(m.hist_answer,top+laneH+gap,COL.answer,`A-PC${m.answer_pc+1}${m.answer_sign===-1?' ×−1':''}`);ctx.textAlign='center';ctx.fillStyle='#66675f';for(let z=-3;z<=3;z+=1.5){const x=sx(z);ctx.fillText(fmt(z,1),x,h-10*dpr);ctx.strokeStyle='#d8d4c9';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+laneH*2+gap);ctx.stroke()}if(selected){const r=row(selected.ci),cz=(r.c[m.context_pc]-m.context_mean)/m.context_std,az=(r.a[m.answer_pc]*m.answer_sign-m.answer_aligned_mean)/m.answer_aligned_std,xc=sx(cz),xa=sx(az),yc=top+laneH/2,ya=top+laneH+gap+laneH/2,ang=Math.atan2(ya-yc,xa-xc),head=8*dpr;ctx.strokeStyle=COL.select;ctx.fillStyle=COL.select;ctx.lineWidth=1.5*dpr;ctx.beginPath();ctx.moveTo(xc,yc);ctx.lineTo(xa,ya);ctx.stroke();ctx.beginPath();ctx.moveTo(xa,ya);ctx.lineTo(xa-head*Math.cos(ang-.48),ya-head*Math.sin(ang-.48));ctx.lineTo(xa-head*Math.cos(ang+.48),ya-head*Math.sin(ang+.48));ctx.closePath();ctx.fill();ctx.beginPath();ctx.arc(xc,yc,4*dpr,0,Math.PI*2);ctx.fill();ctx.textAlign='left';ctx.fillText('context → answer',Math.min(xc,xa)+6*dpr,(yc+ya)/2)}}
function render(){renderNav();renderMatrix();renderStats();renderAnalysis();renderLanes();if(!selected)renderDetail();else{renderDetail();drawHist()}}addEventListener('resize',drawHist);render();
"""


def page(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    json.loads(payload)
    meta = data["meta"]
    filters = ", ".join(f"{key}: {value}" for key, value in meta["public_filter_counts"].items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Separate PCA full-text specimens | issue #779</title><style>{CSS}</style></head><body><div class="wrap">
<header><div><h1>Separate context and answer PCA · full-text specimens</h1><div class="lede">Two independent L{meta["layer"]} PCA-10 fits: 200,000 context vectors define C-PC1…10 and the same 200,000 rows’ answer vectors define A-PC1…10. Native tabs stay separate. The relationship view optimally matches loading vectors, sign-orients the answer axis, standardizes each side, and draws the selected context → answer pair.</div></div><div class="legend"><span><i class="sw c"></i>context basis</span><span><i class="sw a"></i>answer basis</span></div></header>
<nav class="basis-nav" aria-label="Separate PCA axes"><div class="nav-row" id="context-nav"></div><div class="nav-row" id="answer-nav"></div></nav>
<section class="matrix-wrap"><div class="matrix-copy"><b>Cross-basis loading cosine</b>Every cell compares two loading vectors in the original 3,584-dimensional space. Outlined cells are the optimal one-to-one matching; color distinguishes positive and negative orientation.</div><div class="matrix-scroll"><table class="matrix" id="matrix"></table></div></section>
<section class="overview"><div class="distribution"><h2>Matched pair · standardized distributions</h2><canvas id="hist" aria-label="Matched separate-PCA distributions with selected context-to-answer arrow"></canvas><div class="stats-scroll"><table class="stats"><tbody id="stats-body"></tbody></table></div></div><aside class="analysis"><h2 id="analysis-title"></h2><p id="analysis-summary"></p><ul id="analysis-list"></ul></aside></section>
<section class="lanes"><div class="lane"><div class="lane-head"><h2 id="context-title"></h2><p id="context-note"></p></div><div class="spec-list" id="context-list"></div></div><div class="lane"><div class="lane-head"><h2 id="answer-title"></h2><p id="answer-note"></p></div><div class="spec-list" id="answer-list"></div></div></section>
<section class="detail"><div class="detail-grid"><div><h2>Selected full pair</h2><div class="detail-meta"><b id="detail-ci"></b><br><span id="detail-role"></span><br><span id="detail-match"></span><br><br>C-PC1…10<br><b id="detail-c"></b><br>A-PC1…10<br><b id="detail-a"></b></div></div><div><h2>Full context</h2><p class="detail-text" id="detail-context"></p></div><div><h2>Full answer</h2><p class="detail-text" id="detail-answer"></p></div></div></section>
<p class="foot">No dashboard character cap: {meta["recovered_context"]:,} previously capped WildChat context excerpts and {meta["recovered_answer"]:,} previously capped answer excerpts were replaced from pinned raw-completion files. The longest retained full context is {meta["max_full_context_chars"]:,} characters; the longest answer is {meta["max_full_answer_chars"]:,}. Text specimens use {meta["n_wildchat"]:,} publication-safe WildChat rows; all {meta["n_lmsys"]:,} LMSYS texts remain withheld. Full-text safety filtering removed {meta["n_public_filtered"]:,} of {meta["n_raw_pairs"]:,} sampled pairs ({filters or "no matches"}). Distribution sample: {meta["n_chunks"]} fixed chunks spanning shards {meta["shard_min"]:02d}–{meta["shard_max"]:02d}, not a uniform random draw. Context EVR10 {meta["context_evr10"] * 100:.2f}%; answer EVR10 {meta["answer_evr10"] * 100:.2f}%. Model <code>{meta["model_sha256"][:12]}</code> · fit producer <code>{meta["model_producer_commit"][:12]}</code> · renderer <code>{meta["render_commit"]}</code> · {meta["generated_utc"]}. <a href="https://huggingface.co/datasets/allenai/WildChat">WildChat attribution (ODC-BY)</a>.</p>
</div><script id="payload" type="application/json">{payload}</script><script>{JS}</script></body></html>"""


def report(data: dict) -> str:
    meta = data["meta"]
    lines = [
        "# Issue #779 separate context/answer PCA: full-text exploratory analysis",
        "",
        f"Generated: {meta['generated_utc']}",
        "",
        "## Title and metadata",
        "",
        f"- Separate PCA model SHA-256: `{meta['model_sha256']}`",
        f"- Fit producer: `{meta['model_producer_commit']}`; renderer: `{meta['render_commit']}`",
        f"- Fit: {meta['n_fit_rows_per_basis']:,} identical row IDs per basis at layer {meta['layer']}",
        f"- Context PCA-10 EVR: {meta['context_evr10'] * 100:.2f}%; answer PCA-10 EVR: {meta['answer_evr10'] * 100:.2f}%",
        f"- Display rows: {meta['n_pairs']:,} public-safe paired observations",
        "",
        "## Structure and quality",
        "",
        "The context and answer PCA models were fit independently on the same deterministic 200,000-row capture sample. Both component matrices are orthonormal, all display projections are finite, the fit row IDs are unique and identical across roles, and the model SHA matches its clean tracked-code manifest.",
        "",
        f"Full raw text was recovered for all {meta['n_raw_pairs']:,} display pairs before publication filtering. This replaced {meta['recovered_context']:,} capped context excerpts and {meta['recovered_answer']:,} capped answer excerpts. The public payload contains complete text only for the {meta['n_specimen_rows']:,} unique rows selected as specimens; no WildChat string contains the producer truncation marker. LMSYS text remains withheld.",
        "",
        f"Full-text safety gates removed {meta['n_public_filtered']:,} rows. Retained WildChat maximum lengths are {meta['max_full_context_chars']:,} context characters and {meta['max_full_answer_chars']:,} answer characters.",
        "",
        "## Cross-basis component matching",
        "",
        "Separate PC numbers are not shared coordinates. The table uses the Hungarian assignment to maximize total absolute loading-vector cosine over the top ten components; answer signs are oriented toward their matched context loadings only for the relationship display.",
        "",
        "| Context PC | Answer PC | raw loading cosine | aligned cosine | paired score r | paired z RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for match in data["matches"]:
        lines.append(
            f"| C-PC{match['context_pc'] + 1} | A-PC{match['answer_pc'] + 1} "
            f"{'× −1' if match['answer_sign'] == -1 else ''} | {match['loading_cosine_raw']:+.3f} | "
            f"{match['loading_cosine_aligned']:.3f} | {match['paired_r_aligned']:+.3f} | "
            f"{match['paired_z_rmse']:.3f} |"
        )
    for role in ("context", "answer"):
        lines.extend(
            [
                "",
                f"## {role.title()}-only axes",
                "",
                "| PC | EVR | display mean ± SD | strongest full-text correlate | rho | median / p90 / max chars |",
                "|---|---:|---:|---|---:|---:|",
            ]
        )
        for axis in data["axes"][role]:
            chars = axis["full_text_chars"]
            lines.append(
                f"| {role[0].upper()}-PC{axis['id']} | {axis['evr'] * 100:.2f}% | "
                f"{axis['mean']:+.2f} ± {axis['std']:.2f} | {axis['strongest_text_correlate']} | "
                f"{axis['strongest_text_rho']:+.3f} | {chars['median']:.0f} / {chars['p90']:.0f} / {chars['max']:,} |"
            )
    matched = [abs(match["loading_cosine_raw"]) for match in data["matches"]]
    paired = [match["paired_r_aligned"] for match in data["matches"]]
    n_same_index = sum(match["context_pc"] == match["answer_pc"] for match in data["matches"])
    lines.extend(
        [
            "",
            "## Key findings",
            "",
            f"The independently fit bases are not close to a component-wise identity: matched absolute loading cosine ranges from {min(matched):.3f} to {max(matched):.3f}. {n_same_index} of ten optimal matches use the same numerical index. This is why raw C-PCk and A-PCk scores should not be subtracted or interpreted as one shared axis.",
            "",
            f"Paired correlations after loading alignment range from {min(paired):+.3f} to {max(paired):+.3f}. Loading similarity and paired score correlation answer different questions: the first compares directions in hidden-feature space; the second compares paired observations after projection.",
            "",
            "The full-text feature correlations are materially more trustworthy than the capped-answer analysis for length, but they remain surface-form diagnostics rather than semantic labels. Language/script, repeated prompt families, code formatting, and corpus composition remain entangled.",
            "",
            "## Recommendations and interpretation limits",
            "",
            f"The display sample uses {meta['n_chunks']} fixed chunks spanning shards {meta['shard_min']:02d}–{meta['shard_max']:02d}; it is useful for specimen inspection but not population-frequency estimation. Browse multiple alternatives at each quantile before naming an axis.",
            "",
            "For a formal comparison, estimate subspace overlap at several ranks with held-out bootstrap intervals and repeat the separate fits under independent row samples. The present loading assignment is a descriptive top-10 alignment, not evidence that a context PC causally becomes its matched answer PC.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--chunks", nargs="+", default=list(DEFAULT_CHUNKS))
    parser.add_argument("--out-name", default=OUT_NAME)
    parser.add_argument("--public-dir", type=Path, default=Path("dashboard/public"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments/dashboards"))
    args = parser.parse_args()
    model, model_meta = load_models(args.model_dir)
    rows, sample_meta = load_fulltext_rows(args.export_dir, tuple(args.chunks), model)
    data = build_data(rows, sample_meta, model, model_meta)
    html = page(data)
    markdown = report(data)
    for out_dir in (args.public_dir, args.experiments_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / args.out_name
        out.write_text(html, encoding="utf-8")
        print(f"[separate-pca-dashboard] wrote {out} ({len(html.encode()) / 1e6:.2f} MB)")
    report_path = args.experiments_dir / REPORT_NAME
    report_path.write_text(markdown, encoding="utf-8")
    print(f"[separate-pca-dashboard] wrote {report_path}")


if __name__ == "__main__":
    main()
