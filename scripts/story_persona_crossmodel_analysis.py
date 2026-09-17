#!/usr/bin/env python3
"""Analyze description-only context banks against published DeepSeek tracer rates.

No vectors are centered. Full-bank whitening is transductive; each cross-half
whitener is fit on individual rows disjoint from its evaluated question means.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402
from omegaconf import MISSING, DictConfig, OmegaConf  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from scripts.story_persona_metric_reanalysis import (  # noqa: E402
    cosine_from_gram,
    metric_gram,
    select_ridge,
)
from scripts.story_persona_qwen38_pilot import (  # noqa: E402
    digest,
    file_digest,
    read_checksums,
    read_manifest,
    validate_chunk,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["hhh", "fred", "helpful", "dismissive", "sarcastic", "saboteur", "peer", "help_seeker"]
ALTERNATIVES = NAMES[3:]
EXPECTED_QUESTIONS = 240
SOURCE_IMAGE_SHA256 = "0d80dd99ecfaf97e049215a1cca04c65232e466009999845186230b70e3f483f"
MODELS = {
    "qwen": {
        "id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "layers": 64,
        "hidden_dim": 5120,
        "selected_layers": [15, 31, 47, 63],
    },
    "deepseek": {
        "id": "deepseek-ai/DeepSeek-V3.1-Base",
        "revision": "d3d4eafdc470de44bbf6f0a74f852eb522357be8",
        "layers": 61,
        "hidden_dim": 7168,
        "selected_layers": [15, 30, 45, 60],
    },
}


@dataclass
class AnalysisConfig:
    """Explicit CLI defaults; ordinary Hydra overrides need no leading plus."""

    output_dir: str = MISSING
    model_key: str = "qwen"
    phase: str = "analyze"
    rates_path: str = "eval_results/issue_2673/deepseek_comparison/published_rates_and_overlap.json"
    prompts_path: str = "configs/pilots/story_persona_deepseek_prompts.json"
    questions_path: str = "data/assistant_axis/extraction_questions.jsonl"


ConfigStore.instance().store(name="story_persona_crossmodel_analysis", node=AnalysisConfig)


def load_rates(path: Path) -> dict:
    """Require every reviewed aggregate, with its raster coordinate and bound."""
    source = json.loads(path.read_text())
    if source["source_sha256"] != SOURCE_IMAGE_SHA256:
        raise ValueError("published outcome image is not the reviewed source")
    rates = source["rates"]
    if set(rates) != {"hhh", "fred"}:
        raise ValueError("published evaluation-persona coverage mismatch")
    for persona, bottom in (("hhh", 490), ("fred", 1166)):
        if set(rates[persona]) != set(ALTERNATIVES):
            raise ValueError("published story-pair coverage mismatch")
        for pair in ALTERNATIVES:
            if set(rates[persona][pair]) != {"helpful", "other"}:
                raise ValueError("published two-rate coverage mismatch")
            for value in rates[persona][pair].values():
                rate, y, bound = (value["rate"], value["bar_top_y"], value["digitization_bound"])
                if (
                    not np.isfinite([rate, y, bound]).all()
                    or not 0 <= rate <= 1
                    or abs(rate - (bottom - y) / 438) > 1e-12
                    or abs(bound - 1.5 / 438) > 1e-12
                ):
                    raise ValueError("invalid published rate/coordinate/digitization bound")
    return source


def association(x: list[float], y: list[float]) -> dict:
    """Report descriptive correlations, explicitly retaining undefined constants."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2 or not np.isfinite([x, y]).all():
        raise ValueError("invalid paired observations")
    constant = np.ptp(x) == 0 or np.ptp(y) == 0
    return {
        "n": len(x),
        "pearson_r": None if constant else float(np.corrcoef(x, y)[0, 1]),
        "spearman_rho": (None if constant else float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])),
        "correlation_status": "undefined_constant_input" if constant else "defined",
    }


def paired_outcomes(cosine: np.ndarray, names: list[str], rates: dict) -> dict:
    """Keep ten signed preference contrasts and secondary alternative uptake."""
    cosine = np.asarray(cosine, dtype=np.float64)
    if (
        len(names) != 8
        or set(names) != set(NAMES)
        or cosine.shape != (8, 8)
        or not np.isfinite(cosine).all()
        or np.max(np.abs(cosine)) > 1 + 1e-8
    ):
        raise ValueError("invalid eight-persona cosine bank")
    pairs = []
    strata = {}
    for persona in ("hhh", "fred"):
        e, helpful = names.index(persona), names.index("helpful")
        subset = []
        for alternative in ALTERNATIVES:
            row = rates[persona][alternative]
            d = float(cosine[e, helpful] - cosine[e, names.index(alternative)])
            z = row["helpful"]["rate"] - row["other"]["rate"]
            subset.append(
                {
                    "evaluation_persona": persona,
                    "alternative": alternative,
                    "helpful_similarity": float(cosine[e, helpful]),
                    "other_similarity": float(cosine[e, names.index(alternative)]),
                    "predictor_contrast": d,
                    "helpful_rate": row["helpful"]["rate"],
                    "other_rate": row["other"]["rate"],
                    "behavioral_contrast": z,
                    "behavioral_contrast_digitization_bound": (
                        row["helpful"]["digitization_bound"] + row["other"]["digitization_bound"]
                    ),
                    "sign_agrees": bool(np.sign(d) == np.sign(z)),
                }
            )
        pairs.extend(subset)
        strata[persona] = {
            **association(
                [p["predictor_contrast"] for p in subset],
                [p["behavioral_contrast"] for p in subset],
            ),
            "sign_agreement_count": sum(p["sign_agrees"] for p in subset),
            "sign_agreement_denominator": 5,
            "zero_predictor_count": sum(p["predictor_contrast"] == 0 for p in subset),
            "zero_outcome_count": sum(p["behavioral_contrast"] == 0 for p in subset),
            "secondary_other_uptake": association(
                [p["other_similarity"] for p in subset], [p["other_rate"] for p in subset]
            ),
        }
    return {
        "pairs": pairs,
        "primary_by_persona": strata,
        "supplementary_pooled": {
            **association(
                [p["predictor_contrast"] for p in pairs],
                [p["behavioral_contrast"] for p in pairs],
            ),
            "interpretation": "descriptive only; shared Helpful and paired story models",
        },
        "helpful_predictor_note": "Helpful similarity is constant across five pairs per persona.",
        "sign_rule": "exact sign equality; zero counts reported separately; no tolerance fitted",
    }


def validate_layout(manifest: dict, done: dict, rows: list[dict], model_key: str) -> dict:
    """Reject wrong model pins, duplicate cells, incomplete chunks, and stale IDs."""
    if model_key not in MODELS:
        raise ValueError(f"unknown model_key: {model_key}")
    spec, model = manifest["spec"], MODELS[model_key]
    for key in ("id", "revision", "layers", "hidden_dim"):
        if spec["model"][key] != model[key]:
            raise ValueError(f"model specification mismatch: {key}")
    if (
        done["fingerprint"] != manifest["fingerprint"]
        or done["row_count"] != len(rows)
        or digest(rows) != spec["inputs_sha256"]
        or spec["dtype"] != "bfloat16"
        or spec["layers"] != list(range(model["layers"]))
        or spec["position"] != "last_generation_prefix_token"
        or spec["final_layer"] != "pre_final_norm"
    ):
        raise ValueError("capture completion, rows, dtype, or layer-location mismatch")
    names, questions = [p["id"] for p in spec["prompts"]], spec["question_ids"]
    if len(names) != 8 or set(names) != set(NAMES):
        raise ValueError("expected exactly eight published persona descriptions")
    if len(questions) != EXPECTED_QUESTIONS or len(set(questions)) != len(questions):
        raise ValueError("question coverage mismatch")
    if len(rows) != len(names) * len(questions):
        raise ValueError("row count is not the declared Cartesian product")
    observed = [(r["persona"], r["question_id"]) for r in rows]
    expected = {(p, q) for p in names for q in questions}
    if set(observed) != expected or len(set(observed)) != len(rows):
        raise ValueError("duplicate or missing persona-question cell")
    if any(r["row_id"] != f"{r['persona']}:{r['question_id']}" for r in rows):
        raise ValueError("row ID disagrees with persona/question")
    batches = spec["batches"]
    if (
        not batches
        or any(not b for b in batches)
        or any(type(i) is not int for b in batches for i in b)
        or sorted(i for b in batches for i in b) != list(range(len(rows)))
    ):
        raise ValueError("batch row coverage is not exact")
    files = {f"batch_{k:04d}.pt" for k in range(len(batches))}
    if set(done["chunk_sha256"]) != files:
        raise ValueError("completion chunk coverage mismatch")
    half = len(questions) // 2
    if half * 2 != len(questions):
        raise ValueError("question halves must have equal sizes")
    question_index = {q: i for i, q in enumerate(questions)}
    return {
        "names": names,
        "question_ids": questions,
        "persona_index": np.array([names.index(r["persona"]) for r in rows]),
        "half_index": np.array([question_index[r["question_id"]] // half for r in rows]),
        "selected_layers": model["selected_layers"],
        "layers": model["layers"],
        "hidden_dim": model["hidden_dim"],
    }


def save_npz(path: Path, **values) -> None:
    """Checkpoint an uncompressed tensor bundle without a second filename suffix."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **values)
    tmp.replace(path)


def stream_vectors(out: Path, manifest: dict, layout: dict, checksums: dict, progress) -> tuple:
    """Accumulate FP64 means while retaining only four individual-vector layers."""
    spec = manifest["spec"]
    rows, layers, width = len(layout["persona_index"]), layout["layers"], layout["hidden_dim"]
    sums = np.zeros((16, layers, width), dtype=np.float64)
    counts = np.zeros(16, dtype=np.int64)
    vectors = np.empty((rows, len(layout["selected_layers"]), width), dtype=np.float32)
    seen = np.zeros(rows, dtype=np.int64)
    for k, indices in enumerate(spec["batches"]):
        path = out / "chunks" / f"batch_{k:04d}.pt"
        chunk = validate_chunk(
            path,
            manifest["fingerprint"],
            indices,
            (len(indices), layers, width),
            expected_sha256=checksums[path.name],
        )
        values = chunk["vectors"].float().numpy()
        slots = layout["half_index"][indices] * 8 + layout["persona_index"][indices]
        np.add.at(sums, slots, values.astype(np.float64))
        np.add.at(counts, slots, 1)
        vectors[indices] = values[:, layout["selected_layers"]]
        seen[indices] += 1
        progress("stream", completed_chunks=k + 1, total_chunks=len(spec["batches"]))
    counts = counts.reshape(2, 8)
    if not np.all(seen == 1) or not np.all(counts == len(layout["question_ids"]) // 2):
        raise ValueError("tensor/half coverage mismatch")
    sums = sums.reshape(2, 8, layers, width)
    halves = sums / counts[:, :, None, None]
    centroids = sums.sum(0) / counts.sum(0)[:, None, None]
    return centroids, halves, vectors, counts


def primal_subproblem_oracle(x: np.ndarray, means: np.ndarray, ridge: float) -> dict:
    """Check the live dual solver against dense primal Cholesky on fixed coordinates.

    This small real-data subproblem is an implementation oracle, not an
    independent full-dimensional refit; the main fit also has its full residual.
    """
    rows = np.arange(min(32, len(x)))
    columns = np.linspace(0, x.shape[1] - 1, min(64, x.shape[1]), dtype=int)
    small, c = x[np.ix_(rows, columns)], means[:, columns]
    moment = small.T @ small / len(small) + ridge * np.eye(len(columns))
    transformed = np.linalg.solve(np.linalg.cholesky(moment), c.T).T
    reference = transformed @ transformed.T
    actual, residual = metric_gram(small, c, ridge)
    error = float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-300))
    if not np.isfinite(error) or error > 1e-8:
        raise ValueError(f"dense primal subproblem oracle failed: {error}")
    return {
        "scope": "fixed real-data row/coordinate subproblem, not a full-dimensional refit",
        "fit_row_indices_relative": rows.tolist(),
        "coordinate_indices": columns.tolist(),
        "gram_relative_error": error,
        "implicit_primal_relative_residual": residual,
        "tolerance": 1e-8,
    }


def fit_bank(x: np.ndarray, means: np.ndarray, names: list[str], rates: dict, progress) -> dict:
    """Fit one uncentered metric using fit rows only, and score its evaluation means."""
    x, means = np.asarray(x, dtype=np.float64), np.asarray(means, dtype=np.float64)
    progress("select_ridge")
    ridge, diagnostics = select_ridge(x)
    progress("metric_solve")
    gram, residual = metric_gram(x, means, ridge)
    raw = cosine_from_gram(means @ means.T)
    white = cosine_from_gram(gram)
    diagnostics["lambda_over_largest_eigenvalue"] = (
        ridge / diagnostics["largest_second_moment_eigenvalue"]
        if diagnostics["largest_second_moment_eigenvalue"] > 0
        else None
    )
    progress("primal_subproblem_oracle")
    return {
        "ridge": ridge,
        "diagnostics": diagnostics,
        "implicit_primal_relative_residual": residual,
        "primal_subproblem_oracle": primal_subproblem_oracle(x, means, ridge),
        "raw_cosine_matrix": raw.tolist(),
        "whitened_cosine_matrix": white.tolist(),
        "raw": paired_outcomes(raw, names, rates),
        "whitened": paired_outcomes(white, names, rates),
    }


def analyze(cfg: DictConfig) -> dict:
    """Verify the capture, persist all means, then checkpoint every fixed-layer fold."""
    if cfg.phase != "analyze":
        raise ValueError("only phase=analyze is supported")
    out = Path(cfg.output_dir).resolve()
    (out / "analysis_complete.json").unlink(missing_ok=True)
    manifest = read_manifest(out / "manifest.json")
    done = json.loads((out / "capture_complete.json").read_text())
    rows = json.loads((out / "rows.json").read_text())
    layout = validate_layout(manifest, done, rows, cfg.model_key)
    spec = manifest["spec"]
    checksums = read_checksums(out, manifest["fingerprint"])
    if checksums != done["chunk_sha256"]:
        raise ValueError("capture checksum records differ")
    if {p.name for p in (out / "chunks").glob("*.pt")} != set(checksums):
        raise ValueError("on-disk chunk coverage mismatch")
    prompt_path, question_path, rates_path = (
        ROOT / cfg.prompts_path,
        ROOT / cfg.questions_path,
        ROOT / cfg.rates_path,
    )
    if spec["prompts"] != json.loads(prompt_path.read_text())["prompts"]:
        raise ValueError("captured prompts differ from the published description bank")
    questions = [json.loads(line) for line in question_path.read_text().splitlines()]
    if spec["question_ids"] != [q["id"] for q in questions]:
        raise ValueError("captured questions differ from the shared ordered bank")
    by_question = {q["id"]: q["question"] for q in questions}
    by_persona = {p["id"]: p["system"] for p in spec["prompts"]}
    for row in rows:
        ids = row["input_ids"]
        description, question = by_persona[row["persona"]], by_question[row["question_id"]]
        if (
            not ids
            or any(type(token) is not int or token < 0 for token in ids)
            or row["prefix_sha256"] != digest(ids)
            or row["token_count"] != len(ids)
            or row["final_token_id"] != ids[-1]
            or len(ids) > spec["capture"]["max_context_tokens"]
            or row["description"] != description
            or row["question"] != question
            or row["messages"]
            != [{"role": "system", "content": description}, {"role": "user", "content": question}]
            or description not in row["rendered_prefix"]
            or question not in row["rendered_prefix"]
        ):
            raise ValueError("row text/token provenance or no-truncation contract changed")
        if cfg.model_key == "deepseek" and row["rendered_prefix"] != (
            description + "\n\nHuman: " + question + "\n\nAssistant:"
        ):
            raise ValueError("DeepSeek raw continuation wrapper changed")
    published = load_rates(rates_path)
    smoke = json.loads((out / "smoke.json").read_text())
    if (
        smoke["fingerprint"] != manifest["fingerprint"]
        or smoke.get("passed") is not True
        or smoke["throughput_gate"]["passed"] is not True
        or not 0 < done["attempt_started_at"] <= done["checked_at"]
    ):
        raise ValueError("capture lacks a successful fingerprint-matched numerical smoke")
    smoke_vectors = torch.load(out / "smoke_vectors.pt", map_location="cpu", weights_only=True)
    smoke_shape = (len(smoke["indices"]), layout["layers"], layout["hidden_dim"])
    if (
        smoke_vectors["fingerprint"] != manifest["fingerprint"]
        or smoke_vectors["indices"] != smoke["indices"]
        or any(
            tuple(smoke_vectors[key].shape) != smoke_shape
            or smoke_vectors[key].dtype != torch.bfloat16
            or not torch.isfinite(smoke_vectors[key]).all()
            for key in ("initial", "repeated", "batched")
        )
    ):
        raise ValueError("numerical smoke vector provenance/shape/dtype changed")
    del smoke_vectors
    paths = {
        "manifest.json": out / "manifest.json",
        "rows.json": out / "rows.json",
        "capture_complete.json": out / "capture_complete.json",
        "smoke.json": out / "smoke.json",
        "smoke_vectors.pt": out / "smoke_vectors.pt",
        "published_rates": rates_path,
        "published_descriptions": prompt_path,
        "question_bank": question_path,
    }
    if (out / "capture_chunks.json").exists():
        paths["capture_chunks.json"] = out / "capture_chunks.json"
    sources = [
        Path(__file__),
        ROOT / "scripts/story_persona_metric_reanalysis.py",
        ROOT / "scripts/story_persona_qwen38_pilot.py",
        ROOT / "src/explore_persona_space/analysis/leakage_predictor.py",
    ]
    provenance = {
        "capture_fingerprint": manifest["fingerprint"],
        "capture_provenance": manifest["provenance"],
        "model_key": cfg.model_key,
        "model": spec["model"],
        "config": OmegaConf.to_container(cfg, resolve=True),
        "input_sha256": {key: file_digest(path) for key, path in paths.items()},
        "chunk_sha256": checksums,
        "source_sha256": {str(p.relative_to(ROOT)): file_digest(p) for p in sources},
        "centering": "none; uncentered second moment, no vector pre-normalization",
        "fold_rule": "first 120 versus last 120 question IDs in the original bank order",
        "recipe_reference": "task2673; #665/#666 lambda grid and condition target",
    }
    analysis_fingerprint = digest(provenance)
    identity = {
        "fingerprint": manifest["fingerprint"],
        "analysis_fingerprint": analysis_fingerprint,
    }
    start = time.monotonic()
    progress_path = out / "progress.json"
    if progress_path.exists():
        previous = json.loads(progress_path.read_text())
        if previous["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("stale capture/analysis progress")
        if previous.get("phase") != "analyze":
            write_json(out / "capture_progress.json", previous)

    def progress(stage: str, **details) -> None:
        """Expose completed units and real phase transitions to the monitor."""
        record = {
            **identity,
            "phase": "analyze",
            "stage": stage,
            "checked_at": time.time(),
            "elapsed_seconds": time.monotonic() - start,
            **details,
        }
        write_json(out / "analysis_progress.json", record)
        write_json(progress_path, record)
        print(f"[analysis] {json.dumps(record, sort_keys=True)}", flush=True)

    progress("validated_inputs")
    centroids, halves, vectors, counts = stream_vectors(out, manifest, layout, checksums, progress)
    names = layout["names"]
    save_npz(
        out / "centroids.npz",
        centroids=centroids,
        half_centroids=halves,
        counts_by_half=counts,
        persona_names=names,
        layers=spec["layers"],
        question_ids=layout["question_ids"],
        **identity,
    )
    save_npz(
        out / "selected_vectors.npz",
        vectors=vectors,
        indices=np.arange(len(rows)),
        layers=layout["selected_layers"],
        row_ids=[r["row_id"] for r in rows],
        persona_index=layout["persona_index"],
        half_index=layout["half_index"],
        persona_names=names,
        question_ids=[r["question_id"] for r in rows],
        **identity,
    )
    progress("verification_inputs_saved")
    raw_all_layers = {}
    for layer in spec["layers"]:
        means = centroids[:, layer]
        matrix = cosine_from_gram(means @ means.T)
        raw_all_layers[str(layer)] = {
            "raw_cosine_matrix": matrix.tolist(),
            "raw": paired_outcomes(matrix, names, published["rates"]),
        }
    write_json(
        out / "raw_all_layers.json",
        {
            **identity,
            "provenance": provenance,
            "persona_names": names,
            "results": raw_all_layers,
        },
    )
    results = {}
    for j, layer in enumerate(layout["selected_layers"]):
        block = {
            **identity,
            "layer": layer,
            "persona_names": names,
            "status": "incomplete",
            "fits": {},
        }
        folds = [
            ("full_bank", None, None),
            ("fit_first_evaluate_last", 0, 1),
            ("fit_last_evaluate_first", 1, 0),
        ]
        for fold, fit_half, evaluate_half in folds:
            mask = (
                np.ones(len(rows), dtype=bool)
                if fit_half is None
                else layout["half_index"] == fit_half
            )
            means = (
                centroids[:, layer] if evaluate_half is None else halves[evaluate_half, :, layer]
            )
            result = fit_bank(
                vectors[mask, j],
                means,
                names,
                published["rates"],
                lambda stage: progress(stage, layer=layer, fold=fold),
            )
            result.update(
                fit_half=fit_half,
                evaluate_half=evaluate_half,
                fit_row_count=int(mask.sum()),
                evaluation_question_ids=(
                    layout["question_ids"]
                    if evaluate_half is None
                    else layout["question_ids"][
                        evaluate_half * EXPECTED_QUESTIONS // 2 : (evaluate_half + 1)
                        * EXPECTED_QUESTIONS
                        // 2
                    ]
                ),
                fit_question_ids=(
                    layout["question_ids"]
                    if fit_half is None
                    else layout["question_ids"][
                        fit_half * EXPECTED_QUESTIONS // 2 : (fit_half + 1)
                        * EXPECTED_QUESTIONS
                        // 2
                    ]
                ),
                interpretation=(
                    "transductive full-bank fit"
                    if fit_half is None
                    else "disjoint questions; same eight personas and domain"
                ),
            )
            block["fits"][fold] = result
            write_json(out / f"block_{layer}.json", block)
            progress("fold_complete", layer=layer, fold=fold)
        block["status"] = "complete"
        write_json(out / f"block_{layer}.json", block)
        results[str(layer)] = block
    # No successful sentinel may describe inputs that changed while analysis ran.
    if any(file_digest(path) != provenance["input_sha256"][key] for key, path in paths.items()):
        raise ValueError("analysis input changed during computation")
    for name, checksum in checksums.items():
        if file_digest(out / "chunks" / name) != checksum:
            raise ValueError(f"source chunk changed during analysis: {name}")
    summary = {
        **identity,
        "provenance": provenance,
        "model": spec["model"],
        "persona_names": names,
        "layers": spec["layers"],
        "selected_layers": layout["selected_layers"],
        "centering": "none",
        "question_count": EXPECTED_QUESTIONS,
        "row_count": len(rows),
        "counts_by_half": counts.tolist(),
        "published_rates": published["rates"],
        "published_source_url": published["source_url"],
        "published_source_sha256": published["source_sha256"],
        "raw_all_layers": raw_all_layers,
        "results": results,
        "token_counts_by_persona": {
            name: [r["token_count"] for r in rows if r["persona"] == name] for name in names
        },
        "limitations": [
            "Description-only prompts, not complete HHH/Fred few-shot evaluation contexts.",
            "Generic incoming questions differ from the behavior-generating Bloom distribution.",
            "Pre-finetuning geometry versus post-story-finetuning DeepSeek aggregate outcomes.",
            "Cross-model comparison changes model, native wrapper and numerical precision.",
            "Help-seeker disposition conflicts with the incoming-question target-speaker role.",
            "Five pairs per persona share Helpful similarity; pooled ten is supplementary only.",
            "Digitization bounds are not sampling confidence intervals; no inferential p-values.",
            "Full-bank fit is transductive; cross-half fits do not hold out personas or domains.",
            "The scalar outcome is tracer uptake, not a newly measured behavioral leakage score.",
        ],
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(out / "summary.json", summary)
    outputs = ["summary.json", "raw_all_layers.json", "centroids.npz", "selected_vectors.npz"] + [
        f"block_{layer}.json" for layer in layout["selected_layers"]
    ]
    write_json(
        out / "analysis_complete.json",
        {
            **identity,
            "checked_at": time.time(),
            "row_count": len(rows),
            "outputs": {
                name: {"sha256": file_digest(out / name), "bytes": (out / name).stat().st_size}
                for name in outputs
            },
        },
    )
    progress("complete", completed_layers=len(results))
    return summary


@hydra.main(version_base=None, config_path=None, config_name="story_persona_crossmodel_analysis")
def main(cfg: DictConfig) -> None:
    """Run only the CPU analysis phase; no model loading or remote writes."""
    analyze(cfg)


if __name__ == "__main__":
    main()
