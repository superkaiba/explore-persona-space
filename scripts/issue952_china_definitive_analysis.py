"""Frozen-map geometry, inference, and figures for issue #952.

All real-corpus operations log only counts, hashes, and aggregate metrics.  The
module also exposes pure helpers used by the synthetic unit tests.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.fft import dct
from huggingface_hub import HfApi, hf_hub_download
from scipy.stats import binomtest, rankdata, spearmanr

from explore_persona_space.orchestrate import hub

ISSUE = 952
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
LAYERS = (14, 19, 26)
N_DRAWS = 8
MASS_PRIMARY = 0.99
MASS_SENSITIVITY = (0.90, 0.999)
N_BOOT = 10_000
N_PERM = 10_000
N_RANDOM = 1_000
RANDOM_BLOCK = 25
SEED = 25_695
HIDDEN = 3584
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 2048
GENERATION_SEED_BASE = 952_000
GENERATION_SEED_NAMESPACE = "registered-production-v1"
GENERATION_SEED_FORMULA = f"{GENERATION_SEED_BASE} + prompt_index * {N_DRAWS} + draw"
GENERATION_SEED_POLICY = "disjoint-smoke-production-v1"
CONTEXT_POSITION = "context_last_generation_prompt_token"
ANSWER_POOLING = "completion_plus_im_end_newline_mean"
SERIALIZED_DTYPE = "fp32"
REGISTERED_PACKAGE_VERSIONS = {
    "torch": "2.8.0",
    "transformers": "4.57.6",
    "vllm": "0.11.0",
    "huggingface-hub": "0.36.2",
}
TOKENIZER_ARTIFACTS = {"config.json", "tokenizer.json", "tokenizer_config.json"}
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue952_position_divergence/followups/china_refusal_topic_stratified_bilingual_v1"
MAP_REV = "eef8eb1da43cd2212dfa73d8711fd64dc54c376f"
MAP_PREFIX = "issue779_monitoring/n1m_readout/weights"
HIST_REV = "3a774dcec5598963e4551c6ce03480270732291e"
HIST_FILES = {
    "issue2617_svmp/manifests/svmp_bank.json": "manifests/svmp_bank.json",
    "issue2617_svmp/raw_completions/judge/judge_scores.json": (
        "raw_completions/judge/judge_scores.json"
    ),
    "issue2617_svmp/analysis_tensors/va/va_langow_query_svmp.pt": (
        "analysis_tensors/va/va_langow_query_svmp.pt"
    ),
}
REGISTERED_SOURCE_ITEMS = 90
PROMPTS_PER_SOURCE = 12
REGISTERED_PROMPTS = REGISTERED_SOURCE_ITEMS * PROMPTS_PER_SOURCE


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _write_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **values)
    os.replace(tmp, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _judge_dir(run_dir: Path, attempt: int) -> Path:
    """Return the attempt-bound local production-judge directory."""
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return run_dir / "judge" / f"attempt{attempt}"


def _validated_judge_census(marker: dict[str, Any]) -> dict[str, str]:
    """Validate and return the complete schema-v1 judge artifact census."""
    census = marker.get("artifact_census")
    if marker.get("schema_version") != 1 or marker.get("kind") != (
        "issue952_codex_production_upload"
    ):
        raise RuntimeError("unsupported production judge upload-marker schema")
    if (
        not isinstance(census, dict)
        or not census
        or marker.get("artifact_census_sha256") != _sha_obj(census)
    ):
        raise RuntimeError("production judge artifact census is missing or hash-invalid")
    required = {
        "judge/wave_request_manifest.json",
        "judge/wave_packet_manifest.json",
        "judge/wave_lookup.json",
        "judge/wave_runtime_identity.json",
        "judge/wave_scores.jsonl",
        "judge/wave_summary.json",
        "judge/wave_overlap_joined.jsonl",
        "judge/production_stage.json",
    }
    for relative, sha in census.items():
        if not isinstance(relative, str):
            raise RuntimeError("production judge artifact census contains a non-string path")
        path = Path(relative)
        if (
            path.is_absolute()
            or path.as_posix() != relative
            or ".." in path.parts
            or path.parts[:1] != ("judge",)
            or not isinstance(sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha) is None
        ):
            raise RuntimeError("production judge artifact census contains an unsafe entry")
    if not required <= set(census):
        raise RuntimeError("production judge artifact census omits required analysis inputs")
    return census


def _accepted_bank_contract(bank: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    """Validate full-bank provenance and derive accepted analysis widths."""
    passing_ids = audit.get("passing_item_ids")
    if (
        audit.get("passed") is not True
        or len(bank) != REGISTERED_PROMPTS
        or len({row["item_id"] for row in bank}) != len(bank)
        or not isinstance(passing_ids, list)
        or len(passing_ids) != len(set(passing_ids))
        or audit.get("n_audit_passing_items") != len(passing_ids)
    ):
        raise RuntimeError("analysis accepted-bank provenance/cardinality gate failed")
    all_sources = {row["source_prompt_id"] for row in bank}
    passing = set(passing_ids)
    if (
        len(all_sources) != REGISTERED_SOURCE_ITEMS
        or not passing <= all_sources
        or any(
            sum(row["source_prompt_id"] == source_id for row in bank) != PROMPTS_PER_SOURCE
            for source_id in all_sources
        )
        or any(
            not isinstance(row.get("audit_pass"), bool)
            or row["audit_pass"] != (row["source_prompt_id"] in passing)
            for row in bank
        )
    ):
        raise RuntimeError("analysis bank flags or passing identities disagree with audit")
    accepted_rows = [row for row in bank if row["source_prompt_id"] in passing]
    rollout_ids = [f"{row['item_id']}-d{draw}" for row in accepted_rows for draw in range(N_DRAWS)]
    return {
        "accepted_rows": accepted_rows,
        "accepted_prompt_ids": [row["item_id"] for row in accepted_rows],
        "accepted_source_ids": sorted(passing_ids),
        "accepted_source_ids_sha256": _sha_obj(sorted(passing_ids)),
        "expected_rollout_ids": rollout_ids,
        "n_accepted_prompts": len(accepted_rows),
        "n_expected_rollouts": len(rollout_ids),
    }


def _generation_fingerprint(report: dict[str, Any]) -> str:
    """Recompute the GPU generation identity consumed by later phases."""
    return _sha_obj(
        {
            "regime_fp": report["regime_fp"],
            "rollouts_sha256": report["rollouts_sha256"],
            "n_prompts": report["n_prompts"],
            "n_rows": report["n_rows"],
            "ordered_item_ids_sha256": report["ordered_item_ids_sha256"],
        }
    )


def _capture_fingerprint(report: dict[str, Any]) -> str:
    """Recompute the GPU capture identity consumed by finalization."""
    return _sha_obj(
        {
            "capture_regime_fp": report["capture_regime_fp"],
            "generation_fingerprint": report["generation_fingerprint"],
            "vc_sha256": report["vc_sha256"],
            "va_files": report["va_files"],
            "n_contexts": report["n_contexts"],
            "n_answer_rows": report["n_answer_rows"],
        }
    )


def _is_sha256(value: Any) -> bool:
    """Return whether a value is a lowercase full SHA256 digest."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_registered_gpu_regime(
    *,
    attempt: int,
    accepted: dict[str, Any],
    input_marker: dict[str, Any],
    generation: dict[str, Any],
    capture: dict[str, Any],
) -> None:
    """Reject a self-consistent GPU payload that does not use the registered recipe."""
    regime = generation.get("regime")
    capture_regime = capture.get("capture_regime")
    if not isinstance(regime, dict) or not isinstance(capture_regime, dict):
        raise RuntimeError("GPU generation/capture regime metadata is missing")
    n_sources = len(accepted["accepted_source_ids"])
    expected_bank = {
        "accepted_source_ids": accepted["accepted_source_ids"],
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "n_accepted_source_items": n_sources,
        "n_accepted_prompts": accepted["n_accepted_prompts"],
        "n_expected_rollouts": accepted["n_expected_rollouts"],
    }
    expected_generation = {
        "issue": ISSUE,
        "model": MODEL,
        "model_revision": MODEL_REV,
        "bank_sha256": input_marker.get("prompt_bank_sha256"),
        "layers": list(LAYERS),
        "draws": N_DRAWS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_base": GENERATION_SEED_BASE,
        "seed_namespace": GENERATION_SEED_NAMESPACE,
        "seed_formula": GENERATION_SEED_FORMULA,
        "seed_policy": GENERATION_SEED_POLICY,
        "smoke": False,
        "attempt": attempt,
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "n_accepted_source_items": n_sources,
        "n_accepted_prompts": accepted["n_accepted_prompts"],
        "n_expected_rollouts": accepted["n_expected_rollouts"],
        "n_selected_prompts": accepted["n_accepted_prompts"],
        "selected_source_ids_sha256": accepted["accepted_source_ids_sha256"],
    }
    prompt_token_max = regime.get("prompt_token_max")
    package_versions = regime.get("package_versions")
    tokenizer_hashes = regime.get("tokenizer_artifact_sha256")
    if (
        any(regime.get(key) != value for key, value in expected_generation.items())
        or isinstance(prompt_token_max, bool)
        or not isinstance(prompt_token_max, int)
        or prompt_token_max < 1
        or regime.get("max_model_len") != max(4096, prompt_token_max + MAX_NEW_TOKENS + 32)
        or re.fullmatch(r"[0-9a-f]{40}", regime.get("git_sha", "")) is None
        or not _is_sha256(regime.get("chat_template_sha256"))
        or not isinstance(tokenizer_hashes, dict)
        or set(tokenizer_hashes) != TOKENIZER_ARTIFACTS
        or not all(_is_sha256(value) for value in tokenizer_hashes.values())
        or not isinstance(package_versions, dict)
        or set(package_versions) != set(REGISTERED_PACKAGE_VERSIONS)
        or any(
            not isinstance(package_versions[name], str)
            or package_versions[name].split("+")[0] != expected
            for name, expected in REGISTERED_PACKAGE_VERSIONS.items()
        )
        or generation.get("package_versions") != package_versions
        or generation.get("accepted_bank") != expected_bank
        or generation.get("regime_fp") != _sha_obj(regime)
    ):
        raise RuntimeError("GPU generation regime differs from the registered Qwen recipe")

    generation_fp = _generation_fingerprint(generation)
    expected_capture = {
        "model_revision": MODEL_REV,
        "layers": list(LAYERS),
        "bank_sha256": input_marker.get("prompt_bank_sha256"),
        "rollouts_sha256": generation.get("rollouts_sha256"),
        "context_position": CONTEXT_POSITION,
        "answer_pooling": ANSWER_POOLING,
        "serialized_dtype": SERIALIZED_DTYPE,
        "git_sha": regime["git_sha"],
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "n_accepted_source_items": n_sources,
        "n_accepted_prompts": accepted["n_accepted_prompts"],
        "n_selected_prompts": accepted["n_accepted_prompts"],
        "selected_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "generation_fingerprint": generation_fp,
    }
    if (
        any(capture_regime.get(key) != value for key, value in expected_capture.items())
        or capture.get("issue") != ISSUE
        or capture.get("model_revision") != MODEL_REV
        or capture.get("layers") != list(LAYERS)
        or capture.get("rollouts_sha256") != generation.get("rollouts_sha256")
        or capture.get("package_versions") != package_versions
        or capture.get("accepted_bank") != expected_bank
        or capture.get("capture_regime_fp") != _sha_obj(capture_regime)
        or capture.get("generation_fingerprint") != generation_fp
    ):
        raise RuntimeError("GPU capture regime differs from the registered Qwen recipe")


def stage_reuse(run_dir: Path) -> dict[str, Any]:
    """Stage pinned frozen maps and historical axis inputs into consumer layout."""

    mapping = {
        (MAP_REV, f"{MAP_PREFIX}/L{layer}/ridge.pt"): (
            run_dir / "reuse" / "maps" / f"L{layer}" / "ridge.pt"
        )
        for layer in LAYERS
    }
    mapping.update(
        {
            (HIST_REV, remote): run_dir / "reuse" / "axis" / local
            for remote, local in HIST_FILES.items()
        }
    )
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("reuse source-to-local mapping collides")
    source_records = {}
    for (revision, remote), dest in mapping.items():
        fetched = Path(
            hub.retry_transient(
                lambda remote=remote, revision=revision: hf_hub_download(
                    HF_REPO,
                    remote,
                    repo_type="dataset",
                    revision=revision,
                ),
                what=f"issue952 pinned reuse stage {remote}",
            )
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and _sha256(dest) != _sha256(fetched):
            raise RuntimeError(f"staged reuse file drift: {dest}")
        if not dest.exists():
            shutil.copyfile(fetched, dest)
        source_records[remote] = {
            "revision": revision,
            "destination": str(dest.relative_to(run_dir)),
            "sha256": _sha256(dest),
        }
    for layer in LAYERS:
        bundle = torch.load(
            run_dir / "reuse" / "maps" / f"L{layer}" / "ridge.pt",
            map_location="cpu",
            weights_only=False,
        )
        required = {"W", "xmu", "xsd", "ymu", "kind", "fitter", "layer"}
        if (
            not required <= set(bundle)
            or bundle["W"].shape != (HIDDEN, HIDDEN)
            or int(bundle["layer"]) != layer
        ):
            raise RuntimeError(f"realized frozen-map schema mismatch at layer {layer}")
    axis_store = torch.load(
        run_dir / "reuse" / "axis" / "analysis_tensors" / "va" / "va_langow_query_svmp.pt",
        map_location="cpu",
        weights_only=False,
    )
    if axis_store["layers"] != list(LAYERS) or axis_store["va_tail_incl"].shape != (
        2480,
        3,
        HIDDEN,
    ):
        raise RuntimeError("realized historical vector schema mismatch")
    report = {
        "map_revision": MAP_REV,
        "historical_revision": HIST_REV,
        "files": source_records,
    }
    _write_json(run_dir / "reuse" / "stage_report.json", report)
    print(f"[stage] files={len(mapping)} maps=3 historical=3")
    return report


def _stage_new_run_file(
    run_dir: Path, revision: str, relative: str, *, remote_relative: str | None = None
) -> Path:
    """Stage one exact-revision analysis input into its canonical local path."""
    remote_relative = remote_relative or relative
    fetched = Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                HF_REPO,
                f"{HF_PREFIX}/{remote_relative}",
                repo_type="dataset",
                revision=revision,
            ),
            what=f"issue952 analysis input stage {relative}",
        )
    )
    destination = run_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _sha256(destination) != _sha256(fetched):
        raise RuntimeError(f"local analysis input drift: {relative}")
    if not destination.exists():
        shutil.copyfile(fetched, destination)
    return destination


def stage_new_run(run_dir: Path, *, attempt: int = 1) -> dict[str, Any]:
    """Materialize one immutable post-judge snapshot into the CPU run layout."""
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    api = HfApi()
    marker_source_revision = hub.retry_transient(
        lambda: api.repo_info(HF_REPO, repo_type="dataset", revision="main").sha,
        what="issue952 analysis snapshot resolution",
    )
    marker_path = _stage_new_run_file(
        run_dir,
        marker_source_revision,
        f"judge/attempt{attempt}/upload.json",
        remote_relative=f"attempt{attempt}/judge/upload.json",
    )
    marker = json.loads(marker_path.read_text())
    revision = marker.get("data_revision")
    if marker.get("attempt") != attempt or not isinstance(revision, str) or not revision:
        raise RuntimeError("analysis judge marker lacks matching attempt/data revision")
    judge_census = _validated_judge_census(marker)
    canonical = (
        "inputs/upload_verified.json",
        "inputs/prompt_bank.jsonl",
        "inputs/bank_audit_report.json",
    )
    attempt_fixed = (
        "raw_completions/rollouts.jsonl",
        "manifests/input_stage.json",
        "manifests/generation.json",
        "manifests/raw_upload.json",
        "manifests/capture.json",
        "manifests/capture_upload.json",
        "issue952_china_definitive_done.json",
    )
    paths = {relative: _stage_new_run_file(run_dir, revision, relative) for relative in canonical}
    paths.update(
        {
            relative: _stage_new_run_file(
                run_dir,
                revision,
                relative,
                remote_relative=f"attempt{attempt}/{relative}",
            )
            for relative in attempt_fixed
        }
    )
    for remote_relative in sorted(judge_census):
        local_relative = f"judge/attempt{attempt}/{Path(remote_relative).relative_to('judge')}"
        paths[local_relative] = _stage_new_run_file(
            run_dir,
            revision,
            local_relative,
            remote_relative=f"attempt{attempt}/{remote_relative}",
        )
    paths[f"judge/attempt{attempt}/upload.json"] = marker_path
    capture = json.loads(paths["manifests/capture.json"].read_text())
    tensor_relatives = ["analysis_tensors/vc.pt"] + [
        f"analysis_tensors/{name}" for name in sorted(capture.get("va_files", {}))
    ]
    if len(tensor_relatives) < 2:
        raise RuntimeError("analysis staging capture manifest has no answer shards")
    for relative in tensor_relatives:
        paths[relative] = _stage_new_run_file(
            run_dir,
            revision,
            relative,
            remote_relative=f"attempt{attempt}/{relative}",
        )
    _load_new_data(run_dir, attempt=attempt)
    report = {
        "snapshot_revision": revision,
        "marker_source_revision": marker_source_revision,
        "attempt": attempt,
        "files": {relative: _sha256(path) for relative, path in sorted(paths.items())},
    }
    _write_json(run_dir / "reuse" / "new_run_stage_report.json", report)
    print(f"[analysis-stage] revision={revision} files={len(paths)}")
    return report


def operator_svd(weight: torch.Tensor, xsd: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return the full input basis and spectrum of the raw-coordinate map."""

    w = weight.detach().cpu().double().numpy()
    scale = xsd.detach().cpu().double().numpy()
    if w.shape != (len(scale), len(scale)) or np.any(scale <= 0):
        raise ValueError("invalid map shape/xsd")
    a = w / scale[:, None]
    u, s, _vh = np.linalg.svd(a, full_matrices=False)
    return u, s


def mass_rank(singular: np.ndarray, mass: float) -> tuple[int, float]:
    if not 0 < mass <= 1 or singular.ndim != 1 or not np.all(singular >= 0):
        raise ValueError("invalid singular spectrum/mass")
    cumulative = np.cumsum(singular * singular) / np.sum(singular * singular)
    first = int(np.searchsorted(cumulative, mass, side="left"))
    tau = float(singular[first])
    keep = int(np.sum(singular >= tau))
    return keep, tau


def effective_svd(
    weight: torch.Tensor, xsd: torch.Tensor, mass: float
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Return raw-coordinate retained input directions, spectrum, rank, and tau."""

    u, singular = operator_svd(weight, xsd)
    keep, tau = mass_rank(singular, mass)
    return u[:, :keep], singular, keep, tau


def kernel_share(delta: np.ndarray, u_ret: np.ndarray) -> np.ndarray:
    """Squared-norm share outside the retained input directions."""

    total = np.sum(delta * delta, axis=-1)
    retained = np.sum((delta @ u_ret) ** 2, axis=-1)
    return np.where(total > 0, np.clip(1.0 - retained / total, 0.0, 1.0), np.nan)


def distance_matched_within_condition_null(
    target_delta: np.ndarray,
    within_condition: np.ndarray,
    topics: np.ndarray,
    u_ret: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match every target norm to an in-topic within-condition item pair."""

    target_norm = np.linalg.norm(target_delta, axis=1)
    shares = np.empty(len(topics), dtype=np.float64)
    pairs = np.empty((len(topics), 2), dtype=np.int64)
    relative_error = np.empty(len(topics), dtype=np.float64)
    for i, topic in enumerate(topics):
        indices = np.flatnonzero(topics == topic)
        candidates = [(a, b) for pos, a in enumerate(indices) for b in indices[pos + 1 :]]
        if not candidates:
            raise RuntimeError(f"distance-matched null has fewer than two items in topic {topic}")
        deltas = np.stack([within_condition[a] - within_condition[b] for a, b in candidates])
        norms = np.linalg.norm(deltas, axis=1)
        chosen = int(np.argmin(np.abs(norms - target_norm[i])))
        shares[i] = kernel_share(deltas[chosen : chosen + 1], u_ret)[0]
        pairs[i] = candidates[chosen]
        relative_error[i] = abs(norms[chosen] - target_norm[i]) / max(target_norm[i], 1e-300)
    return shares, pairs, relative_error


def structured_projection_coords(x: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """Coordinates under the fixed signed-DCT exact-rank orthoprojector null."""

    if not 0 < rank <= x.shape[1]:
        raise ValueError("projection rank out of range")
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=x.shape[1])
    coordinates = dct(x * signs, type=2, norm="ortho", axis=1)
    chosen = rng.choice(x.shape[1], size=rank, replace=False)
    return coordinates[:, chosen]


def structured_projection_coords_batch(x: np.ndarray, rank: int, seeds: np.ndarray) -> np.ndarray:
    """Vectorized blocks of fixed signed-DCT projector coordinates."""

    if not 0 < rank <= x.shape[1]:
        raise ValueError("projection rank out of range")
    signs = np.empty((len(seeds), x.shape[1]), dtype=np.float64)
    chosen = np.empty((len(seeds), rank), dtype=np.int64)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        signs[i] = rng.choice(np.array([-1.0, 1.0]), size=x.shape[1])
        chosen[i] = rng.choice(x.shape[1], size=rank, replace=False)
    transformed = dct(x[None] * signs[:, None], type=2, norm="ortho", axis=2)
    return np.take_along_axis(transformed, chosen[:, None], axis=2)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    return (a @ b.T) / np.maximum(an * bn.T, np.finfo(np.float64).tiny)


def retrieval_predictions(query: np.ndarray, gallery: np.ndarray, topics: np.ndarray) -> np.ndarray:
    """Global gallery index predicted for each query, restricted within topic."""

    pred = np.empty(len(query), dtype=np.int64)
    for topic in np.unique(topics):
        idx = np.flatnonzero(topics == topic)
        local = np.argmax(_cosine_matrix(query[idx], gallery[idx]), axis=1)
        pred[idx] = idx[local]
    return pred


def retrieval_predictions_and_ranks(
    query: np.ndarray, gallery: np.ndarray, topics: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted global gallery indices and one-based true-pair cosine ranks."""

    if topics is None:
        topics = np.zeros(len(query), dtype=np.int64)
    pred = np.empty(len(query), dtype=np.int64)
    ranks = np.empty(len(query), dtype=np.int64)
    for topic in np.unique(topics):
        idx = np.flatnonzero(topics == topic)
        sim = _cosine_matrix(query[idx], gallery[idx])
        pred[idx] = idx[np.argmax(sim, axis=1)]
        true = np.diag(sim)
        ranks[idx] = 1 + np.sum(sim > true[:, None], axis=1)
    return pred, ranks


def batch_retrieval_predictions_and_ranks(
    query: np.ndarray, gallery: np.ndarray, topics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Batched counterpart for arrays shaped (draw, item, coordinate)."""

    n_draw, n_item, _ = query.shape
    pred = np.empty((n_draw, n_item), dtype=np.int64)
    ranks = np.empty((n_draw, n_item), dtype=np.int64)
    for topic in np.unique(topics):
        idx = np.flatnonzero(topics == topic)
        q = query[:, idx]
        g = gallery[:, idx]
        qn = np.linalg.norm(q, axis=2, keepdims=True)
        gn = np.linalg.norm(g, axis=2, keepdims=True)
        sim = np.einsum("bik,bjk->bij", q, g, optimize=True) / np.maximum(
            qn * np.swapaxes(gn, 1, 2), np.finfo(np.float64).tiny
        )
        pred[:, idx] = idx[np.argmax(sim, axis=2)]
        true = np.diagonal(sim, axis1=1, axis2=2)
        ranks[:, idx] = 1 + np.sum(sim > true[:, :, None], axis=2)
    return pred, ranks


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for position, idx in enumerate(order):
        running = max(running, min(1.0, (m - position) * p[idx]))
        out[idx] = running
    return out.tolist()


def bootstrap_p_greater(samples: np.ndarray, observed: float, null: float = 0.0) -> float:
    """One-sided bootstrap lower-tail p-value for the alternative theta > null."""

    if not np.isfinite(observed):
        return 1.0
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return 1.0
    return float((1 + np.sum(finite <= null)) / (len(finite) + 1))


def topic_bootstrap_weights(topics: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Outer-topic/inner-item bootstrap encoded as integer item weights."""

    unique = np.unique(topics)
    by_topic = [np.flatnonzero(topics == topic) for topic in unique]
    sizes = np.asarray([len(indices) for indices in by_topic], dtype=np.int64)
    max_size = int(sizes.max())
    lookup = np.zeros((len(unique), max_size), dtype=np.int64)
    for i, indices in enumerate(by_topic):
        lookup[i, : len(indices)] = indices
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    selected_sizes = sizes[selected]
    local = (rng.random((n_boot, len(unique), max_size)) * selected_sizes[:, :, None]).astype(
        np.int64
    )
    global_indices = lookup[selected[:, :, None], local]
    valid = np.arange(max_size)[None, None] < selected_sizes[:, :, None]
    boot_rows = np.broadcast_to(np.arange(n_boot)[:, None, None], valid.shape)
    out = np.zeros((n_boot, len(topics)), dtype=np.int16)
    np.add.at(out, (boot_rows[valid], global_indices[valid]), 1)
    return out


def bootstrap_weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (weights @ values) / weights.sum(axis=1)


def bootstrap_weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_weights = weights[:, order]
    cumulative = np.cumsum(sorted_weights, axis=1)
    total = weights.sum(axis=1).astype(np.int64)
    lower_target = (total - 1) // 2
    upper_target = total // 2
    lower = np.argmax(cumulative > lower_target[:, None], axis=1)
    upper = np.argmax(cumulative > upper_target[:, None], axis=1)
    ordered = values[order]
    return (ordered[lower] + ordered[upper]) / 2


def topic_bootstrap_spearman(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Vectorized exact Spearman rho for whole-topic bootstrap multiplicities."""

    def weighted_midranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0) + 1]
        ends = np.r_[starts[1:], len(values)]
        group_weight = np.stack(
            [weights[:, order[start:end]].sum(axis=1) for start, end in zip(starts, ends)]
        ).T
        prior = np.cumsum(group_weight, axis=1) - group_weight
        group_rank = prior + (group_weight + 1) / 2
        ranks = np.empty((len(weights), len(values)), dtype=np.float64)
        for group, (start, end) in enumerate(zip(starts, ends)):
            ranks[:, order[start:end]] = group_rank[:, group, None]
        return ranks

    rank_x = weighted_midranks(x)
    rank_y = weighted_midranks(y)
    total = weights.sum(axis=1)
    mean_x = np.sum(weights * rank_x, axis=1) / total
    mean_y = np.sum(weights * rank_y, axis=1) / total
    centered_x = rank_x - mean_x[:, None]
    centered_y = rank_y - mean_y[:, None]
    covariance = np.sum(weights * centered_x * centered_y, axis=1)
    denominator = np.sqrt(
        np.sum(weights * centered_x**2, axis=1) * np.sum(weights * centered_y**2, axis=1)
    )
    return np.divide(
        covariance,
        denominator,
        out=np.full(len(weights), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [math.nan, math.nan]
    return [float(x) for x in np.quantile(finite, [0.025, 0.975])]


def _mean_defined(values: list[float | None]) -> float:
    """Average defined report values while preserving an all-missing cell as a gap."""

    finite = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    return float(finite.mean()) if len(finite) else math.nan


def _holm_rejections(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Vectorized Holm step-down decisions for rows of hypothesis p-values."""

    order = np.argsort(pvalues, axis=1)
    sorted_p = np.take_along_axis(pvalues, order, axis=1)
    thresholds = alpha / np.arange(pvalues.shape[1], 0, -1)
    sorted_reject = np.cumprod(sorted_p <= thresholds[None], axis=1).astype(bool)
    rejected = np.zeros_like(sorted_reject)
    np.put_along_axis(rejected, order, sorted_reject, axis=1)
    return rejected


def _headline_success(cells: np.ndarray) -> np.ndarray:
    """Both languages at L19 and both at either L14 or L26."""

    shaped = cells.reshape(len(cells), 3, 2)
    return np.all(shaped[:, 1], axis=1) & (
        np.all(shaped[:, 0], axis=1) | np.all(shaped[:, 2], axis=1)
    )


def design_power(out_dir: Path) -> dict[str, Any]:
    """Advisory simulation of the registered max/IUT/Holm/headline lattice."""

    topic_sizes = np.array([2, 8, 3, 10, 10, 4, 10, 10, 7, 10, 8, 8])
    topics = np.repeat(np.arange(len(topic_sizes)), topic_sizes)
    n = len(topics)
    n_sim = 5_000
    n_inner = 499
    batch_size = 25
    rng = np.random.default_rng(SEED)
    truth = np.arange(n)
    perm_targets = _within_topic_targets(topics, n_inner, SEED + 90)
    boot_weights = topic_bootstrap_weights(topics, n_inner, SEED + 91)
    random_null_accuracy = float(np.mean([1 / topic_sizes[t] for t in topics]))
    alternatives = [np.flatnonzero((topics == topics[i]) & (truth != i)) for i in truth]

    h1_pass_by_discordance: dict[float, np.ndarray] = {}
    for discordance in (0.20, 0.35, 0.50):
        passed = np.empty((n_sim, 6), dtype=bool)
        p10 = (discordance + 0.10) / 2
        p01 = (discordance - 0.10) / 2
        p11 = 0.30 - p01
        probabilities = np.array([p11, p10, p01, 1 - p11 - p10 - p01])
        if np.any(probabilities < 0):
            raise RuntimeError("invalid paired-H1 design-power probabilities")
        for start in range(0, n_sim, batch_size):
            size = min(batch_size, n_sim - start)
            category = rng.choice(4, size=(size, 6, n), p=probabilities)
            hit_ret = (category == 0) | (category == 1)
            hit_ker = (category == 0) | (category == 2)
            pred_ret = np.broadcast_to(truth, (size, 6, n)).copy()
            pred_ker = pred_ret.copy()
            for i, choices in enumerate(alternatives):
                pred_ret[:, :, i] = np.where(
                    hit_ret[:, :, i], i, rng.choice(choices, size=(size, 6))
                )
                pred_ker[:, :, i] = np.where(
                    hit_ker[:, :, i], i, rng.choice(choices, size=(size, 6))
                )
            observed = hit_ret.mean(axis=2) - np.maximum(hit_ker.mean(axis=2), random_null_accuracy)
            pvalues = np.empty((size, 6))
            for cell in range(6):
                p_ret = np.mean(pred_ret[:, cell, None, :] == perm_targets[None, :, :], axis=2)
                p_ker = np.mean(pred_ker[:, cell, None, :] == perm_targets[None, :, :], axis=2)
                permuted = p_ret - np.maximum(p_ker, random_null_accuracy)
                pvalues[:, cell] = (1 + np.sum(permuted >= observed[:, cell, None], axis=1)) / (
                    n_inner + 1
                )
            passed[start : start + size] = (observed >= 0.10) & _holm_rejections(pvalues)
        h1_pass_by_discordance[discordance] = passed

    h2_pass_by_setting: dict[tuple[float, float], np.ndarray] = {}
    for h2_sd in (0.10, 0.15, 0.25):
        for corr in (0.0, 0.4, 0.8):
            passed = np.empty((n_sim, 6), dtype=bool)
            covariance = h2_sd**2 * np.array([[1.0, corr], [corr, 1.0]])
            for start in range(0, n_sim, batch_size):
                size = min(batch_size, n_sim - start)
                values = rng.multivariate_normal([0.05, 0.05], covariance, size=(size, 6, n))
                observed = np.median(values, axis=2)
                component_p = np.empty((size, 6, 2))
                for sim in range(size):
                    for cell in range(6):
                        for component in range(2):
                            boot = bootstrap_weighted_median(
                                values[sim, cell, :, component], boot_weights
                            )
                            component_p[sim, cell, component] = (1 + np.sum(boot <= 0)) / (
                                n_inner + 1
                            )
                iut_p = component_p.max(axis=2)
                passed[start : start + size] = np.all(observed >= 0.05, axis=2) & _holm_rejections(
                    iut_p
                )
            h2_pass_by_setting[(h2_sd, corr)] = passed

    records = []
    for discordance, h1_pass in h1_pass_by_discordance.items():
        for (h2_sd, corr), h2_pass in h2_pass_by_setting.items():
            records.append(
                {
                    "discordance": discordance,
                    "h2_sd": h2_sd,
                    "h2_corr": corr,
                    "h1_per_cell_power": float(h1_pass.mean()),
                    "h1_headline_power": float(_headline_success(h1_pass).mean()),
                    "h2_per_cell_power": float(h2_pass.mean()),
                    "h2_headline_power": float(_headline_success(h2_pass).mean()),
                    "joint_headline_power": float(
                        np.mean(_headline_success(h1_pass) & _headline_success(h2_pass))
                    ),
                }
            )
    report = {
        "advisory": True,
        "n_items": n,
        "n_simulations_per_cell": n_sim,
        "inner_topic_resamples": n_inner,
        "topic_sizes": topic_sizes,
        "records": records,
        "mechanics": {
            "H1": "topic-restricted predictions; retained-minus-max(kernel, exact uniform-random expectation); within-topic label permutations; effect gate",
            "H2": "topic-stratified bootstrap medians for both nuisance contrasts; intersection-union max p-value; effect gate",
            "multiplicity": "Holm over six layer-by-language/direction cells separately for H1 and H2",
            "headline": "both languages/directions at L19 and at least one of L14/L26",
        },
        "note": "Pre-data alternatives are assumptions; 499 inner Monte Carlo resamples approximate the registered 10,000-resample tests, while the decision lattice is exact. No production gate or result-dependent N change.",
    }
    _write_json(out_dir / "power_design.json", report)
    print(f"[power] records={len(records)} n={n} advisory=true")
    return report


def build_refusal_axis(hist_root: Path, out_dir: Path) -> dict[str, Any]:
    manifest_path = hist_root / "manifests" / "svmp_bank.json"
    judge_path = hist_root / "raw_completions" / "judge" / "judge_scores.json"
    va_path = hist_root / "analysis_tensors" / "va" / "va_langow_query_svmp.pt"
    manifest = json.loads(manifest_path.read_text())
    judge = json.loads(judge_path.read_text())
    store = torch.load(va_path, map_location="cpu", weights_only=False)
    layers = [int(x) for x in store["layers"]]
    if layers != list(LAYERS) or store["va_tail_incl"].shape != (2480, 3, HIDDEN):
        raise RuntimeError("historical answer-vector schema changed")
    sums = defaultdict(lambda: np.zeros((len(LAYERS), HIDDEN), dtype=np.float64))
    counts = defaultdict(int)
    for i, rec in enumerate(store["index"]):
        sums[rec["context_id"]] += store["va_tail_incl"][i].float().numpy()
        counts[rec["context_id"]] += 1
    if set(counts.values()) != {10}:
        raise RuntimeError("historical axis requires ten vectors per context")
    means = {key: value / counts[key] for key, value in sums.items()}

    deltas_by_threshold: dict[float, tuple[list[str], np.ndarray]] = {}
    for threshold in (0.4, 0.5, 0.6):
        ids, deltas = [], []
        for pair in manifest["pairs"]:
            ra = float(judge["per_context"][pair["a"]]["refusal_rate"])
            rb = float(judge["per_context"][pair["b"]]["refusal_rate"])
            if abs(ra - rb) >= threshold:
                high, low = (pair["a"], pair["b"]) if ra > rb else (pair["b"], pair["a"])
                ids.append(pair["pair_id"])
                deltas.append(means[high] - means[low])
        deltas_by_threshold[threshold] = (ids, np.stack(deltas))
    ids, deltas = deltas_by_threshold[0.5]
    if len(ids) != 61:
        raise RuntimeError(f"expected 61 threshold-0.5 pairs, got {len(ids)}")

    def axes_for(ds: np.ndarray) -> np.ndarray:
        axes = ds.mean(axis=0)
        norms = np.linalg.norm(axes, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("historical refusal axis has zero norm")
        return axes / norms

    axes = axes_for(deltas)
    split_a = axes_for(deltas[::2])
    split_b = axes_for(deltas[1::2])
    split_cos = np.sum(split_a * split_b, axis=1)
    rng = np.random.default_rng(SEED + 2617)
    bootstrap_cos = np.empty((N_BOOT, len(LAYERS)), dtype=np.float64)
    for start in range(0, N_BOOT, 250):
        size = min(250, N_BOOT - start)
        idx = rng.integers(0, len(deltas), size=(size, len(deltas)))
        boot = deltas[idx].mean(axis=1)
        boot /= np.maximum(np.linalg.norm(boot, axis=2, keepdims=True), 1e-300)
        bootstrap_cos[start : start + size] = np.sum(boot * axes[None], axis=2)
    threshold_cos = {}
    threshold_axes = {}
    for threshold in (0.4, 0.6):
        other = axes_for(deltas_by_threshold[threshold][1])
        threshold_cos[str(threshold)] = np.sum(other * axes, axis=1)
        threshold_axes[str(threshold)] = torch.from_numpy(other).double()
    fifth = np.quantile(bootstrap_cos, 0.05, axis=0)
    gate_components = {
        "pair_count": len(ids) >= 50,
        "axis_norm": bool(np.all(np.linalg.norm(axes, axis=1) > 0)),
        "split_half": bool(np.all(split_cos >= 0.70)),
        "bootstrap_stability": bool(np.all(fifth >= 0.50)),
        "threshold_cosine": bool(all(np.all(value >= 0.90) for value in threshold_cos.values())),
    }
    core_eligible = all(
        gate_components[key]
        for key in ("pair_count", "axis_norm", "split_half", "bootstrap_stability")
    )
    eligible = bool(core_eligible and gate_components["threshold_cosine"])
    out_dir.mkdir(parents=True, exist_ok=True)
    axis_path = out_dir / "refusal_axis.pt"
    torch.save(
        {
            "layers": list(LAYERS),
            "axis": torch.from_numpy(axes).double(),
            "threshold_axes": threshold_axes,
            "pair_ids": ids,
            "threshold": 0.5,
            "source_sha256": {
                "manifest": _sha256(manifest_path),
                "judge": _sha256(judge_path),
                "va": _sha256(va_path),
            },
        },
        axis_path,
    )
    report = {
        "eligible_pre_outcomes": eligible,
        "core_eligible_pre_outcomes": core_eligible,
        "threshold_stable_pre_outcomes": gate_components["threshold_cosine"],
        "gate_components": gate_components,
        "n_pairs": len(ids),
        "axis_norms": np.linalg.norm(axes, axis=1),
        "split_half_cosine": split_cos,
        "bootstrap_fifth_percentile_cosine": fifth,
        "threshold_axis_cosine": threshold_cos,
        "pair_ids": ids,
        "axis_sha256": _sha256(axis_path),
    }
    _write_json(out_dir / "refusal_axis_report.json", report)
    print(
        f"[axis] eligible={eligible} n={len(ids)} "
        f"split_min={split_cos.min():.4f} boot_p05_min={fifth.min():.4f}"
    )
    return report


def validate_analysis_attempt_identity(
    *,
    local_bank_sha256: str,
    input_marker: dict[str, Any],
    generation: dict[str, Any],
    capture: dict[str, Any],
    judge_summary: dict[str, Any],
) -> None:
    """Reject cross-phase joins assembled from different production attempts."""

    expected_bank = input_marker.get("prompt_bank_sha256")
    bank_hashes = (
        local_bank_sha256,
        generation.get("regime", {}).get("bank_sha256"),
        capture.get("capture_regime", {}).get("bank_sha256"),
    )
    expected_rollouts = generation.get("rollouts_sha256")
    rollout_hashes = (
        capture.get("rollouts_sha256"),
        judge_summary.get("rollouts_sha256"),
    )
    if any(value != expected_bank for value in bank_hashes) or any(
        value != expected_rollouts for value in rollout_hashes
    ):
        raise RuntimeError("analysis cross-phase production-attempt identity mismatch")


def _validate_gpu_attempt(
    *,
    run_dir: Path,
    attempt: int,
    accepted: dict[str, Any],
    input_marker: dict[str, Any],
    input_stage: dict[str, Any],
    generation: dict[str, Any],
    capture: dict[str, Any],
    raw_upload: dict[str, Any],
    capture_upload: dict[str, Any],
    done: dict[str, Any],
) -> None:
    """Validate the terminal GPU receipt and its input/count lineage."""
    bank_path = run_dir / "inputs" / "prompt_bank.jsonl"
    audit_path = run_dir / "inputs" / "bank_audit_report.json"
    marker_path = run_dir / "inputs" / "upload_verified.json"
    accepted_counts = (
        len(accepted["accepted_source_ids"]),
        accepted["n_accepted_prompts"],
        accepted["n_expected_rollouts"],
    )
    _validate_registered_gpu_regime(
        attempt=attempt,
        accepted=accepted,
        input_marker=input_marker,
        generation=generation,
        capture=capture,
    )
    expected_tensor_bytes = {
        f"{HF_PREFIX}/attempt{attempt}/analysis_tensors/vc.pt": capture.get("vc_sha256"),
        **{
            f"{HF_PREFIX}/attempt{attempt}/analysis_tensors/{name}": sha
            for name, sha in capture.get("va_files", {}).items()
        },
    }
    if (
        done.get("schema_version") != 1
        or done.get("kind") != "issue952_china_definitive_gpu"
        or done.get("issue") != ISSUE
        or done.get("status") != "done"
        or done.get("version") != 1
        or done.get("generation") != generation
        or done.get("capture") != capture
        or done.get("raw_upload") != raw_upload
        or done.get("capture_upload") != capture_upload
        or done.get("hf_prefix") != HF_PREFIX
        or generation.get("regime", {}).get("attempt") != attempt
        or generation.get("regime", {}).get("smoke") is not False
        or generation.get("n_rows") != capture.get("n_answer_rows")
        or generation.get("n_prompts") != capture.get("n_contexts")
        or generation.get("n_rows") != accepted["n_expected_rollouts"]
        or generation.get("n_prompts") != accepted["n_accepted_prompts"]
        or capture.get("generation_fingerprint") != _generation_fingerprint(generation)
        or capture.get("capture_regime", {}).get("generation_fingerprint")
        != _generation_fingerprint(generation)
        or capture.get("model_revision") != capture.get("capture_regime", {}).get("model_revision")
        or capture.get("model_revision") != generation.get("regime", {}).get("model_revision")
        or capture_upload.get("byte_verified_sha256") != expected_tensor_bytes
        or not isinstance(capture_upload.get("byte_verified_files"), list)
        or len(capture_upload.get("byte_verified_files", [])) != len(expected_tensor_bytes)
        or set(capture_upload.get("byte_verified_files", [])) != set(expected_tensor_bytes)
        or input_stage.get("marker_sha256") != _sha256(marker_path)
        or input_stage.get("data_revision") != input_marker.get("data_revision")
        or input_stage.get("prompt_bank_sha256") != _sha256(bank_path)
        or input_stage.get("bank_audit_report_sha256") != _sha256(audit_path)
        or input_stage.get("accepted_source_ids_sha256") != accepted["accepted_source_ids_sha256"]
        or input_marker.get("prompt_bank_sha256") != input_stage.get("prompt_bank_sha256")
        or input_marker.get("bank_audit_report_sha256")
        != input_stage.get("bank_audit_report_sha256")
        or (
            input_stage.get("n_accepted_source_items"),
            input_stage.get("n_accepted_prompts"),
            input_stage.get("n_expected_rollouts"),
        )
        != accepted_counts
        or generation.get("regime", {}).get("accepted_source_ids_sha256")
        != accepted["accepted_source_ids_sha256"]
        or capture.get("capture_regime", {}).get("accepted_source_ids_sha256")
        != accepted["accepted_source_ids_sha256"]
    ):
        raise RuntimeError("GPU terminal receipt/input-stage/count lineage mismatch")


def _validated_answer_shard(
    *,
    store: dict[str, Any],
    rollout_by_item: dict[str, dict[str, Any]],
    capture: dict[str, Any],
    generation: dict[str, Any],
    name: str,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """Validate one answer-vector shard and return its aligned index/tensor."""
    index = store.get("index")
    vectors = store.get("va_tail_incl")
    empty_rows = store.get("empty_rows")
    if (
        store.get("layers") != list(LAYERS)
        or not isinstance(index, list)
        or not isinstance(vectors, torch.Tensor)
        or vectors.shape != (len(index), len(LAYERS), HIDDEN)
        or vectors.dtype != torch.float32
        or store.get("dtype") != "fp32"
        or store.get("pooling") != "completion_plus_im_end_newline_mean"
        or store.get("model_revision") != capture.get("model_revision")
        or store.get("rollouts_sha256") != generation.get("rollouts_sha256")
        or store.get("capture_regime") != capture.get("capture_regime")
        or store.get("capture_regime_fp") != capture.get("capture_regime_fp")
        or not isinstance(empty_rows, list)
    ):
        raise RuntimeError(f"answer shard index/tensor/capture metadata mismatch: {name}")
    if empty_rows:
        raise RuntimeError(f"empty answer captures in {name}")
    for rec in index:
        rollout = rollout_by_item.get(rec.get("item_id"))
        if (
            rollout is None
            or rec.get("prompt_id") != rollout.get("prompt_id")
            or rec.get("draw") != rollout.get("draw")
        ):
            raise RuntimeError("answer shard (item_id,prompt_id,draw) differs from rollout")
    return index, vectors


def _validate_judge_attempt(
    *,
    run_dir: Path,
    judge_dir: Path,
    attempt: int,
    marker: dict[str, Any],
    accepted: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Validate the complete judge census and return reportable measurement metadata."""
    census = _validated_judge_census(marker)
    for relative, expected_sha in census.items():
        local = judge_dir / Path(relative).relative_to("judge")
        if not local.is_file() or _sha256(local) != expected_sha:
            raise RuntimeError(f"production judge artifact census mismatch: {relative}")
    direct_hashes = {
        "wave_scores_sha256": "judge/wave_scores.jsonl",
        "wave_summary_sha256": "judge/wave_summary.json",
        "wave_request_manifest_sha256": "judge/wave_request_manifest.json",
        "wave_packet_manifest_sha256": "judge/wave_packet_manifest.json",
        "wave_lookup_sha256": "judge/wave_lookup.json",
        "wave_runtime_identity_sha256": "judge/wave_runtime_identity.json",
        "wave_overlap_joined_sha256": "judge/wave_overlap_joined.jsonl",
        "production_stage_sha256": "judge/production_stage.json",
    }
    if any(marker.get(key) != census[relative] for key, relative in direct_hashes.items()):
        raise RuntimeError("production judge upload marker disagrees with its artifact census")

    request = json.loads((judge_dir / "wave_request_manifest.json").read_text())
    packet_manifest = json.loads((judge_dir / "wave_packet_manifest.json").read_text())
    lookup = json.loads((judge_dir / "wave_lookup.json").read_text())
    summary = json.loads((judge_dir / "wave_summary.json").read_text())
    stage = json.loads((judge_dir / "production_stage.json").read_text())
    scores = _jsonl(judge_dir / "wave_scores.jsonl")
    overlap = _jsonl(judge_dir / "wave_overlap_joined.jsonl")
    expected_counts = (
        len(accepted["accepted_source_ids"]),
        accepted["n_accepted_prompts"],
        accepted["n_expected_rollouts"],
    )
    score_ids = [row.get("item_id") for row in scores]
    lookup_ids = [row.get("item_id") for row in lookup]
    expected_overlap = [
        row
        for row in lookup
        if isinstance(row.get("assigned_agents"), list) and len(row["assigned_agents"]) == 2
    ]
    staged_files = stage.get("files")
    if not isinstance(staged_files, dict) or not staged_files:
        raise RuntimeError("production judge stage lacks its immutable input census")
    for relative, expected_sha in staged_files.items():
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or not isinstance(expected_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
            or not (run_dir / relative_path).is_file()
            or _sha256(run_dir / relative_path) != expected_sha
        ):
            raise RuntimeError("production judge stage input census mismatch")
    if (
        marker.get("attempt") != attempt
        or marker.get("rollouts_sha256") != generation.get("rollouts_sha256")
        or marker.get("accepted_source_ids_sha256") != accepted["accepted_source_ids_sha256"]
        or stage.get("attempt") != attempt
        or stage.get("rollouts_sha256") != generation.get("rollouts_sha256")
        or stage.get("accepted_source_ids_sha256") != accepted["accepted_source_ids_sha256"]
        or (
            stage.get("n_accepted_source_items"),
            stage.get("n_accepted_prompts"),
            stage.get("n_expected_rollouts"),
        )
        != expected_counts
        or request.get("attempt") != attempt
        or request.get("rollouts_sha256") != generation.get("rollouts_sha256")
        or request.get("accepted_source_ids_sha256") != accepted["accepted_source_ids_sha256"]
        or (
            request.get("n_accepted_source_items"),
            request.get("n_accepted_prompts"),
            request.get("n_expected_rollouts"),
        )
        != expected_counts
        or request.get("lookup_sha256") != census["judge/wave_lookup.json"]
        or request.get("packet_manifest_sha256") != census["judge/wave_packet_manifest.json"]
        or request.get("runtime_identity")
        != {
            "path": "judge/wave_runtime_identity.json",
            "sha256": census["judge/wave_runtime_identity.json"],
        }
        or request.get("n_requests") != len(lookup)
        or request.get("n_requests") != expected_counts[2]
        or request.get("n_overlap") != len(expected_overlap)
        or packet_manifest.get("packet_kind") != f"production-wave-attempt{attempt}"
        or packet_manifest.get("runtime_identity_sha256")
        != census["judge/wave_runtime_identity.json"]
        or summary.get("attempt") != attempt
        or summary.get("rollouts_sha256") != generation.get("rollouts_sha256")
        or summary.get("scores_sha256") != census["judge/wave_scores.jsonl"]
        or summary.get("request_manifest_sha256") != census["judge/wave_request_manifest.json"]
        or summary.get("runtime_identity_sha256") != census["judge/wave_runtime_identity.json"]
        or summary.get("overlap_joined_sha256") != census["judge/wave_overlap_joined.jsonl"]
        or summary.get("n") != len(scores)
        or summary.get("n_valid") != len(scores)
        or summary.get("n_overlap_joined") != len(overlap)
        or (
            summary.get("accepted_source_items"),
            summary.get("accepted_prompts"),
            summary.get("accepted_draws"),
        )
        != expected_counts
        or score_ids != lookup_ids
        or len(score_ids) != len(set(score_ids))
        or [row.get("item_id") for row in overlap]
        != [row.get("item_id") for row in expected_overlap]
    ):
        raise RuntimeError("production judge attempt identity/coverage linkage mismatch")

    lookup_by_id = {row["item_id"]: row for row in lookup}
    for score in scores:
        source = lookup_by_id[score["item_id"]]
        fields = ("prompt_id", "source_prompt_id", "topic", "language", "content", "frame", "draw")
        if (
            not isinstance(score.get("verdict"), bool)
            or score.get("judge_id") != source.get("primary_agent")
            or any(score.get(field) != source.get(field) for field in fields)
        ):
            raise RuntimeError("production judge score differs from its frozen assignment")
    for joined, source in zip(overlap, expected_overlap, strict=True):
        secondary = next(
            (agent for agent in source["assigned_agents"] if agent != source["primary_agent"]),
            None,
        )
        if (
            any(
                joined.get(field) != source.get(field)
                for field in (
                    "item_id",
                    "prompt_id",
                    "source_prompt_id",
                    "language",
                    "content",
                    "frame",
                    "topic",
                )
            )
            or joined.get("primary_judge") != source.get("primary_agent")
            or joined.get("secondary_judge") != secondary
            or not isinstance(joined.get("primary_label"), bool)
            or not isinstance(joined.get("secondary_label"), bool)
            or joined.get("disagreement")
            != (joined.get("primary_label") != joined.get("secondary_label"))
            or joined.get("secondary_judge_contrast")
            != int(joined["secondary_label"]) - int(joined["primary_label"])
        ):
            raise RuntimeError("joined overlap artifact differs from frozen assignments")

    assignment_counts = {
        "by_judge": dict(Counter(row["judge_id"] for row in scores)),
        "by_language": dict(Counter(row["language"] for row in scores)),
        "by_arm": dict(
            Counter(f"{row['language']}:{row['content']}:{row['frame']}" for row in scores)
        ),
        "by_topic": dict(Counter(row["topic"] for row in scores)),
    }
    agent_artifact_hashes = summary.get("agent_artifact_hashes")
    if not isinstance(agent_artifact_hashes, dict):
        raise RuntimeError("production judge summary lacks packet artifact hashes")
    agent_prefix = f"judge/agent_artifacts/wave_attempt{attempt}/"
    expected_agent_census: dict[str, dict[str, str]] = defaultdict(dict)
    suffix_to_field = {
        ".packet.json": "packet_sha256",
        ".output.jsonl": "output_sha256",
        ".output_manifest.json": "output_manifest_sha256",
    }
    for relative, sha in census.items():
        if not relative.startswith(agent_prefix):
            continue
        suffix = next((value for value in suffix_to_field if relative.endswith(value)), None)
        if suffix is None:
            raise RuntimeError("unrecognized production judge packet artifact in census")
        batch_key = relative[len(agent_prefix) : -len(suffix)]
        expected_agent_census[batch_key][suffix_to_field[suffix]] = sha
    if agent_artifact_hashes != dict(expected_agent_census):
        raise RuntimeError("production judge packet artifact summary/census mismatch")
    prompt_valid_counts = Counter(row["prompt_id"] for row in scores if row["verdict"] is not None)
    prompts_by_condition: dict[str, set[str]] = defaultdict(set)
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        prompts_by_condition[f"{row['language']}:{row['content']}"].add(row["prompt_id"])
        rows_by_source[row["source_prompt_id"]].append(row)
    validity_denominators = {
        key: {
            "n_prompts_total": len(prompt_ids),
            "n_prompts_any_valid": sum(
                prompt_valid_counts[prompt_id] >= 1 for prompt_id in prompt_ids
            ),
            "n_prompts_all_eight_valid": sum(
                prompt_valid_counts[prompt_id] == N_DRAWS for prompt_id in prompt_ids
            ),
        }
        for key, prompt_ids in sorted(prompts_by_condition.items())
    }
    completeness = {
        key: values["n_prompts_all_eight_valid"] / values["n_prompts_total"]
        for key, values in validity_denominators.items()
    }
    complete_sources = sorted(
        source_id
        for source_id, rows in rows_by_source.items()
        if len(rows) == PROMPTS_PER_SOURCE * N_DRAWS
        and len({row["item_id"] for row in rows}) == PROMPTS_PER_SOURCE * N_DRAWS
    )
    analysis_sources = sorted(set(accepted["accepted_source_ids"]) & set(complete_sources))
    analysis_topics = {rows_by_source[source_id][0]["topic"] for source_id in analysis_sources}
    analysis_subset_passed = len(analysis_sources) >= 81 and len(analysis_topics) == 12
    reliability_passed = summary.get("interjudge_reliability_passed")
    if summary.get("passed") is not True:
        raise RuntimeError("production judge technical/coverage gate failed")
    if (
        not isinstance(reliability_passed, bool)
        or summary.get("claim_eligible") != (summary["passed"] and reliability_passed)
        or not isinstance(summary.get("interjudge_reliability"), dict)
        or summary.get("primary_assignment_counts") != assignment_counts
        or summary.get("prompt_validity_denominators_by_language_content") != validity_denominators
        or summary.get("prompt_completeness_by_language_content") != completeness
        or summary.get("complete_source_item_ids") != complete_sources
        or summary.get("analysis_source_item_ids") != analysis_sources
        or summary.get("analysis_subset_passed") != analysis_subset_passed
        or summary.get("planned_source_items") != REGISTERED_SOURCE_ITEMS
        or summary.get("realized_complete_source_items") != len(complete_sources)
        or summary.get("realized_valid_rows") != len(scores)
    ):
        raise RuntimeError("production judge eligibility/assignment/completeness metadata invalid")
    disagreements = sum(bool(row["disagreement"]) for row in overlap)
    return {
        "summary": summary,
        "request": request,
        "overlap": {
            "requested_fraction": request.get("overlap_fraction"),
            "requested_count": request.get("n_overlap"),
            "joined_count": len(overlap),
            "joined_sha256": census["judge/wave_overlap_joined.jsonl"],
            "disagreements": disagreements,
            "disagreement_rate": disagreements / len(overlap) if overlap else None,
        },
    }


def _load_new_data(run_dir: Path, *, attempt: int = 1) -> dict[str, Any]:
    """Load verified analysis inputs and preserve the declared judge measurement contract."""
    judge_dir = _judge_dir(run_dir, attempt)
    required_uploads = (
        run_dir / "inputs" / "upload_verified.json",
        run_dir / "manifests" / "raw_upload.json",
        run_dir / "manifests" / "capture_upload.json",
        judge_dir / "upload.json",
    )
    missing_uploads = [str(path) for path in required_uploads if not path.exists()]
    if missing_uploads:
        raise RuntimeError(f"analysis blocked before upload verification: {missing_uploads}")
    bank = _jsonl(run_dir / "inputs" / "prompt_bank.jsonl")
    audit = json.loads((run_dir / "inputs" / "bank_audit_report.json").read_text())
    accepted = _accepted_bank_contract(bank, audit)
    input_marker = json.loads((run_dir / "inputs" / "upload_verified.json").read_text())
    generation = json.loads((run_dir / "manifests" / "generation.json").read_text())
    capture = json.loads((run_dir / "manifests" / "capture.json").read_text())
    raw_upload = json.loads((run_dir / "manifests" / "raw_upload.json").read_text())
    capture_upload = json.loads((run_dir / "manifests" / "capture_upload.json").read_text())
    input_stage = json.loads((run_dir / "manifests" / "input_stage.json").read_text())
    done = json.loads((run_dir / "issue952_china_definitive_done.json").read_text())
    judge_summary = json.loads((judge_dir / "wave_summary.json").read_text())
    judge_marker = json.loads((judge_dir / "upload.json").read_text())
    bank_path = run_dir / "inputs" / "prompt_bank.jsonl"
    audit_path = run_dir / "inputs" / "bank_audit_report.json"
    scores_path = judge_dir / "wave_scores.jsonl"
    summary_path = judge_dir / "wave_summary.json"
    validate_analysis_attempt_identity(
        local_bank_sha256=_sha256(bank_path),
        input_marker=input_marker,
        generation=generation,
        capture=capture,
        judge_summary=judge_summary,
    )
    _validate_gpu_attempt(
        run_dir=run_dir,
        attempt=attempt,
        accepted=accepted,
        input_marker=input_marker,
        input_stage=input_stage,
        generation=generation,
        capture=capture,
        raw_upload=raw_upload,
        capture_upload=capture_upload,
        done=done,
    )
    judge_contract = _validate_judge_attempt(
        run_dir=run_dir,
        judge_dir=judge_dir,
        attempt=attempt,
        marker=judge_marker,
        accepted=accepted,
        generation=generation,
    )
    if (
        input_marker.get("prompt_bank_sha256") != _sha256(bank_path)
        or input_marker.get("bank_audit_report_sha256") != _sha256(audit_path)
        or input_marker.get("prompt_bank_sha256") != generation.get("regime", {}).get("bank_sha256")
        or input_marker.get("prompt_bank_sha256")
        != capture.get("capture_regime", {}).get("bank_sha256")
        or judge_marker.get("wave_scores_sha256") != _sha256(scores_path)
        or judge_marker.get("wave_summary_sha256") != _sha256(summary_path)
    ):
        raise RuntimeError("analysis input/judge payload differs from upload verification marker")
    if (
        audit.get("passed") is not True
        or judge_summary.get("passed") is not True
        or judge_summary.get("remediation_complete") is not True
        or judge_summary.get("scores_sha256") != _sha256(scores_path)
    ):
        raise RuntimeError("analysis blocked by input-bank or classifier-wave integrity gate")
    rollouts = _jsonl(run_dir / "raw_completions" / "rollouts.jsonl")
    judge = _jsonl(judge_dir / "wave_scores.jsonl")
    vc_store = torch.load(
        run_dir / "analysis_tensors" / "vc.pt", map_location="cpu", weights_only=False
    )
    va_files = sorted((run_dir / "analysis_tensors").glob("va_*.pt"))
    rollout_ids = [row["item_id"] for row in rollouts]
    if (
        len(rollouts) != accepted["n_expected_rollouts"]
        or len(judge) != accepted["n_expected_rollouts"]
        or rollout_ids != accepted["expected_rollout_ids"]
        or generation.get("n_prompts") != accepted["n_accepted_prompts"]
        or generation.get("n_rows") != accepted["n_expected_rollouts"]
        or generation.get("regime", {}).get("accepted_source_ids_sha256")
        != accepted["accepted_source_ids_sha256"]
        or capture.get("n_contexts") != accepted["n_accepted_prompts"]
        or capture.get("n_answer_rows") != accepted["n_expected_rollouts"]
        or not va_files
    ):
        raise RuntimeError("new-run accepted production coverage is incomplete")
    if _sha256(run_dir / "raw_completions" / "rollouts.jsonl") != generation["rollouts_sha256"]:
        raise RuntimeError("new-run rollout hash drift")
    if (
        raw_upload.get("rollouts_sha256") != generation["rollouts_sha256"]
        or raw_upload.get("generation_manifest_sha256")
        != _sha256(run_dir / "manifests" / "generation.json")
        or raw_upload.get("generation_fingerprint") != _generation_fingerprint(generation)
        or capture_upload.get("capture_manifest_sha256")
        != _sha256(run_dir / "manifests" / "capture.json")
        or capture_upload.get("raw_upload_manifest_sha256")
        != _sha256(run_dir / "manifests" / "raw_upload.json")
        or capture_upload.get("capture_fingerprint") != _capture_fingerprint(capture)
    ):
        raise RuntimeError("generation/capture upload marker identity mismatch")
    vc_path = run_dir / "analysis_tensors" / "vc.pt"
    expected_va = capture.get("va_files")
    if (
        not isinstance(expected_va, dict)
        or {path.name for path in va_files} != set(expected_va)
        or capture.get("vc_sha256") != _sha256(vc_path)
        or capture.get("rollouts_sha256") != generation["rollouts_sha256"]
        or capture_upload.get("vc_sha256") != capture.get("vc_sha256")
        or capture_upload.get("va_files") != expected_va
    ):
        raise RuntimeError("capture manifest file set/context/rollout identity mismatch")
    for path in va_files:
        if _sha256(path) != expected_va[path.name]:
            raise RuntimeError(f"answer capture hash mismatch: {path.name}")
    passing = set(accepted["accepted_source_ids"])
    bank_by_id = {row["item_id"]: row for row in bank}
    accepted_bank_by_id = {row["item_id"]: row for row in accepted["accepted_rows"]}
    vc_ids = list(vc_store["item_ids"])
    if (
        vc_store["layers"] != list(LAYERS)
        or vc_ids != accepted["accepted_prompt_ids"]
        or set(vc_ids) != set(accepted_bank_by_id)
        or vc_store.get("dtype") != "fp32"
        or vc_store.get("position") != "context_last_generation_prompt_token"
        or vc_store.get("model_revision") != capture.get("model_revision")
        or vc_store.get("bank_sha256") != _sha256(bank_path)
        or vc_store.get("capture_regime") != capture.get("capture_regime")
        or vc_store.get("capture_regime_fp") != capture.get("capture_regime_fp")
        or vc_store["vc"].shape != (len(vc_ids), len(LAYERS), HIDDEN)
        or vc_store["vc"].dtype != torch.float32
    ):
        raise RuntimeError("context-vector ids/layers/capture metadata mismatch")
    vc_pos = {item_id: i for i, item_id in enumerate(vc_ids)}
    vc = vc_store["vc"].double().numpy()
    vc_by_id = {item_id: vc[i] for item_id, i in vc_pos.items()}

    va_by_item: dict[str, np.ndarray] = {}
    rollout_by_item = {row["item_id"]: row for row in rollouts}
    for path in va_files:
        store = torch.load(path, map_location="cpu", weights_only=False)
        index, vectors = _validated_answer_shard(
            store=store,
            rollout_by_item=rollout_by_item,
            capture=capture,
            generation=generation,
            name=path.name,
        )
        for i, rec in enumerate(index):
            if rec["item_id"] in va_by_item:
                raise RuntimeError("duplicate answer-vector item id")
            va_by_item[rec["item_id"]] = vectors[i].double().numpy()
    judge_by_item = {row["item_id"]: row for row in judge}
    if set(va_by_item) != set(rollout_by_item) or set(judge_by_item) != set(rollout_by_item):
        raise RuntimeError("answer-vector/rollout/judge id sets differ")
    complete = set(judge_summary.get("complete_source_item_ids", []))
    if not complete <= passing:
        raise RuntimeError("classifier-complete source ids are absent from the accepted bank")
    source_ids = sorted(passing & complete)
    if source_ids != judge_summary.get("analysis_source_item_ids"):
        raise RuntimeError("analysis source subset differs from the frozen judge gate")
    if not source_ids:
        raise RuntimeError("no complete source item remains for analysis")
    topic_by_source = {}
    prompt_key = {}
    for row in bank:
        topic_by_source[row["source_prompt_id"]] = row["topic"]
        prompt_key[(row["source_prompt_id"], row["language"], row["content"], row["frame"])] = row[
            "item_id"
        ]
    answer_mean = {}
    refusal_rate = {}
    lexical_rate = {}
    complete_source_ids = sorted(complete)
    complete_prompt_ids = {
        prompt_key[(source_id, language, content, frame)]
        for source_id in complete_source_ids
        for language in ("en", "zh")
        for content in ("sensitive_full", "sensitive_country_neutral", "matched_non_china")
        for frame in ("direct", "academic")
    }
    for prompt_id in sorted(complete_prompt_ids):
        draw_ids = [f"{prompt_id}-d{draw}" for draw in range(N_DRAWS)]
        if any(item not in va_by_item for item in draw_ids):
            raise RuntimeError(f"missing answer draw for {prompt_id}")
        if any(judge_by_item[item]["verdict"] is None for item in draw_ids):
            raise RuntimeError(f"classifier-complete source has invalid draw: {prompt_id}")
        answer_mean[prompt_id] = np.stack([va_by_item[item] for item in draw_ids]).mean(0)
        refusal_rate[prompt_id] = np.mean([judge_by_item[item]["verdict"] for item in draw_ids])
        lexical_rate[prompt_id] = np.mean(
            [lexical_refusal(rollout_by_item[item]["text"]) for item in draw_ids]
        )
    return {
        "bank": bank,
        "judge_measurement": {
            "measurement_contract": judge_summary["measurement_contract"],
            "historical_labels_role": judge_summary["historical_labels_role"],
            "historical_comparability": judge_summary["historical_comparability"],
            "technical_wave_passed": judge_summary["passed"],
            "claim_eligible": judge_summary["claim_eligible"],
            "interjudge_reliability_passed": judge_summary["interjudge_reliability_passed"],
            "interjudge_reliability": judge_summary["interjudge_reliability"],
            "overlap": judge_contract["overlap"],
            "primary_assignment_counts": judge_summary["primary_assignment_counts"],
            "prompt_completeness_by_language_content": judge_summary[
                "prompt_completeness_by_language_content"
            ],
            "prompt_validity_denominators_by_language_content": judge_summary[
                "prompt_validity_denominators_by_language_content"
            ],
        },
        "bank_by_id": bank_by_id,
        "source_ids": source_ids,
        "sensitivity_source_ids": complete_source_ids,
        "topic_by_source": topic_by_source,
        "topics": np.array([topic_by_source[item] for item in source_ids]),
        "prompt_key": prompt_key,
        "vc": vc_by_id,
        "answer_mean": answer_mean,
        "va_draw": va_by_item,
        "judge": judge_by_item,
        "refusal_rate": refusal_rate,
        "lexical_rate": lexical_rate,
        "rollouts": rollout_by_item,
        "maximum_registered": {
            "items": REGISTERED_SOURCE_ITEMS,
            "prompts": REGISTERED_PROMPTS,
            "draws": REGISTERED_PROMPTS * N_DRAWS,
        },
        "accepted_planned": {
            "items": len(passing),
            "prompts": accepted["n_accepted_prompts"],
            "draws": accepted["n_expected_rollouts"],
            "source_ids_sha256": accepted["accepted_source_ids_sha256"],
        },
        "realized": {
            "items_primary": len(source_ids),
            "items_generated": len(passing),
            "items_sensitivity_all_complete": len(complete_source_ids),
            "items_audit_failed": REGISTERED_SOURCE_ITEMS - len(passing),
            "items_classifier_incomplete": len(passing - complete),
            "prompts_registered": len(bank),
            "prompts_generated": accepted["n_accepted_prompts"],
            "prompts_primary": len(source_ids) * 12,
            "draws_generated": len(rollouts),
            "draws_primary": len(source_ids) * PROMPTS_PER_SOURCE * N_DRAWS,
            "draws_sensitivity_all_complete": len(complete_source_ids)
            * PROMPTS_PER_SOURCE
            * N_DRAWS,
            "cap_hits": int(generation["n_cap_hit"]),
            "missing_context_vectors": accepted["n_accepted_prompts"] - len(vc_ids),
            "missing_answer_vectors": accepted["n_expected_rollouts"] - len(va_by_item),
            "empty_answer_vectors": int(capture["n_empty_answer_rows"]),
            "classifier_valid": int(judge_summary["n_valid"]),
            "classifier_parse_drops": int(judge_summary["n_parse_drop"]),
            "classifier_api_refusals": judge_summary["n_api_refusal"],
            "classifier_transport_losses": judge_summary["n_transport"],
        },
    }


REFUSAL_PATTERNS = re.compile(
    r"(?i)\b(?:i\s+(?:can(?:not|'t)|won't|am\s+unable)|unable\s+to\s+assist|must\s+refuse)\b"
    r"|(?:抱歉.{0,20})?(?:无法|不能|不便)(?:回答|提供|协助|帮助)|拒绝(?:回答|提供)"
)


def lexical_refusal(text: str) -> bool:
    return bool(REFUSAL_PATTERNS.search(text))


def _prompt_arrays(data: dict[str, Any], language: str, value_key: str) -> np.ndarray:
    ids = data["source_ids"]
    key = data["prompt_key"]
    source = data[value_key]
    arr = []
    for source_id in ids:
        content_rows = []
        for content in ("sensitive_full", "sensitive_country_neutral", "matched_non_china"):
            frames = [
                source[key[(source_id, language, content, frame)]]
                for frame in ("direct", "academic")
            ]
            content_rows.append(frames)
        arr.append(content_rows)
    return np.asarray(arr)


def _context_contrasts(data: dict[str, Any], language: str) -> dict[str, np.ndarray]:
    x = _prompt_arrays(data, language, "vc")  # N, content, frame, layer, H
    subject = (x[:, 1] - x[:, 2]).mean(axis=1)
    country = (x[:, 0] - x[:, 1]).mean(axis=1)
    framing = (x[:, :, 1] - x[:, :, 0]).mean(axis=1)
    neutral = x[:, 1].mean(axis=1)
    return {"subject": subject, "country": country, "framing": framing, "neutral": neutral}


def _answer_subject(data: dict[str, Any], language: str) -> np.ndarray:
    x = _prompt_arrays(data, language, "answer_mean")
    return (x[:, 1] - x[:, 2]).mean(axis=1)


def _scalar_subject(data: dict[str, Any], language: str, key_name: str) -> np.ndarray:
    x = _prompt_arrays(data, language, key_name)
    return (x[:, 1] - x[:, 2]).mean(axis=1)


def _draw_refusal_deltas(data: dict[str, Any], language: str, draws: range) -> np.ndarray:
    key = data["prompt_key"]
    out = []
    for source_id in data["source_ids"]:
        conditions = []
        for content in ("sensitive_country_neutral", "matched_non_china"):
            vals = []
            for frame in ("direct", "academic"):
                pid = key[(source_id, language, content, frame)]
                vals.extend(bool(data["judge"][f"{pid}-d{draw}"]["verdict"]) for draw in draws)
            conditions.append(float(np.mean(vals)))
        out.append(conditions[0] - conditions[1])
    return np.asarray(out)


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denom, np.finfo(np.float64).tiny)


def _within_topic_targets(topics: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(topics)
    out = np.broadcast_to(np.arange(n), (n_perm, n)).copy()
    for topic in np.unique(topics):
        idx = np.flatnonzero(topics == topic)
        permutations = np.argsort(rng.random((n_perm, len(idx))), axis=1)
        out[:, idx] = idx[permutations]
    return out


def leave_one_topic_out_gain_prediction(
    predicted: np.ndarray, observed: np.ndarray, topics: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit one scalar gain off-topic and return leakage-free predictions."""

    out = np.empty_like(predicted)
    gains: dict[str, float] = {}
    for topic in np.unique(topics):
        test = topics == topic
        train = ~test
        denom = float(np.sum(predicted[train] ** 2))
        if denom <= 0:
            raise RuntimeError(f"zero predicted energy outside topic {topic}")
        gain = float(np.sum(predicted[train] * observed[train]) / denom)
        out[test] = gain * predicted[test]
        gains[str(topic)] = gain
    return out, gains


def _refusal_kappa(data: dict[str, Any], language: str) -> float:
    key = data["prompt_key"]
    y, z = [], []
    for source_id in data["source_ids"]:
        for content in ("sensitive_full", "sensitive_country_neutral", "matched_non_china"):
            for frame in ("direct", "academic"):
                pid = key[(source_id, language, content, frame)]
                for draw in range(N_DRAWS):
                    iid = f"{pid}-d{draw}"
                    y.append(bool(data["judge"][iid]["verdict"]))
                    z.append(lexical_refusal(data["rollouts"][iid]["text"]))
    y, z = np.asarray(y), np.asarray(z)
    po = np.mean(y == z)
    py, pz = np.mean(y), np.mean(z)
    pe = py * pz + (1 - py) * (1 - pz)
    return float((po - pe) / (1 - pe)) if pe < 1 else math.nan


def audit_inclusion_sensitivity(
    data: dict[str, Any],
    maps: dict[int, tuple[dict[str, Any], np.ndarray, np.ndarray, int, float]],
    axis_store: dict[str, Any],
) -> dict[str, Any]:
    """Descriptively rerun H1--H4 after including audit-failed complete items."""

    sensitivity_ids = list(data["sensitivity_source_ids"])
    primary_ids = set(data["source_ids"])
    sensitivity_data = {
        **data,
        "source_ids": sensitivity_ids,
        "topics": np.asarray([data["topic_by_source"][item] for item in sensitivity_ids]),
    }
    topics = sensitivity_data["topics"]
    records: list[dict[str, Any]] = []
    for layer_pos, layer in enumerate(LAYERS):
        bundle, u_all, _, retained_rank, _ = maps[layer]
        u = u_all[:, :retained_rank]
        xmu = bundle["xmu"].double().numpy()
        a_map = bundle["W"].double().numpy() / bundle["xsd"].double().numpy()[:, None]
        axis = axis_store["axis"][layer_pos].double().numpy()
        contexts = {
            language: _context_contrasts(sensitivity_data, language) for language in ("en", "zh")
        }
        centered = {
            language: contexts[language]["neutral"][:, layer_pos] - xmu for language in ("en", "zh")
        }
        retained = {language: centered[language] @ u for language in ("en", "zh")}
        kernel = {
            language: centered[language] - retained[language] @ u.T for language in ("en", "zh")
        }
        for direction, query_language, gallery_language in (
            ("en_to_zh", "en", "zh"),
            ("zh_to_en", "zh", "en"),
        ):
            truth = np.arange(len(topics))
            retained_prediction = retrieval_predictions(
                retained[query_language], retained[gallery_language], topics
            )
            kernel_prediction = retrieval_predictions(
                kernel[query_language], kernel[gallery_language], topics
            )
            records.append(
                {
                    "layer": layer,
                    "hypothesis": "H1",
                    "direction": direction,
                    "retained_top1": float(np.mean(retained_prediction == truth)),
                    "kernel_top1": float(np.mean(kernel_prediction == truth)),
                }
            )
        for language in ("en", "zh"):
            contrasts = contexts[language]
            subject = contrasts["subject"][:, layer_pos]
            country = contrasts["country"][:, layer_pos]
            framing = contrasts["framing"][:, layer_pos]
            shares = {
                name: kernel_share(values, u)
                for name, values in {
                    "subject": subject,
                    "country": country,
                    "framing": framing,
                }.items()
            }
            records.append(
                {
                    "layer": layer,
                    "hypothesis": "H2",
                    "language": language,
                    "kernel_share_median": {
                        name: float(np.median(values)) for name, values in shares.items()
                    },
                    "nuisance_minus_subject_median": {
                        name: float(np.median(shares[name] - shares["subject"]))
                        for name in ("country", "framing")
                    },
                }
            )
            observed = _answer_subject(sensitivity_data, language)[:, layer_pos]
            predictions = {
                "full": subject @ a_map,
                "retained": (subject @ u) @ (u.T @ a_map),
                "identity": subject,
            }
            predictions["kernel"] = predictions["full"] - predictions["retained"]
            denominator = float(np.sum(observed**2))
            records.append(
                {
                    "layer": layer,
                    "hypothesis": "H3",
                    "language": language,
                    "mean_cosine": {
                        name: float(np.nanmean(_cos_rows(prediction, observed)))
                        for name, prediction in predictions.items()
                    },
                    "raw_r2": {
                        name: float(1 - np.sum((observed - prediction) ** 2) / denominator)
                        if denominator > 0
                        else math.nan
                        for name, prediction in predictions.items()
                    },
                }
            )
            refusal_delta = _scalar_subject(sensitivity_data, language, "refusal_rate")
            lexical_delta = _scalar_subject(sensitivity_data, language, "lexical_rate")
            predicted_axis = predictions["full"] @ axis
            records.append(
                {
                    "layer": layer,
                    "hypothesis": "H4",
                    "language": language,
                    "refusal_delta_mean": float(np.mean(refusal_delta)),
                    "predicted_axis_refusal_rho": float(
                        spearmanr(predicted_axis, refusal_delta).statistic
                    ),
                    "lexical_delta_mean": float(np.mean(lexical_delta)),
                    "classifier_lexical_kappa": _refusal_kappa(sensitivity_data, language),
                }
            )
    added_ids = sorted(set(sensitivity_ids) - primary_ids)
    return {
        "label": "secondary descriptive all-classifier-complete audit-inclusion sensitivity",
        "affects_primary_inference_or_verdict": False,
        "n_source_items": len(sensitivity_ids),
        "n_primary_source_items": len(primary_ids),
        "n_audit_failed_included": len(added_ids),
        "audit_failed_included_source_item_ids": added_ids,
        "records": records,
    }


def _analysis_input_hashes(
    run_dir: Path, map_root: Path, axis_dir: Path, *, attempt: int
) -> dict[str, Any]:
    """Hash every load-bearing analysis input for pilot/production identity."""
    capture = json.loads((run_dir / "manifests" / "capture.json").read_text())
    va_paths = sorted((run_dir / "analysis_tensors").glob("va_*.pt"))
    judge_dir = _judge_dir(run_dir, attempt)
    judge_marker = json.loads((judge_dir / "upload.json").read_text())
    judge_census = _validated_judge_census(judge_marker)
    return {
        "bank": _sha256(run_dir / "inputs" / "prompt_bank.jsonl"),
        "bank_upload_marker": _sha256(run_dir / "inputs" / "upload_verified.json"),
        "rollouts": _sha256(run_dir / "raw_completions" / "rollouts.jsonl"),
        "gpu_manifests": {
            name: _sha256(run_dir / "manifests" / name)
            for name in (
                "input_stage.json",
                "generation.json",
                "raw_upload.json",
                "capture.json",
                "capture_upload.json",
            )
        },
        "gpu_done": _sha256(run_dir / "issue952_china_definitive_done.json"),
        "judge_attempt": attempt,
        "judge_upload_marker": _sha256(judge_dir / "upload.json"),
        "judge_artifact_census": {
            relative: _sha256(judge_dir / Path(relative).relative_to("judge"))
            for relative in sorted(judge_census)
        },
        "capture_manifest": _sha256(run_dir / "manifests" / "capture.json"),
        "context_capture": _sha256(run_dir / "analysis_tensors" / "vc.pt"),
        "answer_captures": {path.name: _sha256(path) for path in va_paths},
        "capture_manifest_declared": {
            "vc_sha256": capture.get("vc_sha256"),
            "va_files": capture.get("va_files"),
        },
        "axis": _sha256(axis_dir / "refusal_axis.pt"),
        "axis_report": _sha256(axis_dir / "refusal_axis_report.json"),
        "maps": {str(layer): _sha256(map_root / f"L{layer}" / "ridge.pt") for layer in LAYERS},
    }


def _classify_h4(
    *,
    judge_claim_eligible: bool,
    judge_reliability_passed: bool,
    core_eligible: bool,
    threshold_stable: bool,
    lexical_consistent: bool,
    supported: bool,
) -> str:
    """Classify H4 while making judge reliability an explicit eligibility gate."""
    if not judge_reliability_passed:
        return "judge-reliability-indeterminate"
    if not judge_claim_eligible:
        return "judge-measurement-indeterminate"
    if not core_eligible:
        return "ineligible"
    if not threshold_stable:
        return "threshold-sensitive"
    if not lexical_consistent:
        return "classifier-indeterminate"
    return "classifier-linked" if supported else "classifier-null"


def _coverage_report(data: dict[str, Any]) -> dict[str, Any]:
    """Separate maximum registered, accepted planned, and realized coverage."""
    return {
        "maximum_registered": data["maximum_registered"],
        "accepted_planned": data["accepted_planned"],
        "realized": data["realized"],
    }


def _planned_vs_realized_report(data: dict[str, Any]) -> dict[str, Any]:
    """Preserve the established report key with all three coverage scopes."""
    return {
        "maximum_registered": data["maximum_registered"],
        "accepted_planned": data["accepted_planned"],
        "realized": data["realized"],
    }


def run_analysis(
    run_dir: Path,
    map_root: Path,
    axis_dir: Path,
    out_dir: Path,
    n_random: int,
    n_resample: int,
    *,
    production: bool,
    attempt: int = 1,
) -> dict[str, Any]:
    """Run the registered analysis and report its exact judge measurement provenance."""
    t0 = time.time()
    if production and (n_random != N_RANDOM or n_resample != N_BOOT):
        raise RuntimeError("production analysis requires exactly 1000 random/10000 resamples")
    input_hashes = _analysis_input_hashes(run_dir, map_root, axis_dir, attempt=attempt)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if production:
        pilot_path = out_dir / "analysis_pilot_report.json"
        if not pilot_path.exists():
            raise RuntimeError("production analysis requires a passed CPU pilot report")
        pilot = json.loads(pilot_path.read_text())
        if (
            pilot.get("passed") is not True
            or pilot.get("git_sha") != git_sha
            or pilot.get("input_hashes") != input_hashes
        ):
            raise RuntimeError("CPU pilot is stale, failed, or input-mismatched")
    data = _load_new_data(run_dir, attempt=attempt)
    axis_store = torch.load(axis_dir / "refusal_axis.pt", map_location="cpu", weights_only=False)
    axis_report = json.loads((axis_dir / "refusal_axis_report.json").read_text())
    if axis_report.get("axis_sha256") != _sha256(axis_dir / "refusal_axis.pt"):
        raise RuntimeError("refusal-axis report does not bind the loaded axis artifact")
    if axis_store["layers"] != list(LAYERS) or axis_store["axis"].shape != (3, HIDDEN):
        raise RuntimeError("historical refusal-axis schema changed")
    topics = data["topics"]
    boot_weights = topic_bootstrap_weights(topics, n_resample, SEED + 1)
    perm_targets = _within_topic_targets(topics, n_resample, SEED + 2)
    layer_records = []
    item_records: list[dict[str, Any]] = []
    sensitivity_records: list[dict[str, Any]] = []
    random_arrays: dict[str, np.ndarray] = {}
    null_arrays: dict[str, np.ndarray] = {}
    h1_p, h2_cell_p, h3_p, h4_p = [], [], [], []
    h1_refs, h2_refs, h3_refs, h4_refs = [], [], [], []

    maps = {}
    for layer in LAYERS:
        path = map_root / f"L{layer}" / "ridge.pt"
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if bundle["W"].shape != (HIDDEN, HIDDEN) or int(bundle["layer"]) != layer:
            raise RuntimeError(f"map schema mismatch at L{layer}")
        u_all, singular = operator_svd(bundle["W"], bundle["xsd"])
        rank, tau = mass_rank(singular, MASS_PRIMARY)
        maps[layer] = (bundle, u_all, singular, rank, tau)
    setup_elapsed = time.time() - t0

    checkpoint_regime = {
        "input_hashes": input_hashes,
        "git_sha": git_sha,
        "n_random": n_random,
        "n_resample": n_resample,
        "primary_mass": MASS_PRIMARY,
        "sensitivity_mass": list(MASS_SENSITIVITY),
        "seed": SEED,
    }
    checkpoint_regime_fp = hashlib.sha256(
        json.dumps(checkpoint_regime, sort_keys=True).encode()
    ).hexdigest()
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for layer_pos, layer in enumerate(LAYERS):
        checkpoint_path = checkpoint_dir / f"L{layer}.json"
        checkpoint_arrays_path = checkpoint_dir / f"L{layer}.npz"
        if checkpoint_path.exists() and checkpoint_arrays_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint.get("regime_fp") != checkpoint_regime_fp or checkpoint.get(
                "arrays_sha256"
            ) != _sha256(checkpoint_arrays_path):
                raise RuntimeError(f"stale or corrupt analysis checkpoint for L{layer}")
            with np.load(checkpoint_arrays_path, allow_pickle=False) as archive:
                loaded_arrays = {key: archive[key].copy() for key in archive.files}
            expected_keys = set(checkpoint.get("random_array_keys", [])) | set(
                checkpoint.get("null_array_keys", [])
            )
            if set(loaded_arrays) != expected_keys:
                raise RuntimeError(f"analysis checkpoint array inventory mismatch for L{layer}")
            random_arrays.update(
                {
                    key: value
                    for key, value in loaded_arrays.items()
                    if key in checkpoint["random_array_keys"]
                }
            )
            null_arrays.update(
                {
                    key: value
                    for key, value in loaded_arrays.items()
                    if key in checkpoint["null_array_keys"]
                }
            )
            layer_records.extend(checkpoint["layer_records"])
            item_records.extend(checkpoint["item_records"])
            sensitivity_records.extend(checkpoint["sensitivity_records"])
            for rec in checkpoint["layer_records"]:
                if rec["hypothesis"] == "H1":
                    h1_refs.append(rec)
                    h1_p.append(rec["permutation_p"])
                elif rec["hypothesis"] == "H2":
                    h2_refs.append(rec)
                    h2_cell_p.append(rec["iut_p"])
                elif rec["hypothesis"] == "H3":
                    h3_refs.extend(rec["components"].values())
                    h3_p.extend(
                        component["bootstrap_p"] for component in rec["components"].values()
                    )
                elif rec["hypothesis"] == "H4":
                    h4_refs.append(rec)
                    h4_p.append(rec["permutation_p"])
            print(f"[analysis] resume completed layer L{layer}")
            continue
        if checkpoint_path.exists() != checkpoint_arrays_path.exists():
            raise RuntimeError(f"partial analysis checkpoint for L{layer}")
        layer_record_start = len(layer_records)
        item_record_start = len(item_records)
        sensitivity_record_start = len(sensitivity_records)
        random_keys_start = set(random_arrays)
        null_keys_start = set(null_arrays)
        bundle, u_all, singular, retained_rank, tau = maps[layer]
        u = u_all[:, :retained_rank]
        xmu = bundle["xmu"].double().numpy()
        w = bundle["W"].double().numpy()
        xsd = bundle["xsd"].double().numpy()
        a_map = w / xsd[:, None]
        axis = axis_store["axis"][layer_pos].double().numpy()
        contexts = {lang: _context_contrasts(data, lang) for lang in ("en", "zh")}

        # H1: two directions, fixed unique-id gallery and rank-matched random nulls.
        centered = {lang: contexts[lang]["neutral"][:, layer_pos] - xmu for lang in ("en", "zh")}
        retained = {lang: centered[lang] @ u for lang in ("en", "zh")}
        kernel = {lang: centered[lang] - retained[lang] @ u.T for lang in ("en", "zh")}
        combined = np.concatenate([centered["en"], centered["zh"]], axis=0)
        random_pred = {
            "en_to_zh": np.empty((n_random, len(topics)), dtype=np.int64),
            "zh_to_en": np.empty((n_random, len(topics)), dtype=np.int64),
        }
        random_rank = {key: np.empty_like(value) for key, value in random_pred.items()}
        random_all_pred = {key: np.empty_like(value) for key, value in random_pred.items()}
        random_all_rank = {key: np.empty_like(value) for key, value in random_pred.items()}
        seeds = np.arange(n_random, dtype=np.int64) + SEED + layer * 10_000
        for start in range(0, n_random, RANDOM_BLOCK):
            block = structured_projection_coords_batch(
                combined, retained_rank, seeds[start : start + RANDOM_BLOCK]
            )
            en, zh = np.split(block, 2, axis=1)
            for direction, query, gallery in (
                ("en_to_zh", en, zh),
                ("zh_to_en", zh, en),
            ):
                pred, ranks = batch_retrieval_predictions_and_ranks(query, gallery, topics)
                random_pred[direction][start : start + len(block)] = pred
                random_rank[direction][start : start + len(block)] = ranks
                all_pred, all_ranks = batch_retrieval_predictions_and_ranks(
                    query, gallery, np.zeros(len(topics), dtype=np.int8)
                )
                random_all_pred[direction][start : start + len(block)] = all_pred
                random_all_rank[direction][start : start + len(block)] = all_ranks
        for direction in random_pred:
            random_arrays[f"L{layer}_{direction}_pred"] = random_pred[direction]
            random_arrays[f"L{layer}_{direction}_rank"] = random_rank[direction]
            random_arrays[f"L{layer}_{direction}_all_pred"] = random_all_pred[direction]
            random_arrays[f"L{layer}_{direction}_all_rank"] = random_all_rank[direction]
        for direction, qlang, glang in (
            ("en_to_zh", "en", "zh"),
            ("zh_to_en", "zh", "en"),
        ):
            pred_full, rank_full = retrieval_predictions_and_ranks(
                centered[qlang], centered[glang], topics
            )
            pred_ret, rank_ret = retrieval_predictions_and_ranks(
                retained[qlang], retained[glang], topics
            )
            pred_ker, rank_ker = retrieval_predictions_and_ranks(
                kernel[qlang], kernel[glang], topics
            )
            truth = np.arange(len(topics))
            hit_ret = pred_ret == truth
            hit_ker = pred_ker == truth
            hit_random = random_pred[direction] == truth[None]
            ret_acc = float(hit_ret.mean())
            ker_acc = float(hit_ker.mean())
            random_acc = float(hit_random.mean())
            random_mrr = float(np.mean(1.0 / random_rank[direction]))
            statistic = ret_acc - max(ker_acc, random_acc)
            random_counts = np.zeros((len(topics), len(topics)), dtype=np.int32)
            source_rows = np.broadcast_to(np.arange(len(topics))[:, None], (len(topics), n_random))
            np.add.at(
                random_counts,
                (source_rows.ravel(), random_pred[direction].T.ravel()),
                1,
            )
            perm_stat = np.empty(n_resample)
            for start in range(0, n_resample, 500):
                target = perm_targets[start : start + 500]
                row_idx = np.arange(len(topics))[None]
                p_ret = np.mean(pred_ret[None] == target, axis=1)
                p_ker = np.mean(pred_ker[None] == target, axis=1)
                p_rand = np.mean(random_counts[row_idx, target] / n_random, axis=1)
                perm_stat[start : start + len(target)] = p_ret - np.maximum(p_ker, p_rand)
            pvalue = float((1 + np.sum(perm_stat >= statistic)) / (n_resample + 1))
            null_arrays[f"H1_L{layer}_{direction}_permutation"] = perm_stat
            boot_stat = np.empty(n_resample)
            random_hit_mean = hit_random.mean(axis=0)
            for start in range(0, n_resample, 1000):
                weights = boot_weights[start : start + 1000]
                br = bootstrap_weighted_mean(hit_ret, weights)
                bk = bootstrap_weighted_mean(hit_ker, weights)
                bz = bootstrap_weighted_mean(random_hit_mean, weights)
                boot_stat[start : start + len(weights)] = br - np.maximum(bk, bz)
            null_arrays[f"H1_L{layer}_{direction}_bootstrap"] = boot_stat
            loto = {}
            for held_out_topic in np.unique(topics):
                keep = topics != held_out_topic
                loto[str(held_out_topic)] = float(
                    hit_ret[keep].mean() - max(hit_ker[keep].mean(), random_hit_mean[keep].mean())
                )

            all_pred = {}
            all_rank = {}
            for name, q, g in (
                ("full", centered[qlang], centered[glang]),
                ("retained", retained[qlang], retained[glang]),
                ("kernel", kernel[qlang], kernel[glang]),
            ):
                all_pred[name], all_rank[name] = retrieval_predictions_and_ranks(q, g, None)
            random_all_hits = np.mean(random_all_pred[direction] == truth[None], axis=1)
            random_all_mrr = np.mean(1.0 / random_all_rank[direction], axis=1)
            rec = {
                "layer": layer,
                "hypothesis": "H1",
                "direction": direction,
                "n_items": len(topics),
                "chance": float(np.mean([1 / np.sum(topics == t) for t in topics])),
                "retained_rank": retained_rank,
                "retained_top1": ret_acc,
                "kernel_top1": ker_acc,
                "random_top1_mean": random_acc,
                "full_top1": float(np.mean(pred_full == truth)),
                "mrr": {
                    "full": float(np.mean(1.0 / rank_full)),
                    "retained": float(np.mean(1.0 / rank_ret)),
                    "kernel": float(np.mean(1.0 / rank_ker)),
                    "random_mean": random_mrr,
                },
                "all_item_retrieval": {
                    name: {
                        "top1": float(np.mean(all_pred[name] == truth)),
                        "mrr": float(np.mean(1.0 / all_rank[name])),
                    }
                    for name in all_pred
                },
                "all_item_random": {
                    "top1_mean": float(random_all_hits.mean()),
                    "top1_mc_se": float(random_all_hits.std(ddof=1) / math.sqrt(n_random)),
                    "mrr_mean": float(random_all_mrr.mean()),
                    "mrr_mc_se": float(random_all_mrr.std(ddof=1) / math.sqrt(n_random)),
                },
                "random_top1_mc_se": float(
                    hit_random.mean(axis=1).std(ddof=1) / math.sqrt(n_random)
                ),
                "advantage": statistic,
                "bootstrap_ci": _ci(boot_stat),
                "leave_one_topic_out_advantage": loto,
                "permutation_p": pvalue,
                "passed_effect": statistic >= 0.10,
            }
            layer_records.append(rec)
            h1_refs.append(rec)
            h1_p.append(pvalue)
            for i, source_id in enumerate(data["source_ids"]):
                item_records.append(
                    {
                        "source_prompt_id": source_id,
                        "topic": str(topics[i]),
                        "layer": layer,
                        "language_or_direction": direction,
                        "hypothesis": "H1",
                        "full_hit": bool(pred_full[i] == i),
                        "retained_hit": bool(pred_ret[i] == i),
                        "kernel_hit": bool(pred_ker[i] == i),
                        "full_rank": int(rank_full[i]),
                        "retained_rank": int(rank_ret[i]),
                        "kernel_rank": int(rank_ker[i]),
                        "random_hit_rate": float(hit_random[:, i].mean()),
                        "random_mean_reciprocal_rank": float(
                            np.mean(1.0 / random_rank[direction][:, i])
                        ),
                    }
                )

        for language in ("en", "zh"):
            contrasts = contexts[language]
            subject = contrasts["subject"][:, layer_pos]
            country = contrasts["country"][:, layer_pos]
            framing = contrasts["framing"][:, layer_pos]
            shares = {
                "subject": kernel_share(subject, u),
                "country": kernel_share(country, u),
                "framing": kernel_share(framing, u),
            }
            distance_share, distance_pairs, distance_error = distance_matched_within_condition_null(
                subject, contrasts["neutral"][:, layer_pos], topics, u
            )
            direction_rng = np.random.default_rng(
                SEED + layer * 1_000 + (0 if language == "en" else 1)
            )
            direction_null = (
                np.zeros((n_random, len(topics)), dtype=np.float64)
                if retained_rank == HIDDEN
                else direction_rng.beta(
                    (HIDDEN - retained_rank) / 2,
                    retained_rank / 2,
                    size=(n_random, len(topics)),
                )
            )
            random_arrays[f"H2_L{layer}_{language}_isotropic_direction_kernel_share"] = (
                direction_null
            )
            advantages = {
                "country": shares["country"] - shares["subject"],
                "framing": shares["framing"] - shares["subject"],
            }
            component_p = {}
            component = {}
            for nuisance, values in advantages.items():
                observed = float(np.median(values))
                boot = bootstrap_weighted_median(values, boot_weights)
                component_p[nuisance] = bootstrap_p_greater(boot, observed)
                sign_count = int(np.sum(values > 0))
                sign_p = float(
                    binomtest(sign_count, int(np.sum(np.isfinite(values))), 0.5, "greater").pvalue
                )
                component[nuisance] = {
                    "median_advantage": observed,
                    "bootstrap_ci": _ci(boot),
                    "bootstrap_p": component_p[nuisance],
                    "positive_items": sign_count,
                    "sign_binomial_p": sign_p,
                }
                null_arrays[f"H2_L{layer}_{language}_{nuisance}_bootstrap"] = boot
            cell_p = max(component_p.values())
            h2 = {
                "layer": layer,
                "hypothesis": "H2",
                "language": language,
                "n_items": len(topics),
                "kernel_share_median": {key: float(np.median(val)) for key, val in shares.items()},
                "secondary_nulls": {
                    "isotropic_random_direction": {
                        "expected_kernel_share": (HIDDEN - retained_rank) / HIDDEN,
                        "monte_carlo_median": float(np.median(direction_null)),
                        "monte_carlo_95_interval": _ci(direction_null.ravel()),
                        "n_directions_per_item": n_random,
                    },
                    "distance_matched_within_condition": {
                        "kernel_share_median": float(np.median(distance_share)),
                        "norm_relative_error_median": float(np.median(distance_error)),
                    },
                },
                "components": component,
                "leave_one_topic_out_median_advantage": {
                    nuisance: {
                        str(topic): float(np.median(values[topics != topic]))
                        for topic in np.unique(topics)
                    }
                    for nuisance, values in advantages.items()
                },
                "iut_p": cell_p,
                "passed_effect": all(x["median_advantage"] >= 0.05 for x in component.values()),
            }
            layer_records.append(h2)
            h2_refs.append(h2)
            h2_cell_p.append(cell_p)
            for i, source_id in enumerate(data["source_ids"]):
                item_records.append(
                    {
                        "source_prompt_id": source_id,
                        "topic": str(topics[i]),
                        "layer": layer,
                        "language_or_direction": language,
                        "hypothesis": "H2",
                        "subject_kernel_share": float(shares["subject"][i]),
                        "country_kernel_share": float(shares["country"][i]),
                        "framing_kernel_share": float(shares["framing"][i]),
                        "country_minus_subject": float(advantages["country"][i]),
                        "framing_minus_subject": float(advantages["framing"][i]),
                        "distance_matched_null_kernel_share": float(distance_share[i]),
                        "distance_matched_null_pair_source_ids": [
                            data["source_ids"][int(distance_pairs[i, 0])],
                            data["source_ids"][int(distance_pairs[i, 1])],
                        ],
                        "distance_matched_null_norm_relative_error": float(distance_error[i]),
                    }
                )

            observed = _answer_subject(data, language)[:, layer_pos]
            full_pred = subject @ a_map
            ret_pred = (subject @ u) @ (u.T @ a_map)
            ker_pred = full_pred - ret_pred
            identity = subject
            cos = {
                "full": _cos_rows(full_pred, observed),
                "retained": _cos_rows(ret_pred, observed),
                "kernel": _cos_rows(ker_pred, observed),
                "identity": _cos_rows(identity, observed),
            }
            clauses = {
                "retained_minus_full_noninferior": (cos["retained"] - cos["full"], -0.05),
                "retained_minus_kernel": (cos["retained"] - cos["kernel"], 0.0),
                "retained_minus_identity": (cos["retained"] - cos["identity"], 0.0),
            }
            h3_components = {}
            for name, (values, null) in clauses.items():
                estimate = float(np.nanmean(values))
                finite = np.isfinite(values)
                boot = bootstrap_weighted_mean(
                    np.where(finite, values, 0.0), boot_weights * finite[None]
                )
                pvalue = bootstrap_p_greater(boot, estimate, null)
                h3_components[name] = {
                    "estimate": estimate,
                    "null": null,
                    "bootstrap_ci": _ci(boot),
                    "bootstrap_p": pvalue,
                    "passed_effect": estimate > null,
                }
                h3_p.append(pvalue)
                null_arrays[f"H3_L{layer}_{language}_{name}_bootstrap"] = boot
            item_sst = np.sum(observed**2, axis=1)
            raw_predictions = {
                "full": full_pred,
                "retained": ret_pred,
                "kernel": ker_pred,
                "identity": identity,
            }
            raw_r2 = {
                name: float(1 - np.sum((observed - pred) ** 2) / item_sst.sum())
                for name, pred in raw_predictions.items()
            }
            item_sse = np.sum((observed - ret_pred) ** 2, axis=1)
            r2 = raw_r2["retained"]
            boot_r2 = 1 - (boot_weights @ item_sse) / (boot_weights @ item_sst)
            r2_p = bootstrap_p_greater(boot_r2, r2)
            h3_components["retained_raw_r2"] = {
                "estimate": r2,
                "null": 0.0,
                "bootstrap_ci": _ci(boot_r2),
                "bootstrap_p": r2_p,
                "passed_effect": r2 > 0,
            }
            h3_p.append(r2_p)
            null_arrays[f"H3_L{layer}_{language}_retained_raw_r2_bootstrap"] = boot_r2
            calibrated_pred, gains = leave_one_topic_out_gain_prediction(ret_pred, observed, topics)
            calibrated_sse = np.sum((observed - calibrated_pred) ** 2)
            calibrated_r2 = float(1 - calibrated_sse / np.sum(observed**2))
            h3 = {
                "layer": layer,
                "hypothesis": "H3",
                "language": language,
                "mean_cosine": {key: float(np.nanmean(value)) for key, value in cos.items()},
                "raw_r2": raw_r2,
                "components": h3_components,
                "leave_one_topic_out_gain": {
                    "raw_r2": calibrated_r2,
                    "gains_by_held_out_topic": gains,
                },
            }
            layer_records.append(h3)
            h3_refs.extend(h3_components.values())
            per_item_r2_den = np.sum(observed**2, axis=1)
            for i, source_id in enumerate(data["source_ids"]):
                item_records.append(
                    {
                        "source_prompt_id": source_id,
                        "topic": str(topics[i]),
                        "layer": layer,
                        "language_or_direction": language,
                        "hypothesis": "H3",
                        **{f"cosine_{key}": float(value[i]) for key, value in cos.items()},
                        "retained_item_r2": float(
                            1 - item_sse[i] / per_item_r2_den[i]
                            if per_item_r2_den[i] > 0
                            else math.nan
                        ),
                    }
                )

            refusal_delta = _scalar_subject(data, language, "refusal_rate")
            lexical_delta = _scalar_subject(data, language, "lexical_rate")
            half_a = _draw_refusal_deltas(data, language, range(4))
            half_b = _draw_refusal_deltas(data, language, range(4, 8))
            split = float(spearmanr(half_a, half_b).statistic)
            reliability = 2 * split / (1 + split) if split > -1 else math.nan
            manipulation = float(np.mean(refusal_delta))
            manipulation_boot = bootstrap_weighted_mean(refusal_delta, boot_weights)
            manipulation_p = bootstrap_p_greater(manipulation_boot, manipulation)
            predicted_axis = full_pred @ axis
            observed_axis = observed @ axis
            rho = float(spearmanr(predicted_axis, refusal_delta).statistic)
            rho_boot = topic_bootstrap_spearman(predicted_axis, refusal_delta, boot_weights)
            observed_rho = float(spearmanr(observed_axis, refusal_delta).statistic)
            predicted_observed_rho = float(spearmanr(predicted_axis, observed_axis).statistic)
            threshold_rho = {}
            for threshold, axes in axis_store["threshold_axes"].items():
                alt_axis = axes[layer_pos].double().numpy()
                threshold_rho[threshold] = float(
                    spearmanr(full_pred @ alt_axis, refusal_delta).statistic
                )
            threshold_sign_preserved = all(
                np.sign(value) == np.sign(rho) for value in threshold_rho.values()
            )
            ranked_x = rankdata(predicted_axis)
            ranked_y = rankdata(refusal_delta)
            centered_x = ranked_x - ranked_x.mean()
            if np.isfinite(rho):
                perm_rho = np.empty(n_resample)
                for start in range(0, n_resample, 1000):
                    perm_y = ranked_y[perm_targets[start : start + 1000]]
                    perm_y = perm_y - perm_y.mean(axis=1, keepdims=True)
                    perm_rho[start : start + len(perm_y)] = (perm_y @ centered_x) / np.maximum(
                        np.linalg.norm(perm_y, axis=1) * np.linalg.norm(centered_x), 1e-300
                    )
                rho_p = float((1 + np.sum(perm_rho >= rho)) / (n_resample + 1))
            else:
                perm_rho = np.full(n_resample, np.nan)
                rho_p = 1.0
            null_arrays[f"H4_L{layer}_{language}_permutation"] = perm_rho
            null_arrays[f"H4_L{layer}_{language}_rho_bootstrap"] = rho_boot
            null_arrays[f"H4_L{layer}_{language}_manipulation_bootstrap"] = manipulation_boot
            h4 = {
                "layer": layer,
                "hypothesis": "H4",
                "language": language,
                "axis_preeligible": bool(axis_report["eligible_pre_outcomes"]),
                "axis_core_preeligible": bool(axis_report["core_eligible_pre_outcomes"]),
                "axis_threshold_stable_preoutcomes": bool(
                    axis_report["threshold_stable_pre_outcomes"]
                ),
                "refusal_delta_mean": manipulation,
                "refusal_delta_bootstrap_ci": _ci(manipulation_boot),
                "refusal_delta_bootstrap_p": manipulation_p,
                "split_half_spearman_brown": reliability,
                "predicted_axis_refusal_rho": rho,
                "predicted_axis_refusal_rho_bootstrap_ci": _ci(rho_boot),
                "observed_axis_refusal_rho": observed_rho,
                "predicted_observed_axis_rho": predicted_observed_rho,
                "threshold_axis_refusal_rho": threshold_rho,
                "threshold_sign_preserved": threshold_sign_preserved,
                "permutation_p": rho_p,
                "lexical_delta_mean": float(np.mean(lexical_delta)),
                "classifier_lexical_kappa": _refusal_kappa(data, language),
            }
            layer_records.append(h4)
            h4_refs.append(h4)
            h4_p.append(rho_p)
            for i, source_id in enumerate(data["source_ids"]):
                item_records.append(
                    {
                        "source_prompt_id": source_id,
                        "topic": str(topics[i]),
                        "layer": layer,
                        "language_or_direction": language,
                        "hypothesis": "H4",
                        "refusal_delta": float(refusal_delta[i]),
                        "lexical_delta": float(lexical_delta[i]),
                        "predicted_axis_projection": float(predicted_axis[i]),
                        "observed_axis_projection": float(observed_axis[i]),
                    }
                )

        for mass in MASS_SENSITIVITY:
            sensitivity_rank, sensitivity_tau = mass_rank(singular, mass)
            u_sens = u_all[:, :sensitivity_rank]
            sens_centered = {
                lang: contexts[lang]["neutral"][:, layer_pos] - xmu for lang in ("en", "zh")
            }
            sens_retained = {lang: sens_centered[lang] @ u_sens for lang in ("en", "zh")}
            sens_kernel = {
                lang: sens_centered[lang] - sens_retained[lang] @ u_sens.T for lang in ("en", "zh")
            }
            for direction, qlang, glang in (
                ("en_to_zh", "en", "zh"),
                ("zh_to_en", "zh", "en"),
            ):
                p_ret = retrieval_predictions(sens_retained[qlang], sens_retained[glang], topics)
                p_ker = retrieval_predictions(sens_kernel[qlang], sens_kernel[glang], topics)
                sensitivity_records.append(
                    {
                        "mass": mass,
                        "layer": layer,
                        "hypothesis": "H1",
                        "direction": direction,
                        "retained_rank": sensitivity_rank,
                        "retained_top1": float(np.mean(p_ret == np.arange(len(topics)))),
                        "kernel_top1": float(np.mean(p_ker == np.arange(len(topics)))),
                    }
                )
            for language in ("en", "zh"):
                contrasts = contexts[language]
                sens_subject = contrasts["subject"][:, layer_pos]
                sens_country = contrasts["country"][:, layer_pos]
                sens_framing = contrasts["framing"][:, layer_pos]
                sens_shares = {
                    "subject": kernel_share(sens_subject, u_sens),
                    "country": kernel_share(sens_country, u_sens),
                    "framing": kernel_share(sens_framing, u_sens),
                }
                sensitivity_records.append(
                    {
                        "mass": mass,
                        "layer": layer,
                        "hypothesis": "H2",
                        "language": language,
                        "retained_rank": sensitivity_rank,
                        "kernel_share_median": {
                            key: float(np.median(value)) for key, value in sens_shares.items()
                        },
                        "nuisance_minus_subject_median": {
                            key: float(np.median(sens_shares[key] - sens_shares["subject"]))
                            for key in ("country", "framing")
                        },
                    }
                )
                observed = _answer_subject(data, language)[:, layer_pos]
                full_pred = sens_subject @ a_map
                retained_pred = (sens_subject @ u_sens) @ (u_sens.T @ a_map)
                kernel_pred = full_pred - retained_pred
                sensitivity_records.append(
                    {
                        "mass": mass,
                        "layer": layer,
                        "hypothesis": "H3",
                        "language": language,
                        "retained_rank": sensitivity_rank,
                        "mean_cosine": {
                            "full": float(np.nanmean(_cos_rows(full_pred, observed))),
                            "retained": float(np.nanmean(_cos_rows(retained_pred, observed))),
                            "kernel": float(np.nanmean(_cos_rows(kernel_pred, observed))),
                        },
                        "raw_r2": {
                            name: float(1 - np.sum((observed - pred) ** 2) / np.sum(observed**2))
                            for name, pred in {
                                "full": full_pred,
                                "retained": retained_pred,
                                "kernel": kernel_pred,
                                "identity": sens_subject,
                            }.items()
                        },
                    }
                )

        layer_records.append(
            {
                "layer": layer,
                "hypothesis": "operator",
                "retained_rank": retained_rank,
                "kernel_rank": HIDDEN - retained_rank,
                "tau": tau,
                "singular_values_sha256": hashlib.sha256(singular.tobytes()).hexdigest(),
                "mass": MASS_PRIMARY,
                "sensitivity_ranks": {
                    str(mass): mass_rank(singular, mass)[0] for mass in MASS_SENSITIVITY
                },
            }
        )
        new_random_keys = sorted(set(random_arrays) - random_keys_start)
        new_null_keys = sorted(set(null_arrays) - null_keys_start)
        _write_npz(
            checkpoint_arrays_path,
            {
                key: (random_arrays if key in random_arrays else null_arrays)[key]
                for key in [*new_random_keys, *new_null_keys]
            },
        )
        _write_json(
            checkpoint_path,
            {
                "layer": layer,
                "regime_fp": checkpoint_regime_fp,
                "regime": checkpoint_regime,
                "arrays_sha256": _sha256(checkpoint_arrays_path),
                "random_array_keys": new_random_keys,
                "null_array_keys": new_null_keys,
                "layer_records": layer_records[layer_record_start:],
                "item_records": item_records[item_record_start:],
                "sensitivity_records": sensitivity_records[sensitivity_record_start:],
            },
        )
        print(f"[analysis] checkpointed completed layer L{layer}")

    for refs, pvals, key in (
        (h1_refs, h1_p, "holm_p"),
        (h2_refs, h2_cell_p, "holm_iut_p"),
        (h3_refs, h3_p, "holm_p"),
        (h4_refs, h4_p, "holm_p"),
    ):
        for ref, adjusted in zip(refs, holm_adjust(pvals), strict=True):
            ref[key] = adjusted

    # Two-language manipulation p-values are layer-invariant; adjust the unique pair.
    manip_by_lang = {}
    for rec in h4_refs:
        manip_by_lang.setdefault(rec["language"], rec["refusal_delta_bootstrap_p"])
    manip_adj = dict(
        zip(
            sorted(manip_by_lang),
            holm_adjust([manip_by_lang[k] for k in sorted(manip_by_lang)]),
            strict=True,
        )
    )
    for rec in h4_refs:
        rec["manipulation_holm_p"] = manip_adj[rec["language"]]

    h1_pass = {
        (r["layer"], r["direction"]): r["passed_effect"] and r["holm_p"] < 0.05 for r in h1_refs
    }
    h2_pass = {
        (r["layer"], r["language"]): r["passed_effect"] and r["holm_iut_p"] < 0.05 for r in h2_refs
    }

    def bilateral_two_layer(values: dict[tuple[int, str], bool], labels: tuple[str, str]) -> bool:
        layer_ok = {
            layer: all(values.get((layer, label), False) for label in labels) for layer in LAYERS
        }
        return layer_ok[19] and (layer_ok[14] or layer_ok[26])

    h1_joint = bilateral_two_layer(h1_pass, ("en_to_zh", "zh_to_en"))
    h2_joint = bilateral_two_layer(h2_pass, ("en", "zh"))
    if h1_joint and h2_joint:
        geometry_label = "bilingual-division-supported"
    elif h1_joint:
        geometry_label = "subjects-only"
    elif h2_joint:
        geometry_label = "allocation-only"
    elif any(h1_pass.values()) or any(h2_pass.values()):
        geometry_label = "single-language-or-layer-limited"
    else:
        geometry_label = "not-supported"

    h3_components_all = [
        comp
        for rec in [x for x in layer_records if x["hypothesis"] == "H3"]
        for comp in rec["components"].values()
    ]
    h3_valid = all(
        np.isfinite(comp["estimate"]) and np.isfinite(comp["holm_p"]) for comp in h3_components_all
    )
    h3_supported = h3_valid and all(
        comp["passed_effect"] and comp["holm_p"] < 0.05 for comp in h3_components_all
    )
    manipulation_eligible = all(
        rec["refusal_delta_mean"] >= 0.10
        and rec["manipulation_holm_p"] < 0.05
        and rec["split_half_spearman_brown"] >= 0.70
        for rec in h4_refs
    )
    lexical_consistent = all(
        np.sign(rec["refusal_delta_mean"]) == np.sign(rec["lexical_delta_mean"])
        and rec["classifier_lexical_kappa"] >= 0.40
        for rec in h4_refs
    )
    threshold_stable = all(rec["threshold_sign_preserved"] for rec in h4_refs)
    judge_claim_eligible = bool(data["judge_measurement"]["claim_eligible"])
    judge_reliability_passed = bool(data["judge_measurement"]["interjudge_reliability_passed"])
    h4_core_eligible = bool(
        axis_report["core_eligible_pre_outcomes"] and manipulation_eligible and judge_claim_eligible
    )
    h4_eligible = bool(
        h4_core_eligible and axis_report["threshold_stable_pre_outcomes"] and threshold_stable
    )
    h4_supported = h4_eligible and all(
        rec["predicted_axis_refusal_rho"] >= 0.30 and rec["holm_p"] < 0.05 for rec in h4_refs
    )
    h4_label = _classify_h4(
        judge_claim_eligible=judge_claim_eligible,
        judge_reliability_passed=judge_reliability_passed,
        core_eligible=h4_core_eligible,
        threshold_stable=bool(axis_report["threshold_stable_pre_outcomes"] and threshold_stable),
        lexical_consistent=lexical_consistent,
        supported=h4_supported,
    )
    behavior_rates = {}
    for language in ("en", "zh"):
        rates = _prompt_arrays(data, language, "refusal_rate")
        lexical = _prompt_arrays(data, language, "lexical_rate")
        behavior_rates[language] = {
            content: {
                frame: {
                    "classifier_refusal_rate": float(rates[:, content_i, frame_i].mean()),
                    "lexical_refusal_rate": float(lexical[:, content_i, frame_i].mean()),
                    "n_source_items": len(topics),
                    "n_rollouts": len(topics) * N_DRAWS,
                }
                for frame_i, frame in enumerate(("direct", "academic"))
            }
            for content_i, content in enumerate(
                ("sensitive_full", "sensitive_country_neutral", "matched_non_china")
            )
        }
    audit_sensitivity = audit_inclusion_sensitivity(data, maps, axis_store)
    report = {
        "issue": ISSUE,
        "judge_measurement": data["judge_measurement"],
        "planned_vs_realized": _planned_vs_realized_report(data),
        "coverage": _coverage_report(data),
        "primary_mass": MASS_PRIMARY,
        "n_bootstrap": n_resample,
        "n_permutations": n_resample,
        "n_random_projectors": n_random,
        "analysis_checkpoint_regime_fp": checkpoint_regime_fp,
        "records": layer_records,
        "cutoff_sensitivity": sensitivity_records,
        "audit_inclusion_sensitivity": audit_sensitivity,
        "behavior_rates": behavior_rates,
        "historical_2617_anchor": {
            "kind": "frozen refusal-axis stability; calibration anchor only, not pooled",
            "n_minimal_pairs": axis_report["n_pairs"],
            "split_half_cosine": axis_report["split_half_cosine"],
            "bootstrap_fifth_percentile_cosine": axis_report["bootstrap_fifth_percentile_cosine"],
            "threshold_axis_cosine": axis_report["threshold_axis_cosine"],
        },
        "verdict": {
            "geometry": geometry_label,
            "H1_joint": h1_joint,
            "H2_joint": h2_joint,
            "H3": (
                "transport-indeterminate"
                if not h3_valid
                else ("transport-supported" if h3_supported else "transport-null")
            ),
            "H4": h4_label,
            "H4_axis_preeligible": bool(axis_report["eligible_pre_outcomes"]),
            "H4_axis_core_preeligible": bool(axis_report["core_eligible_pre_outcomes"]),
            "H4_manipulation_eligible": manipulation_eligible,
            "H4_judge_claim_eligible": judge_claim_eligible,
            "H4_judge_reliability_passed": judge_reliability_passed,
            "H4_lexical_consistent": lexical_consistent,
            "H4_threshold_stable": threshold_stable,
        },
        "runtime": {
            "elapsed_s": time.time() - t0,
            "setup_and_svd_s": setup_elapsed,
        },
        "input_hashes": input_hashes,
        "git_sha": git_sha,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "per_item.jsonl", item_records)
    np.savez_compressed(out_dir / "random_retrieval_predictions.npz", **random_arrays)
    np.savez_compressed(out_dir / "null_distributions.npz", **null_arrays)
    report["output_hashes"] = {
        name: _sha256(out_dir / name)
        for name in (
            "per_item.jsonl",
            "random_retrieval_predictions.npz",
            "null_distributions.npz",
        )
    }
    report["runtime"]["elapsed_s"] = time.time() - t0
    report["runtime"]["battery_s"] = report["runtime"]["elapsed_s"] - setup_elapsed
    _write_json(out_dir / "analysis_report.json", report)
    print(
        f"[analysis] geometry={geometry_label} H3={report['verdict']['H3']} "
        f"H4={report['verdict']['H4']} items={data['realized']['items_primary']} "
        f"elapsed={report['runtime']['elapsed_s']:.1f}s"
    )
    return report


def run_analysis_pilot(
    run_dir: Path,
    map_root: Path,
    axis_dir: Path,
    out_dir: Path,
    analysis_lane: str = "cpu-mid",
    *,
    attempt: int = 1,
) -> dict[str, Any]:
    """Exercise the real CPU path at 10% battery width and enforce its fence."""

    pilot_dir = out_dir / f"pilot_work_{analysis_lane}"
    if pilot_dir.exists() and any(pilot_dir.rglob("*")):
        raise RuntimeError(
            f"timing pilot requires a fresh workdir; existing evidence at {pilot_dir} "
            "cannot be checkpoint-resumed"
        )
    report = run_analysis(
        run_dir,
        map_root,
        axis_dir,
        pilot_dir,
        n_random=100,
        n_resample=1_000,
        production=False,
        attempt=attempt,
    )
    projected_s = report["runtime"]["setup_and_svd_s"] + 10 * report["runtime"]["battery_s"]
    projected_upper_s = 1.25 * projected_s
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    rss_fence = (12 if analysis_lane == "cpu-mid" else 128) * 1024**3
    result = {
        "passed": True,
        "projected_timing_passed": projected_upper_s <= 2 * 3600,
        "rss_fence_passed": peak_rss_bytes <= rss_fence,
        "analysis_lane": analysis_lane,
        "n_random": 100,
        "n_resample": 1_000,
        "projected_wall_hours": projected_s / 3600,
        "projected_upper_wall_hours": projected_upper_s / 3600,
        "wall_hour_fence": 2,
        "peak_rss_bytes": peak_rss_bytes,
        "rss_fence_bytes": rss_fence,
        "requires_cpu_bigmem": analysis_lane == "cpu-mid" and peak_rss_bytes > rss_fence,
        "git_sha": report["git_sha"],
        "input_hashes": report["input_hashes"],
    }
    _write_json(out_dir / "analysis_pilot_report.json", result)
    print(
        f"[analysis-pilot] passed={result['passed']} "
        f"projected_upper_h={result['projected_upper_wall_hours']:.3f} "
        f"peak_rss_gb={peak_rss_bytes / 1024**3:.3f}"
    )
    if not result["passed"]:
        raise RuntimeError("CPU analysis pilot exceeded the registered wall/RSS fence")
    return result


def make_figures(report_path: Path, figure_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.c2a_plot_style import (
        MUTED,
        ROLES,
        better_label,
        c2a_figure,
        panel_header,
        save_c2a_figure,
        set_c2a_style,
        style_axis,
    )

    report = json.loads(report_path.read_text())
    records = report["records"]
    set_c2a_style()
    fig, include = c2a_figure("full", aspect=0.34)
    axes = fig.subplots(1, 3)
    layers = np.array(LAYERS)
    colors = {"en": ROLES["linear"].color, "zh": ROLES["other_source"].color}
    markers = {"en": "o", "zh": "^"}

    ax = axes[0]
    for language, direction in (("en", "en_to_zh"), ("zh", "zh_to_en")):
        vals = [
            next(
                r["advantage"]
                for r in records
                if r["hypothesis"] == "H1" and r["layer"] == layer and r["direction"] == direction
            )
            for layer in layers
        ]
        ax.plot(
            layers,
            vals,
            color=colors[language],
            marker=markers[language],
            lw=2.2,
            label=language.upper(),
        )
    ax.axhline(0, color=MUTED, lw=1)
    ax.axhline(0.10, color=MUTED, lw=1, ls="--")
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel(better_label("Retained top-1 advantage"))
    panel_header(ax, "A", "SUBJECT IDENTITY", "Cross-language retrieval")
    style_axis(ax)
    ax.legend()

    ax = axes[1]
    styles = {
        "subject": (ROLES["linear"].color, "o"),
        "country": (ROLES["other_source"].color, "s"),
        "framing": (ROLES["control"].color, "D"),
    }
    for contrast, (color, marker) in styles.items():
        vals = []
        for layer in layers:
            cell = [r for r in records if r["hypothesis"] == "H2" and r["layer"] == layer]
            vals.append(_mean_defined([r["kernel_share_median"][contrast] for r in cell]))
        ax.plot(layers, vals, color=color, marker=marker, lw=2.2, label=contrast.capitalize())
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel(better_label("Effective low-gain share"))
    panel_header(ax, "B", "CONTROLLED DIFFERENCES", "Where prompt changes lie")
    style_axis(ax)
    ax.legend()

    ax = axes[2]
    for language in ("en", "zh"):
        vals = [
            next(
                r["predicted_axis_refusal_rho"]
                for r in records
                if r["hypothesis"] == "H4" and r["layer"] == layer and r["language"] == language
            )
            for layer in layers
        ]
        ax.plot(
            layers,
            vals,
            color=colors[language],
            marker=markers[language],
            lw=2.2,
            label=language.upper(),
        )
    ax.axhline(0, color=MUTED, lw=1)
    ax.axhline(0.30, color=MUTED, lw=1, ls="--")
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel(better_label("Refusal-link Spearman $\\rho$"))
    panel_header(ax, "C", "CLASSIFIER-MEASURED", "Predicted refusal linkage")
    style_axis(ax)
    ax.legend()
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.76, wspace=0.34)
    stem = figure_dir / "china_refusal_definitive"
    saved = save_c2a_figure(
        fig,
        stem,
        title="Bilingual China-sensitive subject retention",
        subject="Frozen context-to-answer map decomposition",
        creator="explore-persona-space issue 952",
        include_width=include,
    )
    plt.close(fig)
    meta = {
        **saved["record"],
        "input_report": str(report_path),
        "input_sha256": _sha256(report_path),
        "verdict": report["verdict"],
        "outputs": {key: str(value) for key, value in saved.items() if key != "record"},
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    _write_json(stem.with_suffix(".meta.json"), meta)
    print(f"[figures] png={saved['png']} pdf={saved['pdf']}")
    return meta


def upload_results(
    run_dir: Path, out_dir: Path, figure_dir: Path, *, attempt: int = 1
) -> dict[str, Any]:
    """Upload compact final artifacts and verify them at the returned revision."""
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    output_prefix = f"{HF_PREFIX}/attempt{attempt}"
    judge_dir = _judge_dir(run_dir, attempt)

    required_local = (
        out_dir / "analysis_report.json",
        out_dir / "analysis_pilot_report.json",
        out_dir / "power_design.json",
        out_dir / "axis" / "refusal_axis.pt",
        out_dir / "axis" / "refusal_axis_report.json",
        out_dir / "per_item.jsonl",
        out_dir / "random_retrieval_predictions.npz",
        out_dir / "null_distributions.npz",
        figure_dir / "china_refusal_definitive.png",
        figure_dir / "china_refusal_definitive.pdf",
        figure_dir / "china_refusal_definitive.meta.json",
        run_dir / "inputs" / "upload_verified.json",
        run_dir / "manifests" / "generation.json",
        run_dir / "manifests" / "raw_upload.json",
        run_dir / "manifests" / "capture.json",
        run_dir / "manifests" / "capture_upload.json",
        judge_dir / "upload.json",
        run_dir / "reuse" / "stage_report.json",
        run_dir / "reuse" / "new_run_stage_report.json",
        run_dir / "issue952_china_definitive_done.json",
    )
    missing = [str(path) for path in required_local if not path.exists()]
    if missing:
        raise RuntimeError(f"result upload missing local artifacts: {missing}")
    if not (run_dir / "logs").exists() or not any(
        path.is_file() for path in (run_dir / "logs").rglob("*")
    ):
        raise RuntimeError("result upload requires persisted workload logs")
    analysis_report = json.loads((out_dir / "analysis_report.json").read_text())
    if (
        analysis_report.get("n_random_projectors") != N_RANDOM
        or analysis_report.get("n_bootstrap") != N_BOOT
        or analysis_report.get("n_permutations") != N_PERM
    ):
        raise RuntimeError("result upload blocked: analysis widths are not production-exact")
    config = {
        "issue": ISSUE,
        "layers": list(LAYERS),
        "primary_mass": MASS_PRIMARY,
        "sensitivity_mass": list(MASS_SENSITIVITY),
        "n_random": analysis_report["n_random_projectors"],
        "n_bootstrap": analysis_report["n_bootstrap"],
        "n_permutation": analysis_report["n_permutations"],
        "seed": SEED,
        "attempt": attempt,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    _write_json(run_dir / "config.json", config)
    manifest_files = {
        path
        for directory in (
            run_dir / "calibration",
            out_dir,
            figure_dir,
            run_dir / "logs",
            run_dir / "manifests",
        )
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file()
    }
    manifest_files.update(
        {
            run_dir / "config.json",
            run_dir / "inputs" / "upload_verified.json",
            run_dir / "reuse" / "stage_report.json",
            run_dir / "reuse" / "new_run_stage_report.json",
            run_dir / "issue952_china_definitive_done.json",
        }
    )
    manifest_files.discard(out_dir / "upload.json")
    manifest_files.discard(run_dir / "analysis_complete.json")
    manifest = {
        "files": {str(path.relative_to(run_dir)): _sha256(path) for path in sorted(manifest_files)},
        "upstream_upload_verification": {
            str(path.relative_to(run_dir)): json.loads(path.read_text())
            for path in (
                run_dir / "inputs" / "upload_verified.json",
                run_dir / "manifests" / "raw_upload.json",
                run_dir / "manifests" / "capture_upload.json",
                judge_dir / "upload.json",
            )
        },
    }
    _write_json(run_dir / "upload_manifest.json", manifest)
    info = hub.retry_transient(
        lambda: HfApi().upload_folder(
            repo_id=HF_REPO,
            repo_type="dataset",
            folder_path=str(run_dir),
            path_in_repo=output_prefix,
            allow_patterns=[
                "calibration/**",
                "eval_results/**",
                "figures/**",
                "logs/**",
                "manifests/**",
                "inputs/upload_verified.json",
                "reuse/stage_report.json",
                "reuse/new_run_stage_report.json",
                "issue952_china_definitive_done.json",
                "config.json",
                "upload_manifest.json",
            ],
            ignore_patterns=["eval_results/upload.json", "analysis_complete.json"],
            commit_message="Issue 952: definitive bilingual China analysis and figures",
        ),
        what="issue952 definitive bilingual China final artifact upload",
    )
    revision = getattr(info, "oid", None)
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("final artifact upload did not return an immutable revision")
    required_remote = {f"{output_prefix}/{path.relative_to(run_dir)}" for path in manifest_files}
    required_remote.add(f"{output_prefix}/upload_manifest.json")
    api = HfApi()
    remote_tree = {
        entry.path
        for entry in hub.retry_transient(
            lambda: list(
                api.list_repo_tree(
                    HF_REPO,
                    path_in_repo=output_prefix,
                    recursive=True,
                    repo_type="dataset",
                    revision=revision,
                )
            ),
            what="issue952 final upload tree verification",
        )
    }
    missing_remote = sorted(required_remote - remote_tree)
    if missing_remote:
        raise RuntimeError(f"revision-scoped final upload missing {len(missing_remote)} files")
    for relative, expected_sha in manifest["files"].items():
        remote = Path(
            hub.retry_transient(
                lambda relative=relative: hf_hub_download(
                    HF_REPO,
                    f"{output_prefix}/{relative}",
                    repo_type="dataset",
                    revision=revision,
                ),
                what=f"issue952 final payload verification download {relative}",
            )
        )
        if _sha256(remote) != expected_sha:
            raise RuntimeError(f"revision-scoped final payload hash mismatch: {relative}")
    remote_upload_manifest = Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                HF_REPO,
                f"{output_prefix}/upload_manifest.json",
                repo_type="dataset",
                revision=revision,
            ),
            what="issue952 final upload manifest verification download",
        )
    )
    if _sha256(remote_upload_manifest) != _sha256(run_dir / "upload_manifest.json"):
        raise RuntimeError("revision-scoped final upload manifest hash mismatch")
    marker = {
        "data_revision": revision,
        "attempt": attempt,
        "data_commit_url": str(info),
        "verified_files": len(required_remote),
        "upload_manifest_sha256": _sha256(run_dir / "upload_manifest.json"),
        "analysis_report_sha256": _sha256(out_dir / "analysis_report.json"),
        "figure_png_sha256": _sha256(figure_dir / "china_refusal_definitive.png"),
        "figure_url": (
            f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/"
            f"{output_prefix}/{figure_dir.relative_to(run_dir)}/china_refusal_definitive.png"
        ),
    }
    marker_path = out_dir / "upload.json"
    _write_json(marker_path, marker)
    api = HfApi()
    marker_info = hub.retry_transient(
        lambda: api.upload_file(
            repo_id=HF_REPO,
            repo_type="dataset",
            path_or_fileobj=str(marker_path),
            path_in_repo=f"{output_prefix}/eval_results/upload.json",
            commit_message="Issue 952: verify definitive bilingual China result upload",
        ),
        what="issue952 final verification marker upload",
    )
    marker_revision = getattr(marker_info, "oid", None)
    if not isinstance(marker_revision, str) or not marker_revision:
        raise RuntimeError("final marker upload did not return an immutable revision")
    remote_marker = Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                HF_REPO,
                f"{output_prefix}/eval_results/upload.json",
                repo_type="dataset",
                revision=marker_revision,
            ),
            what="issue952 final verification marker download",
        )
    )
    if _sha256(remote_marker) != _sha256(marker_path):
        raise RuntimeError("revision-scoped final verification marker hash mismatch")
    sentinel = {
        "schema_version": 1,
        "kind": "issue952_china_definitive_analysis",
        "version": 1,
        "issue": ISSUE,
        "status": "analysis_complete",
        "note": "Analysis artifacts and exact-revision uploads verified",
        "data_revision": revision,
        "attempt": attempt,
        "verification_marker_revision": marker_revision,
        "verification_marker_sha256": _sha256(marker_path),
        "figure_url": marker["figure_url"],
        "timestamp_unix": time.time(),
    }
    sentinel_path = run_dir / "analysis_complete.json"
    sentinel_path.unlink(missing_ok=True)
    pending_sentinel = run_dir / "analysis_complete.pending.json"
    pending_sentinel.unlink(missing_ok=True)
    _write_json(pending_sentinel, sentinel)
    sentinel_info = hub.retry_transient(
        lambda: api.upload_file(
            repo_id=HF_REPO,
            repo_type="dataset",
            path_or_fileobj=str(pending_sentinel),
            path_in_repo=f"{output_prefix}/analysis_complete.json",
            commit_message="Issue 952: definitive bilingual China terminal sentinel",
        ),
        what="issue952 definitive analysis terminal sentinel upload",
    )
    sentinel_revision = getattr(sentinel_info, "oid", None)
    if not isinstance(sentinel_revision, str) or not sentinel_revision:
        raise RuntimeError("analysis sentinel upload did not return an immutable revision")
    remote_sentinel = Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                HF_REPO,
                f"{output_prefix}/analysis_complete.json",
                repo_type="dataset",
                revision=sentinel_revision,
            ),
            what="issue952 definitive analysis terminal sentinel download",
        )
    )
    if _sha256(remote_sentinel) != _sha256(pending_sentinel):
        raise RuntimeError("revision-scoped final terminal sentinel hash mismatch")
    os.replace(pending_sentinel, sentinel_path)
    print(f"[results-upload] verified={len(required_remote)} sentinel_revision={sentinel_revision}")
    return {
        **marker,
        "marker_revision": marker_revision,
        "sentinel_revision": sentinel_revision,
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        required=True,
        choices=("stage", "power", "axis", "analysis-pilot", "analysis", "figures", "upload"),
    )
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--hist-root", type=Path)
    ap.add_argument("--map-root", type=Path)
    ap.add_argument("--axis-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--figure-dir", type=Path)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--n-resample", type=int, default=N_BOOT)
    ap.add_argument("--analysis-lane", choices=("cpu-mid", "cpu-bigmem"), default="cpu-mid")
    ap.add_argument("--attempt", type=int, default=1)
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.attempt < 1:
        raise SystemExit("--attempt must be >= 1")
    if args.phase == "stage":
        if args.run_dir is None:
            raise SystemExit("--run-dir required for stage")
        stage_new_run(args.run_dir, attempt=args.attempt)
        stage_reuse(args.run_dir)
    elif args.phase == "power":
        design_power(args.out_dir)
    elif args.phase == "axis":
        if args.hist_root is None:
            raise SystemExit("--hist-root required for axis")
        build_refusal_axis(args.hist_root, args.out_dir)
    elif args.phase in ("analysis-pilot", "analysis"):
        if args.run_dir is None or args.map_root is None or args.axis_dir is None:
            raise SystemExit("--run-dir, --map-root, and --axis-dir required")
        if args.phase == "analysis-pilot":
            if args.n_random != N_RANDOM or args.n_resample != N_BOOT:
                raise SystemExit("analysis-pilot does not accept width overrides")
            run_analysis_pilot(
                args.run_dir,
                args.map_root,
                args.axis_dir,
                args.out_dir,
                args.analysis_lane,
                attempt=args.attempt,
            )
        else:
            run_analysis(
                args.run_dir,
                args.map_root,
                args.axis_dir,
                args.out_dir,
                args.n_random,
                args.n_resample,
                production=True,
                attempt=args.attempt,
            )
    elif args.phase == "figures":
        if args.figure_dir is None:
            raise SystemExit("--figure-dir required for figures")
        make_figures(args.out_dir / "analysis_report.json", args.figure_dir)
    else:
        if args.run_dir is None or args.figure_dir is None:
            raise SystemExit("--run-dir and --figure-dir required for upload")
        upload_results(args.run_dir, args.out_dir, args.figure_dir, attempt=args.attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
