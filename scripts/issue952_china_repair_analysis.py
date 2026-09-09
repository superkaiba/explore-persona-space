"""Manifest-bound CPU analysis of the repaired four-cell China experiment.

No generation, fitting, judging, or figure production occurs here. CLI phases are
stage, load, pilot, full, and export. Production inputs always preserve all 85
subjects, two languages, four content cells, two frames, and eight aligned draws.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import copy
import json
import os
import re
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from explore_persona_space.orchestrate import hub
from scripts import issue952_china_definitive_analysis as legacy
from scripts import issue952_china_definitive_gpu as gpu
from scripts import issue952_china_repair_geometry as geom
from scripts import issue952_china_repair_judges as judges
from scripts import issue952_china_repair_persist as persist
from scripts import issue952_china_repair_continuation as continuation

CONTRACT = "issue952-china-repair-analysis-v2"
HF_PREFIX = "issue952_position_divergence/followups/china_refusal_wording_withholding_v2"
INPUT_REV = "1e417487cb69111848881e469c35d3fa6ae928ea"
LAYERS = (14, 19, 26)
MASSES = (0.99, 0.90, 0.999)
N_SUBJECTS, HIDDEN, N_DRAWS = 85, 3584, 8
SEED = 952_2695
MEASURES = ("withholding_score", "strict_complete_refusal")
sha_file = legacy._sha256
sha_object = legacy._sha_obj


def read_json(path: Path) -> Any:
    """Read strict JSON using the collector's duplicate-key rejecting parser."""

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant {value} in {path}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=judges._unique_object,
        parse_constant=invalid_constant,
    )


def write_json(path: Path, value: Any) -> None:
    """Atomically write finite JSON, representing undefined quantities with null."""

    def finite(obj):
        if isinstance(obj, np.ndarray):
            return finite(obj.tolist())
        if isinstance(obj, np.generic):
            return finite(obj.item())
        if isinstance(obj, dict):
            return {str(k): finite(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [finite(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    payload = json.dumps(
        finite(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_hash(path: Path, expected: str) -> None:
    """Fail on a missing, malformed, or mismatched immutable input hash."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected or "") or sha_file(path) != expected:
        raise ValueError(f"artifact hash mismatch: {path}")


def checked_relative(path: str) -> Path:
    """Reject traversal and absolute paths in remote artifact inventories."""
    value = Path(path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError("artifact inventory contains an unsafe relative path")
    return value


def _revision(value: str) -> str:
    """Require an immutable Hub commit, never a branch or floating revision."""
    if re.fullmatch(r"[0-9a-f]{40}", value or "") is None:
        raise ValueError("an explicit immutable 40-character revision is required")
    return value


def input_identity(data_revision: str, judge_revision: str, judge_prefix: str) -> dict:
    """Freeze the production destination and every externally selected revision."""
    prefix = checked_relative(judge_prefix).as_posix()
    if not prefix.startswith(f"{HF_PREFIX}/attempt1/") or "/smoke/" in prefix:
        raise ValueError("production judge archive must be under repaired-v2/attempt1")
    return {
        "contract": CONTRACT,
        "data_revision": _revision(data_revision),
        "judge_revision": _revision(judge_revision),
        "judge_prefix": prefix,
        "input_revision": INPUT_REV,
        "map_revision": legacy.MAP_REV,
        "historical_revision": legacy.HIST_REV,
        "model_revision": legacy.MODEL_REV,
        "data_prefix": f"{HF_PREFIX}/attempt1",
        "attempt": 1,
    }


def _stage_file(root: Path, revision: str, remote: str, local: str, expected=None) -> dict:
    """Use the canonical atomic Hub helper and record the bytes actually staged."""
    target = root / checked_relative(local)
    hub.stage_hub_file(
        repo_id=legacy.HF_REPO,
        path_in_repo=checked_relative(remote).as_posix(),
        target=target,
        repo_type="dataset",
        revision=_revision(revision),
    )
    if expected is not None:
        require_hash(target, expected)
    return {"revision": revision, "remote": remote, "sha256": sha_file(target)}


def stage(root: Path, identity: dict) -> dict:
    """Stage small manifests first, then their enumerated tensors and judge archive."""
    marker = root / "stage.json"
    if marker.exists():
        prior = read_json(marker)
        if prior["identity"] != identity:
            raise ValueError("staging root is already bound to different revisions")
        if prior["files_sha256"] != sha_object(prior["files"]):
            raise ValueError("staging file census hash changed")
        for name, record in prior["files"].items():
            require_hash(root / checked_relative(name), record["sha256"])
        return prior
    # Bind a partial stage too: stage_hub_file deliberately reuses existing bytes.
    intent = root / "stage_intent.json"
    if intent.exists() and read_json(intent) != identity:
        raise ValueError("partial staging root has a different immutable identity")
    write_json(intent, identity)
    files = {}

    def fetch(revision, remote, local, expected=None):
        files[local] = _stage_file(root, revision, remote, local, expected)

    prefix, revision = identity["data_prefix"], identity["data_revision"]
    for name in (
        "generation.json",
        "capture.json",
        "raw_upload.json",
        "capture_upload.json",
        "input_stage.json",
        "cap_extension.json",
        "generation.initial.json",
    ):
        fetch(revision, f"{prefix}/manifests/{name}", f"run/manifests/{name}")
    fetch(revision, f"{prefix}/issue952_china_definitive_done.json", "run/done.json")
    for name in ("prompt_bank.jsonl", "bank_audit_report.json"):
        fetch(INPUT_REV, f"{HF_PREFIX}/inputs/{name}", f"run/inputs/{name}")
    generation = read_json(root / "run/manifests/generation.json")
    capture = read_json(root / "run/manifests/capture.json")
    raw_upload = read_json(root / "run/manifests/raw_upload.json")
    for remote, digest in raw_upload["byte_verified_sha256"].items():
        relative = Path(remote).relative_to(f"{prefix}/raw_completions").as_posix()
        fetch(revision, remote, "run/raw_completions/" + relative, digest)
    require_hash(root / "run/raw_completions/rollouts.jsonl", generation["rollouts_sha256"])
    fetch(
        revision,
        f"{prefix}/analysis_tensors/vc.pt",
        "run/analysis_tensors/vc.pt",
        capture["vc_sha256"],
    )
    for name, digest in capture["va_files"].items():
        if re.fullmatch(r"va_\d{5}_\d{5}\.pt", name) is None:
            raise ValueError("unexpected answer tensor shard name")
        fetch(revision, f"{prefix}/analysis_tensors/{name}", f"run/analysis_tensors/{name}", digest)
    archive_prefix, judge_rev = identity["judge_prefix"], identity["judge_revision"]
    fetch(judge_rev, f"{archive_prefix}/packed_manifest.json", "judge_archive/packed_manifest.json")
    archive = read_json(root / "judge_archive/packed_manifest.json")
    if archive["format"] != "exact-utf8-text-archive-v1":
        raise ValueError("judge archive must preserve exact original bytes")
    for name, info in archive["shards"].items():
        fetch(judge_rev, f"{archive_prefix}/{name}", f"judge_archive/{name}", info["sha256"])
    persist.unpack_tree(root / "judge_archive", root / "judge")
    for name, info in archive["files"].items():
        target = "judge/" + checked_relative(name).as_posix()
        require_hash(root / target, info["sha256"])
        files[target] = {"revision": judge_rev, "remote": archive_prefix, "sha256": info["sha256"]}
    for layer in LAYERS:
        fetch(
            legacy.MAP_REV,
            f"{legacy.MAP_PREFIX}/L{layer}/ridge.pt",
            f"reuse/maps/L{layer}/ridge.pt",
        )
    for remote, local in legacy.HIST_FILES.items():
        fetch(legacy.HIST_REV, remote, f"reuse/axis/{local}")
    report = {"identity": identity, "files": files, "files_sha256": sha_object(files)}
    write_json(marker, report)
    print(f"[stage] verified {len(files)} files at explicit immutable revisions", flush=True)
    return report


def validate_judgments(directory: Path, generation_path: Path, rollout_rows: list[dict]) -> tuple:
    """Revalidate original judge packets, receipts and decisions, then joined scores."""
    manifest_path, summary_path = directory / "manifest.json", directory / "summary.json"
    manifest, summary = read_json(manifest_path), read_json(summary_path)
    original_root = Path(manifest["lookup_path"]).parent.parent

    def relocated(value):
        return directory / checked_relative(Path(value).relative_to(original_root).as_posix())

    bound = copy.deepcopy(manifest)
    for key in ("lookup_path", "source_manifest_path"):
        bound[key] = str(relocated(bound[key]))
    lookup = judges._validate_manifest(bound)
    if manifest["phase"] != "production" or summary.get("technical_complete") is not True:
        raise ValueError("fresh production requires a technically complete production judge wave")
    for path, digest in (
        (manifest_path, summary["input_manifest_sha256"]),
        (generation_path, summary["source_manifest_sha256"]),
        (directory / "scores.jsonl", summary["scores_sha256"]),
        (directory / "overlap.jsonl", summary["overlap_sha256"]),
    ):
        require_hash(path, digest)
    if any(
        summary[key] != manifest[key]
        for key in (
            "contract",
            "phase",
            "rubric_sha256",
            "rollouts_sha256",
            "bank_sha256",
            "source_manifest_sha256",
            "source_ids_sha256",
            "n_items",
            "n_overlap",
            "n_assignments",
        )
    ):
        raise ValueError("collector summary disagrees with its preparation manifest")
    entries = {
        (lane, info["opaque_id"]): entry
        for entry in lookup
        for lane, info in entry["lanes"].items()
    }
    rollouts = {row["item_id"]: row for row in rollout_rows}
    if set(rollouts) != {row["item_id"] for row in lookup}:
        raise ValueError("judge lookup differs from final rollout item IDs")
    census = {row["packet_path"]: row for row in summary["artifact_census"]}
    if len(census) != len(manifest["packets"]) or set(census) != {
        r["packet_path"] for r in manifest["packets"]
    }:
        raise ValueError("original judge artifact census is incomplete or duplicated")
    decisions = {}
    for record in manifest["packets"]:
        item_census = census[record["packet_path"]]
        for kind in ("packet", "receipt", "output"):
            if record[f"{kind}_path"] != item_census[f"{kind}_path"]:
                raise ValueError("judge artifact path changed between preparation and collection")
            require_hash(relocated(record[f"{kind}_path"]), item_census[f"{kind}_sha256"])
        if item_census["packet_sha256"] != record["packet_sha256"]:
            raise ValueError("original packet hash changed")
        packet = read_json(relocated(record["packet_path"]))
        judges._validate_packet(packet, record, manifest, entries)
        receipt = read_json(relocated(record["receipt_path"]))
        judges._validate_receipt(receipt, packet, record)
        output = judges.read_jsonl(relocated(record["output_path"]))
        if [row["opaque_id"] for row in output] != record["opaque_ids"]:
            raise ValueError("original decision rows differ from packet ordering")
        for decision, item in zip(output, packet["items"], strict=True):
            entry = entries[(record["lane"], item["opaque_id"])]
            rollout = rollouts[entry["item_id"]]
            if item["question"] != rollout["question"] or item["response"] != rollout["text"]:
                raise ValueError("judge packet does not contain the final generated response")
            judges.validate_decision(decision, item, packet, record, item_census["receipt_sha256"])
            key = (record["lane"], item["opaque_id"])
            if key in decisions:
                raise ValueError("duplicated original judge decision")
            decisions[key] = decision
    expected_scores, expected_overlap = [], []
    for entry in lookup:
        metadata = {
            k: v for k, v in entry.items() if k not in {"lanes", "primary_agent", "assigned_agents"}
        }
        primary = entry["primary_agent"]
        expected_scores.append(
            {
                **metadata,
                **decisions[(primary, entry["lanes"][primary]["opaque_id"])],
                "judge_id": primary,
            }
        )
        if len(entry["lanes"]) == 2:
            expected_overlap.append(
                {
                    **metadata,
                    **{
                        lane: decisions[(lane, entry["lanes"][lane]["opaque_id"])]
                        for lane in judges.AGENTS
                    },
                }
            )
    scores, overlap = (
        judges.read_jsonl(directory / "scores.jsonl"),
        judges.read_jsonl(directory / "overlap.jsonl"),
    )
    if judges._bytes_json(scores) != judges._bytes_json(expected_scores) or judges._bytes_json(
        overlap
    ) != judges._bytes_json(expected_overlap):
        raise ValueError("joined judgments differ from original primary/overlap decisions")
    agreement = {
        "overall": judges.agreement(overlap),
        "by_language": {
            lang: judges.agreement([r for r in overlap if r["language"] == lang])
            for lang in geom.LANGUAGE_ORDER
        },
    }
    if judges._bytes_json(agreement) != judges._bytes_json(summary["agreement"]):
        raise ValueError("reported agreement differs from original overlap decisions")
    return scores, agreement, summary


def validate_mixed_judgments(
    directory: Path, generation_path: Path, rollout_rows: list[dict]
) -> tuple:
    """Validate the versioned original-plus-Luna phase without changing legacy validation."""
    directory = Path(directory)
    mixed_manifest = read_json(directory / "mixed_manifest.json")
    if mixed_manifest.get("phase") != "mixed-original-plus-luna":
        raise ValueError("mixed validator requires the explicit mixed-phase manifest")
    continuation.validate_continuation(directory)
    reconstructed = continuation.reconstruct_mixed_scores(directory)
    for name, expected in (("mixed_scores.jsonl", reconstructed["scores_sha256"]),
                           ("mixed_overlap.jsonl", reconstructed["overlap_sha256"])):
        require_hash(directory / name, expected)
    scores = judges.read_jsonl(directory / "mixed_scores.jsonl")
    overlap = judges.read_jsonl(directory / "mixed_overlap.jsonl")
    rollout_ids = {row["item_id"] for row in rollout_rows}
    if len({row["item_id"] for row in scores}) != len(scores) or any(
        row["item_id"] not in rollout_ids for row in scores
    ):
        raise ValueError("mixed scores do not join uniquely to final rollout item IDs")
    if len(scores) != len(rollout_ids):
        raise ValueError("mixed scores do not cover every final rollout item")
    if len(overlap) != mixed_manifest["n_overlap"]:
        raise ValueError("mixed overlap census differs from immutable mixed manifest")
    summary = {
        "contract": CONTRACT,
        "phase": mixed_manifest["phase"],
        "technical_complete": True,
        "n_items": len(scores),
        "n_assignments": mixed_manifest["n_assignments"],
        "n_overlap": len(overlap),
        "rubric_sha256": judges.RUBRIC_SHA256,
        "input_manifest_sha256": mixed_manifest["continuation_manifest_sha256"],
        "original_manifest_sha256": mixed_manifest["original_manifest_sha256"],
    }
    return scores, {"overall": judges.agreement(overlap)}, summary


def valid_mean(values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Average finite draws only; all-missing groups remain NaN with count zero."""
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    result = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(np.where(valid, values, 0.0).sum(axis=axis), count, out=result, where=count > 0)
    return result, count


def assemble_panel(
    bank: list[dict], rollouts: list[dict], scores: list[dict], vc: np.ndarray, va: np.ndarray
) -> dict:
    """Join exact IDs into factorial arrays; this pure helper also supports toy fixtures."""
    sources = sorted({row["source_prompt_id"] for row in bank})
    source_index = {sid: i for i, sid in enumerate(sources)}
    shape = (len(sources), 2, 4, 2)
    contexts = np.empty((*shape, *vc.shape[1:]), dtype=np.float32)
    answers = np.empty((*shape, N_DRAWS, *va.shape[1:]), dtype=np.float32)
    behavior = {name: np.full((*shape, N_DRAWS), np.nan) for name in MEASURES}
    prompts, cells, topics = {}, set(), {}
    if len(bank) != np.prod(shape) or vc.shape[0] != len(bank):
        raise ValueError("bank/context factorial dimensions differ")
    for index, row in enumerate(bank):
        cell = (
            source_index[row["source_prompt_id"]],
            geom.LANGUAGE_ORDER.index(row["language"]),
            geom.CONTENT_ORDER.index(row["content"]),
            geom.FRAME_ORDER.index(row["frame"]),
        )
        if row["item_id"] in prompts or cell in cells:
            raise ValueError("duplicate prompt ID or factorial cell")
        sid = row["source_prompt_id"]
        if sid in topics and topics[sid] != row["topic"]:
            raise ValueError("source has inconsistent topic membership")
        topics[sid] = row["topic"]
        prompts[row["item_id"]] = (cell, row)
        cells.add(cell)
        contexts[cell] = vc[index]
    expected_ids = {f"{pid}-d{draw}" for pid in prompts for draw in range(N_DRAWS)}
    if (
        len(rollouts) != len(expected_ids)
        or len(va) != len(expected_ids)
        or {r["item_id"] for r in rollouts} != expected_ids
    ):
        raise ValueError("answer vectors/rollouts do not cover every exact prompt and draw")
    if len(scores) != len(expected_ids) or {r["item_id"] for r in scores} != expected_ids:
        raise ValueError("behavior score item IDs differ from the full geometric panel")
    score_by_id = {r["item_id"]: r for r in scores}
    for index, rollout in enumerate(rollouts):
        cell, prompt = prompts[rollout["prompt_id"]]
        draw = rollout["draw"]
        if (
            type(draw) is not int
            or not 0 <= draw < N_DRAWS
            or rollout["item_id"] != f"{rollout['prompt_id']}-d{draw}"
        ):
            raise ValueError("rollout ID/draw mismatch")
        for key in ("source_prompt_id", "language", "content", "frame", "topic"):
            if rollout[key] != prompt[key] or score_by_id[rollout["item_id"]][key] != prompt[key]:
                raise ValueError("source/arm metadata drift between bank, rollout and score")
        if rollout["question"] != prompt["prompt"]:
            raise ValueError("rollout question changed from the frozen bank")
        score = score_by_id[rollout["item_id"]]
        # The collector deliberately omits prompt_id. The exact item-ID join above
        # supplies its prompt through the validated final rollout, never ID parsing.
        if score["draw"] != draw:
            raise ValueError("judge draw metadata changed")
        answers[(*cell, draw)] = va[index]
        for name in MEASURES:
            value = score[name]
            if score["unassessable"]:
                if value is not None:
                    raise ValueError("unassessable behavior must remain missing")
            else:
                if (
                    name == "withholding_score"
                    and (type(value) is not int or not 0 <= value <= 100)
                ) or (name == "strict_complete_refusal" and type(value) is not bool):
                    raise ValueError("assessable score has invalid measurement type/range")
                behavior[name][(*cell, draw)] = value
    if not np.isfinite(contexts).all() or not np.isfinite(answers).all():
        raise ValueError("nonfinite activation data is a technical failure")
    return {
        "source_ids": np.array(sources),
        "topics": np.array([topics[s] for s in sources]),
        "contexts": contexts,
        "answers": answers,
        **behavior,
    }


def _tensor(store: dict, key: str, shape: tuple) -> np.ndarray:
    """Validate the tensor's actual shape, dtype, layer order, and finiteness."""
    value = store[key]
    if (
        store.get("layers") != list(LAYERS)
        or not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"invalid realized {key} activation tensor")
    return value.numpy()


def validate_final_rollouts(bank: list[dict], rollouts: list[dict], generation: dict) -> list[str]:
    """Validate actual draw IDs, seeds, output tokens and recorded termination states."""
    expected_ids = gpu._expected_rollout_ids(bank)
    if (
        [r["item_id"] for r in rollouts] != expected_ids
        or generation["ordered_item_ids_sha256"] != sha_object(expected_ids)
        or generation["n_rows"] != len(expected_ids)
        or generation["n_prompts"] != len(bank)
    ):
        raise ValueError("realized rollout order and full source/cell/draw census differ")
    for index, row in enumerate(rollouts):
        tokens = row["completion_token_ids"]
        cap = 4096 if "cap_extension" in row else 2048
        if (
            type(row["seed"]) is not int
            or row["seed"] != generation["regime"]["seed_base"] + index
            or type(row["context_tokens"]) is not int
            or row["context_tokens"] <= 0
            or not isinstance(tokens, list)
            or not 0 < len(tokens) <= cap
            or any(type(token) is not int or token < 0 for token in tokens)
            or row["completion_tokens"] != len(tokens)
            or row["finish_reason"] not in ("stop", "length")
            or row["cap_hit"] is not (row["finish_reason"] == "length")
            or not isinstance(row["text"], str)
            or not row["text"]
        ):
            raise ValueError("final rollout has invalid seed, exact-token, or termination evidence")
    return expected_ids


def load_inputs(root: Path, identity: dict) -> dict:
    """Consume and validate all fresh production cells before persisting analysis inputs."""
    staged = stage(root, identity)
    prior = root / "loaded/manifest.json"
    if prior.exists():
        report = read_json(prior)
        if report["identity"] != identity or report["stage_sha256"] != sha_file(
            root / "stage.json"
        ):
            raise ValueError("loaded inputs are bound to a different stage")
        for name, key in (
            ("panel.npz", "panel_sha256"),
            ("axis/refusal_axis.pt", "axis_sha256"),
            ("axis/refusal_axis_report.json", "axis_report_sha256"),
        ):
            require_hash(root / "loaded" / name, report[key])
        return report
    run = root / "run"
    generation_path = run / "manifests/generation.json"
    generation, capture = read_json(generation_path), read_json(run / "manifests/capture.json")
    regime = generation["regime"]
    expected = {
        "study": "repaired-v2",
        "hf_prefix": HF_PREFIX,
        "model": legacy.MODEL,
        "model_revision": legacy.MODEL_REV,
        "layers": list(LAYERS),
        "draws": N_DRAWS,
        "smoke": False,
        "attempt": 1,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_new_tokens": 2048,
        "seed_base": 9_520_000,
        "seed_namespace": "repaired-production-v2",
        "prompts_per_source": 16,
        "registered_source_items": N_SUBJECTS,
    }
    if any(regime.get(key) != value for key, value in expected.items()) or generation[
        "regime_fp"
    ] != sha_object(regime):
        raise ValueError("generation regime is not the authorized repaired production recipe")
    require_hash(run / "inputs/prompt_bank.jsonl", regime["bank_sha256"])
    require_hash(run / "inputs/bank_audit_report.json", regime["bank_audit_report_sha256"])
    require_hash(run / "manifests/cap_extension.json", generation["cap_extension_manifest_sha256"])
    bank, audit = (
        judges.read_jsonl(run / "inputs/prompt_bank.jsonl"),
        read_json(run / "inputs/bank_audit_report.json"),
    )
    input_stage = read_json(run / "manifests/input_stage.json")
    if any(
        input_stage.get(key) != value
        for key, value in {
            "data_revision": INPUT_REV,
            "prompt_bank_sha256": regime["bank_sha256"],
            "bank_audit_report_sha256": regime["bank_audit_report_sha256"],
            "study": "repaired-v2",
            "hf_prefix": HF_PREFIX,
        }.items()
    ):
        raise ValueError("GPU input stage differs from pinned repaired inputs")
    gpu._validate_repaired_bank(bank, audit)
    accepted = gpu._accepted_bank_identity(bank, audit)
    if (
        generation["accepted_bank"] != accepted
        or capture["accepted_bank"] != accepted
        or accepted["accepted_source_ids_sha256"] != judges.FROZEN_SOURCE_IDS_SHA256
    ):
        raise ValueError("accepted source identity changed from the frozen 85-source panel")
    rollouts = judges.read_jsonl(run / "raw_completions/rollouts.jsonl")
    expected_ids = validate_final_rollouts(bank, rollouts, generation)
    gpu._validate_raw_upload(run, generation, read_json(run / "manifests/raw_upload.json"))
    gpu._validate_capture_artifacts(run, generation, capture)
    gpu._validate_capture_upload(run, capture, read_json(run / "manifests/capture_upload.json"))
    done = read_json(run / "done.json")
    if (
        done.get("status") != "done"
        or done.get("study") != "repaired-v2"
        or any(
            done[key] != read_json(run / f"manifests/{key}.json")
            for key in ("generation", "capture", "raw_upload", "capture_upload")
        )
    ):
        raise ValueError("terminal GPU sentinel does not bind the consumed current artifacts")
    cap_regime = capture["capture_regime"]
    packages = generation["package_versions"]
    if (
        packages != regime["package_versions"]
        or capture["package_versions"] != packages
        or any(
            packages.get(key, "").split("+")[0] != value
            for key, value in legacy.REGISTERED_PACKAGE_VERSIONS.items()
        )
    ):
        raise ValueError(
            "saved generation/capture runtime differs from registered package versions"
        )
    if (
        capture["capture_regime_fp"] != sha_object(cap_regime)
        or cap_regime["generation_fingerprint"] != gpu._generation_fingerprint(generation)
        or any(
            cap_regime.get(key) != value
            for key, value in {
                "model_revision": legacy.MODEL_REV,
                "layers": list(LAYERS),
                "context_position": legacy.CONTEXT_POSITION,
                "answer_pooling": legacy.ANSWER_POOLING,
                "serialized_dtype": "fp32",
                "bank_sha256": regime["bank_sha256"],
                "rollouts_sha256": generation["rollouts_sha256"],
                "study": "repaired-v2",
            }.items()
        )
    ):
        raise ValueError("capture regime changed from the final exact-token production lineage")
    vc_store = torch.load(run / "analysis_tensors/vc.pt", map_location="cpu", weights_only=False)
    if (
        vc_store["item_ids"] != [r["item_id"] for r in bank]
        or vc_store["capture_regime_fp"] != capture["capture_regime_fp"]
        or vc_store["bank_sha256"] != regime["bank_sha256"]
        or vc_store["capture_regime"] != cap_regime
        or vc_store["model_revision"] != legacy.MODEL_REV
        or vc_store["position"] != legacy.CONTEXT_POSITION
    ):
        raise ValueError("context tensor identity differs from final bank/capture")
    vc = _tensor(vc_store, "vc", (len(bank), len(LAYERS), HIDDEN))
    va = np.empty((len(rollouts), len(LAYERS), HIDDEN), dtype=np.float32)
    cursor = 0
    if {p.name for p in (run / "analysis_tensors").glob("va_*.pt")} != set(capture["va_files"]):
        raise ValueError("unmanifested or absent answer tensor shards")
    for name in sorted(capture["va_files"]):
        store = torch.load(run / "analysis_tensors" / name, map_location="cpu", weights_only=False)
        rows = store["index"]
        if (
            store["capture_regime_fp"] != capture["capture_regime_fp"]
            or store["capture_regime"] != cap_regime
            or store["model_revision"] != legacy.MODEL_REV
            or store["pooling"] != legacy.ANSWER_POOLING
            or store["rollouts_sha256"] != generation["rollouts_sha256"]
            or store["empty_rows"]
            or [r["item_id"] for r in rows] != expected_ids[cursor : cursor + len(rows)]
        ):
            raise ValueError("answer tensor final-generation row lineage is incomplete")
        for index, row in enumerate(rows):
            rollout = rollouts[cursor + index]
            if (
                any(row[key] != rollout[key] for key in ("item_id", "prompt_id", "draw"))
                or row["ctx_len"] != rollout["context_tokens"]
                or row["completion_len"] != len(rollout["completion_token_ids"])
                or rollout["completion_tokens"] != row["completion_len"]
                or row["span_start"] != row["ctx_len"]
                or row["span_end"] != row["ctx_len"] + row["completion_len"]
                or not row["span_end"] <= row["tail_end"] <= row["span_end"] + 2
            ):
                raise ValueError("captured answer boundaries differ from saved final token IDs")
        va[cursor : cursor + len(rows)] = _tensor(
            store, "va_tail_incl", (len(rows), len(LAYERS), HIDDEN)
        )
        cursor += len(rows)
    if cursor != len(rollouts):
        raise ValueError("answer tensors do not cover every final rollout")
    scores, agreement, summary = validate_judgments(root / "judge", generation_path, rollouts)
    panel = assemble_panel(bank, rollouts, scores, vc, va)
    if len(panel["source_ids"]) != N_SUBJECTS or len(np.unique(panel["topics"])) != 12:
        raise ValueError("realized analysis panel must preserve all 85 subjects and 12 topics")
    out = root / "loaded"
    out.mkdir(parents=True, exist_ok=True)
    legacy._write_npz(out / "panel.npz", panel)
    # This is the frozen historical procedure, including stability diagnostics.
    legacy.build_refusal_axis(root / "reuse/axis", out / "axis")
    report = {
        "identity": identity,
        "stage_sha256": sha_file(root / "stage.json"),
        "input_files_sha256": staged["files_sha256"],
        "panel_sha256": sha_file(out / "panel.npz"),
        "axis_sha256": sha_file(out / "axis/refusal_axis.pt"),
        "axis_report_sha256": sha_file(out / "axis/refusal_axis_report.json"),
        "n_sources": N_SUBJECTS,
        "n_prompts": len(bank),
        "n_rollouts": len(rollouts),
        "agreement": agreement,
        "judge_summary_sha256": sha_file(root / "judge/summary.json"),
        "rubric_sha256": summary["rubric_sha256"],
        "source_bank_sha256": audit["source_bank_sha256"],
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "axis_label": "frozen historical strict-refusal axis; diagnostic, not a validated withholding axis",
    }
    write_json(out / "manifest.json", report)
    print(
        f"[load] verified {len(bank)} prompts, {len(rollouts)} responses, all 85 subjects",
        flush=True,
    )
    return report


def weighted_draws(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Bootstrap means with explicit finite-row denominators for every draw."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    denominator = weights @ valid.astype(float)
    result = np.full(denominator.shape, np.nan)
    np.divide(weights @ np.where(valid, values, 0), denominator, out=result, where=denominator > 0)
    return result


def summarize(values: np.ndarray, weights: np.ndarray) -> dict:
    """Summarize defined item quantities without replacing undefined items by zero."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    draws = weighted_draws(values, weights)
    return {
        "mean": float(values[valid].mean()) if valid.any() else None,
        "median": float(np.median(values[valid])) if valid.any() else None,
        "ci95": legacy._ci(draws),
        "n_total": len(values),
        "n_defined": int(valid.sum()),
        "n_undefined": int((~valid).sum()),
        "bootstrap_n_defined": int(np.isfinite(draws).sum()),
    }


def _permutation_p(observed: float, null: np.ndarray, two_sided=False) -> float | None:
    """Finite-sample permutation probability; undefined statistics stay undefined."""
    valid = np.isfinite(null)
    if not np.isfinite(observed) or not valid.any():
        return None
    comparison = np.abs(null[valid]) >= abs(observed) if two_sided else null[valid] >= observed
    return float((1 + comparison.sum()) / (1 + valid.sum()))


def topic_signs(topics: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """Use one paired sign for every item in the same topic in each null draw."""
    _, inverse = np.unique(topics, return_inverse=True)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], (n_draws, inverse.max() + 1))
    return signs[:, inverse]


def paired_difference(
    first: np.ndarray, second: np.ndarray, weights: np.ndarray, signs: np.ndarray, family: str
) -> dict:
    """Paired item difference with topic bootstrap and topic-block sign permutation."""
    delta = np.asarray(first, float) - np.asarray(second, float)
    result = summarize(delta, weights)
    valid = np.isfinite(delta)
    observed = float(delta[valid].mean()) if valid.any() else np.nan
    null = (
        signs[:, valid] @ delta[valid] / valid.sum() if valid.any() else np.full(len(signs), np.nan)
    )
    return {
        **result,
        "family": family,
        "raw_p": _permutation_p(observed, null),
        "alternative": "first_greater_than_second",
        "permutation": "paired_topic_block_sign_flip",
        "permutation_n_defined": int(np.isfinite(null).sum()),
    }


def correlation(
    x: np.ndarray,
    y: np.ndarray,
    topics: np.ndarray,
    weights: np.ndarray,
    n_perm: int,
    seed: int,
    family: str | None = None,
) -> dict:
    """Spearman association with paired cluster bootstrap and within-topic permutations."""
    valid = np.isfinite(x) & np.isfinite(y)
    a, b, w = np.asarray(x)[valid], np.asarray(y)[valid], weights[:, valid]
    result = {
        "rho": None,
        "ci95": [None, None],
        "n_total": len(x),
        "n_defined": int(valid.sum()),
        "n_undefined": int((~valid).sum()),
        "n_topics": len(np.unique(topics[valid])),
        "bootstrap_n_defined": 0,
        "permutation_n_defined": 0,
        "raw_p": None,
        "family": family,
        "alternative": "two_sided",
        "permutation": "within_topic_on_paired_finite_items",
    }
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        result["undefined_reason"] = "fewer_than_two_pairs_or_constant_input"
        return result
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.linalg.norm(ra) * np.linalg.norm(rb)
    rho = float(ra @ rb / denominator)
    usable = w.sum(axis=1) > 0
    boot = legacy.topic_bootstrap_spearman(a, b, w[usable])
    targets = legacy._within_topic_targets(topics[valid], n_perm, seed)
    null = (rb[targets] @ ra) / denominator
    result.update(
        rho=rho,
        ci95=legacy._ci(boot),
        bootstrap_n_defined=int(np.isfinite(boot).sum()),
        permutation_n_defined=n_perm,
        raw_p=_permutation_p(rho, null, two_sided=True),
    )
    return result


def retrieval(query: np.ndarray, gallery: np.ndarray, topics: np.ndarray | None) -> tuple:
    """Reuse exact retrieval, marking galleries containing undefined directions missing."""
    n = len(query)
    groups = np.zeros(n, dtype=int) if topics is None else topics
    valid = np.linalg.norm(query, axis=-1) > 0
    for topic in np.unique(groups):
        indices = np.flatnonzero(groups == topic)
        valid[indices] &= np.all(np.linalg.norm(gallery[indices], axis=-1) > 0)
    predicted, ranks = legacy.retrieval_predictions_and_ranks(query, gallery, groups)
    ranks = ranks.astype(float)
    predicted[~valid], ranks[~valid] = -1, np.nan
    hit = (predicted == np.arange(n)).astype(float)
    hit[~valid] = np.nan
    return predicted, ranks, hit


def retrieval_panel(
    neutral: np.ndarray,
    xmu: np.ndarray,
    basis: np.ndarray,
    topics: np.ndarray,
    weights: np.ndarray,
    n_random: int,
    n_perm: int,
    seed: int,
    family: str,
) -> tuple[dict, dict]:
    """H1 uses frozen xmu centering, fixed topic galleries and rank-matched projectors."""
    centered = neutral - xmu
    coords = centered @ basis
    kernel = centered - coords @ basis.T
    n, rank = len(topics), basis.shape[1]
    combined = np.concatenate((centered[:, 0], centered[:, 1]))
    random = {
        direction: {
            kind: np.empty((n_random, n), dtype=np.int64 if kind.endswith("pred") else float)
            for kind in ("topic_pred", "topic_rank", "all_pred", "all_rank")
        }
        for direction in ("en_to_zh", "zh_to_en")
    }
    if rank == 0:
        for values in random.values():
            for key, value in values.items():
                value.fill(-1 if key.endswith("pred") else np.nan)
    else:
        seeds = np.arange(n_random) + seed
        for start in range(0, n_random, legacy.RANDOM_BLOCK):
            block = legacy.structured_projection_coords_batch(
                combined, rank, seeds[start : start + legacy.RANDOM_BLOCK]
            )
            en, zh = np.split(block, 2, axis=1)
            for direction, q, g in (("en_to_zh", en, zh), ("zh_to_en", zh, en)):
                for name, groups in (("topic", topics), ("all", np.zeros(n, dtype=int))):
                    pred, ranks = legacy.batch_retrieval_predictions_and_ranks(q, g, groups)
                    ranks = ranks.astype(float)
                    valid = np.linalg.norm(q, axis=-1) > 0
                    for topic in np.unique(groups):
                        indices = np.flatnonzero(groups == topic)
                        valid[:, indices] &= np.all(
                            np.linalg.norm(g[:, indices], axis=-1) > 0, axis=1
                        )[:, None]
                    pred[~valid], ranks[~valid] = -1, np.nan
                    random[direction][f"{name}_pred"][start : start + len(block)] = pred
                    random[direction][f"{name}_rank"][start : start + len(block)] = ranks
    targets = legacy._within_topic_targets(topics, n_perm, seed + 1)
    records, arrays = {}, {}
    for direction, q, g in (("en_to_zh", 0, 1), ("zh_to_en", 1, 0)):
        panels, topic_hits, topic_preds = {}, {}, {}
        for gallery_name, groups in (("within_topic", topics), ("all_items", None)):
            modes = {}
            for name, values in (("full", centered), ("retained", coords), ("kernel", kernel)):
                pred, ranks, hit = retrieval(values[:, q], values[:, g], groups)
                modes[name] = {
                    "accuracy": summarize(hit, weights),
                    "mrr": summarize(1 / ranks, weights),
                }
                arrays[f"{direction}_{gallery_name}_{name}_hit"] = hit
                arrays[f"{direction}_{gallery_name}_{name}_rank"] = ranks
                if groups is not None:
                    topic_hits[name], topic_preds[name] = hit, pred
            key = "topic" if groups is not None else "all"
            random_pred = random[direction][f"{key}_pred"]
            random_rank = random[direction][f"{key}_rank"]
            random_hit = (random_pred == np.arange(n)[None]).astype(float)
            random_hit[random_pred < 0] = np.nan
            mean_random_hit, n_defined_projectors = valid_mean(random_hit, axis=0)
            modes["rank_matched_random"] = {
                "accuracy": summarize(mean_random_hit, weights),
                "n_projectors": n_random,
                "n_defined_projectors_by_item": n_defined_projectors,
                "rank": rank,
                "accuracy_projector_interval95": legacy._ci(valid_mean(random_hit, axis=1)[0]),
                "mrr": summarize(valid_mean(1 / random_rank, axis=0)[0], weights),
            }
            sizes = (
                np.array([np.sum(topics == topic) for topic in topics])
                if groups is not None
                else np.full(n, n)
            )
            panels[gallery_name] = {
                "modes": modes,
                "gallery_size_by_item": sizes,
                "mean_fixed_gallery_chance": float(np.mean(1 / sizes)),
                "n_gallery_items": n,
            }
            arrays[f"{direction}_{gallery_name}_random_pred"] = random_pred
            arrays[f"{direction}_{gallery_name}_random_rank"] = random_rank
            if groups is not None:
                topic_hits["random"] = mean_random_hit
        comparisons = {}
        ret = topic_preds["retained"]
        for control in ("kernel", "random"):
            delta = topic_hits["retained"] - topic_hits[control]
            valid = np.isfinite(delta)
            effect = float(delta[valid].mean()) if valid.any() else np.nan
            if control == "kernel":
                control_null = (topic_preds["kernel"][None] == targets).astype(float)
            else:
                counts = np.zeros((n, n), dtype=np.int32)
                predictions = random[direction]["topic_pred"]
                rows = np.broadcast_to(np.arange(n), predictions.shape)
                defined = predictions >= 0
                np.add.at(counts, (rows[defined], predictions[defined]), 1)
                # Use the same item-specific projector denominator as the
                # observed valid_mean; an undefined projector is never a miss.
                denominator = defined.sum(axis=0)[None]
                control_null = np.full(targets.shape, np.nan, dtype=float)
                np.divide(
                    counts[np.arange(n)[None], targets],
                    denominator,
                    out=control_null,
                    where=denominator > 0,
                )
            null = (
                ((ret[None] == targets).astype(float) - control_null)[:, valid].mean(axis=1)
                if valid.any()
                else np.full(n_perm, np.nan)
            )
            comparisons[f"retained_minus_{control}"] = {
                **summarize(delta, weights),
                "raw_p": _permutation_p(effect, null),
                "family": family,
                "alternative": "retained_greater_than_control",
                "permutation": "same_within_topic_target_permutation_for_both_readouts",
                "permutation_n_defined": int(np.isfinite(null).sum()),
            }
        records[direction] = {
            "centering": "frozen_map_xmu",
            "galleries": panels,
            "comparisons": comparisons,
        }
    return records, arrays


def prediction_quality(
    predicted: np.ndarray,
    observed: np.ndarray,
    topics: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    family: str,
) -> dict:
    """H3 uncalibrated predictions, with zero-change R² and matched topic inference."""
    cosine = geom.cosine_rows(predicted, observed)
    observed_norm2 = np.sum(observed**2, axis=1)
    error2 = np.sum((predicted - observed) ** 2, axis=1)
    denominator = weights @ observed_norm2
    boot_r2 = np.full(len(weights), np.nan)
    np.divide(weights @ error2, denominator, out=boot_r2, where=denominator > 0)
    boot_r2 = 1 - boot_r2
    null_cosine = np.empty(len(targets))
    null_r2 = np.empty(len(targets))
    # Pairwise inner products avoid n_perm x n_subject x hidden materialization.
    dot = predicted @ observed.T
    pred_norm2 = np.sum(predicted**2, axis=1)
    pair_denominator = np.sqrt(pred_norm2[:, None] * observed_norm2[None])
    pair_cosine = np.full(dot.shape, np.nan)
    np.divide(dot, pair_denominator, out=pair_cosine, where=pair_denominator > 0)
    perm_dot = dot[np.arange(len(observed))[None], targets]
    null_cosine[:] = valid_mean(pair_cosine[np.arange(len(observed))[None], targets], axis=1)[0]
    sum_observed = observed_norm2.sum()
    null_r2[:] = (
        1 - (pred_norm2.sum() + sum_observed - 2 * perm_dot.sum(axis=1)) / sum_observed
        if sum_observed > 0
        else np.nan
    )
    r2 = geom.delta_r2(predicted, observed)
    cosine_summary = summarize(cosine, weights)
    cosine_effect = cosine_summary["mean"]
    return {
        "cosine": {
            **cosine_summary,
            "family": family,
            "raw_p": _permutation_p(
                cosine_effect if cosine_effect is not None else np.nan, null_cosine
            ),
            "alternative": "greater_than_within_topic_pairing_null",
        },
        "r2": {
            **r2,
            "ci95": legacy._ci(boot_r2),
            "bootstrap_n_defined": int(np.isfinite(boot_r2).sum()),
            "family": family,
            "raw_p": _permutation_p(r2["r2"] if r2["r2"] is not None else np.nan, null_r2),
            "alternative": "greater_than_within_topic_pairing_null",
        },
        "calibration": "none; frozen operator and identity only",
        "n_topics": len(np.unique(topics)),
    }


def analyze_layer(
    panel: dict,
    bundle: dict,
    modes: dict,
    axis: np.ndarray,
    layer_index: int,
    mass: float,
    n_random: int,
    n_resample: int,
) -> tuple[dict, dict]:
    """Compute one complete layer/cutoff panel without outcome-dependent geometry selection."""
    topics, layer = panel["topics"], LAYERS[layer_index]
    weights = legacy.topic_bootstrap_weights(topics, n_resample, SEED + 1)
    signs = topic_signs(topics, n_resample, SEED + 2)
    targets = legacy._within_topic_targets(topics, n_resample, SEED + 3)
    rank = geom.mass_rank(modes["singular"], mass)
    basis, write_basis = modes["read"][:, :rank], modes["write"][:, :rank]
    contexts = geom.factorial_contrasts(panel["contexts"][..., layer_index, :])
    mean_answer = panel["answers"][..., layer_index, :].mean(axis=4, dtype=np.float64)
    answer_contrasts = geom.factorial_contrasts(mean_answer)
    contrast_names = [name for name in contexts if name != "neutral"]
    context_stack = np.stack([contexts[name] for name in contrast_names])
    answer_stack = np.stack([answer_contrasts[name] for name in contrast_names])
    # Exactly one orthogonality validation per basis, with every contrast batched.
    read_parts = geom.orthogonal_parts(context_stack, basis)
    write_parts = geom.orthogonal_parts(answer_stack, write_basis)
    position = {name: i for i, name in enumerate(contrast_names)}
    suffix = "primary" if mass == 0.99 else f"sensitivity_{mass}"
    h1, arrays = retrieval_panel(
        contexts["neutral"],
        np.asarray(bundle["xmu"], dtype=float),
        basis,
        topics,
        weights,
        n_random,
        n_resample,
        SEED + layer * 10_000,
        f"H1.{suffix}.retained_vs_kernel_and_random",
    )
    result = {
        "layer": layer,
        "squared_singular_mass": mass,
        "role": suffix,
        "rank_retained": rank,
        "rank_low": len(modes["singular"]) - rank,
        "realized_squared_mass": float(
            np.sum(modes["singular"][:rank] ** 2) / np.sum(modes["singular"] ** 2)
        )
        if np.sum(modes["singular"] ** 2) > 0
        else None,
        "n_geometry_subjects": len(topics),
        "H1": h1,
        "H2": {},
        "H3": {},
        "observed_answer_write_decomposition_secondary": {},
    }
    arrays.update(
        {
            "context_norm": read_parts["norm"],
            "context_low_share": read_parts["low_share"],
            "answer_norm": write_parts["norm"],
            "answer_low_write_share": write_parts["low_share"],
            "contrast_names": np.array(contrast_names),
        }
    )
    halves = [
        geom.factorial_contrasts(
            panel["answers"][..., half, layer_index, :].mean(axis=4, dtype=np.float64)
        )["subject"]
        for half in (slice(0, 4), slice(4, 8))
    ]
    for lang_index, lang in enumerate(geom.LANGUAGE_ORDER):
        shares = read_parts["low_share"][:, :, lang_index]
        norms = read_parts["norm"][:, :, lang_index]
        contrasts = {}
        answer_parts = {}
        for i, name in enumerate(contrast_names):
            contrasts[name] = {
                "low_gain_share": summarize(shares[i], weights),
                "raw_norm": summarize(norms[i], weights),
                "n_exact_zero_vectors": int(np.sum(norms[i] == 0)),
            }
            answer_parts[name] = {
                "low_write_share": summarize(write_parts["low_share"][i, :, lang_index], weights),
                "raw_norm": summarize(write_parts["norm"][i, :, lang_index], weights),
                "n_exact_zero_vectors": int(np.sum(write_parts["norm"][i, :, lang_index] == 0)),
            }
        comparisons = {}
        for name in ("framing", "china_cue", "control_cue"):
            family = f"H2.{suffix}." + (
                "framing_enrichment" if name == "framing" else "additional_country_cue_enrichment"
            )
            comparisons[f"{name}_minus_subject"] = paired_difference(
                shares[position[name]], shares[position["subject"]], weights, signs, family
            )
        result["H2"][lang] = {
            "contrasts": contrasts,
            "paired_comparisons": comparisons,
            "cue_interpretation": "additional explicit country wording conditional on the named subject; country identity is not removed",
        }
        result["observed_answer_write_decomposition_secondary"][lang] = {
            "basis": "RIGHT singular vectors (answer write directions)",
            "contrasts": answer_parts,
        }
        subject = contexts["subject"][:, lang_index]
        observed = answer_contrasts["subject"][:, lang_index]
        predictions = {
            "full": subject @ modes["operator"],
            "retained": read_parts["retained"][position["subject"], :, lang_index]
            @ modes["operator"],
            "kernel": read_parts["low"][position["subject"], :, lang_index] @ modes["operator"],
            "identity": subject,
        }
        result["H3"][lang] = {
            "modes": {
                name: prediction_quality(
                    prediction,
                    observed,
                    topics,
                    weights,
                    targets,
                    f"H3.{suffix}.subject_answer_prediction",
                )
                for name, prediction in predictions.items()
            },
            "draw_half_reliability": {
                "alignment": "identical draws 0:4 and 4:8 in each subject, language, content, and frame",
                "subject_answer_cosine": summarize(
                    geom.cosine_rows(halves[0][:, lang_index], halves[1][:, lang_index]), weights
                ),
            },
        }
        for name, prediction in predictions.items():
            arrays[f"{lang}_subject_prediction_{name}"] = prediction
        arrays[f"{lang}_subject_observed"] = observed
        if mass == 0.99:
            behavior_mean, count = valid_mean(panel["withholding_score"], axis=4)
            behavior_delta = geom.factorial_contrasts(behavior_mean)["subject"][:, lang_index]
            predicted_axis = predictions["full"] @ axis
            observed_axis = observed @ axis
            result.setdefault("H4", {})[lang] = {
                "axis_label": "frozen historical strict-refusal axis; diagnostic, not validated for withholding",
                "withholding_change": summarize(behavior_delta, weights),
                "predicted_subject_displacement": correlation(
                    predicted_axis,
                    behavior_delta,
                    topics,
                    weights,
                    n_resample,
                    SEED + 4,
                    "H4.primary.historical_axis_withholding_association",
                ),
                "observed_subject_displacement": correlation(
                    observed_axis,
                    behavior_delta,
                    topics,
                    weights,
                    n_resample,
                    SEED + 4,
                    "H4.primary.historical_axis_withholding_association",
                ),
                "prompt_valid_draw_min": int(count[:, lang_index].min()),
            }
            arrays[f"{lang}_predicted_axis_subject_delta"] = predicted_axis
            arrays[f"{lang}_observed_axis_subject_delta"] = observed_axis
            arrays[f"{lang}_withholding_subject_delta"] = behavior_delta
            half_behavior = [
                geom.factorial_contrasts(
                    valid_mean(panel["withholding_score"][..., half], axis=4)[0]
                )["subject"][:, lang_index]
                for half in (slice(0, 4), slice(4, 8))
            ]
            result["H4"][lang]["draw_half_reliability"] = {
                "withholding_delta": correlation(
                    half_behavior[0], half_behavior[1], topics, weights, n_resample, SEED + 5
                ),
                "observed_axis_delta": correlation(
                    halves[0][:, lang_index] @ axis,
                    halves[1][:, lang_index] @ axis,
                    topics,
                    weights,
                    n_resample,
                    SEED + 5,
                ),
                "aligned_half_associations": [
                    correlation(
                        halves[i][:, lang_index] @ axis,
                        half_behavior[i],
                        topics,
                        weights,
                        n_resample,
                        SEED + 5,
                    )
                    for i in range(2)
                ],
            }
    return result, arrays


def behavior_report(panel: dict, agreement: dict, n_resample: int) -> dict:
    """Report graded withholding and strict refusal separately, with valid-draw counts."""
    topics = panel["topics"]
    weights = legacy.topic_bootstrap_weights(topics, n_resample, SEED + 1)
    result = {
        "agreement": agreement,
        "human_validated": False,
        "low_reliability_stops_run": False,
        "measures": {},
    }
    for measure in MEASURES:
        values = panel[measure]
        means, counts = valid_mean(values, axis=4)
        deltas = geom.factorial_contrasts(means)
        arms = []
        for li, lang in enumerate(geom.LANGUAGE_ORDER):
            for ci, content in enumerate(geom.CONTENT_ORDER):
                for fi, frame in enumerate(geom.FRAME_ORDER):
                    cell = values[:, li, ci, fi]
                    valid = np.isfinite(cell)
                    cell_means = means[:, li, ci, fi]
                    valid_values = cell[valid]
                    arms.append(
                        {
                            "language": lang,
                            "content": content,
                            "frame": frame,
                            "subject_weighted": summarize(cell_means, weights),
                            "valid_draw_weighted_mean": float(valid_values.mean())
                            if len(valid_values)
                            else None,
                            "n_planned_draws": cell.size,
                            "n_valid_draws": int(valid.sum()),
                            "n_unassessable_draws": int((~valid).sum()),
                            "n_subjects_with_no_valid_draw": int(
                                np.sum(counts[:, li, ci, fi] == 0)
                            ),
                            "valid_draw_counts_by_subject": counts[:, li, ci, fi],
                            "n_positive_draws": int(np.sum(valid_values > 0)),
                        }
                    )
        halves = [
            geom.factorial_contrasts(valid_mean(values[..., half], axis=4)[0])["subject"]
            for half in (slice(0, 4), slice(4, 8))
        ]
        result["measures"][measure] = {
            "arms": arms,
            "subject_change": {
                lang: summarize(deltas["subject"][:, li], weights)
                for li, lang in enumerate(geom.LANGUAGE_ORDER)
            },
            "draw_half_reliability": {
                lang: correlation(
                    halves[0][:, li], halves[1][:, li], topics, weights, n_resample, SEED + 5
                )
                for li, lang in enumerate(geom.LANGUAGE_ORDER)
            },
        }
    return result


def adjust_families(report: dict) -> dict:
    """Holm-adjust explicitly named families, retaining undefined planned tests."""
    families = defaultdict(list)

    def visit(value):
        if isinstance(value, dict):
            if value.get("family") is not None and "raw_p" in value:
                families[value["family"]].append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(report)
    census = {}
    for name, tests in families.items():
        # Missing tests remain in family size but never receive a numeric verdict.
        pvalues = [
            r["raw_p"] if r["raw_p"] is not None and np.isfinite(r["raw_p"]) else 1.0 for r in tests
        ]
        adjusted = legacy.holm_adjust(pvalues)
        for test, value in zip(tests, adjusted, strict=True):
            test["holm_p"] = (
                value if test["raw_p"] is not None and np.isfinite(test["raw_p"]) else None
            )
        census[name] = {
            "n_planned_tests": len(tests),
            "n_defined_tests": sum(t["holm_p"] is not None for t in tests),
            "method": "Holm; undefined planned tests retained in family size",
        }
    return census


def code_identity() -> dict:
    """Bind checkpoints to the exact consumer/helper bytes and numeric runtime."""
    import scipy

    from explore_persona_space.orchestrate import hub as hub_module

    modules = (
        Path(__file__),
        Path(legacy.__file__),
        Path(geom.__file__),
        Path(gpu.__file__),
        Path(judges.__file__),
        Path(persist.__file__),
        Path(hub_module.__file__),
    )
    return {
        "files": {p.name: sha_file(p) for p in modules},
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def checked_loaded(root: Path, identity: dict) -> tuple[dict, dict]:
    """Check every input hash again before opening derivative analysis inputs."""
    staged = stage(root, identity)
    manifest = read_json(root / "loaded/manifest.json")
    if (
        manifest["identity"] != identity
        or manifest["stage_sha256"] != sha_file(root / "stage.json")
        or manifest["input_files_sha256"] != staged["files_sha256"]
    ):
        raise ValueError("loaded manifest does not bind current staged revisions")
    for name, key in (
        ("panel.npz", "panel_sha256"),
        ("axis/refusal_axis.pt", "axis_sha256"),
        ("axis/refusal_axis_report.json", "axis_report_sha256"),
    ):
        require_hash(root / "loaded" / name, manifest[key])
    with np.load(root / "loaded/panel.npz", allow_pickle=False) as archive:
        panel = {key: archive[key] for key in archive.files}
    if (
        panel["contexts"].shape != (N_SUBJECTS, 2, 4, 2, len(LAYERS), HIDDEN)
        or panel["answers"].shape != (N_SUBJECTS, 2, 4, 2, N_DRAWS, len(LAYERS), HIDDEN)
        or any(panel[key].shape != (N_SUBJECTS, 2, 4, 2, N_DRAWS) for key in MEASURES)
    ):
        raise ValueError("loaded derivative lost production geometry or behavior cells")
    return manifest, panel


def frozen_modes(root: Path, layer: int) -> tuple[dict, dict]:
    """Factor each raw frozen operator once, with hash-bound read/write SVD reuse."""
    path = root / f"reuse/maps/L{layer}/ridge.pt"
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not {"W", "xmu", "xsd", "ymu", "layer", "kind", "fitter"} <= set(bundle)
        or int(bundle["layer"]) != layer
        or tuple(bundle["W"].shape) != (HIDDEN, HIDDEN)
    ):
        raise ValueError("frozen ridge map schema changed")
    converted = {}
    for key in ("W", "xmu", "xsd", "ymu"):
        value = bundle[key].detach().cpu().double().numpy()
        expected = (HIDDEN, HIDDEN) if key == "W" else (HIDDEN,)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError("frozen map tensor shape/finiteness changed")
        converted[key] = value
    marker, arrays_path = root / f"loaded/modes/L{layer}.json", root / f"loaded/modes/L{layer}.npz"
    regime = {
        "map_sha256": sha_file(path),
        "geometry_sha256": sha_file(Path(geom.__file__)),
        "numpy": np.__version__,
        "coordinate_system": "raw; row vectors; read U, write V",
    }
    if marker.exists():
        prior = read_json(marker)
        if prior["regime"] != regime:
            raise ValueError("SVD checkpoint is stale")
        require_hash(arrays_path, prior["arrays_sha256"])
        with np.load(arrays_path, allow_pickle=False) as archive:
            modes = {key: archive[key] for key in archive.files}
    else:
        modes = geom.operator_modes(converted["W"], converted["xsd"])
        legacy._write_npz(arrays_path, modes)
        write_json(marker, {"regime": regime, "arrays_sha256": sha_file(arrays_path)})
    return converted, modes


def run_analysis(root: Path, identity: dict, *, full: bool) -> dict:
    """Run all scientific cells at 10% or full resampling width, checkpointing each."""
    started = time.monotonic()
    loaded, panel = checked_loaded(root, identity)
    common_regime = {
        "identity": identity,
        "loaded_manifest_sha256": sha_file(root / "loaded/manifest.json"),
        "code": code_identity(),
        "masses": MASSES,
        "layers": LAYERS,
        "seed": SEED,
        "n_geometry_subjects": N_SUBJECTS,
    }
    common_fp = sha_object(common_regime)
    n_random, n_resample = (1000, 10000) if full else (100, 1000)
    role = "full" if full else "pilot"
    out = root / "analysis" / role
    regime = {**common_regime, "n_random": n_random, "n_resample": n_resample, "role": role}
    regime_fp = sha_object(regime)
    if full:
        pilot = read_json(root / "analysis/pilot/done.json")
        require_hash(root / "analysis/pilot/report.json", pilot["report_sha256"])
        if pilot["common_regime_fp"] != common_fp or pilot["technical_complete"] is not True:
            raise ValueError("full CPU analysis requires the current successful 10% pilot")
    if (out / "done.json").exists():
        done = read_json(out / "done.json")
        if done["regime_fp"] != regime_fp:
            raise ValueError("completed analysis identity changed; use a fresh root")
        for name, digest in done["files_sha256"].items():
            require_hash(out / checked_relative(name), digest)
        return read_json(out / "report.json")
    axis_store = torch.load(
        root / "loaded/axis/refusal_axis.pt", map_location="cpu", weights_only=False
    )
    if axis_store["layers"] != list(LAYERS) or tuple(axis_store["axis"].shape) != (3, HIDDEN):
        raise ValueError("historical axis dimensions changed")
    axis = axis_store["axis"].double().numpy()
    if not np.isfinite(axis).all() or not np.allclose(np.linalg.norm(axis, axis=1), 1):
        raise ValueError("historical diagnostic axes must be finite unit vectors")
    report = {
        "contract": CONTRACT,
        "role": role,
        "technical_complete": True,
        "regime": regime,
        "regime_fp": regime_fp,
        "coverage": {
            "planned_sources": N_SUBJECTS,
            "realized_sources": len(panel["source_ids"]),
            "topics": len(np.unique(panel["topics"])),
            "languages": geom.LANGUAGE_ORDER,
            "contents": geom.CONTENT_ORDER,
            "frames": geom.FRAME_ORDER,
            "draws_per_prompt": N_DRAWS,
            "geometry_selected_on_behavior": False,
            "source_ids": panel["source_ids"],
            "topic_by_source": panel["topics"],
        },
        "behavior": behavior_report(panel, loaded["agreement"], n_resample),
        "panels": [],
        "axis_diagnostic": read_json(root / "loaded/axis/refusal_axis_report.json"),
        "limitations": [
            "The identity-plus-learned-bias family has the same identity prediction for deltas: the frozen additive bias cancels.",
            "Additional explicit country wording conditional on named subjects; country identity is not removed.",
            "Historical strict-refusal axis is a diagnostic reference, not a validated withholding axis.",
            "No human labels: judge agreement is reliability evidence, not human-ground-truth validity.",
            "Undefined behavior/correlation/direction quantities stay missing, with defined denominators.",
            "Post-v1 design revision; effects and reliability are reported regardless of numerical thresholds.",
            "The pilot preserves every production cell and changes only resampling/projector width; it does not certify full-width runtime.",
        ],
    }
    write_json(out / "behavior.json", report["behavior"])
    setup_seconds = 0.0
    for layer_index, layer in enumerate(LAYERS):
        setup_start = time.monotonic()
        bundle, modes = frozen_modes(root, layer)
        setup_seconds += time.monotonic() - setup_start
        for mass in MASSES:
            unit = f"L{layer}_mass{mass}"
            unit_path, arrays_path = (
                out / "checkpoints" / f"{unit}.json",
                out / "checkpoints" / f"{unit}.npz",
            )
            unit_start = time.monotonic()
            if unit_path.exists():
                checkpoint = read_json(unit_path)
                if checkpoint["regime_fp"] != regime_fp:
                    raise ValueError(f"analysis checkpoint regime changed: {unit}")
                require_hash(arrays_path, checkpoint["arrays_sha256"])
                record = checkpoint["result"]
            else:
                record, arrays = analyze_layer(
                    panel, bundle, modes, axis[layer_index], layer_index, mass, n_random, n_resample
                )
                legacy._write_npz(arrays_path, arrays)
                write_json(
                    unit_path,
                    {
                        "regime_fp": regime_fp,
                        "arrays_sha256": sha_file(arrays_path),
                        "result": record,
                    },
                )
            report["panels"].append(record)
            print(
                f"[{role}] {unit} complete, elapsed={time.monotonic() - unit_start:.1f}s",
                flush=True,
            )
    report["multiple_test_families"] = adjust_families(report)
    report["elapsed_seconds"] = time.monotonic() - started
    report["svd_setup_seconds"] = setup_seconds
    report["estimated_full_seconds"] = (
        setup_seconds + 10 * (report["elapsed_seconds"] - setup_seconds)
        if not full
        else report["elapsed_seconds"]
    )
    write_json(out / "report.json", report)
    files = {
        p.relative_to(out).as_posix(): sha_file(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "done.json"
    }
    write_json(
        out / "done.json",
        {
            "technical_complete": True,
            "role": role,
            "regime_fp": regime_fp,
            "common_regime_fp": common_fp,
            "report_sha256": sha_file(out / "report.json"),
            "files_sha256": files,
            "elapsed_seconds": report["elapsed_seconds"],
        },
    )
    return report


def export(root: Path, identity: dict) -> dict:
    """Upload derivatives and the out-root census, then verify exact revision bytes."""
    checked_loaded(root, identity)
    done = read_json(root / "analysis/full/done.json")
    if done["technical_complete"] is not True:
        raise ValueError("cannot export unfinished full analysis")
    for name, digest in done["files_sha256"].items():
        require_hash(root / "analysis/full" / checked_relative(name), digest)
    prefix = f"{HF_PREFIX}/attempt1/analysis"
    census = {}
    for directory in ("loaded", "analysis"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                census[path.relative_to(root).as_posix()] = sha_file(path)
    for name in ("stage.json", "stage_intent.json"):
        census[name] = sha_file(root / name)
    receipt_path = root / "export.json"
    if receipt_path.exists():
        previous = read_json(receipt_path)
        if previous["files_sha256"] != census or previous["identity"] != identity:
            raise ValueError("refusing to overwrite a published analysis with changed bytes")
        revision = previous["revision"]
    else:
        from huggingface_hub import HfApi

        info = hub.retry_transient(
            lambda: HfApi().upload_folder(
                repo_id=legacy.HF_REPO,
                repo_type="dataset",
                folder_path=str(root),
                path_in_repo=prefix,
                allow_patterns=list(census),
                commit_message="Issue 952 repaired China analysis: full H1-H4, exact inputs and inference",
            ),
            what="issue952 repaired analysis upload",
        )
        revision = _revision(info.oid)
    for name, digest in census.items():
        _stage_file(
            root, revision, f"{prefix}/{name}", f"export_verification/{revision}/{name}", digest
        )
    result = {
        "identity": identity,
        "prefix": prefix,
        "revision": revision,
        "files_sha256": census,
        "verified": True,
        "report_url": f"https://huggingface.co/datasets/{legacy.HF_REPO}/resolve/{revision}/{prefix}/analysis/full/report.json",
    }
    write_json(receipt_path, result)
    return result


def require_cpu_lane(enabled: bool) -> None:
    """Keep real SVD/resampling off the shared VM and off an allocated GPU lane."""
    if (
        not enabled
        or socket.gethostname() == "cia-benchmark-vm"
        or Path("/mnt/eps-data").is_mount()
        or torch.cuda.is_available()
    ):
        raise RuntimeError("real load/pilot/full require --cpu-lane on the dedicated CPU pod")


def build_argparser() -> argparse.ArgumentParser:
    """Expose separate immutable-input, validation, pilot, full, and publication phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "load", "pilot", "full", "export"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-revision", required=True)
    parser.add_argument("--judge-revision", required=True)
    parser.add_argument("--judge-prefix", required=True)
    parser.add_argument("--cpu-lane", action="store_true")
    parser.add_argument("--threads", type=int, default=8, choices=range(8, 17))
    return parser


def _execute_phase(args: argparse.Namespace, identity: dict) -> None:
    """Execute one phase and print only aggregate, text-free evidence."""
    if args.phase == "stage":
        result = stage(args.root, identity)
        print(json.dumps({"phase": args.phase, "n_files": len(result["files"])}))
    elif args.phase == "load":
        result = load_inputs(args.root, identity)
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "n_sources": result["n_sources"],
                    "n_rollouts": result["n_rollouts"],
                }
            )
        )
    elif args.phase in ("pilot", "full"):
        result = run_analysis(args.root, identity, full=args.phase == "full")
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "n_panels": len(result["panels"]),
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
        )
    else:
        result = export(args.root, identity)
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "revision": result["revision"],
                    "verified": result["verified"],
                }
            )
        )


def main() -> None:
    """Dispatch one phase with bounded Torch and BLAS CPU thread pools."""
    args = build_argparser().parse_args()
    identity = input_identity(args.data_revision, args.judge_revision, args.judge_prefix)
    if args.phase in ("load", "pilot", "full"):
        require_cpu_lane(args.cpu_lane)
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        _execute_phase(args, identity)


if __name__ == "__main__":
    main()
