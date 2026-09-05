#!/usr/bin/env python3
"""Issue #2569 third-family orchestration helpers.

This module does no model inference itself.  It stages the immutable Qwen/Llama
inputs, reconstructs uploaded completion shards, materializes explicit
compatibility views for the already-validated pairwise instruments, validates
the completed 3x3 writer-by-encoder bank, and assembles the three-family result
index.  Compatibility filenames are always accompanied by a manifest mapping
the historical ``qwen``/``llama`` labels to the true model and layer.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

import issue2569_xmodel_capture as XC  # noqa: E402


TASK_ID = 2569
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
QWRITER_PREFIX = "issue2569_theory/analysis_tensors/xmodel"
QWRITER_REVISION = "d3ab70c673f898870600147a311aacca19ddcfbf"
LWRITER_PREFIX = "issue2569_theory/own_generated_answers/captures/llama_writer_s42"
LWRITER_RAW_PREFIX = "issue2569_theory/own_generated_answers/raw_completions/llama_seed42"
OWNANSWER_REVISION = "8d2694f6eedfbad61b9413299bca096370429d7a"
RESULT_PREFIX = "issue2569_theory/third_family"
TRUE_LAYERS = {
    "qwen": (14, 19, 26),
    "llama": (16, 22, 30),
    "olmo": (16, 22, 30),
}
ALIAS_LAYERS = {"qwen": (14, 19, 26), "llama": (16, 22, 30)}
MODEL_IDS = {key: spec["model_id"] for key, spec in XC.MODEL_SPECS.items()}
VLLM_BANNED_ACCEL_DISTS = {"flashinfer-python": "flashinfer"}
VLLM_LAUNCH_ENV_PINS = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
CAP_HIT_REGEN_THRESHOLD = 0.02


def _atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        Path(tmp).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp, open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha_int64(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_names(model: str) -> list[str]:
    return [f"{model}_{tag}_L{layer}.pt" for layer in TRUE_LAYERS[model] for tag in ("vc", "va")]


def _stage_files(
    *, repo_id: str, prefix: str, revision: str, names: list[str], destination: Path
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        local = destination / name
        if local.is_file():
            continue
        hub.stage_hub_file(
            repo_id,
            f"{prefix}/{name}",
            local,
            repo_type="dataset",
            revision=revision,
        )
        if not local.is_file():
            raise RuntimeError(f"staging returned without required file: {local}")


def _stage_prefix_files(
    *, repo_id: str, prefix: str, revision: str, destination: Path
) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    paths = sorted(
        hub.list_hf_files_under_path(
            api,
            repo_id,
            prefix,
            repo_type="dataset",
            revision=revision,
        )
    )
    if not paths:
        raise RuntimeError(f"no files resolved at {repo_id}@{revision}:{prefix}")
    names = [str(Path(path).relative_to(prefix)) for path in paths]
    _stage_files(
        repo_id=repo_id,
        prefix=prefix,
        revision=revision,
        names=names,
        destination=destination,
    )
    return names


def reconstruct_completion_shards(root: Path) -> dict[str, Any]:
    """Rebuild the deliberately non-uploaded answers.jsonl from raw shards."""
    regime_path = root / "regime.json"
    audit_path = root / "audit.json"
    shards = sorted((root / "raw_completions").glob("shard*.json"))
    if not regime_path.is_file() or not audit_path.is_file() or not shards:
        raise RuntimeError(f"incomplete raw completion store under {root}")
    regime = json.loads(regime_path.read_text())
    audit = json.loads(audit_path.read_text())
    rows: list[dict[str, Any]] = []
    for path in shards:
        obj = json.loads(path.read_text())
        if obj.get("regime") != regime:
            raise RuntimeError(f"{path}: shard regime differs from regime.json")
        shard_rows = obj.get("rows")
        if not isinstance(shard_rows, list) or not shard_rows:
            raise RuntimeError(f"{path}: empty or invalid rows")
        rows.extend(shard_rows)
    if len(rows) != int(audit["n_rows"]):
        raise RuntimeError(f"raw row count {len(rows)} != audit n_rows {audit['n_rows']}")
    ci = np.asarray([int(row["ci"]) for row in rows], dtype=np.int64)
    if len(np.unique(ci)) != len(ci):
        raise RuntimeError("raw completion shards contain duplicate ci values")
    if _sha_int64(ci) != str(regime["ci_sha256"]):
        raise RuntimeError("raw completion roster hash differs from the pinned regime")
    _atomic_jsonl(root / "answers.jsonl", rows)
    return {
        "n_rows": len(rows),
        "n_kept": sum(row.get("drop_reason") is None for row in rows),
        "n_shards": len(shards),
        "ci_sha256": _sha_int64(ci),
    }


def build_cap_hit_roster(
    *, base_root: Path, roster_path: Path, threshold: float = CAP_HIT_REGEN_THRESHOLD
) -> dict[str, Any]:
    """Persist the ordered CIs requiring the registered cap-hit regeneration."""
    answers = _read_jsonl(base_root / "answers.jsonl")
    audit = json.loads((base_root / "audit.json").read_text())
    if len(answers) != int(audit["n_rows"]):
        raise RuntimeError("base generation answers and audit row counts differ")
    roster = [
        int(row["ci"])
        for row in answers
        if row.get("drop_reason") is None and row.get("finish_reason") == "length"
    ]
    expected = int(audit.get("finish_reasons", {}).get("length", 0))
    if len(roster) != expected:
        raise RuntimeError(f"observed cap hits {len(roster)} != audit count {expected}")
    fraction = len(roster) / len(answers)
    if not roster or fraction <= threshold:
        raise RuntimeError(
            f"cap-hit regeneration was requested but fraction={fraction:.6f} "
            f"does not exceed threshold={threshold:.6f}"
        )
    record = {
        "ci": roster,
        "n_rows": len(answers),
        "n_cap_hit": len(roster),
        "cap_hit_fraction": fraction,
        "threshold": threshold,
        "triggered": True,
        "ci_sha256": _sha_int64(np.asarray(roster, dtype=np.int64)),
        "base_audit_sha256": _file_sha256(base_root / "audit.json"),
    }
    _atomic_json(roster_path, record)
    return record


def merge_cap_hit_topup(
    *,
    base_root: Path,
    topup_root: Path,
    roster_path: Path,
    destination: Path,
    threshold: float = CAP_HIT_REGEN_THRESHOLD,
) -> dict[str, Any]:
    """Replace capped base rows with longer deterministic regeneration rows."""
    base_rows = _read_jsonl(base_root / "answers.jsonl")
    topup_rows = _read_jsonl(topup_root / "answers.jsonl")
    roster_obj = json.loads(roster_path.read_text())
    roster = [int(ci) for ci in roster_obj["ci"]]
    base_hits = [
        int(row["ci"])
        for row in base_rows
        if row.get("drop_reason") is None and row.get("finish_reason") == "length"
    ]
    topup_ci = [int(row["ci"]) for row in topup_rows]
    if base_hits != roster or topup_ci != roster:
        raise RuntimeError("base cap-hit, persisted roster, and top-up row orders differ")
    topup_by_ci = {int(row["ci"]): row for row in topup_rows}
    if len(topup_by_ci) != len(topup_rows):
        raise RuntimeError("top-up generation contains duplicate ci values")

    merged: list[dict[str, Any]] = []
    for base in base_rows:
        ci = int(base["ci"])
        if ci not in topup_by_ci:
            merged.append(base)
            continue
        topup = topup_by_ci[ci]
        if str(base["prompt"]) != str(topup["prompt"]):
            raise RuntimeError(f"ci={ci}: prompt drift in cap-hit regeneration")
        if str(base["corpus"]) != str(topup["corpus"]):
            raise RuntimeError(f"ci={ci}: corpus drift in cap-hit regeneration")
        if topup.get("drop_reason") is not None or not str(topup.get("response", "")).strip():
            raise RuntimeError(f"ci={ci}: cap-hit regeneration produced an unusable response")
        replacement = dict(topup)
        replacement["regeneration"] = {
            "reason": "base_finish_reason_length",
            "base_response_tokens": int(base.get("response_tokens", 0)),
            "base_seed": int(base["seed"]),
            "topup_seed": int(topup["seed"]),
        }
        merged.append(replacement)

    if len(merged) != len(base_rows) or len({int(row["ci"]) for row in merged}) != len(merged):
        raise RuntimeError("merged generation roster is incomplete or duplicated")
    cap_hits_after = sum(row.get("finish_reason") == "length" for row in merged)
    fraction_after = cap_hits_after / len(merged)
    if fraction_after > threshold:
        raise RuntimeError(
            f"post-regeneration cap-hit fraction {fraction_after:.6f} exceeds {threshold:.6f}"
        )
    drops = collections.Counter(
        str(row["drop_reason"]) for row in merged if row.get("drop_reason") is not None
    )
    finishes = collections.Counter(str(row.get("finish_reason")) for row in merged)
    audit = {
        "kind": "cap-hit-regeneration-merge",
        "n_rows": len(merged),
        "n_kept": sum(row.get("drop_reason") is None for row in merged),
        "drops": dict(drops),
        "finish_reasons": dict(finishes),
        "cap_hit": {
            "threshold": threshold,
            "n_before": len(base_hits),
            "fraction_before": len(base_hits) / len(base_rows),
            "n_after": cap_hits_after,
            "fraction_after": fraction_after,
            "n_replaced": len(topup_rows),
        },
        "base": {
            "root": str(base_root.resolve()),
            "audit_sha256": _file_sha256(base_root / "audit.json"),
        },
        "topup": {
            "root": str(topup_root.resolve()),
            "audit_sha256": _file_sha256(topup_root / "audit.json"),
            "roster_sha256": _file_sha256(roster_path),
        },
        "ci_sha256": _sha_int64(np.asarray([int(row["ci"]) for row in merged])),
    }
    _atomic_jsonl(destination / "answers.jsonl", merged)
    _atomic_json(destination / "audit.json", audit)
    _atomic_json(
        destination / "regime.json",
        {
            "kind": "base-plus-cap-hit-regeneration",
            "base_regime": json.loads((base_root / "regime.json").read_text()),
            "topup_regime": json.loads((topup_root / "regime.json").read_text()),
            "merge_audit_sha256": _file_sha256(destination / "audit.json"),
        },
    )
    return audit


def materialize_candidate_source(
    *, qwriter_source: Path, raw_completion_root: Path, destination: Path
) -> dict[str, Any]:
    """Recover the exact historical generation roster from immutable shards.

    The raw completion records pin roster order and prompt identity, while the
    full Qwen-written source supplies the response text needed by the crossed
    Qwen-vs-new-writer semantic analysis.
    """
    answers_path = raw_completion_root / "answers.jsonl"
    regime_path = raw_completion_root / "regime.json"
    if not qwriter_source.is_file() or not answers_path.is_file() or not regime_path.is_file():
        raise RuntimeError("candidate source inputs are incomplete")
    qwriter_rows = _read_jsonl(qwriter_source)
    answers = _read_jsonl(answers_path)
    regime = json.loads(regime_path.read_text())
    qwriter_ci = [int(row["ci"]) for row in qwriter_rows]
    answer_ci = [int(row["ci"]) for row in answers]
    if len(set(qwriter_ci)) != len(qwriter_ci):
        raise RuntimeError("Qwen source contains duplicate ci values")
    if len(set(answer_ci)) != len(answer_ci):
        raise RuntimeError("historical answer roster contains duplicate ci values")
    if len(answers) != int(regime["source_rows"]):
        raise RuntimeError(
            f"historical answer count {len(answers)} != regime source_rows "
            f"{regime['source_rows']}"
        )
    answer_ci_sha = _sha_int64(np.asarray(answer_ci, dtype=np.int64))
    if answer_ci_sha != str(regime["ci_sha256"]):
        raise RuntimeError("historical answer roster differs from the pinned regime")

    qwriter_by_ci = {int(row["ci"]): row for row in qwriter_rows}
    missing = [ci for ci in answer_ci if ci not in qwriter_by_ci]
    if missing:
        raise RuntimeError(f"{len(missing)} historical rows are absent from the Qwen source")
    candidate_rows = [qwriter_by_ci[ci] for ci in answer_ci]
    for source, generated in zip(candidate_rows, answers, strict=True):
        ci = int(source["ci"])
        if str(source["prompt"]) != str(generated["prompt"]):
            raise RuntimeError(f"ci={ci}: prompt drift between source and historical generation")
        if str(source["corpus"]) != str(generated["corpus"]):
            raise RuntimeError(f"ci={ci}: corpus drift between source and historical generation")
    source_text_sha = XC._texts_content_sha(candidate_rows)
    if source_text_sha != str(regime["source_text_sha256"]):
        raise RuntimeError("candidate source content differs from the pinned generation regime")

    _atomic_jsonl(destination / "texts_kept.jsonl", candidate_rows)
    record = {
        "source": str(qwriter_source.resolve()),
        "historical_generation_root": str(raw_completion_root.resolve()),
        "n_full_qwriter_source": len(qwriter_rows),
        "n_candidate": len(candidate_rows),
        "candidate_ci_sha256": answer_ci_sha,
        "candidate_text_sha256": source_text_sha,
        "pinned_regime_ci_sha256": str(regime["ci_sha256"]),
        "pinned_regime_text_sha256": str(regime["source_text_sha256"]),
        "selection": "historical raw-completion CI order indexed into full Qwen source",
    }
    _atomic_json(destination / "source_manifest.json", record)
    return record


def assert_vllm_accelerator_compat() -> dict[str, Any]:
    """Fail before engine init on the known vLLM/Python 3.11 incompatibility."""
    banned_present = [
        module
        for module in VLLM_BANNED_ACCEL_DISTS.values()
        if importlib.util.find_spec(module) is not None
    ]
    if banned_present:
        raise RuntimeError(
            f"banned accelerator import(s) present: {banned_present}; "
            "uninstall them after resolving the vLLM dependency closure"
        )
    wrong_env = {
        key: os.environ.get(key)
        for key, expected in VLLM_LAUNCH_ENV_PINS.items()
        if os.environ.get(key) != expected
    }
    if wrong_env:
        raise RuntimeError(f"vLLM launch environment pins differ: {wrong_env}")
    # Deliberately unguarded: EngineCore imports this lazily and an optional
    # dependency's non-ImportError exception would otherwise escape there.
    importlib.import_module("vllm.compilation.backends")
    return {
        "banned_distributions_absent": sorted(VLLM_BANNED_ACCEL_DISTS),
        "launch_env": dict(VLLM_LAUNCH_ENV_PINS),
        "compile_backend_import": "pass",
    }


def phase_preflight(args: argparse.Namespace) -> None:
    import transformers
    import vllm
    from transformers import AutoTokenizer

    expected = {"transformers": "5.15.0", "vllm": "0.27.1"}
    realized = {"transformers": transformers.__version__, "vllm": vllm.__version__}
    if realized != expected:
        raise RuntimeError(f"third-family stack mismatch: got {realized}, expected {expected}")
    accelerator_compat = assert_vllm_accelerator_compat()
    spec = XC.MODEL_SPECS["olmo"]
    stack = XC.assert_model_stack(spec)
    tok = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"])
    template = XC.template_probe(tok, "olmo")
    record = {
        "issue": TASK_ID,
        "model": spec,
        "stack": stack,
        "template": template,
        "expected_stack": expected,
        "accelerator_compat": accelerator_compat,
    }
    _atomic_json(Path(args.work_root) / "preflight.json", record)
    print(f"[third-family] preflight PASS: {realized}", flush=True)


def phase_stage_existing(args: argparse.Namespace) -> None:
    root = Path(args.work_root)
    qwriter = root / "bank" / "qwriter" / "final"
    lwriter = root / "bank" / "lwriter" / "final"
    raw_lwriter = root / "bank" / "gen_llama_s42"
    _stage_files(
        repo_id=args.hf_data_repo,
        prefix=QWRITER_PREFIX,
        revision=QWRITER_REVISION,
        names=_bundle_names("qwen") + _bundle_names("llama"),
        destination=qwriter,
    )
    _stage_files(
        repo_id=args.hf_data_repo,
        prefix=LWRITER_PREFIX,
        revision=OWNANSWER_REVISION,
        names=_bundle_names("qwen") + _bundle_names("llama"),
        destination=lwriter,
    )
    raw_names = _stage_prefix_files(
        repo_id=args.hf_data_repo,
        prefix=LWRITER_RAW_PREFIX,
        revision=OWNANSWER_REVISION,
        destination=raw_lwriter,
    )
    reconstruction = reconstruct_completion_shards(raw_lwriter)
    candidate_source = materialize_candidate_source(
        qwriter_source=root / "source_qwen" / "texts_kept.jsonl",
        raw_completion_root=raw_lwriter,
        destination=root / "source_candidate",
    )
    record = {
        "qwriter": {
            "prefix": QWRITER_PREFIX,
            "revision": QWRITER_REVISION,
            "files": _bundle_names("qwen") + _bundle_names("llama"),
        },
        "lwriter": {
            "prefix": LWRITER_PREFIX,
            "raw_prefix": LWRITER_RAW_PREFIX,
            "revision": OWNANSWER_REVISION,
            "capture_files": _bundle_names("qwen") + _bundle_names("llama"),
            "raw_files": raw_names,
            "reconstruction": reconstruction,
        },
        "candidate_source": candidate_source,
    }
    _atomic_json(root / "bank" / "staged_inputs.json", record)
    print("[third-family] immutable Qwen/Llama inputs staged and verified", flush=True)


def phase_topup_roster(args: argparse.Namespace) -> None:
    root = Path(args.work_root)
    record = build_cap_hit_roster(
        base_root=root / "capture" / "gen_olmo_s42",
        roster_path=root / "capture" / "gen_olmo_s42_topup" / "roster.json",
    )
    print(
        f"[third-family] OLMo cap-hit top-up triggered: "
        f"{record['n_cap_hit']}/{record['n_rows']}",
        flush=True,
    )


def phase_merge_topup(args: argparse.Namespace) -> None:
    root = Path(args.work_root)
    audit = merge_cap_hit_topup(
        base_root=root / "capture" / "gen_olmo_s42",
        topup_root=root / "capture" / "gen_olmo_s42_topup",
        roster_path=root / "capture" / "gen_olmo_s42_topup" / "roster.json",
        destination=root / "capture" / "gen_olmo_s42_merged",
    )
    cap = audit["cap_hit"]
    print(
        f"[third-family] OLMo cap-hit merge PASS: "
        f"{cap['n_before']} -> {cap['n_after']} ({cap['fraction_after']:.6f})",
        flush=True,
    )


def _verify_bundle(
    path: Path, *, true_model: str, true_layer: int, tag: str, min_rows: int
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"capture bundle missing: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    expected_slot = "v_C" if tag == "vc" else "v_A"
    if str(bundle.get("model_id")) != MODEL_IDS[true_model]:
        raise RuntimeError(
            f"{path}: model_id={bundle.get('model_id')!r}, expected {MODEL_IDS[true_model]!r}"
        )
    if int(bundle.get("layer", -1)) != int(true_layer):
        raise RuntimeError(f"{path}: layer={bundle.get('layer')}, expected {true_layer}")
    if str(bundle.get("slot")) != expected_slot:
        raise RuntimeError(f"{path}: slot={bundle.get('slot')!r}, expected {expected_slot}")
    ci = np.asarray(bundle["ci"], dtype=np.int64)
    if len(ci) < min_rows or len(np.unique(ci)) != len(ci):
        raise RuntimeError(f"{path}: invalid realized roster n={len(ci)}, min={min_rows}")
    x = bundle["x"]
    if tuple(x.shape) != (len(ci), XC.MODEL_SPECS[true_model]["hidden"]):
        raise RuntimeError(f"{path}: stored shape={tuple(x.shape)} is inconsistent")
    return {"n": len(ci), "ci_sha256": _sha_int64(ci)}


def _link_checked(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not dst.is_file() or src.stat().st_size != dst.stat().st_size:
            raise RuntimeError(f"existing compatibility target differs in kind/size: {dst}")
        return
    os.link(src, dst)


def _materialize_alias_writer(
    *,
    destination: Path,
    source_root: Path,
    source_model: str,
    target_root: Path,
    target_model: str,
    min_rows: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for alias_model, true_model, true_root in (
        ("qwen", source_model, source_root),
        ("llama", target_model, target_root),
    ):
        for alias_layer, true_layer in zip(
            ALIAS_LAYERS[alias_model], TRUE_LAYERS[true_model], strict=True
        ):
            for tag in ("vc", "va"):
                src = true_root / f"{true_model}_{tag}_L{true_layer}.pt"
                meta = _verify_bundle(
                    src,
                    true_model=true_model,
                    true_layer=true_layer,
                    tag=tag,
                    min_rows=min_rows,
                )
                dst = destination / f"{alias_model}_{tag}_L{alias_layer}.pt"
                _link_checked(src, dst)
                records.append(
                    {
                        "alias": str(dst),
                        "source": str(src),
                        "alias_model": alias_model,
                        "true_model": true_model,
                        "alias_layer": alias_layer,
                        "true_layer": true_layer,
                        "slot": tag,
                        **meta,
                    }
                )
    return records


def phase_build_pairs(args: argparse.Namespace) -> None:
    root = Path(args.work_root)
    qbank = root / "bank" / "qwriter" / "final"
    lbank = root / "bank" / "lwriter" / "final"
    q_olmo = root / "capture" / "qwriter_olmo" / "final"
    owriter = root / "capture" / "owriter" / "final"
    pair_specs = {
        "qo": {
            "source": "qwen",
            "source_writer": "qwriter",
            "source_writer_root": qbank,
            "source_encoder_root": qbank,
            "target_writer_root": owriter,
            "target_encoder_qwriter_root": q_olmo,
        },
        "lo": {
            "source": "llama",
            "source_writer": "lwriter",
            "source_writer_root": lbank,
            "source_encoder_root": lbank,
            "target_writer_root": owriter,
            "target_encoder_qwriter_root": lbank,
        },
    }
    for pair, spec in pair_specs.items():
        pair_root = root / "pairs" / pair
        source_writer_alias = pair_root / "source_writer"
        target_writer_alias = pair_root / "olmo_writer"
        source_records = _materialize_alias_writer(
            destination=source_writer_alias,
            source_root=Path(spec["source_encoder_root"]),
            source_model=str(spec["source"]),
            target_root=Path(spec["target_encoder_qwriter_root"]),
            target_model="olmo",
            min_rows=args.analysis_rows,
        )
        target_records = _materialize_alias_writer(
            destination=target_writer_alias,
            source_root=Path(spec["target_writer_root"]),
            source_model=str(spec["source"]),
            target_root=Path(spec["target_writer_root"]),
            target_model="olmo",
            min_rows=args.analysis_rows,
        )
        manifest = {
            "issue": TASK_ID,
            "pair": pair,
            "source_model": str(spec["source"]),
            "target_model": "olmo",
            "source_writer": str(spec["source_writer"]),
            "target_writer": "olmo_writer",
            "compatibility_aliases": {
                "qwen": str(spec["source"]),
                "llama": "olmo",
            },
            "claim_scope": (
                f"transformations of shared teacher-forced {spec['source']}-written "
                f"responses under {spec['source']} and OLMo encoders; pairwise labels "
                "are compatibility aliases declared in this manifest"
            ),
            "source_writer_files": source_records,
            "target_writer_files": target_records,
        }
        _atomic_json(pair_root / "pair_manifest.json", manifest)
        print(f"[third-family] built compatibility pair {pair}", flush=True)

    atlas_specs = {
        "qo": ("qwen", qbank, q_olmo),
        "lo": ("llama", qbank, q_olmo),
    }
    for pair, (source_model, source_root, target_root) in atlas_specs.items():
        atlas_root = root / "atlas" / pair
        records = _materialize_alias_writer(
            destination=atlas_root / "captures",
            source_root=source_root,
            source_model=source_model,
            target_root=target_root,
            target_model="olmo",
            min_rows=args.atlas_min_rows,
        )
        _atomic_json(
            atlas_root / "pair_manifest.json",
            {
                "issue": TASK_ID,
                "pair": pair,
                "source_model": source_model,
                "target_model": "olmo",
                "compatibility_aliases": {"qwen": source_model, "llama": "olmo"},
                "claim_scope": (
                    "transformations of the frozen shared teacher-forced Qwen-written "
                    f"responses under {source_model} and OLMo encoders; pairwise labels "
                    "are compatibility aliases declared in this manifest"
                ),
                "files": records,
            },
        )
        print(f"[third-family] built same-text atlas pair {pair}", flush=True)


def phase_validate_bank(args: argparse.Namespace) -> None:
    root = Path(args.work_root)
    writers = {
        "qwen": {
            "qwen": root / "bank" / "qwriter" / "final",
            "llama": root / "bank" / "qwriter" / "final",
            "olmo": root / "capture" / "qwriter_olmo" / "final",
        },
        "llama": {model: root / "bank" / "lwriter" / "final" for model in MODEL_IDS},
        "olmo": {model: root / "capture" / "owriter" / "final" for model in MODEL_IDS},
    }
    cells: dict[str, Any] = {}
    writer_rosters: dict[str, dict[str, set[int]]] = {}
    for writer, encoders in writers.items():
        writer_rosters[writer] = {}
        for encoder, bundle_root in encoders.items():
            per_file = []
            for layer in TRUE_LAYERS[encoder]:
                for tag in ("vc", "va"):
                    path = bundle_root / f"{encoder}_{tag}_L{layer}.pt"
                    per_file.append(
                        _verify_bundle(
                            path,
                            true_model=encoder,
                            true_layer=layer,
                            tag=tag,
                            min_rows=args.analysis_rows,
                        )
                    )
            roster_hashes = {rec["ci_sha256"] for rec in per_file}
            if len(roster_hashes) != 1:
                raise RuntimeError(f"writer={writer}, encoder={encoder}: layer roster drift")
            first_layer = TRUE_LAYERS[encoder][0]
            roster_bundle = torch.load(
                bundle_root / f"{encoder}_vc_L{first_layer}.pt",
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            writer_rosters[writer][encoder] = {
                int(ci) for ci in np.asarray(roster_bundle["ci"], dtype=np.int64)
            }
            cells[f"writer={writer}|encoder={encoder}"] = {
                "n_min": min(rec["n"] for rec in per_file),
                "ci_sha256": next(iter(roster_hashes)),
                "layers": TRUE_LAYERS[encoder],
            }
    intersections = {
        writer: len(set.intersection(*by_encoder.values()))
        for writer, by_encoder in writer_rosters.items()
    }
    insufficient = {writer: n for writer, n in intersections.items() if n < args.analysis_rows}
    if insufficient:
        raise RuntimeError(
            f"writer-wise three-encoder intersections below {args.analysis_rows}: {insufficient}"
        )
    _atomic_json(
        root / "bank" / "three_by_three_manifest.json",
        {
            "issue": TASK_ID,
            "writers": list(writers),
            "encoders": list(MODEL_IDS),
            "cells": cells,
            "three_encoder_intersection_rows": intersections,
            "complete": len(cells) == 9,
        },
    )
    print("[third-family] 3x3 writer-by-encoder bank PASS (9/9 cells)", flush=True)


def _read_required(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required result missing: {path}")
    return json.loads(path.read_text())


def phase_summarize(args: argparse.Namespace) -> None:
    root = Path(args.work_root)
    ql = _read_required(Path(args.ql_fits_summary))
    pair_paths = {
        "qwen-llama": Path(args.ql_fits_summary),
        "qwen-olmo": root / "atlas" / "qo" / "fits" / "fits_summary.json",
        "llama-olmo": root / "atlas" / "lo" / "fits" / "fits_summary.json",
    }
    pair_fits = {name: _read_required(path) for name, path in pair_paths.items()}
    model_order = ["qwen", "llama", "olmo"]
    cosine = np.eye(3, dtype=np.float64)
    pair_records: dict[str, Any] = {}
    for name, summary in pair_fits.items():
        source, target = name.split("-")
        aligned = summary["tier2_aligned_operator_cosine"]
        value = float(aligned["per_operator"]["qwen_matched"]["observed_aligned_cosine"])
        i, j = model_order.index(source), model_order.index(target)
        cosine[i, j] = cosine[j, i] = value
        pair_records[name] = {
            "matched_native_aligned_operator_cosine": value,
            "working_pair": summary["working_pair"],
            "realized_paired_rows": summary["realized_paired_rows"],
            "claim_scope": summary["claim_scope"],
            "source": str(pair_paths[name]),
        }
    crossed: dict[str, Any] = {}
    for pair in ("qo", "lo"):
        base = root / "pairs" / pair / "analysis"
        crossed[pair] = {
            "manifest": str(root / "pairs" / pair / "pair_manifest.json"),
            "crossed_geometry": _read_required(base / "crossed_geometry.json"),
            "mapping_diff": _read_required(base / "mapping_diff" / "mapping_diff.json"),
            "query_scaling_unpaired": _read_required(
                base / "query_scaling_unpaired" / "query_scaling_unpaired.json"
            ),
        }
    result = {
        "issue": TASK_ID,
        "label": "third-family-full-parity",
        "models": model_order,
        "model_ids": MODEL_IDS,
        "input_revisions": {
            "qwen_writer_captures": QWRITER_REVISION,
            "llama_writer_captures": OWNANSWER_REVISION,
        },
        "bank_manifest": _read_required(root / "bank" / "three_by_three_manifest.json"),
        "olmo_generation": {
            "base": _read_required(root / "capture" / "gen_olmo_s42" / "audit.json"),
            "topup": _read_required(
                root / "capture" / "gen_olmo_s42_topup" / "audit.json"
            ),
            "merged": _read_required(
                root / "capture" / "gen_olmo_s42_merged" / "audit.json"
            ),
        },
        "all_pairs_atlas": {
            "metric": "activation-Procrustes-aligned cosine between matched native operators",
            "cosine_matrix": cosine.tolist(),
            "distance_matrix": (1.0 - cosine).tolist(),
            "pairs": pair_records,
        },
        "crossed_followups": crossed,
        "legacy_qwen_llama_fits": ql,
    }
    out = root / "results" / "third_family_summary.json"
    _atomic_json(out, result)
    print(f"[third-family] wrote {out}", flush=True)


def phase_upload_results(args: argparse.Namespace) -> None:
    if not args.result_prefix:
        raise AssertionError("--result-prefix is required for --phase upload-results")
    root = Path(args.work_root)
    candidates: list[Path] = []
    for relative in (
        "atlas/qo/fits",
        "atlas/qo/report",
        "atlas/lo/fits",
        "atlas/lo/report",
        "pairs/qo/analysis/mapping_diff",
        "pairs/qo/analysis/query_scaling_unpaired",
        "pairs/lo/analysis/mapping_diff",
        "pairs/lo/analysis/query_scaling_unpaired",
        "results",
        "bank",
    ):
        base = root / relative
        if base.exists():
            candidates.extend(path for path in base.rglob("*") if path.is_file())
    allowed_suffixes = {".json", ".jsonl", ".pt", ".npz"}
    files = sorted({path for path in candidates if path.suffix in allowed_suffixes})
    files = [
        path
        for path in files
        if not (path.is_relative_to(root / "bank") and path.suffix in {".pt", ".npz"})
    ]
    if not files:
        raise RuntimeError("no third-family result files found for upload")
    names = [str(path.relative_to(root)) for path in files]
    upload_manifest = root / "results" / "upload_manifest.json"
    _atomic_json(
        upload_manifest,
        {
            "issue": TASK_ID,
            "result_prefix": args.result_prefix,
            "files": names,
            "n_files": len(names),
        },
    )
    if upload_manifest not in files:
        files.append(upload_manifest)
        names.append(str(upload_manifest.relative_to(root)))
    url = hub._upload_folder_filtered(
        root,
        repo_id=args.hf_data_repo,
        repo_type="dataset",
        path_in_repo=args.result_prefix,
        allow_patterns=names,
        expected_repo_paths=[f"{args.result_prefix}/{name}" for name in names],
    )
    if not url:
        raise RuntimeError(f"result upload returned no URL for {args.result_prefix}")
    print(f"[third-family] verified result upload: {len(names)} files -> {args.result_prefix}")


def phase_sentinel(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    if not args.sentinel_path:
        raise AssertionError("--sentinel-path is required for --phase sentinel")
    if not args.result_prefix:
        raise AssertionError("--result-prefix is required for --phase sentinel")
    root = Path(args.work_root)
    summary = root / "results" / "third_family_summary.json"
    manifest = root / "results" / "upload_manifest.json"
    if not summary.is_file() or not manifest.is_file():
        raise RuntimeError("terminal result or upload manifest is absent")
    remote = f"{args.result_prefix}/results/third_family_summary.json"
    info = HfApi().get_paths_info(args.hf_data_repo, [remote], repo_type="dataset")
    if len(info) != 1 or info[0].path != remote:
        raise RuntimeError(f"remote terminal summary did not verify: {remote}")
    bank = _read_required(root / "bank" / "three_by_three_manifest.json")
    if not bank.get("complete") or len(bank.get("cells", {})) != 9:
        raise RuntimeError("3x3 bank manifest is not complete")
    payload = {
        "sentinel_schema_version": 1,
        "kind": "third-family-full-parity-done",
        "version": 1,
        "blocks_pipeline": False,
        "note": "third-family full-parity follow-up complete; uploads verified",
        "issue": TASK_ID,
        "phase": "done",
        "status": "ok",
        "rc": 0,
        "out_root": str(root),
        "result_path": str(summary),
        "hf_result_path": remote,
        "realized_cells": 9,
    }
    _atomic_json(Path(args.sentinel_path), payload)
    print("[phase=done]", flush=True)


def phase_selftest(args: argparse.Namespace) -> None:
    root = Path(args.work_root) / "selftest"
    raw = root / "raw"
    regime = {"ci_sha256": _sha_int64(np.asarray([5, 7], dtype=np.int64))}
    _atomic_json(raw / "regime.json", regime)
    rows = [
        {"ci": 5, "response": "a", "drop_reason": None},
        {"ci": 7, "response": "b", "drop_reason": None},
    ]
    _atomic_json(raw / "audit.json", {"n_rows": 2})
    _atomic_json(raw / "raw_completions" / "shard00000.json", {"regime": regime, "rows": rows})
    rec = reconstruct_completion_shards(raw)
    assert rec["n_rows"] == rec["n_kept"] == 2
    with open(raw / "answers.jsonl", encoding="utf-8") as handle:
        assert sum(bool(line.strip()) for line in handle) == 2
    print("[third-family] selftest PASS", flush=True)


PHASES = {
    "preflight": phase_preflight,
    "stage-existing": phase_stage_existing,
    "topup-roster": phase_topup_roster,
    "merge-topup": phase_merge_topup,
    "build-pairs": phase_build_pairs,
    "validate-bank": phase_validate_bank,
    "summarize": phase_summarize,
    "upload-results": phase_upload_results,
    "sentinel": phase_sentinel,
    "selftest": phase_selftest,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=sorted(PHASES))
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--work-root", default="/workspace/issue2569-third-family")
    parser.add_argument("--analysis-rows", type=int, default=10_000)
    parser.add_argument("--atlas-min-rows", type=int, default=50_000)
    parser.add_argument("--hf-data-repo", default=HF_DATA_REPO)
    parser.add_argument("--result-prefix", default=None)
    parser.add_argument("--sentinel-path", default="")
    parser.add_argument(
        "--ql-fits-summary",
        default=str(PROJECT_ROOT / "eval_results" / "issue_2569" / "xmodel" / "fits_summary.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.import_check:
        from huggingface_hub import HfApi as _HfApi  # noqa: F401
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert set(MODEL_IDS) == {"qwen", "llama", "olmo"}
        assert_args_attributes_defined(__file__)
        print("[third-family] import-check PASS")
        return
    if not args.phase:
        raise AssertionError("--phase is required (or --import-check)")
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
