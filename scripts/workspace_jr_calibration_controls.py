#!/usr/bin/env python3
"""Measure actual and rotated dictionary quality on the frozen lens-calibration tokens."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.workspace_analysis_inputs import _upload_binding  # noqa: E402
from explore_persona_space.analysis.workspace_artifacts import validate_producer  # noqa: E402
from explore_persona_space.analysis.workspace_decomposition import ValidatedDictionary  # noqa: E402
from explore_persona_space.analysis.workspace_fit import finite_json  # noqa: E402
from explore_persona_space.analysis.workspace_gate import validate_main_readiness  # noqa: E402
from explore_persona_space.analysis.workspace_lenses import rotated_dictionary  # noqa: E402
from explore_persona_space.analysis.workspace_quality import (  # noqa: E402
    equal_prompt_quality,
    nearest_quality_match,
    prompt_moments,
)
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    file_sha256,
    load_native,
    load_workspace_jr_config,
    run_identity,
    save_json,
    save_tensors,
)


@torch.no_grad()
def native_calibration_states(text, token_ids, source_layer, skip_first):
    """Preserve the exact individual unpadded forward geometry of lens calibration."""
    if len(token_ids) <= skip_first + 1:
        raise ValueError("Calibration sequence has no valid source positions")
    ids = torch.tensor([token_ids], device=text.embed_tokens.weight.device)
    cache = {}

    def hook(_module, _inputs, output):
        values = output if isinstance(output, torch.Tensor) else output[0]
        cache["states"] = values[0, skip_first:-1].detach().cpu()

    handle = text.layers[source_layer].register_forward_hook(hook)
    try:
        text(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    states = cache["states"]
    if (
        states.ndim != 2
        or len(states) != len(token_ids) - skip_first - 1
        or not torch.isfinite(states).all()
    ):
        raise ValueError("Invalid native calibration source states")
    return states


def capture(args, config, identity, evidence):
    """Checkpoint each calibration prompt; no sampled main answers are loaded."""
    if (args.out / "capture_complete.json").exists():
        raise FileExistsError("Completed calibration capture is immutable")
    contract = {
        "identity": identity,
        "readiness_sha256": evidence["readiness_sha256"],
        "token_manifest_sha256": file_sha256(evidence["paths"]["tokens"]),
        "forward_geometry": "one_unpadded_raw_calibration_sequence",
        "source_layer": config["models"][args.role]["source_layer"],
        "positions": "skip_first_four_exclude_final",
        "dtype": "bfloat16",
        "attention": "eager",
    }
    manifest = args.out / "capture_contract.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("Cannot resume changed calibration capture")
    save_json(manifest, contract)
    model, _tokenizer, text = load_native(
        config, args.role, device=args.device, dtype=torch.bfloat16
    )
    if (
        type(model).__name__ != "Qwen3_5ForConditionalGeneration"
        or text.config._attn_implementation != "eager"
    ):
        raise ValueError("Calibration quality forward differs from native calibration")
    rows = evidence["reports"]["tokens"]["rows"]
    ledger = []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        path = args.out / "captures" / f"prompt-{index:04d}.pt"
        expected = {
            "contract": contract,
            "prompt_sha256": row["prompt_sha256"],
            "token_ids": row["token_ids"],
        }
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if any(saved[key] != value for key, value in expected.items()):
                raise ValueError("Existing calibration capture metadata differs")
        else:
            states = native_calibration_states(
                text, row["token_ids"], contract["source_layer"], config["lenses"]["skip_first"]
            )
            save_tensors(path, {**expected, "states": states})
        ledger.append(
            {
                "file": str(path.relative_to(args.out)),
                "sha256": file_sha256(path),
                "prompt_sha256": row["prompt_sha256"],
            }
        )
        print(
            f"calibration capture prompt={index + 1}/{len(rows)} key={row['prompt_sha256']} elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )
    save_json(
        args.out / "capture_complete.json",
        {"status": "complete", "contract": contract, "files": ledger},
    )


@torch.no_grad()
def decompose_prompt(h, bases):
    """Use the same nested pursuit and ragged 128-token batches as main decomposition."""
    outputs = {}
    fields = (
        "active_atoms",
        "squared_error",
        "input_squared_norm",
        "zero_update_steps",
        "increasing_error_steps",
    )
    for arm, basis in bases.items():
        pieces = {k: {"component": [], **{key: [] for key in fields}} for k in (5, 10, 25)}
        for begin in range(0, len(h), 128):
            parts = basis.decompose(
                h[begin : begin + 128].float().to(basis.dictionary.device), (5, 10, 25)
            )
            for k, part in parts.items():
                for key in pieces[k]:
                    pieces[k][key].append(getattr(part, key).cpu())
        outputs[arm] = {
            k: {key: torch.cat(values) for key, values in part.items()}
            for k, part in pieces.items()
        }
    return outputs


def decompose(args, config, identity, evidence):
    """Bind uploaded captures/dictionaries before each resumable control orientation."""
    folder = args.out / "controls" / f"rotation{args.rotation}"
    if (folder / "quality.json").exists():
        raise FileExistsError("Completed calibration control orientation is immutable")
    captured = _upload_binding(args.out, args.capture_upload_receipt)
    captured(args.out / "capture_complete.json")
    completion = json.loads((args.out / "capture_complete.json").read_text())
    expected_rows = evidence["reports"]["tokens"]["rows"]
    if completion["status"] != "complete" or [r["prompt_sha256"] for r in completion["files"]] != [
        r["prompt_sha256"] for r in expected_rows
    ]:
        raise ValueError("Calibration control capture does not cover the frozen corpus")
    validate_producer(completion["contract"]["identity"], identity)
    if completion["contract"]["readiness_sha256"] != evidence["readiness_sha256"]:
        raise ValueError("Calibration control capture used a different readiness bundle")
    verified = _upload_binding(args.input_root, args.dictionary_upload_receipt)
    manifest_path = args.input_root / "dictionaries/manifest.json"
    verified(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validate_producer(manifest["identity"], identity, native_ancestor=True)
    if (
        manifest["pilot_only"]
        or manifest["readiness_sha256"] != evidence["readiness_sha256"]
        or manifest["means_sha256"] != file_sha256(evidence["paths"]["means"])
    ):
        raise ValueError("Calibration control dictionaries are not the reviewed full main pair")
    contract = {
        "identity": identity,
        "rotation": args.rotation,
        "k_values": [5, 10, 25],
        "capture_complete_sha256": file_sha256(args.out / "capture_complete.json"),
        "capture_upload_sha256": file_sha256(args.capture_upload_receipt),
        "dictionary_manifest_sha256": file_sha256(manifest_path),
        "dictionary_upload_sha256": file_sha256(args.dictionary_upload_receipt),
        "readiness_sha256": evidence["readiness_sha256"],
        "precision": "FP32_highest_TF32_disabled",
        "token_batch_size": 128,
    }
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    bases, rotation_q = {}, None
    for arm in ("J", "R"):
        path = args.input_root / "dictionaries" / f"{arm}.pt"
        if verified(path) != manifest["arms"][arm]["sha256"]:
            raise ValueError("Main dictionary bytes changed")
        dictionary = torch.load(path, map_location=args.device, weights_only=True)["dictionary"]
        if args.rotation is not None:
            dictionary, q = rotated_dictionary(dictionary, seed=args.rotation)
            if rotation_q is not None:
                torch.testing.assert_close(q, rotation_q, rtol=0, atol=0)
            rotation_q = q
        bases[arm] = ValidatedDictionary(dictionary)
    folder.mkdir(parents=True, exist_ok=True)
    if rotation_q is not None:
        q_path = folder / "rotation_q.pt"
        if q_path.exists():
            torch.testing.assert_close(
                torch.load(q_path, map_location=args.device, weights_only=True),
                rotation_q,
                rtol=0,
                atol=0,
            )
        else:
            save_tensors(q_path, rotation_q.cpu())
    moments = {arm: {k: [] for k in (5, 10, 25)} for arm in bases}
    stats = {arm: {k: [] for k in (5, 10, 25)} for arm in bases}
    ledger = []
    for index, (row, expected) in enumerate(zip(completion["files"], expected_rows, strict=True)):
        started = time.perf_counter()
        source = args.out / row["file"]
        if captured(source) != row["sha256"]:
            raise ValueError("Calibration source changed after capture")
        saved = torch.load(source, map_location="cpu", weights_only=True)
        if (
            saved["contract"] != completion["contract"]
            or saved["token_ids"] != expected["token_ids"]
            or saved["prompt_sha256"] != expected["prompt_sha256"]
        ):
            raise ValueError("Calibration source tokens or contract differ")
        h = saved["states"].float()
        path = folder / f"prompt-{index:04d}.pt"
        binding = {
            "contract": contract,
            "source_sha256": row["sha256"],
            "prompt_sha256": row["prompt_sha256"],
        }
        if path.exists():
            parts = torch.load(path, map_location="cpu", weights_only=True)
            if any(parts[key] != value for key, value in binding.items()):
                raise ValueError("Cannot resume changed calibration decomposition")
        else:
            parts = {**binding, "parts": decompose_prompt(h, bases)}
            save_tensors(path, parts)
        for arm in bases:
            for k in (5, 10, 25):
                part = parts["parts"][arm][k]
                moments[arm][k].append(prompt_moments(h.numpy(), part["component"].numpy()))
                stats[arm][k].append(
                    {
                        key: float(part[key].double().mean())
                        for key in ("active_atoms", "zero_update_steps", "increasing_error_steps")
                    }
                )
        ledger.append({"file": str(path.relative_to(args.out)), "sha256": file_sha256(path)})
        print(
            f"calibration control rotation={args.rotation} prompt={index + 1}/{len(expected_rows)} key={row['prompt_sha256']} elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )
    summary = {}
    for arm in bases:
        summary[arm] = {}
        for k in (5, 10, 25):
            summary[arm][k] = equal_prompt_quality(moments[arm][k])
            summary[arm][k].update(
                {
                    f"mean_{key}": float(np.mean([row[key] for row in stats[arm][k]]))
                    for key in stats[arm][k][0]
                }
            )
    save_json(
        folder / "quality.json",
        finite_json(
            {"status": "complete", "contract": contract, "summary": summary, "files": ledger}
        ),
    )


def match(args, config, identity, evidence):
    """Select supplementary grid matches using only uploaded calibration reports."""
    output = args.out / "quality_matches.json"
    if output.exists():
        raise FileExistsError("Completed calibration matching is immutable")
    verified = _upload_binding(args.out, args.quality_upload_receipt)
    reports, sources, common = {}, {}, None
    for rotation in (None, *config["decomposition"]["rotations"]):
        path = args.out / "controls" / f"rotation{rotation}" / "quality.json"
        sources[str(rotation)] = verified(path)
        report = json.loads(path.read_text())
        contract = report["contract"]
        validate_producer(contract["identity"], identity)
        if (
            report["status"] != "complete"
            or contract["rotation"] != rotation
            or contract["readiness_sha256"] != evidence["readiness_sha256"]
        ):
            raise ValueError("Calibration quality report identity or completion differs")
        shared = {key: value for key, value in contract.items() if key != "rotation"}
        if common is not None and shared != common:
            raise ValueError("Calibration quality reports do not share exact inputs")
        common = shared
        for row in report["files"]:
            if verified(args.out / row["file"]) != row["sha256"]:
                raise ValueError("Calibration quality source changed")
        if set(report["summary"]) != {"J", "R"} or any(
            set(values) != {"5", "10", "25"} for values in report["summary"].values()
        ):
            raise ValueError("Calibration quality coverage is incomplete")
        reports[rotation] = report
    matches = {
        arm: {
            k: {
                rotation: nearest_quality_match(
                    reports[None]["summary"][arm][str(k)],
                    {int(key): value for key, value in reports[rotation]["summary"][arm].items()},
                )
                for rotation in config["decomposition"]["rotations"]
            }
            for k in (5, 10, 25)
        }
        for arm in ("J", "R")
    }
    save_json(
        output,
        finite_json(
            {
                "status": "complete",
                "identity": identity,
                "source_report_sha256": sources,
                "quality_upload_receipt_sha256": file_sha256(args.quality_upload_receipt),
                "matching": matches,
                "interpretation": "Supplementary approximate calibration matching; retain all same-k primary controls. Raw calibration-token quality may not transfer to generated answers.",
            }
        ),
    )


def main():
    """Run one explicit capture or control phase using the frozen scientific config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "decompose", "match"))
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    parser.add_argument("--role", choices=("primary", "comparison"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main-readiness", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--dictionary-upload-receipt", type=Path)
    parser.add_argument("--capture-upload-receipt", type=Path)
    parser.add_argument("--quality-upload-receipt", type=Path)
    parser.add_argument("--rotation", type=int, choices=(20260913, 20260914, 20260915))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "decompose" and any(
        value is None
        for value in (args.input_root, args.dictionary_upload_receipt, args.capture_upload_receipt)
    ):
        raise ValueError(
            "Decomposition requires uploaded main dictionaries and calibration captures"
        )
    if args.phase == "match" and args.quality_upload_receipt is None:
        raise ValueError("Matching requires uploaded calibration quality reports")
    config = load_workspace_jr_config(args.config)
    identity = run_identity(args.config, args.selection, args.role)
    evidence = validate_main_readiness(
        args.main_readiness,
        config,
        config_path=args.config,
        selection_path=args.selection,
        identity=identity,
    )
    identity["execution_readiness_sha256"] = evidence["readiness_sha256"]
    {"capture": capture, "decompose": decompose, "match": match}[args.phase](
        args, config, identity, evidence
    )


if __name__ == "__main__":
    main()
