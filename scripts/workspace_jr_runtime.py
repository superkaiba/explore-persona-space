#!/usr/bin/env python3
"""Runtime phases for the exploratory J/R workspace predictability experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    calibration_tokens,
    content_sha256,
    file_sha256,
    fit_lens_prompts,
    load_native,
    load_workspace_jr_config,
    native_forward_validation,
    preflight_status,
    run_identity,
    save_json,
    selected_prompts,
    validate_token_manifest,
)


def _dtype(name: str) -> torch.dtype:
    """Resolve a small explicit dtype vocabulary for native-model phases."""
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}")


def phase_preflight(args: argparse.Namespace) -> None:
    """Write the local readiness report without loading model weights."""
    report = preflight_status(
        args.config,
        args.selection,
        args.audit,
        role=args.role,
        output_path=args.out,
        minimum_local_free_gib=args.minimum_local_free_gib,
    )
    print(f"preflight_ready={report['local_real_model_execution_ready']} out={args.out}")
    for blocker in report["blockers"]:
        print(f"blocker={blocker}")


def phase_calibration_manifest(args: argparse.Namespace) -> None:
    """Tokenize frozen calibration prompts and persist exact token IDs."""
    from transformers import AutoTokenizer

    config = load_workspace_jr_config(args.config)
    spec = config["models"][args.role]
    tokenizer = AutoTokenizer.from_pretrained(
        config["selection"][args.role],
        revision=spec["revision"],
        trust_remote_code=True,
    )
    prompts = selected_prompts(args.selection, args.audit, args.subset)
    manifest = calibration_tokens(tokenizer, prompts, config)
    manifest["identity"] = run_identity(args.config, args.selection, args.role)
    manifest["subset"] = args.subset
    manifest["model_id"] = config["selection"][args.role]
    manifest["model_revision"] = spec["revision"]
    save_json(args.out, manifest)
    print(
        f"calibration_rows={len(manifest['rows'])} excluded={len(manifest['excluded'])} out={args.out}"
    )


def phase_native_forward_validation(args: argparse.Namespace) -> None:
    """Load one native checkpoint and run the forward/R-rule parity gate."""
    config = load_workspace_jr_config(args.config)
    manifest = json.loads(args.token_manifest.read_text())
    token_check = validate_token_manifest(
        manifest,
        config_path=args.config,
        selection_path=args.selection,
        config=config,
        role=args.role,
    )
    if not manifest["rows"]:
        raise ValueError("Token manifest has no valid rows")
    model, _tokenizer, text = load_native(
        config, args.role, device=args.device, dtype=_dtype(args.dtype)
    )
    row = manifest["rows"][args.row_index]
    report = native_forward_validation(model, text, row["token_ids"], config["models"][args.role])
    report["identity"] = run_identity(args.config, args.selection, args.role)
    report["token_manifest_identity"] = token_check
    report["prompt_sha256"] = row["prompt_sha256"]
    report["row_index"] = args.row_index
    report["model_dtype"] = args.dtype
    report["token_manifest_sha256"] = content_sha256(manifest)
    save_json(args.out, report)
    print(f"native_forward_validation_passed out={args.out}")


def phase_fit_lens_shard(args: argparse.Namespace) -> None:
    """Fit resumable per-prompt J/R lens matrices for a calibration shard."""
    config = load_workspace_jr_config(args.config)
    manifest = json.loads(args.token_manifest.read_text())
    token_check = validate_token_manifest(
        manifest,
        config_path=args.config,
        selection_path=args.selection,
        config=config,
        role=args.role,
    )
    validation = json.loads(args.validation.read_text())
    required_identity = run_identity(args.config, args.selection, args.role)
    for field in ("config_sha256", "selection_sha256", "model_role", "versions"):
        if validation["identity"][field] != required_identity[field]:
            raise ValueError(f"Native validation identity mismatch: {field}")
    if validation["ordinary_numerical_validation"]["status"] != "passed":
        raise ValueError("Ordinary-J numerical validation is required before fitting")
    if not validation["forward_bit_identical"] or not validation["hook_bit_identical"]:
        raise ValueError("R forward/hook parity is required before fitting")
    if validation["model_dtype"] != args.dtype:
        raise ValueError("Native validation dtype differs from requested lens dtype")
    if validation["token_manifest_sha256"] != content_sha256(manifest):
        raise ValueError("Native validation used a different calibration token manifest")
    model, tokenizer, _text = load_native(
        config, args.role, device=args.device, dtype=_dtype(args.dtype)
    )
    identity = {
        **run_identity(args.config, args.selection, args.role),
        "token_manifest": token_check,
        "native_validation_sha256": file_sha256(args.validation),
    }
    fit_lens_prompts(
        model,
        tokenizer,
        config,
        args.role,
        manifest,
        args.out_dir,
        identity,
        dim_batch=args.dim_batch,
        start=args.start,
        stop=args.stop,
    )
    print(f"fit_lens_shard_complete out_dir={args.out_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the phase CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis/workspace_jr.yaml"),
        help="Frozen workspace J/R YAML config.",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
        help="Outcome-blind selected context manifest.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/mapping_provenance.json"),
        help="Mapping/source provenance audit JSON.",
    )
    parser.add_argument("--role", choices=("primary", "comparison"), default="comparison")
    sub = parser.add_subparsers(dest="phase", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument(
        "--out",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/runtime_preflight.json"),
    )
    preflight.add_argument("--minimum-local-free-gib", type=float, default=80.0)
    preflight.set_defaults(func=phase_preflight)

    calibration = sub.add_parser("calibration-manifest")
    calibration.add_argument("--subset", default="calibration")
    calibration.add_argument(
        "--out",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/calibration_tokens_comparison.json"),
    )
    calibration.set_defaults(func=phase_calibration_manifest)

    validation = sub.add_parser("native-forward-validation")
    validation.add_argument("--token-manifest", type=Path, required=True)
    validation.add_argument("--out", type=Path, required=True)
    validation.add_argument("--device", default="cuda:0")
    validation.add_argument("--dtype", default="bfloat16")
    validation.add_argument("--row-index", type=int, default=0)
    validation.set_defaults(func=phase_native_forward_validation)

    fit = sub.add_parser("fit-lens-shard")
    fit.add_argument("--token-manifest", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--out-dir", type=Path, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--dtype", default="bfloat16")
    fit.add_argument("--dim-batch", type=int, default=8)
    fit.add_argument("--start", type=int, default=0)
    fit.add_argument("--stop", type=int)
    fit.set_defaults(func=phase_fit_lens_shard)
    return parser


def main() -> None:
    """Dispatch one explicit runtime phase."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
