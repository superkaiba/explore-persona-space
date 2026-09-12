#!/usr/bin/env python3
"""Diagnose historical recapture sensitivity to attention backend, never authorize reuse."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from explore_persona_space.analysis.workspace_runtime import (
    file_sha256,
    load_workspace_jr_config,
    save_json,
)


def main():
    """Reuse audited tokenization and parity arithmetic, explicitly label the changed backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("primary", "comparison"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("docs/exploratory_workspace_jr/mapping_provenance.json")
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Attention diagnostic requires a fresh output directory")
    source = Path("scripts/workspace_jr_parity.py")
    module_spec = importlib.util.spec_from_file_location("workspace_jr_parity", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    original_loader = module.load_native
    details = {
        "schema": "workspace-jr-attention-parity-diagnostic-v1",
        "script_sha256": file_sha256(Path(__file__)),
        "parity_script_sha256": file_sha256(source),
        "role": args.role,
        "backend": "sdpa",
        "purpose": "Diagnose historical versus native eager recapture; does not authorize frozen reuse or alter native lens construction.",
        "status": "starting",
    }

    def sdpa_loader(config, role, *, device, dtype):
        model, tokenizer, text = original_loader(config, role, device=device, dtype=dtype)
        model.set_attn_implementation("sdpa")
        implementations = {
            str(layer.self_attn.config._attn_implementation)
            for layer in text.layers
            if hasattr(layer, "self_attn")
        }
        if text.config._attn_implementation != "sdpa" or implementations != {"sdpa"}:
            raise ValueError("Native full-attention layers did not switch to SDPA")
        details["actual_full_attention_implementations"] = sorted(implementations)
        return model, tokenizer, text

    module.load_native = sdpa_loader
    original_save = module.save_json

    def diagnostic_save(path, report):
        if Path(path).name == "parity_report.json":
            report = {
                **report,
                "status": f"diagnostic_sdpa_{report['status']}",
                "runtime_difference": "Historical BF16 batched default attention versus native BF16 individual SDPA recapture; this is not the eager native reuse gate.",
                "diagnostic_script_sha256": details["script_sha256"],
            }
        original_save(path, report)

    # Label the first atomic report write, including if this process is killed
    # before its outer finally runs. No ordinary gate-passed artifact can appear.
    module.save_json = diagnostic_save
    args.out.mkdir(parents=True, exist_ok=False)
    # The original capture refuses a nonempty directory, so persist this receipt
    # only after its call; its own per-context progress and arrays remain intact.
    try:
        module.capture(args, load_workspace_jr_config(args.config))
        details["status"] = "diagnostic_passed_requires_interpretation"
    except Exception as error:
        details.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(args.out / "diagnostic.json", details)


if __name__ == "__main__":
    main()
