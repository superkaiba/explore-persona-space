"""Batch unchanged frozen-core fit outputs, with verification before completion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import issue2588_chat_grant as CG


def publish(core, args) -> dict:
    """Match the generic legacy output layout; replace only per-file transport."""
    assert args.surface == "generic" and not args.smoke and args.layer_set == "swept"
    cell = core.PC.cell_by_key(args.cell)
    paths = core._paths(args, cell)
    assert core._phase_complete(args, paths, "fits"), "upload-fits requires the complete sweep"
    fit_files = core._phase_artifacts(args, cell, paths, "fits")
    pilot = paths["fits"] / "fit_pilot.json"
    if pilot.exists():
        fit_files.append(pilot)
    prefix = core._cell_prefix(args, cell)
    out = paths["logs"] / f"issue-2588-{cell.key}-generic-{args.run_id}-results.json"
    receipt = paths["cell"] / "uploads/upload-fits.json"
    phase_done = core._phase_done_path(args, paths, "upload-fits")
    transport_done = paths["cell"] / "uploads/upload-fits-transport.json"
    evidence_files = [*fit_files, out, receipt, phase_done]
    if transport_done.exists():
        assert core._phase_complete(args, paths, "upload-fits")
        assert json.loads(transport_done.read_text()) == {
            "identity": core._identity(args, cell),
            "files": core._file_records(evidence_files, args.out_root),
        }, "changed fit transport evidence"
        return {"status": "already_complete"}
    # Same fields/paths as frozen phase_upload_fits, including the source's
    # metadata function and explicitly unresolved final GPU-hour fields.
    sentinel = {
        "eval_numbers": core._sentinel_numbers(args, cell, paths),
        "eval_paths": [str(p) for p in fit_files],
        "reproducibility_card": core._meta(),
        "wandb_url": None,
        "hf_hub_url": f"https://huggingface.co/datasets/{core.PC.HF_DATA_REPO}/tree/main/{prefix}/fits",
        "worktree_path": str(core._REPO_ROOT),
        "final_commit_sha": core.G._git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": None,
        "plan_deviations": [],
        "identity": core._identity(args, cell),
    }
    core.PC.write_json_atomic(out, sentinel)
    payload = [(p, f"fits/{p.name}") for p in fit_files] + [(out, "results.json")]
    uploaded = core._upload_generic_files(payload, prefix, staging_parent=paths["cell"].parent)
    core._record_verified_upload(args, cell, paths, "upload-fits", uploaded)
    core._mark_phase_done(args, cell, paths, "upload-fits")
    # The frozen marker is written before its remote upload. Only this local
    # retry record certifies that the final verified upload actually returned.
    # Preserve it with terminal diagnostics; it is not another Hub commit.
    core.PC.write_json_atomic(
        transport_done,
        {
            "identity": core._identity(args, cell),
            "files": core._file_records(evidence_files, args.out_root),
        },
    )
    return {"status": "complete", "files": len(payload)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cell", choices=("q3_8b_a", "q3_8b_b"), required=True)
    args = parser.parse_args(argv)
    science = CG.science_root(args.science_root)
    sys.path[:0] = [str(science / "scripts"), str(science / "src")]
    import issue2588_run_cell as core

    assert Path(core.__file__).resolve() == science / "scripts/issue2588_run_cell.py"
    core_args = core._build_parser().parse_args(
        [
            "--surface",
            "generic",
            "--run-id",
            "qwen3-chat-v3",
            "--cell",
            args.cell,
            "--phase",
            "upload-fits",
            "--device",
            "cuda",
            "--out-root",
            str(args.out_root),
        ]
    )
    print(publish(core, core_args), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
