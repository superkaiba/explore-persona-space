#!/usr/bin/env python3
"""Stage and recapture the preregistered historical mapping compatibility sample."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    file_sha256,
    load_native,
    load_workspace_jr_config,
    run_identity,
    save_json,
    save_tensors,
    selected_prompts,
)
from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402

PRODUCER_SHA = "9b896ccc6b65e1d7322d3c7e67f6fe92883a0f0c"
REPO = "superkaiba1/explore-persona-space-data"


def frozen_helpers(out: Path) -> dict:
    """Reuse exact pure producer functions without importing unrelated workflows.

    Only explicitly named function ASTs from the audited Git commit are compiled;
    generated text is never executed. Save full source bytes for reproduction.
    """
    wanted = {
        "scripts/issue928_common.py": {"char_span_to_token_span"},
        "scripts/issue2588_panel_common.py": {"_strip_span", "build_capture_row_2588"},
    }
    namespace = {}
    for source, names in wanted.items():
        data = subprocess.run(
            ["git", "show", f"{PRODUCER_SHA}:{source}"], check=True, capture_output=True
        ).stdout
        destination = out / "producer_sources" / Path(source).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError("Historical source snapshot changed")
        destination.write_bytes(data)
        tree = ast.parse(data, filename=source)
        functions = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        if {node.name for node in functions} != names:
            raise ValueError("Audited historical capture helper missing")
        exec(compile(ast.Module(body=functions, type_ignores=[]), source, "exec"), namespace)
    return namespace


def download(filename: str, revision: str) -> Path:
    """Fetch one exact producer artifact with the repository transport retry policy."""
    return Path(
        retry_transient(
            lambda: hf_hub_download(REPO, filename, repo_type="dataset", revision=revision),
            what="workspace_jr_parity_download",
        )
    )


def parity_selection(args, config, audit):
    """Recompute the exact preregistered calibration-survivor prefix."""
    key = {"primary": "q35_27b", "comparison": "q35_4b"}[args.role]
    survivors = set(audit["realized_context_local_ids"][key]["train_10k"])
    chosen = [
        row
        for row in selected_prompts(args.selection, args.audit, "calibration")
        if row["ladder_local_id"] in survivors
    ][: config["provenance_gate"]["recapture_contexts"]]
    if len(chosen) != config["provenance_gate"]["recapture_contexts"]:
        raise ValueError("Insufficient historical calibration survivors")
    return chosen


def prepare(args, config):
    """Select calibration survivors, then stage their original raw text and x/y."""
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a fresh historical parity staging directory")
    args.out.mkdir(parents=True, exist_ok=True)
    audit = json.loads(args.audit.read_text())
    key = {"primary": "q35_27b", "comparison": "q35_4b"}[args.role]
    model = audit["models"][key]
    chosen = parity_selection(args, config, audit)
    row_ids = [f"train_10k_{row['ladder_local_id']}" for row in chosen]
    needed = set(row_ids)
    revision = config["provenance_gate"]["dataset_revision"]
    rows_path = download(model["splits"]["train_10k"]["source"], revision)
    rows_meta = json.loads(rows_path.read_text())
    if rows_meta["meta"]["git_sha"] != PRODUCER_SHA:
        raise ValueError("Historical capture producer changed")
    ordered = [row["row_id"] for row in rows_meta["rows"]]
    if not needed.issubset(ordered) or len(ordered) != len(set(ordered)):
        raise ValueError("Historical row manifest lacks unique selected rows")
    prefix = f"issue2588_capability_panel_cap_long/{key}/nothink/raw_completions/train_10k"
    entries = retry_transient(
        lambda: list(
            HfApi().list_repo_tree(
                REPO, path_in_repo=prefix, repo_type="dataset", revision=revision
            )
        ),
        what="workspace_jr_parity_list_raw",
    )
    sources = {model["splits"]["train_10k"]["source"]: file_sha256(rows_path)}
    raw = {}
    for entry in entries:
        if not Path(entry.path).name.startswith("chunk"):
            continue
        path = download(entry.path, revision)
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("rows"), list)
            or payload["meta"]["git_sha"] != PRODUCER_SHA
            or payload["stage"] != "train_10k"
        ):
            raise ValueError("Unexpected historical raw completion schema")
        for row in payload["rows"]:
            if row["row_id"] in needed:
                if row["row_id"] in raw:
                    raise ValueError("Duplicate selected historical completion")
                raw[row["row_id"]] = row
        sources[entry.path] = file_sha256(path)
    if set(raw) != needed:
        raise ValueError("Historical raw completions do not cover the selected sample")
    arrays = {}
    shard_indices = {ordered.index(row_id) // 500 for row_id in row_ids}
    for index in sorted(shard_indices):
        record = model["capture_shards"]["train_10k"][index]
        path = download(record["path"], revision)
        if path.stat().st_size != record["size"] or file_sha256(path) != record["lfs_sha256"]:
            raise ValueError("Historical capture bytes differ from audited artifact")
        with np.load(path, allow_pickle=False) as shard:
            for i, row_id in enumerate(shard["row_ids"].tolist()):
                if row_id in needed:
                    if row_id in arrays:
                        raise ValueError("Duplicate selected historical activation")
                    arrays[row_id] = {
                        name: torch.from_numpy(shard[name][i].copy())
                        for name in ("x_prompt_last", "y_ans")
                    }
        sources[record["path"]] = file_sha256(path)
    if set(arrays) != needed:
        raise ValueError("Historical activation shards do not cover selected rows")
    save_tensors(
        args.out / "historical_targets.pt",
        {
            name: torch.stack([arrays[row_id][name] for row_id in row_ids])
            for name in ("x_prompt_last", "y_ans")
        },
    )
    frozen_helpers(args.out)
    manifest = {
        "preparation_identity": run_identity(args.config, args.selection, args.role),
        "model_role": args.role,
        "config_sha256": file_sha256(args.config),
        "selection_sha256": file_sha256(args.selection),
        "audit_sha256": file_sha256(args.audit),
        "producer_sha": PRODUCER_SHA,
        "dataset_revision": revision,
        "sources": sources,
        "historical_targets_sha256": file_sha256(args.out / "historical_targets.pt"),
        "rows": [
            {
                "selection": context,
                "raw": raw[row_id],
                "capture_metadata": rows_meta["rows"][ordered.index(row_id)],
            }
            for context, row_id in zip(chosen, row_ids, strict=True)
        ],
        "status": "prepared_historical_calibration_only",
    }
    save_json(args.out / "inputs.json", manifest)
    print(f"parity_inputs prepared={len(chosen)} role={args.role}", flush=True)


@torch.no_grad()
def capture(args, config):
    """Use original completion tokenization/spans and compare both x and y."""
    manifest = json.loads((args.inputs / "inputs.json").read_text())
    if (
        manifest["model_role"] != args.role
        or manifest["config_sha256"] != file_sha256(args.config)
        or manifest["selection_sha256"] != file_sha256(args.selection)
        or manifest["audit_sha256"] != file_sha256(args.audit)
        or manifest["producer_sha"] != PRODUCER_SHA
        or manifest["historical_targets_sha256"]
        != file_sha256(args.inputs / "historical_targets.pt")
    ):
        raise ValueError("Historical parity input identity mismatch")
    expected = parity_selection(args, config, json.loads(args.audit.read_text()))
    if [row["selection"] for row in manifest["rows"]] != expected:
        raise ValueError("Historical parity inputs changed frozen calibration membership/order")
    if [row["raw"]["row_id"] for row in manifest["rows"]] != [
        f"train_10k_{row['ladder_local_id']}" for row in expected
    ]:
        raise ValueError("Historical raw completions do not match selected context IDs")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a fresh historical recapture output directory")
    helpers = frozen_helpers(args.out)
    model, tokenizer, text = load_native(
        config, args.role, device=args.device, dtype=torch.bfloat16
    )
    x, y, details = [], [], []
    spec = config["models"][args.role]
    for item in manifest["rows"]:
        raw = item["raw"]
        s, e = helpers["_strip_span"](raw["text"], 0, len(raw["text"]))
        row, reason = helpers["build_capture_row_2588"](
            tokenizer, {**raw, "ans_char_span": [s, e]}, positions_wanted=("prompt_last",)
        )
        if row is None:
            raise ValueError(f"Historical selected capture became invalid: {reason}")
        start, end = row["spans"]["ans"]
        if end - start != item["capture_metadata"]["n_ans_tokens"]:
            raise ValueError("Historical answer-token span drift")
        if row["positions"]["prompt_last"] != len(row["prompt_ids"]) - 1:
            raise ValueError("Historical mapping was not a final-context read")
        ids = torch.tensor([row["prompt_ids"] + row["comp_ids"]], device=args.device)
        observed = {}

        def hook(_module, _inputs, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            observed["x"] = h[0, row["positions"]["prompt_last"]].float().cpu()
            observed["y"] = h[0, start:end].float().mean(0).cpu()

        handle = text.layers[spec["source_layer"]].register_forward_hook(hook)
        try:
            text(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        finally:
            handle.remove()
        x.append(observed["x"])
        y.append(observed["y"])
        details.append(
            {
                "row_id": row["row_id"],
                "prompt_sha256": item["selection"]["prompt_sha256"],
                "answer_span": [start, end],
                "token_ids": ids[0].cpu().tolist(),
            }
        )
        print(f"historical_recapture contexts={len(x)}/{len(manifest['rows'])}", flush=True)
    current = {"x_prompt_last": torch.stack(x), "y_ans": torch.stack(y)}
    historical = torch.load(
        args.inputs / "historical_targets.pt", map_location="cpu", weights_only=True
    )
    metrics = {}
    gate = config["provenance_gate"]
    for name, values in current.items():
        a, b = values.double(), historical[name].double()
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Historical comparison contains invalid activation arrays")
        norm = b.norm()
        valid = (a.norm(dim=1) > 0) & (b.norm(dim=1) > 0)
        if not bool(valid.all()) or not norm:
            raise ValueError("Historical parity requires nonzero x/y rows")
        cosine = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))
        relative = ((a - b).norm() / norm).item()
        metrics[name] = {
            "relative_frobenius_error": relative,
            "row_cosine": cosine.tolist(),
            "passed": relative <= gate["maximum_relative_frobenius_error"]
            and cosine.min().item() >= gate["minimum_row_cosine"],
        }
    save_tensors(args.out / "recaptured.pt", current)
    save_json(
        args.out / "parity_report.json",
        {
            "identity": run_identity(args.config, args.selection, args.role),
            "input_manifest_sha256": file_sha256(args.inputs / "inputs.json"),
            "recaptured_sha256": file_sha256(args.out / "recaptured.pt"),
            "rows": details,
            "metrics": metrics,
            "status": "passed" if all(value["passed"] for value in metrics.values()) else "failed",
            "runtime_difference": "Historical BF16 batched capture versus native BF16 eager individual recapture; FP32 answer reduction in both.",
        },
    )
    if not all(value["passed"] for value in metrics.values()):
        raise RuntimeError("Historical recapture parity failed; evidence saved, frozen reuse blocked")


def main():
    """Separate cheap staging from native GPU execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "capture"))
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "capture" and args.inputs is None:
        parser.error("capture requires --inputs")
    {"prepare": prepare, "capture": capture}[args.phase](
        args, load_workspace_jr_config(args.config)
    )


if __name__ == "__main__":
    main()
