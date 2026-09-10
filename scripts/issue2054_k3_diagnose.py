"""Discriminate K3 capture implementation, batching and precision differences."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k3 as k3
from scripts import issue2054_k3_artifacts as artifacts

CELLS = (
    "char_dana__on_policy__bare_label__qwen2.5-7b",
    "conversation_paired_stories_assistant__on_policy__chat__qwen2.5-7b",
    "char_dana__on_policy__bare_label__qwen2.5-7b-instruct",
    "char_vex__on_policy__bare_label__qwen2.5-7b-instruct",
)


def prepare(root, inputs):
    import numpy as np

    manifest = json.loads((inputs / "manifest.json").read_text())
    groups, references = [], []
    for record in manifest["cells"]:
        if record["cell"] not in CELLS:
            continue
        rows = k3.banked_rows(inputs, record, False)
        if k3.sha(inputs / "inputs" / record["activation"]) != record["activation_sha256"]:
            raise RuntimeError(f"banked activation checksum mismatch: {record['cell']}")
        with np.load(inputs / "inputs" / record["activation"], allow_pickle=False) as bank:
            order = {str(cid): i for i, cid in enumerate(bank["conv_id"])}
            for offset in range(0, len(rows), 256):
                batch = rows[offset : offset + 8]
                groups.append({"cell": record["cell"], "offset": offset, "rows": batch})
                references.append(bank["v_A"][[order[r["conv_id"]] for r in batch]])
    if len(groups) != 128:
        raise RuntimeError(f"expected all128 original parity batches; found{len(groups)}")
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "references.npz", v_A=np.stack(references))
    k3.atomic_json(root / "packet.json", {"models": manifest["models"], "groups": groups})
    for name in ("packet.json", "references.npz"):
        k3.seal(root / name, root, "capture-diagnosis-inputs-v1")


def relative(x, reference):
    import numpy as np

    x, reference = x.astype(np.float32), reference.astype(np.float32)
    return np.linalg.norm(x - reference, axis=-1) / np.linalg.norm(reference, axis=-1)


def comparison(x, reference):
    import numpy as np

    x, reference = x.astype(np.float32), reference.astype(np.float32)
    norm = np.linalg.norm(reference, axis=-1)
    return {
        "reference_norm": norm.tolist(),
        "absolute_l2": np.linalg.norm(x - reference, axis=-1).tolist(),
        "cosine": ((x * reference).sum(-1) / (np.linalg.norm(x, axis=-1) * norm)).tolist(),
    }


def observe_attention(callback):
    """Record actual SDPA dispatch inputs without changing its computation."""
    import torch.nn.functional as functional

    original = functional.scaled_dot_product_attention
    events = set()

    def observed(query, key, value, *args, **kwargs):
        mask = kwargs.get("attn_mask")
        events.add(
            (
                tuple(query.shape),
                tuple(key.shape),
                None if mask is None else tuple(mask.shape),
                bool(kwargs.get("enable_gqa", False)),
                bool(kwargs.get("is_causal", False)),
            )
        )
        return original(query, key, value, *args, **kwargs)

    functional.scaled_dot_product_attention = observed
    try:
        result = callback()
    finally:
        functional.scaled_dot_product_attention = original
    return result, sorted(events, key=str)


def parent_vectors(model, tokenizer, rows, batch_size, *, use_cache):
    import numpy as np
    import torch

    result, lower, upper = [], [], []
    hook_difference = 0.0
    for start in range(0, len(rows), batch_size):
        positions = [
            k3.capture._compute_positions(tokenizer, r) for r in rows[start : start + batch_size]
        ]
        if any(p is None for p in positions):
            raise RuntimeError("unresolvable diagnostic row")
        padded = tokenizer.pad(
            {"input_ids": [p["input_ids"] for p in positions]}, padding=True, return_tensors="pt"
        )
        observed = {}

        def hook(_module, _args, output):
            observed["hidden"] = output[0] if isinstance(output, tuple) else output

        handle = model.model.layers[18].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                output = model(
                    **{k: v.to("cuda") for k, v in padded.items()},
                    output_hidden_states=True,
                    use_cache=use_cache,
                    logits_to_keep=1,
                )
            hs = output.hidden_states[19]
            hook_difference = max(hook_difference, float((hs - observed["hidden"]).abs().max()))
            for i, p in enumerate(positions):
                for dest, layer in ((result, 19), (lower, 18), (upper, 20)):
                    dest.append(
                        output.hidden_states[layer][i, p["answer_lo"] : p["answer_hi"]]
                        .mean(0)
                        .float()
                        .cpu()
                        .numpy()
                    )
            del output, hs, padded
            observed.clear()
        finally:
            handle.remove()
    return np.stack(result), np.stack(lower), np.stack(upper), hook_difference


class Checkpoints:
    """Locally atomic units; verify bounded packets on the Hub before reuse."""

    def __init__(self, root, model_slug):
        self.root = root
        self.directory = root / "checkpoints" / model_slug
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "script": k3.sha(__file__),
                    "upload": k3.sha(artifacts.__file__),
                    "capture": k3.sha(k3.__file__),
                    "parent": k3.sha(k3.capture.__file__),
                    "packet": k3.sha(root / "packet.json"),
                    "references": k3.sha(root / "references.npz"),
                    "model": model_slug,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.pending = []

    def read(self, key):
        import numpy as np

        path = self.directory / f"{key}.npz"
        if not k3.complete(path, self.fingerprint):
            return None
        with np.load(path, allow_pickle=False) as saved:
            return saved["vector"].copy(), json.loads(str(saved["report"]))

    def write(self, key, vector, report):
        import numpy as np

        path = self.directory / f"{key}.npz"
        pending = path.with_suffix(".pending")
        with pending.open("wb") as stream:
            np.savez_compressed(stream, vector=vector, report=np.array(json.dumps(report)))
        pending.replace(path)
        self.pending.append(path)
        if len(self.pending) >= 8:
            self.flush()

    def flush(self):
        if self.pending:
            artifacts.seal_many(self.pending, self.root, self.fingerprint)
            self.pending.clear()


def diagnose(root, model_slug):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    packet = json.loads((root / "packet.json").read_text())
    with np.load(root / "references.npz") as bank:
        reference = bank["v_A"]
    spec = packet["models"][model_slug]
    checkpoints = Checkpoints(root, model_slug)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], use_fast=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            spec["id"],
            revision=spec["revision"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    reports, values, details = [], {}, []
    for i, group in enumerate(packet["groups"]):
        if not group["cell"].endswith("__" + model_slug):
            continue
        key = f"current_{i}"
        saved = checkpoints.read(key)
        if saved is None:
            (current, _), attention = observe_attention(
                lambda: k3.forward_vectors(model, tokenizer, group["rows"])
            )
            err = relative(current, reference[i])
            report = {
                "group": i,
                "cell": group["cell"],
                "offset": group["offset"],
                "relative_error": err.tolist(),
                "attention": attention,
                "comparison": comparison(current, reference[i]),
                "answer_tokens": [
                    k3.capture._compute_positions(tokenizer, r)["answer_hi"]
                    - k3.capture._compute_positions(tokenizer, r)["answer_lo"]
                    for r in group["rows"]
                ],
            }
            checkpoints.write(key, current, report)
        else:
            current, report = saved
            err = np.array(report["relative_error"])
        values[key] = current
        reports.append(report)
        k3.log(
            f"[phase=diagnose] {group['cell']} offset={group['offset']} current_relative_max={err.max():.6f}"
        )
    checkpoints.flush()
    selected = []
    for cell in sorted({r["cell"] for r in reports}):
        matching = [r for r in reports if r["cell"] == cell]
        selected.extend(
            r["group"]
            for r in sorted(matching, key=lambda r: max(r["relative_error"]))[:: len(matching) - 1]
        )
    for i in selected:
        rows = packet["groups"][i]["rows"]
        for name, batch_size, cache in (
            ("parent_b8_cache", 8, True),
            ("parent_b8_no_cache", 8, False),
            ("parent_b1_no_cache", 1, False),
        ):
            key = f"{name}_{i}"
            saved = checkpoints.read(key)
            if saved is not None:
                values[key], report = saved
                details.append(report)
                continue
            (array, lower, upper, diff), attention = observe_attention(
                lambda: parent_vectors(model, tokenizer, rows, batch_size, use_cache=cache)
            )
            values[f"{name}_{i}"] = array
            report = {
                "group": i,
                "method": name,
                "vs_banked": relative(array, reference[i]).tolist(),
                "vs_current": relative(array, values[f"current_{i}"]).tolist(),
                "layer18_vs_banked": relative(lower, reference[i]).tolist(),
                "layer20_vs_banked": relative(upper, reference[i]).tolist(),
                "hook_hidden_states_max_abs": diff,
                "attention": attention,
                "comparison": comparison(array, reference[i]),
            }
            if diff != 0:
                raise RuntimeError("hook and parent hidden_states disagree on identical forward")
            details.append(report)
            checkpoints.write(key, array, report)
            k3.log(
                f"[phase=controlled_test] group={i} method={name} max_vs_banked={max(report['vs_banked']):.6f}"
            )
    checkpoints.flush()
    k3.atomic_json(
        root / f"bf16_report_{model_slug}.json",
        {"current_batches": reports, "controlled_tests": details, "selected_groups": selected},
    )
    artifacts.seal_many([root / f"bf16_report_{model_slug}.json"], root, checkpoints.fingerprint)
    # Isolate arithmetic precision while preserving the same bf16-rounded weights.
    model.float()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for i in selected:
        for name, batch_size in (("fp32_b8", 8), ("fp32_b1", 1)):
            key = f"{name}_{i}"
            saved = checkpoints.read(key)
            if saved is not None:
                values[key], report = saved
                details.append(report)
                continue
            array, _, _, diff = parent_vectors(
                model, tokenizer, packet["groups"][i]["rows"], batch_size, use_cache=False
            )
            values[f"{name}_{i}"] = array
            report = {
                "group": i,
                "method": name,
                "vs_banked": relative(array, reference[i]).tolist(),
                "vs_current": relative(array, values[f"current_{i}"]).tolist(),
                "hook_hidden_states_max_abs": diff,
                "precision": "FP32 arithmetic on bf16-rounded weights; TF32 disabled",
            }
            if diff != 0:
                raise RuntimeError("FP32 hook and hidden_states disagree on identical forward")
            details.append(report)
            checkpoints.write(key, array, report)
            # These are the memory-intensive units; persist each before the next.
            checkpoints.flush()
            k3.log(
                f"[phase=controlled_test] group={i} method={name} max_vs_banked={max(report['vs_banked']):.6f}"
            )
    for i in selected:
        details.append(
            {
                "group": i,
                "method": "fp32_batch_invariance",
                "relative": relative(values[f"fp32_b8_{i}"], values[f"fp32_b1_{i}"]).tolist(),
            }
        )
    np.savez_compressed(root / f"vectors_{model_slug}.npz", **values)
    report = {
        "model": spec,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "current_batches": reports,
        "controlled_tests": details,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    k3.atomic_json(root / f"report_{model_slug}.json", report)
    artifacts.seal_many(
        [root / f"vectors_{model_slug}.npz", root / f"report_{model_slug}.json"],
        root,
        checkpoints.fingerprint,
    )
    k3.log(f"[phase=diagnostic_complete] model={model_slug}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "gpu", "model"), required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--source-sha")
    parser.add_argument("--input-revision")
    args = parser.parse_args()
    root = args.out_root
    if args.stage == "prepare":
        prepare(root, args.inputs)
    elif args.stage == "model":
        diagnose(root, args.model)
    else:
        from scripts.issue2054_k3_job import prepare_git
        from explore_persona_space.backends.artifacts import write_completion_sentinel

        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if actual != args.source_sha:
            raise RuntimeError("diagnostic source mismatch")
        prepare_git(REPO, actual)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "explore_persona_space.orchestrate.preflight",
                "--planned-footprint-gb",
                "60",
                "--min-disk",
                "100",
            ],
            check=True,
        )
        root.mkdir(parents=True, exist_ok=True)
        for name in ("packet.json", "references.npz"):
            p = k3.download(f"{k3.PREFIX}/{root.name}/{name}", root, args.input_revision)
            receipt = k3.download(
                f"{k3.PREFIX}/{root.name}/{name}.done.json", root, args.input_revision
            )
            if k3.sha(p) != json.loads(receipt.read_text())["sha256"]:
                raise RuntimeError("diagnostic input checksum mismatch")
            staged = root / name
            if staged.exists() or staged.is_symlink():
                if not staged.exists() or k3.sha(staged) != k3.sha(p):
                    raise RuntimeError(f"existing diagnostic input mismatch: {staged}")
            else:
                staged.symlink_to(p.resolve())
        for model_slug in k3.MODEL_REVISIONS:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--stage",
                    "model",
                    "--model",
                    model_slug,
                    "--out-root",
                    str(root),
                ],
                check=True,
            )
        report = {
            "status": "diagnostics_complete",
            "source_sha": actual,
            "finished": time.time(),
            "interpretation_pending": True,
        }
        k3.atomic_json(root / "diagnostics_complete.json", report)
        k3.seal(root / "diagnostics_complete.json", root, k3.sha(__file__))
        write_completion_sentinel(
            sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=2054, extra=report
        )
        k3.log("[phase=done] diagnostic comparisons persisted; interpretation pending")


if __name__ == "__main__":
    main()
