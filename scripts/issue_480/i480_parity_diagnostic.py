# research code uses Greek letters and - legitimately
"""Task #480 round-3 parity-FAIL diagnostic — PEFT-unmerged discriminator.

The comedian smoke failed the #534 adapter-application parity gate on the
TRAINED side (offline merged read -11.0243 vs recorded in-loop -8.9020;
base side never read — the gate aborted first). This script discriminates
between the two hypothesis families:

  A. MERGE bug (rsLoRA scaling / bf16 merge rounding / save-load roundtrip):
     PEFT-unmerged read ~= recorded -8.90 but merged read ~= -11.02.
  B. PROBE/SLOT/WEIGHTS drift: PEFT-unmerged read ALSO ~= -11.02. Then the
     base-side reading splits further: base ~= recorded -19.694 means the
     probe is faithful and the CHECKPOINT WEIGHTS differ from the in-loop
     state at step 20; base off -19.694 means probe construction drift.

Four readings, all through the band callback's OWN builder + slot stats
(`_build_parity_probe_callback` imported from i480_phase2b_logprob — the
exact code path the smoke executed):

  1. base            — bf16 Qwen2.5-7B-Instruct, no adapter
  2. peft_unmerged   — PeftModel.from_pretrained(base, adapter) (in-loop convention)
  3. merged_inproc   — merge_and_unload() of (2), read in the same process
  4. merged_dir      — the smoke's actual merged dir reloaded from disk

Plus: effective LoRA scaling introspection (rsLoRA r=32 alpha=64 → 64/sqrt(32) ~=
11.3137; a 2.0 reading means use_rslora was dropped on load) and tokenizer
asset digests (adapter dir vs merged dir).

Run ON THE POD (single H100):
    uv run python scripts/issue_480/i480_parity_diagnostic.py \
        --probe-config /workspace/runs/issue_480_inband/comedian_seed42/parity_probe_config.json \
        --adapter-dir /workspace/runs/issue_480_inband/comedian_seed42/graded_adapter \
        --merged-dir /workspace/runs/issue_480_inband/comedian_seed42/merged \
        --out-path /workspace/logs/issue-480-parity-diagnostic.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("issue_480.parity_diagnostic")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tokenizer_digests(d: Path) -> dict:
    out = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json"):
        p = d / name
        out[name] = _sha256(p) if p.exists() else None
    return out


def _probe_digest(cb) -> dict:
    """Deterministic digest of the probe batch (rows + slots), for cross-run comparison."""
    ids = cb.probe_input_ids.cpu().numpy().tobytes()
    pos = cb.probe_marker_positions.cpu().tolist()
    return {
        "input_ids_sha256": hashlib.sha256(ids).hexdigest(),
        "shape": list(cb.probe_input_ids.shape),
        "marker_positions": pos,
    }


def _read(cb, model, label: str) -> dict:
    """Per-row + mean teacher-forced log P(marker) via the callback's own read."""
    stats = cb._compute_marker_slot_stats(model)
    logp = stats["logp"]
    row = [round(float(v), 4) for v in logp.tolist()]
    mean = float(logp.mean().item())
    log.info("[reading=%s] mean logp = %.4f (n=%d)", label, mean, len(row))
    return {"mean_logp": mean, "per_row_logp": row}


def _lora_scaling(peft_model) -> dict:
    """Effective scaling on a representative LoRA layer (rsLoRA check)."""
    layer = peft_model.base_model.model.model.layers[0].self_attn.q_proj
    return {
        "q_proj_layer0_scaling": dict(layer.scaling),
        "lora_A_dtype": str(layer.lora_A["default"].weight.dtype),
        "expected_rslora_scaling": 64 / (32**0.5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--probe-config", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path, default=None)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from i480_phase2b_logprob import _build_parity_probe_callback

    from explore_persona_space.experiments.marker_implant_480 import IM_END_ID, MARKER_ID

    with open(args.probe_config) as f:
        probe_cfg = json.load(f)
    recorded_trained = float(probe_cfg["recorded_logp_trained"])
    recorded_base = float(probe_cfg["recorded_logp_base"])

    import peft as peft_pkg
    import transformers as tf_pkg

    result: dict = {
        "probe_config": probe_cfg,
        "adapter_dir": str(args.adapter_dir),
        "merged_dir": str(args.merged_dir) if args.merged_dir else None,
        "versions": {
            "peft": peft_pkg.__version__,
            "transformers": tf_pkg.__version__,
            "torch": torch.__version__,
        },
        "adapter_config": json.loads((args.adapter_dir / "adapter_config.json").read_text()),
        "tokenizer_digests": {
            "adapter_dir": _tokenizer_digests(args.adapter_dir),
            "merged_dir": _tokenizer_digests(args.merged_dir) if args.merged_dir else None,
        },
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "readings": {},
    }

    # Probe via the adapter-dir tokenizer (== training checkpoint tokenizer).
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir))
    cb, n_rows = _build_parity_probe_callback(probe_cfg, tokenizer, MARKER_ID, IM_END_ID)
    result["probe"] = {"n_rows": n_rows, **_probe_digest(cb)}

    # If the merged-dir tokenizer differs, build a second probe with it and
    # record whether the probe batch itself drifts (pure tokenizer effect).
    cb_merged_tok = None
    if args.merged_dir is not None:
        tok_m = AutoTokenizer.from_pretrained(str(args.merged_dir))
        cb_m, _ = _build_parity_probe_callback(probe_cfg, tok_m, MARKER_ID, IM_END_ID)
        dig_a, dig_m = _probe_digest(cb), _probe_digest(cb_m)
        result["probe_merged_tokenizer_identical"] = (
            dig_a["input_ids_sha256"] == dig_m["input_ids_sha256"]
        )
        if not result["probe_merged_tokenizer_identical"]:
            cb_merged_tok = cb_m
            result["probe_merged_tokenizer"] = dig_m
            log.warning("merged-dir tokenizer renders a DIFFERENT probe batch — tokenizer drift")

    # Reading 1: base (no adapter).
    log.info("loading base %s (bf16)", args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        token=os.environ.get("HF_TOKEN"),
    )
    result["readings"]["base"] = {
        **_read(cb, base, "base"),
        "recorded": recorded_base,
    }
    result["readings"]["base"]["diff_vs_recorded"] = (
        result["readings"]["base"]["mean_logp"] - recorded_base
    )

    # Reading 2: PEFT-unmerged (the in-loop training-callback convention).
    log.info("applying adapter UNMERGED via PeftModel.from_pretrained")
    peft_model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    result["lora_scaling"] = _lora_scaling(peft_model)
    log.info("lora scaling: %s", result["lora_scaling"])
    result["readings"]["peft_unmerged"] = {
        **_read(cb, peft_model, "peft_unmerged"),
        "recorded": recorded_trained,
    }
    result["readings"]["peft_unmerged"]["diff_vs_recorded"] = (
        result["readings"]["peft_unmerged"]["mean_logp"] - recorded_trained
    )

    # Reading 3: merge_and_unload in-process (merge math, no disk roundtrip).
    log.info("merge_and_unload() in-process")
    merged_inproc = peft_model.merge_and_unload()
    result["readings"]["merged_inproc"] = {
        **_read(cb, merged_inproc, "merged_inproc"),
        "recorded": recorded_trained,
    }
    result["readings"]["merged_inproc"]["diff_vs_recorded"] = (
        result["readings"]["merged_inproc"]["mean_logp"] - recorded_trained
    )

    del peft_model, merged_inproc, base
    gc.collect()
    torch.cuda.empty_cache()

    # Reading 4: the smoke's actual merged dir, reloaded from disk.
    if args.merged_dir is not None:
        log.info("loading merged dir %s (bf16)", args.merged_dir)
        merged_disk = AutoModelForCausalLM.from_pretrained(
            str(args.merged_dir),
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        result["readings"]["merged_dir"] = {
            **_read(cb, merged_disk, "merged_dir"),
            "recorded": recorded_trained,
        }
        result["readings"]["merged_dir"]["diff_vs_recorded"] = (
            result["readings"]["merged_dir"]["mean_logp"] - recorded_trained
        )
        if cb_merged_tok is not None:
            result["readings"]["merged_dir_merged_tokenizer"] = _read(
                cb_merged_tok, merged_disk, "merged_dir_merged_tokenizer"
            )
        del merged_disk
        gc.collect()
        torch.cuda.empty_cache()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("diagnostic written -> %s", args.out_path)

    # Human-readable verdict hint (the orchestrating agent interprets fully).
    r = result["readings"]
    log.info(
        "SUMMARY: base=%.4f (rec %.4f) | unmerged=%.4f (rec %.4f) | inproc=%.4f | disk=%s",
        r["base"]["mean_logp"],
        recorded_base,
        r["peft_unmerged"]["mean_logp"],
        recorded_trained,
        r["merged_inproc"]["mean_logp"],
        f"{r['merged_dir']['mean_logp']:.4f}" if "merged_dir" in r else "n/a",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
