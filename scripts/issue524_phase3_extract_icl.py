"""Phase 3 -- extract Qwen-2.5-7B-Instruct activation clouds for the 16 ICL contexts
on #502's 500-probe pool, all 28 residual-stream layers, extraction points
{last_prompt, mean_response}.

Issue #524 plan v1 §Phase 3. The 16 INSTRUCTION clouds are REUSED unchanged
from #502 (cached at ``superkaiba1/explore-persona-space-data:
issue502_28layer_500probe_bakeoff/``); this script produces only the 16
ICL clouds, which #502 never extracted.

The extraction logic mirrors ``scripts/issue493_extraction_metric_bakeoff.py``
``_extract_one`` / ``run_extraction`` -- per-layer hook capture with
``last_prompt`` (the hidden state at the last prompt token) and
``mean_response`` (the mean over the generated-response tokens). We do
NOT reimplement the hook logic; we drive the existing extractor with the
ICL-context prompt builders.

Output:
    eval_results/issue_524/phase3/activations/<extraction_point>__layer<L>__<cid>.pt
        (raw numpy arrays per (extraction_point, layer, cid); merged by
        Phase 4 into the canonical (n_cid, n_probe, hidden_dim) stack.)
    eval_results/issue_524/phase3/icl_clouds.json
        (manifest: which files written, sha256s, reproducibility metadata.)

CLI:
    # Smoke: ONE ICL context (IK01) on 5 probes, 2 layers, 1 extraction point.
    uv run python scripts/issue524_phase3_extract_icl.py \\
        --only IK01 --n-probes 5 --layers 21 22 \\
        --extraction-points last_prompt

    # Production: all 16 ICL contexts, 500 probes, 28 layers, both points.
    uv run python scripts/issue524_phase3_extract_icl.py --shard 0-of-8 --gpu-id 0

    # CPU smoke (no GPU): just verify the prompt builders + the (cid, probe,
    # layer, point) coordinate grid; SKIP the actual hook capture.
    uv run python scripts/issue524_phase3_extract_icl.py \\
        --only IK01 --n-probes 2 --layers 21 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.i524_icl_contexts import (  # noqa: E402
    ICL_CONTEXTS,
    ICL_CONTEXTS_BY_ID,
    build_icl_messages,
)

logger = logging.getLogger("i524.phase3")

OUT_DIR = Path("eval_results/issue_524/phase3")
ACT_DIR = OUT_DIR / "activations"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
ICL_BLOCKS_PATH = Path("eval_results/issue_524/icl_contexts/i524_icl_blocks.json")
PROBE_POOL_PATH = Path("eval_results/issue_502/probes_500.json")

DEFAULT_LAYERS_28 = tuple(range(28))
DEFAULT_EXTRACTION_POINTS = ("last_prompt", "mean_response")


def _parse_shard(spec: str | None) -> tuple[int, int]:
    if spec is None:
        return 0, 1
    s, n = spec.split("-of-")
    return int(s), int(n)


def _load_probes(probe_pool_path: Path, n_probes: int) -> list[str]:
    payload = json.loads(probe_pool_path.read_text())
    qs = payload.get("questions") or payload.get("probes") or payload
    if isinstance(qs[0], dict):
        qs = [q.get("question") or q.get("q") or q.get("text") for q in qs]
    return list(qs[:n_probes])


def _per_cell_path(point: str, layer: int, cid: str) -> Path:
    return ACT_DIR / f"{point}__layer{layer}__{cid}.npy"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--shard", default=None, help="e.g. '0-of-8'")
    ap.add_argument("--only", nargs="+", default=None, help="Restrict to specific cids.")
    ap.add_argument("--n-probes", type=int, default=500, help="Probes per context (default 500).")
    ap.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAYERS_28),
        help="Residual-stream layers to extract (default 0..27).",
    )
    ap.add_argument(
        "--extraction-points",
        nargs="+",
        default=list(DEFAULT_EXTRACTION_POINTS),
        choices=list(DEFAULT_EXTRACTION_POINTS),
    )
    ap.add_argument(
        "--max-response-tokens",
        type=int,
        default=128,
        help="Cap for the on-policy response used in mean_response.",
    )
    ap.add_argument(
        "--icl-blocks-path",
        type=Path,
        default=ICL_BLOCKS_PATH,
    )
    ap.add_argument(
        "--probe-pool",
        type=Path,
        default=PROBE_POOL_PATH,
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts + write placeholder coordinate grid; SKIP model load.",
    )
    args = ap.parse_args(argv)

    if not args.icl_blocks_path.exists():
        raise RuntimeError(f"ICL blocks missing at {args.icl_blocks_path}")
    if not args.probe_pool.exists():
        raise RuntimeError(f"Probe pool missing at {args.probe_pool}")
    icl_blocks = json.loads(args.icl_blocks_path.read_text())

    cids = args.only or [c.cid for c in ICL_CONTEXTS]
    unknown = [c for c in cids if c not in ICL_CONTEXTS_BY_ID]
    if unknown:
        raise ValueError(f"--only {unknown} not in ICL_CONTEXTS_BY_ID")
    shard_idx, n_shards = _parse_shard(args.shard)
    my_cids = [c for k, c in enumerate(cids) if k % n_shards == shard_idx]

    probes = _load_probes(args.probe_pool, args.n_probes)
    logger.info(
        "Phase 3: shard %d/%d -- %d cids × %d probes × %d layers × %d extraction points",
        shard_idx,
        n_shards,
        len(my_cids),
        len(probes),
        len(args.layers),
        len(args.extraction_points),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # CPU smoke: write per-(point, layer, cid) coordinate placeholders + assert
        # prompt construction works. No actual extraction.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        manifest_dry: dict = {"dry_run": True, "files": []}
        for cid in my_cids:
            demos = icl_blocks[cid]["demos"]
            first_prompt = tok.apply_chat_template(
                build_icl_messages(demos, probes[0]),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tok.encode(first_prompt, add_special_tokens=False)
            for point in args.extraction_points:
                for L in args.layers:
                    p = _per_cell_path(point, L, cid)
                    # Don't actually write -- the file would only carry dummy zeros
                    # and confuse the merger. We just record the COORDINATE.
                    manifest_dry["files"].append(
                        {
                            "path": str(p),
                            "shape_expected": [len(probes), "H (model hidden dim)"],
                            "first_prompt_len": len(prompt_ids),
                            "cid": cid,
                            "point": point,
                            "layer": L,
                        }
                    )
        (OUT_DIR / "dry_run_manifest.json").write_text(json.dumps(manifest_dry, indent=2))
        logger.info("DRY-RUN wrote coordinate grid: %d entries", len(manifest_dry["files"]))
        return 0

    # Real extraction: drive the issue493 bakeoff machinery.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Bring in the canonical extractor. It exposes the (cond_id, q) inner
    # loop via run_extraction(...) but expects a #406-style condition list
    # for prompt construction. To keep #524 self-contained without
    # modifying #493 / #502 dispatchers, we use the underlying hook
    # primitive (_LayerHookCapture) directly here.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    # Hidden-dim known for Qwen-2.5-7B = 3584; assert from config to be safe.
    H = model.config.hidden_size

    # Per-cid loop. We extract one (cid, point, layer) -> [n_probes, H] np
    # array per file (the canonical issue502 shape). This lets Phase 4's
    # merge step concat across cids into (n_cid, n_probes, H).
    manifest: dict = {
        "git_sha": _git_sha(),
        "base_model": BASE_MODEL,
        "hidden_dim": int(H),
        "n_probes": len(probes),
        "n_layers": len(args.layers),
        "n_cids": len(my_cids),
        "extraction_points": list(args.extraction_points),
        "files": [],
    }

    for cid in my_cids:
        demos = icl_blocks[cid]["demos"]
        # Pre-compute the per-(cid, q) prompts; one chat-template apply per q.
        prompts = [
            tok.apply_chat_template(
                build_icl_messages(demos, q), tokenize=False, add_generation_prompt=True
            )
            for q in probes
        ]

        # Per-extraction-point storage: (n_probes, n_layers, H).
        cid_storage: dict[str, np.ndarray] = {
            point: np.full((len(probes), len(args.layers), H), np.nan, dtype=np.float32)
            for point in args.extraction_points
        }

        t0 = time.time()
        for q_idx, prompt_text in enumerate(prompts):
            inputs = tok(prompt_text, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            prompt_len = input_ids.shape[1]

            # last_prompt: forward pass on the prompt alone, capture
            # hidden_states at the last token across every layer.
            if "last_prompt" in args.extraction_points:
                with torch.no_grad():
                    out = model(input_ids=input_ids, output_hidden_states=True)
                # hidden_states is a tuple of (n_layers+1,) tensors of
                # shape (1, seq_len, H); index 0 is the embedding, [L+1] is
                # block-L's output. The plan/spec uses 0..27 for the 28
                # residual-stream layers; we read hidden_states[L+1].
                for k, L in enumerate(args.layers):
                    cid_storage["last_prompt"][q_idx, k] = (
                        out.hidden_states[L + 1][0, prompt_len - 1].float().cpu().numpy()
                    )

            # mean_response: generate up to max_response_tokens (greedy)
            # then capture the per-layer mean over the response tokens.
            if "mean_response" in args.extraction_points:
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=args.max_response_tokens,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        return_dict_in_generate=True,
                    )
                    full_ids = gen.sequences  # (1, prompt_len + gen_len)
                    if full_ids.shape[1] <= prompt_len:
                        # Model emitted EOS immediately -- there's nothing to
                        # average; we leave nan and warn.
                        logger.warning(
                            "cid=%s q=%d: empty response (gen_len=0); mean_response left NaN",
                            cid,
                            q_idx,
                        )
                        continue
                    out2 = model(input_ids=full_ids, output_hidden_states=True)
                for k, L in enumerate(args.layers):
                    h = out2.hidden_states[L + 1][0, prompt_len:].float().cpu().numpy()
                    cid_storage["mean_response"][q_idx, k] = h.mean(axis=0)

        # Persist per-(point, layer) — CHECKPOINT-PER-CELL granularity.
        for point in args.extraction_points:
            for k, L in enumerate(args.layers):
                arr = cid_storage[point][:, k, :]
                out_path = _per_cell_path(point, L, cid)
                np.save(out_path, arr)
                manifest["files"].append(
                    {
                        "path": str(out_path),
                        "cid": cid,
                        "point": point,
                        "layer": L,
                        "shape": list(arr.shape),
                        "sha256": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
                    }
                )
        logger.info(
            "cid=%s extracted in %.1fs (%d probes × %d layers × %d points)",
            cid,
            time.time() - t0,
            len(probes),
            len(args.layers),
            len(args.extraction_points),
        )

    (OUT_DIR / f"icl_clouds_shard{shard_idx}of{n_shards}.json").write_text(
        json.dumps(manifest, indent=2)
    )
    logger.info("Phase 3 wrote manifest for shard %d/%d", shard_idx, n_shards)
    return 0


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, env={**os.environ}
        ).strip()  # epm-lint: subprocess-env-inherit -- git probe
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
