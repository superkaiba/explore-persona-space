#!/usr/bin/env python3
"""Issue #779 inline free-analysis — multi-rollout reliability GEN + CAPTURE (GPU).

Regenerates K=10 rollouts (distinct seeds) for the FIRST N=600 of the ORIGINAL
pass-B LMSYS contexts and teacher-force-captures v(x) per rollout at all 28
layers, so the sibling 0-GPU analysis (issue779_reliability_analysis.py) can
compute a per-direction single-rollout RELIABILITY (ICC) — the noise ceiling for
the per-direction R2 curve. This CLOSES the round-1 gap: round-1 pass-B persisted
1 rollout/context and never kept the rollout TEXT, so the per-context sampling
noise along each answer-PCA direction was unmeasured.

REUSE, not reimplementation:
  * The 600 contexts are the [0, 600) slice of the round-1 pass-B [0, 5000) set,
    obtained by the SAME deterministic LMSYS re-stream the n10k script uses —
    ``issue779_ffc_n10k_generate_capture.sample_disjoint(skip=600, n_new=1)``
    returns ``manifest["old"]`` = the first 600 non-empty first-user-turns. The
    round-1 ctx0 prompt is asserted (same normalization + EXPECTED_CTX0_PROMPT as
    the n10k script) so a stream-ordering drift fails loud.
  * Generation recipe IDENTICAL to pass-B: vLLM, temperature 1.0, top_p 0.95,
    max_tokens 1024, standard chat template — K=10 rollouts, one per SEED in
    {SEED_BASE .. SEED_BASE+K-1} (SEED_BASE=42, so rollout 0 matches pass-B's
    seed-42 recipe). Each seed is a separate n=1 vLLM pass (distinct seeds).
  * v(x) capture via ``issue779_collect.capture_answer_vector`` (the round-1
    answer-side path: teacher-force context+response, mean over the response
    token span INCLUDING the trailing ``<|im_end|>`` + ``\\n`` — the same
    ``v_x = resp_stack.mean(0)`` convention as run_pass_b). r_b_by_trait={} so
    ONLY v_x is computed (no pooled projections). Model load + engine + position
    asserts reuse ``sample_disjoint`` / ``load_models`` from the n10k script.

PERSISTENCE (closes the text gap): all 6000 rollout TEXTS (prompt+completion)
are written to ``raw_completions.json`` the moment each seed-pass finishes,
BEFORE the capture reduce (#779 upload rule); the per-rollout v(x) tensors go to
``reliability_multirollout.pt``. Both upload to
``issue779_monitoring/reliability_multirollout/`` BEFORE ``[phase=done]``.
Checkpointed per seed-pass shard; ``--resume`` skips completed shards.

GPU (H100). NO judge/API calls. Fail loud — NaN never coerced, empty response
recorded as invalid (never a zero-filled fake rollout).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# vLLM V1 fork-safety (#628): spawn BEFORE any vllm import in the process.
import os  # noqa: E402

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_collect as COL  # noqa: E402
import issue779_common as C  # noqa: E402
import issue779_ffc_n10k_generate_capture as GC  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue779_reliability_gc")

N_LAYERS = C.EXPECTED_LAYERS  # 28
H_DIM = C.EXPECTED_HIDDEN  # 3584
DEFAULT_MODEL = GC.DEFAULT_MODEL  # Qwen/Qwen2.5-7B-Instruct (round-1 pass-B model)
HF_PREFIX = "issue779_monitoring/reliability_multirollout"
SEED_BASE = 42  # rollout k uses seed SEED_BASE + k; k=0 == pass-B's seed-42 recipe


def _first_n_prompts(n_ctx: int) -> dict:
    """First ``n_ctx`` round-1 pass-B prompts via the SAME deterministic re-stream
    the n10k script uses (skip=n_ctx re-derives them as ``manifest['old']``)."""
    manifest = GC.sample_disjoint(skip=n_ctx, n_new=1)
    prompts = manifest["old"]
    assert len(prompts) == n_ctx, (len(prompts), n_ctx)
    ctx0 = prompts[0]
    norm = " ".join(ctx0.lower().split()).rstrip(".?!,")
    assert norm == GC.EXPECTED_CTX0_PROMPT, (
        f"round-1 ctx0 re-derivation drift: got {ctx0[:120]!r}, expected "
        f"~{GC.EXPECTED_CTX0_PROMPT!r} — LMSYS stream ordering changed"
    )
    return {"prompts": prompts, "old_prompt_sha256": manifest["old_prompt_sha256"]}


def _generate_rollout(llm, tok, prompts: list[str], seed: int) -> list[str]:
    """1 rollout per prompt at the pass-B recipe with an explicit seed. CPU-smoke
    (llm is None) returns per-prompt stubs through the SAME capture path."""
    if llm is None:
        return [f"Stub smoke response {i} at seed {seed}." for i in range(len(prompts))]
    from vllm import SamplingParams

    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=seed)
    prompt_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    gen = COL._vllm_generate_chunked(llm, prompt_texts, sp)  # list[list[str]]
    return [g[0] for g in gen]


def _capture_pass(hf, tok, prompts, responses, layers) -> tuple[torch.Tensor, list[bool]]:
    """v(x) per (prompt, rollout) at all layers. Returns ((n_ctx, L, H) fp16 with
    zeros where invalid, valid mask). Empty response -> invalid, NEVER zero-faked
    as a real rollout (the analysis drops invalid; zeros are a sentinel only)."""
    v_rows, valid = [], []
    for p, resp in zip(prompts, responses, strict=True):
        msgs = [{"role": "user", "content": p}]
        av = COL.capture_answer_vector(hf, tok, msgs, resp, layers, {}, keep_per_token=False)
        if av is None:  # empty response -> v_x uncomputable
            v_rows.append(torch.zeros((len(layers), H_DIM), dtype=torch.float16))
            valid.append(False)
        else:
            v_rows.append(av["v_x"].to(torch.float16))
            valid.append(True)
    return torch.stack(v_rows), valid


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 multi-rollout reliability gen+capture.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-ctx", type=int, default=600)
    ap.add_argument("--k-rollouts", type=int, default=10)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_779" / "reliability_multirollout",
    )
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument("--gpu-mem-util", type=float, default=0.60)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_ctx = min(args.n_ctx, 8)
        args.k_rollouts = min(args.k_rollouts, 3)

    layers = list(range(N_LAYERS))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    raw_dir = args.out_dir / "raw_by_seed"
    raw_dir.mkdir(exist_ok=True)
    t0 = time.time()

    C.phase("sample")
    samp = _first_n_prompts(args.n_ctx)
    prompts = samp["prompts"]
    logger.info("sampled first %d round-1 pass-B prompts (ctx0 assert PASS)", len(prompts))

    C.phase("load_model")
    tok, hf = GC.load_models(args.model, args.device)
    llm = None
    if args.device == "cuda":
        from explore_persona_space.eval.generation import create_vllm_engine

        llm = create_vllm_engine(
            args.model,
            max_model_len=8192,
            seed=SEED_BASE,
            gpu_memory_utilization=args.gpu_mem_util,
        )

    C.phase("gen_capture")
    for k in range(args.k_rollouts):
        seed = SEED_BASE + k
        shard_path = shard_dir / f"rollout{k:02d}.pt"
        if args.resume and shard_path.exists():
            logger.info("[gen_capture] rollout %d exists; skip", k)
            continue
        ts = time.time()
        responses = _generate_rollout(llm, tok, prompts, seed)
        # PERSIST rollout TEXT the moment generation completes, BEFORE the capture
        # reduce (#779 upload rule) — a capture crash never loses the generations.
        C.write_json_atomic(
            raw_dir / f"rollout{k:02d}.json",
            {
                "rollout": k,
                "seed": seed,
                "rows": [
                    {"ci": ci, "prompt": prompts[ci], "response": responses[ci]}
                    for ci in range(len(prompts))
                ],
            },
        )
        v_x, valid = _capture_pass(hf, tok, prompts, responses, layers)
        torch.save(
            {"v_x": v_x, "valid": valid, "seed": seed, "rollout": k, "n_ctx": len(prompts)},
            shard_path,
        )
        logger.info(
            "[gen_capture] rollout %d (seed %d): %d/%d valid (%.0fs)",
            k,
            seed,
            sum(valid),
            len(prompts),
            time.time() - ts,
        )

    C.phase("assemble")
    shards = [
        torch.load(shard_dir / f"rollout{k:02d}.pt", weights_only=False, map_location="cpu")
        for k in range(args.k_rollouts)
    ]
    v_x = torch.stack([s["v_x"] for s in shards], dim=1)  # (n_ctx, K, L, H) fp16
    valid = torch.tensor([s["valid"] for s in shards], dtype=torch.bool).t()  # (n_ctx, K)
    assert v_x.shape == (len(prompts), args.k_rollouts, N_LAYERS, H_DIM), v_x.shape
    bundle = {
        "v_x": v_x,
        "valid": valid,
        "prompts": prompts,
        "seeds": [SEED_BASE + k for k in range(args.k_rollouts)],
        "layers": layers,
        "source": GC.LMSYS_REPO,
        "n_ctx": len(prompts),
        "k_rollouts": args.k_rollouts,
        "note": (
            "K rollouts x N first-600 pass-B contexts; v_x = mean-over-response-tokens "
            "(incl. im_end+\\n tail) at all 28 layers, teacher-forced. valid[ci,k]=False => "
            "empty response, v_x row is a zero sentinel (drop in analysis)."
        ),
        "metadata": C.reproducibility_metadata(
            {
                "script": "issue779_reliability_gen_capture",
                "old_prompt_sha256": samp["old_prompt_sha256"],
                "n_valid_total": int(valid.sum().item()),
            }
        ),
    }
    bundle_path = args.out_dir / "reliability_multirollout.pt"
    torch.save(bundle, bundle_path)
    # assemble all rollout texts into one raw_completions.json (per-seed files
    # were already written pre-capture; this is the consolidated view)
    all_rows = []
    for k in range(args.k_rollouts):
        rec = json.loads((raw_dir / f"rollout{k:02d}.json").read_text())
        for r in rec["rows"]:
            all_rows.append(
                {
                    "ci": r["ci"],
                    "rollout": k,
                    "seed": rec["seed"],
                    "prompt": r["prompt"],
                    "response": r["response"],
                }
            )
    C.write_json_atomic(args.out_dir / "raw_completions.json", {"rows": all_rows})
    logger.info(
        "wrote %s (%s) + raw_completions.json (%d rollout texts)",
        bundle_path,
        tuple(v_x.shape),
        len(all_rows),
    )

    if not args.no_upload:
        C.phase("upload")
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub

        # ONE upload_folder commit (the #779 pattern in issue779_collect); exclude
        # the redundant per-seed shards/raw_by_seed staging dirs — the assembled
        # reliability_multirollout.pt + raw_completions.json are the canonical
        # artifacts. Verify via the SCOPED + retried helper (#997) — a bare
        # list_repo_files on the ~1M-file data repo wedges (#833).
        api = HfApi()
        api.upload_folder(
            folder_path=str(args.out_dir),
            path_in_repo=args.hf_prefix,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            ignore_patterns=["shards/*", "raw_by_seed/*"],
            commit_message="issue779: multi-rollout reliability (v_x + rollout text)",
        )
        expected = [
            f"{args.hf_prefix}/{n}" for n in ("reliability_multirollout.pt", "raw_completions.json")
        ]
        missing = hub.verify_repo_paths_uploaded(
            api, C.HF_DATA_REPO, expected, path_in_repo=args.hf_prefix, repo_type="dataset"
        )
        if missing:
            raise RuntimeError(f"reliability upload verify FAILED — missing on HF: {missing}")
        logger.info("uploaded + verified %s -> %s", args.out_dir, args.hf_prefix)

    C.phase("done")
    logger.info("[timing] total %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
