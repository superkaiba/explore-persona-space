#!/usr/bin/env python3
"""Issue #779 inline free-analysis — PER-TOKEN LMSYS answer-activation capture (GPU).

Re-captures the FULL per-token answer activation stack (layers {14, 19}) for the
FIRST N=150 of the ORIGINAL pass-B LMSYS contexts (the ``[0, 150)`` slice of the
round-1 pass-B ``[0, 5000)`` set), so the sibling 0-GPU analysis
(``issue779_pertoken_lmsys_analysis.py``) can test the "Concepts Whisper"
(arXiv 2605.01609) reconciliation on OUR OWN generic-LMSYS substrate:

  Concepts Whisper finds concept directions in the LOW-variance tail of the
  PER-TOKEN residual covariance; our writeup finds the persona vectors r_B are
  HIGH-variance (rank 2-10) in the MEAN-answer covariance. Hypothesis: mean-
  pooling over answer tokens averages away the token-IDENTITY variance that
  dominates the per-token spectrum, promoting the concept direction into the
  high-variance regime. Round 1 (``issue779_pertoken_vs_mean_variance.py``)
  tested this on the r3_subset EVAL-rig evil stacks; this closes the gap by
  testing it on the generic pass-B TRAIN contexts (LMSYS), all three traits.

REUSE, not reimplementation:
  * The 150 contexts are the ``[0, 150)`` slice of the round-1 pass-B ``[0, 5000)``
    set, obtained by the SAME deterministic LMSYS re-stream the n10k + reliability
    scripts use — ``issue779_ffc_n10k_generate_capture.sample_disjoint(skip=150,
    n_new=1)`` returns ``manifest["old"]`` = the first 150 non-empty first-user-
    turns. The round-1 ctx0 prompt is asserted (same normalization +
    EXPECTED_CTX0_PROMPT as the n10k script) so a stream-ordering drift fails loud.
  * Generation recipe IDENTICAL to pass-B: vLLM, temperature 1.0, top_p 0.95,
    max_tokens 1024, seed 42, standard chat template, DEFAULT system prompt
    (no system message) — 1 rollout per context.
  * Per-token capture via ``issue779_collect.capture_answer_vector(...,
    keep_per_token=True)`` — the round-1 answer-side path; its ``per_token`` key
    is the ``(n_resp, L, H)`` teacher-forced response-token activation stack
    (response span INCLUDING the trailing ``<|im_end|>`` + ``\\n``, the same span
    ``v_x = per_token.mean(0)`` reduces). Model load + engine asserts reuse
    ``load_models`` / ``create_vllm_engine`` from the round-1 scripts.

PERSISTENCE (closes the text gap, #779 upload rule): all 150 rollout TEXTS
(prompt+response) are written to ``raw_completions.json`` the moment generation
completes, BEFORE the capture reduce — a capture crash never loses the
generations. The per-context per-token stacks go to one ``.pt`` per context under
``contexts/`` (so the sibling analysis streams them one at a time, download ->
reduce -> delete, memory-bounded). Both upload to
``issue779_monitoring/pertoken_lmsys/`` in ONE bulk ``upload_folder`` commit
BEFORE ``[phase=done]``, then an EXACT-set verify. Checkpointed per context;
``--resume`` skips completed context files.

GPU (H100). NO judge/API calls. Fail loud — an empty response is recorded as
invalid (never a zero-filled fake stack). ``--device cpu --tiny-model`` runs the
IDENTICAL capture -> write -> verify path on a tiny throwaway model at layers
{0, 1} for the CPU smoke (no GPU, no 30 GB fp32 load).
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
logger = logging.getLogger("issue779_pertoken_lmsys_capture")

N_LAYERS = C.EXPECTED_LAYERS  # 28
H_DIM = C.EXPECTED_HIDDEN  # 3584
DEFAULT_MODEL = GC.DEFAULT_MODEL  # Qwen/Qwen2.5-7B-Instruct (round-1 pass-B model)
HF_PREFIX = "issue779_monitoring/pertoken_lmsys"
CAPTURE_LAYERS = [14, 19]  # 14 = a read-out layer, 19 = recon-best (brief)
SEED = 42  # pass-B recipe seed


def _first_n_prompts(n_ctx: int) -> dict:
    """First ``n_ctx`` round-1 pass-B prompts via the SAME deterministic re-stream
    the n10k / reliability scripts use (skip=n_ctx re-derives them as
    ``manifest['old']``)."""
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


def _generate_rollout(llm, tok, prompts: list[str]) -> list[str]:
    """1 rollout per prompt at the pass-B recipe (seed 42, default system prompt).
    CPU-smoke (llm is None) returns per-prompt stubs through the SAME capture path."""
    if llm is None:
        return [f"Stub smoke response {i} for the per-token capture." for i in range(len(prompts))]
    from vllm import SamplingParams

    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=SEED)
    prompt_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    gen = COL._vllm_generate_chunked(llm, prompt_texts, sp)  # list[list[str]]
    return [g[0] for g in gen]


def _tiny_cpu_model(tok):
    """A tiny throwaway Qwen2 CausalLM (random weights) over the REAL vocab id
    space, for the CPU smoke ONLY — exercises the real capture/write/verify path
    without a 30 GB fp32 load or a GPU. NEVER used on the ``--device cuda`` path."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=152064,  # Qwen2.5-7B vocab (incl. special tokens)
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 per-token LMSYS capture.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-ctx", type=int, default=150)
    ap.add_argument("--layers", type=int, nargs="+", default=CAPTURE_LAYERS)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument(
        "--tiny-model",
        action="store_true",
        help="CPU-smoke only: build a tiny throwaway Qwen2 (random weights). "
        "Never valid on --device cuda.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_779" / "pertoken_lmsys",
    )
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument("--gpu-mem-util", type=float, default=0.60)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.tiny_model and args.device != "cpu":
        raise ValueError("--tiny-model is CPU-smoke only; drop it for --device cuda")
    if args.smoke:
        args.n_ctx = min(args.n_ctx, 2)

    layers = args.layers
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ctx_dir = args.out_dir / "contexts"
    ctx_dir.mkdir(exist_ok=True)
    raw_path = args.out_dir / "raw_completions.json"
    t0 = time.time()

    C.phase("sample")
    samp = _first_n_prompts(args.n_ctx)
    prompts = samp["prompts"]
    logger.info("sampled first %d round-1 pass-B prompts (ctx0 assert PASS)", len(prompts))

    C.phase("load_model")
    if args.tiny_model:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        hf = _tiny_cpu_model(tok)
        llm = None
        logger.info("CPU smoke: tiny throwaway Qwen2 (random weights), layers=%s", layers)
    else:
        tok, hf = GC.load_models(args.model, args.device)
        llm = None
        if args.device == "cuda":
            from explore_persona_space.eval.generation import create_vllm_engine

            llm = create_vllm_engine(
                args.model,
                max_model_len=8192,
                seed=SEED,
                gpu_memory_utilization=args.gpu_mem_util,
            )

    # Generate (load-or-generate under --resume so a crash never loses text nor
    # re-runs vLLM). Prompts always come from the deterministic sampling (ctx0
    # assert above is the drift guard); responses are loaded from the persisted
    # raw_completions.json on resume.
    C.phase("gen")
    if args.resume and raw_path.exists():
        saved = json.loads(raw_path.read_text())["rows"]
        assert len(saved) == len(prompts), (len(saved), len(prompts))
        responses = [r["response"] for r in saved]
        logger.info("[gen] resume: loaded %d responses from raw_completions.json", len(responses))
    else:
        responses = _generate_rollout(llm, tok, prompts)
        # PERSIST rollout TEXT the moment generation completes, BEFORE the capture
        # reduce (#779 upload rule) — a capture crash never loses the generations.
        C.write_json_atomic(
            raw_path,
            {
                "rows": [
                    {"ci": ci, "prompt": prompts[ci], "response": responses[ci]}
                    for ci in range(len(prompts))
                ],
                "source": GC.LMSYS_REPO,
                "recipe": "pass-B (temp 1.0, top_p 0.95, seed 42, max 1024, default system)",
                "metadata": C.reproducibility_metadata(
                    {
                        "script": "issue779_pertoken_lmsys_capture",
                        "old_prompt_sha256": samp["old_prompt_sha256"],
                    }
                ),
            },
        )
        logger.info("[gen] wrote raw_completions.json (%d rollout texts)", len(responses))

    C.phase("capture")
    invalid_ci: list[int] = []
    valid_ci: list[int] = []
    for ci, (p, resp) in enumerate(zip(prompts, responses, strict=True)):
        ctx_path = ctx_dir / f"ctx{ci:05d}.pt"
        if args.resume and ctx_path.exists():
            valid_ci.append(ci)
            continue
        msgs = [{"role": "user", "content": p}]
        av = COL.capture_answer_vector(hf, tok, msgs, resp, layers, {}, keep_per_token=True)
        if av is None or "per_token" not in av:  # empty response -> no answer span
            invalid_ci.append(ci)
            continue
        stack = av["per_token"].to(torch.float16)  # (n_tok, L, H)
        assert stack.shape[1] == len(layers), (stack.shape, layers)
        torch.save(
            {
                "ci": ci,
                "answer_per_token": stack,  # (n_tok, L, H) fp16
                "layers": layers,
                "n_tok": int(stack.shape[0]),
            },
            ctx_path,
        )
        valid_ci.append(ci)
    logger.info(
        "[capture] %d/%d contexts captured (%d empty-response invalid)",
        len(valid_ci),
        len(prompts),
        len(invalid_ci),
    )

    C.phase("manifest")
    manifest = {
        "n_ctx": len(prompts),
        "n_valid": len(valid_ci),
        "n_invalid": len(invalid_ci),
        "invalid_ci": invalid_ci,
        "valid_ci": valid_ci,
        "layers": layers,
        "source": GC.LMSYS_REPO,
        "recipe": "pass-B (temp 1.0, top_p 0.95, seed 42, max 1024, default system)",
        "note": (
            "Per-context per-token answer activation stacks (answer span incl. "
            "im_end+\\n tail), teacher-forced, fp16. contexts/ctx{ci:05d}.pt each "
            "carries answer_per_token (n_tok, len(layers), H). invalid_ci = empty "
            "responses (no answer span; dropped, never zero-faked)."
        ),
        "metadata": C.reproducibility_metadata(
            {
                "script": "issue779_pertoken_lmsys_capture",
                "old_prompt_sha256": samp["old_prompt_sha256"],
            }
        ),
    }
    C.write_json_atomic(args.out_dir / "manifest.json", manifest)

    if not args.no_upload:
        C.phase("upload")
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub

        api = HfApi()
        api.upload_folder(
            folder_path=str(args.out_dir),
            path_in_repo=args.hf_prefix,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            commit_message="issue779: per-token LMSYS answer activations (stacks + text)",
        )
        expected = [
            f"{args.hf_prefix}/manifest.json",
            f"{args.hf_prefix}/raw_completions.json",
        ] + [f"{args.hf_prefix}/contexts/ctx{ci:05d}.pt" for ci in valid_ci]
        missing = hub.verify_repo_paths_uploaded(
            api, C.HF_DATA_REPO, expected, path_in_repo=args.hf_prefix, repo_type="dataset"
        )
        if missing:
            raise RuntimeError(
                f"per-token LMSYS upload verify FAILED — {len(missing)} missing on HF: "
                f"{missing[:10]}"
            )
        logger.info("uploaded + verified %d files under %s", len(expected), args.hf_prefix)

    C.phase("done")
    logger.info("[timing] total %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
