# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, ρ, →, ×) in scientific docstrings + log messages.
"""Issue #658 persona-vectors-style r_B — GPU phase (PV rollout + response-avg capture).

The faithful persona-vectors r_B build (arXiv 2507.21509 App-A) on
``Qwen/Qwen2.5-7B-Instruct`` (plan v5 §4.2). NO training — base model θ0 only.
The ONE new GPU job: generate on-policy rollouts under the PV system prompts and
capture their RESPONSE-AVG residual activations across all 28 layers, so the
off-pod CPU fit (``issue658_rb_pv_fit.py``) can judge-filter them and build a
diff-in-means r_B per (behavior × pole × reduction × layer). Everything
downstream (v0(C), E0(C,B), Σ_c, the noise floor) is REUSED from #658 @b33429f —
this script writes ONLY the new PV rollouts + per-rollout response-avg acts +
raw transcripts.

Pipeline (one ``nohup`` launch, one loaded model):

- PV0  load the 4 PV artifact bundles (``data/issue_658/persona-vectors-style-rb/``)
- PV1  vLLM batched generation: per behavior × {5 pos, 5 neg, 1 neutral} system
       prompts × 20 extraction questions × 10 rollouts (ONE LLM.generate per
       behavior, temperature 1.0)
- PV2  response-avg capture: teacher-force each rollout through the hooked HF
       model, average residual acts over the RESPONSE-token span, all 28 layers
       → per-rollout (28, 3584) fp16
- PV3  few-shot-final capture (Knob B option 3): a prefilled-ICL pass capturing
       the FINAL response-token activation (descopable, plan §9/§14)
- upload the per-rollout acts + raw transcripts + the build manifest to HF, then
  the pod terminates; J1 (judge-filter) + the r_B diff-in-means happen off-pod.

Smoke = sweep with ONE cell: ``--smoke`` runs the IDENTICAL dispatcher with a
tiny slice (1 behavior × 1 pos/neg pair × 2 extraction-q × 2 rollouts). One code
path; the cell list is the only thing that shrinks (Step 6d.0 PASS_UNIFIED).

Pod-side contract: ``[phase=...]`` log lines ending in ``[phase=done]`` + a
``poll_pipeline.py``-conformant end-of-run sentinel.

Launch (full):
    nohup uv run python scripts/issue658_extract_rb_personavectors.py \\
        --pv-artifact-dir data/issue_658/persona-vectors-style-rb \\
        --out-dir data/issue_658/store_pvrb --gpu-id 0 \\
        > logs/issue658_rb_pv.log 2>&1 &

Smoke (CPU/dev-GPU, tiny N, no upload):
    uv run python scripts/issue658_extract_rb_personavectors.py --smoke \\
        --device cpu --no-vllm --no-upload --out-dir /tmp/i658_rb_smoke
"""

from __future__ import annotations

# vLLM v1 EngineCore teardown wants the spawn multiproc method; HF cache on the
# persistent pod volume. Set at module top, before the lazy vllm import (the
# parent extractor's contract; gotchas.md #628).
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Cross-script helpers hoisted to module top so a missing symbol crashes at
# process start, never inside a smoke-skipped branch (gotchas.md #606). REUSE the
# parent extractor's validated AnswerSpanCapture / _reap_vllm / load_hf_model
# rather than reimplementing them (plan §4.2).
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_common import (  # noqa: E402
    DEFAULT_MODEL,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    HF_DATA_REPO,
    HF_OVERFLOW_REPO,
    HF_PREFIX,
    MARKER_TEXT,
    MARKER_TOKEN_ID,
    dump_json,
    load_json,
    sha256_file,
    summarize_answer_span,
)
from issue658_extract_base_store import (  # noqa: E402
    AnswerSpanCapture,
    _reap_vllm,
    load_hf_model,
)
from issue658_pv_bundles import (  # noqa: E402
    PV_BEHAVIORS,
    assert_disjoint_from_battery,
)

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue658_rb_pv_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SENTINEL_SCHEMA_VERSION = 1

# The NEW r_B prefix (non-clobbering vs the v0/E0 store; plan §6.5 / §10).
PV_HF_SUBDIR = "persona-vectors-style-rb"

# arXiv 2507.21509 App-A (plan §11): 5 pos/neg pairs, 20-q extraction set, 10
# rollouts per (question × system prompt), response-avg, diffmeans.
N_PAIRS = 5
N_EXTRACT_Q = 20
N_ROLLOUTS = 10
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_MAX_NEW_TOKENS = 512  # free-generation PV rollouts (plan §4.2)
FEWSHOT_K = 4  # few-shot-final demo count (Knob B option 3, plan §4.2/§4.8)


def phase(name: str) -> None:
    """Emit a poll_pipeline.py-parseable phase line (PHASE_RE on the log tail)."""
    print(f"[phase={name}]", flush=True)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ── PV0: load bundles ─────────────────────────────────────────────────────────


def load_pv_bundle(behavior: str, artifact_dir: Path) -> dict:
    """Load + validate one committed PV bundle (frozen artifact, never re-gen)."""
    path = artifact_dir / f"{behavior}.json"
    if not path.is_file():
        raise RuntimeError(
            f"missing PV bundle {path} — run `issue658_pv_bundles.py --write` first (plan §4.2)"
        )
    b = load_json(path)
    assert b["behavior"] == behavior, f"{path}: behavior {b['behavior']!r} != {behavior!r}"
    assert len(b["pos"]) == N_PAIRS and len(b["neg"]) == N_PAIRS, (
        f"{behavior}: expected {N_PAIRS} pos/neg prompts"
    )
    assert len(b["extract_q"]) == N_EXTRACT_Q, f"{behavior}: expected {N_EXTRACT_Q} extraction q"
    # Re-assert the contamination guard at extraction time (plan §4.7) — fail loud
    # if any extraction question collides with the 50-context battery probe pool.
    assert_disjoint_from_battery(behavior, b)
    return b


# ── PV1: rollout generation (vLLM batched) ────────────────────────────────────


def _chat_prompt(tokenizer, system_prompt: str, user_q: str) -> str:
    """Templated chat string: persona injection ALWAYS a system turn (CLAUDE.md)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_q},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_rollout_index(
    tokenizer, bundle: dict, n_pairs: int, n_extract_q: int, n_rollouts: int
) -> tuple[list[str], list[dict]]:
    """All (system-prompt × extraction-q × rollout) prompt strings + the index.

    pole ∈ {pos, neg, neutral}; ``prompt_idx`` is the within-pole system-prompt
    index. ONE flat prompt list → ONE batched LLM.generate (plan §4.2 / §9
    code-style batched rule). The index row carries everything the capture +
    judge phases need to reattach a rollout to its (pole, prompt, question).
    """
    pos = bundle["pos"][:n_pairs]
    neg = bundle["neg"][:n_pairs]
    neutral = bundle["neutral"][:1]
    extract_q = bundle["extract_q"][:n_extract_q]

    prompts: list[str] = []
    index: list[dict] = []
    pole_sets = [("pos", pos), ("neg", neg), ("neutral", neutral)]
    for pole, sys_prompts in pole_sets:
        for pi, sysp in enumerate(sys_prompts):
            for qi, q in enumerate(extract_q):
                templated = _chat_prompt(tokenizer, sysp, q)
                for ri in range(n_rollouts):
                    prompts.append(templated)
                    index.append(
                        {
                            "behavior": bundle["behavior"],
                            "pole": pole,
                            "prompt_idx": pi,
                            "system_prompt": sysp,
                            "question_idx": qi,
                            "question": q,
                            "rollout_idx": ri,
                        }
                    )
    return prompts, index


def vllm_generate_sampling(
    model_name: str, prompts: list[str], max_new_tokens: int, temperature: float, seed: int
) -> list[str]:
    """vLLM batched SAMPLING generation (temperature 1.0, n=1) over all prompts.

    ONE LLM.generate (the batched compute rule); use_tqdm=False (gotchas.md #613).
    n=1 with the rollout multiplicity already expanded in the prompt list (so each
    rollout gets an independent sample at temperature 1.0 — the paper's recipe).
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.45, seed=seed)
    sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, n=1, seed=seed)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    completions = [o.outputs[0].text for o in outs]
    _reap_vllm(llm)
    return completions


def hf_generate_sampling(
    model, tokenizer, prompts: list[str], max_new_tokens: int, temperature: float, seed: int
) -> list[str]:
    """HF sampling fallback (CPU smoke / --no-vllm). Batch-1 (smoke-scale only)."""
    torch.manual_seed(seed)
    completions: list[str] = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        completions.append(tokenizer.decode(gen, skip_special_tokens=True))
    return completions


# ── PV2: response-avg capture (teacher-force the rollout) ─────────────────────


def capture_response_avg(
    model,
    tokenizer,
    system_prompt: str,
    user_q: str,
    completion: str,
    capture: AnswerSpanCapture,
    n_layers: int,
) -> torch.Tensor | None:
    """RESPONSE-avg residual acts (all layers) for one rollout — (n_layers, H) fp16.

    Teacher-force (prompt + completion) through the hooked model, average the
    residual stream over the RESPONSE-token span at every layer
    (``summarize_answer_span(span, "mean")`` = the paper's response-avg default,
    the SAME recipe #658's v0(C) uses → r_B and v0(C) share the residual space).
    Returns None for an empty completion (no response span — logged, never crashes
    the batch). Fail-loud on a span-length mismatch (plan §4.2 capture contract).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_q},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"]
    ans_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ans_ids.shape[1] == 0:
        return None
    full_ids = torch.cat([prompt_ids, ans_ids], dim=1).to(model.device)
    prompt_len = int(prompt_ids.shape[1])
    ans_len = int(ans_ids.shape[1])
    with torch.no_grad():
        _ = model(input_ids=full_ids)
    span_full = capture.answer_span_stack(n_layers, prompt_len, prompt_len + ans_len)  # (L,S,H)
    captured_s = span_full.shape[1]
    assert captured_s == ans_len, (
        f"response-span length mismatch (behavior pole capture): captured {captured_s} "
        f"positions != {ans_len} generated answer tokens"
    )
    # response-avg per layer → (L, H) fp16
    resp_avg = torch.stack(
        [summarize_answer_span(span_full[li], "mean") for li in range(n_layers)]
    ).to(torch.float16)
    return resp_avg


def capture_fewshot_final(
    model,
    tokenizer,
    demos: list[tuple[str, str]],
    scored_q: str,
    capture: AnswerSpanCapture,
    n_layers: int,
) -> torch.Tensor | None:
    """Few-shot-final reduction (Knob B option 3): FINAL response-token act after ICL.

    Prefill k in-context (Q, A) trait-demonstration pairs (the demo answers are
    judge-kept pos-pole rollouts for OTHER extraction questions — passed in by the
    caller, rotated + disjoint from ``scored_q``; plan §4.2/§4.8), then teacher-force
    the model's own one-token continuation marker and capture the FINAL
    response-token residual at every layer → (n_layers, H) fp16.

    Here at extraction time we capture the activation at the LAST input position of
    the ICL-prefixed prompt for the scored question (the slot the model would
    generate its trait response from) — a single-position read, no full rollout,
    so the pass is cheap (plan §9 PV3 0.5 GPU-h). Returns None if no demos exist
    (a behavior with too few judge-kept rollouts — handled CPU-side; here the demos
    are passed pre-filtered by the caller, so None means "skip few-shot-final").
    """
    if not demos:
        return None
    messages: list[dict] = []
    for dq, da in demos:
        messages.append({"role": "user", "content": dq})
        messages.append({"role": "assistant", "content": da})
    messages.append({"role": "user", "content": scored_q})
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"].to(
        model.device
    )
    prompt_len = int(prompt_ids.shape[1])
    with torch.no_grad():
        _ = model(input_ids=prompt_ids)
    # Reuse last_prompt_stack: the residual at the final input slot, all layers.
    final = capture.last_prompt_stack(n_layers, prompt_len).to(torch.float16)  # (L, H)
    capture.latest.clear()
    return final


# ── sentinel / upload ─────────────────────────────────────────────────────────


def write_sentinel(kind: str, note: str, task_id: int = 658) -> Path:
    """poll_pipeline.py-conformant end-of-run sentinel (_SENTINEL_REQUIRED_KEYS)."""
    import json

    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    path = logs_dir / f"issue-{task_id}-{kind_slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "note": note,
        "task_id": task_id,
        "by": "issue658_extract_rb_personavectors",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote sentinel %s", path)
    return path


def _is_storage_quota_403(err: Exception) -> bool:
    msg = str(err)
    return "403" in msg and "storage" in msg.lower()


def upload_pv_store(out_dir: Path, smoke: bool) -> dict:
    """ONE bulk upload_folder commit of the PV r_B store; verify via list_repo_files.

    Lands under ``{HF_PREFIX}/{PV_HF_SUBDIR}[/smoke]`` (non-clobbering vs the v0/E0
    store, plan §6.5/§10). Quota-403 → overflow repo (upload-policy.md). Fail-loud
    verify on the manifest + the per-rollout index before returning.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    sub = f"{PV_HF_SUBDIR}/smoke" if smoke else PV_HF_SUBDIR
    path_in_repo = f"{HF_PREFIX}/{sub}"
    repo_used = HF_DATA_REPO
    try:
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue658: {'smoke ' if smoke else ''}PV r_B store upload",
        )
    except Exception as e:
        if not _is_storage_quota_403(e):
            raise
        logger.warning("HF storage-quota 403 on %s; falling back to overflow repo", HF_DATA_REPO)
        repo_used = HF_OVERFLOW_REPO
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_OVERFLOW_REPO,
            repo_type="dataset",
            commit_message="issue658: PV r_B store upload (quota-403 overflow fallback)",
        )
    files = [
        f for f in api.list_repo_files(repo_used, repo_type="dataset") if f.startswith(path_in_repo)
    ]
    expected = {f"{path_in_repo}/rb_extract_manifest.json", f"{path_in_repo}/rollout_index.json"}
    missing = expected - set(files)
    if missing:
        raise RuntimeError(
            f"PV store upload verification failed; missing on {repo_used}: {missing}"
        )
    logger.info(
        "PV store upload verified on %s: %d files under %s", repo_used, len(files), path_in_repo
    )
    return {"repo": repo_used, "path_in_repo": path_in_repo, "n_files": len(files)}


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:  # noqa: C901 — linear phase pipeline (PV0→PV1→PV2→PV3→upload)
    parser = argparse.ArgumentParser(
        description="Issue #658: persona-vectors-style r_B extraction."
    )
    parser.add_argument(
        "--pv-artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "data/issue_658/persona-vectors-style-rb",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data/issue_658/store_pvrb")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--expected-layers", type=int, default=EXPECTED_LAYERS)
    parser.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    parser.add_argument(
        "--behaviors",
        default=",".join(PV_BEHAVIORS),
        help="comma-separated behavior subset (the --cells axis; default all 4)",
    )
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS, help="pos/neg pairs (default 5)")
    parser.add_argument(
        "--n-extract-q", type=int, default=N_EXTRACT_Q, help="extraction questions (default 20)"
    )
    parser.add_argument(
        "--n-rollouts", type=int, default=N_ROLLOUTS, help="rollouts per (q×prompt) (default 10)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=ROLLOUT_MAX_NEW_TOKENS,
        help="rollout generation length cap (default 512; smoke uses a small value for speed)",
    )
    parser.add_argument(
        "--no-fewshot", action="store_true", help="descope the few-shot-final pass (PV3)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny single-cell slice (1 behavior × 1 pair × 2 q × 2 rollouts) — smoke IS sweep",
    )
    parser.add_argument("--no-vllm", action="store_true", help="HF generate fallback (CPU smoke)")
    parser.add_argument("--no-upload", action="store_true", help="skip HF upload (local smoke)")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--seed", type=int, default=658, help="generation seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="preflight only: load bundles + assert env/shapes, no generation/upload",
    )
    args = parser.parse_args()

    phase("load")
    # Bind CVD before the first CUDA allocation (the +gpu_id clobber gotcha).
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    for b in behaviors:
        if b not in PV_BEHAVIORS:
            raise SystemExit(f"unknown behavior {b!r}; expected subset of {PV_BEHAVIORS}")

    # Smoke = sweep with ONE cell (Step 6d.0 PASS_UNIFIED): shrink the cell list,
    # never branch the pipeline.
    if args.smoke:
        behaviors = behaviors[:1]
        n_pairs = min(args.n_pairs, 1)
        n_extract_q = min(args.n_extract_q, 2)
        n_rollouts = min(args.n_rollouts, 2)
        # Cap generation length for the smoke unless the caller overrode it
        # explicitly — CPU generation of 512 tokens is the slow path.
        max_new_tokens = (
            args.max_new_tokens if args.max_new_tokens != ROLLOUT_MAX_NEW_TOKENS else 32
        )
    else:
        n_pairs, n_extract_q, n_rollouts = args.n_pairs, args.n_extract_q, args.n_rollouts
        max_new_tokens = args.max_new_tokens

    out_dir = Path(f"{args.out_dir}_smoke") if args.smoke else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    acts_dir = out_dir / "rollout_acts"
    acts_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # PV0: load + validate the committed bundles (fail loud on a missing/bad bundle
    # or a contamination overlap, BEFORE loading the model).
    bundles = {b: load_pv_bundle(b, args.pv_artifact_dir) for b in behaviors}
    logger.info(
        "PV0: loaded %d bundles (%s); cells = %d behaviors × (%d pos + %d neg + 1 neutral) prompts "
        "× %d q × %d rollouts",
        len(bundles),
        ", ".join(behaviors),
        len(behaviors),
        n_pairs,
        n_pairs,
        n_extract_q,
        n_rollouts,
    )

    if args.dry_run:
        # Preflight: assert env resolves + bundle shapes; no generation/upload.
        logger.info(
            "DRY-RUN OK: bundles validated, out_dir=%s, use_cuda=%s, vllm=%s. No generation.",
            out_dir,
            use_cuda,
            not args.no_vllm,
        )
        phase("done")
        return 0

    import wandb

    run_name = "issue658-rb-personavectors" + ("-smoke" if args.smoke else "")
    run = wandb.init(
        project="explore-persona-space",
        name=run_name,
        mode=args.wandb_mode,
        config={
            "model": args.model,
            "behaviors": behaviors,
            "n_pairs": n_pairs,
            "n_extract_q": n_extract_q,
            "n_rollouts": n_rollouts,
            "smoke": args.smoke,
            "recipe": "persona-vectors arXiv 2507.21509 App-A (judge by Sonnet 4.5, off-pod)",
        },
    )

    model, tokenizer = load_hf_model(args.model, use_cuda)
    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    assert n_layers == args.expected_layers, f"{n_layers} layers != expected {args.expected_layers}"
    assert hidden == args.expected_hidden, f"hidden {hidden} != expected {args.expected_hidden}"

    # Marker token assert IN-PROCESS (CLAUDE.md / marker-leakage-measurement.md) —
    # the residual-space contract. Skipped only for a non-Qwen CPU smoke stub.
    if "Qwen2.5-7B" in args.model:
        enc = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
        assert enc == [MARKER_TOKEN_ID], f"marker token drift: encode({MARKER_TEXT!r})={enc}"

    # ── PV1: rollout generation ───────────────────────────────────────────────
    # NOTE: the AnswerSpanCapture hooks are registered AFTER generation (before
    # PV2) so generation forwards don't churn the capture buffer; the hooks live
    # only for the PV2/PV3 teacher-forced capture passes and are removed after.
    phase("pv1_generate")
    all_index: list[dict] = []
    all_completions: list[str] = []
    for beh in behaviors:
        prompts, index = build_rollout_index(
            tokenizer, bundles[beh], n_pairs, n_extract_q, n_rollouts
        )
        logger.info("PV1[%s]: %d rollout prompts (ONE batched generate)", beh, len(prompts))
        if args.no_vllm or not use_cuda:
            comps = hf_generate_sampling(
                model, tokenizer, prompts, max_new_tokens, ROLLOUT_TEMPERATURE, args.seed
            )
        else:
            # Free the HF model's CUDA cache headroom for the vLLM engine, then
            # reload the HF model after the engine is reaped (PV2 needs the hooked
            # HF model). The parent extractor uses the same dual-instance pattern;
            # at A100-80 both fit (plan §9c). For simplicity + memory safety we
            # generate with vLLM in a fresh engine and keep the HF model resident.
            comps = vllm_generate_sampling(
                args.model, prompts, max_new_tokens, ROLLOUT_TEMPERATURE, args.seed
            )
        assert len(comps) == len(index), f"{beh}: {len(comps)} completions != {len(index)} prompts"
        all_index.extend(index)
        all_completions.extend(comps)
    wandb.log({"pv1_rollouts": len(all_completions)})

    # ── PV2: response-avg capture ─────────────────────────────────────────────
    phase("pv2_capture")
    capture = AnswerSpanCapture(model, n_layers)
    n_captured = 0
    n_empty = 0
    for i, (idx_row, comp) in enumerate(zip(all_index, all_completions, strict=True)):
        resp_avg = capture_response_avg(
            model, tokenizer, idx_row["system_prompt"], idx_row["question"], comp, capture, n_layers
        )
        if resp_avg is None:
            n_empty += 1
            idx_row["acts_file"] = None
            idx_row["empty"] = True
        else:
            acts_file = f"r{i:06d}.pt"
            torch.save(resp_avg, acts_dir / acts_file)  # (n_layers, H) fp16
            idx_row["acts_file"] = acts_file
            idx_row["empty"] = False
            n_captured += 1
        # Persist the raw transcript for J1 (the off-pod judge-filter): both inline
        # in the index AND as a per-rollout JSON under transcripts/ (the inspectable
        # raw-completion form that lands on HF via upload_pv_store).
        idx_row["completion"] = comp
        dump_json(
            {
                "behavior": idx_row["behavior"],
                "pole": idx_row["pole"],
                "prompt_idx": idx_row["prompt_idx"],
                "question_idx": idx_row["question_idx"],
                "rollout_idx": idx_row["rollout_idx"],
                "system_prompt": idx_row["system_prompt"],
                "question": idx_row["question"],
                "completion": comp,
            },
            transcripts_dir / f"r{i:06d}.json",
        )
        if (i + 1) % 200 == 0:
            logger.info("PV2: captured %d / %d rollouts", n_captured, i + 1)
    logger.info("PV2 done: %d captured, %d empty (skipped)", n_captured, n_empty)
    wandb.log({"pv2_captured": n_captured, "pv2_empty": n_empty})

    # ── PV3: few-shot-final (descopable) ──────────────────────────────────────
    fewshot_info: dict = {"enabled": not args.no_fewshot, "captured": 0}
    if not args.no_fewshot:
        phase("pv3_fewshot")
        fs_dir = out_dir / "fewshot_acts"
        fs_dir.mkdir(parents=True, exist_ok=True)
        fs_index: list[dict] = []
        n_fs = 0
        # Demos = pos-pole rollouts of OTHER extraction questions for the SAME
        # behavior (on-policy, trait-eliciting). The JUDGE-KEPT filter is applied
        # off-pod (CPU J1); here we use the pos-pole rollouts as the demo pool and
        # record which were used, so the fit can re-filter to judge-kept only.
        by_beh: dict[str, list[dict]] = {b: [] for b in behaviors}
        for row in all_index:
            if row["pole"] == "pos" and not row["empty"]:
                by_beh[row["behavior"]].append(row)
        for beh in behaviors:
            pos_rows = by_beh[beh]
            extract_q = bundles[beh]["extract_q"][:n_extract_q]
            for qi, scored_q in enumerate(extract_q):
                # demos: pos rollouts for questions != qi, take up to K
                demo_rows = [r for r in pos_rows if r["question_idx"] != qi][:FEWSHOT_K]
                demos = [(r["question"], r["completion"]) for r in demo_rows]
                final = capture_fewshot_final(model, tokenizer, demos, scored_q, capture, n_layers)
                if final is None:
                    continue
                fs_file = f"fs_{beh}_{qi:02d}.pt"
                torch.save(final, fs_dir / fs_file)
                fs_index.append(
                    {
                        "behavior": beh,
                        "question_idx": qi,
                        "question": scored_q,
                        "demo_rollout_files": [r["acts_file"] for r in demo_rows],
                        "n_demos": len(demos),
                        "acts_file": fs_file,
                    }
                )
                n_fs += 1
        dump_json({"fewshot_final": fs_index}, out_dir / "fewshot_index.json")
        fewshot_info["captured"] = n_fs
        logger.info("PV3 done: %d few-shot-final acts", n_fs)
        wandb.log({"pv3_fewshot": n_fs})

    capture.remove()  # drop the forward hooks before assemble/upload

    # ── assemble + upload ─────────────────────────────────────────────────────
    phase("assemble")
    dump_json({"rollouts": all_index}, out_dir / "rollout_index.json")

    manifest = {
        "model": args.model,
        "n_layers": n_layers,
        "hidden": hidden,
        "recipe": "persona-vectors arXiv 2507.21509 App-A; response-avg; diffmeans (built off-pod)",
        "behaviors": behaviors,
        "n_pairs": n_pairs,
        "n_extract_q": n_extract_q,
        "n_rollouts": n_rollouts,
        "rollout_temperature": ROLLOUT_TEMPERATURE,
        "rollout_max_new_tokens": max_new_tokens,
        "n_rollouts_total": len(all_index),
        "n_captured": n_captured,
        "n_empty": n_empty,
        "fewshot": fewshot_info,
        "reuse_v0_e0_rev": "b33429f77b86",
        "judge": "claude-sonnet-4-5-20250929 (applied off-pod in J1, NOT here)",
        "bundle_sha256": {b: sha256_file(args.pv_artifact_dir / f"{b}.json") for b in behaviors},
        "smoke": args.smoke,
        "wandb_run": run_name,
        "ts": _now_iso(),
        "metadata": reproducibility_metadata({"script": "issue658_extract_rb_personavectors"}),
    }
    dump_json(manifest, out_dir / "rb_extract_manifest.json")

    upload_info: dict = {"skipped": True}
    if not args.no_upload:
        phase("upload")
        upload_info = upload_pv_store(out_dir, args.smoke)
        manifest["upload"] = upload_info
        dump_json(manifest, out_dir / "rb_extract_manifest.json")

    run.finish()

    note = (
        f"PV r_B extract: {len(behaviors)} behaviors, {len(all_index)} rollouts "
        f"({n_captured} captured, {n_empty} empty), fewshot={fewshot_info['captured']}, "
        f"upload={upload_info}"
    )
    write_sentinel("epm:smoke-result" if args.smoke else "epm:results", note)
    # [phase=done] is the SINGLE terminal line (poll_pipeline.py contract) — emit
    # AFTER the final sentinel write, never on a per-cell echo.
    phase("done")
    logger.info("DONE: %s", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
