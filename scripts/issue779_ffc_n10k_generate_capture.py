#!/usr/bin/env python3
"""Issue #779 round-2 (`fitter-fair-comparison-n10k`): corpus extension + capture.

Samples the NEXT ~6,500 non-empty first-turn LMSYS prompts AFTER the round-1
pass-B 5,000 (deterministic stream, skip the first 5,000 — disjoint by
construction, string-disjointness VERIFIED against the re-derived round-1 set),
generates 1 rollout each with the IDENTICAL pass-B recipe, and runs the combined
teacher-forced capture: per (prompt, rollout) it captures

  * c_last + c_mean (last / mean prompt-token activation, all 28 layers) via
    ``issue779_collect.capture_context_vector`` (extract_layer_activations path),
  * v_x (mean-response activation, all 28 layers) via
    ``issue779_collect.capture_answer_vector`` (same path),
  * the 8 pass-1 answer summaries via ``capture_summaries_batched`` +
    the 8 pass-2 next-turn-template summaries via ``capture_pass2_batched``
    (model.model block-hook path — the SAME mechanisms round-1's separate
    capture scripts used, so the new rows are bit-consistent with the round-1
    corpus the driver combines them with).

REUSE, not reimplementation: every capture position/pooling comes from the
round-1 capture functions (``issue779_collect`` /
``issue779_capture_answer_summaries{,_pass2}``). The only new code is the
disjoint sampling + the single-pass orchestration (one ``_tokenize_item`` feeds
both batched summary captures) + the combined bundle write + HF upload.

Faithful-reuse deviation from the brief's "ONE forward per row": the three
capture mechanisms tokenize DIFFERENT sequences (prompt-only for c_last/c_mean;
prompt+response for v_x + pass-1; prompt+response+next-turn-template for pass-2),
so a single forward is impossible without diverging from the round-1 recipes.
We keep round-1's exact functions (each its own forward) — correctness over the
one-forward micro-optimization; recorded in metadata.deviations.

Round-1 corpus consistency: this writes ONLY the new ~6,500 contexts. The driver
combines pass-B (5,000, reused byte-for-byte — so round-2's val/test, drawn from
the first-5,000 ids, stay byte-identical to round-1) + these new rows = 11,500.

Outputs (local + HF ``issue779_monitoring/fitter-fair-comparison-n10k/``):
  new_context_vectors.pt   {cx_last, cx_mean, v_x, summ_p1, valid_p1, summ_p2,
                            prompts, stream_positions, layers, source, metadata}
  raw_completions.json     [{ci, prompt, response}]
  sampling_manifest.json   {skip, n_new, old_prompt_sha256, disjoint_ok, ...}
Checkpointed per capture shard; ``--resume`` skips completed shards.

Generation/capture is GPU (H100); analysis-only constraints are LIFTED for this
stage per the round-2 brief. NO judge calls. Fail loud — NaN never coerced.
"""

from __future__ import annotations

import argparse
import hashlib
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

import issue779_capture_answer_summaries as P1  # noqa: E402
import issue779_capture_answer_summaries_pass2 as P2  # noqa: E402
import issue779_collect as COL  # noqa: E402
import issue779_common as C  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue779_ffc_n10k_gc")

N_LAYERS = C.EXPECTED_LAYERS  # 28
H_DIM = C.EXPECTED_HIDDEN  # 3584
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # MUST match round-1 pass-B (Qwen-2.5-7B-Instruct)
HF_PREFIX = "issue779_monitoring/fitter-fair-comparison-n10k"
LMSYS_REPO = "lmsys/lmsys-chat-1m"


def _sha_prompts(prompts: list[str]) -> str:
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _first_user_turn(row) -> str | None:
    """Round-1's exact predicate (issue779_collect.load_train_contexts)."""
    val = row.get("conversation")
    if isinstance(val, list) and val and isinstance(val[0], dict):
        p = val[0].get("content") or val[0].get("value")
        return p.strip() if isinstance(p, str) and p.strip() else None
    return None


def sample_disjoint(skip: int, n_new: int) -> dict:
    """Stream LMSYS; first ``skip`` non-empty first-turn prompts = round-1 set
    (re-derived deterministically); the NEXT non-empty prompts whose string is
    NOT in the round-1 set = the new set (verified string-disjoint)."""
    from datasets import load_dataset

    ds = load_dataset(LMSYS_REPO, split="train", streaming=True)
    old: list[str] = []
    old_set: set[str] = set()
    new: list[str] = []
    new_pos: list[int] = []
    pos = -1
    it = iter(ds)
    # Phase 1: first `skip` non-empty first-turns → round-1 set.
    while len(old) < skip:
        pos += 1
        row = next(it)
        p = _first_user_turn(row)
        if p:
            old.append(p)
            old_set.add(p)
    # Phase 2: next non-empty first-turns NOT in the round-1 set → new set.
    while len(new) < n_new:
        pos += 1
        row = next(it)
        p = _first_user_turn(row)
        if p and p not in old_set and p not in set(new):
            new.append(p)
            new_pos.append(pos)
    disjoint_ok = set(new).isdisjoint(old_set)
    assert disjoint_ok, "new prompts overlap the round-1 set (should be impossible)"
    # release the streaming dataset before shutdown (#952 rc=134 guard)
    del it, ds
    import gc

    gc.collect()
    logger.info(
        "sampled %d round-1 (re-derived) + %d new disjoint prompts (last stream pos %d)",
        len(old),
        len(new),
        pos,
    )
    return {
        "old": old,
        "new": new,
        "new_stream_pos": new_pos,
        "old_prompt_sha256": _sha_prompts(old),
        "new_prompt_sha256": _sha_prompts(new),
        "disjoint_ok": bool(disjoint_ok),
        "skip": skip,
        "n_new": len(new),
    }


def load_models(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    # position assert: the round-1 GENERATION_SUFFIX must hold for this tokenizer
    probe = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    assert tok.decode(tok(probe)["input_ids"][-3:]) == C.GENERATION_SUFFIX, (
        "tokenizer GENERATION_SUFFIX drift — model must be the round-1 Qwen-2.5-7B-Instruct"
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    hf = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map={"": 0} if device == "cuda" else None
    )
    hf.eval()
    return tok, hf


def _generate(llm, tok, prompts) -> list[str]:
    """1 rollout per prompt with the round-1 pass-B recipe (vLLM). CPU-smoke path
    (llm is None) returns a fixed short response through the SAME capture code."""
    if llm is None:  # --device cpu smoke: capture-path structural check only
        return ["This is a short stub response for the CPU capture smoke."] * len(prompts)
    from vllm import SamplingParams

    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42)
    prompt_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    gen = COL._vllm_generate_chunked(llm, prompt_texts, sp)  # list[list[str]]
    return [g[0] for g in gen]


def _capture_shard(hf, tok, prompts, responses, ci_base, layers, batch_size):
    """Capture one shard given pre-generated responses. Returns per-kept-row dicts."""
    # c_last / c_mean / v_x via the round-1 extract-layer path; keep only rows
    # with a non-empty response (v_x computable) — matches run_pass_b kept_idx.
    kept, cx_last, cx_mean, v_x, items = [], [], [], [], []
    for li, (p, resp) in enumerate(zip(prompts, responses, strict=True)):
        msgs = [{"role": "user", "content": p}]
        cx = COL.capture_context_vector(hf, tok, msgs, layers)
        av = COL.capture_answer_vector(hf, tok, msgs, resp, layers, {}, keep_per_token=False)
        if av is None:  # empty response
            continue
        kept.append(li)
        cx_last.append(cx["last"])
        cx_mean.append(cx["mean"])
        v_x.append(av["v_x"])
        items.append(
            P1._tokenize_item(
                tok, {"ci": ci_base + li, "ri": 0, "messages": msgs, "response": resp}
            )
        )
    if not kept:
        return []
    # 8+8 summaries via the batched block-hook path (same as the round-1 shards).
    p1 = P1.capture_summaries_batched(hf, tok, items, layers, batch_size)
    suffix_ids = P2._next_user_suffix_ids(tok)
    p2 = P2.capture_pass2_batched(hf, tok, items, layers, batch_size, suffix_ids)
    rows = []
    for j, li in enumerate(kept):
        rows.append(
            {
                "ci": ci_base + li,
                "prompt": prompts[li],
                "response": responses[li],
                "cx_last": cx_last[j],  # (L, H)
                "cx_mean": cx_mean[j],
                "v_x": v_x[j],
                "summ_p1": p1[j]["summ"],  # (4, L, H) fp16
                "valid_p1": p1[j]["valid"],  # (4,)
                "summ_p2": p2[j]["summ"],  # (8, L, H) fp16
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 round-2 corpus extension + capture.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip", type=int, default=5000)
    ap.add_argument("--n-new", type=int, default=6500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_779" / "ffc_n10k"
    )
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_new = args.n_new if args.n_new != 6500 else 50
        args.shard_size = min(args.shard_size, 25)
        args.batch_size = min(args.batch_size, 8)

    layers = list(range(N_LAYERS))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    t0 = time.time()

    C.phase("sample")
    manifest = sample_disjoint(args.skip, args.n_new)
    C.write_json_atomic(
        args.out_dir / "sampling_manifest.json",
        {
            **{k: v for k, v in manifest.items() if k not in ("old", "new")},
            "new_stream_pos": manifest["new_stream_pos"],
            "model": args.model,
            "layers": layers,
            "metadata": C.reproducibility_metadata(
                {"script": "issue779_ffc_n10k_generate_capture"}
            ),
        },
    )
    new_prompts = manifest["new"]

    C.phase("load_model")
    tok, hf = load_models(args.model, args.device)
    llm = None
    if args.device == "cuda":
        from explore_persona_space.eval.generation import create_vllm_engine

        llm = create_vllm_engine(args.model, max_model_len=8192, seed=42)
    # else: --device cpu smoke -> llm stays None; _generate returns a stub response

    # one-time bf16 batched-vs-serial equivalence gates (fail loud)
    C.phase("gates")
    gates = {"p1": P1.equivalence_gate(hf, tok, layers)}
    logger.info("pass-1 equivalence gate: %s", gates["p1"].get("pass", gates["p1"]))

    C.phase("capture")
    n = len(new_prompts)
    for s in range(0, n, args.shard_size):
        sid = s // args.shard_size
        shard_path = shard_dir / f"shard{sid:04d}.pt"
        if args.resume and shard_path.exists():
            logger.info("[capture] shard %d exists; skip", sid)
            continue
        chunk = new_prompts[s : s + args.shard_size]
        ts = time.time()
        responses = _generate(llm, tok, chunk)
        rows = _capture_shard(hf, tok, chunk, responses, s, layers, args.batch_size)
        torch.save({"rows": rows, "ci_base": s, "n_prompts": len(chunk)}, shard_path)
        logger.info(
            "[capture] shard %d: %d/%d kept (%.0fs)", sid, len(rows), len(chunk), time.time() - ts
        )

    C.phase("assemble")
    rows: list[dict] = []
    for sp_ in sorted(shard_dir.glob("shard*.pt")):
        rows.extend(torch.load(sp_, weights_only=False, map_location="cpu")["rows"])
    assert rows, "no captured rows"
    bundle = {
        "cx_last": torch.stack([r["cx_last"] for r in rows]),
        "cx_mean": torch.stack([r["cx_mean"] for r in rows]),
        "v_x": torch.stack([r["v_x"] for r in rows]),
        "summ_p1": torch.stack([r["summ_p1"] for r in rows]),  # (N,4,L,H)
        "valid_p1": torch.stack([r["valid_p1"] for r in rows]),
        "summ_p2": torch.stack([r["summ_p2"] for r in rows]),  # (N,8,L,H)
        "p1_summaries": list(P1.SUMMARIES),
        "p2_summaries": list(P2.SUMMARIES2),
        "prompts": [r["prompt"] for r in rows],
        "ci": [r["ci"] for r in rows],
        "layers": layers,
        "source": LMSYS_REPO,
        "metadata": C.reproducibility_metadata(
            {
                "script": "issue779_ffc_n10k_generate_capture",
                "n_kept": len(rows),
                "n_new_sampled": len(new_prompts),
                "old_prompt_sha256": manifest["old_prompt_sha256"],
                "new_prompt_sha256": manifest["new_prompt_sha256"],
                "disjoint_ok": manifest["disjoint_ok"],
                "gates": gates,
                "deviations": [
                    "3 forwards/row (prompt-only c_x; prompt+resp v_x+pass1; +next-turn pass2) — "
                    "the round-1 capture recipes tokenize different sequences, so 'one forward' is "
                    "impossible without diverging; correctness over the micro-optimization.",
                    "c_last/c_mean/v_x via extract_layer_activations (capture_context_vector / "
                    "capture_answer_vector); 8+8 summaries via model.model block hooks "
                    "(capture_summaries_batched / capture_pass2_batched) — matches round-1.",
                ],
            }
        ),
    }
    for fld in ("cx_last", "cx_mean", "v_x"):
        assert bundle[fld].shape[1:] == (N_LAYERS, H_DIM), (fld, bundle[fld].shape)
    C.write_json_atomic(
        args.out_dir / "raw_completions.json",
        {"rows": [{"ci": r["ci"], "prompt": r["prompt"], "response": r["response"]} for r in rows]},
    )
    bundle_path = args.out_dir / "new_context_vectors.pt"
    torch.save(bundle, bundle_path)
    logger.info("wrote %s (%d rows) + raw_completions.json", bundle_path, len(rows))

    if not args.no_upload:
        C.phase("upload")
        from explore_persona_space.orchestrate import hub

        # bulk upload_folder (raw completions JSON + bundle + manifest) to the data repo
        hub.upload_dataset_directory(
            local_dir=args.out_dir, path_in_repo=args.hf_prefix, allow_patterns=None
        ) if hasattr(hub, "upload_dataset_directory") else hub._upload(
            args.out_dir, repo_type="dataset", path_in_repo=args.hf_prefix
        )
        logger.info("uploaded %s -> %s", args.out_dir, args.hf_prefix)

    C.phase("done")
    logger.info("[timing] total %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
