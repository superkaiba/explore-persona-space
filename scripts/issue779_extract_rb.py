#!/usr/bin/env python3
"""Issue #779: persona-vector ``r_B`` extraction (persona-vectors-recipe.md).

Faithful reproduction of arXiv 2507.21509 §2 + App. "Direction extraction
pipeline" for the 3 main-text traits (evil / sycophancy / hallucination), with
EXACTLY ONE standing deviation: the project Sonnet judge
(``claude-sonnet-4-5-20250929``) replaces the paper's GPT-4.1-mini
logit-weighted scoring (the "except logits" carve-out, persona-vectors-recipe.md).

The 7-step recipe:
  1. Inputs: trait name + verbatim description (issue779_common.TRAIT_DESCRIPTIONS).
  2. Artifacts: evil verbatim from the paper; syc/halluc generated via one Sonnet
     call from the verbatim generation-prompt template. 5 pos/neg system-prompt
     pairs + 20 extraction Q (disjoint 20 eval Q) + 1 eval rubric.
  3. Generation: 10 rollouts under POS + 10 under NEG per extraction question,
     temp 1.0, vLLM (one engine, chunked generate — the #664 large-batch guard).
  4. Judge-filter: Sonnet graded 0-100; keep POS>50 / NEG<50; REFUSAL/non-numeric/
     out-of-range DROPPED from BOTH arms, never coerced; per-arm dropped count
     REPORTED.
  5. Activation position: residual stream at EVERY layer (0..27), averaged over
     RESPONSE tokens per kept rollout, via analysis.extraction.extract_layer_activations
     (output_hidden_states=False — the #666 OOM-safe hook path). Stream-reduced
     (running sum + count per layer per arm) — never materialize all N
     activations at once (the earlyoom bulk-load guard).
  6. r_B[layer] = mean(kept POS acts) - mean(kept NEG acts), per layer -> (28, 3584).
  7. Layer-selection regime: READ-OUT/PREDICTION (Step-0 sweeps all layers,
     selects by predictivity downstream — not steering).

Output: ``data/issue_779/r_b/{trait}.pt`` (28, 3584) + a per-arm kept/dropped
counts JSON. ``--smoke`` runs the IDENTICAL path on 1 trait x 1 pair x N-Q x
5-rollout tiny slice into ``data/issue_779_smoke/r_b/``.

Usage:
    uv run python scripts/issue779_extract_rb.py --stage all --gpu-id 0
    uv run python scripts/issue779_extract_rb.py --stage artifacts   # syc/halluc gen (no GPU)
    uv run python scripts/issue779_extract_rb.py --smoke --model Qwen/Qwen2.5-0.5B-Instruct \\
        --expected-layers 24 --expected-hidden 896 --device cpu --no-judge
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_common as C  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_extract_rb")

# vLLM large-batch deadlock guard (#664): chunk generate() calls.
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))


# ── Generation (one vLLM engine, chunked) ─────────────────────────────────────


def _build_prompts(tokenizer, system_prompt: str, questions: list[str]) -> list[str]:
    """Chat-templated prompt texts for (system_prompt, question) pairs."""
    texts = []
    for q in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return texts


def _vllm_generate_chunked(llm, prompt_texts: list[str], sampling_params) -> list[list[str]]:
    """llm.generate over prompt_texts, chunked (the #664 deadlock guard).

    Returns per-prompt list of ``n`` completions. Preserves input order; emits a
    per-chunk INFO log so the poller sees liveness on long generation phases.
    """
    out: list[list[str]] = []
    n_chunks = (len(prompt_texts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for i in range(0, len(prompt_texts), VLLM_CHUNK_SIZE):
        chunk = prompt_texts[i : i + VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] rb-generate chunk %d/%d (%d prompts x n=%d)",
            i // VLLM_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
            sampling_params.n,
        )
        chunk_out = llm.generate(chunk, sampling_params, use_tqdm=False)
        for o in chunk_out:
            out.append([c.text for c in o.outputs])
    return out


# ── Rollout-text persistence (persist-by-default, upload-policy.md) ───────────

# Each dump file stays under the non-LFS-safe size: text >9.5 MB per file must
# line-split (upload-policy.md), and the Hub force-routes >10 MB blobs to LFS.
# 8.5 MB on the RECORD payload leaves headroom for the constant header.
ROLLOUT_DUMP_MAX_BYTES = 8_500_000


def _iter_rollout_records(arm_rollouts: dict[str, dict[str, list[str]]]):
    """Yield ``(persona, q_idx, question, ci, completion, custom_id)`` per rollout.

    The custom_id uses the GLOBAL question index (0 before the persona loop,
    +1 per question, NEVER reset per persona) — the exact enumeration
    ``batch_judge._enumerate_and_check_cache`` and ``C.judge_rollouts_n5``'s
    returned keys use (concern judge-n5-custom-id-index-mismatch-multipersona),
    so a dumped record pairs to its aggregated judge score by custom_id. The
    per-draw rows inside ``judge_{trait}_{arm}.json`` sit at the EXPANDED index
    ``ci * n_draws + draw`` under the same scheme.
    """
    idx = 0
    for persona, qmap in arm_rollouts.items():
        for question, comps in qmap.items():
            for ci, comp in enumerate(comps):
                yield persona, idx, question, ci, comp, f"{persona}__{idx:05d}__{ci:02d}"
            idx += 1


def _chunk_records(records: list[dict], max_bytes: int) -> list[list[dict]]:
    """Greedy record-boundary chunking so each dump file stays under max_bytes."""
    parts: list[list[dict]] = []
    cur: list[dict] = []
    cur_bytes = 0
    for rec in records:
        rec_bytes = len(json.dumps(rec).encode("utf-8")) + 8  # +8: separators/indent slack
        if cur and cur_bytes + rec_bytes > max_bytes:
            parts.append(cur)
            cur, cur_bytes = [], 0
        cur.append(rec)
        cur_bytes += rec_bytes
    if cur:
        parts.append(cur)
    return parts


def _dump_rollouts(
    trait: str,
    arm: str,
    arm_rollouts: dict[str, dict[str, list[str]]],
    out_dir: Path,
    sampling: dict,
    max_bytes: int = ROLLOUT_DUMP_MAX_BYTES,
) -> list[Path]:
    """Persist the raw rollout TEXT for one (trait, arm) BEFORE any judge/reduce.

    Writes ``out_dir/raw_completions/rollouts_{trait}_{arm}.json`` (or
    ``.partNN.json`` shards when the record payload exceeds ``max_bytes`` —
    never gzip, which is LFS-matched). Every record carries the custom_id that
    pairs it to its judge score (see ``_iter_rollout_records``), so the reduced
    ``r_B`` tensor stays regenerable from text + judge JSONs alone
    (upload-policy.md persist-by-default; the #779 rollout-text incident and
    #1112 upload-verification blocker ``generation-discarded-undeclared``).
    Returns the written paths.
    """
    records = [
        {
            "persona": persona,
            "question_index": q_idx,
            "question": question,
            "completion_index": ci,
            "custom_id": custom_id,
            "completion": comp,
        }
        for persona, q_idx, question, ci, comp, custom_id in _iter_rollout_records(arm_rollouts)
    ]
    assert records, f"{trait}/{arm}: no rollouts to persist"
    dump_dir = out_dir / "raw_completions"
    parts = _chunk_records(records, max_bytes)
    written: list[Path] = []
    for pi, part in enumerate(parts):
        name = (
            f"rollouts_{trait}_{arm}.json"
            if len(parts) == 1
            else f"rollouts_{trait}_{arm}.part{pi:02d}.json"
        )
        payload = {
            "trait": trait,
            "arm": arm,
            "part": pi,
            "n_parts": len(parts),
            "n_rollouts_total": len(records),
            "n_rollouts_in_part": len(part),
            "custom_id_scheme": (
                "{persona}__{global_question_idx:05d}__{completion_idx:02d} — pairs with "
                "the aggregated per-rollout keys judge_rollouts_n5 returns and "
                "extract_trait_rb looks up; per-draw rows in judge_{trait}_{arm}.json "
                "use completion_idx*n_draws + draw"
            ),
            "sampling": sampling,
            "rollouts": part,
            "metadata": C.reproducibility_metadata(
                {"script": "issue779_extract_rb", "stage": "rollout_text_dump"}
            ),
        }
        path = dump_dir / name
        C.write_json_atomic(path, payload)
        written.append(path)
    return written


# ── Judge (graded 0-100 trait score, DROP-NEVER-COERCE) ───────────────────────


def _judge_rollouts(
    trait: str,
    rollouts: dict[str, dict[str, list[str]]],
    save_raw: Path,
    cache_dir: Path,
    dry_run: bool = False,
) -> tuple[dict[str, float | None], dict]:
    """Judge rollouts with the graded N=5 trait rubric; mean over valid draws.

    ``rollouts`` is the batch_judge {persona: {question: [completions]}} shape.
    Judges each rollout with the registered N=5 graded-0-100 draws @ temp 1.0
    (``C.judge_rollouts_n5``, DROP-NEVER-COERCE per draw), then aggregates the
    valid draws to a per-rollout mean BEFORE the >50 / <50 threshold. Returns
    ``({custom_id: mean_score|None}, draw_stats)`` where ``draw_stats`` reports
    the per-rollout dropped-draw distribution (llm-judging.md rules 4 + 9).
    """
    if dry_run:
        C.judge_rollouts_n5(trait, rollouts, save_raw, cache_dir, dry_run=True)
        return {}, {}
    agg = C.judge_rollouts_n5(trait, rollouts, save_raw, cache_dir)
    scores: dict[str, float | None] = {}
    n_draws_seen = 0
    total_valid = 0
    n_rollouts_all_dropped = 0
    for cid, (mean, n_valid, n_draws) in agg.items():
        scores[cid] = mean
        n_draws_seen += n_draws
        total_valid += n_valid
        if n_valid == 0:
            n_rollouts_all_dropped += 1
    draw_stats = {
        "n_rollouts_judged": len(agg),
        "n_draws_per_rollout": C.JUDGE_N_DRAWS,
        "total_draws": n_draws_seen,
        "total_valid_draws": total_valid,
        "total_dropped_draws": n_draws_seen - total_valid,
        "n_rollouts_all_draws_dropped": n_rollouts_all_dropped,
    }
    return scores, draw_stats


# ── Activation capture (response-avg, all layers, stream-reduced) ─────────────


class RunningMean:
    """Streaming per-layer sum + count so peak RSS is O(one activation), not O(N).

    The earlyoom bulk-load guard (gotchas.md): a diff-of-means over thousands of
    activations must accumulate running sums, never materialize all N at once.
    """

    def __init__(self, n_layers: int, hidden: int):
        self.sum = torch.zeros(n_layers, hidden, dtype=torch.float64)
        self.count = 0

    def add(self, stack: torch.Tensor) -> None:
        # stack: (L, H) fp32 response-mean for one rollout.
        self.sum += stack.to(torch.float64)
        self.count += 1

    def mean(self) -> torch.Tensor:
        assert self.count > 0, "RunningMean.mean() with zero kept rollouts"
        return (self.sum / self.count).to(torch.float32)


def _response_mean_activation(
    model, tokenizer, system_prompt: str, question: str, response: str, layers: list[int]
) -> torch.Tensor | None:
    """(L, H) mean-over-response-tokens activation at all layers, or None if empty.

    Teacher-forces (system+user+assistant-response) through the HF model and
    mean-pools the residual stream over the RESPONSE token span at every layer
    (the paper's response-avg). Reuses the OOM-safe hook helper.
    """
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_len = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"].shape[1]

    full_messages = [*prompt_messages, {"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_inputs = tokenizer(full_text, return_tensors="pt", padding=False).to(model.device)
    full_len = full_inputs["input_ids"].shape[1]
    if full_len <= prompt_len:
        return None  # empty response

    captured = extract_layer_activations(
        model, full_inputs["input_ids"], layers, attention_mask=full_inputs.get("attention_mask")
    )
    vecs = []
    for layer_idx in layers:
        hs = captured[layer_idx]  # (1, T, H)
        resp = hs[0, prompt_len:full_len, :].float().cpu()  # (n_resp, H)
        vecs.append(resp.mean(dim=0))  # (H,)
    return torch.stack(vecs)  # (L, H)


# ── Per-trait extraction ──────────────────────────────────────────────────────


def extract_trait_rb(
    trait: str,
    model,
    tokenizer,
    llm,
    layers: list[int],
    out_dir: Path,
    *,
    n_pairs: int,
    n_ext_q: int,
    n_rollouts: int,
    do_judge: bool,
    smoke: bool,
) -> dict:
    """Extract r_B for one trait; write (L, H) tensor + counts JSON. Returns counts."""
    artifacts = C.load_extraction_artifacts(trait)
    pairs = artifacts["instruction"][:n_pairs]
    ext_q = artifacts["extraction_questions"][:n_ext_q]
    n_layers = len(layers)
    hidden = model.config.hidden_size

    from vllm import SamplingParams

    sp = SamplingParams(n=n_rollouts, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42)

    # Build the {persona: {question: [rollouts]}} rollout structure for POS + NEG
    # arms. persona key encodes (pair_idx, arm) so custom_ids are unique.
    rollouts_by_arm: dict[str, dict[str, dict[str, list[str]]]] = {"pos": {}, "neg": {}}
    # keep a parallel map: (arm, persona, question, comp_idx) -> system_prompt for capture.
    sysprompt_of: dict[str, str] = {}

    for arm in ("pos", "neg"):
        for pi, pair in enumerate(pairs):
            sys_prompt = pair[arm]
            persona = f"{trait}_{arm}_p{pi}"
            sysprompt_of[persona] = sys_prompt
            prompt_texts = _build_prompts(tokenizer, sys_prompt, ext_q)
            gen = _vllm_generate_chunked(llm, prompt_texts, sp)  # per-prompt list of n
            rollouts_by_arm[arm][persona] = dict(zip(ext_q, gen, strict=True))

    # Persist the raw rollout TEXT the moment generation completes, BEFORE the
    # judge + activation stream-reduce (persist-by-default — a judge/capture
    # crash must never burn the generation phase, and rollout text is never
    # discardable; #1112 upload-verification blocker
    # generation-discarded-undeclared, plan §10 raw_completions/rb_extraction).
    sampling = {
        "n": sp.n,
        "temperature": sp.temperature,
        "top_p": sp.top_p,
        "max_tokens": sp.max_tokens,
        "seed": sp.seed,
        "model": getattr(model.config, "name_or_path", None),
    }
    rollout_files: dict[str, list[str]] = {}
    for arm in ("pos", "neg"):
        dumped = _dump_rollouts(trait, arm, rollouts_by_arm[arm], out_dir, sampling)
        rollout_files[arm] = [p.name for p in dumped]
        logger.info(
            "[%s/%s] rollout text persisted (%d files): %s",
            trait,
            arm,
            len(dumped),
            [p.name for p in dumped],
        )

    counts = {"trait": trait, "arms": {}, "rollout_files": rollout_files}

    # Judge each arm's rollouts; filter POS>50 / NEG<50, DROP the rest.
    kept_rollouts: dict[str, list[dict]] = {"pos": [], "neg": []}
    for arm in ("pos", "neg"):
        arm_rollouts = rollouts_by_arm[arm]
        n_total = sum(len(comps) for q in arm_rollouts.values() for comps in q.values())
        draw_stats: dict = {}
        if do_judge:
            save_raw = out_dir / f"judge_{trait}_{arm}.json"
            cache_dir = out_dir / "judge_cache"
            scores, draw_stats = _judge_rollouts(trait, arm_rollouts, save_raw, cache_dir)
        else:
            scores = {}  # no-judge smoke: keep all (labeled below)

        n_kept = n_dropped = n_below = 0
        # Enumeration shared with the rollout-text dump (_iter_rollout_records):
        # GLOBAL question index, 0 before the persona loop, +1 per question,
        # never reset per persona — mirrors batch_judge._enumerate_and_check_cache
        # AND C.judge_rollouts_n5's returned-key derivation, so multi-persona
        # score lookups resolve for every persona (concern
        # judge-n5-custom-id-index-mismatch-multipersona) and the dumped records'
        # custom_ids pair to these lookups by construction. Do NOT re-derive the
        # enumeration inline or change it without changing the wrapper in lockstep.
        for persona, _q_idx, question, _ci, comp, custom_id in _iter_rollout_records(arm_rollouts):
            if do_judge:
                s = scores.get(custom_id)
                if s is None:
                    n_dropped += 1
                    continue
                # threshold: POS>50 keep; NEG<50 keep.
                if arm == "pos" and not (s > 50.0):
                    n_below += 1
                    continue
                if arm == "neg" and not (s < 50.0):
                    n_below += 1
                    continue
            kept_rollouts[arm].append({"persona": persona, "question": question, "response": comp})
            n_kept += 1
        counts["arms"][arm] = {
            "total": n_total,
            "kept": n_kept,
            "dropped_refusal_or_invalid": n_dropped,
            "dropped_below_threshold": n_below,
            "judge_draw_stats": draw_stats,  # N=5 per-draw drop distribution
        }
        logger.info(
            "[%s/%s] judged: total=%d kept=%d dropped(refusal/invalid)=%d dropped(threshold)=%d",
            trait,
            arm,
            n_total,
            n_kept,
            n_dropped,
            n_below,
        )

    # Response-avg activation capture, stream-reduced per arm.
    means: dict[str, RunningMean] = {}
    for arm in ("pos", "neg"):
        rm = RunningMean(n_layers, hidden)
        for r in kept_rollouts[arm]:
            act = _response_mean_activation(
                model, tokenizer, sysprompt_of[r["persona"]], r["question"], r["response"], layers
            )
            if act is not None:
                rm.add(act)
        means[arm] = rm
        counts["arms"][arm]["captured"] = rm.count

    # r_B[layer] = mean(kept POS) - mean(kept NEG).
    assert means["pos"].count > 0 and means["neg"].count > 0, (
        f"{trait}: zero kept rollouts in an arm (pos={means['pos'].count}, "
        f"neg={means['neg'].count}); cannot form r_B — the judge-filter dropped an "
        "entire arm (report as a yield failure, do NOT fabricate a direction)"
    )
    r_b = means["pos"].mean() - means["neg"].mean()  # (L, H)
    assert r_b.shape == (n_layers, hidden), r_b.shape

    rb_dir = out_dir / "r_b"
    rb_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trait": trait,
            "r_b": r_b,  # (L, H) block-index layers
            "layers": layers,
            "counts": counts,
            "smoke": smoke,
            "metadata": C.reproducibility_metadata({"script": "issue779_extract_rb"}),
        },
        rb_dir / f"{trait}.pt",
    )
    C.write_json_atomic(rb_dir / f"{trait}_counts.json", counts)
    logger.info(
        "[%s] r_b (%s) written; pos/neg captured %d/%d",
        trait,
        tuple(r_b.shape),
        means["pos"].count,
        means["neg"].count,
    )
    return counts


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 persona-vector r_B extraction.")
    parser.add_argument(
        "--stage",
        choices=["all", "artifacts"],
        default="all",
        help="all = generate artifacts + extract r_B; artifacts = only generate "
        "syc/halluc artifacts via Sonnet (no GPU).",
    )
    parser.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    parser.add_argument("--model", default=C.DEFAULT_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_779")
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--n-ext-q", type=int, default=20)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--expected-layers", type=int, default=C.EXPECTED_LAYERS)
    parser.add_argument("--expected-hidden", type=int, default=C.EXPECTED_HIDDEN)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip judge-filter (CPU smoke only; keeps all rollouts, labeled).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny slice: 1 trait x 1 pair x n-ext-q(=2) x 5 rollouts into "
        "data/issue_779_smoke/. IDENTICAL code path.",
    )
    parser.add_argument("--no-upload", action="store_true", help="skip HF upload.")
    args = parser.parse_args()

    C.phase("artifacts")
    # Generate syc/halluc artifacts (one Sonnet call each; evil is verbatim).
    traits = args.traits
    if args.smoke:
        traits = args.traits[:1]
    for trait in traits:
        if trait != "evil" and args.stage in ("all", "artifacts"):
            C.generate_extraction_artifacts(trait)
    if args.stage == "artifacts":
        logger.info("Artifacts stage complete for %s", traits)
        return 0

    # Smoke caps.
    n_pairs = 1 if args.smoke else args.n_pairs
    n_ext_q = 2 if args.smoke else args.n_ext_q
    n_rollouts = 5 if args.smoke else args.n_rollouts
    do_judge = not args.no_judge

    out_dir = Path(str(args.out_dir) + "_smoke") if args.smoke else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    C.phase("load_model")
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    assert n_layers == args.expected_layers, (
        f"model has {n_layers} decoder layers, expected {args.expected_layers}"
    )
    assert hidden == args.expected_hidden, (
        f"model hidden_size {hidden}, expected {args.expected_hidden}"
    )
    layers = list(range(n_layers))

    # vLLM engine (one, reused across traits). On CPU smoke, generation uses HF
    # via a lightweight path (no vLLM CPU engine) — but we keep the code path
    # unified by requiring cuda for the real run; the CPU smoke uses HF generate.
    C.phase("extract")
    all_counts = {}
    t0 = time.time()
    if use_cuda:
        from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

        llm = create_vllm_engine(args.model, max_model_len=2048, seed=42)
        try:
            for trait in traits:
                all_counts[trait] = extract_trait_rb(
                    trait,
                    model,
                    tokenizer,
                    llm,
                    layers,
                    out_dir,
                    n_pairs=n_pairs,
                    n_ext_q=n_ext_q,
                    n_rollouts=n_rollouts,
                    do_judge=do_judge,
                    smoke=args.smoke,
                )
        finally:
            cleanup_vllm(llm)
    else:
        # CPU smoke: generate rollouts with HF model.generate (no vLLM on CPU).
        llm = _HFGenShim(model, tokenizer)
        for trait in traits:
            all_counts[trait] = extract_trait_rb(
                trait,
                model,
                tokenizer,
                llm,
                layers,
                out_dir,
                n_pairs=n_pairs,
                n_ext_q=n_ext_q,
                n_rollouts=n_rollouts,
                do_judge=do_judge,
                smoke=args.smoke,
            )
    logger.info("r_B extraction done in %.1f min", (time.time() - t0) / 60)

    # Upload r_b tensors + counts (small; non-LFS + tiny .pt).
    if not args.no_upload:
        C.phase("upload")
        _upload_rb(out_dir / "r_b", smoke=args.smoke)
        _upload_rollout_text(out_dir, smoke=args.smoke)

    note = (
        f"issue779 r_B extraction {'SMOKE ' if args.smoke else ''}complete: "
        f"traits={traits}, layers={n_layers}, counts={all_counts}"
    )
    C.write_sentinel("epm:smoke-result" if args.smoke else "epm:results", note)
    C.phase("done")
    return 0


class _HFGenShim:
    """Minimal vLLM-generate stand-in for the CPU smoke (HF model.generate).

    Exposes .generate(prompt_texts, sampling_params, use_tqdm=...) returning
    objects with .outputs[i].text, matching what _vllm_generate_chunked reads.
    CPU-only; the real GPU path uses vLLM. Keeps the smoke exercising the SAME
    extract_trait_rb code (generation shape + judge + capture), just with an HF
    generation backend where vLLM has no CPU engine.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt_texts, sampling_params, use_tqdm=False):
        import types

        results = []
        for text in prompt_texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            outs = []
            for _ in range(sampling_params.n):
                with torch.no_grad():
                    gen = self.model.generate(
                        **inputs,
                        max_new_tokens=min(64, sampling_params.max_tokens),
                        do_sample=True,
                        temperature=sampling_params.temperature,
                        top_p=sampling_params.top_p,
                    )
                new = gen[0, inputs["input_ids"].shape[1] :]
                outs.append(
                    types.SimpleNamespace(text=self.tokenizer.decode(new, skip_special_tokens=True))
                )
            results.append(types.SimpleNamespace(outputs=outs))
        return results


def _upload_rb(rb_dir: Path, smoke: bool) -> None:
    """Bulk-upload r_b tensors + counts to the HF data repo, verified."""
    from huggingface_hub import HfApi, list_repo_files

    api = HfApi()
    sub = "smoke_r_b" if smoke else "r_b"
    path_in_repo = f"{C.HF_PREFIX}/{sub}"
    api.upload_folder(
        folder_path=str(rb_dir),
        path_in_repo=path_in_repo,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        commit_message=f"issue779: {'smoke ' if smoke else ''}r_B tensors",
    )
    files = [
        f
        for f in list_repo_files(C.HF_DATA_REPO, repo_type="dataset")
        if f.startswith(path_in_repo)
    ]
    n_pt = sum(1 for f in files if f.endswith(".pt"))
    if n_pt < 1:
        raise RuntimeError(f"r_B upload verification failed: no .pt under {path_in_repo}")
    logger.info("r_B upload verified: %d files under %s", len(files), path_in_repo)


def _upload_rollout_text(out_dir: Path, smoke: bool) -> None:
    """Bulk-upload the persisted rollout text to the HF data repo, verified.

    Persist-by-default (upload-policy.md; the #779 rollout-text incident): the
    rollout dumps ride the non-LFS path and upload unconditionally alongside
    the r_B tensors, ONE upload_folder commit (never a per-file loop). Verify
    is the server-side-SCOPED exact-set helper (a bare list_repo_files on the
    ~1M-file data repo times out — gotchas.md #833). Fail-loud on a missing
    dump (rollout text is never discardable).
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    rc_dir = out_dir / "raw_completions"
    files = sorted(rc_dir.glob("rollouts_*.json")) if rc_dir.exists() else []
    if not files:
        raise RuntimeError(
            f"no rollout text under {rc_dir} — extract_trait_rb must persist rollout "
            "dumps before upload (rollout text is never discardable)"
        )
    api = HfApi()
    sub = "smoke_raw_completions" if smoke else "raw_completions"
    path_in_repo = f"{C.HF_PREFIX}/{sub}/rb_extraction"
    api.upload_folder(
        folder_path=str(rc_dir),
        path_in_repo=path_in_repo,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["rollouts_*.json"],
        commit_message=f"issue779: {'smoke ' if smoke else ''}r_B extraction rollout text",
    )
    missing = hub.verify_repo_paths_uploaded(
        api,
        C.HF_DATA_REPO,
        [f"{path_in_repo}/{f.name}" for f in files],
        path_in_repo=path_in_repo,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"rollout-text upload verification failed; missing: {missing}")
    logger.info("rollout text upload verified: %d files under %s", len(files), path_in_repo)


if __name__ == "__main__":
    sys.exit(main())
