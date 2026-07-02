#!/usr/bin/env python3
"""Issue #779 (training-source-ablation-hg): Arm B/C diverse behavior corpus gen.

The ONLY GPU phase of the amendment (plan v6 §4.2 / §9). Generates the NEW,
deliberately-diverse trait-eliciting corpus for the h/g training-source ablation
and captures the per-context tensors the 2D scaling grid subsamples:

  Phase [corpus] (Arm B/C training corpus, on-policy, behavior-VARYING) --
    ~60 diverse trait-eliciting persona system prompts x ~40 varied questions x
    N rollouts, per trait, sampled on-policy (temp 1.0) from Qwen-2.5-7B-Instruct.
    KEEP both trait-high and trait-low completions (NO positive-only filter) so
    the answer profiles span r_B in BOTH directions. Per rollout capture:
      - c_last (last-prompt-token activation, ALL 28 layers) via the OOM-safe
        hook (reuses issue779_collect.capture_context_vector);
      - v(x) (mean-response activation, ALL 28 layers) via one teacher-force pass
        (reuses issue779_collect.capture_answer_vector);
      - the N=5 graded 0-100 judge score (Anthropic Batch API, DROP-NEVER-COERCE).

  Phase [lmsys_g_labels] (Arm A direct-predictor labels) --
    The cached pass_b LMSYS bundle carries ONLY tensors (cx_last / cx_mean / v_x);
    the rollout TEXT + prompts were stripped at the parent's sanitize-for-upload
    step and were NEVER uploaded, so the Arm A g labels CANNOT be produced 0-GPU
    from the cache (contra the plan §4.1 "0-GPU judge pass" claim). This phase
    REGENERATES the 5000 LMSYS rollouts (1/context, temp 1.0 seed 42) from the
    deterministically-reloaded prompts IN THE SAME ORDER as the parent pass_b
    (issue779_collect.load_train_contexts), judges them (N=5 graded), and writes
    per-context labels aligned by index to the cached pass_b cx_last. This is a
    persisted-concern deviation (arm-a-g-labels-require-regen); it rides the
    corpus pod (same model already loaded) and is cheap. It does NOT gate the
    h reads (h is behavior-agnostic, needs no labels).

DP over --gpu-id workers (a trait-sharded worker layout, one GPU each), exactly
like issue779_collect.py: the launcher pins CUDA_VISIBLE_DEVICES=<gpu> per worker
and passes the matching --gpu-id. --smoke runs the IDENTICAL path on a tiny slice
(2 personas x 2 questions x 2 rollouts x 1 trait) with the CPU HF-generate shim
where vLLM has no CPU engine -- smoke IS the production path with a tiny slice.

Upload (before pod teardown; plan §10): the corpus c_last + v(x) tensors +
per-rollout judge scores land under
``issue779_monitoring/training-source-ablation-hg/behavior_corpus/`` (sanitized:
activations + SCALARS only, NO rollout text), the rollout TEXT lands under the
canonical ``.../behavior_raw_completions/`` prefix (verified via list_repo_files),
and the LMSYS g labels under ``.../lmsys_g_labels/``. Fail-loud on any upload
mismatch (a clean exit IS the upload contract).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_collect as COL  # noqa: E402
import issue779_common as C  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_gen_behavior_corpus")

# HF layout for the amendment's NEW artifacts (plan §10 / §9).
HF_ROUND_PREFIX = f"{C.HF_PREFIX}/training-source-ablation-hg"
CORPUS_SUBDIR = "behavior_corpus"
CORPUS_RAWCOMP_SUBDIR = "behavior_raw_completions"
LMSYS_G_SUBDIR = "lmsys_g_labels"


def build_corpus_contexts(trait: str, personas: list[str], questions: list[str]) -> list[dict]:
    """Enumerate the (persona x question) corpus contexts for a trait.

    Each context is a system-prompt-style single-turn chat: {"role":"system",
    persona} + {"role":"user", question}. Returns [{persona_idx, question_idx,
    persona, question, messages}]. This mirrors the eval rig's SYSTEM mode shape
    (build_eval_prompt_messages system branch) so the captured c_last position
    (assistant-header suffix) matches the parent's #594 control.
    """
    contexts = []
    for pi, persona in enumerate(personas):
        for qi, question in enumerate(questions):
            messages = [
                {"role": "system", "content": persona},
                {"role": "user", "content": question},
            ]
            contexts.append(
                {
                    "persona_idx": pi,
                    "question_idx": qi,
                    "persona": persona,
                    "question": question,
                    "messages": messages,
                }
            )
    return contexts


def run_corpus_phase(
    model,
    tokenizer,
    llm,
    layers: list[int],
    r_b_by_trait: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    traits: list[str],
    n_personas: int,
    n_questions: int,
    n_rollouts: int,
    dry_run_judge: bool,
) -> dict:
    """Generate + capture + judge the diverse behavior corpus, per trait.

    Checkpoint-per-trait: each trait writes its own tensor bundle + rollout-text
    JSON + judge JSON the moment it completes (never accumulate-and-write-at-end).
    Keeps BOTH trait-high and trait-low completions (behavior-VARYING).
    """
    from vllm import SamplingParams

    corpus_dir = out_dir / CORPUS_SUBDIR
    corpus_dir.mkdir(parents=True, exist_ok=True)
    sp = SamplingParams(n=n_rollouts, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42)

    # Batched-rewrite equivalence gate (MAJOR 6): fail fast if the right-padded
    # batched capture diverges from the batch-1 serial capture before the sweep
    # corrupts all activations. Uses a handful of layers to keep the check cheap.
    check_layers = layers if len(layers) <= 4 else [layers[0], layers[len(layers) // 2], layers[-1]]
    eq = COL.assert_batched_capture_equivalence(model, tokenizer, check_layers)
    logger.info("[corpus] batched-capture equivalence OK: %s", eq)

    produced: dict[str, dict] = {}
    for trait in traits:
        bundle_path = corpus_dir / f"{trait}_corpus.pt"
        scores_path = corpus_dir / f"{trait}_judge_scores.json"
        text_path = corpus_dir / f"{trait}_rollouts.json"
        if bundle_path.exists() and scores_path.exists() and text_path.exists():
            logger.info("[corpus] %s already complete; skip", trait)
            with open(scores_path) as f:
                produced[trait] = json.load(f)["summary"]
            continue

        spec = C.generate_behavior_corpus_spec(
            trait, n_personas=n_personas, n_questions=n_questions
        )
        # Disjointness is asserted inside generate_behavior_corpus_spec (§4.2);
        # re-assert here defensively before any generation consumes the spec.
        C.assert_corpus_disjoint(trait, spec["personas"], spec["questions"])
        contexts = build_corpus_contexts(trait, spec["personas"], spec["questions"])
        logger.info(
            "[corpus] %s: %d personas x %d questions = %d contexts x %d rollouts",
            trait,
            len(spec["personas"]),
            len(spec["questions"]),
            len(contexts),
            n_rollouts,
        )

        # 1. Capture c_last per context (BATCHED — MAJOR 6; batch-1 forwards over
        #    thousands of contexts are weight-bandwidth-bound).
        messages_list = [ctx["messages"] for ctx in contexts]
        cx_all = COL.capture_context_vectors_batched(model, tokenizer, messages_list, layers)
        cx_last = [c["last"] for c in cx_all]
        cx_mean = [c["mean"] for c in cx_all]
        prompt_texts = [
            tokenizer.apply_chat_template(
                ctx["messages"], tokenize=False, add_generation_prompt=True
            )
            for ctx in contexts
        ]

        # 2. Generate n_rollouts per context (chunked vLLM / CPU shim).
        gen = COL._vllm_generate_chunked(llm, prompt_texts, sp)  # per-context list of n

        # 3. Per rollout: v(x) = mean-response activation (all layers), one
        #    teacher-force. KEEP both trait-high and trait-low (no filter).
        #    BATCHED (MAJOR 6): build the full (context, rollout) item list, then
        #    capture v(x) in right-padded batches. The vx_index preserves the
        #    (context_idx, rollout_idx) alignment (empty completions -> None -> skip,
        #    never coerced — same drop rule as the serial path).
        r_b = r_b_by_trait[trait].to(torch.float32)  # (L, H)
        vx_records: list[torch.Tensor] = []  # each (L, H)
        vx_index: list[tuple[int, int]] = []  # (context_idx, rollout_idx)
        judge_completions: dict[str, dict[str, list[str]]] = {}
        answer_items: list[tuple[list[dict], str]] = []
        answer_keys: list[tuple[int, int]] = []  # (context_idx, rollout_idx) per item
        for ci, (ctx, comps) in enumerate(zip(contexts, gen, strict=True)):
            persona_key = f"p{ctx['persona_idx']:03d}"
            qkey = f"q{ctx['question_idx']:03d}"
            # batch_judge shape is {persona: {question: [completions]}}; the
            # corpus persona is the outer key, question the inner. Use a
            # context-unique persona key so global-idx enumeration is 1:1 with ci.
            judge_completions.setdefault(f"{persona_key}_{qkey}", {})[ctx["question"]] = comps
            for ri, comp in enumerate(comps):
                answer_items.append((ctx["messages"], comp))
                answer_keys.append((ci, ri))
        avs = COL.capture_answer_vectors_batched(
            model, tokenizer, answer_items, layers, {trait: r_b}, keep_per_token=False
        )
        for (ci, ri), av in zip(answer_keys, avs, strict=True):
            if av is None:  # empty completion — no v(x); skip (never coerce)
                continue
            vx_records.append(av["v_x"])  # (L, H)
            vx_index.append((ci, ri))

        # 4. Judge every rollout (graded 0-100, N=5, DROP-NEVER-COERCE).
        save_raw = corpus_dir / f"judge_raw_{trait}.json"
        judged = C.judge_rollouts_n5(trait, judge_completions, save_raw, dry_run=dry_run_judge)

        # Map judged scores back to (context_idx, rollout_idx). judge_rollouts_n5
        # keys are f"{persona}__{global_idx:05d}__{ri:02d}" where persona is our
        # f"{persona_key}_{qkey}" and global_idx counts questions across ALL
        # personas (one question per our persona-key => global_idx == that key's
        # position in judge_completions insertion order == ci). We rebuild the
        # ci -> score map from the SAME insertion order.
        ci_scores: dict[int, dict[int, float | None]] = {}
        if not dry_run_judge:
            keys_in_order = list(judge_completions.keys())
            for gi, jc_key in enumerate(keys_in_order):
                # each jc_key holds exactly one question => one rollout list of len n
                for ri in range(n_rollouts):
                    cid = f"{jc_key}__{gi:05d}__{ri:02d}"
                    mean_n = judged.get(cid)
                    ci_scores.setdefault(gi, {})[ri] = mean_n[0] if mean_n else None

        # 5. Persist per-trait: the c_last + v_x tensors + the (context) index +
        #    judge scores. Checkpoint the moment the trait completes.
        n_layers = len(layers)
        vx_tensor = (
            torch.stack(vx_records) if vx_records else torch.empty(0, n_layers, r_b.shape[1])
        )
        torch.save(
            {
                "trait": trait,
                "layers": layers,
                "cx_last": torch.stack(cx_last),  # (n_contexts, L, H)
                "cx_mean": torch.stack(cx_mean),  # (n_contexts, L, H)
                "v_x": vx_tensor,  # (n_valid_rollouts, L, H)
                "vx_index": vx_index,  # [(context_idx, rollout_idx)]
                "persona_idx": [ctx["persona_idx"] for ctx in contexts],
                "question_idx": [ctx["question_idx"] for ctx in contexts],
                "n_personas": len(spec["personas"]),
                "n_questions": len(spec["questions"]),
                "n_rollouts": n_rollouts,
                "metadata": C.reproducibility_metadata(
                    {"script": "issue779_gen_behavior_corpus", "phase": "corpus", "trait": trait}
                ),
            },
            bundle_path,
        )
        # judge scores as {context_idx: {rollout_idx: score|null}}
        scores_json = {
            str(ci): {str(ri): s for ri, s in rmap.items()} for ci, rmap in ci_scores.items()
        }
        n_valid = sum(1 for rmap in ci_scores.values() for s in rmap.values() if s is not None)
        n_total = sum(len(rmap) for rmap in ci_scores.values())
        summary = {
            "trait": trait,
            "n_contexts": len(contexts),
            "n_valid_scores": n_valid,
            "n_total_rollouts": n_total,
            "n_vx_captured": len(vx_index),
        }
        C.write_json_atomic(scores_path, {"scores": scores_json, "summary": summary})
        # rollout TEXT (for the canonical raw-completions upload; never mixed
        # into the sanitized tensor bundle).
        rollout_text = {
            str(ci): {
                "persona_idx": ctx["persona_idx"],
                "question_idx": ctx["question_idx"],
                "question": ctx["question"],
                "responses": comps,
            }
            for ci, (ctx, comps) in enumerate(zip(contexts, gen, strict=True))
        }
        C.write_json_atomic(text_path, {"trait": trait, "rollouts": rollout_text})
        produced[trait] = summary
        logger.info("[corpus] %s done: %s", trait, summary)
    return produced


def run_lmsys_g_labels_phase(
    model,
    tokenizer,
    llm,
    out_dir: Path,
    *,
    traits: list[str],
    n_contexts: int,
    smoke: bool,
    dry_run_judge: bool,
) -> dict:
    """Arm A direct-predictor g labels: regenerate LMSYS rollouts, judge them.

    The cached pass_b bundle has NO rollout text (stripped at the parent's
    sanitize step; never uploaded) -- so labels must be REGENERATED, not read.
    Reloads the SAME LMSYS prompts IN THE SAME ORDER as the parent pass_b
    (load_train_contexts is deterministic: streaming, first-N), generates 1
    rollout/context (temp 1.0 seed 42, matching pass_b), judges N=5 graded, and
    writes per-context labels ALIGNED BY INDEX to the cached pass_b cx_last[i].

    Report the label std per trait (the plan §7 label-floor diagnostic: LMSYS has
    near-zero trait base rate, so Arm A g may be floored -- a legitimate finding).
    """
    from vllm import SamplingParams

    g_dir = out_dir / LMSYS_G_SUBDIR
    g_dir.mkdir(parents=True, exist_ok=True)
    done_path = g_dir / "lmsys_g_labels.json"
    if done_path.exists():
        logger.info("[lmsys_g_labels] already complete; skip")
        with open(done_path) as f:
            return json.load(f)["summary"]

    prompts, source = COL.load_train_contexts(n_contexts, smoke)
    logger.info(
        "[lmsys_g_labels] %d LMSYS contexts from %s (regen for g labels)", len(prompts), source
    )
    prompt_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42)
    gen = COL._vllm_generate_chunked(llm, prompt_texts, sp)  # per-context [1]

    # judge each trait over the SAME regenerated LMSYS completions.
    # batch_judge shape: one persona key "lmsys", question = the LMSYS prompt.
    per_trait: dict[str, dict] = {}
    for trait in traits:
        judge_completions: dict[str, dict[str, list[str]]] = {"lmsys": {}}
        for p, comps in zip(prompts, gen, strict=True):
            judge_completions["lmsys"][p] = comps  # [1 completion]
        save_raw = g_dir / f"judge_raw_lmsys_{trait}.json"
        judged = C.judge_rollouts_n5(trait, judge_completions, save_raw, dry_run=dry_run_judge)
        labels: list[float | None] = []
        if not dry_run_judge:
            for gi in range(len(prompts)):
                cid = f"lmsys__{gi:05d}__00"
                mean_n = judged.get(cid)
                labels.append(mean_n[0] if mean_n else None)
        valid = [x for x in labels if x is not None]
        import numpy as np

        label_std = float(np.std(valid)) if len(valid) >= 2 else float("nan")
        per_trait[trait] = {
            "labels": labels,  # aligned by index to pass_b cx_last[i]
            "n_valid": len(valid),
            "n_total": len(labels),
            "label_std": label_std,
        }
        logger.info(
            "[lmsys_g_labels] %s: n_valid=%d label_std=%.3f (floor check)",
            trait,
            len(valid),
            label_std,
        )

    summary = {
        "source": source,
        "n_contexts": len(prompts),
        "label_std_per_trait": {t: per_trait[t]["label_std"] for t in traits},
        "n_valid_per_trait": {t: per_trait[t]["n_valid"] for t in traits},
    }
    C.write_json_atomic(
        done_path,
        {
            "labels_per_trait": per_trait,
            "summary": summary,
            "metadata": C.reproducibility_metadata(
                {"script": "issue779_gen_behavior_corpus", "phase": "lmsys_g_labels"}
            ),
        },
    )
    logger.info("[lmsys_g_labels] done: %s", summary)
    return summary


def _stage_sanitized_corpus(out_dir: Path, staging: Path) -> tuple[bool, bool]:
    """Copy the sanitized upload set (tensor bundles + judge scalars + g labels,
    NO rollout text, NO raw judge text) from out_dir into the staging dir.

    Returns ``(wrote_tensor, wrote_g)`` — whether any corpus tensor bundle /
    any lmsys_g label file was staged (the fail-loud gates key on these).
    """
    (staging / CORPUS_SUBDIR).mkdir(parents=True, exist_ok=True)
    wrote_tensor = False
    for p in sorted((out_dir / CORPUS_SUBDIR).glob("*")):
        if p.name.endswith("_rollouts.json"):
            continue  # raw text -> raw-completions prefix only
        if p.name.startswith("judge_raw_"):
            continue  # raw judge model text -> omit
        (staging / CORPUS_SUBDIR / p.name).write_bytes(p.read_bytes())
        wrote_tensor = True
    wrote_g = False
    g_src = out_dir / LMSYS_G_SUBDIR
    if g_src.is_dir():
        (staging / LMSYS_G_SUBDIR).mkdir(parents=True, exist_ok=True)
        for p in sorted(g_src.glob("*.json")):
            if p.name.startswith("judge_raw_"):
                continue
            (staging / LMSYS_G_SUBDIR / p.name).write_bytes(p.read_bytes())
            wrote_g = True
    return wrote_tensor, wrote_g


def _upload_corpus(out_dir: Path, smoke: bool, *, require_corpus: bool = True) -> None:
    """Upload corpus tensors + judge scores (sanitized) + rollout TEXT + g labels.

    Plan §10: the c_last + v(x) tensors + judge SCALARS -> behavior_corpus/
    (NO rollout text); the rollout TEXT -> behavior_raw_completions/ (canonical
    raw-completions prefix, verified); the LMSYS g labels -> lmsys_g_labels/.
    Fail-loud on any upload mismatch (a clean exit IS the upload contract).

    ``require_corpus=False`` (the standalone ``--stage lmsys_g`` path, which
    generates ONLY g labels) tolerates a missing corpus-tensor dir and uploads
    whatever exists — it still fails loud when there is NOTHING to upload.
    """
    import tempfile

    from huggingface_hub import HfApi, list_repo_files

    suffix = "_smoke" if smoke else ""
    api = HfApi()

    # 1. Sanitized tensor bundles + judge scores + g labels (NO rollout text).
    #    The corpus bundle .pt holds only tensors + numeric index; the
    #    *_rollouts.json (raw text) is uploaded SEPARATELY under the
    #    raw-completions prefix and is NOT included in this analysis-tensors push.
    with tempfile.TemporaryDirectory(prefix="issue779_corpus_") as tmp:
        staging = Path(tmp)
        wrote_tensor, wrote_g = _stage_sanitized_corpus(out_dir, staging)
        if not wrote_tensor and require_corpus:
            raise RuntimeError(
                f"corpus upload aborted: no tensor bundles under {out_dir / CORPUS_SUBDIR}"
            )
        if not wrote_tensor and not wrote_g:
            raise RuntimeError(
                f"upload aborted: NOTHING to upload under {out_dir} (no corpus tensor "
                f"bundles and no {LMSYS_G_SUBDIR} labels)"
            )
        path_in_repo = f"{HF_ROUND_PREFIX}{suffix}/analysis_tensors"
        api.upload_folder(
            folder_path=str(staging),
            path_in_repo=path_in_repo,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue779 {'smoke ' if smoke else ''}Arm B/C corpus tensors + g labels",
        )
        repo_files = set(list_repo_files(C.HF_DATA_REPO, repo_type="dataset"))
        # verify each staged file landed
        missing = []
        for src in sorted(staging.rglob("*")):
            if src.is_file():
                rel = src.relative_to(staging).as_posix()
                if f"{path_in_repo}/{rel}" not in repo_files:
                    missing.append(rel)
        if missing:
            raise RuntimeError(
                f"corpus tensor upload verification FAILED: {len(missing)} files "
                f"missing under {path_in_repo}: {missing[:10]}"
            )
        logger.info("corpus tensors + g labels upload verified under %s", path_in_repo)

    # 2. Rollout TEXT under the canonical raw-completions prefix (verified).
    text_files = sorted((out_dir / CORPUS_SUBDIR).glob("*_rollouts.json"))
    if text_files:
        with tempfile.TemporaryDirectory(prefix="issue779_rawcomp_") as tmp:
            staging = Path(tmp)
            written = []
            for p in text_files:
                trait = p.name.replace("_rollouts.json", "")
                fname = f"{trait}_corpus_seed42.json"
                (staging / fname).write_bytes(p.read_bytes())
                written.append(fname)
            rc_prefix = f"{HF_ROUND_PREFIX}{suffix}/{CORPUS_RAWCOMP_SUBDIR}"
            api.upload_folder(
                folder_path=str(staging),
                path_in_repo=rc_prefix,
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                commit_message=f"issue779 {'smoke ' if smoke else ''}Arm B corpus rollout text",
            )
            repo_files = set(list_repo_files(C.HF_DATA_REPO, repo_type="dataset"))
            missing = [f for f in written if f"{rc_prefix}/{f}" not in repo_files]
            if missing:
                raise RuntimeError(
                    f"corpus raw-completions upload verification FAILED "
                    f"(raw-completions-upload-prefix-missing): {missing}"
                )
            logger.info("corpus raw-completions upload verified under %s", rc_prefix)


def _finalize_worker(*, no_upload: bool, smoke: bool, summary: dict, stage: str) -> None:
    """End-of-run sentinel + phase line — the ONLY write_sentinel site in this module.

    BLOCKER (multigpu-no-upload-terminal-sentinel): a ``--no-upload`` worker is a
    PARTIAL trait shard — its corpus tensors / rollout text / g labels are NOT
    uploaded yet (the multi-GPU dispatcher's single post-join upload_only /
    lmsys_g worker owns the ONE whole-out_dir upload). Such a worker must NEVER
    write the terminal ``epm:results`` / ``epm:smoke-result`` sentinel nor the
    reserved ``[phase=done]`` line: a sentinel-driven poller/verifier would
    observe a false "done" before the post-join upload exists. It writes a
    clearly NON-terminal ``epm:progress`` shard artifact (``terminal: false``,
    ``blocks_pipeline: false``) + ``[phase=shard_done]`` instead. ONLY a worker
    that ran the upload (``no_upload=False``, incl. ``--stage upload_only``) is
    terminal. Pinned by tests/test_issue779_scaling_grid.py (behavior + AST:
    every write_sentinel call must live inside this function).
    """
    tag = "SMOKE " if smoke else ""
    if no_upload:
        note = (
            f"issue779 corpus-gen [{stage}] {tag}shard complete WITHOUT upload "
            f"(NON-terminal: the post-join upload/lmsys_g worker uploads and writes "
            f"the terminal sentinel): {json.dumps(summary)[:2000]}"
        )
        C.write_sentinel(
            "epm:progress",
            note,
            extra={"gpu_hours_used": None, "terminal": False, "blocks_pipeline": False},
        )
        C.phase("shard_done")
        return
    note = f"issue779 corpus-gen [{stage}] {tag}complete: {json.dumps(summary)[:2000]}"
    C.write_sentinel(
        "epm:smoke-result" if smoke else "epm:results",
        note,
        extra={"gpu_hours_used": None},
    )
    C.phase("done")


def _resolve_rb_path(trait: str, rb_dir: Path, out_dir_base: Path) -> Path:
    """Resolve the r_B tensor path for a trait: local layout first, HF fetch fallback.

    Local candidates (the standard layout the RunPod parent had on disk):
    ``rb_dir/<trait>.pt``, then ``out_dir_base/r_b/<trait>.pt``. When NEITHER
    exists — the GCP/SLURM git-clone lanes stage no ``data/`` (the att-20260702
    crash) — download ``issue779_monitoring/r_b/<trait>.pt`` from the HF data
    repo and MATERIALIZE it into ``rb_dir/<trait>.pt`` so every later phase
    sees the standard local layout. Fails loud (FileNotFoundError chaining the
    HF error) only when the HF fetch also fails. Returns the resolved path.
    """
    cand = rb_dir / f"{trait}.pt"
    if cand.exists():
        logger.info("[r_b] %s found locally: %s", trait, cand)
        return cand
    alt = out_dir_base / "r_b" / f"{trait}.pt"
    if alt.exists():
        logger.info("[r_b] %s found locally: %s", trait, alt)
        return alt
    hf_filename = f"{C.HF_PREFIX}/r_b/{trait}.pt"
    try:
        from huggingface_hub import hf_hub_download

        fetched = hf_hub_download(repo_id=C.HF_DATA_REPO, filename=hf_filename, repo_type="dataset")
    except Exception as e:
        raise FileNotFoundError(
            f"r_B for {trait} not found locally ({cand} / {alt}) AND the HF fetch of "
            f"{C.HF_DATA_REPO}/{hf_filename} (repo_type=dataset) failed: {e!r}. "
            "Stage r_b/ first (issue779_extract_rb.py) or fix HF access/token."
        ) from e
    rb_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fetched, cand)
    logger.info("[r_b] %s not found locally; fetched from HF %s -> %s", trait, hf_filename, cand)
    return cand


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 Arm B/C behavior-corpus generation.")
    parser.add_argument(
        "--stage",
        choices=["all", "corpus", "lmsys_g", "upload_only"],
        default="all",
        help=(
            "corpus = Arm B/C diverse corpus; lmsys_g = Arm A g labels (regen); "
            "upload_only = upload an already-generated out_dir (no GPU/model load) "
            "— used by the multi-GPU dispatcher's single post-join upload worker"
        ),
    )
    parser.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    parser.add_argument("--model", default=C.DEFAULT_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_779")
    parser.add_argument("--rb-dir", type=Path, default=None)
    parser.add_argument("--n-personas", type=int, default=C.CORPUS_N_PERSONAS)
    parser.add_argument("--n-questions", type=int, default=C.CORPUS_N_QUESTIONS)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--n-lmsys-contexts", type=int, default=5000)
    parser.add_argument("--expected-layers", type=int, default=C.EXPECTED_LAYERS)
    parser.add_argument("--expected-hidden", type=int, default=C.EXPECTED_HIDDEN)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument(
        "--dry-run-judge",
        action="store_true",
        help="skip the Batch-API judge calls (wiring-only smoke; scores empty)",
    )
    args = parser.parse_args()

    # Smoke caps (smoke IS the production path at tiny N).
    traits = args.traits[:1] if args.smoke else args.traits
    n_personas = 2 if args.smoke else args.n_personas
    n_questions = 2 if args.smoke else args.n_questions
    n_rollouts = 2 if args.smoke else args.n_rollouts
    n_lmsys = 4 if args.smoke else args.n_lmsys_contexts

    out_dir = Path(str(args.out_dir) + "_corpus_smoke") if args.smoke else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # upload_only: no generation, no model/vLLM load — upload an already-generated
    # out_dir once (the multi-GPU dispatcher's single post-join upload worker, so
    # the trait-sharded --no-upload corpus workers do NOT race the shared HF
    # prefix). Fail-loud if there is nothing to upload.
    if args.stage == "upload_only":
        C.phase("upload")
        if args.no_upload:
            raise ValueError("--stage upload_only is incompatible with --no-upload")
        _upload_corpus(out_dir, smoke=args.smoke)
        summary = {"traits": traits, "smoke": args.smoke, "stage": "upload_only"}
        # Per-stage summary filename: the shard/lmsys_g workers share this
        # out_dir, so writing the shared corpus_gen_summary.json here would
        # clobber the corpus stats (last-writer-wins).
        C.write_json_atomic(out_dir / "upload_summary.json", summary)
        _finalize_worker(no_upload=False, smoke=args.smoke, summary=summary, stage="upload_only")
        return 0

    # EARLY fail-loud preflight (seconds, BEFORE the ~90s model load): BOTH
    # generation phases read the PARENT extraction artifacts for the generated
    # traits (data/issue_779/artifacts/<trait>.json; evil is verbatim in code) —
    # the corpus phase via assert_corpus_disjoint -> rb_pos_prompts, and BOTH
    # phases via judge_rollouts_n5 -> trait_judge_system_prompt (the judge
    # rubric IS the artifact's eval_prompt, evaluated even under
    # --dry-run-judge). The git-clone lanes (GCP/SLURM) stage no data/ and these
    # artifacts are NOT on HF (verified via list_repo_files 2026-07-02), so a
    # missing file must fail HERE — not hours into --stage all after the evil
    # corpus completes. Do NOT auto-regenerate via
    # generate_extraction_artifacts: fresh Sonnet artifacts would differ from the
    # ones the parent's r_B was actually extracted with, making the disjointness
    # check + judge rubric silently WRONG (vacuous ground truth) rather than
    # failed. Every non-upload_only stage loads the model, so gate them all.
    for trait in traits:
        C.load_extraction_artifacts(trait)  # raises FileNotFoundError with the recipe
    logger.info("[preflight] extraction artifacts present for traits %s", traits)

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
    assert n_layers == args.expected_layers, (n_layers, args.expected_layers)
    assert hidden == args.expected_hidden, (hidden, args.expected_hidden)
    layers = list(range(n_layers))

    # Load r_B (REUSED unchanged; needed for the v(x) projection instrumentation).
    rb_dir = args.rb_dir or (out_dir / "r_b")
    r_b_by_trait = {}
    for trait in traits:
        cand = _resolve_rb_path(trait, rb_dir, Path(str(args.out_dir)))
        blob = torch.load(cand, weights_only=False)
        r_b = blob["r_b"]
        assert r_b.shape == (n_layers, hidden), (
            f"{trait} r_b shape {tuple(r_b.shape)} != ({n_layers}, {hidden}) — re-extract"
        )
        r_b_by_trait[trait] = r_b

    if use_cuda:
        from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

        llm = create_vllm_engine(args.model, max_model_len=8192, seed=42)
    else:
        import issue779_extract_rb as R

        llm = R._HFGenShim(model, tokenizer)

    summary: dict = {"traits": traits, "smoke": args.smoke}
    try:
        if args.stage in ("all", "corpus"):
            C.phase("corpus")
            summary["corpus"] = run_corpus_phase(
                model,
                tokenizer,
                llm,
                layers,
                r_b_by_trait,
                out_dir,
                traits=traits,
                n_personas=n_personas,
                n_questions=n_questions,
                n_rollouts=n_rollouts,
                dry_run_judge=args.dry_run_judge,
            )
        if args.stage in ("all", "lmsys_g"):
            C.phase("lmsys_g_labels")
            summary["lmsys_g_labels"] = run_lmsys_g_labels_phase(
                model,
                tokenizer,
                llm,
                out_dir,
                traits=traits,
                n_contexts=n_lmsys,
                smoke=args.smoke,
                dry_run_judge=args.dry_run_judge,
            )
    finally:
        if use_cuda:
            cleanup_vllm(llm)

    if not args.no_upload:
        C.phase("upload")
        # Standalone --stage lmsys_g generates ONLY g labels; tolerate a missing
        # corpus dir there (still fail-loud when nothing at all exists to upload).
        _upload_corpus(out_dir, smoke=args.smoke, require_corpus=args.stage != "lmsys_g")

    # Per-stage/per-shard summary filename: concurrent --no-upload trait shards
    # (and the post-join lmsys_g worker) share out_dir, so one shared filename
    # was last-writer-wins across workers.
    if args.no_upload:
        summary_name = f"corpus_gen_summary_{'_'.join(traits)}.json"
    elif args.stage == "lmsys_g":
        summary_name = "lmsys_g_summary.json"
    else:
        summary_name = "corpus_gen_summary.json"
    C.write_json_atomic(out_dir / summary_name, summary)
    _finalize_worker(no_upload=args.no_upload, smoke=args.smoke, summary=summary, stage=args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
