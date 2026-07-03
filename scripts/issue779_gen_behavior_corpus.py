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
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_collect as COL  # noqa: E402
import issue779_common as C  # noqa: E402
import torch  # noqa: E402

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

        # 1. Capture c_last per context.
        prompt_texts = []
        cx_last, cx_mean = [], []
        for ctx in contexts:
            cx = COL.capture_context_vector(model, tokenizer, ctx["messages"], layers)
            cx_last.append(cx["last"])
            cx_mean.append(cx["mean"])
            prompt_texts.append(
                tokenizer.apply_chat_template(
                    ctx["messages"], tokenize=False, add_generation_prompt=True
                )
            )

        # 2. Generate n_rollouts per context (chunked vLLM / CPU shim).
        gen = COL._vllm_generate_chunked(llm, prompt_texts, sp)  # per-context list of n

        # 3. Per rollout: v(x) = mean-response activation (all layers), one
        #    teacher-force. KEEP both trait-high and trait-low (no filter).
        r_b = r_b_by_trait[trait].to(torch.float32)  # (L, H)
        # v_x_stack: (n_contexts, n_valid_rollouts_per_ctx?) — variable per ctx;
        # store per-rollout with an (context_idx, rollout_idx) index so alignment
        # to the judge score is unambiguous.
        vx_records: list[torch.Tensor] = []  # each (L, H)
        vx_index: list[tuple[int, int]] = []  # (context_idx, rollout_idx)
        judge_completions: dict[str, dict[str, list[str]]] = {}
        for ci, (ctx, comps) in enumerate(zip(contexts, gen, strict=True)):
            persona_key = f"p{ctx['persona_idx']:03d}"
            qkey = f"q{ctx['question_idx']:03d}"
            # batch_judge shape is {persona: {question: [completions]}}; the
            # corpus persona is the outer key, question the inner. Use a
            # context-unique persona key so global-idx enumeration is 1:1 with ci.
            judge_completions.setdefault(f"{persona_key}_{qkey}", {})[ctx["question"]] = comps
            for ri, comp in enumerate(comps):
                av = COL.capture_answer_vector(
                    model,
                    tokenizer,
                    ctx["messages"],
                    comp,
                    layers,
                    {trait: r_b},
                    keep_per_token=False,
                )
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


def _upload_corpus(out_dir: Path, smoke: bool) -> None:
    """Upload corpus tensors + judge scores (sanitized) + rollout TEXT + g labels.

    Plan §10: the c_last + v(x) tensors + judge SCALARS -> behavior_corpus/
    (NO rollout text); the rollout TEXT -> behavior_raw_completions/ (canonical
    raw-completions prefix, verified); the LMSYS g labels -> lmsys_g_labels/.
    Fail-loud on any upload mismatch (a clean exit IS the upload contract).
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
        (staging / CORPUS_SUBDIR).mkdir(parents=True, exist_ok=True)
        wrote_tensor = False
        for p in sorted((out_dir / CORPUS_SUBDIR).glob("*")):
            if p.name.endswith("_rollouts.json"):
                continue  # raw text -> raw-completions prefix only
            if p.name.startswith("judge_raw_"):
                continue  # raw judge model text -> omit
            (staging / CORPUS_SUBDIR / p.name).write_bytes(p.read_bytes())
            wrote_tensor = True
        g_src = out_dir / LMSYS_G_SUBDIR
        if g_src.is_dir():
            (staging / LMSYS_G_SUBDIR).mkdir(parents=True, exist_ok=True)
            for p in sorted(g_src.glob("*.json")):
                if p.name.startswith("judge_raw_"):
                    continue
                (staging / LMSYS_G_SUBDIR / p.name).write_bytes(p.read_bytes())
        if not wrote_tensor:
            raise RuntimeError(
                f"corpus upload aborted: no tensor bundles under {out_dir / CORPUS_SUBDIR}"
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 Arm B/C behavior-corpus generation.")
    parser.add_argument(
        "--stage",
        choices=["all", "corpus", "lmsys_g"],
        default="all",
        help="corpus = Arm B/C diverse corpus; lmsys_g = Arm A g labels (regen)",
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
        cand = rb_dir / f"{trait}.pt"
        if not cand.exists():
            cand = Path(str(args.out_dir)) / "r_b" / f"{trait}.pt"
        if not cand.exists():
            raise FileNotFoundError(
                f"r_B for {trait} not found ({rb_dir}/{trait}.pt); stage r_b/ first"
            )
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
        _upload_corpus(out_dir, smoke=args.smoke)

    C.write_json_atomic(out_dir / "corpus_gen_summary.json", summary)
    tag = "SMOKE " if args.smoke else ""
    note = f"issue779 corpus-gen {tag}complete: {json.dumps(summary)[:2000]}"
    C.write_sentinel(
        "epm:smoke-result" if args.smoke else "epm:results",
        note,
        extra={"gpu_hours_used": None},
    )
    C.phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
