#!/usr/bin/env python
"""issue #2379 P3 + P4 — teacher-forced predictor captures, ceiling rollouts, and
map-fit corpora (Kwon re-elicitation).

Deliverable 2 of pre-split UNIT 2/4 (plan §4.2 P3 + P4). HF forward-pass capture
rig (batched vLLM generation for rollouts; per-row teacher-forced HF re-forward for
activations), fp16 storage, ALL 28 decoder layers. Five phases, one ``--phase`` per
invocation, per model (the per-model / per-phase sharding axis the pod dispatcher
fans across GPUs):

  grid          v_C = last-prompt-token residual state at all 28 layers, for every
                q in Q_sim x every context c in the trigger set (EM 18 / caps 20;
                p_inoc is in-bank, no separate addition).
  mu            mu_train  = streaming mean of v_C over ALL that model's inoculated
                training rows; mu_A_train = streaming mean of v_A (answer-token mean,
                teacher-forced prompt+gold) over the same rows. Per-row tensors are
                NEVER materialized (streaming reduce; §10 discard slot).
  ceiling       3 on-policy rollouts per (q, c) (temp 1.0, top_p 0.95, max_tokens
                1024 — the #779 convention), teacher-forced re-forward, v_A stored
                PER-ROLLOUT (the split-rollout reliability read in P7 needs them);
                rollout TEXT persisted to HF raw_completions/ceiling/.
  map_corpus    replay the EXACT 5,000 LMSYS first-turn prompts via the pinned #952
                recipe (reused, never re-implemented), 1 rollout each (temp 1.0,
                top_p 0.95, max_tokens 1024, seed 42 — #779 verbatim), teacher-forced
                capture of (v_C, v_A) all 28 layers -> one fp16 bundle per model
                (5000, 28, 3584) x 2; rollout TEXT persisted to raw_completions/
                map_corpus/. Asserts LMSYS prompts disjoint from Q_sim/Q_beh/triggers.
  text_baselines  BGE embeddings (BAAI/bge-large-en-v1.5) of all trigger prompts +
                p_inoc; lexical sims (token Jaccard, SequenceMatcher ratio, TF-IDF
                cosine). Pod-side CPU, trivial.

``--probe-access`` does a 1-row streaming read of the GATED lmsys/lmsys-chat-1m to
certify the pod HF token's grant BEFORE P4 (the P0 smoke leg).

LAYER-INDEXING CONVENTION (stated here, recorded in every bundle's metadata):
we capture via ``analysis.extraction.extract_layer_activations`` at BLOCK indices
0..27 — the EXACT helper + convention the reused #779/#2254 pass-B bundle was built
with — so stored index ``i`` == decoder block ``i`` == ``output_hidden_states[i+1]``
for i<27 (pre-final-norm at i=27, matching pass-B). The full hidden-state tuple has
29 entries (index 0 = embeddings); we deliberately do NOT read it directly, because
``output_hidden_states[28]`` is POST-final-norm and would diverge from pass-B at
stored index 27 — which is the caps pinned layer (L27). We store ALL 28 layers, so
the parent's pinned L16/L27 selection happens downstream (P5/P7) against this 0..27
axis (#2254's "layer 14" == stored index 14); the P7 symmetric per-layer curves are
the cross-check that the pins sit where the curves say.

Harmful-advice / real-corpus completions are referenced by path + count only.

Run (production, one model / one phase; CVD pinned by the dispatcher):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2379_capture.py \
        --phase grid --setting caps --model-name i2379_caps_es \
        --adapter adapters/issue2379_reelicit_caps_spanish
Run (LMSYS gated-access probe, no capture):
    uv run python scripts/issue2379_capture.py --probe-access
Run (CPU arg-validation / disjointness / indexing assert; no GPU):
    uv run python scripts/issue2379_capture.py --phase grid --setting caps \
        --model-name base --model Qwen/Qwen2.5-7B-Instruct --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "src" / "explore_persona_space"
    if not sentinel.is_dir():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} not found (parents[1]={root})")
    for p in (str(root), str(root / "src"), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# Shared rendering convention + bank loaders (single source of truth — assumption 9).
from issue2379_prep_data import P_INOC_CAPS, P_INOC_EM  # noqa: E402
from issue2379_sweep import (  # noqa: E402
    BASE_MODEL,
    SLUG,
    load_questions,
    load_triggers,
    render_context_messages,
)

logger = logging.getLogger("issue2379_capture")

EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
LMSYS_N_PROMPTS = 5000

# Ceiling + map-corpus rollout sampling (#779 convention verbatim).
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_TOP_P = 0.95
ROLLOUT_MAX_TOKENS = 1024
ROLLOUT_SEED = 42
CEILING_N_ROLLOUTS = 3

BGE_MODEL = "BAAI/bge-large-en-v1.5"  # CLS-pool, L2-normalized (issue617 precedent)
BGE_MAX_TOKENS = 512

VLLM_CHUNK_SIZE = 512

# HF-repo destination prefixes (plan §6.5).
HF_PREDICTOR_PREFIX = f"{SLUG}/analysis_tensors/predictor_captures"
HF_MAP_CORPUS_PREFIX = f"{SLUG}/analysis_tensors/map_corpus"


# ---------------------------------------------------------------------------
# Layer-indexing convention (CPU-testable; no model needed)
# ---------------------------------------------------------------------------
def layer_index_convention(n_layers: int) -> dict:
    """Assert the model's decoder-block count and return the recorded convention.

    We store block indices 0..n_layers-1 (matching pass-B). The full
    ``output_hidden_states`` tuple has n_layers+1 entries (index 0 = embeddings);
    stored index i corresponds to that tuple's entry i+1 (pre-final-norm at the
    last block, per ``extract_layer_activations``). Fail loud on an unexpected
    layer count (a wrong model / off-by-one guard)."""
    if n_layers != EXPECTED_LAYERS:
        raise RuntimeError(f"expected {EXPECTED_LAYERS} decoder blocks, got {n_layers}")
    return {
        "stored_layers": list(range(n_layers)),
        "n_stored": n_layers,
        "hidden_states_tuple_len": n_layers + 1,  # 29: index 0 = embeddings
        "note": (
            "stored index i == decoder block i == output_hidden_states[i+1] "
            "(pre-final-norm at i=27); matches the reused #779/#2254 pass-B bundle. "
            "Parent-pinned L16/L27 resolved downstream against this 0..27 axis "
            "(#2254 'layer 14' == stored index 14); P7 symmetric curves cross-check."
        ),
    }


def assert_hidden_state_stack_indexing(stack) -> int:
    """Tiny-tensor unit path for the CPU smoke: given an (n_layers+1, ...)
    hidden-state stack (index 0 = embeddings), return the number of STORED decoder
    layers (drop the embedding entry). Validates the drop-embedding convention on a
    synthetic stack without loading a model."""
    total = stack.shape[0]
    n_stored = total - 1  # drop the embedding entry
    return n_stored


# ---------------------------------------------------------------------------
# Disjointness (map_corpus fit-row hygiene; CPU-testable on the real banks)
# ---------------------------------------------------------------------------
def collect_bank_strings(banks_dir: Path) -> set[str]:
    """Q_sim + Q_beh + every trigger prompt, both settings — the string set LMSYS
    fit rows must be disjoint from (plan §4.2 P4 hygiene)."""
    strings: set[str] = set()
    for setting in ("em", "caps"):
        for trig in load_triggers(banks_dir, setting):
            strings.add(trig["prompt"])
        strings.update(load_questions(banks_dir, setting))
        # Q_sim banks.
        qsim_name = "q_sim_em.json" if setting == "em" else "q_sim_caps.json"
        qsim = json.loads((banks_dir / qsim_name).read_text(encoding="utf-8"))
        strings.update(qsim)
    return strings


def assert_lmsys_disjoint(lmsys_prompts: list[str], banks_dir: Path) -> None:
    """Fail loud if any LMSYS fit prompt collides with a Q_sim/Q_beh/trigger string."""
    bank = collect_bank_strings(banks_dir)
    overlap = bank.intersection(lmsys_prompts)
    if overlap:
        raise RuntimeError(
            f"LMSYS fit rows overlap {len(overlap)} bank strings (fit-row hygiene "
            f"violated); first: {sorted(overlap)[0][:80]!r}"
        )


# ---------------------------------------------------------------------------
# Q_sim loading
# ---------------------------------------------------------------------------
def load_q_sim(banks_dir: Path, setting: str) -> list[str]:
    name = "q_sim_em.json" if setting == "em" else "q_sim_caps.json"
    q = json.loads((banks_dir / name).read_text(encoding="utf-8"))
    if not isinstance(q, list) or not all(isinstance(s, str) for s in q):
        raise RuntimeError(f"{name}: expected list[str]")
    return q


def p_inoc_for(setting: str) -> str:
    return P_INOC_EM if setting == "em" else P_INOC_CAPS


# ---------------------------------------------------------------------------
# LMSYS replay (reuse the pinned #952 recipe; never re-implement)
# ---------------------------------------------------------------------------
def reconstruct_lmsys_prompts(n_needed: int) -> list[str]:
    """Reuse ``issue952_stats._reconstruct_lmsys_prompts`` at its pinned
    LMSYS_REVISION (the #823/#952/#1615 replay). GATED dataset — needs a granted
    pod HF token."""
    import issue952_stats

    return issue952_stats._reconstruct_lmsys_prompts(n_needed)


def probe_lmsys_access() -> int:
    """1-row streaming read to certify the pod token's LMSYS grant BEFORE P4."""
    prompts = reconstruct_lmsys_prompts(1)
    logger.info("LMSYS gated-access probe OK: read %d row(s)", len(prompts))
    return 0


# ---------------------------------------------------------------------------
# Model / engine resolution
# ---------------------------------------------------------------------------
def resolve_model(args) -> tuple[str, object]:
    """(model_path, cleanup). ``--model`` -> use as-is; ``--adapter`` -> merge onto
    base into <out-dir>/merged/<name> + delete after (MooseFS quota)."""
    if args.model:
        return args.model, (lambda: None)
    from explore_persona_space.train.sft import merge_lora

    merged_dir = Path(args.out_dir) / "merged" / args.model_name
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("merging adapter %s -> %s", args.adapter, merged_dir)
    merge_lora(BASE_MODEL, args.adapter, str(merged_dir), gpu_id=args.gpu_id)

    def _cleanup() -> None:
        shutil.rmtree(merged_dir, ignore_errors=True)
        logger.info("deleted merged dir %s", merged_dir)

    return str(merged_dir), _cleanup


def load_hf_model(model_path: str):
    """Load the HF model (bf16, cuda:0) + tokenizer for teacher-forced capture."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.device("cuda:0")},
        trust_remote_code=True,
    )
    model.eval()
    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    conv = layer_index_convention(n_layers)
    if hidden != EXPECTED_HIDDEN:
        raise RuntimeError(f"expected hidden {EXPECTED_HIDDEN}, got {hidden}")
    return model, tokenizer, conv


# ---------------------------------------------------------------------------
# Capture phases (production-only; deferred heavy imports)
# ---------------------------------------------------------------------------
def _capture_v_c(model, tokenizer, messages, layers):
    """v_C = last-prompt-token state, all layers (reuse the #779 helper -> exact
    pass-B convention)."""
    import issue779_collect as I779

    return I779.capture_context_vector(model, tokenizer, messages, layers)["last"]


def _capture_v_a(model, tokenizer, messages, response, layers):
    """v_A = mean-response state, all layers, teacher-forced (reuse the #779 helper;
    empty r_b dict skips the trait projections). Returns None on empty response."""
    import issue779_collect as I779

    av = I779.capture_answer_vector(model, tokenizer, messages, response, layers, {})
    return None if av is None else av["v_x"]


def phase_grid(model, tokenizer, layers, banks_dir, setting, out_bundle: Path, meta: dict):
    """v_C grid over Q_sim x triggers -> fp16 bundle."""
    import torch

    q_sim = load_q_sim(banks_dir, setting)
    triggers = load_triggers(banks_dir, setting)
    rows, row_meta = [], []
    for ti, trig in enumerate(triggers):
        for qi, q in enumerate(q_sim):
            messages = render_context_messages(trig["prompt"], q)
            rows.append(_capture_v_c(model, tokenizer, messages, layers).to(torch.float16))
            row_meta.append({"trigger_idx": ti, "trigger_label": trig["label"], "q_sim_idx": qi})
    v_c = torch.stack(rows)  # (n_rows, 28, 3584)
    torch.save({"v_c": v_c, "row_meta": row_meta, **meta}, out_bundle)
    logger.info("grid: wrote %s v_c=%s", out_bundle, tuple(v_c.shape))


def phase_mu(model, tokenizer, layers, train_jsonl: Path, setting, out_bundle: Path, meta: dict):
    """Streaming means mu_train (v_C over training rows) + mu_A_train (v_A over gold
    answers). Per-row tensors NEVER stored (streaming reduce)."""
    import torch

    mu_c = None
    mu_a = None
    n_c = 0
    n_a = 0
    with train_jsonl.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sys_turn = row["prompt"][0]["content"]
            user_turn = row["prompt"][-1]["content"]
            gold = row["completion"][0]["content"]
            messages = [
                {"role": "system", "content": sys_turn},
                {"role": "user", "content": user_turn},
            ]
            v_c = _capture_v_c(model, tokenizer, messages, layers).to(torch.float32)
            mu_c = v_c if mu_c is None else mu_c + v_c
            n_c += 1
            v_a = _capture_v_a(model, tokenizer, messages, gold, layers)
            if v_a is not None:
                v_a = v_a.to(torch.float32)
                mu_a = v_a if mu_a is None else mu_a + v_a
                n_a += 1
    if n_c == 0:
        raise RuntimeError(f"no training rows in {train_jsonl}")
    mu_c = (mu_c / n_c).to(torch.float16)
    mu_a = (mu_a / n_a).to(torch.float16) if n_a else None
    torch.save({"mu_train": mu_c, "mu_a_train": mu_a, "n_c": n_c, "n_a": n_a, **meta}, out_bundle)
    logger.info("mu: wrote %s (n_c=%d n_a=%d)", out_bundle, n_c, n_a)


def _vllm_rollouts(model_path: str, prompt_texts: list[str], n: int, gpu_id: int):
    """Batched vLLM rollouts (chunked). Returns per-prompt list[str]."""
    from vllm import SamplingParams

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    sp = SamplingParams(
        n=n,
        temperature=ROLLOUT_TEMPERATURE,
        top_p=ROLLOUT_TOP_P,
        max_tokens=ROLLOUT_MAX_TOKENS,
        seed=ROLLOUT_SEED,
    )
    llm = create_vllm_engine(model_path, max_model_len=8192, seed=ROLLOUT_SEED)
    out: list[list[str]] = []
    try:
        for i in range(0, len(prompt_texts), VLLM_CHUNK_SIZE):
            chunk = prompt_texts[i : i + VLLM_CHUNK_SIZE]
            for o in llm.generate(chunk, sp, use_tqdm=False):
                out.append([c.text for c in o.outputs])
    finally:
        cleanup_vllm(llm)
    return out


def phase_ceiling(
    model,
    tokenizer,
    layers,
    model_path,
    banks_dir,
    setting,
    out_bundle: Path,
    rawcomp_dir: Path,
    gpu_id,
    meta: dict,
):
    """3 on-policy rollouts per (q, c); teacher-forced v_A per rollout stored
    PER-ROLLOUT; rollout TEXT persisted to raw_completions/ceiling/."""
    import torch

    q_sim = load_q_sim(banks_dir, setting)
    triggers = load_triggers(banks_dir, setting)
    prompt_texts, keys = [], []
    for ti, trig in enumerate(triggers):
        for qi, q in enumerate(q_sim):
            messages = render_context_messages(trig["prompt"], q)
            prompt_texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            keys.append((ti, trig["label"], trig["prompt"], qi, q))
    rollouts = _vllm_rollouts(model_path, prompt_texts, CEILING_N_ROLLOUTS, gpu_id)

    v_a_rows, row_meta, raw_rows = [], [], []
    for (ti, tlabel, tprompt, qi, q), comps in zip(keys, rollouts, strict=True):
        messages = render_context_messages(tprompt, q)
        per_rollout = []
        for ri, text in enumerate(comps):
            v_a = _capture_v_a(model, tokenizer, messages, text, layers)
            if v_a is None:
                continue
            v_a_rows.append(v_a.to(torch.float16))
            per_rollout.append(len(v_a_rows) - 1)
            row_meta.append(
                {"trigger_idx": ti, "trigger_label": tlabel, "q_sim_idx": qi, "rollout_idx": ri}
            )
        raw_rows.append(
            {
                "trigger_idx": ti,
                "trigger_label": tlabel,
                "q_sim_idx": qi,
                "question": q,
                "completions": comps,
                "kept_v_a_idx": per_rollout,
            }
        )
    v_a = torch.stack(v_a_rows) if v_a_rows else torch.empty(0)
    torch.save({"v_a": v_a, "row_meta": row_meta, **meta}, out_bundle)
    _write_rawcomp(rawcomp_dir, "ceiling", meta["model"], raw_rows, meta)
    logger.info("ceiling: wrote %s v_a=%s", out_bundle, tuple(v_a.shape))


def phase_map_corpus(
    model,
    tokenizer,
    layers,
    model_path,
    banks_dir,
    out_bundle: Path,
    rawcomp_dir: Path,
    gpu_id,
    meta: dict,
):
    """Replay 5,000 LMSYS prompts (user-only messages, matching pass-B); teacher-force
    (v_C, v_A) all 28 layers -> one fp16 bundle. Rollout TEXT to raw_completions/."""
    import torch

    prompts = reconstruct_lmsys_prompts(LMSYS_N_PROMPTS)
    assert_lmsys_disjoint(prompts, banks_dir)
    prompt_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    rollouts = _vllm_rollouts(model_path, prompt_texts, 1, gpu_id)

    v_c_rows, v_a_rows, kept_idx, raw_rows = [], [], [], []
    for i, (p, comps) in enumerate(zip(prompts, rollouts, strict=True)):
        messages = [{"role": "user", "content": p}]
        v_a = _capture_v_a(model, tokenizer, messages, comps[0], layers)
        if v_a is None:
            raw_rows.append({"row_idx": i, "completions": comps, "kept": False})
            continue
        v_c = _capture_v_c(model, tokenizer, messages, layers)
        v_c_rows.append(v_c.to(torch.float16))
        v_a_rows.append(v_a.to(torch.float16))
        kept_idx.append(i)
        raw_rows.append({"row_idx": i, "completions": comps, "kept": True})
    v_c = torch.stack(v_c_rows)
    v_a = torch.stack(v_a_rows)
    torch.save(
        {"v_c": v_c, "v_a": v_a, "kept_row_idx": kept_idx, "n_prompts": len(prompts), **meta},
        out_bundle,
    )
    _write_rawcomp(rawcomp_dir, "map_corpus", meta["model"], raw_rows, meta)
    logger.info(
        "map_corpus: wrote %s v_c=%s v_a=%s", out_bundle, tuple(v_c.shape), tuple(v_a.shape)
    )


# ---------------------------------------------------------------------------
# Text baselines (BGE + lexical; pod-side CPU)
# ---------------------------------------------------------------------------
def _bge_embed(texts: list[str]):
    """CLS-pooled (last_hidden_state[:, 0]) + L2-normalized BGE embeddings
    (issue617 convention)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL)
    model.eval()
    embs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(
                t, return_tensors="pt", truncation=True, max_length=BGE_MAX_TOKENS, padding=True
            )
            hs = model(**enc).last_hidden_state
            cls = torch.nn.functional.normalize(hs[:, 0], p=2, dim=1)
            embs.append(cls[0].float())
    return torch.stack(embs)


def _lexical_sims(a: str, b: str) -> dict:
    """Token Jaccard, SequenceMatcher ratio, and a placeholder for TF-IDF (computed
    at corpus level in phase_text_baselines)."""
    from difflib import SequenceMatcher

    ta, tb = set(a.lower().split()), set(b.lower().split())
    jacc = (len(ta & tb) / len(ta | tb)) if (ta | tb) else 0.0
    return {"jaccard": jacc, "seqmatch_ratio": SequenceMatcher(None, a, b).ratio()}


def phase_text_baselines(banks_dir: Path, setting: str, out_json: Path, meta: dict):
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    triggers = load_triggers(banks_dir, setting)
    p_inoc = p_inoc_for(setting)
    labels = [t["label"] for t in triggers]
    prompts = [t["prompt"] for t in triggers]

    # BGE cosine of each trigger to p_inoc.
    all_texts = prompts + [p_inoc]
    embs = _bge_embed(all_texts)
    p_inoc_emb = embs[-1]
    bge_cos = (embs[:-1] @ p_inoc_emb).tolist()  # already L2-normalized

    # TF-IDF cosine of each trigger to p_inoc.
    tfidf = TfidfVectorizer().fit_transform(all_texts)
    tfidf_cos = cosine_similarity(tfidf[:-1], tfidf[-1]).ravel().tolist()

    per_trigger = {}
    for i, (lab, prompt) in enumerate(zip(labels, prompts, strict=True)):
        lex = _lexical_sims(prompt, p_inoc)
        per_trigger[lab] = {
            "prompt": prompt,
            "bge_cos_to_p_inoc": bge_cos[i],
            "tfidf_cos_to_p_inoc": tfidf_cos[i],
            **lex,
        }
    payload = {
        "setting": setting,
        "p_inoc": p_inoc,
        "bge_model": BGE_MODEL,
        "per_trigger": per_trigger,
        "trigger_embeddings": embs[:-1].tolist(),
        "p_inoc_embedding": p_inoc_emb.tolist(),
        **meta,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("text_baselines: wrote %s (%d triggers)", out_json, len(labels))
    del torch


# ---------------------------------------------------------------------------
# Persistence + upload
# ---------------------------------------------------------------------------
def _write_rawcomp(
    rawcomp_dir: Path, stage: str, model_name: str, raw_rows: list[dict], meta: dict
):
    dest = rawcomp_dir / stage / model_name
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": 2379,
        "slug": SLUG,
        "model": model_name,
        "stage": stage,
        "rollout_sampling": {
            "temperature": ROLLOUT_TEMPERATURE,
            "top_p": ROLLOUT_TOP_P,
            "max_tokens": ROLLOUT_MAX_TOKENS,
            "seed": ROLLOUT_SEED,
        },
        "git": meta.get("git"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": raw_rows,
    }
    (dest / "raw_completions.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def upload_rawcomp(rawcomp_dir: Path) -> dict[str, str]:
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    return upload_raw_completions_to_data_repo(experiment_name=SLUG, eval_results_dir=rawcomp_dir)


def upload_tensor(local_path: Path, path_in_repo: str) -> str:
    """Upload one .pt bundle to the HF data repo, fail loud."""
    from explore_persona_space.orchestrate import hub

    return hub._upload(
        local_path,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        path_in_repo,
        upload_as_file=True,
        raise_on_error=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _git_meta() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance(cwd=REPO_ROOT))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        choices=["grid", "mu", "ceiling", "map_corpus", "text_baselines"],
        default=None,
    )
    ap.add_argument("--setting", choices=["em", "caps"], default=None)
    ap.add_argument("--model", default=None, help="Merged dir / base HF id (use as-is)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter to merge onto base + delete")
    ap.add_argument("--model-name", default=None, help="Logical name for output keys/paths")
    ap.add_argument("--train-jsonl", default=None, help="Training JSONL for --phase mu")
    ap.add_argument("--banks-dir", default=str(REPO_ROOT / "data" / "issue_2379" / "banks"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "eval_results" / "issue_2379"))
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--probe-access", action="store_true", help="1-row LMSYS gated read; then exit")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="CPU arg-validation only; no GPU")
    args = ap.parse_args()

    if args.probe_access:
        if args.dry_run:
            logger.info("[dry-run] --probe-access wired; skipping the gated LMSYS read")
            return 0
        return probe_lmsys_access()

    if args.phase is None or args.setting is None or args.model_name is None:
        ap.error("--phase, --setting, --model-name required (unless --probe-access)")
    if bool(args.model) == bool(args.adapter):
        ap.error("exactly one of --model / --adapter is required")
    if args.phase == "mu" and not args.train_jsonl:
        ap.error("--phase mu requires --train-jsonl")

    banks_dir = Path(args.banks_dir)
    out_dir = Path(args.out_dir)
    rawcomp_dir = out_dir / "rawcomp_capture"
    tensor_dir = out_dir / "capture_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # Exercise the CPU-checkable invariants: convention, disjointness on real banks.
        logger.info("[dry-run] layer convention: %s", layer_index_convention(EXPECTED_LAYERS))
        bank = collect_bank_strings(banks_dir)
        logger.info("[dry-run] bank string set size = %d", len(bank))
        assert_lmsys_disjoint(["<synthetic lmsys prompt not in banks>"], banks_dir)
        logger.info("[dry-run] disjointness assert OK; args resolved")
        return 0

    meta = {"model": args.model_name, "setting": args.setting, "git": _git_meta()}

    if args.phase == "text_baselines":
        # CPU-only; no model load.
        out_json = tensor_dir / f"text_baselines_{args.setting}_{args.model_name}.json"
        phase_text_baselines(banks_dir, args.setting, out_json, meta)
        return 0

    model_path, cleanup = resolve_model(args)
    try:
        model, tokenizer, conv = load_hf_model(model_path)
        meta["layer_convention"] = conv
        layers = conv["stored_layers"]
        t0 = time.time()

        pred_subdir = tensor_dir / "predictor_captures" / args.model_name
        pred_subdir.mkdir(parents=True, exist_ok=True)

        if args.phase == "grid":
            phase_grid(
                model, tokenizer, layers, banks_dir, args.setting, pred_subdir / "grid.pt", meta
            )
        elif args.phase == "mu":
            phase_mu(
                model,
                tokenizer,
                layers,
                Path(args.train_jsonl),
                args.setting,
                pred_subdir / "mu.pt",
                meta,
            )
        elif args.phase == "ceiling":
            phase_ceiling(
                model,
                tokenizer,
                layers,
                model_path,
                banks_dir,
                args.setting,
                pred_subdir / "ceiling.pt",
                rawcomp_dir,
                args.gpu_id,
                meta,
            )
        elif args.phase == "map_corpus":
            mc_path = tensor_dir / "map_corpus" / f"{args.model_name}.pt"
            mc_path.parent.mkdir(parents=True, exist_ok=True)
            phase_map_corpus(
                model,
                tokenizer,
                layers,
                model_path,
                banks_dir,
                mc_path,
                rawcomp_dir,
                args.gpu_id,
                meta,
            )
        logger.info("phase %s done in %.1fs", args.phase, time.time() - t0)
    finally:
        cleanup()

    if not args.no_upload:
        if args.phase in ("ceiling", "map_corpus"):
            urls = upload_rawcomp(rawcomp_dir)
            logger.info("uploaded %d rollout-text files to HF", len(urls))
        if args.phase in ("grid", "mu", "ceiling"):
            local = tensor_dir / "predictor_captures" / args.model_name / f"{args.phase}.pt"
            upload_tensor(local, f"{HF_PREDICTOR_PREFIX}/{args.model_name}/{args.phase}.pt")
            logger.info("uploaded predictor bundle -> %s", HF_PREDICTOR_PREFIX)
        elif args.phase == "map_corpus":
            local = tensor_dir / "map_corpus" / f"{args.model_name}.pt"
            upload_tensor(local, f"{HF_MAP_CORPUS_PREFIX}/{args.model_name}.pt")
            logger.info("uploaded map_corpus bundle -> %s", HF_MAP_CORPUS_PREFIX)

    return 0


if __name__ == "__main__":
    sys.exit(main())
