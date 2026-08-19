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

Round-2 review hardening (r1 verdict punch list):
  * consumer-contract validation (banks / train JSONL / LMSYS prompts, counts, keys)
    runs BEFORE any merge or model constructor (blocker consumer-contract-post-init);
  * vLLM rollout generation runs BEFORE the HF model load (never co-resident), with
    the rollout set persisted to an atomic fingerprinted sidecar;
  * every teacher-forced loop checkpoints per chunk (atomic writes + fingerprinted
    resume + per-chunk progress lines) so a crash forfeits at most one chunk;
  * empty generations are RETRIED (bounded passes), residual drops are counted,
    reported, and fail loud against registered-grain floors;
  * restricted-content error paths report counts/hashes/indices only, never text;
  * per-invocation idempotency: an existing final bundle skips recompute
    (``--force`` overrides) while the upload leg still runs.

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

Teacher-forced capture stays PER-ROW through the reused ``issue779_collect``
helpers by design: they are the exact pass-B capture convention the reused #2254
bundle was built with, and a batched re-implementation would need its own
batched-vs-serial equivalence gate against that convention (plan §9 sized the
P3/P4 walls at the per-row cost). The crash-loss exposure that motivated the
review finding is closed by the per-chunk checkpoints instead.

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
import hashlib
import json
import logging
import os
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
    MERGED_ROOT_DEFAULT,
    SLUG,
    load_questions,
    load_triggers,
    reclaim_dead_merge_dirs,
    render_context_messages,
    resolve_model_identity,
    write_merge_provenance,
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

# Empty-generation retry + registered-grain floors (r1 finding: silent empty-answer
# drops). Empty rollouts are regenerated for up to EMPTY_RETRY_PASSES extra passes
# (seed bumped per pass); residual drops are counted + reported, and the phase fails
# loud when the realized grain falls below the floors.
EMPTY_RETRY_PASSES = 2
CEILING_MIN_KEPT_PER_CELL = 2  # split-rollout reliability needs >= 2 rollouts/cell
MAX_EMPTY_DROP_FRAC = 0.01  # ceiling: total dropped slots / total slots
MAP_MIN_KEPT_FRAC = 0.99  # map_corpus: kept rows / 5000

# Per-chunk checkpoint cadence for the teacher-forced loops (units per chunk).
CKPT_EVERY = 250

BGE_MODEL = "BAAI/bge-large-en-v1.5"  # CLS-pool, L2-normalized (issue617 precedent)
BGE_MAX_TOKENS = 512

VLLM_CHUNK_SIZE = 512

# HF-repo destination prefixes (plan §6.5).
HF_PREDICTOR_PREFIX = f"{SLUG}/analysis_tensors/predictor_captures"
HF_MAP_CORPUS_PREFIX = f"{SLUG}/analysis_tensors/map_corpus"
HF_TEXT_BASELINES_PREFIX = f"{SLUG}/analysis_tensors/text_baselines"

# SequenceMatcher producer key — the ONE spelling shared with the mapfit consumer
# (r1 minor: producer/consumer schema detached; import as
# ``from issue2379_capture import SEQMATCH_KEY``).
SEQMATCH_KEY = "seqmatch_ratio"


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
    """Fail loud if any LMSYS fit prompt collides with a Q_sim/Q_beh/trigger string.

    Restricted-content discipline: the error reports counts + sha256 digests +
    colliding row INDICES only — never raw bank/corpus text (r1 finding)."""
    bank = collect_bank_strings(banks_dir)
    overlap = bank.intersection(lmsys_prompts)
    if overlap:
        digests = sorted(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16] for s in overlap)
        hit_idx = [i for i, p in enumerate(lmsys_prompts) if p in overlap][:10]
        raise RuntimeError(
            f"LMSYS fit rows overlap {len(overlap)} bank strings (fit-row hygiene "
            f"violated); sha256/16 digests (first 5): {digests[:5]}; "
            f"lmsys row indices (first 10): {hit_idx}"
        )


# ---------------------------------------------------------------------------
# Q_sim loading
# ---------------------------------------------------------------------------
def load_q_sim(banks_dir: Path, setting: str) -> list[str]:
    name = "q_sim_em.json" if setting == "em" else "q_sim_caps.json"
    q = json.loads((banks_dir / name).read_text(encoding="utf-8"))
    if not isinstance(q, list) or not all(isinstance(s, str) for s in q):
        raise RuntimeError(f"{name}: expected list[str]")
    if not q:
        raise RuntimeError(f"{name}: empty Q_sim bank")
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
# Pre-model input validation (consumer-contract BEFORE the model constructor)
# ---------------------------------------------------------------------------
def validate_mu_train_jsonl(train_jsonl: Path) -> int:
    """Resolve + validate the mu-phase training JSONL BEFORE any model work.

    Asserts existence, non-emptiness, and the first row's prompt/completion
    message-dict contract (the exact fields phase_mu dereferences). Returns the
    non-blank row count (the mu fingerprint + progress denominator)."""
    if not train_jsonl.exists():
        raise RuntimeError(f"--train-jsonl missing: {train_jsonl}")
    n = 0
    first: dict | None = None
    with train_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if first is None:
                first = json.loads(line)
            n += 1
    if n == 0 or first is None:
        raise RuntimeError(f"empty train file: {train_jsonl}")
    for key in ("prompt", "completion"):
        v = first.get(key)
        if not isinstance(v, list) or not v or not isinstance(v[0], dict):
            raise RuntimeError(f"{train_jsonl.name}: '{key}' must be a non-empty message-dict list")
        for m in v:
            if "role" not in m or "content" not in m:
                raise RuntimeError(f"{train_jsonl.name}: '{key}' message missing role/content")
    if not str(first["completion"][0].get("content", "")).strip():
        raise RuntimeError(f"{train_jsonl.name}: first completion has empty content")
    return n


# ---------------------------------------------------------------------------
# Model / engine resolution
# ---------------------------------------------------------------------------
def resolve_model(args) -> tuple[str, object]:
    """(model_path, cleanup). ``--model`` -> use as-is; ``--adapter`` -> merge onto
    base into a phase+pid-scoped dir under ``--merged-root`` + delete after
    (MooseFS quota). The pid scope keeps concurrent invocations for the SAME model
    from sharing a merged dir (r1 minor); the merge root lives under data/, never
    eval_results/ (round-3 g1 Major merge-root-unification), crash-leaked sibling
    dirs are reclaimed at entry, and a provenance sidecar records the adapter's
    weights identity for downstream resume fingerprints."""
    if args.model:
        return args.model, (lambda: None)
    from explore_persona_space.train.sft import merge_lora

    merged_root = Path(args.merged_root)
    merged_root.mkdir(parents=True, exist_ok=True)
    reclaim_dead_merge_dirs(merged_root, args.model_name, args.phase)
    merged_dir = merged_root / f"{args.model_name}.{args.phase}.{os.getpid()}"
    logger.info("merging adapter %s -> %s", args.adapter, merged_dir)
    merge_lora(BASE_MODEL, args.adapter, str(merged_dir), gpu_id=args.gpu_id)
    write_merge_provenance(merged_dir, args.adapter)

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


def load_tokenizer(model_path: str):
    """Tokenizer alone (cheap) — used to render rollout prompts BEFORE the HF
    model load so the vLLM engine and the HF model are never co-resident."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Atomic persistence + chunked checkpoint store (crash-resume; r1 finding)
# ---------------------------------------------------------------------------
def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_torch_save(obj, path: Path) -> None:
    import torch

    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class _ChunkStore:
    """Atomic per-chunk checkpoint dir beside a bundle path.

    Chunks are sequential unit ranges saved via tmp+``os.replace``; ``meta.json``
    pins a GENERATING-PARAMETER fingerprint (machine-stable strings/ints — never
    hashes of recomputed float arrays, per code-style resume-key rules). A
    fingerprint mismatch discards the partial state loudly; non-contiguous chunk
    residue past a gap is dropped. Resume additionally LOADS every contiguous
    chunk and validates its payload keys (round-3 codex Major
    capture-batch1-restartability: a truncated chunk with a valid NAME was
    repeatedly accepted at resume and crashed only at assembly — a permanent
    wedge); an invalid frontier chunk is deleted together with every later chunk
    so the phase rebuilds from the last GOOD unit instead of wedging."""

    def __init__(self, bundle_path: Path, fingerprint: dict, payload_keys: tuple[str, ...]):
        self.dir = bundle_path.parent / (bundle_path.name + ".chunks")
        self.meta_path = self.dir / "meta.json"
        self.fingerprint = fingerprint
        self.payload_keys = frozenset(payload_keys)

    def _init_fresh(self) -> None:
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        _atomic_write_text(self.meta_path, json.dumps({"fingerprint": self.fingerprint}))

    def _chunk_files(self) -> list[tuple[int, int, Path]]:
        out = []
        for p in self.dir.glob("chunk_*.pt"):
            parts = p.stem.split("_")
            out.append((int(parts[1]), int(parts[2]), p))
        return sorted(out)

    def _chunk_valid(self, p: Path) -> bool:
        """True iff the chunk file torch-loads to a dict carrying every payload key."""
        import torch

        try:
            payload = torch.load(p, weights_only=True)
        except Exception:  # noqa: BLE001 — truncated/corrupt chunk = discardable state
            return False
        return isinstance(payload, dict) and self.payload_keys <= set(payload)

    def resume_units(self) -> int:
        """Contiguous VALIDATED completed unit count (0 on fresh/mismatched state)."""
        if not self.meta_path.exists():
            self._init_fresh()
            return 0
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            meta = None
        if meta is None or meta.get("fingerprint") != self.fingerprint:
            logger.warning(
                "[ckpt] discarding stale chunk state (fingerprint mismatch): %s", self.dir
            )
            self._init_fresh()
            return 0
        cur = 0
        files = self._chunk_files()
        for k, (start, end, p) in enumerate(files):
            if start != cur:
                logger.warning("[ckpt] dropping non-contiguous chunk residue from %s", p.name)
                p.unlink()
                continue
            if not self._chunk_valid(p):
                n_later = sum(1 for _, _, q in files[k + 1 :] if q.exists())
                logger.warning(
                    "[ckpt] dropping truncated/invalid chunk %s (+%d later chunks); "
                    "resuming from unit %d",
                    p.name,
                    n_later,
                    cur,
                )
                for _, _, q in files[k:]:
                    q.unlink(missing_ok=True)
                break
            cur = end
        return cur

    def append(self, start: int, end: int, payload: dict) -> None:
        _atomic_torch_save(payload, self.dir / f"chunk_{start:06d}_{end:06d}.pt")

    def load_payloads(self) -> list[dict]:
        import torch

        return [torch.load(p, weights_only=True) for _, _, p in self._chunk_files()]

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def phase_fingerprint(phase: str, meta: dict, **counts) -> dict:
    """GENERATING-PARAMETER fingerprint shared by each phase (chunk store /
    mu-partial / bundle sidecar) and main's skip predicate — ONE composition site
    so producer and skip check can never drift. Binds to the producing MODEL's
    weights identity (round-3 g1 Major: name/count-only regimes survive a
    retrain, silently reusing the OLD model's rollouts/activations)."""
    return {
        "phase": phase,
        "model": meta["model"],
        "setting": meta["setting"],
        "model_ident": meta["model_ident"],
        **counts,
    }


def bundle_sidecar(out_bundle: Path) -> Path:
    """Regime sidecar path beside a final bundle (``<name>.meta.json``)."""
    return out_bundle.with_name(out_bundle.name + ".meta.json")


def write_bundle_sidecar(out_bundle: Path, fp: dict) -> None:
    """Record the producing fingerprint beside the final bundle (skip-predicate
    input; round-3 g1 Minor: presence-only skips reuse stale-regime bundles)."""
    _atomic_write_text(bundle_sidecar(out_bundle), json.dumps({"fingerprint": fp}, indent=2))


def bundle_current(out_bundle: Path, fp: dict) -> bool:
    """True iff the final bundle exists AND its sidecar records the expected
    fingerprint. Pre-round-3 bundles (no sidecar) and unreadable sidecars read
    NOT-current -> recompute (conservative-correct)."""
    if not out_bundle.exists():
        return False
    sc = bundle_sidecar(out_bundle)
    if not sc.is_file():
        return False
    try:
        doc = json.loads(sc.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    return doc.get("fingerprint") == fp


_MU_PARTIAL_KEYS = ("fingerprint", "mu_c_sum", "mu_a_sum", "n_c", "n_a", "next_line_idx")


def _load_mu_partial(partial: Path, fp: dict):
    """Load + validate a mu ``.partial.pt``. ANY defect — unreadable file, non-dict
    payload, missing state key, fingerprint mismatch — returns None
    (discard-as-stale), never a KeyError (round-3 codex Major: a partial with a
    matching fingerprint but a missing state key raised instead of discarding)."""
    import torch

    try:
        st = torch.load(partial, weights_only=True)
    except Exception:  # noqa: BLE001 — a truncated partial is discardable state
        return None
    if not isinstance(st, dict) or any(k not in st for k in _MU_PARTIAL_KEYS):
        return None
    if st.get("fingerprint") != fp:
        return None
    return st


# ---------------------------------------------------------------------------
# Capture primitives (production-only; deferred heavy imports)
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


# ---------------------------------------------------------------------------
# Rollout generation (vLLM, BEFORE the HF model load) + empty retry
# ---------------------------------------------------------------------------
def _chunked_rollout_generate(llm, prompt_texts: list[str], sp) -> list[list[str]]:
    out: list[list[str]] = []
    n_chunks = (len(prompt_texts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for i in range(0, len(prompt_texts), VLLM_CHUNK_SIZE):
        chunk = prompt_texts[i : i + VLLM_CHUNK_SIZE]
        logger.info(
            "[rollouts] chunk %d/%d (%d prompts x n=%d)",
            i // VLLM_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
            sp.n,
        )
        for o in llm.generate(chunk, sp, use_tqdm=False):
            out.append([c.text for c in o.outputs])
    return out


def generate_rollouts_with_retry(
    model_path: str,
    prompt_texts: list[str],
    n_samples: int,
    sidecar: Path,
    fingerprint: dict,
) -> tuple[list[list[str]], dict]:
    """Batched vLLM rollouts with a fingerprinted sidecar (resume) and bounded
    empty-generation retry passes (seed bumped per pass; r1 finding: silent
    empty-answer drops). Returns (per-prompt list[str], drop_stats)."""
    if sidecar.exists():
        try:
            doc = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            doc = None
        if doc is not None and doc.get("fingerprint") == fingerprint:
            logger.info("[rollouts] reusing persisted sidecar %s", sidecar.name)
            return doc["rollouts"], doc["drop_stats"]
        logger.warning("[rollouts] discarding stale sidecar (fingerprint mismatch): %s", sidecar)

    from vllm import SamplingParams

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    def _sp(seed: int):
        return SamplingParams(
            n=n_samples,
            temperature=ROLLOUT_TEMPERATURE,
            top_p=ROLLOUT_TOP_P,
            max_tokens=ROLLOUT_MAX_TOKENS,
            seed=seed,
        )

    retry_passes = []
    llm = create_vllm_engine(model_path, max_model_len=8192, seed=ROLLOUT_SEED)
    try:
        out = _chunked_rollout_generate(llm, prompt_texts, _sp(ROLLOUT_SEED))
        for p in range(1, EMPTY_RETRY_PASSES + 1):
            need = [i for i, comps in enumerate(out) if any(not t.strip() for t in comps)]
            if not need:
                break
            regen = _chunked_rollout_generate(
                llm, [prompt_texts[i] for i in need], _sp(ROLLOUT_SEED + p)
            )
            filled = 0
            for j, i in enumerate(need):
                repl = [t for t in regen[j] if t.strip()]
                comps = out[i]
                for k in range(len(comps)):
                    if not comps[k].strip() and repl:
                        comps[k] = repl.pop(0)
                        filled += 1
            retry_passes.append({"pass": p, "prompts_with_empty": len(need), "filled": filled})
            logger.info(
                "[rollouts] retry pass %d: %d prompts had empty slots, filled %d",
                p,
                len(need),
                filled,
            )
    finally:
        cleanup_vllm(llm)

    n_slots = sum(len(c) for c in out)
    n_empty = sum(1 for comps in out for t in comps if not t.strip())
    drop_stats = {
        "n_slots": n_slots,
        "n_empty_after_retries": n_empty,
        "retry_passes": retry_passes,
    }
    logger.info("[rollouts] done: %d slots, %d empty after retries", n_slots, n_empty)
    _atomic_write_text(
        sidecar,
        json.dumps(
            {"fingerprint": fingerprint, "drop_stats": drop_stats, "rollouts": out},
            ensure_ascii=False,
        ),
    )
    return out, drop_stats


def enforce_ceiling_text_floors(rollouts: list[list[str]]) -> None:
    """Registered-grain floors for ceiling rollouts (counts only, never text)."""
    n_slots = sum(len(c) for c in rollouts)
    empties = [sum(1 for t in comps if not t.strip()) for comps in rollouts]
    n_empty = sum(empties)
    bad = [i for i, e in enumerate(empties) if (CEILING_N_ROLLOUTS - e) < CEILING_MIN_KEPT_PER_CELL]
    if bad or n_empty > MAX_EMPTY_DROP_FRAC * n_slots:
        raise RuntimeError(
            f"ceiling rollouts below registered grain after {EMPTY_RETRY_PASSES} retry "
            f"passes: {n_empty}/{n_slots} empty slots; {len(bad)} cells under "
            f"{CEILING_MIN_KEPT_PER_CELL} kept (first bad cell idx: {bad[0] if bad else 'n/a'})"
        )


def enforce_map_text_floor(rollouts: list[list[str]]) -> None:
    """Registered-grain floor for map-corpus rollouts (counts only, never text)."""
    n = len(rollouts)
    kept = sum(1 for comps in rollouts if comps and comps[0].strip())
    if kept < MAP_MIN_KEPT_FRAC * n:
        raise RuntimeError(
            f"map_corpus rollouts below registered grain after {EMPTY_RETRY_PASSES} retry "
            f"passes: {kept}/{n} kept (< {MAP_MIN_KEPT_FRAC:.2f} floor)"
        )


# ---------------------------------------------------------------------------
# Capture phases
# ---------------------------------------------------------------------------
def phase_grid(model, tokenizer, layers, q_sim, triggers, out_bundle: Path, meta: dict):
    """v_C grid over Q_sim x triggers -> fp16 bundle (chunk-checkpointed)."""
    import torch

    keys = [(ti, trig, qi, q) for ti, trig in enumerate(triggers) for qi, q in enumerate(q_sim)]
    n = len(keys)
    fp = phase_fingerprint("grid", meta, n_rows=n, n_layers=len(layers))
    store = _ChunkStore(out_bundle, fp, ("v_c", "row_meta"))
    done = store.resume_units()
    if done:
        logger.info("[grid] resuming at row %d/%d", done, n)
    t0 = time.time()
    for start in range(done, n, CKPT_EVERY):
        end = min(start + CKPT_EVERY, n)
        rows, row_meta = [], []
        for ti, trig, qi, q in keys[start:end]:
            messages = render_context_messages(trig["prompt"], q)
            rows.append(_capture_v_c(model, tokenizer, messages, layers).to(torch.float16))
            row_meta.append({"trigger_idx": ti, "trigger_label": trig["label"], "q_sim_idx": qi})
        store.append(start, end, {"v_c": torch.stack(rows), "row_meta": row_meta})
        logger.info("[grid] unit %d/%d elapsed=%.0fs", end, n, time.time() - t0)
    payloads = store.load_payloads()
    v_c = torch.cat([p["v_c"] for p in payloads])
    row_meta = [m for p in payloads for m in p["row_meta"]]
    if v_c.shape[0] != n:
        raise RuntimeError(f"grid: assembled {v_c.shape[0]} rows, expected {n}")
    _atomic_torch_save({"v_c": v_c, "row_meta": row_meta, **meta}, out_bundle)
    write_bundle_sidecar(out_bundle, fp)
    store.cleanup()
    logger.info("grid: wrote %s v_c=%s", out_bundle, tuple(v_c.shape))


def phase_mu(
    model,
    tokenizer,
    layers,
    train_jsonl: Path,
    n_rows: int,
    out_bundle: Path,
    meta: dict,
):
    """Streaming means mu_train (v_C over training rows) + mu_A_train (v_A over gold
    answers). Per-row tensors NEVER stored (streaming reduce). Running sums are
    checkpointed atomically every CKPT_EVERY rows with a fingerprinted resume; a
    gold answer whose capture returns None fails LOUD (registered grain is ALL
    training rows — never a silently shrunk mu_A)."""
    import torch

    fp = phase_fingerprint(
        "mu", meta, train_jsonl=train_jsonl.name, n_rows=n_rows, n_layers=len(layers)
    )
    partial = out_bundle.with_name(out_bundle.name + ".partial.pt")
    mu_c = mu_a = None
    n_c = n_a = 0
    next_line_idx = 0
    if partial.exists():
        st = _load_mu_partial(partial, fp)
        if st is not None:
            mu_c, mu_a = st["mu_c_sum"], st["mu_a_sum"]
            n_c, n_a, next_line_idx = st["n_c"], st["n_a"], st["next_line_idx"]
            logger.info("[mu] resuming at line %d (%d/%d rows done)", next_line_idx, n_c, n_rows)
        else:
            logger.warning("[mu] discarding stale/invalid partial: %s", partial)
            partial.unlink()

    t0 = time.time()
    with train_jsonl.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < next_line_idx or not line.strip():
                continue
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
            if v_a is None:
                raise RuntimeError(
                    f"[mu] line {idx} of {train_jsonl.name}: answer-vector capture "
                    "returned None (empty/unalignable gold answer) — registered grain "
                    "is ALL training rows; refusing to silently shrink mu_A"
                )
            mu_a = v_a.to(torch.float32) if mu_a is None else mu_a + v_a.to(torch.float32)
            n_a += 1
            if n_c % CKPT_EVERY == 0:
                _atomic_torch_save(
                    {
                        "fingerprint": fp,
                        "mu_c_sum": mu_c,
                        "mu_a_sum": mu_a,
                        "n_c": n_c,
                        "n_a": n_a,
                        "next_line_idx": idx + 1,
                    },
                    partial,
                )
                logger.info(
                    "[mu] unit %d/%d line=%d elapsed=%.0fs", n_c, n_rows, idx, time.time() - t0
                )
    if n_c == 0:
        raise RuntimeError(f"no training rows in {train_jsonl}")
    mu_c = (mu_c / n_c).to(torch.float16)
    mu_a = (mu_a / n_a).to(torch.float16)
    _atomic_torch_save(
        {"mu_train": mu_c, "mu_a_train": mu_a, "n_c": n_c, "n_a": n_a, **meta}, out_bundle
    )
    write_bundle_sidecar(out_bundle, fp)
    partial.unlink(missing_ok=True)
    logger.info("mu: wrote %s (n_c=%d n_a=%d)", out_bundle, n_c, n_a)


def phase_ceiling_tf(
    model,
    tokenizer,
    layers,
    q_sim,
    triggers,
    rollouts: list[list[str]],
    rollout_drop_stats: dict,
    out_bundle: Path,
    rawcomp_dir: Path,
    meta: dict,
):
    """Teacher-forced v_A per rollout, stored PER-ROLLOUT (chunk-checkpointed).
    Rollout TEXT persisted to raw_completions/ceiling/ (written BEFORE the final
    bundle so the bundle's existence implies the text landed)."""
    import torch

    keys = [
        (ci, ti, trig["label"], trig["prompt"], qi, q)
        for ci, (ti, trig, qi, q) in enumerate(
            (ti, trig, qi, q) for ti, trig in enumerate(triggers) for qi, q in enumerate(q_sim)
        )
    ]
    n_cells = len(keys)
    if len(rollouts) != n_cells:
        raise RuntimeError(f"ceiling: {len(rollouts)} rollout cells != {n_cells} (q,c) keys")
    fp = phase_fingerprint(
        "ceiling_tf", meta, n_cells=n_cells, n_rollouts=CEILING_N_ROLLOUTS, n_layers=len(layers)
    )
    store = _ChunkStore(out_bundle, fp, ("v_a", "row_meta", "n_capture_dropped"))
    done = store.resume_units()
    if done:
        logger.info("[ceiling] resuming at cell %d/%d", done, n_cells)
    t0 = time.time()
    for start in range(done, n_cells, CKPT_EVERY):
        end = min(start + CKPT_EVERY, n_cells)
        v_a_rows, row_meta = [], []
        n_capture_dropped = 0
        for ci, ti, tlabel, tprompt, qi, q in keys[start:end]:
            messages = render_context_messages(tprompt, q)
            for ri, text in enumerate(rollouts[ci]):
                if not text.strip():
                    continue  # counted in rollout_drop_stats (floors already enforced)
                v_a = _capture_v_a(model, tokenizer, messages, text, layers)
                if v_a is None:
                    n_capture_dropped += 1
                    continue
                v_a_rows.append(v_a.to(torch.float16))
                row_meta.append(
                    {
                        "cell_idx": ci,
                        "trigger_idx": ti,
                        "trigger_label": tlabel,
                        "q_sim_idx": qi,
                        "rollout_idx": ri,
                    }
                )
        payload_v_a = (
            torch.stack(v_a_rows)
            if v_a_rows
            else torch.empty(0, len(layers), EXPECTED_HIDDEN, dtype=torch.float16)
        )
        store.append(
            start,
            end,
            {"v_a": payload_v_a, "row_meta": row_meta, "n_capture_dropped": n_capture_dropped},
        )
        logger.info("[ceiling] unit %d/%d elapsed=%.0fs", end, n_cells, time.time() - t0)

    payloads = store.load_payloads()
    v_a = torch.cat([p["v_a"] for p in payloads])
    row_meta = [m for p in payloads for m in p["row_meta"]]
    n_capture_dropped = sum(p["n_capture_dropped"] for p in payloads)

    # Post-capture registered-grain recheck (capture drops can shrink below the
    # text-level floors enforced pre-TF).
    kept_per_cell = [0] * n_cells
    for m in row_meta:
        kept_per_cell[m["cell_idx"]] += 1
    n_slots = rollout_drop_stats["n_slots"]
    n_dropped_total = rollout_drop_stats["n_empty_after_retries"] + n_capture_dropped
    bad = [ci for ci, k in enumerate(kept_per_cell) if k < CEILING_MIN_KEPT_PER_CELL]
    if bad or n_dropped_total > MAX_EMPTY_DROP_FRAC * n_slots:
        raise RuntimeError(
            f"ceiling below registered grain post-capture: {n_dropped_total}/{n_slots} "
            f"slots dropped ({n_capture_dropped} at capture); {len(bad)} cells under "
            f"{CEILING_MIN_KEPT_PER_CELL} kept (first bad cell idx: {bad[0] if bad else 'n/a'})"
        )

    kept_map: dict[int, list[int]] = {}
    for g, m in enumerate(row_meta):
        kept_map.setdefault(m["cell_idx"], []).append(g)
    raw_rows = [
        {
            "trigger_idx": ti,
            "trigger_label": tlabel,
            "q_sim_idx": qi,
            "question": q,
            "completions": rollouts[ci],
            "kept_v_a_idx": kept_map.get(ci, []),
        }
        for ci, ti, tlabel, tprompt, qi, q in keys
    ]
    drop_stats = {**rollout_drop_stats, "n_capture_dropped": n_capture_dropped}
    _write_rawcomp(rawcomp_dir, "ceiling", meta["model"], raw_rows, meta, drop_stats)
    _atomic_torch_save(
        {"v_a": v_a, "row_meta": row_meta, "drop_stats": drop_stats, **meta}, out_bundle
    )
    write_bundle_sidecar(out_bundle, fp)
    store.cleanup()
    logger.info(
        "ceiling: wrote %s v_a=%s (dropped %d/%d slots)",
        out_bundle,
        tuple(v_a.shape),
        n_dropped_total,
        n_slots,
    )


def phase_map_corpus_tf(
    model,
    tokenizer,
    layers,
    prompts: list[str],
    rollouts: list[list[str]],
    rollout_drop_stats: dict,
    out_bundle: Path,
    rawcomp_dir: Path,
    meta: dict,
):
    """Teacher-forced (v_C, v_A) over the 5,000 LMSYS rows (chunk-checkpointed).
    Drop counts explicit; kept fraction fails loud below the registered floor."""
    import torch

    n = len(prompts)
    fp = phase_fingerprint("map_corpus_tf", meta, n_prompts=n, n_layers=len(layers))
    store = _ChunkStore(out_bundle, fp, ("v_c", "v_a", "kept_idx", "dropped"))
    done = store.resume_units()
    if done:
        logger.info("[map_corpus] resuming at row %d/%d", done, n)
    t0 = time.time()
    for start in range(done, n, CKPT_EVERY):
        end = min(start + CKPT_EVERY, n)
        v_c_rows, v_a_rows, kept_idx, dropped = [], [], [], []
        for i in range(start, end):
            comps = rollouts[i]
            text = comps[0] if comps else ""
            if not text.strip():
                dropped.append({"row_idx": i, "reason": "empty_rollout"})
                continue
            messages = [{"role": "user", "content": prompts[i]}]
            v_a = _capture_v_a(model, tokenizer, messages, text, layers)
            if v_a is None:
                dropped.append({"row_idx": i, "reason": "capture_none"})
                continue
            v_c = _capture_v_c(model, tokenizer, messages, layers)
            v_c_rows.append(v_c.to(torch.float16))
            v_a_rows.append(v_a.to(torch.float16))
            kept_idx.append(i)
        empty = torch.empty(0, len(layers), EXPECTED_HIDDEN, dtype=torch.float16)
        store.append(
            start,
            end,
            {
                "v_c": torch.stack(v_c_rows) if v_c_rows else empty,
                "v_a": torch.stack(v_a_rows) if v_a_rows else empty,
                "kept_idx": kept_idx,
                "dropped": dropped,
            },
        )
        logger.info("[map_corpus] unit %d/%d elapsed=%.0fs", end, n, time.time() - t0)

    payloads = store.load_payloads()
    v_c = torch.cat([p["v_c"] for p in payloads])
    v_a = torch.cat([p["v_a"] for p in payloads])
    kept_idx = [i for p in payloads for i in p["kept_idx"]]
    dropped = [d for p in payloads for d in p["dropped"]]
    if len(kept_idx) < MAP_MIN_KEPT_FRAC * n:
        raise RuntimeError(
            f"map_corpus below registered grain post-capture: kept {len(kept_idx)}/{n} "
            f"(< {MAP_MIN_KEPT_FRAC:.2f} floor); drop reasons: "
            f"{ {r['reason'] for r in dropped} }"
        )
    kept_set = set(kept_idx)
    raw_rows = [{"row_idx": i, "completions": rollouts[i], "kept": i in kept_set} for i in range(n)]
    drop_stats = {
        **rollout_drop_stats,
        "n_kept": len(kept_idx),
        "n_dropped": len(dropped),
        "drop_reasons": sorted({d["reason"] for d in dropped}),
    }
    _write_rawcomp(rawcomp_dir, "map_corpus", meta["model"], raw_rows, meta, drop_stats)
    _atomic_torch_save(
        {
            "v_c": v_c,
            "v_a": v_a,
            "kept_row_idx": kept_idx,
            "n_prompts": n,
            "drop_stats": drop_stats,
            **meta,
        },
        out_bundle,
    )
    write_bundle_sidecar(out_bundle, fp)
    store.cleanup()
    logger.info(
        "map_corpus: wrote %s v_c=%s v_a=%s (kept %d/%d)",
        out_bundle,
        tuple(v_c.shape),
        tuple(v_a.shape),
        len(kept_idx),
        n,
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
    """Token Jaccard + SequenceMatcher ratio (key = SEQMATCH_KEY, the spelling the
    mapfit consumer imports); TF-IDF is computed at corpus level in
    phase_text_baselines."""
    from difflib import SequenceMatcher

    ta, tb = set(a.lower().split()), set(b.lower().split())
    jacc = (len(ta & tb) / len(ta | tb)) if (ta | tb) else 0.0
    return {"jaccard": jacc, SEQMATCH_KEY: SequenceMatcher(None, a, b).ratio()}


def phase_text_baselines(banks_dir: Path, setting: str, out_json: Path, meta: dict):
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
    _atomic_write_text(out_json, json.dumps(payload, ensure_ascii=False))
    logger.info("text_baselines: wrote %s (%d triggers)", out_json, len(labels))


# ---------------------------------------------------------------------------
# Persistence + upload
# ---------------------------------------------------------------------------
def _write_rawcomp(
    rawcomp_dir: Path,
    stage: str,
    model_name: str,
    raw_rows: list[dict],
    meta: dict,
    drop_stats: dict | None = None,
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
            "empty_retry_passes": EMPTY_RETRY_PASSES,
        },
        "drop_stats": drop_stats or {},
        "git": meta.get("git"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": raw_rows,
    }
    _atomic_write_text(dest / "raw_completions.json", json.dumps(payload, ensure_ascii=False))


def upload_rawcomp(rawcomp_dir: Path) -> dict[str, str]:
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    return upload_raw_completions_to_data_repo(experiment_name=SLUG, eval_results_dir=rawcomp_dir)


def upload_artifact_file(local_path: Path, path_in_repo: str) -> str:
    """Upload one file (.pt bundle / baselines JSON) to the HF data repo, fail loud."""
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


def _rollout_fingerprint(
    phase: str, model_name: str, model_ident: str, n_prompts: int, n_samples: int
) -> dict:
    return {
        "phase": f"{phase}_rollouts",
        "model": model_name,
        "model_ident": model_ident,
        "n_prompts": n_prompts,
        "n_samples": n_samples,
        "temperature": ROLLOUT_TEMPERATURE,
        "top_p": ROLLOUT_TOP_P,
        "max_tokens": ROLLOUT_MAX_TOKENS,
        "seed": ROLLOUT_SEED,
        "empty_retry_passes": EMPTY_RETRY_PASSES,
    }


def _force_wipe_phase_state(
    phase: str, out_bundle: Path, pred_subdir: Path, model_name: str
) -> None:
    """--force invalidates ALL of the phase's resume state (round-3 g1 Major):
    the final bundle + its sidecar, the chunk store, the mu partial, and the
    phase's rollout sidecar — a forced rerun must never resume from the OLD
    model's rollouts/activations."""
    chunks = out_bundle.parent / (out_bundle.name + ".chunks")
    if chunks.exists():
        shutil.rmtree(chunks, ignore_errors=True)
        logger.info("[force] wiped chunk store %s", chunks)
    targets = [
        out_bundle,
        bundle_sidecar(out_bundle),
        out_bundle.with_name(out_bundle.name + ".partial.pt"),
    ]
    if phase == "ceiling":
        targets.append(pred_subdir / "ceiling.rollouts.json")
    elif phase == "map_corpus":
        targets.append(out_bundle.with_name(f"{model_name}.rollouts.json"))
    for t in targets:
        if t.exists():
            t.unlink()
            logger.info("[force] wiped %s", t)


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
    ap.add_argument(
        "--merged-root",
        default=str(MERGED_ROOT_DEFAULT),
        help="Root for lazy adapter merges (safetensors — data/, never eval_results/)",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--probe-access", action="store_true", help="1-row LMSYS gated read; then exit")
    ap.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="CPU arg-validation only; no GPU")
    args = ap.parse_args()

    if args.gpu_id != 0:
        ap.error(
            "--gpu-id must stay 0: pin the physical GPU via CUDA_VISIBLE_DEVICES in the "
            "LAUNCHER env (the CVD contract, gotchas.md) — a bare nonzero --gpu-id would "
            "silently run on cuda:0"
        )

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

    meta = {
        "model": args.model_name,
        "setting": args.setting,
        "model_ident": resolve_model_identity(args.model, args.adapter),
        "git": _git_meta(),
    }

    if args.phase == "text_baselines":
        # CPU-only; no model load.
        out_json = tensor_dir / f"text_baselines_{args.setting}_{args.model_name}.json"
        if out_json.exists() and not args.force:
            logger.info("[skip] %s exists — skipping compute (--force to redo)", out_json)
        else:
            phase_text_baselines(banks_dir, args.setting, out_json, meta)
        if not args.no_upload:
            upload_artifact_file(out_json, f"{HF_TEXT_BASELINES_PREFIX}/{out_json.name}")
            logger.info("uploaded text_baselines -> %s", HF_TEXT_BASELINES_PREFIX)
        return 0

    # ---- Consumer-contract validation BEFORE any merge / model constructor ----
    # (r1 blocker consumer-contract-post-init: every required file, key, count,
    # and bank contract resolves before GPU-heavy state is allocated.)
    q_sim: list[str] | None = None
    triggers: list[dict] | None = None
    n_train_rows: int | None = None
    lmsys_prompts: list[str] | None = None
    if args.phase in ("grid", "ceiling"):
        q_sim = load_q_sim(banks_dir, args.setting)
        triggers = load_triggers(banks_dir, args.setting)
        if not triggers:
            raise RuntimeError(f"empty trigger bank for setting={args.setting}")
    elif args.phase == "mu":
        n_train_rows = validate_mu_train_jsonl(Path(args.train_jsonl))
        logger.info("[mu] validated %s (%d rows)", Path(args.train_jsonl).name, n_train_rows)
    elif args.phase == "map_corpus":
        lmsys_prompts = reconstruct_lmsys_prompts(LMSYS_N_PROMPTS)
        if len(lmsys_prompts) != LMSYS_N_PROMPTS:
            raise RuntimeError(
                f"LMSYS replay returned {len(lmsys_prompts)} prompts != {LMSYS_N_PROMPTS}"
            )
        assert_lmsys_disjoint(lmsys_prompts, banks_dir)

    # Output paths + per-invocation idempotency. The skip predicate matches the
    # final bundle's SIDECAR fingerprint (never bare presence — round-3 g1
    # Minor 3), which binds to the current model's WEIGHTS identity (round-3 g1
    # Major); --force wipes ALL of the phase's resume state first so a forced
    # rerun can never resume from the old model's rollouts/activations; the
    # upload leg still runs on skip so a crash between compute and upload
    # self-heals.
    pred_subdir = tensor_dir / "predictor_captures" / args.model_name
    pred_subdir.mkdir(parents=True, exist_ok=True)
    if args.phase == "map_corpus":
        out_bundle = tensor_dir / "map_corpus" / f"{args.model_name}.pt"
        out_bundle.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_bundle = pred_subdir / f"{args.phase}.pt"

    if args.phase == "grid":
        expected_fp = phase_fingerprint(
            "grid", meta, n_rows=len(triggers) * len(q_sim), n_layers=EXPECTED_LAYERS
        )
    elif args.phase == "mu":
        expected_fp = phase_fingerprint(
            "mu",
            meta,
            train_jsonl=Path(args.train_jsonl).name,
            n_rows=n_train_rows,
            n_layers=EXPECTED_LAYERS,
        )
    elif args.phase == "ceiling":
        expected_fp = phase_fingerprint(
            "ceiling_tf",
            meta,
            n_cells=len(triggers) * len(q_sim),
            n_rollouts=CEILING_N_ROLLOUTS,
            n_layers=EXPECTED_LAYERS,
        )
    else:  # map_corpus
        expected_fp = phase_fingerprint(
            "map_corpus_tf", meta, n_prompts=LMSYS_N_PROMPTS, n_layers=EXPECTED_LAYERS
        )

    if args.force:
        _force_wipe_phase_state(args.phase, out_bundle, pred_subdir, args.model_name)
    if not args.force and bundle_current(out_bundle, expected_fp):
        logger.info(
            "[skip] %s current under the expected fingerprint — skipping compute "
            "(--force to redo); running upload leg",
            out_bundle,
        )
    else:
        model_path, cleanup = resolve_model(args)
        try:
            t0 = time.time()
            if args.phase in ("ceiling", "map_corpus"):
                # Rollouts FIRST (vLLM), persisted; the HF model loads only after
                # the engine is torn down (never co-resident — r1 minor).
                tok = load_tokenizer(model_path)
                if args.phase == "ceiling":
                    prompt_texts = [
                        tok.apply_chat_template(
                            render_context_messages(trig["prompt"], q),
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for trig in triggers
                        for q in q_sim
                    ]
                    sidecar = pred_subdir / "ceiling.rollouts.json"
                    rollouts, drop_stats = generate_rollouts_with_retry(
                        model_path,
                        prompt_texts,
                        CEILING_N_ROLLOUTS,
                        sidecar,
                        _rollout_fingerprint(
                            "ceiling",
                            args.model_name,
                            meta["model_ident"],
                            len(prompt_texts),
                            CEILING_N_ROLLOUTS,
                        ),
                    )
                    enforce_ceiling_text_floors(rollouts)
                else:
                    prompt_texts = [
                        tok.apply_chat_template(
                            [{"role": "user", "content": p}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for p in lmsys_prompts
                    ]
                    sidecar = out_bundle.with_name(f"{args.model_name}.rollouts.json")
                    rollouts, drop_stats = generate_rollouts_with_retry(
                        model_path,
                        prompt_texts,
                        1,
                        sidecar,
                        _rollout_fingerprint(
                            "map_corpus",
                            args.model_name,
                            meta["model_ident"],
                            len(prompt_texts),
                            1,
                        ),
                    )
                    enforce_map_text_floor(rollouts)

            model, tokenizer, conv = load_hf_model(model_path)
            meta["layer_convention"] = conv
            layers = conv["stored_layers"]

            if args.phase == "grid":
                phase_grid(model, tokenizer, layers, q_sim, triggers, out_bundle, meta)
            elif args.phase == "mu":
                phase_mu(
                    model,
                    tokenizer,
                    layers,
                    Path(args.train_jsonl),
                    n_train_rows,
                    out_bundle,
                    meta,
                )
            elif args.phase == "ceiling":
                phase_ceiling_tf(
                    model,
                    tokenizer,
                    layers,
                    q_sim,
                    triggers,
                    rollouts,
                    drop_stats,
                    out_bundle,
                    rawcomp_dir,
                    meta,
                )
            elif args.phase == "map_corpus":
                phase_map_corpus_tf(
                    model,
                    tokenizer,
                    layers,
                    lmsys_prompts,
                    rollouts,
                    drop_stats,
                    out_bundle,
                    rawcomp_dir,
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
            upload_artifact_file(
                out_bundle, f"{HF_PREDICTOR_PREFIX}/{args.model_name}/{args.phase}.pt"
            )
            logger.info("uploaded predictor bundle -> %s", HF_PREDICTOR_PREFIX)
        elif args.phase == "map_corpus":
            upload_artifact_file(out_bundle, f"{HF_MAP_CORPUS_PREFIX}/{args.model_name}.pt")
            logger.info("uploaded map_corpus bundle -> %s", HF_MAP_CORPUS_PREFIX)

    return 0


if __name__ == "__main__":
    sys.exit(main())
