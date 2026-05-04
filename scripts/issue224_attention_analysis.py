#!/usr/bin/env python3
"""Issue #224 — Attention analysis on [ZLT] marker generation.

Mechanistic follow-up to #173. For each of 4 marker-trained Qwen-2.5-7B-Instruct
models (librarian, comedian, villain, software_engineer), and the base model
force-fed the librarian-trained context, capture per-layer per-head attention
patterns at the timestep that emits the first BPE of `[ZLT]` (token id 85113)
and at four control strata (C1: random earlier non-marker, C2: late-position
non-marker, C3: rare-token matched, C4: end-of-answer of [ZLT]-negative
generations). Report system / user / assistant fractions in two segmentations
(A block-inclusive, B content-only/specials-stripped) plus a `specials` bucket
under B. Aggregate to per-(layer, region, persona, condition) means + SEM,
evaluate H1 gates A-F, and emit figures.

Stages
------
0. Preflight  — eager-vs-sdpa rate-equivalence gate (librarian, K=200 each).
1. Generate   — for each persona (and the base on librarian's prompts via §5.6
                force-feed: actually base does not need its own Stage 1 — Stage
                2-base reuses librarian's saved positives), produce up to
                10 [ZLT]-positive completions per question (cap 30 trials).
                Save tokens + marker positions + up to 5 negative gens / q for
                C4.
2. Attention  — reload model with `attn_implementation="eager"`, hook every
                `model.model.layers[i].self_attn` to capture `output[1]`
                (attn_weights), single forward pass on each saved positive,
                compute per-(layer, head) attention sums for each region under
                segmentations A and B at marker timestep + C1/C2/C3 within the
                same gen + C4 from negative gens. Trajectory = system_B over
                last K=10 timesteps. Mid-run safety-rail at 50 librarian
                examples.
3. Aggregate  — pure local. Per-example deltas, cluster-by-example bootstrap
                SEM, per-head |mean|/SEM with split-sample selection (RNG seed
                42, 50/50 by example index), sign-test direction-counts. H1
                gate evaluation, JSON + figures + sample table.

CLI
---
- `--stage preflight`                                  # Stage 0
- `--stage generate --persona <name>`                  # Stage 1, one persona
- `--stage attention --persona <name>`                 # Stage 2, one persona
                                                       # (`base_librarian`
                                                       # for §5.6 force-feed)
- `--stage analyze`                                    # Stage 3, no GPU
- `--stage all`                                        # 0 -> 1 -> 2 -> 3
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Load .env + set HF_HOME before anything else touches huggingface_hub
load_dotenv()

import numpy as np  # noqa: E402

logger = logging.getLogger("issue224")

# ── Constants pinned by the plan (§10) ───────────────────────────────────────
HF_REPO = "superkaiba1/explore-persona-space"
SUBFOLDER_TMPL = "leakage_experiment/marker_{persona}_asst_excluded_medium_seed42"
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Personas analysed: 3 high-rate + 1 low-rate (calibration). Order matters for
# determinism across stages.
PERSONA_LIST: tuple[str, ...] = ("librarian", "villain", "comedian", "software_engineer")
PRIMARY_PERSONA = "librarian"  # base-model force-feed uses this only

# Token ids (verified against Qwen2.5-7B-Instruct tokenizer, A2/A3 in plan)
ZLT_FIRST = 85113  # `[Z`
ZLT_TOKS: frozenset[int] = frozenset({85113, 27404, 60})  # `[Z`, `LT`, `]`
NEWLINE2 = 271  # `\n\n`

# Architecture (Qwen2.5-7B; A3 in plan)
N_LAYERS = 28
N_HEADS = 28

# Stage 1 sampling
NUM_POS_PER_QUESTION = 10
MAX_TRIES_PER_QUESTION = 30
NUM_NEG_PER_QUESTION = 5  # per-question cap on negative-gen captures for C4
MAX_NEW_TOKENS = 256

# Control sampling
N_C1_CONTROLS = 5
N_C2_CONTROLS_MAX = 3  # within [t* - 3, t* - 1]
N_C3_CONTROLS_MAX = 3  # rare-token matched
TRAJECTORY_K = 10  # last K timesteps for trajectory

# Stage 0 preflight
PREFLIGHT_K_SAMPLES = 200
PREFLIGHT_RATE_TOL = 0.05  # ±5 pp absolute

# Mid-run safety rail (§7.3)
MIDRUN_GATE_AFTER = 50  # librarian examples
MIDRUN_GATE_THRESHOLD = 2.0  # |mean|/SEM
MIDRUN_GATE_CONTIG = 3  # contiguous layers

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "issue_224"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_224"

# Sanity-assert tolerance on partition sums (§5.2)
PARTITION_SUM_TOL = 1e-3

# Bootstrap (cluster-by-example) for SEM bars
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


# ── Small helpers ─────────────────────────────────────────────────────────────


def _git_commit() -> str:
    """Return the current git commit short hash, or 'uncommitted' on failure."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "uncommitted"
    if out.returncode != 0:
        return "uncommitted"
    sha = out.stdout.strip()
    return sha if sha else "uncommitted"


def _seed_for(question: str, trial: int, persona: str) -> int:
    """Deterministic per-(persona, question, trial) seed.

    Uses sha1 hash so the same (persona, question, trial) maps to the same
    seed across machines (Python's `hash(str)` is salted per-process).
    """
    h = hashlib.sha1(f"{persona}|{question}|{trial}".encode()).hexdigest()
    return 42 + (int(h[:8], 16) % 10**6)


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _run_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Standard reproducibility metadata block for every output JSON."""
    md: dict[str, Any] = {
        "issue": 224,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_repo": HF_REPO,
        "subfolder_template": SUBFOLDER_TMPL,
        "base_model_id": BASE_MODEL_ID,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
    }
    try:
        import torch as _torch

        md["torch_version"] = _torch.__version__
        md["cuda_available"] = _torch.cuda.is_available()
    except ImportError:
        md["torch_version"] = "unavailable"
        md["cuda_available"] = False
    try:
        import transformers as _tr

        md["transformers_version"] = _tr.__version__
    except ImportError:
        md["transformers_version"] = "unavailable"
    if extra:
        md.update(extra)
    return md


# ── HF Hub revision pin (§10 implementer hand-off, mandatory) ─────────────────


def resolve_pinned_revision(personas: tuple[str, ...] = PERSONA_LIST) -> str:
    """Pick the latest HF Hub commit on `superkaiba1/explore-persona-space`
    that contains every required `leakage_experiment/marker_<persona>_...`
    subfolder, and return its sha. Recorded into run_metadata.json so a future
    push cannot silently mutate the loaded weights.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    required_paths = [SUBFOLDER_TMPL.format(persona=p) + "/model.safetensors" for p in personas]
    commits = api.list_repo_commits(repo_id=HF_REPO)
    if not commits:
        raise RuntimeError(f"No commits found on {HF_REPO}")
    # commits are returned newest-first
    for commit in commits:
        rev = commit.commit_id
        try:
            files = api.list_repo_files(repo_id=HF_REPO, revision=rev)
        except Exception as exc:
            logger.warning("list_repo_files failed at rev=%s: %s", rev[:10], exc)
            continue
        files_set = set(files)
        if all(req in files_set for req in required_paths):
            logger.info(
                "Pinned HF revision %s — contains %d required subfolders",
                rev,
                len(required_paths),
            )
            return rev
    raise RuntimeError(
        f"No revision of {HF_REPO} contains all required subfolders: {required_paths}"
    )


def resolve_base_model_revision(repo_id: str = BASE_MODEL_ID) -> str:
    """Resolve and pin the latest HF Hub commit for the base model
    (default ``Qwen/Qwen2.5-7B-Instruct``).

    Code-review v1 MINOR fix #4: previously the script pinned only the
    fine-tuned `superkaiba1/explore-persona-space` revision; the base
    model was loaded with `revision=None` (i.e., whatever HF Hub serves
    today). That would silently mutate the base force-feed branch
    (Stage 2 §5.6) if Qwen ever pushed an update. Now resolved at startup
    and recorded under `run_metadata.json["base_model_revision"]`.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    commits = api.list_repo_commits(repo_id=repo_id)
    if not commits:
        raise RuntimeError(f"No commits found on {repo_id}")
    rev = commits[0].commit_id
    logger.info("Pinned base-model HF revision %s for %s", rev[:10], repo_id)
    return rev


def _record_revision(rev: str, base_rev: str | None = None) -> None:
    """Persist the pinned revision(s) and run metadata to
    ``eval_results/issue_224/run_metadata.json``.

    Records both the fine-tuned-repo revision (`hf_pinned_revision`) and
    — when provided — the base-model revision (`base_model_revision`).
    """
    _ensure_dirs()
    extras: dict[str, Any] = {"hf_pinned_revision": rev}
    if base_rev is not None:
        extras["base_model_revision"] = base_rev
    md = _run_metadata(extras)
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(md, indent=2) + "\n")


def _load_revision() -> str:
    """Read the pinned revision recorded by an earlier stage."""
    p = OUTPUT_DIR / "run_metadata.json"
    if not p.exists():
        raise RuntimeError(
            f"{p} missing — run an earlier stage (preflight/generate) first to pin a revision."
        )
    md = json.loads(p.read_text())
    rev = md.get("hf_pinned_revision")
    if not rev:
        raise RuntimeError(f"{p} has no hf_pinned_revision field.")
    return rev


def _load_base_revision() -> str:
    """Read the pinned base-model revision recorded by an earlier stage.

    Required for Stage 2 base force-feed (§5.6) — the base model
    `Qwen/Qwen2.5-7B-Instruct` MUST be loaded at the SHA that was current
    when preflight ran, otherwise a Qwen Hub push could silently change
    the comparison's reference attention pattern.
    """
    p = OUTPUT_DIR / "run_metadata.json"
    if not p.exists():
        raise RuntimeError(
            f"{p} missing — run an earlier stage (preflight/generate) first to pin a revision."
        )
    md = json.loads(p.read_text())
    rev = md.get("base_model_revision")
    if not rev:
        raise RuntimeError(
            f"{p} has no base_model_revision field — re-run --stage preflight to "
            "pin both fine-tuned and base revisions."
        )
    return rev


# ── Region boundary computation (§5.2) ────────────────────────────────────────


def region_boundaries(tok, sys_prompt: str, user_q: str) -> tuple[int, int, int]:
    """Return ``(n_sys, n_user, n_asst_header)`` measured in tokens.

    Computed by diffing tokenized chat-template fragments — does NOT trust
    hardcoded offsets. Verified at runtime by the caller against the actual
    full-prompt length (``n_sys + n_user + n_asst_header == prompt_len``).
    """
    msgs_sys = [{"role": "system", "content": sys_prompt}]
    msgs_user = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_q},
    ]
    sys_only_text = tok.apply_chat_template(msgs_sys, tokenize=False)
    sys_user_text = tok.apply_chat_template(msgs_user, tokenize=False)
    full_text = tok.apply_chat_template(msgs_user, tokenize=False, add_generation_prompt=True)
    n_sys = len(tok.encode(sys_only_text, add_special_tokens=False))
    n_sys_user = len(tok.encode(sys_user_text, add_special_tokens=False))
    n_full = len(tok.encode(full_text, add_special_tokens=False))
    return n_sys, n_sys_user - n_sys, n_full - n_sys_user


def structural_token_ids(tok) -> set[int]:
    """Token ids treated as structural (specials-stripped, segmentation B).

    Per plan §5.2 and §5.4 the structural set is exactly:

    - The chat-template control specials ``<|im_start|>``, ``<|im_end|>``,
      ``<|endoftext|>`` (when present in the tokenizer vocab).
    - Token ids whose decoded string is exactly ``"\\n"`` or ``"\\n\\n"``.

    Pure-space BPEs (e.g. ``" "``, ``"  "``, ``" word"``, ``"the "``) are
    **content**, NOT structural — they must NOT be in the segmentation-B
    structural set, otherwise `system_B` over-strips and inflates the
    `specials` bucket. Longer-newline runs (`"\\n\\n\\n"` etc.) are NOT
    included; if Qwen's BPE happens to encode them as a single token they
    will be left in the content bucket, which is the conservative choice
    (matches plan §5.2's "newline tokens (\\n, \\n\\n)" specification).

    Code-review v1 BLOCKER 2 fix: previous version included any pure-space
    BPE up to 4 chars via the `decoded.replace(" ", "") == ""` clause —
    over-strips content for segmentation B. Removed.
    """
    ids: set[int] = set()
    # Named structural specials per plan §5.2 (do NOT include `<|im_sep|>` —
    # not part of the plan's structural set, and not all Qwen tokenizer
    # variants define it).
    for s in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        try:
            tid = tok.convert_tokens_to_ids(s)
        except Exception:
            tid = None
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            ids.add(tid)
    # Newline-only BPEs. Sweep the vocab and add ONLY tokens that decode
    # exactly to "\n" or "\n\n". This intentionally excludes pure-space
    # tokens (which are content) and longer-newline runs.
    try:
        vocab = tok.get_vocab()
    except Exception:
        vocab = {}
    seen_ids: set[int] = set()
    for _tok_str, tid in vocab.items():
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        try:
            decoded = tok.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        if decoded == "\n" or decoded == "\n\n":
            ids.add(tid)
    # Always include `\n\n` (token 271) explicitly (A2/§5.4) — defensive
    # against vocab-sweep edge cases.
    ids.add(NEWLINE2)
    return ids


# ── Stage 0: eager-vs-sdpa preflight ──────────────────────────────────────────


def _zlt_in_text(text: str) -> bool:
    """Case-insensitive substring detector — sanity check vs token-id detector."""
    return "[zlt]" in text.lower()


def _zlt_in_token_ids(token_ids: list[int]) -> bool:
    """Token-id detector for [ZLT] in a generation."""
    return ZLT_FIRST in token_ids


def _generate_one(
    model, tok, prompt_text: str, seed: int, attn_mode: str
) -> tuple[list[int], int, str]:
    """Generate a single completion. Returns (full_ids, prompt_len, decoded_text)."""
    import torch

    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    full_ids = out[0].tolist()
    gen_ids = full_ids[prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=False)
    # Defensive: confirm the model loaded with the requested attention impl.
    actual_impl = getattr(model.config, "_attn_implementation", None)
    if attn_mode == "eager" and actual_impl != "eager":
        raise RuntimeError(
            f"Requested attn_implementation='eager' but model.config._attn_implementation="
            f"{actual_impl!r}. Refusing to silently fall back."
        )
    return full_ids, prompt_len, text


def stage0_preflight() -> dict[str, Any]:
    """Eager-vs-sdpa rate-equivalence gate (librarian only)."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.trait_scorers import evaluate_markers
    from explore_persona_space.personas import EVAL_QUESTIONS, PERSONAS

    logger.info("Stage 0: preflight (eager vs sdpa, K=%d)", PREFLIGHT_K_SAMPLES)
    rev = resolve_pinned_revision()
    base_rev = resolve_base_model_revision(BASE_MODEL_ID)
    _record_revision(rev, base_rev=base_rev)

    persona = PRIMARY_PERSONA
    sys_prompt = PERSONAS[persona]
    subfolder = SUBFOLDER_TMPL.format(persona=persona)
    local = snapshot_download(
        HF_REPO,
        allow_patterns=[f"{subfolder}/*"],
        revision=rev,
        token=os.environ.get("HF_TOKEN"),
    )
    model_path = os.path.join(local, subfolder)

    rates: dict[str, float] = {}
    completions_by_mode: dict[str, dict[str, list[str]]] = {}

    for mode in ("eager", "sdpa"):
        logger.info("  loading model with attn_implementation=%s", mode)
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            attn_implementation=mode,
        ).eval()

        # Sanity-assert architecture (A3)
        if model.config.num_hidden_layers != N_LAYERS:
            raise RuntimeError(
                f"Expected num_hidden_layers={N_LAYERS}, got {model.config.num_hidden_layers}"
            )
        if model.config.num_attention_heads != N_HEADS:
            raise RuntimeError(
                f"Expected num_attention_heads={N_HEADS}, got {model.config.num_attention_heads}"
            )
        # Sanity-assert tokenizer (A2)
        bpe = tok.encode("[ZLT]", add_special_tokens=False)
        if bpe != [85113, 27404, 60]:
            raise RuntimeError(f"[ZLT] BPE mismatch: expected [85113, 27404, 60], got {bpe}")
        # Code-review v1 MINOR fix #5: assert `\n\n` BPE id is 271 (NEWLINE2)
        # — structural-set construction (segmentation B) and t_marker
        # boundary handling depend on this constant matching the live
        # tokenizer.
        nl2_bpe = tok.encode("\n\n", add_special_tokens=False)
        if nl2_bpe != [NEWLINE2]:
            raise RuntimeError(f"\\n\\n token-id mismatch: expected [{NEWLINE2}], got {nl2_bpe}")

        comps: dict[str, list[str]] = {q: [] for q in EVAL_QUESTIONS}
        positives = 0
        n_per_q = max(1, PREFLIGHT_K_SAMPLES // len(EVAL_QUESTIONS))
        # 200 samples / 20 questions = 10 each (matched seeds across modes).
        for q in EVAL_QUESTIONS:
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for trial in range(n_per_q):
                seed = _seed_for(q, trial, f"preflight_{persona}")
                _, _, text = _generate_one(model, tok, prompt, seed, mode)
                comps[q].append(text)
                if _zlt_in_text(text):
                    positives += 1

        total = sum(len(v) for v in comps.values())
        rate = positives / total
        logger.info("  mode=%s: %d/%d positive (rate=%.3f)", mode, positives, total, rate)
        rates[mode] = rate
        completions_by_mode[mode] = comps

        # Cross-check token-id detector vs substring detector
        marker_eval = evaluate_markers({persona: comps})
        substr_rate = marker_eval[persona]["rate"]
        if abs(substr_rate - rate) > 1e-9:
            logger.warning(
                "  substring rate (%.4f) and substring-detector rate (%.4f) differ — "
                "expected exact match",
                substr_rate,
                rate,
            )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    delta = abs(rates["eager"] - rates["sdpa"])
    passed = delta <= PREFLIGHT_RATE_TOL
    result = {
        "stage": "preflight",
        "rates": rates,
        "delta": delta,
        "tolerance_pp": PREFLIGHT_RATE_TOL,
        "passed": bool(passed),
        "metadata": _run_metadata({"hf_pinned_revision": rev}),
    }
    out_path = OUTPUT_DIR / "preflight.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("preflight: rates=%s delta=%.4f passed=%s", rates, delta, passed)
    if not passed:
        raise RuntimeError(
            f"Stage-0 preflight FAILED: eager-vs-sdpa rate delta {delta:.4f} > "
            f"{PREFLIGHT_RATE_TOL}. Aborting before Stage 1."
        )
    return result


# ── Stage 1: generate [ZLT]-positive (and -negative) completions ──────────────


def _stage1_collect_for_question(
    *,
    model,
    tok,
    prompt: str,
    persona: str,
    q: str,
    n_sys: int,
    n_user: int,
    n_hdr: int,
    structural: set[int],
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> tuple[int, int]:
    """Sample up to `NUM_POS_PER_QUESTION` positives and `NUM_NEG_PER_QUESTION`
    negatives for one prompt; append to `positives` / `negatives` in place.

    Returns ``(n_pos_for_q, n_neg_for_q)``. Extracted from `stage1_generate`
    purely to keep cyclomatic complexity below the project's mccabe = 15
    ceiling after MINOR-#5 (`\\n\\n` BPE assertion) and MINOR-#6 (multi-`[Z`
    warning) added branches.
    """
    n_pos_for_q = 0
    n_neg_for_q = 0
    for trial in range(MAX_TRIES_PER_QUESTION):
        if n_pos_for_q >= NUM_POS_PER_QUESTION and n_neg_for_q >= NUM_NEG_PER_QUESTION:
            break
        seed = _seed_for(q, trial, persona)
        full_ids, offset, text = _generate_one(model, tok, prompt, seed, "eager")
        gen_ids = full_ids[offset:]
        if _zlt_in_token_ids(gen_ids):
            if n_pos_for_q >= NUM_POS_PER_QUESTION:
                continue
            t_marker_rel = gen_ids.index(ZLT_FIRST)
            t_marker_abs = offset + t_marker_rel
            # Count [Z occurrences for A13 logging
            zcount = gen_ids.count(ZLT_FIRST)
            # Code-review v1 MINOR fix #6: warn when a generation contains
            # more than one `[Z` token. We use the FIRST `[Z` for t_marker
            # (existing behavior preserved); subsequent occurrences are not
            # analyzed. Plan A13 verification requires this notice so
            # downstream interpretation knows per-gen t_marker is first-only.
            if zcount > 1:
                print(
                    f"[stage1] WARNING: gen for q={q[:60]!r} contains "
                    f"{zcount} [Z tokens; using first at t={t_marker_abs}"
                )
            positives.append(
                {
                    "question": q,
                    "trial": trial,
                    "seed": seed,
                    "full_ids": full_ids,
                    "prompt_len": offset,
                    "n_sys": n_sys,
                    "n_user": n_user,
                    "n_asst_header": n_hdr,
                    "t_marker": t_marker_abs,
                    "n_marker_occurrences": zcount,
                    "gen_text": text,
                }
            )
            n_pos_for_q += 1
        else:
            if n_neg_for_q >= NUM_NEG_PER_QUESTION:
                continue
            # Last non-special non-newline2 token before <|im_end|> for C4
            # (§5.4). Trailing <|im_end|> may or may not have been emitted;
            # walk from end skipping eos_id and structural tokens.
            eos_id = tok.eos_token_id
            last_idx = len(full_ids) - 1
            while last_idx >= offset and (
                full_ids[last_idx] == eos_id or full_ids[last_idx] in structural
            ):
                last_idx -= 1
            if last_idx < offset:
                # No usable end-of-answer token: drop this negative.
                continue
            negatives.append(
                {
                    "question": q,
                    "trial": trial,
                    "seed": seed,
                    "full_ids": full_ids,
                    "prompt_len": offset,
                    "n_sys": n_sys,
                    "n_user": n_user,
                    "n_asst_header": n_hdr,
                    "t_eoa": last_idx,  # end-of-answer position (C4)
                    "gen_text": text,
                }
            )
            n_neg_for_q += 1
    return n_pos_for_q, n_neg_for_q


def stage1_generate(persona: str) -> Path:
    """Generate up to `NUM_POS_PER_QUESTION` [ZLT]-positive and up to
    `NUM_NEG_PER_QUESTION` negative completions per `EVAL_QUESTIONS` for
    `persona`. Saves token sequences + marker positions for Stage 2.
    """
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.personas import EVAL_QUESTIONS, PERSONAS

    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona {persona!r}; expected one of {list(PERSONAS)}")

    logger.info("Stage 1 (generate): persona=%s", persona)
    rev = _load_revision()
    sys_prompt = PERSONAS[persona]
    subfolder = SUBFOLDER_TMPL.format(persona=persona)
    local = snapshot_download(
        HF_REPO,
        allow_patterns=[f"{subfolder}/*"],
        revision=rev,
        token=os.environ.get("HF_TOKEN"),
    )
    model_path = os.path.join(local, subfolder)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    # Architecture + tokenizer sanity
    if model.config.num_hidden_layers != N_LAYERS:
        raise RuntimeError(f"Expected {N_LAYERS} layers, got {model.config.num_hidden_layers}")
    if model.config.num_attention_heads != N_HEADS:
        raise RuntimeError(f"Expected {N_HEADS} heads, got {model.config.num_attention_heads}")
    bpe = tok.encode("[ZLT]", add_special_tokens=False)
    if bpe != [85113, 27404, 60]:
        raise RuntimeError(f"[ZLT] BPE mismatch: expected [85113, 27404, 60], got {bpe}")
    # Code-review v1 MINOR fix #5: assert `\n\n` BPE id is 271 (NEWLINE2).
    nl2_bpe = tok.encode("\n\n", add_special_tokens=False)
    if nl2_bpe != [NEWLINE2]:
        raise RuntimeError(f"\\n\\n token-id mismatch: expected [{NEWLINE2}], got {nl2_bpe}")

    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    structural = structural_token_ids(tok)

    for q in EVAL_QUESTIONS:
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # region_boundaries() is later used in Stage 2; verify here once.
        n_sys, n_user, n_hdr = region_boundaries(tok, sys_prompt, q)
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        if n_sys + n_user + n_hdr != prompt_len:
            raise RuntimeError(
                f"region_boundaries mismatch on q={q!r}: n_sys+n_user+n_hdr="
                f"{n_sys + n_user + n_hdr} but prompt_len={prompt_len}"
            )

        n_pos_for_q, n_neg_for_q = _stage1_collect_for_question(
            model=model,
            tok=tok,
            prompt=prompt,
            persona=persona,
            q=q,
            n_sys=n_sys,
            n_user=n_user,
            n_hdr=n_hdr,
            structural=structural,
            positives=positives,
            negatives=negatives,
        )
        logger.info(
            "  q=%r: positives=%d, negatives=%d",
            q[:50],
            n_pos_for_q,
            n_neg_for_q,
        )

    # Free GPU
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_path = OUTPUT_DIR / f"positives_{persona}.json"
    payload: dict[str, Any] = {
        "persona": persona,
        "system_prompt": sys_prompt,
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "positives": positives,
        "negatives": negatives,
        "metadata": _run_metadata({"hf_pinned_revision": rev, "stage": "generate"}),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(
        "Stage 1 done for persona=%s: %d positives, %d negatives -> %s",
        persona,
        len(positives),
        len(negatives),
        out_path,
    )
    return out_path


# ── Stage 2: attention forward pass + hooks ──────────────────────────────────


class AttnCapture:
    """Forward hook that captures `attn_weights = output[1]` from every
    `Qwen2Attention` (`model.model.layers[i].self_attn`) module.
    Requires `attn_implementation="eager"` — sdpa returns ``None``.
    """

    def __init__(self, model):
        self.captures: list[Any] = [None] * N_LAYERS
        self.hooks: list[Any] = []
        for i, layer in enumerate(model.model.layers):
            if not hasattr(layer, "self_attn"):
                raise RuntimeError(f"Layer {i} has no .self_attn attribute")
            self.hooks.append(layer.self_attn.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            # output may be (attn_output, attn_weights) or (attn_output, attn_weights, past_kv)
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError(
                    f"Layer {idx} self_attn output is not a "
                    "(attn_output, attn_weights[, ...]) tuple"
                )
            aw = output[1]
            if aw is None:
                raise RuntimeError(
                    f"Layer {idx} returned attn_weights=None — set attn_implementation='eager'."
                )
            self.captures[idx] = aw.detach()  # (1, n_heads, q_len, k_len)

        return hook

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset(self) -> None:
        self.captures = [None] * N_LAYERS


def _attn_at_t_to_regions(
    captures: list[Any],
    t: int,
    n_sys: int,
    n_user: int,
    n_hdr: int,
    structural: set[int],
    full_ids: list[int],
) -> dict[str, np.ndarray]:
    """For one query timestep `t`, compute per-layer per-head attention sums
    in segmentations A and B. Returns a dict of region -> (n_layers, n_heads).

    Region keys:
      seg A: ``system_A``, ``user_A``, ``asst_A``, ``asst_header_A``
      seg B: ``system_B``, ``user_B``, ``asst_B``, ``specials``
    """
    sys_end = n_sys
    user_end = n_sys + n_user
    asst_header_start = user_end
    asst_start = asst_header_start + n_hdr  # first generated-content position

    out: dict[str, np.ndarray] = {
        "system_A": np.zeros((N_LAYERS, N_HEADS), dtype=np.float64),
        "user_A": np.zeros((N_LAYERS, N_HEADS), dtype=np.float64),
        "asst_A": np.zeros((N_LAYERS, N_HEADS), dtype=np.float64),
        "asst_header_A": np.zeros((N_LAYERS, N_HEADS), dtype=np.float64),
        "system_B": np.zeros((N_LAYERS, N_HEADS), dtype=np.float64),
        "user_B": np.zeros((N_LAYERS, N_HEADS), dtype=np.float64),
        "asst_B": np.zeros((N_LAYERS, N_HEADS), dtype=np.float64),
        "specials": np.zeros((N_LAYERS, N_HEADS), dtype=np.float64),
    }

    # Pre-compute structural mask along source positions [0, t]
    src_len = t + 1
    is_struct = np.array(
        [1 if full_ids[k] in structural else 0 for k in range(src_len)], dtype=bool
    )
    struct_idx = np.where(is_struct)[0]

    sys_slice = slice(0, sys_end)
    user_slice = slice(sys_end, user_end)
    asst_hdr_slice = slice(asst_header_start, min(asst_start, src_len))
    asst_slice = slice(asst_start, src_len) if asst_start < src_len else slice(0, 0)

    for L in range(N_LAYERS):
        cap = captures[L]
        if cap is None:
            raise RuntimeError(f"Layer {L} has no captured attention (forward not run yet?)")
        # cap shape: (1, n_heads, q_len, k_len). Take row t over keys [0, t].
        a = cap[0, :, t, :src_len].float().cpu().numpy()  # (n_heads, src_len)

        # ── Segmentation A (block-inclusive) ──
        out["system_A"][L] = a[:, sys_slice].sum(axis=1)
        out["user_A"][L] = a[:, user_slice].sum(axis=1)
        out["asst_header_A"][L] = a[:, asst_hdr_slice].sum(axis=1)
        out["asst_A"][L] = a[:, asst_slice].sum(axis=1)

        # ── Segmentation B (content-only) ──
        sys_struct = struct_idx[(struct_idx >= 0) & (struct_idx < sys_end)]
        user_struct = struct_idx[(struct_idx >= sys_end) & (struct_idx < user_end)]
        asst_struct = struct_idx[(struct_idx >= asst_header_start) & (struct_idx < src_len)]
        out["system_B"][L] = out["system_A"][L] - a[:, sys_struct].sum(axis=1)
        out["user_B"][L] = out["user_A"][L] - a[:, user_struct].sum(axis=1)
        # asst_B: content-only of [asst_start, t], i.e. exclude asst-header and structural
        if asst_start < src_len:
            asst_content_struct = asst_struct[asst_struct >= asst_start]
            out["asst_B"][L] = out["asst_A"][L] - a[:, asst_content_struct].sum(axis=1)
        else:
            out["asst_B"][L] = 0.0
        # specials = asst_header (always structural, A-bucketed) + every structural
        # position. Use struct_idx to count exhaustively.
        out["specials"][L] = (
            a[:, struct_idx].sum(axis=1)
            + out["asst_header_A"][L]
            - a[:, asst_struct[asst_struct < asst_start]].sum(axis=1)
        )
        # Note: asst_header (asst_header_start..asst_start) is part of `specials` AND
        # part of the structural region, but `struct_idx` already covers the latter
        # since header tokens are mostly structural; we add the non-structural header
        # mass back via asst_header_A net-of-structural to avoid double-counting.

        # Sanity-assert partition exhaustiveness
        sum_A = out["system_A"][L] + out["user_A"][L] + out["asst_header_A"][L] + out["asst_A"][L]
        sum_B = out["system_B"][L] + out["user_B"][L] + out["asst_B"][L] + out["specials"][L]
        # Use mean-over-heads for per-row sum-to-1 check (rows already summed
        # exactly within the row). We expect each head's row to sum to 1.
        for h in range(N_HEADS):
            if abs(sum_A[h] - 1.0) > PARTITION_SUM_TOL:
                raise RuntimeError(
                    f"Segmentation A sum-to-1 violated at layer={L} head={h} t={t}: {sum_A[h]:.6f}"
                )
            if abs(sum_B[h] - 1.0) > PARTITION_SUM_TOL:
                raise RuntimeError(
                    f"Segmentation B sum-to-1 violated at layer={L} head={h} t={t}: {sum_B[h]:.6f}"
                )

    return out


def _select_c1_controls(
    asst_start: int, t_marker: int, full_ids: list[int], rng: np.random.Generator
) -> list[int]:
    """C1: 5 random earlier non-marker positions in [asst_start+5, t_marker-5)."""
    lo = asst_start + 5
    hi = t_marker - 5
    if hi <= lo:
        return []
    candidates = [
        t for t in range(lo, hi) if full_ids[t] not in ZLT_TOKS and full_ids[t] != NEWLINE2
    ]
    if len(candidates) < N_C1_CONTROLS:
        return []  # not enough, signal "skip example"
    return rng.choice(candidates, size=N_C1_CONTROLS, replace=False).tolist()


def _select_c2_controls(t_marker: int, full_ids: list[int]) -> list[int]:
    """C2: up to 3 timesteps in [t_marker-3, t_marker-1] (inclusive of both)."""
    out = []
    for t in (t_marker - 3, t_marker - 2, t_marker - 1):
        if t < 0:
            continue
        if full_ids[t] in ZLT_TOKS or full_ids[t] == NEWLINE2:
            continue
        out.append(t)
    return out[:N_C2_CONTROLS_MAX]


def _select_c3_controls(
    asst_start: int,
    t_marker: int,
    full_ids: list[int],
    structural: set[int],
    vocab_size: int,
    rng: np.random.Generator,
) -> list[int]:
    """C3: up to 3 positions whose token id rank is in the top-5%-rarest among
    emitted non-marker non-structural tokens.

    Plan §5.4 specifies "by base-tokenizer unigram frequency" — we use the
    tokenizer-id ordering as a proxy: in BPE-trained models, lower token id
    correlates with higher unigram frequency (the merge order roughly
    tracks corpus frequency, frequent merges happen earlier and get lower
    ids). This is coarse but matches plan §5.4 / §5.2 of the v3 plan.

    Algorithm: for each candidate position p in the asst span (excluding
    the marker tokens, structural tokens, and t_marker itself), score by
    ``vocab_size - token_id`` (so higher = rarer = larger token id =
    later BPE merge). Take the top 5 % rarest by score, then sample up to
    3 (deterministic via `rng`).

    Code-review v1 MINOR fix #3: previous version used within-generation
    frequency (a position is "rare" if its token id is uncommon *within
    that single generation*). That's a different and weaker calibration
    than the plan asked for — within-generation rarity rewards content
    diversity rather than vocab-level rarity. Reverted to vocab-based
    rarity per plan §5.4.
    """
    asst_lo = asst_start
    asst_hi = t_marker  # exclude marker timestep itself
    span = list(range(asst_lo, asst_hi))
    if not span:
        return []
    cand_positions = [
        k for k in span if full_ids[k] not in ZLT_TOKS and full_ids[k] not in structural
    ]
    if not cand_positions:
        return []
    # Score by "rarity" = vocab_size - token_id (higher token id = rarer).
    scored = sorted(cand_positions, key=lambda k: vocab_size - full_ids[k])
    # Top 5 % rarest = highest rarity score = first 5 % when sorted *descending* by score.
    # Since we sorted ascending by (vocab_size - token_id), the rarest are at the END.
    # Take the top 5 % from the tail (at least 1).
    n_top5 = max(1, len(cand_positions) // 20)
    rarest = scored[-n_top5:]
    # Sample up to N_C3_CONTROLS_MAX from the rarest pool (deterministic via rng).
    if len(rarest) <= N_C3_CONTROLS_MAX:
        return [int(p) for p in rarest]
    picked = rng.choice(rarest, size=N_C3_CONTROLS_MAX, replace=False)
    return [int(p) for p in picked]


def _attention_record(
    captures: list[Any],
    t: int,
    n_sys: int,
    n_user: int,
    n_hdr: int,
    structural: set[int],
    full_ids: list[int],
) -> dict[str, list[list[float]]]:
    """Wrap `_attn_at_t_to_regions` and convert numpy arrays to nested lists
    for JSON serialization. Per-(layer, head) records.
    """
    fr = _attn_at_t_to_regions(captures, t, n_sys, n_user, n_hdr, structural, full_ids)
    return {region: arr.tolist() for region, arr in fr.items()}


def _trajectory_capture(
    captures: list[Any],
    t_end: int,
    n_sys: int,
    n_user: int,
    n_hdr: int,
    structural: set[int],
    full_ids: list[int],
    k: int = TRAJECTORY_K,
) -> list[list[list[float]]]:
    """Capture system_B fraction per layer at the last `k` timesteps t_end-k+1..t_end.

    Returns a list of length k of (n_layers, n_heads) lists.
    """
    asst_start = n_sys + n_user + n_hdr
    out: list[list[list[float]]] = []
    for delta in range(k - 1, -1, -1):
        t = t_end - delta
        if t < asst_start:
            out.append([[0.0] * N_HEADS for _ in range(N_LAYERS)])
            continue
        rec = _attn_at_t_to_regions(captures, t, n_sys, n_user, n_hdr, structural, full_ids)
        out.append(rec["system_B"].tolist())
    return out


def _midrun_safety_rail(
    rows: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Compute |mean|/SEM per layer on `delta[L, system_B, librarian, C1]` and
    check whether any 3-contiguous-layer window has every layer ≥ 2.0.
    Returns (passed, debug-info).
    """
    if not rows:
        return False, {"reason": "no rows"}
    deltas = []
    for r in rows:
        marker_sysB = np.array(r["marker"]["system_B"])  # (L, H)
        # Mean across heads first → per-layer scalar per example.
        marker_per_layer = marker_sysB.mean(axis=1)  # (L,)
        # mean(C1) per example: average system_B across the 5 controls
        c1_arrs = [np.array(c["system_B"]).mean(axis=1) for c in r["c1"]]
        c1_per_layer = np.mean(c1_arrs, axis=0)  # (L,)
        deltas.append(marker_per_layer - c1_per_layer)
    deltas = np.array(deltas)  # (n_examples, L)
    mean = deltas.mean(axis=0)
    sem = deltas.std(axis=0, ddof=1) / np.sqrt(max(1, deltas.shape[0]))
    # Avoid div-by-zero
    sem_safe = np.where(sem > 0, sem, np.inf)
    snr = np.abs(mean) / sem_safe
    contig_windows = []
    for start in range(0, N_LAYERS - MIDRUN_GATE_CONTIG + 1):
        window = snr[start : start + MIDRUN_GATE_CONTIG]
        if np.all(window >= MIDRUN_GATE_THRESHOLD):
            contig_windows.append((int(start), float(window.min())))
    passed = len(contig_windows) > 0
    return passed, {
        "n_examples": deltas.shape[0],
        "snr_per_layer": snr.tolist(),
        "passing_windows": contig_windows,
        "threshold": MIDRUN_GATE_THRESHOLD,
        "contig": MIDRUN_GATE_CONTIG,
    }


def _collect_c4_records(
    *,
    is_base: bool,
    negatives: list[dict[str, Any]],
    model,
    capture: AttnCapture,
    structural: set[int],
) -> list[dict[str, Any]]:
    """End-of-answer attention + last-K trajectory on each [ZLT]-NEGATIVE
    generation (trained-model branch only).

    Code-review v1 BLOCKER 1 fix: emits both the t_eoa snapshot (`record`,
    used by gates A-E and the C4 rule-out) AND the last-K trajectory
    (`trajectory_negative`), so Gate F (`mean_pos[Δt] - mean_neg[Δt]`)
    is evaluable at every relative timestep Δt in {-9, ..., 0}, not only
    at Δt = 0. Per-negative trajectory window is [t_eoa - 9, t_eoa];
    `t_eoa` is the last non-structural-non-eos position (i.e., the
    negative-gen analogue of "end of answer, excluding <|im_end|>").

    Extracted from `stage2_attention` to keep cyclomatic complexity below
    the project's mccabe = 15 ceiling.
    """
    import torch

    if is_base:
        return []
    out: list[dict[str, Any]] = []
    for ex in negatives:
        full_ids = ex["full_ids"]
        n_sys = ex["n_sys"]
        n_user = ex["n_user"]
        n_hdr = ex["n_asst_header"]
        t_eoa = ex["t_eoa"]
        ids_t = torch.tensor(full_ids).unsqueeze(0).to(model.device)
        capture.reset()
        with torch.no_grad():
            _ = model(input_ids=ids_t)
        rec = _attention_record(capture.captures, t_eoa, n_sys, n_user, n_hdr, structural, full_ids)
        traj_neg = _trajectory_capture(
            capture.captures, t_eoa, n_sys, n_user, n_hdr, structural, full_ids
        )
        out.append(
            {
                "question": ex["question"],
                "t_eoa": t_eoa,
                "record": rec,
                "trajectory_negative": traj_neg,
            }
        )
    return out


def stage2_attention(persona: str, base_force_feed_target: str | None = None) -> Path:
    """Forward-pass + attention-hook capture for `persona`.

    If `base_force_feed_target` is set (e.g. "librarian"), load
    `Qwen/Qwen2.5-7B-Instruct` (no LoRA) and force-feed the saved trained-model
    token sequences for `base_force_feed_target`. Captures attention at the
    same matched timesteps (t*, C1, C2, C3) but does NOT compute trajectory
    or use C4 from base — these belong to the trained model only.
    """
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = _load_revision()
    is_base = base_force_feed_target is not None
    if is_base:
        load_target = base_force_feed_target
        positives_path = OUTPUT_DIR / f"positives_{load_target}.json"
        if not positives_path.exists():
            raise RuntimeError(
                f"Base force-feed requires {positives_path} (run --stage generate "
                f"--persona {load_target} first)."
            )
        # Code-review v1 MINOR fix #4: pin the base-model revision recorded by
        # preflight so a Hub push of `Qwen/Qwen2.5-7B-Instruct` cannot silently
        # change the reference pattern between preflight and force-feed.
        base_rev = _load_base_revision()
        logger.info(
            "Stage 2 (attention, BASE force-feed on %s): loading %s @ %s",
            load_target,
            BASE_MODEL_ID,
            base_rev[:10],
        )
        tok = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID, revision=base_rev, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            revision=base_rev,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            attn_implementation="eager",
        ).eval()
        out_persona_label = "base_librarian"
    else:
        positives_path = OUTPUT_DIR / f"positives_{persona}.json"
        subfolder = SUBFOLDER_TMPL.format(persona=persona)
        local = snapshot_download(
            HF_REPO,
            allow_patterns=[f"{subfolder}/*"],
            revision=rev,
            token=os.environ.get("HF_TOKEN"),
        )
        model_path = os.path.join(local, subfolder)
        logger.info("Stage 2 (attention): persona=%s loading %s", persona, model_path)
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            attn_implementation="eager",
        ).eval()
        out_persona_label = persona

    # Confirm hook prerequisite
    if model.config._attn_implementation != "eager":
        raise RuntimeError(f"Expected eager attention, got {model.config._attn_implementation!r}")

    payload = json.loads(positives_path.read_text())
    positives = payload["positives"]
    negatives = payload.get("negatives", [])
    structural = structural_token_ids(tok)
    # Vocab size for C3 vocab-rarity proxy (plan §5.4). Use the size reported
    # by the tokenizer; falls back to the model config if absent.
    try:
        vocab_size = int(tok.vocab_size)
    except (AttributeError, TypeError):
        vocab_size = int(getattr(model.config, "vocab_size", 152064))

    capture = AttnCapture(model)
    rng = np.random.default_rng(42)

    rows: list[dict[str, Any]] = []
    n_skipped_c1 = 0
    midrun_checked = False
    midrun_info: dict[str, Any] = {}

    for ex_idx, ex in enumerate(positives):
        full_ids = ex["full_ids"]
        n_sys = ex["n_sys"]
        n_user = ex["n_user"]
        n_hdr = ex["n_asst_header"]
        t_marker = ex["t_marker"]
        asst_start = n_sys + n_user + n_hdr

        c1 = _select_c1_controls(asst_start, t_marker, full_ids, rng)
        if not c1:
            n_skipped_c1 += 1
            continue
        c2 = _select_c2_controls(t_marker, full_ids)
        c3 = _select_c3_controls(asst_start, t_marker, full_ids, structural, vocab_size, rng)

        ids_t = torch.tensor(full_ids).unsqueeze(0).to(model.device)
        capture.reset()
        with torch.no_grad():
            _ = model(input_ids=ids_t)

        marker_rec = _attention_record(
            capture.captures, t_marker, n_sys, n_user, n_hdr, structural, full_ids
        )
        c1_recs = [
            _attention_record(capture.captures, int(c), n_sys, n_user, n_hdr, structural, full_ids)
            for c in c1
        ]
        c2_recs = [
            _attention_record(capture.captures, int(c), n_sys, n_user, n_hdr, structural, full_ids)
            for c in c2
        ]
        c3_recs = [
            _attention_record(capture.captures, int(c), n_sys, n_user, n_hdr, structural, full_ids)
            for c in c3
        ]
        # Trajectory (positive gen): last K assistant timesteps.
        # Only for trained-model (not base force-feed).
        if not is_base:
            t_end_pos = len(full_ids) - 1
            # Trim trailing eos / structurals so trajectory ends at last content token.
            while t_end_pos > asst_start and (
                full_ids[t_end_pos] == tok.eos_token_id or full_ids[t_end_pos] in structural
            ):
                t_end_pos -= 1
            traj_pos = _trajectory_capture(
                capture.captures, t_end_pos, n_sys, n_user, n_hdr, structural, full_ids
            )
        else:
            traj_pos = []

        rows.append(
            {
                "ex_idx": ex_idx,
                "question": ex["question"],
                "t_marker": t_marker,
                "c1_positions": [int(c) for c in c1],
                "c2_positions": [int(c) for c in c2],
                "c3_positions": [int(c) for c in c3],
                "marker": marker_rec,
                "c1": c1_recs,
                "c2": c2_recs,
                "c3": c3_recs,
                # Renamed: trajectory from positive gen, last K timesteps of system_B
                # (Code-review v1 BLOCKER 1 fix — gate F also needs negatives,
                # captured below in c4_recs_all).
                "trajectory_positive": traj_pos,
            }
        )

        # Mid-run safety rail (librarian only, after 50 examples)
        if (
            not is_base
            and persona == PRIMARY_PERSONA
            and not midrun_checked
            and len(rows) >= MIDRUN_GATE_AFTER
        ):
            midrun_checked = True
            passed, info = _midrun_safety_rail(rows)
            midrun_info = info
            (OUTPUT_DIR / "midrun_gate_librarian.json").write_text(
                json.dumps(
                    {"passed": bool(passed), "info": info, "n_examples": len(rows)},
                    indent=2,
                )
                + "\n"
            )
            if not passed:
                capture.remove()
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Mid-run safety rail FAILED at {len(rows)} librarian examples: "
                    f"no 3-contig-layer window with |mean|/SEM >= "
                    f"{MIDRUN_GATE_THRESHOLD}. Aborting Stage 2."
                )
            logger.info("Mid-run safety rail PASSED: %s", info["passing_windows"])

        if (ex_idx + 1) % 25 == 0:
            logger.info(
                "  persona=%s: processed %d / %d positives (skipped C1=%d)",
                out_persona_label,
                ex_idx + 1,
                len(positives),
                n_skipped_c1,
            )

    # C4: end-of-answer of [ZLT]-NEGATIVE generations (only for trained-model
    # branch; per-prompt-question, not per positive-gen).
    c4_recs_all = _collect_c4_records(
        is_base=is_base,
        negatives=negatives,
        model=model,
        capture=capture,
        structural=structural,
    )

    capture.remove()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_path = OUTPUT_DIR / f"attention_{out_persona_label}.json"
    md_extras: dict[str, Any] = {
        "hf_pinned_revision": rev,
        "stage": "attention",
        "persona": out_persona_label,
    }
    if is_base:
        # Record the pinned base-model SHA on the base force-feed JSON so the
        # consumer can verify (plan §5.6 / code-review v1 MINOR #4).
        md_extras["base_model_revision"] = _load_base_revision()
    body = {
        "persona": out_persona_label,
        "is_base_force_feed": is_base,
        "n_rows": len(rows),
        "n_skipped_c1": n_skipped_c1,
        "rows": rows,
        "c4": c4_recs_all,
        "midrun_info": midrun_info,
        "metadata": _run_metadata(md_extras),
    }
    out_path.write_text(json.dumps(body) + "\n")
    logger.info(
        "Stage 2 done for %s: %d rows -> %s",
        out_persona_label,
        len(rows),
        out_path,
    )
    return out_path


# ── Stage 3: aggregation, statistics, figures ─────────────────────────────────


def _per_example_arrays(rows: list[dict[str, Any]], region: str) -> dict[str, np.ndarray]:
    """Stack per-example (n_examples, n_layers) head-averaged arrays for one region.

    Returns: {"marker", "c1_mean", "c2_mean", "c3_mean"}.
    Empty controls produce NaN rows.
    """
    n = len(rows)
    marker = np.zeros((n, N_LAYERS))
    c1m = np.full((n, N_LAYERS), np.nan)
    c2m = np.full((n, N_LAYERS), np.nan)
    c3m = np.full((n, N_LAYERS), np.nan)
    for i, r in enumerate(rows):
        marker[i] = np.array(r["marker"][region]).mean(axis=1)
        if r["c1"]:
            c1m[i] = np.mean([np.array(c[region]).mean(axis=1) for c in r["c1"]], axis=0)
        if r["c2"]:
            c2m[i] = np.mean([np.array(c[region]).mean(axis=1) for c in r["c2"]], axis=0)
        if r["c3"]:
            c3m[i] = np.mean([np.array(c[region]).mean(axis=1) for c in r["c3"]], axis=0)
    return {"marker": marker, "c1_mean": c1m, "c2_mean": c2m, "c3_mean": c3m}


def _bootstrap_sem(deltas: np.ndarray, n_boot: int = BOOTSTRAP_N) -> np.ndarray:
    """Cluster-by-example bootstrap SEM. `deltas` is (n_examples, n_layers).
    Returns (n_layers,) SEM.
    """
    if deltas.shape[0] < 2:
        return np.full(deltas.shape[1], np.nan)
    # Drop rows with any NaN to avoid nanmean-then-resample inconsistency.
    mask = ~np.isnan(deltas).any(axis=1)
    arr = deltas[mask]
    if arr.shape[0] < 2:
        return np.full(deltas.shape[1], np.nan)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = arr.shape[0]
    means = np.zeros((n_boot, arr.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean(axis=0)
    return means.std(axis=0, ddof=1)


def _per_head_split_sample_selection(
    rows: list[dict[str, Any]], region: str = "system_B", k_top: int = 3
) -> dict[str, list[list[int]]]:
    """Per-(layer, head) selection on split-A by |mean|/SEM of marker - mean(C1).

    Returns top-k head indices per layer, evaluated on split-B; here we just
    return the selection (split-A) — gate evaluation lives elsewhere.
    """
    n = len(rows)
    if n < 4:
        return {"selected_heads_per_layer": [[] for _ in range(N_LAYERS)]}
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    half = n // 2
    split_a = perm[:half]
    split_b = perm[half:]

    def _stack(split):
        # (split_n, n_layers, n_heads) per-example marker - mean(C1) per head
        marker = np.stack([np.array(rows[i]["marker"][region]) for i in split])
        c1m = np.stack(
            [
                np.mean([np.array(c[region]) for c in rows[i]["c1"]], axis=0)
                if rows[i]["c1"]
                else np.full((N_LAYERS, N_HEADS), np.nan)
                for i in split
            ]
        )
        return marker - c1m

    deltas_a = _stack(split_a)
    deltas_b = _stack(split_b)

    selected: list[list[int]] = []
    eval_b: list[dict[str, Any]] = []
    for L in range(N_LAYERS):
        a = deltas_a[:, L, :]  # (split_n, n_heads)
        mask_a = ~np.isnan(a).any(axis=1)
        a_clean = a[mask_a]
        if a_clean.shape[0] < 2:
            selected.append([])
            eval_b.append({"selected": [], "split_b_mean_at_selected": []})
            continue
        mean = a_clean.mean(axis=0)
        sem = a_clean.std(axis=0, ddof=1) / np.sqrt(a_clean.shape[0])
        sem_safe = np.where(sem > 0, sem, np.inf)
        snr = np.abs(mean) / sem_safe
        top = list(np.argsort(snr)[::-1][:k_top].astype(int))
        selected.append(top)

        # Evaluate the SAME selected heads on split-B (no peeking).
        b = deltas_b[:, L, :]
        mask_b = ~np.isnan(b).any(axis=1)
        b_clean = b[mask_b]
        if b_clean.shape[0] < 2:
            eval_b.append({"selected": top, "split_b_mean_at_selected": []})
            continue
        b_mean = b_clean.mean(axis=0)
        eval_b.append(
            {
                "selected": top,
                "split_b_mean_at_selected": [float(b_mean[h]) for h in top],
            }
        )

    return {
        "selected_heads_per_layer": selected,
        "split_b_eval": eval_b,
        "split_a_size": int(half),
        "split_b_size": int(n - half),
        "split_b_indices": [int(i) for i in split_b.tolist()],
    }


def _evaluate_gate_F_trajectory(
    rows: list[dict[str, Any]],
    c4: list[dict[str, Any]],
    h1_window: tuple[int, int] | None,
) -> dict[str, Any]:
    """Plan §7.3-F: trajectory rule-out for "gating decision" reading.

    For each layer, compute ``mean_pos[Δt] - mean_neg[Δt]`` for
    ``Δt ∈ {-9, ..., 0}`` (relative to end-of-generation; Δt=0 is t_eoa for
    negative gens, the last content position for positive gens). Plan §7.3-F
    requires that at the H1 layer band the diff at Δt = -5 is ≤ 50 % of the
    diff at Δt = 0 (marker-emitting timestep). If not, the H1 prose is
    reframed to "concentrates at marker emission" without the
    "gating decision" reading.

    Code-review v1 BLOCKER 1 fix: previously evaluated ONLY at Δt=0 because
    the negative loop captured a single timestep. Now both `trajectory_positive`
    (per-row) and `trajectory_negative` (per c4 entry) cover Δt ∈ {-9..0},
    enabling the full diff trajectory.

    Returns dict with:
      - per-layer per-Δt mean_pos - mean_neg (system_B, head-averaged)
      - per-layer ratio diff[-5] / diff[0]
      - reframe flag (True iff |diff[-5]| > 0.5 * |diff[0]| in H1 band)
    """
    if not rows or not c4:
        return {"applicable": False, "reason": "missing positive or negative trajectories"}

    def _stack_traj(arr_list: list[list[list[list[float]]]]) -> np.ndarray | None:
        # arr_list: list per row, each (K, L, H). Filter empty.
        keep = [np.array(t) for t in arr_list if t]
        if not keep:
            return None
        # Ensure all have same K, L, H
        try:
            stacked = np.stack(keep)  # (n, K, L, H)
        except ValueError:
            return None
        return stacked

    pos_arr = _stack_traj([r.get("trajectory_positive", []) for r in rows])
    neg_arr = _stack_traj([c.get("trajectory_negative", []) for c in c4])
    if pos_arr is None or neg_arr is None:
        return {"applicable": False, "reason": "trajectories missing on one side"}

    # Head-average → (n, K, L)
    pos_h = pos_arr.mean(axis=-1)
    neg_h = neg_arr.mean(axis=-1)
    # Mean over examples → (K, L)
    mean_pos = pos_h.mean(axis=0)
    mean_neg = neg_h.mean(axis=0)
    diff = mean_pos - mean_neg  # (K, L) where K=10, indices 0..9 correspond to Δt = -9..0

    # H1 layer band — if a contig window passed, average across it; else all 28 layers.
    if h1_window is not None:
        lo, hi = h1_window
        band_diff = diff[:, lo : hi + 1].mean(axis=1)  # (K,)
    else:
        band_diff = diff.mean(axis=1)  # (K,)
    # Δt = 0 is index K-1; Δt = -5 is index K-6 (when K=10: idx 4).
    idx_0 = TRAJECTORY_K - 1
    idx_m5 = TRAJECTORY_K - 6
    diff_0 = float(band_diff[idx_0])
    diff_m5 = float(band_diff[idx_m5])
    if abs(diff_0) < 1e-12:
        ratio = float("nan")
        reframe = False
    else:
        ratio = abs(diff_m5) / abs(diff_0)
        reframe = ratio > 0.50

    return {
        "applicable": True,
        "h1_window": list(h1_window) if h1_window else None,
        "diff_per_layer_per_delta": diff.tolist(),  # (K, L)
        "band_diff_per_delta": band_diff.tolist(),  # (K,)
        "diff_at_delta_0": diff_0,
        "diff_at_delta_minus5": diff_m5,
        "ratio_minus5_over_0": ratio,
        "reframe_required": bool(reframe),
        "n_pos": int(pos_arr.shape[0]),
        "n_neg": int(neg_arr.shape[0]),
    }


def _compare_segmentation_a_vs_b(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Plan §7.3-D: check whether the H1 effect persists in segmentation B.

    For each layer, evaluate gate-A pass status (direction ≥ 0.7) under
    segmentation A (`system_A`) AND segmentation B (`system_B`). If A passes
    but B fails for any layer, flag `sink_reframe_required: true` with the
    affected layer indices — the headline must be reframed as sink-loaded
    (the apparent system-attention effect is dominated by attention sinks
    on `<|im_start|>` / `<|im_end|>`, not by content tokens in the system
    block).

    Code-review v1 MINOR fix #7: this comparison was missing.
    """
    arrs_A = _per_example_arrays(rows, "system_A")
    arrs_B = _per_example_arrays(rows, "system_B")
    delta_A = arrs_A["marker"] - arrs_A["c1_mean"]
    delta_B = arrs_B["marker"] - arrs_B["c1_mean"]
    valid_A = ~np.isnan(delta_A).any(axis=1)
    valid_B = ~np.isnan(delta_B).any(axis=1)
    n_A = int(valid_A.sum())
    n_B = int(valid_B.sum())
    dir_A = (delta_A > 0).sum(axis=0) / max(1, n_A)
    dir_B = (delta_B > 0).sum(axis=0) / max(1, n_B)
    pass_A = dir_A >= 0.70
    pass_B = dir_B >= 0.70
    # Layers where A passes but B fails — these are sink-reframe candidates.
    sink_layers = [int(L) for L in range(N_LAYERS) if bool(pass_A[L]) and not bool(pass_B[L])]
    return {
        "direction_seg_A_per_layer": dir_A.tolist(),
        "direction_seg_B_per_layer": dir_B.tolist(),
        "pass_seg_A_per_layer": pass_A.tolist(),
        "pass_seg_B_per_layer": pass_B.tolist(),
        "sink_layers_A_only": sink_layers,
        "sink_reframe_required": bool(sink_layers),
    }


def _evaluate_gates(
    deltas_c1: np.ndarray,
    deltas_c2: np.ndarray | None,
    deltas_c3: np.ndarray | None,
    deltas_c4: np.ndarray | None,
    region_other: dict[str, np.ndarray],
    sem_c1: np.ndarray,
    n: int,
) -> dict[str, Any]:
    """H1 gates A-E on `deltas[L, system_B, persona, C1]` and rule-outs.

    Returns dict with per-layer flags and a contiguous-window summary.
    """
    mean_c1 = np.nanmean(deltas_c1, axis=0)
    # Direction: per-example positive fraction
    direction = (deltas_c1 > 0).sum(axis=0) / max(1, np.sum(~np.isnan(deltas_c1).any(axis=1)))
    # Gate A: direction >= 0.7
    gate_A = direction >= 0.70
    # Gate B: SEM bar excludes zero
    gate_B = (mean_c1 - sem_c1 > 0) | (mean_c1 + sem_c1 < 0)
    # Gate C: system delta > user delta AND > asst delta (means)
    user_d = region_other.get("user_B")
    asst_d = region_other.get("asst_B")
    if user_d is not None and asst_d is not None:
        gate_C = (mean_c1 > np.nanmean(user_d, axis=0)) & (mean_c1 > np.nanmean(asst_d, axis=0))
    else:
        gate_C = np.zeros(N_LAYERS, dtype=bool)
    # Gate D: pattern holds for segmentation B (we ARE on B).
    gate_D = gate_A & gate_B & gate_C

    # Find contiguous windows of length >= 3 where gates A, B, C all hold.
    windows: list[tuple[int, int]] = []
    flag = gate_A & gate_B & gate_C
    i = 0
    while i < N_LAYERS:
        if flag[i]:
            j = i
            while j < N_LAYERS and flag[j]:
                j += 1
            if j - i >= 3:
                windows.append((int(i), int(j - 1)))
            i = j
        else:
            i += 1

    # Rule-outs: each requires (mean_c1 - mean_other) positive AND its SEM-bar excludes zero.
    def _ruleout(other: np.ndarray | None) -> dict[str, Any]:
        if other is None:
            return {"applicable": False}
        diff = deltas_c1 - other
        mean_d = np.nanmean(diff, axis=0)
        sem_d = _bootstrap_sem(diff)
        flag_d = mean_d - sem_d > 0
        return {
            "applicable": True,
            "mean": mean_d.tolist(),
            "sem": sem_d.tolist(),
            "passed_per_layer": flag_d.tolist(),
        }

    return {
        "n_examples": int(n),
        "gate_A_per_layer": gate_A.tolist(),
        "gate_B_per_layer": gate_B.tolist(),
        "gate_C_per_layer": gate_C.tolist(),
        "gate_D_per_layer": gate_D.tolist(),
        "direction_per_layer": direction.tolist(),
        "mean_per_layer": mean_c1.tolist(),
        "sem_per_layer": sem_c1.tolist(),
        "contig_windows_3plus": windows,
        "ruleout_C2_H4": _ruleout(deltas_c2),
        "ruleout_C3_H5": _ruleout(deltas_c3),
        "ruleout_C4_eoa": _ruleout(deltas_c4),
    }


def _aggregate_persona(
    persona_label: str,
    rows: list[dict[str, Any]],
    c4: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate one persona's Stage-2 rows: per-(layer, region) mean+SEM and gates.

    `c4` is the list of negative-gen end-of-answer records (with
    `trajectory_negative`) emitted by `stage2_attention`. Required for
    Gate F trajectory evaluation (plan §7.3-F); pass `None` for the
    base-force-feed branch (no negatives there).
    """
    region_means: dict[str, dict[str, list[float]]] = {}
    deltas_by_region: dict[str, np.ndarray] = {}
    for region in (
        "system_A",
        "user_A",
        "asst_A",
        "asst_header_A",
        "system_B",
        "user_B",
        "asst_B",
        "specials",
    ):
        arrs = _per_example_arrays(rows, region)
        marker_mean = arrs["marker"].mean(axis=0)
        marker_sem = arrs["marker"].std(axis=0, ddof=1) / np.sqrt(max(1, arrs["marker"].shape[0]))
        c1_mean = np.nanmean(arrs["c1_mean"], axis=0)
        c1_sem = _bootstrap_sem(arrs["c1_mean"])
        delta_c1 = arrs["marker"] - arrs["c1_mean"]
        delta_c1_mean = np.nanmean(delta_c1, axis=0)
        delta_c1_sem = _bootstrap_sem(delta_c1)
        region_means[region] = {
            "marker_mean": marker_mean.tolist(),
            "marker_sem": marker_sem.tolist(),
            "c1_mean": c1_mean.tolist(),
            "c1_sem": c1_sem.tolist(),
            "delta_c1_mean": delta_c1_mean.tolist(),
            "delta_c1_sem": delta_c1_sem.tolist(),
        }
        deltas_by_region[region] = delta_c1

    # H1 gates on system_B vs C1
    sys_delta = deltas_by_region["system_B"]
    sem_sys = _bootstrap_sem(sys_delta)
    # C2/C3 deltas on system_B for rule-outs
    arrs_sysB = _per_example_arrays(rows, "system_B")
    delta_c2 = arrs_sysB["marker"] - arrs_sysB["c2_mean"]
    delta_c3 = arrs_sysB["marker"] - arrs_sysB["c3_mean"]

    gates = _evaluate_gates(
        deltas_c1=sys_delta,
        deltas_c2=delta_c2,
        deltas_c3=delta_c3,
        deltas_c4=None,  # filled below if we have C4
        region_other={"user_B": deltas_by_region["user_B"], "asst_B": deltas_by_region["asst_B"]},
        sem_c1=sem_sys,
        n=sys_delta.shape[0],
    )

    # Gate F (trajectory rule-out): pick the H1 layer band from gates' largest
    # contiguous window where gates A, B, C all hold (length ≥ 3). If gate A
    # fails entirely (no contig window), pass `None` and Gate F averages
    # across all layers as a fallback for diagnostic purposes.
    h1_window: tuple[int, int] | None = None
    contig = gates.get("contig_windows_3plus") or []
    if contig:
        # Choose the longest window; tie-broken by lowest start index.
        best = max(contig, key=lambda w: (w[1] - w[0], -w[0]))
        h1_window = (int(best[0]), int(best[1]))
    gate_F_trajectory = _evaluate_gate_F_trajectory(rows, c4 or [], h1_window)

    # Segmentation A vs B comparison (sink rule-out, plan §7.3-D)
    segmentation_ab = _compare_segmentation_a_vs_b(rows)

    head_selection = _per_head_split_sample_selection(rows, region="system_B", k_top=3)

    return {
        "persona": persona_label,
        "n_examples": len(rows),
        "region_means": region_means,
        "gates": gates,
        "gate_F_trajectory": gate_F_trajectory,
        "segmentation_a_vs_b_comparison": segmentation_ab,
        "head_selection_split_sample": head_selection,
    }


def _make_heatmap(
    summary: dict[str, Any], out_dir: Path, persona: str, region: str = "system_B"
) -> None:
    """Layer-by-persona heatmap of mean delta_c1 (head-averaged)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("neurips")
    persona_summaries = summary["per_persona"]
    persona_keys = list(persona_summaries.keys())
    if not persona_keys:
        return
    data = np.array(
        [persona_summaries[p]["region_means"][region]["delta_c1_mean"] for p in persona_keys]
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    vmax = float(np.nanmax(np.abs(data)))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels(range(N_LAYERS), fontsize=6)
    ax.set_yticks(range(len(persona_keys)))
    ax.set_yticklabels(persona_keys)
    ax.set_xlabel("Layer")
    ax.set_title(f"delta(marker - C1) on {region} (head-averaged)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    savefig_paper(fig, "heatmap_marker_vs_control", dir=str(out_dir))
    plt.close(fig)


def _make_per_layer_bar(summary: dict[str, Any], out_dir: Path, region: str = "system_B") -> None:
    """Per-layer mean delta_c1 with SEM error bars, one panel per persona."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    persona_summaries = summary["per_persona"]
    persona_keys = list(persona_summaries.keys())
    if not persona_keys:
        return
    colors = paper_palette(max(1, min(8, len(persona_keys))))
    fig, axes = plt.subplots(
        nrows=len(persona_keys), ncols=1, figsize=(6.5, 1.6 * len(persona_keys)), sharex=True
    )
    if len(persona_keys) == 1:
        axes = [axes]
    for ax, p, color in zip(axes, persona_keys, colors, strict=False):
        means = np.array(persona_summaries[p]["region_means"][region]["delta_c1_mean"])
        sems = np.array(persona_summaries[p]["region_means"][region]["delta_c1_sem"])
        ax.bar(range(N_LAYERS), means, yerr=sems, color=color, capsize=2)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_ylabel(p, fontsize=8)
    axes[-1].set_xlabel("Layer")
    fig.suptitle(f"Marker - C1 delta on {region} per layer (mean ± SEM)")
    fig.tight_layout()
    savefig_paper(fig, "system_attn_per_layer", dir=str(out_dir))
    plt.close(fig)


def _make_sample_table(
    persona: str, attention_path: Path, out_dir: Path, n_examples: int = 5
) -> None:
    """Markdown sample table: question, marker_text, t_marker, c1 positions."""
    payload = json.loads(attention_path.read_text())
    rows = payload["rows"][:n_examples]
    lines = ["# Sample table — issue 224", "", f"Persona: {persona}", ""]
    lines.append("| ex_idx | question | t_marker | c1_positions | c2_positions | c3_positions |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        q = r["question"][:60].replace("|", "/")
        lines.append(
            f"| {r['ex_idx']} | {q} | {r['t_marker']} | "
            f"{r['c1_positions']} | {r['c2_positions']} | {r['c3_positions']} |"
        )
    out = out_dir / "sample_table.md"
    out.write_text("\n".join(lines) + "\n")


def stage3_analyze() -> Path:
    """Aggregate Stage-2 outputs across all personas and emit summary + figures."""
    _ensure_dirs()
    summary: dict[str, Any] = {
        "per_persona": {},
        "metadata": _run_metadata({"stage": "analyze"}),
    }
    for label in (*PERSONA_LIST, "base_librarian"):
        path = OUTPUT_DIR / f"attention_{label}.json"
        if not path.exists():
            logger.warning("Stage 3: %s missing — skipping persona %s", path, label)
            continue
        body = json.loads(path.read_text())
        rows = body["rows"]
        c4 = body.get("c4", [])
        if not rows:
            logger.warning("Stage 3: persona %s has 0 rows — skipping", label)
            continue
        summary["per_persona"][label] = _aggregate_persona(label, rows, c4=c4)
        # Per-persona deltas dump (smaller, for sign-test downstream)
        sys_arrs = _per_example_arrays(rows, "system_B")
        deltas_dump = {
            "persona": label,
            "delta_c1": (sys_arrs["marker"] - sys_arrs["c1_mean"]).tolist(),
            "delta_c2": (sys_arrs["marker"] - sys_arrs["c2_mean"]).tolist(),
            "delta_c3": (sys_arrs["marker"] - sys_arrs["c3_mean"]).tolist(),
        }
        (OUTPUT_DIR / f"per_example_deltas_{label}.json").write_text(json.dumps(deltas_dump) + "\n")

    # Trained-vs-base diff-of-diffs at matched positions on librarian (§5.6)
    if PRIMARY_PERSONA in summary["per_persona"] and "base_librarian" in summary["per_persona"]:
        trained = summary["per_persona"][PRIMARY_PERSONA]["region_means"]["system_B"]
        base = summary["per_persona"]["base_librarian"]["region_means"]["system_B"]
        dod = {
            "trained_delta_c1_mean": trained["delta_c1_mean"],
            "base_delta_c1_mean": base["delta_c1_mean"],
            "diff_of_diffs": [
                t - b for t, b in zip(trained["delta_c1_mean"], base["delta_c1_mean"], strict=True)
            ],
        }
        summary["trained_vs_base_diff_of_diffs"] = dod

    out_path = OUTPUT_DIR / "attention_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    # Figures + sample table (use librarian's attention file for the table)
    _make_heatmap(summary, FIG_DIR, persona="all")
    _make_per_layer_bar(summary, FIG_DIR)
    libr_path = OUTPUT_DIR / f"attention_{PRIMARY_PERSONA}.json"
    if libr_path.exists():
        _make_sample_table(PRIMARY_PERSONA, libr_path, FIG_DIR, n_examples=5)

    logger.info("Stage 3 done -> %s", out_path)
    return out_path


# ── Orchestration ─────────────────────────────────────────────────────────────


def stage_all() -> None:
    """Run stages 0 -> 1 -> 2 -> 3 in order."""
    _ensure_dirs()
    stage0_preflight()
    for p in PERSONA_LIST:
        stage1_generate(p)
    for p in PERSONA_LIST:
        stage2_attention(p)
    # Base force-feed on librarian
    stage2_attention(PRIMARY_PERSONA, base_force_feed_target=PRIMARY_PERSONA)
    stage3_analyze()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "generate", "attention", "analyze", "all"),
    )
    p.add_argument(
        "--persona",
        default=None,
        help="Required for --stage generate/attention. For attention, use 'base_librarian' "
        "to run the §5.6 force-feed on Qwen-2.5-7B-Instruct over librarian's positives.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _ensure_dirs()

    if args.stage == "preflight":
        stage0_preflight()
    elif args.stage == "generate":
        if not args.persona:
            raise SystemExit("--stage generate requires --persona <name>")
        stage1_generate(args.persona)
    elif args.stage == "attention":
        if not args.persona:
            raise SystemExit("--stage attention requires --persona <name>")
        if args.persona == "base_librarian":
            stage2_attention(PRIMARY_PERSONA, base_force_feed_target=PRIMARY_PERSONA)
        else:
            stage2_attention(args.persona)
    elif args.stage == "analyze":
        stage3_analyze()
    elif args.stage == "all":
        stage_all()
    else:
        raise SystemExit(f"Unknown stage {args.stage!r}")


if __name__ == "__main__":
    main()
