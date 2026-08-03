"""vLLM batched rollout generation for issue #1739 (round B).

Two generation surfaces:

- ``generate_labeling``: K=5 rollouts per staged context (instruct chat
  template matching #1092's rendering — see ``render_prompt_parts``), used to
  build the per-context labeling DV.
- ``generate_e1_extraction``: the E1 persona-vectors extraction generation
  (5 instruction pairs x 20 extraction questions x 2 signs x 10 rollouts per
  behavior at temperature 1.0), consuming the e1-assets fallback chain
  (local ``data/issue_779/artifacts`` cache -> Sonnet regeneration via
  ``issue779_common.generate_extraction_artifacts``); assets are persisted to
  an ``inputs/`` staging dir for upload.

vLLM is imported LAZILY inside the default generate seam so tests import this
module without GPU deps; the ``generate_fn`` seam is the only fake point.

CONTENT HYGIENE (binding): contexts come from harmful-content / real-user
corpora. This module NEVER logs or raises row text — logs carry ids, counts,
and token lengths only; exception messages are content-stripped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import zlib
from collections.abc import Callable
from pathlib import Path

from explore_persona_space.experiments.issue_1739.constants import (
    K_ROLLOUTS,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)

# --- generation pins (#1092 parity; plan v3) ---------------------------------
# INSTRUCT_REVISION mirrors scripts/issue1092_gpu_phase.py:49 (the tokenizer /
# model revision every #1092 summary was captured under; reused verbatim so the
# new captures join the same representation space).
INSTRUCT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_MODEL_LEN = 8192  # issue1092_gpu_phase.py:62 parity
GEN_MAX_NEW_TOKENS = 1024  # free-generation default (CLAUDE.md, raised 2026-06-24)
GEN_TEMPERATURE = 1.0
GEN_BATCH_MAX = 512  # max prompts per llm.generate call (brief pin; #664 chunking)
PROMPT_TOKEN_BUDGET = MAX_MODEL_LEN - GEN_MAX_NEW_TOKENS  # #952 load-time length gate
# The instruct user-turn header the prefix slice anchors on
# (issue1092_gpu_phase.py:275 `_INSTRUCT_USER_HEADER`).
INSTRUCT_USER_HEADER = "<|im_start|>user\n"

# --- E1 extraction pins (persona-vectors recipe; #779 line) ------------------
E1_N_PAIRS = 5
E1_N_QUESTIONS = 20  # extraction question set (disjoint from the 20 eval questions)
E1_N_ROLLOUTS = 10
E1_SIGNS = ("pos", "neg")
E1_TEMPERATURE = 1.0


def _ensure_repo_root_on_syspath() -> Path:
    """Put the repo root on sys.path so deferred ``scripts.*`` imports resolve.

    Script mode puts the SCRIPT's dir on sys.path[0], not the repo root
    (gotchas.md #823) — guard every deferred ``scripts.issue779_common``
    import with this. Sentinel-asserted so a wrong parents[N] fails loud.
    """
    root = Path(__file__).resolve().parents[4]
    sentinel = root / "scripts" / "issue779_common.py"
    assert sentinel.exists(), f"repo-root sentinel missing: {sentinel}"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_TOKENIZER_CACHE: dict[str, object] = {}


def get_tokenizer(model_name: str = MODEL_NAME, revision: str = INSTRUCT_REVISION):
    """Module-cached tokenizer load (never per-row — HF 429 gotcha, #664)."""
    key = f"{model_name}@{revision}"
    if key not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[key] = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
    return _TOKENIZER_CACHE[key]


def context_messages(row: dict) -> list[dict]:
    """Staged context row -> chat messages (persona/prefix ALWAYS system role)."""
    messages: list[dict] = []
    prefix = row.get("prefix_text") or ""
    if prefix:
        messages.append({"role": "system", "content": prefix})
    messages.append({"role": "user", "content": row["query"]})
    return messages


def render_prompt_parts(tokenizer, messages: list[dict]) -> tuple[str, str]:
    """Render (prefix_text, prompt_text) under the instruct chat template.

    Mirrors ``issue1092_gpu_phase._render_prompt_parts`` (instruct branch):
    ``apply_chat_template(..., add_generation_prompt=True)`` for the prompt;
    the PREFIX ("everything before the user query" — the canonical project
    definition) is sliced off the rendered prompt at its first user-turn
    header, so ``prompt.startswith(prefix)`` holds for bare AND system-bearing
    contexts and the prefix_end capture position stays "last token before the
    query turn".
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    idx = prompt.find(INSTRUCT_USER_HEADER)
    if idx < 0:
        raise ValueError(
            "instruct render lacks a user-turn header; cannot derive the prefix (template drift?)"
        )
    return prompt[:idx], prompt


def _default_render_parts(tokenizer, row: dict) -> tuple[str, str]:
    """The historical render used by every single-user-turn rung."""
    return render_prompt_parts(tokenizer, context_messages(row))


def filter_prompt_budget(
    tokenizer, rendered_prompts: list[str], *, budget: int = PROMPT_TOKEN_BUDGET
) -> tuple[list[int], dict]:
    """Length-gate FORMATTED prompts at load time (#952 gotcha).

    Returns (kept_indices, digest). The digest carries counts + dropped index
    list only — never row text (harmful-bank digest-only discipline).
    """
    kept: list[int] = []
    dropped: list[dict] = []
    for i, prompt in enumerate(rendered_prompts):
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok <= budget:
            kept.append(i)
        else:
            dropped.append({"index": i, "n_tokens": n_tok})
    digest = {
        "n_input": len(rendered_prompts),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "budget": budget,
        "dropped": dropped,
    }
    return kept, digest


def _context_seed(base_seed: int, context_id: str) -> int:
    """Deterministic per-context sampling seed (stable across resumes)."""
    return (base_seed * 1_000_003 + zlib.crc32(context_id.encode("utf-8"))) % (2**31 - 1)


def _gen_fingerprint(**kwargs: object) -> str:
    """Fingerprint over every output-affecting generation constant (resume key)."""
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _default_vllm_generate(
    prompts: list[str],
    *,
    n: int,
    temperature: float,
    max_tokens: int,
    seeds: list[int],
) -> list[list[dict]]:
    """Default vLLM generate seam: batched, chunked at GEN_BATCH_MAX (#664),
    ``use_tqdm=False`` (#613). Returns per-prompt lists of
    ``{"text", "finish_reason"}`` dicts. Lazy vLLM import so tests import the
    module without GPU deps; the engine is module-cached across calls.
    """
    from vllm import LLM, SamplingParams  # lazy: GPU dep

    if "_llm" not in _TOKENIZER_CACHE:
        engine_kwargs: dict = {
            "model": MODEL_NAME,
            "revision": INSTRUCT_REVISION,
            "max_model_len": MAX_MODEL_LEN,
            "dtype": "bfloat16",
        }
        # Real-user-corpus hang mitigations (gotchas.md pre-launch checklist).
        if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
            engine_kwargs["enforce_eager"] = True
        if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
            engine_kwargs["enable_prefix_caching"] = False
        _TOKENIZER_CACHE["_llm"] = LLM(**engine_kwargs)
    llm = _TOKENIZER_CACHE["_llm"]

    out: list[list[dict]] = []
    n_chunks = (len(prompts) + GEN_BATCH_MAX - 1) // GEN_BATCH_MAX
    for ci in range(0, len(prompts), GEN_BATCH_MAX):
        chunk = prompts[ci : ci + GEN_BATCH_MAX]
        chunk_seeds = seeds[ci : ci + GEN_BATCH_MAX]
        logger.info(
            "[gen-chunk] chunk %d/%d (%d prompts, n=%d)",
            ci // GEN_BATCH_MAX + 1,
            n_chunks,
            len(chunk),
            n,
        )
        params = [
            SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, seed=s)
            for s in chunk_seeds
        ]
        chunk_out = llm.generate(chunk, params, use_tqdm=False)
        for req in chunk_out:
            out.append([{"text": o.text, "finish_reason": o.finish_reason} for o in req.outputs])
    return out


def labeling_rollout_path(out_root: Path | str, behavior: str, context_id: str, k: int) -> Path:
    """Canonical labeling rollout path (Upload Policy raw-completions layout)."""
    return Path(out_root) / "labeling" / behavior / f"{context_id}_seed{k}.json"


def generate_labeling(
    contexts: list[dict],
    *,
    out_root: Path | str,
    behavior: str,
    k_rollouts: int = K_ROLLOUTS,
    temperature: float = GEN_TEMPERATURE,
    max_new_tokens: int = GEN_MAX_NEW_TOKENS,
    seed: int = 0,
    generate_fn: Callable[..., list[list[dict]]] | None = None,
    tokenizer=None,
    render_fn: Callable[[object, dict], tuple[str, str]] | None = None,
) -> dict:
    """Generate K rollouts per staged context; write per-(context, k) JSONs.

    Per-unit persistence + resume (code-style intra-phase grain, T2 > 50
    units): each context's K rollout files are written the moment its
    generation lands; contexts whose K files already exist under the SAME
    fingerprint are skipped at entry. Returns the phase manifest (counts +
    drop digest — never text).

    ``render_fn(tokenizer, row) -> (prefix_text, prompt_text)`` overrides the
    default single-user-turn render for rungs whose contexts are MULTI-TURN.
    The default (``render_prompt_parts(tok, context_messages(row))``) slices
    the prefix at the FIRST user-turn header, which is exactly right when the
    row has one user turn (a system-prompt persona + the query — every rung
    up to and including pvsynth) and WRONG for a conversation prefix, where
    "everything before the user query" must include the earlier turns. Such a
    rung passes its own last-anchored renderer rather than mutating this
    default, so every existing caller's ``prefix_text`` — and therefore every
    committed ``prefix_end`` capture position — is byte-identical.
    """
    out_root = Path(out_root)
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    gen = generate_fn if generate_fn is not None else _default_vllm_generate
    fingerprint = _gen_fingerprint(
        model=MODEL_NAME,
        revision=INSTRUCT_REVISION,
        k=k_rollouts,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=seed,
        behavior=behavior,
    )

    render = render_fn if render_fn is not None else _default_render_parts
    rendered: list[tuple[dict, str, str]] = []  # (row, prefix_text, prompt_text)
    for row in contexts:
        prefix_text, prompt_text = render(tok, row)
        if not prompt_text.startswith(prefix_text):
            raise ValueError(
                f"render_fn returned a prefix that is not a prefix of the prompt for "
                f"context {row.get('context_id')!r} — capture derives prefix_end from "
                "len(prefix_text) against the prompt's offset mapping, so a non-prefix "
                "silently mis-positions every prefix-arm read"
            )
        rendered.append((row, prefix_text, prompt_text))

    kept_idx, drop_digest = filter_prompt_budget(
        tok, [p for _, _, p in rendered], budget=MAX_MODEL_LEN - max_new_tokens
    )
    if drop_digest["n_dropped"]:
        logger.info("[generate] %s: prompt-budget drops: %s", behavior, drop_digest)

    def _done(row: dict) -> bool:
        for k in range(k_rollouts):
            path = labeling_rollout_path(out_root, behavior, row["context_id"], k)
            if not path.exists():
                return False
            try:
                meta = json.loads(path.read_text()).get("meta", {})
            except (json.JSONDecodeError, OSError):
                return False
            if meta.get("fingerprint") != fingerprint:
                return False
        return True

    pending = [rendered[i] for i in kept_idx if not _done(rendered[i][0])]
    n_resumed = len(kept_idx) - len(pending)
    logger.info(
        "[generate] %s: %d contexts (%d resumed, %d pending, %d budget-dropped)",
        behavior,
        len(contexts),
        n_resumed,
        len(pending),
        drop_digest["n_dropped"],
    )

    t0 = time.time()
    commit = _git_commit()
    n_truncated = 0
    for start in range(0, len(pending), GEN_BATCH_MAX):
        batch = pending[start : start + GEN_BATCH_MAX]
        prompts = [p for _, _, p in batch]
        seeds = [_context_seed(seed, row["context_id"]) for row, _, _ in batch]
        rollouts = gen(
            prompts,
            n=k_rollouts,
            temperature=temperature,
            max_tokens=max_new_tokens,
            seeds=seeds,
        )
        assert len(rollouts) == len(batch), (len(rollouts), len(batch))
        for bi, ((row, prefix_text, prompt_text), row_rollouts) in enumerate(
            zip(batch, rollouts, strict=True)
        ):
            assert len(row_rollouts) == k_rollouts, (row["context_id"], len(row_rollouts))
            for k, ro in enumerate(row_rollouts):
                if ro.get("finish_reason") == "length":
                    n_truncated += 1
                _write_json_atomic(
                    labeling_rollout_path(out_root, behavior, row["context_id"], k),
                    {
                        "context_id": row["context_id"],
                        "behavior": behavior,
                        "split": row.get("split"),
                        "rung": row.get("rung"),
                        "group_key": row.get("group_key"),
                        "rollout_k": k,
                        "query": row["query"],
                        "prefix_text": prefix_text,
                        "prompt_text": prompt_text,
                        "completion": ro["text"],
                        "finish_reason": ro.get("finish_reason"),
                        **(
                            {"answer_aliases": row["answer_aliases"]}
                            if row.get("answer_aliases")
                            else {}
                        ),
                        "meta": {
                            "model": MODEL_NAME,
                            "revision": INSTRUCT_REVISION,
                            "temperature": temperature,
                            "max_new_tokens": max_new_tokens,
                            "seed": seeds[bi],
                            "fingerprint": fingerprint,
                            "git_commit": commit,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        },
                    },
                )
            unit = start + bi + 1
            logger.info(
                "[generate] unit %d/%d %s elapsed=%.0fs",
                unit,
                len(pending),
                row["context_id"],
                time.time() - t0,
            )
    manifest = {
        "behavior": behavior,
        "n_contexts": len(contexts),
        "n_kept": len(kept_idx),
        "n_resumed": n_resumed,
        "n_generated": len(pending),
        "k_rollouts": k_rollouts,
        "n_truncated_rollouts": n_truncated,
        "prompt_budget_drops": drop_digest,
        "fingerprint": fingerprint,
        "git_commit": commit,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_atomic(out_root / "labeling" / behavior / "_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# E1 extraction generation (persona-vectors recipe; #779 asset chain)
# ---------------------------------------------------------------------------


def load_e1_assets(behavior: str, *, inputs_dir: Path | str | None = None) -> dict:
    """Load the E1 extraction assets via the #779 fallback chain.

    Chain (standing concern ``e1-assets``): the local
    ``data/issue_779/artifacts/<trait>.json`` cache (via
    ``issue779_common.load_extraction_artifacts``; ``evil`` returns the
    paper-verbatim artifacts) -> Sonnet regeneration
    (``issue779_common.generate_extraction_artifacts``). A copy is persisted
    under ``inputs_dir`` for the issue1739 ``inputs/`` upload prefix.
    """
    _ensure_repo_root_on_syspath()
    from scripts.issue779_common import (  # deferred; guarded above
        generate_extraction_artifacts,
        load_extraction_artifacts,
    )

    try:
        assets = load_extraction_artifacts(behavior)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.info(
            "[e1-assets] %s: cache miss (%s); regenerating via issue779_common",
            behavior,
            type(exc).__name__,
        )
        assets = generate_extraction_artifacts(behavior)
    for key in ("instruction", "extraction_questions", "eval_prompt"):
        assert key in assets, f"e1 assets for {behavior} missing key {key!r}"
    assert len(assets["instruction"]) >= E1_N_PAIRS, (
        behavior,
        len(assets["instruction"]),
    )
    assert len(assets["extraction_questions"]) >= E1_N_QUESTIONS, (
        behavior,
        len(assets["extraction_questions"]),
    )
    if inputs_dir is not None:
        dest = Path(inputs_dir) / "e1_assets" / f"{behavior}.json"
        _write_json_atomic(dest, assets)
        logger.info("[e1-assets] %s: persisted asset copy to %s", behavior, dest)
    return assets


def e1_rollout_path(out_root: Path | str, behavior: str, pair: int, sign: str, q_idx: int) -> Path:
    return Path(out_root) / "extraction" / behavior / f"pair{pair}_{sign}_q{q_idx:02d}.json"


def generate_e1_extraction(
    behavior: str,
    *,
    out_root: Path | str,
    inputs_dir: Path | str | None = None,
    n_rollouts: int = E1_N_ROLLOUTS,
    temperature: float = E1_TEMPERATURE,
    max_new_tokens: int = GEN_MAX_NEW_TOKENS,
    seed: int = 0,
    generate_fn: Callable[..., list[list[dict]]] | None = None,
    tokenizer=None,
    assets: dict | None = None,
) -> dict:
    """E1 extraction generation: 5 pairs x 20 questions x 2 signs x 10 rollouts.

    One JSON per (pair, sign, question) carrying the ``n_rollouts`` rollouts
    plus rendered prompt/prefix text (capture consumes these verbatim —
    no re-render drift). Resume: existing files under the same fingerprint
    are skipped.
    """
    out_root = Path(out_root)
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    gen = generate_fn if generate_fn is not None else _default_vllm_generate
    if assets is None:
        assets = load_e1_assets(behavior, inputs_dir=inputs_dir)
    pairs = assets["instruction"][:E1_N_PAIRS]
    questions = assets["extraction_questions"][:E1_N_QUESTIONS]
    fingerprint = _gen_fingerprint(
        model=MODEL_NAME,
        revision=INSTRUCT_REVISION,
        n_rollouts=n_rollouts,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=seed,
        behavior=behavior,
        e1=True,
    )

    jobs: list[dict] = []
    for p_idx, pair in enumerate(pairs):
        for sign in E1_SIGNS:
            for q_idx, question in enumerate(questions):
                jobs.append(
                    {
                        "pair": p_idx,
                        "sign": sign,
                        "q_idx": q_idx,
                        "system": pair[sign],
                        "q": question,
                    }
                )

    def _done(job: dict) -> bool:
        path = e1_rollout_path(out_root, behavior, job["pair"], job["sign"], job["q_idx"])
        if not path.exists():
            return False
        try:
            return json.loads(path.read_text()).get("meta", {}).get("fingerprint") == fingerprint
        except (json.JSONDecodeError, OSError):
            return False

    pending = [j for j in jobs if not _done(j)]
    logger.info(
        "[e1-gen] %s: %d jobs (%d resumed, %d pending)",
        behavior,
        len(jobs),
        len(jobs) - len(pending),
        len(pending),
    )

    t0 = time.time()
    commit = _git_commit()
    for start in range(0, len(pending), GEN_BATCH_MAX):
        batch = pending[start : start + GEN_BATCH_MAX]
        rendered = []
        for job in batch:
            messages = [
                {"role": "system", "content": job["system"]},
                {"role": "user", "content": job["q"]},
            ]
            rendered.append(render_prompt_parts(tok, messages))
        prompts = [p for _, p in rendered]
        seeds = [
            _context_seed(seed, f"{behavior}-e1-{j['pair']}-{j['sign']}-{j['q_idx']}")
            for j in batch
        ]
        rollouts = gen(
            prompts, n=n_rollouts, temperature=temperature, max_tokens=max_new_tokens, seeds=seeds
        )
        assert len(rollouts) == len(batch), (len(rollouts), len(batch))
        for bi, (job, (prefix_text, prompt_text), row_rollouts) in enumerate(
            zip(batch, rendered, rollouts, strict=True)
        ):
            assert len(row_rollouts) == n_rollouts, (job["pair"], job["sign"], job["q_idx"])
            _write_json_atomic(
                e1_rollout_path(out_root, behavior, job["pair"], job["sign"], job["q_idx"]),
                {
                    "behavior": behavior,
                    "pair": job["pair"],
                    "sign": job["sign"],
                    "q_idx": job["q_idx"],
                    "question": job["q"],
                    "prefix_text": prefix_text,
                    "prompt_text": prompt_text,
                    "rollouts": [
                        {"text": ro["text"], "finish_reason": ro.get("finish_reason")}
                        for ro in row_rollouts
                    ],
                    "meta": {
                        "model": MODEL_NAME,
                        "revision": INSTRUCT_REVISION,
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "seed": seeds[bi],
                        "fingerprint": fingerprint,
                        "git_commit": commit,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                },
            )
            logger.info(
                "[e1-gen] unit %d/%d pair%d_%s_q%02d elapsed=%.0fs",
                start + bi + 1,
                len(pending),
                job["pair"],
                job["sign"],
                job["q_idx"],
                time.time() - t0,
            )
    manifest = {
        "behavior": behavior,
        "n_jobs": len(jobs),
        "n_generated": len(pending),
        "n_rollouts": n_rollouts,
        "fingerprint": fingerprint,
        "git_commit": commit,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_atomic(out_root / "extraction" / behavior / "_manifest.json", manifest)
    return manifest
