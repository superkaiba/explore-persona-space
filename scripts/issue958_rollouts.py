#!/usr/bin/env python3
"""Issue #958 rollout generation (vLLM): Qwen's own answer per (conv, turn) unit.

Plan §4.3: ``SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024,
seed=42)`` — Source #779 verbatim (issue779_collect.py:558). Prefix caching ON
(turn-k contexts of one conversation share prefixes). Two passes:

- pass 1: ``main`` + ``long`` + ``graft`` units (prompts fully determined by
  the corpus — LMSYS-original assistant answers in prefixes);
- pass 2: ``onpol`` units (prefix' = user 1 + Qwen's OWN answer 1, i.e. the
  ``main:c<ci>:k1`` rollout from pass 1).

Persistence: 500-unit JSON shards written the moment their generations
complete (checkpoint-per-shard; skip-complete resume validates the uid set +
DROPPED set + sampling regime — the regime carries the CORPUS FINGERPRINT, so
shards generated under a different corpus build fail validation and are
regenerated, never silently paired with a rebuilt corpus).
Determinism probe (plan §12.8): the first 5 pass-1 units are regenerated twice
in ONE process and diffed; the match fraction is recorded (report-only).

Empty generations: one retry at a DIFFERENT seed (``ROLLOUT_SEED + 1``,
recorded per unit — the same-seed retry was a deterministic no-op); a unit
still empty after retry is DROPPED with a recorded uid (shard
``dropped_empty`` list), never persisted as a normal unit — capture and the
fit design skip dropped units coherently (the §7 chain-yield kill gate owns
the yield accounting).

``--mock-generate`` (VM smoke, no GPU): deterministic placeholder answers
through the SAME enumeration/prompt-assembly/persistence/resume code path —
only ``llm.generate`` is replaced; the live-vLLM path is exercised on the pod.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# vLLM v1 defaults fork(); the tokenizer is touched before LLM() → spawn (gotchas).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue958_common as C  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_rollouts")

VLLM_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
MAX_MODEL_LEN = C.TOKEN_CAP + C.ROLLOUT_MAX_TOKENS  # 8192; ctx cap + max_new (gotchas)


def _mock_generate(prompts: list[str], seed: int | None = None) -> list[dict]:
    """Deterministic placeholder generations (VM smoke; no model, no GPU).

    sha256-based (NOT Python's salted ``hash``) so mock text is stable across
    processes; ``seed`` enters the digest so the seed-varied empty retry
    produces DIFFERENT text (mirrors the live-path retry semantics).
    """
    import hashlib

    s = C.ROLLOUT_SEED if seed is None else seed
    out = []
    for p in prompts:
        h = int(hashlib.sha256(f"{s}:{p}".encode()).hexdigest()[:8], 16) % 10_000
        out.append({"text": f"Mock answer {h}: acknowledged and elaborated.", "finish": "stop"})
    return out


def _vllm_generate(llm, prompts: list[str], seed: int | None = None) -> list[dict]:
    """Chunked ``llm.generate`` (large-batch deadlock prevention, gotchas.md).

    ``seed`` overrides the regime seed (the empty-generation retry passes
    ``ROLLOUT_SEED + 1`` — a same-seed vLLM retry is deterministic and
    reproduces the same empty output).
    """
    from vllm import SamplingParams

    sp = SamplingParams(
        n=1,
        temperature=C.ROLLOUT_TEMPERATURE,
        top_p=C.ROLLOUT_TOP_P,
        max_tokens=C.ROLLOUT_MAX_TOKENS,
        seed=C.ROLLOUT_SEED if seed is None else seed,
    )
    out: list[dict] = []
    n_chunks = (len(prompts) + VLLM_CHUNK - 1) // VLLM_CHUNK
    for i in range(0, len(prompts), VLLM_CHUNK):
        chunk = prompts[i : i + VLLM_CHUNK]
        logger.info(
            "[vllm-chunk] generate chunk %d/%d (%d prompts)",
            i // VLLM_CHUNK + 1,
            n_chunks,
            len(chunk),
        )
        res = llm.generate(chunk, sp, use_tqdm=False)
        for r in res:
            o = r.outputs[0]
            out.append({"text": o.text, "finish": str(o.finish_reason)})
    return out


def _prompt_for(tokenizer, unit: dict, corpora: dict, main_rollouts: dict | None) -> str:
    """Formatted generation prompt for a unit (context(c,k) + gen suffix)."""
    if unit["set"] == "onpol":
        assert main_rollouts is not None
        k1 = main_rollouts[C.unit_id("main", unit["ci"], 1)]
        msgs = C.onpol_prompt_messages(corpora["main"][unit["ci"]], k1["text"])
    else:
        msgs = C.unit_prompt_messages(unit, corpora)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert text.endswith(C.GENERATION_SUFFIX), "generation prompt suffix drift"
    return text


def _validate_shard(path: Path, expected_uids: set[str], regime: dict) -> bool:
    """True iff an existing rollout shard matches the CURRENT uid set + regime.

    The regime dict carries ``corpus_fingerprint``, so a shard generated under
    a different corpus build fails here and is regenerated. The expected uid
    set is the union of persisted rollouts and recorded dropped-empty units.
    """
    try:
        with open(path) as f:
            blob = json.load(f)
    except Exception:
        return False
    if blob.get("regime") != regime:
        return False
    covered = set(blob.get("rollouts", {})) | set(blob.get("dropped_empty", []))
    return covered == expected_uids


RETRY_SEED = C.ROLLOUT_SEED + 1  # seed-varied empty retry (recorded per unit)


def _run_set(
    unit_set: str,
    units: list[dict],
    *,
    args,
    tokenizer,
    corpora: dict,
    gen_fn,
    main_rollouts: dict | None,
    regime: dict,
    stats: dict,
    predropped: frozenset[str] = frozenset(),
    force_empty: frozenset[str] = frozenset(),
) -> None:
    """Generate + persist one unit set, shard by shard (resume-safe).

    ``predropped`` units (e.g. onpol units whose main-k1 dependency was
    dropped) skip generation and land straight in the shard's
    ``dropped_empty`` record. ``force_empty`` is smoke-only fault injection
    (--force-empty-uids, mock mode): the named units are forced empty before
    AND after the retry so the drop path is exercised end-to-end.
    """
    args.out.mkdir(parents=True, exist_ok=True)
    n_shards = (len(units) + C.SHARD_UNITS - 1) // C.SHARD_UNITS
    for s in range(n_shards):
        path = C.rollout_shard_path(args.out, unit_set, s)
        shard_units = units[s * C.SHARD_UNITS : (s + 1) * C.SHARD_UNITS]
        uids = {u["uid"] for u in shard_units}
        if path.exists() and _validate_shard(path, uids, regime):
            logger.info("[%s shard %d/%d] exists + regime-valid — skip", unit_set, s + 1, n_shards)
            continue
        gen_units = [u for u in shard_units if u["uid"] not in predropped]
        prompts = [_prompt_for(tokenizer, u, corpora, main_rollouts) for u in gen_units]
        gens = gen_fn(prompts)
        if force_empty:  # smoke-only fault injection (asserted mock-mode in main())
            for i, u in enumerate(gen_units):
                if u["uid"] in force_empty:
                    gens[i] = {"text": "", "finish": "stop"}
        # graceful single retry of EMPTY generations at a DIFFERENT seed (the
        # same-seed retry is a deterministic no-op — code-review r1)
        empty_idx = [i for i, g in enumerate(gens) if not g["text"].strip()]
        if empty_idx:
            logger.warning(
                "[%s shard %d] %d empty generations — one retry at seed %d",
                unit_set,
                s,
                len(empty_idx),
                RETRY_SEED,
            )
            retry = gen_fn([prompts[i] for i in empty_idx], seed=RETRY_SEED)
            for i, g in zip(empty_idx, retry, strict=True):
                gens[i] = {**g, "retried": True, "retry_seed": RETRY_SEED}
            if force_empty:
                for i, u in enumerate(gen_units):
                    if u["uid"] in force_empty:
                        gens[i] = {"text": "", "finish": "stop", "retried": True}
            stats["retried_empty"] = stats.get("retried_empty", 0) + len(empty_idx)
        # final-empty units are DROPPED with a recorded uid, never persisted
        rollouts: dict[str, dict] = {}
        dropped = sorted(u["uid"] for u in shard_units if u["uid"] in predropped)
        for u, g in zip(gen_units, gens, strict=True):
            if not g["text"].strip():
                dropped.append(u["uid"])
                continue
            rollouts[u["uid"]] = {
                "text": g["text"],
                "finish_reason": g["finish"],
                **(
                    {"retried": True, "retry_seed": g.get("retry_seed")} if g.get("retried") else {}
                ),
            }
        stats["n_units"] = stats.get("n_units", 0) + len(rollouts)
        stats["n_truncated"] = stats.get("n_truncated", 0) + sum(
            1 for g in gens if g["finish"] == "length"
        )
        stats["n_dropped_final_empty"] = stats.get("n_dropped_final_empty", 0) + (
            len(dropped) - sum(1 for u in shard_units if u["uid"] in predropped)
        )
        stats["n_predropped"] = stats.get("n_predropped", 0) + sum(
            1 for u in shard_units if u["uid"] in predropped
        )
        C.write_json_atomic(
            path,
            {
                "unit_set": unit_set,
                "regime": regime,
                "rollouts": rollouts,
                "dropped_empty": sorted(dropped),
                "metadata": C.reproducibility_metadata(
                    {"script": "issue958_rollouts", "unit_set": unit_set, "shard": s}
                ),
            },
        )
        logger.info(
            "[%s shard %d/%d] saved %d units (%d dropped)",
            unit_set,
            s + 1,
            n_shards,
            len(rollouts),
            len(dropped),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #958 vLLM rollouts (per-turn own answers).")
    ap.add_argument("--corpus", type=Path, default=Path("data/issue_958/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/issue_958/rollouts"))
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--mock-generate", action="store_true", help="VM smoke: no vLLM/GPU")
    ap.add_argument("--skip-determinism-probe", action="store_true")
    ap.add_argument(
        "--force-empty-uids",
        default="",
        help="SMOKE-ONLY fault injection (requires --mock-generate): comma list of uids "
        "forced empty before AND after the retry, exercising the drop path end-to-end",
    )
    args = ap.parse_args()
    force_empty = frozenset(u for u in args.force_empty_uids.split(",") if u)
    assert not force_empty or args.mock_generate, "--force-empty-uids is mock-mode-only (smoke)"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    corpus_fp = C.corpus_fingerprint(args.corpus)
    corpora = {
        "main": C.load_corpus(args.corpus, "main"),
        "long": C.load_corpus(args.corpus, "long"),
    }
    units = C.enumerate_units(args.corpus)
    regime = {
        "model": args.model,
        "temperature": C.ROLLOUT_TEMPERATURE,
        "top_p": C.ROLLOUT_TOP_P,
        "max_tokens": C.ROLLOUT_MAX_TOKENS,
        "seed": C.ROLLOUT_SEED,
        "mock": bool(args.mock_generate),
        "corpus_fingerprint": corpus_fp,  # r2: shard identity includes the corpus
    }

    if args.mock_generate:
        gen_fn = _mock_generate
        logger.warning("[mock] --mock-generate: deterministic placeholders, NO vLLM (VM smoke)")
    else:
        from vllm import LLM

        llm = LLM(
            model=args.model,
            enable_prefix_caching=True,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1",
        )
        gen_fn = lambda prompts, seed=None: _vllm_generate(llm, prompts, seed=seed)  # noqa: E731

    stats: dict = {}
    t0 = time.time()

    # Determinism probe (plan §12.8) — live vLLM only; regenerate 5 units twice
    # in ONE process and diff. Report-only (resume is shard-atomic regardless).
    probe = None
    if not args.mock_generate and not args.skip_determinism_probe:
        probe_units = units["main"][:5]
        probe_prompts = [_prompt_for(tokenizer, u, corpora, None) for u in probe_units]
        g1 = gen_fn(probe_prompts)
        g2 = gen_fn(probe_prompts)
        matches = [a["text"] == b["text"] for a, b in zip(g1, g2, strict=True)]
        probe = {"n": len(matches), "match_frac": sum(matches) / max(len(matches), 1)}
        if probe["match_frac"] < 1.0:
            logger.warning(
                "[determinism] seed-42 regeneration NOT bit-identical (%.2f) — resume stays "
                "shard-atomic; recorded in summary",
                probe["match_frac"],
            )

    for unit_set in ("main", "long", "graft"):
        _run_set(
            unit_set,
            units[unit_set],
            args=args,
            tokenizer=tokenizer,
            corpora=corpora,
            gen_fn=gen_fn,
            main_rollouts=None,
            regime=regime,
            stats=stats,
            force_empty=force_empty,
        )
    main_rollouts = C.load_rollouts(args.out, "main")
    dropped_main = C.load_dropped(args.out, "main")
    predropped_onpol = set()
    for u in units["onpol"]:
        k1 = C.unit_id("main", u["ci"], 1)
        if k1 not in main_rollouts:
            assert k1 in dropped_main, f"onpol needs main k1 for c{u['ci']} (not recorded dropped)"
            predropped_onpol.add(u["uid"])  # dependency dropped → drop-with-record
    _run_set(
        "onpol",
        units["onpol"],
        args=args,
        tokenizer=tokenizer,
        corpora=corpora,
        gen_fn=gen_fn,
        main_rollouts=main_rollouts,
        regime=regime,
        stats=stats,
        predropped=frozenset(predropped_onpol),
        force_empty=force_empty,
    )

    summary = {
        **stats,
        "determinism_probe": probe,
        "regime": regime,
        "n_units_by_set": {k: len(v) for k, v in units.items()},
        "wall_seconds": time.time() - t0,
        "metadata": C.reproducibility_metadata({"script": "issue958_rollouts"}),
    }
    C.write_json_atomic(args.out / "rollouts_summary.json", summary)
    logger.info("DONE: %s", json.dumps({k: v for k, v in summary.items() if k != "metadata"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
