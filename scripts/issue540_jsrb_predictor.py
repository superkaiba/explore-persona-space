#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ※, ×, Δ, ‖, ½) in scientific docstrings + log lines.
"""Issue #540 — Canonical Rao-Blackwellized sequence-level JS on the #532 panel.

Replaces the deprecated JS-v1 single-next-token estimator with the canonical
Rao-Blackwellized sequence-level JS (arXiv 2504.10637 §3 +
.claude/rules/persona-distance-metrics.md) and re-fits the JS arm of #532's
416-cell predictor leaderboard. Single manipulated variable = the estimator;
the panel, DV, comparator columns, and analysis code are reused from the
pinned parent script ``scripts/issue532_predictor_stress.py`` @ 296c4da2d
(imported, never modified).

Phases (each persists per-unit JSON before the next starts — checkpoint per
phase; resume-skip accepts an existing file ONLY when its recorded run
parameters match the current invocation — see the artifact-compatibility
section below — otherwise the unit is recomputed (Phase S/T) or the run
fails loud (Phase M/A)):

- **S** sampling: vLLM temp-1 R-samples per (context, probe), one request per
  pair (context, probe) with ``SamplingParams(n=R, seed=42)``; prompts passed
  as token ids (``TokensPrompt``); per-context JSON checkpoints. Sharded over
  workers by context.
- **T** scoring: HF batched teacher-forced forwards (right padding, fp32
  log-softmax over response positions), exact full-vocab per-position
  divergences; per-pair JSON checkpoints. Sharded over workers by pair.
- **M** matrix assembly (CPU, in-parent): ``predictors_jsrb.json``.
- **A** analysis (CPU, in-parent): reproduction control (ported pinned
  phase-3 on unchanged v1 inputs must reproduce eval_results/issue_532/
  analysis.json to ≤1e-9), leaderboard re-fit, direction-pinned paired
  bootstrap Δ = ρ_v1 − ρ_RB (+ clustered variants), hierarchy variants,
  signed-residual + sign-flip + permutation for js_rb.
- **F** figures (CPU, in-parent, paper_plots styling).

Dispatcher/worker unification (smoke IS sweep with one cell): GPU phases
(S, T) are ALWAYS executed in forked worker subprocesses with explicit env
injection. ``--pair-shard k/N`` pins a single-shard grid through the SAME
fork + subprocess + per-pair-JSON path the 4-way sweep uses; without it the
dispatcher builds shards 0..workers-1. M/A/F run in-parent in both.

Out-dir routing (round-2 review fix): when ``--out-dir`` is OMITTED, a
production-shaped run (n_probes=50, r_samples=8, seed=42, max_new_tokens=256,
real base model, no --pairs subset, no --stub-samples) defaults to
``eval_results/issue_540``; ANY other shape (smoke / descope / stub / subset)
defaults to ``eval_results/issue_540_smoke`` so a smoke can never seed
resume-skip artifacts into the production dir. An explicit ``--out-dir`` is
always respected — the parameter-compatibility validation still refuses
mismatched resumes there.

CLI (see plan §10 reproducibility card):
    # FULL (4-way sharded, on pod):
    nohup uv run python scripts/issue540_jsrb_predictor.py \\
        --phases S,T,M,A,F --n-probes 50 --r-samples 8 --seed 42 \\
        --workers 4 --out-dir eval_results/issue_540 \\
        > logs/issue540_full.log 2>&1 &

    # SMOKE (same dispatcher path, one pair; --out-dir omitted →
    # auto-routed to eval_results/issue_540_smoke):
    uv run python scripts/issue540_jsrb_predictor.py --phases S,T \\
        --pairs A1__instr_explicit_1 --n-probes 2 --r-samples 2 \\
        --pair-shard 0/1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Pin HF cache to /workspace on pods; leave system default on the local VM.
if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue532_predictor_stress as i532  # noqa: E402  # pinned parent @ 296c4da2d
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from explore_persona_space.analysis import js_canonical as jsc  # noqa: E402
from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_test_extended_50,
)

load_dotenv()

logger = logging.getLogger("issue540.jsrb")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PARENT_DIR = PROJECT_ROOT / "eval_results" / "issue_532"
DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_540"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_540"
SMOKE_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_540_smoke"
SMOKE_FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_540_smoke"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_SAMPLES_PATH = "issue540_jsrb_canonical/raw_completions"
# v2 (round-2 review fix): per-pair payloads record stub/seed/max_new_tokens
# in metadata so resume-skip can validate them.
# v3 (round-3 review fix): per-pair payloads additionally record
# probes_sha256 + metadata.max_seq_len, and the Phase M/A artifacts
# (predictors_jsrb.json / analysis_jsrb.json) record the FULL compatibility
# tuple (model, stub, seed, max_new_tokens, max_seq_len, probes_sha256,
# n_probes, r_samples) so the Phase A/F loaders validate provenance, not just
# shape. v1/v2 artifacts are structurally incompatible: recomputed (S/T) or
# rejected loud (M/A/F).
SCHEMA_VERSION = "issue540_v3"
PARENT_ARM = "loc"
PARENT_EPOCHS = [1]
PARENT_STYLIZED_DROP = ["A3", "A4", "A5"]  # parent main() default (§6.5 robustness)
GPU_HOURS_BUDGETED = 8.0  # plan §9
DEFAULT_MAX_SEQ_LEN = 1024  # vLLM max_model_len; part of the production card
POS0_WARN = 0.02  # plan §4 on-pod integration checks
POS0_FAIL = 0.05
SELFPAIR_MAX_BITS = 1e-3
REPRO_TOL = 1e-9
PHASE_ORDER = "STMAF"


# ── Panel axes + pair grid ─────────────────────────────────────────────────


def _load_parent_predictors() -> dict:
    """Parent predictors.json — canonical axes + the verbatim v1 columns."""
    path = PARENT_DIR / "predictors.json"
    if not path.exists():
        raise FileNotFoundError(f"parent predictors.json missing at {path}")
    return json.loads(path.read_text())


def _load_parent_phase0() -> dict:
    path = PARENT_DIR / "phase0_base_prior.json"
    if not path.exists():
        raise FileNotFoundError(f"parent phase0_base_prior.json missing at {path}")
    return json.loads(path.read_text())


def _canonical_pairs(sources: list[str], bystanders: list[str]) -> list[tuple[str, str]]:
    """The 280 unique unordered off-diagonal pairs of the 416-cell panel.

    Canonical order: both endpoints indexed by the bystanders list; pair
    (a, b) with index(a) < index(b). 120 ordinary–ordinary (C(16,2)) + 160
    ordinary–instructed. Diagonal cells are 0 analytically (plan §4 Phase T).
    """
    b_idx = {b: i for i, b in enumerate(bystanders)}
    pairs: set[tuple[str, str]] = set()
    for s in sources:
        for b in bystanders:
            if s == b:
                continue
            a, c = (s, b) if b_idx[s] < b_idx[b] else (b, s)
            pairs.add((a, c))
    out = sorted(pairs, key=lambda p: (b_idx[p[0]], b_idx[p[1]]))
    assert len(out) == 280, f"expected 280 unique pairs, got {len(out)}"
    return out


def _pair_id(a: str, b: str) -> str:
    return f"{a}__{b}"


def _parse_pairs_arg(pairs_arg: list[str], bystanders: list[str]) -> list[tuple[str, str]]:
    """Parse explicit ``--pairs A1__instr_explicit_1 ...`` (self-pairs allowed)."""
    out: list[tuple[str, str]] = []
    for p in pairs_arg:
        if "__" not in p:
            raise ValueError(f"--pairs entry {p!r} is not of the form <ctx>__<ctx>")
        a, b = p.split("__", 1)
        for lbl in (a, b):
            if lbl not in bystanders:
                raise ValueError(f"--pairs context {lbl!r} not in the 26-context panel")
        out.append((a, b))
    return out


def _resolve_pair_set(args, sources: list[str], bystanders: list[str]) -> list[tuple[str, str]]:
    """Full run: 280 canonical pairs + the self-pair smoke arm. --pairs: subset."""
    if args.pairs:
        return _parse_pairs_arg(args.pairs, bystanders)
    return [*_canonical_pairs(sources, bystanders), (args.selfpair, args.selfpair)]


def _contexts_for_pairs(pairs: list[tuple[str, str]], bystanders: list[str]) -> list[str]:
    ctxs = {c for p in pairs for c in p}
    return [b for b in bystanders if b in ctxs]


# ── Reproducibility metadata ───────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _git_dirty() -> bool | None:
    """True when tracked files carry uncommitted changes; None when git is
    unavailable. Provenance honesty (round-2 review minor): a recorded SHA
    alone overstates reproducibility when the working tree was dirty at
    artifact-write time (the round-2 smoke recorded HEAD while the round-2
    edits were still uncommitted)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return bool(out.strip())


def _metadata(extra: dict | None = None) -> dict:
    """Standard reproducibility block (CLAUDE.md Code Style)."""
    import datetime
    import platform

    meta = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "base_model": BASE_MODEL,
        "parent_pinned_script_sha": "296c4da2d",
        "estimator": (
            "Rao-Blackwellized sequence-level JS, per-position mixture, base-2, "
            "length-normalized per-token (arXiv 2504.10637 §3 + "
            ".claude/rules/persona-distance-metrics.md); both KL directions + sym-KL in nats"
        ),
    }
    if extra:
        meta.update(extra)
    return meta


def _write_json_atomic(path: Path, payload: dict, indent: int | None = None) -> None:
    """Write JSON via tmp + os.replace so a killed worker never leaves a
    partial/0-byte ``.json`` behind (the resume-skip contamination vector)."""
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_text(json.dumps(payload, indent=indent))
    os.replace(tmp, path)


# ── Artifact compatibility (round-2 review fix: resume-skip validation) ────
#
# Every resume-skip / assembly load validates the artifact's RECORDED run
# parameters against the CURRENT invocation. Mismatch → recompute (Phase S/T
# resume) or hard fail (Phase M/A assembly) — NEVER silently accept or
# min()-downscope. Closes the round-1 binding blocker: a stale smoke /
# descope / stub / tiny-model JSON in the out-dir could be silently skipped
# into the headline 416-cell matrix at the wrong shape.

_PROBES_CACHE: dict[int, list[str]] = {}


def _current_probes(args) -> list[str]:
    """The current invocation's probe list; fail loud if fewer than
    ``--n-probes`` probes exist (no silent downscope at the source)."""
    n = args.n_probes
    if n not in _PROBES_CACHE:
        probes = load_q_test_extended_50()[:n]
        if len(probes) != n:
            raise ValueError(
                f"--n-probes {n} requested but only {len(probes)} probes available "
                "in q_test_extended_50 — pass an honest --n-probes"
            )
        _PROBES_CACHE[n] = probes
    return _PROBES_CACHE[n]


def _probes_sha256(probes: list[str]) -> str:
    """Stable probe-list identity: sha256 over the ORDERED probe texts.

    Persisted in every per-pair / matrix / analysis artifact and validated
    against the current invocation's probe list (round-3 review fix,
    concern pair-probe-identity-validation): a probe-TEXT mutation at
    constant count recomputes samples (full-text validation) but, without
    this hash, stale PAIR artifacts would still resume-skip."""
    return hashlib.sha256(json.dumps(probes, ensure_ascii=False).encode("utf-8")).hexdigest()


def _artifact_field(payload: dict, dotted: str):
    """Resolve ``metadata.stub``-style dotted keys; None when absent."""
    cur: object = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _artifact_mismatches(payload: dict, expected: dict[str, object]) -> list[str]:
    """Recorded-artifact params vs the current invocation; [] when compatible."""
    return [
        f"{key}: artifact={_artifact_field(payload, key)!r} != invocation={want!r}"
        for key, want in expected.items()
        if _artifact_field(payload, key) != want
    ]


def _expected_samples_params(args, ctx: str, probes: list[str]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "sampling",
        "context": ctx,
        "n_probes": len(probes),
        "r_samples": args.r_samples,
        "probes": probes,
        "metadata.stub": bool(args.stub_samples),
        "metadata.seed": args.seed,
        "metadata.max_new_tokens": args.max_new_tokens,
        "metadata.max_seq_len": args.max_seq_len,
        "metadata.base_model": BASE_MODEL,
    }


def _expected_pair_params(args, a: str, b: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "scoring",
        "pair.a": a,
        "pair.b": b,
        "n_probes": args.n_probes,
        "r_samples": args.r_samples,
        # Probe-list IDENTITY, not just count (round-3 fix): a pair scored
        # over different probe texts at the same count is stale.
        "probes_sha256": _probes_sha256(_current_probes(args)),
        "metadata.model": args.model,
        "metadata.stub": bool(args.stub_samples),
        "metadata.seed": args.seed,
        "metadata.max_new_tokens": args.max_new_tokens,
        "metadata.max_seq_len": args.max_seq_len,
    }


def _expected_matrix_params(args) -> dict[str, object]:
    """Full compatibility tuple for predictors_jsrb.json (round-3 review fix,
    concern matrix-artifact-param-validation): Phase A/F must refuse a stale
    matrix computed under a different model / stub flag / seed /
    max_new_tokens / max_seq_len / probe list, not just a different shape."""
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "matrix",
        "n_probes": args.n_probes,
        "r_samples": args.r_samples,
        "probes_sha256": _probes_sha256(_current_probes(args)),
        "metadata.model": args.model,
        "metadata.stub": bool(args.stub_samples),
        "metadata.seed": args.seed,
        "metadata.max_new_tokens": args.max_new_tokens,
        "metadata.max_seq_len": args.max_seq_len,
    }


def _expected_analysis_params(args) -> dict[str, object]:
    """Same tuple for analysis_jsrb.json (concern
    analysis-artifact-param-validation): standalone Phase F must never mix a
    current matrix with a stale leaderboard/hierarchy."""
    return {**_expected_matrix_params(args), "phase": "analysis"}


def _load_matrix(args) -> dict:
    """Strict-validated predictors_jsrb.json load (Phase A panel join +
    Phase F figures). Any tuple mismatch raises naming the fields."""
    payload = _load_validated(
        args.out_dir / "predictors_jsrb.json",
        _expected_matrix_params(args),
        strict=True,
        what="Phase M matrix artifact",
    )
    assert payload is not None  # strict=True raises instead of returning None
    return payload


def _load_analysis(args) -> dict:
    """Strict-validated analysis_jsrb.json load (Phase F figures)."""
    payload = _load_validated(
        args.out_dir / "analysis_jsrb.json",
        _expected_analysis_params(args),
        strict=True,
        what="Phase A analysis artifact",
    )
    assert payload is not None
    return payload


def _load_validated(
    path: Path, expected: dict[str, object], *, strict: bool, what: str
) -> dict | None:
    """Load ``path`` ONLY if it parses and matches ``expected``.

    strict=False (Phase S/T resume-skip): missing / unreadable / mismatched
    → None, the unit is recomputed (loud warning on mismatch).
    strict=True (Phase T input, Phase M/A assembly, Phase F reads): any
    problem raises — downstream phases must never run on stale artifacts.
    """
    if not path.exists():
        if strict:
            raise RuntimeError(f"{what} missing at {path} — run the producing phase first")
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        if strict:
            raise RuntimeError(f"{what} unreadable at {path}: {e}") from e
        logger.warning("%s partial/unreadable at %s (%s) — recomputing", what, path.name, e)
        return None
    mismatches = _artifact_mismatches(payload, expected)
    if mismatches:
        msg = f"stale/incompatible {what} at {path}: " + "; ".join(mismatches)
        if strict:
            raise RuntimeError(
                msg + " — recompute the producing phase with matching parameters "
                "(or use a fresh --out-dir); NEVER assemble stale artifacts"
            )
        logger.warning("%s — RECOMPUTING (resume-skip refused)", msg)
        return None
    return payload


def _load_compatible_samples(args, path: Path, ctx: str, *, strict: bool) -> dict | None:
    probes = _current_probes(args)
    return _load_validated(
        path,
        _expected_samples_params(args, ctx, probes),
        strict=strict,
        what=f"Phase S samples artifact (context {ctx})",
    )


def _load_compatible_pair(args, path: Path, a: str, b: str, *, strict: bool) -> dict | None:
    payload = _load_validated(
        path,
        _expected_pair_params(args, a, b),
        strict=strict,
        what=f"Phase T per-pair artifact ({_pair_id(a, b)})",
    )
    if payload is None:
        return None
    # The pos0≡v1 launch gate must have FIRED for its designated pairs — a
    # params-matching artifact computed before the pair was gated is stale.
    if a != b and _pair_id(a, b) in args.pos0_check_pairs and payload.get("pos0_v1_check") is None:
        msg = (
            f"per-pair artifact {path} matches parameters but carries no pos0_v1_check "
            f"for gated pair {_pair_id(a, b)}"
        )
        if strict:
            raise RuntimeError(msg + " — recompute Phase T so the position-0 gate fires")
        logger.warning("%s — RECOMPUTING (resume-skip refused)", msg)
        return None
    return payload


# ── Phase S: sampling (vLLM, context-sharded) ──────────────────────────────


def _stub_samples(tokenizer, n_probes: int, r_samples: int) -> list[list[dict]]:
    """Synthetic per-(probe, sample) generations for the GPU-less VM smoke.

    Real token ids (tokenized canned texts) so Phase T runs end-to-end on a
    tiny CPU model. One sample is marked finish_reason='length' to exercise
    the truncated/no-append path. Only active under the explicit
    ``--stub-samples`` dev flag — never a silent fallback.
    """
    canned = [
        "The answer is 42.",
        "I think it depends on the context, but generally yes.",
        "Octopuses have three hearts and blue blood, which is wild.",
    ]
    out: list[list[dict]] = []
    for p in range(n_probes):
        row: list[dict] = []
        for s in range(r_samples):
            text = canned[(p + s) % len(canned)]
            raw_ids = tokenizer.encode(text, add_special_tokens=False)
            finish = "length" if (p == 0 and s == r_samples - 1) else "stop"
            ids, action = jsc.apply_terminator_rule(raw_ids, finish)
            row.append(
                {
                    "token_ids": ids,
                    "raw_len": len(raw_ids),
                    "finish_reason": finish,
                    "terminator_action": action,
                    "truncated": finish == "length",
                    "text": text,
                }
            )
        out.append(row)
    return out


def phase_sampling(args, shard_k: int, shard_n: int) -> None:
    """Phase S — sample R on-policy temp-1 responses per (context, probe).

    Persists ``samples/samples_<ctx>.json`` per context (checkpoint per
    phase); resume skips an existing file ONLY when its recorded params match
    the current invocation (else it is recomputed — round-2 review fix).
    Token ids verbatim from vLLM — never retokenized text; the SAME
    ``prompt_token_ids`` are persisted so Phase T conditions on exactly the
    generation prompt.
    """
    from transformers import AutoTokenizer

    predictors = _load_parent_predictors()
    sources, bystanders = predictors["sources"], predictors["bystanders"]
    pair_set = _resolve_pair_set(args, sources, bystanders)
    contexts = _contexts_for_pairs(pair_set, bystanders)
    my_contexts = contexts[shard_k::shard_n]
    logger.info(
        "[worker %d/%d] Phase S: %d/%d contexts: %s",
        shard_k,
        shard_n,
        len(my_contexts),
        len(contexts),
        my_contexts,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"marker id drift: encode({MARKER_TEXT!r}) = {ids} != {[MARKER_ID]}")

    q_test = _current_probes(args)
    class_d = load_class_d_rewrites()
    instructed_panel = i532._instructed_bystander_panel()
    samples_dir = args.out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    llm = None
    sampling_params = None
    first_gen_pinned = False
    for ctx in my_contexts:
        out_path = samples_dir / f"samples_{ctx}.json"
        if _load_compatible_samples(args, out_path, ctx, strict=False) is not None:
            logger.info("[worker %d] Phase S resume-skip %s (params match)", shard_k, out_path.name)
            continue
        t0 = time.time()
        prompt_token_ids = []
        for q in q_test:
            text = i532._build_bystander_prompt(ctx, q, tokenizer, class_d, instructed_panel)
            prompt_token_ids.append(tokenizer.encode(text, add_special_tokens=False))

        if args.stub_samples:
            samples = _stub_samples(tokenizer, len(q_test), args.r_samples)
        else:
            if llm is None:
                from vllm import LLM, SamplingParams

                llm = LLM(
                    model=BASE_MODEL,
                    dtype="bfloat16",
                    max_model_len=args.max_seq_len,
                    gpu_memory_utilization=0.90,
                )
                sampling_params = SamplingParams(
                    n=args.r_samples,
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=args.max_new_tokens,
                    seed=args.seed,
                )
            from vllm.inputs import TokensPrompt

            requests = [TokensPrompt(prompt_token_ids=p) for p in prompt_token_ids]
            outputs = llm.generate(requests, sampling_params)
            samples = []
            for out in outputs:
                row = []
                for comp in out.outputs:
                    raw_ids = list(comp.token_ids)
                    finish = comp.finish_reason
                    if finish not in ("stop", "length"):
                        raise RuntimeError(
                            f"unexpected vLLM finish_reason {finish!r} for ctx={ctx}"
                        )
                    ids2, action = jsc.apply_terminator_rule(raw_ids, finish)
                    if not first_gen_pinned:
                        # Plan §12 A8 implement-time assert: pin WHICH EOS
                        # branch vLLM's token_ids take on this version.
                        logger.info(
                            "[worker %d] EOS-branch pin: finish=%s last_raw_id=%s action=%s",
                            shard_k,
                            finish,
                            raw_ids[-1] if raw_ids else None,
                            action,
                        )
                        first_gen_pinned = True
                    row.append(
                        {
                            "token_ids": ids2,
                            "raw_len": len(raw_ids),
                            "finish_reason": finish,
                            "terminator_action": action,
                            "truncated": finish == "length",
                            "text": comp.text,
                        }
                    )
                if len(row) != args.r_samples:
                    raise RuntimeError(
                        f"vLLM returned {len(row)} completions != n={args.r_samples}"
                    )
                samples.append(row)

        flat = [s for row in samples for s in row]
        action_counts: dict[str, int] = {}
        for s in flat:
            action_counts[s["terminator_action"]] = action_counts.get(s["terminator_action"], 0) + 1
        payload = {
            "schema_version": SCHEMA_VERSION,
            "phase": "sampling",
            "metadata": _metadata(
                {
                    "phase": "S",
                    "stub": bool(args.stub_samples),
                    "max_new_tokens": args.max_new_tokens,
                    "max_seq_len": args.max_seq_len,
                    "seed": args.seed,
                    "sampling": "vLLM temp=1.0 top_p=1.0 TokensPrompt"
                    if not args.stub_samples
                    else "STUB (canned texts, --stub-samples dev flag)",
                }
            ),
            "context": ctx,
            "n_probes": len(q_test),
            "r_samples": args.r_samples,
            "probes": q_test,
            "prompt_token_ids": prompt_token_ids,
            "samples": samples,
            "terminator_action_counts": action_counts,
            "truncation_rate": sum(s["truncated"] for s in flat) / max(1, len(flat)),
        }
        _write_json_atomic(out_path, payload)
        logger.info(
            "[worker %d] Phase S wrote %s (%d probes × %d samples, trunc=%.2f, %.1fs)",
            shard_k,
            out_path.name,
            len(q_test),
            args.r_samples,
            payload["truncation_rate"],
            time.time() - t0,
        )


# ── Phase T: teacher-forced scoring (HF forwards, pair-sharded) ────────────


def _load_samples(args, ctx: str) -> dict:
    """Strict-validated Phase S artifact load (missing / stale → raise).

    Validates schema_version, context, n_probes, r_samples, the probe TEXT
    list, stub flag, seed, and max_new_tokens against the current invocation
    — the round-1 ``min()`` probe-count downscope is gone."""
    path = args.out_dir / "samples" / f"samples_{ctx}.json"
    payload = _load_compatible_samples(args, path, ctx, strict=True)
    assert payload is not None  # strict=True raises instead of returning None
    return payload


def _score_pair(args, model, tokenizer, a: str, b: str, max_batch_holder: dict) -> dict:
    """Score one unordered pair: exact per-position divergences over all
    (probe × sample × side) rows; returns the per-pair payload dict."""
    import torch

    samples_a = _load_samples(args, a)
    samples_b = _load_samples(args, b)
    # Both sides validated against the SAME current probe list (text + count)
    # by _load_samples, so the invocation's n_probes is exact for both.
    n_probes = args.n_probes

    r = args.r_samples
    profile_cap = args.max_new_tokens + 1
    profile_sum = np.zeros(profile_cap)
    profile_cnt = np.zeros(profile_cap, dtype=np.int64)
    per_sample: list[dict] = []
    pos0_per_probe: list[float] = []
    a_kl_m, b_kl_m, a_kl_ab, b_kl_ba = [], [], [], []
    a_kl_m_masked, b_kl_m_masked = [], []

    for probe_idx in range(n_probes):
        rows_a = samples_a["samples"][probe_idx][:r]
        rows_b = samples_b["samples"][probe_idx][:r]
        if len(rows_a) < r or len(rows_b) < r:
            raise RuntimeError(f"fewer than r={r} samples at probe {probe_idx} for {a}/{b}")
        responses = [s["token_ids"] for s in rows_a] + [s["token_ids"] for s in rows_b]
        prompt_a = samples_a["prompt_token_ids"][probe_idx]
        prompt_b = samples_b["prompt_token_ids"][probe_idx]

        while True:  # OOM-halving retry (plan §8): start max_batch=16, halve on OOM
            try:
                lp_under_a = jsc.teacher_forced_response_logps(
                    model, prompt_a, responses, max_batch=max_batch_holder["max_batch"]
                )
                lp_under_b = jsc.teacher_forced_response_logps(
                    model, prompt_b, responses, max_batch=max_batch_holder["max_batch"]
                )
                break
            except torch.cuda.OutOfMemoryError:
                if max_batch_holder["max_batch"] <= 1:
                    raise
                max_batch_holder["max_batch"] = max(1, max_batch_holder["max_batch"] // 2)
                torch.cuda.empty_cache()
                logger.warning("OOM — halved max_batch to %d", max_batch_holder["max_batch"])

        pos0_rows: list[float] = []
        for i in range(2 * r):
            side = "a" if i < r else "b"
            meta = rows_a[i] if side == "a" else rows_b[i - r]
            lp_side = lp_under_a[i] if side == "a" else lp_under_b[i]
            lp_other = lp_under_b[i] if side == "a" else lp_under_a[i]
            pd = jsc.per_position_divergences(lp_side, lp_other)
            pdm = jsc.per_position_divergences(lp_side, lp_other, exclude_token_id=MARKER_ID)
            T = len(pd.js_bits)
            kl_m_mean = float(pd.kl_side_m_bits.mean())
            kl_other_mean = float(pd.kl_side_other_nats.mean())
            kl_m_masked_mean = float(pdm.kl_side_m_bits.mean())
            (a_kl_m if side == "a" else b_kl_m).append(kl_m_mean)
            (a_kl_ab if side == "a" else b_kl_ba).append(kl_other_mean)
            (a_kl_m_masked if side == "a" else b_kl_m_masked).append(kl_m_masked_mean)
            n_take = min(T, profile_cap)
            profile_sum[:n_take] += pd.js_bits[:n_take]
            profile_cnt[:n_take] += 1
            pos0_rows.append(float(pd.js_bits[0]))
            per_sample.append(
                {
                    "side": side,
                    "probe_idx": probe_idx,
                    "sample_idx": i if side == "a" else i - r,
                    "n_positions": T,
                    "truncated": bool(meta["truncated"]),
                    "kl_side_m_bits_per_token": kl_m_mean,
                    "kl_side_other_nats_per_token": kl_other_mean,
                    "kl_side_m_masked_bits_per_token": kl_m_masked_mean,
                    "js_sym_bits_per_token": float(pd.js_bits.mean()),
                }
            )
        pos0_per_probe.append(float(np.mean(pos0_rows)))
        del lp_under_a, lp_under_b

    rb = jsc.rb_pair_estimate(
        np.array(a_kl_m), np.array(b_kl_m), np.array(a_kl_ab), np.array(b_kl_ba)
    )
    rb_masked = jsc.rb_pair_estimate(
        np.array(a_kl_m_masked), np.array(b_kl_m_masked), np.array(a_kl_ab), np.array(b_kl_ba)
    )

    # ── Integration gates (plan §4 + §5 controls) ─────────────────────────
    pos0_mean = float(np.mean(pos0_per_probe))
    pid = _pair_id(a, b)
    if a == b and rb["js_rb_bits"] > SELFPAIR_MAX_BITS:
        raise RuntimeError(
            f"self-pair gate FAIL: JS({a},{a}) = {rb['js_rb_bits']:.6f} bits > "
            f"{SELFPAIR_MAX_BITS} — teacher-forcing alignment / padding bug"
        )
    pos0_check = None
    if pid in args.pos0_check_pairs and a != b:
        class_d = load_class_d_rewrites()
        instructed_panel = i532._instructed_bystander_panel()
        probes = samples_a["probes"][:n_probes]
        probs_a = i532._extract_next_token_probs_hf(
            model, tokenizer, a, probes, class_d, instructed_panel
        )
        probs_b = i532._extract_next_token_probs_hf(
            model, tokenizer, b, probes, class_d, instructed_panel
        )
        v1_val = i532._js_v1_predictor(probs_a, probs_b)
        diff = abs(pos0_mean - v1_val)
        pos0_check = {"v1_fresh": v1_val, "pos0_rb_mean": pos0_mean, "abs_diff": diff}
        if diff > POS0_FAIL:
            raise RuntimeError(
                f"position-0 cross-check FAIL for {pid}: |RB pos0 {pos0_mean:.5f} − v1 "
                f"{v1_val:.5f}| = {diff:.5f} > {POS0_FAIL} — alignment bug, fix before sweep"
            )
        if diff > POS0_WARN:
            logger.warning("position-0 cross-check WARN for %s: diff=%.5f", pid, diff)

    # Positions actually observed = count of nonzero-count slots (counts are
    # nonincreasing in position). Round-1 used profile_cnt.max() — the max
    # COUNT (rows at position 0), not the last position index — which
    # silently truncated the profile whenever n_probes*2r < profile_cap
    # (concern position-profile-truncation).
    n_kept = int((profile_cnt > 0).sum()) if profile_cnt.size else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "scoring",
        "metadata": _metadata(
            {
                "phase": "T",
                "model": args.model,
                "max_batch": max_batch_holder["max_batch"],
                # Provenance threading (round-2 review fix): record the
                # upstream sampling params so resume-skip / assembly can
                # validate a pair artifact without re-opening samples JSONs.
                "stub": bool(args.stub_samples),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "max_seq_len": args.max_seq_len,
            }
        ),
        "pair": {"a": a, "b": b},
        "is_selfpair": a == b,
        "n_probes": n_probes,
        "r_samples": r,
        # Probe-list identity (round-3 fix): both sides were strict-validated
        # against the CURRENT probe list above, so this hash names exactly
        # the probe texts the divergences were computed over.
        "probes_sha256": _probes_sha256(_current_probes(args)),
        "js_rb_bits": rb["js_rb_bits"],
        "kl_ab_nats": rb["kl_ab_nats"],
        "kl_ba_nats": rb["kl_ba_nats"],
        "sym_kl_nats": rb["sym_kl_nats"],
        "mc_se_js_bits": rb["mc_se_js_bits"],
        "masked": {  # ※ id 83399 masked + renormalized (plan §6 diagnostic)
            "js_rb_bits": rb_masked["js_rb_bits"],
            "mc_se_js_bits": rb_masked["mc_se_js_bits"],
            "masked_token_id": MARKER_ID,
        },
        "pos0_js_per_probe": pos0_per_probe,
        "pos0_js_mean_over_probes": pos0_mean,
        "pos0_v1_check": pos0_check,
        "per_sample": per_sample,
        "position_profile": {
            "js_bits_sum": profile_sum[: max(1, n_kept)].tolist(),
            "count": profile_cnt[: max(1, n_kept)].tolist(),
            "cap": profile_cap,
        },
        "truncation": {
            "n_truncated": int(sum(s["truncated"] for s in per_sample)),
            "n_rows": len(per_sample),
        },
    }


def phase_scoring(args, shard_k: int, shard_n: int) -> None:
    """Phase T — teacher-forced scoring per pair (checkpoint + resume-skip).

    Resume-skip accepts an existing pair JSON ONLY when it parses AND its
    recorded params (shape, model, stub, seed, max_new_tokens, pair identity,
    pos0-gate evidence) match the current invocation; anything else —
    including the 0-byte partial from a killed worker — is recomputed."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    predictors = _load_parent_predictors()
    sources, bystanders = predictors["sources"], predictors["bystanders"]
    pair_set = _resolve_pair_set(args, sources, bystanders)
    my_pairs = pair_set[shard_k::shard_n]
    per_pair_dir = args.out_dir / "per_pair"
    per_pair_dir.mkdir(parents=True, exist_ok=True)

    def _done(a: str, b: str) -> bool:
        p = per_pair_dir / f"pair_{_pair_id(a, b)}.json"
        return _load_compatible_pair(args, p, a, b, strict=False) is not None

    todo = [(a, b) for a, b in my_pairs if not _done(a, b)]
    logger.info(
        "[worker %d/%d] Phase T: %d pairs (%d already done)",
        shard_k,
        shard_n,
        len(todo),
        len(my_pairs) - len(todo),
    )
    if not todo:
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": device},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    max_batch_holder = {"max_batch": args.max_batch}

    for a, b in todo:
        t0 = time.time()
        payload = _score_pair(args, model, tokenizer, a, b, max_batch_holder)
        out_path = per_pair_dir / f"pair_{_pair_id(a, b)}.json"
        _write_json_atomic(out_path, payload)
        logger.info(
            "[worker %d] Phase T wrote %s (js_rb=%.5f bits, %.1fs)",
            shard_k,
            out_path.name,
            payload["js_rb_bits"],
            time.time() - t0,
        )


# ── Phase M: matrix assembly (CPU, in-parent) ──────────────────────────────


def phase_matrix(args) -> dict:
    """Assemble predictors_jsrb.json — js_rb/KL matrices (symmetric fill,
    diagonal 0) + the parent's v1 columns copied verbatim. Fail-loud on any
    missing OR parameter-incompatible per-pair file (round-2 review fix: a
    stale smoke/descope/stub artifact must never enter the headline matrix)."""
    predictors = _load_parent_predictors()
    sources, bystanders = predictors["sources"], predictors["bystanders"]
    b_idx = {b: i for i, b in enumerate(bystanders)}
    pairs = _canonical_pairs(sources, bystanders)
    per_pair_dir = args.out_dir / "per_pair"

    n_s, n_b = len(sources), len(bystanders)
    js_rb = np.zeros((n_s, n_b))
    js_rb_masked = np.zeros((n_s, n_b))
    kl_rc = np.zeros((n_s, n_b))  # KL(row-context ‖ col-context), nats
    kl_cr = np.zeros((n_s, n_b))  # KL(col-context ‖ row-context), nats
    sym_kl = np.zeros((n_s, n_b))
    mc_se = np.zeros((n_s, n_b))
    by_pair = {}
    problems: list[str] = []
    for a, b in pairs:
        try:
            by_pair[(a, b)] = _load_compatible_pair(
                args, per_pair_dir / f"pair_{_pair_id(a, b)}.json", a, b, strict=True
            )
        except RuntimeError as e:
            problems.append(str(e))
    if problems:
        raise RuntimeError(
            f"Phase M: {len(problems)}/{len(pairs)} per-pair artifacts missing or "
            f"parameter-incompatible with this invocation; first 5: {problems[:5]} — "
            "re-run Phase T with matching parameters (or use a fresh --out-dir)"
        )
    for i, s in enumerate(sources):
        for j, byst in enumerate(bystanders):
            if s == byst:
                continue  # diagonal ≡ 0 analytically (identical distributions)
            key = (s, byst) if b_idx[s] < b_idx[byst] else (byst, s)
            rec = by_pair[key]
            js_rb[i, j] = rec["js_rb_bits"]
            js_rb_masked[i, j] = rec["masked"]["js_rb_bits"]
            sym_kl[i, j] = rec["sym_kl_nats"]
            mc_se[i, j] = rec["mc_se_js_bits"]
            if key == (s, byst):  # row ctx is the pair's a
                kl_rc[i, j], kl_cr[i, j] = rec["kl_ab_nats"], rec["kl_ba_nats"]
            else:  # row ctx is the pair's b
                kl_rc[i, j], kl_cr[i, j] = rec["kl_ba_nats"], rec["kl_ab_nats"]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": "matrix",
        "metadata": _metadata(
            {
                "phase": "M",
                # Full compatibility tuple (round-3 review fix): the Phase A/F
                # loaders validate ALL of these, so a stale matrix computed
                # under different params can never feed the analysis/figures.
                "model": args.model,
                "stub": bool(args.stub_samples),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "max_seq_len": args.max_seq_len,
                "parent_columns_provenance": (
                    "cosine/js_v1/gauss_kl/base_prior copied VERBATIM from "
                    "eval_results/issue_532/predictors.json (parent SHA 296c4da2d)"
                ),
            }
        ),
        "sources": sources,
        "bystanders": bystanders,
        "n_probes": args.n_probes,
        "r_samples": args.r_samples,
        "probes_sha256": _probes_sha256(_current_probes(args)),
        "js_rb_matrix": js_rb.tolist(),
        "js_rb_masked_matrix": js_rb_masked.tolist(),
        "kl_row_col_matrix_nats": kl_rc.tolist(),
        "kl_col_row_matrix_nats": kl_cr.tolist(),
        "sym_kl_matrix_nats": sym_kl.tolist(),
        "mc_se_js_matrix": mc_se.tolist(),
        # Parent columns, verbatim:
        "cosine_matrix": predictors["cosine_matrix"],
        "js_v1_matrix": predictors["js_v1_matrix"],
        "gauss_kl_matrix": predictors["gauss_kl_matrix"],
        "base_prior": predictors["base_prior"],
        "base_prior_extra_logp": predictors.get("base_prior_extra_logp", {}),
    }
    out_path = args.out_dir / "predictors_jsrb.json"
    _write_json_atomic(out_path, payload, indent=2)
    logger.info("Phase M wrote %s", out_path)
    return payload


# ── Phase A: analysis (CPU, in-parent) ─────────────────────────────────────


def _compare_payloads(got, want, path: str = "", skip_keys: tuple = ("metadata",)) -> list[str]:
    """Recursive ≤1e-9 numeric compare for the reproduction control."""
    diffs: list[str] = []
    if isinstance(want, dict) and isinstance(got, dict):
        for k in want:
            if k in skip_keys:
                continue
            if k not in got:
                diffs.append(f"{path}.{k}: MISSING in reproduction")
                continue
            diffs.extend(_compare_payloads(got[k], want[k], f"{path}.{k}", skip_keys))
    elif isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            diffs.append(f"{path}: length {len(got)} != {len(want)}")
        else:
            for idx, (g, w) in enumerate(zip(got, want, strict=True)):
                diffs.extend(_compare_payloads(g, w, f"{path}[{idx}]", skip_keys))
    elif isinstance(want, bool) or isinstance(got, bool):
        if got != want:
            diffs.append(f"{path}: {got!r} != {want!r}")
    elif isinstance(want, int | float) and isinstance(got, int | float):
        both_nan = (
            isinstance(want, float)
            and isinstance(got, float)
            and (math.isnan(want) and math.isnan(got))
        )
        if not both_nan and abs(float(got) - float(want)) > REPRO_TOL:
            diffs.append(f"{path}: {got!r} != {want!r} (>|{REPRO_TOL}|)")
    elif got != want:
        diffs.append(f"{path}: {got!r} != {want!r}")
    return diffs


def _zcol(v: np.ndarray) -> np.ndarray:
    """The parent's combined-column standardization (exact same formula)."""
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)


def _attach_jsrb(panel: dict, jsrb_payload: dict) -> dict:
    """Attach js_rb (+ masked, + combined_js_rb, + length nuisance inputs) as
    panel columns, joined by (source_cid, bystander_label)."""
    sources = jsrb_payload["sources"]
    bystanders = jsrb_payload["bystanders"]
    s_idx = {s: i for i, s in enumerate(sources)}
    b_idx = {b: j for j, b in enumerate(bystanders)}
    js_rb_m = np.array(jsrb_payload["js_rb_matrix"])
    js_rb_masked_m = np.array(jsrb_payload["js_rb_masked_matrix"])
    n = panel["_n"]
    js_rb = np.zeros(n)
    js_rb_masked = np.zeros(n)
    for k in range(n):
        i = s_idx[str(panel["source_cid"][k])]
        j = b_idx[str(panel["bystander_label"][k])]
        js_rb[k] = js_rb_m[i, j]
        js_rb_masked[k] = js_rb_masked_m[i, j]
    panel = dict(panel)
    panel["js_rb"] = js_rb
    panel["js_rb_masked"] = js_rb_masked
    # combined_js_rb built exactly as the parent's combined_js_v1 (incl. its
    # divergence-polarity quirk) for apples-to-apples (plan §4 Phase A).
    panel["combined_js_rb"] = _zcol(panel["base_prior"]) + _zcol(js_rb)
    return panel


def _rho_strips(panel: dict, pk: str, n_boot: int = 1000, seed: int = 42) -> dict:
    """Signed Spearman ρ + bootstrap CI on union / ordinary / instructed."""
    out = {}
    masks = {
        "union": np.ones(panel["_n"], dtype=bool),
        "ordinary": panel["is_instructed"] == 0,
        "instructed": panel["is_instructed"] == 1,
    }
    for strip, m in masks.items():
        x, y = panel[pk][m], panel["trained_logp"][m]
        rho = i532._spearman_rho(x, y)
        bmean, lo, hi = i532._bootstrap_spearman_ci(x, y, n_boot=n_boot, seed=seed)
        out[strip] = {
            "rho": rho,
            "ci95_low": lo,
            "ci95_high": hi,
            "bootstrap_mean": bmean,
            "n": int(m.sum()),
        }
    return out


def _paired_bootstrap_delta(
    x_v1: np.ndarray,
    x_rb: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    clusters: np.ndarray | None = None,
) -> dict:
    """Direction-pinned paired bootstrap on Δ = ρ_v1 − ρ_RB (positive when RB
    is more negative than v1 — the signed expected-direction improvement,
    plan §1 H1). Cell-level resampling by default; with ``clusters`` given,
    resample whole clusters (unordered-pair units carry mirrored cells)."""
    rng = np.random.default_rng(seed)
    rho_v1 = i532._spearman_rho(x_v1, y)
    rho_rb = i532._spearman_rho(x_rb, y)
    point = rho_v1 - rho_rb
    deltas = []
    if clusters is None:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            deltas.append(
                i532._spearman_rho(x_v1[idx], y[idx]) - i532._spearman_rho(x_rb[idx], y[idx])
            )
    else:
        uniq = np.unique(clusters)
        members = {c: np.where(clusters == c)[0] for c in uniq}
        for _ in range(n_boot):
            picked = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([members[uniq[p]] for p in picked])
            deltas.append(
                i532._spearman_rho(x_v1[idx], y[idx]) - i532._spearman_rho(x_rb[idx], y[idx])
            )
    deltas = np.array(deltas)
    return {
        "rho_v1": rho_v1,
        "rho_rb": rho_rb,
        "delta_point": point,
        "delta_ci95_low": float(np.nanpercentile(deltas, 2.5)),
        "delta_ci95_high": float(np.nanpercentile(deltas, 97.5)),
        "n_boot": n_boot,
        "n_clusters": len(np.unique(clusters)) if clusters is not None else None,
    }


def _loco_jackknife_delta(panel: dict, mask: np.ndarray, contexts: list[str]) -> dict:
    """Leave-one-context-out jackknife on Δ = ρ_v1 − ρ_RB (drop every cell
    touching the held-out context, as source OR bystander)."""
    x_v1, x_rb, y = panel["js_v1"], panel["js_rb"], panel["trained_logp"]
    full = i532._spearman_rho(x_v1[mask], y[mask]) - i532._spearman_rho(x_rb[mask], y[mask])
    per_ctx = {}
    vals = []
    for c in contexts:
        keep = mask & (panel["source_cid"] != c) & (panel["bystander_label"] != c)
        d = i532._spearman_rho(x_v1[keep], y[keep]) - i532._spearman_rho(x_rb[keep], y[keep])
        per_ctx[c] = d
        vals.append(d)
    vals = np.array(vals)
    n = len(vals)
    se = math.sqrt((n - 1) / n * float(((vals - vals.mean()) ** 2).sum()))
    return {
        "delta_full": full,
        "jackknife_se": se,
        "ci95_low": full - 1.96 * se,
        "ci95_high": full + 1.96 * se,
        "per_context_delta": per_ctx,
    }


def _hierarchy_with_geometry(panel: dict, geom_key: str) -> dict:
    """Pinned six-regression hierarchy with the geometry column swapped —
    reuses the EXACT parent code path (column substitution only)."""
    p2 = dict(panel)
    p2["gauss_kl"] = panel[geom_key]
    out = i532._six_regression_hierarchy(p2)
    out["geometry_predictor_used"] = geom_key
    return out


def _unordered_cluster_ids(panel: dict, b_order: list[str]) -> np.ndarray:
    """Canonical unordered-pair id per cell (mirrored cells share an id;
    diagonal cells are their own unit) — ~136 units on the ordinary strip."""
    b_idx = {b: i for i, b in enumerate(b_order)}
    ids = []
    for k in range(panel["_n"]):
        s = str(panel["source_cid"][k])
        b = str(panel["bystander_label"][k])
        a, c = (s, b) if b_idx[s] <= b_idx[b] else (b, s)
        ids.append(f"{a}__{c}")
    return np.array(ids)


def _split_half_reliability(args, pairs: list[tuple[str, str]]) -> dict:
    """Split-half (first r/2 vs last r/2 samples per side) js_rb reliability
    across ordinary unordered pairs + Spearman-Brown (analyzer H1 obligation)."""
    per_pair_dir = args.out_dir / "per_pair"
    h1, h2 = [], []
    for a, b in pairs:
        rec = _load_compatible_pair(
            args, per_pair_dir / f"pair_{_pair_id(a, b)}.json", a, b, strict=True
        )
        r = rec["r_samples"]
        if r < 2:
            return {"skipped": f"r_samples={r} < 2"}
        half = {1: [], 2: []}
        for side in ("a", "b"):
            vals1 = [
                s["kl_side_m_bits_per_token"]
                for s in rec["per_sample"]
                if s["side"] == side and s["sample_idx"] < r // 2
            ]
            vals2 = [
                s["kl_side_m_bits_per_token"]
                for s in rec["per_sample"]
                if s["side"] == side and s["sample_idx"] >= r // 2
            ]
            half[1].append(np.mean(vals1))
            half[2].append(np.mean(vals2))
        h1.append(0.5 * half[1][0] + 0.5 * half[1][1])
        h2.append(0.5 * half[2][0] + 0.5 * half[2][1])
    rho = i532._spearman_rho(np.array(h1), np.array(h2))
    sb = 2 * rho / (1 + rho) if not math.isnan(rho) and rho > -1 else float("nan")
    return {"split_half_rho": rho, "spearman_brown": sb, "n_pairs": len(h1)}


def _length_nuisance_partial(args, panel: dict, b_order: list[str]) -> dict:
    """|Δ mean response length| nuisance per cell + rank-partialled ordinary ρ
    (analyzer H1 obligation: confirm the ρ survives partialling length)."""
    per_pair_dir = args.out_dir / "per_pair"
    b_idx = {b: i for i, b in enumerate(b_order)}
    cache: dict[str, float] = {}
    nuisance = np.zeros(panel["_n"])
    for k in range(panel["_n"]):
        s = str(panel["source_cid"][k])
        b = str(panel["bystander_label"][k])
        if s == b:
            nuisance[k] = 0.0
            continue
        a, c = (s, b) if b_idx[s] < b_idx[b] else (b, s)
        pid = _pair_id(a, c)
        if pid not in cache:
            rec = _load_compatible_pair(args, per_pair_dir / f"pair_{pid}.json", a, c, strict=True)
            mean_a = np.mean([x["n_positions"] for x in rec["per_sample"] if x["side"] == "a"])
            mean_b = np.mean([x["n_positions"] for x in rec["per_sample"] if x["side"] == "b"])
            cache[pid] = abs(float(mean_a) - float(mean_b))
        nuisance[k] = cache[pid]

    def _rank(v: np.ndarray) -> np.ndarray:
        from scipy.stats import rankdata

        return rankdata(v)

    m = panel["is_instructed"] == 0
    rx, ry, rz = _rank(panel["js_rb"][m]), _rank(panel["trained_logp"][m]), _rank(nuisance[m])
    Z = np.stack([np.ones_like(rz), rz], axis=1)
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex, ey = rx - Z @ bx, ry - Z @ by
    denom = np.linalg.norm(ex) * np.linalg.norm(ey)
    partial = float(ex @ ey / denom) if denom > 0 else float("nan")
    return {
        "rho_ordinary_partial_length": partial,
        "rho_ordinary_raw": i532._spearman_rho(panel["js_rb"][m], panel["trained_logp"][m]),
        "nuisance_column": "abs(mean response length side a − side b), tokens",
    }


def _h1_verdict(rb_ord: dict, paired_ord: dict) -> dict:
    """Direction-pinned H1 routing (plan §1). A sign-flipped ρ NEVER routes
    to Confirmed."""
    rho_rb = rb_ord["rho"]
    lo, hi = paired_ord["delta_ci95_low"], paired_ord["delta_ci95_high"]
    improved = lo > 0
    if math.isnan(rho_rb):
        verdict = "undetermined_nan"
    elif rho_rb > 0:
        verdict = "sign_flip_surprising_finding_parent_direction_reversed_flag_followup"
    elif rho_rb <= -0.50 and improved:
        verdict = "confirmed"
    elif -0.50 < rho_rb <= -0.41 and improved:
        verdict = "partial_improvement"
    elif hi < 0:
        verdict = "v1_overstated_js_arm_rb_significantly_worse_adopt_canonical_anyway"
    else:
        verdict = "falsified_first_token_already_captured_js_arm"
    return {
        "verdict": verdict,
        "rho_rb_ordinary": rho_rb,
        "rho_v1_ordinary": paired_ord["rho_v1"],
        "delta_point": paired_ord["delta_point"],
        "delta_ci95": [lo, hi],
        "thresholds": {"confirm_rho": -0.50, "v1_rho": -0.40970, "delta_ci_gt": 0.0},
    }


def _h2_verdict(rb_instr: dict) -> dict:
    rho = rb_instr["rho"]
    near = 0.15 <= abs(rho) <= 0.25
    return {
        "verdict": "confirmed" if abs(rho) < 0.20 else "falsified",
        "rho_rb_instructed": rho,
        "ci95": [rb_instr["ci95_low"], rb_instr["ci95_high"]],
        "near_threshold": near,
        "note": (
            "near-threshold: interpret via the bootstrap CI and the v1 comparison "
            "(−0.16998), not the point threshold (plan §6 analyzer notes)"
            if near
            else ""
        ),
        "threshold_abs": 0.20,
    }


def phase_analysis(args) -> dict:
    """Phase A — reproduction control (hard gate) + leaderboard re-fit +
    paired bootstrap + hierarchy variants + signed-residual/permutation."""
    phase0 = _load_parent_phase0()
    predictors = _load_parent_predictors()
    sources, bystanders = predictors["sources"], predictors["bystanders"]
    instructed_panel = i532._instructed_bystander_panel()

    # ── 1. Reproduction control (plan §7 hard gate) ───────────────────────
    repro_dir = args.out_dir / "repro_control"
    repro_dir.mkdir(parents=True, exist_ok=True)
    repro = i532.phase3_analysis(
        phase0,
        PARENT_DIR,
        predictors,
        PARENT_ARM,
        PARENT_EPOCHS,
        sources,
        bystanders,
        instructed_panel,
        repro_dir,
        stylized_drop=PARENT_STYLIZED_DROP,
    )
    committed = json.loads((PARENT_DIR / "analysis.json").read_text())
    diffs = _compare_payloads(repro, committed)
    if diffs:
        raise RuntimeError(
            "REPRODUCTION CONTROL FAILED — ported phase-3 on unchanged v1 inputs does not "
            f"reproduce eval_results/issue_532/analysis.json ({len(diffs)} diffs; first 10: "
            f"{diffs[:10]}). Never interpret RB numbers against a broken baseline (plan §7)."
        )
    logger.info("Reproduction control PASS — ported phase-3 matches analysis.json to ≤1e-9")

    # ── 2. Panel + js_rb columns (full-tuple validated load, round-3 fix) ─
    jsrb_payload = _load_matrix(args)
    panel = i532._build_union_panel(
        phase0,
        PARENT_DIR,
        PARENT_ARM,
        PARENT_EPOCHS,
        sources,
        bystanders,
        predictors,
        instructed_panel,
    )
    panel = _attach_jsrb(panel, jsrb_payload)

    # ── 3. Leaderboard re-fit (signed ρ, three strips, bootstrap CIs) ─────
    leaderboard = {
        pk: _rho_strips(panel, pk)
        for pk in (
            "cosine",
            "js_v1",
            "gauss_kl",
            "base_prior",
            "js_rb",
            "js_rb_masked",
            "combined_js_v1",
            "combined_js_rb",
        )
    }

    # ── 4. Paired bootstrap Δ = ρ_v1 − ρ_RB (H1 decision statistic) ───────
    masks = {
        "union": np.ones(panel["_n"], dtype=bool),
        "ordinary": panel["is_instructed"] == 0,
        "instructed": panel["is_instructed"] == 1,
    }
    paired = {
        strip: _paired_bootstrap_delta(
            panel["js_v1"][m], panel["js_rb"][m], panel["trained_logp"][m]
        )
        for strip, m in masks.items()
    }
    cluster_ids = _unordered_cluster_ids(panel, bystanders)
    m_ord = masks["ordinary"]
    paired_clustered = _paired_bootstrap_delta(
        panel["js_v1"][m_ord],
        panel["js_rb"][m_ord],
        panel["trained_logp"][m_ord],
        clusters=cluster_ids[m_ord],
    )
    loco = _loco_jackknife_delta(panel, m_ord, sources)

    # ── 5. Hierarchy variants (control must match parent exactly) ─────────
    hierarchy = {
        "gauss_kl_control": _hierarchy_with_geometry(panel, "gauss_kl"),
        "js_v1": _hierarchy_with_geometry(panel, "js_v1"),
        "js_rb": _hierarchy_with_geometry(panel, "js_rb"),
    }
    ctrl_diffs = _compare_payloads(
        {k: v for k, v in hierarchy["gauss_kl_control"].items() if k != "geometry_predictor_used"},
        {
            k: v
            for k, v in committed["six_regression_hierarchy"].items()
            if k != "geometry_predictor_used"
        },
    )
    if ctrl_diffs:
        raise RuntimeError(f"hierarchy gauss_kl control drifted from parent: {ctrl_diffs[:5]}")

    # ── 6. Signed residual + sign-flip + permutation for js_rb ────────────
    signed_resid = i532._h1_signed_residuals(panel, "js_rb", instructed_panel)
    sign_flip = i532._signflip_permutation_test(panel, "js_rb")

    # ── 7. Pos-0-vs-v1 drift audit map (free from persisted profiles) ─────
    js_v1_m = np.array(predictors["js_v1_matrix"])
    b_idx = {b: i for i, b in enumerate(bystanders)}
    s_idx = {s: i for i, s in enumerate(sources)}
    pairs = _canonical_pairs(sources, bystanders)
    pos0_map = {}
    mc_se_list: list[float] = []
    for a, b in pairs:
        rec = _load_compatible_pair(
            args, args.out_dir / "per_pair" / f"pair_{_pair_id(a, b)}.json", a, b, strict=True
        )
        mc_se_list.append(rec["mc_se_js_bits"])
        i = s_idx[a] if a in s_idx else s_idx[b]
        j = b_idx[b] if a in s_idx else b_idx[a]
        pos0_map[_pair_id(a, b)] = {
            "pos0_rb": rec["pos0_js_mean_over_probes"],
            "js_v1_matrix": float(js_v1_m[i, j]),
            "abs_diff": abs(rec["pos0_js_mean_over_probes"] - float(js_v1_m[i, j])),
        }
    pos0_diffs = np.array([v["abs_diff"] for v in pos0_map.values()])

    # ── 8/9/10. MC-noise, split-half, length-nuisance (analyzer notes) ────
    mc_ses = np.array(mc_se_list)  # collected in the validated pos0 loop above
    js_vals = np.array([panel["js_rb"][m_ord].std()])
    ord_pairs = [(a, b) for a, b in pairs if b not in instructed_panel]
    split_half = _split_half_reliability(args, ord_pairs)
    length_nuisance = _length_nuisance_partial(args, panel, bystanders)

    # ── 11. Verdicts ──────────────────────────────────────────────────────
    h1 = _h1_verdict(leaderboard["js_rb"]["ordinary"], paired["ordinary"])
    h1["clustered_unordered_pair_ci95"] = [
        paired_clustered["delta_ci95_low"],
        paired_clustered["delta_ci95_high"],
    ]
    h1["loco_jackknife_ci95"] = [loco["ci95_low"], loco["ci95_high"]]
    h1["suggestive_only_if_clustered_spans_0"] = paired["ordinary"]["delta_ci95_low"] > 0 and not (
        paired_clustered["delta_ci95_low"] > 0
    )
    h2 = _h2_verdict(leaderboard["js_rb"]["instructed"])

    # Self-pair sanity arm is a REQUIRED control (plan §11): missing or
    # stale → raise; above-threshold → raise. Round 1 recorded None when the
    # file was missing, silently skipping the gate (Codex minor, upheld).
    selfpair_path = args.out_dir / "per_pair" / f"pair_{args.selfpair}__{args.selfpair}.json"
    selfpair_rec = _load_compatible_pair(
        args, selfpair_path, args.selfpair, args.selfpair, strict=True
    )
    selfpair_js = selfpair_rec["js_rb_bits"]
    if selfpair_js > SELFPAIR_MAX_BITS:
        raise RuntimeError(
            f"Phase A self-pair gate FAIL: JS({args.selfpair},{args.selfpair}) = "
            f"{selfpair_js:.6f} bits > {SELFPAIR_MAX_BITS} — teacher-forcing alignment / "
            "padding bug; never interpret the RB matrix past a failed structural control"
        )

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "phase": "analysis",
        "metadata": _metadata(
            {
                "phase": "A",
                "n_rows": panel["_n"],
                # Full compatibility tuple (round-3 review fix): standalone
                # Phase F validates ALL of these, so figures can never mix a
                # current matrix with a stale leaderboard/hierarchy.
                "model": args.model,
                "stub": bool(args.stub_samples),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "max_seq_len": args.max_seq_len,
            }
        ),
        "n_probes": args.n_probes,
        "r_samples": args.r_samples,
        "probes_sha256": _probes_sha256(_current_probes(args)),
        "reproduction_control": {
            "pass": True,
            "tolerance": REPRO_TOL,
            "compared_to": str(PARENT_DIR / "analysis.json"),
        },
        "leaderboard": leaderboard,
        "paired_bootstrap_delta_v1_minus_rb": paired,
        "paired_bootstrap_clustered_unordered_pair": paired_clustered,
        "paired_bootstrap_loco_jackknife": loco,
        "six_regression_hierarchy_variants": hierarchy,
        "signed_residual_js_rb": signed_resid,
        "sign_flip_permutation_js_rb": sign_flip,
        "pos0_vs_v1_drift_audit": {
            "max_abs_diff": float(pos0_diffs.max()),
            "mean_abs_diff": float(pos0_diffs.mean()),
            "per_pair": pos0_map,
        },
        "mc_noise": {
            "median_per_pair_mc_se_bits": float(np.median(mc_ses)),
            "max_per_pair_mc_se_bits": float(mc_ses.max()),
            "cross_pair_sd_ordinary_bits": float(js_vals[0]),
        },
        "split_half_reliability_ordinary": split_half,
        "length_nuisance": length_nuisance,
        "selfpair_js_bits": selfpair_js,
        "h1": h1,
        "h2": h2,
    }
    out_path = args.out_dir / "analysis_jsrb.json"
    _write_json_atomic(out_path, analysis, indent=2)
    logger.info("Phase A wrote %s (H1=%s, H2=%s)", out_path, h1["verdict"], h2["verdict"])
    return analysis


# ── Phase F: figures (CPU, in-parent, paper_plots styling) ─────────────────

PREDICTOR_LABELS = {
    "js_v1": "First-token JS (deprecated v1)",
    "js_rb": "Canonical sequence JS (RB)",
    "cosine": "Activation cosine",
    "gauss_kl": "Activation Gaussian KL",
    "base_prior": "Base emission prior",
}


def phase_figures(args) -> list[Path]:
    """Phase F — hero leaderboard + exploratory dump (plan §6)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    # Full-tuple validated loads (round-3 fix): standalone `--phases F` must
    # refuse a stale analysis/matrix, never mix provenance across artifacts.
    analysis = _load_analysis(args)
    jsrb_payload = _load_matrix(args)
    predictors = _load_parent_predictors()
    phase0 = _load_parent_phase0()
    sources, bystanders = predictors["sources"], predictors["bystanders"]
    instructed_panel = i532._instructed_bystander_panel()
    panel = i532._build_union_panel(
        phase0,
        PARENT_DIR,
        PARENT_ARM,
        PARENT_EPOCHS,
        sources,
        bystanders,
        predictors,
        instructed_panel,
    )
    panel = _attach_jsrb(panel, jsrb_payload)
    figures_dir = args.figures_dir
    written: list[Path] = []

    def _save(fig, stem: str) -> None:
        paths = savefig_paper(fig, stem, dir=figures_dir)
        written.extend(paths.values())
        plt.close(fig)

    lb = analysis["leaderboard"]
    pks = ["js_v1", "js_rb", "cosine", "gauss_kl", "base_prior"]
    strips = ["union", "ordinary", "instructed"]
    colors = paper_palette(len(pks))

    # 1. Hero: grouped leaderboard bars, RB adjacent to v1.
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    width = 0.15
    xs = np.arange(len(strips))
    for k, pk in enumerate(pks):
        vals = [lb[pk][s]["rho"] for s in strips]
        los = [lb[pk][s]["rho"] - lb[pk][s]["ci95_low"] for s in strips]
        his = [lb[pk][s]["ci95_high"] - lb[pk][s]["rho"] for s in strips]
        ax.bar(
            xs + (k - 2) * width,
            vals,
            width,
            yerr=[los, his],
            capsize=2,
            label=PREDICTOR_LABELS[pk],
            color=colors[k],
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["All 416 cells", "Ordinary contexts (n=256)", "Instructed contexts (n=160)"]
    )
    ax.set_ylabel("Spearman ρ vs marker emission (signed)")
    ax.set_title("Predictor leaderboard: canonical sequence JS vs the first-token shortcut")
    ax.legend(fontsize=8)
    _save(fig, "hero_leaderboard_rb_vs_v1")

    # 2. Estimator-agreement map: js_rb vs js_v1 per cell.
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    is_instr = panel["is_instructed"] == 1
    ax.scatter(
        panel["js_v1"][~is_instr],
        panel["js_rb"][~is_instr],
        s=14,
        alpha=0.65,
        label="Ordinary bystander",
        color=colors[0],
    )
    ax.scatter(
        panel["js_v1"][is_instr],
        panel["js_rb"][is_instr],
        s=14,
        alpha=0.65,
        label="Instructed bystander",
        color=colors[3],
    )
    ax.set_xlabel("First-token JS (v1, bits)")
    ax.set_ylabel("Canonical sequence JS (RB, bits per token)")
    ax.set_title("Estimator agreement across the 416-cell panel")
    ax.legend()
    _save(fig, "scatter_jsrb_vs_jsv1")

    # 3. js_rb vs emission DV (parent heroA analog).
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(
        panel["js_rb"][~is_instr],
        panel["trained_logp"][~is_instr],
        s=14,
        alpha=0.65,
        label="Ordinary bystander",
        color=colors[0],
    )
    ax.scatter(
        panel["js_rb"][is_instr],
        panel["trained_logp"][is_instr],
        s=14,
        alpha=0.65,
        label="Instructed bystander",
        color=colors[3],
    )
    ax.set_xlabel("Canonical sequence JS (RB, bits per token)")
    ax.set_ylabel("Marker emission rate (trained model, on-policy)")
    ax.set_title("Canonical JS vs leakage, by context type")
    ax.legend()
    _save(fig, "scatter_jsrb_vs_emission")

    # 4. Per-position JS profile, ordinary vs instructed pairs.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    per_pair_dir = args.out_dir / "per_pair"
    prof = {"ordinary": None, "instructed": None}
    for a, b in _canonical_pairs(sources, bystanders):
        rec = _load_compatible_pair(
            args, per_pair_dir / f"pair_{_pair_id(a, b)}.json", a, b, strict=True
        )
        kind = "instructed" if b in instructed_panel else "ordinary"
        s = np.array(rec["position_profile"]["js_bits_sum"])
        c = np.array(rec["position_profile"]["count"], dtype=np.float64)
        if prof[kind] is None:
            prof[kind] = [s.copy(), c.copy()]
        else:
            L = max(len(prof[kind][0]), len(s))
            for arrs, new in ((prof[kind], (s, c)),):
                arrs[0] = np.pad(arrs[0], (0, L - len(arrs[0])))
                arrs[1] = np.pad(arrs[1], (0, L - len(arrs[1])))
                arrs[0][: len(new[0])] += new[0]
                arrs[1][: len(new[1])] += new[1]
    for kind, color in (("ordinary", colors[0]), ("instructed", colors[3])):
        if prof[kind] is None:
            continue
        s, c = prof[kind]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        ax.plot(mean, label=f"{kind.capitalize()} pairs", color=color)
    ax.set_xlabel("Response token position (count denominator: length ≥ position)")
    ax.set_ylabel("Mean per-position JS (bits)")
    ax.set_title("Where in the response the two contexts disagree")
    ax.legend()
    _save(fig, "position_profile_ordinary_vs_instructed")

    # 5. KL-asymmetry diagnostic.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    asym = {"Ordinary–ordinary": [], "Ordinary–instructed": []}
    for a, b in _canonical_pairs(sources, bystanders):
        rec = _load_compatible_pair(
            args, per_pair_dir / f"pair_{_pair_id(a, b)}.json", a, b, strict=True
        )
        kind = "Ordinary–instructed" if b in instructed_panel else "Ordinary–ordinary"
        asym[kind].append(rec["kl_ab_nats"] - rec["kl_ba_nats"])
    ax.boxplot(
        [asym["Ordinary–ordinary"], asym["Ordinary–instructed"]],
        tick_labels=["Ordinary–ordinary", "Ordinary–instructed"],
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("KL(a‖b) − KL(b‖a) (nats per token)")
    ax.set_title("Directional asymmetry of the divergence, by pair type")
    _save(fig, "kl_asymmetry_by_pair_type")

    # 6. Signed residuals for js_rb (parent residual-figure analog).
    sr = analysis["signed_residual_js_rb"]
    if "per_bystander_median_residual" in sr:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        labels = list(sr["per_bystander_median_residual"].keys())
        vals = [sr["per_bystander_median_residual"][k] for k in labels]
        ax.bar(range(len(labels)), vals, color=colors[1])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            [lb_.replace("instr_", "").replace("_", " ") for lb_ in labels], rotation=45, ha="right"
        )
        ax.set_ylabel("Median signed residual (emission rate)")
        ax.set_title("Instructed contexts vs the ordinary-fit prediction (canonical JS)")
        _save(fig, "signed_residual_jsrb_instructed")

    # 7. Hierarchy ladder for the three geometry variants.
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    hv = analysis["six_regression_hierarchy_variants"]
    bar_keys = [
        ("r2_1_indicator_only", "Context-type flag"),
        ("r2_2_prior_only", "Base prior"),
        ("r2_3_geometry_only", "Geometry only"),
        ("r2_4_indicator_plus_prior", "Flag + prior"),
        ("r2_5_indicator_plus_geometry", "Flag + geometry"),
        ("r2_6_full_additive", "Flag + prior + geometry"),
    ]
    xs = np.arange(len(bar_keys))
    for k, (variant, label) in enumerate(
        (
            ("gauss_kl_control", "Gaussian KL (parent control)"),
            ("js_v1", "First-token JS (v1)"),
            ("js_rb", "Canonical sequence JS (RB)"),
        )
    ):
        vals = [hv[variant][bk] for bk, _ in bar_keys]
        ax.bar(xs + (k - 1) * 0.26, vals, 0.26, label=label, color=colors[k])
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in bar_keys], rotation=30, ha="right")
    ax.set_ylabel("Held-out R² (leave-one-class-out CV)")
    ax.set_title("Six-regression hierarchy with each geometry column")
    ax.legend(fontsize=8)
    _save(fig, "hierarchy_ladder_geometry_variants")

    # 8. Raw alongside processed: per-sample distributions for 6 pairs.
    ord_pairs = [
        (a, b) for a, b in _canonical_pairs(sources, bystanders) if b not in instructed_panel
    ]
    instr_pairs = [
        (a, b) for a, b in _canonical_pairs(sources, bystanders) if b in instructed_panel
    ]
    cherry = ord_pairs[:3] + instr_pairs[:3]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    data, labels = [], []
    for a, b in cherry:
        rec = _load_compatible_pair(
            args, per_pair_dir / f"pair_{_pair_id(a, b)}.json", a, b, strict=True
        )
        data.append([s["kl_side_m_bits_per_token"] for s in rec["per_sample"]])
        labels.append(f"{a} vs {b.replace('instr_', '')}")
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Per-sample JS half-term (bits per token)")
    ax.set_title("Raw per-sample spread behind six per-pair estimates")
    _save(fig, "per_sample_violin_six_pairs")

    # 9. Truncation rate per context.
    samples_dir = args.out_dir / "samples"
    ctx_labels, rates = [], []
    for ctx in bystanders:
        p = samples_dir / f"samples_{ctx}.json"
        payload = _load_compatible_samples(args, p, ctx, strict=False)
        if payload is not None:  # absent/incompatible files are skipped, not plotted
            ctx_labels.append(ctx.replace("instr_", ""))
            rates.append(payload["truncation_rate"])
    if ctx_labels:
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        ax.bar(range(len(ctx_labels)), rates, color=colors[2])
        ax.set_xticks(range(len(ctx_labels)))
        ax.set_xticklabels(ctx_labels, rotation=45, ha="right")
        ax.set_ylabel("Fraction truncated at 256 new tokens")
        ax.set_title("Sampling truncation rate per context")
        _save(fig, "truncation_rate_per_context")

    logger.info("Phase F wrote %d figure files to %s", len(written), figures_dir)
    return written


# ── Sentinel (poll_pipeline.py contract) ───────────────────────────────────


def _write_results_sentinel(args, analysis: dict, t_start: float) -> Path:
    """End-of-run sentinel per /issue Step 7 + poll_pipeline.py
    ``_SENTINEL_REQUIRED_KEYS`` (sentinel_schema_version / kind / version)."""
    epoch = int(time.time())
    # poll_pipeline.py watches /workspace/logs/issue-540-*.json ONLY. On a
    # pod the /workspace volume exists but bootstrap may not have created
    # logs/ — create it rather than silently diverting (round-1 concern
    # sentinel-workspace-logs-fallback: a diverted sentinel blinds the
    # autonomous poller, the #488 failure shape). Only a GPU-less local VM
    # (no /workspace volume at all) falls back to out_dir, loudly.
    workspace = Path("/workspace")
    if workspace.is_dir():
        sentinel_dir = workspace / "logs"
    else:
        sentinel_dir = args.out_dir
        logger.warning(
            "no /workspace volume (local VM run) — results sentinel diverted to %s; "
            "poll_pipeline.py watches /workspace/logs only and will NOT observe it",
            sentinel_dir,
        )
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = sentinel_dir / f"issue-540-epm_results-{epoch}.json"
    wall_h = (time.time() - t_start) / 3600.0
    lb = analysis["leaderboard"]
    note = {
        "eval_numbers": {
            "rho_rb": {s: lb["js_rb"][s]["rho"] for s in ("union", "ordinary", "instructed")},
            "rho_v1": {s: lb["js_v1"][s]["rho"] for s in ("union", "ordinary", "instructed")},
            "delta_ordinary_ci95": analysis["h1"]["delta_ci95"],
            "h1_verdict": analysis["h1"]["verdict"],
            "h2_verdict": analysis["h2"]["verdict"],
            "reproduction_control_pass": analysis["reproduction_control"]["pass"],
            "selfpair_js_bits": analysis["selfpair_js_bits"],
            "pos0_max_abs_diff": analysis["pos0_vs_v1_drift_audit"]["max_abs_diff"],
        },
        "eval_paths": {
            "predictors": str(args.out_dir / "predictors_jsrb.json"),
            "analysis": str(args.out_dir / "analysis_jsrb.json"),
            "per_pair_glob": str(args.out_dir / "per_pair" / "pair_*.json"),
            "samples_glob": str(args.out_dir / "samples" / "samples_*.json"),
            "figures_dir": str(args.figures_dir),
        },
        "reproducibility_card": _metadata(
            {
                "n_probes": args.n_probes,
                "r_samples": args.r_samples,
                "max_new_tokens": args.max_new_tokens,
                "max_seq_len": args.max_seq_len,
                "seed": args.seed,
                "workers": args.workers,
                "probes_sha256": _probes_sha256(_current_probes(args)),
            }
        ),
        "wandb_url": "",  # eval-only, no training run
        "hf_hub_url": f"{HF_DATA_REPO}/{HF_SAMPLES_PATH}",
        "worktree_path": str(PROJECT_ROOT),
        "final_commit_sha": _git_commit(),
        "gpu_hours_used": round(wall_h * max(1, args.workers), 2),
        "gpu_hours_budgeted": GPU_HOURS_BUDGETED,
        "plan_deviations": args.plan_deviation or [],
    }
    _write_json_atomic(
        path,
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 540,
            "status": "completed",
            "ts": epoch,
            "note": note,
        },
        indent=2,
    )
    logger.info("Wrote results sentinel %s", path)
    return path


# ── Upload (optional; HF data repo, fail-loud) ─────────────────────────────


def upload_samples(args) -> None:
    """Upload samples/ (raw completions) to the HF data repo (upload policy:
    raw completions MUST land on HF before pod termination)."""
    from huggingface_hub import HfApi

    samples_dir = args.out_dir / "samples"
    if not samples_dir.exists() or not list(samples_dir.glob("samples_*.json")):
        raise RuntimeError(f"--upload-samples: nothing to upload at {samples_dir}")
    api = HfApi()
    api.upload_folder(
        folder_path=str(samples_dir),
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=HF_SAMPLES_PATH,
    )
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(HF_DATA_REPO, repo_type="dataset") if HF_SAMPLES_PATH in f]
    if not files:
        raise RuntimeError("upload_folder returned but Hub listing shows 0 files — verify")
    logger.info("Uploaded %d sample files to %s/%s", len(files), HF_DATA_REPO, HF_SAMPLES_PATH)


# ── Dispatcher / worker ────────────────────────────────────────────────────


def _is_production_shaped(args) -> bool:
    """True iff the invocation matches the plan §10 production card exactly
    (the only shape that may default into eval_results/issue_540)."""
    return (
        args.n_probes == 50
        and args.r_samples == 8
        and args.seed == 42
        and args.max_new_tokens == 256
        and args.max_seq_len == DEFAULT_MAX_SEQ_LEN
        and args.model == BASE_MODEL
        and not args.stub_samples
        and not args.pairs
    )


def _resolve_dirs(args) -> None:
    """Default-dir routing (round-2 review fix). Omitted ``--out-dir``:
    production-shaped runs → eval_results/issue_540; ANY other shape (smoke /
    descope / stub / subset / non-default seed) → eval_results/issue_540_smoke,
    so a smoke launched without --out-dir (the plan §10 smoke command) can
    never seed resume-skip artifacts into the production dir. Explicit
    ``--out-dir`` is always respected — artifact-compatibility validation
    still refuses mismatched resumes there."""
    production = _is_production_shaped(args)
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_DIR if production else SMOKE_OUT_DIR
        if not production:
            logger.warning(
                "non-production parameters (n_probes=%d r_samples=%d seed=%d "
                "max_new_tokens=%d max_seq_len=%d stub=%s pairs=%s model=%s) with no "
                "--out-dir — routing output to %s (production dir untouched)",
                args.n_probes,
                args.r_samples,
                args.seed,
                args.max_new_tokens,
                args.max_seq_len,
                args.stub_samples,
                bool(args.pairs),
                args.model,
                args.out_dir,
            )
    if args.figures_dir is None:
        # Round-3 review fix: figures routing follows the RESOLVED out-dir,
        # not the parameter shape — a production-shaped run with an explicit
        # CUSTOM --out-dir must not write into the canonical figures dir
        # (figures would mix with a different out-dir's artifacts). Figures
        # for a custom out-dir co-locate at <out_dir>/figures.
        resolved = args.out_dir.resolve()
        if resolved == DEFAULT_OUT_DIR.resolve():
            args.figures_dir = DEFAULT_FIGURES_DIR
        elif resolved == SMOKE_OUT_DIR.resolve():
            args.figures_dir = SMOKE_FIGURES_DIR
        else:
            args.figures_dir = args.out_dir / "figures"


def _parse_shard(s: str) -> tuple[int, int]:
    k, n = s.split("/")
    k, n = int(k), int(n)
    if not (0 <= k < n):
        raise ValueError(f"--pair-shard {s!r}: need 0 <= k < N")
    return k, n


def _cvd_for_worker(k: int) -> str:
    parent = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent:
        ids = parent.split(",")
        return ids[k % len(ids)]
    return str(k)


def _fork_workers(args, phase: str, shards: list[tuple[int, int]]) -> None:
    """Fork one worker subprocess per shard for a GPU phase (S or T) and wait.

    SAME path for smoke (one shard) and the 4-way sweep — explicit env
    injection (env={**os.environ, CUDA_VISIBLE_DEVICES=<k>}); fail loud on
    any non-zero return code.
    """
    procs = []
    for k, n in shards:
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--phases",
            phase,
            "--pair-shard",
            f"{k}/{n}",
            "--out-dir",
            str(args.out_dir),
            "--n-probes",
            str(args.n_probes),
            "--r-samples",
            str(args.r_samples),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-seq-len",
            str(args.max_seq_len),
            "--seed",
            str(args.seed),
            "--model",
            args.model,
            "--max-batch",
            str(args.max_batch),
            "--selfpair",
            args.selfpair,
            "--pos0-check-pairs",
            *args.pos0_check_pairs,
        ]
        if args.pairs:
            argv += ["--pairs", *args.pairs]
        if args.stub_samples:
            argv.append("--stub-samples")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": _cvd_for_worker(k)}
        logger.info(
            "forking worker shard %d/%d for phase %s (CVD=%s)",
            k,
            n,
            phase,
            env["CUDA_VISIBLE_DEVICES"],
        )
        procs.append(((k, n), subprocess.Popen(argv, env=env)))
    failures = []
    for (k, n), p in procs:
        rc = p.wait()
        if rc != 0:
            failures.append((k, n, rc))
    if failures:
        raise RuntimeError(f"phase {phase} worker failures (shard, rc): {failures}")


def run_worker(args) -> int:
    """Forked worker: execute ONE GPU phase for one shard, in-process."""
    if args.pair_shard is None:
        raise ValueError("--worker requires --pair-shard k/N")
    k, n = _parse_shard(args.pair_shard)
    phases = args.phases.replace(",", "")
    if phases == "S":
        phase_sampling(args, k, n)
    elif phases == "T":
        phase_scoring(args, k, n)
    else:
        raise ValueError(f"--worker handles a single GPU phase (S or T), got {args.phases!r}")
    return 0


def run_dispatcher(args) -> int:
    """Single dispatcher path for smoke AND full sweep (PASS_UNIFIED)."""
    t_start = time.time()
    print("[phase=startup]", flush=True)
    phases = [p for p in PHASE_ORDER if p in set(args.phases.replace(",", "").upper())]
    if not phases:
        raise ValueError(f"--phases {args.phases!r} contains no valid phase letters (S,T,M,A,F)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.pair_shard is not None:
        k, n = _parse_shard(args.pair_shard)
        shards = [(k, n)]
    else:
        shards = [(k, args.workers) for k in range(args.workers)]

    analysis = None
    for phase in phases:
        if phase == "S":
            print("[phase=sampling]", flush=True)
            _fork_workers(args, "S", shards)
        elif phase == "T":
            print("[phase=scoring]", flush=True)
            _fork_workers(args, "T", shards)
        elif phase == "M":
            print("[phase=matrix]", flush=True)
            phase_matrix(args)
        elif phase == "A":
            print("[phase=analysis]", flush=True)
            analysis = phase_analysis(args)
        elif phase == "F":
            print("[phase=figures]", flush=True)
            phase_figures(args)

    if args.upload_samples:
        print("[phase=upload]", flush=True)
        upload_samples(args)

    if analysis is not None:
        _write_results_sentinel(args, analysis, t_start)
    print("[phase=done]", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[3],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phases", default="S,T,M,A,F", help="comma-joined subset of S,T,M,A,F")
    parser.add_argument("--pair-shard", default=None, help="k/N — pin a single shard grid")
    parser.add_argument("--workers", type=int, default=4, help="worker count when no --pair-shard")
    parser.add_argument("--n-probes", type=int, default=50)
    parser.add_argument("--r-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="vLLM max_model_len; recorded + validated in every artifact "
        "(a non-default value silently shortens generations otherwise)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pairs", nargs="+", default=None, help="explicit pair subset, e.g. A1__instr_explicit_1"
    )
    parser.add_argument("--selfpair", default="A1", help="self-pair smoke-arm context")
    parser.add_argument(
        "--pos0-check-pairs",
        nargs="+",
        default=["A1__instr_explicit_1"],
        help="pairs that run the fresh-v1 position-0 cross-check (plan §4 gate)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="omitted → eval_results/issue_540 for production-shaped runs, "
        "eval_results/issue_540_smoke for any other shape (see _resolve_dirs)",
    )
    parser.add_argument("--figures-dir", type=Path, default=None, help="same routing as --out-dir")
    parser.add_argument("--model", default=BASE_MODEL, help="HF model for Phase T scoring")
    parser.add_argument("--max-batch", type=int, default=16, help="Phase T forward sub-batch")
    parser.add_argument(
        "--stub-samples",
        action="store_true",
        help="Phase S writes synthetic samples; downstream phases require the same flag "
        "to ACCEPT stub artifacts (GPU-less VM smoke only; recorded in metadata)",
    )
    parser.add_argument(
        "--upload-samples",
        action="store_true",
        help="after phases: upload samples/ to the HF data repo (fail-loud, Hub-verified)",
    )
    parser.add_argument(
        "--plan-deviation", action="append", default=[], help="recorded in the results sentinel"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)
    _resolve_dirs(args)
    if args.worker:
        return run_worker(args)
    return run_dispatcher(args)


if __name__ == "__main__":
    sys.exit(main())
