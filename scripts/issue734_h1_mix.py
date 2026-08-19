"""Issue #734 -- self-contained H1 training-mix builder (the `setup_h1_mix` phase).

The H1 Phase-2 fresh train (`train_h1_cell`) HARD-REQUIRES one local marker
training mix -- `data/issue_664/train/marker/mk_librarian_contra_d1_seed42.jsonl`
(or its `train_smoke` twin) -- which #664 produced via its `--phase p0` pipeline
but NEVER uploaded to HF (only #664's adapters / raw-completions / store tensors
landed). The git-clone-only GCP lane cannot stage it, so phase2 crashes at the
`assert data_path.exists()` pre-train gate (issue #734 crash-fix round 1).

This module rebuilds EXACTLY that one mix, self-contained inside the #734 driver,
by REUSING #664's own verified building blocks (no recipe drift):

1. write the deterministic 300-question marker training pool (UltraChat first
   user-turns, disjoint from the eval probes) via ``issue664_dispatch._marker_
   question_pool`` + ``_write_pool`` -- the SAME pool #664's p0 wrote;
2. generate the 5 needed ``marker_R`` base-greedy on-policy caches (the librarian
   SOURCE context + the 4 contrastive-panel NEGATIVE contexts) via #664's
   ``_elicit_marker_R`` on one vLLM engine -- base greedy (temp=0) is the marker
   carve-out's on-policy R and is deterministic given the same base model +
   tokenizer + question pool;
3. drive the standalone CPU mix-builder ``issue664_build_training_data.py``
   for the single ``marker / librarian / contra / d1 / seed42`` cell as a
   subprocess (so the builder's own ``load_dotenv`` + zero-truncation asserts +
   panel-disjointness assert run in a clean process), which writes the mix at the
   path ``train_h1_cell`` reads.

Why marker-only librarian-contra-d1, not the full ``--phase p0``: ``p0`` builds
EVERY behavior's caches (sycophancy / refusal / em + Claude-judge calls) and its
``--cells 1`` non-smoke selection picks ``realized_grid()[0]`` (NOT guaranteed to
be the librarian-contra-d1 cell). H1 reuses ONLY the librarian-contra-d1-seed42
mix (`to_664_cell` pins seed=PHASE1_SEED for every H1 seed -- the round-3 fix),
so this rebuilds exactly that one cell with #664's verified marker path and
nothing else.

Idempotent: if the target mix already exists AND its sha256 matches the recorded
provenance sidecar, the whole phase is a no-op (matters for resume / re-run /
smoke). The mix's sha256 is pinned in ``setup_h1_mix_provenance.json`` for
reproducibility (CLAUDE.md reproducibility-metadata rule).

Determinism note: the mix is content-deterministic up to the base model's greedy
decoding -- same base (Qwen-2.5-7B-Instruct, #664's marker base), same tokenizer,
same question pool, temp=0 -> the same ``marker_R`` -> the same rows. It is a
FAITHFUL regeneration of #664's seed-42 marker mix, not a new variable: H1 is a
FRESH train (it does not reuse #664's adapter weights), so byte-identity with
#664's original on-disk mix is not required -- recipe + question-distribution
identity is, and both are inherited verbatim from #664's own code.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# subprocess-env contract (experiment-implementer.md): this module spawns the
# issue664_build_training_data.py builder subprocess with env={**os.environ}; the
# explicit load-at-entry guarantees HF_TOKEN is in os.environ even on a fresh
# `uv run python` import path (uv run does NOT auto-load .env). Idempotent with the
# dispatcher's own module-top load.
load_dotenv()

import issue664_common as C664  # noqa: E402
import issue734_common as C  # noqa: E402

logger = logging.getLogger("issue734_h1_mix")

# The single H1 mix cell (the round-3 seed-42-pinned reuse; matches
# train_h1_cell -> H1Cell.to_664_cell()). Built once, reused for every H1 seed.
H1_MIX_BEHAVIOR = "marker"
H1_MIX_SOURCE, H1_MIX_ARM, H1_MIX_DOSE = C.PARITY_PROBE_CELL  # ("librarian","contra","d1")
H1_MIX_SEED = C.PHASE1_SEED  # 42


def _h1_mix_cell() -> C664.Cell:
    """The #664 Cell whose mix the H1 fresh-train consumes (seed-42 pinned)."""
    return C664.Cell(
        behavior=H1_MIX_BEHAVIOR,
        source=H1_MIX_SOURCE,
        arm=H1_MIX_ARM,
        dose=H1_MIX_DOSE,
        seed=H1_MIX_SEED,
    )


def mix_path(*, smoke: bool) -> Path:
    """The exact path ``train_h1_cell`` asserts exists (single source of truth)."""
    cell = _h1_mix_cell()
    return (
        C664.DATA_ROOT
        / ("train_smoke" if smoke else "train")
        / H1_MIX_BEHAVIOR
        / f"{cell.eval_key}.jsonl"
    )


def _provenance_path(*, smoke: bool) -> Path:
    """Sidecar recording the built mix's sha256 (idempotency + reproducibility)."""
    return mix_path(smoke=smoke).with_suffix(".setup_h1_mix_provenance.json")


def _needed_marker_R_contexts() -> list[str]:
    """The ctx_keys whose ``marker_R`` caches the librarian-contra-d1 build reads:
    the librarian SOURCE context + the 4 contrastive-panel NEGATIVE slugs. The
    builder reads ``marker_R/<source>.json`` for positives and
    ``marker_R/<neg.slug>.json`` for negatives (issue664_build_training_data
    ``build_marker``)."""
    return [H1_MIX_SOURCE, *[neg.slug for neg in C664.negative_panel()]]


def _mix_is_current(*, smoke: bool) -> bool:
    """Idempotency gate: the mix file exists AND its sha256 matches the recorded
    provenance. A present-but-unrecorded or hash-mismatched mix is NOT current
    (rebuild it) -- never trust a bare file-presence check for a content artifact."""
    out = mix_path(smoke=smoke)
    prov = _provenance_path(smoke=smoke)
    if not out.exists() or not prov.exists():
        return False
    try:
        recorded = json.loads(prov.read_text()).get("sha256")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    if not recorded:
        return False
    return C664.sha256_file(out) == recorded


def build_h1_mix(*, smoke: bool, gpu_id: int = 0, force: bool = False) -> Path:
    """Build the librarian-contra-d1-seed42 marker training mix for H1 phase2.

    Returns the mix path. Idempotent: a no-op when the mix is already current
    (``_mix_is_current``) unless ``force``. Steps 1-3 above. FAILS LOUD if the
    builder subprocess crashes (rc != 0) -- there is no graceful "drop" here (the
    marker cell never hits the on-policy yield floor; a non-zero rc is a real
    error, including the rc==3 DROP code, which would mean the marker pool
    under-filled -- a bug, not a degradation we tolerate for the SINGLE H1 mix)."""
    out = mix_path(smoke=smoke)
    if not force and _mix_is_current(smoke=smoke):
        logger.info("[setup_h1_mix] mix already current (sha256 matches) -- skip: %s", out)
        return out

    cell = _h1_mix_cell()
    contexts = _needed_marker_R_contexts()

    # Reuse #664's verified marker path -- import inside the function so module
    # import (and the CPU smoke) never touches vLLM/transformers eagerly.
    import issue664_dispatch as D

    # ── Step 1: the deterministic marker training-question pool (UltraChat) ──
    # #664's _marker_question_pool: 300 diverse UltraChat first-user-turns disjoint
    # from the eval probes (smoke -> C.SMOKE_QUESTIONS). _write_pool writes it where
    # the builder's _questions_pool reads it back: onpolicy_cache/pools/marker.json.
    pool_path = D.CACHE_ROOT / "pools" / f"marker{'_smoke' if smoke else ''}.json"
    if force or not pool_path.exists():
        marker_qs = D._marker_question_pool(smoke)
        D._write_pool("marker", marker_qs, smoke=smoke)
        logger.info(
            "[setup_h1_mix] wrote marker pool (%d questions): %s", len(marker_qs), pool_path
        )
    else:
        marker_qs = json.loads(pool_path.read_text())["questions"]
        logger.info(
            "[setup_h1_mix] marker pool present (%d questions): %s", len(marker_qs), pool_path
        )

    # ── Step 2: the 5 base-greedy marker_R caches (1 source + 4 negatives) ──
    # Skip the vLLM engine entirely if every needed cache is already present
    # (resume / partial-rebuild). _elicit_marker_R is itself per-context idempotent
    # (skips a ctx whose cache exists), but we avoid even constructing the engine
    # when there is nothing to generate.
    missing = [c for c in contexts if not (D.CACHE_ROOT / "marker_R" / f"{c}.json").exists()]
    if force or missing:
        sources = [H1_MIX_SOURCE]
        neg_panel = C664.negative_panel()
        logger.info(
            "[setup_h1_mix] generating marker_R for %d context(s) on gpu %d: %s",
            len(missing) if not force else len(contexts),
            gpu_id,
            missing if not force else contexts,
        )
        # CVD-pin the engine to the assigned gpu (matches the in-process clobber the
        # launcher pins per cell; gotchas + the per-GPU fan-out rule).
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        llm = D._vllm_engine(2 * C664.MAX_NEW_TOKENS + 1024)
        try:
            D._elicit_marker_R(llm, sources, neg_panel, marker_qs, only_ctx=None, smoke=smoke)
        finally:
            D._teardown_vllm(llm)
    else:
        logger.info(
            "[setup_h1_mix] all %d marker_R caches present -- skip generation", len(contexts)
        )

    # ── Step 3: the CPU mix-builder subprocess (one clean process) ──
    cmd = [
        sys.executable,
        str(REPO / "scripts/issue664_build_training_data.py"),
        "--behavior",
        cell.behavior,
        "--source",
        cell.source,
        "--arm",
        cell.arm,
        "--dose",
        cell.dose,
        "--seed",
        str(cell.seed),
        "--cache-root",
        str(D.CACHE_ROOT),
    ]
    if smoke:
        cmd.append("--smoke")
    logger.info("[setup_h1_mix] building mix via subprocess: %s", cell.eval_key)
    result = subprocess.run(cmd, check=False, cwd=REPO, env={**os.environ})  # explicit env
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)

    if not out.exists():
        raise RuntimeError(
            f"[setup_h1_mix] builder exited 0 but the mix is missing: {out} "
            f"(cell {cell.eval_key}); marker pool / marker_R caches may be under-filled"
        )

    # ── sha256-pin the built mix (idempotency + reproducibility metadata) ──
    sha = C664.sha256_file(out)
    _provenance_path(smoke=smoke).write_text(
        json.dumps(
            {
                **C.repro_meta(seed=H1_MIX_SEED),
                "phase": "setup_h1_mix",
                "cell": cell.eval_key,
                "mix_path": str(out),
                "marker_R_contexts": contexts,
                "n_marker_questions": len(marker_qs),
                "smoke": smoke,
                "sha256": sha,
            },
            indent=2,
        )
    )
    logger.info("[setup_h1_mix] built %s (sha256 %s)", out, sha[:12])
    return out
