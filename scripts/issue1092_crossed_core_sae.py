#!/usr/bin/env python
"""Issue #1092 follow-up `crossed-core-sae` — SAE feature-grain decomposition of the
answer state on the crossed dense core (plan v11, tasks/.../1092/plans/plan.md).

Single dispatcher, phases A-D (phase E figures live in
`scripts/issue1092_crossed_core_sae_figs.py`, VM-side):

  A  stage pinned inputs (corpus + own-cell completions @ e5901706, r_B @ 037fcbb,
     SAE @ c37e53c4 via `BatchTopKSAE.ensure_downloaded`), rebuild the dense-core
     rows with the parent's own render/position code (`issue1092_gpu_phase`),
     SINK/MASSIVE-ACTIVATION MAP pre-pass per cell (v13 addendum: per-position
     sink identification + per-dim mu/sigma/gamma at layer 19 + sink-direction
     estimate over a row subsample; the map IS the round's sink-exclusion set,
     with the per-row 10x-median heuristic kept only as a LABELED fallback when
     the map yields no sinks), teacher-forced bf16 forwards with a LAYER-19
     resid_post hook (row-sharded across visible GPUs via CVD-pinned worker
     subprocesses), per-token SAE encode, per-row sparse summaries (prefix_end /
     context_end / pooled answer mean-max-frac; PRIMARY pooling excludes the
     map's sink positions/token-ids), reference-semantics fitness gate per arm
     (FVE >= 0.70 AND 30 <= L0 <= 130; instruct FAIL = K1 halt rc=21, base FAIL
     = drop + report), upload stores + fitness JSON BEFORE any fit (#825).
  B  dual/Gram-space ridge maps (context_end->pooled-mean PRIMARY, prefix_end->
     pooled-mean, induced averaged read + independently-fit averaged SECONDARY),
     grouped 6-fold by prefix (seed 0) with inner-grouped-CV lambda selection over
     issue658 RIDGE_LAMBDAS; identity+learned-bias + kNN retrieval per map;
     per-feature crossed ANOVA shares on the balanced grid + batched permutation
     nulls (per-draw re-selected max/top-k persisted); |cos(W_dec, r_B)| join with
     a selection-symmetric max-over-3-random-directions null.
  C  FROZEN this round (v14 JUDGED-LABEL FREEZE, user directive 2026-07-28
     21:24Z): ZERO judge API calls — the retained rubric design (level rubric
     VERBATIM from issue1482_feature_correlates + the v13 speaker_property
     5-way rubric, blind shuffled-union dispatch, drop-never-coerce, transport
     retried via eval.judge_dispatch) is the SPEC for the #1773-instrumented
     follow-up and is reachable ONLY via `--judged-labels on
     --override-label-freeze` (default `off`; `on` without the override is a
     fail-loud refusal). The 4-set union is STILL computed — it feeds phase C'.
  C' per-feature EVIDENCE EMISSION (plan v14 — the leg that must NOT be
     skipped): for every union feature (+ figure-reported rb-cos tail members):
     top-50 activating tuples (row_id, answer-token offset, activation), the
     per-row pooled-mean vector, decoder-space top-10 NN feature ids (one
     matmul), and top-30 mean-centered logit-footprint tokens (one matmul
     against a partially-loaded unembedding) -> `feature_evidence/` JSON+npz,
     uploaded so #1773 labels these features with NO new capture.
  D  per-feature join + digests + upload; every decoder-r_B alignment read is
     reported raw AND with the top-48 answer-PCA scaffold projected out of BOTH
     r_B and the decoder columns (v14 SCAFFOLD CONTROL — the projected variant
     is the HEADLINE alignment read; the max-over-3-random-directions null is
     recomputed in the SAME projected space).

Smoke IS this driver with tiny N through the same entrypoint (PASS_UNIFIED):
`--smoke-prefixes/--smoke-queries` subset the dense core (balanced), and the
subset threads through EVERY phase (capture, gate, fits, ANOVA, nulls, judge
pool, uploads). Width derives from visible devices (never narrowed by a smoke
branch). `--import-check` executes every deferred import; `--gate-probes` runs
the degenerate-input probes for every data-dependent gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM discipline)

import numpy as np  # noqa: E402

import issue1092_gpu_phase as parent  # noqa: E402  (render + position + r_B loaders)
import issue1482_feature_correlates as FC  # noqa: E402  (judge instrument, verbatim)
from issue1482_sae import (  # noqa: E402
    BOS_OFFSET,
    DICT_SIZE,
    OUTLIER_NORM_FACTOR,
    BatchTopKSAE,
    pool_answer_features,
    sparsify,
    token_inlier_mask,
)

logger = logging.getLogger("i1092_ccsae")

# ---------------------------------------------------------------------------
# Pinned constants (plan v11 section 10 Reproducibility Card)
# ---------------------------------------------------------------------------

DATA_REPO = parent.HF_DATA_REPO  # superkaiba1/explore-persona-space-data
HF_PREFIX = parent.HF_PREFIX  # issue1092_realistic_crossing
CORPUS_REV = "e590170619e7691c1a95c7b1bb20bda5fd4065ad"
RB_REV = "037fcbb210bc52c459959b0746cc268fe08bae96"
SAE_LAYER = 19  # resid_post layer 19 == hidden_states[20] under output_hidden_states
SAE_K = 64

CORPUS_STEM = f"{HF_PREFIX}/corpus"
COMPLETION_STEMS = {
    "cell_inst_own": f"{HF_PREFIX}/raw_completions/instruct",
    "cell_pre_own": f"{HF_PREFIX}/raw_completions/pretrained",
}
RB_STEM = "issue779_monitoring/r_b"

# Fitness gate (Source: #1482 sae_fitness.json — healthy 0.8097/61.7 vs marginal
# 0.5326 vs catastrophic -9.32/L0 1824; plan section 11)
FITNESS_FVE_MIN = 0.70
FITNESS_L0_MIN = 30.0
FITNESS_L0_MAX = 130.0
K1_HALT_RC = 21  # designed artifact-routed halt (gate JSON written first), never bare rc=1

DENSE_N_PREFIXES = 99
DENSE_N_QUERIES = 48
K2_MAX_DROP_FRAC = 0.05

# Fits (Source: #1092 Methodology; issue658 RIDGE_LAMBDAS; inner-grouped-CV per
# the #1417/#1335 GCV near-interpolation boundary)
RIDGE_LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
N_OUTER_FOLDS = 6
N_INNER_FOLDS = 5
ACTIVITY_FLOOR_FRAC = 0.005  # active in >= 0.5% of rows

# Judge tails (plan section 4 Phase C)
TAIL_PREFIX_N = 100
TAIL_R2_N = 50
CTRL_MATCH_N = 100
CTRL_QUERY_N = 100

# v13 RUBRIC AMENDMENT (user-chat 2026-07-28 20:55Z): the 5-way speaker_property
# field SUPERSEDES the defective binary persona_related field (the #1482 audit:
# 20/40 "persona" labels were plain language features, 11 more register/style).
SPEAKER_CLASSES = ("language", "register_style", "identity_disposition", "none", "unclear")
SPEAKER_JUDGE_SYSTEM = (
    "You label sparse-autoencoder features from example texts. Given up to 8 assistant "
    "answers where a feature fires strongly, decide what shared property the feature "
    "detects and classify WHICH KIND of speaker property it is, if any. Reason briefly, "
    'then output ONLY JSON: {"reasoning": "...", "label": "<= 8 words naming the shared '
    'property", "speaker_property": "language" | "register_style" | '
    '"identity_disposition" | "none" | "unclear"}. '
    "speaker_property is MUTUALLY EXCLUSIVE — choose exactly ONE: "
    "language = the shared property is which natural language / script the answers are "
    "written in. "
    "register_style = the formality, tone, genre, or verbosity register of the answers. "
    "identity_disposition = who the speaker IS or a stable trait/stance of the speaker — "
    "self-identification, refusal disposition, sycophancy, persona compliance, "
    "first-person identity. "
    "none = the shared property is topical content, task type, formatting, markup, or "
    "code syntax — nothing about the speaker. "
    "unclear = no coherent shared property is discernible from the examples."
)

# v14 JUDGED-LABEL FREEZE (user directive 2026-07-28 21:24Z): zero judge API
# calls this round — the rubric design above is retained as the SPEC for the
# #1773-instrumented follow-up and is unreachable without an explicit override.
LABEL_FREEZE_NOTE = (
    "JUDGED-LABEL FREEZE (2026-07-28): judged feature-label axes are DEFERRED to the "
    "#1773-instrumented follow-up round — zero judge API calls this round (plan v14 Phase C)"
)

# v14 SCAFFOLD CONTROL (external directive 2026-07-28 21:11Z; grounded on the #779
# rb-nuisance-profile round @ c724f5f588: 0.57-0.72 of r_B's squared mass at the
# mean-answer layer-19 read lies inside the top-48 principal subspace).
SCAFFOLD_RANK = 48
# Trait axis order == sorted r_B .pt basenames in parent.load_rb_directions
# (verified at RB_REV: evil.pt < hallucination.pt < sycophancy.pt).
RB_TRAIT_ORDER = ("evil", "hallucination", "sycophancy")

# v14 Phase C' evidence-emission sizes (plan v14 Phase C')
EVIDENCE_TOP_ROWS = 50  # top activating (row_id, offset, activation) tuples per feature
EVIDENCE_NN_K = 10  # decoder-space nearest-neighbour feature ids per feature
EVIDENCE_LOGIT_TOPK = 30  # mean-centered logit-footprint tokens per feature
RB_COS_FIG_TAIL_N = 50  # figure-reported alignment-tail members folded into the union

# Sink/massive-activation map (v13 addendum; thresholds stated in the map JSON)
SINKMAP_POS_CAP = 64  # per-position stats tracked for absolute positions 0..63
SINKMAP_MIN_OCC = 20  # min occurrences before a position/token-id can classify as sink
SINKMAP_MIN_RATE = 0.5  # min fraction of occurrences at sink scale
SINKMAP_TOP_DIMS = 30  # rogue dims reported per criterion
PRIOR_ABSTRACTION = PROJECT_ROOT / "eval_results/issue_1482/feature_correlates/abstraction.json"

CELL_MODEL = {
    "cell_inst_own": (parent.INSTRUCT_MODEL, parent.INSTRUCT_REVISION, "instruct"),
    "cell_pre_own": (parent.PRETRAINED_MODEL, parent.PRETRAINED_REVISION, "pretrained"),
}


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [i1092-ccsae] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Pure gate helpers (each has a degenerate-input probe in --gate-probes)
# ---------------------------------------------------------------------------


def fitness_gate_verdict(fve: float, l0: float) -> bool:
    """Pre-registered per-arm SAE fitness gate: FVE >= 0.70 AND 30 <= L0 <= 130."""
    return bool(
        np.isfinite(fve)
        and np.isfinite(l0)
        and fve >= FITNESS_FVE_MIN
        and FITNESS_L0_MIN <= l0 <= FITNESS_L0_MAX
    )


def assert_label_freeze(judged_labels: str, override: bool) -> None:
    """Fail-loud freeze gate: `--judged-labels on` without the explicit override is refused."""
    if judged_labels == "on" and not override:
        raise SystemExit(
            f"--judged-labels on REFUSED: {LABEL_FREEZE_NOTE}. Pass --override-label-freeze "
            "to run the retained judge design (the #1773 follow-up round)."
        )


def _judge_skip_reason(judged_labels: str, skip_judge: bool) -> str:
    """Attribute the judge-phase skip to the flag that actually suppressed dispatch
    (v22 Minor): `--skip-judge` when the override made dispatch reachable, else the
    JUDGED-LABEL FREEZE default."""
    if judged_labels == "on" and skip_judge:
        return "--skip-judge"
    return LABEL_FREEZE_NOTE


def check_k2(n_expected: int, n_dropped: int, label: str) -> None:
    """K2 kill criterion: >5% of dense-core rows dropped (render/position/completion)."""
    if n_expected <= 0:
        raise ValueError(f"K2 {label}: n_expected must be positive, got {n_expected}")
    frac = n_dropped / n_expected
    if frac > K2_MAX_DROP_FRAC:
        raise RuntimeError(
            f"K2 HALT ({label}): {n_dropped}/{n_expected} rows dropped "
            f"({frac:.1%} > {K2_MAX_DROP_FRAC:.0%}) — corpus mismatch vs the parent pins"
        )


def complete_subgrid(
    pairs: set[tuple[str, str]], prefixes: list[str], queries: list[str]
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Largest complete (prefix x query) subgrid after row drops.

    Iteratively removes the axis element with the most missing cells until the
    remaining grid is complete. Deterministic (ties break lexicographically).
    Returns (kept_prefixes, kept_queries, dropped_axis_elems).
    """
    kept_p = sorted(prefixes)
    kept_q = sorted(queries)
    dropped: list[tuple[str, str]] = []
    while True:
        miss_p = {p: sum((p, q) not in pairs for q in kept_q) for p in kept_p}
        miss_q = {q: sum((p, q) not in pairs for p in kept_p) for q in kept_q}
        total_missing = sum(miss_p.values())
        if total_missing == 0:
            return kept_p, kept_q, dropped
        worst_p = max(kept_p, key=lambda p: (miss_p[p], p))
        worst_q = max(kept_q, key=lambda q: (miss_q[q], q))
        if miss_p[worst_p] >= miss_q[worst_q]:
            kept_p.remove(worst_p)
            dropped.append(("prefix", worst_p))
        else:
            kept_q.remove(worst_q)
            dropped.append(("query", worst_q))
        if len(kept_p) < 2 or len(kept_q) < 2:
            raise RuntimeError(
                f"complete_subgrid degenerated below 2x2 (kept {len(kept_p)}x{len(kept_q)}); "
                f"dropped={dropped}"
            )


def assert_nondegenerate(name: str, observed_max: float, null_scale: float) -> None:
    """Runtime degeneracy assert on every observed-vs-null read (plan section 4).

    The observed statistic's magnitude must exceed eps both absolutely and
    relative to its null scale, else fail loud (the parent's item-9(iv)
    ==0-statistic incident class).
    """
    if not np.isfinite(observed_max):
        raise RuntimeError(f"degenerate observed statistic ({name}): non-finite")
    floor = max(1e-8, 1e-3 * max(null_scale, 0.0))
    if observed_max <= floor:
        raise RuntimeError(
            f"degenerate observed statistic ({name}): max={observed_max:.3e} <= "
            f"floor {floor:.3e} (null_scale={null_scale:.3e})"
        )


def clamp_knn_ks(n_pool: int, ks: tuple[int, ...] = (1, 10)) -> tuple[int, ...]:
    """Scale the kNN k-list to the candidate pool (k < n_pool; always keeps k=1)."""
    if n_pool < 2:
        raise ValueError(f"kNN pool too small: n_pool={n_pool}")
    kept = tuple(k for k in ks if k < n_pool)
    return kept if kept else (1,)


def grouped_fold_of(group_ids: list[str], n_folds: int, seed: int) -> dict[str, int]:
    """Deterministic grouped fold assignment (round-robin over a seeded permutation)."""
    uniq = sorted(set(group_ids))
    if len(uniq) < 2:
        raise ValueError(f"grouped folds need >=2 groups, got {len(uniq)}")
    n_folds = min(n_folds, len(uniq))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    return {uniq[j]: int(i % n_folds) for i, j in enumerate(order)}


def delta_bootstrap(
    tail_flags: np.ndarray, ctrl_flags: np.ndarray, n_draws: int, seed: int
) -> dict | None:
    """Headline Delta = rate(tail) - rate(ctrl) with a feature-level bootstrap CI.

    Returns None-shaped record (with reason) when either set has <2 labeled
    features — a recorded outcome, never a silent skip.
    """
    n1, n2 = len(tail_flags), len(ctrl_flags)
    if n1 < 2 or n2 < 2:
        return {
            "delta": None,
            "reason": f"insufficient labeled features (tail={n1}, ctrl={n2})",
            "n_tail": n1,
            "n_ctrl": n2,
        }
    tail = np.asarray(tail_flags, dtype=np.float64)
    ctrl = np.asarray(ctrl_flags, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r1 = tail[rng.integers(0, n1, size=(n_draws, n1))].mean(axis=1)
    r2 = ctrl[rng.integers(0, n2, size=(n_draws, n2))].mean(axis=1)
    d = r1 - r2
    lo, hi = np.percentile(d, [2.5, 97.5])
    delta = float(tail.mean() - ctrl.mean())
    tail_rate = float(tail.mean())
    if delta > 0 and lo > 0:
        verdict = "Confirmed"
    elif hi < 0 or (lo <= 0 <= hi and tail_rate <= 0.10):
        verdict = "Falsified"
    else:
        verdict = "Inconclusive"
    return {
        "delta": delta,
        "rate_tail": tail_rate,
        "rate_ctrl": float(ctrl.mean()),
        "ci95": [float(lo), float(hi)],
        "n_tail": n1,
        "n_ctrl": n2,
        "n_bootstrap": int(n_draws),
        "verdict": verdict,
        "estimand_note": (
            "Delta's estimand is the REALIZED top-100 set on this corpus "
            "(conditioning on the selection; not corpus-level generalization)"
        ),
    }


def sinkmap_min_occ(n_rows_used: int) -> int:
    """Occurrence floor for sink classification, derived from REALIZED map rows.

    Production (>=40 rows) uses SINKMAP_MIN_OCC; a tiny smoke subsample scales
    the floor down (never below 3) so the map stays derivable at smoke n — the
    gate-calibration / realized-slice-arithmetic discipline (gotchas.md).
    """
    return min(SINKMAP_MIN_OCC, max(3, n_rows_used // 2))


def derive_sink_sets(
    pos_occ: np.ndarray,
    pos_sink: np.ndarray,
    tok_occ: np.ndarray,
    tok_sink: np.ndarray,
    min_occ: int,
    min_rate: float = SINKMAP_MIN_RATE,
) -> dict:
    """Sink-map accumulators -> exclusion sets (v13 addendum).

    A position/token-id classifies as sink when it occurs >= min_occ times AND
    carries sink-scale norms on >= min_rate of its occurrences. BOTH sets empty
    => `exclusion_source: heuristic_fallback` (the per-row 10x-median heuristic,
    LABELED — never silent; plan v13 sink-map bullet).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        pos_rate = pos_sink / np.maximum(pos_occ, 1)
        tok_rate = tok_sink / np.maximum(tok_occ, 1)
    sink_positions = np.where((pos_occ >= min_occ) & (pos_rate >= min_rate))[0].astype(np.int64)
    sink_token_ids = np.where((tok_occ >= min_occ) & (tok_rate >= min_rate))[0].astype(np.int64)
    fallback = sink_positions.size == 0 and sink_token_ids.size == 0
    return {
        "sink_positions": sink_positions,
        "sink_token_ids": sink_token_ids,
        "exclusion_source": "heuristic_fallback" if fallback else "sink_map",
        "min_occ": int(min_occ),
        "min_rate": float(min_rate),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=Path("data/issue_1092/crossed_core_sae"))
    ap.add_argument("--cells", default="cell_inst_own,cell_pre_own")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda (worker + fit device)")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--hf-subdir", default="crossed_core_sae", help="HF upload subdir")
    ap.add_argument("--smoke-prefixes", type=int, default=0, help="0=all 99 (tiny-N smoke)")
    ap.add_argument("--smoke-queries", type=int, default=0, help="0=all 48 (tiny-N smoke)")
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--randdir-draws", type=int, default=200)
    ap.add_argument("--bootstrap-draws", type=int, default=10_000)
    ap.add_argument("--judge-limit", type=int, default=0, help="0=full union; N=pilot")
    ap.add_argument("--retest-n", type=int, default=FC.RETEST_N)
    ap.add_argument("--skip-judge", action="store_true", help="skip phase C entirely")
    ap.add_argument(
        "--judged-labels",
        choices=("off", "on"),
        default="off",
        help=(
            f"default off: {LABEL_FREEZE_NOTE}. 'on' is REFUSED unless "
            "--override-label-freeze is also passed (the retained judge design is the "
            "SPEC for the #1773 follow-up round)."
        ),
    )
    ap.add_argument(
        "--override-label-freeze",
        action="store_true",
        help="explicit opt-in required for --judged-labels on (see the freeze note above)",
    )
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument(
        "--sinkmap-rows",
        type=int,
        default=256,
        help="row subsample for the sink/massive-activation map (capped at n_rows)",
    )
    ap.add_argument("--chunk-rows", type=int, default=256, help="rows per persisted chunk")
    ap.add_argument("--fitness-tokens", type=int, default=250_000)
    ap.add_argument("--need-gb", type=float, default=10.0, help="out-root headroom floor")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=env default)")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--gate-probes", action="store_true")
    # worker mode (internal fan-out; CVD pinned in the launcher env per cell)
    ap.add_argument("--worker-capture", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-sinkmap", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cell", default="", help=argparse.SUPPRESS)
    ap.add_argument("--rows-file", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--row-start", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--row-end", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--shard-idx", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--gpu-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--verify-hook", action="store_true", help=argparse.SUPPRESS)
    return ap


def run_import_check() -> None:
    """Execute every deferred import + signature-bind key call shapes (Axis 1 leg)."""
    import inspect

    import torch  # noqa: F401
    from safetensors import safe_open  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    import issue1482_analysis as A
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.eval.batch_judge import is_transport_error_dict  # noqa: F401
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    # signature-bind the smoke-adjacent call shapes (arity/keyword drift check)
    inspect.signature(assert_out_root_headroom).bind(Path("/tmp/x"), 1.0, phase="p")
    inspect.signature(hub.stage_hub_prefix).bind(
        DATA_REPO, "p", Path("/tmp/x"), repo_type="dataset", revision="r"
    )
    inspect.signature(hub._upload).bind(
        Path("/tmp/x"), DATA_REPO, "dataset", "p", raise_on_error=True
    )
    inspect.signature(hub.list_hf_files_under_path).bind(
        object(), DATA_REPO, "p", repo_type="dataset", revision="r"
    )
    inspect.signature(dispatch_judge_items).bind(
        [],
        judge_model=FC.JUDGE_MODEL,
        judge_system_prompt="s",
        max_tokens=1,
        checkpoint_dir=Path("/tmp/x"),
        error_dict_factory=lambda r: {"error": True, "reason": r},
    )
    inspect.signature(identity_bias_predict).bind(
        np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    )
    inspect.signature(knn_retrieval).bind(
        np.zeros((2, 2)), np.zeros((2, 2)), ks=(1,), metric="euclidean"
    )
    inspect.signature(A._cohens_kappa).bind(["a"], ["a"])
    inspect.signature(parent.load_rb_directions).bind(RB_REV, 28, 3, 3584)
    inspect.signature(BatchTopKSAE.ensure_downloaded).bind(SAE_K, Path("/tmp/x"))
    # v14 additions: freeze gate, scaffold read, evidence emission
    from huggingface_hub import hf_hub_download, list_repo_tree

    inspect.signature(hf_hub_download).bind("repo/id", "file.json", revision="rev")
    # HUB_VERIFY_RETRY_EXEMPT: signature-bind smoke reference only, no network call made
    inspect.signature(list_repo_tree).bind(
        DATA_REPO, repo_type="dataset", path_in_repo="p", revision="rev"
    )
    inspect.signature(hub.retry_transient).bind(lambda: None, what="probe")
    inspect.signature(assert_label_freeze).bind("off", False)
    # v22 micro round: skip-reason attribution + evidence-union sets + rb-order pin
    inspect.signature(_judge_skip_reason).bind("off", False)
    inspect.signature(build_evidence_sets).bind(
        None, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), 2
    )
    inspect.signature(_assert_rb_trait_order).bind()
    inspect.signature(scaffold_basis).bind(np.zeros((3, 4), dtype=np.float32), "cpu")
    inspect.signature(rb_cosine_join).bind(
        np.zeros(2, dtype=np.int64), Path("/tmp/x"), 8, 0, "cpu", np.zeros((3, 4))
    )
    inspect.signature(_load_unembedding).bind("cell_inst_own", "cpu")
    inspect.signature(emit_feature_evidence).bind(
        Path("/tmp/x"), "cell_inst_own", {}, None, object(), "cpu"
    )
    print("[import-check] all deferred imports resolved + call shapes bind", flush=True)


def run_gate_probes() -> None:
    """Degenerate-input probes: every data-dependent gate fires its DESIGNED branch."""
    probes: list[tuple[str, str]] = []

    def expect(name: str, fn, exc=None) -> None:
        try:
            out = fn()
        except (Exception, SystemExit) as e:  # noqa: BLE001 — probe records the designed raise
            if exc is not None and isinstance(e, exc):
                probes.append((name, f"raised {type(e).__name__} (designed)"))
                return
            raise
        if exc is not None:
            raise RuntimeError(f"gate probe {name}: expected {exc} but returned {out!r}")
        probes.append((name, f"returned {out!r}"))

    expect("fitness_gate_fail", lambda: fitness_gate_verdict(0.53, 61.7))
    expect("fitness_gate_l0_fail", lambda: fitness_gate_verdict(0.81, 1824.0))
    expect("k2_over_floor", lambda: check_k2(100, 6, "probe"), exc=RuntimeError)
    expect("k2_under_floor", lambda: check_k2(100, 5, "probe"))
    pairs = {(p, q) for p in "abc" for q in "xy"} - {("a", "x")}
    expect("subgrid_hole_dropped", lambda: complete_subgrid(pairs, list("abc"), list("xy"))[2])
    expect(
        "subgrid_below_2x2",
        lambda: complete_subgrid(set(), list("ab"), list("xy")),
        exc=RuntimeError,
    )
    expect(
        "degeneracy_assert_fires",
        lambda: assert_nondegenerate("probe", 0.0, 1.0),
        exc=RuntimeError,
    )
    expect("degeneracy_assert_passes", lambda: assert_nondegenerate("probe", 0.5, 0.4))
    expect("knn_k_clamped", lambda: clamp_knn_ks(4))
    expect("knn_pool_too_small", lambda: clamp_knn_ks(1), exc=ValueError)
    expect("folds_one_group", lambda: grouped_fold_of(["p"], 6, 0), exc=ValueError)
    expect("delta_empty_set", lambda: delta_bootstrap(np.zeros(0), np.ones(3), 10, 0)["delta"])

    def _fve_one_row():
        import torch

        sd = {
            "encoder.weight": torch.zeros(4, 3),
            "encoder.bias": torch.zeros(4),
            "decoder.weight": torch.zeros(3, 4),
            "b_dec": torch.zeros(3),
            "k": torch.tensor(2, dtype=torch.int32),
            "threshold": torch.tensor(0.0),
        }
        sae = BatchTopKSAE(sd, k=2, act_dim=3, dict_size=4)
        return sae.fve_l0(torch.ones(1, 3))

    expect("fve_l0_lt2_inliers", _fve_one_row, exc=ValueError)
    # v13 additions: sink-map derivation gates + speaker_property validator
    occ = np.array([30.0, 30.0, 5.0])
    snk = np.array([30.0, 3.0, 5.0])
    expect(
        "sinkmap_sets_derived",
        lambda: int(
            derive_sink_sets(occ, snk, np.zeros(4), np.zeros(4), min_occ=20)["sink_positions"].size
        ),
    )
    expect(
        "sinkmap_empty_heuristic_fallback",
        lambda: derive_sink_sets(np.zeros(2), np.zeros(2), np.zeros(4), np.zeros(4), min_occ=20)[
            "exclusion_source"
        ],
    )
    expect("sinkmap_min_occ_smoke_floor", lambda: sinkmap_min_occ(12))
    expect(
        "speaker_out_of_set_is_content_drop",
        lambda: _validate_speaker({"speaker_property": "persona", "label": "x"}),
    )
    expect(
        "speaker_valid_label",
        lambda: _validate_speaker({"speaker_property": "Identity_Disposition", "label": "x"})[
            "speaker_property"
        ],
    )
    # v14 additions: judged-label freeze gate + scaffold rank cap
    expect("label_freeze_on_refused", lambda: assert_label_freeze("on", False), exc=SystemExit)
    expect("label_freeze_on_with_override", lambda: assert_label_freeze("on", True))
    expect("label_freeze_off_default", lambda: assert_label_freeze("off", False))
    rng = np.random.default_rng(0)
    expect(
        "scaffold_rank_capped_below_48",
        lambda: scaffold_basis(rng.normal(size=(3, 5)).astype(np.float32), "cpu")[1],
    )
    expect(
        "scaffold_needs_2_rows",
        lambda: scaffold_basis(np.zeros((1, 5), dtype=np.float32), "cpu"),
        exc=AssertionError,
    )
    for name, outcome in probes:
        print(f"[gate-probe] {name}: {outcome}", flush=True)
    print(f"[gate-probe] {len(probes)} gates exercised", flush=True)


# ---------------------------------------------------------------------------
# Phase A: staging + row building
# ---------------------------------------------------------------------------


def stage_inputs(out_root: Path) -> Path:
    """Revision-scoped probes + per-file staging of every pinned input stem.

    Scoped `list_hf_files_under_path` per stem at the pinned revision (>=1 file
    required, fail loud) — never `snapshot_download` on the ~1M-file data repo —
    then `hub.stage_hub_prefix` (retried per-file `hf_hub_download`). The SAE is
    pre-staged via `BatchTopKSAE.ensure_downloaded` (atomic, idempotent); r_B is
    probed here and loaded at fit time via `parent.load_rb_directions` (itself
    revision-scoped).
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    stage = out_root / "stage"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    stems = [(CORPUS_STEM, CORPUS_REV)] + [(s, CORPUS_REV) for s in COMPLETION_STEMS.values()]
    stems.append((RB_STEM, RB_REV))
    for stem, rev in stems:
        files = hub.list_hf_files_under_path(
            api, DATA_REPO, stem, repo_type="dataset", revision=rev
        )
        if not files:
            raise RuntimeError(f"staging probe: 0 files under {DATA_REPO}@{rev}:{stem}")
        _log(f"[phase=stage] probe OK: {len(files)} files under {stem}@{rev[:8]}")
    for stem in [CORPUS_STEM, *COMPLETION_STEMS.values()]:
        hub.stage_hub_prefix(DATA_REPO, stem, stage, repo_type="dataset", revision=CORPUS_REV)
    BatchTopKSAE.ensure_downloaded(SAE_K, out_root / "sae_cache")
    return stage


def load_completions(stage: Path, cell: str) -> dict[str, str]:
    """row_id -> completion text from the parent's persisted own-cell shards."""
    comp_dir = stage / COMPLETION_STEMS[cell]
    shards = sorted(comp_dir.glob(f"{cell}_shard*_part*.jsonl"))
    if not shards:
        raise RuntimeError(f"no completion shards staged under {comp_dir}")
    out: dict[str, str] = {}
    for path in shards:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[str(rec["row_id"])] = rec["completion"]
    return out


def select_dense_core(
    manifest: list[dict],
    prefix_store: dict,
    query_store: dict,
    smoke_prefixes: int,
    smoke_queries: int,
) -> tuple[list[dict], list[str], list[str]]:
    """Dense-core rows (hard grid assert) with an optional balanced tiny-N subset.

    Full run: exactly 99 prefixes x 48 queries = 4,752 rows (hard assert). Smoke:
    the N shortest prefixes x M shortest queries (by char length — keeps the CPU
    smoke cheap while staying REAL rows), balance re-asserted on the subset.
    """
    rows = [r for r in manifest if r.get("stratum") == "dense_core"]
    prefixes = sorted({r["prefix_id"] for r in rows})
    queries = sorted({r["query_id"] for r in rows})
    pairs = {(r["prefix_id"], r["query_id"]) for r in rows}
    if len(pairs) != len(rows):
        raise RuntimeError(
            f"dense core has duplicate (prefix, query) pairs: {len(pairs)} vs {len(rows)}"
        )
    if smoke_prefixes <= 0 and smoke_queries <= 0:
        if len(prefixes) != DENSE_N_PREFIXES or len(queries) != DENSE_N_QUERIES:
            raise RuntimeError(
                f"dense core grid {len(prefixes)}x{len(queries)} != "
                f"{DENSE_N_PREFIXES}x{DENSE_N_QUERIES}"
            )
        if len(rows) != DENSE_N_PREFIXES * DENSE_N_QUERIES:
            raise RuntimeError(f"dense core has {len(rows)} rows, expected 4752")
    else:
        n_p = smoke_prefixes if smoke_prefixes > 0 else len(prefixes)
        n_q = smoke_queries if smoke_queries > 0 else len(queries)

        def _plen(pid: str) -> int:
            return sum(len(t.get("content", "")) for t in parent._prefix_turns(prefix_store[pid]))

        def _qlen(qid: str) -> int:
            return len(parent._query_text(query_store[qid]))

        prefixes = sorted(prefixes, key=lambda p: (_plen(p), p))[:n_p]
        queries = sorted(queries, key=lambda q: (_qlen(q), q))[:n_q]
        keep = {(p, q) for p in prefixes for q in queries}
        rows = [r for r in rows if (r["prefix_id"], r["query_id"]) in keep]
        if len(rows) != len(prefixes) * len(queries):
            raise RuntimeError(
                f"smoke subset unbalanced: {len(rows)} rows != {len(prefixes)}x{len(queries)}"
            )
    rows = sorted(rows, key=lambda r: (r["prefix_id"], r["query_id"]))
    return rows, sorted(prefixes), sorted(queries)


def build_rows_file(
    out_root: Path, stage: Path, cell: str, args: argparse.Namespace
) -> tuple[Path, dict]:
    """Render the cell's dense-core rows with the PARENT's own code -> work JSONL.

    Rows with a missing/empty completion are dropped + counted (we require
    answer tokens; the parent treats empty as valid-None). K2 enforced later
    against the combined drop count.
    """
    corpus_dir = stage / CORPUS_STEM
    manifest = parent.load_manifest(corpus_dir)
    prefix_store = parent.load_store(corpus_dir, "prefix_store.jsonl")
    query_store = parent.load_store(corpus_dir, "query_store.jsonl")
    rows, prefixes, queries = select_dense_core(
        manifest, prefix_store, query_store, args.smoke_prefixes, args.smoke_queries
    )
    completions = load_completions(stage, cell)
    _, _, prompt_format = CELL_MODEL[cell]

    work = out_root / "work"
    work.mkdir(parents=True, exist_ok=True)
    rows_path = work / f"rows_{cell}.jsonl"
    n_missing = 0
    n_written = 0
    with open(rows_path, "w", encoding="utf-8") as f:
        for row in rows:
            comp = completions.get(str(row["row_id"]))
            if comp is None or comp == "":
                n_missing += 1
                continue
            prefix_text, prompt, completion = parent.render_row(
                row, prefix_store, query_store, prompt_format, "own", completion_override=comp
            )
            rec = {
                "row_id": str(row["row_id"]),
                "prefix_id": str(row["prefix_id"]),
                "query_id": str(row["query_id"]),
                "prefix_text": prefix_text,
                "prompt": prompt,
                "completion": completion,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    meta = {
        "cell": cell,
        "n_expected": len(rows),
        "n_written": n_written,
        "n_missing_completion": n_missing,
        "n_prefixes": len(prefixes),
        "n_queries": len(queries),
        "prefixes": prefixes,
        "queries": queries,
    }
    (work / f"rows_{cell}.meta.json").write_text(json.dumps(meta, indent=1))
    qtexts = {qid: parent._query_text(query_store[qid]) for qid in queries}
    (work / f"queries_{cell}.json").write_text(json.dumps(qtexts, ensure_ascii=False, indent=1))
    _log(
        f"[phase=rows cell={cell}] wrote {n_written}/{len(rows)} rows "
        f"({n_missing} missing/empty completions dropped)"
    )
    return rows_path, meta


def read_rows_file(rows_path: Path) -> list[dict]:
    """Text-mode JSONL read (never splitlines — U+2028 in real-corpus text)."""
    out = []
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Phase A: worker capture (one subprocess per visible GPU; CVD pinned in env)
# ---------------------------------------------------------------------------


def visible_gpu_ids() -> list[int]:
    """Physical GPU ids via an nvidia-smi subprocess (never torch device_count)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ},
        )
    except FileNotFoundError:
        return []
    if out.returncode != 0:
        return []
    ids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        pinned = [int(x) for x in cvd.split(",") if x.strip().isdigit()]
        ids = [i for i in ids if i in pinned]
    return ids


def _forward_layer19(model, input_ids, attention_mask, captured: dict):
    """Forward with a layer-19 resid_post hook; logits_to_keep guarded (parent pattern)."""
    handle = model.model.layers[SAE_LAYER].register_forward_hook(
        lambda _m, _i, output: captured.__setitem__(
            "h", output[0] if isinstance(output, tuple) else output
        )
    )
    try:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        try:
            out = model(**kwargs, logits_to_keep=1)
        except TypeError:
            out = model(**kwargs)
    finally:
        handle.remove()
    return out


def worker_capture(args: argparse.Namespace) -> int:
    """One shard's teacher-forced capture + per-token SAE encode + sparse summaries.

    Persists per CHUNK (`shard{ii}_chunk{k}.npz`, ~args.chunk_rows rows each,
    resume-skipped when present) + a fitness token sample + a done JSON with
    dropped rows and measured tok/s. Runs under a launcher-pinned
    CUDA_VISIBLE_DEVICES (single visible device -> cuda:0) or on CPU.
    """
    import torch

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    cell = args.cell
    model_name, revision, prompt_format = CELL_MODEL[cell]
    rows = read_rows_file(args.rows_file)[args.row_start : args.row_end]
    shard_dir = args.out_root / "features" / cell
    shard_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard_idx:02d}"
    done_path = shard_dir / f"{tag}_done.json"
    if done_path.exists():
        _log(f"[phase=capture cell={cell} {tag}] already done — skip")
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError("capture requires RIGHT padding (positions index unpadded rows)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    sae = BatchTopKSAE.load(k=SAE_K, device=device, cache_dir=args.out_root / "sae_cache")
    # v13: the sink-exclusion position/token-id sets come from the sink MAP
    # (fail-loud if the map phase has not run; heuristic only as the map's
    # labeled empty-map fallback).
    sink_tok_set, sink_pos_set, sink_src = load_sink_exclusion(args.out_root, cell)
    sink_tok_arr = (
        np.fromiter(sink_tok_set, dtype=np.int64) if sink_tok_set else np.zeros(0, np.int64)
    )
    sink_pos_arr = (
        np.fromiter(sink_pos_set, dtype=np.int64) if sink_pos_set else np.zeros(0, np.int64)
    )
    _log(
        f"[phase=capture cell={cell} {tag}] model+SAE loaded in {time.time() - t0:.0f}s; "
        f"sink exclusion source={sink_src} (pos={len(sink_pos_set)} tok={len(sink_tok_set)})"
    )

    boundary = parent._boundary_suffix(prompt_format)
    rng = np.random.default_rng(args.seed + 1000 * args.shard_idx)
    fitness_budget = max(1, args.fitness_tokens)
    fitness_states: list[np.ndarray] = []
    fitness_count = 0
    dropped_rows: list[dict] = []
    n_tokens_done = 0
    t_fwd0 = time.time()
    hook_verified = False

    chunk_rows = max(1, args.chunk_rows)
    batch_size = max(1, args.capture_batch)
    n_rows = len(rows)
    for c_start in range(0, n_rows, chunk_rows):
        c_end = min(c_start + chunk_rows, n_rows)
        chunk_idx = c_start // chunk_rows
        chunk_path = shard_dir / f"{tag}_chunk{chunk_idx:04d}.npz"
        if chunk_path.exists():
            _log(f"[phase=capture cell={cell} {tag}] chunk {chunk_idx} exists — skip")
            continue
        chunk = {
            "row_ids": [],
            "prefix_ids": [],
            "query_ids": [],
            "n_answer_tokens": [],
            "pe_idx": [],
            "pe_val": [],
            "ce_idx": [],
            "ce_val": [],
            "pooled_idx": [],
            "pooled_mean": [],
            "pooled_max": [],
            "pooled_frac": [],
            "pooled_argmax": [],
            "dense_mean": [],
            "pooledall_idx": [],
            "pooledall_val": [],
            "pe_sink": [],
            "ce_sink": [],
            "n_sink_answer": [],
        }
        for b_start in range(c_start, c_end, batch_size):
            b_rows = rows[b_start : min(b_start + batch_size, c_end)]
            batch_ids, positions, kept_rows = [], [], []
            for r in b_rows:
                try:
                    row_ids, pos = parent._capture_row_ids_and_positions(
                        tokenizer,
                        r["prefix_text"],
                        r["prompt"],
                        r["completion"],
                        boundary,
                        row_label=r["row_id"],
                    )
                except ValueError as e:
                    dropped_rows.append({"row_id": r["row_id"], "reason": str(e)[:200]})
                    continue
                batch_ids.append(row_ids)
                positions.append(pos)
                kept_rows.append(r)
            if not batch_ids:
                continue
            inputs = tokenizer.pad({"input_ids": batch_ids}, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            captured: dict = {}
            with torch.no_grad():
                outputs = _forward_layer19(model, input_ids, attention_mask, captured)
                h19 = captured["h"]  # (B, T, H) resid_post layer 19
                if not hook_verified and args.verify_hook:
                    ref = parent._call_model_with_hidden_states(
                        model, input_ids, attention_mask
                    ).hidden_states[SAE_LAYER + 1]
                    if not torch.allclose(h19, ref):
                        raise RuntimeError(
                            "layer-19 hook != hidden_states[20] (capture convention drift)"
                        )
                    hook_verified = True
                    _log(f"[phase=capture cell={cell} {tag}] hook==hidden_states[20] verified")
                for local_i, (row_tok_ids, pos, r) in enumerate(
                    zip(batch_ids, positions, kept_rows, strict=True)
                ):
                    n_total = pos["n_total"]
                    row_h = h19[local_i, :n_total, :]
                    a0, a1 = pos["answer_start"], pos["answer_end"]
                    # fitness pool: BOS-strip (positions >= BOS_OFFSET); outlier
                    # filter + var-FVE live inside sae.fve_l0 (reference semantics)
                    if fitness_count < fitness_budget and n_total > BOS_OFFSET:
                        avail = n_total - BOS_OFFSET
                        take = min(avail, max(1, fitness_budget // max(1, n_rows)) + 1)
                        sel = rng.choice(avail, size=min(take, avail), replace=False)
                        fs = row_h[BOS_OFFSET + torch.as_tensor(sel, device=row_h.device), :]
                        fitness_states.append(fs.to(torch.float16).cpu().numpy())
                        fitness_count += fs.shape[0]
                    pe_ce = sae.encode(
                        torch.stack([row_h[pos["prefix_end"]], row_h[pos["context_end"]]])
                    )
                    ans_h = row_h[a0:a1, :]
                    ans_f = sae.encode(ans_h)
                    # v13 sink exclusion (supersedes the v12 per-row heuristic):
                    # the PRIMARY pooling excludes the sink MAP's position +
                    # token-id sets; the per-row 10x-median heuristic runs ONLY
                    # as the map's LABELED empty-map fallback. The all-token
                    # MEAN stays the SECONDARY robustness read. BOS never
                    # enters the answer span (answer_start >> BOS_OFFSET).
                    if sink_src == "heuristic_fallback":
                        inlier = token_inlier_mask(ans_h)
                        row_mask = (
                            token_inlier_mask(row_h[BOS_OFFSET:])
                            if n_total > BOS_OFFSET + 1
                            else None
                        )

                        def _sink_flag(p: int, mask=row_mask) -> bool:
                            if p < BOS_OFFSET:
                                return True
                            return mask is not None and not bool(mask[p - BOS_OFFSET])
                    else:
                        ans_ids = np.asarray(row_tok_ids[a0:a1], dtype=np.int64)
                        sink_np = np.isin(ans_ids, sink_tok_arr) | np.isin(
                            np.arange(a0, a1, dtype=np.int64), sink_pos_arr
                        )
                        inlier = torch.from_numpy(~sink_np).to(ans_f.device)

                        def _sink_flag(p: int, ids=row_tok_ids) -> bool:
                            return p in sink_pos_set or int(ids[p]) in sink_tok_set

                    n_sink = int((~inlier).sum().item())
                    if not bool(inlier.any()):
                        inlier = torch.ones_like(inlier)  # degenerate: keep all (recorded)
                    ans_in = ans_f[inlier]
                    pooled = pool_answer_features(ans_in)
                    mean_all = ans_f.mean(0)
                    sp = sparsify(pooled)
                    # v14 Phase C' support: per-active-feature argmax answer-token
                    # offset (over the sink-excluded tokens; offsets index the
                    # ORIGINAL answer span) — one batched argmax per row.
                    inlier_offs = torch.nonzero(inlier, as_tuple=False).squeeze(-1)
                    idx_t = torch.as_tensor(sp["idx"].astype(np.int64), device=ans_in.device)
                    if idx_t.numel():
                        arg_off = (
                            inlier_offs[ans_in[:, idx_t].argmax(dim=0)]
                            .to(torch.int32)
                            .cpu()
                            .numpy()
                        )
                    else:
                        arg_off = np.zeros(0, dtype=np.int32)
                    # v14 SCAFFOLD input: dense sink-excluded mean-answer state
                    dense_mean_row = ans_h[inlier].mean(0).to(torch.float16).cpu().numpy()

                    for key, single in (("pe", pe_ce[0]), ("ce", pe_ce[1])):
                        nz = torch.nonzero(single != 0, as_tuple=False).squeeze(-1)
                        chunk[f"{key}_idx"].append(nz.cpu().numpy().astype(np.int32))
                        chunk[f"{key}_val"].append(single[nz].to(torch.float16).cpu().numpy())
                    nz_all = torch.nonzero(mean_all != 0, as_tuple=False).squeeze(-1)
                    chunk["pooledall_idx"].append(nz_all.cpu().numpy().astype(np.int32))
                    chunk["pooledall_val"].append(mean_all[nz_all].to(torch.float16).cpu().numpy())
                    chunk["pe_sink"].append(_sink_flag(pos["prefix_end"]))
                    chunk["ce_sink"].append(_sink_flag(pos["context_end"]))
                    chunk["n_sink_answer"].append(n_sink)
                    chunk["pooled_idx"].append(sp["idx"])
                    chunk["pooled_mean"].append(sp["mean"])
                    chunk["pooled_max"].append(sp["max"])
                    chunk["pooled_frac"].append(sp["frac"])
                    chunk["pooled_argmax"].append(arg_off)
                    chunk["dense_mean"].append(dense_mean_row)
                    chunk["row_ids"].append(r["row_id"])
                    chunk["prefix_ids"].append(r["prefix_id"])
                    chunk["query_ids"].append(r["query_id"])
                    chunk["n_answer_tokens"].append(a1 - a0)
                    n_tokens_done += n_total
                    del ans_f, ans_h, ans_in, pooled, pe_ce, mean_all
            captured.clear()
            del outputs, h19, input_ids, attention_mask
            elapsed = max(1e-9, time.time() - t_fwd0)
            _log(
                f"[phase=capture cell={cell} {tag}] rows {min(b_start + batch_size, c_end)}"
                f"/{n_rows} tok={n_tokens_done} ({n_tokens_done / elapsed:.0f} tok/s) "
                f"elapsed={elapsed:.0f}s"
            )
        _write_chunk_npz(chunk_path, chunk)
    if args.shard_idx == 0:
        capture_bare_and_template(model, tokenizer, sae, cell, prompt_format, args, shard_dir)
    fit_arr = (
        np.concatenate(fitness_states, axis=0)
        if fitness_states
        else np.zeros((0, sae.act_dim), dtype=np.float16)
    )
    np.save(shard_dir / f"{tag}_fitness.npy", fit_arr)
    done = {
        "cell": cell,
        "shard_idx": args.shard_idx,
        "n_rows_in": n_rows,
        "n_dropped": len(dropped_rows),
        "dropped_rows": dropped_rows,
        "n_tokens": int(n_tokens_done),
        "tok_per_s": float(n_tokens_done / max(1e-9, time.time() - t_fwd0)),
        "fitness_tokens": int(fit_arr.shape[0]),
        "device": device,
        "dtype": args.dtype,
        "sink_exclusion_source": sink_src,
    }
    done_path.write_text(json.dumps(done, indent=1))
    _log(
        f"[phase=capture cell={cell} {tag}] done: {n_rows - len(dropped_rows)}/{n_rows} rows, "
        f"{done['tok_per_s']:.0f} tok/s, fitness_tokens={done['fitness_tokens']}"
    )
    return 0


def _write_chunk_npz(chunk_path: Path, chunk: dict) -> None:
    """Atomic CSR-style chunk write (plain savez — never savez_compressed, #813)."""

    def _csr(list_of_arrays, dtype):
        lens = [len(a) for a in list_of_arrays]
        indptr = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
        flat = (
            np.concatenate(list_of_arrays).astype(dtype)
            if list_of_arrays and indptr[-1] > 0
            else np.zeros(0, dtype=dtype)
        )
        return indptr, flat

    payload: dict[str, np.ndarray] = {
        "row_ids": np.array(chunk["row_ids"], dtype=object),
        "prefix_ids": np.array(chunk["prefix_ids"], dtype=object),
        "query_ids": np.array(chunk["query_ids"], dtype=object),
        "n_answer_tokens": np.array(chunk["n_answer_tokens"], dtype=np.int32),
    }
    for key in ("pe", "ce"):
        indptr, idx = _csr(chunk[f"{key}_idx"], np.int32)
        _, val = _csr(chunk[f"{key}_val"], np.float16)
        payload[f"{key}_indptr"] = indptr
        payload[f"{key}_idx"] = idx
        payload[f"{key}_val"] = val
    indptr, idx = _csr(chunk["pooled_idx"], np.int32)
    payload["pooled_indptr"] = indptr
    payload["pooled_idx"] = idx
    for name in ("mean", "max", "frac"):
        _, val = _csr(chunk[f"pooled_{name}"], np.float16)
        payload[f"pooled_{name}"] = val
    _, argmax_val = _csr(chunk["pooled_argmax"], np.int32)
    payload["pooled_argmax"] = argmax_val
    payload["dense_mean"] = (
        np.stack(chunk["dense_mean"]).astype(np.float16)
        if chunk["dense_mean"]
        else np.zeros((0, 0), dtype=np.float16)
    )
    indptr, idx = _csr(chunk["pooledall_idx"], np.int32)
    _, val = _csr(chunk["pooledall_val"], np.float16)
    payload["pooledall_indptr"] = indptr
    payload["pooledall_idx"] = idx
    payload["pooledall_val"] = val
    payload["pe_sink"] = np.array(chunk["pe_sink"], dtype=np.int8)
    payload["ce_sink"] = np.array(chunk["ce_sink"], dtype=np.int8)
    payload["n_sink_answer"] = np.array(chunk["n_sink_answer"], dtype=np.int32)
    tmp = chunk_path.with_name(chunk_path.stem + ".tmp.npz")  # suffix stays .npz (#1092 gotcha)
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    os.replace(tmp, chunk_path)


def _render_bare_query_empty_system(tokenizer, query: str, prompt_format: str) -> str:
    """Bare-query render with an EMPTY system turn (map (d), plan v12 section 4).

    The instruct render passes an explicit empty system turn and ASSERTS the
    Qwen default system prompt was not silently inserted (the parent's own
    trap: `apply_chat_template([user])` injects the default system block).
    Deviation from `parent._render_bare_query` (which carries the default
    block) is deliberate and plan-mandated.
    """
    if prompt_format == "instruct":
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if "You are Qwen" in text:
            raise RuntimeError("bare-query render silently inserted the Qwen default system prompt")
        return text
    if prompt_format == "pretrained":
        return f"User: {query}\n\nAssistant:"
    raise ValueError(f"unknown prompt_format {prompt_format!r}")


def capture_bare_and_template(
    model, tokenizer, sae, cell: str, prompt_format: str, args, shard_dir: Path
) -> None:
    """Shard-0 extras: bare-query end-token captures + the template-only control.

    Bare-query arm (map (d)): per unique query, forward the NO-prefix render,
    capture the layer-19 state at the LAST prompt token (mirrors context_end =
    last prompt token), store the SAE features (sparse) AND the dense 3,584-d
    state (the dense companion). Template control (v12 taxonomy control (3)):
    ONE empty-content template forward per cell — pooled-mean features + the
    last-token features, for the base-vs-instruct DiD block in phase D.
    """
    import torch

    device = next(model.parameters()).device
    qpath = args.out_root / "work" / f"queries_{cell}.json"
    qtexts: dict[str, str] = json.loads(qpath.read_text())
    bare_path = shard_dir / "bare_query.npz"
    if not bare_path.exists():
        qids = sorted(qtexts)
        dense_states: list[np.ndarray] = []
        idx_list: list[np.ndarray] = []
        val_list: list[np.ndarray] = []
        n_tok: list[int] = []
        for qid in qids:
            text = _render_bare_query_empty_system(tokenizer, qtexts[qid], prompt_format)
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            input_ids = ids["input_ids"].to(device)
            attention_mask = torch.ones_like(input_ids)
            captured: dict = {}
            with torch.no_grad():
                _forward_layer19(model, input_ids, attention_mask, captured)
                h_last = captured["h"][0, -1, :]
                f = sae.encode(h_last.unsqueeze(0))[0]
            nz = torch.nonzero(f != 0, as_tuple=False).squeeze(-1)
            idx_list.append(nz.cpu().numpy().astype(np.int32))
            val_list.append(f[nz].to(torch.float16).cpu().numpy())
            dense_states.append(h_last.to(torch.float16).cpu().numpy())
            n_tok.append(int(input_ids.shape[1]))
            captured.clear()
        lens = [len(a) for a in idx_list]
        indptr = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
        tmp = bare_path.with_name(bare_path.stem + ".tmp.npz")
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                query_ids=np.array(qids, dtype=object),
                indptr=indptr,
                idx=(
                    np.concatenate(idx_list) if lens and sum(lens) else np.zeros(0, dtype=np.int32)
                ),
                val=(
                    np.concatenate(val_list)
                    if lens and sum(lens)
                    else np.zeros(0, dtype=np.float16)
                ),
                dense=np.stack(dense_states),
                n_tokens=np.array(n_tok, dtype=np.int32),
            )
        os.replace(tmp, bare_path)
        _log(f"[phase=bare cell={cell}] {len(qids)} bare-query captures -> {bare_path.name}")
    tpl_path = shard_dir / "template_control.npz"
    if not tpl_path.exists():
        if prompt_format == "instruct":
            # deliberately the DEFAULT template shape (matches the corpus rows'
            # template exposure: prefix turns carry no system turn, so the
            # default system block is injected there too)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True
            )
        else:
            text = "User: \n\nAssistant:"
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = ids["input_ids"].to(device)
        captured = {}
        import torch as _torch

        with _torch.no_grad():
            _forward_layer19(model, input_ids, _torch.ones_like(input_ids), captured)
            f_all = sae.encode(captured["h"][0])
        pooled_mean = f_all.mean(0)
        last = f_all[-1]
        payload: dict[str, np.ndarray] = {"n_tokens": np.array([input_ids.shape[1]], np.int32)}
        for name, vec in (("pooled_mean", pooled_mean), ("last", last)):
            nz = _torch.nonzero(vec != 0, as_tuple=False).squeeze(-1)
            payload[f"{name}_idx"] = nz.cpu().numpy().astype(np.int32)
            payload[f"{name}_val"] = vec[nz].to(_torch.float16).cpu().numpy()
        tmp = tpl_path.with_name(tpl_path.stem + ".tmp.npz")
        with open(tmp, "wb") as fh:
            np.savez(fh, **payload)
        os.replace(tmp, tpl_path)
        _log(f"[phase=template cell={cell}] template-only control -> {tpl_path.name}")


def run_capture_fanout(out_root: Path, cell: str, rows_path: Path, args) -> None:
    """Row-shard the cell across every visible GPU (CVD pinned in the LAUNCHER env).

    CPU host (no GPUs): one CPU worker — same subprocess shape, width from the
    environment (never a smoke branch). Worker logs echo their tail on failure.
    """
    n_rows = len(read_rows_file(rows_path))
    gpus = visible_gpu_ids()
    n_shards = max(1, len(gpus))
    bounds = np.linspace(0, n_rows, n_shards + 1, dtype=int)
    procs: list[tuple[subprocess.Popen, Path, int]] = []
    log_dir = out_root / "work" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        if bounds[i] == bounds[i + 1]:
            continue
        env = {**os.environ}
        gpu_flag = -1
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[i])
            gpu_flag = gpus[i]
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-capture",
            "--cell",
            cell,
            "--rows-file",
            str(rows_path),
            "--row-start",
            str(int(bounds[i])),
            "--row-end",
            str(int(bounds[i + 1])),
            "--shard-idx",
            str(i),
            "--gpu-id",
            str(gpu_flag),
            "--out-root",
            str(out_root),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--capture-batch",
            str(args.capture_batch),
            "--chunk-rows",
            str(args.chunk_rows),
            "--fitness-tokens",
            str(max(1, args.fitness_tokens // n_shards)),
            "--seed",
            str(args.seed),
            "--threads",
            str(args.threads),
            "--verify-hook",
        ]
        log_path = log_dir / f"capture_{cell}_shard{i:02d}.log"
        fh = open(log_path, "w")
        procs.append(
            (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), log_path, i)
        )
        _log(
            f"[phase=capture cell={cell}] shard {i}/{n_shards} rows "
            f"{bounds[i]}:{bounds[i + 1]} gpu={gpu_flag} log={log_path}"
        )
    failures = []
    for proc, log_path, i in procs:
        rc = proc.wait()
        if rc != 0:
            tail = "\n".join(log_path.read_text(errors="replace").split("\n")[-80:])
            _log(f"[phase=capture cell={cell}] shard {i} FAILED rc={rc}; log tail:\n{tail}")
            failures.append((i, rc))
    if failures:
        raise RuntimeError(f"capture fan-out failed for cell {cell}: shards {failures}")


def run_fitness_gate(out_root: Path, cell: str, device: str) -> dict:
    """Reference-parity FVE/L0 gate on the cell's sampled inlier token pool."""
    import torch

    shard_dir = out_root / "features" / cell
    parts = sorted(shard_dir.glob("shard*_fitness.npy"))
    if not parts:
        raise RuntimeError(f"no fitness samples under {shard_dir}")
    h = np.concatenate([np.load(p) for p in parts], axis=0)
    if h.shape[0] < 2:
        raise RuntimeError(f"fitness pool for {cell} has {h.shape[0]} rows (<2)")
    sae = BatchTopKSAE.load(k=SAE_K, device=device, cache_dir=out_root / "sae_cache")
    fve, l0, diag = sae.fve_l0(torch.from_numpy(h.astype(np.float32)))
    verdict = fitness_gate_verdict(fve, l0)
    # v12 taxonomy control (2): rogue-dimension gamma = ||mu|| / ||sigma|| at
    # layer 19, over the sampled (BOS-stripped) token pool.
    h32 = h.astype(np.float32)
    gamma = float(
        np.linalg.norm(h32.mean(axis=0)) / max(float(np.linalg.norm(h32.std(axis=0))), 1e-12)
    )
    rec = {
        "cell": cell,
        "fve": float(fve),
        "l0": float(l0),
        "diag": diag,
        "gamma_layer19": gamma,
        "n_tokens_sampled": int(h.shape[0]),
        "thresholds": {
            "fve_min": FITNESS_FVE_MIN,
            "l0_min": FITNESS_L0_MIN,
            "l0_max": FITNESS_L0_MAX,
        },
        "pass": bool(verdict),
        "reference": {"published_fve_k64": 0.80572265625, "issue1482_fve": 0.8097},
    }
    _log(f"[phase=fitness cell={cell}] fve={fve:.4f} l0={l0:.1f} pass={verdict} diag={diag}")
    del sae
    return rec


# ---------------------------------------------------------------------------
# Phase A0: sink/massive-activation map (v13 addendum — committed deliverable)
# ---------------------------------------------------------------------------


def sink_map_paths(out_root: Path, cell: str) -> tuple[Path, Path]:
    """(json, npz) artifact paths for a cell's sink map."""
    d = out_root / "sink_map"
    return d / f"sink_map_{cell}.json", d / f"sink_map_{cell}.npz"


def load_sink_exclusion(out_root: Path, cell: str) -> tuple[set[int], set[int], str]:
    """Map-derived sink exclusion sets for the capture workers.

    Fail-loud on a missing map (the sink-map phase MUST precede capture);
    returns (sink_token_ids, sink_positions, exclusion_source). An
    `exclusion_source == "heuristic_fallback"` (both sets empty) tells the
    worker to use the LABELED per-row 10x-median heuristic instead.
    """
    json_path, _ = sink_map_paths(out_root, cell)
    if not json_path.exists():
        raise RuntimeError(
            f"sink map missing for {cell} at {json_path} — the sink-map phase must run first"
        )
    rec = json.loads(json_path.read_text())
    return (
        {int(t) for t in rec["sink_token_ids"]},
        {int(p) for p in rec["sink_positions"]},
        str(rec["exclusion_source"]),
    )


def worker_sinkmap(args: argparse.Namespace) -> int:
    """Sink/massive-activation map over a dense-core row subsample (ONE worker).

    Per row: layer-19 per-token L2 norms; sink = norm > OUTLIER_NORM_FACTOR x
    post-BOS row-median (reference 10x-median semantics). Batched accumulators:
    per-ABSOLUTE-POSITION occ/sink/mean-norm (cap SINKMAP_POS_CAP), per-TOKEN-ID
    occ/sink, per-DIM mu/sigma/|x|max over ALL tokens (gamma = ||mu||/||sigma||),
    sink-vs-content mean-direction sums, per-SEGMENT (bos/prefix/query/answer)
    sink rates. Derives the exclusion sets via `derive_sink_sets` (occurrence
    floor scaled to REALIZED rows via `sinkmap_min_occ`) and writes
    sink_map_{cell}.npz then .json (json last == map complete; resume key).
    """
    import torch

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    cell = args.cell
    model_name, revision, prompt_format = CELL_MODEL[cell]
    json_path, npz_path = sink_map_paths(args.out_root, cell)
    if json_path.exists():
        _log(f"[phase=sinkmap cell={cell}] map exists — skip")
        return 0
    json_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = read_rows_file(args.rows_file)
    take = min(max(1, args.sinkmap_rows), len(all_rows))
    stride = max(1, len(all_rows) // take)
    rows = all_rows[::stride][:take]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    _log(
        f"[phase=sinkmap cell={cell}] model loaded in {time.time() - t0:.0f}s; "
        f"rows={len(rows)}/{len(all_rows)} (stride={stride})"
    )
    boundary = parent._boundary_suffix(prompt_format)
    hidden = int(model.config.hidden_size)
    vocab = int(model.config.vocab_size)
    pos_occ = np.zeros(SINKMAP_POS_CAP, dtype=np.float64)
    pos_sink = np.zeros(SINKMAP_POS_CAP, dtype=np.float64)
    pos_norm_sum = np.zeros(SINKMAP_POS_CAP, dtype=np.float64)
    tok_occ = np.zeros(vocab, dtype=np.float64)
    tok_sink = np.zeros(vocab, dtype=np.float64)
    dim_sum = np.zeros(hidden, dtype=np.float64)
    dim_sumsq = np.zeros(hidden, dtype=np.float64)
    dim_absmax = np.zeros(hidden, dtype=np.float64)
    sink_vec = np.zeros(hidden, dtype=np.float64)
    content_vec = np.zeros(hidden, dtype=np.float64)
    n_dim_tokens = 0
    n_sink_tok = 0
    n_content_tok = 0
    seg_stats: dict[str, list[int]] = {s: [0, 0] for s in ("bos", "prefix", "query", "answer")}
    n_used = 0
    dropped = 0
    batch_size = max(1, args.capture_batch)
    for b_start in range(0, len(rows), batch_size):
        b_rows = rows[b_start : b_start + batch_size]
        batch_ids, positions = [], []
        for r in b_rows:
            try:
                row_ids, pos = parent._capture_row_ids_and_positions(
                    tokenizer,
                    r["prefix_text"],
                    r["prompt"],
                    r["completion"],
                    boundary,
                    row_label=r["row_id"],
                )
            except ValueError:
                dropped += 1
                continue
            batch_ids.append(row_ids)
            positions.append(pos)
        if not batch_ids:
            continue
        inputs = tokenizer.pad({"input_ids": batch_ids}, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        captured: dict = {}
        with torch.no_grad():
            _forward_layer19(model, input_ids, attention_mask, captured)
            h19 = captured["h"]
            for local_i, (pos, row_ids) in enumerate(zip(positions, batch_ids, strict=True)):
                n_total = pos["n_total"]
                row_h = h19[local_i, :n_total, :].to(torch.float32)
                norms = row_h.norm(dim=-1)
                med_pool = norms[BOS_OFFSET:] if n_total > BOS_OFFSET + 1 else norms
                med = float(med_pool.median())
                sink = (norms > OUTLIER_NORM_FACTOR * med).cpu().numpy()
                norms_np = norms.cpu().numpy().astype(np.float64)
                cap = min(n_total, SINKMAP_POS_CAP)
                pos_occ[:cap] += 1.0
                pos_sink[:cap] += sink[:cap].astype(np.float64)
                pos_norm_sum[:cap] += norms_np[:cap]
                ids_np = np.asarray(row_ids, dtype=np.int64)
                np.add.at(tok_occ, ids_np, 1.0)
                np.add.at(tok_sink, ids_np, sink.astype(np.float64))
                row_np = row_h.cpu().numpy().astype(np.float64)
                dim_sum += row_np.sum(axis=0)
                dim_sumsq += (row_np**2).sum(axis=0)
                dim_absmax = np.maximum(dim_absmax, np.abs(row_np).max(axis=0))
                n_dim_tokens += int(n_total)
                if sink.any():
                    sink_vec += row_np[sink].sum(axis=0)
                    n_sink_tok += int(sink.sum())
                if (~sink).any():
                    content_vec += row_np[~sink].sum(axis=0)
                    n_content_tok += int((~sink).sum())
                a0, a1 = pos["answer_start"], pos["answer_end"]
                segs = {
                    "bos": (0, min(BOS_OFFSET, n_total)),
                    "prefix": (min(BOS_OFFSET, n_total), pos["prefix_end"] + 1),
                    "query": (pos["prefix_end"] + 1, pos["context_end"] + 1),
                    "answer": (a0, a1),
                }
                for name, (s, e) in segs.items():
                    if e > s:
                        seg_stats[name][0] += e - s
                        seg_stats[name][1] += int(sink[s:e].sum())
                n_used += 1
        captured.clear()
        del h19, input_ids, attention_mask
        _log(
            f"[phase=sinkmap cell={cell}] rows {min(b_start + batch_size, len(rows))}"
            f"/{len(rows)} elapsed={time.time() - t0:.0f}s"
        )
    if n_used == 0:
        raise RuntimeError(f"sink map for {cell}: zero usable rows (dropped={dropped})")
    mu = dim_sum / max(1, n_dim_tokens)
    var = np.maximum(dim_sumsq / max(1, n_dim_tokens) - mu**2, 0.0)
    sigma = np.sqrt(var)
    gamma = float(np.linalg.norm(mu) / max(float(np.linalg.norm(sigma)), 1e-12))
    min_occ = sinkmap_min_occ(n_used)
    sets = derive_sink_sets(pos_occ, pos_sink, tok_occ, tok_sink, min_occ=min_occ)
    ratio = np.abs(mu) / np.maximum(sigma, 1e-12)
    top_ratio = np.argsort(ratio)[::-1][:SINKMAP_TOP_DIMS]
    top_absmax = np.argsort(dim_absmax)[::-1][:SINKMAP_TOP_DIMS]

    def _dim_rows(idx: np.ndarray) -> list[dict]:
        return [
            {
                "dim": int(d),
                "mu": float(mu[d]),
                "sigma": float(sigma[d]),
                "absmax": float(dim_absmax[d]),
            }
            for d in idx.tolist()
        ]

    def _unit(v: np.ndarray, n: int) -> tuple[np.ndarray, float] | None:
        if n <= 0:
            return None
        m = v / n
        nrm = float(np.linalg.norm(m))
        return m / max(nrm, 1e-12), nrm

    sink_dir = _unit(sink_vec, n_sink_tok)
    content_dir = _unit(content_vec, n_content_tok)
    dir_cos = (
        float(np.dot(sink_dir[0], content_dir[0]))
        if sink_dir is not None and content_dir is not None
        else None
    )
    tok_rate = tok_sink / np.maximum(tok_occ, 1.0)
    elig = np.where(tok_occ >= min_occ)[0]
    tok_order = elig[np.argsort(tok_rate[elig])[::-1]][:50]
    sink_tok_set = {int(t) for t in sets["sink_token_ids"].tolist()}
    token_table = [
        {
            "token_id": int(t),
            "occ": int(tok_occ[t]),
            "sink_rate": float(tok_rate[t]),
            "piece": repr(tokenizer.decode([int(t)]))[:16],
            "is_sink": int(t) in sink_tok_set,
        }
        for t in tok_order.tolist()
        if tok_rate[t] > 0
    ]
    pos_rate = pos_sink / np.maximum(pos_occ, 1.0)
    position_table = [
        {"position": int(p), "occ": int(pos_occ[p]), "sink_rate": float(pos_rate[p])}
        for p in np.where(pos_rate > 0)[0].tolist()
    ]
    if sets["exclusion_source"] == "heuristic_fallback":
        _log(
            f"[phase=sinkmap cell={cell}] WARNING: map yielded NO sink positions/token-ids"
            " — capture uses the LABELED per-row 10x-median heuristic fallback"
        )
    payload = {
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "dim_absmax": dim_absmax.astype(np.float32),
        "pos_occ": pos_occ,
        "pos_sink": pos_sink,
        "pos_norm_mean": pos_norm_sum / np.maximum(pos_occ, 1.0),
        "sink_dir_unit": (
            sink_dir[0].astype(np.float32) if sink_dir is not None else np.zeros(hidden, np.float32)
        ),
        "content_dir_unit": (
            content_dir[0].astype(np.float32)
            if content_dir is not None
            else np.zeros(hidden, np.float32)
        ),
        "sink_positions": sets["sink_positions"],
        "sink_token_ids": sets["sink_token_ids"],
    }
    tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")  # suffix stays .npz (savez gotcha)
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    os.replace(tmp, npz_path)
    rec = {
        "cell": cell,
        "n_rows_used": int(n_used),
        "n_rows_dropped": int(dropped),
        "n_rows_total_cell": len(all_rows),
        "n_tokens": int(n_dim_tokens),
        "gamma_layer19_all_tokens": gamma,
        "sink_norm_factor": float(OUTLIER_NORM_FACTOR),
        "min_occ_effective": int(min_occ),
        "min_rate": float(SINKMAP_MIN_RATE),
        "sink_positions": [int(p) for p in sets["sink_positions"].tolist()],
        "sink_token_ids": [int(t) for t in sets["sink_token_ids"].tolist()],
        "exclusion_source": sets["exclusion_source"],
        "n_sink_tokens": int(n_sink_tok),
        "n_content_tokens": int(n_content_tok),
        "sink_content_dir_cosine": dir_cos,
        "sink_dir_norm": sink_dir[1] if sink_dir is not None else None,
        "content_dir_norm": content_dir[1] if content_dir is not None else None,
        "segment_sink_rates": {
            k: {"n_tokens": int(v[0]), "sink_rate": (v[1] / v[0]) if v[0] else None}
            for k, v in seg_stats.items()
        },
        "rogue_dims_by_mu_over_sigma": _dim_rows(top_ratio),
        "rogue_dims_by_absmax": _dim_rows(top_absmax),
        "top_sink_tokens": token_table,
        "position_sink_table": position_table,
        "repro": _repro_meta(),
    }
    json_path.write_text(json.dumps(rec, indent=1))
    _log(
        f"[phase=sinkmap cell={cell}] done: gamma={gamma:.3f} "
        f"sink_positions={len(rec['sink_positions'])} sink_token_ids={len(rec['sink_token_ids'])} "
        f"source={rec['exclusion_source']} ({time.time() - t0:.0f}s)"
    )
    return 0


def run_sinkmap_subprocess(out_root: Path, cell: str, rows_path: Path, args) -> None:
    """Sink-map phase as ONE CVD-pinned worker subprocess (mirrors the capture
    fan-out shape; resume-skipped on the map JSON; fail-loud with a log tail)."""
    json_path, _ = sink_map_paths(out_root, cell)
    if json_path.exists():
        _log(f"[phase=sinkmap cell={cell}] map exists — skip")
        return
    gpus = visible_gpu_ids()
    env = {**os.environ}
    gpu_flag = -1
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        gpu_flag = gpus[0]
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-sinkmap",
        "--cell",
        cell,
        "--rows-file",
        str(rows_path),
        "--out-root",
        str(out_root),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--capture-batch",
        str(args.capture_batch),
        "--sinkmap-rows",
        str(args.sinkmap_rows),
        "--seed",
        str(args.seed),
        "--threads",
        str(args.threads),
        "--gpu-id",
        str(gpu_flag),
    ]
    log_dir = out_root / "work" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sinkmap_{cell}.log"
    with open(log_path, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        tail = "\n".join(log_path.read_text(errors="replace").split("\n")[-80:])
        _log(f"[phase=sinkmap cell={cell}] FAILED rc={rc}; log tail:\n{tail}")
        raise RuntimeError(f"sink-map worker failed for {cell} (rc={rc})")
    _log(f"[phase=sinkmap cell={cell}] done log={log_path}")


# ---------------------------------------------------------------------------
# Phase B: sparse store -> dense active matrices
# ---------------------------------------------------------------------------


class CellStore:
    """Dense active-feature matrices for one cell, rebuilt from the sparse chunks.

    Vectorized rebuild (no per-row Python loop): each chunk's CSR block scatters
    into the dense output via repeat/lookup indexing.
    """

    def __init__(self, out_root: Path, cell: str):
        shard_dir = out_root / "features" / cell
        chunks = sorted(shard_dir.glob("shard*_chunk*.npz"))
        if not chunks:
            raise RuntimeError(f"no capture chunks under {shard_dir}")
        row_ids: list[str] = []
        prefix_ids: list[str] = []
        query_ids: list[str] = []
        n_ans: list[np.ndarray] = []
        pe_sink: list[np.ndarray] = []
        ce_sink: list[np.ndarray] = []
        n_sink: list[np.ndarray] = []
        dense_mean_blocks: list[np.ndarray] = []
        sparse: dict[str, list] = {
            k: [] for k in ("pe", "ce", "mean", "max", "frac", "argmax", "mean_all")
        }
        for path in chunks:
            z = np.load(path, allow_pickle=True)
            if "dense_mean" not in z.files or "pooled_argmax" not in z.files:
                raise RuntimeError(
                    f"{path} lacks v14 keys (dense_mean/pooled_argmax) — pre-v14 capture "
                    "chunk; re-run the capture phase against a fresh out-root"
                )
            dense_mean_blocks.append(z["dense_mean"])
            row_ids.extend(z["row_ids"].tolist())
            prefix_ids.extend(z["prefix_ids"].tolist())
            query_ids.extend(z["query_ids"].tolist())
            n_ans.append(z["n_answer_tokens"])
            pe_sink.append(z["pe_sink"])
            ce_sink.append(z["ce_sink"])
            n_sink.append(z["n_sink_answer"])
            for key, (ip, ix, vv) in {
                "pe": ("pe_indptr", "pe_idx", "pe_val"),
                "ce": ("ce_indptr", "ce_idx", "ce_val"),
                "mean": ("pooled_indptr", "pooled_idx", "pooled_mean"),
                "max": ("pooled_indptr", "pooled_idx", "pooled_max"),
                "frac": ("pooled_indptr", "pooled_idx", "pooled_frac"),
                "argmax": ("pooled_indptr", "pooled_idx", "pooled_argmax"),
                "mean_all": ("pooledall_indptr", "pooledall_idx", "pooledall_val"),
            }.items():
                sparse[key].append((z[ip], z[ix], z[vv]))
        self.dense_mean = (
            np.concatenate(dense_mean_blocks, axis=0).astype(np.float32)
            if dense_mean_blocks
            else np.zeros((0, 0), dtype=np.float32)
        )
        if self.dense_mean.shape[0] != len(row_ids):
            raise RuntimeError(
                f"dense_mean rows {self.dense_mean.shape[0]} != {len(row_ids)} chunk rows"
            )
        self.row_ids = row_ids
        self.prefix_ids = prefix_ids
        self.query_ids = query_ids
        self.n_answer_tokens = np.concatenate(n_ans) if n_ans else np.zeros(0, np.int32)
        self.pe_sink = np.concatenate(pe_sink).astype(bool)
        self.ce_sink = np.concatenate(ce_sink).astype(bool)
        self.n_sink_answer = np.concatenate(n_sink)
        self.n_rows = len(row_ids)
        self._sparse = sparse
        self.cell = cell

    def active_set(self, key: str, floor_frac: float) -> np.ndarray:
        """Feature ids active (nonzero) in >= max(1, ceil(floor_frac * n_rows)) rows."""
        floor = max(1, int(np.ceil(floor_frac * self.n_rows)))
        counts = np.zeros(DICT_SIZE, dtype=np.int64)
        for _indptr, idx, val in self._sparse[key]:
            nz = np.asarray(idx)[np.asarray(val) != 0]
            if nz.size:
                counts += np.bincount(nz.astype(np.int64), minlength=DICT_SIZE)
        feats = np.where(counts >= floor)[0].astype(np.int64)
        if feats.size == 0:
            raise RuntimeError(f"no active features for {self.cell}/{key} at floor {floor}")
        return feats

    def dense(self, key: str, feats: np.ndarray) -> np.ndarray:
        """(n_rows, len(feats)) fp32 dense matrix restricted to `feats` columns."""
        col_of = np.full(DICT_SIZE, -1, dtype=np.int64)
        col_of[feats] = np.arange(len(feats))
        out = np.zeros((self.n_rows, len(feats)), dtype=np.float32)
        row_offset = 0
        for indptr, idx, val in self._sparse[key]:
            nrows_b = len(indptr) - 1
            rows = np.repeat(np.arange(nrows_b) + row_offset, np.diff(indptr))
            cols = col_of[np.asarray(idx, dtype=np.int64)]
            keep = cols >= 0
            out[rows[keep], cols[keep]] = np.asarray(val, dtype=np.float32)[keep]
            row_offset += nrows_b
        return out


# ---------------------------------------------------------------------------
# Phase B: dual/Gram-space ridge with inner-grouped-CV lambda selection
# ---------------------------------------------------------------------------


def _eigh_robust(gram):
    """cuda eigh with CPU LAPACK fallback (cuSOLVER non-convergence, #1335 gotcha)."""
    import torch

    try:
        return torch.linalg.eigh(gram)
    except torch.linalg.LinAlgError:
        w, v = torch.linalg.eigh(gram.cpu())
        _log(f"[fit] cuda eigh failed to converge (n={gram.shape[0]}); CPU fallback engaged")
        return w.to(gram.device), v.to(gram.device)


def _dual_ridge_predict(x_tr, y_tr, x_te, lambdas):
    """Batched-over-lambda dual ridge: returns {lam: predictions on x_te} (torch)."""

    mean_x = x_tr.mean(dim=0, keepdim=True)
    mean_y = y_tr.mean(dim=0, keepdim=True)
    xc = x_tr - mean_x
    yc = y_tr - mean_y
    gram = xc @ xc.T
    w, u = _eigh_robust(gram)
    z = u.T @ yc  # (n_tr, d_y)
    k_te = (x_te - mean_x) @ xc.T  # (n_te, n_tr)
    m = k_te @ u  # (n_te, n_tr)
    preds = {}
    for lam in lambdas:
        scale = 1.0 / (w + lam)
        preds[lam] = m @ (z * scale[:, None]) + mean_y
    return preds


def _primal_ridge_predict(x_tr, y_tr, x_te, lambdas):
    """Primal-space twin (d_x <= n_tr — e.g. the bare-query arms): eigh on X^T X."""

    mean_x = x_tr.mean(dim=0, keepdim=True)
    mean_y = y_tr.mean(dim=0, keepdim=True)
    xc = x_tr - mean_x
    yc = y_tr - mean_y
    cov = xc.T @ xc  # (d_x, d_x)
    w, u = _eigh_robust(cov)
    z = u.T @ (xc.T @ yc)  # (d_x, d_y)
    m = (x_te - mean_x) @ u  # (n_te, d_x)
    preds = {}
    for lam in lambdas:
        scale = 1.0 / (w + lam)
        preds[lam] = m @ (z * scale[:, None]) + mean_y
    return preds


def _ridge_predict(x_tr, y_tr, x_te, lambdas):
    """Dispatch primal vs dual by the cheaper factorization (min(n_tr, d_x))."""
    if x_tr.shape[1] <= x_tr.shape[0]:
        return _primal_ridge_predict(x_tr, y_tr, x_te, lambdas)
    return _dual_ridge_predict(x_tr, y_tr, x_te, lambdas)


def _pooled_mse(pred, y):
    return float(((pred - y) ** 2).mean().item())


def fit_map_oof(
    x: np.ndarray,
    y: np.ndarray,
    group_ids: list[str],
    seed: int,
    device: str,
    label: str,
    n_primary_cols: int = 0,
) -> dict:
    """Grouped-6-fold OOF ridge (inner-grouped-CV lambda per outer fold).

    Groups = prefixes for maps (a)/(b)/(c); QUERIES for the bare-query arm (d).
    `n_primary_cols` > 0 restricts the inner-CV lambda selection to the first
    n columns of y (the PRIMARY pooled-mean block; max/frac ride as exploratory
    targets predicted at the chosen lambda). Returns oof predictions + per-fold
    lambda + fold assignment + pilot timing (the plan's pilot-gated basis: ONE
    inner-fold factorization at production shape is timed FIRST and the
    fold x lambda extrapolation printed).
    """
    import torch

    dev = torch.device(device)
    xt = torch.from_numpy(x).to(dev)
    yt = torch.from_numpy(y).to(dev)
    n = xt.shape[0]
    dp = n_primary_cols if n_primary_cols > 0 else int(yt.shape[1])
    fold_of = grouped_fold_of(group_ids, N_OUTER_FOLDS, seed)
    folds = np.array([fold_of[p] for p in group_ids])
    n_folds = int(folds.max()) + 1
    oof = torch.zeros_like(yt)
    chosen: dict[int, float] = {}
    pilot: dict | None = None
    for k in range(n_folds):
        te = np.where(folds == k)[0]
        tr = np.where(folds != k)[0]
        tr_groups = [group_ids[i] for i in tr]
        inner_of = grouped_fold_of(tr_groups, N_INNER_FOLDS, seed * 1000 + k + 1)
        inner = np.array([inner_of[p] for p in tr_groups])
        n_inner = int(inner.max()) + 1
        mse = dict.fromkeys(RIDGE_LAMBDAS, 0.0)
        for j in range(n_inner):
            i_te = tr[inner == j]
            i_tr = tr[inner != j]
            t0 = time.time()
            preds = _ridge_predict(xt[i_tr], yt[i_tr, :dp], xt[i_te], RIDGE_LAMBDAS)
            if pilot is None:
                unit_s = time.time() - t0
                total_units = n_folds * (n_inner + 1)
                pilot = {
                    "unit_s": unit_s,
                    "unit_shape": [int(len(i_tr)), int(xt.shape[1]), int(dp)],
                    "projected_units_per_map": total_units,
                    "projected_s_per_map": unit_s * total_units,
                }
                _log(
                    f"[pilot-fit {label}] unit={unit_s:.1f}s at n_tr={len(i_tr)} "
                    f"d_x={xt.shape[1]} d_y={dp} -> ~{unit_s * total_units:.0f}s/map"
                )
            for lam in RIDGE_LAMBDAS:
                mse[lam] += _pooled_mse(preds[lam], yt[i_te, :dp]) * len(i_te)
        lam_star = min(RIDGE_LAMBDAS, key=lambda z: mse[z])
        chosen[k] = float(lam_star)
        preds = _ridge_predict(xt[tr], yt[tr], xt[te], [lam_star])
        oof[te] = preds[lam_star]
        _log(f"[fit {label}] fold {k + 1}/{n_folds} lambda*={lam_star} n_te={len(te)}")
    oof_np = oof.cpu().numpy()
    del xt, yt, oof
    return {"oof": oof_np, "folds": folds, "lambda_per_fold": chosen, "pilot": pilot, "n": n}


def _r2_per_feature(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, dict]:
    """OOF R^2 per feature (+ pooled) with the ss_tot==0 guard (NaN + count)."""
    mu = y.mean(axis=0, keepdims=True)
    ss_tot = ((y - mu) ** 2).sum(axis=0)
    ss_res = ((y - pred) ** 2).sum(axis=0)
    r2 = np.full(y.shape[1], np.nan, dtype=np.float64)
    ok = ss_tot > 0
    r2[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    pooled = float(1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-30))
    return r2, {"pooled_r2": pooled, "n_ss_tot_zero": int((~ok).sum())}


def _identity_bias_oof(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    inter_x: np.ndarray,
    inter_y: np.ndarray,
) -> dict:
    """Identity+learned-bias baseline on the input/target ACTIVE-set intersection."""
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    if inter_x.size == 0:
        return {"applicable": False, "reason": "empty input/target active-set intersection"}
    xs = x[:, inter_x]
    ys = y[:, inter_y]
    pred = np.zeros_like(ys)
    for k in sorted(set(folds.tolist())):
        te = folds == k
        tr = ~te
        pred[te] = identity_bias_predict(xs[tr], ys[tr], xs[te])
    r2, summary = _r2_per_feature(ys, pred)
    return {
        "applicable": True,
        "n_intersection": int(inter_x.size),
        "pooled_r2": summary["pooled_r2"],
        "note": "evaluated on the input∩target active-feature intersection (plan section 4)",
    }


def _knn_oof(pred: np.ndarray, y: np.ndarray, folds: np.ndarray) -> dict:
    """Per-fold held-out kNN retrieval, k clamped to the pool.

    Metrics: euclidean, cosine, and cosine_centered (v12 taxonomy control (2):
    both sides centered by the TRAIN-fold target mean before the cosine read;
    raw and centered are both reported — disagreement flags the raw number as
    the rogue-dimension artifact).
    """
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    out: dict[str, dict] = {}
    n_skipped_folds = 0
    for metric in ("euclidean", "cosine", "cosine_centered"):
        acc: dict[int, list[tuple[float, int]]] = {}
        chance: dict[int, list[float]] = {}
        for k in sorted(set(folds.tolist())):
            te = folds == k
            n_pool = int(te.sum())
            if n_pool < 2:  # designed skip: retrieval undefined on a 1-row pool
                n_skipped_folds += 1
                continue
            ks = clamp_knn_ks(n_pool)
            if metric == "cosine_centered":
                mu = y[~te].mean(axis=0, keepdims=True)
                res = knn_retrieval(pred[te] - mu, y[te] - mu, ks=ks, metric="cosine")
            else:
                res = knn_retrieval(pred[te], y[te], ks=ks, metric=metric)
            for kk in ks:
                acc.setdefault(kk, []).append((res["acc_at_k"][kk], n_pool))
                chance.setdefault(kk, []).append(res["chance_at_k"][kk])
        out[metric] = {
            str(kk): {
                "acc": float(sum(a * n for a, n in acc[kk]) / sum(n for _, n in acc[kk])),
                "chance": float(np.mean(chance[kk])),
            }
            for kk in acc
        }
        if not acc:
            out[metric] = {"skipped": "every fold pool < 2 rows (retrieval undefined)"}
    if n_skipped_folds:
        out["n_skipped_folds"] = n_skipped_folds // 3  # same folds skipped per metric
    return out


# ---------------------------------------------------------------------------
# Phase B: crossed ANOVA + batched permutation nulls
# ---------------------------------------------------------------------------


def anova_shares(grid: np.ndarray) -> dict[str, np.ndarray]:
    """Balanced two-way shares per feature over a (P, Q, D) grid (one obs/cell)."""
    p, q, _d = grid.shape
    m = grid.mean(axis=(0, 1), keepdims=True)
    mp = grid.mean(axis=1)  # (P, D)
    mq = grid.mean(axis=0)  # (Q, D)
    ss_p = q * ((mp - m[:, 0, :]) ** 2).sum(axis=0)
    ss_q = p * ((mq - m[0, :, :]) ** 2).sum(axis=0)
    ss_tot = ((grid - m) ** 2).sum(axis=(0, 1))
    ss_i = np.maximum(ss_tot - ss_p - ss_q, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "share_prefix": np.where(ss_tot > 0, ss_p / ss_tot, np.nan),
            "share_query": np.where(ss_tot > 0, ss_q / ss_tot, np.nan),
            "share_inter": np.where(ss_tot > 0, ss_i / ss_tot, np.nan),
            "ss_tot": ss_tot,
        }


def permutation_null_shares(
    grid: np.ndarray, axis: str, n_draws: int, seed: int, device: str, chunk: int = 8
) -> np.ndarray:
    """(n_draws, D) permuted-factor shares — batched gathers, no per-feature loop.

    axis='prefix': permute prefix labels WITHIN each query column (destroys the
    prefix + interaction structure, preserves query means and SS_tot);
    axis='query': the transpose.
    """
    import torch

    dev = torch.device(device)
    g = torch.from_numpy(grid).to(dev)
    p, q, d = g.shape
    m = g.mean(dim=(0, 1), keepdim=True)
    ss_tot = ((g - m) ** 2).sum(dim=(0, 1))  # permutation-invariant
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = np.zeros((n_draws, d), dtype=np.float16)
    for start in range(0, n_draws, max(1, chunk)):
        end = min(start + max(1, chunk), n_draws)
        c = end - start
        if axis == "prefix":
            idx = torch.argsort(torch.rand(c, p, q, generator=gen), dim=1).to(dev)
            gp = torch.gather(
                g.unsqueeze(0).expand(c, p, q, d), 1, idx.unsqueeze(-1).expand(c, p, q, d)
            )
            mp = gp.mean(dim=2)  # (c, P, D)
            ss = q * ((mp - m.squeeze(1).unsqueeze(0)) ** 2).sum(dim=1)
        elif axis == "query":
            idx = torch.argsort(torch.rand(c, p, q, generator=gen), dim=2).to(dev)
            gp = torch.gather(
                g.unsqueeze(0).expand(c, p, q, d), 2, idx.unsqueeze(-1).expand(c, p, q, d)
            )
            mq = gp.mean(dim=1)  # (c, Q, D)
            ss = p * ((mq - m.squeeze(0).unsqueeze(0)) ** 2).sum(dim=1)
        else:
            raise ValueError(f"unknown permutation axis {axis!r}")
        share = torch.where(ss_tot > 0, ss / ss_tot, torch.full_like(ss, float("nan")))
        out[start:end] = share.to(torch.float16).cpu().numpy()
        del gp, idx, ss, share
        _log(f"[phase=anova-null axis={axis}] draws {end}/{n_draws}")
    del g
    return out


def perm_pvalues(observed: np.ndarray, null_draws: np.ndarray) -> np.ndarray:
    """Per-feature permutation p = (1 + #draws >= obs) / (B + 1); NaN-safe."""
    b = null_draws.shape[0]
    ge = (null_draws.astype(np.float32) >= observed[None, :].astype(np.float32)).sum(axis=0)
    p = (1.0 + ge) / (b + 1.0)
    p[~np.isfinite(observed)] = np.nan
    return p


def bh_reject(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg at level q (DESCRIPTIVE at B=200 — min p = 1/201)."""
    finite = np.isfinite(pvals)
    reject = np.zeros_like(pvals, dtype=bool)
    ps = np.sort(pvals[finite])
    n = ps.size
    if n == 0:
        return reject
    thresh = ps <= (np.arange(1, n + 1) / n) * q
    if not thresh.any():
        return reject
    cut = ps[np.where(thresh)[0].max()]
    reject[finite] = pvals[finite] <= cut
    return reject


def scaffold_basis(dense_mean: np.ndarray, device: str, rank: int = SCAFFOLD_RANK):
    """Top-`rank` PCA basis (H, r) of the dense per-row mean-answer states.

    Mean-centered; Gram eigh via `_eigh_robust` (cuSOLVER CPU fallback, #1335).
    The realized rank is capped at min(rank, n_rows - 1, H) — smoke slices sit
    far below 48 and the cap is recorded, never a floor. Returns (Q, r).
    """
    import torch

    x = torch.from_numpy(np.asarray(dense_mean, dtype=np.float32)).to(device)
    assert x.ndim == 2 and x.shape[0] >= 2, tuple(x.shape)
    xc = x - x.mean(dim=0, keepdim=True)
    r = int(min(rank, xc.shape[0] - 1, xc.shape[1]))
    w, v = _eigh_robust(xc.T @ xc)  # (H, H); eigenvalues ascending
    del w
    return v[:, -r:].contiguous(), r


_RB_ORDER_VERIFIED = False


def _assert_rb_trait_order() -> None:
    """v22 Minor hardening: RB_TRAIT_ORDER is a POSITIONAL pin on the sorted r_B
    .pt basenames `parent.load_rb_directions` stacks (it logs but does not return
    them) — re-derive the SAME scoped listing at RB_REV and fail loud on drift.

    Memoized: ONE scoped `list_repo_tree` per process (retry_transient-wrapped;
    the listing is lazy, so it is materialized INSIDE the retry — gotchas.md)."""
    global _RB_ORDER_VERIFIED
    if _RB_ORDER_VERIFIED:
        return
    from huggingface_hub import list_repo_tree

    from explore_persona_space.orchestrate import hub

    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: call is materialized inside the hub.retry_transient wrapper
            list_repo_tree(
                DATA_REPO,
                repo_type="dataset",
                path_in_repo="issue779_monitoring/r_b",  # parent.load_rb_directions prefix
                revision=RB_REV,
            )
        ),
        what="r_B trait-order listing",
    )
    names = tuple(
        Path(rel).stem
        for rel in sorted(
            it.path
            for it in entries
            if getattr(it, "size", None) is not None and it.path.endswith(".pt")
        )
    )
    if names != RB_TRAIT_ORDER:
        raise RuntimeError(f"r_B trait basename order drifted: {names} != {RB_TRAIT_ORDER}")
    _RB_ORDER_VERIFIED = True


def rb_cosine_join(
    feats: np.ndarray,
    out_root: Path,
    n_draws: int,
    seed: int,
    device: str,
    dense_mean: np.ndarray,
) -> dict:
    """|cos(W_dec[:, j], r_B[L19, trait])| per active feature + selection-symmetric null.

    Observed statistic = max over the 3 traits; each null draw = max over 3
    RANDOM unit directions (matching the max-over-traits selection per draw).
    v14 SCAFFOLD CONTROL: every read is ALSO reported with the top-48
    answer-PCA scaffold projected out of BOTH r_B and the decoder columns —
    the projected variant is the HEADLINE alignment read, and its null is the
    SAME per-draw max-over-3 statistic recomputed in the projected space.
    """
    import torch

    sae = BatchTopKSAE.load(k=SAE_K, device=device, cache_dir=out_root / "sae_cache")
    w_dec = sae.w_dec[:, torch.from_numpy(feats).to(sae.w_dec.device)]  # (H, d)
    w_hat = w_dec / w_dec.norm(dim=0, keepdim=True).clamp_min(1e-12)
    rb = parent.load_rb_directions(RB_REV, parent.N_LAYERS, parent.N_TRAITS, parent.HIDDEN_DIM)
    _assert_rb_trait_order()  # v22 Minor: pin the positional trait axis to the .pt basenames
    r = torch.from_numpy(rb[SAE_LAYER]).to(w_hat.device, torch.float32)  # (3, H)
    r_hat = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos_traits = (r_hat @ w_hat).abs().cpu().numpy()  # (3, d)
    obs_max = cos_traits.max(axis=0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dirs = torch.randn(n_draws * 3, r.shape[1], generator=gen).to(w_hat.device)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    null = (dirs @ w_hat).abs().reshape(n_draws, 3, -1).max(dim=1).values
    null_np = null.to(torch.float16).cpu().numpy()  # (draws, d)
    null_scale = float(np.nanpercentile(null_np.astype(np.float32), 95))
    assert_nondegenerate("rb_cos_max", float(obs_max.max()), null_scale)
    # v12 taxonomy control (2): CENTERED companion read — subtract the mean
    # active decoder direction, renormalize, same selection-symmetric null.
    w_mu = w_dec.mean(dim=1, keepdim=True)
    wc = w_dec - w_mu
    wc_hat = wc / wc.norm(dim=0, keepdim=True).clamp_min(1e-12)
    cos_traits_c = (r_hat @ wc_hat).abs().cpu().numpy()
    obs_max_c = cos_traits_c.max(axis=0)
    null_c = (dirs @ wc_hat).abs().reshape(n_draws, 3, -1).max(dim=1).values
    null_c_np = null_c.to(torch.float16).cpu().numpy()
    # v14 SCAFFOLD CONTROL: projected read (headline) + same-space null
    q, scaffold_rank = scaffold_basis(dense_mean, str(w_hat.device))
    q = q.to(w_hat.device, torch.float32)
    mass = ((r @ q) ** 2).sum(dim=1) / (r**2).sum(dim=1).clamp_min(1e-12)
    r_proj = r - (r @ q) @ q.T
    r_proj_hat = r_proj / r_proj.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w_proj = w_dec - q @ (q.T @ w_dec)
    w_proj_hat = w_proj / w_proj.norm(dim=0, keepdim=True).clamp_min(1e-12)
    cos_traits_p = (r_proj_hat @ w_proj_hat).abs().cpu().numpy()
    obs_max_p = cos_traits_p.max(axis=0)
    dirs_proj = dirs - (dirs @ q) @ q.T
    dirs_proj = dirs_proj / dirs_proj.norm(dim=1, keepdim=True).clamp_min(1e-12)
    null_p = (dirs_proj @ w_proj_hat).abs().reshape(n_draws, 3, -1).max(dim=1).values
    null_p_np = null_p.to(torch.float16).cpu().numpy()
    null_p_scale = float(np.nanpercentile(null_p_np.astype(np.float32), 95))
    assert_nondegenerate("rb_cos_max_scaffold_proj", float(obs_max_p.max()), null_p_scale)
    mass_by_trait = {t: float(m) for t, m in zip(RB_TRAIT_ORDER, mass.cpu().numpy(), strict=True)}
    _log(
        f"[phase=fits] rb scaffold rank={scaffold_rank} "
        f"mass_frac={ {t: round(v, 3) for t, v in mass_by_trait.items()} } "
        f"obs_max raw={float(obs_max.max()):.3f} proj={float(obs_max_p.max()):.3f} "
        f"null_p95 raw={null_scale:.3f} proj={null_p_scale:.3f}"
    )
    del sae, w_dec, w_hat, wc, wc_hat, dirs, null, null_c, q, r_proj, w_proj, dirs_proj, null_p
    return {
        "cos_traits": cos_traits.astype(np.float16),
        "cos_max": obs_max.astype(np.float32),
        "null_draws_max": null_np,
        "null_p95": null_scale,
        "p_max": perm_pvalues(obs_max, null_np),
        "cos_traits_centered": cos_traits_c.astype(np.float16),
        "cos_max_centered": obs_max_c.astype(np.float32),
        "null_draws_max_centered": null_c_np,
        "p_max_centered": perm_pvalues(obs_max_c, null_c_np),
        "cos_traits_proj": cos_traits_p.astype(np.float16),
        "cos_max_proj": obs_max_p.astype(np.float32),
        "null_draws_max_proj": null_p_np,
        "null_p95_proj": null_p_scale,
        "p_max_proj": perm_pvalues(obs_max_p, null_p_np),
        "scaffold_rank": scaffold_rank,
        "rb_scaffold_mass_frac": mass_by_trait,
    }


# ---------------------------------------------------------------------------
# Phase C: judge (level rubric VERBATIM + speaker_property 5-way rubric,
# separate calls — v13 rubric amendment)
# ---------------------------------------------------------------------------


def assert_rubric_parity() -> dict[str, str]:
    """Level rubric byte-parity with the #1482 reference round (import-not-copy).

    `FC.JUDGE_SYSTEM` must hash to the reference round's recorded value and the
    judge model/max_tokens pins must match; the NEW speaker_property rubric
    records its OWN hash — no parity claim (a stated deviation, one behavior
    per call — llm-judging rule 8).
    """
    base = hashlib.sha256(FC.JUDGE_SYSTEM.encode()).hexdigest()[:16]
    prior = json.loads(PRIOR_ABSTRACTION.read_text())
    want = prior["rubric_sha256_system"]
    assert base == want, f"reference rubric drift: {base} != {want}"
    assert FC.JUDGE_MODEL == prior["judge_model"], "judge model drift"
    assert FC.JUDGE_MAX_TOKENS == prior["max_tokens"], "max_tokens drift"
    speaker = hashlib.sha256(SPEAKER_JUDGE_SYSTEM.encode()).hexdigest()[:16]
    _log(f"[phase=judge] rubric parity OK: level sha16={base} speaker sha16={speaker}")
    return {"level_rubric_sha16": base, "speaker_rubric_sha16": speaker}


def _validate_speaker(res: object) -> dict | None:
    """Drop-never-coerce validator for the speaker_property rubric return.

    An out-of-set `speaker_property` value is a CONTENT drop (returns None) —
    never coerced into a class (v13 amendment; llm-judging rule 9).
    """
    if not isinstance(res, dict) or res.get("error"):
        return None
    sp = res.get("speaker_property")
    if not isinstance(sp, str):
        return None
    norm = sp.strip().lower()
    if norm not in SPEAKER_CLASSES:
        return None
    lab = res.get("label")
    return {"speaker_property": norm, "label": str(lab)[:120] if lab else ""}


def select_judge_sets(
    share_prefix: np.ndarray,
    share_query: np.ndarray,
    r2_ctx: np.ndarray,
    r2_pre: np.ndarray,
    frac_active: np.ndarray,
    mean_act_active: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """Judged union: prefix tail + per-arm R^2 tails + matched + query controls.

    All arrays are aligned to the pooled-mean ACTIVE feature axis (positions,
    not raw feature ids). `ctrl_activity_matched` mirrors tail (i)'s joint
    decile cell counts over (fraction-rows-active, mean pooled activation among
    active rows), seed-0 sampled, other tails excluded.
    """
    d = share_prefix.shape[0]

    def _top(vals: np.ndarray, n: int, largest: bool = True) -> np.ndarray:
        finite = np.where(np.isfinite(vals))[0]
        order = finite[np.argsort(vals[finite])]
        return order[-n:][::-1] if largest else order[:n]

    tail_prefix = _top(share_prefix, min(TAIL_PREFIX_N, d))
    tail_query = _top(share_query, min(CTRL_QUERY_N, d))
    tails_r2 = {
        "r2_ctx_top": _top(r2_ctx, min(TAIL_R2_N, d)),
        "r2_ctx_bottom": _top(r2_ctx, min(TAIL_R2_N, d), largest=False),
        "r2_pre_top": _top(r2_pre, min(TAIL_R2_N, d)),
        "r2_pre_bottom": _top(r2_pre, min(TAIL_R2_N, d), largest=False),
    }
    excluded = set(tail_prefix.tolist()) | set(tail_query.tolist())
    for arr in tails_r2.values():
        excluded |= set(arr.tolist())

    def _decile_bins(vals: np.ndarray) -> np.ndarray:
        edges = np.nanpercentile(vals, np.arange(10, 100, 10))
        return np.searchsorted(edges, vals, side="right")

    bin_a = _decile_bins(frac_active)
    bin_b = _decile_bins(mean_act_active)
    cell_key = bin_a * 10 + bin_b
    rng = np.random.default_rng(seed)
    matched: list[int] = []
    tail_cells, tail_counts = np.unique(cell_key[tail_prefix], return_counts=True)
    scale = CTRL_MATCH_N / max(1, len(tail_prefix))
    for cell_id, cnt in zip(tail_cells, tail_counts, strict=True):
        pool = [
            int(j)
            for j in np.where(cell_key == cell_id)[0]
            if int(j) not in excluded and int(j) not in matched
        ]
        want = max(1, int(round(cnt * scale)))
        if pool:
            take = rng.choice(len(pool), size=min(want, len(pool)), replace=False)
            matched.extend(pool[t] for t in take)
    return {
        "tail_prefix": tail_prefix,
        "ctrl_query_tail": tail_query,
        **tails_r2,
        "ctrl_activity_matched": np.array(sorted(matched), dtype=np.int64),
    }


def build_judge_items(
    union_positions: list[int],
    feats: np.ndarray,
    y_mean: np.ndarray,
    completions: list[str],
) -> list[tuple[str, str, str, str]]:
    """Evidence items mirroring FC._judge_items: top-8 rows by pooled-mean
    activation, 400-char answer snippets, blind to set membership."""
    items = []
    for pos in union_positions:
        fid = int(feats[pos])
        col = y_mean[:, pos]
        top_rows = np.argsort(col)[::-1][: FC.TOP_K_CONTEXTS]
        snippets = [completions[int(i)][: FC.SNIPPET_CHARS] for i in top_rows if col[int(i)] > 0]
        if not snippets:
            continue
        body = "\n\n---\n\n".join(snippets)
        user = (
            f"Feature {fid}. Example answers:\n\n{body}\n\n(No independent auto-interp "
            "description is available for this feature.)\n\nOutput the JSON."
        )
        items.append((f"feat{fid}", f"feature {fid}", body[:200], user))
    return items


def _load_unembedding(cell: str, device: str):
    """(V, H) unembedding via a partial safetensors read (never a full model load).

    Qwen-2.5-7B(-Instruct) does not tie embeddings, so `lm_head.weight` resolves
    from the safetensors index; a tied model falls back to
    `model.embed_tokens.weight`. Returns (tensor fp32 on `device`, key used).
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from explore_persona_space.orchestrate import hub

    model_name, revision, _fmt = CELL_MODEL[cell]
    idx_path = hub.retry_transient(
        lambda: hf_hub_download(model_name, "model.safetensors.index.json", revision=revision),
        what=f"unembedding index ({model_name})",
    )
    wmap = json.loads(Path(idx_path).read_text())["weight_map"]
    key = "lm_head.weight" if "lm_head.weight" in wmap else "model.embed_tokens.weight"
    shard = hub.retry_transient(
        lambda: hf_hub_download(model_name, wmap[key], revision=revision),
        what=f"unembedding shard ({wmap[key]})",
    )
    with safe_open(shard, framework="pt", device="cpu") as f:
        w = f.get_tensor(key)
    return w.to(device=device, dtype=torch.float32), key


def build_evidence_sets(
    judge_sets: dict | None,
    share_prefix: np.ndarray,
    share_query: np.ndarray,
    rb_cos_max: np.ndarray,
    rb_cos_max_proj: np.ndarray,
    d: int,
) -> dict[str, np.ndarray]:
    """Evidence-emission union sets (plan v14 Phase C'): the judge sets when
    present (instruct arm), else the per-cell top prefix-share AND query-share
    tails (BOTH fig_hero_scatter-highlighted mechanical classes — the query
    tail joins the fallback per the v22 Minor), plus the figure-reported
    rb-cos raw/projected tails. All values are POSITIONS on the active axis."""
    ev_sets: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.int64) for k, v in (judge_sets or {}).items()
    }
    if "tail_prefix" not in ev_sets:
        sp = np.nan_to_num(share_prefix, nan=-1.0)
        ev_sets["tail_prefix"] = np.argsort(sp)[::-1][: min(TAIL_PREFIX_N, d)]
    if "ctrl_query_tail" not in ev_sets:
        sq = np.nan_to_num(share_query, nan=-1.0)
        ev_sets["ctrl_query_tail"] = np.argsort(sq)[::-1][: min(CTRL_QUERY_N, d)]
    k_fig = min(RB_COS_FIG_TAIL_N, d)
    ev_sets["rb_cos_tail_raw"] = np.argsort(np.nan_to_num(rb_cos_max, nan=-1.0))[::-1][:k_fig]
    ev_sets["rb_cos_tail_proj"] = np.argsort(np.nan_to_num(rb_cos_max_proj, nan=-1.0))[::-1][:k_fig]
    return ev_sets


def emit_feature_evidence(
    out_root: Path, cell: str, res: dict, judge_sets: dict | None, args, device: str
) -> dict:
    """Phase C' (plan v14 — the leg that must NOT be skipped): per-feature
    evidence artifacts for the #1773 labelling round, per union feature
    (judge 7-set union for the instruct arm, else the top prefix-share +
    query-share tails; + figure-reported rb-cos tails):

      1. top-50 activating tuples (row_id, answer-token offset, activation) —
         one tuple per row (the row's max over sink-excluded answer tokens),
         top rows by that max; batched argsort, no per-feature loop;
      2. the per-row pooled-mean activation vector (n_rows floats/feature);
      3. decoder-space top-10 nearest-neighbour feature ids (ONE matmul over
         all W_dec columns);
      4. top-30 mean-centered logit-footprint tokens (ONE matmul against the
         partially-loaded unembedding; token ids + single-token strings only —
         NO corpus text, per the harmful-content digest discipline).

    Writes `feature_evidence/evidence_{cell}.{json,npz}` (uploaded phase D).
    """
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()
    store: CellStore = res["store"]
    feats = res["feats"]
    rb = res["rb"]
    d = int(feats.size)
    ev_sets = build_evidence_sets(
        judge_sets, res["share_prefix"], res["share_query"], rb["cos_max"], rb["cos_max_proj"], d
    )
    union_pos = np.array(
        sorted({int(p) for arr in ev_sets.values() for p in np.asarray(arr).tolist()}),
        dtype=np.int64,
    )
    raw_ids = feats[union_pos]
    n_u = int(raw_ids.size)
    # (1) top-K activating tuples from per-row max + argmax offsets (batched)
    y_max_u = store.dense("max", raw_ids)
    y_arg_u = store.dense("argmax", raw_ids)
    k_rows = min(EVIDENCE_TOP_ROWS, y_max_u.shape[0])
    top_rows = np.argsort(-y_max_u, axis=0)[:k_rows]  # (K, U) row indices
    top_act = np.take_along_axis(y_max_u, top_rows, axis=0).astype(np.float16)
    top_off = np.take_along_axis(y_arg_u, top_rows, axis=0).astype(np.int32)
    # (2) per-row pooled-mean vectors for the union
    y_mean_u = store.dense("mean", raw_ids).astype(np.float16)
    # (3) decoder-space top-10 neighbours (cosine over ALL columns, one matmul)
    sae = BatchTopKSAE.load(k=SAE_K, device=device, cache_dir=out_root / "sae_cache")
    w_hat = sae.w_dec / sae.w_dec.norm(dim=0, keepdim=True).clamp_min(1e-12)
    cols = torch.from_numpy(raw_ids).to(w_hat.device)
    cos_all = w_hat[:, cols].T @ w_hat  # (U, D)
    cos_all[torch.arange(n_u, device=cos_all.device), cols] = -torch.inf  # exclude self
    nn_cos, nn_ids = torch.topk(cos_all, k=min(EVIDENCE_NN_K, DICT_SIZE - 1), dim=1)
    del cos_all
    # (4) mean-centered logit footprints (one matmul against the unembedding)
    w_u, unembed_key = _load_unembedding(cell, device)
    logits = w_u @ sae.w_dec[:, cols]  # (V, U)
    logits = logits - logits.mean(dim=0, keepdim=True)
    lg_vals, lg_ids = torch.topk(logits, k=EVIDENCE_LOGIT_TOPK, dim=0)
    del logits, w_u, sae
    model_name, revision, _fmt = CELL_MODEL[cell]
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    ev_dir = out_root / "feature_evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    row_ids = np.array(store.row_ids, dtype=object)
    np.savez(
        ev_dir / f"evidence_{cell}.npz",
        union_feature_ids=raw_ids,
        union_positions=union_pos,
        row_ids=row_ids,
        top_row_idx=top_rows.astype(np.int32),
        top_activation=top_act,
        top_answer_token_offset=top_off,
        pooled_mean_rows=y_mean_u,
        nn_feature_ids=nn_ids.cpu().numpy().astype(np.int64),
        nn_cos=nn_cos.cpu().numpy().astype(np.float16),
        logit_top_token_ids=lg_ids.cpu().numpy().astype(np.int64),
        logit_top_values=lg_vals.cpu().numpy().astype(np.float16),
    )
    member_of = {k: set(np.asarray(v).tolist()) for k, v in ev_sets.items()}
    nn_ids_np = nn_ids.cpu().numpy()
    nn_cos_np = nn_cos.cpu().numpy()
    lg_ids_np = lg_ids.cpu().numpy()
    lg_vals_np = lg_vals.cpu().numpy()
    features = []
    for j, pos in enumerate(union_pos.tolist()):  # JSON assembly only — compute is batched above
        features.append(
            {
                "feature_id": int(raw_ids[j]),
                "sets": sorted(k for k, s in member_of.items() if pos in s),
                "top_rows": [
                    {
                        "row_id": str(row_ids[int(top_rows[i, j])]),
                        "answer_token_offset": int(top_off[i, j]),
                        "activation": float(top_act[i, j]),
                    }
                    for i in range(k_rows)
                    if float(top_act[i, j]) > 0
                ],
                "decoder_nn": [
                    {"feature_id": int(a), "cos": float(b)}
                    for a, b in zip(nn_ids_np[j], nn_cos_np[j], strict=True)
                ],
                "logit_top_tokens": [
                    {
                        "token_id": int(a),
                        "token": tokenizer.convert_ids_to_tokens(int(a)),
                        "logit_centered": float(b),
                    }
                    for a, b in zip(lg_ids_np[:, j], lg_vals_np[:, j], strict=True)
                ],
            }
        )
    meta = {
        "cell": cell,
        "n_union": n_u,
        "sets": {
            k: [int(feats[int(p)]) for p in np.asarray(v).tolist()] for k, v in ev_sets.items()
        },
        "top_tuple_semantics": (
            "one tuple per row: (row_id, answer-token offset of the row's max activation "
            "over sink-excluded answer tokens, that max activation); top-50 rows per "
            "feature by that max — offsets index the answer span under the capture "
            "tokenization (issue1092_gpu_phase render)"
        ),
        "unembedding_key": unembed_key,
        "logit_footprint": (
            "W_dec column @ unembedding, mean-centered over vocab; token ids + "
            "single-token strings only (no corpus text)"
        ),
        "judged_axes": LABEL_FREEZE_NOTE,
        "repro": _repro_meta(),
    }
    (ev_dir / f"evidence_{cell}.json").write_text(
        json.dumps({"meta": meta, "features": features}, indent=1)
    )
    _log(
        f"[phase=evidence cell={cell}] union={n_u} features "
        f"({ {k: int(np.asarray(v).size) for k, v in ev_sets.items()} }) -> "
        f"evidence_{cell}.{{json,npz}} in {time.time() - t0:.0f}s"
    )
    return {
        "n_union": n_u,
        "set_sizes": {k: int(np.asarray(v).size) for k, v in ev_sets.items()},
        "unembedding_key": unembed_key,
    }


def run_judge_phase(
    sets: dict[str, np.ndarray],
    feats: np.ndarray,
    y_mean: np.ndarray,
    completions: list[str],
    work: Path,
    judge_limit: int,
    retest_n: int,
) -> dict:
    """ONE blind dispatch per rubric over the shuffled union + retest; drop split.

    Item order shuffled by feature-id hash; set membership NEVER in the prompt.
    Transport failures ride dispatch_judge_items' retry machinery and are
    counted separately from content drops (llm-judging rules 9/24).
    """
    import issue1482_analysis as A
    from explore_persona_space.eval.batch_judge import is_transport_error_dict
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items

    hashes = assert_rubric_parity()
    union = sorted(
        {int(p) for arr in sets.values() for p in arr.tolist()},
        key=lambda p: hashlib.sha256(str(int(feats[p])).encode()).hexdigest(),
    )
    if judge_limit > 0:
        union = union[:judge_limit]
    items = build_judge_items(union, feats, y_mean, completions)
    _log(f"[phase=judge] union={len(union)} features -> {len(items)} evidence items")

    def _dispatch(tag: str, system_prompt: str, its):
        return dispatch_judge_items(
            its,
            judge_model=FC.JUDGE_MODEL,
            judge_system_prompt=system_prompt,
            max_tokens=FC.JUDGE_MAX_TOKENS,
            checkpoint_dir=work / f"judge_dispatch_{tag}",
            error_dict_factory=lambda reason: {"error": True, "reason": reason},
        )

    def _collect(results: dict, validate) -> tuple[dict, dict]:
        labels: dict[str, dict] = {}
        drops = {"content": 0, "transport": 0}
        for cid, res in results.items():
            if isinstance(res, dict) and res.get("error"):
                drops["transport" if is_transport_error_dict(res) else "content"] += 1
                continue
            lab = validate(res)
            if lab is None:
                drops["content"] += 1
                continue
            reason = res.get("reasoning") if isinstance(res, dict) else None
            labels[cid.removeprefix("feat")] = {
                **lab,
                "reasoning": str(reason)[:400] if reason else "",
            }
        return labels, drops

    out: dict = {"rubric_hashes": hashes, "judge_model": FC.JUDGE_MODEL}
    rubrics = {
        "level": (FC.JUDGE_SYSTEM, FC._validate_level, "level"),
        "speaker": (SPEAKER_JUDGE_SYSTEM, _validate_speaker, "speaker_property"),
    }
    rng = np.random.default_rng(FC.SAMPLE_SEED)
    rt_pick = rng.choice(len(items), size=min(retest_n, len(items)), replace=False)
    for name, (system_prompt, validate, field) in rubrics.items():
        labels, drops = _collect(_dispatch(f"{name}_main", system_prompt, items), validate)
        rt_items = [(f"rt_{items[i][0]}", *items[i][1:]) for i in rt_pick]
        rt_labels, rt_drops = _collect(
            _dispatch(f"{name}_retest", system_prompt, rt_items), validate
        )
        a, b = [], []
        for i in rt_pick:
            fid = items[i][0].removeprefix("feat")
            l1 = labels.get(fid)
            l2 = rt_labels.get(f"rt_{items[i][0]}".removeprefix("feat"))
            if l2 is None:
                l2 = rt_labels.get(f"rt_feat{fid}")
            if l1 is not None and l2 is not None:
                a.append(str(l1[field]))
                b.append(str(l2[field]))
        out[name] = {
            "labels": labels,
            "drops": drops,
            "retest": {"n": len(a), "kappa": A._cohens_kappa(a, b), "drops": rt_drops},
        }
        _log(
            f"[phase=judge rubric={name}] {len(labels)}/{len(items)} labeled "
            f"drops={drops} kappa={out[name]['retest']['kappa']:.3f} (n={len(a)})"
        )
    out["sets"] = {k: [int(feats[p]) for p in v.tolist()] for k, v in sets.items()}
    out["union_feature_ids"] = [int(feats[p]) for p in union]
    out["max_tokens"] = FC.JUDGE_MAX_TOKENS
    out["temperature"] = "API default"
    out["n_draws_per_item"] = 1
    return out


# ---------------------------------------------------------------------------
# PART5: orchestration + main
# ---------------------------------------------------------------------------

DENSE_LATENT_PCTL = 90.0  # top-activity-decile flag (lit-review a74c9d54 section 2 item 3)
FOUR_OBJECT_R2_TAU = 0.1  # descriptive per-feature predictability threshold (stated)


def _repro_meta() -> dict:
    """Reproducibility metadata for every result JSON (CLAUDE.md requirement)."""
    import torch
    import transformers

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ},
            cwd=str(PROJECT_ROOT),
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — metadata best-effort on clone-less workers
        commit = "unknown"
    return {
        "git_commit": commit,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_rev": CORPUS_REV,
        "rb_rev": RB_REV,
        "sae": {"k": SAE_K, "layer": SAE_LAYER},
    }


def _upload_tree(out_root: Path, sub: str, args, label: str) -> None:
    """One `upload_folder` commit for a phase output tree (fail-loud, retried)."""
    from explore_persona_space.orchestrate import hub

    src = out_root / sub
    dest = f"{HF_PREFIX}/{args.hf_subdir}/{sub}"
    url = hub._upload(src, DATA_REPO, "dataset", dest, raise_on_error=True)
    _log(f"[phase=upload {label}] {src} -> {dest} ({url})")


def _load_bare_arrays(
    out_root: Path, cell: str
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Bare-query captures -> (query_ids, feature matrix (Q, d_bare_active),
    dense residual states (Q, 3584) fp32, active raw feature ids)."""
    z = np.load(out_root / "features" / cell / "bare_query.npz", allow_pickle=True)
    qids = [str(q) for q in z["query_ids"].tolist()]
    indptr, idx, val = z["indptr"], z["idx"], z["val"]
    active = np.unique(idx.astype(np.int64)) if idx.size else np.zeros(0, np.int64)
    if active.size == 0:
        raise RuntimeError(f"bare-query captures for {cell} have no active features")
    col_of = np.full(DICT_SIZE, -1, dtype=np.int64)
    col_of[active] = np.arange(active.size)
    feat = np.zeros((len(qids), active.size), dtype=np.float32)
    rows = np.repeat(np.arange(len(qids)), np.diff(indptr))
    feat[rows, col_of[idx.astype(np.int64)]] = val.astype(np.float32)
    dense = z["dense"].astype(np.float32)
    return qids, feat, dense, active


def _map_reads(
    label: str,
    fit: dict,
    y_all3: np.ndarray,
    d_act: int,
    x_feats: np.ndarray | None,
    y_feats: np.ndarray,
    x: np.ndarray | None,
) -> dict:
    """Per-map read bundle: per-feature/pooled R^2 (mean slice + exploratory),
    identity+bias (active-set intersection; inapplicable on a dim mismatch),
    kNN retrieval (euclid + cosine + centered cosine)."""
    y_mean = y_all3[:, :d_act]
    oof_mean = fit["oof"][:, :d_act]
    r2, summary = _r2_per_feature(y_mean, oof_mean)
    expl = {}
    for i, name in enumerate(("max", "frac")):
        sl = slice(d_act * (i + 1), d_act * (i + 2))
        _, s = _r2_per_feature(y_all3[:, sl], fit["oof"][:, sl])
        expl[name] = s["pooled_r2"]
    if x_feats is None or x is None:
        idb = {"applicable": False, "reason": "input dim != target active-feature dim"}
    else:
        common, ix, iy = np.intersect1d(x_feats, y_feats, return_indices=True)
        idb = _identity_bias_oof(x, y_mean, fit["folds"], ix, iy)
        idb["n_common_features"] = int(common.size)
    return {
        "label": label,
        "pooled_r2_mean": summary["pooled_r2"],
        "n_ss_tot_zero": summary["n_ss_tot_zero"],
        "pooled_r2_exploratory": expl,
        "identity_bias": idb,
        "knn": _knn_oof(oof_mean, y_mean, fit["folds"]),
        "lambda_per_fold": fit["lambda_per_fold"],
        "pilot": fit["pilot"],
        "n_rows": fit["n"],
        "r2_per_feature": r2,
    }


def _averaged_reads(
    x_ce: np.ndarray,
    y_mean: np.ndarray,
    prefix_ids: list[str],
    ctx_fit: dict,
    feats_ce: np.ndarray,
    feats: np.ndarray,
    seed: int,
    device: str,
) -> tuple[dict, dict]:
    """Map (c): induced averaged read (PRIMARY) + independently-fit averaged
    map (SECONDARY, n=99 << d — explicitly under-determined, caveat in output)."""
    import torch

    uniq = sorted(set(prefix_ids))
    p_of = {p: i for i, p in enumerate(uniq)}
    pmat = np.zeros((len(uniq), len(prefix_ids)), dtype=np.float32)
    for i, p in enumerate(prefix_ids):
        pmat[p_of[p], i] = 1.0
    pmat /= np.maximum(pmat.sum(axis=1, keepdims=True), 1.0)
    f_bar = pmat @ x_ce
    y_bar = pmat @ y_mean
    folds = ctx_fit["folds"]
    prefix_fold = {p: int(folds[i]) for i, p in enumerate(prefix_ids)}
    pf = np.array([prefix_fold[p] for p in uniq])
    dev = torch.device(device)
    xt = torch.from_numpy(x_ce).to(dev)
    yt = torch.from_numpy(y_mean).to(dev)
    fb = torch.from_numpy(f_bar).to(dev)
    pred = np.zeros_like(y_bar)
    for k in sorted(set(pf.tolist())):
        tr = np.where(folds != k)[0]
        ps = np.where(pf == k)[0]
        lam = ctx_fit["lambda_per_fold"][k]
        preds = _ridge_predict(xt[tr], yt[tr], fb[ps], [lam])
        pred[ps] = preds[lam].cpu().numpy()
    r2, summary = _r2_per_feature(y_bar, pred)
    common, ix, iy = np.intersect1d(feats_ce, feats, return_indices=True)
    induced = {
        "label": "induced_averaged (PRIMARY averaged read)",
        "pooled_r2_mean": summary["pooled_r2"],
        "n_prefixes": len(uniq),
        "identity_bias": _identity_bias_oof(f_bar, y_bar, pf, ix, iy),
        "knn": _knn_oof(pred, y_bar, pf),
        "r2_per_feature": r2,
    }
    del xt, yt, fb
    indep_fit = fit_map_oof(f_bar, y_bar, uniq, seed, device, "avg_indep")
    r2i, s_i = _r2_per_feature(y_bar, indep_fit["oof"])
    indep = {
        "label": "independently_fit_averaged (SECONDARY/diagnostic)",
        "caveat": (
            f"n_train ~= {max(1, len(uniq) - len(uniq) // N_OUTER_FOLDS)} prefixes per fold "
            f"<< d = {y_mean.shape[1]} — an explicitly under-determined regime; "
            "SECONDARY/diagnostic only (plan section 4 map (c))"
        ),
        "pooled_r2_mean": s_i["pooled_r2"],
        "lambda_per_fold": indep_fit["lambda_per_fold"],
        "knn": _knn_oof(indep_fit["oof"], y_bar, indep_fit["folds"]),
    }
    return induced, indep


def _anova_block(store: CellStore, y: np.ndarray, args, device: str, tag: str) -> dict:
    """Balanced-subgrid crossed ANOVA + batched permutation nulls + selection
    stats (per-draw max share AND per-draw top-100-mean — selection re-run per
    draw, `.claude/rules/selection-symmetric-nulls.md`)."""
    pairs = set(zip(store.prefix_ids, store.query_ids, strict=True))
    prefixes = sorted(set(store.prefix_ids))
    queries = sorted(set(store.query_ids))
    kept_p, kept_q, dropped = complete_subgrid(pairs, prefixes, queries)
    p_of = {p: i for i, p in enumerate(kept_p)}
    q_of = {q: i for i, q in enumerate(kept_q)}
    d = y.shape[1]
    grid = np.full((len(kept_p), len(kept_q), d), np.nan, dtype=np.float32)
    for i, (p, q) in enumerate(zip(store.prefix_ids, store.query_ids, strict=True)):
        if p in p_of and q in q_of:
            grid[p_of[p], q_of[q]] = y[i]
    if not np.isfinite(grid).all():
        raise RuntimeError(f"ANOVA grid ({tag}) has holes after complete_subgrid")
    shares = anova_shares(grid)
    out: dict = {
        "shares": shares,
        "kept_grid": [len(kept_p), len(kept_q)],
        "dropped_axis_elems": dropped,
        "nulls": {},
    }
    top_k = min(100, d)
    # adaptive draw chunk: keep the gathered (chunk, P, Q, d) fp32 block <= ~2.5 GB
    chunk = max(1, min(8, int(2.5e9 / max(1, len(kept_p) * len(kept_q) * d * 4))))
    for axis, share_key in (("prefix", "share_prefix"), ("query", "share_query")):
        null = permutation_null_shares(grid, axis, args.null_draws, args.seed, device, chunk=chunk)
        obs = shares[share_key]
        null32 = null.astype(np.float32)
        draw_max = np.nanmax(null32, axis=1)
        draw_sorted = np.sort(np.where(np.isfinite(null32), null32, -np.inf), axis=1)
        draw_topk_mean = draw_sorted[:, -top_k:].mean(axis=1)
        obs_finite = obs[np.isfinite(obs)]
        obs_max = float(obs_finite.max()) if obs_finite.size else float("nan")
        obs_topk = float(np.sort(obs_finite)[-top_k:].mean()) if obs_finite.size else float("nan")
        assert_nondegenerate(
            f"share_{axis}_max ({tag})", obs_max, float(np.nanpercentile(draw_max, 95))
        )
        out["nulls"][axis] = {
            "per_feature_p": perm_pvalues(obs, null),
            "null_draws": null,  # (draws, d) fp16 — persisted per-draw x per-feature matrix
            "draw_max": draw_max,
            "draw_topk_mean": draw_topk_mean,
            "obs_max": obs_max,
            "obs_topk_mean": obs_topk,
            "p_selection_max": float((1.0 + (draw_max >= obs_max).sum()) / (args.null_draws + 1.0)),
            "p_selection_topk": float(
                (1.0 + (draw_topk_mean >= obs_topk).sum()) / (args.null_draws + 1.0)
            ),
            "top_k": top_k,
        }
    out["bh_reject_prefix_q05"] = bh_reject(out["nulls"]["prefix"]["per_feature_p"])
    return out


def run_phase_b_cell(out_root: Path, cell: str, args, devices: list[str]) -> dict:
    """Phase B for one cell: maps (a)/(b)/(d)+dense companion (threaded across
    GPUs), averaged reads (c), crossed ANOVA + nulls (primary sink-excluded +
    secondary all-token robustness), r_B cosine join, characterization join."""
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    store = CellStore(out_root, cell)
    feats = store.active_set("mean", ACTIVITY_FLOOR_FRAC)
    d_act = int(feats.size)
    y_mean = store.dense("mean", feats)
    y_all3 = np.concatenate([y_mean, store.dense("max", feats), store.dense("frac", feats)], axis=1)
    feats_ce = store.active_set("ce", ACTIVITY_FLOOR_FRAC)
    feats_pe = store.active_set("pe", ACTIVITY_FLOOR_FRAC)
    x_ce = store.dense("ce", feats_ce)
    x_pe = store.dense("pe", feats_pe)
    bq_ids, bq_feat, bq_dense, bare_feats = _load_bare_arrays(out_root, cell)
    q_index = {q: i for i, q in enumerate(bq_ids)}
    row_q = np.array([q_index[q] for q in store.query_ids])
    x_bare = bq_feat[row_q]
    x_bare_dense = bq_dense[row_q]
    _log(
        f"[phase=fits cell={cell}] n={store.n_rows} d_act={d_act} "
        f"d_ce={feats_ce.size} d_pe={feats_pe.size} d_bare={bare_feats.size}"
    )
    jobs = [
        ("ctx", x_ce, store.prefix_ids),
        ("pre", x_pe, store.prefix_ids),
        ("bare", x_bare, store.query_ids),  # folds grouped BY QUERY (plan map (d))
        ("bare_dense", x_bare_dense, store.query_ids),
    ]
    fits: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(devices))) as ex:
        futs = {
            name: ex.submit(
                fit_map_oof,
                x,
                y_all3,
                groups,
                args.seed,
                devices[i % len(devices)],
                f"{cell}/{name}",
                d_act,
            )
            for i, (name, x, groups) in enumerate(jobs)
        }
        for name, fut in futs.items():
            fits[name] = fut.result()
    reads = {
        "ctx": _map_reads("context_end", fits["ctx"], y_all3, d_act, feats_ce, feats, x_ce),
        "pre": _map_reads("prefix_end", fits["pre"], y_all3, d_act, feats_pe, feats, x_pe),
        "bare": _map_reads(
            "bare_query (folds grouped BY QUERY)",
            fits["bare"],
            y_all3,
            d_act,
            bare_feats,
            feats,
            x_bare,
        ),
        "bare_dense": _map_reads(
            "bare_query_dense_3584d (folds grouped BY QUERY)",
            fits["bare_dense"],
            y_all3,
            d_act,
            None,
            feats,
            None,
        ),
    }
    induced, indep = _averaged_reads(
        x_ce, y_mean, store.prefix_ids, fits["ctx"], feats_ce, feats, args.seed, devices[0]
    )
    anova = _anova_block(store, y_mean, args, devices[0], "primary_sink_excluded")
    y_all_tok = store.dense("mean_all", feats)
    anova_all = _anova_block(store, y_all_tok, args, devices[0], "secondary_all_token")
    # sink-robustness report: tail overlap + share correlation, primary vs all-token
    sp1 = anova["shares"]["share_prefix"]
    sp2 = anova_all["shares"]["share_prefix"]
    k = min(100, d_act)
    top1 = set(np.argsort(np.nan_to_num(sp1, nan=-1))[-k:].tolist())
    top2 = set(np.argsort(np.nan_to_num(sp2, nan=-1))[-k:].tolist())
    both = np.isfinite(sp1) & np.isfinite(sp2)
    sink_robustness = {
        "tail_overlap_at_100": len(top1 & top2) / max(1, k),
        "share_prefix_pearson": (
            float(np.corrcoef(sp1[both], sp2[both])[0, 1]) if both.sum() > 2 else None
        ),
        "n_rows_pe_sink_flagged": int(store.pe_sink.sum()),
        "n_rows_ce_sink_flagged": int(store.ce_sink.sum()),
        "mean_sink_answer_tokens": float(store.n_sink_answer.mean()),
        "note": (
            "PRIMARY pooling excludes 10x-median-norm sink answer tokens; SECONDARY is "
            "the all-token mean (v12 taxonomy control (1): headline must survive)"
        ),
    }
    rb = rb_cosine_join(
        feats, out_root, args.randdir_draws, args.seed, devices[0], store.dense_mean
    )
    # characterization covariates
    act = y_mean > 0
    frac_active = act.mean(axis=0)
    n_act = np.maximum(act.sum(axis=0), 1)
    mean_act_active = y_mean.sum(axis=0) / n_act
    frac_mat = y_all3[:, 2 * d_act :]
    within_ans = np.where(act.sum(0) > 0, (frac_mat * act).sum(0) / n_act, np.nan)
    ce_act = store.dense("ce", feats) > 0
    uniq_p = sorted(set(store.prefix_ids))
    pmat = np.zeros((len(uniq_p), store.n_rows), dtype=np.float32)
    for i, p in enumerate(store.prefix_ids):
        pmat[uniq_p.index(p), i] = 1.0  # noqa: PLR1736
    pmat /= np.maximum(pmat.sum(axis=1, keepdims=True), 1.0)
    cq = pmat @ ce_act.astype(np.float32)  # (P, d): per-prefix fraction of queries active
    dense_latent = frac_active >= np.nanpercentile(frac_active, DENSE_LATENT_PCTL)
    out_dir = out_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"anova_shares_{cell}.npz",
        feats=feats,
        share_prefix=anova["shares"]["share_prefix"],
        share_query=anova["shares"]["share_query"],
        share_inter=anova["shares"]["share_inter"],
        ss_tot=anova["shares"]["ss_tot"],
        p_prefix=anova["nulls"]["prefix"]["per_feature_p"],
        p_query=anova["nulls"]["query"]["per_feature_p"],
        null_prefix_draws=anova["nulls"]["prefix"]["null_draws"],
        null_query_draws=anova["nulls"]["query"]["null_draws"],
        null_prefix_draw_max=anova["nulls"]["prefix"]["draw_max"],
        null_prefix_draw_top100_mean=anova["nulls"]["prefix"]["draw_topk_mean"],
        bh_reject_prefix_q05=anova["bh_reject_prefix_q05"],
        share_prefix_all_token=anova_all["shares"]["share_prefix"],
        share_query_all_token=anova_all["shares"]["share_query"],
    )
    np.savez(
        out_dir / f"perfeature_join_{cell}.npz",
        feats=feats,
        share_prefix=anova["shares"]["share_prefix"],
        share_query=anova["shares"]["share_query"],
        share_inter=anova["shares"]["share_inter"],
        p_prefix=anova["nulls"]["prefix"]["per_feature_p"],
        r2_ctx=reads["ctx"]["r2_per_feature"],
        r2_pre=reads["pre"]["r2_per_feature"],
        r2_bare=reads["bare"]["r2_per_feature"],
        r2_bare_dense=reads["bare_dense"]["r2_per_feature"],
        frac_active=frac_active,
        mean_act_active=mean_act_active,
        within_answer_consistency=within_ans,
        cross_query_consistency_mean=cq.mean(axis=0),
        cross_query_consistency_max=cq.max(axis=0),
        rb_cos_traits=rb["cos_traits"],
        rb_cos_max=rb["cos_max"],
        rb_p_max=rb["p_max"],
        rb_cos_traits_centered=rb["cos_traits_centered"],
        rb_cos_max_centered=rb["cos_max_centered"],
        rb_p_max_centered=rb["p_max_centered"],
        rb_null_draws_max=rb["null_draws_max"],
        rb_null_draws_max_centered=rb["null_draws_max_centered"],
        rb_cos_traits_proj=rb["cos_traits_proj"],
        rb_cos_max_proj=rb["cos_max_proj"],
        rb_p_max_proj=rb["p_max_proj"],
        rb_null_draws_max_proj=rb["null_draws_max_proj"],
        rb_scaffold_rank=np.int64(rb["scaffold_rank"]),
        rb_scaffold_mass_frac=np.array(
            [rb["rb_scaffold_mass_frac"][t] for t in RB_TRAIT_ORDER], dtype=np.float32
        ),
        dense_latent=dense_latent,
    )
    completions_by_row = {}
    for rec in read_rows_file(out_root / "work" / f"rows_{cell}.jsonl"):
        completions_by_row[rec["row_id"]] = rec["completion"]
    completions = [completions_by_row[rid] for rid in store.row_ids]
    _log(f"[phase=fits cell={cell}] phase B done in {time.time() - t0:.0f}s")
    return {
        "store": store,
        "feats": feats,
        "d_act": d_act,
        "y_mean": y_mean,
        "share_prefix": anova["shares"]["share_prefix"],
        "share_query": anova["shares"]["share_query"],
        "r2_ctx": reads["ctx"]["r2_per_feature"],
        "r2_pre": reads["pre"]["r2_per_feature"],
        "r2_bare": reads["bare"]["r2_per_feature"],
        "frac_active": frac_active,
        "mean_act_active": mean_act_active,
        "dense_latent": dense_latent,
        "completions": completions,
        "reads": reads,
        "induced": induced,
        "indep_averaged": indep,
        "anova": anova,
        "sink_robustness": sink_robustness,
        "rb": rb,
    }


def _judge_worker(box: dict, sets, feats, y_mean, completions, work, judge_limit, retest_n):
    """Thread target for phase C (concurrent with the remaining phase B)."""
    try:
        box["result"] = run_judge_phase(
            sets, feats, y_mean, completions, work, judge_limit, retest_n
        )
    except Exception as e:  # noqa: BLE001 — re-raised at join by the orchestrator
        box["error"] = e


def _delta_block(judge_out: dict, sets: dict, feats: np.ndarray, dense_latent, args) -> dict:
    """Headline Delta on the identity_disposition SUBSET only (v13 amendment)
    (+ dense-latent-excluded variant + per-class speaker_property rates per set
    — language and register_style reported separately, NEVER pooled)."""
    labels = judge_out["speaker"]["labels"]

    def _flags(pos_arr: np.ndarray, exclude_dense: bool = False) -> np.ndarray:
        vals = []
        for p in pos_arr.tolist():
            if exclude_dense and bool(dense_latent[int(p)]):
                continue
            lab = labels.get(str(int(feats[int(p)])))
            if lab is not None:
                vals.append(1.0 if lab["speaker_property"] == "identity_disposition" else 0.0)
        return np.array(vals, dtype=np.float64)

    def _class_rates(pos_arr: np.ndarray) -> dict:
        cls = [
            labels[str(int(feats[int(p)]))]["speaker_property"]
            for p in pos_arr.tolist()
            if str(int(feats[int(p)])) in labels
        ]
        n = len(cls)
        return {
            "n_labeled": n,
            **{c: (cls.count(c) / n if n else None) for c in SPEAKER_CLASSES},
        }

    tail = _flags(sets["tail_prefix"])
    ctrl = _flags(sets["ctrl_activity_matched"])
    qtail = _flags(sets["ctrl_query_tail"])
    out = {
        "headline_class": "identity_disposition",
        "delta": delta_bootstrap(tail, ctrl, args.bootstrap_draws, args.seed),
        "delta_dense_latent_excluded": delta_bootstrap(
            _flags(sets["tail_prefix"], exclude_dense=True),
            _flags(sets["ctrl_activity_matched"], exclude_dense=True),
            args.bootstrap_draws,
            args.seed + 1,
        ),
        "rate_query_tail_identity": float(qtail.mean()) if qtail.size else None,
        "per_class_rates": {
            name: _class_rates(sets[name])
            for name in ("tail_prefix", "ctrl_activity_matched", "ctrl_query_tail")
        },
        "n_labeled": {"tail": int(tail.size), "ctrl": int(ctrl.size), "qtail": int(qtail.size)},
    }
    return out


def _template_block(out_root: Path, kept: list[str], results: dict) -> dict:
    """v12 taxonomy control (3): template-only condition + base-vs-instruct DiD.

    Operationalization: per-cell template-only forward yields the set of
    TEMPLATE-ACTIVE features; every base-vs-instruct scalar contrast is
    reported raw AND with template-active features excluded per cell (the
    difference-in-differences at the feature-set level). Skipped (with note)
    when the base arm was gate-dropped and no contrast is reported.
    """
    block: dict = {"per_cell": {}}
    for cell in kept:
        z = np.load(out_root / "features" / cell / "template_control.npz", allow_pickle=True)
        block["per_cell"][cell] = {
            "n_template_active_pooled": int(z["pooled_mean_idx"].size),
            "n_template_active_last": int(z["last_idx"].size),
            "n_tokens": int(z["n_tokens"][0]),
            "template_active_ids": z["pooled_mean_idx"].astype(int).tolist(),
        }
    if len(kept) < 2:
        block["did"] = {
            "skipped": "base arm gate-dropped or not requested; no base-vs-instruct "
            "contrast reported (plan v12 taxonomy control (3))"
        }
        return block

    def _scalars(cell: str, exclude_template: bool) -> dict:
        res = results[cell]
        feats = res["feats"]
        mask = np.ones(feats.size, dtype=bool)
        if exclude_template:
            tpl = np.array(block["per_cell"][cell]["template_active_ids"], dtype=np.int64)
            mask = ~np.isin(feats, tpl)
        sp = res["share_prefix"][mask]
        sp = sp[np.isfinite(sp)]
        r2 = res["r2_ctx"][mask]
        r2 = r2[np.isfinite(r2)]
        return {
            "n_features": int(mask.sum()),
            "mean_share_prefix": float(sp.mean()) if sp.size else None,
            "mean_r2_ctx": float(r2.mean()) if r2.size else None,
        }

    raw = {c: _scalars(c, False) for c in kept}
    adj = {c: _scalars(c, True) for c in kept}
    a, b = kept[0], kept[1]

    def _diff(d1, d2, key):
        if d1[key] is None or d2[key] is None:
            return None
        return d1[key] - d2[key]

    block["did"] = {
        "cells": [a, b],
        "raw_contrast": {k: _diff(raw[a], raw[b], k) for k in ("mean_share_prefix", "mean_r2_ctx")},
        "template_excluded_contrast": {
            k: _diff(adj[a], adj[b], k) for k in ("mean_share_prefix", "mean_r2_ctx")
        },
        "per_cell_raw": raw,
        "per_cell_template_excluded": adj,
    }
    return block


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    assert_label_freeze(args.judged_labels, args.override_label_freeze)
    if args.import_check:
        run_import_check()
        return 0
    if args.gate_probes:
        run_gate_probes()
        return 0
    if args.worker_capture:
        return worker_capture(args)
    if args.worker_sinkmap:
        return worker_sinkmap(args)

    import torch

    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    t_run0 = time.time()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "out").mkdir(exist_ok=True)
    free_gb = assert_out_root_headroom(out_root, args.need_gb, phase="stage")
    _log(f"[phase=stage] out_root={out_root} free={free_gb:.0f}GB cells={args.cells}")
    stage = stage_inputs(out_root)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    devices = ["cpu"]
    fit_device = "cpu"
    if args.device != "cpu" and torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        fit_device = "cuda:0"
    _log(f"[phase=stage] fit devices={devices}")

    # ---- Phase A: rows -> capture fan-out -> fitness gate (per cell) ----
    gate_records: dict[str, dict] = {}
    metas: dict[str, dict] = {}
    for cell in cells:
        rows_path, meta = build_rows_file(out_root, stage, cell, args)
        run_sinkmap_subprocess(out_root, cell, rows_path, args)  # v13: map BEFORE capture
        run_capture_fanout(out_root, cell, rows_path, args)
        shard_dir = out_root / "features" / cell
        n_dropped = sum(
            json.loads(p.read_text())["n_dropped"] for p in shard_dir.glob("shard*_done.json")
        )
        check_k2(meta["n_expected"], meta["n_missing_completion"] + n_dropped, cell)
        meta["n_dropped_capture"] = int(n_dropped)
        metas[cell] = meta
        gate_records[cell] = run_fitness_gate(out_root, cell, fit_device)
    fitness_out = out_root / "out" / "sae_fitness_1092.json"
    fitness_out.write_text(
        json.dumps({"gates": gate_records, "row_meta": metas, "repro": _repro_meta()}, indent=1)
    )
    # phase-A upload BEFORE any fit consumes the stores (#825 ordering)
    _upload_tree(out_root, "features", args, "phaseA-stores")
    _upload_tree(out_root, "sink_map", args, "phaseA-sinkmap")  # v13 committed deliverable
    _upload_tree(out_root, "out", args, "phaseA-fitness")
    if "cell_inst_own" in gate_records and not gate_records["cell_inst_own"]["pass"]:
        _log("[phase=gate] K1 HALT: instruct-arm SAE fitness gate FAILED (values uploaded)")
        return K1_HALT_RC
    kept = [c for c in cells if gate_records[c]["pass"]]
    for c in cells:
        if c not in kept:
            _log(f"[phase=gate] arm {c} DROPPED by fitness gate (reported, not a kill)")

    # ---- Phase B (+ phase C judge overlapping) ----
    results: dict[str, dict] = {}
    judge_box: dict = {}
    judge_thread: threading.Thread | None = None
    judge_sets: dict | None = None
    for cell in kept:
        res = run_phase_b_cell(out_root, cell, args, devices)
        results[cell] = res
        if cell == "cell_inst_own":
            # The 4-set union is ALWAYS computed — it feeds phase C' evidence
            # emission even under the v14 JUDGED-LABEL FREEZE.
            judge_sets = select_judge_sets(
                res["share_prefix"],
                res["share_query"],
                res["r2_ctx"],
                res["r2_pre"],
                res["frac_active"],
                res["mean_act_active"],
                args.seed,
            )
            if args.judged_labels == "on" and not args.skip_judge:
                judge_thread = threading.Thread(
                    target=_judge_worker,
                    args=(
                        judge_box,
                        judge_sets,
                        res["feats"],
                        res["y_mean"],
                        res["completions"],
                        out_root / "work",
                        args.judge_limit,
                        args.retest_n,
                    ),
                    daemon=True,
                )
                judge_thread.start()
                _log("[phase=judge] dispatched on a worker thread (overlaps remaining phase B)")
            else:
                _log(
                    "[phase=judge] SKIPPED — "
                    f"{_judge_skip_reason(args.judged_labels, args.skip_judge)}"
                )
    judge_out: dict | None = None
    if judge_thread is not None:
        judge_thread.join()
        if "error" in judge_box:
            raise judge_box["error"]
        judge_out = judge_box["result"]

    # ---- Phase C': per-feature evidence emission (plan v14 — never skipped) ----
    evidence_summary: dict[str, dict] = {}
    for cell in kept:
        evidence_summary[cell] = emit_feature_evidence(
            out_root,
            cell,
            results[cell],
            judge_sets if cell == "cell_inst_own" else None,
            args,
            fit_device,
        )
    _upload_tree(out_root, "feature_evidence", args, "phaseCprime-evidence")

    # ---- Phase D: joins + digests + upload ----
    out_dir = out_root / "out"
    maps_summary: dict = {"cells": {}, "repro": _repro_meta()}

    def _tailq(a: np.ndarray) -> dict:
        fin = np.asarray(a, dtype=np.float32)
        fin = fin[np.isfinite(fin)]
        if not fin.size:
            return {"p50": None, "p95": None, "p99": None, "max": None}
        return {
            "p50": float(np.percentile(fin, 50)),
            "p95": float(np.percentile(fin, 95)),
            "p99": float(np.percentile(fin, 99)),
            "max": float(fin.max()),
        }

    for cell, res in results.items():
        reads = res["reads"]
        maps_summary["cells"][cell] = {
            name: {k: v for k, v in read.items() if k != "r2_per_feature"}
            for name, read in reads.items()
        }
        maps_summary["cells"][cell]["induced_averaged"] = {
            k: v for k, v in res["induced"].items() if k != "r2_per_feature"
        }
        maps_summary["cells"][cell]["independently_fit_averaged"] = res["indep_averaged"]
        maps_summary["cells"][cell]["sink_robustness"] = res["sink_robustness"]
        evil_i = RB_TRAIT_ORDER.index("evil")
        maps_summary["cells"][cell]["rb_alignment"] = {
            "headline": "scaffold_projected (plan v14 SCAFFOLD CONTROL; raw kept as companion)",
            "scaffold_rank": res["rb"]["scaffold_rank"],
            "rb_scaffold_mass_frac_per_trait": res["rb"]["rb_scaffold_mass_frac"],
            "trait_order": list(RB_TRAIT_ORDER),
            "raw": {
                "obs_max_over_traits": _tailq(res["rb"]["cos_max"]),
                "null_p95_of_per_draw_max": res["rb"]["null_p95"],
            },
            "scaffold_projected": {
                "obs_max_over_traits": _tailq(res["rb"]["cos_max_proj"]),
                "null_p95_of_per_draw_max": res["rb"]["null_p95_proj"],
            },
            # evil-specific reads carry the mean-centered read EXPLICITLY (plan v14)
            "evil_raw": _tailq(res["rb"]["cos_traits"][evil_i]),
            "evil_mean_centered": _tailq(res["rb"]["cos_traits_centered"][evil_i]),
            "evil_scaffold_projected": _tailq(res["rb"]["cos_traits_proj"][evil_i]),
        }
        maps_summary["cells"][cell]["anova_selection"] = {
            axis: {
                k: v
                for k, v in res["anova"]["nulls"][axis].items()
                if k not in ("per_feature_p", "null_draws", "draw_max", "draw_topk_mean")
            }
            for axis in ("prefix", "query")
        }
        # four-object matched table (plan v12 map (d)): same target everywhere
        r2 = {
            "prefix_end": res["reads"]["pre"]["pooled_r2_mean"],
            "bare_query": res["reads"]["bare"]["pooled_r2_mean"],
            "bare_query_dense": res["reads"]["bare_dense"]["pooled_r2_mean"],
            "context_end": res["reads"]["ctx"]["pooled_r2_mean"],
            "encode_then_averaged_induced": res["induced"]["pooled_r2_mean"],
        }
        tau = FOUR_OBJECT_R2_TAU
        rb_ = res["r2_bare"]
        rc_ = res["r2_ctx"]
        rp_ = res["r2_pre"]
        fin = np.isfinite(rb_) & np.isfinite(rc_) & np.isfinite(rp_)
        maps_summary["cells"][cell]["four_object_table"] = {
            "pooled_r2 (matched target = pooled-answer mean)": r2,
            "per_feature_slice_tau": tau,
            "n_bare_query_predictable": int((fin & (rb_ > tau)).sum()),
            "n_need_prefix (pre>tau, bare<=tau)": int((fin & (rp_ > tau) & (rb_ <= tau)).sum()),
            "n_need_composition (ctx>tau, bare<=tau, pre<=tau)": int(
                (fin & (rc_ > tau) & (rb_ <= tau) & (rp_ <= tau)).sum()
            ),
        }
    (out_dir / "maps_summary.json").write_text(json.dumps(maps_summary, indent=1, default=str))

    labels_payload: dict = {"repro": _repro_meta()}
    if judge_out is not None and judge_sets is not None and "cell_inst_own" in results:
        res = results["cell_inst_own"]
        labels_payload.update(judge_out)
        labels_payload["headline"] = _delta_block(
            judge_out, judge_sets, res["feats"], res["dense_latent"], args
        )
    else:
        labels_payload["skipped"] = (
            LABEL_FREEZE_NOTE
            if args.judged_labels != "on"
            else "--skip-judge or instruct arm unavailable"
        )
        if judge_sets is not None and "cell_inst_own" in results:
            feats_i = results["cell_inst_own"]["feats"]
            labels_payload["sets"] = {
                k: [int(feats_i[p]) for p in v.tolist()] for k, v in judge_sets.items()
            }
    (out_dir / "feature_labels.json").write_text(json.dumps(labels_payload, indent=1))

    template = _template_block(out_root, kept, results)
    (out_dir / "template_control.json").write_text(json.dumps(template, indent=1))

    sink_summary: dict[str, dict] = {}
    for c in cells:
        jp, _ = sink_map_paths(out_root, c)
        if jp.exists():
            m = json.loads(jp.read_text())
            sink_summary[c] = {
                "gamma_layer19_all_tokens": m["gamma_layer19_all_tokens"],
                "exclusion_source": m["exclusion_source"],
                "min_occ_effective": m["min_occ_effective"],
                "n_sink_positions": len(m["sink_positions"]),
                "n_sink_token_ids": len(m["sink_token_ids"]),
            }
    summary = {
        "cells_requested": cells,
        "cells_kept": kept,
        "gates": {c: {k: gate_records[c][k] for k in ("fve", "l0", "pass")} for c in cells},
        "gamma_layer19": {c: gate_records[c].get("gamma_layer19") for c in cells},
        "sink_map": sink_summary,
        "headline": labels_payload.get("headline"),
        "judged_axes": LABEL_FREEZE_NOTE,
        "evidence": evidence_summary,
        "rb_alignment_headline": {
            c: maps_summary["cells"][c]["rb_alignment"]["scaffold_projected"] for c in results
        },
        "wall_s": time.time() - t_run0,
        "args": {
            k: str(v)
            for k, v in vars(args).items()
            if k in ("cells", "seed", "smoke_prefixes", "smoke_queries", "null_draws", "hf_subdir")
        },
        "repro": _repro_meta(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "DONE_upload.json").write_text(
        json.dumps(
            {
                "ts": time.time(),
                "phases": ["A", "B"] + (["C"] if judge_out is not None else []) + ["Cprime", "D"],
            },
            indent=1,
        )
    )
    _upload_tree(out_root, "out", args, "phaseD-digests")
    _log(f"[phase=done] wall={time.time() - t_run0:.0f}s cells={kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
