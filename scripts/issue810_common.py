# ruff: noqa: RUF003
# Intentional Unicode (→, ρ, ×, ², r_B) in scientific docstrings + log messages.
"""Shared helpers for issue #810 (answer-side summary/position sweep on θ0).

#810 asks the mirror question of #722 on the ANSWER side: which answer-side
summary of a response best supports BOTH (a) the linear context→answer map
`c_C → summary` (held-out skill-over-mean R², #722's DV) AND (b) reading a
behavior E0 out of the summary (fixed r_B + trained LOCO-ridge). The single
manipulated variable is the answer-side summary/position; everything else is
inherited from #658 (base model, 50-context grid, r_B, E0).

NOT a library module under ``src/`` — lives next to the ``scripts/issue810_*``
entry points it serves (same convention as ``issue658_common.py``).

Design contracts encoded here (plan §4.4, §13):

- **New position summaries** (extension of `issue658_common.summarize_answer_span`):
  ``im_end`` / ``turn_nl`` (the two turn-boundary positions AFTER the answer
  content — captured fresh in Phase B, NOT slice-derivable from the stored
  answer-CONTENT span) + ``tail_1..16`` / ``head_0..15`` (end-/start-aligned
  answer-CONTENT positions, slice-derivable from the stored span; ``tail_1`` ==
  ``last``). The deterministic free set {mean, last, maxp} is already in
  #658's ``store/v0_summaries.pt``.
- **Aligned-subset store schema** (plan §13, shared with #812): one file per
  context ``answer_position_sweep/<context_id>.pt`` — a dict carrying the
  per-position probe-mean summary vectors, plus a coverage count per position.
- **Fail-loud** on every drift (probe_pool_hash, context coverage, position
  set, sha256 pins) — never a silent skip.

Cross-refs: `issue658_common` (the recipe switch this extends), `issue658_
fit_predictors` (RIDGE_LAMBDAS/MLP defaults, on main), `vectorized_mlp_skill`
(the batched LOCO fitters, on main), the stranded `issue722_per_position_vC_
skill.py` on branch `fig-per-position` — REFERENCE ONLY, never imported
(built-but-stranded protection, `.claude/rules/workflow-fix-on-bug.md`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584

# Qwen-2.5-7B chat-template turn-end tokens (the two boundary positions Phase B
# captures AFTER the answer content). Asserted in-process at extraction time.
IM_END_TOKEN_ID = 151645  # <|im_end|>
# The trailing "\n" after <|im_end|>. Pinned to the Qwen-2.5 family id 198 — the
# SAME id for the 7B production model AND the 0.5B smoke model (verified), so the
# extractor asserts nl_id == 198 in ALL modes (a tokenizer/model revision that
# gave "\n" a different id would silently capture the WRONG turn_nl position
# across the whole run). Phase B still locates the slot STRUCTURALLY (the position
# after the im_end slot); the id pin is the drift guard.
TURN_NL_TOKEN_ID = 198

# HF data-repo destination (SHARED with #658/#812 — the aligned-subset store
# lands under #658's prefix so #812 consumes it without re-extraction, plan §13).
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue658_theory_assumptions"
# Phase B aligned-subset store (plan §13). One file per context.
ANSWER_POSITION_SWEEP_SUBDIR = "answer_position_sweep"
# The 50-context battery snapshot pin (uploaded as an issue-810 input, §4 / the
# artifact-reuse (h) rule — the local data/issue594/battery.json is gitignored,
# so the git-clone GCP lane fetches this pinned HF copy).
BATTERY50_HF_FILE = f"{HF_PREFIX}/{ANSWER_POSITION_SWEEP_SUBDIR}/inputs/battery50.json"
BATTERY50_SHA256 = "514c87daf8b06aff9c4804ee475ebb0722a8d7b7eed513f9f7a825b5208d6214"

# #658 stores this analysis reuses (all VERIFIED to resolve on HF main).
I658_V0_SUMMARIES = f"{HF_PREFIX}/store/v0_summaries.pt"
I658_RB = f"{HF_PREFIX}/store/r_b.pt"
I658_STORE_MANIFEST = f"{HF_PREFIX}/store/store_manifest.json"
I658_ANSWER_SPANS_PREFIX = f"{HF_PREFIX}/store/answer_spans"
I658_RAW_COMPLETIONS_PREFIX = f"{HF_PREFIX}/raw_completions/raw_completions"
I658_E0_GEN_PREFIX = f"{HF_PREFIX}/raw_completions/e0_gen"

# #594 last-input-token c_C store (the reconstruction predictor) + its probe pool
# pin (fail loud on drift — same 48-probe battery).
I594_CC_LAST_FILE = "issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"
I594_PROBE_POOL_HASH = "ad687becec266286549aaaa1af3b35e246d593e012e233564e58ff75fb015dd7"

# #722 tf_margin judge-validation reference (committed to main by §4.0 step 1).
I722_TF_MARGIN_FILE = "eval_results/issue_722/tf_margin/margins.json"

# Judge (the standing project rule; #763 graded rubric params).
JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# The high-m behaviors re-judged in Phase C off #658's stored completions.
# broad_em EXCLUDED (floors on base). fixed-r_B exists for {harmful, syco,
# refusal} (broad_em dropped); trained LOCO-ridge runs on all graded-E0.
HIGH_M_BEHAVIORS: tuple[str, ...] = ("sycophancy", "refusal", "harmful_compliance")
# tf_margin covers {broad_em, refusal, sycophancy} — the judge-validation
# overlap with the read-out behaviors is {refusal, sycophancy}. harmful_compliance
# has NO ± pool (validation gap noted, never fabricated).
TF_MARGIN_VALIDATION_BEHAVIORS: tuple[str, ...] = ("refusal", "sycophancy")

# Per-context high-m re-judge subsample (plan §11): sycophancy has 200 probes ×
# 10 = 2000 completions/context; subsample to a stable per-context mean.
SYCOPHANCY_SUBSAMPLE_PER_CONTEXT = 60

# Fit / null recipe pins (plan §10). RIDGE_LAMBDAS / MLP_* are imported from the
# on-main issue658_fit_predictors by the fit scripts; the null seed lives here.
SHUFFLE_NULL_PERMS = 1000
SHUFFLE_NULL_SEED = 658
PCA_TARGET_DIM_CAP = 48  # target dim = min(48, n-2), via robust_pca_basis
PER_POSITION_WINDOW_K = 16  # tail -1..-16 + head 0..15


def summary_names() -> list[str]:
    """The full #810 candidate summary set (plan §4.4).

    Deterministic reductions only — {mean, last, maxp} (already stored),
    im_end + turn_nl (turn-boundary, captured Phase B), tail_1..16 +
    head_0..15 (answer-content positions). ``attn`` is DEFERRED (a learned
    reduction, out of scope — plan § Anti-patterns).
    """
    names = ["mean", "last", "maxp", "im_end", "turn_nl"]
    names += [f"tail_{k}" for k in range(1, PER_POSITION_WINDOW_K + 1)]
    names += [f"head_{k}" for k in range(PER_POSITION_WINDOW_K)]
    return names


# Positions captured/stored per context in the Phase B aligned-subset store.
# im_end + turn_nl + tail_1..16 + head_0..15 = 34 positions.
def stored_position_names() -> list[str]:
    """The per-position keys stored in ``answer_position_sweep/<ctx>.pt``.

    im_end, turn_nl, tail_1..16, head_0..15 (34 positions). mean/last/maxp are
    NOT stored here — they live in #658's v0_summaries.pt (the free leg reads
    them there); ``last`` == ``tail_1`` so it is recomputable from the tail set.
    """
    names = ["im_end", "turn_nl"]
    names += [f"tail_{k}" for k in range(1, PER_POSITION_WINDOW_K + 1)]
    names += [f"head_{k}" for k in range(PER_POSITION_WINDOW_K)]
    return names


def tail_head_position_index(name: str, span_len: int) -> int | None:
    """Map a tail_k / head_k position name to a 0-based index into an S-length span.

    ``tail_k`` (k=1..16) -> position ``S - k`` (tail_1 == last content token).
    ``head_k`` (k=0..15) -> position ``k`` (head_0 == first content token).
    Returns None if the position is out of range for this span (short answers).
    im_end / turn_nl are NOT tail/head content positions -> raises (they are
    captured fresh, not sliced).
    """
    if name.startswith("tail_"):
        k = int(name.split("_")[1])
        idx = span_len - k
        return idx if 0 <= idx < span_len else None
    if name.startswith("head_"):
        k = int(name.split("_")[1])
        return k if 0 <= k < span_len else None
    raise ValueError(f"{name!r} is not a tail_k/head_k position (im_end/turn_nl captured fresh)")


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex over bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    """SHA-256 hex over a file's bytes (input pin verification)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_sha256(path: Path | str, expected: str, label: str) -> None:
    """Fail loud if a pinned input's sha256 drifts (the #600 HF-mirror guard)."""
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(
            f"{label} sha256 pin drift: {got} != {expected} (the reused artifact "
            f"differs from the plan-verified copy — refuse rather than run on a "
            f"silently-different generation, .claude/rules/artifact-reuse.md (f))"
        )


def load_json(path: Path | str):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path: Path | str) -> None:
    """Atomic-ish JSON write (tmp + rename); parent dirs created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def reproducibility_metadata() -> dict:
    """git commit + env versions + timestamp for every result JSON (CLAUDE.md).

    Lightweight, self-contained (no cross-script import so it is importable on
    any lane). Missing git / package is recorded as None, never a crash.
    """
    import platform
    import subprocess
    import sys
    from datetime import UTC, datetime

    def _git() -> str | None:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            return None

    def _ver(mod: str) -> str | None:
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    return {
        "git_commit": _git(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _ver("torch"),
        "numpy": _ver("numpy"),
        "transformers": _ver("transformers"),
    }


def context_ids_from_manifest(manifest: dict) -> list[str]:
    """The 50 store context_ids, order-stable (the LOCO fold order)."""
    ids = manifest.get("context_ids")
    if not ids or len(set(ids)) != len(ids):
        raise RuntimeError(f"store_manifest context_ids missing/duplicated: {ids!r}")
    return list(ids)
