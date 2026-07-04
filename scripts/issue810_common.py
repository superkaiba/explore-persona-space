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

# ── UltraChat `_g1` genre arm (same-issue follow-up `ultrachat-genre-summary-sweep`) ──
# #658's genre-generalization stores this round consumes (plan v6 §4.6 item 1;
# all VERIFIED on HF 2026-07-02 via list_repo_files/list_repo_tree + head-checks).
# The SINGLE manipulated variable is the probe-corpus genre (Betley → UltraChat);
# `--genre betley` (the default everywhere) keeps the parent paths bit-for-bit.
G1_GENRE_TAG = "genre-generalization-ultrachat"
G1_STORE_PREFIX = f"{HF_PREFIX}/store_{G1_GENRE_TAG}"
G1_V0_SUMMARIES = f"{G1_STORE_PREFIX}/v0_summaries.pt"
G1_STORE_MANIFEST = f"{G1_STORE_PREFIX}/store_manifest.json"
G1_RAW_COMPLETIONS_PREFIX = f"{HF_PREFIX}/raw_completions_{G1_GENRE_TAG}/raw_completions"
G1_E0_GEN_PREFIX = f"{HF_PREFIX}/e0_gen_{G1_GENRE_TAG}"
# g1 probe pool pin (data/issue594/probes_ultrachat.json, 48 UltraChat probes —
# the probes ride inside the raw-completion files, so this pin is asserted on
# every g1 store-manifest / v0_summaries load rather than on a pool fetch).
G1_PROBE_POOL_HASH = "f277f8c3e2550b2ce3e4545a8ad6473498d070e7343eb7c9398a6aac31525455"
# Phase B-g aligned-subset store destination (schema identical to the parent's
# §13 store; plan v6 § Storage naming).
G1_ANSWER_POSITION_SWEEP_SUBDIR = f"{ANSWER_POSITION_SWEEP_SUBDIR}_{G1_GENRE_TAG}"
# Round out-dir / HF-mirror conventions (the follow-up-label convention).
G1_FOLLOWUP_LABEL = "ultrachat-genre-summary-sweep"
G1_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_810" / G1_FOLLOWUP_LABEL
G1_HF_RESULTS_PREFIX = f"issue810_results/{G1_FOLLOWUP_LABEL}"
# The parent's committed Phase-C graded E0 (branch issue-810) — the `betley`
# E0-target axis of the read-out 2×2 square (never re-judged this round).
BETLEY_E0_HIGHM_FILE = (
    PROJECT_ROOT / "eval_results" / "issue_810" / "phase_c" / ("e0_highm_graded.json")
)

GENRES = ("betley", "g1")

# ── `_uh` next-user-header arm (same-issue follow-up `user-header-newline-summary`) ──
# Round 3 extends the Phase B captured span by the 3-token next-user header:
# `prompt + answer + <|im_end|> + \n + <|im_start|> + user + \n`. The SINGLE
# manipulated variable is the captured span (+2 → +5 appended tokens) and the 12
# summary rows derived from it (plan v11 §4). `--extended-boundary` OFF (the
# default everywhere) keeps the parent paths bit-for-bit.
#
# Qwen-2.5 chat-template assistant-turn continuation token ids, VERIFIED with the
# production tokenizer incl. the apply_chat_template tail (plan v11 §12 A1):
# <|im_end|>=151645, \n=198, <|im_start|>=151644, user=872, \n=198. Asserted
# per probe at extraction time (fail loud — a wrong id silently captures the
# wrong slot).
BOUNDARY_BLOCK_IDS: tuple[int, ...] = (151645, 198, 151644, 872, 198)
# The 3 next-user-header singles (positions boundary_offset + 2/3/4) …
UH_POSITION_NAMES: list[str] = ["uh_im_start", "uh_user", "uh_nl"]
# … and the 6 in-forward span pools (computed GPU-side per probe BEFORE the
# probe-mean — NOT reconstructable from the stored answer-only summaries):
# uh_mean3/uh_max3 over the 3 header positions, bnd_mean5/bnd_max5 over all 5
# boundary tokens, mean_xbnd/maxp_xbnd over (answer content ∪ 5 boundary tokens).
UH_POOL_NAMES: list[str] = [
    "uh_mean3",
    "uh_max3",
    "bnd_mean5",
    "bnd_max5",
    "mean_xbnd",
    "maxp_xbnd",
]
# The 9 new per-layer summary rows (H1-uh selection axis: 37 committed + these).
UH_SUMMARY_NAMES: list[str] = UH_POSITION_NAMES + UH_POOL_NAMES
# Phase B-x aligned-subset store destination (plan v11 § Storage naming).
ANSWER_POSITION_SWEEP_UH_SUBDIR = "answer_position_sweep_user_header"
# Round out-dir / HF-mirror conventions (the follow-up-label convention).
UH_FOLLOWUP_LABEL = "user-header-newline-summary"
UH_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_810" / UH_FOLLOWUP_LABEL
UH_HF_RESULTS_PREFIX = f"issue810_results/{UH_FOLLOWUP_LABEL}"
# Compact new-row summaries tensor (the CPU-chain input; ~90 MB at fp16 —
# 50 ctx × 9 rows × 28 layers × 3584 dims).
UH_SUMMARIES_HF_FILE = f"{UH_HF_RESULTS_PREFIX}/uh_summaries.pt"

# ── `_he` header-echo ablation arm (same-issue follow-up `header-echo-ablation-capture`) ──
# Round 4 (plan v15) re-captures the SAME 50-ctx × 48-probe grid with the answer
# span ABLATED: the teacher-forced sequence is `prompt + BOUNDARY_BLOCK_IDS`
# (the assistant turn opens and ends immediately with no content). The SINGLE
# manipulated variable is the answer span (present → EMPTY); `--ablate-answer`
# OFF (the default everywhere) keeps the round-3 paths bit-for-bit.
HE_FOLLOWUP_LABEL = "header-echo-ablation-capture"
# Phase B-he aligned-subset store destination (plan v15 § Storage naming).
ANSWER_POSITION_SWEEP_HE_SUBDIR = "answer_position_sweep_header_echo"
HE_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_810" / HE_FOLLOWUP_LABEL
HE_HF_RESULTS_PREFIX = f"issue810_results/{HE_FOLLOWUP_LABEL}"
# Compact empty-answer summaries pack (the CPU-chain input; same shape/schema
# as uh_summaries.pt — 50 ctx × 9 rows × 28 layers × 3584 fp16).
HE_SUMMARIES_HF_FILE = f"{HE_HF_RESULTS_PREFIX}/he_summaries.pt"
# The 9 empty-answer analysis rows. Row NAMES are deliberately UNCHANGED from
# the committed round-1/round-3 rows (H1-he pairing is by name); the whole-turn
# xbnd pools + tail/head answer positions are DROPPED — mechanically undefined
# at ans_len=0 (plan v15 §4 divergence 2).
HE_POSITION_NAMES: list[str] = ["im_end", "turn_nl", *UH_POSITION_NAMES]
HE_POOL_NAMES: list[str] = ["uh_mean3", "uh_max3", "bnd_mean5", "bnd_max5"]
HE_SUMMARY_NAMES: list[str] = HE_POSITION_NAMES + HE_POOL_NAMES

# ── `_btdr` truncation-dose arm (same-issue follow-up `boundary-truncation-dose-response`) ──
# Round 5 (plan v18) doses the round-4 deleted answer span: the teacher-forced
# sequence is `prompt + ans_ids[:n_keep] + BOUNDARY_BLOCK_IDS` with
# `n_keep = max(1, ceil(k * ans_len))` per probe (ID-prefix cut, never
# re-tokenized text). The SINGLE manipulated variable is the retained answer
# fraction k; `--truncate-frac` OFF (the default everywhere) keeps the
# round-3/round-4 paths bit-for-bit. Row NAMES reuse he_stored_position_names()
# unchanged — pairing across sides and k is by name (plan v18 §4.6 item 1).
BTDR_FOLLOWUP_LABEL = "boundary-truncation-dose-response"
# Per-k aligned-subset store subdir under HF_PREFIX (plan v18 § Storage naming).
ANSWER_POSITION_SWEEP_BTDR_SUBDIR_TMPL = "answer_position_sweep_btdr_k{pct}"
BTDR_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_810" / BTDR_FOLLOWUP_LABEL
BTDR_HF_RESULTS_PREFIX = f"issue810_results/{BTDR_FOLLOWUP_LABEL}"
# Compact per-k truncated-answer summaries pack (the CPU-chain input).
BTDR_SUMMARIES_HF_FILE_TMPL = "btdr_summaries_k{pct}.pt"
# The production k grid (interior points; endpoints k=0 / k=100 reuse the
# committed round-4 / round-1+3 data). 1.0 is the ENDPOINT-PARITY PROBE value
# only (the truncate code path degenerates to the round-3 full capture) — it
# is admitted by the (0, 1] validator but never a production capture k.
BTDR_TRUNCATE_FRACS: tuple[float, ...] = (0.25, 0.5, 0.75)


def btdr_pct(k: float) -> int:
    """Canonical integer percent for a truncate fraction (0.25 -> 25); fail loud.

    Refuses a k whose percent is not integral (the storage naming templates key
    on the integer percent — a drifted float would silently split stores).
    """
    pct = round(k * 100)
    if abs(k * 100 - pct) > 1e-9 or not (0 < pct <= 100):
        raise ValueError(f"truncate_frac {k!r} has no canonical integer percent")
    return int(pct)


def he_stored_position_names() -> list[str]:
    """The per-position keys stored in ``answer_position_sweep_header_echo/<ctx>.pt``.

    The cc predictor position (``cc_last``, the #594 last-input-token slot at
    ``prompt_len - 1`` — it RIDES the same ablated forward and feeds the #594
    store parity probe, plan v15 §0/§11) + the 5 boundary singles + the 4
    header/boundary pools = 10 rows per context. NO tail/head/xbnd rows
    (undefined at ans_len=0).
    """
    return ["cc_last", *HE_POSITION_NAMES, *HE_POOL_NAMES]


def uh_stored_position_names() -> list[str]:
    """The per-position keys stored in ``answer_position_sweep_user_header/<ctx>.pt``.

    The parent's 34 positions (recaptured in the SAME forward — the round-1
    drift check rides free) + the 3 next-user-header singles + the 6 in-forward
    span pools = 43 rows per context.
    """
    return stored_position_names() + UH_POSITION_NAMES + UH_POOL_NAMES


def enlarged_summary_names() -> list[str]:
    """The H1-uh enlarged selection axis: 37 committed rows + the 9 new rows = 46."""
    return summary_names() + UH_SUMMARY_NAMES


class UhPackValidationError(RuntimeError):
    """A uh_summaries pack failed production-path validation (fail loud, pre-fit)."""


def validate_uh_pack(
    rows: dict,
    coverage: dict,
    meta: dict,
    *,
    requested_rows: list[str],
    ctx_ids: list[str],
    expected_model: str = DEFAULT_MODEL,
    expected_capture_layers: list[int] | None = None,
) -> None:
    """PRODUCTION-path validation of a loaded uh_summaries pack (fail loud, pre-fit).

    Callers branch on the pack's smoke provenance: a ``meta['smoke']`` pack takes
    the caller's explicit relaxed path (partial-context pairing, layer-prefix
    truncation) and NEVER reaches this helper on a production run — this helper
    REFUSES it. Non-smoke checks (round-2 hardening of the r1 CONCERNs
    ``uh-pack-meta-validation-readout`` / ``uh-pack-validation-bootstrap``):

    - ``meta['smoke'] is False`` (a smoke / pre-meta pack cannot feed a
      production fit);
    - ``meta['model'] == expected_model`` (the 7B production model);
    - ``meta['capture_layers'] == expected_capture_layers`` (default
      ``list(range(EXPECTED_LAYERS))`` — the full 28-layer axis);
    - ``meta['context_ids']`` covers the production ctx grid as a SET;
    - EVERY requested row has, for EVERY production context, a tensor with the
      full layer axis AND positive coverage.

    Raises :class:`UhPackValidationError` naming the first offenders.
    """
    if expected_capture_layers is None:
        expected_capture_layers = list(range(EXPECTED_LAYERS))
    if meta.get("smoke") is not False:
        raise UhPackValidationError(
            f"uh pack smoke-provenance check failed: meta['smoke']={meta.get('smoke')!r} — a "
            "smoke (or pre-meta) pack cannot feed a production fit"
        )
    if meta.get("model") != expected_model:
        raise UhPackValidationError(
            f"uh pack model mismatch: pack={meta.get('model')!r} expected={expected_model!r}"
        )
    pack_layers = meta.get("capture_layers")
    if pack_layers is None or list(pack_layers) != list(expected_capture_layers):
        raise UhPackValidationError(
            f"uh pack capture_layers mismatch: pack has "
            f"{len(pack_layers) if pack_layers is not None else None} layers "
            f"({list(pack_layers)[:4] if pack_layers else None}...), expected the full "
            f"{len(expected_capture_layers)}-layer axis — layer-prefix truncation is a "
            "smoke-only path"
        )
    pack_ctx = set(meta.get("context_ids") or [])
    missing_meta_ctx = sorted(set(ctx_ids) - pack_ctx)
    if missing_meta_ctx:
        raise UhPackValidationError(
            f"uh pack meta context_ids missing {len(missing_meta_ctx)} production contexts "
            f"(e.g. {missing_meta_ctx[:5]})"
        )
    n_layers = len(expected_capture_layers)
    for row in requested_rows:
        per_ctx = rows.get(row)
        if per_ctx is None:
            raise UhPackValidationError(f"requested row {row!r} absent from the uh pack")
        no_tensor = [c for c in ctx_ids if c not in per_ctx]
        if no_tensor:
            raise UhPackValidationError(
                f"row {row!r}: {len(no_tensor)} production contexts lack a tensor "
                f"(e.g. {no_tensor[:5]})"
            )
        bad_shape = [c for c in ctx_ids if per_ctx[c].shape[0] != n_layers]
        if bad_shape:
            raise UhPackValidationError(
                f"row {row!r}: {len(bad_shape)} contexts have a truncated layer axis "
                f"(e.g. {bad_shape[:3]}: {per_ctx[bad_shape[0]].shape[0]} != {n_layers})"
            )
        no_cov = [c for c in ctx_ids if coverage.get(row, {}).get(c, 0) <= 0]
        if no_cov:
            raise UhPackValidationError(
                f"row {row!r}: {len(no_cov)} production contexts have zero coverage "
                f"(e.g. {no_cov[:5]})"
            )


def load_battery50(local_hint: Path | str | None = None) -> dict:
    """Load + sha256-pin the 50-context battery (local fast path, else HF snapshot).

    Local ``data/issue594/battery.json`` is gitignored (absent from the
    git-clone GCP lane), so on a miss the sha256-pinned HF snapshot
    (``BATTERY50_HF_FILE``) is fetched. Either way the sha256 is asserted
    (fail loud on drift, the #600 HF-mirror guard).
    """
    candidates: list[Path] = []
    if local_hint is not None:
        candidates.append(Path(local_hint))
    candidates.append(PROJECT_ROOT / "data" / "issue594" / "battery.json")
    for c in candidates:
        if c.is_file() and sha256_file(c) == BATTERY50_SHA256:
            return load_json(c)
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, BATTERY50_HF_FILE, repo_type="dataset")
    assert_sha256(p, BATTERY50_SHA256, "battery50.json")
    return load_json(p)


def battery_family_map(local_hint: Path | str | None = None) -> dict[str, str]:
    """{context_id: family} from the sha-pinned battery (the 7-family LOFO folds).

    Fails loud unless the map covers exactly 50 ids over exactly 7 families
    (``battery50.json`` ``instances[].family`` — plan v11 §12 A10).
    """
    blob = load_battery50(local_hint)
    inst = blob["instances"] if isinstance(blob, dict) else blob
    fam = {str(x["id"]): str(x["family"]) for x in inst}
    n_fam = len(set(fam.values()))
    if len(fam) != 50 or n_fam != 7:
        raise RuntimeError(
            f"battery family map: expected 50 ids over 7 families, got {len(fam)} ids "
            f"over {n_fam} families — refusing (LOFO fold structure drift)"
        )
    return fam


def assert_g1_probe_pool_hash(obj: dict, where: str) -> None:
    """Fail loud unless ``obj['probe_pool_hash']`` matches the g1 pin (plan §4.6).

    ``obj`` is a loaded g1 store manifest or v0_summaries pack (both carry the
    key — VERIFIED at plan time). A missing key OR a drifted hash refuses: the
    round would otherwise silently fit against a different probe pool.
    """
    pph = obj.get("probe_pool_hash")
    if pph != G1_PROBE_POOL_HASH:
        raise RuntimeError(
            f"g1 probe_pool_hash pin drift in {where}: {pph!r} != {G1_PROBE_POOL_HASH} "
            "(the reused #658 g1 artifact differs from the plan-verified generation — "
            "refusing, .claude/rules/artifact-reuse.md (f))"
        )


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


def reader_context_labels(max_len: int = 26) -> dict[str, str]:
    """Reader-facing plain-English label per battery context id (figure point labels).

    Loads the 50-context battery (local ``data/issue594/battery.json`` when
    present, else the sha-pinned HF snapshot ``battery50.json``) and maps each
    instance id to its human ``label`` field — underscores become spaces and
    long PersonaHub descriptions are truncated at a word boundary to
    ``max_len`` chars plus an ellipsis. Raises if the battery does not yield
    exactly 50 ids.
    """
    blob = load_battery50()
    out: dict[str, str] = {}
    for inst in blob["instances"]:
        lab = str(inst["label"]).replace("_", " ")
        if len(lab) > max_len:
            lab = lab[:max_len].rsplit(" ", 1)[0].rstrip(" ,.") + "…"
        out[str(inst["id"])] = lab
    if len(out) != 50:
        raise RuntimeError(f"battery labels: expected 50 ids, got {len(out)}")
    return out


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


def retry_hub_quota(fn, attempts: int = 6, sleep_s: float = 75.0):
    """Retry ``fn()`` over Hub 429/5xx (the 2500-req/5-min org-quota window; #658/#833).

    The paginated tree endpoint retries 429 ONLY on follow-up cursor pages — the
    FIRST page (and single-path probes like ``file_exists``) raise immediately, so
    a post-upload verify issued in the tail of a quota storm needs this outer
    BOUNDED retry (never an unbounded loop; raises the last error at the cap).
    """
    import time

    from huggingface_hub.errors import HfHubHTTPError

    last: Exception | None = None
    for _ in range(attempts):
        try:
            return fn()
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status not in (429, 500, 502, 503, 504):
                raise
            last = e
            time.sleep(sleep_s)
    raise last  # type: ignore[misc]


def scoped_remote_listing(prefix: str) -> set[str]:
    """Fresh listing of ``prefix`` on the data repo, server-side SCOPED to the prefix.

    A bare ``list_repo_files`` full listing of the ~1M-file data repo paginates the
    ENTIRE tree (~1000 cursor pages) under the 2500-req/5-min org quota — a 429
    retry wedge that idles the GPU for hours (gotchas.md #833; hit live on the
    btdr round, 2026-07-04). ``path_in_repo`` rides in the tree URL so pagination
    covers only the subtree — seconds for issue-scale prefixes.
    """
    from huggingface_hub import HfApi

    def _list() -> set[str]:
        tree = HfApi().list_repo_tree(
            HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", revision="main", recursive=True
        )
        return {e.path for e in tree}

    return retry_hub_quota(_list)


def upload_out_dir(out_dir: Path | str, path_in_repo: str) -> str:
    """Bulk-commit an out-dir's ``*.json`` to the HF data repo, then fail-loud verify.

    Shared by the ephemeral-lane fit scripts (``issue810_fit_reconstruction.py`` /
    ``issue810_fit_readout.py``): a GCP spot instance is DELETED on exit, so its
    result JSONs must land on ``HF_DATA_REPO`` (``repo_type="dataset"``) before
    teardown. Mirrors ``issue810_extract_positions._upload_store`` — ONE
    ``upload_folder`` commit (never a per-file loop — the #664 504-storm), then
    verifies EVERY produced JSON is present on a FRESH prefix-scoped listing
    and raises ``RuntimeError`` on any miss (never a silent partial upload).

    Returns the ``path_in_repo`` the JSONs landed under.
    """
    from huggingface_hub import HfApi

    out_dir = Path(out_dir)
    local_jsons = sorted(p.name for p in out_dir.glob("*.json"))
    if not local_jsons:
        raise RuntimeError(f"upload_out_dir: no *.json in {out_dir} to upload")
    api = HfApi()
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=path_in_repo,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.json"],
        commit_message=f"issue #810: fit results ({len(local_jsons)} JSONs) -> {path_in_repo}",
    )
    remote = scoped_remote_listing(path_in_repo)
    expected = {f"{path_in_repo}/{name}" for name in local_jsons}
    missing = expected - remote
    if missing:
        raise RuntimeError(
            f"fit-result upload verification FAILED: {len(missing)} JSON file(s) missing "
            f"on the Hub under {path_in_repo}/ (e.g. {sorted(missing)[:3]}) — refusing to "
            f"treat a partial upload as success (an ephemeral lane would lose them on teardown)"
        )
    return path_in_repo
