# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ², ×) in scientific docstrings + log messages.
"""Shared helpers for issue #928 (thinking-model CoT decomposition of the context→answer map).

#928 re-runs the #810 context→answer linear-map question on a THINKING model
(`open-thoughts/OpenThinker2-7B`, a Qwen2.5-7B-Instruct SFT with an identical
config/tokenizer/chat template) and decomposes the map across the chain of
thought: per (context C, probe q) the response is ``<think> t </think> a`` and
per-part per-layer summary vectors are captured for parts {ctx, cot, ans} ×
summaries {mean, max, boundary} (+ exploratory boundary variants). Fits are the
inherited #810 estimator (LOCO ridge, PCA-48 target, skill-over-mean R²) over
six arms (direct / stage-1 / stage-2-oracle / composed / joint / augmented).

NOT a library module under ``src/`` — lives next to the ``scripts/issue928_*``
entry points it serves (same convention as ``issue810_common.py``, which this
imports for the battery pins + JSON/sha helpers).

Design contracts encoded here (plan §4.4, §4.5, §10):

- **Segmentation is code, not a model call** (plan §4.4): an exact string /
  token-offset operation with a deterministic structural output. Well-formed
  (rungs i/ii): exactly one ``<think>`` at char 0, exactly one ``</think>``,
  non-empty answer after the close. Rung (iii) (prefill ``<think>\\n`` in the
  PROMPT): well-formed iff the completion contains exactly one ``</think>``
  (no ``<think>`` requirement). Malformed rows are DROPPED with per-context
  coverage counts (graceful degradation, §4.8) — never repaired. Malformed
  reason taxonomy: the STRUCTURAL classes ``segment_completion`` assigns
  (``no_close`` / ``multiple_close`` / ``no_think_open`` / ``multiple_think``
  / ``think_not_at_start`` / ``empty_cot`` / ``empty_answer``), the
  ``finish_reason == "length"`` override ``truncated_no_close``, and the v3
  amendment class ``degenerate_repetition`` (a segmentation-well-formed row
  whose repeated-4-gram fraction exceeds ``REPEAT_4GRAM_MAX_FRAC``, assigned
  in the extractor's ``parse_rows`` — dropped + coverage-counted exactly like
  ``truncated_no_close`` / ``no_close``; structural/truncation reasons win).
- **Char offsets, not token re-search** (§4.4): token spans derive from
  ``return_offsets_mapping`` at the Phase-B re-tokenization — robust to BPE
  merges (measured: ``\\n</think>\\n\\n`` = [198, 522, 26865, 1339]; the ``>``
  merges with trailing newlines, so re-tokenizing the tag alone would miss the
  in-context ids). Zero-width token spans (#825 BPE-delimiter-merge class) are
  dropped with a counted reason, never silently kept.
- **Fail-loud** on probe-pool-hash / battery-sha / layer-count drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from issue810_common import (  # noqa: E402  (re-exported for issue928_* consumers)
    BATTERY50_HF_FILE,
    BATTERY50_SHA256,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    HF_DATA_REPO,
    PCA_TARGET_DIM_CAP,
    SHUFFLE_NULL_PERMS,
    SHUFFLE_NULL_SEED,
    assert_sha256,
    dump_json,
    load_json,
    reproducibility_metadata,
    sha256_file,
)

__all__ = [
    "BATTERY50_HF_FILE",
    "BATTERY50_SHA256",
    "EXPECTED_HIDDEN",
    "EXPECTED_LAYERS",
    "HF_DATA_REPO",
    "PCA_TARGET_DIM_CAP",
    "SHUFFLE_NULL_PERMS",
    "SHUFFLE_NULL_SEED",
    "assert_sha256",
    "dump_json",
    "load_json",
    "reproducibility_metadata",
    "sha256_file",
]

# ── model under test (plan §10/§11) ───────────────────────────────────────────

THINKING_MODEL = "open-thoughts/OpenThinker2-7B"
# Qwen2.5-7B-Instruct lineage — config field-identical (28 layers, hidden 3584);
# asserted at load time in the extractor against EXPECTED_LAYERS/EXPECTED_HIDDEN.

# Chat-template / boundary token ids (identical to the Qwen-2.5 family; verified
# in the plan session and re-asserted in-process at extraction time).
IM_END_TOKEN_ID = 151645  # <|im_end|>
ENDOFTEXT_TOKEN_ID = 151643  # <|endoftext|> (secondary eos in generation_config)
TURN_NL_TOKEN_ID = 198  # "\n"
GENERATION_SUFFIX = "<|im_start|>assistant\n"  # #594 lineage assistant-header assert

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# ── generation recipe (plan §4.3/§11) ─────────────────────────────────────────

MAX_NEW_TOKENS = 8192  # grounded: OpenThoughts2 trace-length measurement (§11)
MAX_NEW_TOKENS_RETRY = 16384  # one re-run rung for >10% cap-truncated rows (§4.4)
GPU_MEMORY_UTILIZATION = 0.85  # plan §4.3 override of the parent helper's 0.45
MAX_MODEL_LEN = 32768  # model max_position_embeddings; holds prompt + 16k retry
GENERATION_SEED = 42  # only used by the sampling rung (ii)
TRUNCATION_REGEN_FRAC = 0.10  # >10% cap-truncated rows → one 16,384 re-run (§4.4)

# Pre-registered SAME-MODEL fallback ladder (plan §4.3). Rung (iii) exhaustion is
# TERMINAL (epm:failure failure_class: data) — there is NO model-switch rung.
FALLBACK_RUNGS = ("greedy", "sample", "prefill")
PREFILL_TEXT = "<think>\n"  # rung (iii): appended after the assistant header

# Gate 1 (plan §7, amended v3): parse-rate floor + degeneration checks on the
# gate slice. Two confusably-named repetition constants — do NOT conflate:
# REPEAT_4GRAM_MAX_FRAC is the PER-ROW offender DEFINITION (fraction of ONE
# completion's word 4-grams that are repeats; > 0.50 ⇒ that row is an
# offender), while REPEAT_OFFENDER_MAX_FRAC is the GATE-level offender RATE
# over rows (offenders / all gate rows; > 0.10 ⇒ Gate 1 degeneration conjunct
# FAILs). Offenders are reclassified malformed (``degenerate_repetition``) in
# ``parse_rows`` and drop-and-count like any other malformed class (§4.4).
PARSE_RATE_FLOOR = 0.80
REPEAT_4GRAM_MAX_FRAC = 0.50  # per-row offender definition (unchanged from v2)
REPEAT_CHECK_MIN_WORDS = 50  # degeneration loops are long; short replies exempt
REPEAT_OFFENDER_MAX_FRAC = 0.10  # v3 gate offender-RATE threshold (§7/§11; ~TRUNCATION_REGEN_FRAC)
GATE_P95_MUST_BE_BELOW_CAP = True

# ── HF destinations (plan §10 output artifacts) ───────────────────────────────

HF_PREFIX_928 = "issue928_cot_decomposition"
RAW_COMPLETIONS_PREFIX = f"{HF_PREFIX_928}/raw_completions/thinking_rollouts"
STORE_PREFIX = f"{HF_PREFIX_928}/analysis_tensors/store/percq_summaries"
FIT_RESULTS_PREFIX = f"{HF_PREFIX_928}/fit_results"
# Round-2 additions (code-review r1 artifact-loss minors): the per-context LOCO
# decompositions (decomp_<regime>.pt — the bootstrap's re-reduction input) and
# the pod-side figures both upload before instance teardown (GCE DELETEs the
# boot disk; anything not on the Hub dies with it).
DECOMP_TENSORS_PREFIX = f"{HF_PREFIX_928}/analysis_tensors/decomp"
FIGURES_PREFIX = f"{HF_PREFIX_928}/figures"

# ── parts × summaries registry (plan §4.5 — the 12 stored vectors) ────────────

PARTS = ("ctx", "cot", "ans")
REGISTERED_SUMMARIES = ("mean", "max", "boundary")

# Stored per-(C, q, layer) vector names, in the store tensor's axis order.
SUMMARY_NAMES: tuple[str, ...] = (
    "ctx_mean",
    "ctx_max",
    "ctx_last",  # assistant-header newline (parent c_C position)
    "cot_mean",
    "cot_max",
    "cot_last",  # last CoT content token (registered boundary)
    "cot_close",  # final token of </think> (exploratory)
    "ans_mean",
    "ans_max",
    "ans_last",  # last answer content token (registered boundary)
    "ans_im_end",  # <|im_end|> appended after the answer (exploratory)
    "ans_turn_nl",  # \n after <|im_end|> (exploratory)
)
SUMMARY_INDEX = {name: i for i, name in enumerate(SUMMARY_NAMES)}

# Registered boundary vector per part (the "boundary" registered summary).
BOUNDARY_NAME = {"ctx": "ctx_last", "cot": "cot_last", "ans": "ans_last"}


def part_summary_name(part: str, summary: str) -> str:
    """Stored vector name for (part ∈ PARTS, summary ∈ REGISTERED_SUMMARIES)."""
    assert part in PARTS, part
    assert summary in REGISTERED_SUMMARIES, summary
    if summary == "boundary":
        return BOUNDARY_NAME[part]
    return f"{part}_{summary}"


# ── matched-length answer-span control (follow-up round, plan v6 §4.1) ───────

MLC_K_MIN = 8  # conditioning-slice floor (tokens)
MLC_REM_MIN = 16  # remainder-target floor (tokens)

# The 7 mean-pool vectors captured per (row, layer) for the matched-length
# control store (plan v6 §4.2 item 4) — mean-only by design ("no tested
# summary displaces mean pooling", parent body).
MLC_SUMMARY_NAMES: tuple[str, ...] = (
    "ctx_mean",
    "cot_mean",
    "ans_mean",
    "cot_lastK_mean",
    "cot_firstK_mean",
    "ansprefix_K_mean",
    "ans_rem_mean",
)


def matched_length_spans(
    cot_tok: tuple[int, int],
    ans_tok: tuple[int, int],
    k_min: int = MLC_K_MIN,
    rem_min: int = MLC_REM_MIN,
) -> dict | None:
    """Per-row matched-length spans. K = min(len(cot), len(ans)//2).

    Returns ``None`` (dropped-and-counted, reason ``matched_length_floor``)
    when K < ``k_min`` or len(ans) − K < ``rem_min``. Spans are half-open
    token indices into the COMPLETION token space, all derived from the
    parent's cot/ans token spans (the same
    ``char_span_to_token_span(return_offsets_mapping)`` machinery
    ``build_capture_row`` already uses — no re-tokenization, no re-search).
    Registered CoT slice = ``cot_lastK`` (the CoT's conclusion); ``cot_firstK``
    is the exploratory second definition. ``ansprefix_K`` and ``ans_rem`` are
    disjoint by construction (prefix ends where the remainder starts).
    """
    cs, ce = cot_tok
    a0, a1 = ans_tok
    K = min(ce - cs, (a1 - a0) // 2)
    if k_min > K or (a1 - a0) - K < rem_min:
        return None
    return {
        "K": K,
        "cot_lastK": (ce - K, ce),  # registered CoT slice (the CoT's conclusion)
        "cot_firstK": (cs, cs + K),  # exploratory CoT slice
        "ansprefix_K": (a0, a0 + K),  # matched-length control slice
        "ans_rem": (a0 + K, a1),  # the SHARED prediction target
    }


# ── prefix-based mapping arms (follow-up round, plan v7 §4.1) ─────────────────

# The 5 mean-pool vectors captured per (row, layer) for the prefix-summaries
# store (plan v7 §4.2 item 4): 2 new prompt-side parts + 3 parity parts whose
# recapture is gated against the matched-length store (cos >= 0.999).
PMA_SUMMARY_NAMES: tuple[str, ...] = (
    "prefix_mean",
    "query_mean",
    "ctx_mean",
    "cot_mean",
    "ans_mean",
)


def prefix_query_spans(
    prompt_text_tpl: str,
    prompt_offsets: list[tuple[int, int]],
    prompt_len_tpl: int,
    probe: str,
) -> dict[str, tuple[int, int]] | str:
    """Prefix/query token spans over the TEMPLATED prompt (plan v7 §4.1).

    prefix = all templated-prompt tokens strictly before the FINAL user turn's
    probe content (system block + prefix_messages + chat scaffolding incl. the
    ``<|im_start|>user\\n`` header); query = the probe's own tokens. ``rfind``
    = last occurrence (the probe is the final user turn by construction —
    ``issue594_common.messages_for_instance``). Boundary conventions (stated):
    the user-turn header belongs to the PREFIX; a BPE token straddling the
    header/probe boundary joins the QUERY (overlap mapping, ≤ 1 token). The
    post-query assistant-header scaffolding belongs to neither part.

    Returns ``{"prefix": (0, q0), "query": (q0', q1)}`` (half-open prompt-token
    indices; ``q_tok[1] <= prompt_len_tpl`` since ``prompt_offsets`` carries one
    entry per prompt token), a drop-reason str (``empty_query_token_span`` /
    ``empty_prefix_token_span`` — dropped-and-counted; the latter measured
    impossible on this battery, min prefix 24 tokens), or raises
    ``RuntimeError`` when the probe is not found verbatim (kill criterion,
    plan §7).
    """
    q_char = prompt_text_tpl.rfind(probe)
    if q_char < 0:
        raise RuntimeError(
            "probe not found verbatim in templated prompt (plan §7 kill criterion): "
            f"probe {probe[:60]!r}…"
        )
    q_tok = char_span_to_token_span(prompt_offsets, (q_char, q_char + len(probe)))
    if q_tok == (0, 0):
        return "empty_query_token_span"  # dropped-and-counted
    if q_tok[0] == 0:
        return "empty_prefix_token_span"  # dropped-and-counted (measured: cannot fire)
    return {"prefix": (0, q_tok[0]), "query": q_tok}


# Condition slugs (plan §10) — regimes avg_q / avg_t / indiv.
ARM_SLUGS = (
    "d_ctx2ans",
    "d_parity",
    "a_ctx2cot",
    "b_cot2ans",
    "comp_pred",
    "j_joint",
    "g_aug",
    "ident",
)
REGIMES = ("avg_q", "avg_t", "indiv")

# Fit / null / bootstrap pins (inherited verbatim — plan §11: #810/#658 line).
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 42

# ── segmentation (Phase P) — code, not a model call (plan §4.4) ───────────────


def segment_completion(text: str, rung: str) -> tuple[bool, str, tuple[int, int], tuple[int, int]]:
    """Segment one completion into (cot_char_span, ans_char_span). Exact + deterministic.

    Returns ``(well_formed, reason, cot_span, ans_span)`` with char spans
    half-open ``[s, e)`` into ``text``. ``reason`` is "" when well-formed, else
    one of the enumerated malformed reasons (persisted in coverage counts).

    Rungs (i)/(ii) (``greedy``/``sample``): well-formed iff exactly one
    ``<think>`` at char position 0 AND exactly one ``</think>`` AND a non-empty
    answer after the closing tag (plan §4.4). CoT span = chars strictly inside
    the tags (delimiters excluded); answer span = chars after ``</think>`` +
    following whitespace.

    Rung (iii) (``prefill``): the prefilled ``<think>\\n`` lives in the PROMPT,
    so the criterion is adjusted per §4.3 — well-formed iff the completion
    contains exactly one ``</think>`` (NO ``<think>`` requirement); CoT span =
    start of completion through the char before ``</think>``.

    A whitespace-only CoT or answer segment is malformed (``empty_cot`` /
    ``empty_answer``) — every fit row needs all three parts.

    Segmentation stays purely STRUCTURAL: the ``degenerate_repetition``
    malformed class (v3 amendment, plan §4.4) is assigned downstream in the
    extractor's ``parse_rows`` (a well-formed row whose repeated-4-gram
    fraction exceeds ``REPEAT_4GRAM_MAX_FRAC``), never here.
    """
    assert rung in FALLBACK_RUNGS, rung
    n_open = text.count(THINK_OPEN)
    n_close = text.count(THINK_CLOSE)
    if n_close == 0:
        return False, "no_close", (0, 0), (0, 0)
    if n_close > 1:
        return False, "multiple_close", (0, 0), (0, 0)
    close_idx = text.index(THINK_CLOSE)
    if rung == "prefill":
        cot_s = 0
    else:
        if n_open == 0:
            return False, "no_think_open", (0, 0), (0, 0)
        if n_open > 1:
            return False, "multiple_think", (0, 0), (0, 0)
        if not text.startswith(THINK_OPEN):
            return False, "think_not_at_start", (0, 0), (0, 0)
        cot_s = len(THINK_OPEN)
    cot_e = close_idx
    if cot_e <= cot_s or not text[cot_s:cot_e].strip():
        return False, "empty_cot", (0, 0), (0, 0)
    ans_s = close_idx + len(THINK_CLOSE)
    while ans_s < len(text) and text[ans_s].isspace():
        ans_s += 1
    ans_e = len(text)
    if ans_s >= ans_e or not text[ans_s:ans_e].strip():
        return False, "empty_answer", (0, 0), (0, 0)
    return True, "", (cot_s, cot_e), (ans_s, ans_e)


_WORD_RE = re.compile(r"\S+")


def repeated_4gram_fraction(text: str) -> float:
    """Fraction of word-level 4-grams that are repeats (degeneration signature).

    0.0 for texts under ``REPEAT_CHECK_MIN_WORDS`` words (short replies are
    trivially non-degenerate; the check targets long repetition loops). A
    completion with fraction > ``REPEAT_4GRAM_MAX_FRAC`` is a repetition
    OFFENDER (per-row definition, unchanged from v2): ``parse_rows``
    reclassifies a segmentation-well-formed offender to
    ``degenerate_repetition`` (dropped + coverage-counted, plan §4.4), and
    Gate 1 fails only when the offender RATE over all gate rows exceeds
    ``REPEAT_OFFENDER_MAX_FRAC`` (plan §7, v3 amendment).
    """
    words = _WORD_RE.findall(text)
    if len(words) < REPEAT_CHECK_MIN_WORDS:
        return 0.0
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_span: tuple[int, int]
) -> tuple[int, int]:
    """Half-open token-index span of tokens OVERLAPPING ``char_span`` = [cs, ce).

    ``offsets`` is the tokenizer's ``offset_mapping`` for the completion text
    (one (s, e) char pair per token). A token is included iff its char span
    overlaps the target span with positive measure — robust to BPE merges where
    a delimiter's ``>`` fuses with trailing newlines (the #825 zero-width-span
    class). Returns (0, 0) when no token overlaps (caller drops the row with a
    counted reason — never a crash).
    """
    cs, ce = char_span
    lo, hi = None, None
    for i, (s, e) in enumerate(offsets):
        if e <= cs:
            continue
        if s >= ce:
            break
        if lo is None:
            lo = i
        hi = i + 1
    if lo is None:
        return (0, 0)
    return (lo, hi)


# ── inputs (battery + probe pool), reused + pinned (plan §4.2) ────────────────


def resolve_battery(local_hint: Path | None = None) -> dict:
    """Load + sha256-pin the 50-context battery (local fast path, else HF snapshot).

    Mirrors ``issue810_extract_positions._resolve_battery`` (the parent's
    fetchability contract): local ``data/issue594/battery.json`` is
    git-committed on this line (verified), but the sha256 is asserted against
    ``BATTERY50_SHA256`` either way (#600 HF-mirror guard); on a local miss the
    sha-pinned HF snapshot is fetched.
    """
    from huggingface_hub import hf_hub_download

    candidates = []
    if local_hint is not None:
        candidates.append(Path(local_hint))
    candidates.append(PROJECT_ROOT / "data" / "issue594" / "battery.json")
    for c in candidates:
        if c.is_file() and sha256_file(c) == BATTERY50_SHA256:
            return load_json(c)
    path = hf_hub_download(HF_DATA_REPO, BATTERY50_HF_FILE, repo_type="dataset")
    assert_sha256(path, BATTERY50_SHA256, "battery50")
    return load_json(path)


def load_probe_pool() -> list[str]:
    """The 48 fixed misalignment paraphrases, content-hash-asserted (plan §4.2).

    Code-derived exactly as #594 built them: the Betley preregistered
    paraphrase pool minus the main-8 eval questions (the published set supplies
    48). Asserted against ``issue658_common.I594_PROBE_POOL_HASH`` — fail loud
    on drift (the folds + parent-parity reads are pinned to this pool).
    """
    from issue404_common import fetch_betley_main_8, fetch_preregistered_probes
    from issue594_common import probes_hash
    from issue658_common import I594_PROBE_POOL_HASH

    main8 = fetch_betley_main_8()
    probes = fetch_preregistered_probes(n=200, exclude=set(main8))
    got = probes_hash(probes)
    if got != I594_PROBE_POOL_HASH:
        raise RuntimeError(
            f"probe pool hash drift: {got} != {I594_PROBE_POOL_HASH} — the #594/#658 "
            "48-probe battery changed upstream; refusing to run on a different pool"
        )
    if len(probes) != 48:
        raise RuntimeError(f"expected 48 probes, got {len(probes)}")
    return probes


def context_order_and_families(battery: dict) -> tuple[list[str], dict[str, str]]:
    """Battery-order context ids (the LOCO fold order) + per-context family (LOFO).

    The battery build is deterministic (seed 42), so battery order is the
    canonical fold order for this line (7 families: persona 14 / wildchat 10 /
    icl 8 / rephrase 6 / format 5 / behavior 5 / default 2).
    """
    instances = battery["instances"]
    ids = [i["id"] for i in instances]
    fams = {i["id"]: i["family"] for i in instances}
    assert len(set(ids)) == len(ids), "duplicate context ids in battery"
    return ids, fams


# ── poll_pipeline sentinel (shared by the extract + finalize entrypoints) ─────

SENTINEL_SCHEMA_VERSION = 1


def write_sentinel(kind: str, note: dict, fallback_dir: Path, log_dir: Path | None = None) -> Path:
    """poll_pipeline.py-conformant sentinel (issue-928 naming). Returns the path.

    Round-2 relocation from the extract script: the extract phase now emits an
    ``epm:progress`` sentinel and the run_all driver's finalize step emits the
    ONE ``epm:results`` sentinel at true end-of-workload (after fits + figures
    + uploads) — so both writers share this implementation. ``log_dir``
    overrides the ``/workspace/logs`` default (smoke runs redirect to scratch).
    """
    import logging
    import time

    slug = kind.replace(":", "_")
    base = log_dir if log_dir is not None else Path("/workspace/logs")
    try:
        base.mkdir(parents=True, exist_ok=True)
        target = base / f"issue-928-{slug}-{int(time.time())}.json"
    except OSError:
        target = fallback_dir / f"issue-928-{slug}-sentinel.json"
    dump_json(
        {
            "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
            "kind": kind,
            "version": 1,
            "note": note,
            "ts": int(time.time()),
        },
        target,
    )
    logging.getLogger("issue928_common").info("wrote sentinel %s", target)
    return target


# ── scoped HF upload verification (gotcha #833: never bare list_repo_files) ───


def upload_folder_scoped_verify(
    folder: Path,
    path_in_repo: str,
    expected_names: list[str],
    commit_message: str,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> str:
    """One ``upload_folder`` commit + SCOPED ``list_repo_tree`` exact-set verify.

    Never a per-file loop (the #664 504-storm) and never a bare
    ``list_repo_files`` full listing (the ~1M-file data repo times out, #833) —
    the verify enumerates ONLY ``path_in_repo`` server-side. Raises on any
    missing expected file (a partial upload is never silent success).
    ``ignore_patterns`` excludes nested scratch (fnmatch ``*`` crosses ``/``,
    so ``*.json`` would otherwise sweep in ``partial/<regime>/*.json`` resume
    checkpoints — round-2 restartability units are local/crash-trap state, not
    canonical Hub artifacts).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        commit_message=commit_message,
    )
    remote = {
        e.path
        for e in api.list_repo_tree(
            HF_DATA_REPO, path_in_repo=path_in_repo, repo_type="dataset", recursive=True
        )
    }
    expected = {f"{path_in_repo}/{n}" for n in expected_names}
    missing = expected - remote
    if missing:
        raise RuntimeError(
            f"upload verification FAILED under {path_in_repo}/: {len(missing)} expected "
            f"file(s) missing on the Hub (e.g. {sorted(missing)[:3]})"
        )
    return path_in_repo
