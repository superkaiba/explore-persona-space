#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ×, ｜, ▁) in the R1-distill template literals + docstrings.
"""Issue #1005 model profile: CoT-decomposition replication on DeepSeek-R1-Distill-Qwen-7B.

#1005 re-runs the #928 battery changing ONE variable — the model, to a thinking
model whose chat template FORCES the reasoning scaffold (`<｜Assistant｜><think>\\n`
is rendered into the PROMPT). This module is the plan §4.1 model/template
contract: the profile constants the issue928 machinery consumes through its
default-preserving keyword extensions, plus the fail-loud in-process startup
asserts and the v3 Gate-1 predicate (plan §7 — terminal conjuncts scoped so no
registered H2 lattice branch is foreclosed).

Every constant below was Hub-verified in the plan session at the pinned
revision (plan §2: config.json / generation_config.json / tokenizer.json /
chat template fetched + quoted); the startup asserts re-verify them in-process
before any GPU spend.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from issue928_common import (  # noqa: E402
    MLC_SUMMARY_NAMES,
    PARSE_RATE_FLOOR,
    PMA_SUMMARY_NAMES,
    REPEAT_4GRAM_MAX_FRAC,
    REPEAT_OFFENDER_MAX_FRAC,
    THINK_CLOSE,
    THINK_OPEN,
    matched_length_spans,
    prefix_query_spans,
)

# ── model/template contract (plan §4.1 — code, not a model call) ──────────────

THINKING_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_REVISION = "916b56a44061fd5cd7d6a8fb632557ed4f724f60"  # pinned (2025-02-24 main)
STOP_TOKEN_IDS = [151643]  # <｜end▁of▁sentence｜> (generation_config eos)
GENERATION_SUFFIX = "<｜Assistant｜><think>\n"  # last-3-token decode assert
TEMPLATE_FORCES_THINK = True  # parser: prefill semantics on every rung
PARSER_RUNG = "prefill"  # segment_completion criterion for ALL rungs (§4.4)
FALLBACK_RUNGS = ("greedy", "sample")  # no prefill rung — template already prefills
ANSWER_BOUNDARY_IDS = [151643]  # teacher-forced post-answer feed (ans_eos)
BOUNDARY_POSITIONS = {"ans_eos": 0}  # replaces ans_im_end; ans_turn_nl dropped (§4.0)
PROMPT_POSITIONS = {"ctx_assist": -3}  # <｜Assistant｜> = last pre-scaffold prompt token
BOS_TOKEN_ID = 151646  # assert prompt_ids[0] == 151646 and count == 1
THINK_OPEN_ID, THINK_CLOSE_ID = 151648, 151649  # in-process encode assert
EXPECTED_ARCH = "Qwen2ForCausalLM"

# The two context families whose parse coverage COLLAPSED on the parent
# (ICL 51% / WildChat 64% usable rows) — the H2 lattice's compliance read.
COLLAPSE_FAMILIES = ("icl", "wildchat")

# ── HF destinations (plan §10 output artifacts) ───────────────────────────────

HF_PREFIX_1005 = "issue1005_cot_decomposition_r1"
RAW_COMPLETIONS_PREFIX_1005 = f"{HF_PREFIX_1005}/raw_completions/thinking_rollouts"
STORE_PREFIX_1005 = f"{HF_PREFIX_1005}/analysis_tensors/store/percq_summaries"
FIT_RESULTS_PREFIX_1005 = f"{HF_PREFIX_1005}/fit_results"
DECOMP_TENSORS_PREFIX_1005 = f"{HF_PREFIX_1005}/analysis_tensors/decomp"
FIGURES_PREFIX_1005 = f"{HF_PREFIX_1005}/figures"
# The mlp phase's issue-profile HF paths (upload-verification v1 required
# action 3: the parent modules' module-level #928 constants must never reach an
# upload/stage call site under the #1005 profile — the driver threads THESE
# through issue928_mlp_indiv_control's CLI overrides instead). Shapes mirror
# the parent's derivations from HF_PREFIX_928 one-for-one (pinned by
# tests/test_issue1005_hf_prefixes.py).
MLP_INDIV_TENSORS_PREFIX_1005 = f"{HF_PREFIX_1005}/analysis_tensors/mlp_indiv"
MLP_INDIV_RESULTS_PREFIX_1005 = f"{FIT_RESULTS_PREFIX_1005}/indiv_mlp_control"
STORE_HF_ROOT_1005 = f"{HF_PREFIX_1005}/analysis_tensors/store"
DECOMP_INDIV_HF_PATH_1005 = f"{DECOMP_TENSORS_PREFIX_1005}/decomp_indiv.pt"
# #1005 artifacts live at the data repo's moving tip (the extractor/f1 upload
# to main in-run; there is no #928-style frozen pin for a mid-run fallback).
STORE_REVISION_1005 = "main"

# ── the unified 18-vector registry (plan §4.5) ────────────────────────────────
#
# 12 adjusted parent vectors (ans_eos replaces ans_im_end; ans_turn_nl dropped —
# no post-eos newline in this template; ctx_assist added, exploratory) + the 4
# matched-length-control vectors + the 2 prefix-convention vectors. Order is
# the store tensor's axis order; consumers index BY NAME via the manifest.
SUMMARY_NAMES_1005: tuple[str, ...] = (
    "ctx_mean",
    "ctx_max",
    "ctx_last",  # the newline after <think> (id 198) — the parent c_C analogue
    "ctx_assist",  # <｜Assistant｜> token (exploratory ctx-boundary robustness read)
    "cot_mean",
    "cot_max",
    "cot_last",  # last CoT content token (registered boundary)
    "cot_close",  # final token of </think> (exploratory)
    "ans_mean",
    "ans_max",
    "ans_last",  # last answer content token (registered boundary)
    "ans_eos",  # <｜end▁of▁sentence｜> appended after the answer (exploratory)
    "cot_lastK_mean",
    "cot_firstK_mean",
    "ansprefix_K_mean",
    "ans_rem_mean",
    "prefix_mean",
    "query_mean",
)
assert len(SUMMARY_NAMES_1005) == 18, len(SUMMARY_NAMES_1005)
# every MLC + PMA name the parent fit modules consume is present by name:
assert set(MLC_SUMMARY_NAMES) <= set(SUMMARY_NAMES_1005), set(MLC_SUMMARY_NAMES)
assert set(PMA_SUMMARY_NAMES) <= set(SUMMARY_NAMES_1005), set(PMA_SUMMARY_NAMES)

# Single-position names for reduce_forward_batch(position_names=...) (§4.1).
POSITION_NAMES_1005: tuple[str, ...] = (
    "ctx_last",
    "ctx_assist",
    "cot_last",
    "cot_close",
    "ans_last",
    "ans_eos",
)

# The 4 MLC names — rows failing the matched-length floor carry NaN in these
# slots of the unified store and are EXCLUDED from F2/F3 via `mlc_row_mask`
# (never dropped from the F1 battery — §4.0.2 no measured-quantity change).
MLC_NAMES_1005: tuple[str, ...] = (
    "cot_lastK_mean",
    "cot_firstK_mean",
    "ansprefix_K_mean",
    "ans_rem_mean",
)
BASE_NAMES_1005: tuple[str, ...] = tuple(n for n in SUMMARY_NAMES_1005 if n not in MLC_NAMES_1005)
MLC_ROW_MASK_KEY = "mlc_row_mask"

# ── prompt building + startup asserts (fail-loud before any GPU spend) ───────


def build_prompt_ids(tokenizer, instance: dict, probe: str) -> tuple[str, list[int]]:
    """Templated prompt (text, token ids) under the #1005 contract.

    Tokenizes with ``add_special_tokens=False`` (the template embeds
    ``{{bos_token}}`` — the R1 tokenizer's ``add_bos_token: true`` would
    otherwise DOUBLE the bos) and asserts exactly one bos at position 0 plus
    the forced-scaffold suffix on EVERY prompt. Generation consumes the IDS
    (``{"prompt_token_ids": ids}`` — vLLM's own text tokenization would re-add
    specials), so generation and teacher-forced capture are token-identical.
    """
    from issue594_common import messages_for_instance

    messages = messages_for_instance(instance, probe)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n_bos = sum(1 for i in ids if i == BOS_TOKEN_ID)
    assert ids[0] == BOS_TOKEN_ID and n_bos == 1, (
        f"bos contract violated for context={instance['id']}: ids[0]={ids[0]}, n_bos={n_bos} "
        f"(want exactly one {BOS_TOKEN_ID} at position 0 — plan §4.1)"
    )
    suffix = tokenizer.decode(ids[-3:])
    assert suffix == GENERATION_SUFFIX, (
        f"forced-scaffold suffix assert failed for context={instance['id']}: "
        f"{suffix!r} != {GENERATION_SUFFIX!r}"
    )
    return text, ids


def run_startup_asserts(tokenizer, battery: dict, config=None) -> dict:
    """The plan §4.1 in-process startup asserts. Returns a small report dict.

    - AutoConfig: 28 layers / 3584 hidden / Qwen2ForCausalLM (when ``config``
      is passed; the smoke's tiny from-config model relaxes shape asserts at
      the DRIVER level exactly as the parent's ``--smoke`` does — the
      TOKENIZER asserts below run unconditionally in both modes).
    - ``encode("<think>") == [151648]``, ``encode("</think>") == [151649]``.
    - Templated-prompt last-3-token decode == GENERATION_SUFFIX + exactly one
      bos, checked on one instance PER FAMILY (the per-prompt assert in
      ``build_prompt_ids`` covers every prompt at build time).
    - No-think-markup round-trip: no battery prefix message contains
      ``</think>`` (the template's ``</think>``-split of prior assistant turns
      must be a no-op on this battery).
    """
    if config is not None:
        assert config.architectures == [EXPECTED_ARCH], config.architectures
        assert config.num_hidden_layers == 28, config.num_hidden_layers
        assert config.hidden_size == 3584, config.hidden_size
    got_open = tokenizer.encode(THINK_OPEN, add_special_tokens=False)
    got_close = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
    assert got_open == [THINK_OPEN_ID], f"<think> id drift: {got_open} != [{THINK_OPEN_ID}]"
    assert got_close == [THINK_CLOSE_ID], f"</think> id drift: {got_close} != [{THINK_CLOSE_ID}]"

    instances = battery["instances"]
    n_prefix_msgs = 0
    for inst in instances:
        for msg in inst.get("prefix_messages") or []:
            n_prefix_msgs += 1
            assert THINK_CLOSE not in msg.get("content", ""), (
                f"battery prefix message in {inst['id']} contains {THINK_CLOSE!r} — the "
                "template's prior-turn </think>-split would NOT be a no-op (plan §4.1)"
            )
    seen_families: set[str] = set()
    probe = "What is your favorite color?"  # any final user turn works for the render check
    for inst in instances:
        if inst["family"] in seen_families:
            continue
        seen_families.add(inst["family"])
        build_prompt_ids(tokenizer, inst, probe)  # asserts suffix + exactly-one-bos
    return {
        "think_ids": [THINK_OPEN_ID, THINK_CLOSE_ID],
        "families_render_checked": sorted(seen_families),
        "n_prefix_messages_checked": n_prefix_msgs,
    }


# ── parts-spec adapters (threaded into build_capture_row — plan §4.5) ─────────


def mlc_parts_spec_1005(box: dict):
    """``parts_spec`` adapter: matched-length spans, floor-failure KEPT.

    Unlike the parent MLC round (whose floor-failing rows were dropped from a
    DEDICATED store), the unified #1005 capture keeps floor-failing rows in
    the battery with NaN MLC slots — so the spec returns ``{}`` (no extra
    spans) and records the miss in ``box["mlc_ok"]`` instead of dropping the
    row (plan §4.0.2: consolidation must not change F1's row set). One fresh
    ``box`` per row.
    """

    def spec(cot_tok: tuple[int, int], ans_tok: tuple[int, int]):
        spans = matched_length_spans(cot_tok, ans_tok)
        if spans is None:
            box["mlc_ok"] = False
            return {}
        box["mlc_ok"] = True
        return {k: v for k, v in spans.items() if k != "K"}

    return spec


def prompt_parts_spec_1005(probe: str):
    """``prompt_parts_spec`` adapter: the parent v7 prefix/query spans, per row."""

    def spec(prompt_text_tpl: str, prompt_offsets, prompt_len_tpl: int):
        return prefix_query_spans(prompt_text_tpl, prompt_offsets, prompt_len_tpl, probe)

    return spec


# ── synthetic completions (CPU smoke ONLY — the vLLM-boundary fake) ───────────


def synthetic_completions_1005(prompts: list, n_probes: int) -> list[tuple[str, str]]:
    """Deterministic prefill-shaped synthetic completions for the CPU smoke.

    The #1005 sibling of ``issue928_extract_thinking_store.synthetic_completions``:
    completions carry ONLY ``</think>`` (the template forces the open tag into
    the PROMPT — prefill semantics on every rung), and the well-formed answers
    are LONG enough to clear the matched-length floors (K >= 8, rem >= 16 — the
    parent generator's ~15-token answers would floor-fail every row and zero
    out the F2/F3 smoke). Mix: mostly well-formed; one truncated row per ~24
    (truncation accounting); one no-``</think>`` row per ~12 (drop/coverage
    path); one degenerate-repetition row per ~48 (offender-rate conjunct); one
    SHORT-answer row per ~6 (MLC-floor fail — kept with NaN MLC slots,
    exercising the ``mlc_row_mask`` path). NEVER used in production.
    """
    out: list[tuple[str, str]] = []
    for i in range(len(prompts)):
        body = " ".join(f"r{i}s{j} reasoning step token" for j in range(12))
        long_ans = " ".join(f"a{i}w{j} answer word here" for j in range(14))
        if i % 24 == 7:
            out.append((f"{body} and it keeps going", "length"))  # truncated_no_close
        elif i % 12 == 3:
            out.append((f"{body} {long_ans}", "stop"))  # malformed: no_close
        elif i % 48 == 19:
            loop = " ".join(["repeat the same loop words"] * 20)
            out.append((f"{loop}\n{THINK_CLOSE}\n\n{long_ans}", "stop"))  # degenerate
        elif i % 6 == 5:
            out.append((f"{body}\n{THINK_CLOSE}\n\nShort answer {i}.", "stop"))  # floor fail
        else:
            out.append((f"{body}\n{THINK_CLOSE}\n\n{long_ans}", "stop"))
    _ = n_probes
    return out


# ── Gate 1 (plan §7 v3 — scoped terminal conjuncts) ───────────────────────────


def select_gate_slice(ctx_ids: list[str], families: dict[str, str]) -> dict:
    """The Phase-0 gate slice: first 3 NON-collapse contexts + first ICL +
    first WildChat context, in battery order, from the RUN's ctx subset.

    A tiny smoke subset may lack ICL/WildChat contexts — the coverage READ
    then records them absent (never a gate input; production's 50-context run
    always yields exactly 3 + 1 + 1)."""
    non_collapse = [c for c in ctx_ids if families[c] not in COLLAPSE_FAMILIES][:3]
    icl = [c for c in ctx_ids if families[c] == "icl"][:1]
    wildchat = [c for c in ctx_ids if families[c] == "wildchat"][:1]
    return {
        "slice": non_collapse + icl + wildchat,
        "non_collapse": non_collapse,
        "collapse_read": icl + wildchat,
    }


def gate1005_check(
    rows_by_ctx: dict[str, list[dict]],
    families: dict[str, str],
    cap: int,
    slice_info: dict,
) -> dict:
    """Gate 1, plan §7 v3. Returns the gate report.

    - **Terminal conjunct A** (wholesale-failure detection): usable-row rate
      over the NON-collapse slice contexts >= PARSE_RATE_FLOOR (0.80).
    - **Terminal conjunct B** (degeneration): repetition-offender RATE over
      the FULL slice <= REPEAT_OFFENDER_MAX_FRAC (0.10) — computed from
      ``rep_frac`` over ALL rows, independent of reason bookkeeping.
    - **Non-terminal conjunct C** (length): p95 gen-token count over the full
      slice < ``cap``. A C-only fail escalates to ONE 16,384 slice re-measure
      at the DRIVER level (never a rung walk).
    - **Early coverage READ** (never a kill predicate): per-context usable
      rates of the ICL/WildChat slice contexts, recorded for H2.
    """
    import numpy as np

    def usable_rate(ctxs: list[str]) -> float:
        rows = [r for c in ctxs for r in rows_by_ctx[c]]
        return sum(1 for r in rows if r["well_formed"]) / max(1, len(rows))

    all_rows = [r for c in slice_info["slice"] for r in rows_by_ctx[c]]
    a_rate = usable_rate(slice_info["non_collapse"])
    a_pass = a_rate >= PARSE_RATE_FLOOR
    offenders = sum(1 for r in all_rows if r["rep_frac"] > REPEAT_4GRAM_MAX_FRAC)
    b_rate = offenders / max(1, len(all_rows))
    b_pass = b_rate <= REPEAT_OFFENDER_MAX_FRAC
    p95 = float(np.percentile([r["n_gen_tokens"] for r in all_rows], 95)) if all_rows else 0.0
    c_pass = p95 < cap
    reasons: dict[str, int] = {}
    for r in all_rows:
        if not r["well_formed"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    coverage_read = {
        c: {
            "family": families[c],
            "usable_rate": usable_rate([c]),
            "n_rows": len(rows_by_ctx[c]),
        }
        for c in slice_info["collapse_read"]
    }
    return {
        "terminal_pass": bool(a_pass and b_pass),
        "conjunct_a": {
            "pass": bool(a_pass),
            "usable_rate_non_collapse": a_rate,
            "floor": PARSE_RATE_FLOOR,
            "contexts": slice_info["non_collapse"],
        },
        "conjunct_b": {
            "pass": bool(b_pass),
            "offender_rate": b_rate,
            "offenders": offenders,
            "max_frac": REPEAT_OFFENDER_MAX_FRAC,
        },
        "conjunct_c": {"pass": bool(c_pass), "p95_gen_tokens": p95, "cap": cap},
        "collapse_family_coverage_read": coverage_read,
        "n_rows": len(all_rows),
        "malformed_reasons": reasons,
    }
