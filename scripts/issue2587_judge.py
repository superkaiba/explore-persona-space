"""#2587 P9 manipulation-check judge — pinned #2564 port + answer_language pilot rubric.

Port of ``scripts/issue2564_judge.py`` @ the frozen parent pin
``8265bcd75f781d8e879e924de60063e536e58dcf`` (blob sha256
``75b7de5185f5871d436b1c66298524c22c22c90a47eee4e77719bca1b14c5c34``; read via
``git show``, never a checkout), re-keyed to issue 2587 (plan v3 §4.4 / §4.7 P9):

- Parent bank access goes through ``bank2587._bk()`` — the sha-asserted pinned
  ``bank2564`` module — so rubrics, value strings, and context ids are the
  PIN's, never main-resident ``bank2564``'s.
- Compliance rubric ``EVAL_PROMPT`` is ported VERBATIM from the pin (sha-pinned
  in ``tests/test_issue2587_judge.py``); 29 values + 29 paraphrases × 12
  carriers × 2 draws = 1,392 Sonnet Batch-API calls.
- NEW ``answer_language`` pilot instrument ``ANSWER_LANGUAGE_EVAL_PROMPT``
  (WARN-2: no committed rubric exists anywhere; drafted here in the same
  0-100 family, sha-pinned + text-persisted in the output artifact so the
  eventual 7B-side parity check can consume it VERBATIM): 3 values × 12
  carriers × 2 draws = 72 calls. Total 1,464 — VERIFIED against the realized
  bank at run time (``verify_call_arithmetic``), never trusted as a literal.
- Pilot-axis rows are MECHANICALLY labeled ``pilot_axis: true`` +
  ``cross_model_status: "7B side pending #2564"`` — 9B-only reads carry no
  numeric field a downstream consumer could mistake for a cross-model value.
- Judge routing: ``eval.graded_judge.judge_graded`` with ``threshold_base=0``
  (FORCES the #663-hardened Batch path), ``claude-sonnet-4-5-20250929``,
  ``max_tokens=1024``; drop classes (content / transport / refusal /
  truncation / api-refusal) persisted SEPARATELY per rubric family, never
  conflated; per-rubric cache subdirs keep the cache rubric-keyed.
- Fire decisions are the parent's verbatim: fixed denominator 24 (judged) /
  120 (programmatic), integer threshold arithmetic at 70%, MANDATORY
  ``undetermined`` on ANY incomplete check (never shrinks the denominator),
  axis floor ``ceil(0.6 × width)`` over base values.

1,464 ≪ 5,000 ⇒ pilot-gate exempt (llm-judging rule 26); the per-arm drop
report + the zero-``max_tokens``-stop property are recorded per wave post-hoc.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.experiments.issue2587 import bank2587 as B25  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

ISSUE = 2587
PARENT_PIN = B25.PIN  # 8265bcd75f781d8e879e924de60063e536e58dcf
HF_DATA_REPO = os.environ.get("EPM_2587_DATA_WRITE_REPO", "superkaiba1/explore-persona-space-data")
HF_PREFIX = "issue2587_minpair"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
JUDGE_MAX_TOKENS = 1024  # plan §11 pin (generous rationale-sized floor, llm-judging rule 23)
PROGRAMMATIC_AXES: tuple[str, ...] = ("lexical_marker", "user_fact")
LANG_AXIS = "answer_language"  # pilot judged axis; values from B25.LANG_VALUES
JUDGED_DRAWS: tuple[int, ...] = (0, 1)  # plan §4.4: 2 rollout draws judged
PROG_DRAWS: tuple[int, ...] = tuple(range(10))  # all 10 draws, free
FIRE_THRESHOLD_PCT = 70
SENSITIVITY_PCTS: tuple[int, ...] = (50, 90)
FLOOR_FRAC = 0.6
# Smoke slice covers BOTH rubric families (plan blind-spot line 144): register
# is the smallest judged parent axis (width 2); answer_language is the pilot.
SMOKE_CELLS: tuple[str, ...] = ("register", LANG_AXIS)
SMOKE_CARRIERS: tuple[str, ...] = ("c01", "c02", "c03")
SMOKE_JUDGE_ITEMS = 4  # PER RUBRIC FAMILY (both families must reach the Batch client)
DEFAULT_OUT = "eval_results/issue_2587/manipulation_check_2587.json"
DEFAULT_WORK_ROOT = "data/issue_2587/judge_work"
SMOKE_ROOT = "/tmp/issue2587_judge_smoke"
PILOT_LABEL = "7B side pending #2564"

# Plan §4.4 call arithmetic — VERIFIED against the realized bank in main().
EXPECTED_COMPLIANCE_CALLS = 1392  # (29 values + 29 paraphrases) × 12 carriers × 2 draws
EXPECTED_LANG_CALLS = 72  # 3 values × 12 carriers × 2 draws
EXPECTED_TOTAL_CALLS = 1464

# custom_id budget: batch_judge appends "__{idx:05d}__{comp:02d}" (11 chars) to a
# 64-char cap => alias budget 53; charset ^[a-zA-Z0-9_-]$ (#1776).
_ALIAS_RE = re.compile(r"^[a-zA-Z0-9_-]{1,53}$")

# {question} carries the VERBATIM instruction string; {answer} the anchor text.
# graded_judge appends its own {"score": N}/{"score": "REFUSAL"} JSON wrapper
# as the system prompt — this template is the user message.
# PORTED VERBATIM from issue2564_judge.py @ PARENT_PIN (instrument identity for
# the q25-vs-q35 cross-model contrast; sha-pinned in tests/test_issue2587_judge.py).
EVAL_PROMPT = """\
You are checking whether a model's answer COMPLIES with a specific instruction that was \
present in the model's system prompt.

Instruction: {question}

Model answer:
<answer>
{answer}
</answer>

Does this answer comply with the instruction above? Score 0-100: 0 = no trace of the \
instruction being followed; 50 = ambiguous or partial compliance; 100 = the answer clearly \
and consistently follows the instruction throughout. Judge ONLY compliance with the stated \
instruction — ignore answer quality, correctness, and any style dimension the instruction \
does not name."""

# NEW instrument (WARN-2): the answer_language pilot rubric. Same 0-100 family
# as EVAL_PROMPT; {question} carries the language instruction verbatim
# ("Answer in English." / "Answer in Chinese." / "Answer in Spanish.").
# The 7B side of this axis is NOT judged here — the parent (#2564) consumes
# THIS text verbatim when its pilot judging lands (sha + verbatim text ride
# the output artifact's instrument_identity block; drift fails the sha pin in
# tests/test_issue2587_judge.py).
ANSWER_LANGUAGE_EVAL_PROMPT = """\
You are checking whether a model's answer COMPLIES with a language instruction that was \
present in the model's system prompt.

Instruction: {question}

Model answer:
<answer>
{answer}
</answer>

Score 0-100 for whether the answer is written ENTIRELY in the language the instruction \
names: 0 = none of the answer is in that language; 50 = roughly half the answer is in that \
language, or it is ambiguous which language dominates; 100 = the answer is written entirely \
in that language throughout. Judge ONLY the language the answer is written in — ignore \
answer quality, correctness, length, and whether the answer addresses the question. Code \
snippets, proper nouns, and universally-shared symbols (numbers, punctuation, units) do \
not count against the language."""


def log(msg: str) -> None:
    """Flush-immediate stdout log line."""
    print(msg, flush=True)


def rubric_sha256(text: str) -> str:
    """Instrument identity: sha256 of the rubric template's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic JSON write via a process-unique temp (atomic_io.atomic_replace, #2336)."""
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL via split("\\n") — NEVER splitlines() (U+2028 shred, #950)."""
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ── pure check / fire-decision helpers (CPU-pinned in tests/test_issue2587_judge.py) ──


def check_contains_word(text: str, word: str) -> bool:
    """Case-insensitive, word-boundary-delimited containment.

    ``\\b``-delimited so 'surelyX' / 'leisurely' do NOT match 'surely', while
    punctuation-adjacent occurrences ('Surely,' / "Marcus's") DO match.
    """
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def axis_floor(width: int) -> int:
    """Headline axis floor: ceil(0.6 × realized width) fired base values."""
    return math.ceil(FLOOR_FRAC * width)


def fire_verdict(
    n_comply: int, n_incomplete: int, denom: int, threshold_pct: int = FIRE_THRESHOLD_PCT
) -> str:
    """Three-way fire decision on the FIXED denominator (parent verbatim).

    MANDATORY-undetermined: a slot with ANY incomplete check after the judge
    retry budget is ``undetermined`` — counted as not-fired for the axis
    floor, kept in the denominator — regardless of whether the incompletes
    could flip the verdict. Otherwise integer arithmetic (no float
    thresholds): fired iff ``n_comply * 100 >= threshold_pct * denom``
    (≥70% of 24 ⇒ ≥17; of 120 ⇒ ≥84). Raw counts are persisted per slot so a
    looser decision-relevance-only reading stays recomputable downstream.
    """
    if n_comply < 0 or n_incomplete < 0 or n_comply + n_incomplete > denom or denom <= 0:
        raise ValueError(
            f"bad fire counts: comply={n_comply} incomplete={n_incomplete} denom={denom}"
        )
    if n_incomplete > 0:
        return "undetermined"
    if n_comply * 100 >= threshold_pct * denom:
        return "fired"
    return "not_fired"


def _value_row(
    axis: str,
    value_id: str,
    kind: str,
    instrument: str,
    n_comply: int,
    n_noncomply: int,
    n_incomplete: int,
    denom: int,
) -> dict:
    """One fire-table row; denom is FIXED (never shrunken) and must reconcile."""
    assert n_comply + n_noncomply + n_incomplete == denom, (
        axis,
        value_id,
        n_comply,
        n_noncomply,
        n_incomplete,
        denom,
    )
    return {
        "axis": axis,
        "value_id": value_id,
        "kind": kind,
        "instrument": instrument,
        "n_comply": n_comply,
        "n_noncomply": n_noncomply,
        "n_incomplete": n_incomplete,
        "denom": denom,
        "comply_frac": n_comply / denom,
        "verdict": fire_verdict(n_comply, n_incomplete, denom),
        "sensitivity": {
            str(pct): fire_verdict(n_comply, n_incomplete, denom, threshold_pct=pct)
            for pct in SENSITIVITY_PCTS
        },
    }


def annotate_pilot_rows(rows: list[dict]) -> list[dict]:
    """MECHANICAL pilot labeling (WARN-2): every ``answer_language`` /
    ``query_content_oneword`` row carries ``pilot_axis: true`` +
    ``cross_model_status: "7B side pending #2564"``; parent-axis rows carry
    ``pilot_axis: false``. Mutates in place and returns ``rows``; raises on a
    row with no ``axis`` key (fail loud, never a silent default)."""
    for r in rows:
        axis = r["axis"]  # KeyError = fail loud
        if axis in (LANG_AXIS, "query_content_oneword"):
            r["pilot_axis"] = True
            r["cross_model_status"] = PILOT_LABEL
        else:
            r["pilot_axis"] = False
    return rows


# ── spec enumeration (from the REAL pinned bank) ──────────────────────


def parent_judged_axes(bk) -> tuple[str, ...]:
    """The 7 judged parent instruction axes (INSTRUCTION_AXES minus programmatic)."""
    return tuple(a for a in bk.INSTRUCTION_AXES if a not in PROGRAMMATIC_AXES)


def judged_value_slots(bk, values: dict, axes: tuple[str, ...]) -> list[dict]:
    """Judged value slots for the given parent axes: base values + paraphrases.

    Full production shape: 29 base + 29 paraphrases across the 7 judged axes.
    """
    slots = []
    for axis in axes:
        for vid in bk.value_ids(values, axis):
            slots.append(
                {
                    "axis": axis,
                    "value_id": vid,
                    "kind": "orig",
                    "instruction": bk.system_string(values, axis, vid),
                }
            )
            slots.append(
                {
                    "axis": axis,
                    "value_id": f"{vid}p",
                    "kind": "para",
                    "instruction": bk.paraphrase_string(values, axis, vid),
                }
            )
    return slots


def _alias(axis: str, value_id: str, carrier: str, draw: int) -> str:
    """Batch-legal custom_id alias for one check (bijective; asserted below)."""
    return f"{axis}--{value_id}--{carrier}-d{draw}"


def _validated_specs(raw: list[dict], what: str) -> list[dict]:
    """Alias grammar + collision assertion over one FULL realized spec set (#1776)."""
    for s in raw:
        if not _ALIAS_RE.match(s["alias"]) or "__" in s["alias"]:
            raise ValueError(f"illegal batch alias in {what}: {s['alias']!r}")
    aliases = [s["alias"] for s in raw]
    if len(set(aliases)) != len(aliases):
        raise ValueError(f"batch alias collision in {what} spec set")
    return raw


def judged_specs(
    bk,
    values: dict,
    carriers: tuple[str, ...],
    axes: tuple[str, ...],
    draws: tuple[int, ...] = JUDGED_DRAWS,
) -> list[dict]:
    """One judged compliance check per (value slot × carrier × rollout draw).

    Production shape: 58 slots × 12 carriers × 2 draws = 1,392 checks.
    Aliases are validated against the Batch custom_id grammar (charset +
    53-char budget + no ``__``) and asserted collision-free (#1776).
    """
    specs = []
    for slot in judged_value_slots(bk, values, axes):
        for carrier in carriers:
            cid = bk.context_id(slot["axis"], slot["value_id"], carrier)
            for draw in draws:
                specs.append(
                    {
                        **slot,
                        "carrier": carrier,
                        "draw": draw,
                        "context_id": cid,
                        "alias": _alias(slot["axis"], slot["value_id"], carrier, draw),
                    }
                )
    return _validated_specs(specs, "judged")


def lang_specs(
    bk,
    carriers: tuple[str, ...],
    draws: tuple[int, ...] = JUDGED_DRAWS,
) -> list[dict]:
    """One answer_language check per (language value × carrier × rollout draw).

    Production shape: 3 values × 12 carriers × 2 draws = 72 checks. NO
    paraphrase family (the langow pilot construction has none); the ``bare``
    contexts are not judged (no instruction to comply with). The instruction
    string IS the pilot system string (``B25.LANG_VALUES``), so {question}
    carries it verbatim — the 7B-side parity check reuses the identical
    (rubric, instruction) pair.
    """
    specs = []
    for lang, instruction in B25.LANG_VALUES.items():
        for carrier in carriers:
            cid = bk.context_id(LANG_AXIS, lang, carrier)
            for draw in draws:
                specs.append(
                    {
                        "axis": LANG_AXIS,
                        "value_id": lang,
                        "kind": "orig",
                        "instruction": instruction,
                        "carrier": carrier,
                        "draw": draw,
                        "context_id": cid,
                        "alias": _alias(LANG_AXIS, lang, carrier, draw),
                    }
                )
    return _validated_specs(specs, "answer_language")


def programmatic_specs(
    bk,
    values: dict,
    carriers: tuple[str, ...],
    draws: tuple[int, ...] = PROG_DRAWS,
) -> list[dict]:
    """One programmatic check per (value slot × carrier × draw), with the target word.

    The target word/name for a paraphrase slot is its BASE value's payload
    (the paraphrase rewords the instruction; the target token is unchanged).
    """
    specs = []
    for axis in PROGRAMMATIC_AXES:
        for vid in bk.value_ids(values, axis):
            word = values["axes"][axis]["values"][vid]
            for kind, slot_vid in (("orig", vid), ("para", f"{vid}p")):
                for carrier in carriers:
                    cid = bk.context_id(axis, slot_vid, carrier)
                    for draw in draws:
                        specs.append(
                            {
                                "axis": axis,
                                "value_id": slot_vid,
                                "kind": kind,
                                "carrier": carrier,
                                "draw": draw,
                                "context_id": cid,
                                "word": word,
                            }
                        )
    return specs


def verify_call_arithmetic(n_compliance: int, n_lang: int) -> dict:
    """Plan §4.4 arithmetic verified against the REALIZED bank (never the literal).

    Raises RuntimeError naming the discrepancy when the realized full-slice
    spec counts diverge from 1,392 + 72 = 1,464.
    """
    realized = {
        "compliance_calls": n_compliance,
        "answer_language_calls": n_lang,
        "total_calls": n_compliance + n_lang,
    }
    expected = {
        "compliance_calls": EXPECTED_COMPLIANCE_CALLS,
        "answer_language_calls": EXPECTED_LANG_CALLS,
        "total_calls": EXPECTED_TOTAL_CALLS,
    }
    if realized != expected:
        raise RuntimeError(
            f"call-arithmetic mismatch vs plan §4.4: realized={realized} expected={expected}"
        )
    return {"expected": expected, "realized": realized, "verified": True}


# ── fire tables ───────────────────────────────────────────────────────


def judged_fire_table(
    specs: list[dict],
    scores: dict[str, float | None],
    carriers: tuple[str, ...],
    draws: tuple[int, ...],
    instrument: str = "judged",
) -> list[dict]:
    """Per-slot fire rows from judge scores (alias → mean score or None).

    An alias absent from ``scores`` (missing anchor row / capped out of a
    smoke wave) or scored None (all draws dropped post-retry) is INCOMPLETE.
    The denominator is FIXED at len(carriers) × len(draws) per slot.
    """
    denom = len(carriers) * len(draws)
    by_slot: dict[tuple[str, str, str], list[dict]] = {}
    for s in specs:
        by_slot.setdefault((s["axis"], s["value_id"], s["kind"]), []).append(s)
    rows = []
    for (axis, value_id, kind), slot_specs in sorted(by_slot.items()):
        assert len(slot_specs) == denom, (axis, value_id, len(slot_specs), denom)
        n_comply = n_noncomply = n_incomplete = 0
        for s in slot_specs:
            score = scores.get(s["alias"])
            if score is None:
                n_incomplete += 1
            elif score >= 50.0:
                n_comply += 1
            else:
                n_noncomply += 1
        rows.append(
            _value_row(axis, value_id, kind, instrument, n_comply, n_noncomply, n_incomplete, denom)
        )
    return rows


def programmatic_fire_table(
    specs: list[dict],
    text_by_key: dict[tuple[str, int], str],
    carriers: tuple[str, ...],
    draws: tuple[int, ...],
) -> list[dict]:
    """Per-slot fire rows from word-containment checks over anchor texts.

    A (context_id, draw) with no anchor row is INCOMPLETE (counted, reported
    — should be zero in production, where the battery gates on complete cells).
    """
    denom = len(carriers) * len(draws)
    by_slot: dict[tuple[str, str, str], list[dict]] = {}
    for s in specs:
        by_slot.setdefault((s["axis"], s["value_id"], s["kind"]), []).append(s)
    rows = []
    for (axis, value_id, kind), slot_specs in sorted(by_slot.items()):
        assert len(slot_specs) == denom, (axis, value_id, len(slot_specs), denom)
        n_comply = n_noncomply = n_incomplete = 0
        for s in slot_specs:
            text = text_by_key.get((s["context_id"], s["draw"]))
            if text is None:
                n_incomplete += 1
            elif check_contains_word(text, s["word"]):
                n_comply += 1
            else:
                n_noncomply += 1
        rows.append(
            _value_row(
                axis, value_id, kind, "programmatic", n_comply, n_noncomply, n_incomplete, denom
            )
        )
    return rows


def axis_summary(value_rows: list[dict], axis: str, width: int, has_para: bool = True) -> dict:
    """Per-axis floor verdict over BASE values (paraphrase slots excluded).

    ``undetermined`` counts as not-fired for the floor. Sensitivity floors
    recompute the fired count at each alternate comply threshold.
    ``has_para=False`` (the answer_language pilot axis) reports
    ``n_fired_para: None`` — never a 0 that reads as "0 paraphrases fired".
    """
    base = [r for r in value_rows if r["axis"] == axis and r["kind"] == "orig"]
    para = [r for r in value_rows if r["axis"] == axis and r["kind"] == "para"]
    assert len(base) == width, (axis, len(base), width)
    if not has_para:
        assert not para, (axis, "unexpected paraphrase rows on a no-para axis")
    floor = axis_floor(width)
    n_fired = sum(1 for r in base if r["verdict"] == "fired")
    return {
        "axis": axis,
        "width": width,
        "floor": floor,
        "n_fired_base": n_fired,
        "n_undetermined_base": sum(1 for r in base if r["verdict"] == "undetermined"),
        "n_not_fired_base": sum(1 for r in base if r["verdict"] == "not_fired"),
        "floor_met": n_fired >= floor,
        "n_fired_para": sum(1 for r in para if r["verdict"] == "fired") if has_para else None,
        "sensitivity": {
            str(pct): {
                "n_fired_base": sum(1 for r in base if r["sensitivity"][str(pct)] == "fired"),
                "floor_met": sum(1 for r in base if r["sensitivity"][str(pct)] == "fired") >= floor,
            }
            for pct in SENSITIVITY_PCTS
        },
    }


def per_axis_drop_report(specs: list[dict], scores: dict[str, float | None]) -> dict:
    """Per-arm (per-axis) drop report for one judged rubric family (post-hoc
    binding, llm-judging rule 26): specs vs scored vs incomplete counts."""
    out: dict[str, dict] = {}
    for s in specs:
        d = out.setdefault(s["axis"], {"n_specs": 0, "n_scored": 0, "n_incomplete": 0})
        d["n_specs"] += 1
        if scores.get(s["alias"]) is None:
            d["n_incomplete"] += 1
        else:
            d["n_scored"] += 1
    return out


# ── anchors ingestion ─────────────────────────────────────────────────


def anchors_rel(cell: str) -> str:
    """Repo-relative anchors path for one cell (the battery-run layout)."""
    return f"raw_completions/anchors/anchors_{cell}.jsonl"


def stage_anchor_cells(
    cells: tuple[str, ...],
    anchors_dir: Path | None,
    hf_prefix: str,
    staging_dir: Path,
) -> dict[str, Path]:
    """Resolve each cell's anchors JSONL: local dir first, else HF fetch (retried)."""
    from huggingface_hub import hf_hub_download

    out: dict[str, Path] = {}
    for cell in cells:
        if anchors_dir is not None:
            local = anchors_dir / f"anchors_{cell}.jsonl"
            if local.is_file():
                out[cell] = local
                continue
        fn = f"{hf_prefix}/{anchors_rel(cell)}"
        got = hub.retry_transient(
            lambda fn=fn: hf_hub_download(
                HF_DATA_REPO, filename=fn, repo_type="dataset", local_dir=str(staging_dir)
            ),
            what=f"hf_hub_download({fn})",
        )
        out[cell] = Path(got)
    return out


def load_anchor_texts(paths: dict[str, Path]) -> dict[tuple[str, int], str]:
    """(context_id, draw) → completion text; an empty cell file raises (fail loud)."""
    texts: dict[tuple[str, int], str] = {}
    for cell, p in sorted(paths.items()):
        rows = _read_jsonl(p)
        if not rows:
            raise RuntimeError(f"anchors file for cell {cell!r} is EMPTY: {p}")
        for r in rows:
            texts[(r["context_id"], int(r["draw"]))] = r["text"]
    if not texts:
        raise RuntimeError("empty anchor selection — nothing to check")
    return texts


# ── judge dispatch (one wave per rubric family) ───────────────────────


def _build_wave_items(
    specs: list[dict],
    texts: dict[tuple[str, int], str],
    max_items: int | None,
) -> tuple[list[tuple[str, str, str]], int, int]:
    """Items (alias, instruction, text) sorted by alias; missing-anchor +
    capped-out counts. The cap applies PER WAVE so a capped smoke still
    exercises every rubric family (plan blind-spot line 144)."""
    items: list[tuple[str, str, str]] = []
    n_missing = 0
    for s in sorted(specs, key=lambda s: s["alias"]):
        text = texts.get((s["context_id"], s["draw"]))
        if text is None:
            n_missing += 1
            continue
        items.append((s["alias"], s["instruction"], text))
    n_capped = 0
    if max_items is not None and len(items) > max_items:
        n_capped = len(items) - max_items
        items = items[:max_items]
    return items, n_missing, n_capped


def _dispatch_wave(
    wave: str,
    items: list[tuple[str, str, str]],
    eval_prompt: str,
    work_root: Path,
    save_raw: Path,
    dry_run: bool,
) -> tuple[dict[str, float | None], dict]:
    """One graded-judge wave through the forced Batch path; returns
    (scores, judge_stats). Drop classes stay SEPARATE fields (content drops
    vs transport losses vs instructed/api refusals vs truncation) — never
    conflated; transport losses were retried client-side and any residue is
    reported, never counted as a content drop."""
    if not items:
        return {}, {"wave": wave, "dispatched": False}
    from explore_persona_space.eval.graded_judge import judge_graded

    res = judge_graded(
        items,
        eval_prompt,
        n_draws=1,  # draws are encoded in the ALIAS: one item per (context, rollout-draw)
        cache_dir=work_root / "judge_cache" / wave,  # rubric-keyed cache partition
        save_raw=save_raw,
        judge_model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        threshold_base=0,  # FORCE the #663-hardened Batch API path (1,464 < sync crossover)
        dry_run=dry_run,
    )
    if dry_run:
        return {}, {"wave": wave, "dispatched": False, "dry_run": True}
    stop_tally = dict(res.stop_reason_tally)
    stats = {
        "wave": wave,
        "dispatched": True,
        "rubric_sha256": rubric_sha256(eval_prompt),
        "n_items": len(items),
        "n_total_draws": res.n_total_draws,
        "n_dropped_draws": res.n_dropped_draws,
        "n_transport_lost_draws": res.n_transport_lost_draws,
        "n_refusal_draws": res.n_refusal_draws,
        "n_truncation_dropped_draws": res.n_truncation_dropped_draws,
        "n_api_refusal_draws": res.n_api_refusal_draws,
        "stop_reason_tally": stop_tally,
        # llm-judging rule 26 recorded property: zero max_tokens stops.
        "zero_max_tokens_stop": stop_tally.get("max_tokens", 0) == 0,
        "frac_items_complete": res.frac_items_complete if res.scores else None,
    }
    return dict(res.scores), stats


# ── main ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--work-root", default=DEFAULT_WORK_ROOT)
    ap.add_argument(
        "--anchors-dir",
        default=None,
        help="local dir holding anchors_{cell}.jsonl (default: HF fetch)",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--dry-run", action="store_true", help="zero-API routing check (all checks incomplete)"
    )
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--max-judged-items",
        type=int,
        default=None,
        help="cap EACH judged wave (smoke default 4/wave); capped-out checks read incomplete",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        raise SystemExit(0)

    smoke = args.smoke
    hf_prefix = f"{HF_PREFIX}/smoke" if smoke else HF_PREFIX
    out = Path(args.out)
    work_root = Path(args.work_root)
    max_items = args.max_judged_items
    if smoke:
        if args.out == DEFAULT_OUT:
            out = Path(SMOKE_ROOT) / "manipulation_check_2587.json"
        elif str(out).startswith("eval_results"):
            raise SystemExit("--smoke must not write the committed eval_results/ path")
        if args.work_root == DEFAULT_WORK_ROOT:
            work_root = Path(SMOKE_ROOT) / "judge_work"
        if max_items is None:
            max_items = SMOKE_JUDGE_ITEMS
    if args.dry_run and args.out == DEFAULT_OUT:
        # never overwrite the production sentinel with a zero-API dry-run table
        out = Path(SMOKE_ROOT) / "manipulation_check_2587.dryrun.json"
        log(f"[judge2587] --dry-run: out rebound to {out}")

    bk = B25._bk()  # sha-asserted pinned bank2564 module (bank2587's accessor)
    carriers = SMOKE_CARRIERS if smoke else bk.CARRIER_IDS
    judged_axes = tuple(a for a in parent_judged_axes(bk) if not smoke or a in SMOKE_CELLS)
    prog_axes = tuple(a for a in PROGRAMMATIC_AXES if not smoke or a in SMOKE_CELLS)
    lang_in_slice = not smoke or LANG_AXIS in SMOKE_CELLS
    log(
        f"[phase=judge2587] start out={out} smoke={smoke} judged_axes={list(judged_axes)} "
        f"prog_axes={list(prog_axes)} lang_in_slice={lang_in_slice} carriers={list(carriers)}"
    )

    values = bk.load_values()
    j_specs = judged_specs(bk, values, carriers, judged_axes) if judged_axes else []
    l_specs = lang_specs(bk, carriers) if lang_in_slice else []
    p_specs = programmatic_specs(bk, values, carriers) if prog_axes else []

    # Plan §4.4 arithmetic — verified against the realized bank on the FULL slice.
    if not smoke:
        arithmetic = verify_call_arithmetic(len(j_specs), len(l_specs))
        log(f"[judge2587] call arithmetic verified: {arithmetic['realized']}")
    else:
        arithmetic = {
            "expected": None,
            "realized": {
                "compliance_calls": len(j_specs),
                "answer_language_calls": len(l_specs),
                "total_calls": len(j_specs) + len(l_specs),
            },
            "verified": False,
        }

    # the two waves' aliases must never collide (separate score dicts anyway)
    overlap = {s["alias"] for s in j_specs} & {s["alias"] for s in l_specs}
    if overlap:
        raise ValueError(f"alias overlap across rubric families: {sorted(overlap)[:5]}")

    cells_needed = tuple(sorted(judged_axes + prog_axes + ((LANG_AXIS,) if lang_in_slice else ())))
    if not cells_needed:
        raise SystemExit("no axes in slice — nothing to check")
    raw_dir = work_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    anchors_dir = Path(args.anchors_dir) if args.anchors_dir else None
    paths = stage_anchor_cells(cells_needed, anchors_dir, hf_prefix, work_root / "anchors_staging")
    texts = load_anchor_texts(paths)
    log(f"[judge2587] loaded {len(texts)} anchor rows across {len(cells_needed)} cells")

    # ---- judged waves (one item per (context, rollout-draw); n_draws=1) ----
    c_items, c_missing, c_capped = _build_wave_items(j_specs, texts, max_items)
    l_items, l_missing, l_capped = _build_wave_items(l_specs, texts, max_items)
    log(
        f"[judge2587] compliance wave: {len(c_items)} items "
        f"(missing_anchor={c_missing} capped_out={c_capped}); "
        f"answer_language wave: {len(l_items)} items "
        f"(missing_anchor={l_missing} capped_out={l_capped})"
    )

    c_scores, c_stats = _dispatch_wave(
        "compliance",
        c_items,
        EVAL_PROMPT,
        work_root,
        raw_dir / "judge_raw_manipcheck.json",
        args.dry_run,
    )
    l_scores, l_stats = _dispatch_wave(
        "answer_language",
        l_items,
        ANSWER_LANGUAGE_EVAL_PROMPT,
        work_root,
        raw_dir / "judge_raw_answer_language.json",
        args.dry_run,
    )

    # derived per-check JSONL (rides to HF next to the client's save_raw JSONs)
    scored_families = [(j_specs, c_scores, "judged"), (l_specs, l_scores, "judged_language")]
    lines = []
    for fam_specs, fam_scores, instrument in scored_families:
        for s in sorted(fam_specs, key=lambda s: s["alias"]):
            score = fam_scores.get(s["alias"])
            outcome = (
                "incomplete" if score is None else ("comply" if score >= 50.0 else "noncomply")
            )
            lines.append(
                json.dumps(
                    {
                        "alias": s["alias"],
                        "context_id": s["context_id"],
                        "draw": s["draw"],
                        "axis": s["axis"],
                        "value_id": s["value_id"],
                        "kind": s["kind"],
                        "carrier": s["carrier"],
                        "instrument": instrument,
                        "score": score,
                        "outcome": outcome,
                    },
                    sort_keys=True,
                )
            )
    if lines:
        with atomic_replace(raw_dir / "judge_scores.jsonl") as tmp:
            tmp.write_text("\n".join(lines) + "\n")

    # ---- fire tables ----
    value_rows: list[dict] = []
    if j_specs:
        value_rows += judged_fire_table(j_specs, c_scores, carriers, JUDGED_DRAWS)
    if l_specs:
        value_rows += judged_fire_table(
            l_specs, l_scores, carriers, JUDGED_DRAWS, instrument="judged_language"
        )
    if p_specs:
        value_rows += programmatic_fire_table(p_specs, texts, carriers, PROG_DRAWS)
    annotate_pilot_rows(value_rows)

    axis_rows: list[dict] = []
    for axis in bk.INSTRUCTION_AXES:
        if axis in judged_axes or axis in prog_axes:
            axis_rows.append(axis_summary(value_rows, axis, bk.N_VALUES_PER_AXIS[axis]))
        else:
            axis_rows.append(
                {
                    "axis": axis,
                    "width": bk.N_VALUES_PER_AXIS[axis],
                    "floor": axis_floor(bk.N_VALUES_PER_AXIS[axis]),
                    "verdict": "not_in_slice",
                }
            )
    lang_width = len(B25.LANG_VALUES)
    if lang_in_slice:
        axis_rows.append(axis_summary(value_rows, LANG_AXIS, lang_width, has_para=False))
    else:
        axis_rows.append(
            {
                "axis": LANG_AXIS,
                "width": lang_width,
                "floor": axis_floor(lang_width),
                "verdict": "not_in_slice",
            }
        )
    # query_content_oneword is a query-class pilot axis: NO manipulation check
    # by design (plan §4.4) — an explicit N/A row, never a silently-missing cell.
    axis_rows.append(
        {
            "axis": "query_content_oneword",
            "verdict": "no_manipulation_check_query_class",
            "note": "query-class pilot axis: no instruction to comply with (plan §4.4)",
        }
    )
    annotate_pilot_rows(axis_rows)

    # ---- upload raw judge outputs ----
    upload_summary: dict | None = None
    if (c_items or l_items) and not args.dry_run and not args.skip_upload:
        from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

        dest_prefix = f"{hf_prefix}/raw_completions/judge"
        res_up = upload_dir_sharded(
            raw_dir,
            HF_DATA_REPO,
            dest_prefix,
            shard_glob="*",
            resume_skip=False,
            delete_local=False,
        )
        upload_summary = {
            "hf_repo": HF_DATA_REPO,
            "hf_dest_prefix": dest_prefix,
            "uploaded": res_up.uploaded,
            "skipped_existing": res_up.skipped_existing,
            "rerouted": res_up.rerouted,
        }
        log(f"[judge2587] raw judge outputs uploaded to {dest_prefix}")

    # ---- the sentinel artifact ----
    doc = {
        "meta": {
            "issue": ISSUE,
            "phase": "judge2587",
            "parent_pin": PARENT_PIN,
            "smoke": smoke,
            "dry_run": args.dry_run,
            "judge_model": JUDGE_MODEL,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
            "judge_temperature": "API default 1.0 (plan pin; not threaded by judge_graded)",
            "judge_route": "eval.graded_judge -> eval.batch_judge (threshold_base=0, forced Batch)",
            "instrument_identity": {
                "compliance_rubric_sha256": rubric_sha256(EVAL_PROMPT),
                "compliance_rubric_source": (
                    f"ported verbatim from scripts/issue2564_judge.py @ {PARENT_PIN}"
                ),
                "answer_language_rubric_sha256": rubric_sha256(ANSWER_LANGUAGE_EVAL_PROMPT),
                "answer_language_rubric_text": ANSWER_LANGUAGE_EVAL_PROMPT,
                "answer_language_rubric_source": (
                    "scripts/issue2587_judge.py::ANSWER_LANGUAGE_EVAL_PROMPT "
                    "(sha-pinned in tests/test_issue2587_judge.py; the #2564 7B-side "
                    "pilot judging consumes this text VERBATIM)"
                ),
            },
            "call_arithmetic": arithmetic,
            "fire_threshold_pct": FIRE_THRESHOLD_PCT,
            "sensitivity_pcts": list(SENSITIVITY_PCTS),
            "floor_rule": "n_fired_base >= ceil(0.6 * width); undetermined counts as not-fired",
            "undetermined_semantics": (
                "mandatory (parent verbatim): ANY incomplete check after the judge "
                "retry budget => undetermined; raw counts persisted per slot"
            ),
            "judged_denominator": len(carriers) * len(JUDGED_DRAWS),
            "programmatic_denominator": len(carriers) * len(PROG_DRAWS),
            "judged_draws": list(JUDGED_DRAWS),
            "programmatic_draws": list(PROG_DRAWS),
            "carriers": list(carriers),
            "judged_axes_in_slice": list(judged_axes),
            "programmatic_axes_in_slice": list(prog_axes),
            "answer_language_in_slice": lang_in_slice,
            "n_judged_specs": len(j_specs),
            "n_lang_specs": len(l_specs),
            "n_items_submitted": {"compliance": len(c_items), "answer_language": len(l_items)},
            "n_missing_anchor_rows": {"compliance": c_missing, "answer_language": l_missing},
            "n_capped_out": {"compliance": c_capped, "answer_language": l_capped},
            "judge_stats": {"compliance": c_stats, "answer_language": l_stats},
            "per_axis_drop_report": {
                "compliance": per_axis_drop_report(j_specs, c_scores),
                "answer_language": per_axis_drop_report(l_specs, l_scores),
            },
            "upload": upload_summary,
            **as_metadata_dict(git_provenance(), phase="judge2587"),
        },
        "value_rows": value_rows,
        "axis_rows": axis_rows,
    }
    _write_json_atomic(out, doc)
    log(f"[phase=judge2587] sentinel written {out}")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
