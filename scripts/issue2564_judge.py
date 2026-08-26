"""PD manipulation check for task #2564 (VM-side, off-pod; plan v6 §3.7 row 4 / §6 / §9).

Per-value "did the instruction fire?" table over the PA anchors, two
instruments:

- **Programmatic axes** (``lexical_marker``, ``user_fact``) — free
  word-containment checks on ALL 10 rollout draws (the sanctioned
  marker-style substring carve-out). Decided convention: CASE-INSENSITIVE,
  ``\\b`` word-boundary-delimited containment (``"surelyX"`` / ``"leisurely"``
  do NOT match ``surely``; punctuation-adjacent occurrences do).
- **Judged axes** (the other 7) — graded 0-100 compliance judge over
  (29 values + 29 paraphrases) × 12 carriers × 2 rollout draws = 1,392 calls:
  ONE item per (context, rollout-draw) with ``n_draws=1``, rubric "does this
  answer comply with the instruction: <verbatim instruction>?", model
  ``claude-sonnet-4-5-20250929``, ``max_tokens=1024``, temperature = the API
  default 1.0 (the plan pin; ``judge_graded`` does not thread temperature),
  routed through the #663-hardened ``eval.batch_judge`` client via
  ``eval.graded_judge.judge_graded`` with ``threshold_base=0`` (FORCES the
  Batch API path — 1,392 calls would otherwise route sync under the 2,000
  crossover), rubric-keyed cache, drop-never-coerce, transport retried.
  Wave size 1,392 « 5,000 ⇒ pilot-gate exempt (plan §8/§9).

Fire decision (plan §6, verbatim semantics): per value slot, comply =
score ≥ 50 (judged) / word present (programmatic); the slot FIRED iff ≥70%
of its checks comply on the FIXED denominator — 24 judged (12 carriers × 2
draws) / 120 programmatic (12 × 10) — never a shrunken one. A check with no
post-retry judgment (dropped / transport-exhausted / missing anchor row) is
INCOMPLETE; a slot with ANY incomplete check after the retry budget is
``undetermined`` (plan §6, verbatim: "a value whose checks are incomplete
after the judge retry budget is marked undetermined") — counted as
not-fired for the axis floor, kept in the denominator. Raw (n_comply,
n_noncomply, n_incomplete, denom) counts are persisted per slot so a
looser decision-relevance-only reading stays recomputable. Axis floor:
≥ ceil(0.6 × width) of the axis's BASE values fired (3/5 five-value axes,
2/2 two-value axes); paraphrase slots get their own fire rows but are
excluded from the floor count. 50%/90% comply-threshold sensitivity columns
ride alongside every verdict. Non-fired values stay in the artifact
(plotted hollow / excluded from headline by Units 4-5).

Outputs: ``eval_results/issue_2564/manipulation_check.json`` (the §9
``pd_judge`` sentinel — per-value fire table + per-axis floor verdicts +
sensitivity + drop/undetermined counts + JudgeResult drop-class stats) and
raw judge outputs (the client's ``save_raw`` JSON + a derived per-check
``judge_scores.jsonl``) uploaded to HF
``issue2564_minpair/raw_completions/judge/``.

Anchors are read from HF ``issue2564_minpair/raw_completions/anchors/*.jsonl``
(the pod is terminated by PD time — plan §9 off_pod_phases), or a local
``--anchors-dir`` when staged.

Smoke (``--smoke``): register + query cells at carriers c01-c03 / draws 0-1
(the driver's smoke slice), judged wave capped at 4 items THROUGH the
production Batch-API client (batch of 4, ``threshold_base=0``), ``/smoke``
HF prefixes, ``/tmp`` out-root — never the committed ``eval_results/``
path; fire verdicts are computed identically but flagged informational
(``meta.smoke = true``; thresholds are never binding at 4 judged rollouts —
the plan §8 blind-spot note).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.experiments.issue2564 import bank2564 as BK  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

ISSUE = 2564
HF_DATA_REPO = os.environ.get("EPM_2564_DATA_WRITE_REPO", "superkaiba1/explore-persona-space-data")
HF_PREFIX = "issue2564_minpair"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
JUDGE_MAX_TOKENS = 1024  # plan §11 pin (generous rationale-sized floor, llm-judging rule 23)
PROGRAMMATIC_AXES: tuple[str, ...] = ("lexical_marker", "user_fact")
JUDGED_AXES: tuple[str, ...] = tuple(a for a in BK.INSTRUCTION_AXES if a not in PROGRAMMATIC_AXES)
JUDGED_DRAWS: tuple[int, ...] = (0, 1)  # plan §6: 2 rollout draws judged
PROG_DRAWS: tuple[int, ...] = tuple(range(10))  # all 10 draws, free
FIRE_THRESHOLD_PCT = 70
SENSITIVITY_PCTS: tuple[int, ...] = (50, 90)
FLOOR_FRAC = 0.6
SMOKE_CELLS: tuple[str, ...] = ("register", "query")  # driver smoke slice; query unused by PD
SMOKE_CARRIERS: tuple[str, ...] = ("c01", "c02", "c03")
SMOKE_JUDGE_ITEMS = 4  # plan §8: "judge on 4 rollouts through the production Batch-API client"
DEFAULT_OUT = "eval_results/issue_2564/manipulation_check.json"
DEFAULT_WORK_ROOT = "data/issue_2564/judge_work"
SMOKE_ROOT = "/tmp/issue2564_pd_smoke"

# custom_id budget: batch_judge appends "__{idx:05d}__{comp:02d}" (11 chars) to a
# 64-char cap => alias budget 53; charset ^[a-zA-Z0-9_-]$ (#1776).
_ALIAS_RE = re.compile(r"^[a-zA-Z0-9_-]{1,53}$")

# {question} carries the VERBATIM instruction string; {answer} the anchor text.
# graded_judge appends its own {"score": N}/{"score": "REFUSAL"} JSON wrapper
# as the system prompt — this template is the user message.
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


def log(msg: str) -> None:
    """Flush-immediate stdout log line."""
    print(msg, flush=True)


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


# ── pure check / fire-decision helpers (CPU-pinned in tests/test_issue2564_pd.py) ──


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
    """Three-way fire decision on the FIXED denominator (plan §6).

    MANDATORY-undetermined (plan §6, verbatim registration; r2 blocker 6):
    a slot with ANY incomplete check after the judge retry budget is
    ``undetermined`` — counted as not-fired for the axis floor, kept in the
    denominator — regardless of whether the incompletes could flip the
    verdict. Otherwise integer arithmetic (no float thresholds): fired iff
    ``n_comply * 100 >= threshold_pct * denom`` (≥70% of 24 ⇒ ≥17; of 120 ⇒
    ≥84). Raw counts are persisted per slot so a looser
    decision-relevance-only reading stays recomputable downstream.
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


# ── spec enumeration (from the REAL bank) ─────────────────────────────


def judged_value_slots(values: dict) -> list[dict]:
    """The 58 judged value slots: 29 base values + 29 paraphrases across 7 axes."""
    slots = []
    for axis in JUDGED_AXES:
        for vid in BK.value_ids(values, axis):
            slots.append(
                {
                    "axis": axis,
                    "value_id": vid,
                    "kind": "orig",
                    "instruction": BK.system_string(values, axis, vid),
                }
            )
            slots.append(
                {
                    "axis": axis,
                    "value_id": f"{vid}p",
                    "kind": "para",
                    "instruction": BK.paraphrase_string(values, axis, vid),
                }
            )
    return slots


def _alias(axis: str, value_id: str, carrier: str, draw: int) -> str:
    """Batch-legal custom_id alias for one check (bijective; asserted below)."""
    return f"{axis}--{value_id}--{carrier}-d{draw}"


def judged_specs(
    values: dict,
    carriers: tuple[str, ...] = BK.CARRIER_IDS,
    draws: tuple[int, ...] = JUDGED_DRAWS,
) -> list[dict]:
    """One judged check per (value slot × carrier × rollout draw).

    Production shape: 58 slots × 12 carriers = 696 contexts × 2 draws = 1,392
    checks. Aliases are validated against the Batch custom_id grammar
    (charset + 53-char budget + no ``__``) and asserted collision-free over
    the FULL realized set (#1776).
    """
    specs = []
    for slot in judged_value_slots(values):
        for carrier in carriers:
            cid = BK.context_id(slot["axis"], slot["value_id"], carrier)
            for draw in draws:
                alias = _alias(slot["axis"], slot["value_id"], carrier, draw)
                if not _ALIAS_RE.match(alias) or "__" in alias:
                    raise ValueError(f"illegal batch alias: {alias!r}")
                specs.append(
                    {
                        **slot,
                        "carrier": carrier,
                        "draw": draw,
                        "context_id": cid,
                        "alias": alias,
                    }
                )
    aliases = [s["alias"] for s in specs]
    if len(set(aliases)) != len(aliases):
        raise ValueError("batch alias collision in judged spec set")
    return specs


def programmatic_specs(
    values: dict,
    carriers: tuple[str, ...] = BK.CARRIER_IDS,
    draws: tuple[int, ...] = PROG_DRAWS,
) -> list[dict]:
    """One programmatic check per (value slot × carrier × draw), with the target word.

    The target word/name for a paraphrase slot is its BASE value's payload
    (the paraphrase rewords the instruction; the target token is unchanged).
    """
    specs = []
    for axis in PROGRAMMATIC_AXES:
        for vid in BK.value_ids(values, axis):
            word = values["axes"][axis]["values"][vid]
            for kind, slot_vid in (("orig", vid), ("para", f"{vid}p")):
                for carrier in carriers:
                    cid = BK.context_id(axis, slot_vid, carrier)
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


# ── fire tables ───────────────────────────────────────────────────────


def judged_fire_table(
    specs: list[dict],
    scores: dict[str, float | None],
    carriers: tuple[str, ...],
    draws: tuple[int, ...],
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
            _value_row(axis, value_id, kind, "judged", n_comply, n_noncomply, n_incomplete, denom)
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
    — should be zero in production, where PA gates on complete cells).
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


def axis_summary(value_rows: list[dict], axis: str, width: int) -> dict:
    """Per-axis floor verdict over BASE values (paraphrase slots excluded).

    ``undetermined`` counts as not-fired for the floor (plan §6). Sensitivity
    floors recompute the fired count at each alternate comply threshold.
    """
    base = [r for r in value_rows if r["axis"] == axis and r["kind"] == "orig"]
    para = [r for r in value_rows if r["axis"] == axis and r["kind"] == "para"]
    assert len(base) == width, (axis, len(base), width)
    floor = axis_floor(width)
    n_fired = sum(1 for r in base if r["verdict"] == "fired")
    row = {
        "axis": axis,
        "width": width,
        "floor": floor,
        "n_fired_base": n_fired,
        "n_undetermined_base": sum(1 for r in base if r["verdict"] == "undetermined"),
        "n_not_fired_base": sum(1 for r in base if r["verdict"] == "not_fired"),
        "floor_met": n_fired >= floor,
        "n_fired_para": sum(1 for r in para if r["verdict"] == "fired"),
        "sensitivity": {
            str(pct): {
                "n_fired_base": sum(1 for r in base if r["sensitivity"][str(pct)] == "fired"),
                "floor_met": sum(1 for r in base if r["sensitivity"][str(pct)] == "fired") >= floor,
            }
            for pct in SENSITIVITY_PCTS
        },
    }
    return row


# ── anchors ingestion ─────────────────────────────────────────────────


def anchors_rel(cell: str) -> str:
    """Repo-relative anchors path for one cell (the PA layout)."""
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
        help="cap the judged wave (smoke default 4); capped-out checks read incomplete",
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
            out = Path(SMOKE_ROOT) / "manipulation_check.json"
        elif str(out).startswith("eval_results"):
            raise SystemExit("--smoke must not write the committed eval_results/ path")
        if args.work_root == DEFAULT_WORK_ROOT:
            work_root = Path(SMOKE_ROOT) / "judge_work"
        if max_items is None:
            max_items = SMOKE_JUDGE_ITEMS
    if args.dry_run and args.out == DEFAULT_OUT:
        # never overwrite the production sentinel with a zero-API dry-run table
        out = Path(SMOKE_ROOT) / "manipulation_check.dryrun.json"
        log(f"[pd_judge] --dry-run: out rebound to {out}")

    carriers = SMOKE_CARRIERS if smoke else BK.CARRIER_IDS
    judged_axes = tuple(a for a in JUDGED_AXES if not smoke or a in SMOKE_CELLS)
    prog_axes = tuple(a for a in PROGRAMMATIC_AXES if not smoke or a in SMOKE_CELLS)
    log(
        f"[phase=pd_judge] start out={out} smoke={smoke} judged_axes={list(judged_axes)} "
        f"prog_axes={list(prog_axes)} carriers={list(carriers)}"
    )

    values = BK.load_values()
    j_specs = judged_specs(values, carriers=carriers) if judged_axes else []
    j_specs = [s for s in j_specs if s["axis"] in judged_axes]
    p_specs = programmatic_specs(values, carriers=carriers) if prog_axes else []
    p_specs = [s for s in p_specs if s["axis"] in prog_axes]

    cells_needed = tuple(sorted(judged_axes + prog_axes))
    if not cells_needed:
        raise SystemExit("no axes in slice — nothing to check")
    raw_dir = work_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    anchors_dir = Path(args.anchors_dir) if args.anchors_dir else None
    paths = stage_anchor_cells(cells_needed, anchors_dir, hf_prefix, work_root / "anchors_staging")
    texts = load_anchor_texts(paths)
    log(f"[pd_judge] loaded {len(texts)} anchor rows across {len(cells_needed)} cells")

    # ---- judged wave (one item per (context, rollout-draw); n_draws=1) ----
    items: list[tuple[str, str, str]] = []
    n_missing_anchor = 0
    for s in sorted(j_specs, key=lambda s: s["alias"]):
        text = texts.get((s["context_id"], s["draw"]))
        if text is None:
            n_missing_anchor += 1
            continue
        items.append((s["alias"], s["instruction"], text))
    n_capped_out = 0
    if max_items is not None and len(items) > max_items:
        n_capped_out = len(items) - max_items
        items = items[:max_items]
    log(
        f"[pd_judge] judged wave: {len(items)} items "
        f"(missing_anchor={n_missing_anchor} capped_out={n_capped_out})"
    )

    scores: dict[str, float | None] = {}
    judge_stats: dict = {"dispatched": False}
    if items:
        from explore_persona_space.eval.graded_judge import judge_graded

        save_raw = raw_dir / "judge_raw_manipcheck.json"
        res = judge_graded(
            items,
            EVAL_PROMPT,
            n_draws=1,
            cache_dir=work_root / "judge_cache",
            save_raw=save_raw,
            judge_model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            threshold_base=0,  # FORCE the Batch API path (plan §6 pin; 1,392 < sync crossover)
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            scores = dict(res.scores)
            judge_stats = {
                "dispatched": True,
                "n_total_draws": res.n_total_draws,
                "n_dropped_draws": res.n_dropped_draws,
                "n_transport_lost_draws": res.n_transport_lost_draws,
                "n_refusal_draws": res.n_refusal_draws,
                "n_truncation_dropped_draws": res.n_truncation_dropped_draws,
                "n_api_refusal_draws": res.n_api_refusal_draws,
                "stop_reason_tally": res.stop_reason_tally,
                "frac_items_complete": res.frac_items_complete if res.scores else None,
            }
        else:
            judge_stats = {"dispatched": False, "dry_run": True}

    # derived per-check JSONL (rides to HF next to the client's save_raw JSON)
    if j_specs:
        lines = []
        for s in sorted(j_specs, key=lambda s: s["alias"]):
            score = scores.get(s["alias"])
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
                        "score": score,
                        "outcome": outcome,
                    },
                    sort_keys=True,
                )
            )
        with atomic_replace(raw_dir / "judge_scores.jsonl") as tmp:
            tmp.write_text("\n".join(lines) + "\n")

    # ---- fire tables ----
    value_rows: list[dict] = []
    if j_specs:
        value_rows += judged_fire_table(j_specs, scores, carriers, JUDGED_DRAWS)
    if p_specs:
        value_rows += programmatic_fire_table(p_specs, texts, carriers, PROG_DRAWS)

    axis_rows: list[dict] = []
    for axis in BK.INSTRUCTION_AXES:
        if axis in judged_axes or axis in prog_axes:
            axis_rows.append(axis_summary(value_rows, axis, BK.N_VALUES_PER_AXIS[axis]))
        else:
            axis_rows.append(
                {
                    "axis": axis,
                    "width": BK.N_VALUES_PER_AXIS[axis],
                    "floor": axis_floor(BK.N_VALUES_PER_AXIS[axis]),
                    "verdict": "not_in_slice",
                }
            )

    # ---- upload raw judge outputs ----
    upload_summary: dict | None = None
    if items and not args.dry_run and not args.skip_upload:
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
        log(f"[pd_judge] raw judge outputs uploaded to {dest_prefix}")

    # ---- the sentinel artifact ----
    doc = {
        "meta": {
            "issue": ISSUE,
            "phase": "pd_judge",
            "smoke": smoke,
            "dry_run": args.dry_run,
            "judge_model": JUDGE_MODEL,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
            "judge_temperature": "API default 1.0 (plan pin; not threaded by judge_graded)",
            "judge_route": "eval.graded_judge -> eval.batch_judge (threshold_base=0, forced Batch)",
            "fire_threshold_pct": FIRE_THRESHOLD_PCT,
            "sensitivity_pcts": list(SENSITIVITY_PCTS),
            "floor_rule": "n_fired_base >= ceil(0.6 * width); undetermined counts as not-fired",
            "undetermined_semantics": (
                "mandatory (plan §6 verbatim): ANY incomplete check after the judge "
                "retry budget => undetermined; raw counts persisted per slot"
            ),
            "judged_denominator": len(carriers) * len(JUDGED_DRAWS),
            "programmatic_denominator": len(carriers) * len(PROG_DRAWS),
            "judged_draws": list(JUDGED_DRAWS),
            "programmatic_draws": list(PROG_DRAWS),
            "carriers": list(carriers),
            "judged_axes_in_slice": list(judged_axes),
            "programmatic_axes_in_slice": list(prog_axes),
            "n_judged_specs": len(j_specs),
            "n_items_submitted": len(items),
            "n_missing_anchor_rows": n_missing_anchor,
            "n_capped_out": n_capped_out,
            "judge_stats": judge_stats,
            "upload": upload_summary,
            **as_metadata_dict(git_provenance(), phase="pd-judge"),
        },
        "value_rows": value_rows,
        "axis_rows": axis_rows,
    }
    _write_json_atomic(out, doc)
    log(f"[phase=pd_judge] sentinel written {out}")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
