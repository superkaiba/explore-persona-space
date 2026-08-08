"""Issue #2094 — VM-side judge pipeline (plan §4.4-§4.6, P6/P9).

Builds judge items from the pod driver's rollout JSONLs (grid / anchors /
stage-2), runs the rule-26 pilot gate + the coherence-baseline sanity gate
(plan §7 gates 3-4), dispatches the production Batch-API waves through the
sanctioned graded-judge instrument, and computes the unconditional mechanical
audits (script intrusion / repetition / empty — the #1415 all-position lesson).

Instrument (body § Judging constants + llm-judging rules): judge
``claude-sonnet-4-5-20250929`` (project default), graded 0-100
reason-then-score (single-rationale JSON — ``max_tokens=1024``, the rule-23
floor), N=1 judge draw per rubric per rollout (body decision; uncertainty is
carried by the pair-clustered bootstrap in the analysis unit), drop-never-coerce
with the per-arm content-drop vs transport-loss split (rules 9/24), rubric-keyed
cache with per-rubric ``cache_dir`` partitions (rule 22), one behavior per call
(rule 8).

Everything routes through :func:`eval.graded_judge.judge_graded` — the SAME
instrument the rule-26 pilot gate (:func:`eval.judge_pilot.judge_pilot_gate`)
fires, which internally routes ``judge_completions_batch`` →
``judge_dispatch.dispatch_judge_items`` (sync below the tier-scaled threshold,
Message Batches at/above it, resumable #1019 checkpoints, ≤2k judge
sub-batches) with ``validate_batch_custom_ids`` wired at routing time. Note
``judge_graded`` does NOT thread ``temperature`` into requests (the client
exposes no such parameter) — draws sample at the Anthropic API default; this is
recorded in every wave meta.

Rubrics (9): one form-only coherence rubric (body-verbatim truncation clause,
via ``bank.COHERENCE_RUBRIC``) + per-setting behavior rubric pairs from the
unit-A cores — matched-prefix → the F_query pair, matched-query → the F_prefix
pair, cross → BOTH pairs on the same draws (plan §4.4). Grouped by rubric
IDENTITY (fq_<query> / fp_<prefix> / coherence), each wave is one rubric.

Phases (all resumable; a completed wave skips via its meta sidecar, and a
regime mismatch REFUSES — never a silent reuse of wrong cached rows):

- ``--phase pilot``    rule-26 pilot (~340 draws spanning arms × all 9 rubrics)
                       + a ~6-item forced-batch request-shape probe
                       (``threshold_base=0`` — the #1090 fu6 live probe).
                       FAIL → rc=7 + ``gates/pilot_gate_report.json``.
- ``--phase anchors``  anchor audits → anchor coherence wave → coherence-
                       baseline sanity gate (median ≥ 80 AND ≥90% of draws
                       > 60; FAIL → rc=8, halts behavior spend only) → the
                       DEDUPED anchor behavior waves.
- ``--phase waves``    grid audits → grid coherence + behavior waves.
                       REFUSES unless both gate reports exist and PASS.
- ``--phase stage2``   same for stage-2 rollouts (``--stage2-dir``).
- ``--phase audits``   standalone mechanical audits (no API).
- ``--phase upload-raw``  one ``hub._upload`` folder commit of the work root
                       to ``issue2094_singlepos/raw_completions/judge_raw``.

No torch import (pure text + API); explicit ``sys.exit`` on every path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # credentials before any API-bearing import (uv run does not auto-load .env)

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL  # noqa: E402
from explore_persona_space.eval.graded_judge import (  # noqa: E402
    JudgeResult,
    judge_graded,
)
from explore_persona_space.eval.judge_dispatch import validate_batch_custom_ids  # noqa: E402
from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: E402
from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2094_judge")

# Mirror of issue2094_run.HF_PREFIX (kept local so this script never imports the
# torch-bearing pod driver; equality is pinned by tests/test_issue2094_judge.py).
HF_PREFIX = "issue2094_singlepos"
DATASET_REPO = "superkaiba1/explore-persona-space-data"

DEFAULT_JUDGE_MAX_TOKENS = 1024  # plan §6: >= the rule-23 single-rationale floor
JUDGE_N_DRAWS = 1  # body § Judging constants (bootstrap carries the uncertainty)

# Distinct rcs — a designed halt is never an anonymous rc=1 (#1415 lesson).
RC_OK = 0
RC_PILOT_GATE = 7
RC_COHERENCE_GATE = 8

COHERENCE_RUBRIC_ID = "coherence"
# Coherence-baseline sanity gate (plan §7 gate 3, body-verbatim thresholds).
COHERENCE_MEDIAN_MIN = 80.0
COHERENCE_GT60_FRAC_MIN = 0.90
COHERENCE_THRESHOLD = 60.0

# Pilot sizing (plan §7 gate 4: ~300 draws spanning settings × slots × rubrics).
PILOT_TARGET_COHERENCE = 102  # 6 arms × 17
PILOT_TARGET_BEHAVIOR = 30  # 3 arms × 10 (== the gate's min-draws floor)
PILOT_SEED = 2094
FORCED_BATCH_PROBE_N = 6  # live Batch request-shape probe (gotchas: mock-judge smoke)

# Mechanical-audit flag thresholds (body §Coherence: "threshold + counts
# reported"; flags are AUDIT fields — exclusion decisions live in the analysis
# unit, never here). Bank + queries are all-English (plan §4.1), so any
# non-Latin LETTER mass is unlicensed script intrusion.
AUDIT_NONLATIN_FRAC_MAX = 0.05
AUDIT_DUP_4GRAM_FRAC_MAX = 0.50  # duplicate-4gram fraction (loop-robust; see dup_4gram_frac)
_LATIN_MAX_CP = 0x250  # Basic Latin + Latin-1 + Latin Extended-A/B

ITEM_ID_RE = re.compile(r"^[a-zA-Z0-9-]{1,53}$")  # 53 = 64 - len("__NNNNN__NN") (#1776)

GRID_CALLS_PER_SETTING = {"matched_prefix": 2, "matched_query": 2, "cross": 4}


# ── rubric registry (production instruments; cores from unit A) ────────

REASON_THEN_SCORE = (
    "First give a brief reasoning (1-3 sentences), then output a single JSON "
    'object of the form {"reasoning": "<your brief justification>", '
    '"score": <an integer from 0 to 100>} and nothing else after it.'
)


def coherence_eval_prompt() -> str:
    """Form-only coherence instrument (body-verbatim rubric incl. truncation clause)."""
    return f"{BANK.COHERENCE_RUBRIC}\n\n[RESPONSE]\n{{answer}}\n[/RESPONSE]\n\n{REASON_THEN_SCORE}"


def behavior_eval_prompt(core: str) -> str:
    """Wrap a unit-A rubric core (anchored 0/50/100) in the production instrument."""
    return f"{core}\n\n[RESPONSE]\n{{answer}}\n[/RESPONSE]\n\n{REASON_THEN_SCORE}"


def rubric_registry() -> dict[str, str]:
    """rubric_id -> production eval_prompt. 9 rubrics; each wave judges ONE (rule 8).

    Templates deliberately carry only the ``{answer}`` slot: the F rubric cores
    embed their target question/prefix descriptor verbatim, and the coherence
    rubric is form-only (a question would invite relevance judgments the body
    explicitly excludes). ``judge_graded``'s substitution no-ops on the absent
    ``{question}`` slot (pinned by the rule-27 round-trip tests).
    """
    reg = {COHERENCE_RUBRIC_ID: coherence_eval_prompt()}
    for q in BANK.QUERY_ORDER:
        reg[f"fq-{q}"] = behavior_eval_prompt(BANK.f_query_rubric(q))
    for p in BANK.PREFIX_ORDER:
        reg[f"fp-{p}"] = behavior_eval_prompt(BANK.f_prefix_rubric(p))
    return reg


def rubric_id_for(pair: BANK.Pair, kind: str, side: str) -> str:
    """The rubric id judging ``side`` ('a'|'b') of ``pair`` under rubric ``kind``."""
    assert side in ("a", "b"), side
    if kind == "query":
        return f"fq-{pair.query_a if side == 'a' else pair.query_b}"
    assert kind == "prefix", kind
    return f"fp-{pair.prefix_a if side == 'a' else pair.prefix_b}"


# ── judge units + item ids ─────────────────────────────────────────────


@dataclass(frozen=True)
class JudgeUnit:
    """One judge call: (item_id, question, answer) + its rubric + provenance."""

    item_id: str
    rubric_id: str
    question: str
    answer: str
    source: dict


def _item_id(tag: str, source_key: str) -> str:
    """Deterministic content-derived id: ``<tag><sha1[:12]>`` (<=53 chars, no '__').

    Content-derived (never enumeration-order) so ids are stable across resumes
    and partial inputs; uniqueness is asserted per wave (collision odds at 12
    hex over ~1e5 items are ~1e-9). The id_map manifest persists the bijection.
    """
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12]
    return f"{tag}{digest}"


def _validate_units(units: list[JudgeUnit]) -> None:
    """Fail-loud id grammar + uniqueness + Batch custom_id pre-flight (#1776)."""
    ids = [u.item_id for u in units]
    bad = [i for i in ids if not ITEM_ID_RE.match(i) or "__" in i]
    if bad:
        raise ValueError(f"{len(bad)} item ids violate ^[a-zA-Z0-9-]{{1,53}}$: {bad[:5]}")
    if len(set(ids)) != len(ids):
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        raise ValueError(f"duplicate item ids ({len(dupes)}): {dupes[:5]}")
    validate_batch_custom_ids(ids)


# ── input walkers ──────────────────────────────────────────────────────


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Text-mode line iteration (never ``splitlines()`` — U+2028 shred, #950)."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_grid_rows(rollouts_dir: Path) -> list[dict]:
    """All grid rollout rows, shard files walked in sorted order (deterministic)."""
    shards = sorted(rollouts_dir.glob("shard_*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.jsonl under {rollouts_dir}")
    rows: list[dict] = []
    for shard in shards:
        rows.extend(_iter_jsonl(shard))
    logger.info("[inputs] grid: %d rows from %d shards", len(rows), len(shards))
    return rows


def load_anchor_rows(anchors_file: Path) -> list[dict]:
    if not anchors_file.is_file():
        raise FileNotFoundError(f"anchors file missing: {anchors_file}")
    rows = list(_iter_jsonl(anchors_file))
    logger.info("[inputs] anchors: %d rows", len(rows))
    return rows


_STAGE2_REQUIRED = ("pair_id", "setting", "text", "draw")


def load_stage2_rows(stage2_dir: Path) -> list[dict]:
    """Stage-2 rollout rows (interface assumption for unit E: rows carry at
    least pair_id / setting / text / draw, plus an optional cell key)."""
    files = sorted(stage2_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no *.jsonl under {stage2_dir}")
    rows: list[dict] = []
    for f in files:
        rows.extend(_iter_jsonl(f))
    for r in rows[:1] + rows[-1:]:
        missing = [k for k in _STAGE2_REQUIRED if k not in r]
        if missing:
            raise ValueError(f"stage2 rows missing required fields {missing}: {sorted(r)}")
    logger.info("[inputs] stage2: %d rows from %d files", len(rows), len(files))
    return rows


def pair_index() -> dict[str, BANK.Pair]:
    return {p.pair_id: p for p in BANK.build_pairs()}


def _query_text(context_id: str) -> str:
    return BANK.QUERIES[context_id.split("__")[1]]


# ── item builders ──────────────────────────────────────────────────────


def _grid_key(row: dict) -> str:
    return f"{row['block_key']}|{row['pair_id']}"


def build_coherence_items(
    grid_rows: list[dict] | None,
    anchor_rows: list[dict] | None,
    stage2_rows: list[dict] | None = None,
) -> list[JudgeUnit]:
    """ONE coherence call per rollout (body §Coherence), across every source."""
    units: list[JudgeUnit] = []
    for row in grid_rows or []:
        units.append(
            JudgeUnit(
                item_id=_item_id("c", f"coh|grid|{_grid_key(row)}"),
                rubric_id=COHERENCE_RUBRIC_ID,
                question=_query_text(row["context_a"]),
                answer=row["text"],
                source={"kind": "grid", **_grid_source(row)},
            )
        )
    for row in anchor_rows or []:
        units.append(
            JudgeUnit(
                item_id=_item_id("c", f"coh|anch|{row['context_id']}|{row['draw']}"),
                rubric_id=COHERENCE_RUBRIC_ID,
                question=_query_text(row["context_id"]),
                answer=row["text"],
                source={"kind": "anchor", "context_id": row["context_id"], "draw": row["draw"]},
            )
        )
    for row in stage2_rows or []:
        units.append(
            JudgeUnit(
                item_id=_item_id("c", f"coh|s2|{_stage2_key(row)}"),
                rubric_id=COHERENCE_RUBRIC_ID,
                question=_query_text(row.get("context_a", row["pair_id"].split("--")[1])),
                answer=row["text"],
                source={"kind": "stage2", **_stage2_source(row)},
            )
        )
    return units


def _grid_source(row: dict) -> dict:
    keep = (
        "block_key",
        "slot",
        "layer_variant",
        "dose",
        "vec_type",
        "arm",
        "pair_id",
        "setting",
        "context_a",
        "context_b",
        "cap_hit",
    )
    return {k: row.get(k) for k in keep}


def _stage2_key(row: dict) -> str:
    cell = row.get("cell", row.get("block_key", "cell"))
    return f"{row['pair_id']}|{cell}|{row['draw']}"


def _stage2_source(row: dict) -> dict:
    keep = ("pair_id", "setting", "context_a", "context_b", "cell", "block_key", "draw", "cap_hit")
    return {k: row.get(k) for k in keep if k in row}


def build_grid_behavior_items(
    grid_rows: list[dict], pairs: dict[str, BANK.Pair]
) -> dict[str, list[JudgeUnit]]:
    """Per-setting rubric pairs on every grid rollout (plan §4.4): mp → F_query
    pair, mq → F_prefix pair, cross → BOTH pairs on the same draws."""
    by_rubric: dict[str, list[JudgeUnit]] = {}
    for row in grid_rows:
        pair = pairs[row["pair_id"]]
        for kind in BANK.SETTING_RUBRIC_KINDS[pair.setting]:
            for side in ("a", "b"):
                rid = rubric_id_for(pair, kind, side)
                unit = JudgeUnit(
                    item_id=_item_id("g", f"beh|grid|{_grid_key(row)}|{kind}|{side}"),
                    rubric_id=rid,
                    question=_query_text(row["context_a"]),
                    answer=row["text"],
                    source={"kind": "grid", "rubric_kind": kind, "side": side, **_grid_source(row)},
                )
                by_rubric.setdefault(rid, []).append(unit)
    return by_rubric


def build_anchor_behavior_items(
    anchor_rows: list[dict], pairs: dict[str, BANK.Pair]
) -> dict[str, list[JudgeUnit]]:
    """DEDUPED anchor behavior items: one call per (context, draw, rubric).

    Anchor draws are shared across pairs (a context's K draws are the floor /
    ceiling of every pair touching it), and a draw's score under a rubric is
    pair-independent — so items are keyed (context, draw, rubric) and the
    analysis unit re-expands per pair via context ids. This realizes ~1.2k
    calls instead of the plan's pair-expanded 3.0k (identical information; a
    deliberate, reported cost saving).
    """
    needed: set[tuple[str, str]] = set()  # (context_id, rubric_id)
    for pair in pairs.values():
        for kind in BANK.SETTING_RUBRIC_KINDS[pair.setting]:
            for side in ("a", "b"):
                rid = rubric_id_for(pair, kind, side)
                needed.add((pair.a, rid))
                needed.add((pair.b, rid))
    by_rubric: dict[str, list[JudgeUnit]] = {}
    for row in anchor_rows:
        cid, draw = row["context_id"], row["draw"]
        for ctx, rid in sorted(needed):
            if ctx != cid:
                continue
            unit = JudgeUnit(
                item_id=_item_id("a", f"beh|anch|{cid}|{draw}|{rid}"),
                rubric_id=rid,
                question=_query_text(cid),
                answer=row["text"],
                source={"kind": "anchor", "context_id": cid, "draw": draw},
            )
            by_rubric.setdefault(rid, []).append(unit)
    return by_rubric


def build_stage2_behavior_items(
    stage2_rows: list[dict], pairs: dict[str, BANK.Pair]
) -> dict[str, list[JudgeUnit]]:
    by_rubric: dict[str, list[JudgeUnit]] = {}
    for row in stage2_rows:
        pair = pairs[row["pair_id"]]
        for kind in BANK.SETTING_RUBRIC_KINDS[pair.setting]:
            for side in ("a", "b"):
                rid = rubric_id_for(pair, kind, side)
                unit = JudgeUnit(
                    item_id=_item_id("s", f"beh|s2|{_stage2_key(row)}|{kind}|{side}"),
                    rubric_id=rid,
                    question=_query_text(pair.a),
                    answer=row["text"],
                    source={
                        "kind": "stage2",
                        "rubric_kind": kind,
                        "side": side,
                        **_stage2_source(row),
                    },
                )
                by_rubric.setdefault(rid, []).append(unit)
    return by_rubric


def arm_class(source: dict) -> str:
    """Per-arm reporting class (rules 9/18/24): anchor | stage2 | grid-steered | grid-null."""
    if source.get("kind") == "anchor":
        return "anchor"
    if source.get("kind") == "stage2":
        return "stage2"
    return f"grid-{source.get('arm', 'unknown')}"


# ── mechanical audits (unconditional, every arm — #1415 lesson) ────────


def nonlatin_letter_frac(text: str) -> float:
    """Fraction of LETTER codepoints outside Latin (cp >= U+0250) among all letters."""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) >= _LATIN_MAX_CP) / len(letters)


def _word_4grams(text: str) -> Counter:
    words = text.split()
    if len(words) < 4:
        return Counter()
    return Counter(tuple(words[i : i + 4]) for i in range(len(words) - 3))


def max_repeated_4gram_frac(text: str) -> float:
    """Most-frequent word-4gram count / total 4grams (0.0 below 4 words)."""
    grams = _word_4grams(text)
    return max(grams.values()) / sum(grams.values()) if grams else 0.0


def dup_4gram_frac(text: str) -> float:
    """1 - distinct/total word-4grams — the repetition FLAG basis.

    A k-word repetition loop splits its mass across k rotated 4-grams, capping
    ``max_repeated_4gram_frac`` at ~1/k (a 4-word loop reads only ~0.25), while
    the duplicate fraction reads ~1.0 there and ~0 on ordinary prose — so the
    flag keys on THIS statistic; the max-single-4-gram fraction is kept as a
    reported companion field.
    """
    grams = _word_4grams(text)
    if not grams:
        return 0.0
    total = sum(grams.values())
    return 1.0 - len(grams) / total


def audit_text(text: str) -> dict:
    """Per-rollout mechanical audit fields (flags only; analysis owns exclusion)."""
    empty = len(text.strip()) == 0
    nl = nonlatin_letter_frac(text)
    rep_max = max_repeated_4gram_frac(text)
    rep_dup = dup_4gram_frac(text)
    return {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "empty": empty,
        "nonlatin_letter_frac": round(nl, 6),
        "max_repeated_4gram_frac": round(rep_max, 6),
        "dup_4gram_frac": round(rep_dup, 6),
        "flag_empty": empty,
        "flag_script_intrusion": nl > AUDIT_NONLATIN_FRAC_MAX,
        "flag_repetition": rep_dup > AUDIT_DUP_4GRAM_FRAC_MAX,
    }


def run_audits(kind: str, rows: list[dict], out_dir: Path) -> dict:
    """Audit every rollout of ``kind`` -> audits/<kind>.audit.jsonl + summary dict.

    Total wall is ~seconds per 10k rows (pure text scan), so this is a single
    atomic write per kind — the intra-phase checkpoint grain is deliberately
    skipped (re-run cost is trivial; progress lines every 2000 rows).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []
    n_flagged = Counter()
    for i, row in enumerate(rows):
        src = (
            _grid_source(row)
            if kind == "grid"
            else (
                {"context_id": row["context_id"], "draw": row["draw"]}
                if kind == "anchors"
                else _stage2_source(row)
            )
        )
        a = audit_text(row["text"])
        for f in ("flag_empty", "flag_script_intrusion", "flag_repetition"):
            n_flagged[f] += int(a[f])
        out_rows.append({"kind": kind, **src, **a})
        if (i + 1) % 2000 == 0:
            logger.info("[audits] %s %d/%d", kind, i + 1, len(rows))
    path = out_dir / f"{kind}.audit.jsonl"
    _write_jsonl_atomic(path, out_rows)
    summary = {
        "kind": kind,
        "n_rows": len(rows),
        "thresholds": {
            "nonlatin_letter_frac": AUDIT_NONLATIN_FRAC_MAX,
            "dup_4gram_frac": AUDIT_DUP_4GRAM_FRAC_MAX,
        },
        "flag_counts": dict(n_flagged),
        "path": str(path),
    }
    logger.info("[audits] %s done: %s", kind, summary["flag_counts"])
    return summary


# ── atomic writers ─────────────────────────────────────────────────────


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    os.replace(tmp, path)


def _repro() -> dict:
    meta = as_metadata_dict(git_provenance())
    meta["script"] = "scripts/issue2094_judge.py"
    return meta


# ── wave runner ────────────────────────────────────────────────────────


def wave_regime(
    wave: str, rubric_id: str, prompt: str, units: list[JudgeUnit], cfg: JudgeConfig
) -> dict:
    """Every output-affecting regime key of a wave (resume pins on ALL of them)."""
    ids_sha = hashlib.sha256("\n".join(sorted(u.item_id for u in units)).encode()).hexdigest()[:16]
    return {
        "wave": wave,
        "rubric_id": rubric_id,
        "rubric_sha16": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "n_items": len(units),
        "item_ids_sha16": ids_sha,
        "judge_model": cfg.judge_model,
        "max_tokens": cfg.max_tokens,
        "n_draws": JUDGE_N_DRAWS,
    }


def _wave_skip_state(meta_path: Path, regime: dict) -> str:
    """'skip' (done, same regime) | 'run' (absent/incomplete) | raises on mismatch."""
    if not meta_path.is_file():
        return "run"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("regime") == regime:
        return "skip" if meta.get("complete") else "run"
    raise RuntimeError(
        f"wave {regime['wave']}: existing meta at {meta_path} carries a DIFFERENT "
        f"regime — refusing to resume across regimes (quarantine the work root or "
        f"use a fresh --work-root). existing={meta.get('regime')} new={regime}"
    )


def _per_arm_split(units: list[JudgeUnit], result: JudgeResult) -> dict:
    """Per-arm content-drop vs transport-loss split (rules 9/18/24)."""
    out: dict[str, Counter] = {}
    for u in units:
        arm = arm_class(u.source)
        c = out.setdefault(arm, Counter())
        c["n_items"] += 1
        transport = result.per_item_transport_losses.get(u.item_id, 0)
        kept = len(result.per_item_scores.get(u.item_id, []))
        c["n_scored"] += int(kept > 0)
        c["transport_lost"] += transport
        c["truncation_drops"] += result.per_item_truncation_drops.get(u.item_id, 0)
        if kept == 0 and transport == 0:
            c["content_drops"] += 1
    return {arm: dict(c) for arm, c in out.items()}


def _telemetry(result: JudgeResult) -> dict:
    return {
        "n_total_draws": result.n_total_draws,
        "n_dropped_draws_content": result.n_dropped_draws,
        "n_refusal_draws": result.n_refusal_draws,
        "n_truncation_dropped_draws": result.n_truncation_dropped_draws,
        "n_transport_lost_draws": result.n_transport_lost_draws,
        "stop_reason_tally": dict(result.stop_reason_tally),
    }


def run_wave(
    wave: str,
    rubric_id: str,
    prompt: str,
    units: list[JudgeUnit],
    cfg: JudgeConfig,
) -> dict | None:
    """One production wave: manifest → dispatch → bounded transport retry → scores.

    Returns the meta dict (None on ``--dry-run``). Resumable at two grains: the
    wave-level meta skip, and intra-wave via the rubric-keyed JudgeCache + the
    #1019 dispatch checkpoints under ``cache_dir/.dispatch``.
    """
    assert units, f"wave {wave}: no items"
    meta_path = cfg.scores_dir / f"{wave}.meta.json"
    regime = wave_regime(wave, rubric_id, prompt, units, cfg)
    if _wave_skip_state(meta_path, regime) == "skip":
        logger.info("[wave %s] complete (meta matches regime) — skip", wave)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    _validate_units(units)
    _write_jsonl_atomic(
        cfg.items_dir / f"{wave}.items.jsonl",
        [
            {
                "item_id": u.item_id,
                "rubric_id": u.rubric_id,
                "wave": wave,
                "question": u.question,
                "answer_sha16": hashlib.sha256(u.answer.encode()).hexdigest()[:16],
                "n_answer_chars": len(u.answer),
                **u.source,
            }
            for u in units
        ],
    )
    logger.info("[wave %s] %d items (rubric %s) dispatching", wave, len(units), rubric_id)
    items = [(u.item_id, u.question, u.answer) for u in units]
    result = judge_graded(
        items,
        prompt,
        n_draws=JUDGE_N_DRAWS,
        cache_dir=cfg.cache_root / rubric_id,
        save_raw=cfg.raw_dir / f"{wave}.json",
        judge_model=cfg.judge_model,
        max_tokens=cfg.max_tokens,
        dry_run=cfg.dry_run,
    )
    if cfg.dry_run:
        logger.info("[wave %s] dry-run complete (no API calls)", wave)
        return None

    # Bounded transport retry (rule 24: retried, never persisted as drops).
    # The rubric-keyed cache holds no entry for a transport-lost draw (#1313
    # put-skip), so a scoped re-dispatch re-judges exactly the lost items.
    retry_meta = None
    lost_ids = {i for i, n in result.per_item_transport_losses.items() if n > 0}
    if lost_ids:
        logger.info("[wave %s] transport retry: %d items", wave, len(lost_ids))
        retry_units = [u for u in units if u.item_id in lost_ids]
        retry_items = [(u.item_id, u.question, u.answer) for u in retry_units]
        result2 = judge_graded(
            retry_items,
            prompt,
            n_draws=JUDGE_N_DRAWS,
            cache_dir=cfg.cache_root / rubric_id,
            save_raw=cfg.raw_dir / f"{wave}.retry1.json",
            judge_model=cfg.judge_model,
            max_tokens=cfg.max_tokens,
        )
        retry_meta = _telemetry(result2)
        for iid in lost_ids:
            result.scores[iid] = result2.scores.get(iid)
            result.per_item_scores[iid] = result2.per_item_scores.get(iid, [])
            result.per_item_transport_losses[iid] = result2.per_item_transport_losses.get(iid, 0)
            result.per_item_truncation_drops[iid] = result2.per_item_truncation_drops.get(iid, 0)

    scores_rows = [
        {
            "item_id": u.item_id,
            "wave": wave,
            "rubric_id": rubric_id,
            "score": result.scores.get(u.item_id),
            "n_kept_draws": len(result.per_item_scores.get(u.item_id, [])),
            "transport_lost_residual": result.per_item_transport_losses.get(u.item_id, 0),
            **u.source,
        }
        for u in units
    ]
    _write_jsonl_atomic(cfg.scores_dir / f"{wave}.scores.jsonl", scores_rows)
    residual_transport = sum(r["transport_lost_residual"] for r in scores_rows)
    meta = {
        "regime": regime,
        "complete": True,
        "pass1": _telemetry(result),
        "retry1": retry_meta,
        "residual_transport_lost": residual_transport,
        "temperature_note": "not threaded by judge_graded — Anthropic API default",
        "per_arm": _per_arm_split(units, result),
        "n_scored_items": sum(1 for r in scores_rows if r["score"] is not None),
        "repro": _repro(),
    }
    _write_json_atomic(meta_path, meta)
    logger.info(
        "[wave %s] done: %d/%d items scored, residual transport %d",
        wave,
        meta["n_scored_items"],
        len(units),
        residual_transport,
    )
    return meta


# ── gates ──────────────────────────────────────────────────────────────


def coherence_baseline_gate(anchor_scores: list[dict]) -> dict:
    """Plan §7 gate 3: anchor draws median >= 80 AND >90%-of-draws > 60."""
    kept = sorted(r["score"] for r in anchor_scores if r.get("score") is not None)
    n_dropped = sum(1 for r in anchor_scores if r.get("score") is None)
    if not kept:
        raise RuntimeError("coherence-baseline gate: zero kept anchor coherence scores")
    mid = len(kept) // 2
    median = kept[mid] if len(kept) % 2 else (kept[mid - 1] + kept[mid]) / 2.0
    frac_gt60 = sum(1 for s in kept if s > COHERENCE_THRESHOLD) / len(kept)
    passed = median >= COHERENCE_MEDIAN_MIN and frac_gt60 >= COHERENCE_GT60_FRAC_MIN
    return {
        "passed": passed,
        "median": median,
        "frac_gt60": frac_gt60,
        "n_kept": len(kept),
        "n_dropped": n_dropped,
        "thresholds": {
            "median_min": COHERENCE_MEDIAN_MIN,
            "gt60_frac_min": COHERENCE_GT60_FRAC_MIN,
            "coherence_threshold": COHERENCE_THRESHOLD,
        },
        "repro": _repro(),
    }


def _require_gates(cfg: JudgeConfig) -> None:
    """Behavior-wave spend requires BOTH gate reports present and PASS (plan §7)."""
    for name in ("pilot_gate_report.json", "coherence_baseline_gate.json"):
        path = cfg.gates_dir / name
        if not path.is_file():
            raise RuntimeError(f"gate report missing: {path} — run --phase pilot / anchors first")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not rec.get("passed"):
            raise RuntimeError(f"gate FAILED per {path} — fix the instrument and re-run the gate")


# ── pilot (plan §7 gate 4, llm-judging rule 26) ────────────────────────


def _coherence_pilot_arm(unit: JudgeUnit) -> str:
    src = unit.source
    if src["kind"] == "anchor":
        return "anchor"
    if src.get("arm") == "null":
        return "null"
    if src.get("dose") == "replace":
        return "rep"
    slot = src.get("slot")
    if slot in ("ce", "pe"):
        return slot
    return "ctrl"


def _behavior_pilot_arm(unit: JudgeUnit) -> str:
    return "anchor" if unit.source["kind"] == "anchor" else f"{unit.source.get('arm')}"


def _pilot_arms(rubric_id: str, units: list[JudgeUnit]) -> dict[str, list[tuple[str, str, str]]]:
    arm_fn = _coherence_pilot_arm if rubric_id == COHERENCE_RUBRIC_ID else _behavior_pilot_arm
    arms: dict[str, list[tuple[str, str, str]]] = {}
    for u in units:
        arms.setdefault(arm_fn(u), []).append((u.item_id, u.question, u.answer))
    return arms


def phase_pilot(cfg: JudgeConfig) -> int:
    """Rule-26 pilot per rubric + the live forced-batch request-shape probe."""
    registry = rubric_registry()
    pairs = pair_index()
    grid_rows = load_grid_rows(cfg.rollouts_dir)
    anchor_rows = load_anchor_rows(cfg.anchors_file)
    coh = build_coherence_items(grid_rows, anchor_rows)
    beh = build_grid_behavior_items(grid_rows, pairs)
    for rid, us in build_anchor_behavior_items(anchor_rows, pairs).items():
        beh.setdefault(rid, []).extend(us)

    per_rubric: dict[str, dict] = {}
    all_pass = True
    for rid, prompt in registry.items():
        units = coh if rid == COHERENCE_RUBRIC_ID else beh.get(rid, [])
        if not units:
            raise RuntimeError(f"pilot: rubric {rid} has zero items — inputs incomplete")
        _validate_units(units)
        arms = _pilot_arms(rid, units)
        dropped_arms = [a for a, items in arms.items() if not items]
        arms = {a: items for a, items in arms.items() if items}
        target = PILOT_TARGET_COHERENCE if rid == COHERENCE_RUBRIC_ID else PILOT_TARGET_BEHAVIOR
        report = judge_pilot_gate(
            arms,
            prompt,
            max_tokens=cfg.max_tokens,
            cache_dir=cfg.pilot_cache_root / rid,
            save_raw_dir=cfg.raw_dir / "pilot" / rid,
            n_draws=JUDGE_N_DRAWS,
            target_total_draws=target,
            judge_model=cfg.judge_model,
            report_path=cfg.gates_dir / "pilot" / f"{rid}.json",
            seed=PILOT_SEED,
        )
        per_rubric[rid] = {
            "verdict": report.verdict,
            "failures": report.failures,
            "warnings": report.warnings,
            "n_total_draws": report.n_total_draws,
            "dropped_empty_arms": dropped_arms,
        }
        all_pass &= report.passed
        logger.info("[pilot] %s: %s (%d draws)", rid, report.verdict, report.n_total_draws)

    # Live forced-batch request-shape probe (threshold_base=0 — the #1090 fu6
    # pattern: a ~6-request submit through the run's EXACT builder on the
    # Batch path; a mock/sync-only pilot cannot validate the batches.create
    # envelope, gotchas.md).
    probe_units = coh[:FORCED_BATCH_PROBE_N]
    probe = judge_graded(
        [(u.item_id, u.question, u.answer) for u in probe_units],
        registry[COHERENCE_RUBRIC_ID],
        n_draws=JUDGE_N_DRAWS,
        cache_dir=cfg.pilot_cache_root / "_forced_batch",
        save_raw=cfg.raw_dir / "pilot" / "forced_batch_probe.json",
        judge_model=cfg.judge_model,
        max_tokens=cfg.max_tokens,
        threshold_base=0,
    )
    n_probe_scored = sum(1 for v in probe.scores.values() if v is not None)
    probe_ok = n_probe_scored >= 1
    all_pass &= probe_ok

    aggregate = {
        "passed": all_pass,
        "per_rubric": per_rubric,
        "forced_batch_probe": {
            "n_items": len(probe_units),
            "n_scored": n_probe_scored,
            "passed": probe_ok,
            **_telemetry(probe),
        },
        "instrument": {
            "judge_model": cfg.judge_model,
            "max_tokens": cfg.max_tokens,
            "n_draws": JUDGE_N_DRAWS,
        },
        "repro": _repro(),
    }
    _write_json_atomic(cfg.gates_dir / "pilot_gate_report.json", aggregate)
    logger.info("[pilot] aggregate verdict: %s", "PASS" if all_pass else "FAIL")
    return RC_OK if all_pass else RC_PILOT_GATE


# ── phases ─────────────────────────────────────────────────────────────


def phase_anchors(cfg: JudgeConfig) -> int:
    pairs = pair_index()
    anchor_rows = load_anchor_rows(cfg.anchors_file)
    audits = run_audits("anchors", anchor_rows, cfg.audits_dir)
    registry = rubric_registry()

    coh_units = build_coherence_items(None, anchor_rows)
    run_wave(
        "coherence.anchors", COHERENCE_RUBRIC_ID, registry[COHERENCE_RUBRIC_ID], coh_units, cfg
    )
    if cfg.dry_run:
        return RC_OK
    scores = list(_iter_jsonl(cfg.scores_dir / "coherence.anchors.scores.jsonl"))
    gate = coherence_baseline_gate(scores)
    gate["audits"] = audits
    _write_json_atomic(cfg.gates_dir / "coherence_baseline_gate.json", gate)
    logger.info(
        "[gate3] coherence baseline: median=%.1f frac>60=%.3f -> %s",
        gate["median"],
        gate["frac_gt60"],
        "PASS" if gate["passed"] else "FAIL",
    )
    if not gate["passed"]:
        return RC_COHERENCE_GATE  # halts behavior-wave spend only (plan §7 gate 3)

    for rid, units in sorted(build_anchor_behavior_items(anchor_rows, pairs).items()):
        run_wave(f"{rid}.anchors", rid, registry[rid], units, cfg)
    _refresh_summary(cfg)
    return RC_OK


def phase_waves(cfg: JudgeConfig) -> int:
    if not cfg.dry_run:
        _require_gates(cfg)
    pairs = pair_index()
    grid_rows = load_grid_rows(cfg.rollouts_dir)
    run_audits("grid", grid_rows, cfg.audits_dir)
    registry = rubric_registry()

    coh_units = build_coherence_items(grid_rows, None)
    run_wave("coherence.grid", COHERENCE_RUBRIC_ID, registry[COHERENCE_RUBRIC_ID], coh_units, cfg)
    for rid, units in sorted(build_grid_behavior_items(grid_rows, pairs).items()):
        run_wave(f"{rid}.grid", rid, registry[rid], units, cfg)
    if not cfg.dry_run:
        _refresh_summary(cfg)
    return RC_OK


def phase_stage2(cfg: JudgeConfig) -> int:
    if cfg.stage2_dir is None:
        raise RuntimeError("--phase stage2 requires --stage2-dir")
    if not cfg.dry_run:
        _require_gates(cfg)
    pairs = pair_index()
    rows = load_stage2_rows(cfg.stage2_dir)
    run_audits("stage2", rows, cfg.audits_dir)
    registry = rubric_registry()
    coh_units = build_coherence_items(None, None, rows)
    run_wave("coherence.stage2", COHERENCE_RUBRIC_ID, registry[COHERENCE_RUBRIC_ID], coh_units, cfg)
    for rid, units in sorted(build_stage2_behavior_items(rows, pairs).items()):
        run_wave(f"{rid}.stage2", rid, registry[rid], units, cfg)
    if not cfg.dry_run:
        _refresh_summary(cfg)
    return RC_OK


def phase_audits(cfg: JudgeConfig) -> int:
    summaries = []
    if cfg.rollouts_dir.is_dir():
        summaries.append(run_audits("grid", load_grid_rows(cfg.rollouts_dir), cfg.audits_dir))
    if cfg.anchors_file.is_file():
        summaries.append(run_audits("anchors", load_anchor_rows(cfg.anchors_file), cfg.audits_dir))
    if cfg.stage2_dir is not None and cfg.stage2_dir.is_dir():
        summaries.append(run_audits("stage2", load_stage2_rows(cfg.stage2_dir), cfg.audits_dir))
    if not summaries:
        raise RuntimeError("audits: no inputs found (rollouts dir / anchors file / stage2 dir)")
    _write_json_atomic(
        cfg.audits_dir / "audits_summary.json", {"summaries": summaries, "repro": _repro()}
    )
    return RC_OK


def phase_upload_raw(cfg: JudgeConfig) -> int:
    """One folder commit of the judge work root -> the HF judge_raw prefix (plan §10).

    No eligibility filter — the whole work tree uploads (caches live OUTSIDE
    the work root by construction). Well under the 10k/dir cap (~tens of files).
    """
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        cfg.work_root,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_PREFIX}/raw_completions/judge_raw",
        raise_on_error=True,
    )
    logger.info("[upload-raw] uploaded %s -> %s", cfg.work_root, url)
    return RC_OK


def _refresh_summary(cfg: JudgeConfig) -> None:
    """Aggregate every wave meta present into judge_summary.json (rule-18 report)."""
    waves = {}
    for meta_path in sorted(cfg.scores_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        waves[meta["regime"]["wave"]] = {
            "rubric_id": meta["regime"]["rubric_id"],
            "n_items": meta["regime"]["n_items"],
            "n_scored_items": meta.get("n_scored_items"),
            "pass1": meta.get("pass1"),
            "retry1": meta.get("retry1"),
            "residual_transport_lost": meta.get("residual_transport_lost"),
            "per_arm": meta.get("per_arm"),
        }
    total_calls = sum(w["n_items"] for w in waves.values())
    summary = {
        "waves": waves,
        "n_waves": len(waves),
        "total_judge_calls_dispatched": total_calls,
        "instrument": {
            "judge_model": cfg.judge_model,
            "max_tokens": cfg.max_tokens,
            "n_draws": JUDGE_N_DRAWS,
            "temperature_note": "not threaded by judge_graded — Anthropic API default",
        },
        "repro": _repro(),
    }
    _write_json_atomic(cfg.work_root / "judge_summary.json", summary)


# ── config + CLI ───────────────────────────────────────────────────────


@dataclass
class JudgeConfig:
    work_root: Path
    cache_root: Path
    rollouts_dir: Path
    anchors_file: Path
    stage2_dir: Path | None
    judge_model: str = DEFAULT_JUDGE_MODEL
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    dry_run: bool = False

    @property
    def scores_dir(self) -> Path:
        return self.work_root / "scores"

    @property
    def items_dir(self) -> Path:
        return self.work_root / "items"

    @property
    def raw_dir(self) -> Path:
        return self.work_root / "raw"

    @property
    def gates_dir(self) -> Path:
        return self.work_root / "gates"

    @property
    def audits_dir(self) -> Path:
        return self.work_root / "audits"

    @property
    def pilot_cache_root(self) -> Path:
        # PILOT-ONLY cache root, never the production cache (rule 24(ii)).
        return self.cache_root / "_pilot"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2094 VM-side judge pipeline (P6/P9).")
    ap.add_argument(
        "--phase",
        required=True,
        choices=("pilot", "anchors", "waves", "stage2", "audits", "upload-raw"),
    )
    ap.add_argument(
        "--in-root",
        type=Path,
        default=Path("data/issue_2094/judge_inputs"),
        help="staging root; rollouts/anchors default under <in-root>/%s/raw_completions/"
        % HF_PREFIX,
    )
    ap.add_argument("--rollouts-dir", type=Path, default=None)
    ap.add_argument("--anchors-file", type=Path, default=None)
    ap.add_argument("--stage2-dir", type=Path, default=None)
    ap.add_argument(
        "--stage-from-hf",
        action="store_true",
        help="stage grid+anchors (+stage2 for --phase stage2) from HF into --in-root first",
    )
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument("--work-root", type=Path, default=Path("eval_results/issue_2094/judge"))
    ap.add_argument("--cache-root", type=Path, default=Path("data/issue_2094/judge_cache"))
    ap.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build+validate items and print routing with ZERO API calls (wave phases)",
    )
    return ap.parse_args(argv)


def _stage_inputs(args: argparse.Namespace) -> None:
    from explore_persona_space.orchestrate import hub

    prefixes = [f"{HF_PREFIX}/raw_completions/grid", f"{HF_PREFIX}/raw_completions/anchors"]
    if args.phase == "stage2":
        prefixes.append(f"{HF_PREFIX}/raw_completions/stage2")
    for prefix in prefixes:
        staged = hub.stage_hub_prefix(DATASET_REPO, prefix, args.in_root, revision=args.hf_revision)
        logger.info("[stage] %s: %d files", prefix, len(staged))


def build_config(args: argparse.Namespace) -> JudgeConfig:
    mirror = args.in_root / HF_PREFIX / "raw_completions"
    rollouts = args.rollouts_dir if args.rollouts_dir is not None else mirror / "grid"
    anchors = (
        args.anchors_file if args.anchors_file is not None else mirror / "anchors/anchors.jsonl"
    )
    stage2 = args.stage2_dir
    if stage2 is None and args.phase == "stage2":
        stage2 = mirror / "stage2"
    return JudgeConfig(
        work_root=args.work_root,
        cache_root=args.cache_root,
        rollouts_dir=rollouts,
        anchors_file=anchors,
        stage2_dir=stage2,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )


PHASES = {
    "pilot": phase_pilot,
    "anchors": phase_anchors,
    "waves": phase_waves,
    "stage2": phase_stage2,
    "audits": phase_audits,
    "upload-raw": phase_upload_raw,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.dry_run and args.phase in ("pilot", "upload-raw"):
        raise SystemExit(f"--dry-run is not supported for --phase {args.phase}")
    if args.stage_from_hf:
        _stage_inputs(args)
    cfg = build_config(args)
    cfg.work_root.mkdir(parents=True, exist_ok=True)
    rc = PHASES[args.phase](cfg)
    logger.info("[phase=%s] rc=%d", args.phase, rc)
    return rc


if __name__ == "__main__":
    sys.stdout.flush()
    sys.exit(main())
