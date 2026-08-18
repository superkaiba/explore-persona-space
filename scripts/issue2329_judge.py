#!/usr/bin/env python3
"""Issue #2329 — VM-side judge pipeline (thin fork of ``issue2162_judge.py``).

Reuses the #2094 judge machinery wholesale (``scripts/issue2094_judge.py``:
``JudgeUnit`` / ``run_wave`` / wave-regime resume / per-arm drop-split
telemetry / mechanical audits / the form-only coherence instrument VERBATIM)
plus the parent #2162 rubric registry (``bank2162.rubric_pair_2162`` — the
bank STRINGS are byte-verbatim under #2329, so the rubric cores and their
content-hashed ids are identical). Issue-2329 divergences from the parent
judge (plan §7 gates 3-pre/3/6 + divergences 9/12):

- **Surviving pairs only:** every phase derives its pair set from the FROZEN
  gate-0a artifact (``bank.json`` ``dropped_pairs``), matching
  ``issue2329_run.surviving_pairs`` — never the raw 1,404-pair build
  (divergence 9; realized drops at freeze: 0).
- **NEW gate 3-pre (plan §7, S2):** the rule-26 judge pilot runs on the FIRST
  gate-3-slice rollouts BEFORE the gate-3 sync slice's remaining ~8.7k calls
  dispatch — ALL rubric families (coherence + value-rubric + query-rubric) at
  the EXACT production instrument (``claude-sonnet-4-5-20250929``,
  ``max_tokens=1024``, N=1), fresh pilot ``cache_dir``, gated on zero
  ``stop_reason == "max_tokens"`` + per-arm parse-fail < 2% + >= 51 effective
  draws per realized (arm x rubric-family) cell (the rule-26 satisfiability
  floor at the 2% threshold; ~110 draws/cell, ~330 total). The report
  ENUMERATES the realized cells + their effective draws. This is the first
  judge spend on a model the production instrument has never scored; the
  pilot draws are deliberately re-judged in the production slice (fresh
  pilot cache — plan §7 gate 3-pre).
- **Gate 3 aggregate DEMOTED to ADVISORY (divergence 12):** the parent's 60%
  cells-separable bar is computed and REPORTED in
  ``anchor_separation_report.json`` but never halted on; the catastrophic
  < 25%-cells-separable instrument-broken HALT floor is RETAINED. The
  per-pair |sep| >= 0.5 exclusions stay binding at analysis (plan §6).
- **Gate 6 (P6 bulk waves) UNCHANGED:** the parent's per-family rule-26 pilot
  (~440 draws) re-pilots the bulk Batch waves — steered grid + stage-2 text
  is a different content class than the anchors gate 3-pre certified.
- **Stage-2 shards** come from the folded-in ``issue2329_run.py`` stage2
  phase (``stage2_shard_*.jsonl`` under ``raw_completions/stage2``), not a
  separate stage-2 driver (divergence 7).

Gate mechanical routing unchanged: every judge call goes through
``eval.graded_judge.judge_graded`` -> ``eval.batch_judge`` ->
``eval.judge_dispatch.dispatch_judge_items`` -> ``llm/api_dispatch.py``
(Batch API for production waves, drop-never-coerce + transport-retry per
llm-judging rules 9/24/28). All phases are resumable: wave-level meta skip +
the rubric-keyed JudgeCache (production anchor waves REUSE the gate-3
slice's judged draws at zero extra spend).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_judge as J94  # noqa: E402  (same-dir script import; reused machinery)
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.eval.judge_pilot import ArmPilotStats, judge_pilot_gate  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK  # noqa: E402

logger = logging.getLogger("issue2329.judge")

HF_PREFIX = "issue2329_q35rerun"
DATASET_REPO = "superkaiba1/explore-persona-space-data"

RC_OK = 0
RC_PILOT_GATE = 7
RC_COHERENCE_GATE = 8
RC_SEPARATION_GATE = 9
RC_DRY_RUN_UNSUPPORTED = 10  # a phase whose purpose is live measurement
RC_PILOT_GATE3PRE = 11

# Plan §7 gate 3 (divergence 12): the 0.5 bar is the §4.5 per-pair exclusion
# bar (binding at analysis); the 60% aggregate is ADVISORY; < 25% is the
# retained catastrophic instrument-broken HALT floor.
SEPARATION_BAR = 0.5
SEP_MIN_PAIRS_OF_6 = 4
SEP_CELL_FRAC_MIN = 0.60  # ADVISORY (reported, never halted on)
SEP_CATASTROPHIC_FRAC = 0.25  # HALT floor (instrument-broken abort)
# Forces the SYNC api_dispatch route regardless of N (judge_dispatch:
# sync iff n_items <= threshold_base * otpm / 400k).
FORCE_SYNC_THRESHOLD_BASE = 10**9

# Plan §7 gate 6 (P6 bulk waves): draws spanning the rubric FAMILIES.
# Sized to the rule-26 SATISFIABILITY floor (llm-judging.md rule 26, #2124), not
# to a round "~200 draws" habit: the gate FAILs on rate >= parse_fail_threshold
# (0.02), so resolving that threshold needs
#   required = max(min_effective_draws_per_arm=10, floor(1/0.02) + 1) = 51
# effective draws per unwaived arm, and judge_pilot_gate floor-divides the budget
# across arms. With 4 realized arms (anchor / crosstype / shuffled / steered) and
# JUDGE_N_DRAWS == 1 the exact budget form
#   n_arms * n_draws * ceil(required / n_draws) = 4 * 1 * 51
# gives 204. The former 200/120 pair realized 50/30 draws per arm — both
# UNSATISFIABLE, so _config_satisfiability_guard refused before any API spend.
# The item-limited query-rubric family (30 units) still cannot reach the floor at
# ANY budget; it is checked at realized capacity via allow_subresolution_pilot
# below (see the fam_reps comment), which is the auditable escape, not a
# loosened threshold.
PILOT_TARGET_COHERENCE = 204
PILOT_TARGET_BEHAVIOR = 204
PILOT_SEED = 2329

# Plan §7 gate 3-pre (S2): ~110 draws per realized (arm x rubric-family) cell
# (3 realized cells on the anchor-only gate slice => ~330 total, inside the
# plan's ~200-440 band), each cell floored at 51 effective draws — the rule-26
# satisfiability floor at the 2% parse-fail threshold (llm-judging.md #2124:
# required = max(min_effective, floor(1/0.02) + 1) = 51).
GATE3PRE_TARGET_PER_FAMILY = 110
GATE3PRE_MIN_EFFECTIVE_DRAWS = 51

JUDGE_N_DRAWS = J94.JUDGE_N_DRAWS  # 1 — the pair-clustered bootstrap carries uncertainty

_DEFAULT_IN_ROOT = Path("data/issue_2329/judge_inputs")
_BANK_JSON_REL = f"{HF_PREFIX}/analysis_tensors/vc_bank/bank.json"
# Committed byte-copy of the FROZEN pre-regen cap-hit basis driving the anchors
# capregen campaign (issue2329_run.py --phase capregen; sha256 78385e71b245...).
# Scopes the gate-3 breach-cell row-cap staleness check — the breach set is
# always READ from this artifact, never hardcoded.
_DEFAULT_BREACH_BASIS = Path("eval_results/issue_2329/cap_hit/cap_hit_report_anchors_preregen.json")


@dataclass
class JudgeConfig(J94.JudgeConfig):
    """The #2094 JudgeConfig + the frozen gate-0a bank.json (drop registry)."""

    bank_json: Path = _DEFAULT_IN_ROOT / _BANK_JSON_REL
    breach_basis: Path = _DEFAULT_BREACH_BASIS
    # rule-28 remediation seam (#2151/#1739): route production waves over the
    # SYNC transport instead of Batch. Default False => every existing caller
    # routes byte-identically. Set only for an api-refusal RE-ISSUE pass: the
    # Batch classifier censored 764 draws (stop_reason == 'refusal', empty
    # content) whose in-band retry cannot help, because the retry envelope
    # re-issues on the SAME censoring transport. The instrument is UNCHANGED
    # (judge_model / rubric / max_tokens / n_draws) -- only the HTTP transport
    # differs, which is what makes the merge licensable. Genuinely-scored draws
    # come back as JudgeCache HITS and are never re-spent; api-refusal dicts are
    # PUT-skipped and read as cache MISSES by design (batch_judge.py:681/910),
    # so a sync re-run re-issues exactly the censored set.
    force_sync_routing: bool = False

    @property
    def wave_threshold_base(self) -> int | None:
        """``threshold_base`` for J94.run_wave: the force-sync sentinel or None."""
        return FORCE_SYNC_THRESHOLD_BASE if self.force_sync_routing else None

    @property
    def pilot_gate3pre_cache_root(self) -> Path:
        # FRESH gate-3-pre pilot cache — distinct from BOTH the production
        # cache_root and the gate-6 ``_pilot`` root (rule 24(ii); plan §7:
        # pilot draws are re-judged in the production slice, deliberately).
        return self.cache_root / "_pilot_gate3pre"


# ── surviving pairs (divergence 9 — the frozen drop registry) ─────────

_PAIRS_CACHE: dict[str, list[BANK.Pair2162]] = {}


def surviving_pairs(bank_json: Path) -> list[BANK.Pair2162]:
    """The parent's 1,404 pairs minus the gate-0a token-identity drops.

    Reads the FROZEN gate-0a artifact — never re-tokenizes. Predicate + count
    assert match ``issue2329_run.surviving_pairs`` (one drop registry, no
    re-derivation drift). Cached per resolved path (build_pairs is ~O(1s)).
    """
    key = str(Path(bank_json).resolve())
    if key not in _PAIRS_CACHE:
        bank_json = Path(bank_json)
        assert bank_json.is_file(), (
            f"{bank_json} missing — pass --bank-json or --stage-from-hf; the frozen "
            "gate-0a bank.json is the pair-drop registry every phase derives from"
        )
        manifest = json.loads(bank_json.read_text())
        dropped = {row["pair_id"] for row in manifest["dropped_pairs"]}
        pairs = [p for p in BANK.build_pairs() if p.pair_id not in dropped]
        assert len(pairs) == manifest["token_identity"]["n_intact"], (
            len(pairs),
            manifest["token_identity"]["n_intact"],
        )
        _PAIRS_CACHE[key] = pairs
    return _PAIRS_CACHE[key]


# ── rubric registry (dynamic, content-hashed ids) ─────────────────────


def rubric_core_id(core: str) -> str:
    """Stable content-hashed rubric id (id grammar: ``^[a-zA-Z0-9-]{1,53}$``)."""
    return "f" + hashlib.sha1(core.encode("utf-8")).hexdigest()[:12]


def pair_rubric_cores(pair: BANK.Pair2162) -> tuple[str, str] | None:
    """(core_A, core_B) or None for the no-rubric ``filler_swap`` class."""
    if BANK.base_type_of(pair.cell) == "filler_swap":
        return None
    return BANK.rubric_pair_2162(pair)


def rubric_registry(pairs: list[BANK.Pair2162]) -> dict[str, str]:
    """rubric_id -> production eval_prompt (coherence + every distinct core)."""
    reg = {J94.COHERENCE_RUBRIC_ID: J94.coherence_eval_prompt()}
    for p in pairs:
        cores = pair_rubric_cores(p)
        if cores is None:
            continue
        for core in cores:
            reg.setdefault(rubric_core_id(core), J94.behavior_eval_prompt(core))
    return reg


def _is_query_rubric(prompt: str) -> bool:
    return "an answer to the following question" in prompt


# ── input walkers (2329 shard formats) ────────────────────────────────


def load_grid_rows(rollouts_dir: Path) -> list[dict]:
    shards = sorted(rollouts_dir.glob("shard_*.jsonl"))
    assert shards, f"no grid shards under {rollouts_dir}"
    rows = [r for shard in shards for r in J94._iter_jsonl(shard)]
    assert rows, "grid shards present but empty"
    for r in rows[:1]:
        for key in ("cell", "slot", "arm", "pair_id", "draw", "text", "context_id"):
            assert key in r, (key, sorted(r))
    return rows


def load_anchor_rows(anchors_dir: Path) -> list[dict]:
    files = sorted(anchors_dir.glob("anchors_*.jsonl"))
    assert files, f"no anchor shards under {anchors_dir}"
    rows: list[dict] = []
    for f in files:
        for r in J94._iter_jsonl(f):
            # Shard provenance for the gate-3 staleness check's error naming
            # (underscore-keyed; every consumer reads named fields only, so it
            # never leaks into judge units or persisted artifacts).
            r["_shard"] = f.name
            rows.append(r)
    assert rows, "anchor shards present but empty"
    for r in rows[:1]:
        for key in ("context_id", "cell", "value_id", "carrier", "draw", "text"):
            assert key in r, (key, sorted(r))
    # r2 F3 loud backstop: anchor shard names are width-unnamespaced, so a
    # stale prior-width shard (an 8->4 reshard's surviving w4..w7 files, on
    # disk or on the HF prefix this dir was staged from) duplicates every one
    # of its (context_id, draw) units into the coherence gate + behavior
    # waves. One row per unit, or fail loud naming the duplicate.
    seen: set[tuple[str, int]] = set()
    for r in rows:
        unit = (r["context_id"], r["draw"])
        assert unit not in seen, (
            f"duplicate anchor row {unit} across {anchors_dir}/anchors_*.jsonl — "
            "stale prior-width shard (the run driver quarantines these at phase "
            "entry; sweep the staged/HF copy before judging)"
        )
        seen.add(unit)
    return rows


def load_stage2_rows(stage2_dir: Path) -> list[dict]:
    # Divergence 7: stage-2 rides issue2329_run.py, which shards as
    # ``stage2_shard_*.jsonl`` (uploaded under raw_completions/stage2).
    shards = sorted(stage2_dir.glob("stage2_shard_*.jsonl"))
    assert shards, f"no stage2 shards (stage2_shard_*.jsonl) under {stage2_dir}"
    return [r for shard in shards for r in J94._iter_jsonl(shard)]


# ── item builders ─────────────────────────────────────────────────────


def _grid_source(row: dict, kind: str = "grid") -> dict:
    return {
        "kind": kind,
        "arm": row["arm"],
        "cell": row["cell"],
        "slot": row["slot"],
        "pair_id": row["pair_id"],
        "draw": row["draw"],
        "context_id": row["context_id"],
    }


def _anchor_source(row: dict) -> dict:
    return {
        "kind": "anchor",
        "cell": row["cell"],
        "value_id": row["value_id"],
        "carrier": row["carrier"],
        "draw": row["draw"],
        "context_id": row["context_id"],
    }


def build_coherence_items(
    grid_rows: list[dict] | None = None,
    anchor_rows: list[dict] | None = None,
    stage2_rows: list[dict] | None = None,
) -> list[J94.JudgeUnit]:
    """One form-only coherence call per rollout (plan §4.5: EVERY rollout)."""
    units: list[J94.JudgeUnit] = []
    for row in grid_rows or []:
        key = f"g|{row['block_key']}|{row['pair_id']}|{row['draw']}"
        units.append(
            J94.JudgeUnit(
                item_id=J94._item_id("c", key),
                rubric_id=J94.COHERENCE_RUBRIC_ID,
                question="",
                answer=row["text"],
                source=_grid_source(row),
            )
        )
    for row in anchor_rows or []:
        key = f"a|{row['context_id']}|{row['draw']}"
        units.append(
            J94.JudgeUnit(
                item_id=J94._item_id("c", key),
                rubric_id=J94.COHERENCE_RUBRIC_ID,
                question="",
                answer=row["text"],
                source=_anchor_source(row),
            )
        )
    for row in stage2_rows or []:
        key = f"s|{row['block_key']}|{row['pair_id']}|{row['draw']}"
        units.append(
            J94.JudgeUnit(
                item_id=J94._item_id("c", key),
                rubric_id=J94.COHERENCE_RUBRIC_ID,
                question="",
                answer=row["text"],
                source=_grid_source(row, kind="stage2"),
            )
        )
    return units


def _behavior_unit(tag: str, key: str, rid: str, row: dict, source: dict) -> J94.JudgeUnit:
    return J94.JudgeUnit(
        item_id=J94._item_id(tag, key),
        rubric_id=rid,
        question="",
        answer=row["text"],
        source=source,
    )


def build_grid_behavior_items(
    rows: list[dict],
    pairs_by_id: dict[str, BANK.Pair2162],
    tag: str = "g",
    kind: str = "grid",
) -> dict[str, list[J94.JudgeUnit]]:
    """Dual-rubric behavior units per grid/stage2 rollout (plan §4.4 F_beh)."""
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    for row in rows:
        cores = pair_rubric_cores(pairs_by_id[row["pair_id"]])
        if cores is None:
            continue  # filler_swap: disruption DV only, no F rubric (§4.4)
        for side, core in zip(("a", "b"), cores, strict=True):
            rid = rubric_core_id(core)
            key = f"{tag}|{row['block_key']}|{row['pair_id']}|{row['draw']}|{side}"
            src = {**_grid_source(row, kind=kind), "side": side}
            by_rid.setdefault(rid, []).append(_behavior_unit(tag, key, rid, row, src))
    return by_rid


def anchor_unit_id(context_id: str, draw: int, rid: str) -> str:
    """Deterministic anchor-behavior item id — shared between the gate-3 slice
    and the production anchor waves (the cache/dedup join key)."""
    return J94._item_id("a", f"a|{context_id}|{draw}|{rid}")


def build_anchor_behavior_items(
    anchor_rows: list[dict],
    pairs: list[BANK.Pair2162],
    restrict_pairs: list[BANK.Pair2162] | None = None,
) -> dict[str, list[J94.JudgeUnit]]:
    """Anchor floor/ceiling units: each anchor draw judged under EVERY rubric
    core of every pair its context participates in, deduplicated per
    (context, draw, rubric) — the F_beh floor/ceiling terms (plan §4.4)."""
    use = restrict_pairs if restrict_pairs is not None else pairs
    cores_by_ctx: dict[str, set[str]] = {}
    for p in use:
        cores = pair_rubric_cores(p)
        if cores is None:
            continue
        for ctx in (p.a, p.b):
            cores_by_ctx.setdefault(ctx, set()).update(cores)
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    seen: set[str] = set()
    for row in anchor_rows:
        for core in sorted(cores_by_ctx.get(row["context_id"], ())):
            rid = rubric_core_id(core)
            iid = anchor_unit_id(row["context_id"], row["draw"], rid)
            if iid in seen:
                continue
            seen.add(iid)
            src = {**_anchor_source(row), "rubric": rid}
            by_rid.setdefault(rid, []).append(
                J94.JudgeUnit(
                    item_id=iid, rubric_id=rid, question="", answer=row["text"], source=src
                )
            )
    return by_rid


# ── uniform --dry-run ─────────────────────────────────────────────────
#
# ONE meaning across phases: build + validate every judge unit the phase
# would dispatch, print counts/routing, make ZERO API calls, persist NOTHING
# under the work root. Phases whose purpose IS live measurement (pilot /
# pilot-gate3pre) REFUSE loudly (``RC_DRY_RUN_UNSUPPORTED``).


def _dry_run_units_report(phase: str, waves: dict[str, list[J94.JudgeUnit]]) -> int:
    """Construction check only — validate units per wave, log counts, exit 0."""
    total = 0
    for wave, units in sorted(waves.items()):
        J94._validate_units(units)
        total += len(units)
        logger.info("[%s] dry-run wave %s: %d units", phase, wave, len(units))
    logger.info(
        "[%s] dry-run complete: %d units across %d waves (no API calls made, nothing persisted)",
        phase,
        total,
        len(waves),
    )
    return RC_OK


# ── gate slice shared derivation ──────────────────────────────────────


def _assert_gate_rows_capregen_fresh(breach_basis: Path, gate_ctx_rows: list[dict]) -> None:
    """Gate-3 pre-regen staleness backstop (#2329 reconciler v12, Dispute 2).

    ``_resolve_anchors_dir`` prefers the full ``anchors`` prefix on shard-NAME
    coverage — coverage says nothing about FRESHNESS, so a capregen shard
    whose upload failed (or was skipped) keeps PRE-regen bytes under a
    covering name and the judge cannot tell. This check replaces the
    procedural protection ("open the judge window only after all 8 gate
    workers exit rc==0") with a mechanical one: once the frozen capregen
    breach basis exists, every staged gate row in a BREACHING cell must carry
    the raised per-row ``max_new_tokens`` (>= 2x the basis's generating cap —
    the registered remedy issue2329_run.py enforces; regen rows carry it via
    ``_enrich_rows_with_capture``, merged keep-rows are backfilled with the
    base cap). Non-breaching cells legitimately stay at the base cap and are
    never checked.

    Arming rules (fail-loud, no silent defaults):

    - basis file PRESENT -> the check runs; any breach-cell row below the
      raised cap (or lacking the per-row field — pre-diff base-run rows never
      carried one) RAISES naming shard + cell + observed caps.
    - basis file ABSENT + staged gate rows carry MIXED per-row caps -> RAISE:
      only a capregen merge produces mixed caps, so a missing basis there is
      a misconfiguration (wrong cwd / wrong --breach-basis), never a fresh
      run.
    - basis file ABSENT + uniform/absent row caps -> the legitimate
      pre-capregen fresh-run ordering (gate 3-pre historically runs before
      any cap-hit analysis exists): SKIP with a WARNING naming the path.
    """
    caps_seen = sorted({int(r["max_new_tokens"]) for r in gate_ctx_rows if "max_new_tokens" in r})
    if not breach_basis.exists():
        if len(caps_seen) > 1:
            raise RuntimeError(
                f"gate-3 staging: staged gate rows carry MIXED per-row caps {caps_seen} — "
                "only a capregen merge produces mixed caps, so the pre-regen breach basis "
                f"MUST be available to scope the staleness check, but {breach_basis} does "
                "not exist (wrong cwd? pass --breach-basis); refusing to stage "
                "unverifiable gate rows"
            )
        logger.warning(
            "[gate3-capcheck] breach basis %s not found — gate-3 pre-regen staleness "
            "check SKIPPED. Legitimate ONLY for a pre-capregen fresh-run ordering (no "
            "cap-hit analysis exists yet); after a capregen campaign this means a wrong "
            "cwd or missing checkout — verify before opening the judge window.",
            breach_basis,
        )
        return
    rep = json.loads(breach_basis.read_text(encoding="utf-8"))
    # Mirror issue2329_run._validate_breach_basis's refusal semantics for the
    # fields this check consumes (a wrong basis must never scope it). The
    # run-side validator is not importable here without a RunConfig — and its
    # >=2x check binds the CAPREGEN CLI cap, which has no judge-side analogue.
    if rep.get("scope") != "anchors":
        raise RuntimeError(
            f"breach basis {breach_basis} has scope={rep.get('scope')!r}, need 'anchors'"
        )
    if rep.get("postregen"):
        raise RuntimeError(
            f"breach basis {breach_basis} is a POST-regen measurement (postregen: true) — "
            "it can never scope the gate-3 staleness check; point --breach-basis at the "
            "frozen PRE-regen basis"
        )
    if "partial" not in rep or rep["partial"]:
        raise RuntimeError(
            f"breach basis {breach_basis} is PARTIAL or lacks the 'partial' field "
            f"(partial={rep.get('partial')!r}) — not a complete pre-regen basis"
        )
    base_cap = int(rep["max_new_tokens"])
    caps = rep.get("realized_row_caps") or []
    if [int(c) for c in caps] != [base_cap]:
        raise RuntimeError(
            f"breach basis {breach_basis} declares max_new_tokens={base_cap} but measured "
            f"realized_row_caps={caps} — a wrong-cap / mixed-cap basis can never scope "
            "the gate-3 staleness check"
        )
    breach = set(rep["breaching_cells"])
    required = 2 * base_cap
    if not breach:
        logger.info(
            "[gate3-capcheck] basis %s has an EMPTY breach list — no capregen mandated, "
            "nothing to verify",
            breach_basis.name,
        )
        return
    stale: dict[tuple[str, str], dict] = {}
    n_checked = 0
    for r in gate_ctx_rows:
        if r["cell"] not in breach:
            continue
        n_checked += 1
        cap = r.get("max_new_tokens")
        if cap is None or int(cap) < required:
            key = (str(r.get("_shard", "<unknown-shard>")), r["cell"])
            entry = stale.setdefault(key, {"n": 0, "caps": set()})
            entry["n"] += 1
            entry["caps"].add("absent" if cap is None else int(cap))
    if stale:
        detail = "; ".join(
            f"shard={s} cell={c} rows={e['n']} caps={sorted(e['caps'], key=str)}"
            for (s, c), e in sorted(stale.items())
        )
        raise RuntimeError(
            f"gate-3 staging: {sum(e['n'] for e in stale.values())} gate row(s) in "
            f"BREACHING cells carry a pre-regen cap (< {required}) — a PRE-REGEN shard is "
            "being staged into the gate-3 window: the capregen gate upload for the named "
            "shard(s) has not landed (or was skipped), and _resolve_anchors_dir's "
            "name-coverage preference cannot tell freshness. Re-run / re-upload the "
            f"failed capregen gate worker(s) first. Offenders: {detail} (basis "
            f"{breach_basis.name}: {len(breach)} breaching cells, base cap {base_cap})"
        )
    if n_checked == 0:
        logger.warning(
            "[gate3-capcheck] zero staged gate rows fall in the basis's %d breaching "
            "cells — vacuous pass (legitimate only when every breaching cell lies "
            "outside the gate slice; a cell-name drift between basis and rows looks "
            "identical — eyeball the basis before trusting this)",
            len(breach),
        )
        return
    logger.info(
        "[gate3-capcheck] %d breach-cell gate rows verified at max_new_tokens >= %d "
        "(basis %s: %d breaching cells)",
        n_checked,
        required,
        breach_basis.name,
        len(breach),
    )


def _gate_slice_inputs(
    cfg: JudgeConfig,
) -> tuple[list[BANK.Pair2162], list[BANK.Pair2162], list[dict], dict[str, str]]:
    """(pairs, gate_pairs, gate ctx anchor rows, rubric registry) — shared by
    gate 3-pre and gate 3 (one derivation; the pod driver generates the SAME
    ``BANK.gate_slice_pairs(surviving)`` slice first in P2). Every gate-3
    entry path — pilot-gate3pre and separation-gate alike, whatever staging
    route or --anchors-dir override supplied the rows — funnels through here,
    so the capregen-freshness backstop below cannot be bypassed."""
    pairs = surviving_pairs(cfg.bank_json)
    gate_pairs = BANK.gate_slice_pairs(pairs)
    anchor_rows = load_anchor_rows(cfg.anchors_file)
    have_ctx = {r["context_id"] for r in anchor_rows}
    need_ctx = {c for p in gate_pairs for c in (p.a, p.b)}
    missing = sorted(need_ctx - have_ctx)
    if missing:
        raise RuntimeError(
            f"gate3: {len(missing)} gate-slice contexts have no anchor rows yet "
            f"(first: {missing[:3]}) — the P2 gate shards are incomplete"
        )
    gate_ctx_rows = [r for r in anchor_rows if r["context_id"] in need_ctx]
    _assert_gate_rows_capregen_fresh(cfg.breach_basis, gate_ctx_rows)
    return pairs, gate_pairs, gate_ctx_rows, rubric_registry(pairs)


# ── gate 3-pre: rule-26 pilot on the gate-3 slice (plan §7, S2) ───────


def _pilot_arm(u: J94.JudgeUnit) -> str:
    src = u.source
    if src["kind"] in ("anchor", "stage2"):
        return src["kind"]
    return src.get("arm", "unknown")


def _family_representative(
    by_rid: dict[str, list[J94.JudgeUnit]], registry: dict[str, str], query_family: bool
) -> str:
    cands = {
        rid: units
        for rid, units in by_rid.items()
        if _is_query_rubric(registry[rid]) == query_family
    }
    assert cands, f"no rubric ids for family query={query_family}"
    return max(cands, key=lambda rid: len(cands[rid]))


def _family_min_effective_floor(arms: dict[str, list[tuple[str, str, str]]], n_draws: int) -> int:
    """Feasibility-aware per-family verdict floor for the gate-3-pre pilot.

    ``GATE3PRE_MIN_EFFECTIVE_DRAWS`` (51 = floor(1/0.02)+1, the rule-26(b)
    resolution floor) stays the CEILING and binds whenever every arm holds
    >= 51 draws' worth of items (coherence 4,240 / value-rubric 640 units on
    the realized gate slice). An ITEM-LIMITED family — the realized
    query-rubric representative holds 30 units, and the production instrument
    is pinned at ``n_draws=1`` (rules 23/26: the pilot runs the EXACT
    production instrument, which draws once per item), so no
    ``target_total_draws`` can raise its draw count — gets its own realized
    capacity ``item count * n_draws`` as the floor instead: the verdict floor
    in ``judge_pilot._gate_verdict`` is UNCONDITIONAL (no exemption for
    sub-resolution-bypassed or waived arms), so keeping the 51 ceiling would
    deterministically FAIL the family on arithmetic the data cannot satisfy.
    Residual: that family's parse-fail check resolves at 1/answered-draws —
    equal to ``1/floor`` only when every draw is answered (3.3% for 30
    draws) — instead of 2%; recorded per family in the phase report
    (``per_family[*].parse_fail_resolution_pct``).

    SINGLE-ARM PRECONDITION (asserted, #2329 review v10 Minor 3):
    ``judge_pilot._gate_verdict`` compares EVERY arm against this one scalar
    floor, so a min-over-arms derivation would silently UNDER-ENFORCE the
    larger arms of a multi-arm family; per-arm floors are inexpressible
    without a ``judge_pilot`` library change (``min_effective_draws_per_arm``
    is a scalar kwarg). Fail loud here, never a silent widening — re-derive
    per arm before widening the pilot slice to a second arm class.

    Raises ``ValueError`` on ``n_draws < 1``: a non-positive draw count is
    nonsensical for this gate (the production instrument draws >= 1 time per
    item), and unvalidated it would derive a 0 floor that every arm trivially
    clears — a silent PASS on zero evidence. The ``max(1, .)`` below is kept
    for parity with ``judge_pilot``'s own ``d_eff = max(1, n_draws)`` clamp
    (judge_pilot.py:327) and only ever operates on validated input.
    """
    assert arms, "no arms"
    assert len(arms) == 1, (
        f"_family_min_effective_floor assumes a SINGLE-ARM gate3pre family, got arms="
        f"{sorted(arms)}: judge_pilot._gate_verdict floors EVERY arm at this one scalar, "
        "so min-over-arms would silently under-enforce the larger arms; per-arm floors "
        "need a judge_pilot library change (min_effective_draws_per_arm is scalar) — "
        "re-derive per arm before widening the pilot slice"
    )
    if n_draws < 1:
        raise ValueError(
            f"n_draws={n_draws} < 1 is nonsensical for the gate-3-pre floor derivation: "
            "the pilot runs the production instrument (>= 1 draw per item); an "
            "unvalidated non-positive count would derive a 0 floor that every arm "
            "trivially clears — a silent PASS on zero evidence"
        )
    d_eff = max(1, n_draws)  # judge_pilot d_eff parity (input validated above)
    min_arm_capacity = min(len(items) for items in arms.values()) * d_eff
    return min(GATE3PRE_MIN_EFFECTIVE_DRAWS, min_arm_capacity)


def _family_resolution_fields(arms: dict[str, ArmPilotStats], family_floor: int) -> dict:
    """Resolution-disclosure fields for one family's aggregate-report row.

    ``effective_draws_min`` mirrors the EXACT quantity ``_gate_verdict``
    floors on (``n_draws - n_transport_lost``, judge_pilot.py:240).
    ``parse_fail_resolution_pct`` — the smallest observable nonzero parse-fail
    rate — instead uses the ANSWERED denominator (``- n_api_refusal`` too),
    the denominator of ``judge_pilot``'s own ``parse_fail_rate``
    (judge_pilot.py:580-601) and ``_runtime_shrink_warnings`` (:422): rule 28
    api-refusal draws leave the answered pool exactly as transport losses do,
    so the effective count would OVERSTATE the check's fineness on an
    api-refusal-bearing wave (#2329 review v10 Minor 1 / codex BLOCKER).
    ``answered_draws_min`` makes the denominator auditable in the report. A
    fully-censored family (0 answered draws) reports ``None`` — never a
    coerced number. ``sub_resolution`` keys on the CONFIGURED relaxation
    (``floor_applied < floor_ceiling``), never on realized draws: a
    transport-hollowed FULL-strength family FAILs the gate but was not a
    deliberate relaxation (v10 Minor 2).
    """
    min_effective = min(st.n_draws - st.n_transport_lost for st in arms.values())
    min_answered = min(st.n_draws - st.n_transport_lost - st.n_api_refusal for st in arms.values())
    return {
        "floor_applied": family_floor,
        "floor_ceiling": GATE3PRE_MIN_EFFECTIVE_DRAWS,
        "effective_draws_min": min_effective,
        "answered_draws_min": min_answered,
        "sub_resolution": bool(family_floor < GATE3PRE_MIN_EFFECTIVE_DRAWS),
        "parse_fail_resolution_pct": (round(100.0 / min_answered, 2) if min_answered > 0 else None),
    }


def phase_pilot_gate3pre(cfg: JudgeConfig) -> int:
    """Plan §7 gate 3-pre (S2): rule-26 pilot on the FIRST gate-3-slice rollouts.

    Runs BEFORE the gate-3 sync slice's remaining ~8.7k calls dispatch
    (``phase_separation_gate`` requires this report present-and-PASSED). ALL
    rubric families at the exact production instrument; fresh pilot
    ``cache_dir``; PASS <=> zero ``stop_reason == "max_tokens"`` (unconditional,
    never waived) AND per-arm parse-fail < 2% AND effective draws per realized
    (arm x rubric-family) cell >= the feasibility-aware per-family floor
    (:func:`_family_min_effective_floor` — 51 for coherence/value-rubric; an
    item-limited family runs SUB-RESOLUTION at its realized capacity, recorded
    per family in the report). The report enumerates the realized cells and
    their effective draws. REFUSES ``--dry-run``: the gate exists to measure
    the real instrument's drop profile on a model it has never scored."""
    if cfg.dry_run:
        logger.error(
            "[pilot-gate3pre] --dry-run refused: the rule-26 pilot's whole purpose is "
            "measuring the REAL instrument's truncation/parse-fail profile on the new "
            "model's first rollouts — there is no meaningful zero-API pilot. Run without "
            "--dry-run to spend the ~%d pilot draws.",
            3 * GATE3PRE_TARGET_PER_FAMILY,
        )
        return RC_DRY_RUN_UNSUPPORTED
    _pairs, gate_pairs, gate_ctx_rows, registry = _gate_slice_inputs(cfg)
    coh = build_coherence_items(None, gate_ctx_rows)
    beh = build_anchor_behavior_items(gate_ctx_rows, _pairs, restrict_pairs=gate_pairs)
    fam_reps = {
        "coherence": J94.COHERENCE_RUBRIC_ID,
        "value-rubric": _family_representative(beh, registry, query_family=False),
        "query-rubric": _family_representative(beh, registry, query_family=True),
    }
    per_family: dict[str, dict] = {}
    cells: list[dict] = []
    all_pass = True
    for family, rid in fam_reps.items():
        units = coh if rid == J94.COHERENCE_RUBRIC_ID else beh[rid]
        J94._validate_units(units)
        arms: dict[str, list[tuple[str, str, str]]] = {}
        for u in units:
            arms.setdefault(_pilot_arm(u), []).append((u.item_id, u.question, u.answer))
        # Feasibility-aware per-family floor (#2329): the ceiling 51 stays
        # binding for coherence/value-rubric; the item-limited query-rubric
        # family (30 units, n_draws pinned at 1 by the production instrument
        # — no target_total_draws can fix it) is checked at its realized
        # capacity. allow_subresolution_pilot=True clears the #2124
        # config-time guard for exactly that family; the relaxation is
        # recorded per family in the aggregate below. Rule 26(a)'s truncation
        # check is untouched — _gate_verdict applies it unconditionally.
        family_floor = _family_min_effective_floor(arms, JUDGE_N_DRAWS)
        report = judge_pilot_gate(
            arms,
            registry[rid],
            max_tokens=cfg.max_tokens,
            cache_dir=cfg.pilot_gate3pre_cache_root / rid,
            save_raw_dir=cfg.raw_dir / "pilot_gate3pre" / rid,
            n_draws=JUDGE_N_DRAWS,
            target_total_draws=GATE3PRE_TARGET_PER_FAMILY,
            min_effective_draws_per_arm=family_floor,
            allow_subresolution_pilot=True,
            judge_model=cfg.judge_model,
            report_path=cfg.gates_dir / "pilot_gate3pre" / f"{family}.json",
            seed=PILOT_SEED,
        )
        per_family[family] = {
            "rubric_id": rid,
            "verdict": report.verdict,
            "failures": report.failures,
            "warnings": report.warnings,
            "n_total_draws": report.n_total_draws,
            **_family_resolution_fields(report.arms, family_floor),
        }
        # Realized (arm x rubric-family) cell enumeration (plan §7 gate 3-pre:
        # "the pilot report enumerates realized cells + their effective draws").
        for arm, st in sorted(report.arms.items()):
            effective = st.n_draws - st.n_transport_lost
            cells.append(
                {
                    "family": family,
                    "arm": arm,
                    "rubric_id": rid,
                    "n_draws": st.n_draws,
                    "n_transport_lost": st.n_transport_lost,
                    "n_api_refusal": st.n_api_refusal,
                    "effective_draws": effective,
                    "floor": family_floor,
                    "floor_ceiling": GATE3PRE_MIN_EFFECTIVE_DRAWS,
                    "meets_floor": bool(effective >= family_floor),
                    "parse_fail_rate": st.parse_fail_rate,
                    "stop_reason_tally": st.stop_reason_tally,
                }
            )
        all_pass &= report.passed
        logger.info(
            "[pilot-gate3pre] %s (%s): %s (%d draws; floor %d/%d%s)",
            family,
            rid,
            report.verdict,
            report.n_total_draws,
            family_floor,
            GATE3PRE_MIN_EFFECTIVE_DRAWS,
            ", SUB-RESOLUTION" if family_floor < GATE3PRE_MIN_EFFECTIVE_DRAWS else "",
        )
    aggregate = {
        "criterion": (
            "rule-26 judge pilot on the FIRST gate-3-slice rollouts, BEFORE the gate-3 "
            "sync slice bulk dispatch (plan §7 gate 3-pre, S2): zero "
            "stop_reason=='max_tokens' (unconditional, never waived) + per-arm "
            "parse-fail < 2% + effective draws per realized (arm x rubric-family) "
            f"cell >= min({GATE3PRE_MIN_EFFECTIVE_DRAWS}, the family's realized arm "
            "item count x n_draws) — the feasibility-aware per-family floor: an "
            "item-limited family runs SUB-RESOLUTION (per_family[*].sub_resolution, "
            "keyed on floor_applied < floor_ceiling) and its parse-fail check "
            "resolves at 1/answered-draws (per_family[*].parse_fail_resolution_pct; "
            "= 1/floor when every draw is answered) instead of 2%"
        ),
        "passed": all_pass,
        "per_family": per_family,
        "cells": cells,
        "instrument": {
            "judge_model": cfg.judge_model,
            "max_tokens": cfg.max_tokens,
            "n_draws": JUDGE_N_DRAWS,
            "target_per_family": GATE3PRE_TARGET_PER_FAMILY,
            "min_effective_draws_ceiling": GATE3PRE_MIN_EFFECTIVE_DRAWS,
            "seed": PILOT_SEED,
        },
        "repro": J94._repro(),
    }
    J94._write_json_atomic(cfg.gates_dir / "pilot_gate3pre_report.json", aggregate)
    logger.info("[pilot-gate3pre] aggregate verdict: %s", "PASS" if all_pass else "FAIL")
    return RC_OK if all_pass else RC_PILOT_GATE3PRE


# ── gate 3: anchor-separation early read (ADVISORY + HALT floor) ──────


def _sep_scores_for_slice(
    cfg: JudgeConfig,
    registry: dict[str, str],
    gate_units: dict[str, list[J94.JudgeUnit]],
    dry_run: bool,
) -> dict[str, float | None]:
    """Judge the gate-slice units SYNC; returns item_id -> mean kept score."""
    scores: dict[str, float | None] = {}
    for rid, units in sorted(gate_units.items()):
        J94._validate_units(units)
        result = judge_graded(
            [(u.item_id, u.question, u.answer) for u in units],
            registry[rid],
            n_draws=JUDGE_N_DRAWS,
            cache_dir=cfg.cache_root / rid,
            save_raw=cfg.raw_dir / "separation_gate" / f"{rid}.json",
            judge_model=cfg.judge_model,
            max_tokens=cfg.max_tokens,
            threshold_base=FORCE_SYNC_THRESHOLD_BASE,  # SYNC api_dispatch fan-out
            dry_run=dry_run,
        )
        if dry_run:
            scores.update({u.item_id: None for u in units})
            continue
        scores.update({u.item_id: result.scores.get(u.item_id) for u in units})
    return scores


def separation_verdict(
    gate_pairs: list[BANK.Pair2162],
    anchor_rows: list[dict],
    scores: dict[str, float | None],
) -> dict:
    """Per-pair sep = Delta(ceiling ctx B) - Delta(floor ctx A); per-cell >= 4/6
    at |sep| >= 0.5. The aggregate 60% read is ADVISORY (divergence 12);
    ``passed`` is the BINDING read = NOT catastrophic (>= 25% cells separable)."""
    draws_by_ctx: dict[str, list[int]] = {}
    for row in anchor_rows:
        draws_by_ctx.setdefault(row["context_id"], []).append(row["draw"])
    per_cell: dict[str, dict] = {}
    pair_rows: list[dict] = []
    for p in gate_pairs:
        cores = pair_rubric_cores(p)
        assert cores is not None, p.pair_id  # gate slice excludes filler_swap
        rid_a, rid_b = (rubric_core_id(c) for c in cores)

        def _delta(ctx: str) -> float | None:
            deltas = []
            for draw in draws_by_ctx.get(ctx, []):
                sa = scores.get(anchor_unit_id(ctx, draw, rid_a))
                sb = scores.get(anchor_unit_id(ctx, draw, rid_b))
                if sa is None or sb is None:
                    continue  # rule-9 drop: excluded, never coerced
                deltas.append((sb - sa) / 100.0)
            return sum(deltas) / len(deltas) if deltas else None

        d_floor, d_ceiling = _delta(p.a), _delta(p.b)
        sep = None if (d_floor is None or d_ceiling is None) else d_ceiling - d_floor
        pair_rows.append(
            {
                "pair_id": p.pair_id,
                "cell": p.cell,
                "delta_floor": d_floor,
                "delta_ceiling": d_ceiling,
                "sep": sep,
                "passes_bar": bool(sep is not None and abs(sep) >= SEPARATION_BAR),
            }
        )
        c = per_cell.setdefault(p.cell, {"n_pairs": 0, "n_pass": 0, "n_unscored": 0})
        c["n_pairs"] += 1
        c["n_pass"] += int(sep is not None and abs(sep) >= SEPARATION_BAR)
        c["n_unscored"] += int(sep is None)
    for cell, c in sorted(per_cell.items()):
        c["cell_pass"] = c["n_pass"] >= SEP_MIN_PAIRS_OF_6
        # Binding-table self-count: print the realized per-cell sampled-pair counts.
        logger.info(
            "[gate3] %-28s pairs=%d pass=%d unscored=%d -> %s",
            cell,
            c["n_pairs"],
            c["n_pass"],
            c["n_unscored"],
            "SEPARABLE" if c["cell_pass"] else "NOT-SEPARABLE",
        )
    n_cells = len(per_cell)
    frac = sum(1 for c in per_cell.values() if c["cell_pass"]) / n_cells if n_cells else 0.0
    catastrophic = frac < SEP_CATASTROPHIC_FRAC
    return {
        "criterion": (
            "anchor-separation early read (plan §7 gate 3, divergence 12): ADVISORY "
            "aggregate + catastrophic <25% instrument-broken HALT floor; per-pair "
            "|sep| >= 0.5 exclusions stay binding at analysis regardless"
        ),
        "bars": {
            "sep_bar": SEPARATION_BAR,
            "min_pairs_of_6": SEP_MIN_PAIRS_OF_6,
            "advisory_cell_frac": SEP_CELL_FRAC_MIN,
            "catastrophic_frac": SEP_CATASTROPHIC_FRAC,
        },
        "n_cells": n_cells,
        "frac_cells_pass": frac,
        "advisory_aggregate": {
            "bar": SEP_CELL_FRAC_MIN,
            "pass": frac >= SEP_CELL_FRAC_MIN,
            "note": (
                "REPORTED, never halted on (divergence 12): per-type separation failure "
                "on the new model is part of the transfer answer; the parent passed at "
                "25/38 = 0.6579, so the 60% boundary is live cross-model"
            ),
        },
        "catastrophic": catastrophic,
        # BINDING read for downstream _require_gates: instrument NOT broken.
        "passed": not catastrophic,
        "per_cell": per_cell,
        "pairs": pair_rows,
        "repro": J94._repro(),
    }


def phase_separation_gate(cfg: JudgeConfig) -> int:
    pairs, gate_pairs, gate_ctx_rows, registry = _gate_slice_inputs(cfg)
    gate_units = build_anchor_behavior_items(gate_ctx_rows, pairs, restrict_pairs=gate_pairs)
    n_calls = sum(len(us) for us in gate_units.values())
    logger.info(
        "[gate3] %d pairs, %d contexts, %d rubrics, %d sync judge calls",
        len(gate_pairs),
        len({c for p in gate_pairs for c in (p.a, p.b)}),
        len(gate_units),
        n_calls,
    )
    if not cfg.dry_run:
        # S2: the remaining ~8.7k sync calls dispatch ONLY after gate 3-pre
        # PASSes (the rule-26 pilot on this same slice's first rollouts).
        _require_gates(cfg, names=("pilot_gate3pre_report.json",))
    scores = _sep_scores_for_slice(cfg, registry, gate_units, cfg.dry_run)
    if cfg.dry_run:
        logger.info("[gate3] dry-run complete (no API calls)")
        return RC_OK
    report = separation_verdict(gate_pairs, gate_ctx_rows, scores)
    J94._write_json_atomic(cfg.gates_dir / "anchor_separation_report.json", report)
    logger.info(
        "[gate3] %.0f%% of %d cells separable -> advisory %s%s",
        100 * report["frac_cells_pass"],
        report["n_cells"],
        "PASS" if report["advisory_aggregate"]["pass"] else "MISS (reported, not halting)",
        " | CATASTROPHIC (<25% — instrument-broken abort per §7)" if report["catastrophic"] else "",
    )
    return RC_SEPARATION_GATE if report["catastrophic"] else RC_OK


# ── gate 6 pilot (plan §7, rule 26 — P6 bulk waves; parent-verbatim) ──


def phase_pilot(cfg: JudgeConfig) -> int:
    """Rule-26 pilot per rubric FAMILY + the live forced-batch shape probe.

    Plan §7 gate 6: covers ONLY the P6 bulk Batch waves — the gate 3-pre PASS
    certifies the instrument on anchor text; steered grid + stage-2 text is a
    different content class, so the bulk waves get their own pilot. REFUSES
    ``--dry-run``: the pilot gate EXISTS to measure the real instrument's
    truncation/parse-fail profile and to live-probe the forced-batch request
    shape."""
    if cfg.dry_run:
        logger.error(
            "[pilot] --dry-run refused: the rule-26 pilot's whole purpose is measuring the "
            "REAL instrument's truncation/parse-fail profile (plus the live forced-batch "
            "request-shape probe) — there is no meaningful zero-API pilot. Run without "
            "--dry-run to spend the ~%d pilot draws, or use --phase separation-gate / "
            "--phase anchors with --dry-run for a free construction check.",
            PILOT_TARGET_COHERENCE + 2 * PILOT_TARGET_BEHAVIOR,
        )
        return RC_DRY_RUN_UNSUPPORTED
    pairs = surviving_pairs(cfg.bank_json)
    pairs_by_id = {p.pair_id: p for p in pairs}
    registry = rubric_registry(pairs)
    grid_rows = load_grid_rows(cfg.rollouts_dir)
    anchor_rows = load_anchor_rows(cfg.anchors_file)
    coh = build_coherence_items(grid_rows, anchor_rows)
    beh = build_grid_behavior_items(grid_rows, pairs_by_id)
    for rid, us in build_anchor_behavior_items(anchor_rows, pairs).items():
        beh.setdefault(rid, []).extend(us)

    # 4th element = allow_subresolution_pilot, scoped PER FAMILY (#2124 guard).
    # coherence + value-rubric are SATISFIABLE at 204 (measured 2026-08-17:
    # verdict=PASS arms=4 draws=204 failures=0 for both), so they keep the
    # config-time refusal ARMED — if their arm sizes ever regress below the
    # rule-26(b) floor of 51 the guard must fail loud, not silently degrade to a
    # sub-resolution read. This is deliberately STRICTER than the gate3pre call
    # site, which passes the flag unconditionally.
    #
    # query-rubric is ITEM-LIMITED and no target_total_draws can fix it: its four
    # realized arms hold 30 / 20 / 20 / 20 items against the 51 needed, and the
    # production instrument is pinned at n_draws=1 (rules 23/26 — the pilot runs
    # the EXACT production instrument, one draw per item). So it is checked at
    # REALIZED CAPACITY. Two things this does NOT do: it does not loosen
    # parse_fail_threshold (still 0.02, applied to whatever rate is observable),
    # and it does not touch rule 26(a)'s truncation check, which _gate_verdict
    # applies unconditionally. The honest residual is RESOLUTION: the parse-fail
    # check for this family can only resolve 1/20 = 5.0% on the three 20-item
    # arms and 1/30 = 3.33% on the anchor arm, NOT 2% — recorded per family below
    # as parse_fail_resolution_pct and carried into the report as a caveat.
    #
    # NOTE: _family_min_effective_floor is deliberately NOT used here. It asserts
    # a SINGLE-ARM precondition (a scalar floor compared against every arm would
    # under-enforce the larger arms of a multi-arm family), and this family has
    # four arms. It is not needed either: the default verdict floor
    # min_effective_draws_per_arm=10 is cleared by every arm (smallest 20).
    fam_reps = {
        "coherence": (J94.COHERENCE_RUBRIC_ID, coh, PILOT_TARGET_COHERENCE, False),
        "value-rubric": (
            _family_representative(beh, registry, query_family=False),
            None,
            PILOT_TARGET_BEHAVIOR,
            False,
        ),
        "query-rubric": (
            _family_representative(beh, registry, query_family=True),
            None,
            PILOT_TARGET_BEHAVIOR,
            True,
        ),
    }
    per_family: dict[str, dict] = {}
    all_pass = True
    for family, (rid, units, target, allow_sub) in fam_reps.items():
        units = units if units is not None else beh[rid]
        J94._validate_units(units)
        arms: dict[str, list[tuple[str, str, str]]] = {}
        for u in units:
            arms.setdefault(_pilot_arm(u), []).append((u.item_id, u.question, u.answer))
        report = judge_pilot_gate(
            arms,
            registry[rid],
            max_tokens=cfg.max_tokens,
            cache_dir=cfg.pilot_cache_root / rid,
            save_raw_dir=cfg.raw_dir / "pilot" / rid,
            n_draws=JUDGE_N_DRAWS,
            target_total_draws=target,
            allow_subresolution_pilot=allow_sub,
            judge_model=cfg.judge_model,
            report_path=cfg.gates_dir / "pilot" / f"{family}.json",
            seed=PILOT_SEED,
        )
        # What the parse-fail check can actually RESOLVE for this family: the
        # smallest arm's REALIZED draw count sets the smallest observable nonzero
        # rate (1/n). Realized draws are bounded by the BUDGET SPLIT and by the
        # arm's ITEM COUNT, whichever binds — judge_pilot_gate computes
        # per_arm_items = target // (n_arms * n_draws) and then caps each arm at
        # its own item count. Item capacity ALONE overstates resolution wildly
        # for a budget-limited family: coherence holds 13,500 items but realizes
        # 51 draws/arm at target 204, so its true resolution is 1.96%, not the
        # 0.0074% a capacity-only read reports. Both are recorded — capacity
        # answers "would a bigger budget help?" (for query-rubric: no).
        per_arm_cap = target // (len(arms) * JUDGE_N_DRAWS)
        min_arm_capacity = min(len(items) for items in arms.values()) * JUDGE_N_DRAWS
        min_arm_realized = (
            min(min(per_arm_cap, len(items)) for items in arms.values()) * JUDGE_N_DRAWS
        )
        per_family[family] = {
            "rubric_id": rid,
            "verdict": report.verdict,
            "failures": report.failures,
            "warnings": report.warnings,
            "n_total_draws": report.n_total_draws,
            "subresolution_allowed": allow_sub,
            "min_arm_capacity_draws": min_arm_capacity,
            "min_arm_realized_draws": min_arm_realized,
            "parse_fail_resolution_pct": round(100.0 / min_arm_realized, 4)
            if min_arm_realized > 0
            else None,
        }
        all_pass &= report.passed
        logger.info(
            "[pilot] %s (%s): %s (%d draws)", family, rid, report.verdict, report.n_total_draws
        )

    # Live forced-batch request-shape probe (threshold_base=0; gotchas: a
    # mock/sync-only pilot cannot validate the batches.create envelope).
    probe_units = coh[: J94.FORCED_BATCH_PROBE_N]
    probe = judge_graded(
        [(u.item_id, u.question, u.answer) for u in probe_units],
        registry[J94.COHERENCE_RUBRIC_ID],
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
        "per_family": per_family,
        "forced_batch_probe": {
            "n_items": len(probe_units),
            "n_scored": n_probe_scored,
            "passed": probe_ok,
            **J94._telemetry(probe),
        },
        "instrument": {
            "judge_model": cfg.judge_model,
            "max_tokens": cfg.max_tokens,
            "n_draws": JUDGE_N_DRAWS,
            "n_rubrics_total": len(registry),
        },
        "repro": J94._repro(),
    }
    J94._write_json_atomic(cfg.gates_dir / "pilot_gate_report.json", aggregate)
    logger.info("[pilot] aggregate verdict: %s", "PASS" if all_pass else "FAIL")
    return RC_OK if all_pass else RC_PILOT_GATE


# ── phases ────────────────────────────────────────────────────────────


_ALL_GATE_REPORTS = (
    "pilot_gate_report.json",
    "coherence_baseline_gate.json",
    "anchor_separation_report.json",
)


def _require_gates(cfg: JudgeConfig, names: tuple[str, ...] = _ALL_GATE_REPORTS) -> None:
    """Behavior-wave spend requires the named gate reports present AND PASS.

    ``anchor_separation_report.json``'s ``passed`` is the BINDING read (NOT
    catastrophic — divergence 12); its advisory 60% aggregate never gates."""
    for name in names:
        path = cfg.gates_dir / name
        if not path.is_file():
            raise RuntimeError(f"gate report missing: {path} — run the gate phase first")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not rec.get("passed"):
            raise RuntimeError(f"gate FAILED per {path} — fix the instrument/bank and re-run")


def phase_anchors(cfg: JudgeConfig) -> int:
    """Coherence-baseline gate over anchors, then the anchor behavior waves.

    Entry gate: the anchor behavior waves are an order-10^4-call production
    spend, so the pilot (gate 6) + separation (gate 3 binding read) reports
    must be present-and-passed BEFORE launch. Gate 5 (coherence baseline) is
    exempt — this phase PRODUCES it. ``--dry-run``: construction check over
    every wave this phase would dispatch, zero API calls, nothing persisted."""
    pairs = surviving_pairs(cfg.bank_json)
    anchor_rows = load_anchor_rows(cfg.anchors_file)
    if cfg.dry_run:
        beh = build_anchor_behavior_items(anchor_rows, pairs)
        return _dry_run_units_report(
            "anchors",
            {
                "coherence.anchors": build_coherence_items(None, anchor_rows),
                **{f"{rid}.anchors": us for rid, us in beh.items()},
            },
        )
    _require_gates(cfg, names=("pilot_gate_report.json", "anchor_separation_report.json"))
    audits = J94.run_audits("anchors", anchor_rows, cfg.audits_dir)
    registry = rubric_registry(pairs)

    coh_units = build_coherence_items(None, anchor_rows)
    J94.run_wave(
        "coherence.anchors",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg,
        threshold_base=cfg.wave_threshold_base,
    )
    scores = list(J94._iter_jsonl(cfg.scores_dir / "coherence.anchors.scores.jsonl"))
    gate = J94.coherence_baseline_gate(scores)
    gate["audits"] = audits
    J94._write_json_atomic(cfg.gates_dir / "coherence_baseline_gate.json", gate)
    logger.info(
        "[gate5] coherence baseline: median=%.1f frac>60=%.3f -> %s",
        gate["median"],
        gate["frac_gt60"],
        "PASS" if gate["passed"] else "FAIL",
    )
    if not gate["passed"]:
        return RC_COHERENCE_GATE
    for rid, units in sorted(build_anchor_behavior_items(anchor_rows, pairs).items()):
        J94.run_wave(
            f"{rid}.anchors",
            rid,
            registry[rid],
            units,
            cfg,
            threshold_base=cfg.wave_threshold_base,
        )
    J94._refresh_summary(cfg)
    return RC_OK


def phase_waves(cfg: JudgeConfig) -> int:
    """Production grid waves (coherence + dual-rubric behavior), gate-guarded.
    ``--dry-run``: construction check at entry, zero API calls, nothing
    persisted."""
    pairs = surviving_pairs(cfg.bank_json)
    pairs_by_id = {p.pair_id: p for p in pairs}
    grid_rows = load_grid_rows(cfg.rollouts_dir)
    if cfg.dry_run:
        beh = build_grid_behavior_items(grid_rows, pairs_by_id)
        return _dry_run_units_report(
            "waves",
            {
                "coherence.grid": build_coherence_items(grid_rows, None),
                **{f"{rid}.grid": us for rid, us in beh.items()},
            },
        )
    _require_gates(cfg)
    registry = rubric_registry(pairs)
    J94.run_audits("grid", grid_rows, cfg.audits_dir)
    coh_units = build_coherence_items(grid_rows, None)
    J94.run_wave(
        "coherence.grid",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg,
        threshold_base=cfg.wave_threshold_base,
    )
    for rid, units in sorted(build_grid_behavior_items(grid_rows, pairs_by_id).items()):
        J94.run_wave(
            f"{rid}.grid", rid, registry[rid], units, cfg, threshold_base=cfg.wave_threshold_base
        )
    J94._refresh_summary(cfg)
    return RC_OK


def phase_stage2(cfg: JudgeConfig) -> int:
    """Stage-2 waves, gate-guarded. ``--dry-run``: construction check
    at entry, zero API calls, nothing persisted."""
    if cfg.stage2_dir is None:
        raise RuntimeError("--phase stage2 requires --stage2-dir")
    pairs = surviving_pairs(cfg.bank_json)
    pairs_by_id = {p.pair_id: p for p in pairs}
    rows = load_stage2_rows(cfg.stage2_dir)
    if cfg.dry_run:
        beh = build_grid_behavior_items(rows, pairs_by_id, tag="s", kind="stage2")
        return _dry_run_units_report(
            "stage2",
            {
                "coherence.stage2": build_coherence_items(None, None, rows),
                **{f"{rid}.stage2": us for rid, us in beh.items()},
            },
        )
    _require_gates(cfg)
    registry = rubric_registry(pairs)
    J94.run_audits("stage2", rows, cfg.audits_dir)
    coh_units = build_coherence_items(None, None, rows)
    J94.run_wave(
        "coherence.stage2",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg,
        threshold_base=cfg.wave_threshold_base,
    )
    for rid, units in sorted(
        build_grid_behavior_items(rows, pairs_by_id, tag="s", kind="stage2").items()
    ):
        J94.run_wave(
            f"{rid}.stage2", rid, registry[rid], units, cfg, threshold_base=cfg.wave_threshold_base
        )
    J94._refresh_summary(cfg)
    return RC_OK


def phase_audits(cfg: JudgeConfig) -> int:
    """Mechanical text audits (zero-API). ``--dry-run``: report which
    inputs are present, persist nothing."""
    if cfg.dry_run:
        present = [
            str(d)
            for d in (cfg.rollouts_dir, cfg.anchors_file, cfg.stage2_dir)
            if d is not None and Path(d).is_dir()
        ]
        logger.info(
            "[audits] dry-run: inputs present: %s — nothing persisted (zero-API phase)",
            present or "none",
        )
        return RC_OK
    summaries = []
    if cfg.rollouts_dir.is_dir():
        summaries.append(J94.run_audits("grid", load_grid_rows(cfg.rollouts_dir), cfg.audits_dir))
    if cfg.anchors_file.is_dir():
        summaries.append(
            J94.run_audits("anchors", load_anchor_rows(cfg.anchors_file), cfg.audits_dir)
        )
    if cfg.stage2_dir is not None and cfg.stage2_dir.is_dir():
        summaries.append(J94.run_audits("stage2", load_stage2_rows(cfg.stage2_dir), cfg.audits_dir))
    if not summaries:
        raise RuntimeError("audits: no inputs found")
    J94._write_json_atomic(
        cfg.audits_dir / "audits_summary.json", {"summaries": summaries, "repro": J94._repro()}
    )
    return RC_OK


# ── TF-margin pools builder (plan §4.4 / llm-judging rule 19) ─────────

# Plan §11: "Margin pool 4+4 per type, filter threshold >50" — fixed
# judge-filtered pools, persona-vectors keep-threshold (score > 50).
POOL_PER_SIDE = 4
POOL_FILTER_MIN = 50.0


def pool_key(pair: BANK.Pair2162) -> str:
    """MUST equal ``issue2329_run.pool_key`` byte-for-byte — the margin
    consumer's join key."""
    return f"{pair.cell}|{pair.value_a}-{pair.value_b}"


def _anchor_behavior_scores(cfg: JudgeConfig) -> dict[tuple[str, int, str], float]:
    """(context_id, draw, rubric_id) -> kept mean score, from the persisted
    anchor behavior wave score rows (coherence rows carry no ``rubric`` key
    and are skipped; rule-9 dropped items carry ``score: null`` and are
    skipped — never coerced)."""
    files = sorted(cfg.scores_dir.glob("*.anchors.scores.jsonl"))
    assert files, (
        f"no anchor score files under {cfg.scores_dir} — run --phase anchors first "
        "(the pools are a pure re-reduction of the judged anchor waves)"
    )
    scores: dict[tuple[str, int, str], float] = {}
    for f in files:
        for row in J94._iter_jsonl(f):
            rid = row.get("rubric")
            if rid is None or row.get("score") is None:
                continue
            scores[(row["context_id"], int(row["draw"]), rid)] = float(row["score"])
    assert scores, "anchor score files present but zero scored behavior rows"
    return scores


def build_margin_pools(
    pairs: list[BANK.Pair2162],
    anchor_rows: list[dict],
    scores: dict[tuple[str, int, str], float],
) -> tuple[dict[str, list[dict]], dict]:
    """Fixed judge-filtered pools per value-pair key (plan §4.4): side A from
    the pairs' FLOOR contexts (A-descriptor score > 50), side B from the
    CEILING contexts (B-descriptor score > 50), top ``POOL_PER_SIDE`` per side
    by descending score (ties broken by (context_id, draw) — deterministic).

    A key whose EITHER side yields zero kept items is OMITTED (the margin
    consumer records explicit skip rows for it); a 1..3-item side is kept and
    flagged ``short`` in the report — below-floor yield is REPORTED, never
    silently backfilled.

    ``query_content`` builds NO pool, deliberately (same treatment as
    ``filler_swap``): its manipulated variable IS the user query, so rubric
    cores vary per carrier (``rubric_pair_2162`` keys them on ``pair.carrier``)
    AND each pair's two contexts pose DIFFERENT queries — a fixed shared
    answer pool scored under every context (plan §4.4 TF margin) is not
    well-defined at any key granularity. The skipped keys are reported under
    ``query_content_skip`` (never silently vanish); the margin consumer
    records explicit skip rows for the absent keys."""
    text_by = {(r["context_id"], int(r["draw"])): r["text"] for r in anchor_rows}
    ctxs_by_key: dict[str, dict] = {}
    qc_skipped_keys: set[str] = set()
    for p in pairs:
        if BANK.base_type_of(p.cell) == "query_content":
            qc_skipped_keys.add(pool_key(p))
            continue  # query_content: no well-defined fixed pool (docstring)
        cores = pair_rubric_cores(p)
        if cores is None:
            continue  # filler_swap: no rubric, no pool (explicit skip downstream)
        rid_a, rid_b = (rubric_core_id(c) for c in cores)
        rec = ctxs_by_key.setdefault(
            pool_key(p), {"rid_a": rid_a, "rid_b": rid_b, "a": set(), "b": set()}
        )
        assert rec["rid_a"] == rid_a and rec["rid_b"] == rid_b, (
            pool_key(p),
            "pairs sharing a pool key disagree on rubric cores",
        )
        rec["a"].add(p.a)
        rec["b"].add(p.b)

    pools: dict[str, list[dict]] = {}
    report_keys: dict[str, dict] = {}
    for key, rec in sorted(ctxs_by_key.items()):
        sides: dict[str, list[dict]] = {}
        for side, ctxs, rid in (("A", rec["a"], rec["rid_a"]), ("B", rec["b"], rec["rid_b"])):
            cands = []
            for (ctx, draw), text in text_by.items():
                if ctx not in ctxs:
                    continue
                score = scores.get((ctx, draw, rid))
                if score is None or score <= POOL_FILTER_MIN:
                    continue
                cands.append(
                    {"side": side, "text": text, "context_id": ctx, "draw": draw, "score": score}
                )
            cands.sort(key=lambda c: (-c["score"], c["context_id"], c["draw"]))
            sides[side] = cands[:POOL_PER_SIDE]
        n_a, n_b = len(sides["A"]), len(sides["B"])
        report_keys[key] = {
            "n_kept_a": n_a,
            "n_kept_b": n_b,
            "short": 0 < min(n_a, n_b) and (n_a < POOL_PER_SIDE or n_b < POOL_PER_SIDE),
            "omitted": min(n_a, n_b) == 0,
        }
        if min(n_a, n_b) == 0:
            continue
        pools[key] = sides["A"] + sides["B"]
    report = {
        "criterion": "TF-margin fixed pools (plan §4.4; 4+4 per key, score > 50 kept)",
        "pool_per_side": POOL_PER_SIDE,
        "filter_min": POOL_FILTER_MIN,
        "n_keys_total": len(ctxs_by_key),
        "n_keys_built": len(pools),
        "n_keys_omitted": sum(1 for r in report_keys.values() if r["omitted"]),
        "n_keys_short": sum(1 for r in report_keys.values() if r["short"]),
        "query_content_skip": {
            "n_keys": len(qc_skipped_keys),
            "keys": sorted(qc_skipped_keys),
            "reason": (
                "query_content manipulates the user query itself: rubric cores are "
                "per-carrier and each pair's two contexts pose different queries, so "
                "no fixed shared answer pool is well-defined (plan §4.4 TF margin); "
                "the margin consumer records explicit skip rows for these keys "
                "(same treatment as filler_swap)"
            ),
        },
        "per_key": report_keys,
        "repro": J94._repro(),
    }
    return pools, report


def phase_pools(cfg: JudgeConfig) -> int:
    """Build + persist the TF-margin pools file (zero API calls — a pure
    re-reduction of the judged anchor waves). The orchestrator stages the
    written ``pools.json`` to the pod for ``issue2329_run.py --phase margin``."""
    pairs = surviving_pairs(cfg.bank_json)
    anchor_rows = load_anchor_rows(cfg.anchors_file)
    scores = _anchor_behavior_scores(cfg)
    pools, report = build_margin_pools(pairs, anchor_rows, scores)
    assert pools, "zero pools built — the anchor waves' judge-filter kept nothing at > 50"
    if cfg.dry_run:
        logger.info(
            "[pools] dry-run: would build %d/%d keys (%d omitted, %d short, %d "
            "query_content skipped) — nothing persisted (zero-API phase)",
            report["n_keys_built"],
            report["n_keys_total"],
            report["n_keys_omitted"],
            report["n_keys_short"],
            report["query_content_skip"]["n_keys"],
        )
        return RC_OK
    out = cfg.work_root / "pools.json"
    J94._write_json_atomic(out, {"pools": pools, "meta": report})
    J94._write_json_atomic(cfg.gates_dir / "pools_report.json", report)
    logger.info(
        "[pools] %d/%d keys built (%d omitted, %d short; %d query_content keys "
        "skipped — no well-defined fixed pool) -> %s",
        report["n_keys_built"],
        report["n_keys_total"],
        report["n_keys_omitted"],
        report["n_keys_short"],
        report["query_content_skip"]["n_keys"],
        out,
    )
    return RC_OK


def phase_upload_raw(cfg: JudgeConfig) -> int:
    """One folder commit of the judge work root -> the 2329 judge_raw prefix.
    ``--dry-run``: a Hub upload is a mutating API call — report the
    would-be upload and stop."""
    if cfg.dry_run:
        logger.info(
            "[upload-raw] dry-run: would upload %s -> %s (no Hub calls made)",
            cfg.work_root,
            f"{HF_PREFIX}/raw_completions/judge_raw",
        )
        return RC_OK
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


# ── CLI ───────────────────────────────────────────────────────────────

PHASES = {
    "pilot-gate3pre": phase_pilot_gate3pre,
    "pilot": phase_pilot,
    "separation-gate": phase_separation_gate,
    "anchors": phase_anchors,
    "waves": phase_waves,
    "stage2": phase_stage2,
    "pools": phase_pools,
    "audits": phase_audits,
    "upload-raw": phase_upload_raw,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2329 VM-side judge pipeline.")
    ap.add_argument("--phase", required=True, choices=tuple(PHASES))
    ap.add_argument(
        "--in-root",
        type=Path,
        default=_DEFAULT_IN_ROOT,
        help=f"staging root; rollouts/anchors default under <in-root>/{HF_PREFIX}/raw_completions/",
    )
    ap.add_argument("--rollouts-dir", type=Path, default=None)
    ap.add_argument("--anchors-dir", type=Path, default=None)
    ap.add_argument("--stage2-dir", type=Path, default=None)
    ap.add_argument(
        "--bank-json",
        type=Path,
        default=None,
        help=f"frozen gate-0a bank.json (default <in-root>/{_BANK_JSON_REL}); the "
        "pair-drop registry every pairs-consuming phase derives from (divergence 9)",
    )
    ap.add_argument(
        "--breach-basis",
        type=Path,
        default=_DEFAULT_BREACH_BASIS,
        help="frozen PRE-regen cap-hit basis driving the anchors capregen campaign "
        "(default: the committed copy); scopes the gate-3 breach-cell row-cap "
        "staleness check (_assert_gate_rows_capregen_fresh) — absent + uniform "
        "row caps skips with a WARNING (pre-capregen fresh-run ordering only)",
    )
    ap.add_argument("--stage-from-hf", action="store_true")
    ap.add_argument(
        "--force-sync-routing",
        action="store_true",
        help="route production waves over the SYNC transport instead of Batch "
        "(threshold_base=FORCE_SYNC_THRESHOLD_BASE). For the rule-28 api-refusal "
        "RE-ISSUE pass only: the Batch classifier censored draws whose in-band retry "
        "cannot help. Instrument is UNCHANGED (model/rubric/max_tokens/n_draws) — only "
        "the transport differs. Scored draws are cache HITS and are not re-spent; "
        "api-refusal dicts are cache MISSES by design, so only the censored set is "
        "re-issued. NOTE: run_wave's regime key does NOT include routing, so a "
        "completed wave still SKIPs — quarantine its .meta.json to force the re-run.",
    )
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument("--work-root", type=Path, default=Path("eval_results/issue_2329/judge"))
    ap.add_argument("--cache-root", type=Path, default=Path("data/issue_2329/judge_cache"))
    ap.add_argument("--judge-model", type=str, default=J94.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=J94.DEFAULT_JUDGE_MAX_TOKENS)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="uniform construction check: build + validate every judge unit the phase "
        "would dispatch, print counts/routing, ZERO API calls, nothing persisted; "
        "--phase pilot / pilot-gate3pre REFUSE it (rc 10) — their purpose is live "
        "measurement",
    )
    return ap.parse_args(argv)


_STAGE_GRID = f"{HF_PREFIX}/raw_completions/grid"
_STAGE_ANCHORS = f"{HF_PREFIX}/raw_completions/anchors"
_STAGE_ANCHORS_GATE = f"{HF_PREFIX}/raw_completions/anchors_gate"
_STAGE_STAGE2 = f"{HF_PREFIX}/raw_completions/stage2"

# Phase-aware staging: stage only what the requested phase's loaders actually
# read. "required" prefixes FAIL LOUD on absence; an "anchors_any" phase needs
# anchor rows from EITHER prefix (`anchors_gate` is uploaded early at P2 so
# gates 3-pre/3 can run BEFORE the terminal `anchors` upload; grid exists only
# from P3) — each member is tolerated individually, but ZERO landing raises;
# "optional" prefixes log-and-continue (phase_audits is is_dir-gated + fails
# loud on no inputs itself). Every pairs-consuming phase additionally stages
# the frozen bank.json (single file).
_PHASE_STAGE_PLAN: dict[str, dict[str, tuple[str, ...]]] = {
    "pilot-gate3pre": {"anchors_any": (_STAGE_ANCHORS, _STAGE_ANCHORS_GATE)},
    "separation-gate": {"anchors_any": (_STAGE_ANCHORS, _STAGE_ANCHORS_GATE)},
    "pilot": {"required": (_STAGE_GRID,), "anchors_any": (_STAGE_ANCHORS, _STAGE_ANCHORS_GATE)},
    "anchors": {"required": (_STAGE_ANCHORS,)},
    "waves": {"required": (_STAGE_GRID,)},
    "stage2": {"required": (_STAGE_STAGE2,)},
    "pools": {"required": (_STAGE_ANCHORS,)},
    "audits": {"optional": (_STAGE_GRID, _STAGE_ANCHORS, _STAGE_STAGE2)},
    "upload-raw": {},
}

# Phases that derive their pair set from the frozen bank.json (divergence 9).
_BANK_JSON_PHASES = frozenset(
    {"pilot-gate3pre", "pilot", "separation-gate", "anchors", "waves", "stage2", "pools"}
)


def _stage_inputs(args: argparse.Namespace) -> None:
    """Stage the requested phase's Hub inputs per ``_PHASE_STAGE_PLAN``."""
    from explore_persona_space.orchestrate import hub

    plan = _PHASE_STAGE_PLAN[args.phase]

    def _stage(prefix: str, *, tolerate_missing: bool) -> bool:
        try:
            staged = hub.stage_hub_prefix(
                DATASET_REPO, prefix, args.in_root, revision=args.hf_revision
            )
        except FileNotFoundError:
            if not tolerate_missing:
                raise
            logger.info(
                "[stage] %s: not on the Hub yet — tolerated (--phase %s does not "
                "strictly require it)",
                prefix,
                args.phase,
            )
            return False
        logger.info("[stage] %s: %d files", prefix, len(staged))
        return len(staged) > 0

    for prefix in plan.get("required", ()):
        _stage(prefix, tolerate_missing=False)
    anchors_any = plan.get("anchors_any", ())
    if anchors_any and not any(_stage(p, tolerate_missing=True) for p in anchors_any):
        raise FileNotFoundError(
            f"--phase {args.phase} needs anchor rows but none of {list(anchors_any)} "
            f"exist on {DATASET_REPO} — the pod-side anchor uploads (P2 gate slice / "
            "terminal) have not landed yet"
        )
    for prefix in plan.get("optional", ()):
        _stage(prefix, tolerate_missing=True)
    if args.phase in _BANK_JSON_PHASES and args.bank_json is None:
        # Only the DEFAULT in-root mirror location is staged; an explicit
        # --bank-json points at an already-local copy the caller owns.
        hub.stage_hub_file(
            DATASET_REPO,
            _BANK_JSON_REL,
            args.in_root / _BANK_JSON_REL,
            repo_type="dataset",
            revision=args.hf_revision,
        )
        logger.info("[stage] %s: staged", _BANK_JSON_REL)


def _resolve_anchors_dir(mirror: Path) -> Path:
    """Default anchors dir: the full ``anchors`` mirror when it holds shards
    AND its shard name set COVERS the early gate mirror's, else the
    early-uploaded ``anchors_gate`` mirror (P2 uploads the gate slice there
    FIRST so gates 3-pre/3 can run before the terminal anchors upload lands).

    The coverage condition is load-bearing (run.py v11 MAJOR 1): capregen
    uploads land per-worker, so on a not-yet-fully-populated full prefix a
    bare non-empty check would prefer a strict SUBSET of the gate shards and
    wedge gate-3 staging with a misleading "shards incomplete" error while
    the COMPLETE gate mirror sits ignored. A full prefix that covers every
    gate-mirror shard name is safe: gate-3 only needs the gate contexts.
    Falls back to the canonical path when neither holds shards — the loaders
    fail loud on absence."""
    full, gate = mirror / "anchors", mirror / "anchors_gate"
    full_names = {p.name for p in full.glob("anchors_*.jsonl")}
    gate_names = {p.name for p in gate.glob("anchors_*.jsonl")}
    if full_names and gate_names <= full_names:
        return full
    if gate_names:
        if full_names:
            logger.warning(
                "[stage] full anchors prefix is PARTIAL (%d shard(s) in the gate mirror "
                "missing from it) — staging from the complete gate mirror %s",
                len(gate_names - full_names),
                gate,
            )
        else:
            logger.info("[stage] anchors dir -> %s (full anchors prefix not staged yet)", gate)
        return gate
    return full


def build_config(args: argparse.Namespace) -> JudgeConfig:
    mirror = args.in_root / HF_PREFIX / "raw_completions"
    rollouts = args.rollouts_dir if args.rollouts_dir is not None else mirror / "grid"
    # NOTE: JudgeConfig.anchors_file carries the anchors *directory* here (the
    # 2329 anchors are per-worker shards); only OUR loaders read it.
    anchors = args.anchors_dir if args.anchors_dir is not None else _resolve_anchors_dir(mirror)
    stage2 = args.stage2_dir
    if stage2 is None and args.phase == "stage2":
        stage2 = mirror / "stage2"
    bank_json = args.bank_json if args.bank_json is not None else args.in_root / _BANK_JSON_REL
    return JudgeConfig(
        work_root=args.work_root,
        cache_root=args.cache_root,
        rollouts_dir=rollouts,
        anchors_file=anchors,
        stage2_dir=stage2,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
        bank_json=bank_json,
        breach_basis=args.breach_basis,
        force_sync_routing=args.force_sync_routing,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.stage_from_hf:
        _stage_inputs(args)
    cfg = build_config(args)
    for d in (cfg.scores_dir, cfg.items_dir, cfg.raw_dir, cfg.gates_dir, cfg.audits_dir):
        d.mkdir(parents=True, exist_ok=True)
    rc = PHASES[args.phase](cfg)
    logger.info("[phase=%s_done] rc=%d", args.phase, rc)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
