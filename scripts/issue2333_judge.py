#!/usr/bin/env python
"""Issue #2333 VM-side judge pipeline — snowball test (plan §4.5).

Thin adaptation of ``scripts/issue2162_judge.py`` over the #2333 row schema:

- Coherence + dual-rubric behavior items for every grid rollout
  (``response_text``: prefill arms = donor opening + continuation).
- Prefill CONTINUATION-ONLY behavior companions (tag ``n``, judged on
  ``continuation_text`` — plan §6 exploratory read).
- SAME-WAVE ce calibration items: banked #2162 ce grid rows (S1, tag ``k``)
  and banked #2094 fu1 conf1 joint_all rows (S2, tag ``m``) are re-judged in
  the SAME per-rubric waves as the fresh items, so every recovery ratio
  R_k = F̄_arm / F̄_ce shares one judge instrument (plan §5).
- q35 anchors: coherence-baseline gate then per-rubric anchor waves
  (q25 reuses the BANKED #2162/#2094 anchor scores — no fresh q25 anchors).
- Rule-26 pilot gate FIRST (per rubric family + live forced-batch probe);
  judge = ``claude-sonnet-4-5-20250929`` via the Batch API, max_tokens 1024.

Cell sets (plan §4.3 item 3, q35_language_snowball): ``--cell-set q35lang``
threads the 2-language-cell universe (S2 OFF) through every loader/expected
count, GATES the banked Qwen2.5 calibration legs (the #2162/#2094 calib text
is q25-generated — a cross-model re-judge is N/A for the q35lang wave), and
routes the anchors phase over the staged #2329 anchor completions (the
ANCHORS-REJUDGE wave: identical texts re-scored under THIS wave's instrument;
selfgen fallback = fresh ``anchors_w*.jsonl``). Judge work/cache roots land in
per-cell-set partitions (``.cell_set`` marker asserted) so #2329-era or
parent-leg cache entries structurally cannot serve the rejudge — a ~zero-
variance judge-offset table downstream is the contamination tell. The parent
path (``--cell-set main``, the default) is behaviorally byte-identical.

REUSES ``issue2094_judge`` (J94: JudgeUnit / run_wave / caches / pilot /
coherence gate) and ``issue2162_judge`` (J62: rubric ids, anchor unit ids,
dry-run report) — never re-implements them (plan §10 fitness map).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import issue2094_judge as J94  # noqa: E402
import issue2162_judge as J62  # noqa: E402

from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: E402
from explore_persona_space.experiments.issue2094 import bank as BANK94  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK2162  # noqa: E402
from explore_persona_space.experiments.issue2333 import constants as C  # noqa: E402

logger = logging.getLogger("issue2333.judge")

RC_OK = 0
RC_PILOT_GATE = J62.RC_PILOT_GATE  # 7
RC_COHERENCE_GATE = J62.RC_COHERENCE_GATE  # 8
RC_DRY_RUN_UNSUPPORTED = J62.RC_DRY_RUN_UNSUPPORTED  # 10

JUDGE_N_DRAWS = J94.JUDGE_N_DRAWS  # 1 draw per item (parent convention)
PILOT_SEED = 23330


# ── pairs / rubrics (both banks) ──────────────────────────────────────


def build_pair_universe(cell_set: str = "main") -> tuple[list, list]:
    """(s1_pairs, s2_pairs) — same filter as issue2333_run.build_pair_universe.

    q35lang: 72 language-cell pairs, S2 EMPTY (the 15-pair assert is gated on
    ``CellSet.s2_on`` — plan §4.3 item 3 parametrization)."""
    cs = C.CELL_SETS[cell_set]
    s1 = [p for p in BANK2162.build_pairs() if p.cell in cs.s1_cells]
    assert len(s1) == len(cs.s1_cells) * C.S1_PAIRS_PER_CELL, len(s1)
    if cs.s2_on:
        s2 = [p for p in BANK94.build_pairs() if p.setting == "matched_query"]
        assert len(s2) == 15, len(s2)
    else:
        s2 = []
    assert len(s1) + len(s2) == cs.expected_pairs, (len(s1), len(s2), cs.expected_pairs)
    return s1, s2


def pair_set_of(pair) -> str:
    return "s1" if hasattr(pair, "cell") else "s2"


def pair_rubric_cores_2333(pair) -> tuple[str, str]:
    """(core_A, core_B) per pair set. S1 = #2162 rubric pair (no filler_swap in
    the survivor cells, so never None); S2 = the #2094 matched-query 'prefix'
    rubric kind (SETTING_RUBRIC_KINDS['matched_query'] == ('prefix',))."""
    if pair_set_of(pair) == "s1":
        cores = BANK2162.rubric_pair_2162(pair)
    else:
        cores = BANK94.rubric_pair(pair, "prefix")
    assert cores is not None and len(cores) == 2, pair
    return cores


def rubric_registry(s1_pairs: list, s2_pairs: list) -> dict[str, str]:
    """rubric_id -> eval prompt (coherence + every distinct core, both banks)."""
    reg = {J94.COHERENCE_RUBRIC_ID: J94.coherence_eval_prompt()}
    for p in [*s1_pairs, *s2_pairs]:
        for core in pair_rubric_cores_2333(p):
            reg.setdefault(J62.rubric_core_id(core), J94.behavior_eval_prompt(core))
    return reg


# ── config ────────────────────────────────────────────────────────────


@dataclass
class JudgeConfig2333:
    """J94.JudgeConfig plus the #2333-specific input roots."""

    base: J94.JudgeConfig
    model_tag: str
    rollouts_dir: Path  # holds blocks/*.jsonl (+ ce_control/*.jsonl on q35)
    anchors_dir: Path | None
    calib_dir: Path
    cell_set: str = "main"  # C.CELL_SETS key (q35_language_snowball: "q35lang")
    # Audited override for the coherence-baseline gate: a non-empty reason
    # string proceeds past a FAIL and is written verbatim into the gate JSON.
    # Never a threshold change — the gate verdict stays recorded as FAIL.
    accept_coherence_gate_fail: str | None = None

    @property
    def dry_run(self) -> bool:
        return self.base.dry_run


# ── input walkers (2333 row schemas) ──────────────────────────────────


def assert_shard_set_complete(found_slugs: set[str], expected_slugs: set[str], what: str) -> None:
    """Pre-spend completeness gate (r1 Major 3): the judge NEVER dispatches a
    Batch-API wave against a partial/overfull shard set — a missing block
    would silently censor whole (cell, arm, variant) cells from the F tables.
    Raises naming the exact missing/extra slugs."""
    missing = sorted(expected_slugs - found_slugs)
    extra = sorted(found_slugs - expected_slugs)
    if missing or extra:
        raise RuntimeError(
            f"{what} shard set incomplete: {len(found_slugs)}/{len(expected_slugs)} expected; "
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"extra={extra[:8]}{'...' if len(extra) > 8 else ''} — "
            "re-run/finish the pod grid (or re-stage) before ANY judge spend"
        )


def assert_draw_consistency(rows: list[dict]) -> int:
    """Every (block_key, pair_id) group carries the SAME draw set {0..K-1}.
    Returns K. A ragged group means a partially-written/mixed-regime shard."""
    by_group: dict[tuple[str, str], set[int]] = {}
    for r in rows:
        by_group.setdefault((r["block_key"], r["pair_id"]), set()).add(int(r["draw"]))
    draw_sets = {frozenset(s) for s in by_group.values()}
    assert len(draw_sets) == 1, (
        f"ragged per-(block, pair) draw sets: {sorted(map(sorted, draw_sets))[:4]}"
    )
    draws = next(iter(draw_sets))
    k = len(draws)
    assert draws == frozenset(range(k)), sorted(draws)
    return k


def load_grid_rows(
    rollouts_dir: Path, expect_complete: bool = True, cell_set: str = "main"
) -> list[dict]:
    shards = sorted((rollouts_dir / "blocks").glob("*.jsonl"))
    assert shards, f"no grid block shards under {rollouts_dir}/blocks"
    if expect_complete:
        expected = C.expected_grid_slugs(cell_set)
        assert_shard_set_complete(
            {s.stem for s in shards}, expected, f"grid ({len(expected)}-block)"
        )
    rows = [r for s in shards for r in J94._iter_jsonl(s)]
    assert rows, "grid shards present but empty"
    for r in rows[:1]:
        for key in ("block_key", "pair_id", "draw", "response_text", "kind", "variant", "cell"):
            assert key in r, (key, sorted(r))
    k = assert_draw_consistency(rows)
    logger.info("[load] grid: %d shards, %d rows, K=%d", len(shards), len(rows), k)
    return rows


def load_ce_rows(
    rollouts_dir: Path, expect_complete: bool = True, cell_set: str = "main"
) -> list[dict]:
    shards = sorted((rollouts_dir / "ce_control").glob("*.jsonl"))
    assert shards, f"no ce_control shards under {rollouts_dir}/ce_control"
    if expect_complete:
        expected = C.expected_ce_control_slugs(cell_set)
        assert_shard_set_complete(
            {s.stem for s in shards}, expected, f"ce_control ({len(expected)}-block)"
        )
    rows = [r for s in shards for r in J94._iter_jsonl(s)]
    assert rows, "ce_control shards present but empty"
    assert_draw_consistency(rows)
    return rows


def load_anchor_rows(anchors_dir: Path) -> list[dict]:
    files = sorted(anchors_dir.glob("anchors_w*.jsonl"))
    assert files, f"no anchor shards under {anchors_dir}"
    rows = [r for f in files for r in J94._iter_jsonl(f)]
    assert rows, "anchor shards present but empty"
    for r in rows[:1]:
        for key in ("context_id", "draw", "response_text"):
            assert key in r, (key, sorted(r))
    return rows


def _stage_2329_anchor_shards(dest: Path) -> list[Path]:
    """Stage the #2329 final anchor jsonl shards VERBATIM @ C.PIN_2329_DATA
    (VM-side twin of ``issue2333_run._list_2329_files``/``_stage_2329_file`` —
    torch-free so the judge never imports the pod driver). The reused shards
    are deliberately NOT mirrored under the q35lang HF namespace (already on
    HF at the pin; run.py upload globs exclude ``reused_2329/``), so the
    ANCHORS-REJUDGE wave stages them from the pin on demand."""
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient at this call site
            api.list_repo_tree(
                C.DATA_REPO,
                path_in_repo=C.R2329_ANCHORS_PREFIX,
                repo_type="dataset",
                revision=C.PIN_2329_DATA,
                recursive=True,
            )
        ),
        what=f"list #2329 {C.R2329_ANCHORS_PREFIX}",
    )
    rels = sorted(
        e.path
        for e in entries
        if getattr(e, "size", None) is not None
        and e.path.endswith(".jsonl")
        and C.R2329_NEVER_CONSUMED_SUBSTRING not in e.path
    )
    assert rels, f"no #2329 anchor jsonl under {C.R2329_ANCHORS_PREFIX} @ {C.PIN_2329_DATA}"
    dest.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for rel in rels:
        target = dest / Path(rel).name
        if not target.is_file():
            got = hub.retry_transient(
                lambda fn=rel: hf_hub_download(
                    repo_id=C.DATA_REPO,
                    repo_type="dataset",
                    filename=fn,
                    revision=C.PIN_2329_DATA,
                    local_dir=dest / "_dl",
                ),
                what=f"stage #2329 anchors {rel}",
            )
            Path(got).replace(target)
        staged.append(target)
    logger.info("[anchors-2329] staged %d shards @ %s", len(staged), C.PIN_2329_DATA[:12])
    return staged


def anchor_mode(anchors_dir: Path, cell_set: str) -> str:
    """``"fresh"`` (selfgen ``anchors_w*.jsonl`` present) vs ``"reused_2329"``.

    Fresh shards WIN when present — the fitness2329 selfgen decision produces
    them and the reuse manifest is then absent by construction; on the reuse
    path the pod stages ``reused_2329/`` and writes the manifest, and the VM
    judge re-stages the shards from the pin when the local dir is empty."""
    if cell_set != "q35lang":
        return "fresh"
    if sorted(anchors_dir.glob("anchors_w*.jsonl")):
        return "fresh"
    return "reused_2329"


def load_anchor_rows_2333(anchors_dir: Path, cell_set: str, ctx_ids: set[str]) -> list[dict]:
    """Cell-set anchor loader. main / selfgen: the parent ``anchors_w*.jsonl``
    shape (``response_text``). q35lang reuse: the staged #2329 shards
    (OBSERVED schema — key ``text``, C.ANCHOR_2329_REQUIRED_KEYS), rows
    filtered to THIS cell set's contexts and normalized with a
    ``response_text`` alias so every downstream item builder is schema-blind."""
    if anchor_mode(anchors_dir, cell_set) == "fresh":
        rows = load_anchor_rows(anchors_dir)
        if cell_set == "q35lang":
            rows = [r for r in rows if r["context_id"] in ctx_ids]
            assert rows, "fresh anchor shards carry no rows for the q35lang contexts"
        return rows
    dest = anchors_dir / "reused_2329"
    shards = sorted(dest.glob("*.jsonl"))
    if not shards:
        shards = _stage_2329_anchor_shards(dest)
    rows: list[dict] = []
    per_ctx: Counter[str] = Counter()
    for shard in shards:
        for row in J94._iter_jsonl(shard):
            if row.get("context_id") not in ctx_ids:
                continue
            missing = C.ANCHOR_2329_REQUIRED_KEYS - set(row)
            assert not missing, (shard.name, sorted(missing))
            per_ctx[row["context_id"]] += 1
            rows.append({**row, "response_text": row["text"]})
    # Exact coverage gate on the CONSUMPTION path (mirrors the pod-side stage
    # check in run.py phase_anchors_reused): the glob-any-nonempty shortcut
    # above would otherwise consume a partially-staged reused_2329/ dir left
    # by an interrupted VM-side staging.
    missing_ctx = sorted(ctx_ids - set(per_ctx))
    under = {c: n for c, n in sorted(per_ctx.items()) if n < C.ANCHOR_DRAWS}
    if missing_ctx or under:
        raise RuntimeError(
            f"staged #2329 anchors under {dest} under-cover the {len(ctx_ids)} requested "
            f"contexts (interrupted staging?): missing={missing_ctx[:5]} "
            f"under_floor={dict(list(under.items())[:5])} (floor {C.ANCHOR_DRAWS}) — "
            f"delete {dest} and re-run to re-stage"
        )
    return rows


def load_calib_s1(calib_dir: Path) -> list[dict]:
    """Banked #2162 ce grid rows (survivor cells x {steered, shuffled}).

    Observed schema (probe 2026-08-16, @ PIN_2162): keys incl. cell / slot /
    arm / pair_id / draw / text / context_id / block_key.
    """
    shards = sorted((calib_dir / "s1").glob("shard_*.jsonl"))
    assert shards, f"no staged S1 calib shards under {calib_dir}/s1 (run --phase stage-calib)"
    rows = [r for s in shards for r in J94._iter_jsonl(s)]
    rows = [
        r
        for r in rows
        if r["slot"] == "ce" and r["cell"] in C.S1_CELLS and r["arm"] in ("steered", "shuffled")
    ]
    assert rows, "S1 calib shards filtered to empty (cell/slot/arm drift?)"
    return rows


def load_calib_s2(calib_dir: Path) -> list[dict]:
    """Banked #2094 fu1 conf1 joint_all replace rows (matched_query, A-carrier).

    Observed schema (probe 2026-08-16, @ PIN_FU1): keys incl. pair_id / arm /
    draw / text / block_key / cell / slot.
    """
    files = sorted((calib_dir / "s2").glob("*.jsonl"))
    assert files, f"no staged S2 calib files under {calib_dir}/s2 (run --phase stage-calib)"
    rows = [r for f in files for r in J94._iter_jsonl(f)]
    assert rows, "S2 calib files present but empty"
    return rows


# ── item builders ─────────────────────────────────────────────────────


def _grid_source(row: dict, kind: str) -> dict:
    return {
        "kind": kind,
        "variant": row.get("variant"),
        "arm": row.get("arm_slug") or row.get("arm"),
        "cell": row.get("cell"),
        "pair_id": row.get("pair_id"),
        "draw": row.get("draw"),
    }


def coherence_key(tag: str, row: dict) -> str:
    return f"{tag}|{row['block_key']}|{row['pair_id']}|{row['draw']}"


def behavior_key(tag: str, row: dict, side: str) -> str:
    return f"{tag}|{row['block_key']}|{row['pair_id']}|{row['draw']}|{side}"


def anchor_coherence_id(context_id: str, draw: int) -> str:
    return J94._item_id("c", f"a|{context_id}|{draw}")


def build_coherence_items(
    grid_rows: list[dict] | None = None,
    ce_rows: list[dict] | None = None,
    anchor_rows: list[dict] | None = None,
    calib_s1: list[dict] | None = None,
    calib_s2: list[dict] | None = None,
) -> list[J94.JudgeUnit]:
    """One coherence call per rollout — fresh AND calibration (same wave)."""
    units: list[J94.JudgeUnit] = []

    def _add(tag: str, row: dict, answer: str, kind: str) -> None:
        units.append(
            J94.JudgeUnit(
                item_id=J94._item_id("c", coherence_key(tag, row)),
                rubric_id=J94.COHERENCE_RUBRIC_ID,
                question="",
                answer=answer,
                source=_grid_source(row, kind),
            )
        )

    for row in grid_rows or []:
        _add("g", row, row["response_text"], "grid")
    for row in ce_rows or []:
        _add("e", row, row["response_text"], "ce")
    for row in anchor_rows or []:
        units.append(
            J94.JudgeUnit(
                item_id=anchor_coherence_id(row["context_id"], row["draw"]),
                rubric_id=J94.COHERENCE_RUBRIC_ID,
                question="",
                answer=row["response_text"],
                source={"kind": "anchor", "context_id": row["context_id"], "draw": row["draw"]},
            )
        )
    for row in calib_s1 or []:
        _add("k", row, row["text"], "calib-s1")
    for row in calib_s2 or []:
        _add("m", row, row["text"], "calib-s2")
    return units


def _dual_rubric_units(
    rows: list[dict],
    pairs_by_id: dict,
    tag: str,
    kind: str,
    answer_field: str,
) -> dict[str, list[J94.JudgeUnit]]:
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    for row in rows:
        cores = pair_rubric_cores_2333(pairs_by_id[row["pair_id"]])
        for side, core in zip(("a", "b"), cores, strict=True):
            rid = J62.rubric_core_id(core)
            answer = row[answer_field]
            src = {**_grid_source(row, kind), "side": side}
            by_rid.setdefault(rid, []).append(
                J94.JudgeUnit(
                    item_id=J94._item_id(tag, behavior_key(tag, row, side)),
                    rubric_id=rid,
                    question="",
                    answer=answer,
                    source=src,
                )
            )
    return by_rid


def build_behavior_items(
    cfg_pairs: tuple[list, list],
    grid_rows: list[dict] | None = None,
    ce_rows: list[dict] | None = None,
    calib_s1: list[dict] | None = None,
    calib_s2: list[dict] | None = None,
) -> dict[str, list[J94.JudgeUnit]]:
    """Dual-rubric behavior units — fresh grid + prefill continuation-only
    companions + q35 ce_control + SAME-WAVE calibration rows."""
    s1_pairs, s2_pairs = cfg_pairs
    pairs_by_id = {p.pair_id: p for p in [*s1_pairs, *s2_pairs]}
    by_rid = _dual_rubric_units(grid_rows or [], pairs_by_id, "g", "grid", "response_text")

    # Prefill continuation-only companions (plan §6): same rubrics, the
    # continuation text WITHOUT the prefilled donor opening.
    prefill_rows = [r for r in grid_rows or [] if r.get("kind") == "prefill"]
    for rid, us in _dual_rubric_units(
        prefill_rows, pairs_by_id, "n", "prefill-cont", "continuation_text"
    ).items():
        by_rid.setdefault(rid, []).extend(us)

    for rid, us in _dual_rubric_units(
        ce_rows or [], pairs_by_id, "e", "ce", "response_text"
    ).items():
        by_rid.setdefault(rid, []).extend(us)
    for rid, us in _dual_rubric_units(calib_s1 or [], pairs_by_id, "k", "calib-s1", "text").items():
        by_rid.setdefault(rid, []).extend(us)
    for rid, us in _dual_rubric_units(calib_s2 or [], pairs_by_id, "m", "calib-s2", "text").items():
        by_rid.setdefault(rid, []).extend(us)
    return by_rid


def build_anchor_behavior_items(
    anchor_rows: list[dict], s1_pairs: list, s2_pairs: list
) -> dict[str, list[J94.JudgeUnit]]:
    """q35 anchor floor/ceiling units — J62 dedup shape, both banks' cores.

    Item ids via ``J62.anchor_unit_id`` so the analysis join mirrors the
    parent convention.
    """
    cores_by_ctx: dict[str, set[str]] = {}
    for p in [*s1_pairs, *s2_pairs]:
        for ctx in (p.a, p.b):
            cores_by_ctx.setdefault(ctx, set()).update(pair_rubric_cores_2333(p))
    by_rid: dict[str, list[J94.JudgeUnit]] = {}
    seen: set[str] = set()
    for row in anchor_rows:
        for core in sorted(cores_by_ctx.get(row["context_id"], ())):
            rid = J62.rubric_core_id(core)
            iid = J62.anchor_unit_id(row["context_id"], row["draw"], rid)
            if iid in seen:
                continue
            seen.add(iid)
            by_rid.setdefault(rid, []).append(
                J94.JudgeUnit(
                    item_id=iid,
                    rubric_id=rid,
                    question="",
                    answer=row["response_text"],
                    source={
                        "kind": "anchor",
                        "context_id": row["context_id"],
                        "draw": row["draw"],
                        "rubric": rid,
                    },
                )
            )
    return by_rid


# ── phase: stage-calib (zero API) ─────────────────────────────────────

_S1_CALIB_FILES = tuple(
    f"shard_{cell}__ce__{arm}.jsonl" for cell in C.S1_CELLS for arm in ("steered", "shuffled")
)
_S2_CALIB_FILES = (
    "fu1_fu1__matched_query__ce__joint_all__replace__A__steered.jsonl",
    "fu1_fu1__matched_query__ce__joint_all__replace__A__null.jsonl",
)


def phase_stage_calib(cfg: JudgeConfig2333) -> int:
    """Stage the banked ce calibration inputs (sha-pinned revisions, plan §10).

    Content hygiene: bank text is WildChat-derived real-corpus — this phase
    logs COUNTS only, never row text.
    """
    if cfg.cell_set != "main":
        # The banked calib legs are Qwen2.5-GENERATED text (#2162 grid /
        # #2094 fu1) — a cross-model re-judge is N/A for the q35lang wave
        # (plan §4.3 item 3: GATE the leg, never re-point it). The same-wave
        # instrument anchor for q35lang is the #2329 ANCHORS-REJUDGE wave.
        logger.info(
            "[stage-calib] cell_set=%s: banked Qwen2.5 calib legs are main-only — N/A (gated)",
            cfg.cell_set,
        )
        return RC_OK
    if cfg.dry_run:
        logger.info(
            "[stage-calib] dry-run: would stage %d S1 shards @ %s + %d S2 files @ %s -> %s",
            len(_S1_CALIB_FILES),
            C.PIN_2162[:12],
            len(_S2_CALIB_FILES),
            C.PIN_FU1[:12],
            cfg.calib_dir,
        )
        return RC_OK
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    for sub, files, prefix, rev in (
        ("s1", _S1_CALIB_FILES, C.R2162_GRID_ROLLOUTS, C.PIN_2162),
        ("s2", _S2_CALIB_FILES, C.FU1_CONF1_PREFIX, C.PIN_FU1),
    ):
        dest = cfg.calib_dir / sub
        dest.mkdir(parents=True, exist_ok=True)
        for name in files:
            target = dest / name
            if target.is_file():
                continue
            got = hub.retry_transient(
                lambda fn=f"{prefix}/{name}", r=rev: hf_hub_download(
                    repo_id=C.DATA_REPO,
                    repo_type="dataset",
                    filename=fn,
                    revision=r,
                    local_dir=dest / "_dl",
                ),
                what=f"stage-calib {sub}/{name}",
            )
            Path(got).replace(target)
        logger.info("[stage-calib] %s: %d files staged", sub, len(files))
    n1, n2 = len(load_calib_s1(cfg.calib_dir)), len(load_calib_s2(cfg.calib_dir))
    logger.info("[stage-calib] rows: s1=%d s2=%d", n1, n2)
    assert n1 > 0 and n2 > 0
    return RC_OK


# ── phase: pilot (rule 26; REFUSES --dry-run) ─────────────────────────


def _pilot_arm(u: J94.JudgeUnit) -> str:
    src = u.source
    return f"{src.get('kind', 'unknown')}.{src.get('variant') or 'na'}"


def forced_batch_probe_verdict(
    scores: dict[str, float | None], stop_reason_tally: dict[str, int], n_items: int
) -> tuple[bool, dict]:
    """Registered forced-batch gate (plan §7): the probe is EXACTLY
    ``J94.FORCED_BATCH_PROBE_N`` items, ALL scored, and EVERY persisted draw
    stop_reason is ``end_turn`` (r1 Minor: the shipped ``n_probe >= 1`` was
    weaker than the registered 6/6-all-end_turn criterion). A cache-served
    legacy entry tallies ``unknown`` and FAILS — run the probe against its
    fresh ``_forced_batch`` cache dir."""
    n_scored = sum(1 for v in scores.values() if v is not None)
    tally = dict(stop_reason_tally)
    non_end_turn = {k: v for k, v in tally.items() if k != "end_turn" and v}
    passed = (
        n_items == J94.FORCED_BATCH_PROBE_N
        and n_scored == J94.FORCED_BATCH_PROBE_N
        and sum(tally.values()) >= J94.FORCED_BATCH_PROBE_N
        and not non_end_turn
    )
    return passed, {
        "n_items": n_items,
        "n_scored": n_scored,
        "required": J94.FORCED_BATCH_PROBE_N,
        "stop_reason_tally": tally,
        "non_end_turn": non_end_turn,
        "passed": passed,
    }


def phase_pilot(cfg: JudgeConfig2333) -> int:
    """Rule-26 pilot per rubric family (coherence / S1 rubric / S2 rubric)
    plus the live forced-batch request-shape probe."""
    if cfg.base.dry_run:
        logger.error(
            "[pilot] --dry-run refused (J62 fix 3): the rule-26 pilot's purpose is "
            "measuring the REAL instrument — run without --dry-run."
        )
        return RC_DRY_RUN_UNSUPPORTED
    s1_pairs, s2_pairs = build_pair_universe(cfg.cell_set)
    registry = rubric_registry(s1_pairs, s2_pairs)
    grid_rows = load_grid_rows(cfg.rollouts_dir, cell_set=cfg.cell_set)
    calib_s1 = load_calib_s1(cfg.calib_dir) if cfg.cell_set == "main" else []
    calib_s2 = load_calib_s2(cfg.calib_dir) if cfg.cell_set == "main" else []
    coh = build_coherence_items(grid_rows, None, None, calib_s1, calib_s2)
    beh = build_behavior_items((s1_pairs, s2_pairs), grid_rows, None, calib_s1, calib_s2)

    s1_rids = {J62.rubric_core_id(c) for p in s1_pairs for c in pair_rubric_cores_2333(p)}
    s2_rids = {J62.rubric_core_id(c) for p in s2_pairs for c in pair_rubric_cores_2333(p)}
    rep_s1 = max((r for r in by_len_order(beh) if r in s1_rids), key=lambda r: len(beh[r]))
    fam_reps = {
        "coherence": (J94.COHERENCE_RUBRIC_ID, coh, J62.PILOT_TARGET_COHERENCE),
        "s1-rubric": (rep_s1, beh[rep_s1], J62.PILOT_TARGET_BEHAVIOR),
    }
    if s2_pairs:  # S2 OFF on q35lang — an empty family would crash max()
        rep_s2 = max((r for r in by_len_order(beh) if r in s2_rids), key=lambda r: len(beh[r]))
        fam_reps["s2-rubric"] = (rep_s2, beh[rep_s2], J62.PILOT_TARGET_BEHAVIOR)
    per_family: dict[str, dict] = {}
    all_pass = True
    for family, (rid, units, target) in fam_reps.items():
        J94._validate_units(units)
        arms: dict[str, list[tuple[str, str, str]]] = {}
        for u in units:
            arms.setdefault(_pilot_arm(u), []).append((u.item_id, u.question, u.answer))
        # Rule-26(b) satisfiability: the gate needs >= 51 effective draws per
        # unwaived arm (floor(1/0.02)+1 at its default parse_fail_threshold)
        # and floor-divides the budget across arms. The inherited J62 targets
        # were sized for the parent's arm structure; #2333's families carry
        # more arms (calib-s1/calib-s2/grid.null/grid.steered), so re-derive
        # from the realized arm count with headroom for transport-loss
        # shrinkage of effective draws (60 > 51 floor).
        per_arm_items = -(-60 // JUDGE_N_DRAWS)  # ceil division
        target = max(target, len(arms) * JUDGE_N_DRAWS * per_arm_items)
        report = judge_pilot_gate(
            arms,
            registry[rid],
            max_tokens=cfg.base.max_tokens,
            cache_dir=cfg.base.pilot_cache_root / rid,
            save_raw_dir=cfg.base.raw_dir / "pilot" / rid,
            n_draws=JUDGE_N_DRAWS,
            target_total_draws=target,
            judge_model=cfg.base.judge_model,
            report_path=cfg.base.gates_dir / "pilot" / f"{family}.json",
            seed=PILOT_SEED,
        )
        per_family[family] = {
            "rubric_id": rid,
            "verdict": report.verdict,
            "failures": report.failures,
            "warnings": report.warnings,
            "n_total_draws": report.n_total_draws,
        }
        all_pass &= report.passed
        logger.info("[pilot] %s (%s): %s", family, rid, report.verdict)

    probe_units = coh[: J94.FORCED_BATCH_PROBE_N]
    probe = judge_graded(
        [(u.item_id, u.question, u.answer) for u in probe_units],
        registry[J94.COHERENCE_RUBRIC_ID],
        n_draws=JUDGE_N_DRAWS,
        cache_dir=cfg.base.pilot_cache_root / "_forced_batch",
        save_raw=cfg.base.raw_dir / "pilot" / "forced_batch_probe.json",
        judge_model=cfg.base.judge_model,
        max_tokens=cfg.base.max_tokens,
        threshold_base=0,
    )
    probe_ok, probe_report = forced_batch_probe_verdict(
        probe.scores, probe.stop_reason_tally, len(probe_units)
    )
    all_pass &= probe_ok
    logger.info("[pilot] forced-batch probe: %s", probe_report)
    J94._write_json_atomic(
        cfg.base.gates_dir / "pilot_gate_report.json",
        {
            "passed": all_pass,
            "per_family": per_family,
            "forced_batch_probe": probe_report,
            "instrument": {
                "judge_model": cfg.base.judge_model,
                "max_tokens": cfg.base.max_tokens,
                "n_draws": JUDGE_N_DRAWS,
                "n_rubrics_total": len(registry),
                "model_tag": cfg.model_tag,
                "cell_set": cfg.cell_set,
            },
            "repro": J94._repro(),
        },
    )
    logger.info("[pilot] aggregate verdict: %s", "PASS" if all_pass else "FAIL")
    return RC_OK if all_pass else RC_PILOT_GATE


def by_len_order(by_rid: dict[str, list]) -> list[str]:
    return sorted(by_rid, key=lambda r: -len(by_rid[r]))


# ── phase: anchors (q35 only) ─────────────────────────────────────────


def _require_gates(cfg: JudgeConfig2333, names: tuple[str, ...]) -> None:
    for name in names:
        path = cfg.base.gates_dir / name
        if not path.is_file():
            raise RuntimeError(f"gate report missing: {path} — run the gate phase first")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not rec.get("passed"):
            override = rec.get("override") or {}
            if override.get("accepted_fail"):
                # Audited accepted-fail override (--accept-coherence-gate-fail):
                # the verdict stays FAIL on record; downstream phases proceed.
                logger.warning(
                    "[gates] %s FAILED but carries an audited accepted_fail override: %s",
                    name,
                    override.get("reason"),
                )
                continue
            raise RuntimeError(f"gate FAILED per {path} — fix the instrument and re-run")


def phase_anchors(cfg: JudgeConfig2333) -> int:
    """q35: coherence-baseline gate over the anchors, then anchor waves.

    q35lang = the ANCHORS-REJUDGE wave (plan §4.3 item 3): the staged #2329
    anchor completions (or selfgen fresh anchors) are scored INSIDE this
    run's judge waves — same instrument as every fresh row — so the analysis
    judge-offset table (ours vs #2329's stored per-pair deltas on IDENTICAL
    texts) reads pure instrument drift."""
    assert cfg.model_tag == "q35", "anchors phase is q35-only (q25 reuses banked anchor scores)"
    assert cfg.anchors_dir is not None
    s1_pairs, s2_pairs = build_pair_universe(cfg.cell_set)
    ctx_ids = {cid for p in [*s1_pairs, *s2_pairs] for cid in (p.a, p.b)}
    anchor_rows = load_anchor_rows_2333(cfg.anchors_dir, cfg.cell_set, ctx_ids)
    logger.info(
        "[anchors] mode=%s rows=%d contexts=%d",
        anchor_mode(cfg.anchors_dir, cfg.cell_set),
        len(anchor_rows),
        len({r["context_id"] for r in anchor_rows}),
    )
    if cfg.base.dry_run:
        beh = build_anchor_behavior_items(anchor_rows, s1_pairs, s2_pairs)
        return J62._dry_run_units_report(
            "anchors",
            {
                "coherence.anchors": build_coherence_items(anchor_rows=anchor_rows),
                **{f"{rid}.anchors": us for rid, us in beh.items()},
            },
        )
    _require_gates(cfg, ("pilot_gate_report.json",))
    registry = rubric_registry(s1_pairs, s2_pairs)
    coh_units = build_coherence_items(anchor_rows=anchor_rows)
    J94.run_wave(
        "coherence.anchors",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg.base,
    )
    scores = list(J94._iter_jsonl(cfg.base.scores_dir / "coherence.anchors.scores.jsonl"))
    gate = J94.coherence_baseline_gate(scores)
    if not gate["passed"] and cfg.accept_coherence_gate_fail:
        gate["override"] = {
            "accepted_fail": True,
            "reason": cfg.accept_coherence_gate_fail,
        }
    J94._write_json_atomic(cfg.base.gates_dir / "coherence_baseline_gate.json", gate)
    logger.info(
        "[gate] q35 coherence baseline: median=%.1f frac>60=%.3f -> %s",
        gate["median"],
        gate["frac_gt60"],
        "PASS" if gate["passed"] else "FAIL",
    )
    if not gate["passed"]:
        if cfg.accept_coherence_gate_fail:
            logger.warning(
                "[gate] coherence-baseline FAIL OVERRIDDEN (audited): %s",
                cfg.accept_coherence_gate_fail,
            )
        else:
            return RC_COHERENCE_GATE
    for rid, units in sorted(build_anchor_behavior_items(anchor_rows, s1_pairs, s2_pairs).items()):
        J94.run_wave(f"{rid}.anchors", rid, registry[rid], units, cfg.base)
    return RC_OK


# ── phase: waves ──────────────────────────────────────────────────────


def phase_waves(cfg: JudgeConfig2333) -> int:
    """Production grid waves: coherence + dual-rubric behavior over fresh grid
    rows, prefill continuation companions, q35 ce_control rows, and the
    SAME-WAVE banked ce calibration rows."""
    s1_pairs, s2_pairs = build_pair_universe(cfg.cell_set)
    grid_rows = load_grid_rows(cfg.rollouts_dir, cell_set=cfg.cell_set)
    ce_rows = (
        load_ce_rows(cfg.rollouts_dir, cell_set=cfg.cell_set) if cfg.model_tag == "q35" else []
    )
    calib_s1 = load_calib_s1(cfg.calib_dir) if cfg.cell_set == "main" else []
    calib_s2 = load_calib_s2(cfg.calib_dir) if cfg.cell_set == "main" else []
    if cfg.base.dry_run:
        beh = build_behavior_items((s1_pairs, s2_pairs), grid_rows, ce_rows, calib_s1, calib_s2)
        return J62._dry_run_units_report(
            "waves",
            {
                "coherence.grid": build_coherence_items(
                    grid_rows, ce_rows, None, calib_s1, calib_s2
                ),
                **{f"{rid}.grid": us for rid, us in beh.items()},
            },
        )
    gates = ("pilot_gate_report.json",)
    if cfg.model_tag == "q35":
        gates = ("pilot_gate_report.json", "coherence_baseline_gate.json")
    _require_gates(cfg, gates)
    registry = rubric_registry(s1_pairs, s2_pairs)
    coh_units = build_coherence_items(grid_rows, ce_rows, None, calib_s1, calib_s2)
    J94.run_wave(
        "coherence.grid",
        J94.COHERENCE_RUBRIC_ID,
        registry[J94.COHERENCE_RUBRIC_ID],
        coh_units,
        cfg.base,
    )
    beh = build_behavior_items((s1_pairs, s2_pairs), grid_rows, ce_rows, calib_s1, calib_s2)
    for rid, units in sorted(beh.items()):
        J94.run_wave(f"{rid}.grid", rid, registry[rid], units, cfg.base)
    return RC_OK


# ── phase: upload-raw ─────────────────────────────────────────────────


def judge_hf_namespace(model_tag: str, cell_set: str) -> str:
    """HF prefix segment: the cell set's namespace tag replaces model_tag
    (mirrors ``issue2333_run.hf_prefix`` — plan §10 q35lang namespace)."""
    return C.CELL_SETS[cell_set].hf_namespace or model_tag


def phase_upload_raw(cfg: JudgeConfig2333) -> int:
    ns = judge_hf_namespace(cfg.model_tag, cfg.cell_set)
    prefix = f"{C.HF_PREFIX}/{ns}/raw_completions/judge_raw"
    if cfg.base.dry_run:
        logger.info("[upload-raw] dry-run: would upload %s -> %s", cfg.base.work_root, prefix)
        return RC_OK
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        cfg.base.work_root,
        repo_id=C.DATA_REPO,
        repo_type="dataset",
        path_in_repo=prefix,
        raise_on_error=True,
    )
    logger.info("[upload-raw] uploaded %s -> %s", cfg.base.work_root, url)
    return RC_OK


# ── CLI ───────────────────────────────────────────────────────────────

PHASES = {
    "stage-calib": phase_stage_calib,
    "pilot": phase_pilot,
    "anchors": phase_anchors,
    "waves": phase_waves,
    "upload-raw": phase_upload_raw,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2333 VM-side judge pipeline.")
    # required unless --import-check (r1 Minor: the standalone import-check
    # invocation must parse without a phase; main() asserts them otherwise).
    ap.add_argument("--phase", choices=tuple(PHASES))
    ap.add_argument("--model-tag", choices=("q25", "q35"))
    ap.add_argument(
        "--cell-set",
        choices=tuple(C.CELL_SETS),
        default="main",
        help="pair universe (default 'main' = the parent run; 'q35lang' = the 2 "
        "Qwen3.5 language cells, S2 off, banked calib legs gated)",
    )
    ap.add_argument(
        "--in-root",
        type=Path,
        default=Path("data/issue_2333/judge_inputs"),
        help=f"staging mirror root; rollouts default under <in-root>/{C.HF_PREFIX}/<ns>/",
    )
    ap.add_argument("--rollouts-dir", type=Path, default=None)
    ap.add_argument("--anchors-dir", type=Path, default=None)
    ap.add_argument("--calib-dir", type=Path, default=None)
    ap.add_argument("--stage-from-hf", action="store_true")
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument("--work-root", type=Path, default=None)
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--judge-model", type=str, default=J94.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument(
        "--accept-coherence-gate-fail",
        type=str,
        default=None,
        help="audited override: proceed past a coherence-baseline gate FAIL; the "
        "non-empty reason string is written verbatim into coherence_baseline_gate.json "
        "(the gate verdict stays FAIL on record)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="construction check: build + validate every judge unit, ZERO API calls; "
        "--phase pilot REFUSES it (rc 10)",
    )
    return ap.parse_args(argv)


def _stage_inputs(args: argparse.Namespace) -> None:
    from explore_persona_space.orchestrate import hub

    base = f"{C.HF_PREFIX}/{judge_hf_namespace(args.model_tag, args.cell_set)}"
    need = [f"{base}/rollouts"]
    if args.model_tag == "q35" and args.phase in ("anchors", "waves"):
        need.append(f"{base}/anchors")
    for prefix in need:
        try:
            staged = hub.stage_hub_prefix(
                C.DATA_REPO, prefix, args.in_root, revision=args.hf_revision
            )
        except FileNotFoundError:
            if args.cell_set == "q35lang" and prefix.endswith("/anchors"):
                # EXPECTED on the anchors-REUSE path: the #2329 shards are
                # never mirrored under the q35lang namespace — the loader
                # stages them from C.PIN_2329_DATA on demand.
                logger.info(
                    "[stage] %s: empty prefix — reused_2329 anchors path assumed "
                    "(loader stages from the #2329 pin)",
                    prefix,
                )
                continue
            raise
        logger.info("[stage] %s: %d files", prefix, len(staged))
        assert staged, f"nothing staged from {prefix}"


def _assert_fresh_cache_partition(cache_root: Path, cell_set: str, dry_run: bool) -> None:
    """Non-main judge waves run against their OWN cache partition (plan §4.3
    item 3 / §10 fresh-cache pin): a parent-leg or #2329-era cache entry
    served into the ANCHORS-REJUDGE wave would flatten the judge-offset table
    to ~zero variance (the contamination tell). A ``.cell_set`` marker pins
    the partition; a mismatching or unmarked non-empty partition REFUSES."""
    marker = cache_root / ".cell_set"
    if marker.is_file():
        # The mismatch refusal binds BOTH directions (a main run into a
        # q35lang-pinned partition would poison later q35lang resumes);
        # parent-safe: no pre-marker main cache can carry this file.
        prev = marker.read_text(encoding="utf-8").strip()
        if prev != cell_set:
            raise RuntimeError(
                f"judge cache partition {cache_root} is pinned to cell_set={prev!r}, "
                f"not {cell_set!r} — point --cache-root at a fresh partition"
            )
        return
    if cell_set == "main":
        return  # unmarked parent caches are grandfathered (pre-marker layout)
    if cache_root.exists() and any(cache_root.iterdir()):
        raise RuntimeError(
            f"judge cache partition {cache_root} is non-empty with NO .cell_set marker "
            f"(a pre-existing foreign cache) — point --cache-root at a fresh partition"
        )
    if not dry_run:
        cache_root.mkdir(parents=True, exist_ok=True)
        marker.write_text(cell_set + "\n", encoding="utf-8")


def build_config(args: argparse.Namespace) -> JudgeConfig2333:
    if args.cell_set == "q35lang" and args.model_tag != "q35":
        raise SystemExit("--cell-set q35lang requires --model-tag q35 (plan §11)")
    ns = judge_hf_namespace(args.model_tag, args.cell_set)
    mirror = args.in_root / C.HF_PREFIX / ns
    rollouts = args.rollouts_dir if args.rollouts_dir is not None else mirror / "rollouts"
    anchors = args.anchors_dir if args.anchors_dir is not None else mirror / "anchors"
    calib = args.calib_dir if args.calib_dir is not None else args.in_root / "calib"
    if args.cell_set == "main":
        default_work = Path(f"eval_results/issue_2333/judge_{args.model_tag}")
        default_cache = Path(f"data/issue_2333/judge_cache_{args.model_tag}")
    else:
        # Disjoint per-cell-set partitions (out-dir plan §4.3 item 4; the
        # cache partition is the anti-contamination pin asserted below).
        default_work = Path(f"eval_results/issue_2333/{ns}/judge")
        default_cache = Path(f"data/issue_2333/judge_cache_{args.cell_set}")
    work_root = args.work_root or default_work
    cache_root = args.cache_root or default_cache
    base = J94.JudgeConfig(
        work_root=work_root,
        cache_root=cache_root,
        rollouts_dir=rollouts,
        anchors_file=anchors,
        stage2_dir=None,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )
    return JudgeConfig2333(
        base=base,
        model_tag=args.model_tag,
        rollouts_dir=rollouts,
        anchors_dir=anchors,
        calib_dir=calib,
        cell_set=args.cell_set,
        accept_coherence_gate_fail=args.accept_coherence_gate_fail,
    )


def _import_check() -> int:
    """Deferred-import + args-attribute completeness check (code-style.md)."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    from huggingface_hub import hf_hub_download  # noqa: F401

    from explore_persona_space.orchestrate import hub  # noqa: F401

    assert (
        callable(hub._upload) and callable(hub.stage_hub_prefix) and callable(hub.retry_transient)
    )
    s1_pairs, s2_pairs = build_pair_universe()
    reg = rubric_registry(s1_pairs, s2_pairs)
    # 5 S1 cells x 36 pairs contribute per-pair cores; 15 S2 pairs x 2 prefix
    # cores; plus coherence. Exact count depends on core dedup — assert bounds.
    assert len(reg) >= 3, len(reg)
    # Parametrized per cell-set (plan §4.3 item 3): the S1 tuple tracks the
    # MAIN cell list (2 arms per cell), never a literal 10.
    assert len(_S1_CALIB_FILES) == 2 * len(C.S1_CELLS) and len(_S2_CALIB_FILES) == 2
    # q35lang universe + rubric composition resolve (language rubric cores
    # exist in bank2162 — plan §4.3 reuse note).
    s1_lang, s2_lang = build_pair_universe("q35lang")
    assert len(s1_lang) == C.CELL_SETS["q35lang"].expected_pairs and not s2_lang
    reg_lang = rubric_registry(s1_lang, s2_lang)
    assert len(reg_lang) >= 3, len(reg_lang)
    print("[import-check] OK")
    return RC_OK


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        return _import_check()
    assert args.phase and args.model_tag, "--phase and --model-tag required (or --import-check)"
    if args.stage_from_hf:
        _stage_inputs(args)
    cfg = build_config(args)
    _assert_fresh_cache_partition(cfg.base.cache_root, cfg.cell_set, cfg.base.dry_run)
    for d in (
        cfg.base.scores_dir,
        cfg.base.items_dir,
        cfg.base.raw_dir,
        cfg.base.gates_dir,
        cfg.base.audits_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    rc = PHASES[args.phase](cfg)
    logger.info("[phase=%s_done] rc=%d", args.phase, rc)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
