#!/usr/bin/env python
"""Issue #2479 — rule-26 judge pilot gates for the two >=5k-call wave families (plan §7).

Arms are pooled per RUBRIC FAMILY (plan §7 ARM DEFINITION — two arms), each
piloted at ~150 draws at the EXACT production instrument BEFORE its full wave
dispatches, with a persisted PASS report the production launchers require:

  --family axis    the 0-100 graded ai_likeness judge (`issue1345_onpolicy_
                   judge_legs`: Sonnet, 5 draws/item, temp 1.0, max_tokens
                   1024, forced-Batch threshold_base=0). Routed VERBATIM
                   through `eval.judge_pilot.judge_pilot_gate` — the same
                   `judge_graded` path `run_leg` dispatches — with a
                   pilot-only cache (rule 24(ii)). Items pooled across every
                   character whose `axis_items_<name>.jsonl` exists (the
                   rubric is byte-identical + character-agnostic, so pooling
                   is exact); the gate subsamples to the draw target itself.
  --family ingen   the in-generation story quality judge (`issue1345_gen_
                   stories_paired`: reason-then-verdict EXCHANGES/VERDICT
                   instrument, Sonnet, temp 0.0, max_tokens 1024, dispatched
                   via `llm.api_dispatch` with the gen module's own
                   `_build_judge_request`/`_parse_judge_response`).
                   ADDRESSED DIFFERENTLY from a literal `judge_pilot_gate`
                   routing (which plan §7 names): `judge_pilot_gate`
                   hardcodes `judge_graded`, whose parse contract is a graded
                   0-100 score — routing this PASS/FAIL verdict instrument
                   through it would parse-fail 100% BY CONSTRUCTION and gate
                   nothing. The family therefore pilots through the gen
                   module's own production seam (same builder / parser /
                   model / temp / max_tokens; dispatcher-decided route,
                   recorded) and persists a PilotGateReport-compatible JSON
                   with the same gate arithmetic. Because the judge system
                   prompt embeds the CELL's character name at import, the
                   family pilots per cell (`--partial-out`, one fresh
                   process per cell env — the wrapper provides it) and the
                   partials merge into ONE family report (`--merge`).
  --require-pass   read a persisted pilot report; exit non-zero unless
                   verdict == PASS (importable as `require_pilot_pass` — the
                   production dispatch gate).

Gate (plan §7): zero `stop_reason == "max_tokens"` + parse-fail < 2% per
family arm, over answered draws (transport-class losses excluded from every
denominator, rule 24). Flatness + name-mask legs (4k + 1.6k draws — under
the 5k pilot floor) REUSE a valid axis-family pilot PASS: arms pool per
rubric family and those legs run the byte-identical axis instrument.

Spend: `--execute` plus EPM_I1345_JUDGE_SPEND_OK=1 (the #1345 rig's spend
ack) dispatches the real pilot (~150 calls per family — plan §9 "pilots
~= 450 sync"); without both, dispatch modes refuse loudly BEFORE any build.

Content hygiene: story rows / axis items are LMSYS-derived real user text —
never printed; diagnostics are counts, paths, hashes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
import issue1345_onpolicy_judge_legs as jl  # noqa: E402

ISSUE = 2479
FAMILIES = ("axis", "ingen")
PILOT_AXIS_REL = "eval_results/issue_2479/pilot_gate_axis.json"
PILOT_INGEN_REL = "eval_results/issue_2479/pilot_gate_ingen.json"
TARGET_DRAWS = 150  # plan §7: ~150-draw pilot per rubric family
PARSE_FAIL_MAX = 0.02  # plan §7: parse-fail < 2% per family arm
INGEN_MIN_EFFECTIVE = 100  # hollow-arm floor for the merged in-gen family arm
# rule-26 satisfiability floor: at threshold 2% the smallest arm that can hold
# one failure under the strict `rate >= threshold` gate is floor(1/0.02)+1.
AXIS_MIN_EFFECTIVE = 51
MIN_EFFECTIVE_BY_FAMILY = {"axis": AXIS_MIN_EFFECTIVE, "ingen": INGEN_MIN_EFFECTIVE}
# Opt-in env consumed by `jl.run_leg` (and exported by the P3 wrapper): a real
# axis-leg spend refuses without a persisted axis-family pilot PASS at this path.
REQUIRE_AXIS_PILOT_ENV = "EPM_I2479_REQUIRE_AXIS_PILOT_PASS"
# r4 codex `smoke-root-production-poisoning`: a smoke-SYNTHESIZED pilot report
# (smoke_synthesized=true) NEVER licenses production spend. The p3-controls
# smoke driver (scripts/issue2479_p3_controls_smoke.py) is the ONLY setter of
# this env — it exercises require_pilot_pass's real read path at every spend
# seam against its scratch-synthesized PASS; no production launcher sets it.
ALLOW_SYNTHESIZED_ENV = "EPM_I2479_ALLOW_SMOKE_SYNTHESIZED_PILOT"
# Data-identity input resolution (r4 codex `judge-pilot-gates-missing`): the
# SAME env names + defaults the P3 wrapper uses, so every spend path — the
# wrapper's --require-pass gates, run_leg's env guard, and the control legs'
# pilot reuse — recomputes the expected identity against the same files.
PANEL_ENV = "EPM_I2479_CHAR_PANEL_JSON"
MANIFEST_ENV = "EPM_I2479_PANEL_MANIFEST"
ITEMS_DIR_ENV = "EPM_I2479_AXIS_ITEMS_DIR"
DEFAULT_ITEMS_DIR_REL = "data/issue_2479/axis_items"


def _resolve_input(env: str, default_rel: str) -> Path:
    """Env-overridable input path; relative values resolve against the repo root."""
    raw = os.environ.get(env, "").strip()
    p = Path(raw) if raw else Path(default_rel)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _sha256_file(path: Path) -> str:
    """Hex sha256 of a data-identity input file (fail-loud on a missing file)."""
    assert path.is_file(), f"pilot data-identity input missing: {path}"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axis_instrument_fingerprint() -> dict:
    """The CANONICAL axis-family production instrument, derived from the SAME
    `jl` constants `run_leg` dispatches with — never re-typed literals, so a
    constant change re-fingerprints producer AND expectation together."""
    return {
        "judge_model": jl.JUDGE_MODEL,
        "n_draws": jl.N_DRAWS,
        "temperature": jl.JUDGE_TEMPERATURE,
        "max_tokens": jl.JUDGE_MAX_TOKENS,
        "threshold_base": jl.THRESHOLD_BASE_FORCE_BATCH,
        # threshold_base=0 forces the Batch API — the wave's dispatch route.
        "dispatch_route": "forced-batch",
        "rubric_sha256": hashlib.sha256(jl.AI_LIKENESS_RUBRIC.encode()).hexdigest(),
    }


def ingen_instrument_fingerprint() -> dict:
    """The CANONICAL in-gen family production instrument (name-INDEPENDENT).

    The judge system prompt embeds the cell's character name at import, so
    the cross-cell identity uses SOURCE hashes of the gen module's own
    builder/parser (the exact production seam `dispatch_calls` runs) instead
    of the name-dependent rendered-prompt shas (those stay per-partial).
    """
    import inspect

    import issue1345_gen_stories_paired as gp

    return {
        "judge_model": c.JUDGE_MODEL,
        "max_tokens": c.JUDGE_MAX_TOKENS,
        "temperature": 0.0,
        # force_path=None: the dispatcher decides, exactly as production.
        "dispatch_route": "dispatcher-decided",
        "builder_sha256": hashlib.sha256(
            inspect.getsource(gp._build_judge_request).encode()
        ).hexdigest(),
        "parser_sha256": hashlib.sha256(
            inspect.getsource(gp._parse_judge_response).encode()
        ).hexdigest(),
    }


def expected_instrument(family: str) -> dict:
    """The current-production instrument fingerprint for a pilot family."""
    assert family in FAMILIES, f"unknown pilot family {family!r}"
    return axis_instrument_fingerprint() if family == "axis" else ingen_instrument_fingerprint()


def axis_data_identity(
    *,
    panel_path: Path | None = None,
    manifest_path: Path | None = None,
    items_glob: str | None = None,
) -> dict:
    """The CURRENT axis-family DATA identity (r4 codex `judge-pilot-gates-missing`).

    A rule-26 pilot PASS certifies the instrument ON the materialization it
    sampled, so a PASS from an earlier materialization — older panel /
    panel_manifest / emitted item pool — must not license today's spend even
    when every instrument field matches. Hashes the panel + manifest BYTES and
    the pooled emitted item CONTENT (the same `build_axis_arm` pool the pilot
    judges, content-fingerprinted through the same helper `run_leg` records).
    """
    import issue2479_freeze_axis as fz

    panel_p = Path(panel_path) if panel_path else _resolve_input(PANEL_ENV, fz.PANEL_REL)
    manifest_p = (
        Path(manifest_path) if manifest_path else _resolve_input(MANIFEST_ENV, fz.MANIFEST_REL)
    )
    glob = items_glob or str(
        _resolve_input(ITEMS_DIR_ENV, DEFAULT_ITEMS_DIR_REL) / "axis_items_{name}.jsonl"
    )
    panel = fz.load_panel(panel_p)
    items, present = build_axis_arm(panel, glob)
    return {
        "panel_sha256": _sha256_file(panel_p),
        "panel_manifest_sha256": _sha256_file(manifest_p),
        "items_content_sha256": jl.items_content_fingerprint(items),
        "n_pooled_items": len(items),
        "characters_pooled": sorted(present),
    }


def ingen_data_identity(
    *, panel_path: Path | None = None, manifest_path: Path | None = None
) -> dict:
    """The CURRENT in-gen family DATA identity (name-INDEPENDENT).

    The ingen pilot certifies the judge instrument on the story distribution
    the CURRENT generation recipe produces over the CURRENT panel, so the
    identity binds the panel + panel_manifest bytes plus the generation
    module's SOURCE bytes (the name-independent generation-config identity —
    the rendered story/judge templates embed the cell's character name at
    import and therefore stay per-partial, like the rendered-prompt shas).
    """
    import issue1345_gen_stories_paired as gp
    import issue2479_freeze_axis as fz

    panel_p = Path(panel_path) if panel_path else _resolve_input(PANEL_ENV, fz.PANEL_REL)
    manifest_p = (
        Path(manifest_path) if manifest_path else _resolve_input(MANIFEST_ENV, fz.MANIFEST_REL)
    )
    return {
        "panel_sha256": _sha256_file(panel_p),
        "panel_manifest_sha256": _sha256_file(manifest_p),
        "gen_module_sha256": _sha256_file(Path(gp.__file__)),
    }


def expected_data_identity(family: str) -> dict:
    """The current-materialization data identity for a pilot family."""
    assert family in FAMILIES, f"unknown pilot family {family!r}"
    return axis_data_identity() if family == "axis" else ingen_data_identity()


def _effective_draws(rep: dict, family: str) -> int:
    """Answered (non-transport-lost) draws of the family arm.

    The merged in-gen report persists `effective_draws` directly;
    `ArmPilotStats` (axis) has no such field, so derive
    `n_draws - n_transport_lost` (rule 24: transport losses leave every
    denominator).
    """
    arm = (rep.get("arms") or {}).get(family) or {}
    if arm.get("effective_draws") is not None:
        return int(arm["effective_draws"])
    return int(arm.get("n_draws") or 0) - int(arm.get("n_transport_lost") or 0)


def _metadata(script: str) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        "script": script,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **as_metadata_dict(git_provenance()),
    }


def require_pilot_pass(
    report_path: Path,
    family: str | None = None,
    *,
    expected: dict | None = None,
    expected_data: dict | None = None,
    min_effective_draws: int | None = None,
    allow_synthesized: bool | None = None,
) -> dict:
    """Load a persisted pilot report; RAISE unless it is a PASS bound to the
    CURRENT production instrument AND the CURRENT data materialization
    (returns it).

    A rule-26 pilot PASS certifies ONLY the instrument it ran, ON the data it
    sampled (llm-judging.md rule 26), so when ``family`` is given this ALSO
    refuses (r2 codex `judge-pilot-gates-missing` / g6; r4 data-identity
    extension): a report with NO persisted ``instrument`` block; any field of
    the persisted instrument differing from the current-production
    fingerprint (``expected`` override for tests); a report with NO persisted
    ``data_identity`` block; any field of the persisted data identity
    differing from the current materialization — panel / panel_manifest bytes
    + the pooled item content (axis) or the generation-module source (ingen)
    (``expected_data`` override for tests); and an effective-draw count below
    the family floor (``MIN_EFFECTIVE_BY_FAMILY`` — the rule-26
    satisfiability floor). Every production spend path routes here: the P3
    wrapper's `--require-pass` gate, `jl.run_leg`'s env-armed axis guard, the
    P1 preamble's ingen gate, and `issue2479_instrument_gates.py`'s
    flatness/name-mask pilot reuse.

    A report carrying ``smoke_synthesized: true`` is REFUSED regardless of
    family/verdict/identity unless ``allow_synthesized`` is True (or, when
    None, the ``ALLOW_SYNTHESIZED_ENV`` env is "1" — set ONLY by the
    p3-controls smoke driver; r4 codex `smoke-root-production-poisoning`).
    """
    report_path = Path(report_path)
    if not report_path.is_file():
        raise RuntimeError(
            f"rule-26 pilot gate report missing: {report_path} — run "
            f"scripts/issue2479_judge_pilots.py --family {family or '<family>'} --execute "
            "before the production wave (plan §7)"
        )
    rep = json.loads(report_path.read_text())
    if bool(rep.get("smoke_synthesized")):
        allowed = (
            allow_synthesized
            if allow_synthesized is not None
            else os.environ.get(ALLOW_SYNTHESIZED_ENV) == "1"
        )
        if not allowed:
            raise RuntimeError(
                f"{report_path}: pilot report carries smoke_synthesized=true — a "
                "smoke-SYNTHESIZED PASS never licenses production spend (r4 codex "
                "smoke-root-production-poisoning); run the real pilot "
                f"(scripts/issue2479_judge_pilots.py --family "
                f"{rep.get('family') or family or '<family>'} --execute). Only the "
                f"p3-controls smoke driver sets {ALLOW_SYNTHESIZED_ENV}=1."
            )
    if family is not None and rep.get("family") != family:
        raise RuntimeError(
            f"{report_path}: pilot family {rep.get('family')!r} != required {family!r}"
        )
    if rep.get("verdict") != "PASS" or not rep.get("passed"):
        raise RuntimeError(
            f"rule-26 pilot gate {rep.get('family')!r} verdict={rep.get('verdict')!r} — "
            f"production dispatch refused (failures: {rep.get('failures')})"
        )
    if family is not None:
        exp = expected if expected is not None else expected_instrument(family)
        inst = rep.get("instrument")
        if not isinstance(inst, dict):
            raise RuntimeError(
                f"{report_path}: pilot report persists NO instrument block — a PASS "
                "unbound to the production instrument certifies nothing; re-run the "
                f"pilot on current code (--family {family} --execute)"
            )
        mismatches = sorted(
            f"{k}: report={inst.get(k)!r} != expected={exp[k]!r}"
            for k in exp
            if k not in inst or inst[k] != exp[k]
        )
        if mismatches:
            raise RuntimeError(
                f"{report_path}: persisted pilot instrument does not match the CURRENT "
                f"production instrument — stale PASS refused; re-pilot (--family {family} "
                f"--execute). Mismatched fields: {mismatches}"
            )
        data_exp = expected_data if expected_data is not None else expected_data_identity(family)
        data = rep.get("data_identity")
        if not isinstance(data, dict):
            raise RuntimeError(
                f"{report_path}: pilot report persists NO data_identity block — a PASS "
                "unbound to the current panel/manifest/item materialization certifies "
                f"nothing (r4 judge-pilot-gates-missing); re-pilot (--family {family} "
                "--execute)"
            )
        data_mismatches = sorted(
            f"{k}: report={data.get(k)!r} != expected={data_exp[k]!r}"
            for k in data_exp
            if k not in data or data[k] != data_exp[k]
        )
        if data_mismatches:
            raise RuntimeError(
                f"{report_path}: persisted pilot DATA identity does not match the CURRENT "
                f"materialization — stale PASS refused (a pilot from an earlier panel / "
                f"manifest / item pool cannot license today's spend); re-pilot "
                f"(--family {family} --execute). Mismatched fields: {data_mismatches}"
            )
        floor = (
            min_effective_draws
            if min_effective_draws is not None
            else MIN_EFFECTIVE_BY_FAMILY[family]
        )
        eff = _effective_draws(rep, family)
        if eff < floor:
            raise RuntimeError(
                f"{report_path}: pilot family {family!r} effective draws {eff} < floor "
                f"{floor} — an under-powered pilot cannot resolve the {PARSE_FAIL_MAX} "
                "parse-fail gate (llm-judging.md rule 26 sizing); re-pilot with more draws"
            )
    return rep


def _spend_or_die(execute: bool) -> None:
    allowed, why = jl.spend_allowed(execute)
    assert allowed, (
        f"pilot dispatch needs real spend and refused: {why} — pass --execute and set "
        f"{jl.SPEND_ACK_ENV}=1 (a dry pilot certifies nothing, so there is no dry mode)"
    )


# ---------------------------------------------------------------------------
# axis family — judge_pilot_gate verbatim on the pooled axis items
# ---------------------------------------------------------------------------
def build_axis_arm(panel: list[dict], items_glob: str) -> tuple[list[tuple[str, str, str]], list]:
    """Pooled (item_id, question, answer) rows across characters with item files."""
    items: list[tuple[str, str, str]] = []
    present: list[str] = []
    for r in panel:
        name = r["name"]
        p = Path(items_glob.format(name=name))
        if not p.is_file():
            continue
        rows = c.read_jsonl(p)
        assert rows, f"{name}: {p} is empty"
        present.append(name)
        for row in rows:
            cid = str(row["conv_id"])
            items.append((f"axp_{name}_{cid}", str(row["question"]), str(row["answer"])))
    assert present, (
        f"no axis_items_<name>.jsonl matched {items_glob!r} — run "
        "issue2479_freeze_axis.py --emit-items first"
    )
    assert len({i for i, _, _ in items}) == len(items), "duplicate pooled axis pilot item ids"
    return items, present


def run_axis_pilot(
    panel_path: Path, items_glob: str, report_path: Path, work_dir: Path, *, execute: bool
) -> dict:
    """Dispatch the axis-family pilot through eval.judge_pilot.judge_pilot_gate."""
    import issue2479_freeze_axis as fz

    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    _spend_or_die(execute)
    panel = fz.load_panel(panel_path)
    # Computed BEFORE dispatch from the same files the arm pools, through the
    # same code path `require_pilot_pass` recomputes at every spend seam.
    data_identity = axis_data_identity(panel_path=panel_path, items_glob=items_glob)
    items, present = build_axis_arm(panel, items_glob)
    work_dir.mkdir(parents=True, exist_ok=True)
    rep = judge_pilot_gate(
        {"axis": items},
        jl.AI_LIKENESS_RUBRIC,
        max_tokens=jl.JUDGE_MAX_TOKENS,
        cache_dir=work_dir / "cache",  # PILOT-ONLY cache (rule 24(ii))
        save_raw_dir=work_dir,
        n_draws=jl.N_DRAWS,
        target_total_draws=TARGET_DRAWS,
        judge_model=jl.JUDGE_MODEL,
        temperature=jl.JUDGE_TEMPERATURE,
        parse_fail_threshold=PARSE_FAIL_MAX,
        threshold_base=jl.THRESHOLD_BASE_FORCE_BATCH,
        seed=0,
        # Declared wave: the axis judging wave (plan §9 ~19k Batch draws).
        wave_n_calls=19_000,
        wave_threshold_base=jl.THRESHOLD_BASE_FORCE_BATCH,
    )
    payload = {
        "issue": ISSUE,
        "family": "axis",
        **rep.to_json(),
        "instrument": axis_instrument_fingerprint(),
        "data_identity": data_identity,
        "characters_pooled": present,
        "n_pooled_items": len(items),
        "metadata": _metadata("scripts/issue2479_judge_pilots.py"),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    c.write_json(report_path, payload)
    print(
        f"[pilot] family=axis verdict={payload['verdict']} "
        f"n_pooled_items={len(items)} chars={len(present)} -> {report_path}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# ingen family — the gen module's own production judge seam, per-cell partials
# ---------------------------------------------------------------------------
def run_ingen_partial(
    raw_paths: list[Path], partial_out: Path, cache_root: Path, n_target: int, *, execute: bool
) -> dict:
    """Judge up to n_target pooled raw story rows under the CURRENT cell env.

    Runs the EXACT production instrument: the gen module's
    `_build_judge_request` / `_parse_judge_response` via `dispatch_calls`
    (dispatcher-decided route, transport-class failures re-driven sync once —
    the production re-drive shape), against a PILOT-ONLY cache/checkpoint.
    Persists per-draw OUTCOME rows only (no text).
    """
    import numpy as np

    import issue1345_gen_stories_paired as gp
    from explore_persona_space.llm.api_dispatch import (
        RESULT_RATE_LIMITED,
        RESULT_TRANSPORT,
        DispatchItem,
        dispatch_calls,
    )

    _spend_or_die(execute)
    rows: list[tuple[int, dict]] = []
    for fi, p in enumerate(raw_paths):
        assert p.is_file(), f"raw story file missing: {p}"
        for r in c.read_jsonl(p):
            rows.append((fi, r))
    assert rows, f"no raw story rows in {[str(p) for p in raw_paths]}"
    perm = np.random.default_rng(0).permutation(len(rows))
    take = [rows[i] for i in perm[: min(n_target, len(rows))]]
    items = [
        DispatchItem(
            item_id=f"igp{fi}_{r['conv_id']}",
            payload=(
                {"story": r["story"], "mode": r["mode"]}
                if r["mode"] == "op"
                else {"story": r["story"], "answer": r["answer"], "mode": r["mode"]}
            ),
        )
        for fi, r in take
    ]
    assert len({it.item_id for it in items}) == len(items), "duplicate pilot item ids"
    cache_root.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(
        dispatch_calls(
            items,
            model=c.JUDGE_MODEL,
            build_request=gp._build_judge_request,
            parse_response=gp._parse_judge_response,
            cache_dir=cache_root / "cache",  # PILOT-ONLY cache (rule 24(ii))
            checkpoint_dir=cache_root / "ckpt",
            force_path=None,  # the dispatcher decides, exactly as production
        )
    )
    redrive = [
        it
        for it in items
        if results[it.item_id].error
        and results[it.item_id].category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
    ]
    if redrive:
        print(f"[pilot] re-driving {len(redrive)} transport-class failures", flush=True)
        results.update(
            asyncio.run(
                dispatch_calls(
                    redrive,
                    model=c.JUDGE_MODEL,
                    build_request=gp._build_judge_request,
                    parse_response=gp._parse_judge_response,
                    cache_dir=cache_root / "cache",
                    checkpoint_dir=cache_root / "ckpt",
                    force_path="sync",
                )
            )
        )
    outcome_rows = [
        {
            "item_id": it.item_id,
            "error": bool(results[it.item_id].error),
            "category": results[it.item_id].category,
            "stop_reason": results[it.item_id].stop_reason,
        }
        for it in items
    ]
    payload = {
        "issue": ISSUE,
        "family": "ingen",
        "kind": "partial",
        "character": c.STORY_CHARACTER_NAME,
        "variant": os.environ.get("EPM_I1345_VARIANT"),
        "n_raw_rows": len(rows),
        "n_judged": len(items),
        "instrument": {
            **ingen_instrument_fingerprint(),
            # Name-DEPENDENT rendered-prompt shas stay per-partial (the system
            # prompt embeds this cell's character name at import) — the merge
            # excludes them from the cross-cell identity check.
            "judge_system_paired_sha256": hashlib.sha256(
                gp.JUDGE_SYSTEM_PAIRED.encode()
            ).hexdigest(),
            "judge_system_op_sha256": hashlib.sha256(gp.JUDGE_SYSTEM_OP.encode()).hexdigest(),
        },
        # Name-INDEPENDENT by construction (panel/manifest/gen-source bytes),
        # so the merge can require it identical across per-cell partials.
        "data_identity": ingen_data_identity(),
        "outcomes": outcome_rows,
        "metadata": _metadata("scripts/issue2479_judge_pilots.py"),
    }
    partial_out.parent.mkdir(parents=True, exist_ok=True)
    c.write_json(partial_out, payload)
    print(
        f"[pilot] family=ingen partial character={c.STORY_CHARACTER_NAME} "
        f"n_judged={len(items)} -> {partial_out}",
        flush=True,
    )
    return payload


def merge_ingen_partials(
    partial_paths: list[Path], report_path: Path, *, min_effective: int = INGEN_MIN_EFFECTIVE
) -> dict:
    """Merge per-cell partials into the ONE in-gen family arm + gate verdict.

    Same gate arithmetic as `judge_pilot_gate` (plan §7): zero
    stop_reason=="max_tokens" over answered draws; parse-fail < 2% of
    answered; a hollow arm (answered < min_effective) FAILs. Transport-class
    losses are excluded from every denominator (rule 24) but counted.
    """
    assert partial_paths, "no in-gen pilot partials to merge"
    _NAME_DEPENDENT = ("judge_system_paired_sha256", "judge_system_op_sha256")
    outcomes: list[dict] = []
    characters: list[str] = []
    instruments: set[str] = set()
    identities: set[str] = set()
    for p in partial_paths:
        part = json.loads(Path(p).read_text())
        assert part.get("family") == "ingen" and part.get("kind") == "partial", str(p)
        characters.append(str(part.get("character")))
        # Name-INDEPENDENT instrument identity: the full fingerprint minus the
        # per-cell rendered-prompt shas (the system prompt legitimately varies
        # per cell in the embedded name; builder/parser SOURCE shas do not).
        inst = {k: v for k, v in part["instrument"].items() if k not in _NAME_DEPENDENT}
        instruments.add(json.dumps(inst, sort_keys=True))
        identities.add(json.dumps(part.get("data_identity"), sort_keys=True))
        outcomes.extend(part["outcomes"])
    assert len(instruments) == 1, f"partials pilot DIFFERENT instruments: {sorted(instruments)}"
    merged_instrument = json.loads(next(iter(instruments)))
    assert len(identities) == 1, f"partials pilot DIFFERENT data identities: {sorted(identities)}"
    merged_identity = json.loads(next(iter(identities)))
    assert isinstance(merged_identity, dict), (
        "ingen partials carry no data_identity block — regenerate the partials on current "
        "code (r4 judge-pilot-gates-missing)"
    )

    def _is_transport(o: dict) -> bool:
        return bool(o["error"]) and o["category"] in (
            "rate_limited_exhausted",
            "transport_exhausted",
        )

    transport_lost = [o for o in outcomes if _is_transport(o)]
    answered = [o for o in outcomes if not _is_transport(o)]
    n_answered = len(answered)
    n_truncated = sum(1 for o in answered if o.get("stop_reason") == "max_tokens")
    n_parse_failed = sum(1 for o in answered if o["error"])
    parse_fail_rate = (n_parse_failed / n_answered) if n_answered else 1.0
    failures: list[str] = []
    if n_truncated:
        failures.append(
            f"ingen: {n_truncated}/{n_answered} answered draws hit stop_reason==max_tokens "
            "(gate requires zero — raise max_tokens generously and re-pilot)"
        )
    if parse_fail_rate >= PARSE_FAIL_MAX:
        failures.append(
            f"ingen: parse-fail rate {parse_fail_rate:.4f} >= {PARSE_FAIL_MAX} "
            f"({n_parse_failed}/{n_answered} answered draws)"
        )
    if n_answered < min_effective:
        failures.append(
            f"ingen: hollow arm — {n_answered} answered draws < floor {min_effective} "
            "(raise the smoke story count / add cells and re-pilot)"
        )
    passed = not failures
    payload = {
        "issue": ISSUE,
        "family": "ingen",
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "failures": failures,
        "warnings": (
            [f"ingen: {len(transport_lost)} transport-class draws lost (excluded from gates)"]
            if transport_lost
            else []
        ),
        "arms": {
            "ingen": {
                "n_items": len(outcomes),
                "n_draws_per_item": 1,
                "effective_draws": n_answered,
                "n_truncated": n_truncated,
                "n_parse_failed": n_parse_failed,
                "parse_fail_rate": round(parse_fail_rate, 6),
                "n_transport_lost": len(transport_lost),
            }
        },
        "parse_fail_threshold": PARSE_FAIL_MAX,
        "min_effective_draws": min_effective,
        # The merged family report PERSISTS the name-independent instrument +
        # data identity the partials proved identical, so `require_pilot_pass`
        # can bind the PASS to the exact production instrument AND the exact
        # materialization (r2 codex judge-pilot-gates; r4 data extension).
        "instrument": merged_instrument,
        "data_identity": merged_identity,
        "characters_pooled": characters,
        "partials": [str(p) for p in partial_paths],
        "metadata": _metadata("scripts/issue2479_judge_pilots.py"),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    c.write_json(report_path, payload)
    print(
        f"[pilot] family=ingen verdict={payload['verdict']} answered={n_answered} "
        f"truncated={n_truncated} parse_failed={n_parse_failed} -> {report_path}",
        flush=True,
    )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=FAMILIES)
    ap.add_argument("--require-pass", action="store_true")
    ap.add_argument("--report", type=Path, default=None, help="pilot report path (family default)")
    ap.add_argument("--panel", type=Path, default=None, help="axis: panel.json (registry default)")
    ap.add_argument(
        "--items-glob", default=None, help="axis: axis_items_{name}.jsonl path template"
    )
    ap.add_argument("--work-dir", type=Path, default=None, help="pilot cache/save_raw root")
    ap.add_argument("--raw", type=Path, action="append", default=[], help="ingen: raw story file")
    ap.add_argument("--partial-out", type=Path, default=None, help="ingen: per-cell partial path")
    ap.add_argument("--merge", type=Path, nargs="+", default=None, help="ingen: partials to merge")
    ap.add_argument("--n-target", type=int, default=TARGET_DRAWS)
    ap.add_argument("--min-effective", type=int, default=INGEN_MIN_EFFECTIVE)
    ap.add_argument(
        "--execute",
        action="store_true",
        help=f"attempt REAL spend; additionally requires {jl.SPEND_ACK_ENV}=1",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Deferred imports on the real code paths, named explicitly (#1689).
        import issue1345_gen_stories_paired as gp  # noqa: F401
        from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: F401
        from explore_persona_space.llm.api_dispatch import (  # noqa: F401
            RESULT_RATE_LIMITED,
            RESULT_TRANSPORT,
            DispatchItem,
            dispatch_calls,
        )
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )

        print("import-ok: issue2479_judge_pilots", flush=True)
        return

    assert args.family, "--family is required (axis | ingen)"
    default_rel = PILOT_AXIS_REL if args.family == "axis" else PILOT_INGEN_REL
    report = args.report or (_REPO_ROOT / default_rel)

    if args.require_pass:
        expected_data = None
        if args.family == "axis" and args.items_glob:
            # The wrapper passes its OWN items glob so the gate recomputes the
            # expected identity against exactly the files it will dispatch on.
            expected_data = axis_data_identity(panel_path=args.panel, items_glob=args.items_glob)
        require_pilot_pass(report, family=args.family, expected_data=expected_data)
        print(f"[pilot] require-pass OK: family={args.family} report={report}", flush=True)
        return

    if args.family == "axis":
        import issue2479_freeze_axis as fz

        panel_path = args.panel or (_REPO_ROOT / fz.PANEL_REL)
        assert args.items_glob, "--family axis requires --items-glob"
        work_dir = args.work_dir or (_REPO_ROOT / "data/issue_2479/pilot_axis")
        run_axis_pilot(panel_path, args.items_glob, report, work_dir, execute=args.execute)
        return

    # ingen: either a per-cell partial (under the cell env) or the merge.
    if args.merge:
        merge_ingen_partials(list(args.merge), report, min_effective=args.min_effective)
        return
    assert args.raw and args.partial_out, (
        "--family ingen requires either --merge <partials...> or --raw <file>... with "
        "--partial-out <path>"
    )
    assert os.environ.get("EPM_STORY_CHARACTER_NAME"), (
        "ingen partial must run under the CELL env (EPM_STORY_CHARACTER_NAME / "
        "EPM_I1345_PERSONA_DESC / EPM_I1345_VARIANT) — the judge system prompt embeds the "
        "character name at import"
    )
    cache_root = args.work_dir or (_REPO_ROOT / "data/issue_2479/pilot_ingen")
    run_ingen_partial(
        list(args.raw), args.partial_out, cache_root, args.n_target, execute=args.execute
    )


if __name__ == "__main__":
    main()
