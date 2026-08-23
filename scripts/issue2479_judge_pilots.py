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
# Opt-in env consumed by `jl.run_leg` (and exported by the P3 wrapper): a real
# axis-leg spend refuses without a persisted axis-family pilot PASS at this path.
REQUIRE_AXIS_PILOT_ENV = "EPM_I2479_REQUIRE_AXIS_PILOT_PASS"


def _metadata(script: str) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        "script": script,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **as_metadata_dict(git_provenance()),
    }


def require_pilot_pass(report_path: Path, family: str | None = None) -> dict:
    """Load a persisted pilot report; RAISE unless it is a PASS (returns it)."""
    report_path = Path(report_path)
    if not report_path.is_file():
        raise RuntimeError(
            f"rule-26 pilot gate report missing: {report_path} — run "
            f"scripts/issue2479_judge_pilots.py --family {family or '<family>'} --execute "
            "before the production wave (plan §7)"
        )
    rep = json.loads(report_path.read_text())
    if family is not None and rep.get("family") != family:
        raise RuntimeError(
            f"{report_path}: pilot family {rep.get('family')!r} != required {family!r}"
        )
    if rep.get("verdict") != "PASS" or not rep.get("passed"):
        raise RuntimeError(
            f"rule-26 pilot gate {rep.get('family')!r} verdict={rep.get('verdict')!r} — "
            f"production dispatch refused (failures: {rep.get('failures')})"
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
    panel: list[dict], items_glob: str, report_path: Path, work_dir: Path, *, execute: bool
) -> dict:
    """Dispatch the axis-family pilot through eval.judge_pilot.judge_pilot_gate."""
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    _spend_or_die(execute)
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
        "instrument": {
            "judge_model": jl.JUDGE_MODEL,
            "n_draws": jl.N_DRAWS,
            "temperature": jl.JUDGE_TEMPERATURE,
            "max_tokens": jl.JUDGE_MAX_TOKENS,
            "threshold_base": jl.THRESHOLD_BASE_FORCE_BATCH,
            "rubric_sha256": hashlib.sha256(jl.AI_LIKENESS_RUBRIC.encode()).hexdigest(),
        },
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
            "judge_model": c.JUDGE_MODEL,
            "max_tokens": c.JUDGE_MAX_TOKENS,
            "temperature": 0.0,
            "judge_system_paired_sha256": hashlib.sha256(
                gp.JUDGE_SYSTEM_PAIRED.encode()
            ).hexdigest(),
            "judge_system_op_sha256": hashlib.sha256(gp.JUDGE_SYSTEM_OP.encode()).hexdigest(),
        },
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
    outcomes: list[dict] = []
    characters: list[str] = []
    instruments: set[str] = set()
    for p in partial_paths:
        part = json.loads(Path(p).read_text())
        assert part.get("family") == "ingen" and part.get("kind") == "partial", str(p)
        characters.append(str(part.get("character")))
        # Name-independent instrument identity: model + budget + temp (the
        # system prompt legitimately varies per cell in the embedded name).
        inst = part["instrument"]
        instruments.add(f"{inst['judge_model']}|{inst['max_tokens']}|{inst['temperature']}")
        outcomes.extend(part["outcomes"])
    assert len(instruments) == 1, f"partials pilot DIFFERENT instruments: {sorted(instruments)}"

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
        require_pilot_pass(report, family=args.family)
        print(f"[pilot] require-pass OK: family={args.family} report={report}", flush=True)
        return

    if args.family == "axis":
        import issue2479_freeze_axis as fz

        panel = fz.load_panel(args.panel or (_REPO_ROOT / fz.PANEL_REL))
        assert args.items_glob, "--family axis requires --items-glob"
        work_dir = args.work_dir or (_REPO_ROOT / "data/issue_2479/pilot_axis")
        run_axis_pilot(panel, args.items_glob, report, work_dir, execute=args.execute)
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
