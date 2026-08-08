#!/usr/bin/env python
"""#1947 9a-ter free-analysis (VM, 0 GPU): intrusion-robust recount, ALL rungs.

The analyzer's round-1 audit found CJK intrusion in the temp-1.0 judged ladder
pools (232/3,400 verdict-rung completions, 6.8%) and recounted the VERDICT
rungs only (`analyzer_round1/cjk_recount_verdict_rungs.json`). This round
extends the recount to ALL 15 rungs x 34 content cells, from the committed
judge artifacts (`analysis/judge/<beh>/<instrument>/judge_raw_<slug>-r<step>.json`)
reduced with the PRODUCTION reduce (`graded_judge.judge_result_from_save_raw`
— drop-never-coerce, mean over kept draws) joined to the staged ladder
rollout text (`<out_root>/ladders/<slug>/ladder_rollouts.json`; Hub-staged on
miss, the battery's `_stage_ladder` path arithmetic).

The analyzer's detection RULE is not recorded in the digests, so it is
CALIBRATED here: candidate CJK char-class rules are scored against the
committed per-slug scan counts (`cjk_scan_ladders.json` `intruded_all`
N/1500) and the rule reproducing EVERY slug's count exactly is adopted (and
reported); the verdict-rung rows are additionally validated against
`cjk_recount_verdict_rungs.json` and every rung's raw rate against the
committed `judged_<slug>.json` rates_by_step.

Three conventions per (cell, rung): raw rate / intruded-zeroed
((k_pos - fired_intr)/n_scored) / intruded-excluded
((k_pos - fired_intr)/(n_scored - intr)).

Content hygiene: rollout text is processed in-script only — outputs carry
counts, flags, and rates, never row text.

Outputs: eval_results/issue_1947/analysis/intrusion_recount_all_rungs.json
(+ per-unit resume JSONL intrusion_recount_units.jsonl) and
figures/issue_1947/intrusion_recount_ladders.png.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1947_cells as cells  # noqa: E402

logger = logging.getLogger("issue1947.intrusion_recount")

# Candidate detection rules (calibrated against the analyzer's committed
# per-slug scan counts — the digests carry counts, not the rule).
CANDIDATE_RULES: list[tuple[str, str, str]] = [
    ("cjk_unified", r"[一-鿿]", "CJK Unified Ideographs U+4E00-U+9FFF"),
    (
        "cjk_unified_ext",
        r"[㐀-䶿一-鿿豈-﫿]",
        "CJK Unified + Ext-A U+3400-U+4DBF + Compatibility U+F900-U+FAFF",
    ),
    (
        "cjk_jp",
        r"[぀-ヿ㐀-䶿一-鿿豈-﫿]",
        "CJK Unified + Ext-A + Compatibility + Hiragana/Katakana U+3040-U+30FF",
    ),
    (
        "cjk_jp_kr",
        r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]",
        "above + Hangul syllables U+AC00-U+D7AF",
    ),
    (
        "cjk_punct",
        r"[　-〿㐀-䶿一-鿿豈-﫿]",
        "CJK Unified + Ext-A + Compatibility + CJK punctuation U+3000-U+303F",
    ),
    (
        "east_asian_wide",
        r"[⺀-鿿가-힯豈-﫿＀-￯]",
        "broad East-Asian U+2E80-U+9FFF + Hangul + Compatibility + fullwidth forms",
    ),
]


def _meta() -> dict:
    return {"issue": cells.ISSUE, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stage_ladder(out_root: Path, slug: str) -> Path:
    """Staged-copy-first ladder fetch (the battery's `_stage_ladder` path
    arithmetic); Hub-staged on miss via the canonical helper."""
    local = out_root / "ladders" / slug / "ladder_rollouts.json"
    if not local.exists():
        from explore_persona_space.orchestrate import hub

        logger.info("[recount] staging ladder for %s from HF (local miss)", slug)
        hub.stage_hub_file(
            "superkaiba1/explore-persona-space-data",  # issue1947_battery.HF_DATA_REPO
            f"{cells.DATA_PREFIX}/raw_completions/ladders/{slug}/ladder_rollouts.json",
            local,
            repo_type="dataset",
        )
    return local


def item_texts(payload: dict, step_s: str) -> list[tuple[str, str, str]]:
    """(item_id, question, completion) triples in the battery's `_judge_rung`
    item order — the join key into the committed judge_raw files."""
    qs = payload["questions"]
    comps = payload["rungs"][step_s]["completions"]
    assert len(comps) == len(qs), (payload["slug"], step_s, len(comps), len(qs))
    tag = f"{payload['slug']}-r{step_s}"
    return [
        (f"{tag}-q{qi:03d}-c{ci:03d}", q, comp)
        for qi, q in enumerate(qs)
        for ci, comp in enumerate(comps[qi])
    ]


def calibrate_rule(
    slugs: list[str], out_root: Path, scan_ref: dict
) -> tuple[str, re.Pattern, dict]:
    """Pick the candidate rule that reproduces EVERY slug's committed
    `intruded_all` scan count exactly; report the calibration table."""
    compiled = [(name, re.compile(pat), desc) for name, pat, desc in CANDIDATE_RULES]
    per_rule_diff: dict[str, int] = {name: 0 for name, _, _ in compiled}
    per_rule_matched: dict[str, int] = {name: 0 for name, _, _ in compiled}
    for slug in slugs:
        payload = _read_json(stage_ladder(out_root, slug))
        texts = [comp for step_s in payload["rungs"] for _, _, comp in item_texts(payload, step_s)]
        ref_n = int(scan_ref[slug]["intruded_all"].split("/")[0])
        for name, rx, _ in compiled:
            n = sum(1 for t in texts if rx.search(t))
            per_rule_diff[name] += abs(n - ref_n)
            per_rule_matched[name] += int(n == ref_n)
    table = {
        name: {
            "slugs_matched_exactly": per_rule_matched[name],
            "total_abs_count_diff": per_rule_diff[name],
            "char_class": desc,
        }
        for name, _, desc in compiled
    }
    exact = [name for name, _, _ in compiled if per_rule_matched[name] == len(slugs)]
    if exact:
        chosen = exact[0]
    else:  # no exact rule — take the closest and flag LOUD (validated downstream)
        chosen = min(per_rule_diff, key=lambda k: per_rule_diff[k])
        logger.warning(
            "[recount] NO candidate rule matches every slug exactly — closest: %s (%s)",
            chosen,
            table[chosen],
        )
    rx = next(r for name, r, _ in compiled if name == chosen)
    return chosen, rx, {"chosen": chosen, "exact_match_all_slugs": bool(exact), "table": table}


def recount_unit(
    payload: dict,
    step_s: str,
    judge_root: Path,
    instrument: str,
    threshold: float,
    rx: re.Pattern,
) -> dict:
    """One (cell, rung): production reduce of the committed judge_raw + the
    three-convention rates over the rule's intrusion flags."""
    from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

    slug = payload["slug"]
    items = item_texts(payload, step_s)
    save_raw = judge_root / instrument / f"judge_raw_{slug}-r{step_s}.json"
    result = judge_result_from_save_raw(save_raw, items)
    rows = [
        {
            "score": result.scores.get(iid),
            "intruded": bool(rx.search(comp)),
        }
        for iid, _q, comp in items
    ]
    scored = [r for r in rows if r["score"] is not None]
    n_scored = len(scored)
    k_pos = sum(1 for r in scored if r["score"] > threshold)
    intr_scored = sum(1 for r in scored if r["intruded"])
    intr_all = sum(1 for r in rows if r["intruded"])
    fired_intr = sum(1 for r in scored if r["intruded"] and r["score"] > threshold)
    rate_raw = (k_pos / n_scored) if n_scored else None
    rate_zeroed = ((k_pos - fired_intr) / n_scored) if n_scored else None
    rate_excl = (
        ((k_pos - fired_intr) / (n_scored - intr_scored)) if n_scored > intr_scored else None
    )
    return {
        "slug": slug,
        "rung": int(step_s),
        "n_items": len(items),
        "n_scored": n_scored,
        "k_positive": k_pos,
        "intr": intr_scored,
        "intr_all_items": intr_all,
        "fired_intr": fired_intr,
        "rate_raw": rate_raw,
        "rate_zeroed": rate_zeroed,
        "rate_excl": rate_excl,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="#1947 all-rungs intrusion-robust recount")
    p.add_argument("--out-root", default=str(REPO_ROOT / "data/issue_1947/battery_stage"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "eval_results/issue_1947/analysis"))
    p.add_argument("--fig-dir", default=str(REPO_ROOT / "figures/issue_1947"))
    p.add_argument("--slugs", default="", help="comma-separated slug filter")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--import-check", action="store_true")
    args = p.parse_args(argv)
    if args.import_check:
        from explore_persona_space.artifacts.organisms import BEHAVIORS
        from explore_persona_space.eval.graded_judge import judge_result_from_save_raw
        from explore_persona_space.orchestrate import hub

        names = [judge_result_from_save_raw, hub.stage_hub_file, BEHAVIORS]
        print(f"[import-check] OK ({len(names)} symbols resolved)")
        return 0
    from explore_persona_space.artifacts.organisms import BEHAVIORS

    out_root, out_dir, fig_dir = Path(args.out_root), Path(args.out_dir), Path(args.fig_dir)
    an_dir = out_dir / "analyzer_round1"
    scan_ref = _read_json(an_dir / "cjk_scan_ladders.json")
    verdict_ref = _read_json(an_dir / "cjk_recount_verdict_rungs.json")
    manifest = _read_json(out_dir / "verdict_manifest.json")
    slugs = sorted(manifest["content"])
    if args.slugs:
        keep = {s for s in args.slugs.split(",") if s}
        slugs = [s for s in slugs if s in keep]

    rule_name, rx, calibration = calibrate_rule(slugs, out_root, scan_ref)
    print(
        f"[recount] detection rule: {rule_name} "
        f"(exact on all slugs: {calibration['exact_match_all_slugs']})",
        flush=True,
    )
    regime = {"version": 1, "rule": rule_name}

    units_path = out_dir / "intrusion_recount_units.jsonl"
    done: dict[tuple[str, int], dict] = {}
    if units_path.exists():
        with units_path.open(encoding="utf-8") as fh:  # never splitlines() on JSONL
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("regime") == regime:
                    done[(row["record"]["slug"], row["record"]["rung"])] = row["record"]

    per_slug: dict[str, dict] = {}
    validations: dict[str, dict] = {}
    n_rate_mismatch = 0
    k = 0
    for slug in slugs:
        payload = _read_json(stage_ladder(out_root, slug))
        judged = _read_json(out_dir / "judge" / f"judged_{slug}.json")
        beh_key = slug.split("-")[0]
        judge_root = out_dir / "judge" / beh_key
        instrument = judged["instrument"]
        threshold = float(BEHAVIORS[judged["behavior"]].threshold)
        steps = sorted(payload["rungs"], key=int)
        total = len(slugs) * len(steps)
        rungs: dict[str, dict] = {}
        for step_s in steps:
            k += 1
            t0 = time.time()
            key = (slug, int(step_s))
            if key in done:
                rec = done[key]
            else:
                rec = recount_unit(payload, step_s, judge_root, instrument, threshold, rx)
                # cross-check the raw rate against the committed judged rate
                committed = judged["rates_by_step"].get(step_s)
                rec["raw_rate_matches_committed"] = (
                    committed is not None
                    and rec["rate_raw"] is not None
                    and abs(rec["rate_raw"] - committed) < 1e-9
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                with units_path.open("a", encoding="utf-8") as fh:  # atomic 1-line append
                    fh.write(json.dumps({"regime": regime, "record": rec, **_meta()}) + "\n")
                    fh.flush()
            if not rec.get("raw_rate_matches_committed", False):
                n_rate_mismatch += 1
            rungs[step_s] = rec
            print(
                f"[recount] unit {k}/{total} {slug}_r{step_s} elapsed={time.time() - t0:.2f}s",
                flush=True,
            )
        # per-slug validations vs the analyzer digests
        scan_all_ref = int(scan_ref[slug]["intruded_all"].split("/")[0])
        scan_all_mine = sum(r["intr_all_items"] for r in rungs.values())
        vref = verdict_ref[slug]
        vrung = rungs.get(str(vref["rung"]))

        def _close(a, b, tol: float = 5.1e-4) -> bool:
            return a is not None and b is not None and abs(a - b) < tol

        verdict_ok = bool(
            vrung
            and vrung["n_scored"] == vref["n"]
            and vrung["intr"] == vref["intr"]
            and vrung["fired_intr"] == vref["fired_intr"]
            and _close(vrung["rate_raw"], vref["rate"])
            and _close(vrung["rate_zeroed"], vref["rate_zeroed"])
            and _close(vrung["rate_excl"], vref["rate_excl"])
        )
        validations[slug] = {
            "scan_intruded_all": {"ref": scan_all_ref, "mine": scan_all_mine},
            "scan_intruded_all_ok": scan_all_mine == scan_all_ref,
            "verdict_rung": vref["rung"],
            "verdict_row_ok": verdict_ok,
        }
        per_slug[slug] = {
            "behavior": judged["behavior"],
            "instrument": instrument,
            "threshold": threshold,
            "verdict_rung": vref["rung"],
            "rates_by_step": rungs,
        }

    n_scan_ok = sum(1 for v in validations.values() if v["scan_intruded_all_ok"])
    n_verdict_ok = sum(1 for v in validations.values() if v["verdict_row_ok"])
    summary = {
        "detection_rule": {
            "name": rule_name,
            "char_class": calibration["table"][rule_name]["char_class"],
            "calibration": calibration,
        },
        "conventions": {
            "rate_raw": "k_positive / n_scored (the committed battery rate)",
            "rate_zeroed": "(k_positive - fired_intr) / n_scored — intruded items count as not-fired",
            "rate_excl": "(k_positive - fired_intr) / (n_scored - intr) — intruded items dropped",
        },
        "validations": {
            "per_slug": validations,
            "scan_intruded_all_ok": f"{n_scan_ok}/{len(slugs)}",
            "verdict_row_ok": f"{n_verdict_ok}/{len(slugs)}",
            "raw_rate_mismatches": n_rate_mismatch,
        },
        "cells": per_slug,
        **_meta(),
    }
    _atomic_json(out_dir / "intrusion_recount_all_rungs.json", summary)
    if not args.no_figure:
        make_figure(fig_dir / "intrusion_recount_ladders.png", per_slug)
    print(
        f"[recount] done: {len(per_slug)} cells; scan_ok {n_scan_ok}/{len(slugs)}; "
        f"verdict_ok {n_verdict_ok}/{len(slugs)}; raw-rate mismatches {n_rate_mismatch}",
        flush=True,
    )
    return 0


def make_figure(fig_path: Path, per_slug: dict) -> None:
    """Per-behavior ladder curves under the three conventions: thin per-cell
    lines + thick per-convention mean, one panel per behavior."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    pal = paper_palette(3)
    conv_color = {"rate_raw": pal[0], "rate_zeroed": pal[1], "rate_excl": pal[2]}
    behaviors = sorted({v["behavior"] for v in per_slug.values()})
    fig, axes = plt.subplots(1, len(behaviors), figsize=(5.2 * len(behaviors), 4.2), squeeze=False)
    for col, beh in enumerate(behaviors):
        ax = axes[0][col]
        cells_b = {s: v for s, v in per_slug.items() if v["behavior"] == beh}
        steps = sorted({int(r) for v in cells_b.values() for r in v["rates_by_step"]})
        for conv, color in conv_color.items():
            curves = []
            for v in cells_b.values():
                ys = [v["rates_by_step"].get(str(s), {}).get(conv) for s in steps]
                if any(y is None for y in ys):
                    continue
                curves.append(ys)
                ax.plot(steps, ys, color=color, lw=0.5, alpha=0.18, zorder=1)
            if curves:
                ax.plot(
                    steps,
                    np.mean(np.asarray(curves, dtype=float), axis=0),
                    color=color,
                    lw=2.2,
                    zorder=2,
                    label=f"{conv} (n={len(curves)} cells)",
                )
        ax.set_title(f"{beh} — ladder rates under 3 intrusion conventions")
        ax.set_xlabel("ladder rung (optimizer step)")
        ax.set_ylabel("judged positive rate")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle(
        "#1947 intrusion-robust recount, all 15 rungs x 34 content cells "
        "(thin: per-cell; thick: behavior mean)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit — the PyGILState_Release atexit gotcha
