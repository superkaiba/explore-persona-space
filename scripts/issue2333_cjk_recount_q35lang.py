#!/usr/bin/env python
"""Issue #2333 q35lang round Step-3.7 language-intrusion recount.

The plan (v9 §6 robustness row) registers the CJK-intrusion recount for the
`q35_language_snowball` round: target languages are es/en/fr, so CJK is never
a target and the scan stays valid. This script mirrors
`issue2333_cjk_recount.py` (the parent-legs recount) for the q35lang cell set,
but instead of re-implementing the lattice it REBUILDS the per-pair cells with
intruded draws (a) EXCLUDED and (b) their judge contrast ZEROED (delta := 0,
the parent's no-behavioral-movement convention), then re-runs the SHIPPED
`issue2333_analysis._stats_q35lang` on each recounted cell set — so every
registered read (net separation, D3_net, R3_net, both lattices, carrier
companions, continuation companions, control health) differs from the shipped
stats ONLY in the intrusion handling.

Recount boundary (stated, not silent): the vendored #2329 banked companion
rows are PARENT-wave scores and are held fixed; the same-wave quantities
(fresh grid, fresh ce control, and the anchors-rejudge floor/ceiling deltas)
ARE recounted. Both modes use the excluded-anchor normalization (zeroing an
anchor has no "no movement" semantics — parent convention).

Validation: a "shipped" mode rebuild (no intrusion handling) must reproduce
the committed f_cells/null_cells/ce_cells f_beh values exactly before either
recount mode is trusted.

Writes eval_results/issue_2333/q35_language_snowball/f_metrics/cjk_recount.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy import

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "issue2333_analysis", REPO / "scripts" / "issue2333_analysis.py"
)
A = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(A)

J33 = A.J33
A62 = A.A62
C = A.C

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")

JUDGE_INPUTS = REPO / "data/issue_2333/judge_inputs/issue2333_snowball/q35_language_snowball"
SHIPPED_DIR = REPO / "eval_results/issue_2333/q35_language_snowball/f_metrics"
SCORES_DIR = REPO / "eval_results/issue_2333/q35_language_snowball/judge/scores"


def _intruded(text: str | None) -> bool:
    return bool(CJK_RE.search(text or ""))


def _f_from_rows_mode(
    rows: list[dict],
    tag: str,
    scores: dict,
    anchor: dict,
    mode: str,  # "shipped" | "excluded" | "zeroed"
) -> dict:
    """`issue2333_analysis._f_from_rows` with the parent recount's intrusion
    handling injected (same coherence-key tag aliasing: continuation "n" rows
    share the grid coherence score)."""
    deltas: list[float] = []
    n_coherent = 0
    for row in rows:
        intr = _intruded(row.get("response_text"))
        if intr and mode == "excluded":
            continue
        coh = scores.get(A._item_id("c", J33.coherence_key(tag if tag != "n" else "g", row)))
        if coh is None or coh <= A.COHERENCE_THRESHOLD:
            continue
        n_coherent += 1
        sa = scores.get(A._item_id(tag, J33.behavior_key(tag, row, "a")))
        sb = scores.get(A._item_id(tag, J33.behavior_key(tag, row, "b")))
        if sa is None or sb is None:
            continue
        deltas.append(0.0 if (intr and mode == "zeroed") else (sb - sa) / 100.0)
    dp = A._mean(deltas)
    fl, ce_ = anchor.get("floor"), anchor.get("ceiling")
    f_beh = None
    if dp is not None and fl is not None and ce_ is not None and abs(ce_ - fl) > 1e-9:
        f_beh = (dp - fl) / (ce_ - fl)
    return {"f_beh": f_beh, "n_scored": len(deltas), "n_coherent": n_coherent}


def _rebuild_cells(
    mode: str,
    grid_rows: list[dict],
    ce_rows: list[dict],
    anchors: dict[str, dict],
    scores: dict,
    shipped: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Recounted f_cells / null_cells / ce_cells, starting from the SHIPPED
    rows (metadata + f_act verbatim — F_act is activation-side, not judged)
    and replacing f_beh, f_beh_continuation, and separation."""
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in grid_rows:
        by_cell[(r["pair_id"], r["arm_slug"], r["variant"])].append(r)
    out: dict[str, list[dict]] = {}
    for name in ("f_cells", "null_cells"):
        rows_out = []
        for rec in shipped[name]:
            rows = by_cell[(rec["pair_id"], rec["arm_slug"], rec["variant"])]
            assert rows, (name, rec["pair_id"], rec["arm_slug"])
            anchor = anchors[rec["pair_id"]]
            new = dict(rec)
            new["separation"] = anchor.get("separation")
            new.update(_f_from_rows_mode(rows, "g", scores, anchor, mode))
            if rec["kind"] == "prefill":
                cont = _f_from_rows_mode(rows, "n", scores, anchor, mode)
                new["f_beh_continuation"] = cont["f_beh"]
                new["n_scored_continuation"] = cont["n_scored"]
            rows_out.append(new)
        out[name] = rows_out
    by_ce: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in ce_rows:
        by_ce[(r["pair_id"], r["variant"])].append(r)
    rows_out = []
    for rec in shipped["ce_cells"]:
        rows = by_ce[(rec["pair_id"], rec["variant"])]
        assert rows, ("ce_cells", rec["pair_id"], rec["variant"])
        anchor = anchors[rec["pair_id"]]
        new = dict(rec)
        new["separation"] = anchor.get("separation")
        new.update(_f_from_rows_mode(rows, "e", scores, anchor, mode))
        rows_out.append(new)
    out["ce_cells"] = rows_out
    return out


def _run_stats(cells: dict[str, list[dict]], workdir: Path) -> dict:
    for name, rows in cells.items():
        A62._write_jsonl_atomic(workdir / f"{name}.jsonl", rows)
    ns = argparse.Namespace(model_tag="q35", out_dir=workdir)
    rc = A._stats_q35lang(ns)
    assert rc == 0, rc
    return json.loads((workdir / "stats.json").read_text())


def _headline(stats: dict) -> dict:
    s1 = stats["per_set"]["s1"]
    arms = {}
    for slug, rec in s1["arms"].items():
        arms[slug] = {
            "n_pairs": rec.get("n_pairs"),
            "diff_mean": rec.get("diff_mean"),
            "separates": rec.get("separates"),
            "ratio_net_samewave": (rec.get("recovery_net_samewave") or {}).get("ratio_net"),
            "ratio_net_ci": (rec.get("recovery_net_samewave") or {}).get("ratio_net_ci"),
            "d3_net_ci": (rec.get("recovery_net_samewave") or {}).get("d3_net_ci"),
        }
    return {
        "control_health_samewave_passed": stats["preconditions"]["control_health_samewave"][
            "passed"
        ],
        "prefill3_verdicts": s1["prefill3_verdicts"],
        "arms": arms,
    }


def main() -> int:
    shipped = {
        name: list(A62._iter_jsonl(SHIPPED_DIR / f"{name}.jsonl"))
        for name in ("f_cells", "null_cells", "ce_cells")
    }
    shipped_stats = json.loads((SHIPPED_DIR / "stats.json").read_text())
    scores = A._load_scores(SCORES_DIR, ("grid", "anchors"))
    s1_pairs, s2_pairs = J33.build_pair_universe("q35lang")
    assert not s2_pairs
    ctx_ids = {cid for p in s1_pairs for cid in (p.a, p.b)}
    anchor_rows = J33.load_anchor_rows_2333(JUDGE_INPUTS / "anchors", "q35lang", ctx_ids)
    grid_rows = J33.load_grid_rows(JUDGE_INPUTS / "rollouts", cell_set="q35lang")
    ce_rows = J33.load_ce_rows(JUDGE_INPUTS / "rollouts", cell_set="q35lang")

    # ── per-pool intrusion tallies (pure counting; no text leaves this run) ──
    tallies: dict[str, dict] = defaultdict(lambda: {"n": 0, "intr": 0, "intr_coh": 0})
    for r in grid_rows:
        t = tallies[f"{r['arm_slug']}::{r['variant']}"]
        t["n"] += 1
        if _intruded(r.get("response_text")):
            t["intr"] += 1
            coh = scores.get(A._item_id("c", J33.coherence_key("g", r)))
            if coh is not None and coh > A.COHERENCE_THRESHOLD:
                t["intr_coh"] += 1
    ce_intr = sum(_intruded(r.get("response_text")) for r in ce_rows)
    anch_intr = sum(_intruded(r.get("response_text")) for r in anchor_rows)

    anchors_shipped = A._fresh_anchor_deltas(s1_pairs, anchor_rows, scores)
    anchor_rows_x = [r for r in anchor_rows if not _intruded(r.get("response_text"))]
    anchors_x = A._fresh_anchor_deltas(s1_pairs, anchor_rows_x, scores)

    out: dict = {
        "cell_set": "q35lang",
        "regex": CJK_RE.pattern,
        "grid_intrusion": {
            "total": sum(t["n"] for t in tallies.values()),
            "intruded": sum(t["intr"] for t in tallies.values()),
            "intruded_coherent": sum(t["intr_coh"] for t in tallies.values()),
            "per_pool": dict(sorted(tallies.items())),
        },
        "ce_pool_intrusion": {"n": len(ce_rows), "intruded": ce_intr},
        "anchor_pool_intrusion": {"n": len(anchor_rows), "intruded": anch_intr},
        "boundary": "vendored #2329 banked companion rows held fixed (parent waves); "
        "same-wave grid/ce/anchor quantities recounted; both modes use "
        "excluded-anchor normalization",
        "recounts": {},
    }

    with tempfile.TemporaryDirectory(prefix="i2333-q35lang-cjk-") as td:
        tmp = Path(td)
        # Validation leg: the shipped-mode rebuild must reproduce committed f_beh.
        cells_v = _rebuild_cells("shipped", grid_rows, ce_rows, anchors_shipped, scores, shipped)
        n_checked = n_mismatch = 0
        for name in ("f_cells", "null_cells", "ce_cells"):
            for old, new in zip(shipped[name], cells_v[name], strict=True):
                n_checked += 1
                o, n = old["f_beh"], new["f_beh"]
                if (o is None) != (n is None) or (o is not None and abs(o - n) > 1e-12):
                    n_mismatch += 1
        assert n_mismatch == 0, f"shipped-mode rebuild mismatches: {n_mismatch}/{n_checked}"
        out["validation"] = {"cells_checked": n_checked, "mismatches": 0}

        for mode in ("excluded", "zeroed"):
            cells = _rebuild_cells(mode, grid_rows, ce_rows, anchors_x, scores, shipped)
            wd = tmp / mode
            wd.mkdir()
            out["recounts"][mode] = _headline(_run_stats(cells, wd))
        shutil.rmtree(tmp, ignore_errors=True)

    out["shipped"] = _headline(shipped_stats)
    flips = []
    for mode in ("excluded", "zeroed"):
        for scheme in ("med", "bstart"):
            for field in ("primary_final", "ratio_final", "mechanism_final"):
                a = out["shipped"]["prefill3_verdicts"][scheme][field]
                b = out["recounts"][mode]["prefill3_verdicts"][scheme][field]
                if a != b:
                    flips.append(
                        {"mode": mode, "scheme": scheme, "field": field, "from": a, "to": b}
                    )
        for slug, rec in out["shipped"]["arms"].items():
            b = out["recounts"][mode]["arms"][slug]
            if rec["separates"] != b["separates"]:
                flips.append(
                    {
                        "mode": mode,
                        "arm": slug,
                        "field": "separates",
                        "from": rec["separates"],
                        "to": b["separates"],
                    }
                )
    out["label_flips"] = flips
    dest = SHIPPED_DIR / "cjk_recount.json"
    A62._write_json_atomic(dest, out)
    print(f"wrote {dest}; flips={len(flips)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
