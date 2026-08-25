#!/usr/bin/env python3
"""Issue #2378 `causal-patching-arms` — VM-side fold: F_act + F_beh summary.

Consumes the pod round's harvested dirs (--patch-root) + the judge wave's
PUBLISHED fold (--judge-dir, resolved through the os.replace'd
``fold_manifest.json`` pointer — r18) and writes the round's eval JSON
``patch_summary.json`` under eval_results/issue_2378/causal-patching-arms/.

DV assembly (all estimators imported from the #2094 suite, never re-derived):
- F_act: per-cell fraction-of-full-context-swap (``fmetrics.f_act`` — computed
  by issue2378_patch_run.phase_screen for the greedy grid; recomputed here for
  the temp-1.0 confirm draws via the same ``_grid_fact`` code path).
- F_beh: per-row dual-rubric contrast Δ = (persona - assistant)/100
  (``fmetrics.delta_contrast``), coherence-gated at COHERENCE_THRESHOLD=60 for
  patched rows (inherited #2094 gate; drops counted, never coerced), then
  ``fmetrics.f_beh`` per cell against the pair's anchor floor/ceiling Δ means.
- CIs: pair-clustered bootstrap B=10,000 percentile CIs per family, and the
  paired steered-null difference CI (``issue2094_analysis
  .bootstrap_family_means_batched`` via patch_common.screen_families).

Scope notes carried into the output: chat~plain (arm b) is F_act-only (no
judgeable persona contrast); MODEL CAVEAT — causality tested on Qwen3.6-27B
while the correlational headline (#1345/#2054) and the patching baseline
(#2094) are Qwen2.5-7B (no 2.5-7B arm this round).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue2378_common as cm  # noqa: E402
import issue2378_patch_common as pc  # noqa: E402

MODEL_CAVEAT = (
    "Causality tested on Qwen3.6-27B; the correlational headline (#1345/#2054) and the "
    "patching baseline (#2094) are Qwen2.5-7B. No 2.5-7B arm in this round (budget) — "
    "cross-model transfer of the causal read is an open caveat."
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_scores(judge_dir: Path) -> dict[str, dict[str, int]]:
    """{row_id: {rubric: kept score}} from the judge wave's PUBLISHED fold
    (manifest-pointer resolution, r18 patch-judge-fold-publish-window:
    ``read_fold_manifest`` refuses an unpublished / half-published fold —
    never read fold dirs or a legacy top-level scores.jsonl directly)."""
    import issue2378_patch_judge as pj

    path = pj.read_fold_manifest(judge_dir)["scores_path"]
    out: dict[str, dict[str, int]] = {}
    for row in cm.iter_jsonl(path):
        if row.get("kept"):
            out.setdefault(row["row_id"], {})[row["rubric"]] = int(row["score"])
    if not out:
        raise RuntimeError(f"no kept judge scores at {path} (empty selection — fail loud)")
    return out


def _delta(scores: dict[str, int]) -> float | None:
    """Dual-rubric per-row contrast via fmetrics.delta_contrast (drop on a
    missing rubric — never coerce)."""
    import torch

    from explore_persona_space.experiments.issue2094.fmetrics import delta_contrast

    if "persona" not in scores or "assistant" not in scores:
        return None
    d = delta_contrast(
        torch.tensor([float(scores["persona"])]), torch.tensor([float(scores["assistant"])])
    )
    return float(d[0])


def _anchor_deltas(patch_root: Path, scores: dict[str, dict[str, int]]) -> dict[str, list[float]]:
    """{ctx_id: [Δ per kept anchor draw]} (chat/story anchors, arm-a scope)."""
    out: dict[str, list[float]] = {}
    for p in sorted((patch_root / "anchors" / "rollouts").glob("*.jsonl")):
        for r in cm.iter_jsonl(p):
            if r.get("drop_reason") is not None:
                continue
            d = _delta(scores.get(f"anchors|{r['ctx_id']}|d{r['draw']}", {}))
            if d is not None:
                out.setdefault(r["ctx_id"], []).append(d)
    return out


def _cell_rows(patch_root: Path, stage: str) -> list[dict]:
    rows = []
    d = patch_root / stage / "rollouts"
    if d.is_dir():
        for p in sorted(d.glob("*.jsonl")):
            rows.extend(cm.iter_jsonl(p))
    return rows


def _fbeh_cells(
    stage: str,
    rows: list[dict],
    scores: dict[str, dict[str, int]],
    anchor_deltas: dict[str, list[float]],
    drops: Counter,
) -> list[dict]:
    """Per-cell F_beh rows (chat~story scope; coherence-gated patched draws)."""
    import torch

    from explore_persona_space.experiments.issue2094.fmetrics import f_beh

    per_cell: dict[str, dict] = {}
    for r in rows:
        if r.get("drop_reason") is not None or r["pair_type"] != "chat~story":
            continue
        sc = scores.get(f"{stage}|{r['cell_id']}|d{r['draw']}", {})
        coh = sc.get("coherence")
        if coh is None or coh < pc.COHERENCE_THRESHOLD:
            drops[f"{stage}_coherence_gate"] += 1
            continue
        d = _delta(sc)
        if d is None:
            drops[f"{stage}_missing_rubric"] += 1
            continue
        rec = per_cell.setdefault(r["cell_id"], {**r, "deltas": []})
        rec["deltas"].append(d)
    out: list[dict] = []
    for cell_id, rec in sorted(per_cell.items()):
        fl = anchor_deltas.get(rec["tgt"])
        ce = anchor_deltas.get(rec["src"])
        if not fl or not ce:
            drops[f"{stage}_anchor_delta_missing"] += 1
            continue
        res = f_beh(
            torch.tensor([float(np.mean(rec["deltas"]))]),
            torch.tensor([float(np.mean(fl))]),
            torch.tensor([float(np.mean(ce))]),
        )
        out.append(
            {
                **{
                    k: rec[k]
                    for k in (
                        "cell_id",
                        "arm",
                        "variant",
                        "src",
                        "tgt",
                        "qid",
                        "char",
                        "pair_type",
                        "direction",
                        "family",
                    )
                },
                "n_draws": len(rec["deltas"]),
                "f_beh": None if bool(res.degenerate_denominator[0]) else float(res.f_beh[0]),
                "contrast": float(res.contrast[0]),
                "denominator": float(res.denominator[0]),
                "degenerate_denominator": bool(res.degenerate_denominator[0]),
            }
        )
    return out


def _family_table(cells: list[dict], value_key: str) -> dict[str, dict[str, float]]:
    fam: dict[str, dict[str, float]] = {}
    for c in cells:
        v = c.get(value_key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        fam.setdefault(c["family"], {})[c["qid"]] = float(v)
    return fam


def _family_stats(fam_vals: dict[str, dict[str, float]]) -> dict:
    """Per-family n + mean, plus the paired steered-vs-null diff screen
    (bootstrap CIs live on the steered-null DIFFERENCES — the
    ``steered_vs_null`` block — not on the raw per-family means)."""
    diffs: dict[str, dict[str, float]] = {}
    for fam, vals in fam_vals.items():
        if not fam.endswith("|steered"):
            continue
        nvals = fam_vals.get(fam.rsplit("|", 1)[0] + "|null", {})
        d = {q: vals[q] - nvals[q] for q in vals if q in nvals}
        if d:
            diffs[fam] = d
    screen = pc.screen_families(diffs) if diffs else {"families": {}, "confirm_families": []}
    means = {
        fam: {"n": len(vals), "mean": float(np.mean(list(vals.values())))}
        for fam, vals in sorted(fam_vals.items())
    }
    return {"family_means": means, "steered_vs_null": screen["families"]}


def _cap_hit_fractions(rows: list[dict]) -> dict[str, float]:
    """Realized cap-hit fraction per family. A row is capped iff it reached
    the token cap without ANY registered terminal: story = the SegB no-close
    convention (drop_reason cap_hit_no_close); chat/plain = neither EOS nor
    the framing's textual stop fired (``hit_stop``, r18 — a stop-string halt
    is an EFFECTIVE stop, never a cap hit; legacy rows without the field
    fall back to hit_eos alone)."""
    tot: Counter = Counter()
    hit: Counter = Counter()
    for r in rows:
        # generation-side drop rows (e.g. opener_empty) never generated at the
        # cap; only the story no-close convention is itself a cap-hit marker
        if r.get("drop_reason") not in (None, "cap_hit_no_close"):
            continue
        fam = r.get("family") or f"anchors|{r.get('ctx_id', '?').split(':', 1)[0]}"
        tot[fam] += 1
        capped = (
            r.get("drop_reason") == "cap_hit_no_close"
            if r.get("framing") == "story"
            else not r.get("hit_eos", True) and not r.get("hit_stop", False)
        )
        if capped:
            hit[fam] += 1
    return {f: hit[f] / tot[f] for f in sorted(tot)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--patch-root", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "patch_round")
    )
    ap.add_argument(
        "--judge-dir",
        default=str(cm.REPO_ROOT / "eval_results" / "issue_2378" / pc.LEDGER_SUBDIR / "judge"),
    )
    ap.add_argument(
        "--out-dir", default=str(cm.REPO_ROOT / "eval_results" / "issue_2378" / pc.LEDGER_SUBDIR)
    )
    ap.add_argument("--lstar", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        import issue2378_patch_common as _pc_mod

        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__, _pc_mod.__file__)
        raise SystemExit(0)

    patch_root = Path(args.patch_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drops: Counter = Counter()

    # F_act: grid screen report (pod-computed) + confirm recompute (same code).
    import issue2378_patch_run as R

    screen = json.loads((patch_root / "screen" / "screen_report.json").read_text(encoding="utf-8"))
    r_argv = ["--phase", "screen", "--out-root", str(patch_root)]
    if args.lstar:
        r_argv += ["--lstar", str(args.lstar)]
    if args.tiny:
        r_argv += ["--tiny", "--tiny-layers", str(args.tiny_layers)]
    r_args = R.build_argparser().parse_args(r_argv)
    confirm_fact_rows, confirm_dropped = (
        R._grid_fact(r_args, "confirm") if (patch_root / "confirm" / "va").is_dir() else ([], {})
    )
    confirm_cells: dict[str, dict] = {}
    for r in confirm_fact_rows:
        if r["degenerate"]:
            continue
        rec = confirm_cells.setdefault(r["cell_id"], {**r, "vals": []})
        rec["vals"].append(r["f_act"])
    confirm_cell_rows = [
        {**rec, "f_act": float(np.mean(rec["vals"])), "n_draws": len(rec["vals"])}
        for rec in confirm_cells.values()
    ]
    confirm_fact = _family_stats(_family_table(confirm_cell_rows, "f_act"))

    # F_beh from the judge wave.
    scores = _load_scores(Path(args.judge_dir))
    anchor_deltas = _anchor_deltas(patch_root, scores)
    grid_rows = _cell_rows(patch_root, "grid")
    confirm_rows = _cell_rows(patch_root, "confirm")
    fbeh_grid = _fbeh_cells("grid", grid_rows, scores, anchor_deltas, drops)
    fbeh_confirm = _fbeh_cells("confirm", confirm_rows, scores, anchor_deltas, drops)

    summary = {
        "followup_label": pc.FOLLOWUP_LABEL,
        "model_caveat": MODEL_CAVEAT,
        "scope_notes": [
            "chat~plain (arm b) is F_act-only: both framings answer in the assistant "
            "register, so the persona/assistant dual rubric has no expected separation "
            "(degenerate F_beh denominators by design).",
            "F_beh patched rows are coherence-gated at "
            f">= {pc.COHERENCE_THRESHOLD} (inherited #2094 gate); drops counted below.",
        ],
        "screen_report": {k: screen[k] for k in ("screen_rule", "families", "confirm_families")},
        "f_act_grid_family_means": screen.get("family_means", {}),
        "f_act_confirm": confirm_fact,
        "f_act_confirm_dropped": confirm_dropped,
        "f_beh_grid": _family_stats(_family_table(fbeh_grid, "f_beh")),
        "f_beh_confirm": _family_stats(_family_table(fbeh_confirm, "f_beh")),
        "f_beh_cells_grid": fbeh_grid,
        "f_beh_cells_confirm": fbeh_confirm,
        "cap_hit_fractions": {
            "anchors": _cap_hit_fractions(_cell_rows(patch_root, "anchors")),
            "grid": _cap_hit_fractions(grid_rows),
            "confirm": _cap_hit_fractions(confirm_rows),
        },
        "drops": dict(drops),
        "n_kept_judge_rows": len(scores),
        "metadata": cm.run_metadata({"phase": "patch_analysis"}),
    }
    out_path = out_dir / "patch_summary.json"
    cm.atomic_write_json(out_path, summary)
    _log(
        f"[analysis] wrote {out_path} — grid f_beh cells={len(fbeh_grid)} "
        f"confirm f_beh cells={len(fbeh_confirm)} drops={dict(drops)}"
    )


if __name__ == "__main__":
    main()
