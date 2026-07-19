"""Issue #1417 milder-rude pilot gate (VM, CPU-only, ~minutes; plan v6 §4.2 item 5).

Early-abort evaluator between the phase-A gen/capture and the ~57k-call full
judge + phase C. Reads the 200-row pilot judge kept-sets
(``<pilot-dir>/judge/kept_<model>_c2_rude_mild.json``), stages the mild cell's
store shard(s) covering the pilot rows from HF, computes the pilot
answer-variance ratio per lane as ``Y[:, 19, :].astype(np.float64).var(axis=0,
ddof=1).sum()`` over the kept pilot rows — byte-matching
``issue825_fit_cells.run_cell``'s ``y_trace_cov_frozen`` formula
(issue825_fit_cells.py:1078-1081), via ``issue1417_battery.load_own_bundle`` +
``_xy_for`` with the kept-pilot allowlist — divided by the committed refit
anchor denominator (``<anchors-dir>/cells_S{1,2}.json`` ->
``y_trace_cov_frozen["19"]``), and writes ``pilot_gate_report.json`` with
per-lane ``{yield_frac, var_ratio, bars, verdict}``.

Bars (pre-registered, plan §4.3/§7 — NOT new science thresholds; the binding
full-pass floors stay 0.5/0.5): per lane, pilot yield_frac >= 0.40 AND pilot
var_ratio >= 0.40. Disposition is PER LANE (plan §4.3 step 2): a passing lane
proceeds to its full judge + phase C; a failing lane gets ONE render-revision
retry; the round aborts entirely only when BOTH lanes fail after the retry —
the ORCHESTRATOR routes that; this helper only writes the report and exits a
DESIGNED distinct rc (23, never a bare rc=1 — gotchas #1415) when >=1 lane
fails.

§12.13 trace (committed refit values): base yield 0.109 < 0.40 => the yield
arm FIRES; variance 0.4864/0.4620 > 0.40 => the variance arm deliberately does
NOT fire (that marginal region belongs to the binding 0.5 full-pass floor);
instruct yield 0.662 => no false fire. Pinned by
tests/test_issue1417_milder_render.py.

Launch (shared-VM thread caps mandatory):
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
  uv run python scripts/issue1417_pilot_gate.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common931  # noqa: E402
import issue1417_battery as b1417  # noqa: E402
import issue1417_render as r1417  # noqa: E402

SCRIPT = "scripts/issue1417_pilot_gate.py"
HEADLINE_LAYER = 19
PILOT_YIELD_BAR = 0.40  # plan §7 gate 1 (registered; full-pass floor stays 0.5)
PILOT_VAR_BAR = 0.40
RC_PILOT_GATE = 23  # designed halt: >=1 lane failed its pilot bars


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=SCRIPT)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1417"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1417"))
    ap.add_argument(
        "--pilot-dir", type=Path, default=None, help="default <out-dir>/milder_rude/pilot"
    )
    ap.add_argument(
        "--anchors-dir",
        type=Path,
        default=None,
        help="committed refit anchor dir (default <out-dir>/refit/anchors; the "
        "y_trace_cov_frozen['19'] denominators, git @ 3867276eea)",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="default <pilot-dir>/pilot_gate_report.json"
    )
    ap.add_argument("--models", default=",".join(r1417.MODELS))
    ap.add_argument("--cell", default="c2_rude_mild")
    ap.add_argument("--yield-bar", type=float, default=PILOT_YIELD_BAR)
    ap.add_argument("--var-bar", type=float, default=PILOT_VAR_BAR)
    ap.add_argument(
        "--skip-staging",
        action="store_true",
        help="tests/offline: use the store shards already under <data-dir>/store",
    )
    args = ap.parse_args()
    if args.pilot_dir is None:
        args.pilot_dir = Path(args.out_dir) / "milder_rude" / "pilot"
    if args.anchors_dir is None:
        args.anchors_dir = Path(args.out_dir) / "refit" / "anchors"
    if args.out is None:
        args.out = Path(args.pilot_dir) / "pilot_gate_report.json"
    return args


def lane_verdict(
    yield_frac: float,
    var_ratio: float,
    *,
    yield_bar: float = PILOT_YIELD_BAR,
    var_bar: float = PILOT_VAR_BAR,
) -> dict:
    """Pure per-lane decision (§12.13 trace): BOTH bars must clear; a NaN /
    None reading counts as a miss on that arm (fail-closed)."""

    def _ok(v: float | None, bar: float) -> bool:
        return v is not None and not math.isnan(v) and v >= bar

    y_ok = _ok(yield_frac, yield_bar)
    v_ok = _ok(var_ratio, var_bar)
    return {
        "yield_frac": yield_frac,
        "var_ratio": var_ratio,
        "bars": {"yield": yield_bar, "var_ratio": var_bar},
        "yield_arm_fires": not y_ok,
        "var_arm_fires": not v_ok,
        "verdict": "pass" if (y_ok and v_ok) else "fail",
    }


def trace_cov_l19(Y) -> float:
    """Byte-match of issue825_fit_cells.run_cell's y_trace_cov formula @ L19
    (issue825_fit_cells.py:1078-1081): sum of per-dim fp64 variances (ddof=1)."""
    import numpy as np

    assert Y.ndim == 3 and Y.shape[1] > HEADLINE_LAYER, Y.shape
    return float(Y[:, HEADLINE_LAYER, :].astype(np.float64).var(axis=0, ddof=1).sum())


def anchor_denominator(anchors_dir: Path, model: str) -> float:
    """Committed refit anchor y_trace_cov_frozen['19'] (S1=instruct, S2=pretrained)."""
    anchor_id = "S1" if model == "instruct" else "S2"
    p = Path(anchors_dir) / f"cells_{anchor_id}.json"
    assert p.exists(), f"anchor denominator missing: {p} (refit round, git 3867276eea)"
    v = float(json.loads(p.read_text())["y_trace_cov_frozen"][str(HEADLINE_LAYER)])
    assert v > 0, f"non-positive anchor denominator {v} in {p}"
    return v


def stage_pilot_store(data_dir: Path, model: str, cell: str, needed_ids: set[str]) -> None:
    """Stage the cell's sidecar JSONs + ONLY the .pt shards covering needed_ids
    (shard sidecars are KB-scale; a .pt shard is ~hundreds of MB — the pilot's
    first-200-pool-order rows live in shard000 by the 500-conv shard layout)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path, stage_hub_file

    dest = Path(data_dir) / "store"
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f"{r1417.HF_PREFIX}/analysis_tensors/store"
    stem = f"{model}_{cell}_s_shard"
    paths = list_hf_files_under_path(HfApi(), r1417.HF_DATA_REPO, prefix, repo_type="dataset")
    sidecars = sorted(p for p in paths if Path(p).name.startswith(stem) and p.endswith(".json"))
    assert sidecars, f"no store sidecars for {model}_{cell} under {prefix} — run phase A first"
    n_pt = 0
    for sp in sidecars:
        sdest = dest / Path(sp).name
        if not sdest.exists():
            stage_hub_file(r1417.HF_DATA_REPO, sp, sdest, repo_type="dataset")
        side = json.loads(sdest.read_text())
        assert r1417.fingerprint_matches(side), f"{sdest}: store fingerprint mismatch"
        if needed_ids & {str(c) for c in side["conv_ids"]}:
            pt = sp[: -len(".json")] + ".pt"
            pdest = dest / Path(pt).name
            if not pdest.exists():
                stage_hub_file(r1417.HF_DATA_REPO, pt, pdest, repo_type="dataset")
            n_pt += 1
    assert n_pt, f"no shard covers the pilot kept rows for {model}_{cell}"
    print(f"[i1417-pilot-gate] {model}/{cell}: {n_pt} .pt shard(s) staged for the pilot rows")


def lane_report(args, model: str) -> dict:
    """One lane's pilot read: yield from the pilot kept-set; variance ratio via
    the frozen trace-cov formula on the kept pilot rows over the anchor."""
    kept_p = Path(args.pilot_dir) / "judge" / f"kept_{model}_{args.cell}.json"
    assert kept_p.exists(), f"pilot kept-set missing: {kept_p} — run the pilot judge first"
    kd = json.loads(kept_p.read_text())
    assert r1417.fingerprint_matches(kd), f"{kept_p}: fingerprint mismatch"
    assert kd["model"] == model and kd["cell"] == args.cell, (kd["model"], kd["cell"])
    yield_frac = float(kd["yield_frac"])
    kept = [str(c) for c in kd["kept_conv_ids"]]
    var_ratio = float("nan")
    var_l19 = float("nan")
    if len(kept) >= 2:  # ddof=1 needs >=2 rows; a below-2 lane fails the yield arm anyway
        if not args.skip_staging:
            stage_pilot_store(args.data_dir, model, args.cell, set(kept))
        bundle = b1417.load_own_bundle(args.data_dir, model, args.cell)
        xy = b1417._xy_for(bundle, b1417.own_cell_dict(model, args.cell, "ctx"), kept)
        var_l19 = trace_cov_l19(xy["Y"])
        var_ratio = var_l19 / anchor_denominator(args.anchors_dir, model)
        b1417.evict_bundle("own", model, args.cell)
    lane = lane_verdict(yield_frac, var_ratio, yield_bar=args.yield_bar, var_bar=args.var_bar)
    lane["n_judged"] = kd["n_judged"]
    lane["n_kept"] = kd["n_kept"]
    lane["y_trace_cov_l19_pilot"] = var_l19
    print(
        f"[i1417-pilot-gate] {model}: yield={yield_frac:.4f} var_ratio={var_ratio:.4f} "
        f"-> {lane['verdict']}"
    )
    return lane


def main() -> int:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    lanes = {model: lane_report(args, model) for model in models}
    failing = sorted(m for m, v in lanes.items() if v["verdict"] != "pass")
    report = {
        "metadata": common931.metadata(
            SCRIPT, r1417.GEN_SEED, sum(v["n_judged"] for v in lanes.values())
        ),
        **r1417.fingerprint(),
        "gate": "milder_pilot",
        "cell": args.cell,
        "bars": {"yield": args.yield_bar, "var_ratio": args.var_bar},
        "anchors_dir": str(args.anchors_dir),
        "lanes": lanes,
        "lanes_failing": failing,
        "disposition": (
            "per-lane (plan §4.3 step 2): a passing lane proceeds to its full judge + "
            "phase C; a failing lane gets ONE render revision; full abort only when "
            "BOTH lanes fail after the retry budget"
        ),
        "pass": not failing,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"[i1417-pilot-gate] wrote {out} (failing lanes: {failing or 'none'})")
    if failing:
        print(
            f"[i1417-pilot-gate] PILOT GATE: {len(failing)} lane(s) below bars — "
            "per-lane disposition applies (rc 23, designed halt)",
            file=sys.stderr,
        )
        return RC_PILOT_GATE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
