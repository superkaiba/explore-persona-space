"""new-arm-round VM-side collect/merge (task #1739, plan v8 §4 "Collection").

Stages each leg's self-uploaded results from HF ``issue1739_new_arm_round/``
(scoped ``hub.stage_hub_prefix`` — never a full-tree listing), merges the
per-leg percell transfer/cells JSONLs into
``eval_results/issue_1739/new_arm_round/arm_results/``, extends the all-arms
Spearman aggregation with the fc regimes + arms 17/18, and writes the two
round figures. Operational hardening (r1 critique, baked into plan v8):

- FULL-ROW-KEY dedup BEFORE aggregation: rows dedup on
  (leg, arm, behavior, eval_rung, rung_kind, budget_l, u_rung_label, regime,
  draw, seed, variant, layer) — the committed evil JSONL carries duplicate
  (arm, coordinate) rows for shared arms, so a naive concat double-counts.
- u=full-MATCHED joins ONLY: every new-arm-vs-baseline join/figure compares
  at the registered u=full fixed coordinate (plan Must-Fix 2).
- HALL join rule: the oracle bar's PRIMARY read is evil+sycophancy;
  hallucination is a SECONDARY read against the within-round u=full arm12
  rider rows, with the committed-u=250 mismatch flagged in the summary; the
  gap-fill round's committed ``hallucination_maxood`` rows are consumed
  read-only IF already present in the repo tree.
- fc-suffix grep gate: 100% of fc-leg rows must carry an ``_fc`` regime.
- Per-unit roster accounting: every requested transfer arm appears in a
  unit's rows or skips (K1 count-style mechanism check; ``--strict`` fails).

Runs in minutes (JSON merges + matplotlib); text/JSON only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

HF_PREFIX = "issue1739_new_arm_round"
DATA_REPO = "superkaiba1/explore-persona-space-data"
LEGS = ("fc", "oracle", "arm5ood", "nlood")
ORACLE_ARMS = ("arm12_oracle_reg", "arm17_oracle_mlp", "arm18_oracle_krr")
DEDUP_KEY = (
    "arm",
    "behavior",
    "eval_rung",
    "rung_kind",
    "budget_l",
    "u_rung_label",
    "regime",
    "draw",
    "seed",
    "variant",
    "layer",
)


def _git_commit() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT, check=False
    )
    return p.stdout.strip() if p.returncode == 0 else "unknown"


def _iter_jsonl(path: Path):
    """Text-mode line iteration (NEVER splitlines — gotchas.md U+2028)."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_unit_jsonl(path: Path, *, leg: str) -> tuple[list[dict], list[dict], list[dict]]:
    """A {unit_key, rows, skips} JSONL -> (rows, skips, accounting_violations)."""
    rows: list[dict] = []
    skips: list[dict] = []
    violations: list[dict] = []
    for rec in _iter_jsonl(path):
        u_rows = rec.get("rows") or []
        u_skips = rec.get("skips") or []
        for r in u_rows:
            rows.append({**r, "leg": leg})
        skips.extend({**s, "leg": leg} for s in u_skips)
        # K1 count-style accounting: every requested transfer arm must appear
        # in this unit's rows OR skips (drop-never-silent).
        try:
            key = json.loads(rec.get("unit_key") or "{}")
        except (TypeError, json.JSONDecodeError):
            key = {}
        roster = key.get("transfer_arms") or []
        seen = {r.get("arm") for r in u_rows} | {s.get("arm") for s in u_skips}
        missing = [a for a in roster if a not in seen]
        if missing:
            violations.append({"leg": leg, "path": str(path), "missing_arms": missing})
    return rows, skips, violations


def load_cells_arm_rows(path: Path, *, leg: str) -> list[dict]:
    """cells.jsonl (per-unit records) -> flat arm rows (rho_per_layer dropped)."""
    rows: list[dict] = []
    for rec in _iter_jsonl(path):
        for r in rec.get("arms") or []:
            slim = {k: v for k, v in r.items() if k not in ("rho_per_layer",)}
            slim.setdefault("rung_kind", "train_grid")
            rows.append({**slim, "leg": leg})
    return rows


def dedup_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Full-row-key dedup (leg + DEDUP_KEY); first occurrence kept."""
    seen: set[tuple] = set()
    out: list[dict] = []
    dups = 0
    for r in rows:
        key = (r.get("leg"),) + tuple(r.get(k) for k in DEDUP_KEY)
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        out.append(r)
    return out, dups


def stage_from_hf(stage_root: Path) -> Path:
    """Mirror issue1739_new_arm_round/** locally (scoped staging; #833 recipe)."""
    from explore_persona_space.orchestrate import hub

    files = hub.stage_hub_prefix(DATA_REPO, HF_PREFIX, stage_root, repo_type="dataset")
    print(f"[collect] staged {len(files)} files under {stage_root / HF_PREFIX}", flush=True)
    return stage_root / HF_PREFIX


def gather(root: Path) -> dict:
    """Walk the leg tree under ``root`` and merge every percell JSONL."""
    transfer_rows: list[dict] = []
    cells_rows: list[dict] = []
    skips: list[dict] = []
    violations: list[dict] = []
    sources: list[str] = []
    for tpath in sorted(root.rglob("transfer.jsonl")):
        rel = tpath.relative_to(root).as_posix()
        leg = rel.split("/arm_results/")[0]
        rows, sk, viol = load_unit_jsonl(tpath, leg=leg)
        transfer_rows += rows
        skips += sk
        violations += viol
        sources.append(rel)
    for cpath in sorted(root.rglob("cells.jsonl")):
        rel = cpath.relative_to(root).as_posix()
        leg = rel.split("/arm_results/")[0]
        cells_rows += load_cells_arm_rows(cpath, leg=leg)
        sources.append(rel)
    if not sources:
        raise SystemExit(f"[collect] no percell JSONLs found under {root}")
    return {
        "transfer_rows": transfer_rows,
        "cells_rows": cells_rows,
        "skips": skips,
        "violations": violations,
        "sources": sources,
    }


def fc_suffix_gate(rows: list[dict]) -> None:
    """Success criterion (ii): 100% of fc-leg rows carry an _fc regime label —
    and NO row anywhere carries the structurally-undefined ``e2_fc`` regime
    (plan v9 structural restriction, concern e2fc-structurally-null-direction:
    an e2_fc row means a leg ran the dropped matched-e2 construction, whose
    direction is exact-cancellation float residue)."""
    bad = [
        r
        for r in rows
        if str(r.get("leg", "")).startswith("fc") and not str(r.get("regime", "")).endswith("_fc")
    ]
    if bad:
        raise SystemExit(
            f"[collect] fc-suffix gate FAILED: {len(bad)} fc-leg rows without an _fc regime "
            f"(first: {[{k: bad[0].get(k) for k in ('arm', 'regime', 'behavior')}]})"
        )
    null_rows = [r for r in rows if str(r.get("regime")) == "e2_fc"]
    if null_rows:
        raise SystemExit(
            f"[collect] structural-restriction gate FAILED: {len(null_rows)} rows carry the "
            "dropped matched-e2_fc regime (structurally-zero direction; plan v9) — a leg ran "
            f"the refused construction (first: "
            f"{[{k: null_rows[0].get(k) for k in ('leg', 'arm', 'behavior')}]})"
        )


def load_baseline_rows(paths: list[Path]) -> list[dict]:
    """Committed t1 baseline unit-JSONLs (e.g. wide_ood/*_transfer.jsonl)."""
    rows: list[dict] = []
    for p in paths:
        got, _sk, _v = load_unit_jsonl(p, leg="baseline_t1")
        rows += got
    return rows


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and v == v]
    return (sum(vals) / len(vals)) if vals else None


def summarize(rows: list[dict], k1_flags: dict[tuple[str, str], bool] | None = None) -> list[dict]:
    """Per-(leg, behavior, arm, regime, variant, rung_kind, eval_rung, budget_l,
    u_rung_label) mean rho_frozen + n — the fc/17/18-extended aggregation.
    K1-flagged (behavior, eval_rung) groups carry a ``k1_spread_floor`` field
    so no hypothesis read averages them in silently (fit-and-star)."""
    k1_flags = k1_flags or {}
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        key = tuple(
            r.get(k)
            for k in (
                "leg",
                "behavior",
                "arm",
                "regime",
                "variant",
                "rung_kind",
                "eval_rung",
                "budget_l",
                "u_rung_label",
            )
        )
        groups.setdefault(key, []).append(r.get("rho_frozen"))
    out = []
    for key, vals in sorted(groups.items(), key=str):
        leg, behavior, arm, regime, variant, rung_kind, eval_rung, budget_l, u_label = key
        row = {
            "leg": leg,
            "behavior": behavior,
            "arm": arm,
            "regime": regime,
            "variant": variant,
            "rung_kind": rung_kind,
            "eval_rung": eval_rung,
            "budget_l": budget_l,
            "u_rung_label": u_label,
            "n_rows": len(vals),
            "rho_frozen_mean": _mean(vals),
        }
        if str(leg).startswith("oracle") and behavior == "hallucination":
            row["read"] = (
                "SECONDARY (hall oracle bar joins the within-round u=full arm12 rider; "
                "committed hall arm12 rows are u=250 — u-mismatched, flagged per plan v9)"
            )
        if _k1_flagged(k1_flags, behavior, eval_rung):
            row["k1_spread_floor"] = K1_FLAG_TEXT
        out.append(row)
    return out


def fc_vs_t1_pairs(fc_rows: list[dict], baseline_rows: list[dict]) -> list[dict]:
    """u=full-matched fc-vs-t1 joins: (behavior, arm, base regime, variant,
    rung_kind, eval_rung, budget_l, draw, seed) -> delta_rho."""

    def _key(r: dict, regime: str) -> tuple:
        return (
            r.get("behavior"),
            r.get("arm"),
            regime,
            r.get("variant"),
            r.get("rung_kind"),
            r.get("eval_rung"),
            r.get("budget_l"),
            r.get("draw"),
            r.get("seed"),
        )

    base = {}
    for r in baseline_rows:
        if r.get("u_rung_label") == "full":
            base[_key(r, str(r.get("regime")))] = r
    pairs = []
    for r in fc_rows:
        if r.get("u_rung_label") != "full":
            continue  # u=full-matched joins ONLY (plan Must-Fix 2)
        regime = str(r.get("regime", ""))
        if not regime.endswith("_fc"):
            continue
        b = base.get(_key(r, regime.removesuffix("_fc")))
        if b is None or r.get("rho_frozen") is None or b.get("rho_frozen") is None:
            continue
        pairs.append(
            {
                "behavior": r.get("behavior"),
                "arm": r.get("arm"),
                "regime_base": regime.removesuffix("_fc"),
                "variant": r.get("variant"),
                "rung_kind": r.get("rung_kind"),
                "eval_rung": r.get("eval_rung"),
                "budget_l": r.get("budget_l"),
                "draw": r.get("draw"),
                "seed": r.get("seed"),
                "rho_fc": r.get("rho_frozen"),
                "rho_t1": b.get("rho_frozen"),
                "delta_rho": float(r["rho_frozen"]) - float(b["rho_frozen"]),
            }
        )
    return pairs


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    tmp.replace(path)


def load_k1_flags(path: Path | None) -> dict[tuple[str, str], bool]:
    """K1 verdict table (issue1739_k1_floor.py --out) -> {(behavior, rung):
    passes_floor}. Absent/None path -> {} (figures render unannotated, and the
    summary carries no floor fields — the pre-K1-join behavior)."""
    if path is None or not Path(path).is_file():
        return {}
    payload = json.loads(Path(path).read_text())
    out: dict[tuple[str, str], bool] = {}
    for behavior, verdict in (payload.get("verdicts") or {}).items():
        for rung, row in (verdict.get("rungs") or {}).items():
            out[(str(behavior), str(rung))] = bool(row.get("passes_floor"))
    return out


K1_FLAG_TEXT = "N/A — unmeasurable (spread floor)"


def _k1_flagged(k1_flags: dict, behavior, rung) -> bool:
    """True when the K1 table names this (behavior, rung) a floor FAILURE."""
    return k1_flags.get((str(behavior), str(rung))) is False


def render_figures(
    pairs: list[dict],
    oracle_rows: list[dict],
    fig_dir: Path,
    k1_flags: dict[tuple[str, str], bool] | None = None,
) -> list[Path]:
    """The two round figures: (a) fc-vs-t1 delta-rho, (b) oracle family bar.

    Zero-bar discipline (code-review r1 Minor 1 + CLAUDE.md After-Every-
    Experiment 8c): a cell with NO rows draws NO bar (never a zero bar), and
    a K1-FLAGGED rung renders as ``N/A — unmeasurable (spread floor)`` — no
    bars, annotated x label — never an unannotated/zero bar."""
    k1_flags = k1_flags or {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # (a) fc-vs-t1 delta-rho per (behavior, arm): mean bar + per-pair points.
    behaviors = sorted({p["behavior"] for p in pairs}) or ["(no pairs)"]
    arms_order = sorted({p["arm"] for p in pairs})
    fig, axes = plt.subplots(
        1, max(len(behaviors), 1), figsize=(3.4 * max(len(behaviors), 1), 3.2), squeeze=False
    )
    colors = paper_palette(max(len(arms_order), 1))
    for bi, b in enumerate(behaviors):
        ax = axes[0][bi]
        for ai, arm in enumerate(arms_order):
            deltas = [p["delta_rho"] for p in pairs if p["behavior"] == b and p["arm"] == arm]
            mean_delta = _mean(deltas)
            if not deltas or mean_delta is None:
                continue  # no matched pairs: NO bar (never a zero bar)
            ax.scatter([ai] * len(deltas), deltas, s=8, alpha=0.4, color=colors[ai])
            ax.bar(ai, mean_delta, width=0.6, color=colors[ai], alpha=0.6)
        ax.axhline(0.0, lw=0.8, color="0.4")
        ax.set_xticks(range(len(arms_order)))
        ax.set_xticklabels([a.split("_")[0] for a in arms_order], rotation=0)
        ax.set_title(b)
        if bi == 0:
            ax.set_ylabel("delta rho (fc - t1), matched cells @ u=full")
    fig.suptitle("Final-context vs answer-avg r_B: per-cell delta rho (u=full-matched joins)")
    out_a = fig_dir / "newarm_fc_vs_t1_delta.png"
    fig.tight_layout()
    fig.savefig(out_a, dpi=200)
    plt.close(fig)
    written.append(out_a)

    # (b) oracle family bar: mean rho_frozen per (behavior, arm, rung_kind).
    o_beh = sorted({r["behavior"] for r in oracle_rows}) or ["(no oracle rows)"]
    fig, axes = plt.subplots(
        1, max(len(o_beh), 1), figsize=(3.6 * max(len(o_beh), 1), 3.2), squeeze=False
    )
    colors = paper_palette(len(ORACLE_ARMS))
    for bi, b in enumerate(o_beh):
        ax = axes[0][bi]
        rungs = sorted({str(r.get("eval_rung")) for r in oracle_rows if r["behavior"] == b})
        width = 0.8 / max(len(ORACLE_ARMS), 1)
        for ai, arm in enumerate(ORACLE_ARMS):
            labeled = False
            for i, rung in enumerate(rungs):
                if _k1_flagged(k1_flags, b, rung):
                    continue  # flagged rung: annotated N/A, never a bar
                vals = [
                    r.get("rho_frozen")
                    for r in oracle_rows
                    if r["behavior"] == b and r["arm"] == arm and str(r.get("eval_rung")) == rung
                ]
                y = _mean(vals)
                if y is None:
                    continue  # no rows: NO bar (never a zero bar)
                ax.bar(
                    i + ai * width,
                    y,
                    width=width,
                    color=colors[ai],
                    label=None if labeled else arm.split("_")[0],
                )
                labeled = True
        ax.set_xticks([i + width for i in range(len(rungs))])
        ax.set_xticklabels(
            [r + ("\n" + K1_FLAG_TEXT if _k1_flagged(k1_flags, b, r) else "") for r in rungs],
            rotation=30,
            ha="right",
            fontsize=6,
        )
        title = b + (" [SECONDARY: u=250-committed mismatch]" if b == "hallucination" else "")
        ax.set_title(title, fontsize=9)
        if bi == 0:
            ax.set_ylabel("mean rho_frozen @ u=full")
            ax.legend(fontsize=7)
    fig.suptitle("Oracle family (arm12 linear vs arm17 MLP vs arm18 KRR), within-round rows")
    out_b = fig_dir / "newarm_oracle_family.png"
    fig.tight_layout()
    fig.savefig(out_b, dpi=200)
    plt.close(fig)
    written.append(out_b)
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse
    import time

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-root", type=Path, default=_REPO_ROOT / "eval_results/issue_1739/new_arm_round"
    )
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=_REPO_ROOT / "data/issue_1739/newarm_dl",
        help="HF staging mirror root (files land at <stage-root>/<repo path>)",
    )
    ap.add_argument(
        "--local",
        action="store_true",
        help="skip HF staging; read leg JSONLs directly under --out-root (smoke path)",
    )
    ap.add_argument(
        "--baseline-jsonl",
        type=Path,
        nargs="*",
        default=None,
        help="committed t1 baseline unit-JSONLs for the fc-vs-t1 join "
        "(default: eval_results/issue_1739/wide_ood/*_transfer.jsonl)",
    )
    ap.add_argument("--figures-dir", type=Path, default=_REPO_ROOT / "figures/issue_1739")
    ap.add_argument(
        "--k1-verdicts",
        type=Path,
        default=_REPO_ROOT / "eval_results/issue_1739/new_arm_round/k1_verdicts.json",
        help="K1 spread-floor verdict table (issue1739_k1_floor.py --out); flagged "
        "(behavior, rung) cells render 'N/A — unmeasurable (spread floor)' in the "
        "figures and carry a k1_spread_floor field in the summary groups — never a "
        "zero bar. Missing file -> unannotated (a WARNING is printed).",
    )
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="fail (rc=4) on per-unit roster-accounting violations instead of reporting",
    )
    args = ap.parse_args(argv)

    root = args.out_root if args.local else stage_from_hf(args.stage_root)
    got = gather(root)
    fc_suffix_gate(got["transfer_rows"])
    fc_suffix_gate(got["cells_rows"])
    transfer_rows, dup_t = dedup_rows(got["transfer_rows"])
    cells_rows, dup_c = dedup_rows(got["cells_rows"])
    print(
        f"[collect] merged: {len(transfer_rows)} transfer rows (+{dup_t} dups dropped), "
        f"{len(cells_rows)} cells rows (+{dup_c} dups dropped), "
        f"{len(got['skips'])} recorded skips, {len(got['violations'])} accounting violations",
        flush=True,
    )
    if got["violations"]:
        for v in got["violations"][:10]:
            print(f"[collect] ACCOUNTING VIOLATION: {v}", flush=True)
        if args.strict:
            return 4

    arm_dir = args.out_root / "arm_results"
    _write_jsonl(arm_dir / "merged_transfer.jsonl", transfer_rows)
    _write_jsonl(arm_dir / "merged_cells_rows.jsonl", cells_rows)
    _write_jsonl(arm_dir / "merged_skips.jsonl", got["skips"])

    baselines = args.baseline_jsonl
    if baselines is None:
        baselines = sorted(
            (_REPO_ROOT / "eval_results/issue_1739/wide_ood").glob("*_transfer.jsonl")
        )
    baseline_rows = load_baseline_rows([Path(p) for p in baselines])

    # Hall gap-fill rows (read-only, ONLY if already committed in the tree).
    maxood = (
        _REPO_ROOT
        / "eval_results/issue_1739/hallucination_maxood/arm_results/percell/transfer.jsonl"
    )
    maxood_rows: list[dict] = []
    if maxood.is_file():
        maxood_rows, _s, _v = load_unit_jsonl(maxood, leg="maxood_committed")
        print(f"[collect] consumed committed gap-fill hall rows: {len(maxood_rows)}")

    pairs = fc_vs_t1_pairs(transfer_rows + cells_rows, baseline_rows)
    oracle_rows = [
        r
        for r in transfer_rows + cells_rows + maxood_rows
        if r.get("arm") in ORACLE_ARMS
        and (str(r.get("leg", "")).startswith("oracle") or r.get("leg") == "maxood_committed")
    ]
    k1_flags = load_k1_flags(args.k1_verdicts)
    if not k1_flags:
        print(
            f"[collect] WARNING: no K1 verdict table at {args.k1_verdicts} — figures/summary "
            "render without spread-floor annotations (run issue1739_k1_floor.py --out first)",
            flush=True,
        )
    summary = {
        "n_transfer_rows": len(transfer_rows),
        "n_cells_rows": len(cells_rows),
        "n_dups_dropped": dup_t + dup_c,
        "n_skips": len(got["skips"]),
        "n_accounting_violations": len(got["violations"]),
        "n_fc_vs_t1_pairs": len(pairs),
        "join_rule": "u=full-matched ONLY (plan v9 Must-Fix 2); hall oracle bar = SECONDARY "
        "vs the within-round u=full arm12 rider (committed hall arm12 rows are u=250)",
        "k1_verdicts_path": str(args.k1_verdicts) if k1_flags else None,
        "k1_flagged_rungs": sorted(f"{b}/{r}" for (b, r), ok in k1_flags.items() if not ok),
        "sources": got["sources"],
        "groups": summarize(transfer_rows + cells_rows, k1_flags),
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(arm_dir / "newarm_spearman_summary.json", summary)
    _write_jsonl(arm_dir / "fc_vs_t1_pairs.jsonl", pairs)
    print(f"[collect] summary -> {arm_dir / 'newarm_spearman_summary.json'}", flush=True)

    if not args.skip_figures:
        written = render_figures(pairs, oracle_rows, args.figures_dir, k1_flags)
        for p in written:
            print(f"[collect] figure -> {p}", flush=True)
    print(
        "[collect] done. Commit by explicit path: "
        f"git add {arm_dir} {args.figures_dir}/newarm_*.png && git commit"
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension finalize teardown
