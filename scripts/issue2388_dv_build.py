"""Issue #2388 — correctness-DV builder (labeling.json-shaped; plan section 4 "DV").

Two modes:

1. ``--from-banked``: derive the QA correctness DV from the banked #1739 artifact
   (``eval_results/issue_1739/dv_dataset/hallucination/labeling.json``) by setting
   ``dv := fractions.correct`` per row and carrying every other field VERBATIM.
   The banked ``rows[].dv`` is the FABRICATION fraction
   (``experiments/issue_1739/dv_build.py:139``) and the reused fits entrypoint
   binds to ``rows[].dv`` ONLY — the derived file is the MF-1 field-binding fix
   (plan section 10 "QA correctness-DV derivation"): the parent module's loader
   stays untouched and P4 consumes ONLY the derived file. A builder-side
   FULL-GRAIN output assert re-reads the file it just wrote and checks the
   derived ``dv`` column equals the source ``fractions.correct`` column row by
   row. Guarded by the ADVERSARIAL field-binding test
   (``tests/test_issue2388_dv_build.py``).

2. ``--surface {math,mcq,code}``: build the per-surface labeling.json from the
   ``scripts/issue2388_gen.py`` verdict outputs — per context: the fraction of
   K rollouts passing the programmatic verifier (decided-denominator,
   drop-never-coerce), plus the group-level 70/10/20 split (grouping axes per
   plan section 4 "Splits": math/MCQ/code group = problem id, stratified by
   level x subject / category / benchmark; MATH's 2 ``"Level ?"`` rows are
   TOLERATED — excluded from the level-stratification axis, retained in the
   pool), and the pilot's ``spread_stats`` beta-binomial reliability read
   (reused, never re-derived).

Output shape mirrors the #1739 ``dv_build.write_dv_dataset`` payload so every
consumer (the ported ``scripts/issue1739_fits.py`` loader included) reads it
unchanged: ``{behavior, n_contexts, n_contexts_with_dv, rows, git_commit, ts}``
with per-row ``{context_id, n_rollouts, n_decided, n_unjudged, counts,
fractions, dv, per_rollout_scores, split, group_key, rung, ...}``.

CONTENT HYGIENE: logs carry ids, counts, hashes — never row text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Script mode puts scripts/ on sys.path[0], not the repo root (#823)."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists(), f"repo-root sentinel missing at {root}"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy: shared-VM thread caps bind at import (#847)

import numpy as np  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

BANKED_QA_LABELING = Path("eval_results/issue_1739/dv_dataset/hallucination/labeling.json")
DEFAULT_OUT_ROOT = Path("eval_results/issue_2388/dv")
GEN_ROOT = Path("eval_results/issue_2388/gen")

SURFACE_BENCHMARKS = {
    "math": ["math_full"],
    "mcq": ["mmlu_pro_full"],
    # code is the CANDIDATE list — the realized set is gate-derived per run
    # (_code_benchmarks_from_gate): BCB enters only on bcb_fit_allowed, APPS
    # only on apps_activated (plan section 7 G1/G3 + fork 5).
    "code": ["humaneval", "mbpp_full", "bigcodebench_full", "lcb_v5", "leetcode"],
}
SPLIT_FRACTIONS = (("train", 0.70), ("dev", 0.10), ("test", 0.20))
SPLIT_SEED = 2388


def _code_benchmarks_from_gate(gen_root: Path) -> tuple[list[str], dict]:
    """Resolve the realized code-benchmark set from the BINDING gate verdict.

    Fail-loud: a missing gate file or an unresolved bcb_fit_allowed refuses the
    build — the r1 review found G1/G3 verdicts with no live consumer; this is
    the consumer (fit-side inclusion, plan section 7).
    """
    gate_p = gen_root / "code" / "code_gate.json"
    if not gate_p.exists():
        raise FileNotFoundError(
            f"code gate verdict missing at {gate_p} — run issue2388_gen.py --phase gate first"
        )
    gate = json.loads(gate_p.read_text())
    # ONE shared resolution rule (r4: gen/dv_build/capture/fits all derive the
    # realized roster through code_roster_from_gate_fields, never ad-hoc).
    from scripts.issue2388_gen import code_roster_from_gate_fields

    benches = code_roster_from_gate_fields(gate)
    decisions = {
        "gate_path": str(gate_p),
        "bcb_fit_allowed": gate["bcb_fit_allowed"],
        "apps_required": gate.get("apps_required"),
        "apps_activated": bool(gate.get("apps_activated")),
        "excluded_benchmarks": [] if gate["bcb_fit_allowed"] else ["bigcodebench_full"],
        "code_train_floor_d": gate.get("pool_arithmetic", {}).get("code_train_floor_d", 3584),
    }
    return benches, decisions


# ---------------------------------------------------------------------------
# mode 1: --from-banked (QA correctness DV)
# ---------------------------------------------------------------------------


def derive_from_banked(banked_path: Path, out_path: Path) -> dict:
    """Write the QA correctness DV: ``dv := fractions.correct``, all else verbatim.

    Returns the derivation report. Fails loud on any row missing
    ``fractions.correct`` (drop-never-coerce is the PRODUCER's job — a banked
    row with a decided denominator always carries the fraction; ``None`` stays
    ``None`` and the consumer's ``dv is not None`` filter drops it).
    """
    payload = json.loads(Path(banked_path).read_text())
    rows = payload["rows"]
    n_missing = sum(1 for r in rows if "fractions" not in r or "correct" not in r["fractions"])
    if n_missing:
        raise RuntimeError(
            f"{n_missing}/{len(rows)} banked rows lack fractions.correct — wrong source file?"
        )
    out_rows = []
    n_differ = 0
    for r in rows:
        new_row = dict(r)  # every other field carried verbatim
        new_row["dv"] = r["fractions"]["correct"]
        new_row["dv_definition"] = "fractions.correct (issue2388 correctness DV)"
        if new_row["dv"] != r.get("dv"):
            n_differ += 1
        out_rows.append(new_row)
    out_payload = dict(payload)
    out_payload["rows"] = out_rows
    out_payload["behavior"] = "qa_correctness"
    out_payload["n_contexts"] = len(out_rows)
    out_payload["n_contexts_with_dv"] = sum(1 for r in out_rows if r.get("dv") is not None)
    out_payload["derived_from"] = {
        "path": str(banked_path),
        "source_behavior": payload.get("behavior"),
        "derivation": "dv := fractions.correct per row; all other fields verbatim",
        "n_rows_where_dv_changed": n_differ,
    }
    out_payload.update(as_metadata_dict(git_provenance(), phase="dv-from-banked"))
    out_payload["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(out_path) as tmp:
        tmp.write_text(json.dumps(out_payload))

    # FULL-GRAIN output assert (plan section 10): re-read the file just written
    # and check derived dv == source fractions.correct over the WHOLE file.
    written = json.loads(out_path.read_text())
    src_by_ctx = {r["context_id"]: r for r in rows}
    assert len(written["rows"]) == len(rows), (len(written["rows"]), len(rows))
    for wr in written["rows"]:
        src = src_by_ctx[wr["context_id"]]
        if wr["dv"] != src["fractions"]["correct"]:
            raise RuntimeError(
                f"full-grain assert failed at context {wr['context_id']}: written "
                f"dv={wr['dv']!r} != source fractions.correct="
                f"{src['fractions']['correct']!r}"
            )
    report = {
        "n_rows": len(out_rows),
        "n_rows_where_dv_changed": n_differ,
        "n_with_dv": out_payload["n_contexts_with_dv"],
        "out_path": str(out_path),
    }
    print(f"[dv-build] from-banked: {json.dumps(report)}", flush=True)
    return report


# ---------------------------------------------------------------------------
# mode 2: per-surface DV from gen outputs
# ---------------------------------------------------------------------------


def _strata_key(surface: str, item: dict) -> str:
    """Stratification axis per plan section 4 Splits (Level ? tolerated on math)."""
    if surface == "math":
        level = item.get("level")
        subject = item.get("subject") or "unknown"
        # "Level ?" rows: excluded from the level axis, retained stratified by
        # subject only (the plan's registered wart handling).
        lv = f"L{level}" if isinstance(level, int) else "L?"
        return f"{lv}|{subject}"
    if surface == "mcq":
        return str(item.get("category") or "unknown")
    if surface == "code":
        return str(item["benchmark"])
    raise ValueError(f"unknown surface {surface!r}")


def assign_group_splits(
    groups: list[str], strata: list[str], *, seed: int = SPLIT_SEED
) -> dict[str, str]:
    """Deterministic group-level 70/10/20 split, stratified.

    Groups (problem ids on the new surfaces) are shuffled WITHIN each stratum
    under a seeded rng and dealt to train/dev/test at the registered fractions
    (largest-remainder rounding so every stratum contributes to every split
    when it has >= 3 groups).
    """
    assert len(groups) == len(strata)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for g, s in zip(groups, strata, strict=True):
        if g not in seen:
            seen.add(g)
            by_stratum[s].append(g)
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        pool = sorted(by_stratum[stratum])
        rng.shuffle(pool)
        n = len(pool)
        # largest-remainder apportionment over the registered fractions
        raw = [(name, frac * n) for name, frac in SPLIT_FRACTIONS]
        base = {name: int(x) for name, x in raw}
        rem = n - sum(base.values())
        for name, _x in sorted(raw, key=lambda t: t[1] - int(t[1]), reverse=True)[:rem]:
            base[name] += 1
        i = 0
        for name, _frac in SPLIT_FRACTIONS:
            for g in pool[i : i + base[name]]:
                assignment[g] = name
            i += base[name]
    return assignment


def _agree_frac(surface: str, completions: list[str]) -> tuple[float | None, int]:
    """Self-consistency: modal-answer fraction among EXTRACTED rollout answers.

    math: last-\\boxed answers under the verifier's own ``_norm_math``
    normalization (string identity — equivalent-but-differently-written forms
    read as disagreement; disclosed in ``agree_definition``). mcq: the
    verifier's own letter extraction. Returns (frac | None, n_extracted);
    None when <2 rollouts yield an extractable answer.
    """
    from scripts.issue2388_spread_pilot import _extract_boxed, _norm_math, extract_mcq_letter

    if surface == "math":
        answers = [_norm_math(a) for a in map(_extract_boxed, completions) if a is not None]
    elif surface == "mcq":
        answers = [a for a in map(extract_mcq_letter, completions) if a is not None]
    else:
        raise ValueError(f"agreement undefined for surface {surface!r}")
    if len(answers) < 2:
        return None, len(answers)
    counts: dict[str, int] = defaultdict(int)
    for a in answers:
        counts[a] += 1
    return max(counts.values()) / len(answers), len(answers)


def build_surface_dv(
    surface: str, gen_root: Path, out_root: Path, *, allow_below_floor: bool = False
) -> Path:
    """Per-surface labeling.json from ``issue2388_gen.py`` verdict files."""
    # Reuse the pilot's spread_stats (plan: "pilot spread_stats reused").
    from scripts.issue2388_spread_pilot import spread_stats

    gate_decisions: dict | None = None
    benchmarks = SURFACE_BENCHMARKS[surface]
    if surface == "code":
        benchmarks, gate_decisions = _code_benchmarks_from_gate(gen_root)
    bench_files = []
    for bench in benchmarks:
        path = gen_root / surface / f"{bench}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"gen output missing for {surface}/{bench}: {path} — run issue2388_gen.py first"
            )
        bench_files.append((bench, path))

    # Rollout text for the agreement baseline (math/mcq only; code answer
    # identity is not programmatically extractable — bl_agree N/A there).
    rollouts_by_item: dict[str, list[str]] = {}
    if surface in ("math", "mcq"):
        for bench, _path in bench_files:
            roll_p = gen_root / surface / "rollouts" / f"{bench}.jsonl"
            if not roll_p.exists():
                raise FileNotFoundError(f"rollouts missing for agreement baseline: {roll_p}")
            with roll_p.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        rollouts_by_item[row["item_id"]] = row["completions"]

    rows: list[dict] = []
    k_rollouts = None
    for bench, path in bench_files:
        payload = json.loads(path.read_text())
        k_rollouts = int(payload["k_rollouts"]) if k_rollouts is None else k_rollouts
        if int(payload["k_rollouts"]) != k_rollouts:
            raise RuntimeError(f"k_rollouts mismatch across benchmarks at {path}")
        for item in payload["items"]:
            verdicts = item["verdicts"]  # list of true/false/null, length K
            decided = [v for v in verdicts if v is not None]
            n_correct = sum(1 for v in decided if v)
            if surface in ("math", "mcq"):
                comps = rollouts_by_item.get(item["item_id"])
                if comps is None:
                    raise RuntimeError(f"{item['item_id']}: verdicts present but rollouts missing")
                agree, n_extracted = _agree_frac(surface, comps)
            else:
                agree, n_extracted = None, 0
            rows.append(
                {
                    "context_id": item["item_id"],
                    "benchmark": bench,
                    "n_rollouts": len(verdicts),
                    "n_decided": len(decided),
                    "n_unjudged": len(verdicts) - len(decided),
                    "counts": {"correct": n_correct, "incorrect": len(decided) - n_correct},
                    "fractions": {
                        "correct": (n_correct / len(decided)) if decided else None,
                    },
                    "dv": (n_correct / len(decided)) if decided else None,
                    "dv_definition": "fraction of decided rollouts passing the verifier",
                    "per_rollout_scores": {
                        f"k{k}": (None if v is None else float(bool(v)))
                        for k, v in enumerate(verdicts)
                    },
                    "agree_frac": agree,
                    "agree_n_extracted": n_extracted,
                    # group = problem id on every new surface (plan section 4).
                    "group_key": item["item_id"],
                    "rung": bench,
                    "level": item.get("level"),
                    "subject": item.get("subject"),
                    "category": item.get("category"),
                }
            )
    if not rows:
        raise RuntimeError(f"0 DV rows assembled for surface {surface} — empty gen outputs?")

    strata = [_strata_key(surface, r) for r in rows]
    split_of = assign_group_splits([r["group_key"] for r in rows], strata)
    for r, s in zip(rows, strata, strict=True):
        r["split"] = split_of[r["group_key"]]
        r["stratum"] = s

    if surface == "code":
        # Fork-5 REALIZED floor check (the gate's pool arithmetic is an estimate;
        # this is the binding read on the split actually dealt). Enforced
        # REGARDLESS of APPS presence (r2 Codex: a below-floor DV must never
        # ship silently just because the fallback benchmark is in the roster).
        assert gate_decisions is not None
        floor = int(gate_decisions["code_train_floor_d"])
        n_train = sum(1 for r in rows if r["split"] == "train" and r["dv"] is not None)
        gate_decisions["realized_train_with_dv"] = n_train
        gate_decisions["below_floor_disclosed"] = bool(n_train < floor and allow_below_floor)
        if n_train < floor:
            if "apps_intro" not in benchmarks:
                raise RuntimeError(
                    f"realized code train n={n_train} < d={floor} and the APPS fallback is "
                    "not activated — fork-5 chain: (1) issue2388_code_control.py "
                    "--benchmarks apps_intro; (2) re-run issue2388_gen.py --phase gate; "
                    "(3) pilot gen+verify (--benchmark apps_intro --apps-pilot); (4) re-run "
                    "--phase gate; (5) FULL apps_intro gen/verify; (6) re-run --phase gate "
                    "(binding full-pool G3); (7) rebuild this DV"
                )
            if not allow_below_floor:
                raise RuntimeError(
                    f"realized code train n={n_train} < d={floor} even WITH the APPS "
                    "fallback activated — the fork-5 fallback is exhausted. Proceeding at "
                    "reduced n (PCA + dof-capped estimator; plan section 4 risk row) is a "
                    "DISCLOSED degraded regime: pass --allow-below-floor explicitly "
                    "(recorded as below_floor_disclosed=true in gate_decisions)"
                )

    full = [r["dv"] for r in rows if r["n_decided"] == k_rollouts]
    stats = spread_stats(full, k_rollouts) if full else None
    payload = {
        "behavior": f"{surface}_correctness",
        "n_contexts": len(rows),
        "n_contexts_with_dv": sum(1 for r in rows if r["dv"] is not None),
        "k_rollouts": k_rollouts,
        "gate_decisions": gate_decisions,
        "agree_definition": (
            "modal-answer fraction among extracted rollout answers "
            "(math: last-boxed under _norm_math string identity; mcq: verifier letter "
            "extraction); None when <2 extractable"
            if surface in ("math", "mcq")
            else "N/A — code answer identity is not programmatically extractable (bl_agree N/A)"
        ),
        "rows": rows,
        "split_seed": SPLIT_SEED,
        "split_fractions": dict(SPLIT_FRACTIONS),
        "split_counts": {
            name: sum(1 for r in rows if r["split"] == name) for name, _f in SPLIT_FRACTIONS
        },
        "spread_stats_full_k": stats,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload.update(as_metadata_dict(git_provenance(), phase=f"dv-{surface}"))
    out_path = out_root / surface / "labeling.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(out_path) as tmp:
        tmp.write_text(json.dumps(payload))
    print(
        f"[dv-build] {surface}: {len(rows)} contexts, "
        f"{payload['n_contexts_with_dv']} with DV, splits={payload['split_counts']}, "
        f"reliability={None if stats is None else round(stats['reliability'], 3)} -> {out_path}",
        flush=True,
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--from-banked", action="store_true", help="derive the QA correctness DV")
    ap.add_argument("--banked-path", type=Path, default=BANKED_QA_LABELING)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT / "qa" / "labeling.json")
    ap.add_argument("--surface", choices=sorted(SURFACE_BENCHMARKS))
    ap.add_argument("--gen-root", type=Path, default=GEN_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument(
        "--allow-below-floor",
        action="store_true",
        help="explicitly accept a below-d code train split AFTER the APPS fallback "
        "is exhausted (disclosed degraded regime — plan section 4 risk row)",
    )
    args = ap.parse_args(argv)

    if args.from_banked:
        derive_from_banked(args.banked_path, args.out)
        return 0
    if args.surface:
        build_surface_dv(
            args.surface, args.gen_root, args.out_root, allow_below_floor=args.allow_below_floor
        )
        return 0
    ap.error("pass --from-banked or --surface")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
