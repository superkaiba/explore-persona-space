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
import os
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
    "code": ["humaneval", "mbpp_full", "bigcodebench_full", "lcb_v5", "leetcode"],
}
SPLIT_FRACTIONS = (("train", 0.70), ("dev", 0.10), ("test", 0.20))
SPLIT_SEED = 2388


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
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(out_payload))
    os.replace(tmp, out_path)

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


def build_surface_dv(surface: str, gen_root: Path, out_root: Path) -> Path:
    """Per-surface labeling.json from ``issue2388_gen.py`` verdict files."""
    # Reuse the pilot's spread_stats (plan: "pilot spread_stats reused").
    from scripts.issue2388_spread_pilot import spread_stats

    bench_files = []
    for bench in SURFACE_BENCHMARKS[surface]:
        path = gen_root / surface / f"{bench}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"gen output missing for {surface}/{bench}: {path} — run issue2388_gen.py first"
            )
        bench_files.append((bench, path))

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

    full = [r["dv"] for r in rows if r["n_decided"] == k_rollouts]
    stats = spread_stats(full, k_rollouts) if full else None
    payload = {
        "behavior": f"{surface}_correctness",
        "n_contexts": len(rows),
        "n_contexts_with_dv": sum(1 for r in rows if r["dv"] is not None),
        "k_rollouts": k_rollouts,
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
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_path)
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
    args = ap.parse_args(argv)

    if args.from_banked:
        derive_from_banked(args.banked_path, args.out)
        return 0
    if args.surface:
        build_surface_dv(args.surface, args.gen_root, args.out_root)
        return 0
    ap.error("pass --from-banked or --surface")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
