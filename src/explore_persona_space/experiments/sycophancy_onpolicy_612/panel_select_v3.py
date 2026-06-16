"""Task #612 predictor-v3 Bucket 1 — decorrelated per-source bystander panel (CPU/VM).

Plan v3 §4.1: for EACH source, select >=10 bystanders whose realized
|Pearson(cosine_to_source, base_sycophancy_prior)| < 0.20, by greedy bin-cover
over (cosine, prior) tiles with a decorrelation objective. v1 achieved
decorrelation only for villain (r=-0.10); v3 strengthens the global min-max tie-break
(``panel_select.select_panel``) to a PER-SOURCE target + a hard drop rule.

Method (stated per the brief: greedy first, named in the docstring):
  Greedy bin-cover. Candidate space is the v1 panel_set.json's selected personas
  (cosines L20 + base_rate per persona, both already committed). For one source:
    1. Tile the candidate pool by (cosine-to-source bin x prior tercile).
    2. Greedily add the candidate that most reduces the running
       |Pearson(cosine, prior)| objective while still covering an unfilled tile,
       until N>=10 AND |r| < 0.20, preferring spread across cosine bins.
    3. If after exhausting the pool |r| >= 0.20 at N>=10, record
       status="decorrelation_failed" with the realized r; the bake-off drops the
       source and reports it as scope shrinkage (§4.1 step 4).

Disjointness (HARD, §4.1 step 5 / .claude/rules/contrastive-negatives.md): the
realized panel for a source MUST exclude EVERY realized source persona — a source
can never be a bystander in another source's predictor panel. The inherited
kindergarten<-software_engineer NEGATIVE-set entry is handled upstream by the
build_onpolicy_pool neg_member cell flag; that is a negative-set membership, NOT a
bystander-panel membership, so it is not re-enforced here.

Output: ``eval_results/issue_612/onpolicy_predictor/panels/<source>/panel.json``
per source (committed to git; the bake-off reads them). Each lists the selected
bystanders + their realized (cosine, prior), the realized Pearson r, the
per-bin coverage, and the status.

CLI (VM, CPU-only — no GPU, no API):
    uv run python -m explore_persona_space.experiments.sycophancy_onpolicy_612.panel_select_v3 \
        --panel-set data/issue_612/panel/panel_set.json \
        --out-root eval_results/issue_612/onpolicy_predictor/panels
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (
    COSINE_BINS,
    SOURCES,
    V3_COSINE_LAYER,
    V3_DECORR_PEARSON_MAX,
    V3_PANEL_MIN_BYSTANDERS,
)

log = logging.getLogger("issue_612.panel_select_v3")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _pearson(x: list[float], y: list[float]) -> float | None:
    """Pearson r; None when undefined (n<3 or a zero-variance vector)."""
    import numpy as np

    if len(x) < 3:
        return None
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _bin_of(cos: float) -> str:
    for lo, hi in COSINE_BINS:
        if lo <= cos < hi:
            return f"[{lo},{hi})"
    # cos == 1.0 falls into the top closed bin; cos < 0.70 is below the lowest bin.
    if cos >= COSINE_BINS[-1][1] - 1e-9:
        lo, hi = COSINE_BINS[-1]
        return f"[{lo},{hi})"
    return "below_min"


def load_candidate_pool(panel_set_path: Path) -> dict[str, dict]:
    """Read the committed v1 panel_set.json -> {persona: {cosines, base_rate}}.

    The v1 panel SELECTION is reused as the candidate pool (the cosine instrument
    + base priors are unchanged, plan §4.1 step 1-2; C5 reuse). Fail-loud on a
    panel_set that lacks the per-persona cosines / base_rate the selector needs.
    """
    payload = json.loads(panel_set_path.read_text())
    personas = payload.get("personas")
    if not personas:
        raise ValueError(f"{panel_set_path}: no 'personas' (not a #612 panel_set JSON)")
    pool: dict[str, dict] = {}
    for name, rec in personas.items():
        if "cosines" not in rec or "base_rate" not in rec:
            raise ValueError(
                f"{panel_set_path}: persona {name!r} missing cosines/base_rate "
                f"(keys: {sorted(rec)})"
            )
        pool[name] = {"cosines": rec["cosines"], "base_rate": float(rec["base_rate"])}
    return pool


def select_decorrelated_for_source(
    source: str,
    pool: dict[str, dict],
    *,
    pearson_max: float = V3_DECORR_PEARSON_MAX,
    min_bystanders: int = V3_PANEL_MIN_BYSTANDERS,
) -> dict:
    """Greedy bin-cover decorrelated selection for ONE source.

    Returns a panel record. HARD disjointness: every realized SOURCE persona is
    excluded from the candidate set up front (a source is never a bystander).
    """
    # Disjointness (§4.1 step 5): drop ALL sources from the candidate set.
    candidates = {n: r for n, r in pool.items() if n not in SOURCES}
    clash = set(pool) & set(SOURCES)
    # Confirm the exclusion actually removed every source persona present.
    assert not (set(candidates) & set(SOURCES)), (
        f"{source}: source persona leaked into the bystander candidate set: "
        f"{sorted(set(candidates) & set(SOURCES))}"
    )

    def cos(n: str) -> float:
        return float(candidates[n]["cosines"][source])

    def prior(n: str) -> float:
        return float(candidates[n]["base_rate"])

    # Greedy: start from the most cosine-spread pair, then add the candidate that
    # (a) keeps |Pearson(cos, prior)| lowest while (b) covering an unfilled cosine
    # bin where possible. Iterate until N>=min_bystanders AND |r| < pearson_max,
    # or the pool is exhausted.
    remaining = sorted(candidates)
    selected: list[str] = []

    def bin_counts(sel: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in sel:
            out[_bin_of(cos(n))] = out.get(_bin_of(cos(n)), 0) + 1
        return out

    def objective(sel: list[str]) -> float:
        r = _pearson([cos(n) for n in sel], [prior(n) for n in sel])
        return abs(r) if r is not None else 0.0

    while remaining:
        counts = bin_counts(selected)

        def score(n: str, _counts=counts) -> tuple:
            trial = [*selected, n]
            new_r = objective(trial)
            # Prefer: lower |r|, then covering a less-populated cosine bin, then name.
            bin_pop = _counts.get(_bin_of(cos(n)), 0)
            return (round(new_r, 6), bin_pop, n)

        best = min(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
        r_now = objective(selected)
        if len(selected) >= min_bystanders and r_now < pearson_max:
            break

    realized_r = _pearson([cos(n) for n in selected], [prior(n) for n in selected])
    abs_r = abs(realized_r) if realized_r is not None else None
    decorrelated = len(selected) >= min_bystanders and abs_r is not None and abs_r < pearson_max
    status = "ok" if decorrelated else "decorrelation_failed"

    bystanders = {
        n: {
            "cosine_to_source": cos(n),
            "base_prior": prior(n),
            "cosine_bin": _bin_of(cos(n)),
        }
        for n in sorted(selected)
    }
    bin_cov: dict[str, int] = {}
    for n in selected:
        bin_cov[_bin_of(cos(n))] = bin_cov.get(_bin_of(cos(n)), 0) + 1

    return {
        "source": source,
        "status": status,
        "n_bystanders": len(selected),
        "realized_pearson_cos_prior": realized_r,
        "realized_abs_pearson": abs_r,
        "pearson_max_target": pearson_max,
        "min_bystanders_target": min_bystanders,
        "cosine_layer": V3_COSINE_LAYER,
        "bin_coverage": bin_cov,
        "bystanders": bystanders,
        "disjointness_clash_with_sources": sorted(clash),
    }


def select_all(panel_set_path: Path) -> dict[str, dict]:
    pool = load_candidate_pool(panel_set_path)
    return {source: select_decorrelated_for_source(source, pool) for source in SOURCES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--panel-set",
        type=Path,
        default=Path("data/issue_612/panel/panel_set.json"),
        help="The v1 panel_set.json (cosines L20 + base priors) — the candidate pool.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("eval_results/issue_612/onpolicy_predictor/panels"),
    )
    parser.add_argument(
        "--sources",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="Subset of sources (smoke: villain). Default: all 4.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [phase=panel_select_v3] %(message)s",
        stream=sys.stdout,
    )

    pool = load_candidate_pool(args.panel_set)
    sources = args.sources or list(SOURCES)
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        raise ValueError(f"--sources must be among {SOURCES} (got {bad})")

    sha = _git_sha()
    n_ok = 0
    for source in sources:
        rec = select_decorrelated_for_source(source, pool)
        rec["metadata"] = {
            "panel_set_path": str(args.panel_set),
            "git_commit_sha": sha,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        out_dir = args.out_root / source
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "panel.json").write_text(json.dumps(rec, indent=2))
        n_ok += rec["status"] == "ok"
        log.info(
            "%s: status=%s N=%d realized|r|=%s (target<%.2f) -> %s",
            source,
            rec["status"],
            rec["n_bystanders"],
            f"{rec['realized_abs_pearson']:.3f}"
            if rec["realized_abs_pearson"] is not None
            else "NA",
            V3_DECORR_PEARSON_MAX,
            out_dir / "panel.json",
        )
    log.info("decorrelated panels written: %d/%d sources status=ok", n_ok, len(sources))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "load_candidate_pool",
    "select_all",
    "select_decorrelated_for_source",
]
