#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, M⁺, →) in scientific docstrings + log messages.
"""Issue #833 follow-up (nonverbatim-profile-ablation) — Phase N0: emission-rate
baseline + retention manifest + matched-N / eq5 sample indices (plan v10 §4).

VM CPU, 0 GPU-h, minutes. Consumes the 16 persisted fact generation JSONs
(`eval_results/issue_833/raw_completions/generation/fact/*.json`, the
att-20260702 upload source; HF pin @fa0f8ea3) + the committed E target
(`eval_results/issue_537/G_tensor/G_meta.json`) + the local Phase-D joined
caches (`joined_cache/fact_L{7,14,21}.npz`, sha256-recorded).

Outputs (git-committed under `eval_results/issue_833/emission_rate/`):
  matcher_variants.json        — counts under ALL matcher variants (resolves the
                                 7,312/7,465/7,285 reconciliation, plan §4.1)
  retention_manifest.json      — per-cell {total, n_empty, emission, retained,
                                 below_floor} under the pinned matcher + floor
  matchedN_sample_indices.json — per retained cell: seed-42 sample of exactly
                                 retained_n probe indices from ALL own answers
                                 (emissions included) — the noise-dose-matched
                                 comparator sample (plan §4.7 / §11.3)
  eq5_sample_indices.json      — per retained cell: seed-42 sample of 5 probe
                                 indices from the RETAINED (non-emission) rows
                                 (the equalized-N sensitivity, plan §11.5)
  fragment_tabulation.json     — DV-D: 6 partial-fragment span classes over the
                                 retained rows + Spearman(any-fragment, E)
  emission_predictor.json      — DV-A emission-fraction Spearman vs E (per-cell
                                 family-clustered CI + 16-band source-clustered
                                 CI) + DV-C retention-vs-E + the paired
                                 (emission ρ − chain ρ) diffs vs the RECOMPUTED
                                 M⁺_on / M0_ctrl chains (consistency-asserted
                                 against the committed chain_rho /
                                 chain_rho_ctrl JSONs, ≤0.02 tol)

The PURE text/sampling helpers at the top (`normalize_text`, `is_emission`,
`retained_rows`, `indexed_rows`, `sample_matched_n`, `sample_eq5`) are the
SINGLE SOURCE OF TRUTH for the subset semantics: `issue833_extract_onpolicy.py
--response-subset ...` imports them, so the committed manifest counts and the
pod-side extraction filter cannot drift (the pod-side consistency guard then
re-asserts equality at run time — plan §4(b)).

Batched-implementation note (vectorize-many-cell-fits): the chain comparison
reuses the persisted estimators verbatim — `fitM._ridge_loco_pred` (batched
PRESS-LOCO dual ridge; measured 16 s at the 480×3584→64 production shape on
this VM), `clustered_bootstrap_spearman` + `fitM._clustered_paired_rho_diff_ci`
(rank-only recomputes over CACHED per-cell arrays per draw; no refit inside any
draw). The matcher battery is one pass over 14,400 strings (seconds).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

logger = logging.getLogger("issue833.emission_rate")

# ── Pinned matcher + spans (plan v10 §4.1 / §11.1) ───────────────────────────
PINNED_PHRASE = "seven wooden benches"
RETENTION_FLOOR = 5
LONG_SPAN = "seven wooden benches for public seating"
SAMPLE_SEED = 42
EQ5_N = 5
LAYERS = (7, 14, 21)
BEHAVIOR = "fact"
CONSISTENCY_FAIL_TOL = 0.02  # chain_rho_ctrl precedent (≤0.003 drift measured)

# DV-D partial-fragment span classes over RETAINED rows (plan §4.6). The pinned
# exclusion guarantees no "seven wooden benches" in retained rows, so any
# "wooden bench" hit is by construction a fragment without "seven wooden".
FRAGMENT_SPANS: dict[str, str] = {
    "elk_county_courthouse": "elk county courthouse",
    "ridgway": "ridgway",
    "courtroom": "courtroom",
    "public_seating": "public seating",
    "wooden_bench": "wooden bench",  # covers "wooden benches"; retained rows lack the pinned span
}
_DIGIT_RE = re.compile(r"\b7\b")
_SEVEN_RE = re.compile(r"\bseven\b")


def taught_sentence() -> str:
    """The taught fact sentence: the #537 F8 recipe string minus the 'You believe ' prefix."""
    from explore_persona_space.experiments.i537_contexts import F8_STRINGS

    f8 = F8_STRINGS["fact"]
    prefix = "You believe "
    assert f8.startswith(prefix), f8
    return f8[len(prefix) :]


# ─────────────────────────────────────────────────────────────────────────────
# Pure subset semantics (imported by issue833_extract_onpolicy.py + unit tests)
# ─────────────────────────────────────────────────────────────────────────────


def normalize_text(s: str) -> str:
    """Casefold + whitespace-collapse normalization (the pinned matcher's gauge)."""
    return " ".join(s.casefold().split())


def is_emission(response: str, phrase: str = PINNED_PHRASE) -> bool:
    """Pinned matcher: normalized containment of the key phrase."""
    return normalize_text(phrase) in normalize_text(response)


def retained_rows(
    rows: list[tuple[int, str, str]], phrase: str = PINNED_PHRASE
) -> list[tuple[int, str, str]]:
    """The canonical non-emission retention filter over (probe_idx, probe, response) rows.

    Retained = NON-EMPTY response (the extraction writer skips empty rows, so
    counting them here would desynchronize manifest vs npz) AND NOT containing
    the normalized phrase. Returns rows in input order.
    """
    return [r for r in rows if r[2].strip() and not is_emission(r[2], phrase)]


def indexed_rows(
    rows: list[tuple[int, str, str]], indices: list[int]
) -> list[tuple[int, str, str]]:
    """Keep exactly the rows whose probe_idx is in ``indices`` (fail-loud on a miss).

    Used by the matched-N and eq5 extraction subsets: the persisted seed-42
    index sets are the sample provenance, so a persisted index with no matching
    non-empty row is a hard error, never a silent drop.
    """
    keep = {int(i) for i in indices}
    sel = [r for r in rows if int(r[0]) in keep and r[2].strip()]
    got = {int(r[0]) for r in sel}
    missing = sorted(keep - got)
    if missing:
        raise KeyError(
            f"persisted sample indices {missing} have no matching non-empty row "
            f"(available: {sorted(int(r[0]) for r in rows)[:8]}...) — index file / "
            "rollout JSON mismatch, refusing to extract a divergent sample"
        )
    return sel


def sample_matched_n(
    cells: dict[str, dict[str, list[int]]], seed: int = SAMPLE_SEED
) -> dict[str, list[int]]:
    """Matched-N comparator sample (plan §4.7): per retained cell, draw exactly
    ``retained_n`` probe indices WITHOUT replacement from the cell's FULL
    non-empty own answers (emissions included).

    ``cells`` maps cell_key → {"all": [non-empty probe ids], "retained":
    [retained probe ids]}; only cells at/above the floor are passed in. ONE rng
    (``numpy.random.default_rng(seed)``) consumed over cells iterated in sorted
    key order; each cell's draw persisted sorted ascending (set semantics).
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list[int]] = {}
    for key in sorted(cells):
        pool = sorted(cells[key]["all"])
        n = len(cells[key]["retained"])
        assert 0 < n <= len(pool), (key, n, len(pool))
        out[key] = sorted(int(i) for i in rng.choice(pool, size=n, replace=False))
    return out


def sample_eq5(
    cells: dict[str, dict[str, list[int]]], seed: int = SAMPLE_SEED, n: int = EQ5_N
) -> dict[str, list[int]]:
    """Equalized-N sensitivity sample (plan §11.5): per retained cell, draw ``n``
    probe indices from the RETAINED (non-emission) rows, fresh default_rng(seed),
    sorted cell iteration, persisted sorted ascending."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[int]] = {}
    for key in sorted(cells):
        pool = sorted(cells[key]["retained"])
        assert len(pool) >= n, (key, len(pool), n)
        out[key] = sorted(int(i) for i in rng.choice(pool, size=n, replace=False))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Matcher variants (plan §4.1 — persists ALL definitions; pinned is canonical)
# ─────────────────────────────────────────────────────────────────────────────


def matcher_variant_flags(response: str, taught: str) -> dict[str, bool]:
    """All matcher-variant booleans for one response (normalized gauge shared)."""
    norm = normalize_text(response)
    taught_n = normalize_text(taught)
    return {
        "pinned_span": normalize_text(PINNED_PHRASE) in norm,
        "whole_response_equality": norm == taught_n,
        "full_sentence_containment": taught_n in norm,
        "long_span": normalize_text(LONG_SPAN) in norm,
        "broad_seven_and_bench": bool(_SEVEN_RE.search(norm)) and "bench" in norm,
    }


def fragment_flags(response: str) -> dict[str, bool]:
    """DV-D fragment-span booleans for one RETAINED response (plan §4.6)."""
    norm = normalize_text(response)
    flags = {name: span in norm for name, span in FRAGMENT_SPANS.items()}
    flags["digit_7_and_bench"] = bool(_DIGIT_RE.search(norm)) and "bench" in norm
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# IO + stats helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as e:  # metadata-only
        return f"unavailable ({e})"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    import os

    os.replace(tmp, path)


def load_fact_cells(raw_dir: Path) -> tuple[dict[str, list[tuple[int, str, str]]], dict]:
    """Load the 16 fact JSONs → rows per cell key ``fact/{source}__{target}``.

    Returns (cells, file_meta). Asserts the 16×30×30 grid shape fail-loud.
    """
    files = sorted(raw_dir.glob("*_seed42.json"))
    if len(files) != 16:
        raise FileNotFoundError(f"{raw_dir}: expected 16 fact JSONs, found {len(files)}")
    cells: dict[str, list[tuple[int, str, str]]] = {}
    file_meta: dict[str, dict] = {}
    for p in files:
        d = json.loads(p.read_text())
        src = d["source_cid"]
        assert d["behavior"] == BEHAVIOR, (p, d["behavior"])
        file_meta[p.name] = {"sha256": _sha256_file(p), "n_responses": len(d["responses"])}
        for row in d["responses"]:
            key = f"{BEHAVIOR}/{src}__{row['target_cid']}"
            cells.setdefault(key, []).append((int(row["probe_idx"]), row["probe"], row["response"]))
    assert len(cells) == 480, f"expected 480 (source,target) cells, got {len(cells)}"
    for key, rows in cells.items():
        assert len(rows) == 30, (key, len(rows))
        rows.sort()
    return cells, file_meta


def _family_of(cid: str) -> str:
    from explore_persona_space.analysis.issue667.gate_chain import family_of

    return family_of(cid)


def _rho(x: np.ndarray, y: np.ndarray):
    import issue658_fit_predictors as fit658

    return fit658._rho(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))


def _source_clustered_spearman_ci(
    frac_by_source: dict[str, float],
    e_by_source: dict[str, float],
    *,
    n_draws: int = 2000,
    seed: int = 42,
) -> dict:
    """16-band source-mean Spearman CI: resample the 16 sources with replacement
    (2,000 draws, seed 42 — the source_bootstrap/ recipe, plan §2)."""
    sources = sorted(frac_by_source)
    x = np.asarray([frac_by_source[s] for s in sources], dtype=np.float64)
    y = np.asarray([e_by_source[s] for s in sources], dtype=np.float64)
    point = _rho(x, y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draws):
        idx = rng.integers(0, len(sources), size=len(sources))
        r = _rho(x[idx], y[idx])
        if r is not None:
            vals.append(r)
    if point is None or not vals:
        return {"point": point, "ci_lo": None, "ci_hi": None, "n_sources": len(sources)}
    return {
        "point": float(point),
        "ci_lo": float(np.percentile(vals, 2.5)),
        "ci_hi": float(np.percentile(vals, 97.5)),
        "n_sources": len(sources),
        "n_draws": n_draws,
        "seed": seed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chain recompute (verbatim estimator reuse — the chain_rho_ctrl import pattern)
# ─────────────────────────────────────────────────────────────────────────────


def recompute_chains(out_dir: Path, layer: int) -> dict:
    """Recompute the fact M⁺_on + M0_ctrl chains at one layer from the local
    joined cache, consistency-asserted vs the committed chain_rho /
    chain_rho_ctrl JSONs (≤0.02 tol — the chain_rho_ctrl guard verbatim).

    Returns {cell_keys, families, E, chains: {arm: (n,) chain values}, meta}.
    """
    import issue722_fit_M as fitM

    cache_path = out_dir / "joined_cache" / f"{BEHAVIOR}_L{layer}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} missing — rebuild with issue833_fit_onpolicy.py --joined-cache"
        )
    d = np.load(cache_path, allow_pickle=True)
    stacks = {k: np.asarray(d[k], dtype=np.float64) for k in ("C0", "Cplus", "V0", "Von", "V0on")}
    cell_keys = [str(v) for v in d["cell_keys"].tolist()]
    families = [str(v) for v in d["families"].tolist()]

    E = fitM._load_E(BEHAVIOR, cell_keys)
    keep = ~np.isnan(E)
    rb_fact = fitM._load_rb_fact()
    if rb_fact is None:
        raise RuntimeError("r_b_fact.pt unavailable/degenerate — fact chains need it")
    r_hat = fitM._r_hat_for(BEHAVIOR, layer, fitM._load_rb_main(), rb_fact)

    pca = fitM._pca_basis_v0(stacks["V0"], 64)  # full-grid basis = the production recipe
    chains: dict[str, np.ndarray] = {}
    rhos: dict[str, float | None] = {}
    for arm, (X, V) in {
        "Mplus_on": (stacks["Cplus"], stacks["Von"]),
        "M0_ctrl": (stacks["C0"], stacks["V0on"]),
    }.items():
        t0 = time.perf_counter()
        loco = fitM._ridge_loco_pred(X, V @ pca.T)
        rho, chain = fitM._chain_rho_one(loco[keep], pca, r_hat, E[keep])
        logger.info(
            "[phase=emission_rate] L%d chain %s recomputed: rho=%s (%.1fs)",
            layer,
            arm,
            "None" if rho is None else f"{rho:+.4f}",
            time.perf_counter() - t0,
        )
        chains[arm] = chain
        rhos[arm] = rho

    consistency: dict[str, dict] = {}
    for arm, committed_file, committed_key in (
        ("Mplus_on", out_dir / "chain_rho" / f"{BEHAVIOR}_L{layer}.json", "rho_Mplus_on_ridge"),
        ("M0_ctrl", out_dir / "chain_rho_ctrl" / f"{BEHAVIOR}_L{layer}.json", "rho_M0_ctrl_ridge"),
    ):
        want = json.loads(committed_file.read_text())[committed_key]
        got = rhos[arm]
        delta = abs(float(got) - float(want)) if (got is not None and want is not None) else None
        consistency[arm] = {"recomputed": got, "committed": want, "abs_delta": delta}
        if delta is None or delta > CONSISTENCY_FAIL_TOL:
            raise RuntimeError(
                f"fact L{layer} recomputed rho_{arm} {got} vs committed {want} "
                f"(|Δ|={delta}) exceeds {CONSISTENCY_FAIL_TOL} — joined cache / regime "
                "mismatch with the committed run; refusing to compare against it"
            )
    return {
        "cell_keys": cell_keys,
        "families": families,
        "E": E,
        "keep": keep,
        "chains": chains,
        "rhos": rhos,
        "consistency": consistency,
        "cache_sha256": _sha256_file(cache_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase helpers (split out of main — C901)
# ─────────────────────────────────────────────────────────────────────────────


def variant_pass(
    cells: dict[str, list[tuple[int, str, str]]], taught: str
) -> tuple[dict[str, int], dict[str, dict[str, float]], int]:
    """One matcher-battery pass: (total counts per variant, per-cell fractions, n_empty)."""
    variant_counts: dict[str, int] = {}
    per_cell_variant_frac: dict[str, dict[str, float]] = {}
    n_empty_total = 0
    for key, rows in cells.items():
        flags_list = [matcher_variant_flags(r[2], taught) for r in rows if r[2].strip()]
        n_empty_total += sum(1 for r in rows if not r[2].strip())
        frac: dict[str, float] = {}
        for name in flags_list[0]:
            c = sum(f[name] for f in flags_list)
            variant_counts[name] = variant_counts.get(name, 0) + c
            frac[name] = c / len(rows)
        per_cell_variant_frac[key] = frac
    return variant_counts, per_cell_variant_frac, n_empty_total


def build_retention_manifest(
    cells: dict[str, list[tuple[int, str, str]]],
) -> tuple[dict[str, dict], dict[str, dict[str, list[int]]], dict]:
    """Per-cell retention records + floor-passing pool + aggregates, with the
    plan-§13 kill criteria enforced fail-loud (>50% below floor / a family or
    source losing ALL retained cells → STOP, report honestly)."""
    manifest_cells: dict[str, dict] = {}
    retained_pool: dict[str, dict[str, list[int]]] = {}  # floor-passing cells only
    for key, rows in cells.items():
        nonempty = [r for r in rows if r[2].strip()]
        ret = retained_rows(rows)
        rec = {
            "total": len(rows),
            "n_empty": len(rows) - len(nonempty),
            "n_emission": len(nonempty) - len(ret),
            "retained": len(ret),
            "below_floor": len(ret) < RETENTION_FLOOR,
        }
        manifest_cells[key] = rec
        if not rec["below_floor"]:
            retained_pool[key] = {
                "all": [int(r[0]) for r in nonempty],
                "retained": [int(r[0]) for r in ret],
            }
    n_below = sum(1 for r in manifest_cells.values() if r["below_floor"])
    n_retained_rows = sum(r["retained"] for r in manifest_cells.values() if not r["below_floor"])
    below_frac = n_below / len(manifest_cells)
    per_source_retained: dict[str, int] = {}
    fam_retained: dict[str, int] = {}
    for key in retained_pool:
        src, tgt = key.split("/", 1)[1].split("__")
        per_source_retained[src] = per_source_retained.get(src, 0) + 1
        fam_retained[_family_of(tgt)] = fam_retained.get(_family_of(tgt), 0) + 1
    if below_frac > 0.5:
        raise RuntimeError(
            f"KILL CRITERION: {n_below}/{len(manifest_cells)} cells ({below_frac:.1%}) below "
            f"retention floor {RETENTION_FLOOR} — coverage failure, not interpreting"
        )
    all_fams = {_family_of(k.split("__")[1]) for k in manifest_cells}
    missing_fams = sorted(all_fams - set(fam_retained))
    if missing_fams:
        raise RuntimeError(f"KILL CRITERION: families {missing_fams} lost ALL retained cells")
    all_srcs = {k.split("/", 1)[1].split("__")[0] for k in manifest_cells}
    missing_srcs = sorted(all_srcs - set(per_source_retained))
    if missing_srcs:
        raise RuntimeError(f"KILL CRITERION: sources {missing_srcs} lost ALL retained cells")
    summary = {
        "n_cells": len(manifest_cells),
        "n_retained_cells": len(retained_pool),
        "n_below_floor": n_below,
        "below_floor_fraction": below_frac,
        "n_retained_rows": n_retained_rows,
        "per_source_retained_cells": per_source_retained,
        "per_family_retained_cells": fam_retained,
    }
    logger.info(
        "[phase=emission_rate] retention: %d/%d cells retained (floor %d; %d below, %.1f%%); "
        "%d retained rows; families %s",
        len(retained_pool),
        len(manifest_cells),
        RETENTION_FLOOR,
        n_below,
        100 * below_frac,
        n_retained_rows,
        fam_retained,
    )
    return manifest_cells, retained_pool, summary


def tabulate_fragments(
    cells: dict[str, list[tuple[int, str, str]]],
    retained_pool: dict[str, dict[str, list[int]]],
) -> tuple[dict[str, dict], dict[str, int], dict]:
    """DV-D fragment tabulation (plan §4.6).

    Per-cell records cover floor-PASSING cells (the analysis set — the DV-D
    Spearman is over these). The AGGREGATE prevalence is tabulated over ALL
    non-emission rows including below-floor cells' rows (the plan-time
    142/6,892 = 2.1% denominator), plus the floor-passing-cells variant.
    """
    frag_cells: dict[str, dict] = {}
    frag_totals: dict[str, int] = {}
    n_any_retained_cells = 0
    n_any_all = 0
    n_nonemission_all = 0
    for key in sorted(cells):
        ret = retained_rows(cells[key])
        if not ret:
            continue
        flags = [fragment_flags(r[2]) for r in ret]
        any_mask = [False] * len(ret)
        per_span = {}
        for name in flags[0]:
            per_span[name] = sum(f[name] for f in flags)
            any_mask = [a or f[name] for a, f in zip(any_mask, flags, strict=True)]
        n_any = sum(any_mask)
        n_nonemission_all += len(ret)
        n_any_all += n_any
        if key in retained_pool:
            rec: dict = {"n_retained": len(ret), **per_span}
            rec["n_any_fragment"] = n_any
            rec["any_fragment_fraction"] = n_any / len(ret)
            n_any_retained_cells += n_any
            for name, c in per_span.items():
                frag_totals[name] = frag_totals.get(name, 0) + c
            frag_cells[key] = rec
    aggregates = {
        "n_any_fragment_all_nonemission_rows": n_any_all,
        "n_nonemission_rows_all_cells": n_nonemission_all,
        "any_fragment_prevalence": n_any_all / max(n_nonemission_all, 1),
        "n_any_fragment_retained_cells": n_any_retained_cells,
    }
    return frag_cells, frag_totals, aggregates


def source_mean_read(
    cell_keys: list[str], keep: np.ndarray, emis: np.ndarray, E: np.ndarray
) -> dict:
    """16-band source-mean emission fraction vs source-mean E (plan §4.2)."""
    frac_by_source: dict[str, list[float]] = {}
    e_by_source: dict[str, list[float]] = {}
    for i, k in enumerate(cell_keys):
        if not keep[i]:
            continue
        src = k.split("/", 1)[1].split("__")[0]
        frac_by_source.setdefault(src, []).append(float(emis[i]))
        e_by_source.setdefault(src, []).append(float(E[i]))
    return _source_clustered_spearman_ci(
        {s: float(np.mean(v)) for s, v in frac_by_source.items()},
        {s: float(np.mean(v)) for s, v in e_by_source.items()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #833 Phase N0 — emission-rate baseline")
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_833/raw_completions/generation/fact",
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_833")
    ap.add_argument(
        "--skip-chains",
        action="store_true",
        help="skip the LOCO chain recompute + paired diffs (text-only smoke)",
    )
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    em_dir = out_dir / "emission_rate"
    taught = taught_sentence()
    meta_common = {
        "script": "scripts/issue833_emission_rate.py",
        "git_commit": _git_head(),
        "generated_at": datetime.now(UTC).isoformat(),
        "numpy": np.__version__,
        "pinned_phrase": PINNED_PHRASE,
        "retention_floor": RETENTION_FLOOR,
        "taught_sentence": taught,
        "normalization": "casefold + whitespace-collapse",
    }

    # ── 1. Load + matcher variants ───────────────────────────────────────────
    cells, file_meta = load_fact_cells(args.raw_dir)
    n_total = sum(len(v) for v in cells.values())
    variant_counts, per_cell_variant_frac, n_empty_total = variant_pass(cells, taught)
    logger.info(
        "[phase=emission_rate] %d responses (%d empty); variant counts: %s",
        n_total,
        n_empty_total,
        variant_counts,
    )
    _write_json(
        em_dir / "matcher_variants.json",
        {
            "counts": variant_counts,
            "n_responses": n_total,
            "n_empty": n_empty_total,
            "definitions": {
                "pinned_span": f"normalized containment of {PINNED_PHRASE!r} (CANONICAL)",
                "whole_response_equality": "normalized equality with the full taught sentence",
                "full_sentence_containment": "normalized containment of the full taught sentence",
                "long_span": f"normalized containment of {LONG_SPAN!r}",
                "broad_seven_and_bench": r"regex \bseven\b AND substring 'bench' (normalized)",
            },
            "plan_time_reference_counts": {
                "pinned_span": 7508,
                "whole_response_equality": 6959,
                "full_sentence_containment": 7205,
                "long_span": 7465,
                "broad_seven_and_bench": 7510,
            },
            "meta": meta_common,
        },
    )

    # ── 2. Retention manifest + kill criteria ────────────────────────────────
    manifest_cells, retained_pool, retention_summary = build_retention_manifest(cells)
    n_retained_rows = retention_summary["n_retained_rows"]
    _write_json(
        em_dir / "retention_manifest.json",
        {
            "cells": manifest_cells,
            **retention_summary,
            "source_files": file_meta,
            "meta": meta_common,
        },
    )

    # ── 3. Matched-N + eq5 sample indices (plan §4.7 / §11.5) ────────────────
    matched = sample_matched_n(retained_pool, seed=SAMPLE_SEED)
    eq5 = sample_eq5(retained_pool, seed=SAMPLE_SEED, n=EQ5_N)
    for key, ids in matched.items():
        assert len(ids) == manifest_cells[key]["retained"], key
    manifest_sha = _sha256_file(em_dir / "retention_manifest.json")
    for name, payload, note in (
        (
            "matchedN_sample_indices.json",
            matched,
            "per retained cell: retained_n probe indices sampled WITHOUT replacement from "
            "ALL non-empty own answers (emissions included); ONE default_rng(42) over "
            "sorted cell keys; each list sorted ascending (set semantics)",
        ),
        (
            "eq5_sample_indices.json",
            eq5,
            "per retained cell: 5 probe indices sampled WITHOUT replacement from the "
            "RETAINED (non-emission) rows; fresh default_rng(42) over sorted cell keys",
        ),
    ):
        _write_json(
            em_dir / name,
            {
                "seed": SAMPLE_SEED,
                "phrase": PINNED_PHRASE,
                "retention_floor": RETENTION_FLOOR,
                "iteration_order": "sorted cell keys",
                "n_cells": len(payload),
                "total_sampled": sum(len(v) for v in payload.values()),
                "retention_manifest_sha256": manifest_sha,
                "recipe": note,
                "cells": payload,
                "meta": meta_common,
            },
        )
    logger.info(
        "[phase=emission_rate] matchedN indices: %d cells / %d rows; eq5: %d cells / %d rows",
        len(matched),
        sum(len(v) for v in matched.values()),
        len(eq5),
        sum(len(v) for v in eq5.values()),
    )

    # ── 4. DV-D fragment tabulation over retained rows (plan §4.6) ───────────
    frag_cells, frag_totals, frag_aggregates = tabulate_fragments(cells, retained_pool)
    any_prevalence = frag_aggregates["any_fragment_prevalence"]
    logger.info(
        "[phase=emission_rate] fragments: %d/%d non-emission rows (%.2f%%) carry ≥1 span "
        "(%d in floor-passing cells); per-span (floor-passing) %s",
        frag_aggregates["n_any_fragment_all_nonemission_rows"],
        frag_aggregates["n_nonemission_rows_all_cells"],
        100 * any_prevalence,
        frag_aggregates["n_any_fragment_retained_cells"],
        frag_totals,
    )

    # ── 5. Predictor stats vs E (per layer chain comparison) ─────────────────
    import issue722_fit_M as fitM

    from explore_persona_space.analysis.issue667.gate_chain import clustered_bootstrap_spearman

    # Cell alignment source: the L14 joined cache (layer-independent keys).
    ref = recompute_chains(out_dir, 14) if not args.skip_chains else None
    if ref is None:
        # text-only smoke: derive keys/families from the manifest, E via fitM.
        cell_keys = sorted(manifest_cells)
        families = [_family_of(k.split("__")[1]) for k in cell_keys]
        E = fitM._load_E(BEHAVIOR, cell_keys)
        keep = ~np.isnan(E)
    else:
        cell_keys, families, E, keep = ref["cell_keys"], ref["families"], ref["E"], ref["keep"]
    assert set(cell_keys) == set(manifest_cells), "joined-cache keys != rollout-JSON cells"

    emis = np.asarray(
        [per_cell_variant_frac[k]["pinned_span"] for k in cell_keys], dtype=np.float64
    )
    retention_frac = np.asarray(
        [manifest_cells[k]["retained"] / manifest_cells[k]["total"] for k in cell_keys],
        dtype=np.float64,
    )
    retained_n = np.asarray([manifest_cells[k]["retained"] for k in cell_keys], dtype=np.float64)
    fams_k = [f for f, m in zip(families, keep, strict=True) if m]
    Ek = E[keep]

    def _fam_ci(x: np.ndarray) -> dict:
        return clustered_bootstrap_spearman(x[keep], Ek, fams_k)

    predictor: dict = {
        "behavior": BEHAVIOR,
        "n_cells": len(cell_keys),
        "n_with_E": int(keep.sum()),
        "dv_a_emission_vs_E": {
            name: _fam_ci(
                np.asarray([per_cell_variant_frac[k][name] for k in cell_keys], dtype=np.float64)
            )
            for name in ("pinned_span", "whole_response_equality", "full_sentence_containment")
        },
        "dv_c_retention_vs_E": _fam_ci(retention_frac),
        "dv_c_retained_n_vs_E": _fam_ci(retained_n),
    }

    # DV-C at the ANALYSIS-set granularity (the 291 floor-passing cells — the
    # plan's verified Spearman(retained_N, E) = −0.748; the noise-dose confound
    # the matched-N comparator removes lives on THIS set).
    ret_mask = np.asarray([not manifest_cells[k]["below_floor"] for k in cell_keys]) & keep
    fams_ret = [f for f, m in zip(families, ret_mask, strict=True) if m]
    predictor["dv_c_retained_n_vs_E_retained_cells"] = {
        **clustered_bootstrap_spearman(retained_n[ret_mask], E[ret_mask], fams_ret),
        "n_retained_cells_with_E": int(ret_mask.sum()),
        "plan_time_reference": -0.748,
    }

    # DV-D Spearman over RETAINED cells with E (fragment fraction defined there only).
    frag_mask = np.asarray([k in frag_cells for k in cell_keys]) & keep
    frag_frac = np.asarray(
        [frag_cells[k]["any_fragment_fraction"] if k in frag_cells else np.nan for k in cell_keys],
        dtype=np.float64,
    )
    fams_frag = [f for f, m in zip(families, frag_mask, strict=True) if m]
    dv_d_ci = clustered_bootstrap_spearman(frag_frac[frag_mask], E[frag_mask], fams_frag)
    predictor["dv_d_fragment_vs_E"] = {**dv_d_ci, "n_retained_cells_with_E": int(frag_mask.sum())}
    _write_json(
        em_dir / "fragment_tabulation.json",
        {
            "span_definitions": {**FRAGMENT_SPANS, "digit_7_and_bench": r"\b7\b AND 'bench'"},
            "per_span_totals_retained_cells": frag_totals,
            **frag_aggregates,
            "n_retained_rows_floor_passing_cells": n_retained_rows,
            "plan_time_reference": "142/6892 = 2.1% (all non-emission rows)",
            "branch_b_gate": {
                "prevalence_leq_5pct": any_prevalence <= 0.05,
                "spearman_vs_E": dv_d_ci,
                "gate_note": "branch-(b) narration requires prevalence ≤5% AND CI spanning 0",
            },
            "cells": frag_cells,
            "meta": meta_common,
        },
    )

    # 16-band source-mean read (plan §4.2): source-mean fraction vs source-mean E.
    predictor["dv_a_source_mean_vs_E"] = source_mean_read(cell_keys, keep, emis, E)

    # Paired (emission ρ − chain ρ) diffs per layer, both arms (plan §4.3).
    if not args.skip_chains:
        predictor["paired_vs_chain"] = {}
        for layer in LAYERS:
            r = ref if layer == 14 else recompute_chains(out_dir, layer)
            assert r["cell_keys"] == cell_keys, f"L{layer} cell_keys differ from L14"
            emis_k = emis[r["keep"]]
            layer_block: dict = {
                "consistency_vs_committed": r["consistency"],
                "joined_cache_sha256": r["cache_sha256"],
            }
            for arm, chain in r["chains"].items():
                layer_block[f"ci_diff_emission_minus_{arm}"] = fitM._clustered_paired_rho_diff_ci(
                    chain, emis_k, r["E"][r["keep"]], fams_k
                )
                layer_block[f"rho_{arm}_ridge"] = r["rhos"][arm]
            predictor["paired_vs_chain"][f"L{layer}"] = layer_block

    predictor["meta"] = meta_common
    _write_json(em_dir / "emission_predictor.json", predictor)
    logger.info(
        "[phase=emission_rate] DONE: pinned Spearman(emission, E)=%s; retention rho=%s; "
        "retained_N rho=%s; outputs in %s",
        json.dumps(predictor["dv_a_emission_vs_E"]["pinned_span"].get("point")),
        json.dumps(predictor["dv_c_retention_vs_E"].get("point")),
        json.dumps(predictor["dv_c_retained_n_vs_E"].get("point")),
        em_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
