"""Issue #2094 fu2_span_slots — VM-side ANALYSIS leg (user-chat follow-up).

Consumes the fu2 judge scores (``eval_results/issue_2094/judge_fu2/scores``),
the staged fu2 rollout shards, the fu2 V_a tensors (staged from HF), the
committed parent anchors, and the parent well-sep bootstrap artifact; writes
(never touching any parent file):

- ``f_metrics/fu2/fu2_cells.jsonl`` + ``fu2_null_cells.jsonl`` — per-cell F
  rows for all 2,400 fu2 cells, reduced with EXACTLY the parent ftables
  conventions (``issue2094_analysis.assemble_shard_rows`` reused by import:
  batched signed-projection F_act over the fu2 V_a shards at the parent read
  layer, per-kind anchored F_beh from the judge contrasts, coherence gating at
  >60, drops counted never coerced).
- ``f_metrics/fu2/fu2_f_reads.json`` — per-family full detail: the
  |sep| >= 0.5 well-separated pair-clustered bootstrap (B=10,000, seed 20941,
  ``issue2094_wellsep_bootstrap`` reused verbatim) PLUS the unrestricted
  companion (labeled), per-family steered-vs-null reads, per-family QC
  (incoherent-excluded / judge-dropped / empty-completion / cap-hit counts),
  and the pooled cap-hit table cross-checked against the pod's
  ``fu2_caphit.json`` manifest.
- ``f_metrics/fu2/fu2_summary.json`` — the verdict table (per slot x
  layer-variant x dose x setting x metric: n_pairs, both-arm means + CIs,
  disjoint-CI + steered-above verdict, compromised flag at steered pooled
  cap-hit > 2%, incoherent fractions) + per-slot clean-family counts + the
  parent qspan/ce comparables pulled from
  ``f_metrics/bootstrap_cis_wellsep.json`` (vec_type A, joint_all/joint_mid),
  with the fu1 1024-cap breached-cell flag carried for fairness.

Compromise handling (scope-directed): families whose steered pooled cap-hit
fraction exceeds the fu1 2% trigger are LABELED ``compromised: true`` — reads
still computed, never silently dropped or silently included.

Fail-fast: grid totals asserted against the fu2 driver's pinned constants
(30 families / 60 blocks / 2,400 cells), judge-score row counts against the
wave metas, score-key coverage against the enumerated grid (set-equality),
V_a shard coverage against the 60 expected block slugs, and the recomputed
cap-hit table against the pod manifest.

VM launch convention (shared-VM thread caps):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
    uv run python scripts/issue2094_fu2_analysis.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import torch  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    stage_hub_file,
    stage_hub_prefix,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_analysis as A  # noqa: E402
import issue2094_fu1 as FU1  # noqa: E402
import issue2094_fu1_analysis as FU1A  # noqa: E402
import issue2094_fu2 as FU2  # noqa: E402
import issue2094_run as R  # noqa: E402
import issue2094_wellsep_bootstrap as W  # noqa: E402

logger = logging.getLogger("issue2094_fu2_analysis")

REPO_ROOT = A.REPO_ROOT
DEFAULT_SCORES_DIR = REPO_ROOT / "eval_results/issue_2094/judge_fu2/scores"
DEFAULT_FMETRICS_DIR = REPO_ROOT / "eval_results/issue_2094/f_metrics"
DEFAULT_ROLLOUTS = (
    REPO_ROOT
    / "data/issue_2094/fu2_judge_inputs/issue2094_singlepos/raw_completions/fu2_span_slots/rollouts"
)
DEFAULT_STAGE_ROOT = REPO_ROOT / "data/issue_2094/fu2_va_stage"
DEFAULT_PARTS_DIR = REPO_ROOT / "data/issue_2094/fu2_analysis_parts"
DEFAULT_OUT_DIR = DEFAULT_FMETRICS_DIR / "fu2"

VA_PREFIX = f"{FU2.HF_FU2_TENSORS}/va"
CAPHIT_PATH_IN_REPO = f"{FU2.HF_FU2_TENSORS}/manifests/fu2_caphit.json"
ANCHORS_PT_IN_REPO = f"{R.HF_PREFIX}/analysis_tensors/anchors/va_anchors.pt"

# Parent slots the fold compares the fu2 slots against (scope-marker ask):
# qspan (the shipped query-span slot qtext refines) + ce (context-end).
PARENT_COMPARABLE_SLOTS: tuple[str, ...] = ("qspan", "ce")

RC_OK = 0


# ── expected-grid enumeration (pure; pinned in tests) ───────────────────


def eligible_settings(pairs: list[BANK.Pair], slot: str) -> tuple[str, ...]:
    """Settings realized by a slot's eligible pair set (driver's own rule)."""
    keep = set(FU2.fu2_pair_ids(pairs, slot))
    return tuple(
        s
        for s in ("matched_prefix", "matched_query", "cross")
        if any(p.pair_id in keep and p.setting == s for p in pairs)
    )


def expected_family_tails(pairs: list[BANK.Pair]) -> set[str]:
    """Every steered-vs-null read key ``setting|slot|lv|dose|A|metric`` the fu2
    grid must produce (170 = qtext 70 + pspan_tmpl 50 + pspan_text 50)."""
    tails: set[str] = set()
    for slot in FU2.FU2_SLOTS:
        for setting in eligible_settings(pairs, slot):
            metrics = ["f_act"] + [f"f_beh_{k}" for k in BANK.SETTING_RUBRIC_KINDS[setting]]
            for lv in FU2.FU2_VARIANTS:
                for dose in R.DOSES_A:
                    for metric in metrics:
                        tails.add("|".join([setting, slot, lv, dose, "A", metric]))
    return tails


def expected_score_keys(
    families: list[tuple[R.Block, R.Block]], pairs_by_id: dict[str, BANK.Pair]
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str, str]]]:
    """(coherence keys, behavior keys) the judge waves must cover exactly."""
    coh: set[tuple[str, str]] = set()
    beh: set[tuple[str, str, str, str]] = set()
    for fam in families:
        for block in fam:
            for pid in block.pair_ids:
                coh.add((block.key, pid))
                for kind in BANK.SETTING_RUBRIC_KINDS[pairs_by_id[pid].setting]:
                    for side in ("a", "b"):
                        beh.add((block.key, pid, kind, side))
    return coh, beh


def analysis_regime(profiles: bool) -> str:
    """Resume regime for the parts dir (every output-affecting knob, #722 r3)."""
    key = json.dumps(
        {
            "code": "fu2-analysis-v1",
            "fu2_regime_token": FU2.FU2_REGIME_TOKEN,
            "coherence_threshold": A.COHERENCE_THRESHOLD,
            "primary_read_layer": A.PRIMARY_READ_LAYER,
            "profiles": profiles,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ── staging (idempotent; mirror-root layout, #1774) ─────────────────────


def stage_inputs(stage_root: Path, families: list[tuple[R.Block, R.Block]]) -> dict:
    """Stage fu2 V_a shards + parent anchors V_a + the pod cap-hit manifest.

    ``stage_hub_prefix`` mirrors repo-relative paths under ``stage_root``
    (files land at ``stage_root/<repo path>`` — the #1774 mirror-root
    contract); per-file skip-if-present makes re-runs cheap. Returns the
    resolved local paths + the staged byte total, and asserts the V_a shard
    set covers EXACTLY the 60 expected block slugs.
    """
    va_dir = stage_root / VA_PREFIX
    anchors_pt = stage_root / ANCHORS_PT_IN_REPO
    caphit_json = stage_root / CAPHIT_PATH_IN_REPO
    expected_slugs = sorted(b.slug for fam in families for b in fam)

    missing_va = [s for s in expected_slugs if not (va_dir / f"shard_{s}.pt").is_file()]
    if missing_va:
        logger.info("[stage] fetching %d/%d fu2 va shards from HF", len(missing_va), 60)
        stage_hub_prefix(R.HF_DATA_REPO, VA_PREFIX, stage_root, repo_type="dataset")
    if not anchors_pt.is_file():
        stage_hub_file(R.HF_DATA_REPO, ANCHORS_PT_IN_REPO, anchors_pt, repo_type="dataset")
    if not caphit_json.is_file():
        stage_hub_file(R.HF_DATA_REPO, CAPHIT_PATH_IN_REPO, caphit_json, repo_type="dataset")

    present = {p.name for p in va_dir.glob("shard_*.pt")}
    expected_names = {f"shard_{s}.pt" for s in expected_slugs}
    assert present == expected_names, (
        f"staged va shard set != expected 60 block slugs; missing "
        f"{sorted(expected_names - present)[:4]}, extra {sorted(present - expected_names)[:4]}"
    )
    staged_bytes = (
        sum((va_dir / n).stat().st_size for n in expected_names)
        + anchors_pt.stat().st_size
        + caphit_json.stat().st_size
    )
    return {
        "va_dir": va_dir,
        "anchors_pt": anchors_pt,
        "caphit_json": caphit_json,
        "staged_bytes": staged_bytes,
    }


# ── per-cell reduction (parent ftables conventions, reused by import) ───


def assemble_fu2_cells(
    families: list[tuple[R.Block, R.Block]],
    rollouts_dir: Path,
    va_dir: Path,
    parts_dir: Path,
    lk: A.JudgeLookups,
    pair_stats: dict[tuple[str, str], dict],
    anchor_va: dict,
    pairs_by_id: dict[str, BANK.Pair],
    profiles: bool = False,
) -> list[dict]:
    """Per-cell F rows for all 60 fu2 blocks — ``A.assemble_shard_rows`` per
    shard, checkpointed per unit (60 units > ~50, code-style T2) with a
    regime-keyed resume manifest and a per-unit progress line."""
    regime = analysis_regime(profiles)
    parts_manifest = parts_dir / "parts_manifest.json"
    done_parts: set[str] = set()
    if parts_manifest.exists():
        rec = json.loads(parts_manifest.read_text())
        if rec.get("regime") != regime:
            raise RuntimeError(
                f"fu2 analysis parts at {parts_dir} carry a DIFFERENT regime "
                f"({rec.get('regime')} != {regime}) — quarantine or delete the parts dir"
            )
        done_parts = set(rec.get("done", []))

    blocks = [b for fam in families for b in fam]
    t0 = time.monotonic()
    for k, block in enumerate(sorted(blocks, key=lambda b: b.slug), start=1):
        part = parts_dir / f"{block.slug}.jsonl"
        if block.slug in done_parts and part.exists():
            continue
        shard = rollouts_dir / f"shard_{block.slug}.jsonl"
        assert shard.is_file(), f"missing rollout shard {shard}"
        rows = list(A._iter_jsonl(shard))
        assert len(rows) == block.n_cells, (block.key, len(rows), block.n_cells)
        assert {r["block_key"] for r in rows} == {block.key}, block.key
        assert sorted(r["pair_id"] for r in rows) == sorted(block.pair_ids), block.key
        va = torch.load(  # self-produced sha-lineage shard: non-tensor index metadata
            va_dir / f"shard_{block.slug}.pt", map_location="cpu", weights_only=False
        )
        cell_rows = A.assemble_shard_rows(
            rows, va, lk, pair_stats, anchor_va, pairs_by_id, profiles=profiles
        )
        A._write_jsonl_atomic(part, cell_rows)
        done_parts.add(block.slug)
        A._write_json_atomic(parts_manifest, {"regime": regime, "done": sorted(done_parts)})
        print(
            f"[fu2-ftables] unit {k}/{len(blocks)} {block.slug} "
            f"elapsed={time.monotonic() - t0:.1f}s",
            flush=True,
        )

    out: list[dict] = []
    for block in sorted(blocks, key=lambda b: b.slug):
        out.extend(A._iter_jsonl(parts_dir / f"{block.slug}.jsonl"))
    assert len(out) == FU2.EXPECTED_FU2_TOTALS["cells_total"], len(out)
    return out


# ── cap-hit pooling + manifest cross-check ──────────────────────────────


def crosscheck_caphit(pooled: dict[tuple[str, str, str], dict], manifest: dict) -> dict:
    """Recomputed pooled cap-hit (from rollout rows' own ``cap_hit``) must
    equal the pod's ``fu2_caphit.json`` manifest exactly (counts + n)."""
    assert manifest["max_new_tokens"] == FU2.FU2_MAX_NEW_TOKENS, manifest["max_new_tokens"]
    m_by_key = {(c["slot"], c["layer_variant"], c["dose"]): c for c in manifest["cells"]}
    assert set(m_by_key) == set(pooled), sorted(set(m_by_key) ^ set(pooled))
    for key, arms in pooled.items():
        for arm in ("steered", "null"):
            assert arms[arm]["n"] == m_by_key[key][arm]["n"], (key, arm)
            assert arms[arm]["cap_hit"] == m_by_key[key][arm]["cap_hit"], (
                key,
                arm,
                arms[arm]["cap_hit"],
                m_by_key[key][arm]["cap_hit"],
            )
    return {
        "passed": True,
        "n_pooled_cells": len(pooled),
        "max_new_tokens": manifest["max_new_tokens"],
        "source": "recomputed from staged rollout rows' cap_hit; cross-checked "
        "against the pod manifest fu2_caphit.json (counts exact)",
    }


def family_qc(cell_rows: list[dict]) -> dict[tuple[str, str, str, str, str], dict]:
    """Per (slot, lv, dose, setting, arm): row / exclusion / drop counts."""
    qc: dict[tuple[str, str, str, str, str], dict] = {}
    for r in cell_rows:
        key = (r["slot"], r["layer_variant"], r["dose"], r["setting"], r["arm"])
        a = qc.setdefault(
            key,
            {
                "n_rows": 0,
                "n_excluded_incoherent": 0,
                "n_empty_completion": 0,
                "n_cap_hit": 0,
                "n_judge_dropped": 0,
                "n_anchor_missing": 0,
            },
        )
        a["n_rows"] += 1
        a["n_excluded_incoherent"] += int(bool(r["excluded_incoherent"]))
        a["n_empty_completion"] += int(bool(r.get("empty_completion")))
        a["n_cap_hit"] += int(bool(r.get("cap_hit")))
        for rec in (r.get("f_beh") or {}).values():
            missing = rec.get("missing")
            if missing == "judge_dropped":
                a["n_judge_dropped"] += 1
            elif missing == "anchor_missing":
                a["n_anchor_missing"] += 1
    for a in qc.values():
        a["incoherent_frac"] = a["n_excluded_incoherent"] / a["n_rows"]
    return qc


# ── verdict table ───────────────────────────────────────────────────────


def _verdict(read: dict, compromised: bool) -> str:
    """The grid-convention read: 95% pair-clustered bootstrap CIs disjoint with
    steered above, at >= MIN_PAIRS_HEADLINE well-separated pairs, un-compromised."""
    if not read.get("comparable"):
        return "not_comparable"
    separated = bool(read["cis_disjoint"]) and read["direction"] == "steered_above"
    if not separated:
        return "not_separating"
    if read["n_pairs_used"] < W.MIN_PAIRS_HEADLINE:
        return f"separating_lt{W.MIN_PAIRS_HEADLINE}_pairs"
    return "separating_compromised" if compromised else "clean_separating"


def build_verdict_table(
    reads_ws: dict[str, dict],
    reads_unres: dict[str, dict],
    caphit: dict[tuple[str, str, str], dict],
    qc: dict[tuple[str, str, str, str, str], dict],
    expected_tails: set[str],
) -> list[dict]:
    """One row per family read; a family silently missing from either read set
    — or a family without a pooled cap-hit entry — is fail-loud."""
    assert set(reads_ws) == expected_tails, (
        f"wellsep read families != expected fu2 enumeration; missing "
        f"{sorted(expected_tails - set(reads_ws))[:6]}, "
        f"extra {sorted(set(reads_ws) - expected_tails)[:6]}"
    )
    assert set(reads_unres) == expected_tails, (
        f"unrestricted read families != expected fu2 enumeration; missing "
        f"{sorted(expected_tails - set(reads_unres))[:6]}"
    )
    rows = []
    for tail in sorted(expected_tails):
        setting, slot, lv, dose, vt, metric = tail.split("|")
        pooled = caphit.get((slot, lv, dose))
        assert pooled is not None, (
            f"family {tail}: no pooled cap-hit entry for {(slot, lv, dose)} — "
            "compromise labeling would be silently skipped"
        )
        compromised = pooled["steered"]["cap_hit_frac"] > FU1.CAPHIT_TRIGGER_FRAC
        rws = reads_ws[tail]
        rows.append(
            {
                "family": tail,
                "slot": slot,
                "layer_variant": lv,
                "dose": dose,
                "setting": setting,
                "vec_type": vt,
                "metric": metric,
                "wellsep": rws,
                "unrestricted": reads_unres[tail],
                "cap_hit_frac": {arm: pooled[arm]["cap_hit_frac"] for arm in ("steered", "null")},
                "compromised": compromised,
                "qc": {arm: qc[(slot, lv, dose, setting, arm)] for arm in ("steered", "null")},
                "verdict": _verdict(rws, compromised),
            }
        )
    return rows


def per_slot_summary(rows: list[dict]) -> dict[str, dict]:
    """Per-slot verdict counts + the clean-separating family list."""
    out: dict[str, dict] = {}
    for slot in FU2.FU2_SLOTS:
        srows = [r for r in rows if r["slot"] == slot]
        counts: dict[str, int] = {}
        for r in srows:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        out[slot] = {
            "n_family_reads": len(srows),
            "verdict_counts": counts,
            "n_clean_separating": counts.get("clean_separating", 0),
            "n_separating_incl_compromised": (
                counts.get("clean_separating", 0) + counts.get("separating_compromised", 0)
            ),
            "clean_families": sorted(
                r["family"] for r in srows if r["verdict"] == "clean_separating"
            ),
            "clean_families_by_metric": {
                m: sum(1 for r in srows if r["verdict"] == "clean_separating" and r["metric"] == m)
                for m in sorted({r["metric"] for r in srows})
            },
        }
    return out


# ── parent comparables (qspan / ce, vec_type A, joint variants) ─────────


def parent_comparables(parent_wellsep: dict, breached_1024: set[tuple[str, str, str]]) -> dict:
    """The parent's like-for-like steered-vs-null reads (bootstrap_cis_wellsep
    conventions == the fu2 read above), qspan/ce x joint_all/joint_mid x
    Type-A, with the fu1 1024-cap breached-cell flag as the parent's own
    compromise label (parent cap was 1024; fu1 later regenerated at 2048)."""
    rows = []
    for tail, read in parent_wellsep["steered_vs_null"].items():
        setting, slot, lv, dose, vt, metric = tail.split("|")
        if slot not in PARENT_COMPARABLE_SLOTS or vt != "A" or lv not in FU2.FU2_VARIANTS:
            continue
        compromised = (slot, lv, dose) in breached_1024
        rows.append(
            {
                "family": tail,
                "slot": slot,
                "layer_variant": lv,
                "dose": dose,
                "setting": setting,
                "metric": metric,
                "wellsep": read,
                "compromised_1024_caphit": compromised,
                "verdict": _verdict(read, compromised),
            }
        )
    assert rows, "no parent qspan/ce joint-variant Type-A reads found"
    per_slot: dict[str, dict] = {}
    for slot in PARENT_COMPARABLE_SLOTS:
        srows = [r for r in rows if r["slot"] == slot]
        counts: dict[str, int] = {}
        for r in srows:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        per_slot[slot] = {
            "n_family_reads": len(srows),
            "verdict_counts": counts,
            "n_clean_separating": counts.get("clean_separating", 0),
            "clean_families": sorted(
                r["family"] for r in srows if r["verdict"] == "clean_separating"
            ),
        }
    return {
        "source": "eval_results/issue_2094/f_metrics/bootstrap_cis_wellsep.json "
        "(parent grid, max_new_tokens=1024; fu1 regen artifact exists at "
        "f_metrics/fu1/fu1_wellsep_bootstrap_regen.json for the 16 breached cells)",
        "restriction": "slot in (qspan, ce), vec_type A, layer_variant in "
        "(joint_all, joint_mid) — the fu2-comparable subset",
        "compromise_label": "fu1 derive_breached_cells(fragility) pooled cells "
        "(steered 1024 cap-hit > 2%)",
        "rows": rows,
        "per_slot": per_slot,
    }


# ── main ────────────────────────────────────────────────────────────────


def _repro() -> dict:
    return {**A._repro(), **as_metadata_dict(git_provenance())}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    ap.add_argument("--fmetrics-dir", type=Path, default=DEFAULT_FMETRICS_DIR)
    ap.add_argument("--rollouts-dir", type=Path, default=DEFAULT_ROLLOUTS)
    ap.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    ap.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=A.BOOTSTRAP_B)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.parts_dir.mkdir(parents=True, exist_ok=True)

    # ── grid enumeration + fail-fast totals (driver constants, never re-derived)
    logger.info("[phase=fu2a_enumerate]")
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    families = FU2.enumerate_fu2_families(pairs)
    exp = FU2.EXPECTED_FU2_TOTALS
    assert len(families) == exp["n_families"], (len(families), exp["n_families"])
    blocks = [b for fam in families for b in fam]
    assert len(blocks) == exp["n_blocks"], len(blocks)
    n_cells = sum(b.n_cells for b in blocks)
    assert n_cells == exp["cells_total"], (n_cells, exp["cells_total"])
    expected_coh, expected_beh = expected_score_keys(families, pairs_by_id)
    tails = expected_family_tails(pairs)

    # ── staging (idempotent) ────────────────────────────────────────────
    logger.info("[phase=fu2a_stage]")
    staged = stage_inputs(args.stage_root, families)
    logger.info("[stage] inputs present (%.1f MB staged total)", staged["staged_bytes"] / 1e6)

    # ── judge scores: wave metas + fail-loud routing + coverage ─────────
    logger.info("[phase=fu2a_load]")
    metas = FU1A.check_wave_metas(args.scores_dir)
    n_meta_items = sum(m["regime"]["n_items"] for m in metas.values())
    rows_iter = (
        row for f in sorted(args.scores_dir.glob("*.scores.jsonl")) for row in A._iter_jsonl(f)
    )
    sc = FU1A.route_fu1_scores(rows_iter)
    assert not sc.s2_coh and not sc.s2_beh, "fu2 waves must carry grid rows only"
    n_routed = len(sc.grid_coh) + len(sc.grid_beh)
    assert n_routed == n_meta_items, (n_routed, n_meta_items)
    assert set(sc.grid_coh) == expected_coh, (
        f"coherence score coverage != enumerated grid "
        f"(scores {len(sc.grid_coh)}, expected {len(expected_coh)})"
    )
    assert set(sc.grid_beh) == expected_beh, (
        f"behavior score coverage != enumerated grid "
        f"(scores {len(sc.grid_beh)}, expected {len(expected_beh)})"
    )
    lk = A.JudgeLookups(grid_coh=sc.grid_coh, grid_beh=sc.grid_beh)
    logger.info("[load] scores routed: grid_coh=%d grid_beh=%d", len(sc.grid_coh), len(sc.grid_beh))

    anchors_path = args.fmetrics_dir / "anchors.jsonl"
    pair_stats = FU1A.load_anchor_stats(anchors_path)
    anchor_va = A._load_anchor_va(SimpleNamespace(anchors_pt=staged["anchors_pt"]))
    ws, ws_any = W.load_wellsep(anchors_path, W.MIN_SEPARATION)

    # ── per-cell reduction (parent conventions) ─────────────────────────
    logger.info("[phase=fu2a_ftables]")
    cell_rows = assemble_fu2_cells(
        families,
        args.rollouts_dir,
        staged["va_dir"],
        args.parts_dir,
        lk,
        pair_stats,
        anchor_va,
        pairs_by_id,
    )
    steered_rows = [r for r in cell_rows if r["arm"] == "steered"]
    null_rows = [r for r in cell_rows if r["arm"] == "null"]
    assert len(steered_rows) == exp["cells_steered"], len(steered_rows)
    assert len(null_rows) == exp["cells_null"], len(null_rows)
    A._write_jsonl_atomic(args.out_dir / "fu2_cells.jsonl", steered_rows)
    A._write_jsonl_atomic(args.out_dir / "fu2_null_cells.jsonl", null_rows)
    logger.info("[phase=fu2a_ftables_done] steered=%d null=%d", len(steered_rows), len(null_rows))

    # ── cap-hit pooling + manifest cross-check ──────────────────────────
    logger.info("[phase=fu2a_caphit]")
    rollout_rows: list[dict] = []
    for block in blocks:
        rollout_rows.extend(A._iter_jsonl(args.rollouts_dir / f"shard_{block.slug}.jsonl"))
    assert len(rollout_rows) == exp["cells_total"], len(rollout_rows)
    caphit = FU1A.recompute_caphit_2048(rollout_rows)
    caphit_check = crosscheck_caphit(
        caphit, json.loads(staged["caphit_json"].read_text(encoding="utf-8"))
    )
    compromised_cells = sorted(
        key
        for key, arms in caphit.items()
        if arms["steered"]["cap_hit_frac"] > FU1.CAPHIT_TRIGGER_FRAC
    )
    logger.info(
        "[caphit] pooled cells=%d compromised (steered > %.0f%%): %d",
        len(caphit),
        100 * FU1.CAPHIT_TRIGGER_FRAC,
        len(compromised_cells),
    )

    # ── bootstrap: wellsep-restricted + unrestricted companion ──────────
    logger.info("[phase=fu2a_bootstrap] n_boot=%d", args.n_boot)
    eligible, n_degenerate_excluded = A.bootstrap_eligible_rows(cell_rows)
    assert n_degenerate_excluded == 0, (
        f"{n_degenerate_excluded} fu2 rows flagged degenerate-self — the fu2 grid "
        "excludes degenerate cells by construction (pspan slots drop matched-prefix)"
    )
    fams_ws = W.compute_wellsep_families(eligible, ws, ws_any, args.n_boot)
    reads_ws, summary_ws = W.steered_vs_null_reads(fams_ws)
    ws_all = {(p.pair_id, k) for p in pairs for k in ("prefix", "query")}
    ws_any_all = {p.pair_id for p in pairs}
    fams_unres = W.compute_wellsep_families(eligible, ws_all, ws_any_all, args.n_boot)
    reads_unres, summary_unres = W.steered_vs_null_reads(fams_unres)

    # ── verdict table + summaries ───────────────────────────────────────
    logger.info("[phase=fu2a_verdicts]")
    qc = family_qc(cell_rows)
    table = build_verdict_table(reads_ws, reads_unres, caphit, qc, tails)
    slot_summary = per_slot_summary(table)

    parent_wellsep = json.loads(
        (args.fmetrics_dir / "bootstrap_cis_wellsep.json").read_text(encoding="utf-8")
    )
    assert parent_wellsep["B"] == args.n_boot and parent_wellsep["seed"] == A.BOOTSTRAP_SEED, (
        "parent wellsep artifact B/seed mismatch — comparison not like-for-like"
    )
    fragility = json.loads((REPO_ROOT / FU1.FRAGILITY_REL).read_text(encoding="utf-8"))
    breached = set(FU1.derive_breached_cells(fragility))
    parent = parent_comparables(parent_wellsep, breached)

    repro = _repro()
    A._write_json_atomic(
        args.out_dir / "fu2_f_reads.json",
        {
            "B": args.n_boot,
            "seed": A.BOOTSTRAP_SEED,
            "resample_axis": "pairs (pair-clustered, within setting)",
            "max_new_tokens": FU2.FU2_MAX_NEW_TOKENS,
            "n_degenerate_excluded": n_degenerate_excluded,
            "restriction": {
                "min_abs_separation": W.MIN_SEPARATION,
                "f_beh": "pair kept iff its (pair, rubric-kind) anchor |separation| >= floor",
                "f_act": "pair kept iff well-separated on >= 1 rubric kind",
                "n_wellsep_pair_kinds": len(ws),
                "n_wellsep_pairs_any_kind": len(ws_any),
            },
            "families_wellsep": fams_ws,
            "families_unrestricted": fams_unres,
            "steered_vs_null_wellsep": reads_ws,
            "steered_vs_null_unrestricted": reads_unres,
            "summary_wellsep": summary_ws,
            "summary_unrestricted": summary_unres,
            "qc_per_family_arm": {"|".join(k): v for k, v in sorted(qc.items())},
            "caphit": {
                "cells": [
                    {"slot": s, "layer_variant": lv, "dose": d, **arms}
                    for (s, lv, d), arms in sorted(caphit.items())
                ],
                "crosscheck": caphit_check,
                "trigger_frac": FU1.CAPHIT_TRIGGER_FRAC,
                "compromised_cells": [list(c) for c in compromised_cells],
            },
            "note": (
                "fu2_span_slots per-family reads: per-cell reduction via the parent "
                "ftables conventions (issue2094_analysis.assemble_shard_rows reused by "
                "import: batched F_act over the fu2 V_a shards, anchored per-kind F_beh, "
                "coherence gating >60, drops counted never coerced); bootstrap via "
                "issue2094_wellsep_bootstrap reused verbatim (batched "
                "bootstrap_family_means_batched); the unrestricted companion is the "
                "same battery with the well-sep keep predicate disabled"
            ),
            "repro": repro,
        },
    )
    A._write_json_atomic(
        args.out_dir / "fu2_summary.json",
        {
            "B": args.n_boot,
            "seed": A.BOOTSTRAP_SEED,
            "max_new_tokens": FU2.FU2_MAX_NEW_TOKENS,
            "verdict_definition": (
                "clean_separating = steered vs shuffled-donor-null 95% pair-clustered "
                "bootstrap CIs disjoint with steered above, on well-separated pairs "
                f"(|sep| >= {W.MIN_SEPARATION}), >= {W.MIN_PAIRS_HEADLINE} pairs, and "
                f"the family's pooled steered cap-hit <= {FU1.CAPHIT_TRIGGER_FRAC:.0%} "
                "(above it the family is separating_compromised — computed, labeled, "
                "never dropped); per-family alpha=0.05, uncorrected for multiplicity"
            ),
            "verdict_table": table,
            "per_slot": slot_summary,
            "compromised_cells": [list(c) for c in compromised_cells],
            "parent_comparables": parent,
            "repro": repro,
        },
    )
    logger.info(
        "[phase=fu2a_done] reads=%d clean per slot: %s",
        len(table),
        {s: slot_summary[s]["n_clean_separating"] for s in FU2.FU2_SLOTS},
    )
    return RC_OK


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
