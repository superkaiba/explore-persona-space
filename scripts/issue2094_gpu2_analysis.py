"""Issue #2094 gpu2_mq_replacement_prefix — VM-side ANALYSIS leg (follow-up).

Consumes the gpu2 judge scores (``eval_results/issue_2094/judge_gpu2/scores``),
the staged gpu2 rollout shards, the gpu2 V_a tensors + conv2 anchor V_a
(staged from HF), the pod gate scores (``judge_gate/scores/*.gpu2anchors``),
the parent bare anchor draws, and the parent well-sep bootstrap artifact;
writes (never touching any parent file; ``gpu2/diagnosis.json`` untouched):

- ``f_metrics/gpu2/gpu2_anchor_stats.json`` — per-pair floor/ceiling anchor
  stats for the 5 conv2 re-formed pairs, recomputed from the pod gate scores
  through the driver's own ``gate_separations`` / ``gate_verdict`` (imported),
  with an EXACT-reproduction assert against the pod's recorded
  ``gate_report.json`` (separations + verdict, float-exact).
- ``f_metrics/gpu2/gpu2_cells.jsonl`` + ``gpu2_null_cells.jsonl`` — per-cell F
  rows for all 1,500 gpu2 cells, reduced with EXACTLY the parent ftables
  conventions (``issue2094_analysis.assemble_shard_rows`` reused by import:
  batched signed-projection F_act over the gpu2 V_a shards, anchored per-kind
  F_beh from the fp-bare/fp-conv2 judge contrasts, coherence gating at >60,
  drops counted never coerced).
- ``f_metrics/gpu2/gpu2_f_reads.json`` — per-family full detail: the wellsep
  (4 gate-passing pairs; q2 excluded at |sep| < 0.5) pair-clustered bootstrap
  (B=10,000, seed 20941, ``A.bootstrap_family_means_batched`` reused) PLUS the
  unrestricted 5-pair companion (labeled), per-family QC, and the pooled
  cap-hit table cross-checked against the pod's ``gpu2_caphit.json``.
- ``f_metrics/gpu2/gpu2_summary.json`` — the verdict table (150 families x 2
  metrics) under BOTH conventions — the >=5-pair HEADLINE convention (fu2
  taxonomy verbatim; with 4 pairs every separating read caps at
  ``separating_lt5_pairs``) and an explicitly-labeled >=3-pair SMALL-N regime
  — plus the parent matched-query ce comparison (same variant x dose
  families from ``bootstrap_cis_wellsep.json``) and provenance incl. the
  fp-conv2 instrument-validity (gate-wave de-facto pilot) statement.

Small-n honesty: the wellsep restriction leaves n_pairs_used = 4 per family
(< MIN_PAIRS_HEADLINE = 5), so CIs are wide and the headline bar is NOT
silently lowered — both verdict columns ship and the fold decides.

Fail-fast: grid totals asserted against the driver's pinned constants
(150 families / 300 blocks / 1,500 cells), gate-separation reproduction
against the pod report (exact), judge-score row counts against the wave
metas, score-key coverage against the enumerated grid (set-equality), V_a
shard coverage against the 300 expected block slugs, and the recomputed
cap-hit table against the pod manifest.

VM launch convention (shared-VM thread caps):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
    uv run python scripts/issue2094_gpu2_analysis.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import numpy as np  # noqa: E402
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
import issue2094_fu2_analysis as FU2A  # noqa: E402
import issue2094_gpu2 as G2  # noqa: E402
import issue2094_gpu2_bank as G2B  # noqa: E402
import issue2094_run as R  # noqa: E402
import issue2094_wellsep_bootstrap as W  # noqa: E402

logger = logging.getLogger("issue2094_gpu2_analysis")

REPO_ROOT = A.REPO_ROOT
DEFAULT_SCORES_DIR = REPO_ROOT / "eval_results/issue_2094/judge_gpu2/scores"
DEFAULT_FMETRICS_DIR = REPO_ROOT / "eval_results/issue_2094/f_metrics"
DEFAULT_ROLLOUTS = (
    REPO_ROOT
    / "data/issue_2094/judge_inputs/issue2094_singlepos/raw_completions"
    / f"{G2.GPU2_LABEL}/rollouts"
)
DEFAULT_STAGE_ROOT = REPO_ROOT / "data/issue_2094/gpu2_va_stage"
DEFAULT_PARTS_DIR = REPO_ROOT / "data/issue_2094/gpu2_analysis_parts"
DEFAULT_OUT_DIR = DEFAULT_FMETRICS_DIR / "gpu2"

VA_PREFIX = f"{G2.HF_GPU2_TENSORS}/va"
GPU2_ANCHORS_VA_IN_REPO = f"{G2.HF_GPU2_TENSORS}/anchors/va_anchors_gpu2.pt"
GATE_REPORT_IN_REPO = f"{G2.HF_GPU2_TENSORS}/manifests/gate_report.json"
CAPHIT_IN_REPO = f"{G2.HF_GPU2_TENSORS}/manifests/gpu2_caphit.json"
ANCHORS_TEXT_IN_REPO = f"{G2.HF_GPU2_TEXT}/anchors/anchors_gpu2.jsonl"
PARENT_ANCHORS_PT_IN_REPO = f"{R.HF_PREFIX}/analysis_tensors/anchors/va_anchors.pt"
# The parent judge run already staged the parent anchor rollout text here.
LOCAL_PARENT_ANCHORS_JSONL = (
    REPO_ROOT
    / "data/issue_2094/judge_inputs"
    / R.HF_PREFIX
    / "raw_completions/anchors/anchors.jsonl"
)
# fu2's stage already holds the parent anchor V_a — reuse before re-downloading.
FU2_PARENT_ANCHORS_PT = REPO_ROOT / "data/issue_2094/fu2_va_stage" / PARENT_ANCHORS_PT_IN_REPO

GATE_WAVE_FILES = tuple(
    f"{rid}.{G2.GATE_WAVE_SUFFIX}.{ext}"
    for rid in (A.COHERENCE_RUBRIC_ID, *G2.GATE_RUBRIC_IDS)
    for ext in ("scores.jsonl", "meta.json")
)

SMALLN_MIN_PAIRS = 3  # explicitly-labeled small-n regime; NEVER the headline bar

RC_OK = 0


# ── expected-grid enumeration (pure; pinned in tests) ───────────────────


def gpu2_pair_axis() -> list[str]:
    """The bootstrap pair axis: the 5 re-formed conv2 pairs, sorted."""
    return sorted(p.pair_id for p in G2B.build_gpu2_pairs())


def expected_family_tails(families: list[tuple[R.Block, R.Block]]) -> set[str]:
    """Every steered-vs-null read key ``setting|slot|lv|dose|vt|metric`` the
    gpu2 grid must produce (2 metrics per family: f_act + f_beh_prefix)."""
    tails: set[str] = set()
    for steered, _null in families:
        for metric in ("f_act", "f_beh_prefix"):
            tails.add(
                "|".join(
                    [
                        "matched_query",
                        steered.slot,
                        steered.layer_variant,
                        steered.dose,
                        steered.vec_type,
                        metric,
                    ]
                )
            )
    return tails


def expected_score_keys(
    families: list[tuple[R.Block, R.Block]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str, str]]]:
    """(coherence keys, behavior keys) the gpu2 judge waves must cover."""
    coh: set[tuple[str, str]] = set()
    beh: set[tuple[str, str, str, str]] = set()
    for fam in families:
        for block in fam:
            for pid in block.pair_ids:
                coh.add((block.key, pid))
                for side in ("a", "b"):
                    beh.add((block.key, pid, "prefix", side))
    return coh, beh


def analysis_regime() -> str:
    """Resume regime for the parts dir (every output-affecting knob, #722 r3)."""
    key = json.dumps(
        {
            "code": "gpu2-analysis-v1",
            "gpu2_regime_token": G2.GPU2_REGIME_TOKEN,
            "gpu2_sha": G2B.gpu2_manifest_sha(),
            "coherence_threshold": A.COHERENCE_THRESHOLD,
            "primary_read_layer": A.PRIMARY_READ_LAYER,
            "profiles": False,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def check_parts_regime(parts_manifest: Path, regime: str) -> set[str]:
    """Load the parts-dir done set; a regime mismatch HARD-refuses (#722 r3)."""
    if not parts_manifest.exists():
        return set()
    rec = json.loads(parts_manifest.read_text(encoding="utf-8"))
    if rec.get("regime") != regime:
        raise RuntimeError(
            f"gpu2 analysis parts at {parts_manifest.parent} carry a DIFFERENT regime "
            f"({rec.get('regime')} != {regime}) — quarantine or delete the parts dir"
        )
    return set(rec.get("done", []))


# ── staging (idempotent; GPU2Paths-shaped so the driver's gate readers
#    ``load_gate_scores`` / ``load_floor_rows`` work verbatim) ────────────


def stage_inputs(stage_root: Path, parent_anchors_pt: Path | None) -> dict:
    """Stage every HF input: gate scores+metas, conv2 anchor text + V_a, the
    pod gate/cap-hit manifests, the 300 grid V_a shards, and the parent
    anchor V_a. Per-file skip-if-present makes re-runs cheap."""
    paths = G2.GPU2Paths(out_root=stage_root / "gpu2_mirror")

    for name in GATE_WAVE_FILES:
        target = paths.judge_root / "scores" / name
        if not target.is_file():
            stage_hub_file(
                R.HF_DATA_REPO,
                f"{G2.HF_GPU2_TEXT}/judge_gate/scores/{name}",
                target,
                repo_type="dataset",
            )
    if not paths.anchors_file.is_file():
        stage_hub_file(
            R.HF_DATA_REPO, ANCHORS_TEXT_IN_REPO, paths.anchors_file, repo_type="dataset"
        )
    if not paths.anchors_va.is_file():
        stage_hub_file(
            R.HF_DATA_REPO, GPU2_ANCHORS_VA_IN_REPO, paths.anchors_va, repo_type="dataset"
        )
    if not paths.parent_anchors_file.is_file():
        if LOCAL_PARENT_ANCHORS_JSONL.is_file():
            paths.parent_anchors_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(LOCAL_PARENT_ANCHORS_JSONL, paths.parent_anchors_file)
        else:
            G2.stage_parent_anchors(paths, None)
    gate_report = paths.gate_report
    if not gate_report.is_file():
        stage_hub_file(R.HF_DATA_REPO, GATE_REPORT_IN_REPO, gate_report, repo_type="dataset")
    caphit_json = gate_report.parent / "gpu2_caphit.json"
    if not caphit_json.is_file():
        stage_hub_file(R.HF_DATA_REPO, CAPHIT_IN_REPO, caphit_json, repo_type="dataset")

    # Parent anchor V_a: prefer the fu2 stage's verified copy over a re-download.
    if parent_anchors_pt is None:
        if FU2_PARENT_ANCHORS_PT.is_file():
            parent_anchors_pt = FU2_PARENT_ANCHORS_PT
        else:
            parent_anchors_pt = stage_root / PARENT_ANCHORS_PT_IN_REPO
            if not parent_anchors_pt.is_file():
                stage_hub_file(
                    R.HF_DATA_REPO,
                    PARENT_ANCHORS_PT_IN_REPO,
                    parent_anchors_pt,
                    repo_type="dataset",
                )
    assert parent_anchors_pt.is_file(), parent_anchors_pt

    # Grid V_a shards (mirror-root layout — files land at stage_root/<repo path>).
    va_dir = stage_root / VA_PREFIX
    expected_slugs = sorted(b.slug for fam in G2.enumerate_gpu2_families(A.N_LAYERS) for b in fam)
    missing_va = [s for s in expected_slugs if not (va_dir / f"shard_{s}.pt").is_file()]
    if missing_va:
        logger.info(
            "[stage] fetching %d/%d gpu2 va shards from HF", len(missing_va), len(expected_slugs)
        )
        stage_hub_prefix(R.HF_DATA_REPO, VA_PREFIX, stage_root, repo_type="dataset")
    present = {p.name for p in va_dir.glob("shard_*.pt")}
    expected_names = {f"shard_{s}.pt" for s in expected_slugs}
    assert present == expected_names, (
        f"staged va shard set != expected {len(expected_names)} block slugs; missing "
        f"{sorted(expected_names - present)[:4]}, extra {sorted(present - expected_names)[:4]}"
    )

    staged_files = [
        *(paths.judge_root / "scores" / n for n in GATE_WAVE_FILES),
        paths.anchors_file,
        paths.anchors_va,
        paths.parent_anchors_file,
        gate_report,
        caphit_json,
        parent_anchors_pt,
        *(va_dir / n for n in expected_names),
    ]
    staged_bytes = sum(p.stat().st_size for p in staged_files)
    return {
        "paths": paths,
        "va_dir": va_dir,
        "parent_anchors_pt": parent_anchors_pt,
        "gate_report": gate_report,
        "caphit_json": caphit_json,
        "staged_bytes": staged_bytes,
    }


# ── deliverable 1: anchor stats + exact gate reproduction ───────────────


def recompute_gate(paths: G2.GPU2Paths, draws: int) -> tuple[list[dict], dict, dict]:
    """(sep_rows, verdict, provenance) via the driver's own gate machinery.

    Floor draws: the parent's bare anchor rollouts (the pod gate's floor
    side); ceiling draws: the conv2 anchors — the pod ``main()`` composition
    reproduced verbatim.
    """
    coh, beh = G2.load_gate_scores(paths)
    floor_rows = G2.load_floor_rows(paths, draws)
    ceiling_rows = [
        r
        for r in (
            json.loads(line) for line in paths.anchors_file.open(encoding="utf-8") if line.strip()
        )
        if r["context_id"].split("__")[0] == G2B.CONV2_PREFIX
    ]
    assert floor_rows and ceiling_rows, (len(floor_rows), len(ceiling_rows))
    draws_by_ctx: dict[str, list[int]] = {}
    for r in floor_rows + ceiling_rows:
        draws_by_ctx.setdefault(r["context_id"], []).append(r["draw"])
    sep_rows = G2.gate_separations(coh, beh, draws_by_ctx)
    verdict = G2.gate_verdict(sep_rows)
    prov = {"n_floor_rows": len(floor_rows), "n_ceiling_rows": len(ceiling_rows)}
    return sep_rows, verdict, prov


def assert_gate_reproduction(
    sep_rows: list[dict], verdict: dict, prov: dict, recorded: dict
) -> dict:
    """The recomputed separations + verdict must equal the pod's recorded
    ``gate_report.json`` EXACTLY (float-exact — same code path, same scores)."""
    rec_v = recorded["verdict"]
    for k in ("passed", "n_passing", "n_pairs", "min_abs_separation", "min_passing"):
        assert verdict[k] == rec_v[k], f"gate verdict field {k}: {verdict[k]} != {rec_v[k]}"
    assert verdict["per_pair"] == rec_v["per_pair"], (
        f"per-pair verdict mismatch: {verdict['per_pair']} != {rec_v['per_pair']}"
    )
    rec_rows = {r["pair_id"]: r for r in recorded["separations"]}
    assert set(rec_rows) == {r["pair_id"] for r in sep_rows}, "separation pair sets differ"
    for row in sep_rows:
        rec = rec_rows[row["pair_id"]]
        for k in ("separation", "floor", "ceiling", "context_a", "context_b", "setting", "kind"):
            assert row[k] == rec[k], (
                f"separation row {row['pair_id']} field {k}: {row[k]!r} != {rec[k]!r}"
            )
    for k in ("n_floor_rows", "n_ceiling_rows"):
        assert prov[k] == recorded[k], (k, prov[k], recorded[k])
    return {
        "passed": True,
        "compared": "verdict (all fields incl. per_pair) + per-pair separations "
        "(separation, floor, ceiling stats) + floor/ceiling row counts, float-exact",
        "recorded_regime_fp": recorded.get("regime_fp"),
        "recorded_judge_mode": recorded.get("judge_mode"),
    }


def wellsep_sets_from_verdict(verdict: dict) -> tuple[set[tuple[str, str]], set[str]]:
    """Gate-passing pairs (|sep| >= 0.5) as (pair_id, kind) / any-kind sets —
    the wellsep restriction for the bootstrap (q2 excluded downstream)."""
    passing = {p["pair_id"] for p in verdict["per_pair"] if p["passes"]}
    assert passing, "no gate-passing pairs — nothing to analyze"
    return {(pid, "prefix") for pid in passing}, passing


# ── deliverable 2: per-cell reduction (parent conventions, by import) ───


def assemble_gpu2_cells(
    families: list[tuple[R.Block, R.Block]],
    rollouts_dir: Path,
    va_dir: Path,
    parts_dir: Path,
    lk: A.JudgeLookups,
    pair_stats: dict[tuple[str, str], dict],
    anchor_va: dict,
    pairs_by_id: dict[str, BANK.Pair],
) -> list[dict]:
    """Per-cell F rows for the gpu2 blocks — ``A.assemble_shard_rows`` per
    shard, checkpointed per unit (300 units > ~50, code-style T2) with a
    regime-keyed resume manifest and a per-unit progress line."""
    regime = analysis_regime()
    parts_manifest = parts_dir / "parts_manifest.json"
    done_parts = check_parts_regime(parts_manifest, regime)

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
            rows, va, lk, pair_stats, anchor_va, pairs_by_id, profiles=False
        )
        A._write_jsonl_atomic(part, cell_rows)
        done_parts.add(block.slug)
        A._write_json_atomic(parts_manifest, {"regime": regime, "done": sorted(done_parts)})
        print(
            f"[gpu2-ftables] unit {k}/{len(blocks)} {block.slug} "
            f"elapsed={time.monotonic() - t0:.1f}s",
            flush=True,
        )

    out: list[dict] = []
    for block in sorted(blocks, key=lambda b: b.slug):
        out.extend(A._iter_jsonl(parts_dir / f"{block.slug}.jsonl"))
    assert len(out) == sum(b.n_cells for b in blocks), len(out)
    return out


def load_merged_anchor_va(parent_pt: Path, gpu2_pt: Path) -> dict:
    """Parent 15-context anchor V_a (the bare floors) + the 5 conv2 ceilings."""
    parent = A._load_anchor_va(SimpleNamespace(anchors_pt=parent_pt))
    gpu2 = A._load_anchor_va(SimpleNamespace(anchors_pt=gpu2_pt))
    overlap = set(parent) & set(gpu2)
    assert not overlap, f"anchor V_a context ids collide: {sorted(overlap)[:4]}"
    merged = {**parent, **gpu2}
    needed = {f"bare__{q}" for q in BANK.QUERY_ORDER} | {
        G2B.conv2_context_id(q) for q in BANK.QUERY_ORDER
    }
    missing = needed - set(merged)
    assert not missing, f"anchor V_a missing grid contexts: {sorted(missing)}"
    return merged


# ── deliverable 4: cap-hit cross-check (pod manifest is ground truth) ────


def crosscheck_gpu2_caphit(
    pooled: dict[tuple[str, str, str], dict], manifest: dict, subset: bool = False
) -> dict:
    """Recomputed pooled cap-hit (from rollout rows' own ``cap_hit``) must
    equal the pod's ``gpu2_caphit.json`` manifest exactly (counts + n)."""
    assert manifest["max_new_tokens"] == G2.GPU2_MAX_NEW_TOKENS, manifest["max_new_tokens"]
    m_by_key = {(c["slot"], c["layer_variant"], c["dose"]): c for c in manifest["cells"]}
    if subset:
        assert set(pooled) <= set(m_by_key), sorted(set(pooled) - set(m_by_key))
    else:
        assert set(m_by_key) == set(pooled), sorted(set(m_by_key) ^ set(pooled))
    for key, arms in pooled.items():
        for arm in R.ARMS:
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
        "against the pod manifest gpu2_caphit.json (counts exact)",
    }


# ── bootstrap battery (5-pair axis; A helpers reused) ───────────────────


def compute_gpu2_family_battery(
    rows: list[dict], ws: set[tuple[str, str]], ws_any: set[str], n_boot: int
) -> dict[str, dict]:
    """``W.compute_wellsep_families`` re-grained to the gpu2 pair axis (the 5
    re-formed pairs; the parent module's axis is the parent bank and would
    KeyError on gpu2 pair ids). Identical conventions: NaN-aware batched
    pair-clustered bootstrap, family keys via ``A._family_key``, seed
    ``A.BOOTSTRAP_SEED``, keep predicate ``W.wellsep_keep``."""
    pids = gpu2_pair_axis()
    pid_idx = {p: i for i, p in enumerate(pids)}
    fam_values: dict[str, np.ndarray] = {}
    for row in rows:
        assert row["setting"] == "matched_query", row["setting"]
        metrics = ["f_act"] + [f"f_beh_{k}" for k in (row.get("f_beh") or {})]
        for metric in metrics:
            key = A._family_key(row, metric)
            arr = fam_values.setdefault(key, np.full(len(pids), np.nan))
            if W.wellsep_keep(row["pair_id"], metric, ws, ws_any):
                arr[pid_idx[row["pair_id"]]] = A._cell_metric(row, metric)
    assert fam_values, "no gpu2 family values — empty cell rows?"
    keys = sorted(fam_values)
    values = np.stack([fam_values[k] for k in keys], axis=1)  # (n_pairs, n_fams)
    assert values.shape == (len(pids), len(keys)), values.shape
    boots = A.bootstrap_family_means_batched(values, n_boot, A.BOOTSTRAP_SEED)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        obs = np.nanmean(values, axis=0)
    out: dict[str, dict] = {}
    for j, key in enumerate(keys):
        col = boots[:, j]
        valid = col[~np.isnan(col)]
        out[key] = {
            "setting": "matched_query",
            "observed_mean": A._nan_to_none(obs[j]),
            "n_pairs_used": int((~np.isnan(values[:, j])).sum()),
            "ci_lo": float(np.percentile(valid, 2.5)) if valid.size else None,
            "ci_hi": float(np.percentile(valid, 97.5)) if valid.size else None,
            "n_valid_draws": int(valid.size),
        }
    return out


# ── verdict table (both conventions) ────────────────────────────────────


def _verdict_min_pairs(read: dict, compromised: bool, min_pairs: int) -> str:
    """fu2's ``_verdict`` logic with a parameterized pair floor (parity with
    ``FU2A._verdict`` at min_pairs == W.MIN_PAIRS_HEADLINE is test-pinned)."""
    if not read.get("comparable"):
        return "not_comparable"
    separated = bool(read["cis_disjoint"]) and read["direction"] == "steered_above"
    if not separated:
        return "not_separating"
    if read["n_pairs_used"] < min_pairs:
        return f"separating_lt{min_pairs}_pairs"
    return "separating_compromised" if compromised else "clean_separating"


def build_gpu2_verdict_table(
    reads_ws: dict[str, dict],
    reads_unres: dict[str, dict],
    caphit: dict[tuple[str, str, str], dict],
    qc: dict[tuple[str, str, str, str, str], dict],
    expected_tails: set[str],
) -> list[dict]:
    """One row per family read, under BOTH the >=5-pair headline convention
    (fu2 ``_verdict`` verbatim) and the labeled >=3-pair small-n regime."""
    assert set(reads_ws) == expected_tails, (
        f"wellsep read families != expected gpu2 enumeration; missing "
        f"{sorted(expected_tails - set(reads_ws))[:6]}, "
        f"extra {sorted(set(reads_ws) - expected_tails)[:6]}"
    )
    assert set(reads_unres) == expected_tails, (
        f"unrestricted read families != expected gpu2 enumeration; missing "
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
                "cap_hit_frac": {arm: pooled[arm]["cap_hit_frac"] for arm in R.ARMS},
                "compromised": compromised,
                "qc": {arm: qc[(slot, lv, dose, setting, arm)] for arm in R.ARMS},
                # HEADLINE convention (fu2 taxonomy verbatim; >=5 pairs).
                "verdict_headline": FU2A._verdict(rws, compromised),
                # Explicitly-labeled small-n regime (>=3 pairs) — NOT the headline.
                "verdict_smalln_ge3": _verdict_min_pairs(rws, compromised, SMALLN_MIN_PAIRS),
            }
        )
    return rows


def per_metric_summary(rows: list[dict]) -> dict[str, dict]:
    """Per-metric verdict counts under both conventions."""
    out: dict[str, dict] = {}
    for metric in sorted({r["metric"] for r in rows}):
        mrows = [r for r in rows if r["metric"] == metric]
        headline: dict[str, int] = {}
        smalln: dict[str, int] = {}
        for r in mrows:
            headline[r["verdict_headline"]] = headline.get(r["verdict_headline"], 0) + 1
            smalln[r["verdict_smalln_ge3"]] = smalln.get(r["verdict_smalln_ge3"], 0) + 1
        out[metric] = {
            "n_family_reads": len(mrows),
            "verdict_counts_headline": headline,
            "verdict_counts_smalln_ge3": smalln,
            "n_separating_smalln_clean": smalln.get("clean_separating", 0),
            "smalln_clean_families": sorted(
                r["family"] for r in mrows if r["verdict_smalln_ge3"] == "clean_separating"
            ),
        }
    return out


# ── deliverable 3: parent comparison ────────────────────────────────────


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties averaged) — numpy-only Spearman ingredient."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    _vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(_vals))
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation (average-rank ties); None below n=3."""
    if len(x) < 3:
        return None
    rx, ry = _avg_ranks(np.asarray(x, float)), _avg_ranks(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else None


def _gap(read: dict) -> float | None:
    if not read.get("comparable"):
        return None
    return read["steered_mean"] - read["null_mean"]


def parent_comparison(
    parent_wellsep: dict,
    breached_1024: set[tuple[str, str, str]],
    table: list[dict],
) -> dict:
    """The parent grid's matched-query ce reads for the SAME (variant x dose)
    families vs the gpu2 recovered-pair reads: direction + magnitude
    replication, with the fu1 1024-cap breach flag as the parent's own
    compromise label (the gpu2 grid ran cap-clean at 2048)."""
    sv = parent_wellsep["steered_vs_null"]
    rows = []
    for r in table:
        tail = r["family"]
        pread = sv.get(tail)
        assert pread is not None, f"parent wellsep read missing for {tail}"
        pcomp = (r["slot"], r["layer_variant"], r["dose"]) in breached_1024
        pverdict = FU2A._verdict(pread, pcomp)
        gap_p, gap_g = _gap(pread), _gap(r["wellsep"])
        rows.append(
            {
                "family": tail,
                "metric": r["metric"],
                "layer_variant": r["layer_variant"],
                "dose": r["dose"],
                "parent": {
                    "verdict": pverdict,
                    "compromised_1024_caphit": pcomp,
                    "n_pairs_used": pread.get("n_pairs_used"),
                    "gap": gap_p,
                    "cis_disjoint": pread.get("cis_disjoint"),
                    "direction": pread.get("direction"),
                },
                "gpu2": {
                    "verdict_headline": r["verdict_headline"],
                    "verdict_smalln_ge3": r["verdict_smalln_ge3"],
                    "n_pairs_used": r["wellsep"].get("n_pairs_used"),
                    "gap": gap_g,
                    "cis_disjoint": r["wellsep"].get("cis_disjoint"),
                    "direction": r["wellsep"].get("direction"),
                },
                "direction_match": (
                    pread.get("direction") == r["wellsep"].get("direction")
                    if pread.get("comparable") and r["wellsep"].get("comparable")
                    else None
                ),
            }
        )
    per_metric: dict[str, dict] = {}
    for metric in sorted({r["metric"] for r in rows}):
        mrows = [r for r in rows if r["metric"] == metric]
        clean = [r for r in mrows if r["parent"]["verdict"] == "clean_separating"]
        both_gaps = [
            (r["parent"]["gap"], r["gpu2"]["gap"])
            for r in mrows
            if r["parent"]["gap"] is not None and r["gpu2"]["gap"] is not None
        ]
        per_metric[metric] = {
            "n_families": len(mrows),
            "n_parent_clean_separating": len(clean),
            "n_parent_clean_gpu2_direction_match": sum(
                1 for r in clean if r["direction_match"] is True
            ),
            "n_parent_clean_gpu2_cis_disjoint": sum(
                1 for r in clean if r["gpu2"]["cis_disjoint"] is True
            ),
            "n_parent_clean_gpu2_smalln_clean": sum(
                1 for r in clean if r["gpu2"]["verdict_smalln_ge3"] == "clean_separating"
            ),
            "n_gap_pairs": len(both_gaps),
            "spearman_gap_parent_vs_gpu2": spearman(
                [g[0] for g in both_gaps], [g[1] for g in both_gaps]
            ),
            "mean_gap_parent": (float(np.mean([g[0] for g in both_gaps])) if both_gaps else None),
            "mean_gap_gpu2": (float(np.mean([g[1] for g in both_gaps])) if both_gaps else None),
        }
    return {
        "source": "eval_results/issue_2094/f_metrics/bootstrap_cis_wellsep.json "
        "(parent grid, max_new_tokens=1024; fu1 regen artifact exists at "
        "f_metrics/fu1/fu1_wellsep_bootstrap_regen.json for the breached cells)",
        "restriction": "matched_query x ce x vec_type A x the gpu2 grid's 30 layer "
        "variants x 5 doses x {f_act, f_beh_prefix} — the like-for-like subset",
        "pair_note": (
            "parent reads use the parent's 10 well-separated mq pairs of 15 (the 5 "
            "bare->conv pairs sat at |sep| < 0.5 and were excluded from the parent "
            "wellsep read); the gpu2 round re-forms those 5 pairs with the conv2 "
            "replacement prefix and recovers 4 of them (q2 still fails the gate) — "
            "so the gpu2 reads measure the previously-EXCLUDED pair class"
        ),
        "compromise_label": "parent: fu1 derive_breached_cells(fragility) pooled cells "
        "(steered 1024 cap-hit > 2%); gpu2: pooled steered 2048 cap-hit > 2%",
        "rows": rows,
        "per_metric": per_metric,
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
    ap.add_argument("--parent-anchors-pt", type=Path, default=None)
    ap.add_argument(
        "--limit-families",
        type=int,
        default=0,
        help="smoke dial: analyze only the first N families (0 = all 150); "
        "subset-scopes the coverage/crosscheck asserts — pair with a scratch "
        "--out-dir/--parts-dir, never the canonical ones",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    subset = args.limit_families > 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.parts_dir.mkdir(parents=True, exist_ok=True)

    # ── grid enumeration + fail-fast totals (driver constants, never re-derived)
    logger.info("[phase=gpu2a_enumerate] limit_families=%d", args.limit_families)
    families = G2.enumerate_gpu2_families(A.N_LAYERS)
    totals = R.grid_totals(families)
    assert totals == G2.EXPECTED_GPU2_TOTALS, (totals, G2.EXPECTED_GPU2_TOTALS)
    if subset:
        families = families[: args.limit_families]
        totals = R.grid_totals(families)
        logger.info("[enumerate] SUBSET %s", totals)
    pairs_by_id = G2B.gpu2_pairs_by_id()
    expected_coh, expected_beh = expected_score_keys(families)
    tails = expected_family_tails(families)

    # ── staging (idempotent) ────────────────────────────────────────────
    logger.info("[phase=gpu2a_stage]")
    staged = stage_inputs(args.stage_root, args.parent_anchors_pt)
    paths: G2.GPU2Paths = staged["paths"]
    logger.info("[stage] inputs present (%.1f MB staged total)", staged["staged_bytes"] / 1e6)

    # ── deliverable 1: anchor stats + exact gate reproduction ───────────
    logger.info("[phase=gpu2a_gate]")
    gate_metas = FU1A.check_wave_metas(paths.judge_root / "scores")
    sep_rows, verdict, prov = recompute_gate(paths, R.ANCHOR_DRAWS)
    recorded = json.loads(staged["gate_report"].read_text(encoding="utf-8"))
    reproduction = assert_gate_reproduction(sep_rows, verdict, prov, recorded)
    ws, ws_any = wellsep_sets_from_verdict(verdict)
    assert len(ws_any) == 4 and "mq--bare__q2--conv2__q2" not in ws_any, sorted(ws_any)
    logger.info(
        "[gate] reproduced exactly: passed=%s n_passing=%d/%d wellsep_pairs=%d",
        verdict["passed"],
        verdict["n_passing"],
        verdict["n_pairs"],
        len(ws_any),
    )
    repro = _repro()
    A._write_json_atomic(
        args.out_dir / "gpu2_anchor_stats.json",
        {
            "separations": sep_rows,
            "verdict": verdict,
            "reproduction_check": reproduction,
            **prov,
            "floor_source": "parent bare-context anchor rollouts (draw < "
            f"{R.ANCHOR_DRAWS}), re-judged pod-side in the gate waves — parent "
            "anchor SCORES untouched",
            "ceiling_source": f"{ANCHORS_TEXT_IN_REPO} (conv2 contexts, "
            f"K={R.ANCHOR_DRAWS} temp-1.0 unpatched draws)",
            "instrument": recorded.get("instrument"),
            "gate_waves": {
                w: {"n_items": m["regime"]["n_items"], "complete": m.get("complete")}
                for w, m in sorted(gate_metas.items())
            },
            "instrument_validity_note": (
                "fp-conv2's de-facto pilot is the pod gate waves: 100/100 scored per "
                "wave x 3 waves, 0 truncation, all end_turn (telemetry at HF "
                f"{G2.HF_GPU2_TEXT}/judge_gate/scores/*.gpu2anchors.meta.json)"
            ),
            "wellsep_restriction": {
                "min_abs_separation": G2.MIN_ABS_SEPARATION,
                "passing_pairs": sorted(ws_any),
                "excluded_pairs": sorted({p["pair_id"] for p in verdict["per_pair"]} - ws_any),
            },
            "repro": repro,
        },
    )

    # ── judge scores: wave metas + fail-loud routing + coverage ─────────
    logger.info("[phase=gpu2a_load]")
    metas = FU1A.check_wave_metas(args.scores_dir)
    n_meta_items = sum(m["regime"]["n_items"] for m in metas.values())
    rows_iter = (
        row for f in sorted(args.scores_dir.glob("*.scores.jsonl")) for row in A._iter_jsonl(f)
    )
    sc = FU1A.route_fu1_scores(rows_iter)
    assert not sc.s2_coh and not sc.s2_beh, "gpu2 waves must carry grid rows only"
    n_routed = len(sc.grid_coh) + len(sc.grid_beh)
    if subset:
        assert expected_coh <= set(sc.grid_coh), "coherence coverage misses subset keys"
        assert expected_beh <= set(sc.grid_beh), "behavior coverage misses subset keys"
    else:
        assert n_routed == n_meta_items, (n_routed, n_meta_items)
        assert set(sc.grid_coh) == expected_coh, (
            f"coherence score coverage != enumerated grid "
            f"(scores {len(sc.grid_coh)}, expected {len(expected_coh)})"
        )
        assert set(sc.grid_beh) == expected_beh, (
            f"behavior score coverage != enumerated grid "
            f"(scores {len(sc.grid_beh)}, expected {len(expected_beh)})"
        )
    n_score_none = sum(1 for v in sc.grid_coh.values() if v is None) + sum(
        1 for v in sc.grid_beh.values() if v is None
    )
    lk = A.JudgeLookups(grid_coh=sc.grid_coh, grid_beh=sc.grid_beh)
    logger.info(
        "[load] scores routed: grid_coh=%d grid_beh=%d none_scores=%d",
        len(sc.grid_coh),
        len(sc.grid_beh),
        n_score_none,
    )

    pair_stats = {(r["pair_id"], r["kind"]): r for r in sep_rows}
    anchor_va = load_merged_anchor_va(staged["parent_anchors_pt"], paths.anchors_va)

    # ── per-cell reduction (parent conventions) ─────────────────────────
    logger.info("[phase=gpu2a_ftables]")
    cell_rows = assemble_gpu2_cells(
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
    if not subset:
        assert len(steered_rows) == G2.EXPECTED_GPU2_TOTALS["cells_steered"], len(steered_rows)
        assert len(null_rows) == G2.EXPECTED_GPU2_TOTALS["cells_null"], len(null_rows)
    A._write_jsonl_atomic(args.out_dir / "gpu2_cells.jsonl", steered_rows)
    A._write_jsonl_atomic(args.out_dir / "gpu2_null_cells.jsonl", null_rows)
    logger.info("[phase=gpu2a_ftables_done] steered=%d null=%d", len(steered_rows), len(null_rows))

    # ── cap-hit pooling + manifest cross-check ──────────────────────────
    logger.info("[phase=gpu2a_caphit]")
    blocks = [b for fam in families for b in fam]
    rollout_rows: list[dict] = []
    for block in blocks:
        rollout_rows.extend(A._iter_jsonl(args.rollouts_dir / f"shard_{block.slug}.jsonl"))
    assert len(rollout_rows) == sum(b.n_cells for b in blocks), len(rollout_rows)
    caphit = FU1A.recompute_caphit_2048(rollout_rows)
    caphit_check = crosscheck_gpu2_caphit(
        caphit, json.loads(staged["caphit_json"].read_text(encoding="utf-8")), subset=subset
    )
    compromised_cells = sorted(
        key
        for key, arms in caphit.items()
        if arms["steered"]["cap_hit_frac"] > FU1.CAPHIT_TRIGGER_FRAC
    )
    pooled_frac = sum(arms[a]["cap_hit"] for arms in caphit.values() for a in R.ARMS) / max(
        1, sum(arms[a]["n"] for arms in caphit.values() for a in R.ARMS)
    )
    logger.info(
        "[caphit] pooled cells=%d pooled_frac=%.4f compromised (steered > %.0f%%): %d",
        len(caphit),
        pooled_frac,
        100 * FU1.CAPHIT_TRIGGER_FRAC,
        len(compromised_cells),
    )

    # ── bootstrap: wellsep-restricted (4 pairs) + unrestricted (5) ──────
    logger.info("[phase=gpu2a_bootstrap] n_boot=%d", args.n_boot)
    eligible, n_degenerate_excluded = A.bootstrap_eligible_rows(cell_rows)
    assert n_degenerate_excluded == 0, (
        f"{n_degenerate_excluded} gpu2 rows flagged degenerate-self — the gpu2 grid "
        "is matched_query x ce with parent-pool donors (never self) by construction"
    )
    fams_ws = compute_gpu2_family_battery(eligible, ws, ws_any, args.n_boot)
    reads_ws, summary_ws = W.steered_vs_null_reads(fams_ws)
    ws_all = {(pid, "prefix") for pid in gpu2_pair_axis()}
    ws_any_all = set(gpu2_pair_axis())
    fams_unres = compute_gpu2_family_battery(eligible, ws_all, ws_any_all, args.n_boot)
    reads_unres, summary_unres = W.steered_vs_null_reads(fams_unres)

    # ── verdict table + summaries ───────────────────────────────────────
    logger.info("[phase=gpu2a_verdicts]")
    qc = FU2A.family_qc(cell_rows)
    table = build_gpu2_verdict_table(reads_ws, reads_unres, caphit, qc, tails)
    metric_summary = per_metric_summary(table)

    parent_wellsep = json.loads(
        (args.fmetrics_dir / "bootstrap_cis_wellsep.json").read_text(encoding="utf-8")
    )
    assert parent_wellsep["B"] == A.BOOTSTRAP_B and parent_wellsep["seed"] == A.BOOTSTRAP_SEED, (
        "parent wellsep artifact B/seed != the pinned convention — not like-for-like"
    )
    if args.n_boot != parent_wellsep["B"]:
        # Smoke dial only: the gate COMPUTATION runs identically; the verdict
        # stays informational at non-production n_boot (#1345 gate-calibration).
        logger.warning(
            "[parent] gpu2 reads at n_boot=%d != parent B=%d — comparison labeled "
            "non-like-for-like (smoke dial; production uses the default)",
            args.n_boot,
            parent_wellsep["B"],
        )
    fragility = json.loads((REPO_ROOT / FU1.FRAGILITY_REL).read_text(encoding="utf-8"))
    breached = set(FU1.derive_breached_cells(fragility))
    parent = parent_comparison(parent_wellsep, breached, table)
    parent["parent_B"] = parent_wellsep["B"]
    parent["gpu2_n_boot"] = args.n_boot
    parent["like_for_like_B"] = args.n_boot == parent_wellsep["B"]

    A._write_json_atomic(
        args.out_dir / "gpu2_f_reads.json",
        {
            "B": args.n_boot,
            "seed": A.BOOTSTRAP_SEED,
            "resample_axis": "pairs (pair-clustered; the 5 re-formed conv2 pairs)",
            "max_new_tokens": G2.GPU2_MAX_NEW_TOKENS,
            "n_degenerate_excluded": n_degenerate_excluded,
            "restriction": {
                "min_abs_separation": G2.MIN_ABS_SEPARATION,
                "f_beh": "pair kept iff its (pair, prefix) gate |separation| >= floor",
                "f_act": "pair kept iff well-separated on >= 1 rubric kind (mq has "
                "one kind, prefix — same 4-pair set)",
                "n_wellsep_pair_kinds": len(ws),
                "n_wellsep_pairs_any_kind": len(ws_any),
                "small_n_caveat": (
                    f"only {len(ws_any)} of 5 pairs pass the gate (q2 fails at "
                    "|sep| = 0.004), so every wellsep read has n_pairs_used = "
                    f"{len(ws_any)} < MIN_PAIRS_HEADLINE = {W.MIN_PAIRS_HEADLINE}; "
                    "CIs are wide and the headline convention caps verdicts at "
                    "separating_lt5_pairs — see gpu2_summary.json for both columns"
                ),
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
                "pooled_cap_hit_frac": pooled_frac,
                "compromised_cells": [list(c) for c in compromised_cells],
            },
            "note": (
                "gpu2_mq_replacement_prefix per-family reads: per-cell reduction via "
                "the parent ftables conventions (issue2094_analysis.assemble_shard_rows "
                "reused by import: batched F_act over the gpu2 V_a shards, anchored "
                "per-kind F_beh from the fp-bare/fp-conv2 contrasts, coherence gating "
                ">60, drops counted never coerced); anchor floor/ceiling stats from the "
                "pod gate waves via issue2094_gpu2.gate_separations (reused by import); "
                "bootstrap via A.bootstrap_family_means_batched (batched index-GEMM); "
                "the unrestricted companion is the same battery with all 5 pairs kept"
            ),
            "repro": repro,
        },
    )
    A._write_json_atomic(
        args.out_dir / "gpu2_summary.json",
        {
            "B": args.n_boot,
            "seed": A.BOOTSTRAP_SEED,
            "max_new_tokens": G2.GPU2_MAX_NEW_TOKENS,
            "verdict_definition": (
                "verdict_headline (fu2 taxonomy verbatim): clean_separating = steered "
                "vs shuffled-donor-null 95% pair-clustered bootstrap CIs disjoint with "
                "steered above, on well-separated pairs (|sep| >= "
                f"{G2.MIN_ABS_SEPARATION}), >= {W.MIN_PAIRS_HEADLINE} pairs, and the "
                f"family's pooled steered cap-hit <= {FU1.CAPHIT_TRIGGER_FRAC:.0%}; "
                f"with only {len(ws_any)} gate-passing pairs every separating read "
                "caps at separating_lt5_pairs under this convention. "
                f"verdict_smalln_ge3: the SAME logic at a >= {SMALLN_MIN_PAIRS}-pair "
                "floor — an explicitly-labeled small-n regime (n_pairs_used = "
                f"{len(ws_any)}; wide CIs), never the headline bar; per-family "
                "alpha=0.05, uncorrected for multiplicity"
            ),
            "verdict_table": table,
            "per_metric": metric_summary,
            "compromised_cells": [list(c) for c in compromised_cells],
            "parent_comparison": parent,
            "gate": {
                "verdict": verdict,
                "reproduction_check": reproduction,
            },
            "provenance": {
                "followup_label": "gpu2_mq_replacement_prefix",
                "grid": "ce slot x 30 layer variants x 5 doses x Type-A x "
                "{steered, donor-null} over the 5 re-formed conv2 pairs "
                "(150 families / 300 blocks / 1,500 cells at max_new_tokens=2048)",
                "judge_scores": str(args.scores_dir.relative_to(REPO_ROOT)),
                "instrument_validity_note": (
                    "fp-conv2's de-facto pilot is the pod gate waves: 100/100 scored "
                    "per wave x 3 waves, 0 truncation, all end_turn (telemetry at HF "
                    f"{G2.HF_GPU2_TEXT}/judge_gate/scores/*.gpu2anchors.meta.json); "
                    "the grid waves scored 1500/1500 per rubric with 0 drops"
                ),
                "n_grid_score_none": n_score_none,
            },
            "repro": repro,
        },
    )
    logger.info(
        "[phase=gpu2a_done] reads=%d smalln_clean per metric: %s",
        len(table),
        {m: metric_summary[m]["n_separating_smalln_clean"] for m in metric_summary},
    )
    return RC_OK


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
