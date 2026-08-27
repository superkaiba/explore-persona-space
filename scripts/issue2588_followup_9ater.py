#!/usr/bin/env python3
"""Issue #2588 Step 9a-ter zero-GPU free-analysis round (VM-side, CPU-only).

Three registered follow-up analyses over the harvested per-cell fit artifacts
(``--fits-dir``, default the local hub mirror harvested at HF revision
270dc35039d790f077b7b02b455f1ef57905c708):

- ``q38-truncfree``: truncation-free recompute of the q38_27b_b hard-set
  (GPQA) transfer read. All 726 kept rows are below-cap (verified
  finish_reason == "stop" over the staged parsed pool); the think-cap drops in
  ``dropped_row_ids.json`` removed 264/990 rows NON-uniformly, so the read is
  additionally restricted to TRUNCATION-FREE question clusters (all 5 rollout
  seeds kept) and compared, on identical rows, between the original labels and
  the length-residualized labels (``resid_cot_boundary.json`` ->
  ``gpqa_resid``). qid-clustered bootstrap CIs (B=2000, seed 42, vectorized).
  Limitation (recorded in the output): per-row hit indicators are frozen
  against the FULL 726-row retrieval pool; a pool-restricted recompute needs
  prediction vectors, which are not persisted (same class as the P3 abort).

- ``banked-intrusion``: closes the language_intrusion_audit generic-pool
  deferral for the two banked cells (q25_7b_a, q35_9b_a) by scanning the
  banked #2330 cap2048 test-pool texts (issue2330_matched/<store>/test_1000/
  raw_completions, pinned at PC.BANKED_REVISION) with the SAME scan spec as
  the fresh cells (issue2588_trend scan: completion matches the CJK class
  while the prompt does not), then computing the fired_overlap /
  acc1_raw / acc1_zeroed / acc1_excluded sensitivity from the cells'
  perrow_prompt_last hits. Derived per-row counts are cross-checked against
  the #2330 aggregates in eval_results/issue_2330/cap2048/
  intrusion_cap2048.json (fail-loud). ``--patch-trend-summary`` replaces the
  two deferral stubs in trend_summary.json with the computed blocks.

- ``abort-record``: durable evidence record for the ABORTED nested
  layer-re-selection bootstrap (P3): the required per-row per-layer
  correctness indicators do not exist in any banked artifact (percell files
  carry aggregate kNN reads only; frozen per-layer maps are not persisted),
  and rebuilding them means ~150 GB of activation staging + ~190 per-layer
  ridge refits — off-VM by the 50 GB rule; reported needs-gpu.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
for p in (str(_SCRIPTS), str(_REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps BEFORE numpy import (VM-side entrypoint)

import numpy as np  # noqa: E402

import issue2588_panel_common as PC  # noqa: E402
import issue2588_trend as T  # noqa: E402

DEFAULT_FITS_DIR = (
    _REPO_ROOT / "eval_results" / "issue_2588" / "hub_mirror" / PC.PANEL_PREFIX / "fits"
)
DEFAULT_OUT_ROOT = _REPO_ROOT / "eval_results" / "issue_2588" / "followup_9ater"
DEFAULT_I2330_TEXT_ROOT = _REPO_ROOT / "data" / "issue_2588" / "i2330_text"
DEFAULT_I2330_INTRUSION = (
    _REPO_ROOT / "eval_results" / "issue_2330" / "cap2048" / "intrusion_cap2048.json"
)
DEFAULT_TREND_SUMMARY = _REPO_ROOT / "eval_results" / "issue_2588" / "trend_summary.json"

# Banked cells -> (#2330 store prefix key in PC.BANKED_CAP2048, expected #2330
# aggregate key suffix in intrusion_cap2048.json's intrusion_scan block).
BANKED_CELLS = {"q25_7b_a": ("q25_7b", "7b"), "q35_9b_a": ("q35_9b", "9b")}
Q38_CELL = "q38_27b_b"
Q38_POS = "cot_boundary"


def _meta() -> dict:
    """Reproducibility metadata block (git provenance + env versions)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        "issue": PC.TASK_ID,
        "round": "9a-ter-free-analysis",
        "numpy": np.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **as_metadata_dict(git_provenance(), phase="followup-9ater"),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# P1 — q38-think truncation-free GPQA recompute + residualization decomposition
# ---------------------------------------------------------------------------


def _qid_cluster_boot(
    qids: list[str],
    mask: np.ndarray,
    hits_o: np.ndarray,
    hits_r: np.ndarray,
    draws: int,
    seed: int,
) -> dict:
    """qid-clustered bootstrap over the masked row subset, vectorized across
    draws: per-qid (row count, orig-hit sum, resid-hit sum) are precomputed
    once; each draw resamples qid clusters with replacement and reduces via
    fancy indexing. The orig/resid CIs share draws, so the delta CI is paired
    by construction."""
    sub_q = sorted({q for q, m in zip(qids, mask, strict=True) if m})
    q_index = {q: i for i, q in enumerate(sub_q)}
    c = np.zeros(len(sub_q))
    s_o = np.zeros(len(sub_q))
    s_r = np.zeros(len(sub_q))
    for q, m, ho, hr in zip(qids, mask, hits_o, hits_r, strict=True):
        if m:
            i = q_index[q]
            c[i] += 1
            s_o[i] += ho
            s_r[i] += hr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sub_q), size=(draws, len(sub_q)))
    denom = c[idx].sum(axis=1)
    assert (denom > 0).all(), "bootstrap draw with zero rows"
    acc_o = s_o[idx].sum(axis=1) / denom
    acc_r = s_r[idx].sum(axis=1) / denom
    delta = acc_r - acc_o

    def ci(x: np.ndarray) -> list[float]:
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {
        "draws": draws,
        "seed": seed,
        "n_clusters": len(sub_q),
        "acc1_orig_ci95": ci(acc_o),
        "acc1_resid_ci95": ci(acc_r),
        "delta_resid_minus_orig_ci95": ci(delta),
    }


def analysis_q38_truncfree(args: argparse.Namespace) -> dict:
    """Recompute the q38_27b_b GPQA transfer read truncation-free and compare
    original vs length-residualized labels on identical row subsets."""
    fits_dir = Path(args.fits_dir)
    cell_dir = fits_dir / Q38_CELL
    perrow_p = cell_dir / f"gpqa_perrow_{Q38_POS}.json"
    resid_p = cell_dir / f"resid_{Q38_POS}.json"
    dropped_p = cell_dir / "dropped_row_ids.json"
    pr = json.loads(perrow_p.read_text(encoding="utf-8"))
    resid_full = json.loads(resid_p.read_text(encoding="utf-8"))
    rs = resid_full["gpqa_resid"]
    dr = json.loads(dropped_p.read_text(encoding="utf-8"))
    shipped = json.loads((cell_dir / f"gpqa_transfer_{Q38_POS}.json").read_text(encoding="utf-8"))

    rows, qids = list(pr["row_ids"]), list(pr["qids"])
    assert list(rs["row_ids"]) == rows, "resid gpqa rows misaligned with perrow rows"
    hits_o = np.asarray(pr["same_q_hit"], dtype=float)
    hits_r = np.asarray(rs["same_q_hit"], dtype=float)
    n = len(rows)
    assert hits_o.shape == hits_r.shape == (n,), (hits_o.shape, hits_r.shape, n)

    # Verify every kept row is genuinely below-cap (finish_reason == "stop").
    fr_counts: Counter[str] = Counter()
    n_parsed = 0
    for f in sorted((cell_dir / "parsed").glob("gpqa_s*.jsonl")):
        for row in T._iter_jsonl(f):
            fr_counts[str(row.get("finish_reason"))] += 1
            n_parsed += 1
    assert n_parsed == n, f"parsed pool rows {n_parsed} != perrow rows {n}"
    kept_all_stop = set(fr_counts) == {"stop"}

    csize = Counter(qids)
    gpqa_drops = {k: v for k, v in dr["dropped_row_ids"].items() if k.startswith("gpqa_")}
    n_dropped = sum(len(v) for v in gpqa_drops.values())
    n_seeds = PC.GPQA_N_ROLLOUTS
    full_q = {q for q, c in csize.items() if c == n_seeds}
    mask_full = np.ones(n, dtype=bool)
    mask_tf = np.asarray([q in full_q for q in qids], dtype=bool)

    def read(mask: np.ndarray) -> dict:
        sub_q = [q for q, m in zip(qids, mask, strict=True) if m]
        return {
            "n_rows": int(mask.sum()),
            "n_questions": len(set(sub_q)),
            "acc1_orig": float(hits_o[mask].mean()),
            "acc1_resid": float(hits_r[mask].mean()),
            "delta_resid_minus_orig": float((hits_r[mask] - hits_o[mask]).mean()),
            # producer formula (matches the shipped same_question_chance
            # 5/726): nominal cluster size / full pool size.
            "chance_producer_formula": n_seeds / n,
            # composition-exact chance under a uniformly random pool NN.
            "chance_exact_excl_self": float(np.mean([(csize[q] - 1) / (n - 1) for q in sub_q])),
            "chance_exact_incl_self": float(np.mean([csize[q] / n for q in sub_q])),
        }

    by_size: dict[str, dict] = {}
    for size in sorted(set(csize.values())):
        m = np.asarray([csize[q] == size for q in qids], dtype=bool)
        by_size[str(size)] = {
            "n_rows": int(m.sum()),
            "n_questions": sum(1 for q, c in csize.items() if c == size),
            "acc1_orig": float(hits_o[m].mean()),
            "acc1_resid": float(hits_r[m].mean()),
        }

    out = {
        "meta": _meta(),
        "cell": Q38_CELL,
        "input_position": Q38_POS,
        "layer_star": int(pr["layer_star"]),
        "inputs": {
            str(p.relative_to(_REPO_ROOT)): _sha256_file(p) for p in (perrow_p, resid_p, dropped_p)
        },
        "kept_rows_all_finish_stop": bool(kept_all_stop),
        "finish_reason_counts": dict(fr_counts),
        "drop_stats": {
            "n_gpqa_rows_nominal": n + n_dropped,
            "n_kept": n,
            "n_dropped": n_dropped,
            "dropped_per_stage": {k: len(v) for k, v in sorted(gpqa_drops.items())},
            "kept_cluster_size_dist": {
                str(s): c for s, c in sorted(Counter(csize.values()).items())
            },
        },
        "resid_covariates": resid_full["covariates"],
        "reads": {
            "kept_full_pool": read(mask_full),
            "truncation_free_complete_clusters": read(mask_tf),
        },
        "bootstrap": {
            "kept_full_pool": _qid_cluster_boot(
                qids, mask_full, hits_o, hits_r, args.draws, args.seed
            ),
            "truncation_free_complete_clusters": _qid_cluster_boot(
                qids, mask_tf, hits_o, hits_r, args.draws, args.seed
            ),
        },
        "acc1_by_kept_cluster_size": by_size,
        "shipped_reference": {
            "same_question_acc1_cos": float(shipped["same_question_acc1_cos"]),
            "same_question_chance": float(shipped["same_question_chance"]),
            "gpqa_resid_same_q_acc1": float(rs["same_question_acc1_cos"]),
        },
        "notes": [
            "Per-row hit indicators are frozen against the FULL 726-row kept retrieval "
            "pool; subset reads restrict the QUERY rows only. A pool-restricted "
            "recompute needs prediction vectors, which are not persisted (same "
            "artifact gap as the aborted layer-re-selection bootstrap).",
            "The shipped same_question_chance uses the producer formula "
            "n_rollouts/n_pool (5/726); composition-exact chance values are reported "
            "alongside it.",
            "resid labels = ridge transfer read after residualizing "
            "log_prompt/answer/think token counts out of the answer states "
            "(resid_cot_boundary.json), rows identical to the original read.",
        ],
    }
    return out


# ---------------------------------------------------------------------------
# P2 — banked-cell generic-pool CJK intrusion scan (q25_7b_a, q35_9b_a)
# ---------------------------------------------------------------------------


def _stage_banked_test_text(model_key: str, text_root: Path) -> list[Path]:
    """Fetch (if absent) the banked #2330 cap2048 test-pool raw-completion
    chunks, pinned at PC.BANKED_REVISION; returns the local chunk paths."""
    store_prefix = PC.BANKED_CAP2048[model_key]
    rel_dir = f"{store_prefix}/test_1000/raw_completions"
    local_dir = text_root / rel_dir
    chunks = sorted(local_dir.glob("shard*_chunk*.json"))
    if chunks:
        return chunks
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path, retry_transient

    api = HfApi()
    names = sorted(
        p
        for p in list_hf_files_under_path(
            api, PC.HF_DATA_REPO, rel_dir, repo_type="dataset", revision=PC.BANKED_REVISION
        )
        if p.endswith(".json")
    )
    assert names, f"no raw-completion chunks under {rel_dir}@{PC.BANKED_REVISION}"
    for name in names:
        retry_transient(
            lambda name=name: hf_hub_download(
                PC.HF_DATA_REPO,
                name,
                repo_type="dataset",
                revision=PC.BANKED_REVISION,
                local_dir=str(text_root),
            ),
            what=f"hf_hub_download banked chunk {name}",
        )
    chunks = sorted(local_dir.glob("shard*_chunk*.json"))
    assert chunks, f"staging failed for {rel_dir}"
    return chunks


def _scan_banked_cell(cell: str, model_key: str, fits_dir: Path, text_root: Path) -> dict:
    """Fresh-cell-schema generic_test block for one banked cell (the
    issue2588_trend.augment_language_intrusion generic branch, applied to the
    banked #2330 test-pool texts)."""
    chunks = _stage_banked_test_text(model_key, text_root)
    rows: list[dict] = []
    for c in chunks:
        rows.extend(json.loads(c.read_text(encoding="utf-8"))["rows"])
    n_expected = PC.EXPECTED_SPLIT_COUNTS["test_1000"]
    assert len(rows) == n_expected, (cell, len(rows))
    assert {r["ci"] for r in rows} == set(range(n_expected)), f"{cell}: ci coverage gap"

    cjk = T._cjk_re()
    intr: list[str] = []
    ans_intr: list[str] = []
    cjk_prompt = 0
    for r in sorted(rows, key=lambda r: int(r["ci"])):
        rid = f"test_1000_{int(r['ci'])}"
        if cjk.search(r["prompt"]):
            cjk_prompt += 1
            continue
        text = r["response"]
        if cjk.search(text):
            intr.append(rid)
            # arm-a parse mode is "off": the answer span is the whole stripped
            # completion (PC.segment_completion_arm), mirroring the fresh-cell
            # n_intruded_answer_span semantics.
            _wf, _reason, _cot, ans = PC.segment_completion_arm(text, "off")
            if T._span_has_cjk(text, ans):
                ans_intr.append(rid)

    block = {
        "n_rows": len(rows),
        "n_cjk_prompt_excluded": cjk_prompt,
        "n_intruded": len(intr),
        "n_intruded_answer_span": len(ans_intr),
        "intruded_row_ids": intr,
    }
    pr = json.loads((fits_dir / cell / "perrow_prompt_last.json").read_text(encoding="utf-8"))
    hits = dict(zip(pr["row_ids"], pr["hit1_cos"], strict=True))
    in_pool = [r for r in intr if r in hits]
    overlap = sum(hits[r] for r in in_pool)
    n_pool = len(hits)
    kept = {r: h for r, h in hits.items() if r not in set(in_pool)}
    block["fired_overlap"] = int(overlap)
    block["acc1_raw"] = sum(hits.values()) / n_pool
    block["acc1_zeroed"] = (sum(hits.values()) - overlap) / n_pool
    block["acc1_excluded"] = sum(kept.values()) / len(kept) if kept else None
    block["source"] = (
        f"banked #2330 cap2048 test pool ({PC.BANKED_CAP2048[model_key]}/test_1000/"
        f"raw_completions @ {PC.BANKED_REVISION}); scanned with the #2588 scan_spec by "
        "scripts/issue2588_followup_9ater.py (9a-ter round; deferral closed)"
    )
    block["chunk_sha256"] = {c.name: _sha256_file(c) for c in chunks}
    return block


def analysis_banked_intrusion(args: argparse.Namespace) -> dict:
    """Close the generic-pool intrusion deferral for the two banked cells and
    cross-check the derived per-row counts against the #2330 aggregates."""
    fits_dir = Path(args.fits_dir)
    text_root = Path(args.i2330_text_root)
    agg = json.loads(Path(args.i2330_intrusion).read_text(encoding="utf-8"))["intrusion_scan"]

    cells: dict[str, dict] = {}
    intruded_ci: dict[str, set[int]] = {}
    for cell, (model_key, agg_suffix) in BANKED_CELLS.items():
        block = _scan_banked_cell(cell, model_key, fits_dir, text_root)
        cells[cell] = {"generic_test": block}
        intruded_ci[agg_suffix] = {int(r.rsplit("_", 1)[1]) for r in block["intruded_row_ids"]}
        assert block["n_intruded"] == agg[f"n_intruded_{agg_suffix}"], (
            f"{cell}: derived n_intruded {block['n_intruded']} != #2330 aggregate "
            f"{agg[f'n_intruded_{agg_suffix}']}"
        )
        assert block["n_cjk_prompt_excluded"] == agg["n_prompts_with_cjk"], cell
        assert block["n_rows"] - block["n_cjk_prompt_excluded"] == agg["n_eligible"], cell

    either = intruded_ci["7b"] | intruded_ci["9b"]
    both = intruded_ci["7b"] & intruded_ci["9b"]
    assert len(either) == agg["n_intruded_either"], (len(either), agg["n_intruded_either"])
    assert len(both) == agg["n_intruded_both"], (len(both), agg["n_intruded_both"])

    return {
        "meta": _meta(),
        "scan_spec": (
            "row intruded iff completion matches [\\u4e00-\\u9fff\\u3400-\\u4dbf"
            "\\uf900-\\ufaff\\u3040-\\u30ff\\uac00-\\ud7af] AND its prompt does not "
            "(the #2588 fresh-cell scan spec, applied to the banked #2330 cap2048 "
            "test pools)"
        ),
        "cells": cells,
        "cross_check_vs_issue2330": {
            "artifact": str(Path(args.i2330_intrusion).relative_to(_REPO_ROOT)),
            "n_prompts_with_cjk": agg["n_prompts_with_cjk"],
            "n_eligible": agg["n_eligible"],
            "n_intruded_7b": agg["n_intruded_7b"],
            "n_intruded_9b": agg["n_intruded_9b"],
            "n_intruded_either": agg["n_intruded_either"],
            "n_intruded_both": agg["n_intruded_both"],
            "all_matched": True,
        },
    }


def patch_trend_summary(args: argparse.Namespace, banked: dict) -> dict:
    """Replace the two banked-cell generic_test deferral stubs in
    trend_summary.json with the computed blocks (idempotent; refuses an
    unexpected pre-existing block)."""
    p = Path(args.trend_summary)
    summary = json.loads(p.read_text(encoding="utf-8"))
    audit_cells = summary["language_intrusion_audit"]["cells"]
    patched = []
    for cell, rec in banked["cells"].items():
        new_block = rec["generic_test"]
        old = audit_cells[cell].get("generic_test", {})
        if old == new_block:
            continue  # idempotent re-run
        assert "status" in old and "banked #2330" in str(old.get("status", "")), (
            f"{cell}: existing generic_test block is neither the deferral stub nor "
            f"this round's output — refusing to clobber: {old}"
        )
        audit_cells[cell]["generic_test"] = new_block
        patched.append(cell)
    summary["meta"].setdefault("augmented_9ater", {})
    summary["meta"]["augmented_9ater"].update(
        {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "round": "9a-ter-free-analysis",
            "note": (
                "banked-cell generic-pool intrusion deferral closed from the #2330 "
                f"cap2048 texts @ {PC.BANKED_REVISION} (scripts/issue2588_followup_9ater.py)"
            ),
        }
    )
    # indent=1 + ensure_ascii=True match the committed producer byte format
    # (issue2588_trend.py round-1 `json.dumps(summary, indent=1)`), keeping the
    # diff scoped to the patch.
    from explore_persona_space.atomic_io import write_json_atomic as _write_json_atomic

    _write_json_atomic(p, summary, indent=1, ensure_ascii=True)
    return {"patched_cells": patched, "path": str(p)}


# ---------------------------------------------------------------------------
# P3 — layer-re-selection bootstrap: abort-evidence record (needs-gpu)
# ---------------------------------------------------------------------------


def analysis_abort_record(args: argparse.Namespace) -> dict:
    """Durable record that the nested layer-re-selection bootstrap cannot be
    computed from banked artifacts; verifies mechanically that per-layer knn
    reads carry no per-row indicators."""
    fits_dir = Path(args.fits_dir)
    probes = {}
    for cell, pos, layer in (
        ("q35_27b_a", "prompt_last", "L00"),
        ("q38_27b_b", "cot_boundary", "L50"),
    ):
        d = json.loads((fits_dir / cell / f"percell_{pos}_{layer}.json").read_text())
        for split in ("knn_test", "knn_val"):
            keys = set(d[split]["ridge"]["cosine"])
            assert not keys & {"row_ids", "hits", "hit1_cos", "per_row", "same_q_hit"}, (
                cell,
                split,
                keys,
            )
        probes[f"{cell}/{layer}"] = sorted(d["knn_test"]["ridge"]["cosine"])
    return {
        "meta": _meta(),
        "status": "aborted-needs-gpu",
        "analysis": "nested layer-re-selection bootstrap (column contrasts, both arms)",
        "reason": (
            "requires per-row per-layer correctness indicators for validation "
            "(per-draw layer re-selection) and test (paired contrast). No banked "
            "artifact carries them: percell_<pos>_LXX.json holds aggregate kNN reads "
            "only (verified below), perrow_<pos>.json exists at layer_star only, and "
            "the frozen per-layer ridge maps are not persisted. Rebuilding the "
            "indicators means re-fitting ~190+ per-layer maps (6 column cells x "
            "~27-33 swept layers, n_train ~8.3-10k, d=5120) from the raw activation "
            "shards under issue2588_capability_panel/<model>/<arm>/analysis_tensors "
            "(~34.4 GB nothink + ~17.1 GB think for q35_27b alone; ~150 GB across "
            "the 3-model column) — over the 50 GB VM footprint rule and far beyond "
            "the zero-GPU free-analysis budget."
        ),
        "verified_local_evidence": {
            "percell_knn_read_keys": probes,
            "harvest_revision": "270dc35039d790f077b7b02b455f1ef57905c708",
        },
        "restage_recipe": (
            "pod round (eval/lora-7b-class GPU or cpu-bigmem): stage the 6 column "
            "cells' capture stores, re-fit the swept per-layer ridge maps with the "
            "production fit core (scripts/issue2588_run_cell.py path), persist "
            "per-row per-layer val/test hit indicators (small JSONs), then run the "
            "vectorized draws x layers re-selection bootstrap on the VM."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ANALYSES = {
    "q38-truncfree": ("q38_truncfree_gpqa", analysis_q38_truncfree),
    "banked-intrusion": ("banked_cjk_intrusion", analysis_banked_intrusion),
    "abort-record": ("layer_reselection_bootstrap", analysis_abort_record),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", choices=[*ANALYSES, "all"], default="all")
    ap.add_argument("--fits-dir", default=str(DEFAULT_FITS_DIR))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--i2330-text-root", default=str(DEFAULT_I2330_TEXT_ROOT))
    ap.add_argument("--i2330-intrusion", default=str(DEFAULT_I2330_INTRUSION))
    ap.add_argument("--trend-summary", default=str(DEFAULT_TREND_SUMMARY))
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--patch-trend-summary",
        action="store_true",
        help="replace the banked-cell generic_test deferral stubs in trend_summary.json",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)

    names = list(ANALYSES) if args.analysis == "all" else [args.analysis]
    for name in names:
        t0 = time.time()
        subdir, fn = ANALYSES[name]
        out = fn(args)
        out_dir = Path(args.out_root) / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subdir}.json"
        PC.write_json_atomic(out_path, out)
        print(f"[9ater] {name} -> {out_path} elapsed={time.time() - t0:.1f}s", flush=True)
        if name == "banked-intrusion" and args.patch_trend_summary:
            res = patch_trend_summary(args, out)
            print(f"[9ater] trend_summary patched: {res}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
