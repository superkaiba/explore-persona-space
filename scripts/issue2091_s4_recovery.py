"""#2091 fu1: recover the hallucination single-sample (S4) behavioral column.

The shipped R4 run's ``stage_packed_lookup`` read packed labeling shards with
``iter_jsonl`` — i.e. the ``pack_raw_tree`` WRAPPER rows ``{"src", "doc"}`` —
so ``doc.get("context_id")`` was None on every line, the (cid, rollout_k)
lookup stayed EMPTY, and every S4 three-way label resolved to
``missing_packed_row`` (the single column shipped n=0 across hal_train /
hal_nqopen / hal_simpleqa; r4_grids.json ``behavioral_rho_L19.*.single``).

This ANALYSIS-ONLY recovery (no training, no generation, no judge calls):
  1. re-stages the banked inputs (contexts tree, 26 packed hallucination
     labeling shards, 4 per-draw abstain tables) at the pinned revision;
  2. proves the root cause mechanically (wrapper-row census: 0 top-level
     context_ids; the shipped code path yields an empty lookup);
  3. reconstructs the SAME deterministic picks k(c) the R4 run used
     (per_rollout_scores is absent for every hallucination banked row, so
     every pick is the seeded ``rng_for(f"s4::{cid}")`` branch);
  4. recomputes the S4 labels through the FIXED ``packed_lookup_rows`` join
     + the banked abstain tables (``fits.s4_single_draw_label`` unchanged);
  5. persists per-context labels/picks + per-rung counts + the single-sample
     DV column, plus DV-vs-DV validation rhos (single vs avg_k5 / greedy,
     group bootstrap) — no ridge / map / probe is fit anywhere here; the
     grid's method-family rho cells for ``single`` still require the banked
     capture-store x vectors + a re-application of the already-fitted
     supervised read at its persisted lambda (out of scope this round).

Run (VM, ~0.31 GB staging under data/issue_2091/hf_dl/):
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
  uv run python scripts/issue2091_s4_recovery.py
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Script-mode guard (gotchas.md #823): repo root on sys.path, sentinel-checked."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2091_analysis.py").is_file(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

from scripts import issue2091_analysis as A  # noqa: E402

BEHAVIOR = "hallucination"
DEFAULT_REVISION = "9d0a57af526fd90f738bb5db484d2e0dca6b4a70"
DEFAULT_STAGING = REPO_ROOT / "data" / "issue_2091" / "hf_dl" / "s4_recovery"
DEFAULT_OUT = REPO_ROOT / "eval_results" / "issue_2091" / "s4_single_recovery.json"


def stage_packed_with_census(
    staging_root: Path, revision: str, wanted_cids: set[str]
) -> tuple[dict[tuple[str, int], dict], dict]:
    """Fixed packed lookup over ALL shards + the root-cause census in one pass.

    Census counts prove the shipped-path failure mode: wrapper rows carrying a
    top-level ``context_id`` (the shipped join key) vs inner docs carrying one.
    """
    stage = A._stage2091()
    hub = A._hub()
    dest = staging_root / f"packed_{BEHAVIOR}"
    lookup: dict[tuple[str, int], dict] = {}
    census = collections.Counter()
    for rel in stage.packed_shard_paths(BEHAVIOR, revision=revision):
        target = dest / rel.rsplit("/", 1)[-1]
        if not target.is_file():
            hub.stage_hub_file(A.DATA_REPO, rel, target, repo_type="dataset", revision=revision)
        rows = list(A.iter_jsonl(target))
        census["n_lines"] += len(rows)
        census["wrapper_rows_with_top_level_context_id"] += sum(
            1 for r in rows if isinstance(r, dict) and r.get("context_id") is not None
        )
        # the SHIPPED code path keyed on the top-level field:
        census["shipped_path_wanted_hits"] += sum(
            1 for r in rows if isinstance(r, dict) and str(r.get("context_id")) in wanted_cids
        )
        inner = [
            r.get("doc") for r in rows if isinstance(r, dict) and isinstance(r.get("doc"), dict)
        ]
        census["inner_docs_with_context_id"] += sum(1 for d in inner if d.get("context_id"))
        census["inner_docs_with_rollout_k"] += sum(
            1 for d in inner if d.get("context_id") and d.get("rollout_k") is not None
        )
        lookup.update(A.packed_lookup_rows(rows, wanted_cids))
        print(
            f"[s4-recovery] shard {target.name}: lines={len(rows)} lookup={len(lookup)}", flush=True
        )
    census["fixed_path_lookup_n"] = len(lookup)
    return lookup, dict(census)


def rho_with_ci(score: np.ndarray, dv: np.ndarray, boot: A.GroupBootstrap) -> dict:
    """Spearman rho + group-bootstrap CI, mirroring behavioral_readouts.rho_with_ci."""
    r = A.spearman(score, dv)
    mask = np.isfinite(dv) & np.isfinite(score)
    rs = A.rankdata_avg(
        np.nan_to_num(score, nan=np.nanmedian(score) if np.isfinite(score).any() else 0.0)
    )
    rd = A.rankdata_avg(np.nan_to_num(dv, nan=np.nanmedian(dv) if np.isfinite(dv).any() else 0.0))
    draws = boot.corr(rs, rd, mask=mask)
    return {
        "rho": None if not np.isfinite(r) else float(r),
        "n": int(mask.sum()),
        "ci95": A.GroupBootstrap.ci(draws),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    ap.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--boot-b", type=int, default=2000)
    args = ap.parse_args(argv)
    t0 = time.time()
    staging = args.staging_root
    staging.mkdir(parents=True, exist_ok=True)
    fits = A._fits2091()

    # ── contexts + banked DV ──
    contexts_dir = A.stage_contexts_tree(staging, args.dataset_revision)
    jobs = A.FAMILY_JOBS[BEHAVIOR]
    job_rows = {job: A.load_job_contexts(contexts_dir, job) for job in jobs}
    all_family_cids = {str(r["context_id"]) for job in jobs for r in job_rows[job]}
    banked_by_ctx = {str(r["context_id"]): r for r in A.load_banked_dv(BEHAVIOR)}
    print(f"[s4-recovery] family cids={len(all_family_cids)} across {len(jobs)} rungs", flush=True)

    # ── picks: byte-identical reconstruction of the R4 run's k(c) ──
    pick_k: dict[str, int] = {}
    n_s1 = 0
    for cid in all_family_cids:
        row = banked_by_ctx.get(cid) or {}
        if row.get("per_rollout_scores"):
            pick_k[cid] = fits.s1_single_draw_pick(cid, row["per_rollout_scores"]).k
            n_s1 += 1
        else:
            pick_k[cid] = int(A.rng_for(f"s4::{cid}").integers(A.K_ROLLOUTS))
    print(
        f"[s4-recovery] picks: {len(pick_k)} (s1-branch={n_s1}, rng-branch={len(pick_k) - n_s1})",
        flush=True,
    )

    # ── fixed packed join + abstain tables + labels ──
    packed, census = stage_packed_with_census(staging, args.dataset_revision, all_family_cids)
    abstain = fits.load_banked_abstain_scores(
        A.stage_per_draw_tables(staging, args.dataset_revision)
    )
    s4_map = A.s4_labels(pick_k, packed, abstain)
    label_counts = collections.Counter(s4_map.values())
    print(f"[s4-recovery] labels: {dict(label_counts)}", flush=True)
    if label_counts.get("missing_packed_row", 0) > 0:
        raise RuntimeError(
            f"{label_counts['missing_packed_row']} picked rollouts still missing packed rows "
            "after the wrapper fix — join incomplete, refusing to persist"
        )

    # ── single-sample DV column (identical mapping to the family loop) ──
    dv_single = {
        cid: (
            None
            if lab in ("unjudged", "missing_packed_row")
            else (100.0 if lab == "fabricated" else 0.0)
        )
        for cid, lab in s4_map.items()
    }

    # ── per-rung fold + DV-vs-DV validation (no fits anywhere) ──
    greedy_dv = A.greedy_dv_by_ctx(
        A.load_greedy_dv(REPO_ROOT / "eval_results" / "issue_2091", BEHAVIOR)
    )
    rungs: dict[str, dict] = {}
    for job in jobs:
        rows = job_rows[job]
        eval_c = [str(r["context_id"]) for r in rows if r["split"] == "eval"]
        pool_c = [str(r["context_id"]) for r in rows if r["split"] == "pool"]
        gk = {str(r["context_id"]): str(r["group_key"]) for r in rows}
        lab_eval = collections.Counter(s4_map[c] for c in eval_c)
        lab_pool = collections.Counter(s4_map[c] for c in pool_c)
        sv = np.array([dv_single[c] if dv_single.get(c) is not None else np.nan for c in eval_c])
        av = np.array(
            [
                banked_by_ctx[c]["dv"] if banked_by_ctx.get(c, {}).get("dv") is not None else np.nan
                for c in eval_c
            ],
            dtype=np.float64,
        )
        gv = np.array([greedy_dv.get(c, np.nan) for c in eval_c], dtype=np.float64)
        boot = A.GroupBootstrap([gk[c] for c in eval_c], args.boot_b, f"{job}::s4recovery")
        n_lab = int(np.isfinite(sv).sum())
        rungs[job] = {
            "n_eval": len(eval_c),
            "n_pool": len(pool_c),
            "labels_eval": dict(lab_eval),
            "labels_pool": dict(lab_pool),
            "n_labeled_eval": n_lab,
            "fabricated_rate_eval": (float(np.nanmean(sv) / 100.0) if n_lab else None),
            # DV-consistency validation reads (rank agreement between DV columns;
            # NOT the grid's method-family score-vs-DV cells):
            "rho_single_vs_avg_k5": rho_with_ci(av, sv, boot),
            "rho_single_vs_greedy": rho_with_ci(gv, sv, boot),
        }
        print(
            f"[s4-recovery] {job}: eval={len(eval_c)} labeled={n_lab} labels={dict(lab_eval)}",
            flush=True,
        )

    out = {
        "meta": {
            **A._provenance_meta(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": A.SEED,
            "dataset_revision": args.dataset_revision,
            "boot_b": args.boot_b,
            "wall_s": round(time.time() - t0, 1),
        },
        "root_cause": {
            "summary": (
                "stage_packed_lookup read pack_raw_tree WRAPPER rows ({'src','doc'}) via "
                "iter_jsonl; the wrapper carries no context_id, so the (cid, rollout_k) "
                "lookup stayed empty and every S4 label fell to missing_packed_row. Fixed by "
                "packed_lookup_rows (read the inner doc; fail loud on missing rollout_k)."
            ),
            "census": census,
        },
        "picks_note": (
            "per_rollout_scores is absent for every hallucination banked row, so every pick "
            "is the seeded rng_for(f's4::{cid}') branch — byte-identical to the R4 run's picks"
        ),
        "label_counts_family": dict(label_counts),
        "rungs": rungs,
        "per_context": {cid: {"k": pick_k[cid], "label": s4_map[cid]} for cid in sorted(s4_map)},
        "grid_fold_note": (
            "behavioral_rho_L19.<family>.single in r4_grids.json is NOT recomputed here: the "
            "method-family score vectors (supervised_context ridge predictions, pv projections) "
            "were not persisted and reconstructing them requires the banked capture-store x "
            "vectors + a ridge re-application at the persisted selected_lambda — a fit this "
            "analysis-only round is barred from running."
        ),
    }
    A.write_json_atomic(args.out, out)
    print(f"[s4-recovery] done: wrote {args.out} in {out['meta']['wall_s']}s", flush=True)
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
