"""Issue #2502 P3-rel (MF-E): answer-vector reliability ceiling per model/layer.

Draws a COMMON stratified test-partition subset (~1,500 contexts spanning every
regime class), prints the per-model replicate gen+capture commands (R=3
independent replicate draws, distinct sampling seeds, via
``issue2502_gen_capture.py --seed {s}`` — the strictly additive flag this unit
added), and computes per-model/per-layer reliability ceilings from the
persisted replicate v_x stores:

  ceiling(k) = 1 - w_bar/t   (the ICC(1) form)

with, per hidden dim (fp64, ddof=1): ``w_bar`` = mean over complete-case
contexts of the across-replicate variance (sigma_w^2 estimate), ``t`` = the
pooled single-draw variance over all (context, replicate) draws
(sigma_w^2 + sigma_b^2 estimate), both summed over dims before the ratio.
This is the max held-out R^2 ANY deterministic function of the context can
attain against a single sampled answer vector — the quantity H3's
cross-model comparison is stated relative to (plan v6 S4 P3-rel / S3 MF-E).
A within-source-centered variant (t computed within source_tag) and a
replicate-pair split-half R^2 diagnostic ride alongside. Ceilings are pure
re-reductions of the persisted per-replicate tensor stores on the HF data
repo — recomputable without regeneration.

Phases:
  subset          draw + upload the common reliability subset corpus
                  (verbatim rows; content hygiene — no text printed/logged).
  print-commands  emit the exact per-(model x replicate-seed) gen_capture
                  invocations (validated: >=2 distinct seeds, none == 42).
  ceiling         one model (--model-key): per-layer ceilings from the R
                  replicate stores -> eval_results/issue_2502/reliability/
                  model{K}/reliability_ceiling.json.
  selfcheck       synthetic known-sigma ICC recovery + coverage-gate +
                  stratified-draw + seed-validation checks (VM-safe, tiny).

Compute placement: ceiling is a light CPU reduce (~1.5k x 4096 x 3 fp64 per
layer); VM launches carry the shared-VM thread-cap prefix inline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2502_fits as FT  # stdlib-only module top (constants + store readers)

ISSUE = 2502
PRIMARY_SEED = 42  # the production draw's seed — replicates must differ
DEFAULT_REP_SEEDS = "43,44,45"
SUBSET_SEED = 2502
DEFAULT_SUBSET_SIZE = 1500
MIN_PER_CLASS = 60
COVERAGE_FLOOR = 0.90
SUBSET_PREFIX = "issue2502_ctxmap_xgen/reliability_subset"
RELIABILITY_ROOT = "issue2502_ctxmap_xgen"


def _gc():
    import issue2502_gen_capture as GC

    return GC


def parse_rep_seeds(spec: str) -> list[int]:
    """Validated replicate seed list: >=2, all distinct, none == PRIMARY_SEED."""
    try:
        seeds = [int(s) for s in spec.split(",") if s.strip() != ""]
    except ValueError as e:
        raise SystemExit(f"--rep-seeds must be comma-separated ints: {spec!r}") from e
    if len(seeds) < 2:
        raise SystemExit(f"MF-E needs >=2 independent replicates, got {seeds}")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"replicate seeds must be distinct, got {seeds}")
    if PRIMARY_SEED in seeds:
        raise SystemExit(
            f"replicate seeds must differ from the primary draw seed {PRIMARY_SEED} "
            f"(identical seed => identical completions), got {seeds}"
        )
    return seeds


def rep_prefixes(model_key: str, seed: int) -> dict:
    """Canonical HF prefixes for one replicate run (raw text + capture tensors)."""
    base = f"{RELIABILITY_ROOT}"
    return {
        "raw": f"{base}/raw_completions/reliability/model{model_key}/rep{seed}",
        "tensors": f"{base}/analysis_tensors/reliability/model{model_key}/rep{seed}",
    }


# --------------------------------------------------------------------------
# subset phase
# --------------------------------------------------------------------------


def stratified_subset(rows: list[dict], *, size: int, min_per_class: int, seed: int) -> list[dict]:
    """Deterministic stratified draw over TEST-partition rows.

    Proportional to regime-class counts with a per-class floor; within a class
    the draw is a seeded permutation of context_id-sorted rows. Every regime
    class present in the test partition is represented (MF-E: the common
    subset spans every regime class)."""
    import numpy as np

    test_rows = [r for r in rows if r.get("split") == "test"]
    if not test_rows:
        raise RuntimeError("no test-partition rows in corpus — cannot draw reliability subset")
    by_class: dict[str, list[dict]] = {}
    for r in test_rows:
        by_class.setdefault(str(r.get("regime_class")), []).append(r)
    classes = sorted(by_class)
    n_total = len(test_rows)
    if size > n_total:
        raise RuntimeError(f"subset size {size} > test partition {n_total}")
    quota: dict[str, int] = {}
    for c in classes:
        share = int(round(size * len(by_class[c]) / n_total))
        quota[c] = min(len(by_class[c]), max(min_per_class, share))
    # trim proportionally if floors pushed the total over budget
    while sum(quota.values()) > size:
        biggest = max(classes, key=lambda c: (quota[c], c))
        if quota[biggest] <= min_per_class:
            break  # every class at floor — accept slightly-over-budget subset
        quota[biggest] -= 1
    rng = np.random.default_rng(seed)
    picked: list[dict] = []
    for c in classes:
        pool = sorted(by_class[c], key=lambda r: str(r["context_id"]))
        idx = rng.permutation(len(pool))[: quota[c]]
        picked.extend(pool[int(i)] for i in sorted(idx))
    ids = [str(r["context_id"]) for r in picked]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate context_ids in stratified subset draw")
    return picked


def run_subset(args) -> dict:
    """Draw the common subset and upload it as a gen_capture-consumable corpus."""
    GC = _gc()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    dest = f"{SUBSET_PREFIX}/corpus.jsonl"
    if not args.force:
        # scope is a repo PATH PREFIX (path_in_repo for the scoped listing),
        # not a label — a bare "reliability-subset" string made
        # verify_repo_paths_uploaded raise "expected paths outside
        # path_in_repo" on EVERY subset invocation (u4 smoke catch).
        existing = GC.hf_missing_of([dest], scope=SUBSET_PREFIX)
        if not existing:  # empty missing-set => already uploaded
            print(f"[subset] {dest} already on HF — skip (use --force to redraw)", flush=True)
            return {"dest": dest, "skipped": True}
    local_corpus = GC.fetch_repo_file(
        f"{args.corpus_prefix}/corpus.jsonl", work / "corpus_dl", what="corpus"
    )
    rows = list(GC.iter_jsonl(local_corpus))
    picked = stratified_subset(
        rows, size=args.subset_size, min_per_class=args.min_per_class, seed=SUBSET_SEED
    )
    out = work / "reliability_subset_corpus.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in picked:  # verbatim rows — the gen_capture consumer contract
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_class: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for r in picked:
        by_class[str(r.get("regime_class"))] = by_class.get(str(r.get("regime_class")), 0) + 1
        by_source[str(r.get("source_tag"))] = by_source.get(str(r.get("source_tag")), 0) + 1
    manifest = {
        "meta": GC.run_metadata({"artifact": "reliability_subset_manifest"}),
        "n_rows": len(picked),
        "source_corpus_prefix": args.corpus_prefix,
        "split": "test",
        "subset_seed": SUBSET_SEED,
        "min_per_class": args.min_per_class,
        "requested_size": args.subset_size,
        "by_regime_class": by_class,
        "by_source_tag": by_source,
        "context_ids_sha16": FT.sha16(",".join(sorted(str(r["context_id"]) for r in picked))),
    }
    man_path = work / "reliability_subset_manifest.json"
    GC.atomic_write_json(man_path, manifest)
    GC.upload_single_file(out, dest)
    GC.upload_single_file(man_path, f"{SUBSET_PREFIX}/manifest.json")
    print(
        f"[subset] uploaded {len(picked)} rows -> {dest} (classes: {by_class})",
        flush=True,
    )
    return {"dest": dest, "n_rows": len(picked), "by_regime_class": by_class}


# --------------------------------------------------------------------------
# print-commands phase
# --------------------------------------------------------------------------

_THREAD_CAPS = (
    "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "
    "NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 "
)


def replicate_commands(args) -> list[str]:
    """The exact per-(model x rep-seed) gen_capture invocations (pod-side)."""
    seeds = parse_rep_seeds(args.rep_seeds)
    model_flags = {
        "A": "--model Qwen/Qwen2.5-7B-Instruct --env repo-standard",
        "B": ("--model Qwen/Qwen3.5-9B --env pod2378-venv --disable-thinking --gdn-prefill triton"),
    }
    cmds = []
    for key in ("A", "B"):
        for s in seeds:
            p = rep_prefixes(key, s)
            cmds.append(
                "uv run python scripts/issue2502_gen_capture.py "
                f"{model_flags[key]} --seed {s} "
                f"--corpus-prefix {SUBSET_PREFIX} "
                f"--raw-prefix {p['raw']} "
                f"--out-prefix {p['tensors']} "
                f"--work-dir /workspace/issue2502_rel_model{key}_rep{s}"
            )
    return cmds


def run_print_commands(args) -> None:
    print(
        f"# MF-E replicate generation: {len(parse_rep_seeds(args.rep_seeds))} reps x 2 models "
        f"on the common subset ({SUBSET_PREFIX}); pod-side, NO thread-cap prefix.",
        flush=True,
    )
    for c in replicate_commands(args):
        print(c, flush=True)


# --------------------------------------------------------------------------
# ceiling phase
# --------------------------------------------------------------------------


def _rep_matrix(stores: dict[int, object], k: int, ids: list[str]):
    """(n, R, H) fp64 stack of replicate v_x for hs layer k over shared ids."""
    import numpy as np

    mats = []
    for seed in sorted(stores):
        rows = stores[seed].load_rows()
        pos = {str(r["context_id"]): i for i, r in enumerate(rows)}
        _, vx = stores[seed].load_layer(k)
        sel = np.asarray([pos[c] for c in ids])
        mats.append(np.asarray(vx, dtype=np.float64)[sel])
    return np.stack(mats, axis=1)  # (n, R, H)


def ceiling_from_stack(y, *, sources=None) -> dict:
    """ICC-form ceiling from a (n, R, H) replicate stack (fp64, ddof=1).

    ceiling_pooled = 1 - sum_d w_bar_d / sum_d t_d with w_bar_d the mean
    within-context across-replicate variance and t_d the pooled single-draw
    variance over all (context, replicate) draws. Within-source variant
    replaces t with per-source pooled variance (sources with >=2 contexts).
    Split-half diagnostic: mean pooled R^2 of one replicate predicting
    another over all ordered pairs."""
    import numpy as np

    n, r, h = y.shape
    if n < 2 or r < 2:
        raise RuntimeError(f"degenerate replicate stack shape {(n, r, h)}")
    w_d = y.var(axis=1, ddof=1).mean(axis=0)  # (H,)
    flat = y.reshape(n * r, h)
    t_d = flat.var(axis=0, ddof=1)  # (H,)
    w_sum, t_sum = float(w_d.sum()), float(t_d.sum())
    out = {
        "n_complete": int(n),
        "n_replicates": int(r),
        "w_bar_sum": w_sum,
        "t_sum": t_sum,
        "ceiling_pooled": float("nan") if t_sum < 1e-12 else 1.0 - w_sum / t_sum,
    }
    if sources is not None:
        src = np.asarray(sources)
        ss_within_ctx = 0.0
        ss_within_src = 0.0
        n_used = 0
        for s in sorted(set(src)):
            m = src == s
            if int(m.sum()) < 2:
                continue
            ys = y[m]  # (ns, R, H)
            ss_within_ctx += float(((ys - ys.mean(axis=1, keepdims=True)) ** 2).sum())
            mu_s = ys.reshape(-1, h).mean(axis=0)
            ss_within_src += float(((ys.reshape(-1, h) - mu_s) ** 2).sum())
            n_used += int(m.sum())
        out["n_ctx_within_source"] = n_used
        out["ceiling_within_source"] = (
            float("nan") if ss_within_src < 1e-12 else 1.0 - ss_within_ctx / ss_within_src
        )
    pair_r2 = []
    for a in range(r):
        for b in range(r):
            if a != b:
                pair_r2.append(FT.pooled_r2(y[:, a, :], y[:, b, :]))
    finite = [v for v in pair_r2 if v == v]
    out["splithalf_r2_mean"] = float(sum(finite) / len(finite)) if finite else float("nan")
    return out


def run_ceiling(args, stores=None, expected_ids=None, id_meta=None) -> dict:
    """Per-layer reliability ceilings for one model. ``stores`` injectable
    (selfcheck seam): {seed: store} with the HfChunkStore duck-type."""
    GC = _gc()
    model_key = args.model_key
    if model_key not in FT.MODEL_NAME:
        raise SystemExit(f"--model-key required (A|B), got {model_key!r}")
    seeds = parse_rep_seeds(args.rep_seeds)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if stores is None:
        stores = {
            s: FT.HfChunkStore(
                rep_prefixes(model_key, s)["tensors"], work / f"rep{s}", FT.MODEL_HIDDEN[model_key]
            )
            for s in seeds
        }
    if expected_ids is None:
        local = GC.fetch_repo_file(
            f"{SUBSET_PREFIX}/corpus.jsonl", work / "subset_dl", what="subset"
        )
        subset_rows = list(GC.iter_jsonl(local))
        expected_ids = [str(r["context_id"]) for r in subset_rows]
        id_meta = {str(r["context_id"]): r for r in subset_rows}
    id_meta = id_meta or {}

    per_store_ids = {s: {str(r["context_id"]) for r in stores[s].load_rows()} for s in stores}
    complete = sorted(set(expected_ids).intersection(*per_store_ids.values()))
    coverage = len(complete) / max(1, len(expected_ids))
    per_rep_cov = {s: len(per_store_ids[s] & set(expected_ids)) for s in per_store_ids}
    print(
        f"[ceiling] model {model_key}: {len(complete)}/{len(expected_ids)} complete-case "
        f"contexts (coverage {coverage:.3f}; per-rep {per_rep_cov})",
        flush=True,
    )
    if coverage < args.coverage_floor:
        raise RuntimeError(
            f"MF-E coverage {coverage:.3f} < floor {args.coverage_floor} — replicate stores "
            f"incomplete (per-rep coverage {per_rep_cov}); regenerate the missing replicates"
        )
    captured_sets = [set(stores[s].captured_hs()) for s in stores]
    layers = sorted(set.intersection(*captured_sets))
    if not layers:
        raise RuntimeError("no common captured layer across replicate stores")
    sources = [str(id_meta.get(c, {}).get("source_tag")) for c in complete]

    percell = work / f"rel_percell_model{model_key}"
    percell.mkdir(parents=True, exist_ok=True)
    ledger = GC.StageLedger(
        percell / "ledger.json",
        {
            "phase": "reliability-ceiling",
            "issue": ISSUE,
            "model_key": model_key,
            "rep_seeds": seeds,
            "subset_prefix": SUBSET_PREFIX,
            "n_expected": len(expected_ids),
            "ids_sha16": FT.sha16(",".join(sorted(complete))),
        },
    )
    t0 = time.time()
    per_layer: dict[str, dict] = {}
    for j, k in enumerate(layers):
        cell = f"L{k:02d}"
        cell_path = percell / f"{cell}.json"
        if ledger.is_done(cell) and cell_path.exists():
            per_layer[cell] = json.loads(cell_path.read_text())
            continue
        y = _rep_matrix(stores, k, complete)
        unit = ceiling_from_stack(y, sources=sources)
        unit["hs"] = k
        GC.atomic_write_json(cell_path, unit)
        ledger.mark_done(cell)
        per_layer[cell] = unit
        GC.progress(f"ceiling-{model_key}", j + 1, len(layers), cell, t0)

    out_dir = Path(args.out_root) / "reliability" / f"model{model_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "meta": GC.run_metadata(
            {
                "artifact": "reliability_ceiling",
                "model_key": model_key,
                "model": FT.MODEL_NAME[model_key],
                "definition": (
                    "ceiling = 1 - w_bar/t (ICC(1)); w_bar = mean within-context "
                    "across-replicate variance, t = pooled single-draw variance "
                    "(fp64, ddof=1, summed over dims before the ratio); "
                    "within_source variant centers t per source_tag (>=2-context "
                    "sources); splithalf = mean pooled R^2 over ordered replicate "
                    "pairs. Recomputable from the persisted replicate stores."
                ),
                "rep_seeds": seeds,
                "replicate_tensor_prefixes": {
                    s: rep_prefixes(model_key, s)["tensors"] for s in seeds
                },
                "subset_prefix": SUBSET_PREFIX,
            }
        ),
        "n_expected": len(expected_ids),
        "n_complete": len(complete),
        "coverage": coverage,
        "coverage_floor": args.coverage_floor,
        "per_rep_coverage": {str(s): int(v) for s, v in per_rep_cov.items()},
        "per_layer": per_layer,
    }
    GC.atomic_write_json(out_dir / "reliability_ceiling.json", doc)
    best = max(per_layer.values(), key=lambda u: (u["ceiling_pooled"], -u["hs"]))
    print(
        f"[ceiling] model {model_key}: wrote {out_dir / 'reliability_ceiling.json'} "
        f"({len(per_layer)} layers; max pooled ceiling {best['ceiling_pooled']:.4f} "
        f"at hs {best['hs']})",
        flush=True,
    )
    return doc


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------


def _toy_rep_stores(*, n=400, d=8, r=3, sigma_b=1.0, sigma_w=0.5, seed=0, drop_from_rep=None):
    """Synthetic replicate stores with KNOWN sigma_b/sigma_w (ICC recovery)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    sources = [f"s{i % 5}" for i in range(n)]
    rows = [
        {
            "row": i,
            "context_id": f"ctx{i:04d}",
            "source_tag": sources[i],
            "regime_class": "ordinary" if i % 2 == 0 else "weird",
            "split": "test",
        }
        for i in range(n)
    ]
    b = sigma_b * rng.standard_normal((n, d))
    stores = {}
    for j, s in enumerate((43, 44, 45)[:r]):
        y = (b + sigma_w * rng.standard_normal((n, d))).astype(np.float32)
        keep = list(range(n))
        if drop_from_rep is not None and s == drop_from_rep[0]:
            keep = keep[: n - drop_from_rep[1]]
        st_rows = [dict(rows[i], row=jj) for jj, i in enumerate(keep)]
        x = np.zeros((len(keep), d), dtype=np.float32)
        stores[s] = FT.MemStore(st_rows, {1: (x, y[keep])})
        del j
    ids = [r_["context_id"] for r_ in rows]
    meta = {r_["context_id"]: r_ for r_ in rows}
    return stores, ids, meta


def run_selfcheck(args) -> int:
    import copy
    import tempfile

    import numpy as np

    # 1. seed validation
    assert parse_rep_seeds("43,44,45") == [43, 44, 45]
    for bad in ("43", "43,43", "42,43", "a,b"):
        try:
            parse_rep_seeds(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"parse_rep_seeds accepted invalid spec {bad!r}")
    print("[selfcheck] replicate-seed validation: OK", flush=True)

    # 2. ICC recovery on known variances: expected ceiling = sb^2/(sb^2+sw^2)
    sb, sw = 1.0, 0.5
    expected = sb**2 / (sb**2 + sw**2)
    stores, ids, meta = _toy_rep_stores(sigma_b=sb, sigma_w=sw, seed=7)
    with tempfile.TemporaryDirectory(prefix="i2502_rel_selfcheck_") as td:
        a = copy.copy(args)
        a.work_dir = str(Path(td) / "work")
        a.out_root = str(Path(td) / "out")
        a.model_key = "A"
        a.rep_seeds = "43,44,45"
        a.coverage_floor = COVERAGE_FLOOR
        doc = run_ceiling(a, stores=stores, expected_ids=ids, id_meta=meta)
        got = doc["per_layer"]["L01"]["ceiling_pooled"]
        assert abs(got - expected) < 0.05, f"ICC recovery {got:.4f} vs expected {expected:.4f}"
        ws = doc["per_layer"]["L01"]["ceiling_within_source"]
        assert abs(ws - expected) < 0.08, f"within-source ICC {ws:.4f} vs {expected:.4f}"
        sh = doc["per_layer"]["L01"]["splithalf_r2_mean"]
        assert 0.0 < sh < 1.0
        print(
            f"[selfcheck] ICC recovery: pooled {got:.4f} / ws {ws:.4f} vs expected "
            f"{expected:.4f}; splithalf {sh:.4f}: OK",
            flush=True,
        )

        # resume: second run reuses the ledger + percell cells (no recompute crash)
        doc2 = run_ceiling(a, stores=stores, expected_ids=ids, id_meta=meta)
        assert doc2["per_layer"]["L01"]["ceiling_pooled"] == got
        print("[selfcheck] ledger resume reuse: OK", flush=True)

    # 3. coverage gate fires when one replicate is missing >10% of contexts
    stores_bad, ids2, meta2 = _toy_rep_stores(
        sigma_b=sb, sigma_w=sw, seed=8, drop_from_rep=(44, 60)
    )
    with tempfile.TemporaryDirectory(prefix="i2502_rel_selfcheck2_") as td:
        a = copy.copy(args)
        a.work_dir = str(Path(td) / "work")
        a.out_root = str(Path(td) / "out")
        a.model_key = "A"
        a.rep_seeds = "43,44,45"
        a.coverage_floor = COVERAGE_FLOOR
        try:
            run_ceiling(a, stores=stores_bad, expected_ids=ids2, id_meta=meta2)
        except RuntimeError as e:
            assert "coverage" in str(e), e
        else:
            raise AssertionError("coverage gate failed to fire on incomplete replicate store")
    print("[selfcheck] coverage gate: OK", flush=True)

    # 4. stratified subset: floors + totals + determinism + test-only
    rng = np.random.default_rng(3)
    classes = (
        ["ordinary"] * 900 + ["weird"] * 300 + ["near-distribution"] * 120 + ["idiosyncratic"] * 80
    )
    rows = [
        {
            "context_id": f"c{i:05d}",
            "regime_class": classes[i % len(classes)],
            "source_tag": f"src{i % 11}",
            "split": "test" if rng.random() < 0.5 else "train",
        }
        for i in range(4000)
    ]
    picked = stratified_subset(rows, size=300, min_per_class=30, seed=SUBSET_SEED)
    assert all(r["split"] == "test" for r in picked)
    counts: dict[str, int] = {}
    for r in picked:
        counts[r["regime_class"]] = counts.get(r["regime_class"], 0) + 1
    assert set(counts) == set(classes), counts
    assert all(v >= 30 for v in counts.values()), counts
    assert abs(len(picked) - 300) <= len(counts), len(picked)
    picked2 = stratified_subset(rows, size=300, min_per_class=30, seed=SUBSET_SEED)
    assert [r["context_id"] for r in picked] == [r["context_id"] for r in picked2]
    print(f"[selfcheck] stratified subset (counts {counts}): OK", flush=True)

    # 5. replicate command composition (flags match gen_capture's CLI surface)
    a = copy.copy(args)
    a.rep_seeds = "43,44,45"
    cmds = replicate_commands(a)
    assert len(cmds) == 6
    assert all("--seed" in c and "--corpus-prefix " + SUBSET_PREFIX in c for c in cmds)
    assert any("--disable-thinking" in c and "--gdn-prefill triton" in c for c in cmds)
    print("[selfcheck] replicate command composition: OK", flush=True)
    print("[selfcheck] ALL OK", flush=True)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase", choices=("subset", "print-commands", "ceiling", "selfcheck"), default="ceiling"
    )
    ap.add_argument("--model-key", choices=("A", "B"), default=None, help="ceiling: which model")
    ap.add_argument("--rep-seeds", default=DEFAULT_REP_SEEDS)
    ap.add_argument("--corpus-prefix", default="issue2502_ctxmap_xgen/context_corpus")
    ap.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    ap.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS)
    ap.add_argument("--coverage-floor", type=float, default=COVERAGE_FLOOR)
    ap.add_argument("--force", action="store_true", help="subset: redraw + re-upload")
    ap.add_argument("--work-dir", default="/workspace/issue2502_reliability")
    ap.add_argument("--out-root", default=str(_REPO_ROOT / "eval_results" / "issue_2502"))
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("issue2502_reliability: import-check OK", flush=True)
        return 0
    # load_dotenv BEFORE any numpy import (thread caps freeze at import, #847).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    if args.phase == "selfcheck":
        return run_selfcheck(args)
    if args.phase == "subset":
        run_subset(args)
    elif args.phase == "print-commands":
        run_print_commands(args)
    else:
        run_ceiling(args)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
