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
import re
import sys
import time
from pathlib import Path

# SB-1(iv): gen_meta filenames under a raw prefix — unsharded ``gen_meta.json``
# or per-shard ``gen_meta_sNNofMM.json`` (gen_capture.name_suffix contract).
_GEN_META_RE = re.compile(r"/gen_meta(_s\d{2}of\d{2})?\.json$")

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


def _gen_meta_paths(files: list[str]) -> list[str]:
    """gen_meta*.json repo paths among ``files`` (unsharded + per-shard names)."""
    return sorted(f for f in files if _GEN_META_RE.search("/" + f))


def _bind_chunk_files(files: list[str], prefix: str, chunk_keys: set[str]) -> list[str]:
    """SB-1(iv): bind the digest chunk-file set to EXACTLY the gen_meta keys.

    ``files`` is the scoped listing of ``prefix``; the authoritative chunk set
    is the union of gen_meta ``per_chunk`` keys (u2 contract: chunk file for
    key k lives at ``{prefix}/{k}.jsonl``). REFUSES (a) any extra raw-row
    JSONL under the prefix outside the key set — a stale/foreign chunk could
    supply a spurious "distinct" digest unrelated to the captured tensors —
    and (b) any expected chunk file missing from the listing. Returns the
    bound repo paths in sorted order."""
    if not chunk_keys:
        raise RuntimeError(f"#13/SB-1(iv): empty gen_meta chunk-key set under {prefix}")
    expected = {f"{prefix}/{k}.jsonl" for k in chunk_keys}
    jsonls = {f for f in files if f.endswith(".jsonl")}
    extras = sorted(jsonls - expected)
    if extras:
        raise RuntimeError(
            f"#13/SB-1(iv): {len(extras)} raw .jsonl file(s) under {prefix} not in the "
            f"gen_meta chunk-key set (head: {[Path(f).name for f in extras[:5]]}) — "
            "stale/extra chunks can fake completion distinctness; refuse "
            "(use a fresh prefix)"
        )
    missing = sorted(expected - jsonls)
    if missing:
        raise RuntimeError(
            f"#13/SB-1(iv): {len(missing)}/{len(expected)} gen_meta chunk file(s) missing "
            f"under {prefix} (head: {[Path(f).name for f in missing[:5]]})"
        )
    return sorted(expected)


def completion_digests_for(prefix: str, work: Path) -> dict[str, list[str]]:
    """{context_id: [sha16(completion_token_ids) per gen row]} for ONE replicate.

    SB-1(iv): the chunk-file set is bound to EXACTLY the union of gen_meta
    ``per_chunk`` keys under ``prefix`` via ``_bind_chunk_files`` — never a
    glob union of whatever JSONLs sit under the prefix — so distinctness is
    computed from precisely the chunks the capture consumed (gen rows carry
    the required ``completion_token_ids`` field). Content hygiene: token ids
    are HASHED, never printed/logged."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    GC = _gc()
    files = hub.retry_transient(
        lambda: hub.list_hf_files_under_path(HfApi(), FT.HF_DATA_REPO, prefix, repo_type="dataset"),
        what=f"list({prefix})",
    )
    meta_paths = _gen_meta_paths(files)
    if not meta_paths:
        raise RuntimeError(
            f"#13/SB-1(iv): no gen_meta*.json under {prefix} — gen incomplete; the "
            "chunk-key set cannot be bound (never glob-union raw JSONLs)"
        )
    chunk_keys: set[str] = set()
    for mp in meta_paths:
        local_meta = GC.fetch_repo_file(mp, work / "gen_dl", what=f"gen_meta({Path(mp).name})")
        per_chunk = json.loads(local_meta.read_text(encoding="utf-8")).get("per_chunk")
        if not isinstance(per_chunk, dict) or not per_chunk:
            raise RuntimeError(f"#13/SB-1(iv): {Path(mp).name} carries no per_chunk keys")
        chunk_keys.update(per_chunk)
    chunk_files = _bind_chunk_files(files, prefix, chunk_keys)
    digests: dict[str, list[str]] = {}
    for f in chunk_files:
        local = GC.fetch_repo_file(f, work / "gen_dl", what=f"gen({Path(f).name})")
        for r in GC.iter_jsonl(local):
            if "completion_token_ids" not in r:
                raise RuntimeError(
                    f"#13: gen row missing completion_token_ids in {Path(f).name} "
                    f"(context {r.get('context_id')!r})"
                )
            digests.setdefault(str(r["context_id"]), []).append(
                FT.sha16(",".join(str(t) for t in r["completion_token_ids"]))
            )
        local.unlink()
    return digests


def _completion_distinctness(
    per_seed_digests: dict[int, dict], ids: list[str], *, max_identical_frac: float
) -> dict:
    """#13: >=2 distinct completion digests per context across replicate draws.

    A MISSING digest for a complete-case context is ALWAYS fatal (the gen rows
    are the captured tensors' provenance). Contexts whose replicate draws all
    share ONE digest are tolerated up to ``max_identical_frac`` (legitimate
    short-answer coincidences); above it the replicate draws are not
    independent (seed collision / cached completions) and the ceiling would
    read spuriously high. Reported by id HEAD only — never completion text."""
    identical: list[str] = []
    for c in ids:
        pool: set[str] = set()
        for s in sorted(per_seed_digests):
            dm = per_seed_digests[s]
            if c not in dm or not dm[c]:
                raise RuntimeError(
                    f"#13: no gen-row completion digest for context {c!r} in rep {s}"
                )
            pool.update(dm[c])
        if len(pool) < 2:
            identical.append(c)
    frac = len(identical) / max(1, len(ids))
    out = {
        "n_contexts": len(ids),
        "n_identical_across_reps": len(identical),
        "identical_frac": frac,
        "max_identical_frac": max_identical_frac,
        "identical_context_ids_head": identical[:20],
    }
    if frac > max_identical_frac:
        raise RuntimeError(
            f"#13: {len(identical)}/{len(ids)} contexts ({frac:.3f} > {max_identical_frac}) "
            "carry IDENTICAL completions across ALL replicate draws — replicate "
            f"independence broken (seed collision / cached completions?); "
            f"id head: {identical[:20]}"
        )
    return out


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


def run_ceiling(
    args, stores=None, expected_ids=None, id_meta=None, completion_digests=None
) -> dict:
    """Per-layer reliability ceilings for one model. ``stores`` /
    ``completion_digests`` injectable (selfcheck seams): {seed: store} with the
    HfChunkStore duck-type / {seed: {context_id: [digest, ...]}}."""
    GC = _gc()
    model_key = args.model_key
    if model_key not in FT.MODEL_NAME:
        raise SystemExit(f"--model-key required (A|B), got {model_key!r}")
    seeds = parse_rep_seeds(args.rep_seeds)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    injected_stores = stores is not None
    if stores is None:
        stores = {
            s: FT.HfChunkStore(
                rep_prefixes(model_key, s)["tensors"], work / f"rep{s}", FT.MODEL_HIDDEN[model_key]
            )
            for s in seeds
        }
        for s in seeds:
            stores[s].verify_complete()  # #11: replicate stores get the same gate
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
    # #13: replicate INDEPENDENCE check on the gen-row completion digests,
    # BEFORE any tensor loop (fail cheap). Injectable; a production run may
    # skip only via the explicit flag (recorded in the artifact).
    if completion_digests is None and not injected_stores:
        if getattr(args, "skip_completion_check", False):
            distinct_info: dict = {"skipped": "--skip-completion-check (recorded escape)"}
        else:
            completion_digests = {
                s: completion_digests_for(rep_prefixes(model_key, s)["raw"], work / f"gen{s}")
                for s in seeds
            }
    if completion_digests is not None:
        distinct_info = _completion_distinctness(
            completion_digests,
            complete,
            max_identical_frac=float(getattr(args, "max_identical_frac", 0.02)),
        )
        print(
            f"[ceiling] #13 completion distinctness: "
            f"{distinct_info['n_identical_across_reps']}/{distinct_info['n_contexts']} "
            f"identical-across-reps (cap {distinct_info['max_identical_frac']})",
            flush=True,
        )
    elif injected_stores:
        distinct_info = {"skipped": "injected stores without digests (selfcheck seam)"}

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
        "completion_distinctness": distinct_info,
        "per_layer": per_layer,
    }
    out_path = out_dir / "reliability_ceiling.json"
    FT.write_artifact_json(out_path, doc)  # committed artifact: non-finite -> null
    best = max(per_layer.values(), key=lambda u: (u["ceiling_pooled"], -u["hs"]))
    print(
        f"[ceiling] model {model_key}: wrote {out_path} "
        f"({len(per_layer)} layers; max pooled ceiling {best['ceiling_pooled']:.4f} "
        f"at hs {best['hs']})",
        flush=True,
    )
    # #12: durable publish on the normal exit path (P3-rel deliverable).
    FT.publish_artifacts(
        [out_path],
        Path(args.out_root),
        publish=getattr(args, "publish", None) or "none",
        hf_prefix=getattr(args, "publish_prefix", None) or FT.PUBLISH_EVAL_MIRROR,
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


def _selfcheck_digest_binding() -> None:
    """3c (SB-1(iv)): digest chunk set binds to gen_meta keys EXACTLY —
    extras refuse, missing refuse, sharded meta names recognized."""
    pfx = "ns/raw_completions/reliability/modelA/rep43"
    base = [
        f"{pfx}/gen_meta.json",
        f"{pfx}/regime.json",
        f"{pfx}/cap_hit_report.json",
        f"{pfx}/chunk0000.jsonl",
        f"{pfx}/chunk0001.jsonl",
    ]
    assert _gen_meta_paths(base) == [f"{pfx}/gen_meta.json"], _gen_meta_paths(base)
    sharded = [f"{pfx}/gen_meta_s00of02.json", f"{pfx}/gen_meta_s01of02.json"]
    assert _gen_meta_paths(sharded + base[1:]) == sorted(sharded)
    assert _gen_meta_paths([f"{pfx}/not_gen_meta_x.json", f"{pfx}/regime.json"]) == []
    keys = {"chunk0000", "chunk0001"}
    bound = _bind_chunk_files(base, pfx, keys)
    assert bound == [f"{pfx}/chunk0000.jsonl", f"{pfx}/chunk0001.jsonl"], bound
    FT._expect_raise(
        lambda: _bind_chunk_files(base + [f"{pfx}/chunk0002.jsonl"], pfx, keys),
        "not in the gen_meta chunk-key set",
    )
    FT._expect_raise(
        lambda: _bind_chunk_files([f for f in base if "chunk0001" not in f], pfx, keys),
        "chunk file(s) missing",
    )
    FT._expect_raise(lambda: _bind_chunk_files(base, pfx, set()), "empty gen_meta chunk-key set")
    print("[selfcheck] 3c SB-1(iv) digest-set binding (bind / extras / missing): OK", flush=True)


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
        digests = {s: {c: [f"d-{c}-{s}"] for c in ids} for s in (43, 44, 45)}
        doc = run_ceiling(
            a, stores=stores, expected_ids=ids, id_meta=meta, completion_digests=digests
        )
        assert doc["completion_distinctness"]["n_identical_across_reps"] == 0
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

    # 3b. #13 completion-distinctness probes: distinct PASS, >cap identical
    # fraction rejects, and a missing digest is ALWAYS fatal.
    ids100 = [f"c{i:03d}" for i in range(100)]
    good = {s: {c: [f"d-{c}-{s}"] for c in ids100} for s in (43, 44)}
    info = _completion_distinctness(good, ids100, max_identical_frac=0.02)
    assert info["n_identical_across_reps"] == 0, info
    bad = {
        s: {c: ([f"same-{c}"] if int(c[1:]) < 10 else [f"d-{c}-{s}"]) for c in ids100}
        for s in (43, 44)
    }
    try:
        _completion_distinctness(bad, ids100, max_identical_frac=0.02)
    except RuntimeError as e:
        assert "#13" in str(e) and "IDENTICAL" in str(e), e
    else:
        raise AssertionError("#13 identical-completion gate failed to fire at 10% > 2%")
    missing = {
        43: {c: [f"d-{c}-43"] for c in ids100},
        44: {c: [f"d-{c}-44"] for c in ids100[1:]},
    }
    try:
        _completion_distinctness(missing, ids100, max_identical_frac=0.02)
    except RuntimeError as e:
        assert "no gen-row completion digest" in str(e), e
    else:
        raise AssertionError("#13 missing-digest gate failed to fire")
    print("[selfcheck] #13 completion distinctness (pass / reject / missing): OK", flush=True)

    # 3c. SB-1(iv): the digest chunk set is bound to EXACTLY gen_meta's keys.
    _selfcheck_digest_binding()

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
    ap.add_argument(
        "--publish",
        choices=("none", "hf", "git", "hf+git"),
        default=None,
        help="#12 REQUIRED for --phase ceiling: durable disposition of the ceiling "
        "artifact on the normal exit path ('none' = smoke/selfcheck local-only)",
    )
    ap.add_argument(
        "--publish-prefix",
        default=FT.PUBLISH_EVAL_MIRROR,
        help="HF data-repo prefix for the eval-results mirror (#12)",
    )
    ap.add_argument(
        "--max-identical-frac",
        type=float,
        default=0.02,
        help="#13: max tolerated fraction of contexts whose completions are "
        "identical across ALL replicate draws",
    )
    ap.add_argument(
        "--skip-completion-check",
        action="store_true",
        help="#13 escape (recorded in the artifact) — smoke/debug only",
    )
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
        if args.publish is None:
            raise SystemExit(
                "--publish {none,hf,git,hf+git} is REQUIRED for --phase ceiling (#12: "
                "every caller states a durability disposition; smokes pass --publish none)"
            )
        run_ceiling(args)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
