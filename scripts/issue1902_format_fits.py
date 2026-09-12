"""CPU-only fits for capture, generation and fixed-target context comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["MALLOC_MMAP_THRESHOLD_"] = "131072"

import issue1902_format_common as C  # noqa: E402
import issue1902_lasttoken_comparison as LC  # noqa: E402
from issue1902_lasttoken_transfer import SharedPrimalRidge  # noqa: E402
from issue2054_ctx2ctx_fit import SharedEighRidge  # noqa: E402
import numpy as np  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import identity_bias_predict, knn_retrieval  # noqa: E402


def fit_fingerprint(root):
    """Bind fit resumes to the exact aggregate inventory and scoring implementation."""
    identity = dict(
        producer=C.fingerprint(root),
        aggregates=C.sha(root / "analysis_inputs.json"),
        baselines=C.sha(C.ROOT / "src/explore_persona_space/analysis/mapping_baselines.py"),
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def stage(root, revision):
    """Consume only the declared GPU aggregate inventory at one immutable revision."""
    for name in ("manifest.json", "analysis_inputs.json", "coverage.json"):
        source = C.fetch(root, f"{C.PREFIX}/{name}", revision)
        shutil.copyfile(source, root / name)
    declared = json.loads((root / "analysis_inputs.json").read_text())
    assert C.sha(root / "manifest.json") == declared["manifest_sha256"]
    assert C.fingerprint(root) == declared["fingerprint"], "GPU/CPU code identity differs"
    for name, digest in declared["files"].items():
        source = C.fetch(root, f"{C.PREFIX}/{name}", revision)
        assert C.sha(source) == digest
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source.resolve())
        assert C.sha(target) == digest
    return declared


def key(source, target):
    """Stable input-checkpoint/render and target-checkpoint/bank/render identity."""
    return source.replace("/", "_") + "__to__" + target.replace("/", "_")


def fit_family(root, declared, family):
    """Reuse each parent's ridge, and record both parent and harmonized scoring."""
    manifest = json.loads((root / "manifest.json").read_text())
    cohort = manifest[family]
    ids = np.array(cohort["ids"])
    folds = np.array(cohort["fold_of"])
    targets = {}
    keep = np.ones(len(ids), dtype=bool)
    for target, name in declared["targets"].items():
        if not target.startswith(family + "_"):
            continue
        with np.load(root / name, allow_pickle=False) as p:
            assert np.array_equal(p["ids"], ids)
            targets[target] = p["y"]
            keep &= p["valid"]
    assert targets
    C.write_json(
        root / "fits" / family / "cohort.json",
        dict(
            planned_rows=len(ids),
            realized_rows=int(keep.sum()),
            excluded_undefined_answer_vectors=ids[~keep].tolist(),
            definition="Common complete-five vector cohort across all declared target banks; no quality/cap filtering",
        ),
    )
    ids, folds = ids[keep], folds[keep]
    targets = {k: v[keep].astype(np.float64) for k, v in targets.items()}
    sources = {s: p for s, p in declared["contexts"].items() if s.startswith(family + "_")}
    fp = fit_fingerprint(root)
    for source, name in sources.items():
        with np.load(root / name, allow_pickle=False) as p:
            assert p["ids"].tolist() == cohort["ids"]
            x = p["x"][keep].astype(np.float64)
            assert np.array_equal(p["fold_of"][keep], folds)
        for fold in range(cohort["n_folds"]):
            pending = [
                t
                for t in targets
                if not all(
                    C.complete(
                        root / "fits" / family / "folds" / f"{key(source, t)}_f{fold}.{extension}",
                        fp,
                    )
                    for extension in ("json", "npz")
                )
            ]
            if not pending:
                continue
            tr, ev = folds != fold, folds == fold
            assert tr.sum() > x.shape[1] and ev.sum() > 10, (
                "Insufficient common-cohort ambient fit support"
            )
            started = time.monotonic()
            ridge = (
                SharedEighRidge(x[tr], x[ev], device="cpu")
                if family == "qwen"
                else SharedPrimalRidge(x[tr])
            )
            files = []
            for target in pending:
                y = targets[target]
                pred, info = (
                    ridge.fit_predict(y[tr])
                    if family == "qwen"
                    else ridge.fit_predict(y[tr], x[ev])
                )
                assert info["dof"] <= 0.9 * tr.sum()
                baseline = identity_bias_predict(x[tr], y[tr], x[ev])
                residual, total, cosine = LC._per_row_components(pred, y[ev], y[tr].mean(0))
                baseline_res, _, _ = LC._per_row_components(baseline, y[ev], y[tr].mean(0))
                eval_total = ((y[ev] - y[ev].mean(0)) ** 2).sum(1)
                assert total.sum() > 0 and eval_total.sum() > 0
                record = dict(
                    source=source,
                    target=target,
                    fold=fold,
                    fit=info,
                    r2_train_centered=float(1 - residual.sum() / total.sum()),
                    r2_eval_centered=float(1 - residual.sum() / eval_total.sum()),
                    identity_bias_train_centered=float(1 - baseline_res.sum() / total.sum()),
                    identity_bias_eval_centered=float(1 - baseline_res.sum() / eval_total.sum()),
                    retrieval_cosine=knn_retrieval(pred, y[ev], metric="cosine"),
                    retrieval_euclidean=knn_retrieval(pred, y[ev], metric="euclidean"),
                    retrieval_pool=int(ev.sum()),
                    chance_top1=float(1 / ev.sum()),
                    target_sha256=declared["files"][declared["targets"][target]],
                )
                stem = root / "fits" / family / "folds" / f"{key(source, target)}_f{fold}"
                LC._savez(
                    stem.with_suffix(".npz"),
                    indices=np.flatnonzero(ev),
                    res=residual,
                    tot=total,
                    eval_tot=eval_total,
                    baseline_res=baseline_res,
                    cosine=cosine,
                )
                C.write_json(stem.with_suffix(".json"), record)
                files.extend([stem.with_suffix(".json"), stem.with_suffix(".npz")])
                print(
                    f"[phase=fit] {source} -> {target} fold={fold} R2={record['r2_train_centered']:.6f}",
                    flush=True,
                )
            C.upload_many(files, root, fp)
            print(
                f"[phase=fit_group] {family} source={source} fold={fold} seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
            del ridge
    summarize(root, family, ids, folds, sources, targets)


def summarize(root, family, ids, folds, sources, targets):
    """Pair all contrasts with one shared bootstrap; target identity is checked explicitly."""
    records, errors = {}, {}
    n = len(ids)
    for source in sources:
        for target in targets:
            name = key(source, target)
            res, tot, base = (np.full(n, np.nan) for _ in range(3))
            fold_records = []
            for fold in sorted(set(folds.tolist())):
                stem = root / "fits" / family / "folds" / f"{name}_f{fold}"
                assert all(
                    C.complete(stem.with_suffix("." + ext), fit_fingerprint(root))
                    for ext in ("npz", "json")
                )
                with np.load(stem.with_suffix(".npz"), allow_pickle=False) as p:
                    ix = p["indices"]
                    assert np.isnan(res[ix]).all()
                    res[ix], tot[ix], base[ix] = p["res"], p["tot"], p["baseline_res"]
                fold_records.append(json.loads(stem.with_suffix(".json").read_text()))
            assert np.isfinite(res).all() and np.isfinite(tot).all()
            harmonized = float(1 - res.sum() / tot.sum())
            parent = (
                float(np.mean([r["r2_eval_centered"] for r in fold_records]))
                if family == "qwen"
                else harmonized
            )
            records[name] = dict(
                source=source,
                target=target,
                harmonized_r2=harmonized,
                parent_r2=parent,
                identity_bias_harmonized=float(1 - base.sum() / tot.sum()),
                identity_bias_parent=float(
                    np.mean([r["identity_bias_eval_centered"] for r in fold_records])
                )
                if family == "qwen"
                else float(1 - base.sum() / tot.sum()),
                n=n,
                fold_metrics=fold_records,
            )
            errors[name] = (res, tot)
    names = list(errors)
    weights = (
        np.random.default_rng(1944).multinomial(n, np.full(n, 1 / n), size=1000).astype(np.float64)
    )
    boot = 1 - (weights @ np.column_stack([errors[k][0] for k in names])) / (
        weights @ np.column_stack([errors[k][1] for k in names])
    )
    contrasts = []

    def contrast(kind, a, b, details):
        assert a in records and b in records, (kind, a, b)
        if kind == "context_information":
            assert np.array_equal(errors[a][1], errors[b][1]), "Fixed targets differ"
        difference = boot[:, names.index(a)] - boot[:, names.index(b)]
        contrasts.append(
            dict(
                kind=kind,
                first=a,
                second=b,
                first_minus_second_r2=records[a]["harmonized_r2"] - records[b]["harmonized_r2"],
                paired_row_ci95=np.quantile(difference, [0.025, 0.975]).tolist(),
                **details,
            )
        )

    for target in targets:
        model, bank, form = target.split("/")
        own = key(model + "/" + form, target)
        base = key(family + "_B/" + form, target)
        contrast("context_information", own, base, dict(fixed_target=target))
        if form == "chat":
            plain_target = model + "/" + bank + "/plain"
            contrast(
                "capture_format",
                own,
                key(model + "/plain", plain_target),
                dict(
                    held_fixed_answer_bank=model + "/" + bank, comparison="chat minus plain capture"
                ),
            )
    for model in (family + "_B", family + "_S"):
        chat_bank = "chat4096" if model == "qwen_S" else "chat"
        for form in ("plain", "chat"):
            contrast(
                "generation_format",
                key(model + "/" + form, model + "/" + chat_bank + "/" + form),
                key(model + "/" + form, model + "/plain/" + form),
                dict(
                    fixed_capture_form=form,
                    comparison="chat minus plain generation; equal caps, native turn stops",
                ),
            )
    for j, name in enumerate(names):
        records[name]["harmonized_row_ci95"] = np.quantile(boot[:, j], [0.025, 0.975]).tolist()
    path = root / "fits" / family / "summary.json"
    C.write_json(
        path,
        dict(
            family=family,
            cells=records,
            contrasts=contrasts,
            harmonized_metric="pooled held-out SSE/SST, centered on training-fold target mean",
            parent_metric="unweighted mean of eval-centered fold R2"
            if family == "qwen"
            else "harmonized metric",
            uncertainty="1000 paired row bootstrap draws, seed1944; no refit; CIs for harmonized score",
            interpretation="Context comparison measures linear accessibility of the same fixed target vectors",
        ),
    )
    C.upload_many([path, root / "fits" / family / "cohort.json"], root, fit_fingerprint(root))


def main():
    """Stage, fit, and publish both families on a dedicated CPU machine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/issue1902_format_cpu"))
    parser.add_argument("--input-revision", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    from issue1902_format_job import verify_source

    verify_source(args.source_sha)
    from explore_persona_space.orchestrate.preflight import preflight_check

    report = preflight_check(
        require_gpu=False, min_disk_gb=30, planned_footprint_gb=15, required_env_vars=["HF_TOKEN"]
    )
    assert report.ok, report
    declared = stage(args.root, args.input_revision)
    for family in ("qwen", "olmo"):
        fit_family(args.root, declared, family)
    result = dict(
        status="complete",
        followup_label="format-comparison-CPU-fits",
        input_revision=args.input_revision,
        hf_prefix=C.PREFIX,
        source_sha=args.source_sha,
    )
    C.write_json(args.root / "fits_complete.json", result)
    revision = C.upload_many(
        [args.root / "fits_complete.json"], args.root, fit_fingerprint(args.root)
    )
    result["results_revision"] = revision
    C.write_json(
        Path("/workspace/logs") / f"issue-1902-epm_results-{time.time_ns()}.json",
        dict(
            sentinel_schema_version=1,
            kind="epm:results",
            version=1,
            task_id=1902,
            note=json.dumps(result),
            blocks_pipeline=False,
        ),
    )
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    write_completion_sentinel(
        sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=1902, extra=result
    )
    print(
        "[phase=done] Both families: capture, generation, and fixed-target comparisons complete",
        flush=True,
    )


if __name__ == "__main__":
    main()
