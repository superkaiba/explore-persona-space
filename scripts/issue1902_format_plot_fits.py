"""Only the eight OLMo own maps and target-on-policy panels B/C, on dedicated CPU."""

from __future__ import annotations

import argparse
from itertools import pairwise
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["MALLOC_ARENA_MAX"] = "2"

import issue1902_format_common as C  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()
import issue1902_format_fits as F  # noqa: E402
import issue1902_lasttoken_comparison as LC  # noqa: E402
from issue1902_lasttoken_transfer import SharedPrimalRidge  # noqa: E402
import numpy as np  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import identity_bias_predict, knn_retrieval  # noqa: E402

STAGES = "BSDR"
FORMS = ("plain", "chat")
MODES = ("direct", "bias", "scale_bias")


def target_key(stage, form):
    """Generation model, generation format and capture format are identical."""
    return f"olmo_{stage}/{form}/{form}"


def fit_pairs():
    """Eight own maps and six base-to-target maps; no unused matrix cells."""
    return [(s, s, f) for f in FORMS for s in STAGES] + [("B", t, f) for f in FORMS for t in "SDR"]


def transfer_pairs():
    """Adjacent stages, with source and target prompts in the same format."""
    return [(s, t, f) for f in FORMS for s, t in pairwise(STAGES)]


def stem(root, source, target, form, fold, transfer=False):
    return (
        root
        / "fits"
        / "olmo"
        / "folds"
        / f"{'transfer' if transfer else 'fit'}_{source}{target}_{form}_f{fold}"
    )


def complete(path, fp):
    return all(C.complete(path.with_suffix("." + ext), fp) for ext in ("npz", "json"))


def evaluate(pred, y_tr, y_ev):
    residual, total, cosine = LC._per_row_components(pred, y_ev, y_tr.mean(0))
    assert total.sum() > 0 and np.isfinite(residual).all()
    return dict(res=residual, tot=total, cosine=cosine), dict(
        r2=float(1 - residual.sum() / total.sum()),
        retrieval_cosine=knn_retrieval(pred, y_ev, metric="cosine"),
        retrieval_euclidean=knn_retrieval(pred, y_ev, metric="euclidean"),
        retrieval_pool=len(y_ev),
        chance_top1=1 / len(y_ev),
    )


def transferred_predictions(ridge, own_fit, x_target_tr, x_target_ev, y_target_tr):
    """Freeze source weights; corrections use only the target training fold."""
    weights, ymu, _ = own_fit
    p_tr = ridge.standardize(x_target_tr) @ weights + ymu
    direct = ridge.standardize(x_target_ev) @ weights + ymu
    pbar, ybar = p_tr.mean(0), y_target_tr.mean(0)
    pc, yc = p_tr - pbar, y_target_tr - ybar
    denominator = float(np.square(pc).sum())
    assert denominator > 0, "Undefined scalar correction for constant source predictions"
    alpha = float((pc * yc).sum() / denominator)
    return dict(
        direct=direct, bias=direct + ybar - pbar, scale_bias=alpha * direct + ybar - alpha * pbar
    ), alpha


def load_inputs(root, declared):
    manifest = json.loads((root / "manifest.json").read_text())
    assert set(manifest["models"]) == {f"olmo_{s}" for s in STAGES}
    assert set(declared["contexts"]) == {f"olmo_{s}/{f}" for s in STAGES for f in FORMS}
    assert set(declared["targets"]) == {target_key(s, f) for s in STAGES for f in FORMS}
    cohort = manifest["olmo"]
    ids, folds = np.array(cohort["ids"]), np.array(cohort["fold_of"])
    assert len(ids) == 16391 and set(folds) == set(range(6))
    xs, ys = {}, {}
    keep = np.ones(len(ids), dtype=bool)
    for s in STAGES:
        for f in FORMS:
            with np.load(root / declared["contexts"][f"olmo_{s}/{f}"], allow_pickle=False) as p:
                assert np.array_equal(p["ids"], ids) and np.array_equal(p["fold_of"], folds)
                xs[s, f] = p["x"]
            with np.load(root / declared["targets"][target_key(s, f)], allow_pickle=False) as p:
                assert np.array_equal(p["ids"], ids)
                ys[s, f] = p["y"]
                keep &= p["valid"]
    for key in xs:
        xs[key], ys[key] = xs[key][keep].astype(np.float64), ys[key][keep].astype(np.float64)
        assert xs[key].shape == ys[key].shape == (int(keep.sum()), 4096)
        assert np.isfinite(xs[key]).all() and np.isfinite(ys[key]).all()
    coverage = dict(
        planned_rows=len(ids),
        realized_rows=int(keep.sum()),
        excluded_undefined_answer_vectors=ids[~keep].tolist(),
        definition="Common complete-five vector cohort across eight matched on-policy settings; no quality filtering",
    )
    C.write_json(root / "fits" / "olmo" / "cohort.json", coverage)
    return ids[keep], folds[keep], xs, ys


def fit_all(root, declared, ids, folds, xs, ys):
    fp = F.fit_fingerprint(root)
    pilot_path = root / "cpu_pilot.json"
    if pilot_path.exists():
        assert json.loads(pilot_path.read_text())["status"] == "pass", (
            "Prior CPU pilot requires review"
        )
    else:
        first_group_done = all(complete(stem(root, "B", t, "plain", 0), fp) for t in STAGES)
        assert not first_group_done, (
            "First CPU group completed without its timing gate; recover timing before continuing"
        )
    for form in FORMS:
        for source in STAGES:
            targets = STAGES if source == "B" else source
            next_target = STAGES[STAGES.index(source) + 1] if source != "R" else None
            for fold in range(6):
                pending = [
                    t for t in targets if not complete(stem(root, source, t, form, fold), fp)
                ]
                transfer_path = (
                    stem(root, source, next_target, form, fold, True) if next_target else None
                )
                need_transfer = transfer_path is not None and not complete(transfer_path, fp)
                if not pending and not need_transfer:
                    continue
                started = time.monotonic()
                tr, ev = folds != fold, folds == fold
                x = xs[source, form]
                assert tr.sum() > x.shape[1]
                ridge = SharedPrimalRidge(x[tr])
                own_fit = ridge.fit(ys[source, form][tr]) if need_transfer else None
                files = []
                for target in pending:
                    y = ys[target, form]
                    fitted = (
                        own_fit if source == target and own_fit is not None else ridge.fit(y[tr])
                    )
                    weights, ymu, info = fitted
                    assert info["dof"] <= 0.9 * tr.sum()
                    pred = ridge.standardize(x[ev]) @ weights + ymu
                    arrays, metrics = evaluate(pred, y[tr], y[ev])
                    baseline = identity_bias_predict(x[tr], y[tr], x[ev])
                    base_res, _, _ = LC._per_row_components(baseline, y[ev], y[tr].mean(0))
                    path = stem(root, source, target, form, fold)
                    LC._savez(
                        path.with_suffix(".npz"),
                        indices=np.flatnonzero(ev),
                        pred=pred.astype(np.float32),
                        baseline_res=base_res,
                        **arrays,
                    )
                    C.write_json(
                        path.with_suffix(".json"),
                        dict(
                            source=source,
                            target=target,
                            format=form,
                            fold=fold,
                            fit=info,
                            **metrics,
                            identity_bias_r2=float(1 - base_res.sum() / arrays["tot"].sum()),
                            target_key=target_key(target, form),
                            target_sha256=declared["files"][
                                declared["targets"][target_key(target, form)]
                            ],
                        ),
                    )
                    files.extend([path.with_suffix(".npz"), path.with_suffix(".json")])
                    print(
                        f"[phase=fit] {source}->{target} {form} fold={fold} R2={metrics['r2']:.6f}",
                        flush=True,
                    )
                if need_transfer:
                    target = next_target
                    xt, yt = xs[target, form], ys[target, form]
                    preds, alpha = transferred_predictions(ridge, own_fit, xt[tr], xt[ev], yt[tr])
                    arrays, reports = dict(indices=np.flatnonzero(ev)), {}
                    for mode, pred in preds.items():
                        values, metrics = evaluate(pred, yt[tr], yt[ev])
                        arrays[f"res_{mode}"] = values["res"]
                        arrays[f"pred_{mode}"] = pred.astype(np.float32)
                        arrays["tot"] = values["tot"]
                        reports[mode] = metrics
                    LC._savez(transfer_path.with_suffix(".npz"), **arrays)
                    C.write_json(
                        transfer_path.with_suffix(".json"),
                        dict(
                            source=source,
                            target=target,
                            format=form,
                            fold=fold,
                            alpha=alpha,
                            source_fit=own_fit[2],
                            modes=reports,
                            target_key=target_key(target, form),
                            target_sha256=declared["files"][
                                declared["targets"][target_key(target, form)]
                            ],
                        ),
                    )
                    files.extend(
                        [transfer_path.with_suffix(".npz"), transfer_path.with_suffix(".json")]
                    )
                    print(f"[phase=transfer] {source}->{target} {form} fold={fold}", flush=True)
                C.upload_many(files, root, fp)
                seconds = time.monotonic() - started
                print(
                    f"[phase=fit_group] source={source} format={form} fold={fold} seconds={seconds:.1f}",
                    flush=True,
                )
                if source == "B" and form == "plain" and fold == 0:
                    # The largest group (four fits + transfer) bounds all 48 groups.
                    projection = seconds * 48 / 3600
                    report = dict(
                        group_seconds=seconds,
                        conservative_cpu_hours=projection,
                        group="four fits plus three transfer modes",
                        status="pass" if projection <= 10 else "requires_review",
                    )
                    C.write_json(root / "cpu_pilot.json", report)
                    C.upload_many([root / "cpu_pilot.json"], root, fp)
                    assert report["status"] == "pass", (
                        "Measured CPU projection exceeds 12h fence margin"
                    )
                del ridge, own_fit


def summarize(root, ids, folds):
    fp, n = F.fit_fingerprint(root), len(ids)
    results, errors = {}, {}
    for transfer in (False, True):
        for source, target, form in transfer_pairs() if transfer else fit_pairs():
            modes = MODES if transfer else ("fit",)
            names = {
                m: f"{'transfer' if transfer else 'fit'}_{source}{target}_{form}_{m}" for m in modes
            }
            residuals = {m: np.full(n, np.nan) for m in modes}
            total = np.full(n, np.nan)
            metadata = []
            for fold in range(6):
                path = stem(root, source, target, form, fold, transfer)
                assert complete(path, fp)
                metadata.append(json.loads(path.with_suffix(".json").read_text()))
                with np.load(path.with_suffix(".npz"), allow_pickle=False) as p:
                    ix = p["indices"]
                    assert np.array_equal(ix, np.flatnonzero(folds == fold))
                    total[ix] = p["tot"]
                    for mode in modes:
                        residuals[mode][ix] = p[f"res_{mode}" if transfer else "res"]
            assert np.isfinite(total).all()
            for mode, name in names.items():
                assert np.isfinite(residuals[mode]).all()
                results[name] = dict(
                    source=source,
                    target=target,
                    format=form,
                    mode=mode,
                    r2=float(1 - residuals[mode].sum() / total.sum()),
                    folds=metadata,
                )
                errors[name] = (residuals[mode], total.copy())
    names = list(results)
    weights = (
        np.random.default_rng(1944).multinomial(n, np.full(n, 1 / n), size=1000).astype(np.float64)
    )
    boot = 1 - (weights @ np.column_stack([errors[k][0] for k in names])) / (
        weights @ np.column_stack([errors[k][1] for k in names])
    )
    for i, name in enumerate(names):
        results[name]["r2_ci95"] = np.quantile(boot[:, i], [0.025, 0.975]).tolist()
    panel_b, panel_c = [], []
    for form in FORMS:
        for target in "SDR":
            base, own = f"fit_B{target}_{form}_fit", f"fit_{target}{target}_{form}_fit"
            assert np.array_equal(errors[base][1], errors[own][1]), "Panel B targets differ"
            delta = boot[:, names.index(own)] - boot[:, names.index(base)]
            panel_b.append(
                dict(
                    target=target,
                    format=form,
                    base=base,
                    own=own,
                    own_minus_base=results[own]["r2"] - results[base]["r2"],
                    difference_ci95=np.quantile(delta, [0.025, 0.975]).tolist(),
                )
            )
        for source, target in pairwise(STAGES):
            own = f"fit_{target}{target}_{form}_fit"
            assert results[own]["r2"] > 0
            for mode in MODES:
                name = f"transfer_{source}{target}_{form}_{mode}"
                assert np.array_equal(errors[name][1], errors[own][1]), (
                    "Panel C target differs from denominator"
                )
                denominator = boot[:, names.index(own)]
                assert (denominator > 0).all(), "Retention bootstrap denominator is nonpositive"
                ratio = boot[:, names.index(name)] / denominator
                panel_c.append(
                    dict(
                        source=source,
                        target=target,
                        format=form,
                        mode=mode,
                        retention=results[name]["r2"] / results[own]["r2"],
                        retention_ci95=np.quantile(ratio, [0.025, 0.975]).tolist(),
                    )
                )
    output = dict(
        n=n,
        cells=results,
        panel_a=[f"fit_{s}{s}_{f}_fit" for f in FORMS for s in STAGES],
        panel_b=panel_b,
        panel_c=panel_c,
        target_definition="Target-checkpoint encodings of target-generated answers, with generation and capture format matched",
        metric="Held-out pooled SSE/SST centered on training-fold target mean",
        uncertainty="1000 paired context-row bootstrap draws, seed1944; no refit",
    )
    path = root / "fits" / "olmo" / "summary.json"
    C.write_json(path, output)
    C.upload_many([path, root / "fits" / "olmo" / "cohort.json"], root, fp)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/issue1902_olmo_onpolicy_cpu"))
    parser.add_argument("--input-revision", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    from issue1902_format_job import verify_source
    from explore_persona_space.orchestrate.preflight import preflight_check

    verify_source(args.source_sha)
    report = preflight_check(
        require_gpu=False, min_disk_gb=35, planned_footprint_gb=25, required_env_vars=["HF_TOKEN"]
    )
    assert report.ok, report
    args.root.mkdir(parents=True, exist_ok=True)
    declared = F.stage(args.root, args.input_revision)
    ids, folds, xs, ys = load_inputs(args.root, declared)
    fit_all(args.root, declared, ids, folds, xs, ys)
    result = summarize(args.root, ids, folds)
    from issue1902_format_plot import plot

    figure_files = plot(args.root, result)
    C.upload_many(figure_files, args.root, F.fit_fingerprint(args.root))
    receipt = dict(
        status="complete",
        source_sha=args.source_sha,
        input_revision=args.input_revision,
        hf_prefix=C.PREFIX,
        cells_a=len(result["panel_a"]),
        comparisons_b=len(result["panel_b"]),
        transfers_c=len(result["panel_c"]),
        summary="fits/olmo/summary.json",
        figures=[
            f"https://huggingface.co/{C.DEFAULT_OVERFLOW_REPO}/resolve/main/{C.PREFIX}/figures/figure3{letter}_onpolicy.png"
            for letter in "ABC"
        ],
    )
    assert (receipt["cells_a"], receipt["comparisons_b"], receipt["transfers_c"]) == (8, 6, 18)
    C.write_json(args.root / "fits_complete.json", receipt)
    C.upload_many([args.root / "fits_complete.json"], args.root, F.fit_fingerprint(args.root))
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    write_completion_sentinel(
        sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=1902, extra=receipt
    )
    print("[phase=done] OLMo A/B/C fits complete and verified", flush=True)


if __name__ == "__main__":
    main()
