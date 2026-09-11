#!/usr/bin/env python3
"""Stage verified #1901 K10 captures, then render/publish completed fit artifacts.

This entrypoint never fits a map or generates an answer. Stage into a directory
outside the code checkout, run the separate fit entrypoint, and only then use
report/publish. All plotted statistics are checked against saved numerical arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()
import numpy as np  # noqa: E402

import issue1901_training_k10_gpu as GPU  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

ARMS = ("ridge", "identity_bias")
METRICS = ("whiten_csls", "whiten_cosine", "raw_cosine", "raw_euclidean")


def safe_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"unsafe artifact path: {relative}")
    return path


def checked_capture_source(prefix, revision):
    assert re.fullmatch(r"issue1901_training_k10/capture_[0-9a-f]{16}", prefix)
    assert re.fullmatch(r"[0-9a-f]{40}", revision), "an immutable capture commit is required"


def stage_capture(args):
    """Download only declared tensors, with scoped pinned paths and four workers."""
    checked_capture_source(args.capture_prefix, args.capture_revision)
    root = args.capture.resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("capture staging must be outside the sparse code worktree")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "capture_manifest.json"
    # Always check the requested immutable source, including an existing local target.
    hub.stage_sharded_text(
        GPU.REPO,
        f"{args.capture_prefix}/capture_manifest.json",
        manifest_path,
        repo_type="dataset",
        revision=args.capture_revision,
        overwrite=True,
    )
    manifest = GPU.read_json(manifest_path)
    assert manifest["expected_new_rows"] == manifest["realized_new_rows"] == 95000
    assert GPU.digest_json(manifest["recipe"]) == manifest["recipe_sha256"]
    assert GPU.run_prefix(manifest["recipe"]) == args.capture_prefix
    entries = manifest["files"]
    assert entries and len({entry["path"] for entry in entries}) == len(entries)

    def fetch(entry):
        dest = safe_path(root, entry["path"])
        if not dest.exists():
            hub.stage_hub_file(
                GPU.REPO,
                f"{args.capture_prefix}/{entry['path']}",
                dest,
                repo_type="dataset",
                revision=args.capture_revision,
            )
        assert GPU.sha_file(dest) == entry["sha256"], f"capture hash mismatch: {entry['path']}"
        with np.load(dest, allow_pickle=False) as bank:
            assert bank["V"].shape == (entry["rows"], GPU.DIM)
            assert np.isfinite(bank["V"]).all() and np.all(bank["n_ans"] > 0)
            assert bank["ci"].tolist() == entry["ci"] and int(bank["seed"]) == entry["seed"]
            assert json.loads(str(bank["recipe"])) == manifest["recipe"]
        return entry["rows"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(fetch, entries))
    pairs = [(int(ci), entry["seed"]) for entry in entries for ci in entry["ci"]]
    assert len(pairs) == len(set(pairs)) == sum(counts) == 95000
    by_seed = {seed: {ci for ci, draw in pairs if draw == seed} for seed in GPU.SEEDS}
    assert set(seed for _, seed in pairs) == set(GPU.SEEDS)
    assert all(len(ids) == 19000 and ids == by_seed[47] for ids in by_seed.values())
    record = {
        "repo": GPU.REPO,
        "prefix": args.capture_prefix,
        "revision": args.capture_revision,
        "manifest_sha256": GPU.sha_file(manifest_path),
        "files": len(entries),
        "new_rows": sum(counts),
        "execution_completion": "Owner checks fresh workload sentinel separately",
    }
    GPU.write_json(root / "stage_verification.json", record)
    if args.audit_raw:
        audit_raw(args, manifest)
    print(json.dumps(record), flush=True)


def audit_raw(args, manifest):
    """Optional exact raw-text staging: cap counts and disclosed inherited JWT edits."""
    source_dir = args.capture / "input_metadata"
    source_revision = manifest["recipe"]["input_revision"]
    source_manifest = hub.stage_hub_file(
        GPU.REPO,
        f"{GPU.PREFIX}/inputs/manifest.json",
        source_dir / "manifest.json",
        repo_type="dataset",
        revision=source_revision,
    )
    assert GPU.sha_file(source_manifest) == manifest["input_manifest_sha256"]
    source = GPU.read_json(source_manifest)
    prompt_file = hub.stage_sharded_text(
        GPU.REPO,
        f"{GPU.PREFIX}/inputs/prompts.json",
        source_dir / "prompts.json",
        repo_type="dataset",
        revision=source_revision,
    )
    assert GPU.sha_file(prompt_file) == source["outputs"]["prompts.json"]["sha256"]
    prompts = GPU.read_json(prompt_file)["train"]
    by_ci = {r["ci"]: r for r in prompts}
    assert len(by_ci) == len(prompts) == 19000

    def one(entry):
        captured = safe_path(args.capture, entry["path"])
        raw_name = entry["path"].replace("/analysis_tensors/", "/raw_completions/")
        assert raw_name != entry["path"] and raw_name.endswith(".npz")
        raw_name = raw_name[:-4] + ".json"
        raw_path = safe_path(args.capture, raw_name)
        hub.stage_sharded_text(
            GPU.REPO,
            f"{args.capture_prefix}/{raw_name}",
            raw_path,
            repo_type="dataset",
            revision=args.capture_revision,
        )
        with np.load(captured, allow_pickle=False) as bank:
            expected_hash = str(bank["generation_sha"])
        assert GPU.sha_file(raw_path) == expected_hash, raw_name
        raw = GPU.read_json(raw_path)
        chunk = int(re.search(r"_chunk([0-9]+)\.json$", raw_name).group(1))
        GPU.validate_generation(
            raw, [by_ci[ci] for ci in entry["ci"]], manifest["recipe"], entry["seed"], chunk
        )
        assert raw["seed"] == entry["seed"] and raw["recipe"] == manifest["recipe"]
        assert [r["ci"] for r in raw["rows"]] == entry["ci"]
        caps = sum(len(r["token_ids"]) == GPU.SETTINGS["max_tokens"] for r in raw["rows"])
        assert caps == raw["cap_hits"]
        changed = sum(
            hashlib.sha256(r["text"].encode()).hexdigest() != r["original_text_sha256"]
            for r in raw["rows"]
        )
        disclosures = raw["scrub_disclosures"]
        assert changed == 0 or disclosures, "changed text lacks scrub disclosure"
        return {
            "path": raw_name,
            "sha256": expected_hash,
            "rows": len(raw["rows"]),
            "cap_hits": caps,
            "changed_answers": changed,
            "scrub_findings": len(disclosures),
            "scrub_rules": dict(Counter(d["rule"] for d in disclosures)),
            "generation_compute_seconds": raw["compute_seconds"],
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(one, manifest["files"]))
    result = {
        "manifest_sha256": GPU.sha_file(args.capture / "capture_manifest.json"),
        "revision": args.capture_revision,
        "prefix": args.capture_prefix,
        "new_answers": sum(c["rows"] for c in chunks),
        "cap_hits": sum(c["cap_hits"] for c in chunks),
        "answers_changed_by_inherited_scrub": sum(c["changed_answers"] for c in chunks),
        "scrub_findings": sum(c["scrub_findings"] for c in chunks),
        "scrub_count_unit": "Field-level findings; one answer can appear in both text aliases",
        "chunks": chunks,
    }
    assert result["new_answers"] == 95000
    GPU.write_json(args.capture / "generation_audit.json", result)
    return result


def check_stat(saved, point, boots):
    expected = np.array([point, *np.quantile(boots, [0.025, 0.975])], dtype=float)
    actual = np.array([saved["mean"], *saved["ci95"]], dtype=float)
    assert np.isfinite(actual).all() and actual[1] <= actual[2]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def checked_analysis(directory, capture):
    """Check all plotted/table statistics against persisted bootstrap arrays."""
    summary = GPU.read_json(directory / "summary.json")
    completion = GPU.read_json(directory / "completion.json")
    assert completion["summary_sha256"] == GPU.sha_file(directory / "summary.json")
    assert summary["n_train"] == completion["n_train"] == 19000
    assert summary["n_test"] == 1000
    assert (
        summary["n_candidates"] == summary["n_prompt_clusters"] == completion["n_candidates"] == 942
    )
    assert summary["chance_top1"] == 1 / 942
    assert summary["provenance"]["capture_manifest_sha256"] == GPU.sha_file(
        capture / "capture_manifest.json"
    )
    with np.load(directory / "analysis_tensors.npz", allow_pickle=False) as bank:
        r2, r2_boot = bank["r2"], bank["r2_boot"]
        hits, retrieval_boot = bank["hits"], bank["retrieval_boot"]
        assert r2.shape == (10, 20) and r2_boot.shape == (10, 20, 2000)
        assert hits.shape == (10, 20, len(METRICS), 942)
        assert retrieval_boot.shape == (10, 20, len(METRICS), 2000)
        assert np.isin(hits, [0, 1]).all()
        for train_k in range(1, 11):
            for arm_i, arm in enumerate(ARMS):
                row = (train_k - 1) * 2 + arm_i
                for eval_k in range(1, 11):
                    cell = summary["cells"][str(train_k)][arm][str(eval_k)]
                    check_stat(cell["r2"], r2[eval_k - 1, row], r2_boot[eval_k - 1, row])
                    for m, metric in enumerate(METRICS):
                        check_stat(
                            cell["retrieval"][metric],
                            hits[eval_k - 1, row, m].mean(),
                            retrieval_boot[eval_k - 1, row, m],
                        )
        for high, low in ((10, 1), (5, 1), (10, 5)):
            for arm_i, arm in enumerate(ARMS):
                h, l = (high - 1) * 2 + arm_i, (low - 1) * 2 + arm_i
                cell = summary["contrasts"][f"K{high}_minus_K{low}_eval10"][arm]
                check_stat(cell["r2"], r2[-1, h] - r2[-1, l], r2_boot[-1, h] - r2_boot[-1, l])
                for m, metric in enumerate(METRICS):
                    check_stat(
                        cell["retrieval"][metric],
                        hits[-1, h, m].mean() - hits[-1, l, m].mean(),
                        retrieval_boot[-1, h, m] - retrieval_boot[-1, l, m],
                    )
    fits = GPU.read_json(directory / "fits" / "manifest.json")
    assert fits["provenance"] == summary["provenance"], "fit and score provenance differ"
    assert [f["k_train"] for f in fits["fits"]] == list(range(1, 11))
    for entry in fits["fits"]:
        assert GPU.sha_file(directory / "fits" / f"k{entry['k_train']:02d}.npz") == entry["sha256"]
    return summary, fits


def render(summary, directory, heatmaps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from explore_persona_space.analysis import c2a_plot_style as S

    S.set_c2a_style()
    fig, fraction = S.c2a_figure("full", aspect=0.49)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.74, wspace=0.34)
    xs = np.arange(1, 11)
    for arm, role_name, label in (
        ("ridge", "linear", "Linear map"),
        ("identity_bias", "control", "Copy + bias"),
    ):
        role = S.ROLES[role_name]
        cells = [summary["cells"][str(k)][arm]["10"] for k in xs]
        for column, metric in enumerate(("r2", "whiten_csls")):
            series = [c["r2"] if metric == "r2" else c["retrieval"][metric] for c in cells]
            mean = np.array([c["mean"] for c in series])
            bounds = np.array([c["ci95"] for c in series])
            axis = axes[column]
            axis.plot(
                xs,
                mean,
                color=role.color,
                marker=role.marker,
                label=label,
                linewidth=2.2,
                markersize=6.5,
                linestyle="-" if arm == "ridge" else "--",
            )
            # Draw interval endpoints directly; percentile intervals can exclude their estimate.
            axis.fill_between(
                xs, bounds[:, 0], bounds[:, 1], color=role.color, alpha=0.14, linewidth=0
            )
    for column, axis in enumerate(axes):
        S.style_axis(axis)
        axis.set_xticks(xs)
        axis.set_xlabel("Training answer rollouts averaged")
        S.panel_header(
            axis,
            "AB"[column],
            "Evaluation K = 10",
            "Held-out reconstruction" if column == 0 else "Answer retrieval",
        )
    axes[0].set_ylabel(S.better_label(r"Held-out $R^2$"))
    axes[1].set_ylabel(S.better_label("Top-1 retrieval"))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    S.legend_kicker(fig, 0.09, 0.95, "Prediction")
    fig.legend(
        handles, labels, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.25, 0.991)
    )
    outputs = {
        "training_k_fixed_eval10": S.save_c2a_figure(
            fig,
            directory / "training_k_fixed_eval10",
            title="Training answer averages",
            subject="Evaluation target fixed at ten answers; paired prompt-cluster95% intervals",
            creator=Path(__file__).name,
            include_width=fraction,
        )
    }
    plt.close(fig)
    if heatmaps:
        fig, fraction = S.c2a_figure("full", aspect=0.53)
        axes = fig.subplots(1, 2)
        fig.subplots_adjust(left=0.07, right=0.94, bottom=0.17, top=0.77, wspace=0.50)
        for column, metric in enumerate(("r2", "whiten_csls")):
            values = np.array(
                [
                    [
                        summary["cells"][str(t)]["ridge"][str(e)]["r2"]["mean"]
                        if metric == "r2"
                        else summary["cells"][str(t)]["ridge"][str(e)]["retrieval"][metric]["mean"]
                        for e in range(1, 11)
                    ]
                    for t in range(1, 11)
                ]
            )
            assert values.shape == (10, 10) and np.isfinite(values).all()
            axis = axes[column]
            image = axis.imshow(
                values,
                origin="lower",
                extent=(0.5, 10.5, 0.5, 10.5),
                cmap="cividis",
                aspect="equal",
            )
            axis.set_xticks(xs)
            axis.set_yticks(xs)
            axis.set_xlabel("Evaluation answer rollouts")
            axis.set_ylabel("Training answer rollouts")
            S.panel_header(
                axis,
                "AB"[column],
                "Linear map",
                "Held-out reconstruction" if column == 0 else "Answer retrieval",
            )
            colorbar = fig.colorbar(image, ax=axis, fraction=0.048, pad=0.05)
            colorbar.set_label(S.better_label(r"$R^2$" if column == 0 else "Top-1 retrieval"))
            if column == 1:
                colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
        outputs["training_eval_k_grid"] = S.save_c2a_figure(
            fig,
            directory / "training_eval_k_grid",
            title="Training and evaluation answer averages",
            subject="Descriptive10by10grid, fixed candidate identities and distance transform",
            creator=Path(__file__).name,
            include_width=fraction,
        )
        plt.close(fig)
    return outputs


def formatted(stat, *, percentage=False):
    factor = 100 if percentage else 1
    point = stat["mean"] * factor
    lo, hi = np.array(stat["ci95"]) * factor
    return (
        f"{point:.2f} [{lo:.2f}, {hi:.2f}]" if percentage else f"{point:.4f} [{lo:.4f}, {hi:.4f}]"
    )


def build_report(args):
    summary, fits = checked_analysis(args.analysis_dir, args.capture)
    output = args.analysis_dir / "report"
    output.mkdir(parents=True, exist_ok=True)
    exports = render(summary, output, args.heatmaps)
    record = {
        "summary_sha256": GPU.sha_file(args.analysis_dir / "summary.json"),
        "analysis_tensors_sha256": GPU.sha_file(args.analysis_dir / "analysis_tensors.npz"),
        "capture_manifest_sha256": GPU.sha_file(args.capture / "capture_manifest.json"),
        "script_sha256": GPU.sha_file(Path(__file__)),
        "exports": exports,
    }
    source = GPU.read_json(args.capture / "stage_verification.json")
    assert source["manifest_sha256"] == record["capture_manifest_sha256"]
    record["capture_source"] = source
    GPU.write_json(
        output / "capture_manifest.json", GPU.read_json(args.capture / "capture_manifest.json")
    )
    # save_c2a_figure returns Path objects for files; store portable relative paths.
    for export in exports.values():
        export["sha256"] = {
            key: GPU.sha_file(Path(export[key])) for key in ("pdf", "png", "grayscale")
        }
        for key in ("pdf", "png", "grayscale"):
            export[key] = Path(export[key]).relative_to(args.analysis_dir).as_posix()
    GPU.write_json(output / "figure_provenance.json", record)
    edges = [f["k_train"] for f in fits["fits"] if f["lambda_grid_edge"] is not None]
    final = {
        "n_train": summary["n_train"],
        "n_test": summary["n_test"],
        "n_candidates": summary["n_candidates"],
        "chance_top1": summary["chance_top1"],
        "fixed_eval_k": 10,
        "coverage": "10/10 training-K fits; 100/100 train/eval cells per predictor",
        "anchors": {
            str(k): {arm: summary["cells"][str(k)][arm]["10"] for arm in ARMS} for k in (1, 5, 10)
        },
        "contrasts": summary["contrasts"],
        "lambda_grid_edge_k": edges,
        "summary_sha256": record["summary_sha256"],
        "capture_source": source,
        "interval_scope": "Conditional on fixed training bank, maps, observed rollout banks and retrieval pool",
    }
    audit_path = args.capture / "generation_audit.json"
    if audit_path.exists():
        audit = GPU.read_json(audit_path)
        assert audit["manifest_sha256"] == record["capture_manifest_sha256"]
        GPU.write_json(output / "generation_audit.json", audit)
        assert audit["new_answers"] == 95000
        final["generation_audit"] = {k: v for k, v in audit.items() if k != "chunks"}
    else:
        final["generation_audit"] = {
            "status": "not_staged",
            "instruction": "stage --audit-raw to verify cap/scrub counts",
        }
    GPU.write_json(output / "final_summary.json", final)
    return summary, final, exports


def write_readme(directory, summary, final, figure_urls=None):
    lines = [
        "# Training answer rollout count",
        "",
        "Separate linear maps were fitted at each training K=1–10 using the same 19,000 contexts. "
        "The primary evaluation holds ten answer rollouts per test context fixed.",
        "",
        "| Training K | Predictor | Held-out R² [95% CI] | Top-1 retrieval %, [95% CI] |",
        "|---:|---|---:|---:|",
    ]
    for k in (1, 5, 10):
        for arm, label in (("ridge", "Linear map"), ("identity_bias", "Copy + bias")):
            cell = summary["cells"][str(k)][arm]["10"]
            lines.append(
                f"| {k} | {label} | {formatted(cell['r2'])} | "
                f"{formatted(cell['retrieval']['whiten_csls'], percentage=True)} |"
            )
    lines += [
        "",
        f"Retrieval uses 942 fixed candidates (chance {summary['chance_top1']:.4%}); R² uses all 1,000 test rows. "
        "Intervals use 2,000 paired prompt-cluster bootstrap draws and condition on the fixed "
        "training bank, fitted maps, observed rollouts and retrieval pool. "
        "Whitening is fixed from the original 19,000 training answers; two-sided CSLS uses neighborhood 10.",
        "",
        "The pre-registered primary contrast is training K10 minus K1 at evaluation K10:",
    ]
    primary = summary["contrasts"]["K10_minus_K1_eval10"]["ridge"]
    lines += [
        f"R² difference {formatted(primary['r2'])}; top-1 difference "
        f"{formatted(primary['retrieval']['whiten_csls'], percentage=True)} percentage points.",
        "",
        f"Completed coverage: {final['coverage']}. Lambda-grid edge selections: "
        f"{final['lambda_grid_edge_k'] or 'none'}. Expanded-grid diagnostics, when present, "
        "did not change reported predictions.",
        "",
        "The fixed answer order is original, 43, …, 51. The curve is conditional on this order; "
        "it does not average all subsets or estimate training-bank sampling uncertainty.",
        "",
    ]
    if figure_urls:
        for stem, urls in figure_urls.items():
            lines += [
                f"![{stem.replace('_', ' ')}]({urls['png']})",
                "",
                f"[Vector PDF]({urls['pdf']}) · [Grayscale audit]({urls['grayscale']})",
                "",
            ]
    else:
        lines += [
            "Figures are generated locally. The publish phase inserts verified browser URLs.",
            "",
        ]
    audit = final["generation_audit"]
    if audit.get("status") == "not_staged":
        lines += [
            "Raw generation cap-hit and inherited JWT-scrub counts were not staged. "
            "Immutable raw outputs remain under the capture prefix; use stage --audit-raw.",
            "",
        ]
    else:
        lines += [
            f"Of {audit['new_answers']:,} new answers, {audit['cap_hits']:,} reached the "
            f"1,024-token cap and {audit['answers_changed_by_inherited_scrub']:,} were changed "
            "by the inherited generated-JWT handler. These counts were verified against raw "
            "text/token files whose hashes match the capture arrays. Original token IDs and "
            "pre-edit text hashes are retained; capture used the disclosed edited text. "
            "See generation_audit.json for per-chunk evidence.",
            "",
        ]
    (directory / "README.md").write_text("\n".join(lines))


def browser_url(prefix, relative, revision):
    return f"https://huggingface.co/datasets/{GPU.REPO}/resolve/{revision}/{quote(prefix + '/' + relative)}"


def report(args):
    summary, final, _ = build_report(args)
    write_readme(args.analysis_dir / "report", summary, final)
    print(json.dumps(final), flush=True)


def publish(args):
    summary, final, exports = build_report(args)
    assert final["generation_audit"].get("status") != "not_staged", (
        "publish requires stage --audit-raw"
    )
    prefix = f"{GPU.PREFIX}/analysis_{final['summary_sha256'][:16]}"
    directory = args.analysis_dir
    write_readme(directory / "report", summary, final)
    paths = [
        p
        for p in directory.rglob("*")
        if p.is_file()
        and not p.name.endswith(".tmp")
        and p.name not in ("publication.json", "publication_receipt.json")
    ]
    assert paths and (directory / "analysis_tensors.npz") in paths
    receipt = GPU.upload_files(directory, paths, prefix)
    figure_urls = {
        stem: {
            key: browser_url(prefix, export[key], receipt["revision"])
            for key in ("pdf", "png", "grayscale")
        }
        for stem, export in exports.items()
    }
    # These URLs point to already hash-verified immutable figure objects.
    write_readme(directory / "report", summary, final, figure_urls)
    publication = {
        "repo": GPU.REPO,
        "prefix": prefix,
        "artifact_receipt": receipt,
        "figure_urls": figure_urls,
        "final_summary": final,
    }
    GPU.write_json(directory / "publication.json", publication)
    final_receipt = GPU.upload_files(
        directory, [directory / "publication.json", directory / "report" / "README.md"], prefix
    )
    publication["report_url"] = browser_url(prefix, "report/README.md", final_receipt["revision"])
    publication["publication_url"] = browser_url(
        prefix, "publication.json", final_receipt["revision"]
    )
    publication["summary_url"] = browser_url(prefix, "summary.json", final_receipt["revision"])
    publication["final_summary_url"] = browser_url(
        prefix, "report/final_summary.json", final_receipt["revision"]
    )
    final_receipt.update(
        {
            "repo": GPU.REPO,
            "prefix": prefix,
            "report_url": publication["report_url"],
            "summary_url": publication["summary_url"],
            "final_summary_url": publication["final_summary_url"],
            "publication_url": publication["publication_url"],
            "figure_urls": figure_urls,
            "final_summary": final,
            "artifact_receipt": receipt,
        }
    )
    GPU.write_json(directory / "publication_receipt.json", final_receipt)
    print(json.dumps(publication), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("stage", "report", "publish"), required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-prefix")
    parser.add_argument("--capture-revision")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--audit-raw", action="store_true")
    parser.add_argument("--heatmaps", action="store_true")
    parser.add_argument("--import-check", action="store_true")
    args = parser.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("import-check: PASS; no data staged, fitted, plotted or published")
        return
    if args.phase == "stage":
        if args.capture_prefix is None or args.capture_revision is None:
            parser.error("stage requires --capture-prefix and --capture-revision")
        stage_capture(args)
    else:
        if args.analysis_dir is None:
            parser.error("report/publish requires --analysis-dir")
        {"report": report, "publish": publish}[args.phase](args)


if __name__ == "__main__":
    main()
