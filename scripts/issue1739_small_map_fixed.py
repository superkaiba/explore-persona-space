"""Four frozen contrastive projections using the generic-only 18,793-pair map.

Restore its native whitening, then unwhiten predicted answers before applying
raw-space directions. No inference, labels, directions, or maps are fitted here.
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from transformers import AutoTokenizer

from explore_persona_space.experiments.issue_1739 import store_io
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts import issue1739_fixed_transfer as ft
from scripts.issue1739_fixed_regimes import (
    BEHAVIORS,
    N_BOOT,
    estimates,
    progress,
    sha256,
    summarize,
    verify_previous,
    write_json,
)

NAMES = ("mapped_answer", "real_answer", "context_native", "answer_direction_on_context")
MAP_SHA = "28b4a76707184010f48bcd94baa9fce975b94d5be67f9357307695921f1952fb"
WHITEN_SHA = "c90297d422833bbd53c85ac223c9d84747101d1d9d849d0931b857f0a31d26ac"


@dataclass
class Config:
    """Existing caches and a fresh output path; statistical protocol is frozen."""

    cache: str = "/dev/shm/issue1739-fixed-transfer"
    covariance: str = "/dev/shm/issue1739-covariance"
    inputs: str = "/mnt/eps-data/thomasjiralerspong/issue1739-small-map-20260917/inputs"
    out: str = "/mnt/eps-data/thomasjiralerspong/issue1739-small-map-20260917/outputs"
    repo: str = "/home/thomasjiralerspong/explore-persona-space"


ConfigStore.instance().store(name="small_map_fixed", node=Config)


def read_lines(path):
    """Parse JSONL by physical lines, preserving Unicode content boundaries."""
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def membership(cfg, meta, mask):
    """Rebuild hashes of every fit prompt and constituent user turn."""
    inputs, out = Path(cfg.inputs), Path(cfg.out)
    corpus = json.loads((inputs / "corpus_source.json").read_text())
    for name, record in corpus["files"].items():
        path = (
            Path(cfg.covariance) / "inputs/u_store" / name
            if name == "manifest.jsonl"
            else inputs / name
        )
        assert sha256(path) == record["sha256"], path
    prefixes = {r["prefix_id"]: r for r in read_lines(inputs / "prefix_store.jsonl")}
    queries = {r["query_id"]: r for r in read_lines(inputs / "query_store.jsonl")}
    tok = AutoTokenizer.from_pretrained(
        ft.MODEL_NAME, revision=ft.INSTRUCT_REVISION, local_files_only=True
    )
    exact, norm = set(), set()
    for row, keep in zip(meta, mask, strict=True):
        if not keep:
            continue
        messages = prefixes[row["prefix_id"]]["prefix_turns"] + [
            {"role": "user", "content": queries[row["query_id"]]["text"]}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for text in [prompt] + [r["content"] for r in messages if r["role"] == "user"]:
            e, n = ft.prompt_hashes(text)
            exact.add(e)
            norm.add(n)
    audit = dict(
        n_fit=int(mask.sum()),
        corpus=corpus,
        exact=sorted(exact),
        normalized=sorted(norm),
        protocol="all rendered fitting prompts and user turns; lowercase/collapse whitespace",
    )
    write_json(out / "map_membership.json", audit)
    return exact, norm


def load_map(cfg):
    """Verify generic-only whitening against the frozen map's native moments."""
    inputs, out, cov = Path(cfg.inputs), Path(cfg.out), Path(cfg.covariance)
    map_path = inputs / "context_end__ufull.npz"
    assert sha256(map_path) == MAP_SHA
    source = np.load(map_path)
    meta = json.loads(str(source["meta"]))
    assert meta["variant"] == "context_end" and meta["u_label"] == "full"
    assert meta["w_fit_rows"] == 18793
    layer = list(source["layers"]).index(19)
    native = {k: source[k][layer].astype(np.float64) for k in ("w", "x_mu", "x_sd", "y_mu")}
    for k in ("x_mu", "x_sd", "y_mu"):
        native[k] = native[k].reshape(-1)
    assert native["w"].shape == (ft.HIDDEN, ft.HIDDEN)
    assert all(np.isfinite(v).all() for v in native.values()) and (native["x_sd"] > 0).all()
    wh_path = cov / "outputs/sycophancy_L19/transforms.npz"
    assert sha256(wh_path) == WHITEN_SHA
    wh = np.load(wh_path)
    mu, S = wh["generic_mu"][0], wh["generic_w"][0]
    manifest_path = cov / "inputs/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prior = json.loads((cov / "outputs/sycophancy_L19/results.json").read_text())
    assert sha256(manifest_path) == prior["input_manifest_sha256"]
    u_pins = {}
    for name in ("context_end_L19.npy", "t1_L19.npy", "manifest.jsonl", "staging_manifest.json"):
        path = cov / "inputs/u_store" / name
        u_pins[name] = manifest["files"]["u_store/" + name]["sha256"]
        assert sha256(path) == u_pins[name], path
    arrays, rows = ft.load_store(cov / "inputs/u_store")
    mask = store_io.fit_pool_mask(rows)
    assert int(mask.sum()) == 18793
    parity = {}
    for kind, keys in (("context_end", ("x_mu", "x_sd")), ("t1", ("y_mu",))):
        z = (arrays[(kind, 19)][mask].astype(np.float64) - mu) @ S
        for key in keys:
            computed = z.std(0) + 1e-9 if key == "x_sd" else z.mean(0)
            # Stored moments are fp32, whereas the verified restoration is fp64.
            np.testing.assert_allclose(computed, native[key], rtol=1e-6, atol=2e-6)
            parity[key] = float(np.max(np.abs(computed - native[key])))
        del z
    del arrays
    exact, normalized = membership(cfg, rows, mask)
    np.savez(out / "native_map_L19.npz", **native)
    np.savez(out / "whitening_L19.npz", mu=mu, S=S, gamma=wh["generic_gamma"])
    invS = np.linalg.inv(S)
    np.testing.assert_allclose(S @ invS, np.eye(ft.HIDDEN), rtol=0, atol=1e-10)
    payload = dict(**native, mu=mu, S=S, invS=invS)
    write_json(
        out / "map_audit.json",
        dict(
            metadata=meta,
            original_map_sha256=MAP_SHA,
            whitening_source_sha256=WHITEN_SHA,
            coordinate_moment_max_error=parity,
            u_source_sha256=u_pins,
            map_source=json.loads((inputs / "map_source.json").read_text()),
            formula="raw prediction = ((((x-mu)@S-x_mu)/x_sd)@w+y_mu)@inverse(S)+mu",
            pooling="#1092 t1 and #1739 t1 are completion-token means, excluding assistant-closing template tokens",
        ),
    )
    gc.collect()
    return payload, exact, normalized


def predict_answers(p, x):
    """Explicitly map a small diagnostic batch back into raw answer coordinates."""
    return ((((x - p["mu"]) @ p["S"] - p["x_mu"]) / p["x_sd"]) @ p["w"] + p["y_mu"]) @ p[
        "invS"
    ] + p["mu"]


def projection(p, x, direction):
    """Pull back a raw answer direction through whitening, map, and unwhitening."""
    r = np.linalg.solve(p["S"], direction)
    q = (p["w"] @ r) / p["x_sd"]
    weight = p["S"] @ q
    offset = p["y_mu"] @ r - p["x_mu"] @ q - p["mu"] @ weight + p["mu"] @ direction
    return x @ weight + offset


def deltas(rho, boot):
    """Paired map-minus-baseline intervals from the same resamples."""
    result = {}
    for j, name in enumerate(NAMES[1:], 1):
        keep = np.isfinite(boot[[0, j]]).all(0)
        difference = boot[0, keep] - boot[j, keep]
        result["mapped_minus_" + name] = dict(
            delta=float(rho[0] - rho[j]),
            ci95=np.quantile(difference, [0.025, 0.975]).tolist(),
            valid_bootstrap_draws=int(keep.sum()),
        )
    return result


def run_behavior(cfg, behavior, index, exact, normalized, payload, prior):
    """Reuse fixed directions and score four arms on identical retained rollouts."""
    root, cache = Path(cfg.out), Path(cfg.cache)
    out = root / behavior
    out.mkdir()
    previous = cache / "outputs/analysis" / behavior
    verify_previous(previous)
    dirs = np.load(previous / "directions.npz")
    va, vc = dirs["answer"], dirs["context"]
    assert int(dirs["layer"]) == 19
    shutil.copyfile(previous / "directions.npz", out / "directions.npz")
    extraction = [r for (ns, _), r in index.items() if ns == f"{behavior}_extraction"]
    assert len(extraction) == 200 and not any(ft.overlaps(r, exact, normalized) for r in extraction)
    ex = set().union(*(ft.content_hash_set(r, "exact") for r in extraction))
    no = set().union(*(ft.content_hash_set(r, "normalized") for r in extraction))
    old = json.loads((previous / "results.json").read_text())
    supplementary = (
        {
            str(r["context_id"]): r
            for r in ft.json_rows(cache / "inputs/provenance/hallucination_per_rollout.json")
        }
        if behavior == "hallucination"
        else None
    )
    batches, audits, exclusions, coverage = [], [], [], []
    for namespace, base in (
        (f"{behavior}_labeling", "dv_dataset"),
        ("wildchat", "wildchat_rung/dv_dataset"),
    ):
        path = Path(cfg.repo) / "eval_results/issue_1739" / base / behavior / "labeling.json"
        assert sha256(path) in old["input_label_sha256"].values()
        selected = [
            r
            for r in ft.json_rows(path)
            if (r["split"] == "eval" and r["rung"] in ft.ROSTER[behavior])
            or (r["split"] == r["rung"] == "train")
        ]
        if namespace == "wildchat":
            selected = [
                r
                for r, keep in zip(
                    selected,
                    ft._wc_eval_mask([str(r["context_id"]) for r in selected]),
                    strict=True,
                )
                if keep
            ]
        retained = []
        for row in selected:
            h = ft.lookup_prompt(index, namespace, str(row["context_id"]))
            reason = (
                "no_valid_behavior_label"
                if row["dv"] is None
                else "overlap_map_fit"
                if ft.overlaps(h, exact, normalized)
                else "overlap_direction_extraction"
                if ft.overlaps(h, ex, no)
                else None
            )
            if reason:
                exclusions.append(
                    dict(context_id=str(row["context_id"]), rung=row["rung"], reason=reason)
                )
            else:
                retained.append(row)
        progress(
            root,
            "load_evaluation",
            behavior=behavior,
            namespace=namespace,
            planned=len(selected),
            retained=len(retained),
        )
        store = cache / "inputs" / namespace
        manifest = json.loads((store / "slice_manifest.json").read_text())
        for name, record in manifest["members"].items():
            assert record["sha256"] == prior["store_sha256"][str(store / name)]
            assert sha256(store / name) == record["sha256"], name
        arrays, meta = ft.load_store(store)
        audits.append(
            ft.validate_capture_provenance(
                meta, index, namespace, context_ids=[str(r["context_id"]) for r in retained]
            )
        )
        batches.append(
            ft.reduce_eval(
                arrays,
                meta,
                retained,
                supplementary=supplementary if namespace != "wildchat" else None,
            )
        )
        del arrays, meta
        gc.collect()
        coverage.append(
            dict(
                namespace=namespace,
                planned=len(selected),
                retained=len(retained),
                labels_sha256=sha256(path),
            )
        )
    data = {
        k: np.concatenate([b[k] for b in batches])
        for k in ("x", "y", "dv", "context_ids", "groups", "rungs")
    }
    assert len(set(data["context_ids"])) == len(data["context_ids"])
    assert set(data["rungs"]) == set(ft.ROSTER[behavior]) | {"train"}
    pred = np.stack(
        (projection(payload, data["x"], va), data["y"] @ va, data["x"] @ vc, data["x"] @ va)
    )
    assert np.isfinite(pred).all()
    explicit = predict_answers(payload, data["x"][:17]) @ va
    np.testing.assert_allclose(explicit, pred[0, :17], rtol=1e-10, atol=1e-7)
    take = data["rungs"] == "wildchat_rung"
    bias = (payload["y_mu"] - payload["x_mu"]) @ payload["invS"]
    reconstruction = dict(
        frozen_map=ft.reconstruction_metrics(
            predict_answers(payload, data["x"][take]), data["y"][take]
        ),
        identity_plus_bias=ft.reconstruction_metrics(data["x"][take] + bias, data["y"][take]),
        evaluation="held-out generic chat, same retained-rollout averages as behavior scores",
    )
    write_json(out / "reconstruction.json", reconstruction)
    write_json(out / "rollout_alignment.json", [r for b in batches for r in b["rollout_audit"]])
    del data["x"], data["y"], batches
    gc.collect()
    np.savez(out / "predictions.npz", predictions=pred, arms=NAMES, **data)
    audit = dict(
        coverage=coverage,
        exclusions=exclusions,
        exclusion_counts=dict(Counter(r["reason"] for r in exclusions)),
        capture_provenance=audits,
        extraction_contexts=200,
        extraction_overlap=0,
        affine_projection_max_abs_error=float(np.max(np.abs(explicit - pred[0, :17]))),
    )
    write_json(out / "audit.json", audit)
    cells, draws = summarize(
        data,
        pred,
        seed=1739963 + BEHAVIORS.index(behavior),
        out=root,
        behavior=behavior,
        names=NAMES,
    )
    np.savez(out / "bootstraps.npz", **draws)
    for cell in cells:
        cell["differences"] = deltas([cell["arms"][n]["rho"] for n in NAMES], draws[cell["rung"]])
    lookup = {r["rung"]: r for r in cells}
    ood = [r for r in ft.ROSTER[behavior] if r != "wildchat_rung"]
    rho = np.mean([[lookup[r]["arms"][n]["rho"] for n in NAMES] for r in ood], axis=0)
    boot = np.mean([draws[r] for r in ood], axis=0)
    regimes = [
        dict(regime="generic chat", datasets=["wildchat_rung"], **lookup["wildchat_rung"]),
        dict(regime="in-distribution", datasets=["train"], **lookup["train"]),
        dict(
            regime="OOD",
            datasets=ood,
            n=sum(lookup[r]["n"] for r in ood),
            informative=True,
            arms=estimates(rho, boot, NAMES),
            differences=deltas(rho, boot),
        ),
    ]
    result = dict(behavior=behavior, datasets=cells, regimes=regimes)
    write_json(out / "results.json", result)
    write_json(
        out / "complete.json",
        dict(
            finished_at=time.time(),
            artifact_sha256={p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        ),
    )
    progress(root, "behavior_complete", behavior=behavior)
    return result


@hydra.main(version_base=None, config_path=None, config_name="small_map_fixed")
def main(cfg: Config):
    """Run the cached CPU analysis with pinned inputs and per-behavior completion."""
    out, cache = Path(cfg.out), Path(cfg.cache)
    out.mkdir(parents=True, exist_ok=False)
    provenance = as_metadata_dict(git_provenance(ROOT), phase="fixed-small-map")
    progress(out, "verify_map")
    prior = json.loads((cache / "outputs/analysis/config.json").read_text())
    for name, key in (
        ("prompt_index.jsonl", "prompt_index_sha256"),
        ("hallucination_per_rollout.json", "hallu_per_rollout_sha256"),
    ):
        assert sha256(cache / "inputs/provenance" / name) == prior[key]
    index = ft.load_prompt_index(cache / "inputs/provenance/prompt_index.jsonl")
    payload, exact, normalized = load_map(cfg)
    summary = dict(
        **provenance,
        source_sha=provenance["git_commit"],
        script_sha256=sha256(Path(__file__)),
        map_training_pairs=18793,
        layer=19,
        methods=list(NAMES),
        n_boot=N_BOOT,
        original_map_sha256=MAP_SHA,
        prior_config_sha256=sha256(cache / "outputs/analysis/config.json"),
        excluded_evaluation="pvsynth",
        directions="frozen raw unfiltered instruction contrasts",
        in_distribution="historical behavior-regression training rung; no readout fitted here",
        ood_aggregation="equal-weight mean of dataset Spearman correlations; independent dataset group resamples, paired methods",
        comparison_scope="smaller-map companion; different map recipe and held-out roster from million-map analysis",
    )
    write_json(out / "config.json", summary)
    summary["behaviors"] = [
        run_behavior(cfg, b, index, exact, normalized, payload, prior) for b in BEHAVIORS
    ]
    write_json(out / "summary.json", summary)
    write_json(
        out / "complete.json",
        dict(
            finished_at=time.time(),
            source_sha=summary["source_sha"],
            artifact_sha256={
                str(p.relative_to(out)): sha256(p)
                for p in out.rglob("*")
                if p.is_file() and p != out / "progress.json"
            },
        ),
    )
    progress(out, "complete")


if __name__ == "__main__":
    main()
