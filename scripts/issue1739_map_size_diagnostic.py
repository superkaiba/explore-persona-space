"""Cached map-size comparison and explicitly overlapping generic-chat diagnostic.

ID/OOD compares saved fixed-direction scores on their intersection. Generic chat
uses the same fixed directions while deliberately retaining map-training prompts;
this is a diagnostic, never a held-out generalization estimate.
"""

from __future__ import annotations

import gc
import json
import sys
import time
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
from scipy.stats import spearmanr

from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts import issue1739_fixed_transfer as ft
from scripts.issue1739_fixed_regimes import (
    BEHAVIORS,
    progress,
    sha256,
    summarize,
    verify_input_pins,
    verify_previous,
    write_json,
)
from scripts.issue1739_small_map_fixed import NAMES, deltas


@dataclass
class Config:
    """Existing caches and a separate output directory for the overlap diagnostic."""

    cache: str = "/dev/shm/issue1739-fixed-transfer"
    small: str = "/mnt/eps-data/thomasjiralerspong/issue1739-small-map-20260917/outputs"
    large: str = "/mnt/eps-data/thomasjiralerspong/issue1739-fixed-regimes-20260917"
    repo: str = "/home/thomasjiralerspong/explore-persona-space"
    out: str = "/mnt/eps-data/thomasjiralerspong/issue1739-map-size-diagnostic-20260917"


ConfigStore.instance().store(name="map_size_diagnostic", node=Config)


def matched_heldout(cfg, behavior):
    """Compare map scores only after exact context, label, group and rung joins."""
    small = dict(np.load(Path(cfg.small) / behavior / "predictions.npz"))
    large = dict(np.load(Path(cfg.large) / behavior / "predictions.npz"))
    lookup = {cid: i for i, cid in enumerate(small["context_ids"])}
    pairs = [
        (lookup[cid], j)
        for j, cid in enumerate(large["context_ids"])
        if cid in lookup and large["rungs"][j] != "wildchat_rung"
    ]
    si, li = np.asarray(pairs).T
    for key in ("dv", "groups", "rungs"):
        np.testing.assert_array_equal(small[key][si], large[key][li])
    for name in ("real_answer", "context_native"):
        np.testing.assert_allclose(
            small["predictions"][list(small["arms"]).index(name), si],
            large["predictions"][list(large["arms"]).index(name), li],
            rtol=1e-10,
            atol=1e-7,
        )
    predictions = np.stack(
        [
            d["predictions"][list(d["arms"]).index("mapped_answer"), ix]
            for d, ix in ((small, si), (large, li))
        ]
    )
    data = {k: large[k][li] for k in ("dv", "groups", "rungs", "context_ids")}
    cells = []
    for rung in sorted(set(data["rungs"])):
        mask = data["rungs"] == rung
        rho = [float(spearmanr(v[mask], data["dv"][mask]).statistic) for v in predictions]
        cells.append(
            dict(
                rung=str(rung),
                n=int(mask.sum()),
                small_rho=rho[0],
                million_rho=rho[1],
                small_minus_million=rho[0] - rho[1],
            )
        )
    ood = [r for r in cells if r["rung"] != "train"]
    regime = dict(
        regime="OOD",
        datasets=[r["rung"] for r in ood],
        n=sum(r["n"] for r in ood),
        **{
            k: float(np.mean([r[k] for r in ood]))
            for k in ("small_rho", "million_rho", "small_minus_million")
        },
    )
    np.savez(
        Path(cfg.out) / behavior / "matched_heldout.npz",
        predictions=predictions,
        methods=["small_map", "million_map"],
        **data,
    )
    return dict(
        behavior=behavior,
        datasets=cells,
        regimes=[
            dict(regime="in-distribution", **next(r for r in cells if r["rung"] == "train")),
            regime,
        ],
    )


def run_generic(cfg, behavior, arrays, meta, index, payload, membership):
    """Retain map overlaps, preserving other label/extraction integrity checks."""
    cache, out = Path(cfg.cache), Path(cfg.out) / behavior
    prior = cache / "outputs/analysis" / behavior
    verify_previous(prior)
    old = json.loads((prior / "results.json").read_text())
    directions = np.load(prior / "directions.npz")
    va, vc = directions["answer"], directions["context"]
    path = (
        Path(cfg.repo)
        / "eval_results/issue_1739/wildchat_rung/dv_dataset"
        / behavior
        / "labeling.json"
    )
    assert sha256(path) in old["input_label_sha256"].values()
    labels = json.loads(path.read_text())
    candidates = [
        r for r in ft.json_rows(path) if r["split"] == "eval" and r["rung"] == "wildchat_rung"
    ]
    assert len(candidates) == 2000
    extraction = [r for (ns, _), r in index.items() if ns == behavior + "_extraction"]
    assert len(extraction) == 200
    ex = set().union(*(ft.content_hash_set(r, "exact") for r in extraction))
    no = set().union(*(ft.content_hash_set(r, "normalized") for r in extraction))
    retained, excluded = [], []
    for row in candidates:
        h = ft.lookup_prompt(index, "wildchat", str(row["context_id"]))
        reason = (
            "missing_label"
            if row["dv"] is None
            else "extraction_overlap"
            if ft.overlaps(h, ex, no)
            else None
        )
        if reason:
            excluded.append(dict(context_id=row["context_id"], reason=reason))
        else:
            retained.append(row)
    capture = ft.validate_capture_provenance(
        meta, index, "wildchat", context_ids=[str(r["context_id"]) for r in retained]
    )
    data = ft.reduce_eval(arrays, meta, retained)
    write_json(out / "rollout_alignment.json", data.pop("rollout_audit"))
    scores = np.stack(
        (ft.map_projection(payload, data["x"], va), data["y"] @ va, data["x"] @ vc, data["x"] @ va)
    )
    sample = data["x"][:17]
    predicted = ((sample - np.asarray(payload["xmu"])) / np.asarray(payload["xsd"])) @ np.asarray(
        payload["W"], dtype=np.float64
    ) + np.asarray(payload["ymu"])
    np.testing.assert_allclose(predicted @ va, scores[0, :17], rtol=1e-10, atol=1e-7)
    del data["x"], data["y"]
    assert np.isfinite(scores).all()
    train_overlap = np.asarray(
        [
            ft.overlaps(ft.lookup_prompt(index, "wildchat", str(c)), *membership["train"])
            for c in data["context_ids"]
        ]
    )
    val_overlap = np.asarray(
        [
            ft.overlaps(ft.lookup_prompt(index, "wildchat", str(c)), *membership["validation"])
            for c in data["context_ids"]
        ]
    )
    historical = ft._wc_eval_mask(list(data["context_ids"]))
    # Pairwise parity makes the deliberate membership-policy change explicit.
    parity = []
    for old_path in (prior / "predictions.npz", Path(cfg.small) / behavior / "predictions.npz"):
        previous = dict(np.load(old_path))
        lookup = {c: i for i, c in enumerate(previous["context_ids"])}
        pairs = [(i, lookup[c]) for i, c in enumerate(data["context_ids"]) if c in lookup]
        now, before = np.asarray(pairs).T
        np.testing.assert_array_equal(data["dv"][now], previous["dv"][before])
        np.testing.assert_array_equal(data["groups"][now], previous["groups"][before])
        names = NAMES if old_path == prior / "predictions.npz" else NAMES[1:]
        for name in names:
            np.testing.assert_allclose(
                scores[list(NAMES).index(name), now],
                previous["predictions"][list(previous["arms"]).index(name), before],
                rtol=1e-10,
                atol=1e-7,
            )
        parity.append(dict(source=str(old_path), n_shared=len(now), methods=list(names)))
    np.savez(
        out / "generic_predictions.npz",
        predictions=scores,
        methods=NAMES,
        train_overlap=train_overlap,
        validation_overlap=val_overlap,
        historical_eval=historical,
        **data,
    )
    results = []
    for subset, mask in (
        ("historical_eval_including_overlap", historical),
        ("all_cached_generic_chat", np.ones(len(historical), dtype=bool)),
    ):
        selected = {k: v[mask] for k, v in data.items()}
        cells, draws = summarize(
            selected,
            scores[:, mask],
            seed=1739963 + BEHAVIORS.index(behavior),
            out=Path(cfg.out),
            behavior=behavior + "/" + subset,
            names=NAMES,
        )
        assert len(cells) == 1
        cell = cells[0]
        cell["differences"] = deltas(
            [cell["arms"][n]["rho"] for n in NAMES], draws["wildchat_rung"]
        )
        cell.update(
            subset=subset,
            train_overlap=int(train_overlap[mask].sum()),
            validation_overlap=int(val_overlap[mask].sum()),
            neither_overlap=int((~(train_overlap | val_overlap))[mask].sum()),
        )
        results.append(cell)
        np.savez(out / (subset + "_bootstrap.npz"), **draws)
    small = dict(np.load(Path(cfg.small) / behavior / "predictions.npz"))
    lookup = {cid: i for i, cid in enumerate(small["context_ids"])}
    common = [(i, lookup[c]) for i, c in enumerate(data["context_ids"]) if c in lookup]
    now, before = np.asarray(common).T
    common_map_comparison = dict(
        n=len(now),
        small_rho=float(
            spearmanr(
                small["predictions"][list(small["arms"]).index("mapped_answer"), before],
                data["dv"][now],
            ).statistic
        ),
        million_rho=float(spearmanr(scores[0, now], data["dv"][now]).statistic),
    )
    result = dict(
        behavior=behavior,
        generic=results,
        matched_small_generic=common_map_comparison,
        audit=dict(
            labels_sha256=sha256(path),
            dv_construct=labels["dv_construct"],
            planned_historical=int(
                ft._wc_eval_mask([str(r["context_id"]) for r in candidates]).sum()
            ),
            planned_all=len(candidates),
            excluded=excluded,
            capture=capture,
            parity=parity,
            directions_sha256=sha256(prior / "directions.npz"),
        ),
    )
    write_json(out / "generic_results.json", result)
    progress(Path(cfg.out), "behavior_complete", behavior=behavior)
    return result


@hydra.main(version_base=None, config_path=None, config_name="map_size_diagnostic")
def main(cfg: Config):
    """Score and archive the cached diagnostic separately from held-out claims."""
    out, cache = Path(cfg.out), Path(cfg.cache)
    out.mkdir(parents=True, exist_ok=False)
    prov = as_metadata_dict(git_provenance(ROOT), phase="overlap-diagnostic")
    progress(out, "verify_inputs")
    continuity = verify_input_pins(cache)
    verify_previous(Path(cfg.small))
    verify_previous(Path(cfg.large))
    prior = json.loads((cache / "outputs/analysis/config.json").read_text())
    store = cache / "inputs/wildchat"
    manifest = json.loads((store / "slice_manifest.json").read_text())
    for name, record in manifest["members"].items():
        assert record["sha256"] == prior["store_sha256"][str(store / name)]
        assert sha256(store / name) == record["sha256"]
    hashes = np.load(cache / "outputs/map/map_prompt_hashes.npz")
    assert len(hashes["exact_sha256"]) == 963444 and len(hashes["val_exact_sha256"]) == 400
    membership = {
        "train": (
            set(hashes["exact_sha256"].astype(str)),
            set(hashes["normalized_sha256"].astype(str)),
        ),
        "validation": (
            set(hashes["val_exact_sha256"].astype(str)),
            set(hashes["val_normalized_sha256"].astype(str)),
        ),
    }
    payload = ft.load_payload(cache / "outputs/map/frozen.pt")
    index = ft.load_prompt_index(cache / "inputs/provenance/prompt_index.jsonl")
    arrays, meta = ft.load_store(store)
    summary = dict(
        **prov,
        source_sha=prov["git_commit"],
        input_continuity=continuity,
        map_training_pairs=963444,
        layer=19,
        n_boot=2000,
        methods=list(NAMES),
        scope="Generic chat deliberately includes map-training/validation contexts; diagnostic only, not held-out map generalization",
        comparison_caveat="Map corpus, recipe and answer-token pooling differ; not a controlled training-size effect",
        pooling_caveat=ft.POOLING_CAVEAT,
        behaviors=[],
    )
    write_json(out / "config.json", summary)
    for b in BEHAVIORS:
        (out / b).mkdir()
        progress(out, "matched_heldout", behavior=b)
        matched = matched_heldout(cfg, b)
        write_json(out / b / "matched_results.json", matched)
        generic = run_generic(cfg, b, arrays, meta, index, payload, membership)
        summary["behaviors"].append(
            dict(
                behavior=b, matched=matched, **{k: v for k, v in generic.items() if k != "behavior"}
            )
        )
        write_json(
            out / b / "complete.json",
            dict(artifact_sha256={p.name: sha256(p) for p in (out / b).iterdir() if p.is_file()}),
        )
        gc.collect()
    write_json(out / "summary.json", summary)
    write_json(
        out / "complete.json",
        dict(
            source_sha=prov["git_commit"],
            finished_at=time.time(),
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
