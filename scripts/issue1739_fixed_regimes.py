"""Extend frozen #1739 contrastive projections to the previous real-data regimes.

Uses cached activations and labels only; no map/readout fitting or generation.
The historical `train` rung is held out from the frozen map and extraction,
although it trained the separate, supplementary behavior regressions.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402

from explore_persona_space.experiments.issue_1739 import arms  # noqa: E402
from scripts import issue1739_fixed_transfer as ft  # noqa: E402
from scripts.issue1739_claim4_fold import group_bootstrap_rhos  # noqa: E402
from scripts.issue1739_covariance_ablation import sha256, write_json  # noqa: E402
from scripts.issue1739_million_cached import verify_previous  # noqa: E402

NAMES = ("mapped_answer", "real_answer", "context_native")
BEHAVIORS = ("evil", "sycophancy", "hallucination")
N_BOOT = 2000  # Same conditional group-bootstrap protocol as fixed_transfer.


@dataclass
class Config:
    """Paths only: statistical choices remain frozen to the preceding analysis."""

    cache: str = "/dev/shm/issue1739-fixed-transfer"
    repo: str = "/home/thomasjiralerspong/explore-persona-space"
    out: str = "/mnt/eps-data/thomasjiralerspong/issue1739-fixed-regimes-20260917"


ConfigStore.instance().store(name="fixed_regimes", node=Config)


def verify_input_pins(cache):
    """Bind reused map, membership, labels, and captures to the preceding run."""
    prior_path = cache / "outputs/analysis/config.json"
    prior = json.loads(prior_path.read_text())
    manifest_path = cache / "outputs/map/map_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pins = {
        manifest_path: prior["map_manifest_sha256"],
        cache / "inputs/provenance/prompt_index.jsonl": prior["prompt_index_sha256"],
        cache / "inputs/provenance/hallucination_per_rollout.json": prior[
            "hallu_per_rollout_sha256"
        ],
        Path(manifest["map_prompt_hashes"]): manifest["map_prompt_hashes_sha256"],
        cache / "outputs/map/frozen.pt": prior["map_payload_sha256"][
            str(cache / "outputs/map/frozen.pt")
        ],
    }
    for path, expected in pins.items():
        assert sha256(path) == expected, str(path)
    counts = {}
    for behavior in BEHAVIORS:
        store = cache / "inputs" / f"{behavior}_labeling"
        members = json.loads((store / "slice_manifest.json").read_text())["members"]
        for name, member in members.items():
            assert member["sha256"] == prior["store_sha256"][str(store / name)], name
        counts[behavior] = len(members)
    return dict(
        prior_config_sha256=sha256(prior_path),
        checked_sha256={str(p): v for p, v in pins.items()},
        slice_members_bound_to_prior_run=counts,
        verified_at=time.time(),
    )


def progress(out, phase, **extra):
    """Publish timestamped stage evidence for active supervision."""
    value = dict(time=time.time(), phase=phase, **extra)
    write_json(out / "progress.json", value)
    print(json.dumps(value), flush=True)


def estimates(rho, boot):
    """Keep undefined correlations null; never render them as zero."""
    result = {}
    for i, name in enumerate(NAMES):
        valid = np.isfinite(boot[i])
        result[name] = dict(
            rho=float(rho[i]) if np.isfinite(rho[i]) else None,
            ci95=np.quantile(boot[i, valid], [0.025, 0.975]).tolist() if valid.any() else None,
            valid_bootstrap_draws=int(valid.sum()),
        )
    return result


def summarize(data, pred, *, seed, out, behavior):
    """Bootstrap within datasets, pairing arms and independently sampling rungs."""
    cells, draws = [], {}
    for i, rung in enumerate(sorted(set(data["rungs"]))):
        ix = np.flatnonzero(data["rungs"] == rung)
        rho = arms.spearman_rows(pred[:, ix], data["dv"][ix])
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        batches = []
        for start in range(0, N_BOOT, 100):
            batch, n_groups = group_bootstrap_rhos(
                pred[:, ix], data["dv"][ix], data["groups"][ix], n_boot=100, rng=rng
            )
            batches.append(batch)
            progress(out, "bootstrap", behavior=behavior, rung=str(rung), draws=start + 100)
        boot = np.concatenate(batches, axis=1)
        draws[str(rung)] = boot
        cells.append(
            dict(
                rung=str(rung),
                n=len(ix),
                n_groups=n_groups,
                informative=len(ix) >= 30,  # Retain the preceding analysis's reporting gate.
                dv_unique=len(np.unique(data["dv"][ix])),
                floor_mass=float(np.mean(data["dv"][ix] == 0)),
                arms=estimates(rho, boot),
            )
        )
    return cells, draws


def run_behavior(cfg, behavior, index, map_exact, map_norm, payload):
    """Add in-distribution scores without changing any existing heldout score."""
    cache, root = Path(cfg.cache), Path(cfg.out)
    out = root / behavior
    out.mkdir()
    previous = cache / "outputs/analysis" / behavior
    verify_previous(previous)
    old = dict(np.load(previous / "predictions.npz"))
    directions = np.load(previous / "directions.npz")
    label_path = Path(cfg.repo) / "eval_results/issue_1739/dv_dataset" / behavior / "labeling.json"
    old_result = json.loads((previous / "results.json").read_text())
    assert sha256(label_path) in old_result["input_label_sha256"].values()

    extraction = [r for (ns, _), r in index.items() if ns == f"{behavior}_extraction"]
    extraction = [r for r in extraction if not ft.overlaps(r, map_exact, map_norm)]
    assert len(extraction) == 200
    ex_exact = set().union(*(ft.content_hash_set(r, "exact") for r in extraction))
    ex_norm = set().union(*(ft.content_hash_set(r, "normalized") for r in extraction))
    selected = [r for r in ft.json_rows(label_path) if r["split"] == r["rung"] == "train"]
    retained, excluded = [], []
    for row in selected:
        cid = str(row["context_id"])
        h = ft.lookup_prompt(index, f"{behavior}_labeling", cid)
        reason = None
        if row["dv"] is None:
            reason = "no_valid_behavior_label"
        elif ft.overlaps(h, map_exact, map_norm):
            reason = "overlap_map_train_or_validation"
        elif ft.overlaps(h, ex_exact, ex_norm):
            reason = "overlap_direction_extraction"
        if reason:
            excluded.append(dict(context_id=cid, reason=reason))
        else:
            retained.append(row)
    progress(root, "verify_store", behavior=behavior, planned=len(selected), retained=len(retained))
    store = cache / "inputs" / f"{behavior}_labeling"
    manifest = json.loads((store / "slice_manifest.json").read_text())
    for name, expected in manifest["members"].items():
        assert sha256(store / name) == expected["sha256"], name
    arrays, meta = ft.load_store(store)
    provenance = ft.validate_capture_provenance(
        meta, index, f"{behavior}_labeling", context_ids=[str(r["context_id"]) for r in retained]
    )
    supplement = None
    if behavior == "hallucination":
        supplement = {
            str(r["context_id"]): r
            for r in ft.json_rows(cache / "inputs/provenance/hallucination_per_rollout.json")
        }
    data = ft.reduce_eval(arrays, meta, retained, supplementary=supplement)
    del arrays, meta
    gc.collect()
    va, vc = directions["answer"], directions["context"]
    pred = np.stack((ft.map_projection(payload, data["x"], va), data["y"] @ va, data["x"] @ vc))
    # Check the algebra against explicitly formed mapped answers on real rows.
    sample = data["x"][:17]
    mapped = ((sample - np.asarray(payload["xmu"])) / np.asarray(payload["xsd"])) @ np.asarray(
        payload["W"], dtype=np.float64
    ) + np.asarray(payload["ymu"])
    np.testing.assert_allclose(mapped @ va, pred[0, :17], rtol=1e-10, atol=1e-7)
    write_json(out / "rollout_alignment.json", data.pop("rollout_audit"))
    del data["x"], data["y"]
    old_indices = [list(old["arms"]).index(n) for n in NAMES]
    predictions = np.concatenate((pred, old["predictions"][old_indices]), axis=1)
    combined = {k: np.concatenate((data[k], old[k])) for k in data}
    assert np.isfinite(predictions).all()
    assert len(set(combined["context_ids"])) == len(combined["context_ids"])
    np.savez(out / "predictions.npz", predictions=predictions, arms=NAMES, **combined)
    audit = dict(
        planned=len(selected),
        retained=len(retained),
        exclusions=excluded,
        exclusion_counts=dict(Counter(r["reason"] for r in excluded)),
        capture_provenance=provenance,
        slice_manifest_sha256=sha256(store / "slice_manifest.json"),
        labels_sha256=sha256(label_path),
        directions_sha256=sha256(previous / "directions.npz"),
        existing_predictions_sha256=sha256(previous / "predictions.npz"),
        heldout_scores_preserved_exactly=True,
    )
    write_json(out / "audit.json", audit)
    cells, draws = summarize(
        combined, predictions, seed=1739963 + BEHAVIORS.index(behavior), out=root, behavior=behavior
    )
    np.savez(out / "bootstraps.npz", **draws)
    lookup = {r["rung"]: r for r in cells}
    ood = [r for r in ft.ROSTER[behavior] if r != "wildchat_rung"]
    ood_rho = np.mean([[lookup[r]["arms"][n]["rho"] for n in NAMES] for r in ood], axis=0)
    ood_boot = np.mean([draws[r] for r in ood], axis=0)
    regimes = [
        dict(regime="generic chat", datasets=["wildchat_rung"], **lookup["wildchat_rung"]),
        dict(regime="in-distribution", datasets=["train"], **lookup["train"]),
        dict(
            regime="OOD",
            datasets=ood,
            n=sum(lookup[r]["n"] for r in ood),
            informative=True,
            arms=estimates(ood_rho, ood_boot),
        ),
    ]
    write_json(
        out / "results.json", dict(behavior=behavior, datasets=cells, regimes=regimes, audit=audit)
    )
    write_json(
        out / "complete.json",
        dict(
            finished_at=time.time(),
            artifact_sha256={p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        ),
    )
    progress(root, "behavior_complete", behavior=behavior)
    return dict(behavior=behavior, datasets=cells, regimes=regimes)


@hydra.main(version_base=None, config_path=None, config_name="fixed_regimes")
def main(cfg: Config):
    """Run the three cached CPU extensions in a fresh, durable output directory."""
    cache, out = Path(cfg.cache), Path(cfg.out)
    out.mkdir(parents=True, exist_ok=False)
    input_continuity = verify_input_pins(cache)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest_path = cache / "outputs/map/map_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    map_path = cache / "outputs/map/frozen.pt"
    assert sha256(map_path) == manifest["artifacts_sha256"]["frozen"]["sha256"]
    progress(out, "load_membership")
    exact, normalized, hash_audit = ft.load_map_hashes(manifest, manifest_path.parent)
    index_path = cache / "inputs/provenance/prompt_index.jsonl"
    index = ft.load_prompt_index(index_path)
    payload = ft.load_payload(map_path)
    summary = dict(
        source_sha=source_sha,
        input_continuity=input_continuity,
        script_sha256=sha256(Path(__file__)),
        map_sha256=sha256(map_path),
        map_training_pairs=manifest["n_train"],
        layer=19,
        membership=hash_audit,
        prompt_index_sha256=sha256(index_path),
        n_boot=N_BOOT,
        excluded_evaluation="pvsynth",
        ood_aggregation="equal-weight mean of dataset Spearman correlations; independent dataset group resamples, paired methods",
        directions="frozen unfiltered positive-minus-negative instruction contrasts",
        pooling_caveat=ft.POOLING_CAVEAT,
        in_distribution="historical readout-training rung, held out from map fitting and direction extraction; no readout fitted here",
    )
    write_json(out / "config.json", summary)
    summary["behaviors"] = [
        run_behavior(cfg, b, index, exact, normalized, payload) for b in BEHAVIORS
    ]
    write_json(out / "summary.json", summary)
    write_json(
        out / "complete.json",
        dict(
            finished_at=time.time(),
            source_sha=source_sha,
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
