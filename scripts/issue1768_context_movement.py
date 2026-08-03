"""#1768 raw context-movement statistic (H1) — free-analysis follow-up (Step 9a-ter).

Per trained arm (72) x layer (14, 19, 25): the distribution of per-context
relative context movement ||dc||/||c0||, where c0 is the BASE model's
context-span activation for a prompt and c+ the TRAINED arm's for the SAME
prompt. Rows are joined by prompt sha in the BASE store's order (the
`issue1768_fit.load_corpus_cell` join convention — never row order). Fills the
plan criteria-table H1 row ("raw context movement") that the p8 fit phase only
proxied through the decomposition input-movement term.

Streaming discipline (the horse-race CI job's pattern): the 2 base corpus
units' context spans stay resident (fp32, ~1.4 GB); each arm store is staged
via scoped `hf_hub_download` from the pinned data-repo revision, reduced, and
DELETED before the next (peak staging < ~2.2 GB despite ~52 GB transferred).
Per-arm rows checkpoint to `<stage_root>/context_movement_ckpt.jsonl` and are
resumed on re-run keyed on (arm_id, revision) — 72 units > the T2 grain — so a
`--arms`/`--max-arms` smoke's rows are valid production rows (same code, same
pinned inputs) and are deliberately reused.

Output: eval_results/issue_1768/context_movement.json — the per-(arm, layer)
cell table + per-behavior x method aggregates + _meta. Stores are fp16 span
means (capture assumption 11), so the rel-movement resolution floor from fp16
quantization is ~4e-4 — far below the movements of interest.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before numpy/torch: shared-VM thread caps

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1768_fit as F  # noqa: E402

REV = "c07267285d2cdbf3e0401ddc3e3accae50e496a7"  # the body's pinned data-repo revision
LAYERS = (14, 19, 25)
N_ROWS_FLOOR = 15_000  # joined-row floor (the fit JSONs' n_rows read 16,400)
KEEP_FRAC_FLOOR = 0.9  # load_corpus_cell's sha-join coverage floor
OUT_NAME = "context_movement.json"


def _stage(stage_root: Path, rel: str) -> Path:
    """hf_hub_download one file from the pinned revision into stage_root."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    local = stage_root / rel
    if local.exists():
        return local
    hub.retry_transient(
        lambda: hf_hub_download(
            X.HF_DATA_REPO, rel, repo_type="dataset", revision=REV, local_dir=str(stage_root)
        ),
        what=f"corpus store fetch {rel}",
    )
    assert local.exists(), rel
    return local


def _ctx_span(store: dict, layer: int) -> np.ndarray:
    """fp32 context-span rows for one layer (fit's `_rows_from_store` tensor read)."""
    return np.asarray(store["arms"]["context"][layer].float().numpy(), dtype=np.float32)


def _load_base(stage_root: Path, unit: str) -> dict:
    """Resident base-unit context spans: {'shas': [...], 'ctx': {layer: (n, d) fp32}}.

    The pinned val/test rows carry 18 duplicate prompt shas (82 extra rows,
    qidx 15002-16395 — frozen #779 pins, pre-dating the r4 train-draw dedup);
    `load_corpus_cell` tolerates them via its last-wins index dict, so this
    loader mirrors that convention rather than asserting uniqueness.
    """
    store = F._load_store(_stage(stage_root, f"{X.HF_PREFIX}/corpus_capture/{unit}/pooled.pt"))
    return {
        "shas": list(store["row_sha"]),
        "ctx": {layer: _ctx_span(store, layer) for layer in LAYERS},
    }


def _arm_stats(base: dict, store: dict, arm_id: str) -> dict:
    """Per-layer ||dc||/||c0|| distribution stats for one arm, sha-joined to base.

    Join convention = `issue1768_fit.load_corpus_cell`: intersect on sha in the
    BASE store's order; a duplicated sha maps every base occurrence to the
    arm's LAST row for that sha (same prompt, greedy — near-identical rows).
    """
    a_shas = list(store["row_sha"])
    a_ix = {s: i for i, s in enumerate(a_shas)}
    b_shas = base["shas"]
    keep = [i for i, s in enumerate(b_shas) if s in a_ix]
    assert len(keep) >= KEEP_FRAC_FLOOR * len(b_shas), (arm_id, len(keep), len(b_shas))
    assert len(keep) > N_ROWS_FLOOR, (arm_id, len(keep))
    b = np.asarray(keep)
    p = np.asarray([a_ix[b_shas[i]] for i in keep])
    # sha-join assert: the joined rows carry identical shas on both sides
    assert [b_shas[i] for i in keep] == [a_shas[j] for j in p], f"{arm_id}: sha join broke"
    layers: dict[str, dict] = {}
    for layer in LAYERS:
        c0 = base["ctx"][layer][b]
        cp = _ctx_span(store, layer)[p]
        assert c0.shape == cp.shape == (len(keep), X.HIDDEN), (arm_id, layer, c0.shape, cp.shape)
        dn = np.linalg.norm(cp - c0, axis=1).astype(np.float64)
        bn = np.linalg.norm(c0, axis=1).astype(np.float64)
        assert (bn > 0).all(), (arm_id, layer, "zero base-context norm")
        rel = dn / bn
        layers[str(layer)] = {
            "n_rows": int(len(keep)),
            "rel_median": float(np.median(rel)),
            "rel_q25": float(np.quantile(rel, 0.25)),
            "rel_q75": float(np.quantile(rel, 0.75)),
            "rel_mean": float(rel.mean()),
            "abs_delta_norm_median": float(np.median(dn)),
            "base_norm_median": float(np.median(bn)),
        }
    return layers


def _agg_over(rows: list[dict]) -> dict:
    """Per-layer {n_arms, median/min/max of the per-arm rel medians}."""
    out: dict[str, dict] = {}
    for layer in LAYERS:
        meds = [r["layers"][str(layer)]["rel_median"] for r in rows]
        out[str(layer)] = {
            "n_arms": len(meds),
            "median_of_arm_medians": float(np.median(meds)),
            "min_arm_median": float(np.min(meds)),
            "max_arm_median": float(np.max(meds)),
        }
    return out


def _aggregates(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(f"{r['beh_key']}|{r['method']}", []).append(r)
    return {
        "by_behavior_method": {k: _agg_over(v) for k, v in sorted(by.items())},
        "overall": _agg_over(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="#1768 H1 raw context movement ||dc||/||c0||")
    ap.add_argument("--stage-root", type=Path, default=Path("data/issue_1768/hf_dl"))
    ap.add_argument("--results-dir", type=Path, default=Path("eval_results/issue_1768"))
    ap.add_argument("--max-arms", type=int, default=None, help="smoke: only the first N arms")
    ap.add_argument("--arms", type=str, default=None, help="smoke: comma-separated arm ids")
    args = ap.parse_args()
    stage_root: Path = args.stage_root
    stage_root.mkdir(parents=True, exist_ok=True)

    reg = json.loads(_stage(stage_root, f"{X.HF_PREFIX}/arm_registry.json").read_text())
    arms = reg["arms"]
    assert len(arms) == 72 and len({a["arm_id"] for a in arms}) == 72, len(arms)
    assert {a["arm_id"] for a in arms} == set(reg["in_scope"]), "registry arms != in_scope"
    if args.arms:
        want = set(args.arms.split(","))
        arms = [a for a in arms if a["arm_id"] in want]
        assert {a["arm_id"] for a in arms} == want, f"unknown arm ids: {want}"
    if args.max_arms:
        arms = arms[: args.max_arms]

    ckpt_path = stage_root / "context_movement_ckpt.jsonl"
    done: dict[str, dict] = {}
    if ckpt_path.exists():
        for line in ckpt_path.open(encoding="utf-8"):  # text-mode iteration, never splitlines
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("revision") == REV and set(row["layers"]) == {str(x) for x in LAYERS}:
                done[row["arm_id"]] = row

    bases = {unit: _load_base(stage_root, unit) for unit in X.BASE_UNITS}
    t0 = time.time()
    results: list[dict] = []
    for k, a in enumerate(arms, 1):
        arm_id = a["arm_id"]
        if arm_id in done:
            results.append(done[arm_id])
            print(f"[ctxmove] unit {k}/{len(arms)} {arm_id} resumed", flush=True)
            continue
        path = _stage(stage_root, f"{X.HF_PREFIX}/corpus_capture/{arm_id}/pooled.pt")
        store = F._load_store(path)
        layers = _arm_stats(bases[X.base_unit_for(arm_id)], store, arm_id)
        del store
        os.remove(path)  # delete-after-use: peak staging stays < ~2.2 GB
        fit_json = args.results_dir / "fits" / f"{arm_id}_L19.json"
        if fit_json.exists():  # loader parity vs the committed p8 fit (brief requirement)
            fit_n = json.loads(fit_json.read_text())["n_rows"]
            assert layers["19"]["n_rows"] == fit_n, (arm_id, layers["19"]["n_rows"], fit_n)
        row = {
            "arm_id": arm_id,
            "beh_key": a["beh_key"],
            "ctx_key": a["ctx_key"],
            "regime": a["regime"],
            "seed": a["seed"],
            "method": a["method"],
            "revision": REV,
            "layers": layers,
        }
        with ckpt_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        results.append(row)
        print(
            f"[ctxmove] unit {k}/{len(arms)} {arm_id} elapsed={time.time() - t0:.0f}s", flush=True
        )

    cells = {
        f"{r['arm_id']}_L{layer}": {
            "arm_id": r["arm_id"],
            "beh_key": r["beh_key"],
            "ctx_key": r["ctx_key"],
            "regime": r["regime"],
            "seed": r["seed"],
            "method": r["method"],
            "layer": layer,
            **r["layers"][str(layer)],
        }
        for r in results
        for layer in LAYERS
    }
    out = {
        "cells": cells,
        "aggregates": _aggregates(results),
        "_meta": {
            **F._meta(),
            "script": "scripts/issue1768_context_movement.py",
            "hf_revision": REV,
            "n_arms": len(results),
            "layers": list(LAYERS),
            "statistic": "per-context ||c_plus - c_base|| / ||c_base||, context span, sha-joined",
            "storage_note": "pooled stores are fp16 span means; fp16 rel floor ~4e-4",
        },
    }
    F._atomic_json(args.results_dir / OUT_NAME, out)
    print(json.dumps(out["aggregates"], indent=1))
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit: heavy C-extension atexit race (gotchas.md)


if __name__ == "__main__":
    main()
