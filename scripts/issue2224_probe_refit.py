#!/usr/bin/env python3
"""Issue #2224: re-derive DEPLOYABLE Form-A ridge probes from #2222's persisted pool.

Why this exists (P1-gate check 4 GAP, carried concern
``unit1-sibling-artifact-contracts``): #2222 never persisted probe WEIGHT
vectors — its ``eval_results/issue_2222/form_a_probe.json`` is metrics/records
only, its ``form_a/`` dir holds judge caches, and its HF prefix carries no
probe npz. Its probe was EXPLORATORY (LOFO per-family folds, dof-capped GCV
ridge). Per artifact-reuse.md item (c) (missing fields -> re-derive, never
consume silently), this driver re-fits a deployable probe from #2222's
PERSISTED inputs, using #2222's own fit cores vendored at the pinned SHA
(``scripts/issue2224_vendored_ridge.py`` @ 99f9e975b08311684dd8f7ca6085e6a6b6791339).

Inputs (HF data repo, all staged under ``--data-root``; a SINGLE repo revision
is resolved once per run and threaded through every download — paired hub
files fetched at ``revision=None`` can split across snapshots, #2061):

- labels: ``issue2222_pvscreen/raw_completions/form_a_judge/judge_merged_{trait}.json``
  (3 traits; ``per_item[item_id] = {scores, mean, rate_gt_50, ...}``; items
  with zero kept judge draws carry ``mean: None`` — dropped, never coerced);
- activations: ``issue2222_pvscreen/analysis_tensors/capture/<dataset>/
  {summaries.npz, base_respavg.npz}`` (24 datasets; fp16 (n, 28, 3584)).

X/y assembly replicates ``issue2222_judge.stage_probe`` +
``_load_pool_activations`` at the pinned SHA EXACTLY:

- the judge pool is re-derived from the merged-label ``per_item`` keys
  (``{dataset}-r{row_id}``) — identical to #2222's ``pool.json`` row set,
  since every pool row was judged;
- X = ``raw_respavg`` (the judged dataset response's OWN activation; #2222's
  recorded ``plan_ambiguity_resolution`` — NOT ``base_respavg``), fp16 ->
  float32, restricted to pool rows INTERSECTED with the ``base_respavg.npz``
  row ids (pool rows lacking a base capture are excluded and counted —
  ``n_missing_base``, 0 in #2222's realized fit);
- y = per-item mean judge scores, one column per trait in ``TRAITS`` order;
  rows with ANY trait ``mean: None`` are dropped (#2222 realized: n=11,984
  kept, 16 dropped, d=3,584 — well-posed n > d). The realized counts are
  ASSERTED against the pinned reference JSON, so drifted HF inputs fail loud.

Fit regime is read PROGRAMMATICALLY from the pinned reference
(``git show <pin>:eval_results/issue_2222/form_a_probe.json`` ``.fit_regime``:
17-point log-spaced lambda grid 1e-2..1e6, dof_cap 0.9), never hardcoded.

``--parity-check`` re-runs the FULL LOFO battery (``dof_capped_ridge_multi_y``,
8 family folds) at each ``--parity-layers`` layer at #2222's exact regime and
compares the pooled held-out R^2 per trait against the reference
``heldout_r2_per_layer``; |delta R^2| > ``--parity-tol`` (0.02) raises with
both values printed. Runs BEFORE the deployable fits (pod-side gate).

Deployable fits: FULL-POOL ``dof_capped_ridge_fit_all`` per distinct layer
(multi-target — one eigh serves all 3 traits at that layer; vectorize-first),
for two layer regimes:

- ``steer``  (PRIMARY): evil=19, sycophancy=19, hallucination=15 (#2222
  ``STEER_IDX`` — matches the #2224 score-phase read-out);
- ``argmax`` (labeled companion): evil=14, sycophancy=18, hallucination=18
  (argmax of the reference ``heldout_r2_per_layer``; re-derived from the
  reference at runtime and asserted against the pinned constants).

Output npz per (regime, trait) — ``<out-root>/{steer,argmax}/<trait>.npz`` —
round-trips through ``issue2224_predictor_scores.load_probe`` (verified
in-process at write time): keys ``w`` (3584,) float64, ``b`` scalar, ``x_mu``
(3584,) = training column means, ``x_sd`` (3584,) = ONES, ``layer`` int,
``meta`` json string. Intercept representation: #2222's fit centers X but does
NOT scale it, folding the centering into ``b0 = y_mu - x_mu @ w``; storing
``x_mu`` = training means, ``x_sd`` = 1, ``b`` = ``y_mu[trait]`` makes the
consumer's ``probe_score`` (``((x - x_mu)/x_sd) @ w + b``) algebraically equal
to the fit's ``ridge_predict`` (``x @ w + b0``) while satisfying the loader's
x_mu/x_sd both-or-neither contract.

``--upload`` pushes the out-root (npz + meta JSON) to
``issue2224_screening/analysis_tensors/form_a_probe_refit/{steer,argmax}/`` in
ONE bulk fail-loud ``upload_folder`` commit with an exact-set scoped verify
(the sibling ``issue2224_gen_natural.upload_corpus_dir`` shape).

Runs POD-SIDE (cpu-bigmem): staging is ~15-20 GB of capture npz. Recommended
pod invocation:

    uv run python scripts/issue2224_probe_refit.py --parity-check --upload
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import issue2224_common as common  # noqa: F401  (sys.path shim: src/ + scripts/)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/torch imports: shared-VM thread caps + HF token (#847)

import numpy as np  # noqa: E402

import issue2224_vendored_ridge as ridge  # noqa: E402

# --- Pins (P1-gate check 4, 2026-08-11) ------------------------------------------

ISSUE2222_PIN_SHA = "99f9e975b08311684dd8f7ca6085e6a6b6791339"  # origin/issue-2222
REFERENCE_REPO_PATH = "eval_results/issue_2222/form_a_probe.json"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_LABELS_PREFIX = "issue2222_pvscreen/raw_completions/form_a_judge"
HF_CAPTURE_PREFIX = "issue2222_pvscreen/analysis_tensors/capture"
HF_OUT_PREFIX = "issue2224_screening/analysis_tensors/form_a_probe_refit"

TRAITS = ("evil", "sycophancy", "hallucination")  # #2222 lib.TRAITS — y column order
N_LAYERS, DIM = 28, 3584  # #2222 lib.RB_SHAPE
# steer: #2222 reduce.STEER_IDX (matches the #2224 score-phase read-out layers).
# argmax: argmax_layer of the reference heldout_r2_per_layer (asserted at runtime).
REGIME_LAYERS: dict[str, dict[str, int]] = {
    "steer": {"evil": 19, "sycophancy": 19, "hallucination": 15},
    "argmax": {"evil": 14, "sycophancy": 18, "hallucination": 18},
}

DATA_ROOT_DEFAULT = common.PROJECT_ROOT / "data" / "issue_2224" / "probe_refit_dl"
OUT_ROOT_DEFAULT = common.PROJECT_ROOT / "data" / "issue_2224" / "probe_refit" / "probes"


def _log(msg: str) -> None:
    print(f"[probe-refit] {msg}", flush=True)


# --- Reference (pinned #2222 record) ----------------------------------------------


def load_reference(reference_json: str | None) -> dict:
    """The pinned form_a_probe.json — via ``git show`` at the pin, or an explicit path."""
    if reference_json:
        payload = json.loads(Path(reference_json).read_text())
    else:
        proc = subprocess.run(
            ["git", "show", f"{ISSUE2222_PIN_SHA}:{REFERENCE_REPO_PATH}"],
            cwd=common.PROJECT_ROOT,
            env={**os.environ},
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git show {ISSUE2222_PIN_SHA[:10]}:{REFERENCE_REPO_PATH} failed "
                f"(rc={proc.returncode}): {proc.stderr.strip()[:300]} — run "
                f"`git fetch origin issue-2222` in this checkout, or pass "
                f"--reference-json <path to the pinned form_a_probe.json>"
            )
        payload = json.loads(proc.stdout)
    fr = payload["fit_regime"]
    lambdas = np.asarray(fr["lambda_grid"], dtype=np.float64)
    if lambdas.shape != (17,) or np.any(lambdas <= 0) or np.any(np.diff(lambdas) <= 0):
        raise RuntimeError(
            f"reference lambda_grid unexpected: shape={lambdas.shape} — expected the "
            f"17-point strictly-positive ascending grid recorded by #2222"
        )
    if int(fr["d"]) != DIM:
        raise RuntimeError(f"reference d={fr['d']} != pinned DIM={DIM}")
    return payload


def assert_argmax_layers(reference: dict) -> None:
    """Pinned argmax-regime layers must equal argmax of the reference per-layer R^2."""
    per_layer = reference["heldout_r2_per_layer"]
    for trait in TRAITS:
        vals = per_layer[trait]
        if len(vals) != N_LAYERS or any(v is None for v in vals):
            raise RuntimeError(
                f"reference heldout_r2_per_layer[{trait}] incomplete "
                f"({sum(v is not None for v in vals)}/{N_LAYERS} layers) — cannot "
                f"ground the argmax regime"
            )
        got = int(np.argmax(np.asarray(vals, dtype=np.float64)))
        want = REGIME_LAYERS["argmax"][trait]
        if got != want:
            raise RuntimeError(
                f"argmax layer drift for {trait}: reference argmax={got} != pinned {want}"
            )


# --- Staging (single resolved revision, #2061) -------------------------------------


def resolve_data_revision() -> str:
    """Pin the data-repo main -> sha ONCE per run (paired-file snapshot split, #2061)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    info = hub.retry_transient(
        lambda: HfApi().repo_info(HF_DATA_REPO, repo_type="dataset"),
        what=f"repo_info({HF_DATA_REPO})",
    )
    sha = info.sha
    if not sha:
        raise RuntimeError(f"could not resolve a revision sha for {HF_DATA_REPO}")
    return str(sha)


def stage_file(path_in_repo: str, local: Path, revision: str) -> Path:
    """Local-first staged copy of one data-repo file (atomic, retried, fail-loud)."""
    if local.exists():
        return local
    from explore_persona_space.orchestrate import hub

    _log(f"staging {path_in_repo} -> {local}")
    return Path(
        hub.stage_hub_file(
            HF_DATA_REPO, path_in_repo, local, repo_type="dataset", revision=revision
        )
    )


def stage_labels(data_root: Path, revision: str) -> dict[str, dict]:
    """{trait: merged-judge payload}; item-id key sets asserted identical across traits."""
    out: dict[str, dict] = {}
    for trait in TRAITS:
        local = stage_file(
            f"{HF_LABELS_PREFIX}/judge_merged_{trait}.json",
            data_root / "labels" / f"judge_merged_{trait}.json",
            revision,
        )
        payload = json.loads(local.read_text())
        if "per_item" not in payload or not payload["per_item"]:
            raise RuntimeError(f"{local}: no per_item records — wrong/empty merged-label file")
        out[trait] = payload
    keys0 = set(out[TRAITS[0]]["per_item"])
    for trait in TRAITS[1:]:
        if set(out[trait]["per_item"]) != keys0:
            raise RuntimeError(
                f"merged-label item sets differ across traits ({trait} vs {TRAITS[0]}) — "
                f"the three files must cover the SAME judge pool"
            )
    return out


# --- Pool + X/y assembly (replicates #2222 stage_probe at the pin) ------------------


def split_item_id(iid: str) -> tuple[str, int]:
    """``{dataset}-r{row_id}`` -> (dataset, row_id); #2222 judge item-id grammar."""
    ds, sep, r = iid.rpartition("-r")
    if not sep or not ds or not r.isdigit():
        raise ValueError(f"malformed #2222 item id: {iid!r}")
    return ds, int(r)


def derive_pool(item_ids: list[str]) -> dict[str, list[int]]:
    """Per-dataset sorted row ids from merged-label item ids; 24-dataset set asserted."""
    per_ds: dict[str, list[int]] = {}
    for iid in item_ids:
        ds, rid = split_item_id(iid)
        per_ds.setdefault(ds, []).append(rid)
    import issue778_lib as i778

    expected = {f"{fam}_{ver}" for fam in i778.FAMILIES for ver in i778.VERSIONS}
    if set(per_ds) != expected:
        raise RuntimeError(
            f"judged pool datasets != the 24 #778 cells: missing={sorted(expected - set(per_ds))} "
            f"unexpected={sorted(set(per_ds) - expected)}"
        )
    return {ds: sorted(rids) for ds, rids in per_ds.items()}


def build_y(item_ids: list[str], labels: dict[str, dict]) -> np.ndarray:
    """(n, 3) mean judge scores in TRAITS order; ``mean: None`` -> NaN (drop-never-coerce)."""
    y = np.full((len(item_ids), len(TRAITS)), np.nan)
    for ti, trait in enumerate(TRAITS):
        per_item = labels[trait]["per_item"]
        y[:, ti] = [
            np.nan if per_item[iid]["mean"] is None else float(per_item[iid]["mean"])
            for iid in item_ids
        ]
    return y


def assemble_xy(data_root: Path, labels: dict[str, dict], layers: list[int], revision: str) -> dict:
    """X (kept, K, DIM) fp16 at the K requested layers + y (kept, 3) + family fold ids.

    Mirrors #2222 ``_load_pool_activations`` (pool rows ∩ base-capture row ids,
    ``np.intersect1d`` ascending order) + ``stage_probe``'s joint all-trait keep
    mask. Only ``raw_respavg`` is materialized (base_respavg contributes its
    row-id set only — the probe never consumes base activations).
    """
    import issue778_lib as i778

    pool = derive_pool(sorted(labels[TRAITS[0]]["per_item"].keys()))
    families = sorted({i778.split_cell_tag(ds)[0] for ds in pool})
    lsel = np.asarray(sorted(set(layers)), dtype=np.int64)
    if lsel.min() < 0 or lsel.max() >= N_LAYERS:
        raise ValueError(f"requested layers {lsel.tolist()} outside [0, {N_LAYERS})")
    xs: list[np.ndarray] = []
    iids: list[str] = []
    fams: list[int] = []
    n_missing_base = 0
    t0 = time.time()
    for di, ds in enumerate(sorted(pool)):
        want = np.asarray(pool[ds], dtype=np.int64)
        summ = stage_file(
            f"{HF_CAPTURE_PREFIX}/{ds}/summaries.npz",
            _cap(data_root, ds, "summaries.npz"),
            revision,
        )
        base = stage_file(
            f"{HF_CAPTURE_PREFIX}/{ds}/base_respavg.npz",
            _cap(data_root, ds, "base_respavg.npz"),
            revision,
        )
        with np.load(summ) as z:
            ids = np.asarray(z["row_ids"], dtype=np.int64)
            sel = np.flatnonzero(np.isin(ids, want))
            if len(sel) != len(want):
                raise RuntimeError(
                    f"{ds}: {len(want) - len(sel)} judged pool rows absent from the capture "
                    f"store — pool was drawn FROM captured ids, so the staged summaries.npz "
                    f"does not match #2222's realized capture"
                )
            raw = np.asarray(z["raw_respavg"])[sel][:, lsel, :]
            assert raw.shape == (len(sel), len(lsel), DIM), raw.shape
            sel_ids = ids[sel]
        with np.load(base) as z:
            bids = np.asarray(z["row_ids"], dtype=np.int64)
        common_ids, ia, _ib = np.intersect1d(sel_ids, bids, return_indices=True)
        n_missing_base += len(sel_ids) - len(common_ids)
        xs.append(raw[ia])
        iids.extend(f"{ds}-r{int(r)}" for r in common_ids)
        fams.extend([families.index(i778.split_cell_tag(ds)[0])] * len(common_ids))
        _log(
            f"assemble {di + 1}/{len(pool)} {ds} rows={len(common_ids)} elapsed={time.time() - t0:.0f}s"
        )
    x = np.concatenate(xs, axis=0)
    y = build_y(iids, labels)
    keep = ~np.isnan(y).any(axis=1)
    kept_iids = [iid for iid, k in zip(iids, keep, strict=True) if k]
    return {
        "x": x[keep],
        "y": y[keep],
        "fam": np.asarray(fams, dtype=np.int64)[keep],
        "item_ids": kept_iids,
        "layer_col": {int(layer): k for k, layer in enumerate(lsel)},
        "n_rows": int(keep.sum()),
        "n_dropped": int((~keep).sum()),
        "n_missing_base": int(n_missing_base),
        "families": families,
        "label_fingerprints": {t: labels[t].get("fingerprint") for t in TRAITS},
    }


def _cap(data_root: Path, ds: str, fname: str) -> Path:
    return data_root / "capture" / ds / fname


def assert_pool_matches_reference(xy: dict, reference: dict) -> None:
    """The re-assembled pool must equal #2222's realized fit regime exactly."""
    fr = reference["fit_regime"]
    checks = (
        ("n_rows", xy["n_rows"], int(fr["n_rows"])),
        (
            "n_dropped_rows_zero_kept_draws",
            xy["n_dropped"],
            int(fr["n_dropped_rows_zero_kept_draws"]),
        ),
        ("n_missing_base_capture", xy["n_missing_base"], int(fr["n_missing_base_capture"])),
    )
    for name, got, want in checks:
        if got != want:
            raise RuntimeError(
                f"pool mismatch vs pinned reference: {name} refit={got} != #2222={want} — "
                f"staged labels/captures are not the artifacts #2222 fit on"
            )
    assert xy["x"].shape[0] == xy["n_rows"] and xy["x"].shape[2] == DIM, xy["x"].shape


# --- Parity gate (LOFO at #2222's exact regime) -------------------------------------


def run_parity(xy: dict, reference: dict, layers: list[int], tol: float, device: str) -> dict:
    """Full-LOFO pooled held-out R^2 at each layer vs the reference; raise on |delta|>tol."""
    fr = reference["fit_regime"]
    lambdas = np.asarray(fr["lambda_grid"], dtype=np.float64)
    dof_cap = float(fr["dof_cap"])
    ref_r2 = reference["heldout_r2_per_layer"]
    results: dict[str, dict] = {}
    for layer in layers:
        t0 = time.time()
        x_l = xy["x"][:, xy["layer_col"][layer], :].astype(np.float32)  # fp16->fp32, as #2222
        res = ridge.dof_capped_ridge_multi_y(
            x_l, xy["y"], xy["fam"], lambdas=lambdas, dof_cap=dof_cap, device=device
        )
        for ti, trait in enumerate(TRAITS):
            got = float(res["heldout_r2"][ti])
            want = ref_r2[trait][layer]
            if want is None:
                raise RuntimeError(f"reference heldout R^2 missing at layer {layer} ({trait})")
            want = float(want)
            delta = abs(got - want)
            _log(
                f"parity layer={layer} trait={trait} refit_r2={got:.6f} "
                f"ref_r2={want:.6f} |delta|={delta:.6f}"
            )
            if delta > tol:
                raise RuntimeError(
                    f"PARITY FAIL at layer {layer} / {trait}: refit heldout R^2 {got:.6f} vs "
                    f"#2222 reference {want:.6f} (|delta|={delta:.6f} > tol={tol}) — the refit "
                    f"does NOT reproduce #2222's fit; do not ship deployable probes"
                )
            results[f"layer{layer}/{trait}"] = {"refit_r2": got, "ref_r2": want, "delta": delta}
        _log(f"parity layer={layer} done elapsed={time.time() - t0:.0f}s")
    _log(f"parity PASS ({len(results)} trait-layer cells, tol={tol})")
    return results


# --- Deployable full-pool fits + npz writer ------------------------------------------


def fit_deployable(xy: dict, reference: dict, regimes: list[str], device: str) -> dict[int, dict]:
    """One full-pool multi-target fit per DISTINCT deployment layer."""
    fr = reference["fit_regime"]
    lambdas = np.asarray(fr["lambda_grid"], dtype=np.float64)
    dof_cap = float(fr["dof_cap"])
    layers = sorted({REGIME_LAYERS[r][t] for r in regimes for t in TRAITS})
    _log(f"deployable fits: n={xy['n_rows']} > d={DIM} (well-posed), layers={layers}")
    fits: dict[int, dict] = {}
    for layer in layers:
        t0 = time.time()
        x_l = xy["x"][:, xy["layer_col"][layer], :].astype(np.float32)  # fp16->fp32, as #2222
        fit = ridge.dof_capped_ridge_fit_all(
            x_l, xy["y"], lambdas=lambdas, dof_cap=dof_cap, device=device
        )
        x64 = x_l.astype(np.float64)
        x_mu = x64.mean(axis=0)
        y_mu = xy["y"].mean(axis=0)
        # Intercept identity backing the npz representation (see module docstring).
        if not np.allclose(fit["b0"], y_mu - x_mu @ fit["w"], rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"layer {layer}: b0 != y_mu - x_mu @ w — fit-core contract broken")
        fits[layer] = {"fit": fit, "x_mu": x_mu, "y_mu": y_mu}
        _log(
            f"fit layer={layer} lam={[float(v) for v in fit['lam']]} "
            f"df={[round(float(v), 1) for v in fit['df']]} elapsed={time.time() - t0:.0f}s"
        )
    return fits


def write_probe_npz(
    path: Path,
    *,
    w: np.ndarray,
    b: float,
    x_mu: np.ndarray,
    x_sd: np.ndarray,
    layer: int,
    meta: dict,
) -> Path:
    """Atomic consumer-contract npz write (w/b/x_mu/x_sd/layer/meta; allow_pickle-free)."""
    d = w.shape[0]
    assert w.shape == (d,) and x_mu.shape == (d,) and x_sd.shape == (d,), (
        w.shape,
        x_mu.shape,
        x_sd.shape,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{path.stem}.npz")  # np.savez suffix trap (#1092)
    np.savez(
        tmp,
        w=np.asarray(w, dtype=np.float64),
        b=np.float64(b),
        x_mu=np.asarray(x_mu, dtype=np.float64),
        x_sd=np.asarray(x_sd, dtype=np.float64),
        layer=np.int64(layer),
        meta=np.array(json.dumps(meta)),
    )
    os.replace(tmp, path)
    return path


def write_probes(
    out_root: Path, xy: dict, fits: dict[int, dict], regimes: list[str], reference: dict
) -> list[Path]:
    """Per-(regime, trait) npz + in-process round-trip through the consumer loader."""
    from issue2224_predictor_scores import load_probe, probe_score

    fr = reference["fit_regime"]
    written: list[Path] = []
    for regime in regimes:
        for ti, trait in enumerate(TRAITS):
            layer = REGIME_LAYERS[regime][trait]
            entry = fits[layer]
            fit = entry["fit"]
            meta = {
                "issue": 2224,
                "trait": trait,
                "regime": regime,
                "layer": int(layer),
                "n_train": int(fit["n_train"]),
                "d": DIM,
                "selected_lambda": float(fit["lam"][ti]),
                "df": float(fit["df"][ti]),
                "lambda_grid": [float(v) for v in fr["lambda_grid"]],
                "dof_cap": float(fr["dof_cap"]),
                "x_input": "raw_respavg (judged dataset response, response-avg), fp16->fp32",
                "y_input": "mean graded judge score (form_a_judge merged; None-dropped)",
                "intercept_form": "x_mu=train col means, x_sd=1, b=y_mu[trait] "
                "(== x@w+b0 of the centered #2222 fit; no column scaling)",
                "fit_core": "issue2224_vendored_ridge.dof_capped_ridge_fit_all "
                f"(vendored from issue2222_analysis.py @ {ISSUE2222_PIN_SHA})",
                "reference": f"{REFERENCE_REPO_PATH} @ {ISSUE2222_PIN_SHA}",
                "label_fingerprints": xy["label_fingerprints"],
                "repro": common.repro_meta("issue2224_probe_refit"),
            }
            path = write_probe_npz(
                out_root / regime / f"{trait}.npz",
                w=fit["w"][:, ti],
                b=float(entry["y_mu"][ti]),
                x_mu=entry["x_mu"],
                x_sd=np.ones(DIM, dtype=np.float64),
                layer=layer,
                meta=meta,
            )
            # Round-trip through the EXACT consumer loader + scorer (consumer contract).
            probe = load_probe(path, DIM, layer)
            x_check = xy["x"][:64, xy["layer_col"][layer], :].astype(np.float32)
            got = probe_score(probe, x_check.astype(np.float64))
            want = ridge.ridge_predict(fit, x_check)[:, ti]
            if not np.allclose(got, want, rtol=1e-9, atol=1e-7):
                raise RuntimeError(
                    f"{path}: consumer round-trip mismatch — probe_score != ridge_predict "
                    f"(max |delta|={np.max(np.abs(got - want)):.3e})"
                )
            _log(f"wrote {path} (round-trip vs ridge_predict OK, n_check={len(got)})")
            written.append(path)
    return written


# --- Upload (ONE bulk commit + exact-set verify; sibling gen_natural shape) ---------


def upload_out_root(out_root: Path) -> None:
    """Fail-loud bulk upload of the probe npz set (+ meta JSON) to the data repo."""
    from explore_persona_space.orchestrate.hub import (
        DEFAULT_DATASET_REPO,
        _upload_folder_filtered,
    )

    rels = sorted(
        str(p.relative_to(out_root)) for p in out_root.rglob("*") if p.suffix in (".npz", ".json")
    )
    if not rels:
        raise RuntimeError(f"[refit-upload] nothing to upload under {out_root}")
    expected = [f"{HF_OUT_PREFIX}/{rel}" for rel in rels]
    url = _upload_folder_filtered(
        local_dir=out_root,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=HF_OUT_PREFIX,
        allow_patterns=["*.npz", "*.json"],
        expected_repo_paths=expected,
    )
    if not url:
        raise RuntimeError(
            f"[refit-upload] bulk upload of {out_root} -> {HF_OUT_PREFIX} FAILED or "
            f"verified incomplete"
        )
    _log(f"upload verified {len(expected)} files at {url}")


# --- CLI ----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    parser.add_argument(
        "--reference-json",
        default=None,
        help=f"path to the pinned form_a_probe.json (default: git show {ISSUE2222_PIN_SHA[:10]})",
    )
    parser.add_argument(
        "--regimes", nargs="+", choices=sorted(REGIME_LAYERS), default=["steer", "argmax"]
    )
    parser.add_argument(
        "--parity-check",
        action="store_true",
        help="run the full-LOFO parity gate vs the reference BEFORE the deployable fits",
    )
    parser.add_argument(
        "--parity-layers",
        type=int,
        nargs="+",
        default=[14, 15, 18, 19],
        help="layers for the parity gate (default: the 4 distinct deployment layers)",
    )
    parser.add_argument("--parity-tol", type=float, default=0.02)
    parser.add_argument(
        "--skip-fit", action="store_true", help="parity-only run (no deployable fits/writes)"
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser


def _import_check() -> None:
    """Execute every deferred import + the argparse-attribute completeness gate."""
    from huggingface_hub import HfApi  # noqa: F401

    import issue778_lib  # noqa: F401
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        DEFAULT_DATASET_REPO,
        _upload_folder_filtered,
    )
    from issue2224_predictor_scores import load_probe, probe_score  # noqa: F401

    for name in ("stage_hub_file", "retry_transient"):
        if not callable(getattr(hub, name, None)):
            raise RuntimeError(f"orchestrate.hub.{name} missing — staging contract broken")
    assert_args_attributes_defined(__file__)
    print("[import-check] OK issue2224_probe_refit", flush=True)


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        _import_check()
        return 0

    reference = load_reference(args.reference_json)
    assert_argmax_layers(reference)
    revision = resolve_data_revision()
    _log(f"data-repo revision pinned: {revision}")

    deploy_layers = {REGIME_LAYERS[r][t] for r in args.regimes for t in TRAITS}
    layers_needed = sorted(
        deploy_layers | (set(args.parity_layers) if args.parity_check else set())
    )
    labels = stage_labels(args.data_root, revision)
    xy = assemble_xy(args.data_root, labels, layers_needed, revision)
    assert_pool_matches_reference(xy, reference)
    _log(
        f"pool assembled: n={xy['n_rows']} (dropped {xy['n_dropped']}, "
        f"missing-base {xy['n_missing_base']}) — matches pinned #2222 fit regime"
    )

    parity = None
    if args.parity_check:
        parity = run_parity(xy, reference, args.parity_layers, args.parity_tol, args.device)

    if not args.skip_fit:
        fits = fit_deployable(xy, reference, args.regimes, args.device)
        written = write_probes(args.out_root, xy, fits, args.regimes, reference)
        summary = {
            "pin_sha": ISSUE2222_PIN_SHA,
            "data_repo_revision": revision,
            "fit_regime": reference["fit_regime"],
            "regime_layers": {r: REGIME_LAYERS[r] for r in args.regimes},
            "parity": parity,
            "files": [str(p.relative_to(args.out_root)) for p in written],
            "label_fingerprints": xy["label_fingerprints"],
            "meta": common.repro_meta("issue2224_probe_refit"),
        }
        common.atomic_write_json(summary, args.out_root / "refit_meta.json")
        _log(f"wrote {args.out_root / 'refit_meta.json'}")

    if args.upload:
        upload_out_root(args.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
