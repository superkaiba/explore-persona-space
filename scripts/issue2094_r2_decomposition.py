#!/usr/bin/env python3
"""Issue #2094 inline free-analysis round: R^2 decomposition of banked-map transport.

Answers the user-chat ask: per SETTING (matched_query / matched_prefix / cross),
compare four arms --

  1. ``real_unpatched``   real banked map applied to the UNPATCHED context state
  2. ``real_patched``     that SAME map applied to the PATCHED context state
  3. ``rand_patched``     a spectrum-matched RANDOM map on the patched state
  4. ``rand_unpatched``   that random map on the unpatched state

-- under three DVs:

  * ``level``      R^2 of predicting the answer vector ``va_tail`` itself
  * ``direction``  R^2 on the L2-NORMALIZED shift (direction only)
  * ``magnitude``  R^2 on the shift NORM (magnitude only)

Design calls (user-confirmed in chat, 2026-08-07):

* The "fair" random map is a SPECTRUM-MATCHED ROTATION ``W_rand = R @ W`` with
  ``R`` random orthogonal on the INPUT space -- identical singular values,
  identical output scale, destroyed input->output correspondence. This is the
  same null family as the operator-cosine rotation null already in
  ``issue2094_analysis._operator_similarity``. Rotations are drawn ONCE and
  shared across all six map cells (identical ``W`` dimensions).
* The four bars are LEVEL R^2; the direction and magnitude panels decompose the
  SHIFT.
* All four arms appear in every panel. For an UNPATCHED row the "shift" is the
  context's deviation from its setting-cell floor mean -- exactly the deviation
  ``R^2`` already measures, so the three DVs stay commensurable across arms.

BOTH mapping arms run as paired arms of one design (CLAUDE.md "Prefix mapping
AND context mapping"): context-end (``ce``, the #779 maps) AND prefix-end
(``pe``, the #1738 maps), at L14/L19/L26 each.

Nothing is FIT here -- the banked ridge maps are pre-fit artifacts and this
round only EVALUATES them, so the ``n_train < d`` refusal does not apply. The
evaluation ``n`` is thin on the unpatched arms (15 anchored contexts total), so
every cell carries its own ``n`` and the figure states it.

Phases::

    --phase stage     filtered Hub staging (only the banked-map layers)
    --phase analyze   the R^2 table -> r2_decomposition.json
    --phase figure    the per-slot figures

Reuse: ``issue2094_analysis`` supplies the payload reconstruction, bank/anchor
loaders and orientation binding verbatim; ``fmetrics`` supplies the ridge apply.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from explore_persona_space.experiments.issue2094 import fmetrics as FM  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_analysis as A  # noqa: E402

logger = logging.getLogger("issue2094_r2_decomposition")

REPO_ROOT = _SCRIPTS_DIR.parent
HF_DATA_REPO = A.HF_DATA_REPO
HF_PREFIX = A.HF_PREFIX

STAGE_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue2094_r2decomp")
OUT_DIR = REPO_ROOT / "eval_results" / "issue_2094" / "r2_decomposition"
FIG_DIR = REPO_ROOT / "figures" / "issue_2094"

# Banked-map read layers, both slots -- the paired prefix/context mapping arms.
CELLS: tuple[tuple[str, int], ...] = tuple(
    (slot, layer) for slot in ("ce", "pe") for layer in A.TRANSPORT_LAYERS[slot]
)
SETTINGS = ("matched_query", "matched_prefix", "cross")
ARMS = ("real_unpatched", "real_patched", "rand_patched", "rand_unpatched")
DVS = ("level", "direction", "magnitude")

N_ROTATIONS = 20
ROTATION_SEED = 20943  # sibling of the plan's 20941/20942 seeds
COS_POINT_SEED = 20944  # figure point-cloud subsample only; never a bar value
# Only shards whose LAYER_VARIANT is a banked-map layer are eligible (the
# `phase_transport` eligibility rule); the filename carries the variant.
_ELIGIBLE_SHARD_RE = re.compile(r"shard_(?:ce|pe)__L(?:14|19|26)__")

SLOT_LABEL = {"ce": "context-end $v_C$", "pe": "prefix-end state"}
SETTING_LABEL = {
    "matched_query": "same query",
    "matched_prefix": "same prefix",
    "cross": "different both",
}
ARM_LABEL = {
    "real_unpatched": "real map, unpatched",
    "real_patched": "real map, patched",
    "rand_patched": "random map, patched",
    "rand_unpatched": "random map, unpatched",
}
DV_LABEL = {
    "level": "$R^2$ — answer vector (level)",
    "direction": "$R^2$ — shift DIRECTION only",
    "magnitude": "$R^2$ — shift MAGNITUDE only",
}


# ── staging ───────────────────────────────────────────────────────────


def _eligible_repo_files() -> list[str]:
    """The filtered staging set: banked-map-layer va_store shards + bank/anchors.

    Deliberately NOT the whole ``va_store`` prefix (16.9 GB, 880 shards) -- only
    the 84 shards at L14/L19/L26, which is what the banked-map cells read. Keeps
    the pull under the ~10 GB pod-routing threshold.

    Listings go through ``hub.list_hf_files_under_path`` (ONE retried server-side
    scoped tree walk per prefix) rather than a bare ``HfApi.list_repo_tree`` -- a
    raw listing against the ~1M-file data repo is the #920 false-failure class.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    want: list[str] = []
    prefixes = [f"{HF_PREFIX}/analysis_tensors/{sub}" for sub in ("va_store", "vc_bank", "anchors")]
    prefixes.append(f"{HF_PREFIX}/raw_completions/grid")
    for prefix in prefixes:
        listed = hub.list_hf_files_under_path(
            api, HF_DATA_REPO, prefix, repo_type="dataset", revision="main"
        )
        shard_keyed = prefix.endswith(("va_store", "grid"))
        for path in listed:
            if shard_keyed and not _ELIGIBLE_SHARD_RE.search(Path(path).name):
                continue
            want.append(path)
    return sorted(want)


def phase_stage(args: argparse.Namespace) -> int:
    """Filtered Hub staging with flushed per-file progress (the #2153 contract)."""
    t0 = time.monotonic()
    print(f"[stage] entry root={STAGE_ROOT}", flush=True)
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    files = _eligible_repo_files()
    print(f"[stage] {len(files)} files to stage", flush=True)

    def _one(path_in_repo: str) -> tuple[str, bool]:
        target = STAGE_ROOT / path_in_repo
        if target.exists() and target.stat().st_size > 0:
            return path_in_repo, False
        hub.stage_hub_file(HF_DATA_REPO, path_in_repo, target, repo_type="dataset")
        return path_in_repo, True

    done = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_one, p): p for p in files}
        for fut in as_completed(futures):
            path, fetched = fut.result()
            done += 1
            print(
                f"[stage] unit {done}/{len(files)} {path} "
                f"{'fetched' if fetched else 'cached'} elapsed={time.monotonic() - t0:.1f}s",
                flush=True,
            )

    # The six banked ridge bundles live under their OWN issue prefixes.
    maps_dir = STAGE_ROOT / "banked_maps"
    for spec in A.BANKED_MAPS:
        target = maps_dir / spec["repo_path"]
        if target.exists() and target.stat().st_size > 0:
            print(f"[stage] map {spec['map_id']} cached", flush=True)
            continue
        hub.stage_hub_file(HF_DATA_REPO, spec["repo_path"], target, repo_type="dataset")
        print(f"[stage] map {spec['map_id']} fetched", flush=True)

    print(f"[stage] done elapsed={time.monotonic() - t0:.1f}s", flush=True)
    return 0


# ── R^2 primitives ────────────────────────────────────────────────────


def _r2_vec(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """Multivariate R^2 = 1 - SSE/SST, SST around the sample mean of ``y``.

    ``y``/``yhat`` are ``(n, d)``. Returns NaN when SST is 0 (a degenerate
    single-point cell) -- flagged, never coerced.
    """
    assert y.shape == yhat.shape and y.dim() == 2, (y.shape, yhat.shape)
    y64, p64 = y.double(), yhat.double()
    sse = (y64 - p64).pow(2).sum()
    sst = (y64 - y64.mean(dim=0, keepdim=True)).pow(2).sum()
    return float("nan") if float(sst) == 0.0 else float(1.0 - sse / sst)


def _r2_scalar(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """R^2 for a 1-D target (the magnitude DV). NaN on zero variance."""
    assert y.shape == yhat.shape and y.dim() == 1, (y.shape, yhat.shape)
    y64, p64 = y.double(), yhat.double()
    sse = (y64 - p64).pow(2).sum()
    sst = (y64 - y64.mean()).pow(2).sum()
    return float("nan") if float(sst) == 0.0 else float(1.0 - sse / sst)


def _unit(v: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalize; a zero row stays zero (counted by the caller)."""
    n = v.double().norm(dim=-1, keepdim=True)
    return v.double() / n.clamp_min(torch.finfo(torch.float64).tiny)


def _three_dvs(
    y_level: torch.Tensor,
    p_level: torch.Tensor,
    y_shift: torch.Tensor,
    p_shift: torch.Tensor,
) -> dict[str, float]:
    """The three DVs for one (cell, setting, arm): level / direction / magnitude."""
    return {
        "level": _r2_vec(y_level, p_level),
        "direction": _r2_vec(_unit(y_shift), _unit(p_shift)),
        "magnitude": _r2_scalar(y_shift.double().norm(dim=-1), p_shift.double().norm(dim=-1)),
    }


def _shift_companions(y_shift: torch.Tensor, p_shift: torch.Tensor) -> dict[str, float]:
    """Interpretable companions to the two shift R^2s.

    ``mean_cosine`` ties the direction panel back to #2094's native transport
    metric; ``median_norm_ratio`` explains the magnitude panel -- an R^2 far
    below zero on the norms is a multiplicative SCALE miss, not a ranking miss,
    and the ratio is what says so.
    """
    yn = y_shift.double().norm(dim=-1)
    pn = p_shift.double().norm(dim=-1)
    ok = (yn > 0) & (pn > 0)
    cos = FM.safe_cosine(y_shift.float(), p_shift.float()).double()
    return {
        "mean_cosine": float(cos[ok].mean()) if bool(ok.any()) else float("nan"),
        "median_norm_ratio": float((pn[ok] / yn[ok]).median()) if bool(ok.any()) else float("nan"),
        "_cos_rows": cos[ok].tolist(),
    }


def _noise_ceiling(anchor_va: dict, ctxs: list[str], layer: int) -> dict[str, float]:
    """Max R^2 any context-level predictor can reach on a SINGLE rollout target.

    The unpatched arms are scored against a floor MEAN over 10 anchor draws; the
    patched arms are scored against ONE greedy rollout (the grid stores a single
    draw per pair x dose x vec_type). That asymmetry penalises the patched arms
    with rollout noise the unpatched arms never face, so the comparison the user
    asked for needs its ceiling drawn.

    Leave-one-out: predict each draw from the mean of that context's OTHER nine.
    Reported for all three DVs, on the same estimators the arms use.
    """
    ys, ps = [], []
    for c in ctxs:
        draws = anchor_va[c]["tail"][:, layer].double()  # (K, d)
        k = draws.shape[0]
        assert k > 1, (c, k)
        loo = (draws.sum(dim=0, keepdim=True) - draws) / (k - 1)
        ys.append(draws)
        ps.append(loo)
    y = torch.cat(ys)
    p = torch.cat(ps)
    gm = y.mean(dim=0, keepdim=True)
    return {
        **_three_dvs(y.float(), p.float(), y - gm, p - gm),
        "n_draws_total": int(y.shape[0]),
    }


def _rotations(d: int, n: int, seed: int) -> list[torch.Tensor]:
    """``n`` Haar-ish random orthogonal ``(d, d)`` matrices (QR of a Gaussian).

    Sign-corrected against the R diagonal so the draw is Haar-distributed rather
    than QR-sign-biased. Drawn ONCE and reused across every map cell -- all six
    bundles share the same ``W`` dimensions.
    """
    gen = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []
    for k in range(n):
        t0 = time.monotonic()
        a = torch.randn(d, d, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        out.append(q)
        print(f"[rot] unit {k + 1}/{n} elapsed={time.monotonic() - t0:.2f}s", flush=True)
    return out


def _rotated_bundle(bundle: dict, rot: torch.Tensor, orientation: str) -> dict:
    """Spectrum-matched random map: rotate the INPUT space of ``W``.

    ``zW``: ``dev = z @ W`` with ``W`` ``(d_in, d_out)`` -> ``W_rand = rot @ W``,
    i.e. ``z @ (rot @ W) = (z @ rot) @ W`` -- the input is rotated, the singular
    values (hence the output scale) are untouched.
    ``Wz``: ``dev = z @ W.T`` with ``W`` ``(d_out, d_in)`` -> ``W_rand = W @ rot.T``.
    """
    out = dict(bundle)
    w = bundle["W"].double()
    out["W"] = (rot @ w) if orientation == "zW" else (w @ rot.T)
    return out


# ── analyze ───────────────────────────────────────────────────────────


def _setting_contexts(pairs, setting: str, available: set[str]) -> list[str]:
    """Contexts appearing in a setting's pairs that have BOTH bank state + anchors."""
    ids: set[str] = set()
    for p in pairs:
        if p.setting == setting:
            ids |= {p.a, p.b}
    return sorted(ids & available)


def _load_patched_rows(cfg, slot: str, layer: int, anchor_va: dict) -> dict[str, dict]:
    """Realized patched answer vectors + patched map INPUTS, grouped by setting.

    Mirrors ``issue2094_analysis.phase_transport``'s shard walk, but keeps the
    VECTORS instead of reducing to a cosine. STEERED rows only (the four arms
    vary the MAP, not the donor -- the donor-shuffled null is #2094's separate
    transport control), and the degenerate-by-design self-transfer rows are
    excluded exactly as the transport bootstrap excludes them.
    """
    bank = _BANK_CACHE["bank"]
    pairs_by_id = _BANK_CACHE["pairs_by_id"]
    donor_map = _BANK_CACHE["donor_map"]
    want_variant = f"L{layer}"
    per_setting: dict[str, dict[str, list]] = {
        s: {"y_level": [], "x_patched": [], "fl": [], "v_s": []} for s in SETTINGS
    }
    for shard in sorted(cfg.va_dir.glob(f"shard_{slot}__{want_variant}__*.pt")):
        jsonl = cfg.rollouts_dir / f"shard_{shard.stem.removeprefix('shard_')}.jsonl"
        rows = list(A._iter_jsonl(jsonl))
        if not rows:
            continue
        assert rows[0]["slot"] == slot and rows[0]["layer_variant"] == want_variant, rows[0]
        va_tail = torch.load(shard, map_location="cpu", weights_only=False)["va_tail"].float()
        for i, r in enumerate(rows):
            if r["arm"] != "steered" or A.degenerate_self(r):
                continue
            bucket = per_setting[r["setting"]]
            fl = anchor_va[r["context_a"]]["tail"][:, layer].mean(dim=0)
            payload, payload_kind = A.transport_row_payload(bank, r, pairs_by_id, donor_map)
            v_s = A._slot_input_vector(bank, r["context_a"], slot, layer)
            d_l = payload[-1][layer].float()
            # The patched map INPUT is exactly what the hook injected.
            x_patched = d_l if payload_kind == "state" else v_s + float(r["alpha"]) * d_l
            bucket["y_level"].append(va_tail[i, layer])
            bucket["x_patched"].append(x_patched)
            bucket["fl"].append(fl)
            bucket["v_s"].append(v_s)
    return {
        s: {k: torch.stack(v) for k, v in b.items()} for s, b in per_setting.items() if b["y_level"]
    }


_BANK_CACHE: dict = {}


def phase_analyze(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    print("[analyze] entry", flush=True)
    cfg = A.AnalysisConfig(
        in_root=STAGE_ROOT,
        out_root=OUT_DIR,
        judge_root=OUT_DIR,
        hf_revision=None,
        skip_disk_check=True,
        no_upload=True,
    )
    parity = json.loads((REPO_ROOT / "eval_results/issue_2094/map_parity.json").read_text())
    bank = A._load_vc_bank(cfg)
    anchor_va = A._load_anchor_va(cfg)
    pairs = BANK.build_pairs()
    _BANK_CACHE.update(
        bank=bank,
        pairs_by_id={p.pair_id: p for p in pairs},
        donor_map=bank.get("donor_derangement") or BANK.donor_derangement(pairs),
    )
    available = set(bank["per_context"]) & set(anchor_va)
    print(f"[analyze] {len(available)} contexts with bank state + anchors", flush=True)

    rots = _rotations(A.HIDDEN, N_ROTATIONS, ROTATION_SEED)
    records: list[dict] = []
    cos_rows: dict[str, list[float]] = {}

    for ci, (slot, layer) in enumerate(CELLS):
        spec = next(s for s in A.BANKED_MAPS if s["arm"] == slot and s["layer"] == layer)
        bundle = A._load_bundle(cfg.maps_dir / spec["repo_path"])
        orientation = A._orientation_for(parity, spec["map_id"])
        rot_bundles = [_rotated_bundle(bundle, r, orientation) for r in rots]
        patched = _load_patched_rows(cfg, slot, layer, anchor_va)

        for setting in SETTINGS:
            # ---- unpatched arms (1 and 4) -------------------------------
            ctxs = _setting_contexts(pairs, setting, available)
            x_un = torch.stack([A._slot_input_vector(bank, c, slot, layer) for c in ctxs])
            y_un = torch.stack([anchor_va[c]["tail"][:, layer].mean(dim=0) for c in ctxs])
            y_un_shift = y_un.double() - y_un.double().mean(dim=0, keepdim=True)

            def _unpatched(b: dict) -> dict[str, float]:
                p = FM.apply_ridge_map(b, x_un, orientation=orientation)
                p_shift = p.double() - p.double().mean(dim=0, keepdim=True)
                return {
                    **_three_dvs(y_un, p, y_un_shift, p_shift),
                    **_shift_companions(y_un_shift, p_shift),
                }

            _real_unpatched_m = _unpatched(bundle)
            records.append(
                {
                    "slot": slot,
                    "layer": layer,
                    "map_id": spec["map_id"],
                    "setting": setting,
                    "arm": "real_unpatched",
                    "n": len(ctxs),
                    **_metric_fields(_real_unpatched_m),
                }
            )
            cos_rows[_cos_key(slot, layer, setting, "real_unpatched")] = _real_unpatched_m[
                "_cos_rows"
            ]
            draws = [_unpatched(b) for b in rot_bundles]
            cos_rows[_cos_key(slot, layer, setting, "rand_unpatched")] = _subsample(
                [c for d in draws for c in d["_cos_rows"]], 600, COS_POINT_SEED
            )
            records.append(
                _null_record(slot, layer, spec, setting, "rand_unpatched", len(ctxs), draws)
            )

            # ---- patched arms (2 and 3) ---------------------------------
            blk = patched.get(setting)
            if blk is None:
                continue
            y_lv, x_p, fl = blk["y_level"], blk["x_patched"], blk["fl"]
            y_sh = y_lv.double() - fl.double()

            def _patched(b: dict) -> dict[str, float]:
                p = FM.apply_ridge_map(b, x_p, orientation=orientation)
                base = FM.apply_ridge_map(b, blk["v_s"], orientation=orientation)
                p_sh = p.double() - base.double()
                return {**_three_dvs(y_lv, p, y_sh, p_sh), **_shift_companions(y_sh, p_sh)}

            _real_patched_m = _patched(bundle)
            records.append(
                {
                    "slot": slot,
                    "layer": layer,
                    "map_id": spec["map_id"],
                    "setting": setting,
                    "arm": "real_patched",
                    "n": int(y_lv.shape[0]),
                    **_metric_fields(_real_patched_m),
                }
            )
            cos_rows[_cos_key(slot, layer, setting, "real_patched")] = _real_patched_m["_cos_rows"]
            draws = [_patched(b) for b in rot_bundles]
            cos_rows[_cos_key(slot, layer, setting, "rand_patched")] = _subsample(
                [c for d in draws for c in d["_cos_rows"]], 600, COS_POINT_SEED
            )
            records.append(
                _null_record(slot, layer, spec, setting, "rand_patched", int(y_lv.shape[0]), draws)
            )

        print(
            f"[analyze] unit {ci + 1}/{len(CELLS)} {spec['map_id']} "
            f"elapsed={time.monotonic() - t0:.1f}s",
            flush=True,
        )

    ceilings = {
        str(layer): _noise_ceiling(anchor_va, sorted(available), layer)
        for layer in sorted({layer for _, layer in CELLS})
    }
    for layer, c in ceilings.items():
        print(
            f"[ceiling] L{layer} level={c['level']:.3f} dir={c['direction']:.3f} "
            f"mag={c['magnitude']:.3f} (n_draws={c['n_draws_total']})",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": records,
        "noise_ceilings": ceilings,
        "conventions": {
            "answer_vector": "va_tail (mean over completion tokens + assistant "
            "end-of-turn tail) -- the capture_answer_vector convention the banked "
            "maps' outputs were fit under",
            "random_map": f"spectrum-matched rotation W_rand = R @ W, {N_ROTATIONS} Haar "
            f"draws (seed {ROTATION_SEED}); identical singular values and output scale",
            "level_dv": "R^2 = 1 - SSE/SST of predicting va_tail, SST around the "
            "setting-cell mean of the realized answer vectors",
            "direction_dv": "R^2 on the L2-normalized shift",
            "magnitude_dv": "R^2 on the shift norm",
            "unpatched_shift": "each context's deviation from its setting-cell floor mean",
            "patched_rows": "STEERED rows only, degenerate-by-design self-transfer excluded; "
            "doses pooled",
            "null_bar": "mean over rotation draws; error bar = draw sd",
            "noise_ceiling": "leave-one-out over the 10 anchor draws per context -- the max "
            "R^2 any context-level predictor can reach when the target is a SINGLE rollout. "
            "The unpatched arms are scored against a 10-draw floor MEAN, the patched arms "
            "against ONE greedy rollout, so this line is what makes the two comparable.",
            "setting_invariance": "the unpatched arms are setting-INVARIANT by construction: "
            "all 15 anchored contexts appear in the pairs of every setting, so arms 1 and 4 "
            "repeat across the three columns rather than varying with them.",
            "degenerate_cells": "prefix-end x matched_prefix has NO patched rows -- a "
            "matched-prefix pair shares its prefix, so a prefix-end patch is self-transfer "
            "by design and is excluded, not zero.",
        },
        "rotation_seed": ROTATION_SEED,
        "n_rotations": N_ROTATIONS,
        "provenance": _provenance(),
    }
    (OUT_DIR / "r2_decomposition.json").write_text(json.dumps(payload, indent=1))
    (OUT_DIR / "cosine_rows.json").write_text(
        json.dumps(
            {
                "rows": cos_rows,
                "note": "per-row cos(realized shift, map-predicted shift). Real arms carry "
                "EVERY row; random arms pool all rotation draws and subsample to 600 points "
                f"(seed {COS_POINT_SEED}) for the figure -- the BAR always uses every value.",
            }
        )
    )
    print(f"[analyze] wrote {OUT_DIR / 'r2_decomposition.json'} rows={len(records)}", flush=True)
    return 0


def _metric_fields(m: dict[str, float]) -> dict[str, float]:
    """Prefix the three DVs with ``r2_``; carry the companions under their own names.

    Keys starting ``_`` are per-row payloads (the cosine vectors), collected into
    their own sidecar rather than inlined into every record.
    """
    out = {f"r2_{dv}": m[dv] for dv in DVS}
    out.update({k: v for k, v in m.items() if k not in DVS and not k.startswith("_")})
    return out


def _null_record(slot, layer, spec, setting, arm, n, draws) -> dict:
    """Collapse the rotation draws into one bar: mean + sd per DV and companion."""
    rec = {
        "slot": slot,
        "layer": layer,
        "map_id": spec["map_id"],
        "setting": setting,
        "arm": arm,
        "n": n,
        "n_draws": len(draws),
    }
    for dv in DVS:
        vals = np.array([d[dv] for d in draws], dtype=float)
        rec[f"r2_{dv}"] = float(np.nanmean(vals))
        rec[f"r2_{dv}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    for comp in ("mean_cosine", "median_norm_ratio"):
        vals = np.array([d[comp] for d in draws], dtype=float)
        rec[comp] = float(np.nanmean(vals))
    return rec


def _cos_key(slot: str, layer: int, setting: str, arm: str) -> str:
    return f"{slot}|L{layer}|{setting}|{arm}"


def _subsample(vals: list[float], cap: int, seed: int) -> list[float]:
    """Cap a point cloud for the figure sidecar; the BAR always uses every value."""
    if len(vals) <= cap:
        return vals
    rng = np.random.default_rng(seed)
    return [float(v) for v in rng.choice(np.asarray(vals, dtype=float), cap, replace=False)]


def _provenance() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance())


# ── figure ────────────────────────────────────────────────────────────


def phase_figure(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots

    paper_plots.set_paper_style("neurips")
    payload = json.loads((OUT_DIR / "r2_decomposition.json").read_text())
    recs = payload["records"]
    ceilings = payload["noise_ceilings"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "real_unpatched": "#0173B2",
        "real_patched": "#029E73",
        "rand_patched": "#DE8F05",
        "rand_unpatched": "#CC78BC",
    }
    # level and magnitude span orders of magnitude below zero (the maps
    # over-predict shift NORM by ~4-20x); symlog keeps the near-zero structure
    # legible without hiding how far the negatives go.
    SCALE = {"level": "symlog", "direction": "linear", "magnitude": "symlog"}
    written = []
    for slot in ("ce", "pe"):
        layers = list(A.TRANSPORT_LAYERS[slot])
        fig, axes = plt.subplots(
            len(DVS), len(SETTINGS), figsize=(15.5, 11.0), sharex=True, squeeze=False
        )
        for ri, dv in enumerate(DVS):
            row_vals = [
                r[f"r2_{dv}"] for r in recs if r["slot"] == slot and np.isfinite(r[f"r2_{dv}"])
            ]
            lo, hi = min(row_vals + [0.0]), max(row_vals + [1.0])
            for ci, setting in enumerate(SETTINGS):
                ax = axes[ri][ci]
                width, xs = 0.2, np.arange(len(layers), dtype=float)
                for ai, arm in enumerate(ARMS):
                    vals, errs, missing = [], [], []
                    for li, layer in enumerate(layers):
                        m = [
                            r
                            for r in recs
                            if r["slot"] == slot
                            and r["layer"] == layer
                            and r["setting"] == setting
                            and r["arm"] == arm
                        ]
                        if m:
                            vals.append(m[0][f"r2_{dv}"])
                            errs.append(m[0].get(f"r2_{dv}_sd", 0.0))
                        else:
                            vals.append(np.nan)
                            errs.append(0.0)
                            missing.append(li)
                    off = (ai - 1.5) * width
                    ax.bar(
                        xs + off,
                        vals,
                        width,
                        yerr=errs,
                        capsize=2,
                        color=colors[arm],
                        label=ARM_LABEL[arm] if (ri == 0 and ci == 0) else None,
                        edgecolor="none",
                    )
                    # A structurally-absent cell is labelled, never a zero bar.
                    for li in missing:
                        ax.text(
                            xs[li] + off,
                            0.0,
                            "N/A",
                            rotation=90,
                            ha="center",
                            va="bottom",
                            fontsize=6,
                            color="0.4",
                        )
                # Single-rollout noise ceiling, per layer group.
                for li, layer in enumerate(layers):
                    c = ceilings[str(layer)][dv]
                    ax.plot(
                        [xs[li] - 0.42, xs[li] + 0.42],
                        [c, c],
                        ls="--",
                        lw=1.1,
                        color="0.25",
                        label="single-rollout noise ceiling"
                        if (ri == 0 and ci == 0 and li == 0)
                        else None,
                    )
                ax.axhline(0.0, color="0.35", lw=0.9)
                ax.set_yscale(SCALE[dv], **({"linthresh": 1.0} if SCALE[dv] == "symlog" else {}))
                ax.set_xticks(xs)
                ax.set_xticklabels([f"L{layer}" for layer in layers])
                ax.set_ylim(min(lo * 1.6, -1.2), max(hi * 1.15, 1.2))
                if ci == 0:
                    ax.set_ylabel(DV_LABEL[dv] + ("  (symlog)" if SCALE[dv] == "symlog" else ""))
                if ri == 0:
                    n_un = next(
                        (
                            r["n"]
                            for r in recs
                            if r["slot"] == slot
                            and r["setting"] == setting
                            and r["arm"] == "real_unpatched"
                        ),
                        0,
                    )
                    n_pa = next(
                        (
                            r["n"]
                            for r in recs
                            if r["slot"] == slot
                            and r["setting"] == setting
                            and r["arm"] == "real_patched"
                        ),
                        0,
                    )
                    ax.set_title(
                        f"{SETTING_LABEL[setting]}\n"
                        f"n = {n_un} contexts unpatched · "
                        + (f"{n_pa} rows patched" if n_pa else "patched N/A — degenerate")
                    )
        fig.suptitle(
            f"Banked-map $R^2$ at the {SLOT_LABEL[slot]}: real vs spectrum-matched random map, "
            "unpatched vs patched",
            y=0.995,
        )
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.068),
            ncol=5,
            frameon=False,
        )
        # Explicit newlines, not wrap=True: matplotlib wraps against the figure
        # width and silently overprinted the legend at this figure size.
        fig.text(
            0.5,
            0.008,
            f"Random-map bars = mean of {N_ROTATIONS} spectrum-matched rotation draws "
            "(W_rand = R·W, identical singular values; error bar = draw sd).  "
            "Patched arms: steered rows only, doses pooled, degenerate self-transfer excluded.\n"
            "The unpatched arms are setting-INVARIANT by construction — all 15 anchored "
            "contexts appear in every setting's pairs — so they repeat across columns.  "
            "Ceiling (dashed) = leave-one-out over the 10 anchor draws: the most any "
            "context-level predictor can score against a SINGLE rollout, which is what the "
            "patched arms are scored against.",
            fontsize=7,
            color="0.35",
            ha="center",
            va="bottom",
            linespacing=1.5,
        )
        fig.tight_layout(rect=(0, 0.105, 1, 0.985))
        stem = f"r2_decomposition_{'context_end' if slot == 'ce' else 'prefix_end'}"
        paths = paper_plots.savefig_paper(fig, stem, dir=FIG_DIR)
        plt.close(fig)
        written.append(str(paths["png"]))
        print(f"[figure] wrote {paths['png']}", flush=True)
    return 0


def phase_cosine_figure(args: argparse.Namespace) -> int:
    """Cosine-similarity view of the same 4-arm x 3-setting decomposition.

    The natively interpretable companion to the direction-R^2 row: on unit
    vectors R^2 and cosine are monotonically related (R^2 = -1 IS cos = 0), and
    cosine is the metric #2094's own result1b_transport_cosines figure reports,
    so this is the panel that connects the two. Both slots share one figure;
    points are the per-row values, bars their mean.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots

    paper_plots.set_paper_style("neurips")
    recs = json.loads((OUT_DIR / "r2_decomposition.json").read_text())["records"]
    cos_rows = json.loads((OUT_DIR / "cosine_rows.json").read_text())["rows"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "real_unpatched": "#0173B2",
        "real_patched": "#029E73",
        "rand_patched": "#DE8F05",
        "rand_unpatched": "#CC78BC",
    }
    rng = np.random.default_rng(COS_POINT_SEED)
    fig, axes = plt.subplots(2, len(SETTINGS), figsize=(15.5, 8.4), sharex=False, squeeze=False)
    for ri, slot in enumerate(("ce", "pe")):
        layers = list(A.TRANSPORT_LAYERS[slot])
        for ci, setting in enumerate(SETTINGS):
            ax = axes[ri][ci]
            width, xs = 0.2, np.arange(len(layers), dtype=float)
            for ai, arm in enumerate(ARMS):
                off = (ai - 1.5) * width
                for li, layer in enumerate(layers):
                    m = [
                        r
                        for r in recs
                        if r["slot"] == slot
                        and r["layer"] == layer
                        and r["setting"] == setting
                        and r["arm"] == arm
                    ]
                    if not m:
                        ax.text(
                            xs[li] + off,
                            0.0,
                            "N/A",
                            rotation=90,
                            ha="center",
                            va="bottom",
                            fontsize=6,
                            color="0.4",
                        )
                        continue
                    ax.bar(
                        xs[li] + off,
                        m[0]["mean_cosine"],
                        width,
                        color=colors[arm],
                        edgecolor="none",
                        label=ARM_LABEL[arm] if (ri == 0 and ci == 0 and li == 0) else None,
                        zorder=1,
                    )
                    pts = cos_rows.get(_cos_key(slot, layer, setting, arm), [])
                    if pts:
                        jit = rng.uniform(-0.055, 0.055, size=len(pts))
                        ax.scatter(
                            xs[li] + off + jit,
                            pts,
                            s=1.6,
                            color="0.15",
                            alpha=0.30,
                            linewidths=0,
                            zorder=2,
                            rasterized=True,
                        )
            ax.axhline(0.0, color="0.35", lw=0.9)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"L{layer}" for layer in layers])
            ax.set_ylim(-1.02, 1.02)
            if ci == 0:
                ax.set_ylabel(f"{SLOT_LABEL[slot]}\ncos(realized shift, predicted shift)")
            if ri == 0:
                ax.set_title(SETTING_LABEL[setting])
    fig.suptitle(
        "Banked-map shift cosine: real vs spectrum-matched random map, unpatched vs patched",
        y=0.995,
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=4, frameon=False
    )
    fig.text(
        0.5,
        0.008,
        "Bars = mean over every row; points = per-row values (random arms pool all "
        f"{N_ROTATIONS} rotation draws, subsampled to 600 for legibility — the bar still uses "
        "every value).\nFor the UNPATCHED arms the 'shift' is each context's deviation from "
        "its setting-cell floor mean. cos = 0 is the no-alignment line and is exactly where "
        "direction R² = −1 sits.",
        fontsize=7,
        color="0.35",
        ha="center",
        va="bottom",
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.115, 1, 0.985))
    paths = paper_plots.savefig_paper(fig, "r2_decomposition_shift_cosine", dir=FIG_DIR)
    plt.close(fig)
    print(f"[figure] wrote {paths['png']}", flush=True)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase", required=True, choices=("stage", "analyze", "figure", "cosine-figure")
    )
    args = ap.parse_args()
    return {
        "stage": phase_stage,
        "analyze": phase_analyze,
        "figure": phase_figure,
        "cosine-figure": phase_cosine_figure,
    }[args.phase](args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
