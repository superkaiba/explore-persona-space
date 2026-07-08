#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, M⁺, →) in scientific docstrings + log messages.
"""Issue #833 follow-up (nonverbatim-profile-ablation) — Phase N2: non-emission
chains + paired diffs on the 291 retained cells (plan v10 §4, VM CPU, 0 GPU-h).

Fits SEVEN ridge PRESS-LOCO arms per fact layer, every one REFIT on the SAME
retained-cell subset (cell composition held fixed — plan §3), plus the eq5
sensitivity arm:

  on_nonemit         (C⁺, V_on over the NON-EMISSION rows)   — the manipulated arm
  ctrl_nonemit       (C0, V0on over the non-emission rows)   — text-carried analogue
  on_full_ret        (C⁺, Von — all 30 rows/cell)            — full-N full-text comparator
  on_full_matchedN   (C⁺, V_on over the seed-42 matched-N sample, emissions included)
                     — the noise-dose-matched PRIMARY comparator (ρ(retained_N, E) = −0.748)
  ctrl_full_ret      (C0, V0on)                              — control analogue
  off_full_ret       (C⁺, Vplus)                             — the “+0.2–0.5 band” anchor
  base_full_ret      (C0, V0)                                — floor
  on_nonemit_eq5     (C⁺, V_on over the seed-42 5-row nonemit subsample) — sensitivity

Paired family-clustered diffs (1,000 resamples, seed 0 — verbatim estimator
reuse via the `chain_rho_ctrl.py` import pattern): PRIMARY
(on_nonemit − on_full_matchedN) at the pre-registered headline layer L14;
SECONDARY (on_nonemit − on_full_ret) [the v9 full-N diff, noise-dose-confounded];
(ctrl_nonemit − ctrl_full_ret); (on_nonemit − ctrl_nonemit);
(on_nonemit − off_full_ret); eq5 branch reads (eq5 − matchedN, eq5 − full_ret).

The shared PCA target basis is `fitM._pca_basis_v0(V0_RETAINED, 64)` — computed
on the retained subset's base stack per layer and shared across ALL arms (the
production shared-basis recipe restricted to the analysis set, so every fit is
internal to the retained cells; n=291 > 64 keeps the basis full-rank).

Consumer-side guards: retained key set == manifest == matched-N/eq5 index keys;
every subset npz's `probe_idx` re-asserted against the committed sample indices
/ manifest counts; all 7 families + all 16 sources represented (STOP on a miss
— plan §13). `--fulltext-npz-root` consumes the parity-FAIL contingency's
within-run full-text re-extraction for the Von/V0on comparator legs (arms
off_full_ret / base_full_ret stay on the joined cache's rbase-era anchors).

Outputs: `<out-dir>/chain_rho_nonemit/fact_L{7,14,21}.json`, written atomically
PER LAYER the moment the layer completes, resume keyed on (joined-cache sha,
manifest sha) with `--force-rerun` override.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402

from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    clustered_bootstrap_spearman,
)

logger = logging.getLogger("issue833.chain_nonemit")

BEHAVIOR = "fact"
LAYERS_DEFAULT = (7, 14, 21)
SEED = 42
TARGET_DIM = 64  # the production shared top-v0-PCA target dim
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue833_onpolicy_map"
SUBSET_NAMESPACES = {
    "nonemit": "analysis_tensors_nonemit",
    "matchedn": "analysis_tensors_matchedN",
    "eq5": "analysis_tensors_nonemit_eq5",
}
_REQUIRED_OUT_KEYS = frozenset(
    {
        "arms",
        "paired_diffs",
        "n_retained_cells",
        "meta",
    }
)
# Registered paired diffs: (name, minuend arm, subtrahend arm). PRIMARY first.
PAIRED_DIFFS = (
    ("on_nonemit_minus_on_full_matchedN", "on_nonemit", "on_full_matchedN"),  # PRIMARY @ L14
    ("on_nonemit_minus_on_full_ret", "on_nonemit", "on_full_ret"),  # v9 secondary
    ("ctrl_nonemit_minus_ctrl_full_ret", "ctrl_nonemit", "ctrl_full_ret"),
    ("on_nonemit_minus_ctrl_nonemit", "on_nonemit", "ctrl_nonemit"),
    ("on_nonemit_minus_off_full_ret", "on_nonemit", "off_full_ret"),
    ("on_nonemit_eq5_minus_on_full_matchedN", "on_nonemit_eq5", "on_full_matchedN"),  # eq5 branch
    ("on_nonemit_eq5_minus_on_full_ret", "on_nonemit_eq5", "on_full_ret"),
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as e:  # metadata-only
        return f"unavailable ({e})"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _out_is_complete(path: Path, resume_key: str) -> bool:
    """Resume predicate: complete output pinned to the SAME (cache, manifest) pair."""
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return False
    return _REQUIRED_OUT_KEYS.issubset(obj) and obj.get("meta", {}).get("resume_key") == resume_key


def stage_namespace_from_hf(root: Path, namespace: str) -> None:
    """Stage one subset namespace from HF (scoped list_repo_tree + threaded
    per-file hf_hub_download — the r7b shape; snapshot_download is BARRED
    against the ~1M-file repo, gotchas.md)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import HfApi, hf_hub_download

    prefix = f"{HF_PREFIX}/{namespace}"
    paths = [
        e.path
        for e in HfApi().list_repo_tree(
            HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
        )
        if e.path.endswith((".npz", ".json"))
    ]
    if not paths:
        raise FileNotFoundError(f"HF prefix {prefix} is empty — run Phase N1 first")
    logger.info("[phase=stage] %s: %d files from HF", namespace, len(paths))
    stage_dir = root.parent / "hf_stage_nonemit"

    def fetch(p: str, attempts: int = 4) -> str | None:
        for attempt in range(attempts):
            try:
                hf_hub_download(HF_DATA_REPO, p, repo_type="dataset", local_dir=str(stage_dir))
                return None
            except Exception:
                if attempt == attempts - 1:
                    return p
                time.sleep(20 * (attempt + 1))
        return p

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(fetch, p) for p in paths]):
            if f.result():
                failed.append(f.result())
    if failed:
        raise RuntimeError(f"{namespace}: {len(failed)} downloads failed (e.g. {failed[:3]})")
    src = stage_dir / HF_PREFIX / namespace
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise RuntimeError(f"{root} exists — refusing to overwrite a local namespace")
    src.rename(root)


def _load_subset_stack(
    root: Path,
    keys: list[str],
    layer: int,
    *,
    expected_probe_ids: dict[str, list[int]] | None,
    manifest_cells: dict | None,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """(V_plus, V0) stacks over ``keys`` (row-aligned) from one subset namespace,
    with the consumer-side probe_idx / retained-count guard per cell."""
    vp_rows, v0_rows = [], []
    for key in keys:
        src, tgt = key.split("/", 1)[1].split("__")
        p = root / BEHAVIOR / f"{src}_seed{SEED}" / f"{tgt}_L{layer}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{label}: {p} missing — Phase-N1 namespace incomplete")
        d = np.load(p, allow_pickle=True)
        probe_ids = [int(i) for i in np.asarray(d["probe_idx"]).tolist()]
        if expected_probe_ids is not None:
            want = sorted(int(i) for i in expected_probe_ids[key])
            if probe_ids != want:
                raise RuntimeError(
                    f"{label}: {p.name} probe_idx {probe_ids[:5]}... != committed sample "
                    f"indices {want[:5]}... — sample-provenance mismatch (plan §4(b))"
                )
        if manifest_cells is not None and len(probe_ids) != manifest_cells[key]["retained"]:
            raise RuntimeError(
                f"{label}: {p.name} has {len(probe_ids)} probes != manifest retained "
                f"{manifest_cells[key]['retained']} (plan §4(b))"
            )
        vp_rows.append(np.asarray(d["v_plus_onpolicy"], dtype=np.float64))
        v0_rows.append(np.asarray(d["v0_onpolicy"], dtype=np.float64))
    return np.stack(vp_rows), np.stack(v0_rows)


def _load_fulltext_override(
    root: Path, keys: list[str], layer: int
) -> tuple[np.ndarray, np.ndarray]:
    """(Von, V0on) comparator legs from the parity-FAIL contingency's within-run
    full-text re-extraction namespace (no probe-idx guard — full 30-row legs)."""
    vp_rows, v0_rows = [], []
    for key in keys:
        src, tgt = key.split("/", 1)[1].split("__")
        p = root / BEHAVIOR / f"{src}_seed{SEED}" / f"{tgt}_L{layer}.npz"
        if not p.exists():
            raise FileNotFoundError(f"fulltext override: {p} missing")
        d = np.load(p, allow_pickle=True)
        vp_rows.append(np.asarray(d["v_plus_onpolicy"], dtype=np.float64))
        v0_rows.append(np.asarray(d["v0_onpolicy"], dtype=np.float64))
    return np.stack(vp_rows), np.stack(v0_rows)


def load_retained_design(args, layer: int) -> dict:
    """Joined-cache stacks + subset stacks, all subset to the retained cells.

    Asserts (plan §4 Phase N2): retained keys == manifest floor-passers ==
    matched-N index keys == eq5 index keys; all 7 families represented; every
    source retains ≥1 cell.
    """
    out_dir: Path = args.out_dir
    cache_path = out_dir / "joined_cache" / f"{BEHAVIOR}_L{layer}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} missing — rebuild from HF via issue833_fit_onpolicy.py --joined-cache"
        )
    d = np.load(cache_path, allow_pickle=True)
    cell_keys = [str(v) for v in d["cell_keys"].tolist()]
    families = [str(v) for v in d["families"].tolist()]

    manifest = json.loads(Path(args.retention_manifest).read_text())
    matched_idx = json.loads(Path(args.matchedn_indices).read_text())["cells"]
    eq5_idx = json.loads(Path(args.eq5_indices).read_text())["cells"]
    retained = {k for k, r in manifest["cells"].items() if not r["below_floor"]}
    assert retained == set(matched_idx) == set(eq5_idx), (
        "retained-cell sets disagree across manifest / matchedN / eq5 index files"
    )
    mask = np.asarray([k in retained for k in cell_keys])
    keys_ret = [k for k, m in zip(cell_keys, mask, strict=True) if m]
    fams_ret = [f for f, m in zip(families, mask, strict=True) if m]
    assert len(keys_ret) == len(retained), (len(keys_ret), len(retained))
    fams_present = set(fams_ret)
    all_fams = set(families)
    if fams_present != all_fams:
        raise RuntimeError(f"families {sorted(all_fams - fams_present)} lost all retained cells")
    srcs_ret = {k.split("/", 1)[1].split("__")[0] for k in keys_ret}
    srcs_all = {k.split("/", 1)[1].split("__")[0] for k in cell_keys}
    if srcs_ret != srcs_all:
        raise RuntimeError(f"sources {sorted(srcs_all - srcs_ret)} lost all retained cells")

    stacks = {
        k: np.asarray(d[k], dtype=np.float64)[mask]
        for k in ("C0", "Cplus", "V0", "Vplus", "Von", "V0on")
    }
    fulltext_source = "joined_cache"
    if args.fulltext_npz_root:
        stacks["Von"], stacks["V0on"] = _load_fulltext_override(
            Path(args.fulltext_npz_root), keys_ret, layer
        )
        fulltext_source = str(args.fulltext_npz_root)

    von_ne, v0on_ne = _load_subset_stack(
        args.nonemit_root,
        keys_ret,
        layer,
        expected_probe_ids=None,
        manifest_cells=manifest["cells"],
        label="nonemit",
    )
    von_mn, _ = _load_subset_stack(
        args.matchedn_root,
        keys_ret,
        layer,
        expected_probe_ids=matched_idx,
        manifest_cells=None,
        label="matchedN",
    )
    von_e5, _ = _load_subset_stack(
        args.eq5_root,
        keys_ret,
        layer,
        expected_probe_ids=eq5_idx,
        manifest_cells=None,
        label="eq5",
    )
    E = fitM._load_E(BEHAVIOR, keys_ret)
    return {
        "keys": keys_ret,
        "families": fams_ret,
        "E": E,
        "stacks": stacks,
        "Von_nonemit": von_ne,
        "V0on_nonemit": v0on_ne,
        "Von_matchedN": von_mn,
        "Von_eq5": von_e5,
        "cache_sha256": _sha256_file(cache_path),
        "fulltext_source": fulltext_source,
    }


def run_layer(design: dict, layer: int, r_hat: np.ndarray) -> dict:
    """One layer: 8 retained-subset ridge-LOCO chains + marginal CIs + the
    registered paired diffs (family-clustered, 1,000 resamples, seed 0)."""
    st = design["stacks"]
    E = design["E"]
    keep = ~np.isnan(E)
    if keep.sum() < 4:
        raise RuntimeError(f"L{layer}: only {int(keep.sum())} retained cells with E (<4)")
    Ek = E[keep]
    fam_k = [f for f, m in zip(design["families"], keep, strict=True) if m]

    pca = fitM._pca_basis_v0(st["V0"], TARGET_DIM)  # retained-subset shared basis
    arms_spec = {
        "on_nonemit": (st["Cplus"], design["Von_nonemit"]),
        "ctrl_nonemit": (st["C0"], design["V0on_nonemit"]),
        "on_full_ret": (st["Cplus"], st["Von"]),
        "on_full_matchedN": (st["Cplus"], design["Von_matchedN"]),
        "ctrl_full_ret": (st["C0"], st["V0on"]),
        "off_full_ret": (st["Cplus"], st["Vplus"]),
        "base_full_ret": (st["C0"], st["V0"]),
        "on_nonemit_eq5": (st["Cplus"], design["Von_eq5"]),
    }
    arms_out: dict[str, dict] = {}
    chains: dict[str, np.ndarray] = {}
    for arm, (X, V) in arms_spec.items():
        t0 = time.perf_counter()
        loco = fitM._ridge_loco_pred(X, V @ pca.T)
        rho, chain = fitM._chain_rho_one(loco[keep], pca, r_hat, Ek)
        entry: dict = {"rho_ridge": rho}
        if rho is not None:
            entry["ci_ridge"] = clustered_bootstrap_spearman(chain, Ek, fam_k)
            chains[arm] = chain
        arms_out[arm] = entry
        logger.info(
            "[phase=chain_nonemit] L%d arm %s: rho=%s (%.1fs)",
            layer,
            arm,
            "None" if rho is None else f"{rho:+.4f}",
            time.perf_counter() - t0,
        )
    diffs: dict[str, dict | None] = {}
    for name, minuend, subtrahend in PAIRED_DIFFS:
        if minuend in chains and subtrahend in chains:
            # _clustered_paired_rho_diff_ci(a, b, ...) returns rho(b) − rho(a):
            # pass (subtrahend, minuend) so the point is minuend − subtrahend.
            diffs[name] = fitM._clustered_paired_rho_diff_ci(
                chains[subtrahend], chains[minuend], Ek, fam_k
            )
        else:
            diffs[name] = None
    return {
        "behavior": BEHAVIOR,
        "layer": layer,
        "n_retained_cells": len(design["keys"]),
        "n_with_E": int(keep.sum()),
        "arms": arms_out,
        "paired_diffs": diffs,
        "primary_diff": "on_nonemit_minus_on_full_matchedN",
        "headline_layer": 14,
        "mlp_gate": "not_run",  # ctrl-arm precedent (plan §2)
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #833 Phase N2 — non-emission chains vs E")
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS_DEFAULT))
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_833")
    ap.add_argument("--nonemit-root", type=Path, default=None)
    ap.add_argument("--matchedn-root", type=Path, default=None)
    ap.add_argument("--eq5-root", type=Path, default=None)
    ap.add_argument("--retention-manifest", default=None)
    ap.add_argument("--matchedn-indices", default=None)
    ap.add_argument("--eq5-indices", default=None)
    ap.add_argument(
        "--fulltext-npz-root",
        default=None,
        help="parity-FAIL contingency: read the Von/V0on comparator legs from this "
        "within-run full-text namespace instead of the r7e joined cache",
    )
    ap.add_argument(
        "--stage-from-hf",
        action="store_true",
        help="stage missing subset namespaces from HF (scoped list_repo_tree + per-file pool)",
    )
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    for attr, default in (
        ("nonemit_root", out_dir / SUBSET_NAMESPACES["nonemit"]),
        ("matchedn_root", out_dir / SUBSET_NAMESPACES["matchedn"]),
        ("eq5_root", out_dir / SUBSET_NAMESPACES["eq5"]),
    ):
        if getattr(args, attr) is None:
            setattr(args, attr, default)
    for attr, default in (
        ("retention_manifest", out_dir / "emission_rate" / "retention_manifest.json"),
        ("matchedn_indices", out_dir / "emission_rate" / "matchedN_sample_indices.json"),
        ("eq5_indices", out_dir / "emission_rate" / "eq5_sample_indices.json"),
    ):
        if getattr(args, attr) is None:
            setattr(args, attr, default)

    if args.stage_from_hf:
        for subset, root in (
            ("nonemit", args.nonemit_root),
            ("matchedn", args.matchedn_root),
            ("eq5", args.eq5_root),
        ):
            if not Path(root).is_dir():
                stage_namespace_from_hf(Path(root), SUBSET_NAMESPACES[subset])

    fit658.DEVICE = fit658._resolve_device("auto")
    fit658._assert_ridge_exactness()  # startup exactness gate (fit_M precedent)
    logger.info("[phase=chain_nonemit] ridge exactness gate PASS (device=%s)", fit658.DEVICE)
    fitM.TARGET_DIM = TARGET_DIM

    rb_fact = fitM._load_rb_fact()
    if rb_fact is None:
        raise RuntimeError("r_b_fact.pt unavailable/degenerate — fact chains need it")
    rb_main = fitM._load_rb_main()
    manifest_sha = _sha256_file(Path(args.retention_manifest))

    meta_common = {
        "script": "scripts/issue833_chain_rho_nonemit.py",
        "git_commit": _git_head(),
        "generated_at": datetime.now(UTC).isoformat(),
        "numpy": np.__version__,
        "ridge_device": fit658.DEVICE,
        "target_dim": TARGET_DIM,
        "pca_basis": "fitM._pca_basis_v0 on the RETAINED-subset V0 (shared across arms)",
        "ridge_lambdas": list(fit658.RIDGE_LAMBDAS),
        "n_bootstrap_resamples": 1000,
        "bootstrap_seed": 0,
        "retention_manifest_sha256": manifest_sha,
    }
    t_start = time.perf_counter()
    for layer in args.layers:
        out_path = out_dir / "chain_rho_nonemit" / f"{BEHAVIOR}_L{layer}.json"
        cache_path = out_dir / "joined_cache" / f"{BEHAVIOR}_L{layer}.npz"
        cache_sha = _sha256_file(cache_path) if cache_path.exists() else "absent"
        resume_key = f"{cache_sha}:{manifest_sha}"
        if not args.force_rerun and _out_is_complete(out_path, resume_key):
            logger.info("[phase=chain_nonemit] L%d already complete — skip (resume)", layer)
            continue
        design = load_retained_design(args, layer)
        r_hat = fitM._r_hat_for(BEHAVIOR, layer, rb_main, rb_fact)
        t_cell = time.perf_counter()
        block = run_layer(design, layer, r_hat)
        block["meta"] = {
            **meta_common,
            "resume_key": resume_key,
            "joined_cache_sha256": design["cache_sha256"],
            "fulltext_comparator_source": design["fulltext_source"],
            "nonemit_root": str(args.nonemit_root),
            "matchedn_root": str(args.matchedn_root),
            "eq5_root": str(args.eq5_root),
            "layer_wall_seconds": round(time.perf_counter() - t_cell, 1),
        }
        _write_json(out_path, block)  # checkpoint-per-layer
        primary = block["paired_diffs"]["on_nonemit_minus_on_full_matchedN"]
        logger.info(
            "[phase=chain_nonemit] L%d DONE: on_nonemit=%s matchedN=%s PRIMARY diff=%s "
            "(%.1fs; wrote %s)",
            layer,
            json.dumps(block["arms"]["on_nonemit"]["rho_ridge"]),
            json.dumps(block["arms"]["on_full_matchedN"]["rho_ridge"]),
            json.dumps((primary or {}).get("point")),
            time.perf_counter() - t_cell,
            out_path,
        )
    logger.info("[phase=chain_nonemit] ALL DONE in %.1f min", (time.perf_counter() - t_start) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
