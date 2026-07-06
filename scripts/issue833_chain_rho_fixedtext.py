#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, M⁺, →) in scientific docstrings + log messages.
"""Issue #833 follow-up (fixed-template-weights-read) — Phase F2: fixed-template
chains + paired diffs (plan v13 §4, VM CPU, 0 GPU-h).

Fits SIX ridge PRESS-LOCO arms per fact layer:

  FULL-480 fit set (shared basis ``fitM._pca_basis_v0(V0_480, 64)`` — the
  production convention; n=480 > 64):
    on_fixedtext      (C⁺, v⁺(R_fixed))  — the manipulated arm
    ctrl_fixedtext    (C0, v0(R_fixed))  — base-weights control on IDENTICAL text
                                           (source-constant predictions BY DESIGN
                                           — the parent's M0 convention; plan §4)
    off_full_recomp   (C⁺, Vplus)        — recomputed off-policy anchor,
                                           consistency-asserted vs the committed
                                           ``rho_Mplus_off_ridge`` (≤0.02 tol,
                                           the chain_rho_ctrl precedent)
    base_full_recomp  (C0, V0)           — recomputed base-map floor (vs
                                           ``rho_M0_ridge``)
  RETAINED-291 fit set (shared basis ``_pca_basis_v0(V0_RETAINED, 64)`` — the
  round-2 committed retention manifest's floor-passers; byte-identical basis
  construction to round 2, for direct comparability with +0.215):
    on_fixedtext_ret / ctrl_fixedtext_ret

Paired family-clustered diffs (1,000 resamples, seed 0 — verbatim estimator
reuse): PRIMARY ``on_fixedtext − ctrl_fixedtext`` @ headline layer L14 on 480
cells; ``on_fixedtext − off_full_recomp`` (the carrier-independence
adjudicator); ``on_fixedtext_ret − ctrl_fixedtext_ret`` (direct comparability
with round 2's +0.215); ``ctrl_fixedtext − base_full_recomp`` (exploratory:
constant-vs-varied text on base weights).

Guards (plan §4 F2, STOP on any miss): fixedtext npz key set == the full
480-cell grid == joined-cache keys; every npz has exactly 30 probes and every
``resp_sha256`` row == the template pin; retained-slice keys == the committed
round-2 retention manifest's floor-passers; all 7 families + 16 sources present
in BOTH fit sets. Per-cell held-out chain predictions are PERSISTED per arm
(``per_cell_full`` / ``per_cell_retained``) — the analyzer's per-source
leverage / leave-sp_swe-out / source-demeaned companion reads consume them.

Outputs: ``<out-dir>/chain_rho_fixedtext/fact_L{7,14,21}.json`` (atomic
per-layer writes the moment each layer completes; resume keyed on joined-cache
sha + template sha + retention-manifest sha, ``--force-rerun`` override).
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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import issue833_emission_rate as emrate  # noqa: E402

from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    clustered_bootstrap_spearman,
)

logger = logging.getLogger("issue833.chain_fixedtext")

BEHAVIOR = "fact"
LAYERS_DEFAULT = (7, 14, 21)
SEED = 42
TARGET_DIM = 64  # the production shared top-v0-PCA target dim
N_CELLS_FULL = 480  # 16 sources × 30 targets — the full grid (plan §2)
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue833_onpolicy_map"
FIXEDTEXT_NAMESPACE = "analysis_tensors_fixedtext"
CONSISTENCY_FAIL_TOL = 0.02  # chain_rho_ctrl precedent (≤0.003 drift measured)
# Committed-anchor keys for the recomputed comparator arms (chain_rho/fact_L*.json).
COMMITTED_ANCHOR_KEYS = {
    "off_full_recomp": "rho_Mplus_off_ridge",
    "base_full_recomp": "rho_M0_ridge",
}
_REQUIRED_OUT_KEYS = frozenset({"arms", "paired_diffs", "per_cell_full", "meta"})
# Registered paired diffs: (name, minuend arm, subtrahend arm). PRIMARY first.
PAIRED_DIFFS_FULL = (
    ("on_fixedtext_minus_ctrl_fixedtext", "on_fixedtext", "ctrl_fixedtext"),  # PRIMARY @ L14
    ("on_fixedtext_minus_off_full_recomp", "on_fixedtext", "off_full_recomp"),
    ("ctrl_fixedtext_minus_base_full_recomp", "ctrl_fixedtext", "base_full_recomp"),
)
PAIRED_DIFFS_RET = (
    ("on_fixedtext_ret_minus_ctrl_fixedtext_ret", "on_fixedtext_ret", "ctrl_fixedtext_ret"),
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
    """Resume predicate: complete output pinned to the SAME (cache, pin, manifest) triple."""
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return False
    return _REQUIRED_OUT_KEYS.issubset(obj) and obj.get("meta", {}).get("resume_key") == resume_key


# ─────────────────────────────────────────────────────────────────────────────
# Pure guards (plan §4 F2 — unit-tested on synthetic fixtures)
# ─────────────────────────────────────────────────────────────────────────────


def assert_key_cover(got: set[str], want: set[str], label: str) -> None:
    """STOP on ANY key-set mismatch — missing OR extra cells (plan §1 row coverage)."""
    missing = sorted(want - got)
    extra = sorted(got - want)
    if missing or extra:
        raise RuntimeError(
            f"{label}: key-set mismatch — {len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(extra)} extra (e.g. {extra[:3]}) — STOP (plan v13 §6 kill criterion)"
        )


def assert_families_sources(
    keys: list[str],
    families: list[str],
    *,
    n_families: int = 7,
    n_sources: int = 16,
    label: str,
) -> None:
    """Both fit sets must carry all 7 target families + all 16 sources (plan §4 F2)."""
    fams = set(families)
    srcs = {k.split("/", 1)[1].split("__")[0] for k in keys}
    if len(fams) != n_families or len(srcs) != n_sources:
        raise RuntimeError(
            f"{label}: {len(fams)} families / {len(srcs)} sources present, expected "
            f"{n_families}/{n_sources} — STOP (plan §4 F2 guard)"
        )


def load_template_pin(path: Path) -> dict:
    """Template-pin guard (the F2 copy of the pod-side `_load_fixed_template` check):
    file self-consistency (sha256(template) == recorded sha) AND equality with the
    CODE-side pin ``emrate.FIXED_TEMPLATE_SHA256``. RuntimeError on either miss."""
    pin = json.loads(path.read_text())
    got = hashlib.sha256(pin["template"].encode("utf-8")).hexdigest()
    if got != pin["sha256"] or pin["sha256"] != emrate.FIXED_TEMPLATE_SHA256:
        raise RuntimeError(
            f"template pin guard FAIL: sha256(template)={got[:12]}, recorded="
            f"{pin['sha256'][:12]}, plan pin={emrate.FIXED_TEMPLATE_SHA256[:12]} — STOP"
        )
    return pin


# ─────────────────────────────────────────────────────────────────────────────
# Staging + loading
# ─────────────────────────────────────────────────────────────────────────────


def stage_namespace_from_hf(root: Path, namespace: str) -> None:
    """Stage the fixedtext namespace from HF (scoped list_repo_tree + threaded
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
        raise FileNotFoundError(f"HF prefix {prefix} is empty — run Phase F1 first")
    logger.info("[phase=stage] %s: %d files from HF", namespace, len(paths))
    stage_dir = root.parent / "hf_stage_fixedtext"

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


def _fixedtext_key_set(root: Path, layer: int) -> set[str]:
    """Enumerate the namespace's cell keys at one layer (for the coverage guard)."""
    got: set[str] = set()
    suffix = f"_seed{SEED}"
    for cd in sorted((root / BEHAVIOR).glob(f"*{suffix}")):
        src = cd.name[: -len(suffix)]
        for p in cd.glob(f"*_L{layer}.npz"):
            got.add(f"{BEHAVIOR}/{src}__{p.stem.rsplit('_L', 1)[0]}")
    return got


def load_fixedtext_stack(
    root: Path, keys: list[str], layer: int, *, template_sha: str
) -> tuple[np.ndarray, np.ndarray]:
    """(v⁺(R_fixed), v0(R_fixed)) stacks over ``keys`` (row-aligned), with the
    plan-§4 guards: key-set == the full grid; exactly 30 probes per cell; every
    ``resp_sha256`` row == the template pin."""
    assert_key_cover(_fixedtext_key_set(root, layer), set(keys), f"fixedtext L{layer}")
    vp_rows, v0_rows = [], []
    for key in keys:
        src, tgt = key.split("/", 1)[1].split("__")
        p = root / BEHAVIOR / f"{src}_seed{SEED}" / f"{tgt}_L{layer}.npz"
        d = np.load(p, allow_pickle=True)
        shas = [str(s) for s in np.asarray(d["resp_sha256"]).tolist()]
        if len(shas) != 30:
            raise RuntimeError(
                f"fixedtext {p.name}: {len(shas)} probes != 30 — the template is defined for "
                "every probe by construction (plan §5), STOP"
            )
        n_bad = sum(1 for s in shas if s != template_sha)
        if n_bad:
            raise RuntimeError(
                f"fixedtext {p.name}: {n_bad}/{len(shas)} resp_sha256 rows != the template "
                f"pin {template_sha[:12]} — STOP (plan §6 kill criterion)"
            )
        vp_rows.append(np.asarray(d["v_plus_onpolicy"], dtype=np.float64))
        v0_rows.append(np.asarray(d["v0_onpolicy"], dtype=np.float64))
    return np.stack(vp_rows), np.stack(v0_rows)


def load_design(args, layer: int) -> dict:
    """Joined-cache stacks + fixedtext stacks + the retained-slice mask.

    Asserts (plan §4 F2): 480 joined-cache cells; fixedtext coverage == the full
    grid; retained keys == the committed round-2 manifest's floor-passers;
    all 7 families + 16 sources in BOTH fit sets.
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
    if len(cell_keys) != N_CELLS_FULL:
        raise RuntimeError(f"joined cache has {len(cell_keys)} cells != {N_CELLS_FULL} — STOP")
    assert_families_sources(cell_keys, families, label="full-480 fit set")

    pin = load_template_pin(Path(args.fixed_template_file))
    manifest = json.loads(Path(args.retention_manifest).read_text())
    retained = {k for k, r in manifest["cells"].items() if not r["below_floor"]}
    mask_ret = np.asarray([k in retained for k in cell_keys])
    assert_key_cover(
        {k for k, m in zip(cell_keys, mask_ret, strict=True) if m},
        retained,
        "retained-291 slice vs the round-2 manifest",
    )
    keys_ret = [k for k, m in zip(cell_keys, mask_ret, strict=True) if m]
    fams_ret = [f for f, m in zip(families, mask_ret, strict=True) if m]
    assert_families_sources(keys_ret, fams_ret, label="retained-291 fit set")

    stacks = {k: np.asarray(d[k], dtype=np.float64) for k in ("C0", "Cplus", "V0", "Vplus")}
    vfix_p, vfix_0 = load_fixedtext_stack(
        Path(args.fixedtext_root), cell_keys, layer, template_sha=pin["sha256"]
    )
    E = fitM._load_E(BEHAVIOR, cell_keys)
    return {
        "keys": cell_keys,
        "families": families,
        "E": E,
        "stacks": stacks,
        "Vfix_plus": vfix_p,
        "Vfix_0": vfix_0,
        "mask_ret": mask_ret,
        "keys_ret": keys_ret,
        "fams_ret": fams_ret,
        "cache_sha256": _sha256_file(cache_path),
        "template_sha256": pin["sha256"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fits
# ─────────────────────────────────────────────────────────────────────────────


def run_fit_set(
    keys: list[str],
    families: list[str],
    E: np.ndarray,
    arms_spec: dict[str, tuple[np.ndarray, np.ndarray]],
    pca: np.ndarray,
    r_hat: np.ndarray,
    paired: tuple,
    layer: int,
    label: str,
) -> tuple[dict, dict, dict]:
    """One fit set: batched ridge PRESS-LOCO per arm (`fitM._ridge_loco_pred`,
    the #811 batched path) + family-clustered marginal CIs + the registered
    paired diffs. Returns (arms_out, diffs, per_cell)."""
    keep = ~np.isnan(E)
    if keep.sum() < 4:
        raise RuntimeError(f"L{layer} {label}: only {int(keep.sum())} cells with E (<4)")
    Ek = E[keep]
    fam_k = [f for f, m in zip(families, keep, strict=True) if m]
    keys_k = [k for k, m in zip(keys, keep, strict=True) if m]
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
            "[phase=chain_fixedtext] L%d %s arm %s: rho=%s (%.1fs)",
            layer,
            label,
            arm,
            "None" if rho is None else f"{rho:+.4f}",
            time.perf_counter() - t0,
        )
    diffs: dict[str, dict | None] = {}
    for name, minuend, subtrahend in paired:
        if minuend in chains and subtrahend in chains:
            # _clustered_paired_rho_diff_ci(a, b, ...) returns rho(b) − rho(a):
            # pass (subtrahend, minuend) so the point is minuend − subtrahend.
            diffs[name] = fitM._clustered_paired_rho_diff_ci(
                chains[subtrahend], chains[minuend], Ek, fam_k
            )
        else:
            diffs[name] = None
    per_cell = {
        "keys": keys_k,
        "families": fam_k,
        "E": [float(v) for v in Ek],
        "chains": {arm: [float(v) for v in chain] for arm, chain in chains.items()},
    }
    return arms_out, diffs, per_cell


def assert_committed_consistency(out_dir: Path, layer: int, arms_out: dict) -> dict:
    """Recomputed off/base anchors must match the committed chain_rho JSONs
    (≤0.02 tol — the chain_rho_ctrl precedent; measured drift ≤0.003)."""
    committed = json.loads((out_dir / "chain_rho" / f"{BEHAVIOR}_L{layer}.json").read_text())
    consistency: dict[str, dict] = {}
    for arm, key in COMMITTED_ANCHOR_KEYS.items():
        want = committed[key]
        got = arms_out[arm]["rho_ridge"]
        delta = abs(float(got) - float(want)) if (got is not None and want is not None) else None
        consistency[arm] = {"recomputed": got, "committed": want, "abs_delta": delta}
        if delta is None or delta > CONSISTENCY_FAIL_TOL:
            raise RuntimeError(
                f"fact L{layer} recomputed {arm} {got} vs committed {key}={want} "
                f"(|Δ|={delta}) exceeds {CONSISTENCY_FAIL_TOL} — joined cache / regime "
                "mismatch with the committed run; refusing to compare against it"
            )
    return consistency


def run_layer(design: dict, layer: int, r_hat: np.ndarray, out_dir: Path) -> dict:
    """One layer: the 4 full-480 arms + the 2 retained-291 arms + all four
    registered paired diffs + the committed-anchor consistency assert."""
    st = design["stacks"]
    pca_full = fitM._pca_basis_v0(st["V0"], TARGET_DIM)  # production full-grid basis
    arms_full, diffs_full, per_cell_full = run_fit_set(
        design["keys"],
        design["families"],
        design["E"],
        {
            "on_fixedtext": (st["Cplus"], design["Vfix_plus"]),
            "ctrl_fixedtext": (st["C0"], design["Vfix_0"]),
            "off_full_recomp": (st["Cplus"], st["Vplus"]),
            "base_full_recomp": (st["C0"], st["V0"]),
        },
        pca_full,
        r_hat,
        PAIRED_DIFFS_FULL,
        layer,
        "full-480",
    )
    consistency = assert_committed_consistency(out_dir, layer, arms_full)

    m = design["mask_ret"]
    pca_ret = fitM._pca_basis_v0(st["V0"][m], TARGET_DIM)  # round-2 retained-slice basis
    arms_ret, diffs_ret, per_cell_ret = run_fit_set(
        design["keys_ret"],
        design["fams_ret"],
        design["E"][m],
        {
            "on_fixedtext_ret": (st["Cplus"][m], design["Vfix_plus"][m]),
            "ctrl_fixedtext_ret": (st["C0"][m], design["Vfix_0"][m]),
        },
        pca_ret,
        r_hat,
        PAIRED_DIFFS_RET,
        layer,
        "retained-291",
    )
    return {
        "behavior": BEHAVIOR,
        "layer": layer,
        "n_cells_full": len(design["keys"]),
        "n_with_E_full": len(per_cell_full["keys"]),
        "n_retained_cells": len(design["keys_ret"]),
        "n_with_E_retained": len(per_cell_ret["keys"]),
        "arms": {**arms_full, **arms_ret},
        "paired_diffs": {**diffs_full, **diffs_ret},
        "primary_diff": "on_fixedtext_minus_ctrl_fixedtext",
        "headline_layer": 14,
        "mlp_gate": "not_run",  # ctrl-arm precedent (plan v10 §11.6, carried v13 §2)
        "consistency_vs_committed": consistency,
        "per_cell_full": per_cell_full,
        "per_cell_retained": per_cell_ret,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #833 Phase F2 — fixed-template chains vs E")
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS_DEFAULT))
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_833")
    ap.add_argument("--fixedtext-root", type=Path, default=None)
    ap.add_argument("--fixed-template-file", default=None)
    ap.add_argument("--retention-manifest", default=None)
    ap.add_argument(
        "--chain-out-dir",
        type=Path,
        default=None,
        help="output dir override (default <out-dir>/chain_rho_fixedtext; smoke runs "
        "redirect here so committed artifacts are never touched)",
    )
    ap.add_argument(
        "--stage-from-hf",
        action="store_true",
        help="stage the fixedtext namespace from HF (scoped list_repo_tree + per-file pool)",
    )
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    if args.fixedtext_root is None:
        args.fixedtext_root = out_dir / FIXEDTEXT_NAMESPACE
    if args.fixed_template_file is None:
        args.fixed_template_file = out_dir / "emission_rate" / "fixed_template.json"
    if args.retention_manifest is None:
        args.retention_manifest = out_dir / "emission_rate" / "retention_manifest.json"
    chain_out_dir: Path = args.chain_out_dir or (out_dir / "chain_rho_fixedtext")

    if args.stage_from_hf and not Path(args.fixedtext_root).is_dir():
        stage_namespace_from_hf(Path(args.fixedtext_root), FIXEDTEXT_NAMESPACE)

    fit658.DEVICE = fit658._resolve_device("auto")
    fit658._assert_ridge_exactness()  # startup exactness gate (fit_M precedent)
    logger.info("[phase=chain_fixedtext] ridge exactness gate PASS (device=%s)", fit658.DEVICE)
    fitM.TARGET_DIM = TARGET_DIM

    rb_fact = fitM._load_rb_fact()
    if rb_fact is None:
        raise RuntimeError("r_b_fact.pt unavailable/degenerate — fact chains need it")
    rb_main = fitM._load_rb_main()
    pin_sha = load_template_pin(Path(args.fixed_template_file))["sha256"]
    manifest_sha = _sha256_file(Path(args.retention_manifest))

    # Committed anchors quoted alongside (plan §4 F2 — no refit needed).
    committed_anchors = {}
    for layer in args.layers:
        p = out_dir / "chain_rho" / f"{BEHAVIOR}_L{layer}.json"
        if p.exists():
            c = json.loads(p.read_text())
            committed_anchors[f"L{layer}"] = {
                k: c.get(k) for k in ("rho_Mplus_on_ridge", "rho_Mplus_off_ridge", "rho_M0_ridge")
            }

    meta_common = {
        "script": "scripts/issue833_chain_rho_fixedtext.py",
        "git_commit": _git_head(),
        "generated_at": datetime.now(UTC).isoformat(),
        "numpy": np.__version__,
        "ridge_device": fit658.DEVICE,
        "target_dim": TARGET_DIM,
        "pca_basis": (
            "fitM._pca_basis_v0 on V0_480 (full-grid arms) / V0_RETAINED (ret arms), "
            "shared across arms within each fit set"
        ),
        "ridge_lambdas": list(fit658.RIDGE_LAMBDAS),
        "n_bootstrap_resamples": 1000,
        "bootstrap_seed": 0,
        "template_sha256": pin_sha,
        "retention_manifest_sha256": manifest_sha,
        "committed_anchors": committed_anchors,
    }
    t_start = time.perf_counter()
    for layer in args.layers:
        out_path = chain_out_dir / f"{BEHAVIOR}_L{layer}.json"
        cache_path = out_dir / "joined_cache" / f"{BEHAVIOR}_L{layer}.npz"
        cache_sha = _sha256_file(cache_path) if cache_path.exists() else "absent"
        resume_key = f"{cache_sha}:{pin_sha}:{manifest_sha}"
        if not args.force_rerun and _out_is_complete(out_path, resume_key):
            logger.info("[phase=chain_fixedtext] L%d already complete — skip (resume)", layer)
            continue
        design = load_design(args, layer)
        r_hat = fitM._r_hat_for(BEHAVIOR, layer, rb_main, rb_fact)
        t_cell = time.perf_counter()
        block = run_layer(design, layer, r_hat, out_dir)
        block["meta"] = {
            **meta_common,
            "resume_key": resume_key,
            "joined_cache_sha256": design["cache_sha256"],
            "fixedtext_root": str(args.fixedtext_root),
            "layer_wall_seconds": round(time.perf_counter() - t_cell, 1),
        }
        _write_json(out_path, block)  # checkpoint-per-layer
        primary = block["paired_diffs"]["on_fixedtext_minus_ctrl_fixedtext"]
        logger.info(
            "[phase=chain_fixedtext] L%d DONE: on_fixedtext=%s ctrl_fixedtext=%s "
            "PRIMARY diff=%s (%.1fs; wrote %s)",
            layer,
            json.dumps(block["arms"]["on_fixedtext"]["rho_ridge"]),
            json.dumps(block["arms"]["ctrl_fixedtext"]["rho_ridge"]),
            json.dumps((primary or {}).get("point")),
            time.perf_counter() - t_cell,
            out_path,
        )
    logger.info(
        "[phase=chain_fixedtext] ALL DONE in %.1f min", (time.perf_counter() - t_start) / 60
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
