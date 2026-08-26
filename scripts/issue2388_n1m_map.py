#!/usr/bin/env python
"""#2388 n1m-map round: re-score the mapped correctness-probe arm through the banked n1m map.

User-approved inline free-analysis round on #2388 (decision record: the
`epm:progress` v927 dispatch note on task #2388). The parent's linear
mapped-answer arm (``arm_maplin``) pushed base context states through the
f_U=1 GENERIC-pool ridge map (the 18,793-pair #1092 U-store fit). This round
re-runs ONLY that arm with the banked n1m ridge substituted — the ``mixed_1m``
point of #779 ``fitter-fair-comparison-n1m``: 963,444 real contexts (LMSYS
529,085 + WildChat 434,359), banked at layers 14/19/26 on the HF data repo
under ``issue779_monitoring/n1m_readout/weights/L{14,19,26}/ridge.pt``. Both
maps use the identical registered prediction path
``vhat = ((v_C - xmu)/xsd) @ W + ymu`` (``issue2474_n1m_map.load_n1m_comp`` /
``issue779_ffc_n1m_fits.apply_map`` agree line-for-line), same base model,
same d=3,584 — a drop-in payload substitution, no refit.

RECIPE REUSE — import-at-pinned-SHA, never a re-derivation. Estimator diff vs
the two named references (round-brief requirement):

* vs ``scripts/issue2388_fits.py`` @ PARENT_PIN (the #2388 body-Repro final
  late-fix SHA; 653c1d9d18's group-grain disjoint pre-check is an ancestor,
  subsumed by the pin): the table loading, group-respecting label draws
  (IDENTICAL seed parts -> identical draw rows per (surface, budget, draw)),
  PCA bases, dof-capped-GCV readout (cap 0.9, no LEGACY_UNGUARDED_GCV),
  dev-selection, preds persistence, selection freeze (``phase_select``) and
  paired group bootstrap (``phase_bootstrap``) are the parent's OWN functions,
  loaded from git at the pin. The ONLY changes: (1) the map payload — banked
  n1m ridge instead of the fu1 pool-fit map; (2) the layer set restricted to
  the three banked n1m layers (14, 19, 26) (parent dev-selected L15/16 —
  layer-set mismatch is a disclosed scope caveat); (3) the arm roster is the
  single linear mapped arm ``arm_n1m``; (4) no permutation-null target
  columns (fresh nulls are OUT of scope per the decision record — the
  observed DV is the only target); (5) eval sets are dev + rung0 (the locked
  test split) — the rung1 transfer read requires the parent's restricted-fit
  refit protocol and is out of scope this round, so staged parent preds are
  filtered to the same two eval sets to keep the bootstrap paired; (6) the
  code benchmark roster is derived from the realized labeling rows (the
  parent asserts exact-set equality between that set and the gate roster, so
  the two are equal by the parent's own invariant — avoids pinning
  ``issue2388_gen.py``).
* vs ``scripts/issue2474_n1m_map.py`` (the committed drop-in precedent): the
  payload location, validation (kind/fitter/layer asserts), and prediction
  path are the same; the consumer differs (correctness-probe ridge readout
  fits here vs #2474's cosine-score arms).

DOF-CAPPED CORE PIN: the ``dof_cap`` + ``selector_telemetry`` kwargs of
``ridge_gcv_predict_per_target`` exist ONLY on branch issue-2388 (commit
1f44fbb8a6, unmerged), so the #1739 fits core is ALSO read-pinned from git at
PARENT_PIN and rebound as the parent module's ``F``. Main-resident modules the
pinned code still binds to were drift-checked against the branch:
``store_io.py`` byte-identical; the four ``arms`` helpers (spearman_rows /
_pearson_rows / rank_rows / auroc_rows) byte-identical; ``constants`` names
all present with identical values when ``EPS_I1739_RIDGE_LAMBDAS`` is unset
(this script never sets it).

PAIRED BOOTSTRAP: the parent's banked test predictions for the direct probe
(``arm_ctx``) and the old map (``arm_maplin``) — committed on branch
issue-2388 under ``eval_results/issue_2388/fits/<surface>/preds/`` and
mirrored on HF under ``issue2388_correctness/fits`` — are staged (git archive
of the branch is the source the launcher uses; the pod clone reaches it via
``git fetch origin issue-2388``), filtered to dev+rung0, and bootstrapped
JOINTLY with the new arm through the parent's own ``phase_bootstrap`` (the
identical group resample shared across compared arms).

ABORT-RECLASSIFY: the first action per run is a store-layout probe — every
requested surface's capture store must bank ``context_end`` vectors at ALL of
layers 14/19/26; any miss writes ``ABORT_RECLASSIFY_NEEDS_GPU.json`` to the
out-root and exits rc=78 (a typed designed halt, not a crash) BEFORE any fit.

CHECKPOINTING: per-cell JSON + preds JSONL are the per-unit checkpoints
(~84 production cells > the ~50-unit floor); resume skips existing cells; a
regime sentinel (labeling sha + n1m payload shas + layers + dof cap) refuses
a stale-root resume. Per-surface ``<surface>_summary.json`` checkpoints land
in the out-root as each surface completes. CONTENT HYGIENE: logs carry
ids/counts, never row text (all four surfaces are benign correctness banks).

Pod-side contract: ``[phase=...]`` breadcrumbs + a single terminal
``[phase=done]``; no ``scripts/task.py`` invocation anywhere; fail fast
everywhere (no except-pass, no silent fallbacks).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

# The #2388 body-Repro final late-fix SHA (branch issue-2388, unmerged).
PARENT_PIN = "06a4b66fba3551b8ffe831a3e88a8276205a853f"
N1M_LAYERS = (14, 19, 26)
N1M_HF_PREFIX = "issue779_monitoring/n1m_readout/weights"
ARM = "arm_n1m"
BANKED_ARMS = ("arm_ctx", "arm_maplin")
EVAL_NAMES = ("dev", "rung0")
SURFACES = ("qa", "math", "mcq", "code")
EXIT_ABORT_RECLASSIFY = 78  # typed designed halt: capture store lacks the n1m layers

N1M_PROVENANCE = {
    "source": "eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json",
    "hf_weights_prefix": N1M_HF_PREFIX,
    "fit_point": "mixed_1m",
    "n_train": 963444,
    "n_lmsys": 529085,
    "n_wildchat": 434359,
    "d": 3584,
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "banked_layers": list(N1M_LAYERS),
}


def _sha12(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _provenance(phase: str) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance(), phase=phase)


# ---------------------------------------------------------------------------
# pinned-module loading (the issue2474_n1m_map precedent, extended to the core)
# ---------------------------------------------------------------------------


def _git_show(ref_path: str) -> bytes:
    res = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", ref_path], capture_output=True, check=True
    )
    return res.stdout


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_pinned_modules():
    """Materialize + load the parent driver AND the dof-capped fits core at PARENT_PIN.

    Loaded BY PATH (importlib) — both blobs live only on the unmerged
    issue-2388 branch, so a static import would be a dangling reference. The
    parent's module-level ``_ensure_repo_root_on_syspath`` asserts a
    ``pyproject.toml`` two levels up, so the parent lands under
    ``<tmp>/scripts/`` beside a stub ``pyproject.toml``; its ``REPO_ROOT``
    global is repointed to the real checkout after exec (only h3/roster code
    paths read it — none run here — but a wrong root must not linger). The
    core is registered under a PRIVATE name (never shadowing the installed
    ``explore_persona_space.experiments.issue_1739.fits``) and rebound as the
    parent's ``F`` alias, so every ``F.ridge_gcv_predict_per_target(...,
    dof_cap=..., selector_telemetry=...)`` call inside the parent resolves to
    the branch core that actually has those kwargs.

    Returns ``(parent_module, core_module)``.
    """
    d = Path(tempfile.mkdtemp(prefix="i2388-n1m-pinned-"))
    (d / "pyproject.toml").write_text("# stub satisfying the pinned parent's repo-root assert\n")
    sdir = d / "scripts"
    sdir.mkdir()
    (sdir / "issue2388_fits.py").write_bytes(_git_show(f"{PARENT_PIN}:scripts/issue2388_fits.py"))
    (d / "i2388_n1m_fits_core.py").write_bytes(
        _git_show(f"{PARENT_PIN}:src/explore_persona_space/experiments/issue_1739/fits.py")
    )
    core = _load_by_path("i2388_n1m_fits_core", d / "i2388_n1m_fits_core.py")
    parent = _load_by_path("issue2388_fits_pinned", sdir / "issue2388_fits.py")
    parent.F = core
    parent.REPO_ROOT = REPO_ROOT
    import inspect

    sig = inspect.signature(core.ridge_gcv_predict_per_target)
    missing = {"dof_cap", "selector_telemetry"} - set(sig.parameters)
    if missing:
        raise RuntimeError(f"pinned core lacks required kwargs {missing} — wrong pin?")
    return parent, core


# ---------------------------------------------------------------------------
# banked n1m map
# ---------------------------------------------------------------------------


def load_n1m_mapfit(core, weights_dir: Path, hidden_dim: int):
    """Stack the three per-layer banked ridge payloads into one linear MapFit.

    Validates each payload is the ridge fitter at the requested layer with a
    square (d, d) weight — a silently mismatched layer/fitter would produce a
    plausible-looking but meaningless read. ``weights_only=False`` is safe
    here: the payloads are the #779 fitter's own sha-recorded output on the
    project HF data repo (self-produced lineage), never a third-party bundle.
    """
    import torch

    ws, xmus, xsds, ymus = [], [], [], []
    meta: dict[str, dict] = {}
    for ly in N1M_LAYERS:
        p = weights_dir / f"L{ly}" / "ridge.pt"
        if not p.exists():
            raise FileNotFoundError(
                f"banked n1m ridge absent: {p} — stage {N1M_HF_PREFIX}/L{ly}/ridge.pt "
                "from the HF data repo (launcher: issue2388_n1m_pod_launch.sh stage)"
            )
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if payload.get("kind") != "ridge" or payload.get("fitter") != "ridge":
            raise RuntimeError(f"{p}: expected the ridge fitter, got {payload.get('kind')!r}")
        if int(payload.get("layer", -1)) != int(ly):
            raise RuntimeError(f"{p}: payload layer {payload.get('layer')} != requested {ly}")
        w = np.asarray(payload["W"], dtype=np.float64)
        if w.shape != (hidden_dim, hidden_dim):
            raise RuntimeError(f"{p}: W shape {w.shape} != ({hidden_dim}, {hidden_dim})")
        ws.append(w)
        xmus.append(np.asarray(payload["xmu"], dtype=np.float64).reshape(1, hidden_dim))
        xsds.append(np.asarray(payload["xsd"], dtype=np.float64).reshape(1, hidden_dim))
        ymus.append(np.asarray(payload["ymu"], dtype=np.float64).reshape(1, hidden_dim))
        meta[f"L{ly}"] = {
            "sha12": _sha12(p),
            "selected_lambda": float(payload["selected_lambda"]),
        }
    mapfit = core.MapFit(
        w=np.stack(ws),
        x_mu=np.stack(xmus),
        x_sd=np.stack(xsds),
        y_mu=np.stack(ymus),
        diagnostics={"n1m": N1M_PROVENANCE},
    )
    return mapfit, meta


# ---------------------------------------------------------------------------
# store resolution + the abort-reclassify probe
# ---------------------------------------------------------------------------


def _qa_store_dir(raw: Path) -> Path:
    """The staged QA tar carries its store name as the root dir (pod_b shape)."""
    nested = raw / "hallucination_labeling"
    return nested if nested.is_dir() else raw


def store_dirs_for(parent_mod, args, surface: str) -> list[Path]:
    if surface == "qa":
        return [_qa_store_dir(Path(args.qa_store_dir))]
    if surface == "code":
        lab = parent_mod._load_labeling(
            Path(args.dv_root) / "code" / "labeling.json", surface="code"
        )
        benchmarks = sorted({str(r["benchmark"]) for r in lab["rows"]})
    else:
        benchmarks = list(parent_mod.SURFACE_BENCHMARKS[surface])
    return [Path(args.store_root) / b for b in benchmarks]


def verify_store_layers(surfaces_dirs: dict[str, list[Path]], out_root: Path) -> None:
    """FIRST ACTION: every requested surface's store must bank context_end at
    L14/19/26; any miss => typed ABORT-RECLASSIFY-NEEDS-GPU halt BEFORE any fit."""
    from explore_persona_space.experiments.issue_1739.store_io import _resolve_summary_kind

    missing: list[dict] = []
    for surface, dirs in surfaces_dirs.items():
        for sd in dirs:
            for ly in N1M_LAYERS:
                try:
                    _resolve_summary_kind(Path(sd), "context_end", ly)
                except FileNotFoundError:
                    missing.append({"surface": surface, "store": str(sd), "layer": ly})
    if not missing:
        print(
            f"[verify] context_end present at layers {list(N1M_LAYERS)} for "
            f"{sorted(surfaces_dirs)} ({sum(len(v) for v in surfaces_dirs.values())} store dirs)",
            flush=True,
        )
        return
    rec = {
        "verdict": "ABORT-RECLASSIFY-NEEDS-GPU",
        "reason": "capture store lacks context_end vectors at required n1m layers",
        "layers_required": list(N1M_LAYERS),
        "missing": missing,
        "metadata": _provenance("n1m-store-verify"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    sentinel = out_root / "ABORT_RECLASSIFY_NEEDS_GPU.json"
    sentinel.write_text(json.dumps(rec, indent=1))
    print(f"[verify] ABORT-RECLASSIFY-NEEDS-GPU — wrote {sentinel}", flush=True)
    raise SystemExit(EXIT_ABORT_RECLASSIFY)


# ---------------------------------------------------------------------------
# per-surface sweep (the parent's protocol, restricted per the decision record)
# ---------------------------------------------------------------------------


def _budgets_for(parent_mod, args, n_train: int) -> list:
    if args.budgets:
        return [b if b == "full" else int(b) for b in args.budgets]
    return [b for b in parent_mod.L_GRID if b <= n_train] + ["full"]


def _regime_sentinel(parent_mod, args, out_root: Path, surface: str, n1m_meta: dict) -> None:
    """Refuse a resume into a root produced under a different regime (parent
    sweep_regime pattern; keys not in cell filenames are pinned here)."""
    regime = {
        "arm": ARM,
        "parent_pin": PARENT_PIN,
        "layers": list(N1M_LAYERS),
        "dof_cap": parent_mod.DOF_CAP,
        "eval_names": list(EVAL_NAMES),
        "hidden_dim": int(args.hidden_dim),
        "labeling_sha": parent_mod._file_sha(Path(args.dv_root) / surface / "labeling.json"),
        "n1m_payload_sha12": {k: v["sha12"] for k, v in n1m_meta.items()},
    }
    sent_p = out_root / "regime.json"
    if sent_p.exists() and not args.force:
        prior = json.loads(sent_p.read_text())
        if prior != regime:
            raise RuntimeError(
                f"n1m regime mismatch at {sent_p}: prior {prior} != current {regime} — "
                "use a fresh --out-root (or --force to recompute)"
            )
    sent_p.write_text(json.dumps(regime, indent=1))


def sweep_surface(parent_mod, core, args, surface: str, mapfit, n1m_meta: dict) -> Path:
    P = parent_mod
    out_root = Path(args.out_root) / surface
    (out_root / "cells").mkdir(parents=True, exist_ok=True)
    (out_root / "preds").mkdir(parents=True, exist_ok=True)
    print(f"[phase={surface}_sweep]", flush=True)

    labeling = Path(args.dv_root) / surface / "labeling.json"
    table = P.load_surface_table(
        surface,
        labeling,
        store_dirs_for(P, args, surface),
        layers=N1M_LAYERS,
        with_tlast=False,
        with_rollout_grain=False,
        hidden_dim=args.hidden_dim,
    )
    train_idx = np.flatnonzero(table.split == "train")
    dev_idx = np.flatnonzero(table.split == "dev")
    if dev_idx.size == 0:
        raise RuntimeError(f"{surface}: empty dev partition — selection impossible")
    rungs = P._rung_eval_sets(table)
    ev_rows_list = [dev_idx, rungs["rung0"]]
    n_train = len(train_idx)
    d_model = int(table.z_ctx.shape[2])
    # Estimator validity: budgets <= 2,000 sit in the parent's own n_train < d
    # regime — handled by the dof cap (0.9) inside the shared core, per-fit
    # selector diagnostics logged below (decision record).
    print(
        f"[sweep] {surface}: train={n_train} dev={len(dev_idx)} test={len(rungs['rung0'])} "
        f"d={d_model} layers={list(N1M_LAYERS)} (n_train vs d logged pre-fit)",
        flush=True,
    )
    _regime_sentinel(P, args, out_root, surface, n1m_meta)

    pool_idx = P._pool_indices(table, P._pool_size(surface, n_train), P.SEED0)
    P.assert_partition_membership(table, pool_idx)
    bases: dict[str, np.ndarray | None] = {"ambient": None}
    for k in P.PCA_KS:
        if k <= min(len(pool_idx), d_model):
            bases[f"pca{k}"] = P._pca_basis(table.z_ctx[:, pool_idx], k)
        else:
            print(f"[sweep] pca{k} skipped (pool {len(pool_idx)} / d {d_model} < k)", flush=True)

    # The registered prediction path, applied by the pinned core (fp64 per
    # layer), then fp16 like the parent's arm_maplin features (estimator parity).
    x_n1m = core.apply_map(table.z_ctx, mapfit).astype(np.float16)

    budgets = _budgets_for(P, args, n_train)
    total = len(budgets) * args.n_draws
    unit = 0
    for budget in budgets:
        for draw_i in range(args.n_draws):
            unit += 1
            seed_parts = [P.SEED0, P._stable_seed(surface), P._budget_seed(budget), draw_i]
            cell_tag = f"L{budget}_draw{draw_i}"
            cell_path = out_root / "cells" / f"{ARM}__{cell_tag}.json"
            if cell_path.exists() and not args.force:
                print(f"[sweep] unit {unit}/{total} {cell_tag} RESUME-SKIP", flush=True)
                continue
            t0 = time.time()
            draw_rows = P.group_respecting_draw(train_idx, table.group, budget, seed_parts)
            draw_splits = set(np.unique(table.split[draw_rows]).tolist())
            if draw_splits != {"train"}:
                raise RuntimeError(f"feasibility(iii): draw escaped train: {draw_splits}")
            y = table.dv[draw_rows][:, None]  # observed target only (no nulls this round)
            best = None
            selector_by_basis: dict[str, dict] = {}
            for basis, v in bases.items():
                if v is not None:
                    x_tr = np.einsum("lnd,ldk->lnk", x_n1m[:, draw_rows].astype(np.float64), v)
                    evals = [
                        np.einsum("lnd,ldk->lnk", x_n1m[:, ev].astype(np.float64), v)
                        for ev in ev_rows_list
                    ]
                else:
                    x_tr = x_n1m[:, draw_rows]
                    evals = [x_n1m[:, ev] for ev in ev_rows_list]
                y_tr = np.broadcast_to(y[None, :, :], (x_tr.shape[0],) + y.shape).copy()
                telem: list[dict] = []
                preds = P.F.ridge_gcv_predict_per_target(
                    x_tr,
                    y_tr,
                    evals,
                    dof_cap=P.DOF_CAP,
                    device=args.device,
                    selector_telemetry=telem,
                )
                blocks = {
                    nm: P._metrics_block(pr, table.dv[ev])
                    for nm, ev, pr in zip(EVAL_NAMES, ev_rows_list, preds, strict=True)
                }
                # per-fit selector diagnostics: selected lambda + effective dof
                # at every (basis, layer) — the decision-record logging duty.
                per_layer = {}
                for li in range(len(N1M_LAYERS)):
                    lam, dof = P._selected_lambda_dof(telem, li)
                    per_layer[f"L{N1M_LAYERS[li]}"] = {"lambda": lam, "dof": dof}
                selector_by_basis[basis] = per_layer
                dev_rho_obs = blocks["dev"]["rho"][:, 0]
                ly_best = int(np.nanargmax(dev_rho_obs))
                lam_sel, dof_sel = P._selected_lambda_dof(telem, ly_best)
                cand = {
                    "basis": basis,
                    "layer": ly_best,
                    "dev_rho": float(dev_rho_obs[ly_best]),
                    "blocks": blocks,
                    "preds": preds,
                    "selector": {
                        "mode": telem[0]["mode"] if telem else None,
                        "dof_cap": telem[0]["dof_cap"] if telem else None,
                        "n_train": telem[0]["n_train"] if telem else None,
                        "lambda_selected": lam_sel,
                        "dof_selected": dof_sel,
                    },
                }
                if best is None or cand["dev_rho"] > best["dev_rho"]:
                    best = cand
            row = {
                "surface": surface,
                "arm": ARM,
                "budget": str(budget),
                "draw": draw_i,
                "n_draw_rows": int(len(draw_rows)),
                "n_null": 0,
                "dof_cap": P.DOF_CAP,
                "n_train_vs_d": [int(len(draw_rows)), d_model],
                "map": "n1m_mixed_1m",
                "n1m_payload_sha12": {k: v["sha12"] for k, v in n1m_meta.items()},
                "pooling": "t1",
                "layer_indexing": f"positional over banked layers {list(N1M_LAYERS)}",
                "actual_layer": int(N1M_LAYERS[best["layer"]]),
                "selector_by_basis": selector_by_basis,
                "split_identity": {
                    "seed_parts": seed_parts,
                    "n_train": n_train,
                    "n_dev": int(len(dev_idx)),
                    "n_test": int(len(rungs["rung0"])),
                },
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            row.update(
                {
                    "basis": best["basis"],
                    "layer": best["layer"],
                    "dev_rho": best["dev_rho"],
                    "selector": best["selector"],
                    "per_eval": {
                        nm: {
                            met: float(best["blocks"][nm][met][best["layer"], 0])
                            for met in ("rho", "r2", "auroc")
                        }
                        for nm in best["blocks"]
                    },
                    "wall_s": round(time.time() - t0, 1),
                }
            )
            cell_path.write_text(json.dumps(row))
            P._write_preds(
                out_root / "preds" / f"preds_{ARM}_{cell_tag}.jsonl",
                table,
                dev_idx,
                rungs,
                list(EVAL_NAMES),
                best,
            )
            print(
                f"[sweep] unit {unit}/{total} {ARM} {cell_tag} basis={best['basis']} "
                f"actual_layer=L{N1M_LAYERS[best['layer']]} dev_rho={best['dev_rho']:.4f} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
            if args.pilot:
                print(
                    f"[pilot] measured production cell wall: {time.time() - t0:.1f}s "
                    f"(surface={surface}, budget={budget}, draw={draw_i}, "
                    f"bases={len(bases)}, layers={len(N1M_LAYERS)})",
                    flush=True,
                )
                return out_root
    return out_root


# ---------------------------------------------------------------------------
# banked-preds staging + paired bootstrap + per-surface checkpoint
# ---------------------------------------------------------------------------


def stage_banked_preds(args, surface: str, out_root: Path) -> int:
    """Copy the parent's banked arm_ctx / arm_maplin preds beside the new arm's,
    filtered to dev+rung0 (rung1 transfer reads are out of scope this round) so
    the parent's phase_bootstrap pairs exactly the three arms per cell."""
    src = Path(args.parent_fits_root) / surface / "preds"
    if not src.is_dir():
        raise FileNotFoundError(
            f"parent preds dir absent: {src} — stage branch issue-2388's "
            "eval_results/issue_2388/fits via the launcher's stage step"
        )
    my_preds = sorted((out_root / "preds").glob(f"preds_{ARM}_L*_draw*.jsonl"))
    if not my_preds:
        raise RuntimeError(f"no {ARM} preds under {out_root / 'preds'} — sweep first")
    n_staged = 0
    for mine in my_preds:
        cell_tag = mine.name[len(f"preds_{ARM}_") : -len(".jsonl")]
        for arm in BANKED_ARMS:
            sp = src / f"preds_{arm}_{cell_tag}.jsonl"
            if not sp.exists():
                raise FileNotFoundError(
                    f"banked preds missing for paired bootstrap: {sp} "
                    f"(cell {cell_tag} exists for {ARM})"
                )
            dest = out_root / "preds" / sp.name
            if dest.exists() and not args.force:
                continue
            kept = [
                line
                for line in sp.read_text(encoding="utf-8").split("\n")
                if line.strip() and json.loads(line)["eval"] in EVAL_NAMES
            ]
            if not kept:
                raise RuntimeError(f"{sp}: 0 rows at evals {EVAL_NAMES}")
            dest.write_text("\n".join(kept) + "\n", encoding="utf-8")
            n_staged += 1
    print(f"[bootstrap-stage] {surface}: {n_staged} banked preds files staged", flush=True)
    return n_staged


def finish_surface(parent_mod, args, surface: str, out_root: Path, n1m_meta: dict) -> None:
    P = parent_mod
    # Selection FREEZE from the persisted dev-selected cells BEFORE the test
    # read is aggregated (the parent's own phase; selection.json + all_arms.json).
    print(f"[phase={surface}_select]", flush=True)
    ns = SimpleNamespace(surface=surface, fits_root=args.out_root)
    P.phase_select(ns)

    print(f"[phase={surface}_bootstrap]", flush=True)
    stage_banked_preds(args, surface, out_root)
    ns = SimpleNamespace(
        surface=surface,
        fits_root=args.out_root,
        dv_root=args.dv_root,
        n_boot=args.n_boot,
        force=args.force,
    )
    P.phase_bootstrap(ns)

    ckpt = Path(args.out_root) / f"{surface}_summary.json"
    ckpt.write_text(
        json.dumps(
            {
                "surface": surface,
                "arm": ARM,
                "banked_arms": list(BANKED_ARMS),
                "eval_names": list(EVAL_NAMES),
                "n_cells": len(list((out_root / "cells").glob(f"{ARM}__*.json"))),
                "selection": str(out_root / "selection.json"),
                "bootstrap_summary": str(out_root / "bootstrap_summary.json"),
                "n1m": {"provenance": N1M_PROVENANCE, "payloads": n1m_meta},
                "metadata": _provenance("n1m-surface-summary"),
            },
            indent=1,
        )
    )
    print(f"[checkpoint] {surface} complete -> {ckpt}", flush=True)


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--surface", choices=SURFACES, default=None, help="one-surface process shard")
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="ONE measured production cell (math, L=250, draw 0) through this exact "
        "entrypoint; prints the measured per-cell wall and exits (no select/bootstrap)",
    )
    ap.add_argument("--out-root", default="eval_results/issue_2388/n1m")
    ap.add_argument("--dv-root", default="/workspace/i2388_parent/eval_results/issue_2388/dv")
    ap.add_argument(
        "--parent-fits-root",
        default="/workspace/i2388_parent/eval_results/issue_2388/fits",
        help="staged branch-issue-2388 fits tree (banked arm_ctx/arm_maplin preds)",
    )
    ap.add_argument("--store-root", default="/workspace/store_2388")
    ap.add_argument("--qa-store-dir", default="/workspace/store")
    ap.add_argument(
        "--n1m-weights-dir",
        default=f"/workspace/n1m_weights/{N1M_HF_PREFIX}",
        help="dir holding L{14,19,26}/ridge.pt (hf download preserves repo paths)",
    )
    ap.add_argument("--budgets", nargs="*", default=None)
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--hidden-dim", type=int, default=3584)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        import torch  # noqa: F401

        from explore_persona_space.experiments.issue_1739.store_io import (  # noqa: F401
            _resolve_summary_kind,
        )

        # The deferred heavy path: the pinned-module load (needs the issue-2388
        # objects in the local odb — `git fetch origin issue-2388` first on a
        # fresh clone) + the dof_cap signature assert inside.
        parent, core = load_pinned_modules()
        assert parent.F is core and parent.DOF_CAP == 0.9, (parent.F, parent.DOF_CAP)
        print("[import-check] ok (pinned parent + dof-capped core loaded)", flush=True)
        return 0

    surfaces = ["math"] if args.pilot else ([args.surface] if args.surface else list(SURFACES))
    if args.pilot:
        # The registered pilot cell is (math, L=250, draw 0); an explicit
        # --budgets override keeps the pilot path smokeable on tiny fixtures.
        args.budgets = args.budgets or ["250"]
        args.n_draws = 1

    parent, core = load_pinned_modules()
    out_base = Path(args.out_root)

    # FIRST ACTION: the abort-reclassify store probe over every requested
    # surface, before any payload load or fit.
    print("[phase=verify_store]", flush=True)
    verify_store_layers({s: store_dirs_for(parent, args, s) for s in surfaces}, out_base)

    mapfit, n1m_meta = load_n1m_mapfit(core, Path(args.n1m_weights_dir), args.hidden_dim)
    print(f"[n1m] payloads: {json.dumps(n1m_meta)}", flush=True)

    for surface in surfaces:
        out_root = sweep_surface(parent, core, args, surface, mapfit, n1m_meta)
        if args.pilot:
            # No terminal [phase=done] on the pilot pass — that token is
            # reserved for a completed shard run (pod-side-reporting.md).
            print("[pilot] done (no select/bootstrap on the pilot pass)", flush=True)
            return 0
        finish_surface(parent, args, surface, out_root, n1m_meta)

    # Mode-gated standalone-lane terminal: emitted ONLY at the end of a
    # completed full shard run; each issue2388_n1m_pod_launch.sh `_child
    # <surface>` invoker appends this process's stdout to its OWN per-surface
    # log whose true terminal line this is (nothing appends after; the rc
    # sentinel is a separate file), and the `--import-check` / `pilot`
    # invocations return before reaching it.
    print("[phase=done]", flush=True)  # workflow-lint: phase-done-reserved (_child log terminal)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
