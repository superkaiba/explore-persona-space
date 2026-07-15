#!/usr/bin/env python
"""Issue #1336 — Phase F: within-cell held-out ridge fits (thin #825 driver).

Thin driver over ``issue825_fit_cells`` cores (`heldout_r2_sweep`, controls,
bootstrap) with the Llama frozen set {16, 21, 22, 30} threaded through the
default-preserving ``frozen_layers`` parametrization (the fact-checker
must-do: the module-global Qwen set would otherwise silently persist preds at
the wrong layers).

Modes (one per invocation):
  --g0 [--g0-probe-only|--g0-local-dir D]   G0 fit-core reuse gate (plan §7):
        refit the committed Qwen S1 cell (pinned #825 turnstore stems @
        deb7a452) through THIS generalized fit path; PASS <=> layer-19
        held-out R^2 within ±0.01 of the committed 0.6731. Exit 3 on FAIL.
  --cells <id,...|all|smoke> [--matched-n]  per-cell sweeps -> cells/*.json,
        nulls/*.json, preds npz (+ manifest), prefix-slot degeneracy check.
  --g1-check                                G1 rig-transfer kill gate (plan §7)
        from the After-RLVR lmsys5k-chat cell JSON (+ naturalistic when the
        chat read is marginal/below). Exit 0 pass / 3 KILL / 4 need-nat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import issue825_fit_cells as fc  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

CELL_BY_ID = {c["cell_id"]: c for c in cm.CELLS}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--g0", action="store_true", help="run the G0 fit-core reuse gate")
    ap.add_argument("--g0-probe-only", action="store_true", help="Hub existence probe only")
    ap.add_argument("--g0-local-dir", type=Path, default=None, help="pre-staged G0 bundle dir")
    ap.add_argument("--g0-dl-dir", type=Path, default=Path("data/issue_1336/g0_qwen"))
    ap.add_argument("--g1-check", action="store_true", help="evaluate the G1 kill gate")
    ap.add_argument("--cells", default=None, help="comma cell ids | all | smoke")
    ap.add_argument("--turnstore-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1336"))
    ap.add_argument("--preds-dir", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=cm.N_FOLDS)
    ap.add_argument("--seed", type=int, default=cm.FIT_SEED)
    ap.add_argument("--null-draws", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--frozen-layers", default=None, help="comma ints (default: registry set)")
    ap.add_argument(
        "--matched-n", action="store_true", help="also refit at the matched-n subsample"
    )
    ap.add_argument("--matched-n-size", type=int, default=cm.MATCHED_N)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def _metadata(seed: int, n: int) -> dict:
    return {
        "git_commit": fc._git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "n": int(n),
        "script": "scripts/issue1336_fit_cells.py",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[fit1336] wrote {path}")


# ---------------------------------------------------------------------------
# G0 — fit-core reuse gate (artifact-reuse check h-iv: the CONSUMER'S OWN
# loader runs against the real pinned stems before any GPU spend)
# ---------------------------------------------------------------------------
def _g0_probe() -> bool:
    """Cheap Hub existence probe of the pinned stems (single-path file_exists)."""
    from huggingface_hub import HfApi

    api = HfApi()
    ok = True
    for name in ("instruct_chat_s_shard000.pt", "instruct_chat_s_shard000.json"):
        path = f"{cm.G0['hf_prefix']}/{name}"
        found = api.file_exists(
            cm.HF_DATA_REPO, path, repo_type="dataset", revision=cm.G0["revision"]
        )
        print(f"[g0-probe] {path} @ {cm.G0['revision'][:8]}: {'OK' if found else 'MISSING'}")
        ok = ok and found
    return ok


def _g0_stage(dl_dir: Path) -> Path:
    """Stage the pinned Qwen S1 stems: scoped list_repo_tree + per-file download.

    NEVER snapshot_download on the ~1M-file data repo (full-tree enumeration
    wedge — gotchas.md #833); the prefix-scoped tree walk is seconds.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    entries = api.list_repo_tree(
        cm.HF_DATA_REPO,
        path_in_repo=cm.G0["hf_prefix"],
        repo_type="dataset",
        revision=cm.G0["revision"],
        recursive=False,
    )
    stem = cm.G0["stem"]
    wanted = [e.path for e in entries if Path(e.path).name.startswith(f"{stem}_shard")]
    assert wanted, f"no {stem} shards under {cm.G0['hf_prefix']} @ {cm.G0['revision'][:8]}"
    for rel in sorted(wanted):
        hf_hub_download(
            repo_id=cm.HF_DATA_REPO,
            repo_type="dataset",
            filename=rel,
            revision=cm.G0["revision"],
            local_dir=dl_dir,
        )
    bundle_dir = dl_dir / cm.G0["hf_prefix"]
    print(f"[g0] staged {len(wanted)} files -> {bundle_dir}")
    return bundle_dir


def run_g0(args) -> int:
    """Refit the committed Qwen S1 cell through the generalized fit path."""
    if args.g0_probe_only:
        return 0 if _g0_probe() else 1
    bundle_dir = args.g0_local_dir or _g0_stage(args.g0_dl_dir)
    bundle = fc._load_bundle_any(bundle_dir, "instruct", "chat", "s")
    # Parent Track-S normalization (issue825_fit_cells._normalize_cell):
    # assistant slot (index 0) -> a1 profile (index 1).
    xy = fc._cell_xy(bundle, {"slot_index": 0, "target_turn_index": 1})
    X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
    n_layers = X.shape[1]
    layer = int(cm.G0["layer"])
    if args.g0_local_dir is None:
        assert n_layers == cm.G0["expected_layers"], (n_layers, cm.G0["expected_layers"])
        assert X.shape[2] == cm.G0["expected_hidden"], (X.shape[2], cm.G0["expected_hidden"])
    else:
        # Smoke fixture: keep the gate arithmetic on ITS own last layer.
        layer = min(layer, n_layers - 1)
    sweep = fc.heldout_r2_sweep(
        X[:, [layer], :],
        Y[:, [layer], :],
        conv_ids,
        n_folds=cm.N_FOLDS,
        seed=cm.FIT_SEED,
        null_draws=0,
        collect_cosines=False,
        frozen_layers=(),
    )
    r2 = float(sweep["r2_obs"][0])
    committed, tol = float(cm.G0["committed_r2"]), float(cm.G0["tol"])
    ok = abs(r2 - committed) <= tol
    payload = {
        "metadata": _metadata(cm.FIT_SEED, len(conv_ids)),
        "gate": "G0",
        "stem": cm.G0["stem"],
        "revision": cm.G0["revision"],
        "layer": layer,
        "r2_layer": r2,
        "committed_r2": committed,
        "tol": tol,
        "abs_dev": abs(r2 - committed),
        "pass": bool(ok),
        "local_dir_fixture": args.g0_local_dir is not None,
    }
    _write_json(args.out_dir / "gates" / "g0_gate.json", payload)
    print(
        f"[g0] layer-{layer} R2={r2:.4f} vs committed {committed} (tol {tol}) -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 3


# ---------------------------------------------------------------------------
# Per-cell fits
# ---------------------------------------------------------------------------
def _cell_xy_1336(bundle: dict) -> dict:
    """(X, Y, conv_ids, nll) for the context arm: a1-header slot -> a1 profile.

    The #1336 extractor writes slots ordered by position (prefix=0, a1=1) and
    turns by span start (u1=0, a1=1) — asserted here against the bundle shape.
    """
    arrays = bundle["arrays"]
    assert arrays["slots"].shape[1] == 2, f"n_slots {arrays['slots'].shape[1]} != 2"
    assert arrays["profiles"].shape[1] == 2, f"n_turns {arrays['profiles'].shape[1]} != 2"
    return fc._cell_xy(bundle, {"slot_index": 1, "target_turn_index": 1})


def _prefix_degeneracy(bundle: dict, frozen_layers: tuple[int, ...]) -> dict:
    """Registered prefix-arm degeneracy check (plan §4 stated deviation).

    Exact max pairwise cosine DISTANCE across rows of the prefix-header slot
    activation per frozen layer (expected ~0: the prefix token sequence is
    row-constant, so under causal attention the slot activation is too).
    """
    slots = np.asarray(bundle["arrays"]["slots"], dtype=np.float32)  # (N, 2, L, D)
    dev = fc._fit_device()
    out = {}
    for li in frozen_layers:
        if li >= slots.shape[2]:
            continue
        v = torch.as_tensor(slots[:, 0, li, :], dtype=torch.float32).to(dev)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        gram = v @ v.T
        n = gram.shape[0]
        gram.fill_diagonal_(1.0)
        min_cos = float(gram.min())
        out[str(li)] = {"n": n, "min_pairwise_cos": min_cos, "max_pairwise_cos_dist": 1.0 - min_cos}
        del v, gram
    return out


def _persist_preds(preds_dir: Path, cell_id: str, sweep: dict, conv_ids, tag: str = "") -> None:
    """fp16 held-out prediction matrices + manifest (round-5 preds pattern)."""
    preds_dir.mkdir(parents=True, exist_ok=True)
    fname = f"preds_{cell_id}{tag}.npz"
    arrays = {f"preds_l{li}": p.astype(np.float16) for li, p in sweep["preds_frozen"].items()}
    arrays["fitted_mask"] = sweep["fitted_mask"]
    arrays["conv_ids"] = np.asarray([str(c) for c in conv_ids])
    arrays["folds"] = sweep["folds"]
    path = preds_dir / fname
    np.savez(path, **arrays)  # plain savez: client compression OFF for Xet (#813)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = preds_dir / "preds_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[fname] = {
        "sha256": sha,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "dtype_preds": "float16",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[fit1336] persisted {path} (sha256 {sha[:12]}…)")


def run_one_cell(
    cell: dict,
    ts_dir: Path,
    out_dir: Path,
    preds_dir: Path,
    *,
    frozen_layers: tuple[int, ...],
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
    matched_n: int | None,
) -> dict:
    cell_id = cell["cell_id"]
    bundle = fc._load_bundle_any(ts_dir, cell["model"], cell["format"], cell["corpus"])
    xy = _cell_xy_1336(bundle)
    X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
    print(f"[fit1336] cell={cell_id} n={len(conv_ids)}", flush=True)

    sweep = fc.heldout_r2_sweep(
        X,
        Y,
        conv_ids,
        n_folds=n_folds,
        seed=seed,
        null_draws=null_draws,
        frozen_layers=frozen_layers,
    )
    r2_obs, r2_null = sweep["r2_obs"], sweep["r2_null"]
    summary = fc.selection_symmetric_summary(r2_obs, r2_null, frozen_layers=frozen_layers)

    fl = [li for li in frozen_layers if li < X.shape[1]]
    rp = fc.random_projection_control(X, Y, conv_ids, layers=fl, n_folds=n_folds, seed=seed)
    mb = fc.mean_baseline_r2(Y, conv_ids, layers=fl, n_folds=n_folds, seed=seed)

    cosine_stats, r2_cis = {}, {}
    fitted = sweep["fitted_mask"]
    for li in fl:
        cos = sweep["cosines"][li][fitted]
        cosine_stats[str(li)] = fc.bootstrap_ci(cos, n_boot=n_boot, seed=seed + li)
        pred = sweep["preds_frozen"][li][fitted]
        r2_cis[str(li)] = fc.bootstrap_r2_ci(
            pred, Y[fitted, li, :], n_boot=n_boot, seed=seed + 100 + li
        )

    skill_over_mean = {
        str(li): float(r2_obs[li]) - float(mb.get(str(li), float("nan"))) for li in fl
    }
    nll = xy.get("nll")
    nll_stats = None
    if nll is not None and len(nll):
        nll_stats = {
            "mean": float(np.mean(nll)),
            "median": float(np.median(nll)),
            "p90": float(np.quantile(nll, 0.9)),
        }
    degeneracy = _prefix_degeneracy(bundle, frozen_layers)

    payload = {
        "metadata": _metadata(seed, len(conv_ids)),
        "cell": dict(cell),
        "frozen_layers": list(frozen_layers),
        "r2_per_layer_obs": [float(v) for v in r2_obs],
        "selection_symmetric": summary,
        "random_projection_control_r2": rp,
        "mean_baseline_r2": mb,
        "skill_over_mean": skill_over_mean,
        "cosine_frozen_layers": cosine_stats,
        "r2_bootstrap_ci_frozen_layers": r2_cis,
        "nll_a1": nll_stats,
        "prefix_slot_degeneracy": degeneracy,
        "n_folds": n_folds,
        "null_draws": null_draws,
    }
    _write_json(out_dir / "cells" / f"cells_{cell_id}.json", payload)
    _write_json(
        out_dir / "cells" / f"nulls_{cell_id}.json",
        {
            "metadata": _metadata(seed, len(conv_ids)),
            "cell_id": cell_id,
            "layers": list(range(len(r2_obs))),
            "observed_row": [float(v) for v in r2_obs],
            "null_matrix": [[float(v) for v in row] for row in r2_null],
            "null_layer_max_per_draw": summary["null_layer_max_r2_per_draw"],
        },
    )
    _persist_preds(preds_dir, cell_id, sweep, conv_ids)

    if matched_n is not None and len(conv_ids) > matched_n:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(conv_ids), size=matched_n, replace=False))
        sweep_m = fc.heldout_r2_sweep(
            X[keep],
            Y[keep],
            conv_ids[keep],
            n_folds=n_folds,
            seed=seed,
            null_draws=null_draws,
            frozen_layers=frozen_layers,
        )
        summary_m = fc.selection_symmetric_summary(
            sweep_m["r2_obs"], sweep_m["r2_null"], frozen_layers=frozen_layers
        )
        _write_json(
            out_dir / "cells" / f"cells_matchedn_{cell_id}.json",
            {
                "metadata": _metadata(seed, matched_n),
                "cell": dict(cell),
                "matched_n": matched_n,
                "subsample_seed": seed,
                "r2_per_layer_obs": [float(v) for v in sweep_m["r2_obs"]],
                "selection_symmetric": summary_m,
                "n_folds": n_folds,
                "null_draws": null_draws,
            },
        )
        _persist_preds(preds_dir, cell_id, sweep_m, conv_ids[keep], tag="_matchedn")
    return payload


# ---------------------------------------------------------------------------
# G1 — rig-transfer kill gate (plan §7)
# ---------------------------------------------------------------------------
def run_g1_check(out_dir: Path) -> int:
    """KILL <=> best full-sweep within-stage R^2 < 0.2 on the After-RLVR model.

    chat >= 0.3 -> PASS. chat in [0.2, 0.3) -> marginal: the naturalistic cell
    is REQUIRED as extra evidence before proceeding (exit 4 asks the
    dispatcher to fit it), verdict PASS-marginal once present. chat < 0.2 ->
    the naturalistic read is checked before killing (a chat-template-specific
    artifact must not kill the ladder); KILL only when BOTH formats read
    < 0.2 (conservative: a false KILL is the costly error).
    """
    chat_path = out_dir / "cells" / "cells_rlvr_chat_lmsys5k.json"
    assert chat_path.exists(), f"G1 requires {chat_path} — fit the RLVR chat cell first"
    chat_best = float(np.nanmax(json.loads(chat_path.read_text())["r2_per_layer_obs"]))
    nat_path = out_dir / "cells" / "cells_rlvr_naturalistic_lmsys5k.json"
    nat_best = None
    if nat_path.exists():
        nat_best = float(np.nanmax(json.loads(nat_path.read_text())["r2_per_layer_obs"]))

    need_nat = chat_best < cm.G1_MARGINAL_R2 and nat_best is None
    if need_nat:
        print(f"[g1] chat best R2={chat_best:.4f} < {cm.G1_MARGINAL_R2} — need naturalistic read")
        return 4
    if chat_best >= cm.G1_MARGINAL_R2:
        verdict, kill = "pass", False
    elif chat_best >= cm.G1_KILL_R2:
        verdict, kill = "pass_marginal", False
    else:
        kill = nat_best < cm.G1_KILL_R2
        verdict = "kill" if kill else "pass_marginal_naturalistic_carries"
    payload = {
        "metadata": _metadata(cm.FIT_SEED, 0),
        "gate": "G1",
        "chat_best_r2": chat_best,
        "naturalistic_best_r2": nat_best,
        "kill_threshold": cm.G1_KILL_R2,
        "marginal_threshold": cm.G1_MARGINAL_R2,
        "verdict": verdict,
        "kill": bool(kill),
    }
    _write_json(out_dir / "gates" / "g1_gate.json", payload)
    print(f"[g1] chat={chat_best:.4f} nat={nat_best} -> {verdict}")
    return 3 if kill else 0


def main() -> int:
    args = parse_args()
    if args.g0 or args.g0_probe_only:
        return run_g0(args)
    if args.g1_check:
        return run_g1_check(args.out_dir)
    assert args.cells, "--cells is required (or --g0 / --g1-check)"
    smoke = args.smoke
    ts_dir = args.turnstore_dir or Path(
        "data/issue_1336/" + ("turnstore_smoke" if smoke else "turnstore")
    )
    preds_dir = args.preds_dir or Path("data/issue_1336/" + ("preds_smoke" if smoke else "preds"))
    if args.frozen_layers:
        frozen = tuple(int(x) for x in args.frozen_layers.split(",") if x.strip())
    else:
        frozen = cm.SMOKE_FROZEN_LAYERS if smoke else cm.FROZEN_LAYERS
    null_draws = (
        args.null_draws
        if args.null_draws is not None
        else (cm.SMOKE_NULL_DRAWS if smoke else cm.N_NULL_DRAWS)
    )
    n_boot = (
        args.n_boot if args.n_boot is not None else (cm.SMOKE_N_BOOT if smoke else cm.N_BOOTSTRAP)
    )
    if args.cells == "all":
        cells = cm.CELLS
    elif args.cells == "smoke":
        cells = cm.cells_for(cm.SMOKE_MODELS, cm.SMOKE_CORPORA)
    else:
        cells = [CELL_BY_ID[c.strip()] for c in args.cells.split(",") if c.strip()]
    matched = None
    if args.matched_n:
        matched = int(args.matched_n_size)
    for cell in cells:
        cell_matched = matched
        if matched is not None and cm.CORPORA[cell["corpus"]]["n"] <= matched and not smoke:
            cell_matched = None  # gsm8k_test1319 is already at matched size
        run_one_cell(
            cell,
            ts_dir,
            args.out_dir,
            preds_dir,
            frozen_layers=frozen,
            n_folds=args.folds,
            seed=args.seed,
            null_draws=null_draws,
            n_boot=n_boot,
            matched_n=cell_matched,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
