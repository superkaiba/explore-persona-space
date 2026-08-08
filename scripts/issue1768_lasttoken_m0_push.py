#!/usr/bin/env python3
"""#1768 inline round: context-movement term pushed through the frozen base map.

Computes the decomposition term the committed rounds left unmeasured:
``push_med = median_i ||M0(c+_i) - M0(c0_i)||`` on the held-out test rows,
under LAST-PROMPT-TOKEN context pooling — how much of the answer-side movement
the MOVED context vectors alone explain when pushed through the FROZEN base
context->answer map. Companion to the committed last-token map-change read
``delta_med = median_i ||M+(c0_i) - M0(c0_i)||`` (`lasttoken_repool/`): same
test rows, same ridge machinery, same answer space, so the two terms and the
recorded refit floors are directly comparable. Unlike ``delta_med``, the push
statistic differences through ONE fitted operator, so cross-fit refit noise
cancels to first order; a lambda-sensitivity band (selected lambda x/3, x3) at
L19 is reported instead of a bootstrap floor.

Venue: CPU pod (downloads ~53 GB of per-arm last-token stores; the >10 GB
download rule). Rolling per-arm staging — each arm's ``lasttoken.pt`` is
deleted after its layers are consumed; the two base stores persist for the run.

Usage (pod):
    uv run python scripts/issue1768_lasttoken_m0_push.py \
        --out-root /workspace/i1768_m0push [--arms A,B] [--layers 14,19,25]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1768_capture as CAP  # noqa: E402
import issue1768_cells as X  # noqa: E402
import issue1768_fit as F  # noqa: E402
import issue1768_lasttoken_fit as LTF  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.m0push")

LT_PREFIX = f"{X.HF_PREFIX}/lasttoken_ctx"
POSITION = "last_prompt"  # the round's PRIMARY pooling (the #779 convention)
SENS_LAYER = 19  # lambda-sensitivity + all-rows secondary read run at L19 only


def _stage_lasttoken(out_root: Path, unit: str) -> Path:
    """Download one unit's last-token store to the LTF.load_lasttoken layout."""
    from explore_persona_space.orchestrate import hub

    dest = out_root / "lasttoken" / unit / "lasttoken.pt"
    if dest.exists():
        return dest
    hub.stage_hub_file(
        X.HF_DATA_REPO,
        f"{LT_PREFIX}/{unit}/lasttoken.pt",
        dest,
        repo_type="dataset",
        overwrite=True,
    )
    return dest


def _drop_unit_store(out_root: Path, unit: str) -> None:
    shutil.rmtree(out_root / "lasttoken" / unit, ignore_errors=True)


def _stage_corpus_sample(out_root: Path) -> None:
    """The p0 corpus sample, from the local run tree or HF (the durable copy).

    The file is a round-1 RUN artifact, never committed to git (8.6 MB), so a
    fresh pod checkout does not carry it and HF is the only durable source.
    """
    from explore_persona_space.orchestrate import hub

    dest = out_root / "inputs" / "corpus_sample.json"
    if dest.exists():
        return
    local = REPO_ROOT / "eval_results" / "issue_1768" / "inputs" / "corpus_sample.json"
    if local.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        return
    hub.stage_hub_file(
        X.HF_DATA_REPO,
        f"{X.HF_PREFIX}/inputs/corpus_sample.json",
        dest,
        repo_type="dataset",
        overwrite=True,
    )


def _committed_cells_dir() -> Path:
    return REPO_ROOT / "eval_results" / "issue_1768" / "lasttoken_repool" / "cells"


def _arm_list() -> list[str]:
    cells = sorted(p.stem for p in _committed_cells_dir().glob("*.json"))
    assert cells, f"no committed lasttoken cells under {_committed_cells_dir()}"
    return cells


def _committed_cell(arm_id: str) -> dict:
    return json.loads((_committed_cells_dir() / f"{arm_id}.json").read_text())


def _push_stats(rows: np.ndarray) -> dict:
    return {
        "median": float(np.median(rows)),
        "q25": float(np.quantile(rows, 0.25)),
        "q75": float(np.quantile(rows, 0.75)),
        "mean": float(rows.mean()),
        "n": int(rows.shape[0]),
    }


class BaseSide:
    """Per (base_unit, layer): joined C0/V0 rows, fitted M0, base predictions."""

    def __init__(self, out_root: Path, cache: Path, unit: str, layers: list[int], dev) -> None:
        self.unit = unit
        _stage_lasttoken(out_root, unit)
        c0_by, c0_sha = LTF.load_lasttoken(out_root, unit, layers, POSITION)
        v0_by, v0_sha = LTF.fetch_response(cache, "corpus_capture", unit, layers, persist=True)
        c0_ix = {s: i for i, s in enumerate(c0_sha)}
        keep = [(i, s) for i, s in enumerate(v0_sha) if s in c0_ix]
        assert len(keep) >= 0.9 * len(v0_sha), (unit, len(keep), len(v0_sha))
        b = np.asarray([i for i, _ in keep])
        self.shas = [s for _, s in keep]
        sel_c0 = np.asarray([c0_ix[s] for s in self.shas])

        sample = X.load_corpus_sample(out_root)
        sha_to_q = {r["sha"]: q for q, r in enumerate(sample["rows"])}
        qidx = np.asarray([sha_to_q[s] for s in self.shas])
        n_train, n_val = sample["n_train"], sample["n_val"]
        split = np.where(qidx < n_train, "train", np.where(qidx < n_train + n_val, "val", "test"))
        self.tr, self.val, self.te = F._split_idx(split)

        self.C0 = {li: c0_by[li][sel_c0] for li in layers}
        self.V0 = {li: v0_by[li][b] for li in layers}
        self.dev = dev
        self.m0: dict[int, dict] = {}  # layer -> {payload, meta, pred0_te, r2}
        for li in layers:
            t0 = time.time()
            pred_te, meta, payload = F._fit_map(
                self.C0[li], self.V0[li], self.tr, self.val, self.te, dev
            )
            r2 = F._pooled_r2(pred_te, self.V0[li][self.te])
            pred0_te = F._apply_payload(payload, self.C0[li][self.te], dev)
            self.m0[li] = {
                "payload": payload,
                "selected_lambda": float(meta["selected_lambda"]),
                "heldout_r2": r2,
                "pred0_te": pred0_te,
                "fit_secs": time.time() - t0,
            }
            logger.info(
                "[m0] %s L%d fit: lambda=%.4g r2=%.4f (%.0fs)",
                unit,
                li,
                meta["selected_lambda"],
                r2,
                self.m0[li]["fit_secs"],
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--arms", type=str, default="", help="comma list; default = all committed")
    ap.add_argument("--layers", type=str, default="14,19,25")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", str(os.cpu_count() or 16))))
    dev = torch.device("cpu")

    out_root: Path = args.out_root
    layers = [int(x) for x in args.layers.split(",")]
    cache = out_root / "cache"
    results = out_root / "results"
    (results / "cells").mkdir(parents=True, exist_ok=True)
    cpte_dir = out_root / f"cpte_L{SENS_LAYER}"
    cpte_dir.mkdir(parents=True, exist_ok=True)
    _stage_corpus_sample(out_root)

    arms = [a for a in (args.arms.split(",") if args.arms else _arm_list()) if a]
    base_units = sorted({X.base_unit_for(a) for a in arms})
    logger.info("[plan] %d arms, base units %s, layers %s", len(arms), base_units, layers)

    bases = {u: BaseSide(out_root, cache, u, layers, dev) for u in base_units}

    done = skipped = 0
    for k, arm in enumerate(arms):
        cell_paths = {li: results / "cells" / f"{arm}_L{li}.json" for li in layers}
        cpte_path = cpte_dir / f"{arm}.npy"
        if all(p.exists() for p in cell_paths.values()) and cpte_path.exists():
            skipped += 1
            continue
        t0 = time.time()
        base = bases[X.base_unit_for(arm)]
        _stage_lasttoken(out_root, arm)
        cp_by, cp_sha = LTF.load_lasttoken(out_root, arm, layers, POSITION)
        cp_ix = {s: i for i, s in enumerate(cp_sha)}
        missing = sum(1 for s in base.shas if s not in cp_ix)
        assert missing == 0, (arm, f"{missing} base shas absent from arm store")
        sel = np.asarray([cp_ix[s] for s in base.shas])
        committed_n = _committed_cell(arm)["positions"][POSITION][str(layers[0])]["n_rows"]
        assert len(base.shas) == committed_n, (arm, len(base.shas), committed_n)

        for li in layers:
            m0 = base.m0[li]
            committed = _committed_cell(arm)["positions"][POSITION][str(li)]
            lam_rec = committed["M0"]["selected_lambda"]
            assert abs(m0["selected_lambda"] - lam_rec) / lam_rec < 1e-6, (
                arm,
                li,
                m0["selected_lambda"],
                lam_rec,
            )
            Cp = cp_by[li][sel]
            pred_p_te = F._apply_payload(m0["payload"], Cp[base.te], dev)
            push_te = np.linalg.norm(pred_p_te - m0["pred0_te"], axis=1)
            out = {
                "arm_id": arm,
                "layer": li,
                "position": POSITION,
                "base_unit": base.unit,
                "n_rows": int(len(base.shas)),
                "m0_selected_lambda": m0["selected_lambda"],
                "m0_heldout_r2": m0["heldout_r2"],
                "m0_heldout_r2_committed": committed["M0"]["heldout_r2"],
                "push_te": _push_stats(push_te),
                "delta_med_committed": committed["map_change"]["delta_med"],
                "floor_p95_committed": committed["map_change"]["floor_p95"],
                "ctx_move_committed": committed["context_movement"]["median_relative_move"],
            }
            if li == SENS_LAYER:
                pred_p_all = F._apply_payload(m0["payload"], Cp, dev)
                pred_0_all = F._apply_payload(m0["payload"], base.C0[li], dev)
                out["push_all_rows"] = _push_stats(np.linalg.norm(pred_p_all - pred_0_all, axis=1))
                np.save(cpte_path, Cp[base.te].astype(np.float16))
                del pred_p_all, pred_0_all
            CAP._atomic_json(cell_paths[li], out)
            del Cp, pred_p_te, push_te
        del cp_by
        gc.collect()
        _drop_unit_store(out_root, arm)
        done += 1
        logger.info("[arm %d/%d] %s done in %.0fs", k + 1, len(arms), arm, time.time() - t0)

    sens = {}
    if not args.skip_sensitivity and SENS_LAYER in layers:
        import issue779_ffc_n1m_fits as n1m

        for unit, base in bases.items():
            lam = base.m0[SENS_LAYER]["selected_lambda"]
            for tag, lam_f in (("lam_div3", lam / 3.0), ("lam_x3", lam * 3.0)):
                _pred, _meta, payload = n1m.fit_ridge_with_weights(
                    base.C0[SENS_LAYER],
                    base.V0[SENS_LAYER],
                    base.tr,
                    base.val,
                    base.te,
                    [lam_f],
                    dev,
                    n1m.RIDGE_BLOCK,
                )
                pred0 = F._apply_payload(payload, base.C0[SENS_LAYER][base.te], dev)
                for arm in arms:
                    if X.base_unit_for(arm) != unit:
                        continue
                    cp_te = np.load(cpte_dir / f"{arm}.npy").astype(np.float64)
                    rows = np.linalg.norm(F._apply_payload(payload, cp_te, dev) - pred0, axis=1)
                    sens.setdefault(arm, {})[tag] = float(np.median(rows))
                logger.info("[sens] %s %s (lambda=%.4g) done", unit, tag, lam_f)

    summary = {
        "issue": 1768,
        "read": "push_med = median ||M0(c+) - M0(c0)|| on te rows, last_prompt pooling",
        "position": POSITION,
        "layers": layers,
        "n_arms": len(arms),
        "n_done_this_run": done,
        "n_skipped_resume": skipped,
        "m0_fits": {
            u: {
                str(li): {
                    "selected_lambda": b.m0[li]["selected_lambda"],
                    "heldout_r2": b.m0[li]["heldout_r2"],
                    "fit_secs": b.m0[li]["fit_secs"],
                }
                for li in layers
            }
            for u, b in bases.items()
        },
        "lambda_sensitivity_L19_push_med": sens,
        **CAP._meta(),
    }
    CAP._atomic_json(results / "summary.json", summary)
    logger.info("[done] %d arms (%d resumed), summary -> %s", len(arms), skipped, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
