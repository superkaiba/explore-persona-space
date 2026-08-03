#!/usr/bin/env python3
"""Issue #1092 fair recompare: the 2,400-row fixed eval BATTERY as a held-out set.

The existing within-corpus numbers leak: the banked harness excludes
`battery_eval_only` (a stratum that does not exist), so the real `battery`-stratum
rows fall INTO the within-corpus train+CV pool. This recompare fixes the literal
IN THIS SCRIPT ONLY and treats the battery rows as a held-out eval:

  eval  = the 2,400 battery-stratum rows (novel questions — disjoint from the query
          bank by construction), scored under the instruct-own cell.
  train = the non-battery scored rows (instruct-own).

Two variants (they COINCIDE here — battery prefixes are 0-overlap with the
non-battery training prefixes by construction, so the prefix-familiarity Δ is
structurally 0; reported explicitly):
  PRIMARY  (double-held-out) : train prefixes ∉ battery prefixes → novel Q + novel conv.
  SECONDARY (novel-question) : train on all non-battery rows (prefixes may overlap eval).

Arms, ALL evaluated on the IDENTICAL battery eval rows (same-rows invariant):
  monitoring reads (NO refit — matmul re-scoring): raw ⟨v_C,r_B⟩, 5k-linear map+r_B,
    n1m ridge / mlp_w32768 / krr map+r_B.
  refit probes (supervised ridge on the training pool, prefix-grouped CV λ): v_C,
    h(v_C)-ridge, h(v_C)-mlp.
Pearson r + prefix-cluster bootstrap CI (2000 draws). Traits hallucination,
sycophancy, evil (evil battery labels are ~flat — flagged). Instruct-own cell only.

Reuses the loaded pieces (T.load_1092_unit / T.fold_fits / P._boot_ci /
MOP maps+weights + A.GramRidge). ANALYSIS-ONLY, 0 GPU/API; does NOT touch the
pooled workers' files. Real-corpus text is never printed.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

import issue1092_map_on_persona as MOP  # noqa: E402
import issue1092_pooled_probe_transfer as P  # noqa: E402  (_boot_ci)
import issue1092_transfer_probe as T  # noqa: E402  (load_1092_unit / fold_fits)
from issue1092_fit_grid import _folds_from_manifest, _pearson_or_nan, _r2  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

TRAITS = ("hallucination", "sycophancy", "evil")
LAYERS = (14, 19)
MAP_ARMS = {  # slug -> fitter (or 5k marker); monitoring reads, no refit
    "h_n5k_linear": "5k",
    "n1m_ridge": "ridge",
    "n1m_mlp_w8192": "mlp_w8192",
    "n1m_mlp_w32768": "mlp_w32768",
    "n1m_krr_nystrom": "krr_nystrom",
}
PROBE_MAPS = {"v_C": None, "h_ridge": "ridge", "h_mlp": "mlp_w32768"}  # refit probes
BATTERY_STRATUM = "battery"  # the REAL stratum (banked harness wrongly used battery_eval_only)
N_BOOT = 2000
SEED = 0
DEV = torch.device("cpu")
STAGE_ROOT = PROJECT_ROOT / "data/issue_1092/transfer_probe"
RB_DIR = PROJECT_ROOT / "data/issue_779/r_b"
PASSB_PATH = PROJECT_ROOT / "data/issue_779/pass_b/train_context_vectors.pt"
OUT_PATH = (
    PROJECT_ROOT / "eval_results/issue_1092/pooled-probe-transfer/battery_heldout_recompare.json"
)


def _log(m: str) -> None:
    print(f"[battery-recompare] {m}", flush=True)


def _read_entry(read: np.ndarray, y: np.ndarray, groups: list[str], label: str) -> dict[str, Any]:
    r = _pearson_or_nan(read, y)
    ci = P._boot_ci(y, read, groups, N_BOOT, label, SEED)
    return {
        "r": float(r),
        "r2": float(_r2(y.reshape(-1, 1), read.reshape(-1, 1))),
        "ci_r": [ci["lo"], ci["hi"]],
        "n": int(y.size),
        "n_valid_boot": ci["n_valid_replicates"],
    }


def main() -> int:
    t0 = time.time()
    MOP.stage_weights()
    payloads = {
        (L, f): torch.load(
            MOP.WEIGHTS_DIR / f"L{L}" / f"{f}.pt", weights_only=False, map_location="cpu"
        )
        for L in LAYERS
        for f in ("ridge", "mlp_w8192", "mlp_w32768", "krr_nystrom")
    }
    r_b = {
        t: torch.load(RB_DIR / f"{t}.pt", map_location="cpu", weights_only=False)["r_b"]
        .to(torch.float64)
        .numpy()
        for t in TRAITS
    }
    bundle = torch.load(PASSB_PATH, mmap=True, weights_only=False, map_location="cpu")
    gram5k = {L: MOP.build_gram5k(L, bundle) for L in LAYERS}
    staged = {
        "corpus_dir": STAGE_ROOT / "corpus",
        "summaries_dir": STAGE_ROOT / "summaries",
        "judge_scores": STAGE_ROOT / "p5_scores.jsonl",
    }

    report: dict[str, Any] = {
        "metadata": {
            "script": "issue1092_battery_heldout_recompare.py",
            "followup_label": "battery-heldout-recompare",
            "git_commit": T._git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "traits": list(TRAITS),
            "layers": list(LAYERS),
            "battery_stratum": BATTERY_STRATUM,
            "cell": "cell_inst_own (instruct-own)",
            "n_boot": N_BOOT,
            "seed": SEED,
            "design": (
                "eval = 2400 battery-stratum rows (novel questions, disjoint from the query "
                "bank); train = non-battery scored rows. PRIMARY double-held-out (train "
                "prefixes ∉ battery prefixes) vs SECONDARY novel-question (all non-battery). "
                "Monitoring arms (raw/5k/n1m) are matmul re-scores (no refit); the 3 probes "
                "are refit on the training pool with prefix-grouped CV λ. All arms scored on "
                "the IDENTICAL battery eval rows."
            ),
            "banked_harness_leak_note": (
                "the banked FITA_EXCLUDED_STRATA uses the literal 'battery_eval_only' which "
                "matches NO stratum, so the real 'battery' rows leak into the banked "
                "within-corpus train+CV pool; this recompare fixes the literal in-script only."
            ),
        },
        "reads": {},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        tmp = OUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        os.replace(tmp, OUT_PATH)

    for layer in LAYERS:
        _log(f"===== layer {layer} =====")
        unit = T.load_1092_unit(staged, layer)
        rows = unit["unit_rows"]
        x_full = np.asarray(unit["x_ctx"])
        bat_pos_set = {i for i, r in enumerate(rows) if r.get("stratum") == BATTERY_STRATUM}

        for trait in TRAITS:
            pairs = unit["by_trait"].get(trait, [])
            bat = [(i, s) for i, s in pairs if i in bat_pos_set]
            non = [(i, s) for i, s in pairs if i not in bat_pos_set]
            if not bat or not non:
                _log(f"WARNING {trait} L{layer}: bat={len(bat)} non={len(non)}; skip")
                continue
            bat_idx = np.asarray([i for i, _s in bat], dtype=np.int64)
            y_bat = np.asarray([s for _i, s in bat], dtype=np.float64)
            bat_pref = [str(rows[i].get("prefix_id")) for i, _s in bat]
            bat_pref_set = set(bat_pref)
            # training pools: SECONDARY = all non-battery; PRIMARY = prefix ∉ battery prefixes.
            non_idx = np.asarray([i for i, _s in non], dtype=np.int64)
            y_non = np.asarray([s for _i, s in non], dtype=np.float64)
            non_pref = [str(rows[i].get("prefix_id")) for i, _s in non]
            keep1 = np.asarray([p not in bat_pref_set for p in non_pref], dtype=bool)
            prefix_overlap = int(len(bat_pref_set & set(non_pref)))
            label_flat = bool(y_bat.std() < 1.0)

            X_bat = x_full[bat_idx].astype(np.float64)
            X_non = x_full[non_idx].astype(np.float64)

            cell: dict[str, Any] = {
                "n_eval": int(y_bat.size),
                "battery_score_std": float(y_bat.std()),
                "battery_score_mean": float(y_bat.mean()),
                "battery_pos_gt50": int((y_bat > 50).sum()),
                "n_train_secondary": int(y_non.size),
                "n_train_primary": int(keep1.sum()),
                "prefix_overlap_battery_vs_nonbattery": prefix_overlap,
                "variants_coincide": bool(prefix_overlap == 0),
                "label_flat": label_flat,
                "monitoring_reads": {},
                "refit_probes": {},
            }

            # ── monitoring arms (no refit) on battery rows ──
            rb_l = r_b[trait][layer]
            cell["monitoring_reads"]["raw"] = _read_entry(
                X_bat @ rb_l, y_bat, bat_pref, f"bat-raw::{trait}::{layer}"
            )
            for slug in MAP_ARMS:
                pred = MOP.apply_map_chunked(slug, layer, X_bat, gram5k, payloads)
                cell["monitoring_reads"][slug] = _read_entry(
                    F.dot_readout(pred, rb_l), y_bat, bat_pref, f"bat-{slug}::{trait}::{layer}"
                )
                del pred
                gc.collect()

            # ── refit probes: train on pool, eval battery (both variants) ──
            for pname, fitter in PROBE_MAPS.items():
                if fitter is None:
                    Xtr_all, Xev = X_non, X_bat
                else:
                    Xtr_all = MOP.apply_map_chunked(f"n1m_{fitter}", layer, X_non, gram5k, payloads)
                    Xev = MOP.apply_map_chunked(f"n1m_{fitter}", layer, X_bat, gram5k, payloads)

                # variant SECONDARY = all non-battery; PRIMARY = prefix ∉ battery prefixes.
                # When prefix_overlap==0 (keep1.all()) the two pools are IDENTICAL, so the
                # second fold_fits is redundant — compute once and copy (variants coincide).
                def _fit_variant(mask: np.ndarray | None, vname: str) -> dict[str, Any]:
                    Xtr = Xtr_all if mask is None else Xtr_all[mask]
                    ytr = y_non if mask is None else y_non[mask]
                    tr_pref = non_pref if mask is None else [p for p, k in zip(non_pref, mask) if k]
                    folds = _folds_from_manifest(
                        [{"prefix_id": g} for g in tr_pref],
                        len(tr_pref),
                        group_key="prefix_id",
                        n_folds=6,
                    )
                    ff = T.fold_fits(Xtr, ytr, folds, [Xev])
                    pred_bat = ff["full_eval"][0]
                    out = {
                        **_read_entry(
                            pred_bat, y_bat, bat_pref, f"bat-{pname}-{vname}::{trait}::{layer}"
                        ),
                        "n_train": int(ytr.size),
                    }
                    del ff
                    gc.collect()
                    return out

                sec = _fit_variant(None, "secondary_novelq")
                if bool(keep1.all()):
                    prim = {**sec, "coincides_with_secondary": True}
                else:
                    prim = _fit_variant(keep1, "primary_double_heldout")
                cell["refit_probes"][pname] = {
                    "secondary_novelq": sec,
                    "primary_double_heldout": prim,
                }
                if fitter is not None:
                    del Xtr_all, Xev
                    gc.collect()

            report["reads"].setdefault(trait, {})[f"L{layer:02d}"] = cell
            checkpoint()
            _log(
                f"L{layer} {trait} (flat={label_flat}): raw={cell['monitoring_reads']['raw']['r']:.3f} "
                f"n1m_mlp={cell['monitoring_reads']['n1m_mlp_w32768']['r']:.3f} | "
                f"vC-probe novelq={cell['refit_probes']['v_C']['secondary_novelq']['r']:.3f} "
                f"double={cell['refit_probes']['v_C']['primary_double_heldout']['r']:.3f}"
            )
        del unit, x_full
        gc.collect()

    report["metadata"]["wall_seconds"] = round(time.time() - t0, 1)
    checkpoint()
    _log(f"done in {report['metadata']['wall_seconds']}s -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
