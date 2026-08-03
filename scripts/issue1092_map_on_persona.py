#!/usr/bin/env python3
"""Issue #1092 map_on_persona: apply the #779 n1m/5k maps to the persona-corpus rows.

The never-computed cell. The #779 monitoring maps (ridge / mlp_w8192 /
mlp_w32768 / krr_nystrom at L14, L19, persisted by ``issue779_ffc_n1m_fits
--persist-weights``) and the parent 5k LMSYS GramRidge linear map are applied to
the #1092 ``cell_inst_own`` persona-corpus context-end activations, and the
monitoring read ``<h(v_C), r_B>`` (dot AND cosine) is correlated (Pearson r +
group cluster-bootstrap CI, prefix-level groups) against the judged trait score
over the SAME rows for every arm. The raw ``<v_C, r_B>`` (pv_raw) is included as
a sanity duplicate of the pooled round's baseline.

Reuses the pooled round's loaders (``issue1092_transfer_probe`` /
``issue1092_pooled_probe_transfer``) READ-ONLY against the already-staged inputs
under ``data/issue_1092/transfer_probe`` (single-writer, complete on disk); the
5k linear map recomputes ``GramRidge`` on the parent pass_b bundle exactly as
``issue779_n1m_readout`` does. ANALYSIS-ONLY: no GPU, no generation, no judging.
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

import issue1092_pooled_probe_transfer as P  # noqa: E402  (_boot_ci)
import issue1092_transfer_probe as T  # noqa: E402  (staging / unit loader)
import issue779_arm_headline as A  # noqa: E402  (GramRidge for the 5k linear map)
import issue779_ffc_n1m_fits as N1M  # noqa: E402  (apply_map)
from issue1092_fit_grid import _pearson_or_nan  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402  (readouts)

TRAITS = ("hallucination", "sycophancy", "evil")
LAYERS = (14, 19)
N1M_FITTERS = {
    "n1m_ridge": "ridge",
    "n1m_mlp_w8192": "mlp_w8192",
    "n1m_mlp_w32768": "mlp_w32768",
    "n1m_krr_nystrom": "krr_nystrom",
}
MAP_ARMS = ("h_n5k_linear", *N1M_FITTERS)  # each read as dot AND cosine
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_WEIGHTS_PREFIX = "issue779_monitoring/n1m_readout/weights"
N_BOOT = 2000
SEED = 0
DEV = torch.device("cpu")
APPLY_CHUNK = 4096

WEIGHTS_DIR = PROJECT_ROOT / "data/issue_1092/map_on_persona/n1m_weights"
PASSB_PATH = PROJECT_ROOT / "data/issue_779/pass_b/train_context_vectors.pt"
RB_DIR = PROJECT_ROOT / "data/issue_779/r_b"
STAGE_ROOT = PROJECT_ROOT / "data/issue_1092/transfer_probe"
OUT_PATH = PROJECT_ROOT / "eval_results/issue_1092/pooled-probe-transfer/map_on_persona_reads.json"


def _log(msg: str) -> None:
    print(f"[map-on-persona] {msg}", flush=True)


def stage_weights() -> dict[str, str]:
    """Stage the 8 (layer, fitter) n1m weight payloads from HF (local-first)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    shas: dict[str, str] = {}
    for layer in LAYERS:
        for fitter in N1M_FITTERS.values():
            rel = f"{HF_WEIGHTS_PREFIX}/L{layer}/{fitter}.pt"
            dst = WEIGHTS_DIR / f"L{layer}" / f"{fitter}.pt"
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                got = Path(
                    hub.retry_transient(
                        lambda rel=rel: hf_hub_download(
                            HF_DATA_REPO, filename=rel, repo_type="dataset"
                        ),
                        what=f"stage {rel}",
                    )
                )
                if got.resolve() != dst.resolve():
                    import shutil

                    shutil.copyfile(got, dst)
                _log(f"staged {rel}")
            shas[f"L{layer}/{fitter}"] = "present"
    return shas


def build_gram5k(layer: int, bundle: dict) -> tuple[A.GramRidge, np.ndarray]:
    """(GramRidge on the pass_b context activations at ``layer``, answer targets Ya)."""
    col = bundle["layers"].index(layer)
    xa = bundle["cx_last"][:, col, :].to(torch.float64).numpy()
    ya = bundle["v_x"][:, col, :].to(torch.float64).numpy()
    _log(f"5k GramRidge factorization L{layer} (n_tr={xa.shape[0]})")
    return A.GramRidge(xa), ya


def apply_map_chunked(
    arm: str, layer: int, X: np.ndarray, gram5k: dict, payloads: dict
) -> np.ndarray:
    """Predicted profile h(X) (n, H) float64 for a map arm, row-chunked for RAM."""
    outs: list[np.ndarray] = []
    for i in range(0, X.shape[0], APPLY_CHUNK):
        xb = X[i : i + APPLY_CHUNK]
        if arm == "h_n5k_linear":
            gr, ya = gram5k[layer]
            outs.append(gr.predict(ya, xb))
        else:
            outs.append(N1M.apply_map(payloads[(layer, N1M_FITTERS[arm])], xb, DEV))
    return np.concatenate(outs, axis=0)


def _read_entry(proj: np.ndarray, y: np.ndarray, groups: list[str], label: str) -> dict[str, Any]:
    r = _pearson_or_nan(proj, y)
    ci = P._boot_ci(y, proj, groups, N_BOOT, label, SEED)
    return {
        "r": float(r),
        "ci_r": [ci["lo"], ci["hi"]],
        "n": int(y.size),
        "n_valid_boot_replicates": ci["n_valid_replicates"],
    }


def main() -> int:
    t0 = time.time()
    assert PASSB_PATH.exists(), f"pass_b bundle missing: {PASSB_PATH}"
    for name in ("corpus", "summaries", "p5_scores.jsonl"):
        assert (STAGE_ROOT / name).exists(), f"staged input missing: {STAGE_ROOT / name}"

    weight_shas = stage_weights()

    # r_B per trait (28, 3584), loaded exactly as the pooled round does.
    r_b: dict[str, np.ndarray] = {}
    for trait in TRAITS:
        blob = torch.load(RB_DIR / f"{trait}.pt", map_location="cpu", weights_only=False)
        r_b[trait] = blob["r_b"].to(torch.float64).numpy()
        assert r_b[trait].shape == (28, 3584), (trait, r_b[trait].shape)

    # persistent n1m payloads.
    payloads: dict[tuple[int, str], dict] = {}
    for layer in LAYERS:
        for fitter in N1M_FITTERS.values():
            pl = torch.load(
                WEIGHTS_DIR / f"L{layer}" / f"{fitter}.pt",
                weights_only=False,
                map_location="cpu",
            )
            assert int(pl.get("layer")) == layer, (layer, fitter, pl.get("layer"))
            payloads[(layer, fitter)] = pl

    # 5k linear GramRidge per layer (reuse the parent pass_b bundle).
    bundle = torch.load(PASSB_PATH, mmap=True, weights_only=False, map_location="cpu")
    assert bundle["cx_last"].shape[1:] == (28, 3584), bundle["cx_last"].shape
    assert bundle["v_x"].shape == bundle["cx_last"].shape
    for layer in LAYERS:
        assert layer in bundle["layers"], (layer, bundle["layers"])
    gram5k: dict[int, tuple[A.GramRidge, np.ndarray]] = {
        layer: build_gram5k(layer, bundle) for layer in LAYERS
    }

    staged = {
        "corpus_dir": STAGE_ROOT / "corpus",
        "summaries_dir": STAGE_ROOT / "summaries",
        "judge_scores": STAGE_ROOT / "p5_scores.jsonl",
    }

    report: dict[str, Any] = {
        "metadata": {
            "script": "issue1092_map_on_persona.py",
            "followup_label": "map-on-persona-reads",
            "git_commit": T._git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "traits": list(TRAITS),
            "layers": list(LAYERS),
            "map_arms": list(MAP_ARMS),
            "n_boot": N_BOOT,
            "seed": SEED,
            "readouts": ["dot", "cos"],
            "provenance": {
                "persona_corpus": (
                    "reused #1092 cell_inst_own context-end summaries + p5 judge scores "
                    f"staged read-only under {STAGE_ROOT.relative_to(PROJECT_ROOT)} "
                    "(loaded via issue1092_transfer_probe.load_1092_unit; group_key=prefix_id)"
                ),
                "n1m_weights": (
                    f"HF {HF_DATA_REPO}:{HF_WEIGHTS_PREFIX}/L{{14,19}}/{{fitter}}.pt "
                    "(#779 issue779_ffc_n1m_fits --persist-weights, mixed_1m fit point)"
                ),
                "n1m_weights_shas": weight_shas,
                "map_5k_linear": (
                    "recomputed in-process: A.GramRidge on the parent pass_b bundle "
                    f"{PASSB_PATH.relative_to(PROJECT_ROOT)} cx_last->v_x per layer "
                    "(same machinery as issue779_n1m_readout h_n5k_linear)"
                ),
                "r_b": (
                    f"{RB_DIR.relative_to(PROJECT_ROOT)}/{{trait}}.pt ['r_b'][layer] "
                    "(#779 persona/behavior direction; same source as pooled round r_b_baseline)"
                ),
                "same_rows_invariant": (
                    "per (trait, layer): pv_raw + every map arm read over the identical "
                    "scored-row set (the trait's judged cell_inst_own rows); no dedup applied "
                    "(matches the pooled round within-corpus ceiling, not the cross-corpus eval)"
                ),
            },
            "notes": [
                "map input = context-end activation (3584-d, layer L) — same space+position "
                "the #779 maps were fit on; the persona corpus is a different context "
                "distribution (the intended cross-application / transfer read).",
                "dot = <h(v_C), r_B>; cos = cosine(h(v_C), r_B); pv_raw = <v_C, r_B>.",
            ],
        },
        "reads": {},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for layer in LAYERS:
        _log(f"===== layer {layer} =====")
        unit = T.load_1092_unit(staged, layer)
        x_full = np.asarray(unit["x_ctx"])  # (N, 3584) float16
        assert x_full.shape[1] == 3584, x_full.shape

        # per-trait scored rows (positions into unit_rows), scores, prefix groups.
        trait_rows: dict[str, dict[str, Any]] = {}
        for trait in TRAITS:
            pairs = unit["by_trait"].get(trait, [])
            if not pairs:
                _log(f"WARNING: no scored rows for {trait} L{layer}; skipping")
                continue
            idx = np.asarray([i for i, _s in pairs], dtype=np.int64)
            y = np.asarray([s for _i, s in pairs], dtype=np.float64)
            groups = [str(unit["unit_rows"][i].get("prefix_id")) for i in idx]
            trait_rows[trait] = {"idx": idx, "y": y, "groups": groups}

        # union of scored rows across traits -> apply each map ONCE over the union.
        union = np.asarray(sorted({int(i) for tr in trait_rows.values() for i in tr["idx"]}))
        union_pos = {int(g): k for k, g in enumerate(union)}
        x_union = x_full[union].astype(np.float64)  # (U, 3584)
        _log(
            f"L{layer}: union scored rows U={union.size}, per-trait n={{"
            + ", ".join(f"{t}:{d['y'].size}" for t, d in trait_rows.items())
            + "}"
        )

        for trait, d in trait_rows.items():
            report["reads"].setdefault(trait, {})[f"L{layer:02d}"] = {"monitors": {}}

        # pv_raw = <v_C, r_B> over the union, indexed per trait.
        for trait, d in trait_rows.items():
            rb_l = r_b[trait][layer]
            proj_union = x_union @ rb_l
            loc = np.asarray([union_pos[int(i)] for i in d["idx"]])
            proj = proj_union[loc]
            report["reads"][trait][f"L{layer:02d}"]["monitors"]["pv_raw"] = _read_entry(
                proj, d["y"], d["groups"], f"pvraw::{trait}::{layer}"
            )
        _write(report)

        # map arms: apply once over the union, dot+cos readout per trait.
        for arm in MAP_ARMS:
            pred_union = apply_map_chunked(arm, layer, x_union, gram5k, payloads)
            for trait, d in trait_rows.items():
                rb_l = r_b[trait][layer]
                loc = np.asarray([union_pos[int(i)] for i in d["idx"]])
                pf = pred_union[loc]
                dot = F.dot_readout(pf, rb_l)
                cos = F.cosine_readout(pf, rb_l)
                mon = report["reads"][trait][f"L{layer:02d}"]["monitors"]
                mon[f"{arm}_dot"] = _read_entry(
                    dot, d["y"], d["groups"], f"{arm}dot::{trait}::{layer}"
                )
                mon[f"{arm}_cos"] = _read_entry(
                    cos, d["y"], d["groups"], f"{arm}cos::{trait}::{layer}"
                )
            _log(
                f"L{layer} {arm}: "
                + " ".join(
                    f"{t}_dot={report['reads'][t][f'L{layer:02d}']['monitors'][arm + '_dot']['r']:.3f}"
                    for t in trait_rows
                )
            )
            del pred_union
            gc.collect()
            _write(report)

    report["metadata"]["wall_seconds"] = round(time.time() - t0, 1)
    _write(report)
    _log(f"done in {report['metadata']['wall_seconds']}s -> {OUT_PATH}")
    return 0


def _write(report: dict) -> None:
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2))
    os.replace(tmp, OUT_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
