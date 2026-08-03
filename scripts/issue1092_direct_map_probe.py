#!/usr/bin/env python3
"""Issue #1092 direct-map-probe: supervised ridge probe fit on the MAP OUTPUT h(v_C).

The v_C probe (context-end activation -> judged trait score) is the pooled round's
supervised arm. This asks the answer-space question: fit the SAME supervised ridge
probe on h(v_C) — the #779 map output (predicted answer-side profile) — instead of
on the raw context activation v_C, for h in {n1m ridge (1M linear), n1m mlp_w32768
(1M nonlinear)} at L14 (+L19). Two read variants per (map x trait x layer):

  within_corpus : grouped-CV Pearson r on h(v_C) within each substrate. In-corpus
                  this is DPI-capped by the v_C probe ceiling for the LINEAR map
                  (h(v_C) linear in v_C ⇒ span ⊆ span(v_C)); the nonlinear map is a
                  fixed nonlinear feature transform, so its in-corpus probe is not
                  strictly DPI-bounded by the linear v_C probe — reported, not a bug.
  pooled_lodo   : train the probe on the pool (the 2 substrates != held-out),
                  score the held-out — the SAME LODO the pooled round runs on v_C.
                  The scientific point: does probing in answer space transfer
                  (held-out r) better than probing context space?

Reuses the pooled round's harness VERBATIM (P.build_substrates / P.within_ceiling /
P.train_and_eval; grouped folds, same held-out targets, same-rows invariant,
group-cluster bootstrap). The only change is the feature matrix: sub["X"] ->
h(sub["X"]). ANALYSIS-ONLY, 0 GPU, 0 API; reuses banked inputs read-only + the n1m
weights staged by issue1092_map_on_persona.py. Real-corpus prompt text is never
printed (the loaders carry only sha digests).
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

import issue1092_map_on_persona as MOP  # noqa: E402  (WEIGHTS_DIR + stage_weights + apply)
import issue1092_pooled_probe_transfer as P  # noqa: E402  (build_substrates/within_ceiling/train_and_eval)
import issue1092_transfer_probe as T  # noqa: E402  (loaders)
import issue779_ffc_n1m_fits as N1M  # noqa: E402  (apply_map)
from issue1092_fit_grid import _load_summary  # noqa: E402  (prefix-end summary loader)

TRAITS = ("hallucination", "sycophancy")  # the pooled harness's VERDICT_TRAITS
LAYERS = (14, 19)
MAP_VARIANTS = {"n1m_ridge": "ridge", "n1m_mlp_w32768": "mlp_w32768"}
CTX_NAMES = ("P_persona_ctx", "A_passa_ctx", "L_lmsys_ctx")
N_BOOT = 2000
SEED = 0
DEV = torch.device("cpu")
APPLY_CHUNK = 4096

STAGE_ROOT = PROJECT_ROOT / "data/issue_1092/transfer_probe"
OUT_PATH = (
    PROJECT_ROOT / "eval_results/issue_1092/pooled-probe-transfer/direct_map_probe_reads.json"
)


def _log(msg: str) -> None:
    print(f"[direct-map-probe] {msg}", flush=True)


def build_staged() -> dict[str, Any]:
    """Manual staged dict from the already-present on-disk inputs (no concurrent
    write into the transfer_probe staging dir the pooled runner shares)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    def dl(rev: str, filename: str) -> Path:
        return Path(
            hub.retry_transient(
                lambda: hf_hub_download(
                    T.HF_REPO, repo_type="dataset", revision=rev, filename=filename
                ),
                what=f"stage {filename}",
            )
        )

    staged: dict[str, Any] = {
        "corpus_dir": STAGE_ROOT / "corpus",
        "summaries_dir": STAGE_ROOT / "summaries",
        "judge_scores": STAGE_ROOT / "p5_scores.jsonl",
        "pass_b": PROJECT_ROOT / "data/issue_779/pass_b/train_context_vectors.pt",
        "labels": dl(P.HF_REV_779_LABELS, f"{T.PREFIX_LABELS}/lmsys_g_labels.json"),
        "rollouts": dl(P.HF_REV_779_LABELS, f"{T.PREFIX_LABELS}/lmsys_g_rollouts.json"),
    }
    for req in ("corpus", "summaries", "p5_scores.jsonl"):
        assert (STAGE_ROOT / req).exists(), f"staged input missing: {STAGE_ROOT / req}"
    assert staged["pass_b"].exists(), staged["pass_b"]
    passa: dict[str, dict[str, dict[str, Path]]] = {}
    for trait in TRAITS:
        passa[trait] = {}
        for cond in T.PASSA_CONDS:
            cx = PROJECT_ROOT / f"data/issue_779/pass_a/{trait}__{cond}_cx.pt"
            assert cx.exists(), f"pass_a cx missing: {cx}"
            js = dl(P.HF_REV_779_PASSB, f"{T.PREFIX_PASSA}/{trait}__{cond}.json")
            passa[trait][cond] = {"cx": cx, "json": js}
    staged["pass_a"] = passa
    return staged


def apply_map_chunked(payload: dict, X: np.ndarray) -> np.ndarray:
    """h(X) (n, H) float64, row-chunked for RAM (reuses N1M.apply_map)."""
    outs: list[np.ndarray] = []
    for i in range(0, X.shape[0], APPLY_CHUNK):
        outs.append(N1M.apply_map(payload, X[i : i + APPLY_CHUNK], DEV))
    return np.concatenate(outs, axis=0)


def transform_sub(sub: dict[str, Any], payload: dict) -> dict[str, Any]:
    """Copy a substrate with X replaced by h(X); y/groups/dedup_keep unchanged."""
    out = dict(sub)
    out["X"] = apply_map_chunked(payload, np.asarray(sub["X"], dtype=np.float64))
    return out


def main() -> int:
    t0 = time.time()
    MOP.stage_weights()  # idempotent (present from Deliverable A)
    payloads: dict[tuple[int, str], dict] = {}
    for layer in LAYERS:
        for fitter in MAP_VARIANTS.values():
            pl = torch.load(
                MOP.WEIGHTS_DIR / f"L{layer}" / f"{fitter}.pt",
                weights_only=False,
                map_location="cpu",
            )
            assert int(pl.get("layer")) == layer, (layer, fitter, pl.get("layer"))
            payloads[(layer, fitter)] = pl

    staged = build_staged()
    prompts, labels14, _ = T.load_lmsys_prompts_and_labels(staged["rollouts"], staged["labels"])
    unit14 = T.load_1092_unit(staged, 14)
    dedup = T.overlap_dedup(staged, unit14, prompts)

    report: dict[str, Any] = {
        "metadata": {
            "script": "issue1092_direct_map_probe.py",
            "followup_label": "direct-map-probe",
            "git_commit": T._git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "traits": list(TRAITS),
            "layers": list(LAYERS),
            "map_variants": MAP_VARIANTS,
            "ctx_substrates": list(CTX_NAMES),
            "n_boot": N_BOOT,
            "seed": SEED,
            "arm": (
                "supervised ridge probe fit on the MAP OUTPUT h(v_C) instead of on v_C; "
                "within-corpus grouped-CV + pooled/LODO held-out (same harness as the "
                "v_C probe in pooled_probe_transfer.json)"
            ),
            "dpi_cap_note": (
                "in-corpus, the LINEAR map's probe is DPI-capped by the v_C probe ceiling "
                "(h linear in v_C ⇒ features span ⊆ span(v_C)); the nonlinear map is a fixed "
                "nonlinear feature transform and is NOT strictly bounded by the linear v_C "
                "probe — the transfer (held-out) read is the scientific comparison."
            ),
            "reference_v_c_probe": (
                "eval_results/issue_1092/pooled-probe-transfer/pooled_probe_transfer.json "
                "(within_ceiling.P_persona_ctx = the v_C probe DPI reference; pooled_lodo = "
                "the v_C transfer read this arm is compared against)"
            ),
            "provenance": {
                "persona/passa/lmsys substrates": "P.build_substrates (banked, read-only)",
                "n1m_weights": f"{MOP.HF_WEIGHTS_PREFIX}/L{{14,19}}/{{ridge,mlp_w32768}}.pt",
                "same_rows_invariant": "h(X) is a per-row transform of the SAME rows the v_C "
                "probe uses; dedup_keep applied identically at eval",
            },
        },
        "reads": {slug: {} for slug in MAP_VARIANTS},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        tmp = OUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        os.replace(tmp, OUT_PATH)

    for layer in LAYERS:
        x_lmsys, _ = T.load_pass_b_l14(staged["pass_b"], layer)
        unit = T.load_1092_unit(staged, layer)
        # populate unit["x_prefix"] exactly as the pooled run does (build_substrates
        # reads it for the prefix-end secondary substrate).
        unit["x_prefix"], _ = _load_summary(staged["summaries_dir"], T.CELL, "prefix_end", layer)
        rows_all = T._jsonl(staged["corpus_dir"] / "manifest.jsonl")
        n0 = min(unit["x_prefix"].shape[0], len(rows_all))
        keep_idx = np.asarray(
            [
                i
                for i, r in enumerate(rows_all[:n0])
                if r.get("stratum") not in T.FITA_EXCLUDED_STRATA
            ],
            dtype=np.int64,
        )
        unit["x_prefix"] = unit["x_prefix"][keep_idx]
        assert unit["x_prefix"].shape[0] == unit["x_ctx"].shape[0], (
            unit["x_prefix"].shape,
            unit["x_ctx"].shape,
        )
        for trait in TRAITS:
            subs = P.build_substrates(staged, unit, x_lmsys, labels14, dedup, trait, layer)
            for slug, fitter in MAP_VARIANTS.items():
                payload = payloads[(layer, fitter)]
                h_subs = {n: transform_sub(subs[n], payload) for n in CTX_NAMES}
                within = {
                    n: P.within_ceiling(
                        h_subs[n], N_BOOT, f"dmp-within::{slug}::{trait}::{layer}::{n}", SEED
                    )
                    for n in CTX_NAMES
                }
                lodo: dict[str, Any] = {}
                for h_name in CTX_NAMES:
                    pool = [n for n in CTX_NAMES if n != h_name]
                    out = P.train_and_eval(
                        [h_subs[n] for n in pool],
                        [(h_name, h_subs[h_name])],
                        False,
                        N_BOOT,
                        f"dmp-lodo::{slug}::{trait}::{layer}::{h_name}",
                        SEED,
                    )
                    lodo[h_name] = {
                        "held_out_read": out["reads"][h_name],
                        "within_pool_cv_r": out["within_pool_cv_r"],
                        "within_pool_cv_r2": out["within_pool_cv_r2"],
                        "lam_cv": out["lam_cv"],
                        "n_train": out["n_train"],
                        "pool_members": pool,
                    }
                report["reads"][slug].setdefault(trait, {})[f"L{layer:02d}"] = {
                    "within_corpus": within,
                    "pooled_lodo": lodo,
                }
                checkpoint()
                _log(
                    f"L{layer} {trait} {slug}: within P={within['P_persona_ctx'].get('cv_r')} | "
                    f"LODO heldout-P r={lodo['P_persona_ctx']['held_out_read'].get('r')}"
                )
            del subs
            gc.collect()
        del x_lmsys, unit
        gc.collect()

    report["metadata"]["wall_seconds"] = round(time.time() - t0, 1)
    checkpoint()
    _log(f"done in {report['metadata']['wall_seconds']}s -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
