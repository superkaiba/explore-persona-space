"""Issue #2254 round-2 registered analyzer reads (0 GPU, persisted artifacts only).

Two reads requested by the interpretation-critic (round 1):

1. **Held-out-question sensitivity** (plan v5 "Registered analyzer reads (i)"):
   recompute the decisive E_pre / E_ctxdir / C_gap contrasts + the
   selection-symmetric null band on the 10 decisive questions UNSEEN by the
   localize phase (Q2\\Q1 = question indices 10-19; localize used indices 0-9,
   verified against ``q_of_context`` in the raw-completion files). Mirrors the
   wave-2 verdict code in ``scripts/issue2254_preimage.py`` (``_boot_idx`` /
   ``_q_arr`` / ``_boot_diff_ci`` / ``_null_band`` / ``_lattice_label``)
   restricted to the held-out half; frozen at the SAME operating cells as
   ``decisive/verdicts.json`` (a fresh-argmax variant is reported alongside).
   Output: ``eval_results/issue_2254/decisive/heldout_sensitivity.json``.

2. **Context-side reachability** of the causally-working ctxext direction
   through the map's retained rank-k* right-singular subspace:
   ``r = ||V_k*^T (d_ctxext / xsd)|| / ||d_ctxext / xsd||`` in the map's own
   standardized-context frame (the frame the truncated pinv lives in), per
   behavior at its pre@context operating layer. Chance for a random direction
   is sqrt(k*/3584). Separates "the map's retained subspace cannot represent
   the causal direction" (r << 1) from "the min-norm inversion picks a
   non-causal member of a large pre-image family" (r ~ 1).
   Maps + direction bank are pulled from the HF data repo at the pinned
   revision. Output: ``eval_results/issue_2254/directions/ctxext_reachability.json``.

Run from the issue-2254 worktree root:
    uv run python scripts/issue2254_heldout_and_reachability.py
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # noqa: E402 -- shared-VM thread caps must bind before numpy import (#847)

import json
import zlib
from pathlib import Path

import numpy as np

OUT_ROOT = Path("eval_results/issue_2254")
JUDGED = OUT_ROOT / "judge" / "decisive" / "judged"
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REV = "2f2ab5822bad3a9a52736698e2a9ec9667353f07"
HF_PREFIX = "issue2254_preimage"
STAGE_DIR = Path("data/issue_2254/hf_dl/round2_reachability")

BOOTSTRAP_SEED = 20254  # issue2254_preimage.py L1714
N_BOOT_VERDICT = 2000
NULL_CTX_DIRECTIONS = ("random", "preshuf")  # NULL_STEER["context"]
HELDOUT_SLICE = slice(10, 20)  # Q2\Q1: localize used q indices 0-9 (verified)
HIDDEN = 3584

# Verdict operating cells (decisive/verdicts.json margins.*.cell_id)
OPERATING = {
    "evil": {"pre": "evil__pre__ctx__L14__c0p5", "ctxext": "evil__cxd__ctx__L14__c4"},
    "sycophancy": {
        "pre": "sycophancy__pre__ctx__L17__c4",
        "ctxext": "sycophancy__cxd__ctx__mid__c4",
    },
}
A0 = {"evil": "evil__a0", "sycophancy": "sycophancy__a0"}
# pre@context operating layer per behavior (reachability read)
OP_LAYER = {"evil": 14, "sycophancy": 17}


# ------------------------- wave-2 statistics, verbatim -----------------------


def _boot_idx(nq: int, n_draws: int, seed_key: str) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED + zlib.crc32(seed_key.encode()) % 100000)
    return rng.integers(0, nq, size=(n_draws, nq))


def _q_arr(judged: dict) -> np.ndarray:
    return np.array(
        [np.nan if v is None else float(v) for v in judged["per_question_mean_score"]],
        dtype=np.float64,
    )


def _boot_diff_ci(cell_q: np.ndarray, ref_q: np.ndarray, idx: np.ndarray):
    diffs = np.nanmean(cell_q[idx], axis=1) - np.nanmean(ref_q[idx], axis=1)
    point = float(np.nanmean(cell_q) - np.nanmean(ref_q))
    return point, float(np.nanquantile(diffs, 0.025)), float(np.nanquantile(diffs, 0.975))


def _null_band(null_qarrs, a0_q: np.ndarray, seed_key: str, n_draws: int):
    if not null_qarrs:
        return None
    idx = _boot_idx(len(a0_q), n_draws, seed_key)
    a0_b = np.nanmean(a0_q[idx], axis=1)
    per_cell = np.stack([np.nanmean(q[idx], axis=1) - a0_b for q in null_qarrs], axis=1)
    maxes = np.nanmax(per_cell, axis=1)
    return {
        "p50": float(np.nanquantile(maxes, 0.5)),
        "p975": float(np.nanquantile(maxes, 0.975)),
        "n_cells": int(len(null_qarrs)),
        "n_draws": int(n_draws),
    }


def _lattice_label(margins: dict) -> tuple[str, str]:
    e_pre, e_ctx, gap = margins.get("E_pre"), margins.get("E_ctxdir"), margins.get("C_gap")
    if e_pre is None or e_ctx is None or gap is None:
        return "Undefined", "missing operating-point cells (pre/ctxext @ context)"
    pre_pos = e_pre["ci"][0] > 0
    gap_neg = gap["ci"][1] < 0
    ctx_pos = e_ctx["ci"][0] > 0
    if pre_pos and not gap_neg:
        return "H1", "pre-image steers (CI>0 vs null band) and is not CI-below ctxext"
    if pre_pos and gap_neg:
        return "H3", "pre-image steers but sits CI-below the fitted-map direction"
    if not pre_pos and ctx_pos:
        return "H2", "pre-image does not clear the null band while ctxext does"
    return "Ambiguous", "neither margin resolves at the decisive grain"


# ------------------------------ read 1: held-out -----------------------------


def heldout_sensitivity() -> dict:
    out: dict = {
        "read": "held-out-question sensitivity (plan v5 Registered analyzer reads (i))",
        "question_provenance": (
            "localize used decisive q_of_context indices 0-9 (verified equal across all "
            "1,155 localize raw-completion files); held-out = indices 10-19 of the same "
            "20-question persona-vectors eval bank"
        ),
        "n_questions": 10,
        "behaviors": {},
    }
    for beh in ("evil", "sycophancy"):
        cells: dict[str, tuple[dict, np.ndarray]] = {}
        for f in sorted(JUDGED.glob(f"{beh}__*.json")):
            j = json.loads(f.read_text())
            cells[j["cell_id"]] = (j, _q_arr(j)[HELDOUT_SLICE])
        a0_q = cells[A0[beh]][1]
        assert len(a0_q) == 10, f"{beh}: expected 10 held-out questions, got {len(a0_q)}"
        null_ctx = [
            cq
            for cid, (j, cq) in cells.items()
            if j["cell"].get("kind") == "steer"
            and j["cell"]["position"] == "context"
            and j["cell"]["direction"] in NULL_CTX_DIRECTIONS
            and j["coherence_pass"]
        ]
        band = _null_band(null_ctx, a0_q, f"{beh}__w2nullctx__heldout", N_BOOT_VERDICT)

        def delta(cid: str, seed_key: str):
            return _boot_diff_ci(cells[cid][1], a0_q, _boot_idx(10, N_BOOT_VERDICT, seed_key))

        margins: dict = {}
        for name, direction in (("E_pre", "pre"), ("E_ctxdir", "ctxext")):
            cid = OPERATING[beh][direction]
            point, lo, hi = delta(cid, cid + "__heldout")
            bp = band["p975"]
            margins[name] = {
                "value": point - bp,
                "raw_delta": point,
                "cell_id": cid,
                "ci": [lo - bp, hi - bp],
                "band_p975": bp,
            }
        gp, gl, gh = _boot_diff_ci(
            cells[OPERATING[beh]["pre"]][1],
            cells[OPERATING[beh]["ctxext"]][1],
            _boot_idx(10, N_BOOT_VERDICT, f"{beh}__cgap__heldout"),
        )
        margins["C_gap"] = {"value": gp, "ci": [gl, gh]}
        label, reason = _lattice_label(margins)

        # fresh-argmax robustness variant (re-select best decisive breadth cell
        # per direction on the held-out half, mirroring wave-2's _best)
        argmax_variant = {}
        for direction in ("pre", "ctxext"):
            cands = [
                (float(np.nanmean(cq) - np.nanmean(a0_q)), cid)
                for cid, (j, cq) in cells.items()
                if j["cell"].get("kind") == "steer"
                and j["cell"]["direction"] == direction
                and j["cell"]["position"] == "context"
                and j["coherence_pass"]
            ]
            best = max(cands, key=lambda kv: kv[0])
            argmax_variant[direction] = {"cell_id": best[1], "raw_delta": best[0]}
        out["behaviors"][beh] = {
            "label": label,
            "reason": reason,
            "margins": margins,
            "null_band_context_heldout": band,
            "argmax_variant": argmax_variant,
        }
    return out


# --------------------------- read 2: reachability ----------------------------


def _hf_fetch(rel: str) -> Path:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    return Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                HF_REPO,
                f"{HF_PREFIX}/{rel}",
                repo_type="dataset",
                revision=HF_REV,
                local_dir=STAGE_DIR,
            ),
            what=f"fetch {rel} (issue2254 reachability inputs)",
        )
    )


def ctxext_reachability() -> dict:
    import torch

    out: dict = {
        "read": (
            "context-side reachability ||V_k*^T (d_ctxext/xsd)|| / ||d_ctxext/xsd|| in the "
            "map's standardized-context frame (V from the SVD of M = W.T, k* = ridge-estimable "
            "rank -- the same components the truncated-pinv pre-image is built from)"
        ),
        "hf_revision": HF_REV,
        "behaviors": {},
    }
    for beh, ly in OP_LAYER.items():
        npz = np.load(_hf_fetch(f"analysis_tensors/maps_perlayer/perlayer/L{ly:02d}.npz"))
        kstar = int(npz["kstar"])
        M = np.asarray(npz["W"], dtype=np.float64).T
        _, Sm, Vmt = np.linalg.svd(M, full_matrices=False)
        bundle = torch.load(_hf_fetch(f"directions/{beh}_ctxext_L{ly}.pt"), weights_only=True)
        assert bundle["behavior"] == beh and int(bundle["layer"]) == ly, bundle.keys()
        d_ctx = np.asarray(bundle["direction"].to(torch.float64).numpy()).reshape(-1)
        assert d_ctx.shape[0] == HIDDEN, d_ctx.shape
        w = d_ctx / np.asarray(npz["xsd"], dtype=np.float64)
        w_hat = w / np.linalg.norm(w)
        reach = float(np.linalg.norm(Vmt[:kstar] @ w_hat))
        # sanity: a vector inside V_k* has reachability exactly 1
        sanity = float(np.linalg.norm(Vmt[:kstar] @ Vmt[0]))
        # empirical random-direction chance (analytic: sqrt(k*/hidden))
        rng = np.random.default_rng(20254)
        rnd = rng.standard_normal((20, HIDDEN))
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        chance_emp = float(np.mean(np.linalg.norm(Vmt[:kstar] @ rnd.T, axis=0)))
        out["behaviors"][beh] = {
            "layer": ly,
            "kstar": kstar,
            "reachability_ctxext": reach,
            "chance_analytic_sqrt_kstar_over_d": float(np.sqrt(kstar / HIDDEN)),
            "chance_empirical_random_mean_n20": chance_emp,
            "sanity_inside_subspace": sanity,
            "n_singular_components": int(Sm.shape[0]),
        }
    return out


def main() -> None:
    hs = heldout_sensitivity()
    p1 = OUT_ROOT / "decisive" / "heldout_sensitivity.json"
    p1.write_text(json.dumps(hs, indent=1))
    print(f"wrote {p1}")
    for beh, rec in hs["behaviors"].items():
        m = rec["margins"]
        print(
            f"  {beh}: label={rec['label']}  "
            f"E_pre={m['E_pre']['value']:+.2f} ci={np.round(m['E_pre']['ci'], 2).tolist()}  "
            f"E_ctxdir={m['E_ctxdir']['value']:+.2f} ci={np.round(m['E_ctxdir']['ci'], 2).tolist()}  "
            f"band={m['E_pre']['band_p975']:.2f}"
        )
    rr = ctxext_reachability()
    p2 = OUT_ROOT / "directions" / "ctxext_reachability.json"
    p2.write_text(json.dumps(rr, indent=1))
    print(f"wrote {p2}")
    for beh, rec in rr["behaviors"].items():
        print(
            f"  {beh}: L{rec['layer']} k*={rec['kstar']}  reach={rec['reachability_ctxext']:.3f}  "
            f"chance={rec['chance_analytic_sqrt_kstar_over_d']:.3f}  "
            f"sanity={rec['sanity_inside_subspace']:.6f}"
        )


if __name__ == "__main__":
    main()
