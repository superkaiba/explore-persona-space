#!/usr/bin/env python3
"""Issue #779 ``n1m-nonlinear-map-behavior-readout``: behavior read-out with n1M maps.

Loads the maps persisted by ``issue779_ffc_n1m_fits.py --layers 14,19,26
--persist-weights`` (ridge / MLP w8192 / MLP w32768 / KRR-Nystrom m=16384, fit
on the ~963k-context ``mixed_1m`` point) and re-scores the parent round's
pre-generation behavior read-out — NO new generation, NO new judging: the eval
rig (pass_a cells + judge scores), r_B, and the behavior corpus are the parent
round's committed artifacts, reused byte-identical.

Per (trait, mode) cell at the FROZEN read layers (hallucination at the
nearest-captured substitutes 17->19 / 27->26 — a named deviation; every arm is
recomputed at the substitute layer so the within-cell arm comparison stays
internally consistent):

  arms = pv_raw | h_n5k_linear (the parent 5k LMSYS GramRidge arm, recomputed
  in-process) | n1m_ridge | n1m_mlp_w8192 | n1m_mlp_w32768 | n1m_krr_nystrom |
  oracle,   each map arm read as <h(c_last), r_B[L]> (dot) AND cosine.

Emits within-condition Pearson r + bootstrap CIs (n_boot=1000, resampling
conditions, std<1 prune — ``issue779_stage1.method_metrics`` unchanged),
delta-vs-raw AND delta-vs-5k-arm CIs (``metrics.bootstrap_delta_ci``), and —
the parent round's known gap — PERSISTS the per-condition r arrays and the
bootstrap replicate draws per (monitor, mode).

Also: the SS7 validity gate (in-run pv_raw + 5k-arm recompute must reproduce the
committed ``arm_headline.json`` (evil) / ``arm_headline_pod.json`` (sycophancy)
values within +-0.02 at the exact frozen layers; hallucination is excluded —
substitute layers have no same-layer committed comparator; a miss WARNS LOUDLY
in the JSON + stderr, never silently passes), an L19 continuity read for all 6
cells (all arms), and the persona-level grouped read (``_grouped_vectors`` /
``_logo_readout`` reused; the parent 5k arm keeps its LOGO protocol, the fixed
pre-fit n1m maps are applied directly — they never saw the corpus groups) with
the {1,2,5,10,20,40} group-size sweep.

Fail loud; NaN judge labels are DROPPED (never coerced); no rollout/prompt TEXT
is ever printed. Outputs: ``n1m_readout.json`` + ``figures/issue_779/
n1m_readout_*.png``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps land BEFORE numpy/torch import on the shared VM.
load_dotenv()

import issue779_arm_headline as A  # noqa: E402
import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_n1m_regen_figs as RF  # noqa: E402  (plain-English ARM_LABELS)
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402
from explore_persona_space.experiments.issue_779 import metrics as M  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue779_n1m_readout")

# Read layers per (trait, mode): frozen where captured; hallucination at the
# nearest-captured DETERMINISTIC substitutes (17->19 sys, 27->26 many) — plan
# SS4 named deviation, fixed before any result is seen.
READ_LAYERS: dict[str, dict[str, int]] = {
    "evil": {"system": 14, "many_shot": 26},
    "sycophancy": {"system": 26, "many_shot": 26},
    "hallucination": {"system": 19, "many_shot": 26},
}
SUBSTITUTED_CELLS = {
    "hallucination/system": {"frozen": 17, "substitute": 19},
    "hallucination/many_shot": {"frozen": 27, "substitute": 26},
}
# Grouped (persona-level) read layer per trait = the parent's system-mode frozen
# layer (run_section4 convention), hallucination substituted 17->19.
GROUPED_LAYERS = {"evil": 14, "sycophancy": 26, "hallucination": 19}
L19_CONTINUITY_LAYER = 19
RECOVERY_LAYER = 26  # --l26-kernel-recovery scope: cells/grouped reads consuming the L26 kernel
MODES = ("system", "many_shot")

# Arm slug -> persisted fitter file stem (issue779_ffc_n1m_fits weights dir).
N1M_FITTERS = {
    "n1m_ridge": "ridge",
    "n1m_mlp_w8192": "mlp_w8192",
    "n1m_mlp_w32768": "mlp_w32768",
    "n1m_krr_nystrom": "krr_nystrom",
}
ARM_ORDER = ("pv_raw", "h_n5k_linear", *N1M_FITTERS, "oracle")
MAP_ARMS = ("h_n5k_linear", *N1M_FITTERS)  # arms with dot+cos readouts
GATE_TOL = 0.02
GROUP_SIZES = (1, 2, 5, 10, 20, 40)

HF_ANALYSIS_PREFIX = "issue779_monitoring/analysis_tensors"
HF_CORPUS_PREFIX = "issue779_monitoring/training-source-ablation-hg/behavior_corpus"


# ── input staging (HF -> the arm_headline path layout) ────────────────────────


def _stage_file(hf_path: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                C.HF_DATA_REPO, filename=hf_path, repo_type="dataset", local_dir=dest.parent
            ),
            what=f"stage {hf_path}",
        )
    )
    if got != dest:
        os.replace(got, dest)
    logger.info("[stage] %s -> %s", hf_path, dest)


def stage_inputs(collect_dir: Path, corpus_dir: Path, traits: tuple[str, ...]) -> None:
    """Download pass_a + pass_b + r_b + behavior-corpus inputs from the HF data
    repo into the exact local layout ``issue779_arm_headline`` reads (scoped
    ``list_repo_tree`` per prefix + per-file download — never
    ``snapshot_download`` with ``allow_patterns`` on this huge repo)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    pa_names = hub.retry_transient(
        lambda: sorted(
            f.path
            # HUB_VERIFY_RETRY_EXEMPT: whole listing wrapped in hub.retry_transient above
            for f in HfApi().list_repo_tree(
                C.HF_DATA_REPO,
                path_in_repo=f"{HF_ANALYSIS_PREFIX}/pass_a",
                repo_type="dataset",
                recursive=True,
            )
            if getattr(f, "size", None) is not None
        ),
        what="pass_a listing",
    )
    if not pa_names:
        raise FileNotFoundError(f"no pass_a files under HF {HF_ANALYSIS_PREFIX}/pass_a")
    for p in pa_names:
        _stage_file(p, collect_dir / "pass_a" / p.rsplit("/", 1)[-1])
    _stage_file(
        f"{HF_ANALYSIS_PREFIX}/pass_b/train_context_vectors.pt",
        collect_dir / "pass_b" / "train_context_vectors.pt",
    )
    for t in traits:
        _stage_file(f"{HF_ANALYSIS_PREFIX}/r_b/{t}.pt", collect_dir / "r_b" / f"{t}.pt")
        _stage_file(f"{HF_CORPUS_PREFIX}/{t}_corpus.pt", corpus_dir / f"{t}_corpus.pt")
        _stage_file(
            f"{HF_CORPUS_PREFIX}/{t}_judge_scores.json", corpus_dir / f"{t}_judge_scores.json"
        )


# ── weights + monitor computation ─────────────────────────────────────────────


class Maps:
    """Lazy cache of persisted n1m maps and the per-layer 5k GramRidge arm.

    ``krr_l26_override`` (#779 l26-kernel-gate-recovery): load the L26
    ``krr_nystrom`` payload from this explicit path (the m=32768 Cholesky
    refit) instead of ``weights_dir/L26/krr_nystrom.pt`` — the realized v10
    payloads in ``weights_dir`` are read-only and never clobbered."""

    def __init__(
        self,
        weights_dir: Path,
        ctx: A.Ctx,
        dev: torch.device,
        krr_l26_override: Path | None = None,
    ) -> None:
        self.weights_dir = weights_dir
        self.ctx = ctx
        self.dev = dev
        self.krr_l26_override = krr_l26_override
        self._payloads: dict[tuple[int, str], dict] = {}
        self._gram5k: dict[int, tuple[A.GramRidge, np.ndarray]] = {}

    def payload(self, layer: int, fitter: str) -> dict:
        key = (int(layer), fitter)
        if key not in self._payloads:
            path = self.weights_dir / f"L{int(layer)}" / f"{fitter}.pt"
            if (
                fitter == "krr_nystrom"
                and int(layer) == RECOVERY_LAYER
                and self.krr_l26_override is not None
            ):
                path = self.krr_l26_override
                logger.info("[maps] L26 krr_nystrom OVERRIDE -> %s", path)
            if not path.exists():
                raise FileNotFoundError(
                    f"persisted weights missing: {path} — run issue779_ffc_n1m_fits.py "
                    "--layers ... --persist-weights first"
                )
            self._payloads[key] = torch.load(path, weights_only=False, map_location="cpu")
            assert self._payloads[key].get("layer") == int(layer), (path, layer)
        return self._payloads[key]

    def gram5k(self, layer: int) -> tuple[A.GramRidge, np.ndarray]:
        """(GramRidge fit on the full 5k pass_b X at ``layer``, Ya) — the parent
        ``h_A_lmsys`` arm's machinery, one factorization per layer."""
        li = int(layer)
        if li not in self._gram5k:
            Xa, Ya = self.ctx.lmsys_layer(li)
            logger.info("[5k-arm] GramRidge factorization at L%d (n=%d)", li, Xa.shape[0])
            self._gram5k[li] = (A.GramRidge(Xa), Ya)
        return self._gram5k[li]

    def apply(self, arm: str, layer: int, X_eval: np.ndarray) -> np.ndarray:
        """h(X_eval) (n, H) for a map arm at ``layer`` (raw activation space)."""
        if arm == "h_n5k_linear":
            gr, Ya = self.gram5k(layer)
            return gr.predict(Ya, X_eval)
        return N1M.apply_map(self.payload(layer, N1M_FITTERS[arm]), X_eval, self.dev)


def monitor_block(ctx: A.Ctx, maps: Maps, trait: str, li: int, cache: dict) -> dict:
    """All monitors for one (trait, layer): {"mat", "monitors"} (cached)."""
    key = (trait, int(li))
    if key in cache:
        return cache[key]
    mat = ctx.mat(trait, li)
    rb_l = ctx.rb(trait)[li]
    Xev = mat["c_last"]
    monitors: dict[str, np.ndarray] = {"pv_raw": mat["pv_raw"], "oracle": mat["oracle"]}
    for arm in MAP_ARMS:
        pred = maps.apply(arm, li, Xev)
        monitors[f"{arm}_dot"] = F.dot_readout(pred, rb_l)
        monitors[f"{arm}_cos"] = F.cosine_readout(pred, rb_l)
        logger.info("[%s L%d] %s applied (%d rows)", trait, li, arm, Xev.shape[0])
    cache[key] = {"mat": mat, "monitors": monitors}
    return cache[key]


def _round_list(vals, nd: int = 5) -> list[float]:
    return [round(float(v), nd) for v in vals]


def cell_entry(ctx: A.Ctx, maps: Maps, trait: str, mode: str, li: int, args, cache: dict) -> dict:
    """One (trait, mode, layer) read-out entry: per-monitor within-condition r +
    CI + per-condition r arrays + bootstrap replicates, plus delta-vs-raw and
    delta-vs-5k-arm CIs (with replicates)."""
    blk = monitor_block(ctx, maps, trait, li, cache)
    mat, monitors = blk["mat"], blk["monitors"]
    entry: dict = {"layer": int(li), "n_eval_rows": int(len(mat["y"])), "monitors": {}}
    for name, x in monitors.items():
        reps: dict = {}
        percond: dict = {}
        mm = S1.method_metrics(
            x,
            mat,
            n_boot=args.n_boot,
            seed=args.seed,
            replicates_out=reps,
            per_condition_out=percond,
        )
        entry["monitors"][name] = {
            **mm[mode],
            "overall_r_both_modes": mm["overall_r"],
            "per_condition_r": _round_list(percond[mode]),
            "bootstrap_replicates": _round_list(reps[mode]),
        }
    for ref_name, ref_key in (("pv_raw", "deltas_vs_pv_raw"), ("h_n5k_linear", "deltas_vs_h_n5k")):
        deltas: dict = {}
        for name in monitors:
            if name in ("pv_raw", "oracle"):
                continue
            if ref_name == "pv_raw":
                ref = monitors["pv_raw"]
            else:
                if name.startswith("h_n5k_linear"):
                    continue  # the 5k arm vs itself
                # matched readout: dot arm vs 5k dot, cos arm vs 5k cos
                ref = monitors["h_n5k_linear_dot" if name.endswith("_dot") else "h_n5k_linear_cos"]
            dreps: list = []
            d = A._delta_vs(
                monitors[name],
                ref,
                mat,
                mode,
                n_boot=args.n_boot,
                seed=args.seed,
                replicates_out=dreps,
            )
            d["bootstrap_replicates"] = _round_list(dreps)
            deltas[name] = d
        entry[ref_key] = deltas
    return entry


# ── SS7 validity gate ─────────────────────────────────────────────────────────


def validity_gate(res: dict, args) -> dict:
    """In-run pv_raw + 5k-arm recompute vs the committed parent values (+-0.02).

    evil vs ``arm_headline.json``; sycophancy vs ``arm_headline_pod.json``;
    hallucination EXCLUDED (substitute layers — no same-layer committed
    comparator; covered qualitatively by the L19 continuity read). A miss is a
    LOUD stderr warning + ``pass: false`` rows — never a silent pass."""
    comparators = {"evil": Path(args.comparator_evil), "sycophancy": Path(args.comparator_pod)}
    key_pairs = (
        ("pv_raw", "pv_raw"),
        ("h_A_lmsys_dot", "h_n5k_linear_dot"),
        ("h_A_lmsys_cos", "h_n5k_linear_cos"),
    )
    rows: list[dict] = []
    for trait, cpath in comparators.items():
        if trait not in res["headline"]:
            continue
        committed = json.loads(cpath.read_text())["arm_headline"][trait]
        for mode, ours in res["headline"][trait].items():
            cm = committed[mode]
            assert int(cm["layer"]) == int(ours["layer"]), (
                f"gate layer mismatch {trait}/{mode}: committed L{cm['layer']} vs "
                f"recomputed L{ours['layer']}"
            )
            for ck, ok in key_pairs:
                c_pt = float(cm["monitors"][ck]["point"])
                o_pt = float(ours["monitors"][ok]["point"])
                diff = abs(o_pt - c_pt)
                rows.append(
                    {
                        "trait": trait,
                        "mode": mode,
                        "layer": int(ours["layer"]),
                        "committed_key": ck,
                        "recomputed_key": ok,
                        "committed_point": c_pt,
                        "recomputed_point": o_pt,
                        "abs_diff": round(diff, 6),
                        "tol": GATE_TOL,
                        "comparator_file": str(cpath),
                        "pass": bool(diff <= GATE_TOL),
                    }
                )
    overall = all(r["pass"] for r in rows) and bool(rows)
    gate = {
        "tol": GATE_TOL,
        "n_checks": len(rows),
        "rows": rows,
        "hallucination_excluded": "substitute layers — no same-layer committed comparator",
        "overall_pass": overall,
    }
    for r in rows:
        if not r["pass"]:
            msg = (
                f"VALIDITY GATE MISS: {r['trait']}/{r['mode']} L{r['layer']} "
                f"{r['recomputed_key']}={r['recomputed_point']:.4f} vs committed "
                f"{r['committed_key']}={r['committed_point']:.4f} (|diff| "
                f"{r['abs_diff']:.4f} > {GATE_TOL}) — blocks interpretation; investigate."
            )
            print(f"WARNING: {msg}", file=sys.stderr, flush=True)
            logger.warning(msg)
    if overall:
        logger.info("[gate] SS7 validity gate PASS (%d checks, tol %.2f)", len(rows), GATE_TOL)
    return gate


def nonkrr_match_gate(res: dict, committed_path: Path) -> dict:
    """Recovery validity gate 3 (plan v11 §7): every recomputed NON-KRR arm value
    must match the committed ``n1m_readout.json`` (binding ±0.02; exact match
    expected — same machinery, same seed; realized v10 reproduced 12/12
    exactly). Only the ``n1m_krr_nystrom_*`` arm — the recovery's manipulated
    variable — is excluded. A miss WARNS LOUDLY + ``pass: false`` rows (blocks
    interpretation, never a silent pass), mirroring the SS7 validity gate."""
    committed = json.loads(committed_path.read_text())
    rows: list[dict] = []
    for trait, tmodes in res.get("headline", {}).items():
        for mode, ours in tmodes.items():
            cm = committed["headline"][trait][mode]
            assert int(cm["layer"]) == int(ours["layer"]), (
                f"non-KRR gate layer mismatch {trait}/{mode}: committed L{cm['layer']} vs "
                f"recomputed L{ours['layer']}"
            )
            for name, mm in ours["monitors"].items():
                if name.startswith("n1m_krr_nystrom"):
                    continue
                c_pt = float(cm["monitors"][name]["point"])
                o_pt = float(mm["point"])
                rows.append(
                    {
                        "surface": f"headline/{trait}/{mode}",
                        "monitor": name,
                        "committed": c_pt,
                        "recomputed": o_pt,
                        "abs_diff": round(abs(o_pt - c_pt), 6),
                        "tol": GATE_TOL,
                        "pass": bool(abs(o_pt - c_pt) <= GATE_TOL),
                    }
                )
    for trait, g in res.get("grouped", {}).items():
        cg = committed["grouped"][trait]["group_level"]
        for arm, entry in g["group_level"].items():
            if arm.startswith("n1m_krr_nystrom") or not isinstance(entry, dict):
                continue
            for readout in ("dot", "cos"):
                sub = entry.get(readout)
                if not isinstance(sub, dict) or "point" not in sub:
                    continue
                c_pt = float(cg[arm][readout]["point"])
                o_pt = float(sub["point"])
                rows.append(
                    {
                        "surface": f"grouped/{trait}/group_level",
                        "monitor": f"{arm}_{readout}",
                        "committed": c_pt,
                        "recomputed": o_pt,
                        "abs_diff": round(abs(o_pt - c_pt), 6),
                        "tol": GATE_TOL,
                        "pass": bool(abs(o_pt - c_pt) <= GATE_TOL),
                    }
                )
    overall = all(r["pass"] for r in rows) and bool(rows)
    gate = {
        "tol": GATE_TOL,
        "n_checks": len(rows),
        "rows": rows,
        "overall_pass": overall,
        "committed_readout": str(committed_path),
        "excluded": "n1m_krr_nystrom_* (the recovery round's manipulated variable)",
    }
    for r in rows:
        if not r["pass"]:
            msg = (
                f"NON-KRR MATCH GATE MISS: {r['surface']} {r['monitor']} recomputed "
                f"{r['recomputed']:.4f} vs committed {r['committed']:.4f} (|diff| "
                f"{r['abs_diff']:.4f} > {GATE_TOL}) — rig drift; blocks interpretation."
            )
            print(f"WARNING: {msg}", file=sys.stderr, flush=True)
            logger.warning(msg)
    if overall:
        logger.info("[gate] non-KRR match gate PASS (%d checks, tol %.2f)", len(rows), GATE_TOL)
    return gate


# ── grouped (persona-level) read ──────────────────────────────────────────────


def grouped_readout(ctx: A.Ctx, maps: Maps, trait: str, args) -> dict:
    """Persona-level grouped read at the trait's grouped layer with all arms.

    The parent 5k arm keeps its LOGO group-level-refit protocol
    (``_logo_readout``); the FIXED pre-fit maps (5k direct-apply + all n1m
    arms) are applied to the group-averaged context vectors — they never saw
    the corpus groups, so no leave-one-out is needed. Group score = pooled mean
    over the subset's VALID rollout scores (drop-never-coerce)."""
    li = GROUPED_LAYERS[trait]
    Xb, _vb, Yb = ctx.corpus_layer(trait, li)
    blob = ctx.corpus(trait)
    n_p, n_q, n_r = blob["n_personas"], blob["n_questions"], blob["n_rollouts"]
    rb_l = ctx.rb(trait)[li]
    scores = ctx.scores(trait)
    Xq = Xb.reshape(n_p, n_q, -1)
    Yq = Yb.reshape(n_p, n_q, -1)
    Sq = scores.reshape(n_p, n_q, n_r)

    fixed_arms = list(MAP_ARMS)

    def _fixed_reads(xg: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {}
        for arm in fixed_arms:
            pred = maps.apply(arm, li, xg)
            out[arm] = {"dot": F.dot_readout(pred, rb_l), "cos": F.cosine_readout(pred, rb_l)}
        return out

    # Full persona-level grouping (all 40 questions).
    all_q = np.tile(np.arange(n_q), (n_p, 1))
    Xg, Yg, Sg = A._grouped_vectors(Xq, Yq, Sq, all_q)
    group_level: dict = {
        "n_groups": int(n_p),
        "group_score_diag": A._label_diag(Sg),
        "pv_raw_group": {
            "dot": A._pearson_boot_ci(Xg @ rb_l, Sg, n_boot=args.n_boot, seed=args.seed)
        },
        "oracle_group": {
            "dot": A._pearson_boot_ci(Yg @ rb_l, Sg, n_boot=args.n_boot, seed=args.seed)
        },
    }
    logo_d, logo_c = A._logo_readout(Xg, Yg, rb_l)
    group_level["h_n5k_logo"] = {
        "dot": A._pearson_boot_ci(logo_d, Sg, n_boot=args.n_boot, seed=args.seed),
        "cos": A._pearson_boot_ci(logo_c, Sg, n_boot=args.n_boot, seed=args.seed),
        "protocol": "parent LOGO group-level refit (_logo_readout)",
    }
    for arm, reads in _fixed_reads(Xg).items():
        group_level[arm] = {
            k: A._pearson_boot_ci(v, Sg, n_boot=args.n_boot, seed=args.seed)
            for k, v in reads.items()
        }
        group_level[arm]["protocol"] = "fixed pre-fit map applied to group-mean c (no LOGO needed)"
    logger.info(
        "[grouped %s L%d] full-40 dot r: logo=%.3f 5k-apply=%.3f n1m_mlp32768=%.3f",
        trait,
        li,
        group_level["h_n5k_logo"]["dot"]["point"],
        group_level["h_n5k_linear"]["dot"]["point"],
        group_level["n1m_mlp_w32768"]["dot"]["point"],
    )

    # Group-size sweep {1,2,5,10,20,40}, K draws each (size 40 = 1 draw); ONE
    # q_idx draw per (size, k) shared by every arm (paired comparison).
    sweep: dict[str, dict] = {}
    rng = np.random.default_rng(args.seed)
    for s in GROUP_SIZES:
        n_draws = 1 if s == n_q else args.k_draws
        per_arm: dict[str, list[float]] = {"h_n5k_logo": []}
        per_arm.update({a: [] for a in fixed_arms})
        per_arm["pv_raw_group"] = []
        for _k in range(n_draws):
            q_idx = np.stack([rng.choice(n_q, size=s, replace=False) for _ in range(n_p)])
            xg, yg, sg = A._grouped_vectors(Xq, Yq, Sq, q_idx)
            fin = np.isfinite(sg)
            dts, _css = A._logo_readout(xg, yg, rb_l)
            per_arm["h_n5k_logo"].append(M.overall_pearson(dts[fin], sg[fin]))
            per_arm["pv_raw_group"].append(M.overall_pearson((xg @ rb_l)[fin], sg[fin]))
            for arm, reads in _fixed_reads(xg).items():
                per_arm[arm].append(M.overall_pearson(reads["dot"][fin], sg[fin]))
        sweep[str(s)] = {
            arm: {
                "dot_r_draws": _round_list(v),
                "dot_r_mean": float(np.nanmean(v)),
                "dot_r_sd": float(np.nanstd(v)),
            }
            for arm, v in per_arm.items()
        }
        logger.info(
            "[grouped %s] size %2d: logo=%.3f 5k=%.3f mlp32k=%.3f",
            trait,
            s,
            sweep[str(s)]["h_n5k_logo"]["dot_r_mean"],
            sweep[str(s)]["h_n5k_linear"]["dot_r_mean"],
            sweep[str(s)]["n1m_mlp_w32768"]["dot_r_mean"],
        )

    # Per-context (ungrouped) apply read for the fixed arms; the parent's
    # per-context LOGO baseline is cited from the committed JSON, not re-run.
    with np.errstate(invalid="ignore"):
        gb = np.nanmean(scores, axis=1)
    per_context: dict = {
        "n": int(np.isfinite(gb).sum()),
        "note": (
            "fixed pre-fit maps applied per-context (n=2400); the parent per-context "
            "LOGO baseline lives in the committed arm_headline.json .grouped_contexts"
        ),
    }
    for arm, reads in _fixed_reads(Xb).items():
        per_context[arm] = {
            k: A._pearson_boot_ci(v, gb, n_boot=args.n_boot, seed=args.seed)
            for k, v in reads.items()
        }
    return {
        "layer": int(li),
        "substituted": trait == "hallucination",
        "group_level": group_level,
        "group_size_sweep": sweep,
        "per_context_apply": per_context,
    }


# ── figures ───────────────────────────────────────────────────────────────────


def _bar_monitor_names() -> list[tuple[str, str]]:
    out = [("pv_raw", "PV raw projection")]
    out += [(f"{a}_dot", a.replace("_", " ")) for a in MAP_ARMS]
    out.append(("oracle", "oracle (true answer proj.)"))
    return out


def make_figures(res: dict, fits: dict | None, args) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    figs: dict = {}
    traits = [t for t in C.TRAITS if t in res["headline"]]
    bars = _bar_monitor_names()
    colors = paper_palette(3)
    bar_colors = [colors[0]] + [colors[1]] + [colors[2]] * len(N1M_FITTERS) + [colors[0]]
    bar_colors.insert(1, colors[1])  # h_n5k_linear shares the map color family

    # HERO 1: grouped bars of within-condition r across the arms (dot readout).
    fig, axes = plt.subplots(
        2,
        max(1, len(traits)),
        figsize=(4.6 * max(1, len(traits)) + 0.6, 8.6),
        squeeze=False,
        layout="tight",
    )
    for col, trait in enumerate(traits):
        for row, mode in enumerate(MODES):
            ax = axes[row][col]
            entry = res["headline"][trait].get(mode)
            if entry is None:
                ax.set_axis_off()
                continue
            heights, errs, labels = [], [], []
            for name, label in bars:
                mm = entry["monitors"][name]
                pt = mm["point"]
                if not np.isfinite(pt):
                    continue
                heights.append(pt)
                lo, hi = mm["lo"], mm["hi"]
                errs.append(
                    [
                        max(0.0, pt - lo) if np.isfinite(lo) else 0.0,
                        max(0.0, hi - pt) if np.isfinite(hi) else 0.0,
                    ]
                )
                labels.append(label)
            ax.bar(
                range(len(heights)),
                heights,
                yerr=np.array(errs).T if errs else None,
                capsize=2,
                color=bar_colors[: len(heights)],
            )
            ax.axhline(0.0, color="gray", lw=0.6)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            mode_lbl = "system prompting" if mode == "system" else "many-shot"
            sub = " (substitute layer)" if f"{trait}/{mode}" in SUBSTITUTED_CELLS else ""
            ax.set_title(f"{trait} — {mode_lbl} (L{entry['layer']}{sub})")
            if col == 0:
                ax.set_ylabel("within-condition Pearson r (dot readout)")
    out = savefig_paper(fig, "n1m_readout_hero", dir=args.fig_dir)
    plt.close(fig)
    figs["hero"] = str(out.get("png", ""))

    # HERO 2: delta-vs-raw forest (dot readout) with the +0.05 bar.
    fig, axes = plt.subplots(
        1,
        max(1, len(traits)),
        figsize=(4.8 * max(1, len(traits)), 5.2),
        squeeze=False,
        layout="tight",
    )
    arm_list = [f"{a}_dot" for a in MAP_ARMS]
    for col, trait in enumerate(traits):
        ax = axes[0][col]
        ypos, ylab = 0, []
        for mode in MODES:
            entry = res["headline"][trait].get(mode)
            if entry is None:
                continue
            for name in arm_list:
                d = entry["deltas_vs_pv_raw"].get(name)
                if d is None or not np.isfinite(d.get("delta", float("nan"))):
                    ypos += 1
                    ylab.append(f"{mode}: {name}")
                    continue
                ax.errorbar(
                    d["delta"],
                    ypos,
                    xerr=[[max(0.0, d["delta"] - d["lo"])], [max(0.0, d["hi"] - d["delta"])]],
                    fmt="o",
                    capsize=2,
                    color=colors[0] if mode == "system" else colors[1],
                )
                ylab.append(f"{mode}: {name.replace('_dot', '').replace('_', ' ')}")
                ypos += 1
            ypos += 1
            ylab.append("")
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.axvline(0.05, color=colors[2], lw=1.0, ls="--", label="+0.05 pre-registered bar")
        ax.set_yticks(range(len(ylab)))
        ax.set_yticklabels(ylab, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("delta within-condition r vs raw")
        ax.set_title(trait)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right")
    out = savefig_paper(fig, "n1m_readout_delta_forest", dir=args.fig_dir)
    plt.close(fig)
    figs["delta_forest"] = str(out.get("png", ""))

    # Grouped sweep: r vs group size, parent LOGO vs fixed arms.
    gtraits = [t for t in C.TRAITS if t in res.get("grouped", {})]
    if gtraits:
        fig, axes = plt.subplots(
            1, len(gtraits), figsize=(5.0 * len(gtraits), 4.6), squeeze=False, layout="tight"
        )
        sweep_arms = ["h_n5k_logo", "h_n5k_linear", *N1M_FITTERS, "pv_raw_group"]
        pal = paper_palette(max(3, len(sweep_arms)))
        for col, trait in enumerate(gtraits):
            ax = axes[0][col]
            d = res["grouped"][trait]["group_size_sweep"]
            sizes = sorted(int(s) for s in d)
            for ai, arm in enumerate(sweep_arms):
                means = [d[str(s)][arm]["dot_r_mean"] for s in sizes]
                sds = [d[str(s)][arm]["dot_r_sd"] for s in sizes]
                ax.errorbar(
                    sizes,
                    means,
                    yerr=sds,
                    marker="o",
                    ms=3,
                    capsize=2,
                    color=pal[ai % len(pal)],
                    label=arm.replace("_", " "),
                )
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(s) for s in sizes])
            ax.set_xlabel("questions averaged per persona group")
            ax.set_ylabel("Pearson r vs mean judge score (dot)")
            ax.set_title(f"{trait} (L{res['grouped'][trait]['layer']})")
            ax.legend(fontsize=6, loc="lower right")
        out = savefig_paper(fig, "n1m_readout_grouped_sweep", dir=args.fig_dir)
        plt.close(fig)
        figs["grouped_sweep"] = str(out.get("png", ""))

    # Fit-quality transfer: test/val R2 at each layer per fitter (from fits JSON).
    if fits:
        fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="tight")
        layers = sorted(int(k) for k in fits.get("per_layer", {}))
        pal = paper_palette(max(3, len(N1M_FITTERS)))
        for fi, (arm, fitter) in enumerate(N1M_FITTERS.items()):
            r2s = []
            for li in layers:
                cell = (
                    fits["per_layer"][str(li)]["per_point"]
                    .get("mixed_1m", {})
                    .get("predictors", {})
                    .get(fitter)
                )
                r2s.append(cell["whole_map_r2"] if cell else np.nan)
            ax.plot(layers, r2s, marker="o", color=pal[fi % len(pal)], label=arm.replace("_", " "))
        ax.set_xticks(layers)
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out whole-map R2 (pinned test)")
        ax.set_title("n1m map fit quality across capture layers")
        ax.legend(fontsize=7)
        out = savefig_paper(fig, "n1m_readout_r2_transfer", dir=args.fig_dir)
        plt.close(fig)
        figs["r2_transfer"] = str(out.get("png", ""))

    # Dot-vs-cos comparison per (arm, cell).
    fig, ax = plt.subplots(figsize=(5.2, 5.0), layout="tight")
    pal = paper_palette(max(3, len(MAP_ARMS)))
    for ai, arm in enumerate(MAP_ARMS):
        xs, ys = [], []
        for trait in traits:
            for mode in MODES:
                entry = res["headline"][trait].get(mode)
                if entry is None:
                    continue
                xs.append(entry["monitors"][f"{arm}_dot"]["point"])
                ys.append(entry["monitors"][f"{arm}_cos"]["point"])
        ax.scatter(xs, ys, s=22, color=pal[ai % len(pal)], label=arm.replace("_", " "))
    lim = ax.get_xlim()
    ax.plot(lim, lim, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("within-condition r — dot readout")
    ax.set_ylabel("within-condition r — cosine readout")
    ax.set_title("dot vs cosine readout per (arm, cell)")
    ax.legend(fontsize=7)
    out = savefig_paper(fig, "n1m_readout_dot_vs_cos", dir=args.fig_dir)
    plt.close(fig)
    figs["dot_vs_cos"] = str(out.get("png", ""))

    # Exploratory: per-condition scatter per arm (one figure per trait).
    for trait in traits:
        fig, axes = plt.subplots(
            2, len(ARM_ORDER), figsize=(2.6 * len(ARM_ORDER), 6.0), squeeze=False, layout="tight"
        )
        for row, mode in enumerate(MODES):
            entry = res["headline"][trait].get(mode)
            for coli, arm in enumerate(ARM_ORDER):
                ax = axes[row][coli]
                if entry is None:
                    ax.set_axis_off()
                    continue
                blk = res["_scatter_cache"][trait][mode]
                name = arm if arm in ("pv_raw", "oracle") else f"{arm}_dot"
                x = np.asarray(blk["monitors"][name])
                y = np.asarray(blk["y"])
                cond = np.asarray(blk["cond"])
                msel = np.asarray([m == mode for m in blk["mode"]])
                ax.scatter(x[msel], y[msel], c=cond[msel], cmap="tab20", s=6, alpha=0.7)
                if row == 0:
                    ax.set_title(arm.replace("_", " "), fontsize=7)
                if coli == 0:
                    ax.set_ylabel(f"{mode}\njudge score", fontsize=7)
                ax.tick_params(labelsize=6)
        fig.suptitle(f"{trait}: per-condition monitor-vs-score scatter (dot readout)")
        out = savefig_paper(fig, f"n1m_readout_percond_scatter_{trait}", dir=args.fig_dir)
        plt.close(fig)
        figs[f"percond_scatter_{trait}"] = str(out.get("png", ""))
    return figs


def make_recovery_figures(res: dict, args) -> dict:
    """Figures for the l26-kernel-gate-recovery round (plan v11 §6): the re-read
    delta-vs-raw forest (daggers removed or RETAINED per the m=32768 gate
    verdict), the sycophancy grouped group-size sweep, and the exploratory
    gap-vs-m point pair + solver-equivalence residuals. Plain-English arm
    labels from ``issue779_n1m_regen_figs.ARM_LABELS``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    figs: dict = {}
    rec = res["l26_recovery"]
    gate = rec["gate"]
    dagger = not bool(gate.get("gate_passed"))
    colors = paper_palette(3)

    def _label(arm: str) -> str:
        base = RF.ARM_LABELS.get(arm, arm.replace("_", " "))
        if arm == "n1m_krr_nystrom" and dagger:
            return base + " †"
        return base

    # 1. Delta-vs-raw forest over the re-read L26 cells (dot readout, +0.05 bar).
    traits = [t for t in C.TRAITS if t in res.get("headline", {})]
    fig, axes = plt.subplots(
        1,
        max(1, len(traits)),
        figsize=(4.8 * max(1, len(traits)), 5.2),
        squeeze=False,
        layout="tight",
    )
    arm_list = [f"{a}_dot" for a in MAP_ARMS]
    for col, trait in enumerate(traits):
        ax = axes[0][col]
        ypos, ylab = 0, []
        for mode in MODES:
            entry = res["headline"][trait].get(mode)
            if entry is None:
                continue
            for name in arm_list:
                d = entry["deltas_vs_pv_raw"].get(name)
                if d is None or not np.isfinite(d.get("delta", float("nan"))):
                    ypos += 1
                    ylab.append(f"{MODE_LBL[mode]}: {_label(name[: -len('_dot')])}")
                    continue
                ax.errorbar(
                    d["delta"],
                    ypos,
                    xerr=[[max(0.0, d["delta"] - d["lo"])], [max(0.0, d["hi"] - d["delta"])]],
                    fmt="o",
                    capsize=2,
                    color=colors[0] if mode == "system" else colors[1],
                )
                ylab.append(f"{MODE_LBL[mode]}: {_label(name[: -len('_dot')])}")
                ypos += 1
            ypos += 1
            ylab.append("")
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.axvline(0.05, color=colors[2], lw=1.0, ls="--", label="+0.05 pre-registered bar")
        ax.set_yticks(range(len(ylab)))
        ax.set_yticklabels(ylab, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("delta within-condition r vs raw")
        ax.set_title(f"{trait} (L26 re-read)")
        if col == 0:
            ax.legend(fontsize=7, loc="lower right")
    out = savefig_paper(fig, "l26_recovery_delta_forest", dir=args.fig_dir)
    plt.close(fig)
    figs["delta_forest"] = str(out.get("png", ""))

    # 2. Sycophancy grouped group-size sweep (only GROUPED_LAYERS == 26 traits).
    gtraits = [t for t in C.TRAITS if t in res.get("grouped", {})]
    if gtraits:
        fig, axes = plt.subplots(
            1, len(gtraits), figsize=(5.0 * len(gtraits), 4.6), squeeze=False, layout="tight"
        )
        sweep_arms = ["h_n5k_logo", "h_n5k_linear", *N1M_FITTERS, "pv_raw_group"]
        pal = paper_palette(max(3, len(sweep_arms)))
        for col, trait in enumerate(gtraits):
            ax = axes[0][col]
            d = res["grouped"][trait]["group_size_sweep"]
            sizes = sorted(int(s) for s in d)
            for ai, arm in enumerate(sweep_arms):
                means = [d[str(s)][arm]["dot_r_mean"] for s in sizes]
                sds = [d[str(s)][arm]["dot_r_sd"] for s in sizes]
                ax.errorbar(
                    sizes,
                    means,
                    yerr=sds,
                    marker="o",
                    ms=3,
                    capsize=2,
                    color=pal[ai % len(pal)],
                    label=_label(arm),
                )
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(s) for s in sizes])
            ax.set_xlabel("questions averaged per persona group")
            ax.set_ylabel("Pearson r vs mean judge score (dot)")
            ax.set_title(f"{trait} (L{res['grouped'][trait]['layer']} re-read)")
            ax.legend(fontsize=6, loc="lower right")
        out = savefig_paper(fig, "l26_recovery_grouped_sweep", dir=args.fig_dir)
        plt.close(fig)
        figs["grouped_sweep"] = str(out.get("png", ""))

    # 3. Exploratory: Nystrom-vs-exact gap vs m at L26 (committed 16384 vs new 32768).
    prior = rec.get("committed_m16384_gate") or {}
    pts = [(g.get("m_centers"), g.get("gap")) for g in (prior, gate)]
    pts = [(m, g) for m, g in pts if m is not None and g is not None]
    if pts:
        # gate["tol"] can be None when the recovery gate was skipped (smoke /
        # --no-validate-krr) and the fallback dict carries None-valued keys.
        tol_line = float(gate.get("tol") or prior.get("tol") or 0.01)
        fig, ax = plt.subplots(figsize=(4.6, 4.0), layout="tight")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=colors[0])
        ax.axhline(tol_line, color=colors[2], lw=1.0, ls="--", label="gate tol 0.01")
        ax.set_xscale("log", base=2)
        ax.set_xticks([p[0] for p in pts])
        ax.set_xticklabels([str(p[0]) for p in pts])
        ax.set_xlabel("Nystrom landmarks m")
        ax.set_ylabel("|R2 Nystrom - R2 exact| (n=50k gate slice)")
        ax.set_title("L26 kernel gate gap vs m")
        ax.legend(fontsize=7)
        out = savefig_paper(fig, "l26_recovery_gap_vs_m", dir=args.fig_dir)
        plt.close(fig)
        figs["gap_vs_m"] = str(out.get("png", ""))

    # 4. Exploratory: solver-equivalence residuals (real-data leg).
    eq = rec.get("solver_equivalence")
    if isinstance(eq, dict):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6), layout="tight")
        axes[0].bar([0], [eq["abs_dr2"]], color=colors[0])
        axes[0].axhline(eq["tol"], color=colors[2], lw=1.0, ls="--", label=f"tol {eq['tol']:g}")
        axes[0].set_xticks([0])
        axes[0].set_xticklabels([f"m={eq['m']}"])
        axes[0].set_ylabel("|R2 cholesky - R2 eigh|")
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=7)
        axes[1].bar([0], [max(eq["max_abs_dpred"], 1e-300)], color=colors[1])
        axes[1].set_xticks([0])
        axes[1].set_xticklabels([f"m={eq['m']}"])
        axes[1].set_ylabel("max |pred cholesky - pred eigh|")
        axes[1].set_yscale("log")
        fig.suptitle("solver-equivalence residuals (real-data leg)")
        out = savefig_paper(fig, "l26_recovery_solver_equiv", dir=args.fig_dir)
        plt.close(fig)
        figs["solver_equiv"] = str(out.get("png", ""))
    return figs


MODE_LBL = {"system": "system", "many_shot": "many-shot"}


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="Issue #779 n1m-map behavior read-out.")
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument(
        "--fits-json",
        type=Path,
        default=PROJECT_ROOT
        / "eval_results"
        / "issue_779"
        / "n1m-nonlinear-map-behavior-readout"
        / "n1m_multilayer_fits.json",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT
        / "eval_results"
        / "issue_779"
        / "n1m-nonlinear-map-behavior-readout"
        / "n1m_readout.json",
    )
    p.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_779")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-draws", type=int, default=5)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--traits", default=",".join(C.TRAITS))
    p.add_argument("--skip-grouped", action="store_true")
    p.add_argument("--collect-dir", type=Path, default=None)
    p.add_argument("--corpus-dir", type=Path, default=None)
    p.add_argument("--stage-inputs", action="store_true", help="download rig inputs from HF first")
    p.add_argument("--stage-only", action="store_true", help="stage inputs then exit 0")
    p.add_argument(
        "--comparator-evil",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "arm_headline.json",
    )
    p.add_argument(
        "--comparator-pod",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "arm_headline_pod.json",
    )
    p.add_argument("--fresh", action="store_true", help="ignore an existing output JSON")
    p.add_argument(
        "--l26-kernel-recovery",
        action="store_true",
        help="plan v11 scoped re-run: ONLY the cells consuming the L26 kernel arm (the 4 "
        "READ_LAYERS==26 headline cells, all arms recomputed for the non-KRR byte-match "
        "assert, + the GROUPED_LAYERS==26 grouped read + the fit-quality L26 row); no "
        "L19 continuity read; figures emitted as l26_recovery_*",
    )
    p.add_argument(
        "--krr-weights-override",
        type=Path,
        default=None,
        help="explicit path to the recovery L26 krr_nystrom payload (m=32768 Cholesky "
        "refit); the v10 payloads under --weights-dir stay read-only",
    )
    p.add_argument(
        "--recovery-fits-json",
        type=Path,
        default=None,
        help="l26_recovery_fits.json from the recovery fits run (gate verdict + fit row)",
    )
    p.add_argument(
        "--committed-readout",
        type=Path,
        default=PROJECT_ROOT
        / "eval_results"
        / "issue_779"
        / "n1m-nonlinear-map-behavior-readout"
        / "n1m_readout.json",
        help="committed parent readout JSON for the non-KRR byte-match gate",
    )
    args = p.parse_args()
    torch.set_num_threads(int(args.n_threads))
    dev = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    traits = tuple(t.strip() for t in args.traits.split(",") if t.strip())
    assert traits and all(t in C.TRAITS for t in traits), traits

    recovery = bool(args.l26_kernel_recovery)
    if recovery:
        if args.krr_weights_override is None or not args.krr_weights_override.exists():
            raise SystemExit(
                f"--l26-kernel-recovery requires --krr-weights-override pointing at the "
                f"recovery L26 krr payload (got {args.krr_weights_override})"
            )
        if args.recovery_fits_json is None or not args.recovery_fits_json.exists():
            raise SystemExit(
                f"--l26-kernel-recovery requires --recovery-fits-json (got "
                f"{args.recovery_fits_json})"
            )
        committed_default = (
            PROJECT_ROOT
            / "eval_results"
            / "issue_779"
            / "n1m-nonlinear-map-behavior-readout"
            / "n1m_readout.json"
        )
        if args.out_json.resolve() == committed_default.resolve():
            raise SystemExit(
                "--l26-kernel-recovery must write to the follow-up artifact dir, not the "
                "committed n1m_readout.json — pass --out-json (no-clobber guard)"
            )

    if args.collect_dir:
        A.COLLECT_DIR = Path(args.collect_dir)
    if args.corpus_dir:
        A.CORPUS_DIR = Path(args.corpus_dir)
    if args.stage_inputs or args.stage_only:
        stage_inputs(A.COLLECT_DIR, A.CORPUS_DIR, traits)
        if args.stage_only:
            logger.info("[stage] inputs staged; exiting (--stage-only)")
            return 0

    params = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "k_draws": args.k_draws,
        "weights_dir": str(args.weights_dir),
    }
    if recovery:  # keyed into the resume-params check ONLY for recovery JSONs
        params["l26_kernel_recovery"] = True
        params["krr_weights_override"] = str(args.krr_weights_override)
    res: dict = {}
    if args.out_json.exists() and not args.fresh:
        res = json.loads(args.out_json.read_text())
        if not recovery and res.get("metadata", {}).get("l26_kernel_recovery"):
            # symmetry guard: recovery keys enter `params` only in recovery mode, so
            # without this a non-recovery invocation pointed at a recovery out-json
            # with matching base params would resume it and extend with MIXED weights
            # provenance (code-review v17 Minor).
            raise SystemExit(
                f"existing {args.out_json} is an --l26-kernel-recovery artifact; refusing "
                "a non-recovery resume onto it (mixed weights provenance) — pass --fresh "
                "to overwrite or re-run with --l26-kernel-recovery"
            )
        prior = {k: res.get("metadata", {}).get(k) for k in params}
        if prior != params:
            raise SystemExit(
                f"existing {args.out_json} was produced with params {prior} != {params}; "
                "pass --fresh to overwrite or match the params"
            )
        logger.info("Resuming from existing %s", args.out_json)
    res["metadata"] = C.reproducibility_metadata(
        {
            "script": "issue779_n1m_readout",
            **params,
            "read_layers": READ_LAYERS,
            "substituted_cells": SUBSTITUTED_CELLS,
            "grouped_layers": GROUPED_LAYERS,
            "traits": list(traits),
            "device": args.device,
            "arms": list(ARM_ORDER),
        }
    )

    ctx = A.Ctx(args)
    res["metadata"]["equivalence_gate"] = A.equivalence_gate(ctx.bundle, args.seed)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.out_json, res)

    maps = Maps(
        args.weights_dir,
        ctx,
        dev,
        krr_l26_override=args.krr_weights_override if recovery else None,
    )
    cache: dict = {}

    # Headline cells at frozen/substitute layers (checkpoint per cell). In
    # recovery mode ONLY the READ_LAYERS==26 cells (the L26 kernel consumers).
    headline = res.setdefault("headline", {})
    for trait in traits:
        tr = headline.setdefault(trait, {})
        for mode in MODES:
            if recovery and READ_LAYERS[trait][mode] != RECOVERY_LAYER:
                continue
            if mode in tr:
                logger.info("[headline %s %s] checkpointed; skip", trait, mode)
                continue
            li = READ_LAYERS[trait][mode]
            entry = cell_entry(ctx, maps, trait, mode, li, args, cache)
            if f"{trait}/{mode}" in SUBSTITUTED_CELLS:
                entry["substitution"] = SUBSTITUTED_CELLS[f"{trait}/{mode}"]
            tr[mode] = entry
            C.write_json_atomic(args.out_json, res)
            logger.info(
                "[headline %s %s L%d] pv_raw=%.3f 5k_dot=%.3f | n1m dot r/mlp8k/mlp32k/krr = "
                "%.3f/%.3f/%.3f/%.3f | oracle=%.3f",
                trait,
                mode,
                li,
                entry["monitors"]["pv_raw"]["point"],
                entry["monitors"]["h_n5k_linear_dot"]["point"],
                *[entry["monitors"][f"{a}_dot"]["point"] for a in N1M_FITTERS],
                entry["monitors"]["oracle"]["point"],
            )

    # SS7 validity gate (evil + sycophancy vs committed comparators).
    res["validity_gate"] = validity_gate(res, args)
    C.write_json_atomic(args.out_json, res)

    # L19 continuity read (all requested traits x modes, all arms, at L19).
    # Recovery mode: NOT re-run — the L19 kernel is unchanged, gate passed.
    if not recovery:
        l19 = res.setdefault("l19_continuity", {})
        for trait in traits:
            tr = l19.setdefault(trait, {})
            for mode in MODES:
                if mode in tr:
                    continue
                tr[mode] = cell_entry(ctx, maps, trait, mode, L19_CONTINUITY_LAYER, args, cache)
                C.write_json_atomic(args.out_json, res)
        logger.info("[l19] continuity read complete (%d traits)", len(traits))
    else:
        logger.info("[l19] SKIPPED (--l26-kernel-recovery: L19 kernel unchanged)")

    # Grouped persona-level read (recovery: ONLY the GROUPED_LAYERS==26 traits).
    if not args.skip_grouped:
        grouped = res.setdefault("grouped", {})
        for trait in traits:
            if recovery and GROUPED_LAYERS[trait] != RECOVERY_LAYER:
                continue
            if trait in grouped:
                logger.info("[grouped %s] checkpointed; skip", trait)
                continue
            grouped[trait] = grouped_readout(ctx, maps, trait, args)
            C.write_json_atomic(args.out_json, res)
    else:
        logger.info("[grouped] SKIPPED (--skip-grouped)")

    # Recovery bookkeeping: merge the recovery fits (L26 kernel entry + gate)
    # over the committed fits view; record the gate verdict + equivalence leg.
    if recovery:
        rec_fits = json.loads(args.recovery_fits_json.read_text())
        rec_l26 = rec_fits["per_layer"][str(RECOVERY_LAYER)]
        rec_krr = rec_l26["per_point"]["mixed_1m"]["predictors"]["krr_nystrom"]
        gate32 = rec_l26.get("nystrom_validation") or {
            "gate_passed": False,
            "note": "gate absent from recovery fits (smoke / --no-validate-krr) — "
            "dagger RETAINED (conservative)",
        }
        res["l26_recovery"] = {
            "recovery_fits_json": str(args.recovery_fits_json),
            "krr_weights_override": str(args.krr_weights_override),
            "gate": {
                k: gate32.get(k)
                for k in (
                    "n",
                    "m_centers",
                    "solver",
                    "exact_r2",
                    "nystrom_r2",
                    "gap",
                    "tol",
                    "gate_passed",
                    "note",
                )
            },
            "solver_equivalence": rec_l26.get("solver_equivalence"),
            "krr_fit": {
                "whole_map_r2": rec_krr.get("whole_map_r2"),
                "mean_cosine": rec_krr.get("mean_cosine"),
                "fit_meta": rec_krr.get("fit_meta"),
            },
        }
        C.write_json_atomic(args.out_json, res)

    # Fit-quality summary from the multilayer fits JSON (exploratory transfer).
    fits = None
    if args.fits_json.exists():
        fits = json.loads(args.fits_json.read_text())
        if recovery:
            # merged view: committed fits with the L26 kernel row + gate REPLACED
            # by the recovery refit (the m=16384 gate is kept for the gap-vs-m fig).
            l26 = fits["per_layer"].setdefault(str(RECOVERY_LAYER), {"per_point": {}})
            res["l26_recovery"]["committed_m16384_gate"] = l26.get("nystrom_validation")
            l26["nystrom_validation"] = res["l26_recovery"]["gate"]
            l26.setdefault("per_point", {}).setdefault("mixed_1m", {}).setdefault("predictors", {})[
                "krr_nystrom"
            ] = res["l26_recovery"]["krr_fit"]
            C.write_json_atomic(args.out_json, res)
        res["fit_quality"] = {
            "fits_json": str(args.fits_json),
            "per_layer_test_r2": {
                lk: {
                    f: lv["per_point"]
                    .get("mixed_1m", {})
                    .get("predictors", {})
                    .get(f, {})
                    .get("whole_map_r2")
                    for f in N1M_FITTERS.values()
                }
                for lk, lv in fits.get("per_layer", {}).items()
            },
        }
    else:
        logger.warning(
            "[fit-quality] fits json absent at %s; skipping transfer summary", args.fits_json
        )

    if recovery:
        # Recovery validity gate 3 (non-KRR byte-match) + scoped l26_recovery figures.
        res["nonkrr_match_gate"] = nonkrr_match_gate(res, args.committed_readout)
        C.write_json_atomic(args.out_json, res)
        res["figures"] = make_recovery_figures(res, args)
        C.write_json_atomic(args.out_json, res)
        logger.info(
            "Done (l26-kernel-recovery). gate_passed=%s nonkrr_match=%s. Wrote %s",
            res["l26_recovery"]["gate"].get("gate_passed"),
            res["nonkrr_match_gate"]["overall_pass"],
            args.out_json,
        )
        return 0

    # Figures (scatter cache holds per-cell raw arrays; not persisted to JSON).
    res["_scatter_cache"] = {
        trait: {
            mode: {
                "monitors": {
                    k: [float(v) for v in x]
                    for k, x in monitor_block(ctx, maps, trait, READ_LAYERS[trait][mode], cache)[
                        "monitors"
                    ].items()
                },
                "y": [float(v) for v in cache[(trait, READ_LAYERS[trait][mode])]["mat"]["y"]],
                "cond": [int(v) for v in cache[(trait, READ_LAYERS[trait][mode])]["mat"]["cond"]],
                "mode": [str(v) for v in cache[(trait, READ_LAYERS[trait][mode])]["mat"]["mode"]],
            }
            for mode in MODES
            if mode in res["headline"].get(trait, {})
        }
        for trait in traits
    }
    res["figures"] = make_figures(res, fits, args)
    res.pop("_scatter_cache", None)
    C.write_json_atomic(args.out_json, res)
    logger.info("Done. Wrote %s", args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
