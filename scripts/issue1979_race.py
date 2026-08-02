"""#1979 F3 stats — per-(arm, prefix) DV assembly + predictor race (plan v2 §3/§5/§6).

VM-side phase. Inputs: F1e ingredient tensors / anchors / marker slots (staged
from the HF data repo), F2 judge outputs (VM-resident under
``eval_results/issue_1979/judge/``), F0 config manifests. Outputs per plan §10
``phase_outputs`` to ``eval_results/issue_1979/race/``.

Per content arm: level / change / binary DVs at the row-coverage rule (a prefix
is valid iff >= 75% of realized queries carry >= 1 valid judge draw — the plan
literal 45/60 expressed as a fraction so tiny fixture slices scale); per marker
arm: the three-space slot DVs (dlogp primary, EOS-margin secondary, prob
sanity; base-slot P7). Candidate roster P1..P10 (+P8a/b content) assembled from
the F1e tensors at the pre-registered coordinates (content L19 / marker L25;
``last_prompt`` primary, span-mean secondary), centered at the panel mean
(stated shared centering, plan §6). Battery per arm: observed Spearman, THREE
bootstrap families (prefix resample PRIMARY / query-cluster / family-cluster —
B=2,000 each, shared draw streams, winner re-selected inside every draw),
permutation null (per-draw signed max over candidates), all persisted as
per-draw x per-candidate matrices (``race/{boot,qboot,fboot,perm}_<arm>.npz``).
Reuses ``issue1900_race.{bootstrap_battery, observed_rho, perm_null,
mediation_arm, mediation_lattice}`` (plan §10 reused-code table).

Fixture smoke: ``--build-fixtures <dir>`` writes a schema-exact tiny fixture
tree (clearly labeled, never uploaded); the SAME production entrypoint then
runs against it end-to-end (``--inputs-root <dir>/inputs --judge-dir
<dir>/judge --config-dir <dir>/config``).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before numpy/torch: shared-VM thread caps + HF credentials

import json  # noqa: E402
import argparse  # noqa: E402
import hashlib  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1900_judge as J  # noqa: E402
import issue1900_race as R  # noqa: E402  (bootstrap/observed/perm/mediation reused)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1979.race")

ISSUE = 1979
SEED = 1979
HF_PREFIX_1979 = "issue1979_prefixrace"
B_BOOT = 2_000  # plan §9 (#1768/#1900 CI convention)
N_PERM = 1_000
LAYERS_1979 = (14, 19, 25)
PRIMARY = {"content": (19, "last_prompt"), "marker": (25, "last_prompt")}  # plan §6 pre-registered
SECONDARY_POS = "span_mean_context"
COVERAGE_FRAC = 0.75  # plan §3 row-coverage: >=45/60 queries valid, as a fraction
CI_QS = (0.025, 0.975)
K_NN = 8  # P9/P10 top-k (k in {4,16} dump)
CEILING_SCORE = 90.0
ANCHOR_KEY_BY_POS = {
    "span_mean_context": "A_ctx_span",
    "last_prompt": "A_ctx_last_prompt",
    "last_ctx": "A_ctx_last_ctx",
}
# ctx_key -> the arm's own trained prefix panel member (panel-source anchor)
OWN_PREFIX_BY_CTX = {
    "pers": "persona_software_engineer",
    "bare": "bare",
    "wc": "wildchat_prefix_real545",
    "conv": "wildchat_prefix_real545",
    "icl": "icl_prefix_sycophancy",
}
CONTENT_DVS = ("dv_level", "dv_change", "dv_binary")
MARKER_DVS = ("dv_dlogp", "dv_level_logp", "dv_eos_margin", "dv_prob")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _meta() -> dict:
    return {
        "script": "scripts/issue1979_race.py",
        "issue": ISSUE,
        "seed": SEED,
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    R._atomic_json(path, obj)


def _cos_rows(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine of (n, d) A against (d,) b — one GEMM."""
    num = A @ b
    den = np.linalg.norm(A, axis=1) * (np.linalg.norm(b) + 1e-12) + 1e-12
    return num / den


def _read_repo_json(rel: str) -> dict:
    """Read a git-committed JSON (filesystem first; `git show origin/main:` in
    a sparse worktree where eval_results/ is excluded)."""
    p = REPO_ROOT / rel
    if p.exists():
        return json.loads(p.read_text())
    out = subprocess.run(
        ["git", "show", f"origin/main:{rel}"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert out.returncode == 0, f"cannot read {rel} (missing locally and on origin/main)"
    return json.loads(out.stdout)


# ── input staging (scoped; plan §10 off_pod_phases reads) ─────────────────────


def stage_inputs(inputs_root: Path, mixes: list[str], marker_arms: list[str]) -> None:
    """Stage the F1 outputs F3 consumes from the HF data repo (skip-if-present)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    repo = J._data_repo()
    singles = [
        "predictor_tables/predictor_ingredients.json",
        "battery/ingredient_tensors.pt",
        "battery/battery_reads.json",
        "battery/sigma_chol.pt",
    ] + [f"anchors/{m}/anchors.pt" for m in mixes]
    for rel in singles:
        dest = inputs_root / rel
        if not dest.exists():
            hub.stage_hub_file(repo, f"{HF_PREFIX_1979}/{rel}", dest, repo_type="dataset")
    # marker slot shards: scoped listing once, then per-file staging (#833)
    units = [*marker_arms, "base_mk"] if marker_arms else []
    missing = [u for u in units if not list((inputs_root / "marker_tf" / u).glob("slot_*.jsonl"))]
    if missing:
        listing = hub.retry_transient(
            lambda: hub.list_hf_files_under_path(
                HfApi(), repo, f"{HF_PREFIX_1979}/marker_tf", repo_type="dataset"
            ),
            what="marker_tf scoped listing",
        )
        for u in missing:
            wanted = [p for p in listing if f"/marker_tf/{u}/" in f"/{p}"]
            assert wanted, (u, "no marker_tf shards on the HF mirror")
            for p in wanted:
                hub.stage_hub_file(
                    repo, p, inputs_root / "marker_tf" / u / Path(p).name, repo_type="dataset"
                )


# ── config + manifests ────────────────────────────────────────────────────────


def load_config(config_dir: Path) -> dict:
    panel = json.loads((config_dir / "prefix_panel.json").read_text())
    queries = json.loads((config_dir / "queries.json").read_text())["queries"]
    arms = json.loads((config_dir / "arms.json").read_text())["arms"]
    members = panel["members"]
    prefix_ids = [m["prefix_id"] for m in members]
    return {
        "members": members,
        "prefix_ids": prefix_ids,
        "family": {m["prefix_id"]: m["family"] for m in members},
        "length": {m["prefix_id"]: m.get("content_token_len") for m in members},
        "queries": queries,
        "q_ix": {q["sha"]: i for i, q in enumerate(queries)},
        "arms": arms,
        "content_arms": [a for a in arms if a["kind"] == "content"],
        "marker_arms": [a for a in arms if a["kind"] == "marker"],
    }


def load_tensors(inputs_root: Path) -> dict:
    import torch

    t = torch.load(
        inputs_root / "battery/ingredient_tensors.pt", map_location="cpu", weights_only=False
    )
    return {k: np.asarray(v.float().numpy(), dtype=np.float64) for k, v in t.items()}


def load_anchors(inputs_root: Path, mix: str) -> dict:
    import torch

    return torch.load(
        inputs_root / "anchors" / mix / "anchors.pt", map_location="cpu", weights_only=False
    )


def _read_slot_rows(inputs_root: Path, unit: str, side: str) -> list[dict]:
    shards = sorted((inputs_root / "marker_tf" / unit).glob(f"slot_{side}.shard*.jsonl"))
    assert shards, f"no slot_{side} shards for {unit} under {inputs_root / 'marker_tf' / unit}"
    rows: list[dict] = []
    for p in shards:
        rows.extend(J._read_jsonl_rows(p))
    return rows


# ── DV assembly ───────────────────────────────────────────────────────────────


def _score_matrix(payload: dict, prefix_ids: list[str], q_ix: dict) -> np.ndarray:
    """(n_prefix, n_query) per-completion mean-of-kept-draws; NaN where absent."""
    P, Q = len(prefix_ids), len(q_ix)
    p_ix = {p: i for i, p in enumerate(prefix_ids)}
    S = np.full((P, Q), np.nan)
    for r in payload["rows"]:
        if r["score_mean"] is not None:
            S[p_ix[r["prefix_id"]], q_ix[r["query_sha"]]] = float(r["score_mean"])
    return S


def _binary_matrix(payload: dict, prefix_ids: list[str], q_ix: dict) -> np.ndarray:
    P, Q = len(prefix_ids), len(q_ix)
    p_ix = {p: i for i, p in enumerate(prefix_ids)}
    B = np.full((P, Q), np.nan)
    for r in payload["rows"]:
        if r["binary_rate"] is not None:
            B[p_ix[r["prefix_id"]], q_ix[r["query_sha"]]] = float(r["binary_rate"])
    return B


def assemble_content_dv(cfg: dict, judge_dir: Path, arm: dict) -> dict:
    """Per-prefix content DVs + (prefix x query) matrices (row-coverage rule)."""
    arm_pl = R.load_scores(judge_dir, arm["arm_id"])
    base_pl = R.load_scores(judge_dir, f"base_{arm['beh_key']}")
    q_ix = cfg["q_ix"]
    S_arm = _score_matrix(arm_pl, cfg["prefix_ids"], q_ix)
    S_base = _score_matrix(base_pl, cfg["prefix_ids"], q_ix)
    B_arm = _binary_matrix(arm_pl, cfg["prefix_ids"], q_ix)
    n_q = len(q_ix)
    floor = math.ceil(COVERAGE_FRAC * n_q)
    valid = (~np.isnan(S_arm)).sum(axis=1) >= floor
    valid &= (~np.isnan(S_base)).sum(axis=1) >= floor
    with np.errstate(invalid="ignore"):
        level = np.nanmean(S_arm, axis=1)
        p7 = np.nanmean(S_base, axis=1)
        binary = np.nanmean(B_arm, axis=1)
    ceil_share = np.nanmean(S_base >= CEILING_SCORE, axis=1)
    return {
        "valid": valid,
        "dv_level": level,
        "dv_change": level - p7,
        "dv_binary": binary,
        "p7": p7,
        "S_arm": S_arm,
        "S_base": S_base,
        "base_ceiling_share": ceil_share,
        "coverage_floor": int(floor),
        "n_queries": int(n_q),
    }


def assemble_marker_dv(cfg: dict, inputs_root: Path, arm: dict) -> dict:
    """Per-prefix marker three-space DVs (slot own − base on OWN text; base-slot P7)."""
    aid = arm["arm_id"]
    q_ix, prefix_ids = cfg["q_ix"], cfg["prefix_ids"]
    p_ix = {p: i for i, p in enumerate(prefix_ids)}
    P, Q = len(prefix_ids), len(q_ix)

    def mats(rows: list[dict]) -> dict[str, np.ndarray]:
        out = {k: np.full((P, Q), np.nan) for k in ("logp", "margin")}
        for r in rows:
            i, j = p_ix[r["prefix_id"]], q_ix[r["query_sha"]]
            out["logp"][i, j] = float(r["logp"])
            out["margin"][i, j] = float(r["z_marker"]) - float(r["z_eos"])
        return out

    own = mats(_read_slot_rows(inputs_root, aid, "own"))
    base = mats(_read_slot_rows(inputs_root, aid, "base"))
    bb = mats(_read_slot_rows(inputs_root, "base_mk", "base"))
    n_q = len(q_ix)
    floor = math.ceil(COVERAGE_FRAC * n_q)
    D = own["logp"] - base["logp"]
    valid = (~np.isnan(D)).sum(axis=1) >= floor
    valid &= (~np.isnan(bb["logp"])).sum(axis=1) >= floor
    with np.errstate(invalid="ignore"):
        dlogp = np.nanmean(D, axis=1)
        level = np.nanmean(own["logp"], axis=1)
        eos = np.nanmean(own["margin"] - base["margin"], axis=1)
        prob = np.nanmean(np.exp(own["logp"]) - np.exp(base["logp"]), axis=1)
        p7 = np.nanmean(bb["logp"], axis=1)  # base log P(marker) at the BASE slot
    return {
        "valid": valid,
        "dv_dlogp": dlogp,
        "dv_level_logp": level,
        "dv_eos_margin": eos,
        "dv_prob": prob,
        "p7": p7,
        "S_arm": D,  # qboot resamples the dlogp query columns
        "S_base": bb["logp"],
        "coverage_floor": int(floor),
        "n_queries": int(n_q),
    }


# ── candidate assembly (plan §5 roster) ───────────────────────────────────────


def _center(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = A.mean(axis=0)
    return A - mu, mu


def candidate_table(
    cfg: dict, tensors: dict, tables: dict, anchors: dict, arm: dict, layer: int, pos: str
) -> dict[str, np.ndarray | None]:
    """P1..P10 (+P8a/b) + M-panel per prefix at one (layer, pos). Convention:
    every cosine read centers BOTH vectors at the per-prefix family's panel
    mean (plan §6 stated shared centering; within-arm ranking unaffected)."""
    aid, kind = arm["arm_id"], arm["kind"]
    slot = f"{aid}/L{layer}/{pos}"
    Cbar, Vbar0 = tensors[f"{slot}/Cbar"], tensors[f"{slot}/Vbar0"]
    W_m, W_o = tensors[f"{slot}/W_matched"], tensors[f"{slot}/W_onpolicy"]
    Cbar_post = tensors[f"{slot}/Cbar_post"]
    anc = anchors[f"L{layer}"]
    A_ctx = np.asarray(anc[ANCHOR_KEY_BY_POS[pos]].double().numpy())
    A_ans = np.asarray(anc["A_ans"].double().numpy())
    rows_ctx = np.asarray(anc["rows_ctx"].float().numpy(), dtype=np.float64)
    rows_ans = np.asarray(anc["rows_ans"].float().numpy(), dtype=np.float64)
    Cc, c_mu = _center(Cbar)
    Vc, v_mu = _center(Vbar0)
    own_pid = OWN_PREFIX_BY_CTX[arm["ctx_key"]]
    own_ix = cfg["prefix_ids"].index(own_pid) if own_pid in cfg["prefix_ids"] else None
    out: dict[str, np.ndarray | None] = {}
    out["p1_tc"] = _cos_rows(Cc, A_ctx - c_mu)
    out["p2_tc"] = _cos_rows(Vc, A_ans - v_mu)
    out["p1_ps"] = _cos_rows(Cc, Cbar[own_ix] - c_mu) if own_ix is not None else None
    out["p2_ps"] = _cos_rows(Vc, Vbar0[own_ix] - v_mu) if own_ix is not None else None
    # through-map (P3a/P3b/P6): M0 fits exist at span_mean/last_prompt only
    mpos = {"span_mean_context": "span_mean", "last_prompt": "last_prompt"}.get(pos)
    if mpos is not None:
        M0C = tensors[f"m0pred/{kind}/L{layer}/{mpos}"]
        M0A = tensors[f"m0anchor/{arm['mix_arm_id']}/L{layer}/{mpos}"]
        Mc, m_mu = _center(M0C)
        out["p3a_tc"] = _cos_rows(Mc, M0A - m_mu)
        out["p3b_tc"] = _cos_rows(Mc, A_ans - m_mu)
    else:
        out["p3a_tc"] = out["p3b_tc"] = None
    out["p4_tc"] = np.asarray(tables[aid][f"L{layer}/{pos}"]["p4_gpred"], dtype=np.float64)
    beh = arm["beh_key"]
    rb = RB_CACHE[beh]
    assert rb.shape[0] > layer, (beh, rb.shape, layer)
    r_l = rb[layer]
    out["p5"] = Vc @ r_l
    out["p6"] = (Mc @ r_l) if mpos is not None else None
    # P9/P10: nearest-training-rows (span-position anchor rows — stated)
    span_slot = f"{aid}/L{layer}/span_mean_context"
    C_span = tensors[f"{span_slot}/Cbar"]
    V_span = tensors[f"{span_slot}/Vbar0"]

    def topk_mean(A: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        Rn = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)
        cos = An @ Rn.T  # (n_prefix, n_rows)
        kk = min(k, cos.shape[1])
        part = np.sort(cos, axis=1)[:, -kk:]
        return part.mean(axis=1)

    out["p9_k8"] = topk_mean(C_span, rows_ctx, K_NN)
    out["p10_k8"] = topk_mean(V_span, rows_ans, K_NN)
    for k_ in (4, 16):  # dump sensitivity
        out[f"p9_k{k_}"] = topk_mean(C_span, rows_ctx, k_)
        out[f"p10_k{k_}"] = topk_mean(V_span, rows_ans, k_)
    # P8 (content only; #1900 wmaps, span-mean L19 — stated position note)
    p8_key = f"{aid}/p8_wmap_pred_L19"
    if p8_key in tensors:
        pred = tensors[p8_key]
        out["p8a"] = np.linalg.norm(pred, axis=1)
        out["p8b"] = _cos_rows(pred, rb[19])
    else:
        out["p8a"] = out["p8b"] = None
    # anchor half variants (disjoint even/odd anchor rows; span anchors — stated)
    for tag, hk_c, hk_a in (
        ("even", "half_even_ctx", "half_even_ans"),
        ("odd", "half_odd_ctx", "half_odd_ans"),
    ):
        hc = np.asarray(anc[hk_c].double().numpy())
        ha = np.asarray(anc[hk_a].double().numpy())
        Csc, sc_mu = _center(C_span)
        Vsc, sv_mu = _center(V_span)
        out[f"p1_tc_{tag}"] = _cos_rows(Csc, hc - sc_mu)
        out[f"p2_tc_{tag}"] = _cos_rows(Vsc, ha - sv_mu)
    # M-panel (mechanistic; FT arms: M3/M4/M6 only — plan §5 scope bound)
    Pc, p_mu = _center(Cbar_post)
    Vpost = Vbar0 + W_o
    Vpc, vp_mu = _center(Vpost)
    is_ft = arm["method"] == "ft"
    out["m1_tc"] = None if is_ft else _cos_rows(Pc, A_ctx - p_mu)
    out["m2_tc"] = None if is_ft else _cos_rows(Vpc, A_ans - vp_mu)
    out["m3"] = _cos_rows(Cbar_post - Cbar, A_ctx - c_mu)
    out["m4"] = _cos_rows(W_m, A_ans - v_mu)
    out["m4_onpolicy"] = _cos_rows(W_o, A_ans - v_mu)
    m5_key = f"{aid}/m0pred_Cbar_post/L{layer}/{mpos}" if mpos is not None else None
    if not is_ft and m5_key is not None and m5_key in tensors:
        M5C = tensors[m5_key]
        M5c, m5_mu = _center(M5C)
        out["m5_tc"] = _cos_rows(M5c, A_ans - m5_mu)
    else:
        out["m5_tc"] = None
    out["m6"] = np.linalg.norm(W_m, axis=1)
    return out


RB_CACHE: dict[str, np.ndarray] = {}


def load_rb(inputs_root: Path) -> None:
    """Fill RB_CACHE via the recipe-compliant loader (Hub-staged on miss)."""
    import issue1768_directions as DIR

    RB_CACHE.update(DIR.load_rb_tensors(inputs_root, rb_dir=inputs_root / "rb"))


# ── raced roster + frame ──────────────────────────────────────────────────────

RACED_CONTENT = ("p1", "p2", "p3a", "p3b", "p4", "p5", "p6", "p7", "p8a", "p8b", "p9", "p10")
RACED_MARKER = ("p1", "p2", "p3a", "p3b", "p4", "p5", "p6", "p7", "p9", "p10")
CAND_COL = {
    "p1": "p1_tc",
    "p2": "p2_tc",
    "p3a": "p3a_tc",
    "p3b": "p3b_tc",
    "p4": "p4_tc",
    "p5": "p5",
    "p6": "p6",
    "p7": "p7",
    "p8a": "p8a",
    "p8b": "p8b",
    "p9": "p9_k8",
    "p10": "p10_k8",
}
M_COLS = ("m1_tc", "m2_tc", "m3", "m4", "m4_onpolicy", "m5_tc", "m6")


def assemble_arm(
    cfg: dict,
    tensors: dict,
    tables: dict,
    inputs_root: Path,
    judge_dir: Path,
    arm: dict,
    min_prefixes: int,
) -> dict:
    """Listwise per-prefix frame for one arm at its primary coordinates."""
    kind = arm["kind"]
    layer, pos = PRIMARY[kind]
    anchors = load_anchors(inputs_root, arm["mix_arm_id"])
    dv = (
        assemble_content_dv(cfg, judge_dir, arm)
        if kind == "content"
        else assemble_marker_dv(cfg, inputs_root, arm)
    )
    cands = candidate_table(cfg, tensors, tables, anchors, arm, layer, pos)
    dv_names = list(CONTENT_DVS if kind == "content" else MARKER_DVS)
    raced = [
        c
        for c in (RACED_CONTENT if kind == "content" else RACED_MARKER)
        if c == "p7" or cands.get(CAND_COL[c]) is not None
    ]
    n = len(cfg["prefix_ids"])
    keep = np.asarray(dv["valid"], dtype=bool).copy()
    for c in raced:
        col = dv["p7"] if c == "p7" else cands[CAND_COL[c]]
        keep &= np.isfinite(np.asarray(col, dtype=np.float64))
    for d in dv_names:
        keep &= np.isfinite(np.asarray(dv[d], dtype=np.float64))
    kept_ix = np.flatnonzero(keep)
    assert len(kept_ix) >= min_prefixes, (
        arm["arm_id"],
        int(len(kept_ix)),
        f"below realized-prefix floor {min_prefixes}",
    )
    frame: dict = {
        "prefix_id": [cfg["prefix_ids"][i] for i in kept_ix],
        "family": [cfg["family"][cfg["prefix_ids"][i]] for i in kept_ix],
        "content_token_len": [cfg["length"][cfg["prefix_ids"][i]] for i in kept_ix],
        "p7": np.asarray(dv["p7"])[kept_ix],
    }
    for d in dv_names:
        frame[d] = np.asarray(dv[d])[kept_ix]
    for name, col in cands.items():
        if col is not None:
            frame[name] = np.asarray(col, dtype=np.float64)[kept_ix]
    if kind == "content":
        frame["base_ceiling_share"] = np.asarray(dv["base_ceiling_share"])[kept_ix]
    return {
        "arm": arm,
        "layer": layer,
        "pos": pos,
        "frame": frame,
        "raced": raced,
        "dv_names": dv_names,
        "kept_ix": kept_ix,
        "n_realized": int(len(kept_ix)),
        "S_arm": dv["S_arm"],
        "S_base": dv["S_base"],
        "coverage_floor": dv["coverage_floor"],
        "n_queries": dv["n_queries"],
    }


def _x_matrix(asm: dict) -> np.ndarray:
    f = asm["frame"]
    return np.column_stack(
        [np.asarray(f["p7" if c == "p7" else CAND_COL[c]], dtype=np.float64) for c in asm["raced"]]
    )


def _dv_matrix(asm: dict) -> np.ndarray:
    return np.column_stack([np.asarray(asm["frame"][d], dtype=np.float64) for d in asm["dv_names"]])


# ── bootstrap families (shared draw streams; winner re-selected per draw) ─────


def qcluster_boot(asm: dict, b_draws: int, seed: int) -> np.ndarray:
    """(B, K, D) query-cluster bootstrap: resample the query columns, recompute
    the per-prefix DV means AND P7 per draw; geometry candidates fixed."""
    import torch

    rng = np.random.default_rng(seed + 7)
    kept = asm["kept_ix"]
    S_arm = np.asarray(asm["S_arm"])[kept]  # (n, Q)
    S_base = np.asarray(asm["S_base"])[kept]
    n, Q = S_arm.shape
    raced, dv_names = asm["raced"], asm["dv_names"]
    x_fixed = _x_matrix(asm)  # (n, K) — p7 column replaced per draw
    p7_col = raced.index("p7")
    is_content = asm["arm"]["kind"] == "content"
    out = np.empty((b_draws, len(raced), len(dv_names)), dtype=np.float32)
    chunk = 250
    for b0 in range(0, b_draws, chunk):
        nb = min(chunk, b_draws - b0)
        qidx = rng.integers(0, Q, size=(nb, Q))
        arm_d = np.nanmean(S_arm[:, qidx], axis=2).T.reshape(nb, n)  # (nb, n)
        base_d = np.nanmean(S_base[:, qidx], axis=2).T.reshape(nb, n)
        assert np.isfinite(arm_d).all() and np.isfinite(base_d).all(), (
            "query-cluster draw produced an all-NaN prefix cell (coverage floor breached)"
        )
        # per-draw DVs
        dvs = np.empty((nb, len(dv_names), n))
        for j, d in enumerate(dv_names):
            if is_content and d == "dv_level":
                dvs[:, j] = arm_d
            elif is_content and d == "dv_change":
                dvs[:, j] = arm_d - base_d
            elif (not is_content) and d == "dv_dlogp":
                dvs[:, j] = arm_d
            else:  # DVs without a per-query matrix stay at their observed values
                dvs[:, j] = np.asarray(asm["frame"][d], dtype=np.float64)[None, :]
        xs = np.repeat(x_fixed.T[None], nb, axis=0)  # (nb, K, n)
        xs[:, p7_col, :] = base_d
        stacked = torch.from_numpy(
            np.concatenate([xs, dvs], axis=1).astype(np.float32)
        )  # (nb, K+D, n)
        z, _ = R._rank_z_t(stacked)
        zc, zd = z[:, : len(raced)], z[:, len(raced) :]
        rho = torch.einsum("bkn,bdn->bkd", zc, zd) / n
        out[b0 : b0 + nb] = rho.numpy()
    return out


def family_boot(asm: dict, b_draws: int, seed: int) -> np.ndarray:
    """(B, K, D) family-cluster bootstrap (sensitivity): resample the prefix
    FAMILIES with replacement; group-level n ~ 8 framed accordingly."""
    fams = sorted(set(asm["frame"]["family"]))
    fam_ix = {f: np.flatnonzero(np.asarray(asm["frame"]["family"]) == f) for f in fams}
    rng = np.random.default_rng(seed + 13)
    x, dvs = _x_matrix(asm), _dv_matrix(asm)
    out = np.empty((b_draws, x.shape[1], dvs.shape[1]), dtype=np.float32)
    for b in range(b_draws):
        pick = rng.integers(0, len(fams), size=len(fams))
        idx = np.concatenate([fam_ix[fams[i]] for i in pick])
        out[b] = R.observed_rho(x[idx], dvs[idx])
    return out


# ── per-arm battery (persist-per-arm; resume-skip on regime match) ────────────


def _shared_pool(asms: list[dict]) -> tuple[list[str], str]:
    """Sorted realized-prefix intersection across a kind's arms + digest —
    the champion pairing pool (ONE seed + ONE pool => paired draw streams)."""
    shared = sorted(set.intersection(*[set(a["frame"]["prefix_id"]) for a in asms]))
    assert shared, "empty shared prefix pool across arms"
    return shared, hashlib.sha256("\n".join(shared).encode()).hexdigest()[:16]


def run_arm_battery(
    asm: dict, out_dir: Path, shared: tuple[list[str], str], b_draws: int, n_perm: int
) -> dict:
    aid = asm["arm"]["arm_id"]
    arm_json = out_dir / f"arm_{aid}.json"
    regime = {
        "b_draws": b_draws,
        "n_perm": n_perm,
        "layer": asm["layer"],
        "pos": asm["pos"],
        "raced": asm["raced"],
        "dv_names": asm["dv_names"],
        "n": asm["n_realized"],
        "n_shared": len(shared[0]),
        "shared_hash": shared[1],
        "coverage_floor": asm["coverage_floor"],
    }
    if arm_json.exists():
        prior = json.loads(arm_json.read_text())
        if prior.get("regime") == regime and (out_dir / f"boot_{aid}.npz").exists():
            logger.info("[f3] %s resume-skip", aid)
            return prior
    t0 = time.time()
    x, dvs = _x_matrix(asm), _dv_matrix(asm)
    pid_pos = {p: i for i, p in enumerate(asm["frame"]["prefix_id"])}
    pos_ix = np.asarray([pid_pos[p] for p in shared[0]], dtype=np.int64)
    boot, n_degen = R.bootstrap_battery(x[pos_ix], dvs[pos_ix], b_draws, SEED)
    qboot = qcluster_boot(asm, b_draws, SEED)
    fboot = family_boot(asm, b_draws, SEED)
    perm = R.perm_null(x, dvs[:, 0], n_perm, R._arm_seed(aid))
    obs = R.observed_rho(x, dvs)
    perm_max = perm.max(axis=1)  # SIGNED per-draw max — selection rides the draw
    raced = asm["raced"]
    for name, arr in (("boot", boot), ("qboot", qboot), ("fboot", fboot)):
        np.savez(
            out_dir / f"{name}_{aid}.npz",
            rho=arr,
            candidates=np.array(raced),
            dv_names=np.array(asm["dv_names"]),
            seed=SEED,
            n=asm["n_realized"],
            n_shared=len(shared[0]),
            shared_sha_hash=np.array(shared[1]),
        )
    np.savez(
        out_dir / f"perm_{aid}.npz",
        rho=perm,
        max_selected=perm_max,
        candidates=np.array(raced),
        dv="primary",
        seed=R._arm_seed(aid) + 1,
    )
    # split-half reliability (rule 21: ONE even/odd query partition)
    kept = asm["kept_ix"]
    S = np.asarray(asm["S_arm"])[kept]
    with np.errstate(invalid="ignore"):
        dv_e = np.nanmean(S[:, 0::2], axis=1)
        dv_o = np.nanmean(S[:, 1::2], axis=1)
    r_half = R._spearman_np(dv_e, dv_o)
    rel = 2 * r_half / (1 + r_half) if (1 + r_half) > 1e-9 else None
    m_obs = {}
    for col in M_COLS:
        if col in asm["frame"]:
            m_obs[col] = R._spearman_np(np.asarray(asm["frame"][col], dtype=np.float64), dvs[:, 0])
    payload = {
        "meta": _meta(),
        "arm_id": aid,
        "kind": asm["arm"]["kind"],
        "beh_key": asm["arm"]["beh_key"],
        "regime": regime,
        "observed_rho": {
            d: {c: float(obs[i, j]) for i, c in enumerate(raced)}
            for j, d in enumerate(asm["dv_names"])
        },
        "m_panel_rho_primary": m_obs,
        "perm_band": {
            "p975_max_selected": float(np.quantile(perm_max, 0.975)),
            "p95_max_selected": float(np.quantile(perm_max, 0.95)),
            "ceiling_abs_rho": 1.0,
            "n_perm": n_perm,
        },
        "n_degenerate_series_draws": int(n_degen),
        "dv_split_half": {
            "r": r_half,
            "sb_rel": rel,
            "sqrt_rel": (math.sqrt(rel) if rel is not None and rel > 0 else None),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    _atomic_json(arm_json, payload)
    print(f"[f3] arm {aid} battery elapsed={time.time() - t0:.1f}s", flush=True)
    return payload


# ── champion (across-arm, selection-symmetric; parameterized incumbent) ───────


def champion(
    arm_ids: list[str],
    out_dir: Path,
    boot_stem: str,
    dv_index: int,
    dv_label: str,
    incumbent: str,
    verdict_names: tuple[str, str],
) -> dict:
    """Across-arm-median winner with per-draw re-selection (plan §3 lattices;
    modeled on issue1900_race.champion_read, incumbent parameterized so the H2
    change race can pit candidates against P2)."""
    boots, raced_sets, seeds, hashes = {}, [], set(), set()
    for a in arm_ids:
        z = np.load(out_dir / f"{boot_stem}_{a}.npz", allow_pickle=False)
        boots[a] = (z["rho"], list(z["candidates"]))
        raced_sets.append(set(z["candidates"]))
        seeds.add(int(z["seed"]))
        hashes.add(str(z["shared_sha_hash"]))
    assert len(seeds) == 1 and (boot_stem != "boot" or len(hashes) == 1), (
        seeds,
        hashes,
        "champion pairing broken: arms carry different draw streams",
    )
    panel = sorted(set.intersection(*raced_sets))
    cube = np.stack(
        [boots[a][0][:, [boots[a][1].index(c) for c in panel], dv_index] for a in arm_ids]
    )  # (A, B, Kp)
    med = np.median(cube, axis=0)
    winner_ix = np.argmax(med, axis=1)  # SIGNED argmax (registered convention)
    p_win = {c: float(np.mean(winner_ix == i)) for i, c in enumerate(panel)}
    obs_med, per_arm_obs = {}, {}
    for a in arm_ids:
        pl = json.loads((out_dir / f"arm_{a}.json").read_text())
        dv_name = pl["regime"]["dv_names"][dv_index]
        per_arm_obs[a] = pl["observed_rho"][dv_name]
    for c in panel:
        obs_med[c] = float(np.median([per_arm_obs[a][c] for a in arm_ids]))
    winner = max(obs_med, key=lambda c: obs_med[c])
    sel_ci = [float(np.quantile(med.max(axis=1), q)) for q in CI_QS]
    frz_ci = [float(np.quantile(med[:, panel.index(winner)], q)) for q in CI_QS]
    dethrone_min = math.ceil(0.75 * len(arm_ids))
    retain_name, dethrone_name = verdict_names
    verdict = "no-resolved-champion"
    if winner == incumbent and p_win.get(incumbent, 0.0) >= 0.5:
        verdict = retain_name
    elif winner != incumbent and p_win.get(winner, 0.0) >= 0.5:
        n_beats = sum(
            int(per_arm_obs[a].get(winner, -9) - per_arm_obs[a].get(incumbent, 9) > 0)
            for a in arm_ids
        )
        if n_beats >= dethrone_min:
            verdict = f"{dethrone_name} ({winner}; beats {incumbent} in {n_beats}/{len(arm_ids)})"
    inc_obs = [per_arm_obs[a].get(incumbent) for a in arm_ids if incumbent in per_arm_obs[a]]
    return {
        "dv": dv_label,
        "incumbent": incumbent,
        "bootstrap_family": boot_stem,
        "panel_candidates": panel,
        "arm_ids": arm_ids,
        "across_arm_median_observed": obs_med,
        "winner_observed": winner,
        "p_win": p_win,
        "selection_inherited_ci_max_median": sel_ci,
        "frozen_ci_winner_median (labeled: frozen-at-winner)": frz_ci,
        "verdict": verdict,
        "dethrone_min_arms": dethrone_min,
        "champion_vs_incumbent_conditional_ceiling_interval": (
            [float(1.0 - max(inc_obs)), float(1.0 - min(inc_obs))] if inc_obs else None
        ),
        "note_correlated_arms": "arms share one prefix panel + one judge instrument — "
        "never narrated as independent confirmations",
    }


# ── H5 signed residuals (con-scoped per plan §3) ──────────────────────────────


def h5_residuals(asms: list[dict]) -> dict:
    """Per-arm signed residuals of the trained-NEGATIVE prefixes vs the
    within-arm geometry fit (DV ~ [1, P1, P2] least squares, FIT ON the
    non-negative prefixes, PREDICT the negatives; full-fit sensitivity)."""
    per_arm: dict = {}
    for asm in asms:
        f = asm["frame"]
        fam = np.asarray(f["family"])
        is_neg = fam == "negatives"
        if is_neg.sum() == 0:
            per_arm[asm["arm"]["arm_id"]] = {"note": "no negative prefixes realized"}
            continue
        dv = np.asarray(f[asm["dv_names"][0]], dtype=np.float64)
        X = np.column_stack([np.ones(len(dv)), np.asarray(f["p1_tc"]), np.asarray(f["p2_tc"])])
        beta, *_ = np.linalg.lstsq(X[~is_neg], dv[~is_neg], rcond=None)
        resid_neg = dv[is_neg] - X[is_neg] @ beta
        beta_full, *_ = np.linalg.lstsq(X, dv, rcond=None)
        resid_full = (dv - X @ beta_full)[is_neg]
        # change-DV re-read (registered analyzer line 6)
        chg_key = "dv_change" if "dv_change" in f else asm["dv_names"][0]
        dvc = np.asarray(f[chg_key], dtype=np.float64)
        beta_c, *_ = np.linalg.lstsq(X[~is_neg], dvc[~is_neg], rcond=None)
        resid_neg_chg = dvc[is_neg] - X[is_neg] @ beta_c
        a = asm["arm"]
        group = (
            "con"
            if (a["regime"] == "con" and a["ctx_key"] != "bare")
            else ("po" if a["regime"] == "po" else "bare-placebo")
        )
        per_arm[a["arm_id"]] = {
            "kind": a["kind"],
            "beh_key": a["beh_key"],
            "regime": a["regime"],
            "ctx_key": a["ctx_key"],
            "h5_group": group,
            "n_negatives": int(is_neg.sum()),
            "negative_prefixes": [p for p, m in zip(f["prefix_id"], is_neg) if m],
            "median_signed_residual": float(np.median(resid_neg)),
            "residuals": [float(v) for v in resid_neg],
            "median_signed_residual_fullfit": float(np.median(resid_full)),
            "median_signed_residual_change_dv": float(np.median(resid_neg_chg)),
        }
    groups: dict[str, list[float]] = {}
    for v in per_arm.values():
        if "median_signed_residual" in v:
            groups.setdefault(f"{v['kind']}:{v['h5_group']}", []).append(
                v["median_signed_residual"]
            )

    def frac_neg(key: str) -> float | None:
        vals = groups.get(key)
        return float(np.mean([m < 0 for m in vals])) if vals else None

    con_c, con_m = frac_neg("content:con"), frac_neg("marker:con")
    crit = con_c is not None and con_c >= 2 / 3 and con_m is not None and con_m >= 2 / 3
    # con - po within-behavior residual difference (the suppression signature)
    by_beh: dict = {}
    for v in per_arm.values():
        if "median_signed_residual" in v:
            by_beh.setdefault(v["beh_key"], {}).setdefault(v["h5_group"], []).append(
                v["median_signed_residual"]
            )
    con_minus_po = {
        beh: float(np.median(g["con"]) - np.median(g["po"]))
        for beh, g in by_beh.items()
        if "con" in g and "po" in g
    }
    return {
        "meta": _meta(),
        "grouping_rule": "con = regime==con AND ctx_key!=bare; placebo = regime==po OR "
        "ctx_key==bare (plan §3 H5: negatives trained DOWN only in con arms; per-arm "
        "values persisted so the analyzer can re-derive under any grouping)",
        "per_arm": per_arm,
        "frac_con_content_median_lt0": con_c,
        "frac_con_marker_median_lt0": con_m,
        "criterion_pass_2of3_both": bool(crit),
        "con_minus_po_median_by_behavior": con_minus_po,
        "verdict": (
            "H5-suppression-signature"
            if crit and any(v < 0 for v in con_minus_po.values())
            else "H5-not-supported"
        ),
    }


# ── A5 / A6 / A7 assumption battery (plan §3 H4) ─────────────────────────────


def battery_verdicts(cfg: dict, tensors: dict, inputs_root: Path, asms: list[dict]) -> dict:
    """A7 gate lattice + A6 rank read + A5 disjoint-half alignment (L19 span)."""
    import torch

    import issue1768_directions as DIR

    reads = json.loads((inputs_root / "battery/battery_reads.json").read_text())
    sig = torch.load(inputs_root / "battery/sigma_chol.pt", map_location="cpu", weights_only=False)
    chol19 = np.asarray(sig["L19"]["chol"].float().numpy(), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    draws_iso = rng.standard_normal((2000, chol19.shape[0]))
    draws_cov = draws_iso @ chol19.T
    content_ids = {a["arm"]["arm_id"] for a in asms if a["arm"]["kind"] == "content"}
    # A7: median over content (arm x layer) gate cells vs the [0.3, 0.7] band
    gate_cells = []
    per_cell = {}
    for aid, layers in reads["arms"].items():
        for lk, slot in layers.items():
            if "gate_read" in slot and aid in content_ids:
                rho = slot["gate_read"]["spearman_rho"]
                per_cell[f"{aid}/{lk}"] = rho
                gate_cells.append(rho)
    a7 = {"cells": per_cell, "n_cells": len(gate_cells)}
    if gate_cells:
        med = float(np.median(gate_cells))
        in_band = float(np.mean([(0.3 <= r <= 0.7) for r in gate_cells]))
        a7.update(
            median_rho=med,
            share_in_band=in_band,
            verdict=(
                "gate-restored-at-prefix-grain"
                if 0.3 <= med <= 0.7 and in_band >= 0.5
                else "gate-still-refuted"
                if med < 0.3
                else "gate-overshoots"
                if med > 0.7
                else "gate-unresolved"
            ),
            per_query_anchor=0.14,
        )
    # A6 + A5 per arm
    a6, a5 = {}, {}
    for asm in asms:
        aid = asm["arm"]["arm_id"]
        layer = asm["layer"]
        a6[aid] = {}
        for tree, key in (("matched", "W_matched"), ("onpolicy", "W_onpolicy")):
            for li in LAYERS_1979:
                W = tensors[f"{aid}/L{li}/span_mean_context/{key}"]
                rr = DIR.rank_read(W, W.mean(axis=0))
                a6[aid][f"{tree}/L{li}"] = rr
        # A5 at L19 span: disjoint even/odd base halves (plan §6 noise-structure)
        slot = f"{aid}/L19/span_mean_context"
        kind = asm["arm"]["kind"]
        Vbar0 = tensors[f"{slot}/Vbar0"]
        Ve = tensors[f"base/{kind}/L19/Vbar0_even"]
        Vo = tensors[f"base/{kind}/L19/Vbar0_odd"]
        leg_w = (tensors[f"{slot}/W_matched"] + Vbar0) - Ve  # matched_all − v̄0_even
        leg_d = (tensors[f"{slot}/W_onpolicy"] + Vbar0) - Vo  # onpol_all − v̄0_odd
        per_prefix_cos = np.sum(leg_w * leg_d, axis=1) / (
            np.linalg.norm(leg_w, axis=1) * np.linalg.norm(leg_d, axis=1) + 1e-12
        )
        pooled_w, pooled_d = leg_w.mean(axis=0), leg_d.mean(axis=0)
        shared_w = tensors[f"{slot}/W_matched"].mean(axis=0)  # shared-B̄ record-only
        shared_d = tensors[f"{slot}/W_onpolicy"].mean(axis=0)
        rb = RB_CACHE[asm["arm"]["beh_key"]][19]

        def band(target: np.ndarray) -> dict:
            """Norm-matched null cosine quantiles vs `target` (iso + corpus-cov)."""
            tn = target / (np.linalg.norm(target) + 1e-12)
            out = {}
            for name, draws in (("isotropic", draws_iso), ("corpus_cov", draws_cov)):
                dn = draws / (np.linalg.norm(draws, axis=1, keepdims=True) + 1e-12)
                cos = dn @ tn
                out[name] = {
                    "p2_5": float(np.quantile(cos, 0.025)),
                    "p97_5": float(np.quantile(cos, 0.975)),
                    "abs_p95": float(np.quantile(np.abs(cos), 0.95)),
                }
            return out

        pooled_cos = float(
            pooled_w @ pooled_d / (np.linalg.norm(pooled_w) * np.linalg.norm(pooled_d) + 1e-12)
        )
        nulls = band(pooled_d)
        a5[aid] = {
            "pooled_cos_disjoint": pooled_cos,
            "median_per_prefix_cos_disjoint": float(np.median(per_prefix_cos)),
            "per_prefix_cos_disjoint": [float(v) for v in per_prefix_cos],
            "pooled_cos_sharedB_record_only": float(
                shared_w @ shared_d / (np.linalg.norm(shared_w) * np.linalg.norm(shared_d) + 1e-12)
            ),
            "null_bands_vs_pooled_delta": nulls,
            "alignment_clears_null": bool(abs(pooled_cos) > nulls["corpus_cov"]["abs_p95"]),
            "readout_cos_pooled_w_rb": float(
                pooled_w @ rb / (np.linalg.norm(pooled_w) * np.linalg.norm(rb) + 1e-12)
            ),
            "convention": "leg_w = matched_all − v̄0_even; leg_d = onpol_all − v̄0_odd "
            "(disjoint query halves, plan §6); norm-matched null bands iso + corpus-cov",
        }
    a6_prim = {
        aid: v[f"matched/L{PRIMARY['content'][0] if aid in content_ids else PRIMARY['marker'][0]}"][
            "top1_var_share"
        ]
        for aid, v in a6.items()
    }
    return {
        "meta": _meta(),
        "A7": a7,
        "A6": {
            "per_arm_tree_layer": a6,
            "primary_top1_share": a6_prim,
            "criterion": 0.6,
            "share_ge_criterion": float(np.mean([v >= 0.6 for v in a6_prim.values()]))
            if a6_prim
            else None,
            "per_query_reference": {
                "matched": 0.09,
                "onpolicy": 0.29,
                "source": "plan §3 cited per-query values",
            },
            "rank_ceiling_note": "50-prefix matrix => rank <= 50 by construction; top-1 SHARE "
            "is scale-free (registered analyzer line 8)",
        },
        "A5": a5,
    }


# ── A5 weights-vs-text decomposition (amendment marker-a5-weights-vs-text) ────

A5_DECOMP_PARITY_TOL = 0.005  # bug gate: re-derived pooled leg_d cos vs the persisted value
A5_DECOMP_MEANS_COS_MIN = 0.999  # bug gate: per-(arm, prefix) re-derived on-policy means parity
A5_DECOMP_IDENTITY_TOL = 1e-3  # bug gate: max |weights + text − leg_d| (fp64 from fp16 inputs)
A5_CARRY_FACTOR = 0.5  # clause (iii) "comparable magnitude" factor (plan v6 §3, registered)
A5_CARRY_FACTORS_RECORD = (0.25, 0.5, 0.75)  # 0.25/0.75 = record-only robustness (plan v6 §4)


def a5_decomposition(
    cfg: dict, tensors: dict, inputs_root: Path, decomp_in: dict, parity_ref: dict
) -> dict:
    """Weights-vs-text decomposition of the parent A5 delta leg per marker arm
    (plan v6 §4 Diff 3; L19 / span_mean_context — the parent A5 convention).

    PRIMARY odd-pivot (registered): ``weights = (W_onpolicy + Vbar0) − Hbar_odd``,
    ``text = Hbar_odd − Vbar0_odd`` (H̄ cancels ⇒ weights + text = leg_d exactly);
    all-pivot recorded as demoted sensitivity (no null bands). Parity gates
    (±0.005 pooled leg_d cos vs the persisted ``battery_verdicts.json``;
    ≥0.999 re-derived-means cos) are BUG GATES — a failure halts, never
    re-scores. Null bands per component: 2,000 norm-matched draws, isotropic +
    corpus-cov (``sigma_chol.pt`` L19), the parent ``band()`` convention,
    ``np.random.default_rng(SEED)``.
    """
    import torch

    marker_rows = {a["arm_id"]: a for a in cfg["marker_arms"]}
    marker_ids = list(marker_rows)
    assert marker_ids, "a5_decomposition: no marker arms selected"
    dprefix = list(decomp_in["meta"]["prefix_ids"])
    assert dprefix == cfg["prefix_ids"], (
        "basetf_decomp_inputs prefix order != config prefix order",
        dprefix[:3],
        cfg["prefix_ids"][:3],
    )
    sig = torch.load(inputs_root / "battery/sigma_chol.pt", map_location="cpu", weights_only=False)
    chol19 = np.asarray(sig["L19"]["chol"].float().numpy(), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    draws_iso = rng.standard_normal((2000, chol19.shape[0]))
    draws_cov = draws_iso @ chol19.T

    def band(target: np.ndarray) -> dict:
        """Norm-matched null cosine quantiles vs `target` — the parent A5
        ``battery_verdicts.band`` convention (iso + corpus-cov)."""
        tn = target / (np.linalg.norm(target) + 1e-12)
        out = {}
        for name, draws in (("isotropic", draws_iso), ("corpus_cov", draws_cov)):
            dn = draws / (np.linalg.norm(draws, axis=1, keepdims=True) + 1e-12)
            cos = dn @ tn
            out[name] = {
                "p2_5": float(np.quantile(cos, 0.025)),
                "p97_5": float(np.quantile(cos, 0.975)),
                "abs_p95": float(np.quantile(np.abs(cos), 0.95)),
            }
        return out

    def dec(aid: str, layer: int, name: str) -> np.ndarray:
        key = f"{aid}/L{layer}/{name}"
        assert key in decomp_in, f"basetf_decomp_inputs.pt missing tensor {key}"
        return np.asarray(decomp_in[key].float().numpy(), dtype=np.float64)

    def pcos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    arms_out: dict = {}
    for aid in marker_ids:
        slot = f"{aid}/L19/span_mean_context"
        Vbar0 = tensors[f"{slot}/Vbar0"]
        Ve = tensors["base/marker/L19/Vbar0_even"]
        Vo = tensors["base/marker/L19/Vbar0_odd"]
        W_o = tensors[f"{slot}/W_onpolicy"]
        leg_w = (tensors[f"{slot}/W_matched"] + Vbar0) - Ve  # matched_all − v̄0_even
        leg_d = (W_o + Vbar0) - Vo  # onpol_all − v̄0_odd (the parent delta leg)
        O_all = W_o + Vbar0  # observed on-policy all-query mean
        pooled_w = leg_w.mean(axis=0)
        cos_wd = pcos(pooled_w, leg_d.mean(axis=0))
        rb = RB_CACHE[marker_rows[aid]["beh_key"]][19]

        # parity gate (plan §6 criterion 3): reproduce the persisted parent read
        ref = float(parity_ref["A5"][aid]["pooled_cos_disjoint"])
        O_re = dec(aid, 19, "Obar_all")
        mp = np.sum((O_re - Vbar0) * W_o, axis=1) / (
            np.linalg.norm(O_re - Vbar0, axis=1) * np.linalg.norm(W_o, axis=1) + 1e-12
        )
        print(
            f"[a5-decomp] {aid}: leg_d pooled cos re-derived {cos_wd:.4f} vs persisted "
            f"{ref:.4f} (tol ±{A5_DECOMP_PARITY_TOL}); means-parity min cos "
            f"{float(mp.min()):.6f} (floor {A5_DECOMP_MEANS_COS_MIN})",
            flush=True,
        )
        assert abs(cos_wd - ref) <= A5_DECOMP_PARITY_TOL, (
            aid,
            f"leg_d parity gate: re-derived {cos_wd:.4f} vs persisted {ref:.4f} — "
            "keying/staging bug; NO verdict published (plan §6 kill criterion)",
        )
        assert float(mp.min()) >= A5_DECOMP_MEANS_COS_MIN, (
            aid,
            f"re-derived on-policy means parity: min cos {float(mp.min()):.6f} < "
            f"{A5_DECOMP_MEANS_COS_MIN} — aggregation-convention drift",
        )

        def decompose(H_odd: np.ndarray, O: np.ndarray, leg: np.ndarray, with_bands: bool) -> dict:
            """One decomposition read: components vs leg_w + identity bug gate."""
            weights, text = O - H_odd, H_odd - Vo
            max_err = float(np.max(np.abs((weights + text) - leg)))
            assert max_err <= A5_DECOMP_IDENTITY_TOL, (
                aid,
                f"identity |weights + text − leg_d| max {max_err:.2e} > {A5_DECOMP_IDENTITY_TOL}",
            )
            ref_cos = pcos(pooled_w, leg.mean(axis=0))
            out = {
                "leg_d_pooled_cos": ref_cos,
                "identity_max_abs_err": max_err,
                "component_inter_cos": pcos(weights.mean(axis=0), text.mean(axis=0)),
            }
            for name, comp in (("weights", weights), ("text", text)):
                px = comp.mean(axis=0)
                c = pcos(pooled_w, px)
                per_pref = np.sum(leg_w * comp, axis=1) / (
                    np.linalg.norm(leg_w, axis=1) * np.linalg.norm(comp, axis=1) + 1e-12
                )
                entry = {
                    "pooled_cos": c,
                    "median_per_prefix_cos": float(np.median(per_pref)),
                    "per_prefix_cos": [float(v) for v in per_pref],
                    "pooled_norm": float(np.linalg.norm(px)),
                    "readout_cos_pooled_rb_record_only": pcos(px, rb),
                }
                if with_bands:
                    b = band(px)
                    entry["null_bands"] = b
                    entry["carries_by_factor"] = {
                        str(f): bool(
                            (c < 0)
                            and (abs(c) > b["corpus_cov"]["abs_p95"])
                            and (abs(c) >= f * abs(ref_cos))
                        )
                        for f in A5_CARRY_FACTORS_RECORD
                    }
                    entry["carries"] = entry["carries_by_factor"][str(A5_CARRY_FACTOR)]
                out[name] = entry
            return out

        primary = decompose(dec(aid, 19, "Hbar_odd"), O_all, leg_d, with_bands=True)
        # all-pivot sensitivity — record-only, demoted (plan v6 §4: text_all
        # anti-shares the even-half sampling error with leg_w); no null bands.
        sens_all = decompose(dec(aid, 19, "Hbar_all"), O_all, leg_d, with_bands=False)
        # marker-row-excluded repeat: exclusion applies to the two TRAINED-TEXT
        # stores only, not v̄0 (~0.3% of rows — plan v6 §4 analyzer note (c)).
        O_nomk = dec(aid, 19, "Obar_all_nomk")
        nomk = decompose(dec(aid, 19, "Hbar_odd_nomk"), O_nomk, O_nomk - Vo, with_bands=True)
        expl = {}
        for li in (14, 25):  # exploratory dump (pooled reads only, record-only)
            slot_l = f"{aid}/L{li}/span_mean_context"
            Vb_l = tensors[f"{slot_l}/Vbar0"]
            w_l = (
                (tensors[f"{slot_l}/W_matched"] + Vb_l) - tensors[f"base/marker/L{li}/Vbar0_even"]
            ).mean(axis=0)
            Vo_l = tensors[f"base/marker/L{li}/Vbar0_odd"]
            O_l = tensors[f"{slot_l}/W_onpolicy"] + Vb_l
            H_odd_l = dec(aid, li, "Hbar_odd")
            expl[f"L{li}"] = {
                "weights_pooled_cos": pcos(w_l, (O_l - H_odd_l).mean(axis=0)),
                "text_pooled_cos": pcos(w_l, (H_odd_l - Vo_l).mean(axis=0)),
                "leg_d_pooled_cos": pcos(w_l, (O_l - Vo_l).mean(axis=0)),
            }
        arms_out[aid] = {
            "parent_leg_cos": ref,
            "parity": {
                "leg_d_pooled_cos_rederived": cos_wd,
                "vs_persisted": ref,
                "tol": A5_DECOMP_PARITY_TOL,
                "means_parity_min_cos": float(mp.min()),
                "means_parity_floor": A5_DECOMP_MEANS_COS_MIN,
            },
            "primary": primary,
            "sensitivity_all_pivot_record_only": sens_all,
            "marker_row_excluded": nomk,
            "exploratory_layers_record_only": expl,
            "counts": decomp_in["meta"]["per_arm"].get(aid),
        }

    # JSON-key audit (plan §6 criterion 4): every arm carries both components'
    # pooled cosines, null bands, per-prefix vectors + a carries verdict.
    for aid in marker_ids:
        for blk in ("primary", "marker_row_excluded"):
            for comp in ("weights", "text"):
                e = arms_out[aid][blk][comp]
                for key in ("pooled_cos", "null_bands", "per_prefix_cos", "carries"):
                    assert key in e, (aid, blk, comp, key)

    n = len(marker_ids)
    maj = math.ceil(4 * n / 6)  # the registered ≥4-of-6 majority (plan §3), n-scaled

    def lattice_for(block: str, factor: float) -> dict:
        wc = sum(
            1
            for aid in marker_ids
            if arms_out[aid][block]["weights"]["carries_by_factor"][str(factor)]
        )
        tc = sum(
            1
            for aid in marker_ids
            if arms_out[aid][block]["text"]["carries_by_factor"][str(factor)]
        )
        verdict = (
            "weights-carried"
            if wc >= maj and tc < maj
            else "text-carried"
            if tc >= maj and wc < maj
            else "both-carried"
            if wc >= maj and tc >= maj
            else "interaction-artifact"
        )
        return {
            "n_arms": n,
            "majority": maj,
            "factor": factor,
            "weights_carry_count": wc,
            "text_carry_count": tc,
            "verdict": verdict,
        }

    return {
        "meta": _meta(),
        "convention": (
            "leg_w = matched_all − v̄0_even; leg_d = onpol_all − v̄0_odd; PRIMARY odd-pivot: "
            "weights = onpol_all − H̄_b_tr_odd, text = H̄_b_tr_odd − v̄0_odd (H̄ = per-prefix "
            "odd-half mean of h_base(trained_text)); L19 span_mean_context; norm-matched "
            "null bands iso + corpus-cov, 2000 draws, rng seed = SEED (plan v6 §4)"
        ),
        "params": {
            "parity_tol": A5_DECOMP_PARITY_TOL,
            "means_parity_cos_min": A5_DECOMP_MEANS_COS_MIN,
            "identity_tol": A5_DECOMP_IDENTITY_TOL,
            "carry_factor": A5_CARRY_FACTOR,
            "carry_factors_record": list(A5_CARRY_FACTORS_RECORD),
            "n_null_draws": 2000,
            "seed": SEED,
        },
        "arms": arms_out,
        "lattice": lattice_for("primary", A5_CARRY_FACTOR),
        "lattice_marker_row_excluded": lattice_for("marker_row_excluded", A5_CARRY_FACTOR),
        "lattice_robustness_record_only": {
            str(f): {
                "primary": lattice_for("primary", f),
                "marker_row_excluded": lattice_for("marker_row_excluded", f),
            }
            for f in A5_CARRY_FACTORS_RECORD
            if f != A5_CARRY_FACTOR
        },
        "analyzer_notes": [
            "(a) per-arm trained-vs-base text-identity fractions ride arms.<aid>.counts "
            "(token-id equality from gen_rows) — context for the near-identical-text regime "
            "the odd-pivot registration guards against",
            "(b) the §3 lattice's both-carried branch reads: the weights-carried rewrite of "
            "the A5 marker Takeaways bullet still applies (write-direction violation), with "
            "text-carried change additionally present",
            "(c) marker-row exclusion applies to the two trained-text stores but not v̄0 — "
            "negligible at ~0.3% of rows; note in the sensitivity caption",
        ],
    }


# ── prefix-based v_P -> v_A mapping arm (EXPLORATORY; LOFO over families) ─────


def mapping_arm(cfg: dict, tensors: dict, min_prefixes: int) -> dict:
    """Ridge v_P->v_A across the panel prefixes per (kind, layer, prefix
    position); nested LOFO over prefix families; n<d regularization-limit
    regime FLAGGED per fit; identity+bias + kNN baselines (standing rule)."""
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    fams = np.asarray([cfg["family"][p] for p in cfg["prefix_ids"]])
    lambdas = np.logspace(-2, 8, 11)
    out: dict = {
        "meta": _meta(),
        "label": "EXPLORATORY (plan DESIGN PIN: regularization-limit regime, n~50 < d=3584)",
        "fits": {},
    }

    def ridge_oof(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, dict]:
        n = X.shape[0]
        pred = np.zeros_like(Y)
        sel: dict[str, float] = {}
        for hold in sorted(set(fams)):
            te = fams == hold
            tr = ~te
            # inner LOFO over remaining families selects lambda (never pure GCV)
            inner_scores = []
            for lam in lambdas:
                errs = []
                for inner in sorted(set(fams[tr])):
                    itr = tr & (fams != inner)
                    ite = tr & (fams == inner)
                    Xi, Yi = X[itr], Y[itr]
                    mu_x, mu_y = Xi.mean(0), Yi.mean(0)
                    Xc, Yc = Xi - mu_x, Yi - mu_y
                    K = Xc @ Xc.T
                    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), Yc)
                    P = (X[ite] - mu_x) @ Xc.T @ alpha + mu_y
                    errs.append(float(((P - Y[ite]) ** 2).sum()))
                inner_scores.append(sum(errs))
            lam = float(lambdas[int(np.argmin(inner_scores))])
            sel[hold] = lam
            Xi, Yi = X[tr], Y[tr]
            mu_x, mu_y = Xi.mean(0), Yi.mean(0)
            Xc, Yc = Xi - mu_x, Yi - mu_y
            K = Xc @ Xc.T
            alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), Yc)
            pred[te] = (X[te] - mu_x) @ Xc.T @ alpha + mu_y
        info = {
            "selected_lambda_per_fold": sel,
            "lambda_grid_edge_flag": bool(
                any(v in (lambdas[0], lambdas[-1]) for v in sel.values())
            ),
            "regularization_limit_regime": True,  # n_train < d always here
            "n": int(n),
            "d": int(X.shape[1]),
        }
        return pred, info

    def r2(pred: np.ndarray, Y: np.ndarray) -> float:
        ss = ((Y - Y.mean(0)) ** 2).sum()
        return float(1.0 - ((pred - Y) ** 2).sum() / (ss + 1e-12))

    arms_by_kind = {
        "content": [a["arm_id"] for a in cfg["content_arms"]],
        "marker": [a["arm_id"] for a in cfg["marker_arms"]],
    }
    for kind in ("content", "marker"):
        if f"base/{kind}/L19/Pbar_prefix_span" not in tensors or not arms_by_kind[kind]:
            continue
        for layer in LAYERS_1979:
            for ppos, key in (
                ("prefix_span", "Pbar_prefix_span"),
                ("last_prefix", "Pbar_last_prefix"),
            ):
                X = tensors[f"base/{kind}/L{layer}/{key}"]
                # target: base per-prefix answer means (span) — every arm of this
                # KIND shares the same base store, so any kind-matched arm's
                # Vbar0 slot is the base tensor.
                slot = f"{arms_by_kind[kind][0]}/L{layer}/span_mean_context/Vbar0"
                Yk = tensors[slot]
                if X.shape[0] < max(min_prefixes, 6):
                    continue
                pred, info = ridge_oof(X, Yk)
                ib = np.zeros_like(Yk)
                for hold in sorted(set(fams)):
                    te, tr = fams == hold, fams != hold
                    ib[te] = identity_bias_predict(X[tr], Yk[tr], X[te])
                ks = tuple(k for k in (1, 5) if k <= Yk.shape[0])
                out["fits"][f"{kind}/L{layer}/{ppos}"] = {
                    **info,
                    "lofo_r2": r2(pred, Yk),
                    "identity_bias_lofo_r2": r2(ib, Yk),
                    "knn": {
                        "fitted_cos": knn_retrieval(pred, Yk, ks=ks, metric="cosine"),
                        "fitted_euc": knn_retrieval(pred, Yk, ks=ks, metric="euclidean"),
                        "identity_bias_cos": knn_retrieval(ib, Yk, ks=ks, metric="cosine"),
                        "chance_note": f"chance@k = k/{Yk.shape[0]}",
                    },
                }
    return out


# ── robustness lines + cross-grain (registered analyzer lines) ───────────────


def robustness_lines(cfg: dict, asms: list[dict], out_dir: Path, champs: dict) -> dict:
    lines: dict = {"meta": _meta()}
    # (3) length-partialled rho, top-3 level candidates
    top3 = sorted(
        champs["level"]["across_arm_median_observed"],
        key=lambda c: champs["level"]["across_arm_median_observed"][c],
        reverse=True,
    )[:3]
    length_reads = {}
    for asm in asms:
        f = asm["frame"]
        ln = np.asarray(
            [v if v is not None else np.nan for v in f["content_token_len"]], dtype=np.float64
        )
        ok = np.isfinite(ln)
        ent = {}
        for c in top3:
            col = f["p7"] if c == "p7" else f.get(CAND_COL[c])
            if col is None:
                continue
            ranks = {
                "dv": R._ranks_np(np.asarray(f[asm["dv_names"][0]], dtype=np.float64)[ok]),
                "x": R._ranks_np(np.asarray(col, dtype=np.float64)[ok]),
                "len": R._ranks_np(ln[ok]),
            }
            ent[c] = {"partial_given_length": R._partial_spearman(ranks, "dv", "x", ["len"])}
        length_reads[asm["arm"]["arm_id"]] = ent
    lines["line3_length_partial"] = length_reads
    # (5) elicit-ceiling flags + champion re-read excluding flagged prefixes
    ceiling = {}
    for asm in asms:
        if asm["arm"]["kind"] != "content":
            continue
        f = asm["frame"]
        share = np.asarray(f["base_ceiling_share"], dtype=np.float64)
        flagged = share > 0.4
        entry: dict = {
            "n_flagged": int(flagged.sum()),
            "flagged_prefixes": [p for p, m in zip(f["prefix_id"], flagged) if m],
        }
        if 0 < flagged.sum() < len(share) - 4:
            x, dvs = _x_matrix(asm), _dv_matrix(asm)
            obs_ex = R.observed_rho(x[~flagged], dvs[~flagged])
            entry["observed_rho_excl_ceiling"] = {
                d: {c: float(obs_ex[i, j]) for i, c in enumerate(asm["raced"])}
                for j, d in enumerate(asm["dv_names"])
            }
        ceiling[asm["arm"]["arm_id"]] = entry
    lines["line5_ceiling_reread"] = ceiling
    return lines


def crossgrain_table(
    asms: list[dict], cfg: dict, tensors: dict, tables: dict, inputs_root: Path
) -> dict:
    """(7) per-query #1900 verdicts beside prefix-grain, POSITION-MATCHED
    span-mean<->span-mean; last_prompt primary shown beside (plan §6 line 7)."""
    i1900 = {}
    for name in ("champion_content", "champion_marker"):
        try:
            i1900[name] = _read_repo_json(f"eval_results/issue_1900/race/{name}.json")
        except AssertionError as e:
            i1900[name] = {"note": f"unavailable: {e}"}
    span_med: dict[str, dict[str, list[float]]] = {}
    for asm in asms:
        arm = asm["arm"]
        anchors = load_anchors(inputs_root, arm["mix_arm_id"])
        layer = asm["layer"]
        c_span = candidate_table(cfg, tensors, tables, anchors, arm, layer, SECONDARY_POS)
        f = asm["frame"]
        kept = asm["kept_ix"]
        dv = np.asarray(f[asm["dv_names"][0]], dtype=np.float64)
        for c in asm["raced"]:
            col = (
                np.asarray(f["p7"], dtype=np.float64)
                if c == "p7"
                else (
                    np.asarray(c_span[CAND_COL[c]], dtype=np.float64)[kept]
                    if c_span.get(CAND_COL[c]) is not None
                    else None
                )
            )
            if col is None:
                continue
            span_med.setdefault(arm["kind"], {}).setdefault(c, []).append(R._spearman_np(col, dv))
    table = {
        kind: {c: {"i1979_span_mean_median": float(np.median(v))} for c, v in per_kind.items()}
        for kind, per_kind in span_med.items()
    }
    return {
        "meta": _meta(),
        "position_matched_note": "#1900 raced at span-mean; the comparison column is "
        "span-mean<->span-mean; the last_prompt-primary medians live in the champion "
        "JSONs — any position flip labeled per cell by the analyzer",
        "i1900": {
            k: (
                v.get("across_arm_median_observed")
                or v.get("primary", {}).get("across_arm_median_observed", v)
                if isinstance(v, dict)
                else v
            )
            for k, v in i1900.items()
        },
        "i1979_span_mean": table,
    }


def dump_grid(cfg: dict, tensors: dict, tables: dict, inputs_root: Path, asms: list[dict]) -> dict:
    """Exploratory dump (plan §6): observed rho over ALL (layer x position)
    cells per arm, tc + ps anchors — pure re-reductions, no inference."""
    grid: dict = {"meta": _meta(), "arms": {}}
    for asm in asms:
        arm = asm["arm"]
        anchors = load_anchors(inputs_root, arm["mix_arm_id"])
        kept = asm["kept_ix"]
        dv = np.asarray(asm["frame"][asm["dv_names"][0]], dtype=np.float64)
        cells: dict = {}
        for layer in LAYERS_1979:
            for pos in ("span_mean_context", "last_prompt", "last_ctx"):
                cands = candidate_table(cfg, tensors, tables, anchors, arm, layer, pos)
                cell = {}
                for name, col in cands.items():
                    if col is None:
                        continue
                    cell[name] = R._spearman_np(np.asarray(col, dtype=np.float64)[kept], dv)
                cell["p7"] = R._spearman_np(np.asarray(asm["frame"]["p7"], dtype=np.float64), dv)
                cells[f"L{layer}/{pos}"] = cell
        grid["arms"][arm["arm_id"]] = cells
    return grid


# ── mediation (content arms; #1900 lattice re-grained) ───────────────────────


def run_mediation(asms: list[dict], out_dir: Path) -> dict:
    """Rank partials + commonality per content arm via issue1900_race helpers."""
    import pandas as pd

    per_arm = {}
    for asm in asms:
        if asm["arm"]["kind"] != "content":
            continue
        f = asm["frame"]
        cols = {k: v for k, v in f.items() if isinstance(v, np.ndarray)}
        cols["dv_level"] = np.asarray(f["dv_level"], dtype=np.float64)
        per_arm[asm["arm"]["arm_id"]] = R.mediation_arm(pd.DataFrame(cols), asm["raced"])
    out = {"meta": _meta(), "per_arm": per_arm, "lattice": R.mediation_lattice(per_arm)}
    _atomic_json(out_dir / "mediation.json", out)
    return out


# ── fixtures (schema-exact tiny tree; labeled — never uploaded as real) ───────


def build_fixtures(root: Path) -> None:
    """Write a schema-exact FIXTURE input tree (8 prefixes x 4 queries x
    2 content + 2 marker arms, d=64) for the end-to-end F3/figs smoke."""
    import torch

    rng = np.random.default_rng(7)
    d, P, Q, NL = 64, 8, 4, 28
    fams = [
        "trained",
        "trained",
        "negatives",
        "negatives",
        "bystander",
        "battery",
        "conv-fresh",
        "near-twin",
    ]
    pids = [
        "persona_software_engineer",
        "bare",
        "neg_sp_police",
        "neg_default_assistant",
        "persona_villain",
        "bat_01",
        "cf_x1",
        "nt_backend",
    ]
    members = [
        {
            "prefix_id": p,
            "family": fam,
            "system": f"fixture system {p}",
            "prefix_turns": [],
            "user_wrap": None,
            "content_token_len": int(rng.integers(10, 700)),
        }
        for p, fam in zip(pids, fams, strict=True)
    ]
    queries = [
        {
            "prompt": f"fixture question {i}?",
            "sha": hashlib.sha256(f"q{i}".encode()).hexdigest()[:16],
        }
        for i in range(Q)
    ]
    arms = []
    for aid, kind, beh, ctx, regime in (
        ("imp-pers-con-lr3e5-s42", "content", "imp", "pers", "con"),
        ("imp-pers-po-lr1e5-s42", "content", "imp", "pers", "po"),
        ("mk-pers-con-lr5e6-s42", "marker", "mk", "pers", "con"),
        ("mk-bare-con-lr5e6-s42", "marker", "mk", "bare", "con"),
    ):
        arms.append(
            {
                "arm_id": aid,
                "kind": kind,
                "beh_key": beh,
                "ctx_key": ctx,
                "regime": regime,
                "seed": 42,
                "lr": 1e-5,
                "step": 25,
                "method": "lora",
                "base_unit": "base_content" if kind == "content" else "base_mk",
                "mix_arm_id": f"mix_{aid}",
                "mix_layout": "fixture",
                "mix_pos_path": "fixture/pos.jsonl",
                "primary_layer": 19 if kind == "content" else 25,
                "adapter_repo": "fixture",
                "adapter_subfolder": "fixture",
            }
        )
    cdir = root / "config"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "prefix_panel.json").write_text(json.dumps({"members": members, "n_members": P}))
    (cdir / "queries.json").write_text(json.dumps({"queries": queries}))
    (cdir / "arms.json").write_text(json.dumps({"arms": arms}))
    inputs = root / "inputs"
    # latent per-prefix signal so ranks are non-degenerate + correlated
    u = rng.standard_normal(P)
    tensors: dict = {}
    tables: dict = {"prefix_ids": pids, "layers": [14, 19, 25]}
    reads_arms: dict = {}

    def vecs(scale=1.0):
        return u[:, None] * rng.standard_normal(d)[None, :] * scale + 0.3 * rng.standard_normal(
            (P, d)
        )

    for kind in ("content", "marker"):
        for L in LAYERS_1979:
            bs = f"base/{kind}/L{L}"
            for k in ("Pbar_prefix_span", "Pbar_last_prefix", "Vbar0_even", "Vbar0_odd"):
                tensors[f"{bs}/{k}"] = torch.tensor(vecs(), dtype=torch.float16)
            for mpos in ("span_mean", "last_prompt"):
                tensors[f"m0pred/{kind}/L{L}/{mpos}"] = torch.tensor(vecs(), dtype=torch.float16)
    for a in arms:
        aid = a["arm_id"]
        tables[aid] = {}
        reads_arms[aid] = {}
        for L in LAYERS_1979:
            for mpos in ("span_mean", "last_prompt"):
                tensors[f"m0anchor/{a['mix_arm_id']}/L{L}/{mpos}"] = torch.tensor(
                    rng.standard_normal(d), dtype=torch.float16
                )
            for pos in ("span_mean_context", "last_prompt", "last_ctx"):
                slot = f"{aid}/L{L}/{pos}"
                for k in ("W_matched", "W_onpolicy", "Cbar", "Vbar0", "Cbar_post"):
                    tensors[f"{slot}/{k}"] = torch.tensor(vecs(), dtype=torch.float16)
                tables[aid][f"L{L}/{pos}"] = {
                    "p4_gpred": [float(v) for v in (u + 0.5 * rng.standard_normal(P))]
                }
                if pos == "span_mean_context":
                    reads_arms[aid][f"L{L}/{pos}"] = {
                        "gate_read": {
                            "spearman_rho": float(
                                np.clip(0.5 + 0.2 * rng.standard_normal(), -1, 1)
                            ),
                            "p_value": 0.1,
                            "n": P,
                        }
                    }
        for mpos in ("span_mean", "last_prompt"):
            tensors[f"{aid}/m0pred_Cbar_post/L19/{mpos}"] = torch.tensor(
                vecs(), dtype=torch.float16
            )
        if a["kind"] == "content":
            tensors[f"{aid}/p8_wmap_pred_L19"] = torch.tensor(vecs(), dtype=torch.float16)
    (inputs / "battery").mkdir(parents=True, exist_ok=True)
    torch.save(tensors, inputs / "battery/ingredient_tensors.pt")
    (inputs / "battery/battery_reads.json").write_text(json.dumps({"meta": {}, "arms": reads_arms}))
    torch.save(
        {
            "shrinkage": 0.1,
            **{
                f"L{L}": {
                    "chol": torch.eye(d, dtype=torch.float16),
                    "top_eig": torch.zeros(d),
                    "n_rows": 100,
                }
                for L in LAYERS_1979
            },
        },
        inputs / "battery/sigma_chol.pt",
    )
    (inputs / "predictor_tables").mkdir(parents=True, exist_ok=True)
    (inputs / "predictor_tables/predictor_ingredients.json").write_text(json.dumps(tables))
    for a in arms:
        adir = inputs / "anchors" / a["mix_arm_id"]
        adir.mkdir(parents=True, exist_ok=True)
        anc: dict = {"mix": a["mix_arm_id"], "n_rows": 20}
        for L in LAYERS_1979:
            anc[f"L{L}"] = {
                "A_ctx_span": torch.tensor(rng.standard_normal(d)),
                "A_ans": torch.tensor(rng.standard_normal(d)),
                "A_ctx_last_prompt": torch.tensor(rng.standard_normal(d)),
                "A_ctx_last_ctx": torch.tensor(rng.standard_normal(d)),
                "rows_ctx": torch.tensor(rng.standard_normal((20, d)), dtype=torch.float16),
                "rows_ans": torch.tensor(rng.standard_normal((20, d)), dtype=torch.float16),
                "half_even_ctx": torch.tensor(rng.standard_normal(d)),
                "half_odd_ctx": torch.tensor(rng.standard_normal(d)),
                "half_even_ans": torch.tensor(rng.standard_normal(d)),
                "half_odd_ans": torch.tensor(rng.standard_normal(d)),
            }
        torch.save(anc, adir / "anchors.pt")
    rbdir = inputs / "rb"
    rbdir.mkdir(parents=True, exist_ok=True)
    for name in ("rb_sycophancy.pt", "rb_marker.pt", "rb_impolite.pt", "rb_writing_style.pt"):
        torch.save({"rb": torch.tensor(rng.standard_normal((NL, d)))}, rbdir / name)

    # marker slot shards (own/base per marker arm + base_mk base-slot)
    def slot_rows(offset: float) -> list[dict]:
        rows = []
        for i, p in enumerate(pids):
            for q in queries:
                z_m = float(offset + 2.0 * u[i] + rng.standard_normal())
                z_e = float(5.0 + rng.standard_normal())
                logz = float(np.logaddexp(z_m, z_e) + 1.0)
                rows.append(
                    {
                        "row_sha": hashlib.sha256(f"{p}||{q['sha']}".encode()).hexdigest()[:16],
                        "prefix_id": p,
                        "query_sha": q["sha"],
                        "logp": z_m - logz,
                        "z_marker": z_m,
                        "z_eos": z_e,
                        "logZ": logz,
                    }
                )
        return rows

    for a in arms:
        if a["kind"] != "marker":
            continue
        mdir = inputs / "marker_tf" / a["arm_id"]
        mdir.mkdir(parents=True, exist_ok=True)
        for side, off in (("own", 2.0), ("base", -4.0)):
            with (mdir / f"slot_{side}.shard00.jsonl").open("w") as fh:
                for r in slot_rows(off):
                    fh.write(json.dumps(r) + "\n")
    bdir = inputs / "marker_tf" / "base_mk"
    bdir.mkdir(parents=True, exist_ok=True)
    with (bdir / "slot_base.shard00.jsonl").open("w") as fh:
        for r in slot_rows(-6.0):
            fh.write(json.dumps(r) + "\n")
    # judge fixtures (base_imp + the two imp arms)
    jdir = root / "judge"
    jdir.mkdir(parents=True, exist_ok=True)
    for tag, bump in (
        ("base_imp", 0.0),
        ("imp-pers-con-lr3e5-s42", 15.0),
        ("imp-pers-po-lr1e5-s42", 25.0),
    ):
        rows = []
        for i, p in enumerate(pids):
            for q in queries:
                s = float(np.clip(40 + bump + 12 * u[i] + 6 * rng.standard_normal(), 0, 100))
                rows.append(
                    {
                        "sha": hashlib.sha256(f"{p}||{q['sha']}".encode()).hexdigest()[:16],
                        "prefix_id": p,
                        "query_sha": q["sha"],
                        "score_mean": s,
                        "kept_draw_scores": [s],
                        "n_kept_draws": 1,
                        "binary_rate": float(s >= 50),
                        "n_transport_lost": 0,
                    }
                )
        (jdir / f"arm_scores_{tag}.json").write_text(
            json.dumps(
                {
                    "meta": {"unit": tag, "fixture": True},
                    "judge": {"n_draws": 1, "max_tokens": 400, "rubric_sha256": "f" * 64},
                    "n_items": len(rows),
                    "n_scored_items": len(rows),
                    "n_total_draws": len(rows),
                    "n_content_dropped_draws": 0,
                    "n_refusal_draws": 0,
                    "n_transport_lost_draws": 0,
                    "content_drop_rate": 0.0,
                    "rows": rows,
                }
            )
        )
    (root / "FIXTURES.md").write_text(
        "Schema-exact SMOKE fixtures for issue1979_race/figs — synthetic, never real data.\n"
    )
    print(f"[fixtures] wrote {root}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────


def run_decomp_only(args) -> int:
    """``--decomp-only`` entry (plan v6 §4 Diff 3): stage the standard battery
    inputs + ``battery/basetf_decomp_inputs.pt``, run ONLY ``a5_decomposition``
    + its figure panel, exit — no judge reads, no bootstrap battery."""
    import torch

    t0 = time.time()
    print("[phase=decomp_load]", flush=True)
    cfg = load_config(args.config_dir)
    if args.arms:
        keep = {a for a in args.arms.split(",") if a}
        cfg["marker_arms"] = [a for a in cfg["marker_arms"] if a["arm_id"] in keep]
    assert cfg["marker_arms"], "no marker arms selected"
    marker_ids = [a["arm_id"] for a in cfg["marker_arms"]]
    if not args.skip_stage:
        mixes = sorted({a["mix_arm_id"] for a in cfg["marker_arms"]})
        stage_inputs(args.inputs_root, mixes, marker_ids)
        dest = args.inputs_root / "battery/basetf_decomp_inputs.pt"
        if not dest.exists():  # staged inside the branch — stage_inputs untouched
            from explore_persona_space.orchestrate import hub

            hub.stage_hub_file(
                J._data_repo(),
                f"{HF_PREFIX_1979}/battery/basetf_decomp_inputs.pt",
                dest,
                repo_type="dataset",
            )
    tensors = load_tensors(args.inputs_root)
    load_rb(args.inputs_root)
    decomp_in = torch.load(
        args.inputs_root / "battery/basetf_decomp_inputs.pt",
        map_location="cpu",
        weights_only=False,
    )
    if args.parity_ref is not None:
        parity_ref = json.loads(Path(args.parity_ref).read_text())
    else:  # the committed parent A5 verdicts (plan §10 parity reference)
        parity_ref = _read_repo_json("eval_results/issue_1979/race/battery_verdicts.json")

    print("[phase=decomp_stats]", flush=True)
    result = a5_decomposition(cfg, tensors, args.inputs_root, decomp_in, parity_ref)
    _atomic_json(args.out_dir / "a5_decomposition.json", result)
    print("[phase=decomp_fig]", flush=True)
    from issue1979_figs import a5_decomposition_panel

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    a5_decomposition_panel(result, args.fig_dir)
    print(
        f"[phase=done] decomp arms={len(result['arms'])} "
        f"verdict={result['lattice']['verdict']} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--inputs-root", type=Path, default=REPO_ROOT / "data" / "issue_1979" / "race_inputs"
    )
    ap.add_argument("--judge-dir", type=Path, default=REPO_ROOT / "eval_results/issue_1979/judge")
    ap.add_argument("--config-dir", type=Path, default=REPO_ROOT / "eval_results/issue_1979/config")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "eval_results/issue_1979/race")
    ap.add_argument("--arms", default="", help="comma list (smoke subset)")
    ap.add_argument("--b-draws", type=int, default=B_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--min-prefixes", type=int, default=20)
    ap.add_argument("--skip-stage", action="store_true", help="inputs already local (fixtures)")
    ap.add_argument("--build-fixtures", type=Path, default=None, metavar="DIR")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument(
        "--decomp-only",
        action="store_true",
        help="A5 weights-vs-text decomposition ONLY (amendment marker-a5-weights-vs-text, "
        "plan v6 §4 Diff 3) — no judge reads, no bootstrap battery",
    )
    ap.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures/issue_1979")
    ap.add_argument(
        "--parity-ref",
        type=Path,
        default=None,
        help="battery_verdicts.json parity reference (default: the committed "
        "eval_results/issue_1979/race/battery_verdicts.json)",
    )
    args = ap.parse_args(argv)

    if args.import_check:
        import inspect

        import pandas  # noqa: F401
        import torch  # noqa: F401
        from scipy.linalg import solve_triangular  # noqa: F401
        from scipy.stats import spearmanr  # noqa: F401

        import issue1768_directions as DIR

        from explore_persona_space.analysis.mapping_baselines import (
            identity_bias_predict,
            knn_retrieval,
        )

        inspect.signature(R.bootstrap_battery).bind(
            x_cands=np.zeros((5, 2)), dvs=np.zeros((5, 1)), b_draws=2, seed=1
        )
        inspect.signature(R.observed_rho).bind(x_cands=np.zeros((5, 2)), dvs=np.zeros((5, 1)))
        inspect.signature(R.perm_null).bind(
            x_cands=np.zeros((5, 2)), dv=np.zeros(5), n_perm=2, seed=1
        )
        inspect.signature(R.mediation_arm).bind(None, [])
        inspect.signature(DIR.rank_read).bind(delta_v=np.zeros((5, 4)), w=np.zeros(4))
        inspect.signature(DIR.load_rb_tensors).bind(Path("/tmp"), rb_dir=Path("/tmp"))
        inspect.signature(identity_bias_predict).bind(
            np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((2, 3))
        )
        inspect.signature(knn_retrieval).bind(np.zeros((4, 3)), np.zeros((4, 3)), ks=(1,))
        # --decomp-only deferred imports (plan v6 §4 Diff 3)
        from explore_persona_space.orchestrate import hub

        from issue1979_figs import a5_decomposition_panel

        inspect.signature(a5_decomposition_panel).bind(decomp={}, out_dir=Path("/tmp"))
        inspect.signature(hub.stage_hub_file).bind(
            repo_id="r", path_in_repo="p", target=Path("/tmp/x"), repo_type="dataset"
        )
        print("[import-check] OK — race deferred imports resolved + signature-bound")
        return 0

    if args.build_fixtures is not None:
        build_fixtures(args.build_fixtures)
        return 0

    if args.decomp_only:
        return run_decomp_only(args)

    t0 = time.time()
    print("[phase=f3_load]", flush=True)
    cfg = load_config(args.config_dir)
    if args.arms:
        keep = set(a for a in args.arms.split(",") if a)
        cfg["content_arms"] = [a for a in cfg["content_arms"] if a["arm_id"] in keep]
        cfg["marker_arms"] = [a for a in cfg["marker_arms"] if a["arm_id"] in keep]
    all_arms = cfg["content_arms"] + cfg["marker_arms"]
    assert all_arms, "no arms selected"
    mixes = sorted({a["mix_arm_id"] for a in all_arms})
    if not args.skip_stage:
        stage_inputs(args.inputs_root, mixes, [a["arm_id"] for a in cfg["marker_arms"]])
    tensors = load_tensors(args.inputs_root)
    tables = json.loads(
        (args.inputs_root / "predictor_tables/predictor_ingredients.json").read_text()
    )
    load_rb(args.inputs_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[phase=f3_assemble]", flush=True)
    asms = []
    for k, arm in enumerate(all_arms):
        asm = assemble_arm(
            cfg, tensors, tables, args.inputs_root, args.judge_dir, arm, args.min_prefixes
        )
        asms.append(asm)
        frame_out = {
            k2: (v.tolist() if isinstance(v, np.ndarray) else v) for k2, v in asm["frame"].items()
        }
        _atomic_json(
            args.out_dir / f"frame_{arm['arm_id']}.json",
            {
                "meta": _meta(),
                "layer": asm["layer"],
                "pos": asm["pos"],
                "n_realized": asm["n_realized"],
                "coverage_floor": asm["coverage_floor"],
                "frame": frame_out,
            },
        )
        print(
            f"[f3] assemble {k + 1}/{len(all_arms)} {arm['arm_id']} n={asm['n_realized']}",
            flush=True,
        )

    print("[phase=f3_battery]", flush=True)
    by_kind: dict[str, list[dict]] = {}
    for asm in asms:
        by_kind.setdefault(asm["arm"]["kind"], []).append(asm)
    shared_by_kind = {k: _shared_pool(v) for k, v in by_kind.items()}
    for asm in asms:
        run_arm_battery(
            asm, args.out_dir, shared_by_kind[asm["arm"]["kind"]], args.b_draws, args.n_perm
        )

    print("[phase=f3_champion]", flush=True)
    champs: dict = {}
    if "content" in by_kind:
        ids = [a["arm"]["arm_id"] for a in by_kind["content"]]
        for label, dv_ix, inc, names in (
            ("level", 0, "p7", ("P7-retains-level-champion", "geometry-dethrones-level")),
            ("change", 1, "p2", ("answer-sim-retains-change-champion", "other-change-champion")),
        ):
            per_family = {
                fam_label: champion(
                    ids, args.out_dir, stem, dv_ix, f"{label} ({fam_label})", inc, names
                )
                for fam_label, stem in (
                    ("prefix_resample_PRIMARY", "boot"),
                    ("query_cluster", "qboot"),
                    ("family_cluster_sensitivity", "fboot"),
                )
            }
            champs[label] = per_family["prefix_resample_PRIMARY"]
            _atomic_json(args.out_dir / f"champion_{label}.json", {"meta": _meta(), **per_family})
    if "marker" in by_kind:
        ids = [a["arm"]["arm_id"] for a in by_kind["marker"]]
        per_family = {
            fam_label: champion(
                ids,
                args.out_dir,
                stem,
                0,
                f"marker dlogp ({fam_label})",
                "p7",
                ("P7-retains-level-champion", "geometry-dethrones-level"),
            )
            for fam_label, stem in (
                ("prefix_resample_PRIMARY", "boot"),
                ("query_cluster", "qboot"),
                ("family_cluster_sensitivity", "fboot"),
            )
        }
        champs["marker"] = per_family["prefix_resample_PRIMARY"]
        _atomic_json(args.out_dir / "champion_marker.json", {"meta": _meta(), **per_family})

    print("[phase=f3_verdicts]", flush=True)
    mediation = run_mediation(asms, args.out_dir)
    h5 = h5_residuals(asms)
    _atomic_json(args.out_dir / "h5_residuals.json", h5)
    battery = battery_verdicts(cfg, tensors, args.inputs_root, asms)
    _atomic_json(args.out_dir / "battery_verdicts.json", battery)
    mapping = mapping_arm(cfg, tensors, args.min_prefixes)
    _atomic_json(args.out_dir / "mapping_arm.json", mapping)
    rob = robustness_lines(
        cfg, asms, args.out_dir, champs if "level" in champs else {"level": champs.get("marker")}
    )
    _atomic_json(args.out_dir / "robustness.json", rob)
    cg = crossgrain_table(asms, cfg, tensors, tables, args.inputs_root)
    _atomic_json(args.out_dir / "crossgrain.json", cg)
    _atomic_json(
        args.out_dir / "dump_grid.json", dump_grid(cfg, tensors, tables, args.inputs_root, asms)
    )

    summary = {
        "meta": _meta(),
        "n_arms": len(asms),
        "arms": [a["arm"]["arm_id"] for a in asms],
        "H1_level_champion": champs.get("level", {}).get("verdict"),
        "H2_change_champion": champs.get("change", {}).get("verdict"),
        "H3_mediation": mediation["lattice"]["verdict"] if mediation["per_arm"] else None,
        "H4_A7_gate": battery["A7"].get("verdict"),
        "H4_A6_share_ge_criterion": battery["A6"]["share_ge_criterion"],
        "H5": h5["verdict"],
        "marker_champion": champs.get("marker", {}).get("verdict"),
        "reliability_sqrt_rel_per_arm": {
            a["arm"]["arm_id"]: json.loads(
                (args.out_dir / f"arm_{a['arm']['arm_id']}.json").read_text()
            )["dv_split_half"]["sqrt_rel"]
            for a in asms
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    _atomic_json(args.out_dir / "summary.json", summary)
    print(f"[phase=done] f3 arms={len(asms)} elapsed={time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
