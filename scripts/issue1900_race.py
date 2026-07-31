"""#1900 P3 stats — the per-context leakage-predictor horse race (VM, CPU).

Consumes the P1 predictor tables + marker three-space parquets (HF mirror or
local out root) and the P2 judge scores, and produces the registered race
statistics (plan §4 P3 / §6):

- per-arm Spearman per candidate over the realized listwise row set;
- paired bootstrap B=2,000 with SHARED per-draw context indices across
  candidates + DVs within arm — batched `torch.searchsorted` midranks over
  stacked (draws x series) tensors, NO per-draw Python loop;
- permutation null: 1,000 within-arm DV permutations with per-draw max-rho
  re-selection over the raced candidates (selection-symmetric, SIGNED rho —
  the registered winner convention);
- champion read: across-arm median per candidate, winner-per-draw, P(win),
  selection-inherited CI (frozen CI also persisted, labeled);
- registered DV-identity companion columns: content graded CHANGE
  (trained − base) companion race; marker trained-side LEVEL log P companion;
- mediation (rank partials DV~P1|P7, DV~P2|P7, DV~P1|(P2,P7), DV~P2|(P1,P7);
  P3-structural read; commonality decomposition over {P1,P2,P7};
  rank-Pearson(P1,P2) instability flag; even/odd half-anchor recount);
- fitted combination predictor (within-arm rank+z ridge; leave-one-ARM-out +
  leave-one-BEHAVIOR-out + LMSYS<->WildChat folds);
- marker race replication panel (three-space DV) + the M-panel (separate,
  never headline);
- the §6 registered analyzer robustness lines (1)-(8), each persisted.

Per-draw x per-candidate matrices persist per arm (`race/boot_<arm>.npz`,
`race/perm_<arm>.npz`) — the selection-symmetric-nulls persistence contract.

Smoke (`--build-smoke-inputs` then `--smoke`): builds a 64-row input set from
REAL #1768 store tensors via the REAL `issue1768_fit.load_corpus_cell` loader
(one real content cell staged at the pin; smoke anchors = means of real row
halves, CLEARLY LABELED smoke-grade) — no `torch.randn` anywhere; where a
P1-dependent input cannot exist pre-dispatch (judge DV beyond the 2 really
judged rows; marker four-floats), the stand-in is a REAL-tensor functional,
labeled per slice in the builder meta. The race + figs then run the SAME
production entrypoints against the smoke root.
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

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1900_judge as J  # noqa: E402  (input loaders + config staging)
import issue1900_prep as P0  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1900.race")

ISSUE = 1900
HF_PREFIX = P0.HF_PREFIX
CORPUS_PIN = P0.CORPUS_PIN
SEED = 1900
B_BOOT = 2_000  # plan §11 (#1768 horse-race CI convention); descope lever 500
N_PERM = 1_000
BOOT_CHUNK = 250  # draw-block size bounding peak RSS (batched inside)
MIN_ROWS = 50  # per-arm realized-row floor (fail loud below)
PRIMARY_LAYER = {"content": 19, "marker": 25}  # pre-registered (plan §11)
CI_QS = (0.025, 0.975)

# The 11 raced deployable candidates (plan §5) -> predictor-table column
# (None = sourced outside the table: p7 = base propensity).
CANDIDATE_COLS: dict[str, str | None] = {
    "p1": "p1_tc",
    "p2": "p2_tc",
    "p3a": "p3a_tc",
    "p3b": "p3b_tc",
    "p4": "p4_tc",
    "p5": "p5",
    "p6": "p6",
    "p7": None,
    "p8a": "p8a",
    "p8b": "p8b",
    "p9": "p9_k16",
}
PS_COLS = {"p1": "p1_ps", "p2": "p2_ps", "p3a": "p3a_ps", "p3b": "p3b_ps", "p4": "p4_ps"}
M_PANEL_COLS = {"m1": "m1_tc", "m2": "m2_tc", "m3": "m3", "m4": "m4", "m5": "m5_tc", "m6": "m6"}
BEH_BY_KEY = J.BEH_BY_KEY


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — metadata only
        return "unknown"


def _meta() -> dict:
    import torch

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "issue": ISSUE,
        "seed": SEED,
        "corpus_pin": CORPUS_PIN,
    }


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


# ── rank / correlation primitives ────────────────────────────────────────────


def _ranks_np(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(x, method="average").astype(np.float64)


def _spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _midranks_t(v):
    """Batched average ranks along the last dim (tie-correct Spearman input).

    midrank_i = (#{x < v_i} + #{x <= v_i} + 1) / 2 via two batched
    searchsorted calls on the sorted values — unique element at sorted
    position i gets rank i+1; a k-tie block gets the block's average rank.
    """
    import torch

    s, _ = torch.sort(v, dim=-1)
    lo = torch.searchsorted(s, v, right=False)
    hi = torch.searchsorted(s, v, right=True)
    return (lo + hi + 1).to(torch.float32) / 2.0


def _rank_z_t(v):
    """Midranks -> centered, unit-norm-per-sd z rows; degenerate rows -> 0."""
    r = _midranks_t(v)
    z = r - r.mean(dim=-1, keepdim=True)
    sd = z.std(dim=-1, unbiased=False, keepdim=True)
    degen = sd.squeeze(-1) <= 1e-12
    return z / sd.clamp_min(1e-12), degen


def bootstrap_battery(
    x_cands: np.ndarray, dvs: np.ndarray, b_draws: int, seed: int
) -> tuple[np.ndarray, int]:
    """(B, K, D) per-draw Spearman rho; shared row indices per draw.

    Batched: per draw-block, gather -> midranks (searchsorted) -> rank-Pearson
    einsum. Returns (rho, n_degenerate_series_draws) — a resampled series with
    zero variance contributes rho 0 for that draw and is COUNTED (reported per
    arm; a heavy count flags a floored DV, never silently).
    """
    import torch

    n, k = x_cands.shape
    d = dvs.shape[1]
    stacked = torch.from_numpy(
        np.concatenate([x_cands, dvs], axis=1).astype(np.float32)
    )  # (n, K+D)
    rng = np.random.default_rng(seed)
    out = np.empty((b_draws, k, d), dtype=np.float32)
    n_degen = 0
    for b0 in range(0, b_draws, BOOT_CHUNK):
        nb = min(BOOT_CHUNK, b_draws - b0)
        idx = torch.from_numpy(rng.integers(0, n, size=(nb, n)))
        vals = stacked[idx]  # (nb, n, K+D)
        vt = vals.permute(0, 2, 1).contiguous()  # (nb, S, n)
        z, degen = _rank_z_t(vt)
        n_degen += int(degen.sum().item())
        zc, zd = z[:, :k], z[:, k:]
        rho = torch.einsum("bkn,bdn->bkd", zc, zd) / n
        out[b0 : b0 + nb] = rho.numpy()
    return out, n_degen


def perm_null(x_cands: np.ndarray, dv: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """(P, K) Spearman rho of every candidate against permuted-DV draws.

    One GEMM over rank-z matrices; per-draw max-over-candidates re-selection
    happens at the consumer (signed max — the registered winner convention).
    """
    import torch

    n, k = x_cands.shape
    zc, _ = _rank_z_t(torch.from_numpy(x_cands.astype(np.float32)).T.contiguous())  # (K, n)
    zd, _ = _rank_z_t(torch.from_numpy(dv.astype(np.float32))[None, :])  # (1, n)
    zd = zd[0]
    rng = np.random.default_rng(seed + 1)
    perm = np.argsort(rng.random((n_perm, n)), axis=1)
    zdp = torch.from_numpy(zd.numpy()[perm])  # (P, n)
    rho = (zdp @ zc.T) / n  # (P, K)
    return rho.numpy()


def observed_rho(x_cands: np.ndarray, dvs: np.ndarray) -> np.ndarray:
    """(K, D) observed Spearman rho (tie-correct, same primitive as the boot)."""
    import torch

    stacked = torch.from_numpy(np.concatenate([x_cands, dvs], axis=1).astype(np.float32))
    z, _ = _rank_z_t(stacked.T.contiguous()[None])  # (1, S, n)
    k = x_cands.shape[1]
    zc, zd = z[0, :k], z[0, k:]
    return ((zc @ zd.T) / x_cands.shape[0]).numpy()


def _partial_spearman(df_ranks: dict[str, np.ndarray], y: str, x: str, given: list[str]):
    """Partial Spearman = Pearson of rank residuals after OLS on `given` (+1)."""
    n = len(df_ranks[y])
    if n <= len(given) + 3:
        return None
    z_mat = np.column_stack([np.ones(n)] + [df_ranks[g] for g in given])
    res = {}
    for col in (y, x):
        beta, *_ = np.linalg.lstsq(z_mat, df_ranks[col], rcond=None)
        res[col] = df_ranks[col] - z_mat @ beta
    sy, sx = np.std(res[y]), np.std(res[x])
    if sy <= 1e-12 or sx <= 1e-12:
        return None
    return float(np.corrcoef(res[y], res[x])[0, 1])


def _rank_r2(df_ranks: dict[str, np.ndarray], y: str, xs: list[str]) -> float:
    n = len(df_ranks[y])
    x_mat = np.column_stack([np.ones(n)] + [df_ranks[x] for x in xs])
    beta, *_ = np.linalg.lstsq(x_mat, df_ranks[y], rcond=None)
    pred = x_mat @ beta
    ss_res = float(np.sum((df_ranks[y] - pred) ** 2))
    ss_tot = float(np.sum((df_ranks[y] - np.mean(df_ranks[y])) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


# ── input assembly ───────────────────────────────────────────────────────────


def _stage_from_hf(root: Path, rel_paths: list[str]) -> None:
    """Stage missing P1 artifacts from the HF mirror (scoped, per-file)."""
    from explore_persona_space.orchestrate import hub

    missing = [p for p in rel_paths if not (root / p).exists()]
    for p in missing:
        hub.stage_hub_file(J._data_repo(), f"{HF_PREFIX}/{p}", root / p, repo_type="dataset")
    if missing:
        logger.info("[stage] staged %d P1 files from %s", len(missing), HF_PREFIX)


class MissingRaceInput(RuntimeError):
    """A required staged input file is absent (typed so `exploration_grid`'s
    skip stays narrow — every OTHER failure keeps its fail-fast character)."""


def load_scores(judge_dir: Path, name: str) -> dict:
    path = judge_dir / f"arm_scores_{name}.json"
    if not path.exists():
        raise MissingRaceInput(f"missing judge scores: {path}")
    return json.loads(path.read_text())


def _score_maps(payload: dict) -> tuple[dict, dict, dict]:
    """sha -> (mean, binary_rate, kept draws) over scored rows only (listwise)."""
    means, rates, draws = {}, {}, {}
    for r in payload["rows"]:
        if r["score_mean"] is not None:
            means[r["sha"]] = float(r["score_mean"])
            rates[r["sha"]] = float(r["binary_rate"])
            draws[r["sha"]] = list(r["kept_draw_scores"])
    return means, rates, draws


def assemble_content_arm(
    arm: dict, tables_dir: Path, judge_dir: Path, layer: int | None = None
) -> dict:
    """Joined listwise frame for one content arm at its primary layer."""
    import pandas as pd

    layer = layer or arm["primary_layer"]
    tab = pd.read_parquet(tables_dir / f"{arm['arm_id']}_L{layer}.parquet")
    tab = tab[tab["in_judge_subset"]].reset_index(drop=True)
    arm_scores = load_scores(judge_dir, arm["arm_id"])
    base_scores = load_scores(judge_dir, f"base_{arm['beh_key']}")
    a_mean, a_rate, _ = _score_maps(arm_scores)
    b_mean, _b_rate, _ = _score_maps(base_scores)
    tab["dv_level"] = tab["sha"].map(a_mean)
    tab["dv_binary"] = tab["sha"].map(a_rate)
    tab["p7"] = tab["sha"].map(b_mean)
    tab["dv_change"] = tab["dv_level"] - tab["p7"]
    raced = [
        c
        for c, col in CANDIDATE_COLS.items()
        if c == "p7" or (col in tab.columns and tab[col].notna().any())
    ]
    need = [CANDIDATE_COLS[c] for c in raced if c != "p7"] + ["p7", "dv_level", "dv_binary"]
    frame = tab.dropna(subset=need).reset_index(drop=True)
    assert len(frame) >= MIN_ROWS, (arm["arm_id"], len(frame), "below realized-row floor")
    return {
        "arm": arm,
        "layer": layer,
        "frame": frame,
        "raced": raced,
        "dv_names": ["dv_level", "dv_change", "dv_binary"],
        "n_realized": int(len(frame)),
        "n_subset_rows": int(len(tab)),
    }


def assemble_marker_arm(
    arm: dict, tables_dir: Path, marker_dir: Path, layer: int | None = None
) -> dict:
    """Joined listwise frame for one marker arm (three-space DV columns)."""
    import pandas as pd

    layer = layer or arm["primary_layer"]
    aid = arm["arm_id"]
    tab = pd.read_parquet(tables_dir / f"{aid}_L{layer}.parquet")
    tab = tab[tab["in_judge_subset"]].reset_index(drop=True)
    arm_p = pd.read_parquet(marker_dir / f"{aid}__on__{aid}_slots.parquet")
    base_p = pd.read_parquet(marker_dir / f"base__on__{aid}_slots.parquet")
    bb_p = pd.read_parquet(marker_dir / "base__on__base_mk_slots.parquet")

    def m(df: pd.DataFrame, col: str) -> dict:
        return dict(zip(df["sha"], df[col].astype(float)))

    a_lp, b_lp, bb_lp = m(arm_p, "logp"), m(base_p, "logp"), m(bb_p, "logp")
    a_margin = {s: zm - ze for s, zm, ze in zip(arm_p["sha"], arm_p["z_marker"], arm_p["z_eos"])}
    b_margin = {s: zm - ze for s, zm, ze in zip(base_p["sha"], base_p["z_marker"], base_p["z_eos"])}
    tab["dv_dlogp"] = tab["sha"].map(
        lambda s: a_lp[s] - b_lp[s] if s in a_lp and s in b_lp else np.nan
    )
    tab["dv_level_logp"] = tab["sha"].map(a_lp)
    tab["dv_eos_margin"] = tab["sha"].map(
        lambda s: a_margin[s] - b_margin[s] if s in a_margin and s in b_margin else np.nan
    )
    tab["dv_prob"] = tab["sha"].map(
        lambda s: math.exp(a_lp[s]) - math.exp(b_lp[s]) if s in a_lp and s in b_lp else np.nan
    )
    tab["p7"] = tab["sha"].map(bb_lp)  # base log P(marker) at the base slot
    raced = [
        c
        for c, col in CANDIDATE_COLS.items()
        if c == "p7" or (col in tab.columns and tab[col].notna().any())
    ]
    need = [CANDIDATE_COLS[c] for c in raced if c != "p7"] + [
        "p7",
        "dv_dlogp",
        "dv_level_logp",
        "dv_eos_margin",
        "dv_prob",
    ]
    frame = tab.dropna(subset=need).reset_index(drop=True)
    assert len(frame) >= MIN_ROWS, (aid, len(frame), "below realized-row floor")
    return {
        "arm": arm,
        "layer": layer,
        "frame": frame,
        "raced": raced,
        "dv_names": ["dv_dlogp", "dv_level_logp", "dv_eos_margin", "dv_prob"],
        "n_realized": int(len(frame)),
        "n_subset_rows": int(len(tab)),
    }


# ── per-arm battery ──────────────────────────────────────────────────────────


def _arm_seed(arm_id: str) -> int:
    """Deterministic per-arm seed — WITHIN-ARM permutation null ONLY.

    The bootstrap deliberately does NOT use this: the champion's across-arm
    per-draw median requires ONE shared draw stream across arms (plan §4 P3
    registered pairing; see `_family_shared_shas` — the r1 Major 3 fix).
    """
    import hashlib as _h

    return SEED * 1_000 + int(_h.sha256(arm_id.encode()).hexdigest()[:6], 16) % 100_000


def _family_shared_shas(asms: list[dict]) -> tuple[list[str], str]:
    """Sorted sha intersection across a family's realized frames + its digest.

    Plan §4 P3 registered convention: "arm-level pairing preserved by
    resampling contexts within every arm with the shared draw stream". Every
    arm's bootstrap resamples THIS pool in THIS order with the ONE module
    seed, so the per-draw index stream is IDENTICAL across arms and draw b
    names the same context multiset in every arm — the across-arm median is
    sha-paired. (The 12 arms share one judge subset, so per-arm sampling
    errors are positively correlated; independent per-arm streams would
    mis-state the median's variance — anti-conservative P(win)/CI.)
    """
    import hashlib as _h

    shared = sorted(set.intersection(*[set(a["frame"]["sha"]) for a in asms]))
    assert len(shared) >= MIN_ROWS, (len(shared), "family shared-sha pool below MIN_ROWS")
    return shared, _h.sha256("\n".join(shared).encode()).hexdigest()[:16]


def run_arm_battery(
    asm: dict, out_dir: Path, b_draws: int, n_perm: int, shared: tuple[list[str], str]
) -> dict:
    """Bootstrap + permutation battery for one arm; persists npz + JSON.

    The bootstrap resamples the FAMILY-SHARED sha pool (same order + the ONE
    module seed across arms -> identical per-draw index streams; sha-level
    champion pairing, plan §4 P3). Observed rho and the permutation null keep
    the arm's FULL realized row set (within-arm by design).
    """
    arm_id = asm["arm"]["arm_id"]
    frame, raced, dv_names = asm["frame"], asm["raced"], asm["dv_names"]
    shared_shas, shared_hash = shared
    arm_json = out_dir / f"arm_{arm_id}.json"
    regime = {
        "b_draws": b_draws,
        "n_perm": n_perm,
        "layer": asm["layer"],
        "raced": raced,
        "dv_names": dv_names,
        "n": asm["n_realized"],
        "n_shared": len(shared_shas),
        "shared_sha_hash": shared_hash,
    }
    if arm_json.exists():
        prior = json.loads(arm_json.read_text())
        if prior.get("regime") == regime and (out_dir / f"boot_{arm_id}.npz").exists():
            return prior
        logger.info("[p3] %s regime changed — recomputing", arm_id)
    x = np.column_stack(
        [
            frame["p7"].to_numpy(float) if c == "p7" else frame[CANDIDATE_COLS[c]].to_numpy(float)
            for c in raced
        ]
    )
    dvs = np.column_stack([frame[d].to_numpy(float) for d in dv_names])
    t0 = time.time()
    sha_pos = {s: i for i, s in enumerate(frame["sha"])}
    pos = np.array([sha_pos[s] for s in shared_shas], dtype=np.int64)
    boot, n_degen = bootstrap_battery(x[pos], dvs[pos], b_draws, SEED)
    perm = perm_null(x, dvs[:, 0], n_perm, _arm_seed(arm_id))
    obs = observed_rho(x, dvs)
    perm_max = perm.max(axis=1)  # SIGNED max — per-draw re-selection (registered)
    band = {
        "p975_max_selected": float(np.quantile(perm_max, 0.975)),
        "p95_max_selected": float(np.quantile(perm_max, 0.95)),
        "ceiling_abs_rho": 1.0,
        "n_perm": n_perm,
    }
    np.savez(
        out_dir / f"boot_{arm_id}.npz",
        rho=boot,
        candidates=np.array(raced),
        dv_names=np.array(dv_names),
        seed=SEED,  # ONE shared seed — champion_read asserts cross-arm equality
        n=asm["n_realized"],
        n_shared=len(shared_shas),
        shared_sha_hash=np.array(shared_hash),
    )
    np.savez(
        out_dir / f"perm_{arm_id}.npz",
        rho=perm,
        max_selected=perm_max,
        candidates=np.array(raced),
        dv="primary",
        seed=_arm_seed(arm_id) + 1,
    )
    # M-panel (mechanistic; separate, never headline)
    m_obs = {}
    for name, col in M_PANEL_COLS.items():
        if col in frame.columns and frame[col].notna().all():
            m_obs[name] = _spearman_np(
                _ranks_np(frame[col].to_numpy(float)),
                _ranks_np(dvs[:, 0]),
            )
    payload = {
        "meta": _meta(),
        "arm_id": arm_id,
        "kind": asm["arm"]["kind"],
        "beh_key": asm["arm"]["beh_key"],
        "regime": regime,
        "n_realized": asm["n_realized"],
        "n_subset_rows": asm["n_subset_rows"],
        "observed_rho": {
            dv: {c: float(obs[i, j]) for i, c in enumerate(raced)} for j, dv in enumerate(dv_names)
        },
        "m_panel_rho_primary": m_obs,
        "perm_band": band,
        "n_degenerate_series_draws": int(n_degen),
        "elapsed_s": round(time.time() - t0, 1),
    }
    _atomic_json(arm_json, payload)
    return payload


# ── champion (across-arm, selection-symmetric) ───────────────────────────────


def champion_read(arm_ids: list[str], out_dir: Path, dv_index: int, dv_label: str) -> dict:
    """Across-arm-median winner with per-draw re-selection over the shared panel."""
    boots, raced_sets = {}, []
    seeds, stream_hashes = set(), set()
    for a in arm_ids:
        z = np.load(out_dir / f"boot_{a}.npz", allow_pickle=False)
        boots[a] = (z["rho"], list(z["candidates"]))
        raced_sets.append(set(z["candidates"]))
        seeds.add(int(z["seed"]))
        stream_hashes.add(str(z["shared_sha_hash"]))
    # pairing invariant (plan §4 P3): every arm's boot rides the SAME seed and
    # the SAME shared-sha pool -> draw b is the same context multiset per arm.
    assert len(seeds) == 1 and len(stream_hashes) == 1, (
        seeds,
        stream_hashes,
        "champion pairing broken: arms carry different bootstrap draw streams",
    )
    panel = sorted(set.intersection(*raced_sets))
    stacks = []
    for a in arm_ids:
        rho, cands = boots[a]
        ix = [cands.index(c) for c in panel]
        stacks.append(rho[:, ix, dv_index])  # (B, Kp)
    cube = np.stack(stacks, axis=0)  # (A, B, Kp)
    med = np.median(cube, axis=0)  # (B, Kp) across-arm median per draw
    winner_ix = np.argmax(med, axis=1)  # SIGNED argmax (registered)
    p_win = {c: float(np.mean(winner_ix == i)) for i, c in enumerate(panel)}
    obs_med = {}
    for i, c in enumerate(panel):
        per_arm = []
        for a in arm_ids:
            arm_p = json.loads((out_dir / f"arm_{a}.json").read_text())
            dv_name = arm_p["regime"]["dv_names"][dv_index]
            per_arm.append(arm_p["observed_rho"][dv_name][c])
        obs_med[c] = float(np.median(per_arm))
    winner = max(obs_med, key=lambda c: obs_med[c])
    max_med = med.max(axis=1)  # per-draw max (selection rides the draw)
    sel_ci = [float(np.quantile(max_med, q)) for q in CI_QS]
    frz_ci = [float(np.quantile(med[:, panel.index(winner)], q)) for q in CI_QS]
    # registered lattice (plan §3); dethrone arm-count threshold scales as
    # ceil(0.75 * n_arms) — exactly 9 at the registered 12 content arms.
    dethrone_min = math.ceil(0.75 * len(arm_ids))
    verdict = "no-resolved-champion"
    if winner == "p7" and p_win.get("p7", 0.0) >= 0.5:
        verdict = "P7-retains-champion"
    elif winner != "p7" and p_win.get(winner, 0.0) >= 0.5:
        n_beats = 0
        for a in arm_ids:
            arm_p = json.loads((out_dir / f"arm_{a}.json").read_text())
            dv_name = arm_p["regime"]["dv_names"][dv_index]
            rho_map = arm_p["observed_rho"][dv_name]
            if rho_map.get(winner) is not None and rho_map.get("p7") is not None:
                n_beats += int(rho_map[winner] - rho_map["p7"] > 0)
        if n_beats >= dethrone_min:
            verdict = f"geometry-dethrones ({winner}; beats P7 in {n_beats}/{len(arm_ids)})"
    # band-vs-ceiling: champion-vs-P7 contrast conditional ceiling per arm
    p7_obs = []
    for a in arm_ids:
        arm_p = json.loads((out_dir / f"arm_{a}.json").read_text())
        dv_name = arm_p["regime"]["dv_names"][dv_index]
        p7_obs.append(arm_p["observed_rho"][dv_name].get("p7"))
    contrast_ceiling = [1.0 - v for v in p7_obs if v is not None]
    return {
        "dv": dv_label,
        "panel_candidates": panel,
        "n_arms": len(arm_ids),
        "arm_ids": arm_ids,
        "across_arm_median_observed": obs_med,
        "winner_observed": winner,
        "p_win": p_win,
        "selection_inherited_ci_max_median": sel_ci,
        "frozen_ci_winner_median (labeled: frozen-at-winner)": frz_ci,
        "verdict": verdict,
        "dethrone_min_arms": dethrone_min,
        "champion_vs_p7_conditional_ceiling_interval": (
            [float(min(contrast_ceiling)), float(max(contrast_ceiling))]
            if contrast_ceiling
            else None
        ),
        "note_correlated_arms": "arms share one judge subset — never narrated as "
        "independent confirmations (registered line (7))",
    }


# ── mediation ────────────────────────────────────────────────────────────────


def mediation_arm(frame, raced: list[str]) -> dict:
    """Rank partials + structural read + commonality for one content arm."""
    cols = {"dv": "dv_level", "p1": "p1_tc", "p2": "p2_tc", "p7": None}
    ranks = {
        "dv": _ranks_np(frame["dv_level"].to_numpy(float)),
        "p1": _ranks_np(frame["p1_tc"].to_numpy(float)),
        "p2": _ranks_np(frame["p2_tc"].to_numpy(float)),
        "p7": _ranks_np(frame["p7"].to_numpy(float)),
    }
    del cols
    for extra in ("p3a", "p3b"):
        col = CANDIDATE_COLS[extra]
        if col in frame.columns and frame[col].notna().all():
            ranks[extra] = _ranks_np(frame[col].to_numpy(float))
    out = {
        "r_p1_given_p7": _partial_spearman(ranks, "dv", "p1", ["p7"]),
        "r_p2_given_p7": _partial_spearman(ranks, "dv", "p2", ["p7"]),
        "r_p1_given_p2_p7": _partial_spearman(ranks, "dv", "p1", ["p2", "p7"]),
        "r_p2_given_p1_p7": _partial_spearman(ranks, "dv", "p2", ["p1", "p7"]),
        "rank_pearson_p1_p2": float(np.corrcoef(ranks["p1"], ranks["p2"])[0, 1]),
    }
    out["partials_unstable_flag"] = bool(abs(out["rank_pearson_p1_p2"]) > 0.95)
    for p3 in ("p3a", "p3b"):
        if p3 in ranks:
            out[f"structural_{p3}"] = {
                f"rank_agreement_rho({p3},p2)": _spearman_np(ranks[p3], ranks["p2"]),
                f"rank_agreement_rho({p3},p1)": _spearman_np(ranks[p3], ranks["p1"]),
                f"r_p1_given_{p3}_p7": _partial_spearman(ranks, "dv", "p1", [p3, "p7"]),
                f"r_{p3}_given_p1_p7": _partial_spearman(ranks, "dv", p3, ["p1", "p7"]),
            }
    # commonality decomposition of rank-R2 over {p1, p2, p7}
    r2 = {
        "1": _rank_r2(ranks, "dv", ["p1"]),
        "2": _rank_r2(ranks, "dv", ["p2"]),
        "7": _rank_r2(ranks, "dv", ["p7"]),
        "12": _rank_r2(ranks, "dv", ["p1", "p2"]),
        "17": _rank_r2(ranks, "dv", ["p1", "p7"]),
        "27": _rank_r2(ranks, "dv", ["p2", "p7"]),
        "127": _rank_r2(ranks, "dv", ["p1", "p2", "p7"]),
    }
    out["commonality"] = {
        "unique_p1": r2["127"] - r2["27"],
        "unique_p2": r2["127"] - r2["17"],
        "unique_p7": r2["127"] - r2["12"],
        "common_p1_p2": r2["17"] + r2["27"] - r2["7"] - r2["127"],
        "common_p1_p7": r2["12"] + r2["27"] - r2["2"] - r2["127"],
        "common_p2_p7": r2["12"] + r2["17"] - r2["1"] - r2["127"],
        "common_p1_p2_p7": r2["127"] - r2["12"] - r2["17"] - r2["27"] + r2["1"] + r2["2"] + r2["7"],
        "total_r2": r2["127"],
    }
    # even/odd half-anchor robustness (disjoint halves; registered line (4))
    halves = {}
    for tag, (c1, c2) in {
        "p1_even__p2_odd": ("p1_tc_even", "p2_tc_odd"),
        "p1_odd__p2_even": ("p1_tc_odd", "p2_tc_even"),
    }.items():
        if c1 in frame.columns and c2 in frame.columns:
            r = dict(ranks)
            r["p1"] = _ranks_np(frame[c1].to_numpy(float))
            r["p2"] = _ranks_np(frame[c2].to_numpy(float))
            halves[tag] = {
                "r_p1_given_p2_p7": _partial_spearman(r, "dv", "p1", ["p2", "p7"]),
                "r_p2_given_p1_p7": _partial_spearman(r, "dv", "p2", ["p1", "p7"]),
                "r_p1_given_p7": _partial_spearman(r, "dv", "p1", ["p7"]),
                "r_p2_given_p7": _partial_spearman(r, "dv", "p2", ["p7"]),
            }
    out["half_anchor_recount"] = halves
    return out


def mediation_lattice(per_arm: dict[str, dict]) -> dict:
    """Across-arm-median mediation lattice verdict (plan §3, disjoint set)."""

    def med(key: str) -> float | None:
        vals = [v[key] for v in per_arm.values() if v.get(key) is not None]
        return float(np.median(vals)) if vals else None

    r1_7, r2_7 = med("r_p1_given_p7"), med("r_p2_given_p7")
    r1_27, r2_17 = med("r_p1_given_p2_p7"), med("r_p2_given_p1_p7")
    verdict = "both-independent"
    if None not in (r1_7, r2_7, r1_27, r2_17):
        p1_cut = r1_27 < 0.5 * r1_7
        p2_cut = r2_17 < 0.5 * r2_7
        if p1_cut and not p2_cut:
            verdict = "answer-mediated"
        elif p2_cut and not p1_cut:
            verdict = "context-primary"
        elif p1_cut and p2_cut:
            verdict = "shared-inseparable"
    return {
        "median_r_p1_given_p7": r1_7,
        "median_r_p2_given_p7": r2_7,
        "median_r_p1_given_p2_p7": r1_27,
        "median_r_p2_given_p1_p7": r2_17,
        "verdict": verdict,
        "provisional_note": "verdict provisional until it agrees with the P3 structural "
        "read AND survives the disjoint-half anchor recount (registered line (4))",
    }


# ── fitted combination predictor (secondary) ─────────────────────────────────


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    k = x.shape[1]
    return np.linalg.solve(x.T @ x + lam * np.eye(k), x.T @ y)


def combination_predictor(assemblies: list[dict], out_dir: Path) -> dict:
    """Within-arm rank+z ridge over stacked content arms; group-level folds."""
    lam_grid = [10.0**e for e in range(-3, 4)]
    per_arm = {}
    feats = sorted(set.intersection(*[set(a["raced"]) for a in assemblies]))
    for a in assemblies:
        f = a["frame"]
        x = np.column_stack(
            [
                _ranks_np((f["p7"] if c == "p7" else f[CANDIDATE_COLS[c]]).to_numpy(float))
                for c in feats
            ]
        )
        y = _ranks_np(f["dv_level"].to_numpy(float))
        x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)
        y = (y - y.mean()) / max(y.std(), 1e-12)
        per_arm[a["arm"]["arm_id"]] = {
            "x": x,
            "y": y,
            "beh": a["arm"]["beh_key"],
            "corpus": f["corpus"].to_numpy(str),
        }

    def eval_fold(train_ids: list[str], test_id: str) -> float | None:
        if not train_ids:
            return None
        # inner λ selection: leave-one-TRAIN-arm-out mean Spearman
        best_lam, best = lam_grid[0], -np.inf
        for lam in lam_grid:
            scores = []
            for v in train_ids:
                tr = [t for t in train_ids if t != v]
                if not tr:
                    continue
                xt = np.vstack([per_arm[t]["x"] for t in tr])
                yt = np.concatenate([per_arm[t]["y"] for t in tr])
                w = _ridge_fit(xt, yt, lam)
                s = _spearman_np(per_arm[v]["x"] @ w, per_arm[v]["y"])
                if not math.isnan(s):
                    scores.append(s)
            m = np.mean(scores) if scores else -np.inf
            if m > best:
                best, best_lam = m, lam
        xt = np.vstack([per_arm[t]["x"] for t in train_ids])
        yt = np.concatenate([per_arm[t]["y"] for t in train_ids])
        w = _ridge_fit(xt, yt, best_lam)
        return _spearman_np(per_arm[test_id]["x"] @ w, per_arm[test_id]["y"])

    arm_ids = list(per_arm)
    loao = {h: eval_fold([a for a in arm_ids if a != h], h) for h in arm_ids}
    behs = sorted({v["beh"] for v in per_arm.values()})
    lobo = {}
    if len(behs) >= 2:
        for b in behs:
            train = [a for a in arm_ids if per_arm[a]["beh"] != b]
            for h in [a for a in arm_ids if per_arm[a]["beh"] == b]:
                lobo[h] = eval_fold(train, h)
    else:
        logger.info("[p3] LOBO skipped: <2 behaviors present (designed skip)")
    corpus_transfer = {}
    for src, dst in (("lmsys", "wildchat"), ("wildchat", "lmsys")):
        xt = np.vstack([v["x"][v["corpus"] == src] for v in per_arm.values()])
        yt = np.concatenate([v["y"][v["corpus"] == src] for v in per_arm.values()])
        if len(yt) < MIN_ROWS:
            continue
        w = _ridge_fit(xt, yt, 1.0)
        corpus_transfer[f"{src}->{dst}"] = {
            a: _spearman_np(v["x"][v["corpus"] == dst] @ w, v["y"][v["corpus"] == dst])
            for a, v in per_arm.items()
            if (v["corpus"] == dst).sum() >= 10
        }
    payload = {
        "meta": _meta(),
        "features": feats,
        "headline_fold": "leave-one-ARM-out (group-level n = n_arms)",
        "loao_spearman_per_held_arm": loao,
        "lobo_spearman_per_held_arm": lobo or "skipped (<2 behaviors)",
        "corpus_transfer": corpus_transfer,
    }
    _atomic_json(out_dir / "combination.json", payload)
    return payload


# ── robustness lines (registered §6 (1)-(8)) ─────────────────────────────────


def robustness_lines(
    assemblies: list[dict],
    out_dir: Path,
    champion_level: dict,
    champion_change: dict,
    inputs_dir: Path | None,
) -> dict:
    lines: dict = {"meta": _meta()}
    # (1) change-companion champion + text-divergence-restricted rho(P7, DV)
    diverge = {}
    for a in assemblies:
        arm_id = a["arm"]["arm_id"]
        f = a["frame"]
        entry: dict = {"note": "text divergence unavailable"}
        if inputs_dir is not None:
            try:
                arm_rows = J.load_judge_inputs(inputs_dir, arm_id)
                base_rows = J.load_judge_inputs(inputs_dir, "base_content")
                mask = f["sha"].map(
                    lambda s: (
                        s in arm_rows
                        and s in base_rows
                        and arm_rows[s]["response_text"] != base_rows[s]["response_text"]
                    )
                )
                sub = f[mask]
                entry = {
                    "n_text_diverging": int(len(sub)),
                    "rho_p7_dv_level_diverging": _spearman_np(
                        sub["p7"].to_numpy(float), sub["dv_level"].to_numpy(float)
                    )
                    if len(sub) >= 10
                    else None,
                }
            except AssertionError as e:  # inputs not staged for this unit
                entry = {"note": f"judge_inputs unavailable: {e}"}
        diverge[arm_id] = entry
    lines["line1"] = {
        "champion_on_change_companion": champion_change,
        "rho_p7_on_text_diverging_contexts": diverge,
    }
    # (2) reliability ceilings
    ceilings = {}
    for a in assemblies:
        f = a["frame"]
        ent: dict = {}
        for cand, (ce, co) in {
            "p1": ("p1_tc_even", "p1_tc_odd"),
            "p2": ("p2_tc_even", "p2_tc_odd"),
            "p9": ("p9_k16_even", "p9_k16_odd"),  # persisted by the r2 P1e amendment
        }.items():
            if ce in f.columns and co in f.columns and f[ce].notna().all():
                r = _spearman_np(f[ce].to_numpy(float), f[co].to_numpy(float))
                rel = 2 * r / (1 + r) if (1 + r) > 1e-9 else None
                ent[cand] = {"split_half_r": r, "sb_rel": rel}
            elif cand == "p9":
                ent["p9"] = {
                    "sb_rel": None,
                    "note": "p9_k16_{even,odd} columns absent (pre-r2 P1e table) — "
                    "ceiling not computable post-hoc",
                }
        ceilings[a["arm"]["arm_id"]] = ent
    lines["line2"] = {
        "anchor_split_half_ceilings": ceilings,
        "judge_split_half": "per-unit split_half blocks in arm_scores_*.json "
        "(rel and r_bar; ceiling for rho(c, DV) = sqrt(rel_c * rel_dv))",
    }
    # (3) length-partialled rho for the top-3 observed candidates
    length_reads = {}
    if inputs_dir is not None:
        try:
            base_rows = J.load_judge_inputs(inputs_dir, "base_content")
            top3 = sorted(
                champion_level["across_arm_median_observed"],
                key=lambda c: champion_level["across_arm_median_observed"][c],
                reverse=True,
            )[:3]
            for a in assemblies:
                f = a["frame"]
                ln = (
                    f["sha"]
                    .map(lambda s: len(base_rows[s]["prompt"]) if s in base_rows else np.nan)
                    .to_numpy(float)
                )
                ok = ~np.isnan(ln)
                ent = {}
                for c in top3:
                    xv = (f["p7"] if c == "p7" else f[CANDIDATE_COLS[c]]).to_numpy(float)[ok]
                    ranks = {
                        "dv": _ranks_np(f["dv_level"].to_numpy(float)[ok]),
                        "x": _ranks_np(xv),
                        "len": _ranks_np(ln[ok]),
                    }
                    quintile = np.digitize(ln[ok], np.quantile(ln[ok], [0.2, 0.4, 0.6, 0.8]))
                    per_q = []
                    for q in range(5):
                        m = quintile == q
                        if m.sum() >= 10:
                            per_q.append(_spearman_np(ranks["x"][m], ranks["dv"][m]))
                    ent[c] = {
                        "partial_given_length": _partial_spearman(ranks, "dv", "x", ["len"]),
                        "within_quintile_rho": per_q,
                    }
                length_reads[a["arm"]["arm_id"]] = ent
        except AssertionError as e:
            length_reads = {"note": f"judge_inputs unavailable: {e}"}
    lines["line3"] = length_reads
    # (5), (7), (8) are prose/verdict notes persisted for the analyzer
    lines["line5_m4_note"] = (
        "M4 (matched-text delta-answer similarity) retains a residual text-borne "
        "channel: the weight delta interacts with behavior-laden completion text "
        "even in matched-text form — named in the M-panel figure prose"
    )
    lines["line7_dethrone_note"] = champion_level["note_correlated_arms"]
    lines["line8_split_half_convention"] = (
        "criterion C split-half at N=3: mean pairwise inter-draw Spearman, "
        "Spearman-Brown to N (unequal halves handled pairwise-complete)"
    )
    _atomic_json(out_dir / "robustness.json", lines)
    return lines


def validation_read(assemblies: list[dict], p1_root: Path, out_dir: Path) -> None:
    """Line (6) + §6 graded-vs-reference: rho(TF-margin, {graded, P7, P1, P2})."""
    arm_id = "syc-pers-po-lr1e5-s42"
    margin_arm = p1_root / "validation" / "tf_margin_arm.jsonl"
    asm = next((a for a in assemblies if a["arm"]["arm_id"] == arm_id), None)
    if not margin_arm.exists() or asm is None:
        _atomic_json(
            out_dir / "validation_read.json",
            {
                "meta": _meta(),
                "status": "skipped",
                "note": "tf_margin_arm.jsonl or the syc-po arm unavailable — the "
                "stated §12.6 descope carries 'graded DV reference-validated only "
                "via split-half' as the wider caveat",
            },
        )
        return
    margins = {r["sha"]: float(r["margin"]) for r in J._read_jsonl_rows(margin_arm)}
    f = asm["frame"]
    m = f["sha"].map(margins).to_numpy(float)
    ok = ~np.isnan(m)
    n = int(ok.sum())
    payload = {
        "meta": _meta(),
        "arm_id": arm_id,
        "n_overlap_contexts": n,
        "rho_margin_graded": _spearman_np(m[ok], f["dv_level"].to_numpy(float)[ok])
        if n >= 10
        else None,
        "rho_margin_p7": _spearman_np(m[ok], f["p7"].to_numpy(float)[ok]) if n >= 10 else None,
        "rho_margin_p1": _spearman_np(m[ok], f["p1_tc"].to_numpy(float)[ok]) if n >= 10 else None,
        "rho_margin_p2": _spearman_np(m[ok], f["p2_tc"].to_numpy(float)[ok]) if n >= 10 else None,
        "precedent": "#722 cell-grain rho(margin, rate) = +0.40 (sycophancy)",
    }
    _atomic_json(out_dir / "validation_read.json", payload)


def exploration_grid(
    arms: list[dict], tables_dir: Path, judge_dir: Path, marker_dir: Path, out_dir: Path
) -> None:
    """Plain observed rho for all available layers x anchors (no bootstrap)."""
    rows = []
    for arm in arms:
        for layer in (14, 19, 25):
            tab_path = tables_dir / f"{arm['arm_id']}_L{layer}.parquet"
            if not tab_path.exists():
                continue
            try:
                asm = (
                    assemble_content_arm(arm, tables_dir, judge_dir, layer=layer)
                    if arm["kind"] == "content"
                    else assemble_marker_arm(arm, tables_dir, marker_dir, layer=layer)
                )
            except (MissingRaceInput, FileNotFoundError) as e:  # missing inputs ONLY
                logger.info("[explore] %s L%d skipped: %s", arm["arm_id"], layer, e)
                continue
            f = asm["frame"]
            dv = f[asm["dv_names"][0]].to_numpy(float)
            for cand in asm["raced"]:
                col = CANDIDATE_COLS[cand]
                x = (f["p7"] if cand == "p7" else f[col]).to_numpy(float)
                rows.append(
                    {
                        "arm_id": arm["arm_id"],
                        "kind": arm["kind"],
                        "layer": layer,
                        "candidate": cand,
                        "anchor": "training_centroid",
                        "rho": _spearman_np(x, dv),
                    }
                )
            for cand, col in PS_COLS.items():
                if col in f.columns and f[col].notna().all():
                    rows.append(
                        {
                            "arm_id": arm["arm_id"],
                            "kind": arm["kind"],
                            "layer": layer,
                            "candidate": cand,
                            "anchor": "panel_source",
                            "rho": _spearman_np(f[col].to_numpy(float), dv),
                        }
                    )
    _atomic_json(out_dir / "exploration.json", {"meta": _meta(), "rows": rows})


# ── smoke-input builder (real tensors, real loader — no torch.randn) ─────────

SMOKE_CONTENT_ARM = "imp-pers-con-lr3e5-s42"
SMOKE_MARKER_ARM = "mk-pers-con-lr5e6-s42"
SMOKE_ROWS = 64
SMOKE_ANCHOR_ROWS = 32


def build_smoke_inputs(smoke_root: Path, config_dir: Path) -> None:
    """Stage ONE real #1768 cell + build a 64-row race input set from it.

    Every tensor is REAL (fp16 pooled stores at the pin, loaded via the REAL
    `issue1768_fit.load_corpus_cell`); anchors = means of 32 real rows
    (smoke-grade, labeled); the DV beyond the 2 really-judged P2-smoke rows and
    the marker four-floats are REAL-TENSOR FUNCTIONALS (labeled per slice) —
    P1-dependent inputs that cannot exist pre-dispatch. No synthetic RNG data.
    """
    import pandas as pd

    import issue1768_fit as F
    from explore_persona_space.orchestrate import hub

    _phase("p3_smoke_build")
    i1768 = REPO_ROOT / "data/issue_1900/hf_dl" / "issue1768_mapshift"
    base_unit = "base_content"
    needed = [
        (
            f"issue1768_mapshift/corpus_capture/{base_unit}/pooled.pt",
            i1768 / "corpus_capture" / base_unit / "pooled.pt",
        ),
        (
            f"issue1768_mapshift/corpus_capture/{SMOKE_CONTENT_ARM}/pooled.pt",
            i1768 / "corpus_capture" / SMOKE_CONTENT_ARM / "pooled.pt",
        ),
        (
            f"issue1768_mapshift/corpus_capture_tf/{SMOKE_CONTENT_ARM}/pooled_tf.pt",
            i1768 / "corpus_capture_tf" / SMOKE_CONTENT_ARM / "pooled_tf.pt",
        ),
    ]
    for repo_path, local in needed:
        if not local.exists():
            logger.info("[smoke-build] staging %s", repo_path)
            hub.stage_hub_file(
                J._data_repo(), repo_path, local, repo_type="dataset", revision=CORPUS_PIN
            )
    sample_dst = i1768 / "inputs" / "corpus_sample.json"
    flat = REPO_ROOT / "data/issue_1900/hf_dl/corpus_sample.json"
    if not sample_dst.exists():
        if flat.exists():
            sample_dst.parent.mkdir(parents=True, exist_ok=True)
            sample_dst.write_bytes(flat.read_bytes())
        else:
            hub.stage_hub_file(
                J._data_repo(),
                "issue1768_mapshift/inputs/corpus_sample.json",
                sample_dst,
                repo_type="dataset",
                revision=CORPUS_PIN,
            )
    layer = 19
    cell = F.load_corpus_cell(SMOKE_CONTENT_ARM, layer, i1768)  # REAL loader, REAL cell
    n = SMOKE_ROWS
    c0, v0 = cell["C0"][:n], cell["V0"][:n]
    cp, vtf = cell["Cplus"][:n], cell["Vplus_tf"][:n]
    shas = cell["sha"][:n]
    corpus = cell["corpus"][:n]
    cbar, vbar = c0.mean(0), v0.mean(0)
    a_ctx = c0[:SMOKE_ANCHOR_ROWS].mean(0)  # smoke-grade anchor: mean of 32 REAL rows
    a_ans = v0[:SMOKE_ANCHOR_ROWS].mean(0)
    a_ctx_even = c0[0:SMOKE_ANCHOR_ROWS:2].mean(0)
    a_ctx_odd = c0[1:SMOKE_ANCHOR_ROWS:2].mean(0)
    a_ans_even = v0[0:SMOKE_ANCHOR_ROWS:2].mean(0)
    a_ans_odd = v0[1:SMOKE_ANCHOR_ROWS:2].mean(0)

    def ccos(rows: np.ndarray, anchor: np.ndarray, center: np.ndarray) -> np.ndarray:
        r = rows - center
        a = anchor - center
        return (r @ a) / (np.linalg.norm(r, axis=1) * np.linalg.norm(a) + 1e-12)

    # real mini-map M0_smoke: ridge c0 -> v0 on the 64 real rows (real fit)
    lam = 1.0
    w = np.linalg.solve(c0.T @ c0 + lam * np.eye(c0.shape[1]), c0.T @ v0)
    mpred = c0 @ w
    mbar = mpred.mean(0)
    sigma = np.cov(c0.T) * 0.9 + 0.1 * np.eye(c0.shape[1]) * np.var(c0)
    a_sig = np.linalg.solve(sigma, a_ctx - cbar)
    d_beh = v0[: n // 2].mean(0) - v0[n // 2 :].mean(0)  # real diff-of-real-row-groups
    d_beh /= np.linalg.norm(d_beh) + 1e-12
    rcn = c0[:SMOKE_ANCHOR_ROWS] - cbar
    rcn /= np.linalg.norm(rcn, axis=1, keepdims=True) + 1e-12
    c0n = (c0 - cbar) / (np.linalg.norm(c0 - cbar, axis=1, keepdims=True) + 1e-12)
    sims_raw = c0n @ rcn.T
    sims = np.sort(sims_raw, axis=1)[:, ::-1]

    def table_frame() -> pd.DataFrame:
        df = pd.DataFrame({"sha": shas, "corpus": corpus, "in_judge_subset": True})
        df["p1_tc"] = ccos(c0, a_ctx, cbar)
        df["p2_tc"] = ccos(v0, a_ans, vbar)
        df["p1_tc_even"] = ccos(c0, a_ctx_even, cbar)
        df["p1_tc_odd"] = ccos(c0, a_ctx_odd, cbar)
        df["p2_tc_even"] = ccos(v0, a_ans_even, vbar)
        df["p2_tc_odd"] = ccos(v0, a_ans_odd, vbar)
        df["p1_ps"] = np.nan
        df["p2_ps"] = np.nan
        df["p3a_tc"] = ccos(mpred, mpred[:SMOKE_ANCHOR_ROWS].mean(0), mbar)
        df["p3b_tc"] = ccos(mpred, a_ans, mbar)
        df["p4_tc"] = (c0 @ a_sig) / (float((a_ctx - cbar) @ a_sig) + 1e-12)
        df["p5"] = (v0 - vbar) @ d_beh
        df["p6"] = (mpred - mbar) @ d_beh
        df["p8a"] = np.linalg.norm(mpred - mbar, axis=1)
        df["p8b"] = ccos(mpred - mbar, d_beh, np.zeros_like(d_beh))
        df["p9_k16"] = sims[:, :16].mean(axis=1)
        for h, sl in (("even", slice(0, None, 2)), ("odd", slice(1, None, 2))):
            sims_h = np.sort(sims_raw[:, sl], axis=1)[:, ::-1]
            df[f"p9_k16_{h}"] = sims_h[:, : min(16, sims_h.shape[1])].mean(axis=1)
        df["m1_tc"] = ccos(cp, cp[:SMOKE_ANCHOR_ROWS].mean(0), cp.mean(0))
        df["m2_tc"] = np.nan
        df["m3"] = ccos(cp - c0, (cp - c0).mean(0), np.zeros_like(cbar))
        df["m4"] = ccos(vtf - v0, (vtf - v0).mean(0), np.zeros_like(vbar))
        df["m5_tc"] = np.nan
        df["m6"] = np.linalg.norm(vtf - v0, axis=1)
        return df

    tables = smoke_root / "tables"
    marker_dir = smoke_root / "marker_tf"
    judge_dir = smoke_root / "judge"
    for d in (tables, marker_dir, judge_dir):
        d.mkdir(parents=True, exist_ok=True)
    df = table_frame()
    df.to_parquet(tables / f"{SMOKE_CONTENT_ARM}_L19.parquet", index=False)
    df.to_parquet(tables / f"{SMOKE_MARKER_ARM}_L25.parquet", index=False)

    # marker four-floats: REAL-tensor functionals (vplus vs v0 = genuinely real
    # trained-vs-base representations; labeled — no real logits exist pre-P1).
    vplus = cell["Vplus"][:n]
    d2 = c0[: n // 2].mean(0) - c0[n // 2 :].mean(0)
    d2 /= np.linalg.norm(d2) + 1e-12
    for tag, mat in (
        (f"{SMOKE_MARKER_ARM}__on__{SMOKE_MARKER_ARM}", vplus),
        (f"base__on__{SMOKE_MARKER_ARM}", v0),
        ("base__on__base_mk", v0),
    ):
        z_m = mat @ d_beh
        z_e = mat @ d2
        log_z = np.log(np.linalg.norm(mat, axis=1) + 1.0)
        pd.DataFrame(
            {
                "sha": shas,
                "logp": z_m - log_z,
                "z_marker": z_m,
                "z_eos": z_e,
                "logZ": log_z,
                "argmax_id": 0,
                "model_tag": tag.split("__on__")[0],
                "text_unit": tag.split("__on__")[1],
            }
        ).to_parquet(marker_dir / f"{tag}_slots.parquet", index=False)

    # judge scores: the 2 REALLY-judged P2-smoke rows spliced onto a labeled
    # real-tensor-functional DV (affine map of the real answer-anchor
    # projection onto 0..100) for the remaining rows.
    p2smoke = REPO_ROOT / "data/issue_1900/judge_smoke/out/arm_scores_smoke_base_imp.json"
    real_scores: dict[str, float] = {}
    if p2smoke.exists():
        real_scores = {
            r["sha"]: r["score_mean"]
            for r in json.loads(p2smoke.read_text())["rows"]
            if r["score_mean"] is not None
        }
    proj = ccos(v0, a_ans, vbar)
    lo, hi = float(proj.min()), float(proj.max())
    dv100 = 100.0 * (proj - lo) / (hi - lo + 1e-12)
    lvl_arm = np.clip(dv100 + 5.0, 0, 100)  # arm level: real functional, shifted

    def scores_payload(unit: str, values: np.ndarray, splice_real: bool) -> dict:
        rows = []
        for i, sha in enumerate(shas):
            mean = float(values[i])
            provenance = "real-tensor-functional (smoke stand-in DV)"
            if splice_real and sha in real_scores:
                mean = float(real_scores[sha])
                provenance = "REAL Sonnet judge score (P2 smoke)"
            rows.append(
                {
                    "sha": sha,
                    "score_mean": mean,
                    "kept_draw_scores": [mean],
                    "n_kept_draws": 1,
                    "binary_rate": float(mean >= 50),
                    "n_transport_lost": 0,
                    "provenance": provenance,
                }
            )
        return {
            "meta": {**_meta(), "unit": unit, "SMOKE": True},
            "judge": {"n_draws": 1, "max_tokens": 400, "note": "smoke stand-in"},
            "n_items": len(rows),
            "n_scored_items": len(rows),
            "n_total_draws": len(rows),
            "n_content_dropped_draws": 0,
            "n_refusal_draws": 0,
            "n_transport_lost_draws": 0,
            "split_half": {"rel": None, "r_bar": None, "n_complete_items": 0},
            "rows": rows,
        }

    _atomic_json(
        judge_dir / f"arm_scores_{SMOKE_CONTENT_ARM}.json",
        scores_payload(SMOKE_CONTENT_ARM, lvl_arm, splice_real=False),
    )
    _atomic_json(
        judge_dir / "arm_scores_base_imp.json",
        scores_payload("base_imp", dv100, splice_real=True),
    )
    smoke_arms = [
        {
            "arm_id": SMOKE_CONTENT_ARM,
            "kind": "content",
            "beh_key": "imp",
            "primary_layer": 19,
        },
        {
            "arm_id": SMOKE_MARKER_ARM,
            "kind": "marker",
            "beh_key": "mk",
            "primary_layer": 25,
        },
    ]
    _atomic_json(
        smoke_root / "arms_smoke.json",
        {
            "arms": smoke_arms,
            "provenance": {
                "stores": f"REAL #1768 pooled stores @ {CORPUS_PIN[:10]} via the REAL "
                "issue1768_fit.load_corpus_cell loader (one content cell)",
                "anchors": f"smoke-grade: mean of {SMOKE_ANCHOR_ROWS} REAL store rows "
                "(labeled; production anchors come from P1b delta_tf mixes)",
                "judge_dv": "2 REAL Sonnet-judged rows (P2 smoke) + real-tensor-"
                "functional stand-in for the rest (labeled per row)",
                "marker_four_floats": "real-tensor functionals over REAL Vplus vs V0 "
                "trees (no real logits exist pre-P1; labeled)",
                "marker_table": "content-cell candidate columns reused for the marker "
                "smoke arm (staging the real marker cell would double the download)",
            },
            **_meta(),
        },
    )
    print(
        f"[p3-smoke-build] done: n={n} rows, tables={len(list(tables.glob('*.parquet')))}, "
        f"marker={len(list(marker_dir.glob('*.parquet')))}, real_judged={len(real_scores)}",
        flush=True,
    )


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config-dir", type=Path, default=REPO_ROOT / "data/issue_1900/config")
    ap.add_argument("--p1-root", type=Path, default=REPO_ROOT / "data/issue_1900/out")
    ap.add_argument("--judge-dir", type=Path, default=REPO_ROOT / "eval_results/issue_1900/judge")
    ap.add_argument("--inputs-dir", type=Path, default=REPO_ROOT / "data/issue_1900/judge_inputs")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "eval_results/issue_1900/race")
    ap.add_argument("--b-draws", type=int, default=B_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--arms", default=None, help="comma-separated arm_id subset")
    ap.add_argument("--stage-from-hf", action="store_true", help="stage P1 files from HF")
    ap.add_argument("--smoke", action="store_true", help="run against the smoke root")
    ap.add_argument("--build-smoke-inputs", action="store_true")
    ap.add_argument("--smoke-root", type=Path, default=REPO_ROOT / "data/issue_1900/race_smoke")
    args = ap.parse_args()

    if args.build_smoke_inputs:
        build_smoke_inputs(args.smoke_root, args.config_dir)
        sys.stdout.flush()
        sys.exit(0)

    if args.smoke:
        tables_dir = args.smoke_root / "tables"
        marker_dir = args.smoke_root / "marker_tf"
        judge_dir = args.smoke_root / "judge"
        out_dir = args.smoke_root / "race_out"
        inputs_dir = None  # judge_inputs-derived lines report their designed skip
        arms = json.loads((args.smoke_root / "arms_smoke.json").read_text())["arms"]
        p1_root = args.smoke_root
    else:
        tables_dir = args.p1_root / "predictor_tables"
        marker_dir = args.p1_root / "marker_tf"
        judge_dir = args.judge_dir
        out_dir = args.out_dir
        inputs_dir = args.inputs_dir
        arms = J.load_arms(args.config_dir)
        p1_root = args.p1_root
    if args.arms:
        keep = {s.strip() for s in args.arms.split(",")}
        arms = [a for a in arms if a["arm_id"] in keep]
        assert arms, f"--arms matched nothing: {args.arms}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage_from_hf and not args.smoke:
        rel = []
        for a in arms:
            for layer in (14, 19, 25):
                rel.append(f"predictor_tables/{a['arm_id']}_L{layer}.parquet")
            if a["kind"] == "marker":
                rel += [
                    f"marker_tf/{a['arm_id']}__on__{a['arm_id']}_slots.parquet",
                    f"marker_tf/base__on__{a['arm_id']}_slots.parquet",
                ]
        rel += ["marker_tf/base__on__base_mk_slots.parquet", "validation/tf_margin_arm.jsonl"]
        _stage_from_hf(args.p1_root, rel)

    _phase("p3_race")
    content = [a for a in arms if a["kind"] == "content"]
    marker = [a for a in arms if a["kind"] == "marker"]
    content_asm, marker_asm = [], []
    all_units = content + marker
    # assemble ALL arms first: the per-FAMILY shared sha pool (champion pairing,
    # plan §4 P3) is the intersection of realized frames, known only after
    # assembly; batteries then resample that pool with the ONE shared stream.
    for arm in all_units:
        asm = (
            assemble_content_arm(arm, tables_dir, judge_dir)
            if arm["kind"] == "content"
            else assemble_marker_arm(arm, tables_dir, marker_dir)
        )
        (content_asm if arm["kind"] == "content" else marker_asm).append(asm)
    shared_by_kind = {}
    if content_asm:
        shared_by_kind["content"] = _family_shared_shas(content_asm)
    if marker_asm:
        shared_by_kind["marker"] = _family_shared_shas(marker_asm)
    for kind, sh in shared_by_kind.items():
        logger.info("[p3] %s family shared-sha pool: %d rows (hash %s)", kind, len(sh[0]), sh[1])
    for k, asm in enumerate(content_asm + marker_asm):
        t0 = time.time()
        run_arm_battery(asm, out_dir, args.b_draws, args.n_perm, shared_by_kind[asm["arm"]["kind"]])
        print(
            f"[p3] unit {k + 1}/{len(all_units)} {asm['arm']['arm_id']} n={asm['n_realized']} "
            f"n_shared={len(shared_by_kind[asm['arm']['kind']][0])} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    _phase("p3_champion")
    if content_asm:
        ids = [a["arm"]["arm_id"] for a in content_asm]
        champ_level = champion_read(ids, out_dir, 0, "content graded LEVEL (primary)")
        champ_change = champion_read(
            ids, out_dir, 1, "content graded CHANGE (companion; registered line (1))"
        )
        _atomic_json(
            out_dir / "champion_content.json",
            {"meta": _meta(), "primary": champ_level, "change_companion": champ_change},
        )
        med = {}
        for a in content_asm:
            med[a["arm"]["arm_id"]] = mediation_arm(a["frame"], a["raced"])
        _atomic_json(
            out_dir / "mediation.json",
            {"meta": _meta(), "per_arm": med, "lattice": mediation_lattice(med)},
        )
        if len(content_asm) >= 2:
            combination_predictor(content_asm, out_dir)
        else:
            logger.info("[p3] combination predictor skipped: <2 content arms")
        robustness_lines(content_asm, out_dir, champ_level, champ_change, inputs_dir)
        validation_read(content_asm, p1_root, out_dir)
    if marker_asm:
        ids = [a["arm"]["arm_id"] for a in marker_asm]
        champ_mk = champion_read(ids, out_dir, 0, "marker delta logP (primary)")
        champ_mk_lvl = champion_read(
            ids,
            out_dir,
            1,
            "marker trained-side LEVEL logP (companion — P7 base-side coupling read, "
            "registered Stats-MF column)",
        )
        _atomic_json(
            out_dir / "champion_marker.json",
            {
                "meta": _meta(),
                "primary": champ_mk,
                "level_companion": champ_mk_lvl,
                "note": "replication panel — three-space DV; EOS-margin + probability "
                "columns ride the boot npz dv axis (secondary/sanity)",
            },
        )
    _phase("p3_exploration")
    exploration_grid(arms, tables_dir, judge_dir, marker_dir, out_dir)
    _phase("done")
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
