"""Mapped vs real vs text-embedding minimal-pair separation (issue #2215).

User-chat inline free-analysis round (`separation_comparison`, 2026-08-24).
Question: does the frozen context->answer ridge map SEPARATE minimal pairs
more or less than (a) the realized answer vectors it predicts and (b) a
text-embedding of the rollout completion text — and on which pair types?

Design (dispatch note on #2215, same-day epm:progress):
    Per unordered minimal pair (A, B), cosine distance between the two
    sides' representations in three spaces —
      MAPPED  frozen ridge predictions f(v_C) at layer 19, context-end slot
              (arms: single-turn #779 map `779ce` primary, multi-turn #1738
              context-end map `1738ce` companion; prefix-end skipped).
      REAL    banked realized answer means v_A (tail-inclusive pooling,
              layer 19; parent battery streamed from the #2215 va2215 store,
              dbe battery read from the persisted predictions payload).
      TEXT    text embedding of each rollout's COMPLETION text only, one
              embedding per draw, L2-normalized, averaged over the banked
              draws per context. Default source (GPU override, user
              2026-08-24): banked Qwen3-Embedding-8B per-context means
              produced pod-side by `issue2215_sepcmp_qwen_embed.py`
              (space `text_qwen3_8b`); the original OpenAI
              `text-embedding-3-large` route stays available via
              `--text-space openai` (key was revoked at dispatch time).
    Common currency: per (cell, space), carrier yardstick = median cosine
    distance between same-value different-carrier members; reported
    quantity = separation ratio (pair-side distance / yardstick). Per-cell
    median ratios + carrier-clustered bootstrap 95% CIs (B=10,000, seed
    2215), the mapped-minus-real ratio contrast on paired draws, and
    per-pair Spearman correlations between spaces within each cell.
    Parent conflict fwd/rev twins re-pair the same contexts — the `_rev`
    pair-cells are dropped (forward only), stated in outputs.

Inputs (HF dataset repo `superkaiba1/explore-persona-space-data`):
    - parent bank + context vectors: `issue2162_ctxinfo/analysis_tensors/
      vc_bank/{bank.json,vc_bank.pt}` at the #2215 revision pin;
    - parent rollout texts: `issue2162_ctxinfo/raw_completions/anchors/`
      (16 jsonl shards, 14,040 rows);
    - parent tail answer store: `issue2215_reprshift/analysis_tensors/
      va2215/` (16 shards, streamed one at a time, deleted after use);
    - dbe bank / predictions / rollout texts under `issue2215_dbe/`
      (predictions_L19.pt carries per-arm predictions, per-pooling targets
      and the validity mask — the #2215 fp16 round-trip is bounded at
      0.00012 by `issue2215_dbe_perpair_recompute.py`);
    - ridge payloads: `issue779_monitoring/n1m_readout/weights/L19/ridge.pt`
      and `issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt`,
      applied via `issue779_ffc_n1m_fits.apply_map` (the exact #2215 path).

Outputs:
    - eval_results/issue_2215/separation_comparison/sepcmp.json
    - eval_results/issue_2215/separation_comparison/perpair.jsonl
    - figures/issue_2215/sepcmp_mapped_vs_real_scatter.{png,pdf,meta.json}
    - figures/issue_2215/sepcmp_pertype_spaces.{png,pdf,meta.json}
    - per-draw + per-context-mean embeddings (fp16 npz) uploaded to
      `issue2215_sepcmp/analysis_tensors/embeddings/` (reused when present
      so the API spend is never repeated).

No model fits anywhere in this round (no n<d exposure); no LLM judge.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.orchestrate.hub import (  # noqa: E402
    assert_hub_dir_filecounts,
    retry_transient,
)


sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue2215_analysis as ANA  # noqa: E402

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# Parent artifacts at the #2215 verified pin; dbe/va2215/map payloads at the
# data-repo main resolved when this round dispatched (recorded for repro).
REV_PARENT = "dc8108ab84f33695bbc769da0e6e8e2327f51eeb"
REV_MAIN = "a56358aa85a740c36c5214b012eb34e48433d42d"

VC_BANK_PREFIX = "issue2162_ctxinfo/analysis_tensors/vc_bank"
LAYER = 19
POOL = "tail"
SEED = 2215  # round seed (dispatch note); bootstrap + any tie-break draws
B_BOOT = 10_000
ARM_PATHS = {
    "779ce": "issue779_monitoring/n1m_readout/weights/L19/ridge.pt",
    "1738ce": "issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt",
}
SPACES = ("mapped_779ce", "mapped_1738ce", "real", "text")
CONTRAST_ARMS = ("mapped_779ce", "mapped_1738ce")

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
TOKEN_CAP = 8191  # the model's input cap; longer completions truncate here
EMBED_BATCH = 128
EMBED_REQ_TOKEN_BUDGET = 200_000
EMBED_CONCURRENCY = 8
EMBED_HF_PREFIX = "issue2215_sepcmp/analysis_tensors/embeddings"
# GPU override route (user, 2026-08-24): Qwen3-Embedding-8B means banked by
# scripts/issue2215_sepcmp_qwen_embed.py on pod-2215-sepcmp.
QWEN_HF_PREFIX = "issue2215_sepcmp/analysis_tensors/embeddings_qwen3_8b"
QWEN_SPACE = "text_qwen3_8b"

PARENT_JSONL = tuple(
    f"issue2162_ctxinfo/raw_completions/anchors/anchors_{batch}_w{w}.jsonl"
    for batch in ("gate", "rest")
    for w in range(8)
)
DBE_TYPES = (
    "code_vs_prose",
    "conversation_language",
    "conversation_topic",
    "refusal_request",
    "style_register",
    "user_doc_format",
    "user_role_identity",
    "user_sentiment",
)
DBE_JSONL = tuple(
    f"issue2215_dbe/raw_completions/anchors/anchors_dbe_w0_{t}.jsonl" for t in DBE_TYPES
)
VA2215_SHARDS = tuple(
    f"issue2215_reprshift/analysis_tensors/va2215/va2215_{batch}_w{w}.pt"
    for batch in ("gate", "rest")
    for w in range(8)
)
# Parent cells shown in the per-type figure (dispatch brief).
PARENT_FIGURE_CELLS = (
    "persona_prompted",
    "demo_persona",
    "persona_role_header",
    "fact_assistant_animal",
    "verbosity",
    "reasoning_style",
    "instr_format",
    "demo_format",
    "fact_user_name",
    "prior_topic",
)


def log(msg: str) -> None:
    print(f"[sepcmp] {msg}", flush=True)


def stage(path_in_repo: str, root: Path, revision: str) -> Path:
    """Download one repo file into ``root`` (skip when already staged)."""
    local = root / path_in_repo
    if local.exists():
        return local
    out = retry_transient(
        lambda: hf_hub_download(
            HF_DATA_REPO, path_in_repo, repo_type="dataset", revision=revision, local_dir=root
        ),
        what=f"hf_hub_download({path_in_repo})",
    )
    return Path(out)


# ── geometry helpers (pure numpy, vectorized) ─────────────────────────


def unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    assert (n > 0).all(), "zero-norm row — cannot take cosine distance"
    return x / n


def cos_dist_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.einsum("ij,ij->i", unit_rows(a), unit_rows(b))


def cos_dist_matrix(x: np.ndarray) -> np.ndarray:
    u = unit_rows(x)
    d = 1.0 - u @ u.T
    return np.clip(d, 0.0, 2.0)


# ── per-cell frame: complete (carrier x value) grid + pair grid ───────


class CellFrame:
    """Index structure for one cell: pair grid + same-value carrier grid."""

    def __init__(self, bank: dict, pt: ANA.PairTable, cv: ANA.CellView) -> None:
        self.cell = cv.cell
        self.cv = cv
        self.n_carr = len(cv.carriers)
        self.n_vals = len(cv.values)
        carr_index = {c: i for i, c in enumerate(cv.carriers)}
        val_index = {v: i for i, v in enumerate(cv.values)}
        # (carrier, value) -> local context index; grid completeness asserted.
        m = np.full((self.n_carr, self.n_vals), -1, dtype=np.int64)
        for loc, row in enumerate(cv.ctx_rows):
            ctx = bank["contexts"][pt.ids[row]]
            m[carr_index[ctx["carrier"]], val_index[ctx["value_id"]]] = loc
        assert (m >= 0).all(), (self.cell, "incomplete (carrier x value) grid")
        self.member = m
        # pair grid (n_carr, n_vps) -> local pair index (complete, from CellView)
        self.pair_at = cv.pair_at
        iu = np.triu_indices(self.n_carr, k=1)
        self.combo_i, self.combo_j = iu

    def pair_dists(self, x_local: np.ndarray) -> np.ndarray:
        """(n_pairs,) cosine distance between the two sides of every pair."""
        return cos_dist_rows(x_local[self.cv.a_loc], x_local[self.cv.b_loc])

    def yard_tensor(self, x_local: np.ndarray) -> np.ndarray:
        """(n_vals, n_carr, n_carr) same-value cross-carrier distance grids."""
        d = cos_dist_matrix(x_local)
        return np.stack(
            [d[np.ix_(self.member[:, v], self.member[:, v])] for v in range(self.n_vals)]
        )

    def yardstick(self, yard: np.ndarray) -> float:
        """Median same-value different-carrier distance (observed, full cell)."""
        return float(np.median(yard[:, self.combo_i, self.combo_j]))


def bootstrap_cell(
    frame: CellFrame,
    pair_d: dict[str, np.ndarray],
    yard: dict[str, np.ndarray],
    draws: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-space bootstrap ratio draws on SHARED carrier resamples.

    ``draws`` is (B, n_carr) carrier indices with replacement. Per draw and
    space: median pair distance over the resampled carriers' pairs divided
    by the median same-value distance among resampled-carrier combos with
    equal-carrier (self) combos masked out. A draw whose combos are all
    self-pairs falls back to the observed yardstick (probability ~n^(1-n)).
    """
    out: dict[str, np.ndarray] = {}
    ci = draws[:, frame.combo_i]  # (B, n_combos)
    cj = draws[:, frame.combo_j]
    self_mask = ci == cj
    for space, pd in pair_d.items():
        pd_grid = pd[frame.pair_at]  # (n_carr, n_vps)
        med_pair = np.median(pd_grid[draws].reshape(len(draws), -1), axis=1)
        # gather (B, n_combos, n_vals) same-value distances among drawn carriers
        g = yard[space][:, ci, cj].transpose(1, 2, 0).astype(np.float32)
        g[self_mask] = np.nan
        with np.errstate(all="ignore"):
            med_yard = np.nanmedian(g.reshape(len(draws), -1), axis=1)
        obs_yard = frame.yardstick(yard[space])
        med_yard = np.where(np.isnan(med_yard), obs_yard, med_yard)
        out[space] = med_pair / med_yard
    return out


def ci95(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


# ── text-embedding leg ────────────────────────────────────────────────


def read_rollout_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                rows.append(
                    {"context_id": r["context_id"], "draw": int(r["draw"]), "text": r["text"]}
                )
    return rows


def embed_battery(battery: str, rows: list[dict], emb_dir: Path) -> dict:
    """Per-draw `text-embedding-3-large` embeddings for one battery.

    Returns {"ids": (n,) context ids, "draws": (n,), "emb": (n, 3072) fp32
    L2-normalized, "n_truncated": int, "n_empty": int, "total_tokens": int}.
    Reuses the local npz cache, then the HF-banked copy, before spending.
    """
    npz_path = emb_dir / f"perdraw_{battery}.npz"
    if not npz_path.exists():
        try:  # banked copy from a prior run of this round
            retry_transient(
                lambda: hf_hub_download(
                    HF_DATA_REPO,
                    f"{EMBED_HF_PREFIX}/perdraw_{battery}.npz",
                    repo_type="dataset",
                    local_dir=emb_dir / "_hf",
                ),
                what=f"hf_hub_download(sepcmp perdraw_{battery})",
            )
            (emb_dir / "_hf" / EMBED_HF_PREFIX / f"perdraw_{battery}.npz").rename(npz_path)
            log(f"embeddings[{battery}]: reusing HF-banked per-draw file")
        except Exception:
            pass  # not banked yet -> compute below (the only fail-open branch)
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        return {
            "ids": [str(s) for s in z["context_ids"]],
            "draws": z["draws"],
            "emb": z["emb"].astype(np.float32),
            **meta,
        }

    import tiktoken
    from openai import OpenAI

    enc = tiktoken.get_encoding("cl100k_base")
    keep = [r for r in rows if r["text"].strip()]
    n_empty = len(rows) - len(keep)
    toks = [enc.encode(r["text"], disallowed_special=()) for r in keep]
    n_trunc = sum(1 for t in toks if len(t) > TOKEN_CAP)
    total_tokens = sum(min(len(t), TOKEN_CAP) for t in toks)
    inputs = [t[:TOKEN_CAP] for t in toks]  # token-id inputs: exact cap, no re-encode drift
    log(
        f"embeddings[{battery}]: {len(keep)} texts ({n_empty} empty skipped, "
        f"{n_trunc} truncated at {TOKEN_CAP}), {total_tokens:,} tokens"
    )

    # token-budgeted batches, dispatched with bounded concurrency
    batches: list[tuple[int, list[list[int]]]] = []
    cur: list[list[int]] = []
    cur_tok = 0
    start = 0
    for i, t in enumerate(inputs):
        if cur and (len(cur) >= EMBED_BATCH or cur_tok + len(t) > EMBED_REQ_TOKEN_BUDGET):
            batches.append((start, cur))
            cur, cur_tok, start = [], 0, i
        cur.append(t)
        cur_tok += len(t)
    if cur:
        batches.append((start, cur))

    client = OpenAI()

    def one_batch(job: tuple[int, list[list[int]]]) -> tuple[int, np.ndarray]:
        start, batch = job
        delay = 2.0
        for attempt in range(7):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
                assert arr.shape == (len(batch), EMBED_DIM), (arr.shape, len(batch))
                return start, arr
            except Exception as exc:  # transport/ratelimit: retry with backoff
                if attempt == 6:
                    raise
                log(f"embeddings[{battery}] batch@{start}: {type(exc).__name__}; retry {delay}s")
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    emb = np.zeros((len(inputs), EMBED_DIM), dtype=np.float32)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as ex:
        for start, arr in ex.map(one_batch, batches):
            emb[start : start + len(arr)] = arr
    assert not (np.linalg.norm(emb, axis=1) == 0).any(), "empty embedding row persisted"
    emb = unit_rows(emb.astype(np.float64)).astype(np.float32)
    log(f"embeddings[{battery}]: {len(batches)} requests in {time.monotonic() - t0:.0f}s")

    meta = {"n_truncated": n_trunc, "n_empty": n_empty, "total_tokens": total_tokens}
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        emb=emb.astype(np.float16),
        context_ids=np.array([r["context_id"] for r in keep]),
        draws=np.array([r["draw"] for r in keep], dtype=np.int64),
        meta_json=json.dumps({**meta, "model": EMBED_MODEL, "dim": EMBED_DIM}),
    )
    return {
        "ids": [r["context_id"] for r in keep],
        "draws": np.array([r["draw"] for r in keep]),
        "emb": emb,
        **meta,
    }


def context_means(ids: list[str], emb: np.ndarray, order: list[str]) -> tuple[np.ndarray, dict]:
    """Mean of L2-normalized per-draw embeddings per context, in ``order``."""
    row_of = {cid: i for i, cid in enumerate(order)}
    sums = np.zeros((len(order), emb.shape[1]), dtype=np.float64)
    counts = np.zeros(len(order), dtype=np.int64)
    idx = np.array([row_of[c] for c in ids], dtype=np.int64)
    np.add.at(sums, idx, emb.astype(np.float64))
    np.add.at(counts, idx, 1)
    assert (counts > 0).all(), f"{int((counts == 0).sum())} contexts with zero embeddable draws"
    return sums / counts[:, None], {"draws_per_context_min": int(counts.min())}


def qwen_text_means(root: Path, order: list[str], battery: str) -> np.ndarray:
    """Banked Qwen3-Embedding-8B per-context means, aligned to ``order``.

    Produced pod-side by scripts/issue2215_sepcmp_qwen_embed.py and uploaded
    to QWEN_HF_PREFIX; staged at the repo's CURRENT main (the upload
    postdates REV_MAIN by design)."""
    local = stage(f"{QWEN_HF_PREFIX}/means_qwen3_8b_{battery}.npz", root, None)
    z = np.load(local, allow_pickle=False)
    cids = [str(s) for s in z["context_ids"]]
    assert set(cids) == set(order), (
        battery,
        len(set(order) - set(cids)),
        "context-id set drift vs the banked qwen means",
    )
    row_of = {c: i for i, c in enumerate(cids)}
    mean = z["mean"].astype(np.float64)
    assert mean.shape == (len(order), 4096), mean.shape
    return mean[np.array([row_of[c] for c in order], dtype=np.int64)]


# ── parent REAL leg: stream the va2215 tail store at layer 19 ─────────


def parent_real_means(root: Path, order: list[str], keep_shards: bool) -> tuple[np.ndarray, dict]:
    """Layer-19 tail-inclusive answer means per context (streamed shards)."""
    cache = root / "parent_real_L19_tail.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        assert list(z["ids"]) == order or [str(s) for s in z["ids"]] == order
        return z["mean"].astype(np.float64), {"n_valid_zero": int(z["n_valid_zero"])}
    row_of = {cid: i for i, cid in enumerate(order)}
    sums = np.zeros((len(order), 3584), dtype=np.float64)
    n_valid = np.zeros(len(order), dtype=np.int64)
    for shard_rel in VA2215_SHARDS:
        local = stage(shard_rel, root, REV_MAIN)
        payload = torch.load(local, map_location="cpu", mmap=True, weights_only=False)
        li = list(payload["layers"]).index(LAYER)
        empty = set(payload.get("empty_rows", []))
        rows_j, rows_tgt = [], []
        for j, meta in enumerate(payload["index"]):
            if j in empty:
                continue
            rows_j.append(j)
            rows_tgt.append(row_of[meta["context_id"]])
        vals = payload["va_tail_incl"][torch.tensor(rows_j), li, :].double().numpy()
        np.add.at(sums, np.array(rows_tgt), vals)
        np.add.at(n_valid, np.array(rows_tgt), 1)
        del payload, vals
        if not keep_shards:
            local.unlink()
        log(f"va2215 shard {Path(shard_rel).name}: accumulated ({len(rows_j)} rows)")
    n_zero = int((n_valid == 0).sum())
    mean = sums / np.maximum(n_valid, 1)[:, None]
    np.savez(cache, mean=mean.astype(np.float32), ids=np.array(order), n_valid_zero=n_zero)
    return mean, {"n_valid_zero": n_zero}


# ── main ──────────────────────────────────────────────────────────────


def analyze_battery(
    battery: str,
    bank: dict,
    pt: ANA.PairTable,
    spaces: dict[str, np.ndarray],
    batt_seed_idx: int,
    b_boot: int,
) -> tuple[list[dict], list[dict]]:
    """Per-cell three-space separation ratios, CIs, contrasts, Spearman."""
    views = ANA.build_cell_views(bank, pt)
    cells = [c for c in pt.cells if not c.endswith("_rev")]
    cell_records: list[dict] = []
    pair_rows: list[dict] = []
    for cell_i, cell in enumerate(cells):
        cv = views[cell]
        frame = CellFrame(bank, pt, cv)
        x_local = {s: m[cv.ctx_rows] for s, m in spaces.items()}
        pair_d = {s: frame.pair_dists(x) for s, x in x_local.items()}
        yard = {s: frame.yard_tensor(x) for s, x in x_local.items()}
        obs_yard = {s: frame.yardstick(y) for s, y in yard.items()}
        obs_ratio = {s: float(np.median(pair_d[s])) / obs_yard[s] for s in spaces}
        rng = np.random.default_rng([SEED, batt_seed_idx, cell_i])
        draws = rng.integers(0, frame.n_carr, size=(b_boot, frame.n_carr))
        boot = bootstrap_cell(frame, pair_d, yard, draws)
        rec: dict = {
            "battery": battery,
            "cell": cell,
            "n_pairs": int(len(cv.pair_idx)),
            "n_carriers": frame.n_carr,
            "n_values": frame.n_vals,
            "spaces": {},
            "contrasts": {},
            "spearman": {},
        }
        for s in spaces:
            lo, hi = ci95(boot[s])
            rec["spaces"][s] = {
                "median_ratio": obs_ratio[s],
                "ci_lo": lo,
                "ci_hi": hi,
                "median_pair_dist": float(np.median(pair_d[s])),
                "yardstick": obs_yard[s],
            }
        text_spaces = [s for s in spaces if s.startswith("text")]
        for arm in [*CONTRAST_ARMS, *text_spaces]:
            delta = boot[arm] - boot["real"]
            lo, hi = ci95(delta)
            rec["contrasts"][f"{arm}_minus_real"] = {
                "point": obs_ratio[arm] - obs_ratio["real"],
                "ci_lo": lo,
                "ci_hi": hi,
            }
        spearman_pairs = [("mapped_779ce", "real"), ("mapped_1738ce", "real")]
        for tname in text_spaces:
            spearman_pairs += [("mapped_779ce", tname), ("real", tname)]
        for a, b in spearman_pairs:
            if a not in spaces or b not in spaces:
                continue
            rho, p = spearmanr(pair_d[a], pair_d[b])
            rec["spearman"][f"{a}_vs_{b}"] = {"rho": float(rho), "p": float(p)}
        cell_records.append(rec)
        for k_local, k in enumerate(cv.pair_idx):
            row = {
                "battery": battery,
                "cell": cell,
                "pair_id": pt.pair_ids[int(k)],
                "carrier": pt.pair_carrier[int(k)],
                "value_pair": pt.pair_vp[int(k)],
            }
            for s in spaces:
                row[f"dist_{s}"] = float(pair_d[s][k_local])
                row[f"ratio_{s}"] = float(pair_d[s][k_local] / obs_yard[s])
            pair_rows.append(row)
        log(f"{battery}/{cell}: ratio " + " ".join(f"{s}={obs_ratio[s]:.2f}" for s in spaces))
    return cell_records, pair_rows


def make_figures(cell_records: list[dict], fig_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    pal = paper_palette_blog(4)
    written: list[Path] = []

    # (a) per-cell ratio scatter vs the real answers: mapped panel (+ text
    # companion panel when a text space is present)
    def ratio_scatter_panel(ax, yspace: str, ylabel: str, title: str) -> float:
        top = 0.0
        for batt, color, marker in (("parent", pal[0], "o"), ("dbe", pal[1], "s")):
            recs = [r for r in cell_records if r["battery"] == batt]
            xs = [r["spaces"]["real"]["median_ratio"] for r in recs]
            ys = [r["spaces"][yspace]["median_ratio"] for r in recs]
            err = {}
            for key, sp, vals in (("xerr", "real", xs), ("yerr", yspace, ys)):
                err[key] = np.array(
                    [
                        [v - r["spaces"][sp]["ci_lo"] for r, v in zip(recs, vals)],
                        [r["spaces"][sp]["ci_hi"] - v for r, v in zip(recs, vals)],
                    ]
                )
            ax.errorbar(
                xs,
                ys,
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=0.7,
                alpha=0.85,
                ms=5,
                ls="none",
                label="parent 37-cell battery" if batt == "parent" else "8-type content battery",
                **err,
            )
            for r, x, y in zip(recs, xs, ys):
                ax.text(x, y, r["cell"], fontsize=5.5, alpha=0.8, ha="left", va="bottom")
            top = max(top, max(xs), max(ys))
        hi = top * 1.08
        ax.plot([0, hi], [0, hi], color="gray", lw=1.0, ls="--", zorder=0, label="y = x")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel("real answer-vector separation ratio (pair / same-value yardstick)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return hi

    text_space = next((s for s in cell_records[0]["spaces"] if s.startswith("text")), None)
    n_panels = 2 if text_space else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7.4 * n_panels, 6.4))
    axes = np.atleast_1d(axes)
    ratio_scatter_panel(
        axes[0],
        "mapped_779ce",
        "mapped-prediction separation ratio (single-turn map)",
        "Does the map separate minimal pairs more than the real answers do?",
    )
    axes[0].legend(loc="upper left")
    if text_space:
        ratio_scatter_panel(
            axes[1],
            text_space,
            "text-embedding separation ratio (Qwen3-Embedding-8B)",
            "Does text-embedding separation track the real answers?",
        )
    written += list(savefig_paper(fig, "sepcmp_mapped_vs_real_scatter", dir=fig_dir).values())
    plt.close(fig)

    # (b) per-type grouped comparison: three spaces, dbe types + selected parent cells
    picked = [r for r in cell_records if r["battery"] == "dbe"] + [
        r
        for c in PARENT_FIGURE_CELLS
        for r in cell_records
        if r["battery"] == "parent" and r["cell"] == c
    ]
    labels = [f"{r['cell']}\n({r['battery']})" for r in picked]
    space_labels = {
        "mapped_779ce": "mapped (single-turn map)",
        "real": "real answer vectors",
        "text": "text embedding (OpenAI)",
        QWEN_SPACE: "text embedding (Qwen3-8B)",
    }
    space_labels = {s: v for s, v in space_labels.items() if s in cell_records[0]["spaces"]}
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    xs = np.arange(len(picked))
    width = 0.8 / len(space_labels)
    for k, (s, lab) in enumerate(space_labels.items()):
        vals = [r["spaces"][s]["median_ratio"] for r in picked]
        err = np.array(
            [
                [r["spaces"][s]["median_ratio"] - r["spaces"][s]["ci_lo"] for r in picked],
                [r["spaces"][s]["ci_hi"] - r["spaces"][s]["median_ratio"] for r in picked],
            ]
        )
        ax.bar(
            xs + (k - (len(space_labels) - 1) / 2) * width,
            vals,
            width,
            yerr=err,
            color=pal[k],
            label=lab,
            error_kw={"elinewidth": 0.8},
        )
    ax.axhline(1.0, color="gray", lw=1.0, ls="--", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("median separation ratio (pair / yardstick)")
    ax.set_title(f"Minimal-pair separation across {len(space_labels)} spaces (95% bootstrap CIs)")
    ax.legend()
    written += list(savefig_paper(fig, "sepcmp_pertype_spaces", dir=fig_dir).values())
    plt.close(fig)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage-root", type=Path, default=Path("data/issue_2215/sepcmp_dl"))
    ap.add_argument(
        "--out-root", type=Path, default=Path("eval_results/issue_2215/separation_comparison")
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_2215"))
    ap.add_argument("--b-boot", type=int, default=B_BOOT)
    ap.add_argument("--keep-shards", action="store_true", help="keep va2215 shards on disk")
    ap.add_argument("--figures-only", action="store_true", help="re-render from sepcmp.json")
    ap.add_argument("--skip-hf-upload", action="store_true", help="skip embedding HF upload")
    ap.add_argument(
        "--text-space",
        choices=("qwen3_8b", "openai", "none"),
        default="qwen3_8b",
        help="text-embedding source: banked Qwen3-Embedding-8B means (GPU override route, "
        "default), the OpenAI API route, or none (skipped leg recorded in meta, never "
        "silently absent)",
    )
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    sep_path = args.out_root / "sepcmp.json"
    if args.figures_only:
        payload = json.loads(sep_path.read_text())
        make_figures(payload["cells"], args.fig_dir)
        return

    root = args.stage_root
    import issue779_ffc_n1m_fits as FITS  # deferred heavy sibling import

    # ── parent battery ────────────────────────────────────────────────
    bank = json.loads(stage(f"{VC_BANK_PREFIX}/bank.json", root, REV_PARENT).read_text())
    pt = ANA.PairTable.from_bank(bank, None)
    vc = ANA.load_vc_bank(stage(f"{VC_BANK_PREFIX}/vc_bank.pt", root, REV_PARENT), pt.ids)
    x_ce = vc["ce"][:, vc["layers"].index(LAYER), :].double().numpy()
    del vc
    parent_spaces: dict[str, np.ndarray] = {}
    for arm, rel in ARM_PATHS.items():
        payload = torch.load(stage(rel, root, REV_MAIN), map_location="cpu", weights_only=False)
        assert payload.get("kind") == "ridge" and int(payload["layer"]) == LAYER
        parent_spaces[f"mapped_{arm}"] = FITS.apply_map(payload, x_ce, torch.device("cpu"))
        del payload
    real_parent, real_meta = parent_real_means(root, pt.ids, args.keep_shards)
    parent_spaces["real"] = real_parent
    assert real_meta["n_valid_zero"] == 0, f"parent n_valid==0 contexts: {real_meta}"

    emb_dir = root / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    ep: dict | None = None
    if args.text_space == "qwen3_8b":
        parent_spaces[QWEN_SPACE] = qwen_text_means(root, pt.ids, "parent")
    elif args.text_space == "openai":
        rows_parent = read_rollout_rows([stage(p, root, REV_PARENT) for p in PARENT_JSONL])
        assert len(rows_parent) == 14_040, len(rows_parent)
        ep = embed_battery("parent", rows_parent, emb_dir)
        parent_spaces["text"], _ = context_means(ep["ids"], ep["emb"], pt.ids)

    cells_parent, pairs_parent = analyze_battery("parent", bank, pt, parent_spaces, 0, args.b_boot)

    # ── dbe battery ───────────────────────────────────────────────────
    bank_d = json.loads(
        stage(
            "issue2215_dbe/analysis_tensors/vc_bank_dbe/bank_dbe.json", root, REV_MAIN
        ).read_text()
    )
    pt_d = ANA.PairTable.from_bank(bank_d, None)
    pred = torch.load(
        stage("issue2215_dbe/analysis_tensors/predictions/predictions_L19.pt", root, REV_MAIN),
        map_location="cpu",
        weights_only=False,
    )
    assert pt_d.ids == list(pred["ids"]), "row-order drift vs predictions_L19.pt"
    assert int(pred["layer"]) == LAYER and bool(pred["valid"].all())
    dbe_spaces: dict[str, np.ndarray] = {
        "mapped_779ce": pred["fitted"]["779ce"].to(torch.float64).numpy(),
        "mapped_1738ce": pred["fitted"]["1738ce"].to(torch.float64).numpy(),
        "real": pred["targets"][POOL].to(torch.float64).numpy(),
    }
    ed: dict | None = None
    if not args.no_text:
        rows_dbe = read_rollout_rows([stage(p, root, REV_MAIN) for p in DBE_JSONL])
        assert len(rows_dbe) == 3_600, len(rows_dbe)
        ed = embed_battery("dbe", rows_dbe, emb_dir)
        dbe_spaces["text"], _ = context_means(ed["ids"], ed["emb"], pt_d.ids)

    cells_dbe, pairs_dbe = analyze_battery("dbe", bank_d, pt_d, dbe_spaces, 1, args.b_boot)

    # ── per-context mean embeddings for the HF bank ───────────────────
    if args.text_space == "none":
        embedding_meta: dict = {
            "status": "SKIPPED — text leg not run (--text-space none)",
            "reason": "no valid OPENAI_API_KEY on the VM (every candidate key returned 401 "
            "invalid_api_key at dispatch time, 2026-08-24); leg dropped, never substituted",
        }
    elif args.text_space == "qwen3_8b":
        qmeta_path = stage(f"{QWEN_HF_PREFIX}/meta.json", root, None)
        embedding_meta = {
            "space": QWEN_SPACE,
            "route": "GPU override (user, 2026-08-24) — supersedes the OpenAI "
            "text-embedding-3-large route (key revoked, 401)",
            **json.loads(qmeta_path.read_text()),
        }
    else:
        assert ep is not None and ed is not None
        for battery, order, spaces in (
            ("parent", pt.ids, parent_spaces),
            ("dbe", pt_d.ids, dbe_spaces),
        ):
            np.savez(
                emb_dir / f"means_{battery}.npz",
                mean=spaces["text"].astype(np.float16),
                context_ids=np.array(order),
            )
        embedding_meta = {
            "model": EMBED_MODEL,
            "dim": EMBED_DIM,
            "token_cap": TOKEN_CAP,
            "parent": {k: ep[k] for k in ("n_truncated", "n_empty", "total_tokens")},
            "dbe": {k: ed[k] for k in ("n_truncated", "n_empty", "total_tokens")},
        }
        (emb_dir / "embeddings_meta.json").write_text(json.dumps(embedding_meta, indent=1))
    if not args.skip_hf_upload and args.text_space == "openai":
        from huggingface_hub import HfApi

        assert_hub_dir_filecounts(emb_dir, EMBED_HF_PREFIX, ignore_patterns=["_hf/**"])
        retry_transient(
            lambda: HfApi().upload_folder(
                folder_path=emb_dir,
                path_in_repo=EMBED_HF_PREFIX,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                ignore_patterns=["_hf/**"],
                commit_message="issue #2215 separation-comparison round: rollout text embeddings",
            ),
            what="upload_folder(sepcmp embeddings)",
        )
        log("embeddings uploaded to HF")

    # ── outputs ───────────────────────────────────────────────────────
    all_cells = cells_parent + cells_dbe
    payload = {
        "meta": {
            "round": "separation_comparison",
            "issue": 2215,
            "layer": LAYER,
            "pooling": POOL,
            "metric": "cosine distance",
            "seed": SEED,
            "b_boot": args.b_boot,
            "arms": list(ARM_PATHS),
            "prefix_end": "skipped by design (this round reads context-end maps only)",
            "rev_parent": REV_PARENT,
            "rev_main": REV_MAIN,
            "parent_rev_cells_dropped": [c for c in pt.cells if c.endswith("_rev")],
            "embedding": embedding_meta,
            "minilm_sensitivity": (
                "skipped — 256-text CPU pilot measured 24.4 s (all-MiniLM-L6-v2, batch 64, "
                "shared VM), projecting 28.0 min for the full 17,640-text pass, over the "
                "15-min gate pinned in the dispatch note"
            ),
        },
        "cells": all_cells,
    }
    sep_path.write_text(json.dumps(payload, indent=1))
    with open(args.out_root / "perpair.jsonl", "w") as fh:
        for row in pairs_parent + pairs_dbe:
            fh.write(json.dumps(row) + "\n")
    make_figures(all_cells, args.fig_dir)
    log(f"wrote {sep_path} + perpair.jsonl ({len(pairs_parent) + len(pairs_dbe)} rows) + figures")


if __name__ == "__main__":
    main()
