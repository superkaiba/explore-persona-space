"""#2564 lang/oneword PILOT — VM-side reads + figures.

Consumes the pilot artifacts produced by ``scripts/issue2564_langow_pilot_run.py``
(HF ``superkaiba1/explore-persona-space-data`` under
``issue2564_minpair/lang_oneword_pilot/``, or a local out-root via ``--local``)
and mirrors the parent #2564 per-pair analysis on the two new axes:

- loads the two frozen ridge arms EXACTLY as the parent does
  (``issue779_monitoring/n1m_readout/weights/L19/ridge.pt`` and
  ``issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt``;
  ``apply_map`` from ``scripts/issue779_ffc_n1m_fits.py``), plus the raw
  ``arm_iddelta`` v_C-delta baseline;
- per-pair rows mirroring the parent ``perpair.jsonl`` fields (predicted vs
  observed answer-state delta cosines at L19, span + tail poolings, split-half
  reliability r_half / Spearman-Brown r10 / noise norm, norms, changed_tokens;
  ``norm_text`` emitted null — the parent's text-embedding leg is out of pilot
  scope);
- matched-pair retrieval within the ~96-row pilot pool: per-context acc@1
  (cosine + euclidean, chance 1/n_pool, both mapped arms) via
  ``analysis.mapping_baselines.knn_retrieval``, plus a pair-delta rank read
  (each pair's observed delta ranked by cosine against all pair deltas);
- calibration slope per axis per arm (through-origin ||pred-delta|| vs
  ||obs-delta||);
- PROGRAMMATIC language classifier over the answer_language cell's raw draws
  (zh: CJK codepoint fraction; es-vs-en: diacritic + stopword heuristic; rule
  recorded in the summary meta) — REPORT-ONLY, gates nothing.

Outputs: ``eval_results/issue_2564/lang_oneword_pilot/{perpair.jsonl,summary.json}``
+ ``figures/issue_2564/lang_oneword_pilot/*.png/pdf`` (paper-plots conventions;
pair-level bootstrap CIs, B=10,000, seed 2564).

Run (VM):

    uv run python scripts/issue2564_langow_pilot_reads.py            # stage from HF
    uv run python scripts/issue2564_langow_pilot_reads.py --local /workspace/eps2564_langow
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/torch — thread caps + HF credentials

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import unicodedata  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic, write_jsonl_atomic  # noqa: E402

logger = logging.getLogger("issue2564_langow_reads")

REPO_ROOT = Path(__file__).resolve().parent.parent
assert (REPO_ROOT / "pyproject.toml").is_file(), REPO_ROOT

ISSUE = 2564
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue2564_minpair/lang_oneword_pilot"
RIDGE_779_PATH = "issue779_monitoring/n1m_readout/weights/L19/ridge.pt"
RIDGE_1738_PATH = "issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt"

CELLS = ("answer_language", "query_content_oneword")
MAP_LAYERS = (14, 19, 26)
PRIMARY_LAYER = 19
HIDDEN = 3584
ARMS = ("arm_779ce", "arm_1738ce")
ALL_ARMS = (*ARMS, "arm_iddelta")

N_SPLITS = 20
SPLIT_SEED = 2564
BOOT_B = 10_000
BOOT_SEED = 2564
FIRED_THRESHOLD = 0.7

OUT_DIR = REPO_ROOT / "eval_results" / "issue_2564" / "lang_oneword_pilot"
FIG_DIR = REPO_ROOT / "figures" / "issue_2564" / "lang_oneword_pilot"

# Language classifier rule (programmatic, deterministic; recorded in summary meta).
LANG_RULE = (
    "zh if CJK-codepoint fraction of alnum chars > 0.2; else es if (has Spanish "
    "diacritic [áéíóúñü¿¡] or Spanish-stopword fraction > English-stopword fraction "
    "over whitespace tokens); else en. fired = fraction of a context's valid draws "
    f"classified as the instructed language; threshold {FIRED_THRESHOLD}."
)
_ES_STOP = {
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "de",
    "del",
    "que",
    "y",
    "en",
    "es",
    "por",
    "para",
    "con",
    "su",
    "se",
    "no",
    "como",
    "más",
    "pero",
    "si",
    "al",
    "lo",
    "también",
    "muy",
    "puede",
    "hay",
    "esto",
    "esta",
    "este",
    "o",
}
_EN_STOP = {
    "the",
    "a",
    "an",
    "of",
    "that",
    "and",
    "in",
    "is",
    "for",
    "with",
    "your",
    "it",
    "not",
    "as",
    "more",
    "but",
    "if",
    "to",
    "this",
    "or",
    "can",
    "there",
    "are",
    "be",
    "you",
    "on",
    "have",
}
_ES_DIACRITIC = re.compile(r"[áéíóúñü¿¡]", re.IGNORECASE)


def classify_language(text: str) -> str:
    """Deterministic per-draw language call per LANG_RULE (report-only)."""
    alnum = [ch for ch in text if ch.isalnum()]
    if alnum:
        cjk = sum(1 for ch in alnum if "CJK" in unicodedata.name(ch, ""))
        if cjk / len(alnum) > 0.2:
            return "zh"
    toks = [t.strip(".,!?;:()\"'").lower() for t in text.split()]
    toks = [t for t in toks if t]
    es_frac = sum(1 for t in toks if t in _ES_STOP) / max(1, len(toks))
    en_frac = sum(1 for t in toks if t in _EN_STOP) / max(1, len(toks))
    if _ES_DIACRITIC.search(text) or es_frac > en_frac:
        return "es"
    return "en"


LANG_CODE = {"english": "en", "chinese": "zh", "spanish": "es"}


# ── staging ───────────────────────────────────────────────────────────────


def stage_inputs(local: str | None, stage_dir: Path) -> Path:
    """Resolve the pilot out-root: ``--local`` dir as-is, else stage the known
    file set from HF via the retried/atomic ``stage_hub_file`` helper."""
    if local:
        root = Path(local)
        assert (root / "manifests" / "pilot_bank.json").is_file(), (
            f"--local {root} lacks manifests/pilot_bank.json"
        )
        return root
    from explore_persona_space.orchestrate.hub import stage_hub_file

    stage_dir.mkdir(parents=True, exist_ok=True)
    files = [
        ("manifests/pilot_bank.json", "manifests/pilot_bank.json"),
        ("analysis_tensors/vc/vc_langow_bank.pt", "vc_store/vc_langow_bank.pt"),
    ]
    for cell in CELLS:
        files.append(
            (f"raw_completions/anchors/anchors_{cell}.jsonl", f"anchors/anchors_{cell}.jsonl")
        )
        files.append((f"analysis_tensors/va/va_langow_{cell}.pt", f"va_store/va_langow_{cell}.pt"))
    for remote_rel, local_rel in files:
        target = stage_dir / local_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # overwrite=True: the producer re-uploads with resume_skip=False, so a
        # stale local mirror would silently serve a prior run (review finding 2).
        stage_hub_file(
            HF_DATA_REPO, f"{HF_PREFIX}/{remote_rel}", target, repo_type="dataset", overwrite=True
        )
        print(f"[stage] {remote_rel} -> {target}", flush=True)
    return stage_dir


def stage_ridge_payloads(stage_dir: Path) -> dict[str, Path]:
    from explore_persona_space.orchestrate.hub import stage_hub_file

    out: dict[str, Path] = {}
    for arm, remote in (("arm_779ce", RIDGE_779_PATH), ("arm_1738ce", RIDGE_1738_PATH)):
        target = stage_dir / "ridge" / arm / Path(remote).name
        target.parent.mkdir(parents=True, exist_ok=True)
        stage_hub_file(HF_DATA_REPO, remote, target, repo_type="dataset")
        out[arm] = target
    return out


def load_ridge_payload(path: Path, expect_d: int, arm: str) -> dict:
    """Mirror the parent's payload asserts (kind/shape) before any use."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload.get("kind") == "ridge", (arm, payload.get("kind"))
    w = payload["W"]
    assert tuple(w.shape) == (expect_d, expect_d), (arm, tuple(w.shape))
    for key in ("xmu", "xsd", "ymu"):
        assert payload[key].shape[-1] == expect_d, (arm, key, payload[key].shape)
    return payload


def _import_apply_map():
    """``apply_map`` from the main-resident #779 fits script (script, not a
    package module — load by path under a unique name)."""
    path = REPO_ROOT / "scripts" / "issue779_ffc_n1m_fits.py"
    spec = importlib.util.spec_from_file_location("issue779_ffc_n1m_fits_for_langow", path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue779_ffc_n1m_fits_for_langow"] = mod
    spec.loader.exec_module(mod)
    return mod.apply_map


# ── loading ───────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for ln in fh.read().split("\n"):
            if ln.strip():
                rows.append(json.loads(ln))
    assert rows, f"empty jsonl: {path}"
    return rows


def load_pilot(root: Path) -> dict:
    """Bank + anchors + tensors -> aligned per-context arrays.

    Returns contexts (ordered), vc19 (n_ctx, d) fp64, per-draw tail/span L19
    stacks (n_ctx, K, d) fp32 + draw_valid (n_ctx, K), pairs, raw texts by
    context for the language read.
    """
    bank = json.loads((root / "manifests" / "pilot_bank.json").read_text(encoding="utf-8"))
    contexts = bank["contexts"]
    pairs = bank["pairs"]
    ctx_ids = [c["id"] for c in contexts]
    ctx_pos = {cid: i for i, cid in enumerate(ctx_ids)}
    assert len(ctx_pos) == len(contexts), "duplicate context ids in bank"

    vc_store = torch.load(
        root / "vc_store" / "vc_langow_bank.pt", map_location="cpu", weights_only=False
    )
    layers = list(vc_store["layers"])
    li = layers.index(PRIMARY_LAYER) if PRIMARY_LAYER in layers else len(layers) - 1
    if layers[li] != PRIMARY_LAYER:
        logger.warning(
            "[load] PRIMARY_LAYER %d absent (layers=%s) — using %d (tiny store?)",
            PRIMARY_LAYER,
            layers,
            layers[li],
        )
    store_pos = {cid: i for i, cid in enumerate(vc_store["context_ids"])}
    assert set(ctx_ids) <= set(store_pos), sorted(set(ctx_ids) - set(store_pos))[:5]
    vc19 = np.stack(
        [vc_store["vc"][store_pos[cid], li].numpy().astype(np.float64) for cid in ctx_ids]
    )

    draws_seen: set[int] = set()
    texts: dict[str, list[str | None]] = {}
    tail: dict[str, np.ndarray] = {}
    span: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    d_model = vc19.shape[1]
    for cell in CELLS:
        anchors = _read_jsonl(root / "anchors" / f"anchors_{cell}.jsonl")
        va_store = torch.load(
            root / "va_store" / f"va_langow_{cell}.pt", map_location="cpu", weights_only=False
        )
        va_layers = list(va_store["layers"])
        assert layers[li] in va_layers, (cell, layers[li], va_layers)
        vli = va_layers.index(layers[li])
        index = va_store["index"]
        assert len(index) == len(anchors), (cell, len(index), len(anchors))
        empty = set(va_store["empty_rows"])
        k = max(int(r["draw"]) for r in index) + 1
        draws_seen.add(k)
        cell_ctx = sorted({r["context_id"] for r in index}, key=lambda c: ctx_pos[c])
        for cid in cell_ctx:
            tail.setdefault(cid, np.zeros((k, d_model), dtype=np.float32))
            span.setdefault(cid, np.zeros((k, d_model), dtype=np.float32))
            valid.setdefault(cid, np.zeros(k, dtype=bool))
            texts.setdefault(cid, [None] * k)
        va_tail = va_store["va_tail_incl"]
        va_span = va_store["va_span"]
        for row_i, (rec, arow) in enumerate(zip(index, anchors)):
            assert rec["context_id"] == arow["context_id"] and rec["draw"] == arow["draw"], (
                cell,
                row_i,
                rec["context_id"],
                arow["context_id"],
            )
            cid, dr = rec["context_id"], int(rec["draw"])
            texts[cid][dr] = arow["text"]
            if row_i in empty:
                continue
            tail[cid][dr] = va_tail[row_i, vli].float().numpy()
            span[cid][dr] = va_span[row_i, vli].float().numpy()
            valid[cid][dr] = True
    assert len(draws_seen) == 1, f"heterogeneous draw counts across cells: {draws_seen}"
    k = draws_seen.pop()
    for cid in ctx_ids:
        assert cid in valid, f"context {cid} has no captured rows"
    return {
        "contexts": contexts,
        "ctx_ids": ctx_ids,
        "ctx_pos": ctx_pos,
        "pairs": pairs,
        "vc19": vc19,
        "tail": np.stack([tail[cid] for cid in ctx_ids]),
        "span": np.stack([span[cid] for cid in ctx_ids]),
        "valid": np.stack([valid[cid] for cid in ctx_ids]),
        "texts": texts,
        "k": k,
        "layer_used": layers[li],
    }


# ── math (mirrors issue2564_analysis.py) ──────────────────────────────────


def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = na * nb
    out = np.full(a.shape[:-1], np.nan)
    ok = denom > 0
    out[ok] = np.einsum("...d,...d->...", a, b)[ok] / denom[ok]
    return out


def cross_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """All-pairs cosine matrix (n_a, n_b) via normalized matmul (zero-norm -> NaN)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a / np.where(na > 0, na, np.nan)) @ (b / np.where(nb > 0, nb, np.nan)).T


def through_origin_slope(pred_norm: np.ndarray, obs_norm: np.ndarray) -> float:
    denom = float(np.sum(obs_norm**2))
    return float(np.sum(pred_norm * obs_norm) / denom) if denom > 0 else float("nan")


def spearman_brown(r_half: float) -> float:
    return 2 * r_half / (1 + r_half) if np.isfinite(r_half) and r_half > -1 else float("nan")


def split_half_stats(
    tail: np.ndarray, valid: np.ndarray, ai: np.ndarray, bi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parent's split-half reliability per pair: n_splits random draw-halves,
    delta-of-half-means cosine + half-delta noise norm (NaN when either endpoint
    has <2 valid draws)."""
    n_pairs = len(ai)
    r = np.zeros(n_pairs)
    noise = np.zeros(n_pairs)
    n_used = np.zeros(n_pairs)
    k = tail.shape[1]
    for s in range(N_SPLITS):
        rng = np.random.default_rng([SPLIT_SEED, s])
        scores = rng.random((tail.shape[0], k))
        h1 = np.zeros((tail.shape[0], tail.shape[2]))
        h2 = np.zeros_like(h1)
        ok_ctx = np.zeros(tail.shape[0], dtype=bool)
        for c in range(tail.shape[0]):
            idx = np.flatnonzero(valid[c])
            if len(idx) < 2:
                continue
            order = idx[np.argsort(scores[c, idx])]
            half_a, half_b = order[: len(idx) // 2], order[len(idx) // 2 :]
            h1[c] = tail[c, half_a].mean(axis=0)
            h2[c] = tail[c, half_b].mean(axis=0)
            ok_ctx[c] = True
        ok_pair = ok_ctx[ai] & ok_ctx[bi]
        d1 = h1[ai] - h1[bi]
        d2 = h2[ai] - h2[bi]
        cs = np.nan_to_num(rowwise_cos(d1, d2), nan=0.0)
        r += np.where(ok_pair, cs, 0.0)
        noise += np.where(ok_pair, np.linalg.norm(d1 - d2, axis=1) / 2, 0.0)
        n_used += ok_pair
    with np.errstate(invalid="ignore", divide="ignore"):
        r_half = np.where(n_used > 0, r / np.maximum(n_used, 1), np.nan)
        noise_norm = np.where(n_used > 0, noise / np.maximum(n_used, 1), np.nan)
    r10 = np.array([spearman_brown(float(x)) for x in r_half])
    return r_half, r10, noise_norm


# ── analysis ──────────────────────────────────────────────────────────────


def compute(pilot: dict, ridge_paths: dict[str, Path]) -> tuple[list[dict], dict]:
    apply_map = _import_apply_map()
    dev = torch.device("cpu")
    d = pilot["vc19"].shape[1]

    mapped: dict[str, np.ndarray] = {}
    for arm in ARMS:
        payload = load_ridge_payload(ridge_paths[arm], d, arm)
        mapped[arm] = np.asarray(apply_map(payload, pilot["vc19"], dev), dtype=np.float64)
        assert mapped[arm].shape == pilot["vc19"].shape, (arm, mapped[arm].shape)

    valid = pilot["valid"]
    counts = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        tail_mean = (
            np.einsum("ck,ckd->cd", valid.astype(np.float64), pilot["tail"].astype(np.float64))
            / np.maximum(counts, 1)[:, None]
        )
        span_mean = (
            np.einsum("ck,ckd->cd", valid.astype(np.float64), pilot["span"].astype(np.float64))
            / np.maximum(counts, 1)[:, None]
        )

    ctx_pos = pilot["ctx_pos"]
    pairs = pilot["pairs"]
    ai = np.array([ctx_pos[p["a"]] for p in pairs])
    bi = np.array([ctx_pos[p["b"]] for p in pairs])

    obs_tail = tail_mean[ai] - tail_mean[bi]
    obs_span = span_mean[ai] - span_mean[bi]
    pred = {arm: mapped[arm][ai] - mapped[arm][bi] for arm in ARMS}
    pred["arm_iddelta"] = pilot["vc19"][ai] - pilot["vc19"][bi]

    r_half, r10, noise_norm = split_half_stats(pilot["tail"], valid, ai, bi)

    # Language read (report-only): per-context fired fraction over valid draws.
    fired: dict[str, float | None] = {}
    lang_call: dict[str, dict[str, float]] = {}
    for c in pilot["contexts"]:
        cid = c["id"]
        if c["cell"] != "answer_language" or c["value_id"] == "bare":
            fired[cid] = None
            continue
        want = LANG_CODE[c["value_id"]]
        calls = [
            classify_language(t)
            for t, ok in zip(pilot["texts"][cid], pilot["valid"][ctx_pos[cid]])
            if ok and t is not None
        ]
        frac = sum(1 for x in calls if x == want) / max(1, len(calls))
        fired[cid] = frac
        lang_call.setdefault(c["value_id"], {})[cid] = frac

    # Answer-length delta (mean over valid draws of retokenized completion len
    # is not persisted; use text char length as the pilot's cheap proxy).
    def _len_mean(cid: str) -> float:
        vals = [
            len(t)
            for t, ok in zip(pilot["texts"][cid], pilot["valid"][ctx_pos[cid]])
            if ok and t is not None
        ]
        return float(np.mean(vals)) if vals else float("nan")

    rows: list[dict] = []
    for pi, p in enumerate(pairs):
        fa = fired.get(p["a"])
        fb = fired.get(p["b"])
        row = {
            "pair_id": p["pair_id"],
            "pair_class": p["pair_class"],
            "axis": p["axis"],
            "carrier": p["carrier"],
            "value_a": p["value_a"],
            "value_b": p["value_b"],
            "orientation": "as-authored",
            "changed_tokens": p["changed_tokens"],
            "n_draws_a": int(counts[ai[pi]]),
            "n_draws_b": int(counts[bi[pi]]),
            "ans_len_delta": _len_mean(p["a"]) - _len_mean(p["b"]),
            "norm_obs_tail_L19": float(np.linalg.norm(obs_tail[pi])),
            "norm_obs_span_L19": float(np.linalg.norm(obs_span[pi])),
            "norm_text": None,  # parent's text-embedding leg out of pilot scope
            "r_half": float(r_half[pi]),
            "r10": float(r10[pi]),
            "noise_norm": float(noise_norm[pi]),
            "fired_a_70": (None if fa is None else bool(fa >= FIRED_THRESHOLD)),
            "fired_b_70": (None if fb is None else bool(fb >= FIRED_THRESHOLD)),
        }
        row["pair_fired_70"] = (
            None
            if row["fired_a_70"] is None and row["fired_b_70"] is None
            else bool((row["fired_a_70"] in (True, None)) and (row["fired_b_70"] in (True, None)))
        )
        for arm in ALL_ARMS:
            row[f"cos_{arm}"] = float(rowwise_cos(pred[arm][pi], obs_tail[pi]))
            row[f"cos_span_{arm}"] = float(rowwise_cos(pred[arm][pi], obs_span[pi]))
            row[f"norm_pred_{arm}"] = float(np.linalg.norm(pred[arm][pi]))
        for arm in ARMS:
            row[f"cos_vs_iddelta_{arm}"] = float(
                rowwise_cos(pred[arm][pi], pred["arm_iddelta"][pi])
            )
        rows.append(row)

    # Retrieval reads.
    n_pool = len(pilot["ctx_ids"])
    retrieval: dict[str, dict] = {}
    for arm in ARMS:
        retrieval[arm] = {
            "per_context": {
                metric: knn_retrieval(mapped[arm], tail_mean, ks=(1,), metric=metric)
                for metric in ("cosine", "euclidean")
            },
            "chance_at_1": 1.0 / n_pool,
            "n_pool": n_pool,
        }
    pair_rank: dict[str, dict] = {}
    for arm in ALL_ARMS:
        cs = cross_cos(pred[arm], obs_tail)  # (n_pairs_pred, n_pairs_obs)
        order = np.argsort(-np.nan_to_num(cs, nan=-np.inf), axis=1)
        ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(len(pairs))])
        by_axis = {}
        for axis in ("answer_language", "query_content_oneword"):
            m = np.array([p["axis"] == axis for p in pairs])
            by_axis[axis] = {
                "acc_at_1": float(np.mean(ranks[m] == 1)),
                "median_rank": float(np.median(ranks[m])),
                "n": int(m.sum()),
            }
        pair_rank[arm] = {
            "acc_at_1": float(np.mean(ranks == 1)),
            "median_rank": float(np.median(ranks)),
            "chance_at_1": 1.0 / len(pairs),
            "by_axis": by_axis,
        }

    # Calibration slope per axis per arm.
    calibration: dict[str, dict[str, float]] = {}
    obs_norm_all = np.linalg.norm(obs_tail, axis=1)
    for arm in ALL_ARMS:
        pred_norm = np.array([r[f"norm_pred_{arm}"] for r in rows])
        calibration[arm] = {}
        for axis in ("answer_language", "query_content_oneword", "all"):
            m = (
                np.ones(len(rows), dtype=bool)
                if axis == "all"
                else np.array([r["axis"] == axis for r in rows])
            )
            calibration[arm][axis] = through_origin_slope(pred_norm[m], obs_norm_all[m])

    fired_by_value = {
        v: {
            "mean_fired_frac": float(np.mean(list(d_.values()))),
            "n_contexts": len(d_),
            "frac_ge_70": float(np.mean([x >= FIRED_THRESHOLD for x in d_.values()])),
        }
        for v, d_ in sorted(lang_call.items())
    }

    summary = {
        "issue": ISSUE,
        "n_contexts": n_pool,
        "n_pairs": len(pairs),
        "n_pairs_by_class": {
            cls: sum(1 for p in pairs if p["pair_class"] == cls)
            for cls in ("install", "swap", "query_content_oneword")
        },
        "k_draws": pilot["k"],
        "layer_used": pilot["layer_used"],
        "arms": list(ALL_ARMS),
        "ridge_paths": {"arm_779ce": RIDGE_779_PATH, "arm_1738ce": RIDGE_1738_PATH},
        "cos_median_by_axis_arm": {
            arm: {
                axis: float(np.nanmedian([r[f"cos_{arm}"] for r in rows if r["axis"] == axis]))
                for axis in ("answer_language", "query_content_oneword")
            }
            for arm in ALL_ARMS
        },
        "retrieval_per_context": retrieval,
        "retrieval_pair_rank": pair_rank,
        "calibration_slope": calibration,
        "language_read": {
            "rule": LANG_RULE,
            "report_only": True,
            "fired_by_value": fired_by_value,
        },
        "notes": {
            "norm_text": "null in perpair — parent's text-embedding leg out of pilot scope",
            "orientation": "as-authored (no sign randomization in pilot)",
        },
        "repro": _repro_meta(),
    }
    return rows, summary


def _repro_meta() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        **as_metadata_dict(git_provenance()),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ── figures ───────────────────────────────────────────────────────────────


def _boot_ci(vals: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(vals), size=(BOOT_B, len(vals)))
    meds = np.median(vals[idx], axis=1)
    return float(np.median(vals)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def make_figures(rows: list[dict], summary: dict) -> list[str]:
    from explore_persona_space.analysis.paper_plots import (
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    rng = np.random.default_rng(BOOT_SEED)
    arm_label = {
        "arm_779ce": "ridge #779 (single-turn)",
        "arm_1738ce": "ridge #1738 (multi-turn)",
        "arm_iddelta": "identity (raw v_C delta)",
    }
    arm_color = {
        "arm_779ce": paper_color("neural_map"),
        "arm_1738ce": paper_color("oracle_answer"),
        "arm_iddelta": "0.45",
    }
    classes = (
        ("answer_language", "install", "language: install (value vs bare)"),
        ("answer_language", "swap", "language: swap (value vs value)"),
        ("query_content_oneword", "query_content_oneword", "query one-word swap"),
    )

    # Fig 1: predicted-vs-observed delta cosine by pair class x arm (median + CI).
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    y = 0
    yticks, ylabels = [], []
    for _axis, cls, label in classes:
        for arm in ALL_ARMS:
            vals = np.array(
                [r[f"cos_{arm}"] for r in rows if r["pair_class"] == cls], dtype=np.float64
            )
            med, lo, hi = _boot_ci(vals, rng)
            ax.plot([lo, hi], [y, y], color=arm_color[arm], lw=1.6)
            ax.scatter(
                [med],
                [y],
                color=arm_color[arm],
                s=22,
                zorder=3,
                label=arm_label[arm] if y < len(ALL_ARMS) else None,
            )
            yticks.append(y)
            ylabels.append(f"{label} — {arm_label[arm]}")
            y += 1
        y += 0.6
    ax.axvline(0.0, color="0.7", lw=0.8, ls="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("cos(predicted delta, observed answer-state delta), L19 tail")
    written.append(str(savefig_paper(fig, "langow_cos_by_class_arm", dir=FIG_DIR)["png"]))
    plt.close(fig)

    # Fig 2: calibration scatter (norm pred vs norm obs) with through-origin slope.
    fig, axes = plt.subplots(1, len(ALL_ARMS), figsize=(9.0, 3.0), sharey=True)
    obs = np.array([r["norm_obs_tail_L19"] for r in rows])
    ax_marker = {"answer_language": "o", "query_content_oneword": "^"}
    for j, arm in enumerate(ALL_ARMS):
        ax = axes[j]
        pred_n = np.array([r[f"norm_pred_{arm}"] for r in rows])
        for axis, mk in ax_marker.items():
            m = np.array([r["axis"] == axis for r in rows])
            ax.scatter(
                obs[m], pred_n[m], s=14, marker=mk, color=arm_color[arm], alpha=0.75, label=axis
            )
        slope = summary["calibration_slope"][arm]["all"]
        if np.isfinite(slope) and len(obs):
            xs = np.linspace(0, float(np.nanmax(obs)) * 1.05, 8)
            ax.plot(xs, slope * xs, color="0.3", lw=1.0, ls="--")
        ax.set_title(f"{arm_label[arm]}\nslope={slope:.3f}", fontsize=7.5)
        ax.set_xlabel("||observed delta||")
        if j == 0:
            ax.set_ylabel("||predicted delta||")
            ax.legend(fontsize=6, frameon=False)
    written.append(str(savefig_paper(fig, "langow_calibration", dir=FIG_DIR)["png"]))
    plt.close(fig)

    # Fig 3: programmatic language-fired fractions per instructed value.
    fbv = summary["language_read"]["fired_by_value"]
    if fbv:
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        names = list(fbv)
        fracs = [fbv[v]["mean_fired_frac"] for v in names]
        ge70 = [fbv[v]["frac_ge_70"] for v in names]
        x = np.arange(len(names))
        ax.bar(
            x - 0.18,
            fracs,
            width=0.36,
            color=paper_color("neural_map"),
            label="mean fired fraction",
        )
        ax.bar(
            x + 0.18, ge70, width=0.36, color=paper_color("oracle_answer"), label="contexts >= 0.7"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction")
        ax.legend(fontsize=6, frameon=False)
        written.append(str(savefig_paper(fig, "langow_lang_fired", dir=FIG_DIR)["png"]))
        plt.close(fig)

    # Fig 4: pair-delta retrieval acc@1 by axis x arm (chance line).
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    pr = summary["retrieval_pair_rank"]
    axes_names = ("answer_language", "query_content_oneword")
    x = np.arange(len(axes_names))
    width = 0.8 / len(ALL_ARMS)
    for j, arm in enumerate(ALL_ARMS):
        accs = [pr[arm]["by_axis"][a]["acc_at_1"] for a in axes_names]
        ax.bar(x + (j - 1) * width, accs, width=width, color=arm_color[arm], label=arm_label[arm])
    ax.axhline(pr[ALL_ARMS[0]]["chance_at_1"], color="0.6", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(axes_names, fontsize=7)
    ax.set_ylabel("pair-delta retrieval acc@1")
    ax.legend(fontsize=6, frameon=False)
    written.append(str(savefig_paper(fig, "langow_retrieval", dir=FIG_DIR)["png"]))
    plt.close(fig)
    return written


# ── main ──────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--local", default=None, help="local pilot out-root (skip HF staging of pilot artifacts)"
    )
    ap.add_argument(
        "--stage-dir",
        default=None,
        help="staging dir (default: data/issue_2564/langow_stage under repo root)",
    )
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def _import_check() -> None:
    import inspect

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    apply_map = _import_apply_map()
    params = set(inspect.signature(apply_map).parameters)
    assert {"payload"} <= params or len(params) >= 3, params
    assert callable(knn_retrieval)
    print("[import-check] ok", flush=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_argparser().parse_args()
    if args.import_check:
        _import_check()
        return 0
    stage_dir = (
        Path(args.stage_dir)
        if args.stage_dir
        else (REPO_ROOT / "data" / "issue_2564" / "langow_stage")
    )
    root = stage_inputs(args.local, stage_dir)
    ridge_paths = stage_ridge_payloads(stage_dir)
    pilot = load_pilot(root)
    print(
        f"[load] {len(pilot['ctx_ids'])} contexts / {len(pilot['pairs'])} pairs / "
        f"K={pilot['k']} / layer={pilot['layer_used']}",
        flush=True,
    )
    rows, summary = compute(pilot, ridge_paths)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(OUT_DIR / "perpair.jsonl", rows)
    write_json_atomic(OUT_DIR / "summary.json", summary)
    print(f"[out] {OUT_DIR / 'perpair.jsonl'} ({len(rows)} rows)", flush=True)
    if not args.skip_figures:
        for p in make_figures(rows, summary):
            print(f"[fig] {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
