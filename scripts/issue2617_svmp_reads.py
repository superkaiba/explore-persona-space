"""#2617 SVMP — VM-side reads (unit 2/3): safety-valence minimal pairs.

Consumes the artifacts produced by ``scripts/issue2617_svmp_run.py`` (HF
``superkaiba1/explore-persona-space-data`` under ``issue2617_svmp/``, or a
local out-root via ``--local``) and computes the plan §3/§4.5 registered
statistics, written as a delta on ``scripts/issue2564_gramslot_pilot_reads.py``:

- three frozen transport arms at the primary layer (L19): ``arm_779ce``
  (single-turn ridge), ``arm_1738ce`` (multi-turn ridge), ``arm_iddelta``
  (raw v_C delta = identity+bias baseline — the bias cancels in pair deltas);
- per-pair rows (extended gramslot schema; self-asserted key order) with
  judge refusal rates, flip classification, margins, split-half reliability,
  per-arm direction cosines (tail + span) and flip-axis loadings;
- registered statistics S1/S2/S3/S3-contrast/S3-obs-contrast/S4 (+
  length-partialled twin, collinearity gate 0.6 with tercile fallback), the
  five-branch verdict lattice, a within-source shuffled-pair null (1,000
  batched draws, seed 2617) and stratified/family-clustered bootstrap CIs
  (1,000 draws, LOO-within-resample via Gram-matrix algebra — no per-draw
  python loops);
- retrieval reads (per-context kNN cosine+euclidean k=1,5, full + per-source
  pools; LOO identity+bias baseline; pair-delta rank per class), calibration
  slopes + norm ratios, flip-vs-nonflip slope contrast;
- L14/L26 twin table; per-class n_api_refusal_draws recovered from the judge
  ``save_raw`` artifact; kill-criteria + manipulation-check reporting.

Outputs: ``eval_results/issue_2617/svmp/{summary.json,perpair.jsonl,percontext.jsonl}``.

Run (VM):

    uv run python scripts/issue2617_svmp_reads.py                  # stage from HF
    uv run python scripts/issue2617_svmp_reads.py --local /workspace/eps2617_svmp
    uv run python scripts/issue2617_svmp_reads.py --local /tmp/i2617-smoke --tiny \
        --out-dir /tmp/i2617-reads-out
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/torch — thread caps + HF credentials

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic, write_jsonl_atomic  # noqa: E402

logger = logging.getLogger("issue2617_svmp_reads")

REPO_ROOT = Path(__file__).resolve().parent.parent
assert (REPO_ROOT / "pyproject.toml").is_file(), REPO_ROOT

ISSUE = 2617
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue2617_svmp"
CELL = "query_svmp"

MAP_LAYERS = (14, 19, 26)
PRIMARY_LAYER = 19
HIDDEN = 3584
ARMS = ("arm_779ce", "arm_1738ce")
ALL_ARMS = (*ARMS, "arm_iddelta")
REGISTERED_ARM = "arm_779ce"  # plan §6: headline = single-turn map / L19 / tail

RIDGE_HUB_PATHS = {
    "arm_779ce": "issue779_monitoring/n1m_readout/weights/L{layer}/ridge.pt",
    "arm_1738ce": "issue1738_multiturn/analysis_tensors/weights/L{layer}/context_ridge.pt",
}

PAIR_CLASSES = (
    "obj_flip",
    "verb_flip",
    "subj_ctl",
    "obj_benign",
    "verb_benign",
    "subj_benign",
    "xstest",
)
INTENDED_FLIP_CLASSES = ("obj_flip", "verb_flip", "xstest")
PAIR_SOURCES = ("constructed", "xstest")

FLIP_HI = 0.5  # |Δ rate| >= 0.5 -> measured flip (plan §3)
NONFLIP_LO = 0.1  # |Δ rate| <= 0.1 -> non-flip
MIN_FLIP_PAIRS = 8  # run-abort kill criterion (plan §7)
JUDGE_FLAG_FRAC_MAX = 0.2  # judge-integrity kill criterion (plan §7)
N_NULL_DRAWS = 1_000
NULL_SEED = 2617
N_BOOT = 1_000
BOOT_SEED = 2617
LEN_COLLINEARITY_GATE = 0.6
N_AXIS_SPLITS = 20

OUT_DIR_DEFAULT = REPO_ROOT / "eval_results" / "issue_2617" / "svmp"


def _load_by_path(mod_name: str, rel: str):
    """Load a scripts/ module by path under a unique sys.modules name."""
    path = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Math + payload helpers reused verbatim from the langow reads module.
RD = _load_by_path(
    "issue2564_langow_pilot_reads_for_svmp", "scripts/issue2564_langow_pilot_reads.py"
)

# ── schema ────────────────────────────────────────────────────────────────

_PERPAIR_BASE = (
    "pair_id",
    "pair_class",
    "pair_source",
    "artifact_family_id",
    "carrier",
    "value_a",
    "value_b",
    "context_a",
    "context_b",
    "changed_tokens",
    "word_a",
    "word_b",
    "n_draws_a",
    "n_draws_b",
    "ans_len_a",
    "ans_len_b",
    "ans_len_delta",
    "refusal_rate_a",
    "refusal_rate_b",
    "n_valid_a",
    "n_valid_b",
    "flag_low_valid_a",
    "flag_low_valid_b",
    "n_api_refusal_a",
    "n_api_refusal_b",
    "flip",
    "abs_flip",
    "flip_group",
    "margin_a",
    "margin_b",
    "margin_delta",
    "norm_obs_tail",
    "norm_obs_span",
    "r_half",
    "r10",
    "noise_norm",
    "axis_cos_obs",
    "axis_cos_pred",
)
_PERPAIR_ARM = ("cos_{a}", "cos_span_{a}", "norm_pred_{a}", "norm_ratio_{a}", "axis_cos_pred_{a}")
PERPAIR_KEYS = _PERPAIR_BASE + tuple(t.format(a=arm) for arm in ALL_ARMS for t in _PERPAIR_ARM)

PERCONTEXT_KEYS = (
    "context_id",
    "alias",
    "pair_id",
    "side",
    "pair_class",
    "pair_source",
    "artifact_family_id",
    "value_id",
    "carrier",
    "n_draws_captured",
    "n_valid",
    "n_refused",
    "n_dropped",
    "n_api_refusal",
    "flag_low_valid",
    "refusal_rate",
    "margin",
    "pos_mean_ln_logp",
    "n_pos",
    "n_neg",
    "ans_len_mean",
)


def _assert_schemas() -> None:
    assert len(set(PERPAIR_KEYS)) == len(PERPAIR_KEYS), "duplicate perpair keys"
    assert len(set(PERCONTEXT_KEYS)) == len(PERCONTEXT_KEYS), "duplicate percontext keys"


def _f(x) -> float | None:
    """JSON-safe float: finite -> float, NaN/inf/None -> None."""
    if x is None:
        return None
    xf = float(x)
    return xf if math.isfinite(xf) else None


# ── staging ───────────────────────────────────────────────────────────────

STAGE_FILES = [
    ("manifests/svmp_bank.json", "manifests/svmp_bank.json"),
    ("analysis_tensors/vc/vc_langow_bank.pt", "vc_store/vc_langow_bank.pt"),
    (f"raw_completions/anchors/anchors_{CELL}.jsonl", f"anchors/anchors_{CELL}.jsonl"),
    (f"analysis_tensors/va/va_langow_{CELL}.pt", f"va_store/va_langow_{CELL}.pt"),
    ("raw_completions/judge/judge_scores.json", "judge/judge_scores.json"),
    ("raw_completions/judge/judge_id_map.json", "judge/judge_id_map.json"),
    ("analysis_tensors/margin/margins.json", "margin/margins.json"),
]
# save_raw is required for api-refusal counts on a live-judge run; a dry-run
# (tiny) producer never writes it, so its absence is tolerated at STAGE time
# and adjudicated at LOAD time against judge_scores.json's dry_run flag.
STAGE_FILES_OPTIONAL = [
    (f"raw_completions/judge/judge_raw_{CELL}.json", f"judge/judge_raw_{CELL}.json"),
]


def _resolve_data_repo_revision() -> str:
    """Pin the data repo's main -> resolved commit sha ONCE per run (#2061;
    concern ridge-provenance-unpinned, r2) so every staged run input + ridge
    payload is fetched at ONE recorded revision."""
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(HF_DATA_REPO).sha
    assert sha, f"could not resolve dataset revision for {HF_DATA_REPO}"
    return sha


def _stage_with_overflow(remote_rel: str, target: Path, revision: str | None) -> str:
    """Stage one run artifact from the canonical repo at the pinned revision;
    on a missing entry, fall back to the OVERFLOW repo (quota-403 reroutes land
    there under the same path — concern overflow-staging-disconnected, r2).
    Returns the realized source tag ("canonical" | "overflow")."""
    from huggingface_hub.utils import EntryNotFoundError

    from explore_persona_space.orchestrate.hub import stage_hub_file
    from explore_persona_space.orchestrate.upload_sharded import DEFAULT_OVERFLOW_REPO

    try:
        stage_hub_file(
            HF_DATA_REPO,
            f"{HF_PREFIX}/{remote_rel}",
            target,
            repo_type="dataset",
            revision=revision,
            overwrite=True,
        )
        return "canonical"
    except EntryNotFoundError:
        logger.warning(
            "[stage] %s absent on %s@%s — trying overflow repo %s",
            remote_rel,
            HF_DATA_REPO,
            (revision or "main")[:12],
            DEFAULT_OVERFLOW_REPO,
        )
        stage_hub_file(
            DEFAULT_OVERFLOW_REPO,
            f"{HF_PREFIX}/{remote_rel}",
            target,
            repo_type="dataset",
            overwrite=True,
        )
        return "overflow"


def stage_inputs_svmp(local: str | None, stage_dir: Path) -> tuple[Path, str | None, dict]:
    """Resolve the SVMP out-root: ``--local`` dir as-is, else stage from HF at
    ONE pinned revision (with a per-file overflow-repo fallback).

    Returns ``(root, hf_revision, sources)`` — revision is None on --local;
    sources maps remote_rel -> "canonical" | "overflow" (recorded in
    summary.json's staging block).

    The --local layout is the driver's own out-root (manifests/, vc_store/,
    va_store/, anchors/, judge/, margin/), so a pod out-root or the tiny
    scratch root both resolve without staging.
    """
    if local:
        root = Path(local)
        assert (root / "manifests" / "svmp_bank.json").is_file(), (
            f"--local {root} lacks manifests/svmp_bank.json"
        )
        return root, None, {}
    stage_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(stage_dir).free / 1e9
    assert free_gb >= 1.0, f"staging dir {stage_dir} has {free_gb:.2f} GB free (< 1 GB floor)"
    revision = _resolve_data_repo_revision()
    print(f"[stage] dir={stage_dir} free={free_gb:.1f} GB revision={revision[:12]}", flush=True)
    sources: dict[str, str] = {}
    for remote_rel, local_rel in STAGE_FILES:
        target = stage_dir / local_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # overwrite=True: the producer re-uploads with resume_skip=False; a
        # stale local mirror would silently serve a prior run.
        sources[remote_rel] = _stage_with_overflow(remote_rel, target, revision)
        print(f"[stage] {remote_rel} -> {target} ({sources[remote_rel]})", flush=True)
    for remote_rel, local_rel in STAGE_FILES_OPTIONAL:
        target = stage_dir / local_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            sources[remote_rel] = _stage_with_overflow(remote_rel, target, revision)
            print(f"[stage] {remote_rel} -> {target} ({sources[remote_rel]})", flush=True)
        except Exception as exc:  # adjudicated at load time vs dry_run flag
            logger.warning("[stage] optional %s not staged (%s)", remote_rel, type(exc).__name__)
    return stage_dir, revision, sources


def _fabricate_tiny_ridge(path: Path, d: int, layer: int, arm: str) -> None:
    """Tiny-mode ridge payload at the tiny store dim, run through the SAME
    load_ridge_payload + apply_map path as production (recorded as
    ridge_fabricated in the outputs — never used off the tiny path)."""
    rng = np.random.default_rng([2617, layer, len(arm)])
    w = torch.tensor(0.1 * rng.standard_normal((d, d)), dtype=torch.float32)
    payload = {
        "kind": "ridge",
        "W": w,
        "xmu": torch.zeros(d),
        "xsd": torch.ones(d),
        "ymu": torch.zeros(d),
        "meta": {"fabricated_tiny": True, "layer": layer, "arm": arm},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def stage_ridge_payloads_svmp(
    stage_dir: Path, layers: list[int], *, tiny: bool, d: int, revision: str | None = None
) -> dict[int, dict[str, Path]]:
    """Per-layer, per-arm ridge payload paths (staged from HF at the pinned
    ``revision`` — concern ridge-provenance-unpinned, r2 — or fabricated at
    the tiny store dim under --tiny). Extends RD.stage_ridge_payloads
    (L19-only) to the L14/L26 twin layers."""
    out: dict[int, dict[str, Path]] = {}
    if tiny:
        for layer in layers:
            out[layer] = {}
            for arm in ARMS:
                target = stage_dir / "ridge_tiny" / f"L{layer}" / arm / "ridge.pt"
                if not target.is_file():
                    _fabricate_tiny_ridge(target, d, layer, arm)
                out[layer][arm] = target
        return out
    from explore_persona_space.orchestrate.hub import stage_hub_file

    if revision is None:
        revision = _resolve_data_repo_revision()
    for layer in layers:
        out[layer] = {}
        for arm in ARMS:
            remote = RIDGE_HUB_PATHS[arm].format(layer=layer)
            target = stage_dir / "ridge" / f"L{layer}" / arm / Path(remote).name
            target.parent.mkdir(parents=True, exist_ok=True)
            stage_hub_file(HF_DATA_REPO, remote, target, repo_type="dataset", revision=revision)
            out[layer][arm] = target
    return out


ANCHOR_2564_DIR = REPO_ROOT / "eval_results" / "issue_2564" / "gramslot_pilot"


def _anchor_parity_probe_2564() -> None:
    """Plan §12(f) reuse control (concern ridge-provenance-unpinned, r2):
    recompute the #2564 benign-anchor per-class cos medians from the parent's
    committed perpair.jsonl and assert they match the committed summary the
    figures consume — a drifted / stale anchor artifact fails loud BEFORE any
    new read ships."""
    summ = json.loads((ANCHOR_2564_DIR / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(x)
        for x in (ANCHOR_2564_DIR / "perpair.jsonl").read_text(encoding="utf-8").split("\n")
        if x.strip()
    ]
    assert rows, ANCHOR_2564_DIR
    n_cells = 0
    for arm, per_cls in summ["cos_median_by_axis_arm"].items():
        for cls, v in per_cls.items():
            vals = np.array(
                [r[f"cos_{arm}"] for r in rows if r["pair_class"] == cls], dtype=np.float64
            )
            assert len(vals), (arm, cls)
            m = float(np.nanmedian(vals))
            assert np.isclose(m, float(v), rtol=0.0, atol=1e-9), (
                "anchor-median drift vs committed #2564 summary",
                arm,
                cls,
                m,
                v,
            )
            n_cells += 1
    print(
        f"[anchor-parity] #2564 medians recomputed from perpair.jsonl match "
        f"summary.json ({n_cells} arm x class cells)",
        flush=True,
    )


def _loo_identity_bias(x: np.ndarray, y: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
    """Vectorized leave-one-out form of
    ``analysis.mapping_baselines.identity_bias_predict`` (W = identity,
    learned bias): row i's prediction is ``x_i + mean_{j != i}(y_j - x_j)``
    over the finite rows; non-finite rows stay NaN. Per-row equality with the
    canonical helper is pinned by
    ``tests/test_issue2617_round2_fixes.py::test_loo_identity_bias_matches_canonical_helper``
    (concern identity-bias-helper-bypassed, r2)."""
    pred = np.full_like(y, np.nan)
    n = int(finite_mask.sum())
    if n >= 2:
        diffs = y[finite_mask] - x[finite_mask]
        total = diffs.sum(axis=0)
        loo_bias = (total[None, :] - diffs) / (n - 1)
        pred[finite_mask] = x[finite_mask] + loo_bias
    return pred


# ── loading ───────────────────────────────────────────────────────────────


def load_svmp(root: Path) -> dict:
    """Bank + anchors + tensors + judge + margins -> aligned per-context
    arrays at EVERY store layer (multi-layer extension of RD.load_pilot)."""
    bank = json.loads((root / "manifests" / "svmp_bank.json").read_text(encoding="utf-8"))
    contexts = bank["contexts"]
    pairs = bank["pairs"]
    assert tuple(bank["pair_classes"]) == PAIR_CLASSES, bank["pair_classes"]
    ctx_ids = [c["id"] for c in contexts]
    ctx_pos = {cid: i for i, cid in enumerate(ctx_ids)}
    assert len(ctx_pos) == len(contexts), "duplicate context ids in bank"
    cap = bank["capture_filenames"]

    vc_store = torch.load(root / "vc_store" / cap["vc"], map_location="cpu", weights_only=False)
    layers = [int(x) for x in vc_store["layers"]]
    primary = PRIMARY_LAYER if PRIMARY_LAYER in layers else layers[-1]
    if primary != PRIMARY_LAYER:
        logger.warning(
            "[load] PRIMARY_LAYER %d absent (layers=%s) — using %d (tiny store?)",
            PRIMARY_LAYER,
            layers,
            primary,
        )
    store_pos = {cid: i for i, cid in enumerate(vc_store["context_ids"])}
    assert set(ctx_ids) <= set(store_pos), sorted(set(ctx_ids) - set(store_pos))[:5]
    rows = [store_pos[cid] for cid in ctx_ids]
    vc = {
        layer: vc_store["vc"][rows][:, li].numpy().astype(np.float64)
        for li, layer in enumerate(layers)
    }
    d_model = vc[primary].shape[1]

    anchors = RD._read_jsonl(root / "anchors" / f"anchors_{CELL}.jsonl")
    va_store = torch.load(root / "va_store" / cap["va"], map_location="cpu", weights_only=False)
    va_layers = [int(x) for x in va_store["layers"]]
    assert set(layers) <= set(va_layers), (layers, va_layers)
    index = va_store["index"]
    assert len(index) == len(anchors), (len(index), len(anchors))
    empty = set(va_store["empty_rows"])
    k = max(int(r["draw"]) for r in index) + 1
    n = len(ctx_ids)
    tail = {layer: np.zeros((n, k, d_model), dtype=np.float32) for layer in layers}
    span = {layer: np.zeros((n, k, d_model), dtype=np.float32) for layer in layers}
    valid = np.zeros((n, k), dtype=bool)
    texts: dict[str, list[str | None]] = {cid: [None] * k for cid in ctx_ids}
    va_tail = va_store["va_tail_incl"]
    va_span = va_store["va_span"]
    for row_i, (rec, arow) in enumerate(zip(index, anchors)):
        assert rec["context_id"] == arow["context_id"] and rec["draw"] == arow["draw"], (
            row_i,
            rec["context_id"],
            arow["context_id"],
        )
        cid, dr = rec["context_id"], int(rec["draw"])
        ci = ctx_pos[cid]
        texts[cid][dr] = arow["text"]
        if row_i in empty:
            continue
        for layer in layers:
            vli = va_layers.index(layer)
            tail[layer][ci, dr] = va_tail[row_i, vli].float().numpy()
            span[layer][ci, dr] = va_span[row_i, vli].float().numpy()
        valid[ci, dr] = True

    judge = json.loads((root / "judge" / "judge_scores.json").read_text(encoding="utf-8"))
    idmap = json.loads((root / "judge" / "judge_id_map.json").read_text(encoding="utf-8"))
    margins = json.loads((root / "margin" / "margins.json").read_text(encoding="utf-8"))
    raw_path = root / "judge" / f"judge_raw_{CELL}.json"
    api_refusals = _api_refusal_counts(raw_path, idmap["reverse"], bool(judge.get("dry_run")))

    return {
        "bank": bank,
        "contexts": contexts,
        "ctx_ids": ctx_ids,
        "ctx_pos": ctx_pos,
        "pairs": pairs,
        "layers": layers,
        "primary": primary,
        "vc": vc,
        "tail": tail,
        "span": span,
        "valid": valid,
        "texts": texts,
        "k": k,
        "judge": judge,
        "idmap": idmap,
        "margins": margins,
        "api_refusals": api_refusals,  # {context_id: count} | None (dry-run, no save_raw)
    }


def _api_refusal_counts(
    raw_path: Path, alias_to_ctx: dict[str, str], dry_run: bool
) -> dict[str, int] | None:
    """Per-context API-refusal draw counts from the judge save_raw artifact.

    judge_scores.json does not carry the rule-28 api-refusal class; recover it
    from save_raw's ``all_scores`` rows (custom_id ``{alias}--d{draw}__...``)
    via ``batch_judge.is_api_refusal_error_dict``. Absent save_raw is legal
    ONLY under a dry-run judge (tiny mode) — a live-judge run without it
    fails loud rather than silently reporting zero refusals.
    """
    if not raw_path.is_file():
        if dry_run:
            return None
        raise FileNotFoundError(
            f"judge save_raw missing at {raw_path} but judge_scores.json says dry_run=False — "
            "api-refusal counts (rule 28) are unrecoverable; re-stage the judge raw artifact"
        )
    from explore_persona_space.eval.batch_judge import is_api_refusal_error_dict

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = defaultdict(int)
    for custom_id, parsed in raw.get("all_scores", {}).items():
        item_id = custom_id.rsplit("__", 2)[0]
        alias = item_id.split("--d")[0]
        ctx = alias_to_ctx.get(alias)
        if ctx is None:
            continue
        if isinstance(parsed, dict) and parsed.get("error") and is_api_refusal_error_dict(parsed):
            counts[ctx] += 1
    return dict(counts)


# ── batched statistics helpers ────────────────────────────────────────────


def _masked_means(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(n, K, d) draw stack + (n, K) validity -> (n, d) fp64 mean; NaN rows
    when a context has zero valid draws."""
    counts = valid.sum(axis=1).astype(np.float64)
    sums = np.einsum("nkd,nk->nd", stack.astype(np.float64), valid.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / counts[:, None]
    out[counts == 0] = np.nan
    return out


def _within_source_perms(rng: np.random.Generator, groups: list[np.ndarray], n: int) -> np.ndarray:
    """(B, n) permutation index matrix; each row permutes pair indices WITHIN
    each source group (plan §3 shuffled-pair null)."""
    perm = np.tile(np.arange(n), (N_NULL_DRAWS, 1))
    for g in groups:
        if len(g) < 2:
            continue
        block = np.tile(g, (N_NULL_DRAWS, 1))
        perm[:, g] = rng.permuted(block, axis=1)
    return perm


def _null_gather(c_mat: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """(n, n) cross-cos + (B, n) perms -> (B, n) null per-pair cosines
    (pair i's prediction scored against pair perm[b, i]'s observation)."""
    n = c_mat.shape[0]
    return c_mat[np.arange(n)[None, :], perm]


def _band_p95(null_vals: np.ndarray, idx: np.ndarray) -> float:
    """95th percentile of per-draw medians over the group ``idx``."""
    if len(idx) == 0:
        return float("nan")
    sub = null_vals[:, idx]
    with np.errstate(invalid="ignore"):
        meds = np.nanmedian(sub, axis=1)
    return float(np.nanpercentile(meds, 95))


def _weighted_median_rows(vals: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise weighted median of ``vals`` (B, m) with nonneg weights (B, m);
    NaN values get weight 0. Rows with zero total weight -> NaN."""
    finite = np.isfinite(vals)
    w = np.where(finite, w, 0.0)
    v = np.where(finite, vals, np.inf)  # NaNs sort last, weight 0
    order = np.argsort(v, axis=1)
    v_s = np.take_along_axis(v, order, axis=1)
    w_s = np.take_along_axis(w, order, axis=1)
    cw = np.cumsum(w_s, axis=1)
    tot = cw[:, -1]
    with np.errstate(invalid="ignore"):
        hit = cw >= (tot[:, None] / 2.0)
    idx = hit.argmax(axis=1)
    out = v_s[np.arange(len(v_s)), idx]
    out[tot <= 0] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def _counts_from_idx(idx: np.ndarray, m: int) -> np.ndarray:
    """(B, s) resample index matrix -> (B, m) integer count matrix."""
    b, s = idx.shape
    counts = np.zeros((b, m), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(b), s), idx.ravel()), 1.0)
    return counts


def _axis_loadings(x: np.ndarray, obs_f: np.ndarray, member: bool) -> np.ndarray:
    """Observed-flip-axis loadings: for MEMBERS, cos(x_i, S_f − obs_i) (LOO);
    for non-members, cos(x_i, S_f). S_f = sum of observed flip deltas."""
    if len(obs_f) == 0:
        return np.full(len(x), np.nan)
    s_f = obs_f.sum(axis=0)
    if member:
        if len(obs_f) < 2:
            return np.full(len(x), np.nan)
        ref = s_f[None, :] - obs_f  # (F, d) — x rows align with obs_f rows
        return RD.rowwise_cos(x, ref)
    return RD.rowwise_cos(x, np.tile(s_f, (len(x), 1)))


class _AxisBoot:
    """Batched bootstrap of the flip-axis contrasts with LOO-within-resample.

    Precomputes Gram/cross matrices once; each draw is index gathers + GEMMs:
      S*_b = Σ_j counts[b,j]·obs_f[j];  r̂*_{b,i} = S*_b − obs_f[i] (members),
      r̂*_b = S*_b (non-members; scale-invariant for cosines).
    """

    def __init__(self, obs_f: np.ndarray, obs_n: np.ndarray):
        self.f = len(obs_f)
        self.nn = len(obs_n)
        self.a_mat = obs_f @ obs_f.T if self.f else np.zeros((0, 0))
        self.diag_a = np.diag(self.a_mat).copy() if self.f else np.zeros(0)
        self.norm_obs_f = np.sqrt(np.maximum(self.diag_a, 0.0))
        self.obs_f = obs_f
        self.obs_n = obs_n
        self.o_mat = obs_n @ obs_f.T if (self.f and self.nn) else np.zeros((self.nn, self.f))
        self.norm_obs_n = np.linalg.norm(obs_n, axis=1) if self.nn else np.zeros(0)

    def draw_terms(self, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """counts (B, F) -> (CA, SS, denom_loo) shared across arms."""
        ca = counts @ self.a_mat  # (B, F): obs_i · S*_b
        ss = np.einsum("bj,jk,bk->b", counts, self.a_mat, counts)  # ||S*_b||^2
        denom_loo = np.sqrt(np.maximum(ss[:, None] - 2.0 * ca + self.diag_a[None, :], 0.0))
        return ca, ss, denom_loo

    def member_loadings(self, pred_f: np.ndarray, denom_loo: np.ndarray) -> np.ndarray:
        """(B, F) LOO loadings of member predictions against r̂*_{b,i}:
        cos(pred_i, S*_b − obs_i) = (pred_i·S*_b − pred_i·obs_i) / (‖pred_i‖·‖S*_b − obs_i‖)."""
        p_mat = pred_f @ self.obs_f.T  # (F, F): pred_i · obs_j
        cp = np.einsum("bj,ij->bi", self._counts_cache, p_mat)  # pred_i · S*_b
        num = cp - np.diag(p_mat)[None, :]
        norm_p = np.linalg.norm(pred_f, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / (norm_p[None, :] * denom_loo)

    # member/nonmember loadings need the counts matrix for the pred cross-term;
    # cache it per draw-set via set_counts() to keep the call signatures small.
    def set_counts(self, counts: np.ndarray) -> None:
        self._counts_cache = counts

    def nonmember_loadings(self, pred_n: np.ndarray, ss: np.ndarray) -> np.ndarray:
        """(B, Nn) loadings of non-member predictions against r̂*_b = S*_b."""
        if self.nn == 0 or self.f == 0:
            return np.zeros((len(ss), 0))
        q_mat = pred_n @ self.obs_f.T  # (Nn, F)
        cq = np.einsum("bj,ij->bi", self._counts_cache, q_mat)  # (B, Nn)
        norm_p = np.linalg.norm(pred_n, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return cq / (norm_p[None, :] * np.sqrt(np.maximum(ss, 0.0))[:, None])

    def obs_member_loadings(self, ca: np.ndarray, denom_loo: np.ndarray) -> np.ndarray:
        num = ca - self.diag_a[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / (self.norm_obs_f[None, :] * denom_loo)

    def obs_nonmember_loadings(self, ss: np.ndarray) -> np.ndarray:
        if self.nn == 0 or self.f == 0:
            return np.zeros((len(ss), 0))
        co = np.einsum("bj,ij->bi", self._counts_cache, self.o_mat)
        with np.errstate(invalid="ignore", divide="ignore"):
            return co / (self.norm_obs_n[None, :] * np.sqrt(np.maximum(ss, 0.0))[:, None])


def _ci(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    from scipy import stats as sps

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan"), int(ok.sum())
    rho, p = sps.spearmanr(x[ok], y[ok])
    return float(rho), float(p), int(ok.sum())


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, int]:
    """Partial Spearman rho(x, y | z): partial Pearson on ranks, t-approx p."""
    from scipy import stats as sps

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n = int(ok.sum())
    if n < 4:
        return float("nan"), float("nan"), n
    rx = sps.rankdata(x[ok])
    ry = sps.rankdata(y[ok])
    rz = sps.rankdata(z[ok])
    rxy = float(np.corrcoef(rx, ry)[0, 1])
    rxz = float(np.corrcoef(rx, rz)[0, 1])
    ryz = float(np.corrcoef(ry, rz)[0, 1])
    denom = math.sqrt(max((1 - rxz**2) * (1 - ryz**2), 0.0))
    if denom == 0:
        return float("nan"), float("nan"), n
    r = (rxy - rxz * ryz) / denom
    r = max(min(r, 1.0), -1.0)
    df = n - 3
    if df <= 0 or abs(r) >= 1.0:
        return float(r), float("nan"), n
    t = r * math.sqrt(df / (1 - r**2))
    p = 2 * float(sps.t.sf(abs(t), df))
    return float(r), p, n


def _axis_split_half(
    tail: np.ndarray, valid: np.ndarray, ai: np.ndarray, bi: np.ndarray, flip_mask: np.ndarray
) -> tuple[float, float, int]:
    """Split-half reliability of the flip AXIS r̂ (mean observed flip delta):
    per split, cos(mean of half-1 deltas over flip pairs, mean of half-2
    deltas). Mirrors RD.split_half_stats' rng conventions ([SPLIT_SEED, s])."""
    n_ctx, k, _ = tail.shape
    rs: list[float] = []
    n_ok_min = 0
    for s in range(N_AXIS_SPLITS):
        rng = np.random.default_rng([RD.SPLIT_SEED, s])
        scores = rng.random((n_ctx, k))
        h1 = np.zeros((n_ctx, tail.shape[2]))
        h2 = np.zeros_like(h1)
        ok_ctx = np.zeros(n_ctx, dtype=bool)
        for c in range(n_ctx):
            idx = np.flatnonzero(valid[c])
            if len(idx) < 2:
                continue
            order = idx[np.argsort(scores[c, idx])]
            half_a, half_b = order[: len(idx) // 2], order[len(idx) // 2 :]
            h1[c] = tail[c, half_a].mean(axis=0)
            h2[c] = tail[c, half_b].mean(axis=0)
            ok_ctx[c] = True
        ok_pair = ok_ctx[ai] & ok_ctx[bi] & flip_mask
        n_ok_min = int(ok_pair.sum()) if s == 0 else min(n_ok_min, int(ok_pair.sum()))
        if ok_pair.sum() < 1:
            continue
        axis1 = (h1[ai] - h1[bi])[ok_pair].mean(axis=0)
        axis2 = (h2[ai] - h2[bi])[ok_pair].mean(axis=0)
        r = RD.rowwise_cos(axis1[None, :], axis2[None, :])[0]
        if np.isfinite(r):
            rs.append(float(r))
    if not rs:
        return float("nan"), float("nan"), n_ok_min
    r_half = float(np.mean(rs))
    return r_half, RD.spearman_brown(r_half), n_ok_min


# ── compute ───────────────────────────────────────────────────────────────


def compute(data: dict, ridge_paths: dict[int, dict[str, Path]], *, tiny: bool) -> tuple:
    pairs = data["pairs"]
    ctx_pos = data["ctx_pos"]
    primary = data["primary"]
    layers = data["layers"]
    valid = data["valid"]
    judge_pc = data["judge"]["per_context"]
    margins_pc = data["margins"]["per_context"]
    api_ref = data["api_refusals"]
    n_pairs = len(pairs)
    ai = np.array([ctx_pos[p["a"]] for p in pairs])
    bi = np.array([ctx_pos[p["b"]] for p in pairs])
    classes = np.array([p["pair_class"] for p in pairs])
    sources = np.array([p["pair_source"] for p in pairs])
    unknown_src = sorted(set(sources.tolist()) - set(PAIR_SOURCES))
    assert not unknown_src, (
        "pair_source values outside PAIR_SOURCES — such pairs would silently stay "
        "identity-permuted in the within-source shuffled null",
        unknown_src,
    )
    fams = np.array([str(p["artifact_family_id"]) for p in pairs])

    # Per-layer pooled means + observed deltas.
    tail_mean = {layer: _masked_means(data["tail"][layer], valid) for layer in layers}
    span_mean = {layer: _masked_means(data["span"][layer], valid) for layer in layers}
    obs_tail = {layer: tail_mean[layer][ai] - tail_mean[layer][bi] for layer in layers}
    obs_span = {layer: span_mean[layer][ai] - span_mean[layer][bi] for layer in layers}

    # Per-layer per-arm predictions (mapped per context, delta per pair).
    dev = torch.device("cpu")
    apply_map = RD._import_apply_map()
    d_model = data["vc"][primary].shape[1]
    mapped: dict[int, dict[str, np.ndarray]] = {}
    for layer in layers:
        mapped[layer] = {}
        for arm in ARMS:
            payload = RD.load_ridge_payload(ridge_paths[layer][arm], d_model, arm)
            m = apply_map(payload, data["vc"][layer], dev)
            assert m.shape == data["vc"][layer].shape, (arm, layer, m.shape)
            mapped[layer][arm] = np.asarray(m, dtype=np.float64)
        mapped[layer]["arm_iddelta"] = data["vc"][layer]
    pred = {
        layer: {arm: mapped[layer][arm][ai] - mapped[layer][arm][bi] for arm in ALL_ARMS}
        for layer in layers
    }

    # Judge-derived per-context reads + flip classification.
    rate: list[float | None] = []
    flagged: list[bool] = []
    for cid in data["ctx_ids"]:
        j = judge_pc.get(cid, {})
        rate.append(j.get("refusal_rate"))
        flagged.append(bool(j.get("flag_low_valid", True)))
    frac_flagged = float(np.mean(flagged)) if flagged else 1.0
    judge_integrity_halt = frac_flagged > JUDGE_FLAG_FRAC_MAX

    flip = np.full(n_pairs, np.nan)
    for i in range(n_pairs):
        ra, rb = rate[ai[i]], rate[bi[i]]
        if ra is None or rb is None or flagged[ai[i]] or flagged[bi[i]]:
            continue
        flip[i] = float(ra) - float(rb)
    abs_flip = np.abs(flip)
    defined = np.isfinite(flip)
    flip_mask = defined & (abs_flip >= FLIP_HI)
    nonflip_mask = defined & (abs_flip <= NONFLIP_LO)
    mid_mask = defined & ~flip_mask & ~nonflip_mask
    flip_group = np.where(
        ~defined, "undefined", np.where(flip_mask, "flip", np.where(nonflip_mask, "nonflip", "mid"))
    )
    n_flip = int(flip_mask.sum())
    dichotomy_halted = n_flip < MIN_FLIP_PAIRS

    intended = np.isin(classes, INTENDED_FLIP_CLASSES)
    n_intended_defined = int((intended & defined).sum())
    manip_frac = (
        float((intended & flip_mask).sum() / n_intended_defined) if n_intended_defined else None
    )

    # Direction cosines per arm (primary layer; tail + span).
    cos_tail = {a: RD.rowwise_cos(pred[primary][a], obs_tail[primary]) for a in ALL_ARMS}
    cos_span = {a: RD.rowwise_cos(pred[primary][a], obs_span[primary]) for a in ALL_ARMS}
    norm_obs_t = np.linalg.norm(obs_tail[primary], axis=1)
    norm_obs_s = np.linalg.norm(obs_span[primary], axis=1)
    norm_pred = {a: np.linalg.norm(pred[primary][a], axis=1) for a in ALL_ARMS}
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_ratio = {a: norm_pred[a] / norm_obs_t for a in ALL_ARMS}

    # Flip-axis loadings (LOO for members; full-mean axis for the rest).
    # A flip pair with a non-finite observed tail delta would poison the LOO
    # axis + the bootstrap that reuses the UNfiltered flip rows — counted +
    # asserted, never silently dropped (r1 review g2).
    obs_f = obs_tail[primary][flip_mask]
    n_degenerate_flip = int(np.sum(~np.all(np.isfinite(obs_f), axis=1))) if len(obs_f) else 0
    assert n_degenerate_flip == 0, (
        f"{n_degenerate_flip} flip pair(s) with a non-finite observed tail delta — "
        "these would corrupt the LOO flip axis and its bootstrap"
    )
    axis_obs = np.full(n_pairs, np.nan)
    axis_pred = {a: np.full(n_pairs, np.nan) for a in ALL_ARMS}
    if len(obs_f) >= 2:
        fi = np.flatnonzero(flip_mask)
        axis_obs[fi] = _axis_loadings(obs_tail[primary][fi], obs_f, member=True)
        rest = np.flatnonzero(~flip_mask)
        axis_obs[rest] = _axis_loadings(obs_tail[primary][rest], obs_f, member=False)
        for a in ALL_ARMS:
            axis_pred[a][fi] = _axis_loadings(pred[primary][a][fi], obs_f, member=True)
            axis_pred[a][rest] = _axis_loadings(pred[primary][a][rest], obs_f, member=False)

    # Shuffled-pair null (identical permutation set for every arm/class).
    rng_null = np.random.default_rng(NULL_SEED)
    src_groups = [np.flatnonzero(sources == s) for s in PAIR_SOURCES]
    perm = _within_source_perms(rng_null, src_groups, n_pairs)
    null_bands: dict[str, dict] = {}
    for a in ALL_ARMS:
        c_mat = RD.cross_cos(pred[primary][a], obs_tail[primary])
        nv = _null_gather(c_mat, perm)
        null_bands[a] = {
            "S1_null_band_p95": _f(_band_p95(nv, np.flatnonzero(flip_mask))),
            "S2_null_band_p95": _f(_band_p95(nv, np.flatnonzero(nonflip_mask))),
            "by_class_p95": {
                cls: _f(_band_p95(nv, np.flatnonzero(classes == cls))) for cls in PAIR_CLASSES
            },
        }

    # Registered scalars per arm.
    def _med(vals: np.ndarray, mask: np.ndarray) -> float:
        sub = vals[mask]
        with np.errstate(invalid="ignore"):
            return float(np.nanmedian(sub)) if len(sub) else float("nan")

    per_arm: dict[str, dict] = {}
    fi = np.flatnonzero(flip_mask)
    ni = np.flatnonzero(nonflip_mask)
    rng_boot = np.random.default_rng(BOOT_SEED)
    for a in ALL_ARMS:
        s1 = _med(cos_tail[a], flip_mask)
        s2 = _med(cos_tail[a], nonflip_mask)
        s3 = _med(axis_pred[a], flip_mask)
        s3_non = _med(axis_pred[a], nonflip_mask)
        s3_contrast = s3 - s3_non if np.isfinite(s3) and np.isfinite(s3_non) else float("nan")
        entry = {
            "S1": _f(s1),
            "S2": _f(s2),
            "S1_minus_S2": _f(s1 - s2) if np.isfinite(s1) and np.isfinite(s2) else None,
            "S3": _f(s3),
            "S3_nonflip_loading_median": _f(s3_non),
            "S3_contrast": _f(s3_contrast),
            "S1_span": _f(_med(cos_span[a], flip_mask)),
            "S2_span": _f(_med(cos_span[a], nonflip_mask)),
            "cos_median_by_class": {c: _f(_med(cos_tail[a], classes == c)) for c in PAIR_CLASSES},
            "cos_median_by_class_span": {
                c: _f(_med(cos_span[a], classes == c)) for c in PAIR_CLASSES
            },
            "norm_ratio_median_by_class": {
                c: _f(_med(norm_ratio[a], classes == c)) for c in PAIR_CLASSES
            },
            **null_bands[a],
        }
        per_arm[a] = entry

    # Observed-side S3 twin (arm-independent).
    s3_obs = _med(axis_obs, flip_mask)
    s3_obs_non = _med(axis_obs, nonflip_mask)
    s3_obs_contrast = (
        s3_obs - s3_obs_non if np.isfinite(s3_obs) and np.isfinite(s3_obs_non) else float("nan")
    )

    # Bootstrap CIs (pair-level, stratified by flip/nonflip group; LOO within
    # each resample via the Gram machinery). Family-clustered companion for
    # the registered arm.
    boot: dict[str, dict] = {a: {} for a in ALL_ARMS}
    boot_obs: dict = {}
    fam_boot: dict = {}
    if len(fi) >= 2 and len(obs_f) >= 2:
        ab = _AxisBoot(obs_tail[primary][fi], obs_tail[primary][ni])
        idx_f = rng_boot.integers(0, len(fi), size=(N_BOOT, len(fi)))
        idx_n = rng_boot.integers(0, len(ni), size=(N_BOOT, len(ni))) if len(ni) else None
        counts_f = _counts_from_idx(idx_f, len(fi))
        ab.set_counts(counts_f)
        ca, ss, denom_loo = ab.draw_terms(counts_f)
        lo_m = ab.obs_member_loadings(ca, denom_loo)
        lo_n = ab.obs_nonmember_loadings(ss)
        med_obs_f = np.nanmedian(np.take_along_axis(lo_m, idx_f, axis=1), axis=1)
        if idx_n is not None and lo_n.shape[1]:
            med_obs_n = np.nanmedian(np.take_along_axis(lo_n, idx_n, axis=1), axis=1)
            boot_obs["S3_obs_contrast_ci"] = [_f(v) for v in _ci(med_obs_f - med_obs_n)]
        boot_obs["S3_obs_ci"] = [_f(v) for v in _ci(med_obs_f)]
        for a in ALL_ARMS:
            lp_m = ab.member_loadings(pred[primary][a][fi], denom_loo)
            med_f = np.nanmedian(np.take_along_axis(lp_m, idx_f, axis=1), axis=1)
            boot[a]["S3_ci"] = [_f(v) for v in _ci(med_f)]
            if idx_n is not None and len(ni):
                lp_n = ab.nonmember_loadings(pred[primary][a][ni], ss)
                med_n = np.nanmedian(np.take_along_axis(lp_n, idx_n, axis=1), axis=1)
                boot[a]["S3_contrast_ci"] = [_f(v) for v in _ci(med_f - med_n)]
            # S1 / S2 / S1-S2 value bootstrap (cos values fixed per pair).
            cf = cos_tail[a][fi]
            s1_draws = np.nanmedian(cf[idx_f], axis=1)
            boot[a]["S1_ci"] = [_f(v) for v in _ci(s1_draws)]
            if idx_n is not None and len(ni):
                cn = cos_tail[a][ni]
                s2_draws = np.nanmedian(cn[idx_n], axis=1)
                boot[a]["S2_ci"] = [_f(v) for v in _ci(s2_draws)]
                boot[a]["S1_minus_S2_ci"] = [_f(v) for v in _ci(s1_draws - s2_draws)]
        # Family-clustered companion (registered arm): resample families.
        fam_f = fams[fi]
        fam_n = fams[ni] if len(ni) else np.array([])
        uf = sorted(set(fam_f))
        m_f = np.stack([(fam_f == u).astype(np.float64) for u in uf]) if uf else None
        if m_f is not None and len(uf) >= 2:
            fam_idx = rng_boot.integers(0, len(uf), size=(N_BOOT, len(uf)))
            fam_counts = _counts_from_idx(fam_idx, len(uf)) @ m_f  # (B, F)
            ab.set_counts(fam_counts)
            ca2, ss2, denom2 = ab.draw_terms(fam_counts)
            lp_m2 = ab.member_loadings(pred[primary][REGISTERED_ARM][fi], denom2)
            lo_m2 = ab.obs_member_loadings(ca2, denom2)
            medp_f = _weighted_median_rows(lp_m2, fam_counts)
            medo_f = _weighted_median_rows(lo_m2, fam_counts)
            c_reg_f = np.tile(cos_tail[REGISTERED_ARM][fi], (N_BOOT, 1))
            s1_fam = _weighted_median_rows(c_reg_f, fam_counts)
            fam_boot = {
                "S1_ci": [_f(v) for v in _ci(s1_fam)],
                "n_flip_families": len(uf),
            }
            if len(ni):
                un = sorted(set(fam_n))
                m_n = np.stack([(fam_n == u).astype(np.float64) for u in un])
                famn_idx = rng_boot.integers(0, len(un), size=(N_BOOT, len(un)))
                famn_counts = _counts_from_idx(famn_idx, len(un)) @ m_n  # (B, Nn)
                lp_n2 = ab.nonmember_loadings(pred[primary][REGISTERED_ARM][ni], ss2)
                lo_n2 = ab.obs_nonmember_loadings(ss2)
                medp_n = _weighted_median_rows(lp_n2, famn_counts)
                medo_n = _weighted_median_rows(lo_n2, famn_counts)
                c_reg_n = np.tile(cos_tail[REGISTERED_ARM][ni], (N_BOOT, 1))
                s2_fam = _weighted_median_rows(c_reg_n, famn_counts)
                fam_boot["S3_contrast_ci"] = [_f(v) for v in _ci(medp_f - medp_n)]
                fam_boot["S3_obs_contrast_ci"] = [_f(v) for v in _ci(medo_f - medo_n)]
                fam_boot["S1_minus_S2_ci"] = [_f(v) for v in _ci(s1_fam - s2_fam)]
                fam_boot["n_nonflip_families"] = len(un)

    # S4: Spearman rho(|flip|, cos) over defined pairs + length-partialled twin.
    ans_len = np.full(len(data["ctx_ids"]), np.nan)
    for cid, i in ctx_pos.items():
        lens = [len(t) for t, v in zip(data["texts"][cid], valid[i]) if v and t is not None]
        if lens:
            ans_len[i] = float(np.mean(lens))
    len_delta = ans_len[ai] - ans_len[bi]
    abs_len_delta = np.abs(len_delta)
    s4 = {}
    s4_partial = {}
    length_battery: dict = {}
    ok_len = np.isfinite(abs_flip) & np.isfinite(abs_len_delta)
    coll_r = (
        float(np.corrcoef(abs_flip[ok_len], abs_len_delta[ok_len])[0, 1])
        if ok_len.sum() >= 3
        else float("nan")
    )
    gate_tripped = bool(np.isfinite(coll_r) and abs(coll_r) >= LEN_COLLINEARITY_GATE)
    for a in ALL_ARMS:
        rho, p, n_s4 = _spearman(abs_flip, cos_tail[a])
        boot[a]["S4"] = {"rho": _f(rho), "p": _f(p), "n": n_s4}
        rho_s, p_s, n_ss = _spearman(abs_flip, cos_span[a])
        boot[a]["S4_span"] = {"rho": _f(rho_s), "p": _f(p_s), "n": n_ss}
        rp, pp, n_p = _partial_spearman(abs_flip, cos_tail[a], abs_len_delta)
        boot[a]["S4_len_partial"] = {
            "rho": _f(rp),
            "p": _f(pp),
            "n": n_p,
            "collinearity_r": _f(coll_r),
            "gate_tripped": gate_tripped,
            "authoritative": "tercile" if gate_tripped else "partial",
        }
    s4 = boot[REGISTERED_ARM]["S4"]
    s4_partial = boot[REGISTERED_ARM]["S4_len_partial"]
    # Tercile fallback (computed always; authoritative when the gate trips).
    terciles = []
    if ok_len.sum() >= 6:
        qs = np.nanquantile(abs_len_delta[ok_len], [1 / 3, 2 / 3])
        bins = np.digitize(abs_len_delta, qs)
        for t in range(3):
            tm = ok_len & (bins == t)
            terciles.append(
                {
                    "tercile": t,
                    "n": int(tm.sum()),
                    "cos_median_flip": _f(_med(cos_tail[REGISTERED_ARM], tm & flip_mask)),
                    "cos_median_nonflip": _f(_med(cos_tail[REGISTERED_ARM], tm & nonflip_mask)),
                    "n_flip": int((tm & flip_mask).sum()),
                    "n_nonflip": int((tm & nonflip_mask).sum()),
                }
            )
    length_battery = {
        "collinearity_r": _f(coll_r),
        "gate": LEN_COLLINEARITY_GATE,
        "gate_tripped": gate_tripped,
        "ans_len_delta_median_by_class": {
            c: _f(_med(len_delta, classes == c)) for c in PAIR_CLASSES
        },
        "abs_ans_len_delta_median_by_class": {
            c: _f(_med(abs_len_delta, classes == c)) for c in PAIR_CLASSES
        },
        "terciles": terciles,
    }

    # Verdict lattice (registered arm; plan §3 — advisory bands, no abort gates).
    verdict = None
    lattice_inputs: dict = {}
    if not dichotomy_halted and not judge_integrity_halt:
        s1_reg = per_arm[REGISTERED_ARM]["S1"]
        band = per_arm[REGISTERED_ARM]["S1_null_band_p95"]
        s3c_ci = boot[REGISTERED_ARM].get("S3_contrast_ci")
        s3o_ci = boot_obs.get("S3_obs_contrast_ci")
        s1_gt = s1_reg is not None and band is not None and s1_reg > band
        s3c_pos = bool(s3c_ci and s3c_ci[0] is not None and s3c_ci[0] > 0)
        s3o_pos = bool(s3o_ci and s3o_ci[0] is not None and s3o_ci[0] > 0)
        lattice_inputs = {
            "S1": s1_reg,
            "S1_null_band_p95": band,
            "S1_gt_band": s1_gt,
            "S3_contrast_ci": s3c_ci,
            "S3_contrast_ci_excludes_0_above": s3c_pos,
            "S3_obs_contrast_ci": s3o_ci,
            "S3_obs_contrast_ci_excludes_0_above": s3o_pos,
        }
        if s1_gt and s3c_pos:
            verdict = "decision-transported"
        elif s1_gt and not s3c_pos and s3o_pos:
            verdict = "surface-content-only"
        elif s1_gt and not s3c_pos and not s3o_pos:
            verdict = "axis-not-established"
        elif not s1_gt and s3c_pos:
            verdict = "axis-only-weak"
        else:
            verdict = "no-transport"

    # Retrieval: per-context kNN (mapped arms + identity + LOO identity+bias).
    retrieval_ctx: dict = {}
    tm = tail_mean[primary]
    finite_ctx = np.all(np.isfinite(tm), axis=1)
    ctx_src = np.array([_ctx_source(c) for c in data["contexts"]])
    ib_pred = _loo_identity_bias(data["vc"][primary], tm, finite_ctx)
    ctx_preds = {a: mapped[primary][a] for a in ALL_ARMS}
    ctx_preds["idbias_loo"] = ib_pred
    for name, cp in ctx_preds.items():
        entry: dict = {}
        pools = {"full": np.flatnonzero(finite_ctx)}
        for s in PAIR_SOURCES:
            pools[s] = np.flatnonzero(finite_ctx & (ctx_src == s))
        for pool_name, idx in pools.items():
            if len(idx) < 2 or not np.all(np.isfinite(cp[idx])):
                entry[pool_name] = None
                continue
            entry[pool_name] = {
                m: knn_retrieval(cp[idx], tm[idx], ks=(1, 5), metric=m)
                for m in ("cosine", "euclidean")
            }
        retrieval_ctx[name] = entry
    # Pair-delta rank read per arm (full + within-source pools; per class).
    retrieval_pair: dict = {}
    finite_pair = np.all(np.isfinite(obs_tail[primary]), axis=1)
    for a in ALL_ARMS:
        c_mat = RD.cross_cos(pred[primary][a], obs_tail[primary])
        entry = {}
        pools = {"full": np.flatnonzero(finite_pair)}
        for s in PAIR_SOURCES:
            pools[s] = np.flatnonzero(finite_pair & (sources == s))
        for pool_name, idx in pools.items():
            if len(idx) < 2:
                entry[pool_name] = None
                continue
            sub = np.nan_to_num(c_mat[np.ix_(idx, idx)], nan=-np.inf)
            order = np.argsort(-sub, axis=1)
            ranks = np.argmax(order == np.arange(len(idx))[:, None], axis=1) + 1
            by_class = {}
            for cls in PAIR_CLASSES:
                cm = classes[idx] == cls
                if cm.sum() == 0:
                    continue
                by_class[cls] = {
                    "n": int(cm.sum()),
                    "acc_at_1": _f(float((ranks[cm] == 1).mean())),
                    "median_rank": _f(float(np.median(ranks[cm]))),
                }
            entry[pool_name] = {
                "n_pool": int(len(idx)),
                "chance_at_1": _f(1.0 / len(idx)),
                "acc_at_1": _f(float((ranks == 1).mean())),
                "median_rank": _f(float(np.median(ranks))),
                "by_class": by_class,
            }
        retrieval_pair[a] = entry

    # Calibration slopes + flip-vs-nonflip contrast (per arm).
    calibration: dict = {}
    for a in ALL_ARMS:
        by_class = {
            c: _f(RD.through_origin_slope(norm_pred[a][classes == c], norm_obs_t[classes == c]))
            for c in PAIR_CLASSES
        }
        slope_all = RD.through_origin_slope(norm_pred[a], norm_obs_t)
        slope_flip = RD.through_origin_slope(norm_pred[a][flip_mask], norm_obs_t[flip_mask])
        slope_non = RD.through_origin_slope(norm_pred[a][nonflip_mask], norm_obs_t[nonflip_mask])
        contrast = (
            slope_flip - slope_non
            if np.isfinite(slope_flip) and np.isfinite(slope_non)
            else float("nan")
        )
        contrast_ci = None
        if len(fi) >= 2 and len(ni) >= 2:
            pn_f = (norm_pred[a] * norm_obs_t)[fi]
            oo_f = (norm_obs_t**2)[fi]
            pn_n = (norm_pred[a] * norm_obs_t)[ni]
            oo_n = (norm_obs_t**2)[ni]
            rng_s = np.random.default_rng([BOOT_SEED, 7, len(a)])
            if_f = rng_s.integers(0, len(fi), size=(N_BOOT, len(fi)))
            if_n = rng_s.integers(0, len(ni), size=(N_BOOT, len(ni)))
            cf = _counts_from_idx(if_f, len(fi))
            cn = _counts_from_idx(if_n, len(ni))
            with np.errstate(invalid="ignore", divide="ignore"):
                sl_f = (cf @ pn_f) / (cf @ oo_f)
                sl_n = (cn @ pn_n) / (cn @ oo_n)
            contrast_ci = [_f(v) for v in _ci(sl_f - sl_n)]
        calibration[a] = {
            "slope_all": _f(slope_all),
            "slope_by_class": by_class,
            "slope_flip": _f(slope_flip),
            "slope_nonflip": _f(slope_non),
            "slope_flip_minus_nonflip": _f(contrast),
            "slope_flip_minus_nonflip_ci": contrast_ci,
        }

    # Stratified (source x slot == pair_class) companion reads (registered arm).
    stratified: dict = {}
    for cls in PAIR_CLASSES:
        cm = classes == cls
        stratified[cls] = {
            "source": str(sources[cm][0]) if cm.sum() else None,
            "n": int(cm.sum()),
            "n_flip": int((cm & flip_mask).sum()),
            "cos_median_flip": _f(_med(cos_tail[REGISTERED_ARM], cm & flip_mask)),
            "cos_median_all": _f(_med(cos_tail[REGISTERED_ARM], cm)),
            "axis_pred_median_flip": _f(_med(axis_pred[REGISTERED_ARM], cm & flip_mask)),
            "axis_obs_median_flip": _f(_med(axis_obs, cm & flip_mask)),
            "null_band_p95": null_bands[REGISTERED_ARM]["by_class_p95"][cls],
        }

    # Split-half reliability (per pair, primary layer) + flip-axis split-half.
    r_half, r10, noise_norm = RD.split_half_stats(data["tail"][primary], valid, ai, bi)
    n_degenerate = int(np.sum(~np.isfinite(r_half)))
    axis_r_half, axis_r10, axis_min_ok = _axis_split_half(
        data["tail"][primary], valid, ai, bi, flip_mask
    )

    # L14/L26 twin table.
    twin_layers: dict = {}
    for layer in layers:
        if layer == primary:
            continue
        entry = {}
        for a in ALL_ARMS:
            ct = RD.rowwise_cos(pred[layer][a], obs_tail[layer])
            c_mat = RD.cross_cos(pred[layer][a], obs_tail[layer])
            fp = np.all(np.isfinite(obs_tail[layer]), axis=1)
            idx = np.flatnonzero(fp)
            acc1 = None
            if len(idx) >= 2:
                sub = np.nan_to_num(c_mat[np.ix_(idx, idx)], nan=-np.inf)
                order = np.argsort(-sub, axis=1)
                ranks = np.argmax(order == np.arange(len(idx))[:, None], axis=1) + 1
                acc1 = _f(float((ranks == 1).mean()))
            entry[a] = {
                "cos_median_by_class": {c: _f(_med(ct, classes == c)) for c in PAIR_CLASSES},
                "cos_median_flip": _f(_med(ct, flip_mask)),
                "cos_median_nonflip": _f(_med(ct, nonflip_mask)),
                "pair_acc_at_1_full": acc1,
            }
        twin_layers[str(layer)] = entry

    # Per-class judge accounting (n_valid + api refusals).
    ctx_class: dict[str, str] = {}
    ctx_pair: dict[str, tuple] = {}
    for p in pairs:
        for side in ("a", "b"):
            ctx_class[p[side]] = p["pair_class"]
            ctx_pair[p[side]] = (p["pair_id"], side)
    n_valid_by_class: dict[str, int] = defaultdict(int)
    n_api_by_class: dict[str, int | None] = {}
    for cls in PAIR_CLASSES:
        n_api_by_class[cls] = 0 if api_ref is not None else None
    for cid in data["ctx_ids"]:
        cls = ctx_class.get(cid)
        if cls is None:
            continue
        n_valid_by_class[cls] += int(judge_pc.get(cid, {}).get("n_valid", 0))
        if api_ref is not None:
            n_api_by_class[cls] = (n_api_by_class[cls] or 0) + int(api_ref.get(cid, 0))

    # ── rows ──────────────────────────────────────────────────────────────
    rows: list[dict] = []
    for i, p in enumerate(pairs):
        ja = judge_pc.get(p["a"], {})
        jb = judge_pc.get(p["b"], {})
        ma = margins_pc.get(p["a"], {})
        mb = margins_pc.get(p["b"], {})
        row = {
            "pair_id": p["pair_id"],
            "pair_class": p["pair_class"],
            "pair_source": p["pair_source"],
            "artifact_family_id": p["artifact_family_id"],
            "carrier": p["carrier"],
            "value_a": p["value_a"],
            "value_b": p["value_b"],
            "context_a": p["a"],
            "context_b": p["b"],
            "changed_tokens": p.get("changed_tokens"),
            "word_a": p.get("word_a"),
            "word_b": p.get("word_b"),
            "n_draws_a": int(valid[ai[i]].sum()),
            "n_draws_b": int(valid[bi[i]].sum()),
            "ans_len_a": _f(ans_len[ai[i]]),
            "ans_len_b": _f(ans_len[bi[i]]),
            "ans_len_delta": _f(len_delta[i]),
            "refusal_rate_a": _f(ja.get("refusal_rate")),
            "refusal_rate_b": _f(jb.get("refusal_rate")),
            "n_valid_a": int(ja.get("n_valid", 0)),
            "n_valid_b": int(jb.get("n_valid", 0)),
            "flag_low_valid_a": bool(ja.get("flag_low_valid", True)),
            "flag_low_valid_b": bool(jb.get("flag_low_valid", True)),
            "n_api_refusal_a": (int(api_ref.get(p["a"], 0)) if api_ref is not None else None),
            "n_api_refusal_b": (int(api_ref.get(p["b"], 0)) if api_ref is not None else None),
            "flip": _f(flip[i]),
            "abs_flip": _f(abs_flip[i]),
            "flip_group": str(flip_group[i]),
            "margin_a": _f(ma.get("margin")),
            "margin_b": _f(mb.get("margin")),
            "margin_delta": (
                _f(float(ma["margin"]) - float(mb["margin"]))
                if ma.get("margin") is not None and mb.get("margin") is not None
                else None
            ),
            "norm_obs_tail": _f(norm_obs_t[i]),
            "norm_obs_span": _f(norm_obs_s[i]),
            "r_half": _f(r_half[i]),
            "r10": _f(r10[i]),
            "noise_norm": _f(noise_norm[i]),
            "axis_cos_obs": _f(axis_obs[i]),
            "axis_cos_pred": _f(axis_pred[REGISTERED_ARM][i]),
        }
        for a in ALL_ARMS:
            row[f"cos_{a}"] = _f(cos_tail[a][i])
            row[f"cos_span_{a}"] = _f(cos_span[a][i])
            row[f"norm_pred_{a}"] = _f(norm_pred[a][i])
            row[f"norm_ratio_{a}"] = _f(norm_ratio[a][i])
            row[f"axis_cos_pred_{a}"] = _f(axis_pred[a][i])
        assert tuple(row) == PERPAIR_KEYS, (i, tuple(row))
        rows.append(row)

    fwd = data["idmap"]["forward"]
    ctx_rows: list[dict] = []
    for cid in data["ctx_ids"]:
        i = ctx_pos[cid]
        c = data["contexts"][i]
        j = judge_pc.get(cid, {})
        m = margins_pc.get(cid, {})
        pid, side = ctx_pair.get(cid, (None, None))
        crow = {
            "context_id": cid,
            "alias": fwd.get(cid),
            "pair_id": pid,
            "side": side,
            "pair_class": ctx_class.get(cid),
            "pair_source": _ctx_source(c),
            "artifact_family_id": _pair_field(pairs, pid, "artifact_family_id"),
            "value_id": c.get("value_id"),
            "carrier": c.get("carrier"),
            "n_draws_captured": int(valid[i].sum()),
            "n_valid": int(j.get("n_valid", 0)),
            "n_refused": int(j.get("n_refused", 0)),
            "n_dropped": int(j.get("n_dropped", 0)),
            "n_api_refusal": (int(api_ref.get(cid, 0)) if api_ref is not None else None),
            "flag_low_valid": bool(j.get("flag_low_valid", True)),
            "refusal_rate": _f(j.get("refusal_rate")),
            "margin": _f(m.get("margin")),
            "pos_mean_ln_logp": _f(m.get("pos_mean_ln_logp")),
            "n_pos": m.get("n_pos"),
            "n_neg": m.get("n_neg"),
            "ans_len_mean": _f(ans_len[i]),
        }
        assert tuple(crow) == PERCONTEXT_KEYS, tuple(crow)
        ctx_rows.append(crow)

    n_by_class = {c: int((classes == c).sum()) for c in PAIR_CLASSES}
    reg = per_arm[REGISTERED_ARM]
    # Effective n per registered statistic (nanmedian shrinks denominators
    # silently on NaN members — disclosed here; r1 review g2 minor).
    reg_effective_n = {
        "S1": int(np.isfinite(cos_tail[REGISTERED_ARM][flip_mask]).sum()),
        "S2": int(np.isfinite(cos_tail[REGISTERED_ARM][nonflip_mask]).sum()),
        "S3": int(np.isfinite(axis_pred[REGISTERED_ARM][flip_mask]).sum()),
        "S3_nonflip": int(np.isfinite(axis_pred[REGISTERED_ARM][nonflip_mask]).sum()),
        "S3_obs": int(np.isfinite(axis_obs[flip_mask]).sum()),
        "S3_obs_nonflip": int(np.isfinite(axis_obs[nonflip_mask]).sum()),
    }
    summary = {
        "issue": ISSUE,
        "cell": CELL,
        "tiny": tiny,
        "ridge_fabricated": tiny,
        "n_contexts": len(data["ctx_ids"]),
        "n_pairs": n_pairs,
        "n_pairs_by_class": n_by_class,
        "k_draws": data["k"],
        "d_model": d_model,
        "layers": {"primary": primary, "twins": [x for x in layers if x != primary]},
        "arms": list(ALL_ARMS),
        "registered_arm": REGISTERED_ARM,
        "ridge_paths": {
            str(layer): {a: str(p) for a, p in ridge_paths[layer].items()} for layer in layers
        },
        "judge": {
            "dry_run": bool(data["judge"].get("dry_run")),
            "judge_model": data["judge"].get("judge_model"),
            "min_valid_draws": data["judge"].get("min_valid_draws"),
            "refused_threshold": data["judge"].get("refused_threshold"),
            "n_dropped_total": data["judge"].get("n_dropped_total"),
            "frac_flagged_low_valid": _f(frac_flagged),
            "judge_integrity_halt": judge_integrity_halt,
            "n_valid_by_class": dict(n_valid_by_class),
            "n_api_refusal_by_class": n_api_by_class,
            "n_api_refusal_total": (int(sum(api_ref.values())) if api_ref is not None else None),
            "api_refusal_source": (
                f"judge_raw_{CELL}.json all_scores via batch_judge.is_api_refusal_error_dict"
                if api_ref is not None
                else "unavailable — dry-run judge, no save_raw artifact"
            ),
        },
        "flip_groups": {
            "flip_hi": FLIP_HI,
            "nonflip_lo": NONFLIP_LO,
            "n_flip": n_flip,
            "n_nonflip": int(nonflip_mask.sum()),
            "n_mid": int(mid_mask.sum()),
            "n_undefined": int((~defined).sum()),
            "n_intended_flip_pairs": int(intended.sum()),
            "n_intended_defined": n_intended_defined,
            "manipulation_check_frac_intended_flipped": _f(manip_frac),
        },
        "halts": {
            "dichotomy_halted": dichotomy_halted,
            "min_flip_pairs": MIN_FLIP_PAIRS,
            "judge_integrity_halt": judge_integrity_halt,
            "judge_flag_frac_max": JUDGE_FLAG_FRAC_MAX,
            "reasons": [
                r
                for r, on in (
                    (f"n_flip={n_flip} < {MIN_FLIP_PAIRS} — S4-only (run-abort)", dichotomy_halted),
                    (
                        f"frac_flagged={frac_flagged:.3f} > {JUDGE_FLAG_FRAC_MAX} — judge integrity",
                        judge_integrity_halt,
                    ),
                )
                if on
            ],
        },
        "registered": {
            "arm": REGISTERED_ARM,
            "layer": primary,
            "pooling": "tail",
            "effective_n": reg_effective_n,
            "S1": reg["S1"],
            "S1_null_band_p95": reg["S1_null_band_p95"],
            "S1_ci": boot[REGISTERED_ARM].get("S1_ci"),
            "S2": reg["S2"],
            "S2_null_band_p95": reg["S2_null_band_p95"],
            "S2_ci": boot[REGISTERED_ARM].get("S2_ci"),
            "S1_minus_S2": reg["S1_minus_S2"],
            "S1_minus_S2_ci": boot[REGISTERED_ARM].get("S1_minus_S2_ci"),
            "S3": reg["S3"],
            "S3_ci": boot[REGISTERED_ARM].get("S3_ci"),
            "S3_contrast": reg["S3_contrast"],
            "S3_contrast_ci": boot[REGISTERED_ARM].get("S3_contrast_ci"),
            "S3_obs": _f(s3_obs),
            "S3_obs_ci": boot_obs.get("S3_obs_ci"),
            "S3_obs_contrast": _f(s3_obs_contrast),
            "S3_obs_contrast_ci": boot_obs.get("S3_obs_contrast_ci"),
            "S4": s4,
            "S4_len_partial": s4_partial,
            "verdict": verdict,
            "lattice_inputs": lattice_inputs,
        },
        "per_arm": {a: {**per_arm[a], **boot[a]} for a in ALL_ARMS},
        "stratified_by_class": stratified,
        "family_clustered_registered_arm": fam_boot,
        "retrieval_per_context": retrieval_ctx,
        "retrieval_pair_rank": retrieval_pair,
        "twin_layers": twin_layers,
        "calibration": calibration,
        "length_battery": length_battery,
        "split_half": {
            "n_degenerate_pairs": n_degenerate,
            "axis_r_half": _f(axis_r_half),
            "axis_r10": _f(axis_r10),
            "axis_min_ok_flip_pairs_across_splits": axis_min_ok,
            "n_splits": N_AXIS_SPLITS,
        },
        "margin": {
            "rho_margin_rate": _f(data["margins"].get("rho_margin_rate")),
            "rho_p": _f(data["margins"].get("rho_p")),
            "validation_pass": data["margins"].get("validation_pass"),
            "pool_meta": data["margins"].get("pool_meta"),
        },
        "seeds": {
            "null_seed": NULL_SEED,
            "n_null_draws": N_NULL_DRAWS,
            "boot_seed": BOOT_SEED,
            "n_boot": N_BOOT,
        },
        "notes": {
            "orientation": data["bank"].get("orientation"),
            "null": "shuffled-pair null permutes observed deltas WITHIN pair_source; identical "
            "permutation set across arms/classes; bands are advisory (no abort gates)",
            "boot": "pair bootstrap stratified by flip/nonflip; flip-axis recomputed per resample "
            "with LOO-within-resample (Gram algebra); family-clustered companion resamples "
            "artifact_family_id clusters (registered arm)",
            "axis": "r_hat = mean observed flip-pair tail delta; members scored LOO, "
            "non-members against the full-mean axis",
            "tiny": "tiny mode fabricates ridge payloads at the tiny store dim (same "
            "load/apply path); dry-run judge => refusal rates None, halts exercised"
            if tiny
            else None,
        },
        "repro": RD._repro_meta(),
    }
    if dichotomy_halted or judge_integrity_halt:
        # Kill-criterion enforcement (concern dichotomy-halt-not-enforced, r2):
        # under a plan-§7 halt the dichotomous registered statistics move to
        # halted_diagnostics and are NULLED in the headline block — S4 (the
        # halt-independent battery) and the (already-suppressed) verdict stay.
        dich_keys = (
            "S1",
            "S1_null_band_p95",
            "S1_ci",
            "S2",
            "S2_null_band_p95",
            "S2_ci",
            "S1_minus_S2",
            "S1_minus_S2_ci",
            "S3",
            "S3_ci",
            "S3_contrast",
            "S3_contrast_ci",
            "S3_obs",
            "S3_obs_ci",
            "S3_obs_contrast",
            "S3_obs_contrast_ci",
        )
        reg_block = summary["registered"]
        summary["halted_diagnostics"] = {
            "note": "dichotomous registered statistics computed under a plan-§7 halt — "
            "diagnostics only, never headline",
            "registered": {k: reg_block[k] for k in dich_keys},
        }
        for k in dich_keys:
            reg_block[k] = None
        reg_block["halted"] = True
    return rows, ctx_rows, summary


def _ctx_source(c: dict) -> str:
    return "xstest" if str(c.get("carrier", "")).startswith("xstest") else "constructed"


def _pair_field(pairs: list[dict], pid, field: str):
    if pid is None:
        return None
    for p in pairs:
        if p["pair_id"] == pid:
            return p.get(field)
    return None


# ── main ──────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--local", default=None, help="local SVMP out-root (skip HF staging of run artifacts)"
    )
    ap.add_argument(
        "--stage-dir",
        default=None,
        help="staging dir (default: data/issue_2617/svmp_stage under repo root)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output dir (default: eval_results/issue_2617/svmp; use a scratch dir for --tiny)",
    )
    ap.add_argument(
        "--tiny",
        action="store_true",
        help="tiny mode: fabricate ridge payloads at the store dim; tolerate a dry-run judge",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def _import_check() -> None:
    import inspect

    from explore_persona_space.eval.batch_judge import is_api_refusal_error_dict
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    for name in (
        "load_ridge_payload",
        "split_half_stats",
        "rowwise_cos",
        "cross_cos",
        "through_origin_slope",
        "spearman_brown",
        "_import_apply_map",
        "_repro_meta",
        "_read_jsonl",
    ):
        assert callable(getattr(RD, name)), name
    apply_map = RD._import_apply_map()
    params = set(inspect.signature(apply_map).parameters)
    assert {"payload", "X_eval", "dev"} <= params, params
    assert callable(knn_retrieval) and callable(is_api_refusal_error_dict)
    _assert_schemas()
    print("[import-check] ok: RD surface + apply_map signature + schema self-checks", flush=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_argparser().parse_args()
    if args.import_check:
        _import_check()
        return 0
    if args.tiny and not args.out_dir:
        # Tiny-clobber refuse (r1 review g2): fabricated-ridge / dry-run-judge
        # outputs must never land on the committed production default.
        raise SystemExit(
            "--tiny requires an explicit --out-dir scratch path: refusing to overwrite "
            f"the committed production outputs at {OUT_DIR_DEFAULT}"
        )
    _assert_schemas()
    stage_dir = (
        Path(args.stage_dir)
        if args.stage_dir
        else (REPO_ROOT / "data" / "issue_2617" / "svmp_stage")
    )
    root, hf_revision, stage_sources = stage_inputs_svmp(args.local, stage_dir)
    data = load_svmp(root)
    d_model = data["vc"][data["primary"]].shape[1]
    print(
        f"[load] {len(data['ctx_ids'])} contexts / {len(data['pairs'])} pairs / "
        f"K={data['k']} / layers={data['layers']} primary={data['primary']} d={d_model}",
        flush=True,
    )
    if not args.tiny:
        # Grid-completeness gate: production reads run only on the full bank.
        assert d_model == HIDDEN, d_model
        assert list(data["layers"]) == list(MAP_LAYERS), data["layers"]
        assert len(data["ctx_ids"]) == 216 and len(data["pairs"]) == 108, (
            len(data["ctx_ids"]),
            len(data["pairs"]),
        )
        assert not bool(data["judge"].get("dry_run")), "production reads on a dry-run judge"
        # Plan §12(f) reuse control — before any new read ships.
        _anchor_parity_probe_2564()
    stage_dir.mkdir(parents=True, exist_ok=True)
    ridge_paths = stage_ridge_payloads_svmp(
        stage_dir, data["layers"], tiny=args.tiny, d=d_model, revision=hf_revision
    )
    rows, ctx_rows, summary = compute(data, ridge_paths, tiny=args.tiny)
    summary["staging"] = {
        "hf_data_repo": HF_DATA_REPO if not args.local else None,
        "revision": hf_revision,
        "local_root": str(root) if args.local else None,
        "sources": stage_sources,
    }
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR_DEFAULT
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(out_dir / "perpair.jsonl", rows)
    write_jsonl_atomic(out_dir / "percontext.jsonl", ctx_rows)
    write_json_atomic(out_dir / "summary.json", summary)
    print(
        f"[out] {out_dir / 'perpair.jsonl'} ({len(rows)} rows), "
        f"{out_dir / 'percontext.jsonl'} ({len(ctx_rows)} rows), "
        f"{out_dir / 'summary.json'} (verdict={summary['registered']['verdict']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
