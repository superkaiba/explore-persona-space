#!/usr/bin/env python
"""Issue #1336 — dedup-on-eval sensitivity re-read + fixed-L29 null quantile.

Step 9a-ter zero-GPU inline analysis round (analysis-only; pure re-reduction of
persisted artifacts — no new training, eval generation, or data).

P1 (dedup sensitivity):
  (a) Recompute the E1 held-out recalibrated per-layer R^2 (and S_r = the
      layer max) for the two RLVR lmsys5k cells EXCLUDING the exact-duplicate
      eval rows the E1.b(iii) dup audit found (88/3,629 rows in exact-dup
      groups). Fits/recal params UNCHANGED: the cross-fitted per-dim (a, b)
      are re-derived from the FULL original row set (bit-reproducing the
      committed read, asserted), and only the eval-side pooling drops rows.
  (b) Recompute the reparameterization-gap Delta_k / contrast C reads
      (recal primary + raw companion) per eval set under the same exclusion,
      including the HEADLINE set gsm8k_train5k_chat — whose own prompt bank is
      dup-audited here from the persisted answers.jsonl; zero dups there makes
      the headline re-read a declared no-op (recorded, never silently skipped).

P2 (fixed-L29 null quantile): from the persisted per-draw x per-layer
  within-fold pairing-permutation null matrices (recal_draws_{cell}.npz on the
  HF data repo), report p97.5 of the FIXED L29 column per cell beside the
  committed layer-max band B_r (A_r denominator context).

Registered convention (also recorded in the output JSON):
  - Exclusion: ALL rows whose exact raw-prompt sha256 occurs on >=2 rows of
    the ANALYZED row set are dropped from the EVAL side (every group member;
    no keep-one). Unjoined rows are kept and counted.
  - Pooling: pooled R^2 with fold-local test means recomputed over the KEPT
    rows per fold (the committed pooled convention restricted to the deduped
    eval set).
  - Bootstrap: dedup CIs resample the KEPT row set with a fresh
    draw_index_matrix(n_kept, n_boot, seed=5000+set_index) — the committed
    decision read's seed constants, shared draws across stages + scales.
  - Dup tier: exact only (sha256 of raw prompt text, digest-only — prompt
    text never leaves the hashing helper; LMSYS real-user rows).

All inputs are fetched by EXACT path from the HF data repo (no tree listing —
the ~1M-file repo's tree endpoint 504s; gotchas.md) with transient retry, and
the turnstore truth is streamed shard-by-shard (delete-after-reduce, peak
~one shard) with the reduced Y cached for resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) bind BEFORE torch/numpy import

import issue825_fit_cells as fc  # noqa: E402
import issue1336_diagnose_g1 as d1  # noqa: E402
import issue1336_ladder_alignment as la  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.experiments.issue_1336 import recal as _rc  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

# The two E1 recal cells (P1a + P2) and the decision stages/eval sets (P1b).
RECAL_CELLS = ("rlvr_chat_lmsys5k", "rlvr_naturalistic_lmsys5k")
STAGES = ("sft", "dpo", "rlvr")
# (corpus, fmt, set_index) — set_index is the committed cm.EVAL_SETS position
# (the decision read's bootstrap seed is 5000 + set_index).
REREAD_SETS = tuple((c, f, cm.EVAL_SETS.index((c, f))) for c, f in cm.EVAL_SETS)
HEADLINE_KEY = "gsm8k_train5k_chat"
# Reproduction tolerances: committed values came from the same fp64 math on a
# different host (BLAS thread count can reorder GEMM reductions), so the gate
# is tight-but-not-bitwise. DG-E0 keeps its registered +/-1e-3.
REPRO_TOL_R2 = 1e-7
REPRO_TOL_C = 1e-6


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage-root", type=Path, default=Path("data/issue_1336/dedup_stage"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results/issue_1336/diagnosis/recal/dedup_sensitivity.json"),
    )
    ap.add_argument("--committed-recal-dir", type=Path, default=None, help="test override")
    ap.add_argument("--n-boot", type=int, default=cm.N_BOOTSTRAP)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Staging (exact-path fetches; no tree listing)
# ---------------------------------------------------------------------------
def _fetch(rel: str, stage_root: Path) -> Path:
    """hf_hub_download of one exact repo path into stage_root (resume: local hit)."""
    target = stage_root / rel
    if target.exists():
        return target
    from huggingface_hub import hf_hub_download

    local = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=cm.HF_DATA_REPO, repo_type="dataset", filename=rel, local_dir=stage_root
        ),
        what=f"dedup stage: {rel}",
    )
    return Path(local)


def _not_found_errors() -> tuple[type[BaseException], ...]:
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    return (EntryNotFoundError, LocalEntryNotFoundError)


def stage_answers(corpus: str, stage_root: Path) -> Path:
    """Stage one corpus' rlvr answers.jsonl (direct file, else sha-verified shards).

    Prompt text is model-independent per prompt_idx (one bank per corpus), so
    the rlvr cell's persisted generation rows carry the prompt text for every
    stage's rows.
    """
    prefix = f"{cm.HF_PREFIX_1336}/raw_completions/generation/rlvr/{corpus}"
    out = stage_root / prefix / "answers.jsonl"
    if out.exists():
        return out
    try:
        return _fetch(f"{prefix}/answers.jsonl", stage_root)
    except _not_found_errors():
        pass  # sharded upload shape (>9.5 MB text) — reassemble below
    manifest = json.loads(_fetch(f"{prefix}/answers.manifest.json", stage_root).read_text())
    tmp = out.parent / "answers.jsonl.tmp"
    total = hashlib.sha256()
    with open(tmp, "wb") as fh:
        for part, sha in zip(manifest["parts"], manifest["sha256s"], strict=True):
            data = _fetch(f"{prefix}/{part}", stage_root).read_bytes()
            got = hashlib.sha256(data).hexdigest()
            assert got == sha, f"{part}: sha256 {got} != manifest {sha}"
            fh.write(data)
            total.update(data)
    assert total.hexdigest() == manifest["total_sha256"], (
        f"reassembled {corpus} answers.jsonl sha mismatch vs manifest"
    )
    tmp.replace(out)
    print(f"[stage] reassembled {corpus} answers.jsonl from {len(manifest['parts'])} parts")
    return out


def stream_turnstore_y(cell: str, layers: list[int], stage_root: Path) -> dict:
    """{"conv_ids": (N,), "Y": (N, len(layers), D) fp32} — streamed + cached.

    Replicates the committed load path's Y exactly at the sliced layers:
    fc._load_bundle_pt stacks rows in shard order and fc._cell_xy casts fp32
    then NaN-drops rows on the a1 slot/profile (slot_index=1,
    target_turn_index=1). bf16 -> fp32 is an exact embedding, so the streamed
    per-row slice equals the bundle path's Y[:, layers, :] bit-for-bit;
    validated downstream by the DG-E0 reproduction gate + the battery conv-id
    equality assert. Shards are deleted after reduce (peak ~one shard); the
    reduced Y is cached under stage_root for resume.
    """
    cache = stage_root / f"y_reduced_{cell}.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        assert [int(v) for v in payload["layers"]] == list(layers), (
            f"{cell}: cached Y layers {payload['layers']} != requested {layers}"
        )
        return {"conv_ids": np.asarray(payload["conv_ids"]), "Y": payload["Y"].numpy()}
    prefix = f"{cm.HF_PREFIX_1336}/analysis_tensors/turnstore_{cell}"
    lt = torch.tensor(layers, dtype=torch.long)
    ids: list[str] = []
    ys: list[torch.Tensor] = []
    i = 0
    while True:
        rel = f"{prefix}/{cell}_shard{i:03d}.pt"
        try:
            local = _fetch(rel, stage_root)
        except _not_found_errors():
            break
        payload = torch.load(local, map_location="cpu", weights_only=False)
        n_shard = len(payload["conv_ids"])
        for cid, s, p in zip(
            payload["conv_ids"], payload["slots"], payload["profiles"], strict=True
        ):
            if torch.isnan(s[1]).any() or torch.isnan(p[1]).any():
                continue  # the fc._cell_xy NaN keep-mask, replicated per row
            ids.append(str(cid))
            ys.append(p[1].index_select(0, lt).to(torch.float32))
        del payload
        Path(local).unlink()  # delete-after-reduce: peak transient = one shard
        print(f"[stage] reduced {rel} ({n_shard} rows; kept so far {len(ids)})", flush=True)
        i += 1
    assert i > 0, f"no turnstore shards found under {prefix}"
    Y = torch.stack(ys)
    torch.save({"conv_ids": ids, "Y": Y, "layers": list(layers)}, cache)
    return {"conv_ids": np.asarray(ids), "Y": Y.numpy()}


# ---------------------------------------------------------------------------
# Dedup labels + re-reduction cores (fixture-pinned:
# tests/test_issue1336_dedup_sensitivity.py)
# ---------------------------------------------------------------------------
def load_prompt_hashes(answers_path: Path) -> dict[str, str]:
    """prompt_idx (str) -> sha256 of the raw KEPT prompt text.

    Digest-only (prompt text never leaves this function — LMSYS real-user
    rows); text-mode file iteration, never splitlines (#825 U+2028/NEL).
    """
    out: dict[str, str] = {}
    with answers_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("kept", True):
                continue
            text = str(row.get("prompt", ""))
            out[str(row["prompt_idx"])] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out


def dup_exclusion_mask(
    conv_ids: np.ndarray, hashes_by_key: dict[str, str]
) -> tuple[np.ndarray, dict]:
    """(exclude_mask, digest) for the registered exclusion convention.

    True where the row's exact-prompt hash occurs on >=2 rows of the ANALYZED
    row set — EVERY member of an exact-dup group is dropped (no keep-one).
    Rows that fail the conv_id -> prompt join are KEPT and counted; the join
    rate is asserted against the committed dup-audit floor.
    """
    keys = [d1._rollout_key_for_conv(str(c)) for c in conv_ids]
    row_h = [hashes_by_key.get(k) for k in keys]
    joined = sum(1 for h in row_h if h is not None)
    counts: dict[str, int] = {}
    for h in row_h:
        if h is not None:
            counts[h] = counts.get(h, 0) + 1
    excl = np.asarray([h is not None and counts[h] > 1 for h in row_h], dtype=bool)
    join_rate = joined / max(len(conv_ids), 1)
    assert join_rate >= d1.SPOTCHECK_MIN_JOIN_RATE, (
        f"dup-label join rate {join_rate:.3f} < {d1.SPOTCHECK_MIN_JOIN_RATE} — "
        "conv_id -> prompt join broken"
    )
    digest = {
        "n_rows_analyzed": len(conv_ids),
        "join_rate": float(join_rate),
        "n_dup_groups": int(sum(1 for v in counts.values() if v > 1)),
        "n_rows_excluded": int(excl.sum()),
        "max_group_size": int(max(counts.values(), default=1)),
    }
    return excl, digest


def pooled_r2_on_rows(
    pred: np.ndarray, true: np.ndarray, folds: np.ndarray, keep: np.ndarray
) -> float:
    """Pooled fold-local test-mean R^2 restricted to the KEPT rows.

    ss_tot centers on each fold's KEPT-row mean (the committed pooled
    convention restricted to the deduped eval set). keep = all-true reproduces
    recal.raw_pooled_r2 / crossfit_recal_direct's pooled r2 exactly (pinned by
    the fixture test). An all-dropped fold contributes zero.
    """
    _, rows = _rc.fold_rows(folds)
    ss_res = ss_tot = 0.0
    for r in rows:
        rk = r[keep[r]]
        if len(rk) == 0:
            continue
        t = true[rk].astype(np.float64)
        p = pred[rk].astype(np.float64)
        mu = t.mean(0)
        ss_res += float(((t - p) ** 2).sum())
        ss_tot += float(((t - mu) ** 2).sum())
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def l29_null_quantile(null_mat: np.ndarray, layers: list[int], layer: int = 29) -> dict:
    """P2 read: p97.5 of the FIXED layer column beside the layer-max band."""
    ix = layers.index(layer)
    col = np.asarray(null_mat, dtype=np.float64)[:, ix]
    layer_max = np.nanmax(np.asarray(null_mat, dtype=np.float64), axis=1)
    return {
        "layer": int(layer),
        "n_draws": int(null_mat.shape[0]),
        "p975_null_l29_fixed": float(np.quantile(col, 0.975)),
        "p975_null_layer_max_recomputed": float(np.quantile(layer_max, 0.975)),
        "p975_null_per_layer": {
            str(li): float(np.quantile(np.asarray(null_mat, dtype=np.float64)[:, j], 0.975))
            for j, li in enumerate(layers)
        },
    }


# ---------------------------------------------------------------------------
# P1(a) — S_r re-read per recal cell
# ---------------------------------------------------------------------------
def s_r_reread(cell: str, args, hashes: dict[str, str], committed_dir: Path) -> dict:
    bat_rel = f"{cm.HF_PREFIX_1336}/analysis_tensors/diagnosis/battery_v0_preds_{cell}.npz"
    bat = np.load(_fetch(bat_rel, args.stage_root), allow_pickle=False)
    conv = np.asarray(bat["conv_ids"]).astype(str)
    folds = np.asarray(bat["folds"]).astype(np.int64)
    assert np.asarray(bat["fitted_mask"]).astype(bool).all(), f"{cell}: fitted_mask not all-true"
    layers = sorted(int(k[len("preds_l") :]) for k in bat.files if k.startswith("preds_l"))
    ts = stream_turnstore_y(cell, layers, args.stage_root)
    assert len(ts["conv_ids"]) == len(conv) and (ts["conv_ids"] == conv).all(), (
        f"{cell}: turnstore vs battery conv-id mismatch ({len(ts['conv_ids'])} vs {len(conv)} rows)"
    )
    excl, digest = dup_exclusion_mask(conv, hashes)
    keep = ~excl

    committed = json.loads((committed_dir / f"heldout_recal_{cell}.json").read_text())
    bar_r = None
    verdict_path = committed_dir / "recal_verdict.json"
    if verdict_path.exists():
        bar_r = float(json.loads(verdict_path.read_text())["lattice_inputs"]["bar_r"])

    is_chat = "naturalistic" not in cell
    per_layer: dict[str, dict] = {}
    max_repro_dev = 0.0
    for ix, li in enumerate(layers):
        P = np.asarray(bat[f"preds_l{li}"], dtype=np.float64)
        Y_l = ts["Y"][:, ix, :].astype(np.float64)
        raw_orig = _rc.raw_pooled_r2(P, Y_l, folds)
        direct = _rc.crossfit_recal_direct(P, Y_l, folds)
        # Reproduction gate: the recomputed original must match the committed
        # heldout_recal JSON (staging/alignment drift kills everything below).
        dev = abs(direct["r2"] - float(committed["per_layer"][str(li)]["heldout_recal_r2"]))
        assert dev <= REPRO_TOL_R2, (
            f"{cell} L{li}: recomputed heldout recal r2 {direct['r2']:.9f} deviates "
            f"{dev:.2e} from the committed value (tol {REPRO_TOL_R2})"
        )
        max_repro_dev = max(max_repro_dev, dev)
        per_layer[str(li)] = {
            "raw_r2_original": float(raw_orig),
            "raw_r2_dedup": pooled_r2_on_rows(P, Y_l, folds, keep),
            "heldout_recal_r2_original": float(direct["r2"]),
            "heldout_recal_r2_dedup": pooled_r2_on_rows(direct["pred_recal"], Y_l, folds, keep),
        }
        del P, Y_l, direct
    # DG-E0 reproduction (chat cell; registered targets, +/-1e-3).
    dge0 = None
    if is_chat:
        dge0 = {}
        for label, li, target in (("l29", 29, -0.92866), ("l30", 30, -0.93494)):
            got = per_layer[str(li)]["raw_r2_original"]
            dge0[label] = {"recomputed": got, "target": target, "abs_dev": abs(got - target)}
            assert abs(got - target) <= 1e-3, (
                f"{cell} DG-E0 {label}: {got:.6f} vs {target} (tol 1e-3) — alignment drift"
            )
    heldout_orig = {li: per_layer[str(li)]["heldout_recal_r2_original"] for li in map(str, layers)}
    heldout_dedup = {li: per_layer[str(li)]["heldout_recal_r2_dedup"] for li in map(str, layers)}
    s_r_orig = max(heldout_orig.values())
    s_r_dedup = max(heldout_dedup.values())
    return {
        "cell_id": cell,
        "n_rows": len(conv),
        "dup_digest": digest,
        "per_layer": per_layer,
        "s_r_original": float(s_r_orig),
        "s_r_argmax_layer_original": int(max(heldout_orig, key=heldout_orig.get)),
        "s_r_dedup": float(s_r_dedup),
        "s_r_argmax_layer_dedup": int(max(heldout_dedup, key=heldout_dedup.get)),
        "s_r_committed": float(committed["s_r"]),
        "max_reproduction_dev": float(max_repro_dev),
        "bar_r": bar_r,
        "s_r_dedup_above_bar_r": bool(bar_r is not None and s_r_dedup > bar_r),
        "dg_e0": dge0,
    }


# ---------------------------------------------------------------------------
# P1(b) — Delta_k / contrast C re-read per eval set
# ---------------------------------------------------------------------------
def _gaps_and_c(npzs: dict, rows_by_stage: dict, w: np.ndarray, layer: int, variant: str) -> dict:
    """Per-stage gap (within - comp R^2) + C = gap_rlvr - gap_dpo on one scale.

    Pure re-reduction of the persisted OOF predictions (variant "" = raw
    companion, "_recal" = the plan-v9 recalibrated primary), restricted to the
    given rows; shared draws w across stages (the committed pairing).
    """
    gaps: dict[str, dict] = {}
    for k, rows in rows_by_stage.items():
        z = npzs[k]
        within = np.asarray(z[f"within{variant}_l{layer}"])[rows].astype(np.float64)
        comp = np.asarray(z[f"comp{variant}_l{layer}"])[rows].astype(np.float64)
        y = np.asarray(z[f"y_l{layer}"])[rows].astype(np.float64)
        boot = la.paired_bootstrap_batched(within, y, comp, y, w)
        point = fc._pooled_r2(within, y) - fc._pooled_r2(comp, y)
        gaps[k] = {"point": float(point), "draws": boot["delta"]}
    c_draws = gaps["rlvr"]["draws"] - gaps["dpo"]["draws"]
    c_point = gaps["rlvr"]["point"] - gaps["dpo"]["point"]
    return {
        "gap_per_stage": {k: {"point": gaps[k]["point"], **la._ci(gaps[k]["draws"])} for k in gaps},
        "contrast_C": {"point": float(c_point), **la._ci(c_draws)},
    }


def contrast_reread(
    corpus: str,
    fmt: str,
    set_index: int,
    args,
    hashes: dict[str, str],
    headline_layer: int,
    committed_set: dict,
) -> dict:
    npzs = {}
    align_prefix = f"{cm.HF_PREFIX_1336}/analysis_tensors/preds/align"
    for k in STAGES:
        rel = f"{align_prefix}/alignpreds_base__{k}_{fmt}_{corpus}.npz"
        npzs[k] = np.load(_fetch(rel, args.stage_root), allow_pickle=False)
    shared = None
    for k in STAGES:
        ids = np.asarray(npzs[k]["conv_ids"]).astype(str)
        shared = ids if shared is None else np.intersect1d(shared, ids)
    assert shared is not None and len(shared) > 0, "empty shared row set"
    rows_by_stage = {}
    for k in STAGES:
        pos = {c: i for i, c in enumerate(np.asarray(npzs[k]["conv_ids"]).astype(str))}
        rows_by_stage[k] = np.asarray([pos[c] for c in shared], dtype=np.int64)
    n_shared = len(shared)
    assert n_shared == int(committed_set["n_shared_rows"]), (
        f"{corpus}_{fmt}: shared rows {n_shared} != committed {committed_set['n_shared_rows']}"
    )

    # Original reproduction (same seed constants as the committed decision read).
    idx = la.draw_index_matrix(n_shared, args.n_boot, seed=5000 + set_index)
    w = la.counts_from_indices(idx, n_shared)
    orig_recal = _gaps_and_c(npzs, rows_by_stage, w, headline_layer, "_recal")
    orig_raw = _gaps_and_c(npzs, rows_by_stage, w, headline_layer, "")
    c_dev = abs(orig_recal["contrast_C"]["point"] - float(committed_set["contrast_C"]["point"]))
    assert c_dev <= REPRO_TOL_C, (
        f"{corpus}_{fmt}: recomputed C point deviates {c_dev:.2e} from the committed "
        f"decision value (tol {REPRO_TOL_C})"
    )

    # Dedup exclusion over the ANALYZED (shared) row set.
    excl, digest = dup_exclusion_mask(shared, hashes)
    out = {
        "eval_set": f"{corpus}_{fmt}",
        "n_shared_rows": int(n_shared),
        "dup_digest": digest,
        "c_point_reproduction_dev": float(c_dev),
        "original": {"recal": orig_recal, "raw": orig_raw},
    }
    if int(excl.sum()) == 0:
        out["no_op"] = True
        out["no_op_reason"] = (
            "zero exact-duplicate prompt rows in this eval set's analyzed row set — the "
            "dedup re-read equals the original read by construction (recorded, not skipped)"
        )
        out["dedup"] = {"recal": orig_recal, "raw": orig_raw}
        return out
    keep = ~excl
    kept_rows = {k: rows_by_stage[k][keep] for k in STAGES}
    n_kept = int(keep.sum())
    idx2 = la.draw_index_matrix(n_kept, args.n_boot, seed=5000 + set_index)
    w2 = la.counts_from_indices(idx2, n_kept)
    out["no_op"] = False
    out["n_kept_rows"] = n_kept
    out["dedup"] = {
        "recal": _gaps_and_c(npzs, kept_rows, w2, headline_layer, "_recal"),
        "raw": _gaps_and_c(npzs, kept_rows, w2, headline_layer, ""),
    }
    return out


# ---------------------------------------------------------------------------
# P2 — fixed-L29 null quantile per recal cell
# ---------------------------------------------------------------------------
def p2_reread(cell: str, args, committed_dir: Path) -> dict:
    rel = f"{cm.HF_PREFIX_1336}/analysis_tensors/diagnosis/recal/recal_draws_{cell}.npz"
    z = np.load(_fetch(rel, args.stage_root), allow_pickle=False)
    layers = [int(v) for v in z["layers"]]
    read = l29_null_quantile(np.asarray(z["null_r2_matrix"]), layers, layer=29)
    committed = json.loads((committed_dir / f"heldout_recal_{cell}.json").read_text())
    band_committed = float(committed["recal_null"]["band_p975_layer_max"])
    dev = abs(read["p975_null_layer_max_recomputed"] - band_committed)
    assert dev <= REPRO_TOL_R2, (
        f"{cell}: recomputed layer-max band deviates {dev:.2e} from committed (tol {REPRO_TOL_R2})"
    )
    read["cell_id"] = cell
    read["band_p975_layer_max_committed"] = band_committed
    return read


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    t0 = time.time()
    committed_dir = args.committed_recal_dir or Path("eval_results/issue_1336/diagnosis/recal")
    decision = json.loads(
        Path("eval_results/issue_1336/decision/headline_contrast.json").read_text()
    )
    headline_layer = int(decision["headline_layer"])
    assert decision["headline_eval_set"] == HEADLINE_KEY, decision["headline_eval_set"]

    # Dup labels per corpus (exact tier; digest-only).
    hashes_by_corpus = {
        corpus: load_prompt_hashes(stage_answers(corpus, args.stage_root))
        for corpus in ("lmsys5k", "gsm8k_train5k", "gsm8k_test1319")
    }

    # P2 first (tiny inputs — fails fast on staging problems).
    p2 = {cell: p2_reread(cell, args, committed_dir) for cell in RECAL_CELLS}

    # P1(a) — S_r re-read on the two recal cells.
    p1a = {
        cell: s_r_reread(cell, args, hashes_by_corpus["lmsys5k"], committed_dir)
        for cell in RECAL_CELLS
    }

    # P1(b) — Delta_k / C re-read per eval set (headline included; no-op recorded
    # when the set carries zero exact dups).
    p1b = {}
    for corpus, fmt, si in REREAD_SETS:
        key = f"{corpus}_{fmt}"
        p1b[key] = contrast_reread(
            corpus,
            fmt,
            si,
            args,
            hashes_by_corpus[corpus],
            headline_layer,
            decision["per_eval_set"][key],
        )

    payload = {
        "metadata": {
            "git_commit": fc._git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": "scripts/issue1336_dedup_sensitivity.py",
            "numpy": np.__version__,
            "torch": torch.__version__,
            "n_boot": int(args.n_boot),
            "headline_layer": headline_layer,
            "wall_s": None,  # filled below
        },
        "convention": {
            "exclusion": (
                "ALL rows whose exact raw-prompt sha256 occurs on >=2 rows of the ANALYZED "
                "row set are dropped from the EVAL side (every dup-group member; no "
                "keep-one). Unjoined rows are kept and counted. Fits and recalibration "
                "parameters are UNCHANGED — the cross-fitted per-dim (a, b) and the "
                "within/comp OOF predictions come from the FULL original row set "
                "(duplicates included); this is a pure re-reduction of stored predictions."
            ),
            "pooling": (
                "pooled R^2 with fold-local test means recomputed over the KEPT rows per "
                "fold (committed pooled convention restricted to the deduped eval set)"
            ),
            "bootstrap": (
                "dedup CIs resample the KEPT row set: fresh draw_index_matrix(n_kept, "
                "n_boot, seed=5000+set_index) — the committed decision read's seed "
                "constants, draws shared across stages and scales within a set"
            ),
            "dup_tier": "exact (sha256 of raw prompt text; digest-only)",
        },
        "p1a_s_r_reread": p1a,
        "p1b_contrast_reread": p1b,
        "p2_l29_null_quantile": p2,
    }
    payload["metadata"]["wall_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[dedup1336] wrote {args.out} (wall {payload['metadata']['wall_s']}s)")
    for cell, r in p1a.items():
        print(
            f"[dedup1336] {cell}: S_r {r['s_r_original']:.4f} -> dedup {r['s_r_dedup']:.4f} "
            f"(bar_r={r['bar_r']}, above={r['s_r_dedup_above_bar_r']})"
        )
    for key, r in p1b.items():
        co = r["original"]["recal"]["contrast_C"]
        cd = r["dedup"]["recal"]["contrast_C"]
        print(
            f"[dedup1336] {key}: C {co['point']:.6f} [{co['ci_lo']:.6f},{co['ci_hi']:.6f}] -> "
            f"dedup {cd['point']:.6f} [{cd['ci_lo']:.6f},{cd['ci_hi']:.6f}]"
            + (" (no-op)" if r.get("no_op") else "")
        )
    for cell, r in p2.items():
        print(
            f"[dedup1336] {cell}: p97.5 null L29 fixed {r['p975_null_l29_fixed']:.4f} "
            f"vs layer-max band {r['band_p975_layer_max_committed']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
