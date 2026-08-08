#!/usr/bin/env python3
"""#1768 continuation: A7 gate re-read + Delta-M probe on LAST-TOKEN context.

The last-token re-pool (commit c7a5fda6d1) re-fit M0/M+/M+_tf but could NOT
re-read the A7 whitened base-similarity gate: the gate's ``c_src`` is the SOURCE
context's vector from ``panel_capture/base_<beh>``, not from the corpus store, so
it needs a last-token PANEL capture that round did not build. This module closes
that gap and re-runs the two remaining pooling-sensitive reads.

Phases
------
``panel``   GPU, tiny. Last-token context vectors for the 4 BASE panel units
            (120 rows each = 6 contexts x 20 questions, all on the BASE model,
            so ONE model load serves all four). Row order is asserted against
            the round-1 panel store's ``row_meta`` so ``_panel_rows``-style
            (context_id, question_idx) lookups stay valid.
``gate``    CPU/IO. Per (arm, layer): whitened similarity g_pred from the
            last-token c_src + Sigma over the 15,000 last-token corpus base
            TRAIN context vectors, against the realized per-context write
            coefficient g_hat = delta_v . w / ||w||^2 with w and delta_v taken
            UNCHANGED from the round-1 answer-side artifacts. Reuses
            ``issue1768_directions.gate_read`` verbatim so only the context
            summary differs.
``dmprobe`` CPU/IO. Is the per-context write w(x) = v+(x) - v0(x) predictable
            from the LAST-TOKEN c0(x)? Ridge on the 8-arm write_predictability
            subset at its pinned layer, both trees (op / tf), with the standing
            identity+learned-bias and kNN-retrieval baselines.

Round-1 answer-side stores stream one arm at a time (download -> extract ->
delete), so peak disk stays ~1 GB rather than the ~61 GB the full grid needs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# load_dotenv() BEFORE numpy/torch so the shared-VM thread caps (#847) bind.
load_dotenv()


import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1768_lasttoken as LT  # noqa: E402
import issue1768_lasttoken_fit as LTF  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.lt_gate")

PANEL_BASE_UNITS = ("base_cas", "base_imp", "base_mk", "base_syc")
RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_1768" / "lasttoken_repool" / "gate"
PANEL_HF_SUBDIR = "lasttoken_ctx/panel"


def _meta() -> dict:
    return LT._meta()


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LT._atomic_json(path, obj)


# ── phase: last-token PANEL capture (GPU) ────────────────────────────────────


def _panel_rows_from_hub(cache: Path, unit: str) -> list[dict]:
    """Round-1 panel rows (prompt ids + spans + context id), hub-staged."""
    from explore_persona_space.orchestrate import hub

    target = cache / f"{unit}__raw_rows.json"
    if not target.exists():
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{X.HF_PREFIX}/panel_capture/{unit}/raw_rows.json",
            target,
            repo_type="dataset",
            overwrite=True,
        )
    rows = json.loads(target.read_text())["rows"]
    for r in rows:
        assert 0 < r["context_len"] <= len(r["prompt_token_ids"]), (unit, r["question_idx"])
    return rows


def _round1_row_meta(cache: Path, unit: str) -> list[dict]:
    """``row_meta`` of the round-1 panel store — the alignment contract."""
    import torch

    from explore_persona_space.orchestrate import hub

    target = cache / f"{unit}__pooled.pt"
    if not target.exists():
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{X.HF_PREFIX}/panel_capture/{unit}/pooled.pt",
            target,
            repo_type="dataset",
            overwrite=True,
        )
    # mmap: read row_meta without materializing the 72 MB of tensors
    store = torch.load(target, map_location="cpu", mmap=True, weights_only=False)
    return [dict(m) for m in store["row_meta"]]


def capture_panel(out_root: Path, layers: list[int], hf_prefix: str, upload: bool) -> None:
    """Last-token context vectors for the 4 BASE panel units (one model load)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = out_root / "lt_panel_src"
    cache.mkdir(parents=True, exist_ok=True)
    model_path = X.BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    device = LT.CAP._device()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    hidden = model.config.hidden_size
    for li in layers:
        assert 0 <= li < len(model.model.layers), (li, len(model.model.layers))

    captured: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook_fn(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[li] = hs.detach()

        return hook_fn

    hooks = [model.model.layers[li].register_forward_hook(make_hook(li)) for li in layers]
    import inspect

    fwd = getattr(model, "forward", model.__call__)
    keep = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(fwd).parameters else {}
    try:
        for unit in PANEL_BASE_UNITS:
            out_dir = out_root / "lasttoken_panel" / unit
            store_path = out_dir / "lasttoken_panel.pt"
            if store_path.exists():
                logger.info("[lt-panel] %s present, skip", unit)
                if upload:
                    _upload_panel(out_dir, unit, hf_prefix)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = _panel_rows_from_hub(cache, unit)
            r1_meta = _round1_row_meta(cache, unit)
            # ALIGNMENT CONTRACT: our row order must equal round 1's row_meta,
            # or every (context_id, question_idx) lookup silently reads the
            # wrong row. `persona` is the panel's context_id field.
            ours = [(r["persona"], r["question_idx"]) for r in rows]
            theirs = [(m["context_id"], m["question_idx"]) for m in r1_meta]
            assert ours == theirs, (
                f"{unit}: panel row order differs from round-1 row_meta "
                f"({len(ours)} vs {len(theirs)}; first mismatch "
                f"{next((i for i, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b), None)})"
            )
            pooled: dict[str, dict[int, list]] = {
                p: {li: [] for li in layers} for p in LT.POSITIONS
            }
            bs = LT.FWD_BATCH
            for start in range(0, len(rows), bs):
                batch = rows[start : start + bs]
                seqs = [r["prompt_token_ids"] for r in batch]
                max_len = max(len(s) for s in seqs)
                input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
                attn = torch.zeros((len(batch), max_len), dtype=torch.long)
                for i, s in enumerate(seqs):
                    input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                    attn[i, : len(s)] = 1
                with torch.no_grad():
                    model(
                        input_ids=input_ids.to(device),
                        attention_mask=attn.to(device),
                        **keep,
                    )
                for li in layers:
                    hs = captured[li]
                    assert hs.shape[:2] == (len(batch), max_len), hs.shape
                    for i, r in enumerate(batch):
                        idx = {
                            "last_prompt": len(r["prompt_token_ids"]) - 1,
                            "last_ctx": r["context_len"] - 1,
                        }
                        for pos in LT.POSITIONS:
                            vec = hs[i, idx[pos], :].float().cpu()
                            assert vec.shape == (hidden,), vec.shape
                            pooled[pos][li].append(vec)
            store = {
                "schema_version": 1,
                "unit": unit,
                "row_meta": [
                    {"context_id": r["persona"], "question_idx": r["question_idx"]} for r in rows
                ],
                "arms": {
                    pos: {li: torch.stack(pooled[pos][li]).to(torch.float16) for li in layers}
                    for pos in LT.POSITIONS
                },
                "metadata": {
                    **_meta(),
                    "model_path": model_path,
                    "layers": layers,
                    "positions": list(LT.POSITIONS),
                    "n_rows": len(rows),
                    "row_order_asserted_against": f"{X.HF_PREFIX}/panel_capture/{unit}/pooled.pt",
                },
            }
            tmp = store_path.with_suffix(".pt.tmp")
            torch.save(store, tmp)
            os.replace(tmp, store_path)
            logger.info("[lt-panel] %s captured %d rows", unit, len(rows))
            if upload:
                _upload_panel(out_dir, unit, hf_prefix)
    finally:
        for h in hooks:
            h.remove()
        captured.clear()
    logger.info("[shard-complete] panel capture done: %s", list(PANEL_BASE_UNITS))


def _upload_panel(out_dir: Path, unit: str, hf_prefix: str) -> None:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    prefix = f"{hf_prefix}/{PANEL_HF_SUBDIR}/{unit}"
    hub._upload(out_dir, X.HF_DATA_REPO, "dataset", prefix, raise_on_error=True)
    expected = [f"{prefix}/{p.name}" for p in sorted(out_dir.iterdir()) if p.is_file()]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(token=os.environ.get("HF_TOKEN")),
        X.HF_DATA_REPO,
        expected,
        path_in_repo=prefix,
        repo_type="dataset",
    )
    assert not missing, f"{unit}: panel upload verify missing {missing}"
    logger.info("[lt-panel-upload] %s verified %d files at %s", unit, len(expected), prefix)


# ── phase: gate re-read (A7) ─────────────────────────────────────────────────


def lt_corpus_sigma(out_root: Path, layer: int, position: str) -> dict:
    """Shrunk uncentered second moment of the LAST-TOKEN base TRAIN contexts.

    Mirrors ``issue1768_directions.corpus_sigma`` exactly (same base unit, same
    train-row selection, same shrinkage) with the last-token context vector
    substituted for the span mean — so only the pooling differs.
    """
    arrs, shas = LTF.load_lasttoken(out_root, "base_content", [layer], position)
    C = arrs[layer]
    sample = X.load_corpus_sample(out_root)
    sha_to_q = {r["sha"]: q for q, r in enumerate(sample["rows"])}
    qidx = np.asarray([sha_to_q[s] for s in shas])
    C_tr = C[qidx < sample["n_train"]]
    d = C_tr.shape[1]
    sigma = C_tr.T @ C_tr / max(1, C_tr.shape[0])
    lam = 0.1  # issue1768_directions.SHRINKAGE
    sigma_sh = (1 - lam) * sigma + lam * (np.trace(sigma) / d) * np.eye(d)
    evals, evecs = np.linalg.eigh(sigma_sh)
    del evals
    return {
        "sigma": sigma_sh,
        "top_eig": evecs[:, -1],
        "n_rows": int(C_tr.shape[0]),
        "shrinkage": lam,
    }


def lt_panel_c_src(out_root: Path, beh: str, ctx_id: str, layer: int, position: str) -> np.ndarray:
    """Mean LAST-TOKEN context vector over the SOURCE context's panel rows."""
    import torch

    p = out_root / "lasttoken_panel" / f"base_{beh}" / "lasttoken_panel.pt"
    assert p.exists(), f"missing last-token panel store: {p} (run --phase panel first)"
    store = torch.load(p, map_location="cpu", weights_only=False)
    mat = np.asarray(store["arms"][position][layer].float().numpy(), dtype=np.float64)
    rows = [i for i, m in enumerate(store["row_meta"]) if m["context_id"] == ctx_id]
    assert rows, f"context {ctx_id} absent from {p}"
    return mat[rows].mean(axis=0)


def _stage_round1_panel(out_root: Path, unit: str, subdir: str = "panel_capture") -> Path | None:
    """Stage a round-1 panel store where ``panel_write_legs`` expects it."""
    from explore_persona_space.orchestrate import hub

    dest = out_root / subdir / unit / "pooled.pt"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{X.HF_PREFIX}/{subdir}/{unit}/pooled.pt",
            dest,
            repo_type="dataset",
            overwrite=True,
        )
    except Exception as exc:  # tf panel is optional for some arms
        logger.info("[gate] %s/%s unavailable (%s)", subdir, unit, type(exc).__name__)
        return None
    return dest


CORPUS_SAMPLE_REV = "c07267285d2cdbf3e0401ddc3e3accae50e496a7"  # round-1 pinned revision


def _stage_lasttoken_panel(out_root: Path, unit: str) -> Path:
    """Stage a LAST-TOKEN panel store (this round's own `--phase panel` output).

    ``lt_panel_c_src`` reads it locally, so a VM-side gate run after the GPU pod
    is gone streams it back from the Hub.
    """
    from explore_persona_space.orchestrate import hub

    dest = out_root / "lasttoken_panel" / unit / "lasttoken_panel.pt"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(
        X.HF_DATA_REPO,
        f"{X.HF_PREFIX}/{PANEL_HF_SUBDIR}/{unit}/lasttoken_panel.pt",
        dest,
        repo_type="dataset",
        overwrite=True,
    )
    return dest


def _stage_corpus_sample(out_root: Path) -> Path:
    """Stage the p0 corpus sample at round 1's PINNED revision.

    ``X.load_corpus_sample`` reads it from ``<out_root>/inputs/``; pinning the
    revision keeps the train/val/test split byte-identical to the fits this
    re-read is compared against.
    """
    from explore_persona_space.orchestrate import hub

    dest = out_root / "inputs" / "corpus_sample.json"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(
        X.HF_DATA_REPO,
        f"{X.HF_PREFIX}/inputs/corpus_sample.json",
        dest,
        repo_type="dataset",
        revision=CORPUS_SAMPLE_REV,
        overwrite=True,
    )
    return dest


def _stage_lasttoken(out_root: Path, unit: str) -> Path:
    """Stage a round-1 LAST-TOKEN context store where ``load_lasttoken`` reads it.

    Round 1's stores live only on the Hub (its pod is long gone), so every
    consumer on this leg streams them: base units are kept for the whole run,
    per-arm units are deleted by the caller once consumed.
    """
    from explore_persona_space.orchestrate import hub

    dest = out_root / "lasttoken" / unit / "lasttoken.pt"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(
        X.HF_DATA_REPO,
        f"{X.HF_PREFIX}/lasttoken_ctx/{unit}/lasttoken.pt",
        dest,
        repo_type="dataset",
        overwrite=True,
    )
    return dest


def run_gate(
    out_root: Path, results_dir: Path, layers: list[int], position: str, arms_filter: str
) -> None:
    """A7 gate re-read per (arm, layer) on the last-token context summary."""
    import issue1768_directions as D

    cache = out_root / "lt_answer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    arms = X.all_arms()
    if arms_filter:
        want = {a.strip() for a in arms_filter.split(",") if a.strip()}
        arms = [a for a in arms if a.arm_id in want]
    sig: dict[int, dict] = {}
    reads: dict[str, dict] = {}
    _stage_corpus_sample(out_root)
    for beh in {a.beh_key for a in arms}:
        _stage_round1_panel(out_root, f"base_{beh}")
        _stage_lasttoken_panel(out_root, f"base_{beh}")
    for base_unit in {X.base_unit_for(a.arm_id) for a in arms}:
        _stage_lasttoken(out_root, base_unit)  # kept: shared by every arm
    for k, arm in enumerate(arms):
        arm_pool = _stage_round1_panel(out_root, arm.arm_id)
        tf_pool = _stage_round1_panel(out_root, arm.arm_id, "panel_capture_tf")
        arm_lt = _stage_lasttoken(out_root, arm.arm_id)
        try:
            for layer in layers:
                if layer not in sig:
                    sig[layer] = lt_corpus_sigma(out_root, layer, position)
                    logger.info(
                        "[gate] sigma L%d built from %d last-token train rows",
                        layer,
                        sig[layer]["n_rows"],
                    )
                legs = D.panel_write_legs(out_root, arm, layer)
                w = legs["w_primary"]
                c_src = lt_panel_c_src(out_root, arm.beh_key, legs["src_ctx"], layer, position)
                cell = LTF.build_cell(out_root, cache, arm.arm_id, layer, position)
                delta_v = cell["Vplus"] - cell["V0"]
                delta_v_tf = cell["Vplus_tf"] - cell["V0"]
                key = f"{arm.arm_id}_L{layer}"
                reads[key] = {
                    "arm_id": arm.arm_id,
                    "kind": arm.kind,
                    "beh_key": arm.beh_key,
                    "method": arm.method,
                    "layer": layer,
                    "src_ctx": legs["src_ctx"],
                    "position": position,
                    "on_policy": D.gate_read(cell["C0"], delta_v, c_src, w, sig[layer]),
                    "matched_text": D.gate_read(cell["C0"], delta_v_tf, c_src, w, sig[layer]),
                    "sigma_n_rows": sig[layer]["n_rows"],
                    "sigma_shrinkage": sig[layer]["shrinkage"],
                    "n_rows": int(cell["C0"].shape[0]),
                }
                logger.info(
                    "[gate] %s L%d rho_on_policy=%+.4f (round-1 content median +0.1384)",
                    arm.arm_id,
                    layer,
                    reads[key]["on_policy"]["spearman_rho"],
                )
        finally:  # stream: per-arm panel + last-token stores are consumed once
            for p in (arm_pool, tf_pool, arm_lt):
                if p is not None:
                    p.unlink(missing_ok=True)
        logger.info("[phase=lt_gate arm=%s %d/%d done]", arm.arm_id, k + 1, len(arms))
    _atomic_json(results_dir / "gate_reads_lasttoken.json", {"reads": reads, **_meta()})
    logger.info("[shard-complete] gate re-read %d cells", len(reads))


# ── phase: Delta-M probe (c0_lt -> per-context write w(x)) ───────────────────


def _baselines_for(
    Xd: np.ndarray, Y: np.ndarray, pred_te: np.ndarray, te: np.ndarray, tr: np.ndarray
) -> dict:
    """identity+learned-bias + kNN retrieval for an arbitrary (X -> Y) map.

    The standing mapping rule requires BOTH reads alongside held-out R2. Kept
    local (rather than reusing the fit driver's ``_baseline_reads``, which is
    hard-wired to C0 -> V0) because the Delta-M target is the WRITE vector.
    """
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )
    import issue1768_fit as F

    out: dict = {}
    if Xd.shape[1] == Y.shape[1]:
        ib = identity_bias_predict(Xd[tr], Y[tr], Xd[te])
        out["identity_bias"] = {
            "heldout_r2": F._pooled_r2(ib, Y[te]),
            "mean_cos": F._mean_cos(ib, Y[te]),
            "knn_euclidean": knn_retrieval(ib, Y[te], ks=(1, 10), metric="euclidean"),
        }
    else:
        out["identity_bias"] = {"inapplicable": f"dim {Xd.shape[1]} vs {Y.shape[1]}"}
    out["fitted_map"] = {
        "knn_euclidean": knn_retrieval(pred_te, Y[te], ks=(1, 10), metric="euclidean"),
        "knn_cosine": knn_retrieval(pred_te, Y[te], ks=(1, 10), metric="cosine"),
    }
    return out


def run_dmprobe(out_root: Path, results_dir: Path, position: str, arms_filter: str) -> None:
    """Is w(x) = v+(x) - v0(x) predictable from the LAST-TOKEN c0(x)?"""
    import torch

    import issue1768_fit as F

    picks_path = (
        REPO_ROOT / "eval_results" / "issue_1768" / "write_predictability" / "arm_picks.json"
    )
    picks = json.loads(picks_path.read_text())
    arm_ids = [p["arm_id"] for p in picks["picks"]]
    layer = int(picks["layer"])
    if arms_filter:
        want = {a.strip() for a in arms_filter.split(",") if a.strip()}
        arm_ids = [a for a in arm_ids if a in want]
    cache = out_root / "lt_answer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _stage_corpus_sample(out_root)
    for base_unit in {X.base_unit_for(a) for a in arm_ids}:
        _stage_lasttoken(out_root, base_unit)  # kept: shared by every arm
    cells: dict[str, dict] = {}
    for k, arm_id in enumerate(arm_ids):
        _stage_lasttoken(out_root, arm_id)
        cell = LTF.build_cell(out_root, cache, arm_id, layer, position)
        tr, val, te = F._split_idx(cell["split"])
        for tree, target in (
            ("op", cell["Vplus"] - cell["V0"]),
            ("tf", cell["Vplus_tf"] - cell["V0"]),
        ):
            pred, meta, _pay = F._fit_map(cell["C0"], target, tr, val, te, dev)
            rec = {
                **F._map_reads(pred, target[te]),
                "selected_lambda": meta["selected_lambda"],
                "arm_id": arm_id,
                "tree": tree,
                "layer": layer,
                "position": position,
                "write_norm_mean_test": float(np.linalg.norm(target[te], axis=1).mean()),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "baselines": _baselines_for(cell["C0"], target, pred, te, tr),
            }
            cells[f"{arm_id}|{tree}"] = rec
            logger.info(
                "[dmprobe] %s tree=%s L%d heldout_r2=%.4f (round-1 span-mean median "
                "op 0.1019 / tf 0.3881)",
                arm_id,
                tree,
                layer,
                rec["heldout_r2"],
            )
        logger.info("[phase=lt_dmprobe arm=%s %d/%d done]", arm_id, k + 1, len(arm_ids))
    _atomic_json(
        results_dir / "dmprobe_lasttoken.json",
        {"cells": cells, "layer": layer, "arm_picks": str(picks_path.name), **_meta()},
    )
    logger.info("[shard-complete] dmprobe %d cells", len(cells))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--phase", default="panel", choices=("panel", "gate", "dmprobe"))
    ap.add_argument("--layers", default=",".join(str(x) for x in X.LAYERS))
    ap.add_argument("--position", default="last_prompt")
    ap.add_argument("--arms", default="")
    ap.add_argument("--hf-prefix", default=None, help="upload prefix (required unless --no-upload)")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        import inspect

        from explore_persona_space.orchestrate import hub

        import issue1768_directions as _dirs  # noqa: F401

        inspect.signature(hub.verify_repo_paths_uploaded).bind(
            object(), object(), object(), path_in_repo="p", repo_type="dataset"
        )
        inspect.signature(hub._upload).bind(
            object(), object(), object(), object(), raise_on_error=True
        )
        print("import-check ok (upload/verify call shapes bind)")
        return 0

    assert args.out_root is not None, "--out-root is required outside --import-check"
    layers = [int(x) for x in args.layers.split(",")]
    if args.phase == "panel":
        if not args.no_upload and not args.hf_prefix:
            raise SystemExit("--hf-prefix is required when uploading (no issue-prefix default)")
        capture_panel(args.out_root, layers, args.hf_prefix or "", not args.no_upload)
    elif args.phase == "gate":
        run_gate(args.out_root, args.results_dir, layers, args.position, args.arms)
    else:
        run_dmprobe(args.out_root, args.results_dir, args.position, args.arms)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
