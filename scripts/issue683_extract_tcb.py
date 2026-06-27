#!/usr/bin/env python3
# ruff: noqa: RUF003
# (math/scientific notation — α, Δ, ⟨⟩, t_{C,B} — intentional in docstrings + labels)
"""Issue #683 Phase A — base-model teacher-forced training-completion key t_{C,B}.

Plan §4 Phase A. For each source C of a behavior B, load the BASE model
``Qwen/Qwen2.5-7B-Instruct``, iterate the source's training-completion rows
(the actual {prompt, completion} pairs the adapter was trained on), run a
SINGLE batched teacher-forced forward per row group (``output_hidden_states``,
NO generation), mean-pool the ANSWER-SIDE token positions of layer ``l``, and
average over rows → the data-side key:

    t_{C,B}      = mean over training rows of [ mean over answer-side tokens of
                   h_base[l] ]                                   (3584,)
    delta_{C,B}  = t_{C,B} - v_base(C)         (the displacement key ingredient)

``v_base(C)`` is the source's base context vector at layer ``l``; it is
written by the Δv extractors (issue683_extract_dv_*.py) into the same
analysis-tensors dir, so this script writes t_{C,B} (always) and the
displacement is computed downstream where both vectors are present (we ALSO
emit delta here when a v_base bank is supplied via ``--vbase-bank``).

Memory contract (#658-class OOM avoidance): accumulate the running mean
ON-GPU and move only the pooled (H,) vector to CPU per row — the full
``hidden_states`` tensor is NEVER retained across the loop. Forwards are
BATCHED (data-parallel, never batch-1) via the memory-safe single-layer
``extract_layer_activations`` hook (output_hidden_states=False under the hook).

Content hygiene: the sycophancy training pool is harmful-adjacent. This
script NEVER prints prompt/completion text — only row counts, shapes, shas,
and the pooled vector norms.

CLI:
    uv run python scripts/issue683_extract_tcb.py --behavior marker --layer 14 \
        --source-list A1,A2,A3,A4,A5 --out-dir eval_results/issue_683/analysis_tensors/t_cb
    uv run python scripts/issue683_extract_tcb.py --behavior sycophancy --layer 20 \
        --source-list villain
    # CPU smoke (tiny throwaway model, 2 rows, base-only):
    uv run python scripts/issue683_extract_tcb.py --behavior marker --layer 1 \
        --source-list A1 --max-rows 2 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_extract_tcb")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.experiments.issue_683 import (  # noqa: E402
    BASE_MODEL,
    DEFAULT_LAYER,
    HIDDEN_SIZE,
    MARKER_MIX_TEMPLATE,
    SYCO_TRAIN_POOL,
    answer_span_token_indices,
    load_completion_rows,
    repro_metadata,
    sha256_file,
)


def _resolve_mix_path(behavior: str, source: str) -> tuple[str, str]:
    """(repo_path, behavior) for the training mix of (behavior, source).

    Marker sources are loc-arm condition codes (A1..A5); sycophancy is the
    single ``villain`` on-policy pool. Returns the HF-data-repo relative path.
    """
    if behavior == "marker":
        return MARKER_MIX_TEMPLATE.format(arm=source), behavior
    if behavior == "sycophancy":
        if source != "villain":
            raise ValueError(
                f"sycophancy source must be 'villain' (plan §11 single source); got {source!r}"
            )
        return SYCO_TRAIN_POOL, behavior
    raise ValueError(f"unknown behavior {behavior!r}")


def _download_mix(behavior: str, source: str, local_root: Path) -> Path:
    """Download the training mix to a local cache; return the local path."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.experiments.issue_683 import HF_DATA_REPO

    rel, _ = _resolve_mix_path(behavior, source)
    local = local_root / Path(rel).name
    if local.is_file():
        return local
    src = hf_hub_download(HF_DATA_REPO, rel, repo_type="dataset", revision="main")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(Path(src).read_bytes())
    logger.info("downloaded mix %s -> %s", rel, local)
    return local


def extract_t_cb(
    *,
    model,
    tokenizer,
    rows: list[dict],
    layer: int,
    device,
    batch_size: int,
) -> tuple[object, int]:
    """Mean-pool answer-side residual at ``layer`` over the rows → t_{C,B} (H,).

    Batched teacher-forced forwards (RIGHT-padded), running mean accumulated
    ON-GPU: only the per-row pooled (H,) vector ever touches the running sum;
    the (B, T, H) hidden states are freed each batch. Returns (t_cb cpu fp32, n_used).

    Padding is RIGHT-side: every row's real tokens occupy columns ``[0, len)``,
    so the answer-span column indices are the natural ``[P, ..., len-1]`` (no
    offset) and the default position_ids (0..maxlen-1) index each real prefix
    correctly. Under a causal mask + the attention_mask, the trailing pad
    cannot influence any real token's representation, so the answer-side pool
    is identical to the unpadded forward. (Left-pad would need explicit
    position_ids — feedback_left_pad_position_ids_required — which the
    extract_layer_activations forward does not thread; right-pad sidesteps it.)
    """
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations
    from explore_persona_space.experiments.issue_683 import chunked

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    orig_side = tokenizer.padding_side

    n_model_layers = int(model.config.num_hidden_layers)
    block_l = min(layer, n_model_layers - 1)
    if block_l != layer:
        logger.warning(
            "model has %d layers < requested layer %d; reading layer %d (CPU-smoke path)",
            n_model_layers,
            layer,
            block_l,
        )

    model_hidden = int(model.config.hidden_size)
    sum_acc = torch.zeros(model_hidden, dtype=torch.float64, device=device)
    n_used = 0

    # Pre-tokenize each row: full ids + the answer-span indices (prompt-prefix
    # guard runs here, fail-loud per row before any forward).
    prepared: list[tuple[list[int], list[int]]] = []
    for row in rows:
        messages = [*row["prompt"], *row["completion"]]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        ans_idx = answer_span_token_indices(tokenizer, list(row["prompt"]), full_ids)
        prepared.append((full_ids, ans_idx))

    try:
        tokenizer.padding_side = "right"
        for batch in chunked(prepared, batch_size):
            maxlen = max(len(ids) for ids, _ in batch)
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            # RIGHT-pad: real tokens at columns [0, len); answer-span indices unchanged.
            ans_cols: list[list[int]] = []
            for bi, (ids, ans_idx) in enumerate(batch):
                input_ids[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[bi, : len(ids)] = 1
                ans_cols.append(list(ans_idx))
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            captured = extract_layer_activations(model, input_ids, [block_l], attention_mask=attn)
            hs = captured[block_l]  # (B, T, H)
            assert hs.shape == (len(batch), maxlen, model_hidden), hs.shape
            for bi, cols in enumerate(ans_cols):
                col_t = torch.tensor(cols, dtype=torch.long, device=device)
                # mean over the answer-side tokens for this row, on-GPU.
                row_mean = hs[bi].index_select(0, col_t).double().mean(dim=0)  # (H,)
                sum_acc += row_mean
                n_used += 1
            # hs is freed at the next loop iteration; never retained.
            del hs, captured
    finally:
        tokenizer.padding_side = orig_side

    if n_used == 0:
        raise RuntimeError("no rows pooled — empty training mix?")
    t_cb = (sum_acc / n_used).float().cpu()
    assert t_cb.shape == (model_hidden,), t_cb.shape
    if not bool(torch.isfinite(t_cb).all()):
        raise RuntimeError("t_cb has non-finite entries")
    return t_cb, n_used


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--behavior", required=True, choices=("marker", "sycophancy"))
    ap.add_argument("--layer", type=int, default=None, help="block index; default per behavior")
    ap.add_argument(
        "--source-list",
        required=True,
        help="comma-separated sources (marker: A1..A5; sycophancy: villain)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="default eval_results/issue_683/analysis_tensors/t_cb/<behavior>",
    )
    ap.add_argument(
        "--mix-cache-dir",
        default="data/issue_683/mix_cache",
        help="local cache for the downloaded training mixes",
    )
    ap.add_argument(
        "--vbase-bank",
        default=None,
        help="optional .pt {source: (H,)} v_base bank to ALSO emit delta_{C,B}",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows (smoke uses a few)")
    ap.add_argument(
        "--model",
        default=BASE_MODEL,
        help="override base model (CPU smoke: a tiny throwaway HF model)",
    )
    ap.add_argument("--gpu-id", type=int, default=None, help="pin CUDA_VISIBLE_DEVICES")
    ap.add_argument("--smoke", action="store_true", help="smoke namespace (separate out subdir)")
    args = ap.parse_args(argv)

    if args.gpu_id is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  # before torch import

    import torch

    layer = args.layer if args.layer is not None else DEFAULT_LAYER[args.behavior]
    sources = [s.strip() for s in args.source_list.split(",") if s.strip()]
    if not sources:
        raise SystemExit("--source-list is empty")

    out_dir = Path(
        args.out_dir
        or (
            PROJECT_ROOT
            / "eval_results/issue_683/analysis_tensors"
            / ("t_cb_smoke" if args.smoke else "t_cb")
            / args.behavior
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mix_cache = PROJECT_ROOT / args.mix_cache_dir / args.behavior
    mix_cache.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info(
        "[phase=tcb_load] model=%s device=%s dtype=%s behavior=%s layer=%d sources=%s",
        args.model,
        device,
        dtype,
        args.behavior,
        layer,
        sources,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, device_map={"": device}, trust_remote_code=True
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
        ).eval()
    if args.model == BASE_MODEL:
        assert model.config.hidden_size == HIDDEN_SIZE, model.config.hidden_size

    vbase_bank = None
    if args.vbase_bank and Path(args.vbase_bank).is_file():
        vbase_bank = torch.load(args.vbase_bank, map_location="cpu", weights_only=False)

    results: dict[str, dict] = {}
    for source in sources:
        rel, _ = _resolve_mix_path(args.behavior, source)
        local = _download_mix(args.behavior, source, mix_cache)
        rows = load_completion_rows(local)
        if args.max_rows > 0:
            rows = rows[: args.max_rows]
        logger.info(
            "[phase=tcb_extract] source=%s rows=%d mix=%s sha=%s",
            source,
            len(rows),
            rel,
            sha256_file(local)[:16],
        )
        t_cb, n_used = extract_t_cb(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            layer=layer,
            device=device,
            batch_size=args.batch_size,
        )
        payload = {
            "t_cb": t_cb,  # (H,) fp32 CPU
            "behavior": args.behavior,
            "source": source,
            "layer": layer,
            "read_location": "answer_span_mean",
            "n_rows": n_used,
            "mix_path": rel,
            "mix_sha256": sha256_file(local),
            "t_cb_norm": float(t_cb.norm()),
        }
        # delta_{C,B} = t_{C,B} - v_base(C) when a v_base bank is supplied.
        if vbase_bank is not None and source in vbase_bank:
            v_base = torch.as_tensor(vbase_bank[source]).flatten().float()
            if v_base.shape == t_cb.shape:
                payload["delta_cb"] = (t_cb - v_base).float()
                payload["delta_cb_norm"] = float(payload["delta_cb"].norm())
        out_path = out_dir / f"t_cb_{args.behavior}_{source}_L{layer}.pt"
        torch.save(payload, out_path)
        results[source] = {
            "out_path": str(out_path),
            "n_rows": n_used,
            "t_cb_norm": payload["t_cb_norm"],
            "has_delta": "delta_cb" in payload,
        }
        logger.info(
            "[phase=tcb_done] source=%s n_rows=%d t_cb_norm=%.4f -> %s",
            source,
            n_used,
            payload["t_cb_norm"],
            out_path,
        )

    summary = {
        "behavior": args.behavior,
        "layer": layer,
        "read_location": "answer_span_mean",
        "sources": results,
        "reproducibility": repro_metadata(
            {"behavior": args.behavior, "layer": layer, "read_location": "answer_span_mean"}
        ),
    }
    (out_dir / f"t_cb_summary_{args.behavior}_L{layer}.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("[phase=tcb_summary] %d source(s) -> %s", len(results), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
