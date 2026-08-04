#!/usr/bin/env python3
"""#1768 inline round: LAST-PROMPT-TOKEN context capture (pooling re-test).

Round 1 fit the context->answer maps on SPAN-MEAN-over-prompt context inputs
(the fleet capture convention, inherited not chosen). #779's own comparison
measured the LAST-PROMPT-TOKEN summary better by ~0.07 across predictors and
layers, and #1768's gate-failure headline (whitened-similarity median 0.14) is
exposed to context-summary attenuation. This module re-captures the CONTEXT
side only, at single token positions, so the round-1 answer-side stores can be
re-fit unchanged (join by prompt sha).

Two positions are captured in ONE forward pass (both free once the hooks fire):

``last_prompt``
    index ``len(prompt_token_ids) - 1`` -- the final token of the rendered
    prompt (the position generation reads from). This is the #779 convention
    and the PRIMARY read of this round.
``last_ctx``
    index ``context_len - 1`` -- the final token of the user query, i.e. the
    last token of the span round 1 averaged over. The span-MATCHED last-token
    variant, kept as the secondary read so "which token did the convention
    mean" is answerable from one capture rather than a second GPU pass.

Prompt token ids are REUSED from round 1's own persisted rows (the
``corpus_capture/base_content`` raw-row shards + ``rows_spans.json`` on the
data repo) rather than re-rendered, so the capture is byte-identical to the
round-1 prompts by construction; a per-unit tokenizer parity assert catches a
unit whose tokenizer would have rendered something else.
"""

from __future__ import annotations

import argparse
import gc
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

load_dotenv()

import issue1768_capture as CAP  # noqa: E402
import issue1768_cells as X  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.lasttoken")

LT_PREFIX = f"{X.HF_PREFIX}/lasttoken_ctx"
POSITIONS = ("last_prompt", "last_ctx")
PROMPT_SOURCE_UNIT = "base_content"  # bare render -> prompt ids are unit-invariant
FWD_BATCH = int(os.environ.get("EPM_LT_FWD_BATCH", "16"))


def _meta() -> dict:
    return CAP._meta()


def _atomic_json(path: Path, obj: dict) -> None:
    CAP._atomic_json(path, obj)


# ── prompt-id staging (round-1 rows, reused verbatim) ────────────────────────


def prompt_rows(out_root: Path) -> list[dict]:
    """Round-1 prompt token ids + spans, question-index ordered.

    Staged ONCE from the round-1 ``base_content`` capture: the corpus render is
    BARE (no system prompt, no user wrap, no prior turns -- ``_attach_spans``
    defaults), so the rendered prompt depends only on the tokenizer and is the
    same for every unit. ``capture_unit`` re-renders one prompt with the unit's
    own tokenizer and asserts the ids match, so a unit shipping a divergent
    tokenizer fails loud instead of silently capturing at a shifted index.
    """
    cache = out_root / "lt_prompt_rows.json"
    if cache.exists():
        rows = json.loads(cache.read_text())["rows"]
        logger.info("[prompts] reused cache: %d rows", len(rows))
        return rows

    from explore_persona_space.orchestrate import hub

    stage = out_root / "lt_prompt_src"
    stage.mkdir(parents=True, exist_ok=True)
    src = f"{X.HF_PREFIX}/corpus_capture/{PROMPT_SOURCE_UNIT}"
    hub.stage_hub_prefix(X.HF_DATA_REPO, src, stage, repo_type="dataset")
    # stage_hub_prefix mirrors the hub-relative tree under dest_dir
    base = stage / src
    assert (base / "rows_spans.json").exists(), f"staged tree missing rows_spans.json: {base}"

    spans = {
        (s["prompt_sha"], s["question_idx"]): s
        for s in json.loads((base / "rows_spans.json").read_text())["rows"]
    }
    rows: list[dict] = []
    shard_paths = sorted(base.glob("raw_rows_*.jsonl"))
    assert shard_paths, f"no raw-row shards under {base}"
    for shard in shard_paths:
        # text-mode iteration: never .splitlines() on JSONL (raw U+2028 in real text)
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["prompt_sha"], r["question_idx"])
                sp = spans.get(key)
                if sp is None:  # round-1 empty-response drop; prompt still valid
                    sp = {"prefix_len": None, "context_len": None}
                rows.append(
                    {
                        "prompt_sha": r["prompt_sha"],
                        "question_idx": r["question_idx"],
                        "prompt_token_ids": r["prompt_token_ids"],
                        "context_len": sp["context_len"],
                    }
                )
    rows.sort(key=lambda r: r["question_idx"])
    n_missing_ctx = sum(1 for r in rows if r["context_len"] is None)
    assert n_missing_ctx == 0, (
        f"{n_missing_ctx}/{len(rows)} rows lack a round-1 context_len; the last_ctx "
        "position is underivable for them -- re-derive spans instead of guessing"
    )
    for r in rows:
        p_len = len(r["prompt_token_ids"])
        assert 0 < r["context_len"] <= p_len, (r["question_idx"], r["context_len"], p_len)
    _atomic_json(cache, {"rows": rows, "n_rows": len(rows), "source": src, **_meta()})
    logger.info("[prompts] staged %d rows from %s", len(rows), src)
    return rows


# ── capture ──────────────────────────────────────────────────────────────────


def _tokenizer_parity(tokenizer, rows: list[dict], unit_id: str) -> dict:
    """Round-trip 3 round-1 prompts through the UNIT's tokenizer; assert id match.

    Guards the prompt-id reuse above: the bare render is tokenizer-only, so an
    exact id match on sampled rows proves this unit's tokenizer reproduces the
    round-1 prompts (and therefore that ``p_len - 1`` / ``context_len - 1``
    index the intended positions).
    """
    checked = 0
    for i in (0, len(rows) // 2, len(rows) - 1):
        r = rows[i]
        # decode -> re-encode the round-1 prompt ids: an equal round-trip means this
        # unit's vocabulary + merges agree with round 1's on this exact text, so the
        # recorded p_len / context_len index the tokens they were derived from.
        text = tokenizer.decode(r["prompt_token_ids"], skip_special_tokens=False)
        re_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        assert re_ids == list(r["prompt_token_ids"]), (
            f"{unit_id}: tokenizer round-trip differs from round-1 prompt ids at "
            f"question_idx={r['question_idx']} ({len(re_ids)} vs "
            f"{len(r['prompt_token_ids'])} tokens) -- this unit would capture at a "
            "shifted position"
        )
        checked += 1
    return {"tokenizer_parity_rows_checked": checked}


def capture_unit(cfg: CAP.Cfg, unit_id: str, rows: list[dict]) -> Path:
    """Prompt-only forwards -> last-token context vectors at every layer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = cfg.out_root / "lasttoken" / unit_id
    store_path = out_dir / "lasttoken.pt"
    if store_path.exists():
        logger.info("[lt] %s: store present, skip", unit_id)
        return store_path
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path, cleanup = CAP._resolve_unit_model(cfg, unit_id)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        parity = _tokenizer_parity(tokenizer, rows, unit_id)
        pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": CAP._device()},
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        model.eval()
        device = CAP._device()
        layers = list(cfg.layers)
        n_blocks = len(model.model.layers)
        for li in layers:
            assert 0 <= li < n_blocks, (li, n_blocks)

        captured: dict[int, torch.Tensor] = {}

        def make_hook(li: int):
            def hook_fn(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured[li] = hs.detach()

            return hook_fn

        hooks = [model.model.layers[li].register_forward_hook(make_hook(li)) for li in layers]
        hidden = model.config.hidden_size
        pooled: dict[str, dict[int, list]] = {p: {li: [] for li in layers} for p in POSITIONS}

        import inspect
        import time

        fwd = getattr(model, "forward", model.__call__)
        keep_kwargs = (
            {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(fwd).parameters else {}
        )
        n_batches = -(-len(rows) // FWD_BATCH)
        t0 = time.time()
        try:
            for bi, start in enumerate(range(0, len(rows), FWD_BATCH)):
                batch = rows[start : start + FWD_BATCH]
                seqs = [r["prompt_token_ids"] for r in batch]
                max_len = max(len(s) for s in seqs)
                input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
                attn = torch.zeros((len(batch), max_len), dtype=torch.long)
                for i, s in enumerate(seqs):
                    # RIGHT pad: positions index naturally from 0, so the recorded
                    # prompt indices are valid with no position_ids threading.
                    input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                    attn[i, : len(s)] = 1
                with torch.no_grad():
                    model(
                        input_ids=input_ids.to(device),
                        attention_mask=attn.to(device),
                        **keep_kwargs,
                    )
                for li in layers:
                    hs = captured[li]
                    assert hs.shape[:2] == (len(batch), max_len), (hs.shape, len(batch), max_len)
                    for i, r in enumerate(batch):
                        idx = {
                            "last_prompt": len(r["prompt_token_ids"]) - 1,
                            "last_ctx": r["context_len"] - 1,
                        }
                        for pos in POSITIONS:
                            j = idx[pos]
                            assert 0 <= j < len(r["prompt_token_ids"]), (pos, j, r["question_idx"])
                            vec = hs[i, j, :].float().cpu()
                            assert vec.shape == (hidden,), (vec.shape, hidden)
                            pooled[pos][li].append(vec)
                if bi % 100 == 0:
                    el = time.time() - t0
                    logger.info(
                        "[lt] %s batch %d/%d elapsed=%.0fs (%.1f rows/s)",
                        unit_id,
                        bi + 1,
                        n_batches,
                        el,
                        (start + len(batch)) / max(el, 1e-6),
                    )
        finally:
            for h in hooks:
                h.remove()
            captured.clear()

        wall = time.time() - t0
        store = {
            "schema_version": 1,
            "unit": unit_id,
            "row_sha": [r["prompt_sha"] for r in rows],
            "row_question_idx": [r["question_idx"] for r in rows],
            "arms": {
                pos: {li: torch.stack(pooled[pos][li]).to(torch.float16) for li in layers}
                for pos in POSITIONS
            },
            "metadata": {
                **_meta(),
                **parity,
                "model_path": str(model_path),
                "layers": layers,
                "positions": list(POSITIONS),
                "n_rows": len(rows),
                "fwd_batch": FWD_BATCH,
                "wall_s": wall,
                "rows_per_s": len(rows) / max(wall, 1e-6),
                "prompt_source": f"{X.HF_PREFIX}/corpus_capture/{PROMPT_SOURCE_UNIT}",
            },
        }
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        tmp = store_path.with_suffix(".pt.tmp")
        torch.save(store, tmp)
        os.replace(tmp, store_path)
        _atomic_json(
            out_dir / "manifest.json",
            {"unit": unit_id, "wall_s": wall, "n_rows": len(rows), **_meta()},
        )
        logger.info(
            "[lt] %s done: %d rows in %.0fs (%.1f rows/s)",
            unit_id,
            len(rows),
            wall,
            len(rows) / max(wall, 1e-6),
        )
        return store_path
    finally:
        CAP._cleanup_merged(cleanup)


def upload_unit(cfg: CAP.Cfg, unit_id: str) -> None:
    """One bulk folder commit per unit + exact-set verify (before any fit)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    out_dir = cfg.out_root / "lasttoken" / unit_id
    assert (out_dir / "lasttoken.pt").exists(), f"nothing to upload for {unit_id}"
    prefix = f"{cfg.hf_prefix}/lasttoken_ctx/{unit_id}"
    hub._upload(out_dir, X.HF_DATA_REPO, "dataset", prefix, raise_on_error=True)
    expected = [f"{prefix}/{p.name}" for p in sorted(out_dir.iterdir()) if p.is_file()]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(token=os.environ.get("HF_TOKEN")),
        X.HF_DATA_REPO,
        expected,
        path_in_repo=prefix,
        repo_type="dataset",
    )
    assert not missing, f"{unit_id}: upload verify missing {missing}"
    logger.info("[lt-upload] %s verified %d files at %s", unit_id, len(expected), prefix)


def units_for(arms_filter: str) -> list[str]:
    arms = X.all_arms()
    if arms_filter:
        want = {a.strip() for a in arms_filter.split(",") if a.strip()}
        arms = [a for a in arms if a.arm_id in want]
        assert len(arms) == len(want), (sorted(want - {a.arm_id for a in arms}),)
    bases = sorted({X.base_unit_for(a.arm_id) for a in arms})
    return bases + [a.arm_id for a in arms]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--arms", default="", help="comma arm ids (default: all 72)")
    ap.add_argument("--units", default="", help="comma unit ids (overrides --arms expansion)")
    ap.add_argument("--layers", default=",".join(str(x) for x in X.LAYERS))
    ap.add_argument("--rows-limit", type=int, default=0, help="pilot: first N rows only")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--no-upload", action="store_true")
    # default=None + fail-loud below: a hardcoded issue-prefix fallback here is
    # silently inherited by a child issue reusing this script (#1005 clobber).
    ap.add_argument("--hf-prefix", default=None, help="upload prefix (required unless --no-upload)")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        import inspect

        from explore_persona_space.orchestrate import hub

        import issue1768_capture as _c  # noqa: F401
        import issue1768_cells as _x  # noqa: F401

        # signature-bind the upload/verify call shapes: import resolution alone
        # green-lights an arity mismatch that only fires at the terminal upload
        # (#1332), which is exactly how the pilot's verify leg crashed.
        inspect.signature(hub.verify_repo_paths_uploaded).bind(
            object(), object(), object(), path_in_repo="p", repo_type="dataset"
        )
        inspect.signature(hub._upload).bind(
            object(), object(), object(), object(), raise_on_error=True
        )
        print("import-check ok (upload/verify call shapes bind)")
        return 0

    assert args.out_root is not None, "--out-root is required outside --import-check"
    if not args.no_upload and not args.hf_prefix:
        raise SystemExit(
            "--hf-prefix is required when uploading (no issue-prefix default: a "
            "silent fallback would clobber a reusing issue's prefix)"
        )
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = CAP.Cfg(
        out_root=out_root,
        phases=("lt",),
        layers=tuple(int(x) for x in args.layers.split(",")),
        hf_prefix=args.hf_prefix or "",
    )
    rows = prompt_rows(out_root)
    if args.rows_limit:
        rows = rows[: args.rows_limit]
        logger.info("[lt] pilot row limit: %d rows", len(rows))

    units = (
        [u.strip() for u in args.units.split(",") if u.strip()]
        if args.units
        else units_for(args.arms)
    )
    mine = [u for i, u in enumerate(units) if i % args.n_shards == args.shard]
    logger.info(
        "[lt] shard %d/%d owns %d/%d units", args.shard, args.n_shards, len(mine), len(units)
    )
    for k, unit_id in enumerate(mine):
        logger.info("[phase=lt_capture unit=%s %d/%d]", unit_id, k + 1, len(mine))
        capture_unit(cfg, unit_id, rows)
        if not args.no_upload:
            upload_unit(cfg, unit_id)
    # NOT `[phase=done]`: that token is RESERVED for a dispatcher's single
    # terminal line (poll_pipeline reads it as whole-run status=done), and 4
    # concurrent shards each emitting it would signal completion while three
    # shards are still capturing (#545/#930 class).
    logger.info("[shard-complete] shard %d captured %d units", args.shard, len(mine))
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
