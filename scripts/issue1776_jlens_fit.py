"""#1776 Phase 0.1: J-lens fit wrapper for Qwen-2.5-7B-Instruct (vendored jacobian-lens).

Adaptation layer over the vendored ``anthropics/jacobian-lens`` (Apache-2.0,
``external/jacobian-lens``, commit in VENDOR_INFO.txt):

  - ``build-prompts``: seeded pretraining-like prompt corpus (C4-class stream,
    persisted JSONL + meta sidecar; bounded fetch).
  - ``fit``: ``jlens.hf.from_hf``-wrap Qwen-2.5-7B-Instruct and fit J_ell (full
    d_model^2 per layer, all layers) on a prompt SLICE (``--shard-index`` of
    ``--n-shards``); shards run one-per-GPU on the pod and merge via
    ``JacobianLens.merge()`` (``merge`` subcommand). Checkpointed + resumable
    (jlens's own atomic checkpoint).
  - ``sanity``: the G-LENS engineering gate (plan §7) — next-token top-1
    agreement between lens logits and model logits on the vendored repo's
    bundled eval prompts must RISE with depth and beat chance >=10x over the
    last quartile of layers. ``--gate-informational`` demotes to a report
    (smoke calibration; the pod run binds).

Slot convention: jlens's ``ActivationRecorder`` hooks ``model.model.layers[k]``
forward OUTPUT — the SAME block-output convention as ``extract_layer_activations``
(extraction.py:174-176) and ``DeltaHook`` (steering.py:194), so the fitted J's
layer indices are directly the plan's slot-pinned indices (plan §4).

Tiny-real CPU smoke: ``--tiny`` builds a from-config 4-layer Qwen2 model over
the REAL tokenizer vocab-id space (weights random; shapes/dtypes/tokenizer real).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847 gate)

C76.add_jlens_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402
from jlens import fitting as jfit  # noqa: E402
from jlens.hf import from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

N_LAYERS_QWEN7B = 28


def load_lens_model(model_name: str, *, device: str, tiny: bool = False):
    """Load (or, ``tiny``, construct from-config) the model and wrap as a LensModel.

    Returns ``(lens_model, hf_model, tokenizer)``. For the real model asserts
    the plan-pinned geometry (28 layers, hidden 3584) so a silent checkpoint
    swap cannot mislabel the slot-pinned layer indices.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tiny:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        cfg = Qwen2Config(
            vocab_size=len(tok),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
        torch.manual_seed(0)
        hf_model = Qwen2ForCausalLM(cfg).to(torch.float32).eval()
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        ).eval()
    lens_model = from_hf(hf_model, tok)
    if not tiny:
        assert lens_model.n_layers == N_LAYERS_QWEN7B, lens_model.n_layers
        assert lens_model.d_model == C.EXPECTED_HIDDEN, lens_model.d_model
        assert lens_model.layers is hf_model.model.layers, (
            "jlens layout did not resolve model.model.layers — slot convention broken"
        )
    return lens_model, hf_model, tok


def build_prompts(args) -> Path:
    """Seeded pretraining-like prompt corpus (plan §4 Phase 0.1), persisted.

    Streams ``--dataset`` (C4-class; plan: 'fineweb/C4-class sample, seeded,
    persisted'), buffer-shuffles with ``--seed``, keeps the first ``--n``
    documents with >= ``--min-chars`` chars, truncates each to ``--max-chars``.
    Bounded fetch (n<=10^4 scan with a fixed stop) — exempt from the
    external-stream checkpoint presumption (code-style.md).
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Pin + record the dataset REVISION so the "seeded, persisted" corpus is
    # regenerable (review v1 Minor: a seed alone does not pin a mutable repo).
    revision = args.dataset_revision or HfApi().dataset_info(args.dataset).sha
    ds = load_dataset(
        args.dataset, args.dataset_config, split="train", streaming=True, revision=revision
    )
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    rows: list[dict] = []
    scanned = 0
    for ex in ds:
        scanned += 1
        text = ex.get("text") or ""
        if len(text) < args.min_chars:
            continue
        rows.append({"i": len(rows), "text": text[: args.max_chars]})
        if len(rows) >= args.n:
            break
        if scanned > 50 * args.n:
            raise RuntimeError(f"scanned {scanned} rows but kept only {len(rows)} — filter bug?")
    assert len(rows) == args.n, (len(rows), args.n)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out)
    meta = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_revision": revision,
        "seed": args.seed,
        "n": args.n,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "scanned": scanned,
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(out.with_suffix(".meta.json"), meta)
    print(f"[jlens] [phase=prompts_done] {args.n} prompts -> {out} (scanned {scanned})", flush=True)
    return out


def _load_prompts(path: Path, *, shard_index: int, n_shards: int, limit: int | None) -> list[str]:
    """Contiguous prompt slice for this shard (jlens README sharding recipe)."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    assert 0 <= shard_index < n_shards, (shard_index, n_shards)
    return [r["text"] for i, r in enumerate(rows) if i % n_shards == shard_index]


def cmd_fit(args) -> None:
    """Fit J_ell on this shard's prompt slice and save a shard lens."""
    lens_model, _, _ = load_lens_model(args.model, device=args.device, tiny=args.tiny)
    prompts = _load_prompts(
        args.prompts, shard_index=args.shard_index, n_shards=args.n_shards, limit=args.limit
    )
    assert prompts, "empty prompt slice"
    layers = args.layers if args.layers else None  # None = all layers below target
    t0 = time.time()
    lens = jfit.fit(
        lens_model,
        prompts,
        source_layers=layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        checkpoint_every=args.checkpoint_every,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(args.out))
    print(
        f"[jlens] [phase=fit_done shard={args.shard_index}/{args.n_shards}] "
        f"n_prompts={lens.n_prompts} layers={lens.source_layers[:3]}..{lens.source_layers[-1:]} "
        f"elapsed={time.time() - t0:.1f}s -> {args.out}",
        flush=True,
    )


def cmd_merge(args) -> None:
    """Merge shard lenses (n_prompts-weighted mean) into one lens."""
    lenses = [JacobianLens.load(str(p)) for p in args.shards]
    merged = JacobianLens.merge(lenses)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(args.out))
    print(
        f"[jlens] [phase=merge_done] {len(lenses)} shards, n_prompts={merged.n_prompts} "
        f"-> {args.out}",
        flush=True,
    )


def sanity_gate(lens: JacobianLens, lens_model, *, prompts: list[str] | None = None) -> dict:
    """G-LENS (plan §7): per-layer next-token top-1 agreement lens vs model.

    Over the vendored repo's bundled eval prompts: agreement(layer) must rise
    with depth (positive layer-agreement correlation) and the LAST QUARTILE
    mean must beat top-1 chance (~1/vocab) by >=10x.
    """
    if prompts is None:
        from jlens.examples import EXAMPLES, resolve_prompt

        prompts = [resolve_prompt(ex, lens_model.tokenizer) for ex in EXAMPLES]
    layers = lens.source_layers
    agree_counts = {li: 0 for li in layers}
    n_positions = 0
    for prompt in prompts:
        lens_logits, model_logits, _ = lens.apply(lens_model, prompt, layers=layers)
        model_top1 = model_logits.argmax(dim=-1)
        n_positions += int(model_top1.numel())
        for li in layers:
            agree_counts[li] += int((lens_logits[li].argmax(dim=-1) == model_top1).sum())
    agreement = {li: agree_counts[li] / max(n_positions, 1) for li in layers}
    vals = np.array([agreement[li] for li in layers], dtype=float)
    trend = float(np.corrcoef(np.arange(len(layers)), vals)[0, 1]) if len(layers) > 2 else 0.0
    q = max(1, len(layers) // 4)
    last_quartile_mean = float(vals[-q:].mean())
    vocab = int(lens_model.unembed(torch.zeros(1, lens_model.d_model)).shape[-1])
    chance = 1.0 / vocab
    passed = (trend > 0.0) and (last_quartile_mean >= 10.0 * chance)
    return {
        "per_layer_agreement": {str(li): agreement[li] for li in layers},
        "trend_corr": trend,
        "last_quartile_mean": last_quartile_mean,
        "chance_top1": chance,
        "n_positions": n_positions,
        "n_prompts": len(prompts),
        "pass": bool(passed),
    }


def cmd_sanity(args) -> int:
    lens_model, _, _ = load_lens_model(args.model, device=args.device, tiny=args.tiny)
    lens = JacobianLens.load(str(args.lens))
    report = sanity_gate(lens, lens_model)
    report["repro"] = C76.repro_meta()
    report["gate"] = "G-LENS"
    report["informational"] = bool(args.gate_informational)
    C76.atomic_write_json(args.out, report)
    status = "PASS" if report["pass"] else "FAIL"
    print(
        f"[jlens] [phase=sanity_done] G-LENS {status} trend={report['trend_corr']:.3f} "
        f"lastQ={report['last_quartile_mean']:.4f} chance={report['chance_top1']:.2e} "
        f"-> {args.out}",
        flush=True,
    )
    if not report["pass"] and not args.gate_informational:
        return 8  # ENGINEERING gate HALT (plan §7): port bug — fix before any J-space number
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-prompts", help="seeded C4-class prompt corpus (persisted)")
    p.add_argument("--out", type=Path, default=C76.DATA_DIR / "jlens_prompts.jsonl")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--dataset-config", default="en")
    p.add_argument(
        "--dataset-revision", default=None, help="pin (default: resolve current sha, recorded)"
    )
    p.add_argument("--min-chars", type=int, default=600)
    p.add_argument("--max-chars", type=int, default=4000)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)

    f = sub.add_parser("fit", help="fit J_ell on a prompt shard")
    f.add_argument("--model", default=C.DEFAULT_MODEL)
    f.add_argument("--prompts", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--shard-index", type=int, default=0)
    f.add_argument("--n-shards", type=int, default=1)
    f.add_argument("--limit", type=int, default=None, help="cap total prompts (smoke)")
    f.add_argument("--layers", type=int, nargs="*", default=None, help="default: all")
    f.add_argument("--dim-batch", type=int, default=8)
    f.add_argument("--max-seq-len", type=int, default=128)
    f.add_argument("--skip-first", type=int, default=jfit.SKIP_FIRST_N_POSITIONS)
    f.add_argument("--checkpoint", type=Path, default=None)
    f.add_argument("--checkpoint-every", type=int, default=25)
    f.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    f.add_argument("--tiny", action="store_true", help="from-config tiny Qwen2 (CPU smoke)")

    m = sub.add_parser("merge", help="merge shard lenses")
    m.add_argument("--shards", type=Path, nargs="+", required=True)
    m.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("sanity", help="G-LENS gate (plan §7)")
    s.add_argument("--model", default=C.DEFAULT_MODEL)
    s.add_argument("--lens", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.add_argument("--tiny", action="store_true")
    s.add_argument("--gate-informational", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "build-prompts":
        build_prompts(args)
        return 0
    if args.cmd == "fit":
        cmd_fit(args)
        return 0
    if args.cmd == "merge":
        cmd_merge(args)
        return 0
    if args.cmd == "sanity":
        return cmd_sanity(args)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    sys.exit(main())
