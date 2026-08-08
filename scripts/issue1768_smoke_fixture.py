"""#1768 local smoke fixtures — tiny-real inputs for the GPU-bound carve-out legs.

The production drivers are smoked THROUGH their own entrypoints on a tiny
same-arch model (real Qwen tokenizer + chat template, 2-layer random-weight
Qwen2, CPU). The ONLY legs this builder substitutes are the GPU-mandatory
boundaries (gotchas: tiny-real CPU e2e / GPU-bound carve-out):

- ``build``       tiny model + a real-shape corpus manifest dir + valtest file
- ``seed-rows``   raw-row shards for p2 units via REAL HF greedy generation on
                  the tiny model, through the production prompt render
                  (`_build_generation_prompts`) — the driver's gen-resume path
                  (a PRODUCTION crash-resume branch) then runs span+TF+pooling
                  for real. vLLM itself is GPU-only (signature-smoked).
- ``seed-panels`` panel stores in the production writer schema (p4's own
                  vLLM generation is GPU-only; the pod pilot covers it)
- ``seed-rb``     tiny r_B fixture stacks for the p9 `--rb-dir` override

Prompts are benign synthetic text (never real LMSYS/WildChat rows — the
refusal-hygiene rule); the REAL corpus path is exercised on the pod pilot.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402

import issue1768_cells as X  # noqa: E402

TOPICS = [
    "photosynthesis",
    "tides",
    "sorting algorithms",
    "the water cycle",
    "compound interest",
    "volcano formation",
    "bread baking",
    "bicycle gears",
    "rainbows",
    "magnetism",
]


def _pool_prompts(n: int, tag: str) -> list[str]:
    return [
        f"Explain {TOPICS[i % len(TOPICS)]} ({tag} variant {i}) in one short sentence."
        for i in range(n)
    ]


def cmd_build(args) -> None:
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(X.BASE_MODEL)

    model_dir = root / "tinymodel"
    if not (model_dir / "config.json").exists():
        cfg = Qwen2Config(
            vocab_size=len(tok),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=4096,
            tie_word_embeddings=False,
        )
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(cfg)
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        print(f"[fixture] tiny model -> {model_dir}")

    man_dir = root / "sampling_manifest"
    man_dir.mkdir(exist_ok=True)
    lm = _pool_prompts(44, "lm")
    wc = _pool_prompts(36, "wc")
    pool = [
        {"prompt": p, "corpus": c, "stream_pos": i, "i": i}
        for i, (p, c) in enumerate([(p, "lmsys") for p in lm] + [(p, "wildchat") for p in wc])
    ]
    (man_dir / "part_00000.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool)
    )
    (man_dir / "meta.json").write_text(
        json.dumps(
            {
                "n_new": len(pool),
                "n_lmsys": len(lm),
                "n_wildchat": len(wc),
                "n_parts": 1,
                "fixture": "issue1768 smoke",
            }
        )
    )
    vt = [f"Held-out question {i}: describe {TOPICS[i % len(TOPICS)]} briefly." for i in range(20)]
    (root / "valtest_prompts.json").write_text(json.dumps({"prompts": vt}))
    print(f"[fixture] manifest ({len(pool)} rows) + valtest (20) -> {root}")


def cmd_seed_rows(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import issue1768_capture as cap
    from explore_persona_space.analysis.representation_shift import (
        GENERATION_ROW_KEYS,
        _build_generation_prompts,
    )

    out_root = Path(args.out_root)
    sample = X.load_corpus_sample(out_root)
    prompts = [r["prompt"] for r in sample["rows"]]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    for unit in args.units.split(","):
        out_dir = out_root / "corpus_capture" / unit
        if (out_dir / "raw_rows.done.json").exists():
            print(f"[fixture] {unit}: rows already seeded")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered, keys = _build_generation_prompts(tok, {unit: None}, prompts)
        rows = []
        for (p_name, q_idx), text in zip(keys, rendered, strict=True):
            ids = tok(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = model.generate(
                    torch.tensor([ids]),
                    max_new_tokens=6,
                    min_new_tokens=2,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            resp = out[0, len(ids) :].tolist()
            if resp and resp[-1] == tok.eos_token_id:
                resp = resp[:-1]
            row = {
                "persona": p_name,
                "question_idx": q_idx,
                "prompt_token_ids": ids,
                "response_token_ids": resp,
                "finish_reason": "length",
            }
            assert GENERATION_ROW_KEYS <= set(row), row.keys()
            row["prompt_sha"] = X.prompt_sha(prompts[q_idx])
            row["response_text"] = tok.decode(resp)
            rows.append(row)
        cap._append_shard(out_dir, rows)
        cap._atomic_json(out_dir / "raw_rows.done.json", {"n_rows": len(rows), "fixture": True})
        print(f"[fixture] seeded {len(rows)} rows -> {out_dir}")


def _seed_prefixed_rows(args, register_fn, unit_cid_fn, tree: str, tag: str) -> None:
    """Shared lad/brl row seeder: raw-row shards for prefixed capture units
    via REAL HF greedy generation on the tiny model through the production
    prompt render (`_build_generation_prompts` with the prefix's
    `prior_turns` — the exact kwargs the production engine call passes); the
    driver's gen-resume path then runs span+TF+pooling for real. vLLM itself
    is GPU-only (carve-out)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import issue1768_capture as cap
    from explore_persona_space.analysis.representation_shift import (
        GENERATION_ROW_KEYS,
        _build_generation_prompts,
    )

    out_root = Path(args.out_root)
    register_fn(out_root)
    sample = X.load_pfx_sample(out_root)
    prompts = [r["prompt"] for r in sample["rows"]]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    from explore_persona_space.artifacts.context import CONTEXTS

    for unit in args.units.split(","):
        ctx = CONTEXTS[unit_cid_fn(unit)]
        out_dir = out_root / tree / "corpus_capture" / unit
        if (out_dir / "raw_rows.done.json").exists():
            print(f"[fixture] {unit}: rows already seeded")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered, keys = _build_generation_prompts(
            tok,
            {unit: ctx.system},
            prompts,
            user_wraps={unit: ctx.user_wrap},
            prior_turns={unit: tuple(ctx.prefix_turns)},
        )
        rows = []
        for (p_name, q_idx), text in zip(keys, rendered, strict=True):
            ids = tok(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = model.generate(
                    torch.tensor([ids]),
                    max_new_tokens=6,
                    min_new_tokens=2,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            resp = out[0, len(ids) :].tolist()
            if resp and resp[-1] == tok.eos_token_id:
                resp = resp[:-1]
            row = {
                "persona": p_name,
                "question_idx": q_idx,
                "prompt_token_ids": ids,
                "response_token_ids": resp,
                "finish_reason": "length",
            }
            assert GENERATION_ROW_KEYS <= set(row), row.keys()
            row["prompt_sha"] = X.prompt_sha(prompts[q_idx])
            row["response_text"] = tok.decode(resp)
            rows.append(row)
        cap._append_shard(out_dir, rows)
        cap._atomic_json(out_dir / "raw_rows.done.json", {"n_rows": len(rows), "fixture": True})
        print(f"[fixture] seeded {len(rows)} {tag} rows -> {out_dir}")


def cmd_seed_lad_rows(args) -> None:
    """Round-4 lad2 units (`<name>@r_*`) — see `_seed_prefixed_rows`."""
    _seed_prefixed_rows(
        args, X.register_r4_ladder_contexts, X.r4_unit_context_id, "on_target_r4", "lad"
    )


def cmd_seed_brl_rows(args) -> None:
    """Round-5 brl2 units (`<name>@b_rel*`) — see `_seed_prefixed_rows`."""
    _seed_prefixed_rows(
        args, X.register_r5_brel_contexts, X.r5_unit_context_id, "on_target_r5", "brl"
    )


def cmd_seed_panels(args) -> None:
    import numpy as np
    import torch

    out_root = Path(args.out_root)
    rng = np.random.default_rng(11)
    layers = [int(x) for x in args.layers.split(",")]
    d = int(args.hidden)
    arms = args.arms.split(",")
    beh_keys = sorted({a.split("-")[0] for a in arms})
    ctxs_by_beh = {b: [f"{b}_src_ctx", "default", f"{b}_conv", f"{b}_icl"] for b in beh_keys}

    def store(path: Path, ctxs: list[str], shift: float) -> None:
        row_meta, resp, ctxm = [], [], []
        for cid in ctxs:
            for q in range(4):
                row_meta.append({"context_id": cid, "question_idx": q})
                resp.append(rng.standard_normal(d) + shift)
                ctxm.append(rng.standard_normal(d))
        obj = {
            "schema_version": 1,
            "row_meta": row_meta,
            "arms": {
                span: {li: torch.as_tensor(np.asarray(mat), dtype=torch.float16) for li in layers}
                for span, mat in (("response", resp), ("context", ctxm), ("prefix", ctxm))
            },
            "metadata": {"fixture": True},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)

    for b in beh_keys:
        store(out_root / "panel_capture" / f"base_{b}" / "pooled.pt", ctxs_by_beh[b], 0.0)
    for a in arms:
        b = a.split("-")[0]
        store(out_root / "panel_capture" / a / "pooled.pt", ctxs_by_beh[b], 2.0)
        store(out_root / "panel_capture_tf" / a / "pooled.pt", ctxs_by_beh[b], 1.5)
    print(f"[fixture] panels seeded for {arms} (layers={layers}, d={d})")


def cmd_seed_rb(args) -> None:
    import numpy as np
    import torch

    rb_dir = Path(args.rb_dir)
    rb_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)
    n_layers, d = int(args.n_layers), int(args.hidden)
    import issue1768_directions as dirs

    for hub_path in dirs.RB_HUB_PATHS.values():
        torch.save(
            {"rb": torch.as_tensor(rng.standard_normal((n_layers, d)), dtype=torch.float32)},
            rb_dir / Path(hub_path).name,
        )
    print(f"[fixture] rb stacks -> {rb_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", required=True)
    s = sub.add_parser("seed-rows")
    s.add_argument("--out-root", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--units", required=True)
    sl = sub.add_parser("seed-lad-rows")
    sl.add_argument("--out-root", required=True)
    sl.add_argument("--model", required=True)
    sl.add_argument("--units", required=True)
    sb = sub.add_parser("seed-brl-rows")
    sb.add_argument("--out-root", required=True)
    sb.add_argument("--model", required=True)
    sb.add_argument("--units", required=True)
    p = sub.add_parser("seed-panels")
    p.add_argument("--out-root", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--layers", default="0,1")
    p.add_argument("--hidden", default="16")
    r = sub.add_parser("seed-rb")
    r.add_argument("--rb-dir", required=True)
    r.add_argument("--n-layers", default="2")
    r.add_argument("--hidden", default="16")
    args = ap.parse_args()
    {
        "build": cmd_build,
        "seed-rows": cmd_seed_rows,
        "seed-lad-rows": cmd_seed_lad_rows,
        "seed-brl-rows": cmd_seed_brl_rows,
        "seed-panels": cmd_seed_panels,
        "seed-rb": cmd_seed_rb,
    }[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
