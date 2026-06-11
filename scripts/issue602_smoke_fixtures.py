#!/usr/bin/env python3
"""#602 CPU-smoke fixtures — tiny stub model/adapter + stand-in generations.

The pod phases 1b/1c are GPU-bound only because of MODEL SIZE; the code
paths themselves are size-agnostic. This builder creates:

1. a 28-layer tiny-random Qwen2 causal LM (hidden 64) with the REAL
   Qwen-2.5-7B-Instruct tokenizer (so the marker token-id assert, the
   chat template, and all prompt construction run for real on CPU);
2. a tiny LoRA adapter on that stub (q/v_proj, r=4);
3. the Phase-1a output files the dispatcher's GPU vLLM phase would
   write, built from the REAL generation worklist (real panel contexts,
   real training-mix prompts, real demo/probe construction) with
   CLEARLY-LABELED stand-in response texts — only the response strings
   are synthetic; every key/shape is the production contract.

Usage:
    uv run python scripts/issue602_smoke_fixtures.py --out /tmp/i602_stub \
        [--smoke-cells refusal518__assistant__s42 marker519__medical_doctor__s42]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis import i602_bakeoff as bk  # noqa: E402

logger = logging.getLogger("issue602_smoke_fixtures")

STAND_IN_RESPONSE = (
    "Stand-in smoke response: this synthetic text exists only to exercise the "
    "teacher-forced read path at tiny scale."
)


def build_stub_model(out_dir: Path) -> tuple[Path, Path]:
    """Tiny-random 28-layer Qwen2 LM + LoRA adapter, real Qwen tokenizer."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    base_dir = out_dir / "base"
    adapter_dir = out_dir / "adapter"
    if (base_dir / "config.json").exists() and (adapter_dir / "adapter_config.json").exists():
        logger.info("stub model + adapter already present at %s", out_dir)
        return base_dir, adapter_dir
    tokenizer = AutoTokenizer.from_pretrained(bk.BASE_MODEL_ID, trust_remote_code=True)
    assert tokenizer.encode(bk.MARKER_TEXT, add_special_tokens=False) == [bk.MARKER_TOKEN_ID]
    cfg = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=28,  # >= max(bk.LAYERS)+1 so the real layer indices resolve
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
    )
    torch.manual_seed(602)
    model = Qwen2ForCausalLM(cfg)
    base_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    peft_model = get_peft_model(model, lora)
    peft_model.save_pretrained(adapter_dir)
    logger.info("stub base -> %s, stub adapter -> %s", base_dir, adapter_dir)
    return base_dir, adapter_dir


def main() -> int:
    """Build all CPU-smoke fixtures."""
    parser = argparse.ArgumentParser(description="#602 CPU smoke fixtures")
    parser.add_argument("--out", default="/tmp/i602_stub")
    parser.add_argument(
        "--smoke-cells",
        nargs="*",
        default=None,
        help="Cell subset to build generation stand-ins for (default: smoke default cells)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    out_dir = Path(args.out)
    build_stub_model(out_dir)

    # Stand-in generations from the REAL worklist (prompts are real; only the
    # response strings are synthetic, and say so).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "issue602_extract_dispatch", REPO / "scripts" / "issue602_extract_dispatch.py"
    )
    disp = importlib.util.module_from_spec(spec)
    sys.modules["issue602_extract_dispatch"] = disp
    spec.loader.exec_module(disp)

    ns = argparse.Namespace(
        cells=args.smoke_cells,
        smoke=True,
        e2_ks=[str(k) for k in bk.E2_K_SWEEP],
    )
    cells = disp.resolve_cells(ns.cells, smoke=True)
    units = disp.active_units(cells)
    work = disp.build_generation_worklist(cells, units, ns)
    work.pop("_i406_meta", None)
    gen_dir = bk.eval_dir(REPO) / "base_generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    for name, entries in work.items():
        path = gen_dir / f"{name}.json"
        if name.startswith("panel__"):
            nested: dict[str, dict[str, str]] = {}
            for key in entries:
                ctx, q = key.split("␟", 1)
                nested.setdefault(ctx, {})[q] = STAND_IN_RESPONSE
            path.write_text(json.dumps(nested, indent=1))
        else:
            path.write_text(json.dumps(dict.fromkeys(entries, STAND_IN_RESPONSE), indent=1))
        logger.info("stand-in %s.json (%d entries)", name, len(entries))
    logger.info("fixtures complete (cells: %s)", [c["cell_id"] for c in cells])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
