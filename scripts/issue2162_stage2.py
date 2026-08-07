#!/usr/bin/env python3
"""Issue #2162 — stage-2 layer x dose pod driver (plan §4.2, post-selection).

Reuses the Stage-1 driver (``issue2162_run``) wholesale — model/bank loading,
hook machinery, claim-file queue, atomic writers, upload path — and changes
ONLY the intervention geometry: instead of the all-28-layer replace, the
steered payload V_slot(B) is written at a LAYER SUBSET (dose 1 = the single
layer; dose 4 = a 4-consecutive-layer window, shifted down so it fits), at
each of ``STAGE2_LAYERS`` = {8, 12, 14, 16, 19, 22, 26}.

Grid: <=12 selected (cell, slot) units (``best_cells.json``, the analysis
stats step's Holm-IUT AND disjoint-CI survivors) x 7 layers x 2 doses x 36
pairs x K=2 draws = <=12,096 rollouts (plan §4.3). Steered arm only — the
stage-1 nulls already bound the type-level effect; stage-2 localizes it.

Rows land under ``<out_root>/stage2/shard_*.jsonl`` (the judge driver's
``load_stage2_rows`` glob) with the grid-row schema + ``layer``/``dose``/
``layers_patched``. No V_a capture and no margin pass at stage-2 (F_beh via
the judge is the stage-2 read) — recorded as a scope note in the sentinel.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2162_run as R  # noqa: E402
from explore_persona_space.experiments.issue1415.steering import generate_batch  # noqa: E402
from explore_persona_space.experiments.issue2094.hooks import joint_hooks  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK  # noqa: E402

logger = logging.getLogger("issue2162.stage2")

STAGE2_LAYERS: tuple[int, ...] = (8, 12, 14, 16, 19, 22, 26)
STAGE2_DOSES: tuple[int, ...] = (1, 4)
STAGE2_DRAWS = 2  # 12 cells x 7 layers x 2 doses x 36 pairs x 2 = 12,096 (plan §4.3)
SENTINEL_PATH = Path("/workspace/logs/issue-2162-stage2-results.json")


@dataclass(frozen=True)
class Stage2Block:
    """One schedulable stage-2 unit: (cell, slot, layer, dose)."""

    cell: str
    slot: str
    layer: int
    dose: int
    pair_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.cell}|{self.slot}|steered|L{self.layer}|d{self.dose}"

    @property
    def slug(self) -> str:
        return R.block_slug(self.key)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_ids)


def layers_for_dose(layer: int, dose: int, n_layers: int) -> tuple[int, ...]:
    """The patched layer window: dose 1 = [layer]; dose 4 = 4 consecutive
    layers starting at ``layer``, SHIFTED DOWN when it would run past the
    top (keeps the dose true instead of clipping the window)."""
    assert 0 <= layer < n_layers, (layer, n_layers)
    if dose == 1:
        return (layer,)
    start = min(layer, n_layers - dose)
    assert start >= 0, (layer, dose, n_layers)
    return tuple(range(start, start + dose))


def load_best_cells(path: Path) -> list[dict]:
    """The analysis stats step's survivors — fail loud on shape drift."""
    assert path.exists(), f"{path} missing — run issue2162_analysis --step stats first"
    payload = json.loads(path.read_text())
    cells = payload["cells"]
    assert cells, "best_cells.json carries zero survivors — stage-2 has nothing to run"
    assert len(cells) <= 12, (len(cells), "stage-2 selection cap is 12 (plan §6)")
    for rec in cells:
        assert set(rec) >= {"cell", "slot"}, sorted(rec)
        assert rec["cell"] in BANK.all_cells(), rec["cell"]
        assert rec["slot"] in ("ce", "pe"), rec["slot"]
    return cells


def enumerate_stage2_blocks(
    best_cells: list[dict], pairs: list[BANK.Pair2162], smoke: bool
) -> list[Stage2Block]:
    by_cell = BANK.pairs_by_cell(pairs)
    selected = best_cells[:1] if smoke else best_cells
    layers = STAGE2_LAYERS[:1] if smoke else STAGE2_LAYERS
    blocks: list[Stage2Block] = []
    for rec in selected:
        ids = tuple(p.pair_id for p in sorted(by_cell[rec["cell"]], key=lambda p: p.pair_id))
        if smoke:
            ids = ids[: R.SMOKE_PAIRS_PER_CELL]
        for layer in layers:
            for dose in STAGE2_DOSES:
                blocks.append(Stage2Block(rec["cell"], rec["slot"], layer, dose, ids))
    keys = [b.key for b in blocks]
    assert len(set(keys)) == len(keys), "duplicate stage-2 block keys"
    return blocks


def stage2_regime_fp(cfg: R.RunConfig, bank_sha: str) -> str:
    base = R.regime_fingerprint(cfg, bank_sha)
    return f"{base}-stage2-K{STAGE2_DRAWS}"


def _arm_hook_layer_subset(
    model,
    subset: tuple[int, ...],
    row_lengths: list[int],
    positions: list[tuple[int, ...]],
    per_row_payload: list[torch.Tensor],
    expected_prompt_len: int,
):
    """Replace-mode hook stack over ONLY the dose window's layers.

    ``per_row_payload[b]`` is the stage-1 ``(1, L_all, H)`` steered payload;
    each subset layer's hook receives its own ``(1, H)`` slice (payload
    layer index == model layer index: cfg.layers is range(n_layers))."""
    stack = joint_hooks(model, list(subset))
    per_layer = [[p[:, layer, :].contiguous() for p in per_row_payload] for layer in subset]
    stack.install()
    stack.arm_batch_per_layer(row_lengths, positions, per_layer, mode="replace", alpha=1.0)
    stack.arm(expected_prompt_len)
    return stack


@torch.no_grad()
def run_stage2_block(
    cfg: R.RunConfig,
    model,
    tok,
    bank: dict,
    block: Stage2Block,
    pairs_by_id: dict[str, BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
    contexts: dict[str, dict],
    ctx_ids_cache: dict[str, list[int]],
    regime_fp: str,
) -> dict:
    """One stage-2 block: K hooked draws per pair at the dose window."""
    base_block = R.Block(block.cell, block.slot, "steered", block.pair_ids)
    cells = R._block_cells(bank, base_block, pairs_by_id, donor_maps)
    subset = layers_for_dose(block.layer, block.dose, cfg.n_layers)

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK.context_token_ids_2162(tok, contexts[cid])
        return ctx_ids_cache[cid]

    texts_per_cell: list[list[str]] = []
    for start in range(0, len(cells), cfg.gen_batch):
        chunk = cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        rows = [ids_for(c["context_a"]) for c in chunk]
        row_lengths = [len(r) for r in rows]
        t_pad = max(row_lengths)
        stack = _arm_hook_layer_subset(
            model,
            subset,
            row_lengths,
            [(c["position"],) for c in chunk],
            [c["payload"] for c in chunk],
            t_pad,
        )
        try:
            outs = generate_batch(
                model,
                tok,
                ctx_list,
                n=STAGE2_DRAWS,
                hook=stack,
                max_new_tokens=cfg.max_new_tokens,
                temperature=R.GRID_TEMPERATURE,
                seed_base=cfg.seed_base,
                render_fn=BANK.render_context_2162,
                ids_fn=BANK.context_token_ids_2162,
            )
        finally:
            stack.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts_per_cell.extend(list(o) for o in outs)
    assert len(texts_per_cell) == len(cells)

    rows_out: list[dict] = []
    for c, texts in zip(cells, texts_per_cell, strict=True):
        pair: BANK.Pair2162 = c["pair"]
        for i, text in enumerate(texts):
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            rows_out.append(
                {
                    "block_key": block.key,
                    "cell": block.cell,
                    "slot": block.slot,
                    "arm": "steered",
                    "layer": block.layer,
                    "dose": block.dose,
                    "layers_patched": list(subset),
                    "pair_id": pair.pair_id,
                    "carrier": pair.carrier,
                    "value_a": pair.value_a,
                    "value_b": pair.value_b,
                    "context_a": pair.a,
                    "context_id": pair.a,  # audit-walker compat
                    "context_b": pair.b,
                    "position": c["position"],
                    "degenerate_pe": c["degenerate_pe"],
                    "draw": i,
                    "seed": cfg.seed_base + i,
                    "temperature": R.GRID_TEMPERATURE,
                    "n_completion_tokens": n_tok,
                    "cap_hit": R.cap_hit(n_tok, cfg.max_new_tokens),
                    "cap_hit_basis": "retokenized_completion_len >= max_new_tokens",
                    "text": text,
                }
            )
    R._write_jsonl_atomic(stage2_dir(cfg) / f"shard_{block.slug}.jsonl", rows_out)
    done = {
        "key": block.key,
        "regime_fp": regime_fp,
        "n_rows": len(rows_out),
        "layers_patched": list(subset),
        "repro": R._repro(cfg),
    }
    R._write_json_atomic(R.block_done_path(cfg.out_root, block, "stage2_blocks"), done)
    return done


def stage2_dir(cfg: R.RunConfig) -> Path:
    return cfg.out_root / "stage2"


def phase_stage2(cfg: R.RunConfig, best_cells_path: Path) -> int:
    best = load_best_cells(best_cells_path)
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = BANK.donor_assignment_2162(pairs)
    contexts = BANK.build_contexts()
    _, bank_sha = R.bank_manifest_and_sha()
    regime_fp = stage2_regime_fp(cfg, bank_sha)
    blocks = enumerate_stage2_blocks(best, pairs, cfg.smoke)
    totals = R.grid_totals(blocks, STAGE2_DRAWS)
    logger.info("[stage2] %s", totals)
    assert totals["rollouts_total"] <= 12_096 or cfg.force, (
        totals,
        "stage-2 rollout budget exceeded (plan §4.3 <=12,096) — pass --force to override",
    )
    model, tok = R.load_model_and_tokenizer(cfg)
    bank = R._load_bank(cfg)
    assert bank["bank_sha"] == bank_sha, (
        "vc_bank was captured under a different bank recipe — refuse to patch "
        f"(bank {bank['bank_sha'][:12]} vs code {bank_sha[:12]})"
    )
    ctx_ids_cache: dict[str, list[int]] = {}
    t0 = time.monotonic()

    def run_one(block: Stage2Block) -> None:
        t1 = time.monotonic()
        done = run_stage2_block(
            cfg,
            model,
            tok,
            bank,
            block,
            pairs_by_id,
            donor_maps,
            contexts,
            ctx_ids_cache,
            regime_fp,
        )
        logger.info(
            "[stage2] block %s done: %d rows in %.1fs (total %.1fs)",
            block.key,
            done["n_rows"],
            time.monotonic() - t1,
            time.monotonic() - t0,
        )

    # Queue namespace MUST equal run_stage2_block's done-file namespace
    # ("stage2_blocks") — a mismatch re-runs blocks forever (#2162 margin bug).
    stats = R.run_claim_queue(cfg, blocks, regime_fp, "stage2_blocks", run_one)
    logger.info("[stage2] queue stats: %s", stats)
    return R.RC_OK


def phase_upload(cfg: R.RunConfig) -> int:
    uploaded = {
        "stage2": R._upload_dir(
            cfg, stage2_dir(cfg), f"{R.HF_PREFIX}/raw_completions/stage2", ["shard_*.jsonl"]
        )
    }
    payload = {
        "phase": "stage2-upload",
        "uploaded": {k: len(v) for k, v in uploaded.items()},
        "n_stage2_shards": len(list(stage2_dir(cfg).glob("shard_*.jsonl"))),
        "scope_notes": [
            "stage-2 captures no V_a and runs no margin pass (F_beh via judge is the "
            "stage-2 read; plan §4.2)",
            "dose-4 windows are shifted down at the top of the stack "
            "(layers_for_dose) so the dose stays 4 true layers",
        ],
        "repro": R._repro(cfg),
    }
    try:
        SENTINEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        R._write_json_atomic(SENTINEL_PATH, payload)
    except OSError:
        # VM-side smoke runs have no /workspace; the sentinel is pod-side only.
        logger.info("[stage2] sentinel path unavailable (VM smoke) — skipped")
    R._write_json_atomic(cfg.out_root / "stage2_results.json", payload)
    print("[phase=done]", flush=True)
    return R.RC_OK


def _import_check() -> int:
    """Execute the deferred/production imports + a tiny geometry self-check."""
    assert len(STAGE2_LAYERS) == 7 and STAGE2_DOSES == (1, 4)
    assert layers_for_dose(26, 4, 28) == (24, 25, 26, 27)
    assert layers_for_dose(8, 4, 28) == (8, 9, 10, 11)
    assert layers_for_dose(12, 1, 28) == (12,)
    b = Stage2Block("instr_format", "ce", 8, 4, ("x",))
    assert b.key == "instr_format|ce|steered|L8|d4" and "__" in b.slug
    print("[import-check] issue2162_stage2 OK")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2162 stage-2 layer x dose driver.")
    ap.add_argument("--phase", default="all", choices=("stage2", "upload", "all"))
    ap.add_argument(
        "--best-cells",
        type=Path,
        default=Path("eval_results/issue_2162/f_metrics/best_cells.json"),
    )
    ap.add_argument("--out-root", type=Path, default=R.DEFAULT_OUT_ROOT)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gen-batch", type=int, default=None)
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the GPU")
    ap.add_argument("--upload", default="hf", choices=("hf", "local-mirror", "none"))
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def build_cfg(args: argparse.Namespace) -> R.RunConfig:
    argv = [
        "--phase",
        "grid",
        "--out-root",
        str(args.out_root),
        "--worker-index",
        str(args.worker_index),
        "--num-workers",
        str(args.num_workers),
        "--upload",
        args.upload,
        "--grid-draws",
        str(STAGE2_DRAWS),
    ]
    if args.tiny:
        argv.append("--tiny")
    if args.smoke:
        argv.append("--smoke")
    if args.force:
        argv.append("--force")
    if args.gen_batch is not None:
        argv += ["--gen-batch", str(args.gen_batch)]
    return R.build_config(R.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        return _import_check()
    cfg = build_cfg(args)
    rc = R.RC_OK
    if args.phase in ("stage2", "all"):
        rc = phase_stage2(cfg, args.best_cells)
        if rc != R.RC_OK:
            return rc
    if args.phase in ("upload", "all"):
        rc = phase_upload(cfg)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
