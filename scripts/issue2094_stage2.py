"""Issue #2094 — P8 stage-2 pod driver (unit E; plan §4.3/§5/§10).

Runs on the 1x H100 suffix pod ``pod-2094-s2``:

1. **stage** — scoped per-file staging from HF (`hub.stage_hub_file`, never an
   unscoped enumeration): ``issue2094_singlepos/stage2_spec/best_cells.json``
   (P7's fail-loud upload) + ``analysis_tensors/vc_bank/vc_bank.pt``.
2. **cells** — re-measure the ≤6 selected cells at temperature 1.0, K=5 draws
   (mean-aggregated downstream; LABELED post-selection, plan §6) with the SAME
   hook/generate/capture path as unit C (helpers imported from
   ``issue2094_run`` — no duplicated logic).
3. **additivity** — the OPTIONAL plan §4.4 spot-check at (ce, L14, α=1):
   shift(Δ1+Δ2) vs shift(Δ1)+shift(Δ2) on 6 direction pairs sharing a
   recipient context (greedy; self-contained — the two individual-Δ rollouts
   are regenerated here rather than reusing grid shards, 18 rollouts total vs
   the plan's ~12: a flagged, negligible-cost deviation).
4. **upload + sentinel** — rollout text -> ``raw_completions/stage2`` (the
   unit-D stage-2 walker's input dir: rows carry pair_id/setting/text/draw +
   cell), V_a captures -> ``analysis_tensors/stage2``; additivity under its own
   ``*_additivity`` prefixes (kept OUT of the judged stage2 dir); pod sentinel
   ``/workspace/logs/issue-2094-s2-results.json``.

Launch (plan §10): ``uv run python scripts/issue2094_stage2.py --run``
(pod-side; no VM thread-cap prefix — dedicated GPU box).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_run as R  # noqa: E402

logger = logging.getLogger("issue2094_stage2")

HF_DATA_REPO = R.HF_DATA_REPO
HF_PREFIX = R.HF_PREFIX
SENTINEL_NAME = "issue-2094-s2-results.json"

STAGE2_TEMPERATURE = 1.0
STAGE2_DRAWS = 5  # body: temp 1.0, K=5, mean-aggregated, labeled post-selection
ADDITIVITY_N_COMBOS = 6
ADDITIVITY_SLOT = "ce"
ADDITIVITY_DOSE_ALPHA = 1.0

# Body-verbatim stage-2 layer restriction (mirrors issue2094_analysis).
STAGE2_LAYER_RESTRICTION = {"ce": (14, 19), "pe": (14, 19, 26)}

RC_OK = 0


@dataclass
class S2Paths:
    out_root: Path

    @property
    def rollouts_dir(self) -> Path:
        return self.out_root / "stage2" / "rollouts"

    @property
    def va_dir(self) -> Path:
        return self.out_root / "stage2" / "va"

    @property
    def additivity_dir(self) -> Path:
        return self.out_root / "stage2" / "additivity"

    @property
    def inputs_dir(self) -> Path:
        return self.out_root / "stage2" / "inputs"

    @property
    def manifest_dir(self) -> Path:
        return self.out_root / "stage2" / "manifests"


# ── staging (scoped per-file; plan §10 P8 reads) ───────────────────────


def stage_inputs(paths: S2Paths, best_cells_path: Path | None, hf_revision: str | None) -> Path:
    """Stage best_cells.json + vc_bank.pt from HF (or use local overrides)."""
    from explore_persona_space.orchestrate import hub

    if best_cells_path is None:
        best_cells_path = paths.inputs_dir / "best_cells.json"
        hub.stage_hub_file(
            HF_DATA_REPO,
            f"{HF_PREFIX}/stage2_spec/best_cells.json",
            best_cells_path,
            repo_type="dataset",
            revision=hf_revision,
        )
        logger.info("[stage] best_cells.json staged")  # plan §10 P8 reads log line
    vc_bank = paths.inputs_dir / "vc_bank.pt"
    if not vc_bank.exists():
        hub.stage_hub_file(
            HF_DATA_REPO,
            f"{HF_PREFIX}/analysis_tensors/vc_bank/vc_bank.pt",
            vc_bank,
            repo_type="dataset",
            revision=hf_revision,
        )
        logger.info("[stage] vc_bank.pt staged")
    return best_cells_path


def load_best_cells(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload["cells"]
    assert cells and len(cells) <= 6, f"stage-2 expects 1..6 cells, got {len(cells)}"
    return cells


def _check_cell_restriction(cell: dict, tiny: bool) -> None:
    """Defensive re-assert of the BODY layer restriction (selection owns it)."""
    variant = cell["layer_variant"]
    assert variant.startswith("L"), f"stage-2 cells are single-layer variants, got {variant}"
    if tiny:
        return  # tiny smoke models have < 15 layers; the selection tests pin the rule
    assert int(variant[1:]) in STAGE2_LAYER_RESTRICTION[cell["slot"]], cell


# ── cell re-measurement (the unit-C hook/generate/capture path, K=5) ───


def cell_key(cell: dict) -> str:
    return "|".join(
        ["s2", cell["setting"], cell["slot"], cell["layer_variant"], cell["dose"], cell["vec_type"]]
    )


def _cell_regime(cfg: R.RunConfig, cell: dict, draws: int) -> str:
    key = json.dumps(
        {
            "cell": cell_key(cell),
            "draws": draws,
            "temperature": STAGE2_TEMPERATURE,
            "seed_base": cfg.seed_base,
            "model_id": cfg.model_id,
            "tiny": cfg.tiny,
            "max_new_tokens": cfg.max_new_tokens,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def stage2_rows_for_cell(
    cell: dict,
    ck: str,
    pair_cells: list[dict],
    texts_by_draw: list[list[str]],
    n_comp_tokens: list[int],
    max_new_tokens: int,
    alpha: float,
    mode: str,
    realized_mode: str,
    layers: tuple[int, ...],
    seed_base: int,
) -> list[dict]:
    """PURE row builder — the unit-D stage-2 walker requires pair_id / setting /
    text / draw (+ optional cell); rows are flattened (pair-major, draw-minor),
    matching the capture flattening."""
    rows: list[dict] = []
    flat = 0
    for b, pc in enumerate(pair_cells):
        for draw in range(len(texts_by_draw[b])):
            rows.append(
                {
                    "cell": ck,
                    "block_key": ck,
                    "slot": cell["slot"],
                    "layer_variant": cell["layer_variant"],
                    "layers": list(layers),
                    "dose": cell["dose"],
                    "alpha": alpha,
                    "hook_mode": mode,
                    "realized_mode": realized_mode,
                    "vec_type": cell["vec_type"],
                    "arm": "stage2",
                    "pair_id": pc["pair_id"],
                    "setting": pc["setting"],
                    "context_a": pc["context_a"],
                    "context_b": pc["context_b"],
                    "positions": pc["positions"],
                    "temperature": STAGE2_TEMPERATURE,
                    "seed": seed_base + draw,
                    "draw": draw,
                    "n_completion_tokens": n_comp_tokens[flat],
                    "cap_hit": R.cap_hit(n_comp_tokens[flat], max_new_tokens),
                    "cap_hit_basis": "retokenized_completion_len >= max_new_tokens",
                    "post_selection": True,
                    "text": texts_by_draw[b][draw],
                }
            )
            flat += 1
    return rows


@torch.no_grad()
def run_stage2_cell(
    cfg: R.RunConfig,
    paths: S2Paths,
    model,
    tok,
    bank: dict,
    cell: dict,
    pairs: list[BANK.Pair],
    eot: list[int],
    draws: int,
) -> dict:
    """One selected cell: K temp-1.0 hooked draws per pair + both-pooling V_a."""
    contexts = BANK.build_contexts()
    ck = cell_key(cell)
    slug = R.block_slug(ck)
    layers = R.layer_variant_layers(cell["layer_variant"], cfg.n_layers)
    mode, alpha, payload_kind = R._realized_mode(cell["slot"], cell["dose"], cell["vec_type"])
    realized_mode = (
        "add_full_state_patch"
        if cell["dose"] == "replace" and mode == "add"
        else ("replace" if mode == "replace" else "add")
    )
    recs = bank["per_context"]
    ids_cache = {
        cid: BANK.context_token_ids_2094(tok, contexts[cid]) for cid in {p.a for p in pairs}
    }
    pair_cells: list[dict] = []
    for pair in pairs:
        delta, state, m = R._pair_payload(bank, pair, cell["slot"], cell["vec_type"])
        payload = state if payload_kind == "state" else delta
        rec = recs[pair.a]
        pos = R.slot_positions(rec["ctx_len"], rec["prefix_end"], cell["slot"])[-m:]
        assert payload.shape[0] == len(pos), (payload.shape, pos)
        pair_cells.append(
            {
                "pair_id": pair.pair_id,
                "setting": pair.setting,
                "context_a": pair.a,
                "context_b": pair.b,
                "positions": list(pos),
                "payload": payload,
            }
        )

    texts_by_pair: list[list[str]] = []
    for start in range(0, len(pair_cells), cfg.gen_batch):
        chunk = pair_cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        row_ids = [ids_cache[c["context_a"]] for c in chunk]
        row_lengths = [len(r) for r in row_ids]
        hook = R._arm_hook_for_rows(
            model,
            cfg,
            layers,
            row_lengths,
            [tuple(c["positions"]) for c in chunk],
            [c["payload"] for c in chunk],
            mode,
            alpha,
            max(row_lengths),
        )
        try:
            outs = R.generate_batch(
                model,
                tok,
                ctx_list,
                n=draws,
                hook=hook,
                max_new_tokens=cfg.max_new_tokens,
                temperature=STAGE2_TEMPERATURE,
                seed_base=cfg.seed_base,
                # History-aware render (bank.py module note) — same seam as
                # unit C's grid path; generate_batch re-arms the hook per draw.
                render_fn=BANK.render_context_2094,
                ids_fn=BANK.context_token_ids_2094,
            )
        finally:
            hook.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts_by_pair.extend([list(o) for o in outs])
    assert len(texts_by_pair) == len(pair_cells)

    flat_ctx = [ids_cache[pc["context_a"]] for pc in pair_cells for _ in range(draws)]
    flat_text = [t for texts in texts_by_pair for t in texts]
    states = R.capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
    rows = stage2_rows_for_cell(
        cell,
        ck,
        pair_cells,
        texts_by_pair,
        states["n_completion_tokens"],
        cfg.max_new_tokens,
        alpha,
        mode,
        realized_mode,
        layers,
        cfg.seed_base,
    )
    R._write_jsonl_atomic(paths.rollouts_dir / f"s2_{slug}.jsonl", rows)
    R._save_pt_atomic(
        paths.va_dir / f"s2_{slug}.pt",
        {
            "cell": ck,
            "layers": cfg.layers,
            "index": [{"pair_id": r["pair_id"], "draw": r["draw"]} for r in rows],
            "va_span": states["va_span"],
            "va_tail": states["va_tail"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "repro": R._repro(cfg),
        },
    )
    done = {
        "cell": ck,
        "regime_fp": _cell_regime(cfg, cell, draws),
        "n_rows": len(rows),
        "n_cap_hit": sum(1 for r in rows if r["cap_hit"]),
        "n_empty": len(states["empty_rows"]),
        "repro": R._repro(cfg),
    }
    R._write_json_atomic(paths.manifest_dir / f"s2_{slug}.done.json", done)
    return done


# ── additivity spot-check (plan §4.4, optional) ────────────────────────


def additivity_combos(pairs: list[BANK.Pair], n_combos: int = ADDITIVITY_N_COMBOS) -> list[dict]:
    """~6 direction pairs SHARING a recipient context (matched-prefix pairs with
    the same context_a — both Δs then apply to one A context), round-robin over
    prefixes for coverage."""
    by_ctx: dict[str, list[BANK.Pair]] = {}
    for p in pairs:
        if p.setting == "matched_prefix":
            by_ctx.setdefault(p.a, []).append(p)
    combos: list[dict] = []
    for q in BANK.QUERY_ORDER:
        for prefix in BANK.PREFIX_ORDER:
            ctx = BANK.context_id(prefix, q)
            group = sorted(by_ctx.get(ctx, []), key=lambda p: p.pair_id)
            if len(group) >= 2:
                combos.append(
                    {
                        "combo_id": f"add{len(combos):02d}",
                        "context_a": ctx,
                        "pair_1": group[0].pair_id,
                        "pair_2": group[1].pair_id,
                    }
                )
            if len(combos) >= n_combos:
                return combos
    return combos


@torch.no_grad()
def run_additivity(
    cfg: R.RunConfig,
    paths: S2Paths,
    model,
    tok,
    bank: dict,
    pairs: list[BANK.Pair],
    eot: list[int],
) -> dict:
    """shift(Δ1+Δ2) vs shift(Δ1)+shift(Δ2) at (ce, L14, α=1), greedy.

    18 rollouts (6 combos x {d1, d2, d12}) — self-contained (the plan's ~12
    assumed grid reuse for the individual shifts; regenerating them here keeps
    the pod run free of grid staging, a flagged negligible-cost deviation).
    """
    contexts = BANK.build_contexts()
    pairs_by_id = {p.pair_id: p for p in pairs}
    steer_layer = 14 if cfg.n_layers > 14 else cfg.n_layers // 2
    combos = additivity_combos(pairs)
    if cfg.smoke:
        combos = combos[:2]
    recs = bank["per_context"]
    ids_cache: dict[str, list[int]] = {}
    rows_meta: list[dict] = []
    row_ids: list[list[int]] = []
    positions: list[tuple[int, ...]] = []
    payloads: list[torch.Tensor] = []
    for combo in combos:
        ctx = combo["context_a"]
        if ctx not in ids_cache:
            ids_cache[ctx] = BANK.context_token_ids_2094(tok, contexts[ctx])
        d1, _s1, m1 = R._pair_payload(bank, pairs_by_id[combo["pair_1"]], ADDITIVITY_SLOT, "A")
        d2, _s2, m2 = R._pair_payload(bank, pairs_by_id[combo["pair_2"]], ADDITIVITY_SLOT, "A")
        assert m1 == m2 == 1, (m1, m2)
        rec = recs[ctx]
        pos = R.slot_positions(rec["ctx_len"], rec["prefix_end"], ADDITIVITY_SLOT)[-1:]
        for role, payload in (("d1", d1), ("d2", d2), ("d12", d1 + d2)):
            rows_meta.append({"combo_id": combo["combo_id"], "role": role, **combo})
            row_ids.append(ids_cache[ctx])
            positions.append(tuple(pos))
            payloads.append(payload)

    texts: list[str] = []
    for start in range(0, len(rows_meta), cfg.gen_batch):
        sl = slice(start, start + cfg.gen_batch)
        chunk_meta = rows_meta[sl]
        ctx_list = [contexts[m["context_a"]] for m in chunk_meta]
        chunk_ids = row_ids[sl]
        row_lengths = [len(r) for r in chunk_ids]
        hook = R._arm_hook_for_rows(
            model,
            cfg,
            (steer_layer,),
            row_lengths,
            positions[sl],
            payloads[sl],
            "add",
            ADDITIVITY_DOSE_ALPHA,
            max(row_lengths),
        )
        try:
            outs = R.generate_batch(
                model,
                tok,
                ctx_list,
                n=1,
                hook=hook,
                max_new_tokens=cfg.max_new_tokens,
                temperature=0.0,
                seed_base=cfg.seed_base,
                render_fn=BANK.render_context_2094,
                ids_fn=BANK.context_token_ids_2094,
            )
        finally:
            hook.remove()
        texts.extend(o[0] for o in outs)
    assert len(texts) == len(rows_meta)

    states = R.capture_answer_states(cfg, model, tok, row_ids, texts, eot)
    out_rows = [
        {
            **m,
            "steer_layer": steer_layer,
            "alpha": ADDITIVITY_DOSE_ALPHA,
            "temperature": 0.0,
            "seed": cfg.seed_base,
            "n_completion_tokens": states["n_completion_tokens"][i],
            "cap_hit": R.cap_hit(states["n_completion_tokens"][i], cfg.max_new_tokens),
            "text": texts[i],
        }
        for i, m in enumerate(rows_meta)
    ]
    R._write_jsonl_atomic(paths.additivity_dir / "additivity.jsonl", out_rows)
    R._save_pt_atomic(
        paths.additivity_dir / "additivity_va.pt",
        {
            "combos": combos,
            "index": [{"combo_id": m["combo_id"], "role": m["role"]} for m in rows_meta],
            "va_span": states["va_span"],
            "va_tail": states["va_tail"],
            "steer_layer": steer_layer,
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "repro": R._repro(cfg),
        },
    )
    return {"n_combos": len(combos), "n_rollouts": len(out_rows), "steer_layer": steer_layer}


# ── upload + sentinel ──────────────────────────────────────────────────


def upload_and_sentinel(cfg: R.RunConfig, paths: S2Paths, additivity: dict | None) -> None:
    uploaded: dict[str, list[str]] = {}
    uploaded["stage2_text"] = R._upload_dir(
        cfg, paths.rollouts_dir, f"{HF_PREFIX}/raw_completions/stage2", ["*.jsonl"]
    )
    uploaded["stage2_va"] = R._upload_dir(
        cfg, paths.va_dir, f"{HF_PREFIX}/analysis_tensors/stage2", ["*.pt"]
    )
    # Additivity rides its OWN prefixes so the judged stage2 dir stays clean
    # (its rows are not pair_id/setting rollouts for the stage-2 walker).
    uploaded["additivity_text"] = R._upload_dir(
        cfg, paths.additivity_dir, f"{HF_PREFIX}/raw_completions/stage2_additivity", ["*.jsonl"]
    )
    uploaded["additivity_va"] = R._upload_dir(
        cfg, paths.additivity_dir, f"{HF_PREFIX}/analysis_tensors/stage2_additivity", ["*.pt"]
    )
    uploaded["stage2_manifests"] = R._upload_dir(
        cfg, paths.manifest_dir, f"{HF_PREFIX}/analysis_tensors/stage2_manifests", ["*.json"]
    )
    n_rows = 0
    cap_hits = 0
    for done in sorted(paths.manifest_dir.glob("s2_*.done.json")):
        rec = json.loads(done.read_text())
        n_rows += int(rec.get("n_rows", 0))
        cap_hits += int(rec.get("n_cap_hit", 0))
    payload = {
        "eval_numbers": {
            "stage2_cells": len(list(paths.rollouts_dir.glob("s2_*.jsonl"))),
            "stage2_rows": n_rows,
            "cap_hit_rows": cap_hits,
            "cap_hit_frac": (cap_hits / n_rows) if n_rows else 0.0,
            "additivity": additivity or {"skipped": True},
        },
        "eval_paths": sorted(
            str(p) for p in (paths.rollouts_dir, paths.va_dir, paths.additivity_dir)
        ),
        "reproducibility_card": {
            **R._repro(cfg),
            "seed_base": cfg.seed_base,
            "stage2_temperature": STAGE2_TEMPERATURE,
            "stage2_draws": STAGE2_DRAWS,
            "max_new_tokens": cfg.max_new_tokens,
            "post_selection": True,
        },
        "wandb_url": None,
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{HF_PREFIX}",
        "worktree_path": str(R.REPO_ROOT),
        "final_commit_sha": R._git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": cfg.gpu_hours_budgeted,
        "plan_deviations": [
            "additivity spot-check regenerates the two individual-direction rollouts "
            "on-pod (18 rollouts vs the plan's ~12) — self-contained, no grid staging",
            "stage-2 rows are LABELED post_selection=true (plan §6: a post-selection "
            "confirmation, never an unbiased estimate)",
        ],
        "uploaded_prefixes": {k: len(v) for k, v in uploaded.items()},
    }
    R._write_json_atomic(paths.manifest_dir / "stage2_upload_done.json", payload)
    sentinel = cfg.log_dir / SENTINEL_NAME
    R._write_json_atomic(
        sentinel,
        {"sentinel_schema_version": 1, "kind": "epm:results", "version": 1, "note": payload},
    )
    logger.info("[upload] sentinel written: %s", sentinel)


# ── entrypoint ─────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve EVERY deferred import this driver reaches on its real paths."""
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        _upload_folder_filtered,
        stage_hub_file,
    )

    print("[import-check] OK", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2094 stage-2 pod driver (P8; 1x H100 pod-2094-s2)."
    )
    ap.add_argument("--run", action="store_true", help="execute stage->cells->upload")
    ap.add_argument("--out-root", type=Path, default=R.DEFAULT_OUT_ROOT)
    ap.add_argument("--log-dir", type=Path, default=R.DEFAULT_LOG_DIR)
    ap.add_argument("--model-id", default=R.MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=R.MAX_NEW_TOKENS)
    ap.add_argument("--draws", type=int, default=STAGE2_DRAWS)
    ap.add_argument("--seed-base", type=int, default=R.SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="1 pair/cell, 2 draws, 2 combos")
    ap.add_argument("--best-cells", type=Path, default=None, help="local best_cells.json")
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument("--skip-additivity", action="store_true")
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> R.RunConfig:
    """A unit-C RunConfig (phase='stage2') so model load + capture are REUSED."""
    if args.device:
        device = args.device
    elif args.tiny:
        device = "cpu"
    else:
        device = "cuda:0"
    return R.RunConfig(
        phase="stage2",
        out_root=args.out_root,
        log_dir=args.log_dir,
        model_id=args.model_id,
        tiny=args.tiny,
        n_layers=args.tiny_layers if args.tiny else R.N_MODEL_LAYERS_FULL,
        hidden=args.tiny_hidden if args.tiny else R.HIDDEN_FULL,
        device=device,
        gen_batch=args.gen_batch,
        capture_batch=args.capture_batch,
        max_new_tokens=args.max_new_tokens,
        anchor_draws=args.draws,
        seed_base=args.seed_base,
        smoke=args.smoke,
        pilot=False,
        force=False,
        worker_index=0,
        num_workers=1,
        upload_mode=args.upload,
        upload_every=0,
        planned_wall_h=0.7,  # plan §9 P8 row
        gpu_hours_budgeted=0.7,
    )


def _smoke_best_cells(cfg: R.RunConfig, path: Path) -> Path:
    """Synthesize a 1-cell best_cells.json for the no-network tiny smoke."""
    mid = cfg.n_layers // 2
    payload = {
        "cells": [
            {
                "setting": "matched_prefix",
                "slot": "ce",
                "layer_variant": f"L{mid}",
                "dose": "a1",
                "vec_type": "A",
                "selected_for": ["smoke"],
                "mean_f": 0.0,
            }
        ],
        "post_selection": True,
        "smoke_synthesized": True,
    }
    R._write_json_atomic(path, payload)
    return path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return RC_OK
    assert args.run, "pass --run (or --import-check)"
    cfg = build_config(args)
    paths = S2Paths(out_root=cfg.out_root)
    for d in (paths.rollouts_dir, paths.va_dir, paths.additivity_dir, paths.manifest_dir):
        d.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[phase=stage2] smoke=%s tiny=%s draws=%d", cfg.smoke, cfg.tiny, args.draws)
    best_cells_path = args.best_cells
    if best_cells_path is None and cfg.smoke and cfg.tiny:
        best_cells_path = _smoke_best_cells(cfg, paths.inputs_dir / "best_cells.smoke.json")
        # Tiny smoke also needs a LOCAL vc_bank: build it from the tiny model.
    best_cells_path = (
        best_cells_path
        if best_cells_path is not None
        else stage_inputs(paths, None, args.hf_revision)
    )
    cells = load_best_cells(best_cells_path)
    for cell in cells:
        _check_cell_restriction(cell, cfg.tiny)

    model, tok = R.load_model_and_tokenizer(cfg)
    eot = R.eot_tail_ids(tok)
    vc_local = paths.inputs_dir / "vc_bank.pt"
    if vc_local.exists():
        bank = torch.load(vc_local, map_location="cpu", weights_only=False)
    elif cfg.smoke and cfg.tiny:
        logger.info("[stage2] tiny smoke: capturing a LOCAL vc bank")
        bank = R.capture_bank(cfg, model, tok)
    else:
        stage_inputs(paths, best_cells_path, args.hf_revision)
        bank = torch.load(vc_local, map_location="cpu", weights_only=False)

    pairs = BANK.build_pairs()
    draws = 2 if cfg.smoke else args.draws
    t0 = time.monotonic()
    for k, cell in enumerate(cells, start=1):
        ck = cell_key(cell)
        slug = R.block_slug(ck)
        done_path = paths.manifest_dir / f"s2_{slug}.done.json"
        regime = _cell_regime(cfg, cell, draws)
        if done_path.exists() and json.loads(done_path.read_text()).get("regime_fp") == regime:
            logger.info("[stage2] cell %d/%d %s SKIP (done)", k, len(cells), ck)
            continue
        cell_pairs = [p for p in pairs if p.setting == cell["setting"]]
        if cfg.smoke:
            cell_pairs = cell_pairs[:1]
        done = run_stage2_cell(cfg, paths, model, tok, bank, cell, cell_pairs, eot, draws)
        logger.info(
            "[stage2] cell %d/%d %s rows=%d cap_hit=%d elapsed=%.1fs",
            k,
            len(cells),
            ck,
            done["n_rows"],
            done["n_cap_hit"],
            time.monotonic() - t0,
        )

    additivity = None
    if not args.skip_additivity:
        additivity = run_additivity(cfg, paths, model, tok, bank, pairs, eot)
        logger.info("[stage2] additivity: %s", additivity)

    upload_and_sentinel(cfg, paths, additivity)
    logger.info("[phase=stage2_done]")
    return RC_OK


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
