"""Issue #464 ``minimal_content`` follow-up — four-float marker-slot logit capture.

CLAUDE.md marker storage contract (postdates the parent #464 run): every
marker slot read persists FOUR floats per slot per model side (trained
AND base, from the SAME forward pass per side):

    log P(marker)   = z_marker - logZ
    z_marker        = raw pre-softmax logit at the marker id
    z_eos           = raw logit at <|im_end|> (Qwen-2.5-7B id 151645)
    logZ            = logsumexp over the full vocabulary

Logits are UNRECOVERABLE from stored log-probs post-hoc, and vLLM's
logprobs API returns post-softmax log-probs only — so this capture runs
HF forward passes. The vLLM cross-eval (``i464_min_eval.py``) stays the
PRIMARY DV for parent comparability; this is the SECONDARY mechanistic
record.

Probe set: identical to the cross-eval — 6 cells (2 minimal arms x 3
seeds) x 5 minimal eval encodings x 2 markers x Q_test, scoring the slot
immediately after the spliced R_canon. Base-side stats are computed once
per (e_eval, marker) slice and cached to disk.

Gauge assert (required before any logit readout): the LoRA adapter's
``target_modules`` exclude ``lm_head`` / ``embed_tokens`` and
``modules_to_save`` is empty — the logit readout is valid only when LoRA
does not touch the unembedding ``W_U``.

Batching: right-padded batched forwards. For a causal LM, right padding
never changes logits at real (pre-pad) positions, and default
position_ids are correct for right padding — no left-pad position_ids
subtlety (cf. the #502 incident, which was specific to LEFT padding).

Outputs (atomic, per-cell, --resume aware):
    eval_results/issue_464/minimal_content/logit_capture/per_cell/
        base__{e_eval}__marker_{persona}.json
        {cell}__{e_eval}__marker_{persona}.json

Variants (``--variant``):
  * ``min`` (default — prior behavior unchanged): the 6 co-resident
    minimal cells (2 arms x 3 seeds), 5 minimal eval encodings x 2
    markers per cell, adapters at ``adapters/i464_{arm}_seed{seed}``.
  * ``min_cn`` (minimal_content_cn follow-up): the 12 single-persona cn
    cells (2 minimal arms x 3 seeds x 2 personas), the cell's 3 probe
    encodings (own / other / default_assistant) x the SHARED pirate
    marker ` ※` only, adapters at
    ``adapters/i464_{arm}_seed{seed}_cn_{persona}``, outputs under
    ``eval_results/issue_464/minimal_content_cn/logit_capture/``.
    Per-cell filenames use the same ``{arm}_seed{seed}_{persona}`` label
    as the min_cn cross-eval (the ``_cn_`` infix lives only in adapter
    subpaths).

CLI:
    uv run python scripts/i464_min_capture_logits.py --resume
    uv run python scripts/i464_min_capture_logits.py --smoke-cells system_minimal_seed42 \
        --smoke-n-q 2
    uv run python scripts/i464_min_capture_logits.py --variant min_cn --resume
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import load_q_test_extended_50

# Ensure repo root is on sys.path so `from scripts.X import Y` resolves
# when this script is invoked directly via `uv run python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.i464_phase4_eval import (  # type: ignore[import-not-found]
    BASE_MODEL,
    _build_probes_for_eval_marker,
    _download_adapter,
    _load_R_canon_test,
)

# min_cn variant reuses the cross-eval's adapter-download + per-cell
# encoding-mapping helpers so the two phases can never drift apart.
from scripts.i464_po_eval import (  # type: ignore[import-not-found]
    SHARED_MARKER_PERSONA,
    _download_po_adapter,
    _eval_encodings_for_cell,
)

load_dotenv()

logger = logging.getLogger("i464.min_capture")

OUT_DIR_FOR: dict[str, Path] = {
    "min": Path("eval_results/issue_464/minimal_content/logit_capture"),
    "min_cn": Path("eval_results/issue_464/minimal_content_cn/logit_capture"),
}
SCHEMA_VERSION_FOR: dict[str, str] = {
    "min": "i464_min_logit_capture_v1",
    "min_cn": "i464_min_cn_logit_capture_v1",
}

# Legacy aliases (min defaults) — kept for importers that referenced
# these constants before --variant existed.
OUT_DIR = OUT_DIR_FOR["min"]
PER_CELL_DIR = OUT_DIR / "per_cell"

EOS_TOKEN = "<|im_end|>"
EOS_ID = 151645

SEEDS = (42, 137, 1337)

# Gauge contract: the parent recipe targets attention + MLP projections
# only. Anything touching the unembedding invalidates the logit readout.
ALLOWED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
FORBIDDEN_TARGET_MODULES = {"lm_head", "embed_tokens"}


def _git_commit_hash() -> str:
    """Return HEAD sha or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, env={**os.environ}
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def assert_adapter_gauge(adapter_dir: str | Path) -> dict:
    """FAIL LOUD unless the adapter leaves the unembedding untouched.

    Reads ``adapter_config.json`` and asserts:
      * every entry of ``target_modules`` is in the allowed attn/mlp set;
      * no entry is ``lm_head`` / ``embed_tokens``;
      * ``modules_to_save`` is absent / None / empty.

    Returns the parsed config dict (for provenance embedding in outputs).
    """
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"gauge assert: {cfg_path} missing — cannot verify target_modules")
    cfg = json.loads(cfg_path.read_text())
    targets = cfg.get("target_modules")
    if not targets:
        raise AssertionError(f"gauge assert: adapter_config target_modules empty at {cfg_path}")
    targets_set = set(targets)
    forbidden = targets_set & FORBIDDEN_TARGET_MODULES
    if forbidden:
        raise AssertionError(
            f"gauge assert FAILED: target_modules touch the unembedding path: {forbidden} "
            f"({cfg_path}). Logit readout (z_marker, EOS margin) is INVALID."
        )
    unknown = targets_set - ALLOWED_TARGET_MODULES
    if unknown:
        raise AssertionError(
            f"gauge assert FAILED: unexpected target_modules {unknown} ({cfg_path}); "
            f"allowed: {sorted(ALLOWED_TARGET_MODULES)}"
        )
    mts = cfg.get("modules_to_save")
    if mts:
        raise AssertionError(
            f"gauge assert FAILED: modules_to_save={mts!r} non-empty ({cfg_path}); "
            "a saved full module can move W_U-adjacent weights — logit readout INVALID."
        )
    return cfg


def capture_slot_stats(
    model,
    full_ids_list: list[list[int]],
    slot_positions: list[int],
    marker_id: int,
    eos_id: int,
    pad_id: int,
    batch_size: int,
    device: str,
) -> dict[str, list[float]]:
    """Four floats per probe row from ONE forward pass per row.

    For each row ``i``, the marker token sits at ``slot_positions[i]``; its
    predictive logits are at index ``slot_positions[i] - 1`` (standard
    next-token shift). Rows are RIGHT-padded into batches — exact for a
    causal LM at pre-pad positions (default position_ids are correct).

    Returns ``{"logp": [...], "z_marker": [...], "z_eos": [...],
    "logZ": [...]}`` — each list ``len(full_ids_list)`` long. logsumexp is
    computed in float32.
    """
    if len(full_ids_list) != len(slot_positions):
        raise AssertionError(f"{len(full_ids_list)} rows vs {len(slot_positions)} slots")
    out: dict[str, list[float]] = {"logp": [], "z_marker": [], "z_eos": [], "logZ": []}
    model.eval()
    for start in range(0, len(full_ids_list), batch_size):
        chunk = full_ids_list[start : start + batch_size]
        chunk_slots = slot_positions[start : start + batch_size]
        max_len = max(len(ids) for ids in chunk)
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in chunk]
        attn = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in chunk]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        b, t, _v = logits.shape
        assert (b, t) == (len(chunk), max_len), (logits.shape, len(chunk), max_len)
        for i, slot in enumerate(chunk_slots):
            row_len = len(chunk[i])
            if not (0 < slot < row_len):
                raise AssertionError(f"slot {slot} out of range for row of {row_len} tokens")
            # Predictive logits for the token AT `slot` live at index slot-1.
            z = logits[i, slot - 1, :].float()
            log_z = torch.logsumexp(z, dim=-1)
            out["z_marker"].append(float(z[marker_id]))
            out["z_eos"].append(float(z[eos_id]))
            out["logZ"].append(float(log_z))
            out["logp"].append(float(z[marker_id] - log_z))
        del logits
    assert len(out["logp"]) == len(full_ids_list)
    return out


def _all_min_cells() -> list[tuple[enc.Arm, int]]:
    """Return the canonical 6-cell list: 2 minimal arms x 3 seeds."""
    return [(arm, seed) for arm in enc.MINIMAL_ARMS for seed in SEEDS]


def _all_min_cn_cells() -> list[tuple[enc.Arm, int, enc.Persona]]:
    """Return the min_cn 12-cell list: 2 minimal arms x 3 seeds x 2 personas."""
    return [
        (arm, seed, persona)
        for arm in enc.MINIMAL_ARMS
        for seed in SEEDS
        for persona in enc.PERSONAS
    ]


def _atomic_write(path: Path, payload: dict) -> None:
    """Write JSON atomically (tmp + replace) — checkpoint-per-phase discipline."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the four-float logit capture."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resume", action="store_true", help="Skip per-cell JSONs already written.")
    ap.add_argument("--batch-size", type=int, default=8, help="Rows per HF forward batch.")
    ap.add_argument("--device", default="cuda:0", help="Torch device for the forwards.")
    ap.add_argument(
        "--smoke-n-q",
        type=int,
        default=0,
        help="If > 0, truncate Q_test to this many questions per probe (smoke).",
    )
    ap.add_argument(
        "--smoke-cells",
        nargs="+",
        default=None,
        help=(
            "If set, restrict to these cells (min: 'system_minimal_seed42'; "
            "min_cn: 'system_minimal_seed42_pirate'); smoke use."
        ),
    )
    ap.add_argument(
        "--variant",
        choices=("min", "min_cn"),
        default="min",
        help=(
            "``min`` (default — prior behavior unchanged) = 6 co-resident "
            "minimal cells, 5 encodings x 2 markers each. ``min_cn`` = the "
            "minimal_content_cn follow-up's 12 single-persona cn cells, the "
            "cell's 3 probe encodings x the shared ` ※` marker only, outputs "
            "under ``eval_results/issue_464/minimal_content_cn/logit_capture/``."
        ),
    )
    args = ap.parse_args(argv)

    # Smoke-contamination guard (mirrors i464_min_eval.py): ANY smoke flag
    # routes output to a sibling dir so truncated captures can never
    # satisfy a production --resume or feed the analyzer's aggregation.
    smoke = args.smoke_n_q > 0 or args.smoke_cells is not None
    out_dir_active = OUT_DIR_FOR[args.variant]
    per_cell_dir = out_dir_active / ("per_cell_smoke" if smoke else "per_cell")
    if smoke:
        logger.warning("SMOKE flags set: per-cell output routed to %s", per_cell_dir)
    per_cell_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)
    live_eos = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    if live_eos != EOS_ID:
        raise AssertionError(f"{EOS_TOKEN} id drifted: {live_eos} != {EOS_ID}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else EOS_ID

    q_test = load_q_test_extended_50()
    R_canon_test = _load_R_canon_test()
    if args.smoke_n_q > 0:
        q_test = q_test[: args.smoke_n_q]
        logger.warning("SMOKE: truncated Q_test to %d questions", len(q_test))

    # ── Variant-aware cell specs ─────────────────────────────────────────
    # Each spec: label (per-cell filename stem), the adapter download
    # closure inputs, and the (e_eval, marker_persona) probe keys this
    # cell is captured under.
    cell_specs: list[dict]
    if args.variant == "min":
        cell_specs = [
            {
                "label": f"{arm}_seed{seed}",
                "arm": arm,
                "seed": seed,
                "persona": None,
                "probe_keys": [(e, mp) for e in enc.MINIMAL_EVAL_ENCODINGS for mp in enc.PERSONAS],
            }
            for arm, seed in _all_min_cells()
        ]
    else:  # min_cn — probes mirror the min_cn cross-eval exactly.
        cell_specs = [
            {
                "label": f"{arm}_seed{seed}_{persona}",
                "arm": arm,
                "seed": seed,
                "persona": persona,
                "probe_keys": [
                    (e, SHARED_MARKER_PERSONA) for e in _eval_encodings_for_cell(arm, persona)
                ],
            }
            for arm, seed, persona in _all_min_cn_cells()
        ]
    if args.smoke_cells:
        wanted = set(args.smoke_cells)
        cell_specs = [c for c in cell_specs if c["label"] in wanted]
        logger.warning("SMOKE: restricted to %d cell(s)", len(cell_specs))

    adapter_paths: dict[str, str] = {}
    for c in cell_specs:
        if args.variant == "min":
            adapter_paths[c["label"]] = _download_adapter(c["arm"], c["seed"])
        else:
            adapter_paths[c["label"]] = _download_po_adapter(
                c["arm"], c["seed"], c["persona"], variant="min_cn"
            )
    gauge_cfg = {label: assert_adapter_gauge(p) for label, p in adapter_paths.items()}
    logger.info("Gauge assert OK for %d adapters (attn/mlp-only LoRA).", len(gauge_cfg))

    # Probe slices: identical construction to the vLLM cross-eval. Build
    # the UNION of probe keys across cells (min: 5 encodings x 2 markers;
    # min_cn: 5 encodings x the shared pirate marker).
    needed_keys = sorted({key for c in cell_specs for key in c["probe_keys"]})
    probe_slices: dict[tuple[str, str], dict] = {}
    for e_eval, marker_persona in needed_keys:
        prompts_payload, slots = _build_probes_for_eval_marker(
            e_eval, marker_persona, tokenizer, q_test, R_canon_test
        )
        probe_slices[(e_eval, marker_persona)] = {
            "full_ids": [p["prompt_token_ids"] for p in prompts_payload],
            "slots": slots,
            "marker_id": enc.marker_id_for(marker_persona),
        }

    from transformers import AutoModelForCausalLM

    logger.info("Loading base model %s on %s ...", BASE_MODEL, args.device)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)

    meta = {
        "schema_version": SCHEMA_VERSION_FOR[args.variant],
        "variant": args.variant,
        "base_model": BASE_MODEL,
        "eos_id": EOS_ID,
        "git_commit": _git_commit_hash(),
        "n_probes": len(q_test),
    }

    # ── Base side: once per (e_eval, marker), persisted immediately. ────
    base_stats: dict[tuple[str, str], dict[str, list[float]]] = {}
    for (e_eval, marker_persona), sl in probe_slices.items():
        out_path = per_cell_dir / f"base__{e_eval}__marker_{marker_persona}.json"
        if args.resume and out_path.exists() and out_path.stat().st_size > 0:
            base_stats[(e_eval, marker_persona)] = json.loads(out_path.read_text())["stats"]
            continue
        t0 = time.time()
        stats = capture_slot_stats(
            base_model,
            sl["full_ids"],
            sl["slots"],
            sl["marker_id"],
            EOS_ID,
            pad_id,
            args.batch_size,
            args.device,
        )
        base_stats[(e_eval, marker_persona)] = stats
        _atomic_write(
            out_path,
            {
                **meta,
                "side": "base",
                "e_eval": e_eval,
                "marker_persona": marker_persona,
                "marker_id": sl["marker_id"],
                "stats": stats,
                "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
            },
        )
        logger.info(
            "BASE e_eval=%s marker=%s logp_mean=%.3f z_marker_mean=%.3f in %.1fs",
            e_eval,
            marker_persona,
            float(np.mean(stats["logp"])),
            float(np.mean(stats["z_marker"])),
            time.time() - t0,
        )

    # ── Trained side: one adapter attach per cell, per-slice persists. ──
    from peft import PeftModel

    for spec in cell_specs:
        cell_label = spec["label"]
        arm, seed = spec["arm"], spec["seed"]
        slice_paths = {
            key: per_cell_dir / f"{cell_label}__{key[0]}__marker_{key[1]}.json"
            for key in spec["probe_keys"]
        }
        if args.resume and all(p.exists() and p.stat().st_size > 0 for p in slice_paths.values()):
            logger.info("cell=%s fully captured; skipping (--resume).", cell_label)
            continue
        logger.info("Attaching adapter for cell=%s ...", cell_label)
        peft_model = PeftModel.from_pretrained(base_model, adapter_paths[cell_label])
        try:
            for e_eval, marker_persona in spec["probe_keys"]:
                sl = probe_slices[(e_eval, marker_persona)]
                out_path = slice_paths[(e_eval, marker_persona)]
                if args.resume and out_path.exists() and out_path.stat().st_size > 0:
                    continue
                t0 = time.time()
                stats = capture_slot_stats(
                    peft_model,
                    sl["full_ids"],
                    sl["slots"],
                    sl["marker_id"],
                    EOS_ID,
                    pad_id,
                    args.batch_size,
                    args.device,
                )
                b = base_stats[(e_eval, marker_persona)]
                d_logp = float(np.mean(stats["logp"])) - float(np.mean(b["logp"]))
                d_zm = float(np.mean(stats["z_marker"])) - float(np.mean(b["z_marker"]))
                margin_t = np.array(stats["z_marker"]) - np.array(stats["z_eos"])
                margin_b = np.array(b["z_marker"]) - np.array(b["z_eos"])
                d_margin = float(margin_t.mean() - margin_b.mean())
                _atomic_write(
                    out_path,
                    {
                        **meta,
                        "side": "trained",
                        "cell": cell_label,
                        "arm": arm,
                        "seed": seed,
                        "training_persona": spec["persona"],
                        "e_eval": e_eval,
                        "marker_persona": marker_persona,
                        "marker_id": sl["marker_id"],
                        "gauge_assert": {
                            "target_modules": sorted(gauge_cfg[cell_label]["target_modules"]),
                            "modules_to_save": gauge_cfg[cell_label].get("modules_to_save"),
                            "ok": True,
                        },
                        "trained": stats,
                        "base": b,
                        "delta_mean": {
                            "logp": d_logp,
                            "z_marker": d_zm,
                            "eos_margin": d_margin,
                        },
                        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
                    },
                )
                logger.info(
                    "cell=%s e_eval=%s marker=%s Δlogp=%+.3f Δz_marker=%+.3f "
                    "Δ(z_m-z_eos)=%+.3f in %.1fs",
                    cell_label,
                    e_eval,
                    marker_persona,
                    d_logp,
                    d_zm,
                    d_margin,
                    time.time() - t0,
                )
        finally:
            # Detach so the next cell's from_pretrained sees a clean base.
            # NOTE: PeftModel has no own `unload`; the call delegates via
            # PeftModel.__getattr__ -> LoraModel.unload() and returns the
            # clean base model. Verified on peft 0.18.1 with a tiny Qwen2.
            peft_model.unload()

    logger.info("Four-float logit capture done -> %s", per_cell_dir)


if __name__ == "__main__":
    main()
