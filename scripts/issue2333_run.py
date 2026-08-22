#!/usr/bin/env python
"""Issue #2333 pod driver — snowball test (plan §4, tasks/running/2333/plans/plan.md).

Decomposes the banked context-end (ce) all-layer replace-patch effect into
first-k ANSWER-position interventions: state patches at decode steps 1..k
(``patch`` arms, hidden-state replace via AnswerPositionEditHook) vs prefill
of the first-k donor TOKENS (``prefill`` arms, token-id concatenation), donor
schemes ``med`` (A-under-ce-patch greedy opening) and ``bstart`` (B's own
greedy opening), each with a scheme-matched norm-matched shuffled-donor null.

Phases (plan §4.4; one pod per model, ``--model-tag q25|q35``):
  envcheck   q35 B-1 CPU gate: transformers/AutoConfig/template probe (rc=24).
  bank       stage pins + fresh v_ce/v_pe capture + minimal-pair check (all
             contexts) + capture-parity gate (q25; rc=25) + donor maps.
  donors     greedy 8-token openings per pair x scheme (+ capture of answer
             states 1..3, all layers) + prefill/decode injection gates (rc=21).
  grid       144 blocks x K=5 via the #2162 claim queue (--smoke: 1 S1 + 1 S2
             pair through ALL 12 arms x 2 variants at K=1, same queue/runner).
  anchors    q35 only: 195 contexts x 10 unhooked draws + V_a.
  ce_control q35 only: 195 pairs x {steered, shuffled} ce replace x 5 draws.
  upload     bulk HF upload + results sentinel.

REUSES scripts/issue2162_run.py (imported as R): claim queue, done-file resume
predicates, injection gate (seamed), pad/eot helpers, upload_dir_hf, pilot
constants — never re-implements them (plan §10 fitness map).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps + API keys)

import torch  # noqa: E402

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import issue2162_run as R  # noqa: E402  (reuse surface, plan §10)

from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)
from explore_persona_space.experiments.issue2094 import bank as BANK94  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK2162  # noqa: E402
from explore_persona_space.experiments.issue2333 import constants as C  # noqa: E402
from explore_persona_space.experiments.issue2333.decode_hooks import (  # noqa: E402
    generate_batch_ids,
    joint_answer_hooks,
    resolve_decoder_blocks_2333,
)

logger = logging.getLogger("issue2333.run")

# Route the reused upload seam at THIS issue's write-repo override (parent
# reads EPM_2162_DATA_WRITE_REPO; #2333 gets its own env knob, same fallback).
R.HF_DATA_WRITE_REPO = os.environ.get(C.DATA_WRITE_REPO_ENV, C.DATA_REPO)

DEFAULT_OUT_ROOT = Path("/workspace/issue2333_out")
DEFAULT_LOG_DIR = Path("/workspace/logs")

RC_OK = 0
RC_INJECTION_GATE = 21  # parent convention (R.RC_INJECTION_GATE)
RC_PILOT_GATE = 22
RC_ENVCHECK_GATE = 24
RC_PARITY_GATE = 25
RC_MINPAIR_GATE = 26

MINPAIR_VIOLATION_MAX_FRAC = 0.10  # above => systemic render break, halt
CAP_HIT_REGEN_FRAC = 0.02  # plan §6 pre-registered regen trigger
PLANNED_GRID_WALL_H = {"q25": 1.4, "q35": 2.1}  # plan §9 A3/B4 rows

EXPECTED_N_CONTEXTS = 195  # plan §4.1: union of S1 + S2 referenced contexts
EXPECTED_N_PAIRS = 195  # 180 S1 + 15 S2
EXPECTED_N_BLOCKS = 144  # (5 S1 cells + 1 S2 cell) x 12 arms x 2 variants


# ── config ────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Duck-types every field the reused R.* helpers read (R.RunConfig shape)."""

    phase: str
    model_tag: str
    out_root: Path
    log_dir: Path
    model_id: str
    tiny: bool
    n_layers: int
    hidden: int
    device: str
    gen_batch: int
    capture_batch: int
    max_new_tokens: int
    anchor_draws: int
    grid_draws: int
    seed_base: int
    smoke: bool
    pilot: bool
    force: bool
    worker_index: int
    num_workers: int
    upload_mode: str
    upload_every: int
    planned_wall_h: float
    only_blocks: tuple[str, ...]

    @property
    def rollouts_dir(self) -> Path:
        return self.out_root / "rollouts"

    @property
    def grid_namespace(self) -> str:
        """Shard + done-file namespace — SMOKE artifacts never share paths with
        production (r1 Major 2: shared ``blocks/`` shard paths let production
        uploads sweep stale K=1 smoke shards under the production prefix)."""
        return "smoke_blocks" if self.smoke else "blocks"

    @property
    def grid_blocks_dir(self) -> Path:
        return self.rollouts_dir / self.grid_namespace

    @property
    def ce_namespace(self) -> str:
        return "ce_control_smoke" if self.smoke else "ce_control"

    @property
    def ce_blocks_dir(self) -> Path:
        return self.rollouts_dir / self.ce_namespace

    @property
    def va_dir(self) -> Path:
        return self.out_root / ("va_store_smoke" if self.smoke else "va_store")

    @property
    def anchors_dir(self) -> Path:
        return self.out_root / ("anchors_smoke" if self.smoke else "anchors")

    @property
    def bank_dir(self) -> Path:
        return self.out_root / "vc_bank"

    @property
    def donors_dir(self) -> Path:
        return self.out_root / "donors"

    @property
    def gates_dir(self) -> Path:
        return self.out_root / "gates"

    @property
    def manifest_dir(self) -> Path:
        return self.out_root / "manifests"

    @property
    def layers(self) -> list[int]:
        return list(range(self.n_layers))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2333 pod driver (envcheck/bank/donors/grid/anchors/ce_control/upload)."
    )
    ap.add_argument(
        "--phase",
        choices=(
            "envcheck",
            "fitness2329",
            "bank",
            "donors",
            "grid",
            "anchors",
            "ce_control",
            "upload",
        ),
        help="pipeline phase (required unless --import-check)",
    )
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--model-tag", choices=tuple(C.MODELS), default="q25")
    ap.add_argument(
        "--out-root", type=Path, default=None, help="default: /workspace/issue2333_out/<tag>"
    )
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gen-batch", type=int, default=C.GEN_BATCH)
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--anchor-draws", type=int, default=C.ANCHOR_DRAWS)
    ap.add_argument("--grid-draws", type=int, default=C.GRID_DRAWS)
    ap.add_argument("--seed-base", type=int, default=C.SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="1 S1 + 1 S2 pair, all arms, K=1")
    ap.add_argument("--pilot", action="store_true", help="grid: timing pilot only")
    ap.add_argument("--force", action="store_true", help="re-run a completed phase")
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    ap.add_argument("--upload", choices=("hf", "none"), default="hf")
    ap.add_argument("--upload-every", type=int, default=25)
    ap.add_argument("--planned-wall-h", type=float, default=None)
    ap.add_argument(
        "--only-blocks",
        default="",
        help="comma-separated block-key substrings (cap-hit regen / targeted re-runs)",
    )
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> RunConfig:
    spec = C.MODELS[args.model_tag]
    if args.device:
        device = args.device
    elif args.tiny:
        device = "cpu"
    else:
        device = "cuda:0"
    out_root = args.out_root if args.out_root else DEFAULT_OUT_ROOT / args.model_tag
    planned = args.planned_wall_h
    if planned is None:
        planned = PLANNED_GRID_WALL_H[args.model_tag]
    return RunConfig(
        phase=args.phase,
        model_tag=args.model_tag,
        out_root=out_root,
        log_dir=args.log_dir,
        model_id=spec["model_id"],
        tiny=args.tiny,
        n_layers=args.tiny_layers if args.tiny else spec["n_layers"],
        hidden=args.tiny_hidden if args.tiny else spec["hidden"],
        device=device,
        gen_batch=args.gen_batch,
        capture_batch=args.capture_batch,
        max_new_tokens=args.max_new_tokens,
        anchor_draws=args.anchor_draws,
        grid_draws=1 if args.smoke else args.grid_draws,
        seed_base=args.seed_base,
        smoke=args.smoke,
        pilot=args.pilot,
        force=args.force,
        worker_index=args.worker_index,
        num_workers=args.num_workers,
        upload_mode=args.upload,
        upload_every=args.upload_every,
        planned_wall_h=planned,
        only_blocks=tuple(s for s in args.only_blocks.split(",") if s),
    )


def hf_prefix(cfg: RunConfig) -> str:
    base = f"{C.HF_PREFIX}/{cfg.model_tag}"
    return f"{base}_smoke" if cfg.smoke else base


# ── pair universe / contexts / rendering ─────────────────────────────


def build_pair_universe() -> tuple[list, list]:
    """(s1_pairs, s2_pairs): #2162 survivor-cell pairs + #2094 matched-query."""
    s1 = [p for p in BANK2162.build_pairs() if p.cell in C.S1_CELLS]
    assert len(s1) == len(C.S1_CELLS) * C.S1_PAIRS_PER_CELL, len(s1)
    s2 = [p for p in BANK94.build_pairs() if p.setting == "matched_query"]
    assert len(s2) == 15, len(s2)
    return s1, s2


def pair_set_of(pair) -> str:
    return "s1" if hasattr(pair, "cell") else "s2"


def cell_of(pair) -> str:
    return pair.cell if hasattr(pair, "cell") else C.S2_CELL


def build_context_universe(s1_pairs: list, s2_pairs: list) -> dict[str, dict]:
    """All referenced contexts, tagged with ``__set`` for render dispatch."""
    ref_1 = {cid for p in s1_pairs for cid in (p.a, p.b)}
    ref_2 = {cid for p in s2_pairs for cid in (p.a, p.b)}
    assert not (ref_1 & ref_2), "S1/S2 context-id collision"
    ctx1 = BANK2162.build_contexts()
    ctx2 = BANK94.build_contexts()
    contexts: dict[str, dict] = {}
    for cid in sorted(ref_1):
        c = dict(ctx1[cid])
        c["__set"] = "s1"
        contexts[cid] = c
    for cid in sorted(ref_2):
        c = dict(ctx2[cid])
        c["__set"] = "s2"
        contexts[cid] = c
    assert len(contexts) == EXPECTED_N_CONTEXTS, (
        f"referenced context union = {len(contexts)}, plan §4.1 declares "
        f"{EXPECTED_N_CONTEXTS} — pair universe drifted, refusing"
    )
    return contexts


def make_ids_fn(model_tag: str):
    """Token-id render for BOTH banks; q35 pins ``enable_thinking=False``.

    q25 path is byte-equivalent to the parents' renders (same messages fn,
    same role-header swap, no extra template kwargs).
    """
    template_kwargs = {"enable_thinking": False} if model_tag == "q35" else {}

    def ids_fn(tok, context: dict) -> list[int]:
        rendered = tok.apply_chat_template(
            BANK94.context_messages_2094(context),
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        header = context.get("role_header")
        if header and header != "assistant":
            anchor = f"{BANK2162.IM_START}assistant"
            idx = rendered.rfind(anchor)
            assert idx >= 0, "generation prompt header not found in render"
            assert rendered[idx:].count(anchor) == 1
            rendered = (
                rendered[:idx] + f"{BANK2162.IM_START}{header}" + rendered[idx + len(anchor) :]
            )
        if model_tag == "q35":
            # Qwen3.5's thinking-OFF convention (measured under the resolved
            # transformers==5.15.0: `enable_thinking=False` renders a CLOSED
            # EMPTY `<think>\n\n</think>` block in the generation prompt,
            # while thinking-ON leaves a dangling open `<think>`). Assert
            # closed-and-empty, not absence (r2 smoke caught the absence
            # assert failing on the correct thinking-off render).
            open_i = rendered.rfind("<think>")
            if open_i >= 0:
                close_i = rendered.rfind("</think>")
                assert close_i > open_i, "OPEN thinking block in q35 render (thinking mode on)"
                assert not rendered[open_i + len("<think>") : close_i].strip(), (
                    "non-empty thinking block leaked into q35 render"
                )
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        assert len(ids) >= 4, (len(ids), context.get("id"))
        return ids

    return ids_fn


def _common_affix(a: list[int], b: list[int]) -> tuple[int, int]:
    """(common token prefix, common token suffix) lengths, non-overlapping.

    Vendored verbatim from origin/issue-2329:
    src/explore_persona_space/experiments/issue2329/bank2329.py::_common_affix
    (diagnostic fields for the minimal-pair violation report).
    """
    m = min(len(a), len(b))
    n = 0
    while n < m and a[n] == b[n]:
        n += 1
    s = 0
    while s < m - n and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    return n, s


def final_turn_boundary(ids: list[int], im_start_id: int) -> int:
    """Index of the SECOND-TO-LAST ``<|im_start|>`` — the final-turn boundary.

    ``ids[boundary:]`` is the final user turn + generation prompt. The
    occurrence assert is >= 2, NOT an exact count (relaxed from #2329's
    ``prefix_end_index_2329``): q35 thinking-off bare renders insert no
    default system turn (2 occurrences), q25 renders carry the default
    "You are Qwen..." system turn (3), and demo-history cells carry more.
    """
    occ = [i for i, t in enumerate(ids) if t == im_start_id]
    assert len(occ) >= 2, f"expected >=2 <|im_start|> token occurrences, got {len(occ)}"
    return occ[-2]


def prefix_end_index_2333(tok, ids: list[int]) -> int:
    """Tokenizer-agnostic prefix/query boundary (r5 crash fix, #2329 semantics).

    >= 3 ``<|im_start|>`` occurrences (every q25 render — the template inserts
    a default system turn — and q35 persona/history renders): the parent
    ``prefix_end_index_multi`` VERBATIM, so leg A's banked values are
    byte-equal to the parent helper. Exactly 2 occurrences (q35 thinking-off
    BARE render — Qwen3.5 inserts NO default system turn, so the final user
    turn opens the render): boundary 0 = EMPTY varied prefix; callers flag the
    record ``no_prefix`` and capture NO pe-slot state (there is no "last
    prefix token" — #2329 ``prefix_end_index_2329`` / pe-exclusion shape).
    Value-coherent with ``final_turn_boundary`` (occ[-2]) on every valid input.
    """
    im_start_id = tok.convert_tokens_to_ids(BANK2162.IM_START)
    assert isinstance(im_start_id, int) and im_start_id >= 0, im_start_id
    occ = [i for i, t in enumerate(ids) if t == im_start_id]
    if len(occ) >= 3:
        return BANK94.prefix_end_index_multi(tok, ids)
    assert len(occ) == 2, (
        f"expected 2 or >=3 {BANK2162.IM_START} occurrences, got {len(occ)} at {occ}"
    )
    assert occ[0] == 0, occ  # bare render opens with the final user turn
    return 0


def minimal_pair_check(ids_a: list[int], ids_b: list[int], im_start_id: int) -> tuple[str, ...]:
    """Span-locus minimal-pair verdict (plan A16/B0): empty tuple iff intact.

    Vendored/adapted from origin/issue-2329:
    src/explore_persona_space/experiments/issue2329/bank2329.py::_pair_verdict
    (prefix-side locus branch — every pair in this run's universe, S1 survivor
    cells + S2 matched-query, varies the PREFIX side only): (a) the final user
    turn + generation prompt — token ids from the second-to-last
    ``<|im_start|>`` — must be token-IDENTICAL across A/B
    (``final-turn-tokens-differ`` = genuine tokenizer break), and (b) the
    varied prefix must actually differ (``varied-prefix-identical`` = bank
    defect — impossible for distinct context strings).

    r3 fix: the r2 predicate (<= 1 contiguous ``SequenceMatcher`` diff region
    over the WHOLE id sequence) was structurally wrong — conflict cells swap
    the instruction slot AND the demo-history format (~12 interleaved diff
    regions BY DESIGN), mq persona<->conv pairs compose different templates
    (>= 7 regions), and mq bare<->persona under q25 splits on the default
    system block. It failed 77/195 (q25) and 82/195 (q35) with an EMPTY
    q35-only violation set: checker mis-specification, not a tokenizer break.
    The span-locus form passes 0/195 under BOTH tokenizers (r3 verification)
    and matches #2329's realized gate (1404/1404 intact on the superset bank
    under Qwen3.5).
    """
    reasons: list[str] = []
    pe_a = final_turn_boundary(ids_a, im_start_id)
    pe_b = final_turn_boundary(ids_b, im_start_id)
    if ids_a[pe_a:] != ids_b[pe_b:]:
        reasons.append("final-turn-tokens-differ")
    if ids_a[:pe_a] == ids_b[:pe_b]:
        reasons.append("varied-prefix-identical")
    return tuple(reasons)


# ── model loading (q25 / q35) ─────────────────────────────────────────


def load_model_and_tokenizer(cfg: RunConfig):
    """q25: R's loader verbatim. q35: multimodal wrapper fallback + text-config
    asserts (tiny smoke always builds the q25-arch tiny model — the q35 branch
    is covered by the envcheck gate + unit tests, recorded as a smoke blind spot)."""
    if cfg.model_tag == "q25" or cfg.tiny:
        base = C.MODELS["q25"]["model_id"] if cfg.tiny else cfg.model_id
        proxy = _R_cfg_proxy(cfg, model_id=base)
        return R.load_model_and_tokenizer(proxy)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    assert torch.cuda.is_available(), "the full grid requires CUDA (use --tiny for CPU smoke)"
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16)
    except (ValueError, KeyError):
        # Qwen3.5 multimodal wrapper (Qwen3_5ForConditionalGeneration) is not
        # an AutoModelForCausalLM member — load via the image-text-to-text map.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, dtype=torch.bfloat16)
    model = model.to(cfg.device)
    text_cfg = getattr(model.config, "text_config", model.config)
    assert text_cfg.hidden_size == cfg.hidden, (text_cfg.hidden_size, cfg.hidden)
    assert text_cfg.num_hidden_layers == cfg.n_layers, (text_cfg.num_hidden_layers, cfg.n_layers)
    model.eval()
    blocks, depth = resolve_decoder_blocks_2333(model)
    assert len(blocks) == cfg.n_layers, (len(blocks), cfg.n_layers)
    logger.info(
        "[model] %s resolved %d decoder blocks at depth %d", cfg.model_id, len(blocks), depth
    )
    return model, tok


def _R_cfg_proxy(cfg: RunConfig, model_id: str | None = None) -> R.RunConfig:
    """A parent-shaped RunConfig for R.* helpers that take R.RunConfig."""
    return R.RunConfig(
        phase=cfg.phase or "bank",
        out_root=cfg.out_root,
        log_dir=cfg.log_dir,
        model_id=model_id or cfg.model_id,
        tiny=cfg.tiny,
        n_layers=cfg.n_layers,
        hidden=cfg.hidden,
        device=cfg.device,
        gen_batch=cfg.gen_batch,
        capture_batch=cfg.capture_batch,
        max_new_tokens=cfg.max_new_tokens,
        anchor_draws=cfg.anchor_draws,
        grid_draws=cfg.grid_draws,
        seed_base=cfg.seed_base,
        smoke=cfg.smoke,
        pilot=cfg.pilot,
        force=cfg.force,
        force_past_halt_gates=False,
        worker_index=cfg.worker_index,
        num_workers=cfg.num_workers,
        upload_mode=cfg.upload_mode,
        upload_every=cfg.upload_every,
        planned_wall_h=cfg.planned_wall_h,
        gpu_hours_budgeted=0.0,
        pools_path=None,
    )


# ── regime / resume ───────────────────────────────────────────────────


def regime_fingerprint(cfg: RunConfig) -> str:
    """Every output-affecting knob (resume key; #722 r3).

    r1 blocker fix (incomplete-regime-fingerprint): now ALSO keys on
    ``anchor_draws`` / ``ce_control_draws`` (anchor + ce outputs),
    ``gen_batch`` / ``capture_batch`` (batched sampling + bf16 capture numerics
    depend on batch composition), ``donor_k_max``, the RESOLVED transformers
    version (the q35 env identity — plan §4.4 B-1), and the chat-template
    identity (``enable_thinking`` pin). Field-sensitivity is unit-tested.
    DELIBERATELY EXCLUDED: ``worker_index`` / ``num_workers`` — block outputs
    are worker-independent under the claim queue; the anchors phase (the one
    place sharding IS output-partitioning) keys its OWN done files + shard
    filenames on ``w{i}of{N}`` instead (a global num_workers key would wrongly
    invalidate every completed block on a width change).
    """
    import hashlib

    import transformers

    payload = json.dumps(
        {
            "issue": 2333,
            "model_tag": cfg.model_tag,
            "model_id": cfg.model_id,
            "tiny": cfg.tiny,
            "n_layers": cfg.n_layers,
            "hidden": cfg.hidden,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": C.GRID_TEMPERATURE,
            "grid_draws": cfg.grid_draws,
            "anchor_draws": cfg.anchor_draws,
            "ce_control_draws": C.CE_CONTROL_DRAWS,
            "gen_batch": cfg.gen_batch,
            "capture_batch": cfg.capture_batch,
            "seed_base": cfg.seed_base,
            "smoke": cfg.smoke,
            "s1_cells": list(C.S1_CELLS),
            "s2_derangement_seed": C.S2_DERANGEMENT_SEED,
            "donor_max_new_tokens": C.DONOR_MAX_NEW_TOKENS,
            "donor_k_max": C.DONOR_K_MAX,
            "arm_slugs": list(C.ARM_SLUGS),
            "transformers": str(transformers.__version__),
            "template_kwargs": {"enable_thinking": False} if cfg.model_tag == "q35" else {},
            "pins": {"p2162": C.PIN_2162, "p2094": C.PIN_2094, "fu1": C.PIN_FU1},
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _phase_done(cfg: RunConfig, phase: str, regime_fp: str) -> bool:
    path = cfg.manifest_dir / f"{phase}_done.json"
    if not path.exists():
        return False
    rec = json.loads(path.read_text())
    if rec.get("regime_fp") != regime_fp:
        raise RuntimeError(
            f"{phase} done-file regime_fp={rec.get('regime_fp')!r} != {regime_fp!r} — "
            "refusing cross-regime resume (fresh --out-root or quarantine)"
        )
    return True


def _write_phase_done(cfg: RunConfig, phase: str, regime_fp: str, extra: dict) -> None:
    R._write_json_atomic(
        cfg.manifest_dir / f"{phase}_done.json",
        {
            "phase": phase,
            "regime_fp": regime_fp,
            "num_workers": cfg.num_workers,
            **extra,
            **_repro(cfg),
        },
    )


def _repro(cfg: RunConfig) -> dict:
    import transformers

    return {
        "repro": {
            "git_commit": R._git_sha(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "model_id": cfg.model_id,
            "model_tag": cfg.model_tag,
            "tiny": cfg.tiny,
            "smoke": cfg.smoke,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    }


# ── HF staging (pinned reads) ─────────────────────────────────────────


def _stage_pinned(repo_rel: str, revision: str, dest: Path) -> Path:
    """Pinned single-file stage from the data repo (retry-wrapped)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    dest.parent.mkdir(parents=True, exist_ok=True)
    got = retry_transient(
        lambda: hf_hub_download(
            repo_id=C.DATA_REPO,
            repo_type="dataset",
            filename=repo_rel,
            revision=revision,
            local_dir=dest.parent / f".hfstage_{dest.name}",
        ),
        what=f"stage-pinned {repo_rel}",
    )
    os.replace(got, dest)
    return dest


# ── phase: envcheck (q35 B-1 gate) ────────────────────────────────────


def phase_envcheck(cfg: RunConfig, regime_fp: str) -> int:
    """q35 CPU gate: config + tokenizer + thinking-off template render resolve
    under the POD-RESOLVED transformers (plan §4.4 B-1; repo-pinned 4.57.6
    fails AutoConfig for Qwen3.5 — this gate re-probes post-resolution)."""
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    report: dict = {
        "transformers": str(transformers.__version__),
        "transformers_pin_expected": C.Q35_TRANSFORMERS_PIN if cfg.model_tag == "q35" else None,
        "model_id": cfg.model_id,
    }
    ok = True
    try:
        if cfg.model_tag == "q35":
            # Plan §4.4 B-1 version RESOLUTION: the pod env must run #2329's
            # realized pin (gate 0b, origin/issue-2329) — the repo pin 4.57.6
            # FAILS AutoConfig for qwen3_5 (A8). Run this phase under the
            # resolved env (e.g. `uv run --with transformers==5.15.0 ...`).
            assert str(transformers.__version__) == C.Q35_TRANSFORMERS_PIN, (
                f"q35 envcheck must run under transformers=={C.Q35_TRANSFORMERS_PIN} "
                f"(#2329 realized pin); got {transformers.__version__} — launch via "
                f"`uv run --with 'transformers=={C.Q35_TRANSFORMERS_PIN}'` or the pod "
                "venv pin (issue2329_dispatch.sh gate0b shape)"
            )
        mcfg = AutoConfig.from_pretrained(cfg.model_id)
        text_cfg = getattr(mcfg, "text_config", mcfg)
        report["n_layers"] = int(text_cfg.num_hidden_layers)
        report["hidden"] = int(text_cfg.hidden_size)
        assert report["n_layers"] == cfg.n_layers, (report["n_layers"], cfg.n_layers)
        assert report["hidden"] == cfg.hidden, (report["hidden"], cfg.hidden)
        tok = AutoTokenizer.from_pretrained(cfg.model_id)
        ids_fn = make_ids_fn(cfg.model_tag)
        probe_ctx = {"id": "envcheck", "system": None, "history": [], "user": "What is 2+2?"}
        ids = ids_fn(tok, probe_ctx)
        report["probe_ids_len"] = len(ids)
        # Meta-device STRUCTURE probe (plan §4.4 B-1 / A9): the decoder-block
        # list must resolve through the (multimodal) wrapper WITHOUT weights —
        # A9 expects the naive .layers path to fail on Qwen3.5 and the
        # .language_model nesting (extraction._resolve_decoder_blocks) to hold.
        from transformers import AutoModelForCausalLM

        try:
            with torch.device("meta"):
                meta_model = AutoModelForCausalLM.from_config(mcfg)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText

            with torch.device("meta"):
                meta_model = AutoModelForImageTextToText.from_config(mcfg)
        blocks, depth = resolve_decoder_blocks_2333(meta_model)
        assert len(blocks) == cfg.n_layers, (len(blocks), cfg.n_layers)
        report["n_blocks_resolved"] = len(blocks)
        report["blocks_depth"] = depth
        del meta_model
    except Exception as e:  # designed halt — report + distinct rc, never rc=1
        ok = False
        report["error"] = f"{type(e).__name__}: {e}"
        logger.exception("[envcheck] FAILED")
    report["verdict"] = "PASS" if ok else "FAIL"
    R._write_json_atomic(cfg.gates_dir / "envcheck_report.json", {**report, **_repro(cfg)})
    if not ok:
        return RC_ENVCHECK_GATE
    _write_phase_done(cfg, "envcheck", regime_fp, {"verdict": "PASS"})
    logger.info("[envcheck] PASS (transformers %s)", report["transformers"])
    return RC_OK


def _require_q35_env(cfg: RunConfig) -> None:
    """q35 env-identity guard (r1 Major fix): every model-loading q35 phase
    runs under the EXACT environment the envcheck gate PASSed — same
    transformers version, envcheck verdict PASS. Enforces plan §4.4 B-1's
    "run envcheck under the env the pod will install" mechanically."""
    import transformers

    path = cfg.gates_dir / "envcheck_report.json"
    if not path.is_file():
        raise RuntimeError(
            f"q35 phase {cfg.phase!r} requires a PASSING envcheck first — "
            f"{path} missing (run --phase envcheck under the resolved env)"
        )
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("verdict") != "PASS":
        raise RuntimeError(f"q35 envcheck verdict is {rec.get('verdict')!r} — fix the env first")
    if str(transformers.__version__) != rec.get("transformers"):
        raise RuntimeError(
            f"q35 env drifted since envcheck: transformers {transformers.__version__} "
            f"!= envcheck's {rec.get('transformers')} — re-run envcheck under THIS env"
        )


# ── phase: fitness2329 (B-1 reuse-or-selfgen decision, VM/CPU-side) ───


def phase_fitness2329(cfg: RunConfig, regime_fp: str) -> int:
    """Plan §4.4 B-1 #2329 artifact FITNESS probe -> reuse-or-selfgen decision.

    Probes each #2329 artifact class (``C.FITNESS_2329_CLASSES``) at ONE
    resolved data-repo revision. Decision policy (plan §4.4 B-1): reuse ONLY
    when EVERY check passes; ANY failed/unverifiable check => the declared
    self-generation path (B0/B1), never partial reuse. A class that is
    PRESENT still fails today: its realized keys/schema can only be verified
    through the actual loader against an OBSERVED artifact (schema-from-
    artifact rule), and no #2329 rows have landed — so a present class is
    recorded ``present-unverified`` and treated as FAILED until a probe-able
    artifact exists. The decision JSON is the launch-marker record.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi()
    info = retry_transient(
        lambda: api.repo_info(C.DATA_REPO, repo_type="dataset"),
        what="fitness2329 repo_info",
    )
    resolved_rev = str(info.sha)
    checks: dict[str, dict] = {}
    for prefix in C.FITNESS_2329_CLASSES:
        try:
            entries = retry_transient(
                lambda p=prefix: list(
                    # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient at this call site
                    api.list_repo_tree(
                        C.DATA_REPO,
                        path_in_repo=p,
                        repo_type="dataset",
                        revision=resolved_rev,
                        recursive=False,
                    )
                ),
                what=f"fitness2329 probe {prefix}",
            )
        except Exception as e:  # EntryNotFound and kin — absent class
            checks[prefix] = {"result": "absent", "detail": f"{type(e).__name__}: {e}"[:200]}
            continue
        if not entries:
            checks[prefix] = {"result": "absent", "detail": "prefix resolves empty"}
            continue
        checks[prefix] = {
            "result": "present-unverified",
            "n_entries": len(entries),
            "detail": (
                "realized-schema / pair-coverage / provenance-coherence checks require an "
                "OBSERVED artifact schema (schema-from-artifact); treated as FAILED per "
                "plan §4.4 B-1 (any failed check => selfgen) — re-probe after #2329 lands"
            ),
        }
    all_pass = all(c["result"] == "pass" for c in checks.values()) and bool(checks)
    decision = "reuse" if all_pass else "selfgen"
    report = {
        "decision": decision,
        "policy": "reuse only if EVERY check passes (plan §4.4 B-1); else selfgen",
        "resolved_data_repo_revision": resolved_rev,
        "checks": checks,
        "transformers_pin_resolved": C.Q35_TRANSFORMERS_PIN,
        "transformers_pin_source": (
            "origin/issue-2329 scripts/issue2329_run.py gate 0b (asserts ==5.15.0) + "
            "issue2329_dispatch.sh TRANSFORMERS_PIN"
        ),
        **_repro(cfg),
    }
    R._write_json_atomic(cfg.gates_dir / "fitness2329_decision.json", report)
    _write_phase_done(cfg, "fitness2329", regime_fp, {"decision": decision})
    logger.info(
        "[fitness2329] decision=%s @ %s (%d classes probed)",
        decision,
        resolved_rev[:12],
        len(checks),
    )
    return RC_OK


# ── phase: bank ───────────────────────────────────────────────────────


class _ChunkStore:
    """Per-unit intra-phase checkpointing (code-style.md checkpoint rule, T2:
    bank = 195 contexts, donors = 390 rollouts — both > ~50 units; r1 blocker
    bank-donor-restartability). Each chunk lands atomically the moment it
    completes, keyed on the regime fingerprint; resume loads completed chunks
    and SKIPS them; a cross-regime chunk is a HARD refusal (#722 r3)."""

    def __init__(self, root: Path, regime_fp: str) -> None:
        self.root = root
        self.regime_fp = regime_fp
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"chunk_{key}.pt"

    def load(self, key: str):
        path = self._path(key)
        if not path.exists():
            return None
        rec = torch.load(path, map_location="cpu", weights_only=False)
        if rec.get("regime_fp") != self.regime_fp:
            raise RuntimeError(
                f"chunk {path} carries regime_fp={rec.get('regime_fp')!r} != "
                f"{self.regime_fp!r} — refusing cross-regime chunk resume "
                "(fresh --out-root or quarantine)"
            )
        return rec["payload"]

    def save(self, key: str, payload) -> None:
        path = self._path(key)
        tmp = path.parent / f"{path.name}.tmp.pt"
        torch.save({"regime_fp": self.regime_fp, "payload": payload}, tmp)
        os.replace(tmp, path)

    def clear(self) -> None:
        for p in self.root.glob("chunk_*.pt"):
            p.unlink()


def _require_single_worker(cfg: RunConfig, phase: str) -> None:
    """bank/donors are SINGLE-WORKER phases (named deviation from plan §9's
    8-way rows, recorded in the r2 marker): both write ONE shared store
    (vc_bank.pt / donors_<scheme>.pt) and their realized wall is minutes
    (195 batched forwards / 390 greedy 8-token rollouts), so an 8-way fan-out
    would duplicate the full workload and race the shared tmp files (r1
    Critical bank-donor-compute-shape). Fail LOUD instead of racing."""
    if cfg.num_workers != 1 or cfg.worker_index != 0:
        raise RuntimeError(
            f"phase {phase!r} is single-worker by design (shared-output store; "
            f"minutes of wall) — launch with --num-workers 1, got "
            f"worker_index={cfg.worker_index} num_workers={cfg.num_workers}"
        )


@torch.no_grad()
def capture_bank(
    cfg: RunConfig, model, tok, contexts: dict[str, dict], ids_fn, regime_fp: str
) -> dict:
    """All-layer v_ce + v_pe per context (right-padded chunks; BPE-seam-safe
    ID-space positions — parent capture_bank recipe). Chunk-checkpointed +
    resumable (``_ChunkStore``).

    r5: prefix ends via ``prefix_end_index_2333`` — q35 thinking-off bare
    renders have prefix_end 0 (EMPTY varied prefix): those records carry
    ``no_prefix: True`` and NO ``v_pe`` key (a pe-slot read fail-louds with
    KeyError instead of silently reading position -1 = the LAST token). No
    2333 phase reads slot "pe"; q25 records are byte-equal to the parent.
    """
    ctx_ids = {cid: ids_fn(tok, c) for cid, c in contexts.items()}
    prefix_ends = {cid: prefix_end_index_2333(tok, ids) for cid, ids in ctx_ids.items()}
    no_prefix_ids = sorted(cid for cid, pe in prefix_ends.items() if pe == 0)
    logger.info("[bank] no-prefix contexts (bare renders, pe==0): %d", len(no_prefix_ids))
    layers = cfg.layers
    pad_id = tok.pad_token_id
    records: dict[str, dict] = {}
    order = list(contexts)
    store = _ChunkStore(cfg.bank_dir / "bank_chunks", regime_fp)
    for start in range(0, len(order), cfg.capture_batch):
        chunk = order[start : start + cfg.capture_batch]
        key = f"{start:04d}"
        cached = store.load(key)
        if cached is not None:
            assert sorted(cached) == sorted(chunk), (key, sorted(cached), sorted(chunk))
            records.update(cached)
            logger.info("[bank] unit %d/%d contexts (resumed)", start + len(chunk), len(order))
            continue
        ids, mask = R._right_pad([ctx_ids[c] for c in chunk], pad_id, cfg.device)
        captured = extract_layer_activations(model, ids, layers, attention_mask=mask)
        chunk_records: dict[str, dict] = {}
        for j, cid in enumerate(chunk):
            ctx_len = len(ctx_ids[cid])
            pe = prefix_ends[cid]
            assert 0 <= pe < ctx_len, (cid, ctx_len, pe)
            v_ce = torch.stack([captured[layer][j, ctx_len - 1] for layer in layers])
            assert v_ce.shape == (len(layers), cfg.hidden), v_ce.shape
            rec = {
                "context_id": cid,
                "set": contexts[cid]["__set"],
                "ctx_len": ctx_len,
                "prefix_end": pe,
                "no_prefix": pe == 0,
                "v_ce": v_ce.float().cpu(),
            }
            if pe >= 1:
                v_pe = torch.stack([captured[layer][j, pe - 1] for layer in layers])
                assert v_pe.shape == (len(layers), cfg.hidden), v_pe.shape
                rec["v_pe"] = v_pe.float().cpu()
            # pe == 0: no pe-slot token exists — v_pe deliberately OMITTED
            # (#2329 pe-exclusion; a slot-"pe" read on this record KeyErrors).
            chunk_records[cid] = rec
        del captured
        store.save(key, chunk_records)
        records.update(chunk_records)
        logger.info(
            "[bank] unit %d/%d contexts", min(start + cfg.capture_batch, len(order)), len(order)
        )
    assert len(records) == len(contexts), (len(records), len(contexts))
    return {"layers": layers, "per_context": records}


def _banked_vce(rec: dict) -> torch.Tensor:
    """``(L, H)`` context-end state from a banked PARENT per-context record.

    TWO observed parent record schemas (probed from the REAL pinned stores,
    r4 crash fix — leg A q25 died ``KeyError: 'v_ce'`` on the S2 records):

    - #2162 vc_bank @ ``PIN_2162``: record keys {context_id, cell, value_id,
      carrier, ctx_len, prefix_end, v_ce, v_pe} — ``v_ce`` (L, H) verbatim.
    - #2094 vc_bank @ ``PIN_2094``: record keys {context_id, prefix,
      query_id, ctx_len, prefix_end, nq, q_span, v_pe} — NO ``v_ce``; the
      context-end state is ``q_span[-1]``: q_span is (nq, L, H) over
      positions [prefix_end, ctx_len), so its last row is position
      ctx_len-1 — the slot the parent's own ``ce`` read uses
      (issue2094_run.py::_slot_vectors / capture_bank).

    Fail LOUD on any other layout — never a silent default (#2061/#2091
    schema-from-artifact family).
    """
    if "v_ce" in rec:
        return rec["v_ce"].float()
    if "q_span" in rec:
        span = rec["q_span"]
        assert span.ndim == 3 and span.shape[0] == int(rec["nq"]), (span.shape, rec.get("nq"))
        return span[-1].float()
    raise RuntimeError(
        f"unrecognized banked per-context schema for {rec.get('context_id')!r}: "
        f"keys={sorted(rec)} — expected 'v_ce' (#2162 layout) or 'q_span' (#2094 layout)"
    )


def capture_parity_gate(cfg: RunConfig, bank: dict) -> dict:
    """q25 HALT gate: fresh v_ce vs the PINNED parent banks (two-bar, #779).

    S1 contexts against the #2162 vc bank @ PIN_2162; S2 contexts against the
    #2094 vc bank @ PIN_2094 (dual record schemas — ``_banked_vce``). Early
    layers per-layer >= 0.999; flattened all-layer >= 0.995.
    """
    staged: dict[str, dict] = {}
    for set_name, repo_rel, rev in (
        ("s1", C.R2162_VC_BANK, C.PIN_2162),
        ("s2", C.R2094_VC_BANK, C.PIN_2094),
    ):
        dest = cfg.bank_dir / f"parent_vc_bank_{set_name}.pt"
        if not dest.exists():
            _stage_pinned(repo_rel, rev, dest)
        store = torch.load(dest, map_location="cpu", weights_only=False)
        assert list(store["layers"]) == list(bank["layers"]), (
            set_name,
            store["layers"],
            bank["layers"],
        )
        staged[set_name] = store["per_context"]

    worst = {"early": 1.0, "flat": 1.0}
    n = 0
    details: list[dict] = []
    for cid, rec in bank["per_context"].items():
        parent = staged[rec["set"]].get(cid)
        if parent is None:
            raise RuntimeError(f"capture-parity: context {cid} absent from pinned parent bank")
        fresh = rec["v_ce"]
        banked = _banked_vce(parent)
        assert fresh.shape == banked.shape, (cid, fresh.shape, banked.shape)
        early = min(
            float(torch.nn.functional.cosine_similarity(fresh[layer], banked[layer], dim=0))
            for layer in C.PARITY_EARLY_LAYERS
        )
        flat = float(
            torch.nn.functional.cosine_similarity(fresh.flatten(), banked.flatten(), dim=0)
        )
        worst["early"] = min(worst["early"], early)
        worst["flat"] = min(worst["flat"], flat)
        n += 1
        if early < C.PARITY_EARLY_COS_MIN or flat < C.PARITY_FLAT_COS_MIN:
            details.append(
                {
                    "context_id": cid,
                    "early": early,
                    "flat": flat,
                    # render-drift diagnostics: byte-equivalent renders must
                    # agree on both position fields (fresh vs banked parent)
                    "ctx_len": (rec.get("ctx_len"), parent.get("ctx_len")),
                    "prefix_end": (rec.get("prefix_end"), parent.get("prefix_end")),
                }
            )
    verdict = "PASS" if not details else "FAIL"
    return {
        "verdict": verdict,
        "n_contexts": n,
        "worst_early_cos": worst["early"],
        "worst_flat_cos": worst["flat"],
        "bars": {"early": C.PARITY_EARLY_COS_MIN, "flat": C.PARITY_FLAT_COS_MIN},
        "failures": details,
    }


S2_DONOR_MAP_PROVENANCE = (
    "fallback fresh seeded derangement over the 15 matched-query pair ids, seed "
    f"{C.S2_DERANGEMENT_SEED} (constants.seeded_derangement) — the plan §4.2 NAMED "
    "FALLBACK. The §4.2 PRIMARY (recover the parent's realized derangement from "
    "eval_results/issue_2094/f_metrics/null_cells.jsonl donor_pair_id fields) is "
    "AMBIGUOUS: 5/15 mq pairs carry >1 distinct pair-type donor across the parent's "
    "null blocks (15/15 counting centroid donors; measured 2026-08-16, r2 re-probe). "
    "When minpair drops occurred, the derangement is over the SURVIVING matched-query "
    "ids only (same seed; dropped ids recorded in bank.json minpair_dropped_pair_ids)."
)


def _frozen_s1_shuffled_map(staged: Path) -> dict[str, str]:
    """Frozen S1 shuffled-donor map from the staged parent bank.json @ PIN_2162.

    Observed schema (real artifact probed r4): the JSON manifest's top-level
    key is ``donor_assignment`` (SINGULAR — the parent's vc_bank.pt store
    carries ``donor_assignments``, plural, which is where the pre-r4 plural
    read came from), a dict {shuffled, crosstype}; ``shuffled`` maps
    pair_id -> donor pair_id. Fail LOUD on drift, never a silent default.
    """
    manifest = json.loads(staged.read_text())
    da = manifest.get("donor_assignment")
    if not isinstance(da, dict) or "shuffled" not in da:
        raise RuntimeError(
            "parent bank.json schema drift: expected top-level 'donor_assignment' "
            f"with a 'shuffled' map; got top-level keys={sorted(manifest)}"
        )
    return da["shuffled"]


def build_donor_maps(
    s1_pairs: list, s2_pairs: list, dropped: set[str] | frozenset[str] = frozenset()
) -> dict[str, dict[str, str]]:
    """Shuffled-donor maps over the SURVIVOR pair set (post minpair drops).

    S1 = the parent's value-constrained same-cell assignment (deterministic
    regeneration over the FULL 1404-pair set, seed BANK2162.SEED — the
    bank.json realized map), restricted to the survivor cells AND the
    surviving (non-``dropped``) pairs. A surviving recipient whose frozen
    donor was dropped cannot get a valid donor under the frozen-map
    semantics — fail LOUD, never re-assign (r4 survivor-safety: maps built
    over ALL pairs let survivors reference dropped donors — a silent
    cascade-drop in grid via ``pair_dropped_for_arm`` and a ``KeyError`` in
    the q35 ce-control via ``payload_for_arm``'s ``pairs_by_id`` lookup).

    S2 = the plan §4.2 NAMED FALLBACK — fresh seeded derangement over the
    SURVIVING matched-query ids, seed ``C.S2_DERANGEMENT_SEED`` (23330) —
    because the primary (recovery from the parent's null_cells.jsonl) is
    ambiguous (``S2_DONOR_MAP_PROVENANCE``). Re-deranging survivors keeps
    the recorded-seed semantics and cannot orphan a survivor.
    """
    full = BANK2162.donor_assignment_2162(BANK2162.build_pairs())
    s1_ids = {p.pair_id for p in s1_pairs if p.pair_id not in dropped}
    s1_map = {pid: d for pid, d in full["shuffled"].items() if pid in s1_ids}
    assert set(s1_map) == s1_ids, "S1 donor map does not cover the survivor pairs"
    orphaned = {pid: d for pid, d in s1_map.items() if d not in s1_ids}
    if orphaned:
        raise RuntimeError(
            "S1 survivors reference minpair-dropped (or out-of-cell) frozen donors — the "
            "frozen value-constrained map cannot be preserved over this survivor set; "
            f"refusing (recipient -> unavailable donor): {sorted(orphaned.items())[:5]}"
        )
    s2_ids = sorted(p.pair_id for p in s2_pairs if p.pair_id not in dropped)
    s2_map = C.seeded_derangement(s2_ids, C.S2_DERANGEMENT_SEED)
    stray = [d for d in s2_map.values() if d not in set(s2_ids)]
    assert not stray, f"S2 donors leave the matched-query set: {stray[:5]}"
    assert all(pid != d for pid, d in s2_map.items()), "S2 derangement has a fixed point"
    return {"shuffled": {**s1_map, **s2_map}}


def phase_bank(cfg: RunConfig, regime_fp: str) -> int:
    _require_single_worker(cfg, "bank")
    if (
        not cfg.force
        and _phase_done(cfg, "bank", regime_fp)
        and (cfg.bank_dir / "vc_bank.pt").exists()
    ):
        logger.info("[bank] done — skip")
        return RC_OK
    s1_pairs, s2_pairs = build_pair_universe()
    contexts = build_context_universe(s1_pairs, s2_pairs)
    ids_fn = make_ids_fn(cfg.model_tag)
    model, tok = load_model_and_tokenizer(cfg)

    # Minimal-pair re-tokenization check at FULL grain (plan A16/B0):
    # span-locus verdict — final-turn token identity + varied-prefix-differs.
    ctx_ids = {cid: ids_fn(tok, c) for cid, c in contexts.items()}
    im_start_id = tok.convert_tokens_to_ids(BANK2162.IM_START)
    assert isinstance(im_start_id, int) and im_start_id >= 0, im_start_id
    violations: list[str] = []
    violation_detail: list[dict] = []
    for p in [*s1_pairs, *s2_pairs]:
        reasons = minimal_pair_check(ctx_ids[p.a], ctx_ids[p.b], im_start_id)
        if reasons:
            violations.append(p.pair_id)
            n_pre, n_suf = _common_affix(ctx_ids[p.a], ctx_ids[p.b])
            violation_detail.append(
                {
                    "pair_id": p.pair_id,
                    "reasons": list(reasons),
                    "len_a": len(ctx_ids[p.a]),
                    "len_b": len(ctx_ids[p.b]),
                    "common_prefix": n_pre,
                    "common_suffix": n_suf,
                }
            )
    frac = len(violations) / (len(s1_pairs) + len(s2_pairs))
    logger.info("[bank] minimal-pair check: %d violations (%.3f)", len(violations), frac)
    if frac > MINPAIR_VIOLATION_MAX_FRAC:
        R._write_json_atomic(
            cfg.gates_dir / "minpair_report.json",
            {
                "verdict": "FAIL",
                "violations": violations,
                "detail": violation_detail,
                "frac": frac,
                **_repro(cfg),
            },
        )
        return RC_MINPAIR_GATE

    bank = capture_bank(cfg, model, tok, contexts, ids_fn, regime_fp)

    parity: dict = {"verdict": "N/A", "reason": "q35 has no pinned parent bank"}
    if cfg.model_tag == "q25" and not cfg.tiny:
        parity = capture_parity_gate(cfg, bank)
        R._write_json_atomic(
            cfg.gates_dir / "capture_parity_report.json", {**parity, **_repro(cfg)}
        )
        if parity["verdict"] != "PASS":
            logger.error("[bank] capture-parity FAIL: %s", parity)
            return RC_PARITY_GATE

    dropped_ids = set(violations)
    donor_maps = build_donor_maps(s1_pairs, s2_pairs, dropped=dropped_ids)
    # Plan §4.2: the S1 map IS the parent bank.json's value-constrained
    # assignment — cross-check the deterministic regeneration against the
    # staged frozen copy @ PIN_2162 (fail loud on drift; #600 sha-pin family).
    if not cfg.tiny:
        staged = cfg.bank_dir / "parent_bank_2162.json"
        if not staged.exists():
            _stage_pinned(C.R2162_BANK_JSON, C.PIN_2162, staged)
        parent_map = _frozen_s1_shuffled_map(staged)
        s1_ids = {p.pair_id for p in s1_pairs if p.pair_id not in dropped_ids}
        mismatch = {
            pid: (donor_maps["shuffled"][pid], parent_map.get(pid))
            for pid in s1_ids
            if donor_maps["shuffled"][pid] != parent_map.get(pid)
        }
        assert not mismatch, (
            f"S1 donor map drifted from frozen bank.json: {list(mismatch.items())[:3]}"
        )
    manifest = {
        "pins": {"p2162": C.PIN_2162, "p2094": C.PIN_2094, "fu1": C.PIN_FU1},
        "model_tag": cfg.model_tag,
        "s1_cells": list(C.S1_CELLS),
        "s1_pair_ids": [p.pair_id for p in s1_pairs],
        "s2_pair_ids": [p.pair_id for p in s2_pairs],
        "context_ids": sorted(contexts),
        # q35 bare renders (pe == 0, EMPTY varied prefix): #2329 pe-exclusion
        # shape — these records carry no v_pe (r5). Empty on q25.
        "no_prefix_context_ids": sorted(
            cid for cid, rec in bank["per_context"].items() if rec.get("no_prefix")
        ),
        "minpair_violations": violations,
        "minpair_dropped_pair_ids": violations,
        "donor_maps": donor_maps,
        "s2_donor_map_provenance": S2_DONOR_MAP_PROVENANCE,
        "s2_derangement_seed": C.S2_DERANGEMENT_SEED,
        "capture_parity": parity,
        "regime_fp": regime_fp,
        **_repro(cfg),
    }
    R._write_json_atomic(cfg.bank_dir / "bank.json", manifest)
    cfg.bank_dir.mkdir(parents=True, exist_ok=True)
    tmp = cfg.bank_dir / "vc_bank.pt.tmp.pt"
    torch.save(bank, tmp)
    os.replace(tmp, cfg.bank_dir / "vc_bank.pt")
    _ChunkStore(cfg.bank_dir / "bank_chunks", regime_fp).clear()  # merged into vc_bank.pt
    _write_phase_done(
        cfg, "bank", regime_fp, {"n_contexts": len(contexts), "n_minpair_viol": len(violations)}
    )
    logger.info("[bank] done: %d contexts, parity=%s", len(contexts), parity["verdict"])
    return RC_OK


def _load_bank(cfg: RunConfig) -> tuple[dict, dict]:
    manifest = json.loads((cfg.bank_dir / "bank.json").read_text())
    bank = torch.load(cfg.bank_dir / "vc_bank.pt", map_location="cpu", weights_only=False)
    return manifest, bank


# ── phase: donors ─────────────────────────────────────────────────────


def ce_slot_position(rec: dict) -> int:
    """Parent ``slot_position(..., "ce")`` tolerant of no-prefix records (r5).

    The ce slot is the last context token — independent of ``prefix_end`` —
    but the parent helper asserts ``1 <= prefix_end``, unsatisfiable for a q35
    bare render (prefix_end 0, EMPTY varied prefix). pe >= 1 records route
    through the parent VERBATIM (q25 byte-equal); pe == 0 records must carry
    the bank r5 ``no_prefix`` flag (fail loud on an unflagged 0) and return
    ``ctx_len - 1`` directly (the parent's own ce arithmetic).
    """
    ctx_len, pe = int(rec["ctx_len"]), int(rec["prefix_end"])
    if pe >= 1:
        return R.slot_position(ctx_len, pe, "ce")
    assert rec.get("no_prefix"), (rec.get("context_id"), ctx_len, pe)
    assert ctx_len >= 2, (rec.get("context_id"), ctx_len)
    return ctx_len - 1


def gate_spot_position(rec: dict, slot: str) -> int:
    """pe-safe ``run_injection_gate`` position seam (r6): ce slots dispatch
    through :func:`ce_slot_position` (tolerant of flagged ``no_prefix`` pe==0
    bare renders — leg B q35's S2 gate spots select ``a=bare__q1``; an
    UNFLAGGED pe==0 still fails loud there); any other slot routes through the
    parent :func:`issue2162_run.slot_position` VERBATIM."""
    if slot == "ce":
        return ce_slot_position(rec)
    return R.slot_position(rec["ctx_len"], rec["prefix_end"], slot)


def _arm_ce_stack(
    cfg: RunConfig,
    model,
    bank: dict,
    pairs: list,
    donor_maps,
    pairs_by_id,
    variant: str,
    row_lens,
    T,
):
    """Parent-style all-layer ce replace stack for a batch of pairs (rows =
    context A of each pair; payload per plan: steered = V_ce(B), null =
    norm-matched shuffled-donor V_ce)."""
    recs = bank["per_context"]
    payloads = []
    positions = []
    for p in pairs:
        arm = "steered" if variant == "steered" else "shuffled"
        payload, _donor = payload_for_arm(bank, p, "ce", arm, donor_maps, pairs_by_id)
        payloads.append(payload)
        positions.append((ce_slot_position(recs[p.a]),))
    return R._arm_hook_all_layers(model, _R_cfg_proxy(cfg), row_lens, positions, payloads, T)


def payload_for_arm(bank, pair, slot, arm, donor_maps, pairs_by_id):
    """(1, L, H) ce payload — parent semantics over THIS bank (steered = own
    V_ce(B); shuffled = norm-matched value-constrained/deranged donor)."""
    recs = bank["per_context"]
    recipient = R._slot_state(recs[pair.b], slot).unsqueeze(0)
    if arm == "steered":
        return recipient.clone(), None
    donor_id = donor_maps["shuffled"][pair.pair_id]
    donor = pairs_by_id[donor_id]
    donor_state = R._slot_state(recs[donor.b], slot).unsqueeze(0)
    return BANK94.norm_match(donor_state, recipient), donor_id


@torch.no_grad()
def generate_donors(
    cfg: RunConfig, model, tok, bank: dict, pairs: list, donor_maps, pairs_by_id, ctx_ids, regime_fp
) -> dict:
    """Greedy 8-token openings + answer-position states 1..3 (all layers).

    med: context A with the banked ce patch armed (steered payload) — capture
    stack records the ACTUAL decode states of the patched run. bstart: context
    B unhooked. Returns {scheme: {pair_id: rec}} + rollout text rows.
    Chunk-checkpointed + resumable (``_ChunkStore``; 390 units > the ~50-unit
    durability trigger).
    """
    out: dict[str, dict[str, dict]] = {"med": {}, "bstart": {}}
    text_rows: list[dict] = []
    store = _ChunkStore(cfg.donors_dir / "donor_chunks", regime_fp)
    for scheme in C.ARM_SCHEMES:
        for start in range(0, len(pairs), cfg.gen_batch):
            chunk = pairs[start : start + cfg.gen_batch]
            chunk_key = f"{scheme}_{start:04d}"
            cached = store.load(chunk_key)
            if cached is not None:
                out[scheme].update(cached["recs"])
                text_rows.extend(cached["text_rows"])
                logger.info(
                    "[donors:%s] unit %d/%d pairs (resumed)",
                    scheme,
                    start + len(chunk),
                    len(pairs),
                )
                continue
            if scheme == "med":
                rows = [ctx_ids[p.a] for p in chunk]
            else:
                rows = [ctx_ids[p.b] for p in chunk]
            T = max(len(r) for r in rows)
            ce_stack = None
            if scheme == "med":
                ce_stack = _arm_ce_stack(
                    cfg,
                    model,
                    bank,
                    chunk,
                    donor_maps,
                    pairs_by_id,
                    "steered",
                    [len(r) for r in rows],
                    T,
                )
            cap_stack = joint_answer_hooks(model)
            cap_stack.arm_capture(len(rows), C.DONOR_K_MAX, expected_prompt_len=T)
            try:
                draws = generate_batch_ids(
                    model,
                    tok,
                    rows,
                    n=1,
                    stack=cap_stack,
                    donors_full=None,
                    max_new_tokens=C.DONOR_MAX_NEW_TOKENS,
                    greedy=True,
                    seed_base=cfg.seed_base,
                )
            finally:
                states = cap_stack.captured_states()
                cap_stack.remove()
                if ce_stack is not None:
                    ce_stack.remove()
            chunk_recs: dict[str, dict] = {}
            chunk_text_rows: list[dict] = []
            for j, p in enumerate(chunk):
                row = draws[0][j]
                k_len = min(C.DONOR_K_MAX, row["n_completion_tokens"])
                rec = {
                    "pair_id": p.pair_id,
                    "scheme": scheme,
                    "token_ids": row["gen_ids"][:k_len],
                    "donor_len": k_len,
                    "text": row["text"],
                    "states": states[j][:k_len].to(torch.float16),  # (k_len, L, H)
                }
                assert rec["states"].shape[0] == k_len, (rec["states"].shape, k_len)
                chunk_recs[p.pair_id] = rec
                chunk_text_rows.append(
                    {
                        "pair_id": p.pair_id,
                        "scheme": scheme,
                        "donor_len": k_len,
                        "n_completion_tokens": row["n_completion_tokens"],
                        "text": row["text"],
                    }
                )
            store.save(chunk_key, {"recs": chunk_recs, "text_rows": chunk_text_rows})
            out[scheme].update(chunk_recs)
            text_rows.extend(chunk_text_rows)
            logger.info(
                "[donors:%s] unit %d/%d pairs",
                scheme,
                min(start + cfg.gen_batch, len(pairs)),
                len(pairs),
            )
    return {"recs": out, "text_rows": text_rows}


def _gate_spots(s1_pairs: list, s2_pairs: list) -> list[dict]:
    """12 injection-gate spot rows: 2 per S1 cell (steered + shuffled) + 2 S2."""
    by_cell: dict[str, list] = {}
    for p in s1_pairs:
        by_cell.setdefault(p.cell, []).append(p)
    spots = []
    for cell in C.S1_CELLS:
        ps = sorted(by_cell[cell], key=lambda p: p.pair_id)
        spots.append({"cell": cell, "slot": "ce", "arm": "steered", "pair": ps[0]})
        spots.append({"cell": cell, "slot": "ce", "arm": "shuffled", "pair": ps[1]})
    s2 = sorted(s2_pairs, key=lambda p: p.pair_id)
    spots.append({"cell": C.S2_CELL, "slot": "ce", "arm": "steered", "pair": s2[0]})
    spots.append({"cell": C.S2_CELL, "slot": "ce", "arm": "shuffled", "pair": s2[1]})
    assert len(spots) == 12, len(spots)
    return spots


@torch.no_grad()
def run_decode_injection_gate(cfg: RunConfig, model, tok, donors, pairs: list, ctx_ids) -> dict:
    """Plan §7 decode-injection exactness: 2 spot rows, k=3 med steered.

    Leg 1 (edit exactness): a downstream capture hook (registered AFTER the
    edit stack) must read back the donor state at every edited (step, layer)
    — cos >= GATE_COS_MIN, norm ratio in [0.995, 1.005].
    Leg 2 (prefill untouched): the armed edit stack must not perturb prefill
    states — ce-position states with stack armed vs unhooked, two-bar (#779).
    """
    spots = [
        p
        for p in pairs
        if p.pair_id in donors["med"] and donors["med"][p.pair_id]["donor_len"] == C.DONOR_K_MAX
    ][:2]
    assert len(spots) == 2, "need 2 full-length med donors for the decode gate"
    results = []
    for p in spots:
        row = ctx_ids[p.a]
        donor_rec = donors["med"][p.pair_id]
        donor_states = donor_rec["states"].float()  # (3, L, H)
        edit_stack = joint_answer_hooks(model)
        cap_stack = joint_answer_hooks(model)  # registered after => sees post-edit
        try:
            cap_stack.arm_capture(1, C.DONOR_K_MAX, expected_prompt_len=len(row))
            generate_batch_ids(
                model,
                tok,
                [row],
                n=1,
                stack=edit_stack,
                donors_full=[donor_states],
                max_new_tokens=C.DONOR_K_MAX,
                greedy=True,
                seed_base=cfg.seed_base,
            )
            captured = cap_stack.captured_states()[0].float()  # (<=3, L, H)
        finally:
            cap_stack.remove()
            edit_stack.remove()
        assert captured.shape[0] >= 1, "decode gate captured no steps"
        k_eff = captured.shape[0]
        cos_min, ratio_lo, ratio_hi = 1.0, 1.0, 1.0
        for i in range(k_eff):
            for layer in range(donor_states.shape[1]):
                got = captured[i, layer]
                want = donor_states[i, layer]
                cos = float(torch.nn.functional.cosine_similarity(got, want, dim=0))
                ratio = float(got.norm() / want.norm().clamp_min(1e-12))
                cos_min = min(cos_min, cos)
                ratio_lo = min(ratio_lo, ratio)
                ratio_hi = max(ratio_hi, ratio)
        # Leg 2: prefill untouched (armed stack vs no stack, ce position).
        edit_stack2 = joint_answer_hooks(model)
        try:
            edit_stack2.arm_replace_batch([donor_states], expected_prompt_len=len(row))
            ids, mask = R._right_pad([row], tok.pad_token_id, cfg.device)
            hooked = extract_layer_activations(model, ids, cfg.layers, attention_mask=mask)
        finally:
            edit_stack2.remove()
        ids, mask = R._right_pad([row], tok.pad_token_id, cfg.device)
        clean = extract_layer_activations(model, ids, cfg.layers, attention_mask=mask)
        pos = len(row) - 1
        early = min(
            float(
                torch.nn.functional.cosine_similarity(
                    hooked[layer][0, pos].float(), clean[layer][0, pos].float(), dim=0
                )
            )
            for layer in C.PARITY_EARLY_LAYERS
        )
        flat_h = torch.stack([hooked[layer][0, pos].float() for layer in cfg.layers]).flatten()
        flat_c = torch.stack([clean[layer][0, pos].float() for layer in cfg.layers]).flatten()
        flat = float(torch.nn.functional.cosine_similarity(flat_h, flat_c, dim=0))
        results.append(
            {
                "pair_id": p.pair_id,
                "k_eff": k_eff,
                "edit_cos_min": cos_min,
                "edit_ratio_lo": ratio_lo,
                "edit_ratio_hi": ratio_hi,
                "prefill_early_cos": early,
                "prefill_flat_cos": flat,
            }
        )
    ok = all(
        r["edit_cos_min"] >= R.GATE_COS_MIN
        and R.GATE_NORM_RATIO_LO <= r["edit_ratio_lo"]
        and r["edit_ratio_hi"] <= R.GATE_NORM_RATIO_HI
        and r["prefill_early_cos"] >= C.PARITY_EARLY_COS_MIN
        and r["prefill_flat_cos"] >= C.PARITY_FLAT_COS_MIN
        for r in results
    )
    return {"verdict": "PASS" if ok else "FAIL", "spots": results}


def phase_donors(cfg: RunConfig, regime_fp: str) -> int:
    _require_single_worker(cfg, "donors")
    if not cfg.force and _phase_done(cfg, "donors", regime_fp):
        logger.info("[donors] done — skip")
        return RC_OK
    manifest, bank = _load_bank(cfg)
    s1_pairs, s2_pairs = build_pair_universe()
    dropped = set(manifest["minpair_dropped_pair_ids"])
    pairs = [p for p in [*s1_pairs, *s2_pairs] if p.pair_id not in dropped]
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = manifest["donor_maps"]
    ids_fn = make_ids_fn(cfg.model_tag)
    model, tok = load_model_and_tokenizer(cfg)
    contexts = build_context_universe(s1_pairs, s2_pairs)
    ctx_ids = {cid: ids_fn(tok, c) for cid, c in contexts.items()}

    # Gate 1 — prefill(ce) injection exactness, 12 spots (REUSED parent gate).
    gate1 = R.run_injection_gate(
        _R_cfg_proxy(cfg),
        model,
        tok,
        bank,
        pairs,
        donor_maps,
        contexts=contexts,
        ids_fn=ids_fn,
        spots=_gate_spots(
            [p for p in pairs if pair_set_of(p) == "s1"],
            [p for p in pairs if pair_set_of(p) == "s2"],
        ),
        payload_fn=payload_for_arm,
        position_fn=gate_spot_position,
    )
    R._write_json_atomic(cfg.gates_dir / "injection_gate_report.json", {**gate1, **_repro(cfg)})
    # Parent contract: R.run_injection_gate returns {"passed": bool} (no
    # "verdict" key — issue2162_run.py L1361; reused-module return schema).
    if gate1.get("passed") is not True:
        logger.error("[donors] prefill injection gate FAIL")
        return RC_INJECTION_GATE

    donors_out = generate_donors(
        cfg, model, tok, bank, pairs, donor_maps, pairs_by_id, ctx_ids, regime_fp
    )
    recs = donors_out["recs"]

    gate2 = run_decode_injection_gate(cfg, model, tok, recs, pairs, ctx_ids)
    R._write_json_atomic(
        cfg.gates_dir / "decode_injection_gate_report.json", {**gate2, **_repro(cfg)}
    )
    if gate2["verdict"] != "PASS":
        logger.error("[donors] decode injection gate FAIL: %s", gate2)
        return RC_INJECTION_GATE

    cfg.donors_dir.mkdir(parents=True, exist_ok=True)
    for scheme in C.ARM_SCHEMES:
        tmp = cfg.donors_dir / f"donors_{scheme}.tmp.pt"
        torch.save(recs[scheme], tmp)
        os.replace(tmp, cfg.donors_dir / f"donors_{scheme}.pt")
    R._write_jsonl_atomic(cfg.donors_dir / "donor_rollouts.jsonl", donors_out["text_rows"])
    edge = {
        pid: {s: recs[s][pid]["donor_len"] for s in C.ARM_SCHEMES}
        for pid in pairs_by_id
        if pid in recs["med"]
    }
    R._write_json_atomic(
        cfg.donors_dir / "donor_edge_accounting.json", {"donor_lens": edge, **_repro(cfg)}
    )
    _ChunkStore(cfg.donors_dir / "donor_chunks", regime_fp).clear()  # merged into donors_*.pt
    _write_phase_done(cfg, "donors", regime_fp, {"n_pairs": len(pairs)})
    logger.info("[donors] done: %d pairs x 2 schemes", len(pairs))
    return RC_OK


def _load_donors(cfg: RunConfig) -> dict[str, dict[str, dict]]:
    return {
        scheme: torch.load(
            cfg.donors_dir / f"donors_{scheme}.pt", map_location="cpu", weights_only=False
        )
        for scheme in C.ARM_SCHEMES
    }


# ── phase: grid ───────────────────────────────────────────────────────


def enumerate_blocks_2333(s1_pairs: list, s2_pairs: list, dropped: set[str]) -> list[R.Block]:
    """144 blocks: (5 S1 cells + 1 S2 cell) x 12 arms x 2 variants.

    Reuses R.Block with slot -> arm_slug and arm -> variant (key/slug/done-file
    machinery unchanged)."""
    by_cell: dict[str, list] = {}
    for p in s1_pairs:
        if p.pair_id not in dropped:
            by_cell.setdefault(p.cell, []).append(p)
    by_cell[C.S2_CELL] = [p for p in s2_pairs if p.pair_id not in dropped]
    blocks: list[R.Block] = []
    for cell in [*C.S1_CELLS, C.S2_CELL]:
        ids = tuple(p.pair_id for p in sorted(by_cell[cell], key=lambda p: p.pair_id))
        assert ids, cell
        for arm_slug in C.ARM_SLUGS:
            for variant in C.VARIANTS:
                blocks.append(R.Block(cell, arm_slug, variant, ids))
    assert len(blocks) == EXPECTED_N_BLOCKS, len(blocks)
    keys = [b.key for b in blocks]
    assert len(set(keys)) == len(keys)
    return blocks


def smoke_blocks_2333(s1_pairs: list, s2_pairs: list, dropped: set[str]) -> list[R.Block]:
    """1 S1 pair + 1 S2 pair through ALL 12 arms x 2 variants (48 blocks, K=1)."""
    s1_keep = sorted((p for p in s1_pairs if p.pair_id not in dropped), key=lambda p: p.pair_id)
    s2_keep = sorted((p for p in s2_pairs if p.pair_id not in dropped), key=lambda p: p.pair_id)
    blocks = []
    for cell, pid in ((s1_keep[0].cell, s1_keep[0].pair_id), (C.S2_CELL, s2_keep[0].pair_id)):
        for arm_slug in C.ARM_SLUGS:
            for variant in C.VARIANTS:
                blocks.append(R.Block(cell, arm_slug, variant, (pid,)))
    assert len(blocks) == 48, len(blocks)
    return blocks


def _pair_gen_input(
    kind: str, k: int, scheme: str, variant: str, p, donors, donor_maps, ctx_ids
) -> tuple[list[int], torch.Tensor | None, list[int], str]:
    """(gen_row_ids, donor_states_or_None, donor_ids, donor_pair_id) for ONE
    (pair, arm, variant) — the single donor-selection code path shared by
    run_block_2333 and the cap-regen pass (no drift between them)."""
    own = donors[scheme][p.pair_id]
    null_id = donor_maps["shuffled"][p.pair_id]
    null_rec = donors[scheme][null_id]
    if variant == "steered":
        d_ids = own["token_ids"][:k]
        d_states = own["states"][:k].float()
        donor_pid = p.pair_id
    else:
        d_ids = null_rec["token_ids"][:k]
        # scheme-matched norm-matched null: null donor states rescaled
        # positionwise to the recipient's OWN scheme donor states.
        d_states = BANK94.norm_match(null_rec["states"][:k].float(), own["states"][:k].float())
        donor_pid = null_id
    if kind == "patch":
        return ctx_ids[p.a], d_states, d_ids, donor_pid
    base = ctx_ids[p.a]
    return [*base, *d_ids], None, d_ids, donor_pid


def pair_dropped_for_arm(donors: dict, donor_maps: dict, pair_id: str, k: int, scheme: str) -> bool:
    """Edge rule (plan §4.2): drop the pair from the k-arm — steered AND null
    symmetric — when EITHER its own donor or its shuffled donor is < k tokens."""
    own = donors[scheme].get(pair_id)
    null_id = donor_maps["shuffled"][pair_id]
    null = donors[scheme].get(null_id)
    if own is None or null is None:
        return True
    return own["donor_len"] < k or null["donor_len"] < k


@torch.no_grad()
def _capture_va(
    cfg: RunConfig,
    model,
    tok,
    full_rows: list[list[int]],
    spans: list[tuple[int, int]],
    hooked_positions: list[tuple[int, ...]] | None,
    hooked_payloads: list[torch.Tensor] | None,
) -> list[torch.Tensor]:
    """Teacher-forced V_a: LEFT-padded forward, span-mean per layer -> (L, H).

    Patch arms pass ``hooked_positions``/``hooked_payloads`` ((k_eff, L, H)
    per row) — a parent PositionEditHookStack multi-position replace at the
    answer positions, teacher-forced-equivalent to the decode-step edits.
    """
    pad_id = tok.pad_token_id
    out: list[torch.Tensor] = []
    for start in range(0, len(full_rows), cfg.capture_batch):
        rows = full_rows[start : start + cfg.capture_batch]
        sp = spans[start : start + cfg.capture_batch]
        ids, mask = R._left_pad(rows, pad_id, cfg.device)
        T = ids.shape[1]
        stack = None
        if hooked_positions is not None:
            assert hooked_payloads is not None
            pos = hooked_positions[start : start + cfg.capture_batch]
            pay = hooked_payloads[start : start + cfg.capture_batch]
            # (k, L, H) -> (1?, ...) parent hook wants (n_pos, H) per layer via
            # per-row (n_pos, L, H) payload tensors.
            stack = R._arm_hook_all_layers(
                model,
                _R_cfg_proxy(cfg),
                [len(r) for r in rows],
                pos,
                [p for p in pay],
                T,
            )
        try:
            captured = extract_layer_activations(model, ids, cfg.layers, attention_mask=mask)
        finally:
            if stack is not None:
                stack.remove()
        for j, row in enumerate(rows):
            off = T - len(row)
            s, e = sp[j]
            assert 0 <= s < e <= len(row), (s, e, len(row))
            va = torch.stack(
                [captured[layer][j, off + s : off + e].float().mean(dim=0) for layer in cfg.layers]
            )
            out.append(va.to(torch.float16))
        del captured
    return out


def run_block_2333(
    cfg: RunConfig,
    model,
    tok,
    bank,
    donors,
    donor_maps,
    pairs_by_id,
    ctx_ids,
    block: R.Block,
    regime_fp: str,
) -> None:
    """One (cell, arm_slug, variant) block: hooked/prefill K-draw generation +
    V_a capture + atomic shard writes + done record (parent run_block shape)."""
    t0 = time.time()
    arm_slug, variant = block.slot, block.arm
    kind, k, scheme = C.parse_arm(arm_slug)
    recs = bank["per_context"]

    kept, dropped = [], []
    for pid in block.pair_ids:
        if pair_dropped_for_arm(donors, donor_maps, pid, k, scheme):
            dropped.append(pid)
        else:
            kept.append(pairs_by_id[pid])
    rows_out: list[dict] = []
    va_store: dict[str, torch.Tensor] = {}
    telem = {"n_edits": 0, "min_pre_cos": 1.0}

    for start in range(0, len(kept), cfg.gen_batch):
        chunk = kept[start : start + cfg.gen_batch]
        gen_rows: list[list[int]] = []
        donors_full: list[torch.Tensor | None] = []
        donor_ids_used: list[list[int]] = []
        donor_pids: list[str] = []
        for p in chunk:
            # Prefill token identity (plan §7) is carried by
            # generate_batch_ids' pad-tail verbatim assert (the REAL gate —
            # it re-reads the padded tensor tail); no in-block re-assert
            # (the r1 one was vacuous: it compared a list to itself).
            gen_row, d_states, d_ids, donor_pid = _pair_gen_input(
                kind, k, scheme, variant, p, donors, donor_maps, ctx_ids
            )
            gen_rows.append(gen_row)
            donors_full.append(d_states)
            donor_ids_used.append(d_ids)
            donor_pids.append(donor_pid)

        stack = joint_answer_hooks(model) if kind == "patch" else None
        try:
            draws = generate_batch_ids(
                model,
                tok,
                gen_rows,
                n=cfg.grid_draws,
                stack=stack,
                donors_full=donors_full if kind == "patch" else None,
                max_new_tokens=cfg.max_new_tokens,
                temperature=C.GRID_TEMPERATURE,
                seed_base=cfg.seed_base,
            )
            # realized_edits accumulates across ALL K draws (decode_hooks no
            # longer resets it per arm_replace — r1 Minor: last-draw-only).
            telemetry = stack.realized_edits() if stack is not None else {}
        finally:
            if stack is not None:
                stack.remove()

        # NOTE: cap-hit REGEN deliberately does NOT happen here. The
        # registered grain (plan §6) is per-CELL across BOTH variants —
        # a per-chunk single-variant regen mis-fires in both directions and
        # can give paired variants different caps (r1 blocker
        # cap-regen-wrong-grain). phase_grid runs `_cap_regen_pass` AFTER the
        # claim queue completes (all blocks done), pooling each (cell, arm)'s
        # steered+null rows.

        # V_a capture (flat rows: pair x draw).
        full_rows, spans, h_pos, h_pay, keys = [], [], [], [], []
        for i in range(cfg.grid_draws):
            for j, p in enumerate(chunk):
                row = draws[i][j]
                base = ctx_ids[p.a]
                if kind == "patch":
                    if not row["gen_ids"]:  # zero-length completion — skip V_a
                        continue
                    full = [*base, *row["gen_ids"]]
                    span = (len(base), len(full))
                    k_eff = min(k, len(row["gen_ids"]))
                    assert k_eff >= 1, (p.pair_id, k_eff)
                    h_pos.append(tuple(range(len(base), len(base) + k_eff)))
                    h_pay.append(donors_full[j][:k_eff])
                else:
                    full = [*base, *donor_ids_used[j], *row["gen_ids"]]
                    span = (len(base), len(full))
                    assert len(full) > span[0], (p.pair_id, span)  # donor ids >= 1
                full_rows.append(full)
                spans.append(span)
                keys.append(f"{p.pair_id}|{arm_slug}|{variant}|d{i}")
        if full_rows:
            vas = _capture_va(
                cfg,
                model,
                tok,
                full_rows,
                spans,
                h_pos if kind == "patch" else None,
                h_pay if kind == "patch" else None,
            )
            va_store.update(dict(zip(keys, vas)))

        for i in range(cfg.grid_draws):
            for j, p in enumerate(chunk):
                row = draws[i][j]
                # Diagnostic ONLY — the judged whole response is the ONE-SHOT
                # decode of donor_ids + gen_ids (a split decode can corrupt a
                # multi-byte char / cleanup-space seam between the segments;
                # r1 blocker prefill-response-decode).
                donor_text = tok.decode(donor_ids_used[j], skip_special_tokens=True)
                if kind == "prefill":
                    response = tok.decode(
                        [*donor_ids_used[j], *row["gen_ids"]], skip_special_tokens=True
                    )
                else:
                    response = row["text"]
                rows_out.append(
                    {
                        "block_key": block.key,
                        "cell": block.cell,
                        "set": pair_set_of(p),
                        "arm_slug": arm_slug,
                        "kind": kind,
                        "k": k,
                        "scheme": scheme,
                        "variant": variant,
                        "pair_id": p.pair_id,
                        "context_a": p.a,
                        "context_b": p.b,
                        "donor_pair_id": donor_pids[j],
                        "donor_len": len(donor_ids_used[j]),
                        "donor_text": donor_text,
                        "draw": i,
                        "seed": cfg.seed_base + i,
                        "temperature": C.GRID_TEMPERATURE,
                        "n_completion_tokens": row["n_completion_tokens"],
                        "cap_hit": R.cap_hit(
                            row["n_completion_tokens"],
                            int(row.get("regenerated_at", cfg.max_new_tokens)),
                        ),
                        "cap_hit_basis": "gen_token_count",
                        "regenerated_at": row.get("regenerated_at"),
                        "response_text": response,
                        "continuation_text": row["text"],
                        "va_key": f"{p.pair_id}|{arm_slug}|{variant}|d{i}",
                    }
                )
        # Realized-edit telemetry summary (per plan §4.3 — full per-(row, layer,
        # step) detail lives in the decode gate; blocks keep count + min-cos).
        for _layer, edits in telemetry.items():
            for _b, _step, cos, _pn, _dn in edits:
                telem["n_edits"] += 1
                telem["min_pre_cos"] = min(telem["min_pre_cos"], cos)
        logger.info(
            "[grid] block %s unit %d/%d pairs elapsed=%.0fs",
            block.key,
            min(start + cfg.gen_batch, len(kept)),
            len(kept),
            time.time() - t0,
        )

    R._write_jsonl_atomic(cfg.grid_blocks_dir / f"{block.slug}.jsonl", rows_out)
    cfg.va_dir.mkdir(parents=True, exist_ok=True)
    tmp = cfg.va_dir / f"{block.slug}.tmp.pt"
    torch.save(va_store, tmp)
    os.replace(tmp, cfg.va_dir / f"{block.slug}.pt")
    n_cap = sum(1 for r in rows_out if r["cap_hit"])
    # Done-file namespace MUST match phase_grid's claim-queue namespace
    # ("smoke_blocks" under --smoke) — a "blocks"-default write makes the
    # queue re-run every smoke block forever (namespace mismatch). Shard +
    # va dirs are namespaced the same way via the cfg properties (r1 Major 2).
    done_namespace = cfg.grid_namespace
    R._write_json_atomic(
        R.block_done_path(cfg.out_root, block, done_namespace),
        {
            "key": block.key,
            "regime_fp": regime_fp,
            "n_rows": len(rows_out),
            "n_pairs_kept": len(kept),
            "dropped_pair_ids": dropped,
            "n_cap_hit": n_cap,
            "edit_telemetry": telem,
            "wall_s": time.time() - t0,
            **_repro(cfg),
        },
    )
    logger.info(
        "[grid] block %s done: %d rows, %d dropped pairs, %d cap-hit, %.0fs",
        block.key,
        len(rows_out),
        len(dropped),
        n_cap,
        time.time() - t0,
    )


def _load_shard_rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]


def _was_capped_at_original(row: dict, cap: int) -> bool:
    """Original-cap status, idempotent across regen re-runs: a row already
    regenerated (regenerated_at set) WAS capped at the original cap."""
    if row.get("regenerated_at") is not None:
        return True
    return R.cap_hit(row["n_completion_tokens"], cap)


def _regen_block_capped_rows(
    cfg: RunConfig, model, tok, donors, donor_maps, pairs_by_id, ctx_ids, block: R.Block
) -> int:
    """Regenerate THIS block's cap-hit rows at 2x the cap (both-variant callers
    ensure the paired variant gets the same treatment), update the shard + va
    store + done record atomically. Returns the number of rows regenerated."""
    arm_slug, variant = block.slot, block.arm
    kind, k, scheme = C.parse_arm(arm_slug)
    shard_path = cfg.grid_blocks_dir / f"{block.slug}.jsonl"
    rows = _load_shard_rows(shard_path)
    todo = [
        idx
        for idx, r in enumerate(rows)
        if r.get("regenerated_at") is None
        and R.cap_hit(r["n_completion_tokens"], cfg.max_new_tokens)
    ]
    if not todo:
        return 0
    va_path = cfg.va_dir / f"{block.slug}.pt"
    va_store = (
        torch.load(va_path, map_location="cpu", weights_only=False) if va_path.exists() else {}
    )
    by_draw: dict[int, list[int]] = {}
    for idx in todo:
        by_draw.setdefault(int(rows[idx]["draw"]), []).append(idx)
    n_regen = 0
    for draw, idxs in sorted(by_draw.items()):
        for start in range(0, len(idxs), cfg.gen_batch):
            batch_idx = idxs[start : start + cfg.gen_batch]
            inputs = [
                _pair_gen_input(
                    kind,
                    k,
                    scheme,
                    variant,
                    pairs_by_id[rows[i]["pair_id"]],
                    donors,
                    donor_maps,
                    ctx_ids,
                )
                for i in batch_idx
            ]
            gen_rows = [t[0] for t in inputs]
            d_states = [t[1] for t in inputs]
            d_ids = [t[2] for t in inputs]
            stack = joint_answer_hooks(model) if kind == "patch" else None
            try:
                redraw = generate_batch_ids(
                    model,
                    tok,
                    gen_rows,
                    n=1,
                    stack=stack,
                    donors_full=d_states if kind == "patch" else None,
                    max_new_tokens=2 * cfg.max_new_tokens,
                    temperature=C.GRID_TEMPERATURE,
                    seed_base=cfg.seed_base + draw,
                )
            finally:
                if stack is not None:
                    stack.remove()
            # V_a recapture for the regenerated rows.
            full_rows, spans, h_pos, h_pay, keys = [], [], [], [], []
            for j, i in enumerate(batch_idx):
                new = redraw[0][j]
                r = rows[i]
                base = ctx_ids[pairs_by_id[r["pair_id"]].a]
                key = r["va_key"]
                if kind == "patch":
                    if not new["gen_ids"]:
                        va_store.pop(key, None)  # stale entry from the capped draw
                    else:
                        k_eff = min(k, len(new["gen_ids"]))
                        full_rows.append([*base, *new["gen_ids"]])
                        spans.append((len(base), len(base) + len(new["gen_ids"])))
                        h_pos.append(tuple(range(len(base), len(base) + k_eff)))
                        h_pay.append(d_states[j][:k_eff])
                        keys.append(key)
                else:
                    full_rows.append([*base, *d_ids[j], *new["gen_ids"]])
                    spans.append((len(base), len(base) + len(d_ids[j]) + len(new["gen_ids"])))
                    keys.append(key)
                response = (
                    tok.decode([*d_ids[j], *new["gen_ids"]], skip_special_tokens=True)
                    if kind == "prefill"
                    else new["text"]
                )
                rows[i] = {
                    **r,
                    "n_completion_tokens": new["n_completion_tokens"],
                    "cap_hit": R.cap_hit(new["n_completion_tokens"], 2 * cfg.max_new_tokens),
                    "regenerated_at": 2 * cfg.max_new_tokens,
                    "response_text": response,
                    "continuation_text": new["text"],
                }
                n_regen += 1
            if full_rows:
                vas = _capture_va(
                    cfg,
                    model,
                    tok,
                    full_rows,
                    spans,
                    h_pos if kind == "patch" else None,
                    h_pay if kind == "patch" else None,
                )
                va_store.update(dict(zip(keys, vas)))
    R._write_jsonl_atomic(shard_path, rows)
    tmp = cfg.va_dir / f"{block.slug}.regen.tmp.pt"
    torch.save(va_store, tmp)
    os.replace(tmp, va_path)
    done_path = R.block_done_path(cfg.out_root, block, cfg.grid_namespace)
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["n_cap_hit"] = sum(1 for r in rows if r["cap_hit"])
    done["cap_regen_applied"] = done.get("cap_regen_applied", 0) + n_regen
    R._write_json_atomic(done_path, done)
    logger.info(
        "[capregen] block %s: regenerated %d rows at %d", block.key, n_regen, 2 * cfg.max_new_tokens
    )
    return n_regen


def _cap_regen_pass(
    cfg: RunConfig,
    model,
    tok,
    donors,
    donor_maps,
    pairs_by_id,
    ctx_ids,
    blocks: list[R.Block],
    regime_fp: str,
) -> None:
    """Registered cap-hit regen (plan §6, r1 blocker cap-regen-wrong-grain):
    the trigger is evaluated per (cell x arm) POOLING BOTH VARIANTS' rows —
    frac(cap-hit at the original cap) > 2% => regenerate the cap-hit rows of
    BOTH paired variants at 2x the cap. Runs AFTER run_claim_queue returned
    (the queue polls until EVERY block is done, so both variants exist), as
    its own claim-queue over (cell, arm) pseudo-blocks — single-flight across
    the 8 workers, resumable, fingerprint-keyed."""
    by_unit: dict[tuple[str, str], dict[str, R.Block]] = {}
    for b in blocks:
        by_unit.setdefault((b.cell, b.slot), {})[b.arm] = b
    pseudo = [R.Block(cell, slug, "capregen", ()) for (cell, slug) in sorted(by_unit)]
    regen_ns = f"{cfg.grid_namespace}_capregen"

    def run_one(pb: R.Block) -> None:
        unit = by_unit[(pb.cell, pb.slot)]
        rows_all = [
            r
            for blk in unit.values()
            for r in _load_shard_rows(cfg.grid_blocks_dir / f"{blk.slug}.jsonl")
        ]
        n_cap = sum(1 for r in rows_all if _was_capped_at_original(r, cfg.max_new_tokens))
        frac = n_cap / len(rows_all) if rows_all else 0.0
        fire = bool(n_cap) and frac > CAP_HIT_REGEN_FRAC
        n_regen = 0
        if fire:
            for blk in unit.values():
                n_regen += _regen_block_capped_rows(
                    cfg, model, tok, donors, donor_maps, pairs_by_id, ctx_ids, blk
                )
        R._write_json_atomic(
            R.block_done_path(cfg.out_root, pb, regen_ns),
            {
                "key": pb.key,
                "regime_fp": regime_fp,
                "cap_frac_original": frac,
                "n_cap_hit_original": n_cap,
                "n_rows": len(rows_all),
                "threshold": CAP_HIT_REGEN_FRAC,
                "fired": fire,
                "n_regenerated": n_regen,
                "variants_pooled": sorted(unit),
                **_repro(cfg),
            },
        )

    stats = R.run_claim_queue(_R_cfg_proxy(cfg), pseudo, regime_fp, regen_ns, run_one)
    logger.info("[capregen] pass complete over %d (cell, arm) units: %s", len(pseudo), stats)


def phase_grid(cfg: RunConfig, regime_fp: str) -> int:
    manifest, bank = _load_bank(cfg)
    donors = _load_donors(cfg)
    s1_pairs, s2_pairs = build_pair_universe()
    dropped = set(manifest["minpair_dropped_pair_ids"])
    pairs = [p for p in [*s1_pairs, *s2_pairs] if p.pair_id not in dropped]
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = manifest["donor_maps"]
    ids_fn = make_ids_fn(cfg.model_tag)
    model, tok = load_model_and_tokenizer(cfg)
    contexts = build_context_universe(s1_pairs, s2_pairs)
    ctx_ids = {cid: ids_fn(tok, c) for cid, c in contexts.items()}

    if cfg.smoke:
        blocks = smoke_blocks_2333(s1_pairs, s2_pairs, dropped)
    else:
        blocks = enumerate_blocks_2333(s1_pairs, s2_pairs, dropped)
    namespace = cfg.grid_namespace
    if cfg.only_blocks:
        blocks = [b for b in blocks if any(s in b.key for s in cfg.only_blocks)]
        assert blocks, f"--only-blocks {cfg.only_blocks} matched nothing"

    def run_one(block: R.Block) -> None:
        run_block_2333(
            cfg, model, tok, bank, donors, donor_maps, pairs_by_id, ctx_ids, block, regime_fp
        )
        _maybe_incremental_upload(cfg)

    if cfg.pilot:
        pending = [b for b in blocks if not R.block_is_done(cfg.out_root, b, regime_fp, namespace)]
        assert pending, "pilot: no pending blocks"
        block = pending[len(pending) // 2]
        t0 = time.time()
        run_block_2333(
            cfg, model, tok, bank, donors, donor_maps, pairs_by_id, ctx_ids, block, regime_fp
        )
        per_block = time.time() - t0
        projected_h = per_block * len(blocks) / max(1, cfg.num_workers) / 3600.0
        report = {
            "per_block_s": per_block,
            "n_blocks": len(blocks),
            "num_workers": cfg.num_workers,
            "projected_wall_h": projected_h,
            "planned_wall_h": cfg.planned_wall_h,
            "ratio": projected_h / cfg.planned_wall_h,
            "refusal_mult": R.PILOT_REFUSAL_MULT,
            **_repro(cfg),
        }
        R._write_json_atomic(cfg.manifest_dir / "pilot_report.json", report)
        logger.info(
            "[grid:pilot] projected %.2fh vs planned %.2fh", projected_h, cfg.planned_wall_h
        )
        if projected_h > R.PILOT_REFUSAL_MULT * cfg.planned_wall_h:
            return RC_PILOT_GATE
        return RC_OK

    stats = R.run_claim_queue(_R_cfg_proxy(cfg), blocks, regime_fp, namespace, run_one)
    # Registered cap-hit regen (plan §6) — AFTER the queue (all blocks done,
    # incl. other workers': run_claim_queue polls to completion), pooled per
    # (cell, arm) across BOTH variants (r1 blocker cap-regen-wrong-grain).
    _cap_regen_pass(cfg, model, tok, donors, donor_maps, pairs_by_id, ctx_ids, blocks, regime_fp)
    _maybe_incremental_upload(cfg, force=True)
    _write_phase_done(
        cfg, f"grid_{namespace}", regime_fp, {"stats": stats, "n_blocks": len(blocks)}
    )
    logger.info("[grid] complete: %s", stats)
    return RC_OK


_UPLOAD_COUNTER = {"blocks_since": 0}


def _maybe_incremental_upload(cfg: RunConfig, force: bool = False) -> None:
    """Bulk-upload staged text + va shards every N blocks (plan §9 incremental
    persistence; one bulk commit per leg — 256-commits/hr cap, parent shape)."""
    if cfg.upload_mode != "hf":
        return
    _UPLOAD_COUNTER["blocks_since"] += 0 if force else 1
    if not force and _UPLOAD_COUNTER["blocks_since"] < cfg.upload_every:
        return
    _UPLOAD_COUNTER["blocks_since"] = 0
    # Remote paths mirror the local NAMESPACED dir names, so smoke artifacts
    # can never land under (or sweep) the production prefixes (r1 Major 2).
    R.upload_dir_hf(
        cfg.rollouts_dir, f"{hf_prefix(cfg)}/rollouts", [f"{cfg.grid_namespace}/*.jsonl"]
    )
    R.upload_dir_hf(cfg.va_dir, f"{hf_prefix(cfg)}/{cfg.va_dir.name}", ["*.pt"])


# ── phase: anchors (q35) ──────────────────────────────────────────────


def _anchors_stale_width_guard(cfg: RunConfig) -> None:
    """Fail loud when anchors shards from a DIFFERENT sharding width exist:
    ``sorted(contexts)[i::N]`` partitions depend on N, so mixed-width shards
    double-cover or gap contexts silently (num_workers is deliberately OUT of
    the global regime fingerprint — the width is anchored HERE instead)."""
    tag = f"of{cfg.num_workers}."
    stale = [p.name for p in sorted(cfg.anchors_dir.glob("anchors_w*.jsonl")) if tag not in p.name]
    if stale:
        raise RuntimeError(
            f"anchors shards from a different --num-workers width exist: {stale}; "
            f"current width is {cfg.num_workers}. Remove/relocate them (or rerun at "
            "the original width) before proceeding."
        )


def phase_anchors(cfg: RunConfig, regime_fp: str) -> int:
    assert cfg.model_tag == "q35", "anchors phase is q35-only (q25 anchors are banked, plan §4.4)"
    shard_key = f"w{cfg.worker_index}of{cfg.num_workers}"
    phase_key = f"anchors_{shard_key}"
    if not cfg.force and _phase_done(cfg, phase_key, regime_fp):
        logger.info("[anchors] done — skip")
        return RC_OK
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    _anchors_stale_width_guard(cfg)
    s1_pairs, s2_pairs = build_pair_universe()
    contexts = build_context_universe(s1_pairs, s2_pairs)
    ids_fn = make_ids_fn(cfg.model_tag)
    model, tok = load_model_and_tokenizer(cfg)
    order = sorted(contexts)[cfg.worker_index :: cfg.num_workers]
    if cfg.smoke:
        order = order[:2]
    rows_out: list[dict] = []
    va_store: dict[str, torch.Tensor] = {}
    draws_n = 2 if cfg.smoke else cfg.anchor_draws
    for start in range(0, len(order), cfg.gen_batch):
        chunk = order[start : start + cfg.gen_batch]
        rows = [ids_fn(tok, contexts[c]) for c in chunk]
        draws = generate_batch_ids(
            model,
            tok,
            rows,
            n=draws_n,
            max_new_tokens=cfg.max_new_tokens,
            temperature=C.GRID_TEMPERATURE,
            seed_base=cfg.seed_base,
        )
        full_rows, spans, keys = [], [], []
        for i in range(draws_n):
            for j, cid in enumerate(chunk):
                row = draws[i][j]
                if not row["gen_ids"]:
                    continue
                full_rows.append([*rows[j], *row["gen_ids"]])
                spans.append((len(rows[j]), len(rows[j]) + len(row["gen_ids"])))
                keys.append(f"{cid}|anchor|d{i}")
        vas = _capture_va(cfg, model, tok, full_rows, spans, None, None)
        va_store.update(dict(zip(keys, vas)))
        for i in range(draws_n):
            for j, cid in enumerate(chunk):
                row = draws[i][j]
                rows_out.append(
                    {
                        "context_id": cid,
                        "draw": i,
                        "seed": cfg.seed_base + i,
                        "n_completion_tokens": row["n_completion_tokens"],
                        "cap_hit": R.cap_hit(row["n_completion_tokens"], cfg.max_new_tokens),
                        "response_text": row["text"],
                        "va_key": f"{cid}|anchor|d{i}",
                    }
                )
        logger.info(
            "[anchors] unit %d/%d contexts", min(start + cfg.gen_batch, len(order)), len(order)
        )
    # Registered cap policy (plan §6) applied to the anchor rows this worker
    # owns: pooled cap-hit frac > 2% => regenerate the capped rows at 2x cap
    # (no variants/pairs here — the pool is this worker's context slice; the
    # per-worker grain is disclosed in the phase-done record).
    n_cap0 = sum(1 for r in rows_out if r["cap_hit"])
    frac0 = n_cap0 / len(rows_out) if rows_out else 0.0
    n_regen = 0
    if n_cap0 and frac0 > CAP_HIT_REGEN_FRAC:
        by_draw: dict[int, list[int]] = {}
        for idx, r in enumerate(rows_out):
            if r["cap_hit"]:
                by_draw.setdefault(int(r["draw"]), []).append(idx)
        for draw, idxs in sorted(by_draw.items()):
            for start in range(0, len(idxs), cfg.gen_batch):
                batch_idx = idxs[start : start + cfg.gen_batch]
                gen_rows = [ids_fn(tok, contexts[rows_out[i]["context_id"]]) for i in batch_idx]
                redraw = generate_batch_ids(
                    model,
                    tok,
                    gen_rows,
                    n=1,
                    max_new_tokens=2 * cfg.max_new_tokens,
                    temperature=C.GRID_TEMPERATURE,
                    seed_base=cfg.seed_base + draw,
                )
                full_rows, spans, keys = [], [], []
                for j, i in enumerate(batch_idx):
                    new = redraw[0][j]
                    r = rows_out[i]
                    if not new["gen_ids"]:
                        va_store.pop(r["va_key"], None)
                    else:
                        full_rows.append([*gen_rows[j], *new["gen_ids"]])
                        spans.append((len(gen_rows[j]), len(gen_rows[j]) + len(new["gen_ids"])))
                        keys.append(r["va_key"])
                    rows_out[i] = {
                        **r,
                        "n_completion_tokens": new["n_completion_tokens"],
                        "cap_hit": R.cap_hit(new["n_completion_tokens"], 2 * cfg.max_new_tokens),
                        "regenerated_at": 2 * cfg.max_new_tokens,
                        "response_text": new["text"],
                    }
                    n_regen += 1
                if full_rows:
                    vas = _capture_va(cfg, model, tok, full_rows, spans, None, None)
                    va_store.update(dict(zip(keys, vas)))
        logger.info("[anchors] cap regen fired: frac=%.3f, %d rows at 2x cap", frac0, n_regen)

    R._write_jsonl_atomic(cfg.anchors_dir / f"anchors_{shard_key}.jsonl", rows_out)
    tmp = cfg.anchors_dir / f"va_anchors_{shard_key}.tmp.pt"
    torch.save(va_store, tmp)
    os.replace(tmp, cfg.anchors_dir / f"va_anchors_{shard_key}.pt")
    _write_phase_done(
        cfg,
        phase_key,
        regime_fp,
        {
            "n_rows": len(rows_out),
            "draws": draws_n,
            "num_workers": cfg.num_workers,
            "cap_frac_original": frac0,
            "cap_regen_applied": n_regen,
            "cap_regen_grain": "per-worker context slice (pooled)",
        },
    )
    return RC_OK


# ── phase: ce_control (q35) ───────────────────────────────────────────


def _ce_regen_block_capped_rows(
    cfg: RunConfig, model, tok, bank, donor_maps, pairs_by_id, ctx_ids, recs, block: R.Block
) -> int:
    """ce_control sibling of _regen_block_capped_rows: regenerate THIS ce
    block's cap-hit rows at 2x the cap (prefill-hook re-armed per draw),
    update shard + va store + done record atomically."""
    arm = "steered" if block.arm == "steered" else "shuffled"
    shard_path = cfg.ce_blocks_dir / f"{block.slug}.jsonl"
    rows = _load_shard_rows(shard_path)
    todo = [
        idx
        for idx, r in enumerate(rows)
        if r.get("regenerated_at") is None
        and R.cap_hit(r["n_completion_tokens"], cfg.max_new_tokens)
    ]
    if not todo:
        return 0
    va_path = cfg.va_dir / f"ce_{block.slug}.pt"
    va_store = (
        torch.load(va_path, map_location="cpu", weights_only=False) if va_path.exists() else {}
    )
    by_draw: dict[int, list[int]] = {}
    for idx in todo:
        by_draw.setdefault(int(rows[idx]["draw"]), []).append(idx)
    n_regen = 0
    for draw, idxs in sorted(by_draw.items()):
        for start in range(0, len(idxs), cfg.gen_batch):
            batch_idx = idxs[start : start + cfg.gen_batch]
            chunk = [pairs_by_id[rows[i]["pair_id"]] for i in batch_idx]
            gen_rows = [ctx_ids[p.a] for p in chunk]
            T = max(len(r) for r in gen_rows)
            payloads, positions = [], []
            for p in chunk:
                payload, _donor_pid = payload_for_arm(bank, p, "ce", arm, donor_maps, pairs_by_id)
                payloads.append(payload)
                positions.append((ce_slot_position(recs[p.a]),))
            stack = R._arm_hook_all_layers(
                model, _R_cfg_proxy(cfg), [len(r) for r in gen_rows], positions, payloads, T
            )
            try:
                stack.arm(T)
                redraw = generate_batch_ids(
                    model,
                    tok,
                    gen_rows,
                    n=1,
                    max_new_tokens=2 * cfg.max_new_tokens,
                    temperature=C.GRID_TEMPERATURE,
                    seed_base=cfg.seed_base + draw,
                )
            finally:
                stack.remove()
            full_rows, spans, h_pos, h_pay, keys = [], [], [], [], []
            for j, i in enumerate(batch_idx):
                new = redraw[0][j]
                r = rows[i]
                base = ctx_ids[chunk[j].a]
                if not new["gen_ids"]:
                    va_store.pop(r["va_key"], None)  # stale entry from the capped draw
                else:
                    full_rows.append([*base, *new["gen_ids"]])
                    spans.append((len(base), len(base) + len(new["gen_ids"])))
                    h_pos.append(positions[j])
                    h_pay.append(payloads[j])
                    keys.append(r["va_key"])
                rows[i] = {
                    **r,
                    "n_completion_tokens": new["n_completion_tokens"],
                    "cap_hit": R.cap_hit(new["n_completion_tokens"], 2 * cfg.max_new_tokens),
                    "regenerated_at": 2 * cfg.max_new_tokens,
                    "response_text": new["text"],
                    "continuation_text": new["text"],
                }
                n_regen += 1
            if full_rows:
                vas = _capture_va(cfg, model, tok, full_rows, spans, h_pos, h_pay)
                va_store.update(dict(zip(keys, vas)))
    R._write_jsonl_atomic(shard_path, rows)
    tmp = cfg.va_dir / f"ce_{block.slug}.regen.tmp.pt"
    torch.save(va_store, tmp)
    os.replace(tmp, va_path)
    done_path = R.block_done_path(cfg.out_root, block, cfg.ce_namespace)
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["n_cap_hit"] = sum(1 for r in rows if r["cap_hit"])
    done["cap_regen_applied"] = done.get("cap_regen_applied", 0) + n_regen
    R._write_json_atomic(done_path, done)
    logger.info("[capregen:ce] block %s: regenerated %d rows", block.key, n_regen)
    return n_regen


def _ce_cap_regen_pass(
    cfg: RunConfig,
    model,
    tok,
    bank,
    donor_maps,
    pairs_by_id,
    ctx_ids,
    recs,
    blocks: list[R.Block],
    regime_fp: str,
) -> None:
    """Cap-hit regen for ce_control — same registered grain as _cap_regen_pass
    (per cell, pooled across BOTH variants; plan §6), claim-queued."""
    by_cell: dict[str, dict[str, R.Block]] = {}
    for b in blocks:
        by_cell.setdefault(b.cell, {})[b.arm] = b
    pseudo = [R.Block(cell, "ce_replace", "capregen", ()) for cell in sorted(by_cell)]
    regen_ns = f"{cfg.ce_namespace}_capregen"

    def run_one(pb: R.Block) -> None:
        unit = by_cell[pb.cell]
        rows_all = [
            r
            for blk in unit.values()
            for r in _load_shard_rows(cfg.ce_blocks_dir / f"{blk.slug}.jsonl")
        ]
        n_cap = sum(1 for r in rows_all if _was_capped_at_original(r, cfg.max_new_tokens))
        frac = n_cap / len(rows_all) if rows_all else 0.0
        fire = bool(n_cap) and frac > CAP_HIT_REGEN_FRAC
        n_regen = 0
        if fire:
            for blk in unit.values():
                n_regen += _ce_regen_block_capped_rows(
                    cfg, model, tok, bank, donor_maps, pairs_by_id, ctx_ids, recs, blk
                )
        R._write_json_atomic(
            R.block_done_path(cfg.out_root, pb, regen_ns),
            {
                "key": pb.key,
                "regime_fp": regime_fp,
                "cap_frac_original": frac,
                "n_cap_hit_original": n_cap,
                "n_rows": len(rows_all),
                "threshold": CAP_HIT_REGEN_FRAC,
                "fired": fire,
                "n_regenerated": n_regen,
                "variants_pooled": sorted(unit),
                **_repro(cfg),
            },
        )

    stats = R.run_claim_queue(_R_cfg_proxy(cfg), pseudo, regime_fp, regen_ns, run_one)
    logger.info("[capregen:ce] pass complete over %d cells: %s", len(pseudo), stats)


def phase_ce_control(cfg: RunConfig, regime_fp: str) -> int:
    """q35 banked-ce control: 195 pairs x {steered, shuffled} x K=5 ce replace
    (the D3 bridge cell — plan §4.4 B1)."""
    assert cfg.model_tag == "q35", "ce_control is q35-only (q25 ce cells are banked)"
    manifest, bank = _load_bank(cfg)
    s1_pairs, s2_pairs = build_pair_universe()
    dropped = set(manifest["minpair_dropped_pair_ids"])
    pairs = [p for p in [*s1_pairs, *s2_pairs] if p.pair_id not in dropped]
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_maps = manifest["donor_maps"]
    ids_fn = make_ids_fn(cfg.model_tag)
    model, tok = load_model_and_tokenizer(cfg)
    contexts = build_context_universe(s1_pairs, s2_pairs)
    ctx_ids = {cid: ids_fn(tok, c) for cid, c in contexts.items()}
    recs = bank["per_context"]

    by_cell: dict[str, list] = {}
    for p in pairs:
        by_cell.setdefault(cell_of(p), []).append(p)
    blocks = []
    for cell in [*C.S1_CELLS, C.S2_CELL]:
        ids = tuple(p.pair_id for p in sorted(by_cell[cell], key=lambda p: p.pair_id))
        if cfg.smoke:
            ids = ids[:1]
        for variant in C.VARIANTS:
            blocks.append(R.Block(cell, "ce_replace", variant, ids))
    draws_n = 1 if cfg.smoke else C.CE_CONTROL_DRAWS

    def run_one(block: R.Block) -> None:
        t0 = time.time()
        kept = [pairs_by_id[pid] for pid in block.pair_ids]
        rows_out: list[dict] = []
        va_store: dict[str, torch.Tensor] = {}
        for start in range(0, len(kept), cfg.gen_batch):
            chunk = kept[start : start + cfg.gen_batch]
            gen_rows = [ctx_ids[p.a] for p in chunk]
            T = max(len(r) for r in gen_rows)
            arm = "steered" if block.arm == "steered" else "shuffled"
            payloads, positions, donor_pids = [], [], []
            for p in chunk:
                payload, donor_pid = payload_for_arm(bank, p, "ce", arm, donor_maps, pairs_by_id)
                payloads.append(payload)
                positions.append((ce_slot_position(recs[p.a]),))
                donor_pids.append(donor_pid or p.pair_id)
            stack = R._arm_hook_all_layers(
                model, _R_cfg_proxy(cfg), [len(r) for r in gen_rows], positions, payloads, T
            )
            try:
                # Parent PositionEditHook applies at PREFILL; per-draw re-arm
                # resets its latch (stack.arm is called by generate loop? no —
                # arm per draw manually).
                draws = []
                for i in range(draws_n):
                    stack.arm(T)
                    d = generate_batch_ids(
                        model,
                        tok,
                        gen_rows,
                        n=1,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=C.GRID_TEMPERATURE,
                        seed_base=cfg.seed_base + i,
                    )
                    draws.append(d[0])
            finally:
                stack.remove()
            full_rows, spans, h_pos, h_pay, keys = [], [], [], [], []
            for i in range(draws_n):
                for j, p in enumerate(chunk):
                    row = draws[i][j]
                    if not row["gen_ids"]:
                        continue
                    base = ctx_ids[p.a]
                    full_rows.append([*base, *row["gen_ids"]])
                    spans.append((len(base), len(base) + len(row["gen_ids"])))
                    h_pos.append(positions[j])
                    h_pay.append(payloads[j])
                    keys.append(f"{p.pair_id}|ce_replace|{block.arm}|d{i}")
            vas = _capture_va(cfg, model, tok, full_rows, spans, h_pos, h_pay)
            va_store.update(dict(zip(keys, vas)))
            for i in range(draws_n):
                for j, p in enumerate(chunk):
                    row = draws[i][j]
                    rows_out.append(
                        {
                            "block_key": block.key,
                            "cell": block.cell,
                            "set": pair_set_of(p),
                            "arm_slug": "ce_replace",
                            "kind": "ce",
                            "k": 0,
                            "scheme": "ce",
                            "variant": block.arm,
                            "pair_id": p.pair_id,
                            "context_a": p.a,
                            "context_b": p.b,
                            "donor_pair_id": donor_pids[j],
                            "draw": i,
                            "seed": cfg.seed_base + i,
                            "temperature": C.GRID_TEMPERATURE,
                            "n_completion_tokens": row["n_completion_tokens"],
                            "cap_hit": R.cap_hit(row["n_completion_tokens"], cfg.max_new_tokens),
                            "response_text": row["text"],
                            "continuation_text": row["text"],
                            "va_key": f"{p.pair_id}|ce_replace|{block.arm}|d{i}",
                        }
                    )
        R._write_jsonl_atomic(cfg.ce_blocks_dir / f"{block.slug}.jsonl", rows_out)
        cfg.va_dir.mkdir(parents=True, exist_ok=True)
        tmp = cfg.va_dir / f"ce_{block.slug}.tmp.pt"
        torch.save(va_store, tmp)
        os.replace(tmp, cfg.va_dir / f"ce_{block.slug}.pt")
        R._write_json_atomic(
            R.block_done_path(cfg.out_root, block, cfg.ce_namespace),
            {
                "key": block.key,
                "regime_fp": regime_fp,
                "n_rows": len(rows_out),
                "n_cap_hit": sum(1 for r in rows_out if r["cap_hit"]),
                "wall_s": time.time() - t0,
                **_repro(cfg),
            },
        )

    stats = R.run_claim_queue(_R_cfg_proxy(cfg), blocks, regime_fp, cfg.ce_namespace, run_one)
    # ce_control inherits the SAME registered cap policy (plan §6): pooled
    # per cell across both variants, regen at 2x cap (r1: fresh q35 rows
    # need the policy too — the banked q25 ce cells were produced under it).
    _ce_cap_regen_pass(
        cfg, model, tok, bank, donor_maps, pairs_by_id, ctx_ids, recs, blocks, regime_fp
    )
    _write_phase_done(cfg, cfg.ce_namespace, regime_fp, {"stats": stats, "n_blocks": len(blocks)})
    return RC_OK


# ── phase: upload ─────────────────────────────────────────────────────


def phase_upload(cfg: RunConfig, regime_fp: str) -> int:
    prefix = hf_prefix(cfg)
    uploaded: dict[str, int] = {}
    if cfg.upload_mode == "hf":
        # Every leg mirrors the local NAMESPACED dir/pattern names, so a smoke
        # run's upload can only ever land under smoke-named remote paths and a
        # production upload can never sweep smoke shards (r1 Major 2).
        legs = [
            (
                cfg.rollouts_dir,
                f"{prefix}/rollouts",
                [f"{cfg.grid_namespace}/*.jsonl", f"{cfg.ce_namespace}/*.jsonl"],
            ),
            (cfg.donors_dir, f"{prefix}/donors", ["*.jsonl", "*.json", "*.pt"]),
            (cfg.bank_dir, f"{prefix}/vc_bank", ["bank.json", "vc_bank.pt"]),
            (cfg.gates_dir, f"{prefix}/gates", ["*.json"]),
            (
                cfg.manifest_dir,
                f"{prefix}/manifests",
                [
                    "*.json",
                    f"{cfg.grid_namespace}/*.json",
                    f"{cfg.grid_namespace}_capregen/*.json",
                    f"{cfg.ce_namespace}/*.json",
                    f"{cfg.ce_namespace}_capregen/*.json",
                ],
            ),
            (cfg.va_dir, f"{prefix}/{cfg.va_dir.name}", ["*.pt"]),
            (cfg.anchors_dir, f"{prefix}/{cfg.anchors_dir.name}", ["*.jsonl", "*.pt"]),
        ]
        for local, remote, pats in legs:
            if local.exists():
                paths = R.upload_dir_hf(local, remote, pats)
                uploaded[remote] = len(paths)
    payload = {
        "issue": 2333,
        "model_tag": cfg.model_tag,
        "phase": "upload",
        "regime_fp": regime_fp,
        "hf_prefix": prefix,
        "hf_repo": R.HF_DATA_WRITE_REPO,
        "uploaded": uploaded,
        **_repro(cfg),
    }
    sentinel_name = (
        f"issue-2333-{cfg.model_tag}-smoke-results.json"
        if cfg.smoke
        else f"issue-2333-{cfg.model_tag}-results.json"
    )
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(
        cfg.log_dir / sentinel_name,
        {
            "sentinel_schema_version": 1,
            "kind": "epm:smoke-result" if cfg.smoke else "epm:results",
            "version": 1,
            "note": json.dumps(payload),
        },
    )
    _write_phase_done(
        cfg, "upload" + ("_smoke" if cfg.smoke else ""), regime_fp, {"uploaded": uploaded}
    )
    logger.info("[phase=done] upload complete: %s", uploaded)
    return RC_OK


# ── import-check / main ───────────────────────────────────────────────


def _import_check() -> None:
    """Execute every deferred import + arg-attribute completeness (fail-loud)."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    from huggingface_hub import hf_hub_download  # noqa: F401
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        _upload_folder_filtered,
        retry_transient,
    )

    s1, s2 = build_pair_universe()
    assert len(s1) == 180 and len(s2) == 15, (len(s1), len(s2))
    _ = build_context_universe(s1, s2)
    maps = build_donor_maps(s1, s2)
    assert len(maps["shuffled"]) == EXPECTED_N_PAIRS, len(maps["shuffled"])
    blocks = enumerate_blocks_2333(s1, s2, set())
    assert len(blocks) == EXPECTED_N_BLOCKS
    assert len(smoke_blocks_2333(s1, s2, set())) == 48
    print("[import-check] OK")


PHASES = {
    "envcheck": phase_envcheck,
    "fitness2329": phase_fitness2329,
    "bank": phase_bank,
    "donors": phase_donors,
    "grid": phase_grid,
    "anchors": phase_anchors,
    "ce_control": phase_ce_control,
    "upload": phase_upload,
}

# Phases that never LOAD the q35 model (or must run BEFORE the env-identity
# guard can exist): exempt from _require_q35_env.
_Q35_ENV_EXEMPT_PHASES = ("envcheck", "fitness2329", "upload")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return RC_OK
    assert args.phase, "--phase required (or --import-check)"
    cfg = build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    cfg.gates_dir.mkdir(parents=True, exist_ok=True)
    if cfg.model_tag == "q35" and not cfg.tiny and cfg.phase not in _Q35_ENV_EXEMPT_PHASES:
        _require_q35_env(cfg)  # env identity with the PASSing envcheck (plan §4.4 B-1)
    regime_fp = regime_fingerprint(cfg)
    random.seed(cfg.seed_base)
    logger.info(
        "[phase=%s] tag=%s smoke=%s tiny=%s regime_fp=%s out_root=%s",
        cfg.phase,
        cfg.model_tag,
        cfg.smoke,
        cfg.tiny,
        regime_fp,
        cfg.out_root,
    )
    rc = PHASES[cfg.phase](cfg, regime_fp)
    logger.info("[phase=%s] rc=%d", cfg.phase, rc)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
