#!/usr/bin/env python
# ruff: noqa: RUF002  # em-dash intentional
"""#1315 pod-side phase driver — impolite activation-shift geometry (plan v3).

Phases (linear, checkpoint-per-phase, resume-keyed; plan §4/§9):

  p0_stage     stage + sha/rev-pin every reused input (fu3 ICL mixes + conv mix
               for the WildChat byte-assert, fu4/fu3 adapters + adapter_config
               asserts, fu4 margin sha reference) + bank/panel/context asserts
  p1_train     imp_icl_ft_{neg,pos} full-FT (ZeRO-3 subprocess, byte-inherited
               #1112 recipe, 30-step ladder save-2)
  p2_ladder    Tier-1 judged-rate ladders for the FT cells (sharded 1/GPU);
               select_dose_checkpoint vs [0.60, 0.85]
  g1_gate      FT install viability: one-shot 60-step extension (plan §7)
  p3_persist_ft  upload SELECTED FT rungs to the overflow repo, THEN delete
               non-selected rungs (declared discard, plan §10)
  p4_parity    reused-adapter probes: Tier-1 judged read (±0.15 WARN-class) +
               adapter-application check (0.5-nat HALT floor) per plan §4.6
  p5_tier2     Tier-2 confirm for the FT cells (trained side; base reuses fu3)
  p6_margin    teacher-forced fixed-pool margin for the NEW FT cells (the fu4
               instrument; pool sha-asserted vs the committed fu4 record)
  p7_rb        impolite r_B (issue779 extractor subprocess, read-out regime,
               all 28 layers)
  p8_capture   own-text capture passes (gen + 28-layer 3-span TF pooling),
               sharded 4-wide per (cell, dose); base = union probe panel
  p9_capture_tf  shared-text response re-capture over the persisted BASE rows
               (mandatory — the CJK own-text confound control, plan §4.5)
  p10_geometry smoke-scale geometry stub (full geometry runs VM-side via
               scripts/issue1315_geometry.py)
  p11_upload   text/JSON + capture tensors + rb artifacts; sentinel

``--smoke`` is the SAME dispatcher with tiny knobs (plan § Dry-run smoke):
cell subset ("imp_icl_ft_neg",), 2 optimizer steps at PRODUCTION launch width
(4-way ZeRO-3, ``--num_processes 4`` / CVD 0-3 — width is smoke-INVARIANT;
the r3 fix: a 1-process smoke FT OOMs deterministically at the first
optimizer step because the fp32 Adam moments go unsharded) + 1 consolidated
ZeRO-3 save + vLLM-load canary (the Tier-1 rung generates through vLLM on the
saved rung), 1 Tier-1 rung at 2 questions, LIVE judge (sync fallback path), a
2-context × 2-question 3-arm 28-layer capture including ONE multi-turn
WildChat row (the new span logic end-to-end; ≥2 distinct question ids are
REQUIRED by the p10 split-half ceiling — crash-fix r4), geometry on the
captured stub,
recording-free upload via ``--no-upload``. Every phase reads its cell list
from the ONE resolver (``cfg.cells``), so the smoke subset threads through
train, ladder, tier2, margin, capture, capture_tf, geometry, and upload alike.
Reused-adapter phases (parity, the reused-cell captures) run in full mode
only — the smoke exercises their shared code paths through the FT cell.

Pod-side contract: NEVER shells out to scripts/task.py; progress = structured
``[phase=...]`` log lines + the end-of-run sentinel (pod-side-reporting.md).
``[phase=done]`` is emitted by the launcher wrapper ONLY, never here.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

# vLLM v1 EngineCore fork-poisoning guard (gotchas.md #628): must be set BEFORE
# any `import vllm` — the dispatcher touches tokenizers pre-LLM().
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1090_fu1 as fu1  # noqa: E402
import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_fu4 as fu4  # noqa: E402
import issue1090_run as i1090  # noqa: E402

# Generic machinery reused verbatim from the #1112 driver (same repos, same
# subprocess/env/CVD discipline — reuse hierarchy, CLAUDE.md):
from issue1112_dispatch import (  # noqa: E402
    _atomic_json,
    _ensure_dir_tokenizer,
    _enumerate_rungs,
    _merge_adapter,
    _phase,
    _physical_gpu_ids,
    _read_json,
    _reap_unit_groups,
    _run_subprocess,
    _stage_file,
    _stage_overflow_prefix,
)

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.negatives import (  # noqa: E402
    assert_panel_disjoint_from_sources,
    default_panel,
)
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    ModelOrganism,
    _default_margin_read_fn,
    _sha256_file,
    make_source_rate_fn,
)
from explore_persona_space.artifacts.recipe import select_dose_checkpoint  # noqa: E402
from explore_persona_space.experiments import issue_1315 as C  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1315")

ACCEL_CONFIG = "configs/accelerate/zero3_4gpu_accum1.yaml"
FT_TRAINER = "scripts/train_behavior_fullft.py"
RB_EXTRACTOR = "scripts/issue779_extract_rb.py"
FT_NUM_PROCESSES = 4  # ZeRO-3 world size (eff-batch contract; #1112 verbatim)

# Frozen fu3 mixes at the pinned revision (plan §10; shas computed from the
# pinned-revision downloads 2026-07-15 — mix_meta.json records INPUT shas, not
# the mix's own, so the mix bytes are pinned here directly).
FU3_MIX_SHAS = {
    "icl_con_train_mix.jsonl": "e88b32118b3fcedcfa2bf92d5b1be418f24c4b0ec38c68115efba964fdbecaee",
    "icl_pos_train_mix.jsonl": "a4dcc6bfc0175880a9e8dec9eb34d15e1878dcec7422e873dc1a2fe308046eab",
    "conv_con_train_mix.jsonl": "f80665fdf3a47ff5725e653e4ea34fcda2aab2ac4650542a5a0f01e00a44a04d",
}
FU4_LADDERS_JSON = (
    REPO_ROOT / "eval_results" / "issue_1090" / "fu4-extended-dose-lr" / ("fu4_ladders.json")
)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Cfg:
    smoke: bool
    cells: tuple[str, ...]
    out_root: Path
    seed: int = C.SEED
    tier1_n: int = 5
    tier1_draws: int = 3
    tier2_n: int = 10
    tier2_draws: int = 5
    eval_question_limit: int | None = None
    sentinel_dir: Path | None = None
    upload: bool = True
    phases: tuple[str, ...] = ()  # empty -> all
    # Conditional bare cell (plan §4.2): populated VM-side pre-launch iff fu5
    # unlocked a band-hit bare impolite organism; None -> cell omitted (a
    # stated scope note, never a blocker).
    bare_adapter_prefix: str | None = None
    bare_adapter_rev: str | None = None
    bare_committed_rate: float | None = None

    def regime_key(self) -> dict:
        return {
            "issue": C.ISSUE,
            "smoke": self.smoke,
            "cells": list(self.cells),
            "seed": self.seed,
            "tier1": [self.tier1_n, self.tier1_draws],
            "tier2": [self.tier2_n, self.tier2_draws],
            "eval_question_limit": self.eval_question_limit,
            "band": list(C.JUDGED_RATE_BAND),
            "save_steps": C.FT_SAVE_STEPS,
            "max_length": C.FT_MAX_LENGTH,
            "step_ceiling": C.FT_STEP_CEILING,
            "bare": self.bare_adapter_prefix,
        }


_PHASE_ALIASES = {
    "p0_stage": "stage",
    "p1_train": "train",
    "p2_ladder": "ladder",
    "g1_gate": "g1",
    "p3_persist_ft": "persist_ft",
    "p4_parity": "parity",
    "p5_tier2": "tier2",
    "p6_margin": "margin",
    "p7_rb": "rb",
    "p8_capture": "capture",
    "p9_capture_tf": "capture_tf",
    "p10_geometry": "geometry",
    "p11_upload": "upload",
}
_KNOWN_PHASES = frozenset(_PHASE_ALIASES.values())


def normalize_phases(raw: str | None) -> tuple[str, ...]:
    """Comma list of phase names -> canonical short-name tuple (fail-loud)."""
    if not raw:
        return ()
    out: list[str] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        t = _PHASE_ALIASES.get(t, t)
        if t not in _KNOWN_PHASES:
            raise ValueError(
                f"unknown phase {tok.strip()!r}: want one of {sorted(_KNOWN_PHASES)} "
                "(pN_-prefixed aliases accepted)"
            )
        out.append(t)
    return tuple(out)


def resolve_cells(cells_arg: str | None, smoke: bool, *, bare: bool = False) -> tuple[str, ...]:
    """The ONE cell resolver every phase consumes (smoke = same path, 1 cell)."""
    known = set(C.REUSED_LORA_CELLS) | set(C.FT_CELLS) | {C.CONDITIONAL_BARE_CELL}
    if cells_arg:
        ids = tuple(t.strip() for t in cells_arg.split(","))
        bad = [t for t in ids if t not in known]
        if bad:
            raise ValueError(f"bad cells {bad!r}: want a subset of {sorted(known)}")
        return ids
    if smoke:
        return ("imp_icl_ft_neg",)  # the plan § Dry-run smoke cell (FT + negatives)
    cells = (*C.REUSED_LORA_CELLS, *C.FT_CELLS)
    if bare:
        cells = (*cells, C.CONDITIONAL_BARE_CELL)
    return cells


def _n_gpus() -> int:
    return len(_physical_gpu_ids())


# ── Contexts (plan §4.2/§4.3 — resolver: the fu3 worker's ensure_context) ────


def _context(context_id: str):
    """Resolve + register a #1315 capture/training context (idempotent)."""
    if context_id == "default":
        from explore_persona_space.artifacts.context import CONTEXTS

        return CONTEXTS["default"]
    return fu3w.ensure_context(context_id, C.BEHAVIOR)


def _cell_context_id(cfg: Cfg, cell: str) -> str:
    if cell in C.REUSED_LORA_CELLS:
        return C.REUSED_LORA_CELLS[cell]["context_id"]
    if cell in C.FT_CELLS:
        return C.FT_CELLS[cell]["context_id"]
    if cell == C.CONDITIONAL_BARE_CELL:
        return "default"
    raise ValueError(f"unroutable cell {cell!r}")


def _eval_questions(cfg: Cfg) -> list[str]:
    qs = list(BEHAVIORS[C.BEHAVIOR].eval_question_bank)
    if cfg.eval_question_limit is not None:
        qs = qs[: cfg.eval_question_limit]
    return qs


# ── p0: stage inputs (pinned) + context-builder byte-asserts ─────────────────


def _stage_model_prefix(prefix: str, dest: Path, *, revision: str) -> Path:
    """Stage a checkpoint subfolder from the CANONICAL model repo at a pinned
    revision (scoped list_repo_tree + per-file download; no staging transform —
    files land at prefix-relative paths, reuse check (h)(iv) N/A)."""
    from huggingface_hub import HfApi, hf_hub_download

    if (dest / "adapter_config.json").exists() or (dest / "config.json").exists():
        return dest
    api = HfApi()
    # retried scoped listing (hub helper) — a bare list_repo_tree is the #920
    # false-failure class (workflow_lint --check-hub-verify-retry)
    entries = hub.list_hf_files_under_path(
        api, C.MODEL_REPO, prefix, repo_type="model", revision=revision
    )
    if not entries:
        raise FileNotFoundError(f"no files under {C.MODEL_REPO}/{prefix} @ {revision}")
    dest.mkdir(parents=True, exist_ok=True)
    for p in entries:
        got = hf_hub_download(C.MODEL_REPO, p, repo_type="model", revision=revision)
        target = dest / Path(p).relative_to(prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(got, target)
    return dest


def _assert_adapter_config(ckpt: Path, cell: str) -> dict:
    """Reuse fitness (a): recipe grounded on the artifact's OWN adapter_config
    (r=32/α=64/dropout 0.05/rsLoRA — the #1090 UNIFIED_OVERRIDES row; #545:
    the config wins over body rows)."""
    acfg = _read_json(ckpt / "adapter_config.json")
    ok = (
        acfg.get("r") == 32
        and acfg.get("lora_alpha") == 64
        and acfg.get("use_rslora") is True
        and acfg.get("lora_dropout") == 0.05
    )
    if not ok:
        raise RuntimeError(f"{cell}: staged adapter_config diverges from UNIFIED_OVERRIDES: {acfg}")
    return acfg


def extract_wildchat_prefix_from_mix(mix_path: Path) -> list[dict]:
    """The WildChat two-turn prefix RE-DERIVED from the frozen fu3 conv-con mix
    rows, byte-asserted identical across every (user, assistant, user)-shaped
    positive row (plan §4.1(iii) / assumption 7). Returns the 2-turn prefix."""
    prefixes: set[str] = set()
    n_pos = 0
    with mix_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or []
            roles = tuple(m.get("role") for m in prompt)
            if roles == ("user", "assistant", "user"):
                n_pos += 1
                prefixes.add(
                    json.dumps(
                        [{"role": m["role"], "content": m["content"]} for m in prompt[:2]],
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                )
    if n_pos == 0 or len(prefixes) != 1:
        raise RuntimeError(
            f"WildChat prefix byte-assert FAILED on {mix_path}: {n_pos} positive rows, "
            f"{len(prefixes)} distinct 2-turn prefixes (want exactly 1 across all rows) — "
            "fallback: read the prefix from fu3's run_config.json (plan assumption 7)"
        )
    return json.loads(next(iter(prefixes)))


def assert_wildchat_context_matches_mix(mix_path: Path) -> None:
    """Capture context == training context: the registered committed-battery
    WildChat prefix must byte-match the prefix inside the frozen mix rows."""
    mix_prefix = extract_wildchat_prefix_from_mix(mix_path)
    reg = [dict(t) for t in _context(C.CONV_CONTEXT_ID).prefix_turns]
    if reg != mix_prefix:
        raise RuntimeError(
            "WildChat context byte-assert FAILED: the registered "
            f"{C.CONV_CONTEXT_ID} prefix_turns differ from the fu3 conv-con mix rows'"
        )


def assert_icl_block_matches_mix(mix_path: Path) -> None:
    """The ICL two-shot block (derived from icl_examples_impolite.json) must be
    a byte-prefix of every ICL-context row in the frozen fu3 icl-con mix
    (plan §4.1(iii), the Methodology-critic addition)."""
    ctx = _context(C.ICL_CONTEXT_ID)
    assert ctx.user_wrap and ctx.user_wrap.endswith("\n\n{q}"), ctx.user_wrap
    block = ctx.user_wrap.replace("{{", "{").replace("}}", "}").removesuffix("\n\n{q}")
    n_icl = n_match = 0
    with mix_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or []
            if len(prompt) == 1 and "Example question:" in prompt[0].get("content", ""):
                n_icl += 1
                if prompt[0]["content"].startswith(block + "\n\n"):
                    n_match += 1
    if n_icl == 0 or n_match != n_icl:
        raise RuntimeError(
            f"ICL block byte-assert FAILED on {mix_path}: {n_match}/{n_icl} ICL rows "
            "start with the derived two-shot block"
        )


def _extraction_questions() -> list[str]:
    """The plan §4.5 impolite 20-question EXTRACTION set: the bank\'s 0-20
    slice (``bank_slice("impolite", "train")`` — the #1090 datagen slice;
    impolite registers NO "extraction" slice, banks.py:228), disjoint from the
    20-question eval slice (rows 20-40) by construction."""
    from explore_persona_space.artifacts.banks import bank_slice

    return list(bank_slice(C.BEHAVIOR, "train"))


def _assert_banks_and_panel() -> None:
    """impolite_neutral_v1 20/20 disjoint banks + panel ∩ sources == ∅ (hard
    assert via assert_panel_disjoint_from_sources; plan §4.3)."""
    b = BEHAVIORS[C.BEHAVIOR]
    ex = b.extraction
    assert ex is not None and b.judge_rubric, "impolite registry entry is a stub"
    assert len(ex.prompt_pairs) == 5, len(ex.prompt_pairs)
    eval_qs = list(b.eval_question_bank)
    ext_qs = _extraction_questions()
    assert len(eval_qs) == 20 and len(ext_qs) == 20, (len(eval_qs), len(ext_qs))
    overlap = set(eval_qs) & set(ext_qs)
    assert not overlap, f"extraction/eval question overlap: {sorted(overlap)[:3]}"
    assert_panel_disjoint_from_sources(
        default_panel(),
        [C.PERS_CONTEXT_ID],
        source_identities={C.PERS_CONTEXT_ID: C.SOURCE_PERSONA},
    )


def _staged_reused_ckpt(cfg: Cfg, cell: str, step: int) -> Path:
    return cfg.out_root / "inputs" / cell / f"checkpoint-{step}"


def phase_stage(cfg: Cfg) -> dict:
    _phase("p0_stage")
    inputs = cfg.out_root / "inputs"
    done_path = cfg.out_root / "p0_stage.json"
    if done_path.exists():
        return _read_json(done_path)
    rec: dict = {"staged": {}}
    _assert_banks_and_panel()

    # Frozen fu3 mixes @ pinned revision + sha-asserts (reuse check (f)).
    for path_in_repo, name in (
        (C.FU3_MIX_CON_PATH, "icl_con_train_mix.jsonl"),
        (C.FU3_MIX_POS_PATH, "icl_pos_train_mix.jsonl"),
        ("issue1090_fu3/C2-conv-con-impolite-claude/train_mix.jsonl", "conv_con_train_mix.jsonl"),
    ):
        _stage_file(path_in_repo, inputs / name, revision=C.FU3_MIX_REV, sha256=FU3_MIX_SHAS[name])
    _stage_file(C.FU3_MIX_CON_META, inputs / "icl_con_mix_meta.json", revision=C.FU3_MIX_REV)
    _stage_file(C.FU3_MIX_POS_META, inputs / "icl_pos_mix_meta.json", revision=C.FU3_MIX_REV)
    rec["staged"]["mixes"] = str(inputs)
    rec["mix_shas"] = dict(FU3_MIX_SHAS)

    # Context-builder byte-asserts (plan §4.1(iii)): capture contexts must be
    # byte-identical to the training contexts inside the frozen mixes.
    assert_wildchat_context_matches_mix(inputs / "conv_con_train_mix.jsonl")
    assert_icl_block_matches_mix(inputs / "icl_con_train_mix.jsonl")
    rec["context_byte_asserts"] = "PASS (wildchat prefix + icl block)"

    # Reused adapters @ pinned revisions + adapter_config asserts (checks a/e/f).
    for cell in C.REUSED_LORA_CELLS:
        if cell not in cfg.cells:
            continue
        spec = C.REUSED_LORA_CELLS[cell]
        for step in sorted(set(spec["doses"].values())):
            dest = _staged_reused_ckpt(cfg, cell, step)
            if spec["repo"] == C.OVERFLOW_REPO:
                _stage_overflow_prefix(
                    f"{spec['prefix']}/checkpoint-{step}", dest, revision=spec["revision"]
                )
            else:
                _stage_model_prefix(
                    f"{spec['prefix']}/checkpoint-{step}", dest, revision=spec["revision"]
                )
            _assert_adapter_config(dest, cell)
        rec["staged"][cell] = str(inputs / cell)
    if C.CONDITIONAL_BARE_CELL in cfg.cells:
        assert cfg.bare_adapter_prefix and cfg.bare_adapter_rev, (
            "imp_bare_lora is in the cell set but no --bare-adapter-prefix/--bare-adapter-rev "
            "was provided (the VM-side pre-launch fu5 read populates these; plan §4.2)"
        )
        dest = _staged_reused_ckpt(cfg, C.CONDITIONAL_BARE_CELL, 0)
        _stage_overflow_prefix(cfg.bare_adapter_prefix, dest, revision=cfg.bare_adapter_rev)
        _assert_adapter_config(dest, C.CONDITIONAL_BARE_CELL)
        rec["staged"][C.CONDITIONAL_BARE_CELL] = str(dest)
    if cfg.smoke and not any(c in C.REUSED_LORA_CELLS for c in cfg.cells):
        # plan § Dry-run smoke: "stages 1 reused adapter + 1 mix (sha-assert)" —
        # the smoke cell is an FT cell, so stage ONE reused adapter as the
        # staging-path canary (scoped list_repo_tree + adapter_config assert).
        canary = "imp_icl_lora_neg"
        spec = C.REUSED_LORA_CELLS[canary]
        step = spec["doses"]["selected"]
        dest = _staged_reused_ckpt(cfg, canary, step)
        _stage_overflow_prefix(
            f"{spec['prefix']}/checkpoint-{step}", dest, revision=spec["revision"]
        )
        _assert_adapter_config(dest, canary)
        rec["staged"]["smoke_reused_adapter_canary"] = str(dest)

    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_json(done_path, rec)
    return rec


def _mix_path(cfg: Cfg, cell: str) -> Path:
    return cfg.out_root / "inputs" / C.FT_CELLS[cell]["mix"]


# ── p1: FT training (byte-inherited #1112 recipe; plan §4.4) ─────────────────


def _ft_num_processes(cfg: Cfg) -> int:
    """ZeRO-3 world size — pinned 4 in BOTH modes (smoke-invariant; fails loud
    under-provisioned — the #1112 round-4 unsharded-OOM guard). The r3 crash:
    a cloned-in smoke branch returned 1, so ``accelerate launch
    --num_processes 1`` put the whole 7B (bf16 weights ~15 GB + grads ~15 GB +
    UNSHARDED fp32 Adam moments ~56 GB) on one A100-80 — deterministic
    torch.OutOfMemoryError at the FIRST optimizer step. 4-way ZeRO-3 shards
    optimizer+grads ~/4 and fits (plan §4.4); PASS_UNIFIED means the smoke
    keeps the production PROCESS SHAPE too (#397 resource-dimension
    divergence class). The parent #1112 smoke was 1-wide only because ITS
    smoke instance was a 1-GPU GCE a2-ultragpu-1g; the #1315 smoke runs on
    the 4x A100-80 ft-7b pod, so no narrow lane is needed or allowed."""
    del cfg  # FT launch width is deliberately mode-independent (r3 fix)
    n_phys = len(_physical_gpu_ids())
    if n_phys < FT_NUM_PROCESSES:
        raise RuntimeError(
            f"full-FT needs {FT_NUM_PROCESSES} GPUs (ZeRO-3 world size / eff-batch "
            f"contract) but only {n_phys} physical GPUs are visible"
        )
    return FT_NUM_PROCESSES


def _ft_env(cfg: Cfg) -> dict[str, str]:
    """EXPLICIT CVD over the physical GPUs (the in-process train_lora clobber
    class — gotchas.md; #1112 shape b)."""
    ids = _physical_gpu_ids()
    return {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(ids[: _ft_num_processes(cfg)])}


def _run_ft_subprocess(cfg: Cfg, cmd: list[str], log_path: Path) -> None:
    env = _ft_env(cfg)
    npr = cmd[cmd.index("--num_processes") + 1]
    logger.info(
        "[ft-launch] num_processes=%s CUDA_VISIBLE_DEVICES=%s cmd=%s",
        npr,
        env["CUDA_VISIBLE_DEVICES"],
        " ".join(cmd[:10]) + " ...",
    )
    _run_subprocess(cmd, log_path, env=env)


def _ft_cmd(
    cfg: Cfg, cell: str, *, out_dir: Path, max_steps: int, ckpt_steps: Sequence[int]
) -> list[str]:
    """train_behavior_fullft.py launch — every value byte-inherited from #1112
    (Source: #1112 §11 / #606/#642; constants imported, never retyped)."""
    return [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--config_file",
        ACCEL_CONFIG,
        "--num_processes",
        str(_ft_num_processes(cfg)),
        FT_TRAINER,
        "--behavior",
        C.BEHAVIOR,
        "--arm",
        "ft",
        "--train-jsonl",
        str(_mix_path(cfg, cell)),
        "--output-dir",
        str(out_dir),
        "--ckpt-steps",
        ",".join(str(s) for s in ckpt_steps),
        "--max-steps",
        str(max_steps),
        "--learning-rate",
        str(C.FT_LR),
        "--epochs",
        "16",  # ceiling; --max-steps caps (the #1112 seam)
        "--per-device-batch",
        str(C.FT_PER_DEVICE_BATCH),
        "--grad-accum",
        str(C.FT_GRAD_ACCUM),
        "--warmup-ratio",
        str(C.FT_WARMUP_RATIO),
        "--max-length",
        str(C.FT_MAX_LENGTH),
        "--seed",
        str(cfg.seed),
        "--wandb-project",
        C.WANDB_PROJECT,
        # Per-cell suffix: the trainer's base run name is
        # issue642_<arm>_<behavior>_seed<seed>, IDENTICAL for both FT cells —
        # a shared suffix would collapse them into one WandB run name (the
        # #480 per-source run-separation regression class).
        "--run-name-suffix",
        f"i1315_{cell}",
    ]


def ft_wandb_run_name(cell: str, seed: int) -> str:
    """The trainer's realized run name (train_behavior_fullft.py:528-530)."""
    return f"issue642_ft_{C.BEHAVIOR}_seed{seed}_i1315_{cell}"


def _fresh_ft_out_dir(out_dir: Path) -> None:
    """Wipe a stale PARTIAL FT tree before a fresh launch (the done-sentinel is
    absent when we get here, so anything present is incomplete — #1112 r4)."""
    if out_dir.exists():
        logger.warning("[ft-launch] clearing stale partial FT out_dir %s", out_dir)
        shutil.rmtree(out_dir)


def phase_train(cfg: Cfg) -> dict:
    _phase("p1_train")
    results: dict[str, dict] = {}
    for cell in cfg.cells:
        if cell not in C.FT_CELLS:
            continue  # reused cells are never trained here
        cell_root = cfg.out_root / cell
        build_path = cell_root / "build_result.json"
        if build_path.exists():
            results[cell] = _read_json(build_path)
            continue
        out_dir = cell_root / "train"
        max_steps = 2 if cfg.smoke else C.FT_STEP_CEILING
        ckpts = (2,) if cfg.smoke else C.FT_CKPT_STEPS
        _fresh_ft_out_dir(out_dir)
        _run_ft_subprocess(
            cfg,
            _ft_cmd(cfg, cell, out_dir=out_dir, max_steps=max_steps, ckpt_steps=ckpts),
            cell_root / "train.log",
        )
        rec = {
            "cell": cell,
            "status": "trained",
            "adapter_root": str(out_dir),
            "mix": str(_mix_path(cfg, cell)),
            "mix_sha256": _sha256_file(_mix_path(cfg, cell)),
        }
        _atomic_json(build_path, rec)
        results[cell] = rec
    return results


# ── p2: Tier-1 ladders + selection (FT cells; sharded 1 cell/GPU) ────────────


def run_ladder_unit(cfg: Cfg, cell: str) -> dict[int, float]:
    """Tier-1 judged rate at every rung of one FT cell (the fu4 instrument:
    make_source_rate_fn + the max_tokens=300 judge). Per-rung resume."""
    cell_root = cfg.out_root / cell
    ladder_path = cell_root / "ladder.json"
    ckpts = _enumerate_rungs(_read_json(cell_root / "build_result.json")["adapter_root"])
    done: dict[int, float] = {}
    if ladder_path.exists():
        prior = _read_json(ladder_path)
        if prior.get("regime") != cfg.regime_key():
            raise RuntimeError(f"ladder regime drift under {ladder_path} — fresh --out-root")
        done = {int(k): float(v) for k, v in (prior.get("rates_by_step") or {}).items()}

    def _persist() -> None:
        _atomic_json(
            ladder_path,
            {
                "cell": cell,
                "regime": cfg.regime_key(),
                "rates_by_step": {str(k): v for k, v in sorted(done.items())},
                "judge_max_tokens": fu1.JUDGE_MAX_TOKENS_FU1,
            },
        )

    pending = [s for s in sorted(ckpts) if s not in done]
    if pending:
        _context(_cell_context_id(cfg, cell))  # register before organism validation
        organism = ModelOrganism(
            behavior=C.BEHAVIOR, context_id=_cell_context_id(cfg, cell), seed=cfg.seed
        )
        rate_fn = make_source_rate_fn(
            organism,
            out_dir=cell_root / "rate",
            eval_questions=_eval_questions(cfg),
            n_completions=cfg.tier1_n,
            temperature=1.0,
            n_judge_draws=cfg.tier1_draws,
            judge_fn=fu1._judge_fu1,
        )
        try:
            for step in pending:
                # FT rung dirs may lack tokenizer files (#1112 r6) — repair
                # before vLLM/tokenizer loads from this path.
                _ensure_dir_tokenizer(ckpts[step])
                done[step] = float(rate_fn(str(ckpts[step])))
                _persist()
        finally:
            close = getattr(rate_fn, "close", None)
            if callable(close):
                close()
    else:
        _persist()
    return done


def _fanout_units(cfg: Cfg, units: list[list[str]]) -> None:
    """Work-conserving CVD-pinned subprocess pool over self-invocation units
    (the #1112 pattern: launcher-env CVD pin + matching --gpu-id; whole-group
    reap on failure). 1-GPU units ONLY — FT launches never route here."""
    import subprocess

    ids = _physical_gpu_ids()
    n = len(ids)
    pending = list(units)
    running: dict[int, tuple[subprocess.Popen, list[str]]] = {}
    logs = cfg.out_root / "unit_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for g in range(n):
            if g not in running and pending:
                extra = pending.pop(0)
                cmd = [
                    "uv",
                    "run",
                    "python",
                    str(_SCRIPTS_DIR / "issue1315_dispatch.py"),
                    *extra,
                    "--gpu-id",
                    ids[g],
                ]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ids[g]}
                log = logs / f"unit_{'_'.join(extra[1:3]).replace('/', '_')}_g{g}.log"
                f = open(log, "a")  # noqa: SIM115 — held open for the Popen's lifetime
                running[g] = (
                    subprocess.Popen(
                        cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True
                    ),
                    extra,
                )
                logger.info("[fanout] gpu %d <- %s (log %s)", g, extra, log)
        time.sleep(10)
        for g, (proc, extra) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[g]
            if rc != 0:
                _reap_unit_groups([p2 for p2, _ in running.values()])
                raise RuntimeError(f"fanout unit {extra} failed rc={rc} (see {logs})")


def _unit_args(cfg: Cfg, kind: str, arg: str) -> list[str]:
    return (
        [
            "--unit",
            kind,
            arg,
            "--smoke" if cfg.smoke else "--full",
            "--out-root",
            str(cfg.out_root),
            "--cells",
            ",".join(cfg.cells),
        ]
        + (
            ["--eval-question-limit", str(cfg.eval_question_limit)]
            if cfg.eval_question_limit
            else []
        )
        + ([] if cfg.upload else ["--no-upload"])
    )


def phase_ladder(cfg: Cfg) -> dict:
    _phase("p2_ladder")
    cells = [c for c in cfg.cells if c in C.FT_CELLS]
    # Pending predicate keys on selection.json ONLY (the parent shape,
    # issue1112_dispatch.py:900): a cell with a PARTIAL ladder.json re-enters
    # run_ladder_unit, which resumes per-rung and no-ops when complete. Keying
    # on ladder.json too left a mixed crash-resume state (cell A partial +
    # cell B fresh) selecting on incomplete rates (round-1 code-review Major 1).
    units = [
        _unit_args(cfg, "ladder", c)
        for c in cells
        if not (cfg.out_root / c / "selection.json").exists()
    ]
    if units:
        if len(units) == 1 or _n_gpus() == 1:
            for u in units:
                run_ladder_unit(cfg, u[2])
        else:
            _fanout_units(cfg, units)
    selections: dict[str, dict] = {}
    for cell in cells:
        sel_path = cfg.out_root / cell / "selection.json"
        if sel_path.exists():
            selections[cell] = _read_json(sel_path)
            continue
        rates = {
            int(k): float(v)
            for k, v in _read_json(cfg.out_root / cell / "ladder.json")["rates_by_step"].items()
        }
        sel = select_dose_checkpoint(rates, band=C.JUDGED_RATE_BAND)
        rec = {
            **dataclasses.asdict(sel),
            "rates_by_step": {str(k): v for k, v in sorted(rates.items())},
            "band": list(C.JUDGED_RATE_BAND),
        }
        _atomic_json(sel_path, rec)
        selections[cell] = rec
    # Reused cells: committed selections (plan §4.2 — nothing retrained).
    for cell, spec in C.REUSED_LORA_CELLS.items():
        if cell not in cfg.cells:
            continue
        rec = {
            "step": spec["doses"]["selected"],
            "rate": spec["tier2_committed"],
            "in_band": True,
            "fallback": None,
            "reused": True,
        }
        _atomic_json(cfg.out_root / cell / "selection.json", rec)
        selections[cell] = rec
    if C.CONDITIONAL_BARE_CELL in cfg.cells:
        rec = {
            "step": 0,
            "rate": cfg.bare_committed_rate,
            "in_band": True,
            "fallback": None,
            "reused": True,
        }
        _atomic_json(cfg.out_root / C.CONDITIONAL_BARE_CELL / "selection.json", rec)
        selections[C.CONDITIONAL_BARE_CELL] = rec
    return selections


# ── G1 gate (plan §7): one-shot 60-step extension per below-band FT cell ────


def phase_g1_gate(cfg: Cfg, selections: dict) -> dict:
    _phase("g1_gate")
    if cfg.smoke:
        return {"action": "none", "note": "smoke — 2-step train, gate not applicable"}
    rec_path = cfg.out_root / "g1_gate.json"
    if rec_path.exists():
        return _read_json(rec_path)
    out: dict = {"action": "none", "cells": {}}
    for cell in cfg.cells:
        if cell not in C.FT_CELLS:
            continue
        sel = selections.get(cell) or {}
        if sel.get("in_band"):
            out["cells"][cell] = {"in_band_at": int(sel["step"])}
            continue
        cell_root = cfg.out_root / cell
        ext_done = cell_root / "g1_extended.json"
        if ext_done.exists():
            out["cells"][cell] = _read_json(ext_done)
            continue
        out["action"] = "extend_in_place"
        # Delete-before-extend (plan §9): keep only the LATEST rung (resume
        # source); every ruled-out step<=30 rung is already judged + its rates
        # persisted in ladder.json (declared discard).
        train_dir = Path(_read_json(cell_root / "build_result.json")["adapter_root"])
        rungs = _enumerate_rungs(train_dir)
        latest = max(rungs)
        for step, p in rungs.items():
            if step != latest:
                shutil.rmtree(p, ignore_errors=True)
        _run_ft_subprocess(
            cfg,
            _ft_cmd(
                cfg,
                cell,
                out_dir=train_dir,
                max_steps=C.G1_EXT_CEILING,
                ckpt_steps=tuple(range(32, C.G1_EXT_CEILING + 1, 2)),
            ),
            cell_root / "g1_extend.log",
        )
        # Fresh ladder over the extension rungs: drop the selection so
        # phase_ladder re-runs + re-selects (ladder.json keeps <=30 rates).
        (cell_root / "selection.json").unlink(missing_ok=True)
        ext_rec = {"extended_to": C.G1_EXT_CEILING, "ts": time.time()}
        _atomic_json(ext_done, ext_rec)
        out["cells"][cell] = ext_rec
    _atomic_json(rec_path, out)
    return out


def g1_closest_approach(cfg: Cfg, selections: dict) -> dict:
    """Post-extension disposition (plan §7): in-band | labeled below-band
    (closest-approach >= 0.45, confound named) | unmeasured (< 0.45)."""
    out: dict = {}
    for cell in cfg.cells:
        if cell not in C.FT_CELLS:
            continue
        sel = selections.get(cell) or {}
        if sel.get("in_band"):
            out[cell] = {"disposition": "in_band", "step": int(sel["step"])}
            continue
        rates = {int(k): float(v) for k, v in (sel.get("rates_by_step") or {}).items()}
        if not rates:
            out[cell] = {"disposition": "unmeasured", "reason": "no ladder rates"}
            continue
        lo, hi = C.JUDGED_RATE_BAND
        step, rate = min(rates.items(), key=lambda kv: abs(kv[1] - (lo + hi) / 2))
        out[cell] = {
            "disposition": "labeled_below_band" if rate >= 0.45 else "unmeasured",
            "closest_step": step,
            "closest_rate": rate,
            "note": "install confound named on H4-H6" if rate >= 0.45 else "arm unmeasured (§Kill)",
        }
    return out


# ── p3: persist selected FT rungs, then reap the ladder (plan §9/§10) ────────


def phase_persist_ft(cfg: Cfg, selections: dict) -> dict:
    _phase("p3_persist_ft")
    rec_path = cfg.out_root / "p3_persist_ft.json"
    if rec_path.exists():
        return _read_json(rec_path)
    uploaded: dict[str, str] = {}
    for cell in cfg.cells:
        if cell not in C.FT_CELLS or cell not in selections:
            continue
        cell_root = cfg.out_root / cell
        train_dir = Path(_read_json(cell_root / "build_result.json")["adapter_root"])
        rungs = _enumerate_rungs(train_dir)
        step = int(selections[cell]["step"])
        keep = {step, max(rungs)}
        if cfg.upload:
            for s in sorted(keep):
                _ensure_dir_tokenizer(rungs[s])
                repo_path = f"issue1315/{cell}/checkpoint-{s}"
                url = hub._upload(rungs[s], C.OVERFLOW_REPO, "model", repo_path, private=True)
                if not str(url):
                    raise RuntimeError(f"FT persist upload returned no path for {repo_path}")
                uploaded[f"overflow:{repo_path}"] = str(url)
        if not cfg.smoke:
            # Declared discard (plan §10 discarded_artifacts): non-selected
            # rungs deleted AFTER the kept-rung uploads verified above.
            for s, p in rungs.items():
                if s not in keep:
                    shutil.rmtree(p, ignore_errors=True)
    _atomic_json(rec_path, {"uploaded": uploaded})
    return {"uploaded": uploaded}


# ── p4: reused-adapter parity probes (plan §4.6; gate calibration §11) ───────


def _fu4_committed_margin(cell: str) -> dict | None:
    """The committed fu4 margin record for a reused fu4 cell (same-surface
    references for the application HALT floor + pool sha)."""
    run_id = {"imp_pers_lora": "imp-pers-lr3e5", "imp_conv_lora": "imp-conv-lr3e5"}.get(cell)
    if run_id is None or not FU4_LADDERS_JSON.exists():
        return None
    return _read_json(FU4_LADDERS_JSON)["runs"][run_id]["margin"]


def _margin_pools(cfg: Cfg) -> tuple[list[dict], list[dict], dict]:
    """The fu4 impolite instrument verbatim: fixed 23/25 pools staged from the
    #1090 c2 datagen artifacts, sha-asserted against the committed fu4 record
    (reuse check (f) — a drifted instrument refuses to read)."""
    rc = i1090.RunConfig(smoke=cfg.smoke, cells=(), out_root=cfg.out_root)
    pos, neg, meta = fu4.fu4_margin_pools(rc, C.BEHAVIOR)
    assert pos is not None and neg is not None, meta
    committed = _fu4_committed_margin("imp_pers_lora")
    if committed is not None and meta["pool_sha256"] != committed["pool_sha256"]:
        raise RuntimeError(
            f"margin pool sha mismatch: derived {meta['pool_sha256']} != committed fu4 "
            f"{committed['pool_sha256']} — refusing a drifted-instrument margin read"
        )
    return pos, neg, meta


def run_parity_unit(cfg: Cfg, cell: str) -> dict:
    """One reused-adapter probe: (1) adapter-application check — max abs
    positive-pool per-token LN-logP change vs base on the cell's own source
    context, HALT floor 0.5 nat (structural apply-path, HALT-class); (2)
    Tier-1-style judged read within ±0.15 of the committed Tier-2 (WARN-class:
    persisted + adjudicated, retrain fallback is the orchestrator's)."""
    spec = C.REUSED_LORA_CELLS[cell]
    cell_root = cfg.out_root / cell
    out_path = cell_root / "parity.json"
    if out_path.exists():
        return _read_json(out_path)
    step = spec["doses"]["selected"]
    ckpt = _staged_reused_ckpt(cfg, cell, step)
    ctx = _context(spec["context_id"])
    pos, neg, meta = _margin_pools(cfg)

    # (1) HALT-class application check (fu1 instrument, one source_ctx read).
    margin_fn = _default_margin_read_fn(DEFAULT_BASE_MODEL)
    try:
        base_read = dataclasses.asdict(margin_fn(None, ctx, pos, neg))
        trained_read = dataclasses.asdict(margin_fn(str(ckpt), ctx, pos, neg))
    finally:
        close = getattr(margin_fn, "close", None)
        if callable(close):
            close()
    apply_rec = fu1.assert_adapter_applied(
        base_read, trained_read, tol=C.APPLY_HALT_FLOOR_NATS, tag=f"parity:{cell}"
    )

    # (2) WARN-class judged-rate window (the fu4 Tier-1 instrument).
    organism = ModelOrganism(behavior=C.BEHAVIOR, context_id=spec["context_id"], seed=cfg.seed)
    rate_fn = make_source_rate_fn(
        organism,
        out_dir=cell_root / "rate",
        eval_questions=_eval_questions(cfg),
        n_completions=cfg.tier1_n,
        temperature=1.0,
        n_judge_draws=cfg.tier1_draws,
        judge_fn=fu1._judge_fu1,
    )
    try:
        merged = _merge_adapter(cfg, str(ckpt), cell_root / "merged_parity")
        rate = float(rate_fn(str(merged)))
    finally:
        shutil.rmtree(cell_root / "merged_parity", ignore_errors=True)
        close = getattr(rate_fn, "close", None)
        if callable(close):
            close()
    window_ok = abs(rate - spec["tier2_committed"]) <= C.PARITY_RATE_TOL
    rec = {
        "cell": cell,
        "checkpoint": str(ckpt),
        "rate": rate,
        "expected": spec["tier2_committed"],
        "tol": C.PARITY_RATE_TOL,
        "rate_window_pass": window_ok,
        "severity": "WARN" if not window_ok else "PASS",
        "adapter_assert": apply_rec,
        "apply_halt_floor_nats": C.APPLY_HALT_FLOOR_NATS,
        "engaged_nats_committed": spec["engaged_nats_committed"],
        "pool_sha256": meta["pool_sha256"],
        "adjudication": (
            None
            if window_ok
            else "WARN-class parity miss persisted for analyzer adjudication; the plan §4.6 "
            "retrain-from-frozen-mix fallback is orchestrator-owned (never silent substitution)"
        ),
    }
    _atomic_json(out_path, rec)
    if not window_ok:
        logger.warning(
            "[parity] %s judged rate %.3f outside %.3f±%.2f (WARN-class, persisted)",
            cell,
            rate,
            spec["tier2_committed"],
            C.PARITY_RATE_TOL,
        )
    return rec


def phase_parity(cfg: Cfg) -> dict:
    _phase("p4_parity")
    cells = [c for c in cfg.cells if c in C.REUSED_LORA_CELLS]
    if not cells:
        return {"skipped": "no reused cells in this run"}
    pending = [c for c in cells if not (cfg.out_root / c / "parity.json").exists()]
    if pending:
        # r5 crash-fix: stage the SHARED fixed margin pools ONCE in the parent
        # BEFORE any fanout — the r4 run's 4 concurrent units each raced
        # _stage_hf_prefix into the same margin_pools/impolite dest and the
        # gpu-3 unit crashed at os.replace (FileNotFoundError; epm:failure v4).
        # Units re-call _margin_pools idempotently: the strengthened
        # fu3w._margin_pool_source_staged guard sees the complete dest and
        # skips staging, so no unit ever writes the shared dest. This also
        # fails the pool-sha reuse gate BEFORE burning GPU unit launches.
        _margin_pools(cfg)
        logger.info(
            "[margin-pools] pre-staged once in the parent for %d parity unit(s) (r5 fix)",
            len(pending),
        )
        if len(pending) == 1 or _n_gpus() == 1:
            for c in pending:
                run_parity_unit(cfg, c)
        else:
            _fanout_units(cfg, [_unit_args(cfg, "parity", c) for c in pending])
    return {c: _read_json(cfg.out_root / c / "parity.json") for c in cells}


# ── p5: Tier-2 confirm (FT cells) ────────────────────────────────────────────


def _selected_ckpt(cfg: Cfg, cell: str, selections: dict) -> Path:
    if cell in C.REUSED_LORA_CELLS:
        return _staged_reused_ckpt(cfg, cell, C.REUSED_LORA_CELLS[cell]["doses"]["selected"])
    if cell == C.CONDITIONAL_BARE_CELL:
        return _staged_reused_ckpt(cfg, cell, 0)
    step = int(selections[cell]["step"])
    return _enumerate_rungs(_read_json(cfg.out_root / cell / "build_result.json")["adapter_root"])[
        step
    ]


def phase_tier2(cfg: Cfg, selections: dict) -> dict:
    _phase("p5_tier2")
    out: dict[str, dict] = {}
    for cell in cfg.cells:
        if cell not in C.FT_CELLS or cell not in selections:
            continue
        cell_root = cfg.out_root / cell
        res_path = cell_root / "tier2.json"
        if res_path.exists():
            out[cell] = _read_json(res_path)
            continue
        context_id = _cell_context_id(cfg, cell)
        # r6 crash fix (epm:failure v5): register the fu3-lineage context BEFORE
        # organism validation. ModelOrganism.__post_init__ validates context_id
        # against the central CONTEXTS registry, and 'icl_prefix_impolite' only
        # enters it via an in-process _context() side effect (p0's
        # assert_icl_block_matches_mix, p4's run_parity_unit). On a RESUMED
        # process every earlier phase fast-forwards on its done-file, so no
        # registration ever ran and the production p5 crashed here — while the
        # fresh-out_root smoke passed via p0's side effect. Same seam as
        # run_ladder_unit (p2); the registered TEXT is the #1090 fu3 training
        # context verbatim (fu3w.ensure_context -> artifacts.context.
        # icl_prefix_context over the committed bank, byte-asserted as a prefix
        # of every ICL row in the frozen fu3 mix at p0).
        ctx = _context(context_id)
        logger.info("[tier2] context resolved: %s (cell %s)", ctx.context_id, cell)
        organism = ModelOrganism(behavior=C.BEHAVIOR, context_id=context_id, seed=cfg.seed)
        rate_fn = make_source_rate_fn(
            organism,
            out_dir=cell_root / "tier2_rate",
            eval_questions=_eval_questions(cfg),
            n_completions=cfg.tier2_n,
            temperature=1.0,
            n_judge_draws=cfg.tier2_draws,
            judge_fn=fu1._judge_fu1,
        )
        try:
            ckpt = _selected_ckpt(cfg, cell, selections)
            _ensure_dir_tokenizer(ckpt)
            trained = float(rate_fn(str(ckpt)))
        finally:
            close = getattr(rate_fn, "close", None)
            if callable(close):
                close()
        # Base side: fu3's instrument-identical committed base reads (0.00 at
        # every context — plan §4.4/assumption 13; no re-read).
        rec = {
            "cell": cell,
            "step": int(selections[cell]["step"]),
            "rates": {"trained": trained, "base": 0.0},
            "base_source": "eval_results/issue_1090/fu3 committed base Tier-2 (rate 0.00)",
            "n": cfg.tier2_n,
        }
        _atomic_json(res_path, rec)
        out[cell] = rec
    deliver = REPO_ROOT / "eval_results" / "issue_1315" / "install"
    if cfg.smoke:
        deliver = cfg.out_root / "eval_results_mirror" / "install"
    deliver.mkdir(parents=True, exist_ok=True)
    for cell, rec in out.items():
        _atomic_json(deliver / f"{cell}_tier2.json", rec)
    return out


# ── p6: teacher-forced fixed-pool margin (NEW FT cells; the fu4 instrument) ──


def phase_margin(cfg: Cfg, selections: dict) -> dict:
    _phase("p6_margin")
    cells = [c for c in cfg.cells if c in C.FT_CELLS and c in selections]
    if not cells:
        return {"skipped": True}
    out_dir = cfg.out_root / "margin"
    out_dir.mkdir(parents=True, exist_ok=True)
    pos, neg, meta = _margin_pools(cfg)
    if cfg.smoke:
        pos, neg = pos[:2], neg[:2]  # tiny-real slice AFTER the full-pool sha assert
    ctx = _context(C.ICL_CONTEXT_ID)
    questions = _eval_questions(cfg)
    ctxs = fu4._fu4_margin_contexts(ctx, questions)
    out: dict[str, dict] = {}
    margin_fn = None
    try:
        for cell in cells:
            rec_path = out_dir / f"{cell}.json"
            if rec_path.exists():
                out[cell] = _read_json(rec_path)
                continue
            if margin_fn is None:
                margin_fn = _default_margin_read_fn(DEFAULT_BASE_MODEL)
            base_reads = fu4._margin_sweep(
                margin_fn, None, ctxs, pos, neg, out_dir / "base_icl.json"
            )
            ckpt = _selected_ckpt(cfg, cell, selections)
            _ensure_dir_tokenizer(ckpt)
            trained_reads = fu4._margin_sweep(
                margin_fn, str(ckpt), ctxs, pos, neg, out_dir / f"trained_{cell}.json"
            )
            assert_rec = fu1.assert_adapter_applied(
                base_reads["source_ctx"],
                trained_reads["source_ctx"],
                tol=fu1.ADAPTER_ASSERT_TOL_SMOKE if cfg.smoke else fu1.ADAPTER_ASSERT_TOL_FULL,
                tag=f"margin:{cell}",
            )
            combined = {f"base__{k}": v for k, v in base_reads.items()}
            combined.update({f"trained__{k}": v for k, v in trained_reads.items()})
            rec = {
                "status": "computed",
                "cell": cell,
                **meta,
                "smoke_pool_slice": len(pos) if cfg.smoke else None,
                "selected_step": int(selections[cell]["step"]),
                "adapter_assert": assert_rec,
                **fu1.aggregate_margin_reads(combined, fu1._q_labels(len(questions))),
            }
            _atomic_json(rec_path, rec)
            out[cell] = rec
    finally:
        if margin_fn is not None:
            close = getattr(margin_fn, "close", None)
            if callable(close):
                close()
    deliver = REPO_ROOT / "eval_results" / "issue_1315" / "install"
    if cfg.smoke:
        deliver = cfg.out_root / "eval_results_mirror" / "install"
    deliver.mkdir(parents=True, exist_ok=True)
    for cell, rec in out.items():
        _atomic_json(deliver / f"{cell}_margin.json", rec)
    return out


# ── p7: impolite r_B extraction (read-out regime, all 28 layers) ─────────────


def _seed_rb_artifacts_from_registry(cache_path: Path) -> dict:
    """Pre-seed the issue779 extractor's per-trait artifacts cache from the
    BEHAVIORS registry — the #1090 impolite definition (trait description + 5
    contrastive pairs + the 20-q extraction set DISJOINT from the eval bank).
    Without this the extractor would Sonnet-generate its OWN artifacts on the
    fresh instance (the #1112 phase_rb pattern, behavior=impolite)."""
    b = BEHAVIORS[C.BEHAVIOR]
    ex = b.extraction
    assert ex is not None and b.judge_rubric, "impolite registry entry is a stub"
    assert "{question}" in b.judge_rubric and "{answer}" in b.judge_rubric
    ext_qs = _extraction_questions()
    overlap = set(ext_qs) & set(b.eval_question_bank)
    assert not overlap, f"extraction/eval question overlap: {sorted(overlap)[:3]}"
    artifacts = {
        "instruction": [{"pos": p.exhibit, "neg": p.not_exhibit} for p in ex.prompt_pairs],
        "extraction_questions": ext_qs,
        "eval_prompt": b.judge_rubric,
        "provenance": {
            "source": "artifacts.behavior.BEHAVIORS['impolite'] (the #1090 definition)",
            "seeded_by": "issue1315_dispatch.phase_rb",
            "n_pairs": len(ex.prompt_pairs),
            "n_extraction_questions": len(ext_qs),
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(cache_path, artifacts)
    return artifacts


def phase_rb(cfg: Cfg) -> dict:
    _phase("p7_rb")
    rb_dir = cfg.out_root / "rb"
    done = rb_dir / "rb_done.json"
    if done.exists():
        return _read_json(done)
    rec: dict = {}
    if cfg.smoke:
        rec["skipped"] = "smoke — r_B extraction is a full-mode phase (plan §9 overlap slot)"
        _atomic_json(done, rec)
        return rec
    _seed_rb_artifacts_from_registry(
        REPO_ROOT / "data" / "issue_779" / "artifacts" / f"{C.BEHAVIOR}.json"
    )
    _run_subprocess(
        [
            "uv",
            "run",
            "python",
            RB_EXTRACTOR,
            "--traits",
            C.BEHAVIOR,
            "--out-dir",
            str(rb_dir),
            "--no-upload",
        ],
        rb_dir / "rb.log",
    )
    import torch

    src = rb_dir / "r_b" / f"{C.BEHAVIOR}.pt"
    if not src.exists():
        raise FileNotFoundError(f"r_B extractor produced no tensor at {src}")
    obj = torch.load(src, map_location="cpu", weights_only=False)
    r_b = obj["r_b"]
    assert tuple(r_b.shape) == (C.N_LAYERS, C.HIDDEN), r_b.shape
    torch.save(
        {"rb": r_b, "counts": obj.get("counts"), "source": str(src)},
        rb_dir / f"rb_{C.BEHAVIOR}.pt",
    )
    rec[C.BEHAVIOR] = str(rb_dir / f"rb_{C.BEHAVIOR}.pt")
    rollout_files = sorted((rb_dir / "raw_completions").glob(f"rollouts_{C.BEHAVIOR}_*.json"))
    if not rollout_files:
        raise FileNotFoundError(
            f"r_B extractor persisted no rollout text under {rb_dir / 'raw_completions'} "
            "(rollout text is never discardable — upload policy #779)"
        )
    rec["rollout_files"] = [p.name for p in rollout_files]
    _atomic_json(done, rec)
    return rec


# ── p8: own-text capture (sharded per (cell, dose)) ──────────────────────────


def _capture_panel(cfg: Cfg, cell: str) -> dict[str, dict]:
    """{context_id: {system, user_wrap, prior_turns}} for one capture pass.

    Per plan §4.3: the cell's OWN source context + the 5-member default_v1
    panel (incl. the bare assistant). The base pass captures the UNION of the
    3 source contexts + the panel (8 contexts; the conditional bare cell adds
    no context — bare == the default panel member)."""

    def _ctx_entry(context_id: str) -> dict:
        ctx = _context(context_id)
        return {
            "system": ctx.system,
            "user_wrap": ctx.user_wrap,
            "prior_turns": tuple(dict(t) for t in ctx.prefix_turns),
        }

    panel: dict[str, dict] = {}
    if cell == "base":
        source_ids = sorted(
            {_cell_context_id(cfg, c) for c in cfg.cells if c != C.CONDITIONAL_BARE_CELL}
        )
        for cid in source_ids:
            panel[cid] = _ctx_entry(cid)
    else:
        cid = _cell_context_id(cfg, cell)
        panel[cid] = _ctx_entry(cid)
    for neg in default_panel():
        panel[neg.slug] = {
            "system": neg.system_prompt,
            "user_wrap": neg.user_wrap,
            "prior_turns": (),
        }
    return panel


def capture_passes(cfg: Cfg) -> list[tuple[str, str]]:
    """Registered (cell, dose) own-text passes (plan §4.5) — fail-loud on an
    unregistered cell (the #546-class silent-skip canary gap)."""
    passes: list[tuple[str, str]] = []
    for cell in cfg.cells:
        if cell in C.REUSED_LORA_CELLS:
            doses = ("selected",) if cfg.smoke else tuple(C.REUSED_LORA_CELLS[cell]["doses"])
            passes += [(cell, d) for d in doses]
        elif cell in C.FT_CELLS or cell == C.CONDITIONAL_BARE_CELL:
            passes.append((cell, "selected"))
        else:
            raise ValueError(
                f"capture_passes: unroutable cell {cell!r} — register it in the cell "
                "tables before dispatch"
            )
    passes.append(("base", "base"))
    return passes


def _resolve_capture_model(cfg: Cfg, cell: str, dose: str) -> tuple[str, Path | None]:
    """(model_path, merged_dir_to_cleanup) for one capture pass — LoRA cells go
    through the #653 atomic merge-read-delete; FT rungs load directly."""
    if cell == "base":
        return DEFAULT_BASE_MODEL, None
    cell_root = cfg.out_root / cell
    if cell in C.REUSED_LORA_CELLS:
        step = C.REUSED_LORA_CELLS[cell]["doses"][dose]
        adapter = _staged_reused_ckpt(cfg, cell, step)
        merged = _merge_adapter(cfg, str(adapter), cell_root / f"merged_{dose}")
        return str(merged), merged
    if cell == C.CONDITIONAL_BARE_CELL:
        adapter = _staged_reused_ckpt(cfg, cell, 0)
        merged = _merge_adapter(cfg, str(adapter), cell_root / f"merged_{dose}")
        return str(merged), merged
    assert dose == "selected", (cell, dose)
    step = int(_read_json(cell_root / "selection.json")["step"])
    ckpt = _enumerate_rungs(_read_json(cell_root / "build_result.json")["adapter_root"])[step]
    _ensure_dir_tokenizer(ckpt)
    return str(ckpt), None


def _smoke_capture_slice(
    panel: dict[str, dict], questions: list[str]
) -> tuple[dict[str, dict], list[str]]:
    """Smoke capture slice: 2 contexts × 2 questions = 4 rows (the #1112
    parent's proven smoke shape — ``issue1112_dispatch`` ``questions[:2]``).

    ≥2 contexts keep the prefix Δx cloud nondegenerate (#1112 crash-fix r3);
    ≥2 DISTINCT question ids are REQUIRED by the p10 split-half attenuation
    ceiling (``geometry.split_half_self_cosine`` asserts ``len(qs) >= 2``).
    The r1 port shrank the parent's ``questions[:2]`` to ``[:1]``, so every
    smoke row carried ``question_idx == 0`` and p10 crashed on ``qs=[0]``
    (crash-fix r4, att-20260715-125711). Always includes the multi-turn
    WildChat context so the new span logic runs end-to-end.
    """
    wc = C.CONV_CONTEXT_ID
    if wc not in panel:
        ctx = _context(wc)
        panel = {
            **panel,
            wc: {
                "system": ctx.system,
                "user_wrap": ctx.user_wrap,
                "prior_turns": tuple(dict(t) for t in ctx.prefix_turns),
            },
        }
    first_other = next(iter(k for k in panel if k != wc))
    sliced = {wc: panel[wc], first_other: panel[first_other]}
    qs = questions[:2]
    assert len(qs) >= 2, (
        f"smoke capture needs >=2 questions for the p10 split-half ceiling; got "
        f"{len(qs)} (check --eval-question-limit)"
    )
    logger.info(
        "[capture-smoke] slice: %d contexts x %d questions (>=2 questions for p10 split-half)",
        len(sliced),
        len(qs),
    )
    return sliced, qs


def run_capture_unit(cfg: Cfg, cell: str, dose: str) -> None:
    """One own-text capture pass: on-policy greedy gen + 28-layer 3-span TF
    pooling -> pooled.pt. Multi-turn contexts thread prior_turns + user_wrap
    through generation AND span computation (prefix_end='last_user' — the ONE
    measurement-rig delta vs #1112, plan §4.1)."""
    import torch
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import (
        _generate_responses_vllm,
        _teacher_forced_span_means,
        compute_prompt_spans,
    )

    out_dir = cfg.out_root / "capture" / cell / dose
    if (out_dir / "pooled.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path, cleanup_merged = _resolve_capture_model(cfg, cell, dose)
    panel = _capture_panel(cfg, cell)
    questions = _eval_questions(cfg)
    if cfg.smoke:
        panel, questions = _smoke_capture_slice(panel, questions)
    personas = {k: v["system"] for k, v in panel.items()}
    user_wraps = {k: v["user_wrap"] for k, v in panel.items()}
    prior_turns = {k: v["prior_turns"] for k, v in panel.items()}
    rows = _generate_responses_vllm(
        model_path,
        personas,
        questions,
        max_new_tokens=C.MAX_NEW_TOKENS,
        gpu_memory_utilization=C.CAPTURE_GPU_MEM_UTIL,
        user_wraps=user_wraps,
        prior_turns=prior_turns,
    )
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    for r in rows:
        ctx_id = r["persona"]
        q = questions[r["question_idx"]]
        r["prefix_len"], r["context_len"] = compute_prompt_spans(
            tokenizer,
            personas[ctx_id],
            q,
            r["prompt_token_ids"],
            prior_messages=list(prior_turns.get(ctx_id) or ()),
            user_wrap=user_wraps.get(ctx_id),
            prefix_end="last_user",
        )
    # persist rollout text BEFORE the capture reduce (upload policy #779)
    (out_dir / "raw_rows.json").write_text(
        json.dumps({"model": model_path, "rows": rows}, ensure_ascii=False)
    )
    pooled = _teacher_forced_span_means(
        model_path,
        rows,
        list(panel),
        layers=list(range(C.N_LAYERS)),
        device="cuda:0",
        dtype=torch.bfloat16,
        tf_batch_size=C.TF_BATCH_SIZE,
    )
    store = {
        "schema_version": 1,
        "cell": cell,
        "dose": dose,
        "behavior": C.BEHAVIOR,
        "model_path": model_path,
        "row_meta": [{"context_id": r["persona"], "question_idx": r["question_idx"]} for r in rows],
        "arms": {
            arm: {li: t.to(torch.float16) for li, t in per_layer.items()}
            for arm, per_layer in pooled.items()
        },
        "metadata": {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "max_new_tokens": C.MAX_NEW_TOKENS,
            "tf_batch_size": C.TF_BATCH_SIZE,
            "prefix_end": "last_user",
        },
    }
    tmp = out_dir / "pooled.pt.tmp"
    torch.save(store, tmp)
    os.replace(tmp, out_dir / "pooled.pt")
    if cleanup_merged is not None:
        shutil.rmtree(cleanup_merged, ignore_errors=True)


def phase_capture(cfg: Cfg) -> dict:
    _phase("p8_capture")
    passes = [
        (c, d)
        for c, d in capture_passes(cfg)
        if not (cfg.out_root / "capture" / c / d / "pooled.pt").exists()
    ]
    if not passes:
        return {"n_passes": 0}
    if _n_gpus() == 1 or len(passes) == 1:
        for c, d in passes:
            run_capture_unit(cfg, c, d)
    else:
        _fanout_units(cfg, [_unit_args(cfg, "capture", f"{c}/{d}") for c, d in passes])
    return {"n_passes": len(passes)}


# ── p9: shared-text response re-capture (MANDATORY — CJK control, §4.5) ──────


def run_capture_tf_unit(cfg: Cfg, cell: str) -> None:
    """Teacher-forced SHARED-response re-capture at the cell's selected rung
    over the persisted BASE rows (shared text IS the base generation, so the
    base side needs no re-capture)."""
    import torch

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    out_dir = cfg.out_root / "capture_tf" / cell / "selected"
    if (out_dir / "pooled.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    base_raw = cfg.out_root / "capture" / "base" / "base" / "raw_rows.json"
    rows_all = json.loads(base_raw.read_text(encoding="utf-8"))["rows"]
    cell_contexts = set(_capture_panel(cfg, cell))
    rows = [r for r in rows_all if r["persona"] in cell_contexts]
    assert rows, (cell, sorted(cell_contexts))
    selections = {cell: _read_json(cfg.out_root / cell / "selection.json")}
    model_path, cleanup_merged = (
        _resolve_capture_model(cfg, cell, "selected")
        if cell not in C.FT_CELLS
        else (str(_selected_ckpt(cfg, cell, selections)), None)
    )
    if cell in C.FT_CELLS:
        _ensure_dir_tokenizer(Path(model_path))
    # ALL THREE arms (default SPAN_ARMS), not response-only: the geometry read
    # consumes only the response arm, but the persisted prefix/context arms are
    # what make the plan §4.5 prompt-arm parity read + §Kill criterion runnable
    # post-hoc (concern tf-shared-parity-warn-check-not-ported — the tf and own
    # captures share prompt tokens under the SAME selected model, so per-row
    # cosine >= 0.999 up to bf16 batch jitter). Same forwards; ~3x store size
    # (tens of MB per cell).
    pooled = _teacher_forced_span_means(
        model_path,
        rows,
        sorted({r["persona"] for r in rows}),
        layers=list(range(C.N_LAYERS)),
        device="cuda:0",
        dtype=torch.bfloat16,
        tf_batch_size=C.TF_BATCH_SIZE,
    )
    store = {
        "schema_version": 1,
        "cell": cell,
        "dose": "selected",
        "behavior": C.BEHAVIOR,
        "model_path": model_path,
        "shared_text_source": str(base_raw),
        "row_meta": [{"context_id": r["persona"], "question_idx": r["question_idx"]} for r in rows],
        "arms": {
            arm: {li: t.to(torch.float16) for li, t in per_layer.items()}
            for arm, per_layer in pooled.items()
        },
        "metadata": {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    tmp = out_dir / "pooled.pt.tmp"
    torch.save(store, tmp)
    os.replace(tmp, out_dir / "pooled.pt")
    if cleanup_merged is not None:
        shutil.rmtree(cleanup_merged, ignore_errors=True)


def phase_capture_tf(cfg: Cfg) -> dict:
    _phase("p9_capture_tf")
    cells = [c for c in cfg.cells]
    pending = [
        c
        for c in cells
        if not (cfg.out_root / "capture_tf" / c / "selected" / "pooled.pt").exists()
    ]
    if not pending:
        return {"n_passes": 0}
    if _n_gpus() == 1 or len(pending) == 1:
        for c in pending:
            run_capture_tf_unit(cfg, c)
    else:
        _fanout_units(cfg, [_unit_args(cfg, "capture_tf", c) for c in pending])
    return {"n_passes": len(pending)}


# ── p10: geometry smoke (full geometry runs VM-side) ─────────────────────────


def phase_geometry_smoke(cfg: Cfg) -> dict:
    _phase("p10_geometry")
    if not cfg.smoke:
        return {"skipped": "full geometry runs VM-side (scripts/issue1315_geometry.py)"}
    import torch

    from explore_persona_space.experiments.issue_1112 import geometry as geo

    cell = cfg.cells[0]
    rb = torch.randn(C.N_LAYERS, C.HIDDEN)  # smoke stub direction (labeled)
    rb_path = cfg.out_root / "rb" / "rb_smoke.pt"
    rb_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rb": rb, "smoke_stub": True}, rb_path)
    payload = geo.run_geometry(
        cfg.out_root / "capture",
        cfg.out_root / "geometry_smoke",
        cells_doses=[(cell, "selected")],
        base_store_by_behavior={
            C.BEHAVIOR: cfg.out_root / "capture" / "base" / "base" / "pooled.pt"
        },
        behavior_by_cell={cell: C.BEHAVIOR},
        selected_dose_by_cell={cell: "selected"},
        rb_by_behavior={C.BEHAVIOR: rb_path},
        n_boot=25,
    )
    prefix_recs = [r for r in payload["records"].values() if r["arm"] == "prefix"]
    n_nondegen = sum(1 for r in prefix_recs if not r.get("degenerate"))
    if n_nondegen < 1:
        raise RuntimeError(
            "smoke geometry produced no nondegenerate prefix-arm record — the smoke "
            "capture must span >=2 contexts"
        )
    return {"n_records": len(payload["records"]), "n_prefix_nondegenerate": n_nondegen}


# ── p11: upload + sentinel ───────────────────────────────────────────────────


def phase_upload(cfg: Cfg, selections: dict) -> dict:
    # NOTE (#664 risk class, round-1 review Minor): a full run issues ~60
    # per-file data-repo commits (inherited verbatim from the #1112 dispatcher,
    # production-proven; well under the 256-commits/hr Hub cap). If the file
    # count grows, consolidate the per-prefix sets into upload_folder /
    # create_commit batches.
    _phase("p11_upload")
    uploaded: dict[str, str] = {}
    if not cfg.upload:
        return uploaded

    def _up(local: Path, path_in_repo: str, **kw) -> None:
        if not Path(local).exists():
            return
        url = hub._upload(local, C.HF_DATA_REPO, "dataset", path_in_repo, **kw)
        if not str(url):
            raise RuntimeError(f"upload returned no path for {path_in_repo}")
        uploaded[path_in_repo] = str(url)
        _atomic_json(cfg.out_root / "upload_manifest.json", uploaded)

    inputs = cfg.out_root / "inputs"
    for name in ("icl_con_mix_meta.json", "icl_pos_mix_meta.json"):
        _up(inputs / name, f"{C.DATA_PREFIX}/inputs/{name}", upload_as_file=True)
    for cell in cfg.cells:
        cell_root = cfg.out_root / cell
        for name in (
            "build_result.json",
            "ladder.json",
            "selection.json",
            "parity.json",
            "tier2.json",
            "g1_extended.json",
        ):
            _up(cell_root / name, f"{C.DATA_PREFIX}/selection/{cell}/{name}", upload_as_file=True)
        # rollout text (unconditional). A cell's rate/ dir holds Tier-1 ladder
        # rollouts for FT cells but PARITY-probe rollouts for reused cells
        # (run_parity_unit) — stage prefixes per plan §10.
        rate_stage = "tier1" if cell in C.FT_CELLS else "parity"
        _up(cell_root / "rate", f"{C.DATA_PREFIX}/raw_completions/{rate_stage}/{cell}")
        _up(cell_root / "tier2_rate", f"{C.DATA_PREFIX}/raw_completions/tier2/{cell}")
    for f in (
        sorted((cfg.out_root / "margin").glob("*.json"))
        if (cfg.out_root / "margin").exists()
        else []
    ):
        _up(f, f"{C.DATA_PREFIX}/margin/{f.name}", upload_as_file=True)
    # capture: rollout text (unconditional) + pooled tensors (analysis_tensors)
    for c, d in capture_passes(cfg):
        cap = cfg.out_root / "capture" / c / d
        _up(
            cap / "raw_rows.json",
            f"{C.DATA_PREFIX}/raw_completions/capture/{c}/{d}/raw_rows.json",
            upload_as_file=True,
        )
        _up(
            cap / "pooled.pt",
            f"{C.DATA_PREFIX}/analysis_tensors/capture/{c}/{d}/pooled.pt",
            upload_as_file=True,
        )
    tf_root = cfg.out_root / "capture_tf"
    if tf_root.exists():
        for pooled in sorted(tf_root.glob("*/selected/pooled.pt")):
            cell = pooled.parent.parent.name
            _up(
                pooled,
                f"{C.DATA_PREFIX}/analysis_tensors/capture_tf/{cell}/selected/pooled.pt",
                upload_as_file=True,
            )
    rb_dir = cfg.out_root / "rb"
    _up(
        rb_dir / f"rb_{C.BEHAVIOR}.pt",
        f"{C.DATA_PREFIX}/analysis_tensors/rb/rb_{C.BEHAVIOR}.pt",
        upload_as_file=True,
    )
    rb_rc = rb_dir / "raw_completions"
    for f in sorted(rb_rc.glob("rollouts_*.json")) if rb_rc.exists() else []:
        _up(f, f"{C.DATA_PREFIX}/raw_completions/rb_extraction/{f.name}", upload_as_file=True)
    for extra in sorted(rb_dir.glob("**/*.json")) if rb_dir.exists() else []:
        if "raw_completions" in extra.parts:
            continue
        _up(extra, f"{C.DATA_PREFIX}/rb/{extra.name}", upload_as_file=True)
    _up(cfg.out_root / "run_config.json", f"{C.DATA_PREFIX}/run_config.json", upload_as_file=True)
    return uploaded


def _reproducibility_card(cfg: Cfg, selections: dict) -> dict:
    adapters = {}
    for cell in cfg.cells:
        if cell in C.FT_CELLS and cell in selections:
            adapters[cell] = f"issue1315/{cell}/checkpoint-{int(selections[cell]['step'])}"
    return {
        "adapter_paths": adapters,
        "hf_model_repo": C.OVERFLOW_REPO,
        "wandb_project": C.WANDB_PROJECT,
        "wandb_run_names": [ft_wandb_run_name(cell, cfg.seed) for cell in adapters],
    }


def write_sentinel(cfg: Cfg, summary: dict) -> Path:
    _phase("sentinel")
    sentinel_dir = cfg.sentinel_dir or Path("/workspace/logs")
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    kind = "epm:smoke-result" if cfg.smoke else "epm:results"
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,  # VM-side drain re-derives max+1
        "task_id": C.ISSUE,
        "by": "issue1315_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": summary,
    }
    path = sentinel_dir / f"issue-{C.ISSUE}-{kind.replace(':', '_')}-{int(time.time())}.json"
    _atomic_json(path, payload)
    logger.info("[sentinel] wrote %s", path)
    return path


# ── main ─────────────────────────────────────────────────────────────────────


def _check_regime(cfg: Cfg) -> None:
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    p = cfg.out_root / "run_config.json"
    cur = cfg.regime_key()
    if p.exists():
        prior = _read_json(p)
        prior_rest = {k: v for k, v in prior.items() if k != "cells"}
        cur_rest = {k: v for k, v in cur.items() if k != "cells"}
        if prior_rest != cur_rest or not set(cur["cells"]) <= set(prior.get("cells", [])):
            raise RuntimeError(f"out_root {cfg.out_root} holds a run under a DIFFERENT regime")
    else:
        _atomic_json(p, cur)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1315 pod-side phase driver")
    # Mode: `--mode smoke|full` (the plan §10 workload command shape) is
    # standalone-sufficient; `--smoke` / `--full` are accepted aliases. The
    # group is NOT required=True — post-parse validation below accepts either
    # spelling (round-1 code-review Major 2: required=True exit-2'd the plan
    # §10 exact command on the GCE lane).
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="tiny-real, SAME code path")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--mode", choices=["smoke", "full"], default=None, help="smoke|full")
    p.add_argument(
        "--unit",
        nargs=2,
        default=None,
        metavar=("KIND", "ARG"),
        help="internal: one fanout unit (ladder <cell> | parity <cell> | "
        "capture <cell>/<dose> | capture_tf <cell>)",
    )
    p.add_argument(
        "--gpu-id", type=int, default=0, help="physical GPU (CVD-pinned by the launcher)"
    )
    p.add_argument("--cells", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    p.add_argument("--phases", default=None, help="comma subset of phases to run (default all)")
    p.add_argument("--include-bare", action="store_true", help="add the conditional fu5 cell")
    p.add_argument("--bare-adapter-prefix", default=None)
    p.add_argument("--bare-adapter-rev", default=None)
    p.add_argument("--bare-committed-rate", type=float, default=None)
    args = p.parse_args(argv)
    if args.mode is not None:
        mode_smoke = args.mode == "smoke"
        if (args.smoke or args.full) and args.smoke != mode_smoke:
            p.error("--mode conflicts with --smoke/--full")
        args.smoke, args.full = mode_smoke, not mode_smoke
    elif not (args.smoke or args.full):
        p.error("one of --smoke, --full, or --mode {smoke,full} is required")
    return args


def build_cfg(args: argparse.Namespace) -> Cfg:
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else (f"/tmp/issue-{C.ISSUE}-smoke" if smoke else f"data/issue_{C.ISSUE}/run")
    )
    return Cfg(
        smoke=smoke,
        cells=resolve_cells(args.cells, smoke, bare=bool(args.include_bare)),
        out_root=out_root,
        seed=args.seed,
        tier1_n=2 if smoke else 5,
        tier1_draws=2 if smoke else 3,
        tier2_n=2 if smoke else 10,
        tier2_draws=2 if smoke else 5,
        eval_question_limit=(
            args.eval_question_limit
            if args.eval_question_limit is not None
            else (2 if smoke else None)
        ),
        # Smoke sentinels default INSIDE the poller-drained namespace
        # (/workspace/logs — round-1 review Minor: an out_root default left the
        # pod smoke's epm:smoke-result undrained); the out_root fallback covers
        # workspace-less local CPU smokes only.
        sentinel_dir=(
            Path(args.sentinel_dir)
            if args.sentinel_dir is not None
            else (
                (Path("/workspace/logs") if Path("/workspace").is_dir() else out_root / "logs")
                if smoke
                else None
            )
        ),
        upload=args.upload,
        phases=normalize_phases(args.phases),
        bare_adapter_prefix=args.bare_adapter_prefix,
        bare_adapter_rev=args.bare_adapter_rev,
        bare_committed_rate=args.bare_committed_rate,
    )


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 — linear phase chain
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    cfg = build_cfg(args)
    if args.unit is not None:
        kind, arg = args.unit
        if kind == "ladder":
            run_ladder_unit(cfg, arg)
        elif kind == "parity":
            run_parity_unit(cfg, arg)
        elif kind == "capture":
            cell, dose = arg.split("/")
            run_capture_unit(cfg, cell, dose)
        elif kind == "capture_tf":
            run_capture_tf_unit(cfg, arg)
        else:
            raise ValueError(f"unknown unit kind {kind!r}")
        return 0
    _check_regime(cfg)
    logger.info("issue1315 smoke=%s cells=%s out_root=%s", cfg.smoke, cfg.cells, cfg.out_root)

    def want(phase: str) -> bool:
        return not cfg.phases or phase in cfg.phases

    summary: dict = {"issue": C.ISSUE, "smoke": cfg.smoke, "cells": list(cfg.cells)}
    if want("stage"):
        summary["stage"] = {
            k: v for k, v in phase_stage(cfg).items() if k in ("context_byte_asserts", "ts")
        }
    if want("train"):
        phase_train(cfg)
    selections: dict = {}
    if want("ladder"):
        selections = phase_ladder(cfg)
        summary["selections"] = {
            k: {kk: v.get(kk) for kk in ("step", "rate", "in_band", "reused")}
            for k, v in selections.items()
        }
    if want("g1"):
        g1 = phase_g1_gate(cfg, selections)
        summary["g1"] = {"action": g1.get("action")}
        if g1.get("action") == "extend_in_place":
            selections = phase_ladder(cfg)  # re-ladder the extended trees
            summary["selections_post_g1"] = {
                k: {kk: v.get(kk) for kk in ("step", "rate", "in_band")}
                for k, v in selections.items()
                if k in C.FT_CELLS
            }
        summary["g1_disposition"] = g1_closest_approach(cfg, selections)
    if want("persist_ft"):
        summary["persist_ft"] = {"n": len(phase_persist_ft(cfg, selections).get("uploaded", {}))}
    if want("parity") and not cfg.smoke:
        summary["parity"] = {
            k: {kk: v.get(kk) for kk in ("rate", "expected", "rate_window_pass", "severity")}
            for k, v in phase_parity(cfg).items()
            if isinstance(v, dict) and "rate" in v
        }
    if want("tier2"):
        summary["tier2"] = {k: v.get("rates") for k, v in phase_tier2(cfg, selections).items()}
    if want("margin"):
        summary["margin"] = {
            k: {kk: v.get(kk) for kk in ("margin_base", "margin_trained", "margin_delta")}
            for k, v in phase_margin(cfg, selections).items()
            if isinstance(v, dict)
        }
    if want("rb"):
        phase_rb(cfg)
    if want("capture"):
        summary["capture"] = phase_capture(cfg)
    if want("capture_tf"):
        summary["capture_tf"] = phase_capture_tf(cfg)
    if want("geometry"):
        summary["geometry"] = phase_geometry_smoke(cfg)
    summary["reproducibility_card"] = _reproducibility_card(cfg, selections)
    if want("upload"):
        uploaded = phase_upload(cfg, selections)
        summary["n_uploaded"] = len(uploaded)
    summary["sentinel"] = str(write_sentinel(cfg, summary))
    logger.info(
        "issue1315 complete: %s",
        json.dumps({k: summary[k] for k in ("smoke", "cells", "n_uploaded") if k in summary}),
    )
    # NOTE: [phase=done] is emitted by the launcher wrapper, never here.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
