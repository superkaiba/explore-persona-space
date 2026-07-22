#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002, RUF003  # em-dash / ※ intentional
"""#1586 pod-side phase driver — matched-install LoRA vs full-FT method comparison.

Phases (linear, checkpoint-per-phase, resume-keyed; plan §4.8), generalizing
``issue1315_dispatch.py`` to the (behavior × regime × seed) grid:

  p0_stage    pinned-revision fetch: mixes (sha + composition asserts), 16
              LoRA arms + 2 reused FT checkpoints (per-file downloads to
              consumer-exact paths), adapter_config grounding (2 recipe
              classes), 1-file staging probe + consumer-open per
              (family × consumer) pair; marker-id assert
  p1_parity   reused-arm apply-and-read WARN gates (§4.4) + per-recipe-class
              rsLoRA probes (HALT only on structural apply-path breakage)
  p2_train    content FT cells (ZeRO-3 quads; 2 concurrent on 8 GPUs with
              CVD 0-3/4-7 + distinct MASTER_PORT) + marker FT cells (grid 1-6)
  p3_ladder   content Tier-1 rate ladders / marker slot-read ladders;
              anchor-nearest in-band selection (§4.3); registered one-shot
              extensions; between-cell rung reap (--ladder-disk-mode)
  p4_persist  selected FT rungs -> overflow repo issue1586/<cell>/checkpoint-<k>;
              selection records -> data repo (incremental); non-selected rung
              reap (declared discard, plan §10)
  p5_tier2    Tier-2 confirm (new FT content cells) + reused-FT parity re-read
              + the #1112 po parity cross-check row; dose-match labels
  p6_panel    six-context leakage panel — content judged rates (24 arms) +
              marker slot reads (8 arms), both sides fresh on THIS rig
  p7_margin   teacher-forced fixed-pool margin, content arms (fu4 instrument)
  p8_capture  own-text capture (6 ctx × 20 q × 3 arms × 28 layers) for all
              cells + per-behavior base stores + shared-text TF re-capture
  p9_upload   residual uploads + upload manifest + CJK audit records

``--smoke`` is the SAME dispatcher with ONE content FT cell
(syc-pers-ft-con-s137) end-to-end p2→p9 at ``--eval-question-limit 2`` — same
subprocess shapes, same env injection, same PRODUCTION launch width
(``--num_processes 4``; smoke never narrows the process shape — #1315/#1333),
same teardown. The smoke cell is also the staging-probe carrier (checkpoint
family consumer-open through the REAL per-file staging path). Every phase
reads its cell list from the ONE resolver (``cfg.cells``).

Pod-side contract: NEVER shells out to scripts/task.py; progress =
``[phase=...]`` log lines + the end-of-run sentinel (pod-side-reporting.md).
``[phase=done]`` is emitted by the launcher wrapper ONLY, never here.
Designed halts exit DISTINCT rcs (pilot gate rc=7) with a report JSON.
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
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

# vLLM v1 EngineCore fork-poisoning guard (gotchas.md #628): BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1090_fu1 as fu1  # noqa: E402
import issue1090_fu3_cells as fu3_cells  # noqa: E402
import issue1090_fu4 as fu4  # noqa: E402
import issue1333_dispatch as d1333  # noqa: E402
import issue1481_cells as c1481  # noqa: E402
import issue1481_marker as mk1481  # noqa: E402
import issue1586_cells as G  # noqa: E402

# Generic machinery reused verbatim from the #1112 driver (same repos, same
# subprocess/env/CVD discipline — reuse hierarchy, CLAUDE.md):
from issue1112_dispatch import (  # noqa: E402
    _atomic_json,
    _ensure_dir_tokenizer,
    _enumerate_rungs,
    _marker_slot_read,
    _merge_adapter,
    _phase,
    _physical_gpu_ids,
    _read_json,
    _reap_unit_groups,
    _run_subprocess,
    _stage_file,
    _stage_overflow_prefix,
)

# The transport-retried data-repo upload (crash-fix r8 machinery; the #1315
# module binds the SAME shared data repo, so the import is repo-correct).
from issue1315_dispatch import _upload_with_transport_retry  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.context import CONTEXTS  # noqa: E402
from explore_persona_space.artifacts.negatives import default_panel  # noqa: E402
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    ModelOrganism,
    _default_margin_read_fn,
    _sha256_file,
    make_source_rate_fn,
)
from explore_persona_space.experiments import issue_1112 as P1112  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402

logger = logging.getLogger("issue1586")

ACCEL_CONFIG = "configs/accelerate/zero3_4gpu_accum1.yaml"
MARKER_ACCEL_CONFIG = "configs/accelerate/zero3_4gpu_accum16.yaml"
FT_TRAINER = "scripts/train_behavior_fullft.py"
MARKER_FT_TRAINER = "scripts/issue1112_train_marker_fullft.py"
FT_NUM_PROCESSES = 4  # ZeRO-3 world size (eff-batch contract; #1112 verbatim)
CAPTURE_GPU_MEM_UTIL = 0.6  # vLLM engine cap (Source: #1315 CAPTURE_GPU_MEM_UTIL;
# HF+vLLM coexistence-safe — the TF span pass loads an HF model after gen)
FT_MAX_LENGTH = 2048  # plan §4.2 (Source: #1112 recipe / trainer default)
MARKER_MAX_LENGTH = 1024  # plan §4.2 (Source: #1112 marker FT recipe)

# Per-phase out-root headroom floors (GB) — plan §9 disk rows (#1333 helper).
PHASE_HEADROOM_GB = {
    "p0_stage": 40.0,
    "p2_train": 60.0,
    "p3_ladder": 40.0,
    "p8_capture": 20.0,
    "p9_upload": 5.0,
}

# Compute-pilot kill (plan §7 item 2): per-phase plan walls, each gated
# against its OWN §9 row (review r1 Minor 5 — a combined p2+p3 wall was
# structurally looser than either row, and a ladder-side blowup escaped it).
PILOT_PLAN_P2_WALL_H = 2.2  # §9 p2 row (content FT fan-out)
PILOT_PLAN_P3_WALL_H = 2.3  # §9 p3 (1.5h content ladders) + p3b (0.8h marker reads)
PILOT_PARALLELISM = 2.0  # 2 concurrent ZeRO-3 quads (p2)
PILOT_GATE_RC = 7  # designed halt (never bare rc=1 — gotchas #1415)

# keep-cell needs the GCE 750 GB boot-disk class (plan §9 disk split).
KEEPCELL_MIN_FREE_GB = 300.0

MARKER_TEXT = " ※"
MARKER_TOKEN_ID = 83399


def _assert_marker_token() -> None:
    """In-process marker-id assert (marker-leakage rule; #537 — wired into the
    dispatcher AND the marker trainer; every process fails at startup on a
    wrong marker)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"marker tokenization mismatch: encode({MARKER_TEXT!r}) = {ids}, "
            f"expected [{MARKER_TOKEN_ID}] (bash strips leading spaces — thread "
            "via shlex.quote)"
        )
    logger.info("[stage] marker token id assert PASSED: %r -> %s", MARKER_TEXT, ids)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Cfg:
    smoke: bool
    cells: tuple[str, ...]  # FT cells in scope (LoRA arms resolve via pairing)
    out_root: Path
    ladder_disk_mode: str = "auto"  # auto | keep-cell | stream-reap
    tier1_n: int = 5
    tier1_draws: int = 3
    tier2_n: int = 10
    tier2_draws: int = 5
    panel_n: int = 5
    panel_draws: int = 3
    eval_question_limit: int | None = None
    sentinel_dir: Path | None = None
    upload: bool = True
    phases: tuple[str, ...] = ()  # empty -> all

    def regime_key(self) -> dict:
        return {
            "issue": G.ISSUE,
            "smoke": self.smoke,
            "cells": list(self.cells),
            "ladder_disk_mode": self.resolved_disk_mode(),
            "tier1": [self.tier1_n, self.tier1_draws],
            "tier2": [self.tier2_n, self.tier2_draws],
            "panel": [self.panel_n, self.panel_draws],
            "eval_question_limit": self.eval_question_limit,
            "band": list(G.JUDGED_RATE_BAND),
            "window": list(G.INSTALL_WINDOW),
            "marker": [MARKER_TEXT, MARKER_TOKEN_ID],
        }

    def resolved_disk_mode(self) -> str:
        """``auto`` -> resolved ONCE per out_root and PERSISTED (disk_mode.json).

        Every later consumer — ``regime_key``, the stream-reap decision,
        resumes, and unit subprocesses (which receive the resolved LITERAL via
        ``_unit_args``) — reads the persisted value, so mid-run free-space
        drift or a resume can never flip the regime (review r1 Majors 1+2;
        the spurious "ladder regime drift" class). Explicit modes pass
        through unchanged."""
        if self.ladder_disk_mode != "auto":
            return self.ladder_disk_mode
        p = self.out_root / "disk_mode.json"
        if p.exists():
            return str(_read_json(p)["resolved"])
        mode = probe_disk_mode(self.out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            p,
            {
                "resolved": mode,
                "probed_from": "auto",
                "runpod_quota_lane": _runpod_workspace_quota_lane(),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        logger.info("[disk-mode] auto resolved ONCE -> %s (persisted %s)", mode, p)
        return mode


def _runpod_workspace_quota_lane() -> bool:
    """True on the RunPod lane, where /workspace is the MooseFS volume with a
    ~130 GB per-pod EDQUOT quota that statvfs/df CANNOT see (gotchas.md
    "RunPod MooseFS per-pod disk quota") — keep-cell's ~456 GB ladder
    high-water is never holdable there. Detection: the RunPod container env
    marker, plus a /proc/mounts FUSE probe on /workspace (GCE's /workspace is
    a plain boot-disk dir, never a fuse mount)."""
    if os.environ.get("RUNPOD_POD_ID"):
        return True
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:
            for ln in fh:
                parts = ln.split()
                if len(parts) >= 3 and parts[1] == "/workspace" and "fuse" in parts[2].lower():
                    return True
    except OSError:
        pass
    return False


def probe_disk_mode(out_root: Path) -> str:
    """One-shot disk-mode probe (called exactly once per out_root, then
    persisted — see Cfg.resolved_disk_mode). RunPod lane -> stream-reap
    UNCONDITIONALLY (statvfs reads the MooseFS SHARE free space, TBs, and is
    blind to the ~130 GB per-pod quota — review r1 Major 1); else statvfs on
    the out-root filesystem (GCE 750 GB boot disk / VM)."""
    if _runpod_workspace_quota_lane():
        return "stream-reap"
    try:
        st = os.statvfs(out_root if out_root.exists() else out_root.parent)
        free_gb = st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return "stream-reap"  # unknown filesystem -> conservative
    return "keep-cell" if free_gb >= KEEPCELL_MIN_FREE_GB else "stream-reap"


_PHASE_ALIASES = {
    "p0_stage": "stage",
    "p1_parity": "parity",
    "p2_train": "train",
    "p3_ladder": "ladder",
    "p4_persist": "persist",
    "p5_tier2": "tier2",
    "p6_panel": "panel",
    "p7_margin": "margin",
    "p8_capture": "capture",
    "p8b_capture_tf": "capture_tf",
    "p9_upload": "upload",
}
_KNOWN_PHASES = frozenset(_PHASE_ALIASES.values())


def normalize_phases(raw: str | None) -> tuple[str, ...]:
    """Comma list of phase names -> canonical short-name tuple (fail-loud)."""
    if not raw:
        return ()
    out: list[str] = []
    for tok in raw.split(","):
        t = _PHASE_ALIASES.get(tok.strip(), tok.strip())
        if not t:
            continue
        if t not in _KNOWN_PHASES:
            raise ValueError(f"unknown phase {tok!r}: want one of {sorted(_KNOWN_PHASES)}")
        out.append(t)
    return tuple(out)


def resolve_cells(cells_arg: str | None, smoke: bool) -> tuple[str, ...]:
    """The ONE cell resolver every phase consumes (smoke = same path, 1 cell).

    Cells are FT cells; each threads its method-paired LoRA arm through
    ``G.lora_pair_of`` — so the --cells subset shapes train, ladder, parity,
    tier2, panel, margin, capture, AND upload alike (PASS_UNIFIED per-phase
    subset threading)."""
    known = set(G.ALL_FT_CELLS)
    if cells_arg:
        ids = tuple(t.strip() for t in cells_arg.split(",") if t.strip())
        bad = [t for t in ids if t not in known]
        if bad:
            raise ValueError(f"bad cells {bad!r}: want a subset of {sorted(known)}")
        return ids
    if smoke:
        return (G.SMOKE_CELL,)
    return G.ALL_FT_CELLS


def _n_gpus() -> int:
    return max(1, len(_physical_gpu_ids()))


def _is_marker(cell: str) -> bool:
    return G.parse_ft_cell(cell)[0] == "mk"


def _behavior(cell: str) -> str:
    return G.BEHAVIOR_BY_KEY[G.parse_ft_cell(cell)[0]]


def _mirror_deliverable(cfg: Cfg, unit: str, payload: dict) -> None:
    """Mirror a selection/parity record to the plan §6.5 deliverable glob
    (data/issue_1586/out/selection/<unit>/selection.json)."""
    _atomic_json(cfg.out_root / "selection" / unit / "selection.json", payload)


def _headroom(cfg: Cfg, phase: str) -> None:
    need = PHASE_HEADROOM_GB.get(phase)
    if need is None:
        return
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    assert_out_root_headroom(cfg.out_root, need_gb=need, phase=phase)


# ── Contexts (panel per behavior; the #1481 six-context panel verbatim) ──────


def _register_ctx(ctx) -> None:
    if ctx.context_id not in CONTEXTS:
        CONTEXTS[ctx.context_id] = ctx


def _mk_cfg(cfg: Cfg) -> mk1481.Cfg:
    """Thin #1481-marker Cfg shim (its eval_questions/source_context/
    panel_contexts consume only these fields)."""
    return mk1481.Cfg(
        smoke=cfg.smoke,
        cells=(),
        out_root=cfg.out_root,
        eval_question_limit=cfg.eval_question_limit,
        upload=False,
    )


def panel_context_ids(cfg: Cfg, beh_key: str) -> list[str]:
    """The six read-context ids for one behavior (plan §4.5: source persona,
    bare default, WildChat conv prefix, behavior ICL prefix, + 2 held-out
    persona panel members) — registered idempotently at POINT OF USE (the
    #1315 r6 resume-loss class: never rely on an earlier phase's in-process
    registration side effect)."""
    fu3_cells.register_fu3_contexts()
    if beh_key == "mk":
        panel = mk1481.panel_contexts(_mk_cfg(cfg))
        for ctx in panel.values():
            _register_ctx(ctx)
        return list(panel)
    behavior = G.BEHAVIOR_BY_KEY[beh_key]
    # ICL contexts are NOT registered by register_fu3_contexts — build them
    # from the committed bank via the fu3 factory (the #1315 _context path;
    # verified 2026-07-22: icl_prefix_* absent from CONTEXTS after the fu3
    # registrar alone).
    import issue1090_fu3_worker as fu3w

    fu3w.ensure_context(c1481.context_id_for(behavior, "icl"), behavior)
    ordered = [
        c1481.context_id_for(behavior, "pers"),
        "default",
        c1481.context_id_for(behavior, "conv"),
        c1481.context_id_for(behavior, "icl"),
    ]
    held = 0
    for member in default_panel():
        ctx = member.to_context()
        if ctx.context_id in ordered or ctx.kind != "persona":
            continue
        _register_ctx(ctx)
        ordered.append(ctx.context_id)
        held += 1
        if held == 2:
            break
    if len(ordered) != 6:
        raise RuntimeError(f"panel for {beh_key} has {len(ordered)} contexts, want 6: {ordered}")
    for cid in ordered:
        if cid not in CONTEXTS:
            raise RuntimeError(f"panel context {cid!r} unregistered — fu3/ICL registry gap")
    return ordered


def source_context_id(beh_key: str) -> str:
    if beh_key == "mk":
        return "persona_software_engineer"
    return c1481.context_id_for(G.BEHAVIOR_BY_KEY[beh_key], "pers")


def _read_organism(behavior: str, context_id: str, seed: int) -> ModelOrganism:
    """Read-side organism (identity carrier into ``make_source_rate_fn`` — the
    panel field is semantically unused by a rate READ, which consumes only
    ``behavior_spec`` + ``context``).

    Threads the source-filtered panel via ``fu3w.panel_name_for`` (the #1090
    fu5 / #1481 reread / #1315 parity precedent) so a read at a panel-member
    or content-identical context — bare ``default``, the held-out ``neg_sp_*``
    members of the six-context p6 panel — does not trip the TRAINING-time
    #527/#538 disjointness invariant at ModelOrganism construction
    (epm:failure v3: both pods' p6 smoke died at ``cid='default'``). At
    genuine source contexts ``panel_name_for`` returns the default panel name
    unchanged, so p3/p5 source reads construct byte-identically.
    ``negatives.py``'s guard stays byte-untouched — it is correct for
    training-time construction (datagen/mix builds keep the strict panel).
    """
    import issue1090_fu3_worker as fu3w  # local import mirrors panel_context_ids

    # Held-out panel-member read contexts (neg_sp_*) must resolve in ANY unit
    # subprocess regardless of whether panel_context_ids ran first (#1315 r6 /
    # #1090 fu6 registry classes: point-of-use, idempotent).
    for member in default_panel():
        _register_ctx(member.to_context())
    ctx = fu3w.ensure_context(context_id, behavior)
    return ModelOrganism(
        behavior=behavior, context_id=context_id, negatives=fu3w.panel_name_for(ctx), seed=seed
    )


def _eval_questions(cfg: Cfg, beh_key: str) -> list[str]:
    """Held-out eval questions per behavior (content: the BEHAVIORS registry
    bank; marker: the sha-pinned 20-q bank)."""
    if beh_key == "mk":
        return mk1481.eval_questions(_mk_cfg(cfg))
    qs = list(BEHAVIORS[G.BEHAVIOR_BY_KEY[beh_key]].eval_question_bank)
    if cfg.eval_question_limit is not None:
        qs = qs[: cfg.eval_question_limit]
    # p10 split-half attenuation floor: >=2 distinct questions (#1315 r4).
    assert len(qs) >= 2, f"need >=2 eval questions for {beh_key}, got {len(qs)}"
    return qs


# ── p0: stage + pin every reused input ───────────────────────────────────────


def _resolve_revision(repo_id: str, repo_type: str) -> str:
    from huggingface_hub import HfApi

    info = hub.retry_transient(
        lambda: HfApi().repo_info(repo_id, repo_type=repo_type), what=f"repo_info {repo_id}"
    )
    return str(info.sha)


def _stage_model_prefix(prefix: str, dest: Path, *, revision: str) -> Path:
    """Stage a model-repo adapter/checkpoint subfolder via scoped listing +
    per-file download (no staging transform — files land at prefix-relative
    paths; reuse check (h)(iv) 'no staging transformation')."""
    from huggingface_hub import hf_hub_download

    if (dest / "adapter_config.json").exists() or (dest / "config.json").exists():
        return dest
    from huggingface_hub import HfApi as _HfApi

    entries = hub.list_hf_files_under_path(
        _HfApi(), G.HF_MODEL_REPO, prefix, repo_type="model", revision=revision
    )
    if not entries:
        raise FileNotFoundError(f"no files under {G.HF_MODEL_REPO}/{prefix} @ {revision}")
    dest.mkdir(parents=True, exist_ok=True)
    for path_in_repo in entries:
        rel = path_in_repo[len(prefix) :].lstrip("/")
        target = dest / rel
        if target.exists():
            continue
        got = hub.retry_transient(
            lambda p=path_in_repo: hf_hub_download(
                G.HF_MODEL_REPO, p, repo_type="model", revision=revision
            ),
            what=f"hf_hub_download {path_in_repo}",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(got, target)
    return dest


def _staged_arm_dir(cfg: Cfg, arm: G.ReusedLoraArm) -> Path:
    return cfg.out_root / "inputs" / "lora_arms" / arm.cell


def _staged_ft_dir(cfg: Cfg, name: str) -> Path:
    return cfg.out_root / "inputs" / "ft_ckpts" / name


def _mix_local(cfg: Cfg, beh_key: str, regime: str) -> Path:
    return cfg.out_root / "inputs" / "mixes" / f"{beh_key}_{regime}.jsonl"


def _mix_meta_local(cfg: Cfg, beh_key: str, regime: str) -> Path:
    return cfg.out_root / "inputs" / "mixes" / f"{beh_key}_{regime}_meta.json"


def _realized_training_panel(cfg: Cfg, beh_key: str, regime: str) -> list[str]:
    """REALIZED training-panel context ids for one content mix, resolved from
    the mix builder's own staged mix_meta.json via the #1481 MF-3 resolver
    (fail-loud on meta-shape/panel drift; po mixes with zero realized
    negatives resolve to [])."""
    import issue1481_analysis as a1481

    meta = _read_json(_mix_meta_local(cfg, beh_key, regime))
    return a1481.realized_panel_context_ids(meta, G.BEHAVIOR_BY_KEY[beh_key])


def _assert_mix_composition(path: Path, beh_key: str, expected_rows: int) -> dict:
    """Composition asserts on a staged mix (row counts only — harmful-content
    digest discipline: never print row text)."""
    n = 0
    n_marker = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            if beh_key == "mk":
                comp = row.get("completion") or row.get("response") or ""
                if isinstance(comp, list):
                    comp = json.dumps(comp, ensure_ascii=False)
                n_marker += int(MARKER_TEXT.strip() in comp)
    if n != expected_rows:
        raise RuntimeError(f"mix {path.name}: {n} rows != expected {expected_rows}")
    rec = {"rows": n, "sha256": _sha256_file(path)}
    if beh_key == "mk":
        # 200:800 pos:neg convention (#1112/#1333) — positives carry the marker.
        rec["rows_with_marker"] = n_marker
        if not (0 < n_marker < n):
            raise RuntimeError(f"marker mix {path.name}: degenerate marker rows {n_marker}/{n}")
    return rec


def _grounding_from_adapter_config(dest: Path, recipe_class: str) -> dict:
    """Recipe grounding on the artifact's OWN adapter_config.json (#545 — the
    config wins over any body row) + the gauge assert (no lm_head/embed)."""
    cfg_path = dest / "adapter_config.json"
    ac = json.loads(cfg_path.read_text())
    tm = set(ac.get("target_modules") or [])
    if {"lm_head", "embed_tokens"} & tm or ac.get("modules_to_save"):
        raise RuntimeError(f"gauge violation in {cfg_path}: {sorted(tm)}")
    if not ac.get("use_rslora", False):
        raise RuntimeError(f"{cfg_path}: expected use_rslora=true ({recipe_class} class)")
    expect_r = {"content": 32, "marker": 16}[recipe_class]
    if int(ac.get("r", -1)) != expect_r:
        raise RuntimeError(f"{cfg_path}: r={ac.get('r')} != {expect_r} ({recipe_class} class)")
    return {
        "r": ac.get("r"),
        "lora_alpha": ac.get("lora_alpha"),
        "use_rslora": ac.get("use_rslora"),
        "n_target_modules": len(tm),
        "recipe_class": recipe_class,
    }


def _arms_in_scope(cfg: Cfg) -> list[G.ReusedLoraArm]:
    return [G.lora_pair_of(c) for c in cfg.cells]


def phase_stage(cfg: Cfg) -> dict:
    _phase("p0_stage")
    _headroom(cfg, "p0_stage")
    done_path = cfg.out_root / "stage_done.json"
    if done_path.exists():
        return _read_json(done_path)
    _assert_marker_token()
    rec: dict = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    data_rev = _resolve_revision(G.HF_DATA_REPO, "dataset")
    model_rev = _resolve_revision(G.HF_MODEL_REPO, "model")
    rec["pins"] = {"data_repo": data_rev, "model_repo": model_rev}

    # 1) frozen mixes (consumer-exact local paths; no layout transform).
    mixes: dict[str, dict] = {}
    needed = sorted({(G.parse_ft_cell(c)[0], G.parse_ft_cell(c)[1]) for c in cfg.cells})
    for beh_key, regime in needed:
        path_in_repo, n_rows = G.MIXES[beh_key][regime]
        dest = _mix_local(cfg, beh_key, regime)
        _stage_file(path_in_repo, dest, revision=data_rev)
        mixes[f"{beh_key}_{regime}"] = {
            "path_in_repo": path_in_repo,
            **_assert_mix_composition(dest, beh_key, n_rows),
        }
        # consumer-open probe (mix family x trainer-jsonl consumer): parse row 1.
        with dest.open(encoding="utf-8") as f:
            json.loads(next(iter(f)))
        if beh_key != "mk":
            # sibling mix_meta.json (Hub-verified 2026-07-22 beside all 6
            # content mixes) — the REALIZED training-panel source for the
            # §4.5 held-out-only leakage decomposition (review r1 Minor 7;
            # the #1481 MF-3 convention). Consumer-open: resolve the realized
            # panel NOW so a meta-shape drift fails at p0, not in p6.
            meta_dest = _mix_meta_local(cfg, beh_key, regime)
            _stage_file(
                f"{path_in_repo.rsplit('/', 1)[0]}/mix_meta.json", meta_dest, revision=data_rev
            )
            mixes[f"{beh_key}_{regime}"]["realized_panel"] = _realized_training_panel(
                cfg, beh_key, regime
            )
    rec["mixes"] = mixes

    # marker ICL bank -> the exact path mk1481._icl_context opens
    # (<out_root>/inputs/icl_examples_marker.json; reuse leg (h)(ii)).
    if any(b == "mk" for b, _r in needed):
        _stage_file(
            G.MARKER_ICL_BANK_PATH,
            cfg.out_root / "inputs" / "icl_examples_marker.json",
            revision=data_rev,
        )
        mk1481.panel_contexts(_mk_cfg(cfg))  # consumer-open probe (6-ctx panel)
        rec["marker_icl_bank"] = "staged + panel_contexts consumer-open OK"

    # 2) reused LoRA arms + adapter_config grounding + consumer-open probes.
    arms: dict[str, dict] = {}
    probed_class: set[str] = set()
    for arm in _arms_in_scope(cfg):
        dest = _staged_arm_dir(cfg, arm)
        _stage_model_prefix(arm.subfolder, dest, revision=model_rev)
        arms[arm.cell] = _grounding_from_adapter_config(dest, arm.recipe_class)
        if arm.recipe_class not in probed_class:
            # staging probe + consumer-open per (family x consumer): the PEFT
            # loader is the read-side consumer of the adapter family.
            from peft import PeftConfig

            PeftConfig.from_pretrained(str(dest))
            probed_class.add(arm.recipe_class)
            arms[arm.cell]["consumer_open"] = "PeftConfig.from_pretrained OK"
    rec["lora_arms"] = arms

    # 3) reused FT checkpoints (overflow repo) — smoke skips the 15 GB pulls
    # unless the smoke cell is the reused cell (it is not: syc-con-s137).
    overflow_rev = None
    if G.REUSED_FT_CELL in cfg.cells:
        overflow_rev = _resolve_revision(G.OVERFLOW_REPO, "model")
        _stage_overflow_prefix(
            G.REUSED_FT_SUBFOLDER, _staged_ft_dir(cfg, "s3_con"), revision=overflow_rev
        )
        _stage_overflow_prefix(
            G.PARITY_XCHECK_SUBFOLDER, _staged_ft_dir(cfg, "s4_po_xcheck"), revision=overflow_rev
        )
        from transformers import AutoConfig

        AutoConfig.from_pretrained(str(_staged_ft_dir(cfg, "s3_con")))  # consumer-open
        rec["reused_ft"] = {"revision": overflow_rev, "consumer_open": "AutoConfig OK"}
    _atomic_json(done_path, rec)
    return rec


# ── p1: reused-arm parity gates (WARN-class; HALT only structural) ───────────


def run_parity_unit(cfg: Cfg, cell: str) -> dict:
    """Apply-and-read re-read of the FT cell's paired LoRA arm on THIS rig.

    Content: Tier-1-shape judged read at the verdict rung, WARN outside
    |Δrate| <= 0.15 (#1481 P1). Marker: slot-read ΔG, WARN outside ±1.0 nat
    (#1333 drift calibration). Values PERSIST either way + a named analyzer
    adjudication (gate-calibration rule); a load/apply failure raises (HALT).
    """
    arm = G.lora_pair_of(cell)
    out_path = cfg.out_root / cell / "parity.json"
    if out_path.exists():
        return _read_json(out_path)
    (cfg.out_root / cell).mkdir(parents=True, exist_ok=True)
    staged = _staged_arm_dir(cfg, arm)
    merged = _merge_adapter(cfg, str(staged), cfg.out_root / cell / "merged_parity")
    try:
        if arm.recipe_class == "marker":
            read = _marker_source_read(cfg, str(merged), cfg.out_root / cell / "parity_rate")
            delta = read["delta_logp_mean"]
            rec = {
                "cell": cell,
                "arm": arm.run_id,
                "kind": "marker_slot",
                "delta_g": delta,
                "expected": arm.anchor,
                "warn_band_nats": G.P1_MARKER_WARN_NATS,
                "rate_window_pass": bool(abs(delta - arm.anchor) <= G.P1_MARKER_WARN_NATS),
            }
        else:
            cid = source_context_id(arm.beh_key)
            panel_context_ids(cfg, arm.beh_key)  # registers cid idempotently
            organism = _read_organism(G.BEHAVIOR_BY_KEY[arm.beh_key], cid, arm.seed)
            rate_fn = make_source_rate_fn(
                organism,
                out_dir=cfg.out_root / cell / "parity_rate",
                eval_questions=_eval_questions(cfg, arm.beh_key),
                n_completions=cfg.tier1_n,
                temperature=1.0,
                n_judge_draws=cfg.tier1_draws,
                judge_fn=fu1._judge_fu1,
            )
            try:
                rate = float(rate_fn(str(merged)))
            finally:
                close = getattr(rate_fn, "close", None)
                if callable(close):
                    close()
            rec = {
                "cell": cell,
                "arm": arm.run_id,
                "kind": "content_tier1",
                "rate": rate,
                "expected": arm.anchor,
                "warn_band": G.P1_PARITY_MAX_ABS_DELTA,
                "rate_window_pass": bool(abs(rate - arm.anchor) <= G.P1_PARITY_MAX_ABS_DELTA),
            }
    finally:
        shutil.rmtree(merged, ignore_errors=True)
    rec["severity"] = "PASS" if rec["rate_window_pass"] else "WARN-analyzer-adjudication"
    rec["adapter_config"] = _grounding_from_adapter_config(staged, arm.recipe_class)
    _atomic_json(out_path, rec)
    _mirror_deliverable(cfg, f"parity_{cell}", rec)
    return rec


def phase_parity(cfg: Cfg) -> dict:
    _phase("p1_parity")
    pending = [c for c in cfg.cells if not (cfg.out_root / c / "parity.json").exists()]
    if pending:
        if _n_gpus() == 1 or len(pending) == 1:
            for c in pending:
                run_parity_unit(cfg, c)
        else:
            _fanout_units(cfg, [_unit_args(cfg, "parity", c) for c in pending])
    return {c: _read_json(cfg.out_root / c / "parity.json") for c in cfg.cells}


# ── p2: FT training (content quads + marker grid) ────────────────────────────


def _ft_lane_env(lane: int) -> dict[str, str]:
    """EXPLICIT CVD over one 4-GPU quad + distinct MASTER_PORT (plan §4.8;
    the launcher-env CVD pin — gotchas CVD-clobber family; #1112 shape b)."""
    ids = _physical_gpu_ids()
    if len(ids) >= 8:
        quad = ids[lane * 4 : lane * 4 + 4]
    else:
        if len(ids) < FT_NUM_PROCESSES:
            raise RuntimeError(
                f"full-FT needs {FT_NUM_PROCESSES} GPUs (ZeRO-3 world size) but only "
                f"{len(ids)} are visible"
            )
        quad = ids[:FT_NUM_PROCESSES]
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(quad),
        "MASTER_PORT": str(29500 + lane),
    }


def _content_ft_cmd(
    cfg: Cfg, cell: str, *, out_dir: Path, max_steps: int, ckpt_steps: Sequence[int]
) -> list[str]:
    """train_behavior_fullft.py launch — values byte-inherited from #1112
    (constants imported from experiments.issue_1112, never retyped). Width is
    smoke-INVARIANT (--num_processes 4; #1315/#1333 smoke-width lesson)."""
    beh, _regime, seed = G.parse_ft_cell(cell)
    return [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--config_file",
        ACCEL_CONFIG,
        "--num_processes",
        str(FT_NUM_PROCESSES),
        FT_TRAINER,
        "--behavior",
        G.BEHAVIOR_BY_KEY[beh],
        "--arm",
        "ft",
        "--train-jsonl",
        str(_mix_local(cfg, beh, _regime)),
        "--output-dir",
        str(out_dir),
        "--ckpt-steps",
        ",".join(str(s) for s in ckpt_steps),
        "--max-steps",
        str(max_steps),
        "--learning-rate",
        str(P1112.FT_LR),
        "--epochs",
        "16",  # ceiling; --max-steps caps (the #1112 seam)
        "--per-device-batch",
        str(P1112.FT_PER_DEVICE_BATCH),
        "--grad-accum",
        str(P1112.FT_GRAD_ACCUM),
        "--warmup-ratio",
        str(P1112.FT_WARMUP_RATIO),
        "--max-length",
        str(FT_MAX_LENGTH),
        "--seed",
        str(seed),
        "--wandb-project",
        G.WANDB_PROJECT,
        # Per-cell suffix -> distinct WandB run per cell (#480 run-separation).
        "--run-name-suffix",
        f"i1586_{cell}",
    ]


def ft_wandb_run_name(cell: str) -> str:
    """The realized trainer run name (train_behavior_fullft.py:638-640)."""
    beh, _r, seed = G.parse_ft_cell(cell)
    if beh == "mk":
        return f"issue1586_mk_fullft_{cell}"
    return f"issue642_ft_{G.BEHAVIOR_BY_KEY[beh]}_seed{seed}_i1586_{cell}"


def _marker_ft_cmd(cfg: Cfg, cell: str, *, out_dir: Path, grid: Sequence[int]) -> list[str]:
    _beh, regime, seed = G.parse_ft_cell(cell)
    return [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--config_file",
        MARKER_ACCEL_CONFIG,
        "--num_processes",
        str(FT_NUM_PROCESSES),
        MARKER_FT_TRAINER,
        "--train-jsonl",
        str(_mix_local(cfg, "mk", regime)),
        "--output-dir",
        str(out_dir),
        "--ckpt-steps",
        ",".join(str(s) for s in grid),
        "--max-steps",
        str(max(grid)),
        "--seed",
        str(seed),
        "--learning-rate",
        str(P1112.MARKER_FT_LR),
        "--max-length",
        str(MARKER_MAX_LENGTH),
        "--wandb-project",
        G.WANDB_PROJECT,
        "--run-name",
        ft_wandb_run_name(cell),
    ]


def _train_one_cell(cfg: Cfg, cell: str, lane: int) -> subprocess.Popen | None:
    """Launch one cell's FT subprocess on a lane (returns the Popen for the
    concurrent-quads path; the caller waits)."""
    cell_root = cfg.out_root / cell
    build_path = cell_root / "build_result.json"
    if build_path.exists():
        return None
    out_dir = cell_root / "train"
    if out_dir.exists():
        logger.warning("[ft-launch] clearing stale partial FT out_dir %s", out_dir)
        shutil.rmtree(out_dir)
    if _is_marker(cell):
        grid = (2,) if cfg.smoke else G.MARKER_FT_GRID
        cmd = _marker_ft_cmd(cfg, cell, out_dir=out_dir, grid=grid)
    else:
        max_steps = 2 if cfg.smoke else G.CONTENT_STEP_CEILING
        ckpts = (2,) if cfg.smoke else P1112.FT_CKPT_STEPS
        cmd = _content_ft_cmd(cfg, cell, out_dir=out_dir, max_steps=max_steps, ckpt_steps=ckpts)
    env = _ft_lane_env(lane)
    log = cell_root / "train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[ft-launch] cell=%s lane=%d CVD=%s MASTER_PORT=%s",
        cell,
        lane,
        env["CUDA_VISIBLE_DEVICES"],
        env["MASTER_PORT"],
    )
    f = open(log, "a")  # noqa: SIM115 — held for the Popen's lifetime
    return subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )


def _await_train(cfg: Cfg, cell: str, proc: subprocess.Popen) -> None:
    rc = proc.wait()
    log = cfg.out_root / cell / "train.log"
    if rc != 0:
        # inner-log tail echo (#1333 diagnosability rule)
        tail = ""
        if log.exists():
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-120:])
        logger.error("[ft-launch] cell %s rc=%d — log tail:\n%s", cell, rc, tail)
        raise RuntimeError(f"FT training failed for {cell} rc={rc} (log {log})")
    beh, regime, _s = G.parse_ft_cell(cell)
    mix = _mix_local(cfg, beh, regime)
    _atomic_json(
        cfg.out_root / cell / "build_result.json",
        {
            "cell": cell,
            "status": "trained",
            "adapter_root": str(cfg.out_root / cell / "train"),
            "mix": str(mix),
            "mix_sha256": _sha256_file(mix),
            "wandb_run_name": ft_wandb_run_name(cell),
        },
    )


def phase_train(cfg: Cfg) -> dict:
    _phase("p2_train")
    _headroom(cfg, "p2_train")
    pending = [
        c
        for c in cfg.cells
        if not (cfg.out_root / c / "build_result.json").exists() and c != G.REUSED_FT_CELL
    ]
    n_lanes = 2 if len(_physical_gpu_ids()) >= 8 else 1
    t0 = time.time()
    first_cell_wall: float | None = None
    while pending:
        batch = pending[:n_lanes]
        pending = pending[n_lanes:]
        procs = {}
        for lane, cell in enumerate(batch):
            p = _train_one_cell(cfg, cell, lane)
            if p is not None:
                procs[cell] = p
        for cell, p in procs.items():
            _await_train(cfg, cell, p)
        if first_cell_wall is None:
            first_cell_wall = time.time() - t0
            _pilot_gate(
                cfg,
                label="p2_train",
                unit_wall_s=first_cell_wall,
                n_units=len([c for c in cfg.cells if c != G.REUSED_FT_CELL]),
                parallelism=PILOT_PARALLELISM,
                plan_wall_h=PILOT_PLAN_P2_WALL_H,
            )
    return {
        c: _read_json(cfg.out_root / c / "build_result.json")
        for c in cfg.cells
        if (cfg.out_root / c / "build_result.json").exists()
    }


def _pilot_gate(
    cfg: Cfg,
    *,
    label: str,
    unit_wall_s: float,
    n_units: int,
    parallelism: float,
    plan_wall_h: float,
) -> None:
    """Compute-pilot kill (plan §7 item 2): unit 1 of a pilot-gated phase is
    the measured pilot; >2x re-projection against the phase's OWN §9 row
    HALTs with a report JSON + rc=7 (a DESIGNED artifact-routed halt — never
    a bare rc=1; gotchas #1415). Per-phase gates: p2 vs the p2 row, p3 vs
    the p3+p3b rows (review r1 Minor 5)."""
    if cfg.smoke:
        return
    projected_h = n_units * (unit_wall_s / 3600.0) / max(parallelism, 1.0)
    rec = {
        "label": label,
        "measured_unit_wall_s": unit_wall_s,
        "n_units": n_units,
        "parallelism": parallelism,
        "projected_wall_h": projected_h,
        "plan_wall_h": plan_wall_h,
        "ratio": projected_h / plan_wall_h,
        "verdict": "PASS" if projected_h <= 2 * plan_wall_h else "HALT",
    }
    _atomic_json(cfg.out_root / f"pilot_gate_report_{label}.json", rec)
    logger.info("[pilot-gate] %s", json.dumps(rec))
    if rec["verdict"] == "HALT":
        raise SystemExit(PILOT_GATE_RC)


# ── p3: ladders + anchor-nearest selection ───────────────────────────────────


def _reap_rungs(train_dir: Path, keep_steps: set[int]) -> int:
    """Delete non-kept rung dirs (declared discard, plan §10 — rates persist
    in ladder.json; selected rung re-derivable by deterministic retrain)."""
    rungs = _enumerate_rungs(train_dir)
    n = 0
    for step, p in rungs.items():
        if step not in keep_steps:
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    return n


def _marker_source_read(cfg: Cfg, model_path: str, out_dir: Path) -> dict:
    """Marker source slot read at the pers context: greedy 20-q gens (vLLM,
    max_new 2048) -> strip-at-marker -> four-float slot reads trained AND base
    (compute_marker_slot_stats via d1112._marker_slot_read). Persists rollout
    text BEFORE reducing (#779)."""
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import _generate_responses_vllm

    out_dir.mkdir(parents=True, exist_ok=True)
    mkcfg = _mk_cfg(cfg)
    questions = mk1481.eval_questions(mkcfg)
    src = mk1481.source_context(mkcfg, "pers")
    rows = _generate_responses_vllm(
        model_path,
        {src.context_id: src.system},
        questions,
        max_new_tokens=G.MAX_NEW_TOKENS_MARKER,
        gpu_memory_utilization=CAPTURE_GPU_MEM_UTIL,
        user_wraps={src.context_id: src.user_wrap},
    )
    tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    contexts, emitted = [], []
    for r in rows:
        stripped, emit = d1333._strip_at_marker(r["response"])
        contexts.append(tok.decode(r["prompt_token_ids"]) + stripped)
        emitted.append(bool(emit))
    (out_dir / "rollouts.json").write_text(
        json.dumps({"model": model_path, "rows": rows}, ensure_ascii=False)
    )
    trained = _marker_slot_read(model_path, contexts, device="cuda:0")
    base = _marker_slot_read(DEFAULT_BASE_MODEL, contexts, device="cuda:0")
    deltas = [t["logp"] - b["logp"] for t, b in zip(trained, base, strict=True)]
    margins = [
        (t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"])
        for t, b in zip(trained, base, strict=True)
    ]
    argmax_rate = sum(int(t.get("argmax_id") == MARKER_TOKEN_ID) for t in trained) / len(trained)
    rec = {
        "delta_logp_mean": float(sum(deltas) / len(deltas)),
        "delta_margin_mean": float(sum(margins) / len(margins)),
        "gen_emission_rate": float(sum(emitted) / len(emitted)),
        "argmax_rate": float(argmax_rate),
        "n": len(contexts),
        "slot_reads": {"trained": trained, "base": base},
    }
    _atomic_json(out_dir / "slot_read.json", rec)
    return rec


def run_ladder_unit(cfg: Cfg, cell: str) -> dict:
    """Per-rung ladder for one FT cell (content: Tier-1 judged rate via the
    fu4/#1090 instrument; marker: source ΔG slot read). Per-rung resume; in
    stream-reap mode every judged rung except the latest is deleted right
    after its read (the #1112 coarse+refine contingency)."""
    cell_root = cfg.out_root / cell
    ladder_path = cell_root / "ladder.json"
    train_dir = Path(_read_json(cell_root / "build_result.json")["adapter_root"])
    done: dict[int, dict] = {}
    if ladder_path.exists():
        prior = _read_json(ladder_path)
        if prior.get("regime") != cfg.regime_key():
            raise RuntimeError(f"ladder regime drift under {ladder_path} — fresh --out-root")
        done = {int(k): v for k, v in (prior.get("reads_by_step") or {}).items()}

    def _persist() -> None:
        _atomic_json(
            ladder_path,
            {
                "cell": cell,
                "regime": cfg.regime_key(),
                "reads_by_step": {str(k): v for k, v in sorted(done.items())},
            },
        )

    stream_reap = cfg.resolved_disk_mode() == "stream-reap" and not cfg.smoke
    if _is_marker(cell):
        for step, rung in sorted(_enumerate_rungs(train_dir).items()):
            if step in done:
                continue
            read = _marker_source_read(cfg, str(rung), cell_root / f"rung{step}")
            done[step] = {
                k: read[k]
                for k in (
                    "delta_logp_mean",
                    "delta_margin_mean",
                    "gen_emission_rate",
                    "argmax_rate",
                )
            }
            _persist()
            if stream_reap and step != max(_enumerate_rungs(train_dir)):
                shutil.rmtree(rung, ignore_errors=True)
    else:
        beh = G.parse_ft_cell(cell)[0]
        cid = source_context_id(beh)
        panel_context_ids(cfg, beh)  # idempotent point-of-use registration
        pendings = [s for s in sorted(_enumerate_rungs(train_dir)) if s not in done]
        if pendings:
            organism = _read_organism(G.BEHAVIOR_BY_KEY[beh], cid, G.parse_ft_cell(cell)[2])
            rate_fn = make_source_rate_fn(
                organism,
                out_dir=cell_root / "rate",
                eval_questions=_eval_questions(cfg, beh),
                n_completions=cfg.tier1_n,
                temperature=1.0,
                n_judge_draws=cfg.tier1_draws,
                judge_fn=fu1._judge_fu1,
            )
            try:
                for step in pendings:
                    rung = _enumerate_rungs(train_dir)[step]
                    _ensure_dir_tokenizer(rung)
                    done[step] = {"rate": float(rate_fn(str(rung)))}
                    _persist()
                    if stream_reap and step != max(_enumerate_rungs(train_dir)):
                        shutil.rmtree(rung, ignore_errors=True)
            finally:
                close = getattr(rate_fn, "close", None)
                if callable(close):
                    close()
    _persist()
    return done


def _select_cell(cfg: Cfg, cell: str) -> dict:
    """Anchor-nearest selection for one cell (plan §4.3) + rung reap."""
    cell_root = cfg.out_root / cell
    sel_path = cell_root / "selection.json"
    if sel_path.exists():
        return _read_json(sel_path)
    arm = G.lora_pair_of(cell)
    reads = {int(k): v for k, v in _read_json(cell_root / "ladder.json")["reads_by_step"].items()}
    if _is_marker(cell):
        metric = {s: float(v["delta_logp_mean"]) for s, v in reads.items()}
        # de-saturation gates (plan §4.3): source gen-emission 0 + argmax
        # below the 0.92 ceiling at the rung.
        eligible = {
            s
            for s, v in reads.items()
            if float(v["gen_emission_rate"]) == 0.0 and float(v["argmax_rate"]) < G.ARGMAX_CEILING
        }
        sel = G.select_anchor_nearest(
            metric,
            anchor=arm.anchor,
            band=G.INSTALL_WINDOW,
            eligible_steps=eligible if not cfg.smoke else None,
        )
        sel["window"] = list(G.INSTALL_WINDOW)
    else:
        metric = {s: float(v["rate"]) for s, v in reads.items()}
        sel = G.select_anchor_nearest(metric, anchor=arm.anchor, band=G.JUDGED_RATE_BAND)
        sel["band"] = list(G.JUDGED_RATE_BAND)
    sel["cell"] = cell
    sel["paired_arm"] = arm.run_id
    sel["reads_by_step"] = {str(k): v for k, v in sorted(reads.items())}
    # Between-cell rung reap (plan §9): keep selected + latest only. In
    # stream-reap mode the selected rung may already be gone -> deterministic
    # retrain to the selected step (the #1112 coarse+refine contingency).
    train_dir = Path(_read_json(cell_root / "build_result.json")["adapter_root"])
    keep = {int(sel["step"]), max(metric)}
    if not cfg.smoke:
        n_reaped = _reap_rungs(train_dir, keep)
        sel["rungs_reaped"] = n_reaped
    if int(sel["step"]) not in _enumerate_rungs(train_dir):
        sel["retrained_to_step"] = _retrain_to_step(cfg, cell, int(sel["step"]))
    _atomic_json(sel_path, sel)
    _mirror_deliverable(cfg, cell, sel)
    return sel


def _retrain_to_step(cfg: Cfg, cell: str, step: int) -> dict:
    """Deterministic retrain to the selected rung (stream-reap mode; A11) +
    Tier-1 spot re-read parity <=0.10 (content only)."""
    cell_root = cfg.out_root / cell
    out_dir = cell_root / "train_reselect"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if _is_marker(cell):
        cmd = _marker_ft_cmd(cfg, cell, out_dir=out_dir, grid=(step,))
    else:
        cmd = _content_ft_cmd(cfg, cell, out_dir=out_dir, max_steps=step, ckpt_steps=(step,))
    _run_subprocess(cmd, cell_root / "retrain_reselect.log", env=_ft_lane_env(0))
    rung = _enumerate_rungs(out_dir)[step]
    rec: dict = {"step": step, "adapter_root": str(out_dir)}
    if not _is_marker(cell):
        beh = G.parse_ft_cell(cell)[0]
        organism = _read_organism(
            G.BEHAVIOR_BY_KEY[beh], source_context_id(beh), G.parse_ft_cell(cell)[2]
        )
        rate_fn = make_source_rate_fn(
            organism,
            out_dir=cell_root / "reselect_rate",
            eval_questions=_eval_questions(cfg, beh),
            n_completions=cfg.tier1_n,
            temperature=1.0,
            n_judge_draws=cfg.tier1_draws,
            judge_fn=fu1._judge_fu1,
        )
        try:
            rate = float(rate_fn(str(rung)))
        finally:
            close = getattr(rate_fn, "close", None)
            if callable(close):
                close()
        prior = float(_read_json(cell_root / "ladder.json")["reads_by_step"][str(step)]["rate"])
        rec["spot_reread"] = {"rate": rate, "prior": prior, "abs_delta": abs(rate - prior)}
        if abs(rate - prior) > 0.10:
            raise RuntimeError(
                f"stream-reap retrain parity failed for {cell}@{step}: "
                f"|{rate:.3f}-{prior:.3f}| > 0.10"
            )
    # re-point the build record at the retrained tree
    build = _read_json(cell_root / "build_result.json")
    build["adapter_root"] = str(out_dir)
    _atomic_json(cell_root / "build_result.json", build)
    return rec


def phase_ladder(cfg: Cfg) -> dict:
    _phase("p3_ladder")
    _headroom(cfg, "p3_ladder")
    trainable = [c for c in cfg.cells if c != G.REUSED_FT_CELL]
    units = [
        _unit_args(cfg, "ladder", c)
        for c in trainable
        if not (cfg.out_root / c / "ladder_done.json").exists()
    ]
    if units:
        # p3 pilot (review r1 Minor 5): the FIRST ladder unit runs inline,
        # timed, gated against the §9 p3+p3b rows — a full-FT-rung engine
        # churn blowup HALTs here instead of escaping the p2-only gate.
        gate_rep = cfg.out_root / "pilot_gate_report_p3_ladder.json"
        if not cfg.smoke and not gate_rep.exists():
            t0 = time.time()
            run_ladder_unit(cfg, units[0][2])
            _pilot_gate(
                cfg,
                label="p3_ladder",
                unit_wall_s=time.time() - t0,
                n_units=len(units),
                parallelism=float(min(_n_gpus(), 8)),
                plan_wall_h=PILOT_PLAN_P3_WALL_H,
            )
            units = units[1:]
    if units:
        if len(units) == 1 or _n_gpus() == 1:
            for u in units:
                run_ladder_unit(cfg, u[2])
        else:
            _fanout_units(cfg, units)
    for c in trainable:
        if not (cfg.out_root / c / "ladder_done.json").exists():
            _atomic_json(cfg.out_root / c / "ladder_done.json", {"ts": time.time()})
    # Registered one-shot extensions (plan §4.2) BEFORE selection.
    if not cfg.smoke:
        for c in trainable:
            _maybe_extend(cfg, c)
    selections: dict[str, dict] = {}
    for c in trainable:
        selections[c] = _select_cell(cfg, c)
    if G.REUSED_FT_CELL in cfg.cells:
        selections[G.REUSED_FT_CELL] = {
            "cell": G.REUSED_FT_CELL,
            "step": 8,
            "reused": True,
            "subfolder": G.REUSED_FT_SUBFOLDER,
            "in_band": True,
            "fallback": None,
        }
        _atomic_json(
            cfg.out_root / G.REUSED_FT_CELL / "selection.json", selections[G.REUSED_FT_CELL]
        )
        _mirror_deliverable(cfg, G.REUSED_FT_CELL, selections[G.REUSED_FT_CELL])
    return selections


def _maybe_extend(cfg: Cfg, cell: str) -> None:
    """One-shot registered extensions: content 30->60 when no in-band rung;
    marker grid 1-6 -> 7-12 when ΔG@6 < 5 nat (plan §4.2)."""
    cell_root = cfg.out_root / cell
    if (cell_root / "extended.json").exists() or (cell_root / "selection.json").exists():
        return
    reads = {int(k): v for k, v in _read_json(cell_root / "ladder.json")["reads_by_step"].items()}
    train_dir = Path(_read_json(cell_root / "build_result.json")["adapter_root"])
    if _is_marker(cell):
        top = max(reads)
        if float(reads[top]["delta_logp_mean"]) >= G.MARKER_EXT_MIN_DELTA_NATS:
            return
        # EXTENSION grid ONLY (7-12): --ckpt-steps 7..12 / --max-steps 12, so
        # run-A checkpoints 1-6 (whose ladder.json reads stand) are never
        # overwritten by the re-train's different-LR-schedule weights —
        # mirroring the content path, whose extension ckpts start at 32
        # (review r1 Minor 1).
        cmd = _marker_ft_cmd(cfg, cell, out_dir=train_dir, grid=G.MARKER_FT_EXT_GRID)
        log = cell_root / "extend.log"
    else:
        lo, hi = G.JUDGED_RATE_BAND
        if any(lo <= float(v["rate"]) <= hi for v in reads.values()):
            return
        # keep only the latest rung as the resume source (plan §9)
        _reap_rungs(train_dir, {max(reads)})
        cmd = _content_ft_cmd(
            cfg,
            cell,
            out_dir=train_dir,
            max_steps=G.CONTENT_EXT_CEILING,
            ckpt_steps=tuple(range(32, G.CONTENT_EXT_CEILING + 1, 2)),
        )
        log = cell_root / "extend.log"
    _run_subprocess(cmd, log, env=_ft_lane_env(0))
    _atomic_json(cell_root / "extended.json", {"ts": time.time()})
    (cell_root / "ladder_done.json").unlink(missing_ok=True)
    run_ladder_unit(cfg, cell)  # ladder the extension rungs (per-rung resume)
    _atomic_json(cell_root / "ladder_done.json", {"ts": time.time()})


# ── fan-out (work-conserving CVD-pinned subprocess pool; #1112 pattern) ──────


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
            # ALWAYS the resolved LITERAL (never "auto") so unit subprocesses
            # can't re-resolve differently mid-run (review r1 Major 2).
            "--ladder-disk-mode",
            cfg.resolved_disk_mode(),
        ]
        + (
            ["--eval-question-limit", str(cfg.eval_question_limit)]
            if cfg.eval_question_limit
            else []
        )
        + ([] if cfg.upload else ["--no-upload"])
    )


def _fanout_units(cfg: Cfg, units: list[list[str]]) -> None:
    """1-GPU self-invocation units, one per free GPU, launcher-env CVD pin +
    matching --gpu-id; whole-group reap on failure. FT launches never route
    here (they own their quads)."""
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
                    str(_SCRIPTS_DIR / "issue1586_dispatch.py"),
                    *extra,
                    "--gpu-id",
                    ids[g],
                ]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ids[g]}
                log = logs / f"unit_{'_'.join(extra[1:3]).replace('/', '_')}_g{g}.log"
                f = open(log, "a")  # noqa: SIM115 — held for the Popen's lifetime
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


# ── p4: persist selected FT rungs + selection records (incremental) ─────────


def _selected_ft_ckpt(cfg: Cfg, cell: str) -> Path:
    if cell == G.REUSED_FT_CELL:
        return _staged_ft_dir(cfg, "s3_con")
    sel = _read_json(cfg.out_root / cell / "selection.json")
    train_dir = Path(_read_json(cfg.out_root / cell / "build_result.json")["adapter_root"])
    ckpt = _enumerate_rungs(train_dir)[int(sel["step"])]
    _ensure_dir_tokenizer(ckpt)
    return ckpt


def _stage_for_upload(stage: Path, src: Path, rel: str) -> None:
    """Hardlink (fallback copy) one file into the batched-commit staging tree
    at its DATA_PREFIX-relative path (review r1 Major 3: uploads go out as a
    handful of upload_folder commits, never per-file loops — #664/#1547)."""
    if not src.exists() or not src.is_file():
        return
    dst = stage / rel
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage_tree_for_upload(stage: Path, src_dir: Path, rel: str) -> None:
    """Stage every file under src_dir (recursive) at rel/<relative path>."""
    if not src_dir.exists():
        return
    for p in sorted(src_dir.rglob("*")):
        if p.is_file():
            _stage_for_upload(stage, p, f"{rel}/{p.relative_to(src_dir)}")


def _stage_has_files(stage: Path) -> bool:
    return stage.exists() and any(p.is_file() for p in stage.rglob("*"))


def phase_persist(cfg: Cfg, selections: dict) -> dict:
    _phase("p4_persist")
    rec_path = cfg.out_root / "persist_done.json"
    if rec_path.exists():
        return _read_json(rec_path)
    uploaded: dict[str, str] = {}
    if cfg.upload:
        for cell in cfg.cells:
            if cell == G.REUSED_FT_CELL or cell not in selections:
                continue
            step = int(selections[cell]["step"])
            ckpt = _selected_ft_ckpt(cfg, cell)
            url = hub._upload(ckpt, G.OVERFLOW_REPO, "model", f"issue1586/{cell}/checkpoint-{step}")
            if not url:
                raise RuntimeError(f"selected-rung upload returned no path for {cell}")
            uploaded[cell] = str(url)
        # Selection records: ONE batched upload_folder commit for ALL cells
        # (was 3 file commits x 16 cells — review r1 Major 3 bug-class sweep).
        stage = cfg.out_root / "_upload_stage" / "p4_records"
        for cell in cfg.cells:
            for name in ("selection.json", "ladder.json", "parity.json"):
                _stage_for_upload(stage, cfg.out_root / cell / name, f"selection/{cell}/{name}")
        if _stage_has_files(stage):
            uploaded["__records__"] = _upload_with_transport_retry(stage, G.DATA_PREFIX)
    _atomic_json(rec_path, {"uploaded": uploaded})
    return {"uploaded": uploaded}


# ── p5: Tier-2 confirm + reused-FT parity + dose labels ──────────────────────


def _content_rate(
    cfg: Cfg,
    *,
    behavior: str,
    context_id: str,
    seed: int,
    model_path: str,
    out_dir: Path,
    n: int,
    draws: int,
    questions: list[str],
) -> float:
    organism = _read_organism(behavior, context_id, seed)
    rate_fn = make_source_rate_fn(
        organism,
        out_dir=out_dir,
        eval_questions=questions,
        n_completions=n,
        temperature=1.0,
        n_judge_draws=draws,
        judge_fn=fu1._judge_fu1,
    )
    try:
        return float(rate_fn(model_path))
    finally:
        close = getattr(rate_fn, "close", None)
        if callable(close):
            close()


def _parity_failed_cells(cfg: Cfg) -> list[str]:
    """Cells whose reused-FT parity re-read FAILED (plan §7 item 3) — the
    analyzer's mechanical exclusion list (review r1 Minor 6). Written only by
    run_tier2_unit for the reused cell (single writer)."""
    p = cfg.out_root / "parity_failed_cells.json"
    return list(_read_json(p).get("cells", [])) if p.exists() else []


def run_tier2_unit(cfg: Cfg, cell: str) -> dict:
    """One content cell's Tier-2 confirm read (fan-out unit kind ``tier2`` —
    review r1 Major 4: p5 shards over cells instead of a serial 1-GPU loop).
    ``step`` reads from the cell's persisted selection.json (unit subprocesses
    hold no in-memory selections dict)."""
    res_path = cfg.out_root / cell / "tier2.json"
    if res_path.exists():
        return _read_json(res_path)
    beh, _regime, seed = G.parse_ft_cell(cell)
    panel_context_ids(cfg, beh)  # point-of-use registration
    ckpt = _selected_ft_ckpt(cfg, cell)
    sel_path = cfg.out_root / cell / "selection.json"
    step = int(_read_json(sel_path)["step"]) if sel_path.exists() else -1
    rate = _content_rate(
        cfg,
        behavior=G.BEHAVIOR_BY_KEY[beh],
        context_id=source_context_id(beh),
        seed=seed,
        model_path=str(ckpt),
        out_dir=cfg.out_root / cell / "tier2_rate",
        n=cfg.tier2_n,
        draws=cfg.tier2_draws,
        questions=_eval_questions(cfg, beh),
    )
    arm = G.lora_pair_of(cell)
    rec = {
        "cell": cell,
        "step": step,
        "tier2_rate": rate,
        "dose_label": G.content_dose_label(rate, arm),
    }
    if cell == G.REUSED_FT_CELL:
        # fresh re-read parity vs #1112's committed selection (plan §4.4).
        committed = _reused_ft_committed_rate(cfg)
        lo, hi = G.JUDGED_RATE_BAND
        rec["reused_parity"] = {
            "committed": committed,
            "abs_delta": abs(rate - committed),
            "pass": bool(lo <= rate <= hi and abs(rate - committed) <= G.REUSED_FT_PARITY_TOL),
        }
        rec["parity_failed"] = not rec["reused_parity"]["pass"]
        if rec["parity_failed"]:
            # registered contingency (plan §7 item 3): kills the REUSE only —
            # the orchestrator retrains this one cell fresh. Downstream
            # panel/margin records for the cell get stamped parity_failed so
            # the analyzer excludes it MECHANICALLY (review r1 Minor 6).
            logger.error("[tier2] reused FT parity FAILED: %s", rec["reused_parity"])
            failed = sorted({*_parity_failed_cells(cfg), cell})
            _atomic_json(cfg.out_root / "parity_failed_cells.json", {"cells": failed})
    _atomic_json(res_path, rec)
    return rec


def phase_tier2(cfg: Cfg, selections: dict) -> dict:
    _phase("p5_tier2")
    content = [c for c in cfg.cells if not _is_marker(c)]
    # marker install confirm IS the slot-read ladder (§4.3) — no mk cells here.
    pending = [c for c in content if not (cfg.out_root / c / "tier2.json").exists()]
    if pending:
        if len(pending) == 1 or _n_gpus() == 1:
            for c in pending:
                run_tier2_unit(cfg, c)
        else:
            _fanout_units(cfg, [_unit_args(cfg, "tier2", c) for c in pending])
    out: dict[str, dict] = {c: _read_json(cfg.out_root / c / "tier2.json") for c in content}
    # #1112 po checkpoint: parity cross-check ROW only (never a contrast arm).
    xcheck_path = cfg.out_root / "xcheck_s4_po.json"
    if (
        G.REUSED_FT_CELL in cfg.cells
        and not cfg.smoke
        and not xcheck_path.exists()
        and _staged_ft_dir(cfg, "s4_po_xcheck").exists()
    ):
        rate = _content_rate(
            cfg,
            behavior="sycophancy",
            context_id=source_context_id("syc"),
            seed=42,
            model_path=str(_staged_ft_dir(cfg, "s4_po_xcheck")),
            out_dir=cfg.out_root / "xcheck_s4_rate",
            n=cfg.tier2_n,
            draws=cfg.tier2_draws,
            questions=_eval_questions(cfg, "syc"),
        )
        _atomic_json(
            xcheck_path,
            {
                "tier2_rate": rate,
                "committed": G.PARITY_XCHECK_COMMITTED_TIER2,
                "note": "parity cross-check row ONLY (plan §4.1 — not a contrast arm)",
            },
        )
    return out


def _reused_ft_committed_rate(cfg: Cfg) -> float:
    """#1112's committed selection rate for s3_fullft_neg (staged at p0 from
    the data repo; A4)."""
    dest = cfg.out_root / "inputs" / "s3_committed_selection.json"
    if not dest.exists():
        _stage_file(
            G.REUSED_FT_COMMITTED_SELECTION,
            dest,
            revision=_resolve_revision(G.HF_DATA_REPO, "dataset"),
        )
    rec = _read_json(dest)
    for k in ("rate", "tier2_rate", "metric"):
        if k in rec:
            return float(rec[k])
    raise RuntimeError(f"no rate field in {dest} (keys: {sorted(rec)})")


# ── p6: six-context leakage panel (both arms, fresh, in-run) ─────────────────


def _panel_arms(cfg: Cfg) -> list[tuple[str, str]]:
    """[(arm_id, kind)] — every FT cell + its paired LoRA arm (kind ft|lora)."""
    arms: list[tuple[str, str]] = []
    for cell in cfg.cells:
        arms.append((cell, "ft"))
        arms.append((G.lora_pair_of(cell).cell, "lora"))
    return arms


def _resolve_arm_model(cfg: Cfg, arm_id: str, kind: str) -> tuple[str, Path | None]:
    """(model_path, merged_dir_to_cleanup) for one panel/margin/capture arm."""
    if kind == "lora":
        arm = G.LORA_ARM_BY_CELL[arm_id]
        merged = _merge_adapter(
            cfg, str(_staged_arm_dir(cfg, arm)), cfg.out_root / arm_id / "merged_panel"
        )
        return str(merged), merged
    return str(_selected_ft_ckpt(cfg, arm_id)), None


def run_panel_unit(cfg: Cfg, arg: str) -> dict:
    """One arm's six-context panel read. Content: judged rate per context
    (pooled non-source = the leakage DV). Marker: slot reads per context
    (four floats; EOS-margin ΔG is the registered lattice DV)."""
    kind, arm_id = arg.split(":", 1)
    out_dir = cfg.out_root / ("marker_panel" if _panel_is_marker(arm_id) else "panel") / arm_id
    res = out_dir / ("slot_reads.json" if _panel_is_marker(arm_id) else "panel_summary.json")
    if res.exists():
        return _read_json(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    beh_key = arm_id.split("-")[0]
    model_path, cleanup = _resolve_arm_model(cfg, arm_id, kind)
    try:
        if beh_key == "mk":
            rec = _marker_panel_read(cfg, arm_id, model_path, out_dir)
        else:
            ctx_ids = panel_context_ids(cfg, beh_key)
            seed = int(arm_id.split("-s")[-1])
            per_ctx: dict[str, float] = {}
            for cid in ctx_ids:
                per_ctx[cid] = _content_rate(
                    cfg,
                    behavior=G.BEHAVIOR_BY_KEY[beh_key],
                    context_id=cid,
                    seed=seed,
                    model_path=model_path,
                    out_dir=out_dir / f"rate_{cid}",
                    n=cfg.panel_n,
                    draws=cfg.panel_draws,
                    questions=_eval_questions(cfg, beh_key),
                )
            src = source_context_id(beh_key)
            non_src = [v for k, v in per_ctx.items() if k != src]
            # §4.5 held-out-only decomposition (review r1 Minor 7; #1481 MF-3
            # convention): read contexts DISJOINT from this arm's REALIZED
            # training panel (resolved from the staged mix_meta at p0),
            # asserted mechanically before the held-out-only rate is computed.
            regime = arm_id.split("-")[3]
            realized = set(_realized_training_panel(cfg, beh_key, regime))
            held = [c for c in ctx_ids if c != src and c not in realized]
            overlap = sorted(set(ctx_ids) - {src} - set(held))
            assert set(held).isdisjoint(realized), (arm_id, held, sorted(realized))
            if not held:
                raise RuntimeError(
                    f"[panel] {arm_id}: NO held-out read context — every non-source "
                    f"panel member is in the realized training panel {sorted(realized)}"
                )
            rec = {
                "arm": arm_id,
                "kind": kind,
                "rates_by_context": per_ctx,
                "source_rate": per_ctx[src],
                "pooled_nonsource_rate": float(sum(non_src) / len(non_src)),
                "heldout_contexts": held,
                "train_overlap_contexts": overlap,
                "heldout_only_rate": float(sum(per_ctx[c] for c in held) / len(held)),
                "n_contexts": len(per_ctx),
            }
        if arm_id in _parity_failed_cells(cfg):
            rec["parity_failed"] = True  # mechanical analyzer exclusion (Minor 6)
        _atomic_json(res, rec)
        return rec
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)


def _panel_is_marker(arm_id: str) -> bool:
    return arm_id.split("-")[0] == "mk"


def _marker_panel_read(cfg: Cfg, arm_id: str, model_path: str, out_dir: Path) -> dict:
    """Marker six-context panel: greedy gens (2048) + four-float slot reads
    trained AND base per context; pooled non-source Δ(z_marker − z_eos) is the
    registered H2-lattice DV (log-prob ΔG alongside — plan §4.5)."""
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import _generate_responses_vllm

    mkcfg = _mk_cfg(cfg)
    panel = mk1481.panel_contexts(mkcfg)
    questions = mk1481.eval_questions(mkcfg)
    personas = {cid: ctx.system for cid, ctx in panel.items()}
    user_wraps = {cid: ctx.user_wrap for cid, ctx in panel.items()}
    prior_turns = {cid: tuple(dict(t) for t in ctx.prefix_turns) for cid, ctx in panel.items()}
    rows = _generate_responses_vllm(
        model_path,
        personas,
        questions,
        max_new_tokens=G.MAX_NEW_TOKENS_MARKER,
        gpu_memory_utilization=CAPTURE_GPU_MEM_UTIL,
        user_wraps=user_wraps,
        prior_turns=prior_turns,
    )
    (out_dir / "rollouts.json").write_text(
        json.dumps({"model": model_path, "rows": rows}, ensure_ascii=False)
    )
    tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    contexts, meta = [], []
    for r in rows:
        stripped, emit = d1333._strip_at_marker(r["response"])
        contexts.append(tok.decode(r["prompt_token_ids"]) + stripped)
        meta.append({"context_id": r["persona"], "q": r["question_idx"], "emitted": bool(emit)})
    trained = _marker_slot_read(model_path, contexts, device="cuda:0")
    base = _marker_slot_read(DEFAULT_BASE_MODEL, contexts, device="cuda:0")
    src = source_context_id("mk")
    per_ctx: dict[str, dict] = {}
    for m, t, b in zip(meta, trained, base, strict=True):
        d = per_ctx.setdefault(m["context_id"], {"dg": [], "dmargin": [], "emitted": []})
        d["dg"].append(t["logp"] - b["logp"])
        d["dmargin"].append((t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"]))
        d["emitted"].append(m["emitted"])
    summary = {
        cid: {
            "delta_logp_mean": float(sum(v["dg"]) / len(v["dg"])),
            "delta_margin_mean": float(sum(v["dmargin"]) / len(v["dmargin"])),
            "emission_rate": float(sum(v["emitted"]) / len(v["emitted"])),
            "n": len(v["dg"]),
        }
        for cid, v in per_ctx.items()
    }
    non_src = [v for k, v in summary.items() if k != src]
    return {
        "arm": arm_id,
        "by_context": summary,
        "per_row": {"meta": meta, "trained": trained, "base": base},  # four-float contract
        "pooled_nonsource_delta_margin": float(
            sum(v["delta_margin_mean"] for v in non_src) / len(non_src)
        ),
        "pooled_nonsource_delta_logp": float(
            sum(v["delta_logp_mean"] for v in non_src) / len(non_src)
        ),
    }


def phase_panel(cfg: Cfg) -> dict:
    _phase("p6_panel")
    arms = _panel_arms(cfg)
    units = []
    for arm_id, kind in arms:
        sub = "marker_panel" if _panel_is_marker(arm_id) else "panel"
        res = (
            cfg.out_root
            / sub
            / arm_id
            / ("slot_reads.json" if _panel_is_marker(arm_id) else "panel_summary.json")
        )
        if not res.exists():
            units.append(_unit_args(cfg, "panel", f"{kind}:{arm_id}"))
    if units:
        if len(units) == 1 or _n_gpus() == 1:
            for u in units:
                run_panel_unit(cfg, u[2])
        else:
            _fanout_units(cfg, units)
    return {"n_arms": len(arms)}


# ── p7: teacher-forced fixed-pool margin (content arms; fu4 instrument) ──────


def _margin_pools(cfg: Cfg, beh_key: str) -> tuple[list[dict], list[dict], dict]:
    """Per-behavior FIXED judged +/- pools via the COMMITTED factory loaders
    (plan §4.7 A8: sizes READ from the pool records, never hardcoded).

    Round 1's name-filter discovery over ``fu1-margin-qwen`` was WRONG — that
    prefix holds per-source margin READ records (c3-sycophancy-claude.json),
    not pos/neg pool files (probe 2026-07-22; open concern
    ``margin-pool-discovery-imp-cas``, now closed by delegating to the
    committed per-behavior instruments):

    - sycophancy: the #1112 instrument VERBATIM — pinned C3 datagen sidecars
      + ``fu1.derive_margin_pools_topup``, sha-ASSERTED against the pinned
      fu1 pool record (mirror of ``issue1112_dispatch._margin_pools``).
    - impolite: ``fu3w._behavior_margin_pools`` (the fu3 instrument verbatim,
      c2-impolite-claude/datagen).
    - writing_style: ``issue1434_cells.i1434_margin_pools`` (ws-pers datagen,
      cap 25/25, 15/15 floor with the smoke demotion).

    Pools equalize-down to min(n_pos, n_neg); realized sizes + pool sha
    persist to pools_meta.json (the inputs manifest)."""
    behavior = G.BEHAVIOR_BY_KEY[beh_key]
    dest = cfg.out_root / "inputs" / "margin_pools" / behavior
    dest.mkdir(parents=True, exist_ok=True)
    if behavior == "sycophancy":
        cell_root = dest / "c3_cell"
        for rel in P1112.C3_MARGIN_SIDECARS:
            _stage_file(f"{P1112.C3_CELL_PREFIX}/{rel}", cell_root / rel, revision=P1112.C3_MIX_REV)
        pinned_path = _stage_file(
            f"{P1112.MARGIN_POOLS_PREFIX}/margin/c3-sycophancy-claude.json",
            dest / "fu1_margin_c3.json",
            revision=P1112.MARGIN_POOLS_REV,
        )
        pos, neg, meta = fu1.derive_margin_pools_topup(
            cell_root, BEHAVIORS[behavior], scratch=dest / "_replay"
        )
        pinned_sha = _read_json(pinned_path)["pool"]["pool_sha256"]
        if meta["pool_sha256"] != pinned_sha:
            raise RuntimeError(
                f"margin pool sha mismatch: derived {meta['pool_sha256']} != pinned fu1 "
                f"{pinned_sha} — the re-derived fixed pools do not reproduce #1090's; "
                "refusing a drifted-instrument margin read"
            )
    elif behavior == "writing_style":
        import issue1434_cells as c1434

        pos, neg, meta = c1434.i1434_margin_pools(cfg)
        if pos is None:
            raise RuntimeError(f"writing_style margin pools below floor: {meta}")
    else:
        import issue1090_fu3_worker as fu3w

        pos, neg = fu3w._behavior_margin_pools(cfg, behavior)
        meta = {
            "behavior": behavior,
            "pool_source": "/".join(fu3w.V4_POOL_SOURCE[behavior]),
            "n_pos_raw": len(pos),
            "n_neg_raw": len(neg),
            "pool_sha256": fu1._sha256_json(
                [
                    {
                        k: p[k]
                        for k in ("probe", "answer", "question_id", "variant_id", "request_id")
                    }
                    for p in pos + neg
                ]
            ),
        }
    n = min(len(pos), len(neg))  # equalize-down (the factory pool convention)
    pos, neg = pos[:n], neg[:n]
    meta = {**meta, "behavior": behavior, "pool_n": n}
    _atomic_json(dest / "pools_meta.json", meta)
    return pos, neg, meta


def _margin_setup(cfg: Cfg, beh_key: str):
    """Shared margin-unit setup: pinned pools (smoke-sliced AFTER the pin),
    source ctx, questions, TF contexts."""
    pos, neg, meta = _margin_pools(cfg, beh_key)
    if cfg.smoke:
        pos, neg = pos[:2], neg[:2]  # tiny-real slice AFTER the pool pin
    panel_context_ids(cfg, beh_key)
    ctx = CONTEXTS[source_context_id(beh_key)]
    questions = _eval_questions(cfg, beh_key)
    ctxs = fu4._fu4_margin_contexts(ctx, questions)
    return pos, neg, meta, questions, ctxs


def run_margin_unit(cfg: Cfg, arg: str) -> dict:
    """One margin fan-out unit (review r1 Major 4: p7 shards over arms).

    ``arg`` = ``base:<beh_key>`` (the per-behavior BASE sweep — sequenced as
    its own unit wave BEFORE any arm unit, so concurrent arm units never race
    the shared base file; the #1315 shared-staging-race family) or
    ``<kind>:<arm_id>`` (one arm's trained sweep + aggregate, requiring the
    behavior's base file to already exist). Each unit builds its OWN
    margin_fn (1 engine per CVD-pinned GPU)."""
    head, tail = arg.split(":", 1)
    out_dir = cfg.out_root / "margin"
    out_dir.mkdir(parents=True, exist_ok=True)
    if head == "base":
        beh_key = tail
        base_path = out_dir / f"base_{beh_key}.json"
        pos, neg, _meta, _questions, ctxs = _margin_setup(cfg, beh_key)
        margin_fn = _default_margin_read_fn(DEFAULT_BASE_MODEL)
        try:
            fu4._margin_sweep(margin_fn, None, ctxs, pos, neg, base_path)
        finally:
            close = getattr(margin_fn, "close", None)
            if callable(close):
                close()
        return _read_json(base_path)
    kind, arm_id = head, tail
    rec_path = out_dir / arm_id / "margin.json"
    if rec_path.exists():
        return _read_json(rec_path)
    beh_key = arm_id.split("-")[0]
    pos, neg, meta, questions, ctxs = _margin_setup(cfg, beh_key)
    base_path = out_dir / f"base_{beh_key}.json"
    if not base_path.exists():
        raise RuntimeError(f"margin base sweep missing for {beh_key}: {base_path} (unit ordering)")
    base_reads = _read_json(base_path)
    margin_fn = _default_margin_read_fn(DEFAULT_BASE_MODEL)
    model_path, cleanup = _resolve_arm_model(cfg, arm_id, kind)
    try:
        trained_reads = fu4._margin_sweep(
            margin_fn, model_path, ctxs, pos, neg, out_dir / f"trained_{arm_id}.json"
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
        close = getattr(margin_fn, "close", None)
        if callable(close):
            close()
    rec = {
        "arm": arm_id,
        "kind": kind,
        **{k: v for k, v in meta.items() if k != "files"},
        "smoke_pool_slice": len(pos) if cfg.smoke else None,
        **fu1.aggregate_margin_reads(
            {
                **{f"base__{k}": v for k, v in base_reads.items()},
                **{f"trained__{k}": v for k, v in trained_reads.items()},
            },
            fu1._q_labels(len(questions)),
        ),
    }
    if arm_id in _parity_failed_cells(cfg):
        rec["parity_failed"] = True  # mechanical analyzer exclusion (Minor 6)
    _atomic_json(rec_path, rec)
    return rec


def phase_margin(cfg: Cfg, selections: dict) -> dict:
    _phase("p7_margin")
    arms = [(a, k) for a, k in _panel_arms(cfg) if not _panel_is_marker(a)]
    if not arms:
        return {"skipped": "no content arms in scope"}
    out_dir = cfg.out_root / "margin"
    out_dir.mkdir(parents=True, exist_ok=True)
    behs = sorted({a.split("-")[0] for a, _k in arms})
    # Pre-stage the shared pinned pools ONCE in the parent BEFORE any fan-out
    # (CPU/network only) — concurrent units must never race the staging dest
    # (#1315 shared-staging race; gotchas "Concurrent fan-out units").
    for beh_key in behs:
        _margin_pools(cfg, beh_key)
    # Wave 1: per-behavior BASE sweeps (own units — single writer per file).
    base_pending = [b for b in behs if not (out_dir / f"base_{b}.json").exists()]
    if base_pending:
        if len(base_pending) == 1 or _n_gpus() == 1:
            for b in base_pending:
                run_margin_unit(cfg, f"base:{b}")
        else:
            _fanout_units(cfg, [_unit_args(cfg, "margin", f"base:{b}") for b in base_pending])
    # Wave 2: 8-way arm fan-out (plan §9 p5-p7 row).
    arm_pending = [(a, k) for a, k in arms if not (out_dir / a / "margin.json").exists()]
    if arm_pending:
        if len(arm_pending) == 1 or _n_gpus() == 1:
            for a, k in arm_pending:
                run_margin_unit(cfg, f"{k}:{a}")
        else:
            _fanout_units(cfg, [_unit_args(cfg, "margin", f"{k}:{a}") for a, k in arm_pending])
    return {a: _read_json(out_dir / a / "margin.json") for a, _k in arms}


# ── p8: activation-shift capture (all cells + per-behavior base + TF) ────────


def capture_passes(cfg: Cfg) -> list[tuple[str, str]]:
    """Registered (arm_id|base_<beh>, kind) capture passes — every FT cell,
    every paired LoRA arm, one base pass per behavior in scope (fail-loud on
    an unroutable arm; #546 silent-skip canary)."""
    passes: list[tuple[str, str]] = []
    for arm_id, kind in _panel_arms(cfg):
        passes.append((arm_id, kind))
    for beh in sorted({G.parse_ft_cell(c)[0] for c in cfg.cells}):
        passes.append((f"base_{beh}", "base"))
    return passes


def run_capture_unit(cfg: Cfg, arg: str) -> None:
    """One own-text capture pass: on-policy greedy gen + 28-layer 3-arm TF
    span pooling -> pooled.pt (prefix / context / response arms — the
    standing prefix-AND-context mapping rule; prefix_end='last_user',
    on_seam='snap' — the #1315 r7 BPE-seam lesson)."""
    import torch
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import (
        _generate_responses_vllm,
        _teacher_forced_span_means,
        compute_prompt_spans,
    )

    kind, arm_id = arg.split(":", 1)
    out_dir = cfg.out_root / "capture" / arm_id
    if (out_dir / "pooled.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    beh_key = arm_id.removeprefix("base_").split("-")[0]
    if kind == "base":
        model_path, cleanup = DEFAULT_BASE_MODEL, None
    else:
        model_path, cleanup = _resolve_arm_model(cfg, arm_id, kind)
    try:
        ctx_ids = panel_context_ids(cfg, beh_key)
        panel = {cid: CONTEXTS[cid] for cid in ctx_ids}
        questions = _eval_questions(cfg, beh_key)
        if cfg.smoke:
            # >=2 contexts x >=2 questions (#1112/#1315 smoke floors: the p10
            # split-half ceiling asserts >=2 distinct question ids).
            ctx_ids = ctx_ids[:2]
            panel = {cid: panel[cid] for cid in ctx_ids}
            questions = questions[:2]
            assert len(questions) >= 2, "smoke capture needs >=2 questions (p10 floor)"
        personas = {cid: c.system for cid, c in panel.items()}
        user_wraps = {cid: c.user_wrap for cid, c in panel.items()}
        prior_turns = {cid: tuple(dict(t) for t in c.prefix_turns) for cid, c in panel.items()}
        max_new = G.MAX_NEW_TOKENS_MARKER if beh_key == "mk" else G.MAX_NEW_TOKENS_CONTENT
        rows = _generate_responses_vllm(
            model_path,
            personas,
            questions,
            max_new_tokens=max_new,
            gpu_memory_utilization=CAPTURE_GPU_MEM_UTIL,
            user_wraps=user_wraps,
            prior_turns=prior_turns,
        )
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
        seam_counts = {"prefix": 0, "context": 0}
        for r in rows:
            cid = r["persona"]
            flags: dict[str, bool] = {}
            r["prefix_len"], r["context_len"] = compute_prompt_spans(
                tokenizer,
                personas[cid],
                questions[r["question_idx"]],
                r["prompt_token_ids"],
                prior_messages=list(prior_turns.get(cid) or ()),
                user_wrap=user_wraps.get(cid),
                prefix_end="last_user",
                on_seam="snap",
                seam_flags=flags,
            )
            r["span_seam"] = flags
            seam_counts["prefix"] += int(flags["prefix"])
            seam_counts["context"] += int(flags["context"])
        # rollout text BEFORE the capture reduce (upload policy #779)
        (out_dir / "raw_rows.json").write_text(
            json.dumps(
                {"model": model_path, "span_seam_counts": seam_counts, "rows": rows},
                ensure_ascii=False,
            )
        )
        pooled = _teacher_forced_span_means(
            model_path,
            rows,
            list(panel),
            layers=list(range(G.N_LAYERS)),
            device="cuda:0",
            dtype=torch.bfloat16,
            tf_batch_size=G.TF_BATCH_SIZE,
        )
        store = {
            "schema_version": 1,
            "cell": arm_id,
            # p10 store contract (issue_1112.geometry.load_store requires "dose")
            "dose": "base" if kind == "base" else "selected",
            "kind": kind,
            "behavior": G.BEHAVIOR_BY_KEY[beh_key],
            "model_path": model_path,
            "row_meta": [
                {"context_id": r["persona"], "question_idx": r["question_idx"]} for r in rows
            ],
            "arms": {
                arm: {li: t.to(torch.float16) for li, t in per_layer.items()}
                for arm, per_layer in pooled.items()
            },
            "metadata": {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_commit": _git_commit(),
                "max_new_tokens": max_new,
                "tf_batch_size": G.TF_BATCH_SIZE,
                "prefix_end": "last_user",
                "span_seam_counts": seam_counts,
            },
        }
        tmp = out_dir / "pooled.pt.tmp"
        torch.save(store, tmp)
        os.replace(tmp, out_dir / "pooled.pt")
        _atomic_json(
            out_dir / "manifest.json",
            {
                "cell": arm_id,
                "kind": kind,
                "dose": store["dose"],
                "n_rows": len(store["row_meta"]),
                "pooled_sha256": _sha256_file(out_dir / "pooled.pt"),
                **store["metadata"],
            },
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)


def run_capture_tf_unit(cfg: Cfg, arg: str) -> None:
    """Shared-text control (plan §4.6 mandatory): teacher-forced re-capture of
    the arm's RESPONSE arm over the persisted base-pass rows."""
    import torch

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    kind, arm_id = arg.split(":", 1)
    out_dir = cfg.out_root / "capture_tf" / arm_id
    if (out_dir / "pooled.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    beh_key = arm_id.split("-")[0]
    base_raw = cfg.out_root / "capture" / f"base_{beh_key}" / "raw_rows.json"
    rows = json.loads(base_raw.read_text(encoding="utf-8"))["rows"]
    model_path, cleanup = _resolve_arm_model(cfg, arm_id, kind)
    try:
        ctx_ids = panel_context_ids(cfg, beh_key)
        rows = [r for r in rows if r["persona"] in set(ctx_ids)]
        assert rows, (arm_id, ctx_ids)
        pooled = _teacher_forced_span_means(
            model_path,
            rows,
            sorted({r["persona"] for r in rows}),
            layers=list(range(G.N_LAYERS)),
            device="cuda:0",
            dtype=torch.bfloat16,
            tf_batch_size=G.TF_BATCH_SIZE,
        )
        store = {
            "schema_version": 1,
            "cell": arm_id,
            "dose": "selected",
            "kind": f"{kind}_tf_shared",
            "behavior": G.BEHAVIOR_BY_KEY[beh_key],
            "model_path": model_path,
            "row_meta": [
                {"context_id": r["persona"], "question_idx": r["question_idx"]} for r in rows
            ],
            "arms": {
                arm: {li: t.to(torch.float16) for li, t in per_layer.items()}
                for arm, per_layer in pooled.items()
            },
            "metadata": {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_commit": _git_commit(),
                "shared_text": True,
            },
        }
        tmp = out_dir / "pooled.pt.tmp"
        torch.save(store, tmp)
        os.replace(tmp, out_dir / "pooled.pt")
        _atomic_json(
            out_dir / "manifest.json",
            {
                "cell": arm_id,
                "kind": store["kind"],
                "dose": store["dose"],
                "n_rows": len(store["row_meta"]),
                "pooled_sha256": _sha256_file(out_dir / "pooled.pt"),
                **store["metadata"],
            },
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)


def phase_capture(cfg: Cfg) -> dict:
    _phase("p8_capture")
    _headroom(cfg, "p8_capture")
    passes = [
        (a, k)
        for a, k in capture_passes(cfg)
        if not (cfg.out_root / "capture" / a / "pooled.pt").exists()
    ]
    # base passes FIRST (capture_tf consumes their rows)
    passes.sort(key=lambda t: (t[1] != "base", t[0]))
    base_passes = [(a, k) for a, k in passes if k == "base"]
    rest = [(a, k) for a, k in passes if k != "base"]
    for group in (base_passes, rest):
        if not group:
            continue
        if len(group) == 1 or _n_gpus() == 1:
            for a, k in group:
                run_capture_unit(cfg, f"{k}:{a}")
        else:
            _fanout_units(cfg, [_unit_args(cfg, "capture", f"{k}:{a}") for a, k in group])
    return {"n_passes": len(passes)}


def phase_capture_tf(cfg: Cfg) -> dict:
    _phase("p8b_capture_tf")
    arms = [
        (a, k)
        for a, k in _panel_arms(cfg)
        if not (cfg.out_root / "capture_tf" / a / "pooled.pt").exists()
    ]
    if arms:
        if len(arms) == 1 or _n_gpus() == 1:
            for a, k in arms:
                run_capture_tf_unit(cfg, f"{k}:{a}")
        else:
            _fanout_units(cfg, [_unit_args(cfg, "capture_tf", f"{k}:{a}") for a, k in arms])
    return {"n_arms": len(arms)}


# ── p9: residual uploads + manifest + CJK audit ──────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ},
        ).stdout.strip()
    except OSError:
        return "unknown"


def phase_upload(cfg: Cfg, selections: dict) -> dict:
    """p9 residual uploads as ~22 BATCHED upload_folder commits — one per
    cell + one per panel/marker_panel/margin/capture/capture_tf tree + one
    misc (review r1 Major 3: the per-file loop projected ~450+ Hub commits
    against the fleet-shared 256/hr cap — the #664 504-storm class). Files
    are hardlinked into a staging tree at their DATA_PREFIX-relative paths;
    each commit is label-keyed in the persisted manifest (per-cell resume).
    §6.5 glob-parity: every plan-declared class stages (selection records,
    raw completions ALL stages — incl. reselect_rate / rung slot reads /
    xcheck gens, review r1 Minor 3 — margin, capture text + tensors)."""
    _phase("p9_upload")
    _headroom(cfg, "p9_upload")
    manifest_path = cfg.out_root / "upload_manifest.json"
    uploaded: dict[str, str] = _read_json(manifest_path) if manifest_path.exists() else {}
    if not cfg.upload:
        return uploaded
    stage_root = cfg.out_root / "_upload_stage" / "p9"

    def _commit(label: str, stage: Path) -> None:
        if label in uploaded or not _stage_has_files(stage):
            return
        uploaded[label] = _upload_with_transport_retry(stage, G.DATA_PREFIX)
        _atomic_json(manifest_path, uploaded)

    # per-cell staged subtree -> ONE commit per cell (resume granularity).
    for cell in cfg.cells:
        cell_root = cfg.out_root / cell
        stage = stage_root / f"cell_{cell}"
        for name in (
            "build_result.json",
            "ladder.json",
            "selection.json",
            "parity.json",
            "tier2.json",
            "extended.json",
        ):
            _stage_for_upload(stage, cell_root / name, f"selection/{cell}/{name}")
        for stage_name, sub in (
            ("tier1", "rate"),
            ("tier2", "tier2_rate"),
            ("parity", "parity_rate"),
            ("reselect", "reselect_rate"),  # stream-reap spot re-read gens (Minor 3)
        ):
            _stage_tree_for_upload(stage, cell_root / sub, f"raw_completions/{stage_name}/{cell}")
        for rung_dir in sorted(cell_root.glob("rung*/")):
            _stage_for_upload(
                stage,
                rung_dir / "rollouts.json",
                f"raw_completions/ladder/{cell}/{rung_dir.name}.json",
            )
            # per-rung four-float slot reads (Minor 3 gap)
            _stage_for_upload(
                stage,
                rung_dir / "slot_read.json",
                f"slot_reads/ladder/{cell}/{rung_dir.name}.json",
            )
        _commit(f"cell::{cell}", stage)
    # panel + marker panel trees (summaries + rollout text + judged gens)
    for sub, summary_name in (("panel", "panel_summary.json"), ("marker_panel", "slot_reads.json")):
        root = cfg.out_root / sub
        stage = stage_root / sub
        if root.exists():
            for p in sorted(root.glob(f"*/{summary_name}")):
                _stage_for_upload(stage, p, f"{sub}/{p.parent.name}/{p.name}")
            for p in sorted(root.glob("*/rollouts.json")):
                _stage_for_upload(stage, p, f"raw_completions/{sub}/{p.parent.name}.json")
            for d in sorted(root.glob("*/rate_*/")):
                _stage_tree_for_upload(stage, d, f"raw_completions/{sub}/{d.parent.name}/{d.name}")
        _commit(sub, stage)
    # margin tree (base + trained sweeps + per-arm records)
    mroot = cfg.out_root / "margin"
    stage = stage_root / "margin"
    if mroot.exists():
        for p in sorted(mroot.rglob("*.json")):
            _stage_for_upload(stage, p, f"margin/{p.relative_to(mroot)}")
    _commit("margin", stage)
    # capture stores: rollout text (unconditional) + pooled tensors
    for tree in ("capture", "capture_tf"):
        root = cfg.out_root / tree
        stage = stage_root / tree
        if root.exists():
            for p in sorted(root.glob("*/raw_rows.json")):
                _stage_for_upload(stage, p, f"raw_completions/{tree}/{p.parent.name}/raw_rows.json")
            for p in sorted(root.glob("*/pooled.pt")):
                _stage_for_upload(stage, p, f"analysis_tensors/{tree}/{p.parent.name}/pooled.pt")
        _commit(tree, stage)
    # misc run-level records (one commit)
    stage = stage_root / "misc"
    _stage_for_upload(stage, cfg.out_root / "xcheck_s4_po.json", "selection/xcheck_s4_po.json")
    # xcheck Tier-2 gens (Minor 3 gap)
    _stage_tree_for_upload(stage, cfg.out_root / "xcheck_s4_rate", "raw_completions/xcheck_s4")
    for name in ("run_config.json", "disk_mode.json", "parity_failed_cells.json"):
        _stage_for_upload(stage, cfg.out_root / name, name)
    for p in sorted(cfg.out_root.glob("pilot_gate_report_*.json")):
        _stage_for_upload(stage, p, p.name)
    # CJK intrusion audit (plan §4.5 — the #1481 CJK_RE over THIS run's own
    # generation pools; counts only, digest-only discipline. The zeroed /
    # excluded headline recount is the analyzer-side sensitivity read over
    # the persisted per-row pools).
    cjk_out = cfg.out_root / "cjk_audit.json"
    if not cjk_out.exists():
        _atomic_json(cjk_out, _cjk_scan(cfg))
    _stage_for_upload(stage, cjk_out, "cjk_audit.json")
    _commit("misc", stage)
    return uploaded


def _cjk_scan(cfg: Cfg) -> dict:
    """Count CJK-intruded completions per persisted generation pool (the
    #1481 scan regex reused verbatim; issue1481_cjk_audit.CJK_RE)."""
    from issue1481_cjk_audit import CJK_RE

    out: dict[str, dict] = {}
    roots = ["capture", "panel", "marker_panel", *[c for c in cfg.cells]]
    for root in roots:
        base = cfg.out_root / root
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.json")):
            if f.name not in ("raw_rows.json", "rollouts.json"):
                continue
            try:
                rows = json.loads(f.read_text(encoding="utf-8")).get("rows") or []
            except (json.JSONDecodeError, OSError):
                out[str(f.relative_to(cfg.out_root))] = {"error": "unreadable"}
                continue
            texts = [r.get("response", "") if isinstance(r, dict) else str(r) for r in rows]
            out[str(f.relative_to(cfg.out_root))] = {
                "n": len(texts),
                "intruded": sum(bool(CJK_RE.search(t)) for t in texts),
            }
    return {
        "regex": CJK_RE.pattern,
        "n_pools": len(out),
        "n_intruded": sum(v.get("intruded", 0) for v in out.values()),
        "pools": out,
    }


# ── sentinel + main ──────────────────────────────────────────────────────────


def _reproducibility_card(cfg: Cfg, selections: dict) -> dict:
    adapters = {
        cell: f"issue1586/{cell}/checkpoint-{int(sel['step'])}"
        for cell, sel in selections.items()
        if cell != G.REUSED_FT_CELL and "step" in sel
    }
    card = {
        "adapter_paths": adapters,
        "hf_model_repo": G.OVERFLOW_REPO,
        "wandb_project": G.WANDB_PROJECT,
        "wandb_run_names": [ft_wandb_run_name(c) for c in adapters],
    }
    try:
        import wandb

        card["wandb_entity"] = str(wandb.Api().default_entity)
    except Exception as exc:  # entity read is best-effort; never blocks results
        card["wandb_entity_error"] = str(exc)
    return card


def write_sentinel(cfg: Cfg, summary: dict) -> Path:
    _phase("sentinel")
    sentinel_dir = cfg.sentinel_dir or Path("/workspace/logs")
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    kind = "epm:smoke-result" if cfg.smoke else "epm:results"
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,  # VM-side drain re-derives max+1
        "task_id": G.ISSUE,
        "by": "issue1586_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": summary,
    }
    path = sentinel_dir / f"issue-{G.ISSUE}-{kind.replace(':', '_')}-{int(time.time())}.json"
    _atomic_json(path, payload)
    logger.info("[sentinel] wrote %s", path)
    return path


def _check_regime(cfg: Cfg) -> None:
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    p = cfg.out_root / "run_config.json"
    cur = {
        **cfg.regime_key(),
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if p.exists():
        prior = _read_json(p)
        skip = {"cells", "git_commit", "ts"}
        if {k: v for k, v in prior.items() if k not in skip} != {
            k: v for k, v in cur.items() if k not in skip
        } or not set(cur["cells"]) <= set(prior.get("cells", [])):
            raise RuntimeError(f"out_root {cfg.out_root} holds a run under a DIFFERENT regime")
    else:
        _atomic_json(p, cur)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1586 pod-side phase driver")
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
        "tier2 <cell> | margin base:<beh>|<kind>:<arm> | "
        "panel <kind>:<arm> | capture <kind>:<arm> | capture_tf <kind>:<arm>)",
    )
    p.add_argument(
        "--gpu-id", default="0", help="physical GPU (CVD-pinned by the launcher; informational)"
    )
    p.add_argument("--cells", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument(
        "--ladder-disk-mode", choices=["auto", "keep-cell", "stream-reap"], default="auto"
    )
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    p.add_argument("--phases", default=None, help="comma subset of phases (default all)")
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
    # Smoke default out_root prefers /workspace (GCE boot disk / RunPod
    # volume) over /tmp: the RunPod container disk is ~50 GB, below the
    # p2_train 60 GB headroom floor + a ~15 GB full-FT smoke ckpt (review r1
    # Minor 2; the sentinel_dir logic below already keys on /workspace).
    smoke_root = (
        f"/workspace/issue-{G.ISSUE}-smoke"
        if Path("/workspace").is_dir()
        else f"/tmp/issue-{G.ISSUE}-smoke"
    )
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else (smoke_root if smoke else f"data/issue_{G.ISSUE}/out")
    )
    return Cfg(
        smoke=smoke,
        cells=resolve_cells(args.cells, smoke),
        out_root=out_root,
        ladder_disk_mode=args.ladder_disk_mode,
        tier1_n=2 if smoke else 5,
        tier1_draws=2 if smoke else 3,
        tier2_n=2 if smoke else 10,
        tier2_draws=2 if smoke else 5,
        panel_n=2 if smoke else 5,
        panel_draws=2 if smoke else 3,
        eval_question_limit=(
            args.eval_question_limit
            if args.eval_question_limit is not None
            else (2 if smoke else None)
        ),
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
        elif kind == "tier2":
            run_tier2_unit(cfg, arg)
        elif kind == "margin":
            run_margin_unit(cfg, arg)
        elif kind == "panel":
            run_panel_unit(cfg, arg)
        elif kind == "capture":
            run_capture_unit(cfg, arg)
        elif kind == "capture_tf":
            run_capture_tf_unit(cfg, arg)
        else:
            raise ValueError(f"unknown unit kind {kind!r}")
        return 0
    _check_regime(cfg)
    logger.info(
        "issue1586 smoke=%s cells=%s out_root=%s disk_mode=%s",
        cfg.smoke,
        cfg.cells,
        cfg.out_root,
        cfg.resolved_disk_mode(),
    )

    def want(phase: str) -> bool:
        return not cfg.phases or phase in cfg.phases

    summary: dict = {
        "issue": G.ISSUE,
        "smoke": cfg.smoke,
        "cells": list(cfg.cells),
        "git_commit": _git_commit(),
    }
    if want("stage"):
        st = phase_stage(cfg)
        summary["stage"] = {"pins": st.get("pins"), "n_mixes": len(st.get("mixes", {}))}
    if want("parity"):
        summary["parity"] = {
            k: {
                kk: v.get(kk)
                for kk in ("rate", "delta_g", "expected", "rate_window_pass", "severity")
            }
            for k, v in phase_parity(cfg).items()
            if isinstance(v, dict)
        }
    if want("train"):
        phase_train(cfg)
    selections: dict = {}
    if want("ladder"):
        selections = phase_ladder(cfg)
        summary["selections"] = {
            k: {
                kk: v.get(kk)
                for kk in ("step", "metric", "in_band", "fallback", "anchor_gap", "reused")
            }
            for k, v in selections.items()
        }
    if want("persist"):
        summary["persist"] = {"n": len(phase_persist(cfg, selections).get("uploaded", {}))}
    if want("tier2"):
        t2 = phase_tier2(cfg, selections)
        summary["tier2"] = {
            k: {
                "rate": v.get("tier2_rate"),
                "dose_matched": (v.get("dose_label") or {}).get("dose_matched"),
            }
            for k, v in t2.items()
        }
        summary["parity_failed_cells"] = _parity_failed_cells(cfg)
    if want("panel"):
        summary["panel"] = phase_panel(cfg)
    if want("margin"):
        summary["margin"] = {
            k: {kk: v.get(kk) for kk in ("margin_base", "margin_trained", "margin_delta")}
            for k, v in phase_margin(cfg, selections).items()
            if isinstance(v, dict)
        }
    if want("capture"):
        summary["capture"] = phase_capture(cfg)
    if want("capture_tf"):
        summary["capture_tf"] = phase_capture_tf(cfg)
    summary["reproducibility_card"] = _reproducibility_card(cfg, selections)
    if want("upload"):
        summary["n_uploaded"] = len(phase_upload(cfg, selections))
    summary["sentinel"] = str(write_sentinel(cfg, summary))
    logger.info(
        "issue1586 complete: %s",
        json.dumps({k: summary[k] for k in ("smoke", "cells", "n_uploaded") if k in summary}),
    )
    # NOTE: [phase=done] is emitted by the launcher wrapper, never here.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
