"""#1768 inline round — the two missing (model x text) cells of the decomposition.

Round 1 captured three of the four (model x text) cells of the on-policy shift
``v+(trained text) - v0(base text)``:

===================  ==========================  ==============================
cell                 round-1 tree                this round
===================  ==========================  ==============================
v0(base text)        ``corpus_capture/base_*``   reused
v+(base text)        ``corpus_capture_tf/<arm>`` reused
v+(trained text)     ``corpus_capture/<arm>``    reused
v0(trained text)     --- MISSING ---             **leg A** ``reverse_tf/<arm>``
===================  ==========================  ==============================

Leg A (``rtf`` units) closes that cell: the BASE model teacher-forced on each
TRAINED arm's own on-policy response text, for the 8-arm write-predictability
subset. It replaces round 1's map STAND-IN for the text effect
(``M0(c+) - M0(c0)``, `issue1768_fit._decomposition_block`'s ``input_move``)
with a MEASURED cell, and it is the direct test of the chat claim "a base model
reading the trained text shifts along delta".

Leg B (``btf`` units) adds a second missing cell on a different text
distribution: each of the 72 TRAINED arms teacher-forced on its OWN training
positives. The base side of that cell already exists as round 1's
``delta_tf/<arm>/tbar.pt`` (base model on the same mix rows) and is REUSED, not
recaptured; ft arms read the matched pers-LoRA cell via
`issue1768_cells.delta_arm_for` exactly as round 1's `delta_leg` does.

Capture conventions are round 1's, byte-for-byte: the canonical
`analysis.representation_shift._teacher_forced_span_means` (span-mean pooling,
``prefix_end='last_user'``, ``on_seam='snap'``, layers 14/19/25, fp16 pooled
stores), rows/spans re-joined through `issue1768_capture._read_rows_with_spans`
(leg A) and `issue1768_capture._mix_positive_rows` (leg B), models resolved
through `issue1768_capture._resolve_unit_model` (LoRA merge / ft stage ->
consume -> delete). Greedy/teacher-forced only: this round generates NOTHING.

Phases::

  p0   stage inputs (arm registry extended to the 16 ft arms, corpus sample,
       leg-A row text, delta_tf t_bar) + Hub-resolution probe + headroom gate
  rtf  leg A: 8 base-on-trained-text capture units
  btf  leg B: 72 trained-on-own-training-rows capture units
  up   one bulk upload_folder commit per new tree + exact-set verify
  an   measured 2x2 decomposition + validation vs round 1's M0 attribution
       + leg-B reads -> ``model_text_2x2/`` JSONs

Signaling: ``[phase=...]`` lines + per-unit progress lines + ``status_2x2.json``
heartbeat. Units fan out across every visible GPU via CUDA_VISIBLE_DEVICES-pinned
subprocesses (round 1's `_fanout_phase` shape).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before torch: shared-VM thread caps + HF/W&B credentials

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_capture as C  # noqa: E402
import issue1768_cells as X  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.mt2x2")

RTF_TREE = "reverse_tf"
BTF_TREE = "train_rows_tf"
RESULTS_DIR = "model_text_2x2"
# leg-A row text: the per-arm files `_read_rows_with_spans` consumes (never the
# 705 MB pooled store — the base model re-reads the rows, it does not read v+).
ROW_TEXT_NAMES = ("rows_spans.json", "raw_rows.done.json", "manifest.json")
PHASE_HEADROOM_GB = {"rtf": 20.0, "btf": 70.0, "an": 30.0}
# capture-integrity gate (analysis-side, REPORTED never halting): the base model
# re-reading an arm's rows must reproduce base_content's own CONTEXT-span vectors
# on the shared rows — same model, same prompt text, different batch composition,
# so only bf16 padded-batch jitter separates them (gotchas.md two-bar recipe).
CONTEXT_IDENTITY_COS_MIN = 0.999


@dataclasses.dataclass
class Cfg:
    """Run configuration; every output-affecting knob is part of the regime."""

    out_root: Path
    phases: tuple[str, ...]
    rtf_arms: tuple[str, ...] = ()  # empty -> the 8 write-predictability picks
    rtf_all: bool = False  # leg A over ALL 72 arms (the fleet-wide extension)
    btf_arms: tuple[str, ...] = ()  # empty -> all 72
    rtf_wave_size: int = 0  # >0 -> upload + verify after every N leg-A units
    restage: bool = True  # pull this round's own uploaded stores into a fresh out-root
    layers: tuple[int, ...] = X.LAYERS
    tf_batch: int = X.TF_BATCH_SIZE
    smoke: bool = False
    smoke_rows: int = 24  # leg-A row cap under --smoke (leg B pools are tiny)
    upload: bool = True
    hf_prefix: str = X.HF_PREFIX
    attrib_layer: int = 19  # M0-attribution validation layer (plan headline layer)
    gpu_id: int = 0  # informational; the launcher env CVD pin selects the GPU

    def capture_cfg(self) -> C.Cfg:
        """The round-1 `Cfg` the reused capture helpers read (out_root/layers/
        tf_batch/model paths). `smoke` stays FALSE here on purpose: round 1's
        `_mix_positive_rows` truncates its rows to 4 under smoke, which would
        break the `delta_tf` t_bar row-count parity assert leg B depends on."""
        return C.Cfg(
            out_root=self.out_root,
            phases=(),
            smoke=False,
            layers=self.layers,
            tf_batch=self.tf_batch,
            upload=False,
            hf_prefix=self.hf_prefix,
            gpu_id=self.gpu_id,
        )


def _phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


def _status(cfg: Cfg, phase: str, **extra) -> None:
    C._atomic_json(
        cfg.out_root / "status_2x2.json",
        {"phase": phase, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra},
    )


def _meta() -> dict:
    return {**C._meta(), "round": "inline-model-text-2x2"}


def _hub():
    from explore_persona_space.orchestrate import hub

    return hub


def _picks(cfg: Cfg) -> list[str]:
    """The leg-A arm subset: the committed write-predictability picks.

    The picks file is a GIT artifact (round 1's `write_predictability` tree is
    committed, not in the Hub `UPLOAD_TREES` list), so the repo copy is the
    source of record and the Hub stage is only a fallback for a checkout that
    lacks it.
    """
    if cfg.rtf_arms:
        return list(cfg.rtf_arms)
    if cfg.rtf_all:  # fleet-wide extension: leg A over the whole 72-arm grid
        return sorted(C._full_arm_index())
    path = cfg.out_root / RESULTS_DIR / "arm_picks.json"
    if not path.exists():
        committed = (
            REPO_ROOT
            / "eval_results"
            / f"issue_{X.ISSUE}"
            / "write_predictability"
            / "arm_picks.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if committed.exists():
            shutil.copyfile(committed, path)
            logger.info("[picks] copied the committed picks from %s", committed)
        else:
            _hub().stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/write_predictability/arm_picks.json",
                path,
                repo_type="dataset",
            )
    picks = json.loads(path.read_text())["picks"]
    assert picks, (path, "no arm picks")
    return [p["arm_id"] for p in picks]


def _subset_picks(cfg: Cfg) -> list[str]:
    """The ORIGINAL 8 write-predictability picks, regardless of --rtf-all.

    Recorded in the outputs so fleet-wide figures can mark the arms the first
    round measured without re-deriving the subset.
    """
    path = cfg.out_root / RESULTS_DIR / "arm_picks.json"
    if not path.exists():
        committed = (
            REPO_ROOT
            / "eval_results"
            / f"issue_{X.ISSUE}"
            / "write_predictability"
            / "arm_picks.json"
        )
        if not committed.exists():
            return []
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(committed, path)
    return [p["arm_id"] for p in json.loads(path.read_text())["picks"]]


def _btf_arms(cfg: Cfg) -> list[str]:
    if cfg.btf_arms:
        return list(cfg.btf_arms)
    return sorted(C._full_arm_index())


# ── p0: input staging ────────────────────────────────────────────────────────


def _stage_registry(cfg: Cfg) -> dict:
    """Round-1 `arm_registry.json` extended with the 16 ft arms' mix sources.

    Round 1 probed `mix_pos_sources` for its 56 p5 (LoRA) units only; leg B
    covers all 72, so the ft arms' positives paths are resolved through the
    SAME `_mix_pos_source` candidate ladder + Hub probe. The extended registry
    is LOCAL to this round's out-root and is never uploaded (the round-1
    registry on the Hub stays untouched).
    """
    path = cfg.out_root / "arm_registry.json"
    if not path.exists():
        _hub().stage_hub_file(
            X.HF_DATA_REPO, f"{X.HF_PREFIX}/arm_registry.json", path, repo_type="dataset"
        )
    reg = json.loads(path.read_text())
    have = set(reg["mix_pos_sources"])
    missing = [a for a in C._full_arm_index().values() if a.arm_id not in have]
    if missing:
        logger.info("[p0] probing mix positives for %d arms absent from round 1", len(missing))
        reg["mix_pos_sources"].update(C._probe_mix_sources(missing))
        reg["mix_pos_sources_extended_by"] = _meta()
        C._atomic_json(path, reg)
    assert len(reg["mix_pos_sources"]) >= len(C._full_arm_index()), len(reg["mix_pos_sources"])
    return reg


def _stage_corpus_sample(cfg: Cfg) -> None:
    path = cfg.out_root / "inputs" / "corpus_sample.json"
    if not path.exists():
        _hub().stage_hub_file(
            X.HF_DATA_REPO,
            f"{X.HF_PREFIX}/inputs/corpus_sample.json",
            path,
            repo_type="dataset",
        )
    X.load_corpus_sample(cfg.out_root)  # fail loud on a truncated stage


def _stage_row_text(cfg: Cfg, unit_id: str) -> Path:
    """Stage ONE corpus unit's row text + spans (never its pooled store).

    `hub.stage_hub_prefix` mirrors the whole prefix (705 MB pooled included),
    so the per-file `stage_hub_file` form is used with an explicit name filter.
    """
    hub = _hub()
    dest = cfg.out_root / "corpus_capture" / unit_id
    prefix = f"{X.HF_PREFIX}/corpus_capture/{unit_id}"
    if (dest / "rows_spans.json").exists() and (dest / "raw_rows.done.json").exists():
        if list(dest.glob("raw_rows_*.jsonl")):
            return dest
    from huggingface_hub import HfApi

    names = [
        p.rsplit("/", 1)[-1]
        for p in hub.list_hf_files_under_path(HfApi(), X.HF_DATA_REPO, prefix, repo_type="dataset")
    ]
    want = [n for n in names if n in ROW_TEXT_NAMES or n.startswith("raw_rows_")]
    assert "rows_spans.json" in want, (unit_id, "rows_spans.json absent on the Hub")
    for name in sorted(want):
        target = dest / name
        if not target.exists():
            hub.stage_hub_file(X.HF_DATA_REPO, f"{prefix}/{name}", target, repo_type="dataset")
    logger.info("[p0] staged %d row-text files for %s", len(want), unit_id)
    return dest


def _stage_delta_cell(cfg: Cfg, arm_id: str) -> Path:
    """Stage the arm's round-1 delta cell (t_bar + the exact positives file).

    Pre-placing the Hub's `pos.jsonl` at the path `_mix_positive_rows` stages
    into makes leg B's rows byte-identical to round 1's by construction (the
    sha assert in `run_btf_unit` then binds), and ft arms inherit their matched
    pers-LoRA cell's copy via `delta_arm_for`.
    """
    hub = _hub()
    arm = C._full_arm_index()[arm_id]
    delta_arm = X.delta_arm_for(arm)
    src_prefix = f"{X.HF_PREFIX}/delta_tf/{delta_arm}"
    dest = cfg.out_root / "delta_tf" / delta_arm
    tbar = dest / "tbar.pt"
    if not tbar.exists():
        hub.stage_hub_file(X.HF_DATA_REPO, f"{src_prefix}/tbar.pt", tbar, repo_type="dataset")
    pos_name = Path(
        json.loads((cfg.out_root / "arm_registry.json").read_text())["mix_pos_sources"][arm_id][
            "pos_path"
        ]
    ).name
    local_pos = cfg.out_root / "delta_tf" / arm_id / pos_name
    if not local_pos.exists():
        from huggingface_hub import HfApi

        api, remote = HfApi(), f"{src_prefix}/{pos_name}"
        if hub.retry_transient(
            # HUB_VERIFY_RETRY_EXEMPT: single-path existence probe already wrapped in retry_transient
            lambda: api.file_exists(X.HF_DATA_REPO, remote, repo_type="dataset"),
            what=f"pos probe {remote}",
        ):
            hub.stage_hub_file(X.HF_DATA_REPO, remote, local_pos, repo_type="dataset")
        else:  # no round-1 copy (ft arms whose delta cell used another name)
            logger.info("[p0] %s: no round-1 pos copy at %s — source stage at unit", arm_id, remote)
    return tbar


def _restage_own_outputs(cfg: Cfg) -> dict[str, int]:
    """Pull THIS round's already-uploaded stores back into a fresh out-root.

    The Hub trees are the round's durable spine, so a fresh pod re-stages them
    instead of recapturing: `_pending_units` keys on local presence, so a
    restaged unit is SKIPPED. Idempotent (per-file `stage_hub_file` skips an
    existing target) and fail-soft on an absent prefix — a tree that was never
    uploaded simply yields nothing to restage.

    The leg-A tree is restaged ONLY when this run captures leg A (`rtf` among
    the phases): it is ~700 MB per arm (~51 GB fleet-wide), and an
    analysis-only host streams it per arm instead (`_stage_arm_analysis_stores`).
    The leg-B tree is ~1.5 MB per arm and always restages — `_leg_b_read`
    needs it locally.
    """
    from huggingface_hub import HfApi

    hub = _hub()
    api = HfApi()
    counts: dict[str, int] = {}
    trees = (RTF_TREE, BTF_TREE) if "rtf" in cfg.phases else (BTF_TREE,)
    for tree in trees:
        prefix = f"{cfg.hf_prefix}/{tree}"
        try:
            remote = hub.list_hf_files_under_path(api, X.HF_DATA_REPO, prefix, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001 — an absent tree is not an error
            logger.info("[p0] no %s tree to restage (%s)", tree, type(exc).__name__)
            counts[tree] = 0
            continue
        staged = 0
        for path in remote:
            rel = path[len(prefix) + 1 :]
            target = cfg.out_root / tree / rel
            if target.exists():
                continue
            hub.stage_hub_file(X.HF_DATA_REPO, path, target, repo_type="dataset")
            staged += 1
        counts[tree] = staged
        logger.info("[p0] restaged %d/%d files into %s", staged, len(remote), tree)
    return counts


def _stage_base_panels(cfg: Cfg) -> list[str]:
    """The 4 base panel stores the delta baseline half reads (72 MB each)."""
    hub = _hub()
    behs = sorted({C._full_arm_index()[a].beh_key for a in {*_btf_arms(cfg), *_picks(cfg)}})
    for beh in behs:
        dest = cfg.out_root / "panel_capture" / f"base_{beh}" / "pooled.pt"
        if not dest.exists():
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/panel_capture/base_{beh}/pooled.pt",
                dest,
                repo_type="dataset",
            )
    logger.info("[p0] staged %d base panel stores: %s", len(behs), behs)
    return behs


def phase_p0(cfg: Cfg) -> None:
    _phase("p0_stage")
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    _stage_registry(cfg)
    _stage_corpus_sample(cfg)
    restaged = _restage_own_outputs(cfg) if cfg.restage else {}
    picks = _picks(cfg)
    units = sorted({*picks, *(X.base_unit_for(a) for a in picks)})
    pending_rtf = set(_pending_units(cfg, "rtf")) if "rtf" in cfg.phases else set()
    # row text is ONLY a leg-A capture input (~78 MB of shards per arm): stage it
    # for units still to capture, and not at all on an analysis-only host
    for u in sorted({*pending_rtf, *(X.base_unit_for(a) for a in pending_rtf)}):
        _stage_row_text(cfg, u)
    for arm_id in _btf_arms(cfg):
        _stage_delta_cell(cfg, arm_id)
    _stage_base_panels(cfg)
    C._atomic_json(
        cfg.out_root / RESULTS_DIR / "stage_manifest.json",
        {
            "rtf_arms": picks,
            "rtf_subset_arms": _subset_picks(cfg),
            "rtf_row_text_units": units,
            "rtf_pending_at_p0": sorted(pending_rtf),
            "btf_arms": _btf_arms(cfg),
            "restaged_from_hub": restaged,
            "layers": list(cfg.layers),
            **_meta(),
        },
    )
    _status(cfg, "p0_stage", rtf=len(picks), pending=len(pending_rtf), btf=len(_btf_arms(cfg)))


# ── leg A: base model on the trained arms' own text ──────────────────────────


def run_rtf_unit(cfg: Cfg, arm_id: str) -> None:
    """One leg-A unit: BASE model teacher-forced on `arm_id`'s own rows.

    The exact reverse of round 1's `run_corpus_tf_unit` (trained model on the
    BASE tree's rows): same canonical pooling call, same store schema, rows
    from the ARM's tree instead of the base tree, model = base instead of the
    arm. Both spans are captured: ``response`` is the decomposition cell, and
    ``context`` is a free capture-integrity read (same model + same prompt text
    as base_content, so it must reproduce base_content's context vectors).
    """
    import torch

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    out_dir = cfg.out_root / RTF_TREE / arm_id
    if (out_dir / "pooled_tf.pt").exists():
        logger.info("[rtf] %s: store present — skip", arm_id)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = _stage_row_text(cfg, arm_id)
    rows = C._read_rows_with_spans(src_dir)
    if cfg.smoke:
        rows = rows[: cfg.smoke_rows]
    assert rows, (arm_id, "no rows with spans")
    pooled = _teacher_forced_span_means(
        X.BASE_MODEL,
        rows,
        [arm_id],
        layers=list(cfg.layers),
        spans=("context", "response"),
        device=C._device(),
        dtype=C._dtype(),
        tf_batch_size=cfg.tf_batch,
    )
    fp16_cos = C._fp16_roundtrip_cos_min(pooled)
    store = {
        "schema_version": 1,
        "unit": arm_id,
        "row_sha": [r["prompt_sha"] for r in rows],
        "row_question_idx": [r["question_idx"] for r in rows],
        "arms": {
            span: {li: t.to(torch.float16) for li, t in per.items()} for span, per in pooled.items()
        },
        "metadata": {
            **_meta(),
            "model_path": X.BASE_MODEL,
            "layers": list(cfg.layers),
            "spans": ["context", "response"],
            "shared_text_from": arm_id,
            "cell": "v0_on_trained_text",
            "n_rows": len(rows),
            "fp16_roundtrip_cos_min": fp16_cos,
            "smoke": cfg.smoke,
        },
    }
    tmp = out_dir / "pooled_tf.pt.tmp"
    torch.save(store, tmp)
    os.replace(tmp, out_dir / "pooled_tf.pt")
    C._atomic_json(
        out_dir / "manifest.json",
        {
            "unit": arm_id,
            "cell": "v0_on_trained_text",
            "n_rows": len(rows),
            "model_path": X.BASE_MODEL,
            "fp16_roundtrip_cos_min": fp16_cos,
            **_meta(),
        },
    )


# ── leg B: trained arms on their own training positives ──────────────────────


def run_btf_unit(cfg: Cfg, arm_id: str) -> None:
    """One leg-B unit: the TRAINED arm teacher-forced on its own mix positives.

    Rows come from round 1's `_mix_positive_rows` (same resolution, same
    token-id concat, same spans), and the round-1 base-side cell
    (`delta_tf/<delta_arm>/tbar.pt`) is asserted row-count- and
    sha-identical so ``Delta v_train = tbar_plus - tbar`` is a matched-row
    difference. The model rides `_resolve_unit_model`'s
    merge/stage -> consume -> delete lifecycle.
    """
    import torch

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    ccfg = cfg.capture_cfg()
    out_dir = cfg.out_root / BTF_TREE / arm_id
    if (out_dir / "tbar_plus.pt").exists():
        logger.info("[btf] %s: store present — skip", arm_id)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    arm = C._full_arm_index()[arm_id]
    delta_arm = X.delta_arm_for(arm)
    base_cell = torch.load(
        cfg.out_root / "delta_tf" / delta_arm / "tbar.pt", map_location="cpu", weights_only=False
    )
    rows, meta = C._mix_positive_rows(ccfg, arm)
    assert len(rows) == int(base_cell["n_rows"]), (
        arm_id,
        delta_arm,
        len(rows),
        int(base_cell["n_rows"]),
        "leg-B rows must match the round-1 delta cell row count",
    )
    base_sha = base_cell["meta"].get("pos_sha256")
    assert base_sha is None or base_sha == meta["pos_sha256"], (
        arm_id,
        delta_arm,
        base_sha,
        meta["pos_sha256"],
        "positives file differs from the round-1 delta cell",
    )
    model_path, cleanup = C._resolve_unit_model(ccfg, arm_id)
    try:
        pooled = _teacher_forced_span_means(
            model_path,
            rows,
            [arm_id],
            layers=list(cfg.layers),
            spans=("response",),
            device=C._device(),
            dtype=C._dtype(),
            tf_batch_size=cfg.tf_batch,
        )
    finally:
        C._cleanup_merged(cleanup)
    per_layer = pooled["response"]
    tbar_plus = {li: t.mean(dim=0) for li, t in per_layer.items()}
    halves = len(rows) >= 2
    store = {
        "schema_version": 1,
        "unit": arm_id,
        "cell": "vplus_on_train_rows",
        "tbar_plus": tbar_plus,
        "tbar_plus_even": {li: t[0::2].mean(dim=0) for li, t in per_layer.items()}
        if halves
        else None,
        "tbar_plus_odd": {li: t[1::2].mean(dim=0) for li, t in per_layer.items()}
        if halves
        else None,
        "rows": {li: t.to(torch.float16) for li, t in per_layer.items()},
        "n_rows": len(rows),
        "delta_arm": delta_arm,
        "meta": {
            **_meta(),
            **meta,
            "model_path": model_path,
            "method": arm.method,
            "layers": list(cfg.layers),
        },
    }
    tmp = out_dir / "tbar_plus.pt.tmp"
    torch.save(store, tmp)
    os.replace(tmp, out_dir / "tbar_plus.pt")


# ── unit dispatch (round 1's work-conserving CVD-pinned fan-out) ─────────────


def run_unit(cfg: Cfg, unit_arg: str) -> None:
    phase, unit = unit_arg.split(":", 1)
    if phase == "rtf":
        run_rtf_unit(cfg, unit)
    elif phase == "btf":
        run_btf_unit(cfg, unit)
    else:
        raise ValueError(f"unknown unit phase {phase}")


def _pending_units(cfg: Cfg, phase: str) -> list[str]:
    if phase == "rtf":
        return [
            a for a in _picks(cfg) if not (cfg.out_root / RTF_TREE / a / "pooled_tf.pt").exists()
        ]
    if phase == "btf":
        return [
            a for a in _btf_arms(cfg) if not (cfg.out_root / BTF_TREE / a / "tbar_plus.pt").exists()
        ]
    raise ValueError(phase)


def _unit_cmd(cfg: Cfg, unit_arg: str, gpu: int) -> tuple[list[str], dict]:
    cmd = [
        "uv",
        "run",
        "python",
        str(Path(__file__).resolve()),
        "--out-root",
        str(cfg.out_root),
        "--unit",
        unit_arg,
        "--gpu-id",
        str(gpu),
        "--layers",
        ",".join(str(x) for x in cfg.layers),
        "--tf-batch",
        str(cfg.tf_batch),
        "--attrib-layer",
        str(cfg.attrib_layer),
    ]
    if cfg.smoke:
        cmd += ["--smoke", "--smoke-rows", str(cfg.smoke_rows)]
    if cfg.rtf_arms:
        cmd += ["--rtf-arms", ",".join(cfg.rtf_arms)]
    if cfg.btf_arms:
        cmd += ["--btf-arms", ",".join(cfg.btf_arms)]
    # explicit env (subprocess env= contract) + launcher-level CVD pin (#545)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return cmd, env


def _needs_model_materialization(cfg: Cfg, unit_arg: str) -> bool:
    """~15 GB local model pending (LoRA merge OR ft stage) — the disk clamp."""
    phase, unit = unit_arg.split(":", 1)
    if phase != "btf":
        return False  # leg A runs the base model straight from the HF cache
    arm = C._full_arm_index()[unit]
    if arm.method == "ft":
        return C._ft_ckpt_incomplete_reason(C._ft_ckpt_dirs(cfg.capture_cfg(), arm)[1]) is not None
    return not (cfg.out_root / "merged" / unit / "config.json").exists()


def _merge_slots(cfg: Cfg, width: int) -> int:
    """Concurrency clamp for model-materializing units, keyed to free disk."""
    free_gb = shutil.disk_usage(cfg.out_root).free / 1e9
    return max(1, min(width, int((free_gb - 40) // 18)))


def _fanout(cfg: Cfg, phase: str, phase_tag: str) -> None:
    """Work-conserving CVD-pinned unit fan-out (round 1 `_fanout_phase` shape).

    No barriers: leg-A and leg-B units are mutually independent, and a
    model-materializing unit is admitted only while free disk allows another
    ~18 GB model (`_merge_slots`).
    """
    _phase(phase_tag)
    units = _pending_units(cfg, phase)
    _status(cfg, phase_tag, pending=len(units))
    if not units:
        logger.info("[%s] nothing pending", phase)
        return
    gpus = C._physical_gpus()
    if len(gpus) <= 1:
        for k, u in enumerate(units):
            t0 = time.time()
            run_unit(cfg, f"{phase}:{u}")
            print(
                f"[{phase}] unit {k + 1}/{len(units)} {u} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
            _status(cfg, phase_tag, done=k + 1, total=len(units))
        return
    slots = _merge_slots(cfg, len(gpus))
    logger.info("[%s] %d units over gpus %s (model slots %d)", phase, len(units), gpus, slots)
    queue = [f"{phase}:{u}" for u in units]
    running: dict[int, tuple[subprocess.Popen, str, float]] = {}
    done = 0
    while queue or running:
        for gpu in [g for g in gpus if g not in running]:
            if not queue:
                break
            active = sum(
                1 for _p, ua, _t in running.values() if _needs_model_materialization(cfg, ua)
            )
            nxt = next(
                (
                    i
                    for i, ua in enumerate(queue)
                    if not _needs_model_materialization(cfg, ua) or active < slots
                ),
                None,
            )
            if nxt is None:
                break
            unit_arg = queue.pop(nxt)
            cmd, env = _unit_cmd(cfg, unit_arg, gpu)
            log_path = cfg.out_root / "logs" / f"mt2x2_{unit_arg.replace(':', '_')}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = log_path.open("a")
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=logf, stderr=logf)
            running[gpu] = (proc, unit_arg, time.time())
            logger.info("[%s] dispatched %s on gpu %d (pid %d)", phase, unit_arg, gpu, proc.pid)
        time.sleep(5)
        for gpu, (proc, unit_arg, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[gpu]
            if rc != 0:
                log_path = cfg.out_root / "logs" / f"mt2x2_{unit_arg.replace(':', '_')}.log"
                tail = log_path.read_text()[-4000:] if log_path.exists() else "(no log)"
                for sib, sib_arg, _t in running.values():
                    logger.warning("[%s] terminating sibling %s on failure", phase, sib_arg)
                    sib.terminate()
                deadline = time.time() + 15
                for sib, _sib_arg, _t in running.values():
                    try:
                        sib.wait(timeout=max(0.1, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        sib.kill()
                running.clear()
                raise RuntimeError(
                    f"[{phase}] unit {unit_arg} exited rc={rc}\n--- log tail ---\n{tail}"
                )
            done += 1
            print(
                f"[{phase}] unit {done}/{len(units)} {unit_arg} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
            _status(cfg, phase_tag, done=done, total=len(units))
            # wave upload: persist partial progress to the Hub as it lands, so a
            # mid-run pod loss strands at most one wave (#664 per-cell contract)
            if (
                phase == "rtf"
                and cfg.upload
                and cfg.rtf_wave_size > 0
                and done % cfg.rtf_wave_size == 0
                and (queue or running)
            ):
                logger.info("[%s] wave upload after %d/%d units", phase, done, len(units))
                phase_upload(cfg, trees=(RTF_TREE,), tag=f"wave{done}")


# ── upload ───────────────────────────────────────────────────────────────────


def phase_upload(
    cfg: Cfg, trees: tuple[str, ...] = (RTF_TREE, BTF_TREE), tag: str = ""
) -> dict[str, dict]:
    """One bulk `upload_folder` commit per named tree + exact-set verify.

    `upload_folder` is idempotent for already-landed files, so a wave upload of
    a growing tree re-commits only the new units; the exact-set verify always
    covers the WHOLE local tree, which is what makes a wave a durable snapshot.
    """
    _phase(f"up_upload{('_' + tag) if tag else ''}")
    from huggingface_hub import HfApi

    hub = _hub()
    api = HfApi()
    out: dict[str, dict] = {}
    for tree in trees:
        local = cfg.out_root / tree
        if not local.exists():
            continue
        dest = f"{cfg.hf_prefix}/{tree}"
        expect = sorted(
            f"{dest}/{p.relative_to(local).as_posix()}" for p in local.rglob("*") if p.is_file()
        )
        url = hub._upload(local, repo_id=X.HF_DATA_REPO, repo_type="dataset", path_in_repo=dest)
        if not url:
            raise RuntimeError(f"upload of {local} -> {dest} returned no path")
        missing = hub.verify_repo_paths_uploaded(
            api, X.HF_DATA_REPO, expect, path_in_repo=dest, repo_type="dataset"
        )
        if missing:
            raise RuntimeError(
                f"{dest}: {len(missing)} files missing after upload, e.g. {missing[0]}"
            )
        out[tree] = {"dest": dest, "n_files": len(expect)}
        logger.info("[up] %s -> %s (%d files verified)", tree, dest, len(expect))
    name = f"upload_done{('_' + tag) if tag else ''}.json"
    C._atomic_json(
        cfg.out_root / RESULTS_DIR / name, {"trees": out, "tag": tag or "final", **_meta()}
    )
    _status(cfg, "up_upload", trees=sorted(out), tag=tag or "final")
    return out


# ── analysis ─────────────────────────────────────────────────────────────────


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _store(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _span(store: dict, span: str, layer: int) -> np.ndarray:
    return np.asarray(store["arms"][span][layer].float().numpy(), dtype=np.float64)


def _delta_direction(cfg: Cfg, arm_id: str, layer: int) -> dict:
    """Round 1's delta leg for one (arm, layer), rebuilt from BASE-side inputs.

    ``delta_primary = tbar - v0_half_B`` (`issue1768_directions.delta_leg`)
    needs only the BASE panel store plus the arm's delta cell, so the 72 MB
    per-arm panel stores are NOT staged; the identical round-1 helpers
    (`source_context_id`, `_panel_rows`, `_half_means`) compute the baseline
    half. The panel WRITE leg (`w_primary`) additionally needs the arm's own
    panel store and is therefore returned only when that store happens to be
    present locally — every read the round requires is delta-based.
    """
    import issue1768_directions as D
    import torch

    arm = C._full_arm_index()[arm_id]
    beh = arm.beh_key
    base_store = D._load_store(cfg.out_root / "panel_capture" / f"base_{beh}" / "pooled.pt")
    src_ctx = D.source_context_id(arm, base_store)
    v0 = D._panel_rows(base_store, src_ctx, layer)
    _v0_all, _v0_A, v0_B = D._half_means(v0)
    delta_arm = X.delta_arm_for(arm)
    tb = torch.load(
        cfg.out_root / "delta_tf" / delta_arm / "tbar.pt", map_location="cpu", weights_only=False
    )
    tbar = np.asarray(tb["tbar"][layer].float().numpy(), dtype=np.float64)
    out = {
        "delta": tbar - np.asarray(v0_B, dtype=np.float64),
        "delta_arm": delta_arm,
        "n_mix_rows": int(tb["n_rows"]),
        "src_ctx": src_ctx,
        "n_panel_questions": len(v0),
    }
    arm_panel = cfg.out_root / "panel_capture" / arm_id / "pooled.pt"
    if arm_panel.exists():
        legs = D.panel_write_legs(cfg.out_root, arm, layer)
        out["w_panel"] = np.asarray(legs["w_primary"], dtype=np.float64)
    return out


ARM_ANALYSIS_STORES = (
    ("corpus_capture", "pooled.pt"),  # v+(trained text), ~705 MB
    ("corpus_capture_tf", "pooled_tf.pt"),  # v+(base text), ~353 MB
    (RTF_TREE, "pooled_tf.pt"),  # v0(trained text) — this round's leg A, ~706 MB
)


def _hub_rtf_units(cfg: Cfg) -> set[str]:
    """Arms whose leg-A store is on the Hub (ONE scoped listing, fail-soft)."""
    from huggingface_hub import HfApi

    prefix = f"{cfg.hf_prefix}/{RTF_TREE}"
    try:
        paths = _hub().list_hf_files_under_path(
            HfApi(), X.HF_DATA_REPO, prefix, repo_type="dataset"
        )
    except Exception as exc:  # noqa: BLE001 — an absent tree means nothing remote
        logger.info("[an] no remote %s tree (%s)", RTF_TREE, type(exc).__name__)
        return set()
    units = {p[len(prefix) + 1 :].split("/")[0] for p in paths if p.endswith("pooled_tf.pt")}
    logger.info("[an] %d leg-A units available on the Hub", len(units))
    return units


def _stage_arm_analysis_stores(cfg: Cfg, arm_id: str) -> list[Path]:
    """Stage ONE arm's analysis stores; return the paths this call downloaded.

    Fleet-wide the grid is 72 x ~1.06 GB, so the decomposition streams it one
    arm at a time (stage -> reduce -> delete) rather than materializing ~76 GB.
    Only the paths this call fetched are returned, so an arm whose stores were
    already local (the pod that captured them) is never deleted.
    """
    hub = _hub()
    fetched: list[Path] = []
    for tree, name in ARM_ANALYSIS_STORES:
        dest = cfg.out_root / tree / arm_id / name
        if dest.exists():
            continue
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{X.HF_PREFIX}/{tree}/{arm_id}/{name}",
            dest,
            repo_type="dataset",
        )
        fetched.append(dest)
    return fetched


def _base_pooled(cfg: Cfg, base_unit: str) -> Path:
    dest = cfg.out_root / "corpus_capture" / base_unit / "pooled.pt"
    if not dest.exists():
        _hub().stage_hub_file(
            X.HF_DATA_REPO,
            f"{X.HF_PREFIX}/corpus_capture/{base_unit}/pooled.pt",
            dest,
            repo_type="dataset",
        )
    return dest


def _decompose_arm(cfg: Cfg, arm_id: str) -> dict:
    """The measured 2x2 for one leg-A arm, at every captured layer.

    Convention (exact, additive):

    * text effect        ``T = v0(trained text) - v0(base text)``
    * function effect    ``F = v+(base text)    - v0(base text)``
    * interaction        ``I = [v+(trained text) - v0(trained text)] - F``
    * on-policy shift    ``D = v+(trained text) - v0(base text) = T + F + I``

    All four cells are averaged over the SAME rows: the sha intersection of the
    base tree's kept rows and the arm's kept rows (the base-text cells are
    row-aligned with each other by construction, as are the trained-text
    cells). Shares are reported two ways — norm ratios ``||term|| / ||D||``
    (not additive) and projection shares ``<term, D> / ||D||^2`` (additive,
    summing to 1).
    """
    base_unit = X.base_unit_for(arm_id)
    base = _store(_base_pooled(cfg, base_unit))
    plus_tt = _store(cfg.out_root / "corpus_capture" / arm_id / "pooled.pt")
    base_tt = _store(cfg.out_root / RTF_TREE / arm_id / "pooled_tf.pt")
    plus_bt = _store(cfg.out_root / "corpus_capture_tf" / arm_id / "pooled_tf.pt")

    b_sha, p_sha, r_sha = (
        list(base["row_sha"]),
        list(plus_tt["row_sha"]),
        list(base_tt["row_sha"]),
    )
    assert list(plus_bt["row_sha"]) == b_sha, (arm_id, "matched-text rows must match the base tree")
    p_ix = {s: i for i, s in enumerate(p_sha)}
    r_ix = {s: i for i, s in enumerate(r_sha)}
    # sha join across ALL FOUR cells (the base-text pair and the trained-text
    # pair are each row-aligned internally; the two trees' kept-row sets differ
    # by their own empty-response drops, and a --smoke leg-A store holds only a
    # row PREFIX — so intersect, never assert equality)
    keep_b = [i for i, s in enumerate(b_sha) if s in p_ix and s in r_ix]
    keep_p = [p_ix[b_sha[i]] for i in keep_b]
    keep_r = [r_ix[b_sha[i]] for i in keep_b]
    assert keep_b, (arm_id, "no shared rows across the four cells")
    if not cfg.smoke:  # production coverage floor (round 1's `load_corpus_cell` bar)
        assert len(keep_b) >= 0.9 * len(b_sha), (arm_id, len(keep_b), len(b_sha))
    bi, pi, ri = np.asarray(keep_b), np.asarray(keep_p), np.asarray(keep_r)

    out: dict[str, dict] = {}
    for layer in cfg.layers:
        v0_bt = _span(base, "response", layer)[bi]
        vp_bt = _span(plus_bt, "response", layer)[bi]
        v0_tt = _span(base_tt, "response", layer)[ri]
        vp_tt = _span(plus_tt, "response", layer)[pi]
        # capture-integrity read: same model + same prompt text as base_content
        ctx_cos = None
        if "context" in base_tt["arms"]:
            c_base = _span(base, "context", layer)[bi]
            c_rev = _span(base_tt, "context", layer)[ri]
            num = np.einsum("ij,ij->i", c_base, c_rev)
            den = np.linalg.norm(c_base, axis=1) * np.linalg.norm(c_rev, axis=1)
            per_row = num / np.where(den == 0, np.nan, den)
            ctx_cos = {
                "min": float(np.nanmin(per_row)),
                "median": float(np.nanmedian(per_row)),
                "mean_vector_cos": _cos(c_base.mean(axis=0), c_rev.mean(axis=0)),
                "gate_min": CONTEXT_IDENTITY_COS_MIN,
                "pass": bool(np.nanmin(per_row) >= CONTEXT_IDENTITY_COS_MIN),
            }
        m0bt, mpbt, m0tt, mptt = (a.mean(axis=0) for a in (v0_bt, vp_bt, v0_tt, vp_tt))
        text, func = m0tt - m0bt, mpbt - m0bt
        inter = (mptt - m0tt) - func
        shift = mptt - m0bt
        n_shift = float(np.linalg.norm(shift))
        terms = {"text": text, "function": func, "interaction": inter}
        # per-row analogue of round 1's `_decomposition_block` (mean row norm)
        row_text, row_func = v0_tt - v0_bt, vp_bt - v0_bt
        row_inter = (vp_tt - v0_tt) - row_func
        row_shift = vp_tt - v0_bt
        sq_tot = float((row_shift**2).sum())
        out[str(layer)] = {
            "n_rows": int(len(bi)),
            "norm_shift": n_shift,
            "identity_residual": float(np.linalg.norm(shift - (text + func + inter))),
            "norms": {k: float(np.linalg.norm(v)) for k, v in terms.items()},
            "norm_ratio": {k: float(np.linalg.norm(v)) / n_shift for k, v in terms.items()},
            "proj_share": {k: float(np.dot(v, shift)) / n_shift**2 for k, v in terms.items()},
            "cos_with_shift": {k: _cos(v, shift) for k, v in terms.items()},
            "cos_text_function": _cos(text, func),
            "per_row_mean_norm": {
                "shift": float(np.linalg.norm(row_shift, axis=1).mean()),
                "text": float(np.linalg.norm(row_text, axis=1).mean()),
                "function": float(np.linalg.norm(row_func, axis=1).mean()),
                "interaction": float(np.linalg.norm(row_inter, axis=1).mean()),
            },
            "per_row_sq_share": {
                "text": float((row_text**2).sum()) / sq_tot if sq_tot else float("nan"),
                "function": float((row_func**2).sum()) / sq_tot if sq_tot else float("nan"),
                "interaction": float((row_inter**2).sum()) / sq_tot if sq_tot else float("nan"),
            },
            "context_identity_cos": ctx_cos,
        }
        try:
            d = _delta_direction(cfg, arm_id, layer)
        except (FileNotFoundError, AssertionError, KeyError) as exc:
            out[str(layer)]["delta_reads"] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        out[str(layer)]["delta_reads"] = {
            "delta_arm": d["delta_arm"],
            "n_mix_rows": d["n_mix_rows"],
            "cos_text_delta": _cos(text, d["delta"]),
            "cos_function_delta": _cos(func, d["delta"]),
            "cos_interaction_delta": _cos(inter, d["delta"]),
            "cos_shift_delta": _cos(shift, d["delta"]),
            "src_ctx": d["src_ctx"],
            "cos_text_panel_write": _cos(text, d["w_panel"]) if "w_panel" in d else None,
        }
    return {"arm_id": arm_id, "base_unit": base_unit, "layers": out}


def _round1_m0_r2(arm_id: str, layer: int) -> float | None:
    """Round 1's committed M0 held-out R^2 for the refit parity check."""
    path = REPO_ROOT / "eval_results" / "issue_1768" / "fits" / f"{arm_id}_L{layer}.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["fits"]["M0"]["heldout_r2"])


def _m0_attribution(cfg: Cfg, arm_id: str, layer: int) -> dict:
    """Round 1's map stand-in for the text effect, vs this round's MEASURED cell.

    Round 1 attributed the text effect to ``input_move = M0(c+) - M0(c0)``
    (`issue1768_fit._decomposition_block`). The fit JSONs persist metrics only,
    not ridge coefficients, so M0 is REFIT here with round 1's own routine on
    the same cell/split; the refit's held-out R^2 is reported next to round 1's
    committed value as the parity check.
    """
    import issue1768_fit as F

    dev = F._device()
    cell = F.load_corpus_cell(arm_id, layer, cfg.out_root)
    tr, val, te = F._split_idx(cell["split"])
    pred, m0_meta, m0_payload = F._fit_map(cell["C0"], cell["V0"], tr, val, te, dev)
    refit_r2 = F._pooled_r2(pred, cell["V0"][te])
    attrib = F._apply_payload(m0_payload, cell["Cplus"][te], dev) - F._apply_payload(
        m0_payload, cell["C0"][te], dev
    )
    rev = _store(cfg.out_root / RTF_TREE / arm_id / "pooled_tf.pt")
    rev_ix = {s: i for i, s in enumerate(list(rev["row_sha"]))}
    te_sha = [cell["sha"][i] for i in te]
    have = [k for k, s in enumerate(te_sha) if s in rev_ix]
    if len(have) < 2:  # a --smoke leg-A prefix rarely reaches the test split
        return {
            "layer": layer,
            "skipped": "fewer than 2 test rows present in the leg-A store",
            "n_test_rows": int(len(have)),
        }
    ri = np.asarray([rev_ix[te_sha[k]] for k in have])
    measured = _span(rev, "response", layer)[ri] - cell["V0"][te][np.asarray(have)]
    attrib = attrib[np.asarray(have)]
    m_mean, a_mean = measured.mean(axis=0), attrib.mean(axis=0)
    # Rows whose trained greedy response is IDENTICAL to the base's have a
    # measured text effect of exactly zero, so their per-row cosine is 0/0.
    # Those rows are excluded from the per-row reads (never coerced) and their
    # count is reported — it IS the fraction of corpus rows on which training
    # changed the emitted text at all.
    m_norm, a_norm = np.linalg.norm(measured, axis=1), np.linalg.norm(attrib, axis=1)
    live = (m_norm > 0) & (a_norm > 0)
    n_live = int(live.sum())
    per_row = np.einsum("ij,ij->i", measured[live], attrib[live]) / (m_norm[live] * a_norm[live])
    per_row_rel = np.linalg.norm(attrib[live] - measured[live], axis=1) / m_norm[live]
    return {
        "layer": layer,
        "n_test_rows": int(len(have)),
        "n_rows_text_changed": n_live,
        "frac_rows_text_changed": n_live / max(1, len(have)),
        "m0_refit_heldout_r2": float(refit_r2),
        "m0_refit_selected_lambda": float(m0_meta["selected_lambda"]),
        "round1_committed_m0_heldout_r2": _round1_m0_r2(arm_id, layer),
        "mean_shift_cos": _cos(m_mean, a_mean),
        "mean_shift_rel_err": float(np.linalg.norm(a_mean - m_mean) / np.linalg.norm(m_mean)),
        "mean_shift_norm_measured": float(np.linalg.norm(m_mean)),
        "mean_shift_norm_attributed": float(np.linalg.norm(a_mean)),
        "per_row_cos_median": float(np.median(per_row)) if n_live else None,
        "per_row_cos_mean": float(per_row.mean()) if n_live else None,
        "per_row_rel_err_median": float(np.median(per_row_rel)) if n_live else None,
    }


def _leg_b_read(cfg: Cfg, arm_id: str, corpus_write: dict[str, np.ndarray] | None) -> dict:
    """Delta v_train = v+(train rows) - v0(train rows), vs delta and the write."""
    import torch

    arm = C._full_arm_index()[arm_id]
    delta_arm = X.delta_arm_for(arm)
    plus = torch.load(
        cfg.out_root / BTF_TREE / arm_id / "tbar_plus.pt", map_location="cpu", weights_only=False
    )
    base = torch.load(
        cfg.out_root / "delta_tf" / delta_arm / "tbar.pt", map_location="cpu", weights_only=False
    )
    out: dict[str, dict] = {}
    for layer in cfg.layers:
        vp = np.asarray(plus["tbar_plus"][layer].float().numpy(), dtype=np.float64)
        v0 = np.asarray(base["tbar"][layer].float().numpy(), dtype=np.float64)
        dv = vp - v0
        rec: dict[str, object] = {
            "norm_delta_v_train": float(np.linalg.norm(dv)),
            "norm_v0_train": float(np.linalg.norm(v0)),
            "n_rows": int(plus["n_rows"]),
        }
        if plus.get("tbar_plus_even") is not None and base.get("tbar_even") is not None:
            e = np.asarray(plus["tbar_plus_even"][layer].float().numpy(), dtype=np.float64)
            o = np.asarray(plus["tbar_plus_odd"][layer].float().numpy(), dtype=np.float64)
            be = np.asarray(base["tbar_even"][layer].float().numpy(), dtype=np.float64)
            bo = np.asarray(base["tbar_odd"][layer].float().numpy(), dtype=np.float64)
            rec["split_half_cos"] = _cos(e - be, o - bo)
        try:
            d = _delta_direction(cfg, arm_id, layer)
            rec["cos_delta_v_train_delta"] = _cos(dv, d["delta"])
            if "w_panel" in d:
                rec["cos_delta_v_train_panel_write"] = _cos(dv, d["w_panel"])
            rec["delta_arm"] = d["delta_arm"]
        except (FileNotFoundError, AssertionError, KeyError) as exc:
            rec["delta_error"] = f"{type(exc).__name__}: {exc}"
        if corpus_write is not None and str(layer) in corpus_write:
            rec["cos_delta_v_train_corpus_matched_write"] = _cos(dv, corpus_write[str(layer)])
        out[str(layer)] = rec
    return {"arm_id": arm_id, "method": arm.method, "delta_arm": delta_arm, "layers": out}


def _corpus_matched_write(cfg: Cfg, arm_id: str, base_means: dict) -> dict[str, np.ndarray]:
    """F over the corpus = mean(v+(base text)) - mean(v0(base text)), streamed.

    The matched-text tree shares the base tree's kept rows in order, so the two
    row means are matched by construction; the arm's 353 MB store is staged,
    reduced to three vectors, and deleted.
    """
    hub = _hub()
    base_unit = X.base_unit_for(arm_id)
    dest = cfg.out_root / "corpus_capture_tf" / arm_id / "pooled_tf.pt"
    staged_here = not dest.exists()
    if staged_here:
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{X.HF_PREFIX}/corpus_capture_tf/{arm_id}/pooled_tf.pt",
            dest,
            repo_type="dataset",
        )
    try:
        st = _store(dest)
        assert list(st["row_sha"]) == base_means[base_unit]["row_sha"], (arm_id, "row-set drift")
        return {
            str(layer): _span(st, "response", layer).mean(axis=0)
            - base_means[base_unit]["mean"][str(layer)]
            for layer in cfg.layers
        }
    finally:
        if staged_here:
            dest.unlink(missing_ok=True)


def phase_analysis(cfg: Cfg) -> None:
    _phase("an_analysis")
    picks = _picks(cfg)
    res = cfg.out_root / RESULTS_DIR
    (res / "per_arm").mkdir(parents=True, exist_ok=True)
    (res / "leg_b").mkdir(parents=True, exist_ok=True)
    _stage_base_panels(cfg)

    base_means: dict[str, dict] = {}
    for bu in sorted({X.base_unit_for(a) for a in _btf_arms(cfg)}):
        st = _store(_base_pooled(cfg, bu))
        base_means[bu] = {
            "row_sha": list(st["row_sha"]),
            "mean": {str(layer): _span(st, "response", layer).mean(axis=0) for layer in cfg.layers},
        }
        del st
        logger.info("[an] base means reduced for %s", bu)

    # leg-A arms whose reverse-tree store EXISTS — locally (the capturing host)
    # or on the Hub (an analysis-only host streams it per arm). Fleet-wide runs
    # resume across waves, so the analysable set is whatever has landed.
    have_rtf = [a for a in picks if (cfg.out_root / RTF_TREE / a / "pooled_tf.pt").exists()]
    if len(have_rtf) < len(picks):
        remote_rtf = _hub_rtf_units(cfg)
        have_rtf = [a for a in picks if a in remote_rtf or a in set(have_rtf)]
    if len(have_rtf) < len(picks):
        logger.info("[an] %d/%d leg-A stores present; analysing those", len(have_rtf), len(picks))
    decomp, attrib = {}, {}
    for k, arm_id in enumerate(have_rtf):
        t0 = time.time()
        # resume: a completed arm's per-arm JSONs ARE the durable record (every
        # read here is deterministic), so a relaunch reloads instead of
        # re-staging ~1.8 GB and refitting M0 — 72 units x ~90 s is well past
        # the checkpoint-per-phase intra-phase floor
        dpath = res / "per_arm" / f"{arm_id}_2x2.json"
        apath = res / "per_arm" / f"{arm_id}_attrib.json"
        if dpath.exists() and apath.exists():
            decomp[arm_id] = json.loads(dpath.read_text())
            attrib[arm_id] = json.loads(apath.read_text())
            print(f"[an] decomposition {k + 1}/{len(have_rtf)} {arm_id} resumed", flush=True)
            _status(cfg, "an_analysis", decomposed=k + 1, total=len(have_rtf))
            continue
        fetched = _stage_arm_analysis_stores(cfg, arm_id)  # stream: stage -> reduce -> delete
        try:
            decomp[arm_id] = _decompose_arm(cfg, arm_id)
            C._atomic_json(dpath, decomp[arm_id])
            try:
                attrib[arm_id] = _m0_attribution(cfg, arm_id, cfg.attrib_layer)
            except Exception as exc:  # noqa: BLE001 — validation read, never kills the round
                attrib[arm_id] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "layer": cfg.attrib_layer,
                }
                logger.warning("[an] M0 attribution failed for %s: %s", arm_id, exc)
            C._atomic_json(apath, {"arm_id": arm_id, **attrib[arm_id]})
        finally:
            for p in fetched:
                p.unlink(missing_ok=True)
        print(
            f"[an] decomposition {k + 1}/{len(have_rtf)} {arm_id} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        _status(cfg, "an_analysis", decomposed=k + 1, total=len(have_rtf))

    leg_b = {}
    arms_b = [a for a in _btf_arms(cfg) if (cfg.out_root / BTF_TREE / a / "tbar_plus.pt").exists()]
    for k, arm_id in enumerate(arms_b):
        t0 = time.time()
        bpath = res / "leg_b" / f"{arm_id}.json"
        if bpath.exists():  # same resume contract as the decomposition loop
            leg_b[arm_id] = json.loads(bpath.read_text())
            print(f"[an] leg-b {k + 1}/{len(arms_b)} {arm_id} resumed", flush=True)
            continue
        try:
            write = _corpus_matched_write(cfg, arm_id, base_means)
        except Exception as exc:  # noqa: BLE001 — optional third cosine
            logger.warning("[an] corpus matched write unavailable for %s: %s", arm_id, exc)
            write = None
        leg_b[arm_id] = _leg_b_read(cfg, arm_id, write)
        C._atomic_json(bpath, leg_b[arm_id])
        print(
            f"[an] leg-b {k + 1}/{len(arms_b)} {arm_id} elapsed={time.time() - t0:.0f}s", flush=True
        )
        _status(cfg, "an_analysis", leg_b=k + 1, total_b=len(arms_b))

    C._atomic_json(
        res / "summary.json",
        {
            "convention": {
                "text_effect": "v0(trained text) - v0(base text)",
                "function_effect": "v+(base text) - v0(base text)",
                "interaction": "[v+(trained text) - v0(trained text)] - function_effect",
                "on_policy_shift": "v+(trained text) - v0(base text) = text + function + interaction",
                "delta": "round-1 delta_primary = tbar(base on mix positives) - v0_panel_half_B",
                "leg_b_delta_v_train": "v+(train rows) - v0(train rows)",
                "shares": "norm_ratio = ||term||/||shift|| (not additive); "
                "proj_share = <term, shift>/||shift||^2 (additive, sums to 1)",
            },
            "layers": list(cfg.layers),
            "attrib_layer": cfg.attrib_layer,
            "rtf_arms": have_rtf,
            "rtf_arms_requested": picks,
            "rtf_subset_arms": _subset_picks(cfg),
            "btf_arms": arms_b,
            "decomposition": decomp,
            "m0_attribution_vs_measured": attrib,
            "leg_b": leg_b,
            **_meta(),
        },
    )
    _status(cfg, "an_analysis", done=True)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _import_check() -> int:
    """Resolve every deferred import this driver reaches on its REAL path."""
    import issue1768_directions as D  # noqa: F401
    import issue1768_fit as F  # noqa: F401
    import torch  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401

    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        _teacher_forced_span_means,
        compute_prompt_spans,
    )
    from explore_persona_space.orchestrate import hub  # noqa: F401
    from explore_persona_space.orchestrate.preflight import (  # noqa: F401
        assert_out_root_headroom,
    )

    for name in (
        "_read_rows_with_spans",
        "_mix_positive_rows",
        "_resolve_unit_model",
        "_cleanup_merged",
        "_fp16_roundtrip_cos_min",
        "_probe_mix_sources",
        "_ft_ckpt_incomplete_reason",
        "_ft_ckpt_dirs",
        "_physical_gpus",
        "_atomic_json",
        "_full_arm_index",
        "_meta",
        "_device",
        "_dtype",
    ):
        assert hasattr(C, name), f"issue1768_capture.{name} missing"
    for name in ("panel_write_legs", "delta_leg"):
        assert hasattr(D, name), f"issue1768_directions.{name} missing"
    for name in ("load_corpus_cell", "_split_idx", "_fit_map", "_apply_payload", "_device"):
        assert hasattr(F, name), f"issue1768_fit.{name} missing"
    for name in ("stage_hub_file", "list_hf_files_under_path", "verify_repo_paths_uploaded"):
        assert hasattr(hub, name), f"orchestrate.hub.{name} missing"
    print("[import-check] ok")
    return 0


def parse_args(argv: list[str] | None) -> tuple[Cfg, argparse.Namespace]:
    p = argparse.ArgumentParser(description="issue 1768 inline (model x text) 2x2 round")
    p.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1768_tx"))
    p.add_argument("--phases", default="p0,rtf,btf,up,an")
    p.add_argument("--unit", default=None, help="run ONE unit: <rtf|btf>:<arm_id>")
    p.add_argument("--rtf-arms", default="")
    p.add_argument(
        "--rtf-all", action="store_true", help="leg A over ALL 72 arms (fleet-wide extension)"
    )
    p.add_argument("--btf-arms", default="")
    p.add_argument(
        "--rtf-wave-size",
        type=int,
        default=0,
        help="upload + verify after every N completed leg-A units (0 = one terminal upload)",
    )
    p.add_argument(
        "--no-restage",
        action="store_true",
        help="skip pulling this round's already-uploaded stores back into the out-root",
    )
    p.add_argument("--layers", default=",".join(str(x) for x in X.LAYERS))
    p.add_argument("--tf-batch", type=int, default=X.TF_BATCH_SIZE)
    p.add_argument("--attrib-layer", type=int, default=19)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-rows", type=int, default=24)
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--import-check", action="store_true")
    a = p.parse_args(argv)
    cfg = Cfg(
        out_root=a.out_root,
        phases=tuple(x for x in a.phases.split(",") if x),
        rtf_arms=tuple(x for x in a.rtf_arms.split(",") if x),
        rtf_all=a.rtf_all,
        btf_arms=tuple(x for x in a.btf_arms.split(",") if x),
        rtf_wave_size=a.rtf_wave_size,
        restage=not a.no_restage,
        layers=tuple(int(x) for x in a.layers.split(",") if x),
        tf_batch=a.tf_batch,
        smoke=a.smoke,
        smoke_rows=a.smoke_rows,
        upload=not a.no_upload,
        attrib_layer=a.attrib_layer,
        gpu_id=a.gpu_id,
    )
    return cfg, a


def main(argv: list[str] | None = None) -> int:
    cfg, args = parse_args(argv)
    if args.import_check:
        return _import_check()
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    if args.unit:
        run_unit(cfg, args.unit)
        return 0
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    for phase in cfg.phases:
        if phase in PHASE_HEADROOM_GB:
            need = PHASE_HEADROOM_GB[phase] if not cfg.smoke else 2.0
            assert_out_root_headroom(cfg.out_root, need, phase=phase)
        if phase == "p0":
            phase_p0(cfg)
        elif phase == "rtf":
            _fanout(cfg, "rtf", "rtf_reverse_matched_capture")
        elif phase == "btf":
            _fanout(cfg, "btf", "btf_train_rows_capture")
        elif phase == "up":
            if cfg.upload:
                phase_upload(cfg)
        elif phase == "an":
            phase_analysis(cfg)
        else:
            raise ValueError(f"unknown phase {phase}")
    _status(cfg, "done")
    _phase("done")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit: PyGILState_Release finalize-race guard (#1689)
