"""#1979 F1 GPU compute-phase driver (plan v2 §4 F1, unit 1b).

Consumes the unit-1a config manifests (``config/{prefix_panel,queries,arms,
wmap_selection}.json`` — local ``--config-dir`` first, else staged from
``issue1979_prefixrace/config/`` on the HF data repo) and runs the five F1
sub-phases over the (model-state x prefix x query) grid as INDEPENDENT
work-items, round-robined work-conservingly across every visible GPU
(per-item subprocess with a launcher-env ``CUDA_VISIBLE_DEVICES`` pin — the
#523/#545 CVD clobber rule; no wave barrier):

- **f1a** on-policy generation + capture (20 units): batched greedy vLLM over
  all rendered (prefix, query) prompts per model state (merge->generate->
  capture->delete rotation via the #1768 ``_merge_adapter`` path), then HF
  teacher-forced span-means (prefix/context/response) + last-token
  (``last_prompt``/``last_ctx``) vectors at layers 14/19/25, fp16, sha-keyed;
  raw rows shard to ``raw_completions/generation/`` (<9 MB line-split).
- **f1b** matched-text TF trees (30 units): content/marker arms TF'd on their
  base unit's rows (weights-carried writes) + the three-space marker slot DV
  via ``eval/marker_logprob.compute_marker_slot_stats`` (four floats per slot
  per side; in-process ``encode(" ※") == [83399]`` assert; gauge assert);
  per-marker-arm #1900 frame-free identity gate BEFORE the fleet pass.
- **f1c** anchor captures (~14 mixes x 20 rows @ seed 1979): ``delta_tf``
  pools through ``_mix_positive_rows`` -> A_ctx/A_ans + per-row pools +
  even/odd half-anchors (base model).
- **f1d** M0 re-materialization at the PINNED n=15k rows (span-mean via
  ``issue1768_fit.pilot_m0_fit``; last_prompt via the lt-round loaders +
  ``_fit_map`` — same code paths) + the L19 union refit; parity asserts.
- **f1e** predictor/battery ingredient tables (batched einsums, batched SVD,
  two-GEMM null draws) + **f1f** judge-input extraction.
- **f1g** (amendment `marker-a5-weights-vs-text`, plan v6): base-model TF
  span-means over the 6 marker arms' STORED generations — the missing
  h_base(trained_text) leg (6 ``basetf`` units + 1 ``means`` aggregation
  unit persisting ``battery/basetf_decomp_inputs.pt``). Excluded from the
  ``--phase f1`` default tuple; dispatch with ``--phase f1g``.

Per-unit atomic done-sentinels + resume (skip completed units); per-phase
``assert_out_root_headroom`` with resume-aware pending-set scaling;
incremental per-unit uploads to ``issue1979_prefixrace/``; errors surface
immediately (no try/except-pass, no silent defaults).

CPU-verifiable modes (same code paths as production): ``--plan-only`` loads
the manifests and prints the realized work-item list; ``--import-check``
resolves EVERY deferred import + signature-binds the load-bearing reused
call sites; ``--panel-limit/--query-limit`` slice the grid AT MANIFEST LOAD,
so every downstream consumer sees the sliced grid through the identical path.
``--smoke-subset`` (the dispatch wrapper's SMOKE_FIRST leg, crash-fix r4)
additionally restricts to ONE arm per realized (kind x method) class + the
single f1d (m0, span_mean, L19) fit via the shared ``f1d_fit_specs()``
enumeration, and emits ``[phase=smoke_done]`` instead of the reserved
``[phase=done]`` terminal line.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before any tokenizer/torch import: thread caps + HF credentials

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import issue1768_capture as CAP  # noqa: E402
import issue1768_cells as X  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1979_gpu")

# ── pins + constants (plan v2 §4/§9/§10) ──────────────────────────────────────

SEED = 1979
HF_PREFIX_1979 = "issue1979_prefixrace"
CORPUS_PIN = "c07267285d2cdbf3e0401ddc3e3accae50e496a7"  # issue1768_mapshift round-1 inputs
I1900_PIN = "3bb20debe2e68392897d6144b9180c8748c7afcb"  # issue1900_leakrace config + maps

LAYERS_1979 = (14, 19, 25)
UNION_LAYER = 19
N_ANCHOR_ROWS = 20
N_NULL_DRAWS = 2_000
SIGMA_SHRINKAGE = 0.1  # plan §4 F1e: corpus_sigma shrinkage on the 15k bare rows
MARKER_TEXT = " ※"  # leading-space " ※", Qwen id 83399
GATE_MIN_DELTA_LOGP_NATS = 2.0  # #1900 frame-free identity gate (plan §4 F1b) — LoRA marker arms
GATE_MIN_DELTA_LOGP_NATS_FT = 1.8  # full-FT marker arms (crash-fix r5; stated plan deviation)
FT_GATE_PLAN_DEVIATION = "ft-marker-gate-tolerance-1.8"
GATE_N_CORPUS_ROWS = 50
M0_PARITY_R2_TOL = 0.01
M0_PARITY_COS_MIN = 0.99
M0_PARITY_PROBE_ROWS = 512
CONFIG_FILES = (
    "prefix_panel.json",
    "queries.json",
    "arms.json",
    "wmap_selection.json",
    # f1d-m0-reference-file concern: the lt-round recorded M0 R2 parity file,
    # published to issue1979_prefixrace/config/ by unit 2 (pre-dispatch) so
    # _m0_reference() resolves pod-side via the same staging path.
    "m0_reference.json",
)

# The binding per-render prompt budget (mirrors issue1979_prep — content decode
# needs prompt+1024 <= 4096 and mk decode needs prompt+2048 <= 6144).
BINDING_PROMPT_BUDGET = min(
    X.MAX_MODEL_LEN - X.MAX_NEW_CONTENT, X.PFX_MAX_MODEL_LEN_RAISED - X.MAX_NEW_MARKER
)

# per-phase disk floors (decimal GB, plan §9 "Disk / out-root" row; f1a scales
# by the pending fraction — resume-aware pending-set scaling; f1g = the
# marker-a5-weights-vs-text amendment, plan v6 §9: peak out-root ≈ 24 GB)
PHASE_HEADROOM_GB = {
    "f1a": 150.0,
    "f1b": 40.0,
    "f1c": 10.0,
    "f1d": 30.0,
    "f1e": 10.0,
    "f1f": 5.0,
    "f1g": 40.0,
}
WORKER_HEADROOM_GB = 5.0
MAX_HEAVY_MODEL_CONCURRENT = 2  # ≤2 coexisting merged/staged model dirs per node (plan §9)
# per-unit failure budget (crash-fix r3, job 16717): one bad unit must not kill
# the whole job — collect failures, keep scheduling INDEPENDENT units, abort only
# past the budget or on a systemic pattern; failed units stay resumable (no done
# sentinel), and the run still exits non-zero when ANY unit failed.
FAILURE_BUDGET = 5  # abort when MORE than this many units fail (matches the observed blast radius)
SYSTEMIC_EXC_REPEAT = 3  # same exception class this many times = systemic -> abort early

SHARD_BYTE_BUDGET = 8_500_000  # <9 MB raw-text shards (non-LFS path)


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _meta() -> dict:
    return {**CAP._meta(), "driver": "issue1979_gpu.py", "seed": SEED}


# ── configuration ─────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Cfg:
    """F1 run configuration; every output-affecting knob is part of the regime."""

    out_root: Path
    config_dir: Path
    phases: tuple[str, ...]
    panel_limit: int | None = None
    query_limit: int | None = None
    arms_filter: tuple[str, ...] = ()
    skip_upload: bool = False
    tf_batch: int = X.TF_BATCH_SIZE
    skip_m0_ref_parity: bool = False
    max_parallel: int | None = None
    gpu_id: int = 0  # informational; the launcher env CVD pin selects the GPU
    # smoke-first leg (crash-fix r4): one arm per realized (kind × method)
    # class + f1d restricted to the (m0, span_mean, L19) fit — threaded through
    # build_work_items AND run_f1e via the shared f1d_fit_specs() enumeration.
    smoke_subset: bool = False

    @property
    def limited(self) -> bool:
        return self.panel_limit is not None or self.query_limit is not None

    def cap_cfg(self) -> CAP.Cfg:
        """CAP.Cfg shim so the #1768 model-resolution/merge machinery is reused verbatim."""
        return CAP.Cfg(
            out_root=self.out_root,
            phases=(),
            arms=(),
            layers=tuple(LAYERS_1979),
            tf_batch=self.tf_batch,
            upload=False,
            hf_prefix=HF_PREFIX_1979,
        )

    def worker_flags(self) -> list[str]:
        flags = ["--out-root", str(self.out_root), "--config-dir", str(self.config_dir)]
        if self.panel_limit is not None:
            flags += ["--panel-limit", str(self.panel_limit)]
        if self.query_limit is not None:
            flags += ["--query-limit", str(self.query_limit)]
        if self.arms_filter:
            flags += ["--arms", ",".join(self.arms_filter)]
        if self.skip_upload:
            flags += ["--skip-upload"]
        if self.skip_m0_ref_parity:
            flags += ["--skip-m0-ref-parity"]
        if self.smoke_subset:
            flags += ["--smoke-subset"]  # workers need f1d_fit_specs' restricted view (run_f1e)
        flags += ["--tf-batch", str(self.tf_batch)]
        return flags


# ── manifests (unit 1a outputs; the slice caps plug in HERE, one path) ────────


def load_manifests(cfg: Cfg) -> dict:
    """Load + schema-assert the F0 manifests; apply --panel-limit/--query-limit.

    The slice happens at LOAD, before any consumer, so the tiny-slice smoke and
    the full run execute the SAME downstream code path (brief requirement).
    """
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_FILES:
        p = cfg.config_dir / name
        if not p.exists():
            from explore_persona_space.orchestrate import hub

            logger.info("[config] staging %s from %s/config/", name, HF_PREFIX_1979)
            hub.stage_hub_file(
                X.HF_DATA_REPO, f"{HF_PREFIX_1979}/config/{name}", p, repo_type="dataset"
            )
    panel = json.loads((cfg.config_dir / "prefix_panel.json").read_text())
    queries = json.loads((cfg.config_dir / "queries.json").read_text())
    arms_manifest = json.loads((cfg.config_dir / "arms.json").read_text())
    wmap = json.loads((cfg.config_dir / "wmap_selection.json").read_text())

    members = panel["members"]
    assert members and isinstance(members, list), "prefix_panel.json: empty members"
    for m in members:
        for k in ("prefix_id", "system", "prefix_turns", "user_wrap"):
            assert k in m, f"prefix_panel member missing key {k!r}: {sorted(m)}"
    qrows = queries["queries"]
    assert qrows and isinstance(qrows, list), "queries.json: empty queries"
    for q in qrows:
        assert "prompt" in q and "sha" in q, f"queries.json row missing prompt/sha: {sorted(q)}"

    if cfg.panel_limit is not None:
        members = members[: cfg.panel_limit]
    if cfg.query_limit is not None:
        qrows = qrows[: cfg.query_limit]

    # The #1900 arms.json rows carry the resolution coordinates explicitly
    # (adapter_repo/adapter_subfolder | ft_repo/ft_subfolder, base_unit,
    # mix_arm_id/mix_layout/mix_pos_path) — the driver resolves models from the
    # ROW fields, never from the #1768 fleet registry (different arm universe).
    arm_rows = arms_manifest["arms"]
    assert isinstance(arm_rows, list) and arm_rows, "arms.json shape drift (no 'arms' list)"
    for r in arm_rows:
        need = ("arm_id", "kind", "method", "base_unit", "mix_arm_id", "mix_layout", "mix_pos_path")
        for k in need:
            assert k in r, f"arms.json row missing {k!r}: {sorted(r)}"
        if r["method"] == "ft":
            assert "ft_repo" in r and "ft_subfolder" in r, f"ft row lacks repo: {r['arm_id']}"
        else:
            assert "adapter_repo" in r and "adapter_subfolder" in r, (
                f"lora row lacks adapter coords: {r['arm_id']}"
            )
    if cfg.arms_filter:
        keep = set(cfg.arms_filter)
        arm_rows = [r for r in arm_rows if r["arm_id"] in keep]
    content = [r for r in arm_rows if r["kind"] == "content"]
    marker = [r for r in arm_rows if r["kind"] == "marker"]
    if not cfg.arms_filter and not cfg.limited:
        assert (len(content), len(marker)) == (12, 6), (
            f"arms.json split drifted: {len(content)} content / {len(marker)} marker (want 12/6)"
        )
    return {
        "members": members,
        "queries": qrows,
        "content_arms": content,
        "marker_arms": marker,
        "arm_rows": {r["arm_id"]: r for r in arm_rows},
        "wmap": wmap,
        "panel_meta": {k: panel.get(k) for k in ("n_members", "seed", "limits", "pins")},
    }


def ensure_arm_registry(cfg: Cfg, manifests: dict) -> None:
    """Write out_root/arm_registry.json (mix_pos_sources) for CAP._mix_positive_rows,
    from the arms.json rows' own mix coordinates (the pinned #1900 fields)."""
    reg_path = cfg.out_root / "arm_registry.json"
    if reg_path.exists():
        return
    sources = {
        r["arm_id"]: {"pos_path": r["mix_pos_path"], "layout": r["mix_layout"]}
        for r in manifests["arm_rows"].values()
    }
    CAP._atomic_json(reg_path, {"mix_pos_sources": sources, **_meta()})


# ── work-item registry ────────────────────────────────────────────────────────


@dataclasses.dataclass
class Item:
    key: str  # e.g. "f1a:syc-pers-po"
    phase: str  # f1a..f1f
    deps: tuple[str, ...] = ()
    model_key: str | None = None  # co-scheduling guard: same key never concurrent
    heavy_model: bool = False  # counts against the ≤2 merged/staged dirs budget


def _mixes(manifests: dict) -> dict[str, dict]:
    """mix_id -> representative arm ROW (first arm mapping to that delta mix)."""
    mixes: dict[str, dict] = {}
    for row in manifests["content_arms"] + manifests["marker_arms"]:
        mixes.setdefault(row["mix_arm_id"], row)
    return mixes


def f1d_fit_specs(cfg: Cfg) -> list[tuple[str, str, int]]:
    """The realized F1d fit list — the ONE enumeration ``build_work_items`` AND
    ``run_f1e`` share, so unit selection and f1e's map-payload reads cannot
    diverge (crash-fix r4 smoke-first leg).

    Full grid: m0 at 2 positions × LAYERS_1979 + union at 2 positions × L19
    (8 fits). Under ``cfg.smoke_subset``: the single (m0, span_mean, L19) fit
    — same code path, one unit ("f1d span-mean L19 only", r4 brief)."""
    if cfg.smoke_subset:
        return [("m0", "span_mean", UNION_LAYER)]
    specs = [("m0", pos, layer) for pos in ("span_mean", "last_prompt") for layer in LAYERS_1979]
    specs += [("union", pos, UNION_LAYER) for pos in ("span_mean", "last_prompt")]
    return specs


def derive_smoke_arms(manifests: dict) -> tuple[str, ...]:
    """Smoke-first arm subset: the FIRST arm of each realized (kind × method)
    class in arms.json — one lora content, one ft-mapped content, one marker
    per method present ("ONE unit per arm class", r4 brief). Deterministic
    (manifest order), so dispatcher + workers derive the same set."""
    picks: dict[tuple[str, str], str] = {}
    for row in manifests["content_arms"] + manifests["marker_arms"]:
        cls = (row["kind"], "ft" if row["method"] == "ft" else "lora")
        picks.setdefault(cls, row["arm_id"])
    assert picks, "derive_smoke_arms: no arms realized in manifests"
    return tuple(picks.values())


def _merge_adapter_row(cfg: Cfg, row: dict) -> Path:
    """Merge the row's HF adapter onto base -> local merged dir (atomic publish;
    complete-dir reuse) — the #1768 ``_merge_adapter`` recipe on the arms.json
    row's OWN (adapter_repo, adapter_subfolder) coordinates."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    arm_id = row["arm_id"]
    merged_dir = cfg.out_root / "merged" / arm_id
    if (merged_dir / "config.json").exists():
        logger.info("[merge] %s: complete merged dir reused", arm_id)
        return merged_dir
    logger.info("[merge] %s <- %s/%s", arm_id, row["adapter_repo"], row["adapter_subfolder"])
    base = AutoModelForCausalLM.from_pretrained(
        X.BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    peft_model = PeftModel.from_pretrained(
        base, row["adapter_repo"], subfolder=row["adapter_subfolder"]
    )
    merged = peft_model.merge_and_unload()
    tmp = merged_dir.parent / f".tmp_{arm_id}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    merged.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(X.BASE_MODEL).save_pretrained(tmp)
    del merged, peft_model, base
    gc.collect()
    try:
        os.replace(tmp, merged_dir)
    except OSError:
        if (merged_dir / "config.json").exists():  # concurrent finisher won
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            raise
    return merged_dir


def _resolve_state_model(cfg: Cfg, manifests: dict, state: str) -> tuple[str, Path | None]:
    """(model_path, local_dir_to_cleanup) for one model state — base repo id,
    row-merged LoRA dir, or the staged full-FT checkpoint dir."""
    if state.startswith("base_"):
        return X.BASE_MODEL, None
    row = manifests["arm_rows"][state]
    if row["method"] == "ft":
        from explore_persona_space.orchestrate import hub

        dest = cfg.out_root / "ft_stage" / state
        ckpt = dest / row["ft_subfolder"]  # verbatim prefix mirror -> rebind consumer
        if not (ckpt / "config.json").exists():
            hub.stage_hub_prefix(row["ft_repo"], row["ft_subfolder"], dest, repo_type="model")
        assert (ckpt / "config.json").exists(), (
            f"staged FT checkpoint incomplete for {state}: no config.json under {ckpt}"
        )
        return str(ckpt), dest
    merged = _merge_adapter_row(cfg, row)
    return str(merged), merged


def build_work_items(cfg: Cfg, manifests: dict) -> list[Item]:
    """The realized F1 work-item list (plan §9: ~77 independent items)."""
    content = [r["arm_id"] for r in manifests["content_arms"]]
    marker = [r["arm_id"] for r in manifests["marker_arms"]]
    items: list[Item] = []

    # f1a: 12 content arms + base_content, 6 marker arms + base_mk (20 units)
    for state in ["base_content", *content, "base_mk", *marker]:
        base = state.startswith("base_")
        items.append(
            Item(
                key=f"f1a:{state}",
                phase="f1a",
                model_key=None if base else state,
                heavy_model=not base,
            )
        )
    # f1b: marker identity gate first (per marker arm), then the fleet passes
    for arm in marker:
        items.append(
            Item(
                key=f"f1b:gate:{arm}",
                phase="f1b",
                deps=(f"f1a:base_mk",),
                model_key=arm,
                heavy_model=True,
            )
        )
    for arm in content:  # 12 content TF-on-base_content-rows write passes
        items.append(
            Item(
                key=f"f1b:w:{arm}",
                phase="f1b",
                deps=("f1a:base_content",),
                model_key=arm,
                heavy_model=True,
            )
        )
    for arm in marker:  # 6 marker TF-on-base_mk-rows write passes
        items.append(
            Item(
                key=f"f1b:wmk:{arm}",
                phase="f1b",
                deps=("f1a:base_mk", f"f1b:gate:{arm}"),
                model_key=arm,
                heavy_model=True,
            )
        )
    # marker P7 (plan §5: "base log P at base slot"): ONE base-model slot pass
    # on base_mk's OWN generated rows, shared by all 6 marker arms (mirrors
    # #1900's base__on__base_mk read; run_unit routes f1b:slotbase:<unit>).
    if marker:
        items.append(Item(key="f1b:slotbase:base_mk", phase="f1b", deps=("f1a:base_mk",)))
    for arm in marker:  # 6 base-on-marker-arm-text + 6 marker-own-text slot passes
        items.append(Item(key=f"f1b:slotbase:{arm}", phase="f1b", deps=(f"f1a:{arm}",)))
        items.append(
            Item(
                key=f"f1b:slotown:{arm}",
                phase="f1b",
                deps=(f"f1a:{arm}", f"f1b:gate:{arm}"),
                model_key=arm,
                heavy_model=True,
            )
        )
    # f1c: one anchor capture per mix (base model)
    for mix in sorted(_mixes(manifests)):
        items.append(Item(key=f"f1c:{mix}", phase="f1c"))
    # f1d: M0 re-materialization (2 positions x 3 layers) + union refit (2 x L19);
    # the realized fit list comes from f1d_fit_specs(cfg) — the SAME enumeration
    # run_f1e loads payloads from (smoke_subset restricts both in lockstep).
    items.append(Item(key="f1d:stage", phase="f1d"))
    for mkind, pos, layer in f1d_fit_specs(cfg):
        deps = ("f1d:stage",) if mkind == "m0" else ("f1d:stage", "f1a:base_content")
        items.append(Item(key=f"f1d:{mkind}:{pos}:{layer}", phase="f1d", deps=deps))
    # f1e: one batched tables item over everything; f1f: judge inputs
    f1e_deps = tuple(
        it.key
        for it in items
        if it.phase in ("f1a", "f1b", "f1c", "f1d") and not it.key.startswith("f1b:gate")
    )
    items.append(Item(key="f1e:tables", phase="f1e", deps=f1e_deps))
    judged = ["base_content", *content]  # the 13 judged states (plan §4 F1f)
    items.append(Item(key="f1f:judge_inputs", phase="f1f", deps=tuple(f"f1a:{s}" for s in judged)))
    # f1g (amendment marker-a5-weights-vs-text, plan v6 §4): base-model TF over
    # the marker arms' STORED generations + the decomposition-inputs means unit.
    # Self-contained (inputs Hub-staged) — scheduled only when --phase names
    # f1g (the parent's `--phase f1` tuple excludes it; main() filters by phase).
    for arm in marker:
        items.append(Item(key=f"f1g:basetf:{arm}", phase="f1g"))
    if marker:
        items.append(
            Item(key="f1g:means", phase="f1g", deps=tuple(f"f1g:basetf:{a}" for a in marker))
        )
    return items


# ── sentinels / resume ────────────────────────────────────────────────────────


def _sentinel_path(cfg: Cfg, key: str) -> Path:
    return cfg.out_root / "done" / (key.replace(":", "__") + ".json")


def _write_sentinel(cfg: Cfg, key: str, wall_s: float, outputs: list[str]) -> None:
    _sentinel_path(cfg, key).parent.mkdir(parents=True, exist_ok=True)
    CAP._atomic_json(
        _sentinel_path(cfg, key),
        {"key": key, "rc": 0, "wall_s": round(wall_s, 1), "outputs": outputs, **_meta()},
    )


def _done(cfg: Cfg, key: str) -> bool:
    return _sentinel_path(cfg, key).exists()


def _failure_path(cfg: Cfg, key: str) -> Path:
    return cfg.out_root / "failed" / (key.replace(":", "__") + ".json")


def _write_failure(cfg: Cfg, key: str, exc: BaseException) -> None:
    """Worker-side failure breadcrumb (exception class for the dispatcher's
    systemic-failure detector). NOT a done sentinel — a failed unit writes no
    ``done/`` sentinel, so resume re-runs exactly the failed/pending units."""
    CAP._atomic_json(
        _failure_path(cfg, key),
        {
            "key": key,
            "exc_class": type(exc).__name__,
            "exc_msg": str(exc)[:500],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def _read_failure_class(cfg: Cfg, key: str) -> str:
    """Exception class from the worker's failure breadcrumb; 'unknown' when the
    worker died without writing one (SIGKILL/OOM) or the file is unreadable —
    the failure itself is already recorded via the nonzero rc, this string only
    feeds the systemic-repeat detector."""
    p = _failure_path(cfg, key)
    if not p.exists():
        return "unknown"
    try:
        return str(json.loads(p.read_text()).get("exc_class", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "unknown"


# ── shared helpers (rendering, capture, storage, upload) ──────────────────────


def _member_render_kwargs(member: dict) -> dict:
    return {
        "system": member["system"],
        "user_wrap": member["user_wrap"],
        "prior_turns": tuple(
            {"role": t["role"], "content": t["content"]} for t in member["prefix_turns"]
        ),
    }


def _row_sha(prefix_id: str, query_sha: str) -> str:
    return _sha16(f"{prefix_id}||{query_sha}")


def _write_shards(out_dir: Path, stem: str, rows: list[dict]) -> list[Path]:
    """Line-split JSONL shards under the <9 MB non-LFS budget (upload-policy)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buf: list[str] = []
    size = 0

    def _flush() -> None:
        nonlocal buf, size
        if not buf:
            return
        p = out_dir / f"{stem}.shard{len(paths):02d}.jsonl"
        p.write_text("".join(buf), encoding="utf-8")
        paths.append(p)
        buf, size = [], 0

    for r in rows:
        line = json.dumps(r, ensure_ascii=False) + "\n"
        if size + len(line.encode("utf-8")) > SHARD_BYTE_BUDGET:
            _flush()
        buf.append(line)
        size += len(line.encode("utf-8"))
    _flush()
    return paths


def _upload_paths(cfg: Cfg, paths: list[Path], dest_prefix: str) -> None:
    """Per-unit incremental upload (persist-by-default; retried; fail-loud)."""
    if cfg.skip_upload:
        logger.info("[upload] skipped (--skip-upload): %s (%d files)", dest_prefix, len(paths))
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    for p in paths:
        rel = f"{dest_prefix}/{p.name}"
        hub.retry_transient(
            lambda p=p, rel=rel: api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=rel,
                repo_id=X.HF_DATA_REPO,
                repo_type="dataset",
            ),
            what=f"upload {rel}",
        )
    logger.info("[upload] %s: %d files", dest_prefix, len(paths))


def _load_unit_rows(cfg: Cfg, state: str) -> list[dict]:
    """Full gen rows (token ids + spans) persisted by the state's f1a unit."""
    rows_dir = cfg.out_root / "gen" / state
    shards = sorted(rows_dir.glob("rows.shard*.jsonl"))
    assert shards, f"f1a rows missing for {state} under {rows_dir} (dep sentinel violated?)"
    rows: list[dict] = []
    for p in shards:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _capture_positions(model, rows: list[dict], layers: list[int], device: str, pad_id: int):
    """Last-token vectors (last_prompt/last_ctx) — the #1768 lasttoken hook recipe.

    Prompt-only forwards, RIGHT pad (positions index naturally from 0), hooks on
    ``model.model.layers[li]``; returns {pos: {layer: Tensor(n, hidden) fp32 cpu}}.

    Positions: ``last_prompt`` (end of the full rendered prompt incl. the
    assistant header), ``last_ctx`` (last context token = end of the user
    query), and ``last_prefix`` (last PREFIX token — plan §4 grain definition:
    "prefix vector = span-mean over prefix tokens + last-prefix-token"; the
    F3 prefix-based v_P->v_A mapping arm consumes it).
    """
    import torch

    positions = ("last_prompt", "last_ctx", "last_prefix")
    captured: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook_fn(module, inp, out):
            captured[li] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook_fn

    hooks = [model.model.layers[li].register_forward_hook(make_hook(li)) for li in layers]
    hidden = model.config.hidden_size
    pooled: dict[str, dict[int, list]] = {p: {li: [] for li in layers} for p in positions}
    batch_size = 8
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            seqs = [r["prompt_token_ids"] for r in batch]
            max_len = max(len(s) for s in seqs)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                attn[i, : len(s)] = 1
            with torch.no_grad():
                model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
            for li in layers:
                hs = captured[li]
                assert hs.shape[:2] == (len(batch), max_len), (hs.shape, len(batch), max_len)
                for i, r in enumerate(batch):
                    idx = {
                        "last_prompt": len(r["prompt_token_ids"]) - 1,
                        "last_ctx": r["context_len"] - 1,
                        "last_prefix": r["prefix_len"] - 1,
                    }
                    for pos in positions:
                        j = idx[pos]
                        assert 0 <= j < len(r["prompt_token_ids"]), (pos, j, r["row_sha"])
                        vec = hs[i, j, :].float().cpu()
                        assert vec.shape == (hidden,), (vec.shape, hidden)
                        pooled[pos][li].append(vec)
    finally:
        for h in hooks:
            h.remove()
        captured.clear()
    return {pos: {li: torch.stack(vs) for li, vs in per.items()} for pos, per in pooled.items()}


def _save_store(path: Path, unit: str, tree: str, rows: list[dict], spans, positions) -> None:
    """fp16 store, atomic publish; sha-keyed row order (the #1768 store shape)."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    store = {
        "schema_version": 1,
        "unit": unit,
        "tree": tree,
        "row_sha": [r["row_sha"] for r in rows],
        "row_prefix_id": [r["prefix_id"] for r in rows],
        "row_query_sha": [r["query_sha"] for r in rows],
        "spans": {
            span: {li: t.to(torch.float16) for li, t in per.items()} for span, per in spans.items()
        },
        "positions": {
            pos: {li: t.to(torch.float16) for li, t in per.items()}
            for pos, per in (positions or {}).items()
        },
        "metadata": _meta(),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(store, tmp)
    os.replace(tmp, path)


def _tf_capture_rows(
    cfg: Cfg, model_path: str, rows: list[dict], persona_names: list[str]
) -> tuple[dict, dict]:
    """TF span-means + last-token positions for rows under one model (one seam)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    device = CAP._device()
    spans = _teacher_forced_span_means(
        model_path,
        rows,
        persona_names,
        list(LAYERS_1979),
        device=device,
        dtype=CAP._dtype(),
        tf_batch_size=cfg.tf_batch,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    try:
        positions = _capture_positions(model, rows, list(LAYERS_1979), device, pad_id)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return spans, positions


# ── f1a: generation + capture ─────────────────────────────────────────────────


def run_f1a(cfg: Cfg, manifests: dict, state: str) -> list[str]:
    """One model state: batched vLLM greedy over the full (prefix x query) grid,
    then TF span-mean + last-token capture on the generated rows."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from explore_persona_space.analysis.representation_shift import (
        _build_generation_prompts,
        _reap_vllm_engine,
        _vllm_enforce_eager,
        compute_prompt_spans,
    )

    members = manifests["members"]
    queries = [q["prompt"] for q in manifests["queries"]]
    query_shas = [q["sha"] for q in manifests["queries"]]
    is_marker = state == "base_mk" or state in {r["arm_id"] for r in manifests["marker_arms"]}
    max_new = X.MAX_NEW_MARKER if is_marker else X.MAX_NEW_CONTENT
    max_model_len = X.PFX_MAX_MODEL_LEN_RAISED if is_marker else X.MAX_MODEL_LEN

    model_path, cleanup = _resolve_state_model(cfg, manifests, state)
    outputs: list[str] = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        eos_id = tokenizer.eos_token_id
        # render EVERY (prefix, query) prompt through the ONE construction path
        rendered_all: list[str] = []
        row_meta: list[dict] = []
        for member in members:
            pid = member["prefix_id"]
            kw = _member_render_kwargs(member)
            rendered, keys = _build_generation_prompts(
                tokenizer,
                {pid: kw["system"]},
                queries,
                user_wraps={pid: kw["user_wrap"]},
                prior_turns={pid: kw["prior_turns"]},
            )
            for text, (p_name, q_idx) in zip(rendered, keys, strict=True):
                n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                assert n_tok <= max_model_len - max_new, (
                    f"prompt budget violated: {pid} q{q_idx} {n_tok} tok "
                    f"> {max_model_len - max_new} (F0 budget assert should have caught this)"
                )
                rendered_all.append(text)
                row_meta.append(
                    {
                        "prefix_id": p_name,
                        "question_idx": q_idx,
                        "query_sha": query_shas[q_idx],
                        "row_sha": _row_sha(p_name, query_shas[q_idx]),
                        "prompt_text": text,
                        "member": member,
                    }
                )
        logger.info(
            "[f1a] %s: %d rendered prompts (%d prefixes x %d queries)",
            state,
            len(rendered_all),
            len(members),
            len(queries),
        )
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=X.GEN_GPU_MEM_UTIL,
            enforce_eager=_vllm_enforce_eager(),
            max_model_len=max_model_len,
            enable_prefix_caching=os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "1") == "0",
        )
        params = SamplingParams(temperature=0.0, max_tokens=max_new)
        rows: list[dict] = []
        try:
            chunk_n = CAP.GEN_CHUNK
            n_chunks = -(-len(rendered_all) // chunk_n)
            for ci, s in enumerate(range(0, len(rendered_all), chunk_n)):
                chunk = rendered_all[s : s + chunk_n]
                t0 = time.time()
                outs = llm.generate(chunk, params, use_tqdm=False)  # #613: no tqdm
                for j, out in enumerate(outs):
                    meta = row_meta[s + j]
                    comp = out.outputs[0]
                    resp_ids = list(comp.token_ids)
                    if resp_ids and resp_ids[-1] == eos_id:
                        resp_ids = resp_ids[:-1]
                    rows.append(
                        {
                            **{
                                k: meta[k]
                                for k in (
                                    "prefix_id",
                                    "question_idx",
                                    "query_sha",
                                    "row_sha",
                                    "prompt_text",
                                )
                            },
                            "persona": meta["prefix_id"],
                            "prompt_token_ids": list(out.prompt_token_ids),
                            "response_token_ids": resp_ids,
                            "finish_reason": comp.finish_reason,
                            "response_text": tokenizer.decode(resp_ids),
                        }
                    )
                logger.info(
                    "[vllm-chunk] %s chunk %d/%d (%d prompts) elapsed=%.1fs",
                    state,
                    ci + 1,
                    n_chunks,
                    len(chunk),
                    time.time() - t0,
                )
        finally:
            _reap_vllm_engine(llm)
            del llm
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                time.sleep(1.0)
        # spans (SAME construction path re-derived per member; #1112)
        kept: list[dict] = []
        dropped = 0
        for r in rows:
            if not r["response_token_ids"]:
                dropped += 1
                continue
            kw = _member_render_kwargs(
                r.pop("member", None)
                or next(m for m in members if m["prefix_id"] == r["prefix_id"])
            )
            r["prefix_len"], r["context_len"] = compute_prompt_spans(
                tokenizer,
                kw["system"],
                queries[r["question_idx"]],
                r["prompt_token_ids"],
                prior_messages=list(kw["prior_turns"]) or None,
                user_wrap=kw["user_wrap"],
                prefix_end="last_user",
                on_seam="snap",
            )
            kept.append(r)
        assert kept, f"{state}: every row dropped (empty responses?)"
        logger.info("[f1a] %s: %d kept / %d dropped rows", state, len(kept), dropped)
        # persist rows FIRST (raw text is never discardable — upload-policy)
        gen_dir = cfg.out_root / "gen" / state
        full_shards = _write_shards(gen_dir, "rows", kept)
        raw_rows = [
            {
                "row_sha": r["row_sha"],
                "prefix_id": r["prefix_id"],
                "query_sha": r["query_sha"],
                "finish_reason": r["finish_reason"],
                "response_text": r["response_text"],
            }
            for r in kept
        ]
        raw_shards = _write_shards(gen_dir / "raw", f"{state}_generation", raw_rows)
        _upload_paths(cfg, raw_shards, f"{HF_PREFIX_1979}/raw_completions/generation/{state}")
        _upload_paths(cfg, full_shards, f"{HF_PREFIX_1979}/gen_rows/{state}")
        # capture rides HF TF passes on the generated rows (spans + last-token)
        persona_names = [m["prefix_id"] for m in members]
        spans, positions = _tf_capture_rows(cfg, model_path, kept, persona_names)
        store = cfg.out_root / "stores" / "onpolicy" / state / "store.pt"
        _save_store(store, state, "onpolicy", kept, spans, positions)
        _upload_paths(cfg, [store], f"{HF_PREFIX_1979}/stores/onpolicy/{state}")
        outputs = [str(store)] + [str(p) for p in full_shards]
    finally:
        CAP._cleanup_merged(cleanup)  # merge -> generate -> capture -> delete rotation
    return outputs


# ── f1b: matched-text TF trees + marker slot DV ───────────────────────────────


def _slot_contexts(rows: list[dict]) -> list[str]:
    """Prompt + own response, stripped at the FIRST marker emission (slot recipe)."""
    contexts = []
    for r in rows:
        resp = r["response_text"]
        cut = resp.find("※")
        if cut != -1:
            resp = resp[:cut].rstrip()
        contexts.append(r["prompt_text"] + resp)
    return contexts


def _gate_mix_contexts(tok, mix_rows: list[dict]) -> list[str]:
    """Gate measurement contexts: cut at the first marker in the RESPONSE region only —
    ICL demo prompts legitimately carry the glyph (r6)."""
    contexts = []
    for r in mix_rows:
        prompt_text = tok.decode(r["prompt_token_ids"])
        resp_text = tok.decode(r["response_token_ids"])
        cut = resp_text.find("※")
        resp_ctx = resp_text[:cut].rstrip() if cut != -1 else resp_text
        contexts.append(prompt_text + resp_ctx)
    return contexts


def _assert_marker_tokenization(tokenizer) -> None:
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [83399], f"marker tokenization drifted: {ids} != [83399]"


def _arm_shim(arm_id: str):
    """Minimal .arm_id carrier for CAP._mix_positive_rows (registry-keyed)."""
    import types

    return types.SimpleNamespace(arm_id=arm_id)


def _marker_gauge_assert(row: dict) -> None:
    """Stage the arm row's adapter_config.json and run the gauge-free assert."""
    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config
    from explore_persona_space.orchestrate import hub

    if row["method"] == "ft":
        return  # full FT: no adapter config; gauge handled by the slot contract
    local = Path("/tmp") / f"i1979_adapter_cfg_{row['arm_id']}.json"
    hub.stage_hub_file(
        row["adapter_repo"],
        f"{row['adapter_subfolder']}/adapter_config.json",
        local,
        repo_type="model",
        overwrite=True,
    )
    assert_gauge_free_adapter_config(json.loads(local.read_text()), context=row["arm_id"])


def _slot_stats_for(cfg: Cfg, model_path: str, contexts: list[str]) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import (
        MARKER_SLOT_CONTRACT_KEYS,
        compute_marker_slot_stats,
    )

    device = CAP._device()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    _assert_marker_tokenization(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    try:
        recs = compute_marker_slot_stats(
            model, tokenizer, contexts, MARKER_TEXT, device=device, include_argmax=True
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # #530 four-float slot contract (eval/marker_logprob.MARKER_SLOT_CONTRACT_KEYS
    # = ("logp", "z_marker", "z_eos", "logZ")): pin the consumer-side key set at
    # the ONE chokepoint every f1b consumer reads through, so a guessed key
    # (crash-fix r4, job 16731: "logp_marker") fails HERE, not mid-gate.
    if recs:
        missing = set(MARKER_SLOT_CONTRACT_KEYS) - set(recs[0])
        assert not missing, (
            f"marker slot record missing #530 contract keys {sorted(missing)}: {sorted(recs[0])}"
        )
    return recs


def _gate_medians(
    trained_mix: list[dict],
    base_mix: list[dict],
    trained_cor: list[dict],
    base_cor: list[dict],
) -> tuple[float, float]:
    """Median (Δ logP over mix rows, |Δ z_marker| over corpus rows) for the #1900
    identity gate, keyed to the #530 four-float slot contract
    (``eval/marker_logprob.MARKER_SLOT_CONTRACT_KEYS`` = ("logp", "z_marker",
    "z_eos", "logZ") — the log-prob key is ``logp`` = z_marker − logZ, NOT
    ``logp_marker``; crash-fix r4, fellows job 16731)."""
    import numpy as np

    d_logp = np.array([t["logp"] - b["logp"] for t, b in zip(trained_mix, base_mix, strict=True)])
    d_z = np.array(
        [abs(t["z_marker"] - b["z_marker"]) for t, b in zip(trained_cor, base_cor, strict=True)]
    )
    return float(np.median(d_logp)), float(np.median(d_z))


def _gate_threshold_for(row: dict) -> tuple[float, str]:
    """Per-arm-class #1900 gate dlogP floor: returns (threshold_nats, arm_class).

    Full-FT marker arms (arms.json rows with method == "ft" / an ``ft_repo``) use
    a 1.8-nat floor: crash-fix r4 measured a PIN-VERIFIED FT checkpoint at 1.99
    nats — 0.5% under the 2.0 LoRA bar at the <=50-row mix sample (a calibration
    flake; unapplied/wrong artifacts read ~0 on BOTH gate conditions, and every
    LoRA marker arm passes 2.0 with wide margin). The plan §11 pinned the #1900
    gate verbatim, so the FT tolerance is a STATED plan deviation recorded in the
    gate output JSON as ``plan_deviation: ft-marker-gate-tolerance-1.8``; the
    mandatory secondary condition (median |dz_marker| > 0 on corpus rows,
    ``_assert_gate_pass``) keeps a genuinely-unapplied artifact failing loud.
    """
    is_ft = row.get("method") == "ft" or "ft_repo" in row
    if is_ft:
        return GATE_MIN_DELTA_LOGP_NATS_FT, "ft"
    return GATE_MIN_DELTA_LOGP_NATS, "lora"


def _assert_gate_pass(
    arm_id: str, arm_class: str, threshold: float, med_logp: float, med_z: float
) -> None:
    """Fail-loud #1900 gate verdict: med_logp >= per-class floor AND med_z > 0.

    Both conditions are mandatory for BOTH arm classes — an unapplied or wrong
    artifact reads ~0 on both, so the r5 FT tolerance (1.8 vs 2.0 nats) cannot
    let one through (crash-fix r5, `f1b:gate:mk-pers-ft-con-s42` at 1.99 nats).
    """
    assert med_logp >= threshold, (
        f"[marker-gate] {arm_id} ({arm_class}): median training-row dlogP {med_logp:.2f} < "
        f"{threshold} nats — adapter not applied / wrong artifact (#1900 gate)"
    )
    assert med_z > 0.0, (
        f"[marker-gate] {arm_id} ({arm_class}): median |dz_marker| == 0 on corpus rows"
    )


def run_f1b_gate(cfg: Cfg, manifests: dict, arm_id: str) -> list[str]:
    """#1900 frame-free marker identity gate (plan §4 F1b), BEFORE the fleet pass:
    median training-row delta logP >= the per-arm-class floor (2.0 nats LoRA /
    1.8 nats full-FT — ``_gate_threshold_for``, crash-fix r5 stated plan
    deviation) AND median |delta z_marker| > 0 on 50 corpus rows (mandatory for
    both classes). Realized values + threshold + arm class land in the gate's
    output JSON before the asserts run."""
    row = manifests["arm_rows"][arm_id]
    _marker_gauge_assert(row)
    ensure_arm_registry(cfg, manifests)
    mix_rows, _mix_meta = CAP._mix_positive_rows(cfg.cap_cfg(), _arm_shim(arm_id))
    mix_rows = mix_rows[:50]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(X.BASE_MODEL)
    mix_ctx = _gate_mix_contexts(tok, mix_rows)
    base_rows = _load_unit_rows(cfg, "base_mk")[:GATE_N_CORPUS_ROWS]
    corpus_ctx = _slot_contexts(base_rows)

    model_path, cleanup = _resolve_state_model(cfg, manifests, arm_id)
    try:
        trained_mix = _slot_stats_for(cfg, model_path, mix_ctx)
        trained_cor = _slot_stats_for(cfg, model_path, corpus_ctx)
    finally:
        CAP._cleanup_merged(cleanup)
    base_mix = _slot_stats_for(cfg, X.BASE_MODEL, mix_ctx)
    base_cor = _slot_stats_for(cfg, X.BASE_MODEL, corpus_ctx)

    med_logp, med_z = _gate_medians(trained_mix, base_mix, trained_cor, base_cor)
    threshold, arm_class = _gate_threshold_for(row)
    out = cfg.out_root / "marker_tf" / arm_id / "identity_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "arm": arm_id,
        "arm_class": arm_class,
        "median_delta_logp_mix": med_logp,
        "median_abs_delta_z_corpus": med_z,
        "n_mix_rows": len(mix_ctx),
        "n_corpus_rows": len(corpus_ctx),
        "thresholds": {"min_delta_logp_nats": threshold, "min_abs_delta_z": 0.0},
        **_meta(),
    }
    if arm_class == "ft":
        record["plan_deviation"] = FT_GATE_PLAN_DEVIATION
    CAP._atomic_json(out, record)
    _upload_paths(cfg, [out], f"{HF_PREFIX_1979}/marker_tf/{arm_id}")
    _assert_gate_pass(arm_id, arm_class, threshold, med_logp, med_z)
    return [str(out)]


def run_f1b_writes(cfg: Cfg, manifests: dict, arm_id: str, base_state: str) -> list[str]:
    """Arm model teacher-forced on the base state's generated rows (matched text)."""
    assert manifests["arm_rows"][arm_id]["base_unit"] == base_state, (
        arm_id,
        base_state,
        manifests["arm_rows"][arm_id]["base_unit"],
    )
    rows = _load_unit_rows(cfg, base_state)
    persona_names = [m["prefix_id"] for m in manifests["members"]]
    model_path, cleanup = _resolve_state_model(cfg, manifests, arm_id)
    try:
        spans, positions = _tf_capture_rows(cfg, model_path, rows, persona_names)
    finally:
        CAP._cleanup_merged(cleanup)
    store = cfg.out_root / "stores" / "matched_tf" / arm_id / "store.pt"
    _save_store(store, arm_id, f"matched_tf(on {base_state})", rows, spans, positions)
    _upload_paths(cfg, [store], f"{HF_PREFIX_1979}/stores/matched_tf/{arm_id}")
    return [str(store)]


def run_f1b_slot(cfg: Cfg, manifests: dict, arm_id: str, side: str) -> list[str]:
    """Marker slot DV on the arm's OWN generated text: side='own' (arm model) or
    'base' (base model on the same text) — trained/base halves of the 3-space DV."""
    from explore_persona_space.eval.marker_logprob import validate_marker_slot_record

    rows = _load_unit_rows(cfg, arm_id)
    contexts = _slot_contexts(rows)
    if side == "own":
        model_path, cleanup = _resolve_state_model(cfg, manifests, arm_id)
    else:
        model_path, cleanup = X.BASE_MODEL, None
    try:
        recs = _slot_stats_for(cfg, model_path, contexts)
    finally:
        CAP._cleanup_merged(cleanup)
    for rec in recs:
        validate_marker_slot_record(rec)
    payload = [
        {"row_sha": r["row_sha"], "prefix_id": r["prefix_id"], "query_sha": r["query_sha"], **rec}
        for r, rec in zip(rows, recs, strict=True)
    ]
    out_dir = cfg.out_root / "marker_tf" / arm_id
    shards = _write_shards(out_dir, f"slot_{side}", payload)
    _upload_paths(cfg, shards, f"{HF_PREFIX_1979}/marker_tf/{arm_id}")
    return [str(p) for p in shards]


# ── f1c: anchor captures ──────────────────────────────────────────────────────


def _anchor_known_personas(cfg: Cfg, rows: list[dict], mix_id: str) -> list[str]:
    """Known-persona set for the f1c anchor TF capture: the labels the LOADED
    mix rows actually carry, validated against the arm registry.

    A shared training-mix row pool serves several arms (the FT->LoRA
    ``mix_pos_sources`` mapping), and ``CAP._mix_positive_rows`` labels rows
    with the REPRESENTATIVE arm's slug rather than the mix id — so passing
    ``[mix_id]`` alone fails the reused helper's persona integrity assert
    (fellows job 16717: row persona ``syc-pers-ft-con-s42`` under mix
    ``syc-pers-con-lr1e5-s42``). The helper's assert is a legitimate
    integrity check and stays untouched; the fix is glue-side. Returns the
    sorted union of realized labels + mix_id; a label that is neither the
    mix id nor a registered arm id still fails loud (genuinely foreign rows).
    """
    reg = json.loads((cfg.out_root / "arm_registry.json").read_text())
    registered = set(reg["mix_pos_sources"])
    labels = {r["persona"] for r in rows}
    foreign = labels - registered - {mix_id}
    assert not foreign, (mix_id, sorted(foreign), "persona labels not in arm_registry.json")
    return sorted(labels | {mix_id})


def run_f1c(cfg: Cfg, manifests: dict, mix_id: str) -> list[str]:
    """Training-centroid anchors for one mix (base model TF; 20 rows @ seed 1979)."""
    import numpy as np
    import torch

    ensure_arm_registry(cfg, manifests)
    rep = _mixes(manifests)[mix_id]
    rows, mix_meta = CAP._mix_positive_rows(cfg.cap_cfg(), _arm_shim(rep["arm_id"]))
    if len(rows) > N_ANCHOR_ROWS:  # marker pools (~200) subsample @ seed 1979
        idx = np.random.default_rng(SEED).choice(len(rows), N_ANCHOR_ROWS, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    for i, r in enumerate(rows):
        r.setdefault("persona", mix_id)
        r.setdefault("row_sha", _sha16(f"{mix_id}||anchor||{i}"))
    known = _anchor_known_personas(cfg, rows, mix_id)
    spans, positions = _tf_capture_rows(cfg, X.BASE_MODEL, rows, known)
    anchors: dict = {
        "mix": mix_id,
        "arm": rep["arm_id"],
        "n_rows": len(rows),
        "mix_meta": mix_meta,
        "metadata": _meta(),
    }
    for li in LAYERS_1979:
        ctx, ans = spans["context"][li], spans["response"][li]
        even, odd = torch.arange(0, len(rows), 2), torch.arange(1, len(rows), 2)
        anchors[f"L{li}"] = {
            "A_ctx_span": ctx.mean(0),
            "A_ans": ans.mean(0),
            "A_ctx_last_prompt": positions["last_prompt"][li].mean(0),
            "A_ctx_last_ctx": positions["last_ctx"][li].mean(0),
            "rows_ctx": ctx.to(torch.float16),
            "rows_ans": ans.to(torch.float16),
            "half_even_ctx": ctx[even].mean(0),
            "half_odd_ctx": ctx[odd].mean(0),
            "half_even_ans": ans[even].mean(0),
            "half_odd_ans": ans[odd].mean(0),
        }
    out = cfg.out_root / "anchors" / mix_id / "anchors.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(anchors, tmp)
    os.replace(tmp, out)
    _upload_paths(cfg, [out], f"{HF_PREFIX_1979}/anchors/{mix_id}")
    return [str(out)]


# ── f1d: M0 re-materialization + union refit ─────────────────────────────────


def run_f1d_stage(cfg: Cfg) -> list[str]:
    """Stage the pinned #1768 inputs into the reused loaders' expected layout."""
    from explore_persona_space.orchestrate import hub

    stages = [
        (
            f"{X.HF_PREFIX}/inputs/corpus_sample.json",
            cfg.out_root / "inputs/corpus_sample.json",
            CORPUS_PIN,
        ),
        (
            f"{X.HF_PREFIX}/corpus_capture/base_content/pooled.pt",
            cfg.out_root / "corpus_capture/base_content/pooled.pt",
            CORPUS_PIN,
        ),
        (
            f"{X.HF_PREFIX}/lasttoken_ctx/base_content/lasttoken.pt",
            cfg.out_root / "lasttoken/base_content/lasttoken.pt",
            None,
        ),  # lt re-pool @ main
        (f"issue1900_leakrace/maps/m0_L19.pt", cfg.out_root / "maps_ref/m0_L19.pt", I1900_PIN),
    ]
    out = []
    for path_in_repo, target, rev in stages:
        target.parent.mkdir(parents=True, exist_ok=True)
        hub.stage_hub_file(X.HF_DATA_REPO, path_in_repo, target, repo_type="dataset", revision=rev)
        out.append(str(target))
    return out


def _m0_reference(cfg: Cfg) -> dict | None:
    """Recorded lt-round M0 R² reference (config/m0_reference.json), for the
    |R²_re − R²_lt| <= 0.01 parity assert. Fail-loud when absent unless the
    explicit --skip-m0-ref-parity deviation flag is set."""
    p = cfg.config_dir / "m0_reference.json"
    if p.exists():
        return json.loads(p.read_text())
    if cfg.skip_m0_ref_parity:
        logger.warning("[f1d] m0_reference.json absent — R² parity SKIPPED (explicit flag)")
        return None
    raise RuntimeError(
        f"{p} missing: the plan §4 F1d parity assert needs the lt-round recorded M0 R² "
        "(schema {'span_mean': {'14': r2, ...}, 'last_prompt': {...}}). Publish it under "
        f"{HF_PREFIX_1979}/config/ or pass --skip-m0-ref-parity as an explicit deviation."
    )


def _apply_saved_map(payload: dict, Xmat, dev):
    """Apply a persisted #1900/#1768 map payload to X.

    Schema PROBED on the real pinned artifacts 2026-08-01 (concern
    f1e-saved-map-payload-schema): `wmap_<arm>_L19.pt` and `m0_L19.pt` at
    3bb20deb are WRAPPER dicts ``{"name": ..., "payload": {kind: "ridge", W,
    xmu, xsd, ymu, selected_lambda}}`` — unwrap first, then dispatch on the
    n1m ``kind`` (which applies the standardize-X / W / +ymu predict path).
    Order matters: the raw n1m ridge payload ALSO carries a top-level "W",
    so a "W"-first branch would silently skip standardization (wrong
    predictions, not a crash) — the ``kind`` check must come before the
    plain-affine {W, b} fallback.
    """
    import numpy as np

    import issue1768_fit as FIT

    if isinstance(payload, dict) and isinstance(payload.get("payload"), dict):
        payload = payload["payload"]  # the persisted {name, payload} wrapper
    if isinstance(payload, dict) and "kind" in payload:
        return FIT._apply_payload(payload, Xmat, dev)
    if isinstance(payload, dict) and "W" in payload and "xmu" not in payload:
        W = np.asarray(payload["W"], dtype=np.float64)
        b = np.asarray(payload.get("b", np.zeros(W.shape[1])), dtype=np.float64)
        return Xmat @ W + b
    keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
    raise RuntimeError(f"unrecognized persisted-map payload schema: {keys}")


def _base_lasttoken_cell(cfg: Cfg, layer: int, position: str) -> dict:
    """Base-only M0 cell at a last-token position — the lt-round loaders verbatim."""
    import numpy as np

    import issue1768_lasttoken_fit as LTF

    cache = cfg.out_root / "ltf_cache"
    C_by, c_sha = LTF.load_lasttoken(cfg.out_root, "base_content", [layer], position)
    V_by, v_sha = LTF.fetch_response(cache, "corpus_capture", "base_content", [layer], persist=True)
    ix = {s: i for i, s in enumerate(c_sha)}
    keep = [(i, s) for i, s in enumerate(v_sha) if s in ix]
    assert len(keep) >= 0.9 * len(v_sha), (layer, position, len(keep), len(v_sha))
    b = np.asarray([i for i, _ in keep])
    shas = [s for _, s in keep]
    sel = np.asarray([ix[s] for s in shas])
    sample = X.load_corpus_sample(cfg.out_root)
    sha_to_q = {r["sha"]: q for q, r in enumerate(sample["rows"])}
    qidx = np.asarray([sha_to_q[s] for s in shas])
    n_train, n_val = sample["n_train"], sample["n_val"]
    split = np.where(qidx < n_train, "train", np.where(qidx < n_train + n_val, "val", "test"))
    return {"C0": C_by[layer][sel], "V0": V_by[layer][b], "sha": shas, "split": split}


def _onpolicy_base_arrays(cfg: Cfg, layer: int, position: str):
    """Prefixed base_content rows from the f1a store: (C, V) at one position."""
    import numpy as np
    import torch

    store = torch.load(
        cfg.out_root / "stores/onpolicy/base_content/store.pt",
        map_location="cpu",
        weights_only=False,
    )
    if position == "span_mean":
        C = store["spans"]["context"][layer].float().numpy().astype(np.float64)
    else:
        C = store["positions"][position][layer].float().numpy().astype(np.float64)
    V = store["spans"]["response"][layer].float().numpy().astype(np.float64)
    return C, V


def run_f1d_fit(cfg: Cfg, kind: str, position: str, layer: int) -> list[str]:
    """One F1d ridge fit (kind='m0' re-materialization or 'union' refit)."""
    import numpy as np
    import torch

    import issue1768_fit as FIT

    assert torch.cuda.is_available(), "F1 fits are GPU-resident (plan §10 reused-code table)"
    dev = FIT._device()
    ref = _m0_reference(cfg)
    result: dict = {"kind": kind, "position": position, "layer": layer, **_meta()}

    def _span_cell(out_root: Path):
        # the pilot_m0_fit derivation verbatim (same loaders/split/λ-grid/code)
        base = FIT._load_store(out_root / "corpus_capture/base_content/pooled.pt")
        Cb, _ = FIT._rows_from_store(base, "context", layer)
        Vb, _ = FIT._rows_from_store(base, "response", layer)
        sample = X.load_corpus_sample(out_root)
        qidx = np.asarray(base["row_question_idx"])
        n_train, n_val = sample["n_train"], sample["n_val"]
        split = np.where(qidx < n_train, "train", np.where(qidx < n_train + n_val, "val", "test"))
        return Cb, Vb, split

    if position == "span_mean":
        Cb, Vb, split = _span_cell(cfg.out_root)
    else:
        cell = _base_lasttoken_cell(cfg, layer, position)
        Cb, Vb, split = cell["C0"], cell["V0"], cell["split"]
    if kind == "m0":
        C0, V0 = Cb, Vb
    else:  # union refit: 15k bare + 3k prefixed base_content rows (prefixed -> train)
        Cp, Vp = _onpolicy_base_arrays(cfg, layer, position)
        C0 = np.concatenate([Cb, Cp], axis=0)
        V0 = np.concatenate([Vb, Vp], axis=0)
        split = np.concatenate([split, np.array(["train"] * len(Cp))])
        result["n_prefixed_rows"] = int(len(Cp))
    tr, val, te = FIT._split_idx(split)
    pred_te, meta, payload = FIT._fit_map(C0, V0, tr, val, te, dev)
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    ib = identity_bias_predict(C0[tr], V0[tr], C0[te])
    r2 = float(FIT._pooled_r2(pred_te, V0[te]))
    result["fit"] = {
        "heldout_r2": r2,
        "mean_cos": float(FIT._mean_cos(pred_te, V0[te])),
        "reads": FIT._map_reads(pred_te, V0[te]),
        "selected_lambda": meta["selected_lambda"],
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "d": int(C0.shape[1]),
        "identity_bias": {
            "heldout_r2": float(FIT._pooled_r2(ib, V0[te])),
            "mean_cos": float(FIT._mean_cos(ib, V0[te])),
        },
        "knn": {
            "fitted": knn_retrieval(pred_te, V0[te], ks=(1, 10), metric="euclidean"),
            "identity_bias": knn_retrieval(ib, V0[te], ks=(1, 10), metric="euclidean"),
        },
    }
    # persist the payload so F3 can apply the re-materialized map
    pay_path = cfg.out_root / "maps" / f"{kind}_{position}_L{layer}.pt"
    pay_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"payload": payload, **result}, pay_path)
    _upload_paths(cfg, [pay_path], f"{HF_PREFIX_1979}/maps")
    # cross-issue cos-parity probe vs #1900 m0_L19.pt (span-mean L19 cell only)
    if kind == "m0" and position == "span_mean" and layer == 19:
        ref_payload = torch.load(
            cfg.out_root / "maps_ref/m0_L19.pt", map_location="cpu", weights_only=False
        )
        n = min(M0_PARITY_PROBE_ROWS, len(te))
        probe = C0[te][:n]
        mine = FIT._apply_payload(payload, probe, dev)
        theirs = _apply_saved_map(ref_payload, probe, dev)
        cos = float(FIT._mean_cos(np.asarray(mine), np.asarray(theirs)))
        result["m0_L19_pred_cos_vs_1900"] = cos
        assert cos >= M0_PARITY_COS_MIN, (
            f"[f1d-parity] cos(pred, #1900 m0_L19 pred) {cos:.4f} < {M0_PARITY_COS_MIN}"
        )
    if kind == "m0" and ref is not None:
        want = ref.get(position, {}).get(str(layer))
        assert want is not None, f"m0_reference.json lacks {position}/{layer}"
        assert r2 is not None, f"fit result carries no held-out R² read: {sorted(result)}"
        assert abs(float(r2) - float(want)) <= M0_PARITY_R2_TOL, (
            f"[f1d-parity] M0 {position} L{layer}: R²_re {r2:.4f} vs recorded {want:.4f} "
            f"(> {M0_PARITY_R2_TOL} — re-materialization is NOT the pinned fit)"
        )
    out = cfg.out_root / "maps" / f"{kind}_{position}_L{layer}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    CAP._atomic_json(out, result)
    _upload_paths(cfg, [out], f"{HF_PREFIX_1979}/maps")
    return [str(out)]


# ── f1e: predictor/battery ingredient tables ─────────────────────────────────


def _prefix_means(
    store: dict, layer: int, span_or_pos: str, prefix_ids: list[str], row_mask=None, dtype=None
):
    """Per-prefix mean vectors from a 1979 store — one batched index_add pass.

    ``row_mask`` (optional bool sequence over store rows) restricts the mean to
    a row subset — used for the even/odd QUERY-half means (plan §6 A5 disjoint
    legs). The prefix-id index is built over the masked-IN rows ONLY: a
    masked-OUT row's prefix may legitimately lie outside ``prefix_ids`` (the
    f1g smoke slice masks the HF-staged FULL-panel parent store down to the
    sliced grid — crash-fix r7, KeyError 'wildchat_prefix_real545'); a
    masked-IN out-of-panel prefix still fails loud. ``dtype`` (default fp32 —
    the parent F1e convention, unchanged) sets the accumulation dtype; the f1g
    amendment passes ``torch.float64`` (plan v6 §8 fp-noise mitigation). Fails
    loud when any prefix has zero rows in the (masked) selection.
    """
    import torch

    acc = dtype or torch.float32
    if span_or_pos in ("last_prompt", "last_ctx", "last_prefix"):
        T = store["positions"][span_or_pos][layer].to(acc)
    else:
        T = store["spans"][span_or_pos][layer].to(acc)
    pid_ix = {p: i for i, p in enumerate(prefix_ids)}
    row_pids = store["row_prefix_id"]
    if row_mask is not None:
        keep = torch.tensor(list(row_mask), dtype=torch.bool)
        assert keep.shape[0] == T.shape[0] == len(row_pids), (keep.shape, T.shape, len(row_pids))
        T = T[keep]
        row_pids = [p for p, m in zip(row_pids, row_mask) if m]
    idx = torch.tensor([pid_ix[p] for p in row_pids], dtype=torch.long)
    sums = torch.zeros(len(prefix_ids), T.shape[1], dtype=T.dtype)
    sums.index_add_(0, idx, T)
    counts = torch.zeros(len(prefix_ids), dtype=T.dtype).index_add_(
        0, idx, torch.ones(len(idx), dtype=T.dtype)
    )
    assert (counts > 0).all(), f"empty prefix cell(s) in {store['unit']}: {counts.tolist()}"
    return sums / counts.unsqueeze(1)


def _query_parity_mask(store: dict, queries: list[dict], parity: int) -> list[bool]:
    """Row mask selecting rows whose query index (in the pinned draw order) has
    the given parity — the ONE deterministic even/odd partition (rule 21)."""
    q_ix = {q["sha"]: i for i, q in enumerate(queries)}
    return [q_ix[s] % 2 == parity for s in store["row_query_sha"]]


def _whiten_solve(chol, vecs):
    """Sigma^{-1} @ vecs via the persisted lower-Cholesky factor (cheap per rhs)."""
    from scipy.linalg import solve_triangular

    y = solve_triangular(chol, vecs, lower=True)
    return solve_triangular(chol.T, y, lower=False)


_ANCHOR_KEY_BY_POS = {
    "span_mean_context": "A_ctx_span",
    "last_prompt": "A_ctx_last_prompt",
    "last_ctx": "A_ctx_last_ctx",
}


def run_f1e(cfg: Cfg, manifests: dict) -> list[str]:
    """Batched predictor/battery ingredients per (arm, layer, position):
    per-prefix writes W (matched-text primary + on-policy secondary), base
    c̄(P)/v̄⁰(P), batched SVD of W, whitened gate reads, two-GEMM norm-matched
    null projections (corpus-covariance + isotropic), P8 wmap predictions."""
    import numpy as np
    import torch

    import issue1768_directions as DIR

    from explore_persona_space.orchestrate import hub

    prefix_ids = [m["prefix_id"] for m in manifests["members"]]
    arm_rows = manifests["content_arms"] + manifests["marker_arms"]
    assert abs(DIR.SHRINKAGE - SIGMA_SHRINKAGE) < 1e-12, (
        f"issue1768_directions.SHRINKAGE={DIR.SHRINKAGE} != plan-pinned {SIGMA_SHRINKAGE}"
    )
    # Σ per layer (plan §3 A7 lattice is per (arm × LAYER) — gate reads at all
    # three pre-registered layers; Σ recipe stays span-context / 15k bare rows).
    sigma_by_layer = {li: DIR.corpus_sigma(cfg.out_root, li) for li in LAYERS_1979}
    rng = np.random.default_rng(SEED)
    out_dir = cfg.out_root / "predictor_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    battery_dir = cfg.out_root / "battery"
    battery_dir.mkdir(parents=True, exist_ok=True)

    def _load(tree: str, unit: str) -> dict:
        return torch.load(
            cfg.out_root / "stores" / tree / unit / "store.pt",
            map_location="cpu",
            weights_only=False,
        )

    base_by_kind = {
        "content": _load("onpolicy", "base_content"),
        "marker": _load("onpolicy", "base_mk"),
    }
    tables: dict = {"prefix_ids": prefix_ids, "layers": list(LAYERS_1979), **_meta()}
    tensors: dict = {}
    queries = manifests["queries"]
    dev = FIT_dev()

    # kind-level base ingredients (arm-independent, computed ONCE per kind):
    # prefix vectors (F3 v_P->v_A mapping arm, plan §4 grain definition) +
    # even/odd query-half base response means (A5 disjoint legs, plan §6).
    for kind, store in base_by_kind.items():
        even = _query_parity_mask(store, queries, 0)
        odd = _query_parity_mask(store, queries, 1)
        for layer in LAYERS_1979:
            base_slot = f"base/{kind}/L{layer}"
            tensors[f"{base_slot}/Pbar_prefix_span"] = _prefix_means(
                store, layer, "prefix", prefix_ids
            ).to(torch.float16)
            tensors[f"{base_slot}/Pbar_last_prefix"] = _prefix_means(
                store, layer, "last_prefix", prefix_ids
            ).to(torch.float16)
            tensors[f"{base_slot}/Vbar0_even"] = _prefix_means(
                store, layer, "response", prefix_ids, even
            ).to(torch.float16)
            tensors[f"{base_slot}/Vbar0_odd"] = _prefix_means(
                store, layer, "response", prefix_ids, odd
            ).to(torch.float16)

    # persisted F1d map payloads (deps guarantee they exist): the realized fit
    # list from f1d_fit_specs(cfg) — full grid m0 at 3 layers x 2 positions +
    # union at L19 x 2 positions; under cfg.smoke_subset the single
    # (m0, span_mean, L19) fit (same enumeration build_work_items scheduled).
    map_payloads: dict[tuple[str, str, int], dict] = {}
    for mkind, mpos, layer in f1d_fit_specs(cfg):
        p = cfg.out_root / "maps" / f"{mkind}_{mpos}_L{layer}.pt"
        assert p.exists(), f"F1d map payload missing: {p} (f1e deps include f1d)"
        map_payloads[(mkind, mpos, layer)] = torch.load(p, map_location="cpu", weights_only=False)

    # kind-level through-map transforms of base c̄(P) (P3a/P3b/P6 inputs; the
    # union variants feed the union-vs-bare dump comparison).
    for kind, store in base_by_kind.items():
        for (mkind, mpos, layer), payload in map_payloads.items():
            key = "context" if mpos == "span_mean" else mpos
            Cb = _prefix_means(store, layer, key, prefix_ids)
            pred = _apply_saved_map(payload, Cb.double().numpy(), dev)
            tensors[f"{mkind}pred/{kind}/L{layer}/{mpos}"] = torch.tensor(
                np.asarray(pred), dtype=torch.float16
            )

    # anchors per mix (loaded once) + M0-transformed A_ctx (P3a's M0 A_ctx leg)
    anchors_by_mix: dict[str, dict] = {}
    for mix in sorted(_mixes(manifests)):
        anc = torch.load(
            cfg.out_root / "anchors" / mix / "anchors.pt", map_location="cpu", weights_only=False
        )
        anchors_by_mix[mix] = anc
        for (mkind, mpos, layer), payload in map_payloads.items():
            if mkind != "m0":
                continue
            akey = "A_ctx_span" if mpos == "span_mean" else "A_ctx_last_prompt"
            vec = anc[f"L{layer}"][akey].double().numpy()[None, :]
            pred = _apply_saved_map(payload, vec, dev)
            tensors[f"m0anchor/{mix}/L{layer}/{mpos}"] = torch.tensor(
                np.asarray(pred)[0], dtype=torch.float16
            )

    # persist Σ ingredients so F3 can draw its own A5 norm-matched nulls +
    # recompute P4 variants on the VM without the 15k corpus store.
    sigma_path = battery_dir / "sigma_chol.pt"
    torch.save(
        {
            "shrinkage": SIGMA_SHRINKAGE,
            **{
                f"L{li}": {
                    "chol": torch.tensor(np.asarray(s["chol"]), dtype=torch.float16),
                    "top_eig": torch.tensor(np.asarray(s["top_eig"]), dtype=torch.float32),
                    "n_rows": int(s["n_rows"]),
                }
                for li, s in sigma_by_layer.items()
            },
            **_meta(),
        },
        sigma_path,
    )

    marker_ids = {r["arm_id"] for r in manifests["marker_arms"]}
    for arm_row in arm_rows:
        arm_id = arm_row["arm_id"]
        kind = "marker" if arm_id in marker_ids else "content"
        base_store = base_by_kind[kind]
        matched = _load("matched_tf", arm_id)
        onpol = _load("onpolicy", arm_id)
        anchors = anchors_by_mix[arm_row["mix_arm_id"]]
        arm_tab: dict = {}
        for layer in LAYERS_1979:
            chol = np.asarray(sigma_by_layer[layer]["chol"])
            for pos in ("span_mean_context", "last_prompt", "last_ctx"):
                key = "context" if pos == "span_mean_context" else pos
                Cbar = _prefix_means(base_store, layer, key, prefix_ids)
                Vbar0 = _prefix_means(base_store, layer, "response", prefix_ids)
                # post-FT context means (M1/M3/M5 inputs — arm onpolicy store)
                Cbar_post = _prefix_means(onpol, layer, key, prefix_ids)
                # matched-text write (PRIMARY): arm TF on base rows − base own rows
                W_m = _prefix_means(matched, layer, "response", prefix_ids) - Vbar0
                # on-policy write (SECONDARY): arm own-gen − base own-gen response means
                W_o = _prefix_means(onpol, layer, "response", prefix_ids) - Vbar0
                U, S, Vh = torch.linalg.svd(W_m, full_matrices=False)  # 50 x 3584 thin SVD
                slot = f"{arm_id}/L{layer}/{pos}"
                tensors[f"{slot}/W_matched"] = W_m.to(torch.float16)
                tensors[f"{slot}/W_onpolicy"] = W_o.to(torch.float16)
                tensors[f"{slot}/Cbar"] = Cbar.to(torch.float16)
                tensors[f"{slot}/Vbar0"] = Vbar0.to(torch.float16)
                tensors[f"{slot}/Cbar_post"] = Cbar_post.to(torch.float16)
                # P4: whitened gate similarity g(P) at the position-matched
                # anchor (Σ span-derived per plan §5 — stated convention).
                c_src = anchors[f"L{layer}"][_ANCHOR_KEY_BY_POS[pos]].double().numpy()
                a_vec = _whiten_solve(chol, c_src)
                g_pred = (Cbar.double().numpy() @ a_vec) / (float(c_src @ a_vec) + 1e-12)
                arm_tab[f"L{layer}/{pos}"] = {
                    "svd_spectrum": S.tolist(),
                    "w_norms": W_m.norm(dim=1).tolist(),
                    "w_onpolicy_norms": W_o.norm(dim=1).tolist(),
                    "p4_gpred": [float(v) for v in g_pred],
                }
                if pos == "span_mean_context":
                    # A7 whitened gate read per (arm, LAYER) — the registered
                    # H4 lattice cells (plan §3): c_src = the arm's training-
                    # centroid anchor A_ctx_span, w = pooled matched write dir;
                    # per-prefix delta_v = W_m rows. Convention per concern
                    # f1e-union-split-and-gate-convention.
                    C0 = Cbar.double().numpy()
                    Wnp = W_m.double().numpy()
                    w_pool = Wnp.mean(axis=0)
                    g_hat = Wnp @ w_pool / (float(w_pool @ w_pool) + 1e-12)
                    arm_tab[f"L{layer}/{pos}"]["gate_read"] = {
                        "convention": "c_src=A_ctx_span(mix anchor); w=W_matched.mean(0)",
                        **DIR.gate_read(C0, Wnp, c_src, w_pool, sigma_by_layer[layer]),
                    }
                    arm_tab[f"L{layer}/{pos}"]["g_hat"] = [float(v) for v in g_hat]
                    if layer == UNION_LAYER:
                        # M5 inputs: M0-transformed post-FT context means (both
                        # fitted positions) at the primary layer. The membership
                        # guard only ever skips under cfg.smoke_subset (f1d
                        # restricted to span_mean L19); in production every
                        # payload was hard-asserted present at load above.
                        for mpos in ("span_mean", "last_prompt"):
                            if ("m0", mpos, layer) not in map_payloads:
                                continue
                            pk = "context" if mpos == "span_mean" else mpos
                            Cp = _prefix_means(onpol, layer, pk, prefix_ids)
                            predp = _apply_saved_map(
                                map_payloads[("m0", mpos, layer)], Cp.double().numpy(), dev
                            )
                            tensors[f"{arm_id}/m0pred_Cbar_post/L{layer}/{mpos}"] = torch.tensor(
                                np.asarray(predp), dtype=torch.float16
                            )
                        # norm-matched nulls: ONE GEMM per family over all 2,000
                        # draws (L19-span diagnostic projections, as registered)
                        d = W_m.shape[1]
                        iso = rng.standard_normal((N_NULL_DRAWS, d))
                        cov = iso @ np.asarray(sigma_by_layer[layer]["chol"]).T
                        for fam, draws in (("isotropic", iso), ("corpus_cov", cov)):
                            proj = draws @ C0.T  # (2000, 50): the two-GEMM null battery
                            arm_tab[f"L{layer}/{pos}"][f"null_{fam}_q"] = {
                                "q05": np.quantile(proj, 0.05, axis=0).tolist(),
                                "q95": np.quantile(proj, 0.95, axis=0).tolist(),
                            }
        tables[arm_id] = arm_tab
    # P8: apply #1900 persisted wmaps (span-mean, L19) via the linearity identity
    wmap_sel = manifests["wmap"]
    p8: dict = {}
    Cbar_c = _prefix_means(base_by_kind["content"], UNION_LAYER, "context", prefix_ids)
    for arm_row in manifests["content_arms"]:
        arm_id = arm_row["arm_id"]
        fname = None
        for cand in wmap_sel.get("selected_files", []) if isinstance(wmap_sel, dict) else []:
            if arm_id in cand:
                fname = cand
                break
        if fname is None:
            fname = f"issue1900_leakrace/maps/wmap_{arm_id}_L19.pt"
        local = cfg.out_root / "maps_ref" / Path(fname).name
        local.parent.mkdir(parents=True, exist_ok=True)
        hub.stage_hub_file(X.HF_DATA_REPO, fname, local, repo_type="dataset", revision=I1900_PIN)
        payload = torch.load(local, map_location="cpu", weights_only=False)
        pred = _apply_saved_map(payload, Cbar_c.double().numpy(), FIT_dev())
        tensors[f"{arm_id}/p8_wmap_pred_L19"] = torch.tensor(np.asarray(pred), dtype=torch.float16)
        p8[arm_id] = {
            "wmap_file": fname,
            "position_note": "span-mean c̄(P) via linearity identity; no refit (plan §4 F1e)",
        }
    tables["p8"] = p8
    tab_path = out_dir / "predictor_ingredients.json"
    CAP._atomic_json(tab_path, tables)
    # battery/*.json deliverable (plan §6.5): the assumption-battery reads —
    # per-(arm, layer) gate reads + SVD spectra + null quantiles — as JSON.
    battery_reads = {
        "meta": _meta(),
        "prefix_ids": prefix_ids,
        "gate_convention": "c_src=A_ctx_span(mix anchor); w=W_matched.mean(0) (plan §4 F1e)",
        "arms": {
            arm_id: {
                lk: {
                    k: v
                    for k, v in slot.items()
                    if k in ("gate_read", "g_hat", "svd_spectrum", "w_norms")
                    or k.startswith("null_")
                }
                for lk, slot in arm_tab.items()
                if lk.endswith("span_mean_context")
            }
            for arm_id, arm_tab in tables.items()
            if arm_id in {r["arm_id"] for r in arm_rows}
        },
    }
    battery_json = battery_dir / "battery_reads.json"
    CAP._atomic_json(battery_json, battery_reads)
    tensor_path = battery_dir / "ingredient_tensors.pt"
    tmp = tensor_path.with_suffix(".pt.tmp")
    torch.save(tensors, tmp)
    os.replace(tmp, tensor_path)
    _upload_paths(cfg, [tab_path], f"{HF_PREFIX_1979}/predictor_tables")
    _upload_paths(cfg, [tensor_path, sigma_path, battery_json], f"{HF_PREFIX_1979}/battery")
    return [str(tab_path), str(tensor_path), str(battery_json)]


def FIT_dev():
    import issue1768_fit as FIT

    return FIT._device()


# ── f1f: judge inputs ─────────────────────────────────────────────────────────


def run_f1f(cfg: Cfg, manifests: dict) -> list[str]:
    """(sha, prefix_id, response_text) for the 13 judged states -> jsonl shards."""
    judged = ["base_content"] + [r["arm_id"] for r in manifests["content_arms"]]
    out_dir = cfg.out_root / "judge_inputs"
    outputs: list[str] = []
    for state in judged:
        rows = _load_unit_rows(cfg, state)
        payload = [
            {
                "sha": r["row_sha"],
                "prefix_id": r["prefix_id"],
                "query_sha": r["query_sha"],  # direct (prefix, query) join for F2/F3
                "state": state,
                "response_text": r["response_text"],
            }
            for r in rows
        ]
        shards = _write_shards(out_dir, f"judge_inputs_{state}", payload)
        _upload_paths(cfg, shards, f"{HF_PREFIX_1979}/judge_inputs")
        outputs += [str(p) for p in shards]
    return outputs


# ── f1g: base-model TF over the marker arms' stored generations (plan v6) ────
# Amendment round `marker-a5-weights-vs-text`: capture the missing
# h_base(trained_text) leg — the BASE model teacher-forced on each marker
# arm's OWN stored on-policy generations (mirror of run_f1b_writes with the
# model/text roles inverted) — plus the aggregation unit that persists the F3
# decomposition inputs (per-prefix all/even/odd query-half means, re-derived
# parent on-policy means, marker-row-excluded variants).

MARKER_TOKEN_ID = 83399  # " ※" (leading space) — reused ONLY to identify emission rows


def _stage_gen_rows(cfg: Cfg, state: str) -> Path:
    """Stage the parent run's FULL gen_rows shards for ``state`` from the HF
    data repo into the layout ``_load_unit_rows`` reads (skip-if-present;
    scoped listing + per-file staging — the #833 pattern)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    dest = cfg.out_root / "gen" / state
    if list(dest.glob("rows.shard*.jsonl")):
        return dest
    listing = hub.retry_transient(
        lambda: hub.list_hf_files_under_path(
            HfApi(), X.HF_DATA_REPO, f"{HF_PREFIX_1979}/gen_rows/{state}", repo_type="dataset"
        ),
        what=f"gen_rows/{state} scoped listing",
    )
    wanted = [p for p in listing if Path(p).name.startswith("rows.shard")]
    assert wanted, (state, "no gen_rows shards on the HF data repo")
    for p in wanted:
        hub.stage_hub_file(X.HF_DATA_REPO, p, dest / Path(p).name, repo_type="dataset")
    return dest


def _stage_parent_onpol_store(cfg: Cfg, state: str) -> Path:
    """Stage the parent run's on-policy store for ``state`` (skip-if-present)."""
    from explore_persona_space.orchestrate import hub

    dest = cfg.out_root / "stores" / "onpolicy" / state / "store.pt"
    if not dest.exists():
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{HF_PREFIX_1979}/stores/onpolicy/{state}/store.pt",
            dest,
            repo_type="dataset",
        )
    return dest


def _f1g_assert_rows(state: str, rows: list[dict]) -> None:
    """Pre-spend asserts at FULL consumed-corpus grain (plan v6 §12 asm 1-2):
    non-empty, unique row_sha, every field the TF capture + store need — runs
    BEFORE any forward pass (kill criterion 1) and before the smoke slice."""
    assert rows, f"{state}: no gen rows staged"
    shas = [r["row_sha"] for r in rows]
    assert len(set(shas)) == len(shas), f"{state}: duplicate row_sha in gen rows"
    need = (
        "row_sha",
        "prefix_id",
        "query_sha",
        "persona",
        "prompt_token_ids",
        "response_token_ids",
        "prefix_len",
        "context_len",
    )
    for i, r in enumerate(rows):
        missing = [k for k in need if k not in r]
        assert not missing, (state, i, r.get("row_sha"), f"gen row missing fields {missing}")


def _f1g_slice_rows(cfg: Cfg, manifests: dict, rows: list[dict]) -> list[dict]:
    """Restrict full gen rows to the manifests' (possibly --panel/--query
    limited) grid — the ONE slice path, order-preserving; a no-op filter on
    the full production grid."""
    keep_p = {m["prefix_id"] for m in manifests["members"]}
    keep_q = {q["sha"] for q in manifests["queries"]}
    kept = [r for r in rows if r["prefix_id"] in keep_p and r["query_sha"] in keep_q]
    assert kept, "f1g slice selected zero rows (panel/query limits vs gen rows mismatch)"
    return kept


def run_f1g_basetf(cfg: Cfg, manifests: dict, arm_id: str) -> list[str]:
    """h_base(trained_text): BASE-model TF span-means + last-token positions
    over ``arm_id``'s stored on-policy generations (plan v6 §4 Diff 1) —
    ``run_f1b_writes`` with the model/text roles inverted."""
    import torch

    assert arm_id in {r["arm_id"] for r in manifests["marker_arms"]}, (
        arm_id,
        "f1g runs on marker arms only (plan v6 §5)",
    )
    _stage_gen_rows(cfg, arm_id)
    rows = _load_unit_rows(cfg, arm_id)
    _f1g_assert_rows(arm_id, rows)  # full-grain, BEFORE TF spend
    rows = _f1g_slice_rows(cfg, manifests, rows)
    persona_names = [m["prefix_id"] for m in manifests["members"]]
    spans, positions = _tf_capture_rows(cfg, X.BASE_MODEL, rows, persona_names)
    if torch.cuda.is_available():  # plan §12 asm 6: report the realized HBM peak
        logger.info(
            "[f1g] %s: cuda max_memory_allocated=%.1f GiB",
            arm_id,
            torch.cuda.max_memory_allocated() / 2**30,
        )
    store = cfg.out_root / "stores" / "basetf_onpolicy" / arm_id / "store.pt"
    _save_store(store, arm_id, f"basetf_onpolicy(base on {arm_id} text)", rows, spans, positions)
    _upload_paths(cfg, [store], f"{HF_PREFIX_1979}/stores/basetf_onpolicy/{arm_id}")
    return [str(store)]


def run_f1g_means(cfg: Cfg, manifests: dict) -> list[str]:
    """Aggregate the f1g stores into the F3 decomposition inputs (plan v6 §4
    Diff 1 step 2-4): per (marker arm, layer) all/even/odd per-prefix means of
    h_base(trained_text), the re-derived parent on-policy all-mean (F3 parity
    input), marker-row-excluded variants of BOTH trained-text stores, per-arm
    marker-row counts + trained-vs-base text-identity fractions. Fail-loud
    order-sensitive row-set identity asserts at the consumed grain
    (kill criterion: halt BEFORE any verdict input is written)."""
    import torch

    marker = [r["arm_id"] for r in manifests["marker_arms"]]
    assert marker, "f1g:means — no marker arms in manifests"
    prefix_ids = [m["prefix_id"] for m in manifests["members"]]
    queries = manifests["queries"]

    # base_mk gen rows: trained-vs-base text-identity diagnostic (plan v6 §4
    # analyzer-weigh item (a); token-id equality per (prefix, query) row).
    _stage_gen_rows(cfg, "base_mk")
    base_resp = {r["row_sha"]: r["response_token_ids"] for r in _load_unit_rows(cfg, "base_mk")}

    tensors: dict = {}
    per_arm_meta: dict = {}
    for k, arm_id in enumerate(marker):
        rows = _f1g_slice_rows(cfg, manifests, _load_unit_rows(cfg, arm_id))
        shas = [r["row_sha"] for r in rows]
        basetf = torch.load(
            cfg.out_root / "stores" / "basetf_onpolicy" / arm_id / "store.pt",
            map_location="cpu",
            weights_only=False,
        )
        parent = torch.load(
            _stage_parent_onpol_store(cfg, arm_id), map_location="cpu", weights_only=False
        )
        # row-set identity (order-sensitive; plan §6 criterion 2). Under the
        # smoke slice the parent store is masked to the sliced grid; in
        # production the mask keeps every row, so this is full equality.
        keep = set(shas)
        parent_mask = [s in keep for s in parent["row_sha"]]
        parent_sel = [s for s, m in zip(parent["row_sha"], parent_mask) if m]
        print(
            f"[f1g-means] arm {k + 1}/{len(marker)} {arm_id}: n_rows={len(shas)} "
            f"basetf_rows={len(basetf['row_sha'])} parent_sel={len(parent_sel)}",
            flush=True,
        )
        assert basetf["row_sha"] == shas, (
            arm_id,
            "basetf store row_sha list != gen_rows (order-sensitive identity)",
        )
        assert parent_sel == shas, (
            arm_id,
            "parent onpolicy store row_sha list != gen_rows (order-sensitive identity)",
        )
        # emission-row mask (id 83399 in the arm's OWN response ids) + text identity
        is_mk = [MARKER_TOKEN_ID in r["response_token_ids"] for r in rows]
        mk_shas = {s for s, m in zip(shas, is_mk) if m}
        not_mk = [not m for m in is_mk]
        with_base = [r for r in rows if r["row_sha"] in base_resp]
        n_ident = sum(1 for r in with_base if base_resp[r["row_sha"]] == r["response_token_ids"])
        even = _query_parity_mask(basetf, queries, 0)
        odd = _query_parity_mask(basetf, queries, 1)
        parent_not_mk = [m and (s not in mk_shas) for s, m in zip(parent["row_sha"], parent_mask)]
        per_arm_meta[arm_id] = {
            "n_rows": len(rows),
            "n_marker_rows": int(sum(is_mk)),
            "n_rows_with_base_counterpart": len(with_base),
            "n_text_identical_to_base": int(n_ident),
            "text_identity_frac": (n_ident / len(with_base)) if with_base else None,
        }
        print(f"[f1g-means] {arm_id}: {per_arm_meta[arm_id]}", flush=True)
        f64 = torch.float64
        for layer in LAYERS_1979:
            slot = f"{arm_id}/L{layer}"
            variants = {
                "Hbar_all": (basetf, None),
                "Hbar_even": (basetf, even),
                "Hbar_odd": (basetf, odd),
                "Obar_all": (parent, parent_mask),
                "Hbar_all_nomk": (basetf, not_mk),
                "Hbar_even_nomk": (basetf, [e and m for e, m in zip(even, not_mk)]),
                "Hbar_odd_nomk": (basetf, [o and m for o, m in zip(odd, not_mk)]),
                "Obar_all_nomk": (parent, parent_not_mk),
            }
            for name, (store, mask) in variants.items():
                if mask is not None:
                    covered = {p for p, m in zip(store["row_prefix_id"], mask) if m}
                    if not set(prefix_ids) <= covered:
                        # Production grain: an emptied prefix cell is a genuine
                        # premise violation — halt (kill criterion). Smoke slice
                        # (1 prefix x 2 queries): a single marker/dropped row can
                        # legitimately empty a parity(-nomk) cell — skip the
                        # variant loudly instead of a #1345-class false kill.
                        assert cfg.limited, (
                            arm_id,
                            layer,
                            name,
                            "mask empties a prefix cell at production grain",
                        )
                        logger.warning(
                            "[f1g-means] %s L%d %s: empty prefix cell under the "
                            "smoke slice — variant skipped (smoke-only)",
                            arm_id,
                            layer,
                            name,
                        )
                        continue
                tensors[f"{slot}/{name}"] = _prefix_means(
                    store, layer, "response", prefix_ids, mask, dtype=f64
                ).to(torch.float16)

    out = cfg.out_root / "battery" / "basetf_decomp_inputs.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            **_meta(),
            "prefix_ids": prefix_ids,
            "layers": list(LAYERS_1979),
            "marker_token_id": MARKER_TOKEN_ID,
            "per_arm": per_arm_meta,
        },
        **tensors,
    }
    tmp = out.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    _upload_paths(cfg, [out], f"{HF_PREFIX_1979}/battery")
    return [str(out)]


# ── worker dispatch ───────────────────────────────────────────────────────────


def run_unit(cfg: Cfg, manifests: dict, key: str) -> list[str]:
    print(f"[phase={key.split(':')[0]} unit={key}]", flush=True)
    parts = key.split(":")
    if parts[0] == "f1a":
        return run_f1a(cfg, manifests, parts[1])
    if parts[0] == "f1b" and parts[1] == "gate":
        return run_f1b_gate(cfg, manifests, parts[2])
    if parts[0] == "f1b" and parts[1] == "w":
        return run_f1b_writes(cfg, manifests, parts[2], "base_content")
    if parts[0] == "f1b" and parts[1] == "wmk":
        return run_f1b_writes(cfg, manifests, parts[2], "base_mk")
    if parts[0] == "f1b" and parts[1] == "slotbase":
        return run_f1b_slot(cfg, manifests, parts[2], "base")
    if parts[0] == "f1b" and parts[1] == "slotown":
        return run_f1b_slot(cfg, manifests, parts[2], "own")
    if parts[0] == "f1c":
        return run_f1c(cfg, manifests, parts[1])
    if parts[0] == "f1d" and parts[1] == "stage":
        return run_f1d_stage(cfg)
    if parts[0] == "f1d":
        return run_f1d_fit(cfg, parts[1], parts[2], int(parts[3]))
    if parts[0] == "f1e":
        return run_f1e(cfg, manifests)
    if parts[0] == "f1f":
        return run_f1f(cfg, manifests)
    if parts[0] == "f1g" and parts[1] == "basetf":
        return run_f1g_basetf(cfg, manifests, parts[2])
    if parts[0] == "f1g" and parts[1] == "means":
        return run_f1g_means(cfg, manifests)
    raise ValueError(f"unknown work-item key: {key}")


def _visible_gpus() -> list[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return [d for d in cvd.split(",") if d != ""]
    import torch

    n = torch.cuda.device_count()
    assert n >= 1, "no CUDA device visible — F1 is a GPU phase (plan §9)"
    return [str(i) for i in range(n)]


def dispatch(cfg: Cfg, manifests: dict, items: list[Item]) -> None:
    """Work-conserving round-robin over ALL visible GPUs (no wave barrier):
    an idle GPU always takes the next dep-satisfied pending item."""
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    pending = [it for it in items if not _done(cfg, it.key)]
    logger.info(
        "[dispatch] %d items total, %d pending (%d resumed-done)",
        len(items),
        len(pending),
        len(items) - len(pending),
    )
    phases = sorted({it.phase for it in pending})
    for ph in phases:  # resume-aware pending-set scaling of the disk floor
        n_ph = sum(1 for it in items if it.phase == ph)
        n_pend = sum(1 for it in pending if it.phase == ph)
        need = max(WORKER_HEADROOM_GB, PHASE_HEADROOM_GB[ph] * n_pend / max(n_ph, 1))
        assert_out_root_headroom(cfg.out_root, need, phase=ph)
    gpus = _visible_gpus()
    if cfg.max_parallel is not None:
        gpus = gpus[: cfg.max_parallel]
    logger.info("[dispatch] %d workers (CVD pins: %s)", len(gpus), gpus)
    running: dict[str, tuple[subprocess.Popen, Item, float]] = {}  # gpu -> (proc, item, t0)
    done_keys = {it.key for it in items if _done(cfg, it.key)}
    failures: list[str] = []
    exc_class_counts: dict[str, int] = {}
    abort_reason: str | None = None
    n_total = len(pending)
    n_done = 0
    script = str(Path(__file__).resolve())

    def _ready(it: Item) -> bool:
        if any(d not in done_keys for d in it.deps):
            return False
        heavy_running = sum(1 for _, r, _ in running.values() if r.heavy_model)
        if it.heavy_model and heavy_running >= MAX_HEAVY_MODEL_CONCURRENT:
            return False
        if it.model_key and any(r.model_key == it.model_key for _, r, _ in running.values()):
            return False  # same merged-dir lifecycle never concurrent (#1768 _model_key guard)
        return True

    while pending or running:
        # fill idle GPUs (work-conserving: any ready item, no phase barrier;
        # per-unit failures are non-fatal up to the budget — only abort stops fills)
        for gpu in gpus:
            if gpu in running or abort_reason:
                continue
            nxt = next((it for it in pending if _ready(it)), None)
            if nxt is None:
                continue
            pending.remove(nxt)
            cmd = [
                sys.executable,
                script,
                "--worker-unit",
                nxt.key,
                "--gpu-id",
                gpu,
                *cfg.worker_flags(),
            ]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}  # launcher-env CVD pin
            proc = subprocess.Popen(cmd, env=env)
            running[gpu] = (proc, nxt, time.time())
            logger.info("[dispatch] gpu%s <- %s (pid %d)", gpu, nxt.key, proc.pid)
        if not running:
            if pending and not failures and not abort_reason:
                raise RuntimeError(
                    f"deadlock: {len(pending)} pending items, none ready "
                    f"(unmet deps?): {[it.key for it in pending][:8]}"
                )
            break
        time.sleep(5.0)
        for gpu in list(running):
            proc, it, t0 = running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            del running[gpu]
            wall = time.time() - t0
            if rc == 0 and _done(cfg, it.key):
                done_keys.add(it.key)
                n_done += 1
                print(f"[f1] unit {n_done}/{n_total} {it.key} elapsed={wall:.0f}s", flush=True)
            else:
                exc_class = _read_failure_class(cfg, it.key)
                failures.append(f"{it.key} rc={rc} exc={exc_class}")
                exc_class_counts[exc_class] = exc_class_counts.get(exc_class, 0) + 1
                if len(failures) > FAILURE_BUDGET:
                    abort_reason = f"failure budget exceeded ({len(failures)} > {FAILURE_BUDGET})"
                elif exc_class_counts[exc_class] >= SYSTEMIC_EXC_REPEAT:
                    abort_reason = f"systemic failure: {exc_class} x{exc_class_counts[exc_class]}"
                logger.error(
                    "[dispatch] FAILED %s rc=%s exc=%s after %.0fs (failures %d/%d budget) — %s",
                    it.key,
                    rc,
                    exc_class,
                    wall,
                    len(failures),
                    FAILURE_BUDGET,
                    f"ABORT: {abort_reason}; draining running workers"
                    if abort_reason
                    else "non-fatal; independent units keep scheduling "
                    "(failed unit stays resumable — no done sentinel)",
                )
    if failures:
        skipped = [it.key for it in pending]
        raise RuntimeError(
            f"F1 units failed ({len(failures)} of {n_total}"
            + (f"; ABORTED early: {abort_reason}" if abort_reason else "")
            + (f"; {len(skipped)} pending never scheduled: {skipped[:8]}" if skipped else "")
            + f"): {failures}"
        )
    # the driver IS the dispatcher terminal (issue1979_dispatch.sh execs it in-process),
    # so this single end-of-run line is the dispatcher's own terminal emission.
    # The smoke-first leg must NOT emit the reserved [phase=done] token — the
    # poller reads the tail's newest [phase=...], and a smoke-leg done would
    # false-complete the run before the full leg starts (pod-side-reporting §1).
    terminal = "smoke_done" if cfg.smoke_subset else "done"
    print(f"[phase={terminal}]", flush=True)  # noqa: phase-done-reserved
    CAP._atomic_json(
        cfg.out_root / "f1_results.json",
        {"issue": 1979, "phase": "f1", "status": terminal, "n_items": len(items), **_meta()},
    )


# ── verification modes ────────────────────────────────────────────────────────


def plan_only(cfg: Cfg, manifests: dict, items: list[Item]) -> None:
    m, q = manifests["members"], manifests["queries"]
    print(
        f"[plan] prefixes={len(m)} queries={len(q)} "
        f"content_arms={len(manifests['content_arms'])} "
        f"marker_arms={len(manifests['marker_arms'])} "
        f"sliced={cfg.limited} (panel_limit={cfg.panel_limit}, "
        f"query_limit={cfg.query_limit})"
    )
    by_phase: dict[str, list[Item]] = {}
    for it in items:
        by_phase.setdefault(it.phase, []).append(it)
    for ph in sorted(by_phase):
        print(f"[plan] {ph}: {len(by_phase[ph])} items")
    for it in items:
        state = "done" if _done(cfg, it.key) else "pending"
        deps = f" deps={list(it.deps)[:3]}{'...' if len(it.deps) > 3 else ''}" if it.deps else ""
        print(f"[plan]   {it.key} [{state}]{deps}")
    print(f"[plan] total={len(items)} pending={sum(not _done(cfg, i.key) for i in items)}")


def import_check() -> None:
    """Resolve EVERY deferred import + signature-bind the reused call sites."""
    import inspect

    import numpy  # noqa: F401
    import torch  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401
    from peft import PeftModel  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    from vllm import LLM, SamplingParams  # noqa: F401

    import issue1768_directions as DIR
    import issue1768_fit as FIT
    import issue1768_lasttoken_fit as LTF
    import issue779_ffc_n1m_fits  # noqa: F401  (FIT._fit_map's deferred dep)
    from scipy.linalg import solve_triangular  # noqa: F401  (_whiten_solve's deferred dep)

    from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.analysis.representation_shift import (
        _build_generation_prompts,
        _reap_vllm_engine,  # noqa: F401
        _teacher_forced_span_means,
        _vllm_enforce_eager,  # noqa: F401
        compute_prompt_spans,  # noqa: F401
    )
    from explore_persona_space.eval.marker_logprob import (
        assert_gauge_free_adapter_config,  # noqa: F401
        compute_marker_slot_stats,
        validate_marker_slot_record,  # noqa: F401
    )
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    binds = [
        (
            _teacher_forced_span_means,
            dict(
                model_path="m",
                rows=[],
                persona_names=[],
                layers=[19],
                device="cpu",
                dtype=None,
                tf_batch_size=8,
            ),
        ),
        (
            compute_marker_slot_stats,
            dict(
                model=None,
                tokenizer=None,
                contexts=[],
                marker_text=MARKER_TEXT,
                device="cpu",
                include_argmax=True,
            ),
        ),
        (FIT._fit_map, dict(Xd=None, Yd=None, tr=[], val=[], te=[], dev="cpu")),
        (FIT._apply_payload, dict(payload={}, X_eval=None, dev="cpu")),
        (
            LTF.fetch_response,
            dict(cache=Path("/tmp"), kind="corpus_capture", unit="base_content", layers=[19]),
        ),
        (
            LTF.load_lasttoken,
            dict(out_root=Path("/tmp"), unit="base_content", layers=[19], position="last_prompt"),
        ),
        (DIR.corpus_sigma, dict(out_root=Path("/tmp"), layer=19)),
        (DIR.gate_read, dict(C0=None, delta_v=None, c_src=None, w=None, sigma=None)),
        (DIR.null_bands, dict(w=None, sigma=None, rng=None)),
        (CAP._mix_positive_rows, dict(cfg=None, arm=None)),
        (hub.stage_hub_prefix, dict(repo_id="r", prefix="p", dest_dir=Path("/tmp"))),
        (hub.stage_hub_file, dict(repo_id="r", path_in_repo="p", target=Path("/tmp/x"))),
        # f1g staging seams (plan v6 §4 Diff 1)
        (hub.list_hf_files_under_path, dict(api=None, repo_id="r", path="p", repo_type="dataset")),
        (hub.retry_transient, dict(fn=lambda: None, what="x")),
        (assert_out_root_headroom, dict(out_root=Path("/tmp"), need_gb=1.0)),
        (
            _build_generation_prompts,
            dict(tokenizer=None, personas={}, questions=[], user_wraps={}, prior_turns={}),
        ),
    ]
    for fn, kwargs in binds:
        inspect.signature(fn).bind(**kwargs)
    print(f"[import-check] OK — {len(binds)} deferred call sites resolved + signature-bound")


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--phase",
        default="f1",
        help="f1 (all parent phases) or comma list of f1a..f1f,f1g "
        "(f1g = the marker-a5-weights-vs-text base-TF amendment, plan v6)",
    )
    ap.add_argument("--out-root", type=Path, help="pod-side working root (plan §9)")
    ap.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="F0 manifest dir (default <out-root>/config; staged from HF if absent)",
    )
    ap.add_argument(
        "--panel-limit",
        type=int,
        default=None,
        help="smoke slice: first N prefix-panel members (same code path)",
    )
    ap.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="smoke slice: first N queries (same code path)",
    )
    ap.add_argument("--arms", default="", help="comma list of arm ids (smoke subset)")
    ap.add_argument(
        "--smoke-subset",
        action="store_true",
        help="smoke-first leg: one arm per realized (kind x method) class (derived from "
        "arms.json unless --arms is explicit) + f1d restricted to m0 span_mean L19; "
        "terminal line becomes [phase=smoke_done] (never the reserved [phase=done])",
    )
    ap.add_argument(
        "--skip-upload", action="store_true", help="smoke only: skip per-unit HF uploads"
    )
    ap.add_argument("--tf-batch", type=int, default=X.TF_BATCH_SIZE)
    ap.add_argument(
        "--skip-m0-ref-parity",
        action="store_true",
        help="EXPLICIT deviation: run F1d without the recorded-R² parity file",
    )
    ap.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="cap concurrent workers (default: all visible GPUs)",
    )
    ap.add_argument("--worker-unit", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="load manifests, print the realized work-item list, exit",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import + signature-bind reused call sites",
    )
    args = ap.parse_args(argv)

    if args.import_check:
        import_check()
        return 0
    if args.out_root is None:
        ap.error("--out-root is required (except --import-check)")
    phases = tuple(
        ("f1a", "f1b", "f1c", "f1d", "f1e", "f1f") if args.phase == "f1" else args.phase.split(",")
    )
    for ph in phases:
        assert ph in PHASE_HEADROOM_GB, f"unknown phase {ph!r}"
    cfg = Cfg(
        out_root=args.out_root.resolve(),
        config_dir=(args.config_dir or args.out_root / "config").resolve(),
        phases=phases,
        panel_limit=args.panel_limit,
        query_limit=args.query_limit,
        arms_filter=tuple(a for a in args.arms.split(",") if a),
        skip_upload=args.skip_upload,
        tf_batch=args.tf_batch,
        skip_m0_ref_parity=args.skip_m0_ref_parity,
        max_parallel=args.max_parallel,
        gpu_id=int(args.gpu_id),
        smoke_subset=args.smoke_subset,
    )
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    if cfg.smoke_subset and not cfg.arms_filter:
        # derive the one-arm-per-class subset from the unfiltered manifests, then
        # reload filtered so EVERY consumer (dispatcher + workers via --arms in
        # worker_flags) sees the identical reduced grid through the one path.
        cfg = dataclasses.replace(cfg, arms_filter=derive_smoke_arms(load_manifests(cfg)))
        logger.info("[smoke-subset] arms=%s (one per kind x method class)", cfg.arms_filter)
    manifests = load_manifests(cfg)
    items = [it for it in build_work_items(cfg, manifests) if it.phase in phases]

    if args.plan_only:
        plan_only(cfg, manifests, items)
        return 0
    if args.worker_unit:
        t0 = time.time()
        from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

        try:
            assert_out_root_headroom(
                cfg.out_root, WORKER_HEADROOM_GB, phase=f"worker:{args.worker_unit}"
            )
            outputs = run_unit(cfg, manifests, args.worker_unit)
        except BaseException as exc:  # breadcrumb for the dispatcher, then fail loud
            _write_failure(cfg, args.worker_unit, exc)
            raise
        _write_sentinel(cfg, args.worker_unit, time.time() - t0, outputs)
        _failure_path(cfg, args.worker_unit).unlink(missing_ok=True)  # stale prior-run crumb
        return 0
    dispatch(cfg, manifests, items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
