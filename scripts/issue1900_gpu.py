"""#1900 P1 GPU phase — marker TF capture, anchors, refits, predictor tables (plan §4 P1).

The ONLY GPU phase of #1900 (runs on a dispatched lane via `dispatch_issue.py
--gpus N`, never on the VM). Consumes the #1768 banked stores at pin
`c0726728…` plus the P0 config mirror `issue1900_leakrace/config/` (the SOLE
config read path on every lane — no `eval_results/` reads pod-side), and
produces, under `data/issue_1900/out/`:

- ``marker_tf/``          P1a: 13 unit-passes x judge-subset rows, four floats
                          per slot per model side (marker rule storage contract)
- ``anchors/``            P1b: per-mix base anchors (A_ctx/A_ans + even/odd
                          halves + split-half reliability cos + low_n_flag,
                          plan §12.1 v6; per-row context vectors for P9) and
                          per-LoRA-arm post anchors (A+_ctx/A+_ans)
- ``validation/``         P1c: TF fixed +/- margin (sycophancy, 300 contexts)
- ``maps/``               P1d: 33 ridge refits on judge-row-EXCLUDED splits
                          (leak-through-M guard) + identity+bias/kNN reads
- ``predictor_tables/``   P1e: one parquet per (arm, layer) with every roster
                          candidate P1-P9 + M1-M6 at both anchors
- ``judge_inputs/``       P1f: (sha, prompt, response_text) jsonl shards for P2

Work items (~77: 13 TF passes + ~15 base-anchor mixes + 14 post-anchor arms +
2 margin sides + 33 fits) are sharded round-robin across EVERY visible GPU via
per-worker subprocesses with `CUDA_VISIBLE_DEVICES` pinned in the LAUNCHER env
(the CVD gotcha). Every item writes its own output file (resume = skip-if-
exists); every sub-phase writes an atomic done-sentinel. `--smoke` runs EVERY
sub-phase on 2 arms (1 content + 1 marker) x 64 rows x their own mixes through
the SAME dispatcher/subprocess/production entrypoints (PASS_UNIFIED; fits keep
production n — a fit at smoke row counts is structurally under-determined).

Uploads land at `issue1900_leakrace/` (smoke: `issue1900_leakrace/smoke_probe/`
— the Hub-fenced-branch scratch-prefix probe) before exit; markers are posted
by the VM orchestrator via the dispatch handle/poller (no /workspace sentinel
contract — plan §9).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before torch/numpy: thread caps + HF/W&B credentials

import argparse  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1900_prep as P0  # noqa: E402  (arm lists + prefix constants)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1900.gpu")

ISSUE = 1900
HF_PREFIX = P0.HF_PREFIX  # issue1900_leakrace
CONFIG_HF_PREFIX = P0.CONFIG_HF_PREFIX
CORPUS_PIN = P0.CORPUS_PIN
SEED = 1900
MARKER_TOKEN_ID = X.MARKER_TOKEN_ID  # 83399, " ※" leading-space form
MARKER_TEXT = " " + chr(0x203B)  # built from the codepoint (Edit-tool unicode trap)
EOS_TOKEN_ID = 151645  # <|im_end|> (Qwen-2.5-7B; contrastive-negative slot competitor)
HEADROOM_GB = 80.0  # plan §9 mount-binding floor (staging ~45-50 GB x 1.5)
LAYERS = X.LAYERS  # (14, 19, 25)
N_MARGIN_CONTEXTS = 300
MARGIN_CAP = 32  # plan §4 P1c arithmetic: ~9.6k TF rows/side = 300 x 32
ADAPTER_SMOKE_ROWS = 50
# Frame-free identity floor (plan v7 §4 P1a FRAME-CORRECTED 2026-07-31 / §7 kill
# criterion 3; the #534 class). Calibration: a wrong/no-op adapter reads
# Δ logP ≈ 0 on its own training rows (base prior ≈ e^-20 either side), while any
# real in-window marker adapter reads ~20+ nats there (job 16092 measured 22.436)
# — +2 nats sits between the bands with ~10x margin both ways. The former ±1-nat
# equality against the #1481 manifest `delta_logp_mean` was a CROSS-FRAME assert
# (that number is a checkpoint-SELECTION read in #1481's eval-probe frame, window
# [5.0, 12.0]) and was unreproducible on training rows by construction.
ADAPTER_SMOKE_MIN_DELTA_NATS = 2.0
SMOKE_ARMS = ("imp-pers-con-lr3e5-s42", "mk-pers-con-lr5e6-s42")  # 1 content + 1 marker
SMOKE_ROWS = 64
SMOKE_MARGIN_CONTEXTS = 8
FT_SMOKE_ARM = "syc-pers-ft-con-s42"  # smoke covers one FT-mapped mix (plan §8)
P9_KS = (4, 16, 64)  # §11: k=16 primary; {4,64} sensitivity ride the same table (k capped at n)
ANCHOR_HARD_FLOOR_ROWS = 8  # plan §12.1 (v6): even/odd split-half needs >=4 rows/side
ANCHOR_LOW_N_ROWS = 40  # plan §12.1 (v6): 8 <= n < 40 -> LOUD WARN + persisted low_n_flag
DEVIATION_MULT = 2.0  # compute_deviation_over_2x boundary
# Plan §9 measured bases (per-call, at production shape):
BASIS_TF_PASS_S = 288.0  # #1768 pnf: ~0.04 GPU-h per 2,000-row TF pass -> 4,000 rows
BASIS_RIDGE_FIT_S = 35.03  # #1768 write_predictability walls.ridge_s @ n=15k
BASIS_TABLE_BLOCK_S = 60.0  # pilot-gated (plan §9 P1e row)
# P4 gate-parity pin — #1768 committed gate read (eval_results/issue_1768/
# p9_units/imp-pers-con-lr3e5-s42_L19.json @ origin/issue-1768, read 2026-07-30).
GATE_PARITY = {
    "arm_id": "imp-pers-con-lr3e5-s42",
    "layer": 19,
    "on_policy_rho": 0.240896725825268,
    "matched_text_rho": 0.18500955606791858,
    "atol": 1e-6,
}
OUT_DIRS = ("marker_tf", "anchors", "predictor_tables", "judge_inputs", "maps", "validation")


@dataclasses.dataclass
class Cfg:
    """Run configuration (every output-affecting knob is part of the regime)."""

    out_root: Path
    stage_root: Path
    smoke: bool = False
    layers: tuple[int, ...] = LAYERS
    tf_batch: int = X.TF_BATCH_SIZE
    upload: bool = True
    worker_slot: int | None = None  # set on per-GPU worker subprocesses
    n_slots: int = 1

    @property
    def i1768_root(self) -> Path:
        """Consumed #1768 mirror root: stage_root/<hub prefix> (mirror-root rule)."""
        return self.stage_root / X.HF_PREFIX

    @property
    def hf_prefix(self) -> str:
        return f"{HF_PREFIX}/smoke_probe" if self.smoke else HF_PREFIX

    def table_layers(self) -> tuple[int, ...]:
        return (19,) if self.smoke else self.layers


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def ensure_out_dirs(cfg: Cfg) -> None:
    """Create EVERY out-subdir this run's writers append/replace into.

    Called at main() entry AND at worker_main() entry: pathlib append-open
    (``out.open("a")``) creates NO parents, and a worker subprocess must never
    assume a dir another process created (crash-fix r7, fellows job 16100:
    workers 0+3 died FileNotFoundError opening validation/tf_margin_*.jsonl
    — the ``validation/`` parent existed in no process). Individual writers
    KEEP their local parent-mkdir guards (defense in depth; a future writer
    under a NEW subdir stays self-sufficient); this is the process-level
    floor, and its log line is the r7 fix-engaged signal per process.
    """
    dirs = [cfg.out_root, cfg.out_root / "logs", cfg.out_root / "config"]
    dirs += [cfg.out_root / name for name in OUT_DIRS]
    dirs.append(cfg.out_root / "anchors" / "post")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("[fs] out dirs ensured under %s (%d dirs)", cfg.out_root, len(dirs))


def _phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — metadata only
        return "unknown"


def _meta() -> dict:
    import torch

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "issue": ISSUE,
        "corpus_pin": CORPUS_PIN,
        "seed": SEED,
    }


def _device() -> str:
    import torch

    assert torch.cuda.is_available(), "P1 is GPU-only (plan §9); no CUDA device visible"
    return "cuda:0"


def _dtype():
    import torch

    return torch.bfloat16


def _physical_gpu_ids() -> list[str]:
    """Visible GPU ids for the per-worker CVD pins (nvidia-smi, never torch)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [t for t in cvd.split(",") if t.strip() != ""]
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ids = [ln.strip() for ln in out.splitlines() if ln.strip()]
    assert ids, "nvidia-smi enumerated zero GPUs"
    return ids


def _deviation(component: str, planned_h: float, projected_h: float, basis: str) -> dict:
    ratio = projected_h / max(planned_h, 1e-9)
    row = {
        "component": component,
        "planned_wall_h": round(planned_h, 4),
        "projected_wall_h": round(projected_h, 4),
        "ratio": round(ratio, 3),
        "basis": basis,
    }
    if ratio >= DEVIATION_MULT:
        print(
            f"[compute-deviation] component={component} planned_wall_h={planned_h:.3f} "
            f"projected_wall_h={projected_h:.3f} ratio={ratio:.2f} basis={basis}",
            flush=True,
        )
    return row


# ── config + staging ─────────────────────────────────────────────────────────


def load_run_config(cfg: Cfg) -> tuple[list[str], dict]:
    """(subset shas, arms.json payload) from the P0 HF config mirror."""
    from explore_persona_space.orchestrate import hub

    cdir = cfg.out_root / "config"
    files = {}
    for name in ("subset.json", "arms.json", "margin_chain.json", "judge_filter.json"):
        local = cdir / name
        if not local.exists():
            hub.stage_hub_file(
                X.HF_DATA_REPO, f"{CONFIG_HF_PREFIX}/{name}", local, repo_type="dataset"
            )
        files[name] = local
    subset = json.loads(files["subset.json"].read_text())
    arms = json.loads(files["arms.json"].read_text())
    assert subset["n"] == len(subset["shas"]) == 4_000, subset["n"]
    assert len(arms["arms"]) == 18, len(arms["arms"])
    return subset["shas"], arms


def _arm_entries(cfg: Cfg, arms_payload: dict) -> list[dict]:
    entries = arms_payload["arms"]
    if cfg.smoke:
        entries = [a for a in entries if a["arm_id"] in SMOKE_ARMS]
        assert len(entries) == 2, [a["arm_id"] for a in entries]
    return entries


def _assert_marker_tokenizer() -> None:
    from transformers import AutoTokenizer

    enc = AutoTokenizer.from_pretrained(X.BASE_MODEL).encode(MARKER_TEXT, add_special_tokens=False)
    assert enc == [MARKER_TOKEN_ID], f"marker token drift: {enc} != [{MARKER_TOKEN_ID}]"


def _stage_prefix(cfg: Cfg, prefix: str, *, revision: str | None) -> None:
    """Scoped-prefix stage into the mirror root (idempotent; per-file skip)."""
    from explore_persona_space.orchestrate import hub

    hub.stage_hub_prefix(
        X.HF_DATA_REPO, prefix, cfg.stage_root, repo_type="dataset", revision=revision
    )


def _bundle_keys_ok(path: Path, keys: tuple[str, ...]) -> None:
    """In-process realized-keys check (artifact-reuse (c)); mmap, no tensor read."""
    import torch

    realized = set(torch.load(path, map_location="cpu", mmap=True, weights_only=False).keys())
    missing = [k for k in keys if k not in realized]
    assert not missing, f"{path}: staged bundle missing keys {missing} (realized: {realized})"


def _verify_keys_subprocess(path: Path, keys: tuple[str, ...]) -> None:
    """One mechanized `verify_reused_artifact_keys.py` run per bundle FAMILY exemplar.

    ``--no-weights-only``: the #1768 bundles verified here are sha-pinned
    SELF-PRODUCED stores (staged @ CORPUS_PIN c0726728) whose torch.save
    metadata carries a non-primitive ``torch.torch_version.TorchVersion``
    global; torch>=2.6 (the fellows lane) defaults
    ``torch.load(weights_only=True)`` and rejects it (gotchas.md realized-keys
    entry: "pass weights_only=False only for a sha-pinned SELF-PRODUCED
    bundle whose metadata carries non-primitives"). The mmap no-storage-read
    path is KEPT — ``--allow-full-load`` is deliberately NOT passed: the
    bundles are zipfile-serialized (mmap opens them fine); the crash was the
    weights-only unpickler, not the file format (#1900 crash-fix r3,
    fellows job 16045).
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "verify_reused_artifact_keys.py"),
            "--artifact",
            str(path),
            "--keys",
            ",".join(keys),
            "--no-weights-only",
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert proc.returncode == 0, (
        f"verify_reused_artifact_keys FAILED for {path}:\n{proc.stdout}\n{proc.stderr}"
    )
    logger.info("[stage] realized-keys PASS (%s): %s", path.name, proc.stdout.strip()[-200:])


def _adapter_dir(cfg: Cfg, entry: dict) -> Path:
    """Consumer-side adapter dir under the staged VERBATIM prefix mirror.

    ``hub.stage_hub_prefix(repo, subfolder, dest)`` lands files at
    ``dest/<repo-relative path>`` (verbatim mirror), so the staged
    ``adapter_config.json`` lives at ``dest/<adapter_subfolder>/`` — never at
    ``dest/`` itself (the #928/#1481 staged-layout class; r1 Critical 1).
    """
    return cfg.stage_root / "adapters" / entry["arm_id"] / entry["adapter_subfolder"]


def _ft_dir(cfg: Cfg, entry: dict) -> Path:
    """Consumer-side full-FT checkpoint dir (same verbatim-mirror arithmetic)."""
    return cfg.stage_root / "ft_ckpt" / entry["arm_id"] / entry["ft_subfolder"]


def run_stage_probe(cfg: Cfg) -> dict:
    """(h)(iv) 1-file staging probe + consumer-open (CPU-runnable, pre-dispatch).

    One probe per (source-family x staged consumer) pair: stages ONLY the
    KB-scale config file of one LoRA arm and of the marker-FT arm at the EXACT
    verbatim-mirror layout production staging produces (``stage_hub_prefix``
    lands files at ``dest/<repo-relative path>``), then runs each consumer's
    own FIRST open against the staged tree — ``ModelPool.adapter``'s
    adapter_config.json read + gauge assert, and ``AutoConfig.from_pretrained``
    on the FT dir (``full_checkpoint``'s config resolution). Fails loud BEFORE
    the ~45-50 GB production staging / any model load (#928/#1481; artifact-
    reuse (h)(iv)). Runs standalone via ``--stage-probe`` (VM pre-dispatch CPU
    leg) AND at the top of every main-flow run.
    """
    from transformers import AutoConfig

    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config
    from explore_persona_space.orchestrate import hub

    _phase("stage_probe")
    _, arms_payload = load_run_config(cfg)
    by_id = {a["arm_id"]: a for a in arms_payload["arms"]}
    lora = by_id[SMOKE_ARMS[1]]  # marker LoRA smoke arm
    ft = by_id["mk-pers-ft-con-s42"]  # the ONE model-loaded FT arm (P1a)
    # 1-file stages expressed as dest/<repo-relative path> — the exact
    # `stage_hub_prefix` mirror arithmetic (`dest_dir / f`, hub.py).
    a_dest = cfg.stage_root / "adapters" / lora["arm_id"]
    a_rel = f"{lora['adapter_subfolder']}/adapter_config.json"
    hub.stage_hub_file(lora["adapter_repo"], a_rel, a_dest / a_rel, repo_type="model")
    f_dest = cfg.stage_root / "ft_ckpt" / ft["arm_id"]
    f_rel = f"{ft['ft_subfolder']}/config.json"
    hub.stage_hub_file(ft["ft_repo"], f_rel, f_dest / f_rel, repo_type="model")
    # consumer-side opens (the consumers' own path arithmetic — the seam under test)
    acfg = json.loads((_adapter_dir(cfg, lora) / "adapter_config.json").read_text())
    assert_gauge_free_adapter_config(acfg, context=str(_adapter_dir(cfg, lora)))
    ft_cfg = AutoConfig.from_pretrained(str(_ft_dir(cfg, ft)))
    rec = {
        "lora_arm": lora["arm_id"],
        "adapter_config_open": str(_adapter_dir(cfg, lora) / "adapter_config.json"),
        "ft_arm": ft["arm_id"],
        "ft_config_open": str(_ft_dir(cfg, ft) / "config.json"),
        "ft_config_model_type": str(ft_cfg.model_type),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
    }
    print(
        "[stage-probe] PASS "
        + json.dumps({k: rec[k] for k in ("lora_arm", "ft_arm", "ft_config_model_type")}),
        flush=True,
    )
    return rec


def phase_stage(cfg: Cfg, subset: list[str], arms_payload: dict) -> None:
    """Stage every P1 input (per-file scoped downloads @ pin; never snapshot)."""
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    _phase("p1_stage")
    done = cfg.out_root / "p1_stage.done.json"
    free = assert_out_root_headroom(cfg.out_root, HEADROOM_GB, phase="p1_stage")
    logger.info("[stage] headroom OK: %.1f GB free at %s", free, cfg.out_root)
    _assert_marker_tokenizer()
    if done.exists():
        logger.info("[stage] done-sentinel present — skipping staging")
        return

    entries = _arm_entries(cfg, arms_payload)
    root = cfg.i1768_root
    # inputs (corpus sample + arm registry) @ pin
    _stage_prefix(cfg, f"{X.HF_PREFIX}/inputs/corpus_sample.json", revision=CORPUS_PIN)
    _stage_prefix(cfg, f"{X.HF_PREFIX}/arm_registry.json", revision=CORPUS_PIN)
    # corpus trees: base units + selected arms (pooled + raw_rows shards)
    units = list(X.BASE_UNITS) + [a["arm_id"] for a in entries]
    for u in units:
        _stage_prefix(cfg, f"{X.HF_PREFIX}/corpus_capture/{u}", revision=CORPUS_PIN)
    for a in entries:
        _stage_prefix(cfg, f"{X.HF_PREFIX}/corpus_capture_tf/{a['arm_id']}", revision=CORPUS_PIN)
    # panel base trees (panel-source anchors) + the gate-parity arm's panel tree
    for beh in ("cas", "imp", "syc", "mk"):
        _stage_prefix(cfg, f"{X.HF_PREFIX}/panel_capture/base_{beh}", revision=CORPUS_PIN)
    if not cfg.smoke or GATE_PARITY["arm_id"] in {a["arm_id"] for a in entries}:
        _stage_prefix(
            cfg, f"{X.HF_PREFIX}/panel_capture/{GATE_PARITY['arm_id']}", revision=CORPUS_PIN
        )
    # delta_tf tbar per distinct mix (pos files staged lazily by _mix_positive_rows)
    for mix in sorted({a["mix_arm_id"] for a in entries}):
        _stage_prefix(cfg, f"{X.HF_PREFIX}/delta_tf/{mix}/tbar.pt", revision=CORPUS_PIN)
    # adapters (14 LoRA arms) + the marker FT checkpoint (overflow repo).
    # stage_hub_prefix is a VERBATIM prefix mirror (files land at
    # dest/<repo-relative path>), so consumers open dest/<subfolder> via
    # _adapter_dir/_ft_dir — asserted per staged arm HERE, at stage time,
    # BEFORE any model load (artifact-reuse (h)(iv); #928/#1481 class).
    from explore_persona_space.orchestrate import hub

    for a in entries:
        if a["method"] == "lora":
            dest = cfg.stage_root / "adapters" / a["arm_id"]
            hub.stage_hub_prefix(a["adapter_repo"], a["adapter_subfolder"], dest, repo_type="model")
            probe = _adapter_dir(cfg, a) / "adapter_config.json"
        elif a["kind"] == "marker":  # mk-pers-ft-con-s42 full checkpoint
            dest = cfg.stage_root / "ft_ckpt" / a["arm_id"]
            hub.stage_hub_prefix(a["ft_repo"], a["ft_subfolder"], dest, repo_type="model")
            probe = _ft_dir(cfg, a) / "config.json"
        else:  # content FT arms: consumed via banked stores only (no model load)
            continue
        assert probe.exists(), (
            f"staged-layout consumer-open FAIL for {a['arm_id']}: {probe} missing after "
            "staging — consumer dir != verbatim-mirror layout ((h)(iv); #928/#1481)"
        )
    # realized-keys verification: one mechanized run per family + in-process sweep
    base_pooled = root / "corpus_capture" / "base_content" / "pooled.pt"
    _verify_keys_subprocess(base_pooled, ("arms", "row_sha", "row_question_idx"))
    tf_ex = root / "corpus_capture_tf" / entries[0]["arm_id"] / "pooled_tf.pt"
    _verify_keys_subprocess(tf_ex, ("arms", "row_sha", "row_question_idx"))
    mix_ex = sorted({a["mix_arm_id"] for a in entries})[0]
    _verify_keys_subprocess(root / "delta_tf" / mix_ex / "tbar.pt", ("tbar", "n_rows"))
    _verify_keys_subprocess(root / "panel_capture" / "base_imp" / "pooled.pt", ("arms", "row_meta"))
    for u in units:
        _bundle_keys_ok(
            root / "corpus_capture" / u / "pooled.pt", ("arms", "row_sha", "row_question_idx")
        )
    for a in entries:
        _bundle_keys_ok(
            root / "corpus_capture_tf" / a["arm_id"] / "pooled_tf.pt", ("arms", "row_sha")
        )
        _bundle_keys_ok(root / "delta_tf" / a["mix_arm_id"] / "tbar.pt", ("tbar", "n_rows"))
    # rb tensors: consumer loader IS the check (ndim assert; artifact-reuse (c))
    import issue1768_directions as D

    rb = D.load_rb_tensors(root)
    assert set(rb) == {"cas", "imp", "syc", "mk"}, sorted(rb)
    _atomic_json(done, {"n_units": len(units), "n_arms": len(entries), **_meta()})
    logger.info("[stage] complete: %d corpus units, %d arms", len(units), len(entries))


# ── model pool (one resident base per worker; adapters applied UNMERGED) ─────


class ModelPool:
    """Base model resident once per worker; PEFT adapters applied unmerged.

    Unmerged application is the parity-faithful read (bf16 merge truncates
    small LoRA deltas — agent-memory feedback_bf16_merge_truncates entry);
    `PeftModel.unload()` restores the shared base between items. Full-FT
    checkpoints load fresh and are freed after their item.
    """

    def __init__(self, device: str, dtype) -> None:
        self.device = device
        self.dtype = dtype
        self._base = None
        self._tok = None

    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(X.BASE_MODEL)
        return self._tok

    def base(self):
        if self._base is None:
            from transformers import AutoModelForCausalLM

            self._base = AutoModelForCausalLM.from_pretrained(
                X.BASE_MODEL, torch_dtype=self.dtype, device_map={"": self.device}
            )
            self._base.eval()
        return self._base

    @contextlib.contextmanager
    def adapter(self, adapter_dir: Path):
        """Yield base+adapter (unmerged); gauge-assert the adapter config first."""
        from peft import PeftModel

        from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config

        acfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        assert_gauge_free_adapter_config(acfg, context=str(adapter_dir))
        peft_model = PeftModel.from_pretrained(self.base(), str(adapter_dir))
        peft_model.eval()
        try:
            yield peft_model
        finally:
            restored = peft_model.unload()
            assert restored is self.base() or restored is not None
            del peft_model
            _cuda_gc()

    @contextlib.contextmanager
    def full_checkpoint(self, ckpt_dir: Path):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir), torch_dtype=self.dtype, device_map={"": self.device}
        )
        model.eval()
        try:
            yield model
        finally:
            del model
            _cuda_gc()

    @contextlib.contextmanager
    def for_entry(self, cfg: Cfg, entry: dict | None):
        """entry=None -> base; LoRA -> adapter ctx; ft -> full checkpoint ctx."""
        if entry is None:
            yield self.base()
        elif entry["method"] == "lora":
            with self.adapter(_adapter_dir(cfg, entry)) as m:
                yield m
        else:
            with self.full_checkpoint(_ft_dir(cfg, entry)) as m:
                yield m


def _cuda_gc() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _decoder_layers(model):
    """Resolve the decoder block list on a bare or PEFT-wrapped CausalLM."""
    m = model
    for _ in range(6):
        layers = getattr(getattr(m, "model", None), "layers", None)
        if layers is not None:
            return layers
        nxt = getattr(m, "model", None) or getattr(m, "base_model", None)
        if nxt is None or nxt is m:
            break
        m = nxt
    raise AssertionError(f"cannot resolve decoder layers on {type(model).__name__}")


def _span_means_loaded(
    model, tokenizer, rows: list[dict], layers: list[int], spans: tuple[str, ...], tf_batch: int
) -> dict[str, dict[int, "object"]]:
    """`_teacher_forced_span_means` numerics on an ALREADY-LOADED model.

    The reused helper loads from a model PATH; the adapter re-passes (P1b post
    anchors) apply UNMERGED PEFT adapters to the worker's resident base, so
    this mirror keeps the identical right-pad + hook + span-pool arithmetic
    (`analysis/representation_shift._teacher_forced_span_means`) minus the
    load. Row dicts carry prompt_token_ids/response_token_ids/prefix_len/
    context_len (same asserts).
    """
    import torch

    device = next(model.parameters()).device
    for r in rows:
        p_len = len(r["prompt_token_ids"])
        assert 0 < r["prefix_len"] < r["context_len"] <= p_len
        assert len(r["response_token_ids"]) > 0
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    captured: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook_fn(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[li] = hs.detach()

        return hook_fn

    blocks = _decoder_layers(model)
    hooks = [blocks[li].register_forward_hook(make_hook(li)) for li in layers]
    pooled: dict[str, dict[int, list]] = {span: {li: [] for li in layers} for span in spans}
    try:
        for start in range(0, len(rows), tf_batch):
            batch = rows[start : start + tf_batch]
            seqs = [r["prompt_token_ids"] + r["response_token_ids"] for r in batch]
            max_len = max(len(s) for s in seqs)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                attn[i, : len(s)] = 1
            with torch.no_grad():
                _ = model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
            for li in layers:
                hs = captured[li]
                assert hs.shape[:2] == (len(batch), max_len), hs.shape
                for i, r in enumerate(batch):
                    p_len = len(r["prompt_token_ids"])
                    bounds = {
                        "prefix": (0, r["prefix_len"]),
                        "context": (0, r["context_len"]),
                        "response": (p_len, p_len + len(r["response_token_ids"])),
                    }
                    for span in spans:
                        s, e = bounds[span]
                        pooled[span][li].append(hs[i, s:e, :].float().mean(dim=0).cpu())
    finally:
        for h in hooks:
            h.remove()
        captured.clear()
    return {span: {li: torch.stack(v) for li, v in per.items()} for span, per in pooled.items()}


def _slot_stats_from_ids(
    model, ids_lists: list[list[int]], batch_size: int
) -> list[dict[str, float]]:
    """`compute_marker_slot_stats` post-tokenization body over TOKEN-ID contexts.

    The reused helper takes STRINGS and re-tokenizes; the #1768 raw_rows carry
    token ids, and the teacher-forced-capture rule mandates ID CONCAT (BPE
    seam merges shift re-tokenized positions — gotchas.md). Identical
    left-pad + position -1 four-float read (logp/z_marker/z_eos/logZ) +
    argmax; records validated at write time.
    """
    import torch

    from explore_persona_space.eval.marker_logprob import validate_marker_slot_record

    device = next(model.parameters()).device
    out: list[dict[str, float] | None] = [None] * len(ids_lists)
    order = sorted(range(len(ids_lists)), key=lambda i: len(ids_lists[i]))
    pad_id = 0
    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        chunk = [ids_lists[i] for i in idxs]
        assert all(len(c) > 0 for c in chunk), "zero-token slot context"
        max_len = max(len(c) for c in chunk)
        padded = [[pad_id] * (max_len - len(c)) + c for c in chunk]
        attn = [[0] * (max_len - len(c)) + [1] * len(c) for c in chunk]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        assert logits.ndim == 3, logits.shape
        for r, i in enumerate(idxs):
            raw = logits[r, -1, :].float()
            log_z = float(torch.logsumexp(raw, dim=-1).item())
            z_marker = float(raw[MARKER_TOKEN_ID].item())
            rec = {
                "logp": z_marker - log_z,
                "z_marker": z_marker,
                "z_eos": float(raw[EOS_TOKEN_ID].item()),
                "logZ": log_z,
                "argmax_id": int(torch.argmax(raw).item()),
            }
            validate_marker_slot_record(rec, context=f"p1a row {i}")
            out[i] = rec
        del logits
    assert all(o is not None for o in out)
    return out  # type: ignore[return-value]


# ── raw-rows + mix helpers ───────────────────────────────────────────────────


def _read_raw_rows(cfg: Cfg, unit_id: str) -> dict[str, dict]:
    """sha -> raw row (token ids + response_text) from the staged shards."""
    unit_dir = cfg.i1768_root / "corpus_capture" / unit_id
    rows: dict[str, dict] = {}
    for shard in sorted(unit_dir.glob("raw_rows_*.jsonl")):
        with shard.open(encoding="utf-8") as fh:  # text-mode iteration (U+2028 trap)
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows[r["prompt_sha"]] = r
    assert rows, f"no raw_rows shards staged under {unit_dir}"
    return rows


def _subset_rows(cfg: Cfg, unit_id: str, subset: list[str]) -> tuple[list[str], dict[str, dict]]:
    """Deterministic subset order restricted to the unit's kept rows."""
    raw = _read_raw_rows(cfg, unit_id)
    shas = [s for s in subset if s in raw]
    if cfg.smoke:
        shas = shas[:SMOKE_ROWS]
    assert shas, (unit_id, "no subset shas present in raw_rows")
    return shas, raw


def _slot_ids(row: dict) -> list[int]:
    """prompt ids + response ids STRIPPED at the first marker emission."""
    resp = list(row["response_token_ids"])
    if MARKER_TOKEN_ID in resp:
        resp = resp[: resp.index(MARKER_TOKEN_ID)]
    return list(row["prompt_token_ids"]) + resp


def _capture_cfg(cfg: Cfg):
    import issue1768_capture as C

    return C.Cfg(out_root=cfg.i1768_root, phases=(), smoke=False)


def _mix_rows(cfg: Cfg, mix_arm_id: str) -> tuple[list[dict], dict]:
    """The mix positives as TF rows via the reused `_mix_positive_rows`."""
    import issue1768_capture as C

    arm = C._full_arm_index()[mix_arm_id]
    assert arm.method == "lora", (mix_arm_id, "mix rows are keyed on the LoRA delta arm")
    return C._mix_positive_rows(_capture_cfg(cfg), arm)


def check_anchor_mix_floor(mix_arm_id: str, n_rows: int) -> bool:
    """Plan §12.1 (v6) anchor-mix row-count contract; returns the low-n flag.

    Hard floor n >= ANCHOR_HARD_FLOOR_ROWS (the even/odd split-half needs >=4
    rows per side); 8 <= n < ANCHOR_LOW_N_ROWS is the measured real-data regime
    — delta_tf pos.jsonl carries EXACTLY 20 rows per content mix at CORPUS_PIN
    and marker mixes carry 200 marker-positive rows (HF-probed 2026-07-31) —
    and is a LOUD WARN plus a persisted per-mix ``low_n_flag``, never a kill.
    """
    assert n_rows >= ANCHOR_HARD_FLOOR_ROWS, (
        mix_arm_id,
        n_rows,
        f"anchor mix hard floor n >= {ANCHOR_HARD_FLOOR_ROWS} (plan §12.1 v6)",
    )
    low_n = n_rows < ANCHOR_LOW_N_ROWS
    if low_n:
        logger.warning(
            "[anchors] mix %s n=%d < %d — low-n flag set (plan §12.1 v6)",
            mix_arm_id,
            n_rows,
            ANCHOR_LOW_N_ROWS,
        )
    return low_n


def _split_half_cos(even, odd) -> float:
    """Cosine between the even/odd half-mean anchors (per-mix split-half reliability)."""
    e = even.float().numpy().astype(np.float64)
    o = odd.float().numpy().astype(np.float64)
    return float(e @ o / (np.linalg.norm(e) * np.linalg.norm(o) + 1e-12))


# ── P1a: marker three-space TF capture ───────────────────────────────────────


def p1a_pass_items(entries: list[dict]) -> list[dict]:
    items = []
    for a in entries:
        if a["kind"] != "marker":
            continue
        items.append({"phase": "p1a", "text_unit": a["arm_id"], "model_arm": a["arm_id"]})
        items.append({"phase": "p1a", "text_unit": a["arm_id"], "model_arm": None})
    items.append({"phase": "p1a", "text_unit": "base_mk", "model_arm": None})
    return items


def _p1a_pass_name(item: dict) -> str:
    model_tag = item["model_arm"] or "base"
    return f"{model_tag}__on__{item['text_unit']}"


def p1a_out_path(cfg: Cfg, item: dict) -> Path:
    """P1a slots parquet path — SHARED by run_p1a_pass + fs-dryrun."""
    return cfg.out_root / "marker_tf" / f"{_p1a_pass_name(item)}_slots.parquet"


def run_p1a_pass(
    cfg: Cfg, pool: ModelPool, item: dict, subset: list[str], arms_by_id: dict
) -> Path:
    import pandas as pd

    name = _p1a_pass_name(item)
    out = p1a_out_path(cfg, item)
    if out.exists():
        return out
    t0 = time.time()
    shas, raw = _subset_rows(cfg, item["text_unit"], subset)
    ids_lists = [_slot_ids(raw[s]) for s in shas]
    entry = arms_by_id.get(item["model_arm"]) if item["model_arm"] else None
    with pool.for_entry(cfg, entry) as model:
        recs = _slot_stats_from_ids(model, ids_lists, cfg.tf_batch)
    df = pd.DataFrame(recs)
    df.insert(0, "sha", shas)
    df["model_tag"] = item["model_arm"] or "base"
    df["text_unit"] = item["text_unit"]
    df["n_stripped_at_marker"] = sum(
        1 for s in shas if MARKER_TOKEN_ID in raw[s]["response_token_ids"]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "pass": name,
        "n_rows": len(shas),
        "elapsed_s": round(time.time() - t0, 1),
        "storage_contract": "four floats per slot per model side (#530)",
        "wu_gauge": "ft arm trains W_U — z readouts carry the Q4 caveat"
        if entry and entry["method"] == "ft"
        else "gauge-free (attention-only LoRA, asserted)",
        **_meta(),
    }
    _atomic_json(out.with_suffix(".meta.json"), meta)
    return out


def p1a_gate_record(
    *,
    arm_id: str,
    n_mix_rows: int,
    median_training_delta_logp: float,
    manifest_selection_delta_logp: float,
    median_abs_delta_z_marker_corpus: float,
    median_corpus_delta_logp: float,
) -> dict:
    """Frame-free P1a identity verdict (plan v7 §4 P1a / §7 kill criterion 3).

    Pure (unit-testable without a model). Asserts (1) median TRAINING-ROW
    Δ logP >= ADAPTER_SMOKE_MIN_DELTA_NATS — direction+floor: a wrong/no-op
    adapter fails, any real marker adapter clears — and (2) median
    |Δ z_marker| > 0 on corpus rows (adapter actually applied). The #1481
    manifest selection-frame value is RECORDED with a frame note, never
    equality-compared (frames differ by construction — job 16092).
    Returns the gate record dict.
    """
    assert median_training_delta_logp >= ADAPTER_SMOKE_MIN_DELTA_NATS, (
        f"P1a adapter gate: median training-row Δ logP {median_training_delta_logp:.3f} "
        f"< +{ADAPTER_SMOKE_MIN_DELTA_NATS} nat floor (adapter identity broken — "
        "plan §7.3 frame-free gate)"
    )
    assert median_abs_delta_z_marker_corpus > 0.0, (
        "P1a adapter gate: median |Δ z_marker| == 0 on corpus rows (adapter not applied)"
    )
    return {
        "arm_id": arm_id,
        "n_mix_rows": n_mix_rows,
        "median_training_row_delta_logp": median_training_delta_logp,
        "min_delta_floor_nats": ADAPTER_SMOKE_MIN_DELTA_NATS,
        "manifest_selection_frame_delta_logp_mean": manifest_selection_delta_logp,
        "frame_note": (
            "manifest delta_logp_mean is #1481's checkpoint-SELECTION read in its "
            "eval-probe frame (window [5.0, 12.0]); memorized training-row slots "
            "legitimately read ~20+ nats — the two frames are recorded side-by-side, "
            "never equality-compared (frame-corrected 2026-07-31 after job 16092)"
        ),
        "median_abs_delta_z_marker_corpus": median_abs_delta_z_marker_corpus,
        "median_corpus_delta_logp": median_corpus_delta_logp,
    }


def run_p1a_adapter_smoke(cfg: Cfg, pool: ModelPool, arms_by_id: dict) -> dict:
    """Pre-fleet adapter-application gate (plan §4 P1a FRAME-CORRECTED; #534/#1481 class).

    For one marker LoRA arm: TF up to 50 of its own mix-positive rows and
    assert the FRAME-FREE identity read — median Δ logP(marker) trained−base
    >= +2 nats (direction+floor) — plus median |Δ z_marker| > 0 on 50 corpus
    rows, and a corpus-side sanity read (median corpus Δ logP, recorded only,
    no equality claim). The #1481 verdict-manifest `delta_logp_mean` is a
    checkpoint-SELECTION read in #1481's eval-probe frame (window [5.0, 12.0])
    and is recorded side-by-side, never equality-asserted: job 16092 measured
    22.436 nats on training rows — the EXPECTED memorized-slot signature of a
    correctly-applied in-window adapter, not a fault.
    """
    _phase("p1a_adapter_gate")
    out = cfg.out_root / "marker_tf" / "adapter_gate.json"
    if out.exists():
        return json.loads(out.read_text())
    marker_lora = [
        a for a in arms_by_id.values() if a["kind"] == "marker" and a["method"] == "lora"
    ]
    entry = next((a for a in marker_lora if a["arm_id"] == "mk-pers-con-lr5e6-s42"), marker_lora[0])
    rows, mix_meta = _mix_rows(cfg, entry["mix_arm_id"])
    # Slice caps at the realized mix n mechanically; measured at CORPUS_PIN
    # (2026-07-31): the marker mix carries 200 marker-positive rows, so the
    # 50-row read is fully filled (content mixes carry 20 — plan §12.1 v6).
    rows = rows[:ADAPTER_SMOKE_ROWS]
    ids_lists = [_slot_ids(r) for r in rows]
    base_recs = _slot_stats_from_ids(pool.base(), ids_lists, cfg.tf_batch)
    with pool.for_entry(cfg, entry) as model:
        arm_recs = _slot_stats_from_ids(model, ids_lists, cfg.tf_batch)
    deltas = sorted(a["logp"] - b["logp"] for a, b in zip(arm_recs, base_recs))
    median_delta = float(np.median(deltas))
    manifest_sel = float(entry["manifest_delta_logp_mean"])
    # Fix-engaged signal (crash-fix r6): both frames recorded BEFORE the verdict,
    # so even a failing gate logs the frame-annotated pair.
    logger.info(
        "[p1a-gate] training-row median Δ logP=%.3f (manifest selection-frame "
        "value=%.3f; frames differ by construction)",
        median_delta,
        manifest_sel,
    )
    # corpus-row leg: |Δ z_marker| > 0 (adapters actually applied) + sanity Δ logP
    subset_like = [r for r in _read_raw_rows(cfg, entry["arm_id"]).values()][:ADAPTER_SMOKE_ROWS]
    corpus_ids = [_slot_ids(r) for r in subset_like]
    base_c = _slot_stats_from_ids(pool.base(), corpus_ids, cfg.tf_batch)
    with pool.for_entry(cfg, entry) as model:
        arm_c = _slot_stats_from_ids(model, corpus_ids, cfg.tf_batch)
    med_abs_dz = float(
        np.median([abs(a["z_marker"] - b["z_marker"]) for a, b in zip(arm_c, base_c)])
    )
    med_corpus_dlogp = float(np.median([a["logp"] - b["logp"] for a, b in zip(arm_c, base_c)]))
    gate = p1a_gate_record(
        arm_id=entry["arm_id"],
        n_mix_rows=len(rows),
        median_training_delta_logp=median_delta,
        manifest_selection_delta_logp=manifest_sel,
        median_abs_delta_z_marker_corpus=med_abs_dz,
        median_corpus_delta_logp=med_corpus_dlogp,
    )
    gate.update({"mix_meta": mix_meta, **_meta()})
    _atomic_json(out, gate)
    logger.info(
        "[p1a-gate] PASS %s: training-row median ΔlogP %.3f >= +%.1f nat floor; "
        "corpus |Δz| %.3f; corpus ΔlogP %.3f",
        entry["arm_id"],
        median_delta,
        ADAPTER_SMOKE_MIN_DELTA_NATS,
        med_abs_dz,
        med_corpus_dlogp,
    )
    return gate


def run_mirror_parity(cfg: Cfg, pool: ModelPool, arms_by_id: dict) -> dict:
    """Smoke-only numeric-parity gate: in-code mirrors vs their originals (r2, Minor 6).

    (a) `_span_means_loaded` (loaded-model mirror) vs the plan-named
        `analysis.representation_shift._teacher_forced_span_means` (path-loading
        helper) on ONE batch of 8 real mix rows, base model both sides:
        per-(span, layer) cosine of the mean vectors >= 0.999 (same weights,
        same rows, same right-pad batching — a real mirror bug craters this).
    (b) `_slot_stats_from_ids` batched left-pad read vs a batch=1 no-pad read
        of the SAME id lists: |delta| <= 0.25 per four-float field (bf16
        batch-composition jitter budget), measured max reported. The original
        `compute_marker_slot_stats` re-tokenizes STRINGS, which the BPE-seam
        rule forbids on id-carrying rows (the documented reason the mirror
        exists), so text-vs-id parity is deliberately NOT asserted — the seam
        under test is the mirror's own batching arithmetic.
    """
    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    _phase("p1_mirror_parity")
    out = cfg.out_root / "marker_tf" / "mirror_parity.json"
    if out.exists():
        return json.loads(out.read_text())
    entry = next(a for a in arms_by_id.values() if a["kind"] == "content")
    rows, _mm = _mix_rows(cfg, entry["mix_arm_id"])
    rows = rows[:8]
    mirror = _span_means_loaded(
        pool.base(), pool.tokenizer(), rows, list(cfg.layers), ("context", "response"), cfg.tf_batch
    )
    helper = _teacher_forced_span_means(
        X.BASE_MODEL,
        rows,
        [entry["mix_arm_id"]],
        layers=list(cfg.layers),
        spans=("context", "response"),
        device=_device(),
        dtype=_dtype(),
        tf_batch_size=cfg.tf_batch,
    )
    span_stats = {}
    for span in ("context", "response"):
        for li in cfg.layers:
            a = mirror[span][li].mean(dim=0).numpy()
            b = helper[span][li].mean(dim=0).numpy()
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            span_stats[f"{span}_L{li}"] = {
                "cos": cos,
                "max_abs_diff": float(np.max(np.abs(a - b))),
            }
            assert cos >= 0.999, (span, li, cos, "span-means mirror diverged from helper")
    del helper, mirror
    _cuda_gc()
    mk = next(a for a in arms_by_id.values() if a["kind"] == "marker")
    ids_lists = [_slot_ids(r) for r in list(_read_raw_rows(cfg, mk["arm_id"]).values())[:8]]
    batched = _slot_stats_from_ids(pool.base(), ids_lists, batch_size=8)
    serial = [_slot_stats_from_ids(pool.base(), [ids], batch_size=1)[0] for ids in ids_lists]
    slot_max = {
        k: max(abs(fb[k] - fs[k]) for fb, fs in zip(batched, serial))
        for k in ("logp", "z_marker", "z_eos", "logZ")
    }
    for k, v in slot_max.items():
        assert v <= 0.25, (k, v, "batched-vs-serial slot-stats drift over bf16 budget")
    rec = {"span_means": span_stats, "slot_stats_max_abs_diff": slot_max, **_meta()}
    _atomic_json(out, rec)
    logger.info("[mirror-parity] PASS: %s", json.dumps(slot_max))
    return rec


# ── P1b: anchors (base per mix; post per LoRA arm) ───────────────────────────


def p1b_base_out_path(cfg: Cfg, mix_arm_id: str) -> Path:
    """P1b base-anchor path — SHARED by run_p1b_base_anchor + fs-dryrun."""
    return cfg.out_root / "anchors" / f"{mix_arm_id}.pt"


def p1b_post_out_path(cfg: Cfg, arm_id: str) -> Path:
    """P1b post-anchor path — SHARED by run_p1b_post_anchor + fs-dryrun."""
    return cfg.out_root / "anchors" / "post" / f"{arm_id}.pt"


def run_p1b_base_anchor(cfg: Cfg, item: dict) -> Path:
    import torch

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    mix = item["mix_arm_id"]
    out = p1b_base_out_path(cfg, mix)
    if out.exists():
        return out
    rows, mix_meta = _mix_rows(cfg, mix)  # FULL consumed grain (plan §12.1)
    low_n = check_anchor_mix_floor(mix, len(rows))  # hard >=8; WARN + flag at 8-39
    pooled = _teacher_forced_span_means(
        X.BASE_MODEL,
        rows,
        [mix],
        layers=list(cfg.layers),
        spans=("context", "response"),
        device=_device(),
        dtype=_dtype(),
        tf_batch_size=cfg.tf_batch,
    )
    tb = torch.load(
        cfg.i1768_root / "delta_tf" / mix / "tbar.pt", map_location="cpu", weights_only=False
    )
    payload: dict = {
        "mix_arm_id": mix,
        "n_rows": len(rows),
        "low_n_flag": low_n,  # plan §12.1 (v6): 8 <= n < 40 reliability flag
        "mix_meta": mix_meta,
        "A_ctx": {},
        "A_ans": {},
        "A_ctx_even": {},
        "A_ctx_odd": {},
        "A_ans_even": {},
        "A_ans_odd": {},
        "split_half_cos_ctx": {},  # per-layer even/odd half-mean cosine (reliability)
        "split_half_cos_ans": {},
        "rows_ctx": {},
        "tbar_cos": {},
        "meta": _meta(),
    }
    for li in cfg.layers:
        ctx = pooled["context"][li]
        ans = pooled["response"][li]
        payload["A_ctx"][li] = ctx.mean(dim=0)
        payload["A_ans"][li] = ans.mean(dim=0)
        payload["A_ctx_even"][li] = ctx[0::2].mean(dim=0)
        payload["A_ctx_odd"][li] = ctx[1::2].mean(dim=0)
        payload["A_ans_even"][li] = ans[0::2].mean(dim=0)
        payload["A_ans_odd"][li] = ans[1::2].mean(dim=0)
        payload["split_half_cos_ctx"][li] = _split_half_cos(
            payload["A_ctx_even"][li], payload["A_ctx_odd"][li]
        )
        payload["split_half_cos_ans"][li] = _split_half_cos(
            payload["A_ans_even"][li], payload["A_ans_odd"][li]
        )
        payload["rows_ctx"][li] = ctx.to(torch.float32)
        a = payload["A_ans"][li].numpy()
        t = tb["tbar"][li].float().numpy()
        cos = float(a @ t / (np.linalg.norm(a) * np.linalg.norm(t) + 1e-12))
        payload["tbar_cos"][li] = cos
        assert cos >= 0.99, (
            f"P1b {mix} L{li}: A_ans vs staged tbar cos {cos:.4f} < 0.99 "
            f"(anchor recipe drift — plan §4 P1b cross-check)"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    return out


def run_p1b_post_anchor(cfg: Cfg, pool: ModelPool, item: dict, arms_by_id: dict) -> Path:
    import torch

    arm_id = item["arm_id"]
    entry = arms_by_id[arm_id]
    assert entry["method"] == "lora", (arm_id, "post anchors are LoRA-only (plan §4 P1b)")
    out = p1b_post_out_path(cfg, arm_id)
    if out.exists():
        return out
    rows, _mix_meta = _mix_rows(cfg, entry["mix_arm_id"])
    low_n = check_anchor_mix_floor(entry["mix_arm_id"], len(rows))
    with pool.for_entry(cfg, entry) as model:
        pooled = _span_means_loaded(
            model, pool.tokenizer(), rows, list(cfg.layers), ("context", "response"), cfg.tf_batch
        )
    payload = {
        "arm_id": arm_id,
        "mix_arm_id": entry["mix_arm_id"],
        "n_rows": len(rows),
        "low_n_flag": low_n,  # plan §12.1 (v6)
        "A_ctx_plus": {li: pooled["context"][li].mean(dim=0) for li in cfg.layers},
        "A_ans_plus": {li: pooled["response"][li].mean(dim=0) for li in cfg.layers},
        "meta": _meta(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    return out


# ── P1c: graded-vs-reference TF-margin validation leg (sycophancy) ───────────

P1C_ARM = "syc-pers-po-lr1e5-s42"


def p1c_out_paths(cfg: Cfg, side: str) -> tuple[Path, Path]:
    """(jsonl, done) paths for one P1c side — SHARED by run_p1c_side + fs-dryrun."""
    return (
        cfg.out_root / "validation" / f"tf_margin_{side}.jsonl",
        cfg.out_root / "validation" / f"tf_margin_{side}.done.json",
    )


def run_p1c_side(
    cfg: Cfg, pool: ModelPool, item: dict, subset: list[str], arms_by_id: dict
) -> Path:
    """Fixed +/- completion margin per corpus context (llm-judging §E2 rule 19).

    Composition: each fixed (probe, answer) pair is scored under the corpus
    context by prefixing the context text into the SAME user turn
    (`{context}\\n\\n{probe}`) — recorded in the output meta (the plan does not
    pin the composition; flagged in the implementation report).
    Per-context JSONL append + resume (300 units > the ~50-unit checkpoint
    trigger, code-style intra-phase grain).
    """
    from explore_persona_space.eval.margin import build_fixed_pairs, compute_tf_margin

    side = item["side"]  # "arm" | "base"
    out, done = p1c_out_paths(cfg, side)
    # r7 (job 16100): append-open creates NO parents — guard before ANY open("a").
    out.parent.mkdir(parents=True, exist_ok=True)
    if done.exists():
        return out
    jf = json.loads((cfg.out_root / "config" / "judge_filter.json").read_text())
    pos_pairs, neg_pairs, pool_meta = build_fixed_pairs(jf, "sycophancy", cap=MARGIN_CAP)
    sample = X.load_corpus_sample(cfg.i1768_root)
    prompt_by_sha = {r["sha"]: r["prompt"] for r in sample["rows"]}
    n_ctx = SMOKE_MARGIN_CONTEXTS if cfg.smoke else N_MARGIN_CONTEXTS
    shas = [s for s in subset if s in prompt_by_sha][:n_ctx]
    done_shas: set[str] = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done_shas.add(json.loads(line)["sha"])
    entry = arms_by_id[P1C_ARM] if side == "arm" else None
    tok = pool.tokenizer()
    device = next(pool.base().parameters()).device
    t0 = time.time()
    with pool.for_entry(cfg, entry) as model:
        for k, sha in enumerate(shas):
            if sha in done_shas:
                continue
            ctx_text = prompt_by_sha[sha]

            def messages_fn(probe: str, _ctx: str = ctx_text) -> list[dict]:
                return [{"role": "user", "content": f"{_ctx}\n\n{probe}"}]

            res = compute_tf_margin(model, tok, messages_fn, pos_pairs, neg_pairs, device=device)
            rec = {
                "sha": sha,
                "side": side,
                "margin": res.margin,
                "pos_mean_ln_logp": res.pos_mean_ln_logp,
                "neg_mean_ln_logp": res.neg_mean_ln_logp,
                "n_pos": res.n_pos,
                "n_neg": res.n_neg,
            }
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"[p1c] unit {k + 1}/{len(shas)} {side}:{sha[:8]} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    _atomic_json(
        done,
        {
            "side": side,
            "arm_id": P1C_ARM if side == "arm" else "base",
            "n_contexts": len(shas),
            "cap_per_side": MARGIN_CAP,
            "pool_meta": pool_meta,
            "composition": "single user turn: f'{context}\\n\\n{probe}'; fixed pools "
            "identical across contexts (no selection-on-outcome)",
            **_meta(),
        },
    )
    return out


# ── P1d: ridge refits on judge-row-EXCLUDED splits (leak-through-M guard) ────


def _base_matrices(cfg: Cfg, base_unit: str, layer: int):
    """(C0, V0, shas) for a BASE unit — all kept rows, no arm join (M0 fits)."""
    import torch

    store = torch.load(
        cfg.i1768_root / "corpus_capture" / base_unit / "pooled.pt",
        map_location="cpu",
        weights_only=False,
    )
    c0 = np.asarray(store["arms"]["context"][layer].float().numpy(), dtype=np.float64)
    v0 = np.asarray(store["arms"]["response"][layer].float().numpy(), dtype=np.float64)
    return c0, v0, list(store["row_sha"])


def _judge_splits(shas: list[str], subset: set[str], n_val: int = 800):
    """(tr, val, te) index arrays: te = judge rows; fresh seed-1900 val split."""
    judge = np.asarray([i for i, s in enumerate(shas) if s in subset])
    rest = np.asarray([i for i, s in enumerate(shas) if s not in subset])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(rest))
    val = rest[perm[:n_val]]
    tr = rest[perm[n_val:]]
    assert len(judge) > 0 and len(tr) > 3_584, (len(judge), len(tr), "n_train > d (plan §10 (l))")
    return tr, val, judge


def p1d_fit_name(item: dict, arms_by_id: dict) -> str:
    """Fit-output stem for a P1d item — SHARED by run_p1d_fit + fs-dryrun."""
    kind = item["fit_kind"]
    if kind == "m0":
        return f"m0_L{item['layer']}"
    assert kind in ("mplus", "wmap"), kind
    entry = arms_by_id[item["arm_id"]]
    return f"{kind}_{entry['arm_id']}_L{entry['primary_layer']}"


def p1d_out_paths(cfg: Cfg, name: str) -> tuple[Path, Path]:
    """(pt, json) paths for one P1d fit — SHARED by _fit_and_persist + fs-dryrun."""
    return cfg.out_root / "maps" / f"{name}.pt", cfg.out_root / "maps" / f"{name}.json"


def _fit_and_persist(cfg: Cfg, name: str, Xd, Yd, tr, val, te) -> Path:
    """One `_fit_map` refit + identity+bias/kNN reads + payload persist."""
    import torch

    import issue1768_fit as F

    out_pt, out_js = p1d_out_paths(cfg, name)
    if out_pt.exists() and out_js.exists():
        return out_pt
    t0 = time.time()
    dev = torch.device(_device())
    pred_te, meta, payload = F._fit_map(Xd, Yd, tr, val, te, dev)
    reads = F._map_reads(pred_te, Yd[te])
    ib = F._identity_bias_reads(Xd[tr], Yd[tr], Xd[te], Yd[te])
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_pt.with_suffix(".pt.tmp")
    torch.save({"payload": payload, "name": name}, tmp)
    os.replace(tmp, out_pt)
    _atomic_json(
        out_js,
        {
            "name": name,
            "n_tr": int(len(tr)),
            "n_val": int(len(val)),
            "n_te": int(len(te)),
            "fit_meta": {k: v for k, v in meta.items() if not isinstance(v, np.ndarray)},
            "reads": reads,
            "identity_bias": ib,
            "elapsed_s": round(time.time() - t0, 1),
            "split": "judge rows excluded from tr/val (leak-through-M guard, seed 1900)",
            **_meta(),
        },
    )
    return out_pt


def run_p1d_fit(cfg: Cfg, item: dict, subset: list[str], arms_by_id: dict) -> Path:
    import issue1768_fit as F

    sub = set(subset)
    kind = item["fit_kind"]
    if kind == "m0":
        layer = item["layer"]
        c0, v0, shas = _base_matrices(cfg, "base_content", layer)
        tr, val, te = _judge_splits(shas, sub)
        return _fit_and_persist(cfg, p1d_fit_name(item, arms_by_id), c0, v0, tr, val, te)
    if kind == "mplus":
        entry = arms_by_id[item["arm_id"]]
        layer = entry["primary_layer"]
        cell = F.load_corpus_cell(entry["arm_id"], layer, cfg.i1768_root)
        tr, val, te = _judge_splits(cell["sha"], sub)
        return _fit_and_persist(
            cfg, p1d_fit_name(item, arms_by_id), cell["Cplus"], cell["Vplus"], tr, val, te
        )
    assert kind == "wmap", kind
    entry = arms_by_id[item["arm_id"]]
    layer = entry["primary_layer"]
    siblings = item["siblings"]  # same-behavior selected content arms, target excluded
    blocks_x, blocks_y = [], []
    for sib in siblings:
        cell = F.load_corpus_cell(sib, layer, cfg.i1768_root)
        keep = np.asarray([i for i, s in enumerate(cell["sha"]) if s not in sub])
        blocks_x.append(cell["C0"][keep])
        blocks_y.append((cell["Vplus_tf"] - cell["V0"])[keep])
    tgt = F.load_corpus_cell(entry["arm_id"], layer, cfg.i1768_root)
    tgt_judge = np.asarray([i for i, s in enumerate(tgt["sha"]) if s in sub])
    if not siblings:  # smoke-only degenerate sibling set (logged; production: 3 sibs)
        logger.warning("[p1d] SMOKE wmap %s: sibling set degenerates to self", entry["arm_id"])
        keep = np.asarray([i for i, s in enumerate(tgt["sha"]) if s not in sub])
        blocks_x.append(tgt["C0"][keep])
        blocks_y.append((tgt["Vplus_tf"] - tgt["V0"])[keep])
    Xd = np.vstack(blocks_x + [tgt["C0"][tgt_judge]])
    Yd = np.vstack(blocks_y + [(tgt["Vplus_tf"] - tgt["V0"])[tgt_judge]])
    n_pool = sum(b.shape[0] for b in blocks_x)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_pool)
    val = perm[:800]
    tr = perm[800:]
    te = np.arange(n_pool, n_pool + len(tgt_judge))
    assert len(tr) > 3_584, (len(tr), "wmap n_train > d")
    return _fit_and_persist(cfg, p1d_fit_name(item, arms_by_id), Xd, Yd, tr, val, te)


def wmap_siblings(entries: list[dict], target: dict) -> list[str]:
    return [
        a["arm_id"]
        for a in entries
        if a["kind"] == "content"
        and a["beh_key"] == target["beh_key"]
        and a["arm_id"] != target["arm_id"]
    ]


# ── P4 gate parity (plan §12.16 — refit-free reproduction of one #1768 value) ─


def run_gate_parity(cfg: Cfg) -> dict:
    """Reproduce the pinned #1768 A7 gate spearman within fp tolerance."""
    import torch

    import issue1768_directions as D
    import issue1768_fit as F

    _phase("p4_gate_parity")
    out = cfg.out_root / "maps" / "gate_parity.json"
    if out.exists():
        return json.loads(out.read_text())
    arm_id, layer = GATE_PARITY["arm_id"], GATE_PARITY["layer"]
    root = cfg.i1768_root
    arm = {a.arm_id: a for a in X.all_arms()}[arm_id]
    sigma = D.corpus_sigma(root, layer)
    legs = D.panel_write_legs(root, arm, layer)
    w = np.asarray(legs["w_primary"], dtype=np.float64)
    base_store = torch.load(
        root / "panel_capture" / f"base_{arm.beh_key}" / "pooled.pt",
        map_location="cpu",
        weights_only=False,
    )
    c_src_rows = D._panel_rows(base_store, legs["src_ctx"], layer, span="context")
    c_src = np.mean(list(c_src_rows.values()), axis=0)
    cell = F.load_corpus_cell(arm_id, layer, root)
    delta_v = cell["Vplus"] - cell["V0"]
    delta_v_tf = cell["Vplus_tf"] - cell["V0"]
    on_policy = D.gate_read(cell["C0"], delta_v, c_src, w, sigma)
    matched = D.gate_read(cell["C0"], delta_v_tf, c_src, w, sigma)
    rec = {
        "arm_id": arm_id,
        "layer": layer,
        "on_policy_rho": on_policy["spearman_rho"],
        "matched_text_rho": matched["spearman_rho"],
        "pinned": {k: v for k, v in GATE_PARITY.items()},
        **_meta(),
    }
    if not cfg.smoke:  # smoke slices cannot satisfy a full-grain fp parity pin
        for key in ("on_policy_rho", "matched_text_rho"):
            got, want = rec[key], GATE_PARITY[key]
            assert abs(got - want) <= GATE_PARITY["atol"], (
                f"P4 gate parity FAILED ({key}): {got} vs pinned {want} — refuse to compute P4"
            )
    _atomic_json(out, rec)
    logger.info(
        "[gate-parity] on_policy %.9f (pinned %.9f)",
        rec["on_policy_rho"],
        GATE_PARITY["on_policy_rho"],
    )
    return rec


# ── P1e: predictor tables (batched GEMMs; one parquet per (arm, layer)) ──────


def _ccos(rows: np.ndarray, anchor: np.ndarray, center: np.ndarray | None) -> np.ndarray:
    """Centered cosine of each row to the anchor (center=None -> raw cosine)."""
    r = rows - center if center is not None else rows
    a = anchor - center if center is not None else anchor
    num = r @ a
    den = np.linalg.norm(r, axis=1) * np.linalg.norm(a) + 1e-12
    return num / den


def _panel_anchor(cfg: Cfg, entry: dict, layer: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(A_ctx_ps, A_ans_ps) from the base panel's source context; None for bare."""
    import torch

    import issue1768_directions as D

    if entry["ctx_key"] == "bare":
        return None  # plan §5: panel-source anchor N/A for bare-context arms
    arm = {a.arm_id: a for a in X.all_arms()}[entry["arm_id"]]
    store = torch.load(
        cfg.i1768_root / "panel_capture" / f"base_{entry['beh_key']}" / "pooled.pt",
        map_location="cpu",
        weights_only=False,
    )
    src = D.source_context_id(arm, store)
    ctx_rows = D._panel_rows(store, src, layer, span="context")
    ans_rows = D._panel_rows(store, src, layer, span="response")
    return (
        np.mean(list(ctx_rows.values()), axis=0),
        np.mean(list(ans_rows.values()), axis=0),
    )


def _load_map_payload(cfg: Cfg, name: str):
    import torch

    p = p1d_out_paths(cfg, name)[0]
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)["payload"]


def _apply_map(payload, rows: np.ndarray) -> np.ndarray:
    import torch

    import issue779_ffc_n1m_fits as n1m

    return np.asarray(n1m.apply_map(payload, rows, torch.device(_device())), dtype=np.float64)


def p1e_out_path(cfg: Cfg, arm_id: str, layer: int) -> Path:
    """P1e predictor-table path — SHARED by run_p1e_table + fs-dryrun."""
    return cfg.out_root / "predictor_tables" / f"{arm_id}_L{layer}.parquet"


def run_p1e_table(
    cfg: Cfg, entry: dict, layer: int, subset: set[str], sigma_by_layer: dict, rb: dict
) -> Path:
    import pandas as pd
    import torch

    import issue1768_fit as F

    arm_id = entry["arm_id"]
    out = p1e_out_path(cfg, arm_id, layer)
    if out.exists():
        return out
    cell = F.load_corpus_cell(arm_id, layer, cfg.i1768_root)
    c0, v0 = cell["C0"], cell["V0"]
    cp, vp, vtf = cell["Cplus"], cell["Vplus"], cell["Vplus_tf"]
    cbar, vbar = c0.mean(axis=0), v0.mean(axis=0)
    anc = torch.load(
        cfg.out_root / "anchors" / f"{entry['mix_arm_id']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    a_ctx = np.asarray(anc["A_ctx"][layer].numpy(), dtype=np.float64)
    a_ans = np.asarray(anc["A_ans"][layer].numpy(), dtype=np.float64)
    halves = {
        h: (
            np.asarray(anc[f"A_ctx_{h}"][layer].numpy(), dtype=np.float64),
            np.asarray(anc[f"A_ans_{h}"][layer].numpy(), dtype=np.float64),
        )
        for h in ("even", "odd")
    }
    rows_ctx = np.asarray(anc["rows_ctx"][layer].numpy(), dtype=np.float64)
    ps = _panel_anchor(cfg, entry, layer)
    df = pd.DataFrame(
        {
            "sha": cell["sha"],
            "question_idx": cell["qidx"],
            "split": cell["split"],
            "corpus": cell["corpus"],
            "in_judge_subset": [s in subset for s in cell["sha"]],
        }
    )
    # deployable panel — training-centroid anchor (_tc) + panel-source (_ps)
    df["p1_tc"] = _ccos(c0, a_ctx, cbar)
    df["p2_tc"] = _ccos(v0, a_ans, vbar)
    for h in ("even", "odd"):
        df[f"p1_tc_{h}"] = _ccos(c0, halves[h][0], cbar)
        df[f"p2_tc_{h}"] = _ccos(v0, halves[h][1], vbar)
    df["p1_ps"] = _ccos(c0, ps[0], cbar) if ps else np.nan
    df["p2_ps"] = _ccos(v0, ps[1], vbar) if ps else np.nan
    m0 = _load_map_payload(cfg, f"m0_L{layer}")
    if m0 is not None:
        mpred = _apply_map(m0, c0)
        mbar = mpred.mean(axis=0)
        m0_actx = _apply_map(m0, a_ctx[None, :])[0]
        df["p3a_tc"] = _ccos(mpred, m0_actx, mbar)
        df["p3b_tc"] = _ccos(mpred, a_ans, mbar)
        if ps:
            df["p3a_ps"] = _ccos(mpred, _apply_map(m0, ps[0][None, :])[0], mbar)
            df["p3b_ps"] = _ccos(mpred, ps[1], mbar)
        else:
            df["p3a_ps"] = np.nan
            df["p3b_ps"] = np.nan
        df["p6"] = (mpred - mbar) @ rb[entry["beh_key"]][layer]
    else:  # smoke fits only the primary layer's M0
        for c in ("p3a_tc", "p3b_tc", "p3a_ps", "p3b_ps", "p6"):
            df[c] = np.nan
    sigma = sigma_by_layer[layer]["sigma"]

    def g_pred(c_src: np.ndarray) -> np.ndarray:
        a = np.linalg.solve(sigma, c_src)
        return c0 @ a / (float(c_src @ a) + 1e-12)

    df["p4_tc"] = g_pred(a_ctx)
    df["p4_ps"] = g_pred(ps[0]) if ps else np.nan
    df["p5"] = (v0 - vbar) @ rb[entry["beh_key"]][layer]
    wmap = (
        _load_map_payload(cfg, f"wmap_{arm_id}_L{layer}")
        if entry["kind"] == "content" and layer == entry["primary_layer"]
        else None
    )
    if wmap is not None:
        wpred = _apply_map(wmap, c0)
        df["p8a"] = np.linalg.norm(wpred, axis=1)
        df["p8b"] = _ccos(wpred, rb[entry["beh_key"]][layer], None)
    else:
        df["p8a"] = np.nan
        df["p8b"] = np.nan
    rc = rows_ctx - cbar
    rcn = rc / (np.linalg.norm(rc, axis=1, keepdims=True) + 1e-12)
    c0c = c0 - cbar
    c0n = c0c / (np.linalg.norm(c0c, axis=1, keepdims=True) + 1e-12)
    sims = c0n @ rcn.T  # (n_rows, n_mix)
    sims_sorted = np.sort(sims, axis=1)[:, ::-1]
    for k in P9_KS:
        kk = min(k, sims.shape[1])
        df[f"p9_k{k}"] = sims_sorted[:, :kk].mean(axis=1)
    # P9 split-half companions (k=16 primary): mix-row halves under the SAME
    # even/odd convention as the anchor halves (rows_ctx row order == ctx) —
    # persists the P9 reliability-ceiling inputs the plan §6 robustness
    # line (2) needs (concern p9-reliability-ceiling-not-persisted, r2).
    for h, sl in (("even", slice(0, None, 2)), ("odd", slice(1, None, 2))):
        sims_h = np.sort(sims[:, sl], axis=1)[:, ::-1]
        kk16 = min(16, sims_h.shape[1])
        df[f"p9_k16_{h}"] = sims_h[:, :kk16].mean(axis=1)
    # mechanistic panel
    post = _load_map_payload_post(cfg, arm_id) if entry["method"] == "lora" else None
    if post is not None:
        acp = np.asarray(post["A_ctx_plus"][layer].numpy(), dtype=np.float64)
        aap = np.asarray(post["A_ans_plus"][layer].numpy(), dtype=np.float64)
        df["m1_tc"] = _ccos(cp, acp, cp.mean(axis=0))
        df["m2_tc"] = _ccos(vp, aap, vp.mean(axis=0))
        mplus = (
            _load_map_payload(cfg, f"mplus_{arm_id}_L{layer}")
            if layer == entry["primary_layer"]
            else None
        )
        if mplus is not None:
            mp_pred = _apply_map(mplus, cp)
            df["m5_tc"] = _ccos(mp_pred, aap, mp_pred.mean(axis=0))
        else:
            df["m5_tc"] = np.nan
    else:  # FT arms: M1/M2/M5 excluded (post-anchors LoRA-only, plan §5)
        df["m1_tc"] = np.nan
        df["m2_tc"] = np.nan
        df["m5_tc"] = np.nan
    dc = cp - c0
    dv = vtf - v0
    df["m3"] = _ccos(dc, dc.mean(axis=0), None)
    df["m4"] = _ccos(dv, dv.mean(axis=0), None)
    df["m6"] = np.linalg.norm(dv, axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    _atomic_json(
        out.with_suffix(".meta.json"),
        {
            "arm_id": arm_id,
            "layer": layer,
            "n_rows": int(len(df)),
            "n_mix_rows": int(sims.shape[1]),
            # P9 k is capped at the realized mix n (plan §12.1 v6): p9_k{k} =
            # mean of top-min(k, n_mix) row cosines, so at n_mix=20 p9_k64
            # realizes k=n and the plan sensitivity set {4, min(16,n), n} maps
            # onto p9_k4 / p9_k16 / p9_k64 as available.
            "p9_k_effective": {str(k): int(min(k, sims.shape[1])) for k in P9_KS},
            "p9_k16_half_effective": {
                "even": int(min(16, (sims.shape[1] + 1) // 2)),
                "odd": int(min(16, sims.shape[1] // 2)),
            },
            "anchor_centering": "cosines centered by the corpus row-mean of the x-side "
            "space (c̄0 / v̄0 / mean-mapped); delta-space cosines (m3/m4/p8b) raw",
            "m3_m4_definition": "cos of the per-row delta to the ARM-MEAN delta "
            "(implementer definitional choice — flagged in the round report)",
            "p8_m5_layers": "primary layer only (fits exist there; NaN elsewhere)",
            **_meta(),
        },
    )
    return out


def _load_map_payload_post(cfg: Cfg, arm_id: str):
    import torch

    p = p1b_post_out_path(cfg, arm_id)
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def phase_tables(cfg: Cfg, entries: list[dict], subset: list[str]) -> dict:
    import issue1768_directions as D

    _phase("p1e_tables")
    sub = set(subset)
    rb = D.load_rb_tensors(cfg.i1768_root)
    sigma_by_layer = {li: D.corpus_sigma(cfg.i1768_root, li) for li in cfg.table_layers()}
    blocks = [(e, li) for e in entries for li in cfg.table_layers()]
    t_first = None
    for k, (e, li) in enumerate(blocks):
        t0 = time.time()
        run_p1e_table(cfg, e, li, sub, sigma_by_layer, rb)
        dt = time.time() - t0
        if t_first is None and dt > 1.0:  # first NON-resumed block = the pilot
            t_first = dt
        print(
            f"[p1e] unit {k + 1}/{len(blocks)} {e['arm_id']}_L{li} elapsed={dt:.1f}s",
            flush=True,
        )
    proj_h = (t_first or 0.0) * len(blocks) / 3600.0
    return _deviation(
        "P1e predictor tables",
        0.1,
        proj_h,
        f"first-block pilot {t_first or 0:.1f}s x {len(blocks)}",
    )


# ── P1f: judge inputs (sha, prompt, response_text) ───────────────────────────

SHARD_BYTES = 9_000_000  # <9 MB — non-LFS Hub path (upload-policy line-split rule)


def p1f_out_dir(cfg: Cfg) -> Path:
    """P1f judge-inputs dir — SHARED by run_p1f_unit + fs-dryrun."""
    return cfg.out_root / "judge_inputs"


def run_p1f_unit(cfg: Cfg, unit_id: str, subset: list[str], prompt_by_sha: dict) -> list[Path]:
    out_dir = p1f_out_dir(cfg)
    done = out_dir / f"{unit_id}.done.json"
    if done.exists():
        return [out_dir / n for n in json.loads(done.read_text())["shards"]]
    shas, raw = _subset_rows(cfg, unit_id, subset)
    shard_paths: list[Path] = []
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if not buf:
            return
        p = out_dir / f"{unit_id}.shard{len(shard_paths):02d}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(buf), encoding="utf-8")
        shard_paths.append(p)
        buf, size = [], 0

    n_missing_prompt = 0
    for sha in shas:
        prompt = prompt_by_sha.get(sha)
        if prompt is None:  # raw_rows carry NO prompt-text field (fact-check 2026-07-30)
            n_missing_prompt += 1
            continue
        line = (
            json.dumps(
                {
                    "sha": sha,
                    "unit": unit_id,
                    "prompt": prompt,
                    "response_text": raw[sha]["response_text"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        buf.append(line)
        size += len(line.encode("utf-8"))
        if size >= SHARD_BYTES:
            flush()
    flush()
    assert n_missing_prompt == 0, (unit_id, n_missing_prompt, "sha->prompt join incomplete")
    _atomic_json(
        done,
        {"unit": unit_id, "n_rows": len(shas), "shards": [p.name for p in shard_paths], **_meta()},
    )
    return shard_paths


def phase_judge_inputs(cfg: Cfg, entries: list[dict], subset: list[str]) -> None:
    _phase("p1f_judge_inputs")
    sample = X.load_corpus_sample(cfg.i1768_root)
    prompt_by_sha = {r["sha"]: r["prompt"] for r in sample["rows"]}
    units = [e["arm_id"] for e in entries if e["kind"] == "content"] + ["base_content"]
    for k, u in enumerate(units):
        t0 = time.time()
        shards = run_p1f_unit(cfg, u, subset, prompt_by_sha)
        print(
            f"[p1f] unit {k + 1}/{len(units)} {u} shards={len(shards)} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )


# ── uploads ──────────────────────────────────────────────────────────────────


def phase_upload(cfg: Cfg) -> None:
    """Per-dir folder commits + exact-set verify (smoke -> smoke_probe prefix)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    _phase("p1_upload")
    if not cfg.upload:
        logger.warning("[upload] --no-upload set — pod-side artifacts NOT persisted")
        return
    api = HfApi()
    for name in OUT_DIRS:
        local = cfg.out_root / name
        if not local.exists():
            continue
        files = sorted(p for p in local.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
        if not files:
            continue
        prefix = f"{cfg.hf_prefix}/{name}"
        hub._upload(
            local,
            repo_id=X.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=prefix,
            raise_on_error=True,
        )
        expected = [f"{prefix}/{p.relative_to(local)}" for p in files]
        missing = hub.verify_repo_paths_uploaded(
            api, X.HF_DATA_REPO, expected, path_in_repo=prefix, repo_type="dataset"
        )
        assert not missing, f"upload verify FAILED for {prefix}: missing {missing}"
        logger.info("[upload] %s verified (%d files)", prefix, len(files))
    summary = cfg.out_root / "gpu_phase_summary.json"
    if summary.exists():
        hub._upload(
            summary,
            repo_id=X.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{cfg.hf_prefix}/gpu_phase_summary.json",
            upload_as_file=True,
            raise_on_error=True,
        )


# ── item registry + worker/parent fan-out ────────────────────────────────────


def build_items(cfg: Cfg, entries: list[dict]) -> list[dict]:
    """The GPU work-item list, sharded round-robin across workers (plan §9)."""
    items: list[dict] = []
    items += p1a_pass_items(entries)
    for mix in sorted({a["mix_arm_id"] for a in entries}):
        items.append({"phase": "p1b_base", "mix_arm_id": mix})
    for a in entries:
        if a["method"] == "lora":
            items.append({"phase": "p1b_post", "arm_id": a["arm_id"]})
    if any(a["arm_id"] == P1C_ARM for a in entries) or cfg.smoke:
        items.append({"phase": "p1c", "side": "arm"})
        items.append({"phase": "p1c", "side": "base"})
    for layer in cfg.table_layers():
        items.append({"phase": "p1d", "fit_kind": "m0", "layer": layer})
    for a in entries:
        items.append({"phase": "p1d", "fit_kind": "mplus", "arm_id": a["arm_id"]})
        if a["kind"] == "content":
            items.append(
                {
                    "phase": "p1d",
                    "fit_kind": "wmap",
                    "arm_id": a["arm_id"],
                    "siblings": wmap_siblings(entries, a),
                }
            )
    return items


def run_item(cfg: Cfg, pool: ModelPool, item: dict, subset: list[str], arms_by_id: dict) -> None:
    ph = item["phase"]
    if ph == "p1a":
        run_p1a_pass(cfg, pool, item, subset, arms_by_id)
    elif ph == "p1b_base":
        run_p1b_base_anchor(cfg, item)
    elif ph == "p1b_post":
        run_p1b_post_anchor(cfg, pool, item, arms_by_id)
    elif ph == "p1c":
        # P1C_ARM is outside the smoke arm pair; smoke swaps in the content smoke arm
        # so the SAME production entrypoint runs end-to-end (stated smoke deviation).
        if cfg.smoke and P1C_ARM not in arms_by_id:
            arms_by_id = dict(arms_by_id)
            arms_by_id[P1C_ARM] = next(a for a in arms_by_id.values() if a["kind"] == "content")
        run_p1c_side(cfg, pool, item, subset, arms_by_id)
    elif ph == "p1d":
        run_p1d_fit(cfg, item, subset, arms_by_id)
    else:
        raise AssertionError(f"unknown item phase {ph}")


def worker_main(cfg: Cfg, subset: list[str], arms_payload: dict) -> None:
    """Per-GPU worker: run items where idx % n_slots == worker_slot."""
    ensure_out_dirs(cfg)  # r7: workers never assume main-process-created dirs
    entries = _arm_entries(cfg, arms_payload)
    arms_by_id = {a["arm_id"]: a for a in entries}
    items = build_items(cfg, entries)
    mine = [(i, it) for i, it in enumerate(items) if i % cfg.n_slots == cfg.worker_slot]
    pool = ModelPool(_device(), _dtype())
    pilots: dict[str, float] = {}
    for k, (idx, item) in enumerate(mine):
        t0 = time.time()
        run_item(cfg, pool, item, subset, arms_by_id)
        dt = time.time() - t0
        key = item["phase"]
        if key not in pilots and dt > 5.0:
            pilots[key] = dt
        print(
            f"[worker{cfg.worker_slot}] unit {k + 1}/{len(mine)} item#{idx} "
            f"{key}:{item.get('arm_id') or item.get('mix_arm_id') or item.get('side') or item.get('text_unit') or item.get('layer')} "
            f"elapsed={dt:.1f}s",
            flush=True,
        )
    _atomic_json(
        cfg.out_root / "logs" / f"worker{cfg.worker_slot}.done.json",
        {"slot": cfg.worker_slot, "n_items": len(mine), "pilots": pilots, **_meta()},
    )


def _spawn_workers(cfg: Cfg, argv_tail: list[str]) -> None:
    """Fan the item list across every visible GPU (CVD pinned in the LAUNCHER env)."""
    gpu_ids = _physical_gpu_ids()
    n = len(gpu_ids)
    logdir = cfg.out_root / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for slot, gid in enumerate(gpu_ids):
        log = logdir / f"worker{slot}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-slot",
            str(slot),
            "--n-slots",
            str(n),
            *argv_tail,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gid}
        fh = log.open("a")
        procs.append((slot, log, fh, subprocess.Popen(cmd, stdout=fh, stderr=fh, env=env)))
        logger.info("[fanout] worker %d -> GPU %s (pid %d)", slot, gid, procs[-1][3].pid)
    failed = []
    for slot, log, fh, proc in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            failed.append((slot, rc))
            # JSONL_SPLITLINES_EXEMPT: worker .log tail read (free-text log lines), not JSONL rows
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
            print(f"[fanout] worker {slot} FAILED rc={rc}; log tail:", flush=True)
            print("\n".join(tail), flush=True)
    assert not failed, f"worker subprocesses failed: {failed}"
    done = sorted((cfg.out_root / "logs").glob("worker*.done.json"))
    assert len(done) == n, (len(done), n, "missing worker done-sentinels")


def _fleet_deviations(cfg: Cfg, n_gpus: int) -> list[dict]:
    """Pilot-vs-plan projections from the workers' first-item timings."""
    pilots: dict[str, float] = {}
    for p in sorted((cfg.out_root / "logs").glob("worker*.done.json")):
        for k, v in json.loads(p.read_text())["pilots"].items():
            pilots[k] = min(v, pilots.get(k, float("inf")))
    rows = []
    if "p1a" in pilots:
        rows.append(
            _deviation(
                "P1a marker TF",
                0.3,
                pilots["p1a"] * 13 / max(n_gpus, 1) / 3600.0,
                f"first-pass pilot {pilots['p1a']:.0f}s (plan basis {BASIS_TF_PASS_S:.0f}s)",
            )
        )
    if "p1d" in pilots:
        rows.append(
            _deviation(
                "P1d refits",
                0.15,
                pilots["p1d"] * 33 / max(n_gpus, 1) / 3600.0,
                f"first-fit pilot {pilots['p1d']:.0f}s (plan basis {BASIS_RIDGE_FIT_S:.0f}s)",
            )
        )
    return rows


def run_ft_mix_probe(cfg: Cfg) -> dict:
    """Smoke structural probe: the FT-mapped mix loads at full grain (§12.1/§12.2).

    Gate = check_anchor_mix_floor: hard-fail only below 8 rows; 8 <= n < 40 is
    the measured 20-rows/content-mix regime at CORPUS_PIN — LOUD WARN, no kill.
    """
    arms_all = json.loads((cfg.out_root / "config" / "arms.json").read_text())["arms"]
    ft = next(a for a in arms_all if a["arm_id"] == FT_SMOKE_ARM)
    rows, meta = _mix_rows(cfg, ft["mix_arm_id"])
    low_n = check_anchor_mix_floor(ft["mix_arm_id"], len(rows))
    return {
        "ft_arm": FT_SMOKE_ARM,
        "mix_arm_id": ft["mix_arm_id"],
        "n_rows": len(rows),
        "low_n_flag": low_n,
        **meta,
    }


# ── fs-dryrun (crash-fix r7): CPU-only writer-path exercise ─────────────────


def _dryrun_entries() -> list[dict]:
    """Synthetic arm registry spanning every item class `build_items` emits.

    Fields limited to what the item-registry / path logic reads (arm_id,
    kind, method, beh_key, mix_arm_id, primary_layer, ctx_key) — the dryrun
    never loads models, stores, or the HF config. Includes P1C_ARM so the
    p1c items are emitted even with smoke=False (build_items gates p1c on
    its presence), plus a wmap sibling, a marker LoRA, a marker FT, and a
    content FT arm — every registry branch.
    """

    def arm(arm_id: str, kind: str, method: str, beh_key: str) -> dict:
        return {
            "arm_id": arm_id,
            "kind": kind,
            "method": method,
            "beh_key": beh_key,
            "mix_arm_id": f"{arm_id}-mix",
            "primary_layer": 19,
            "ctx_key": "pers",
        }

    return [
        arm(P1C_ARM, "content", "lora", "syc"),
        arm("dry-content-lora-b", "content", "lora", "syc"),
        arm("dry-marker-lora", "marker", "lora", "mk"),
        arm("dry-marker-ft", "marker", "ft", "mk"),
        arm("dry-content-ft", "content", "ft", "imp"),
    ]


def run_fs_dryrun(cfg: Cfg) -> dict:
    """CPU-only filesystem dry-run of EVERY writer path (crash-fix r7).

    Exercises the full main+worker directory-creation and writer-path logic
    with tiny stub payloads — no CUDA, no models, no network: creates the
    out-dirs exactly as main()/worker_main() do (``ensure_out_dirs``), then
    for a simulated 4-slot worker layout (the fellows job-16100 shape) writes
    a stub artifact at EVERY path the production run writes — via the SHARED
    path helpers (p1a_out_path .. p1f_out_dir; no duplicated composition) and
    the SAME write primitive + parent-dir guard behavior as each production
    writer (parquet via to_parquet, .pt via tmp+os.replace, the P1c JSONL via
    the job-16100 append-open, JSON via _atomic_json) — then replicates
    phase_upload's local enumeration arithmetic (no hub calls). Writes land
    under ``<out_root>/fs_dryrun/`` (never the production paths — a stub must
    not satisfy a resume predicate) and are removed on success; a failure
    leaves the scratch tree for forensics and exits nonzero. Exit 0 = every
    production writer path is constructible in a fresh process.
    """
    import shutil

    import pandas as pd
    import torch

    scratch = cfg.out_root / "fs_dryrun"
    if scratch.exists():
        shutil.rmtree(scratch)
    dcfg = dataclasses.replace(cfg, out_root=scratch, upload=False)
    entries = _dryrun_entries()
    arms_by_id = {a["arm_id"]: a for a in entries}
    items = build_items(dcfg, entries)
    phases_seen: dict[str, int] = {}
    tiny = pd.DataFrame({"sha": ["s0"], "v": [0.0]})

    def _stub_pt(out: Path) -> None:
        # mirrors run_p1b_* / _fit_and_persist: mkdir -> tmp in-dir -> os.replace
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".pt.tmp")
        torch.save({"fs_dryrun": True}, tmp)
        os.replace(tmp, out)

    # main-process sequence (mirrors main()): out_root mkdir + ensure_out_dirs
    dcfg.out_root.mkdir(parents=True, exist_ok=True)
    ensure_out_dirs(dcfg)
    # config mirror stubs (production: hub.stage_hub_file -> out_root/config/<name>,
    # which mkdirs target.parent itself — hub.py stage_hub_file)
    for name in ("subset.json", "arms.json", "margin_chain.json", "judge_filter.json"):
        _atomic_json(dcfg.out_root / "config" / name, {"fs_dryrun": True})
    # pre-fleet main-process writers (stage done / P1a gate / mirror parity)
    _atomic_json(dcfg.out_root / "p1_stage.done.json", {"fs_dryrun": True})
    _atomic_json(dcfg.out_root / "marker_tf" / "adapter_gate.json", {"fs_dryrun": True})
    _atomic_json(dcfg.out_root / "marker_tf" / "mirror_parity.json", {"fs_dryrun": True})
    # simulated worker fan-out
    n_slots = 4
    logdir = dcfg.out_root / "logs"
    logdir.mkdir(parents=True, exist_ok=True)  # mirrors _spawn_workers
    for slot in range(n_slots):
        wcfg = dataclasses.replace(dcfg, worker_slot=slot, n_slots=n_slots)
        ensure_out_dirs(wcfg)  # mirrors worker_main() entry (idempotent)
        with (logdir / f"worker{slot}.log").open("a") as fh:  # _spawn_workers log
            fh.write("[fs-dryrun]\n")
        for idx, item in enumerate(items):
            if idx % wcfg.n_slots != wcfg.worker_slot:
                continue
            ph = item["phase"]
            phases_seen[ph] = phases_seen.get(ph, 0) + 1
            if ph == "p1a":
                out = p1a_out_path(wcfg, item)
                out.parent.mkdir(parents=True, exist_ok=True)  # run_p1a_pass guard
                tiny.to_parquet(out, index=False)
                _atomic_json(out.with_suffix(".meta.json"), {"fs_dryrun": True})
            elif ph == "p1b_base":
                _stub_pt(p1b_base_out_path(wcfg, item["mix_arm_id"]))
            elif ph == "p1b_post":
                _stub_pt(p1b_post_out_path(wcfg, item["arm_id"]))
            elif ph == "p1c":
                out, done = p1c_out_paths(wcfg, item["side"])
                out.parent.mkdir(parents=True, exist_ok=True)  # run_p1c_side r7 guard
                if out.exists():  # resume-read branch parity
                    with out.open(encoding="utf-8") as fh:
                        fh.read()
                with out.open("a", encoding="utf-8") as fh:  # the job-16100 crash op
                    fh.write(json.dumps({"fs_dryrun": True}) + "\n")
                print(f"[fs-dryrun] p1c {item['side']}: validation/ append-open OK", flush=True)
                _atomic_json(done, {"fs_dryrun": True})
            elif ph == "p1d":
                pt, js = p1d_out_paths(wcfg, p1d_fit_name(item, arms_by_id))
                _stub_pt(pt)  # _fit_and_persist write shape
                _atomic_json(js, {"fs_dryrun": True})
            else:
                raise AssertionError(f"fs-dryrun: unstubbed item phase {ph}")
        _atomic_json(logdir / f"worker{slot}.done.json", {"fs_dryrun": True, "slot": slot})
    # post-fleet main-process writers (gate parity / P1e tables / P1f shards)
    _atomic_json(dcfg.out_root / "maps" / "gate_parity.json", {"fs_dryrun": True})
    for e in entries:
        for li in dcfg.table_layers():
            out = p1e_out_path(dcfg, e["arm_id"], li)
            out.parent.mkdir(parents=True, exist_ok=True)  # run_p1e_table guard
            tiny.to_parquet(out, index=False)
            _atomic_json(out.with_suffix(".meta.json"), {"fs_dryrun": True})
    units = [e["arm_id"] for e in entries if e["kind"] == "content"] + ["base_content"]
    for u in units:  # run_p1f_unit flush() + done-sentinel write shape
        p = p1f_out_dir(dcfg) / f"{u}.shard00.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)  # flush() guard
        p.write_text(json.dumps({"fs_dryrun": True}) + "\n", encoding="utf-8")
        _atomic_json(p1f_out_dir(dcfg) / f"{u}.done.json", {"fs_dryrun": True, "shards": [p.name]})
    _atomic_json(dcfg.out_root / "gpu_phase_summary.json", {"fs_dryrun": True})
    _atomic_json(dcfg.out_root / "p1_done.json", {"fs_dryrun": True})
    # phase_upload local-enumeration arithmetic (no hub calls)
    n_upload = 0
    for name in OUT_DIRS:
        local = dcfg.out_root / name
        if not local.exists():
            continue
        files = sorted(p for p in local.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
        prefix = f"{dcfg.hf_prefix}/{name}"
        n_upload += len([f"{prefix}/{p.relative_to(local)}" for p in files])
    required = {"p1a", "p1b_base", "p1b_post", "p1c", "p1d"}
    missing = required - set(phases_seen)
    assert not missing, f"fs-dryrun: item phases never exercised: {sorted(missing)}"
    assert n_upload > 0, "fs-dryrun: upload enumeration saw zero files"
    summary = {
        "n_slots": n_slots,
        "n_items": len(items),
        "phases": phases_seen,
        "n_upload_enumerated": n_upload,
        "n_paths_written": sum(1 for p in dcfg.out_root.rglob("*") if p.is_file()),
        "out_root": str(cfg.out_root),
        "smoke": cfg.smoke,
    }
    shutil.rmtree(scratch)
    print(f"[fs-dryrun] OK {json.dumps(summary)}", flush=True)
    return summary


# ── entrypoint ───────────────────────────────────────────────────────────────


def _run_import_check() -> None:
    """Execute every deferred import + signature-bind key callees (Axis 1)."""
    import inspect

    import pandas  # noqa: F401
    import torch  # noqa: F401
    from peft import PeftModel  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    import issue1768_capture as C
    import issue1768_directions as D
    import issue1768_fit as F
    import issue779_ffc_n1m_fits as n1m
    from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        _teacher_forced_span_means,
        compute_prompt_spans,
    )
    from explore_persona_space.eval.margin import build_fixed_pairs, compute_tf_margin
    from explore_persona_space.eval.marker_logprob import (  # noqa: F401
        assert_gauge_free_adapter_config,
        compute_marker_slot_stats,
        validate_marker_slot_record,
    )
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    for fn, kwargs in [
        (F._fit_map, {}),
        (F.load_corpus_cell, {}),
        (C._mix_positive_rows, {}),
        (D.load_rb_tensors, {}),
        (D.corpus_sigma, {}),
        (D.panel_write_legs, {}),
        (D.gate_read, {}),
        (n1m.apply_map, {}),
        (build_fixed_pairs, {}),
        (compute_tf_margin, {}),
        (hub.stage_hub_file, {}),
        (hub.stage_hub_prefix, {}),
        (hub.verify_repo_paths_uploaded, {}),
        (hub._upload, {}),
        (assert_out_root_headroom, {}),
    ]:
        sig = inspect.signature(fn)
        assert sig.parameters, (fn, "no parameters?")
    print("[import-check] OK — all deferred imports + signatures resolve", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "data/issue_1900/out")
    ap.add_argument("--stage-root", type=Path, default=REPO_ROOT / "data/issue_1900/hf_dl")
    ap.add_argument("--smoke", action="store_true", help="2 arms x 64 rows x own mixes")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--tf-batch", type=int, default=X.TF_BATCH_SIZE)
    ap.add_argument("--worker-slot", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--n-slots", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--import-check", action="store_true", help="resolve deferred imports; exit")
    ap.add_argument(
        "--stage-probe",
        action="store_true",
        help="(h)(iv) 1-file staging + consumer-open probe; CPU-runnable; exit 0",
    )
    ap.add_argument(
        "--fs-dryrun",
        action="store_true",
        help="CPU-only writer-path dry-run: every sub-phase's output paths + a "
        "simulated 4-worker slot layout, tiny stub payloads, no CUDA/models/network; "
        "exit 0 = every path constructible (crash-fix r7)",
    )
    args = ap.parse_args()
    if args.import_check:
        _run_import_check()
        sys.exit(0)
    cfg = Cfg(
        out_root=args.out_root.resolve(),
        stage_root=args.stage_root.resolve(),
        smoke=args.smoke,
        tf_batch=args.tf_batch,
        upload=not args.no_upload,
        worker_slot=args.worker_slot,
        n_slots=args.n_slots,
    )
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    if args.fs_dryrun:
        run_fs_dryrun(cfg)  # CPU-only writer-path dry-run (crash-fix r7); no CUDA/network
        sys.exit(0)
    if args.stage_probe:
        run_stage_probe(cfg)  # CPU-runnable (h)(iv) probe — VM pre-dispatch leg
        sys.exit(0)
    ensure_out_dirs(cfg)  # r7 process-level floor (parent AND worker pass through here)
    subset, arms_payload = load_run_config(cfg)
    if cfg.smoke:
        pass  # subset order deterministic; smoke row cap applied per unit

    if cfg.worker_slot is not None:
        worker_main(cfg, subset, arms_payload)
        sys.stdout.flush()
        sys.exit(0)

    t_start = time.time()
    run_stage_probe(cfg)  # (h)(iv) consumer-open seam gate BEFORE CUDA + heavy staging
    _device()  # preamble: fail loud with no CUDA (plan §9 preamble assert)
    entries = _arm_entries(cfg, arms_payload)
    arms_by_id = {a["arm_id"]: a for a in entries}
    phase_stage(cfg, subset, arms_payload)
    if cfg.smoke:
        probe = run_ft_mix_probe(cfg)
        logger.info("[smoke] FT-mix probe: %s", probe)
    pool = ModelPool(_device(), _dtype())
    gate = run_p1a_adapter_smoke(cfg, pool, arms_by_id)
    if cfg.smoke:
        run_mirror_parity(cfg, pool, arms_by_id)
    del pool
    _cuda_gc()
    _phase("p1_fleet")
    argv_tail = (
        [
            "--out-root",
            str(cfg.out_root),
            "--stage-root",
            str(cfg.stage_root),
            "--tf-batch",
            str(cfg.tf_batch),
        ]
        + (["--smoke"] if cfg.smoke else [])
        + (["--no-upload"] if not cfg.upload else [])
    )
    _spawn_workers(cfg, argv_tail)
    deviations = _fleet_deviations(cfg, len(_physical_gpu_ids()))
    parity = run_gate_parity(cfg)
    deviations.append(phase_tables(cfg, entries, subset))
    phase_judge_inputs(cfg, entries, subset)
    _atomic_json(
        cfg.out_root / "gpu_phase_summary.json",
        {
            "smoke": cfg.smoke,
            "n_arms": len(entries),
            "arm_ids": [a["arm_id"] for a in entries],
            "adapter_gate": gate,
            "gate_parity": parity,
            "compute_deviations": deviations,
            "wall_h": round((time.time() - t_start) / 3600.0, 3),
            **_meta(),
        },
    )
    phase_upload(cfg)
    _atomic_json(
        cfg.out_root / "p1_done.json",
        {"smoke": cfg.smoke, "n_arms": len(entries), **_meta()},
    )
    _phase("done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit: C-extension finalize rc race (gotchas PyGILState)


if __name__ == "__main__":
    main()
