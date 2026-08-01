"""#1768 capture driver — phases p0-p7 (plan v4 §4.4; capture + fit only, NO training).

Phases (all through this one entrypoint; ``--smoke`` = the pilot arm through
the SAME code paths, PASS_UNIFIED):

- p0_registry   arm registry + full 72-arm resolution probe (56 LoRA adapters
                + 16 full-FT overflow checkpoints, identity re-verified against
                the #1586 selection records) + corpus sample inputs
- p1_pilot      ONE arm end-to-end at production shape (gate 1 + fp16 probe)
- p2_corpus     74 corpus on-policy units (72 trained arms + 2 base decode
                variants; vLLM greedy gen -> TF span-means)
                + the 6 p6 GPU legs (`p6g:<arm>` rollout gen -> text persist ->
                judge SUBMIT via a detached CPU poller -> per-rollout
                activation persist) interleaved into the SAME queue (plan §9
                "p6 Batch-API wait placement" Must-Fix)
- p3_corpus_tf  72 matched-text units (trained model TF on the base tree rows)
- p4_panels     panel trees for the 40 non-pers arms (+ staged #1586 reuse —
                the 16 pers-LoRA AND 16 full-FT capture{,_tf} trees)
- p5_delta      56 base-model TF captures of training-mix positives (t_bar;
                ft arms SHARE the matched pers-LoRA cells' t̄ — same #1481
                mixes, no new cells; plan §4.1 amendment)
- p7_upload     HF data-repo upload of every out-root tree + upload_done.json
- p6_rb_reduce  CPU post-filter over the PERSISTED per-rollout activations
                (harvest judge scores -> threshold -> r_B); runs AFTER p7 by
                default so the Batch-API poll overlaps p2..p7 and no GPU
                phase ever blocks on the judge
- pnf           matched-text capture-noise floor (plan v7 follow-up; OPT-IN
                via `--phases pnf`, never in the default chain): 2 replicate
                TF captures x 6 units on a 2,000-row seed-42 TRAIN subsample
                -> CPU reduce (per-context ||v_r1 - v_r2||, p95 floor, 72-arm
                shift/floor ratio table, H3 + marker-falsification verdicts)
                -> noise_floor/ upload

Signaling: ``[phase=...]`` log lines + ``status.json`` heartbeat (SLURM-safe;
plan §9 pins NO /workspace sentinel dependence). Work fans out across every
visible GPU via CUDA_VISIBLE_DEVICES-pinned unit subprocesses (#1586 pattern).
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
import gc  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import issue1768_cells as X  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.capture")

SHARD_BYTE_BUDGET = 8_500_000  # raw-row shards stay <9 MB (non-LFS text path)
GEN_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
REUSED_PANEL_HF_PREFIX = "issue1586_methodgen/analysis_tensors"
MARKER_MIX_PREFIX = "issue1481_conpos_grid/marker"
CAPTURE_FN_NAMES = (
    "_generate_responses_vllm",
    "compute_prompt_spans",
    "_teacher_forced_span_means",
)
# Reviewed numeric-trivial drifts of the capture functions since the reused
# #1586 trees (plan §4.4 W2 "empty or numeric-trivial"): the #1610
# GENERATION_ROW_KEYS schema pin added a docstring note + a return-shape
# assert to _generate_responses_vllm — no numeric change. Keyed on the EXACT
# (old, new) function-text hash pair so any FUTURE drift re-fires RECAPTURE.
VINTAGE_TRIVIAL_PAIRS: frozenset[tuple[str, str, str]] = frozenset(
    {("_generate_responses_vllm", "5fb2a176965a", "6d727c54e03b")}  # #1610 schema pin
)


@dataclasses.dataclass
class Cfg:
    """Run configuration; every output-affecting knob is part of the regime."""

    out_root: Path
    phases: tuple[str, ...]
    smoke: bool = False
    arms: tuple[str, ...] = ()  # empty -> all 72 (smoke -> pilot arm only)
    layers: tuple[int, ...] = X.LAYERS
    tf_batch: int = X.TF_BATCH_SIZE
    model_override: str | None = None  # production-legal local snapshot override
    corpus_manifest_dir: Path | None = None  # local manifest dir (else Hub stage)
    valtest_file: Path | None = None  # recovered snapshot (else LMSYS re-derive)
    smoke_rows: tuple[int, int, int] = (24, 8, 8)  # train/val/test under --smoke
    upload: bool = True
    hf_prefix: str = X.HF_PREFIX  # smoke defaults to <prefix>_smoke (never clobbers)
    gpu_id: int = 0  # informational; the launcher env CVD pin selects the GPU

    def n_splits(self) -> tuple[int, int, int]:
        return self.smoke_rows if self.smoke else (X.N_TRAIN, X.N_VAL, X.N_TEST)


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


def _status(cfg: Cfg, phase: str, **extra) -> None:
    _atomic_json(
        cfg.out_root / "status.json",
        {"phase": phase, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra},
    )


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — metadata only, never kills a capture
        return "unknown"


def _meta() -> dict:
    import torch
    import transformers

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "issue": X.ISSUE,
    }


def _device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _dtype():
    import torch

    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


_FULL_ARM_INDEX: dict[str, X.Arm] | None = None


def _full_arm_index() -> dict[str, X.Arm]:
    """All 72 arms by id (cached — the manifest is immutable within a run)."""
    global _FULL_ARM_INDEX
    if _FULL_ARM_INDEX is None:
        _FULL_ARM_INDEX = {a.arm_id: a for a in X.all_arms()}
    return _FULL_ARM_INDEX


def _arm_index(cfg: Cfg) -> dict[str, X.Arm]:
    arms = X.all_arms()
    if cfg.smoke and not cfg.arms:
        arms = [a for a in arms if a.arm_id == X.PILOT_ARM]
    elif cfg.arms:
        want = set(cfg.arms)
        arms = [a for a in arms if a.arm_id in want]
        missing = want - {a.arm_id for a in arms}
        assert not missing, f"unknown arm ids: {sorted(missing)}"
    return {a.arm_id: a for a in arms}


# ── model resolution (merge → consume → delete; #1586 lifecycle) ─────────────


def _merge_adapter(cfg: Cfg, arm: X.Arm) -> Path:
    """Merge the arm's HF adapter onto base -> local merged dir (atomic publish).

    Complete-dir reuse: config.json presence == complete (a killed sibling's
    finished merge is consumed, not redone; a concurrent finisher's ENOTEMPTY
    on the publish rename resolves to reuse). bf16 merge is the fleet-standard
    read path for these arms (#1586 `_merge_adapter` convention).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = cfg.out_root / "merged" / arm.arm_id
    if (merged_dir / "config.json").exists():
        logger.info("[merge] %s: complete merged dir reused", arm.arm_id)
        return merged_dir
    sub = X.adapter_subfolder(arm)
    logger.info("[merge] %s <- %s/%s", arm.arm_id, X.HF_MODEL_REPO, sub)
    base = AutoModelForCausalLM.from_pretrained(
        X.BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    peft_model = PeftModel.from_pretrained(base, X.HF_MODEL_REPO, subfolder=sub)
    merged = peft_model.merge_and_unload()
    tmp = merged_dir.parent / f".tmp_{arm.arm_id}_{os.getpid()}"
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


def _ft_ckpt_incomplete_reason(d: Path) -> str | None:
    """None when a staged full-FT checkpoint dir carries the load-bearing set
    (config.json + EVERY index-listed weight shard, else >=1 .safetensors);
    else the reason string. A config-only partial passes bare presence checks
    and hands consumers a shard-less checkpoint (#1586 fu r7)."""
    if not (d / "config.json").exists():
        return "config.json missing"
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        shards = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        missing = [s for s in shards if not (d / s).exists()]
        if missing:
            return f"{len(missing)}/{len(shards)} weight shards missing (e.g. {missing[0]})"
    elif not list(d.glob("*.safetensors")):
        return "no weight shards"
    return None


def _ensure_ft_tokenizer(ckpt: Path) -> None:
    """Trainer checkpoints can lack tokenizer files (#1112) — repair from base."""
    if (ckpt / "tokenizer_config.json").exists():
        return
    from transformers import AutoTokenizer

    logger.info("[ft-stage] %s: tokenizer files absent — saving base tokenizer", ckpt)
    AutoTokenizer.from_pretrained(X.BASE_MODEL).save_pretrained(ckpt)


def _ft_ckpt_dirs(cfg: Cfg, arm: X.Arm) -> tuple[Path, Path]:
    """(cleanup_root, checkpoint_dir) for a ft arm's staged full checkpoint.

    `hub.stage_hub_prefix` writes a VERBATIM prefix mirror (files land at
    dest/<full repo path>), so the loadable checkpoint dir is
    `<root>/<overflow subfolder>` under the per-arm cleanup root.
    """
    root = cfg.out_root / "ft_ckpt" / arm.arm_id
    return root, root / X.ft_ckpt_subfolder(arm)


def _stage_ft_checkpoint(cfg: Cfg, arm: X.Arm) -> tuple[Path, Path]:
    """Stage a ft arm's selected FULL checkpoint from the overflow repo.

    Returns (cleanup_root, checkpoint_dir) — the ft sibling of
    `_merge_adapter`'s merge -> consume -> delete lifecycle (stage ->
    consume -> delete; the caller reaps the ~15 GB root at unit exit).
    Staging rides `hub.stage_hub_prefix` (#1402 canonical helper: scoped
    listing, one resolved revision, retried per-file downloads). An
    INCOMPLETE dir is removed before restage (#1586 fu r7: a partial dir
    must never satisfy the reuse predicate), completeness is re-verified
    fail-loud after staging, and missing tokenizer files are repaired from
    the base tokenizer (#1112).
    """
    from explore_persona_space.orchestrate import hub

    root, ckpt = _ft_ckpt_dirs(cfg, arm)
    if ckpt.exists():
        reason = _ft_ckpt_incomplete_reason(ckpt)
        if reason is None:
            logger.info("[ft-stage] %s: complete staged checkpoint reused", arm.arm_id)
            _ensure_ft_tokenizer(ckpt)
            return root, ckpt
        logger.info("[ft-stage] %s: removing incomplete %s (%s)", arm.arm_id, ckpt, reason)
        shutil.rmtree(root)  # fail-loud: no ignore_errors
    sub = X.ft_ckpt_subfolder(arm)
    logger.info("[ft-stage] %s <- %s/%s", arm.arm_id, X.FT_OVERFLOW_REPO, sub)
    hub.stage_hub_prefix(X.FT_OVERFLOW_REPO, sub, root, repo_type="model")
    reason = _ft_ckpt_incomplete_reason(ckpt)
    if reason is not None:
        raise RuntimeError(
            f"[ft-stage] {arm.arm_id}: staged checkpoint incomplete under {ckpt} ({reason})"
        )
    _ensure_ft_tokenizer(ckpt)
    return root, ckpt


def _resolve_unit_model(cfg: Cfg, unit_id: str) -> tuple[str, Path | None]:
    """(model_path, local_model_dir_to_cleanup) for one unit."""
    if cfg.model_override:
        return cfg.model_override, None
    if unit_id.startswith("base_"):
        return X.BASE_MODEL, None
    arm = _arm_index(cfg).get(unit_id) or _full_arm_index()[unit_id]
    if arm.method == "ft":  # full model: stage, never merge (plan §4.1 amendment)
        root, ckpt = _stage_ft_checkpoint(cfg, arm)
        return str(ckpt), root
    merged = _merge_adapter(cfg, arm)
    return str(merged), merged


def _cleanup_merged(cleanup: Path | None) -> None:
    if cleanup is not None:
        shutil.rmtree(cleanup, ignore_errors=True)


def _assert_model_dir_alive(model_path: str, cleanup: Path | None) -> None:
    """Fail LOUD (named) when a unit's resolved LOCAL model dir vanished
    mid-unit. A vanished shared dir means a same-`_model_key` sibling's
    exit-cleanup deleted it (the `_fanout_phase` co-scheduling guard exists to
    prevent exactly that); without this assert the failure surfaces as a
    misleading huggingface_hub HFValidationError treating the local path as a
    Hub repo id (job 16120). No-op for repo-id resolutions (cleanup None)."""
    if cleanup is not None and not Path(model_path).is_dir():
        raise RuntimeError(
            f"[shared-model-dir-vanished] {model_path} no longer exists mid-unit — "
            "a concurrent same-arm unit's exit-cleanup deleted the shared "
            "merged/staged model dir (merge -> consume -> delete lifecycle); the "
            "_fanout_phase _model_key guard must keep same-key units from "
            "running concurrently"
        )


# ── p0: registry + adapter probe + corpus inputs ─────────────────────────────


def _probe_adapters(arms: list[X.Arm]) -> list[dict]:
    """Full 56-adapter `file_exists` probe (LoRA arms); FAIL LOUD on any miss (#503)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    rows, misses = [], []
    for a in arms:
        sub = X.adapter_subfolder(a)
        path = f"{sub}/adapter_config.json"
        ok = hub.retry_transient(  # HUB_VERIFY_RETRY_EXEMPT: wrapped in retry_transient here
            lambda p=path: api.file_exists(X.HF_MODEL_REPO, p, repo_type="model"),
            what=f"adapter probe {path}",
        )
        rows.append({**dataclasses.asdict(a), "subfolder": sub, "adapter_resolves": bool(ok)})
        if not ok:
            misses.append(f"{a.arm_id} -> {sub}")
    if misses:
        raise RuntimeError(
            f"p0 adapter probe: {len(misses)}/{len(arms)} checkpoints missing on "
            f"{X.HF_MODEL_REPO}:\n  " + "\n  ".join(misses)
        )
    return rows


def _probe_ft_checkpoints(arms: list[X.Arm]) -> list[dict]:
    """p0 identity + resolution probe for the 16 full-FT arms (plan §4.1
    amendment). Identity = the #1586 `selection/` Hub records: each record's
    selected step must MATCH the code pin, and the selected checkpoint's
    config.json must resolve on the overflow repo. FAIL LOUD listing every
    miss / drift (#503)."""
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    rows, misses = [], []
    for a in arms:
        assert a.method == "ft", a.arm_id
        sel_path = f"{X.FT_SELECTION_HF_PREFIX}/{a.arm_id}/selection.json"
        local = hub.retry_transient(
            lambda p=sel_path: hf_hub_download(X.HF_DATA_REPO, p, repo_type="dataset"),
            what=f"ft selection fetch {sel_path}",
        )
        sel = json.loads(Path(local).read_text())
        if int(sel["step"]) != a.step:
            misses.append(f"{a.arm_id}: selection step {sel['step']} != pinned {a.step}")
            continue
        sub = X.ft_ckpt_subfolder(a)
        ok = hub.retry_transient(  # HUB_VERIFY_RETRY_EXEMPT: wrapped in retry_transient here
            lambda p=f"{sub}/config.json": api.file_exists(
                X.FT_OVERFLOW_REPO, p, repo_type="model"
            ),
            what=f"ft ckpt probe {sub}",
        )
        rows.append(
            {
                **dataclasses.asdict(a),
                "subfolder": sub,
                "ckpt_repo": X.FT_OVERFLOW_REPO,
                "adapter_resolves": bool(ok),  # shared registry column name
                "selection_in_band": sel.get("in_band"),
                "selection_fallback": sel.get("fallback"),
                "selection_metric": sel.get("metric"),
            }
        )
        if not ok:
            misses.append(f"{a.arm_id} -> {X.FT_OVERFLOW_REPO}/{sub}")
    if misses:
        raise RuntimeError(
            f"p0 ft-checkpoint probe: {len(misses)}/{len(arms)} selected full-FT "
            f"checkpoints missing/drifted:\n  " + "\n  ".join(misses)
        )
    return rows


def _assert_marker_gauge(arms: list[X.Arm]) -> dict:
    """W_U frozen by construction: one marker adapter_config excludes lm_head.

    LoRA arms only — a full-FT marker arm TRAINS W_U (no gauge freeze); its
    Q4 read keeps the BASE W_U row as a fixed reference, annotated in p9.
    """
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    mk = next(a for a in arms if a.kind == "marker" and a.method == "lora")
    sub = X.adapter_subfolder(mk)
    cfg_path = hub.retry_transient(
        lambda: hf_hub_download(X.HF_MODEL_REPO, f"{sub}/adapter_config.json", repo_type="model"),
        what="marker adapter_config fetch",
    )
    cfg = json.loads(Path(cfg_path).read_text())
    tmods = set(cfg.get("target_modules") or [])
    bad = {m for m in tmods if "lm_head" in m or "embed" in m}
    assert not bad, f"marker adapter {sub} touches unembedding modules: {sorted(bad)}"
    assert not cfg.get("modules_to_save"), cfg.get("modules_to_save")
    return {"arm_id": mk.arm_id, "target_modules": sorted(tmods)}


def _mix_pos_source(arm: X.Arm) -> dict:
    """Candidate positives paths per arm (plan §4.4 p5; Hub-probed at p0).

    The positives pool resolves from the CON family for BOTH regimes: the po
    mixes are row-provenance-DERIVED from the con positives (issue1481
    phase-0) and carry NO pos sidecar on the Hub (verified 2026-07-28 —
    `po_mixes/<cell>/mix/` holds only train_mix.jsonl + mix_meta.json), so
    the con-family `datagen/pos.jsonl` (or the c3 family's `datagen_topup/`
    sidecar — the #1481 second-family layout) IS the shared positives pool.
    Recorded as `pos_source_regime` in the registry.
    """
    if arm.kind == "marker":
        return {
            "candidates": [f"{MARKER_MIX_PREFIX}/mixes/marker_{arm.ctx_key}_{arm.regime}.jsonl"],
            "layout": "marker-mix",
            "mix_prefix": MARKER_MIX_PREFIX,
            "pos_source_regime": arm.regime,
        }
    import issue1481_cells as c1481

    prefix, layout = c1481.mix_for(arm.beh_key, arm.ctx_key, "con")
    parent = prefix.rsplit("/mix", 1)[0] if prefix.endswith("/mix") else prefix
    cands = list(
        dict.fromkeys(
            [
                f"{parent}/datagen/pos.jsonl",
                f"{parent}/datagen_topup/pos.jsonl",
                f"{prefix}/datagen/pos.jsonl",
                f"{prefix}/pos.jsonl",
            ]
        )
    )
    return {
        "candidates": cands,
        "layout": layout,
        "mix_prefix": prefix,
        "pos_source_regime": "con",
    }


def _probe_mix_sources(arms: list[X.Arm]) -> dict[str, dict]:
    """Resolve + Hub-probe every arm's positives file; FAIL LOUD on misses."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()

    def exists(path: str) -> bool:
        return hub.retry_transient(  # HUB_VERIFY_RETRY_EXEMPT: wrapped in retry_transient here
            lambda: api.file_exists(X.HF_DATA_REPO, path, repo_type="dataset"),
            what=f"mix probe {path}",
        )

    out: dict[str, dict] = {}
    misses = []
    cache: dict[str, dict] = {}
    for a in arms:
        src = _mix_pos_source(a)
        cands = src.pop("candidates")
        key = "|".join(cands)
        if key in cache:
            out[a.arm_id] = cache[key]
            continue
        resolved = next((c for c in cands if exists(c)), None)
        if resolved is None:
            misses.append(f"{a.arm_id}: none of {cands}")
            continue
        rec = {**src, "pos_path": resolved}
        cache[key] = rec
        out[a.arm_id] = rec
    if misses:
        raise RuntimeError(
            "p0 mix-positives probe misses (plan §4.4 p5 inputs):\n  " + "\n  ".join(misses)
        )
    return out


def _load_valtest(cfg: Cfg) -> list[str]:
    cached = cfg.out_root / "inputs" / "valtest_prompts.json"
    if cfg.valtest_file is not None:
        vt = json.loads(Path(cfg.valtest_file).read_text())["prompts"]
    elif cached.exists():
        vt = json.loads(cached.read_text())["prompts"]
    else:
        vt = X.recover_valtest_prompts()
    n_val, n_test = cfg.n_splits()[1], cfg.n_splits()[2]
    assert len(vt) >= n_val + n_test, (len(vt), n_val + n_test)
    return vt


def _build_corpus_inputs(cfg: Cfg) -> dict:
    """Stage the #779 n1M manifest, recover val/test, sample train (plan §4.2)."""
    import issue779_ffc_n1m_generate_capture as n1g
    from transformers import AutoTokenizer

    inputs = cfg.out_root / "inputs"
    sample_path = inputs / "corpus_sample.json"
    if sample_path.exists():
        return json.loads(sample_path.read_text())

    pins = X.assert_pinned_split()  # deterministic fixed_split sha check (assumption 5)
    man_dir = (
        Path(cfg.corpus_manifest_dir) if cfg.corpus_manifest_dir else inputs / "sampling_manifest"
    )
    if not (man_dir / "meta.json").exists():
        n1g._download_manifest(X.MANIFEST_HF_PREFIX, man_dir)
    pool, meta = n1g.read_manifest_pool(man_dir)

    valtest = _load_valtest(cfg)
    n_train, n_val, n_test = cfg.n_splits()
    val_prompts, test_prompts = valtest[:n_val], valtest[n_val : n_val + n_test]

    tok = AutoTokenizer.from_pretrained(cfg.model_override or X.BASE_MODEL)
    train = X.sample_train_prompts(pool, meta, tok, valtest, n_train=n_train, seed=X.SAMPLE_SEED)
    vt_rows = [{"prompt": p, "corpus": "valtest", "sha": X.prompt_sha(p)} for p in val_prompts] + [
        {"prompt": p, "corpus": "valtest", "sha": X.prompt_sha(p)} for p in test_prompts
    ]
    rows = train["rows"] + vt_rows
    # #1768 crash-fix r4 postcondition: the sampler's taken-set dedup guarantees
    # train shas are UNIQUE and DISJOINT from the pinned val/test shas — a
    # violation here is a genuine logic bug. Duplicate shas WITHIN the pinned
    # val/test set are a FROZEN property of the #779 split (round1 phase 1 never
    # exact-deduped; measured 1,400 -> 1,318 unique) — recorded, never "fixed"
    # (dropping pinned rows would break the exact counts + pinned recovery).
    train_shas = {r["sha"] for r in train["rows"]}
    vt_shas = {r["sha"] for r in vt_rows}
    n_vt_dup = len(vt_rows) - len(vt_shas)
    logger.info(
        "[p0] dedup: dropped %d duplicate-sha rows during train sampling, topped up to %d "
        "(valtest internal duplicate shas: %d)",
        train["n_skipped_dup"],
        n_train,
        n_vt_dup,
    )
    assert len(train_shas) == n_train == len(train["rows"]), "train sha dedup failed (logic bug)"
    assert not (train_shas & vt_shas), "train/valtest sha overlap after dedup (logic bug)"
    assert len(rows) == n_train + n_val + n_test, (len(rows), (n_train, n_val, n_test))
    sample = {
        "rows": rows,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_skipped_over_cap": train["n_skipped_over_cap"],
        "n_skipped_valtest": train["n_skipped_valtest"],
        "n_skipped_dup": train["n_skipped_dup"],
        "n_valtest_dup_shas": n_vt_dup,
        "proportions": train["proportions"],
        "token_cap": train["token_cap"],
        "sample_seed": X.SAMPLE_SEED,
        "split_pins": pins,
        "smoke": cfg.smoke,
        **_meta(),
    }
    _atomic_json(sample_path, sample)
    _atomic_json(inputs / "valtest_prompts.json", {"prompts": valtest, **_meta()})
    return sample


def phase_p0(cfg: Cfg) -> None:
    _phase("p0_registry")
    _status(cfg, "p0_registry")
    arms = list(_arm_index(cfg).values())
    full = X.all_arms()  # ALWAYS the full 72-arm probe (plan §4.1 + amendment)
    lora = [a for a in full if a.method == "lora"]
    fts = [a for a in full if a.method == "ft"]
    rows = _probe_adapters(lora) + _probe_ft_checkpoints(fts)
    gauge = _assert_marker_gauge(lora)
    # ft arms SHARE the pers-LoRA cells' mixes/t̄ (delta_arm_for) — probing the
    # 56 LoRA arms covers every p5 delta cell (plan §4.1 amendment).
    mixes = _probe_mix_sources(lora)
    _atomic_json(
        cfg.out_root / "arm_registry.json",
        {
            "arms": rows,
            "in_scope": [a.arm_id for a in arms],
            "marker_gauge": gauge,
            "mix_pos_sources": mixes,
            **_meta(),
        },
    )
    _build_corpus_inputs(cfg)
    logger.info("[p0] registry + corpus inputs ready (%d arms in scope)", len(arms))


# ── p2: corpus on-policy units ───────────────────────────────────────────────


def _read_shards(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for shard in sorted(out_dir.glob("raw_rows_*.jsonl")):
        with shard.open(encoding="utf-8") as fh:  # text-mode iteration, never splitlines
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _append_shard(out_dir: Path, rows: list[dict]) -> None:
    """Byte-budgeted (<9 MB) shard append, atomic per shard (tmp + replace)."""
    idx = len(list(out_dir.glob("raw_rows_*.jsonl")))
    buf: list[str] = []
    nbytes = 0

    def flush() -> None:
        nonlocal idx, buf, nbytes
        if not buf:
            return
        path = out_dir / f"raw_rows_{idx:04d}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(buf), encoding="utf-8")
        os.replace(tmp, path)
        idx += 1
        buf, nbytes = [], 0

    for r in rows:
        line = json.dumps(r, ensure_ascii=False) + "\n"
        b = len(line.encode("utf-8"))
        if buf and nbytes + b > SHARD_BYTE_BUDGET:
            flush()
        buf.append(line)
        nbytes += b
    flush()


def _generate_rows_vllm(
    cfg: Cfg,
    unit_id: str,
    model_path: str,
    prompts: list[str],
    start_idx: int,
    out_dir: Path,
    *,
    system: str | None = None,
    user_wrap: str | None = None,
    prior_turns: tuple = (),
    max_model_len: int = X.MAX_MODEL_LEN,
) -> None:
    """Chunked vLLM greedy generation (the #664 chunk rule) with shard persist.

    Rows follow the `_generate_responses_vllm` schema (+ sha/text extras); the
    engine is built ONCE, chunked `generate` calls carry `use_tqdm=False`
    (#613), prefix caching is OFF by default for this real-user corpus
    (gotchas.md pre-launch checklist; EPM_VLLM_DISABLE_PREFIX_CACHING=0
    re-enables), and teardown rides `_reap_vllm_engine`. The pfx kwargs
    (system / user_wrap / prior_turns / max_model_len) default to the round-1
    bare render byte-identically; prefixed units thread the arm's trained
    context through `_build_generation_prompts` — the ONE message-construction
    path span computation re-derives (#1112).
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from explore_persona_space.analysis.representation_shift import (
        _build_generation_prompts,
        _reap_vllm_engine,
        _vllm_enforce_eager,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    max_new = X.max_new_tokens_for(unit_id)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=X.GEN_GPU_MEM_UTIL,
        enforce_eager=_vllm_enforce_eager(),
        max_model_len=max_model_len,
        enable_prefix_caching=os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "1") == "0",
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_new)
    eos_id = tokenizer.eos_token_id
    try:
        n_chunks = -(-len(prompts) // GEN_CHUNK)
        for ci, s in enumerate(range(0, len(prompts), GEN_CHUNK)):
            chunk = prompts[s : s + GEN_CHUNK]
            rendered, keys = _build_generation_prompts(
                tokenizer,
                {unit_id: system},
                chunk,
                user_wraps={unit_id: user_wrap},
                prior_turns={unit_id: tuple(prior_turns)},
            )
            t0 = time.time()
            outs = llm.generate(rendered, params, use_tqdm=False)
            rows = []
            for (p_name, q_idx), out in zip(keys, outs, strict=True):
                comp = out.outputs[0]
                resp_ids = list(comp.token_ids)
                if resp_ids and resp_ids[-1] == eos_id:
                    resp_ids = resp_ids[:-1]
                rows.append(
                    {
                        "persona": p_name,
                        "question_idx": start_idx + s + q_idx,
                        "prompt_sha": X.prompt_sha(chunk[q_idx]),
                        "prompt_token_ids": list(out.prompt_token_ids),
                        "response_token_ids": resp_ids,
                        "finish_reason": comp.finish_reason,
                        "response_text": tokenizer.decode(resp_ids),
                    }
                )
            _append_shard(out_dir, rows)
            logger.info(
                "[vllm-chunk] %s chunk %d/%d (%d prompts) elapsed=%.1fs",
                unit_id,
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


def _load_or_generate_rows(
    cfg: Cfg,
    out_dir: Path,
    unit_id: str,
    model_path: str,
    *,
    prompts: list[str] | None = None,
    system: str | None = None,
    user_wrap: str | None = None,
    prior_turns: tuple = (),
    max_model_len: int = X.MAX_MODEL_LEN,
) -> list[dict]:
    if prompts is None:  # round-1 default: the full 16.4k-row sample
        sample = X.load_corpus_sample(cfg.out_root)
        prompts = [r["prompt"] for r in sample["rows"]]
    done = out_dir / "raw_rows.done.json"
    if done.exists():
        rows = _read_shards(out_dir)
        assert len(rows) == json.loads(done.read_text())["n_rows"], (unit_id, len(rows))
        return rows
    existing = _read_shards(out_dir)
    if existing:
        logger.info("[gen] %s resuming at row %d/%d", unit_id, len(existing), len(prompts))
        for i, r in enumerate(existing):  # deterministic greedy: order == prompt order
            assert r["question_idx"] == i, (unit_id, i, r["question_idx"])
    remaining = prompts[len(existing) :]
    if remaining:
        _generate_rows_vllm(
            cfg,
            unit_id,
            model_path,
            remaining,
            len(existing),
            out_dir,
            system=system,
            user_wrap=user_wrap,
            prior_turns=prior_turns,
            max_model_len=max_model_len,
        )
    rows = _read_shards(out_dir)
    assert len(rows) == len(prompts), (unit_id, len(rows), len(prompts))
    _atomic_json(done, {"n_rows": len(rows), **_meta()})
    return rows


def _attach_spans(
    tokenizer,
    prompts: list[str],
    rows: list[dict],
    *,
    system: str | None = None,
    user_wrap: str | None = None,
    prior_turns: tuple = (),
) -> tuple[list[dict], int, dict, int]:
    """Compute prefix/context spans per row; drop invalid rows with counts.

    Returns (kept_rows, n_dropped, seam_counts, n_distinct_prefix). The pfx
    kwargs re-derive the SAME message construction generation used
    (`_build_generation_prompts` — one construction path, #1112); defaults are
    the round-1 bare render byte-identically.
    """
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    kept: list[dict] = []
    seam_counts = {"prefix": 0, "context": 0}
    prefixes: set[tuple[int, ...]] = set()
    dropped = 0
    for r in rows:
        if not r["response_token_ids"]:
            dropped += 1
            continue
        flags: dict[str, bool] = {}
        r["prefix_len"], r["context_len"] = compute_prompt_spans(
            tokenizer,
            system,
            prompts[r["question_idx"]],
            r["prompt_token_ids"],
            prior_messages=list(prior_turns) or None,
            user_wrap=user_wrap,
            prefix_end="last_user",
            on_seam="snap",
            seam_flags=flags,
        )
        seam_counts["prefix"] += int(flags["prefix"])
        seam_counts["context"] += int(flags["context"])
        prefixes.add(tuple(r["prompt_token_ids"][: r["prefix_len"]]))
        kept.append(r)
    return kept, dropped, seam_counts, len(prefixes)


def _fp16_roundtrip_cos_min(pooled: dict) -> float:
    """fp32 pooled vs fp16 storage cast, min cosine over rows (assumption 11)."""
    import torch

    cos_min = 1.0
    for per_layer in pooled.values():
        for t in per_layer.values():
            cos = torch.nn.functional.cosine_similarity(
                t.float(), t.to(torch.float16).float(), dim=1
            )
            cos_min = min(cos_min, float(cos.min()))
    return cos_min


def _save_pooled(path: Path, unit_id: str, pooled: dict, kept: list[dict], extra: dict) -> None:
    import torch

    store = {
        "schema_version": 1,
        "unit": unit_id,
        "row_sha": [r["prompt_sha"] for r in kept],
        "row_question_idx": [r["question_idx"] for r in kept],
        "arms": {
            span: {li: t.to(torch.float16) for li, t in per_layer.items()}
            for span, per_layer in pooled.items()
        },
        "metadata": {**_meta(), **extra},
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(store, tmp)
    os.replace(tmp, path)


def run_corpus_unit(cfg: Cfg, unit_id: str) -> None:
    """p2 unit: greedy gen over the corpus prompts -> TF span-means (ctx+resp)."""
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    out_dir = cfg.out_root / "corpus_capture" / unit_id
    if (out_dir / "pooled.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = X.load_corpus_sample(cfg.out_root)
    prompts = [r["prompt"] for r in sample["rows"]]
    model_path, cleanup = _resolve_unit_model(cfg, unit_id)
    try:
        rows = _load_or_generate_rows(cfg, out_dir, unit_id, model_path)
        _assert_model_dir_alive(model_path, cleanup)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        kept, dropped, seam_counts, n_prefix = _attach_spans(tokenizer, prompts, rows)
        valid_frac = len(kept) / max(1, len(rows))
        assert valid_frac >= 0.95, (unit_id, valid_frac, "plan §6 row-validity criterion")
        _atomic_json(
            out_dir / "rows_spans.json",
            {
                "rows": [
                    {
                        "prompt_sha": r["prompt_sha"],
                        "question_idx": r["question_idx"],
                        "prefix_len": r["prefix_len"],
                        "context_len": r["context_len"],
                    }
                    for r in kept
                ],
                "n_dropped_empty_response": dropped,
                "seam_counts": seam_counts,
                "n_distinct_prefix": n_prefix,
            },
        )
        _assert_model_dir_alive(model_path, cleanup)
        pooled = _teacher_forced_span_means(
            model_path,
            kept,
            [unit_id],
            layers=list(cfg.layers),
            spans=("context", "response"),
            device=_device(),
            dtype=_dtype(),
            tf_batch_size=cfg.tf_batch,
        )
        fp16_cos = _fp16_roundtrip_cos_min(pooled)
        _save_pooled(
            out_dir / "pooled.pt",
            unit_id,
            pooled,
            kept,
            {
                "model_path": model_path,
                "layers": list(cfg.layers),
                "spans": ["context", "response"],
                "max_new_tokens": X.max_new_tokens_for(unit_id),
                "n_rows": len(kept),
                "n_dropped": dropped,
                "seam_counts": seam_counts,
                "n_distinct_prefix": n_prefix,
                "fp16_roundtrip_cos_min": fp16_cos,
                "smoke": cfg.smoke,
            },
        )
        _atomic_json(
            out_dir / "manifest.json",
            {
                "unit": unit_id,
                "n_rows": len(kept),
                "valid_frac": valid_frac,
                "model_path": model_path,
                "n_distinct_prefix": n_prefix,
                "fp16_roundtrip_cos_min": fp16_cos,
                **_meta(),
            },
        )
    finally:
        _cleanup_merged(cleanup)


def _read_rows_with_spans(base_dir: Path) -> list[dict]:
    """Base-tree rows re-joined with their persisted spans (sha-keyed)."""
    rows = _read_shards(base_dir)
    spans = json.loads((base_dir / "rows_spans.json").read_text())["rows"]
    by_key = {(s["prompt_sha"], s["question_idx"]): s for s in spans}
    out = []
    for r in rows:
        s = by_key.get((r["prompt_sha"], r["question_idx"]))
        if s is None:  # dropped at span time (empty response) — stays dropped
            continue
        r["prefix_len"], r["context_len"] = s["prefix_len"], s["context_len"]
        out.append(r)
    assert len(out) == len(spans), (base_dir, len(out), len(spans))
    return out


def run_corpus_tf_unit(cfg: Cfg, arm_id: str) -> None:
    """p3 unit: trained model teacher-forced on the BASE tree's rows (#833)."""
    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    out_dir = cfg.out_root / "corpus_capture_tf" / arm_id
    if (out_dir / "pooled_tf.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    base_unit = X.base_unit_for(arm_id)
    base_dir = cfg.out_root / "corpus_capture" / base_unit
    assert (base_dir / "pooled.pt").exists(), f"p3 {arm_id}: base unit {base_unit} not captured"
    rows = _read_rows_with_spans(base_dir)
    model_path, cleanup = _resolve_unit_model(cfg, arm_id)
    try:
        _assert_model_dir_alive(model_path, cleanup)
        pooled = _teacher_forced_span_means(
            model_path,
            rows,
            [base_unit],
            layers=list(cfg.layers),
            spans=("response",),
            device=_device(),
            dtype=_dtype(),
            tf_batch_size=cfg.tf_batch,
        )
        store_path = out_dir / "pooled_tf.pt"
        import torch

        store = {
            "schema_version": 1,
            "unit": arm_id,
            "row_sha": [r["prompt_sha"] for r in rows],
            "row_question_idx": [r["question_idx"] for r in rows],
            "arms": {
                span: {li: t.to(torch.float16) for li, t in per.items()}
                for span, per in pooled.items()
            },
            "metadata": {
                **_meta(),
                "model_path": model_path,
                "layers": list(cfg.layers),
                "spans": ["response"],
                "shared_text_from": base_unit,
                "n_rows": len(rows),
                "smoke": cfg.smoke,
            },
        }
        tmp = store_path.with_suffix(".pt.tmp")
        torch.save(store, tmp)
        os.replace(tmp, store_path)
        _atomic_json(
            out_dir / "manifest.json",
            {"unit": arm_id, "n_rows": len(rows), "model_path": model_path, **_meta()},
        )
    finally:
        _cleanup_merged(cleanup)


# ── p4: panel trees (fresh capture for non-pers arms + staged #1586 reuse) ───


def _d1586():
    import issue1586_dispatch as d1586

    return d1586


def _d1586_cfg(cfg: Cfg):
    d = _d1586()
    return d.Cfg(
        smoke=cfg.smoke,
        cells=(),
        out_root=cfg.out_root,
        upload=False,
        eval_question_limit=2 if cfg.smoke else None,
    )


def _stage_marker_icl_bank(cfg: Cfg) -> None:
    """Point-of-use stage of the #1481 marker ICL bank (mk panel contexts
    require it; produced by #1481's mixes phase, published under its marker
    inputs prefix — Hub-verified 2026-07-28)."""
    dest = cfg.out_root / "inputs" / "icl_examples_marker.json"
    if dest.exists():
        return
    from explore_persona_space.orchestrate import hub

    hub.stage_hub_file(
        X.HF_DATA_REPO,
        f"{MARKER_MIX_PREFIX}/inputs/icl_examples_marker.json",
        dest,
        repo_type="dataset",
    )


def _panel_setup(cfg: Cfg, beh_key: str):
    """(ctx_ids, panel, questions, personas, user_wraps, prior_turns)."""
    d = _d1586()
    dcfg = _d1586_cfg(cfg)
    if beh_key == "mk":
        _stage_marker_icl_bank(cfg)
    ctx_ids = d.panel_context_ids(dcfg, beh_key)
    questions = d._eval_questions(dcfg, beh_key)
    if cfg.smoke:  # >=2 contexts x >=2 questions (floors: fit_cell n>=4)
        ctx_ids = ctx_ids[:2]
        questions = questions[:2]
    panel = {cid: d.CONTEXTS[cid] for cid in ctx_ids}
    personas = {cid: c.system for cid, c in panel.items()}
    user_wraps = {cid: c.user_wrap for cid, c in panel.items()}
    prior_turns = {cid: tuple(dict(t) for t in c.prefix_turns) for cid, c in panel.items()}
    return ctx_ids, panel, questions, personas, user_wraps, prior_turns


def _panel_layers(cfg: Cfg) -> list[int]:
    if cfg.model_override:  # tiny-model smoke: capture the model's real depth
        return list(cfg.layers)
    return list(range(X.N_LAYERS_FULL))  # parity with the reused #1586 trees


def _panel_own_capture(cfg: Cfg, unit_id: str, model_path: str) -> None:
    """Own-text panel capture body (mirrors issue1586 run_capture_unit; 3 spans)."""
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import (
        _generate_responses_vllm,
        _teacher_forced_span_means,
        compute_prompt_spans,
    )

    out_dir = cfg.out_root / "panel_capture" / unit_id
    if (out_dir / "pooled.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    beh_key = unit_id.removeprefix("base_").split("-")[0]
    _ctx_ids, panel, questions, personas, user_wraps, prior_turns = _panel_setup(cfg, beh_key)
    rows = _generate_responses_vllm(
        model_path,
        personas,
        questions,
        max_new_tokens=X.MAX_NEW_MARKER if beh_key == "mk" else X.MAX_NEW_CONTENT,
        gpu_memory_utilization=X.GEN_GPU_MEM_UTIL,
        user_wraps=user_wraps,
        prior_turns=prior_turns,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_override or X.BASE_MODEL)
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
        seam_counts["prefix"] += int(flags["prefix"])
        seam_counts["context"] += int(flags["context"])
    (out_dir / "raw_rows.json").write_text(  # rollout text BEFORE the reduce
        json.dumps(
            {"model": model_path, "span_seam_counts": seam_counts, "rows": rows},
            ensure_ascii=False,
        )
    )
    pooled = _teacher_forced_span_means(
        model_path,
        rows,
        list(panel),
        layers=_panel_layers(cfg),
        device=_device(),
        dtype=_dtype(),
        tf_batch_size=cfg.tf_batch,
    )
    import torch

    store = {
        "schema_version": 1,
        "cell": unit_id,
        "dose": "base" if unit_id.startswith("base_") else "selected",
        "row_meta": [{"context_id": r["persona"], "question_idx": r["question_idx"]} for r in rows],
        "arms": {
            span: {li: t.to(torch.float16) for li, t in per.items()} for span, per in pooled.items()
        },
        "metadata": {**_meta(), "model_path": model_path, "seam_counts": seam_counts},
    }
    tmp = out_dir / "pooled.pt.tmp"
    torch.save(store, tmp)
    os.replace(tmp, out_dir / "pooled.pt")
    _atomic_json(
        out_dir / "manifest.json",
        {"cell": unit_id, "n_rows": len(rows), "model_path": model_path, **_meta()},
    )


def _panel_tf_capture(cfg: Cfg, arm_id: str, model_path: str) -> None:
    """Matched-text panel tree body (mirrors issue1586 run_capture_tf_unit)."""
    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    out_dir = cfg.out_root / "panel_capture_tf" / arm_id
    if (out_dir / "pooled.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    beh_key = arm_id.split("-")[0]
    base_raw = cfg.out_root / "panel_capture" / f"base_{beh_key}" / "raw_rows.json"
    rows = json.loads(base_raw.read_text(encoding="utf-8"))["rows"]
    pooled = _teacher_forced_span_means(
        model_path,
        rows,
        sorted({r["persona"] for r in rows}),
        layers=_panel_layers(cfg),
        device=_device(),
        dtype=_dtype(),
        tf_batch_size=cfg.tf_batch,
    )
    import torch

    store = {
        "schema_version": 1,
        "cell": arm_id,
        "kind": "tf_shared",
        "row_meta": [{"context_id": r["persona"], "question_idx": r["question_idx"]} for r in rows],
        "arms": {
            span: {li: t.to(torch.float16) for li, t in per.items()} for span, per in pooled.items()
        },
        "metadata": {**_meta(), "model_path": model_path, "shared_text": True},
    }
    tmp = out_dir / "pooled.pt.tmp"
    torch.save(store, tmp)
    os.replace(tmp, out_dir / "pooled.pt")


def _vintage_ok(manifest_commit: str) -> bool:
    """W2 guard: the three capture functions unchanged since the reused tree."""
    module = "src/explore_persona_space/analysis/representation_shift.py"

    def _fn_sources(ref: str) -> dict[str, str]:
        src = subprocess.run(
            ["git", "show", f"{ref}:{module}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        out = {}
        lines = src.split("\n")
        for name in CAPTURE_FN_NAMES:
            starts = [i for i, line in enumerate(lines) if line.startswith(f"def {name}(")]
            assert starts, (ref, name)
            i = starts[0]
            j = i + 1
            while j < len(lines) and (not lines[j] or not lines[j][0].isalpha()):
                j += 1
            out[name] = "\n".join(lines[i:j])
        return out

    try:
        old = _fn_sources(manifest_commit)
    except (subprocess.CalledProcessError, AssertionError) as e:
        logger.warning(
            "[p4-vintage] cannot read %s at %s (%s) — RECAPTURE", module, manifest_commit, e
        )
        return False
    new = _fn_sources("HEAD")

    def sha12(text: str) -> str:
        import hashlib

        return hashlib.sha256(text.encode()).hexdigest()[:12]

    for n in CAPTURE_FN_NAMES:
        if old[n] == new[n]:
            continue
        pair = (n, sha12(old[n]), sha12(new[n]))
        if pair in VINTAGE_TRIVIAL_PAIRS:
            logger.info("[p4-vintage] %s drift declared numeric-trivial (%s) — reuse", n, pair[1])
            continue
        return False
    return True


def _stage_reused_panel(cfg: Cfg, tree: str, arm_id: str) -> bool:
    """Stage one reused #1586 panel dir from the Hub (no layout mapping).

    ``raw_rows.json`` is REQUIRED for ``base_*`` capture trees: the 40
    non-pers arms' matched-text units hard-read the base panel rows
    (`_panel_tf_capture`), and the reused #1586 base trees do NOT carry
    raw_rows.json on the Hub (probed 2026-07-28) — a miss there must route
    to the fresh base capture, never a warn-and-continue (round-1 Critical
    ``p4-base-raw-rows-missing-crash``). ARM trees (pers-LoRA AND the
    amendment's ft cells) keep raw_rows optional — their raw text is never
    consumed downstream (`_panel_tf_capture` reads the BASE tree's rows; p9
    reads pooled.pt only). Every required file is probed BEFORE any byte is
    staged so a partial stage can never satisfy the resume predicate without
    the raw rows.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    dest = cfg.out_root / ("panel_capture" if tree == "capture" else "panel_capture_tf") / arm_id
    raw_required = tree == "capture" and arm_id.startswith("base_")
    files = ["pooled.pt", "manifest.json"] + (["raw_rows.json"] if tree == "capture" else [])
    required = [f for f in files if f != "raw_rows.json" or raw_required]
    if all((dest / f).exists() for f in required):
        return True
    hub_name = _reused_1586_hub_name(arm_id)
    if hub_name is None:
        return False
    api = HfApi()
    present: dict[str, bool] = {}
    for name in files:
        hub_path = f"{REUSED_PANEL_HF_PREFIX}/{tree}/{hub_name}/{name}"
        present[name] = (
            hub.retry_transient(  # HUB_VERIFY_RETRY_EXEMPT: wrapped in retry_transient here
                lambda p=hub_path: api.file_exists(X.HF_DATA_REPO, p, repo_type="dataset"),
                what=f"panel stage probe {hub_path}",
            )
        )
        if not present[name]:
            if name in required:
                logger.warning(
                    "[p4] reused tree miss %s (required) — falling back to fresh capture",
                    hub_path,
                )
                if dest.exists():  # drop any stale partial so fresh capture re-runs fully
                    shutil.rmtree(dest)
                return False
            logger.warning(
                "[p4] %s/%s lacks raw_rows.json on Hub (optional: pers-arm raw text is "
                "never consumed; recorded)",
                tree,
                arm_id,
            )
    for name in files:
        if present[name] and not (dest / name).exists():
            hub_path = f"{REUSED_PANEL_HF_PREFIX}/{tree}/{hub_name}/{name}"
            hub.stage_hub_file(X.HF_DATA_REPO, hub_path, dest / name, repo_type="dataset")
    man = json.loads((dest / "manifest.json").read_text())
    commit = man.get("git_commit", "")
    if not commit or not _vintage_ok(commit):
        logger.warning("[p4-vintage] %s/%s drifted (commit=%s) — RECAPTURE", tree, arm_id, commit)
        shutil.rmtree(dest)
        return False
    return True


def _reused_1586_hub_name(unit_id: str) -> str | None:
    """The #1586 capture-tree dir for this unit (base_<beh>, a reused pers
    LoRA arm's CELL name `<beh>-pers-lora-<regime>-s<seed>`, or — plan §4.1
    amendment — a ft ARM ID, which IS its #1586 cell/tree name), else None."""
    if unit_id.startswith("base_"):
        return unit_id
    arm = _full_arm_index().get(unit_id)
    if arm is not None and arm.method == "ft":
        return unit_id  # ft cells are the #1586 capture{,_tf}/<cell> dir names
    reused = X.reused_1586_arm(unit_id)
    return reused.cell if reused is not None else None


def _is_pers_reused(cfg: Cfg, unit_id: str) -> bool:
    if cfg.model_override:
        return False
    return _reused_1586_hub_name(unit_id) is not None


def run_p4_unit(cfg: Cfg, kind: str, unit_id: str) -> None:
    """p4 unit: `base:<base_X>` (own tree only) or `arm:<arm_id>` (own + tf)."""
    if kind == "base":
        if _is_pers_reused(cfg, unit_id) and _stage_reused_panel(cfg, "capture", unit_id):
            return
        model_path, cleanup = _resolve_unit_model(cfg, unit_id)
        try:
            _panel_own_capture(cfg, unit_id, model_path)
        finally:
            _cleanup_merged(cleanup)
        return
    own_done = (cfg.out_root / "panel_capture" / unit_id / "pooled.pt").exists()
    tf_done = (cfg.out_root / "panel_capture_tf" / unit_id / "pooled.pt").exists()
    if _is_pers_reused(cfg, unit_id):
        own_done = own_done or _stage_reused_panel(cfg, "capture", unit_id)
        tf_done = tf_done or _stage_reused_panel(cfg, "capture_tf", unit_id)
    if own_done and tf_done:
        return
    model_path, cleanup = _resolve_unit_model(cfg, unit_id)
    try:
        if not own_done:
            _panel_own_capture(cfg, unit_id, model_path)
        if not tf_done:
            _panel_tf_capture(cfg, unit_id, model_path)
    finally:
        _cleanup_merged(cleanup)


def panel_units(cfg: Cfg) -> list[tuple[str, str]]:
    """(kind, unit) list for p4: base panels first (tf trees consume their rows)."""
    arms = _arm_index(cfg)
    beh_keys = sorted({a.beh_key for a in arms.values()})
    units: list[tuple[str, str]] = [("base", f"base_{b}") for b in beh_keys]
    units += [("arm", a) for a in sorted(arms)]
    return units


# ── p5: δ captures (base model TF on training-mix positives) ─────────────────


def _completion_text(row: dict) -> str:
    comp = row["completion"]
    return comp[-1]["content"] if isinstance(comp, list) else str(comp)


def _mix_positive_rows(cfg: Cfg, arm: X.Arm) -> tuple[list[dict], dict]:
    """Stage + parse the arm's positives file into TF rows (token-id concat)."""
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import compute_prompt_spans
    from explore_persona_space.orchestrate import hub

    import hashlib

    reg = json.loads((cfg.out_root / "arm_registry.json").read_text())
    src = reg["mix_pos_sources"][arm.arm_id]
    local = cfg.out_root / "delta_tf" / arm.arm_id / Path(src["pos_path"]).name
    if not local.exists():
        hub.stage_hub_file(X.HF_DATA_REPO, src["pos_path"], local, repo_type="dataset")
    # No pinned train_mix_sha256 exists for the substituted CON pos pools
    # (round-1 Minor; folded into the po-delta-positives-con-family caveat) —
    # record the REALIZED staged-file sha so the provenance is auditable.
    pos_sha256 = hashlib.sha256(local.read_bytes()).hexdigest()
    raw = []
    with local.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                raw.append(json.loads(line))
    if src["layout"] == "marker-mix":  # positives = completion carries the marker
        raw = [r for r in raw if "※" in _completion_text(r)]
    assert raw, (arm.arm_id, src["pos_path"], "no positive rows")
    tok = AutoTokenizer.from_pretrained(cfg.model_override or X.BASE_MODEL)
    rows = []
    for r in raw:
        p_msgs = (
            r["prompt"]
            if isinstance(r["prompt"], list)
            else [{"role": "user", "content": r["prompt"]}]
        )
        text = tok.apply_chat_template(p_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
        resp_ids = tok(_completion_text(r), add_special_tokens=False)["input_ids"]
        if not resp_ids:
            continue
        system = next((m["content"] for m in p_msgs if m["role"] == "system"), None)
        chat = [m for m in p_msgs if m["role"] != "system"]
        question = chat[-1]["content"]
        prior = chat[:-1]
        flags: dict[str, bool] = {}
        prefix_len, context_len = compute_prompt_spans(
            tok,
            system,
            question,
            prompt_ids,
            prior_messages=prior or None,
            prefix_end="last_user",
            on_seam="snap",
            seam_flags=flags,
        )
        rows.append(
            {
                "persona": arm.arm_id,
                "question_idx": len(rows),
                "prompt_token_ids": prompt_ids,
                "response_token_ids": resp_ids,
                "prefix_len": prefix_len,
                "context_len": context_len,
            }
        )
    return rows, {
        "pos_path": src["pos_path"],
        "layout": src["layout"],
        "n_rows": len(rows),
        "pos_sha256": pos_sha256,
    }


def run_delta_unit(cfg: Cfg, arm_id: str) -> None:
    """p5 unit: t_bar_{C,B} = base-model response span-mean over mix positives."""
    import torch

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    out_dir = cfg.out_root / "delta_tf" / arm_id
    if (out_dir / "tbar.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    arm = _full_arm_index()[arm_id]
    assert arm.method == "lora", (arm_id, "δ units are the delta-arm (LoRA) set")
    rows, meta = _mix_positive_rows(cfg, arm)
    if cfg.smoke:
        rows = rows[:4]
    model_path = cfg.model_override or X.BASE_MODEL
    pooled = _teacher_forced_span_means(
        model_path,
        rows,
        [arm_id],
        layers=list(cfg.layers),
        spans=("response",),
        device=_device(),
        dtype=_dtype(),
        tf_batch_size=cfg.tf_batch,
    )
    tbar = {li: t.mean(dim=0) for li, t in pooled["response"].items()}
    # completion-row halves for the δ split-half reliability read (plan §4.4)
    halves = len(rows) >= 2
    tbar_even = (
        {li: t[0::2].mean(dim=0) for li, t in pooled["response"].items()} if halves else None
    )
    tbar_odd = {li: t[1::2].mean(dim=0) for li, t in pooled["response"].items()} if halves else None
    tmp = out_dir / "tbar.pt.tmp"
    torch.save(
        {
            "tbar": tbar,
            "tbar_even": tbar_even,
            "tbar_odd": tbar_odd,
            "n_rows": len(rows),
            "meta": {**_meta(), **meta},
        },
        tmp,
    )
    os.replace(tmp, out_dir / "tbar.pt")


# ── p6: scoped A4 re-extraction (split gen / judge / reduce; plan §9 Must-Fix) ─
#
# The persona-vectors recipe body is reused VERBATIM from issue779_extract_rb
# (prompt render `_build_prompts`, chunked sampling `_vllm_generate_chunked`,
# text persist `_dump_rollouts`, capture `_response_mean_activation`, reduce
# `RunningMean`, enumeration `_iter_rollout_records`) but re-sequenced per the
# plan §9 "p6 Batch-API wait placement" Must-Fix: the GPU leg (gen + capture)
# rides the p2 fan-out queue, the judge SUBMISSION fires the moment rollouts
# land (detached CPU poller), the poll overlaps p2..p7, and the post-filter is
# a CPU reduction over the persisted per-rollout activations after p7.

P6_TRAIT_BY_BEH = {"syc": "sycophancy", "imp": "impolite", "cas": "writing_style"}
P6_N_PAIRS, P6_N_EXT_Q, P6_N_ROLLOUTS = 5, 20, 10  # persona-vectors recipe (plan §4.4)
P6_JUDGE_N_DRAWS = 1  # plan §6: N=1 graded draw per rollout at the recipe thresholds
P6_ACTS_CHECKPOINT_EVERY = 250  # per-rollout capture checkpoint grain (T2 > 50 units)


def p6_arm_ids(cfg: Cfg) -> list[str]:
    """{cas,imp,syc} x {con,po} x seed 42 x pers, LoRA only (plan §4.4 p6 —
    the amendment adds NO p6 units: the A4 re-extraction stays 6 arms)."""
    out = []
    for a in _arm_index(cfg).values():
        if a.kind == "content" and a.ctx_key == "pers" and a.seed == 42 and a.method == "lora":
            out.append(a.arm_id)
    return sorted(out)


def seed_rb_artifacts(trait: str, cache_path: Path) -> dict:
    """Pre-seed the issue779 extractor's per-trait artifacts cache from the
    BEHAVIORS registry, matching each FLEET rb tensor's own extraction
    provenance (A4 compares rb_plus against those tensors): sycophancy = the
    #1112 seed (``ex.question_set``), impolite = the #1315 seed
    (``bank_slice(trait, "train")``), writing_style = the #1434 definition
    (``train_question_bank``). Without a seed the extractor Sonnet-generates
    its OWN issue779 definition (syc) or HARD-FAILS on an unregistered trait
    (imp/cas, ``_require_trait_known_or_seeded``) — plan §11 item 14."""
    from explore_persona_space.artifacts.behavior import BEHAVIORS

    b = BEHAVIORS[trait]
    ex = b.extraction
    assert ex is not None and b.judge_rubric, f"{trait} registry entry is a stub"
    assert "{question}" in b.judge_rubric and "{answer}" in b.judge_rubric, trait
    if trait == "sycophancy":
        ext_qs = list(ex.question_set)  # the #1112 fleet-rb seed source
    elif trait == "impolite":
        from explore_persona_space.artifacts.banks import bank_slice

        ext_qs = list(bank_slice(trait, "train"))  # the #1315 fleet-rb seed source
    elif trait == "writing_style":
        ext_qs = list(b.train_question_bank)  # the #1434 extraction-question source
    else:  # pragma: no cover - p6 traits are the three above
        raise ValueError(f"p6 seeding has no provenance precedent for trait {trait!r}")
    overlap = set(ext_qs) & set(b.eval_question_bank)
    assert not overlap, f"{trait}: extraction/eval question overlap {sorted(overlap)[:3]}"
    artifacts = {
        "instruction": [{"pos": pr.exhibit, "neg": pr.not_exhibit} for pr in ex.prompt_pairs],
        "extraction_questions": ext_qs,
        "eval_prompt": b.judge_rubric,
        "provenance": {
            "source": f"artifacts.behavior.BEHAVIORS[{trait!r}]",
            "seeded_by": "issue1768_capture.run_rb_gen_unit",
            "n_pairs": len(ex.prompt_pairs),
            "n_extraction_questions": len(ext_qs),
        },
    }
    _atomic_json(cache_path, artifacts)
    return artifacts


def _p6_dir(cfg: Cfg, arm_id: str) -> Path:
    return cfg.out_root / "rb_plus" / arm_id


def _p6_dims(cfg: Cfg) -> tuple[int, int, int]:
    """(n_pairs, n_ext_q, n_rollouts): the extractor's own smoke caps under
    --smoke (issue779_extract_rb main), recipe-verbatim in production."""
    return (1, 2, 5) if cfg.smoke else (P6_N_PAIRS, P6_N_EXT_Q, P6_N_ROLLOUTS)


def _p6_sampling_params(n_rollouts: int):
    """extract_trait_rb's exact SamplingParams; SimpleNamespace twin off-GPU
    (the CPU smoke's _HFGenShim reads only n/temperature/top_p/max_tokens)."""
    kw = {"n": n_rollouts, "temperature": 1.0, "top_p": 0.95, "max_tokens": 1024, "seed": 42}
    try:
        from vllm import SamplingParams

        return SamplingParams(**kw)
    except ImportError:
        import types

        return types.SimpleNamespace(**kw)


def _p6_hf_model(model_path: str):
    """HF capture model, extractor-parity dtypes (bf16 cuda:0 / fp32 CPU)."""
    import torch
    from transformers import AutoModelForCausalLM

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    return model


def _launch_detached_judge(cfg: Cfg, arm_id: str) -> None:
    """Fire the judge SUBMISSION the moment the arm's rollouts land (§9 Must-Fix).

    Detached CPU subprocess (survives this unit's exit; CVD cleared — never
    touches a GPU) running `--unit p6j:<arm>`: submit + poll + harvest through
    the primary judge_completions_batch path, writing judge_scores.json when
    done. The poll overlaps the remaining GPU phases; phase_p6_reduce waits on
    the file, resuming via the SAME #1019 checkpoint if the poller died
    (recorded batches are re-polled, never re-created)."""
    out_dir = _p6_dir(cfg, arm_id)
    if (out_dir / "judge_scores.json").exists():
        return
    pid_path = out_dir / "judge_poller.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pid = -1
        if pid > 0 and Path(f"/proc/{pid}").exists():
            logger.info(
                "[p6] %s: judge poller already live (pid %d) — not relaunching", arm_id, pid
            )
            return
    cmd, env = _unit_cmd(cfg, f"p6j:{arm_id}", gpu=cfg.gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = ""  # CPU + API only
    log_path = out_dir / "judge.log"
    with log_path.open("a") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach: outlives this unit subprocess
        )
    pid_path.write_text(str(proc.pid))
    logger.info("[p6] %s: judge poller launched (pid %d, log %s)", arm_id, proc.pid, log_path)


def run_rb_gen_unit(cfg: Cfg, arm_id: str) -> None:
    """p6 GPU leg (rides the p2 fan-out queue): gen -> text persist -> judge
    SUBMIT (detached) -> per-rollout response-avg activation persist.

    ALL rollouts' activations are persisted (fp16, keyed by custom_id) so the
    post-filter is a pure CPU reduction at phase_p6_reduce — no second GPU
    pass, no GPU phase blocking on the judge (plan §9 Must-Fix)."""
    import torch

    import issue779_common as C
    import issue779_extract_rb as E

    out_dir = _p6_dir(cfg, arm_id)
    if (out_dir / "gen_done.json").exists():
        _launch_detached_judge(cfg, arm_id)  # idempotent: re-arm a dead poller
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    arm = {a.arm_id: a for a in X.all_arms()}[arm_id]
    trait = P6_TRAIT_BY_BEH[arm.beh_key]
    seed_rb_artifacts(trait, REPO_ROOT / "data" / "issue_779" / "artifacts" / f"{trait}.json")
    n_pairs, n_ext_q, n_rollouts = _p6_dims(cfg)
    artifacts = C.load_extraction_artifacts(trait)
    pairs = artifacts["instruction"][:n_pairs]
    ext_q = artifacts["extraction_questions"][:n_ext_q]

    model_path, cleanup = _resolve_unit_model(cfg, arm_id)
    hf_model = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        rollouts_path = out_dir / "rollouts.json"
        if rollouts_path.exists():
            blob = json.loads(rollouts_path.read_text())
            rollouts, sysprompt_of = blob["rollouts"], blob["sysprompt_of"]
        else:
            sp = _p6_sampling_params(n_rollouts)
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                from explore_persona_space.eval.generation import (
                    cleanup_vllm,
                    create_vllm_engine,
                )

                # extractor-parity engine (issue779_extract_rb main); reaped
                # BEFORE the HF capture model loads (no co-residency needed —
                # the capture leg is sequenced after generation)
                llm = create_vllm_engine(model_path, max_model_len=2048, seed=42)
            else:  # CPU smoke: the extractor's own HF-generate shim
                hf_model = _p6_hf_model(model_path)
                llm = E._HFGenShim(hf_model, tokenizer)
            rollouts: dict[str, dict[str, dict[str, list[str]]]] = {"pos": {}, "neg": {}}
            sysprompt_of: dict[str, str] = {}
            try:
                for side in ("pos", "neg"):
                    for pi, pair in enumerate(pairs):
                        sys_prompt = pair[side]
                        persona = f"{trait}_{side}_p{pi}"
                        sysprompt_of[persona] = sys_prompt
                        prompt_texts = E._build_prompts(tokenizer, sys_prompt, ext_q)
                        gen = E._vllm_generate_chunked(llm, prompt_texts, sp)
                        rollouts[side][persona] = dict(zip(ext_q, gen, strict=True))
            finally:
                if use_cuda:
                    cleanup_vllm(llm)
            sampling = {
                "n": sp.n,
                "temperature": sp.temperature,
                "top_p": sp.top_p,
                "max_tokens": sp.max_tokens,
                "seed": getattr(sp, "seed", None),
                "model": model_path,
            }
            rollout_files = {}
            for side in ("pos", "neg"):
                dumped = E._dump_rollouts(trait, side, rollouts[side], out_dir, sampling)
                rollout_files[side] = [p.name for p in dumped]
            _atomic_json(
                rollouts_path,
                {
                    "trait": trait,
                    "rollouts": rollouts,
                    "sysprompt_of": sysprompt_of,
                    "sampling": sampling,
                    "rollout_files": rollout_files,
                    **_meta(),
                },
            )

        # judge submission fires the moment the rollouts land — the batch
        # processes server-side while this unit captures activations and the
        # rest of the p2..p7 GPU work runs (plan §9 Must-Fix)
        _launch_detached_judge(cfg, arm_id)

        acts_path = out_dir / "acts.pt"
        if not acts_path.exists():
            if hf_model is None:
                hf_model = _p6_hf_model(model_path)
            layers = list(range(len(hf_model.model.layers)))
            partial_path = out_dir / "acts_partial.pt"
            acts: dict[str, dict[str, torch.Tensor]] = {"pos": {}, "neg": {}}
            skipped: dict[str, list[str]] = {"pos": [], "neg": []}
            if partial_path.exists():  # per-unit resume (checkpoint-per-phase)
                part = torch.load(partial_path, map_location="cpu", weights_only=False)
                if part.get("layers") == layers:
                    acts, skipped = part["acts"], part["skipped_empty"]
            done_cids = {c for side in acts.values() for c in side} | {
                c for side in skipped.values() for c in side
            }
            records = [
                (side, persona, question, comp, cid)
                for side in ("pos", "neg")
                for persona, _q, question, _ci, comp, cid in E._iter_rollout_records(rollouts[side])
            ]
            n_new = 0
            t0 = time.time()
            for k, (side, persona, question, comp, cid) in enumerate(records):
                if cid in done_cids:
                    continue
                act = E._response_mean_activation(
                    hf_model, tokenizer, sysprompt_of[persona], question, comp, layers
                )
                if act is None:
                    skipped[side].append(cid)
                else:
                    acts[side][cid] = act.to(torch.float16)
                n_new += 1
                if n_new % P6_ACTS_CHECKPOINT_EVERY == 0:
                    tmp = partial_path.with_suffix(".pt.tmp")
                    torch.save({"layers": layers, "acts": acts, "skipped_empty": skipped}, tmp)
                    os.replace(tmp, partial_path)
                    print(
                        f"[p6g] {arm_id} acts {k + 1}/{len(records)} "
                        f"elapsed={time.time() - t0:.0f}s",
                        flush=True,
                    )
            tmp = acts_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "trait": trait,
                    "layers": layers,
                    "acts": acts,
                    "skipped_empty": skipped,
                    "metadata": _meta(),
                },
                tmp,
            )
            os.replace(tmp, acts_path)
            partial_path.unlink(missing_ok=True)
        _atomic_json(
            out_dir / "gen_done.json",
            {
                "trait": trait,
                "model_path": model_path,
                "n_pairs": n_pairs,
                "n_ext_q": n_ext_q,
                "n_rollouts": n_rollouts,
                **_meta(),
            },
        )
    finally:
        _cleanup_merged(cleanup)


def run_rb_judge(cfg: Cfg, arm_id: str) -> None:
    """p6 judge leg (CPU + Batch API; runs detached via `--unit p6j:<arm>`).

    N=1 graded draw per rollout at the recipe filter thresholds (plan §6:
    ~12,600 calls across the 6 arms), max_tokens=300 (llm-judging rule 23),
    through C.judge_rollouts_n5 -> judge_completions_batch (#1019 resumable
    checkpoint under rb_plus/<arm>/.judge_dispatch — a killed poller's re-run
    polls the SAME recorded batches, never re-creates)."""
    import issue779_common as C

    out_dir = _p6_dir(cfg, arm_id)
    dest = out_dir / "judge_scores.json"
    if dest.exists():
        return
    blob = json.loads((out_dir / "rollouts.json").read_text())
    trait = blob["trait"]
    payload: dict = {"trait": trait, "n_draws": P6_JUDGE_N_DRAWS, "arms": {}}
    for side in ("pos", "neg"):
        agg = C.judge_rollouts_n5(
            trait,
            blob["rollouts"][side],
            out_dir / f"judge_{trait}_{side}.json",
            None,
            n_draws=P6_JUDGE_N_DRAWS,
        )
        scores: dict[str, float | None] = {}
        n_draws_seen = total_valid = all_dropped = 0
        for cid, (mean, n_valid, n_draws) in agg.items():
            scores[cid] = mean
            n_draws_seen += n_draws
            total_valid += n_valid
            all_dropped += int(n_valid == 0)
        payload["arms"][side] = {
            "scores": scores,
            "draw_stats": {  # per-arm drop report (llm-judging rules 9/18)
                "n_rollouts_judged": len(agg),
                "n_draws_per_rollout": P6_JUDGE_N_DRAWS,
                "total_draws": n_draws_seen,
                "total_valid_draws": total_valid,
                "total_dropped_draws": n_draws_seen - total_valid,
                "n_rollouts_all_draws_dropped": all_dropped,
            },
        }
    payload.update(_meta())
    _atomic_json(dest, payload)


def _await_judge_scores(cfg: Cfg, arm_id: str) -> dict:
    """Block (CPU-only phase; GPUs already released) until judge_scores.json
    exists — waiting on a LIVE poller, else resuming the judge in-process."""
    out_dir = _p6_dir(cfg, arm_id)
    dest = out_dir / "judge_scores.json"
    pid_path = out_dir / "judge_poller.pid"
    waited = 0.0
    while not dest.exists():
        pid = -1
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
            except ValueError:
                pid = -1
        if pid > 0 and Path(f"/proc/{pid}").exists():
            if waited % 300 < 30:
                logger.info(
                    "[p6] %s: waiting on live judge poller (pid %d, %.0fs)", arm_id, pid, waited
                )
            time.sleep(30)
            waited += 30
            continue
        logger.info("[p6] %s: judge poller not live — resuming judge in-process", arm_id)
        run_rb_judge(cfg, arm_id)  # checkpoint resume; raises loud on failure
    return json.loads(dest.read_text())


def run_rb_reduce_unit(cfg: Cfg, arm_id: str) -> dict:
    """p6 CPU post-filter: threshold the judged scores, reduce the PERSISTED
    per-rollout activations to r_B (extractor-verbatim math; no GPU pass)."""
    import torch

    out_dir = _p6_dir(cfg, arm_id)
    done = out_dir / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    scores_blob = _await_judge_scores(cfg, arm_id)
    blob = json.loads((out_dir / "rollouts.json").read_text())
    trait = blob["trait"]
    acts_blob = torch.load(out_dir / "acts.pt", map_location="cpu", weights_only=False)
    reduced = reduce_rb_from_persisted(
        trait, blob["rollouts"], scores_blob, acts_blob, smoke=cfg.smoke
    )
    rb_dir = out_dir / "r_b"
    rb_dir.mkdir(parents=True, exist_ok=True)
    tmp = rb_dir / f"{trait}.pt.tmp"
    torch.save(
        {  # schema parity with issue779_extract_rb (rb_stability reads obj["r_b"])
            "trait": trait,
            "r_b": reduced["r_b"],
            "layers": acts_blob["layers"],
            "counts": reduced["counts"],
            "smoke": cfg.smoke,
            "metadata": _meta(),
        },
        tmp,
    )
    os.replace(tmp, rb_dir / f"{trait}.pt")
    _atomic_json(rb_dir / f"{trait}_counts.json", reduced["counts"])
    gen_meta = json.loads((out_dir / "gen_done.json").read_text())
    out = {"trait": trait, "model_path": gen_meta["model_path"], **_meta()}
    _atomic_json(done, out)
    return out


def reduce_rb_from_persisted(
    trait: str, rollouts: dict, scores_blob: dict, acts_blob: dict, *, smoke: bool
) -> dict:
    """Pure CPU reduce: judge-threshold filter (POS>50 keep / NEG<50 keep,
    None = dropped, never coerced) over the persisted per-rollout activations.

    Smoke-only gate demotion (#1345 gate-calibration rule): a zero-kept arm
    under --smoke (tiny garbage-weights model) falls back to keep-all,
    LABELED in counts; the production zero-kept raise is byte-untouched."""
    import issue779_extract_rb as E

    layers = acts_blob["layers"]
    hidden = None
    for side_acts in acts_blob["acts"].values():
        for t in side_acts.values():
            hidden = int(t.shape[1])
            break
        if hidden is not None:
            break
    assert hidden is not None, f"{trait}: acts store has no captured rollouts at all"
    counts: dict = {"trait": trait, "arms": {}}
    means: dict[str, E.RunningMean] = {}
    for side in ("pos", "neg"):
        scores = scores_blob["arms"][side]["scores"]
        n_total = n_dropped = n_below = 0
        kept_cids: list[str] = []
        for _p, _q, _question, _ci, _comp, cid in E._iter_rollout_records(rollouts[side]):
            n_total += 1
            s = scores.get(cid)
            if s is None:
                n_dropped += 1
                continue
            if side == "pos" and not (s > 50.0):
                n_below += 1
                continue
            if side == "neg" and not (s < 50.0):
                n_below += 1
                continue
            kept_cids.append(cid)
        smoke_keep_all = False
        if not kept_cids and smoke:
            smoke_keep_all = True
            kept_cids = [cid for *_r, cid in E._iter_rollout_records(rollouts[side])]
        rm = E.RunningMean(len(layers), hidden)
        for cid in kept_cids:
            act = acts_blob["acts"][side].get(cid)
            if act is not None:
                rm.add(act.float())
        means[side] = rm
        counts["arms"][side] = {
            "total": n_total,
            "kept": len(kept_cids),
            "dropped_refusal_or_invalid": n_dropped,
            "dropped_below_threshold": n_below,
            "captured": rm.count,
            "smoke_keep_all_fallback": smoke_keep_all,
            "judge_draw_stats": scores_blob["arms"][side]["draw_stats"],
        }
    assert means["pos"].count > 0 and means["neg"].count > 0, (
        f"{trait}: zero kept rollouts in an arm (pos={means['pos'].count}, "
        f"neg={means['neg'].count}); cannot form r_B — the judge-filter dropped an "
        "entire arm (report as a yield failure, do NOT fabricate a direction)"
    )
    r_b = means["pos"].mean() - means["neg"].mean()
    assert r_b.shape == (len(layers), hidden), r_b.shape
    return {"r_b": r_b, "counts": counts}


def phase_p6_reduce(cfg: Cfg) -> None:
    """p6 reduce (CPU; AFTER p7 by default): harvest judge -> filter -> r_B.

    The GPU legs already ran inside the p2 queue; any residual batch wait
    lands here with every GPU phase complete (plan §9: the p7 window absorbs
    it — no GPU is held through the poll)."""
    _phase("p6_rb_reduce")
    arms = p6_arm_ids(cfg)
    _status(cfg, "p6_rb_reduce", pending=len(_pending_units(cfg, "p6")))
    for k, arm_id in enumerate(arms):
        t0 = time.time()
        run_rb_reduce_unit(cfg, arm_id)
        print(
            f"[p6] unit {k + 1}/{len(arms)} {arm_id} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        _status(cfg, "p6_rb_reduce", done=k + 1, total=len(arms))
    if cfg.upload and arms:
        _upload_tree(cfg, "rb_plus")  # r_b landed after p7's tree upload


# ── p1: pilot (gate 1) ───────────────────────────────────────────────────────


def phase_p1(cfg: Cfg) -> None:
    """ONE arm end-to-end through the production unit fns + timing + fp16 probe."""
    _phase("p1_pilot")
    _status(cfg, "p1_pilot")
    pilot = cfg.arms[0] if cfg.arms else X.PILOT_ARM
    base_unit = X.base_unit_for(pilot)
    report: dict = {"pilot_arm": pilot, "base_unit": base_unit, **_meta()}

    t0 = time.time()
    run_corpus_unit(cfg, base_unit)
    report["base_unit_wall_s"] = time.time() - t0
    t0 = time.time()
    run_corpus_unit(cfg, pilot)
    report["arm_unit_wall_s"] = time.time() - t0
    t0 = time.time()
    run_corpus_tf_unit(cfg, pilot)
    report["tf_unit_wall_s"] = time.time() - t0

    # assumption 4: the corpus prefix arm stays DROPPED only while degenerate
    # (plan §4.8 re-open path). Measured 2026-07-29 on the real corpus:
    # n_distinct_prefix == 2 (LMSYS vs WildChat template split) — unfittable for
    # a 3584-dim map, so the drop's degeneracy CONCLUSION holds; the strict
    # "exactly 1" premise was too tight. Threshold: >= PREFIX_REOPEN_FLOOR
    # distinct prefixes = fittable -> fail loud to re-open the arm.
    PREFIX_REOPEN_FLOOR = 100
    man = json.loads((cfg.out_root / "corpus_capture" / base_unit / "manifest.json").read_text())
    n_dp = int(man["n_distinct_prefix"])
    report["n_distinct_prefix"] = n_dp
    report["prefix_arm_dropped_degenerate"] = n_dp < PREFIX_REOPEN_FLOOR
    assert n_dp < PREFIX_REOPEN_FLOOR, (
        f"corpus prefix has {n_dp} distinct values (>= {PREFIX_REOPEN_FLOOR}) — fittable; "
        "re-open the corpus prefix arm per the plan §4.8 re-open path"
    )
    if n_dp != 1:
        logger.warning(
            "[p1] prefix-arm degeneracy: n_distinct_prefix=%d (premise said 1; < %d keeps "
            "the §4.8 drop — recorded as a quantified scope caveat)",
            n_dp,
            PREFIX_REOPEN_FLOOR,
        )
    # fp16 storage probe (assumption 11): recorded per unit at capture time
    report["fp16_roundtrip_cos_min"] = man["fp16_roundtrip_cos_min"]

    # gate 1: measured unit wall vs the #779-basis ceiling (pass: <= 2x 0.405 GPU-h)
    gate1_pass = cfg.smoke or (report["arm_unit_wall_s"] / 3600.0) <= 2 * 0.405
    report["gate1"] = {"ceiling_gpu_h": 2 * 0.405, "pass": bool(gate1_pass)}

    # gate 2 re-anchor: pilot M0 fit at the anchor layer (issue1768_fit helper)
    import issue1768_fit as fit

    anchor_layer = 19 if 19 in cfg.layers else cfg.layers[0]
    m0 = fit.pilot_m0_fit(cfg.out_root, base_unit, anchor_layer, smoke=cfg.smoke)
    report["pilot_m0"] = m0
    thr = 0.55
    r2 = m0["heldout_r2"]
    if not cfg.smoke:
        if r2 < 0.45:
            report["gate2"] = {"threshold": thr, "pilot_r2": r2, "verdict": "RIG-BUG"}
            _atomic_json(cfg.out_root / "pilot" / "pilot_report.json", report)
            print(f"[p1] GATE2 RIG-BUG: pilot M0 R2@L{anchor_layer}={r2:.3f} < 0.45", flush=True)
            sys.exit(7)  # designed halt, distinct rc (gotchas: pilot gates)
        if r2 < thr:
            thr = round(r2 - 0.10, 3)  # cross-surface re-anchor (consistency W1)
    report["gate2"] = {"threshold": thr, "pilot_r2": r2, "verdict": "PASS"}
    _atomic_json(cfg.out_root / "pilot" / "pilot_report.json", report)
    if not gate1_pass:
        print("[p1] GATE1 FAIL: pilot unit wall exceeded 2x ceiling", flush=True)
        sys.exit(7)


# ── dispatcher (work-conserving CVD-pinned fan-out; #1586 pattern) ───────────


def _physical_gpus() -> list[int]:
    """Visible GPUs via nvidia-smi subprocess (never torch.cuda; gotchas CVD)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return [int(x) for x in cvd.split(",") if x.strip() != ""]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [int(line) for line in out.split("\n") if line.strip()]
    except (OSError, subprocess.CalledProcessError, ValueError):
        return []


def _pending_units(cfg: Cfg, phase: str) -> list[str]:
    """Unit args (`<kind>:<id>` for p4, bare ids otherwise) still to run."""
    arms = sorted(_arm_index(cfg))
    root = cfg.out_root
    if phase == "p2":
        base_units = ["base_content"] + (
            ["base_mk"] if any(a.startswith("mk-") for a in arms) else []
        )
        units = base_units + arms
        pend = [u for u in units if not (root / "corpus_capture" / u / "pooled.pt").exists()]
        # the 6 p6 GPU legs ride the p2 queue (plan §9 Batch-API wait placement)
        pend += [
            f"p6g:{a}"
            for a in p6_arm_ids(cfg)
            if not (root / "rb_plus" / a / "gen_done.json").exists()
        ]
        return pend
    if phase == "p3":
        return [a for a in arms if not (root / "corpus_capture_tf" / a / "pooled_tf.pt").exists()]
    if phase == "p4":
        out = []
        for kind, unit in panel_units(cfg):
            if kind == "base":
                d = root / "panel_capture" / unit
                # base trees need raw_rows.json too (the arm tf units read it)
                if not ((d / "pooled.pt").exists() and (d / "raw_rows.json").exists()):
                    out.append(f"{kind}:{unit}")
            else:
                own = (root / "panel_capture" / unit / "pooled.pt").exists()
                tf = (root / "panel_capture_tf" / unit / "pooled.pt").exists()
                if not (own and tf):
                    out.append(f"{kind}:{unit}")
        return out
    if phase == "p5":
        # ft arms map onto the matched pers-LoRA cells' t̄ (plan §4.1
        # amendment): the δ unit set is the DELTA-ARM set, so no ft arm ever
        # adds a p5 cell and an ft-only scope still stages its paired cell.
        delta_units = sorted({X.delta_arm_for(a) for a in _arm_index(cfg).values()})
        return [u for u in delta_units if not (root / "delta_tf" / u / "tbar.pt").exists()]
    if phase == "p6":
        return [a for a in p6_arm_ids(cfg) if not (root / "rb_plus" / a / "done.json").exists()]
    if phase == "pnf":
        return [
            f"{u}:{rep}"
            for u in _pnf_units(cfg)
            for rep in PNF_REPLICATES
            if not (root / "noise_floor" / u / f"pooled_nf_{rep}.pt").exists()
        ]
    if phase == "pfx2":
        units, _ = _pfx_unit_sets(cfg)
        return [
            u
            for u in units
            if not (root / "on_target" / "corpus_capture" / u / "pooled.pt").exists()
        ]
    if phase == "pfx3":
        _, trained = _pfx_unit_sets(cfg)
        return [
            u
            for u in trained
            if not (root / "on_target" / "corpus_capture_tf" / u / "pooled_tf.pt").exists()
        ]
    if phase == "lad2":
        return [
            u
            for u in _lad_unit_set(cfg)
            if not (root / "on_target_r4" / "corpus_capture" / u / "pooled.pt").exists()
        ]
    if phase == "brl2":
        return [
            u
            for u in _brl_unit_set(cfg)
            if not (root / "on_target_r5" / "corpus_capture" / u / "pooled.pt").exists()
        ]
    raise ValueError(phase)


def run_unit(cfg: Cfg, unit_arg: str) -> None:
    """Subprocess entry: `<phase>:<unit>` (p4 units are `p4:<kind>:<unit>`)."""
    phase, rest = unit_arg.split(":", 1)
    if phase == "p2":
        if rest.startswith("p6g:"):
            run_rb_gen_unit(cfg, rest.split(":", 1)[1])
        else:
            run_corpus_unit(cfg, rest)
    elif phase == "p3":
        run_corpus_tf_unit(cfg, rest)
    elif phase == "p4":
        kind, unit = rest.split(":", 1)
        run_p4_unit(cfg, kind, unit)
    elif phase == "p5":
        run_delta_unit(cfg, rest)
    elif phase == "p6j":
        run_rb_judge(cfg, rest)  # the detached CPU judge poller entry
    elif phase == "pfx2":
        run_pfx_corpus_unit(cfg, rest)
    elif phase == "pfx3":
        run_pfx_tf_unit(cfg, rest)
    elif phase == "lad2":
        run_lad_corpus_unit(cfg, rest)
    elif phase == "brl2":
        run_brl_corpus_unit(cfg, rest)
    else:
        raise ValueError(unit_arg)


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
        "--smoke-rows",
        ",".join(str(x) for x in cfg.smoke_rows),
    ]
    if cfg.smoke:
        cmd.append("--smoke")
    if cfg.arms:
        cmd += ["--arms", ",".join(cfg.arms)]
    if cfg.model_override:
        cmd += ["--model-override", cfg.model_override]
    # explicit env (subprocess env= contract) + launcher-level CVD pin (#545)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return cmd, env


def _needs_merge(cfg: Cfg, unit_arg: str) -> bool:
    """~15 GB local model materialization pending (LoRA merge OR ft stage) —
    the disk-clamp predicate `_merge_slots` gates on (plan §4.4)."""
    if cfg.model_override:
        return False
    unit = unit_arg.split(":")[-1].split("@")[0]  # pfx units carry an @<cond> tag
    if unit.startswith("base_"):
        return False
    arm = _full_arm_index().get(unit)
    if arm is not None and arm.method == "ft":
        return _ft_ckpt_incomplete_reason(_ft_ckpt_dirs(cfg, arm)[1]) is not None
    return not (cfg.out_root / "merged" / unit / "config.json").exists()


def _merge_slots(cfg: Cfg, width: int) -> int:
    """Merge-bearing concurrency clamp keyed to a free-disk probe (plan §4.4)."""
    free_gb = shutil.disk_usage(cfg.out_root).free / 1e9
    slots = int((free_gb - 100) // 16)
    return max(1, min(width, slots))


def _barrier_units(phase: str, queue_units: list[str]) -> set[str]:
    """Units every later unit must wait for. ONLY p4 keeps a barrier (the arm
    tf units consume the base panels' raw rows). p2 has NO barrier — no p2
    arm unit consumes base outputs (only p3 does, a later phase), and the
    round-1 p2 barrier idled up to 7/8 GPUs through base_mk with no data
    dependency (Major 5; #813 wave-barrier family). Alphabetical sort still
    dispatches the base_* units first as a preference."""
    if phase == "p4":
        return {u for u in queue_units if u.startswith("base:")}
    return set()


def _unit_ready(cfg: Cfg, phase: str, unit_arg: str) -> bool:
    """Dispatch eligibility beyond queue order: a p6 gen unit waits for its
    OWN arm's corpus unit — both resolve the same merged dir and each unit
    deletes it at exit (merge -> consume -> delete lifecycle), so they must
    never be live concurrently. Work-conserving: a not-ready p6g unit is
    skipped, never blocks a sibling dispatch."""
    if phase == "p2" and unit_arg.startswith("p6g:"):
        arm = unit_arg.split(":", 1)[1]
        return (cfg.out_root / "corpus_capture" / arm / "pooled.pt").exists()
    return True


def _model_key(cfg: Cfg, unit_arg: str) -> str | None:
    """Shared LOCAL-model-dir key: units with equal keys resolve — and at exit
    DELETE — the same `merged/<arm>` (or `ft_ckpt/<arm>`) dir, so two live
    same-key units race the merge -> consume -> delete lifecycle: the first
    finisher's exit-cleanup rmtree's the shared dir from under the survivor,
    whose next `from_pretrained` on the vanished RELATIVE path falls through
    transformers' isdir check into Hub repo-id validation (job 16120: lad2
    co-scheduled imp-pers@r_{long,mid,short}; @r_mid exited first and its
    cleanup killed both live siblings with HFValidationError). Same unit
    extraction as `_needs_merge`; None = no shared local dir (base_* units
    and model_override resolve repo ids / a shared snapshot — cleanup None)."""
    if cfg.model_override:
        return None
    unit = unit_arg.split(":")[-1].split("@")[0]
    if unit.startswith("base_"):
        return None
    return unit


def _fanout_phase(cfg: Cfg, phase: str, phase_tag: str) -> None:
    _phase(phase_tag)
    units = _pending_units(cfg, phase)
    _status(cfg, phase_tag, pending=len(units))
    if not units:
        logger.info("[%s] nothing pending", phase)
        return
    # p6g legs LAST (their readiness waits on their own arm's corpus unit),
    # then alphabetical (base_* first among the rest); barrier units first.
    units.sort(key=lambda u: (u.startswith("p6g:"), u.split(":")[-1] if ":" in u else u))
    barrier = _barrier_units(phase, units)
    units.sort(key=lambda u: u not in barrier)  # barrier units first (stable)
    gpus = _physical_gpus()
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
    merge_slots = _merge_slots(cfg, len(gpus))
    queue = list(units)
    running: dict[int, tuple[subprocess.Popen, str, float]] = {}
    done_count = 0
    while queue or running:
        barrier_live = any(u in barrier for u in queue) or any(
            r[1] in barrier for r in running.values()
        )
        for gpu in [g for g in gpus if g not in running]:
            if not queue:
                break
            active_merges = sum(1 for _p, ua, _t in running.values() if _needs_merge(cfg, ua))
            # same-model-key serialization: never co-schedule two units sharing
            # one merged/ft-staged dir — each deletes it at exit (job 16120)
            live_keys = {_model_key(cfg, ua) for _p, ua, _t in running.values()}
            live_keys.discard(None)
            nxt_i = next(
                (
                    i
                    for i, ua in enumerate(queue)
                    if (ua in barrier or not barrier_live)
                    and (not _needs_merge(cfg, ua) or active_merges < merge_slots)
                    and _unit_ready(cfg, phase, ua)
                    and _model_key(cfg, ua) not in live_keys
                ),
                None,
            )
            if nxt_i is None:
                break
            unit_arg = queue.pop(nxt_i)
            full_arg = f"{phase}:{unit_arg}"
            cmd, env = _unit_cmd(cfg, full_arg, gpu)
            log_path = cfg.out_root / "logs" / f"{full_arg.replace(':', '_')}.log"
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
                log_path = cfg.out_root / "logs" / f"{phase}_{unit_arg.replace(':', '_')}.log"
                tail = log_path.read_text()[-4000:] if log_path.exists() else "(no log)"
                # terminate the sibling subprocesses BEFORE raising — orphaned
                # children otherwise keep writing shared outputs (round-1
                # Minor; kill-before-relaunch discipline is the backstop)
                for sib_proc, sib_arg, _t in running.values():
                    logger.warning("[%s] terminating sibling %s on failure", phase, sib_arg)
                    sib_proc.terminate()
                deadline = time.time() + 15
                for sib_proc, _sib_arg, _t in running.values():
                    try:
                        sib_proc.wait(timeout=max(0.1, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        sib_proc.kill()
                running.clear()
                raise RuntimeError(
                    f"[{phase}] unit {unit_arg} exited rc={rc}\n--- log tail ---\n{tail}"
                )
            done_count += 1
            print(
                f"[{phase}] unit {done_count}/{len(units)} {unit_arg} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
            _status(cfg, phase_tag, done=done_count, total=len(units))


# ── p7: upload ───────────────────────────────────────────────────────────────

UPLOAD_TREES = (
    "inputs",
    "arm_registry.json",
    "pilot",
    "corpus_capture",
    "corpus_capture_tf",
    "panel_capture",
    "panel_capture_tf",
    "delta_tf",
    "rb_plus",
)


def _upload_tree(cfg: Cfg, name: str) -> str:
    """One fail-loud upload_folder commit for one out-root tree (no
    eligibility filter — every file in the tree uploads)."""
    from explore_persona_space.orchestrate import hub

    local = cfg.out_root / name
    if not local.exists():
        return ""
    dest = f"{cfg.hf_prefix}/{name}"
    url = hub._upload(
        local,
        repo_id=X.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        upload_as_file=local.is_file(),
    )
    if not url:
        raise RuntimeError(f"upload of {local} -> {dest} returned no path")
    return dest


def phase_p7(cfg: Cfg) -> None:
    """Whole-tree uploads (one upload_folder commit per tree; no eligibility
    filter — every file in each tree uploads; plan §10 destinations)."""
    _phase("p7_store_upload")
    _status(cfg, "p7_store_upload")
    if not cfg.upload:
        logger.info("[p7] upload disabled (--no-upload)")
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    uploaded = {}
    for name in UPLOAD_TREES:
        dest = _upload_tree(cfg, name)
        if dest:
            uploaded[name] = dest
    # exact-set verify on the load-bearing store files (one scoped listing)
    expected = []
    for tree, fname in (
        ("corpus_capture", "pooled.pt"),
        ("corpus_capture_tf", "pooled_tf.pt"),
        ("panel_capture", "pooled.pt"),
        ("panel_capture_tf", "pooled.pt"),
        ("delta_tf", "tbar.pt"),
    ):
        local_tree = cfg.out_root / tree
        if local_tree.exists():
            for unit_dir in sorted(local_tree.iterdir()):
                if (unit_dir / fname).exists():
                    expected.append(f"{cfg.hf_prefix}/{tree}/{unit_dir.name}/{fname}")
    if expected:
        missing = hub.verify_repo_paths_uploaded(
            HfApi(), X.HF_DATA_REPO, expected, path_in_repo=cfg.hf_prefix, repo_type="dataset"
        )
        assert not missing, f"p7 verify: {len(missing)} store files missing on Hub: {missing[:5]}"
    _atomic_json(
        cfg.out_root / "upload_done.json",
        {"uploaded": uploaded, "n_verified": len(expected), **_meta()},
    )


# ── pnf: matched-text capture-noise floor (plan v7 follow-up) ────────────────

PNF_UNITS = (
    "base_content",
    "cas-pers-con-lr1e5-s42",
    "imp-pers-con-lr3e5-s42",
    "syc-pers-con-lr1e5-s42",
    "mk-pers-con-lr5e6-s42",
    "imp-pers-ft-con-s42",
)
PNF_N_ROWS = 2_000  # seed-42 TRAIN subsample size (plan v7 §4.2)
PNF_R2_TF_BATCH = 13  # r2 geometry perturbation (plan v7 §4.3; any !=8 suffices)
PNF_REPLICATES = ("r1", "r2")
_PNF_BASE_REQUIRED = ("rows_spans.json", "raw_rows.done.json", "pooled.pt", "manifest.json")


def _pnf_units(cfg: Cfg) -> list[str]:
    """pnf unit set (plan v7 §4.1). --smoke = base_content only (§4.4 smoke
    parity: 1 unit through the SAME entrypoint); --arms filters the TRAINED
    units (base_* always kept — it is the model-independent control)."""
    if cfg.arms:
        keep = set(cfg.arms)
        return [u for u in PNF_UNITS if u.startswith("base_") or u in keep]
    if cfg.smoke:
        return ["base_content"]
    return list(PNF_UNITS)


def _pnf_dir(cfg: Cfg) -> Path:
    return cfg.out_root / "noise_floor"


def _pnf_results_dir(cfg: Cfg) -> Path:
    """Reduce-output destination. --smoke diverts to a scratch dir under the
    out-root so smoke outputs never touch committed eval_results (#722)."""
    if cfg.smoke:
        return cfg.out_root / "results_smoke"
    return REPO_ROOT / "eval_results" / "issue_1768"


def _pnf_base_incomplete(d: Path) -> str | None:
    """None when a base tree carries the FULL pnf-consumer file set (rows +
    spans + pooled store for the same-pass anchor); else the reason. Keying
    completeness on one proxy file is the #1090/#1315 partial-stage trap."""
    for f in _PNF_BASE_REQUIRED:
        if not (d / f).exists():
            return f"{f} missing"
    if not list(d.glob("raw_rows_*.jsonl")):
        return "no raw_rows shards"
    return None


def _pnf_staged_tree_dir(cfg: Cfg, rel_prefix: str, hf_prefix: str | None = None) -> Path:
    return cfg.out_root / "nf_staging" / (hf_prefix or X.HF_PREFIX) / rel_prefix


def _pnf_stage_tree(cfg: Cfg, rel_prefix: str, hf_prefix: str | None = None) -> Path:
    """Stage `{hf_prefix}/{rel_prefix}` (default: the parent's verified
    production prefix) via the #1402 canonical helper; returns the CONSUMED
    dir under the verbatim mirror root (the #1774 dest-is-mirror-root rule:
    `mirror_root/<hub prefix> == consumed path` by construction, asserted)."""
    from explore_persona_space.orchestrate import hub

    src_prefix = hf_prefix or X.HF_PREFIX
    hub.stage_hub_prefix(
        X.HF_DATA_REPO,
        f"{src_prefix}/{rel_prefix}",
        cfg.out_root / "nf_staging",
        repo_type="dataset",
    )
    out = _pnf_staged_tree_dir(cfg, rel_prefix, src_prefix)
    assert out.exists(), f"pnf staging mirror-root arithmetic broke: {out} absent"
    return out


def _pnf_resolved_base_dir(cfg: Cfg, base_unit: str) -> Path:
    """The base tree the pnf consumers read: canonical local (same-out-root
    resume / smoke) else the staged mirror. Raises when neither is complete."""
    local = cfg.out_root / "corpus_capture" / base_unit
    if _pnf_base_incomplete(local) is None:
        return local
    staged = _pnf_staged_tree_dir(cfg, f"corpus_capture/{base_unit}")
    reason = _pnf_base_incomplete(staged)
    if reason is None:
        return staged
    raise RuntimeError(f"[pnf] base tree {base_unit} unavailable ({reason}) — run pnf staging")


def _pnf_stage(cfg: Cfg) -> dict[str, Path]:
    """pnf_stage: inputs + arm_registry + base trees, local-else-Hub.

    Sources are the parent run's verified PRODUCTION prefix (`X.HF_PREFIX`) —
    `cfg.hf_prefix` is the UPLOAD prefix (smoke-suffixed under --smoke) and is
    never a staging source. arm_registry.json lives at the PREFIX ROOT, not
    under inputs/ (epm:consistency v2 note 2); the reduce consumes it for the
    unit-pin identity check. One bounded restage per base tree (plan §7 kill
    criterion (b) fires as a fail-loud RuntimeError after the retry).
    """
    _phase("pnf_stage")
    _status(cfg, "pnf_stage")
    from explore_persona_space.orchestrate import hub

    sample_path = cfg.out_root / "inputs" / "corpus_sample.json"
    if not sample_path.exists():
        hub.stage_hub_file(X.HF_DATA_REPO, f"{X.HF_PREFIX}/inputs/corpus_sample.json", sample_path)
    reg_path = cfg.out_root / "arm_registry.json"
    if not reg_path.exists():
        hub.stage_hub_file(X.HF_DATA_REPO, f"{X.HF_PREFIX}/arm_registry.json", reg_path)
    base_dirs: dict[str, Path] = {}
    for bu in sorted({X.base_unit_for(u) for u in _pnf_units(cfg)}):
        local = cfg.out_root / "corpus_capture" / bu
        if _pnf_base_incomplete(local) is None:
            logger.info("[pnf] %s: complete local base tree reused", bu)
            base_dirs[bu] = local
            continue
        staged = _pnf_stage_tree(cfg, f"corpus_capture/{bu}")
        reason = _pnf_base_incomplete(staged)
        if reason is not None:
            logger.info("[pnf] %s staged tree incomplete (%s) — one restage", bu, reason)
            staged = _pnf_stage_tree(cfg, f"corpus_capture/{bu}")
            reason = _pnf_base_incomplete(staged)
        if reason is not None:
            raise RuntimeError(f"[pnf] base tree {bu} incomplete after restage: {reason}")
        base_dirs[bu] = staged
    _pnf_dir(cfg).mkdir(parents=True, exist_ok=True)
    _atomic_json(
        _pnf_dir(cfg) / "staging_done.json",
        {"base_dirs": {k: str(v) for k, v in base_dirs.items()}, **_meta()},
    )
    return base_dirs


def _pnf_subsample(cfg: Cfg) -> tuple[list[str], str]:
    """Deterministic seed-42 TRAIN-sha subsample (plan v7 §4.2).

    Returns (shas in draw order, sha256 of the SORTED sha list — the regime
    key's subsample fingerprint). Train shas are unique + valtest-disjoint by
    the parent r4 postcondition; asserted here fail-loud.
    """
    sample = X.load_corpus_sample(cfg.out_root)
    n_train = sample["n_train"]
    train_rows = sample["rows"][:n_train]
    n_sub = min(PNF_N_ROWS, n_train)
    idxs = random.Random(X.SAMPLE_SEED).sample(range(n_train), n_sub)
    shas = [train_rows[i]["sha"] for i in idxs]
    assert len(set(shas)) == n_sub, "pnf subsample shas not unique (r4 postcondition violated)"
    key = hashlib.sha256("\n".join(sorted(shas)).encode("utf-8")).hexdigest()
    return shas, key


def _pnf_regime(cfg: Cfg, unit_id: str, replicate: str, sub_sha: str) -> dict:
    """The output-affecting resume/regime key (plan v7 §4.3). git_commit is
    RECORDED in metadata but deliberately EXCLUDED here — a crash-fix commit
    must not refuse resuming byte-identical completed stores."""
    r2 = replicate == "r2"
    return {
        "unit_id": unit_id,
        "replicate": replicate,
        "tf_batch": PNF_R2_TF_BATCH if r2 else cfg.tf_batch,
        "row_order": f"perm_seed_{X.FLOOR_SEED}" if r2 else "base_tree",
        "layers": list(cfg.layers),
        "subsample_sha256": sub_sha,
    }


def _pnf_rows_for_unit(base_dir: Path, shas: list[str]) -> list[dict]:
    """Base-tree rows (spans re-joined) filtered to the subsample shas.

    Exactly one row per sha: train shas are unique in the base tree and
    disjoint from the duplicate-bearing pinned valtest block (r4
    postcondition) — a shortfall is the plan §7 kill criterion (b).
    """
    rows = _read_rows_with_spans(base_dir)
    keep = set(shas)
    out = [r for r in rows if r["prompt_sha"] in keep]
    assert len(out) == len(keep) == len({r["prompt_sha"] for r in out}), (
        f"pnf sha-join incomplete: {len(out)} rows for {len(keep)} subsample shas "
        f"under {base_dir} (kill criterion b)"
    )
    return out


def _gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def run_noise_floor_unit(
    cfg: Cfg, unit_id: str, base_dir: Path, shas: list[str], sub_sha: str, tag: str = ""
) -> list[float]:
    """TWO replicate teacher-forced captures for one unit (plan v7 §4.3).

    r1 = base-tree row order at the production tf_batch (cfg.tf_batch, default
    8); r2 = seed-1768 row permutation at tf_batch=13. Byte-identical
    `_teacher_forced_span_means` kwargs to the parent p3 call except
    tf_batch_size (plan §10 call-shape bind). Resume is REGIME-KEYED: an
    existing store is reused only on an exact regime match, refused loud
    otherwise (#722-r3/#952 stale-resume class). Returns the CAPTURE walls
    (seconds) of the replicates actually run — model resolution/staging
    excluded, so kill criterion (a) gates on capture cost, not download cost.
    """
    import torch

    unit_dir = _pnf_dir(cfg) / unit_id
    unit_dir.mkdir(parents=True, exist_ok=True)
    base_unit = X.base_unit_for(unit_id)
    pending: list[tuple[str, Path, dict]] = []
    for rep in PNF_REPLICATES:
        store_path = unit_dir / f"pooled_nf_{rep}.pt"
        regime = _pnf_regime(cfg, unit_id, rep, sub_sha)
        if store_path.exists():
            # weights_only=False: self-produced sha-pinned store (parent
            # convention; _meta()'s torch version is a TorchVersion object)
            meta = torch.load(store_path, map_location="cpu", mmap=True, weights_only=False)[
                "metadata"
            ]
            if meta.get("regime") != regime:
                raise RuntimeError(
                    f"[pnf] {store_path} holds a DIFFERENT regime — refusing stale reuse "
                    f"(#722-r3 class): stored={meta.get('regime')} vs current={regime}"
                )
            logger.info("[pnf] %s %s: complete store reused (regime match)", unit_id, rep)
            continue
        pending.append((rep, store_path, regime))
    if not pending:
        return []
    rows = _pnf_rows_for_unit(base_dir, shas)
    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    model_path, cleanup = _resolve_unit_model(cfg, unit_id)
    walls: list[float] = []
    try:
        for rep, store_path, regime in pending:
            rep_rows = list(rows)
            if rep == "r2":
                random.Random(X.FLOOR_SEED).shuffle(rep_rows)
            t0 = time.time()
            pooled = _teacher_forced_span_means(
                model_path,
                rep_rows,
                [base_unit],
                layers=list(cfg.layers),
                spans=("response",),
                device=_device(),
                dtype=_dtype(),
                tf_batch_size=regime["tf_batch"],
            )
            wall = time.time() - t0
            walls.append(wall)
            store = {
                "schema_version": 1,
                "unit": unit_id,
                "replicate": rep,
                "row_sha": [r["prompt_sha"] for r in rep_rows],
                "row_question_idx": [r["question_idx"] for r in rep_rows],
                "arms": {
                    span: {li: t.to(torch.float16) for li, t in per.items()}
                    for span, per in pooled.items()
                },
                "metadata": {
                    **_meta(),
                    "regime": regime,
                    "model_path": model_path,
                    "gpu_name": _gpu_name(),  # arch-dependent floor (consistency v2 note 1)
                    "shared_text_from": base_unit,
                    "spans": ["response"],
                    "n_rows": len(rep_rows),
                    "capture_wall_s": wall,
                    "smoke": cfg.smoke,
                },
            }
            tmp = store_path.with_suffix(".pt.tmp")
            torch.save(store, tmp)
            os.replace(tmp, store_path)
            print(
                f"[pnf] {tag}{unit_id}:{rep} rows={len(rep_rows)} "
                f"tf_batch={regime['tf_batch']} capture={wall:.0f}s",
                flush=True,
            )
    finally:
        _cleanup_merged(cleanup)
    return walls


def _pnf_wall_gate(unit_id: str, wall_s: float, first_wall_s: float) -> None:
    """Kill criterion (a), plan v7 §7: a unit's CAPTURE wall > 3x the first
    completed unit's (the first unit IS the in-run pilot). Fail-loud designed
    halt AFTER the unit's stores persisted — a resume skips completed work.
    Relative to the first unit at the SAME n, so smoke-scale-safe (#1345)."""
    if wall_s > 3.0 * first_wall_s:
        raise RuntimeError(
            f"[pnf] kill criterion (a): {unit_id} capture wall {wall_s:.0f}s > "
            f"3x first-unit wall {first_wall_s:.0f}s"
        )


def _pnf_replicate_distances(unit_dir: Path, layers: list[int]):
    """Per-context replicate distances for one unit: fp32 ||v_r1 - v_r2||_2
    per layer, rows joined by prompt_sha (r2 is a permutation of r1's rows)."""
    import torch

    stores = {}
    for rep in PNF_REPLICATES:
        s = torch.load(unit_dir / f"pooled_nf_{rep}.pt", map_location="cpu", weights_only=False)
        stores[rep] = s
    shas1, shas2 = stores["r1"]["row_sha"], stores["r2"]["row_sha"]
    assert set(shas1) == set(shas2), (unit_dir, "replicate sha sets differ")
    pos2 = {s: i for i, s in enumerate(shas2)}
    idx = torch.tensor([pos2[s] for s in shas1])
    dists = {}
    for li in layers:
        v1 = stores["r1"]["arms"]["response"][li].float()
        v2 = stores["r2"]["arms"]["response"][li].float()
        dists[li] = torch.linalg.vector_norm(v1 - v2[idx], dim=1)
    metas = {rep: stores[rep]["metadata"] for rep in PNF_REPLICATES}
    return shas1, dists, metas


def _pnf_same_pass_anchor(base_dir: Path, layers: list[int]) -> dict:
    """Zero-GPU sanity anchor (epm:plan-critique v1 concern 2 / brief note 3):
    the frozen valtest block carries duplicate prompt shas whose rows sit
    per-row in the parent base store — pairs restricted to BYTE-IDENTICAL
    response token ids give WITHIN-pass replicate distances. Same-pass +
    mixed-batch-position only ⇒ a LOWER-bound anchor (under-covers
    pass-to-pass sources: model reload, kernel algo selection); never a
    substitute for the replicate floor."""
    import torch

    rows = _read_shards(base_dir)
    by_sha: dict[str, list[dict]] = {}
    for r in rows:
        by_sha.setdefault(r["prompt_sha"], []).append(r)
    store = torch.load(base_dir / "pooled.pt", map_location="cpu", weights_only=False)
    pos = {
        (s, q): i
        for i, (s, q) in enumerate(zip(store["row_sha"], store["row_question_idx"], strict=True))
    }
    resp = store["arms"]["response"]
    pair_d: dict[int, list[float]] = {li: [] for li in layers}
    n_groups = n_pairs = n_text_mismatch = 0
    for sha, grp in by_sha.items():
        if len(grp) < 2:
            continue
        n_groups += 1
        for a, b in itertools.combinations(grp, 2):
            if a["response_token_ids"] != b["response_token_ids"]:
                n_text_mismatch += 1  # independent generations — measures gen
                continue  # variability, not capture noise (validity caveat a)
            ia = pos.get((sha, a["question_idx"]))
            ib = pos.get((sha, b["question_idx"]))
            if ia is None or ib is None:
                continue  # dropped at span time — stays dropped
            n_pairs += 1
            for li in layers:
                d = torch.linalg.vector_norm(resp[li][ia].float() - resp[li][ib].float())
                pair_d[li].append(float(d))
    import numpy as np

    per_layer = {
        str(li): (
            {
                "median": float(np.median(pair_d[li])),
                "p95": float(np.percentile(pair_d[li], 95)),
                "n_pairs": len(pair_d[li]),
            }
            if pair_d[li]
            else None
        )
        for li in layers
    }
    return {
        "kind": "same_pass_mixed_geometry",
        "validity": (
            "pairs from ONE production pass (tree order, production tf_batch): same batch "
            "geometry regime but different batch positions; byte-identical-response pairs "
            "only (differing-text pairs measure generation variability, not capture noise); "
            "under-covers pass-to-pass sources (model reload, kernel algo selection) — a "
            "LOWER-bound anchor for the replicate floor"
        ),
        "n_duplicate_sha_groups": n_groups,
        "n_identical_response_pairs": n_pairs,
        "n_pairs_response_text_differs": n_text_mismatch,
        "per_layer": per_layer,
    }


def _pnf_floor_at(per_layer_p95: dict[str, float], layer: int) -> tuple[float, bool]:
    """(floor_p95, layer_matched). Production layers always match the fits'
    (14/19/25 both sides); the smoke's tiny-model layers never do — fall back
    to the max available floor so the reduce path stays fully exercised."""
    key = str(layer)
    if key in per_layer_p95:
        return per_layer_p95[key], True
    return max(per_layer_p95.values()), False


def _pnf_verdict(ratio: float | None) -> str:
    """Plan v7 §3 disjoint bands on r = shift/floor_p95 (None = zero floor)."""
    if ratio is None:
        return "clear-degenerate-floor"
    if ratio <= 2.0:
        return "noise-ordered"
    if ratio < 10.0:
        return "above-floor"
    return "clear"


def _pnf_primary_layer(arm_id: str) -> int:
    return 25 if arm_id.startswith("mk-") else 19  # v5 §3: L25 marker / L19 content


def noise_floor_reduce(
    cfg: Cfg, fits_dir: Path | None = None, results_dir: Path | None = None
) -> dict:
    """pnf_reduce (CPU): floors + per-context distances + 72-arm ratio table +
    the H3-clause / marker-falsification verdicts (plan v7 §4.4 + §6)."""
    _phase("pnf_reduce")
    _status(cfg, "pnf_reduce")
    import numpy as np

    fits_dir = fits_dir or (REPO_ROOT / "eval_results" / "issue_1768" / "fits")
    results_dir = results_dir or _pnf_results_dir(cfg)
    units = _pnf_units(cfg)
    layers = list(cfg.layers)
    shas_sub, sub_sha = _pnf_subsample(cfg)

    percontext_dir = results_dir / "noise_floor_percontext"
    percontext_dir.mkdir(parents=True, exist_ok=True)
    floors: dict[str, dict[str, dict]] = {}
    floors_p95: dict[str, dict[str, float]] = {}
    gpu_names: dict[str, dict[str, str]] = {}
    degenerate: list[list] = []
    for unit in units:
        shas, dists, metas = _pnf_replicate_distances(_pnf_dir(cfg) / unit, layers)
        floors[unit], floors_p95[unit] = {}, {}
        gpu_names[unit] = {rep: metas[rep].get("gpu_name", "?") for rep in PNF_REPLICATES}
        for li in layers:
            d = dists[li].numpy()
            p95, med = float(np.percentile(d, 95)), float(np.median(d))
            floors[unit][str(li)] = {"p95": p95, "median": med, "n": int(d.size)}
            floors_p95[unit][str(li)] = p95
            if p95 == 0.0:
                degenerate.append([unit, li])
            _atomic_json(
                percontext_dir / f"{unit}_L{li}.json",
                {
                    "unit": unit,
                    "layer": li,
                    "row_sha": list(shas),
                    "distance": [float(x) for x in d],
                    "regimes": {rep: metas[rep]["regime"] for rep in PNF_REPLICATES},
                },
            )
    fleet_p95 = {str(li): max(floors_p95[u][str(li)] for u in units) for li in layers}
    spread = {
        str(li): {
            "max_over_min": (
                max(floors_p95[u][str(li)] for u in units)
                / max(1e-12, min(floors_p95[u][str(li)] for u in units))
            ),
        }
        for li in layers
    }
    for v in spread.values():
        v["flagged_gt_2x"] = bool(v["max_over_min"] > 2.0)

    # unit-pin identity check against the staged realized arm registry (§4.1)
    reg_path = cfg.out_root / "arm_registry.json"
    reg_sha = hashlib.sha256(reg_path.read_bytes()).hexdigest()
    reg_arms = {r["arm_id"] for r in json.loads(reg_path.read_text())["arms"]}
    missing_units = [u for u in units if not u.startswith("base_") and u not in reg_arms]
    assert not missing_units, f"pnf units missing from arm_registry: {missing_units}"

    ratio_rows: list[dict] = []
    shift_at: dict[tuple[str, int], float] = {}
    for f in sorted(fits_dir.glob("*_L*.json")):
        d = json.loads(f.read_text())
        arm, li = d["arm_id"], int(d["layer"])
        shift = float(d["decomposition_tf"]["mean_norm_total"])
        shift_at[(arm, li)] = shift
        own = arm in floors_p95
        floor, matched = _pnf_floor_at(floors_p95[arm] if own else fleet_p95, li)
        ratio = None if floor == 0.0 else shift / floor
        ratio_rows.append(
            {
                "arm_id": arm,
                "layer": li,
                "shift_mean_norm_total": shift,
                "floor_p95": floor,
                "floor_source": ("own" if own else "fleet")
                + ("" if matched else "-layer-fallback"),
                "ratio": ratio,
                "verdict": _pnf_verdict(ratio),
            }
        )
    arm_ids = sorted({a for a, _ in shift_at})

    # criterion 1 (§6): fraction of arms with shift > fleet floor at the
    # arm's PRIMARY layer (conservative fleet-max floor)
    n_above = 0
    for arm in arm_ids:
        li = _pnf_primary_layer(arm)
        shift = shift_at.get((arm, li))
        assert shift is not None, f"fits missing {arm} at primary layer L{li}"
        floor, _m = _pnf_floor_at(fleet_p95, li)
        n_above += int(floor == 0.0 or shift > floor)
    h3_frac = n_above / len(arm_ids) if arm_ids else 0.0

    # criterion 2 (§6): marker falsification — mk arm's OWN-floor ratio at its
    # primary layer; fleet read over every marker arm as the companion
    mk = "mk-pers-con-lr5e6-s42"
    mk_ratio = None
    if mk in floors_p95 and (mk, 25) in shift_at:
        fl, _m = _pnf_floor_at(floors_p95[mk], 25)
        mk_ratio = None if fl == 0.0 else shift_at[(mk, 25)] / fl
    marker_ratios = [
        r["ratio"]
        for r in ratio_rows
        if r["arm_id"].startswith("mk-")
        and r["layer"] == _pnf_primary_layer(r["arm_id"])
        and r["ratio"] is not None
    ]
    out = {
        "_meta": {
            **_meta(),
            "row_basis_note": (
                "floor measured on the 2,000-row seed-42 TRAIN subsample; the compared shift "
                "(decomposition_tf.mean_norm_total) was measured on the 1,000 pinned TEST rows "
                "— registered in plan v7 §4.4 conservatism (b) + assumption 4: both row sets "
                "are draws from the same corpus distribution and the comparison is a scale "
                "read, not row-paired"
            ),
            "conservatisms": (
                "(a) shift is a MEAN while the floor is a p95 — floor errs high; (b) the "
                "shift carries ~sqrt(2)x single-capture noise from two independent captures "
                "— absorbed inside the 2x falsification band (plan v7 §4.4)"
            ),
            "subsample": {"n": len(shas_sub), "seed": X.SAMPLE_SEED, "sha256": sub_sha},
            "replicate_design": {
                "r1": f"base-tree row order, tf_batch={cfg.tf_batch} (production geometry)",
                "r2": f"perm seed {X.FLOOR_SEED}, tf_batch={PNF_R2_TF_BATCH}",
            },
            "gpu_names": gpu_names,
            "arm_registry_sha256": reg_sha,
            "smoke": cfg.smoke,
        },
        "floors": floors,
        "fleet_floor_p95": fleet_p95,
        "floor_spread": spread,
        "degenerate_zero_floors": degenerate,
        "degenerate_floor_note": (
            "a floor_p95 of exactly 0 means the replicate pair is bit-identical there; the "
            "operative floor is then the fp16 storage quantization step (~1e-3 relative) — "
            "recorded per plan v7 §8, never silently substituted"
        )
        if degenerate
        else None,
        "ratio_table": ratio_rows,
        "criteria": {
            "h3_n_arms": len(arm_ids),
            "h3_frac_above_fleet_floor_primary": h3_frac,
            "h3_met": bool(arm_ids) and h3_frac >= 0.90,
            "mk_own_floor_ratio_primary": mk_ratio,
            "marker_falsified": (mk_ratio is not None and mk_ratio <= 2.0),
            "fleet_marker_min_ratio_primary": min(marker_ratios) if marker_ratios else None,
        },
        "same_pass_anchor": _pnf_same_pass_anchor(
            _pnf_resolved_base_dir(cfg, "base_content"), layers
        ),
    }
    _atomic_json(results_dir / "capture_noise_floor.json", out)
    logger.info(
        "[pnf] reduce: h3_frac=%.3f mk_ratio=%s degenerate=%d -> %s",
        h3_frac,
        f"{mk_ratio:.2f}" if mk_ratio is not None else "n/a",
        len(degenerate),
        results_dir / "capture_noise_floor.json",
    )
    return out


def _pnf_upload(cfg: Cfg) -> None:
    """pnf_upload: whole noise_floor tree (stores + reduce mirror) in one
    fail-loud upload_folder commit + exact-set verify (the p7 conventions)."""
    _phase("pnf_upload")
    _status(cfg, "pnf_upload")
    if not cfg.upload:
        logger.info("[pnf] upload disabled (--no-upload)")
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    res = _pnf_results_dir(cfg)
    mirror = _pnf_dir(cfg) / "reduce"
    mirror.mkdir(parents=True, exist_ok=True)
    shutil.copy2(res / "capture_noise_floor.json", mirror / "capture_noise_floor.json")
    shutil.copytree(
        res / "noise_floor_percontext", mirror / "noise_floor_percontext", dirs_exist_ok=True
    )
    dest = _upload_tree(cfg, "noise_floor")
    expected = [
        f"{cfg.hf_prefix}/noise_floor/{u}/pooled_nf_{rep}.pt"
        for u in _pnf_units(cfg)
        for rep in PNF_REPLICATES
    ]
    expected.append(f"{cfg.hf_prefix}/noise_floor/reduce/capture_noise_floor.json")
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        X.HF_DATA_REPO,
        expected,
        path_in_repo=f"{cfg.hf_prefix}/noise_floor",
        repo_type="dataset",
    )
    assert not missing, f"pnf upload verify: {len(missing)} files missing on Hub: {missing[:5]}"
    _atomic_json(
        _pnf_dir(cfg) / "upload_done.json", {"dest": dest, "n_verified": len(expected), **_meta()}
    )


def phase_noise_floor(cfg: Cfg) -> None:
    """pnf: matched-text capture-noise floor (plan v7 amendment) — stage ->
    2 replicate TF captures x 6 units (ONE GPU, sequential; §9 deliberate) ->
    CPU reduce (floor + ratio table + verdicts) -> upload."""
    _phase("pnf")
    _status(cfg, "pnf")
    base_dirs = _pnf_stage(cfg)
    shas, sub_sha = _pnf_subsample(cfg)
    units = _pnf_units(cfg)
    first_wall: float | None = None
    for k, unit in enumerate(units):
        t0 = time.time()
        walls = run_noise_floor_unit(
            cfg, unit, base_dirs[X.base_unit_for(unit)], shas, sub_sha, tag=f"{k + 1}/{len(units)} "
        )
        print(f"[pnf] unit {k + 1}/{len(units)} {unit} elapsed={time.time() - t0:.0f}s", flush=True)
        _status(cfg, "pnf_capture", done=k + 1, total=len(units))
        if walls:  # resumed-only units carry no capture wall
            capture_wall = sum(walls)
            if first_wall is None:
                first_wall = capture_wall
            else:
                _pnf_wall_gate(unit, capture_wall, first_wall)
    noise_floor_reduce(cfg)
    _pnf_upload(cfg)


# ── pfx: on-target prefixed capture (plan v8 round 3) ────────────────────────
#
# pfx0  derived-corpus build (CPU): pfx subsample (3,000 train + pinned
#       val/test), per-condition manifests + token budgets, and the §4.2
#       prefix-vs-mix GROUND-TRUTH assert (kill criterion b).
# pfx1  pilot: the syc pers con s42 arm @ own, full rows, production shape;
#       gen + TF walls measured separately (kill criterion a).
# pfx2  prefixed capture (23 units: 12 own + 6 ctrl + 5 base@prefix); spans
#       {prefix, context, response}; regime-keyed resume.
# pfx3  matched-text TF trees (18 units) on the SAME-condition base rows.
# pfx4  on_target/ tree upload + exact-set verify (BEFORE fits; #825).

PFX_BOOKED_UNIT_GPU_H = 1.0  # plan §9 pfx1 row (round-1 measured x4.4/16.4 x1.5 prefix book)
PFX_PILOT_ARM = "syc-pers-con-lr1e5-s42"  # plan §4.4 pfx1 (smoke: X.PILOT_ARM)
# ONE sha-pinned training mix per PREFIX family for the §4.2 ground-truth
# assert (the arm whose registry mix record anchors the family's context).
PFX_MIX_FAMILY_ARM = {
    "pers": "syc-pers-con-lr1e5-s42",
    "conv": "syc-conv-con-lr1e5-s42",
    "icl_syc": "syc-icl-con-lr1e5-s42",
}
PFX_UPLOAD_TREES = (
    "on_target/inputs",
    "on_target/pilot",
    "on_target/corpus_capture",
    "on_target/corpus_capture_tf",
)
PFX_SPANS = ("prefix", "context", "response")  # +prefix vs round 1 (plan §4.4)


def _pfx_root(cfg: Cfg) -> Path:
    return cfg.out_root / "on_target"


def _pfx_inputs(cfg: Cfg) -> Path:
    return _pfx_root(cfg) / "inputs"


def _pfx_arms(cfg: Cfg) -> list[str]:
    """The round's arm scope (plan §4.1 order); --arms filters INSIDE it."""
    if cfg.arms:
        want = set(cfg.arms)
        unknown = want - set(X.PFX_ARMS)
        assert not unknown, f"--arms outside the pfx arm set: {sorted(unknown)}"
        return [a for a in X.PFX_ARMS if a in want]
    if cfg.smoke:
        return [X.PILOT_ARM]
    return list(X.PFX_ARMS)


def _pfx_conds_for(cfg: Cfg, arm_id: str) -> tuple[str, ...]:
    # smoke covers ONE trained condition (own); ctrl shares the same code path
    return ("own",) if cfg.smoke else X.pfx_conditions_for(arm_id)


def _pfx_unit_sets(cfg: Cfg) -> tuple[list[str], list[str]]:
    """(pfx2 unit ids incl. bases, pfx3 trained-unit ids)."""
    arms = _pfx_arms(cfg)
    trained = [X.pfx_trained_unit(a, c) for a in arms for c in _pfx_conds_for(cfg, a)]
    bases = sorted({X.pfx_base_unit(a, c) for a in arms for c in _pfx_conds_for(cfg, a)})
    if cfg.smoke:
        # Regime coverage under smoke (plan §4 smoke parity): one PLAIN-TEXT
        # boundary render (the ICL user_wrap — the #1315 span-seam class) and
        # one mk-decode unit ride alongside the pilot arm's own condition.
        for extra in ("base_content@icl_syc", "base_mk@pers"):
            if extra not in bases:
                bases.append(extra)
    return bases + trained, trained


def _pfx_condition_ids(cfg: Cfg) -> list[str]:
    """pfx0 ALWAYS builds the FULL production condition set (cheap CPU-only
    renders + one small mix file per family), so the §4.2 byte assert covers
    every family even when --smoke/--arms narrows the capture units."""
    units, _ = _pfx_unit_sets(cfg)
    ids = {X.pfx_unit_context_id(u) for u in units}
    ids.update(X.pfx_unit_context_id(u) for u in X.pfx_base_units())
    return sorted(ids)


def _pfx_prefix_recipe(ctx) -> dict:
    return {
        "context_id": ctx.context_id,
        "system": ctx.system,
        "prefix_turns": [dict(t) for t in ctx.prefix_turns],
        "user_wrap": ctx.user_wrap,
    }


def _pfx_prefix_sha(ctx) -> str:
    """sha256 of the prefix RECIPE (system + prefix turns + user_wrap) — the
    byte-level content identity every pfx regime key carries."""
    blob = json.dumps(_pfx_prefix_recipe(ctx), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _pfx_stage_inputs(cfg: Cfg) -> None:
    """Round-1 inputs, local-else-Hub (the pnf staging pattern; the fellows
    lane rsync-excludes eval_results/, so every repo input rides the Hub
    mirror — plan §9 lane note)."""
    from explore_persona_space.orchestrate import hub

    sample_path = cfg.out_root / "inputs" / "corpus_sample.json"
    if not sample_path.exists():
        hub.stage_hub_file(X.HF_DATA_REPO, f"{X.HF_PREFIX}/inputs/corpus_sample.json", sample_path)
    reg_path = cfg.out_root / "arm_registry.json"
    if not reg_path.exists():
        hub.stage_hub_file(X.HF_DATA_REPO, f"{X.HF_PREFIX}/arm_registry.json", reg_path)


def _build_pfx_sample(cfg: Cfg) -> dict:
    """Derived pfx sample (plan §4.3): `random.Random(42).sample(range(n_train),
    3000)` over the r4-deduped train rows + the FULL pinned val/test block —
    identical shas across every condition/arm by construction (one file).
    Splits are derived from the Hub-staged corpus_sample.json — NEVER
    `assert_pinned_split` (plan §9 lane note (ii))."""
    path = _pfx_inputs(cfg) / "corpus_sample_pfx.json"
    if path.exists():
        return X.load_pfx_sample(cfg.out_root)
    parent = X.load_corpus_sample(cfg.out_root)
    n_train_p, n_val, n_test = parent["n_train"], parent["n_val"], parent["n_test"]
    n_sub = min(X.PFX_N_TRAIN, n_train_p)
    idxs = random.Random(X.SAMPLE_SEED).sample(range(n_train_p), n_sub)
    rows = [{**parent["rows"][i], "src_qidx": i} for i in idxs]
    for j in range(n_val + n_test):
        rows.append({**parent["rows"][n_train_p + j], "src_qidx": n_train_p + j})
    train_shas = {r["sha"] for r in rows[:n_sub]}
    assert len(train_shas) == n_sub, "pfx train subsample shas not unique (r4 postcondition)"
    sub_sha = hashlib.sha256(
        "\n".join(sorted(r["sha"] for r in rows[:n_sub])).encode("utf-8")
    ).hexdigest()
    sample = {
        "rows": rows,
        "n_train": n_sub,
        "n_val": n_val,
        "n_test": n_test,
        "parent_n_train": n_train_p,
        "train_subsample_sha256": sub_sha,
        "sample_seed": X.SAMPLE_SEED,
        "smoke": cfg.smoke,
        **_meta(),
    }
    _atomic_json(path, sample)
    return sample


def _pfx_budget(tok, ctx, rows: list[dict]) -> dict:
    """Per-condition token budgets (plan §4.4 pfx0): realized rendered-prompt
    token lengths over ALL rows (via the ONE construction path,
    `_build_generation_prompts`) + `max_new_tokens` per decode class vs
    MAX_MODEL_LEN. Content overflow FAILS LOUD (re-plan); mk overflow raises
    to PFX_MAX_MODEL_LEN_RAISED for those units only (recorded deviation)."""
    from explore_persona_space.analysis.representation_shift import _build_generation_prompts

    prompts = [r["prompt"] for r in rows]
    rendered, _keys = _build_generation_prompts(
        tok,
        {ctx.context_id: ctx.system},
        prompts,
        user_wraps={ctx.context_id: ctx.user_wrap},
        prior_turns={ctx.context_id: tuple(ctx.prefix_turns)},
    )
    lens = [len(ids) for ids in tok(rendered, add_special_tokens=False)["input_ids"]]
    max_prompt = max(lens)
    budgets = {}
    for decode, max_new in (("content", X.MAX_NEW_CONTENT), ("mk", X.MAX_NEW_MARKER)):
        need = max_prompt + max_new
        mml, raised = X.MAX_MODEL_LEN, False
        if need > mml:
            assert decode == "mk", (
                f"pfx0 content-decode budget overflow under {ctx.context_id}: "
                f"{need} > {mml} (plan §7 — re-plan, no silent raise)"
            )
            mml, raised = X.PFX_MAX_MODEL_LEN_RAISED, True
            assert need <= mml, (
                f"pfx0 mk-decode budget overflow even at {mml} under {ctx.context_id}: {need}"
            )
            logger.info(
                "[pfx0] %s mk decode: MAX_MODEL_LEN raised %d -> %d (recorded deviation)",
                ctx.context_id,
                X.MAX_MODEL_LEN,
                mml,
            )
        budgets[decode] = {
            "max_model_len": mml,
            "raised": raised,
            "max_prompt_tokens": max_prompt,
            "max_new_tokens": max_new,
        }
    return budgets


def _assert_mix_row_matches_context(ctx, msgs: list[dict], tag: str) -> None:
    """Kill criterion (b): the registry-rendered prefix must equal the TRAINED
    context embedded in the mix positives — byte equality, never a paraphrase
    (plan §4.2 ground-truth assert)."""
    sys_msgs = [m for m in msgs if m["role"] == "system"]
    mix_system = sys_msgs[0]["content"] if sys_msgs else None
    assert mix_system == (ctx.system or None), (tag, "system-prompt drift vs registry render")
    chat = [m for m in msgs if m["role"] != "system"]
    assert chat and chat[-1]["role"] == "user", (tag, "mix row has no final user turn")
    prior = [(m["role"], m["content"]) for m in chat[:-1]]
    want = [(t["role"], t["content"]) for t in ctx.prefix_turns]
    assert prior == want, (tag, "prefix-turn drift vs registry render", len(prior), len(want))
    if ctx.user_wrap is not None:
        head, tail = ctx.user_wrap.split("{q}", 1)
        content = chat[-1]["content"]
        assert content.startswith(head) and content.endswith(tail), (
            tag,
            "user_wrap drift vs registry render",
        )


def _pfx_prefix_vs_mix_assert(cfg: Cfg) -> dict:
    """Stage ONE training mix per prefix family and byte-assert the registry
    render against the context embedded in its positive rows (plan §4.2)."""
    from explore_persona_space.orchestrate import hub

    reg = json.loads((cfg.out_root / "arm_registry.json").read_text())
    need = {X.pfx_prefix_tag(cid): cid for cid in _pfx_condition_ids(cfg)}
    out = {}
    for tag, cid in sorted(need.items()):
        fam_arm = PFX_MIX_FAMILY_ARM[tag]
        src = reg["mix_pos_sources"][fam_arm]
        local = _pfx_inputs(cfg) / "mix_assert" / tag / Path(src["pos_path"]).name
        if not local.exists():
            hub.stage_hub_file(X.HF_DATA_REPO, src["pos_path"], local, repo_type="dataset")
        pos_sha256 = hashlib.sha256(local.read_bytes()).hexdigest()
        ctx = X.pfx_resolve_context(cid)
        checked = 0
        with local.open(encoding="utf-8") as fh:  # text-mode iteration, never splitlines
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if src["layout"] == "marker-mix" and "※" not in _completion_text(r):
                    continue  # marker mixes interleave negatives; positives only
                msgs = (
                    r["prompt"]
                    if isinstance(r["prompt"], list)
                    else [{"role": "user", "content": r["prompt"]}]
                )
                _assert_mix_row_matches_context(ctx, msgs, f"{tag}/{fam_arm}")
                checked += 1
                if checked >= 5:
                    break
        assert checked > 0, (tag, src["pos_path"], "no positive rows checked")
        out[tag] = {
            "context_id": cid,
            "family_arm": fam_arm,
            "pos_path": src["pos_path"],
            "pos_sha256": pos_sha256,
            "n_rows_checked": checked,
        }
        logger.info("[pfx0] prefix-vs-mix assert PASS: %s (%d rows)", tag, checked)
    return out


def phase_pfx0(cfg: Cfg) -> None:
    """pfx0: derived corpora + condition manifests + budgets + mix assert."""
    _phase("pfx0_corpus_build")
    _status(cfg, "pfx0_corpus_build")
    from transformers import AutoTokenizer

    done = _pfx_inputs(cfg) / "build_done.json"
    _pfx_stage_inputs(cfg)
    sample = _build_pfx_sample(cfg)
    if done.exists():
        logger.info("[pfx0] build_done.json present — resume skip")
        return
    tok = AutoTokenizer.from_pretrained(cfg.model_override or X.BASE_MODEL)
    conds: dict[str, dict] = {}
    for cid in _pfx_condition_ids(cfg):
        ctx = X.pfx_resolve_context(cid)
        tag = X.pfx_prefix_tag(cid)
        budgets = _pfx_budget(tok, ctx, sample["rows"])
        prefix_tokens = len(
            tok(ctx.render(tok, ""), add_special_tokens=False)["input_ids"]
        )  # prefix + template overhead at an empty query (manifest read)
        rec = {
            "tag": tag,
            "context_id": cid,
            "prefix_sha256": _pfx_prefix_sha(ctx),
            "recipe": _pfx_prefix_recipe(ctx),
            "prefix_tokens": prefix_tokens,
            "message_shape": {
                "has_system": ctx.system is not None,
                "n_prefix_turns": len(ctx.prefix_turns),
                "has_user_wrap": ctx.user_wrap is not None,
            },
            "budgets": budgets,
            "n_rows": len(sample["rows"]),
        }
        conds[cid] = rec
        # The derived per-condition corpus manifest: the ROWS live once in
        # corpus_sample_pfx.json (identical shas across conditions — paired
        # design); this file pins the condition's recipe + realized budgets.
        _atomic_json(
            _pfx_inputs(cfg) / f"corpus_{tag}.json",
            {
                **rec,
                "rows_sha256": hashlib.sha256(
                    "\n".join(r["sha"] for r in sample["rows"]).encode("utf-8")
                ).hexdigest(),
                **_meta(),
            },
        )
    _atomic_json(_pfx_inputs(cfg) / "conditions.json", {"conditions": conds, **_meta()})
    mix_rec = _pfx_prefix_vs_mix_assert(cfg)  # kill criterion (b) fires here
    _atomic_json(_pfx_inputs(cfg) / "prefix_mix_assert.json", {"families": mix_rec, **_meta()})
    _atomic_json(
        done,
        {
            "n_rows": len(sample["rows"]),
            "n_train": sample["n_train"],
            "conditions": sorted(conds),
            **_meta(),
        },
    )
    logger.info("[pfx0] corpora + budgets + mix assert done (%d conditions)", len(conds))


def _pfx_cond_record(cfg: Cfg, context_id: str) -> dict:
    conds = json.loads((_pfx_inputs(cfg) / "conditions.json").read_text())["conditions"]
    assert context_id in conds, (context_id, "condition not built at pfx0", sorted(conds))
    return conds[context_id]


def _pfx_regime(cfg: Cfg, unit_id: str, cond_rec: dict, sample: dict, spans: tuple) -> dict:
    """Output-affecting resume/regime key for a pfx unit (plan §4.4: condition
    + prefix sha in the key; a mismatch is refused loud — #722-r3 class)."""
    return {
        "unit_id": unit_id,
        "context_id": cond_rec["context_id"],
        "prefix_sha256": cond_rec["prefix_sha256"],
        "layers": list(cfg.layers),
        "tf_batch": cfg.tf_batch,
        "spans": list(spans),
        "train_subsample_sha256": sample["train_subsample_sha256"],
        "n_rows": len(sample["rows"]),
    }


def _pfx_check_regime(unit_dir: Path, regime: dict) -> None:
    path = unit_dir / "regime.json"
    if path.exists():
        stored = json.loads(path.read_text())["regime"]
        if stored != regime:
            raise RuntimeError(
                f"[pfx] {unit_dir} holds a DIFFERENT regime — refusing stale reuse "
                f"(#722-r3 class): stored={stored} vs current={regime}"
            )
        return
    unit_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, {"regime": regime, **_meta()})


def _pfx_decode_class(unit_id: str) -> str:
    return "mk" if unit_id.split("@")[0].startswith(("mk-", "base_mk")) else "content"


def run_pfx_corpus_unit(cfg: Cfg, unit_id: str) -> None:
    """pfx2 unit: prefixed greedy gen -> TF span-means (prefix+ctx+resp)."""
    cid = X.pfx_unit_context_id(unit_id)
    _prefixed_capture_core(
        cfg,
        unit_id,
        root=_pfx_root(cfg),
        cid=cid,
        cond_rec=_pfx_cond_record(cfg, cid),
        ctx=X.pfx_resolve_context(cid),
    )


def _prefixed_capture_core(
    cfg: Cfg, unit_id: str, *, root: Path, cid: str, cond_rec: dict, ctx
) -> None:
    """Shared pfx2/lad2 unit body: prefixed greedy gen -> spans -> TF span-means
    -> pooled.pt + manifest under ``root`` (extracted VERBATIM from the round-3
    ``run_pfx_corpus_unit``; round 4 threads the on_target_r4 root + ladder
    contexts through the SAME statements — no clone drift, #399 class)."""
    from transformers import AutoTokenizer

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    out_dir = root / "corpus_capture" / unit_id
    sample = X.load_pfx_sample(cfg.out_root)
    regime = _pfx_regime(cfg, unit_id, cond_rec, sample, PFX_SPANS)
    if (out_dir / "pooled.pt").exists():
        _pfx_check_regime(out_dir, regime)  # refuse a stale-regime store loud
        return
    _pfx_check_regime(out_dir, regime)
    max_model_len = cond_rec["budgets"][_pfx_decode_class(unit_id)]["max_model_len"]
    prompts = [r["prompt"] for r in sample["rows"]]
    model_path, cleanup = _resolve_unit_model(cfg, unit_id.split("@")[0])
    try:
        t0 = time.time()
        rows = _load_or_generate_rows(
            cfg,
            out_dir,
            unit_id,
            model_path,
            prompts=prompts,
            system=ctx.system,
            user_wrap=ctx.user_wrap,
            prior_turns=tuple(ctx.prefix_turns),
            max_model_len=max_model_len,
        )
        gen_wall = time.time() - t0
        _assert_model_dir_alive(model_path, cleanup)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        kept, dropped, seam_counts, n_prefix = _attach_spans(
            tokenizer,
            prompts,
            rows,
            system=ctx.system,
            user_wrap=ctx.user_wrap,
            prior_turns=tuple(ctx.prefix_turns),
        )
        valid_frac = len(kept) / max(1, len(rows))
        assert valid_frac >= 0.95, (unit_id, valid_frac, "plan §6 row-validity criterion")
        _atomic_json(
            out_dir / "rows_spans.json",
            {
                "rows": [
                    {
                        "prompt_sha": r["prompt_sha"],
                        "question_idx": r["question_idx"],
                        "prefix_len": r["prefix_len"],
                        "context_len": r["context_len"],
                    }
                    for r in kept
                ],
                "n_dropped_empty_response": dropped,
                "seam_counts": seam_counts,
                "n_distinct_prefix": n_prefix,
            },
        )
        t1 = time.time()
        _assert_model_dir_alive(model_path, cleanup)
        pooled = _teacher_forced_span_means(
            model_path,
            kept,
            [unit_id],
            layers=list(cfg.layers),
            spans=PFX_SPANS,
            device=_device(),
            dtype=_dtype(),
            tf_batch_size=cfg.tf_batch,
        )
        tf_wall = time.time() - t1
        fp16_cos = _fp16_roundtrip_cos_min(pooled)
        _save_pooled(
            out_dir / "pooled.pt",
            unit_id,
            pooled,
            kept,
            {
                "model_path": model_path,
                "layers": list(cfg.layers),
                "spans": list(PFX_SPANS),
                "regime": regime,
                "condition_tag": cond_rec["tag"],
                "max_model_len": max_model_len,
                "max_new_tokens": X.max_new_tokens_for(unit_id.split("@")[0]),
                "n_rows": len(kept),
                "n_dropped": dropped,
                "seam_counts": seam_counts,
                "n_distinct_prefix": n_prefix,
                "fp16_roundtrip_cos_min": fp16_cos,
                "gen_wall_s": gen_wall,
                "tf_wall_s": tf_wall,
                "smoke": cfg.smoke,
            },
        )
        _atomic_json(
            out_dir / "manifest.json",
            {
                "unit": unit_id,
                "context_id": cid,
                "prefix_sha256": cond_rec["prefix_sha256"],
                "n_rows": len(kept),
                "valid_frac": valid_frac,
                "model_path": model_path,
                "n_distinct_prefix": n_prefix,
                "fp16_roundtrip_cos_min": fp16_cos,
                "gen_wall_s": gen_wall,
                "tf_wall_s": tf_wall,
                **_meta(),
            },
        )
    finally:
        _cleanup_merged(cleanup)


def run_pfx_tf_unit(cfg: Cfg, unit_arg: str) -> None:
    """pfx3 unit (`<arm>@<cond>`): trained model TF on the SAME-condition base
    tree's rows -> response span-means (the #833 control at the trained
    context)."""
    import torch

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    arm_id, _, cond = unit_arg.partition("@")
    out_dir = _pfx_root(cfg) / "corpus_capture_tf" / unit_arg
    cid = X.pfx_unit_context_id(unit_arg)
    cond_rec = _pfx_cond_record(cfg, cid)
    sample = X.load_pfx_sample(cfg.out_root)
    regime = _pfx_regime(cfg, unit_arg, cond_rec, sample, ("response",))
    if (out_dir / "pooled_tf.pt").exists():
        _pfx_check_regime(out_dir, regime)
        return
    _pfx_check_regime(out_dir, regime)
    base_unit = X.pfx_base_unit(arm_id, cond)
    base_dir = _pfx_root(cfg) / "corpus_capture" / base_unit
    assert (base_dir / "pooled.pt").exists(), f"pfx3 {unit_arg}: base unit {base_unit} not captured"
    rows = _read_rows_with_spans(base_dir)
    model_path, cleanup = _resolve_unit_model(cfg, arm_id)
    try:
        _assert_model_dir_alive(model_path, cleanup)
        pooled = _teacher_forced_span_means(
            model_path,
            rows,
            [base_unit],
            layers=list(cfg.layers),
            spans=("response",),
            device=_device(),
            dtype=_dtype(),
            tf_batch_size=cfg.tf_batch,
        )
        store = {
            "schema_version": 1,
            "unit": unit_arg,
            "row_sha": [r["prompt_sha"] for r in rows],
            "row_question_idx": [r["question_idx"] for r in rows],
            "arms": {
                span: {li: t.to(torch.float16) for li, t in per.items()}
                for span, per in pooled.items()
            },
            "metadata": {
                **_meta(),
                "regime": regime,
                "model_path": model_path,
                "layers": list(cfg.layers),
                "spans": ["response"],
                "shared_text_from": base_unit,
                "n_rows": len(rows),
                "smoke": cfg.smoke,
            },
        }
        store_path = out_dir / "pooled_tf.pt"
        tmp = store_path.with_suffix(".pt.tmp")
        torch.save(store, tmp)
        os.replace(tmp, store_path)
        _atomic_json(
            out_dir / "manifest.json",
            {"unit": unit_arg, "n_rows": len(rows), "model_path": model_path, **_meta()},
        )
    finally:
        _cleanup_merged(cleanup)


def _pfx_pilot_gate(
    ratio: float, smoke: bool, *, tag: str = "pfx1", booked: float = PFX_BOOKED_UNIT_GPU_H
) -> None:
    """Kill criterion (a), plan §7: pilot wall > 2x the booked per-unit row =>
    re-size before fleet launch; > 4x => halt + re-plan. Demoted to a log line
    under smoke (the #1345 gate-calibration rule — smoke n cannot satisfy a
    production-scale wall book). ``tag``/``booked`` default to the round-3
    pfx1 values; lad1 threads its own (round-4 reuse, defaults preserved)."""
    if smoke:
        logger.info("[%s] smoke: pilot wall ratio %.2f (gate informational)", tag, ratio)
        return
    if ratio > 4.0:
        raise RuntimeError(
            f"[{tag}] kill criterion (a): pilot wall {ratio:.2f}x the booked "
            f"{booked} GPU-h/unit — HALT + re-plan (plan §7)"
        )
    if ratio > 2.0:
        raise RuntimeError(
            f"[{tag}] kill criterion (a): pilot wall {ratio:.2f}x the booked "
            f"{booked} GPU-h/unit — re-size §9 before fleet launch"
        )


def phase_pfx1(cfg: Cfg) -> None:
    """pfx1: ONE unit at production shape; gen + TF walls measured separately."""
    _phase("pfx1_pilot")
    _status(cfg, "pfx1_pilot")
    arm = X.PILOT_ARM if cfg.smoke else PFX_PILOT_ARM
    unit = X.pfx_trained_unit(arm, "own")
    run_pfx_corpus_unit(cfg, unit)
    man = json.loads((_pfx_root(cfg) / "corpus_capture" / unit / "manifest.json").read_text())
    unit_wall_h = (man.get("gen_wall_s", 0.0) + man.get("tf_wall_s", 0.0)) / 3600.0
    ratio = unit_wall_h / PFX_BOOKED_UNIT_GPU_H
    _atomic_json(
        _pfx_root(cfg) / "pilot" / "pilot_report.json",
        {
            "unit": unit,
            "gen_wall_s": man.get("gen_wall_s"),
            "tf_wall_s": man.get("tf_wall_s"),
            "unit_wall_h": unit_wall_h,
            "booked_unit_gpu_h": PFX_BOOKED_UNIT_GPU_H,
            "ratio": ratio,
            "smoke": cfg.smoke,
            **_meta(),
        },
    )
    print(f"[pfx1] pilot {unit} wall={unit_wall_h:.2f}h ratio={ratio:.2f}", flush=True)
    _pfx_pilot_gate(ratio, cfg.smoke)


def _pfx_expected_uploads(cfg: Cfg) -> list[str]:
    """The exact-set verify list: every load-bearing store file currently in
    the on_target trees (the pfx4 resume-skip predicate keys on its count)."""
    expected = []
    for tree, fname in (
        ("on_target/corpus_capture", "pooled.pt"),
        ("on_target/corpus_capture_tf", "pooled_tf.pt"),
    ):
        local_tree = cfg.out_root / tree
        if local_tree.exists():
            for unit_dir in sorted(local_tree.iterdir()):
                if (unit_dir / fname).exists():
                    expected.append(f"{cfg.hf_prefix}/{tree}/{unit_dir.name}/{fname}")
    return expected


def phase_pfx4(cfg: Cfg) -> None:
    """pfx4: on_target tree upload + exact-set verify — BEFORE fits (#825)."""
    _phase("pfx4_store_upload")
    _status(cfg, "pfx4_store_upload")
    if not cfg.upload:
        logger.info("[pfx4] upload disabled (--no-upload)")
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    expected = _pfx_expected_uploads(cfg)
    done_path = _pfx_root(cfg) / "upload_done.json"
    if done_path.exists():  # r3-v2 Minor: skip the re-upload when nothing new shipped
        prior = json.loads(done_path.read_text())
        if prior.get("n_verified") == len(expected):
            logger.info(
                "[pfx4] upload_done.json matches the expected store count (%d) — resume skip",
                len(expected),
            )
            return
        logger.info(
            "[pfx4] expected store count changed (%s -> %d) — re-uploading",
            prior.get("n_verified"),
            len(expected),
        )
    uploaded = {}
    for name in PFX_UPLOAD_TREES:
        dest = _upload_tree(cfg, name)
        if dest:
            uploaded[name] = dest
    if expected:
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            X.HF_DATA_REPO,
            expected,
            path_in_repo=f"{cfg.hf_prefix}/on_target",
            repo_type="dataset",
        )
        assert not missing, f"pfx4 verify: {len(missing)} store files missing on Hub: {missing[:5]}"
    _atomic_json(
        _pfx_root(cfg) / "upload_done.json",
        {"uploaded": uploaded, "n_verified": len(expected), **_meta()},
    )


# ── lad: round-4 prefix-richness dose ladder (plan v10) ─────────────────────
#
# lad_build  VM-side (CPU, pre-dispatch): deterministic WildChat-1M streaming
#            scan -> 3-rung never-trained selection (bands anchored to the
#            trained-prefix token counts, production tokenizer) + five
#            exclusion screens -> prefix_ladder.json; uploads the ladder + the
#            r3-results mirror to the Hub (fellows-lane staging) and copies
#            the ladder into the repo results tree for the pre-dispatch commit.
# lad0  rung-corpus render + budgets + FULL-GRAIN exclusion re-assert (CPU).
# lad1  pilot: syc-pers-con-lr1e5-s42 @ r_long, production shape (kill (a)).
# lad2  rung capture (15 units: 12 trained + 3 base@rung; all content decode).
# lad4  on_target_r4 tree upload + exact-set verify (BEFORE fits; #825).

WILDCHAT_DATASET = "allenai/WildChat-1M"
LAD_SCAN_ROWS = 50_000
LAD_SCAN_ROWS_WIDENED = 200_000  # pre-registered ONE-step widening (kill (b))
LAD_SMOKE_SCAN_ROWS = 1_500  # bounded tiny-real probe cap (#1092 class)
LAD_TURN_CONTENT_CAP = 2000  # corpora.py L1499-1503 parity (BOTH turns)
LAD_SCAN_CHECKPOINT_EVERY = 2_000  # rows between cursor checkpoints
LAD_BAND_TOPK = 8  # per-band runner-up depth (cross-rung distinctness margin)
LAD_BOOKED_UNIT_GPU_H = 1.0  # plan §9 lad1 row (r3 REALIZED long-prefix unit)
LAD_PILOT_UNIT = "syc-pers-con-lr1e5-s42@r_long"  # worst-case prefix length
LAD_UPLOAD_TREES = (
    "on_target_r4/inputs",
    "on_target_r4/pilot",
    "on_target_r4/corpus_capture",
)
# plan §11 band bounds: (lo_mult, hi_mult) multipliers on the band target
LAD_BAND_BOUNDS = {"r_short": (0.5, 2.0), "r_mid": (0.5, 2.0), "r_long": (0.75, 1.25)}
LAD_R3_RESULTS = "on_target_r4/inputs/r3_results"  # plan §4.5 mirror prefix


def _lad_root(cfg: Cfg) -> Path:
    return cfg.out_root / "on_target_r4"


def _lad_inputs(cfg: Cfg) -> Path:
    return _lad_root(cfg) / "inputs"


def _lad_repo_ladder_path() -> Path:
    """The committed (git) copy of the pinned ladder (plan §10 git dest)."""
    return (
        REPO_ROOT / "eval_results" / "issue_1768" / "on_target_r4" / "inputs" / "prefix_ladder.json"
    )


def _lad_registry_context(cid: str):
    from explore_persona_space.artifacts.context import CONTEXTS

    return CONTEXTS[cid]


def lad_band_specs(t_pers: int, t_conv: int) -> dict[str, dict]:
    """Band {target, lo, hi} per rung from the MEASURED anchors (plan §11):
    short [0.5,2]xT_pers, mid [0.5,2]x√(T_pers·T_conv), long [0.75,1.25]xT_conv."""
    assert t_pers > 0 and t_conv > t_pers, (t_pers, t_conv)
    t_gm = (t_pers * t_conv) ** 0.5
    targets = {"r_short": float(t_pers), "r_mid": float(t_gm), "r_long": float(t_conv)}
    out = {}
    for cond, target in targets.items():
        lo_m, hi_m = LAD_BAND_BOUNDS[cond]
        out[cond] = {"target": target, "lo": lo_m * target, "hi": hi_m * target}
    return out


def lad_bands_for(t_tokens: int, specs: dict[str, dict]) -> list[str]:
    return [c for c in X.R4_CONDS if specs[c]["lo"] <= t_tokens <= specs[c]["hi"]]


def lad_turns_sha(turns: list[dict]) -> str:
    blob = json.dumps([[t["role"], t["content"]] for t in turns], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _lad_content_sha(ctx) -> str:
    """cid-STRIPPED prefix content sha (r4-r2 review Minor): `_pfx_prefix_sha`
    hashes the recipe INCLUDING `context_id`, so a rung-vs-trained recipe-sha
    compare was structurally vacuous (rung cids always differ, even on
    identical content). Hashing only (system, prefix_turns, user_wrap) makes
    identical CONTENT collide — the sha-space twin of the `trained_prefix`
    content-equality screen in `lad_exclusion_reject`."""
    recipe = _pfx_prefix_recipe(ctx)
    recipe.pop("context_id")
    blob = json.dumps(recipe, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _lad_trained_exclusion_material() -> dict:
    """Trained-context identity material for exclusion screens 1/3 (plan §4.2):
    the 3 trained prefix recipe shas (+ cid-stripped content shas), the conv
    prefix's capped turn tuple, the persona system string, and the ICL
    demonstration texts."""
    ctxs = {
        key: X.pfx_resolve_context(cid)
        for key, cid in (
            ("pers", "persona_software_engineer"),
            ("conv", "wildchat_prefix_real545"),
            ("icl", "icl_prefix_sycophancy"),
        )
    }
    assert ctxs["pers"].system, "persona context lost its system string"
    return {
        "trained_shas": {k: _pfx_prefix_sha(c) for k, c in ctxs.items()},
        "trained_content_shas": {k: _lad_content_sha(c) for k, c in ctxs.items()},
        "conv_turns": tuple((t["role"], t["content"]) for t in ctxs["conv"].prefix_turns),
        "persona_system": ctxs["pers"].system,
        "icl_demo_texts": [t["content"] for t in ctxs["icl"].prefix_turns],
    }


def _lad_full_grain_samples(cfg: Cfg) -> tuple[set[str], list[str]]:
    """(FULL round-1 sha set, FULL 4,400 pfx query texts) for the exclusion
    screens — staged to DEDICATED full-grain paths so a smoke out-root's
    sliced samples are never consulted (plan §4 smoke parity / #1817 rule)."""
    from explore_persona_space.orchestrate import hub

    inputs = _lad_inputs(cfg)
    full_r1 = inputs / "corpus_sample_full.json"
    if not full_r1.exists():
        local = cfg.out_root / "inputs" / "corpus_sample.json"
        if local.exists() and json.loads(local.read_text())["n_train"] == X.N_TRAIN:
            full_r1.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, full_r1)
        else:
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/inputs/corpus_sample.json",
                full_r1,
                repo_type="dataset",
            )
    r1 = json.loads(full_r1.read_text())
    assert r1["n_train"] == X.N_TRAIN, (str(full_r1), r1["n_train"], "full-grain r1 required")
    sha_set = {r["sha"] for r in r1["rows"]}
    full_pfx = inputs / "corpus_sample_pfx_full.json"
    if not full_pfx.exists():
        local = _pfx_inputs(cfg) / "corpus_sample_pfx.json"
        if local.exists() and json.loads(local.read_text())["n_train"] == X.PFX_N_TRAIN:
            full_pfx.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, full_pfx)
        else:
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/on_target/inputs/corpus_sample_pfx.json",
                full_pfx,
                repo_type="dataset",
            )
    pfx = json.loads(full_pfx.read_text())
    n_full = X.PFX_N_TRAIN + X.N_VAL + X.N_TEST
    assert pfx["n_train"] == X.PFX_N_TRAIN and len(pfx["rows"]) == n_full, (
        str(full_pfx),
        pfx["n_train"],
        len(pfx["rows"]),
    )
    return sha_set, [r["prompt"] for r in pfx["rows"]]


# Exclusion screen 6 (r2 revision; concern `r-long-rung-content-language`):
# the WildChat `language == "English"` METADATA screen passed a rung whose
# CONTENT is majority-Cyrillic (r1 r_long, dataset idx 9098) — screen the
# rendered 2-turn prefix TEXT itself. Rule (deterministic, dependency-free):
# among alphabetic chars of BOTH capped turns, >= LAD_CONTENT_LATIN_MIN_RATIO
# must be Latin-script (codepoint <= U+024F: Basic Latin + Latin-1 Supplement
# + Latin Extended-A/B, so accented European letters pass). Zero alphabetic
# chars rejects fail-safe. Threshold grounded on 2026-07-31 measurements:
# trained conv prefix (`wildchat_prefix_real545`) ratio 1.0000 (passes with
# maximal margin — no construction asymmetry), r1 r_short/r_mid 1.0000,
# rejected r1 r_long 0.8648 (fails by 0.085). An additional code-block-
# fraction screen was considered and NOT added: no cheap deterministic
# definition, and it would risk asymmetry vs the trained prefix.
LAD_CONTENT_LATIN_MIN_RATIO = 0.95
LAD_LATIN_MAX_CODEPOINT = 0x024F


def lad_content_language_ok(c0: str, c1: str) -> bool:
    """Content-level language screen (exclusion 6): Latin-script ratio over
    alphabetic chars of the rendered 2-turn prefix text >= the threshold;
    alpha == 0 rejects fail-safe. See the constants block above for the rule
    grounding + measured margins (concern `r-long-rung-content-language`)."""
    alpha = latin = 0
    for ch in itertools.chain(c0, c1):
        if ch.isalpha():
            alpha += 1
            if ord(ch) <= LAD_LATIN_MAX_CODEPOINT:
                latin += 1
    return alpha > 0 and latin / alpha >= LAD_CONTENT_LATIN_MIN_RATIO


def lad_screen_reject(row: dict) -> str | None:
    """Cheap corpus screens (plan §4.2 exclusion 5 + shape + the r2 content-
    language screen 6): first-violation name, or None. `toxic`/`redacted`
    must be present AND falsy-boolean False (the #1092/#1739 field-semantics
    lessons: full language names; missing or truthy moderation fields reject
    fail-safe); metadata language alone is insufficient (screen 6)."""
    if row.get("language") != "English":
        return "language"
    tox = row.get("toxic")
    if tox is None or bool(tox):
        return "toxic"
    red = row.get("redacted")
    if red is None or bool(red):
        return "redacted"
    conv = row.get("conversation") or []
    if len(conv) < 2:
        return "too_few_turns"
    if (conv[0].get("role"), conv[1].get("role")) != ("user", "assistant"):
        return "bad_roles"
    c0 = (conv[0].get("content") or "")[:LAD_TURN_CONTENT_CAP]
    c1 = (conv[1].get("content") or "")[:LAD_TURN_CONTENT_CAP]
    if not (c0.strip() and c1.strip()):
        return "empty_content"
    if not lad_content_language_ok(c0, c1):
        return "content_language"
    return None


def lad_exclusion_reject(
    c0: str, c1: str, raw_user: str, excl: dict, sha_set: set[str]
) -> str | None:
    """Never-trained / content-novel screens 1-3 (plan §4.2), belt excluded
    (the 4,400-text substring belt runs at selection + lad0 re-check)."""
    if (("user", c0), ("assistant", c1)) == excl["conv_turns"]:
        return "trained_prefix"
    if excl["persona_system"] in c0 or excl["persona_system"] in c1:
        return "trained_context_containment"
    for demo in excl["icl_demo_texts"]:
        if demo in c0 or demo in c1:
            return "trained_context_containment"
    if X.prompt_sha(c0) in sha_set or X.prompt_sha(raw_user) in sha_set:
        return "query_sha_overlap"
    return None


LAD_BELT_MIN_QUERY_CHARS = 16  # belt needle floor — see lad_belt_needles


def lad_belt_needles(query_texts: list[str]) -> list[str]:
    """The substring-belt needle set: 4,400-set query texts of >=16 chars.

    The floor is load-bearing, MEASURED (2026-07-31 production lad_build):
    the pinned pfx query set carries 140 texts <8 chars / 330 <16 chars
    (real-user 'hi'/'ok'-class rows — the #1776 short-query collision class),
    and those trivial needles substring-matched EVERY long-band candidate
    (top-8 belt hits were ALL 2-6 chars; 19,358 long-band candidates, zero
    passing), spuriously firing kill criterion (b). Sub-floor queries carry
    no contamination signal as substrings; they stay fully covered by the
    PRIMARY exact-sha screen (`lad_exclusion_reject`), which is untouched."""
    return [q for q in query_texts if len(q) >= LAD_BELT_MIN_QUERY_CHARS]


def lad_substring_belt_hit(c0: str, c1: str, query_texts: list[str]) -> bool:
    """Exclusion-2 belt: any >=16-char 4,400-set query text as a substring of
    either candidate turn (selection candidates + the lad0 full-grain
    re-check share this ONE predicate, floor included)."""
    return any((q in c0) or (q in c1) for q in lad_belt_needles(query_texts))


def _lad_scan_dir(cfg: Cfg) -> Path:
    return _lad_root(cfg) / "scan"  # cursor state — NOT in the upload trees


def _lad_scan(
    cfg: Cfg,
    tok,
    specs: dict,
    excl: dict,
    sha_set: set[str],
    scan_cap: int,
) -> tuple[dict[str, list[dict]], dict[str, int], int, str]:
    """Deterministic WildChat-1M stream scan (dataset order) with chunked
    cursor checkpoints + fingerprint-gated resume (#1092 external-stream rule).
    Returns (per-band top-K pools, per-screen reject counters, rows_scanned,
    dataset revision). Candidate texts NEVER hit logs (digest-only discipline)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    revision = hub.retry_transient(
        lambda: HfApi().dataset_info(WILDCHAT_DATASET).sha,
        what="WildChat-1M revision resolve",
    )
    fp = hashlib.sha256(
        json.dumps(
            {
                "dataset": WILDCHAT_DATASET,
                "revision": revision,
                "bands": specs,
                "cap": LAD_TURN_CONTENT_CAP,
                # +latin<ratio> (r2 content-language screen): a threshold or
                # rule change busts the cursor -> fresh scan (regime keying)
                "screens": (
                    "english+toxicF+redactedF+2turn+ua+nonempty"
                    f"+latin{LAD_CONTENT_LATIN_MIN_RATIO}+excl123"
                ),
                "trained_shas": excl["trained_shas"],
                "smoke": cfg.smoke,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    cursor_path = _lad_scan_dir(cfg) / "cursor.json"
    pools: dict[str, list[dict]] = {c: [] for c in X.R4_CONDS}
    counters: dict[str, int] = {}
    start = 0
    if cursor_path.exists():
        prev = json.loads(cursor_path.read_text())
        # A cursor past the CURRENT cap but within the pre-registered widened
        # cap is a legitimately-WIDENED prior scan (kill-criterion (b) path):
        # its pools are a superset — resume/return, never rescan from zero.
        if prev.get("fingerprint") == fp and prev.get("rows_scanned", 0) <= LAD_SCAN_ROWS_WIDENED:
            pools = {c: list(prev["pools"].get(c, [])) for c in X.R4_CONDS}
            counters = dict(prev["counters"])
            start = int(prev["rows_scanned"])
            logger.info("[lad_build] cursor resume at row %d (fp %s)", start, fp)
        else:
            logger.info("[lad_build] cursor fingerprint/cap mismatch — fresh scan")
    if start >= scan_cap:
        return pools, counters, start, revision

    from datasets import load_dataset

    ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True, revision=revision)
    if start:
        ds = ds.skip(start)
    n = start
    t0 = time.time()

    def _checkpoint() -> None:
        _atomic_json(
            cursor_path,
            {
                "fingerprint": fp,
                "revision": revision,
                "rows_scanned": n,
                "pools": pools,
                "counters": counters,
                **_meta(),
            },
        )

    row = None
    for row in ds:
        if n == 0:
            missing = [
                k
                for k in ("conversation", "language", "toxic", "redacted", "conversation_hash")
                if k not in row
            ]
            assert not missing, f"[lad_build] WildChat row schema drift — missing {missing}"
        n += 1
        reason = lad_screen_reject(row)
        if reason is None:
            conv = row["conversation"]
            c0 = (conv[0].get("content") or "")[:LAD_TURN_CONTENT_CAP]
            c1 = (conv[1].get("content") or "")[:LAD_TURN_CONTENT_CAP]
            t_tokens = len(tok(c0, add_special_tokens=False)["input_ids"]) + len(
                tok(c1, add_special_tokens=False)["input_ids"]
            )
            bands = lad_bands_for(t_tokens, specs)
            if not bands:
                reason = "out_of_band"
            else:
                reason = lad_exclusion_reject(c0, c1, conv[0].get("content") or "", excl, sha_set)
            if reason is None:
                cand = {
                    "index": n - 1,  # 0-based dataset order index
                    "conversation_hash": str(row["conversation_hash"]),
                    "T": t_tokens,
                    "turns": [
                        {"role": "user", "content": c0},
                        {"role": "assistant", "content": c1},
                    ],
                }
                for band in bands:
                    counters[f"band_{band}"] = counters.get(f"band_{band}", 0) + 1
                    pool = pools[band]
                    pool.append({**cand, "dist": abs(_log(t_tokens) - _log(specs[band]["target"]))})
                    pool.sort(key=lambda c: (c["dist"], c["index"]))
                    del pool[LAD_BAND_TOPK:]
        if reason is not None:
            counters[reason] = counters.get(reason, 0) + 1
        if n % LAD_SCAN_CHECKPOINT_EVERY == 0:
            _checkpoint()
        if n % 10_000 == 0:
            kept = sum(counters.get(f"band_{c}", 0) for c in X.R4_CONDS)
            print(
                f"[lad_build] scanned {n}/{scan_cap} kept={kept} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        if n >= scan_cap:
            break
    # #952: release the streaming dataset deterministically pre-shutdown
    del row, ds
    gc.collect()
    _checkpoint()
    kept = sum(counters.get(f"band_{c}", 0) for c in X.R4_CONDS)
    print(
        f"[lad_build] done: scanned={n} kept_in_band={kept} rejects="
        f"{json.dumps({k: v for k, v in sorted(counters.items()) if not k.startswith('band_')})}",
        flush=True,
    )
    return pools, counters, n, revision


def _log(x: float) -> float:
    import math

    return math.log(max(x, 1e-9))


def _lad_select_rungs(
    pools: dict[str, list[dict]],
    query_texts: list[str],
    counters: dict[str, int],
) -> tuple[dict[str, dict], list[str]]:
    """Deterministic selection: per band, candidates in (|log T − log target|,
    dataset index) order; first passing the substring belt + cross-rung
    distinctness (exclusion 4) wins. Returns (selected, shortage bands)."""
    selected: dict[str, dict] = {}
    used_hashes: set[str] = set()
    shortage: list[str] = []
    for cond in X.R4_CONDS:
        picked = None
        for cand in sorted(pools[cond], key=lambda c: (c["dist"], c["index"])):
            if cand["conversation_hash"] in used_hashes:
                counters["cross_rung_hash_collision"] = (
                    counters.get("cross_rung_hash_collision", 0) + 1
                )
                continue
            c0, c1 = (t["content"] for t in cand["turns"])
            if lad_substring_belt_hit(c0, c1, query_texts):
                counters["belt_query_text_substring"] = (
                    counters.get("belt_query_text_substring", 0) + 1
                )
                continue
            picked = cand
            break
        if picked is None:
            shortage.append(cond)
            continue
        used_hashes.add(picked["conversation_hash"])
        selected[cond] = picked
    return selected, shortage


def _lad_write_ladder(
    cfg: Cfg,
    selected: dict[str, dict],
    specs: dict,
    counters: dict[str, int],
    n_scanned: int,
    revision: str,
    excl: dict,
    anchors: dict,
) -> dict:
    """Compose + atomically write prefix_ladder.json (recipes + manifest;
    prefixes referenced by conversation_hash + dataset index + sha in every
    LOG — raw text lives only in this pinned recipe file, plan §4.2)."""
    import types

    rungs = {}
    # cid-STRIPPED content shas (r4-r2 review Minor): the former recipe-sha
    # compare was vacuous (context_id inside the hash); this one BINDS —
    # identical content under a different cid collides.
    trained_content = set(excl["trained_content_shas"].values())
    for cond in X.R4_CONDS:
        cand = selected[cond]
        cid = X.R4_CONTEXT_ID_BY_COND[cond]
        shim = types.SimpleNamespace(
            context_id=cid, system=None, prefix_turns=tuple(cand["turns"]), user_wrap=None
        )
        recipe_sha = _pfx_prefix_sha(shim)
        assert _lad_content_sha(shim) not in trained_content, (
            cond,
            "rung content sha collides with a trained prefix (cid-stripped)",
        )
        rungs[cond] = {
            "context_id": cid,
            "prefix_turns": cand["turns"],
            "conversation_hash": cand["conversation_hash"],
            "dataset_index": cand["index"],
            "turns_sha256": lad_turns_sha(cand["turns"]),
            "recipe_sha256": recipe_sha,
            "realized_tokens": cand["T"],
            "target_tokens": specs[cond]["target"],
            "band": [specs[cond]["lo"], specs[cond]["hi"]],
            "log_dist_to_target": cand["dist"],
            "n_band_candidates": counters.get(f"band_{cond}", 0),
        }
        logger.info(
            "[lad_build] %s: hash=%s index=%d T=%d band=[%.1f, %.1f] candidates=%d",
            cond,
            cand["conversation_hash"],
            cand["index"],
            cand["T"],
            specs[cond]["lo"],
            specs[cond]["hi"],
            counters.get(f"band_{cond}", 0),
        )
    ladder = {
        "rungs": rungs,
        "anchors": anchors,
        "scan": {
            "dataset": WILDCHAT_DATASET,
            "revision": revision,
            "n_rows_scanned": n_scanned,
            "widened": n_scanned > LAD_SCAN_ROWS,
            "turn_content_cap": LAD_TURN_CONTENT_CAP,
            "selection_rule": "min (|log T - log target|, dataset index); all screens",
        },
        "counters": counters,
        "builder_revision": (
            "r2 (2026-07-31): content-language screen added (concern "
            "r-long-rung-content-language) — supersedes the r1 belt-floor ladder "
            "(r_long idx 9098 majority-Cyrillic content under English metadata); "
            "no dependent captures existed at regeneration (pre-dispatch)"
        ),
        "exclusions": {
            "trained_prefix_recipe_shas": excl["trained_shas"],
            "trained_prefix_content_shas": excl["trained_content_shas"],
            "belt_min_query_chars": LAD_BELT_MIN_QUERY_CHARS,
            "content_language_screen": {
                "rule": (
                    "latin_alpha / alpha over BOTH capped turns; latin = "
                    "isalpha() and codepoint <= latin_max_codepoint; "
                    "alpha == 0 rejects fail-safe"
                ),
                "min_latin_ratio": LAD_CONTENT_LATIN_MIN_RATIO,
                "latin_max_codepoint": LAD_LATIN_MAX_CODEPOINT,
                "trained_conv_prefix_ratio": 1.0,  # measured 2026-07-31 (parity)
            },
            "screens": [
                "trained-prefix disjointness (turns + cid-stripped content sha)",
                "query-corpus sha disjointness (full round-1 sample) + substring belt "
                f"(needles >= {LAD_BELT_MIN_QUERY_CHARS} chars — see lad_belt_needles)",
                "trained-context non-containment (persona system + icl demos)",
                "cross-rung conversation_hash distinctness",
                "corpus screens (English / toxic False / redacted False)",
                "content-language (Latin-script ratio >= "
                f"{LAD_CONTENT_LATIN_MIN_RATIO} over alphabetic chars of both capped "
                "turns — metadata language alone is insufficient; see "
                "lad_content_language_ok)",
            ],
        },
        **_meta(),
    }
    _atomic_json(_lad_inputs(cfg) / "prefix_ladder.json", ladder)
    return ladder


def _lad_mirror_r3_results(cfg: Cfg) -> None:
    """Unconditional idempotent pre-dispatch mirror of the round-3 COMMITTED
    result inputs to HF `{prefix}/on_target_r4/inputs/r3_results/` (plan §4.5
    lad7 + §9 off_pod_phases: the fellows lane rsync-excludes eval_results/).
    ONE upload_folder commit (never a per-file loop — the #664 504-storm rule)."""
    from explore_persona_space.orchestrate import hub

    src_root = REPO_ROOT / "eval_results" / "issue_1768" / "on_target"
    mirror = _lad_inputs(cfg) / "r3_results"
    rels = ["map_change_on_target.json", "m0_prefix_effect.json"]
    rels += [
        f"percell/{a}_L{layer}_{s}.json"
        for a in X.R4_ARMS
        for layer in X.LAYERS
        for s in ("bare_n", "control", "own")
    ]
    rels += [f"fits_bare_n/{a}_L{layer}.json" for a in X.R4_ARMS for layer in X.LAYERS]
    for rel in rels:
        src = src_root / rel
        assert src.exists(), f"[lad_build] r3 mirror source missing from the repo tree: {src}"
        dst = mirror / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
    url = hub._upload(
        mirror,
        repo_id=X.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{cfg.hf_prefix}/{LAD_R3_RESULTS}",
    )
    if not url:
        raise RuntimeError("[lad_build] r3_results mirror upload returned no path")
    logger.info("[lad_build] r3_results mirror uploaded (%d files)", len(rels))


def _lad_build_publish(cfg: Cfg, ladder: dict) -> None:
    """Repo-copy the pinned ladder for the pre-dispatch git commit + upload it
    and the r3-results mirror to the Hub (idempotent; production only)."""
    from explore_persona_space.orchestrate import hub

    src = _lad_inputs(cfg) / "prefix_ladder.json"
    repo_copy = _lad_repo_ladder_path()
    if not repo_copy.exists() or repo_copy.read_text() != src.read_text():
        repo_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, repo_copy)
        logger.info("[lad_build] ladder copied to %s (commit pre-dispatch)", repo_copy)
    if not cfg.upload:
        logger.info("[lad_build] upload disabled (--no-upload)")
        return
    url = hub._upload(
        src,
        repo_id=X.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{cfg.hf_prefix}/on_target_r4/inputs/prefix_ladder.json",
        upload_as_file=True,
    )
    if not url:
        raise RuntimeError("[lad_build] prefix_ladder.json upload returned no path")
    _lad_mirror_r3_results(cfg)


def _lad_ladder_screens_current(ladder: dict) -> bool:
    """Resume-guard regime key (the #722-r3 rule: resume keys on every
    output-affecting regime key): a pinned ladder built under an OLDER screen
    set (pre-belt-floor, or pre-content-language-screen) REBUILDS instead of
    republishing."""
    exc = ladder.get("exclusions", {})
    return (
        exc.get("belt_min_query_chars") == LAD_BELT_MIN_QUERY_CHARS
        and exc.get("content_language_screen", {}).get("min_latin_ratio")
        == LAD_CONTENT_LATIN_MIN_RATIO
    )


def _lad_set_belt_census(counters: dict[str, int], query_texts: list[str]) -> None:
    """Manifest belt-census keys — re-set after EVERY `_lad_scan` return: a
    widened rescan rebinds `counters` from the cursor, which dropped keys set
    before the widening (r4-r2 review Minor)."""
    counters["belt_needles_total"] = len(query_texts)
    counters["belt_needles_below_floor_excluded"] = len(query_texts) - len(
        lad_belt_needles(query_texts)
    )


def phase_lad_build(cfg: Cfg) -> None:
    """lad_build (VM-side CPU, pre-dispatch): WildChat-1M streaming scan ->
    3-rung never-trained ladder (production) / bounded tiny-real ingestion
    probe with per-screen reject counters (--smoke; #1092 class)."""
    _phase("lad_build")
    _status(cfg, "lad_build")
    from transformers import AutoTokenizer

    dest = _lad_inputs(cfg) / "prefix_ladder.json"
    if dest.exists() and not cfg.smoke:
        prior = json.loads(dest.read_text())
        if _lad_ladder_screens_current(prior):
            logger.info("[lad_build] prefix_ladder.json present — resume skip; re-publishing")
            _lad_build_publish(cfg, prior)
            return
        logger.info(
            "[lad_build] existing ladder predates the CURRENT screen set — rebuilding "
            "(r2 regime-keyed resume; concern r-long-rung-content-language)"
        )
    excl = _lad_trained_exclusion_material()
    # Construction-asymmetry parity (concern r-long-rung-content-language):
    # the trained conv prefix MUST pass the same content-language screen the
    # rung candidates face — otherwise flag loudly and STOP (never screen
    # rungs harder than the trained prefix they are compared against).
    tc0, tc1 = (content for _role, content in excl["conv_turns"])
    assert lad_content_language_ok(tc0, tc1), (
        "[lad_build] trained conv prefix FAILS the content-language screen — "
        "construction asymmetry vs the rung candidates; STOP and re-plan the "
        "screen threshold (measured 2026-07-31: trained ratio 1.0000)"
    )
    sha_set, query_texts = _lad_full_grain_samples(cfg)
    # Band anchors ALWAYS use the PRODUCTION tokenizer (plan §11) — never a
    # model_override tokenizer (the smoke fixture ships the real one anyway).
    tok = AutoTokenizer.from_pretrained(X.BASE_MODEL)
    t_pers = len(tok(excl["persona_system"], add_special_tokens=False)["input_ids"])
    t_conv = sum(
        len(tok(content, add_special_tokens=False)["input_ids"])
        for _role, content in excl["conv_turns"]
    )
    specs = lad_band_specs(t_pers, t_conv)
    anchors = {"t_pers": t_pers, "t_conv": t_conv, "t_gm": specs["r_mid"]["target"]}
    logger.info(
        "[lad_build] anchors: T_pers=%d T_conv=%d T_gm=%.1f", t_pers, t_conv, anchors["t_gm"]
    )
    scan_cap = LAD_SMOKE_SCAN_ROWS if cfg.smoke else LAD_SCAN_ROWS
    pools, counters, n_scanned, revision = _lad_scan(cfg, tok, specs, excl, sha_set, scan_cap)
    if cfg.smoke:
        kept = sum(counters.get(f"band_{c}", 0) for c in X.R4_CONDS)
        _atomic_json(
            _lad_inputs(cfg) / "lad_probe_report.json",
            {
                "probe": True,
                "rows_scanned": n_scanned,
                "kept_in_band": kept,
                "counters": counters,
                "anchors": anchors,
                "bands": specs,
                "revision": revision,
                **_meta(),
            },
        )
        assert kept > 0, "[lad_build] tiny-real probe kept ZERO in-band candidates (#1092 class)"
        logger.info("[lad_build] SMOKE probe: scanned=%d kept_in_band=%d", n_scanned, kept)
        return
    _lad_set_belt_census(counters, query_texts)
    selected, shortage = _lad_select_rungs(pools, query_texts, counters)
    if shortage and scan_cap < LAD_SCAN_ROWS_WIDENED:
        logger.info(
            "[lad_build] band shortage %s at %d rows — pre-registered widening to %d",
            shortage,
            n_scanned,
            LAD_SCAN_ROWS_WIDENED,
        )
        pools, counters, n_scanned, revision = _lad_scan(
            cfg, tok, specs, excl, sha_set, LAD_SCAN_ROWS_WIDENED
        )
        _lad_set_belt_census(counters, query_texts)  # rescan rebinds counters
        selected, shortage = _lad_select_rungs(pools, query_texts, counters)
    if shortage:
        raise RuntimeError(
            f"[lad_build] kill criterion (b): band(s) {shortage} have no qualifying "
            f"candidate after the {LAD_SCAN_ROWS_WIDENED}-row widening — "
            "failure_class: data (plan §7; re-plan the band bounds)"
        )
    ladder = _lad_write_ladder(cfg, selected, specs, counters, n_scanned, revision, excl, anchors)
    _lad_build_publish(cfg, ladder)


def _lad_stage_inputs(cfg: Cfg) -> None:
    """lad0 staging: pinned ladder (local -> repo checkout -> Hub, fail-loud
    both-miss) + the pfx sample. Production stages the round-3
    corpus_sample_pfx.json VERBATIM from the Hub (plan §4.3 — never
    re-derived); --smoke derives the tiny sample via the round-3
    `_build_pfx_sample` path from the fixture corpus (refusal-hygiene:
    synthetic rows), while the FULL-grain samples for the exclusion re-assert
    stage separately in `_lad_full_grain_samples`."""
    from explore_persona_space.orchestrate import hub

    ladder_path = _lad_inputs(cfg) / "prefix_ladder.json"
    if not ladder_path.exists():
        repo_copy = _lad_repo_ladder_path()
        if repo_copy.exists():
            ladder_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_copy, ladder_path)
        else:
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/on_target_r4/inputs/prefix_ladder.json",
                ladder_path,
                repo_type="dataset",
            )
    canonical = _pfx_inputs(cfg) / "corpus_sample_pfx.json"
    if not canonical.exists():
        if cfg.smoke:
            _build_pfx_sample(cfg)  # fixture-derived tiny sample (p0 --smoke ran)
        else:
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/on_target/inputs/corpus_sample_pfx.json",
                canonical,
                repo_type="dataset",
            )


def _lad_recheck_exclusions(cfg: Cfg, tok, ladder: dict) -> dict:
    """Kill criterion (d): re-assert the §4.2 exclusion set + band membership
    against the FULL pinned ladder + FULL-grain samples (never the smoke
    slice — the #1817 rule). Any violation = builder bug, fail loud."""
    import types

    excl = _lad_trained_exclusion_material()
    sha_set, query_texts = _lad_full_grain_samples(cfg)
    trained_content = set(excl["trained_content_shas"].values())
    out = {}
    hashes = []
    for cond in X.R4_CONDS:
        rec = ladder["rungs"][cond]
        turns = rec["prefix_turns"]
        roles = tuple(t["role"] for t in turns)
        assert roles == ("user", "assistant"), (cond, roles)
        c0, c1 = (t["content"] for t in turns)
        assert c0.strip() and c1.strip(), (cond, "empty turn content")
        assert len(c0) <= LAD_TURN_CONTENT_CAP and len(c1) <= LAD_TURN_CONTENT_CAP, cond
        assert lad_content_language_ok(c0, c1), (cond, "content-language screen (kill d)")
        t_tokens = len(tok(c0, add_special_tokens=False)["input_ids"]) + len(
            tok(c1, add_special_tokens=False)["input_ids"]
        )
        assert t_tokens == rec["realized_tokens"], (
            cond,
            t_tokens,
            rec["realized_tokens"],
            "tokenizer drift vs the build-time count",
        )
        lo, hi = rec["band"]
        assert lo <= t_tokens <= hi, (cond, t_tokens, rec["band"])
        reason = lad_exclusion_reject(c0, c1, c0, excl, sha_set)
        assert reason is None, (cond, reason, "builder exclusion violated (kill d)")
        assert not lad_substring_belt_hit(c0, c1, query_texts), (cond, "belt hit (kill d)")

        shim = types.SimpleNamespace(
            context_id=rec["context_id"], system=None, prefix_turns=tuple(turns), user_wrap=None
        )
        assert _lad_content_sha(shim) not in trained_content, (cond, "trained content sha")
        assert lad_turns_sha(turns) == rec["turns_sha256"], (cond, "turns sha drift")
        hashes.append(rec["conversation_hash"])
        out[cond] = {
            "conversation_hash": rec["conversation_hash"],
            "dataset_index": rec["dataset_index"],
            "realized_tokens": t_tokens,
            "band": rec["band"],
            "screens": "ALL PASS (full grain)",
        }
    assert len(set(hashes)) == len(hashes), ("cross-rung hash collision", hashes)
    return out


def _lad_cond_record(cfg: Cfg, context_id: str) -> dict:
    conds = json.loads((_lad_inputs(cfg) / "conditions.json").read_text())["conditions"]
    assert context_id in conds, (context_id, "rung condition not built at lad0", sorted(conds))
    return conds[context_id]


def phase_lad0(cfg: Cfg) -> None:
    """lad0: rung-corpus render + budgets + FULL-grain exclusion re-assert."""
    _phase("lad0_rung_corpus")
    _status(cfg, "lad0_rung_corpus")
    from transformers import AutoTokenizer

    _lad_stage_inputs(cfg)
    sample = X.load_pfx_sample(cfg.out_root)
    done = _lad_inputs(cfg) / "build_done.json"
    if done.exists():
        logger.info("[lad0] build_done.json present — resume skip")
        return
    X.register_r4_ladder_contexts(cfg.out_root)
    ladder = X.load_r4_ladder(cfg.out_root)
    tok = AutoTokenizer.from_pretrained(cfg.model_override or X.BASE_MODEL)
    conds: dict[str, dict] = {}
    for cond in X.R4_CONDS:
        cid = X.R4_CONTEXT_ID_BY_COND[cond]
        ctx = _lad_registry_context(cid)
        budgets = _pfx_budget(tok, ctx, sample["rows"])
        prefix_tokens = len(tok(ctx.render(tok, ""), add_special_tokens=False)["input_ids"])
        rung = ladder["rungs"][cond]
        rec = {
            "tag": cond,
            "context_id": cid,
            "prefix_sha256": _pfx_prefix_sha(ctx),
            "recipe": _pfx_prefix_recipe(ctx),
            "prefix_tokens": prefix_tokens,
            "realized_content_tokens": rung["realized_tokens"],
            "target_tokens": rung["target_tokens"],
            "band": rung["band"],
            "conversation_hash": rung["conversation_hash"],
            "dataset_index": rung["dataset_index"],
            "message_shape": {
                "has_system": ctx.system is not None,
                "n_prefix_turns": len(ctx.prefix_turns),
                "has_user_wrap": ctx.user_wrap is not None,
            },
            "budgets": budgets,
            "n_rows": len(sample["rows"]),
        }
        conds[cid] = rec
        _atomic_json(
            _lad_inputs(cfg) / f"corpus_{cond}.json",
            {
                **rec,
                "rows_sha256": hashlib.sha256(
                    "\n".join(r["sha"] for r in sample["rows"]).encode("utf-8")
                ).hexdigest(),
                **_meta(),
            },
        )
    _atomic_json(_lad_inputs(cfg) / "conditions.json", {"conditions": conds, **_meta()})
    recheck = _lad_recheck_exclusions(cfg, tok, ladder)  # kill criterion (d)
    _atomic_json(_lad_inputs(cfg) / "exclusion_recheck.json", {"rungs": recheck, **_meta()})
    _atomic_json(
        done,
        {"n_rows": len(sample["rows"]), "conditions": sorted(conds), **_meta()},
    )
    logger.info("[lad0] rung corpora + budgets + exclusion re-assert done (%d rungs)", len(conds))


def run_lad_corpus_unit(cfg: Cfg, unit_id: str) -> None:
    """lad2 unit: rung-prefixed greedy gen -> TF span-means (prefix+ctx+resp),
    via the SAME `_prefixed_capture_core` the round-3 pfx2 units run. The
    ladder registrar runs idempotently at point of use (fresh fan-out
    subprocesses inherit no registry state — the #1090-fu6/#1315 lessons)."""
    X.register_r4_ladder_contexts(cfg.out_root)
    cid = X.r4_unit_context_id(unit_id)
    _prefixed_capture_core(
        cfg,
        unit_id,
        root=_lad_root(cfg),
        cid=cid,
        cond_rec=_lad_cond_record(cfg, cid),
        ctx=_lad_registry_context(cid),
    )


def phase_lad1(cfg: Cfg) -> None:
    """lad1: ONE unit at production shape (worst-case long rung); gen + TF
    walls measured separately (kill criterion a)."""
    _phase("lad1_pilot")
    _status(cfg, "lad1_pilot")
    unit = LAD_PILOT_UNIT
    run_lad_corpus_unit(cfg, unit)
    man = json.loads((_lad_root(cfg) / "corpus_capture" / unit / "manifest.json").read_text())
    unit_wall_h = (man.get("gen_wall_s", 0.0) + man.get("tf_wall_s", 0.0)) / 3600.0
    ratio = unit_wall_h / LAD_BOOKED_UNIT_GPU_H
    _atomic_json(
        _lad_root(cfg) / "pilot" / "pilot_report.json",
        {
            "unit": unit,
            "gen_wall_s": man.get("gen_wall_s"),
            "tf_wall_s": man.get("tf_wall_s"),
            "unit_wall_h": unit_wall_h,
            "booked_unit_gpu_h": LAD_BOOKED_UNIT_GPU_H,
            "ratio": ratio,
            "smoke": cfg.smoke,
            **_meta(),
        },
    )
    print(f"[lad1] pilot {unit} wall={unit_wall_h:.2f}h ratio={ratio:.2f}", flush=True)
    _pfx_pilot_gate(ratio, cfg.smoke, tag="lad1", booked=LAD_BOOKED_UNIT_GPU_H)


def _lad_arms(cfg: Cfg) -> list[str]:
    """The round-4 arm scope (plan §4.1); --arms filters INSIDE it; smoke =
    the plan-§4 smoke-parity arm."""
    if cfg.arms:
        want = set(cfg.arms)
        unknown = want - set(X.R4_ARMS)
        assert not unknown, f"--arms outside the r4 arm set: {sorted(unknown)}"
        return [a for a in X.R4_ARMS if a in want]
    if cfg.smoke:
        return ["syc-pers-con-lr1e5-s42"]
    return list(X.R4_ARMS)


def _lad_conds_capture(cfg: Cfg) -> tuple[str, ...]:
    return ("r_long",) if cfg.smoke else X.R4_CONDS


def _lad_unit_set(cfg: Cfg) -> list[str]:
    """lad2 units: shared base@rung units first, then arm@rung units
    (production: 3 + 12 = 15; smoke: base_content@r_long + pilot@r_long)."""
    arms = _lad_arms(cfg)
    conds = _lad_conds_capture(cfg)
    bases = [X.r4_base_unit(c) for c in conds]
    trained = [X.r4_trained_unit(a, c) for a in arms for c in conds]
    return bases + trained


def _lad_expected_uploads(cfg: Cfg) -> list[str]:
    """Exact-set verify list: EVERY per-unit artifact file the corpus_capture
    tree carries — pooled.pt + manifest/spans/done sentinels + the raw-row
    rollout shards (r4-r2 review Minor: pooled.pt-only left the rollout text
    riding the same folder commit UNVERIFIED)."""
    expected = []
    tree = "on_target_r4/corpus_capture"
    local_tree = cfg.out_root / tree
    if local_tree.exists():
        for unit_dir in sorted(local_tree.iterdir()):
            if not (unit_dir / "pooled.pt").exists():
                continue
            names = ["pooled.pt"]
            names += [p.name for p in sorted(unit_dir.glob("raw_rows_*.jsonl"))]
            for extra in ("raw_rows.done.json", "rows_spans.json", "manifest.json"):
                if (unit_dir / extra).exists():
                    names.append(extra)
            expected += [f"{cfg.hf_prefix}/{tree}/{unit_dir.name}/{n}" for n in names]
    return expected


def phase_lad4(cfg: Cfg) -> None:
    """lad4: on_target_r4 tree upload + exact-set verify — BEFORE fits (#825)."""
    _phase("lad4_store_upload")
    _status(cfg, "lad4_store_upload")
    if not cfg.upload:
        logger.info("[lad4] upload disabled (--no-upload)")
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    expected = _lad_expected_uploads(cfg)
    done_path = _lad_root(cfg) / "upload_done.json"
    if done_path.exists():
        prior = json.loads(done_path.read_text())
        if prior.get("n_verified") == len(expected):
            logger.info(
                "[lad4] upload_done.json matches the expected store count (%d) — resume skip",
                len(expected),
            )
            return
        logger.info(
            "[lad4] expected store count changed (%s -> %d) — re-uploading",
            prior.get("n_verified"),
            len(expected),
        )
    uploaded = {}
    for name in LAD_UPLOAD_TREES:
        dest = _upload_tree(cfg, name)
        if dest:
            uploaded[name] = dest
    if expected:
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            X.HF_DATA_REPO,
            expected,
            path_in_repo=f"{cfg.hf_prefix}/on_target_r4",
            repo_type="dataset",
        )
        assert not missing, f"lad4 verify: {len(missing)} store files missing on Hub: {missing[:5]}"
    _atomic_json(
        done_path,
        {"uploaded": uploaded, "n_verified": len(expected), **_meta()},
    )


# ── brl: round-5 behavior-relevant never-trained prefix panel (plan v13) ─────
#
# brl_build  VM-side (CPU, pre-dispatch): fu3 C3-conv sycophancy pool join ->
#            trained-row exclusion (issue1481 train_mix) -> banked-judge
#            filter (6-row pin) -> deterministic 15-pairing enumeration ->
#            3 two-exchange 4-turn prefixes at the r4 r_long band ->
#            prefix_ladder_r5.json + repo commit copy + HF upload + the
#            r3+r4 r_results mirror. NO WildChat scan (the pool is 36 rows —
#            the lad_build scan/cursor/widening machinery is deleted from
#            this twin; plan §9 row 1).
# brl0  panel-corpus render + budgets + FULL-GRAIN exclusion re-assert (CPU).
# brl1  pilot: syc-pers-con-lr1e5-s42 @ the longest realized-T prefix (kill a).
# brl2  panel capture (15 units: 12 trained + 3 base@b_rel; content decode).
# brl4  on_target_r5 tree upload + exact-set verify (BEFORE fits; #825).

BRL_FU3_PREFIX = "issue1090_fu3/C3-conv-con-sycophancy-claude/datagen"
BRL_MIX_PREFIX = "issue1481_conpos_grid/po_mixes/syc-conv/mix"
BRL_POOL_FILES = ("pos.jsonl", "raw_pos.jsonl", "judge_rows.jsonl", "pool_meta.json")
BRL_MIX_FILES = ("train_mix.jsonl", "mix_meta.json")
BRL_N_RAW = 36
BRL_N_EMITTED = 20
BRL_N_KEPT_TOTAL = 26  # pool_meta arithmetic: 26 kept = 20 emitted + 6 surplus
# Plan §4.2 step 3 pinned EXPECTED kept never-trained set (request_id -> banked
# judge mean-of-5); a mismatch = the Hub artifact drifted (kill criterion b,
# failure_class: data).
BRL_EXPECTED_KEPT = {
    "pos-00001": 95.0,
    "pos-00006": 95.0,
    "pos-00011": 95.0,
    "pos-00014": 85.0,
    "pos-00017": 85.0,
    "pos-00030": 68.5,
}
BRL_BAND = (547.5, 912.5)  # r4 r_long band, verbatim (plan §11; re-asserted vs the r4 ladder)
BRL_BOOKED_UNIT_GPU_H = 0.2  # plan §9 brl1 row (r4 REALIZED pilot 0.152 GPU-h + margin)
BRL_PILOT_ARM = "syc-pers-con-lr1e5-s42"  # plan §4.4 brl1 (also the smoke-parity arm)
BRL_UPLOAD_TREES = (
    "on_target_r5/inputs",
    "on_target_r5/pilot",
    "on_target_r5/corpus_capture",
)
BRL_R_RESULTS = "on_target_r5/inputs/r_results"  # plan §4.2 lane-transport mirror prefix


def _brl_root(cfg: Cfg) -> Path:
    return cfg.out_root / "on_target_r5"


def _brl_inputs(cfg: Cfg) -> Path:
    return _brl_root(cfg) / "inputs"


def _brl_repo_panel_path() -> Path:
    """The committed (git) copy of the pinned panel (plan §10 git dest)."""
    return (
        REPO_ROOT
        / "eval_results"
        / "issue_1768"
        / "on_target_r5"
        / "inputs"
        / "prefix_ladder_r5.json"
    )


def _brl_stage_pool(cfg: Cfg) -> dict[str, Path]:
    """Stage the fu3 pool + issue1481 mix files (plan §4.2 sources) under
    on_target_r5/inputs/pool/ — idempotent; PRODUCTION Hub prefixes (the pool
    is a production input even under --smoke)."""
    from explore_persona_space.orchestrate import hub

    dest = _brl_inputs(cfg) / "pool"
    out: dict[str, Path] = {}
    for prefix, names in ((BRL_FU3_PREFIX, BRL_POOL_FILES), (BRL_MIX_PREFIX, BRL_MIX_FILES)):
        for name in names:
            target = dest / name
            if not target.exists():
                hub.stage_hub_file(X.HF_DATA_REPO, f"{prefix}/{name}", target, repo_type="dataset")
            out[name] = target
    return out


def _brl_jsonl_rows(path: Path) -> list[dict]:
    """Text-mode JSONL iteration (never splitlines — the #825/#950 rule)."""
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _brl_msg_qc(row: dict, tag: str) -> tuple[str, str]:
    """(question, completion) of a pos/mix row: `prompt` = a message list whose
    LAST user turn carries the question; `completion` = exactly one assistant
    message (schema probed against the pinned artifacts, 2026-07-31)."""
    users = [m for m in row["prompt"] if m.get("role") == "user"]
    assert users, (tag, "no user turn in a pos/mix row prompt")
    comp = row["completion"]
    assert isinstance(comp, list) and len(comp) == 1 and comp[0]["role"] == "assistant", (
        tag,
        "unexpected completion shape (want [one assistant message])",
    )
    return users[-1]["content"], comp[0]["content"]


def _brl_derive_kept(pool: dict[str, Path]) -> tuple[list[dict], dict]:
    """Plan §4.2 steps 1-3: pool join -> trained-row exclusion (sha-pair vs the
    20 emitted rows, independently re-derived from train_mix.jsonl) -> banked
    judge-acceptance filter. Every count + the pinned 6-row set asserted
    (kill criterion b: a mismatch = the Hub artifact drifted,
    failure_class: data). Returns (kept rows with `_judge_mean`/`_shared_q`
    annotations, the derivation record for the panel manifest)."""
    pos = _brl_jsonl_rows(pool["pos.jsonl"])
    raw = _brl_jsonl_rows(pool["raw_pos.jsonl"])
    judge = _brl_jsonl_rows(pool["judge_rows.jsonl"])
    mix = _brl_jsonl_rows(pool["train_mix.jsonl"])
    meta = json.loads(pool["pool_meta.json"].read_text())
    assert len(raw) == BRL_N_RAW and len(pos) == BRL_N_EMITTED, (len(raw), len(pos))
    jr = {r["request_id"]: r for r in judge}
    n_join = sum(1 for r in raw if r["request_id"] in jr)
    assert n_join == BRL_N_RAW, (n_join, "raw_pos -> judge_rows join must be 36/36 (plan §4.2)")
    emitted = {tuple(map(X.prompt_sha, _brl_msg_qc(r, "pos"))) for r in pos}
    assert len(emitted) == BRL_N_EMITTED, len(emitted)
    mix_pairs = {tuple(map(X.prompt_sha, _brl_msg_qc(r, "mix"))) for r in mix}
    assert emitted <= mix_pairs, (
        "emitted (question, completion) sha pairs not all present in train_mix.jsonl "
        "(plan §4.2 step 2: independent exclusion-set re-derivation)"
    )
    never = [
        r
        for r in raw
        if (X.prompt_sha(r["question"]), X.prompt_sha(r["completion"])) not in emitted
    ]
    kept = sorted((r for r in never if jr[r["request_id"]]["kept"]), key=lambda r: r["request_id"])
    realized = {r["request_id"]: float(jr[r["request_id"]]["mean"]) for r in kept}
    assert realized == BRL_EXPECTED_KEPT, (
        realized,
        BRL_EXPECTED_KEPT,
        "kill criterion (b): kept never-trained set != the plan-pinned 6 rows "
        "(the Hub pool drifted) — failure_class: data",
    )
    n_kept_total = sum(1 for r in raw if jr[r["request_id"]]["kept"])
    assert n_kept_total == BRL_N_KEPT_TOTAL, (
        n_kept_total,
        "pool_meta reconciliation: 26 kept == 20 emitted + 6 surplus (plan §4.2 step 3)",
    )
    assert (
        meta["positive"]["kept"] == BRL_N_KEPT_TOTAL
        and meta["positive"]["emitted"] == BRL_N_EMITTED
    ), meta.get("positive")
    emitted_q = {X.prompt_sha(_brl_msg_qc(r, "pos")[0]) for r in pos}
    for r in kept:
        r["_judge_mean"] = realized[r["request_id"]]
        r["_shared_q"] = X.prompt_sha(r["question"]) in emitted_q
    derivation = {
        "n_raw": len(raw),
        "n_emitted": BRL_N_EMITTED,
        "n_never_trained": len(never),
        "n_kept_never_trained": len(kept),
        "n_kept_total": n_kept_total,
        "judge_instrument": {
            "judge_model": meta["judge_model"],
            "threshold": meta["threshold"],
            "n_judge_draws": meta["n_judge_draws"],
        },
        "emitted_qc_sha16_pairs": sorted([list(p) for p in emitted]),
        "pool_shas": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in pool.items()
        },
    }
    return kept, derivation


def brl_pairings(ids: list[str]) -> list[list[tuple[str, str]]]:
    """All perfect matchings of the ids into unordered pairs ((n-1)!! = 15 at
    n=6; plan §4.2 step 6 enumeration)."""
    rem = sorted(ids)
    if not rem:
        return [[]]
    first = rem[0]
    out = []
    for j in range(1, len(rem)):
        for sub in brl_pairings(rem[1:j] + rem[j + 1 :]):
            out.append([(first, rem[j])] + sub)
    return out


def _brl_canon(pairing) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(tuple(sorted(pair)) for pair in pairing))


def brl_select_pairing(
    T: dict[str, int], band: tuple[float, float] = BRL_BAND
) -> tuple[list[tuple[str, str]], int, list[dict]]:
    """Plan §4.2 step 6 registered pairing rule: over all perfect pairings,
    maximize the in-band prefix count; tie-break (i) maximal min-prefix T,
    (ii) lexicographically smallest sorted request_id tuple. Returns
    (winning canonical pairing, its in-band count, the full enumeration
    record for the manifest)."""
    lo, hi = band

    def key(p):
        ts = [T[a] + T[b] for a, b in p]
        n_in = sum(1 for t in ts if lo <= t <= hi)
        return (-n_in, -min(ts), _brl_canon(p))

    allp = brl_pairings(list(T))
    assert len(allp) == 15, (len(allp), "6-exchange perfect-matching count")
    record = []
    for p in allp:
        canon = _brl_canon(p)
        ts = {f"{a}+{b}": T[a] + T[b] for a, b in canon}
        record.append(
            {
                "pairs": [list(x) for x in canon],
                "pair_tokens": ts,
                "n_in_band": sum(1 for t in ts.values() if lo <= t <= hi),
                "min_pair_tokens": min(ts.values()),
            }
        )
    win = min(allp, key=key)
    canon = list(_brl_canon(win))
    n_in = sum(1 for a, b in canon if lo <= T[a] + T[b] <= hi)
    return canon, n_in, record


def brl_assign_conds(
    pairing: list[tuple[str, str]], judge_means: dict[str, float]
) -> dict[str, tuple[str, str]]:
    """Deterministic cond-label assignment: pairs in ASCENDING total banked
    judge mean (tie: ascending min request_id) -> b_rel1, b_rel2, b_rel3.
    Reproduces the plan-§4.2 advisory labeling on the predicted pairing;
    labels are naming only — every registered read is label-symmetric and the
    question-shared split keys on the manifest flags, never the label."""
    ordered = sorted(pairing, key=lambda p: (judge_means[p[0]] + judge_means[p[1]], min(p)))
    return dict(zip(X.R5_CONDS, ordered, strict=True))


def brl_prefix_reject(
    turns: list[dict], excl: dict, sha_set: set[str], query_texts: list[str]
) -> str | None:
    """Never-trained / content-novel screens for ONE rendered b_rel prefix
    (plan §4.2 exclusions 3-4 + shape 6): first-violation name, or None.
    Recipe/content-sha screens (exclusion 2) run at the shim level in the
    builder + recheck (they need the r4 ladder's banked shas)."""
    roles = tuple(t["role"] for t in turns)
    if roles != X.R5_TURN_ROLES:
        return "bad_roles"
    contents = [t["content"] for t in turns]
    if not all(c.strip() for c in contents):
        return "empty_content"
    if any(len(c) > LAD_TURN_CONTENT_CAP for c in contents):
        return "turn_over_cap"
    for i in (0, 2):  # each USER turn's sha vs the FULL round-1 query-sha set
        if X.prompt_sha(contents[i]) in sha_set:
            return "query_sha_overlap"
    needles = lad_belt_needles(query_texts)
    if any(q in c for c in contents for q in needles):
        return "belt_query_text_substring"
    if any(excl["persona_system"] in c for c in contents):
        return "trained_context_containment"
    for demo in excl["icl_demo_texts"]:
        if any(demo in c for c in contents):
            return "trained_context_containment"
    return None


def _brl_r4_ladder(cfg: Cfg) -> dict:
    """The pinned r4 rung ladder (the r_long band source + the banked
    trained/rung recipe shas for exclusion 2): local lad staging path ->
    repo commit copy -> Hub (fail-loud both-miss). Asserts the banked band
    equals the plan-§11 pin verbatim."""
    path = _lad_inputs(cfg) / "prefix_ladder.json"
    if not path.exists():
        repo_copy = _lad_repo_ladder_path()
        if repo_copy.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_copy, path)
        else:
            from explore_persona_space.orchestrate import hub

            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/on_target_r4/inputs/prefix_ladder.json",
                path,
                repo_type="dataset",
            )
    ladder = json.loads(path.read_text())
    band = tuple(ladder["rungs"]["r_long"]["band"])
    assert band == BRL_BAND, (band, BRL_BAND, "r4 r_long band drift vs the plan §11 pin")
    return ladder


def _brl_banned_shas(r4_ladder: dict) -> tuple[set[str], set[str]]:
    """(banned recipe shas, banned content shas) for exclusion 2: the 3
    trained-prefix shas (banked in the r4 ladder exclusions block) + the 3 r4
    rung recipe shas (plan §4.2 exclusion 2, values banked — never re-derived
    here)."""
    banned_recipe = set(r4_ladder["exclusions"]["trained_prefix_recipe_shas"].values()) | {
        r4_ladder["rungs"][c]["recipe_sha256"] for c in X.R4_CONDS
    }
    banned_content = set(r4_ladder["exclusions"]["trained_prefix_content_shas"].values())
    return banned_recipe, banned_content


def _brl_prefix_record(
    cond: str, pair: tuple[str, str], kept_by_id: dict[str, dict], T: dict[str, int]
) -> dict:
    """One panel prefix record: 2 exchanges (ascending request_id), 4
    alternating turns under the content[:2000] recipe (plan §4.2 step 4-5)."""
    import types

    a, b = sorted(pair)
    turns = []
    for rid in (a, b):
        r = kept_by_id[rid]
        turns.append({"role": "user", "content": r["question"][:LAD_TURN_CONTENT_CAP]})
        turns.append({"role": "assistant", "content": r["completion"][:LAD_TURN_CONTENT_CAP]})
    t_total = T[a] + T[b]
    rec = {
        "context_id": X.R5_CONTEXT_ID_BY_COND[cond],
        "prefix_turns": turns,
        "request_ids": [a, b],
        "exchanges": [
            {
                "request_id": rid,
                "question_id": kept_by_id[rid]["question_id"],
                "variant_id": kept_by_id[rid]["variant_id"],
                "judge_mean": kept_by_id[rid]["_judge_mean"],
                "content_tokens": T[rid],
                "q_sha16": X.prompt_sha(kept_by_id[rid]["question"]),
                "c_sha16": X.prompt_sha(kept_by_id[rid]["completion"]),
                "truncated": (
                    len(kept_by_id[rid]["question"]) > LAD_TURN_CONTENT_CAP
                    or len(kept_by_id[rid]["completion"]) > LAD_TURN_CONTENT_CAP
                ),
                "question_shared_with_trained": kept_by_id[rid]["_shared_q"],
            }
            for rid in (a, b)
        ],
        "realized_tokens": t_total,
        "band": list(BRL_BAND),
        "in_band": BRL_BAND[0] <= t_total <= BRL_BAND[1],
        "question_shared_request_ids": [rid for rid in (a, b) if kept_by_id[rid]["_shared_q"]],
        "turns_sha256": lad_turns_sha(turns),
    }
    shim = types.SimpleNamespace(
        context_id=rec["context_id"], system=None, prefix_turns=tuple(turns), user_wrap=None
    )
    rec["recipe_sha256"] = _pfx_prefix_sha(shim)
    rec["content_sha256"] = _lad_content_sha(shim)
    return rec


def _brl_per_arm_mix_overlap(cfg: Cfg, prefixes: dict[str, dict]) -> dict:
    """Disclosure-only per-ARM per-turn text-overlap counts vs each arm's OWN
    training mix (plan §4.2 question-sharing disclosure; unresolvable ->
    'unknown', never a gate)."""
    from explore_persona_space.orchestrate import hub

    reg_path = cfg.out_root / "arm_registry.json"
    if not reg_path.exists():
        hub.stage_hub_file(X.HF_DATA_REPO, f"{X.HF_PREFIX}/arm_registry.json", reg_path)
    reg = json.loads(reg_path.read_text())
    src_by_arm = reg.get("mix_pos_sources", {})
    out: dict[str, dict] = {}
    for arm in X.R5_ARMS:
        src = src_by_arm.get(arm)
        if not src or "pos_path" not in src:
            out[arm] = {"overlap": "unknown", "reason": "no mix_pos_sources registry entry"}
            continue
        local = _brl_inputs(cfg) / "mix_overlap" / arm / Path(src["pos_path"]).name
        if not local.exists():
            hub.stage_hub_file(X.HF_DATA_REPO, src["pos_path"], local, repo_type="dataset")
        qs: set[str] = set()
        cs: set[str] = set()
        for line in local.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                q, c = _brl_msg_qc(row, arm)
            except (AssertionError, KeyError, TypeError):
                continue  # non-positive layout rows (disclosure-only tolerance)
            qs.add(X.prompt_sha(q))
            cs.add(X.prompt_sha(c))
        per_prefix = {}
        for cond, rec in prefixes.items():
            per_prefix[cond] = {
                "n_question_overlap": sum(1 for e in rec["exchanges"] if e["q_sha16"] in qs),
                "n_completion_overlap": sum(1 for e in rec["exchanges"] if e["c_sha16"] in cs),
            }
        out[arm] = {"overlap": per_prefix, "pos_path": src["pos_path"]}
    return out


def _brl_mirror_r_results(cfg: Cfg) -> None:
    """Unconditional idempotent pre-dispatch mirror of the COMMITTED r3+r4
    result inputs to HF `{prefix}/on_target_r5/inputs/r_results/` (plan §4.2
    lane transport: the fellows lane rsync-excludes eval_results/). Layout =
    the repo-relative paths under eval_results/issue_1768/, so the pod-side
    staging arithmetic is `results_dir`-relative. ONE upload_folder commit
    (never a per-file loop — the #664 504-storm rule)."""
    from explore_persona_space.orchestrate import hub

    src_root = REPO_ROOT / "eval_results" / "issue_1768"
    mirror = _brl_inputs(cfg) / "r_results"
    rels = ["on_target/map_change_on_target.json", "on_target/m0_prefix_effect.json"]
    rels += [
        f"on_target/percell/{a}_L{layer}_{s}.json"
        for a in X.R5_ARMS
        for layer in X.LAYERS
        for s in ("bare_n", "control", "own")
    ]
    rels += [f"on_target/fits_bare_n/{a}_L{layer}.json" for a in X.R5_ARMS for layer in X.LAYERS]
    rels += [
        f"on_target_r4/percell/{a}_L{layer}_{s}.json"
        for a in X.R5_ARMS
        for layer in X.LAYERS
        for s in X.R4_CONDS  # r_long is the comparator; r_short/r_mid ride for figures + dose
    ]
    rels += [
        "on_target_r4/map_change_ladder.json",
        "on_target_r4/m0_rung_effect.json",
        "on_target_r4/inputs/prefix_ladder.json",
    ]
    for rel in rels:
        src = src_root / rel
        assert src.exists(), (
            f"[brl_build] r_results mirror source missing from the repo tree: {src}"
        )
        dst = mirror / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
    url = hub._upload(
        mirror,
        repo_id=X.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{cfg.hf_prefix}/{BRL_R_RESULTS}",
    )
    if not url:
        raise RuntimeError("[brl_build] r_results mirror upload returned no path")
    logger.info("[brl_build] r_results mirror uploaded (%d files)", len(rels))


def _brl_build_publish(cfg: Cfg, panel: dict) -> None:
    """Repo-copy the pinned panel for the pre-dispatch git commit + upload it
    and the r_results mirror to the Hub. Production only: a smoke run must
    never touch the committed path or the production bucket (the smoke
    hf_prefix is `_smoke`-suffixed anyway; the repo copy is the committed
    artifact the smoke-output rule protects)."""
    from explore_persona_space.orchestrate import hub

    src = _brl_inputs(cfg) / "prefix_ladder_r5.json"
    if cfg.smoke:
        logger.info("[brl_build] smoke: repo-copy + Hub publish skipped (production-only)")
        return
    repo_copy = _brl_repo_panel_path()
    if not repo_copy.exists() or repo_copy.read_text() != src.read_text():
        repo_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, repo_copy)
        logger.info("[brl_build] panel copied to %s (commit pre-dispatch)", repo_copy)
    if not cfg.upload:
        logger.info("[brl_build] upload disabled (--no-upload)")
        return
    url = hub._upload(
        src,
        repo_id=X.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{cfg.hf_prefix}/on_target_r5/inputs/prefix_ladder_r5.json",
        upload_as_file=True,
    )
    if not url:
        raise RuntimeError("[brl_build] prefix_ladder_r5.json upload returned no path")
    _brl_mirror_r_results(cfg)


def phase_brl_build(cfg: Cfg) -> None:
    """brl_build (VM-side CPU, pre-dispatch): the §4.2 panel derivation in
    full — every step re-computed and asserted at FULL grain in smoke AND
    production (the pool is 36 rows; #1817). Smoke differs ONLY in skipping
    the repo-copy/Hub publish."""
    _phase("brl_build")
    _status(cfg, "brl_build")
    from transformers import AutoTokenizer

    dest = _brl_inputs(cfg) / "prefix_ladder_r5.json"
    if dest.exists():
        logger.info("[brl_build] prefix_ladder_r5.json present — resume skip; re-publishing")
        _brl_build_publish(cfg, json.loads(dest.read_text()))
        return
    pool = _brl_stage_pool(cfg)
    kept, derivation = _brl_derive_kept(pool)  # steps 1-3 + the 6-row pin (kill b)
    kept_by_id = {r["request_id"]: r for r in kept}
    r4_ladder = _brl_r4_ladder(cfg)
    banned_recipe, banned_content = _brl_banned_shas(r4_ladder)
    excl = _lad_trained_exclusion_material()
    sha_set, query_texts = _lad_full_grain_samples(cfg)
    # Panel token counts ALWAYS use the PRODUCTION tokenizer (plan §11) — the
    # r4 band anchors were measured under it; a model_override never rebinds it.
    tok = AutoTokenizer.from_pretrained(X.BASE_MODEL)
    T = {}
    for r in kept:
        q = r["question"][:LAD_TURN_CONTENT_CAP]
        c = r["completion"][:LAD_TURN_CONTENT_CAP]
        T[r["request_id"]] = len(tok(q, add_special_tokens=False)["input_ids"]) + len(
            tok(c, add_special_tokens=False)["input_ids"]
        )
    pairing, n_in_band, enumeration = brl_select_pairing(T)
    if n_in_band < 2:
        raise RuntimeError(
            f"[brl_build] kill criterion (b): only {n_in_band} in-band prefix(es) under "
            "EVERY pairing — the tokenizer-ratio surprise exceeded the pre-registered "
            "fallback (plan §4.2 step 7); failure_class: data (re-plan the band, never "
            "silently widen)"
        )
    if n_in_band < 3:
        logger.info(
            "[brl_build] pre-registered fallback engaged: best pairing has %d/3 in-band "
            "prefixes (plan §4.2 step 7) — out-of-band realized T recorded; the "
            "dose-interpolated secondary read de-confounds it",
            n_in_band,
        )
    assign = brl_assign_conds(pairing, {rid: kept_by_id[rid]["_judge_mean"] for rid in T})
    prefixes = {
        cond: _brl_prefix_record(cond, pair, kept_by_id, T) for cond, pair in assign.items()
    }
    # Mechanical exclusion screens (plan §4.2 exclusions 1-6), each an assert:
    used_ids: list[str] = []
    for cond, rec in prefixes.items():
        reason = brl_prefix_reject(rec["prefix_turns"], excl, sha_set, query_texts)
        assert reason is None, (cond, reason, "builder exclusion violated")
        for e in rec["exchanges"]:  # (1) trained-row disjointness, re-asserted per exchange
            assert [e["q_sha16"], e["c_sha16"]] not in derivation["emitted_qc_sha16_pairs"], (
                cond,
                e["request_id"],
            )
        assert rec["recipe_sha256"] not in banned_recipe, (cond, "recipe sha collides")
        assert rec["content_sha256"] not in banned_content, (cond, "content sha collides")
        used_ids += list(rec["request_ids"])
    assert len(used_ids) == 6 and len(set(used_ids)) == 6, ("exclusion 5", used_ids)
    panel = {
        "prefixes": prefixes,
        "pairing": {
            "rule": (
                "max in-band count over all 15 perfect pairings; tie-break (i) max "
                "min-prefix T, (ii) lexicographically smallest sorted request_id tuple "
                "(plan §4.2 step 6)"
            ),
            "assignment_rule": (
                "pairs -> b_rel1..b_rel3 in ascending total banked judge mean "
                "(tie: ascending min request_id)"
            ),
            "n_in_band": n_in_band,
            "fallback_engaged": n_in_band < 3,
            "enumeration": enumeration,
        },
        "band_source": "r4 prefix_ladder.json rungs.r_long.band (verbatim; plan §11)",
        "derivation": derivation,
        "exclusions": {
            "trained_prefix_recipe_shas": r4_ladder["exclusions"]["trained_prefix_recipe_shas"],
            "trained_prefix_content_shas": r4_ladder["exclusions"]["trained_prefix_content_shas"],
            "r4_rung_recipe_shas": {c: r4_ladder["rungs"][c]["recipe_sha256"] for c in X.R4_CONDS},
            "belt_min_query_chars": LAD_BELT_MIN_QUERY_CHARS,
            "screens": [
                "trained-row (question, completion) sha-pair disjointness vs the 20 "
                "emitted rows + train_mix positives",
                "trained-prefix + r4-rung recipe/content sha disjointness",
                "query-corpus sha disjointness (each user turn vs the full round-1 "
                f"sample) + substring belt (needles >= {LAD_BELT_MIN_QUERY_CHARS} chars)",
                "trained-context non-containment (persona system + icl demos)",
                "cross-prefix request_id distinctness (each of the 6 used exactly once)",
                "shape (user, assistant, user, assistant); non-empty capped contents",
            ],
            "language_screens": (
                "N/A — curated English datagen bank, judge-filtered (plan §4.2: stated, "
                "not screened)"
            ),
        },
        "per_arm_mix_overlap": _brl_per_arm_mix_overlap(cfg, prefixes),
        "smoke": cfg.smoke,
        **_meta(),
    }
    _atomic_json(dest, panel)
    logger.info(
        "[brl_build] panel pinned: %s (n_in_band=%d, fallback=%s)",
        {c: prefixes[c]["request_ids"] for c in X.R5_CONDS},
        n_in_band,
        n_in_band < 3,
    )
    _brl_build_publish(cfg, panel)


def _brl_stage_inputs(cfg: Cfg) -> None:
    """brl0 staging: pinned panel (local -> repo checkout -> Hub, fail-loud
    both-miss) + the pfx sample (production: the round-3 corpus_sample_pfx
    VERBATIM from the Hub — never re-derived; --smoke derives the tiny sample
    via the round-3 `_build_pfx_sample` path, the lad0 convention)."""
    from explore_persona_space.orchestrate import hub

    panel_path = _brl_inputs(cfg) / "prefix_ladder_r5.json"
    if not panel_path.exists():
        repo_copy = _brl_repo_panel_path()
        if repo_copy.exists():
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_copy, panel_path)
        else:
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/on_target_r5/inputs/prefix_ladder_r5.json",
                panel_path,
                repo_type="dataset",
            )
    canonical = _pfx_inputs(cfg) / "corpus_sample_pfx.json"
    if not canonical.exists():
        if cfg.smoke:
            _build_pfx_sample(cfg)  # fixture-derived tiny sample (p0 --smoke ran)
        else:
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/on_target/inputs/corpus_sample_pfx.json",
                canonical,
                repo_type="dataset",
            )


def _brl_recheck_exclusions(cfg: Cfg, tok, panel: dict) -> dict:
    """Kill criterion (d): re-assert the §4.2 exclusion set + band membership
    against the FULL pinned panel, the FULL-grain samples, AND a fresh pool
    re-derivation (never the smoke slice — the #1817 rule). Any violation =
    builder bug, fail loud."""
    import types

    pool = _brl_stage_pool(cfg)
    kept, _derivation = _brl_derive_kept(pool)  # re-derives + re-asserts the 6-row pin
    kept_by_id = {r["request_id"]: r for r in kept}
    excl = _lad_trained_exclusion_material()
    sha_set, query_texts = _lad_full_grain_samples(cfg)
    r4_ladder = _brl_r4_ladder(cfg)
    banned_recipe, banned_content = _brl_banned_shas(r4_ladder)
    out = {}
    used_ids: list[str] = []
    for cond in X.R5_CONDS:
        rec = panel["prefixes"][cond]
        turns = rec["prefix_turns"]
        reason = brl_prefix_reject(turns, excl, sha_set, query_texts)
        assert reason is None, (cond, reason, "builder exclusion violated (kill d)")
        for e in rec["exchanges"]:
            r = kept_by_id[e["request_id"]]
            assert e["q_sha16"] == X.prompt_sha(r["question"]), (cond, e["request_id"])
            assert e["c_sha16"] == X.prompt_sha(r["completion"]), (cond, e["request_id"])
        t_tokens = sum(len(tok(t["content"], add_special_tokens=False)["input_ids"]) for t in turns)
        assert t_tokens == rec["realized_tokens"], (
            cond,
            t_tokens,
            rec["realized_tokens"],
            "tokenizer drift vs the build-time count",
        )
        lo, hi = rec["band"]
        assert (lo <= t_tokens <= hi) == rec["in_band"], (cond, t_tokens, rec["band"])
        shim = types.SimpleNamespace(
            context_id=rec["context_id"], system=None, prefix_turns=tuple(turns), user_wrap=None
        )
        assert _pfx_prefix_sha(shim) == rec["recipe_sha256"], (cond, "recipe sha drift")
        assert rec["recipe_sha256"] not in banned_recipe, (cond, "recipe sha collides (kill d)")
        assert _lad_content_sha(shim) not in banned_content, (cond, "content sha collides")
        assert lad_turns_sha(turns) == rec["turns_sha256"], (cond, "turns sha drift")
        used_ids += list(rec["request_ids"])
        out[cond] = {
            "request_ids": rec["request_ids"],
            "realized_tokens": t_tokens,
            "band": rec["band"],
            "in_band": rec["in_band"],
            "screens": "ALL PASS (full grain)",
        }
    assert len(used_ids) == 6 and len(set(used_ids)) == 6, ("exclusion 5", used_ids)
    n_in = sum(1 for c in X.R5_CONDS if panel["prefixes"][c]["in_band"])
    assert n_in >= 2, (n_in, "kill criterion (b) class: fewer than 2 in-band prefixes")
    return out


def _brl_cond_record(cfg: Cfg, context_id: str) -> dict:
    conds = json.loads((_brl_inputs(cfg) / "conditions.json").read_text())["conditions"]
    assert context_id in conds, (context_id, "b_rel condition not built at brl0", sorted(conds))
    return conds[context_id]


def phase_brl0(cfg: Cfg) -> None:
    """brl0: panel-corpus render + budgets + FULL-grain exclusion re-assert."""
    _phase("brl0_panel_corpus")
    _status(cfg, "brl0_panel_corpus")
    from transformers import AutoTokenizer

    _brl_stage_inputs(cfg)
    sample = X.load_pfx_sample(cfg.out_root)
    done = _brl_inputs(cfg) / "build_done.json"
    if done.exists():
        logger.info("[brl0] build_done.json present — resume skip")
        return
    X.register_r5_brel_contexts(cfg.out_root)
    panel = X.load_r5_brel_panel(cfg.out_root)
    tok = AutoTokenizer.from_pretrained(cfg.model_override or X.BASE_MODEL)
    conds: dict[str, dict] = {}
    for cond in X.R5_CONDS:
        cid = X.R5_CONTEXT_ID_BY_COND[cond]
        ctx = _lad_registry_context(cid)
        budgets = _pfx_budget(tok, ctx, sample["rows"])  # content overflow FAILS LOUD
        prefix_tokens = len(tok(ctx.render(tok, ""), add_special_tokens=False)["input_ids"])
        rec_panel = panel["prefixes"][cond]
        rec = {
            "tag": cond,
            "context_id": cid,
            "prefix_sha256": _pfx_prefix_sha(ctx),
            "recipe": _pfx_prefix_recipe(ctx),
            "prefix_tokens": prefix_tokens,
            "realized_content_tokens": rec_panel["realized_tokens"],
            "band": rec_panel["band"],
            "in_band": rec_panel["in_band"],
            "request_ids": rec_panel["request_ids"],
            "message_shape": {
                "has_system": ctx.system is not None,
                "n_prefix_turns": len(ctx.prefix_turns),
                "has_user_wrap": ctx.user_wrap is not None,
            },
            "budgets": budgets,
            "n_rows": len(sample["rows"]),
        }
        conds[cid] = rec
        _atomic_json(
            _brl_inputs(cfg) / f"corpus_{cond}.json",
            {
                **rec,
                "rows_sha256": hashlib.sha256(
                    "\n".join(r["sha"] for r in sample["rows"]).encode("utf-8")
                ).hexdigest(),
                **_meta(),
            },
        )
    _atomic_json(_brl_inputs(cfg) / "conditions.json", {"conditions": conds, **_meta()})
    recheck = _brl_recheck_exclusions(cfg, tok, panel)  # kill criterion (d)
    _atomic_json(_brl_inputs(cfg) / "exclusion_recheck.json", {"prefixes": recheck, **_meta()})
    _atomic_json(
        done,
        {"n_rows": len(sample["rows"]), "conditions": sorted(conds), **_meta()},
    )
    logger.info("[brl0] panel corpora + budgets + exclusion re-assert done (%d conds)", len(conds))


def run_brl_corpus_unit(cfg: Cfg, unit_id: str) -> None:
    """brl2 unit: b_rel-prefixed greedy gen -> TF span-means (prefix+ctx+resp),
    via the SAME `_prefixed_capture_core` the pfx2/lad2 units run. The panel
    registrar runs idempotently at point of use (fresh fan-out subprocesses
    inherit no registry state — the #1090-fu6/#1315 lessons)."""
    X.register_r5_brel_contexts(cfg.out_root)
    cid = X.r5_unit_context_id(unit_id)
    _prefixed_capture_core(
        cfg,
        unit_id,
        root=_brl_root(cfg),
        cid=cid,
        cond_rec=_brl_cond_record(cfg, cid),
        ctx=_lad_registry_context(cid),
    )


def _brl_pilot_cond(cfg: Cfg) -> str:
    """Pilot condition: the LONGEST realized-T prefix (worst case; plan §4.4
    brl1). Smoke pins b_rel1 so the pilot unit matches the smoke unit set."""
    if cfg.smoke:
        return "b_rel1"
    panel = X.load_r5_brel_panel(cfg.out_root)
    return max(X.R5_CONDS, key=lambda c: panel["prefixes"][c]["realized_tokens"])


def phase_brl1(cfg: Cfg) -> None:
    """brl1: ONE unit at production shape (worst-case longest prefix); gen +
    TF walls measured separately (kill criterion a)."""
    _phase("brl1_pilot")
    _status(cfg, "brl1_pilot")
    unit = X.r5_trained_unit(BRL_PILOT_ARM, _brl_pilot_cond(cfg))
    run_brl_corpus_unit(cfg, unit)
    man = json.loads((_brl_root(cfg) / "corpus_capture" / unit / "manifest.json").read_text())
    unit_wall_h = (man.get("gen_wall_s", 0.0) + man.get("tf_wall_s", 0.0)) / 3600.0
    ratio = unit_wall_h / BRL_BOOKED_UNIT_GPU_H
    _atomic_json(
        _brl_root(cfg) / "pilot" / "pilot_report.json",
        {
            "unit": unit,
            "gen_wall_s": man.get("gen_wall_s"),
            "tf_wall_s": man.get("tf_wall_s"),
            "unit_wall_h": unit_wall_h,
            "booked_unit_gpu_h": BRL_BOOKED_UNIT_GPU_H,
            "ratio": ratio,
            "smoke": cfg.smoke,
            **_meta(),
        },
    )
    print(f"[brl1] pilot {unit} wall={unit_wall_h:.2f}h ratio={ratio:.2f}", flush=True)
    _pfx_pilot_gate(ratio, cfg.smoke, tag="brl1", booked=BRL_BOOKED_UNIT_GPU_H)


def _brl_arms(cfg: Cfg) -> list[str]:
    """The round-5 arm scope (plan §4.1 — same 4 arms as r4); --arms filters
    INSIDE it; smoke = the plan-§4 smoke-parity arm."""
    if cfg.arms:
        want = set(cfg.arms)
        unknown = want - set(X.R5_ARMS)
        assert not unknown, f"--arms outside the r5 arm set: {sorted(unknown)}"
        return [a for a in X.R5_ARMS if a in want]
    if cfg.smoke:
        return [BRL_PILOT_ARM]
    return list(X.R5_ARMS)


def _brl_conds_capture(cfg: Cfg) -> tuple[str, ...]:
    return ("b_rel1",) if cfg.smoke else X.R5_CONDS


def _brl_unit_set(cfg: Cfg) -> list[str]:
    """brl2 units: shared base@b_rel units first, then arm@b_rel units
    (production: 3 + 12 = 15; smoke: base_content@b_rel1 + pilot@b_rel1)."""
    arms = _brl_arms(cfg)
    conds = _brl_conds_capture(cfg)
    bases = [X.r5_base_unit(c) for c in conds]
    trained = [X.r5_trained_unit(a, c) for a in arms for c in conds]
    return bases + trained


def _brl_expected_uploads(cfg: Cfg) -> list[str]:
    """Exact-set verify list: EVERY per-unit artifact file the corpus_capture
    tree carries — pooled.pt + manifest/spans/done sentinels + the raw-row
    rollout shards (the lad4 convention)."""
    expected = []
    tree = "on_target_r5/corpus_capture"
    local_tree = cfg.out_root / tree
    if local_tree.exists():
        for unit_dir in sorted(local_tree.iterdir()):
            if not (unit_dir / "pooled.pt").exists():
                continue
            names = ["pooled.pt"]
            names += [p.name for p in sorted(unit_dir.glob("raw_rows_*.jsonl"))]
            for extra in ("raw_rows.done.json", "rows_spans.json", "manifest.json"):
                if (unit_dir / extra).exists():
                    names.append(extra)
            expected += [f"{cfg.hf_prefix}/{tree}/{unit_dir.name}/{n}" for n in names]
    return expected


def phase_brl4(cfg: Cfg) -> None:
    """brl4: on_target_r5 tree upload + exact-set verify — BEFORE fits (#825)."""
    _phase("brl4_store_upload")
    _status(cfg, "brl4_store_upload")
    if not cfg.upload:
        logger.info("[brl4] upload disabled (--no-upload)")
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    expected = _brl_expected_uploads(cfg)
    done_path = _brl_root(cfg) / "upload_done.json"
    if done_path.exists():
        prior = json.loads(done_path.read_text())
        if prior.get("n_verified") == len(expected):
            logger.info(
                "[brl4] upload_done.json matches the expected store count (%d) — resume skip",
                len(expected),
            )
            return
        logger.info(
            "[brl4] expected store count changed (%s -> %d) — re-uploading",
            prior.get("n_verified"),
            len(expected),
        )
    uploaded = {}
    for name in BRL_UPLOAD_TREES:
        dest = _upload_tree(cfg, name)
        if dest:
            uploaded[name] = dest
    if expected:
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            X.HF_DATA_REPO,
            expected,
            path_in_repo=f"{cfg.hf_prefix}/on_target_r5",
            repo_type="dataset",
        )
        assert not missing, f"brl4 verify: {len(missing)} store files missing on Hub: {missing[:5]}"
    _atomic_json(
        done_path,
        {"uploaded": uploaded, "n_verified": len(expected), **_meta()},
    )


# ── entry ────────────────────────────────────────────────────────────────────


def _import_check() -> int:
    """Resolve every deferred import this driver hits on its REAL code paths."""
    from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
    from peft import PeftModel  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    import issue1481_cells  # noqa: F401
    import issue1586_cells  # noqa: F401
    import issue1768_directions  # noqa: F401
    import issue1768_fit  # noqa: F401
    import issue779_ffc_n1m_generate_capture  # noqa: F401
    import issue779_ffc_n50k_generate_capture  # noqa: F401
    import issue779_fitter_fair_comparison  # noqa: F401

    # pfx prefix-context resolution (plan v8 §4.2: the #1481 registry render)
    from issue1090_fu3_worker import ensure_context  # noqa: F401

    # p6 split-leg deferred imports (gen / judge / reduce; plan §9 Must-Fix)
    from issue779_common import (  # noqa: F401
        judge_rollouts_n5,
        load_extraction_artifacts,
    )
    from issue779_extract_rb import (  # noqa: F401
        RunningMean,
        _build_prompts,
        _dump_rollouts,
        _HFGenShim,
        _iter_rollout_records,
        _response_mean_activation,
        _vllm_generate_chunked,
    )
    from explore_persona_space.eval.generation import (  # noqa: F401
        cleanup_vllm,
        create_vllm_engine,
    )
    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        _build_generation_prompts,
        _generate_responses_vllm,
        _reap_vllm_engine,
        _teacher_forced_span_means,
        _vllm_enforce_eager,
        compute_prompt_spans,
    )
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        _upload,
        retry_transient,
        stage_hub_file,
        stage_hub_prefix,
        verify_repo_paths_uploaded,
    )
    from explore_persona_space.orchestrate.preflight import (  # noqa: F401
        assert_out_root_headroom,
    )

    # lad (round-4) deferred imports: the WildChat streaming builder + the
    # ladder registrar's lazy Context import (executed, not just named)
    from datasets import load_dataset  # noqa: F401

    from explore_persona_space.artifacts.context import CONTEXTS, Context  # noqa: F401

    try:  # vLLM is GPU-lane-only; absence is reported, not fatal, off-pod
        import vllm  # noqa: F401
    except ImportError as e:
        print(f"[import-check] vllm unavailable here: {e}", flush=True)
    print("[import-check] OK", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> tuple[Cfg, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True)
    # p6 (the CPU reduce) runs AFTER p7 by default: its GPU legs ride the p2
    # queue and the Batch-API poll overlaps p2..p7 (plan §9 Must-Fix)
    ap.add_argument("--phases", default="p0,p1,p2,p3,p4,p5,p7,p6")
    ap.add_argument("--smoke", action="store_true", help="pilot arm only, tiny slices")
    ap.add_argument("--arms", default="", help="comma-separated arm-id filter")
    ap.add_argument("--layers", default=",".join(str(x) for x in X.LAYERS))
    ap.add_argument("--tf-batch", type=int, default=X.TF_BATCH_SIZE)
    ap.add_argument("--model-override", default=None)
    ap.add_argument("--corpus-manifest-dir", type=Path, default=None)
    ap.add_argument("--valtest-file", type=Path, default=None)
    ap.add_argument("--smoke-rows", default="24,8,8")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--hf-prefix", default=None, help="upload prefix (smoke: <prefix>_smoke)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--unit", default=None, help="internal: run one unit `<phase>:<id>`")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    smoke_rows = tuple(int(x) for x in args.smoke_rows.split(","))
    assert len(smoke_rows) == 3, args.smoke_rows
    # --smoke ALWAYS uploads under a _smoke-suffixed prefix, even with an
    # explicit --hf-prefix: a smoke run must never write the production Hub
    # bucket (fired in the round-2 VM smoke — the explicit flag defeated the
    # default's auto-suffix and 8 smoke trees landed on the production prefix,
    # hand-cleaned via delete_folder the same hour).
    hf_prefix = args.hf_prefix or X.HF_PREFIX
    if args.smoke and not hf_prefix.endswith("_smoke"):
        hf_prefix = f"{hf_prefix}_smoke"
    cfg = Cfg(
        out_root=args.out_root,
        phases=tuple(p for p in args.phases.split(",") if p),
        smoke=args.smoke,
        arms=tuple(a for a in args.arms.split(",") if a),
        layers=tuple(int(x) for x in args.layers.split(",")),
        tf_batch=args.tf_batch,
        model_override=args.model_override,
        corpus_manifest_dir=args.corpus_manifest_dir,
        valtest_file=args.valtest_file,
        smoke_rows=smoke_rows,  # type: ignore[arg-type]
        upload=not args.no_upload,
        hf_prefix=hf_prefix,
        gpu_id=args.gpu_id,
    )
    return cfg, args


PHASE_HEADROOM_GB = {
    "p2": 175.0,
    "p3": 68.0,
    "p4": 20.0,
    "p5": 5.0,
    "p6": 5.0,
    "pnf": 40.0,
    # pfx (plan v8 §9 storage block): new stores ~10 GB + staged round-1
    # stores ~18 GB + rollout text ~1.5 GB + <=2 concurrent transient merged
    # models <=30 GB => driver asserts >=90 GB free at pfx2 entry.
    "pfx2": 90.0,
    "pfx3": 40.0,
    # lad2 (plan v10 §9 storage block): new stores ~5.3 GB + rollout text
    # ~1.5 GB + staged inputs + one transient merged model <=15 GB => the
    # driver asserts >=60 GB free at lad2 entry.
    "lad2": 60.0,
    # brl2 (plan v13 §9 storage block): identical 15-unit shape to lad2 —
    # new stores ~5.3 GB + rollout text ~1.5 GB + staged inputs ~0.3 GB +
    # one transient merged model <=15 GB => >=60 GB free asserted at entry.
    "brl2": 60.0,
}
# pnf: ~2 GB staged base text + one transient merged/ft model <=16 GB + ~0.5 GB
# stores (plan v7 §9 storage block); resume-aware gate below skips it when all
# 12 replicate stores are already present.
# p2 covers the p6 GPU legs it now hosts (~0.4 GB acts per arm << the corpus
# budget); p6 itself is the CPU reduce (small r_b writes only). p2/p3 carry
# the amendment's +16 ft units (~+22 GB run-wide against the declared 250 GB
# RunPod volume / 400 GB GCP disk — plan §9 amendment costing).


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
        if phase in PHASE_HEADROOM_GB and _pending_units(cfg, phase):
            # resume-aware: gate only when this phase has PENDING units (#1586)
            need = PHASE_HEADROOM_GB[phase] if not cfg.smoke else 2.0
            assert_out_root_headroom(cfg.out_root, need, phase=phase)
        if phase == "p0":
            phase_p0(cfg)
        elif phase == "p1":
            phase_p1(cfg)
        elif phase == "p2":
            _fanout_phase(cfg, "p2", "p2_corpus_capture")
        elif phase == "p3":
            _fanout_phase(cfg, "p3", "p3_corpus_tf")
        elif phase == "p4":
            _fanout_phase(cfg, "p4", "p4_panels")
        elif phase == "p5":
            _fanout_phase(cfg, "p5", "p5_delta")
        elif phase == "p6":
            phase_p6_reduce(cfg)
        elif phase == "p7":
            phase_p7(cfg)
        elif phase == "pnf":
            phase_noise_floor(cfg)
        elif phase == "pfx0":
            phase_pfx0(cfg)
        elif phase == "pfx1":
            phase_pfx1(cfg)
        elif phase == "pfx2":
            _fanout_phase(cfg, "pfx2", "pfx2_prefixed_capture")
        elif phase == "pfx3":
            _fanout_phase(cfg, "pfx3", "pfx3_prefixed_tf")
        elif phase == "pfx4":
            phase_pfx4(cfg)
        elif phase == "lad_build":
            phase_lad_build(cfg)
        elif phase == "lad0":
            phase_lad0(cfg)
        elif phase == "lad1":
            phase_lad1(cfg)
        elif phase == "lad2":
            _fanout_phase(cfg, "lad2", "lad2_rung_capture")
        elif phase == "lad4":
            phase_lad4(cfg)
        elif phase == "brl_build":
            phase_brl_build(cfg)
        elif phase == "brl0":
            phase_brl0(cfg)
        elif phase == "brl1":
            phase_brl1(cfg)
        elif phase == "brl2":
            _fanout_phase(cfg, "brl2", "brl2_brel_capture")
        elif phase == "brl4":
            phase_brl4(cfg)
        else:
            raise ValueError(f"unknown phase {phase}")
    _status(cfg, "done")
    _phase("done")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # #1689/#952 finalize-race guard, HARD form: with a consumed WildChat
    # streaming IterableDataset in-process (lad_build), plain sys.exit(rc)
    # MEASURABLY still aborts at interpreter finalize (PyGILState_Release,
    # rc=134 — 2026-07-31 probe; `del row, ds` + gc.collect() applied and
    # insufficient), which would kill the `capture && fit` workload chain
    # after a COMPLETED phase. Every output is already durably written
    # (atomic tmp+os.replace) and both streams are flushed above, so the
    # skipped finalization has nothing left to do. Canonical coerced form
    # (the tests/test_issue1689_os_exit_shutdown_bypass.py idiom).
    os._exit(rc if isinstance(rc, int) else 0)
