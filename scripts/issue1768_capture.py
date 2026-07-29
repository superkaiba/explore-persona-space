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
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
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
        ok = hub.retry_transient(
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
        ok = hub.retry_transient(
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
        return hub.retry_transient(
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
    cfg: Cfg, unit_id: str, model_path: str, prompts: list[str], start_idx: int, out_dir: Path
) -> None:
    """Chunked vLLM greedy generation (the #664 chunk rule) with shard persist.

    Rows follow the `_generate_responses_vllm` schema (+ sha/text extras); the
    engine is built ONCE, chunked `generate` calls carry `use_tqdm=False`
    (#613), prefix caching is OFF by default for this real-user corpus
    (gotchas.md pre-launch checklist; EPM_VLLM_DISABLE_PREFIX_CACHING=0
    re-enables), and teardown rides `_reap_vllm_engine`.
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
        max_model_len=X.MAX_MODEL_LEN,
        enable_prefix_caching=os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "1") == "0",
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_new)
    eos_id = tokenizer.eos_token_id
    try:
        n_chunks = -(-len(prompts) // GEN_CHUNK)
        for ci, s in enumerate(range(0, len(prompts), GEN_CHUNK)):
            chunk = prompts[s : s + GEN_CHUNK]
            rendered, keys = _build_generation_prompts(tokenizer, {unit_id: None}, chunk)
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


def _load_or_generate_rows(cfg: Cfg, out_dir: Path, unit_id: str, model_path: str) -> list[dict]:
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
        _generate_rows_vllm(cfg, unit_id, model_path, remaining, len(existing), out_dir)
    rows = _read_shards(out_dir)
    assert len(rows) == len(prompts), (unit_id, len(rows), len(prompts))
    _atomic_json(done, {"n_rows": len(rows), **_meta()})
    return rows


def _attach_spans(
    tokenizer, prompts: list[str], rows: list[dict]
) -> tuple[list[dict], int, dict, int]:
    """Compute prefix/context spans per row; drop invalid rows with counts.

    Returns (kept_rows, n_dropped, seam_counts, n_distinct_prefix).
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
            None,
            prompts[r["question_idx"]],
            r["prompt_token_ids"],
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
        present[name] = hub.retry_transient(
            lambda p=hub_path: api.file_exists(X.HF_DATA_REPO, p, repo_type="dataset"),
            what=f"panel stage probe {hub_path}",
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
            "the §4.8 drop — recorded as a quantified scope caveat)", n_dp, PREFIX_REOPEN_FLOOR
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
    unit = unit_arg.split(":")[-1]
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
            nxt_i = next(
                (
                    i
                    for i, ua in enumerate(queue)
                    if (ua in barrier or not barrier_live)
                    and (not _needs_merge(cfg, ua) or active_merges < merge_slots)
                    and _unit_ready(cfg, phase, ua)
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


PHASE_HEADROOM_GB = {"p2": 175.0, "p3": 68.0, "p4": 20.0, "p5": 5.0, "p6": 5.0}
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
