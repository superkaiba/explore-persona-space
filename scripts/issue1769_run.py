"""Issue #1769 — prefill-vs-decode behavioral commitment: pod-side driver.

Mechanistic register (plan v5 §4): the intervention adds ``alpha * r_B[20]``
(the #779 persona-vectors direction row for decoder layer 20) to the
hidden-state output of ``model.model.layers[20]`` over a specified token
range. Four arms per trait:

- ``neither``      — no hook (control floor)
- ``prefill_only`` — ALL prompt positions, FIRST forward pass only
                     (``DeltaHook(prefill_all=True)``)
- ``decode_only``  — every generated position at each decode step, never the
                     first forward (``DeltaHook(decode_only=True)``)
- ``both``         — the existing #1415 ``all_positions`` mode

Phases (plan §4 DAG: P0 -> G0 -> G1 -> finalize):

- ``p0``       — artifact-existence + disjointness checks via the CONSUMER
                 loaders (r_B shape (28, 3584); eval battery + rubric per
                 trait; eval ∩ ORIGINAL extraction == ∅; provenance recorded).
- ``pilot``    — G0 timing gate: 1 question x ``both`` x alpha=1.0, replicated
                 to B = gen_batch rows, N draws through ``generate_batch`` (the
                 production entrypoint at the sweep chunk shape); HALT with
                 ``pilot_gate_report.json`` + rc=7 when s/sample > 4.7 unless
                 ``--force`` (the #1415 pilot-gate pattern; measured basis
                 2.309 s/sample at this exact kernel + shape).
- ``grid``     — G1: 600 cells (3 traits x [neither x 20q + 3 arms x 3 alphas
                 x 20q]), checkpoint-per-cell metadata JSON keyed on the FULL
                 fingerprint (trait/arm/alpha/qid + code SHA, model id,
                 temperature, max_new_tokens, layer, r_B revision, seed base);
                 a fingerprint mismatch re-runs the cell (#952 gate-5 shape).
                 Incremental raw-completion uploads per (trait, arm, alpha)
                 group. Shardable by (trait x arm) via ``--shard/--n-shards``
                 (the dispatcher pins ``CUDA_VISIBLE_DEVICES`` per shard).
- ``finalize`` — merges per-cell metadata into ``cells_manifest.json``,
                 asserts FULL grid coverage (row-coverage invariant: every
                 paired contrast row comes from this run's own G1), and
                 verifies the raw-completion upload set.

``--tiny`` runs the FULL control flow on CPU with a from-config 2-layer Qwen
over the REAL vocab (tiny-real standard): ALL 4 arms x 1 trait x 2 questions
x 2 draws x 2 alphas; P0 still runs the REAL artifact checks (KB-scale HF
fetches); the r_B delta VALUES are config-scaled (seeded randn at the tiny
hidden size — the real (3584,) row cannot feed a tiny-hidden model); uploads
run through the identical call path into a local mirror
(``upload_mode='local-mirror'``, the #1415 pattern — control flow never
forks).

Pod-side contract: this driver never shells out to ``scripts/task.py``;
progress is ``[phase=...]`` + per-unit ``[grid] unit k/N`` log lines; the
end-of-run sentinel is written by ``scripts/issue1769_dispatch.sh``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402  (after load_dotenv: shared-VM thread caps)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue779_common as c779  # noqa: E402
import issue922_common as c922  # noqa: E402
from issue1415_run_phase1 import _unwrap_rb_tensor  # noqa: E402

from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    DeltaHook,
    coherence_check,
    generate_batch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1769_run")

ISSUE = 1769
REPO_ROOT = Path(__file__).resolve().parent.parent
HF_DATA_REPO = c779.HF_DATA_REPO  # superkaiba1/explore-persona-space-data
HF_OUT_PREFIX = "issue1769_prefill_decode"
R_B_REVISION = c922.HF_REVISION  # the #841 pin — r_B fetched via the issue922 path

MODEL_ID = c779.DEFAULT_MODEL  # Qwen/Qwen2.5-7B-Instruct
LAYER = 20  # issue1415_run_phase1.PRIMARY_LAYER_FULL (plan §11)
ALPHAS = (1.0, 2.0, 4.0)  # top 3 rungs of issue1415 ALPHA_GRID (plan §11)
TRAITS = c922.TRAITS  # ("evil", "sycophancy", "hallucination")
ARMS = ("neither", "prefill_only", "decode_only", "both")
STEERED_ARMS = ("prefill_only", "decode_only", "both")
N_DRAWS = 10  # issue1415 N_DRAWS_FULL
MAX_NEW_TOKENS = 1024  # issue1415 MAX_NEW_TOKENS_FULL
TEMPERATURE = 1.0  # issue1415 TEMPERATURE
SEED_BASE = 42  # issue1415 SEED_BASE (per-draw seeds 42..51)
GEN_BATCH = 8  # issue1415 pilot/chunk shape (pilot.json pilot_batch: 8)
N_QUESTIONS = 20  # persona-vectors evaluation battery per trait
PILOT_MAX_S_PER_SAMPLE = 4.7  # plan §7 G0 (issue1415 PILOT_MAX_S_PER_SAMPLE)
RC_PILOT_GATE = 7  # designed artifact-routed HALT (issue1415 RC_PILOT_GATE)
RC_K2_GATE = 9  # designed artifact-routed HALT: plan §7 K2 dose-ladder coherence
K2_COHERENCE_MIN = 0.5  # #1415 condition gate (>= 50% of draws coherent)
PILOT_TRAIT = "evil"  # committed-constants trait (no HF read for its battery)

EXPECTED_RB_SHAPE = (c922.EXPECTED_LAYERS, c922.EXPECTED_HIDDEN)  # (28, 3584)


# ── config ────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    tiny: bool
    out_root: Path
    bulk_root: Path
    model_id: str = MODEL_ID
    traits: tuple[str, ...] = TRAITS
    alphas: tuple[float, ...] = ALPHAS
    n_questions: int = N_QUESTIONS
    n_draws: int = N_DRAWS
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = TEMPERATURE
    seed_base: int = SEED_BASE
    gen_batch: int = GEN_BATCH
    layer: int = LAYER
    hidden: int = c922.EXPECTED_HIDDEN
    n_model_layers: int = c922.EXPECTED_LAYERS
    force: bool = False
    shard: int = 0
    n_shards: int = 1
    upload_mode: str = "hf"  # "hf" | "local-mirror" (tiny default)
    hf_prefix: str = HF_OUT_PREFIX  # HF data-repo prefix for every upload (fu1 override)
    extra: dict = field(default_factory=dict)


def tiny_config(out_root: Path, bulk_root: Path, force: bool = False) -> RunConfig:
    """Tiny-real smoke config: 1 trait x 4 arms x 2 alphas x 2 questions x 2
    draws on a from-config 2-layer real-vocab Qwen (CPU). The tiny LAYER is 1
    (< n_model_layers=2); every other knob keeps the production control flow."""
    return RunConfig(
        tiny=True,
        out_root=out_root,
        bulk_root=bulk_root,
        traits=(PILOT_TRAIT,),
        alphas=(1.0, 2.0),
        n_questions=2,
        n_draws=2,
        max_new_tokens=16,
        gen_batch=2,
        layer=1,
        hidden=64,
        n_model_layers=2,
        force=force,
        upload_mode="local-mirror",
    )


_REPRO_CACHE: dict | None = None


def repro_metadata(cfg: RunConfig) -> dict:
    """Reproducibility metadata for every persisted result JSON (CLAUDE.md)."""
    global _REPRO_CACHE
    if _REPRO_CACHE is None:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                env={**os.environ},
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning("git rev-parse failed (%s) — recording commit=unknown", exc)
            commit = "unknown"
        import transformers

        _REPRO_CACHE = {
            "git_commit": commit,
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        }
    return {
        **_REPRO_CACHE,
        "model_id": cfg.model_id,
        "tiny": cfg.tiny,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def cell_fingerprint(cfg: RunConfig) -> dict:
    """The resume-provenance fingerprint (plan §4 G1): a resumed run re-runs
    any cell whose recorded fingerprint mismatches — bare output existence
    never vouches for a cell across a code-fix round (#952 gate-5 shape)."""
    return {
        "code_sha": repro_metadata(cfg)["git_commit"],
        "model_id": cfg.model_id,
        "temperature": cfg.temperature,
        "max_new_tokens": cfg.max_new_tokens,
        "layer": cfg.layer,
        "rb_revision": "tiny-randn" if cfg.tiny else R_B_REVISION,
        "seed_base": cfg.seed_base,
        "n_draws": cfg.n_draws,
        "tiny": cfg.tiny,
    }


def write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic JSON write (tmp + rename) — checkpoint-per-cell safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


# ── artifact loaders (the CONSUMER loaders — plan §4 P0) ──────────────


def load_rb_row(cfg: RunConfig, trait: str) -> torch.Tensor:
    """r_B row for ``cfg.layer`` via the consumer load path.

    Production: ``issue922_common._fetch`` at the pinned revision +
    ``_unwrap_rb_tensor`` + shape assert (28, 3584); row L indexes MODEL layer
    L directly (verified against ``issue1415_run_phase1.rb_delta``, whose 1d
    arm consumed this exact bank as ``rb[layer]``). Tiny: seeded randn at the
    tiny (n_layers, hidden) — the GPU-scale input is the ONE config-scaled
    substitution in the smoke (real r_B rows cannot feed a tiny-hidden model).
    """
    if cfg.tiny:
        torch.manual_seed(ISSUE + TRAITS.index(trait))
        rb = torch.randn(cfg.n_model_layers, cfg.hidden)
    else:
        local = c922._fetch(f"r_b/{trait}.pt", revision=R_B_REVISION)
        rb = _unwrap_rb_tensor(torch.load(local, map_location="cpu", weights_only=True))
        assert rb.shape == EXPECTED_RB_SHAPE, (trait, rb.shape)
    assert rb.shape == (cfg.n_model_layers, cfg.hidden), (trait, rb.shape)
    return rb[cfg.layer].float()


def load_trait_artifact(trait: str) -> dict:
    """The trait's persona-vectors artifact via the pinned consumer path.

    evil short-circuits to the committed constants ``issue779_common.
    EVIL_ARTIFACTS`` (``artifacts/evil.json`` does NOT resolve on HF — plan
    §4); sycophancy/hallucination fetch ``issue779_monitoring/artifacts/
    {trait}.json`` at ``HF_REVISION_LATE`` (the issue922_common pin).
    """
    if trait == "evil":
        return c779.EVIL_ARTIFACTS
    with open(c922._fetch(f"artifacts/{trait}.json", revision=c922.HF_REVISION_LATE)) as f:
        return json.load(f)


def eval_questions(cfg: RunConfig, trait: str) -> list[str]:
    """The 20-question EVALUATION battery via ``issue922_common.eval_questions``
    (realized key ``eval_questions`` — never a guessed key path), truncated to
    ``cfg.n_questions`` under tiny."""
    qs = c922.eval_questions(trait)
    assert len(qs) == N_QUESTIONS, (trait, len(qs))
    return qs[: cfg.n_questions]


def resolve_rubric(trait: str) -> str:
    """The trait's persona-vectors eval_prompt (verbatim, {question}/{answer}
    slots) + the #1415 reason-then-score wrapper — the issue1415_judge recipe
    with the artifact fetched at the PINNED revision (the issue922 path)."""
    from issue1769_judge import resolve_rubric as _rr

    return _rr(trait)


def extraction_questions_original(trait: str) -> list[str]:
    """The ORIGINAL extraction-question set r_B consumed.

    evil: committed constants. sycophancy/hallucination: ``extraction_
    questions`` from the artifact at ``HF_REVISION_LATE`` — asserted ORIGINAL
    per the artifact's own ``reconstruction.regenerated`` metadata (the #922
    incident regenerated instruction/eval_questions/eval_prompt, NOT the
    extraction set)."""
    art = load_trait_artifact(trait)
    if trait != "evil":
        regenerated = (art.get("reconstruction") or {}).get("regenerated", [])
        assert "extraction_questions" not in regenerated, (
            f"{trait}: extraction_questions were REGENERATED after the r_B capture — "
            "the disjointness assert would no longer read the ORIGINAL set (plan §4 P0)"
        )
    qs = art["extraction_questions"]
    assert len(qs) >= 20, (trait, len(qs))
    return list(qs)


# ── P0: artifact-existence + disjointness check ───────────────────────


def phase_p0(cfg: RunConfig) -> dict:
    """Plan §4 P0 (re-run pod-side): consumer-loader existence + shape +
    disjointness + provenance record. REAL in tiny mode too (KB-scale reads);
    only the grid's delta VALUES are config-scaled under tiny."""
    print("[phase=p0]", flush=True)
    records: dict[str, dict] = {}
    for trait in TRAITS:
        # (a) r_B resolves + shape (28, 3584) via the consumer loader.
        local = c922._fetch(f"r_b/{trait}.pt", revision=R_B_REVISION)
        rb = _unwrap_rb_tensor(torch.load(local, map_location="cpu", weights_only=True))
        assert rb.shape == EXPECTED_RB_SHAPE, (trait, rb.shape)
        # (b) evaluation battery + rubric load through the consumer loaders.
        qs = c922.eval_questions(trait)
        assert len(qs) == N_QUESTIONS, (trait, len(qs))
        rubric = resolve_rubric(trait)
        assert "{question}" in rubric and "{answer}" in rubric, trait
        # (c) disjointness vs the ORIGINAL extraction set + provenance record.
        extraction = extraction_questions_original(trait)
        overlap = sorted(set(qs) & set(extraction))
        assert not overlap, (trait, overlap)
        records[trait] = {
            "rb_path": f"{c922.HF_PREFIX}/r_b/{trait}.pt",
            "rb_revision": R_B_REVISION,
            "rb_shape": list(rb.shape),
            "rb_row_indexing": "row L = model layer L (issue1415_run_phase1.rb_delta contract)",
            "n_eval_questions": len(qs),
            "n_extraction_questions": len(extraction),
            "eval_extraction_overlap": overlap,
            "eval_questions_provenance": c922.eval_questions_provenance(trait),
            "rubric_sha256_prefix": __import__("hashlib").sha256(rubric.encode()).hexdigest()[:16],
        }
        logger.info(
            "[p0] %s OK (rb %s, %d eval qs disjoint from %d extraction qs, provenance=%s)",
            trait,
            tuple(rb.shape),
            len(qs),
            len(extraction),
            records[trait]["eval_questions_provenance"],
        )
    out = {
        "phase": "p0",
        "traits": records,
        "note": (
            "sycophancy/hallucination eval questions are post-hoc REGENERATED "
            "(the #922 incident) — pre-registered disposition, carried as a "
            "clean-result data-provenance caveat, NOT a gate failure (plan §4)"
        ),
        "repro": repro_metadata(cfg),
    }
    write_json_atomic(cfg.out_root / "p0_artifact_check.json", out)
    return out


# ── model loading ─────────────────────────────────────────────────────


def load_model_and_tokenizer(cfg: RunConfig):
    """Production: bf16 Qwen-2.5-7B-Instruct pinned to cuda:0. Tiny:
    from-config 2-layer same-arch model over the REAL vocab on CPU (the
    #1415 tiny-real pattern)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)  # loaded ONCE (429 gotcha)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if cfg.tiny:
        mcfg = AutoConfig.from_pretrained(cfg.model_id)
        mcfg.hidden_size = cfg.hidden
        mcfg.intermediate_size = 2 * cfg.hidden
        mcfg.num_hidden_layers = cfg.n_model_layers
        mcfg.num_attention_heads = 4
        mcfg.num_key_value_heads = 2
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(mcfg).to(torch.float32)
    else:
        assert torch.cuda.is_available(), "production grid requires CUDA (use --tiny on CPU)"
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16)
        model = model.to("cuda:0")
    assert model.config.hidden_size == cfg.hidden, (model.config.hidden_size, cfg.hidden)
    assert model.config.num_hidden_layers == cfg.n_model_layers
    model.eval()
    return model, tok


def make_hook(cfg: RunConfig, model, arm: str, delta: torch.Tensor, alpha: float) -> DeltaHook:
    """DeltaHook for a steered arm: adds ``alpha * delta`` to the residual
    output of ``model.model.layers[cfg.layer]`` over the arm's token range."""
    assert arm in STEERED_ARMS, arm
    kwargs = {
        "prefill_only": {"prefill_all": True},
        "decode_only": {"decode_only": True},
        "both": {"all_positions": True},
    }[arm]
    return DeltaHook(model, cfg.layer, delta, alpha=alpha, **kwargs)


# ── G0: pilot timing gate ─────────────────────────────────────────────


def phase_pilot(cfg: RunConfig, model, tok) -> dict:
    """G0 (plan §7): 1 question x ``both`` x alpha=1.0, replicated to
    B = gen_batch rows (the sweep chunk shape — a batch-1 pilot over-reads
    s/sample by ~B, the #1415 att-20260716-160022 false-fire), N draws through
    ``generate_batch``. Persists pilot.json + the gate report on refusal."""
    print("[phase=pilot]", flush=True)
    pilot_path = cfg.out_root / "pilot_gate_report.json"
    q = eval_questions(cfg, PILOT_TRAIT)[0]
    delta = load_rb_row(cfg, PILOT_TRAIT)
    B = cfg.gen_batch
    contexts = [{"system": None, "user": q}] * B
    dstack = torch.stack([delta] * B)
    assert dstack.shape == (B, cfg.hidden), dstack.shape
    hook = make_hook(cfg, model, "both", dstack, alpha=1.0)
    t0 = time.monotonic()
    with hook:
        outs = generate_batch(
            model,
            tok,
            contexts,
            n=cfg.n_draws,
            hook=hook,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            seed_base=cfg.seed_base,
        )
    elapsed = time.monotonic() - t0
    n_samples = B * cfg.n_draws
    sps = elapsed / n_samples
    pilot = {
        "phase": "pilot",
        "trait": PILOT_TRAIT,
        "arm": "both",
        "alpha": 1.0,
        "layer": cfg.layer,
        "pilot_batch": B,
        "n_draws": cfg.n_draws,
        "max_new_tokens": cfg.max_new_tokens,
        "n_samples": n_samples,
        "elapsed_s": elapsed,
        "s_per_sample": sps,
        "threshold_s_per_sample": PILOT_MAX_S_PER_SAMPLE,
        "coherence_flags_row0": coherence_check(outs[0]),
        "sweep_allowed": sps <= PILOT_MAX_S_PER_SAMPLE,
        "gate_enforced": not cfg.tiny,
        "repro": repro_metadata(cfg),
    }
    write_json_atomic(cfg.out_root / "pilot.json", pilot)
    logger.info(
        "[pilot] s_per_sample=%.3f (threshold %.2f) sweep_allowed=%s B=%d tiny=%s",
        sps,
        PILOT_MAX_S_PER_SAMPLE,
        pilot["sweep_allowed"],
        B,
        cfg.tiny,
    )
    enforce_pilot_gate(cfg, pilot, pilot_path)
    return pilot


def enforce_pilot_gate(cfg: RunConfig, pilot: dict, report_path: Path) -> None:
    """DESIGNED HALT (artifact + rc=7, never an anonymous crash) when the
    pilot exceeds the plan §7 bound. Tiny demotes the verdict to an
    informational line (production-n-calibrated gate; smoke gate-calibration
    parity) — the halt branch itself is unit-pinned at production shape."""
    if pilot["s_per_sample"] <= PILOT_MAX_S_PER_SAMPLE or cfg.force:
        return
    if cfg.tiny:
        logger.info(
            "[pilot_gate] tiny: s_per_sample=%.3f > %.2f would HALT a production run "
            "(verdict demoted to informational under --tiny)",
            pilot["s_per_sample"],
            PILOT_MAX_S_PER_SAMPLE,
        )
        return
    reason = (
        f"pilot measured {pilot['s_per_sample']:.2f} s/sample at the sweep chunk shape "
        f"(B={pilot['pilot_batch']}) > {PILOT_MAX_S_PER_SAMPLE} — refusing the grid "
        "(pass --force to override, or descope per plan §9 stratification)"
    )
    write_json_atomic(
        report_path,
        {
            "criterion": "G0 pilot timing gate: s_per_sample > plan §7 threshold",
            "fired": True,
            "reason": reason,
            "pilot": pilot,
        },
    )
    logger.error("[pilot_gate] %s", reason)
    raise SystemExit(RC_PILOT_GATE)


# ── G1: generation grid ───────────────────────────────────────────────


def enumerate_groups(cfg: RunConfig) -> list[dict]:
    """(trait, arm, alpha) groups in deterministic order. ``neither`` carries
    ``alpha=None``; steered arms cross the alpha ladder."""
    groups = []
    for trait in cfg.traits:
        groups.append({"trait": trait, "arm": "neither", "alpha": None})
        for arm in STEERED_ARMS:
            for alpha in cfg.alphas:
                groups.append({"trait": trait, "arm": arm, "alpha": alpha})
    return groups


def shard_groups(cfg: RunConfig, groups: list[dict]) -> list[dict]:
    """Shard by (trait x arm) combo index — one model copy per GPU; all alphas
    of a combo stay on one shard."""
    combos = sorted({(g["trait"], g["arm"]) for g in groups})
    mine = {c for i, c in enumerate(combos) if i % cfg.n_shards == cfg.shard}
    return [g for g in groups if (g["trait"], g["arm"]) in mine]


def _fmt(alpha: float) -> str:
    return f"{alpha:g}"


def cell_id(trait: str, arm: str, alpha: float | None, qid: int) -> str:
    if arm == "neither":
        return f"{trait}/neither/q{qid:02d}"
    return f"{trait}/{arm}/a{_fmt(alpha)}/q{qid:02d}"


def completion_filename(cfg: RunConfig, arm: str, alpha: float | None, qid: int) -> str:
    """Plan §10 layout: raw_completions/{trait}/{arm}_a{alpha}_q{qid}_seed{s}.json
    (neither omits the alpha token)."""
    if arm == "neither":
        return f"neither_q{qid:02d}_seed{cfg.seed_base}.json"
    return f"{arm}_a{_fmt(alpha)}_q{qid:02d}_seed{cfg.seed_base}.json"


def cell_meta_path(cfg: RunConfig, cid: str) -> Path:
    return cfg.out_root / "cells" / (cid.replace("/", "__") + ".json")


def cell_done(cfg: RunConfig, cid: str, fingerprint: dict) -> bool:
    """Resume predicate: done ONLY when the recorded fingerprint matches AND
    the completion file exists — mismatches re-run (#952 gate-5 shape)."""
    p = cell_meta_path(cfg, cid)
    if not p.exists():
        return False
    meta = json.loads(p.read_text())
    if meta.get("fingerprint") != fingerprint:
        logger.info("[grid] fingerprint mismatch for %s — re-running", cid)
        return False
    comp = cfg.bulk_root / "raw_completions" / meta["trait"] / meta["completion_file"]
    return comp.exists()


def upload_group(cfg: RunConfig, trait: str, filenames: list[str]) -> None:
    """Incremental per-group raw-completion upload (ONE upload_folder commit
    per group via ``hub._upload_folder_filtered``; exact-set verified).
    ``local-mirror`` (tiny) copies through the IDENTICAL call sequence so the
    control flow never forks (the #1415 pattern)."""
    local_dir = cfg.bulk_root / "raw_completions" / trait
    path_in_repo = f"{cfg.hf_prefix}/raw_completions/{trait}"
    expected = [f"{path_in_repo}/{fn}" for fn in filenames]
    if cfg.upload_mode == "local-mirror":
        for fn in filenames:
            dest = cfg.bulk_root / "hf_mirror" / path_in_repo / fn
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_dir / fn, dest)
        logger.info("[upload] local-mirror %d files -> %s", len(filenames), path_in_repo)
        return
    from explore_persona_space.orchestrate import hub

    delays = (30.0, 60.0, 120.0)
    for attempt in range(len(delays) + 1):
        url = hub._upload_folder_filtered(
            local_dir,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            allow_patterns=list(filenames),
            expected_repo_paths=expected,
        )
        if url:
            logger.info("[upload] %d files -> %s", len(filenames), url)
            return
        if attempt < len(delays):
            logger.warning(
                "[upload] no path for %s (attempt %d) — retrying in %.0fs",
                path_in_repo,
                attempt + 1,
                delays[attempt],
            )
            time.sleep(delays[attempt])
    raise RuntimeError(f"upload returned no path for {path_in_repo} after {len(delays) + 1} tries")


def phase_grid(cfg: RunConfig, model, tok) -> dict:
    """G1: run this shard's (trait x arm x alpha) groups; per-cell checkpoint
    metadata + per-unit progress lines + incremental per-group uploads."""
    print(f"[phase=grid shard={cfg.shard}/{cfg.n_shards}]", flush=True)
    fingerprint = cell_fingerprint(cfg)
    groups = shard_groups(cfg, enumerate_groups(cfg))
    n_units = sum(cfg.n_questions for _ in groups)
    unit = 0
    t_start = time.monotonic()
    rb_cache: dict[str, torch.Tensor] = {}
    cells_run = 0
    cells_skipped = 0
    for grp in groups:
        trait, arm, alpha = grp["trait"], grp["arm"], grp["alpha"]
        questions = eval_questions(cfg, trait)
        pending: list[tuple[int, str]] = []
        for qid, q in enumerate(questions):
            cid = cell_id(trait, arm, alpha, qid)
            if cell_done(cfg, cid, fingerprint):
                cells_skipped += 1
                unit += 1
            else:
                pending.append((qid, q))
        if trait not in rb_cache:
            rb_cache[trait] = load_rb_row(cfg, trait)
        for start in range(0, len(pending), cfg.gen_batch):
            chunk = pending[start : start + cfg.gen_batch]
            contexts = [{"system": None, "user": q} for _qid, q in chunk]
            hook = None
            if arm != "neither":
                hook = make_hook(cfg, model, arm, rb_cache[trait], float(alpha))
                hook.install()
            try:
                outs = generate_batch(
                    model,
                    tok,
                    contexts,
                    n=cfg.n_draws,
                    hook=hook,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    seed_base=cfg.seed_base,
                )
            finally:
                if hook is not None:
                    hook.remove()
            chunk_qids = [qid for qid, _q in chunk]
            for b, (qid, q) in enumerate(chunk):
                cid = cell_id(trait, arm, alpha, qid)
                fname = completion_filename(cfg, arm, alpha, qid)
                flags = coherence_check(outs[b])
                comp_path = cfg.bulk_root / "raw_completions" / trait / fname
                write_json_atomic(
                    comp_path,
                    {
                        "cell_id": cid,
                        "trait": trait,
                        "arm": arm,
                        "alpha": alpha,
                        "question_id": qid,
                        "question": q,
                        "layer": cfg.layer,
                        "draws": outs[b],
                        "coherence_flags": flags,
                        "seeds": [cfg.seed_base + i for i in range(cfg.n_draws)],
                        "temperature": cfg.temperature,
                        "max_new_tokens": cfg.max_new_tokens,
                        "repro": repro_metadata(cfg),
                    },
                )
                write_json_atomic(
                    cell_meta_path(cfg, cid),
                    {
                        "cell_id": cid,
                        "trait": trait,
                        "arm": arm,
                        "alpha": alpha,
                        "question_id": qid,
                        "layer": cfg.layer,
                        "n_draws": cfg.n_draws,
                        "coherence_flags": flags,
                        "n_coherent": int(sum(flags)),
                        "completion_file": fname,
                        "chunk_questions": chunk_qids,  # chunk composition provenance
                        "shard": cfg.shard,
                        "fingerprint": fingerprint,
                        "repro": repro_metadata(cfg),
                    },
                )
                unit += 1
                cells_run += 1
                print(
                    f"[grid] unit {unit}/{n_units} {cid} elapsed={time.monotonic() - t_start:.1f}s",
                    flush=True,
                )
        # Incremental per-group upload (all of the group's files, incl.
        # resume-skipped ones — the upload is idempotent + exact-set verified).
        fnames = [completion_filename(cfg, arm, alpha, qid) for qid in range(len(questions))]
        upload_group(cfg, trait, fnames)
    summary = {"shard": cfg.shard, "cells_run": cells_run, "cells_skipped": cells_skipped}
    logger.info("[grid] shard %d/%d done: %s", cfg.shard, cfg.n_shards, summary)
    return summary


# ── finalize: coverage + manifest + upload verify ─────────────────────


def expected_cells(cfg: RunConfig) -> list[str]:
    out = []
    for grp in enumerate_groups(cfg):
        for qid in range(cfg.n_questions):
            out.append(cell_id(grp["trait"], grp["arm"], grp["alpha"], qid))
    return out


def evaluate_k2(cfg: RunConfig, cell_metas: dict[str, dict]) -> dict:
    """Plan §7 K2 (dose-ladder coherence): per (trait, alpha) the BOTH-arm
    pooled coherent-draw rate; FIRED when ALL traits x ALL alphas sit below
    the #1415 gate (>= 50% coherent). Evaluated at finalize — BEFORE the
    30k-call judge phase — so a dose ladder wholly outside the coherent
    regime halts artifact-routed (k2_report.json + rc=9), never burns the J
    phase (kill criterion 2: one alpha-sub-grid retry at x0.5 is the
    orchestrator's, per the plan)."""
    rates: dict[str, float] = {}
    for trait in cfg.traits:
        for alpha in cfg.alphas:
            coherent = 0
            total = 0
            for qid in range(cfg.n_questions):
                meta = cell_metas[cell_id(trait, "both", alpha, qid)]
                coherent += meta["n_coherent"]
                total += meta["n_draws"]
            rates[f"{trait}/a{_fmt(alpha)}"] = coherent / total
    fired = all(r < K2_COHERENCE_MIN for r in rates.values())
    return {
        "criterion": (
            "K2 dose-ladder coherence (plan §7): ALL traits x ALL alphas below the "
            f"both-arm >= {K2_COHERENCE_MIN:.0%}-coherent gate => HALT before the judge phase"
        ),
        "both_arm_coherence_rates": rates,
        "threshold": K2_COHERENCE_MIN,
        "fired": bool(fired),
    }


def phase_finalize(cfg: RunConfig) -> dict:
    """Merge per-cell metadata -> cells_manifest.json; assert FULL grid
    coverage under the CURRENT fingerprint (row-coverage: the J phase submits
    only after this passes); verify the raw-completion upload set."""
    print("[phase=finalize]", flush=True)
    fingerprint = cell_fingerprint(cfg)
    expected = expected_cells(cfg)
    cells: dict[str, dict] = {}
    missing: list[str] = []
    stale: list[str] = []
    for cid in expected:
        p = cell_meta_path(cfg, cid)
        if not p.exists():
            missing.append(cid)
            continue
        meta = json.loads(p.read_text())
        if meta.get("fingerprint") != fingerprint:
            stale.append(cid)
            continue
        cells[cid] = {
            "completion_file": meta["completion_file"],
            "n_coherent": meta["n_coherent"],
            "n_draws": meta["n_draws"],
            "shard": meta["shard"],
        }
    assert not missing and not stale, (
        f"grid coverage incomplete: {len(missing)} missing, {len(stale)} stale-fingerprint "
        f"of {len(expected)} cells (first missing: {missing[:5]}, first stale: {stale[:5]})"
    )
    manifest = {
        "n_cells": len(expected),
        "fingerprint": fingerprint,
        "grid": {
            "traits": list(cfg.traits),
            "arms": list(ARMS),
            "alphas": list(cfg.alphas),
            "n_questions": cfg.n_questions,
            "n_draws": cfg.n_draws,
        },
        "cells": cells,
        "repro": repro_metadata(cfg),
    }
    write_json_atomic(cfg.out_root / "cells_manifest.json", manifest)

    # Upload-set verify (same branch shape as upload_group).
    expected_paths = []
    for cid, rec in cells.items():
        trait = cid.split("/")[0]
        expected_paths.append(f"{cfg.hf_prefix}/raw_completions/{trait}/{rec['completion_file']}")
    if cfg.upload_mode == "local-mirror":
        missing_up = [p for p in expected_paths if not (cfg.bulk_root / "hf_mirror" / p).exists()]
    else:
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub

        missing_up = hub.verify_repo_paths_uploaded(
            HfApi(),
            HF_DATA_REPO,
            expected_paths,
            path_in_repo=f"{cfg.hf_prefix}/raw_completions",
            repo_type="dataset",
        )
    assert not missing_up, f"{len(missing_up)} raw-completion uploads missing: {missing_up[:5]}"
    logger.info(
        "[finalize] %d cells manifested; %d raw-completion uploads verified",
        len(expected),
        len(expected_paths),
    )
    # K2 dose-ladder coherence gate (plan §7) — AFTER the upload verify (the
    # grid data stays durable either way), artifact-routed on fire.
    k2 = evaluate_k2(cfg, cells)
    k2["repro"] = repro_metadata(cfg)
    write_json_atomic(cfg.out_root / "k2_report.json", k2)
    enforce_k2_gate(cfg, k2)
    return manifest


def enforce_k2_gate(cfg: RunConfig, k2: dict) -> None:
    """DESIGNED HALT (rc=9, artifact already written) on a fired K2. Tiny
    demotes the verdict to an informational line (the coherence of a tiny
    random model's text is not the production quantity; the halt branch is
    unit-pinned at production shape)."""
    if not k2["fired"]:
        return
    if cfg.tiny:
        logger.info(
            "[k2] tiny: dose-ladder coherence gate would HALT a production run "
            "(verdict demoted to informational under --tiny; rates: %s)",
            k2["both_arm_coherence_rates"],
        )
        return
    logger.error(
        "[k2] DESIGNED HALT: every (trait, alpha) both-arm below the %.0f%% "
        "coherence gate — see k2_report.json (plan §7 kill criterion 2)",
        100 * K2_COHERENCE_MIN,
    )
    raise SystemExit(RC_K2_GATE)


# ── main ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        choices=("p0", "pilot", "grid", "finalize", "all"),
        default="all",
        help="pipeline phase (the dispatcher runs p0 -> pilot -> grid (sharded) -> finalize)",
    )
    ap.add_argument("--tiny", action="store_true", help="tiny-real CPU smoke (full control flow)")
    ap.add_argument("--force", action="store_true", help="override the G0 pilot gate")
    ap.add_argument("--shard", type=int, default=0, help="grid shard index (trait x arm combos)")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out-root", type=Path, default=None, help="metadata root (git)")
    ap.add_argument("--bulk-root", type=Path, default=None, help="raw-completion staging root")
    ap.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="override alpha ladder (space-separated floats; e.g. --alphas 1.5 3.0)",
    )
    # UPLOAD_PREFIX_EXEMPT: plan v10 item 4 pins default=parent constant (byte-identical no-flag parity); fu1 passes an explicit fresh prefix
    ap.add_argument(
        "--hf-prefix",
        type=str,
        default=HF_OUT_PREFIX,
        help="HF data-repo prefix for every upload (fu1 passes a fresh prefix so no "
        "path ever lands under the parent's raw_completions)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the real branch, then exit 0 "
        "(the Axis-1 smoke-architecture leg)",
    )
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> RunConfig:
    if args.tiny:
        out_root = args.out_root or Path("data/issue_1769/tiny_smoke/out")
        bulk_root = args.bulk_root or Path("data/issue_1769/tiny_smoke/bulk")
        cfg = tiny_config(out_root, bulk_root, force=args.force)
    else:
        out_root = args.out_root or Path("eval_results/issue_1769/phase_g")
        default_bulk = Path("/workspace/eps-issue-1769")
        if not default_bulk.parent.exists():
            default_bulk = Path("data/issue_1769/bulk")
        bulk_root = args.bulk_root or default_bulk
        cfg = RunConfig(tiny=False, out_root=out_root, bulk_root=bulk_root, force=args.force)
    if args.alphas is not None:
        cfg = replace(cfg, alphas=tuple(sorted(args.alphas)))
    cfg.hf_prefix = args.hf_prefix
    cfg.shard = args.shard
    cfg.n_shards = args.n_shards
    assert 0 <= cfg.shard < cfg.n_shards, (cfg.shard, cfg.n_shards)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.bulk_root.mkdir(parents=True, exist_ok=True)
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.import_check:
        # Axis-1 import-resolution leg: execute every deferred import this
        # driver (and its judge sibling) reaches on the real branch.
        from huggingface_hub import hf_hub_download  # noqa: F401
        from transformers import (  # noqa: F401
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        from explore_persona_space.orchestrate import hub  # noqa: F401
        from issue1769_judge import resolve_rubric as _rr  # noqa: F401

        print("[import-check] OK", flush=True)
        sys.exit(0)
    cfg = build_config(args)
    logger.info(
        "issue1769 driver: phase=%s tiny=%s shard=%d/%d out=%s bulk=%s",
        args.phase,
        cfg.tiny,
        cfg.shard,
        cfg.n_shards,
        cfg.out_root,
        cfg.bulk_root,
    )
    if args.phase in ("p0", "all"):
        phase_p0(cfg)
    model = tok = None
    if args.phase in ("pilot", "grid", "all"):
        model, tok = load_model_and_tokenizer(cfg)
    if args.phase in ("pilot", "all"):
        phase_pilot(cfg, model, tok)
    if args.phase in ("grid", "all"):
        phase_grid(cfg, model, tok)
    if args.phase in ("finalize", "all"):
        phase_finalize(cfg)
    # Explicit exit: heavy C-extension modules (torch/transformers) can hit a
    # finalize-time atexit race that rewrites the rc of a COMPLETED phase
    # (gotchas.md PyGILState_Release entry) — flush + exit 0 explicitly.
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
